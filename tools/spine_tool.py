#!/usr/bin/env python3
"""spine_tool.py - cut, splice and inspect STATE_DTYPE spines (pbr3b1, 2026-10-24).

The search -> curriculum loop of this lineage (autoresearch/NOTES.md, "The method so far") turns
a searched line into a `train_fast --demo-file` spine by hand every gate: take beam_tas's
`--save-frontier` states (beam_frontier.npz) or a ge_replay line, cut it at its deepest point
on the shaping field before it stalls, splice a Go-Explore chain onto the line at the chain's
root, zero the clocks. This does those steps the same way every time.

    python tools/spine_tool.py info --src runs/x/beam_frontier.npz
    python tools/spine_tool.py cut  --src runs/x/beam_frontier.npz --out runs/x/spine.npy
    python tools/spine_tool.py cat  --part runs/x/line.npy:0:1455 --part runs/y/chain.npy \\
        --out runs/y/spine.npy
    python tools/spine_tool.py splice --line runs/x/spine.npy --chain runs/y/chain.npy \\
        --out runs/y/spine2.npy     # finds the chain's root on the line

* `--src` is a .npy of STATE_DTYPE rows or an .npz holding them under `states`.
* `cut --end auto` (default) ends the spine at the line's deepest state on `--field`
  (argmin d over the reachable states); `--end N` at state N; `--start N` drops the first N.
* `cat --part FILE[:LO[:HI]]` takes rows LO..HI-1 of each part, in order, and reports the gap
  between consecutive parts: a chain spliced at its root has a 0 u gap (the repeated root is
  kept once), the next tick of the same line <= 35 u; more than --max-gap is refused.
* The field defaults to autoresearch/config.json's `field` (the grader's), so the progress
  printed is the score's. Every written spine has tick and stuck_ticks zeroed (a spawned row
  must not inherit its episode clock).
Nothing here reads a demo or a record: the inputs are the policy's own searched lines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE  # noqa: E402
from surfgym.goalfield import GoalField  # noqa: E402


def default_field() -> str:
    cfg = ROOT / "autoresearch" / "config.json"
    if cfg.exists():
        return str(ROOT / json.loads(cfg.read_text(encoding="utf-8"))["field"])
    raise SystemExit("--field is required (no autoresearch/config.json)")


def load_field(path: str) -> GoalField:
    z = np.load(path, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def load_states(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        if "states" not in z.files:
            raise SystemExit(f"{p} has no `states` array (beam_tas --save-frontier writes one)")
        s = z["states"]
    else:
        s = np.load(p, allow_pickle=False)
    if s.dtype != STATE_DTYPE:
        raise SystemExit(f"{p}: dtype is not STATE_DTYPE")
    return np.array(s, dtype=STATE_DTYPE, copy=True)


def depths(states: np.ndarray, f: GoalField) -> np.ndarray:
    """Geodesic d per state; unreachable rows (off the field) are +inf."""
    d = f.sample(states["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    return d


def zero_clocks(states: np.ndarray) -> np.ndarray:
    out = np.array(states, dtype=STATE_DTYPE, copy=True)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    return out


def parse_part(spec: str):
    """FILE[:LO[:HI]] -> (file, lo, hi); a Windows drive letter is not a range."""
    bits = spec.split(":")
    nums = []
    while len(bits) > 1 and (bits[-1] == "" or bits[-1].lstrip("-").isdigit()) and len(nums) < 2:
        nums.insert(0, bits.pop())
    lo = int(nums[0]) if len(nums) >= 1 and nums[0] != "" else None
    hi = int(nums[1]) if len(nums) >= 2 and nums[1] != "" else None
    return ":".join(bits), lo, hi


def describe(states: np.ndarray, f: GoalField, d0: float, label: str = "") -> dict:
    d = depths(states, f)
    v = states["velocity"].astype(np.float64)
    h = states["origin"][:, 2].astype(np.float64) + (v ** 2).sum(1) / (2 * 800.0)
    ok = np.isfinite(d)
    best = int(np.argmin(np.where(ok, d, np.inf))) if ok.any() else -1
    prog = lambda x: 100.0 * (d0 - x) / d0
    info = {"n": int(len(states)),
            "start": [round(float(c), 1) for c in states["origin"][0]],
            "end": [round(float(c), 1) for c in states["origin"][-1]],
            "prog_start": round(prog(d[0]), 3) if ok[0] else None,
            "prog_end": round(prog(d[-1]), 3) if ok[-1] else None,
            "best_idx": best,
            "prog_best": round(prog(d[best]), 3) if best >= 0 else None,
            "h_end": round(float(h[-1]), 1),
            "ducked_frac": round(float(np.mean(states["ducked"] != 0)), 3)}
    print((label + ": " if label else "") + json.dumps(info))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("info", "cut", "cat", "splice"):
        p = sub.add_parser(name)
        p.add_argument("--field", default=None)
        p.add_argument("--d0", type=float, default=None,
                       help="progress reference (default: config.json start_d, else d of row 0)")
        if name in ("info", "cut"):
            p.add_argument("--src", required=True)
        if name == "cut":
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--end", default="auto", help="'auto' = the deepest state, or an index")
        if name == "cat":
            p.add_argument("--part", action="append", required=True)
            p.add_argument("--max-gap", type=float, default=50.0,
                           help="refuse a splice whose consecutive parts are further apart (u); "
                                "consecutive states of one line are one tick (<= 35 u at "
                                "3,500 u/s) apart, a chain spliced at its root 0 u")
        if name == "splice":
            p.add_argument("--line", required=True, help="the line the chain was rooted on")
            p.add_argument("--chain", required=True, help="ge_replay's line (row 0 = its root)")
            p.add_argument("--tol", type=float, default=0.5,
                           help="the root must match a line state within this many u")
        if name in ("cut", "cat", "splice"):
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    f = load_field(a.field or default_field())
    d0 = a.d0
    if d0 is None:
        cfg = ROOT / "autoresearch" / "config.json"
        d0 = (float(json.loads(cfg.read_text(encoding="utf-8")).get("start_d") or 0.0)
              if cfg.exists() else 0.0) or None
    if a.cmd == "info":
        s = load_states(a.src)
        describe(s, f, d0 if d0 else float(depths(s, f)[0]), a.src)
        return 0
    if a.cmd == "cut":
        s = load_states(a.src)
        ref = d0 if d0 else float(depths(s, f)[0])
        describe(s, f, ref, "source")
        if a.end == "auto":
            d = depths(s, f)
            end = int(np.argmin(d))
            if not np.isfinite(d[end]):
                raise SystemExit("no state of the source lies on the field")
        else:
            end = int(a.end)
        if not 0 <= a.start <= end < len(s):
            raise SystemExit(f"cut [{a.start}, {end}] is outside the source's {len(s)} states")
        out = zero_clocks(s[a.start:end + 1])
        np.save(a.out, out)
        describe(out, f, ref, f"spine [{a.start}, {end}] -> {a.out}")
        return 0
    if a.cmd == "splice":
        line, chain = load_states(a.line), load_states(a.chain)
        gap = np.linalg.norm(line["origin"].astype(np.float64)
                             - chain[0]["origin"].astype(np.float64)[None], axis=1)
        dv = np.linalg.norm(line["velocity"].astype(np.float64)
                            - chain[0]["velocity"].astype(np.float64)[None], axis=1)
        i = int(np.argmin(gap + dv))
        print(f"chain root = line state {i} (gap {gap[i]:.3f} u, dv {dv[i]:.3f} u/s)")
        if gap[i] > a.tol or dv[i] > a.tol:
            raise SystemExit(f"the chain's root is not on the line (nearest state {i}: "
                             f"{gap[i]:.1f} u, {dv[i]:.1f} u/s away)")
        out = zero_clocks(np.concatenate([line[:i], chain]))
        np.save(a.out, out)
        describe(out, f, d0 if d0 else float(depths(out, f)[0]),
                 f"line [0, {i}) + chain ({len(chain)}) -> {a.out}")
        return 0
    parts = []
    for spec in a.part:
        path, lo, hi = parse_part(spec)
        s = load_states(path)
        parts.append(s[slice(lo, hi)])
        if not len(parts[-1]):
            raise SystemExit(f"{spec}: empty slice")
    for k in range(1, len(parts)):
        gap = float(np.linalg.norm(parts[k][0]["origin"].astype(np.float64)
                                   - parts[k - 1][-1]["origin"].astype(np.float64)))
        print(f"junction {k}: gap {gap:.2f} u")
        if gap > a.max_gap:
            raise SystemExit(f"junction {k}: the parts are {gap:.1f} u apart (> --max-gap "
                             f"{a.max_gap:g}); splice a chain at its root state")
    # a chain spliced at its root repeats that root: keep one copy
    keep = [parts[0]]
    for k in range(1, len(parts)):
        same = np.array_equal(parts[k][0]["origin"], parts[k - 1][-1]["origin"]) and \
            np.array_equal(parts[k][0]["velocity"], parts[k - 1][-1]["velocity"])
        keep.append(parts[k][1:] if same else parts[k])
    out = zero_clocks(np.concatenate(keep))
    np.save(a.out, out)
    describe(out, f, d0 if d0 else float(depths(out, f)[0]), f"spliced -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
