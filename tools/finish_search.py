#!/usr/bin/env python3
"""finish_search.py - a FINISHING line from the policy's OWN search (zlr3b2, 2026-11-01), the rule
that took surf_za_loop from 0 finishes to 9/9 in round 3, written down so a trainer can run it by
itself (train_fast --auto-finish) instead of an agent choosing roots by eye:

1. GREEDY LINES: tools/beam_tas.py with every env greedy (--envs G --greedy-envs G
   --save-frontier --frontier-last) at --seeds spawn seeds - one exact greedy line of the newest
   weights per seed, replayed open-loop from the true start to where it dies (or --max-secs).
   A greedy line that already finishes is the answer as it is.
2. ROOTS: the deepest line (smallest goal-field d it reached); the searches start from ITS OWN
   state --roots seconds before its end (default 10 and 25 s: the stretch where it was still
   flying when the fatal swing began).
3. ENERGY-RANKED BEAMS: beam_tas --prefix-line <line>:<root> with the policy's own proposals
   (--envs N --greedy-envs 1), lineages ranked by -d + --score-energy x the energy height
   z + |v|^2/2g (--score dh) every 25 decisions, for --span seconds. The energy rank keeps the
   lineages that SURVIVE AND PUMP (a d rank keeps the ones high in their swing at each boundary
   and starves the pumping a loop needs); the first goal crossing is the fastest line.
4. The first crossing's replayed states (beam_frontier.npz, finished = 1; replay bit-exact by
   beam_tas's own assert) are written to <out>/finish_line.npy; the last stdout line is
   ``AUTO_FINISH {json}`` either way.

Measured (surf_za_loop round 3, incumbent weights, 4,096 envs, 105 s spans): energy weight 1 ->
1 / 31 finishers from two roots, weight 2 -> 11 / 36 / 1 finishers from three; a d rank (weight
0) found none from the same roots in 30-42 s spans.

No constant here reads a map: roots are times before the policy's own death, the rank is the
field distance and the energy height, the budgets are times and env counts. No demo enters
(CLAUDE.md sections 0, 0b).

    python tools/finish_search.py --ckpt runs/X/ckpt.pt --out runs/X/fin/k01 --max-secs 230
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def run_beam(args: list, log: Path, timeout: float) -> tuple[int, str]:
    cmd = [sys.executable, "-u", str(ROOT / "tools" / "beam_tas.py")] + [str(a) for a in args]
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -9
    return code, log.read_text(encoding="utf-8", errors="replace")


def crossed(text: str) -> bool:
    return "CROSSING in" in text


def line_states(npz: Path):
    z = np.load(npz, allow_pickle=False)
    return z["states"], int(z["finished"]) if "finished" in z.files else 0


def pick_deepest(lines: list, d_fn) -> int:
    """Index of the line whose states reach the smallest field distance (ties: the longer)."""
    best, bi = None, -1
    for i, st in enumerate(lines):
        if st is None or len(st) == 0:
            continue
        d = float(np.min(d_fn(np.asarray(st["origin"], np.float64))))
        key = (d, -len(st))
        if best is None or key < best:
            best, bi = key, i
    return bi


def root_ticks(n: int, secs: list, tps: float, k: int) -> list:
    """Prefix lengths (ticks, on a decision boundary, inside the line) secs before its end."""
    out = []
    for s in secs:
        t = int(n - round(float(s) * tps))
        t -= t % max(1, int(k))
        if 0 < t < n and t not in out:
            out.append(t)
    return out


def main(argv=None):
    from auto_spine import ckpt_tick
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default=None, help="goal field .npz the lines are ranked on "
                    "(default: the checkpoint's own, as beam_tas builds it)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--greedy-envs", type=int, default=16)
    ap.add_argument("--max-secs", type=float, default=300.0,
                    help="greedy-line cap (the trainer passes its longest greedy episode + 30 s)")
    ap.add_argument("--roots", default="10,25", help="seconds before the line's end")
    ap.add_argument("--span", type=float, default=105.0, help="seconds each beam searches")
    ap.add_argument("--score-energy", type=float, default=2.0)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--act-every", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=3000.0, help="wall-clock cap (s)")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tick_ms = float(ckpt_tick(a.ckpt))
    tps = 1000.0 / tick_ms
    import torch
    cfg = (torch.load(a.ckpt, map_location="cpu", weights_only=False).get("config") or {})
    K = int(a.act_every or cfg.get("act_every", 1))
    res = {"line": None, "finish_s": None, "root": None, "lines": [], "searches": []}

    def left():
        return a.timeout - (time.time() - t0)

    def emit():
        res["wall_s"] = round(time.time() - t0, 1)
        print("AUTO_FINISH " + json.dumps(res), flush=True)

    def take(npz: Path, how: str):
        st, fin = line_states(npz)
        np.save(out / "finish_line.npy", st)
        res.update(line=str(out / "finish_line.npy"), finish_s=round(len(st) / tps, 2),
                   how=how)

    # 1. greedy lines, one per spawn seed
    lines, max_t = [], int(round(a.max_secs * tps))
    for s in range(int(a.seeds)):
        if left() <= 60:
            break
        g = out / f"g{s}"
        code, txt = run_beam([a.ckpt, "--envs", a.greedy_envs, "--greedy-envs", a.greedy_envs,
                              "--score", "d", "--seed", s, "--max-ticks", max_t,
                              "--save-frontier", "--frontier-last", "--allow-nonfinisher",
                              "--skip-gate", "--route-file", "none", "--out-dir", g,
                              "--log-every", 1000], out / f"g{s}.log", left())
        npz = g / "beam_frontier.npz"
        if code != 0 or not npz.exists():
            lines.append(None)
            res["lines"].append({"seed": s, "ok": False, "exit": code})
            continue
        st, fin = line_states(npz)
        lines.append(st)
        res["lines"].append({"seed": s, "ok": True, "ticks": int(len(st)), "finished": fin})
        if crossed(txt) and fin:
            take(npz, f"greedy line seed {s}")
            emit()
            return
    # 2. the deepest line and its roots
    if a.field:
        from surfgym.goalfield import load_goal_field
        gf = load_goal_field(a.field)
        d_fn = gf.sample
    else:
        def d_fn(o):     # without a field: the line's own end is as deep as it got
            return -np.arange(len(o), dtype=np.float64)
    bi = pick_deepest(lines, d_fn)
    if bi < 0:
        res["error"] = "no greedy line"
        emit()
        return
    line_npz = out / f"g{bi}" / "beam_frontier.npz"
    roots = root_ticks(len(lines[bi]), [float(x) for x in a.roots.split(",") if x.strip()],
                       tps, K)
    res["root_line"] = int(bi)
    # 3. energy-ranked beams from the line's own states before its end
    span = int(round(a.span * tps))
    for t in roots:
        if left() <= 120:
            break
        p = out / f"p{t}"
        code, txt = run_beam([a.ckpt, "--envs", a.envs, "--greedy-envs", 1, "--score", "dh",
                              "--score-energy", f"{a.score_energy:g}",
                              "--prefix-line", f"{line_npz}:{t}", "--max-ticks", t + span,
                              "--save-frontier", "--keep-finishers", 16, "--allow-nonfinisher",
                              "--skip-gate", "--route-file", "none", "--out-dir", p,
                              "--log-every", 10], out / f"p{t}.log", left())
        npz = p / "beam_frontier.npz"
        fin = 0
        if npz.exists():
            fin = line_states(npz)[1]
        res["searches"].append({"root_tick": int(t), "root_s": round(t / tps, 2),
                                "exit": code, "crossed": bool(crossed(txt) and fin)})
        if crossed(txt) and fin:
            take(npz, f"energy beam from line {bi} at {t / tps:.1f} s")
            res["root"] = int(t)
            emit()
            return
    emit()


if __name__ == "__main__":
    main()
