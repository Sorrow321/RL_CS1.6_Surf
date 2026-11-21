#!/usr/bin/env python3
"""ge_iter.py - test-time Go-Explore in stages, each rooted at the end of the line before it.

The user (2026-11-21): try search methods overnight on an unseen map (surf_left_turn) with no
training. auto_spine's own Go-Explore stage (tools/explore_phase1.py --exact-chains, uniform
actions, a cell archive; the deepest chain replayed and spliced onto the line at its root) is the
one search that surfed past the first ramps there, in ONE stage. This driver repeats that stage:
stage k roots at the last --roots-secs of line k-1, and its deepest chain extends the line, until
the explorer hits the finish (--goals 1), --stages run out, or --patience stages in a row gain
less than --gain on the field. A stage that gains nothing roots wider the next time (twice the
seconds, up to the whole line). No constant here comes from the map; the start line is any
line the search tools wrote (a spine .npy of STATE_DTYPE rows, or beam_tas's beam_frontier.npz).

    python tools/ge_iter.py --map maps/M.bsp --field maps/M.goalc_32.npz --line spine.npy \\
        --out runs/ge_iter --stages 12 --ge-minutes 30 --tick 7.667 --maxvel 4000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import auto_spine  # noqa: E402


def load_line(path: str) -> np.ndarray:
    from surfgym.core import STATE_DTYPE
    z = np.load(path, allow_pickle=False)
    st = z["states"] if path.endswith(".npz") else z
    return np.asarray(st, STATE_DTYPE)


def field_end(fld, line) -> float:
    return float(fld.sample(np.asarray(line[-1]["origin"], np.float64)[None])[0])


def field_best(fld, line) -> float:
    return float(np.min(fld.sample(np.asarray(line["origin"], np.float64))))


def prune(stage_dir: Path, keep_mb: float = 50.0) -> int:
    """Delete the stage's big files (the archive) once its line is out: a box disk holds a few."""
    freed = 0
    for f in stage_dir.rglob("*"):
        if f.is_file() and f.stat().st_size > keep_mb * 1e6 and "win_" not in f.name:
            freed += f.stat().st_size
            f.unlink()
    return freed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--line", required=True, help="the start line (.npy states or .npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", type=int, default=12)
    ap.add_argument("--ge-minutes", type=float, default=30.0)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--roots-secs", type=float, default=4.0)
    ap.add_argument("--roots-n", type=int, default=48)
    ap.add_argument("--ge-cell", type=float, default=64.0)
    ap.add_argument("--ge-energy", type=float, default=32.0)
    ap.add_argument("--ge-cell-heading", type=int, default=1)
    ap.add_argument("--ge-view-follow", default="-90,-60,-30,-15,0,15,30,60,90")
    ap.add_argument("--ge-select-energy", type=float, default=0.0)
    ap.add_argument("--ge-boost", default="1.0")
    ap.add_argument("--kind", choices=("plain", "held", "contact"), default="held")
    ap.add_argument("--gain", type=float, default=50.0,
                    help="a stage counts as progress when the line's best field d drops by this")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tick", type=float, default=10.0, help="the policy's tick, ms")
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--maxvel", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    auto_spine.TICKS_PER_S = 1000.0 / float(a.tick)
    a.ge_keep_all = 1
    fld = auto_spine.load_field(a.field)
    line = load_line(a.line)
    base_roots = float(a.roots_secs)
    best = field_best(fld, line)
    print(f"ge_iter: start line {len(line)} ticks, end {np.round(line[-1]['origin']).tolist()}, "
          f"end d {field_end(fld, line):,.0f}, best d {best:,.0f}; {a.stages} stages of "
          f"{a.ge_minutes:g} min, roots {a.roots_secs:g} s, kind {a.kind}", flush=True)
    stale = 0
    hist = []
    for k in range(1, int(a.stages) + 1):
        t0 = time.time()
        a.seed = int(a.seed) + 1
        tag = f"st{k:02d}"
        gl, info = auto_spine.go_explore(a, out, line, len(line) - 1, tag=tag, kind=a.kind)
        sd = out / tag
        win = (sd / "win_1.spine.npy").exists()
        freed = 0 if win else prune(sd)      # a goal hit keeps its archive: the exact replay
        rec = {"stage": k, "roots_secs": a.roots_secs, "minutes": round((time.time() - t0) / 60, 1),
               "goal": bool(win), "freed_mb": round(freed / 1e6), **{
                   kk: vv for kk, vv in info.items() if kk in ("ge_rc", "ge_replay_rc",
                                                              "ge_chain_len", "ge_root_index")}}
        if gl is None:
            rec["result"] = "no chain"
            stale += 1
        else:
            nb = field_best(fld, gl)
            rec.update(result="line", ticks=int(len(gl)), end=np.round(gl[-1]["origin"]).tolist(),
                       end_d=round(field_end(fld, gl)), best_d=round(nb))
            if nb < best - float(a.gain) or win:
                stale = 0
                a.roots_secs = base_roots
            else:
                stale += 1
            if nb < best or win:
                best = nb
                line = gl
                np.save(out / f"line_{k:02d}.npy", line)
        if gl is None or stale:
            # no gain: root wider next time (up to the whole line)
            a.roots_secs = min(float(a.roots_secs) * 2.0, len(line) / auto_spine.TICKS_PER_S)
        hist.append(rec)
        print("ge_iter stage " + json.dumps(rec), flush=True)
        (out / "ge_iter.json").write_text(json.dumps({"stages": hist, "best_d": best,
                                                       "ticks": int(len(line))}, indent=1))
        np.save(out / "line_best.npy", line)
        if win:
            print(f"GE_ITER FINISH at stage {k}: {sd / 'win_1.spine.npy'}", flush=True)
            return 0
        if stale >= int(a.patience):
            print(f"ge_iter: {stale} stages without a gain of {a.gain:g} u: stop", flush=True)
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
