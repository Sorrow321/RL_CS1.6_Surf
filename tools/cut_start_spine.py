#!/usr/bin/env python3
"""cut_start_spine.py - the START of a planner line as a pace-gated curriculum spine (pbfr1b2, 2026-10-29).

Takes plan_to_bc spines (per-tick STATE_DTYPE, one wave's best line each, with <spine>.keys.npy), picks the one
that reaches --level percent of the field's start distance soonest, and writes its rows 0..(first arrival +
--extra ticks) with their held keys. Used with train_fast --demo-success-pace (a spine episode succeeds only if it
reaches the line no slower than the spine did from its spawn row): the planner's own faster start taught for its
time. Nothing here reads a demo or a record; the level is a field fraction, the same rule on any map.

    python tools/cut_start_spine.py --field F.npz --start-d 57149 --out OUT.npy runs/plan/r3/spine_*.npy
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from surfgym.goalfield import GoalField  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("spines", nargs="+")
ap.add_argument("--field", required=True)
ap.add_argument("--start-d", type=float, required=True)
ap.add_argument("--level", type=float, default=1.6, help="percent of the start distance")
ap.add_argument("--extra", type=int, default=0, help="ticks kept past the first arrival")
ap.add_argument("--out", required=True)
a = ap.parse_args()
z = np.load(a.field, allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]),
              float(z["reach_max"]))
best = None
for p in a.spines:
    s = np.load(p)
    d = f.sample(np.asarray(s["origin"], np.float64))
    prog = 100.0 * (a.start_d - d) / a.start_d
    i = np.flatnonzero(prog >= a.level)
    if len(i) and (best is None or i[0] < best[1]):
        best = (p, int(i[0]), s)
if best is None:
    raise SystemExit(f"no spine reaches {a.level}%")
p, i, s = best
end = min(len(s), i + 1 + a.extra)
k = np.load(str(Path(p).with_suffix("")) + ".keys.npy")
np.save(a.out, s[:end])
np.save(str(Path(a.out).with_suffix("")) + ".keys.npy", k[:end])
print(f"{p}: {a.level}% at row {i}; wrote rows 0..{end - 1} -> {a.out}")
