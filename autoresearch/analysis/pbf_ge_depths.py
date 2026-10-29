"""pbf_ge_depths.py - for a GE archive rooted at the true start: the SHORTEST chain (fewest ticks
from the start) that reaches each progress level on the grader field (pbfr1b1, 2026-10-29).

    python autoresearch/analysis/pbf_ge_depths.py runs/<ge>/archive.npz [levels...]

Measurement only: it says how long the explorer's quickest way to each depth is, which is how a
curriculum spine's end is picked (earliest arrival at a level, not the deepest meander)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

F, D0 = score.field(), score.START_D
fn = _args[0]
levels = [float(x) for x in _args[1:]] or [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]
z = np.load(fn)
st = z["state"]
o = st["origin"].astype(np.float64)
v = st["velocity"].astype(np.float64)
h = o[:, 2] + (v ** 2).sum(1) / 1600.0
d = F.sample(o).astype(np.float64)
prog = np.where(d < F.reach_max - 0.5 * F.cell, 100.0 * (D0 - d) / D0, -99.0)
depth = z["depth"]
for lv in levels:
    m = np.flatnonzero(prog >= lv)
    if not len(m):
        print(f"{lv:5.2f}%: none")
        continue
    i = m[np.argmin(depth[m])]
    print(f"{lv:5.2f}%: {len(m):7,} cells; shortest chain {int(depth[i]):6d} ticks -> cell {int(i)} at "
          f"{o[i].round(0).tolist()} prog {prog[i]:.2f}% h {h[i]:.0f} |v| {np.linalg.norm(v[i]):.0f}")
