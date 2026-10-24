#!/usr/bin/env python3
"""pb_field_descent.py x y z [drop] - follow the grader field's steepest 26-neighbour descent from a
point (voxel by voxel) for `drop` u of d, printing the path: where the field says the way on is.
Analysis only."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "autoresearch"))
import score as S
f = S.field()
p = np.array([float(v) for v in sys.argv[1:4]])
drop = float(sys.argv[4]) if len(sys.argv) > 4 else 4000.0
c = f.cell
offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                 if (i, j, k) != (0, 0, 0)], np.float64) * c
d0 = float(f.sample(p[None])[0])
print(f"start ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) d {d0:,.0f} ({100*(57149-d0)/57149:.2f}%)")
last = d0
for step in range(2000):
    q = p[None] + offs
    dq = f.sample(q).astype(np.float64)
    j = int(np.argmin(dq))
    if dq[j] >= last:
        print("local minimum"); break
    p, last = q[j], float(dq[j])
    if step % 8 == 0:
        print(f"  ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) d {last:,.0f} ({100*(57149-last)/57149:.2f}%)")
    if d0 - last > drop:
        break
