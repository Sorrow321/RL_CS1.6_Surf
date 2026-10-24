#!/usr/bin/env python3
"""pb_arch_report.py <archive.npz> [n] - deepest cells of a Go-Explore archive on the grader field
(progress %), with chain length (ticks from the root) and the root index. Analysis only."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "autoresearch"))
import score as S
z = np.load(sys.argv[1], allow_pickle=False)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
f = S.field()
st, par, dep = z["state"], z["parent"], z["depth"]
d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
d[d >= f.reach_max - 0.5 * f.cell] = np.inf
p = 100 * (57149.0 - d) / 57149.0
o = np.argsort(d)
print(f"{len(st):,} cells; past 3.3%: {(p > 3.3).sum():,}; past 3.5%: {(p > 3.5).sum():,}; "
      f"past 4%: {(p > 4.0).sum():,}; best {p[o[0]]:.3f}%")
def chain(i):
    c = []
    while i >= 0:
        c.append(int(i)); i = int(par[i])
    return c[::-1]
for i in o[:n]:
    c = chain(i)
    org = st[i]["origin"]; v = st[i]["velocity"]
    print(f"cell {i}: {p[i]:.3f}% ({org[0]:.0f},{org[1]:.0f},{org[2]:.0f}) |v| {np.linalg.norm(v):.0f} "
          f"depth {int(dep[i])} ticks, chain {len(c)} cells, root {c[0]}")
