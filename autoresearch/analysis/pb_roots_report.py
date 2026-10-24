#!/usr/bin/env python3
"""pb_roots_report.py <archive.npz> <roots_source.npy> [thr_pct] - for a Go-Explore archive rooted
on a line (--roots-spine), which line index each chain past thr_pct (grader field) starts from,
the best progress per root index, and the chain lengths. Analysis only (no map coordinates)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "autoresearch"))
import score as S
z = np.load(sys.argv[1], allow_pickle=False)
src = np.load(sys.argv[2], allow_pickle=False)
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 3.3
f = S.field()
st, par, dep = z["state"], z["parent"], z["depth"]
d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
d[d >= f.reach_max - 0.5 * f.cell] = np.inf
p = 100 * (57149.0 - d) / 57149.0
so = src["origin"].astype(np.float64)
def root_of(i):
    while par[i] >= 0:
        i = int(par[i])
    return i
deep = np.flatnonzero(p > thr)
print(f"{len(st):,} cells, {len(deep):,} past {thr}% (best {np.nanmax(p[np.isfinite(p)]):.3f}%)")
best = {}
for i in deep:
    r = root_of(int(i))
    k = int(np.argmin(np.linalg.norm(so - st[r]["origin"].astype(np.float64), axis=1)))
    if k not in best or p[i] > best[k][0]:
        best[k] = (p[i], int(i), int(dep[i]))
for k in sorted(best):
    pi, i, dp = best[k]
    o = st[i]["origin"]
    print(f"root line idx {k}: best {pi:.3f}% cell {i} ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f}) depth {dp} ticks")
