#!/usr/bin/env python3
"""ge_roots_best.py - per-root best progress of a tools/explore_phase1.py --exact-chains archive
(pbr3b4, 2026-10-24): which roots (and, with roots stacked in groups of N - e.g. speed-boosted
copies - which group) reached how deep on the grader field.

    python tools/ge_roots_best.py runs/ge/archive.npz N_PER_GROUP [field.npz]
"""
import sys, numpy as np
sys.path.insert(0, 'python')
from surfgym.goalfield import GoalField
arch, ngroup = sys.argv[1], int(sys.argv[2])
field = sys.argv[3] if len(sys.argv) > 3 else "maps/surf_princessburglar_b26.goalc_32.npz"
z = np.load(arch, allow_pickle=False)
st, parent = z['state'], z['parent']
zf = np.load(field, allow_pickle=False)
f = GoalField(zf['grid'].astype(np.float32) * float(zf['quant']), zf['mins'], float(zf['cell']), float(zf['reach_max']))
d = f.sample(st['origin'].astype(np.float64)).astype(np.float64)
d[d >= f.reach_max - 0.5 * f.cell] = np.inf
n = len(st); root = np.arange(n)
for _ in range(200):
    p = parent[root]; m = p >= 0
    if not m.any(): break
    root[m] = p[m]
D0 = 57149.125
rows = []
for rc in np.unique(root):
    m = root == rc
    k = int(np.argmin(np.where(m, d, np.inf)))
    rows.append((rc, int(np.linalg.norm(st[rc]['velocity'])), np.round(st[rc]['origin']), int(m.sum()), round(100*(D0-d[k])/D0, 2), k))
for r in rows:
    print('root', r[0], 'group', r[0] // ngroup, 'speed', r[1], 'pos', r[2], 'cells', r[3], 'best', r[4], 'cell', r[5])
