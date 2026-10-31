# MEASUREMENT: for the deepest cells of a rooted archive, the energy / progress of the ROOT each chain started from,
# and per root-energy band the best progress reached (how much trough energy a deep exit needs).
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
a = np.load(sys.argv[1], allow_pickle=False)
st, par, dep = a["state"], a["parent"].astype(np.int64), a["depth"]
o = st["origin"].astype(np.float64); v = st["velocity"].astype(np.float64)
d = f.sample(o).astype(np.float64); d[d >= f.reach_max - 0.5 * f.cell] = np.inf
prog = 100 * (30647.5 - d) / 30647.5
E = o[:, 2] + (v ** 2).sum(1) / 1600.0
n = len(st); root = np.arange(n)
for i in range(n):
    if par[i] >= 0: root[i] = root[par[i]]
for lo, hi in ((600, 700), (700, 800), (800, 900), (900, 950), (950, 1000), (1000, 1100)):
    k = np.isfinite(prog) & (E[root] >= lo) & (E[root] < hi)
    if k.any():
        i = np.flatnonzero(k)[np.argmax(prog[k])]
        print(f"root E [{lo},{hi}): cells {k.sum():7d}  best {prog[i]:6.2f}% at ({o[i,0]:6.0f},{o[i,1]:6.0f},{o[i,2]:5.0f}) root {root[i]} E {E[root[i]]:5.0f} prog {prog[root[i]]:5.2f}% at ({o[root[i],0]:6.0f},{o[root[i],1]:6.0f},{o[root[i],2]:5.0f}) chain {int(dep[i]-dep[root[i]])} ticks")
