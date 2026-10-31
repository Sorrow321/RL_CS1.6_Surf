# MEASUREMENT: per archive, the earliest-arriving cell (fewest chain ticks) at each field-progress level.
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
for arch in sys.argv[1:]:
    a = np.load(arch, allow_pickle=False)
    st, dep = a["state"], a["depth"].astype(np.int64)
    o = st["origin"].astype(np.float64); v = st["velocity"].astype(np.float64)
    d = f.sample(o).astype(np.float64); d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    prog = 100 * (30647.5 - d) / 30647.5
    E = o[:, 2] + (v ** 2).sum(1) / 1600.0
    print(arch.split("/")[-2], len(st), "cells")
    for lvl in (4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.8, 8.0):
        k = np.isfinite(prog) & (prog >= lvl)
        if not k.any():
            print(f"  >= {lvl:4.1f}%: none"); continue
        i = np.flatnonzero(k)[np.argmin(dep[k])]
        print(f"  >= {lvl:4.1f}%: n {k.sum():6d}  earliest cell {i:7d} t {dep[i]/100:5.2f}s pos ({o[i,0]:6.0f},{o[i,1]:6.0f},{o[i,2]:5.0f}) |v| {np.linalg.norm(v[i]):5.0f} E {E[i]:5.0f} prog {prog[i]:5.2f}%")
