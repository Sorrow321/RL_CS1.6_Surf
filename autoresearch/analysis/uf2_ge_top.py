# Where are a Go-Explore archive's deepest cells on the grader field? (our own search; measurement)
import sys, json
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
arch = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
a = np.load(arch, allow_pickle=False)
st = a["state"]
o = st["origin"].astype(np.float64); v = st["velocity"].astype(np.float64)
d = f.sample(o).astype(np.float64)
d[d >= f.reach_max - 0.5 * f.cell] = np.inf
d0 = 30647.5
prog = 100 * (d0 - d) / d0
E = o[:, 2] + (v ** 2).sum(1) / 1600.0
sp = np.linalg.norm(v, axis=1)
print("files", a.files, "cells", len(st))
print(f"progress pct quantiles: 50% {np.nanpercentile(prog[np.isfinite(prog)],50):.2f} 90% {np.nanpercentile(prog[np.isfinite(prog)],90):.2f} 99% {np.nanpercentile(prog[np.isfinite(prog)],99):.2f} max {prog[np.isfinite(prog)].max():.2f}")
# below-platform (z < 1000) cells that have positive progress beyond 2%
m = np.isfinite(prog)
low = m & (o[:, 2] < 900)
print(f"cells z<900: {low.sum():,}; of them prog>2%: {(low & (prog>2)).sum():,}; prog>3%: {(low & (prog>3)).sum():,}; prog>5%: {(low & (prog>5)).sum():,}")
print(f"cells in pit (prog < -1%): {(m & (prog < -1)).sum():,}; deepest pit point prog min {prog[m].min():.2f}")
print(f"max E overall {E.max():.0f}; max E in pit (prog<-1) {E[m & (prog < -1)].max() if (m & (prog<-1)).any() else float('nan'):.0f}")
idx = np.argsort(-np.where(m, prog, -1e9))[:n]
dep = a["depth"] if "depth" in a.files else None
par = a["parent"] if "parent" in a.files else None
for i in idx:
    print(f"cell {i:8d} prog {prog[i]:6.2f}% d {d[i]:7.0f} pos ({o[i,0]:6.0f},{o[i,1]:6.0f},{o[i,2]:6.0f}) |v| {sp[i]:5.0f} vz {v[i,2]:6.0f} E {E[i]:6.0f}" + (f" depth {int(dep[i])}" if dep is not None else ""))
