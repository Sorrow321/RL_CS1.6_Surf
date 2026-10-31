# MEASUREMENT: energy height reached per field-progress band, for chains that dove (chain-min < DIP).
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
arch = sys.argv[1]; dip = float(sys.argv[2]) if len(sys.argv) > 2 else -5.0
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
a = np.load(arch, allow_pickle=False)
st, par, dep = a["state"], a["parent"].astype(np.int64), a["depth"]
o = st["origin"].astype(np.float64); v = st["velocity"].astype(np.float64)
d = f.sample(o).astype(np.float64); d[d >= f.reach_max - 0.5 * f.cell] = np.inf
prog = 100 * (30647.5 - d) / 30647.5
E = o[:, 2] + (v ** 2).sum(1) / 1600.0
n = len(st); cmin = prog.copy()
for i in range(n):
    p = par[i]
    if p >= 0: cmin[i] = min(cmin[i], cmin[p])
print("cells", n, "max depth ticks", int(dep.max()))
dove = np.isfinite(prog) & (cmin < dip)
# after the chain's own minimum: is the chain now climbing back (prog > cmin + 2)?
for lo in np.arange(-12, 6, 1.0):
    k = dove & (prog >= lo) & (prog < lo + 1)
    k2 = np.isfinite(prog) & (prog >= lo) & (prog < lo + 1)
    if k2.any():
        i = np.argmax(np.where(k, E, -1e9)) if k.any() else None
        s = (f"  dove: n {k.sum():7,} maxE {E[i]:6.0f} at ({o[i,0]:6.0f},{o[i,1]:6.0f},{o[i,2]:5.0f}) |v| {np.linalg.norm(v[i]):5.0f} t {dep[i]/100:5.2f}" if i is not None else "")
        print(f"prog [{lo:5.1f},{lo+1:5.1f}) all: n {k2.sum():7,} maxE {E[k2].max():6.0f}" + s)
