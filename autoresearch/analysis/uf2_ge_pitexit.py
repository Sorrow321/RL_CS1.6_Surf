# MEASUREMENT: which archive chains went down into the pit (low progress) and came back out (our own search).
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
n = len(st)
cmin = prog.copy()
for i in range(n):
    p = par[i]
    if p >= 0:
        cmin[i] = min(cmin[i], cmin[p])
m = np.isfinite(prog) & (cmin < dip)
print(f"cells whose chain went below {dip}%: {m.sum():,}")
for thr in (-3, -1, 0, 1, 2, 3, 4, 5):
    k = m & (prog > thr)
    print(f"  ... and are now above {thr:3d}%: {k.sum():8,}" + (f"  max E {E[k].max():6.0f}  max |v| {np.linalg.norm(v[k],axis=1).max():5.0f}" if k.any() else ""))
k = m & (prog > -3)
idx = np.argsort(-np.where(k, prog, -1e9))[:12]
for i in idx:
    if not k[i]: break
    print(f"cell {i:7d} t {int(dep[i])/100:5.2f}s prog {prog[i]:6.2f}% chain-min {cmin[i]:6.2f}% pos ({o[i,0]:6.0f},{o[i,1]:6.0f},{o[i,2]:6.0f}) |v| {np.linalg.norm(v[i]):5.0f} vz {v[i,2]:6.0f} E {E[i]:6.0f}")
