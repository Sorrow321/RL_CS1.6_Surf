# MEASUREMENT ONLY (a bench file): archive cells in a progress window with energy height >= EMIN.
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
arch, out, lo, hi, emin = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
a = np.load(arch, allow_pickle=False); st = a["state"]
o = st["origin"].astype(np.float64); v = st["velocity"].astype(np.float64)
d = f.sample(o).astype(np.float64); prog = 100 * (30647.5 - d) / 30647.5
E = o[:, 2] + (v ** 2).sum(1) / 1600.0
k = (prog >= lo) & (prog < hi) & (E >= emin)
s = st[k].copy(); s["tick"] = 0; s["stuck_ticks"] = 0
np.save(out, s)
print(f"{k.sum()} rows -> {out}; E p50 {np.median(E[k]):.0f} max {E[k].max():.0f}; |v| p50 {np.median(np.linalg.norm(v[k],axis=1)):.0f}; z p50 {np.median(o[k,2]):.0f}")
