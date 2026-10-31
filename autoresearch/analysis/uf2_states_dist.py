# MEASUREMENT: a spawn-state file's progress / energy distribution on the grader field.
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
s = np.load(sys.argv[1], allow_pickle=False)
o = s["origin"].astype(np.float64); v = s["velocity"].astype(np.float64)
d = f.sample(o).astype(np.float64)
prog = 100 * (30647.5 - d) / 30647.5
E = o[:, 2] + (v ** 2).sum(1) / 1600.0
print(f"rows {len(s)}; z<100 (pit) {np.mean(o[:,2]<100):.1%}; E>1000 {np.mean(E>1000):.1%}; E>1000 & z<100 {np.mean((E>1000)&(o[:,2]<100)):.1%}")
for lo, hi in ((-20, -5), (-5, -2), (-2, 0), (0, 2), (2, 4), (4, 20)):
    k = (prog >= lo) & (prog < hi)
    if k.any():
        print(f"prog [{lo:3d},{hi:3d}): {k.mean():6.1%}  E p50 {np.median(E[k]):6.0f} p90 {np.percentile(E[k],90):6.0f}  |v| p50 {np.median(np.linalg.norm(v[k],axis=1)):5.0f}  z p50 {np.median(o[k,2]):6.0f}")
