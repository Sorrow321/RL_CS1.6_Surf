# MEASUREMENT: a STATE_DTYPE line's per-step position / speed / energy / field progress (our own search line).
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
s = np.load(sys.argv[1], allow_pickle=False); step = int(sys.argv[2]) if len(sys.argv) > 2 else 50
o = s["origin"].astype(np.float64); v = s["velocity"].astype(np.float64)
d = f.sample(o).astype(np.float64); prog = 100 * (30647.5 - d) / 30647.5
E = o[:, 2] + (v ** 2).sum(1) / 1600.0
# contact: vz rose above the free-fall step (the map pushed back) - the --surf-bonus / pick_selfline rule
dvz = np.diff(v[:, 2]) + 8.0      # free fall: vz drops 800 * 0.01 = 8 u/s per 10 ms tick
cont = np.r_[False, dvz > 1.0]
for k in range(0, len(s), step):
    c = cont[max(0, k - step // 2):k + step // 2 + 1].mean()
    print(f"t {k/100:5.2f} pos ({o[k,0]:6.0f},{o[k,1]:6.0f},{o[k,2]:5.0f}) |v| {np.linalg.norm(v[k]):5.0f} vh {np.hypot(v[k,0],v[k,1]):5.0f} vz {v[k,2]:6.0f} E {E[k]:6.0f} prog {prog[k]:6.2f}% contact {c:4.0%}")
print(f"best prog {prog.max():.2f}% at t {np.argmax(prog)/100:.2f}; last contact tick {np.flatnonzero(cont)[-1] if cont.any() else -1}")
