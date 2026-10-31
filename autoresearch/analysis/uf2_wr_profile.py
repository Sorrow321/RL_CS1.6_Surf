# ANALYSIS ONLY (CLAUDE.md section 0): the WR's first seconds on the grader field, as a yardstick.
import json, sys
import numpy as np
sys.path.insert(0, "/home/steam/ar_uf2/wt/b2/python")
from surfgym.goalfield import GoalField
z = np.load("/home/steam/ar_uf2/wt/b2/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
rows = []
for i, line in enumerate(open("/home/steam/ar_uf2/wr/surf_unitfarmer2_wr.jsonl")):
    if i == 0: continue
    if line.startswith("{"): break
    rows.append(json.loads(line))
r = np.asarray(rows, np.float64)
p = r[:, 1:4]; v = r[:, 4:7]
d = f.sample(p).astype(np.float64)
E = p[:, 2] + (v ** 2).sum(1) / 1600.0
sp = np.linalg.norm(v, axis=1)
d0 = 30647.5
step = int(sys.argv[1]) if len(sys.argv) > 1 else 50
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
for k in range(0, min(len(r), lim), step):
    print(f"t {k/100:5.2f} pos ({p[k,0]:7.0f},{p[k,1]:7.0f},{p[k,2]:6.0f}) |v| {sp[k]:6.0f} vz {v[k,2]:6.0f} E {E[k]:6.0f} d {d[k]:7.0f} prog {100*(d0-d[k])/d0:5.2f}%")
