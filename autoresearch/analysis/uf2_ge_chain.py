# The chain behind one archive cell: positions, energy and field progress per link (our own search).
import sys
import numpy as np
sys.path.insert(0, "/root/RL_Surf/python")
from surfgym.goalfield import GoalField
arch = sys.argv[1]; cell = int(sys.argv[2])
z = np.load("/root/RL_Surf/maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
a = np.load(arch, allow_pickle=False)
st, par, dep = a["state"], a["parent"], a["depth"]
ch = []
i = cell
while i >= 0:
    ch.append(int(i)); i = int(par[i])
ch = ch[::-1]
d0 = 30647.5
for c in ch:
    o = st[c]["origin"].astype(np.float64); v = st[c]["velocity"].astype(np.float64)
    d = float(f.sample(o[None])[0]); E = o[2] + (v ** 2).sum() / 1600
    print(f"cell {c:7d} t {int(dep[c])/100:5.2f}s pos ({o[0]:6.0f},{o[1]:6.0f},{o[2]:6.0f}) |v| {np.linalg.norm(v):5.0f} vz {v[2]:6.0f} E {E:6.0f} prog {100*(d0-d)/d0:6.2f}%")
