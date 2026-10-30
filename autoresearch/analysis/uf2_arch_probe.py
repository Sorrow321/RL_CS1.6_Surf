# ANALYSIS ONLY: what a GE archive holds on uf2, read against the grader field and (as a
# measuring stick only, CLAUDE.md section 0) the WR line's positions/velocities.
import json, sys, numpy as np
sys.path.insert(0, "python")
from surfgym.goalfield import GoalField
z = np.load("maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
D0 = 30647.5
a = np.load(sys.argv[1], allow_pickle=True)
st = a["state"]; dep = a["depth"]
p = st["origin"].astype(np.float64); v = st["velocity"].astype(np.float64)
d = f.sample(p).astype(np.float64)
ok = d < f.reach_max - 16
E = p[:, 2] + (v ** 2).sum(1) / 1600
prog = 100 * (D0 - d) / D0
print(f"cells {len(st):,}  on-field {ok.sum():,}  depth max {dep.max()} ticks")
o = np.argsort(-np.where(ok, prog, -1e9))[:12]
print("top progress cells:")
for i in o:
    print(f"  prog {prog[i]:6.2f}%  pos ({p[i,0]:6.0f},{p[i,1]:6.0f},{p[i,2]:6.0f}) v ({v[i,0]:6.0f},{v[i,1]:6.0f},{v[i,2]:6.0f}) |v| {np.linalg.norm(v[i]):5.0f} E {E[i]:6.0f} depth {dep[i]/100:5.2f}s")
pit = (p[:,0] > -2900) & (p[:,0] < -1470) & (p[:,1] > -3940) & (p[:,1] < 200) & (p[:,2] < 400)
print(f"pit cells (z<400): {pit.sum():,}; E max {E[pit].max() if pit.any() else 0:.0f}; z min {p[pit,2].min() if pit.any() else 0:.0f}")
for zc in (400, 0, -400, -700):
    m = pit & (p[:,2] < zc)
    if m.any():
        print(f"  z<{zc}: {m.sum():,} cells, E max {E[m].max():.0f}, E p90 {np.percentile(E[m],90):.0f}, speed max {np.linalg.norm(v[m],axis=1).max():.0f}")
# WR as a ruler
rows = []
for k, line in enumerate(open(sys.argv[2])):
    if k == 0: continue
    line = line.strip()
    if not line or line[0] == "{": continue
    rows.append(json.loads(line))
r = np.asarray(rows, np.float64); rp = r[:,1:4]; rv = r[:,4:7]; rE = rp[:,2] + (rv**2).sum(1)/1600
print("same-motion best archive E near the record (200 u, cos>0.8):  t  recE  archE  (n)")
vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
for t in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.9, 8.5, 9.0, 10.0, 11.0]:
    i = int(t * 100)
    near = np.linalg.norm(p - rp[i], axis=1) < 200
    rvn = rv[i] / max(np.linalg.norm(rv[i]), 1e-6)
    sm = near & ((vn @ rvn) > 0.8)
    print(f"  t {t:4.1f}  {rE[i]:6.0f}  {E[sm].max() if sm.any() else float('nan'):6.0f}  ({sm.sum()})  any-motion {E[near].max() if near.any() else float('nan'):6.0f}")
