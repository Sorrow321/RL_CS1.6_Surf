import sys, numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
rr = next(iter(ck["respawn_rare"].values()))
st = rr["arch_store"]; ring = int(rr["ring"]); fill = np.asarray(rr["arch_fill"])
nb = len(rr["arch_cell"])
org = np.asarray(st["origin"], np.float64).reshape(nb, ring, 3)
vel = np.asarray(st["velocity"], np.float64).reshape(nb, ring, 3)
ok = np.arange(ring)[None, :] < fill[:, None]
o, v = org[ok], vel[ok]
sp = np.linalg.norm(v, axis=1)
def rep(name, m):
    if not m.any(): print(f"{name:40s} 0"); return
    print(f"{name:40s} n {m.sum():4d}  |v| p50 {np.median(sp[m]):6.0f} p90 {np.percentile(sp[m],90):6.0f} max {sp[m].max():6.0f}  vz>0 {np.mean(v[m,2]>0):.0%}  z min {o[m,2].min():6.0f}")
below = (o[:,0]>1700)&(o[:,0]<2990)&(o[:,2]>9500)&(o[:,2]<10650)
bowl = (o[:,0]>1950)&(o[:,0]<2800)&(o[:,1]>-700)&(o[:,1]<200)&(o[:,2]>9700)&(o[:,2]<10950)
cav = (o[:,0]>2580)&(o[:,0]<2990)&(o[:,2]>10650)&(o[:,2]<11260)
gap = (o[:,0]>2400)&(o[:,0]<2700)&(np.abs(o[:,1])<300)&(o[:,2]>10750)&(o[:,2]<11100)
rep("below shaft", below); rep("human bowl box", bowl); rep("cavity", cav); rep("centre gap", gap)
# fastest states below the shaft
idx = np.flatnonzero(below)
for i in idx[np.argsort(-sp[idx])][:6]:
    print("  fast below:", np.round(o[i]).astype(int).tolist(), np.round(v[i]).astype(int).tolist())
idx = np.flatnonzero(below & (v[:,2] > 0))
print("climbing below-shaft states:", len(idx))
for i in idx[np.argsort(-o[idx,2])][:6]:
    print("  climbing:", np.round(o[i]).astype(int).tolist(), np.round(v[i]).astype(int).tolist())
