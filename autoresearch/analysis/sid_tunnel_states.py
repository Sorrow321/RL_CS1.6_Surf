import sys, numpy as np, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
rr = next(iter(ck["respawn_rare"].values()))
st = rr["arch_store"]; ring = int(rr["ring"]); fill = np.asarray(rr["arch_fill"]); nb = len(rr["arch_cell"])
o = np.asarray(st["origin"], np.float64).reshape(nb, ring, 3); v = np.asarray(st["velocity"], np.float64).reshape(nb, ring, 3)
ok = np.arange(ring)[None, :] < fill[:, None]
o, v = o[ok], v[ok]
m = (o[:,0] > 1900) & (o[:,0] < 2230) & (np.abs(np.abs(o[:,1]) - 900) < 90) & (o[:,2] > 10960) & (o[:,2] < 11200)
print("archived states at/in the window tunnels:", int(m.sum()))
# can the state cross the rest of the tunnel ballistically? need z(x=1936) >= 10994 (standing; ducked 10976)
for i in np.flatnonzero(m)[np.argsort(o[m,0])][:20]:
    x, y, z = o[i]; vx, vy, vz = v[i]
    if vx < -50:
        t = (x - 1936) / -vx; ze = z + vz * t - 400 * t * t
    else:
        ze = float('nan')
    print(f"  pos ({x:6.0f},{y:6.0f},{z:6.0f})  v ({vx:6.0f},{vy:6.0f},{vz:6.0f})  z at tunnel exit {ze:7.0f}")
