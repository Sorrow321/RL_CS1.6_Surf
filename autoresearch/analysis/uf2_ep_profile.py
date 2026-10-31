# MEASUREMENT: one eval episode's path every STEP ticks (position, speed, energy, progress, contact).
import sys, json
import numpy as np
sys.path.insert(0, "/home/steam/ar_uf2/wt/b2/python"); sys.path.insert(0, "/home/steam/ar_uf2/wt/b2/autoresearch")
import score as S
path, epi = sys.argv[1], int(sys.argv[2]); step = int(sys.argv[3]) if len(sys.argv) > 3 else 25
f = S.field()
for h, r, tr in S.episodes(__import__("pathlib").Path(path)):
    if h.get("episode") != epi: continue
    p = r[:, 1:4]; v = r[:, 4:7]
    d = f.sample(p).astype(np.float64); prog = 100 * (d[0] - d) / d[0]
    E = p[:, 2] + (v ** 2).sum(1) / 1600
    dvz = np.diff(v[:, 2]) + 8.0 * float(h.get("tick_ms", 10)) / 10.0
    cont = np.r_[False, dvz > 1.0]
    tick = float(h.get("tick_ms", 10)) / 1000
    for k in range(0, len(r), step):
        c = cont[max(0, k - step // 2):k + step // 2 + 1].mean()
        print(f"t {k*tick:5.2f} pos ({p[k,0]:6.0f},{p[k,1]:6.0f},{p[k,2]:5.0f}) |v| {np.linalg.norm(v[k]):5.0f} vz {v[k,2]:6.0f} E {E[k]:5.0f} prog {prog[k]:6.2f}% contact {c:4.0%}")
    print("end", tr.get("end"), "len", len(r))
