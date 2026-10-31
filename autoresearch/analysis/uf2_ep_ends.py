# MEASUREMENT: per-episode path summary of a greedy eval (min z, max speed, end, best progress).
import sys, json
import numpy as np
sys.path.insert(0, "/home/steam/ar_uf2/wt/b2/python")
sys.path.insert(0, "/home/steam/ar_uf2/wt/b2/autoresearch")
import score as S
f = S.field()
for path in sys.argv[1:]:
    print(path.split("/")[-1])
    for h, r, tr in S.first_episodes(S.episodes(__import__("pathlib").Path(path)), 9):
        p = r[:, 1:4]; v = r[:, 4:7]
        d = f.sample(p).astype(np.float64)
        prog = 100 * (d[0] - d) / d[0]
        sp = np.linalg.norm(v, axis=1)
        E = p[:, 2] + (v ** 2).sum(1) / 1600
        ib = int(np.argmax(prog))
        tick = float(h.get("tick_ms", 10.0)) / 1000
        print(f"  ep{h.get('episode')} {tr.get('end'):5s} t {len(r)*tick:5.1f}s best {prog.max():5.2f}% at t {ib*tick:4.1f} pos ({p[ib,0]:6.0f},{p[ib,1]:6.0f},{p[ib,2]:5.0f}) minz {p[:,2].min():6.0f} vmax {sp.max():5.0f} end ({p[-1,0]:6.0f},{p[-1,1]:6.0f},{p[-1,2]:5.0f}) Emax {E.max():5.0f}")
