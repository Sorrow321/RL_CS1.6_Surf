import json, sys, math
import numpy as np
def episodes(path):
    eps, hdr, cur = [], None, []
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append((np.asarray(cur, float), d)); hdr, cur = None, []
            else: hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps
def wrap(a): return (a + 180) % 360 - 180
for path in sys.argv[1:]:
    print("==", path.split("/")[-1])
    for k, (a, tr) in enumerate(episodes(path)):
        p = a[:, 1:4]; v = a[:, 4:7]
        # first basement entry, then the first U-turn: x > 3300 inside the basement within 4 s of entry
        ib = np.flatnonzero((p[:, 0] > 2280) & (p[:, 2] < 1250))
        if not len(ib): continue
        i0 = ib[0]; i1 = min(len(a), i0 + 400)
        seg = np.arange(i0, i1)
        u = seg[p[seg, 0] > 3300]
        if not len(u): continue
        hd = np.degrees(np.arctan2(v[u, 1], v[u, 0]))
        off = wrap(a[u, 7] - hd)
        vh = np.hypot(v[u, 0], v[u, 1])
        side = a[u, 14]
        # the west leg after the turn: x 2600-3200 heading west within the next 2.5 s
        w = np.arange(u[-1], min(len(a), u[-1] + 250))
        w = w[(p[w, 0] > 2600) & (p[w, 0] < 3200) & (v[w, 0] < -300)]
        wl = (f"west leg |vh| {np.hypot(v[w,0],v[w,1]).mean():5.0f} z {p[w,2].mean():5.0f} |y| {np.abs(p[w,1]).mean():4.0f}"
              if len(w) else "west leg -")
        print(f" ep{k} t_in {a[i0,0]/100:6.1f}  U-turn: offset med {np.median(off):+5.2f} deg  side-left {np.mean(side==0):.0%}  zmax {p[u,2].max():5.0f}  xmax {p[u,0].max():5.0f}  |vh| min {vh.min():5.0f}  | {wl}")
