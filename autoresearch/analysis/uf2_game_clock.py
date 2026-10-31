"""Game clock of finishing episodes: from the tick the player LEAVES the start trigger's clip hull
(model *16, after having been inside it) to the first tick inside the finish trigger's clip hull
(*23; the unrecorded final tick extrapolated along the velocity). Analysis only."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/steam/ar_uf2/wt/b4/autoresearch/analysis'); sys.path.insert(0, '/home/steam/ar_uf2/wt/b4/autoresearch')
import uf2_real_finish as R
v, planes, clip, models = R.load_hulls(Path('/home/steam/ar_uf2/wt/b4/maps/surf_unitfarmer2.bsp'))
def inside(mi, p):
    return any(R.hull_contents(planes, clip, int(models[mi]['head'][h]), p) == -2 for h in (1, 3))
for path in sys.argv[1:]:
    eps, cur, hdr = [], [], None
    for line in open(path):
        if line.startswith('['): cur.append(json.loads(line))
        elif line.startswith('{'):
            d = json.loads(line)
            if 'end' in d: eps.append((hdr, np.array(cur), d)); cur = []
            else: hdr = d
    out = []
    for h, r, tr in eps[:9]:
        dt = float(h.get('tick_ms', 10.0)) / 1000.0
        p, vel = r[:, 1:4], r[:, 4:7]
        ins = [inside(16, p[i]) for i in range(min(len(p), 600))]
        if not any(ins): out.append(('nostart', None)); continue
        last_in = max(i for i, x in enumerate(ins) if x)
        t0 = (last_in + 1) * dt
        fin = None
        for i in range(max(0, len(p) - 400), len(p)):
            if inside(23, p[i]): fin = i; break
        if fin is None:
            for k in (1, 2):
                if inside(23, p[-1] + vel[-1] * dt * k): fin = len(p) - 1 + k; break
        if tr.get('end') != 'done' or fin is None: out.append((tr.get('end'), None)); continue
        out.append(('done', round(fin * dt - t0, 3), round(len(r) * dt, 2)))
    print(Path(path).name, out)
