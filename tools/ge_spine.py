#!/usr/bin/env python3
"""ge_spine.py - a curriculum spine from the policy's own beam line + a Go-Explore exact chain
(pbr3b4, 2026-10-24): line[:root] + the chain (its root as found - possibly a speed-boosted copy
of the line's state - and the explorer's states re-aimed along the velocity,
tools/spine_canon_view.py), cut at the best d on the field. Prints a JSON summary.

    python tools/ge_spine.py runs/beam/beam_frontier.npz runs/ge/line.npy out.npy [field.npz]
"""
import sys, json, hashlib, numpy as np
sys.path.insert(0, 'tools'); sys.path.insert(0, 'python')
from spine_canon_view import canon_view
from surfgym.core import STATE_DTYPE
from surfgym.goalfield import GoalField
beam_npz, chain_npy, out = sys.argv[1], sys.argv[2], sys.argv[3]
field = sys.argv[4] if len(sys.argv) > 4 else "maps/surf_princessburglar_b26.goalc_32.npz"
line = np.load(beam_npz)['states']
gl = np.load(chain_npy)
o = np.asarray(gl[0]['origin'], np.float64)
dist = np.linalg.norm(line['origin'].astype(np.float64) - o, axis=1)
j = int(np.argmin(dist))
assert dist[j] < 1.0, dist[j]
rest, m = canon_view(gl[1:])
sp = np.concatenate([np.asarray(line[:j], STATE_DTYPE), np.asarray(gl[:1], STATE_DTYPE),
                     np.asarray(rest, STATE_DTYPE)])
zf = np.load(field, allow_pickle=False)
f = GoalField(zf['grid'].astype(np.float32) * float(zf['quant']), zf['mins'], float(zf['cell']), float(zf['reach_max']))
d = f.sample(sp['origin'].astype(np.float64)).astype(np.float64)
d[d >= f.reach_max - 0.5 * f.cell] = np.inf
e = int(np.argmin(d))
sp = np.array(sp[:e + 1], copy=True)
sp['tick'] = 0; sp['stuck_ticks'] = 0
np.save(out, sp)
D0 = 57149.125
boost = float(np.linalg.norm(gl[0]['velocity']) / max(1e-6, np.linalg.norm(line[j]['velocity'])))
info = {"root_index": j, "root_boost": round(boost, 3), "n": int(len(sp)), "end_d": float(d[e]),
        "end_pct": round(100 * (D0 - d[e]) / D0, 2), "chain_len": int(len(gl)),
        "md5": hashlib.md5(open(out, 'rb').read()).hexdigest(), "canon_rows": int(m.sum())}
print(json.dumps(info))
