"""ANALYSIS ONLY: harvest the greedy line's own launch-attempt states from eval trajectories
(the state 0.4 s before the first tick rising at vz > 300 near the holes), as a STATE_DTYPE npy."""
import sys, glob, json, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'python')); sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
import score
from surfgym.core import STATE_DTYPE
out = []
for f in sys.argv[2:]:
    for fn in sorted(glob.glob(f)):
        for h, r, tr in score.episodes(fn):
            p = r[:, 1:4]; v = r[:, 4:7]
            att = np.flatnonzero((p[:, 0] > 2440) & (p[:, 0] < 2720) & (v[:, 2] > 300) & (p[:, 2] > 980) & (p[:, 2] < 1180))
            if not len(att):
                continue
            for back in (20, 40, 60):
                t = max(0, att[0] - back)
                s = np.zeros(1, STATE_DTYPE)[0]
                s['origin'] = p[t]; s['velocity'] = v[t]; s['yaw'] = r[t, 7]; s['pitch'] = 0.0
                s['onground'] = -1
                out.append(s)
A = np.array(out, dtype=STATE_DTYPE)
np.save(sys.argv[1], A)
print(len(A), 'states')
