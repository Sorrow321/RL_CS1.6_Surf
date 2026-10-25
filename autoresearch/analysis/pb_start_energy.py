"""The start, per greedy eval (pbr3b4, 2026-10-25): does the line pump before the tunnel?

    python autoresearch/analysis/pb_start_energy.py runs/<agent>/<run>/traj_*.jsonl

Per eval (first 9 episodes): how many episodes go EAST of the spawn (max x over the first 10 s
above 1,100 - the spawn is at x 860, the tunnel west of it), and the median energy height
h = z + |v|^2/1600 and time at each episode's first arrival at 1.4% progress (the tunnel exit;
the record arrives at ~3,950 by ANALYSIS ONLY). Measurement only, never a reward quantity."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

F, D0 = score.field(), score.START_D
for fn in _args:
    east, hs, ts = 0, [], []
    for h, r, tr in score.first_episodes(score.episodes(fn), score.SCORE_EPISODES):
        p, v = r[:, 1:4], r[:, 4:7]
        if p[:1000, 0].max() > 1100:
            east += 1
        d = F.sample(p).astype(np.float64)
        prog = 100 * (D0 - d) / D0
        i = np.flatnonzero(prog >= 1.4)
        if len(i):
            hs.append(p[i[0], 2] + (v[i[0]] ** 2).sum() / 1600.0)
            ts.append(i[0] / 100.0)
    print(f"{Path(fn).name}: {east} east of the spawn; tunnel exit (1.4%) h median "
          f"{np.median(hs) if hs else float('nan'):5.0f} at {np.median(ts) if ts else float('nan'):5.2f} s")
