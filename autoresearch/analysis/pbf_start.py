"""Per greedy eval (pbfr1b2, 2026-10-29): how the line leaves the start room on the corrected core.

    python autoresearch/analysis/pbf_start.py runs/<agent>/<run>/traj_*.jsonl

For the first 9 episodes: did it go east (max x over the first 10 s > 1,100: the half-pipe), did it
stand on the doorway's conveyor slabs (feet at 3,657 = *249 / 3,681 = *250 within the doorway's x/y
box), the max progress on the grader field, and the energy height h = z + |v|^2/1600 and time at
the first arrival at 1.4% (the passage / tunnel exits) and at 2.5% (the pit). Measurement only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

F, D0 = score.field(), score.START_D
for fn in _args:
    east = slab = 0
    best, h14, t14, h25, t25 = [], [], [], [], []
    for h, r, tr in score.first_episodes(score.episodes(fn), score.SCORE_EPISODES):
        tick_s = float(h.get("tick_ms", 10.0)) / 1000.0
        p, v = r[:, 1:4], r[:, 4:7]
        n10 = int(10.0 / tick_s)
        if p[:n10, 0].max() > 1100:
            east += 1
        door = (p[:, 0] > 630) & (p[:, 0] < 735) & (p[:, 1] > -3060) & (p[:, 1] < -2940)
        if (door & (p[:, 2] > 3690)).any():
            slab += 1
        d = F.sample(p).astype(np.float64)
        prog = 100 * (D0 - d) / D0
        best.append(prog.max())
        for lvl, hs, ts in ((1.4, h14, t14), (2.5, h25, t25)):
            i = np.flatnonzero(prog >= lvl)
            if len(i):
                hs.append(p[i[0], 2] + (v[i[0]] ** 2).sum() / 1600.0)
                ts.append(i[0] * tick_s)
    med = lambda a: float(np.median(a)) if a else float("nan")  # noqa: E731
    print(f"{Path(fn).name}: east {east}/9, on the doorway slabs {slab}/9, best progress "
          f"max {max(best):.2f}% median {med(best):.2f}%; 1.4%: {len(h14)}/9 h {med(h14):5.0f} at "
          f"{med(t14):5.2f} s; 2.5%: {len(h25)}/9 h {med(h25):5.0f} at {med(t25):5.2f} s")
