"""Energy height at first arrival at each progress level, per greedy eval (pbr4b4, 2026-10-25).

    python autoresearch/analysis/pb_energy_profile.py runs/<agent>/<run>/traj_*.jsonl

For each eval: the median and max over its first 9 episodes of h = z + |v|^2/1600 at the first
tick each episode reaches 20 / 21 / 22 / 24 / 25.5 / 26.5 % on the grader field (and how many
episodes got there). The levels bracket the energy leaks measured in NOTES.md (gate 8's corridor
landing between 20 and 21 %, the gate-9 wall clip by 22 %, the ramp-foot landing by 24 %, the
upper-floor contact by 25.5 %, the north gap at ~27 %) - a MEASUREMENT of what
--respawn-greedy-impacts is meant to raise, never a reward, spawn or schedule quantity
(CLAUDE.md 0b). Analysis only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

LEVELS = (20.0, 21.0, 22.0, 24.0, 25.5, 26.5)
f = score.field()
print("eval".ljust(26) + "".join(f"{L:>14.1f}%" for L in LEVELS))
for fn in _args:
    cols = {L: [] for L in LEVELS}
    for h, r, tr in score.first_episodes(score.episodes(fn), score.SCORE_EPISODES):
        p, v = r[:, 1:4], r[:, 4:7]
        d = f.sample(p).astype(np.float64)
        pr = 100.0 * (d[0] - d) / d[0]
        hh = p[:, 2] + (v ** 2).sum(1) / 1600.0
        for L in LEVELS:
            i = np.flatnonzero(pr >= L)
            if len(i):
                cols[L].append(hh[i[0]])
    print(Path(fn).name.ljust(26) + "".join(
        f"  {len(cols[L])}:{np.median(cols[L]):5.0f}/{max(cols[L]):5.0f}" if cols[L]
        else f"{'-':>15}" for L in LEVELS))
