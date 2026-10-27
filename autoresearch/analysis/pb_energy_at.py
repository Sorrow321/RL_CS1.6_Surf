"""Energy height h = z + |v|^2/1600 of each greedy episode at its FIRST arrival at given progress
levels on the grader's field (pbr7b4, 2026-10-27; ANALYSIS of the policy's own evals). Where the
greedy loses the energy a gate needs shows as a drop between two levels.

    python autoresearch/analysis/pb_energy_at.py runs/<agent>/<run>/traj_<step>.jsonl 70.5,71.2,71.8,73.44
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

F = score.field()
levels = [float(x) for x in _args[1].split(",")]
eps = score.first_episodes(score.episodes(_args[0]), score.SCORE_EPISODES)
print(Path(_args[0]).name, "levels", levels)
for h, r, tr in eps:
    p, v = r[:, 1:4], r[:, 4:7]
    d = F.sample(p).astype(np.float64)
    # 57,149 u = config.start_d, the grader field's distance at the spawn
    prog = np.maximum.accumulate(np.where(d < F.reach_max - 16, 100 * (57149 - d) / 57149, -1))
    hh = p[:, 2] + (v * v).sum(1) / 1600
    row = []
    for L in levels:
        i = np.flatnonzero(prog >= L)
        row.append(f"{hh[i[0]]:5.0f}" if len(i) else "    -")
    print(f"ep{h.get('episode')} max {prog.max():5.2f}% " + " ".join(row))
