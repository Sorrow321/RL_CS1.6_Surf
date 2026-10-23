"""Per-episode summary of greedy evals on surf_princessburglar_b26 (pbr1b1, 2026-10-23).

    python autoresearch/analysis/pb_evals.py runs/<agent>/<run>/traj_<step>.jsonl [...]

For each episode: length, end kind, best progress % on the grader's field, where the best
d was reached, the furthest x/z, max speed and max energy height h = z + |v|^2/1600 (the
start gate is an energy gate: the way on is ~70-110 u above the spawn floor)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

for fn in _args:
    eps = score.first_episodes(score.episodes(fn), score.SCORE_EPISODES)
    print(Path(fn).name, score.score_file(Path(fn)).get("score"))
    for h, r, tr in eps:
        p, v = r[:, 1:4], r[:, 4:7]
        d = score.field().sample(p).astype(np.float64)
        ok = d < score.field().reach_max - 16
        i = int(np.argmin(np.where(ok, d, 1e9)))
        sp = np.linalg.norm(v, axis=1)
        en = p[:, 2] + (v * v).sum(axis=1) / 1600.0
        tick = float(h.get("tick_ms", 10.0)) / 1000.0
        print(f"  ep{h.get('episode')} {len(r) * tick:6.1f}s {tr.get('end'):5s} "
              f"best {100 * (d[0] - d[i]) / d[0]:5.2f}% at {p[i].round(0)} t={i * tick:5.1f}s  "
              f"end {p[-1].round(0)}  max|v| {sp.max():5.0f}  max h {en.max():6.0f} "
              f"(spawn h {en[0]:5.0f})  max x {p[:, 0].max():6.0f}")
