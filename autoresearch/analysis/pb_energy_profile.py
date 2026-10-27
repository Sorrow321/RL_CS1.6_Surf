"""Energy profile of greedy evals at first arrival per progress level (pbr7b2, 2026-10-27).

    python autoresearch/analysis/pb_energy_profile.py LEVELS file.jsonl [more.jsonl ...]
    python autoresearch/analysis/pb_energy_profile.py 66,70,71.2,72.5,72.9,73.5 runs/<agent>/<run>/traj_<step>.jsonl

For every episode (the first config.score_episodes of an eval) and every progress level (% of the
start's d on the grader's field), at the first tick the episode's best progress reaches the level:
the time, the energy height h = z + |v|^2 / 1600, the speed |v| and the position. Comparing these
rows between an eval and a reference line is how round 7 found where a gate's energy is lost
(gate 20's walkway at 65.9%, the 70.5-72.5% descent before the 74.4% block). A trajectory whose
name contains 'pb_wr' (the parsed world-record demo) is read whole, as a YARDSTICK only: CLAUDE.md
section 0 - it never enters training, a spawn, a reward or a schedule."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import score  # noqa: E402

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
levels = [float(x) for x in sys.argv[1].split(",")]
for fn in sys.argv[2:]:
    eps = score.episodes(fn)
    if "pb_wr" not in fn:
        eps = score.first_episodes(eps, score.SCORE_EPISODES)
    for h, r, tr in eps:
        p, v = r[:, 1:4], r[:, 4:7]
        d = score.field().sample(p).astype(np.float64)
        tick = float(h.get("tick_ms", 10.0)) / 1000.0
        ok = d < score.field().reach_max - 0.5 * score.field().cell
        prog = np.where(ok, 100.0 * (score.START_D - d) / score.START_D, -1e9)
        best = np.maximum.accumulate(prog)
        out = []
        for lev in levels:
            if not (best >= lev).any():
                out.append(f"{lev}:-")
                continue
            i = int(np.argmax(best >= lev))
            sp = float(np.linalg.norm(v[i]))
            out.append(f"{lev}:t{i * tick:.1f} h{p[i, 2] + sp * sp / 1600.0:.0f} v{sp:.0f} "
                       f"({p[i, 0]:.0f},{p[i, 1]:.0f},{p[i, 2]:.0f})")
        print(f"{Path(fn).name[:28]} ep{h.get('episode')} best {best.max():.2f} | " + " | ".join(out))
