"""pbf_start_route.py - HOW each greedy episode leaves the start room (pbfr1b1, 2026-10-29).

    python autoresearch/analysis/pbf_start_route.py runs/<agent>/<run>/traj_*.jsonl

MEASUREMENT ONLY (CLAUDE.md 0 / 0b): the boxes are this map's coordinates, used to label routes in
an eval, never in a reward, a spawn rule or a schedule. Per episode (first 9): did it go onto the
ridge east of the arch (x > 1,200 before its first tunnel tick), did it touch the doorway's conveyor
recess (*249 / *250 under the push fields: x 646-720, y -3,056..-2,944, origin z < 3,720 - the
SUSPECT climb), did it reach the tunnel (x < 600, origin z > 3,690, y -2,920..-2,860), the energy
height h = z + |v|^2/1600 at its tunnel entry, and its best progress on the grader field."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

F, D0 = score.field(), score.START_D
for fn in _args:
    rows = []
    for hdr, r, tr in score.first_episodes(score.episodes(fn), score.SCORE_EPISODES):
        p, v = r[:, 1:4], r[:, 4:7]
        tun = np.flatnonzero((p[:, 0] < 600) & (p[:, 2] > 3690) & (p[:, 1] > -2920) & (p[:, 1] < -2860))
        t_in = int(tun[0]) if len(tun) else len(p)
        east = bool((p[:t_in, 0] > 1200).any())
        door = bool(((p[:t_in, 0] > 646) & (p[:t_in, 0] < 720) & (p[:t_in, 1] > -3056)
                     & (p[:t_in, 1] < -2944) & (p[:t_in, 2] < 3720) & (p[:t_in, 2] > 3650)).any())
        d = F.sample(p).astype(np.float64)
        best = 100.0 * (D0 - d.min()) / D0
        h_in = float(p[t_in, 2] + (v[t_in] ** 2).sum() / 1600.0) if len(tun) else float("nan")
        rows.append((east, door, bool(len(tun)), h_in, best))
    n = len(rows)
    tag = lambda x: ("ridge" if x[0] else "-----") + ("+DOOR" if x[1] else "     ")
    print(f"{Path(fn).name}: {sum(x[2] for x in rows)}/{n} reach the tunnel; ridge {sum(x[0] for x in rows)}, "
          f"door-recess {sum(x[1] for x in rows)}; per episode: "
          + " ".join(f"[{tag(x)} {'T' if x[2] else '-'} h{x[3]:.0f} {x[4]:.1f}%]" for x in rows))
