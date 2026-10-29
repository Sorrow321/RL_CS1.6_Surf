"""pb_stair_eval.py - where a greedy eval's episodes go at the start (pbfr1b4, 2026-10-29).

    python autoresearch/analysis/pb_stair_eval.py runs/<agent>/<run>/traj_*.jsonl [...]

ANALYSIS ONLY (CLAUDE.md 0b: map boxes are measurement instruments, never a reward, spawn rule or
schedule). Per eval (the first 9 episodes, the grader's): for each episode the first time it is
in the doorway basin (func_conveyor *249: x 648..712, y -3043..-2957, feet at its top 3657), on a
*250 strip (feet at 3681), in the crouch tunnel west of the doorway (x < 600, z > 3700), past
the tunnel's west end (progress >= 1.4%), and its max progress on the grader's field and end.
Feet = origin z - 18 when ducked (column 8 buttons bit 4 is the duck KEY, not the hull, so the
hull is inferred from the height: a standing hull on *249 has origin 3693, a ducked one 3675).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402

F, D0 = score.field(), score.START_D


def first(mask, dt):
    i = np.flatnonzero(mask)
    return f"{i[0] * dt:5.1f}" if len(i) else "    -"


for fn in sys.argv[1:]:
    eps = score.first_episodes(score.episodes(fn), score.SCORE_EPISODES)
    print(f"{Path(fn).name}: {len(eps)} episodes")
    print("  ep  basin  strip tunnel  >1.4%   maxprog  end    secs")
    for h, r, tr in eps:
        dt = float(h.get("tick_ms", 10.0)) / 1000.0
        p = r[:, 1:4]
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d = F.sample(p).astype(np.float64)
        ok = d < F.reach_max - 0.5 * F.cell
        prog = np.where(ok, 100.0 * (D0 - d) / D0, -np.inf)
        door = (x > 632) & (x < 728) & (y > -3059) & (y < -2941)
        basin = door & (y > -3043) & (y < -2957) & ((np.abs(z - 3675) < 3) | (np.abs(z - 3693) < 3))
        strip = door & ((np.abs(z - 3699) < 3) | (np.abs(z - 3717) < 3)) & ((y <= -3027) | (y >= -2973))
        tunnel = (x < 600) & (z > 3700) & (y > -2930) & (y < -2840)
        past = prog >= 1.4
        mp = float(np.max(prog[np.isfinite(prog)])) if np.isfinite(prog).any() else float("nan")
        print(f"  {h.get('episode', '?'):>2} {first(basin, dt)}  {first(strip, dt)}  {first(tunnel, dt)}  "
              f"{first(past, dt)}   {mp:6.2f}%  {str(tr.get('end')):5s} {len(r) * dt:6.1f}")
