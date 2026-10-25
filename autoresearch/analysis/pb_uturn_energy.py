"""Energy height at the gate-8 corridor U-turn exit, per greedy eval (pbr3b4, 2026-10-25).

    python autoresearch/analysis/pb_uturn_energy.py runs/<agent>/<run>/traj_*.jsonl

For each eval: the median over its first 9 episodes of h = z + |v|^2/1600 at the episode's
nearest approach to (625, -3,271, 2,740) - the U-turn exit after the corridor ramp, where the
x1.12 Go-Explore root of spine pbr3b4_auto_k03.npy sits (h 3,607; the unboosted beam line had
3,430; the record ~3,570-3,610, ANALYSIS ONLY). A MEASUREMENT point for the energy the gate-10
ramp climb needs; never a reward, spawn or schedule quantity (CLAUDE.md 0b)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import score  # noqa: E402

TGT = np.array([625.0, -3271.0, 2740.0])
for fn in _args:
    hs, ok = [], 0
    for h, r, tr in score.first_episodes(score.episodes(fn), score.SCORE_EPISODES):
        p, v = r[:, 1:4], r[:, 4:7]
        dist = np.linalg.norm(p - TGT, axis=1)
        j = int(np.argmin(dist))
        if dist[j] < 80.0:
            ok += 1
            hs.append(p[j, 2] + (v[j] ** 2).sum() / 1600.0)
    print(f"{Path(fn).name}: {ok} episodes pass the U-turn exit; median h "
          f"{np.median(hs) if hs else float('nan'):6.0f}  max {max(hs) if hs else float('nan'):6.0f}")
