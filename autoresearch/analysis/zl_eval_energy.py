"""Per-episode read-out of greedy evals on surf_za_loop (zlr1b2, 2026-10-31). ANALYSIS ONLY.

For every scored episode of each traj_*.jsonl given: seconds, how it ended, best progress on
config.field (the grader's), the energy height h = z + |v|^2 / (2 g) at the start, its maximum,
the max speed, the z range, and how many times vx changed sign at > 300 u/s (half-pipe swings
along x). Nothing here feeds training.

    python autoresearch/analysis/zl_eval_energy.py runs/<agent>/<run>/traj_*.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "autoresearch"))

import score  # noqa: E402

G = 800.0


def read(path: Path) -> None:
    eps = score.first_episodes(score.episodes(path), int(score.CFG.get("score_episodes", 9)))
    f = score.field()
    print(f"{path.name}: {len(eps)} episodes")
    print("  ep    secs  end    prog%   h0      hmax    vmax   zmin   zmax   swings")
    for hdr, rows, trl in eps:
        if not len(rows):
            continue
        tick = float(hdr.get("tick_ms") or 10.0) / 1000.0
        pos, vel = rows[:, 1:4], rows[:, 4:7]
        d = np.asarray(f.sample(pos), np.float64)
        d0 = float(d[0])
        prog = 100.0 * (d0 - float(np.min(d))) / max(d0, 1.0)
        h = pos[:, 2] + (vel * vel).sum(axis=1) / (2.0 * G)
        spd = np.linalg.norm(vel, axis=1)
        vx = vel[:, 0]
        fast = np.abs(vx) > 300.0
        sx = np.sign(vx[fast])
        swings = int(np.count_nonzero(sx[1:] != sx[:-1])) if len(sx) > 1 else 0
        print(f"  {hdr.get('episode', '?'):>2}  {len(rows) * tick:6.1f}  {trl.get('end', '?'):5}"
              f"  {prog:6.2f}  {h[0]:7.0f} {h.max():7.0f}  {spd.max():5.0f}"
              f"  {pos[:, 2].min():6.0f} {pos[:, 2].max():6.0f}  {swings:4d}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        read(Path(a))
