#!/usr/bin/env python3
"""uf2_splits.py - per greedy eval: the grader's score, the finishes, and the median time of first
arrival at a few progress levels on the grader's field (where a faster eval gains its seconds).
Reporting only (uf2r2b3, 2026-10-31).

    python autoresearch/analysis/uf2_splits.py runs/<agent>/<run>/traj_*.jsonl
    SPLITS=2,5,30,60,70,90 python autoresearch/analysis/uf2_splits.py ...
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "autoresearch"))
import score as S  # noqa: E402

LEVELS = [float(x) for x in os.environ.get("SPLITS", "2,5,30,60,70,90").split(",")]


def main():
    f = S.field()
    print("file                      score   fin  best_s  med_s | " +
          " ".join(f"{lv:>5g}%" for lv in LEVELS))
    for path in sys.argv[1:]:
        p = Path(path)
        eps = S.first_episodes(S.episodes(p), S.SCORE_EPISODES)
        if not eps:
            continue
        sc = S.score_file(p) if hasattr(S, "score_file") else None
        fins, arr = [], {lv: [] for lv in LEVELS}
        for h, r, tr in eps:
            tick = float(h.get("tick_ms", 10.0)) / 1000.0
            pos = r[:, 1:4]
            d = f.sample(pos).astype(np.float64)
            ok = d < f.reach_max - 0.5 * f.cell
            prog = np.where(ok, 100.0 * (S.START_D - d) / S.START_D, -1e9)
            best = np.maximum.accumulate(prog)
            for lv in LEVELS:
                hit = np.flatnonzero(best >= lv)
                if len(hit):
                    arr[lv].append(hit[0] * tick)
            if tr.get("end") == "done":
                fins.append(len(r) * tick)
        med = {lv: (np.median(v) if v else float("nan")) for lv, v in arr.items()}
        s = sc.get("score") if isinstance(sc, dict) else float("nan")
        print(f"{p.name:24s} {s:7.3f} {len(fins)}/{len(eps)} "
              f"{(min(fins) if fins else float('nan')):7.2f} "
              f"{(np.median(fins) if fins else float('nan')):6.2f} | "
              + " ".join(f"{med[lv]:6.2f}" for lv in LEVELS))


if __name__ == "__main__":
    main()
