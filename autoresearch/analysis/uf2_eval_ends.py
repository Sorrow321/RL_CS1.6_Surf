#!/usr/bin/env python3
"""uf2_eval_ends.py - where each greedy eval episode goes on the grader's field: best progress, the
position and time of its best, where it ends, the energy height E = z + |v|^2/1600 along it, and
the lowest z reached (did it enter the start pit). Reporting only.

    python autoresearch/analysis/uf2_eval_ends.py runs/<agent>/<run>/traj_<step>.jsonl [...]
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "autoresearch"))
import score as S  # noqa: E402


def main():
    f = S.field()
    for path in sys.argv[1:]:
        eps = S.first_episodes(S.episodes(Path(path)), S.SCORE_EPISODES)
        print(f"{Path(path).name}: {len(eps)} episodes")
        for h, r, tr in eps:
            tick = float(h.get("tick_ms", 10.0)) / 1000.0
            p, v = r[:, 1:4], r[:, 4:7]
            d = f.sample(p).astype(np.float64)
            d0 = d[0]
            prog = 100.0 * (d0 - d) / d0
            ib = int(np.argmax(prog))
            E = p[:, 2] + (v ** 2).sum(1) / 1600.0
            zmin = int(np.argmin(p[:, 2]))
            sp = np.linalg.norm(v, axis=1)
            print(f"  ep{h.get('episode')}: {tr.get('end'):5s} {len(r)*tick:6.2f}s best {prog[ib]:5.2f}% @ {ib*tick:5.2f}s "
                  f"({p[ib,0]:6.0f},{p[ib,1]:6.0f},{p[ib,2]:5.0f}) | end ({p[-1,0]:6.0f},{p[-1,1]:6.0f},{p[-1,2]:5.0f}) "
                  f"|v| {sp[-1]:5.0f} E {E[-1]:5.0f} | zmin {p[zmin,2]:5.0f} @ {zmin*tick:5.2f}s | Emax {E.max():5.0f} vmax {sp.max():5.0f}")


if __name__ == "__main__":
    main()
