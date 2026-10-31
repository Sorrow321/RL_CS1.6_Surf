#!/usr/bin/env python3
"""zl_eval_ends.py - surf_za_loop: what each greedy eval episode does, on the grader's field.

Per episode: the end (fail = the bottom teleport *21 or a kill, trunc = the cap, done = a finish
slab), its length, the best progress and when, the energy height E = z + |v|^2 / 1600 (start,
max, end; the record gains ~1,000 u on its first loop with NEGATIVE field progress), the share
of the episode spent gaining energy (E at a running max), z range, speed, and whether the
episode was cut by the trainer's eval patience (the trailer's flag, when written). Reporting
only - nothing here reaches training.

    python autoresearch/analysis/zl_eval_ends.py runs/<agent>/<run>/traj_<step>.jsonl [...]
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
        sc = S.score_file(Path(path))
        print(f"{Path(path).name}: {len(eps)} episodes  score {sc.get('score')}")
        for h, r, tr in eps:
            pat = h.get("tick_pattern_ms")
            tick = (float(np.mean(pat)) if pat else float(h.get("tick_ms", 10.0))) / 1000.0
            p, v = r[:, 1:4], r[:, 4:7]
            d = f.sample(p).astype(np.float64)
            d0 = d[0]
            ok = d < f.reach_max - 0.5 * f.cell
            prog = np.where(ok, 100.0 * (d0 - d) / d0, -np.inf)
            ib = int(np.argmax(prog))
            E = p[:, 2] + (v ** 2).sum(1) / 1600.0
            sp = np.linalg.norm(v, axis=1)
            emax_run = np.maximum.accumulate(E)
            gain_share = float(np.mean(E >= emax_run - 1e-3))
            extra = " CUT" if tr.get("patience") or tr.get("cut") else ""
            print(f"  ep{h.get('episode')}: {tr.get('end'):5s}{extra} {len(r)*tick:6.1f}s "
                  f"best {prog[ib]:5.2f}% @{ib*tick:5.1f}s z {p[ib,2]:5.0f} | "
                  f"E {E[0]:6.0f} max {E.max():6.0f} @{int(np.argmax(E))*tick:5.1f}s end {E[-1]:6.0f} "
                  f"| z {p[:,2].min():5.0f}..{p[:,2].max():5.0f} | v max {sp.max():5.0f} "
                  f"| end ({p[-1,0]:6.0f},{p[-1,1]:6.0f},{p[-1,2]:6.0f})")


if __name__ == "__main__":
    main()
