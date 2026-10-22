"""r5_evalsum.py - one line per greedy eval of a run: the grader's score plus where each
episode stopped (ANALYSIS ONLY; it reads the trainer's own traj_<step>.jsonl evals).

    python autoresearch/analysis/r5_evalsum.py runs/<run> [--last N] [--eps]

Per eval: step, score (autoresearch/score.py), best / median progress %, CP, finishes, and the
per-episode best progress sorted (so a line that splits between gates is visible at a glance).
With --eps, the best episode's stop: time of its best tick, position and speed there.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "autoresearch"))
import score  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--last", type=int, default=0, help="only the newest N evals")
    ap.add_argument("--eps", action="store_true", help="the best episode's stop")
    a = ap.parse_args(argv)
    files = sorted(Path(a.run).glob("traj_*.jsonl"))
    files = [x for x in files if re.fullmatch(r"traj_\d+\.jsonl", x.name)]
    if a.last:
        files = files[-a.last:]
    f = score.field()
    for x in files:
        s = score.score_file(x)
        step = int(x.stem.split("_")[1])
        if s["score"] is None:
            print(f"{step:>14,d}  REJECTED {s.get('error')}")
            continue
        eps = score.episodes(x)
        per = []
        stop = None
        for hdr, r, tr in eps:
            p = r[:, 1:4]
            d = f.sample(p).astype(np.float64)
            ok = d < f.reach_max - 0.5 * f.cell
            dd = np.where(ok, d, np.inf)
            k = int(np.argmin(dd))
            pr = 100.0 if tr.get("end") == "done" else max(0.0, 100.0 * (d[0] - dd[k]) / d[0])
            per.append(pr)
            if stop is None or pr > stop[0]:
                v = r[k, 4:7]
                stop = (pr, k * float(hdr.get("tick_ms", 10.0)) / 1000.0, p[k], float(np.hypot(v[0], v[1])),
                        float(v[2]), tr.get("end"), len(r) * float(hdr.get("tick_ms", 10.0)) / 1000.0)
        per_s = " ".join(f"{v:5.1f}" for v in sorted(per, reverse=True))
        line = (f"{step:>14,d}  score {s['score']:7.3f}  best {s['best_progress_pct']:6.2f}  "
                f"med {s['median_progress_pct']:6.2f}  CP{s['max_cp']}  fin {s['finishes']}/{s['episodes']}  "
                f"[{per_s}]")
        if a.eps and stop is not None:
            pr, t, pos, vh, vz, end, tl = stop
            line += (f"  | best ep: {pr:.2f}% at {t:.1f}s ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) "
                     f"|vh| {vh:.0f} vz {vz:.0f} end {end} at {tl:.1f}s")
        print(line, flush=True)


if __name__ == "__main__":
    main()
