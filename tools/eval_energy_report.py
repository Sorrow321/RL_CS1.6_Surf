#!/usr/bin/env python3
"""eval_energy_report.py - per-episode read-out of greedy evals (traj_<step>.jsonl) on the grader's
field and in ENERGY HEIGHT h = z + |v|^2 / (2 g): what a loop map's score cannot show.

For each episode: how it ended and when, the best field progress (the grader's number) and the
time it was reached, the energy height at the start / its max / at the end, the max speed, and
the share of time with h within 32 u of its running record ("at its record" = still pumping or
climbing). Map-agnostic: the field and d0 come from autoresearch/config.json.

    python tools/eval_energy_report.py runs/x/traj_0075497472.jsonl [more ...]
    python tools/eval_energy_report.py runs/x/              # every greedy eval in the run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch"))
sys.path.insert(0, str(ROOT / "python"))

import score  # noqa: E402  (the grader: its field, its episode reader)


def report(path: Path, g: float = 800.0) -> list[str]:
    f = score.field()
    out = []
    eps = score.first_episodes(score.episodes(path), score.SCORE_EPISODES)
    s = score.score_file(path)
    out.append(f"{path.name}: score {s.get('score')}  best {s.get('best_progress_pct')}%  "
               f"median {s.get('median_progress_pct')}%  fin {s.get('finishes')}/{s.get('episodes')}")
    for hdr, r, tr in eps:
        if not len(r):
            continue
        tick_s = float(hdr.get("tick_ms", 10.0)) / 1000.0
        p, v = r[:, 1:4], r[:, 4:7]
        d = f.sample(p).astype(np.float64)
        ok = d < f.reach_max - 0.5 * f.cell
        d = np.where(ok, d, np.inf)
        d0 = d[0]
        prog = 100.0 * (d0 - np.minimum.accumulate(d)) / d0
        h = p[:, 2] + (v * v).sum(1) / (2.0 * g)
        hrec = np.maximum.accumulate(h)
        spd = np.linalg.norm(v, axis=1)
        ib = int(np.argmax(prog))
        near = float(np.mean(h >= hrec - 32.0))
        out.append(f"  ep {hdr.get('episode', '?'):>2} {tr.get('end', '?'):>5} {len(r) * tick_s:6.1f}s"
                   f"  best {prog[ib]:5.1f}% @ {ib * tick_s:6.1f}s  h {h[0]:7.0f} max {h.max():7.0f}"
                   f" end {h[-1]:7.0f}  spd max {spd.max():5.0f}  z end {p[-1, 2]:6.0f}"
                   f"  at-record {near:4.0%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    files = []
    for p in map(Path, a.paths):
        files += (sorted(q for q in p.glob("traj_*.jsonl")
                         if re.fullmatch(r"traj_\d+\.jsonl", q.name)) if p.is_dir() else [p])
    for q in files:
        print("\n".join(report(q)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
