#!/usr/bin/env python3
"""uf2_splits.py - MEASUREMENT ONLY (CLAUDE.md sections 0 / 0b): split times by field progress.

For every episode of each trajectory file: the first time its running-best progress on the
grader's field (config.field) reaches each level, and the energy height E = z + |v|^2/1600 there.
Times count from the episode's first row (the spawn), like the grader's finish time; for the WR
demo pass --curtain to count from the start curtain (y <= -2,400) instead, as its 35.17 s does.
The record is a YARDSTICK for reading results, never a constant in training.

    python autoresearch/analysis/uf2_splits.py [--median] [--curtain] traj.jsonl [...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "autoresearch"))
import score as S  # noqa: E402

LEVELS = [1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50, 57, 60, 65, 70, 75, 80, 85, 90, 95, 98, 99, 99.5]


def splits(h, r, tr, f, curtain=False):
    tick = float(h.get("tick_ms", 10.0)) / 1000.0
    p, v = r[:, 1:4], r[:, 4:7]
    d = f.sample(p).astype(np.float64)
    d0 = S.START_D
    ok = d < f.reach_max - 0.5 * f.cell
    d = np.where(ok, d, np.inf)
    prog = 100.0 * (d0 - np.minimum.accumulate(d)) / d0
    E = p[:, 2] + (v ** 2).sum(1) / 1600.0
    t0 = 0.0
    if curtain:
        c = np.flatnonzero(p[:, 1] <= -2400.0)
        t0 = c[0] * tick if len(c) else 0.0
    out = {}
    for L in LEVELS:
        i = np.flatnonzero(prog >= L)
        out[L] = (i[0] * tick - t0, E[i[0]], np.linalg.norm(v[i[0]])) if len(i) else None
    fin = (len(r) * tick - t0) if tr.get("end") == "done" else None
    return out, fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--median", action="store_true", help="one line per file: the median over episodes")
    ap.add_argument("--curtain", action="store_true", help="time from the start curtain (the WR demo)")
    ap.add_argument("--eps", type=int, default=S.SCORE_EPISODES)
    a = ap.parse_args()
    f = S.field()
    hdr = "level%  " + " ".join(f"{L:>6g}" for L in LEVELS) + "    fin"
    for path in a.files:
        eps = S.first_episodes(S.episodes(Path(path)), a.eps)
        rows = [splits(h, r, tr, f, a.curtain) for h, r, tr in eps]
        print(f"{Path(path).name}: {len(eps)} episodes")
        print(hdr)
        if a.median:
            med = []
            for L in LEVELS:
                ts = [o[L][0] for o, _ in rows if o[L] is not None]
                med.append(f"{np.median(ts):6.2f}" if len(ts) > len(rows) // 2 else "     -")
            fins = [fn for _, fn in rows if fn is not None]
            print("t med   " + " ".join(med) + (f" {np.median(fins):6.2f}" if fins else "      -"))
            Em = []
            for L in LEVELS:
                es = [o[L][1] for o, _ in rows if o[L] is not None]
                Em.append(f"{np.median(es):6.0f}" if len(es) > len(rows) // 2 else "     -")
            print("E med   " + " ".join(Em))
            continue
        for k, (o, fn) in enumerate(rows):
            print(f"t ep{k:<3d} " + " ".join(f"{o[L][0]:6.2f}" if o[L] else "     -" for L in LEVELS)
                  + (f" {fn:6.2f}" if fn else "      -"))
        for k, (o, fn) in enumerate(rows[:1]):
            print(f"E ep{k:<3d} " + " ".join(f"{o[L][1]:6.0f}" if o[L] else "     -" for L in LEVELS))


if __name__ == "__main__":
    main()
