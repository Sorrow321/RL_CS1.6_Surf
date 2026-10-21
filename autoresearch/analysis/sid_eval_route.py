#!/usr/bin/env python3
"""sid_eval_route.py - MEASUREMENT ONLY: which stage of sidistic's CP2 / cavity route each
greedy eval episode reached. Map coordinates are bench metrics (CLAUDE.md 0b), never a reward.

Flags per episode (first tick each was seen, in seconds):
  (all after t = 22 s: the map folds and the same x is flown earlier at other heights)
  E   east under plate *45 after the CP2 U-turn (x > 4,194, 10,700 < z < 11,880): the detour
  S   in the slot past the pit floor's west edge (2,400 < x < 2,592, 11,332 < z < 11,600)
  W   past the thin wall into the shaft (2,208 < x < 2,512, 10,700 < z < 11,520)
  C   in the cavity under the pit slab after W (2,576 < x < 3,040, 9,500 < z < 10,760, |y| < 700)
  R   rising in the cavity (C and vz > 300)
  T   in a tunnel (1,952 < x < 2,208, |y| > 800, 10,900 < z < 11,232)
  M   in the room past the tunnels (1,168 < x < 1,952, 10,400 < z < 11,300)

    python autoresearch/analysis/sid_eval_route.py runs/x/traj_*.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "autoresearch"))
import score  # noqa: E402


def flags(r):
    t = r[:, 0] / 100.0                     # rows: [t, x, y, z, vx, vy, vz, yaw, ...], 10 ms ticks
    x, y, z = r[:, 1], r[:, 2], r[:, 3]
    vz = r[:, 6]
    out = {}

    def first(name, m):
        i = np.flatnonzero(m)
        if len(i):
            out[name] = round(float(t[i[0]]), 2)
        return i

    late = t > 22.0                         # the map folds: the same x is flown earlier at other z
    first("E", late & (x > 4194) & (z > 10700) & (z < 11880))
    first("S", late & (x > 2400) & (x < 2592) & (z > 11332) & (z < 11600))
    w = first("W", late & (x > 2208) & (x < 2512) & (z > 10700) & (z < 11520))
    if len(w):
        cav = (np.arange(len(r)) > w[0]) & (x > 2576) & (x < 3040) & (z > 9500) & (z < 10760) \
            & (np.abs(y) < 700)
        if len(first("C", cav)):
            first("R", cav & (vz > 300))
    first("T", late & (x > 1952) & (x < 2208) & (np.abs(y) > 800) & (z > 10900) & (z < 11232))
    first("M", late & (x > 1168) & (x < 1952) & (z > 10400) & (z < 11300))
    return out


def main(paths):
    f = score.field()
    for path in paths:
        eps = score.episodes(path)
        s = score.score_file(Path(path))
        line = []
        for hdr, r, tr in eps:
            d = f.sample(r[:, 1:4]).astype(np.float64)
            ok = d < f.reach_max - 0.5 * f.cell
            dmin = float(d[ok].min()) if ok.any() else float(d[0])
            prog = 100.0 * (d[0] - dmin) / d[0]
            fl = flags(r)
            tag = "".join(k for k in "ESWCRTM" if k in fl) or "-"
            line.append(f"{prog:5.2f}%/{len(r) / 100.0:4.1f}s/{tag}")
        print(f"{Path(path).name}: score {s.get('score')} med {s.get('median_progress_pct')} | "
              + "  ".join(line))


if __name__ == "__main__":
    main(sys.argv[1:])
