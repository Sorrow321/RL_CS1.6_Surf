#!/usr/bin/env python3
"""ge_frontier_roots.py - the next Go-Explore stage's roots: the K deepest cells of an archive on
the goal field (zlr2b1, 2026-11-01). Standard Go-Explore iteration - continue the search from its
own frontier - for when a stage's frontier is ballistic flight that the stage itself ran out of
time to explore from. Writes the roots (STATE_DTYPE .npy, explore_phase1 --roots-spine) and their
archive indices (.json, for ge_replay --cell when splicing the next stage's chain back).

    python tools/ge_frontier_roots.py runs/ge/archive.npz --field maps/X.goalc_32.npz \\
        --k 2000 --out runs/ge/frontier_roots.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def main(argv=None):
    from surfgym.goalfield import load_goal_field
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("--field", required=True)
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    st = z["state"]
    f = load_goal_field(a.field)
    d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    idx = np.argsort(d)[:a.k]
    idx = idx[np.isfinite(d[idx])]
    roots = np.array(st[idx], copy=True)
    roots["tick"] = 0
    roots["stuck_ticks"] = 0
    np.save(a.out, roots)
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"archive": str(a.archive), "cells": [int(i) for i in idx],
         "d": [float(d[i]) for i in idx]}))
    o = roots["origin"]
    print(f"{len(roots)} roots -> {a.out}: d {d[idx[0]]:,.0f}..{d[idx[-1]]:,.0f}, "
          f"x {o[:, 0].min():.0f}..{o[:, 0].max():.0f}, y {o[:, 1].min():.0f}..{o[:, 1].max():.0f}, "
          f"z {o[:, 2].min():.0f}..{o[:, 2].max():.0f}")


if __name__ == "__main__":
    main()
