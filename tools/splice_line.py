#!/usr/bin/env python3
"""splice_line.py - one per-tick line = a head line up to a state + a tail line from it (pbr2b1).

A Go-Explore search rooted at states OF a line (explore_phase1 --roots-spine) replays, through
tools/ge_replay.py, into a tail whose first state is one of those roots. This finds that root in
the head (origin within --tol u, the first match) and writes head[:k] + tail as one STATE_DTYPE
line for train_fast --demo-file: the head's ticks up to the root, then the tail's exact states.
Nothing is read off the map; both inputs are the run's own search lines.

    python tools/splice_line.py --head runs/a/line_a.npy --tail runs/b/tail.npy --out runs/b/line_b.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from surfgym.core import STATE_DTYPE  # noqa: E402


def splice(head: np.ndarray, tail: np.ndarray, tol: float = 0.5):
    head = np.asarray(head, STATE_DTYPE)
    tail = np.asarray(tail, STATE_DTYPE)
    if not len(tail):
        raise ValueError("empty tail")
    d = np.linalg.norm(head["origin"].astype(np.float64)
                       - tail["origin"][0].astype(np.float64), axis=1)
    hit = np.flatnonzero(d <= tol)
    if not hit.size:
        raise ValueError(f"the tail's first state is not on the head (closest {d.min():.2f} u)")
    k = int(hit[0])
    out = np.concatenate([head[:k], tail]).astype(STATE_DTYPE)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    return out, k


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", required=True)
    ap.add_argument("--tail", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args(argv)
    out, k = splice(np.load(a.head), np.load(a.tail), a.tol)
    np.save(a.out, out)
    print(f"head[:{k}] + tail ({len(np.load(a.tail))} states) = {len(out)} states -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
