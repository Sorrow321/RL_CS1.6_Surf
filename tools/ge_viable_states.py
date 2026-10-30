#!/usr/bin/env python3
"""ge_viable_states.py - a Go-Explore archive (tools/explore_phase1.py --exact-chains) as a training
spawn pool: every cell whose search lineage SURVIVED at least --min-secs after it (uf2r1b1,
2026-10-30).

An --exact-chains archive keeps each cell's parent and depth (physics ticks from the spawn along
its chain). A cell's subtree depth (the deepest descendant) minus its own depth is how long the
search ever kept flying from that state: a state from which every burst died at once (a free fall
onto a kill plane) has ~0, a state on a line that went on has more. The pool is the cells with
subtree survival >= --min-secs, as STATE_DTYPE rows with clocks zeroed (train_fast --spawn-states;
the archive is the run's own reward-free search from the map's true spawn - no demo, no record, no
map constant). Nothing here reads a field, a region or a route.

    python tools/ge_viable_states.py runs/ge/uf2a/archive.npz --out runs/ge/uf2a_viable.npy \\
        --min-secs 1.0 --tick-ms 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def subtree_survival(parent: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """ticks from each cell to its deepest descendant (0 for a leaf). Cells are appended after
    their parent, so one reverse sweep propagates every subtree's maximum depth up."""
    n = len(parent)
    deep = depth.astype(np.int64).copy()
    par = parent.astype(np.int64)
    if n and np.any(par[np.arange(n)] >= np.arange(n)):
        bad = int(np.sum(par >= np.arange(n)))
        raise SystemExit(f"{bad} cells have a parent index >= their own: not an append-order archive")
    for i in range(n - 1, -1, -1):
        p = par[i]
        if p >= 0 and deep[i] > deep[p]:
            deep[p] = deep[i]
    return deep - depth.astype(np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-secs", type=float, default=1.0)
    ap.add_argument("--tick-ms", type=float, default=10.0, help="the archive's physics tick")
    a = ap.parse_args(argv)
    from surfgym.core import STATE_DTYPE
    z = np.load(a.archive, allow_pickle=False)
    st, parent, depth = z["state"], z["parent"], z["depth"]
    surv = subtree_survival(parent, depth)
    keep = surv >= int(round(a.min_secs * 1000.0 / a.tick_ms))
    rows = np.zeros(int(keep.sum()), STATE_DTYPE)
    for f in STATE_DTYPE.names:
        rows[f] = st[f][keep]
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    tmp = Path(a.out).with_suffix(".tmp.npy")
    np.save(tmp, rows)
    tmp.replace(a.out)
    o = rows["origin"]
    v = rows["velocity"].astype(np.float64)
    E = o[:, 2] + (v ** 2).sum(1) / 1600.0
    print(f"{a.archive}: {len(st):,} cells, {int(keep.sum()):,} survive >= {a.min_secs:g} s "
          f"after (subtree depth) -> {a.out}; z {o[:,2].min():.0f}..{o[:,2].max():.0f}, "
          f"E p10/p50/p90 {np.percentile(E,10):.0f}/{np.percentile(E,50):.0f}/{np.percentile(E,90):.0f}")


if __name__ == "__main__":
    main()
