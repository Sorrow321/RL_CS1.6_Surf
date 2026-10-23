#!/usr/bin/env python3
"""gx_spawns.py - a --spawn-states file from a Go-Explore phase-1 archive (tools/explore_phase1.py).

Go-Explore's phase 2 (Ecoffet et al., arXiv 1901.10995 / 2004.12919) robustifies the
trajectories phase 1 found by training from their states. This tool takes an archive.npz, ranks
its cells by progress on the goal field (d at the true start minus d at the cell), keeps the
--top deepest live cells, follows each one's parent chain back to the root (the map's true
start), and writes the union of those chains as a 1-D STATE_DTYPE .npy for the trainer's
--spawn-states (clocks are zeroed by the trainer). Nothing here reads a demo or a record: the
archive is the search's own, rooted at the true start, explored with random actions.

    python tools/gx_spawns.py runs/gx/archive.npz --field maps/<map>.goalc_32.npz \
        --out runs/gx/spawns.npy --top 64

The file is written atomically (tmp + rename), so a running trainer that re-reads it on mtime
change never sees a half-written pool. --min-depth-frac drops chain states shallower than that
fraction of each chain's own progress (0 = keep the whole chain from the start).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE          # noqa: E402
from surfgym.goalfield import GoalField       # noqa: E402


def load_field(path):
    z = np.load(path, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def chains(parent: np.ndarray, picks) -> list[list[int]]:
    """Each pick's chain root -> pick, as archive indices (a cycle guard caps the walk)."""
    out = []
    for i in picks:
        c, j = [], int(i)
        while j >= 0 and len(c) < len(parent) + 1:
            c.append(j)
            j = int(parent[j])
        out.append(c[::-1])
    return out


def build(arch: dict, field, top: int, min_depth_frac: float = 0.0):
    st = arch["state"]
    n = len(st)
    o = st["origin"].astype(np.float64)
    d = field.sample(o).astype(np.float64)
    live = d < field.reach_max - 0.5 * field.cell
    roots = np.flatnonzero(arch["parent"][:n] < 0)
    d0 = float(np.min(d[roots])) if len(roots) else float(d[0])
    prog = np.where(live, d0 - d, -np.inf)
    picks = np.argsort(-prog)[:max(1, int(top))]
    picks = [int(i) for i in picks if np.isfinite(prog[i])]
    keep = set()
    for c, p in zip(chains(arch["parent"][:n], picks), picks):
        if min_depth_frac <= 0.0:
            keep.update(c)
        else:
            lim = float(min_depth_frac) * float(prog[p])
            keep.update(j for j in c if prog[j] >= lim)
    idx = np.array(sorted(keep), np.int64)
    rows = st[idx].copy()
    info = {"cells": int(n), "picked": len(picks), "states": int(len(rows)),
            "d0": d0, "best_progress": float(prog[picks[0]]) if picks else 0.0,
            "best_progress_pct": 100.0 * float(prog[picks[0]]) / d0 if picks else 0.0,
            "depth_max": int(np.max(arch["depth"][idx])) if len(idx) else 0}
    return rows, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="explore_phase1.py archive.npz")
    ap.add_argument("--field", required=True, help="the goal field .npz (progress ranking)")
    ap.add_argument("--out", required=True, help="the --spawn-states .npy to write")
    ap.add_argument("--top", type=int, default=64, help="deepest live cells whose chains are kept")
    ap.add_argument("--min-depth-frac", type=float, default=0.0,
                    help="drop chain states below this fraction of the chain's progress")
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    arch = {k: z[k] for k in z.files}
    if arch["state"].dtype != STATE_DTYPE:
        raise SystemExit(f"{a.archive}: state dtype is not STATE_DTYPE")
    rows, info = build(arch, load_field(a.field), a.top, a.min_depth_frac)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, rows)
    os.replace(tmp, out)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
