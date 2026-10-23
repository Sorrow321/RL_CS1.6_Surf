#!/usr/bin/env python3
"""archive_chain.py - the DENSE chain from a Go-Explore archive's root to one of its cells, for
a backward-curriculum spine (train_fast --demo-file ... --demo-reach R).

explore_phase1.py --dump-paths saves, for every new cell, the burst states (one per decision)
from the burst's restored start to the cell's entry. Walking the archive's parent pointers from
a target cell back to the root and concatenating those paths gives a route at decision
resolution. Two things make the raw concatenation wrong, and this tool fixes the first:

* one burst often enters SEVERAL cells of the chain, so their paths share the burst's prefix
  and the concatenation REPEATS segments. DemoCurriculum.match() attributes a state to the LAST
  row with its origin, so a band of repeated rows never collects an episode and the curriculum
  stalls for good (pbr1b2 arm 12, 280M steps). Every state is kept once, at its FIRST occurrence,
  which restores the order (a prefix repeated later is dropped and the rest follows it).
* a cell's stored state can be replaced by a shorter way in, so consecutive rows can come from
  different bursts (a stitch). A curriculum only resets into the rows, so that is sound for
  spawns; it is not an open-loop replay.

Nothing here reads the map beyond the goal field the reward shapes on (to pick the deepest cell).

    python tools/archive_chain.py runs/ge/archive.npz runs/ge/paths.npy runs/ge/paths_cell.npy \
        runs/ge/chain.npy --field maps/<map>.goalc_32.npz [--cell K] [--cut N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def parent_chain(parent, target: int):
    out, i = [], int(target)
    while i >= 0 and len(out) <= len(parent):
        out.append(i)
        i = int(parent[i])
    return out[::-1]


def dense_rows(paths, paths_cell, chain):
    """The burst rows of every chain cell, in chain order (with the repeats)."""
    parts = [paths[paths_cell == c] for c in chain]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else paths[:0]


def dedup_first(rows, nd: int = 1):
    """Keep each state (origin + velocity, rounded to nd decimals) at its first occurrence."""
    key = np.round(np.concatenate([np.asarray(rows["origin"], np.float64),
                                   np.asarray(rows["velocity"], np.float64)], axis=1), nd)
    _, first = np.unique(key, axis=0, return_index=True)
    return rows[np.sort(first)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("paths")
    ap.add_argument("paths_cell")
    ap.add_argument("out")
    ap.add_argument("--field", required=True, help="goal field .npz (picks the deepest cell)")
    ap.add_argument("--cell", type=int, default=None, help="target cell (default: the deepest)")
    ap.add_argument("--cut", type=int, default=0, help="keep the first N rows (0 = all)")
    a = ap.parse_args(argv)
    from archive_to_spawn import load_field
    z = np.load(a.archive, allow_pickle=False)
    f = load_field(a.field)
    d = f.sample(z["state"]["origin"]).astype(np.float64)
    tgt = int(np.argmin(d)) if a.cell is None else int(a.cell)
    chain = parent_chain(z["parent"], tgt)
    raw = dense_rows(np.load(a.paths, allow_pickle=False),
                     np.load(a.paths_cell, allow_pickle=False), chain)
    rows = dedup_first(raw)
    if a.cut > 0:
        rows = rows[:a.cut]
    rows = np.array(rows, copy=True)
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    np.save(a.out, rows)
    meta = {"archive": a.archive, "target_cell": tgt, "chain_cells": len(chain),
            "raw_rows": int(len(raw)), "rows": int(len(rows)), "cut": int(a.cut),
            "end_d": float(f.sample(rows["origin"][-1:])[0]) if len(rows) else None}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
