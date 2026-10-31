#!/usr/bin/env python3
"""ge_return_states.py - a Go-Explore archive's RETURN distribution as a spawn-state file (uf2r1b2).

Go-Explore returns to an archive cell with probability W = 1/sqrt(C_seen + 1), and under
--select-energy the most energetic cell of each place is drawn (1 + K) times as often
(tools/explore_phase1.py Archive.build_sampler). This draws N rows of the archive with exactly
that weight (with replacement) and writes them as STATE_DTYPE rows, clocks zeroed, for
train_fast --spawn-states (whose draws are uniform over the file's rows, so the file carries the
weight). The archive is the run's own reward-free search from the map's true spawn: no demo, no
record, no reward and no map constant (CLAUDE.md sections 0 / 0b); the only parameters are the
archive's own cell and --select-energy.

    python tools/ge_return_states.py --archive runs/ge/archive.npz --out runs/ge/return.npy \\
        --n 60000 [--select-energy 3] [--seed 0]

--tops writes instead the most energetic cell of EVERY place, once each (the set --select-energy
boosts; uniform over places): on uf2r1b2_ge1 25.6% of those rows are high-energy pit states
against 5.1% of the return draw, which the pit's floor cells (many, slow, doomed) dominate.

--not-behind-roots FIELD keeps only rows that are not behind the search's own roots on the field:
d <= the largest d among the root cells (parent -1). A search rooted at the policy's own line
(--roots-spine) also wanders back down the route; this is princessburglar's "deeper than the
start" rule (round 1), measured from the search's own roots, with no map constant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def energy_tops(state, hasher) -> np.ndarray:
    """Index of the most energetic cell of each place (explore_phase1 Archive.energy_tops)."""
    from explore_phase1 import CellHash
    pos = hasher.pos_keys(state)
    o = state["origin"].astype(np.float64)
    v = state["velocity"].astype(np.float64)
    e = o[:, 2] + (v * v).sum(axis=1) / (2.0 * CellHash.G_ENERGY)
    order = np.lexsort((-e, pos))
    ps = pos[order]
    first = np.ones(len(order), bool)
    first[1:] = ps[1:] != ps[:-1]
    return order[first]


def return_weights(state, seen, hasher, select_energy: float) -> np.ndarray:
    """W = 1/sqrt(seen + 1), x (1 + K) on each place's most energetic cell (explore_phase1)."""
    from explore_phase1 import CellHash
    w = 1.0 / np.sqrt(np.asarray(seen, np.float64) + 1.0)
    if select_energy > 0.0 and len(state):
        w[energy_tops(state, hasher)] *= 1.0 + select_energy
    return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--select-energy", type=float, default=None,
                    help="default: the archive's own (its meta json), else 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tops", action="store_true",
                    help="write each place's most energetic cell once (no draw; --n ignored)")
    ap.add_argument("--not-behind-roots", default=None, metavar="FIELD",
                    help="keep only rows with d <= the largest d of the archive's root cells on "
                         "this GoalField .npz (no row behind where the search was rooted)")
    a = ap.parse_args(argv)
    from explore_phase1 import CellHash
    from surfgym.core import STATE_DTYPE
    z = np.load(a.archive, allow_pickle=False)
    meta = json.loads(Path(a.archive).with_suffix(".meta.json").read_text())
    st, seen = z["state"], z["seen"]
    k = float(meta.get("select_energy", 0.0) if a.select_energy is None else a.select_energy)
    hasher = CellHash(mins=meta["grid_mins"], maxs=[1e9, 1e9, 1e9], cell=float(meta["cell"]))
    hasher.dims = tuple(int(x) for x in meta["grid_dims"])
    keep = None
    if a.not_behind_roots:
        from surfgym.goalfield import GoalField
        zf = np.load(a.not_behind_roots, allow_pickle=False)
        gf = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                       float(zf["cell"]), float(zf["reach_max"]))
        d = gf.sample(st["origin"].astype(np.float64)).astype(np.float64)
        roots = np.flatnonzero(np.asarray(z["parent"]) < 0) if "parent" in z.files else np.arange(0)
        if roots.size == 0:
            raise SystemExit("--not-behind-roots: the archive has no root cells (parent -1)")
        keep = d <= float(d[roots].max())
    if a.tops:
        idx = np.sort(energy_tops(st, hasher))
        if keep is not None:
            idx = idx[keep[idx]]
    else:
        w = return_weights(st, seen, hasher, k)
        if keep is not None:
            w = np.where(keep, w, 0.0)
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(len(st), size=int(a.n), replace=True, p=w / w.sum())
    rows = np.zeros(len(idx), STATE_DTYPE)
    for f in STATE_DTYPE.names:
        rows[f] = st[f][idx]
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    np.save(a.out, rows)
    print(json.dumps({"archive": a.archive, "cells": int(len(st)), "rows": int(len(rows)),
                      "distinct": int(len(np.unique(idx))), "select_energy": k, "seed": a.seed, "tops": bool(a.tops),
                      "not_behind_roots": a.not_behind_roots,
                      "out": a.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
