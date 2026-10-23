#!/usr/bin/env python3
"""archive_spawns.py - training spawn states from a Go-Explore archive: the chain from the map's
TRUE START to the archive cell with the best goal potential.

    python tools/archive_spawns.py runs/ge/archive.npz --map maps/<m>.bsp --out runs/ge/chain.npy

The archive is tools/explore_phase1.py's (reward-free random macro-action bursts from the map
spawn, returned to by state restore; no demo, no policy, no reward). Its cell with the lowest
distance on the map's own goal field (the no-corner-cut field the race reward shapes on, unless
--field says otherwise) is the frontier, and the provenance chain root -> frontier is a route the
simulator itself flew from the true start - the "explore" half of Go-Explore. The output is that
chain's states (STATE_DTYPE, time-ordered, root first) for train_fast.py --spawn-states: the
"robustify" half, done by RL from states along the route instead of a backward curriculum (the
chain stops short of the finish, so there is no goal end to anchor one). The rule has no map
constant: on a deceptive map the best-potential cell may be a dead end, which is the field's own
error, exactly as for tools/archive_route.py.

--all-every K also appends every K-th archive cell (sorted by depth) as coverage starts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def load_field(bsp: Path, kind: str, cell: int):
    from surfgym.goalfield import GoalField
    name = {"goalc": f"{bsp.stem}.goalc_{cell}.npz", "goal": f"{bsp.stem}.goal_{cell}.npz"}[kind]
    z = np.load(bsp.parent / name, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def chain_of(parent: np.ndarray, i: int) -> list[int]:
    out, seen = [], set()
    while i >= 0 and i not in seen:
        seen.add(i)
        out.append(int(i))
        i = int(parent[i])
    out.reverse()
    return out


def build(archive: Path, bsp: Path, field_kind: str = "goalc", cell: int = 32,
          all_every: int = 0):
    z = np.load(archive, allow_pickle=False)
    st, par, dep = z["state"], z["parent"], z["depth"]
    f = load_field(bsp, field_kind, cell)
    org = st["origin"].astype(np.float64)
    d = f.sample(org).astype(np.float64)
    ok = d < f.reach_max - 0.5 * f.cell
    if not ok.any():
        raise SystemExit("archive_spawns: no archive cell is on the field")
    best = int(np.argmin(np.where(ok, d, np.inf)))
    ch = chain_of(par, best)
    if int(par[ch[0]]) != -1 or int(dep[ch[0]]) != 0:
        raise SystemExit("archive_spawns: the chain does not reach a root")
    rows = st[ch].copy()
    extra = np.zeros(0, dtype=st.dtype)
    if all_every > 0:
        order = np.argsort(dep, kind="stable")
        pick = order[::all_every]
        pick = pick[~np.isin(pick, ch)]
        extra = st[pick].copy()
    out = np.concatenate([rows, extra]) if len(extra) else rows
    d0 = float(d[ch[0]])
    info = {
        "archive": str(archive), "field": f"{bsp.stem}.{field_kind}_{cell}",
        "cells": int(len(st)), "best_index": best, "best_d": float(d[best]),
        "root_d": d0, "best_progress_pct": round(100.0 * (d0 - float(d[best])) / d0, 3),
        "best_depth_ticks": int(dep[best]), "chain_len": len(ch), "extra_cells": int(len(extra)),
        "provenance": "tools/explore_phase1.py archive: reward-free random macro-action bursts "
                      "from the map's true start, returned to by state restore; no demo, no "
                      "policy, no reward. Chain = root -> the cell with the lowest goal-field "
                      "distance.",
    }
    return out, info, d[ch], dep[ch]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive")
    ap.add_argument("--map", required=True)
    ap.add_argument("--field", default="goalc", choices=("goalc", "goal"))
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--all-every", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out, info, dch, depch = build(Path(a.archive), Path(a.map), a.field, a.cell, a.all_every)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npy")
    np.save(tmp, out)
    tmp.replace(p)                       # atomic: a running trainer re-reads on mtime change
    p.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    for k in range(0, len(dch), max(1, len(dch) // 12)):
        print(f"  chain[{k:3d}] depth {int(depch[k]):6d}t  d {dch[k]:9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
