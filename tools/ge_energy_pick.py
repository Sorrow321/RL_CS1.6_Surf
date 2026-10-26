#!/usr/bin/env python3
"""ge_energy_pick.py - the most ENERGETIC cell of an exact-chains archive past a progress level
(pbr5b1, 2026-10-25).

    python tools/ge_energy_pick.py ARCHIVE.npz FIELD.npz --d-max D [--top 5] [--line LINE.npy]

Among the archive's cells whose shaping-field distance is at most --d-max (as deep as a line
must get), the --top ones with the highest energy height h = z + |v|^2 / 2g, with their chain
length and root. With --line (the policy's own line the explorer was rooted on), each pick's h is
compared with the line's h at its first arrival at the same depth: an exploration rooted BEFORE
the line's energy losses (hard landings) can come out of them with more energy at matched
progress, and the next gate may need exactly that energy (round 3's "pick by energy, not
depth"). Prints one JSON line per pick; the first is the best. Nothing here names a place on a
map: a depth on the field and an energy height (CLAUDE.md 0b)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def load_field(path):
    from surfgym.goalfield import GoalField
    z = np.load(path, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def energy(states, g=800.0):
    o = np.asarray(states["origin"], np.float64)
    v = np.asarray(states["velocity"], np.float64)
    return o[:, 2] + (v * v).sum(axis=1) / (2.0 * g)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("field")
    ap.add_argument("--d-max", type=float, required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-sep", type=float, default=128.0,
                    help="picks at least this far apart (u), so --top is not one cluster")
    ap.add_argument("--line", default=None)
    ap.add_argument("--alive-margin", type=float, default=300.0,
                    help="keep only cells whose SUBTREE (the explorer's runs restored from them and "
                         "their descendants) reaches at least this many u deeper: free fall conserves "
                         "z + |v|^2/2g, so without it an energy pick chooses runs on their way into a "
                         "kill volume (pbr5b1, 2026-10-25: all four first picks were fatal falls). "
                         "0 = off")
    a = ap.parse_args(argv)
    f = load_field(a.field)
    z = np.load(a.archive, allow_pickle=False)
    st, parent = z["state"], z["parent"]
    d = np.asarray(f.sample(np.asarray(st["origin"], np.float64)), np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    h = energy(st)
    alive = np.ones(len(st), bool)
    if a.alive_margin > 0:
        # the subtree's deepest d, pushed up from children (appended after their parents)
        sub = d.copy()
        par = np.asarray(parent, np.int64)
        for c in range(len(st) - 1, -1, -1):
            p = par[c]
            if p >= 0 and sub[c] < sub[p]:
                sub[p] = sub[c]
        alive = sub <= d - float(a.alive_margin)
    ok = np.flatnonzero((d <= a.d_max) & alive)
    if not len(ok):
        print(json.dumps({"picks": 0, "cells": int(len(st)), "best_d": float(d.min())}))
        return 2
    lh = None
    if a.line:
        ln = np.load(a.line)
        ld = np.asarray(f.sample(np.asarray(ln["origin"], np.float64)), np.float64)
        lbest = np.minimum.accumulate(np.where(ld < f.reach_max - 0.5 * f.cell, ld, np.inf))
        lh = (lbest, energy(ln))
    order = ok[np.argsort(-h[ok])]
    picks = []
    for c in order:
        o = np.asarray(st["origin"][c], np.float64)
        if any(np.linalg.norm(o - np.asarray(st["origin"][p], np.float64)) < a.min_sep
               for p in picks):
            continue
        picks.append(int(c))
        depth, r = 0, int(c)
        while parent[r] >= 0 and depth < 10000:
            r = int(parent[r])
            depth += 1
        row = {"cell": int(c), "d": round(float(d[c]), 1), "h": round(float(h[c]), 1),
               "origin": [round(float(x)) for x in o],
               "speed": round(float(np.linalg.norm(st["velocity"][c])), 1),
               "chain_cells": depth, "root_cell": r}
        if lh is not None:
            j = np.flatnonzero(lh[0] <= d[c])
            if len(j):
                row["line_h_at_d"] = round(float(lh[1][j[0]]), 1)
                row["dh_vs_line"] = round(float(h[c] - lh[1][j[0]]), 1)
        print(json.dumps(row))
        if len(picks) >= a.top:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
