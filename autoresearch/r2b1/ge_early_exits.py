#!/usr/bin/env python3
"""ge_early_exits.py (uf2r2b1) - ANALYSIS of an explore_phase1 archive rooted at a line's own states:
which archived cells are past progress P (on the grader's no-corner-cut field) and how EARLY they were
reached, counted from the map start (the root's row on the line + the chain's ticks, the explorer's whole-ms
tick converted to the line's tick). Reporting only; nothing here trains.

    python autoresearch/r2b1/ge_early_exits.py runs/ge_pit1/archive.npz runs/beam/r2s2_pitroots.npy \
        --row0 450 --stride 10 --ge-tick 8 --line-tick 7.666667 --prog 1.7
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("roots")
    ap.add_argument("--row0", type=int, default=450)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--ge-tick", type=float, default=8.0)
    ap.add_argument("--line-tick", type=float, default=7.666667)
    ap.add_argument("--prog", type=float, default=1.7)
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_unitfarmer2.goalc_32.npz"))
    ap.add_argument("--d0", type=float, default=30647.5)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    from auto_spine import load_field
    f = load_field(a.field)
    z = np.load(a.archive, allow_pickle=False)
    st, par, dep = z["state"], z["parent"].astype(np.int64), z["depth"].astype(np.int64)
    n = len(st)
    roots = np.load(a.roots)
    ro = np.asarray(roots["origin"], np.float64)
    o = np.asarray(st["origin"], np.float64)
    v = np.asarray(st["velocity"], np.float64)
    d = np.asarray(f.sample(o), np.float64)
    bad = d >= f.reach_max - 0.5 * f.cell
    prog = np.where(bad, -999.0, 100.0 * (a.d0 - d) / a.d0)
    E = o[:, 2] + (v * v).sum(1) / 1600.0
    # each cell's root: follow parents to -1
    root_of = np.full(n, -1, np.int64)
    for i in range(n):
        if root_of[i] >= 0:
            continue
        path = []
        j = i
        while j >= 0 and root_of[j] < 0 and par[j] >= 0:
            path.append(j)
            j = int(par[j])
        r = root_of[j] if root_of[j] >= 0 else j
        root_of[j] = r
        for k in path:
            root_of[k] = r
    # root cell -> row on the line (match its origin to the roots file)
    row_of_root = {}
    for r in np.unique(root_of):
        dd = np.linalg.norm(ro - o[r], axis=1)
        k = int(np.argmin(dd))
        row_of_root[int(r)] = a.row0 + a.stride * k if dd[k] < 1.0 else -1
    rows = np.array([row_of_root[int(r)] for r in root_of])
    arrive = rows + dep * (a.ge_tick / a.line_tick)          # in line ticks from the map start
    tsec = arrive * a.line_tick / 1000.0
    ok = (prog >= a.prog) & (rows >= 0)
    print(f"{n:,} cells; {int((rows < 0).sum())} unmatched roots; past {a.prog}%: {int(ok.sum()):,}")
    if ok.any():
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(tsec[idx])][:a.top]
        print("earliest arrivals past the threshold (t from the map start, line tick):")
        for i in idx:
            print(f"  t {tsec[i]:6.2f}s prog {prog[i]:6.2f}% ({o[i,0]:6.0f},{o[i,1]:6.0f},{o[i,2]:6.0f}) "
                  f"|v| {np.linalg.norm(v[i]):5.0f} E {E[i]:6.0f} root row {rows[i]} depth {dep[i]} cell {i}")
    # best progress reached by time
    for T in (6, 7, 8, 9, 10, 11, 12):
        m = (tsec <= T) & (rows >= 0)
        if m.any():
            i = int(np.flatnonzero(m)[np.argmax(prog[m])])
            print(f"  by t {T:2d}s: best prog {prog[i]:6.2f}% at ({o[i,0]:6.0f},{o[i,1]:6.0f},{o[i,2]:6.0f}) "
                  f"E {E[i]:6.0f} t {tsec[i]:5.2f} cell {i}")


if __name__ == "__main__":
    main()
