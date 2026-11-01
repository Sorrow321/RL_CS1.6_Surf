#!/usr/bin/env python3
"""finish_roots.py - roots for tools/cem_finish.py: the policy's own contact states nearest to the
states a finish has already been planned from.

Input: a pool of the policy's own states (tools/frontier_states.py on a fresh greedy
re-recording; consecutive rows are consecutive ticks of an episode) and the finishing lines
found so far (cem_finish fin_*.npz: states[0] is the root each line was planned from). Output:
up to --k pool rows that are in CONTACT (the next row's vz departs from free fall: a surface or
the ground is acting - an airborne state's apex is already fixed by its vz), ranked by the
distance |dpos| / --pos-scale + |dvel| / --vel-scale to the nearest earlier root, at least
--min-gap rows apart. With no earlier line it ranks by the goal field instead (the deepest
contact states first). Generic: the rule is "near where the planner succeeded before"; nothing
names a place.

    python tools/finish_roots.py POOL.npy --lines 'dir/*_fin_*.npz' --k 16 [--tick-ms 7]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "autoresearch"))


def contact_rows(pool: np.ndarray, tick_s: float, gravity: float = 800.0,
                 jump: float = 200.0, tol: float = 1.0) -> np.ndarray:
    o = np.asarray(pool["origin"], np.float64)
    v = np.asarray(pool["velocity"], np.float64)
    n = len(o)
    nxt_ok = np.zeros(n, bool)
    if n > 1:
        nxt_ok[:-1] = np.linalg.norm(np.diff(o, axis=0), axis=1) <= jump
    dvz = np.zeros(n)
    if n > 1:
        dvz[:-1] = v[1:, 2] - v[:-1, 2]
    return nxt_ok & (np.abs(dvz + gravity * tick_s) > tol)


def pick(pool, prev_roots, k: int, tick_s: float, pos_scale=200.0, vel_scale=300.0,
         min_gap=150, field=None) -> list[int]:
    o = np.asarray(pool["origin"], np.float64)
    v = np.asarray(pool["velocity"], np.float64)
    c = contact_rows(pool, tick_s)
    if len(prev_roots):
        po = np.asarray(prev_roots["origin"], np.float64)
        pv = np.asarray(prev_roots["velocity"], np.float64)
        D = np.min([np.linalg.norm(o - po[j], axis=1) / pos_scale
                    + np.linalg.norm(v - pv[j], axis=1) / vel_scale
                    for j in range(len(po))], axis=0)
    elif field is not None:
        D = field.sample(o).astype(np.float64)
    else:
        raise SystemExit("finish_roots: no earlier line and no field to rank by")
    D = np.where(c, D, np.inf)
    out = []
    for i in np.argsort(D):
        if not np.isfinite(D[i]) or len(out) >= k:
            break
        if all(abs(int(i) - p) >= min_gap for p in out):
            out.append(int(i))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pool")
    ap.add_argument("--lines", action="append", default=[])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tick-ms", type=float, default=7.0)
    ap.add_argument("--min-gap", type=int, default=150)
    ap.add_argument("--exclude", default="", help="comma-separated pool rows already tried")
    a = ap.parse_args(argv)
    pool = np.load(a.pool, allow_pickle=False)
    roots = [np.load(f, allow_pickle=False)["states"][:1]
             for g in a.lines for f in sorted(glob.glob(g))]
    prev = np.concatenate(roots) if roots else np.zeros(0, pool.dtype)
    field = None
    if not len(prev):
        import score as S
        field = S.field()
    picks = pick(pool, prev, a.k + 64, a.tick_ms / 1000.0, min_gap=a.min_gap, field=field)
    ex = {int(x) for x in a.exclude.split(",") if x.strip()}
    picks = [p for p in picks if all(abs(p - e) >= a.min_gap for e in ex)][:a.k]
    print(",".join(str(p) for p in picks))


if __name__ == "__main__":
    main()
