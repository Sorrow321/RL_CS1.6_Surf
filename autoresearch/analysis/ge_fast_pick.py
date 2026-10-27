#!/usr/bin/env python3
"""ge_fast_pick.py - the Go-Explore cells that reach a progress line EARLIEST from the true start (pbr9b1).

A Go-Explore archive (tools/explore_phase1.py --exact-chains) rooted at states of the policy's own
greedy line (each root = a line row, possibly with its velocity scaled) holds, for every cell, the
chain of per-tick links from its root. The arrival time from the true start of a cell is

    t = root_row * line_tick + chain_ticks * ge_tick

(the line's own clock to the root, then the explorer's). This lists, for the cells past --line
(d on --field at or below it), the earliest arrivals and their energy height h = z + |v|^2/2g
(the speed the line carries on), as a Pareto table over (t, -h). Nothing here is a reward or a
spawn rule: it picks which of the run's own search lines to replay (tools/ge_replay.py --cell).

    python autoresearch/analysis/ge_fast_pick.py --archive A.npz --roots-json R.json --field F.npz \\
        --line 7800 [--line-tick-ms 7.6667 --ge-tick-ms 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))


def chain_ticks_roots(parent, off):
    """(cumulative per-tick link length from each cell's root, each cell's root cell). A cell is
    archived after the cell its run was restored from (parent < index), so one forward pass."""
    n = len(parent)
    ln = (off[1:n + 1] - off[:n]).astype(np.int64)
    tot = np.zeros(n, np.int64)
    root = np.arange(n, dtype=np.int64)
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            if p >= i:
                raise SystemExit(f"cell {i} has a later parent {p}")
            tot[i] = tot[p] + ln[i]
            root[i] = root[p]
    return tot, root


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--roots-json", required=True, help="[(line row, boost), ...] in root order")
    ap.add_argument("--field", required=True)
    ap.add_argument("--line", type=float, required=True, help="d at or below which a cell counts")
    ap.add_argument("--line-tick-ms", type=float, default=7.6666667)
    ap.add_argument("--ge-tick-ms", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)
    from surfgym.goalfield import GoalField
    z = np.load(a.archive, allow_pickle=False)
    st, parent, off = z["state"], z["parent"].astype(np.int64), z["link_off"].astype(np.int64)
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    roots = json.loads(Path(a.roots_json).read_text())
    o = st["origin"].astype(np.float64)
    v = st["velocity"].astype(np.float64)
    d = f.sample(o).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    h = o[:, 2] + (v * v).sum(1) / 1600.0
    ct, rt = chain_ticks_roots(parent, off)
    nroot = int((parent < 0).sum())
    if nroot != len(roots):
        print(f"!! {nroot} root cells vs {len(roots)} roots in the json")
    row = np.array([roots[int(r)][0] if 0 <= r < len(roots) else -1 for r in rt])
    boost = np.array([roots[int(r)][1] if 0 <= r < len(roots) else np.nan for r in rt])
    t = row * a.line_tick_ms / 1000.0 + ct * a.ge_tick_ms / 1000.0
    ok = np.flatnonzero(d <= a.line)
    print(f"{len(st):,} cells, {nroot} roots; {len(ok):,} past d {a.line:,.0f}")
    if not len(ok):
        return 1
    order = ok[np.argsort(t[ok])]
    print(" cell      t_s    d      h     |vh|    vz   root_row boost chain_ticks  pos")
    for i in order[:a.top]:
        print(f"{i:7d} {t[i]:7.2f} {d[i]:6.0f} {h[i]:6.0f} {np.hypot(v[i,0], v[i,1]):6.0f} "
              f"{v[i,2]:6.0f} {row[i]:7d} {boost[i]:5.2f} {ct[i]:8d}  {o[i].round(0)}")
    # Pareto front over (earliest t, highest h)
    front, hbest = [], -np.inf
    for i in order:
        if h[i] > hbest + 1.0:
            front.append(i)
            hbest = h[i]
    print("Pareto (t up, h up):")
    for i in front[:40]:
        print(f"{i:7d} t {t[i]:7.2f} d {d[i]:6.0f} h {h[i]:6.0f} root {row[i]} x{boost[i]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
