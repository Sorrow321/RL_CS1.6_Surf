#!/usr/bin/env python3
"""energy_pick.py - the Go-Explore cell that keeps the most energy where the policy's line loses it (pbr5b2).

Round 5 passed princessburglar's 56.6% shaft climb by teaching the policy a line found for its
ENERGY, not its depth (autoresearch/NOTES.md, "c5: the energy-gated curriculum"): the explorer's
archive held states that kept ~600 u more energy height through a descent than the policy's own line,
at the same depth. This tool finds such a state in an exact-chains archive (tools/explore_phase1.py
--exact-chains) against a reference line (the policy's beam line or spine):

* energy height h = z + |v|^2 / (2 g); a candidate is a cell in the progress window [--lo, --hi] (%
  of the start's distance on --field) that is not plunging (vz > --vz-min), lies within --near u of
  the line (on the route), and whose chain from its root is not slower than the line from the same
  root to the same progress by more than --slack ticks (a chain that wanders for seconds gathers
  HEIGHT, not speed);
* its gain = h(cell) - h(line at the first tick it reaches that progress); the best gain wins.

It prints a per-bin table and writes the winner as JSON (--out); with --replay it calls
tools/ge_replay.py --cell to write the chain (spine material for tools/stitch_spine.py, then
train_fast --demo-success-energy). Nothing comes from a demo: the archive is the run's own reward-free
search, the line the policy's own.

    python tools/energy_pick.py --archive runs/x/ge/archive.npz --line runs/x/beam/beam_frontier.npz \\
        --field runs/x/field.npz --lo 50 --hi 55 --out runs/x/pick.json [--replay runs/x/chain.npy --map M]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

G_ENERGY = 800.0


def energy_height(o, v) -> np.ndarray:
    o = np.asarray(o, np.float64).reshape(-1, 3)
    v = np.asarray(v, np.float64).reshape(-1, 3)
    return o[:, 2] + (v * v).sum(axis=1) / (2.0 * G_ENERGY)


def load_line(path):
    z = np.load(path, allow_pickle=False)
    return z["states"] if hasattr(z, "files") else z


def progress(field, o, d0: float) -> np.ndarray:
    d = np.asarray(field.sample(np.asarray(o, np.float64)), np.float64)
    d = np.where(d < field.reach_max - 0.5 * field.cell, d, np.inf)
    return 100.0 * (d0 - d) / d0


def roots_of(parent: np.ndarray) -> np.ndarray:
    """Each cell's root (parents precede their children in an archive)."""
    parent = np.asarray(parent, np.int64)
    root = np.empty(len(parent), np.int64)
    for i in range(len(parent)):
        root[i] = i if parent[i] < 0 else root[parent[i]]
    return root


def pick(st, parent, depth, line, field, d0: float, lo: float, hi: float, near: float = 150.0,
         slack: int = 130, vz_min: float = -300.0):
    """-> (best (dict or None), candidates [(gain, cell, ...)] sorted by gain)."""
    from scipy.spatial import cKDTree
    o = np.asarray(st["origin"], np.float64)
    v = np.asarray(st["velocity"], np.float64)
    h = energy_height(o, v)
    p = progress(field, o, d0)
    Lo = np.asarray(line["origin"], np.float64)
    Lh = energy_height(Lo, line["velocity"])
    Lp = progress(field, Lo, d0)
    best_line = np.maximum.accumulate(np.where(np.isfinite(Lp), Lp, -1e9))
    tree = cKDTree(Lo)
    dist, _ = tree.query(o)
    root = roots_of(parent)
    _, rj = tree.query(o[root])
    m = (p >= lo) & (p <= hi) & (v[:, 2] > vz_min) & (dist <= near) & np.isfinite(p)
    cands = []
    for i in np.flatnonzero(m):
        k = np.flatnonzero(best_line >= p[i])
        if not len(k):
            continue
        k = int(k[0])
        t_line = k - int(rj[i])
        if t_line < 0 or int(depth[i]) > t_line + int(slack):
            continue
        cands.append({"cell": int(i), "gain": float(h[i] - Lh[k]), "h": float(h[i]),
                      "h_line": float(Lh[k]), "progress": float(p[i]),
                      "speed": float(np.linalg.norm(v[i])), "depth": int(depth[i]),
                      "line_ticks": int(t_line), "root_row": int(rj[i]),
                      "origin": [float(x) for x in o[i]]})
    cands.sort(key=lambda c: -c["gain"])
    return (cands[0] if cands else None), cands


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", required=True)
    ap.add_argument("--line", required=True, help="the reference line: a STATE_DTYPE .npy or a "
                                                  "beam_tas --save-frontier .npz")
    ap.add_argument("--field", required=True)
    ap.add_argument("--lo", type=float, required=True, help="progress window, %% of the start's d")
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--near", type=float, default=150.0, help="u from the line (on the route)")
    ap.add_argument("--slack", type=int, default=130, help="ticks a chain may be slower than the line")
    ap.add_argument("--vz-min", type=float, default=-300.0, help="drop plunging states")
    ap.add_argument("--d0", type=float, default=None, help="the start's d (default: the line's first)")
    ap.add_argument("--out", default=None, help="the winner as JSON")
    ap.add_argument("--replay", default=None, help="write the winner's chain here (ge_replay --cell)")
    ap.add_argument("--map", default=None, help="for --replay")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(argv)
    from auto_spine import load_field
    field = load_field(a.field)
    z = np.load(a.archive, allow_pickle=False)
    line = load_line(a.line)
    d0 = (float(a.d0) if a.d0 is not None
          else float(field.sample(np.asarray(line["origin"][:1], np.float64))[0]))
    best, cands = pick(z["state"], z["parent"], z["depth"], line, field, d0, a.lo, a.hi,
                       a.near, a.slack, a.vz_min)
    print(f"{len(cands)} candidates in {a.lo:g}-{a.hi:g}% (not plunging, within {a.near:g} u of the "
          f"line, no more than {a.slack} ticks slower than it)")
    for c in cands[:a.top]:
        print(f"  cell {c['cell']}: {c['progress']:.2f}% h {c['h']:.0f} vs line {c['h_line']:.0f} "
              f"(gain {c['gain']:+.0f}) |v| {c['speed']:.0f} depth {c['depth']} vs {c['line_ticks']} "
              f"ticks from root row {c['root_row']}")
    if best is None:
        print("no candidate")
        return 2
    if a.out:
        Path(a.out).write_text(json.dumps(best, indent=1), encoding="utf-8")
    if a.replay:
        if not a.map:
            ap.error("--replay needs --map")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "ge_replay.py"), "--map", a.map,
                            "--archive", a.archive, "--field", a.field, "--cell", str(best["cell"]),
                            "--out", a.replay], cwd=str(ROOT))
        if r.returncode != 0:
            return r.returncode
    print("ENERGY_PICK " + json.dumps(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
