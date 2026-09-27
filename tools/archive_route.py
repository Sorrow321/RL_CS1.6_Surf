"""archive_route.py - a SELF_STATES route to an archive's FRONTIER: the flown path from the root to
the live node with the largest progress on the map's own goal potential.

    python tools/archive_route.py runs/research/<archive>/nodes.npz --map maps_pool/<m>.bsp \
        --out runs/research/<archive>/frontier_route.npz [--goal-cell 32] [--spacing 128]

WHY. The pipeline trains the flat policy on the FIRST FINISHING chain. On a map where the search
has not reached the finish within its budget, the generic fallback is the chain to the frontier:
the node closest to the goal by the map's goal potential (the geodesic field every map has - a
rule with no map constant; on a deceptive map it may pick a dead end, which is the potential's
own error, not a tuned choice). The route is the agent's own flown path (edge_archive.py
--dump-nodes keeps every node's path), resampled at the fan spacing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nodes")
    ap.add_argument("--map", required=True)
    ap.add_argument("--goal-cell", type=int, default=32)
    ap.add_argument("--spacing", type=float, default=128.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from surfgym.goalfield import load_goal_field
    from surfgym.route import resample_polyline
    z = np.load(a.nodes)
    if "all_path_off" not in z.files:
        raise SystemExit("archive_route: this dump has no paths (edge_archive.py --dump-nodes "
                         "after 2026-09-27 07:35)")
    bsp = Path(a.map)
    gf = load_goal_field(str(bsp.with_name(f"{bsp.stem}.goal_{a.goal_cell}.npz")))
    o = z["origin"].astype(np.float64)
    d = gf.sample(o).astype(np.float64)
    d0 = float(gf.sample(z["root"].astype(np.float64)[None])[0])
    prog = np.where(np.isfinite(d), d0 - d, -np.inf)
    j = int(np.argmax(prog))
    nid = int(z["node_id"][j])
    ap_, pts, off = z["all_parent"], z["all_path_pts"], z["all_path_off"]
    chain = []
    while nid >= 0:
        chain.append(nid)
        nid = int(ap_[nid])
    chain = chain[::-1]
    seg = [pts[off[i]:off[i + 1]] for i in chain]
    path = []
    for sgm in seg:
        for q in np.asarray(sgm, np.float64):
            if not path or np.linalg.norm(q - path[-1]) > 1.0:
                path.append(q)
    line, total = resample_polyline(np.asarray(path), float(a.spacing))
    secs = float(z["t"][j]) / 100.0
    np.savez(a.out, route=np.asarray(line, np.float32), spacing=np.float64(a.spacing),
             seconds=np.float64(secs), map=np.array(bsp.stem),
             source=np.array(f"tools/archive_route.py: the chain to the archive node with the "
                             f"largest goal-potential progress ({prog[j]:,.0f} u) in {a.nodes} - "
                             f"the agent's own search, no human input"))
    print(f"archive_route: node {int(z['node_id'][j])} at {np.round(o[j]).astype(int).tolist()} "
          f"progress {prog[j]:,.0f} u, depth {len(chain) - 1} moves, {secs:.1f} s from the root; "
          f"route {total:,.0f} u, {len(line)} vertices -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
