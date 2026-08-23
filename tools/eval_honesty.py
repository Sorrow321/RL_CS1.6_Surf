#!/usr/bin/env python3
"""eval_honesty.py - what race/eval_progress refuses to tell you.

`race/eval_progress` is the geodesic potential's reading at the best point an
episode reached, and on surf_src_cannonball the potential and the task
DECOUPLE near the end: an agent that overshoots the finish and falls into
goal-adjacent space scores ~178,000 of 198,380 (89% of the route) without
ever crossing the line. Round 16 found the same thing from the other side -
every method bottomed out in the same off-route basin at d ~ 21.5k.

So a rising eval_progress is not evidence on its own. This scores recorded
episodes on two things a dive cannot fake:

  * **corridor progress** - how far along the REFERENCE ROUTE the episode
    got while still within `--corridor` units of it. Falling off the route
    stops the clock, because the pit is nowhere near the line. This is the
    honest analogue of eval_progress and is directly comparable between arms
    that share a route file.
  * **where it ended** - inside the finish box, short of it, or BELOW it
    (the dive signature: episodes ending far under the goal's z range).

    python tools/eval_honesty.py --route maps/surf_src_cannonball.route.npz \\
        runs/research/xROUTE/traj_*.jsonl

Route files come from tools/build_route.py. Nothing here needs a GPU, a map
or the goal field - it is pure geometry over the recorded positions, so it
can be run against any arm's trajectories after the box is gone.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from surfgym.route import episodes_from_traj  # noqa: E402


def load_route(path):
    z = np.load(path)
    pts = np.asarray(z["route"], np.float32)
    spacing = float(z["spacing"]) if "spacing" in z.files else 128.0
    return pts, spacing


def corridor_progress(xyz, pts, spacing, corridor):
    """-> (units along the route reached inside the corridor, max lateral).

    The route is monotone in arc length, so "how far did it get" is the
    largest vertex index whose distance to the trajectory ever fell inside
    the corridor - and, crucially, only counting vertices the episode
    actually reached in ORDER, so a trajectory that teleports past the
    corridor (falls, then drifts near a later stretch) cannot claim it.
    """
    d2 = ((xyz[:, None, :] - pts[None, :, :]) ** 2).sum(-1)   # (T, L)
    near = d2.min(axis=1)                                     # per sample
    idx = d2.argmin(axis=1)
    inside = near <= corridor * corridor
    if not inside.any():
        return 0.0, float(np.sqrt(near.min()))
    reached, best = 0, 0
    for k in range(len(idx)):
        if not inside[k]:
            continue
        # only advance; a later sample that snaps to an earlier vertex is
        # backtracking, not progress
        if idx[k] >= best:
            best = int(idx[k])
            reached = best
    return reached * spacing, float(np.sqrt(near.min()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="+")
    ap.add_argument("--route", default="maps/surf_src_cannonball.route.npz")
    ap.add_argument("--map", default="surf_src_cannonball")
    ap.add_argument("--corridor", type=float, default=1500.0,
                    help="how far off the line still counts as on-route")
    ap.add_argument("--pad", type=float, default=64.0)
    a = ap.parse_args()

    rp = Path(a.route)
    if not rp.exists():
        rp = ROOT / a.route
    pts, spacing = load_route(rp)
    end = json.loads((ROOT / "maps" / f"{a.map}.zones.json").read_text(
        encoding="utf-8"))["end"]
    lo, hi = np.asarray(end["mins"]), np.asarray(end["maxs"])

    files = []
    for pat in a.traj:
        files.extend(sorted(glob.glob(pat)) or
                     ([pat] if Path(pat).exists() else []))
    if not files:
        raise SystemExit("no trajectory files matched")

    for f in files:
        eps = episodes_from_traj(f)
        print(f"\n{Path(f).name}: {len(eps)} episodes  "
              f"(route {len(pts)} pts x {spacing:g}u = "
              f"{(len(pts) - 1) * spacing:,.0f}u)")
        prog, fins, dives = [], 0, 0
        for i, ep in enumerate(eps):
            xyz = ep[:, 1:4].astype(np.float32)
            fin = bool(np.any(np.all((xyz >= lo - a.pad) & (xyz <= hi + a.pad),
                                     axis=1)))
            below = bool(xyz[-1, 2] < lo[2] - 256.0)
            p, off = corridor_progress(xyz, pts, spacing, a.corridor)
            prog.append(p)
            fins += fin
            dives += below and not fin
            tag = "FINISH" if fin else ("dive-below" if below else "short")
            print(f"  ep{i}: {len(ep) / 100:6.1f}s  route {p:9,.0f}u "
                  f"({100 * p / max(1.0, (len(pts) - 1) * spacing):5.1f}%)  "
                  f"closest-approach {off:7.0f}u  end z {xyz[-1, 2]:8.0f}  {tag}")
        if prog:
            print(f"  -> corridor progress mean {np.mean(prog):,.0f}u  "
                  f"max {np.max(prog):,.0f}u  |  finishes {fins}/{len(eps)}  "
                  f"dives-below {dives}/{len(eps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
