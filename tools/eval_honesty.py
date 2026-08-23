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


def corridor_progress_ordered(xyz, pts, spacing, corridor, window):
    """-> (units along the route reached, furthest lateral miss at the stop).

    The same quantity as :func:`corridor_progress`, but with the projection
    restricted to a LOCAL window around the previous anchor instead of a
    global argmin - i.e. the rule ``surfgym.route.ArcProgress`` pays under
    ``--race-arc``, so the metric and the reward cannot disagree.

    Why this exists (xARC, round 19): ``corridor_progress`` takes the nearest
    vertex over the WHOLE route at every sample, so wherever the route
    approaches itself the credited index can jump. Measured: two champion
    episodes that died at 87,355 u are credited 133,760 u, and on the final
    descent an episode that leaves the line at 209,664 u and FALLS is
    credited 220,800 u because the route's own bowl passes within ~1,100 u of
    the falling body. Both are the "an off-route fall claims a later stretch"
    failure this file was written to prevent, surviving in the one place the
    frontier now sits.

    Default OFF (``--order-only 0``) so every number already in
    docs/research-results.md reproduces exactly; pass ``--order-only 16`` to
    score the honest way.
    """
    from surfgym.route import ArcProgress
    ap = ArcProgress(np.asarray(pts, np.float64), spacing,
                     corridor=corridor, window=window)
    p = np.asarray(xyz, np.float64)
    ap.reset(p[:1])
    best = float(ap.arc[0])
    for k in range(1, len(p)):
        ap.advance(p[k:k + 1])
        best = max(best, float(ap.arc[0]))
    _arc, off = ap.locate(p)
    return best, float(off.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="+")
    ap.add_argument("--route", default="maps/surf_src_cannonball.route.npz")
    ap.add_argument("--map", default="surf_src_cannonball")
    ap.add_argument("--corridor", type=float, default=1500.0,
                    help="how far off the line still counts as on-route")
    ap.add_argument("--pad", type=float, default=64.0)
    ap.add_argument("--order-only", type=int, default=0, metavar="WINDOW",
                    help="also score with the local-window rule the --race-arc "
                         "reward uses (0 = off, 16 = +/-16 vertices). The "
                         "default global argmin can credit a fall with a "
                         "later stretch wherever the route approaches "
                         "itself; this cannot")
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
        prog, ord_prog, fins, dives = [], [], 0, 0
        for i, ep in enumerate(eps):
            xyz = ep[:, 1:4].astype(np.float32)
            fin = bool(np.any(np.all((xyz >= lo - a.pad) & (xyz <= hi + a.pad),
                                     axis=1)))
            below = bool(xyz[-1, 2] < lo[2] - 256.0)
            p, off = corridor_progress(xyz, pts, spacing, a.corridor)
            prog.append(p)
            extra = ""
            if a.order_only > 0:
                q, _ = corridor_progress_ordered(xyz, pts, spacing, a.corridor,
                                                 a.order_only)
                ord_prog.append(q)
                extra = f"  order-only {q:9,.0f}u"
            fins += fin
            dives += below and not fin
            tag = "FINISH" if fin else ("dive-below" if below else "short")
            print(f"  ep{i}: {len(ep) / 100:6.1f}s  route {p:9,.0f}u "
                  f"({100 * p / max(1.0, (len(pts) - 1) * spacing):5.1f}%)  "
                  f"closest-approach {off:7.0f}u  end z {xyz[-1, 2]:8.0f}"
                  f"{extra}  {tag}")
        if prog:
            print(f"  -> corridor progress mean {np.mean(prog):,.0f}u  "
                  f"max {np.max(prog):,.0f}u  |  finishes {fins}/{len(eps)}  "
                  f"dives-below {dives}/{len(eps)}")
            if ord_prog:
                print(f"  -> ORDER-ONLY      mean {np.mean(ord_prog):,.0f}u  "
                      f"max {np.max(ord_prog):,.0f}u  |  past 205,440u "
                      f"{sum(1 for q in ord_prog if q > 205440)}/{len(eps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
