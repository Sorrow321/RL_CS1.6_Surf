#!/usr/bin/env python3
"""wall_profile.py - why the run dies where it dies, in numbers.

`eval_honesty.py` says HOW FAR an episode got along the reference route.
This says WHAT WENT WRONG at the point it stopped: the speed and the
off-line error as a function of route position, for the agent and for the
champion line side by side.

The finding it was written for (xROUTE, 2026-08-22): the stuck policy is
not slow. At route vertex 1600 it is doing 2,714-2,762 u/s against the
champion's 2,926 - six percent - but it is 2,774-2,908 units OFF the line
and ~1,150 units too low, where the champion is 106 units off. It then
free-falls 2,400 units in 1.5 s while LOSING speed, instead of riding the
final descent down and accelerating to 3,728 u/s as the champion does. The
error is geometric and it starts building 40 vertices (5,000 u) earlier,
where the agent is already 174-330 units off against the champion's 71-114.

    python tools/wall_profile.py --route maps/surf_src_cannonball.route.npz \\
        --champion runs/sISV_par2/traj_8454144000.jsonl \\
        runs/research/xROUTE/traj_*.jsonl

Off-line error at the approach vertices is the diagnostic an arm that is
supposed to help HERE should move, and it moves long before
race/eval_progress does.
"""
import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from surfgym.route import episodes_from_traj  # noqa: E402


def trace(ep, pts):
    xyz = ep[:, 1:4].astype(np.float32)
    vel = ep[:, 4:7].astype(np.float32)
    d2 = ((xyz[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    return (d2.argmin(1), np.sqrt(d2.min(1)),
            np.linalg.norm(vel[:, :2], axis=1), xyz)


def band(idx, near, spd, xyz, v, half=2):
    m = (idx >= v - half) & (idx <= v + half)
    if not m.any():
        return None
    return float(spd[m].mean()), float(xyz[m][:, 2].mean()), float(near[m].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="+")
    ap.add_argument("--route", default="maps/surf_src_cannonball.route.npz")
    ap.add_argument("--champion", default=None,
                    help="trajectory to use as the reference profile")
    ap.add_argument("--from-vertex", type=int, default=1540)
    ap.add_argument("--to-vertex", type=int, default=1700)
    ap.add_argument("--step", type=int, default=20)
    a = ap.parse_args()

    rp = Path(a.route)
    if not rp.exists():
        rp = ROOT / a.route
    z = np.load(rp)
    pts = np.asarray(z["route"], np.float32)
    sp = float(z["spacing"]) if "spacing" in z.files else 128.0
    vs = list(range(a.from_vertex, min(a.to_vertex, len(pts)), a.step))

    ref = {}
    if a.champion:
        eps = episodes_from_traj(a.champion)
        tr = [trace(e, pts) for e in eps]
        bestt = max(tr, key=lambda t: t[0].max())
        print(f"CHAMPION {Path(a.champion).name} (furthest of {len(eps)} eps)")
        print(f"{'vertex':>8} {'units':>10} {'speed':>8} {'z':>9} {'off-line':>9}")
        for v in vs:
            b = band(*bestt, v)
            if b:
                ref[v] = b
                print(f"{v:8d} {v * sp:10,.0f} {b[0]:8.0f} {b[1]:9.0f} {b[2]:9.0f}")

    files = []
    for pat in a.traj:
        files.extend(sorted(glob.glob(pat)) or
                     ([pat] if Path(pat).exists() else []))
    for f in files:
        eps = episodes_from_traj(f)
        tr = [trace(e, pts) for e in eps]
        tops = [int(t[0].max()) for t in tr]
        order = np.argsort(tops)[::-1][:3]        # the three that got furthest
        print(f"\n{Path(f).name}: {len(eps)} eps, furthest vertices "
              f"{sorted(tops, reverse=True)}")
        agg = {}
        for k in order:
            idx, near, spd, xyz = tr[k]
            for v in vs:
                b = band(idx, near, spd, xyz, v)
                if b and v <= tops[k]:
                    agg.setdefault(v, []).append(b)
        print(f"{'vertex':>8} {'speed':>8} {'z':>9} {'off-line':>9}   vs champion")
        for v in vs:
            if v not in agg:
                continue
            s = float(np.mean([b[0] for b in agg[v]]))
            zz = float(np.mean([b[1] for b in agg[v]]))
            o = float(np.mean([b[2] for b in agg[v]]))
            note = ""
            if v in ref:
                note = (f"   dspeed {s - ref[v][0]:+6.0f}  dz {zz - ref[v][1]:+7.0f}"
                        f"  doff {o - ref[v][2]:+8.0f}")
            print(f"{v:8d} {s:8.0f} {zz:9.0f} {o:9.0f}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
