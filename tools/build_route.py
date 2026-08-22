#!/usr/bin/env python3
"""build_route.py - extract a reference route line from recorded trajectories.

The lookahead fan (`python/surfgym/route.py`, `train_fast.py --route`) needs a
polyline through the map. Linesight's rule is the one to follow: the reference
line "does not need to be fast... usually the centerline", and as records fall
it is re-extracted from the AI's own best runs. So this tool takes any set of
recorded episodes, keeps the ones that actually reach the end zone, picks the
FASTEST of those, resamples it at constant arc length and writes a .npz.

    python tools/build_route.py --out maps/cannonball.route.npz \\
        runs/sISV_par2/traj_8454144000.jsonl

    # every finisher in a directory, fastest wins
    python tools/build_route.py --out maps/cannonball.route.npz \\
        "runs/research/xSC2c/traj_*.jsonl"

What this does and does not inject: a route is a PATH through the map. It
carries no actions, no timing, no control, and an agent given one still has to
discover how to fly it. Every system in the survey has the course available a
priori (Sophy's track edges, Linesight's virtual checkpoints, Swift's gate
positions) - a human racer walks the track before driving it. It is still
route knowledge, so a run using it is not honest-perception in the --gps
sense, and that belongs in the ledger entry.

--selftest runs the geometry on a synthetic L-shaped route with no map, no
GPU and no trajectory files.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from surfgym.route import (DEFAULT_SPACING, RouteLine,  # noqa: E402
                           episodes_from_traj, resample_polyline)


def inside(p, mins, maxs, pad=0.0):
    return bool(np.all(p >= np.asarray(mins) - pad)
                and np.all(p <= np.asarray(maxs) + pad))


def pick_route(files, end_mins, end_maxs, tick_ms=10.0, pad=64.0):
    """-> (xyz of the fastest finishing episode, seconds, how many finished).

    "Finished" = some sample lands in the end zone (padded, because a
    recording samples every tick and the box is a 1-unit curtain in y; the
    crossing tick can sit just outside it). The episode is TRIMMED at that
    sample so a route never includes the post-finish coast.
    """
    best, best_s, n_fin, n_ep = None, float("inf"), 0, 0
    for f in files:
        for ep in episodes_from_traj(f):
            n_ep += 1
            xyz = ep[:, 1:4]
            hit = [i for i in range(len(xyz))
                   if inside(xyz[i], end_mins, end_maxs, pad)]
            if not hit:
                continue
            n_fin += 1
            cut = hit[0] + 1
            secs = float(ep[cut - 1, 0] - ep[0, 0]) * tick_ms * 1e-3
            if secs < best_s:
                best, best_s = xyz[:cut], secs
    return best, best_s, n_fin, n_ep


def selftest():
    """Geometry only: an L-shaped route, an agent on it, an agent off it."""
    import torch
    leg = np.linspace(0, 4000, 200)
    line = np.concatenate([
        np.stack([leg, np.zeros_like(leg), np.zeros_like(leg)], 1),
        np.stack([np.full_like(leg, 4000.0), leg, np.zeros_like(leg)], 1)])
    pts, total = resample_polyline(line, 128.0)
    assert abs(total - 8000.0) < 1.0, total
    r = RouteLine(pts, 128.0, offsets=(1.0, 2.0), speed_floor=500.0)
    assert r.n_features == 9, r.n_features

    # (a) on the first leg, facing +x at 1000 u/s: the route ahead is dead
    # ahead, so forward reads ~1 and lateral ~0 at every horizon
    o = torch.tensor([[1000.0, 0.0, 0.0]])
    f = r.features(o, torch.tensor([0.0]), torch.tensor([1000.0])).numpy()[0]
    assert abs(f[3] - 1.0) < 0.05 and abs(f[4]) < 0.05, f[:6]
    assert abs(f[6] - 1.0) < 0.05 and abs(f[7]) < 0.05, f[6:9]

    # (b) same place, YAWED 90 deg left: the route is now to the agent's
    # RIGHT, so lateral goes strongly negative and forward to ~0
    f90 = r.features(o, torch.tensor([90.0]), torch.tensor([1000.0])).numpy()[0]
    assert abs(f90[3]) < 0.05 and f90[4] < -0.9, f90[:6]

    # (c) past the corner the route turns +y: at 2 s / 2000 u ahead from
    # x=3000 the fan must report a LEFT turn (positive lateral)
    o2 = torch.tensor([[3000.0, 0.0, 0.0]])
    f2 = r.features(o2, torch.tensor([0.0]), torch.tensor([1000.0])).numpy()[0]
    assert f2[7] > 0.3, f2[6:9]

    # (d) standing still does not collapse the fan (speed_floor)
    f0 = r.features(o, torch.tensor([0.0]), torch.tensor([0.0])).numpy()[0]
    assert abs(f0[3] - 1.0) < 0.05, f0[:6]

    # (e) off-route laterally: the nearest-point block reports the offset
    o3 = torch.tensor([[1000.0, -400.0, 0.0]])
    f3 = r.features(o3, torch.tensor([0.0]), torch.tensor([1000.0])).numpy()[0]
    assert f3[1] > 0.7, f3[:3]

    # (f) clamped and finite everywhere, including far off the map
    o4 = torch.tensor([[1e6, -1e6, 1e5]])
    f4 = r.features(o4, torch.tensor([33.0]), torch.tensor([9e9])).numpy()
    assert np.isfinite(f4).all() and np.abs(f4).max() <= 4.0 + 1e-6
    print("selftest OK: 6 geometry checks")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="*", help="trajectory .jsonl files or globs")
    ap.add_argument("--out", default=None, help="output .npz")
    ap.add_argument("--map", default="surf_src_cannonball")
    ap.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    ap.add_argument("--pad", type=float, default=64.0,
                    help="end-zone tolerance in map units")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if not a.traj or not a.out:
        ap.error("need trajectory files and --out (or --selftest)")

    files = []
    for pat in a.traj:
        files.extend(sorted(glob.glob(pat)) or ([pat] if Path(pat).exists() else []))
    if not files:
        raise SystemExit("no trajectory files matched")

    zp = ROOT / "maps" / f"{a.map}.zones.json"
    end = json.loads(zp.read_text(encoding="utf-8"))["end"]
    xyz, secs, n_fin, n_ep = pick_route(files, end["mins"], end["maxs"],
                                        pad=a.pad)
    if xyz is None:
        raise SystemExit(f"none of the {n_ep} episodes in {len(files)} file(s) "
                         f"reached the end zone {end['mins']}..{end['maxs']} "
                         f"(pad {a.pad:g}u) - a route must reach the finish")
    pts, total = resample_polyline(xyz, a.spacing)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, route=pts.astype(np.float32),
                        spacing=np.float32(a.spacing),
                        seconds=np.float32(secs), map=a.map,
                        source=";".join(Path(f).name for f in files[:8]))
    print(f"{n_fin}/{n_ep} episodes finished; fastest {secs:.2f}s")
    print(f"route: {len(pts)} points @ {a.spacing:g}u, path {total:,.0f}u")
    print(f"  start {pts[0].round(1).tolist()}  end {pts[-1].round(1).tolist()}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
