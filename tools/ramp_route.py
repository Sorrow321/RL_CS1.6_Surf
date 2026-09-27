"""ramp_route.py - which RAMPS does a trajectory ride, in what order, at what speed? Maps a recorded
episode onto the ramps tools/ramps.py extracted (MEASUREMENT: with a human record it is a ruler,
never training data - CLAUDE.md section 0).

    python tools/ramp_route.py runs/research/ramps_<map>.npz <trajectory.jsonl> [--episode 0]

Contact rule (geometric): the standing hull (half-extents 16, 16, 36) touches a ramp sample p with
normal n when the origin sits at the hull's support distance from p's plane,
|n.(o - p) - s(n)| <= 8 u with s(n) = 16|n_x| + 16|n_y| + 36|n_z|, and the origin projects onto
the plane within 24 u of p. Consecutive contact ticks on one ramp form a visit; the route is the
visit sequence with entry / exit time and speed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

HULL = np.array([16.0, 16.0, 36.0])


def load_episode(path, ep):
    rows, k = [], 0
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("["):
            if k == ep:
                rows.append(json.loads(ln)[:7])
        elif ln.startswith('{"end'):
            if k == ep:
                break
            k += 1
    return np.asarray(rows, np.float64)


def contacts(pts, nrm, rid, traj):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    sup = np.abs(nrm) @ HULL
    out = np.full(len(traj), -1, np.int64)
    for t in range(len(traj)):
        o = traj[t, 1:4]
        idx = tree.query_ball_point(o, r=80.0)
        best, bd = -1, 1e9
        for i in idx:
            dn = float(nrm[i] @ (o - pts[i])) - sup[i]
            if abs(dn) > 8.0:
                continue
            lat = (o - pts[i]) - nrm[i] * float(nrm[i] @ (o - pts[i]))
            dl = float(np.linalg.norm(lat))
            if dl <= 24.0 and dl < bd:
                best, bd = int(rid[i]), dl
        out[t] = best
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ramps")
    ap.add_argument("traj")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--tick-ms", type=float, default=10.0)
    ap.add_argument("--min-ticks", type=int, default=3, help="shorter contacts are grazes")
    a = ap.parse_args(argv)
    z = np.load(a.ramps)
    pts, nrm, rid = z["points"].astype(np.float64), z["normals"].astype(np.float64), z["ramp_id"]
    keep = rid >= 0
    pts, nrm, rid = pts[keep], nrm[keep], rid[keep]
    dmean = z["d_mean"]
    traj = load_episode(a.traj, a.episode)
    c = contacts(pts, nrm, rid, traj)
    sp = np.linalg.norm(traj[:, 4:7], axis=1)
    dt = a.tick_ms / 1000.0
    visits, t = [], 0
    while t < len(c):
        if c[t] < 0:
            t += 1
            continue
        r, t0 = c[t], t
        while t < len(c) and c[t] == r:
            t += 1
        if t - t0 >= a.min_ticks:
            if visits and visits[-1][0] == r and t0 - visits[-1][2] < 10:
                visits[-1][2] = t                      # a short break on the same ramp
            else:
                visits.append([r, t0, t])
    print(f"ramp_route: {len(traj)} ticks ({len(traj) * dt:.1f} s), {int((c >= 0).sum())} ticks in "
          f"ramp contact; {len(visits)} ramp visits:")
    print("   ramp  potential(mean)   enter s  leave s   |v| in   |v| out")
    for r, t0, t1 in visits:
        print(f"   R{r:<4d} {dmean[r]:12,.0f}   {t0 * dt:7.2f}  {t1 * dt:7.2f}   {sp[t0]:6,.0f}   "
              f"{sp[min(t1, len(sp) - 1)]:6,.0f}")
    print("   sequence: " + " -> ".join(f"R{r}" for r, _a, _b in visits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
