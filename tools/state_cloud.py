#!/usr/bin/env python3
"""state_cloud.py - a cloud of synthetic start states around a window of the
agent's OWN line (champion-free), for a start-state exploration arm.

The record's single-touch finish happens 1,141 u sideways and 432 u higher
than where our line first touches ramp 1 (ledger 2026-09-05 10:30). Every
search we have proposes ACTIONS from our line's states and never gets
there. This tool proposes STATES instead: take the spine states of a tick
window (the approach into ramp 1), and jitter position, height, speed and
heading around them. PPO started from the cloud (as its respawn pool, via
--demo-file) has to finish from arrival states it never produced itself;
the ones that finish fast say which arrivals pay, and the trainer's
respawn reservoir keeps them.

    python tools/state_cloud.py runs/research/exitLONG2/spine_r8ep3.npy \
        --t0 62.0 --t1 65.5 --n 4096 --lat 600 --up 600 --speed 0.12 --yaw 20 \
        --out runs/research/exitLONG2/cloud_ramp1.npy

Lateral = the horizontal direction perpendicular to the state's velocity
(so the cloud spreads ACROSS the approach, whichever way it runs). Height
is world z. Speed scales |v| by (1 +- speed); yaw rotates v about z and
turns the view with it, so the state stays self-consistent.
"""
import argparse
import numpy as np

TICK_S = 7.666667 / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spine")
    ap.add_argument("--t0", type=float, required=True, help="window start, seconds from spawn")
    ap.add_argument("--t1", type=float, required=True, help="window end, seconds from spawn")
    ap.add_argument("--tick-s", type=float, default=TICK_S)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--lat", type=float, default=600.0, help="max lateral offset, u (uniform +-)")
    ap.add_argument("--up", type=float, default=600.0, help="max upward offset, u (uniform 0..up)")
    ap.add_argument("--down", type=float, default=100.0, help="max downward offset, u")
    ap.add_argument("--speed", type=float, default=0.12, help="speed scale range +- (fraction)")
    ap.add_argument("--yaw", type=float, default=20.0, help="heading rotation range +- deg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sp = np.load(a.spine, allow_pickle=False)
    ticks = sp["tick"].astype(np.int64) if "tick" in sp.dtype.names else np.arange(len(sp))
    lo, hi = int(a.t0 / a.tick_s), int(a.t1 / a.tick_s)
    idx = np.where((ticks >= lo) & (ticks <= hi))[0]
    if len(idx) == 0:
        raise SystemExit(f"no spine states in ticks {lo}..{hi} (spine ticks {ticks.min()}..{ticks.max()})")
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(idx, size=a.n, replace=True)
    cloud = sp[pick].copy()
    v = cloud["velocity"].astype(np.float64)
    o = cloud["origin"].astype(np.float64)
    # lateral unit vector = horizontal velocity rotated 90 deg
    hv = v.copy(); hv[:, 2] = 0.0
    hn = np.linalg.norm(hv, axis=1, keepdims=True); hn[hn < 1e-6] = 1.0
    lat = np.stack([-hv[:, 1], hv[:, 0], np.zeros(len(hv))], axis=1) / hn
    o += lat * rng.uniform(-a.lat, a.lat, size=(a.n, 1))
    o[:, 2] += rng.uniform(-a.down, a.up, size=a.n)
    # speed and heading
    s = rng.uniform(1.0 - a.speed, 1.0 + a.speed, size=(a.n, 1))
    dyaw = np.deg2rad(rng.uniform(-a.yaw, a.yaw, size=a.n))
    c, sn = np.cos(dyaw), np.sin(dyaw)
    vx, vy = v[:, 0].copy(), v[:, 1].copy()
    v[:, 0] = c * vx - sn * vy
    v[:, 1] = sn * vx + c * vy
    v *= s
    cloud["origin"] = o.astype(cloud["origin"].dtype)
    cloud["velocity"] = v.astype(cloud["velocity"].dtype)
    if "yaw" in cloud.dtype.names:
        cloud["yaw"] = (cloud["yaw"].astype(np.float64) + np.rad2deg(dyaw)).astype(cloud["yaw"].dtype)
    if "onground" in cloud.dtype.names:
        cloud["onground"] = 0            # every cloud state starts airborne
    np.save(a.out, cloud)
    print(f"{a.out}: {a.n} states from {len(idx)} spine states (ticks {lo}..{hi}, "
          f"{a.t0:.1f}-{a.t1:.1f} s); lateral +-{a.lat:.0f} u, height -{a.down:.0f}..+{a.up:.0f} u, "
          f"speed +-{a.speed*100:.0f}%, yaw +-{a.yaw:.0f} deg; "
          f"z range {o[:, 2].min():.0f}..{o[:, 2].max():.0f}, |v| {np.linalg.norm(v, axis=1).min():.0f}..{np.linalg.norm(v, axis=1).max():.0f}")


if __name__ == "__main__":
    main()
