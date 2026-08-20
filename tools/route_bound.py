"""route_bound.py - how fast could THIS route ever be run?

Before spending compute polishing a racing line, ask whether the line can
host the target time at all. This computes, from the exact GoldSrc
movement laws, three things for a recorded trajectory:

  1. RIGOROUS FLOOR - a hard lower bound on the time ANY policy (human,
     TAS or agent) could traverse the same polyline. Air acceleration is
     saturated at our settings (sv_airaccelerate 100, wishspeed 250,
     tau 0.01 -> accelspeed 250 >> the 30u air projection cap), so the
     per-tick impulse is mu = max(0, 30 - v.wishdir) and the best case is
     wishdir exactly perpendicular:
         |v'|^2 = |v|^2 + 900 - (v cos theta)^2  <=  |v|^2 + 900
     Speed-squared therefore grows by at most 900 per 10 ms tick, ever.
     Gravity does work 2*g*dz on descent; ramp contact (PM_ClipVelocity,
     v_c = v - (v.n)n) only ever REMOVES energy. So along the route
         v^2(s, n) <= v0^2 + 2*g*(z0 - z(s)) + 900*n
     and marching a particle at that ceiling gives the minimum ticks.
     Every term is an upper bound on speed, so the result is a true floor
     - and a loose one: it assumes zero energy lost into any ramp normal.

  2. PRACTICAL FLOOR - the same line flown with PERFECT strafe capture (a
     full 450/tick of specific energy) but the SAME fractional kinetic
     energy loss at each ramp contact the recording actually paid, i.e.
     "how fast is this line with flawless execution".

  3. Sensitivity - what the time becomes if ramp losses are cut by
     10-40%, i.e. how much better the CONTACT GEOMETRY has to get.

Caveat: a faster run flies different ballistic arcs, so "the same route"
is an idealization (a particle constrained to the curve). The seconds are
model-dependent; the ranking of the levers is not.

Usage: python tools/route_bound.py <traj.jsonl> [--ep N] [--target 68.0]
"""
import argparse
import json
import math

import numpy as np

TICK = 0.01
G = 800.0
AIR_CAP = 30.0                  # L: wishspeed projection cap in air
MAX_DV2 = AIR_CAP ** 2          # 900: max growth of |v|^2 per tick
MAX_DE = 0.5 * MAX_DV2          # 450: max growth of specific energy/tick
TELEPORT = 110.0


def load_episodes(path):
    eps, rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, dict) and "map" in row:
                rows = []
            elif isinstance(row, list):
                rows.append(row)
            elif isinstance(row, dict) and "end" in row:
                if rows:
                    eps.append(np.asarray(rows, dtype=np.float64))
                rows = []
    return eps


def timed_segment(ep):
    """Trim to the run clock: it starts on the opening cliff drop."""
    z0 = ep[0, 3]
    drop = np.flatnonzero(ep[:, 3] < z0 - 100.0)
    return ep[int(drop[0]):] if len(drop) else ep


def rigorous_floor(pos, v0, cap=4000.0 * math.sqrt(3.0)):
    seg = np.diff(pos, axis=0)
    slen = np.linalg.norm(seg, axis=1)
    slen = np.where(slen <= TELEPORT, slen, 0.0)   # never credit teleports
    cum = np.concatenate(([0.0], np.cumsum(slen)))
    total, z, z0 = cum[-1], pos[:, 2], pos[0, 2]
    s, n, i = 0.0, 0, 0
    while s < total and n < 200_000:
        while i + 1 < len(cum) - 1 and cum[i + 1] <= s:
            i += 1
        span = cum[i + 1] - cum[i]
        f = 0.0 if span <= 0 else (s - cum[i]) / span
        zz = z[i] + f * (z[i + 1] - z[i])
        v2 = v0 * v0 + 2.0 * G * (z0 - zz) + MAX_DV2 * n
        s += min(math.sqrt(max(0.0, v2)), cap) * TICK
        n += 1
    return n * TICK, total


def practical_floor(pos, e0, phi, loss_cut=0.0):
    """Perfect strafe capture, same fractional ramp losses (optionally
    reduced by loss_cut). Returns (total_seconds, per_tick_seconds)."""
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    ev, t, per = e0, 0.0, np.zeros(len(d))
    for i in range(len(d)):
        kin = max(1.0, ev - G * pos[i, 2])
        dt = d[i] / max(math.sqrt(2.0 * kin), 1.0)
        ev += MAX_DE * (dt / TICK)
        kin = max(1.0, ev - G * pos[i + 1, 2]) * (1.0 - phi[i] * (1.0 - loss_cut))
        ev = kin + G * pos[i + 1, 2]
        t += dt
        per[i] = dt
    return t, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--ep", type=int, default=None, help="1-based episode")
    ap.add_argument("--target", type=float, default=68.0, help="record, s")
    ap.add_argument("--segments", type=int, default=10)
    args = ap.parse_args()

    eps = load_episodes(args.traj)
    if args.ep is None:
        print(f"{args.traj}: {len(eps)} episodes")
        for k, e in enumerate(eps, 1):
            print(f"  ep {k:2d}: {len(e):5d} ticks  "
                  f"zdrop {e[0, 3] - e[:, 3].min():8,.0f}  "
                  f"peak {np.hypot(e[:, 4], e[:, 5]).max():6,.0f} u/s")
        return

    a = timed_segment(eps[args.ep - 1])
    p, v = a[:, 1:4], a[:, 4:7]
    sp2 = (v ** 2).sum(1)
    energy = 0.5 * sp2 + G * p[:, 2]        # specific mechanical energy
    de = np.diff(energy)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(d)))
    length = cum[-1]
    actual = len(d) * TICK

    loss = np.where(de < 0, -de, 0.0)       # destroyed into ramp normals
    gain = np.where(de > 0, de, 0.0)        # added by air acceleration
    kin = 0.5 * sp2
    phi = np.zeros(len(de))
    np.divide(loss, kin[:-1], out=phi, where=kin[:-1] > 1e-6)

    floor, _ = rigorous_floor(p, math.sqrt(sp2[0]))
    prac, per = practical_floor(p, energy[0], phi)
    grav = G * (p[0, 2] - p[-1, 2])

    print(f"\n{args.traj}  episode {args.ep}")
    print(f"  route length {length:>12,.0f} u   height drop "
          f"{p[0, 2] - p[:, 2].min():>10,.0f} u")
    print(f"  ACTUAL       {actual:>12.2f} s   mean {length / actual:,.0f} u/s"
          f"   peak {math.sqrt(sp2.max()):,.0f} u/s")
    print(f"\n  RIGOROUS FLOOR  (no ramp loss, perfect strafe)  {floor:>8.2f} s")
    print(f"  PRACTICAL FLOOR (same ramp loss, perfect strafe) {prac:>8.2f} s")
    print(f"  target                                           {args.target:>8.2f} s")
    print(f"  --> route {'CAN' if floor < args.target else 'CANNOT'} host the "
          f"target; this LINE {'can' if prac < args.target else 'cannot'} "
          f"even with flawless execution")

    print("\n  energy budget (specific, per unit mass)")
    print(f"    gravity supplied   {grav:>14,.0f}")
    print(f"    strafe supplied    {gain.sum():>14,.0f}   "
          f"({gain.sum() / (MAX_DE * len(de)) * 100:.0f}% of the 450/tick ceiling)")
    print(f"    destroyed at ramps {loss.sum():>14,.0f}   "
          f"({loss.sum() / (grav + gain.sum()) * 100:.0f}% of all supplied)")

    print(f"\n  per-segment ({args.segments} equal slices of route)")
    print("    seg  actual_s  perfect_s   gain_s    ramp_loss    z_drop  mean_spd")
    for k in range(args.segments):
        lo = k * length / args.segments
        hi = (k + 1) * length / args.segments
        m = (cum[:-1] >= lo) & (cum[:-1] < hi)
        n = int(np.count_nonzero(m))
        if not n:
            continue
        idx = np.flatnonzero(m)
        zdrop = p[idx[0], 2] - p[idx[-1] + 1, 2]
        print(f"    {k + 1:2d}  {n * TICK:8.2f}  {per[m].sum():9.2f}  "
              f"{n * TICK - per[m].sum():7.2f}  {loss[m].sum():11,.0f}  "
              f"{zdrop:8,.0f}  {d[m].sum() / (n * TICK):8,.0f}")

    print("\n  sensitivity to contact quality (perfect strafe +)")
    for cut in (0.0, 0.1, 0.2, 0.3, 0.4):
        t, _ = practical_floor(p, energy[0], phi, loss_cut=cut)
        mark = "  <-- target" if t <= args.target else ""
        print(f"    {int(cut * 100):3d}% less ramp loss -> {t:6.2f} s{mark}")


if __name__ == "__main__":
    main()
