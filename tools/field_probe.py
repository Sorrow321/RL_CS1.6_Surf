#!/usr/bin/env python3
"""field_probe.py - does the shaping field ask for a path the PHYSICS can fly?

The geodesic goal field is a BFS over FREE VOXELS. It knows nothing about
gravity, velocity or whether a surface is ridable, so its "shortest path"
is free to glide laterally across open air. CLAUDE.md already records that
exact deception on cannonball ("a straight ~8,700 u level glide through open
air with 3,584 u of floor clearance ... the BFS believes the player can fly
laterally across a void"), and records petrus's wall as a SPEED GATE at
~1,550 u/s without ever tying the two together.

This tool ties them together. It walks a path - a baked fieldroute, a greedy
descent on the field, or a RECORDED trajectory - and at every sample asks
three questions the field never asked:

  1. SUPPORT   is there a surfable surface (0.1 <= |n_z| <= 0.7, the band
               that is too steep to stand on and shallow enough to ride)
               within reach? Without one the player is a projectile.
  2. CLIMB     does the path demand dz > 0? Unsupported, that is impossible
               at any speed - only banked vertical velocity can pay for it.
  3. SPEED     an unsupported run of horizontal length L that drops h is a
               projectile problem with TWO bounds (see ballistic_speed):
                 v_min  = sqrt(g (sqrt(L^2+h^2) - h))  - optimal launch
                          angle, the floor no policy can beat;
                 v_flat = L sqrt(g / 2h)               - leaving the ramp
                          horizontally, no upward component.
               Both come from the map, not from any policy. A real ramp
               exit lands between them, which is why quoting both bounds
               brackets the gate instead of asserting a single number.

The headline output is the required-speed profile: the map's own answer to
"how fast must I be here", next to the field's claim that the place is free.

Validated against the one case CLAUDE.md already documents by hand - the
cannonball descent from route vertex 1600, recorded there as "191 level
steps, 5 down, 0 up, zero climb ... a straight ~8,700 u level glide through
open air". This tool reports that path independently as arc 8,868 u, 100%
unsupported, zero climb, one run of 8,817 u horizontal against 160 u of
drop: v_flat 13,941 u/s against a --maxvel of 4,000. Same defect, with a
number on it.

Usage
    python tools/field_probe.py --map maps/surf_petrus_lite.bsp \
        --route maps/surf_petrus_lite.fieldroute.npz
    python tools/field_probe.py --map maps/surf_petrus_lite.bsp \
        --from -3839 3066 660          # greedy descent on the field itself
    python tools/field_probe.py --map maps/surf_petrus_lite.bsp \
        --traj runs/<run>/traj_*.jsonl # where a real episode died, and why

Every cache is read from the .bsp's OWN directory, so always pass an
absolute path into the main checkout (CLAUDE.md's worktree trap).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.route import episodes_from_traj  # noqa: E402

GRAVITY = 800.0          # sv_gravity, src/surfcore.h:67
MAXVEL = 4000.0          # --maxvel, the trainer's speed ceiling
GROUND_NZ = 0.7          # src/pm.c:252 - above this you STAND, not surf
WALL_NZ = 0.1            # below this it is a wall; it deflects, it does not carry
SENTINEL_FRAC = 0.995    # d >= reach_max * this  ->  unreachable


# ---------------------------------------------------------------- caches

class Field:
    """The baked grids for one map, on one cell, with the index transform.

    Every grid is [z, y, x] (verified against the goal field: the end-zone
    centre reads d = 0 and the start-zone centre reads d0 only under that
    order). World -> index is floor((p - mins) / cell) per axis, with mins
    given in (x, y, z) order.
    """

    def __init__(self, bsp: Path, cell: int = 32):
        self.bsp, self.cell = Path(bsp), float(cell)
        stem = self.bsp.with_suffix("")

        g = np.load(f"{stem}.goal_{cell}.npz")
        self.d = g["grid"].astype(np.float32) * float(g["quant"])
        self.mins = np.asarray(g["mins"], dtype=np.float64)
        self.reach_max = float(g["reach_max"])
        self.unreach = self.reach_max * SENTINEL_FRAC

        self.occ = np.load(f"{stem}.occ_{cell}.npz")["occ"].astype(bool)
        try:
            self.sdf = np.load(f"{stem}.sdf_{cell}.npz")["sdf"].astype(np.float32)
        except FileNotFoundError:
            self.sdf = None
        try:
            nz = np.load(f"{stem}.surfnz_{cell}.npz")["nz"]
            self.nz = np.abs(nz.astype(np.int16)) / 127.0
        except FileNotFoundError:
            self.nz = None

        self.shape = self.d.shape
        # Two different questions, two different masks - conflating them is
        # how the spawn platform reads as "no support":
        #   holds   anything that can carry the player at all. Flat ground
        #           (|n_z| > 0.7) you STAND on; a ramp (0.1..0.7) you RIDE.
        #           This is the set the BFS should have been walking.
        #   surfy   the ramp band alone. Where `holds` is true but `surfy`
        #           is false the player must be WALKING, which on a surf map
        #           means he is not making race speed.
        if self.nz is not None:
            self.holds = self.nz >= WALL_NZ
            self.surfy = (self.nz >= WALL_NZ) & (self.nz <= GROUND_NZ)
        else:
            self.holds = self.surfy = None

    # -- index helpers -----------------------------------------------
    def idx(self, p):
        p = np.asarray(p, dtype=np.float64)
        i = np.floor((p - self.mins) / self.cell).astype(np.int64)
        return i[..., ::-1]          # (x,y,z) -> (z,y,x)

    def inside(self, zyx):
        zyx = np.asarray(zyx)
        return bool(np.all(zyx >= 0) and np.all(zyx < np.array(self.shape)))

    def at(self, grid, p, default=np.nan):
        i = self.idx(p)
        if not self.inside(i):
            return default
        return float(grid[tuple(i)])

    def dist(self, p):
        return self.at(self.d, p, default=np.nan)

    def solid(self, p) -> bool:
        i = self.idx(p)
        return True if not self.inside(i) else bool(self.occ[tuple(i)])

    # -- the three questions -----------------------------------------
    def support(self, p, reach=192.0, surf_only=False):
        """Distance to the nearest supporting voxel centre, or inf.

        `reach` is a ball radius in world units. 192 u is 6 cells: a player
        is 72 u tall and a ramp he can reach in well under a second.
        `surf_only` restricts the mask to the ridable band, excluding the
        flat ground you can only walk on.
        """
        mask = self.surfy if surf_only else self.holds
        if mask is None:
            return float("nan")
        c, r = self.cell, int(np.ceil(reach / self.cell))
        i = self.idx(p)
        lo = np.maximum(i - r, 0)
        hi = np.minimum(i + r + 1, np.array(self.shape))
        if np.any(lo >= hi):
            return float("inf")
        sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if not sub.any():
            return float("inf")
        zz, yy, xx = np.nonzero(sub)
        # voxel centre in world, back in (x, y, z)
        cx = self.mins[0] + (lo[2] + xx + 0.5) * c
        cy = self.mins[1] + (lo[1] + yy + 0.5) * c
        cz = self.mins[2] + (lo[0] + zz + 0.5) * c
        dd = np.sqrt((cx - p[0]) ** 2 + (cy - p[1]) ** 2 + (cz - p[2]) ** 2)
        m = float(dd.min())
        return m if m <= reach else float("inf")

    def floor_drop(self, p, max_drop=4096.0):
        """Distance straight DOWN to the first solid voxel, or max_drop."""
        c = self.cell
        for k in range(1, int(max_drop / c) + 1):
            q = np.array([p[0], p[1], p[2] - k * c])
            i = self.idx(q)
            if not self.inside(i):
                return float(max_drop)
            if self.occ[tuple(i)]:
                return float(k * c)
        return float(max_drop)


def ballistic_speed(L: float, drop: float):
    """Two honest lower bounds on the speed an unsupported gap costs.

    Returns (v_min, v_flat).

    v_min  - the ABSOLUTE geometric floor: the smallest launch speed, at
             ANY angle, that reaches a target L away and `drop` below.
             Classic minimum-speed projectile result,
                 v_min^2 = g * (sqrt(L^2 + h^2) - h)   with h = drop.
             h = 0 collapses to the 45-degree v = sqrt(g L); L = 0 gives 0.
             No policy crosses the gap slower than this, whatever it does.

    v_flat - the NO-LAUNCH floor: the speed needed entering HORIZONTALLY
             (v_z = 0), i.e. leaving a ramp flat and simply falling,
                 t = sqrt(2h/g),  v = L / t.
             A level gap (h = 0) is unreachable this way, hence inf.

    A real ramp exit lies between the two: it leaves with some upward
    component, rarely the optimal one. Quoting both bounds is the honest
    way to say what the field is demanding.
    """
    L = float(L)
    h = float(drop)
    v_min = float(np.sqrt(max(0.0, GRAVITY * (np.hypot(L, h) - h))))
    v_flat = float("inf") if h <= 1e-6 else L * float(np.sqrt(GRAVITY / (2.0 * h)))
    return v_min, v_flat


# ---------------------------------------------------------------- paths

def greedy_descent(F: Field, start, max_steps=4000):
    """Walk the field the way the BFS believes it flows: 26-connected argmin.

    This is what the shaping reward is actually pointing at, so it is the
    path whose flyability matters.
    """
    i = F.idx(start)
    if not F.inside(i):
        raise SystemExit(f"start {start} is outside the grid")
    offs = np.array([(dz, dy, dx)
                     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if (dz, dy, dx) != (0, 0, 0)])
    pts, seen = [], set()
    for _ in range(max_steps):
        pts.append(F.mins + (i[::-1] + 0.5) * F.cell)
        if F.d[tuple(i)] <= F.cell or tuple(i) in seen:
            break
        seen.add(tuple(i))
        cand = i + offs
        ok = np.all((cand >= 0) & (cand < np.array(F.shape)), axis=1)
        cand = cand[ok]
        if not len(cand):
            break
        vals = F.d[cand[:, 0], cand[:, 1], cand[:, 2]]
        free = ~F.occ[cand[:, 0], cand[:, 1], cand[:, 2]]
        vals = np.where(free, vals, np.inf)
        j = int(np.argmin(vals))
        if not np.isfinite(vals[j]) or vals[j] >= F.d[tuple(i)]:
            break
        i = cand[j]
    return np.array(pts)


def traj_episodes(path):
    """Per-episode (positions, horizontal speed) from a record_rollout jsonl.

    Reuses surfgym.route.episodes_from_traj rather than reparsing: rows are
    ``[tick, x, y, z, vx, vy, vz, yaw, ...]`` with a dict header per episode
    and a footer, and hand-rolling that split is how a recorder without
    headers silently becomes one giant episode.
    """
    out = []
    for ep in episodes_from_traj(path):
        if len(ep) < 2:
            continue
        xyz = ep[:, 1:4].astype(np.float64)
        spd = (np.linalg.norm(ep[:, 4:6], axis=1).astype(np.float64)
               if ep.shape[1] >= 7 else np.full(len(ep), np.nan))
        out.append((xyz, spd))
    return out


# ---------------------------------------------------------------- report

def analyse(F: Field, pts, reach, speeds=None, label="path", profile=True):
    n = len(pts)
    d = np.array([F.dist(p) for p in pts])
    sup = np.array([F.support(p, reach) for p in pts])
    sfy = np.array([F.support(p, reach, surf_only=True) for p in pts])
    flo = np.array([F.floor_drop(p) for p in pts])
    sdf = np.array([F.at(F.sdf, p) if F.sdf is not None else np.nan for p in pts])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    horiz = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    dz = np.diff(pts[:, 2])
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    # The gap analysis keys on the absence of a RIDABLE RAMP, not on the
    # absence of all geometry. Flat ground within reach still means the
    # player cannot carry race speed through here - he lands and walks -
    # so for a race it is a gap exactly as an empty void is.
    freefall = ~np.isfinite(sup)      # nothing at all within reach: a void
    unsup = ~np.isfinite(sfy)         # no ramp to ride: the RACE gap

    print(f"\n=== {label}: {n} samples, arc {arc[-1]:,.0f} u, "
          f"d {d[0]:,.0f} -> {d[-1]:,.0f} u  (cell {F.cell:g}, "
          f"support reach {reach:g} u)")
    reachable = d < F.unreach
    if not reachable.all():
        print(f"  !! {int((~reachable).sum())} samples sit in UNREACHABLE "
              f"voxels (d >= {F.unreach:,.0f}) - the field has no path from there")

    nosurf = unsup
    print(f"  TRUE VOID (nothing within reach): {int(freefall.sum())}/{n} "
          f"({100.0 * freefall.mean():.1f}%)  "
          f"[nothing at all within {reach:g} u - the player is a projectile]")
    print(f"  no RIDABLE ramp:     {int(nosurf.sum())}/{n} "
          f"({100.0 * nosurf.mean():.1f}%)  "
          f"[|nz| in 0.1..0.7 absent; the rest is flat ground you can only "
          f"walk on, which on a surf map is not race speed]")
    climb = dz > 0
    if climb.any():
        print(f"  segments demanding a CLIMB: {int(climb.sum())}/{n-1}, "
              f"total +{dz[climb].sum():,.0f} u, worst +{dz.max():,.0f} u")
    else:
        print(f"  segments demanding a CLIMB: none")

    # --- contiguous unsupported RUNS: the flyability question ---
    runs, i = [], 0
    while i < n:
        if not unsup[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and unsup[j + 1]:
            j += 1
        if j > i:
            L = float(horiz[i:j].sum())
            drop = float(pts[i, 2] - pts[j, 2])
            vmin, vflat = ballistic_speed(L, drop)
            runs.append((i, j, L, drop, vmin, vflat,
                         float(arc[i]), float(d[i])))
        i = j + 1

    if runs:
        runs.sort(key=lambda r: -r[4])
        print(f"\n  UNSUPPORTED RUNS (projectile arcs the field asks for), "
              f"worst first:")
        print(f"    {'arc_u':>9} {'d_u':>9} {'%d0':>6} {'horiz_u':>8} "
              f"{'drop_u':>7} {'v_min':>7} {'v_flat':>8}  verdict")
        for (i, j, L, drop, vmin, vflat, a, dd) in runs[:12]:
            pct = 100.0 * (1.0 - dd / d[0]) if d[0] else float("nan")
            fs = "inf" if not np.isfinite(vflat) else f"{vflat:,.0f}"
            if vmin >= MAXVEL:
                verdict = f"UNFLYABLE - v_min exceeds maxvel {MAXVEL:,.0f}"
            elif vmin >= 1500:
                verdict = "hard - v_min alone clears the petrus gate"
            elif not np.isfinite(vflat):
                verdict = "LEVEL gap - needs an upward launch, not just speed"
            elif vflat >= 1500:
                verdict = "gate-like: a flat ramp exit costs more than the gate"
            else:
                verdict = "flyable"
            print(f"    {a:9,.0f} {dd:9,.0f} {pct:6.1f} {L:8,.0f} "
                  f"{drop:7,.0f} {vmin:7,.0f} {fs:>8}  {verdict}")
        print(f"\n  ==> ABSOLUTE geometric speed floor on this path: "
              f"{max(r[4] for r in runs):,.0f} u/s (optimal launch angle, "
              f"the bound no policy can beat)")
        fin = [r[5] for r in runs if np.isfinite(r[5])]
        if fin:
            print(f"  ==> NO-LAUNCH floor (leaving a ramp horizontally): "
                  f"{max(fin):,.0f} u/s")
        nlev = sum(1 for r in runs if not np.isfinite(r[5]))
        if nlev:
            print(f"  ==> {nlev} run(s) are LEVEL or climbing: no flat exit "
                  f"crosses them at any speed - they need banked VERTICAL "
                  f"velocity, which is a different skill from going fast")
    else:
        print("\n  no unsupported runs: every sample has a surfable surface "
              "in reach")

    # --- WHERE along the map, in the %-of-d0 the petrus ledger uses ---
    if profile and d[0] > 0:
        print(f"\n  PROFILE by progress (the %-of-d0 axis every petrus arm "
              f"was reported on):")
        print(f"    {'%d0 band':>10} {'unsup%':>7} {'climb_u':>8} "
              f"{'worst v_flat':>12} {'min floor':>10}")
        pctp = 100.0 * (1.0 - d / d[0])
        edges = np.arange(0, 101, 5)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pctp >= lo) & (pctp < hi)
            if not m.any():
                continue
            mseg = m[:-1]                      # segments starting in the band
            up = dz[mseg & (dz > 0)]
            cl = float(up.sum()) if up.size else 0.0
            vf = [r[5] for r in runs if lo <= 100.0 * (1 - r[7] / d[0]) < hi
                  and np.isfinite(r[5])]
            worst = f"{max(vf):,.0f}" if vf else "-"
            print(f"    {lo:4.0f}-{hi:<5.0f} {100.0 * unsup[m].mean():7.1f}"
                  f" {cl:8,.0f} {worst:>12} {np.nanmin(flo[m]):10,.0f}")

    # --- where a recorded episode actually stopped ---
    if speeds is not None and len(speeds) == n:
        k = int(np.nanargmin(d))
        print(f"\n  recorded episode: best d = {d[k]:,.0f} u at sample {k} "
              f"(arc {arc[k]:,.0f} u), pos "
              f"({pts[k,0]:,.0f}, {pts[k,1]:,.0f}, {pts[k,2]:,.0f})")
        w = slice(max(0, k - 25), min(n, k + 5))
        print(f"    speed over the last 0.25 s before it: "
              f"{np.nanmin(speeds[w]):,.0f} - {np.nanmax(speeds[w]):,.0f} u/s")
        print(f"    support there: "
              + ("NONE in reach" if unsup[k] else f"{sup[k]:,.0f} u")
              + f", floor {flo[k]:,.0f} u below, sdf {sdf[k]:,.0f} u")

    return dict(d=d, sup=sup, flo=flo, sdf=sdf, arc=arc, runs=runs, unsup=unsup)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="absolute path to the .bsp")
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--route", help="a baked .fieldroute.npz / .route.npz")
    ap.add_argument("--from", dest="start", nargs=3, type=float,
                    metavar=("X", "Y", "Z"),
                    help="greedy-descend the field from here")
    ap.add_argument("--traj", nargs="*", help="recorded traj_*.jsonl")
    ap.add_argument("--reach", type=float, default=192.0,
                    help="support search radius, world units (default 192)")
    a = ap.parse_args()

    F = Field(a.map, a.cell)
    print(f"map {Path(a.map).stem}  grid {F.shape} [z,y,x]  cell {F.cell:g}  "
          f"mins {F.mins}  reach_max {F.reach_max:,.0f}")
    if F.holds is None:
        print("!! no surfnz cache - support analysis unavailable")

    did = False
    if a.route:
        z = np.load(a.route)
        pts = z["route"].astype(np.float64)
        analyse(F, pts, a.reach, label=f"fieldroute {Path(a.route).name}")
        did = True
    if a.start:
        pts = greedy_descent(F, np.array(a.start, dtype=np.float64))
        analyse(F, pts, a.reach,
                label=f"greedy field descent from {tuple(a.start)}")
        did = True
    for pat in (a.traj or []):
        for f in sorted(glob.glob(pat)):
            eps = traj_episodes(f)
            if not eps:
                print(f"  {f}: no episodes found, skipped")
                continue
            for i, (pts, spd) in enumerate(eps):
                analyse(F, pts, a.reach, speeds=spd, profile=False,
                        label=f"traj {Path(f).name} ep{i}")
            did = True
    if not did:
        ap.error("nothing to do: pass --route, --from or --traj")


if __name__ == "__main__":
    main()
