#!/usr/bin/env python3
"""dip_probe.py - how much SHAPING POTENTIAL must a route give up?

The geodesic goal field is a BFS through free space, so it believes the
player can fly across a void (CLAUDE.md, cannonball's barrier). Where the
real route has to leave the field's straight line - unitfarmer2's pit, an
edgeflow detour - the agent must accept a RISE in d before any progress is
paid back. That rise is the dip, and it is what every exploration arm on
this project is actually being asked to cross.

This measures it, champion-free, from geometry alone:

  1. the goal field d (the same cache the trainer shapes on),
  2. a TRAVERSABILITY mask - free voxels that are above the map's kill
     volumes and within ``--near`` units of solid geometry, i.e. places a
     surfer can ride, land on or cross in one hop,
  3. the BOTTLENECK path from spawn to goal inside that mask: the path that
     minimises the MAXIMUM d along it (a minimax / widest-path Dijkstra).

``dip = max_d_on_that_path - d(spawn)`` is then the smallest potential rise
any surface-following route can pay, and ``100 * dip / d0`` is its cost in
reward units under ``--race-shaping`` (the shaping scale is 100/d0).

    python tools/dip_probe.py maps_pool/surf_edgeflow_blue100.bsp
    python tools/dip_probe.py maps_pool/surf_unitfarmer2.bsp --goal-cell 48

Measurement only - nothing here may enter a reward, a spawn rule or a
schedule (CLAUDE.md section 0b).
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import SurfCore, SurfEnvConfig      # noqa: E402
from surfgym.goalfield import build_goal_field, goal_occupancy  # noqa: E402
from surfgym.zones import load_zones, parse_bsp       # noqa: E402


def spawn_points(bsp: str) -> np.ndarray:
    ents, _ = parse_bsp(bsp)
    pts = []
    for e in ents:
        if e.get("classname") not in ("info_player_start",
                                      "info_player_deathmatch"):
            continue
        if not e.get("origin"):
            continue
        try:
            v = [float(x) for x in e["origin"].split()[:3]]
        except ValueError:
            continue
        if len(v) == 3:
            pts.append(v)
    return np.asarray(pts, np.float64)


def kill_ceiling(bsp: str) -> float:
    """Top of the highest fall-net / teleport volume: a route must stay
    above it. -inf when the map has none."""
    ents, bb = parse_bsp(bsp)
    top = -np.inf
    for e in ents:
        if e.get("classname") not in ("trigger_teleport", "trigger_hurt"):
            continue
        mdl = e.get("model", "")
        if not mdl.startswith("*"):
            continue
        try:
            idx = int(mdl[1:])
        except ValueError:
            continue
        if idx < len(bb):
            top = max(top, float(bb[idx][1][2]))
    return top


def supported(occ: np.ndarray, r: int, dead: np.ndarray) -> np.ndarray:
    """A voxel is SUPPORTED when a solid voxel sits within ``r`` voxels
    BELOW it and that support is itself above the map's kill volumes - a
    ramp, a platform or a ledge the player can ride, land on or cross in
    one hop. Support from below is what makes this a route model rather
    than a free-flight model: a ceiling or a wall a metre away supports
    nothing, and the death floor under a void is not a support at all
    (without that second clause a "route" flies straight over the pit,
    held up by the fall net beneath it - measured on the edgeflow maps,
    2026-09-15)."""
    solid = occ.copy()
    solid[dead] = False                   # the fall net and everything under it
    out = np.zeros_like(solid)
    for s in range(1, r + 1):             # solid at z - s  ->  support at z
        out |= np.roll(solid, s, axis=0)
    out[dead] = False
    out &= ~occ          # a voxel INSIDE the wall is not a place to ride: without
                         # this the wall seeds the hop closure and the route hugs
                         # it straight across the pit (measured 2026-09-15)
    return out


def hop(sup: np.ndarray, free: np.ndarray, h: int) -> np.ndarray:
    """Close the supported set under one ballistic HOP of at most ``h``
    voxels in any direction: crossing the gap between two ramps, dropping
    onto a lower one. Anything further from a rideable surface than that is
    not a route, it is a fall - which is what keeps this from flying
    straight across a pit (a fall-closure version did exactly that: the
    route dropped to just above the fall net and walked along the bottom,
    2026-09-15)."""
    out = sup.copy()
    for ax in (0, 1, 2):
        acc = out.copy()
        for s in range(1, h + 1):
            acc |= np.roll(out, s, axis=ax) & free
            acc |= np.roll(out, -s, axis=ax) & free
        out = acc
    return out


def shell_field(shell: np.ndarray, seeds) -> np.ndarray:
    """6-connected BFS distance (in voxels) to the nearest seed, inside the
    ride shell. This is the route model: a surfer travels along surfaces,
    not through the middle of a pit."""
    from collections import deque
    S = np.full(shell.shape, np.inf, np.float32)
    q = deque()
    for v in seeds:
        if shell[v] and not np.isfinite(S[v]):
            S[v] = 0.0
            q.append(v)
    nz, ny, nx = shell.shape
    while q:
        cur = q.popleft()
        base = S[cur] + 1.0
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (cur[0] + dz, cur[1] + dy, cur[2] + dx)
            if not (0 <= n[0] < nz and 0 <= n[1] < ny and 0 <= n[2] < nx):
                continue
            if shell[n] and base < S[n]:
                S[n] = base
                q.append(n)
    return S


def trace(S: np.ndarray, start):
    """Steepest descent of the shell field from ``start`` to a seed."""
    path = [start]
    cur = start
    nz, ny, nx = S.shape
    for _ in range(int(S.size)):
        if S[cur] == 0.0:
            break
        best, bn = S[cur], None
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (cur[0] + dz, cur[1] + dy, cur[2] + dx)
            if not (0 <= n[0] < nz and 0 <= n[1] < ny and 0 <= n[2] < nx):
                continue
            if S[n] < best:
                best, bn = S[n], n
        if bn is None:
            break
        cur = bn
        path.append(cur)
    return path


def dip_of(d: np.ndarray, path) -> tuple:
    """The give-back: the largest rise of d above its own running minimum
    along the route, and where it peaks. This is what the agent
    experiences - progress is banked, then some of it must be handed back
    before the route can continue."""
    run = np.inf
    worst, at, base = 0.0, None, None
    for v in path:
        dv = float(d[v])
        if dv < run:
            run = dv
        elif dv - run > worst:
            worst, at, base = dv - run, v, run
    return worst, at, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bsp")
    ap.add_argument("--goal-cell", type=float, default=32.0)
    ap.add_argument("--hop", type=float, nargs="+", default=[128.0, 256.0, 512.0],
                    help="how far a route may travel HORIZONTALLY away from a "
                         "supported spot (one sensitivity row per value): the gap "
                         "a surfer can cross between two ramps")
    ap.add_argument("--ride", type=float, default=256.0,
                    help="a supported voxel has solid geometry within this many "
                         "units below it (the height a surfer rides at)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    bsp = str(Path(a.bsp))
    zones = load_zones(bsp)
    if not zones.get("end"):
        print(f"!! {bsp}: no end zone"); return 2
    core = SurfCore(bsp, SurfEnvConfig(num_envs=1))
    gf = build_goal_field(core, zones["end"], a.goal_cell, device="cpu")
    occ, omins = goal_occupancy(core, a.goal_cell, None)
    occ = np.asarray(occ, bool)
    cell = float(a.goal_cell)
    d = np.asarray(gf.grid, np.float64)
    if d.shape != occ.shape:
        print(f"!! grid {d.shape} vs occ {occ.shape}"); return 2

    sp = spawn_points(bsp)
    d0 = float(np.median(gf.sample(sp))) if len(sp) else float("nan")
    kz = kill_ceiling(bsp)

    def vox(p):                       # world point -> (iz, iy, ix)
        f = (np.asarray(p, np.float64) - np.asarray(omins, np.float64)) / cell
        i = np.floor(f).astype(np.int64)
        return (int(np.clip(i[2], 0, d.shape[0] - 1)),
                int(np.clip(i[1], 0, d.shape[1] - 1)),
                int(np.clip(i[0], 0, d.shape[2] - 1)))

    free = ~occ
    zs = omins[2] + (np.arange(d.shape[0]) + 0.5) * cell
    above_kill = np.ones_like(free)
    dead = np.zeros(free.shape, bool)
    if np.isfinite(kz):
        above_kill = np.zeros_like(free)
        above_kill[zs > kz, :, :] = True
        dead[zs <= kz, :, :] = True       # a voxel centre at or below the net
    reach = np.isfinite(d) & (d < 1e8)

    goal = zones["end"]
    gmin, gmax = np.asarray(goal["mins"]), np.asarray(goal["maxs"])
    gc = vox((gmin + gmax) / 2.0)
    goal_mask = np.zeros_like(free)
    lo, hi = vox(gmin), vox(gmax)
    goal_mask[min(lo[0], hi[0]):max(lo[0], hi[0]) + 1,
              min(lo[1], hi[1]):max(lo[1], hi[1]) + 1,
              min(lo[2], hi[2]):max(lo[2], hi[2]) + 1] = True
    goal_mask &= free

    starts = [vox(p) for p in sp] if len(sp) else []

    out = {"map": Path(bsp).stem, "goal_cell": cell, "d0": round(d0, 1),
           "reach_max": round(float(gf.reach_max), 1),
           "kill_ceiling": (None if not np.isfinite(kz) else round(float(kz), 1)),
           "straight_line": round(float(np.linalg.norm(
               (gmin + gmax) / 2.0 - np.median(sp, axis=0))), 1) if len(sp) else None,
           "ride": a.ride, "rows": []}
    sup = supported(occ, max(1, int(round(a.ride / cell))), dead)
    gseeds = [tuple(v) for v in np.argwhere(goal_mask)]
    for near in a.hop:
        h = max(1, int(round(near / cell)))
        shell = free & above_kill & reach & hop(sup, free & above_kill, h)
        S = shell_field(shell, gseeds)
        row = {"hop": near, "shell_voxels": int(shell.sum())}
        live = [s0 for s0 in starts if shell[s0] and np.isfinite(S[s0])]
        if not live:
            row["connected"] = False
            out["rows"].append(row)
            continue
        s0 = min(live, key=lambda v: S[v])
        path = trace(S, s0)
        worst, at, base = dip_of(d, path)
        xs_ = [omins[0] + (v[2] + 0.5) * cell for v in path]
        row.update(connected=True, route_voxels=len(path),
                   route_len=round(len(path) * cell, 1),
                   route_x=[round(min(xs_)), round(max(xs_))],
                   dip=round(float(worst), 1),
                   dip_pct_d0=round(100.0 * float(worst) / d0, 1),
                   reward_cost=round(100.0 * float(worst) / d0, 2),
                   dip_at=(None if at is None else [round(omins[0] + (at[2] + .5) * cell),
                                                   round(omins[1] + (at[1] + .5) * cell),
                                                   round(omins[2] + (at[0] + .5) * cell)]),
                   dip_from_d=(None if base is None else round(float(base), 1)),
                   dip_to_d=(None if at is None else round(float(d[at]), 1)))
        out["rows"].append(row)

    if a.json:
        print(json.dumps(out, indent=1))
    else:
        print(f"== {out['map']}  goal cell {cell:g}  d0 {d0:,.0f}u  "
              f"reach_max {out['reach_max']:,.0f}u  "
              f"straight line {out['straight_line']:,.0f}u  "
              f"kill ceiling z={out['kill_ceiling']}")
        for row in out["rows"]:
            if not row.get("connected"):
                print(f"   hop {row['hop']:>5.0f}u: NO route through the ride shell "
                      f"({row['shell_voxels']:,} voxels)")
            else:
                where = ("" if row.get("dip_at") is None else
                         f"| d {row['dip_from_d']:,.0f} -> {row['dip_to_d']:,.0f} "
                         f"at {row['dip_at']} ")
                print(f"   hop {row['hop']:>5.0f}u: DIP {row['dip']:>7,.0f}u "
                      f"({row['dip_pct_d0']:>5.1f}% of d0 = {row['reward_cost']:>6.2f} reward) "
                      f"{where}| route {row['route_len']:,.0f}u, x {row['route_x']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
