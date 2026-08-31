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

--from-field builds the same kind of line with NO recordings at all: a
steepest-descent walk down the map's baked geodesic goal field, from a seed
point to the finish basin. That is what makes a reference line exist on a
map nobody has ever flown - the field is a function of the geometry alone.

    python tools/build_route.py --from-field \\
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp --cell 32 \\
        --seed "-9856,-5568,-2624" \\
        --out maps/surf_src_cannonball.fieldroute.npz

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


# ------------------------------------------------------------- --from-field
#
# The 26-neighbour enumeration order, FIXED: dz outermost, then dy, then dx,
# each ascending (-1, 0, +1), centre skipped. The descent step scans in this
# order with a strict `<`, so an exact tie keeps the FIRST neighbour listed
# here and the walk is reproducible run to run and machine to machine. A
# reference line an arm trains against must not depend on iteration luck.
_NEIGH26 = tuple((dz, dy, dx)
                 for dz in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dx in (-1, 0, 1)
                 if (dz, dy, dx) != (0, 0, 0))

# strict-descent margin, map units. The cached field is quantized to cell/8
# (4 u at cell 32), so this only rejects exact plateaus, never real steps.
DESCENT_EPS = 1e-3

# how far the seed may be snapped, in cells, before we give up
SNAP_CELLS = 4


def _shell(r):
    """Lattice offsets at Chebyshev radius exactly ``r``, in _NEIGH26 order."""
    if r <= 0:
        return ((0, 0, 0),)
    rng = range(-r, r + 1)
    return tuple((dz, dy, dx) for dz in rng for dy in rng for dx in rng
                 if max(abs(dz), abs(dy), abs(dx)) == r)


def voxel_centers(field, idx):
    """(N, 3) lattice indices (ix, iy, iz) -> (N, 3) world voxel centres.

    ``mins`` is the world corner of voxel (0, 0, 0) and GoalField.sample
    reads the lattice at ``(pos - mins) / cell - 0.5``, so the centre of
    voxel ``i`` is ``mins + (i + 0.5) * cell``.
    """
    i = np.asarray(idx, np.float64).reshape(-1, 3)
    return np.asarray(field.mins, np.float64) + (i + 0.5) * float(field.cell)


def snap_seed(field, seed, max_cells: int = SNAP_CELLS):
    """Nearest lattice voxel to ``seed`` carrying an HONEST field value.

    A hand-picked seed lands wherever it lands - inside a ramp, inside a
    wall, or in a pocket the wavefront never reached - and all three read as
    the same sentinel. Search outward by Chebyshev shells and take the
    honest voxel whose CENTRE is closest to the seed, ties broken by shell
    order. -> (ix, iy, iz).
    """
    grid = np.asarray(field.grid)
    nz, ny, nx = grid.shape
    valid_max = float(field._valid_max)
    cell = float(field.cell)
    mins = np.asarray(field.mins, np.float64)
    p = np.asarray(seed, np.float64).reshape(3)
    base = np.floor((p - mins) / cell).astype(np.int64)
    for r in range(int(max_cells) + 1):
        best, best_d2 = None, float("inf")
        for dz, dy, dx in _shell(r):
            ix, iy, iz = int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                continue
            if float(grid[iz, iy, ix]) >= valid_max:   # solid / unreachable
                continue
            c = mins + (np.array([ix, iy, iz], np.float64) + 0.5) * cell
            d2 = float(((c - p) ** 2).sum())
            if d2 < best_d2:
                best, best_d2 = (ix, iy, iz), d2
        if best is not None:
            return best
    raise ValueError(
        f"no honest goal-field voxel within {max_cells} cells "
        f"({max_cells * cell:g}u) of seed {p.tolist()} - it is buried in "
        f"geometry or in space the wavefront never reached; move the seed")


def trace_descent(field, seed, eps: float = DESCENT_EPS,
                  max_cells: int = SNAP_CELLS, max_steps=None):
    """Steepest-descent walk down a baked goal field -> (pts, info).

    ``pts`` is (N, 3) float64 world voxel centres, ordered START -> FINISH
    like any other route; ``info`` carries ``idx`` (N, 3) int32 lattice
    indices, ``vals`` (N,) the field value at each, and the scalars ``d0``,
    ``d_end``, ``n_steps``.

    Deliberately on the RAW GRID, not ``field.sample()``: the trilinear
    sampler renormalizes over honest corners, so near a surface it returns a
    one-sided extrapolation that can descend smoothly INTO a wall. The
    lattice values are what the wavefront actually computed, and a step
    between two honest voxels is a step a path could take.

    Strictly descending (each step must beat the current value by more than
    ``eps``), so a cycle is impossible by construction; the step cap exists
    to make a violated invariant loud instead of infinite. Terminating at
    ``2 * cell`` is the finish basin: the wavefront was seeded across the
    whole (inflated) end box, so the last cell or two of "distance" is
    already inside the zone that ends the episode.
    """
    grid = np.asarray(field.grid)
    nz, ny, nx = grid.shape
    valid_max = float(field._valid_max)
    cell = float(field.cell)
    cap = int(nx) * int(ny) * int(nz) if max_steps is None else int(max_steps)
    stop_at = 2.0 * cell

    start = snap_seed(field, seed, max_cells)
    idx = [start]
    vals = [float(grid[start[2], start[1], start[0]])]
    while vals[-1] > stop_at:
        ix, iy, iz = idx[-1]
        best, best_v = None, vals[-1] - eps
        for dz, dy, dx in _NEIGH26:
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            if not (0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz):
                continue
            v = float(grid[jz, jy, jx])
            if v >= valid_max:          # solid / unreachable sentinel
                continue
            if v < best_v:              # strict: ties keep the earlier one
                best, best_v = (jx, jy, jz), v
        if best is None:                # local minimum of the lattice
            break
        idx.append(best)
        vals.append(best_v)
        if len(idx) > cap:
            raise RuntimeError(
                f"descent exceeded the {cap} step cap - strict descent cannot "
                f"cycle, so the field or the walk is broken")
    return (voxel_centers(field, idx),
            {"idx": np.asarray(idx, np.int32),
             "vals": np.asarray(vals, np.float64),
             "d0": float(vals[0]), "d_end": float(vals[-1]),
             "n_steps": int(len(idx))})


def save_field_route(out, pts, spacing, map_name, seed, info,
                     cell: float = 0.0):
    """Write the field line in the format RouteLine/ArcProgress.load read.

    ``route`` + ``spacing`` are the contract (and the dtypes the trajectory
    path writes); everything else is provenance - a line with no recording
    behind it needs its seed and its endpoints on the record.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, route=np.asarray(pts, np.float32),
                        spacing=np.float32(spacing),
                        map=str(map_name), source="field",
                        seed=np.asarray(seed, np.float32).reshape(3),
                        d0=np.float32(info["d0"]),
                        d_end=np.float32(info["d_end"]),
                        n_steps=np.int32(info["n_steps"]),
                        cell=np.float32(cell))
    return out


def parse_seed(text):
    """'x,y,z' (or 'x y z') -> (3,) float64."""
    parts = [t for t in str(text).replace(",", " ").split() if t]
    if len(parts) != 3:
        raise SystemExit(f"--seed wants three numbers 'x,y,z', got {text!r}")
    try:
        return np.array([float(t) for t in parts], np.float64)
    except ValueError:
        raise SystemExit(f"--seed is not numeric: {text!r}")


def resolve_bsp(spec):
    """``--map`` is a map STEM in trajectory mode and a PATH here.

    Used VERBATIM when it looks like a path: the caller points at the main
    checkout's maps/ on purpose, because a worktree copy has different
    mtimes and every prebaked cache keys on mtime_ns (CLAUDE.md).
    """
    p = Path(spec)
    if p.suffix.lower() == ".bsp" or p.exists():
        return p
    return ROOT / "maps" / f"{spec}.bsp"


def field_route(a):
    """--from-field: extract the line from the map's baked goal field."""
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import build_goal_field
    from surfgym.zones import load_zones

    if not a.out:
        raise SystemExit("--from-field needs --out")
    if not a.seed:
        raise SystemExit("--from-field needs --seed 'x,y,z' (world units)")
    seed = parse_seed(a.seed)
    bsp = resolve_bsp(a.map)
    if not bsp.is_file():
        raise SystemExit(f"no such bsp: {bsp}")

    if a.zones:
        zones = json.loads(Path(a.zones).read_text(encoding="utf-8"))
    else:
        zones = load_zones(str(bsp))
    goal_box = zones.get("end")
    if not goal_box:
        raise SystemExit(f"{bsp.stem} has no end zone - the goal field is "
                         f"seeded from it; hand-label its zones.json first")

    # same core/field path train_fast.py uses, so the SAME cache signature
    # applies and a warm map costs seconds (train_fast.py:2547-2650)
    core = SurfCore(str(bsp), default_config(num_envs=1, spawn_mode=2,
                                             lidar_w=0, lidar_h=0))
    cache_file = bsp.parent / f"{bsp.stem}.goal_{a.cell:g}.npz"
    bar = "=" * 70
    print(bar)
    print(f"opening the goal-field cache {cache_file}")
    print(f"  present: {cache_file.exists()}")
    print("  a HIT costs seconds. A MISS BAKES THE FIELD FROM SCRATCH:")
    print("  ~30 minutes on the GPU, and a pool transferred as a tar misses")
    print("  even with the file present (mtime -> run tools/restamp_maps.py).")
    print("  Ctrl+C NOW if you did not mean to bake.")
    print(bar)
    field = build_goal_field(core, goal_box, cell=a.cell, device=a.device)

    pts, info = trace_descent(field, seed)
    if len(pts) < 2:
        raise SystemExit(
            f"the walk stopped at its first voxel (d = {info['d0']:.0f}u): "
            f"the seed is already in the finish basin or in a dead pocket")
    line, total = resample_polyline(pts, a.spacing)
    out = save_field_route(a.out, line, a.spacing, bsp.stem, seed, info,
                           cell=a.cell)
    print(f"field line: {info['n_steps']} voxel steps -> {len(line)} points "
          f"@ {a.spacing:g}u, path {total:,.0f}u, "
          f"d {info['d0']:,.0f} -> {info['d_end']:,.0f}u")
    if info["d_end"] > 2.0 * a.cell:
        # a healthy walk ends in the finish basin. Stopping anywhere else
        # means the lattice has a LOCAL minimum there - the exact defect
        # this field is documented to have on cannonball - and the line
        # stops short of the goal by however much d_end says.
        print(f"  WARNING: the walk died in a local minimum at "
              f"d = {info['d_end']:,.0f}u, not in the finish basin "
              f"(<= {2.0 * a.cell:g}u). The line STOPS SHORT.")
    print(f"  start {line[0].round(1).tolist()}  end {line[-1].round(1).tolist()}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


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
    ap.add_argument("traj", nargs="*",
                    help="trajectory .jsonl files or globs (unused with "
                         "--from-field)")
    ap.add_argument("--out", default=None, help="output .npz")
    ap.add_argument("--map", default="surf_src_cannonball",
                    help="map stem; with --from-field, the .bsp PATH "
                         "(absolute paths are used verbatim)")
    ap.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    ap.add_argument("--pad", type=float, default=64.0,
                    help="end-zone tolerance in map units")
    ap.add_argument("--from-field", action="store_true",
                    help="extract the line from the baked geodesic goal "
                         "field instead of from recordings")
    ap.add_argument("--cell", type=float, default=32.0,
                    help="goal-field cell size (32 = the goal_32 cache)")
    ap.add_argument("--seed", default=None,
                    help="--from-field: 'x,y,z' world point to descend from")
    ap.add_argument("--zones", default=None,
                    help="--from-field: zones .json override (default: the "
                         "map's own, as train_fast.py loads it)")
    ap.add_argument("--device", default="cuda",
                    help="--from-field: device for a goal-field BAKE; "
                         "unused on a cache hit")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.from_field:
        return field_route(a)
    if not a.traj or not a.out:
        ap.error("need trajectory files and --out "
                 "(or --from-field, or --selftest)")

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
