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

Three extra input modes exist for ONE question - "does the line have to be a
champion's?" (round 19 xARC's open caveat). They are all opt-in and none of
them can be reached without saying so on the command line:

    # a line with no champion in it: the deepest chain of a reward-free
    # Go-Explore phase-1 archive (tools/explore_phase1.py)
    python tools/build_route.py --archive runs/explore/archive.npz \\
        --allow-unfinished --out maps/cb.auto.route.npz

    # a line degraded until it carries the CORRIDOR and not the racing line
    python tools/build_route.py --from-route maps/cb.route.npz \\
        --decimate 32 --out maps/cb.coarse.route.npz

``--allow-unfinished`` is the deliberate, visible relaxation of the rule that
a route must reach the end zone. Without it this tool still refuses, exactly
as before; with it the written .npz carries ``truncated=True`` and the gap in
map units from the last vertex to the finish box, so nothing downstream can
mistake a partial line for a complete one.

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


def longest_episode(files):
    """-> (xyz of the episode with the greatest path length, episodes seen).

    The ``--allow-unfinished`` fallback for trajectory input. "Longest path"
    is deliberately the only ranking used: it needs no goal field, no route
    and no champion, so a line built this way carries nothing the recording
    itself did not.
    """
    best, best_len, n_ep = None, -1.0, 0
    for f in files:
        for ep in episodes_from_traj(f):
            n_ep += 1
            xyz = ep[:, 1:4]
            L = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            if L > best_len:
                best, best_len = xyz, L
    if best is None:
        raise SystemExit("no episodes found")
    return best, n_ep


def archive_chain(npz_path, leaf: str = "depth"):
    """Go-Explore phase-1 archive -> (xyz of the deepest chain, leaf, info).

    ``tools/explore_phase1.py`` writes a cell ARCHIVE, not a trajectory: per
    cell it keeps the best state ever seen there plus ``parent``, the archive
    index of the cell entered just before it on the run that produced that
    state. Walking ``parent`` from a leaf back to a map-spawn root therefore
    yields a time-ordered, single-provenance path - the same construction
    ``dump_win`` uses for its demo spine, minus the requirement that the leaf
    be a goal crossing.

    The chain is STITCHED, not replayed: consecutive rows can come from
    different exploration bursts, so the hop between two cells can be up to a
    couple of cell diagonals rather than one tick of travel. That is fine for
    a reference LINE, which only has to say where the track goes.

    Choosing the leaf is the one judgement call, and only two rules here are
    free of champion information:

    * ``depth`` - the cell whose cheapest known route from a map spawn is the
      LONGEST in physics ticks, i.e. the archive's own frontier. Uses nothing
      but the archive.
    * ``dist`` - the cell closest to the finish in the map's own geodesic
      field. Goal-aware but map-derived. On surf_src_cannonball that field has
      an interior minimum at route vertex 1601 (ledger round 18), so this rule
      will happily stop a line AT the wall; ``depth`` is the default for that
      reason.
    """
    z = np.load(npz_path, allow_pickle=False)
    state, parent, depth = z["state"], z["parent"], z["depth"]
    dist = z["dist"] if "dist" in z.files else np.full(len(state), np.inf)
    if leaf == "depth":
        i = int(np.argmax(depth))
    elif leaf == "dist":
        i = int(np.argmin(dist))
    else:
        raise ValueError(f"unknown --archive-leaf {leaf!r}")
    chain, seen = [], set()
    j = i
    while j >= 0 and j not in seen and len(chain) <= len(state):
        seen.add(j)
        chain.append(j)
        j = int(parent[j])
    chain.reverse()
    idx = np.asarray(chain, np.int64)
    return (np.asarray(state["origin"][idx], np.float64), i,
            {"cells": int(len(state)), "chain": int(len(chain)),
             "leaf_depth_ticks": int(depth[i]), "leaf_dist": float(dist[i])})


def decimate(pts, k: int):
    """Keep every k-th waypoint (first and last always), i.e. replace the line
    by straight chords of ``k * spacing`` units.

    This is a DEGRADATION, on purpose: chords cut every corner, so whatever
    lateral placement made the source line fast is destroyed while the order
    of the corridor it passes through survives. Linesight's reference line
    "does not need to be fast... usually the centerline"; this is how far that
    claim is tested here.
    """
    k = int(k)
    if k < 2:
        return pts
    idx = list(range(0, len(pts), k))
    if idx[-1] != len(pts) - 1:
        idx.append(len(pts) - 1)
    return pts[np.asarray(idx, np.int64)]


def quantize(pts, cell: float):
    """Snap waypoints to the centres of a ``cell``-unit lattice and drop
    consecutive duplicates - the polyline a cell-graph path would give you.

    What survives is exactly "which cells, in which order"; everything finer
    than the lattice is gone.
    """
    cell = float(cell)
    if cell <= 0:
        return pts
    q = np.floor(np.asarray(pts, np.float64) / cell) * cell + cell / 2.0
    keep = np.concatenate(([True], np.abs(np.diff(q, axis=0)).sum(1) > 1e-9))
    return q[keep]


def end_gap(pts, end_mins, end_maxs) -> float:
    """Units from the polyline's last vertex to the finish AABB (0 = inside)."""
    p = np.asarray(pts[-1], np.float64)
    return float(np.linalg.norm(p - np.clip(p, np.asarray(end_mins, np.float64),
                                            np.asarray(end_maxs, np.float64))))


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

    # (g) decimation keeps the ends and drops the middle
    d = decimate(pts, 32)
    assert np.allclose(d[0], pts[0]) and np.allclose(d[-1], pts[-1])
    assert len(d) <= len(pts) // 32 + 2, len(d)

    # (h) quantization snaps to lattice centres and dedups
    q = quantize(pts, 1024.0)
    assert np.allclose((q + 512.0) % 1024.0, 0.0), q[:3]
    assert len(q) < len(pts)

    # (i) an archive chain walks parents root-first
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        n = 6
        st = np.zeros(n, dtype=[("origin", np.float32, 3)])
        st["origin"] = np.stack([np.arange(n) * 100.0, np.zeros(n),
                                 np.zeros(n)], 1)
        p = np.array([-1, 0, 1, 2, 3, 1], np.int64)     # 5 branches off 1
        dep = np.array([0, 10, 20, 30, 40, 15], np.int64)
        dst = np.array([9., 8., 7., 6., 5., 99.])
        ap = Path(td) / "archive.npz"
        np.savez(ap, state=st, parent=p, depth=dep, dist=dst)
        xyz, leaf, info = archive_chain(ap, "depth")
        assert leaf == 4 and info["chain"] == 5, (leaf, info)
        assert np.allclose(xyz[:, 0], [0, 100, 200, 300, 400]), xyz
        xyz2, leaf2, _ = archive_chain(ap, "dist")
        assert leaf2 == 4, leaf2

    # (j) end_gap is 0 inside the box and the true miss outside it
    assert end_gap(np.array([[0., 0., 0.]]), [-1, -1, -1], [1, 1, 1]) == 0.0
    assert abs(end_gap(np.array([[5., 0., 0.]]), [-1, -1, -1], [1, 1, 1])
               - 4.0) < 1e-6
    print("selftest OK: 10 geometry checks")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="*", help="trajectory .jsonl files or globs")
    ap.add_argument("--out", default=None, help="output .npz")
    ap.add_argument("--map", default="surf_src_cannonball")
    ap.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    ap.add_argument("--pad", type=float, default=64.0,
                    help="end-zone tolerance in map units")
    ap.add_argument("--archive", default=None,
                    help="tools/explore_phase1.py archive.npz - build the line "
                         "from the deepest chain of a reward-free Go-Explore "
                         "phase-1 archive instead of from trajectories")
    ap.add_argument("--archive-leaf", default="depth", choices=("depth", "dist"),
                    help="which archive cell ends the chain: 'depth' = the "
                         "archive's own frontier (no goal information at all); "
                         "'dist' = closest to the finish in the map's geodesic "
                         "field (which has an interior minimum at the wall)")
    ap.add_argument("--from-route", default=None,
                    help="an existing route .npz as the input polyline "
                         "(for --decimate / --quantize)")
    ap.add_argument("--decimate", type=int, default=0,
                    help="DEGRADE: keep every Nth waypoint, i.e. replace the "
                         "line by straight chords of N*spacing units")
    ap.add_argument("--quantize", type=float, default=0.0,
                    help="DEGRADE: snap waypoints to the centres of a lattice "
                         "of this size and drop consecutive duplicates")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="do NOT require the source to reach the end zone. The "
                         "written .npz then carries truncated=True and the gap "
                         "to the finish box. Mandatory for --archive.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    n_modes = sum(bool(x) for x in (a.traj, a.archive, a.from_route))
    if n_modes != 1 or not a.out:
        ap.error("need exactly one of: trajectory files / --archive / "
                 "--from-route, plus --out (or --selftest)")

    zp = ROOT / "maps" / f"{a.map}.zones.json"
    end = json.loads(zp.read_text(encoding="utf-8"))["end"]
    secs, provenance, derivation = 0.0, "", []

    if a.archive:
        if not a.allow_unfinished:
            raise SystemExit(
                "--archive lines are discovery frontiers, not finishes; pass "
                "--allow-unfinished and say so in the ledger")
        xyz, leaf, info = archive_chain(a.archive, a.archive_leaf)
        provenance = Path(a.archive).name
        derivation.append(f"go-explore archive chain leaf={a.archive_leaf} "
                          f"cells={info['cells']} chain={info['chain']} "
                          f"leaf_depth={info['leaf_depth_ticks']}t")
        print(f"archive {provenance}: {info['cells']:,} cells, leaf {leaf} "
              f"by {a.archive_leaf} (depth {info['leaf_depth_ticks']}t, "
              f"geodesic d {info['leaf_dist']:,.0f}u), chain {info['chain']}")
    elif a.from_route:
        z = np.load(a.from_route)
        xyz = np.asarray(z["route"], np.float64)
        provenance = Path(a.from_route).name
        derivation.append(f"from-route {provenance} ({len(xyz)} pts)")
        print(f"input route {provenance}: {len(xyz)} points")
    else:
        files = []
        for pat in a.traj:
            files.extend(sorted(glob.glob(pat))
                         or ([pat] if Path(pat).exists() else []))
        if not files:
            raise SystemExit("no trajectory files matched")
        xyz, secs, n_fin, n_ep = pick_route(files, end["mins"], end["maxs"],
                                            pad=a.pad)
        if xyz is None:
            if not a.allow_unfinished:
                raise SystemExit(
                    f"none of the {n_ep} episodes in {len(files)} file(s) "
                    f"reached the end zone {end['mins']}..{end['maxs']} "
                    f"(pad {a.pad:g}u) - a route must reach the finish "
                    f"(--allow-unfinished overrides, deliberately)")
            xyz, n_ep = longest_episode(files)
            derivation.append(f"longest of {n_ep} NON-finishing episodes")
            print(f"0/{n_ep} episodes finished; --allow-unfinished took the "
                  f"longest path instead ({len(xyz)} samples)")
        else:
            print(f"{n_fin}/{n_ep} episodes finished; fastest {secs:.2f}s")
        provenance = ";".join(Path(f).name for f in files[:8])

    if a.quantize > 0:
        before = len(xyz)
        xyz = quantize(xyz, a.quantize)
        derivation.append(f"quantized to a {a.quantize:g}u lattice "
                          f"({before} -> {len(xyz)} waypoints)")
        print(f"  quantize {a.quantize:g}u: {before} -> {len(xyz)} waypoints")
    if a.decimate > 1:
        before = len(xyz)
        xyz = decimate(xyz, a.decimate)
        derivation.append(f"decimated 1/{a.decimate} "
                          f"({before} -> {len(xyz)} waypoints)")
        print(f"  decimate 1/{a.decimate}: {before} -> {len(xyz)} waypoints")

    pts, total = resample_polyline(xyz, a.spacing)
    gap = end_gap(pts, end["mins"], end["maxs"])
    truncated = gap > a.pad
    if truncated and not a.allow_unfinished:
        raise SystemExit(f"the resulting line stops {gap:,.0f}u short of the "
                         f"end zone - pass --allow-unfinished to write it")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, route=pts.astype(np.float32),
                        spacing=np.float32(a.spacing),
                        seconds=np.float32(secs), map=a.map,
                        source=provenance,
                        truncated=np.bool_(truncated),
                        end_gap=np.float32(gap),
                        derivation=" | ".join(derivation) or "fastest finisher")
    print(f"route: {len(pts)} points @ {a.spacing:g}u, path {total:,.0f}u")
    print(f"  start {pts[0].round(1).tolist()}  end {pts[-1].round(1).tolist()}")
    if truncated:
        print(f"  ** TRUNCATED: the last vertex is {gap:,.0f}u from the finish "
              f"box. Arc progress saturates there and the rest of the map pays "
              f"no shaping at all. **")
    else:
        print(f"  reaches the finish box (gap {gap:,.0f}u <= pad {a.pad:g}u)")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
