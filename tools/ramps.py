"""ramps.py - extract a surf map's RAMPS from its collision geometry (the user's ramp-graph idea,
2026-09-27): every surface the engine lets you slide on, grouped into connected ramps, each with
its potential (the map's own geodesic goal field) - no per-map constant.

    python tools/ramps.py maps_pool/<map>.bsp [--cell 16] [--goal-cell 32] [--out ramps.npz]
        [--png ramps.png]

Method:
  1. the free voxels of the map's cached occupancy grid (<map>.occ_<cell>.npz) that touch a solid
     voxel (6-neighbourhood);
  2. from each, a POINT-hull trace (the core's own collision) toward each solid neighbour: the hit
     gives a surface point and its plane normal;
  3. a surface is a RAMP when 0 < normal.z < 0.7: steeper than the engine's walkable limit (0.7,
     PM_CategorizePosition's ground test - an engine constant, not a map one) and facing upward;
     normal.z >= 0.7 is FLOOR, the rest wall / ceiling;
  4. ramp hits are grouped into connected components: two hits join when they are within 1.5
     cells of each other and their normals differ by less than 25 degrees (a curved ramp is many
     planar faces; the threshold is geometric, not map-derived);
  5. each ramp gets its sample count (~ area / cell^2), bounding box, mean normal and the geodesic
     potential over its points (mean, min, max): d = the goal field's distance-to-finish.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

WALKABLE_NZ = 0.7          # the engine's ground test (pm.c: plane normal z >= 0.7 is ground)
MIN_NZ = 0.02              # below this a surface is a wall (vertical): nothing to slide down
JOIN_DEG = 25.0            # neighbouring hits with normals closer than this belong to one ramp
DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def extract(core, occ, mins, cell):
    """-> points (N,3), normals (N,3) of the collision surface seen from the free boundary
    voxels. occ is [nz, ny, nx], 1 = solid."""
    nz, ny, nx = occ.shape
    solid = occ.astype(bool)
    pts, nrm = [], []
    seen = set()
    for d in DIRS:
        dz, dy, dx = int(d[2]), int(d[1]), int(d[0])
        # free voxel whose neighbour in direction d is solid
        a = ~solid
        b = np.zeros_like(solid)
        zs = slice(max(0, -dz), nz - max(0, dz))
        ys = slice(max(0, -dy), ny - max(0, dy))
        xs = slice(max(0, -dx), nx - max(0, dx))
        zt = slice(max(0, dz), nz - max(0, -dz))
        yt = slice(max(0, dy), ny - max(0, -dy))
        xt = slice(max(0, dx), nx - max(0, -dx))
        b[zs, ys, xs] = solid[zt, yt, xt]
        iz, iy, ix = np.nonzero(a & b)
        for k in range(len(iz)):
            c = mins + (np.array([ix[k], iy[k], iz[k]], np.float64) + 0.5) * cell
            e = c + d * 1.5 * cell
            tr = core.trace(c.tolist(), e.tolist(), 2)
            if tr.startsolid or tr.allsolid or tr.fraction >= 1.0:
                continue
            p = np.array(tr.endpos[:], np.float64)
            key = tuple(np.round(p / 4.0).astype(int))
            if key in seen:
                continue
            seen.add(key)
            pts.append(p)
            nrm.append(np.array(tr.normal[:], np.float64))
    return np.asarray(pts), np.asarray(nrm)


def group(pts, nrm, cell):
    """connected components: within 1.5 cells and normals within JOIN_DEG"""
    from scipy.spatial import cKDTree
    n = len(pts)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    tree = cKDTree(pts)
    cos_j = np.cos(np.radians(JOIN_DEG))
    for i, j in tree.query_pairs(1.5 * cell):
        if float(nrm[i] @ nrm[j]) >= cos_j:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bsp")
    ap.add_argument("--cell", type=float, default=16.0)
    ap.add_argument("--goal-cell", type=int, default=32)
    ap.add_argument("--min-samples", type=int, default=8,
                    help="drop ramps smaller than this many surface samples (~cell^2 each)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    a = ap.parse_args(argv)
    from surfgym.core import SurfCore, default_config
    from surfgym.goalfield import load_goal_field
    bsp = Path(a.bsp)
    core = SurfCore(str(bsp), default_config(num_envs=1))
    z = np.load(bsp.with_name(f"{bsp.stem}.occ_{a.cell:g}.npz"))
    occ, mins = z["occ"], np.asarray(z["mins"], np.float64)
    pts, nrm = extract(core, occ, mins, a.cell)
    nzs = nrm[:, 2]
    is_ramp = (nzs > MIN_NZ) & (nzs < WALKABLE_NZ)
    is_floor = nzs >= WALKABLE_NZ
    print(f"ramps.py {bsp.stem}: {len(pts):,} surface samples at cell {a.cell:g}: "
          f"ramp {int(is_ramp.sum()):,}, floor {int(is_floor.sum()):,}, wall/ceiling "
          f"{int((~is_ramp & ~is_floor).sum()):,}")
    rp, rn = pts[is_ramp], nrm[is_ramp]
    lab = group(rp, rn, a.cell)
    gf = None
    gpath = bsp.with_name(f"{bsp.stem}.goal_{a.goal_cell}.npz")
    if gpath.exists():
        gf = load_goal_field(str(gpath))
    ramps = []
    for r in range(lab.max() + 1):
        m = lab == r
        if int(m.sum()) < a.min_samples:
            continue
        p = rp[m]
        d = gf.sample(p + rn[m] * 24.0) if gf is not None else np.full(len(p), np.nan)
        d = d[np.isfinite(d)]
        ramps.append({"n": int(m.sum()), "lo": p.min(0), "hi": p.max(0),
                      "centroid": p.mean(0), "normal": rn[m].mean(0),
                      "d_mean": float(d.mean()) if len(d) else np.nan,
                      "d_min": float(d.min()) if len(d) else np.nan,
                      "d_max": float(d.max()) if len(d) else np.nan, "mask": m})
    ramps.sort(key=lambda r: -r["d_mean"] if np.isfinite(r["d_mean"]) else 0)
    print(f"   {len(ramps)} ramps with >= {a.min_samples} samples (of {lab.max() + 1} components)"
          f"; ordered by potential (far from the finish first):")
    print("    id  samples  centroid (x, y, z)        normal z   potential d mean / min / max")
    for i, r in enumerate(ramps):
        c = r["centroid"]
        print(f"   {i:3d}  {r['n']:7d}  ({c[0]:7.0f},{c[1]:7.0f},{c[2]:6.0f})   {r['normal'][2]:5.2f}"
              f"     {r['d_mean']:8,.0f} / {r['d_min']:8,.0f} / {r['d_max']:8,.0f}")
    if a.out:
        rid = np.full(len(rp), -1, np.int32)
        for i, r in enumerate(ramps):
            rid[r["mask"]] = i
        np.savez_compressed(a.out, points=rp.astype(np.float32), normals=rn.astype(np.float32),
                            ramp_id=rid, d_mean=np.array([r["d_mean"] for r in ramps]),
                            d_min=np.array([r["d_min"] for r in ramps]),
                            d_max=np.array([r["d_max"] for r in ramps]),
                            map=np.array(bsp.stem), cell=np.float32(a.cell))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(pts[is_floor, 0], pts[is_floor, 1], s=1, c="0.85", label="floor")
        for i, r in enumerate(ramps):
            p = rp[r["mask"]]
            ax.scatter(p[:, 0], p[:, 1], s=2, c="0.2")
            ax.annotate(f"R{i}", r["centroid"][:2], fontsize=9, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))
        ax.set_aspect("equal")
        ax.set_title(f"{bsp.stem}: ramps (dark, labelled R<id> by potential, R0 farthest from "
                     f"the finish) and floors (light)")
        ax.set_xlabel("x (u)")
        ax.set_ylabel("y (u)")
        fig.savefig(a.png, dpi=110, bbox_inches="tight")
        print(f"   plot -> {a.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
