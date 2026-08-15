#!/usr/bin/env python3
"""export_goalfield.py — sparse arrow-field export of the race potential.

The race shaping is potential-based on the geodesic distance-to-finish
field; its negative gradient is the direction the reward pulls. This dumps
that field as arrows for the three.js viewer: sampled on a coarse lattice
over REACHABLE free voxels, kept only near geometry (where agents actually
fly), each with the steepest-descent direction and the normalized potential
for coloring.

    python tools/export_goalfield.py maps/surf_src_cannonball.bsp
        -> viewer/assets/surf_src_cannonball.goalfield.json

Output (quantized ints to keep the payload small):
    {"map": ..., "reach_max": ..., "stride": ...,
     "pos": [x0,y0,z0, x1,...],          # world units, rounded
     "dir": [dx0,dy0,dz0, ...],          # descent unit vector * 100
     "pot": [p0, p1, ...]}               # d / reach_max * 1000
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bsp")
    ap.add_argument("--stride", type=int, default=8,
                    help="lattice stride in field voxels (8 = 256u at 32u)")
    ap.add_argument("--near-solid", type=float, default=256.0,
                    help="keep only points within this range of geometry")
    ap.add_argument("--cell", type=float, default=None)
    args = ap.parse_args()

    from surfgym import SurfCore, default_config
    from surfgym.goalfield import build_goal_field
    from surfgym.vision import build_sdf, pick_cell
    from surfgym.zones import load_zones

    core = SurfCore(args.bsp, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    cell = args.cell or pick_cell(core)
    zones = load_zones(core.bsp_path)
    if not zones.get("end"):
        raise SystemExit("map has no end zone — nothing to visualize")
    gf = build_goal_field(core, zones["end"], cell=cell)
    sdf, sdf_mins, sdf_cell = build_sdf(core, cell)

    grid = gf.grid
    ok = grid < gf.reach_max + 0.5 * gf.cell
    nz, ny, nx = grid.shape
    s = args.stride
    iz, iy, ix = np.meshgrid(np.arange(1, nz - 1, s), np.arange(1, ny - 1, s),
                             np.arange(1, nx - 1, s), indexing="ij")
    iz, iy, ix = iz.ravel(), iy.ravel(), ix.ravel()
    keep = ok[iz, iy, ix]
    iz, iy, ix = iz[keep], iy[keep], ix[keep]

    # near geometry only — arrows in empty sky are noise (the SDF shares the
    # grid layout: same bbox, same cell)
    near = sdf[iz, iy, ix].astype(np.float32) <= args.near_solid
    iz, iy, ix = iz[near], iy[near], ix[near]

    # central-difference gradient, masked: a sentinel neighbor falls back to
    # the center value so walls don't fake huge gradients
    def g_axis(dz, dy, dx):
        a = grid[np.clip(iz + dz, 0, nz - 1), np.clip(iy + dy, 0, ny - 1),
                 np.clip(ix + dx, 0, nx - 1)]
        c = grid[iz, iy, ix]
        return np.where(a < gf.reach_max + 0.5 * gf.cell, a, c)

    gx = (g_axis(0, 0, 1) - g_axis(0, 0, -1))
    gy = (g_axis(0, 1, 0) - g_axis(0, -1, 0))
    gz = (g_axis(1, 0, 0) - g_axis(-1, 0, 0))
    d = -np.stack([gx, gy, gz], axis=1)              # descent = -gradient
    n = np.linalg.norm(d, axis=1)
    live = n > 1e-3                                   # flat spots carry no pull
    iz, iy, ix, d, n = iz[live], iy[live], ix[live], d[live], n[live]
    d /= n[:, None]

    pos = np.stack([gf.mins[0] + (ix + 0.5) * gf.cell,
                    gf.mins[1] + (iy + 0.5) * gf.cell,
                    gf.mins[2] + (iz + 0.5) * gf.cell], axis=1)
    pot = grid[iz, iy, ix] / gf.reach_max

    out = {
        "map": Path(args.bsp).stem,
        "reach_max": round(float(gf.reach_max), 1),
        "stride_units": s * gf.cell,
        "pos": np.round(pos).astype(int).ravel().tolist(),
        "dir": np.round(d * 100).astype(int).ravel().tolist(),
        "pot": np.round(pot * 1000).astype(int).tolist(),
    }
    dst = ROOT / "viewer" / "assets" / f"{Path(args.bsp).stem}.goalfield.json"
    dst.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{len(pot):,} arrows ({s * gf.cell:.0f}u lattice, "
          f"<= {args.near_solid:.0f}u from geometry) -> {dst} "
          f"({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
