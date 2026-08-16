"""surfmask.py — per-voxel surfability (|n_z|), the lidar's second channel.

The depth image says where geometry is; a surf agent also has to know what
it is. Floor, rideable ramp and wall are the same pixel to a depth sensor,
and they are three different futures: the physics calls a face walkable at
n_z >= 0.7 (src/pm.c), a surf ramp lives in the ~0.3-0.7 band, and a wall
at 0 only ever costs speed.

This cannot come from the SDF. That field is an unsigned NEAREST-VOXEL EDT,
so on a voxelized ramp its numerical gradient is a 0/1 staircase — central
differences there return a checkerboard, not a slope. So bake the normals
from the source geometry instead: rasterize the exported triangle mesh
(``viewer/assets/<map>.mesh.json``, what the viewer draws) into the SAME
voxel grid the march already indexes, storing |n_z|*127 as int8. The kernel
then gets the normal out of the gather it was going to do anyway.

Two rules the values follow:

* per voxel, the LARGEST-AREA triangle touching it wins. A voxel on a ramp
  usually also clips the trim brush bolted to its edge; the agent cares
  about the surface it will land on, not the decoration.
* |n_z|, not n_z. Winding decides a face's normal sign and both sides of a
  thin brush are exported, so only the magnitude is well defined. Ceilings
  therefore read like floors — harmless, since the depth channel already
  places the pixel above the eye.

0 means "wall" AND "no surface here" (open sky). Again the depth channel
separates them: a sky pixel is at max range.

Known limit, measured: the mesh is the VISIBLE geometry. A solid brush the
compiler left no faces for is invisible to this bake and renders 0, however
ridable it is. On surf_ski_2 that is real — some world ramps have no BSP
face within 200u of where the C tracer hits them — and those pixels read
"wall". On surf_src_cannonball, the map the trainer runs, sampled ramp views
agree with the tracer's own normal 98.7% of the time (99.8% on the surfable
band); tests/python/test_surfmask.py holds both ends of this.

Cached to ``maps/<map>.surfnz_<cell>.npz``. Pure numpy; bake it once per
(map, cell) with ``python -m surfgym.surfmask maps/<map>.bsp``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .vision import SOLID_ENT_CLASSES, _map_sig, grid_dims

__all__ = ["build_surfnz", "load_solid_tris", "rasterize_surfnz"]

ROOT = Path(__file__).resolve().parents[2]

# n1: |n_z|*127 int8, largest-area triangle wins
# n2: exact plane/box claims beat merely-nearby ones (see rasterize_surfnz)
_SURFNZ_SEMANTICS = "n2"

# bounds the scratch block one triangle allocates. Only a map-spanning
# triangle with no small axis can reach it, but that one would otherwise ask
# for the whole grid in float64.
_CHUNK_VOXELS = 4_000_000


def _notsolid_conveyor_models(bsp_path):
    """Brush-model indices of SF_CONVEYOR_NOTSOLID conveyors.

    They are SOLID_NOT in GoldSrc (src/bsp.c), so physics and the occupancy
    grid both ignore them — the mesh export does not, and it carries no
    spawnflags. The BSP entities do."""
    from .zones import parse_bsp
    ents, _ = parse_bsp(Path(bsp_path))
    out = set()
    for ent in ents:
        model = ent.get("model", "")
        if (ent.get("classname") == "func_conveyor" and model.startswith("*")
                and int(float(ent.get("spawnflags", 0))) & 2):
            out.add(int(model[1:]))
    return out


def load_solid_tris(mesh_path, bsp_path=None):
    """Every SOLID triangle of the exported mesh as (T, 3, 3) float64.

    World geometry plus the brush entities physics collides with. Triggers,
    teleports and func_illusionary are dropped on purpose: 109 of
    cannonball's 234 brush models are non-solid volumes draped over the
    track, and a trigger face within a voxel of the ramp under it would
    overwrite the ramp's slope with its own.

    ``bsp_path`` is only read when the mesh actually contains conveyors —
    it exists solely to resolve their NOTSOLID spawnflag."""
    doc = json.loads(Path(mesh_path).read_text(encoding="utf-8"))
    brushes = doc.get("brushes") or []
    notsolid = ()
    if bsp_path and any(b.get("classname") == "func_conveyor" for b in brushes):
        notsolid = _notsolid_conveyor_models(bsp_path)
    parts = [doc["world"]]
    parts += [b for b in brushes
              if b.get("classname") in SOLID_ENT_CLASSES
              and b.get("model") not in notsolid]
    tris = []
    for part in parts:
        idx = np.asarray(part["indices"], dtype=np.int64)
        if not idx.size:
            continue
        pos = np.asarray(part["positions"], dtype=np.float64).reshape(-1, 3)
        tris.append(pos[idx].reshape(-1, 3, 3))
    if not tris:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.concatenate(tris)


def rasterize_surfnz(tris, mins, cell: float, nx: int, ny: int, nz: int):
    """Voxelize |n_z|*127 into a (nz, ny, nx) int8 grid. 0 = no surface.

    Conservative on purpose, in two passes over every triangle:

    * MARGIN — claim every voxel of the AABB (dilated one cell) whose CENTRE
      is within ``0.5*cell*sum|n| + 0.5*cell`` of the plane. The first term
      is the exact plane/box overlap radius; the half cell on top is not
      cosmetic. The march stops in the first voxel the (dilated) occupancy
      calls solid, which can sit most of a cell off the true face, and those
      pixels would otherwise read 0 = "wall" on open floor.
    * EXACT — the same claim at the tight radius, written ON TOP.

    Two passes because one is measurably wrong. A surf ramp runs along the
    arena wall, and the wall's triangles are an order of magnitude larger:
    in a single pass the wall's MARGIN outvoted the ramp face the voxel
    actually contains, and 29% of ramp pixels rendered as wall against the C
    tracer (tests/python/test_surfmask.py measures this end to end). A
    triangle passing through a voxel now always beats one merely near it.

    Within a pass, triangles are written smallest-area first, so the
    largest-area claimant is the one left standing — a voxel on a ramp
    usually also clips the trim brush bolted to its edge. The pass ordering
    is carried in the sign of the stored byte (``-(v+1)`` = written by the
    exact pass, unfolded at the end) rather than in a parallel array: on a
    690M-voxel map a float32 area grid would add 2.7 GB to the bake's RSS.

    In-plane the AABB is not clipped to the triangle. That over-claims the
    corners of a diagonal face — where its coplanar fan neighbours write the
    same value anyway.
    """
    grid = np.zeros((nz, ny, nx), dtype=np.int8)
    tris = np.asarray(tris, dtype=np.float64)
    if tris.size == 0:
        return grid
    mins = np.asarray(mins, dtype=np.float64)
    dims = np.array([nx, ny, nz], dtype=np.int64)

    cr = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    twice_area = np.linalg.norm(cr, axis=1)
    live = twice_area > 1e-9            # degenerate fan triangles carry no face
    tris, cr, twice_area = tris[live], cr[live], twice_area[live]
    if not len(tris):
        return grid
    n = cr / twice_area[:, None]
    val = np.rint(np.abs(n[:, 2]) * 127.0).astype(np.int8)
    exact_val = (-(val.astype(np.int16) + 1)).astype(np.int8)   # -128..-1
    tight = 0.5 * cell * np.abs(n).sum(axis=1)
    plane_d = np.einsum("ij,ij->i", n, tris[:, 0])
    order = np.argsort(twice_area, kind="stable")

    for slab, write in ((tight + 0.5 * cell, val), (tight, exact_val)):
        for k in order:
            lo = np.clip(np.floor((tris[k].min(axis=0) - cell - mins) / cell)
                         .astype(np.int64), 0, dims)
            hi = np.clip(np.floor((tris[k].max(axis=0) + cell - mins) / cell)
                         .astype(np.int64) + 1, 0, dims)
            span = hi - lo
            if not span.all():
                continue                # entirely outside the grid
            # centre coordinates projected on the normal, per axis: the plane
            # distance of a voxel is then one broadcast add
            proj = [((np.arange(lo[a], hi[a]) + 0.5) * cell + mins[a]) * n[k, a]
                    for a in range(3)]
            rows = max(1, int(_CHUNK_VOXELS // max(1, span[0] * span[1])))
            for z0 in range(lo[2], hi[2], rows):
                z1 = min(z0 + rows, int(hi[2]))
                dist = (proj[2][z0 - lo[2]:z1 - lo[2], None, None]
                        + proj[1][None, :, None] + proj[0][None, None, :]
                        - plane_d[k])
                blk = grid[z0:z1, lo[1]:hi[1], lo[0]:hi[0]]
                blk[np.abs(dist) <= slab[k]] = write[k]

    for z0 in range(0, nz, 64):         # unfold the exact pass's sign
        sl = grid[z0:z0 + 64]
        neg = sl < 0
        sl[neg] = (-(sl[neg].astype(np.int16)) - 1).astype(np.int8)
    return grid


def build_surfnz(core, cell: float = 16.0, cache_dir=None, mesh_path=None):
    """Build (or load) the map's per-voxel surfability grid.

    Returns (nz_grid int8 [nz, ny, nx], mins float32 (3,)) on exactly the
    vision SDF's grid — same mins, same cell, same dims, so one voxel index
    reads both. The signature covers the .bsp (which fixes the grid), the
    exported mesh (the source geometry) and the bake's content tag: a
    re-exported mesh must never serve stale normals next to a fresh SDF.
    """
    bsp = Path(core.bsp_path)
    mesh = Path(mesh_path) if mesh_path else \
        ROOT / "viewer" / "assets" / f"{bsp.stem}.mesh.json"
    if not mesh.exists():
        raise FileNotFoundError(
            f"{mesh}: the surfability mask is baked from the exported mesh — "
            f"run `python tools/export_map.py {bsp}` first")
    mst = mesh.stat()
    sig = (f"{_map_sig(bsp)}_m{mst.st_size}_{mst.st_mtime_ns}"
           f"_{_SURFNZ_SEMANTICS}")
    cache = Path(cache_dir) if cache_dir else bsp.parent
    cache_file = cache / f"{bsp.stem}.surfnz_{cell:g}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sig" in z and str(z["sig"]) == sig:
            return z["nz"], z["mins"].astype(np.float32)

    mins, nx, ny, nz = grid_dims(core, cell)
    tris = load_solid_tris(mesh, bsp)
    grid = rasterize_surfnz(tris, mins, cell, nx, ny, nz)
    mins32 = mins.astype(np.float32)
    np.savez_compressed(cache_file, nz=grid, mins=mins32,
                        cell=np.float32(cell), sig=np.str_(sig))
    hit = int(np.count_nonzero(grid))
    walk = int(np.count_nonzero(grid >= 89))          # n_z >= 0.7, src/pm.c
    print(f"surfnz: {len(tris)} solid triangles -> {hit:,} surfaced voxels "
          f"of {grid.size:,} ({walk:,} walkable)")
    return grid, mins32


def main() -> None:
    import argparse

    from . import SurfCore, default_config
    from .vision import pick_cell

    ap = argparse.ArgumentParser(
        description="bake the per-voxel surfability grid for a map")
    ap.add_argument("bsp")
    ap.add_argument("--cell", type=float, default=None,
                    help="voxel size, units (default: vision.pick_cell — it "
                         "MUST match the SDF the lidar renders with)")
    ap.add_argument("--mesh", default=None,
                    help="mesh export (default viewer/assets/<map>.mesh.json)")
    args = ap.parse_args()

    core = SurfCore(args.bsp, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    cell = args.cell or pick_cell(core)
    grid, mins = build_surfnz(core, cell, mesh_path=args.mesh)
    surf = grid[grid > 0].astype(np.float32) / 127.0
    print(f"cell {cell:g}u  grid {grid.shape}  mins {mins}")
    if surf.size:
        edges = [0.0, 0.1, 0.3, 0.7, 0.95, 1.01]
        names = ["wall", "steep", "surfable", "walkable", "flat"]
        counts = np.histogram(surf, bins=edges)[0]
        for name, c in zip(names, counts):
            print(f"  {name:9s} {c:12,}  {100.0 * c / surf.size:5.1f}%")


if __name__ == "__main__":
    main()
