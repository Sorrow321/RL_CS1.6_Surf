"""GPU lidar — high-resolution depth vision via a precomputed SDF.

Exact per-ray BSP traces cost ~0.5us each on the CPU; at 128x64 rays x 2048
envs x 100Hz that is 1.7 billion traces/s — impossible. Instead:

1. Once per map: sample a solid-occupancy voxel grid through the C core
   (``SurfCore.occupancy_grid``), turn it into an unsigned distance field
   with ``scipy.ndimage.distance_transform_edt``, cache to
   ``maps/<map>.sdf_<cell>.npz``.
2. Per tick: sphere-trace all rays on the GPU in a fixed-iteration torch
   loop (nearest-neighbor SDF lookups, conservative 0.9x steps). 16.7M rays
   run in a few milliseconds — vision cost moves off the CPU entirely.

Depth error is on the order of the voxel size (default 16u) — fine for
perception; collision physics stays on the exact C tracer.

The camera convention matches src/surfcore.h write_lidar: row 0 looks up
(+vfov/2), col 0 looks left (+hfov/2), positive pitch = up, eye height
17 standing / 12 ducked, depth = distance/range clamped to 1.

``GpuLidar(surf_mask=True)`` renders a second channel — the hit surface's
|n_z| out of a per-voxel bake, :mod:`surfgym.surfmask`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                       # pragma: no cover
    HAVE_TRITON = False

__all__ = ["GpuLidar", "build_sdf", "map_occupancy", "slab_occupancy",
           "pick_cell", "grid_dims", "SOLID_ENT_CLASSES"]


MARCH_BLOCK = 64        # rays per program
MARCH_WARPS = 2         # = MARCH_BLOCK / 32, i.e. one ray per thread
# Both are measured, not guessed, and the surface is sharp: on a 5090,
# BLOCK 64 / 2 warps renders a 2048-env batch in 1.37 ms while BLOCK 64 /
# 4 warps takes 5.15 ms (half the threads idle and the block-wide alive
# reduction spans twice the warps). BLOCK 128/4 and 256/8 sit in a flatter
# 1.8-1.9 ms basin if this ever needs a safer default. Re-check with
# tools/proto_march.py on a new card.


if HAVE_TRITON:
    @triton.jit
    def _march_kernel(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
                      sdf_ptr, yoff_ptr, poff_ptr,
                      total, HW, W,
                      nx, ny, nz, stride_z, stride_y,
                      mnx, mny, mnz, inv_cell, cell, rng, near,
                      max_steps, BLOCK: tl.constexpr):
        # one lane per ray; a block covers a contiguous pixel patch, so lanes
        # diverge little and each ray stops loading memory once it has hit
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        n = offs // HW
        pix = offs % HW
        r = pix // W
        c = pix % W
        ex = tl.load(eye_ptr + n * 3 + 0, mask=m, other=0.0)
        ey = tl.load(eye_ptr + n * 3 + 1, mask=m, other=0.0)
        ez = tl.load(eye_ptr + n * 3 + 2, mask=m, other=0.0)
        dk = tl.load(duck_ptr + n, mask=m, other=0)
        ez += tl.where(dk != 0, 12.0, 17.0)
        yw = tl.load(yaw_ptr + n, mask=m, other=0.0) + tl.load(yoff_ptr + c, mask=m, other=0.0)
        pt = tl.load(pitch_ptr + n, mask=m, other=0.0) + tl.load(poff_ptr + r, mask=m, other=0.0)
        cp = tl.cos(pt)
        dx = cp * tl.cos(yw)
        dy = cp * tl.sin(yw)
        dz = tl.sin(pt)
        t = tl.zeros([BLOCK], tl.float32)
        alive = m
        hit_eps = 0.6 * cell
        min_step = 0.3 * cell
        # Early exit + a RUNTIME trip bound, both worth ~2x on their own and
        # 4.15x together (5.70 -> 1.37 ms per 2048-env batch), bit-exact.
        # Why they matter: the mean ray finishes in 9.9 steps and only 0.12%
        # of rays reach the 64th, but a constexpr `for` has no break, so every
        # block paid all 64 trips; and a constexpr bound gets fully unrolled,
        # which past ~32 trips costs more in register pressure than the work
        # (1.65 ms at 32 vs 4.90 at 48 — tools/bench_lidar.py).
        # Exactness: `alive` is monotone (only ever &='d) and a dead lane adds
        # 0 to t, so leaving the loop once every lane is dead cannot change an
        # output value. The depth encoding is warm-start ABI — it must not.
        k = 0
        while k < max_steps and tl.max(alive.to(tl.int32)) > 0:
            px = ex + dx * t
            py = ey + dy * t
            pz = ez + dz * t
            ix = tl.minimum(tl.maximum(((px - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
            iy = tl.minimum(tl.maximum(((py - mny) * inv_cell).to(tl.int64), 0), ny - 1)
            iz = tl.minimum(tl.maximum(((pz - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
            d = tl.load(sdf_ptr + iz * stride_z + iy * stride_y + ix,
                        mask=alive, other=0.0).to(tl.float32)
            alive = alive & (d > hit_eps) & (t < rng)
            t += tl.where(alive, tl.maximum(d * 0.9, min_step), 0.0)
            k += 1
        t = tl.minimum(t, rng)
        # near-linear + bounded far tail (near == rng -> legacy t/rng exactly)
        enc = tl.minimum(t, near) / near \
            + 0.25 * (1.0 - tl.exp(-tl.maximum(t - near, 0.0) / 2500.0))
        tl.store(out_ptr + offs, enc, mask=m)

    @triton.jit
    def _march_kernel_nz(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
                         sdf_ptr, snz_ptr, yoff_ptr, poff_ptr,
                         total, HW, W,
                         nx, ny, nz, stride_z, stride_y,
                         mnx, mny, mnz, inv_cell, cell, rng, near,
                         max_steps, BLOCK: tl.constexpr):
        """--surf-mask: the march above, emitting depth AND the hit
        surface's |n_z| (surfgym.surfmask).

        A deliberate copy rather than a constexpr flag on `_march_kernel`.
        The depth encoding is warm-start ABI — every existing checkpoint's
        conv trunk was trained on those exact pixel values — and
        tests/python/test_lidar_march.py pins the single-channel kernel
        bit-exact against a verbatim legacy copy. Leaving that source
        untouched is what keeps that guarantee free of this feature.

        Output is INTERLEAVED per pixel (depth, nz), i.e. NHWC with C
        fastest, because train_fast.py's forward_split turns the flat image
        row into a channels_last conv input by a pure restride. Two planes
        would make that a real transpose (docs/perf-results.md S9).
        """
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        n = offs // HW
        pix = offs % HW
        r = pix // W
        c = pix % W
        ex = tl.load(eye_ptr + n * 3 + 0, mask=m, other=0.0)
        ey = tl.load(eye_ptr + n * 3 + 1, mask=m, other=0.0)
        ez = tl.load(eye_ptr + n * 3 + 2, mask=m, other=0.0)
        dk = tl.load(duck_ptr + n, mask=m, other=0)
        ez += tl.where(dk != 0, 12.0, 17.0)
        yw = tl.load(yaw_ptr + n, mask=m, other=0.0) + tl.load(yoff_ptr + c, mask=m, other=0.0)
        pt = tl.load(pitch_ptr + n, mask=m, other=0.0) + tl.load(poff_ptr + r, mask=m, other=0.0)
        cp = tl.cos(pt)
        dx = cp * tl.cos(yw)
        dy = cp * tl.sin(yw)
        dz = tl.sin(pt)
        t = tl.zeros([BLOCK], tl.float32)
        alive = m
        hit_eps = 0.6 * cell
        min_step = 0.3 * cell
        k = 0
        while k < max_steps and tl.max(alive.to(tl.int32)) > 0:
            px = ex + dx * t
            py = ey + dy * t
            pz = ez + dz * t
            ix = tl.minimum(tl.maximum(((px - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
            iy = tl.minimum(tl.maximum(((py - mny) * inv_cell).to(tl.int64), 0), ny - 1)
            iz = tl.minimum(tl.maximum(((pz - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
            d = tl.load(sdf_ptr + iz * stride_z + iy * stride_y + ix,
                        mask=alive, other=0.0).to(tl.float32)
            alive = alive & (d > hit_eps) & (t < rng)
            t += tl.where(alive, tl.maximum(d * 0.9, min_step), 0.0)
            k += 1
        t = tl.minimum(t, rng)
        enc = tl.minimum(t, near) / near \
            + 0.25 * (1.0 - tl.exp(-tl.maximum(t - near, 0.0) / 2500.0))
        # hit_eps is under one cell and every air voxel's EDT is at least
        # one, so the march can only stop INSIDE a solid voxel: re-deriving
        # the index at the final t lands on the voxel the surface lives in.
        # A ray that ran out to `rng` stops in open air, where the grid is 0
        # — the same value as a wall, which the depth channel separates.
        px = ex + dx * t
        py = ey + dy * t
        pz = ez + dz * t
        ix = tl.minimum(tl.maximum(((px - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
        iy = tl.minimum(tl.maximum(((py - mny) * inv_cell).to(tl.int64), 0), ny - 1)
        iz = tl.minimum(tl.maximum(((pz - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
        snz = tl.load(snz_ptr + iz * stride_z + iy * stride_y + ix,
                      mask=m, other=0).to(tl.float32) * (1.0 / 127.0)
        tl.store(out_ptr + offs * 2 + 0, enc, mask=m)
        tl.store(out_ptr + offs * 2 + 1, snz, mask=m)


_SDF_BUILDER_VERSION = 2   # frozen in _map_sig's format — see below
_SDF_SEMANTICS = "s4"      # s4: NOTSOLID func_conveyors excluded from solids
_OCC_SEMANTICS = "o2"      # base occupancy content (o2: conveyor fix); the
                           # occ cache predates content tags and _map_sig
                           # alone cannot see a solid-set change in the C code


def _map_sig(bsp: Path) -> str:
    # NOTE: the "v2" prefix is a frozen FORMAT artifact — downstream caches
    # (the 30-min geodesic bake) embed this string in their signatures, so
    # it must stay stable; content versioning is appended per-cache instead
    st = bsp.stat()
    return f"v{_SDF_BUILDER_VERSION}_{st.st_size}_{st.st_mtime_ns}"


def pick_cell(core, budget_voxels: float = 700e6) -> float:
    """Smallest power-of-two voxel size (from 16u) whose grid fits the voxel
    budget. surf_ski_2 stays at its historical 16u; a source-port monster
    like surf_src_cannonball (29k x 28k x 27k units) lands on 32u — still
    fine for perception, where structures are hundreds of units wide."""
    mn, mx = core.map_bounds()
    cell = 16.0
    while cell < 256.0:
        pad = 4.0 * cell
        n = 1.0
        for e in (mx - mn) + 2.0 * pad:
            n *= np.ceil(e / cell)
        if n <= budget_voxels:
            break
        cell *= 2.0
    return cell


# brush entities physics collides with (src/bsp.c's solid set). Shared with
# the surfability bake: a trigger volume draped over a ramp must not become
# geometry in EITHER grid.
SOLID_ENT_CLASSES = {"func_wall", "func_breakable", "func_pushable",
                     "func_button", "func_train", "func_conveyor",
                     "func_wall_toggle", "func_rotating",
                     "func_door_rotating", "func_door"}


def grid_dims(core, cell: float):
    """The vision grid's geometry: (mins float64 (3,), nx, ny, nz).

    One definition on purpose. Every grid the march kernel indexes —
    occupancy, the SDF, the surfability mask — has to land on exactly these
    voxels, because one computed voxel index reads all of them. A half-cell
    disagreement would shift the mask off the depth image with no error."""
    mn, mx = core.map_bounds()
    pad = 4.0 * cell                       # margin so rays can leave cleanly
    mins = (mn - pad).astype(np.float64)
    ext = (mx + pad) - mins
    nx, ny, nz = (int(np.ceil(e / cell)) for e in ext)
    return mins, nx, ny, nz


def map_occupancy(core, cell: float = 16.0, cache_dir=None):
    """Sample (or load) the map's solid-occupancy voxel grid.

    Returns (occ uint8 ndarray [nz, ny, nx], mins float64 (3,)). Shared by
    the vision SDF and the race goal-distance field, so it caches separately
    (``maps/<map>.occ_<cell>.npz`` — mostly zeros, compresses tiny)."""
    bsp = Path(core.bsp_path)
    sig = f"{_map_sig(bsp)}_{_OCC_SEMANTICS}"
    cache = Path(cache_dir) if cache_dir else bsp.parent
    cache_file = cache / f"{bsp.stem}.occ_{cell:g}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sig" in z and str(z["sig"]) == sig:
            return z["occ"], z["mins"].astype(np.float64)

    mins, nx, ny, nz = grid_dims(core, cell)
    occ = core.occupancy_grid(mins, cell, nx, ny, nz)      # (nz, ny, nx)
    np.savez_compressed(cache_file, occ=occ, mins=mins, sig=np.str_(sig))
    return occ, mins


def slab_occupancy(core, cell: float = 16.0, cache_dir=None):
    """Occupancy that thin geometry cannot slip through.

    Two layers on top of the base voxel-center sampling:
    1. per-axis shifted samplings on a cell/4 lattice (OR-ed) — catches any
       slab >= ~cell/4 thick regardless of orientation;
    2. exact bbox rasterization of THIN solid brush entities (glass panes,
       skins; min bbox dimension <= 20u) parsed straight from the BSP —
       catches even 1u panes the lattice threads past. Thick entities stay
       lattice-sampled: a sloped conveyor ramp's bbox would wrongly
       solidify the air above it.

    Physics collides with these brushes (they are in the C solid set), so a
    depth sensor that misses them shows the agent free air where the world
    blocks — on surf_src_cannonball, 104 thin panes sit exactly where
    agents crash. Cached to ``maps/<map>.slabocc_<cell>.npz``."""
    bsp = Path(core.bsp_path)
    sig = f"{_map_sig(bsp)}_{_SDF_SEMANTICS}"
    cache = Path(cache_dir) if cache_dir else bsp.parent
    cache_file = cache / f"{bsp.stem}.slabocc_{cell:g}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sig" in z and str(z["sig"]) == sig:
            return z["occ"], z["mins"].astype(np.float64)

    occ, mins = map_occupancy(core, cell, cache_dir)
    occ = occ.copy()
    nz, ny, nx = occ.shape
    step = cell / 4.0
    for axis in range(3):
        for k in (-2, -1, 1, 2):
            off = np.zeros(3)
            off[axis] = k * step
            # shifting the grid origin samples center + off in every voxel
            occ |= core.occupancy_grid(mins + off, cell, nx, ny, nz)

    # thin solid entities: exact AABB rasterization (axis-aligned panes)
    from .zones import parse_bsp
    entities, bboxes = parse_bsp(bsp)
    n_thin = 0
    for ent in entities:
        model = ent.get("model", "")
        if (ent.get("classname") == "func_conveyor"
                and int(float(ent.get("spawnflags", 0))) & 2):
            continue   # SF_CONVEYOR_NOTSOLID: SOLID_NOT in GoldSrc (src/bsp.c)
        if ent.get("classname") in SOLID_ENT_CLASSES and model.startswith("*"):
            mi = int(model[1:])
            if mi >= len(bboxes):
                continue
            bmn, bmx = np.asarray(bboxes[mi][0]), np.asarray(bboxes[mi][1])
            if float((bmx - bmn).min()) > 20.0:
                continue
            lo = np.maximum(np.floor((bmn - mins) / cell - 0.001), 0).astype(int)
            hi = np.minimum(np.ceil((bmx - mins) / cell + 0.001),
                            [nx, ny, nz]).astype(int)
            occ[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = 1
            n_thin += 1
    if n_thin:
        print(f"slab occupancy: rasterized {n_thin} thin solid entities")
    np.savez_compressed(cache_file, occ=occ, mins=mins, sig=np.str_(sig))
    return occ, mins


def build_sdf(core, cell: float = 16.0, cache_dir=None):
    """Build (or load) the map's unsigned distance field.

    Returns (sdf ndarray [nz, ny, nx] in map units, mins (3,), cell).
    The cache is invalidated when the .bsp changes (size+mtime signature) or
    the builder semantics bump — a recompiled map must never serve stale
    geometry to vision while physics uses the new one.
    """
    from scipy.ndimage import distance_transform_edt

    bsp = Path(core.bsp_path)
    sig = f"{_map_sig(bsp)}_{_SDF_SEMANTICS}"
    cache = Path(cache_dir) if cache_dir else bsp.parent
    cache_file = cache / f"{bsp.stem}.sdf_{cell:g}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sig" in z and str(z["sig"]) == sig:
            return z["sdf"], z["mins"], float(z["cell"])

    occ, mins = slab_occupancy(core, cell, cache_dir)
    # outside the sampled box counts as solid (GoldSrc void), so pad with 1
    occ = np.pad(occ, 1, constant_values=1)
    dist = distance_transform_edt(occ == 0, sampling=cell)
    sdf = dist[1:-1, 1:-1, 1:-1]
    # f16 on disk: the render kernels gather f16 anyway, and at giant-map
    # grids (654M voxels for cannonball) f32 doubles a multi-GB artifact
    sdf = sdf.astype(np.float16 if sdf.size > 100e6 else np.float32)
    mins32 = mins.astype(np.float32)
    np.savez_compressed(cache_file, sdf=sdf, mins=mins32, cell=np.float32(cell),
                        sig=np.str_(sig))
    return sdf, mins32, cell


class GpuLidar:
    """Batched depth-image renderer over the cached SDF.

    ``render(origin, yaw_deg, pitch_deg, ducked) -> (N, H, W) float32`` on
    the module's device; values = hit distance / range, 1.0 = clear.

    ``surf_mask=True`` adds the hit surface's |n_z| as a second channel and
    renders (N, H, W, 2) instead, channel-fastest. Off it is byte-for-byte
    the depth-only renderer — same kernel, same buffers, no extra grid.
    """

    def __init__(self, core, width: int = 128, height: int = 64,
                 hfov_deg: float = 120.0, vfov_deg: float = 90.0,
                 range_units: float = 2000.0, cell: float = 16.0,
                 max_steps: int = 64, near_range: float = None,
                 device="cuda", surf_mask: bool = False) -> None:
        # Depth encoding: d/near_range within near_range (identical to the
        # legacy linear code, so warm-started nets keep their features), plus
        # a bounded tail 1 + 0.25*(1 - exp(-(d-near)/2500)) for the far field
        # — where legacy nets only ever saw a flat 1.0. near_range=None (or
        # == range) reproduces the legacy encoding exactly.
        sdf, mins, cell = build_sdf(core, cell)
        self.device = torch.device(device)
        # flat fp16 grid + integer strides: one fused gather per march step
        self.sdf_flat = torch.as_tensor(sdf, device=self.device) \
            .to(torch.float16).reshape(-1)
        self.nz, self.ny, self.nx = sdf.shape
        self.stride_z = self.ny * self.nx
        self.stride_y = self.nx
        self.mins = torch.as_tensor(mins, device=self.device)
        self.mins_f = (float(mins[0]), float(mins[1]), float(mins[2]))
        self.cell = float(cell)
        # second channel: |n_z| of the hit surface (surfgym.surfmask). Kept
        # allocation-free when off — the depth-only path is what every
        # existing checkpoint was trained on and must not move.
        self.surf_mask = bool(surf_mask)
        self.channels = 2 if self.surf_mask else 1
        if self.surf_mask:
            from .surfmask import build_surfnz
            snz, _ = build_surfnz(core, cell)
            if snz.shape != sdf.shape:
                raise RuntimeError(
                    f"surfability grid {snz.shape} != SDF grid {sdf.shape}: "
                    "the march reads both with ONE voxel index")
            self.snz_flat = torch.as_tensor(snz, device=self.device).reshape(-1)
        self.W, self.H = int(width), int(height)
        self.range = float(range_units)
        self.near = float(near_range) if near_range else self.range
        self.max_steps = int(max_steps)
        self._buf_n = 0                                          # lazy buffers
        d2r = np.pi / 180.0
        # per-pixel angular offsets (surfcore.h convention)
        yoff = (hfov_deg * (0.5 - (np.arange(self.W) + 0.5) / self.W)) * d2r
        poff = (vfov_deg * (0.5 - (np.arange(self.H) + 0.5) / self.H)) * d2r
        self.yoff = torch.as_tensor(yoff, dtype=torch.float32,
                                    device=self.device)          # (W,)
        self.poff = torch.as_tensor(poff, dtype=torch.float32,
                                    device=self.device)          # (H,)

    def _ensure_buffers(self, N):
        if self._buf_n == N:
            return
        self._buf_n = N
        sh = (N, self.H, self.W)
        self._dx = torch.empty(sh, device=self.device)
        self._dy = torch.empty(sh, device=self.device)
        self._dz = torch.empty(sh, device=self.device)
        self._t = torch.empty(sh, device=self.device)
        self._alive = torch.empty(sh, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def render(self, origin, yaw_deg, pitch_deg, ducked):
        """origin (N,3), yaw/pitch (N,) degrees, ducked (N,) bool/int ->
        (N, H, W) depths, or (N, H, W, 2) with --surf-mask. Triton kernel
        when available (per-ray early exit), else a lockstep torch sphere
        march."""
        if HAVE_TRITON and self.device.type == "cuda":
            return self._render_triton(origin, yaw_deg, pitch_deg, ducked)
        return self._render_torch(origin, yaw_deg, pitch_deg, ducked)

    @torch.no_grad()
    def _render_triton(self, origin, yaw_deg, pitch_deg, ducked):
        N = origin.shape[0]
        d2r = float(np.pi / 180.0)
        total = N * self.H * self.W
        BLOCK = MARCH_BLOCK
        if self.surf_mask:
            out = torch.empty(N, self.H, self.W, 2, device=self.device)
            _march_kernel_nz[(triton.cdiv(total, BLOCK),)](
                origin.contiguous(), (yaw_deg * d2r).contiguous(),
                (pitch_deg * d2r).contiguous(),
                ducked.to(torch.int32).contiguous(),
                out, self.sdf_flat, self.snz_flat, self.yoff, self.poff,
                total, self.H * self.W, self.W,
                self.nx, self.ny, self.nz, self.stride_z, self.stride_y,
                self.mins_f[0], self.mins_f[1], self.mins_f[2],
                1.0 / self.cell, self.cell, self.range, self.near,
                self.max_steps, BLOCK=BLOCK, num_warps=MARCH_WARPS)
            return out
        out = torch.empty(N, self.H, self.W, device=self.device)
        _march_kernel[(triton.cdiv(total, BLOCK),)](
            origin.contiguous(), (yaw_deg * d2r).contiguous(),
            (pitch_deg * d2r).contiguous(), ducked.to(torch.int32).contiguous(),
            out, self.sdf_flat, self.yoff, self.poff,
            total, self.H * self.W, self.W,
            self.nx, self.ny, self.nz, self.stride_z, self.stride_y,
            self.mins_f[0], self.mins_f[1], self.mins_f[2],
            1.0 / self.cell, self.cell, self.range, self.near,
            self.max_steps, BLOCK=BLOCK, num_warps=MARCH_WARPS)
        return out

    @torch.no_grad()
    def _render_torch(self, origin, yaw_deg, pitch_deg, ducked):
        N = origin.shape[0]
        self._ensure_buffers(N)
        d2r = np.pi / 180.0
        ex = origin[:, 0].view(N, 1, 1)
        ey = origin[:, 1].view(N, 1, 1)
        ez = (origin[:, 2] + torch.where(ducked.bool(), 12.0, 17.0)).view(N, 1, 1)
        p = pitch_deg.view(N, 1, 1) * d2r + self.poff.view(1, self.H, 1)
        y = yaw_deg.view(N, 1, 1) * d2r + self.yoff.view(1, 1, self.W)
        cp = torch.cos(p)
        torch.mul(cp, torch.cos(y), out=self._dx)
        torch.mul(cp, torch.sin(y), out=self._dy)
        self._dz.copy_(torch.sin(p).expand_as(self._dz))
        t, alive = self._t, self._alive
        t.zero_()
        alive.fill_(True)
        hit_eps = 0.6 * self.cell
        min_step = 0.3 * self.cell
        inv_cell = 1.0 / self.cell
        mx, my, mz = self.mins[0], self.mins[1], self.mins[2]
        for it in range(self.max_steps):
            ix = ((ex + self._dx * t - mx) * inv_cell).long().clamp_(0, self.nx - 1)
            iy = ((ey + self._dy * t - my) * inv_cell).long().clamp_(0, self.ny - 1)
            iz = ((ez + self._dz * t - mz) * inv_cell).long().clamp_(0, self.nz - 1)
            d = self.sdf_flat[(iz * self.stride_z + iy * self.stride_y + ix)
                              .reshape(-1)].reshape(N, self.H, self.W).float()
            alive &= (d > hit_eps) & (t < self.range)
            t.add_(torch.clamp(d * 0.9, min=min_step) * alive)
            # early exit costs one host sync — check once per chunk, not per step
            if (it & 7) == 7 and not alive.any():
                break
        t = torch.clamp(t, max=self.range)
        enc = (torch.clamp(t, max=self.near) / self.near
               + 0.25 * (1.0 - torch.exp(-torch.clamp(t - self.near, min=0.0)
                                         / 2500.0)))
        if not self.surf_mask:
            return enc
        # same hit voxel the triton path re-derives, same interleaving
        ix = ((ex + self._dx * t - mx) * inv_cell).long().clamp_(0, self.nx - 1)
        iy = ((ey + self._dy * t - my) * inv_cell).long().clamp_(0, self.ny - 1)
        iz = ((ez + self._dz * t - mz) * inv_cell).long().clamp_(0, self.nz - 1)
        snz = self.snz_flat[(iz * self.stride_z + iy * self.stride_y + ix)
                            .reshape(-1)].reshape(N, self.H, self.W)
        return torch.stack((enc, snz.float() / 127.0), dim=-1)
