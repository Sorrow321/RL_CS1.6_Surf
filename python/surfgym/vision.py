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

__all__ = ["GpuLidar", "build_sdf"]


if HAVE_TRITON:
    @triton.jit
    def _march_kernel(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
                      sdf_ptr, yoff_ptr, poff_ptr,
                      total, HW, W,
                      nx, ny, nz, stride_z, stride_y,
                      mnx, mny, mnz, inv_cell, cell, rng, near,
                      MAX_STEPS: tl.constexpr, BLOCK: tl.constexpr):
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
        for _ in range(MAX_STEPS):
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
        t = tl.minimum(t, rng)
        # near-linear + bounded far tail (near == rng -> legacy t/rng exactly)
        enc = tl.minimum(t, near) / near \
            + 0.25 * (1.0 - tl.exp(-tl.maximum(t - near, 0.0) / 2500.0))
        tl.store(out_ptr + offs, enc, mask=m)


_SDF_BUILDER_VERSION = 2   # bump when occupancy semantics change


def build_sdf(core, cell: float = 16.0, cache_dir=None):
    """Build (or load) the map's unsigned distance field.

    Returns (sdf float32 ndarray [nz, ny, nx] in map units, mins (3,), cell).
    The cache is invalidated when the .bsp changes (size+mtime signature) or
    the builder version bumps — a recompiled map must never serve stale
    geometry to vision while physics uses the new one.
    """
    from scipy.ndimage import distance_transform_edt

    bsp = Path(core.bsp_path)
    st = bsp.stat()
    sig = f"v{_SDF_BUILDER_VERSION}_{st.st_size}_{st.st_mtime_ns}"
    cache = Path(cache_dir) if cache_dir else bsp.parent
    cache_file = cache / f"{bsp.stem}.sdf_{cell:g}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sig" in z and str(z["sig"]) == sig:
            return z["sdf"], z["mins"], float(z["cell"])

    mn, mx = core.map_bounds()
    pad = 4.0 * cell                       # margin so rays can leave cleanly
    mins = (mn - pad).astype(np.float64)
    ext = (mx + pad) - mins
    nx, ny, nz = (int(np.ceil(e / cell)) for e in ext)
    occ = core.occupancy_grid(mins, cell, nx, ny, nz)      # (nz, ny, nx)
    # outside the sampled box counts as solid (GoldSrc void), so pad with 1
    occ = np.pad(occ, 1, constant_values=1)
    dist = distance_transform_edt(occ == 0, sampling=cell).astype(np.float32)
    sdf = dist[1:-1, 1:-1, 1:-1]
    mins32 = mins.astype(np.float32)
    np.savez_compressed(cache_file, sdf=sdf, mins=mins32, cell=np.float32(cell),
                        sig=np.str_(sig))
    return sdf, mins32, cell


class GpuLidar:
    """Batched depth-image renderer over the cached SDF.

    ``render(origin, yaw_deg, pitch_deg, ducked) -> (N, H, W) float32`` on
    the module's device; values = hit distance / range, 1.0 = clear.
    """

    def __init__(self, core, width: int = 128, height: int = 64,
                 hfov_deg: float = 120.0, vfov_deg: float = 90.0,
                 range_units: float = 2000.0, cell: float = 16.0,
                 max_steps: int = 64, near_range: float = None,
                 device="cuda") -> None:
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
        (N, H, W) depths. Triton kernel when available (per-ray early exit),
        else a lockstep torch sphere march."""
        if HAVE_TRITON and self.device.type == "cuda":
            return self._render_triton(origin, yaw_deg, pitch_deg, ducked)
        return self._render_torch(origin, yaw_deg, pitch_deg, ducked)

    @torch.no_grad()
    def _render_triton(self, origin, yaw_deg, pitch_deg, ducked):
        N = origin.shape[0]
        d2r = float(np.pi / 180.0)
        out = torch.empty(N, self.H, self.W, device=self.device)
        total = N * self.H * self.W
        BLOCK = 256
        _march_kernel[(triton.cdiv(total, BLOCK),)](
            origin.contiguous(), (yaw_deg * d2r).contiguous(),
            (pitch_deg * d2r).contiguous(), ducked.to(torch.int32).contiguous(),
            out, self.sdf_flat, self.yoff, self.poff,
            total, self.H * self.W, self.W,
            self.nx, self.ny, self.nz, self.stride_z, self.stride_y,
            self.mins_f[0], self.mins_f[1], self.mins_f[2],
            1.0 / self.cell, self.cell, self.range, self.near,
            MAX_STEPS=self.max_steps, BLOCK=BLOCK)
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
        return (torch.clamp(t, max=self.near) / self.near
                + 0.25 * (1.0 - torch.exp(-torch.clamp(t - self.near, min=0.0)
                                          / 2500.0)))
