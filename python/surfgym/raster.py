"""Pinhole triangle rasterizer - the alternative to sphere-tracing the SDF.

WHY: the shipped renderer marches a precomputed SDF voxel grid. Its cost is
O(pixels x steps) and independent of scene complexity, but it needs a per-map
bake (``maps/<map>.sdf_<cell>.npz``), quantizes depth to the voxel size, and
needs a SECOND bake (``surfmask.build_surfnz``) to know a surface's normal. A
rasterizer has none of those problems: exact depth, true triangle normals, no
precompute. Its cost is O(triangles x cameras), and this project renders
2,048 cameras of 2,048 pixels each - so which one wins is an empirical
question, and this module exists to answer it.

CONVENTION - matches ``vision._march_kernel_pin`` exactly, so the two are
directly comparable::

    eye     = origin + (12 if ducked else 17) in z
    forward = (cos p cos y, cos p sin y, sin p)
    right   = (sin y, -cos y, 0)
    up      = (-sin p cos y, -sin p sin y, cos p)
    ray(r,c) = forward + u[c]*right + v[r]*up      (not normalized)
    u[c] = tan(hfov/2) * (2c/(W-1) - 1)
    v[r] = tan(vfov/2) * (1 - 2r/(H-1))

The march reports RADIAL distance t along the normalized ray. The basis is
orthonormal, so a point at camera-space depth z_c on that pixel's ray has
``t = z_c * sqrt(1 + u^2 + v^2)``. That identity is what makes rasterized
depth comparable to marched depth without a second projection.

APPROXIMATION, stated up front: a triangle is rasterized over a bounded
bbox (``maxbox`` square). Anything larger is CLIPPED, and
``last_clipped_frac`` reports how often - so a timing number is never quoted
without the coverage it was measured at. Triangles crossing the near plane
are dropped rather than clipped.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                    # pragma: no cover
    HAVE_TRITON = False

RASTER_BLOCK = 64
_NEAR_EPS = 1.0
if HAVE_TRITON:
    # a @jit'ed kernel may only read constexpr globals
    NEAR_EPS = tl.constexpr(_NEAR_EPS)


if HAVE_TRITON:
    @triton.jit
    def _raster_kernel(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr,
                       tri_ptr, depth_ptr, clip_ptr,
                       n_tri, HW, W, H, tu, tv, far,
                       total, MAXBOX: tl.constexpr, BLOCK: tl.constexpr):
        """One lane per (camera, triangle): project, then scatter an
        atomic-min depth over the triangle's bounding box.

        Depth is stored as the float32 BIT PATTERN in an int32 buffer: for
        non-negative floats the IEEE ordering is monotonic, so an integer
        atomic_min IS a float min. Triton has no float atomic_min.
        """
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        m = idx < total
        n = idx // n_tri
        t = idx % n_tri

        ex = tl.load(eye_ptr + n * 3 + 0, mask=m, other=0.0)
        ey = tl.load(eye_ptr + n * 3 + 1, mask=m, other=0.0)
        ez = tl.load(eye_ptr + n * 3 + 2, mask=m, other=0.0)
        dk = tl.load(duck_ptr + n, mask=m, other=0)
        ez += tl.where(dk != 0, 12.0, 17.0)
        yw = tl.load(yaw_ptr + n, mask=m, other=0.0)
        pt = tl.load(pitch_ptr + n, mask=m, other=0.0)
        cy = tl.cos(yw)
        sy = tl.sin(yw)
        cp = tl.cos(pt)
        sp = tl.sin(pt)
        fx = cp * cy
        fy = cp * sy
        fz = sp
        rx = sy
        ry = -cy
        ux = -sp * cy
        uy = -sp * sy
        uz = cp

        base = t * 9
        ok = m
        z0 = tl.zeros([BLOCK], tl.float32)
        z1 = tl.zeros([BLOCK], tl.float32)
        z2 = tl.zeros([BLOCK], tl.float32)
        a0 = tl.zeros([BLOCK], tl.float32)
        b0 = tl.zeros([BLOCK], tl.float32)
        a1 = tl.zeros([BLOCK], tl.float32)
        b1 = tl.zeros([BLOCK], tl.float32)
        a2 = tl.zeros([BLOCK], tl.float32)
        b2 = tl.zeros([BLOCK], tl.float32)
        for j in tl.static_range(3):
            wx = tl.load(tri_ptr + base + j * 3 + 0, mask=m, other=0.0) - ex
            wy = tl.load(tri_ptr + base + j * 3 + 1, mask=m, other=0.0) - ey
            wz = tl.load(tri_ptr + base + j * 3 + 2, mask=m, other=0.0) - ez
            cz = wx * fx + wy * fy + wz * fz
            cx = wx * rx + wy * ry
            cyy = wx * ux + wy * uy + wz * uz
            ok = ok & (cz > NEAR_EPS)
            invz = 1.0 / tl.where(cz > NEAR_EPS, cz, 1.0)
            px = (cx * invz / tu + 1.0) * 0.5 * (W - 1).to(tl.float32)
            py = (1.0 - cyy * invz / tv) * 0.5 * (H - 1).to(tl.float32)
            if j == 0:
                a0 = px
                b0 = py
                z0 = cz
            elif j == 1:
                a1 = px
                b1 = py
                z1 = cz
            else:
                a2 = px
                b2 = py
                z2 = cz

        area = (a1 - a0) * (b2 - b0) - (a2 - a0) * (b1 - b0)
        ok = ok & (tl.abs(area) > 1e-9)
        inv_area = 1.0 / tl.where(tl.abs(area) > 1e-9, area, 1.0)

        lox = tl.minimum(tl.minimum(a0, a1), a2)
        hix = tl.maximum(tl.maximum(a0, a1), a2)
        loy = tl.minimum(tl.minimum(b0, b1), b2)
        hiy = tl.maximum(tl.maximum(b0, b1), b2)
        fW = (W - 1).to(tl.float32)
        fH = (H - 1).to(tl.float32)
        x0 = tl.maximum(tl.ceil(lox), 0.0).to(tl.int32)
        y0 = tl.maximum(tl.ceil(loy), 0.0).to(tl.int32)
        x1 = tl.minimum(tl.floor(hix), fW).to(tl.int32)
        y1 = tl.minimum(tl.floor(hiy), fH).to(tl.int32)
        ok = ok & (x1 >= x0) & (y1 >= y0)
        clipped = ok & ((x1 - x0 >= MAXBOX) | (y1 - y0 >= MAXBOX))
        tl.atomic_add(clip_ptr, tl.sum(clipped.to(tl.int32), axis=0))
        x1 = tl.minimum(x1, x0 + MAXBOX - 1)
        y1 = tl.minimum(y1, y0 + MAXBOX - 1)

        for dy in tl.static_range(MAXBOX):
            yy = y0 + dy
            iny = ok & (yy <= y1)
            fy_ = yy.to(tl.float32)
            vv = (1.0 - 2.0 * fy_ / fH) * tv
            for dx in tl.static_range(MAXBOX):
                xx = x0 + dx
                act = iny & (xx <= x1)
                fx_ = xx.to(tl.float32)
                w0 = ((a1 - fx_) * (b2 - fy_) - (a2 - fx_) * (b1 - fy_)) * inv_area
                w1 = ((a2 - fx_) * (b0 - fy_) - (a0 - fx_) * (b2 - fy_)) * inv_area
                w2 = 1.0 - w0 - w1
                act = act & (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
                invz = w0 / z0 + w1 / z1 + w2 / z2
                act = act & (invz > 0.0)
                cz = 1.0 / tl.where(invz > 0.0, invz, 1.0)
                uu = (2.0 * fx_ / fW - 1.0) * tu
                rad = cz * tl.sqrt(1.0 + uu * uu + vv * vv)
                rad = tl.minimum(rad, far)
                bits = rad.to(tl.int32, bitcast=True)
                off = n * HW + yy * W + xx
                tl.atomic_min(depth_ptr + off, bits, mask=act)


if HAVE_TRITON:
    @triton.jit
    def _raster_kernel_exact(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr,
                             tri_ptr, depth_ptr,
                             n_tri, HW, W, H, tu, tv, far,
                             total, BLOCK: tl.constexpr):
        """Same projection, but the bbox loop is RUNTIME-bounded, so nothing
        is clipped and coverage is complete.

        The cost of that: the loop runs to the widest bbox in the BLOCK, so
        one nearby wall triangle drags its whole block to full-screen trip
        count. That is the honest naive cost of rasterizing 2,048 cameras
        without tile binning, and it is why the bounded kernel exists to be
        compared against.
        """
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        m = idx < total
        n = idx // n_tri
        t = idx % n_tri
        ex = tl.load(eye_ptr + n * 3 + 0, mask=m, other=0.0)
        ey = tl.load(eye_ptr + n * 3 + 1, mask=m, other=0.0)
        ez = tl.load(eye_ptr + n * 3 + 2, mask=m, other=0.0)
        dk = tl.load(duck_ptr + n, mask=m, other=0)
        ez += tl.where(dk != 0, 12.0, 17.0)
        yw = tl.load(yaw_ptr + n, mask=m, other=0.0)
        pt = tl.load(pitch_ptr + n, mask=m, other=0.0)
        cy = tl.cos(yw)
        sy = tl.sin(yw)
        cp = tl.cos(pt)
        sp = tl.sin(pt)
        fx = cp * cy
        fy = cp * sy
        fz = sp
        ux = -sp * cy
        uy = -sp * sy
        uz = cp
        base = t * 9
        ok = m
        z0 = tl.zeros([BLOCK], tl.float32)
        z1 = tl.zeros([BLOCK], tl.float32)
        z2 = tl.zeros([BLOCK], tl.float32)
        a0 = tl.zeros([BLOCK], tl.float32)
        b0 = tl.zeros([BLOCK], tl.float32)
        a1 = tl.zeros([BLOCK], tl.float32)
        b1 = tl.zeros([BLOCK], tl.float32)
        a2 = tl.zeros([BLOCK], tl.float32)
        b2 = tl.zeros([BLOCK], tl.float32)
        for j in tl.static_range(3):
            wx = tl.load(tri_ptr + base + j * 3 + 0, mask=m, other=0.0) - ex
            wy = tl.load(tri_ptr + base + j * 3 + 1, mask=m, other=0.0) - ey
            wz = tl.load(tri_ptr + base + j * 3 + 2, mask=m, other=0.0) - ez
            cz = wx * fx + wy * fy + wz * fz
            cx = wx * sy - wy * cy
            cyy = wx * ux + wy * uy + wz * uz
            ok = ok & (cz > NEAR_EPS)
            invz = 1.0 / tl.where(cz > NEAR_EPS, cz, 1.0)
            px = (cx * invz / tu + 1.0) * 0.5 * (W - 1).to(tl.float32)
            py = (1.0 - cyy * invz / tv) * 0.5 * (H - 1).to(tl.float32)
            if j == 0:
                a0 = px
                b0 = py
                z0 = cz
            elif j == 1:
                a1 = px
                b1 = py
                z1 = cz
            else:
                a2 = px
                b2 = py
                z2 = cz
        area = (a1 - a0) * (b2 - b0) - (a2 - a0) * (b1 - b0)
        ok = ok & (tl.abs(area) > 1e-9)
        inv_area = 1.0 / tl.where(tl.abs(area) > 1e-9, area, 1.0)
        fW = (W - 1).to(tl.float32)
        fH = (H - 1).to(tl.float32)
        x0 = tl.maximum(tl.ceil(tl.minimum(tl.minimum(a0, a1), a2)), 0.0).to(tl.int32)
        y0 = tl.maximum(tl.ceil(tl.minimum(tl.minimum(b0, b1), b2)), 0.0).to(tl.int32)
        x1 = tl.minimum(tl.floor(tl.maximum(tl.maximum(a0, a1), a2)), fW).to(tl.int32)
        y1 = tl.minimum(tl.floor(tl.maximum(tl.maximum(b0, b1), b2)), fH).to(tl.int32)
        ok = ok & (x1 >= x0) & (y1 >= y0)
        yy = x0 * 0 + y0
        while tl.max((ok & (yy <= y1)).to(tl.int32)) > 0:
            iny = ok & (yy <= y1)
            fy_ = yy.to(tl.float32)
            vv = (1.0 - 2.0 * fy_ / fH) * tv
            xx = x0 + 0
            while tl.max((iny & (xx <= x1)).to(tl.int32)) > 0:
                act = iny & (xx <= x1)
                fx_ = xx.to(tl.float32)
                w0 = ((a1 - fx_) * (b2 - fy_) - (a2 - fx_) * (b1 - fy_)) * inv_area
                w1 = ((a2 - fx_) * (b0 - fy_) - (a0 - fx_) * (b2 - fy_)) * inv_area
                w2 = 1.0 - w0 - w1
                act = act & (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
                invz = w0 / z0 + w1 / z1 + w2 / z2
                act = act & (invz > 0.0)
                cz = 1.0 / tl.where(invz > 0.0, invz, 1.0)
                uu = (2.0 * fx_ / fW - 1.0) * tu
                rad = tl.minimum(cz * tl.sqrt(1.0 + uu * uu + vv * vv), far)
                tl.atomic_min(depth_ptr + n * HW + yy * W + xx,
                              rad.to(tl.int32, bitcast=True), mask=act)
                xx = xx + 1
            yy = yy + 1


class Rasterizer:
    """Pinhole depth rasterizer over the map's triangle soup.

    ``render`` returns the SAME encoding as :class:`surfgym.vision.GpuLidar`,
    so the two images can be differenced pixel for pixel.
    """

    def __init__(self, tris, width: int = 64, height: int = 32,
                 hfov_deg: float = 120.0, vfov_deg: float = 90.0,
                 range_units: float = 11500.0, near_range=None,
                 device="cuda", maxbox: int = 8, exact: bool = False):
        self.exact = bool(exact)
        if not HAVE_TRITON:
            raise RuntimeError("the rasterizer needs triton")
        t = np.ascontiguousarray(np.asarray(tris, np.float32).reshape(-1, 3, 3))
        self.n_tri = int(t.shape[0])
        self.tri = torch.as_tensor(t.reshape(-1), device=device)
        self.W, self.H = int(width), int(height)
        d2r = float(np.pi / 180.0)
        self.tu = float(np.tan(0.5 * hfov_deg * d2r))
        self.tv = float(np.tan(0.5 * vfov_deg * d2r))
        self.range = float(range_units)
        self.near = float(near_range) if near_range else self.range
        self.device = device
        self.maxbox = int(maxbox)
        self.channels = 1
        self.last_clipped_frac = 0.0
        self._clip = torch.zeros(1, dtype=torch.int32, device=device)

    @torch.no_grad()
    def render(self, origin, yaw_deg, pitch_deg, ducked):
        N = origin.shape[0]
        d2r = float(np.pi / 180.0)
        far_bits = int(torch.tensor([self.range], dtype=torch.float32)
                       .view(torch.int32).item())
        depth = torch.full((N, self.H, self.W), far_bits, dtype=torch.int32,
                           device=self.device)
        self._clip.zero_()
        total = N * self.n_tri
        grid = (triton.cdiv(total, RASTER_BLOCK),)
        if self.exact:
            _raster_kernel_exact[grid](
                origin.contiguous(), (yaw_deg * d2r).contiguous(),
                (pitch_deg * d2r).contiguous(),
                ducked.to(torch.int32).contiguous(),
                self.tri, depth,
                self.n_tri, self.H * self.W, self.W, self.H,
                self.tu, self.tv, self.range,
                total, BLOCK=RASTER_BLOCK, num_warps=2)
            self.last_clipped_frac = 0.0
            t = depth.view(torch.float32).clamp(max=self.range)
            return (torch.clamp(t, max=self.near) / self.near
                    + 0.25 * (1.0 - torch.exp(
                        -torch.clamp(t - self.near, min=0.0) / 2500.0)))
        _raster_kernel[grid](
            origin.contiguous(), (yaw_deg * d2r).contiguous(),
            (pitch_deg * d2r).contiguous(),
            ducked.to(torch.int32).contiguous(),
            self.tri, depth, self._clip,
            self.n_tri, self.H * self.W, self.W, self.H,
            self.tu, self.tv, self.range,
            total, MAXBOX=self.maxbox, BLOCK=RASTER_BLOCK, num_warps=2)
        self.last_clipped_frac = float(self._clip.item()) / max(total, 1)
        t = depth.view(torch.float32).clamp(max=self.range)
        return (torch.clamp(t, max=self.near) / self.near
                + 0.25 * (1.0 - torch.exp(-torch.clamp(t - self.near, min=0.0)
                                          / 2500.0)))


def tris_from_mesh(path):
    """(T, 3, 3) world-space triangles from a viewer mesh export."""
    import json
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    w = d["world"]
    pos = np.asarray(w["positions"], np.float32).reshape(-1, 3)
    idx = np.asarray(w["indices"], np.int64).reshape(-1, 3)
    return pos[idx]
