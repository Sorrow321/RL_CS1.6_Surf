"""The CONTACT BLACKOUT: the depth render went fully black whenever the eye
sat inside a dilated-solid voxel, i.e. whenever the player was touching the
ramp it was surfing.

`slab_occupancy` marks a voxel solid if ANY sample within +-cell/2 of its
centre is solid, so at cell 32 the eye's own voxel reads sdf 0 whenever the
eye is within ~16-30 u of a wall or ramp -- and a surfing hull puts it
exactly there (eye 17 u above the origin, hull half-width 16). Every march
tested `d > hit_eps` at t = 0, so on such a frame EVERY ray died before it
moved and the whole image was 0.0. Measured on 51 recorded cannonball
episodes: 0.44% of moving decisions, 4.5x concentrated at ramp/wall
contacts.

The fix (vision.py, all four triton kernels and `_render_torch`): remember
the voxel the eye starts in and suspend the hit test while the sample is
still inside it. What this file holds:

  1. BIT-IDENTITY on every frame that was not black -- a synthetic scene and
     50 poses recorded off a real cannonball policy, against verbatim copies
     of the pre-fix kernels, on the GPU and on the torch fallback;
  2. the four recorded BLACK poses now render, and the nearest hit lands
     where the C tracer says the surface actually is;
  3. the triton path and the torch path agree on those poses;
  4. the surf-mask and normals channels are untouched on non-black frames.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import surfmask, vision                        # noqa: E402


# ============================================================ synthetic ====
# A 512x512x512 u box on a 16 u grid: one solid slab a voxel thick at
# iz == SLAB_Z, plus a solid wall at ix >= WALL_X. The SDF is the same
# unsigned EDT build_sdf() produces (distance to the nearest solid voxel
# CENTRE), so a march can only ever stop inside a solid voxel.
#
# An eye inside the slab is the blackout: every ray reads d == 0 at t == 0.
# Horizontal rays stay in the slab and must stop at the NEXT voxel; rays up
# or down leave it after one or two min_steps and must march away clean.
SCELL = 16.0
SNX = SNY = SNZ = 32
SLAB_Z = 8                                  # z in [128, 144) is solid
WALL_X = 24                                 # x >= 384 is solid


def _synthetic_grids():
    from scipy.ndimage import distance_transform_edt
    occ = np.zeros((SNZ, SNY, SNX), dtype=np.uint8)
    occ[SLAB_Z] = 1
    occ[:, :, WALL_X:] = 1
    sdf = distance_transform_edt(occ == 0, sampling=SCELL).astype(np.float32)
    nrm = np.zeros((SNZ, SNY, SNX, 3), dtype=np.int8)
    nrm[SLAB_Z] = (0, 0, 127)               # the slab reads as a floor
    nrm[:, :, WALL_X:] = (127, 0, 0)        # the wall's canonical normal
    nrm[SLAB_Z, :, WALL_X:] = (127, 0, 0)
    return sdf, nrm


@pytest.fixture
def scene(monkeypatch):
    """Installs the synthetic grids under build_sdf / the surfmask bakes."""
    sdf, nrm = _synthetic_grids()
    mins = np.zeros(3, dtype=np.float32)
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None: (sdf, mins, cell))
    monkeypatch.setattr(surfmask, "build_surfnormal",
                        lambda core, cell, cache_dir=None, mesh_path=None:
                        (nrm, mins))
    monkeypatch.setattr(surfmask, "build_surfnz",
                        lambda core, cell, cache_dir=None, mesh_path=None:
                        (np.ascontiguousarray(nrm[..., 2]), mins))
    return sdf, nrm


# eye = z + 17 (standing). AIR poses sit two voxels above the slab, BLACK
# poses put the eye inside it.
SYN_AIR = [(200.0, 200.0, 183.0, 0.0, 0.0, 0),        # eye z 200, voxel 12
           (200.0, 200.0, 183.0, 45.0, -30.0, 0),
           (200.0, 200.0, 183.0, 200.0, 20.0, 0),
           (330.0, 120.0, 183.0, 90.0, -10.0, 0),
           (330.0, 120.0, 188.0, 270.0, 10.0, 1)]     # ducked: eye z 200
SYN_BLACK = [(200.0, 200.0, 119.0, 0.0, 0.0, 0),      # eye z 136, in the slab
             (200.0, 200.0, 119.0, 90.0, -25.0, 0),
             (120.0, 330.0, 124.0, 180.0, 15.0, 1)]   # ducked: eye z 136


def _tensors(poses, device="cpu"):
    a = np.asarray(poses, dtype=np.float32)
    return (torch.as_tensor(a[:, 0:3], device=device),
            torch.as_tensor(a[:, 3], device=device),
            torch.as_tensor(a[:, 4], device=device),
            torch.as_tensor(a[:, 5], device=device))


# ------------------------------------------- the pre-fix torch march, verbatim
def _legacy_render_torch(lid, origin, yaw_deg, pitch_deg, ducked):
    """`GpuLidar._render_torch` exactly as it was before the start-voxel
    guard -- the early exit, the encoding and both channel gathers included.
    The reference every non-black frame is held bit-identical to."""
    N = origin.shape[0]
    lid._ensure_buffers(N)
    d2r = np.pi / 180.0
    ex = origin[:, 0].view(N, 1, 1)
    ey = origin[:, 1].view(N, 1, 1)
    ez = (origin[:, 2] + torch.where(ducked.bool(), 12.0, 17.0)).view(N, 1, 1)
    dirs = lid._dirs_pinhole if lid.pinhole else lid._dirs_equiangular
    dirs(N, yaw_deg, pitch_deg, d2r)
    t, alive = lid._t, lid._alive
    t.zero_()
    alive.fill_(True)
    hit_eps = 0.6 * lid.cell
    min_step = 0.3 * lid.cell
    inv_cell = 1.0 / lid.cell
    mx, my, mz = lid.mins[0], lid.mins[1], lid.mins[2]
    for it in range(lid.max_steps):
        ix = ((ex + lid._dx * t - mx) * inv_cell).long().clamp_(0, lid.nx - 1)
        iy = ((ey + lid._dy * t - my) * inv_cell).long().clamp_(0, lid.ny - 1)
        iz = ((ez + lid._dz * t - mz) * inv_cell).long().clamp_(0, lid.nz - 1)
        d = lid.sdf_flat[(iz * lid.stride_z + iy * lid.stride_y + ix)
                         .reshape(-1)].reshape(N, lid.H, lid.W).float()
        alive &= (d > hit_eps) & (t < lid.range)
        t.add_(torch.clamp(d * 0.9, min=min_step) * alive)
        if (it & 7) == 7 and not alive.any():
            break
    t = torch.clamp(t, max=lid.range)
    enc = (torch.clamp(t, max=lid.near) / lid.near
           + 0.25 * (1.0 - torch.exp(-torch.clamp(t - lid.near, min=0.0)
                                     / 2500.0)))
    if not (lid.surf_mask or lid.normals):
        return enc
    ix = ((ex + lid._dx * t - mx) * inv_cell).long().clamp_(0, lid.nx - 1)
    iy = ((ey + lid._dy * t - my) * inv_cell).long().clamp_(0, lid.ny - 1)
    iz = ((ez + lid._dz * t - mz) * inv_cell).long().clamp_(0, lid.nz - 1)
    vox = (iz * lid.stride_z + iy * lid.stride_y + ix).reshape(-1)
    if lid.normals:
        w = lid.snrm_flat[vox.view(-1, 1) * 3
                          + torch.arange(3, device=vox.device).view(1, 3)
                          ].float().reshape(N, lid.H, lid.W, 3)
        wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
        sgn = torch.where(wx * lid._dx + wy * lid._dy + wz * lid._dz > 0.0,
                          -1.0, 1.0)
        wx, wy, wz = wx * sgn, wy * sgn, wz * sgn
        yp = yaw_deg.view(N, 1, 1) * d2r
        cy, sy = torch.cos(yp), torch.sin(yp)
        fx = wx * cy + wy * sy
        fy = wy * cy - wx * sy
        l2 = fx * fx + fy * fy + wz * wz
        inv = torch.where(l2 > 0.0, torch.rsqrt(torch.where(l2 > 0.0, l2, 1.0)),
                          0.0)
        return torch.stack((enc, fx * inv, fy * inv, wz * inv), dim=-1)
    snz = lid.snz_flat[vox].reshape(N, lid.H, lid.W).float() / 127.0
    return snz if lid.mask_only else torch.stack((enc, snz), dim=-1)


# ------------------------------------------- the pre-fix triton march, verbatim
@pytest.fixture(scope="session")
def legacy_kernels():
    """`_march_kernel` and `_march_kernel_nz` as they were before the guard
    -- identical down to the early exit, so any difference this file sees is
    the guard and nothing else."""
    import triton
    import triton.language as tl

    @triton.jit
    def _legacy(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
                sdf_ptr, yoff_ptr, poff_ptr,
                total, HW, W,
                nx, ny, nz, stride_z, stride_y,
                mnx, mny, mnz, inv_cell, cell, rng, near,
                max_steps, BLOCK: tl.constexpr):
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
        tl.store(out_ptr + offs, enc, mask=m)

    @triton.jit
    def _legacy_nz(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
                   sdf_ptr, snz_ptr, yoff_ptr, poff_ptr,
                   total, HW, W,
                   nx, ny, nz, stride_z, stride_y,
                   mnx, mny, mnz, inv_cell, cell, rng, near,
                   max_steps, BLOCK: tl.constexpr):
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

    return _legacy, _legacy_nz


def _render_legacy_triton(lid, kern, p, channels=1, snz=None):
    import triton
    origin, yaw, pitch, duck = p
    N = origin.shape[0]
    d2r = float(np.pi / 180.0)
    total = N * lid.H * lid.W
    BLOCK = vision.MARCH_BLOCK
    shape = (N, lid.H, lid.W) if channels == 1 else (N, lid.H, lid.W, channels)
    out = torch.empty(*shape, device=lid.device)
    extra = () if snz is None else (snz,)
    kern[(triton.cdiv(total, BLOCK),)](
        origin.contiguous(), (yaw * d2r).contiguous(),
        (pitch * d2r).contiguous(), duck.to(torch.int32).contiguous(),
        out, lid.sdf_flat, *extra, lid.yoff, lid.poff,
        total, lid.H * lid.W, lid.W,
        lid.nx, lid.ny, lid.nz, lid.stride_z, lid.stride_y,
        lid.mins_f[0], lid.mins_f[1], lid.mins_f[2],
        1.0 / lid.cell, lid.cell, lid.range, lid.near,
        lid.max_steps, BLOCK=BLOCK, num_warps=vision.MARCH_WARPS)
    return out


# ================================================== synthetic-scene tests ====
def test_the_scene_really_is_a_blackout(scene):
    """Guard the guard: SYN_BLACK must be black under the PRE-FIX march and
    SYN_AIR must not, or the tests below prove nothing."""
    lid = vision.GpuLidar(None, 16, 8, cell=SCELL, device="cpu")
    black = _legacy_render_torch(lid, *_tensors(SYN_BLACK))
    assert float(black.abs().max()) == 0.0, \
        "the eye is not inside a solid voxel -- the scene is wrong"
    air = _legacy_render_torch(lid, *_tensors(SYN_AIR))
    assert float(air.min()) > 0.0 and float(air.max()) > 0.5


def test_eye_in_air_is_bit_identical_on_the_torch_path(scene):
    lid = vision.GpuLidar(None, 16, 8, cell=SCELL, device="cpu")
    p = _tensors(SYN_AIR)
    assert torch.equal(lid.render(*p), _legacy_render_torch(lid, *p)), \
        "the guard moved a frame whose eye is in open air"


def test_eye_in_air_is_bit_identical_with_surf_mask_and_normals(scene):
    """Both extra channels are gathered at the FINAL t, so a moved `t` would
    move them too. Neither may move."""
    for kw, ch in (({"surf_mask": True}, 2), ({"normals": True}, 4)):
        lid = vision.GpuLidar(None, 16, 8, cell=SCELL, device="cpu", **kw)
        p = _tensors(SYN_AIR)
        out = lid.render(*p)
        assert out.shape == (len(SYN_AIR), 8, 16, ch)
        assert torch.equal(out, _legacy_render_torch(lid, *p)), \
            f"the guard moved the {ch}-channel render in open air"


def test_the_blackout_is_gone_and_stops_at_the_next_voxel(scene):
    """With the eye inside the slab: every ray now carries a distance, the
    rays that stay in the slab stop within a voxel or two of it, and the
    rays that leave it march away clean."""
    lid = vision.GpuLidar(None, 64, 32, cell=SCELL, device="cpu")
    p = _tensors(SYN_BLACK)
    out = lid.render(*p)
    assert float(out.min()) > 0.0, "a ray still died at t = 0"
    t = out * lid.near                       # enc == t/near below `near`
    # horizontal rays (the image's middle rows) never leave the slab, so
    # they must stop at the next voxel: between one min_step and the far
    # corner of the start voxel, plus the step that carried them out
    mid = t[:, 15:17, :]
    assert float(mid.max()) < 3.0 * SCELL, \
        f"in-slab rays ran {float(mid.max()):.1f} u, past the next voxel"
    assert float(mid.min()) >= 0.3 * SCELL - 1e-4
    # the top row looks up out of the slab and must reach the sky
    assert float(t[0, 0, :].max()) > 5.0 * SCELL, \
        "rays leaving the start voxel did not march on"


def test_a_ray_cannot_re_enter_its_start_voxel(scene):
    """The reprieve is an interval from t = 0, not a per-step test: a ray
    that left the slab and comes back down onto it must HIT it. Pitch the
    eye just above the slab looking down at a grazing angle."""
    lid = vision.GpuLidar(None, 32, 16, cell=SCELL, device="cpu")
    # eye at z 152 -> voxel 9, directly above the slab; looking down it must
    # stop on the slab, not sail through it
    p = _tensors([(200.0, 200.0, 135.0, 0.0, -60.0, 0)])
    t = lid.render(*p) * lid.near
    assert float(t.max()) < 4.0 * SCELL, \
        f"a downward ray sailed {float(t.max()):.1f} u through the slab"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not vision.HAVE_TRITON, reason="needs triton")
@pytest.mark.parametrize("kw,ch", [({}, 1), ({"surf_mask": True}, 2),
                                   ({"normals": True}, 4),
                                   ({"pinhole": True}, 1)])
def test_triton_matches_the_fallback_on_a_black_eye(scene, kw, ch):
    """All four kernels carry the guard, and each still agrees with the
    torch march it mirrors (cos/sin differ in the last ulp between the two,
    which can move a voxel boundary by one ray -- the same 0.995 bar
    test_normals.py uses)."""
    cpu = vision.GpuLidar(None, 32, 16, cell=SCELL, device="cpu", **kw)
    gpu = vision.GpuLidar(None, 32, 16, cell=SCELL, device="cuda", **kw)
    p = _tensors(SYN_BLACK + SYN_AIR)
    ref = cpu.render(*p)
    out = gpu.render(*[x.cuda() for x in p]).cpu()
    assert out.shape == ref.shape
    ok = torch.isclose(out, ref, atol=1e-3)
    assert float(ok.float().mean()) > 0.995, \
        f"{ch}-channel agreement {float(ok.float().mean()):.4f}"
    assert float(out[..., 0].min() if ch > 1 else out.min()) > 0.0, \
        "the triton kernel still blacks out"


# ======================================================= the real map ====
# 50 poses recorded off a live cannonball policy (runs/research/xsG5n,
# moving, eye in air) plus the four black ones derived the way the audit
# derived them: moving, in contact, eye inside a dilated-solid voxel.
# (x, y, z, yaw, pitch, ducked) -- z is the ORIGIN, the render adds the eye.
BLACK_POSES = [
    (-14152.34, 1831.35, 8716.09, 280.03, 8.62, 1),
    (-14146.10, 1842.45, 8714.27, 267.23, 14.21, 1),
    (-14083.56, 846.20, 8306.52, 272.10, 24.15, 1),
    (-3425.93, -10147.50, 2765.64, 149.53, 10.58, 1),
]

RECORDED_POSES = [
    (-10564.74, -8026.67, 6973.04, 4.72, 15.50, 0),
    (-14400.82, -3887.60, 6928.22, 255.56, 30.00, 0),
    (-7290.40, -3246.62, 4101.77, 272.79, 19.89, 0),
    (-5386.64, 61.53, 2199.76, 48.36, -3.25, 0),
    (-3222.28, -1375.43, 2641.65, 286.90, 17.36, 1),
    (5691.54, -8372.59, 2605.50, 268.89, 30.00, 0),
    (12835.30, 760.57, 485.98, 248.38, 30.00, 1),
    (-4067.87, 983.80, 2104.15, 24.99, -18.94, 0),
    (-8028.04, -4342.10, 1283.30, 79.07, 29.20, 0),
    (-8572.31, 557.36, 4280.41, 123.10, 20.02, 0),
    (-2033.98, 1324.21, 1782.94, 357.65, 13.77, 0),
    (-7803.34, -7163.39, 1737.48, 108.27, -14.69, 1),
    (-3482.63, 1104.76, 3489.84, 280.48, 22.02, 0),
    (-14423.34, -3995.41, 6904.30, 260.12, 27.61, 0),
    (-9285.42, -6864.15, 6027.83, 67.52, -1.92, 0),
    (-9367.30, -7294.20, 6364.25, 58.07, -1.12, 0),
    (-8914.21, -5701.52, 5523.24, 70.22, 17.10, 0),
    (8529.05, -5873.27, 3105.41, 120.74, 30.00, 0),
    (8165.60, -5281.48, 2665.05, 120.74, 30.00, 0),
    (-8968.69, -6155.64, 5582.33, 78.25, 7.52, 0),
    (-8081.17, -336.93, 3807.34, 97.42, 30.00, 0),
    (-894.77, -10974.06, 4244.22, 167.14, -1.39, 0),
    (-8277.31, -5003.72, 1018.46, 87.20, 30.00, 0),
    (-13995.43, -6603.04, 7190.68, 310.85, 30.00, 1),
    (-7679.08, 2884.53, 4801.86, 14.55, 24.68, 0),
    (3233.92, 3389.56, -5310.94, 99.43, 28.67, 0),
    (-9227.58, -6772.93, 5889.80, 54.31, 8.19, 0),
    (-6396.63, -908.38, 3602.82, 226.77, 28.94, 0),
    (3902.88, 6295.30, -43.58, 28.44, 30.00, 0),
    (-3313.70, -5173.02, -2862.05, 79.28, 26.28, 1),
    (3347.01, 2707.45, -5431.10, 91.12, 30.00, 0),
    (-13716.26, -6846.62, 7313.36, 322.99, 30.00, 1),
    (-8060.81, 2745.71, 4753.31, 32.73, 24.68, 0),
    (-14206.59, 1760.61, 8744.72, 276.12, 16.33, 1),
    (-5942.89, -7373.98, 4740.79, 328.23, 30.00, 0),
    (2316.31, -2707.35, 2040.64, 5.24, 30.00, 1),
    (9017.21, -7050.50, 3253.57, 125.81, 30.00, 0),
    (9056.35, -4980.81, -1250.87, 219.65, 12.71, 0),
    (-2105.22, 3236.33, -1469.82, 237.60, 28.00, 0),
    (-13505.60, -1639.01, 8263.41, 277.11, 25.21, 0),
    (725.68, -6285.47, 1191.41, 27.25, 24.81, 0),
    (-3246.28, 3027.95, 4192.86, 238.15, 29.47, 0),
    (2676.77, 4463.21, 1008.33, 82.09, 30.00, 0),
    (-8752.31, 1758.20, 4532.31, 80.63, 28.67, 0),
    (6651.55, -9210.25, 3413.30, 353.48, 30.00, 0),
    (-3494.25, -6954.71, -1598.06, 105.01, 27.87, 0),
    (-8262.18, -4714.78, 1034.54, 94.20, 30.00, 0),
    (-8208.81, 233.75, 3942.94, 112.47, 30.00, 0),
    (-6155.16, 2587.18, 4837.42, 314.82, 5.79, 0),
    (832.01, -6159.98, 1058.08, 16.14, 30.00, 0),
]


def _maps_dir():
    """The maps directory that already holds the cannonball SDF cache.

    A linked worktree's own maps/ has different mtimes, and the cache
    signature embeds mtime_ns, so building there costs a full rebake on a
    30 MB grid. Prefer whichever checkout has the baked cache; the main
    worktree is found through .git rather than hardcoded."""
    cands = []
    env = os.environ.get("SURF_MAPS")
    if env:
        cands.append(Path(env))
    cands.append(ROOT / "maps")
    g = ROOT / ".git"
    if g.is_file():                        # "gitdir: <main>/.git/worktrees/<n>"
        txt = g.read_text(encoding="utf-8").strip()
        if txt.startswith("gitdir:"):
            p = Path(txt.split(":", 1)[1].strip())
            for parent in p.parents:
                if parent.name == ".git":
                    cands.append(parent.parent / "maps")
                    break
    for c in cands:
        if ((c / "surf_src_cannonball.bsp").exists()
                and (c / "surf_src_cannonball.sdf_32.npz").exists()):
            return c
    return None


MAPS = _maps_dir()
mapmark = pytest.mark.skipif(
    MAPS is None, reason="needs a checkout with the baked cannonball SDF")


@pytest.fixture(scope="module")
def core():
    from surfgym import SurfCore, default_config
    return SurfCore(str(MAPS / "surf_src_cannonball.bsp"),
                    default_config(num_envs=1, lidar_w=0, lidar_h=0))


@pytest.fixture(scope="module")
def lidar(core):
    """The live training config: 128x64, range 11500, near 2000, 32u."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return vision.GpuLidar(core, 128, 64, range_units=11500.0,
                           near_range=2000.0, cell=32.0, device=dev)


def _nearest_solid(core, eye, L=400.0):
    """Distance from the eye to the nearest real surface, over the 26 cube
    directions, with the exact C tracer -- the ground truth the dilated
    voxel grid is an approximation of."""
    best = L
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if not (dx or dy or dz):
                    continue
                d = np.array([dx, dy, dz], float)
                d /= np.linalg.norm(d)
                tr = core.trace(eye, eye + d * L, hull=2)
                if tr.startsolid:
                    return 0.0
                if tr.fraction < 1.0:
                    best = min(best, tr.fraction * L)
    return best


def _eye(pose):
    return np.array([pose[0], pose[1], pose[2] + (12.0 if pose[5] else 17.0)])


def _probe_rays(core, pose, lid, L=3000.0):
    """Exact tracer distance along five rays of the image (centre, the four
    edge midpoints), in the renderer's own camera convention."""
    d2r = np.pi / 180.0
    eye = _eye(pose)
    out = []
    for r, c in ((lid.H // 2, lid.W // 2), (0, lid.W // 2), (lid.H - 1, lid.W // 2),
                 (lid.H // 2, 0), (lid.H // 2, lid.W - 1)):
        yw = pose[3] * d2r + float(lid.yoff[c])
        pt = pose[4] * d2r + float(lid.poff[r])
        d = np.array([np.cos(pt) * np.cos(yw), np.cos(pt) * np.sin(yw),
                      np.sin(pt)])
        tr = core.trace(eye, eye + d * L, hull=2)
        out.append(L if tr.fraction >= 1.0 else tr.fraction * L)
    return out


@mapmark
def test_the_recorded_poses_are_what_they_claim(lidar):
    """The fixtures are only meaningful if the 50 are non-black and the 4
    are black on THIS map's grid."""
    sdf = lidar.sdf_flat
    hit_eps = 0.6 * lidar.cell

    def d_at(pose):
        e = _eye(pose)
        i = [int(min(max((e[k] - lidar.mins_f[k]) / lidar.cell, 0),
                     (lidar.nx, lidar.ny, lidar.nz)[k] - 1)) for k in range(3)]
        return float(sdf[i[2] * lidar.stride_z + i[1] * lidar.stride_y + i[0]])

    assert all(d_at(p) > hit_eps for p in RECORDED_POSES), \
        "a recorded pose is black -- it cannot carry a bit-identity claim"
    assert all(d_at(p) <= hit_eps for p in BLACK_POSES), \
        "a black pose is no longer black -- the SDF cache changed"


@mapmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not vision.HAVE_TRITON, reason="needs triton")
def test_recorded_poses_are_bit_identical_on_the_gpu(lidar, legacy_kernels):
    p = _tensors(RECORDED_POSES, lidar.device)
    new = lidar.render(*p)
    old = _render_legacy_triton(lidar, legacy_kernels[0], p)
    diff = float((new - old).abs().max())
    assert diff == 0.0, f"depth ABI drift on non-black frames: {diff:.3e}"


@mapmark
def test_recorded_poses_are_bit_identical_on_the_torch_path(lidar):
    p = _tensors(RECORDED_POSES, lidar.device)
    new = lidar._render_torch(*p)
    old = _legacy_render_torch(lidar, *p)
    assert torch.equal(new, old), "the torch fallback drifted on non-black frames"


@mapmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not vision.HAVE_TRITON, reason="needs triton")
def test_the_surf_mask_channel_is_untouched(core, legacy_kernels):
    """The |n_z| channel is gathered at the final t, so it moves iff depth
    does. Held to the pre-fix 2-channel kernel on the same 50 poses."""
    lid = vision.GpuLidar(core, 64, 32, range_units=11500.0, near_range=2000.0,
                          cell=32.0, device="cuda", surf_mask=True)
    try:
        p = _tensors(RECORDED_POSES, lid.device)
        new = lid.render(*p)
        old = _render_legacy_triton(lid, legacy_kernels[1], p, channels=2,
                                    snz=lid.snz_flat)
        assert new.shape == old.shape == (len(RECORDED_POSES), 32, 64, 2)
        assert torch.equal(new, old), "surf-mask render drifted"
    finally:
        del lid
        torch.cuda.empty_cache()


@mapmark
def test_the_black_poses_now_render(lidar, core):
    """Before: the whole image 0.0. After: every ray carries a distance and
    the nearest hit lands where the C tracer says the surface is (the grid
    is dilated by up to a voxel, so the march may stop early, never late)."""
    p = _tensors(BLACK_POSES, lidar.device)
    old = _legacy_render_torch(lidar, *p)
    assert float(old.abs().max()) == 0.0, "these poses were not black to begin with"

    new = lidar.render(*p).float().cpu()
    assert float(new.min()) > 0.0, "a ray still dies at t = 0"
    t = (new * lidar.near).numpy()           # enc == t/near below `near`
    for i, pose in enumerate(BLACK_POSES):
        truth = _nearest_solid(core, _eye(pose))
        near_hit = float(t[i].min())
        assert 16.0 <= truth <= 30.0, \
            f"pose {i}: the eye is {truth:.1f} u from geometry, not a dilation case"
        assert 0.3 * lidar.cell - 1e-3 <= near_hit <= truth + 2.0 * lidar.cell, \
            f"pose {i}: nearest hit {near_hit:.1f} u vs truth {truth:.1f} u"
        # and the frame carries the world, not a wall of contact pixels: it
        # must see at least half as far as the exact tracer does in the most
        # open of five probe directions. (Pose 3 is genuinely wedged -- the
        # tracer finds nothing past 123 u in any of them -- so an absolute
        # threshold here would be measuring the map, not the renderer.)
        reach = max(_probe_rays(core, pose, lidar))
        assert float(t[i].max()) >= 0.5 * reach, \
            f"pose {i}: sees {float(t[i].max()):.0f} u where the tracer " \
            f"reaches {reach:.0f} u"


@mapmark
def test_four_of_test_lidar_marchs_own_poses_were_blind(lidar):
    """HEADS-UP for whoever finds test_lidar_march.py red.

    Four of its eight fixtures put the eye INSIDE a dilated-solid voxel --
    indices 2 and 3 ("at the glass panes") and 4 and 5 (labelled "mid-track,
    open air", which they are not: the eye is inside geometry, the render
    stops every ray at 9.6-28.8 u). Before this fix they rendered a wall of
    0.0, so that file's bit-exactness assertion was comparing 0 == 0 on half
    its poses. The guard changes exactly those four, by design.

    Fixing it is either the same four lines in ITS legacy copy, or four
    poses that are actually in open air. Until then this test says why."""
    march_poses = [(-14200.0, 2898.0, 10700.0, 0.0, -10.0, 0),
                   (-14200.0, 2898.0, 10700.0, 90.0, 25.0, 0),
                   (-13536.0, 3392.0, 10688.0, 45.0, 0.0, 0),
                   (-13536.0, 3392.0, 10688.0, 225.0, -30.0, 1),
                   (-11000.0, 5000.0, 4000.0, 180.0, 0.0, 0),
                   (-11000.0, 5000.0, 4000.0, 315.0, 29.0, 0),
                   (-11000.0, 7300.0, -1000.0, 90.0, -10.0, 0),
                   (-9000.0, 7000.0, -800.0, 270.0, 10.0, 1)]
    old = _legacy_render_torch(lidar, *_tensors(march_poses, lidar.device))
    blind = [i for i in range(8) if float(old[i].abs().max()) == 0.0]
    assert blind == [2, 3, 4, 5], \
        f"the blind subset of test_lidar_march's poses moved: {blind}"


@mapmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not vision.HAVE_TRITON, reason="needs triton")
def test_triton_and_torch_agree_on_the_black_poses(lidar):
    """The same 0.995 bar test_normals.py holds the two paths to: the last
    ulp of cos/sin can move a voxel boundary by one ray."""
    p = _tensors(BLACK_POSES, lidar.device)
    gpu = lidar.render(*p)
    cpu = lidar._render_torch(*p)
    ok = torch.isclose(gpu, cpu, atol=1e-4)
    assert float(ok.float().mean()) > 0.995, \
        f"triton/torch agreement on black poses {float(ok.float().mean()):.4f}"
