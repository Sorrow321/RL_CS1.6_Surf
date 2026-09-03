"""The SDF march kernel is bit-exact against the pre-early-exit version.

The depth encoding is warm-start ABI: a checkpoint's conv trunk was trained
on specific pixel values, so a march change is only allowed to be FASTER,
never different. The early exit and the runtime trip bound are exactly
equivalent by construction — `alive` is monotone and a dead lane adds 0 to
`t` — and this test holds that claim to the letter (max|diff| == 0, not a
tolerance) against a verbatim copy of the old kernel.

Poses per docs/perf-implementation-plan.md S7: spawn area, mid-track, the
glass at (-13536, 3392, 10688), and sky-heavy views.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import vision                                # noqa: E402

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
SDF_CACHE = ROOT / "maps" / "surf_src_cannonball.sdf_32.npz"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not vision.HAVE_TRITON, reason="needs triton"),
    pytest.mark.skipif(not SDF_CACHE.exists(),
                       reason="needs the baked SDF cache (2 min to build)"),
]

# 8 fixed poses: (x, y, z, yaw, pitch, ducked)
POSES = [
    (-14200.0, 2898.0, 10700.0,   0.0, -10.0, 0),   # spawn platform
    (-14200.0, 2898.0, 10700.0,  90.0,  25.0, 0),   # spawn, gaze up (sky-heavy)
    (-13536.0, 3392.0, 10688.0,  45.0,   0.0, 0),   # at the glass panes
    (-13536.0, 3392.0, 10688.0, 225.0, -30.0, 1),   # glass, ducked, looking down
    (-11000.0, 5000.0,  4000.0, 180.0,   0.0, 0),   # mid-track, open air
    (-11000.0, 5000.0,  4000.0, 315.0,  29.0, 0),   # mid-track, gaze up
    (-11000.0, 7300.0, -1000.0,  90.0, -10.0, 0),   # near the finish
    (-9000.0, 7000.0,  -800.0,  270.0,  10.0, 1),   # past the finish, ducked
]


@pytest.fixture(scope="module")
def lidar():
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0,
                                                    lidar_h=0))
    # the live training config: range 11500, near 2000, 32u voxels
    return vision.GpuLidar(core, 128, 64, range_units=11500.0,
                           near_range=2000.0, cell=32.0, device="cuda")


@pytest.fixture(scope="module")
def legacy_kernel():
    """The kernel exactly as it was before the early exit: constexpr trip
    count, no break."""
    import triton
    import triton.language as tl

    @triton.jit
    def _legacy(eye_ptr, yaw_ptr, pitch_ptr, duck_ptr, out_ptr,
                sdf_ptr, yoff_ptr, poff_ptr,
                total, HW, W,
                nx, ny, nz, stride_z, stride_y,
                mnx, mny, mnz, inv_cell, cell, rng, near,
                MAX_STEPS: tl.constexpr, BLOCK: tl.constexpr):
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
        # the eye's own voxel: the contact-blackout guard (vision.py); the
        # legacy reference carries it too, or four of this file's poses
        # (eye inside dilated solid) would compare black against black
        ix0 = tl.minimum(tl.maximum(((ex - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
        iy0 = tl.minimum(tl.maximum(((ey - mny) * inv_cell).to(tl.int64), 0), ny - 1)
        iz0 = tl.minimum(tl.maximum(((ez - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
        vox0 = iz0 * stride_z + iy0 * stride_y + ix0
        for _ in range(MAX_STEPS):
            px = ex + dx * t
            py = ey + dy * t
            pz = ez + dz * t
            ix = tl.minimum(tl.maximum(((px - mnx) * inv_cell).to(tl.int64), 0), nx - 1)
            iy = tl.minimum(tl.maximum(((py - mny) * inv_cell).to(tl.int64), 0), ny - 1)
            iz = tl.minimum(tl.maximum(((pz - mnz) * inv_cell).to(tl.int64), 0), nz - 1)
            vox = iz * stride_z + iy * stride_y + ix
            d = tl.load(sdf_ptr + vox, mask=alive, other=0.0).to(tl.float32)
            alive = alive & ((d > hit_eps) | (vox == vox0)) & (t < rng)
            t += tl.where(alive, tl.maximum(d * 0.9, min_step), 0.0)
        t = tl.minimum(t, rng)
        enc = tl.minimum(t, near) / near \
            + 0.25 * (1.0 - tl.exp(-tl.maximum(t - near, 0.0) / 2500.0))
        tl.store(out_ptr + offs, enc, mask=m)

    return _legacy


def _tensors(lid):
    a = np.asarray(POSES, dtype=np.float32)
    dev = lid.device
    return (torch.as_tensor(a[:, 0:3], device=dev),
            torch.as_tensor(a[:, 3], device=dev),
            torch.as_tensor(a[:, 4], device=dev),
            torch.as_tensor(a[:, 5], device=dev))


def _render_legacy(lid, kern, origin, yaw, pitch, duck):
    import triton
    N = origin.shape[0]
    d2r = float(np.pi / 180.0)
    out = torch.empty(N, lid.H, lid.W, device=lid.device)
    total = N * lid.H * lid.W
    BLOCK = 256
    kern[(triton.cdiv(total, BLOCK),)](
        origin.contiguous(), (yaw * d2r).contiguous(),
        (pitch * d2r).contiguous(), duck.to(torch.int32).contiguous(),
        out, lid.sdf_flat, lid.yoff, lid.poff,
        total, lid.H * lid.W, lid.W,
        lid.nx, lid.ny, lid.nz, lid.stride_z, lid.stride_y,
        lid.mins_f[0], lid.mins_f[1], lid.mins_f[2],
        1.0 / lid.cell, lid.cell, lid.range, lid.near,
        MAX_STEPS=lid.max_steps, BLOCK=BLOCK)
    return out


def test_march_is_bit_exact_against_the_legacy_kernel(lidar, legacy_kernel):
    origin, yaw, pitch, duck = _tensors(lidar)
    new = lidar.render(origin, yaw, pitch, duck)
    old = _render_legacy(lidar, legacy_kernel, origin, yaw, pitch, duck)
    diff = (new - old).abs().max().item()
    assert diff == 0.0, f"depth ABI drift: max|diff| = {diff:.3e}"


def test_poses_actually_exercise_the_march(lidar):
    """A vacuous test would pass if every ray hit instantly or ran to range.
    The fixtures must produce a real spread of depths."""
    origin, yaw, pitch, duck = _tensors(lidar)
    d = lidar.render(origin, yaw, pitch, duck)
    assert d.min().item() < 0.05, "no near hits — poses are all in open space"
    assert d.max().item() > 1.0, "nothing reached past `near` — no far field"
    assert 0.02 < float((d > 1.0).float().mean()) < 0.98, \
        "depths are degenerate (all near or all far)"


def test_batch_size_is_irrelevant_to_the_result(lidar):
    """The truncation bootstrap renders odd small batches; a block-wide early
    exit must not make a ray depend on which block it landed in."""
    origin, yaw, pitch, duck = _tensors(lidar)
    full = lidar.render(origin, yaw, pitch, duck)
    for i in (0, 3, 7):
        one = lidar.render(origin[i:i + 1], yaw[i:i + 1], pitch[i:i + 1],
                           duck[i:i + 1])
        assert torch.equal(one[0], full[i]), f"pose {i} depends on batch size"
