"""The pinhole camera (--pinhole): same centre ray, straight edges.

The shipped lidar is EQUIANGULAR — yoff/poff are ANGLES added to the view,
so every pixel spans the same angle and straight world edges bow across the
image. ``GpuLidar(pinhole=True)`` samples a regular grid on the tangent
plane instead. Four claims, all checkable without a GPU because the torch
fallback marches the same grid the triton kernel does:

  1. at the image centre the two cameras cast the IDENTICAL ray, for any
     yaw and pitch — otherwise the experiment compares two things at once;
  2. the corner rays sit where the pinhole geometry says they do,
     atan(hypot(tan(hfov/2), tan(vfov/2))), which is NOT where the
     equiangular corner sits;
  3. the discriminating one: a straight world edge lands on a single pixel
     row under pinhole and bows across several under equiangular;
  4. the equiangular direction setup is bit-identical to the code that was
     inline in _render_torch before this feature moved it into a method.

What a GPU still owes: that _march_kernel_pin agrees with this fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import vision                                  # noqa: E402

HFOV, VFOV = 120.0, 90.0                  # GpuLidar's defaults
D2R = np.pi / 180.0


def _dirs(lid, yaw, pitch):
    """The unit ray directions the march will use, (H, W, 3)."""
    lid._ensure_buffers(1)
    f = lid._dirs_pinhole if lid.pinhole else lid._dirs_equiangular
    f(1, torch.tensor([float(yaw)]), torch.tensor([float(pitch)]), D2R)
    return torch.stack((lid._dx[0], lid._dy[0], lid._dz[0]), -1).numpy()


# ------------------------------------------------------------- ray geometry --
@pytest.fixture
def empty_world(monkeypatch):
    """A grid of pure free space: enough for GpuLidar to construct."""
    sdf = np.full((8, 8, 8), 999.0, np.float32)
    mins = np.zeros(3, np.float32)
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None: (sdf, mins, cell))


# odd dims on purpose: both grids then have an exact centre pixel
CW, CH = 9, 5


@pytest.mark.parametrize("yaw,pitch", [(0.0, 0.0), (37.0, 0.0), (0.0, 25.0),
                                       (-140.0, -33.0), (270.0, 60.0),
                                       (95.0, -70.0)])
def test_centre_ray_matches_the_equiangular_camera(empty_world, yaw, pitch):
    eq = vision.GpuLidar(None, CW, CH, device="cpu")
    pin = vision.GpuLidar(None, CW, CH, device="cpu", pinhole=True)
    a = _dirs(eq, yaw, pitch)[CH // 2, CW // 2]
    b = _dirs(pin, yaw, pitch)[CH // 2, CW // 2]
    assert np.abs(a - b).max() < 1e-6, f"centre rays diverge by {a - b}"
    # and it really is the view direction, not just a shared mistake
    p, y = np.radians(pitch), np.radians(yaw)
    fwd = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    assert np.abs(b - fwd).max() < 1e-6


def test_pinhole_rays_are_unit_length_and_keep_the_conventions(empty_world):
    pin = vision.GpuLidar(None, CW, CH, device="cpu", pinhole=True)
    d = _dirs(pin, 0.0, 0.0)
    assert np.abs(np.linalg.norm(d, axis=-1) - 1.0).max() < 1e-6, \
        "the march measures t in units — directions must be normalized"
    # surfcore.h: col 0 looks LEFT (+yaw side), row 0 looks UP
    assert d[CH // 2, 0][1] > 0 and d[CH // 2, -1][1] < 0
    assert d[0, CW // 2][2] > 0 and d[-1, CW // 2][2] < 0
    # the tangent grid spans exactly the nominal fov, endpoints included
    assert np.allclose(pin.uoff.numpy()[[0, -1]],
                       [-np.tan(HFOV / 2 * D2R), np.tan(HFOV / 2 * D2R)])
    assert np.allclose(pin.voff.numpy()[[0, -1]],
                       [np.tan(VFOV / 2 * D2R), -np.tan(VFOV / 2 * D2R)])


def test_rays_are_the_tangent_grid_rotated_into_the_view(empty_world):
    """Every pixel, not just the centre and the axes.

    Derived independently: the camera-space ray is (1, -u, v) — +x forward,
    +z up, and -y to the RIGHT because +y is left — rotated by pitch about
    y and then yaw about z. Sign errors in the right/up basis vanish at
    yaw=0 (u only enters dx through sin(yaw)) and the pitch tilt of `up`
    vanishes at pitch=0, so a camera checked only on-axis can be wrong
    everywhere else and look perfect. Two mutants proved exactly that.
    """
    pin = vision.GpuLidar(None, 7, 5, device="cpu", pinhole=True)
    u = pin.uoff.numpy()
    v = pin.voff.numpy()
    for yaw, pitch in ((0.0, 0.0), (41.0, 0.0), (0.0, -37.0), (41.0, -37.0),
                       (-115.0, 62.0)):
        got = _dirs(pin, yaw, pitch)
        y, p = np.radians(yaw), np.radians(pitch)
        rz = np.array([[np.cos(y), -np.sin(y), 0.0],
                       [np.sin(y), np.cos(y), 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[np.cos(p), 0.0, -np.sin(p)], [0.0, 1.0, 0.0],
                       [np.sin(p), 0.0, np.cos(p)]])          # pitch up = +z
        for r in range(pin.H):
            for c in range(pin.W):
                cam = np.array([1.0, -u[c], v[r]])
                want = rz @ ry @ (cam / np.linalg.norm(cam))
                assert np.abs(got[r, c] - want).max() < 1e-6, \
                    f"yaw {yaw} pitch {pitch} px ({r},{c}): " \
                    f"{got[r, c]} != {want}"


def test_the_kernel_and_the_fallback_share_the_direction_source():
    """The triton kernel and the torch fallback must state the SAME frame.

    A mutation run proved CPU tests are otherwise blind to the kernel:
    flipping a sign in _march_kernel_pin changed nothing any test could see,
    because triton is never invoked here. Nothing on a CPU can execute that
    source — but it can read it, and the three direction lines are written
    identically in both paths precisely so this check is possible.

    This does not replace GPU parity; it only stops one path being fixed
    while the other keeps the bug.
    """
    src = (ROOT / "python" / "surfgym" / "vision.py").read_text(encoding="utf-8")
    for expr in ("dx = cp * cy + u * sy - v * sp * cy",
                 "dy = cp * sy - u * cy - v * sp * sy",
                 "dz = sp + v * cp"):
        assert src.count(expr) == 2, \
            f"'{expr}' appears {src.count(expr)}x — the kernel and the " \
            f"fallback disagree about the camera frame"


def test_corner_spread_is_the_analytic_pinhole_angle(empty_world):
    """A rectilinear camera buys its straight lines with solid angle: the
    corner sits at atan(hypot(tan(h/2), tan(v/2))) — 63.43 deg at the
    default 120x90, where the equiangular corner is 61.11."""
    pin = vision.GpuLidar(None, CW, CH, device="cpu", pinhole=True)
    d = _dirs(pin, 0.0, 0.0)
    fwd = np.array([1.0, 0.0, 0.0])
    want = np.arctan(np.hypot(np.tan(HFOV / 2 * D2R), np.tan(VFOV / 2 * D2R)))
    for r, c in ((0, 0), (0, CW - 1), (CH - 1, 0), (CH - 1, CW - 1)):
        ang = np.arccos(np.clip(d[r, c] @ fwd, -1.0, 1.0))
        assert abs(ang - want) < 1e-6, \
            f"corner ({r},{c}) at {np.degrees(ang):.4f} deg, want " \
            f"{np.degrees(want):.4f}"
    eq = vision.GpuLidar(None, CW, CH, device="cpu")
    ang = np.arccos(np.clip(_dirs(eq, 0.0, 0.0)[0, 0] @ fwd, -1.0, 1.0))
    assert abs(ang - want) > 0.02, \
        "the two cameras agree at the corner — the test is vacuous"


def test_extraction_left_the_equiangular_rays_alone(empty_world):
    """_dirs_equiangular is _render_torch's old inline block, moved. Every
    checkpoint was trained on those pixel values, so 'moved' has to mean
    bit-identical, not 'close'."""
    eq = vision.GpuLidar(None, 16, 8, device="cpu")
    for yaw, pitch in ((0.0, 0.0), (123.0, -41.0)):
        got = _dirs(eq, yaw, pitch)
        # the pre-change code, verbatim
        yd = torch.tensor([yaw])
        pd = torch.tensor([pitch])
        p = pd.view(1, 1, 1) * D2R + eq.poff.view(1, eq.H, 1)
        y = yd.view(1, 1, 1) * D2R + eq.yoff.view(1, 1, eq.W)
        cp = torch.cos(p)
        want = torch.stack((cp * torch.cos(y), cp * torch.sin(y),
                            torch.sin(p).expand(1, eq.H, eq.W)), -1)[0]
        assert torch.equal(torch.as_tensor(got), want), \
            "the equiangular ray directions moved"


# --------------------------------------------------------- the straight edge --
# One straight world edge to photograph: a thin slab filling z in
# [Z0-THICK, Z0] for every x >= X0. Its lip is the line {(X0, y, Z0)} — a
# horizontal world line perpendicular to the view axis, which a rectilinear
# camera must image as one row. The SDF is analytic (distance to that box),
# so no map, no scipy and no DLL are involved.
CELL = 8.0
GNX, GNY, GNZ = 96, 192, 64               # 768 x 1536 x 512 units
X0, Z0, THICK = 512.0, 256.0, 8.0
EYE = torch.tensor([[200.0, 768.0, 383.0]])       # eye height +17 -> z = 400


@pytest.fixture
def edge_world(monkeypatch):
    cx = (np.arange(GNX) + 0.5) * CELL
    cz = (np.arange(GNZ) + 0.5) * CELL
    qx = np.maximum(X0 - cx, 0.0)
    qz = np.maximum.reduce([cz - Z0, (Z0 - THICK) - cz, np.zeros(GNZ)])
    sdf = (np.hypot(qz[:, None, None], qx[None, None, :])
           * np.ones((1, GNY, 1))).astype(np.float32)
    mins = np.zeros(3, np.float32)
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None: (sdf, mins, cell))


def _lip_row_per_column(pinhole):
    lid = vision.GpuLidar(None, 32, 32, range_units=3000.0, cell=CELL,
                          device="cpu", pinhole=pinhole)
    d = lid.render(EYE, torch.tensor([0.0]), torch.tensor([0.0]),
                   torch.tensor([0.0]))[0].numpy()
    hit = d < 0.99                        # below the lip the rays leave
    assert 0.2 < hit.mean() < 0.9, "degenerate frame — nothing to measure"
    rows = np.array([np.flatnonzero(hit[:, c]).max() if hit[:, c].any() else -1
                     for c in range(hit.shape[1])])
    assert (rows >= 0).all(), "a column sees no geometry at all"
    return rows


def test_a_straight_edge_stays_straight_under_pinhole(edge_world):
    rows = _lip_row_per_column(pinhole=True)
    assert rows.max() == rows.min(), \
        f"the lip wanders over {rows.max() - rows.min()} rows: {rows}"


def test_the_same_edge_bows_under_the_equiangular_camera(edge_world):
    """The control. If this ever renders straight too, the fixture stopped
    discriminating and the test above proves nothing."""
    rows = _lip_row_per_column(pinhole=False)
    assert rows.max() - rows.min() >= 3, \
        f"equiangular did not bow the edge: {rows}"
    # and it bows the way a fisheye does — highest in the middle, falling
    # off symmetrically toward both edges of the frame
    assert rows[len(rows) // 2] > rows[0] and rows[len(rows) // 2] > rows[-1]


def test_the_cameras_disagree_across_the_frame_but_not_at_the_centre(edge_world):
    lids = [vision.GpuLidar(None, 33, 33, range_units=3000.0, cell=CELL,
                            device="cpu", pinhole=p) for p in (False, True)]
    a, b = [lid.render(EYE, torch.tensor([0.0]), torch.tensor([0.0]),
                       torch.tensor([0.0]))[0].numpy() for lid in lids]
    assert a[16, 16] == pytest.approx(b[16, 16], abs=1e-6), \
        "the centre pixel must see the same thing in both cameras"
    assert np.abs(a - b).max() > 0.05, "the frames are indistinguishable"


# ------------------------------------------------------------------ gating --
def test_pinhole_and_surf_mask_are_mutually_exclusive(empty_world):
    with pytest.raises(ValueError, match="no combined kernel"):
        vision.GpuLidar(None, 8, 4, device="cpu", surf_mask=True, pinhole=True)


def test_flag_off_allocates_nothing(empty_world):
    off = vision.GpuLidar(None, 8, 4, device="cpu")
    on = vision.GpuLidar(None, 8, 4, device="cpu", pinhole=True)
    tensors = {k for k, v in vars(off).items() if torch.is_tensor(v)}
    assert tensors == {"sdf_flat", "mins", "yoff", "poff"}
    assert {k for k, v in vars(on).items() if torch.is_tensor(v)} - tensors \
        == {"uoff", "voff"}
    assert off.pinhole is False and on.pinhole is True
    # the equiangular render is untouched by the feature existing
    args = (torch.tensor([[100.0, 100.0, 100.0]]), torch.tensor([0.0]),
            torch.tensor([0.0]), torch.tensor([0.0]))
    assert off.render(*args).shape == (1, 4, 8)
    assert on.render(*args).shape == (1, 4, 8)


def test_a_one_pixel_axis_is_the_centre_ray(empty_world):
    """W or H of 1 divides by zero in 2c/(W-1); it has to mean 'centre'."""
    lid = vision.GpuLidar(None, 1, 1, device="cpu", pinhole=True)
    assert float(lid.uoff[0]) == 0.0 and float(lid.voff[0]) == 0.0
    d = _dirs(lid, 30.0, -20.0)[0, 0]
    p, y = np.radians(-20.0), np.radians(30.0)
    assert np.allclose(d, [np.cos(p) * np.cos(y), np.cos(p) * np.sin(y),
                           np.sin(p)], atol=1e-6)
