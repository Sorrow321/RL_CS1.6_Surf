"""--lidar-hfov / --lidar-vfov: EXTEND the camera, never rescale it.

The shipped sensor is 120x90 deg and every checkpoint this project has
trained was rendered through it, so two properties have to hold before a
wide-fov arm means anything:

  1. **the default is the old camera, bit-for-bit.** Not "close" - the
     depth encoding is warm-start ABI, and a run that silently re-aimed
     every ray would be a new experiment wearing the baseline's numbers;
  2. **widening fov and width in the same ratio is a pure extension.** At
     240 deg over 128 columns the deg/column is what it is today, the
     central 64 columns are the SAME rays, and the added columns look
     where nothing looked before. That is what separates "the agent can
     see more" from "the agent sees the same thing more coarsely", and
     precision is the thing under test.

Plus the plumbing that makes the flag real: the trainer must hand both
numbers to GpuLidar and write them into the saved config, the recorder must
read them back (a recording at 120 deg of a 360-deg policy feeds those
weights a crop of the world they trained on), and a checkpoint has to load
into a wider model - `Policy.conv` ends in AdaptiveAvgPool2d((4, 8)), so no
weight carries W in its shape.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import vision                                  # noqa: E402
from train_fast import (LIDAR_HFOV, LIDAR_VFOV,             # noqa: E402
                        N_SCALAR, Policy)

D2R = np.pi / 180.0
# the frame every arm on cannonball and petrus has actually trained on
BASE_W, BASE_H = 64, 32


# ------------------------------------------------------------------ a world --
# A thin slab filling z in [Z0-THICK, Z0] for every x >= X0, plus a wall at
# large |y| so the SIDE rays hit something too - a fov test whose new columns
# all see open sky would pass on a bug that aimed them at the floor.
CELL = 16.0
GNX, GNY, GNZ = 64, 64, 32
X0, Z0, THICK = 512.0, 256.0, 16.0
EYE = torch.tensor([[256.0, 512.0, 383.0]])


@pytest.fixture
def slab_world(monkeypatch):
    cx = (np.arange(GNX) + 0.5) * CELL
    cy = (np.arange(GNY) + 0.5) * CELL
    cz = (np.arange(GNZ) + 0.5) * CELL
    qx = np.maximum(X0 - cx, 0.0)
    qz = np.maximum.reduce([cz - Z0, (Z0 - THICK) - cz, np.zeros(GNZ)])
    slab = np.hypot(qz[:, None, None], qx[None, None, :]) * np.ones((1, GNY, 1))
    wall = np.minimum(np.abs(cy - 128.0), np.abs(cy - 896.0))
    sdf = np.minimum(slab, wall[None, :, None]).astype(np.float32)
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None: (
                            sdf, np.zeros(3, np.float32), cell))


def _render(w, h, hfov, vfov, yaw=0.0, pitch=0.0):
    lid = vision.GpuLidar(None, w, h, hfov_deg=hfov, vfov_deg=vfov,
                          range_units=3000.0, cell=CELL, device="cpu")
    return lid.render(EYE, torch.tensor([yaw]), torch.tensor([pitch]),
                      torch.tensor([0.0]))[0]


# ------------------------------------------------------- 1. the default is old --
def test_gpulidar_defaults_are_the_numbers_the_trainer_defaults_to():
    sig = inspect.signature(vision.GpuLidar.__init__).parameters
    assert sig["hfov_deg"].default == LIDAR_HFOV == 120.0
    assert sig["vfov_deg"].default == LIDAR_VFOV == 90.0


def test_the_trainer_default_renders_the_pre_flag_frame(slab_world):
    """--lidar-hfov absent must be byte-for-byte the camera every existing
    checkpoint was trained on. Rendered, not just compared as angles."""
    old = _render(BASE_W, BASE_H, 120.0, 90.0, yaw=37.0, pitch=-23.0)
    new = vision.GpuLidar(None, BASE_W, BASE_H, hfov_deg=LIDAR_HFOV,
                          vfov_deg=LIDAR_VFOV, range_units=3000.0, cell=CELL,
                          device="cpu").render(
        EYE, torch.tensor([37.0]), torch.tensor([-23.0]), torch.tensor([0.0]))[0]
    assert torch.equal(old, new)
    assert 0.05 < (old < 0.99).float().mean() < 0.95, \
        "degenerate frame - the world shows nothing to compare"


# -------------------------------------------------- 2. widening is extension --
@pytest.mark.parametrize("hfov,w,off", [(240.0, 128, 32), (360.0, 192, 64),
                                        (180.0, 96, 16)])
def test_widening_fov_and_width_together_keeps_deg_per_column(
        slab_world, hfov, w, off):
    assert hfov / w == pytest.approx(120.0 / BASE_W)
    base = vision.GpuLidar(None, BASE_W, BASE_H, device="cpu")
    wide = vision.GpuLidar(None, w, BASE_H, hfov_deg=hfov, device="cpu")
    # the same rays, in the same order, at a known offset - not merely a
    # similar angular span
    assert torch.equal(wide.yoff[off:off + BASE_W], base.yoff)
    assert torch.equal(wide.poff, base.poff), "the vertical camera moved"


@pytest.mark.parametrize("hfov,w,off", [(240.0, 128, 32), (360.0, 192, 64)])
def test_the_central_columns_are_todays_frame_and_the_rest_is_addition(
        slab_world, hfov, w, off):
    base = _render(BASE_W, BASE_H, 120.0, 90.0, yaw=37.0, pitch=-23.0)
    wide = _render(w, BASE_H, hfov, 90.0, yaw=37.0, pitch=-23.0)
    assert torch.equal(wide[:, off:off + BASE_W], base), \
        "the central columns are not the frame the checkpoint was trained on"
    added = torch.cat([wide[:, :off], wide[:, off + BASE_W:]], dim=1)
    assert added.shape[1] == w - BASE_W
    assert not torch.allclose(added, added[:, :1].expand_as(added)), \
        "the new columns are a flat wall - this world cannot tell coverage " \
        "from a bug that aims every extra ray the same way"


def test_widening_at_a_fixed_width_would_rescale_instead(slab_world):
    """The control for the two tests above. Same fov change, width held -
    the frame is then a different sampling of the world at 2x the degrees
    per column, which is the confound the arm is built to avoid."""
    base = _render(BASE_W, BASE_H, 120.0, 90.0, yaw=37.0, pitch=-23.0)
    same_w = _render(BASE_W, BASE_H, 240.0, 90.0, yaw=37.0, pitch=-23.0)
    assert not torch.equal(same_w[:, 16:48], base)
    assert 240.0 / BASE_W == 2 * (120.0 / BASE_W)


def test_a_full_circle_covers_every_azimuth_once(slab_world):
    """hfov 360 is the wrap-around case: the first and last columns are
    adjacent in the world, and must not be the SAME ray."""
    lid = vision.GpuLidar(None, 192, BASE_H, hfov_deg=360.0, device="cpu")
    step = float(lid.yoff[0] - lid.yoff[1])
    wrap = float(lid.yoff[-1] + 2 * np.pi - lid.yoff[0])
    assert wrap == pytest.approx(step, rel=1e-4), \
        "the seam is not one pixel wide - the circle double-covers or gaps"


# ------------------------------------------------------------- 3. plumbing --
def _calls(path, func):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == func]


def test_the_trainer_hands_both_numbers_to_the_lidar():
    kw = {}
    for call in _calls(ROOT / "python" / "train_fast.py", "GpuLidar"):
        kw.update({k.arg: ast.unparse(k.value) for k in call.keywords})
    assert kw.get("hfov_deg") == "args.lidar_hfov"
    assert kw.get("vfov_deg") == "args.lidar_vfov"


def test_the_saved_config_records_the_camera():
    """A run.json without the fov is a run nobody can reproduce or resume:
    every restore path reads the config, not the command line."""
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    for key in ('"lidar_hfov": args.lidar_hfov', '"lidar_vfov": args.lidar_vfov'):
        assert key in src, f"the saved config never writes {key}"
    # ... and reads it back on resume, or a resumed arm silently reverts
    for key in ('ck_cfg.get("lidar_hfov")', 'ck_cfg.get("lidar_vfov")'):
        assert key in src, f"a resume never restores {key}"


def test_the_recorder_mirrors_the_camera():
    kw = {}
    for call in _calls(ROOT / "tools" / "record_ckpt.py", "GpuLidar"):
        kw.update({k.arg: ast.unparse(k.value) for k in call.keywords})
    assert "lidar_hfov" in (kw.get("hfov_deg") or "")
    assert "lidar_vfov" in (kw.get("vfov_deg") or "")


# --------------------------------------------------------- 4. the warm start --
@pytest.mark.parametrize("wide_w", [128, 192])
def test_a_checkpoint_loads_into_a_wider_model(wide_w):
    """AdaptiveAvgPool2d((4, 8)) is what makes the wide-fov arm a warm start
    rather than a fresh run: no weight has W in its shape."""
    narrow = Policy(N_SCALAR + BASE_W * BASE_H, BASE_W, BASE_H)
    wide = Policy(N_SCALAR + wide_w * BASE_H, wide_w, BASE_H)
    missing, unexpected = wide.load_state_dict(narrow.state_dict(), strict=True)
    assert not missing and not unexpected
    for k, v in narrow.state_dict().items():
        assert wide.state_dict()[k].shape == v.shape
    # and it runs: the trunk really does accept the wider image
    obs = torch.zeros(2, N_SCALAR + wide_w * BASE_H)
    logits, value = wide(obs)
    assert logits.shape[0] == 2 and value.shape == (2,)


def test_the_wider_model_is_not_the_same_function(slab_world):
    """Load-compatible is not function-identical: the pooled bins average a
    wider sector, so a resume that widens the fov has a transient. Anyone
    reading a flat first eval should know it was predicted."""
    torch.manual_seed(0)
    narrow = Policy(N_SCALAR + BASE_W * BASE_H, BASE_W, BASE_H).eval()
    wide = Policy(N_SCALAR + 128 * BASE_H, 128, BASE_H).eval()
    wide.load_state_dict(narrow.state_dict())
    base = _render(BASE_W, BASE_H, 120.0, 90.0, yaw=37.0, pitch=-23.0)
    wide_img = _render(128, BASE_H, 240.0, 90.0, yaw=37.0, pitch=-23.0)
    with torch.no_grad():
        a = narrow(torch.cat([torch.zeros(1, N_SCALAR),
                              base.reshape(1, -1)], 1))[1]
        b = wide(torch.cat([torch.zeros(1, N_SCALAR),
                            wide_img.reshape(1, -1)], 1))[1]
    assert not torch.allclose(a, b, atol=1e-4), \
        "the two cameras produced the same value - the test is vacuous"
