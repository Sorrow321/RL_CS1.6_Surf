"""--obs-potential (docs/obs_potential.md): the race potential as the depth
image's second channel, in four encodings, plus --obs-potential-curtain.

(a) LidarPotential.sample is GoalField.sample: the trilinear-over-honest-
    corners geodesic, the honesty mask, the sentinel, on a synthetic field
    (CPU; the codes round-trip the cached grid exactly).
(b) the encodings on hand values: abs = d_hit / d0 in [0, 1.5] with a bad
    sample at 1.5 and the eye ignored; rel = (d_eye - d_hit) / 2000 in
    [-2, 2], goal-ward POSITIVE, a bad sample or a bad eye at -2; with_mode
    shares the grid; the refusals (a field without a grid, a mode that is
    not abs/rel/norm, abs without d0).
(b') norm against a HAND computation on a synthetic frame set: the kernel
    tail's constants under norm (the abs tail at scale 1, unclipped, the
    bad marker -1, the eye ignored), then normalise(): per frame (d_hit -
    mean) / (std + 50 u) over the honest pixels with the population std,
    clipped to [-3, 3]; bad pixels +3; a frame with fewer than 8 honest
    pixels 0 EVERYWHERE (bad pixels included); exactly 8 is standardised;
    a flat frame reads 0 (the floor, not 0/0); the floor damps a
    near-flat frame's outlier; a far outlier clips at +3; zero mean and a
    std of s / (s + 50) over a frame's honest pixels.
(c) the renderer on a synthetic scene (CPU torch path): (N, H, W, 2), the
    depth channel bit-identical to the depth-only lidar, the potential
    sampled ONE FIELD CELL SHORT of the hit (recomputed from the march's
    own t), goal-ward rays positive under rel and backward rays negative,
    an unreachable surface at the bad value, channels == 2, and the
    exclusivity with --surf-mask / --normals / --pinhole (rel and norm).
(c') norm on the same scene: the channel is the hand rule applied to the
    raw abs sample recomputed from the march's own t, and normalise() of
    encode()'s raw output; unreachable rays +3, the honest ones zero-mean
    with the near (farther-from-goal) row positive; a view with no honest
    ray reads 0 everywhere where abs reads 1.5 and rel -2; a constant shift
    of the whole field leaves the channel unchanged (contrast without the
    level).
(d) CUDA, cannonball's caches: the triton tail agrees with the torch
    fallback pixel for pixel on both channels in all three modes, the depth
    channel is bit-exact against the single-channel kernel, the eye sample
    is GoalField.sample at the eye, and one cell back never lands in a
    wall on the fixture poses (16% of raw hit points do).
(b'') logabs against a HAND computation: the kernel tail's constants under
    logabs (norm's raw abs tail), then log_compress(): log1p(d / 1000 u) /
    log1p(d0 / 1000 u) clipped to [0, 1.5], a bad pixel 1.5; d0 reads
    exactly 1 and the goal exactly 0; strictly monotone in d below the
    ceiling and saturating at it, never above; the cannonball numbers the
    mode exists for (the wall 0.033 -> 0.38, the finish room 0.0055 ->
    0.13); with_mode carries d0 both ways. And on the synthetic scene: the
    hand rule on the raw abs sample recomputed from the march's own t, the
    same unreachable set as abs, >= abs inside the run and <= abs past d0
    (log1p is concave and both curves pass through (0, 0) and (d0, 1)), and
    the frame's ORDER unchanged.
(c'') the finish curtain, per mode, on the synthetic scene: OFF is
    bit-identical to a LidarPotential built without the argument at all AND
    to a curtain no ray can reach, and still equals encode(sample(one cell
    short)) + the post-process; ON, a 1-px view aimed straight down through
    a slab reads the GOAL (0 under abs/logabs, d_eye/2000 under rel) where
    the flag off reads the floor, with DEPTH unmoved; on a full frame only
    the rays the slab test catches move (norm excepted - it re-standardises
    the whole frame), and under norm the caught pixels are the frame's most
    goal-ward.
(d') CUDA, cannonball: logabs joins the triton-vs-fallback agreement (spawn
    frame ~1, the deep frame abs squeezes under 0.1 opens to 0.2-0.8), and
    the curtain runs on the real map in all four modes - a box below the
    map is bit-identical to the flag off, the map's own bounding box puts
    every ray at the goal, and a half-map box (a mixed mask) agrees between
    the triton tail and the torch fallback.
(e) trainer smokes (CPU, the toy scratch set of test_view_continuous), one
    per mode: the flag ON trains with finite losses at obs width 15 +
    2*16*8, writes obs_potential / obs_potential_d0 into run.json and the
    checkpoint, in_ch 2 in conv.0; record_ckpt.py mirrors the channel; a
    resume without the flag restores the mode; a resume asking for EITHER
    other mode is refused; --race-dist euclid and --surf-mask alongside it
    are refused (rel and norm). The flag OFF is byte-identical to the
    pre-flag trainer by construction (no key is written when it is off)
    and pinned by test_unstuck.py's flag-off identity against the
    git-history trainer.
(e') a curtain smoke: --obs-potential-curtain reaches the renderer with the
    slot's finish box, writes obs_potential_curtain into run.json and the
    checkpoint, record_ckpt rebuilds the box from the map's zones, a resume
    without the flag restores it, and the flag without --obs-potential is
    refused.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym import vision                                     # noqa: E402
from surfgym.goalfield import EuclidField, GoalField           # noqa: E402
from surfgym.vision import (POTENTIAL_LOG_SCALE,               # noqa: E402
                            POTENTIAL_MODES, LidarPotential)
from test_view_continuous import (CANNONBALL, MAPS, SMOKE_FLAGS,  # noqa: E402
                                  _env, needs_run)

TRAIN = ROOT / "python" / "train_fast.py"
RECORD = ROOT / "tools" / "record_ckpt.py"
ABS = ["--view-continuous", "--view-absolute", "velocity"]


# ------------------------------------------------------------ a synthetic field
CELL = 8.0
NX = NY = NZ = 16                      # a 128 u cube at mins 0


def _field(goal_x: float = 1000.0):
    """d = goal_x - 10 x (the goal is at +x), the slab x >= 96 u and one
    lone voxel unreachable (sentinel). Values are multiples of the cache's
    quant (cell / 8) exactly as build_goal_field stores them."""
    x = (np.arange(NX) + 0.5) * CELL
    grid = np.broadcast_to(goal_x - 10.0 * x[None, None, :],
                           (NZ, NY, NX)).astype(np.float32).copy()
    reach_max = float(grid.max())
    sent = reach_max + 2.0 * CELL
    grid[:, :, 12:] = sent
    grid[3, 5, 2] = sent
    q = CELL / 8.0
    grid = (np.rint(grid / q) * q).astype(np.float32)
    return GoalField(grid, np.zeros(3), CELL, reach_max)


def test_sample_is_goalfield_sample():
    gf = _field()
    P = LidarPotential(gf, "rel", device="cpu")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-8.0, 136.0, size=(4000, 3))
    ref = gf.sample(pts)
    val, ok = P.sample(torch.as_tensor(pts, dtype=torch.float32))
    assert np.abs(val.numpy() - ref).max() < 1e-2          # f32 vs f64 positions
    assert np.array_equal(ok.numpy(), ref < gf._valid_max)
    assert (~ok).sum() > 100 and ok.sum() > 100, "the fixture is degenerate"
    assert torch.all(val[~ok] == gf.sentinel)
    # the codes ARE the cached grid: code * quant == grid, bit for bit
    back = ((P.codes.to(torch.int32) & 0xFFFF).to(torch.float32) * P.quant)
    assert torch.equal(back.reshape(gf.grid.shape), torch.as_tensor(gf.grid))


def test_encodings_and_with_mode():
    gf = _field()
    rel = LidarPotential(gf, "rel", device="cpu")
    ab = rel.with_mode("abs", d0=900.0)
    assert ab.codes is rel.codes and ab.mode == "abs" and rel.mode == "rel"
    vhit = torch.tensor([500.0, 700.0, 0.0, 1e9, 1e9])
    ok = torch.tensor([True, True, True, False, True])
    deye = torch.tensor([600.0])
    # rel: goal-ward (a lower field ahead) POSITIVE, backward negative,
    # clipped at +-2, a bad sample -2
    assert rel.encode(vhit, ok, deye).tolist() == pytest.approx(
        [0.05, -0.05, 0.3, -2.0, -2.0])
    # abs: d_hit / d0, clipped to [0, 1.5], a bad sample 1.5, eye ignored
    assert ab.encode(vhit, ok, deye).tolist() == pytest.approx(
        [500 / 900, 700 / 900, 0.0, 1.5, 1.5])
    assert ab.encode(vhit, ok, torch.tensor([gf.sentinel])).tolist() \
        == pytest.approx([500 / 900, 700 / 900, 0.0, 1.5, 1.5])
    # rel with an eye in unreachable space: nothing in view is goal-ward
    assert rel.encode(vhit, ok, torch.tensor([gf.sentinel])).tolist() \
        == [-2.0] * 5
    assert (rel.lo, rel.hi, rel.bad, rel.scale) == (-2.0, 2.0, -2.0, 2000.0)
    assert (ab.lo, ab.hi, ab.bad, ab.scale) == (0.0, 1.5, 1.5, 900.0)
    assert not rel.norm and not ab.norm
    assert "goal-ward POSITIVE" in rel.describe() and "d0 900" in ab.describe()
    # with_mode round-trips through norm and back without losing the grid
    nm = ab.with_mode("norm")
    assert nm.codes is rel.codes and nm.mode == "norm" and nm.norm
    assert nm.with_mode("abs").scale == 900.0          # d0 rides along


def test_refusals():
    gf = _field()
    with pytest.raises(ValueError):
        LidarPotential(gf, "signed", device="cpu")
    with pytest.raises(ValueError):
        LidarPotential(gf, "abs", d0=None, device="cpu")
    with pytest.raises(ValueError):
        LidarPotential(gf, "abs", d0=0.0, device="cpu")
    with pytest.raises(TypeError):
        LidarPotential(EuclidField({"mins": [0, 0, 0], "maxs": [1, 1, 1]}),
                       "rel", device="cpu")
    # norm and rel need no d0; abs does, also through with_mode
    nm = LidarPotential(gf, "norm", device="cpu")
    assert nm.d0 is None and nm.with_mode("rel").d0 is None
    with pytest.raises(ValueError):
        nm.with_mode("abs")
    with pytest.raises(ValueError):
        nm.with_mode("signed")
    assert POTENTIAL_MODES == ("abs", "rel", "norm", "logabs")
    # logabs needs d0 exactly as abs does
    with pytest.raises(ValueError):
        LidarPotential(gf, "logabs", d0=None, device="cpu")
    with pytest.raises(ValueError):
        LidarPotential(gf, "logabs", d0=0.0, device="cpu")
    with pytest.raises(ValueError):
        nm.with_mode("logabs")
    # a curtain whose mins are above its maxs is a typo, not a box
    with pytest.raises(ValueError):
        LidarPotential(gf, "rel", device="cpu",
                       curtain={"mins": [10, 0, 0], "maxs": [0, 1, 1]})


# ------------------------------------------------------- norm, by hand
def _hand_norm(raw, valid, eps=50.0, clip=3.0, min_valid=8):
    """The norm rule written out in numpy float64, independently of
    LidarPotential.normalise: per frame, the honest pixels' mean and
    POPULATION std, (d - mean) / (std + eps) clipped to [-clip, clip], a
    bad pixel +clip, a frame under min_valid honest pixels 0 everywhere."""
    raw = np.asarray(raw, np.float64)
    valid = np.asarray(valid, bool)
    out = np.zeros(raw.shape, np.float32)
    for i in range(raw.shape[0]):
        v = raw[i][valid[i]]
        if v.size < min_valid:
            continue
        z = np.clip((raw[i] - v.mean()) / (v.std() + eps), -clip, clip)
        z[~valid[i]] = clip
        out[i] = z
    return out


def test_norm_encoding_against_a_hand_computation():
    gf = _field()
    P = LidarPotential(gf, "norm", device="cpu")          # needs no d0
    assert P.norm and not P.rel and P.d0 is None
    # the kernel tail's runtime constants under norm: the abs tail at
    # scale 1 (raw map units), unclipped (valid_max is above every honest
    # value), the bad marker OUT OF BAND at -1
    assert (P.lo, P.hi, P.bad, P.scale, P.scale_inv) \
        == (0.0, gf._valid_max, -1.0, 1.0, 1.0)
    assert (P.norm_eps, P.norm_clip, P.norm_min_valid, P.norm_bad) \
        == (50.0, 3.0, 8, 3.0)
    vhit = torch.tensor([500.0, 700.0, 0.0, 1e9, 1e9])
    ok = torch.tensor([True, True, True, False, True])
    raw = P.encode(vhit, ok, torch.tensor([600.0]))
    assert raw.tolist() == pytest.approx([500.0, 700.0, 0.0, -1.0, gf._valid_max])
    # the eye is ignored, like abs (unlike rel)
    assert torch.equal(P.encode(vhit, ok, torch.tensor([gf.sentinel])), raw)
    assert "farther-from-goal POSITIVE" in P.describe() \
        and "unreachable +3" in P.describe() and "under 8 honest" in P.describe()

    # the post-process against the hand rule on a synthetic frame set
    rng = np.random.default_rng(1)
    N, H, W = 6, 4, 6                                     # 24 pixels a frame
    raw = rng.uniform(1000.0, 9000.0, size=(N, H, W))
    valid = np.ones((N, H, W), bool)
    valid[0, 0, :3] = False                               # 21 honest of 24
    valid[1] = False
    valid[1].reshape(-1)[:7] = True                       # 7 honest: too few
    valid[2] = False
    valid[2].reshape(-1)[:8] = True                       # exactly 8: enough
    raw[3] = 4000.0                                       # flat
    raw[4] = 1000.0
    raw[4, 0, 0] = 1100.0
    valid[4] = False
    valid[4].reshape(-1)[:9] = True                       # 8 x 1000 + 1 x 1100
    raw[5] = 5000.0
    raw[5, 2, 3] = 90000.0                                # one far outlier
    ch = raw.astype(np.float32).copy()
    ch[~valid] = -1.0
    out = P.normalise(torch.as_tensor(ch)).numpy()
    want = _hand_norm(raw, valid)
    assert out.shape == (N, H, W) and out.dtype == np.float32
    assert np.allclose(out, want, atol=1e-6), np.abs(out - want).max()
    # the rules, one at a time
    assert np.all(out[0][~valid[0]] == 3.0), "a bad pixel reads +3"
    assert np.all(out[0][valid[0]] < 3.0)
    v0 = out[0][valid[0]]
    s0 = raw[0][valid[0]].std()
    assert abs(v0.mean()) < 1e-6, "zero mean over the honest pixels"
    assert abs(v0.std() - s0 / (s0 + 50.0)) < 1e-6, "a std of s / (s + 50)"
    assert np.all(out[1] == 0.0), \
        "under 8 honest pixels the WHOLE frame reads 0, bad pixels included"
    assert np.all(out[2][~valid[2]] == 3.0) and np.any(out[2][valid[2]] != 0.0), \
        "exactly 8 honest pixels is standardised"
    assert np.all(out[3] == 0.0), "a flat frame reads 0 (the floor, not 0/0)"
    # the +50 u floor: 8 pixels at 1000 u and one at 1100 u standardise to
    # 88.9 / (31.4 + 50) = 1.09, not the 2.83 a bare z-score would give
    v4 = raw[4][valid[4]]
    z_bare = (1100.0 - v4.mean()) / v4.std()
    assert 2.8 < z_bare < 2.9 and 1.05 < out[4, 0, 0] < 1.15
    assert np.all(out[4][~valid[4]] == 3.0)
    # the far outlier clips at +3; the rest of that frame is one negative value
    assert out[5, 2, 3] == 3.0
    rest = np.delete(out[5].reshape(-1), 2 * W + 3)
    assert np.all(rest < 0.0) and np.ptp(rest) == 0.0
    # a frame of nothing but bad pixels
    empty = torch.full((1, H, W), -1.0)
    assert torch.equal(P.normalise(empty), torch.zeros(1, H, W))


# ---------------------------------------------------------- a synthetic scene
# test_normals' world at cell 8 on the field's grid: a floor (voxel layer 0
# solid) and nothing else, so the field's unreachable slab (x >= 96 u) is
# AIR the rays fly through - a ray that lands on the floor past x = 96
# samples unreachable space, one that lands short of it samples the air one
# cell above the floor. (A solid wall there would never be sampled: the
# sample is one cell SHORT of the hit, on the near side.)
RNX, RNY, RNZ = NX, NY, NZ


def _scene_sdf():
    k = np.arange(RNZ, dtype=np.float32)[:, None, None]
    sdf = np.broadcast_to(k * CELL, (RNZ, RNY, RNX)).copy()
    return sdf.astype(np.float32)


@pytest.fixture
def scene(monkeypatch):
    sdf = _scene_sdf()
    mins = np.zeros(3, dtype=np.float32)
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None: (sdf, mins, cell))
    return sdf


def _pose(x=16.0, y=64.0, z=47.0, yaw=0.0, pitch=0.0, duck=0):
    a = np.array([[x, y, z, yaw, pitch, duck]], dtype=np.float32)
    return (torch.as_tensor(a[:, 0:3]), torch.as_tensor(a[:, 3]),
            torch.as_tensor(a[:, 4]), torch.as_tensor(a[:, 5]))


def _poses(*ps):
    parts = [_pose(**p) for p in ps]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(4))


# eye at (16, 64, 64): facing +x and 45 deg down, the top-centre ray lands
# ~150 u out (past x = 96: unreachable) and the bottom row within 27 u
# (honest); facing -x every ray lands on floor farther from the goal than
# the eye (the grid clamps past x < 0 to its first column, still honest)
VIEWS = ({"yaw": 0.0, "pitch": -45.0}, {"yaw": 180.0, "pitch": -45.0},
         {"pitch": -90.0}, {"yaw": 90.0, "pitch": -20.0, "duck": 1})
# ... and, for norm, an eye deep in the unreachable slab looking straight
# down: every ray lands past x = 96 (the widest ray 97 u out at 56 u up),
# so the frame has no honest pixel at all
EMPTY_VIEW = {"x": 240.0, "pitch": -90.0}


def test_render_on_a_synthetic_scene(scene):
    gf = _field()
    d0 = 900.0
    P = LidarPotential(gf, "rel", device="cpu")
    off = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256)
    on = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256,
                         potential=P)
    assert (off.channels, on.channels) == (1, 2)
    tensors = {k for k, v in vars(off).items() if torch.is_tensor(v)}
    assert {k for k, v in vars(on).items() if torch.is_tensor(v)} == tensors, \
        "the potential grid belongs to the LidarPotential, not the lidar"
    p = _poses(*VIEWS)
    d = off.render(*p)
    out = on.render(*p)
    assert out.shape == (len(VIEWS), 8, 16, 2)
    assert torch.equal(out[..., 0], d), "the potential perturbed depth"
    rel = out[..., 1]
    # facing +x (the goal), 30 deg down: every honest ray reads goal-ward
    # (positive), the rays landing past x = 96 read the bad value
    hit = d[0] < 1.0
    assert hit.all(), "the floor should stop every ray of a downward view"
    bad = rel[0] == -2.0
    assert bad.any() and (~bad).any(), "the view must split honest / unreachable"
    assert torch.all(rel[0][~bad] > 0.0)
    assert bad[0, 8] and not bad[-1].any(), \
        "the top-centre ray lands far (unreachable), the bottom row short (honest)"
    # facing -x: every landing is farther from the goal than the eye
    assert (d[1] < 1.0).all()
    assert torch.all(rel[1] < 0.0) and torch.all(rel[1] > -2.0)
    # straight down, the centre rows: one cell above the floor under the
    # eye, so d_hit is d_eye to within the ray's few units of drift and
    # never the bad value (the hit itself is INSIDE the floor voxel)
    centre = rel[2, 3:5, :]
    assert torch.all(centre != -2.0)
    assert centre.abs().max() < 0.05
    # the sample point: ONE FIELD CELL SHORT of the hit along the ray,
    # recomputed from the march's own t (the fallback keeps it in _t)
    t = torch.clamp(on._t, max=on.range)
    ts = torch.clamp(t - P.cell, min=0.0)
    N = len(VIEWS)
    ex, ey = p[0][:, 0].view(N, 1, 1), p[0][:, 1].view(N, 1, 1)
    ez = (p[0][:, 2] + torch.where(p[3].bool(), 12.0, 17.0)).view(N, 1, 1)
    pts = torch.stack((ex + on._dx * ts, ey + on._dy * ts, ez + on._dz * ts),
                      -1).reshape(-1, 3)
    vhit, ok = P.sample(pts)
    deye = P.eye(p[0], p[3]).view(N, 1, 1)
    want = P.encode(vhit.reshape(N, 8, 16), ok.reshape(N, 8, 16), deye)
    assert torch.equal(rel, want)
    # abs on the same rays: d_hit / d0 on the honest ones, 1.5 on the rest
    on.potential = P.with_mode("abs", d0=d0)
    ab = on.render(*p)[..., 1]
    assert torch.equal(ab[rel == -2.0], torch.full_like(ab[rel == -2.0], 1.5))
    good = rel != -2.0
    assert torch.allclose(ab[good], torch.clamp(vhit.reshape(N, 8, 16)[good] / d0,
                                                0.0, 1.5), atol=1e-6)
    assert torch.all(ab[good] < 1.5)


def test_norm_render_on_a_synthetic_scene(scene):
    gf = _field()
    P = LidarPotential(gf, "norm", device="cpu")
    off = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256)
    on = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256,
                         potential=P)
    assert on.channels == 2
    views = VIEWS + (EMPTY_VIEW,)
    N = len(views)
    p = _poses(*views)
    d = off.render(*p)
    out = on.render(*p)
    assert out.shape == (N, 8, 16, 2) and out.dtype == torch.float32
    assert torch.equal(out[..., 0], d), "the post-process touched depth"
    ch = out[..., 1]
    assert torch.all(ch >= -3.0) and torch.all(ch <= 3.0)
    # the raw abs sample one field cell short of the hit, recomputed from
    # the march's own t, and the hand rule on it
    t = torch.clamp(on._t, max=on.range)
    ts = torch.clamp(t - P.cell, min=0.0)
    ex, ey = p[0][:, 0].view(N, 1, 1), p[0][:, 1].view(N, 1, 1)
    ez = (p[0][:, 2] + torch.where(p[3].bool(), 12.0, 17.0)).view(N, 1, 1)
    pts = torch.stack((ex + on._dx * ts, ey + on._dy * ts, ez + on._dz * ts),
                      -1).reshape(-1, 3)
    vhit, ok = P.sample(pts)
    vhit, ok = vhit.reshape(N, 8, 16), ok.reshape(N, 8, 16)
    want = _hand_norm(vhit.numpy(), ok.numpy())
    assert np.allclose(ch.numpy(), want, atol=1e-5), np.abs(ch.numpy() - want).max()
    # ... which is normalise() of encode()'s raw output (d_hit in u, bad -1)
    raw = P.encode(vhit, ok, P.eye(p[0], p[3]).view(N, 1, 1))
    assert torch.equal(raw[~ok], torch.full_like(raw[~ok], -1.0))
    assert torch.all(raw[ok] >= 0.0)
    assert torch.equal(ch, P.normalise(raw))
    # facing +x (the goal), 45 deg down: the rays past x = 96 are
    # unreachable and read +3 (the top of the range: "the farthest thing in
    # view"); the honest ones average 0, the bottom row - the nearest
    # landings, the farthest from the goal - positive and the far honest
    # landings negative (larger = farther from the goal, abs's sign)
    bad = ~ok[0]
    assert bad.any() and (~bad).any()
    assert torch.all(ch[0][bad] == 3.0)
    assert ch[0][~bad].mean().abs() < 1e-4
    assert ch[0][-1].mean() > 0.0 and (ch[0][~bad] < 0.0).any()
    # facing -x: every landing honest, a real spread (rows land at
    # different x), so the frame is a zero-mean picture with std s/(s+50)
    assert ok[1].all()
    s1 = vhit[1].double().std(unbiased=False).item()
    assert s1 > 10.0                           # not flat: the floor does not dominate
    assert ch[1].mean().abs() < 1e-4
    assert abs(ch[1].double().std(unbiased=False).item() - s1 / (s1 + 50.0)) < 1e-4
    # the empty view: no honest pixel, so the whole frame reads 0 - where
    # abs reads 1.5 and rel -2 everywhere
    assert not ok[-1].any()
    assert torch.all(ch[-1] == 0.0)
    on.potential = P.with_mode("abs", d0=900.0)
    assert torch.all(on.render(*p)[-1, ..., 1] == 1.5)
    on.potential = P.with_mode("rel")
    assert torch.all(on.render(*p)[-1, ..., 1] == -2.0)
    # contrast without the level: the same field shifted by a constant
    # (the goal 4,000 u farther) renders the SAME norm channel
    on.potential = LidarPotential(_field(goal_x=5000.0), "norm", device="cpu")
    ch2 = on.render(*p)[..., 1]
    assert torch.allclose(ch2, ch, atol=1e-4), (ch2 - ch).abs().max()


# ------------------------------------------------------- logabs, by hand
def _hand_logabs(raw, valid, d0, scale=1000.0, clip=1.5):
    """logabs written out in numpy float64, independently of
    LidarPotential.log_compress: log1p(d / 1000 u) / log1p(d0 / 1000 u)
    clipped to [0, 1.5], a bad pixel 1.5. Pointwise - no frame."""
    raw = np.asarray(raw, np.float64)
    valid = np.asarray(valid, bool)
    z = np.clip(np.log1p(np.maximum(raw, 0.0) / scale) / np.log1p(d0 / scale),
                0.0, clip)
    return np.where(valid, z, clip).astype(np.float32)


def test_logabs_encoding_against_a_hand_computation():
    gf = _field()
    d0 = 900.0
    P = LidarPotential(gf, "logabs", d0=d0, device="cpu")
    assert P.logabs and P.post and not P.norm and not P.rel
    assert P.d0 == d0 and P.log_scale == POTENTIAL_LOG_SCALE == 1000.0
    # the kernel tail's constants: norm's raw abs tail (scale 1, unclipped,
    # the bad marker -1) - the post-process does the rest
    assert (P.lo, P.hi, P.bad, P.scale, P.scale_inv) \
        == (0.0, gf._valid_max, -1.0, 1.0, 1.0)
    assert P.log_clip == P.log_bad == 1.5
    vhit = torch.tensor([900.0, 500.0, 0.0, 1e9, 1e9])
    ok = torch.tensor([True, True, True, False, True])
    raw = P.encode(vhit, ok, torch.tensor([600.0]))
    assert raw.tolist() == pytest.approx([900.0, 500.0, 0.0, -1.0, gf._valid_max])
    ch = P.log_compress(raw.view(1, 1, -1)).view(-1)
    assert ch.tolist() == pytest.approx(
        _hand_logabs(raw.numpy(), raw.numpy() >= 0.0, d0).tolist(), abs=1e-6)
    # d0 reads exactly 1 (the spawn), the goal exactly 0, a bad pixel 1.5
    assert ch[0].item() == pytest.approx(1.0, abs=1e-6)
    assert ch[2].item() == 0.0
    assert ch[3].item() == 1.5
    assert P.postprocess(raw.view(1, 1, -1)).view(-1).tolist() == ch.tolist()
    # STRICTLY monotone in d below the ceiling, and in [0, 1.5] above it
    d = torch.linspace(0.0, 1600.0, 512).view(1, 1, -1)
    m = P.log_compress(d).view(-1)
    assert torch.all(m[1:] > m[:-1]) and m.max() < 1.5
    far = P.log_compress(torch.linspace(0.0, 40.0 * d0, 512).view(1, 1, -1))
    assert far.min() == 0.0 and far.max() == 1.5      # saturates, never above
    # the point of the mode: the compressed end of abs is spread out. On
    # cannonball d0 = 198,380 u; the wall sits at 6,568 u and the finish
    # room walls at 550-1,100 u, which abs squeezes under 0.034
    C = LidarPotential(gf, "logabs", d0=198_380.0, device="cpu")
    v = C.log_compress(torch.tensor([[[198_380.0, 6_568.0, 1_100.0, 550.0,
                                       0.0]]])).view(-1)
    assert v[0].item() == pytest.approx(1.0, abs=1e-6)
    assert 0.35 < v[1].item() < 0.42          # the wall, 0.033 under abs
    assert 0.13 < v[2].item() < 0.16          # the finish room, 0.0055 under abs
    assert 0.07 < v[3].item() < 0.10
    assert v[4].item() == 0.0                 # the goal
    # with_mode carries d0 into logabs and back
    ab = C.with_mode("abs")
    assert ab.mode == "abs" and ab.scale == 198_380.0
    assert ab.with_mode("logabs").log_denom == C.log_denom
    assert "log1p(d_hit / 1000 u)" in C.describe() and "d0 198,380" in C.describe()


def test_logabs_render_on_a_synthetic_scene(scene):
    gf = _field()
    d0 = 900.0
    P = LidarPotential(gf, "logabs", d0=d0, device="cpu")
    off = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256)
    on = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256,
                         potential=P)
    assert on.channels == 2
    N = len(VIEWS)
    p = _poses(*VIEWS)
    d = off.render(*p)
    out = on.render(*p)
    assert out.shape == (N, 8, 16, 2) and out.dtype == torch.float32
    assert torch.equal(out[..., 0], d), "the post-process touched depth"
    ch = out[..., 1]
    assert torch.all(ch >= 0.0) and torch.all(ch <= 1.5)
    # the raw abs sample one field cell short of the hit, the hand rule on it
    t = torch.clamp(on._t, max=on.range)
    ts = torch.clamp(t - P.cell, min=0.0)
    ex, ey = p[0][:, 0].view(N, 1, 1), p[0][:, 1].view(N, 1, 1)
    ez = (p[0][:, 2] + torch.where(p[3].bool(), 12.0, 17.0)).view(N, 1, 1)
    pts = torch.stack((ex + on._dx * ts, ey + on._dy * ts, ez + on._dz * ts),
                      -1).reshape(-1, 3)
    vhit, ok = P.sample(pts)
    vhit, ok = vhit.reshape(N, 8, 16), ok.reshape(N, 8, 16)
    want = _hand_logabs(vhit.numpy(), ok.numpy(), d0)
    assert np.allclose(ch.numpy(), want, atol=1e-6), np.abs(ch.numpy() - want).max()
    # unreachable rays read the ceiling, exactly where abs reads its own
    on.potential = P.with_mode("abs", d0=d0)
    ab = on.render(*p)[..., 1]
    assert torch.equal(ch == 1.5, ab == 1.5)
    # honest pixel by honest pixel: log1p is CONCAVE and both curves pass
    # through (0, 0) and (d0, 1), so logabs sits ABOVE abs everywhere
    # inside the run (that is the lift the mode exists for) and below it
    # past the start; the two orders agree on the whole honest set
    lo_g = ok & (ab <= 1.0)
    hi_g = ok & (ab > 1.0)
    assert lo_g.any() and hi_g.any(), "the fixture must span d0"
    assert torch.all(ch[lo_g] >= ab[lo_g] - 1e-6)
    assert torch.all(ch[hi_g] <= ab[hi_g] + 1e-6)
    assert torch.all((ch[ok] > 0.0) == (ab[ok] > 0.0))
    o = ch[ok][torch.argsort(ab[ok])]
    assert torch.all(o[1:] - o[:-1] >= -1e-6), "logabs reordered the frame"


# ------------------------------------------------- the finish curtain
# a slab the straight-down ray of VIEWS[2] crosses at t ~ 33 u, well before
# it lands on the floor at t ~ 58 u
CURTAIN_BOX = {"mins": [10.0, 60.0, 30.0], "maxs": [22.0, 68.0, 31.0]}
# ... and one no ray can reach: the flag ON must then be bit-identical too
FAR_BOX = {"mins": [-1e4, -1e4, -1000.0], "maxs": [1e4, 1e4, -992.0]}


@pytest.mark.parametrize("mode", list(POTENTIAL_MODES))
def test_curtain_off_is_bit_identical(scene, mode):
    """The flag OFF renders exactly what the renderer rendered before it:
    the same channel as a LidarPotential built without the argument at all,
    and the same as a curtain no ray can enter (the slab test's negative
    direction). The kernel's block is behind a constexpr, so on the triton
    path 'off' does not even compile it."""
    gf = _field()
    d0 = 900.0
    p = _poses(*VIEWS)

    def render(curtain):
        kw = {} if curtain is ... else {"curtain": curtain}
        P = LidarPotential(gf, mode, d0=d0, device="cpu", **kw)
        lid = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu",
                              max_steps=256, potential=P)
        return lid.render(*p), P

    ref, P0 = render(...)                       # the pre-flag signature
    assert P0.curtain is None
    off, _ = render(None)
    assert torch.equal(ref, off)
    far, Pf = render(FAR_BOX)
    assert Pf.curtain == ((-1e4, -1e4, -1000.0), (1e4, 1e4, -992.0))
    assert torch.equal(ref, far), "a curtain out of reach changed a pixel"
    # ... and the reference itself is still encode(sample(one cell short))
    lid = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256,
                          potential=P0)
    _ = lid.render(*p)
    N = len(VIEWS)
    t = torch.clamp(lid._t, max=lid.range)
    ts = torch.clamp(t - P0.cell, min=0.0)
    ex, ey = p[0][:, 0].view(N, 1, 1), p[0][:, 1].view(N, 1, 1)
    ez = (p[0][:, 2] + torch.where(p[3].bool(), 12.0, 17.0)).view(N, 1, 1)
    pts = torch.stack((ex + lid._dx * ts, ey + lid._dy * ts, ez + lid._dz * ts),
                      -1).reshape(-1, 3)
    vhit, ok = P0.sample(pts)
    want = P0.encode(vhit.reshape(N, 8, 16), ok.reshape(N, 8, 16),
                     P0.eye(p[0], p[3]).view(N, 1, 1))
    if P0.post:
        want = P0.postprocess(want)
    assert torch.equal(ref[..., 1], want)


@pytest.mark.parametrize("mode", list(POTENTIAL_MODES))
def test_curtain_catches_a_ray_through_the_box(scene, mode):
    """One ray aimed straight down through the box reads the GOAL; the same
    ray with the flag off reads the floor's own field value. Depth never
    moves - the box is not geometry."""
    gf = _field()
    d0 = 900.0
    p = _pose(**VIEWS[2])                      # straight down from (16, 64, 64)
    outs = {}
    for name, box in (("off", None), ("on", CURTAIN_BOX)):
        P = LidarPotential(gf, mode, d0=d0, device="cpu", curtain=box)
        lid = vision.GpuLidar(None, 1, 1, cell=CELL, device="cpu",
                              max_steps=256, potential=P)
        outs[name] = lid.render(*p)[0, 0, 0]
        if name == "on":
            t = torch.clamp(lid._t, max=lid.range)
            hit = P.curtain_mask(
                p[0][:, 0].view(1, 1, 1), p[0][:, 1].view(1, 1, 1),
                (p[0][:, 2] + 17.0).view(1, 1, 1),
                lid._dx, lid._dy, lid._dz, t)
            assert bool(hit.view(-1)[0]), "the box is not on the ray"
            assert t.view(-1)[0] > 40.0, "the ray must reach the floor AFTER the box"
    assert outs["off"][0] == outs["on"][0], "the curtain moved depth"
    off, on = float(outs["off"][1]), float(outs["on"][1])
    if mode == "norm":
        # a 1-pixel frame has fewer than 8 honest pixels, so norm reads 0
        # either way - the mode is checked on the full frame below
        assert on == 0.0 and off == 0.0
    else:
        goal = {"abs": 0.0, "logabs": 0.0, "rel": 840.0 / 2000.0}[mode]
        assert on == pytest.approx(goal, abs=2e-2), (mode, on)
        assert abs(on - off) > 0.05, (mode, on, off)
    # a full frame: only the rays through the box read the goal, and under
    # norm that is the frame's most goal-ward value
    P = LidarPotential(gf, mode, d0=d0, device="cpu", curtain=CURTAIN_BOX)
    lid = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu", max_steps=256,
                          potential=P)
    ch = lid.render(*p)[0, ..., 1]
    t = torch.clamp(lid._t, max=lid.range)
    caught = P.curtain_mask(p[0][:, 0].view(1, 1, 1), p[0][:, 1].view(1, 1, 1),
                            (p[0][:, 2] + 17.0).view(1, 1, 1),
                            lid._dx, lid._dy, lid._dz, t)[0]
    lid.potential = P.with_mode(mode, curtain=None)
    ch0 = lid.render(*p)[0, ..., 1]
    assert caught.any() and not caught.all(), "the box must catch SOME rays"
    # norm re-standardises the WHOLE frame when any pixel moves, so only
    # the other three modes leave the uncaught pixels alone
    if mode != "norm":
        assert torch.equal(ch[~caught], ch0[~caught])
    if mode == "rel":
        assert torch.all(ch[caught] > ch0[caught])          # goal-ward
    elif mode == "norm":
        assert ch[caught].max() < ch[~caught].min()         # most goal-ward
    else:
        assert torch.all(ch[caught] < ch0[caught])          # nearer the goal
        assert torch.all(ch[caught] == 0.0)


@pytest.mark.parametrize("mode", ["rel", "norm"])
def test_exclusive_with_the_other_vision_experiments(scene, mode):
    P = LidarPotential(_field(), mode, device="cpu")
    for kw in ({"surf_mask": True}, {"normals": True}, {"pinhole": True}):
        with pytest.raises(ValueError):
            vision.GpuLidar(None, 8, 4, cell=CELL, device="cpu", potential=P,
                            **kw)
    # a grid on the wrong device is refused before anything is uploaded
    with pytest.raises(ValueError, match="LidarPotential is on cpu"):
        vision.GpuLidar(None, 8, 4, cell=CELL, device="cuda", potential=P)


# ------------------------------------------------------------- CUDA, cannonball
GOAL32 = MAPS / "surf_src_cannonball.goal_32.npz"
SDF32 = MAPS / "surf_src_cannonball.sdf_32.npz"
needs_cuda_caches = pytest.mark.skipif(
    not (torch.cuda.is_available() and vision.HAVE_TRITON
         and CANNONBALL.exists() and GOAL32.exists() and SDF32.exists()),
    reason="needs CUDA + triton + cannonball's baked SDF and goal field "
           "(SURF_TEST_MAPS from a worktree)")

# (x, y, z, yaw, pitch, ducked): the spawn, four states of a recorded
# absolute-view episode (cyABSV, docs/potential_view.png), a ducked one
# and a sky-heavy view
CPOSES = [(-14336.0, 3078.86, 10528.0, 292.03, -10.0, 0),
          (-5727.1, 376.15, 4191.11, 269.58, -13.34, 0),
          (-4448.3, 885.2, 1884.56, 34.69, -15.95, 0),
          (7526.5, 600.14, 2129.96, 153.21, -4.57, 0),
          (-908.13, -3017.16, -4837.39, 342.51, -36.21, 0),
          (-2666.57, 2971.95, -1453.38, 229.5, -28.52, 1),
          (-14200.0, 2898.0, 10700.0, 90.0, 25.0, 0)]


@needs_cuda_caches
def test_triton_tail_agrees_with_the_fallback_on_cannonball():
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import build_goal_field
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0,
                                                    lidar_h=0))
    gf = build_goal_field(core, load_zones(str(CANNONBALL))["end"], cell=32.0)
    d0 = float(np.mean(gf.sample(map_spawn_pool(core)["origin"])))
    assert 190_000 < d0 < 210_000
    a = np.asarray(CPOSES, np.float32)
    kw = dict(range_units=11500.0, near_range=2000.0, cell=32.0)
    Pg = LidarPotential(gf, "abs", d0=d0, device="cuda")
    Pc = LidarPotential(gf, "abs", d0=d0, device="cpu")
    assert Pg.quant == Pc.quant == 4.0 and Pg.codes.numel() == gf.grid.size
    lg = vision.GpuLidar(core, 64, 32, device="cuda", potential=Pg, **kw)
    lc = vision.GpuLidar(core, 64, 32, device="cpu", potential=Pc, **kw)
    plain = vision.GpuLidar(core, 64, 32, device="cuda", **kw)

    def tens(dev):
        return (torch.as_tensor(a[:, 0:3]).to(dev), torch.as_tensor(a[:, 3]).to(dev),
                torch.as_tensor(a[:, 4]).to(dev), torch.as_tensor(a[:, 5]).to(dev))

    dref = plain.render(*tens("cuda"))
    for mode in POTENTIAL_MODES:
        lg.potential, lc.potential = Pg.with_mode(mode), Pc.with_mode(mode)
        og = lg.render(*tens("cuda")).cpu()
        oc = lc.render(*tens("cpu"))
        assert og.shape == oc.shape == (len(CPOSES), 32, 64, 2)
        # depth: the single-channel kernel's pixels, bit for bit
        assert torch.equal(og[..., 0], dref.cpu()), "depth ABI drift"
        # the two paths: cos/sin differ in the last ulp between them, which
        # can move a voxel boundary by one ray. Under norm that one ray
        # also moves its frame's mean and std, i.e. every pixel of the
        # frame by ~1/2048 of the ray's deviation, so the tolerance is wider
        ptol = 1e-2 if mode == "norm" else 1e-3
        d_ok = torch.isclose(og[..., 0], oc[..., 0], atol=1e-4)
        p_ok = torch.isclose(og[..., 1], oc[..., 1], atol=ptol)
        assert d_ok.float().mean() > 0.995 and p_ok.float().mean() > 0.995
        assert (og[..., 1] - oc[..., 1]).abs()[d_ok].max() < ptol
        ch = og[..., 1]
        if mode == "norm":
            # a post-process of the abs sample: every fixture ray is honest
            # (the abs pass above asserts it), so each frame is standardised
            # over all of its pixels - zero mean (to the clip's bite) and a
            # std of s / (s + 50 u), under 1 - and clipped to [-3, 3]
            assert torch.all(ch >= -3.0) and torch.all(ch <= 3.0)
            assert ch.double().mean(dim=(1, 2)).abs().max() < 0.1
            assert torch.all(ch.double().std(dim=(1, 2), unbiased=False) < 1.0)
            continue
        assert torch.all(ch >= lg.potential.lo) and torch.all(ch <= lg.potential.hi)
        assert not (ch == lg.potential.bad).any(), \
            "a fixture ray sampled unreachable space one cell back"
        if mode == "logabs":
            # the same LEVEL as abs - the spawn frame ~1 - on a log axis,
            # so the deep frame abs squeezes under 0.1 opens out
            assert torch.all(ch >= 0.0) and torch.all(ch <= 1.5)
            assert 0.9 < ch[0].mean() < 1.05
            assert 0.2 < ch[5].mean() < 0.8
        elif mode == "abs":
            # the spawn frame reads ~1 (the start), the deep frames less
            assert 0.95 < ch[0].mean() < 1.01 and ch[5].mean() < 0.1
        else:
            # looking down the track from the spawn: goal-ward on average;
            # looking back up (pose 6, yaw 90 pitch 25 at the spawn): not
            assert ch[0].mean() > 0.2 and ch[6].mean() < 0.0
    # --obs-potential-curtain on the real map: the same slab test on both
    # paths, on a box HALF the fixture rays cross (the map's own bounds cut
    # at its centre in x - the finish box itself is 1 u thin and off these
    # views). A box below the map catches nothing and must be bit-identical
    # to the flag off; the map's whole bounding box catches every ray and
    # must read the goal everywhere.
    mn_m, mx_m = core.map_bounds()
    half = {"mins": [float(mn_m[0]), float(mn_m[1]), float(mn_m[2])],
            "maxs": [float(0.5 * (mn_m[0] + mx_m[0])), float(mx_m[1]),
                     float(mx_m[2])]}
    below = {"mins": [float(mn_m[0]), float(mn_m[1]), float(mn_m[2]) - 1e5],
             "maxs": [float(mx_m[0]), float(mx_m[1]), float(mn_m[2]) - 9e4]}
    allb = {"mins": [float(v) for v in mn_m], "maxs": [float(v) for v in mx_m]}
    for mode in POTENTIAL_MODES:
        lg.potential = Pg.with_mode(mode, curtain=None)
        lc.potential = Pc.with_mode(mode, curtain=None)
        base_g, base_c = lg.render(*tens("cuda")).cpu(), lc.render(*tens("cpu"))
        lg.potential = Pg.with_mode(mode, curtain=below)
        lc.potential = Pc.with_mode(mode, curtain=below)
        assert torch.equal(lg.render(*tens("cuda")).cpu(), base_g), mode
        assert torch.equal(lc.render(*tens("cpu")), base_c), mode
        lg.potential = Pg.with_mode(mode, curtain=half)
        lc.potential = Pc.with_mode(mode, curtain=half)
        hg, hc = lg.render(*tens("cuda")).cpu(), lc.render(*tens("cpu"))
        assert torch.equal(hg[..., 0], dref.cpu()), "the curtain moved depth"
        moved = (hg[..., 1] - base_g[..., 1]).abs() > 1e-6
        assert moved.any() and not moved.all(), (mode, moved.float().mean())
        ptol = 1e-2 if mode == "norm" else 1e-3
        assert torch.isclose(hg[..., 1], hc[..., 1], atol=ptol) \
            .float().mean() > 0.99, mode
        lg.potential = Pg.with_mode(mode, curtain=allb)
        ch = lg.render(*tens("cuda")).cpu()[..., 1]
        if mode in ("abs", "logabs"):
            assert torch.all(ch == 0.0), mode        # every ray at the goal
        elif mode == "rel":
            assert torch.all(ch > 0.0), mode         # every ray goal-ward
        else:
            assert torch.all(ch == 0.0), mode        # a flat frame: 0
    lg.potential, lc.potential = Pg, Pc
    # the eye sample is GoalField.sample at the eye
    o, _, _, dk = tens("cuda")
    eye = o.clone()
    eye[:, 2] += torch.where(dk.bool(), 12.0, 17.0)
    assert np.abs(Pg.eye(o, dk).cpu().numpy()
                  - gf.sample(eye.cpu().numpy().astype(np.float64))).max() < 0.1
    # why one cell back: the raw hit point is INSIDE a wall voxel on a
    # measurable share of rays and would read the sentinel there
    lc.potential = Pc
    _ = lc.render(*tens("cpu"))
    N = len(CPOSES)
    o, _, _, dk = tens("cpu")
    t = torch.clamp(lc._t, max=lc.range)
    ex, ey = o[:, 0].view(N, 1, 1), o[:, 1].view(N, 1, 1)
    ez = (o[:, 2] + torch.where(dk.bool(), 12.0, 17.0)).view(N, 1, 1)
    at_hit = torch.stack((ex + lc._dx * t, ey + lc._dy * t, ez + lc._dz * t),
                         -1).reshape(-1, 3)
    _, ok_hit = Pc.sample(at_hit)
    assert 0.05 < 1.0 - ok_hit.float().mean().item() < 0.5


# ------------------------------------------------------------ trainer smokes
def _run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout, encoding="utf-8",
                          errors="replace")


def _train(run, extra, steps="6144", ckpt=None):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE_FLAGS + \
        ["--steps", steps] + list(extra)
    if ckpt is not None:
        cmd += ["--ckpt", str(ckpt)]
    return _run(cmd)


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


@needs_run
@pytest.mark.parametrize("mode", ["rel", "abs", "norm", "logabs"])
def test_trainer_smoke_trains_records_and_resumes(mode):
    run = f"cya_pot_{mode}"
    r = _train(run, ABS + ["--obs-potential", mode])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert f"--obs-potential {mode}: surf_src_cannonball obs-potential {mode}" \
        in r.stdout
    assert "-> in_ch 2" in r.stdout
    if mode == "norm":
        assert "farther-from-goal POSITIVE, unreachable +3" in r.stdout
    assert "bake" not in r.stdout.lower() or "goal_32" in r.stdout   # no re-bake
    d = ROOT / "runs" / run
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["obs_potential"] == mode
    assert 190_000 < c["obs_potential_d0"]["cannonball"] < 210_000
    rows = _csv(run)
    assert len(rows) == 3
    for x in rows:
        for k in ("train/loss", "train/value_loss", "train/approx_kl"):
            assert np.isfinite(float(x[k])), (k, x[k])
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["obs_potential"] == mode
    sd = ck["policy"]
    assert tuple(sd["conv.0.weight"].shape) == (16, 2, 5, 5)      # in_ch 2
    # obs width: 15 core scalars + no latch + 2 channels x 16 x 8 - read off
    # the trainer's own Policy assertion by rebuilding it
    from train_fast import N_SCALAR, Policy
    Policy(N_SCALAR + 2 * 16 * 8, 16, 8, emb=64, hidden=64, in_ch=2,
           view_continuous=True, view_absolute="velocity").load_state_dict(sd)
    with pytest.raises(AssertionError):
        Policy(N_SCALAR + 16 * 8, 16, 8, emb=64, hidden=64, in_ch=2,
               view_continuous=True, view_absolute="velocity")
    # record_ckpt mirrors the channel
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1",
                "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    assert f"--obs-potential {mode} mirrored" in rec.stdout
    assert (d / "rec.jsonl").exists()
    # a resume without the flag restores it; EITHER other mode is refused
    re = _train(f"cya_pot_re_{mode}", ABS, steps="8192", ckpt=d / "ckpt_final.pt")
    assert re.returncode == 0, re.stdout[-4000:] + re.stderr[-4000:]
    assert f"obs_potential={mode}" in re.stdout
    others = [m for m in POTENTIAL_MODES if m != mode]
    assert len(others) == 3
    for other in others:
        bad = _train(f"cya_pot_bad_{mode}", ABS + ["--obs-potential", other],
                     steps="8192", ckpt=d / "ckpt_final.pt")
        assert bad.returncode != 0, other
        assert "--obs-potential changes the conv trunk's input channels" \
            in bad.stdout + bad.stderr, other
    for n in (run, f"cya_pot_re_{mode}", f"cya_pot_bad_{mode}"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


@needs_run
def test_curtain_smoke_trains_records_and_resumes():
    """--obs-potential-curtain end to end on the trainer: the finish box
    reaches the renderer, the key lands in run.json and the checkpoint, a
    resume without the flag restores it, and the flag alone is refused."""
    run = "cya_pot_curtain"
    r = _train(run, ABS + ["--obs-potential", "logabs",
                           "--obs-potential-curtain"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "CURTAIN on: a ray entering the finish box" in r.stdout
    assert "log1p(d_hit / 1000 u)" in r.stdout and "-> in_ch 2" in r.stdout
    d = ROOT / "runs" / run
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["obs_potential"] == "logabs" and c["obs_potential_curtain"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["obs_potential_curtain"] == 1
    for x in _csv(run):
        assert np.isfinite(float(x["train/loss"])), x["train/loss"]
    # record_ckpt rebuilds the box out of the map's own zones
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1",
                "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    assert "CURTAIN on: a ray entering the finish box" in rec.stdout
    # a resume without the flag restores it
    re = _train("cya_pot_curtain_re", ABS, steps="8192",
                ckpt=d / "ckpt_final.pt")
    assert re.returncode == 0, re.stdout[-4000:] + re.stderr[-4000:]
    assert "obs_potential_curtain=1" in re.stdout
    # ... and without a channel to modify it is refused
    bad = _train("cya_pot_curtain_bad", ABS + ["--obs-potential-curtain"])
    assert bad.returncode != 0
    assert "there is no channel without --obs-potential" in bad.stdout + bad.stderr
    for n in (run, "cya_pot_curtain_re", "cya_pot_curtain_bad"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


@needs_run
def test_flag_is_refused_where_it_cannot_be_right():
    for mode in ("rel", "norm"):
        r = _train("cya_pot_euclid", ABS + ["--obs-potential", mode,
                                            "--race-dist", "euclid"])
        assert r.returncode != 0 and "needs the baked geodesic field" in r.stdout + r.stderr
        r = _train("cya_pot_mask", ABS + ["--obs-potential", mode, "--surf-mask", "1"])
        assert r.returncode != 0 and "separate experiments" in r.stdout + r.stderr
    r = _train("cya_pot_mode", ABS + ["--obs-potential", "signed"])
    assert r.returncode != 0 and "invalid choice" in r.stdout + r.stderr
    for n in ("cya_pot_euclid", "cya_pot_mask", "cya_pot_mode"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
