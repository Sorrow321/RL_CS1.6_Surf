"""--obs-views (the user, 2026-10-01): the camera RING - the front camera plus left (+90), right
(-90) and back (180) cameras, each the same render turned about the vertical, stacked as channels.

(a) CPU, test_obs_potential's synthetic scene: the ring is exactly four single-camera renders (the
    side cameras at 1/views_scale the resolution), norm's potential is standardised over the
    whole ring (each view weighted alike), the refusals.
(b) the policy's ring image: the front's channels, then left / right / back, the side cameras
    upsampled (nearest) into place.
(c) a trainer smoke (CPU): in_ch 20 (depth, potential, normal x3 per camera), the config keys,
    record_ckpt mirrors it, a mismatched resume is refused.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym import vision                                         # noqa: E402
from surfgym.vision import VIEW_RING_YAWS, LidarPotential          # noqa: E402
from test_obs_potential import (ABS, CANNONBALL, CELL, CPOSES, RECORD, VIEWS,  # noqa: E402
                                _field, _poses, _run, _train, needs_cuda_caches,
                                scene)  # noqa: F401
from test_view_continuous import needs_run                         # noqa: E402

KW = dict(cell=CELL, device="cpu", max_steps=256)
FACEID = CANNONBALL.parent / "surf_src_cannonball.faceid_32.npz"


def _turned(p, dyaw):
    return (p[0], p[1] + dyaw, p[2], p[3])


@pytest.mark.parametrize("scale", [1, 2])
def test_the_ring_is_four_turned_renders(scene, scale):
    """front = the plain render; side camera j = the plain render at yaw + VIEW_RING_YAWS[j + 1],
    at 1/scale the resolution - bit for bit, depth and (rel, no post-process) potential alike"""
    P = LidarPotential(_field(), "rel", device="cpu")
    one = vision.GpuLidar(None, 16, 8, potential=P, **KW)
    small = vision.GpuLidar(None, 16 // scale, 8 // scale, potential=P, **KW)
    ring = vision.GpuLidar(None, 16, 8, potential=P, views=4, views_scale=scale, **KW)
    assert (ring.channels, ring.conv_channels) == (2, 8)
    assert ring.frame_size == 16 * 8 * 2 + 3 * (16 // scale) * (8 // scale) * 2
    p = _poses(*VIEWS)
    out = ring.render(*p)
    assert out.shape == (len(VIEWS), ring.frame_size)
    front, side = ring.split_views(out)
    assert torch.equal(front, one.render(*p))
    for j, dyaw in enumerate(VIEW_RING_YAWS[1:]):
        assert torch.equal(side[:, j], small.render(*_turned(p, dyaw))), (j, dyaw)
    # the back camera of a pose facing +x is the front camera of the same pose facing -x
    assert torch.equal(side[0, 2], small.render(*_turned(_poses(VIEWS[0]), 180.0))[0])


@pytest.mark.parametrize("scale", [1, 2])
def test_norm_is_standardised_over_the_whole_ring(scene, scale):
    P = LidarPotential(_field(), "norm", device="cpu")
    ring = vision.GpuLidar(None, 16, 8, potential=P, views=4, views_scale=scale, **KW)
    p = _poses(*VIEWS)
    raw_f, raw_s = ring.split_views(ring.render(*p, post=False))
    front, side = ring.split_views(ring.render(*p))
    # depth untouched by the post-process
    assert torch.equal(front[..., 0], raw_f[..., 0]) and torch.equal(side[..., 0], raw_s[..., 0])
    n = len(VIEWS)
    sp = raw_s[..., 1].repeat_interleave(scale, 2).repeat_interleave(scale, 3)
    joint = torch.cat((raw_f[..., 1].reshape(n, -1), sp.reshape(n, -1)), dim=1)
    want = P.normalise(joint[:, None, :])[:, 0, :]
    assert torch.equal(front[..., 1].reshape(n, -1), want[:, :16 * 8])
    assert torch.equal(side[..., 1], want[:, 16 * 8:].reshape(n, 3, 8, 16)[:, :, ::scale, ::scale])
    # ... which is NOT the front camera standardised on its own: the ring keeps which way is
    # goal-ward (facing +x, the goal side, the front reads below the ring's mean)
    alone = P.normalise(raw_f[..., 1])
    assert not torch.equal(front[..., 1], alone)
    ok = raw_f[0, ..., 1] >= 0.0
    assert float(front[0, ..., 1][ok].mean()) < 0.0


def test_the_refusals(scene):
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, views=4, ttc=True, **KW)
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, views=3, **KW)
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, views=4, views_scale=3, **KW)
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 15, 8, views=4, views_scale=2, **KW)
    one = vision.GpuLidar(None, 16, 8, **KW)
    assert (one.views, one.frame_size, one.conv_channels) == (1, 16 * 8, 1)


@pytest.mark.parametrize("scale", [1, 2])
def test_the_policy_stacks_the_ring_as_channels(scale):
    from train_fast import N_SCALAR, Policy
    W, H, c = 16, 8, 5
    frame = W * H * c + 3 * (W // scale) * (H // scale) * c
    pol = Policy(N_SCALAR + frame, W, H, emb=32, hidden=32, in_ch=4 * c, views=4,
                 views_scale=scale)
    # each camera's pixels carry (camera index, row, column, channel) so the stack can be read
    h, w = H // scale, W // scale
    fr = torch.zeros(1, H, W, c)
    sd = torch.zeros(1, 3, h, w, c)
    for r in range(H):
        for q in range(W):
            for k in range(c):
                fr[0, r, q, k] = 1000 * r + 10 * q + k
    for j in range(3):
        for r in range(h):
            for q in range(w):
                for k in range(c):
                    sd[0, j, r, q, k] = 100000 * (j + 1) + 1000 * r + 10 * q + k
    img = torch.cat((fr.reshape(1, -1), sd.reshape(1, -1)), dim=1)
    im = pol._ring_image(img)
    assert im.shape == (1, H, W, 4 * c)
    assert torch.equal(im[..., :c], fr)
    for j in range(3):
        up = sd[:, j].repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        assert torch.equal(im[..., c * (j + 1):c * (j + 2)], up), j
    # and the whole forward runs on it
    scal = torch.zeros(2, N_SCALAR)
    f = pol.features(scal, img.repeat(2, 1))
    assert f.shape[0] == 2


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
def test_ring_trainer_smoke():
    run = "cya_views"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-normal", "1", "--obs-views", "4",
                           "--obs-views-scale", "2"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "in_ch 20" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert (cfg["obs_views"], cfg["obs_views_scale"]) == (4, 2)
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 20, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    bad = _train("cya_views_bad", ABS + ["--obs-views", "1"], ckpt=d / "ckpt_final.pt")
    assert bad.returncode != 0 and "--obs-views changes the image row" in bad.stdout + bad.stderr
    for n in (run, "cya_views_bad"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


@needs_cuda_caches
@pytest.mark.parametrize("scale", [1, 2])
def test_the_ring_on_cannonball_cuda(scale):
    """the triton path: the ring of --obs-normal renders == four single-camera renders, bit for
    bit (the side cameras through their own shallow copy, at 1/scale the resolution)"""
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    kw = dict(range_units=11500.0, near_range=2000.0, cell=32.0, device="cuda",
              texture_mode="normal")
    one = vision.GpuLidar(core, 64, 32, texture=True, **kw)
    small = vision.GpuLidar(core, 64 // scale, 32 // scale, texture=one.texmap, **kw)
    ring = vision.GpuLidar(core, 64, 32, texture=one.texmap, views=4, views_scale=scale, **kw)
    a = np.asarray(CPOSES, np.float32)
    p = (torch.as_tensor(a[:, 0:3]).cuda(), torch.as_tensor(a[:, 3]).cuda(),
         torch.as_tensor(a[:, 4]).cuda(), torch.as_tensor(a[:, 5]).cuda())
    front, side = ring.split_views(ring.render(*p))
    assert torch.equal(front, one.render(*p))
    for j, dyaw in enumerate(VIEW_RING_YAWS[1:]):
        assert torch.equal(side[:, j], small.render(*_turned(p, dyaw))), (j, dyaw)
