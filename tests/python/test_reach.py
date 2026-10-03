"""--obs-reach (the user, 2026-10-03): the free-flight reach channel - can the player's current
momentum carry it to each visible surface point? tanh(slack / 1 s), slack = (when gravity takes
the feet down to the point's height) - (how long the speed and the air control need to get above
it); -1 above the apex and on rays that hit nothing.

(a) CPU, test_obs_potential's synthetic scene (an infinite floor at z = 0): the channel is the
    formula on the march's own depth and rays; the physics reads right - at rest only the floor
    under the feet is reachable, speed reaches ahead and not behind, upward speed only ever
    helps, a floor ABOVE the feet is out of reach until the jump's apex clears it; the depth is
    the lidar's without the flag; a render without the velocity is refused.
(b) a trainer smoke (CPU): in_ch 6 beside --obs-potential + --obs-normal, the config key,
    record_ckpt mirrors it (the policy wrapper passes the eval core's velocity), a mismatched
    resume is refused.
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
from surfgym.vision import AIR_CAP, REACH_TAU                      # noqa: E402
from test_obs_potential import (ABS, CANNONBALL, CELL, RECORD, VIEWS,  # noqa: E402
                                _poses, _run, _train, scene)       # noqa: F401
from test_view_continuous import needs_run                         # noqa: E402

KW = dict(cell=CELL, device="cpu", max_steps=256)
FACEID = CANNONBALL.parent / "surf_src_cannonball.faceid_32.npz"


def _vel(*v):
    return torch.tensor([list(v)], dtype=torch.float32)


def test_the_channel_is_the_formula(scene):
    p = _poses(*VIEWS)
    n = len(VIEWS)
    base = vision.GpuLidar(None, 16, 8, **KW)
    lr = vision.GpuLidar(None, 16, 8, reach=True, **KW)
    assert (lr.channels, lr.reach_channel, lr.reach_g, lr.reach_tick_s) == (2, 1, 800.0, 0.01)
    v = torch.tensor([[900.0, -300.0, 250.0]] * n)
    out = lr.render(*p, velocity=v)
    assert out.shape == (n, 8, 16, 2)
    assert torch.equal(out[..., 0], base.render(*p))
    # the formula, term for term, on the march's own depth and rays
    t = lr.decode_depth(out[..., 0])
    o, dk = p[0].reshape(n, 3, 1, 1), p[3].reshape(n, 1, 1).bool()
    pz = o[:, 2] + torch.where(dk, 12.0, 17.0) + t * lr._dz
    feet = o[:, 2] - torch.where(dk, 18.0, 36.0)
    vv = v.reshape(n, 3, 1, 1)
    disc = vv[:, 2] ** 2 + 2 * 800.0 * (feet - pz)
    tf = (vv[:, 2] + torch.sqrt(disc.clamp(min=0))) / 800.0
    rx, ry = t * lr._dx, t * lr._dy
    d = torch.sqrt(rx ** 2 + ry ** 2)
    vh = torch.sqrt(vv[:, 0] ** 2 + vv[:, 1] ** 2)
    th = torch.acos(((vv[:, 0] * rx + vv[:, 1] * ry) / (vh * d).clamp(min=1e-6)).clamp(-1, 1))
    need = th * vh / (AIR_CAP / 0.01) + d / vh
    want = torch.where((t < lr.range * (1 - 1e-6)) & (disc >= 0) & (tf >= 0),
                       torch.tanh((tf - need) / REACH_TAU), torch.full_like(tf, -1.0))
    assert torch.allclose(out[..., 1], want, atol=1e-5)
    assert float(out[..., 1].min()) >= -1.0 and float(out[..., 1].max()) <= 1.0
    with pytest.raises(ValueError):
        lr.render(*p)


def test_the_physics_reads_right(scene):
    lr = vision.GpuLidar(None, 16, 8, reach=True, **KW)
    down = _poses({"pitch": -90.0})                  # feet 11 u above the floor, looking down
    # at rest the air control only drifts 30 u/s, and the feet are 0.17 s above the floor: the
    # reach falls off away from the point under the feet (the centre pixels look ~7 u out, the
    # corners ~100 u) and nothing far is in reach
    r0 = lr.render(*down, velocity=_vel(0, 0, 0))[0, ..., 1]
    assert float(r0[3:5, 7:9].min()) > float(r0[0, 0]) and float(r0[0, 0]) < -0.5
    assert float(r0[3:5, 7:9].min()) > -0.2
    # 1,000 u/s along +x (the top of a straight-down yaw-0 image): the floor straight ahead
    # (~53 u, 0.05 s away) is reachable, the floor straight behind (a 180 deg turn at 1,000 u/s
    # takes ~1 s) is not
    rf = lr.render(*down, velocity=_vel(1000, 0, 0))[0, ..., 1]
    assert float(rf[0, 7:9].min()) > 0.0 > float(rf[-1, 7:9].max())
    assert float(rf[-1, 7:9].max()) < -0.5
    # upward speed only ever helps (the feet stay up longer); downward only hurts
    p = _poses(*VIEWS)
    n = len(VIEWS)
    base_v = torch.tensor([[600.0, 200.0, 0.0]] * n)
    up = base_v + torch.tensor([0.0, 0.0, 800.0])
    dn = base_v + torch.tensor([0.0, 0.0, -800.0])
    r_mid = lr.render(*p, velocity=base_v)[..., 1]
    assert torch.all(lr.render(*p, velocity=up)[..., 1] >= r_mid - 1e-6)
    assert torch.all(lr.render(*p, velocity=dn)[..., 1] <= r_mid + 1e-6)
    # a floor ABOVE the feet (origin 20 u up: feet at -16): out of reach at rest, in reach once
    # the vertical speed clears 16 u (v_z > sqrt(2 g 16) = 160 u/s)
    low = _poses({"z": 20.0, "pitch": -90.0})
    assert torch.all(lr.render(*low, velocity=_vel(0, 0, 0))[..., 1] == -1.0)
    assert float(lr.render(*low, velocity=_vel(0, 0, 400.0))[0, 3:5, 7:9, 1].min()) > 0.0


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
def test_reach_trainer_smoke():
    run = "cya_reach"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-normal", "1", "--obs-reach", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--obs-reach" in r.stdout and "-> in_ch 6" in r.stdout
    d = ROOT / "runs" / run
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["obs_reach"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 6, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    bad = _train("cya_reach_bad", ABS + ["--obs-reach", "0"], ckpt=d / "ckpt_final.pt")
    assert bad.returncode != 0 and "--obs-reach changes" in bad.stdout + bad.stderr
    for n in (run, "cya_reach_bad"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
