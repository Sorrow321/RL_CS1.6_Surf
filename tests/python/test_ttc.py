"""--obs-ttc (the user, 2026-10-01, idea 2): the velocity folded into the render - the looming /
time-to-contact channel exp(-depth / ((v . ray) x 1 s)), 0 where the player does not close on the
point.

(a) CPU, test_obs_potential's synthetic scene: the channel is the formula on the march's own depth
    and rays; moving away reads 0; the depth (and the potential) channels are the lidar's without
    the flag; a render without the velocity is refused.
(b) a trainer smoke (CPU): trains with in_ch 3 beside the potential, record_ckpt mirrors it (the
    policy wrapper passes the eval core's velocity), a resume restores it, a mismatch is refused.
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
from surfgym.vision import TTC_TAU, LidarPotential                 # noqa: E402
from test_obs_potential import (ABS, CANNONBALL, CELL, RECORD, VIEWS,  # noqa: E402
                                _field, _poses, _run, _train, scene)  # noqa: F401
from test_view_continuous import needs_run                         # noqa: E402

KW = dict(cell=CELL, device="cpu", max_steps=256)


def test_the_channel_is_the_formula(scene):
    p = _poses(*VIEWS)
    n = len(VIEWS)
    base = vision.GpuLidar(None, 16, 8, **KW)
    lt = vision.GpuLidar(None, 16, 8, ttc=True, **KW)
    assert (lt.channels, lt.ttc_channel) == (2, 1)
    v = torch.tensor([[800.0, 0.0, -400.0]] * n)
    out = lt.render(*p, velocity=v)
    assert out.shape == (n, 8, 16, 2)
    assert torch.equal(out[..., 0], base.render(*p))
    t = lt.decode_depth(out[..., 0])
    c = (v[:, 0].view(n, 1, 1) * lt._dx + v[:, 1].view(n, 1, 1) * lt._dy
         + v[:, 2].view(n, 1, 1) * lt._dz)
    want = torch.where(c > 1.0, torch.exp(-t / (c.clamp(min=1.0) * TTC_TAU)), torch.zeros_like(c))
    assert torch.allclose(out[..., 1], want, atol=1e-6)
    assert float(out[..., 1].max()) <= 1.0 and float(out[..., 1].min()) >= 0.0
    # moving straight up, every downward ray reads 0 (the player does not close on the floor)
    up = lt.render(*_poses({"pitch": -90.0}), velocity=torch.tensor([[0.0, 0.0, 500.0]]))
    assert float(up[..., 1].abs().max()) == 0.0
    # falling onto it: the bottom rows of the straight-down view read close to 1
    down = lt.render(*_poses({"pitch": -90.0}), velocity=torch.tensor([[0.0, 0.0, -3000.0]]))
    assert float(down[0, 3:5, :, 1].min()) > 0.9
    with pytest.raises(ValueError):
        lt.render(*p)


def test_with_the_potential(scene):
    p = _poses(*VIEWS)
    gf = _field()
    ref = vision.GpuLidar(None, 16, 8, potential=LidarPotential(gf, "norm", device="cpu"), **KW)
    on = vision.GpuLidar(None, 16, 8, potential=LidarPotential(gf, "norm", device="cpu"),
                         ttc=True, **KW)
    assert (on.channels, on.pot_channel, on.ttc_channel) == (3, 1, 2)
    v = torch.tensor([[600.0, 200.0, 0.0]] * len(VIEWS))
    out = on.render(*p, velocity=v)
    assert torch.equal(out[..., :2], ref.render(*p))


@needs_run
def test_trainer_smoke_trains_records_and_resumes():
    run = "cya_ttc"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-ttc", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--obs-ttc: surf_src_cannonball the looming channel" in r.stdout
    assert "-> in_ch 3" in r.stdout
    d = ROOT / "runs" / run
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["obs_ttc"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 3, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    re = _train("cya_ttc_re", ABS, steps="8192", ckpt=d / "ckpt_final.pt")
    assert re.returncode == 0, re.stdout[-4000:] + re.stderr[-4000:]
    assert "obs_ttc=1" in re.stdout
    bad = _train("cya_ttc_bad", ABS + ["--obs-ttc", "0"], steps="8192", ckpt=d / "ckpt_final.pt")
    assert bad.returncode != 0 and "--obs-ttc changes the conv trunk" in bad.stdout + bad.stderr
    for n in (run, "cya_ttc_re", "cya_ttc_bad"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
