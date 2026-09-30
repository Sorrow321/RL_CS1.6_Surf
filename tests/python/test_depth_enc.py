"""--depth-enc (the user, 2026-09-30): the depth channel's far field made readable.

The legacy encoding is linear to 1.0 at near (2,000 u) and then squashes everything to the range
into a 0.25 tail, so two surfaces 1,000 u apart differ by 0.025 at 5-6k u and 0.0075 at 8-9k u
(0.5 at 500-1,500 u). log = ln(1 + t/d0) / ln(1 + near/d0) keeps 1.0 at near and far differences
in proportion; dual = both, two depth channels ahead of the potential.

(a) CPU, the synthetic scene of test_obs_potential: legacy is the default and its render is the
    lidar's as before; log is the formula on the march's own t, 1.0 at near, and decode_depth
    inverts it; dual is (legacy, log) with channels 2 (3 with the potential, which sits LAST, at
    pot_channel); the potential channel is the same with every depth encoding; the refusals.
(b) the contrast the flag exists for, on the encodings themselves: 1,000 u apart at 5-6k u and
    8-9k u, log against legacy.
(c) CUDA, cannonball's caches: the new kernels agree with the torch reference in log and dual,
    plain and with the potential, and dual's legacy channel is the shipped kernel's, bit for bit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym import vision                                     # noqa: E402
from surfgym.vision import LidarPotential                      # noqa: E402
from test_obs_potential import (CELL, CPOSES, CANNONBALL, VIEWS,  # noqa: E402
                                _field, _poses, needs_cuda_caches, scene)  # noqa: F401

LEG = dict(cell=CELL, device="cpu", max_steps=256)


def _legacy_formula(t, near):
    return (torch.clamp(t, max=near) / near
            + 0.25 * (1.0 - torch.exp(-torch.clamp(t - near, min=0.0) / 2500.0)))


def test_legacy_is_the_default_and_log_is_the_formula(scene):
    p = _poses(*VIEWS)
    base = vision.GpuLidar(None, 16, 8, **LEG)
    leg = vision.GpuLidar(None, 16, 8, depth_enc="legacy", **LEG)
    assert base.depth_enc == "legacy" and base.channels == 1 and base.pot_channel == 1
    d = base.render(*p)
    assert torch.equal(leg.render(*p), d)
    t = torch.clamp(base._t, max=base.range)
    assert torch.equal(d, _legacy_formula(t, base.near))
    lg = vision.GpuLidar(None, 16, 8, depth_enc="log", depth_log_d0=50.0, **LEG)
    assert lg.channels == 1
    e = lg.render(*p)
    want = torch.log(1.0 + t / 50.0) / float(np.log1p(lg.near / 50.0))
    assert torch.allclose(e, want, atol=1e-6)
    # 1.0 at near, like legacy; the decode is the inverse
    assert abs(float(torch.log(torch.tensor(1.0 + lg.near / 50.0)) * lg.lscale) - 1.0) < 1e-6
    assert torch.allclose(lg.decode_depth(e), t, atol=1e-2)
    assert lg.enc_max == pytest.approx(np.log1p(lg.range / 50.0) / np.log1p(lg.near / 50.0))


def test_dual_is_both_and_the_potential_sits_last(scene):
    p = _poses(*VIEWS)
    leg = vision.GpuLidar(None, 16, 8, **LEG)
    lg = vision.GpuLidar(None, 16, 8, depth_enc="log", **LEG)
    du = vision.GpuLidar(None, 16, 8, depth_enc="dual", **LEG)
    assert du.channels == 2 and du.depth_ch == 2
    o = du.render(*p)
    assert o.shape == (len(VIEWS), 8, 16, 2)
    assert torch.equal(o[..., 0], leg.render(*p))
    assert torch.allclose(o[..., 1], lg.render(*p), atol=1e-7)
    # decode reads channel 0, the legacy one, under dual
    assert torch.allclose(du.decode_depth(o[..., 0]), leg.decode_depth(o[..., 0]))
    gf = _field()
    ref = vision.GpuLidar(None, 16, 8, potential=LidarPotential(gf, "rel", device="cpu"), **LEG)
    r = ref.render(*p)
    for enc, ch in (("log", 2), ("dual", 3)):
        on = vision.GpuLidar(None, 16, 8, depth_enc=enc,
                             potential=LidarPotential(gf, "rel", device="cpu"), **LEG)
        assert on.channels == ch and on.pot_channel == ch - 1
        out = on.render(*p)
        assert out.shape == (len(VIEWS), 8, 16, ch)
        # the same rays, the same potential: only the depth encoding moved
        assert torch.equal(out[..., -1], r[..., 1])
        assert torch.allclose(out[..., ch - 2], lg.render(*p), atol=1e-7)
        if enc == "dual":
            assert torch.equal(out[..., 0], r[..., 0])
    # norm's post-process is applied to the POTENTIAL channel wherever it sits
    nrm = vision.GpuLidar(None, 16, 8, depth_enc="dual",
                          potential=LidarPotential(gf, "norm", device="cpu"), **LEG)
    nref = vision.GpuLidar(None, 16, 8, potential=LidarPotential(gf, "norm", device="cpu"), **LEG)
    assert torch.equal(nrm.render(*p)[..., 2], nref.render(*p)[..., 1])


def test_refusals(scene):
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, depth_enc="inverse", **LEG)
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, depth_enc="log", depth_log_d0=0.0, **LEG)
    for kw in ({"pinhole": True}, {"normals": True}, {"surf_mask": True}):
        with pytest.raises(ValueError):
            vision.GpuLidar(None, 16, 8, depth_enc="log", **LEG, **kw)


def test_the_far_field_contrast_the_flag_exists_for():
    near, d0 = 2000.0, 200.0

    def leg(t):
        return min(t, near) / near + 0.25 * (1.0 - np.exp(-max(t - near, 0.0) / 2500.0))

    def lg(t):
        return np.log1p(t / d0) / np.log1p(near / d0)
    for a, b, gain_lo in ((5000.0, 6000.0, 2.5), (8000.0, 9000.0, 5.0), (10000.0, 11000.0, 9.0)):
        g = (lg(b) - lg(a)) / (leg(b) - leg(a))
        assert g > gain_lo, (a, b, g)
    # and near the eye log keeps most of legacy's contrast (74% at 500 vs 1,500 u)
    assert 0.7 < (lg(1500.0) - lg(500.0)) / (leg(1500.0) - leg(500.0)) < 0.8


@needs_cuda_caches
def test_triton_enc_kernels_agree_with_the_reference_on_cannonball():
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import build_goal_field
    from surfgym.zones import load_zones
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    gf = build_goal_field(core, load_zones(str(CANNONBALL))["end"], cell=32.0)
    a = np.asarray(CPOSES, np.float32)
    kw = dict(range_units=11500.0, near_range=2000.0, cell=32.0)

    def tens(dev):
        return (torch.as_tensor(a[:, 0:3]).to(dev), torch.as_tensor(a[:, 3]).to(dev),
                torch.as_tensor(a[:, 4]).to(dev), torch.as_tensor(a[:, 5]).to(dev))
    shipped = vision.GpuLidar(core, 64, 32, device="cuda", **kw).render(*tens("cuda")).cpu()
    for pot in (None, "norm"):
        for enc in ("log", "dual"):
            mk = (lambda dev: dict(potential=LidarPotential(gf, pot, d0=None, device=dev))
                  if pot else {})
            lg_ = vision.GpuLidar(core, 64, 32, device="cuda", depth_enc=enc, **kw, **mk("cuda"))
            lc_ = vision.GpuLidar(core, 64, 32, device="cpu", depth_enc=enc, **kw, **mk("cpu"))
            og = lg_.render(*tens("cuda")).cpu()
            oc = lc_.render(*tens("cpu"))
            assert og.shape == oc.shape and og.shape[-1:] != (0,)
            ch = lg_.channels
            assert (og.shape[-1] if og.dim() == 4 else 1) == ch
            ok = torch.isclose(og, oc, atol=2e-3 if pot == "norm" else 1e-4)
            assert ok.float().mean() > 0.995, (pot, enc, ok.float().mean())
            if enc == "dual":
                # channel 0 is the shipped kernel's depth, bit for bit
                assert torch.equal(og[..., 0], shipped)
            logc = og[..., ch - 2] if og.dim() == 4 else og
            # a clear ray reads the log encoding's ceiling, none above it
            assert float(logc.max()) <= lg_.enc_max + 1e-5 or enc == "dual"


# ------------------------------------------------------------ trainer smokes
from test_obs_potential import ABS, RECORD, _run, _train  # noqa: E402
from test_view_continuous import needs_run  # noqa: E402


@needs_run
@pytest.mark.parametrize("enc", ["log", "dual"])
def test_trainer_smoke_trains_records_and_resumes(enc):
    """--depth-enc end to end (with the scratch default --obs-potential norm): the run trains with
    finite losses, run.json and the checkpoint carry depth_enc / depth_log_d0, conv1 takes 2 + (1
    under dual) channels, record_ckpt mirrors the encoding, a resume without the flag restores it,
    the other encoding is refused, and a legacy run writes neither key."""
    run = f"cya_denc_{enc}"
    r = _train(run, ABS + ["--obs-potential", "norm", "--depth-enc", enc])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert f"--depth-enc {enc}: depth = " in r.stdout
    ch = 3 if enc == "dual" else 2
    assert f"-> in_ch {ch}" in r.stdout
    d = ROOT / "runs" / run
    import json
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["depth_enc"] == enc and c["depth_log_d0"] == 200.0
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["depth_enc"] == enc
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, ch, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    assert (d / "rec.jsonl").exists()
    re = _train(f"cya_denc_re_{enc}", ABS, steps="8192", ckpt=d / "ckpt_final.pt")
    assert re.returncode == 0, re.stdout[-4000:] + re.stderr[-4000:]
    assert f"depth_enc={enc}" in re.stdout
    other = "log" if enc == "dual" else "dual"
    bad = _train(f"cya_denc_bad_{enc}", ABS + ["--depth-enc", other], steps="8192",
                 ckpt=d / "ckpt_final.pt")
    assert bad.returncode != 0 and "--depth-enc changes the depth pixels" in bad.stdout + bad.stderr
    leg = _train(f"cya_denc_leg_{enc}", ABS + ["--obs-potential", "norm"])
    assert leg.returncode == 0, leg.stdout[-4000:] + leg.stderr[-4000:]
    lc = json.loads((ROOT / "runs" / f"cya_denc_leg_{enc}" / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert "depth_enc" not in lc and "depth_log_d0" not in lc
    import shutil
    for n in (run, f"cya_denc_re_{enc}", f"cya_denc_bad_{enc}", f"cya_denc_leg_{enc}"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
