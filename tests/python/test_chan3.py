"""chan3: depth + surfability mask + goal potential as THREE image channels.

Until now --surf-mask and --obs-potential were mutually exclusive for two
separate reasons, and only ONE of them was real:

  1. a METHODOLOGY guard - one vision experiment at a time, so a result can
     be attributed (train_fast.check_vision_exclusive). The user overrode it
     for THIS pair on 2026-09-11; every other pair still refuses.
  2. an IMPLEMENTATION gap - both quantities were already on the march's
     path, but there was no third output plane and the potential's
     post-process wrote out[..., 1], the SAME slot the mask uses.

What this file pins:

(a) surfgym.vision.channel_layout: the whole truth table, and the
    combinations it still refuses (normals+mask, normals+potential,
    mask_only+potential, mask_only without surf_mask).
(b) THE BIT-IDENTITY EVIDENCE. On one synthetic scene, four renders -
    depth only, mask only, potential only, both - and the 3-channel image's
    planes are torch.equal to the corresponding single-flag renders, plane
    for plane, in all four potential modes. So with only --surf-mask the
    output is byte-identical to today, with only --obs-potential likewise,
    and neither one-extra-plane layout moved an index.
(c) norm is a PER-FRAME statistic and must be computed over the POTENTIAL
    plane: the mask plane's values are left alone, and re-baking the mask
    grid to something wild does not move the potential plane by one ulp.
(d) the flat image stays channel-fastest at 3 (train_fast's forward_split
    restrides it into a channels_last conv input).
(e) triton and the torch fallback agree on a mixed scene, both channels,
    all four modes - the same guarantee test_obs_potential.py holds for the
    2-channel kernel, on the NZ=True specialisation.
(f) the trainer end to end: in_ch 3 in conv.0, "img_channels": 3 in
    run.json and the checkpoint config, record_ckpt.py rebuilds the same
    3-channel image, a resume without the flags restores both, and a resume
    that would CHANGE the channel count is refused in both directions.
(g) check_vision_exclusive still refuses every other pair.
"""
from __future__ import annotations

import json
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

from surfgym import surfmask, vision                            # noqa: E402
from surfgym.vision import (CH_DEPTH, CH_MASK, CH_POT_ALONE,    # noqa: E402
                            CH_POT_WITH_MASK, POTENTIAL_MODES,
                            LidarPotential, channel_layout)
from test_obs_potential import (CELL, _field, _poses,           # noqa: E402
                                _scene_sdf, VIEWS)
from test_view_continuous import (CANNONBALL, SMOKE_FLAGS,      # noqa: E402
                                  _env, needs_run)

TRAIN = ROOT / "python" / "train_fast.py"
RECORD = ROOT / "tools" / "record_ckpt.py"
ABS = ["--view-continuous", "--view-absolute", "velocity"]
D0 = 900.0


# ------------------------------------------------------------ (a) the table
def test_channel_layout_is_the_whole_truth_table():
    assert channel_layout() == (1, CH_DEPTH, None, None)
    assert channel_layout(surf_mask=True) == (2, CH_DEPTH, CH_MASK, None)
    assert channel_layout(potential=True) == (2, CH_DEPTH, None, CH_POT_ALONE)
    assert channel_layout(surf_mask=True, potential=True) \
        == (3, CH_DEPTH, CH_MASK, CH_POT_WITH_MASK)
    assert channel_layout(normals=True) == (4, CH_DEPTH, None, None)
    # --surf-mask 2: the mask ALONE, at channel 0, and no depth
    assert channel_layout(surf_mask=True, mask_only=True) == (1, None, 0, None)
    # the single-flag indices are the ones every existing checkpoint saw
    assert (CH_DEPTH, CH_MASK, CH_POT_ALONE) == (0, 1, 1)
    assert CH_POT_WITH_MASK == 2


def test_channel_layout_refuses_what_has_no_kernel():
    with pytest.raises(ValueError, match="same channel twice"):
        channel_layout(surf_mask=True, normals=True)
    with pytest.raises(ValueError, match="no combined kernel"):
        channel_layout(normals=True, potential=True)
    with pytest.raises(ValueError, match="no depth channel"):
        channel_layout(surf_mask=True, mask_only=True, potential=True)
    with pytest.raises(ValueError, match="mask_only requires surf_mask"):
        channel_layout(mask_only=True)


# ------------------------------------------------- the synthetic scene + grids
# test_obs_potential's scene (a floor slab under a field whose goal is at
# +x) with a surfability grid whose values VARY across the floor, so a
# render that read the wrong plane could not pass by looking constant.
def _snz():
    snz = np.zeros((16, 16, 16), dtype=np.int8)
    x = np.arange(16, dtype=np.int16)
    snz[0] = (17 + 7 * x[None, :]).astype(np.int8)      # 17..122 across +x
    snz[1] = 127                                        # a different layer
    return snz


@pytest.fixture
def grids(monkeypatch):
    sdf = _scene_sdf()
    mins = np.zeros(3, dtype=np.float32)
    snz = _snz()
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None: (sdf, mins, cell))
    monkeypatch.setattr(surfmask, "build_surfnz",
                        lambda core, cell, cache_dir=None, mesh_path=None:
                        (snz, mins))
    return sdf, snz


def _lidars(device="cpu", **potkw):
    kw = dict(cell=CELL, device=device, max_steps=256)
    P = LidarPotential(_field(), "abs", d0=D0, device=device, **potkw)
    return (vision.GpuLidar(None, 16, 8, **kw),                      # depth
            vision.GpuLidar(None, 16, 8, surf_mask=True, **kw),      # + mask
            vision.GpuLidar(None, 16, 8, potential=P, **kw),         # + pot
            vision.GpuLidar(None, 16, 8, surf_mask=True,             # all three
                            potential=P, **kw),
            P)


# --------------------------------------------- (b) THE BIT-IDENTITY EVIDENCE
@pytest.mark.parametrize("mode", POTENTIAL_MODES)
def test_three_planes_are_exactly_the_single_flag_renders(grids, mode):
    """The claim the whole change rests on: adding the third channel moved
    NOTHING. Plane 0 is the depth-only render, plane 1 the --surf-mask
    render's mask, plane 2 the --obs-potential render's potential - all
    torch.equal, not allclose."""
    depth, mask, pot, both, P = _lidars()
    for lid in (pot, both):
        lid.potential = P.with_mode(mode, d0=D0)
    assert (depth.channels, mask.channels, pot.channels, both.channels) \
        == (1, 2, 2, 3)
    assert (both.ch_depth, both.ch_mask, both.ch_potential) == (0, 1, 2)
    assert (mask.ch_mask, pot.ch_potential) == (1, 1), \
        "a single-flag layout moved an index and would break every ckpt"

    p = _poses(*VIEWS)
    d, m, q, b = (lid.render(*p) for lid in (depth, mask, pot, both))
    assert d.shape == (len(VIEWS), 8, 16)
    assert m.shape == q.shape == (len(VIEWS), 8, 16, 2)
    assert b.shape == (len(VIEWS), 8, 16, 3)
    assert torch.equal(b[..., 0], d), "depth moved"
    assert torch.equal(b[..., 1], m[..., 1]), "the mask plane moved"
    assert torch.equal(b[..., 2], q[..., 1]), "the potential plane moved"
    # ... and the single-flag renders still agree with each other on depth
    assert torch.equal(m[..., 0], d) and torch.equal(q[..., 0], d)
    # the scene is not degenerate: both extra planes carry real structure
    assert b[..., 1].min() < b[..., 1].max()
    assert b[..., 2].min() < b[..., 2].max()


def test_the_mask_plane_is_the_surfability_bake_at_the_hit(grids):
    """Plane 1 of the 3-channel image is |n_z| / 127 out of the bake, the
    same gather --surf-mask alone does - checked here against the grid
    itself, not just against the other render."""
    _, snz = grids
    _, mask, _, both, _ = _lidars()
    p = _poses(*VIEWS)
    b = both.render(*p)
    nz = b[..., 1]
    # every value in the frame is some voxel's int8 / 127
    allowed = torch.as_tensor(np.unique(snz).astype(np.float32) / 127.0)
    assert torch.isin(nz, allowed).all()
    assert torch.equal(nz, mask.render(*p)[..., 1])


# -------------------------------------------------------- (c) norm is per-plane
def test_norm_normalises_the_potential_plane_and_ignores_the_mask(grids,
                                                                  monkeypatch):
    """norm is `(d - frame mean) / (frame std + 50 u)` over the frame's
    honest pixels. Written into slot 1 it would standardise |n_z| and
    silently hand the policy a normalised MASK as its potential."""
    _, mask, pot, both, P = _lidars()
    for lid in (pot, both):
        lid.potential = P.with_mode("norm")
    p = _poses(*VIEWS)
    m, q, b = mask.render(*p), pot.render(*p), both.render(*p)
    # the potential plane is standardised (zero-mean, clipped) ...
    ch = b[..., 2]
    assert torch.equal(ch, q[..., 1])
    assert torch.all(ch >= -3.0) and torch.all(ch <= 3.0)
    # ... and the mask plane is NOT: it is still |n_z| in [0, 1]
    nz = b[..., 1]
    assert torch.equal(nz, m[..., 1])
    assert torch.all(nz >= 0.0) and torch.all(nz <= 1.0)
    assert nz.mean().item() > 0.05, "the mask plane was standardised"

    # the decisive one: change the mask GRID and the potential plane must
    # not move by one ulp (a per-frame statistic over the wrong plane would)
    wild = _snz()
    wild[0] = 127
    monkeypatch.setattr(surfmask, "build_surfnz",
                        lambda core, cell, cache_dir=None, mesh_path=None:
                        (wild, np.zeros(3, dtype=np.float32)))
    other = vision.GpuLidar(None, 16, 8, cell=CELL, device="cpu",
                            max_steps=256, surf_mask=True,
                            potential=P.with_mode("norm"))
    o = other.render(*p)
    assert not torch.equal(o[..., 1], b[..., 1]), "the fixture did not bite"
    assert torch.equal(o[..., 2], b[..., 2]), \
        "the potential plane followed the MASK's values"
    assert torch.equal(o[..., 0], b[..., 0])


needs_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and vision.HAVE_TRITON),
    reason="needs CUDA + triton for the march kernel")


# --------------------------------------- the curtain composes with the mask
# jtCP / jtCPM / jtCP3 all run `--obs-potential norm --obs-potential-curtain`,
# so the ONE configuration that actually trains is norm + curtain + mask.
CURTAIN_BOX = {"mins": [0.0, 0.0, 0.0], "maxs": [128.0, 128.0, 16.0]}


@pytest.mark.parametrize("mode", POTENTIAL_MODES)
def test_the_finish_curtain_composes_with_the_mask(grids, mode):
    """--obs-potential-curtain is a constexpr slab test in the same kernel
    as NZ. It must bite on the potential plane and leave the other two
    alone, on both march paths."""
    kw = dict(cell=CELL, device="cpu", max_steps=256)
    P = LidarPotential(_field(), mode, d0=D0, device="cpu",
                       curtain=CURTAIN_BOX)
    P0 = LidarPotential(_field(), mode, d0=D0, device="cpu")
    three = vision.GpuLidar(None, 16, 8, surf_mask=True, potential=P, **kw)
    two = vision.GpuLidar(None, 16, 8, potential=P, **kw)
    mask = vision.GpuLidar(None, 16, 8, surf_mask=True, **kw)
    nocur = vision.GpuLidar(None, 16, 8, surf_mask=True, potential=P0, **kw)
    p = _poses(*VIEWS)
    a, b, c, d = (lid.render(*p) for lid in (three, two, mask, nocur))
    assert torch.equal(a[..., 0], c[..., 0])
    assert torch.equal(a[..., 1], c[..., 1]), "the curtain moved the mask"
    assert torch.equal(a[..., 2], b[..., 1])
    assert not torch.equal(a[..., 2], d[..., 2]), "the curtain did not bite"


@needs_triton
def test_the_curtain_and_the_mask_agree_across_march_paths(grids):
    kw = dict(cell=CELL, max_steps=256, surf_mask=True)
    Pg = LidarPotential(_field(), "norm", d0=D0, device="cuda",
                        curtain=CURTAIN_BOX)
    Pc = LidarPotential(_field(), "norm", d0=D0, device="cpu",
                        curtain=CURTAIN_BOX)
    lg = vision.GpuLidar(None, 16, 8, device="cuda", potential=Pg, **kw)
    lc = vision.GpuLidar(None, 16, 8, device="cpu", potential=Pc, **kw)
    p = _poses(*VIEWS)
    og = lg.render(*tuple(t.cuda() for t in p)).cpu()
    oc = lc.render(*p)
    d_ok = torch.isclose(og[..., 0], oc[..., 0], atol=1e-4)
    assert d_ok.float().mean() > 0.99
    assert torch.isclose(og[..., 1], oc[..., 1], atol=1e-6)[d_ok].all()
    assert torch.isclose(og[..., 2], oc[..., 2], atol=1e-2)[d_ok].all()


# ------------------------------------------------------ (d) memory layout
def test_flat_image_is_channel_fastest_at_three(grids):
    """train_fast.Policy.forward_split restrides the flat obs row into a
    channels_last conv input; that is a free restride only if the channel
    is the fastest axis."""
    _, _, _, both, _ = _lidars()
    out = both.render(*_poses(*VIEWS))
    flat = out.reshape(len(VIEWS), -1)
    assert flat.shape[1] == 8 * 16 * 3
    assert torch.equal(flat.reshape(-1, 8, 16, 3), out)
    im = flat.reshape(-1, 8, 16, 3).permute(0, 3, 1, 2)
    assert im.stride()[1] == 1, "channel stride must be 1 (NHWC)"


# ------------------------------------------ (e) triton == the torch fallback
@needs_triton
@pytest.mark.parametrize("mode", POTENTIAL_MODES)
def test_triton_and_fallback_agree_on_a_mixed_scene(grids, mode):
    """The NZ=True specialisation of _march_kernel_pot against the torch
    fallback, on a scene whose frames mix hits, misses, honest and
    unreachable field samples, and a varying mask."""
    Pg = LidarPotential(_field(), mode, d0=D0, device="cuda")
    Pc = LidarPotential(_field(), mode, d0=D0, device="cpu")
    kw = dict(cell=CELL, max_steps=256, surf_mask=True)
    lg = vision.GpuLidar(None, 16, 8, device="cuda", potential=Pg, **kw)
    lc = vision.GpuLidar(None, 16, 8, device="cpu", potential=Pc, **kw)
    # the 2-channel kernel on the same rays: the depth plane must not have
    # moved when NZ switched on, and the mask plane must be the nz kernel's
    plain = vision.GpuLidar(None, 16, 8, device="cuda", cell=CELL,
                            max_steps=256)
    nzonly = vision.GpuLidar(None, 16, 8, device="cuda", cell=CELL,
                            max_steps=256, surf_mask=True)
    assert (lg.channels, lc.channels) == (3, 3)

    p = _poses(*VIEWS)
    pg = tuple(t.cuda() for t in p)
    og = lg.render(*pg).cpu()
    oc = lc.render(*p)
    assert og.shape == oc.shape == (len(VIEWS), 8, 16, 3)
    # depth and mask come out of THIS kernel and must be bit-exact against
    # the kernels that already shipped, on the same card
    dref = plain.render(*pg).cpu()
    mref = nzonly.render(*pg).cpu()
    assert torch.equal(og[..., 0], dref), "depth ABI drift under NZ"
    assert torch.equal(og[..., 1], mref[..., 1]), "the mask plane drifted"
    # the two march paths differ by an ulp of cos/sin, which can move a
    # voxel boundary by one ray (test_obs_potential's tolerance, verbatim)
    ptol = 1e-2 if mode == "norm" else 1e-3
    d_ok = torch.isclose(og[..., 0], oc[..., 0], atol=1e-4)
    assert d_ok.float().mean() > 0.99
    assert torch.isclose(og[..., 1], oc[..., 1],
                         atol=1e-6)[d_ok].all(), "mask disagreement"
    assert torch.isclose(og[..., 2], oc[..., 2],
                         atol=ptol)[d_ok].all(), "potential disagreement"


# ------------------------------------------------------------- (g) the guard
def test_the_guard_still_refuses_every_other_pair():
    from train_fast import check_vision_exclusive as chk
    # the relaxation, and only it
    chk(1, 0, 0, 0, "abs")
    chk(1, 0, 1, 0, "abs")           # --frame-stack 1 is "off"
    chk(1, 0, 0, 0, None)
    chk(0, 0, 0, 0, "abs")
    chk(0, 0, 0, 1, None)
    # --surf-mask 2 is the mask ALONE: no depth channel to ride next to
    with pytest.raises(SystemExit, match="no depth channel"):
        chk(2, 0, 0, 0, "abs")
    # every other pair still has no combined path
    for args in ((1, 1, 0, 0, None),          # mask + pinhole
                 (1, 0, 4, 0, None),          # mask + frame-stack
                 (1, 0, 0, 1, None),          # mask + normals
                 (0, 1, 4, 0, None),          # pinhole + frame-stack
                 (0, 1, 0, 1, None),          # pinhole + normals
                 (0, 0, 4, 1, None),          # frame-stack + normals
                 (0, 1, 0, 0, "abs"),         # pinhole + potential
                 (0, 0, 4, 0, "abs"),         # frame-stack + potential
                 (0, 0, 0, 1, "abs"),         # normals + potential
                 (1, 0, 4, 0, "abs"),         # the trio
                 (1, 1, 0, 0, "abs")):
        with pytest.raises(SystemExit, match="separate experiments"):
            chk(*args)


# ------------------------------------------------------------ (f) the trainer
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


CHAN3 = ["--surf-mask", "1", "--obs-potential", "abs"]


@needs_run
def test_trainer_smoke_three_channels_and_run_json_round_trip():
    run = "cya_chan3"
    r = _train(run, ABS + CHAN3)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "-> in_ch 3" in r.stdout
    d = ROOT / "runs" / run
    # run.json round-trip: both flags plus the RESOLVED width
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["surf_mask"] == 1 and c["obs_potential"] == "abs"
    assert c["img_channels"] == 3
    assert vision.channel_layout(surf_mask=bool(c["surf_mask"]),
                                 potential=bool(c["obs_potential"]))[0] == 3
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["config"]["img_channels"] == 3
    assert ck["config"]["surf_mask"] == 1
    sd = ck["policy"]
    assert tuple(sd["conv.0.weight"].shape) == (16, 3, 5, 5)      # in_ch 3
    from train_fast import N_SCALAR, Policy
    Policy(N_SCALAR + 3 * 16 * 8, 16, 8, emb=64, hidden=64, in_ch=3,
           view_continuous=True, view_absolute="velocity").load_state_dict(sd)
    for x in _csv(run):
        for k in ("train/loss", "train/value_loss", "train/approx_kl"):
            assert np.isfinite(float(x[k])), (k, x[k])
    # record_ckpt rebuilds the same 3-channel image (and cross-checks the
    # recorded width against the layout it derived)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1",
                "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    assert "chan3: image is (depth, |n_z|, potential)" in rec.stdout
    assert (d / "rec.jsonl").exists()
    # a resume with NEITHER flag restores both and rebuilds the 3-ch image
    re = _train("cya_chan3_re", ABS, steps="8192", ckpt=d / "ckpt_final.pt")
    assert re.returncode == 0, re.stdout[-4000:] + re.stderr[-4000:]
    assert "surf_mask=1" in re.stdout and "obs_potential=abs" in re.stdout
    assert "-> in_ch 3" in re.stdout
    for n in (run, "cya_chan3_re"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


@needs_run
def test_a_channel_count_change_cannot_warm_start_a_checkpoint():
    """conv[0] is (16, in_ch, 5, 5). Both directions, both flags: dropping
    one flag from a 3-channel ckpt narrows it, adding one to a 2-channel
    ckpt widens it, and neither is a warm start."""
    three = "cya_chan3_src"
    r = _train(three, ABS + CHAN3)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    src3 = ROOT / "runs" / three / "ckpt_final.pt"
    two = "cya_chan3_two"
    r = _train(two, ABS + ["--obs-potential", "abs"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    src2 = ROOT / "runs" / two / "ckpt_final.pt"

    # 3 -> 2: explicitly ask for no mask
    bad = _train("cya_chan3_bad", ABS + ["--surf-mask", "0",
                                         "--obs-potential", "abs"],
                 steps="8192", ckpt=src3)
    assert bad.returncode != 0
    assert "--surf-mask changes the conv trunk's input channels" \
        in bad.stdout + bad.stderr
    # 3 -> 2 the other way: keep the mask, drop the potential
    bad = _train("cya_chan3_bad", ABS + ["--surf-mask", "1"],
                 steps="8192", ckpt=src3)
    # (--obs-potential absent is RESTORED, so this one must SUCCEED at 3 ch)
    assert bad.returncode == 0, bad.stdout[-4000:] + bad.stderr[-4000:]
    assert "-> in_ch 3" in bad.stdout
    # 2 -> 3: add the mask to a potential-only checkpoint
    bad = _train("cya_chan3_bad2", ABS + CHAN3, steps="8192", ckpt=src2)
    assert bad.returncode != 0
    assert "--surf-mask changes the conv trunk's input channels" \
        in bad.stdout + bad.stderr
    for n in (three, two, "cya_chan3_bad", "cya_chan3_bad2"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
