"""--obs-potential (docs/obs_potential.md): the race potential as the depth
image's second channel, in two encodings.

(a) LidarPotential.sample is GoalField.sample: the trilinear-over-honest-
    corners geodesic, the honesty mask, the sentinel, on a synthetic field
    (CPU; the codes round-trip the cached grid exactly).
(b) the two encodings on hand values: abs = d_hit / d0 in [0, 1.5] with a
    bad sample at 1.5 and the eye ignored; rel = (d_eye - d_hit) / 2000 in
    [-2, 2], goal-ward POSITIVE, a bad sample or a bad eye at -2; with_mode
    shares the grid; the refusals (a field without a grid, a mode that is
    not abs/rel, abs without d0).
(c) the renderer on a synthetic scene (CPU torch path): (N, H, W, 2), the
    depth channel bit-identical to the depth-only lidar, the potential
    sampled ONE FIELD CELL SHORT of the hit (recomputed from the march's
    own t), goal-ward rays positive under rel and backward rays negative,
    an unreachable surface at the bad value, channels == 2, and the
    exclusivity with --surf-mask / --normals / --pinhole.
(d) CUDA, cannonball's caches: the triton tail agrees with the torch
    fallback pixel for pixel on both channels, the depth channel is
    bit-exact against the single-channel kernel, the eye sample is
    GoalField.sample at the eye, and one cell back never lands in a wall
    on the fixture poses (16% of raw hit points do).
(e) trainer smokes (CPU, the toy scratch set of test_view_continuous):
    the flag ON trains with finite losses at obs width 15 + 2*16*8, writes
    obs_potential / obs_potential_d0 into run.json and the checkpoint,
    in_ch 2 in conv.0; record_ckpt.py mirrors the channel; a resume
    without the flag restores the mode; a resume asking for the OTHER
    mode is refused; --race-dist euclid and --surf-mask alongside it are
    refused. The flag OFF is byte-identical to the pre-flag trainer by
    construction (no key is written when it is off) and pinned by
    test_unstuck.py's flag-off identity against the git-history trainer.
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
from surfgym.vision import LidarPotential                      # noqa: E402
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
    assert "goal-ward POSITIVE" in rel.describe() and "d0 900" in ab.describe()


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


def test_exclusive_with_the_other_vision_experiments(scene):
    P = LidarPotential(_field(), "rel", device="cpu")
    for kw in ({"surf_mask": True}, {"normals": True}, {"pinhole": True}):
        with pytest.raises(ValueError):
            vision.GpuLidar(None, 8, 4, cell=CELL, device="cpu", potential=P,
                            **kw)
    with pytest.raises(ValueError):
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
    for mode in ("abs", "rel"):
        lg.potential, lc.potential = Pg.with_mode(mode), Pc.with_mode(mode)
        og = lg.render(*tens("cuda")).cpu()
        oc = lc.render(*tens("cpu"))
        assert og.shape == oc.shape == (len(CPOSES), 32, 64, 2)
        # depth: the single-channel kernel's pixels, bit for bit
        assert torch.equal(og[..., 0], dref.cpu()), "depth ABI drift"
        # the two paths: cos/sin differ in the last ulp between them, which
        # can move a voxel boundary by one ray
        d_ok = torch.isclose(og[..., 0], oc[..., 0], atol=1e-4)
        p_ok = torch.isclose(og[..., 1], oc[..., 1], atol=1e-3)
        assert d_ok.float().mean() > 0.995 and p_ok.float().mean() > 0.995
        assert (og[..., 1] - oc[..., 1]).abs()[d_ok].max() < 1e-3
        ch = og[..., 1]
        assert torch.all(ch >= lg.potential.lo) and torch.all(ch <= lg.potential.hi)
        assert not (ch == lg.potential.bad).any(), \
            "a fixture ray sampled unreachable space one cell back"
        if mode == "abs":
            # the spawn frame reads ~1 (the start), the deep frames less
            assert 0.95 < ch[0].mean() < 1.01 and ch[5].mean() < 0.1
        else:
            # looking down the track from the spawn: goal-ward on average;
            # looking back up (pose 6, yaw 90 pitch 25 at the spawn): not
            assert ch[0].mean() > 0.2 and ch[6].mean() < 0.0
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
@pytest.mark.parametrize("mode", ["rel", "abs"])
def test_trainer_smoke_trains_records_and_resumes(mode):
    run = f"cya_pot_{mode}"
    r = _train(run, ABS + ["--obs-potential", mode])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert f"--obs-potential {mode}: surf_src_cannonball obs-potential {mode}" \
        in r.stdout
    assert "-> in_ch 2" in r.stdout
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
    # a resume without the flag restores it; the other mode is refused
    re = _train(f"cya_pot_re_{mode}", ABS, steps="8192", ckpt=d / "ckpt_final.pt")
    assert re.returncode == 0, re.stdout[-4000:] + re.stderr[-4000:]
    assert f"obs_potential={mode}" in re.stdout
    other = "abs" if mode == "rel" else "rel"
    bad = _train(f"cya_pot_bad_{mode}", ABS + ["--obs-potential", other],
                 steps="8192", ckpt=d / "ckpt_final.pt")
    assert bad.returncode != 0
    assert "--obs-potential changes the conv trunk's input channels" \
        in bad.stdout + bad.stderr
    for n in (run, f"cya_pot_re_{mode}", f"cya_pot_bad_{mode}"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


@needs_run
def test_flag_is_refused_where_it_cannot_be_right():
    r = _train("cya_pot_euclid", ABS + ["--obs-potential", "rel",
                                        "--race-dist", "euclid"])
    assert r.returncode != 0 and "needs the baked geodesic field" in r.stdout + r.stderr
    r = _train("cya_pot_mask", ABS + ["--obs-potential", "rel", "--surf-mask", "1"])
    assert r.returncode != 0 and "separate experiments" in r.stdout + r.stderr
    r = _train("cya_pot_mode", ABS + ["--obs-potential", "signed"])
    assert r.returncode != 0 and "invalid choice" in r.stdout + r.stderr
    for n in ("cya_pot_euclid", "cya_pot_mask", "cya_pot_mode"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
