"""--obs-surf (user, 2026-10-19): one observation column, 1 while the player
rides a surfable face (surfgym/surfflag.py), and the --surf-bonus clamp fix
that came with it.

(a) SurfFlag: free fall, a near-vertical wall, a ceiling, walkable ground and
    a take-off read 0; a ramp's push reads 1; a fall pinned at the
    -sv_maxvelocity clamp reads 0; the first decision of an episode and a
    reset row read 0; the eval path finds episode starts off the tick counter.
(b) RaceReward --surf-bonus: a fall pinned at the clamp no longer pays (the
    per-tick test read the clamp's missing gravity step as support), a ramp
    still does, and a core without sv_maxvelocity keeps the old test.
(c) trainer smoke (CPU, cannonball toy set): the flag widens the scalar side
    by one column and dumps obs_surf 1; a flagless resume restores it, a
    resume with --obs-surf 0 is refused, a plain checkpoint resumed with the
    flag is zero-widened onto it; record_ckpt records an obs_surf checkpoint.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.surfflag import SurfFlag, g_step_of, make_surf_feed  # noqa: E402
from surfgym.rewards import RaceReward                             # noqa: E402
from test_unstuck import ABS, TRAIN, _csv, _run, _train            # noqa: E402
from test_view_continuous import SMOKE_FLAGS, needs_run            # noqa: E402

K = 4
G_STEP = g_step_of(800.0, 10.0, K)       # 32 u/s per decision
VMAX = 4000.0


# ==========================================================================
# (a) the tracker
# ==========================================================================
def test_the_band_over_one_decision():
    f = SurfFlag(7, G_STEP, K, vmax=VMAX)
    og0 = np.array([-1, -1, -1, -1, 3, 3, -1])
    vz0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3990.0])
    assert not f.features(og0, vz0).any(), "the first decision has nothing to compare with"
    #               free fall  wall      ramp   ceiling  ground  take-off  clamp
    vz1 = np.array([-G_STEP,   -G_STEP,  40.0,  -400.0,  0.0,    250.0,    -VMAX])
    og1 = np.array([-1,        -1,       -1,    -1,      3,      -1,       -1])
    got = f.features(og1, vz1)
    assert got.dtype == np.float32
    assert got.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # the ramp keeps paying while the push goes on, and stops in free fall
    vz2 = vz1.copy()
    vz2[2] = 60.0
    assert f.features(og1, vz2)[2] == 1.0
    vz3 = vz2.copy()
    vz3[2] = 60.0 - G_STEP
    assert f.features(og1, vz3)[2] == 0.0


def test_reset_and_out_buffer():
    f = SurfFlag(3, G_STEP, K)
    og = np.full(3, -1)
    f.features(og, np.zeros(3))
    out = np.zeros(3, np.float32)
    f.features(og, np.full(3, 50.0), out=out)
    assert out.tolist() == [1.0, 1.0, 1.0]
    f.reset(np.array([False, True, False]))
    assert f.flag.tolist() == [1.0, 0.0, 1.0]
    got = f.features(og, np.full(3, 100.0))
    assert got.tolist() == [1.0, 0.0, 1.0], "a reset row's first decision reads 0"
    f.reset(np.array([0, 2]))                     # an index array works too
    assert f.features(og, np.full(3, 150.0)).tolist() == [0.0, 1.0, 0.0]


def test_eval_path_detects_episode_starts_off_the_tick_counter():
    f = SurfFlag(2, G_STEP, K)
    og = np.full(2, -1)
    f.eval_features(og, np.zeros(2), np.array([0, 0]))
    assert f.eval_features(og, np.full(2, 50.0), np.array([4, 4])).tolist() == [1.0, 1.0]
    # env 1 respawned in between: its counter advanced by less than K
    got = f.eval_features(og, np.full(2, 100.0), np.array([8, 2]))
    assert got.tolist() == [1.0, 0.0]


class _Core:
    def __init__(self, n):
        self.states_view = np.zeros(n, dtype=[("velocity", np.float32, 3),
                                              ("onground", np.int32),
                                              ("tick", np.int32)])


def test_make_surf_feed_is_the_tracker():
    core = _Core(2)
    core.states_view["onground"] = -1
    feed = make_surf_feed(800.0, 10.0, K, vmax=VMAX)
    assert feed(core, 2).tolist() == [0.0, 0.0]
    core.states_view["velocity"][:, 2] = [50.0, -G_STEP]
    core.states_view["tick"] = K
    assert feed(core, 2).tolist() == [1.0, 0.0]


# ==========================================================================
# (b) --surf-bonus: the clamp
# ==========================================================================
class _Phys:
    sv_gravity = 800.0
    msec = 10
    sv_maxvelocity = VMAX


class _Cfg:
    phys = _Phys()


class _FakeCore:
    def __init__(self, n, phys=None):
        self.num_envs = n
        self.config = _Cfg()
        if phys is not None:
            self.config = type("C", (), {"phys": phys})()
        self.tick_ms = 10.0
        self.states_view = np.zeros(n, dtype=[
            ("origin", np.float32, 3), ("velocity", np.float32, 3),
            ("onground", np.int32), ("yaw", np.float32)])
        self.goal_hits = np.zeros(n, np.uint8)


class _FlatField:
    def sample(self, pos):
        return np.full(len(pos), 10_000.0, np.float64)


def _step(rf, core, vz):
    core.states_view["velocity"][:, 2] = vz
    core.states_view["onground"][:] = -1
    n = core.num_envs
    z = np.zeros((n, 15), np.float32)
    return rf(z, z, z, np.zeros(n, np.float32), np.zeros(n, np.uint8),
              np.zeros(n, np.uint8), core)


def test_surf_bonus_no_longer_pays_a_fall_pinned_at_the_clamp():
    core = _FakeCore(2)
    rf = RaceReward(_FlatField(), scale=0.001, time_pen=0.005, int_coef=0.0, dip=False)
    rf.surf_bonus = 0.0025
    _step(rf, core, np.array([-VMAX, 0.0]))        # on_reset + anchor
    # env 0 stays pinned at the clamp (no gravity step), env 1 is caught by a ramp
    r = _step(rf, core, np.array([-VMAX, -2.0]))
    paid = (r + 0.005) > 1e-9
    assert not paid[0], "a fall pinned at -sv_maxvelocity is free fall"
    assert paid[1], "a ramp catch still pays"


class _PhysNoVmax:
    sv_gravity = 800.0
    msec = 10


def test_a_core_without_sv_maxvelocity_keeps_the_old_test():
    core = _FakeCore(1, phys=_PhysNoVmax())
    rf = RaceReward(_FlatField(), scale=0.001, time_pen=0.005, int_coef=0.0, dip=False)
    rf.surf_bonus = 0.0025
    _step(rf, core, np.array([-VMAX]))
    r = _step(rf, core, np.array([-VMAX]))
    assert (r + 0.005)[0] > 1e-9, "no clamp known: the per-tick test as it was"


# ==========================================================================
# (c) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_obs_surf_smoke_widens_dumps_resumes_refuses_and_records():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    import torch
    run = "cya_obssurf"
    r = _train(run, ABS + ["--obs-surf", "1"], steps="4096")
    assert "--obs-surf: obs column 15 = 1 while riding a surfable face" in r.stdout
    d = ROOT / "runs" / run
    assert _cfg(run)["obs_surf"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    # a plain control of the same smoke, for the tower widths
    run0 = "cya_obssurf_off"
    _train(run0, ABS, steps="2048")
    d0 = ROOT / "runs" / run0
    ck0 = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert "obs_surf" not in _cfg(run0)
    w1, w0 = ck["policy"]["pi.0.weight"], ck0["policy"]["pi.0.weight"]
    assert w1.shape[1] == w0.shape[1] + 1, "one more scalar-side column"
    # a flagless resume restores the flag
    run2 = "cya_obssurf_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "6144"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "obs_surf=1" in r2.stdout
    assert _cfg(run2)["obs_surf"] == 1
    # dropping the trained column is refused
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_obssurf_drop", "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "6144",
                                                         "--obs-surf", "0"])
    assert r3.returncode != 0
    assert "trained with --obs-surf 1" in r3.stdout + r3.stderr
    # a plain checkpoint resumed WITH the flag is zero-widened onto it
    run4 = "cya_obssurf_grow"
    shutil.rmtree(ROOT / "runs" / run4, ignore_errors=True)
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", run4, "--ckpt",
               str(d0 / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "4096",
                                                          "--obs-surf", "1"])
    assert r4.returncode == 0, r4.stdout[-4000:] + r4.stderr[-4000:]
    assert "--obs-surf 1: this checkpoint's towers read" in r4.stdout
    # the recorder mirrors it (the record gate's own backend)
    out = d / "rec_obssurf.jsonl"
    r5 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(d / "ckpt_final.pt"), "--out", str(out), "--episodes", "1",
               "--ep-ticks", "400"])
    assert r5.returncode == 0, r5.stdout[-4000:] + r5.stderr[-4000:]
    assert "--obs-surf: the surfing flag is obs column" in r5.stdout
    assert out.exists() and out.stat().st_size > 0
    for dd in (d, d0, ROOT / "runs" / run2, ROOT / "runs" / "cya_obssurf_drop",
               ROOT / "runs" / run4):
        shutil.rmtree(dd, ignore_errors=True)
