"""--obs-pot-rate (user, 2026-10-19): one observation column, the rate the
shaping potential changes along the player's velocity, -grad d . v / 1000
(surfgym/obsaux.PotRate), on the compass's own gradient.

(a) PotRate on baked fields whose answer is known in closed form: a linear
    field in an arbitrary direction gives exactly a . v / 1000; a slope of 0.5
    halves it (the gradient's magnitude, not just its direction); a slope of 2
    is clipped to 1 (the compass's own |grad| clip); a point off the reachable
    set reads 0; a two-map fleet reads each env against ITS map's field; the
    eval feed is the same class off a core's states.
(b) trainer smoke (CPU, cannonball toy set): with --obs-surf and --obs-compass
    the rate is the LAST scalar column and the row widens by exactly the seven
    columns of the three flags; obs_pot_rate 1 is dumped; a flagless resume
    restores it, a resume with --obs-pot-rate 0 is refused, a plain checkpoint
    resumed with the flag is zero-widened onto it; record_ckpt records it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.goalfield import GoalField                          # noqa: E402
from surfgym.obsaux import PotRate, make_pot_rate_feed           # noqa: E402
from test_unstuck import ABS, TRAIN, _run, _train                # noqa: E402
from test_view_continuous import SMOKE_FLAGS, needs_run          # noqa: E402

CELL = 32.0
MINS = np.array([0.0, 0.0, 0.0])
NX, NY, NZ = 40, 30, 20


def _baked(fn, reach_max=1e6):
    """A GoalField whose voxel centres hold fn(x, y, z)."""
    cx, cy, cz = (MINS[i] + (np.arange(n) + 0.5) * CELL for i, n in enumerate((NX, NY, NZ)))
    z, y, x = np.meshgrid(cz, cy, cx, indexing="ij")
    return GoalField(fn(x, y, z).astype(np.float32), MINS, CELL, reach_max)


def _inside(n, rng):
    # well inside the bake, so every probe one cell away is still on the grid
    lo = MINS + 4 * CELL
    hi = MINS + np.array([NX - 4, NY - 4, NZ - 4]) * CELL
    return lo + rng.random((n, 3)) * (hi - lo)


# ==========================================================================
# (a) the column
# ==========================================================================
def test_a_linear_field_gives_the_speed_along_its_slope():
    a = np.array([0.6, -0.48, 0.64])
    a /= np.linalg.norm(a)
    gf = _baked(lambda x, y, z: 5000.0 - (a[0] * x + a[1] * y + a[2] * z))
    rng = np.random.default_rng(0)
    n = 64
    pos = _inside(n, rng)
    vel = rng.normal(0.0, 1500.0, (n, 3))
    got = PotRate(n, gf).features(pos, vel)
    assert got.dtype == np.float32 and got.shape == (n,)
    np.testing.assert_allclose(got, vel @ a / 1000.0, rtol=1e-4, atol=1e-3)
    # downhill is +a: flying along it is positive, against it negative
    up = PotRate(2, gf).features(pos[:2], np.stack([a * 2000.0, -a * 2000.0]))
    np.testing.assert_allclose(up, [2.0, -2.0], atol=1e-4)


def test_the_magnitude_counts_and_is_clipped_at_one():
    rng = np.random.default_rng(1)
    pos = _inside(16, rng)
    vel = np.zeros((16, 3))
    vel[:, 0] = 1800.0
    half = PotRate(16, _baked(lambda x, y, z: 5000.0 - 0.5 * x)).features(pos, vel)
    np.testing.assert_allclose(half, 0.9, atol=1e-4)
    steep = PotRate(16, _baked(lambda x, y, z: 5000.0 - 2.0 * x)).features(pos, vel)
    np.testing.assert_allclose(steep, 1.8, atol=1e-4)
    # across the slope: 0
    vel[:, 0], vel[:, 1] = 0.0, 1800.0
    np.testing.assert_allclose(
        PotRate(16, _baked(lambda x, y, z: 5000.0 - x)).features(pos, vel), 0.0, atol=1e-4)


def test_off_the_reachable_set_reads_zero():
    reach_max = 6000.0

    def fn(x, y, z):
        d = 5000.0 - x
        return np.where(x > 20 * CELL, reach_max + 10 * CELL, d)   # the far half: sentinel
    gf = _baked(fn, reach_max=reach_max)
    pos = np.array([[10.5 * CELL, 15 * CELL, 10 * CELL],     # reachable, two cells from the edge
                    [30.5 * CELL, 15 * CELL, 10 * CELL]])    # deep in the sentinel
    vel = np.array([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    got = PotRate(2, gf).features(pos, vel)
    assert abs(got[0] - 1.0) < 1e-4 and got[1] == 0.0


def test_each_env_reads_its_own_maps_field():
    f_a = _baked(lambda x, y, z: 5000.0 - x)       # downhill +x
    f_b = _baked(lambda x, y, z: 5000.0 + x)       # downhill -x
    pr = PotRate(4, [(slice(0, 2), f_a), (slice(2, 4), f_b)])
    pos = _inside(4, np.random.default_rng(2))
    vel = np.tile([1000.0, 0.0, 0.0], (4, 1))
    np.testing.assert_allclose(pr.features(pos, vel), [1.0, 1.0, -1.0, -1.0], atol=1e-4)


def test_the_eval_feed_is_the_same_class_off_the_core_states():
    gf = _baked(lambda x, y, z: 5000.0 - y)
    rng = np.random.default_rng(3)
    pos, vel = _inside(3, rng), rng.normal(0.0, 900.0, (3, 3))

    class _Core:
        states_view = {"origin": pos.astype(np.float32), "velocity": vel.astype(np.float32)}
    feed = make_pot_rate_feed(gf)
    got = feed(_Core(), 3)
    want = PotRate(3, gf).features(pos.astype(np.float32), vel.astype(np.float32))
    assert got.shape == (3,) and np.array_equal(got, want)
    np.testing.assert_allclose(got, vel[:, 1] / 1000.0, rtol=1e-3, atol=1e-3)


# ==========================================================================
# (b) the trainer and the recorder
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_obs_pot_rate_smoke_widens_dumps_resumes_refuses_and_records():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    import torch
    run = "cya_potrate"
    r = _train(run, ABS + ["--obs-surf", "1", "--obs-compass", "1", "--obs-pot-rate", "1"],
               steps="4096")
    # 15 core | 5 compass (15..19) | surf 20 | rate 21
    assert "--obs-surf: obs column 20 = 1" in r.stdout
    assert "--obs-pot-rate: obs column 21 = -grad d . v / 1000" in r.stdout
    d = ROOT / "runs" / run
    assert _cfg(run)["obs_pot_rate"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    run0 = "cya_potrate_off"
    _train(run0, ABS, steps="2048")
    d0 = ROOT / "runs" / run0
    ck0 = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert "obs_pot_rate" not in _cfg(run0)
    w1, w0 = ck["policy"]["pi.0.weight"], ck0["policy"]["pi.0.weight"]
    assert w1.shape[1] == w0.shape[1] + 7, "compass 5 + surf 1 + rate 1"
    # a flagless resume restores the flag
    run2 = "cya_potrate_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "6144"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "obs_pot_rate=1" in r2.stdout
    assert _cfg(run2)["obs_pot_rate"] == 1
    # dropping the trained column is refused
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "cya_potrate_drop", "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "6144",
                                                         "--obs-pot-rate", "0"])
    assert r3.returncode != 0
    assert "trained with --obs-pot-rate 1" in r3.stdout + r3.stderr
    # a plain checkpoint resumed WITH the flag is zero-widened onto it
    run4 = "cya_potrate_grow"
    shutil.rmtree(ROOT / "runs" / run4, ignore_errors=True)
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", run4, "--ckpt",
               str(d0 / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "4096",
                                                          "--obs-pot-rate", "1"])
    assert r4.returncode == 0, r4.stdout[-4000:] + r4.stderr[-4000:]
    assert "--obs-pot-rate 1: this checkpoint's towers read" in r4.stdout
    # the recorder mirrors it (the record gate's own backend)
    out = d / "rec_potrate.jsonl"
    r5 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(d / "ckpt_final.pt"), "--out", str(out), "--episodes", "1",
               "--ep-ticks", "400"])
    assert r5.returncode == 0, r5.stdout[-4000:] + r5.stderr[-4000:]
    assert "--obs-pot-rate: the potential's rate along the velocity is obs column" in r5.stdout
    assert out.exists() and out.stat().st_size > 0
    for dd in (d, d0, ROOT / "runs" / run2, ROOT / "runs" / "cya_potrate_drop",
               ROOT / "runs" / run4):
        shutil.rmtree(dd, ignore_errors=True)
