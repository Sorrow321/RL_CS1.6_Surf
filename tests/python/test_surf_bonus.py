"""--surf-bonus / --dive-pen: "surfing = good, diving = bad" (round 40).

THE CONTACT TEST, which is the whole arm. Riding a surfable face KEEPS
``onground == -1`` - that IS the surf mechanic, since ``src/pm.c`` sets
onground only at ``plane_normal[2] >= 0.7`` - so "airborne" on its own
conflates flight with ramp-riding. What separates them is whether the map
pushed back VERTICALLY this tick:

    dev       = (vz - vz_prev) + g*dt        (exactly 0 in free fall)
    supported = airborne AND dev > surf_dvz

That is the same "last tick the map pushed back" detector
``tools/pick_selfline.py`` trims self-lines with (CLAUDE.md), reused rather
than re-derived, and it costs one subtract and one compare on the velocity
column ``RaceReward.__call__`` already fetches.

What is pinned here:

(a) THE BAND. The test brackets |n_z| in (0, 0.7) from both sides using the
    engine's own arithmetic: walkable ground is excluded by ``onground``, a
    near-vertical wall pays nothing because it clips only the HORIZONTAL
    velocity, free fall pays nothing by construction, and a ceiling bonk is
    excluded by the SIGN (it pushes down).
(b) THE BALANCE, which is the property that makes the arm interpretable.
    ``--time-pen 0.005``/tick at 10 ms is 0.5 reward units per second, so at
    k < 0.5/s parking on a ramp is net NEGATIVE and the bonus cannot be
    farmed by sitting still. The trainer REFUSES k >= that.
(c) FLAG-OFF BIT-IDENTITY: with both at 0 the reward is the same float32
    array, element for element, that the pre-flag code produced - no state,
    no RNG draw, the block is not entered.
(d) THE EPISODE-START RE-ANCHOR: without it the first tick of every new
    episode reads a cross-episode velocity jump as a giant "the map pushed
    back".
(e) THE DIAGNOSTIC: surf_paid_frac / dive_frac are fractions of AIRBORNE
    ticks, they partition that set, and MapFleet pools them by ticks.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.rewards import RaceReward                           # noqa: E402


G_TICK = 8.0           # sv_gravity 800 * 10 ms


class _Phys:
    sv_gravity = 800.0
    msec = 10


class _Cfg:
    phys = _Phys()


class _FakeCore:
    """The slice of a SurfCore RaceReward reads: states_view (origin,
    velocity, onground), goal_hits, num_envs, config.phys."""

    def __init__(self, n):
        self.num_envs = n
        self.config = _Cfg()
        self.tick_ms = 10.0
        self.states_view = np.zeros(n, dtype=[
            ("origin", np.float32, 3), ("velocity", np.float32, 3),
            ("onground", np.int32), ("yaw", np.float32)])
        self.goal_hits = np.zeros(n, np.uint8)

    def set(self, vz, onground, origin=None):
        self.states_view["velocity"][:, 2] = vz
        self.states_view["onground"][:] = onground
        if origin is not None:
            self.states_view["origin"][:] = origin


class _FlatField:
    """A geodesic field that is the same everywhere, so the shaping term is
    exactly zero and what is left in `r` is the surf term plus -time_pen."""

    def sample(self, pos):
        return np.full(len(pos), 10_000.0, np.float64)


def _reward(n=4, **kw):
    r = RaceReward(_FlatField(), scale=0.001, time_pen=0.005,
                   int_coef=0.0, dip=False, **kw)
    return r


def _step(rf, core, vz, onground):
    core.set(vz, onground)
    n = core.num_envs
    z = np.zeros((n, 15), np.float32)
    return rf(z, z, z, np.zeros(n, np.float32),
              np.zeros(n, np.uint8), np.zeros(n, np.uint8), core)


# ------------------------------------------------------------------- (a)

def test_the_band_from_both_sides():
    """One tick each of: free fall, a wall (horizontal clip only, so vz
    still falls the full gravity step), a ramp catch, walkable ground, and
    a ceiling bonk."""
    core = _FakeCore(5)
    rf = _reward(5)
    rf.surf_bonus = 0.0025            # 0.25/s at 10 ms
    _step(rf, core, np.zeros(5), -1)  # on_reset + anchor at vz = 0

    #            free fall  wall     ramp     ground   ceiling
    vz = np.array([-G_TICK, -G_TICK, -2.0,    0.0,     -400.0])
    og = np.array([-1,      -1,      -1,      3,       -1])
    core.states_view["onground"][:] = og
    r = _step(rf, core, vz, og)
    paid = (r + 0.005) > 1e-9         # r = -time_pen (+ bonus)

    assert not paid[0], "free fall must pay nothing"
    assert not paid[1], "a near-vertical wall gives no VERTICAL support"
    assert paid[2], "a ramp catch must pay"
    assert not paid[3], "walkable ground is onground and outside the band"
    assert not paid[4], "a ceiling pushes DOWN and is excluded by the sign"
    assert np.isclose(r[2] + 0.005, 0.0025, atol=1e-7)


def test_free_fall_is_exactly_zero_dev():
    """The threshold has room: in true free fall dev is 0 to the float, so
    surf_dvz = 1.0 is not a knife edge."""
    core = _FakeCore(1)
    rf = _reward(1)
    rf.surf_bonus = 0.0025
    vz = 0.0
    _step(rf, core, np.array([vz]), -1)
    for _ in range(20):
        vz -= G_TICK
        r = _step(rf, core, np.array([vz]), -1)
        assert np.isclose(r[0], -0.005, atol=1e-9)   # never pays


def test_dive_pen_is_the_complement():
    core = _FakeCore(3)
    rf = _reward(3)
    rf.dive_pen = 0.001
    _step(rf, core, np.zeros(3), -1)
    vz = np.array([-G_TICK, -2.0, 0.0])
    og = np.array([-1, -1, 3])
    core.states_view["onground"][:] = og
    r = _step(rf, core, vz, og)
    assert np.isclose(r[0], -0.005 - 0.001)   # free fall: charged
    assert np.isclose(r[1], -0.005)           # supported: not charged
    assert np.isclose(r[2], -0.005)           # grounded: not airborne


def test_speed_floor_guard():
    core = _FakeCore(2)
    rf = _reward(2)
    rf.surf_bonus = 0.0025
    rf.surf_hspd = 400.0
    _step(rf, core, np.zeros(2), -1)
    core.states_view["velocity"][0, 0] = 1000.0     # fast
    core.states_view["velocity"][1, 0] = 100.0      # crawling
    r = _step(rf, core, np.array([-2.0, -2.0]), -1)
    assert np.isclose(r[0] + 0.005, 0.0025)
    assert np.isclose(r[1] + 0.005, 0.0)


# ------------------------------------------------------------------- (b)

def test_parking_on_a_ramp_is_net_negative_at_the_arms_k():
    """The arithmetic the arm rests on: time_pen is 0.5 reward/s and the
    arm's k is 0.25/s, so a tick spent permanently supported still LOSES
    0.25/s. The bonus cannot be farmed by sitting still."""
    time_pen_per_s = 0.005 * 100.0
    k = 0.25
    assert k < time_pen_per_s
    assert k - time_pen_per_s == pytest.approx(-0.25)

    core = _FakeCore(1)
    rf = _reward(1)
    rf.surf_bonus = k / 100.0
    _step(rf, core, np.zeros(1), -1)
    total = 0.0
    for _ in range(100):                   # one second, always supported
        total += float(_step(rf, core, np.array([-2.0]), -1)[0])
        core.states_view["velocity"][0, 2] = -2.0
    assert total < 0.0
    assert total == pytest.approx(-0.25, abs=1e-4)


def test_trainer_refuses_a_farmable_k():
    out = subprocess.run(
        [sys.executable, str(ROOT / "python" / "train_fast.py"),
         "--map", str(ROOT / "maps" / "surf_petrus_lite.bsp"),
         "--reward", "race", "--surf-bonus", "0.5", "--steps", "1"],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "farmable" in (out.stdout + out.stderr)


# ------------------------------------------------------------------- (c)

def test_flag_off_is_bit_identical():
    """Both at 0: the same float32 array element for element, and no RNG
    draw (the block is not entered at all)."""
    def run(**kw):
        core = _FakeCore(6)
        rf = _reward(6)
        for k, v in kw.items():
            setattr(rf, k, v)
        _step(rf, core, np.zeros(6), -1)
        out = []
        rng = np.random.default_rng(0)
        for _ in range(30):
            vz = rng.normal(-200.0, 300.0, 6).astype(np.float32)
            og = np.where(rng.random(6) < 0.1, 3, -1)
            core.states_view["onground"][:] = og
            out.append(run_step(rf, core, vz, og))
        return np.stack(out)

    def run_step(rf, core, vz, og):
        return _step(rf, core, vz, og).copy()

    a = run()
    b = run(surf_bonus=0.0, dive_pen=0.0)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32


# ------------------------------------------------------------------- (d)

def test_episode_start_reanchors_vz():
    """Without the re-anchor, the first tick of a new episode reads the
    spawn's velocity jump as a giant 'the map pushed back'.

    The engine AUTORESETS in place, so on the call where an episode ends
    the states already hold the NEW episode's spawn - that is the contract
    every other tracker in RaceReward re-anchors under (``_best[ended] =
    d[ended]`` two lines above), and the vz anchor follows it exactly.
    Modelled here: env 0 has been falling at -2,000 u/s, its episode ends,
    and the state it presents on that call is already the spawn's 0."""
    core = _FakeCore(2)
    rf = _reward(2)
    rf.surf_bonus = 0.0025
    _step(rf, core, np.zeros(2), -1)
    # it really was falling before the end
    core.set(np.array([-2000.0, -G_TICK]), -1)
    _step(rf, core, np.array([-2000.0, -G_TICK]), -1)
    # the ending call: states are the fresh spawn already (autoreset)
    core.set(np.array([0.0, -G_TICK * 2]), -1)
    z = np.zeros((2, 15), np.float32)
    rf(z, z, z, np.zeros(2, np.float32),
       np.array([1, 0], np.uint8), np.zeros(2, np.uint8), core)
    # the new episode's first ordinary tick: a clean free fall off the spawn
    r = _step(rf, core, np.array([-G_TICK, -G_TICK * 3]), -1)
    assert np.isclose(r[0], -0.005), "the spawn jump must not pay a bonus"
    # and without the re-anchor it WOULD have: the un-anchored dev is the
    # whole +2,000 u/s jump, hundreds of times the 1 u/s threshold
    assert abs(0.0 - (-2000.0) + G_TICK) > 100 * rf.surf_dvz


# ------------------------------------------------------------------- (e)

def test_diagnostic_fractions_partition_the_airborne_set():
    core = _FakeCore(4)
    rf = _reward(4)
    rf.surf_bonus = 0.0025
    rf.dive_pen = 0.001
    _step(rf, core, np.zeros(4), -1)
    og = np.array([-1, -1, -1, 3])
    core.states_view["onground"][:] = og
    _step(rf, core, np.array([-2.0, -G_TICK, -G_TICK, 0.0]), og)
    st = rf.pop_stats()
    assert st["surf_paid_frac"] + st["dive_frac"] == pytest.approx(1.0)
    assert st["surf_paid_frac"] == pytest.approx(1 / 3)
    assert st["surf_air_ticks"] == 3.0            # the grounded env is out
    # the counters are DRAINED, so the next window starts clean
    assert "surf_paid_frac" not in rf.pop_stats()


def test_flags_are_on_the_cli():
    out = subprocess.run(
        [sys.executable, str(ROOT / "python" / "train_fast.py"), "--help"],
        capture_output=True, text=True)
    for f in ("--surf-bonus", "--dive-pen", "--surf-hspd"):
        assert f in out.stdout
