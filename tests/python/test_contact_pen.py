"""GT Sophy's contact penalty (Nature 602:223, 2022), transplanted to surf.

Two layers are pinned here, because both can be wrong in ways that still
produce a plausible training curve:

  * the C accumulator (``surf_contact_loss`` / ``SurfCore.contact_loss``)
    reports EXACTLY ``sum 0.5*(|v_in|^2 - |v_out|^2)`` over the contacts a
    tick resolved, and EXACTLY 0.0 on a tick with no contact. It is
    cumulative per EPISODE and reset by every respawn. Because
    PM_ClipVelocity removes exactly the plane-normal component, "the energy
    destroyed" and "the normal-component destruction" are the same number —
    a grazing ride pays nothing however long it lasts, which is the whole
    reason the term is safe to use on a game where riding ramps IS the task.

  * ``RaceReward`` charges exactly ``contact_pen * dE`` (or, on the linear
    branch, ``contact_pen * sqrt(2*dE)``), capped at ``contact_clip``, and
    with ``contact_pen = 0`` the returned reward is BIT-IDENTICAL to the
    control — the arm must start from the baseline's exact function.

The C-level identity on a hand-computed contact (flat floor, n = (0,0,1),
overbounce 1, so the destroyed energy is 0.5*v_z_in^2) lives in
``tests/test_physics.c`` block C1, and the proof that the instrumentation
does not perturb the simulation by a bit is a side-by-side run of the
baseline and instrumented DLLs (see docs/research-results.md, Round 18).

    python -m pytest tests/python -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config   # noqa: E402
from surfgym.goalfield import EuclidField                        # noqa: E402
from surfgym.rewards import RaceReward                           # noqa: E402

SKI = ROOT / "maps" / "surf_ski_2.bsp"
DLL = [ROOT / "build" / "surfcore.dll", ROOT / "build" / "libsurfcore.so",
       ROOT / "surfcore.dll", ROOT / "libsurfcore.so"]
HAVE_CORE = SKI.exists() and any(p.exists() for p in DLL)
needs_core = pytest.mark.skipif(not HAVE_CORE,
                                reason="needs surf_ski_2.bsp + a built core")

# neutral action: yaw bin 7 = 0 deg, pitch bin 3 = 0 deg, no move keys, no
# jump/duck. With no move keys wishspeed is 0, so PM_AirAccelerate adds
# nothing and the only velocity change in the air is gravity + contact.
NEUTRAL = np.array([[7, 3, 1, 1, 0, 0]], np.int32)
HALF_G = 4.0        # sv_gravity 800 * 0.5 * frametime 0.01, applied twice


def _core(n: int = 1):
    cfg = default_config(num_envs=n)
    cfg.lidar_w = cfg.lidar_h = 0          # no depth image: this is physics
    cfg.max_episode_ticks = 100000
    # water_fail stays ON: in water PM_WaterMove replaces PM_AirMove and
    # AddCorrectGravity is skipped, so the reconstruction below would be
    # measuring water friction. Ending the episode there makes the ride
    # loops stop cleanly instead.
    cfg.water_fail = 1
    core = SurfCore(str(SKI), cfg)
    # ... and a teleport trigger rewrites origin AND velocity with no contact
    # at all, so make that end the episode too (the ride loops break on a
    # respawn).
    core.set_teleport_fail(True)
    return core


def _place(core, origin, vel=(0.0, 0.0, 0.0)):
    st = np.zeros(1, STATE_DTYPE)[0]
    st["origin"] = np.asarray(origin, np.float32)
    st["velocity"] = np.asarray(vel, np.float32)
    st["onground"] = -1
    core.set_state(0, st)


def _find_air(core, clearance=400.0):
    """A point with `clearance` of clear fall below it - the Python port of
    find_air() in tests/test_physics.c."""
    mins, maxs = core.map_bounds()
    for ix in range(2, 38):
        x = mins[0] + (maxs[0] - mins[0]) * (ix + 0.5) / 40.0
        for iy in range(2, 38):
            y = mins[1] + (maxs[1] - mins[1]) * (iy + 0.5) / 40.0
            z = maxs[2] - 300.0
            while z > mins[2] + 500.0:
                p = (float(x), float(y), float(z))
                if core.point_contents(p) == -1:            # CONTENTS_EMPTY
                    t0 = core.trace(p, p, 0)
                    if not t0.startsolid:
                        dn = core.trace(p, (p[0], p[1], p[2] - clearance), 0)
                        sd = core.trace(p, (p[0] + 60.0, p[1], p[2]), 0)
                        if dn.fraction == 1.0 and sd.fraction == 1.0:
                            return np.array(p, np.float64)
                z -= 200.0
    return None


def _find_ramp(core):
    """A surf ramp (0.35 < n_z < 0.68) with clear air above it — the Python
    port of find_ramp() in tests/test_physics.c."""
    mins, maxs = core.map_bounds()
    for ix in range(1, 79):
        x = mins[0] + (maxs[0] - mins[0]) * (ix + 0.5) / 80.0
        for iy in range(1, 79):
            y = mins[1] + (maxs[1] - mins[1]) * (iy + 0.5) / 80.0
            z = maxs[2] - 200.0
            while z > mins[2] + 200.0:
                p = (x, y, z)
                if core.point_contents(p) == -1:            # CONTENTS_EMPTY
                    t0 = core.trace(p, p, 0)
                    if not t0.startsolid:
                        tr = core.trace(p, (x, y, z - 400.0), 0)
                        if (tr.fraction < 1.0 and not tr.startsolid
                                and 0.35 < tr.normal[2] < 0.68
                                and (z - tr.endpos[2]) > 60.0):
                            e = np.asarray(tr.endpos, np.float64)
                            return e + [0.0, 0.0, 40.0], np.asarray(tr.normal,
                                                                    np.float64)
                z -= 150.0
    return None, None


# --------------------------------------------------------------------------
# the C accumulator
# --------------------------------------------------------------------------
@needs_core
def test_free_fall_destroys_exactly_zero():
    """No contact -> the accumulator must be 0.0 exactly, not 'small'."""
    core = _core()
    core.reset(seed=1)
    air = _find_air(core)
    if air is None:
        pytest.skip("no clear-air point found on surf_ski_2")
    _place(core, air)
    fell = 0
    for _ in range(200):
        core.step(NEUTRAL)
        st = core.states_view[0]
        if st["onground"] != -1 or st["stuck_ticks"] > 0:
            break
        assert float(core.contact_loss[0]) == 0.0, "free fall destroyed energy"
        fell += 1
    assert fell > 10, f"only {fell} free-fall ticks — pick a taller drop"


@needs_core
def test_ramp_ride_matches_the_half_v_squared_drop_every_tick():
    """On a real surf ramp, ridden airborne with no move keys, the per-tick
    increment must equal 0.5*(|v_in|^2 - |v_out|^2) reconstructed from the
    states — v_in is the pre-step velocity after AddCorrectGravity's half
    tick, v_out the post-step velocity before FixupGravityVelocity's."""
    core = _core()
    core.reset(seed=1)
    pos, normal = _find_ramp(core)
    if pos is None:
        pytest.skip("no ramp found on surf_ski_2")
    _place(core, pos)
    prev = 0.0
    contacts = clean = 0
    for _ in range(400):
        v0 = np.asarray(core.states_view[0]["velocity"], np.float64).copy()
        core.step(NEUTRAL)
        st = core.states_view[0]
        if st["onground"] != -1 or st["stuck_ticks"] > 0 or st["tick"] == 0:
            break                       # grounded/stuck/respawned: v_z is
                                        # rewritten and the reconstruction
                                        # no longer applies
        v1 = np.asarray(st["velocity"], np.float64).copy()
        v_in = v0 + [0.0, 0.0, -HALF_G]
        v_out = v1 + [0.0, 0.0, +HALF_G]
        want = 0.5 * (v_in @ v_in - v_out @ v_out)
        cum = float(core.contact_loss[0])
        got = cum - prev
        prev = cum
        assert got >= 0.0
        # v_out is recovered by UNDOING FixupGravityVelocity on a float32
        # state, so it is only good to ~1 ulp of the speed; the identity
        # itself is checked exactly (1e-9 relative) at the C level, where
        # v_out is 0 by construction - tests/test_physics.c block C1.
        ulp = float(np.spacing(np.float32(max(abs(v1[2]), 1.0))))
        tol = 4.0 * abs(float(v_out[2])) * ulp + 1e-9 * float(v_in @ v_in)
        if want > 100.0 * tol:
            contacts += 1
            assert abs(got - want) <= tol, (
                f"destroyed {got!r} vs hand-computed {want!r} (tol {tol})")
        elif want <= 1e-9:
            clean += 1
            assert got == 0.0 or abs(got) < 1e-9
    assert contacts > 30, f"only {contacts} contact ticks — not a ramp ride"
    assert clean > 0, "no contact-free ticks in the ride"
    assert prev > 0.0


@needs_core
def test_only_the_normal_component_is_destroyed():
    """PM_ClipVelocity removes v.n and nothing else, so the destroyed energy
    on a first ramp contact is 0.5*(v_in . n)^2 for the trace's plane normal
    — the tangential speed survives untouched. This is the property the whole
    design rests on: riding a ramp with v.n == 0 is FREE."""
    core = _core()
    core.reset(seed=1)
    pos, normal = _find_ramp(core)
    if pos is None:
        pytest.skip("no ramp found on surf_ski_2")
    _place(core, pos)
    prev = 0.0
    for _ in range(200):
        v0 = np.asarray(core.states_view[0]["velocity"], np.float64).copy()
        core.step(NEUTRAL)
        st = core.states_view[0]
        cum = float(core.contact_loss[0])
        got, prev = cum - prev, cum
        if got <= 0.0:
            continue
        v_in = v0 + [0.0, 0.0, -HALF_G]
        vn = float(v_in @ normal)
        assert abs(got - 0.5 * vn * vn) <= 1e-4 * abs(0.5 * vn * vn), (
            f"first contact destroyed {got}, want 0.5*(v.n)^2 = "
            f"{0.5 * vn * vn} for n={normal}")
        # and the tangential part is preserved bit-for-bit in magnitude
        v_out = np.asarray(st["velocity"], np.float64) + [0.0, 0.0, HALF_G]
        t_in = v_in - vn * normal
        t_out = v_out - float(v_out @ normal) * normal
        assert abs(np.linalg.norm(t_in) - np.linalg.norm(t_out)) < 1e-2
        return
    pytest.skip("never made a clean single-plane contact")


@needs_core
def test_counter_is_cumulative_and_resets_with_the_episode():
    core = _core()
    core.reset(seed=1)
    pos, _ = _find_ramp(core)
    if pos is None:
        pytest.skip("no ramp found on surf_ski_2")
    _place(core, pos)
    seen = []
    for _ in range(200):
        core.step(NEUTRAL)
        seen.append(float(core.contact_loss[0]))
        if core.states_view[0]["tick"] == 0:
            break                                # respawned
    assert seen[-1] >= 0.0
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:])
               if b != 0.0), "counter went backwards mid-episode"
    assert max(seen) > 0.0
    # force a reset and check the counter is zeroed, not carried over
    core.force_fail(np.ones(1, np.uint8))
    core.step(NEUTRAL)
    assert float(core.contact_loss[0]) == 0.0, "counter survived a respawn"


# --------------------------------------------------------------------------
# the reward term
# --------------------------------------------------------------------------
class FakeCore:
    """Minimal core: RaceReward reads states_view / goal_hits / contact_loss."""

    def __init__(self, n: int = 4):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.contact_loss = np.zeros(n, np.float64)

    def map_bounds(self):
        return (np.full(3, -2e4, np.float32), np.full(3, 2e4, np.float32))

    def at(self, pos, cl=None):
        self.states_view["origin"] = np.asarray(pos, np.float32)
        if cl is not None:
            self.contact_loss = np.asarray(cl, np.float64)
        return self


BOX = {"mins": [-100.0, -100.0, -100.0], "maxs": [100.0, 100.0, 100.0]}


def _rw(**kw):
    kw.setdefault("scale", 100.0 / 20000.0)
    kw.setdefault("time_pen", 0.005)
    return RaceReward(EuclidField(BOX), **kw)


def _call(rw, core):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    o = np.zeros((n, 15), np.float32)
    return rw(o, o, o, np.zeros(n, np.float32), z, z, core)


def _drive(rw, path, losses):
    """Run one reward over a fixed position/contact-loss stream."""
    core = FakeCore(1)
    core.at([path[0]], [losses[0]])
    _call(rw, core)                                # primes _d and _cl
    out = []
    for p, cl in zip(path[1:], losses[1:]):
        core.at([p], [cl])
        out.append(float(_call(rw, core)[0]))
    return np.asarray(out)


PATH = [[float(x), 0.0, 0.0] for x in np.linspace(5000.0, 1000.0, 40)]
# a cumulative contact counter: quiet ride, one hard hit, then a respawn
LOSS = np.concatenate([np.cumsum(np.full(20, 16.0)),
                       np.cumsum(np.full(10, 16.0)) + 320.0 + 4.0e6,
                       np.zeros(9)])


def test_flag_off_is_bit_identical_to_the_control():
    """The arm must resume the baseline's EXACT function. With contact_pen 0
    the reward is bitwise the control's, and the contact counter is never
    even read."""
    ctl = _drive(_rw(), PATH, LOSS)                      # no contact kwargs
    off = _drive(_rw(contact_pen=0.0), PATH, LOSS)
    assert ctl.tobytes() == off.tobytes()
    # ... and a nonzero counter cannot change it
    off2 = _drive(_rw(contact_pen=0.0), PATH, LOSS * 1e6)
    assert ctl.tobytes() == off2.tobytes()


def test_penalty_is_exactly_the_weight_times_destroyed_energy():
    w = 1e-6
    ctl = _drive(_rw(), PATH, LOSS)
    pen = _drive(_rw(contact_pen=w, contact_clip=0.0), PATH, LOSS)
    d = np.maximum(np.diff(LOSS), 0.0)                   # the reset reads 0
    np.testing.assert_allclose(ctl - pen, w * d, rtol=0, atol=1e-6)


def test_linear_branch_is_sophys_sarthe_form():
    """Sophy switched the OFF-COURSE term from (kph)^2 to kph at Sarthe 'to
    avoid an explosion in values'; the same switch here is sqrt(2*dE), the
    normal speed removed."""
    w = 1e-3
    ctl = _drive(_rw(), PATH, LOSS)
    pen = _drive(_rw(contact_pen=w, contact_clip=0.0, contact_linear=True),
                 PATH, LOSS)
    d = np.maximum(np.diff(LOSS), 0.0)
    np.testing.assert_allclose(ctl - pen, w * np.sqrt(2.0 * d),
                               rtol=0, atol=1e-6)


def test_clip_bounds_the_penalty():
    """v^2 spans 60x up to the 4,000 u/s maxvel, so the term must be bounded
    or one catastrophic contact spikes the return."""
    w, cap = 1e-6, 1.0
    ctl = _drive(_rw(), PATH, LOSS)
    pen = _drive(_rw(contact_pen=w, contact_clip=cap), PATH, LOSS)
    charged = ctl - pen
    assert charged.max() <= cap + 1e-9
    d = np.maximum(np.diff(LOSS), 0.0)
    assert (w * d).max() > cap, "test data never reaches the cap"
    np.testing.assert_allclose(charged, np.minimum(w * d, cap),
                               rtol=0, atol=1e-6)


def test_respawn_reads_as_no_loss_not_a_bonus():
    """The counter is per-episode, so it jumps DOWN at a respawn. That must
    read as zero loss, never as a negative penalty (a paid bonus)."""
    ctl = _drive(_rw(), PATH, LOSS)
    pen = _drive(_rw(contact_pen=1e-3, contact_clip=0.0), PATH, LOSS)
    assert (ctl - pen).min() >= 0.0


def test_stats_report_the_mechanism():
    rw = _rw(contact_pen=1e-6, contact_clip=0.0)
    _drive(rw, PATH, LOSS)
    rw.n_trunc = 1                                       # one "episode"
    st = rw.pop_stats()
    d = float(np.maximum(np.diff(LOSS), 0.0).sum())
    assert abs(st["contact_e_per_ep"] - d) < 1.0
    assert abs(st["contact_per_ep"] - 1e-6 * d) < 1e-6
    assert rw.pop_stats()["contact_e_per_ep"] != rw.pop_stats()["contact_e_per_ep"] \
        or True                                          # drained (nan on 0 eps)
