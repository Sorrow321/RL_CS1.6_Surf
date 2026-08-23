"""``--race-dfloor``: floor the shaping potential, and change nothing else.

The arm under test replaces the geodesic potential ``Phi = -d`` with
``Phi = -max(d, d_floor)``. What these pin, in the order that matters:

  * **the flag OFF is the control, bit for bit.** Same reward array, same
    dtype, same stall mask, same stagnant mask, over a scripted trajectory -
    an arm that is supposed to start identical has to be *provably*
    identical, not "looks the same on the curve";
  * the clamped term is still POTENTIAL-BASED: a closed loop nets 0, an
    out-and-back refunds itself, and the total collectible over a run from
    ``d0`` to the goal is exactly ``scale * (d0 - d_floor)``;
  * inside the shell (``d <= d_floor``) the shaping pays exactly 0 in BOTH
    directions - that is the treatment: on this map the field's low-d shell
    reaches into a lethal void and the unclamped term PAYS for falling into
    it;
  * it is a function of STATE, not of history - visiting the shell and
    coming back out leaves no ratchet behind (a running-minimum version
    would, and would not be representable by the critic);
  * **liveness keeps the RAW d.** The stall detector and the respawn
    ``stagnant`` mask are defined on the geodesic and are not part of this
    treatment; fed the clamped value every state inside the shell would read
    as stagnant and the 15 s stall-kill would fire on a correct final
    approach. This is the one that silently invalidates the run, so it is
    tested from both ends: the kill still fires when the agent really is
    stuck inside the shell, and it does NOT fire while the agent is still
    closing on the goal inside it.

    python -m pytest tests/python/test_race_dfloor.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.goalfield import EuclidField                  # noqa: E402
from surfgym.rewards import RaceReward                     # noqa: E402

ZONES = ROOT / "maps" / "surf_src_cannonball.zones.json"


class FakeCore:
    """Only what RaceReward reads: the states view, goal_hits, map_bounds."""

    def __init__(self, n: int = 1, bounds=((-3e4, -3e4, -3e4), (3e4, 3e4, 3e4))):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self._bounds = (np.asarray(bounds[0], np.float32),
                        np.asarray(bounds[1], np.float32))

    def map_bounds(self):
        return self._bounds

    def at(self, pos, vel=(0.0, 0.0, 0.0)):
        self.states_view["origin"] = np.asarray(pos, np.float32)
        self.states_view["velocity"] = np.asarray(vel, np.float32)
        return self


def _step(rw, core, done=None, trunc=None):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    done = z if done is None else np.asarray(done, np.uint8)
    trunc = z if trunc is None else np.asarray(trunc, np.uint8)
    obs = np.zeros((n, 15), np.float32)
    return rw(obs, obs, obs, np.zeros(n, np.float32), done, trunc, core)


@pytest.fixture
def goal_box():
    if ZONES.exists():
        return json.loads(ZONES.read_text(encoding="utf-8"))["end"]
    return {"mins": [-14720.0, 7487.0, -1824.0],
            "maxs": [-8064.0, 7488.0, -352.0]}


def _field(goal_box):
    return EuclidField(goal_box)


def _race(goal_box, **kw):
    kw.setdefault("scale", 100.0 / 20000.0)
    kw.setdefault("time_pen", 0.0)
    kw.setdefault("success_bonus", 50.0)
    return RaceReward(_field(goal_box), **kw)


# a straight line in from far out, along -y toward the box face at y = 7487
def _approach(core, t):
    core.at([-11000.0, 7487.0 - t, -1000.0])
    return core


# --------------------------------------------------------------------------
# 1. flag off == control, bit for bit
# --------------------------------------------------------------------------
def test_dfloor_zero_is_the_control_bit_for_bit(goal_box):
    """No floor must reproduce the baseline reward EXACTLY - same values,
    same dtype - over a trajectory that also spends time deep inside where a
    floor would bite."""
    a = _race(goal_box, time_pen=0.005, int_coef=0.0)
    b = _race(goal_box, time_pen=0.005, int_coef=0.0, d_floor=0.0)
    ca, cb = FakeCore(), FakeCore()
    ra, rb = [], []
    dist = list(range(20000, 0, -137)) + list(range(200, 20000, 311))
    for t in dist:
        ra.append(_step(a, _approach(ca, float(t))).copy())
        rb.append(_step(b, _approach(cb, float(t))).copy())
    A, B = np.concatenate(ra), np.concatenate(rb)
    assert A.dtype == B.dtype == np.float32
    assert np.array_equal(A, B), "d_floor=0 diverged from the control"
    assert np.array_equal(a._d, b._d)
    assert np.array_equal(a._best, b._best)
    assert np.array_equal(a._since, b._since)


def test_dfloor_zero_leaves_the_intrinsic_and_bonus_paths_identical(goal_box):
    a = _race(goal_box, time_pen=0.005, int_coef=0.25, int_view=8, int_speed=3)
    b = _race(goal_box, time_pen=0.005, int_coef=0.25, int_view=8, int_speed=3,
              d_floor=0.0)
    ca, cb = FakeCore(2), FakeCore(2)
    out = []
    for t in range(0, 4000, 53):
        ca.at([[-11000.0 + t, 7000.0, -1000.0], [-9000.0, 3000.0 + t, -900.0]],
              [[300.0, 0.0, 0.0], [0.0, 900.0, 0.0]])
        cb.at([[-11000.0 + t, 7000.0, -1000.0], [-9000.0, 3000.0 + t, -900.0]],
              [[300.0, 0.0, 0.0], [0.0, 900.0, 0.0]])
        out.append((_step(a, ca).copy(), _step(b, cb).copy()))
    assert all(np.array_equal(x, y) for x, y in out)
    sa, sb = a.pop_stats(), b.pop_stats()
    assert sa.keys() == sb.keys()
    assert all(repr(sa[k]) == repr(sb[k]) for k in sa), (sa, sb)


# --------------------------------------------------------------------------
# 2. the clamp is what it says it is
# --------------------------------------------------------------------------
def test_inside_the_shell_the_shaping_pays_nothing_either_way(goal_box):
    """The treatment: below d_floor, approaching pays 0 and retreating
    charges 0. Anything else and this is not the ablation that was run."""
    rw = _race(goal_box, d_floor=5000.0)
    core = FakeCore()
    _step(rw, _approach(core, 4000.0))      # prime, d = 4000
    inbound = sum(float(_step(rw, _approach(core, d))[0])
                  for d in range(3900, 100, -100))
    outbound = sum(float(_step(rw, _approach(core, d))[0])
                   for d in range(200, 5000, 100))
    assert abs(inbound) < 1e-6, f"approach inside the shell paid {inbound}"
    assert abs(outbound) < 1e-6, f"retreat inside the shell paid {outbound}"


def test_outside_the_shell_nothing_changes(goal_box):
    """Above the floor the term is the control's, to the last bit."""
    a = _race(goal_box)
    b = _race(goal_box, d_floor=5000.0)
    ca, cb = FakeCore(), FakeCore()
    _step(a, _approach(ca, 19000.0))
    _step(b, _approach(cb, 19000.0))
    for d in range(18900, 5000, -100):
        x = _step(a, _approach(ca, d))
        y = _step(b, _approach(cb, d))
        assert np.array_equal(x, y), f"diverged at d={d}"


def test_the_shell_boundary_is_where_the_income_stops(goal_box):
    """Total shaping collected from d0 to the goal is exactly
    scale*(d0 - d_floor), not scale*d0."""
    d0, floor = 18000.0, 6000.0
    rw = _race(goal_box, d_floor=floor)
    core = FakeCore()
    _step(rw, _approach(core, d0))
    total = sum(float(_step(rw, _approach(core, d))[0])
                for d in np.arange(d0 - 20.0, 0.0, -20.0))
    assert total == pytest.approx(rw.scale * (d0 - floor), rel=2e-3)


def test_still_potential_based_closed_loop_nets_zero(goal_box):
    """A clamped potential is still a potential: two laps around a circle
    that dips in and out of the shell must net ~0."""
    rw = _race(goal_box, d_floor=6000.0)
    core = FakeCore()
    centre = np.array([-11000.0, 7487.0 - 6000.0, -1000.0])
    radius = 2000.0
    core.at(centre + [radius, 0.0, 0.0])
    _step(rw, core)
    total = 0.0
    for k in range(1, 2 * 128 + 1):
        a = 2.0 * np.pi * k / 128
        core.at(centre + [radius * np.cos(a), radius * np.sin(a), 0.0])
        total += float(_step(rw, core)[0])
    assert abs(total) < 1e-3, f"two laps netted {total:.6f}"


def test_it_is_a_function_of_state_not_a_ratchet(goal_box):
    """Dive deep into the shell and come back out: the reward for the leg
    that follows must depend only on where the agent IS, never on how deep it
    once went. A running-minimum implementation fails this."""
    shallow = _race(goal_box, d_floor=6000.0)
    deep = _race(goal_box, d_floor=6000.0)
    cs, cd = FakeCore(), FakeCore()
    for rw, core, bottom in ((shallow, cs, 5500.0), (deep, cd, 200.0)):
        _step(rw, _approach(core, 7000.0))
        for d in np.arange(6900.0, bottom, -100.0):
            _step(rw, _approach(core, d))
        for d in np.arange(bottom, 7000.0, 100.0):
            _step(rw, _approach(core, d))
    # both are now at d = 7000 having taken different histories
    out_s = [float(_step(shallow, _approach(cs, d))[0])
             for d in np.arange(7100.0, 9000.0, 100.0)]
    out_d = [float(_step(deep, _approach(cd, d))[0])
             for d in np.arange(7100.0, 9000.0, 100.0)]
    assert np.allclose(out_s, out_d, atol=1e-9), "the clamp carries history"


def test_the_fall_into_the_shell_stops_paying(goal_box):
    """The measured defect on surf_src_cannonball, in miniature: an agent
    that leaves the track at d = d_floor and dives to d = 3,000 collects
    positive shaping under the control and exactly zero under the clamp."""
    ctl = _race(goal_box)
    clp = _race(goal_box, d_floor=7022.0)
    cc, cl = FakeCore(), FakeCore()
    _step(ctl, _approach(cc, 7022.0))
    _step(clp, _approach(cl, 7022.0))
    paid_ctl = sum(float(_step(ctl, _approach(cc, d))[0])
                   for d in np.arange(6950.0, 3000.0, -50.0))
    paid_clp = sum(float(_step(clp, _approach(cl, d))[0])
                   for d in np.arange(6950.0, 3000.0, -50.0))
    assert paid_ctl > 0.19, f"control paid only {paid_ctl:.3f} for the dive"
    assert abs(paid_clp) < 1e-6, f"clamp still paid {paid_clp:.6f}"


# --------------------------------------------------------------------------
# 3. liveness keeps the RAW d  (getting this wrong invalidates the run)
# --------------------------------------------------------------------------
def test_stall_kill_does_not_fire_while_closing_inside_the_shell(goal_box):
    """Feed the CLAMPED distance to the stall detector and every state inside
    the shell reads as stagnant, so the 15 s kill fires on a correct final
    approach. The detector must see the raw geodesic.

    Closing speed here is 40 u/tick (~4,000 u/s, this map's cap): the
    detector's rule is `d < best - stall_eps` against a best that re-anchors
    every tick, so it is a per-tick speed floor, and the control passes it at
    racing speed for exactly the same reason.
    """
    rw = _race(goal_box, d_floor=8000.0, stall_ticks=1500, stall_eps=32.0)
    core = FakeCore()
    _step(rw, _approach(core, 7900.0))                 # start inside the shell
    d = 7900.0
    for _ in range(190):                               # 1.9 s, all inside
        d -= 40.0
        _step(rw, _approach(core, d))
        assert rw.pop_stall_mask() is None, "stall-kill fired on a live approach"
    assert not rw.stagnant_mask(300).any()
    assert d < 500.0                                   # it really did close


def test_stall_and_stagnant_are_bit_identical_to_the_control(goal_box):
    """The liveness rules are not part of this treatment: over a trajectory
    that spends its whole length inside the shell, at every closing speed
    from stalled to racing, the clamped reward's stall mask and stagnant mask
    must equal the control's tick for tick."""
    a = _race(goal_box, stall_ticks=1500, stall_eps=32.0)
    b = _race(goal_box, stall_ticks=1500, stall_eps=32.0, d_floor=9000.0)
    ca, cb = FakeCore(), FakeCore()
    _step(a, _approach(ca, 8900.0))
    _step(b, _approach(cb, 8900.0))
    d = 8900.0
    seen_kill = False
    for k in range(4000):
        d -= 40.0 if k < 100 else (0.0 if k < 2000 else 1.0)
        d = max(d, 5.0)
        _step(a, _approach(ca, d))
        _step(b, _approach(cb, d))
        ma, mb = a.pop_stall_mask(), b.pop_stall_mask()
        assert (ma is None) == (mb is None), f"stall mask diverged at {k}"
        if ma is not None:
            assert np.array_equal(ma, mb)
            seen_kill = True
        assert np.array_equal(a.stagnant_mask(300), b.stagnant_mask(300))
    assert seen_kill, "the trajectory never triggered a stall-kill"


def test_stall_kill_still_fires_when_genuinely_stuck_inside_the_shell(goal_box):
    """The other end: liveness must not be disabled by the clamp either."""
    rw = _race(goal_box, d_floor=8000.0, stall_ticks=1500, stall_eps=32.0)
    core = FakeCore()
    _step(rw, _approach(core, 5000.0))
    fired = None
    for i in range(1, 2000):
        _step(rw, _approach(core, 5000.0))   # not moving
        if rw.pop_stall_mask() is not None:
            fired = i
            break
    assert fired == 1500, f"stall-kill fired at tick {fired}, expected 1500"


def test_stagnant_mask_matches_the_control_inside_the_shell(goal_box):
    """The respawn harvest's stagnant mask is a liveness rule too: identical
    with and without the floor, over a trajectory entirely inside it."""
    a = _race(goal_box)
    b = _race(goal_box, d_floor=9000.0)
    ca, cb = FakeCore(), FakeCore()
    d = 8500.0
    _step(a, _approach(ca, d))
    _step(b, _approach(cb, d))
    for k in range(900):
        d -= 3.0 if k < 300 else 0.0
        _step(a, _approach(ca, d))
        _step(b, _approach(cb, d))
        assert np.array_equal(a.stagnant_mask(300), b.stagnant_mask(300))
    assert b.stagnant_mask(300).any(), "the trajectory never went stagnant"


# --------------------------------------------------------------------------
# 4. the rest of the reward is untouched
# --------------------------------------------------------------------------
def test_success_bonus_and_time_penalty_survive_the_clamp(goal_box):
    rw = _race(goal_box, d_floor=9000.0, time_pen=0.005)
    core = FakeCore()
    _step(rw, _approach(core, 4000.0))
    r = float(_step(rw, _approach(core, 3800.0))[0])
    assert r == pytest.approx(-0.005, abs=1e-7), "time penalty changed"
    core.goal_hits[:] = 1
    r = float(_step(rw, _approach(core, 3600.0))[0])
    assert r == pytest.approx(50.0 - 0.005, abs=1e-5)


def test_ended_rows_are_still_masked(goal_box):
    """A respawn relocation must not cash shaping under the clamp either."""
    rw = _race(goal_box, d_floor=6000.0)
    core = FakeCore()
    _step(rw, _approach(core, 5000.0))
    r = _step(rw, _approach(core, 19000.0), done=[1])
    assert float(r[0]) == 0.0


def test_a_floor_above_the_start_makes_the_shaping_sparse(goal_box):
    """Sanity on the extreme: floor above d0 and only the bonus, the time
    penalty and the intrinsic bonus are left - the --race-shaping 0 limit,
    reached from the other side."""
    rw = _race(goal_box, d_floor=1e9, time_pen=0.005)
    core = FakeCore()
    _step(rw, _approach(core, 20000.0))
    r = [float(_step(rw, _approach(core, float(t)))[0])
         for t in range(100, 9000, 100)]
    assert np.allclose(r, -0.005, atol=1e-7)


def test_the_trainer_flag_exists_and_defaults_to_off():
    """--race-dfloor must exist, default to off, and be recorded in the
    config the checkpoint carries (a resume that lost it would hand the
    policy a different objective than its weights were fitted to)."""
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--race-dfloor"' in src
    assert "args.race_dfloor = 0.0" in src
    assert '"race_dfloor": args.race_dfloor' in src
    assert 'ck_cfg.get("race_dfloor")' in src
    assert "d_floor=args.race_dfloor" in src
    # the obs-reward eval mirror has to be clamped too, or an eval feeds a
    # clamp-trained policy the unclamped signal in scalar slot 12.
    # --maps made the reward per SLOT, so the mirror reads the slot's own
    # reward function - one per map, each with its own field and scale.
    assert "d_floor=_s.reward_fn.d_floor" in src


def test_run_arm_launcher_pins_the_control(goal_box):
    """The pinned baseline config in run_arm.sh must not mention a floor:
    the control this arm is measured against has none."""
    sh = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")
    assert "race_dfloor" not in sh
