"""``--race-latch``: once an episode reaches ``d <= L``, the shaping term
pays zero for the rest of it - and nothing else changes.

The arm this pins is one line of reward logic with three ways to be silently
wrong, each of which would make a one-hour ablation unreadable:

  * **the flag OFF is the control, bit for bit.** Same reward array, same
    dtype, same stall mask, same stagnant mask, over a trajectory that also
    spends time deep inside where a latch would bite;
  * **the treatment is the CLIMB BACK OUT, not the shell.** ``--race-dfloor``
    already flattens the potential inside ``d <= L``; what it still charges
    is leaving. On cannonball the route climbs d 6,632 -> 14,976 between
    vertices 1600 and 1680, all of it *above* the floor, charged -4.02 - the
    whole valley, and the reason xCLAMP got 46/81 past the wall with 0
    finishes. The latch has to make that segment cost exactly 0;
  * **it must stay MARKOV.** The switch is episode history, so it is fed to
    the network as one extra observation column. What that column shows at
    t has to be exactly what decides whether t+1 pays shaping, or the critic
    is being asked to predict two different returns from one input. And the
    column has to be the LAST one in the row, where ``widen_for_route``
    zero-pads a checkpoint - a core scalar slot would land in the middle of
    ``feat_idx`` and permute every existing feature.

Plus the one that silently invalidates the run: **liveness keeps the RAW d.**
Past the switch every state pays 0, so on the latched value nothing would
ever look like progress and the 15 s stall-kill would fire on the whole final
descent.

    python -m pytest tests/python/test_race_latch.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.goalfield import EuclidField                  # noqa: E402
from surfgym.rewards import RaceReward                     # noqa: E402
from train_fast import N_SCALAR, Policy, widen_for_route   # noqa: E402

ZONES = ROOT / "maps" / "surf_src_cannonball.zones.json"

# the number the arm actually runs with: d at the last tick the map pushed
# back, route vertex 1595 (tools/pick_dfloor.py, derived on the clamp branch)
LATCH = 6996.0


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


def _race(goal_box, **kw):
    kw.setdefault("scale", 100.0 / 198380.0)   # the real run's scale
    kw.setdefault("time_pen", 0.0)
    kw.setdefault("success_bonus", 50.0)
    return RaceReward(EuclidField(goal_box), **kw)


def _at_d(core, d, i=0):
    """Put env `i` exactly `d` units from the box face at y = 7487."""
    p = core.states_view["origin"].copy()
    p[i] = (-11000.0, 7487.0 + float(d), -1000.0)
    core.states_view["origin"] = p
    return core


# --------------------------------------------------------------------------
# 1. the flag OFF is the control, bit for bit
# --------------------------------------------------------------------------
def test_latch_zero_is_the_control_bit_for_bit(goal_box):
    """No latch must reproduce the baseline reward EXACTLY - values, dtype,
    stall mask, stagnant mask - over a trajectory that dives well inside
    where a latch would bite and then climbs back out."""
    a = _race(goal_box, time_pen=0.005)
    b = _race(goal_box, time_pen=0.005, d_latch=0.0)
    ca, cb = FakeCore(), FakeCore()
    ra, rb = [], []
    path = (list(range(20000, 3000, -137))      # in, well past the threshold
            + list(range(3000, 20000, 311)))    # and back out again
    for d in path:
        ra.append(_step(a, _at_d(ca, d)).copy())
        rb.append(_step(b, _at_d(cb, d)).copy())
    A, B = np.concatenate(ra), np.concatenate(rb)
    assert A.dtype == B.dtype == np.float32
    assert np.array_equal(A, B), "d_latch=0 diverged from the control"
    assert np.array_equal(a.stagnant_mask(), b.stagnant_mask())
    ma, mb = a.pop_stall_mask(), b.pop_stall_mask()
    assert (ma is None and mb is None) or np.array_equal(ma, mb)


def test_latch_zero_touches_no_state(goal_box):
    """Off must not even allocate a live flag path: the array exists (it is
    the obs column) but stays all-False forever."""
    rw = _race(goal_box)
    core = FakeCore()
    for d in list(range(20000, 100, -500)):
        _step(rw, _at_d(core, d))
    assert rw.d_latch == 0.0
    assert not rw.latch_flags().any(), "the latch armed with the flag off"


# --------------------------------------------------------------------------
# 2. the treatment: after the switch, shaping is exactly zero both ways
# --------------------------------------------------------------------------
def test_shaping_is_zero_in_both_directions_after_the_switch(goal_box):
    """Approach past L, then leave. Every tick from the crossing on must pay
    exactly the time penalty and nothing else."""
    rw = _race(goal_box, time_pen=0.005, d_latch=LATCH)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))          # anchor
    seen_cross = False
    for d in [12000.0, 8000.0, LATCH - 1.0, LATCH - 30.0,
              9000.0, 14976.0, 20000.0, 4000.0]:
        r = float(_step(rw, _at_d(core, d))[0])
        if seen_cross:
            assert r == pytest.approx(-0.005, abs=1e-9), \
                f"shaping still paid at d={d}: {r}"
        if d <= LATCH:
            seen_cross = True
    assert seen_cross


def test_the_valley_costs_nothing_and_the_clamp_still_charges_it(goal_box):
    """The claim the arm exists for. Route vertices 1600 -> 1680 raise the
    geodesic 6,632 -> 14,976; every one of those states is ABOVE a 6,996
    floor, so --race-dfloor charges the whole climb and the latch does not."""
    # < max_step (100 u) per tick: the shaping delta is clipped there,
    # and a leg that outran it would be measuring the clip
    seg = np.arange(6632.0, 14976.0, 80.0)

    def run(**kw):
        rw = _race(goal_box, time_pen=0.0, **kw)
        core = FakeCore()
        _step(rw, _at_d(core, 20000.0))
        _step(rw, _at_d(core, LATCH - 1.0))   # cross the threshold once
        return sum(float(_step(rw, _at_d(core, float(d)))[0]) for d in seg)

    clamp = run(d_floor=LATCH)
    latch = run(d_latch=LATCH)
    assert latch == pytest.approx(0.0, abs=1e-9), \
        f"the latch charged the climb back out: {latch}"
    # 100/198380 * (14976 - 6996) = -4.02, the figure in the ledger
    assert clamp == pytest.approx(-4.02, abs=0.02), clamp


def test_total_shaping_is_the_approach_only(goal_box):
    """A full run start -> switch -> goal collects scale*(d0 - L) of shaping
    and not one unit more, whatever it does after the switch."""
    scale = 100.0 / 198380.0
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH)
    core = FakeCore()
    d0 = 20000.0
    _step(rw, _at_d(core, d0))
    total = 0.0
    for d in (list(np.arange(d0, LATCH - 5.0, -90.0)) + [LATCH - 5.0]
              + list(np.arange(LATCH - 5.0, 15000.0, 90.0))
              + list(np.arange(15000.0, 5.0, -90.0))):
        total += float(_step(rw, _at_d(core, float(d)))[0])
    assert total == pytest.approx(scale * (d0 - (LATCH - 5.0)), abs=1e-4)


# --------------------------------------------------------------------------
# 3. episode boundaries
# --------------------------------------------------------------------------
def test_latch_clears_at_an_episode_start(goal_box):
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))
    _step(rw, _at_d(core, 3000.0))                       # latched
    assert rw.latch_flags()[0]
    # the episode ends; the post-autoreset state is the NEW spawn, far out
    _step(rw, _at_d(core, 20000.0), done=[1])
    assert not rw.latch_flags()[0], "the latch survived an episode boundary"
    r = float(_step(rw, _at_d(core, 19000.0))[0])
    assert r > 0.0, "the new episode is not being paid shaping"


def test_a_spawn_inside_the_shell_starts_latched(goal_box):
    """At --respawn-margin 2 the reservoir really does place starts past the
    wall. Such a spawn IS a tick with d <= L, so it arms immediately -
    otherwise the very episodes this arm is about get charged for leaving."""
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))
    _step(rw, _at_d(core, 5000.0), done=[1])   # ends; respawns INSIDE
    assert rw.latch_flags()[0]
    assert float(_step(rw, _at_d(core, 15000.0))[0]) == pytest.approx(0.0)

    # and on_reset does the same for the very first episode
    rw2 = _race(goal_box, time_pen=0.0, d_latch=LATCH)
    rw2.on_reset(_at_d(FakeCore(), 5000.0))
    assert rw2.latch_flags()[0]


def test_envs_are_independent(goal_box):
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH)
    core = FakeCore(3)
    core.states_view["origin"] = np.array(
        [(-11000.0, 7487.0 + 20000.0, -1000.0)] * 3, np.float32)
    _step(rw, core)
    _at_d(core, 20000.0, 0); _at_d(core, 3000.0, 1); _at_d(core, 20000.0, 2)
    _step(rw, core)
    assert list(rw.latch_flags()) == [False, True, False]


# --------------------------------------------------------------------------
# 4. Markov: the observation column IS the switch
# --------------------------------------------------------------------------
def test_the_column_at_t_predicts_whether_t_plus_1_pays(goal_box):
    """The critic sees latch_flags() at t and has to predict the return from
    there. So the flag read at t must be exactly the one that decides t+1's
    shaping - off by one and one input carries two different returns."""
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))
    path = [15000.0, 9000.0, 7000.0, LATCH, 4000.0, 12000.0, 20000.0,
            18000.0, 200.0, 9000.0]
    flag = bool(rw.latch_flags()[0])
    for d in path:
        r = float(_step(rw, _at_d(core, d))[0])
        if flag:
            assert r == 0.0, f"latched at t but t+1 paid {r} (d={d})"
        else:
            assert r != 0.0, f"unlatched at t but t+1 paid nothing (d={d})"
        flag = bool(rw.latch_flags()[0])


def test_latch_boot_is_the_flag_one_call_ago(goal_box):
    """What the truncation bootstrap feeds V(s_T): the autoreset has already
    moved latch_flags() on to the next episode's spawn by then."""
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))
    _step(rw, _at_d(core, 3000.0))
    assert rw.latch_flags()[0] and not rw.latch_boot()[0]
    _step(rw, _at_d(core, 12000.0))
    assert rw.latch_flags()[0] and rw.latch_boot()[0]
    # a truncation: live state is already the new far spawn, boot still says
    # the terminal state of the OLD episode was latched
    _step(rw, _at_d(core, 20000.0), trunc=[1])
    assert not rw.latch_flags()[0]
    assert rw.latch_boot()[0]


# --------------------------------------------------------------------------
# 5. liveness keeps the RAW d
# --------------------------------------------------------------------------
def test_stall_kill_still_fires_when_stuck_inside_the_shell(goal_box):
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH, stall_ticks=100)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))
    _step(rw, _at_d(core, 3000.0))
    for _ in range(120):
        _step(rw, _at_d(core, 3000.0))
    m = rw.pop_stall_mask()
    assert m is not None and m[0] == 1, \
        "the stall kill stopped seeing a genuinely stuck env inside the shell"


def test_stall_kill_does_not_fire_on_a_correct_approach_inside_the_shell(goal_box):
    """The one that silently invalidates the run: past the switch the reward
    is flat, so a liveness rule fed the latched value would kill every
    correct final descent 15 s in."""
    rw = _race(goal_box, time_pen=0.0, d_latch=LATCH, stall_ticks=100)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))
    d = LATCH - 1.0
    for _ in range(150):        # 40 u/tick, above the 32 u stall_eps
        d = max(20.0, d - 40.0)
        _step(rw, _at_d(core, d))
        assert rw.pop_stall_mask() is None, \
            "stall-killed while still closing on the goal inside the shell"
    assert not rw.stagnant_mask()[0]


def test_the_goal_bonus_and_the_time_penalty_are_untouched(goal_box):
    rw = _race(goal_box, time_pen=0.005, d_latch=LATCH, success_bonus=50.0)
    core = FakeCore()
    _step(rw, _at_d(core, 20000.0))
    _step(rw, _at_d(core, 3000.0))
    r = float(_step(rw, _at_d(core, 100.0))[0])
    assert r == pytest.approx(-0.005, abs=1e-9)      # time penalty survives
    core.goal_hits[:] = 1
    r = float(_step(rw, _at_d(core, 10.0), done=[1])[0])
    assert r == pytest.approx(50.0, abs=1e-6)        # ended row: bonus only


# --------------------------------------------------------------------------
# 6. the obs column is the LAST one, and a warm resume is identical
# --------------------------------------------------------------------------
LW, LH = 16, 8
IMG = LW * LH


def _policy(route_dim=0, seed=0):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + route_dim + IMG, LW, LH, emb=16, hidden=12,
                  route_dim=route_dim)


def test_a_one_wide_block_warm_resumes_function_identically():
    """The whole reason the flag rides the route block: widen_for_route
    zero-pads the checkpoint's TRAILING columns, so step 0 is the baseline
    whatever the flag says."""
    base = _policy(0, seed=11)
    opt = torch.optim.Adam(base.parameters(), lr=1e-4)
    base(torch.randn(3, N_SCALAR + IMG))[1].sum().backward()
    opt.step()
    ck = {"policy": base.state_dict(), "optimizer": opt.state_dict()}

    wide = _policy(1, seed=99)                      # a DIFFERENT init
    assert widen_for_route(ck, wide) > 0
    wide.load_state_dict(ck["policy"])
    torch.optim.Adam(wide.parameters(), lr=1e-4).load_state_dict(ck["optimizer"])

    scal, img = torch.randn(5, N_SCALAR), torch.randn(5, IMG)
    want_l, want_v = base(torch.cat([scal, img], 1))
    for flag in (torch.zeros(5, 1), torch.ones(5, 1)):
        got_l, got_v = wide(torch.cat([scal, flag, img], 1))
        assert torch.equal(got_l, want_l), \
            "the latch column is not zero-padded: step 0 is not the baseline"
        assert torch.equal(got_v, want_v)


def test_the_flag_reaches_both_towers():
    """It has to reach the CRITIC (that is the point) and, with
    --route-critic-only off, the actor too."""
    p = _policy(1, seed=5)
    torch.nn.init.normal_(p.vf[0].weight[:, -1:], std=3.0)
    torch.nn.init.normal_(p.pi[0].weight[:, -1:], std=3.0)
    scal, img = torch.randn(4, N_SCALAR), torch.randn(4, IMG)
    l0, v0 = p(torch.cat([scal, torch.zeros(4, 1), img], 1))
    l1, v1 = p(torch.cat([scal, torch.ones(4, 1), img], 1))
    assert not torch.equal(v0, v1), "the critic cannot see the latch"
    assert not torch.equal(l0, l1), "the actor cannot see the latch"


# --------------------------------------------------------------------------
# 7. the flag surface
# --------------------------------------------------------------------------
def test_the_flag_exists_defaults_off_and_is_recorded():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--race-latch"' in src
    assert "args.race_latch = 0.0" in src                  # default off
    assert '"race_latch": args.race_latch' in src          # saved in config
    assert 'ck_cfg.get("race_latch")' in src               # restored on resume
    # --maps resolved the threshold per slot (--race-latch-frac is a
    # fraction of each map's own d0); the absolute flag still lands there
    assert "_s.d_latch = (args.race_latch_frac * _s.rf_d0" in src
    assert "else args.race_latch)" in src
    assert "d_latch=_s.d_latch" in src                     # reaches the reward
    assert "N_LATCH = 1 if (args.race_latch > 0.0" in src
    assert "latch_fn=_s.eval_latch_feed" in src            # and the evals


def test_the_launcher_does_not_bake_the_arm_in():
    sh = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")
    assert "race_latch" not in sh and "race-latch" not in sh
