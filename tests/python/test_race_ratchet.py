"""``--race-ratchet``: inside an episode only NEW progress RECORDS are paid.

The reward keeps a per-episode record ``b`` of the best (smallest) geodesic
``d`` reached - ``b_0`` = the distance at the episode's own start, spawn and
reservoir respawn alike - and the shaping term becomes
``scale * (b_t - b_{t+1}) >= 0``.  ``d 10 -> 8 -> 12 -> 8 -> 6`` pays
``+2, 0, 0, +2`` where the stock signed term pays ``+2, -4, +4, +2``.

Round 19 measured both halves of why this is the arm: a FLAT potential
inside the trap is not enough (xCLAMP, ``--race-dfloor``: 0/99 finishes) and
making LEAVING free is (xLATCH, ``--race-latch``: 52/102).  The latch needs
a threshold picked off a champion trace; the ratchet is the same property
with no threshold and no reference line.

Four ways it could be silently wrong, each of which makes a one-hour arm
unreadable:

  * **the flag OFF is the control, bit for bit** - same array, same dtype,
    same stall and stagnant masks, over a path that backtracks repeatedly;
  * **a RESET must never generate progress reward.**  The record restarts at
    the new spawn, so a reservoir respawn deep in the map neither pays for
    the jump nor leaves the fresh episode unable to earn anything;
  * **it must stay MARKOV.**  The record is episode history, so it is fed to
    the network as one extra column, ``(d - b)/d0``: zero exactly at a
    record, positive by how far the episode has backed off.  With ``d`` in
    the row that column determines ``b``, hence the next transition's pay;
  * **liveness keeps the RAW d** - the stall detector and the respawn
    stagnant mask are not part of this treatment.

    python -m pytest tests/python/test_race_ratchet.py -q
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
D0 = 198380.0


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
    kw.setdefault("scale", 100.0 / D0)
    kw.setdefault("time_pen", 0.0)
    kw.setdefault("success_bonus", 50.0)
    return RaceReward(EuclidField(goal_box), **kw)


def _at(core, d, i=0):
    """Put env `i` so its field distance is `d` units (the box face sits at
    y = 7488, so the offset is d)."""
    p = core.states_view["origin"].copy()
    p[i] = (-11000.0, 7488.0 + float(d), -1000.0)
    core.states_view["origin"] = p
    return core


def _d_of(rw, d):
    """The field's own value where _at() puts the player - the tests assert
    on differences, so the exact face offset never matters."""
    c = FakeCore()
    return float(rw.field.sample(_at(c, d).states_view["origin"])[0])


# --------------------------------------------------------------------------
# 1. the flag OFF is the control, bit for bit
# --------------------------------------------------------------------------
def test_ratchet_off_is_the_control_bit_for_bit(goal_box):
    a = _race(goal_box, time_pen=0.005)
    b = _race(goal_box, time_pen=0.005, ratchet=False)
    ca, cb = FakeCore(), FakeCore()
    ra, rb = [], []
    path = (list(range(20000, 3000, -137))       # in
            + list(range(3000, 12000, 311))      # back out
            + list(range(12000, 200, -97)))      # and in again
    for d in path:
        ra.append(_step(a, _at(ca, d)).copy())
        rb.append(_step(b, _at(cb, d)).copy())
    A, B = np.concatenate(ra), np.concatenate(rb)
    assert A.dtype == B.dtype == np.float32
    assert np.array_equal(A, B), "ratchet=False diverged from the control"
    assert np.array_equal(a.stagnant_mask(), b.stagnant_mask())
    ma, mb = a.pop_stall_mask(), b.pop_stall_mask()
    assert (ma is None and mb is None) or np.array_equal(ma, mb)


def test_off_allocates_no_record_and_no_column(goal_box):
    rw = _race(goal_box)
    core = FakeCore()
    for d in range(20000, 100, -500):
        _step(rw, _at(core, d))
    assert rw.ratchet is False
    assert rw.ratchet_gap() is None and rw.ratchet_boot() is None
    assert rw._rec is None


# --------------------------------------------------------------------------
# 2. the treatment: the 10 -> 8 -> 12 -> 8 -> 6 example
# --------------------------------------------------------------------------
def test_the_worked_example(goal_box):
    """d 10 -> 8 -> 12 -> 8 -> 6 pays +2, 0, 0, +2; the control pays
    +2, -4, +4, +2 over the same path."""
    core_r, core_c = FakeCore(), FakeCore()
    rat = _race(goal_box, scale=1.0, ratchet=True)
    ctl = _race(goal_box, scale=1.0)
    _step(rat, _at(core_r, 10.0))          # anchor at d = 10
    _step(ctl, _at(core_c, 10.0))
    got_r, got_c = [], []
    for d in (8.0, 12.0, 8.0, 6.0):
        got_r.append(float(_step(rat, _at(core_r, d))[0]))
        got_c.append(float(_step(ctl, _at(core_c, d))[0]))
    assert got_r == pytest.approx([2.0, 0.0, 0.0, 2.0], abs=1e-3), got_r
    assert got_c == pytest.approx([2.0, -4.0, 4.0, 2.0], abs=1e-3), got_c
    # the two collect the SAME total over a path that ends at its own best:
    # a ratchet is the signed term with the refunds and the re-charges
    # deleted, not a different budget
    assert sum(got_r) == pytest.approx(sum(got_c), abs=1e-3)


def test_a_detour_of_any_size_costs_exactly_zero(goal_box):
    """The property the arm exists for, stated without a threshold: the
    cannonball valley (route vertices 1600 -> 1680 raise d 6,632 -> 14,976,
    charged -4.02 by the stock term) costs nothing, and so does every other
    climb anywhere on the map."""
    seg = np.arange(6632.0, 14976.0, 80.0)      # < max_step per tick

    def run(**kw):
        rw = _race(goal_box, **kw)
        core = FakeCore()
        _step(rw, _at(core, 20000.0))
        _step(rw, _at(core, 6632.0))            # a record, deep in
        return sum(float(_step(rw, _at(core, float(d)))[0]) for d in seg)

    assert run(ratchet=True) == pytest.approx(0.0, abs=1e-9)
    # the stock signed term charges the whole climb: 100/198380 per unit
    assert run() == pytest.approx(
        -(seg[-1] - seg[0]) * 100.0 / D0, abs=0.02)


def test_re_gaining_lost_ground_is_never_paid_twice(goal_box):
    """Total shaping over an episode is scale*(d_start - d_best), whatever
    it does in between - so a loop nets 0 and the budget is unchanged."""
    scale = 100.0 / D0
    rw = _race(goal_box, scale=scale, ratchet=True)
    core = FakeCore()
    start = 20000.0
    _step(rw, _at(core, start))
    total = 0.0
    best = start
    for d in (list(np.arange(start, 8000.0, -90.0))
              + list(np.arange(8000.0, 16000.0, 90.0))
              + list(np.arange(16000.0, 5000.0, -90.0))):
        total += float(_step(rw, _at(core, float(d)))[0])
        best = min(best, float(d))
    assert total == pytest.approx(
        scale * (_d_of(rw, start) - _d_of(rw, best)), abs=1e-4)


def test_the_term_is_never_negative(goal_box):
    rw = _race(goal_box, ratchet=True)
    core = FakeCore()
    _step(rw, _at(core, 5000.0))
    rng = np.random.default_rng(7)
    for _ in range(400):
        r = float(_step(rw, _at(core, float(rng.uniform(100.0, 20000.0))))[0])
        assert r >= 0.0, r


# --------------------------------------------------------------------------
# 3. episode boundaries: a reset never generates progress reward
# --------------------------------------------------------------------------
def test_a_reset_mid_buffer_pays_zero_and_restarts_the_record(goal_box):
    rw = _race(goal_box, scale=1.0, ratchet=True)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    _step(rw, _at(core, 8000.0))                      # record 8000
    # the episode ends; the post-autoreset state is the NEW spawn, far out
    r = float(_step(rw, _at(core, 20000.0), done=[1])[0])
    assert r == 0.0, f"the ended row paid {r}"
    assert float(rw.ratchet_gap()[0]) == pytest.approx(0.0, abs=1e-6)
    r = float(_step(rw, _at(core, 19950.0))[0])
    assert r == pytest.approx(50.0, abs=1.0), \
        "the new episode's record did not restart at its own spawn"


def test_a_respawn_deep_inside_neither_pays_nor_is_charged(goal_box):
    """At --respawn-margin 2 the reservoir places starts past the wall. The
    jump itself must pay 0 (it is not progress the policy made) and the
    fresh episode must earn from ITS OWN start."""
    rw = _race(goal_box, scale=1.0, ratchet=True)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    r = float(_step(rw, _at(core, 5000.0), done=[1])[0])   # ends; respawns in
    assert r == 0.0
    assert float(rw.ratchet_gap()[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(_step(rw, _at(core, 4950.0))[0]) == pytest.approx(50.0, abs=1.0)


def test_on_reset_anchors_the_record_at_the_spawn(goal_box):
    rw = _race(goal_box, scale=1.0, ratchet=True)
    core = FakeCore()
    rw.on_reset(_at(core, 5000.0))
    assert float(rw.ratchet_gap()[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(_step(rw, _at(core, 6000.0))[0]) == 0.0      # backing off
    assert float(_step(rw, _at(core, 4950.0))[0]) == pytest.approx(50.0, abs=1.0)


def test_envs_are_independent(goal_box):
    rw = _race(goal_box, scale=1.0, ratchet=True)
    core = FakeCore(3)
    for i in range(3):
        _at(core, 10000.0, i)
    _step(rw, core)
    _at(core, 9940.0, 0)          # a record; all moves < max_step (100u)
    _at(core, 10060.0, 1)         # backing off
    _at(core, 10000.0, 2)         # still exactly at its start
    r = _step(rw, core)
    assert float(r[0]) == pytest.approx(60.0, abs=1.0)
    assert float(r[1]) == 0.0
    assert float(r[2]) == 0.0
    _at(core, 9970.0, 0)          # above its own record again
    _at(core, 10010.0, 1)         # still above its start record
    _at(core, 9980.0, 2)          # a new record
    r = _step(rw, core)
    assert float(r[0]) == 0.0
    assert float(r[1]) == 0.0
    assert float(r[2]) == pytest.approx(20.0, abs=1.0)


# --------------------------------------------------------------------------
# 4. Markov: the observation column IS the record
# --------------------------------------------------------------------------
def test_the_column_is_zero_at_a_record_and_positive_on_a_detour(goal_box):
    rw = _race(goal_box, ratchet=True, ratchet_d0=D0)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    assert float(rw.ratchet_gap()[0]) == 0.0
    for d in (15000.0, 9000.0, 6000.0):          # new records all the way in
        _step(rw, _at(core, d))
        assert float(rw.ratchet_gap()[0]) == pytest.approx(0.0, abs=1e-7), d
    prev = 0.0
    for d in (7000.0, 9000.0, 14000.0):          # backing off: it grows
        _step(rw, _at(core, d))
        g = float(rw.ratchet_gap()[0])
        assert g > prev > -1e-9, (d, g, prev)
        prev = g
    assert prev == pytest.approx((14000.0 - 6000.0) / D0, rel=1e-3)
    _step(rw, _at(core, 6000.0))                 # back ON the record
    assert float(rw.ratchet_gap()[0]) == pytest.approx(0.0, abs=1e-7)


def test_the_column_plus_d_determines_the_next_payment(goal_box):
    """The Markov claim. b_t = d_t - gap_t*d0, so the row the network sees
    at t is enough to say what t+1 pays - which is what makes the value
    function learnable at all."""
    scale = 100.0 / D0
    rw = _race(goal_box, scale=scale, ratchet=True, ratchet_d0=D0)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    rng = np.random.default_rng(3)
    for _ in range(300):
        d_t = float(rw.dist_now()[0])
        gap = float(rw.ratchet_gap()[0])
        b_t = d_t - gap * D0
        nxt = float(rng.uniform(200.0, 20000.0))
        d_n = _d_of(rw, nxt)
        want = scale * min(max(0.0, b_t - min(b_t, d_n)), 100.0)
        got = float(_step(rw, _at(core, nxt))[0])
        assert got == pytest.approx(want, abs=1e-6), (b_t, d_n, got, want)


def test_ratchet_boot_lags_the_record_by_one_call(goal_box):
    """What the truncation bootstrap feeds V(s_T): the autoreset has already
    restarted the live record by the time the terminal row is rebuilt."""
    rw = _race(goal_box, ratchet=True, ratchet_d0=D0)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    d20 = _d_of(rw, 20000.0)
    _step(rw, _at(core, 8000.0))
    d8 = _d_of(rw, 8000.0)
    assert float(rw.ratchet_boot()[0]) == pytest.approx(d20, abs=1e-3)
    _step(rw, _at(core, 12000.0))
    assert float(rw.ratchet_boot()[0]) == pytest.approx(d8, abs=1e-3)
    # a truncation: the live record is already the new far spawn, boot still
    # carries the record the ENDED episode held one call ago
    _step(rw, _at(core, 20000.0), trunc=[1])
    assert float(rw.ratchet_gap()[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(rw.ratchet_boot()[0]) == pytest.approx(d8, abs=1e-3)
    # and that is exactly what the terminal row is built from
    d_T = _d_of(rw, 11000.0)
    got = float(rw.ratchet_gap_of(np.array([d_T]), rw.ratchet_boot())[0])
    assert got == pytest.approx((d_T - d8) / D0, rel=1e-4)
    d_T2 = _d_of(rw, 5000.0)                     # a terminal at a NEW record
    got = float(rw.ratchet_gap_of(np.array([d_T2]), rw.ratchet_boot())[0])
    assert got == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# 5. liveness keeps the RAW d, and the outcome terms are untouched
# --------------------------------------------------------------------------
def test_stall_kill_still_fires_on_a_genuinely_stuck_env(goal_box):
    rw = _race(goal_box, ratchet=True, stall_ticks=100)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    _step(rw, _at(core, 3000.0))
    for _ in range(120):
        _step(rw, _at(core, 3000.0))
    m = rw.pop_stall_mask()
    assert m is not None and m[0] == 1


def test_stall_kill_does_not_fire_while_still_closing(goal_box):
    rw = _race(goal_box, ratchet=True, stall_ticks=100)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    d = 19000.0
    for _ in range(150):
        d = max(20.0, d - 40.0)
        _step(rw, _at(core, d))
        assert rw.pop_stall_mask() is None
    assert not rw.stagnant_mask()[0]


def test_time_penalty_bonus_and_fail_pen_are_untouched(goal_box):
    rw = _race(goal_box, time_pen=0.005, ratchet=True, success_bonus=50.0,
               fail_pen=3.0)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    r = float(_step(rw, _at(core, 20500.0))[0])
    assert r == pytest.approx(-0.005, abs=1e-9)     # backing off costs time
    core.goal_hits[:] = 1
    assert float(_step(rw, _at(core, 10.0), done=[1])[0]) == \
        pytest.approx(50.0, abs=1e-6)
    core.goal_hits[:] = 0
    _step(rw, _at(core, 20000.0))
    assert float(_step(rw, _at(core, 20000.0), done=[1])[0]) == \
        pytest.approx(-3.0, abs=1e-6)


# --------------------------------------------------------------------------
# 6. it is one treatment, not two
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{"d_latch": 6996.0}, {"d_floor": 6996.0},
                                {"ng": 1, "ng_d0": D0, "ng_gamma": 0.9995}])
def test_composing_with_the_other_shaping_arms_is_refused(goal_box, kw):
    with pytest.raises(ValueError):
        _race(goal_box, ratchet=True, **kw)


def test_composing_with_the_arc_is_refused(goal_box):
    class _Arc:
        pass

    with pytest.raises(ValueError):
        _race(goal_box, ratchet=True, arc=_Arc(), arc_scale=1.0)


# --------------------------------------------------------------------------
# 7. the obs column is the LAST one, and a warm resume is identical
# --------------------------------------------------------------------------
LW, LH = 16, 8
IMG = LW * LH


def _policy(route_dim=0, seed=0):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + route_dim + IMG, LW, LH, emb=16, hidden=12,
                  route_dim=route_dim)


def test_a_one_wide_block_warm_resumes_function_identically():
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
    for gap in (torch.zeros(5, 1), torch.rand(5, 1)):
        got_l, got_v = wide(torch.cat([scal, gap, img], 1))
        assert torch.equal(got_l, want_l), \
            "the record column is not zero-padded: step 0 is not the baseline"
        assert torch.equal(got_v, want_v)


def test_the_column_reaches_both_towers():
    p = _policy(1, seed=5)
    torch.nn.init.normal_(p.vf[0].weight[:, -1:], std=3.0)
    torch.nn.init.normal_(p.pi[0].weight[:, -1:], std=3.0)
    scal, img = torch.randn(4, N_SCALAR), torch.randn(4, IMG)
    l0, v0 = p(torch.cat([scal, torch.zeros(4, 1), img], 1))
    l1, v1 = p(torch.cat([scal, torch.ones(4, 1), img], 1))
    assert not torch.equal(v0, v1), "the critic cannot see the record gap"
    assert not torch.equal(l0, l1), "the actor cannot see the record gap"


# --------------------------------------------------------------------------
# 8. the flag surface
# --------------------------------------------------------------------------
def test_the_flag_exists_defaults_off_and_is_recorded():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--race-ratchet"' in src
    assert "args.race_ratchet = False" in src                 # default off
    assert '"race_ratchet": bool(args.race_ratchet)' in src   # saved
    assert 'ck_cfg.get("race_ratchet")' in src                # restored
    assert "ratchet=bool(args.race_ratchet), ratchet_d0=_s.rf_d0" in src
    assert "N_RATCHET = 1 if args.race_ratchet else 0" in src
    assert "RATCHET_COL = N_SCALAR + N_ROUTE - 1" in src      # LAST column
    assert "ratchet_fn=_s.eval_ratchet_feed" in src           # and the evals
    assert "fleet.terminal_ratchet(ti, pos_np)" in src        # and the boot


def test_the_launcher_does_not_bake_the_arm_in():
    sh = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")
    assert "race_ratchet" not in sh and "race-ratchet" not in sh
