"""--arc-death-charge: a current-open-line FACE-VALUE DEATH BOND on the executor's arc reward.

Why it exists (ledger 2026-09-27 12:05): the executor's stock objective pays signed arc progress
along its line inside a corridor, and a death keeps whatever was paid. On unitfarmer2's shaft
loop an inside cut ran the arc projection 34% ahead of the line while still inside the corridor,
then the mover died. The bond takes kappa x the post-clip arc credit the reward actually PAID on
the line open at the death back (clamped >= 0); a finish keeps it, a truncation is exempt, every
previously closed line keeps its credit, and kappa 0 is the control byte for byte.

Codex's review (bus 10:33Z) caught the first version charging the endpoint displacement
arc - arc0 instead: after a backward tracker jump clipped to -100 it would have left ~916 of
~1,506 paid units uncharged on the real loop trace. Tests (a)-(h) below pin the paid-bank rule.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE            # noqa: E402
from surfgym.goalarc import MultiArcProgress    # noqa: E402
from surfgym.rewards import RaceReward          # noqa: E402
from surfgym.route import ArcProgress           # noqa: E402

SP = 128.0
SCALE = 0.01                # arc_scale in the tests: 1 u of paid arc = 0.01 reward


class _FlatField:
    def sample(self, pos):
        return np.zeros(len(np.atleast_2d(pos)))


class _FakeCore:
    def __init__(self, n=1):
        from types import SimpleNamespace
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=800.0))

    def at(self, xs, ys=None):
        o = np.zeros((self.num_envs, 3), np.float32)
        o[:, 0] = np.asarray(xs, np.float32)
        if ys is not None:
            o[:, 1] = np.asarray(ys, np.float32)
        self.states_view["origin"][:] = o
        return self


def _line(n=60):
    p = np.zeros((n, 3), np.float64)
    p[:, 0] = np.arange(n) * SP
    return p


def _setup(n=1, kappa=0.0, **kw):
    arc = MultiArcProgress(n, l_max=64, spacing=SP, corridor=384.0, window=16)
    arc.set_lines(np.arange(n), [_line()] * n)
    core = _FakeCore(n).at([0.0] * n)
    rr = RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=SCALE, time_pen=0.0,
                    stall_ticks=10 ** 9, arc_death_charge=kappa, **kw)
    rr.on_reset(core)
    return arc, core, rr


def _step(rr, core, done=0, trunc=0, goal=0):
    n = core.num_envs
    core.goal_hits[:] = goal
    return rr(None, None, None, np.zeros(n, np.float32), np.full(n, done, np.uint8),
              np.full(n, trunc, np.uint8), core)


def _ride(rr, core, xs):
    return [float(_step(rr, core.at([x]))[0]) for x in xs]


# ------------------------------------------------------------------ the bond itself
def test_death_forfeits_the_paid_bank():
    arc, core, rr = _setup(kappa=1.0)
    paid = _ride(rr, core, [30.0 * k for k in range(1, 11)])
    assert sum(paid) == pytest.approx(SCALE * 300.0, abs=1e-4)
    assert arc.bank[0] == pytest.approx(300.0, abs=1e-3)
    r = float(_step(rr, core.at([0.0]), done=1)[0])      # the core has autoreset: pos = spawn
    assert r == pytest.approx(-SCALE * 300.0, abs=1e-4)
    assert sum(paid) + r == pytest.approx(0.0, abs=1e-4)


def test_kappa_scales_the_charge():
    arc, core, rr = _setup(kappa=0.5)
    _ride(rr, core, [100.0, 200.0])
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-0.5 * SCALE * 200.0, abs=1e-4)


def test_a_clipped_backward_jump_charges_the_paid_bank_not_the_endpoint():
    """(a) 15 x +100, then a raw -900 clipped to -100: the reward PAID 1,400, the endpoint
    displacement is 600 - charge the 1,400."""
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0 * k for k in range(1, 16)])
    r_jump = float(_step(rr, core.at([600.0]))[0])
    assert r_jump == pytest.approx(-SCALE * 100.0, abs=1e-4)   # the clip
    assert arc.arc[0] == pytest.approx(600.0, abs=1e-3)        # the endpoint
    assert arc.bank[0] == pytest.approx(1400.0, abs=1e-3)      # what was paid
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-SCALE * 1400.0, abs=1e-3)


def test_a_clipped_forward_jump_charges_only_what_it_paid():
    """(b) a raw +512 projection jump clipped to +100 pays 100 and is charged 100."""
    arc, core, rr = _setup(kappa=1.0)
    r_jump = float(_step(rr, core.at([512.0]))[0])
    assert r_jump == pytest.approx(SCALE * 100.0, abs=1e-4)
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-SCALE * 100.0, abs=1e-4)


def test_a_new_line_with_the_same_zero_origin_clears_the_bank():
    """(c) set_lines anchored at arc 0 again (origin=None) still starts a new bank."""
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    arc.set_lines(np.array([0]), [_line()])
    assert arc.bank[0] == 0.0
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(0.0, abs=1e-6)


def test_masked_resets_clear_only_the_touched_rows():
    """(d)"""
    arc, core, rr = _setup(n=2, kappa=1.0)
    for x in (100.0, 200.0, 300.0):
        _step(rr, core.at([x, x]))
    assert np.allclose(arc.bank, [300.0, 300.0])
    arc.reset(core.states_view["origin"], mask=np.array([True, False]))
    assert np.allclose(arc.bank, [0.0, 300.0])
    arc.set_lines(np.array([1]), [_line()], origin=core.states_view["origin"])
    assert np.allclose(arc.bank, [0.0, 0.0])


def test_the_autoreset_spawn_cannot_enter_the_dying_bank():
    """(e) the death tick's position is the NEXT episode's spawn: its projected delta (here
    +700 raw, +100 after the clip) must not reach the bank before the charge."""
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    r = float(_step(rr, core.at([1000.0]), done=1)[0])
    assert r == pytest.approx(-SCALE * 300.0, abs=1e-4)


@pytest.mark.parametrize("how", ["finish", "trunc"])
def test_finish_and_truncation_pay_no_charge_and_the_next_episode_starts_clean(how):
    """(f)"""
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    kw = {"done": 1, "goal": 1} if how == "finish" else {"trunc": 1}
    r = float(_step(rr, core.at([0.0]), **kw)[0])
    assert r == pytest.approx(rr.success_bonus if how == "finish" else 0.0, abs=1e-4)
    assert arc.bank[0] == 0.0                       # the reward re-anchored the ended row
    r2 = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r2 == pytest.approx(0.0, abs=1e-6)


def test_corridor_exit_freezes_the_bank_and_reentry_resumes_it():
    """(g) outside the corridor the arc is frozen and pays 0, so the bank does not move."""
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    _step(rr, core.at([350.0], [600.0]))            # 600 u off the line: outside 384
    assert arc.bank[0] == pytest.approx(300.0, abs=1e-3)
    _step(rr, core.at([400.0], [0.0]))              # back on the line: +100 from the frozen 300
    assert arc.bank[0] == pytest.approx(400.0, abs=1e-3)
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-SCALE * 400.0, abs=1e-4)


def test_flag_omitted_and_zero_are_identical_rewards_and_tracker_state():
    """(h) kappa 0 / absent: the control byte for byte - rewards AND the tracker's (arc, idx)."""
    xs = [30.0, 90.0, 60.0, 700.0, 180.0, 400.0, 20.0]
    outs, states = [], []
    for kw in ({}, {"arc_death_charge": 0.0}):
        arc = MultiArcProgress(2, l_max=64, spacing=SP, corridor=384.0, window=16)
        arc.set_lines(np.arange(2), [_line()] * 2)
        core = _FakeCore(2).at([0.0, 0.0])
        rr = RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=SCALE, time_pen=0.005,
                        stall_ticks=10 ** 9, **kw)
        rr.on_reset(core)
        seq, st = [], []
        for k, x in enumerate(xs):
            done = np.array([k == 3, k == 5], np.uint8)
            core.goal_hits[:] = 0
            core.at([x, x + 10.0])
            seq.append(rr(None, None, None, np.zeros(2, np.float32), done,
                          np.zeros(2, np.uint8), core).copy())
            st.append(np.concatenate([arc.arc.copy(), arc.idx.astype(np.float64)]))
        outs.append(np.stack(seq))
        states.append(np.stack(st))
    assert np.array_equal(outs[0], outs[1])
    assert np.array_equal(states[0], states[1])


# ------------------------------------------------------------------ the rest of the contract
def test_only_the_open_line_is_forfeited():
    """A new line (a new plan) starts a new bank: what closed lines paid is kept."""
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    arc.set_lines(np.array([0]), [_line()], origin=np.array([[300.0, 0.0, 0.0]]))
    assert arc.bank[0] == 0.0 and arc.arc0[0] == pytest.approx(300.0, abs=1e-3)
    _ride(rr, core, [350.0, 400.0])
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-SCALE * 100.0, abs=1e-4)


def test_dying_after_going_backwards_is_not_paid():
    arc, core, rr = _setup(kappa=1.0)
    core.at([1000.0])
    rr.on_reset(core)                               # re-anchors at 1,000 and clears the bank
    _ride(rr, core, [950.0, 900.0, 850.0])
    assert arc.bank[0] == pytest.approx(-150.0, abs=1e-3)
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(0.0, abs=1e-6)


def test_refusals():
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc_death_charge=1.0)          # no arc
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc=ArcProgress(_line(), SP),  # no per-line bank
                   arc_scale=SCALE, arc_death_charge=1.0)
    arc = MultiArcProgress(1, l_max=64, spacing=SP)
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=SCALE, arc_death_charge=-1.0)
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=SCALE, arc_death_charge=1.0,
                   death_charge=0.5, ng_d0=1000.0)
    for bad in (float("nan"), float("inf")):                              # non-finite kappa
        with pytest.raises(ValueError):
            RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=SCALE, arc_death_charge=bad)
    with pytest.raises(ValueError):                                       # curiosity-scaled pay
        RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=SCALE, arc_death_charge=1.0,
                   cc_tmax=1.0)


def test_advance_never_touches_the_bookkeeping():
    """arc0 and bank move only where the anchor is re-set; the tracker's advance (numba or the
    numpy reference) never touches either."""
    arc = MultiArcProgress(1, l_max=64, spacing=SP, corridor=384.0)
    arc.set_lines(np.array([0]), [_line()])
    arc.bank[0] = 123.0
    arc.advance(np.array([[200.0, 0.0, 0.0]], np.float32))
    assert arc.arc0[0] == 0.0 and arc.bank[0] == 123.0
    assert arc.arc[0] == pytest.approx(200.0, abs=1e-3)
    arc._advance_np(np.array([[300.0, 0.0, 0.0]], np.float32))
    assert arc.bank[0] == 123.0
    arc.reset(np.array([[640.0, 0.0, 0.0]]))
    assert arc.arc0[0] == pytest.approx(640.0, abs=1e-3) and arc.bank[0] == 0.0
