"""--arc-death-charge: a death forfeits the arc banked on the CURRENT goal line.

Why it exists (ledger 2026-09-27 12:05): the executor's stock objective pays signed arc progress
along its line inside a corridor, and a death keeps whatever was banked. On unitfarmer2's shaft
loop an inside cut ran the arc projection 34% ahead of the line while still inside the corridor,
and the mover died under the block keeping it - the reward pays a fatal corner cut. The charge
takes kappa x the current line's bank (clamped >= 0) back at a death; a finish keeps it, a
truncation is exempt, and kappa 0 is the control byte for byte.
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

    def at(self, xs):
        o = np.zeros((self.num_envs, 3), np.float32)
        o[:, 0] = np.asarray(xs, np.float32)
        self.states_view["origin"][:] = o
        return self


def _line(n=40):
    p = np.zeros((n, 3), np.float64)
    p[:, 0] = np.arange(n) * SP
    return p


def _setup(n=1, kappa=0.0, **kw):
    arc = MultiArcProgress(n, l_max=64, spacing=SP, corridor=384.0, window=16)
    arc.set_lines(np.arange(n), [_line()] * n)
    core = _FakeCore(n).at([0.0] * n)
    rr = RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=0.01, time_pen=0.0,
                    stall_ticks=10 ** 9, arc_death_charge=kappa, **kw)
    rr.on_reset(core)
    return arc, core, rr


def _step(rr, core, done=0, trunc=0, goal=0):
    n = core.num_envs
    d = np.full(n, done, np.uint8)
    t = np.full(n, trunc, np.uint8)
    core.goal_hits[:] = goal
    return rr(None, None, None, np.zeros(n, np.float32), d, t, core)


def _ride(rr, core, xs):
    return [float(_step(rr, core.at([x]))[0]) for x in xs]


def test_death_forfeits_the_current_lines_bank():
    arc, core, rr = _setup(kappa=1.0)
    paid = _ride(rr, core, [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0])
    assert sum(paid) == pytest.approx(0.01 * 300.0, abs=1e-4)
    # the death tick: the core has already autoreset, `pos` is the next spawn (x 0)
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-0.01 * 300.0, abs=1e-4)
    assert sum(paid) + r == pytest.approx(0.0, abs=1e-4)


def test_kappa_scales_the_charge():
    arc, core, rr = _setup(kappa=0.5)
    _ride(rr, core, [100.0, 200.0])
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-0.5 * 0.01 * 200.0, abs=1e-4)


def test_a_finish_keeps_its_bank():
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    r = float(_step(rr, core.at([0.0]), done=1, goal=1)[0])
    assert r == pytest.approx(rr.success_bonus, abs=1e-4)


def test_truncation_is_exempt():
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    r = float(_step(rr, core.at([0.0]), trunc=1)[0])
    assert r == pytest.approx(0.0, abs=1e-6)


def test_only_the_current_line_is_forfeited():
    """A new line (a new plan) moves the bank origin: what earlier lines paid is kept."""
    arc, core, rr = _setup(kappa=1.0)
    _ride(rr, core, [100.0, 200.0, 300.0])
    # the next plan starts under the player at x 300: install it anchored there
    arc.set_lines(np.array([0]), [_line()], origin=np.array([[300.0, 0.0, 0.0]]))
    assert arc.arc0[0] == pytest.approx(300.0, abs=1e-3)
    _ride(rr, core, [350.0, 400.0])
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(-0.01 * 100.0, abs=1e-4)


def test_dying_after_going_backwards_is_not_paid():
    arc, core, rr = _setup(kappa=1.0)
    core.at([1000.0])
    arc.reset(core.states_view["origin"])         # a spawn part-way along the line
    rr.on_reset(core)
    _ride(rr, core, [950.0, 900.0, 850.0])
    r = float(_step(rr, core.at([0.0]), done=1)[0])
    assert r == pytest.approx(0.0, abs=1e-6)


def test_kappa_zero_is_the_control_byte_for_byte():
    xs = [30.0, 90.0, 60.0, 200.0, 180.0, 400.0]
    outs = []
    for kw in ({}, {"arc_death_charge": 0.0}):
        arc = MultiArcProgress(2, l_max=64, spacing=SP, corridor=384.0, window=16)
        arc.set_lines(np.arange(2), [_line()] * 2)
        core = _FakeCore(2).at([0.0, 0.0])
        rr = RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=0.01, time_pen=0.005,
                        stall_ticks=10 ** 9, **kw)
        rr.on_reset(core)
        seq = []
        for k, x in enumerate(xs):
            done = np.array([k == 3, k == 5], np.uint8)
            core.goal_hits[:] = 0
            core.at([x, x + 10.0])
            seq.append(rr(None, None, None, np.zeros(2, np.float32), done,
                          np.zeros(2, np.uint8), core).copy())
        outs.append(np.stack(seq))
    assert np.array_equal(outs[0], outs[1])


def test_refusals():
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc_death_charge=1.0)          # no arc
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc=ArcProgress(_line(), SP),  # no per-plan origin
                   arc_scale=0.01, arc_death_charge=1.0)
    arc = MultiArcProgress(1, l_max=64, spacing=SP)
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=0.01, arc_death_charge=-1.0)
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, arc=arc, arc_scale=0.01, arc_death_charge=1.0,
                   death_charge=0.5, ng_d0=1000.0)


def test_arc0_is_bookkeeping_only():
    """arc0 moves only where the anchor is re-set; advance never touches it."""
    arc = MultiArcProgress(1, l_max=64, spacing=SP, corridor=384.0)
    arc.set_lines(np.array([0]), [_line()])
    assert arc.arc0[0] == 0.0
    arc.advance(np.array([[200.0, 0.0, 0.0]], np.float32))
    assert arc.arc0[0] == 0.0 and arc.arc[0] == pytest.approx(200.0, abs=1e-3)
    arc.reset(np.array([[640.0, 0.0, 0.0]]))
    assert arc.arc0[0] == pytest.approx(640.0, abs=1e-3)
