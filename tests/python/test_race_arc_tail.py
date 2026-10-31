"""--race-arc-tail U (zlr1b2, 2026-10-31): with --race-arc, an episode whose arc anchor reaches the
route's last U units switches the shaping field's ratchet on for the rest of the episode.

A straight route along +x from x 0 to x 1,024 (8 x 128 u); the field d = 10,000 - y (progress is
+y, off the route). Before the tail no field progress is paid; inside it every new record of d is
paid at the field scale, a regain is not paid twice, the episode end re-arms it, and U = 0 is the
arc term alone op for op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE             # noqa: E402
from surfgym.rewards import RaceReward           # noqa: E402
from surfgym.route import ArcProgress            # noqa: E402

SCALE, ASCALE = 0.01, 0.001


class _YField:
    reach_max = 1e9
    cell = 32.0

    def sample(self, pos):
        p = np.atleast_2d(np.asarray(pos, np.float64))
        return 10_000.0 - p[:, 1]


class _Core:
    def __init__(self, n=1):
        from types import SimpleNamespace
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=800.0))

    def at(self, x, y):
        self.states_view["origin"][:, 0] = np.float32(x)
        self.states_view["origin"][:, 1] = np.float32(y)
        return self


def _rr(tail):
    route = np.stack([np.arange(9) * 128.0, np.zeros(9), np.zeros(9)], axis=1)
    arc = ArcProgress(route, spacing=128.0, corridor=600.0, window=16)
    core = _Core().at(0.0, 0.0)
    rr = RaceReward(_YField(), scale=SCALE, arc=arc, arc_scale=ASCALE, time_pen=0.0,
                    stall_ticks=10 ** 9, max_step=1e6)
    rr.arc_tail = tail
    rr.on_reset(core)
    return rr, core


def _step(rr, core, done=0):
    return float(rr(None, None, None, np.zeros(1, np.float32), np.full(1, done, np.uint8),
                    np.zeros(1, np.uint8), core)[0])


def test_no_field_pay_before_the_tail_then_records_inside_it():
    rr, core = _rr(200.0)
    # along the route to x 700 with y rising: only the arc pays (field records ignored)
    r1 = _step(rr, core.at(350.0, 100.0))
    r2 = _step(rr, core.at(700.0, 200.0))
    assert r1 == pytest.approx(350.0 * ASCALE, abs=1e-5)
    assert r2 == pytest.approx(350.0 * ASCALE, abs=1e-5)
    # x 900 is inside the last 200 u: the tail switches on with the record at d(900, 250)
    r3 = _step(rr, core.at(900.0, 250.0))
    assert r3 == pytest.approx(200.0 * ASCALE, abs=1e-5)
    # off the route's end, higher y: the field pays the record's improvement
    r4 = _step(rr, core.at(1000.0, 500.0))
    assert r4 == pytest.approx(100.0 * ASCALE + 250.0 * SCALE, abs=1e-5)
    # back down and up to the same height: no second payment, then a new record pays
    assert _step(rr, core.at(1000.0, 300.0)) == pytest.approx(0.0, abs=1e-5)
    assert _step(rr, core.at(1000.0, 500.0)) == pytest.approx(0.0, abs=1e-5)
    assert _step(rr, core.at(1000.0, 600.0)) == pytest.approx(100.0 * SCALE, abs=1e-5)
    st = rr.pop_stats()
    assert rr.arc_tail_paid == 0.0


def test_the_episode_end_rearms_it():
    rr, core = _rr(200.0)
    _step(rr, core.at(900.0, 0.0))
    _step(rr, core.at(1000.0, 400.0))
    _step(rr, core.at(0.0, 0.0), done=1)                     # next spawn at the route's start
    assert not bool(rr._tail_on[0])
    # field progress on the route's first half is not paid in the new episode
    r = _step(rr, core.at(300.0, 500.0))          # inside the 600 u corridor
    assert r == pytest.approx(300.0 * ASCALE, abs=1e-5)


def test_zero_is_the_arc_term_alone():
    a, ca = _rr(0.0)
    b, cb = _rr(200.0)
    pts = [(350.0, 100.0), (700.0, 200.0), (800.0, 220.0)]   # before b's tail: identical
    for x, y in pts:
        assert _step(a, ca.at(x, y)) == _step(b, cb.at(x, y))
    assert a._tail_on is None and a._tail_rec is None


def test_the_recorder_lists_it_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt as rc
    assert "race_arc_tail" in rc.TRAIN_ONLY
