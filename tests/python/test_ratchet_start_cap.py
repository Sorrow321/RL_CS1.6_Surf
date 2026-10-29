"""``--race-ratchet-start-cap``: an episode's ratchet record STARTS at min(its spawn's d, the
map start's d) instead of its spawn's own d (pbfr1b1, 2026-10-29).

Why: a backward curriculum spawns the agent on states BEHIND the map's start (on
surf_princessburglar_b26 the way out runs against the shaping field for ~6 s). With the record
at the spawn's own d, coming back toward the start pays ~2.5 reward units for ground the
agent never chose to give up - the lure that kept the curriculum turning straight back with too
little energy. Capped at the start's d, that return pays nothing, only progress past the start
pays, and the record-gap observation column reads what a real episode from the start would read.

    python -m pytest tests/python/test_ratchet_start_cap.py -q
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
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
    def __init__(self, n: int = 1, bounds=((-3e4, -3e4, -3e4), (3e4, 3e4, 3e4))):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        # RaceReward.on_reset reads the physics (gravity, maxvel) for its energy / surf terms
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=800.0, sv_maxvelocity=3500.0))
        self._bounds = (np.asarray(bounds[0], np.float32), np.asarray(bounds[1], np.float32))

    def map_bounds(self):
        return self._bounds


def _step(rw, core, done=None):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    done = z if done is None else np.asarray(done, np.uint8)
    obs = np.zeros((n, 15), np.float32)
    return rw(obs, obs, obs, np.zeros(n, np.float32), done, z, core)


@pytest.fixture
def goal_box():
    if ZONES.exists():
        return json.loads(ZONES.read_text(encoding="utf-8"))["end"]
    return {"mins": [-14720.0, 7487.0, -1824.0], "maxs": [-8064.0, 7488.0, -352.0]}


def _race(goal_box, cap=0.0, **kw):
    kw.setdefault("scale", 1.0)
    kw.setdefault("time_pen", 0.0)
    kw.setdefault("success_bonus", 50.0)
    rw = RaceReward(EuclidField(goal_box), ratchet=True, **kw)
    rw.ratchet_cap = float(cap)
    return rw


def _at(core, d, i=0):
    p = core.states_view["origin"].copy()
    p[i] = (-11000.0, 7488.0 + float(d), -1000.0)
    core.states_view["origin"] = p
    return core


def _d(rw, d):
    return float(rw.field.sample(_at(FakeCore(), d).states_view["origin"])[0])


def test_off_is_the_control_bit_for_bit(goal_box):
    a, b = _race(goal_box, time_pen=0.003), _race(goal_box, cap=0.0, time_pen=0.003)
    b.ratchet_cap = 0.0
    ca, cb = FakeCore(), FakeCore()
    path = list(range(20000, 9000, -90)) + list(range(9000, 15000, 70)) + list(range(15000, 3000, -95))
    ends = {40, 130, 170}
    ra, rb, ga, gb = [], [], [], []
    for k, dd in enumerate(path):
        done = [1] if k in ends else None
        ra.append(_step(a, _at(ca, dd), done=done).copy())
        rb.append(_step(b, _at(cb, dd), done=done).copy())
        ga.append(a.ratchet_gap().copy())
        gb.append(b.ratchet_gap().copy())
    assert np.array_equal(np.stack(ra), np.stack(rb))
    assert np.array_equal(np.stack(ga), np.stack(gb))


def test_a_spawn_behind_the_start_is_not_paid_for_coming_back(goal_box):
    start = 10000.0
    rw_off, rw_on = _race(goal_box), _race(goal_box, cap=0.0)
    rw_on.ratchet_cap = _d(rw_on, start)
    c_off, c_on = FakeCore(), FakeCore()
    rw_off.on_reset(_at(c_off, 13000.0))           # spawned 3,000 u behind the start
    rw_on.on_reset(_at(c_on, 13000.0))
    # the capped record starts at the start's d: the gap reads 3,000 u behind it
    assert float(rw_on.ratchet_gap()[0]) * rw_on.ratchet_d0 == pytest.approx(3000.0, abs=1.0)
    assert float(rw_off.ratchet_gap()[0]) == pytest.approx(0.0, abs=1e-9)
    pay_off = pay_on = 0.0
    for dd in range(12950, 9999, -50):             # back to the start: paid only without the cap
        pay_off += float(_step(rw_off, _at(c_off, dd))[0])
        pay_on += float(_step(rw_on, _at(c_on, dd))[0])
    assert pay_off == pytest.approx(3000.0, abs=5.0)
    assert pay_on == pytest.approx(0.0, abs=1e-6)
    for dd in range(9950, 8000, -50):              # past the start: both pay the same
        a = float(_step(rw_off, _at(c_off, dd))[0])
        b = float(_step(rw_on, _at(c_on, dd))[0])
        assert a == pytest.approx(b, abs=1e-6)
        assert a == pytest.approx(50.0, abs=1.0)


def test_a_spawn_past_the_start_is_unchanged(goal_box):
    start = 10000.0
    rw_off, rw_on = _race(goal_box), _race(goal_box)
    rw_on.ratchet_cap = _d(rw_on, start)
    c_off, c_on = FakeCore(), FakeCore()
    rw_off.on_reset(_at(c_off, 6000.0))
    rw_on.on_reset(_at(c_on, 6000.0))
    for dd in list(range(6100, 7000, 100)) + list(range(7000, 3000, -60)):
        assert float(_step(rw_off, _at(c_off, dd))[0]) == float(_step(rw_on, _at(c_on, dd))[0])
        assert float(rw_off.ratchet_gap()[0]) == float(rw_on.ratchet_gap()[0])


def test_an_autoreset_behind_the_start_restarts_at_the_cap(goal_box):
    start = 10000.0
    rw = _race(goal_box)
    rw.ratchet_cap = _d(rw, start)
    core = FakeCore()
    rw.on_reset(_at(core, 9000.0))
    _step(rw, _at(core, 8000.0))                   # a record past the start
    r = float(_step(rw, _at(core, 14000.0), done=[1])[0])   # ends; respawns behind the start
    assert r == 0.0
    assert float(rw.ratchet_gap()[0]) * rw.ratchet_d0 == pytest.approx(4000.0, abs=1.0)
    assert float(_step(rw, _at(core, 13950.0))[0]) == 0.0    # coming back pays nothing
    assert rw._rec0[0] == pytest.approx(rw.ratchet_cap)


def test_the_trainer_and_the_tools_mirror_it():
    """The flag exists in the trainer (off by default, dumped only when set) and the two tools
    that rebuild the record-gap column (record_ckpt, beam_tas) mention the config key."""
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--race-ratchet-start-cap"' in src
    assert 'meta["config"]["race_ratchet_start_cap"] = 1' in src
    for tool in ("record_ckpt.py", "beam_tas.py"):
        assert '"race_ratchet_start_cap"' in (ROOT / "tools" / tool).read_text(encoding="utf-8")
