"""--stall-arc (Codex, 2026-09-27): under --race-arc the stall detector's progress coordinate is the
route arc, not the field distance.

The defect it exists for: on a detour map (unitfarmer2) the route first goes AWAY from the finish
for ~15 s, so a detector watching the Euclidean / geodesic field never sees a new record and
kills every episode that follows the route. These tests pin (a) the control path: a flat field
never re-arms, whatever the arc does; (b) --stall-arc re-arms on a new arc record of more than
stall_eps in one call and not on standing still; (c) an ended row re-arms from the NEW episode's
own arc; (d) the flag is refused without an arc.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from surfgym.core import STATE_DTYPE            # noqa: E402
from surfgym.rewards import RaceReward          # noqa: E402
from surfgym.route import ArcProgress           # noqa: E402

SPACING = 128.0
STALL = 20          # calls


class _FlatField:
    def sample(self, pos):
        return np.full(len(np.atleast_2d(pos)), 5000.0)


class _FakeCore:
    def __init__(self, n=1):
        from types import SimpleNamespace
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        # RaceReward.on_reset reads the gravity for its --surf-bonus tracker
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=800.0))

    def at(self, x):
        self.states_view["origin"][:, 0] = np.asarray(x, np.float32)
        return self


def _line():
    p = np.zeros((300, 3), np.float64)
    p[:, 0] = np.arange(300) * SPACING
    return p


def _rr(core, stall_arc):
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=STALL, stall_eps=32.0,
                    arc=ArcProgress(_line(), SPACING, corridor=1500.0), arc_scale=1.0,
                    stall_arc=stall_arc)
    rr.on_reset(core)
    return rr


def _step(rr, core, done=0):
    n = core.num_envs
    d = np.full(n, done, np.uint8)
    return rr(None, None, None, np.zeros(n, np.float32), d, np.zeros(n, np.uint8), core)


def _run(stall_arc, per_call, calls):
    core = _FakeCore().at(0.0)
    rr = _rr(core, stall_arc)
    kills = 0
    for i in range(calls):
        core.at((i + 1) * per_call)
        _step(rr, core)
        if rr.pop_stall_mask() is not None:
            kills += 1
    return kills


def test_the_control_path_never_rearms_on_a_flat_field():
    # the field distance never improves: 40 u of route per call is invisible to it
    assert _run(False, 40.0, 5 * STALL) == 5


def test_stall_arc_rearms_on_route_progress_above_stall_eps():
    assert _run(True, 40.0, 5 * STALL) == 0


def test_stall_arc_still_kills_standing_still_and_creeping():
    assert _run(True, 0.0, 5 * STALL) == 5
    # a record of 10 u per call never beats the running best by 32 u in ONE call: the per-call
    # rule the field detector has (CLAUDE.md), kept unchanged
    assert _run(True, 10.0, 5 * STALL) == 5


def test_an_ended_row_rearms_from_its_new_episodes_own_arc():
    core = _FakeCore().at(0.0)
    rr = _rr(core, True)
    for i in range(10):
        core.at(3000.0 + 0.0 * i)                  # far down the route, then still
        _step(rr, core)
    core.at(0.0)                                   # a respawn at the route's start
    _step(rr, core, done=1)
    assert rr._best[0] == pytest.approx(-float(rr.arc.arc[0]))
    assert rr._since[0] == 0
    core.at(40.0)
    _step(rr, core)
    assert rr._since[0] == 0                       # a new record from the NEW spawn's arc


def test_stall_arc_needs_an_arc():
    with pytest.raises(ValueError):
        RaceReward(_FlatField(), scale=1.0, stall_arc=True)
