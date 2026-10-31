"""The finish is the TRIGGER, not its bounding box (2026-10-31, autoresearch uf2r2b1).

GoldSrc fires a brush trigger only when the player's origin is inside the trigger model's
player-expanded clip hull (SV_TouchLinks); the core scored a finish on the trigger's AABB. The finish
of surf_unitfarmer2 (trigger_multiple *23 -> counter_stop_button) is three 1 u curtains - the west
face x 3328 (y <= -2144), a diagonal (3328, -2144)-(3584, -1888), the north face y -1888 (x >= 3584)
- so its box has an empty north-west corner, and every policy and search line of the lineage
"finished" by skimming it (at about (3,305, -1,880, -590)).

(a) SurfCore.set_goal_box arms the trigger whose model bounds ARE the box (*23 here) and the C core
    then scores a crossing only inside its clip hull: the skim of the empty corner is no finish, a
    crossing of the real west curtain is one; SURF_GOAL_AABB=1 restores the box alone (the skim
    finishes again, as before).
(b) finish_trigger_model: an unmatched box (a hand-written / button box) arms nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

MAP = ROOT / "maps" / "surf_unitfarmer2.bsp"
FIN = ((3328.0, -2592.0, -640.0), (3936.0, -1888.0, 96.0))

pytestmark = pytest.mark.skipif(not MAP.exists(), reason="needs maps/surf_unitfarmer2.bsp")


def _core(aabb_only=False):
    from surfgym.core import SurfCore, default_config
    old = os.environ.get("SURF_GOAL_AABB")
    if aabb_only:
        os.environ["SURF_GOAL_AABB"] = "1"
    else:
        os.environ.pop("SURF_GOAL_AABB", None)
    try:
        core = SurfCore(str(MAP), default_config(num_envs=1, max_episode_ticks=1000))
        core.reset(0)
        core.set_goal_box(*FIN)
    finally:
        if old is None:
            os.environ.pop("SURF_GOAL_AABB", None)
        else:
            os.environ["SURF_GOAL_AABB"] = old
    return core


def _fly(core, origin, velocity, ticks=6):
    """Put env 0 at origin with velocity (airborne) and step with neutral actions; -> whether the
    goal was hit on any tick (the episode ends on it)."""
    st = core.get_states()[0].copy()
    st["origin"] = np.asarray(origin, np.float32)
    st["velocity"] = np.asarray(velocity, np.float32)
    st["onground"] = -1
    core.set_state(0, st)
    from surfgym.core import ACTION_NVEC
    act = np.zeros((1, len(ACTION_NVEC)), np.int32)
    act[0, 0] = ACTION_NVEC[0] // 2          # neutral yaw bin
    act[0, 1] = 3                           # neutral pitch bin
    act[0, 2] = 1                           # no forward key
    act[0, 3] = 1                           # no side key
    for _ in range(ticks):
        _o, _r, done, _t, _ = core.step(act)
        if bool(core.goal_hits[0]):
            return True
        if bool(np.asarray(done)[0]):
            return False
    return False


def test_the_finish_trigger_is_armed_from_the_box():
    core = _core()
    assert core.goal_model == 23


def test_the_empty_corner_skim_is_no_finish_but_was_one_on_the_box():
    # the policies' "finish": into the hull-inflated box's north-west corner, moving east
    skim = ((3302.0, -1880.0, -590.0), (1400.0, 0.0, 0.0))
    assert _fly(_core(aabb_only=True), *skim) is True          # the old test: a finish
    assert _fly(_core(), *skim) is False                       # the trigger: none


def test_a_crossing_of_the_real_west_curtain_finishes_under_both_tests():
    cross = ((3300.0, -2300.0, -590.0), (1400.0, 0.0, 0.0))
    assert _fly(_core(aabb_only=True), *cross) is True
    assert _fly(_core(), *cross) is True


def test_an_unmatched_box_arms_no_trigger():
    from surfgym.core import finish_trigger_model
    m, _o = finish_trigger_model(str(MAP), (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    assert m is None
    m, o = finish_trigger_model(str(MAP), *FIN)
    assert m == 23 and o == [0.0, 0.0, 0.0]
