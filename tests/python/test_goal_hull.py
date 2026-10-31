"""--goal-hull / surf_set_goal_model (uf2r2b4, 2026-10-31): the finish fires on the trigger
model's CLIP HULL at the tick's end position (the engine's SV_TouchLinks rule), not on the
model's AABB.

surf_unitfarmer2's finish (*23) is three thin curtains along its box's west / diagonal / north
sides: the box's north-west corner is a finish for the swept-AABB goal and nothing for the game.
Without the flag the core is unchanged (the swept box), which the first test pins."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
BSP = REPO / "maps" / "surf_unitfarmer2.bsp"
ZONES = REPO / "maps" / "surf_unitfarmer2.zones.json"

pytestmark = pytest.mark.skipif(not (BSP.exists() and ZONES.exists()),
                                reason="surf_unitfarmer2 map files not present")


def _core(goal_model):
    from surfgym import SurfCore, default_config
    z = json.loads(ZONES.read_text())
    core = SurfCore(str(BSP), default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=12000, water_fail=1,
        sv_maxvelocity=3500.0, lidar_w=0, lidar_h=0))
    core.set_goal_box(z["end"]["mins"], z["end"]["maxs"])
    if goal_model is not None:
        core.set_goal_model(goal_model)
    core.reset(0)
    return core


def _fly(core, origin, velocity, ticks=6):
    st = core.get_states()
    st["origin"] = origin
    st["velocity"] = velocity
    st["onground"] = -1
    core.set_state(0, st[0])
    act = np.zeros((1, 6), np.int32)
    act[0, 0], act[0, 1] = 7, 3
    act[0, 2] = act[0, 3] = 1
    for t in range(ticks):
        core.step(act)
        if core.goal_hits[0]:
            return t
    return None


def test_the_model_lookup_finds_the_finish_trigger():
    from surfgym.zones import goal_trigger_model
    z = json.loads(ZONES.read_text())
    assert goal_trigger_model(str(BSP), z["end"]) == 23
    assert goal_trigger_model(str(BSP), {"mins": [0, 0, 0], "maxs": [1, 1, 1]}) == -1


def test_box_goal_counts_the_empty_corner_but_the_hull_goal_does_not():
    corner = [3320.0, -1880.0, -560.0]          # inside the box's NW corner, off every curtain
    drift = [100.0, -100.0, 0.0]                 # slow, toward the diagonal but far from it
    assert _fly(_core(None), corner, drift) is not None, \
        "the swept-AABB goal (default) must still fire in the box corner"
    assert _fly(_core(23), corner, drift) is None, \
        "the clip-hull goal must NOT fire in the empty corner"


def test_hull_goal_fires_on_the_curtains():
    # the diagonal curtain (3328,-2144)-(3584,-1888): fly into it from the north-west
    t = _fly(_core(23), [3440.0, -1990.0, -400.0], [600.0, -600.0, 0.0], ticks=20)
    assert t is not None, "crossing the diagonal curtain must finish"
    # the west face x 3328 (y <= -2144): fly east into it
    t = _fly(_core(23), [3290.0, -2300.0, -400.0], [900.0, 0.0, 0.0], ticks=20)
    assert t is not None, "crossing the west face must finish"


def test_an_invalid_model_falls_back_to_the_box():
    assert _fly(_core(-1), [3320.0, -1880.0, -560.0], [100.0, -100.0, 0.0]) is not None
    assert _fly(_core(10 ** 6), [3320.0, -1880.0, -560.0], [100.0, -100.0, 0.0]) is not None
