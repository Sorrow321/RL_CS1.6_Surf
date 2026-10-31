"""A finish of SEVERAL brush triggers (operator, 2026-10-31): surf_za_loop's five trigger_multiples
*15-*19 all target the stop button. zones.detect_zones gives their union box as "end" plus the
boxes and models; SurfCore.set_goal_box arms the models (surf_set_goal_models), and the step
completes on ANY of them - never in the union's open corners, which are two thirds of the box.
A single-trigger map's zones carry no "models", and nothing changes for it."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
LOOP = ROOT / "maps" / "surf_za_loop.bsp"
ZONES = ROOT / "maps" / "surf_za_loop.zones.json"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else ROOT / "build" / "libsurfcore.so")
needs_loop = pytest.mark.skipif(not (LOOP.exists() and ZONES.exists() and DLL.exists()),
                                reason="needs the built core + surf_za_loop")

IN_16 = (-200.0, 2000.0, 3050.0)          # inside finish slab *16 (x -320..-64, y 1312..3360)
CORNER = (1500.0, 300.0, 3100.0)          # inside the union box, in no trigger: open air
RECORD_ENTRY = (-48.2, 2014.1, 3047.6)    # the WR demo's first frame inside a finish (*16)


def _probe(pos, vel, ticks=4, aabb=False):
    from surfgym.core import SurfCore, default_config
    old = os.environ.get("SURF_GOAL_AABB")
    os.environ["SURF_GOAL_AABB"] = "1" if aabb else "0"
    try:
        z = json.loads(ZONES.read_text())
        core = SurfCore(str(LOOP), default_config(num_envs=1, lidar_w=0, lidar_h=0),
                        dll_path=str(DLL))
        if getattr(core._lib, "surf_set_goal_models", None) is None:
            pytest.skip("this surfcore build predates surf_set_goal_models")
        core.set_goal_box(z["end"]["mins"], z["end"]["maxs"])
        n = core.goal_nmodels
        core.reset(0)
        st = core.get_states()[0].copy()
        st["origin"][:] = pos
        st["velocity"][:] = vel
        st["onground"] = -1
        core.set_state(0, st)
        a = np.zeros((1, 6), np.int32)
        a[0, 0], a[0, 1] = 7, 3
        hits = []
        for _ in range(ticks):
            core.step(a)
            hits.append(int(core.goal_hits[0]))
        return n, hits
    finally:
        if old is None:
            os.environ.pop("SURF_GOAL_AABB", None)
        else:
            os.environ["SURF_GOAL_AABB"] = old


@needs_loop
def test_the_zone_lists_the_five_finish_triggers():
    z = json.loads(ZONES.read_text())
    end = z["end"]
    assert end["models"] == [15, 16, 17, 18, 19] and len(end["boxes"]) == 5
    for k in range(3):
        assert end["mins"][k] == min(b["mins"][k] for b in end["boxes"])
        assert end["maxs"][k] == max(b["maxs"][k] for b in end["boxes"])
    assert z["spawns"][0]["yaw"] == 180.0


@needs_loop
def test_any_trigger_finishes_and_the_union_corner_does_not():
    n, hits = _probe(IN_16, (300.0, 0.0, 0.0))
    assert n == 5 and hits[0] == 1, (n, hits)
    n, hits = _probe(CORNER, (300.0, 0.0, 0.0))
    assert n == 5 and not any(hits), (n, hits)
    n, hits = _probe(RECORD_ENTRY, (300.0, -200.0, 50.0))
    assert hits[0] == 1, hits


@needs_loop
def test_the_aabb_opt_out_is_the_union_box():
    n, hits = _probe(CORNER, (300.0, 0.0, 0.0), aabb=True)
    assert n == 0 and hits[0] == 1, (n, hits)


@needs_loop
def test_the_grader_set_agrees():
    from surfgym.goalhull import for_zone
    z = json.loads(ZONES.read_text())
    g = for_zone(LOOP, z["end"])
    assert g.model == [15, 16, 17, 18, 19]
    assert g.contains(IN_16) and g.contains(RECORD_ENTRY) and not g.contains(CORNER)
    # a recorded episode whose last rows stop one tick short of slab *16 (moving +y into it)
    rows = np.array([[0, -200.0, 1280.0 - 16.0 - 30.0, 3050.0, 0.0, 3000.0, 0.0],
                     [1, -200.0, 1280.0 - 16.0 - 7.0, 3050.0, 0.0, 3000.0, 0.0]])
    assert g.finished(rows, 0.0077)
    far = rows.copy()
    far[:, 1] = 1500.0
    far[:, 2] = 300.0
    assert not g.finished(far, 0.0077)


@needs_loop
def test_bad_model_lists_are_refused_and_empty_clears():
    from surfgym.core import SurfCore, default_config
    z = json.loads(ZONES.read_text())
    core = SurfCore(str(LOOP), default_config(num_envs=1, lidar_w=0, lidar_h=0), dll_path=str(DLL))
    core.set_goal_box(z["end"]["mins"], z["end"]["maxs"])
    for bad in ([0], [10 ** 6], list(range(1, 40))):
        with pytest.raises(ValueError):
            core.set_goal_models(bad)
    assert core.goal_nmodels == 0
    assert core.set_goal_models([15, 16]) == 2 and core.goal_nmodels == 2
    assert core.set_goal_models([]) == 0 and core.goal_nmodels == 0


def test_a_single_trigger_map_has_no_model_list():
    cb = ROOT / "maps" / "surf_src_cannonball.bsp"
    if not cb.exists():
        pytest.skip("surf_src_cannonball not present")
    from surfgym.zones import detect_zones
    z = detect_zones(str(cb))
    assert z["end"] and "models" not in z["end"] and "boxes" not in z["end"]
