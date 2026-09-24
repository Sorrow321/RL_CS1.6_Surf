"""map_spawn_pool moves a spawn whose standing hull starts inside solid (dead on arrival: the core
fails a player trapped in solid for 5 ticks, src/env.c) to the nearest clear height within
SPAWN_UNSTICK_U, up before down, and drops it only when none is clear. blue200's four
front-row spawns (y = -1,248) sit 4 u too low; everything on cannonball is clear."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from surfgym.core import SurfCore, default_config   # noqa: E402
from surfgym.rewards import map_spawn_pool          # noqa: E402

POOL = Path(os.environ.get("SURF_TEST_POOL") or (ROOT / "maps_pool"))


def _core(p):
    if not p.exists():
        pytest.skip(f"{p} not present")
    return SurfCore(str(p), default_config(num_envs=1, spawn_mode=1, lidar_w=0, lidar_h=0))


def test_embedded_spawns_are_lifted_clear():
    core = _core(POOL / "surf_edgeflow_blue200.bsp")
    raw = np.asarray([o for o, _ in core.spawns()], np.float64)
    pool = map_spawn_pool(core)["origin"].astype(np.float64)
    assert len(raw) == 16 and len(pool) == 16
    moved = ~np.all(np.isclose(raw, pool), axis=1)
    assert np.all(np.isclose(raw[moved, 1], -1248.0)) and moved.sum() == 4
    assert np.allclose(pool[moved, 2] - raw[moved, 2], 4.0)          # 4 u up, nothing else
    assert np.allclose(pool[moved, :2], raw[moved, :2])
    for o in pool:                                                   # all clear now
        assert not core.trace(o, o).startsolid
    yaw = np.arange(16, dtype=np.float32)
    assert np.array_equal(map_spawn_pool(core, yaw=yaw)["yaw"], yaw)
    core.close()


def test_a_spawn_with_no_clear_height_is_dropped():
    core = _core(ROOT / "maps" / "surf_src_sidistic.bsp")
    raw = list(core.spawns())
    pool = map_spawn_pool(core)["origin"].astype(np.float64)
    assert len(raw) == 2 and len(pool) == 1
    assert not core.trace(pool[0], pool[0]).startsolid
    core.close()


def test_clear_map_is_unchanged():
    core = _core(ROOT / "maps" / "surf_src_cannonball.bsp")
    raw = np.asarray([o for o, _ in core.spawns()], np.float64)
    pool = map_spawn_pool(core)["origin"].astype(np.float64)
    assert np.array_equal(pool, raw.astype(pool.dtype))
    core.close()
