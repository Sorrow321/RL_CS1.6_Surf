"""Contact telemetry (surf_get_touch, 2026-09-27): the planes the player's movement actually hit on
the last tick, recorded as a side effect that must not change the physics. The ramp-command
search (tools/edge_archive.py --moves ramp) reads it as its collision truth."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import SurfCore, default_config   # noqa: E402

MAP = ROOT / "maps_pool" / "surf_edgeflow_blue025.bsp"


def _core(n=16):
    core = SurfCore(str(MAP), default_config(num_envs=n, max_episode_ticks=100000))
    if getattr(core._lib, "surf_get_touch", None) is None:
        pytest.skip("this surfcore build predates surf_get_touch")
    return core


def _run(read_touch, ticks=300, n=16, seed=0):
    core = _core(n)
    core.reset(seed)
    rng = np.random.default_rng(seed)
    h = hashlib.sha256()
    touched = 0
    for _ in range(ticks):
        acts = np.stack([rng.integers(0, 15, n), rng.integers(0, 7, n), rng.integers(0, 3, n),
                         rng.integers(0, 3, n), rng.integers(0, 2, n), np.zeros(n, int)],
                        1).astype(np.int32)
        core.step(acts)
        h.update(core.states_view.tobytes())
        if read_touch:
            c, nrm, pts = core.get_touch()
            touched += int(c.sum())
    return h.hexdigest(), touched


@pytest.mark.skipif(not MAP.exists(), reason="edgeflow map not present")
def test_reading_touches_does_not_change_the_physics():
    a, _ = _run(read_touch=False)
    b, touched = _run(read_touch=True)
    assert a == b
    assert touched > 0


@pytest.mark.skipif(not MAP.exists(), reason="edgeflow map not present")
def test_standing_on_the_start_floor_records_the_ground_plane():
    core = _core(1)
    core.reset(0)
    neutral = np.array([[7, 3, 1, 1, 0, 0]], np.int32)
    for _ in range(30):                       # settle onto the start floor
        core.step(neutral)
    c, nrm, pts = core.get_touch()
    assert int(c[0]) >= 1
    assert float(nrm[0, 0, 2]) >= 0.7         # a floor (the engine's ground test)
    # the recorded point is the player origin at the contact
    assert np.allclose(pts[0, 0], core.states_view["origin"][0], atol=2.0)


@pytest.mark.skipif(not MAP.exists(), reason="edgeflow map not present")
def test_free_flight_records_nothing():
    core = _core(1)
    core.reset(0)
    st = core.get_states()[0].copy()
    st["origin"][2] += 2000.0                 # far above everything
    st["velocity"][:] = (100.0, 0.0, 0.0)
    st["onground"] = -1
    core.set_state(0, st)
    core.step(np.array([[7, 3, 1, 1, 0, 0]], np.int32))
    c, _n, _p = core.get_touch()
    assert int(c[0]) == 0
