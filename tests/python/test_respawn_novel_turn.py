"""--respawn-novel-turn T: a novel row's horizontal velocity and view yaw are
turned together by U(-T, T) degrees (surfgym.respawn.NovelSpawnSampler).

* T = 0 (the default) is the sampler that shipped, row for row, byte for byte;
* T > 0 gives exactly the T = 0 rows' positions (the turn has its own
  generator), keeps every row's horizontal speed, vertical speed, pitch and
  duck state, and turns the view yaw by the same angle as the heading, so the
  view's offset from the velocity is unchanged;
* T outside [0, 180] is refused, and the recorder lists the flag as
  training-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.goalfield import GoalField                            # noqa: E402
from surfgym.respawn import NovelSpawnSampler, RespawnBuffer       # noqa: E402

SID = ROOT / "maps" / "surf_src_sidistic.bsp"
SID_FIELD = ROOT / "maps" / "surf_src_sidistic.goal_32.npz"
needs_sid = pytest.mark.skipif(not (SID.is_file() and SID_FIELD.is_file()),
                               reason="needs surf_src_sidistic.bsp and its goal_32 field")


@pytest.fixture(scope="module")
def field():
    z = np.load(SID_FIELD, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


@pytest.fixture(scope="module")
def core():
    from surfgym import SurfCore, default_config
    return SurfCore(str(SID), default_config(
        num_envs=4, spawn_mode=2, max_episode_ticks=200,
        sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))


@pytest.fixture(scope="module")
def reservoir(core, field):
    """3,000 states at reachable, hull-clear points, each with its own velocity."""
    rng = np.random.default_rng(0)
    nz, ny, nx = field.grid.shape
    lim = field.reach_max - 0.5 * field.cell
    pts = []
    while len(pts) < 3000:
        ix, iy, iz = rng.integers(0, nx, 20000), rng.integers(0, ny, 20000), rng.integers(0, nz, 20000)
        ok = field.grid[iz, iy, ix] < lim
        for q in np.asarray(field.mins) + (np.stack([ix, iy, iz], 1)[ok] + 0.5) * field.cell:
            if not core.trace(q, q, hull=0).startsolid:
                pts.append(q)
            if len(pts) >= 3000:
                break
    rows = np.zeros(3000, STATE_DTYPE)
    rows["origin"] = np.asarray(pts, np.float32)
    rows["velocity"] = rng.normal(0.0, 900.0, (3000, 3)).astype(np.float32)
    rows["yaw"] = rng.uniform(0.0, 360.0, 3000).astype(np.float32)
    rows["pitch"] = rng.uniform(-30.0, 30.0, 3000).astype(np.float32)
    rows["ducked"] = rng.integers(0, 2, 3000).astype(np.int32)
    res = RespawnBuffer(4, reservoir=8000, map_id="surf_src_sidistic")
    res.push_many(rows)
    return res


def _rows(core, field, reservoir, **kw):
    s = NovelSpawnSampler(core, field, 4, seed=5, **kw)
    store = reservoir._store[:reservoir.size]
    s.d_best = float(np.quantile(field.sample(np.asarray(store["origin"], np.float64)), 0.2))
    return s.sample_states(400, reservoir)


@needs_sid
def test_turn_zero_is_the_shipped_sampler_byte_for_byte(core, field, reservoir):
    a = _rows(core, field, reservoir)
    b = _rows(core, field, reservoir, turn=0.0)
    assert len(a) > 100
    assert a.tobytes() == b.tobytes()


@needs_sid
def test_a_turn_changes_only_the_heading_and_the_yaw_with_it(core, field, reservoir):
    a = _rows(core, field, reservoir)
    b = _rows(core, field, reservoir, turn=180.0)
    assert len(a) == len(b) > 100
    # the same rows in the same places: only velocity x/y and the yaw may differ
    for k in STATE_DTYPE.names:
        if k in ("velocity", "yaw"):
            continue
        assert a[k].tobytes() == b[k].tobytes(), k
    va, vb = a["velocity"].astype(np.float64), b["velocity"].astype(np.float64)
    assert np.array_equal(va[:, 2], vb[:, 2]), "the vertical speed moved"
    np.testing.assert_allclose(np.hypot(vb[:, 0], vb[:, 1]), np.hypot(va[:, 0], va[:, 1]),
                               rtol=1e-5, atol=1e-3)
    dh = np.degrees(np.arctan2(vb[:, 1], vb[:, 0]) - np.arctan2(va[:, 1], va[:, 0]))
    dy = b["yaw"].astype(np.float64) - a["yaw"].astype(np.float64)
    wrap = lambda x: (x + 180.0) % 360.0 - 180.0                   # noqa: E731
    np.testing.assert_allclose(wrap(dh - dy), 0.0, atol=1e-2)      # view offset kept
    assert np.abs(wrap(dh)).max() > 90.0, "a 180-degree turn never turned far"
    assert ((b["yaw"] >= 0.0) & (b["yaw"] < 360.0)).all()


@needs_sid
def test_a_small_turn_stays_inside_its_bound(core, field, reservoir):
    a = _rows(core, field, reservoir)
    b = _rows(core, field, reservoir, turn=20.0)
    dh = np.degrees(np.arctan2(b["velocity"][:, 1], b["velocity"][:, 0])
                    - np.arctan2(a["velocity"][:, 1], a["velocity"][:, 0]))
    dh = (dh + 180.0) % 360.0 - 180.0
    assert np.abs(dh).max() <= 20.0 + 1e-3
    assert np.abs(dh).max() > 5.0


@pytest.mark.parametrize("bad", [-1.0, 181.0])
def test_a_turn_outside_0_180_is_refused(bad):
    class _F:
        grid = np.zeros((2, 2, 2), np.float32)
        mins = (0.0, 0.0, 0.0)
        cell = 32.0
        reach_max = 1e9
    with pytest.raises(ValueError):
        NovelSpawnSampler(None, _F(), 4, turn=bad)


def test_the_recorder_treats_the_flag_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"respawn_novel_turn"' in src
