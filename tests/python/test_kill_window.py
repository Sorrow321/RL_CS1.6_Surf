"""Kill-volume broadphase (Codex 2026-09-27): a kill_zones entry's AABB is MODEL-LOCAL, so its
world box is AABB + the entity's origin, and the HULL-1 containment test accepts standing-player
origins up to 16 u laterally / 36 u vertically outside the brush, so the raster window must be
grown by those extents - otherwise an origin-offset trigger is masked near the world origin and
a thin horizontal kill sheet can have no 32 u cell centre inside its raw box at all."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import kill_window   # noqa: E402
from surfgym.zones import STAND_HALF, kill_world_box   # noqa: E402


def test_world_box_applies_the_origin_and_the_hull():
    k = {"mins": [-8.0, -8.0, -2.0], "maxs": [8.0, 8.0, 2.0], "origin": [1000.0, 2000.0, 300.0]}
    lo, hi = kill_world_box(k)
    assert np.allclose(lo, [1000 - 8 - 16, 2000 - 8 - 16, 300 - 2 - 36])
    assert np.allclose(hi, [1000 + 8 + 16, 2000 + 8 + 16, 300 + 2 + 36])
    lo0, hi0 = kill_world_box({"mins": k["mins"], "maxs": k["maxs"]}, pad=(0, 0, 0))
    assert np.allclose(lo0, k["mins"]) and np.allclose(hi0, k["maxs"])   # no origin key: local


def test_window_of_an_origin_offset_volume_sits_on_the_volume():
    k = {"mins": [-8.0, -8.0, -2.0], "maxs": [8.0, 8.0, 2.0], "origin": [1000.0, 2000.0, 300.0]}
    mins, cell, shape = np.array([0.0, 0.0, 0.0]), 32.0, (40, 100, 60)
    _lo, _hi, pts = kill_window(k, mins, cell, shape)
    assert pts is not None
    c = pts.mean(0)
    assert np.allclose(c, [1000, 2000, 300], atol=cell)      # not near the world origin


def test_a_thin_sheet_between_cell_centres_still_gets_centres():
    # a 4 u thick sheet whose top is 12 u below one row of 32 u centres and whose bottom is 16 u
    # above the row below it: the raw brush box contains no centre, the hull-grown box does
    mins, cell = np.array([0.0, 0.0, 0.0]), 32.0
    z0 = 16.0 + 32.0 * 4 + 16.0            # between the centres at z = 144 and z = 176
    k = {"mins": [64.0, 64.0, z0 - 2.0], "maxs": [512.0, 512.0, z0 + 2.0],
         "origin": [0.0, 0.0, 0.0]}
    zs_raw = [zc for zc in 16.0 + 32.0 * np.arange(20) if z0 - 2.0 <= zc <= z0 + 2.0]
    assert zs_raw == []                    # the k1 window (raw box + 1 u) had no centre row
    _lo, _hi, pts = kill_window(k, mins, cell, (20, 20, 20))
    zs = np.unique(pts[:, 2])
    assert (np.abs(zs - z0) <= 2.0 + STAND_HALF[2]).any()
