"""--race-field-blur: surfgym.goalfield.blur_goal_field.

The blur is masked to the field's honest (reachable free) voxels: sentinel
cells (walls, unreachable pockets) never change and never leak into the
average; sigma 0 is the identity object; values stay inside [0, reach_max];
and a flat honest region stays flat (the masked normalisation, not a
zero-padded blur that would drag values near walls toward 0).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import GoalField, blur_goal_field  # noqa: E402


def _field():
    n = 12
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    grid = (x * 100.0).astype(np.float32)          # a ramp along x
    reach_max = 2000.0
    sentinel = reach_max + 2.0 * 32.0
    grid[:, 5:7, :] = sentinel                       # a wall slab in y
    grid[:, :, 9] = sentinel                         # a wall column in x
    return GoalField(grid, [0.0, 0.0, 0.0], 32.0, reach_max), sentinel


def test_sigma_zero_is_identity():
    gf, _ = _field()
    assert blur_goal_field(gf, 0.0) is gf


def test_sentinel_cells_untouched_and_values_bounded():
    gf, sentinel = _field()
    b = blur_goal_field(gf, 1.5)
    assert b is not gf
    wall = gf.grid >= gf._valid_max
    assert np.array_equal(b.grid[wall], gf.grid[wall])
    assert np.all(b.grid[~wall] >= 0.0) and np.all(b.grid[~wall] <= gf.reach_max)
    assert b.cell == gf.cell and b.reach_max == gf.reach_max


def test_masked_normalisation_keeps_a_flat_region_flat():
    gf, sentinel = _field()
    gf.grid[...] = np.where(gf.grid >= gf._valid_max, sentinel, 700.0).astype(np.float32)
    b = blur_goal_field(gf, 2.0)
    honest = b.grid < b._valid_max
    assert np.allclose(b.grid[honest], 700.0, atol=1e-3)


def test_blur_smooths_the_ramp_away_from_walls():
    gf, _ = _field()
    b = blur_goal_field(gf, 2.0)
    # interior honest cells: the blurred ramp is still monotone in x and
    # closer to its neighbours' mean than the raw ramp is to its own
    row = b.grid[6, 2, :9]
    assert np.all(np.diff(row) >= -1e-3)
    assert abs(float(row[4]) - 400.0) < 60.0
