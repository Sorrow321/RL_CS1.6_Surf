"""GoalField's numba fast path must be BIT-IDENTICAL to the numpy reference.

Why this is a bit-identity test and not an allclose test: the goal field is
the shaping potential. Every checkpoint, every ledger number and every
cross-arm comparison in this project assumes one definition of `d`. A fast
path that is merely close would silently fork the reward from every result
already recorded, and the divergence would look like an experimental effect.

The reference is the numpy body of GoalField.sample; the fast path is
selected by goalfield._FAST_SAMPLE, so setting it to None inside a test
gives the reference on the same object.
"""
import numpy as np
import pytest

import surfgym.goalfield as gf
from surfgym.goalfield import GoalField


def _field(seed=0, shape=(24, 30, 28), cell=32.0):
    rng = np.random.default_rng(seed)
    reach = 5000.0
    grid = rng.uniform(0.0, reach, shape).astype(np.float32)
    # a solid/unreachable region, so the honest-corner masking is exercised
    grid[shape[0] // 3:shape[0] // 2] = reach + 2.0 * cell
    mins = np.array([-400.0, -300.0, -200.0], np.float64)
    return GoalField(grid, mins, cell, reach)


def _both(f, pos):
    fast = f.sample(pos)
    saved, gf._FAST_SAMPLE = gf._FAST_SAMPLE, None
    try:
        ref = f.sample(pos)
    finally:
        gf._FAST_SAMPLE = saved
    return ref, fast


@pytest.mark.skipif(gf._FAST_SAMPLE is None, reason="numba not installed")
def test_bit_identical_on_random_points():
    f = _field()
    rng = np.random.default_rng(1)
    hi = f.mins + np.array(f.grid.shape[::-1]) * f.cell
    pos = rng.uniform(f.mins, hi, (20000, 3))
    ref, fast = _both(f, pos)
    assert np.array_equal(ref, fast)


@pytest.mark.skipif(gf._FAST_SAMPLE is None, reason="numba not installed")
def test_bit_identical_outside_the_grid():
    """Out-of-bounds points clamp to the edge; the clamp must match."""
    f = _field()
    hi = f.mins + np.array(f.grid.shape[::-1]) * f.cell
    pos = np.array([
        f.mins - 1e4, hi + 1e4, f.mins, hi,
        [f.mins[0] - 1.0, hi[1] + 1.0, 0.5 * (f.mins[2] + hi[2])],
    ], np.float64)
    ref, fast = _both(f, pos)
    assert np.array_equal(ref, fast)


@pytest.mark.skipif(gf._FAST_SAMPLE is None, reason="numba not installed")
def test_bit_identical_inside_the_unreachable_region():
    """Where no corner is honest the result is the sentinel, exactly."""
    f = _field()
    s = f.grid.shape
    zc = (s[0] // 3 + s[0] // 2) // 2
    pos = np.stack([
        np.full(64, f.mins[0] + 10 * f.cell),
        np.full(64, f.mins[1] + 10 * f.cell),
        np.full(64, f.mins[2] + zc * f.cell),
    ], axis=1)
    ref, fast = _both(f, pos)
    assert np.array_equal(ref, fast)
    assert np.all(fast == np.float32(f.sentinel))


@pytest.mark.skipif(gf._FAST_SAMPLE is None, reason="numba not installed")
def test_single_point_and_shape():
    f = _field()
    p = np.array([[0.0, 0.0, 0.0]])
    ref, fast = _both(f, p)
    assert fast.shape == (1,) and np.array_equal(ref, fast)
    assert fast.dtype == np.float32
