"""goals._rdp_fast (compiled Douglas-Peucker, the planner's line simplification) keeps exactly the
vertices goals._rdp (the numpy reference) keeps: random 3-D polylines, lattice staircases like a
planner's grid path (distances exactly at the one-cell eps), degenerate and doubling-back lines."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from surfgym.goals import _rdp, _rdp_fast   # noqa: E402


def _same(p, eps):
    a, b = _rdp(p, eps), _rdp_fast(p, eps)
    assert a.shape == b.shape and np.array_equal(a, b), (p, eps)


def test_matches_the_reference():
    rng = np.random.default_rng(0)
    for _ in range(300):                      # random flights
        p = np.cumsum(rng.normal(0.0, 50.0, (int(rng.integers(2, 120)), 3)), axis=0)
        _same(p, float(rng.choice([8.0, 32.0, 128.0, 512.0])))
    for _ in range(300):                      # 26-neighbourhood lattice walks at 32 u (ties at eps)
        steps = rng.integers(-1, 2, (int(rng.integers(2, 300)), 3)) * np.array([1, 1, 0.3]).round()
        p = np.cumsum(steps, axis=0) * 32.0 + np.array([16.0, 16.0, 24.0])
        _same(p, 32.0)
    _same(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 1.0)
    _same(np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [-300.0, 0.0, 0.0], [50.0, 0.0, 0.0]]), 32.0)
    _same(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), 32.0)
