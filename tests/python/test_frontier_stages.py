"""tools/frontier_stages.py: the frontier is the deepest list position the policy's own episodes
touched; its states are cut before that touch and the one before it, each stage balanced."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import frontier_stages as fs  # noqa: E402


def test_the_frontier_is_the_deepest_touch_and_the_one_before_it():
    touches = [[[10, 19], [40, 17]], [[12, 19], [41, 17], [80, 18]], []]
    assert fs.frontier_stages(touches) == (2, [1, 2])
    assert fs.frontier_stages([[[5, 19]]]) == (0, [0])
    assert fs.frontier_stages([[], []]) == (None, [])


def test_balance_resamples_every_stage_to_the_largest():
    st = np.arange(7)
    ks = np.array([3, 3, 3, 3, 3, 4, 4])
    out, k = fs.balance(st, ks, np.random.default_rng(0))
    assert (k == 3).sum() == 5 and (k == 4).sum() == 5
    assert set(out[k == 4]) <= {5, 6} and sorted(out[k == 3]) == [0, 1, 2, 3, 4]


def test_vel_scale_multiplies_velocity_and_one_one_is_exact():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from surfgym.core import STATE_DTYPE
    st = np.zeros(50, STATE_DTYPE)
    st["velocity"] = np.array([100.0, -200.0, 50.0], np.float32)
    rng = np.random.default_rng(0)
    assert fs.jitter_speed(st, 1.0, 1.0, rng) is st
    out = fs.jitter_speed(st, 1.0, 1.3, rng)
    f = out["velocity"][:, 0] / 100.0
    assert np.all((f >= 1.0) & (f <= 1.3)) and f.std() > 0.01
    assert np.allclose(out["velocity"][:, 1], -200.0 * f) and np.allclose(st["velocity"][:, 0], 100.0)
