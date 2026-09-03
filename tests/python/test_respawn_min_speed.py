"""--respawn-min-speed: slow states never enter the respawn reservoir."""
import numpy as np

from surfgym.core import STATE_DTYPE
from surfgym.respawn import RespawnBuffer


def _states(n, speed):
    s = np.zeros(n, dtype=STATE_DTYPE)
    s["origin"] = np.random.default_rng(0).normal(size=(n, 3)) * 100.0
    s["velocity"][:, 0] = speed
    return s


def _run(min_speed, speeds, ticks=400):
    """Drive one env for `ticks` ticks at the given per-tick speeds, then
    end its episode; return how many snapshots were harvested."""
    rb = RespawnBuffer(1, reservoir=1000, margin_ticks=10, snap_every=5,
                       min_speed=min_speed)
    ended = np.zeros(1, bool)
    for t in range(ticks):
        rb.observe(_states(1, speeds[t]), ended)
    rb.observe(_states(1, speeds[-1]), np.ones(1, bool))
    return len(rb._out)


def test_off_is_byte_identical():
    speeds = np.full(400, 50.0)
    assert _run(0.0, speeds) == _run(-1.0, speeds) > 0


def test_slow_states_are_never_snapshotted():
    speeds = np.full(400, 50.0)
    assert _run(500.0, speeds) == 0


def test_only_the_fast_half_is_kept():
    speeds = np.where(np.arange(400) < 200, 1200.0, 100.0)
    n_all = _run(0.0, speeds)
    n_fast = _run(500.0, speeds)
    assert 0 < n_fast < n_all
    # the fast half runs to tick 200, the margin trims only the tail
    assert abs(n_fast - n_all / 2) <= 3
