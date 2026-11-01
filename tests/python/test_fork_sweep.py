"""tools/beam_tas.py --fork-sweep: the per-env fork along a saved line (zlr2b4, 2026-11-01).

What must hold for the search to be what it says:
(a) an env whose fork tick is still ahead flies EXACTLY the line's engine row and view, and at a
    decision tick its held keys become that row's fwd / side / duck (prefix_sync's rule), so its
    first own decision reads the state the line put it in;
(b) after its fork (outside the burst) an env keeps the policy's own row and view, untouched;
(c) inside the burst only the view's yaw column moves, the keys stay the policy's own;
(d) the wrapper's held buffers are never written through;
(e) the burst draws are private and reproducible, and redrawn every `hold` ticks from the fork.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.keyshold import HELD_HEADS  # noqa: E402

bt = pytest.importorskip("beam_tas")

K = 4


class _Keys:
    def __init__(self, n):
        self.state = np.zeros((n, 3), np.int64)
        self.boot = np.zeros((n, 3), np.int64)


def _setup(n=6):
    line_acts = np.zeros((50, 6), np.int32)
    for d in range(50):
        line_acts[d] = [7, 3, (d % 3), (d + 1) % 3, d % 2, (d + 2) % 2]
    line_view = np.stack([np.arange(50, dtype=np.float32) * 0.5,
                          -np.arange(50, dtype=np.float32) * 0.1], axis=1)
    pol_a = np.full((n, 6), 1, np.int32)
    pol_v = np.full((n, 2), 3.0, np.float32)
    fork_tick = np.array([8, 8, 16, 16, 24, 24], np.int64)[:n]
    return line_acts, line_view, pol_a, pol_v, fork_tick


def test_before_the_fork_the_line_and_its_keys():
    line_acts, line_view, pol_a, pol_v, ft = _setup()
    keys = _Keys(len(ft))
    off = np.zeros(len(ft))
    a0, v0 = pol_a.copy(), pol_v.copy()
    t = 12                                    # envs 2..5 still on the line, 0..1 forked
    a, v = bt.fork_sweep_apply(t, K, pol_a, pol_v, ft, line_acts, line_view, off,
                               np.random.default_rng(0), 0, 72, 0.0, "velocity", keys=keys)
    rep = t < ft
    assert np.array_equal(a[rep], np.repeat(line_acts[t // K][None], rep.sum(), 0))
    assert np.allclose(v[rep], line_view[t // K][None])
    # forked envs keep the policy's own row and view (no burst)
    assert np.array_equal(a[~rep], pol_a[~rep]) and np.allclose(v[~rep], pol_v[~rep])
    # held keys follow the line's row for the replaying envs only
    assert np.array_equal(keys.state[rep], np.repeat(
        line_acts[t // K][list(HELD_HEADS)][None], rep.sum(), 0))
    assert not keys.state[~rep].any()
    assert np.array_equal(keys.boot, keys.state)
    # (d) the wrapper's buffers were not written through
    assert np.array_equal(pol_a, a0) and np.array_equal(pol_v, v0)


def test_keys_only_synced_on_a_decision_tick():
    line_acts, line_view, pol_a, pol_v, ft = _setup()
    keys = _Keys(len(ft))
    bt.fork_sweep_apply(13, K, pol_a, pol_v, ft, line_acts, line_view, np.zeros(len(ft)),
                        np.random.default_rng(0), 0, 72, 0.0, "velocity", keys=keys)
    assert not keys.state.any()


def test_burst_moves_only_the_view_yaw_and_redraws_every_hold():
    line_acts, line_view, pol_a, pol_v, ft = _setup()
    off = np.zeros(len(ft))
    rng = np.random.default_rng(3)
    burst, hold = 16, 8
    seen = {}
    for t in range(8, 32):
        a, v = bt.fork_sweep_apply(t, K, pol_a, pol_v, ft, line_acts, line_view, off, rng,
                                   burst, hold, 30.0, "velocity", keys=None)
        inb = (t >= ft) & (t < ft + burst)
        after = t >= ft + burst
        # keys: the policy's own for every forked env, burst or not
        assert np.array_equal(a[t >= ft], pol_a[t >= ft])
        # pitch column never moves; yaw moves only inside the burst, by |off| <= 30
        assert np.allclose(v[t >= ft, 1], pol_v[t >= ft, 1])
        assert np.allclose(v[after, 0], pol_v[after, 0])
        if inb.any():
            dy = (v[inb, 0] - pol_v[inb, 0] + 180.0) % 360.0 - 180.0
            assert np.all(np.abs(dy) <= 30.0 + 1e-4)
            for j in np.flatnonzero(inb):
                seen.setdefault((j, (t - ft[j]) // hold), set()).add(round(float(off[j]), 6))
    # one draw per hold window per env: constant inside a window
    assert all(len(s) == 1 for s in seen.values())
    # and different windows of one env got different draws
    env0 = [next(iter(s)) for (j, w), s in sorted(seen.items()) if j == 0]
    assert len(env0) == 2 and env0[0] != env0[1]


def test_burst_draws_are_reproducible():
    line_acts, line_view, pol_a, pol_v, ft = _setup()
    outs = []
    for _ in range(2):
        off = np.zeros(len(ft))
        rng = np.random.default_rng(11)
        a, v = bt.fork_sweep_apply(24, K, pol_a, pol_v, ft, line_acts, line_view, off, rng,
                                   16, 8, 0.0, "velocity")
        outs.append(v.copy())
    assert np.allclose(outs[0], outs[1])


def test_nothing_to_do_returns_the_inputs():
    line_acts, line_view, pol_a, pol_v, ft = _setup()
    a, v = bt.fork_sweep_apply(100, K, pol_a, pol_v, ft, line_acts, line_view,
                               np.zeros(len(ft)), np.random.default_rng(0), 16, 8, 30.0,
                               "velocity")
    assert a is pol_a and v is pol_v
