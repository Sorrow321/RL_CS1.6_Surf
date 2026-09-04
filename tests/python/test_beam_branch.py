"""tools/beam_tas.py --branch-at: the burst's action override.

The two properties that matter for correctness are (a) the burst touches
ONLY the two heads that steer a surf flight, so a forked lineage is still
the policy's flight with a deviation rather than a random button mash, and
(b) it never writes through the policy wrapper's held action buffer, which
would corrupt the unforked envs' decisions for the rest of the act_every
hold.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from train_fast import H_SIDE, H_YAW, NVEC   # noqa: E402

bt = pytest.importorskip("beam_tas")


def _act(n):
    """A deterministic stand-in for a batch of policy actions."""
    a = np.zeros((n, 6), np.int32)
    for h in range(6):
        a[:, h] = (np.arange(n) + h) % NVEC[h]
    return a


def test_draw_shapes_and_ranges():
    rng = np.random.default_rng(0)
    d0 = bt.branch_draw(rng, 64, 0)
    assert d0.shape == (64, 2) and d0.dtype == np.int32
    assert d0[:, 0].min() >= 0 and d0[:, 0].max() < NVEC[H_YAW]
    assert d0[:, 1].min() >= 0 and d0[:, 1].max() < NVEC[H_SIDE]
    dj = bt.branch_draw(np.random.default_rng(0), 256, 3)
    assert dj[:, 0].min() >= -3 and dj[:, 0].max() <= 3
    assert dj[:, 1].min() >= 0 and dj[:, 1].max() < NVEC[H_SIDE]


def test_draw_is_private_and_reproducible():
    a = bt.branch_draw(np.random.default_rng(7), 32, 2)
    b = bt.branch_draw(np.random.default_rng(7), 32, 2)
    assert np.array_equal(a, b)
    c = bt.branch_draw(np.random.default_rng(8), 32, 2)
    assert not np.array_equal(a, c)


def test_apply_touches_only_yaw_and_side_of_the_forked_envs():
    n, k = 32, 8
    act = _act(n)
    idx = np.arange(n - k, n)
    draw = bt.branch_draw(np.random.default_rng(1), k, 0)
    out = bt.branch_apply(act, idx, draw, 0)
    # unforked envs are untouched, whole row
    assert np.array_equal(out[: n - k], act[: n - k])
    # forked envs keep every head but yaw and side
    for h in range(6):
        if h in (H_YAW, H_SIDE):
            continue
        assert np.array_equal(out[idx, h], act[idx, h])
    assert np.array_equal(out[idx, H_YAW], draw[:, 0])
    assert np.array_equal(out[idx, H_SIDE], draw[:, 1])


def test_apply_does_not_write_through_the_callers_buffer():
    """The caller passes the policy wrapper's HELD action array; writing
    into it would change what the unforked envs do for the rest of the
    act_every hold."""
    act = _act(16)
    before = act.copy()
    bt.branch_apply(act, np.arange(8, 16),
                    bt.branch_draw(np.random.default_rng(2), 8, 0), 0)
    assert np.array_equal(act, before)


def test_jitter_is_an_offset_around_the_mode_and_clips_in_range():
    n, k, J = 64, 64, 5
    act = _act(n)
    idx = np.arange(k)
    draw = bt.branch_draw(np.random.default_rng(3), k, J)
    out = bt.branch_apply(act, idx, draw, J)
    d = out[idx, H_YAW].astype(int) - act[idx, H_YAW].astype(int)
    # every yaw bin moved by at most J, and stayed a legal bin
    assert np.abs(d).max() <= J
    assert out[idx, H_YAW].min() >= 0
    assert out[idx, H_YAW].max() < NVEC[H_YAW]
    # clipping is the only reason a row moves by less than its draw
    exact = np.clip(act[idx, H_YAW] + draw[:, 0], 0, NVEC[H_YAW] - 1)
    assert np.array_equal(out[idx, H_YAW], exact)


def test_jitter_zero_is_an_absolute_bin_not_an_offset():
    act = _act(16)
    idx = np.arange(16)
    draw = bt.branch_draw(np.random.default_rng(4), 16, 0)
    out = bt.branch_apply(act, idx, draw, 0)
    assert np.array_equal(out[idx, H_YAW], draw[:, 0])
