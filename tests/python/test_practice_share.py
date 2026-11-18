"""--practice-share (mm/s7d): the practice mask is floor(n x share) envs of each map, spread
evenly by index, and the same mask for the same n every time (the practice envs are fixed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from train_fast import practice_mask                           # noqa: E402


def test_practice_mask_takes_the_share_spread_by_index():
    m = practice_mask(147, 0.25)
    assert m.dtype == bool and m.shape == (147,)
    assert int(m.sum()) == 36                                  # floor(147 x 0.25)
    gaps = np.diff(np.flatnonzero(m))
    assert set(gaps.tolist()) == {4}                           # every 4th env, evenly
    assert np.array_equal(m, practice_mask(147, 0.25))         # fixed: the same envs every call


def test_practice_mask_edges():
    assert not practice_mask(10, 0.0).any()
    assert int(practice_mask(10, 0.5).sum()) == 5
    assert int(practice_mask(7, 0.3).sum()) == 2               # floor(2.1)
    assert int(practice_mask(1029, 0.25).sum()) == 257
