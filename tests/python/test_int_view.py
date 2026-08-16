"""View-aware novelty keys (--int-view): same voxel, different gaze =
different count key; int_view=0 must reproduce the position-only key."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from surfgym.core import STATE_DTYPE
from surfgym.rewards import RaceReward


class _Flat:
    def sample(self, pos):
        return np.zeros(len(np.atleast_2d(pos)))


def _rr(int_view):
    rr = RaceReward(_Flat(), scale=1.0, int_coef=0.1, int_view=int_view)
    rr._mins = np.zeros(3)
    rr._dims = (100, 100, 100)
    return rr


def _st(x, yaw):
    s = np.zeros(len(x), STATE_DTYPE)
    s["origin"][:, 0] = x
    s["yaw"] = yaw
    return s


def test_view_off_matches_position_only_key():
    a = _rr(0)._cells(_st([1000.0, 1000.0], [10.0, 350.0]))
    assert a[0] == a[1]                      # gaze ignored when off


def test_view_sectors_split_the_same_voxel():
    rr = _rr(8)
    k = rr._cells(_st([1000.0] * 4, [0.0, 90.0, 180.0, 270.0]))
    assert len(set(k.tolist())) == 4         # four gazes, four keys
    # same gaze sector -> same key; wraparound and negatives normalize
    k2 = rr._cells(_st([1000.0, 1000.0, 1000.0], [10.0, 44.0, 370.0]))
    assert k2[0] == k2[1] == k2[2]
    kneg = rr._cells(_st([1000.0], [-350.0]))
    assert kneg[0] == k2[0]                  # -350 == +10 degrees


def test_view_key_is_position_key_times_sectors():
    r0, r8 = _rr(0), _rr(8)
    pos = _rr(0)._cells(_st([2345.0], [0.0]))
    k = r8._cells(_st([2345.0], [0.0]))
    assert k[0] == pos[0] * 8                # sector 0 of the same cell
