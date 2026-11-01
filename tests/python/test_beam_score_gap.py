"""beam_tas --score gap: -max(d, exact hull-to-finish-box gap) (zlr3b1, 2026-11-01)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import beam_tas  # noqa: E402


class ZeroField:
    def sample(self, p):
        return np.zeros(len(np.atleast_2d(p)), np.float32)


class ConstField:
    def __init__(self, v):
        self.v = v

    def sample(self, p):
        return np.full(len(np.atleast_2d(p)), self.v, np.float32)


BOXES = [[[-320.0, 1312.0, 3008.0], [-64.0, 3360.0, 3209.6]],
         [[-2360.0, 536.0, 3008.0], [3008.0, 1312.0, 3209.6]]]


def _states(rows):
    st = np.zeros(len(rows), STATE_DTYPE)
    for i, (o, duck) in enumerate(rows):
        st["origin"][i] = o
        st["ducked"][i] = duck
    return st


def test_gap_exact_beside_a_face():
    sc = beam_tas.make_scorer(ZeroField(), None, 0.0, "gap", goal_boxes=BOXES)
    st = _states([((-368.0, 1350.0, 2980.0), 0),    # 22 u north of the band's face (y 1312)
                  ((-368.0, 1328.0, 2980.0), 0),    # touching it (hull y 1312)
                  ((-368.0, 1328.0, 2980.0), 1),    # ducked: hull top 2998 < 3008
                  ((-48.0, 2017.0, 2972.0), 0)])    # the strip's east face, hull top 3008
    s, dg = sc(st)
    assert np.allclose(dg, [22.0, 0.0, 10.0, 0.0], atol=1e-4), dg
    assert np.allclose(s, -dg)


def test_gap_is_plain_d_when_the_field_is_larger():
    sc = beam_tas.make_scorer(ConstField(500.0), None, 0.0, "gap", goal_boxes=BOXES)
    st = _states([((0.0, 1500.0, 2700.0), 0)])
    s, dg = sc(st)
    assert dg[0] == 500.0 and s[0] == -500.0


def test_d_mode_unchanged():
    sc = beam_tas.make_scorer(ConstField(7.0), None, 0.0, "d")
    s, d = sc(_states([((0.0, 0.0, 0.0), 0)]))
    assert d[0] == 7.0 and s[0] == -7.0
