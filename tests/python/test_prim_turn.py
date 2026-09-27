"""--prim-turn curv (2026-09-27): a primitive's sideways knots keep their CURVATURE at the traced
speed (the rate at the floor speed, scaled by speed / floor); rate (the default) is unchanged."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from surfgym.goalprim import PrimitivePlanner, curve     # noqa: E402


def _heading_change(pts):
    d0 = pts[1] - pts[0]
    d1 = pts[-1] - pts[-2]
    return math.degrees(math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])) % 360.0


def test_rate_is_the_default_and_unchanged():
    p = np.array([90.0, 90.0, 90.0, 0.0, 0.0, 0.0])
    a = curve([0, 0, 0], [1200.0, 0, 0], 0.0, p, 1.0, 3, 300.0)
    b = curve([0, 0, 0], [1200.0, 0, 0], 0.0, p, 1.0, 3, 300.0, turn="rate")
    assert np.array_equal(a, b)
    assert abs(_heading_change(a) - 90.0) < 1.0                # 90 deg/s for 1 s


def test_curv_keeps_the_radius_so_the_rate_scales_with_speed():
    p = np.array([90.0, 90.0, 90.0, 0.0, 0.0, 0.0])
    slow = curve([0, 0, 0], [300.0, 0, 0], 0.0, p, 1.0, 3, 300.0, turn="curv")
    fast = curve([0, 0, 0], [1200.0, 0, 0], 0.0, p, 0.25, 3, 300.0, turn="curv")
    # at the floor speed curv == rate: 90 deg in 1 s
    assert abs(_heading_change(slow) - 90.0) < 1.0
    # at 4x the floor speed the same knots turn 4x as fast - exactly the rate form at 4 x 90 deg/s
    # (the same radius); the end-segment heading loses ~one 3.6 deg tick of the 90
    four = curve([0, 0, 0], [1200.0, 0, 0], 0.0, 4.0 * p, 0.25, 3, 300.0, turn="rate")
    assert np.allclose(fast, four)
    assert abs(_heading_change(fast) - 90.0) < 4.0
    r_slow = 300.0 / math.radians(90.0)
    arc = float(np.sum(np.linalg.norm(np.diff(fast, axis=0), axis=1)))
    assert abs(arc / math.radians(90.0) - r_slow) < 0.05 * r_slow


def test_planner_carries_the_turn_and_refuses_an_unknown_one():
    pp = PrimitivePlanner(turn="curv")
    assert pp.turn == "curv"
    with pytest.raises(ValueError):
        PrimitivePlanner(turn="radius")



def test_curv_uses_the_horizontal_speed_not_the_dive():
    """Codex: in the velocity frame a steep dive's VERTICAL speed must not tighten the horizontal
    turn - at horizontal 600 u/s the rate is 2x the knots, whatever the fall speed."""
    p = np.array([90.0, 90.0, 90.0, 0.0, 0.0, 0.0])
    dive = curve([0, 0, 0], [600.0, 0, -1500.0], 0.0, p, 0.5, 3, 300.0, turn="curv")
    d0 = dive[1] - dive[0]
    d1 = dive[-1] - dive[-2]
    yaw = math.degrees(math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])) % 360.0
    # 90 deg/s at the floor x (600 / 300) = 180 deg/s for 0.5 s = 90 deg (the total-speed form
    # would have turned 1,616 / 300 x 90 x 0.5 = 242 deg)
    assert abs(yaw - 90.0) < 4.0
