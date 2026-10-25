"""tools/spine_canon_keys.py: the wish direction survives putting the view along the velocity."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import spine_canon_keys as sck  # noqa: E402


def test_wish_angles():
    assert sck.wish_angle(2, 1) == 0.0          # forward
    assert sck.wish_angle(1, 0) == 90.0         # left strafe (sidemove -400)
    assert sck.wish_angle(1, 2) == -90.0        # right strafe
    assert sck.wish_angle(0, 1) == 180.0        # back
    assert sck.wish_angle(2, 0) == 45.0         # forward + left
    assert np.isnan(sck.wish_angle(1, 1))       # nothing held


def test_translate_keeps_the_wish_direction():
    # forward with the view 89 deg LEFT of the velocity = a left strafe along it
    k = sck.translate(np.array([[2, 1, 0]]), yaw=[189.0], heading=[100.0])
    assert k.tolist() == [[1, 0, 0]]
    # forward + left with the view 46 deg left (round 4's GE14 ramp) -> left strafe
    k = sck.translate(np.array([[2, 0, 1]]), yaw=[243.0], heading=[197.0])
    assert k.tolist() == [[1, 0, 1]]            # duck untouched
    # already canonical: unchanged; nothing held: unchanged
    k = sck.translate(np.array([[1, 2, 0], [1, 1, 0]]), yaw=[30.0, 30.0], heading=[30.0, 30.0])
    assert k.tolist() == [[1, 2, 0], [1, 1, 0]]
    # wrap-around: view 350, heading 10 -> offset -20; forward -> forward
    k = sck.translate(np.array([[2, 1, 0]]), yaw=[350.0], heading=[10.0])
    assert k.tolist() == [[2, 1, 0]]
