"""tools/energy_waypoint.py (pbr3b2, 2026-10-25): the surplus of usable energy over the policy's own
greedy line, at the same place and moving the same way. Pure numpy, no core."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def _states(pos, vel):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(len(pos), STATE_DTYPE)
    s["origin"] = np.asarray(pos, np.float32)
    s["velocity"] = np.asarray(vel, np.float32)
    return s


def test_usable_energy_ignores_downward_speed():
    import energy_waypoint as ew
    s = _states([[0, 0, 100], [0, 0, 100], [0, 0, 100]], [[400, 0, 0], [0, 0, -400], [0, 0, 400]])
    assert np.allclose(ew.usable_energy(s), [200.0, 100.0, 200.0])
    assert np.allclose(ew.energy(s), [200.0, 200.0, 200.0])


def test_surplus_same_place_same_way():
    import energy_waypoint as ew
    g = _states([[x, 0, 0] for x in range(0, 1000, 10)], [[-400, 0, 0]] * 100)   # the greedy line
    a = _states([[500, 5, 0], [500, 5, 0], [500, 500, 0], [900, 0, 0]],
                [[-600, 0, 0], [600, 0, 0], [-900, 0, 0], [-300, 0, 0]])
    sur, gj, gd = ew.surplus(a, g, near=64.0, min_cos=0.5)
    assert gj[0] == 50 and np.isclose(sur[0], (600 ** 2 - 400 ** 2) / 1600.0)
    assert sur[1] == -np.inf                 # same place, the other way
    assert sur[2] == -np.inf                 # off the line
    assert sur[3] < 0                        # on the line, slower
    sur2, _, _ = ew.surplus(a, g, near=64.0)  # no direction rule
    assert np.isfinite(sur2[1])
