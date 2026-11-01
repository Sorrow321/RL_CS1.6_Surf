"""beam_tas --score gap: -max(d, exact hull gap to the REACHABLE finish region) (zlr3b1,
2026-11-01). A box standing on a solid floor is reachable only with the hull clear of the floor."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
import beam_tas  # noqa: E402


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


def test_gap_without_floors_is_the_hull_box_gap():
    sc = beam_tas.make_scorer(ConstField(0.0), None, 0.0, "gap", goal_boxes=BOXES)
    st = _states([((-368.0, 1350.0, 2980.0), 0),    # 22 u north of the band's face (y 1312)
                  ((-368.0, 1328.0, 2980.0), 0),    # touching it (hull y 1312)
                  ((-368.0, 1328.0, 2980.0), 1),    # ducked: hull top 2998 < 3008
                  ((-48.0, 2017.0, 2972.0), 0)])    # the strip's east face, hull top 3008
    s, dg = sc(st)
    assert np.allclose(dg, [22.0, 0.0, 10.0, 0.0], atol=1e-4), dg
    assert np.allclose(s, -dg)


def test_gap_with_a_floor_needs_the_hull_clear_of_it():
    sc = beam_tas.make_scorer(ConstField(0.0), None, 0.0, "gap", goal_boxes=BOXES,
                              goal_floors=[3000.0, 3000.0])
    st = _states([((-336.0, 1450.0, 2982.5), 0),    # pressed on the strip's side: 53.5 low
                  ((-336.0, 1450.0, 2982.5), 1),    # ducked: needs 3018 -> 35.5 low
                  ((-336.0, 1450.0, 3036.0), 0),    # standing clear of the floor, at the edge
                  ((-200.0, 2000.0, 3044.0), 0),    # standing on the strip
                  ((-48.0, 2017.0, 3048.0), 0)])    # the record's entry (analysis only)
    s, dg = sc(st)
    assert np.allclose(dg, [53.5, 35.5, 0.0, 0.0, 0.0], atol=1e-4), dg


def test_gap_is_plain_d_when_the_field_is_larger():
    sc = beam_tas.make_scorer(ConstField(500.0), None, 0.0, "gap", goal_boxes=BOXES)
    s, dg = sc(_states([((0.0, 1500.0, 2700.0), 0)]))
    assert dg[0] == 500.0 and s[0] == -500.0


def test_d_mode_unchanged():
    sc = beam_tas.make_scorer(ConstField(7.0), None, 0.0, "d")
    s, d = sc(_states([((0.0, 0.0, 0.0), 0)]))
    assert d[0] == 7.0 and s[0] == -7.0


def test_floors_on_za_loop():
    bsp = ROOT / "maps" / "surf_za_loop.bsp"
    if not bsp.exists():
        import pytest
        pytest.skip("map not present")
    import json
    from surfgym.core import SurfCore, default_config
    core = SurfCore(str(bsp), default_config(num_envs=1))
    z = json.loads((ROOT / "maps" / "surf_za_loop.zones.json").read_text())
    fl = beam_tas.goal_box_floors(core, z["end"]["boxes"])
    # the platform top is z 3008 (the trace stops 1/32 u above it), the boxes start there
    assert all(f is not None and abs(f - 3008.0) < 1.0 for f in fl), fl


def test_gaph_adds_the_energy_height():
    sc = beam_tas.make_scorer(ConstField(0.0), None, 0.0, "gaph", energy_k=0.5,
                              gravity=800.0, goal_boxes=BOXES, goal_floors=[3008.0, 3008.0])
    st = _states([((-336.0, 1450.0, 2982.5), 0)])
    st["velocity"][0] = (0.0, 0.0, 400.0)            # h = z + 100
    s, dg = sc(st)
    assert np.isclose(dg[0], 3044.0 - 2982.5) and np.isclose(s[0], -dg[0] + 0.5 * 3082.5)
