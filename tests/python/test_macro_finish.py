"""tools/macro_finish.py: plan layout, plan rows per tick, the inflated-box distance (CPU, no map)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import macro_finish as MF  # noqa: E402


def test_draw_plans_layout():
    rng = np.random.default_rng(0)
    p = MF.draw_plans(rng, 50, 3, (0.1, 1.0), 0.007, 300, 0.1, 0.05, 0.5, 3.0, 120.0)
    assert p.shape == (50, 3, 7)
    end = p[..., MF.F_END]
    assert (end[:, -1] == 300).all()
    assert (np.diff(end, axis=1) >= 0).all()
    assert set(np.unique(p[..., MF.F_FWD])) <= {0.0, 1.0, 2.0}
    assert set(np.unique(p[..., MF.F_SIDE])) <= {0.0, 1.0, 2.0}
    q = MF.perturb(rng, p, 2.0, 300)
    assert q.shape == p.shape and (q[..., MF.F_END][:, -1] == 300).all()
    assert (np.diff(q[..., MF.F_END], axis=1) >= 0).all()


def test_plan_rows_switches_on_end_ticks():
    p = np.zeros((1, 2, 7))
    p[0, 0] = [2, 0, 0, 0, 5.0, 0.0, 10]     # forward, side left, yaw +5 until tick 10
    p[0, 1] = [1, 2, 1, 0, -7.0, 0.0, 20]    # none, side right, ducked, yaw -7 after
    a, v = MF.plan_rows(p, 9)
    assert a[0].tolist() == [7, 3, 2, 0, 0, 0] and v[0].tolist() == [5.0, 0.0]
    a, v = MF.plan_rows(p, 10)
    assert a[0].tolist() == [7, 3, 1, 2, 0, 1] and v[0].tolist() == [-7.0, 0.0]
    a, v = MF.plan_rows(p, 50)                # past the last end: the last macro holds
    assert a[0, 3] == 2


def test_box_dist_uses_the_hull_inflated_box():
    boxes = [((0.0, 0.0, 100.0), (10.0, 10.0, 110.0))]
    o = np.array([[5.0, 5.0, 64.0],     # standing: inflated z min 100 - 36 = 64 -> inside
                  [5.0, 5.0, 63.0],     # 1 u below it
                  [5.0, 5.0, 81.0],     # ducked: inflated z min 100 - 18 = 82 -> 1 u below
                  [-20.0, 5.0, 100.0]])  # 4 u west of the inflated x min (-16)
    d = MF.box_dist(o, np.array([0, 0, 1, 0]), boxes)
    assert np.allclose(d, [0.0, 1.0, 1.0, 4.0])
