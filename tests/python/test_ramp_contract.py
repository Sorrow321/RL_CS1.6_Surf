"""The ramp-command contract (tools/edge_archive.py --moves ramp; Codex reviews 2026-09-27):
the outcome rule, the collision lineage, the line capacity that never cuts the arrival, and the
planner applying the executor's unknown-root first-tick rule."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import edge_archive as ea  # noqa: E402


class _FakeMap:
    """one ramp target: a 20 km x 400 u plane tilted about x, normal (0, -0.78, 0.62)"""

    def __init__(self):
        xs = np.linspace(0.0, 20000.0, 400)
        ys = np.linspace(0.0, 400.0, 5)
        g = np.array([(x, y, 0.62 / 0.78 * y) for x in xs for y in ys], np.float64)
        self.kp = g
        n = np.array([0.0, -0.78, 0.62])
        self.kn = np.tile(n / np.linalg.norm(n), (len(g), 1))
        self.n_surf = 1
        self.cat = np.array([1])
        self.members = [np.arange(len(g))]


def _op(cap):
    op = ea.RampOperator(10.0, np.array([1e6, 0.0, 0.0]), _FakeMap(), 4)
    op.line_cap = cap
    return op


def test_outcome_rule():
    class _P:
        FIN = 3
        targets = [10, 11, 12]
    base = {"fin": False, "ticks": 100, "died": False}
    assert ea.ramp_outcome(dict(base, hit_tick=20, hit_set=[11]), _P, 1) == "direct"
    assert ea.ramp_outcome(dict(base, hit_tick=20, hit_set=[11, 12]), _P, 1) == "wrong"
    assert ea.ramp_outcome(dict(base, hit_tick=20, hit_set=[-4]), _P, 1) == "wrong"
    assert ea.ramp_outcome(dict(base, hit_tick=-1, hit_set=None), _P, 1) == "none"
    # a contact that a death follows is still the command's first contact
    assert ea.ramp_outcome(dict(base, hit_tick=20, hit_set=[11], died=True), _P, 1) == "direct"
    # the finish command: direct only with no contact before the finish
    assert ea.ramp_outcome(dict(base, fin=True, hit_tick=-1, hit_set=None), _P, 3) == "direct"
    assert ea.ramp_outcome(dict(base, fin=True, hit_tick=20, hit_set=[10]), _P, 3) == "wrong"
    assert ea.ramp_outcome(dict(base, fin=True, hit_tick=-1, hit_set=None), _P, 0) == "fin"


def test_lineage_is_capture_set_plus_capture_tick_contacts():
    a = ea.Archive()
    st = np.zeros(1, dtype=[("origin", "f4", 3), ("velocity", "f4", 3)])[0]
    n0 = a.add(st, None, np.zeros(1, np.float32), ("K",), -1, -1, 0, 0, None)
    assert not ea.has_lineage(a, n0)
    ea.note_contact(a, n0, {"hit_set": [5], "end_touch": [5, 7]})
    assert ea.has_lineage(a, n0) and ea.lineage(a, n0) == {5, 7}
    b = ea.Archive()
    assert b.token != a.token                     # a monotonic identity, never id() reuse


def test_capacity_cut_keeps_the_arrival():
    o = np.array([0.0, -2000.0, 3000.0])
    v = np.array([3000.0, 0.0, 0.0])
    full, pts = _op(cap=768).line_and_curve_of(o, v, 0.0, np.zeros(1), 0)
    herm = pts[:max(2, len(pts) - int(ea.RAMP_TAIL / 0.01))]
    arr_len = float(np.linalg.norm(np.diff(herm, axis=0), axis=1).sum())
    step = float(np.linalg.norm(np.diff(full.astype(np.float64), axis=0), axis=1).sum()) / (len(full) - 1)
    n_arr = int(np.ceil(arr_len / step)) + 1          # vertices up to the first one past arrival
    cap = n_arr + 2
    assert cap < len(full), "the scenario must need a cut"
    ln, _p = _op(cap=cap).line_and_curve_of(o, v, 0.0, np.zeros(1), 0)
    assert len(ln) == cap
    arc = float(np.linalg.norm(np.diff(ln.astype(np.float64), axis=0), axis=1).sum())
    assert arc >= arr_len - 1e-2                      # the kept polyline still reaches the arrival


def test_an_arrival_over_capacity_raises():
    op = _op(cap=4)
    o = np.array([0.0, -2000.0, 3000.0])
    v = np.array([3000.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        op.line_and_curve_of(o, v, 0.0, op.choice_nums[0], 0)


MAP = ROOT / "maps_pool" / "surf_edgeflow_blue025.bsp"
RAMPS = ROOT / "runs" / "research" / "ramps3_blue025.npz"


@pytest.mark.skipif(not (MAP.exists() and RAMPS.exists()),
                    reason="needs the edgeflow map and its local ramps3 extraction")
def test_planner_never_ranks_an_unknown_roots_first_tick_contacts():
    from ramps import RampMap
    from surfgym.core import SurfCore, default_config
    core = SurfCore(str(MAP), default_config(num_envs=4, max_episode_ticks=100000))
    if getattr(core._lib, "surf_get_touch", None) is None:
        pytest.skip("this surfcore build predates surf_get_touch")
    core.reset(0)
    neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (4, 1))
    for _ in range(20):                           # settle on the start floor
        core.step(neutral)
    root = core.get_states()[0].copy()
    rm = RampMap(str(RAMPS))
    # what the root touches on its first neutral tick (the executor's source for it)
    core.set_state(0, root)
    core.step(neutral)
    c, n_, p_ = core.get_touch()
    first = {s for s in rm.touch_sets(c[:1], n_[:1], p_[:1])[0] if s >= 0}
    assert first, "the settled root should touch its floor"

    class _Fl:
        pass
    fl = _Fl()
    fl.core = core
    fl.live_ticks = 0
    op = ea.RampOperator(10.0, np.array([0.0, 0.0, 0.0]), rm, 4)
    a = ea.Archive()
    nid = a.add(root, None, np.zeros(1, np.float32), ("R",), -1, -1, 0, 0, None)
    op.plan(fl, a, [nid])
    ranked = {op.targets[k] for k in op.rank[(a.token, nid)] if k < op.FIN}
    assert not (ranked & first)
    op.evict(a.token)
    assert not any(k[0] == a.token for k in op.rank)
