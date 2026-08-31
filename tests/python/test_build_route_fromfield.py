"""build_route.py --from-field: the walk down a baked goal field.

A reference line only exists today on maps somebody has already flown - the
trajectory path needs a FINISHING recording. The field path needs only the
map, so it is what makes the lookahead fan available on an unflown one; the
price is that the line is whatever the wavefront believes, staircase and
all. What has to hold regardless:

  1. it TERMINATES, in the finish basin, inside a cap that cannot be reached
     by a correct walk (strict descent cannot cycle - the cap exists so a
     broken invariant is loud instead of infinite);
  2. it descends STRICTLY, so the line is a path down the potential and not
     a tour of one plateau;
  3. it is DETERMINISTIC - an arm's reference line must not depend on which
     of two equal neighbours the loop happened to see first;
  4. it TURNS. A field line that shortcuts an L is a straight line through
     a wall, which is exactly the failure the geodesic exists to avoid;
  5. it round-trips through the npz into RouteLine, which reads only
     "route" and "spacing".

The field here is synthetic and in memory: a monotone ramp along an L-shaped
corridor with sentinel everywhere else. No map, no bake, no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import build_route                                          # noqa: E402
from surfgym.goalfield import GoalField                     # noqa: E402
from surfgym.route import DEFAULT_SPACING, RouteLine        # noqa: E402

CELL = 32.0
NX, NY, NZ = 40, 20, 8
REACH_MAX = 2000.0            # sentinel = 2064, _valid_max = 2016

# the L: leg A runs -x along iy = 3, leg B runs +y along ix = 2 and steps
# DOWN one cell in z halfway, so the walk has to take a genuine 26-neighbour
# diagonal rather than only face steps
CORNER = (2, 3, 2)            # (ix, iy, iz) where the two legs meet
FAR = (37, 3, 2)              # far end of leg A
GOAL_IY = 16                  # value 0 lives here
BEND_IY = 9                   # last iy at iz = 2


def leg_b_iz(iy):
    return 2 if iy <= BEND_IY else 1


def make_field():
    """An L-shaped corridor of honest distances, sentinel everywhere else.

    Values are a monotone ramp (one cell of distance per cell of corridor),
    not a real geodesic bake - the walk only ever compares neighbours, so
    the ordering is the whole content.
    """
    grid = np.full((NZ, NY, NX), REACH_MAX + 2.0 * CELL, np.float32)
    for iy in range(3, GOAL_IY + 1):                        # leg B, +y
        grid[leg_b_iz(iy), iy, 2] = (GOAL_IY - iy) * CELL
    d_corner = (GOAL_IY - 3) * CELL
    for ix in range(2, NX - 2):                             # leg A, -x
        grid[2, 3, ix] = d_corner + (ix - 2) * CELL
    return GoalField(grid, (0.0, 0.0, 0.0), CELL, REACH_MAX)


def centre(ix, iy, iz):
    return np.array([(ix + 0.5) * CELL, (iy + 0.5) * CELL, (iz + 0.5) * CELL])


def walk(field=None, seed=None, **kw):
    field = make_field() if field is None else field
    return build_route.trace_descent(
        field, centre(*FAR) if seed is None else seed, **kw)


# ------------------------------------------------------------- 1. terminates
def test_walk_reaches_the_goal_basin_inside_the_cap():
    field = make_field()
    pts, info = walk(field)
    assert info["d_end"] <= 2.0 * CELL, info["d_end"]
    assert info["d0"] == pytest.approx((GOAL_IY - 3) * CELL
                                       + (FAR[0] - 2) * CELL)
    assert info["n_steps"] == len(pts) == len(info["idx"])
    assert info["n_steps"] <= NX * NY * NZ
    # it stopped ON leg B, near the goal end, not somewhere off the corridor
    lx, ly, lz = (int(v) for v in info["idx"][-1])
    assert lx == 2 and lz == leg_b_iz(ly)
    assert GOAL_IY - 2 <= ly <= GOAL_IY
    assert np.allclose(pts[-1], centre(lx, ly, lz))


def test_the_step_cap_is_loud():
    """Strict descent cannot cycle, so this can only fire on a real bug -
    but when it does it must raise, not spin."""
    with pytest.raises(RuntimeError, match="step cap"):
        walk(max_steps=3)


# ---------------------------------------------------------- 2. strict descent
def test_values_along_the_path_strictly_decrease():
    field = make_field()
    _, info = walk(field)
    v = info["vals"]
    assert np.all(np.diff(v) < -build_route.DESCENT_EPS), v
    # and every value read back off the grid is the honest one
    for (ix, iy, iz), want in zip(info["idx"], v):
        assert float(field.grid[iz, iy, ix]) == pytest.approx(want)
        assert float(field.grid[iz, iy, ix]) < field._valid_max


def test_every_step_is_a_lattice_neighbour():
    _, info = walk()
    d = np.abs(np.diff(info["idx"].astype(np.int64), axis=0))
    assert d.max() <= 1 and np.all(d.sum(1) >= 1)


# ------------------------------------------------------------ 3. determinism
def test_two_walks_are_identical():
    a_pts, a_info = walk()
    b_pts, b_info = walk()
    assert np.array_equal(a_pts, b_pts)
    assert np.array_equal(a_info["idx"], b_info["idx"])
    assert np.array_equal(a_info["vals"], b_info["vals"])


# ------------------------------------------------------------------ 4. turns
def test_the_bend_is_actually_turned():
    pts, info = walk()
    corner = centre(*CORNER)
    # a 26-connected descent CUTS the corner - from (3, 3) the diagonal
    # neighbour (2, 4) is honest and lower than the corner voxel itself, so
    # the line passes one cell off it rather than through it
    assert np.linalg.norm(pts - corner, axis=1).min() <= 1.5 * CELL, \
        "the path never came near the corner voxel"
    # and it is not the straight line from start to end: measure the
    # perpendicular deviation of the path from that chord
    a, b = pts[0], pts[-1]
    u = (b - a) / np.linalg.norm(b - a)
    rel = pts - a
    perp = rel - np.outer(rel @ u, u)
    assert np.linalg.norm(perp, axis=1).max() > 5.0 * CELL
    # both legs are present in the index path
    idx = info["idx"]
    assert (idx[:, 1] == 3).sum() > 10 and (idx[:, 0] == 2).sum() > 10
    # and the z step on leg B was taken diagonally, not by dropping in place
    assert set(np.unique(idx[:, 2]).tolist()) == {1, 2}


# --------------------------------------------------------------- 5. the seed
def test_a_seed_inside_solid_snaps_to_the_corridor():
    """A hand-picked seed lands in a wall as often as not; one cell of
    search must recover the same walk."""
    want, _ = walk()
    got, _ = walk(seed=centre(*FAR) + np.array([0.0, 0.0, CELL]))
    assert np.array_equal(want, got)


def test_a_seed_in_the_void_aborts_clearly():
    with pytest.raises(ValueError, match="no honest goal-field voxel"):
        walk(seed=centre(FAR[0], FAR[1], FAR[2] + 5))


def test_snap_prefers_the_nearest_honest_voxel():
    field = make_field()
    p = centre(*FAR) + np.array([0.0, 0.6 * CELL, 0.0])   # just into iy = 4
    assert build_route.snap_seed(field, p) == FAR


# ----------------------------------------------------------- 6. npz roundtrip
def test_npz_roundtrips_into_routeline(tmp_path):
    pts, info = walk()
    line, total = build_route.resample_polyline(pts, DEFAULT_SPACING)
    out = build_route.save_field_route(tmp_path / "l.fieldroute.npz", line,
                                       DEFAULT_SPACING, "synthetic_L",
                                       centre(*FAR), info, cell=CELL)
    z = np.load(out, allow_pickle=False)
    assert z["route"].dtype == np.float32 and z["route"].shape == line.shape
    assert float(z["spacing"]) == DEFAULT_SPACING
    assert str(z["source"]) == "field"
    assert float(z["d0"]) == pytest.approx(info["d0"])
    assert float(z["d_end"]) == pytest.approx(info["d_end"])
    assert int(z["n_steps"]) == info["n_steps"]

    r = RouteLine.load(out)
    assert r.n_features == 27                    # 3 * (1 + 8 default horizons)
    assert r.L == len(line) >= 2
    assert r.spacing == DEFAULT_SPACING
    # the stored length agrees with the polyline it was resampled from
    assert r.length == pytest.approx(total, rel=0.02, abs=DEFAULT_SPACING)
    # a route runs START -> FINISH: the field value must fall along it
    assert info["vals"][0] > info["vals"][-1]


def test_parse_seed():
    assert np.array_equal(build_route.parse_seed("-9856,-5568,-2624"),
                          np.array([-9856.0, -5568.0, -2624.0]))
    assert np.array_equal(build_route.parse_seed("1 2 3"),
                          np.array([1.0, 2.0, 3.0]))
    for bad in ("1,2", "1,2,3,4", "a,b,c"):
        with pytest.raises(SystemExit):
            build_route.parse_seed(bad)
