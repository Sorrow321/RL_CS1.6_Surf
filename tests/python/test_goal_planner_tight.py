"""--plan-graph tight (goalplan.BFSPlanner(graph_kind="tight")): the ride shell's cells and edges,
costed by height above the surface below plus a penalty near the rim of that surface's footprint.

On a synthetic surf ramp (a ridge prism along x with a gap to hop, then a turn onto a second
prism along y, whose corner the ride graph's shortest route cuts):
  * tight has exactly the ride graph's nodes and edges (connectivity is unchanged)
  * its route runs on the ridge, the ride route does not
  * Plan.length is the geometric length of the path, not the weighted cost
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from surfgym.goalplan import BFSPlanner   # noqa: E402

CELL = 32.0


def _map():
    nz, ny, nx = 14, 40, 44
    occ = np.zeros((nz, ny, nx), np.uint8)

    def ridge_x(y0, x0, x1):          # a prism along x, centred on row y0, 9 cells wide
        for lvl, half in enumerate((4, 3, 2, 1, 0)):
            occ[2 + lvl, y0 - half:y0 + half + 1, x0:x1] = 1

    def ridge_y(x0, y0, y1):          # a prism along y, centred on column x0
        for lvl, half in enumerate((4, 3, 2, 1, 0)):
            occ[2 + lvl, y0:y1, x0 - half:x0 + half + 1] = 1

    ridge_x(6, 2, 18)                 # leg 1 along x ...
    ridge_x(6, 21, 36)                # ... with a 3-cell gap at x 18..20
    ridge_y(31, 11, 38)               # leg 2 along y, turning at the leg-1 end (gap y 11 -> 10)
    return occ, np.zeros(3)


def _planner(kind, occ, mins):
    fin = {"mins": [31 * CELL - 40, 36 * CELL, 7 * CELL], "maxs": [31 * CELL + 72, 38 * CELL, 9 * CELL]}
    return BFSPlanner(occ, mins, CELL, finish_box=fin, kill_z=-np.inf, n_targets=0, seed=0,
                      graph_kind=kind)


def test_tight_keeps_the_ride_graph_and_rides_the_ridge():
    occ, mins = _map()
    ride = _planner("ride", occ, mins)
    tight = _planner("tight", occ, mins)
    assert ride.wedge is None and tight.wedge is not None
    assert np.array_equal(ride.node_of, tight.node_of)
    assert np.array_equal(ride.nbr, tight.nbr)
    start = np.array([4.5 * CELL, 6.5 * CELL, 7.5 * CELL])       # above the leg-1 ridge
    pr, pt = ride.plan(start, ride.fin), tight.plan(start, tight.fin)
    assert pr is not None and pt is not None
    nodes_t = np.asarray(tight.coords)[tight._descend(
        tight.dist[tight.fin], tight.snap(start[None, :])[0], tight.nbr, tight.wedge)]
    # every node of the tight route on leg 1 (x < 27) sits over the ridge row (y 6) +- 1 cell,
    # on leg 2 (y > 12) over the ridge column (x 31) +- 1 cell
    leg1 = nodes_t[nodes_t[:, 2] < 27]
    leg2 = nodes_t[nodes_t[:, 1] > 12]
    assert len(leg1) and np.all(np.abs(leg1[:, 1] - 6) <= 1), leg1
    assert len(leg2) and np.all(np.abs(leg2[:, 2] - 31) <= 1), leg2
    # the ride route cuts the turn: somewhere it is >= 2 cells off both ridge lines
    nodes_r = np.asarray(ride.coords)[ride._descend(
        ride.dist[ride.fin], ride.snap(start[None, :])[0], ride.nbr, ride.wk)]
    off = (np.abs(nodes_r[:, 1] - 6) >= 2) & (np.abs(nodes_r[:, 2] - 31) >= 2)
    assert off.any()
    # Plan.length: geometric on tight (the field holds the weighted cost), the field on ride
    assert abs(pr.length - float(ride.dist[ride.fin, pr.start])) < 1e-3
    geo = float(np.linalg.norm(np.diff(tight.xyz[tight._descend(
        tight.dist[tight.fin], pt.start, tight.nbr, tight.wedge)], axis=0), axis=1).sum())
    assert abs(pt.length - geo) < 1e-6
    assert float(tight.dist[tight.fin, pt.start]) > pt.length


def test_tight_random_target_band_is_in_map_units():
    """choose()'s dmin/dmax band reads the GEOMETRIC length of the chosen route on tight (the
    fields hold weighted cost), so a tight run draws the same targets by distance as ride."""
    occ, mins = _map()
    fin = {"mins": [31 * CELL - 40, 36 * CELL, 7 * CELL], "maxs": [31 * CELL + 72, 38 * CELL, 9 * CELL]}
    tight = BFSPlanner(occ, mins, CELL, finish_box=fin, kill_z=-np.inf, n_targets=64, seed=0,
                       graph_kind="tight")
    assert tight.glen is not None and tight.glen.shape == tight.dist.shape
    ok = np.isfinite(tight.dist)
    # geometric length <= weighted cost (every cell costs >= 1 per unit), and > 0 off the source
    assert np.all(tight.glen[ok] <= tight.dist[ok] + 1e-3)
    s = int(tight.snap(np.array([[4.5 * CELL, 6.5 * CELL, 7.5 * CELL]]))[0])
    rng = np.random.default_rng(1)
    for _ in range(50):
        t = tight.choose(s, rng, 0.0, 256.0, 1024.0)
        if t >= 0 and np.isfinite(tight.glen[t, s]) and 256.0 <= tight.glen[:tight.n_rand, s].max():
            pl = tight.plan(np.array([4.5 * CELL, 6.5 * CELL, 7.5 * CELL]), t)
            assert pl is not None
            # the plan's own length matches the band's table to within the descent's tie-breaks
            assert abs(pl.length - float(tight.glen[t, s])) <= 2 * CELL
    ride = BFSPlanner(occ, mins, CELL, finish_box=fin, kill_z=-np.inf, n_targets=64, seed=0,
                      graph_kind="ride")
    assert ride.glen is None
