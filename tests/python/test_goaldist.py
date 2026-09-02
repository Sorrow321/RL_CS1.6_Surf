"""GoalDistField: the per-goal distance the goal reward shapes on, in its
Euclidean mode and in the composed-geodesic mode (max of two admissible
lower bounds on the map's one baked field, no rebake)."""
import numpy as np

from surfgym.goals import GoalDistField


class _Line:
    """A fake baked field: the finish is at x = 10,000 along +x, so
    d_F = 10000 - x; beyond it the field holds its sentinel (unreachable)."""
    reach_max = 10000.0

    def sample(self, pos):
        p = np.atleast_2d(np.asarray(pos, np.float64))
        d = 10000.0 - p[:, 0]
        return np.where(d >= 0.0, d, 20000.0).astype(np.float32)


def test_euclid_mode_is_the_straight_line():
    f = GoalDistField(2)
    f.set([0, 1], [[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(f.sample([[0, 0, 0], [3, 4, 0]]), [100.0, 5.0])


def test_geo_mode_is_exact_on_the_descent_and_symmetric_past_the_goal():
    f = GoalDistField(1, geo=_Line())
    f.set([0], [[5000.0, 0.0, 0.0]])
    # upstream on the descent: |dF(x) - dF(g)| = 4000 = the straight line
    assert np.isclose(f.sample([[1000.0, 0.0, 0.0]])[0], 4000.0)
    # past the goal: the same |delta|, so returning pays
    assert np.isclose(f.sample([[7000.0, 0.0, 0.0]])[0], 2000.0)


def test_geo_mode_euclid_pins_the_goal_on_its_level_set():
    f = GoalDistField(1, geo=_Line())
    f.set([0], [[5000.0, 0.0, 0.0]])
    # same depth as the goal, 3000u to the side: the field says 0, the
    # straight line says 3000 - the max keeps the goal a point
    assert np.isclose(f.sample([[5000.0, 3000.0, 0.0]])[0], 3000.0)
    # both terms live: the max picks the larger
    assert np.isclose(f.sample([[1000.0, 3000.0, 0.0]])[0], 5000.0)


def test_geo_mode_falls_back_to_euclid_where_the_field_is_unreachable():
    f = GoalDistField(1, geo=_Line())
    f.set([0], [[5000.0, 0.0, 0.0]])
    assert np.isclose(f.sample([[12000.0, 0.0, 0.0]])[0], 7000.0)   # x unreachable
    f.set([0], [[12000.0, 0.0, 0.0]])                                # goal unreachable
    assert np.isclose(f.sample([[0.0, 0.0, 0.0]])[0], 12000.0)


def test_nan_centre_reads_zero_in_both_modes():
    assert GoalDistField(1).sample([[0.0, 0.0, 0.0]])[0] == 0.0
    assert GoalDistField(1, geo=_Line()).sample([[0.0, 0.0, 0.0]])[0] == 0.0


def test_potential_decreases_monotonically_along_an_approach():
    f = GoalDistField(1, geo=_Line())
    f.set([0], [[8000.0, 500.0, 0.0]])
    xs = np.linspace(0.0, 8000.0, 17)
    d = np.array([f.sample([[x, 500.0 * x / 8000.0, 0.0]])[0] for x in xs])
    assert np.all(np.diff(d) < 0)
