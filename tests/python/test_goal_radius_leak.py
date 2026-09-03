"""Regressions for three goal-system bugs found by adversarial review.

1. THE FINISH-GOAL RADIUS LEAK. ``SphereGoals.set(idx, centers)`` used to
   leave the previous radius in place when none was passed, and
   ``GoalSystem.assign`` passed one only for FINISH goals. So the first time
   an env was handed the map's finish (radius = half the finish box's longest
   side, ~3,328 u on cannonball) it kept that radius for every ordinary 192 u
   goal it was given afterwards - a permanent, per-env, 17x inflation of the
   arrival test that nothing in the logs would show. Every arm with a finish
   goal in its distribution is affected: --goal-fixed, --goal-route-uniform,
   and --goal-frontier at F = 1.

2. THE --goal-route-uniform CRASH. ``_route_goal`` guarded on
   ``s0 >= route_len - radius`` but drew from ``[s0 + 2.5 R, route_len]``, so
   a start projecting into (L - 2.5 R, L - R] handed numpy a low above its
   high and killed the trainer with ValueError.

3. ACHIEVED-GOAL k UNITS. ``assign`` set ``k = len(seg) - 1`` for
   reached-state goals - a count of reservoir SNAPSHOTS - while KCurriculum's
   band and GoalStats' bins are in seconds. At the trainer's 0.25 s goal
   cadence that read 4x high.

CPU only: no map, no DLL, no torch. The fakes are the same shape the audit
scripts used.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.goals import SphereGoals                      # noqa: E402
from surfgym.goalsys import GoalSystem                     # noqa: E402
from surfgym.respawn import RespawnBuffer                  # noqa: E402

ROUTE_LEN = 100_000.0
FINISH_MINS = [97_000.0, -100.0, -700.0]      # 6,000 x 200 x 1,400, like
FINISH_MAXS = [103_000.0, 100.0, 700.0]       # cannonball's 6,656 x 1 x 1,472


class FakeField:
    """d = ROUTE_LEN - x, reachable everywhere, sentinel past the finish."""

    reach_max = ROUTE_LEN

    def sample(self, pos):
        p = np.atleast_2d(np.asarray(pos, np.float64))
        d = ROUTE_LEN - p[:, 0]
        return np.where(d >= 0.0, d, 2.0 * ROUTE_LEN).astype(np.float32)

    def reachable(self, pos):
        return np.ones(len(np.atleast_2d(pos)), bool)


class FakeCore:
    def __init__(self, n):
        self.num_envs = int(n)
        self.states_view = np.zeros(self.num_envs, STATE_DTYPE)
        self.config = types.SimpleNamespace(
            phys=types.SimpleNamespace(sv_gravity=800.0))
        self._bounds = (np.asarray([-2e4, -2e4, -2e4], np.float32),
                        np.asarray([2e5, 2e4, 2e4], np.float32))

    def map_bounds(self):
        return self._bounds

    def at(self, i, pos):
        self.states_view["origin"][i] = np.asarray(pos, np.float32)
        return self


def goal_args(**over):
    a = types.SimpleNamespace(
        goal_radius=192.0, goal_holdout=None, goal_air_frac=0.0,
        goal_kmin=1.0, goal_kmax=5.0, goal_kcap=60.0, goal_curriculum=0,
        goal_frontier=0, goal_route_uniform=0, goal_front_start=0.05,
        goal_front_band=0.05, goal_front_step=0.10, goal_front_rate=0.30,
        goal_front_min_ep=300, goal_route=None, goal_route_frac=0.0,
        goal_fixed=0, goal_fixed_decay=0.0, goal_fixed_spacing=2000.0,
        goal_fixed_air=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def straight_route(path, n=1001):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, ROUTE_LEN, n)
    pts = np.stack([x, np.zeros(n), np.zeros(n)], 1)
    np.savez(path, route=pts.astype(np.float32),
             spacing=np.float32(ROUTE_LEN / (n - 1)))
    return str(path)


def make_system(tmp_path, n_envs=2, **arg_over):
    rp = straight_route(tmp_path / "route.npz")
    core = FakeCore(n_envs)
    args = goal_args(goal_route=rp, goal_route_frac=1.0, **arg_over)
    gs = GoalSystem(core, n_envs, None, FakeField(), ROUTE_LEN, args, "cpu",
                    str(tmp_path / "out"), seed=0)
    gs.set_finish(FINISH_MINS, FINISH_MAXS)
    return core, gs


# ------------------------------------------------------- 1. the radius leak
def test_sphere_set_without_a_radius_returns_to_nominal():
    g = SphereGoals(3, radius=192.0)
    g.set(np.array([0]), np.zeros((1, 3)), radius=np.float32(3328.0))
    assert float(g.radius[0]) == pytest.approx(3328.0)
    g.set(np.array([0]), np.array([[1000.0, 0.0, 0.0]]))
    assert float(g.radius[0]) == pytest.approx(192.0)
    # 1,000 u from the centre is outside a 192 u sphere and inside a 3,328 u
    # one: the leak is exactly the difference between these two answers
    assert not bool(g.hit(np.array([[2000.0, 0.0, 0.0]] * 3))[0])


@pytest.mark.parametrize("mode", ["uniform", "fixed"])
def test_finish_radius_does_not_leak_into_the_next_goal(tmp_path, mode):
    over = ({"goal_route_uniform": 1} if mode == "uniform"
            else {"goal_fixed": 1, "goal_fixed_spacing": 2000.0,
                  "goal_fixed_air": 0})
    core, gs = make_system(tmp_path / mode, n_envs=2, **over)
    nominal, finish = gs.radius, gs.finish_radius
    assert finish > 10.0 * nominal                 # the leak has to matter

    # 1) a start near the end: draw until the goal IS the finish
    core.at(0, [99_000.0, 0.0, 0.0])
    for _ in range(400):
        gs.assign(np.array([0]))
        if gs._last_finish:
            break
    assert gs._last_finish, "never drew the finish goal"
    assert float(gs.sphere.radius[0]) == pytest.approx(finish)

    # 2) respawn at the start; the next goal is an ordinary one
    core.at(0, [0.0, 0.0, 0.0])
    for _ in range(50):
        gs.assign(np.array([0]))
        if not gs._last_finish:
            break
    assert not gs._last_finish, "never drew a non-finish goal"
    assert float(gs.sphere.radius[0]) == pytest.approx(nominal)

    # the observable consequence: the arrival test must not fire 1,000 u out
    c = gs.sphere.center[0].astype(np.float64)
    far = np.repeat((c + np.array([1000.0, 0.0, 0.0]))[None, :], 2, 0)
    assert not bool(gs.sphere.hit(far)[0])
    # and a sibling env that was never given the finish is untouched
    assert float(gs.sphere.radius[1]) == pytest.approx(nominal)


def test_ball_radius_returns_to_nominal():
    """Same reset rule on the depth-channel ball (no GPU: a stub lidar)."""
    torch = pytest.importorskip("torch")
    from surfgym.goalball import GoalBallLidar

    lidar = types.SimpleNamespace(
        channels=1, pinhole=False, W=8, H=4, device=torch.device("cpu"),
        near=32.0, range=8192.0,
        yoff=torch.linspace(0.4, -0.4, 8), poff=torch.linspace(0.2, -0.2, 4))
    gb = GoalBallLidar(lidar, 2, radius=192.0, views=1)
    gb.set_goals([0], [[100.0, 0.0, 0.0]],
                 radius=np.full(1, 3328.0, np.float32))
    assert float(gb.radius[0]) == pytest.approx(3328.0)
    gb.set_goals([0], [[200.0, 0.0, 0.0]])
    assert float(gb.radius[0]) == pytest.approx(192.0)
    assert float(gb.radius[1]) == pytest.approx(192.0)


# --------------------------------------------- 2. the route-uniform crash
# the route vertices are 100 u apart, so a start projects to the nearest of
# them: x = L - 384 lands on s0 = L - 400 and x = L - 193 on s0 = L - 200.
# Both cleared the old `s0 >= L - R` guard (R = 192) and then asked for
# uniform(s0 + 480, L) with the low above the high.
@pytest.mark.parametrize("back", [384.0, 193.0, 400.0, 300.0, 200.0,
                                  100.0, 0.0])
def test_route_uniform_near_the_line_end_returns_none(tmp_path, back):
    """Starts in the last 2.5 R of the line have nothing left to draw."""
    core, gs = make_system(tmp_path / f"b{back:.0f}", n_envs=1,
                           goal_route_uniform=1)
    o = np.array([ROUTE_LEN - back, 0.0, 0.0])
    assert gs._route_goal(o) is None                      # no ValueError
    core.at(0, o)
    gs.assign(np.array([0]))                              # falls back to air
    assert gs.kind[0] != 2


def test_route_uniform_still_draws_with_room_left(tmp_path):
    _core, gs = make_system(tmp_path, n_envs=1, goal_route_uniform=1)
    for back in (5000.0, 500.0):        # 500 = the first vertex that fits
        got = gs._route_goal(np.array([ROUTE_LEN - back, 0.0, 0.0]))
        assert got is not None, back
        g, line, k = got
        assert np.isfinite(g).all() and len(line) >= 2 and k >= 0.0


def test_route_goal_never_raises_anywhere_on_the_line(tmp_path):
    """The sweep the guard exists for: no start position may crash the draw."""
    for over in ({"goal_route_uniform": 1}, {},
                 {"goal_frontier": 1, "goal_front_start": 1.0}):
        tag = "u" if over.get("goal_route_uniform") else (
            "f" if over.get("goal_frontier") else "p")
        _core, gs = make_system(tmp_path / tag, n_envs=1, **over)
        xs = np.concatenate([np.arange(0.0, ROUTE_LEN, 977.0),
                             np.arange(ROUTE_LEN - 2000.0,
                                       ROUTE_LEN + 1.0, 1.0)])
        for x in xs:
            gs._route_goal(np.array([float(x), 0.0, 0.0]))    # must not raise


# ------------------------------------------------ 3. achieved-goal k units
def _reservoir_with_a_chain(snap_every, ticks=2000, step=40.0):
    """One env flying +x at `step` u/tick; harvest its own future as goals."""
    rb = RespawnBuffer(1, reservoir=10_000, margin_ticks=0,
                       snap_every=snap_every, goal_k=(100, 500), seg_max=64,
                       goal_min_dist=480.0, seed=1)
    st = np.zeros(1, STATE_DTYPE)
    st["velocity"][0] = [step * 100.0, 0.0, 0.0]        # u/s
    for t in range(1, ticks + 1):
        st["origin"][0] = [step * t, 0.0, 0.0]
        rb.observe(st, np.array([t == ticks]))
    rows, _ticks, _envs = rb.drain_harvest()
    goals, segs, seglen = rb._last_goals
    rb.push_many(rows, goals=goals, segs=segs, seglen=seglen)
    return rb, rows


@pytest.mark.parametrize("snap_every", [25, 100])
def test_achieved_goal_k_is_in_seconds(tmp_path, snap_every):
    rb, rows = _reservoir_with_a_chain(snap_every)
    pool, pg, ps, psl = rb.build_pool(rows[:1], pool_size=64, fresh_frac=0.0,
                                      vel_scale=(1.0, 1.0), pitch_jitter=0.0,
                                      with_goals=True)
    core = FakeCore(len(pool))
    core.states_view[:] = pool
    gs = GoalSystem(core, len(pool), None, FakeField(), ROUTE_LEN,
                    goal_args(), "cpu", str(tmp_path / f"k{snap_every}"),
                    seed=0, snap_every=snap_every)
    gs.set_pool(pool, pg, ps, psl)
    gs.assign(np.arange(len(pool)))

    ach = gs.kind == 0
    assert ach.any(), "no reached-state goals in the pool"
    # truth: the chain flies at a known speed, so seconds = distance / speed
    speed = float(pool["velocity"][0, 0])
    secs = (gs.sphere.center[ach, 0].astype(np.float64)
            - pool["origin"][ach, 0].astype(np.float64)) / speed
    assert np.median(gs.k[ach] / np.maximum(secs, 1e-9)) == pytest.approx(
        1.0, abs=0.15)
    # and the raw snapshot count - the bug - is off by exactly the cadence
    if snap_every != 100:
        assert np.median(gs.k[ach] / np.maximum(secs, 1e-9)
                         * (100.0 / snap_every)) > 3.0


def test_achieved_goal_k_is_capped_and_floored(tmp_path):
    """kcap bounds it above; one snapshot interval bounds it below."""
    core = FakeCore(2)
    gs = GoalSystem(core, 2, None, FakeField(), ROUTE_LEN,
                    goal_args(goal_kcap=3.0), "cpu", str(tmp_path / "cap"),
                    seed=0, snap_every=25)
    pool = np.zeros(2, STATE_DTYPE)
    pool["origin"][0] = [0.0, 0.0, 0.0]
    pool["origin"][1] = [10_000.0, 0.0, 0.0]
    goals = np.array([[5000.0, 0.0, 0.0], [10_600.0, 0.0, 0.0]], np.float32)
    segs = np.zeros((2, 64, 3), np.float32)
    segs[0, :64, 0] = np.linspace(0.0, 5000.0, 64)      # a long chain
    segs[1, 0] = [10_000.0, 0.0, 0.0]
    segs[1, 1] = [10_600.0, 0.0, 0.0]                   # the shortest chain
    gs.set_pool(pool, goals, segs, np.array([64, 2], np.int32))
    core.states_view[:] = pool
    gs.assign(np.array([0, 1]))
    assert (gs.kind == 0).all()
    assert float(gs.k[0]) == pytest.approx(3.0)         # 63 * 0.25 s -> kcap
    assert float(gs.k[1]) == pytest.approx(0.25)        # 1 * 0.25 s
