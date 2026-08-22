"""--race-arc: Linesight's progress-along-a-reference-line reward.

The defect this exists for (ledger rounds 18-19): the shaping potential is a
geodesic BFS distance over free voxels, and on surf_src_cannonball that graph
believes the player can glide ~8,700 u laterally across open air. The result
is an interior LOCAL MINIMUM of the potential at route vertex 1601 - riding
the champion's own correct line past it RAISES d by 8,344 u and is charged
-4.24 reward. Every control arm stops within a few vertices of there.

Arc length along a reference line cannot do that: it is monotone along the
route by construction. These tests pin (a) that monotonicity on the real
route file, (b) the anti-farming rules that make an agent-position-derived
arc coordinate safe to pay, (c) agreement with the scorer this project
already trusts (tools/eval_honesty.py), and (d) that the control path with
the flag off is bit-identical.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE            # noqa: E402
from surfgym.rewards import RaceReward          # noqa: E402
from surfgym.route import ArcProgress           # noqa: E402

ROUTE = ROOT / "maps" / "surf_src_cannonball.route.npz"
SPACING = 128.0


# ---------------------------------------------------------------- fixtures
class _FlatField:
    """Geodesic stand-in: the arm keeps sampling it for the STALL detector,
    so the reward still needs one even when the shaping is arc-based."""

    def __init__(self, val=0.0):
        self.val = float(val)

    def sample(self, pos):
        return np.full(len(np.atleast_2d(pos)), self.val)


class _FakeCore:
    def __init__(self, n=1):
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)

    def at(self, xyz):
        self.states_view["origin"][:] = np.asarray(xyz, np.float32)
        return self


def _straight(n=200, spacing=SPACING):
    """A straight line down +x at constant spacing."""
    p = np.zeros((n, 3), np.float64)
    p[:, 0] = np.arange(n) * spacing
    return p


def _rr(arc, core, **kw):
    kw.setdefault("time_pen", 0.0)
    kw.setdefault("stall_ticks", 10 ** 9)
    rr = RaceReward(_FlatField(), scale=1.0, arc=arc,
                    arc_scale=kw.pop("arc_scale", 1.0), **kw)
    rr.on_reset(core)
    return rr


def _step(rr, core, done=None, trunc=None):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    done = z.copy() if done is None else np.asarray(done, np.uint8)
    trunc = z.copy() if trunc is None else np.asarray(trunc, np.uint8)
    return rr(None, None, None, np.zeros(n, np.float32), done, trunc, core)


# ------------------------------------------------- the property that matters
def test_arc_is_strictly_monotone_along_the_real_route():
    """The whole point. The geodesic potential has an interior minimum at
    route vertex 1601; an arc coordinate cannot have one at all."""
    z = np.load(ROUTE)
    pts = np.asarray(z["route"], np.float64)
    sp = float(z["spacing"])
    arc = np.arange(len(pts)) * sp
    assert np.all(np.diff(arc) > 0)
    # cumulative shaping banked by riding the line: zero drawdown everywhere
    cum = (100.0 / ((len(pts) - 1) * sp)) * arc
    assert float((cum - np.maximum.accumulate(cum)).min()) == 0.0
    # and the coordinate an ArcProgress reports ON the vertices is that arc
    ap = ArcProgress(pts, sp)
    got, off = ap.locate(pts)
    assert np.allclose(got, arc, atol=1e-3)
    assert float(off.max()) < 1e-3


def test_scale_spends_exactly_the_geodesic_budget():
    """scale = 100/route_length, so a full run collects the same 100 the
    100/d0 geodesic term collects. Derived, not tuned."""
    ap = ArcProgress.load(ROUTE)
    assert ap.length == pytest.approx(231680.0)
    scale = 100.0 / ap.length
    assert scale * ap.length == pytest.approx(100.0)
    assert scale == pytest.approx(4.3163e-4, rel=1e-3)


# ------------------------------------------------------- anti-farming rules
def test_hovering_nets_zero():
    """The failure mode named in the brief: hover near a high-arc vertex and
    collect. The signed (potential) form telescopes, so it pays nothing."""
    ap = ArcProgress(_straight(), SPACING)
    core = _FakeCore(1)
    core.at([[5000.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    total = 0.0
    for x in (5100.0, 5000.0, 5100.0, 5000.0, 5100.0, 5000.0):
        core.at([[x, 0.0, 0.0]])
        total += float(_step(rr, core)[0])
    assert abs(total) < 1e-6


def test_a_loop_out_and_back_nets_zero():
    ap = ArcProgress(_straight(), SPACING)
    core = _FakeCore(1)
    core.at([[1000.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    total = 0.0
    for x in (1030.0, 1060.0, 1090.0, 1060.0, 1030.0, 1000.0):
        core.at([[x, 0.0, 0.0]])
        total += float(_step(rr, core)[0])
    assert abs(total) < 1e-6


def test_telescopes_exactly_over_an_episode():
    ap = ArcProgress(_straight(), SPACING)
    core = _FakeCore(1)
    core.at([[0.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    a0 = float(ap.arc[0])
    total = 0.0
    for k in range(1, 120):
        core.at([[30.0 * k, 40.0 * np.sin(k / 7.0), 0.0]])
        total += float(_step(rr, core)[0])
    assert total == pytest.approx(float(ap.arc[0]) - a0, abs=1e-6)


def test_off_corridor_pays_zero_and_never_a_penalty():
    ap = ArcProgress(_straight(), SPACING, corridor=1000.0)
    core = _FakeCore(1)
    core.at([[2000.0, 0.0, 0.0]])
    rr = _rr(ap, core, time_pen=0.005)
    anchor = float(ap.arc[0])
    for y in (1200.0, 3000.0, 8000.0):
        core.at([[2000.0, y, 0.0]])
        r = float(_step(rr, core)[0])
        assert r == pytest.approx(-0.005)     # time penalty ONLY
        assert float(ap.arc[0]) == pytest.approx(anchor)   # frozen


def test_lateral_deviation_inside_the_corridor_is_not_penalized():
    """Song & Scaramuzza's distinction: pay progress ALONG the line, never
    distance TO it. Moving sideways at constant arc pays exactly zero."""
    ap = ArcProgress(_straight(), SPACING, corridor=1500.0)
    core = _FakeCore(1)
    core.at([[2000.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    for y in (100.0, 400.0, 900.0, 1400.0, 0.0):
        core.at([[2000.0, y, 0.0]])
        assert float(_step(rr, core)[0]) == pytest.approx(0.0, abs=1e-6)


def test_an_off_route_flight_cannot_claim_a_later_stretch():
    """A route that folds back within the corridor of an earlier stretch.
    Teleporting from the early stretch onto the late one must NOT cash the
    arc between them - the local window forbids it. (This is exactly the
    configuration that makes eval_honesty's global argmin over-credit by
    46,000 u on two recorded champion episodes.)"""
    a = _straight(100)                       # x = 0..12,672 at y=0
    b = a.copy()
    b[:, 0] = a[-1, 0] + SPACING + a[:, 0]   # continues on
    b[:, 1] = 200.0                          # ...only 200 u away in y
    b = b[::-1].copy()                       # and doubles back over itself
    pts = np.vstack([a, b])
    ap = ArcProgress(pts, SPACING, corridor=1500.0, window=16)
    core = _FakeCore(1)
    core.at([[1000.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    before = float(ap.arc[0])
    core.at([[1000.0, 200.0, 0.0]])          # 200 u away: inside the corridor
    r = float(_step(rr, core)[0])
    # the far stretch's arc is ~24,000 u ahead; the window caps the move
    assert abs(r) <= ap.window * SPACING
    assert float(ap.arc[0]) - before <= ap.window * SPACING


def test_the_max_step_clip_bounds_a_single_tick():
    ap = ArcProgress(_straight(400), SPACING, window=200)
    core = _FakeCore(1)
    core.at([[0.0, 0.0, 0.0]])
    rr = _rr(ap, core, max_step=100.0)
    core.at([[20000.0, 0.0, 0.0]])
    assert float(_step(rr, core)[0]) == pytest.approx(100.0)


def test_a_respawn_reanchors_with_a_global_search():
    """The local window cannot follow a relocation, so an ended row is
    re-anchored globally - otherwise the whole next episode reads
    off-corridor and earns nothing."""
    ap = ArcProgress(_straight(400), SPACING, corridor=1500.0, window=16)
    core = _FakeCore(1)
    core.at([[0.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    core.at([[30000.0, 0.0, 0.0]])           # post-autoreset spawn, far away
    assert float(_step(rr, core, done=[1])[0]) == 0.0      # ended rows pay 0
    assert float(ap.arc[0]) == pytest.approx(30000.0, abs=SPACING)
    core.at([[30100.0, 0.0, 0.0]])           # and the new episode earns
    assert float(_step(rr, core)[0]) == pytest.approx(100.0, abs=1e-3)


def test_realized_arc_stats_match_the_trajectory():
    """The diagnostic the brief asks for: arc gained per episode, checked
    against the positions actually visited."""
    ap = ArcProgress(_straight(400), SPACING, corridor=1500.0)
    core = _FakeCore(1)
    core.at([[0.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    for k in range(1, 51):
        core.at([[100.0 * k, 0.0, 0.0]])
        _step(rr, core)
    core.at([[0.0, 0.0, 0.0]])
    _step(rr, core, done=[1])
    st = rr.pop_stats()
    assert st["arc_gain"] == pytest.approx(5000.0, abs=1.0)
    assert st["arc_reach"] == pytest.approx(5000.0, abs=1.0)
    assert st["arc_off"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------ agreement with the scorer
def test_agrees_with_eval_honesty_corridor_progress():
    """ArcProgress is the incremental twin of the metric that decides this
    arm. On a trajectory with no self-approach they must agree to within the
    128 u vertex quantization the scorer snaps to."""
    from eval_honesty import corridor_progress

    pts = _straight(300)
    rng = np.random.default_rng(0)
    T = 400
    xyz = np.zeros((T, 3), np.float32)
    xyz[:, 0] = np.linspace(0.0, 20000.0, T)
    xyz[:, 1] = rng.normal(0.0, 120.0, T)
    ref, _ = corridor_progress(xyz, pts.astype(np.float32), SPACING, 1500.0)
    ap = ArcProgress(pts, SPACING, corridor=1500.0)
    p = xyz.astype(np.float64)
    ap.reset(p[:1])
    best = float(ap.arc[0])
    for k in range(1, T):
        ap.advance(p[k:k + 1])
        best = max(best, float(ap.arc[0]))
    assert abs(best - ref) <= SPACING / 2 + 1e-6


# ------------------------------------------------------ the control is safe
def _control_reward(d_prev, d, scale, time_pen, every, max_step):
    delta = d_prev - d
    delta = np.clip(delta, -max_step * every, max_step * every)
    return (delta * scale - time_pen * every).astype(np.float32)


def test_flag_off_reproduces_the_control_formula_bit_for_bit():
    core = _FakeCore(3)
    field = _FlatField()
    rr = RaceReward(field, scale=5.041e-4, time_pen=0.005, stall_ticks=10 ** 9)
    assert rr.arc is None
    rr.on_reset(core)
    rng = np.random.default_rng(7)
    d_prev = np.zeros(3)
    for _ in range(50):
        d = rng.uniform(0.0, 200000.0, 3)
        field.sample = lambda pos, _d=d: _d.copy()          # noqa: E731
        got = _step(rr, core)
        want = _control_reward(d_prev, d, 5.041e-4, 0.005, 1, 100.0)
        assert np.array_equal(got, want)
        d_prev = d


@pytest.mark.parametrize("ref", ["origin/route-obs"])
def test_flag_off_is_bit_identical_to_the_branch_point(ref):
    """The strong form: run the PREVIOUS revision of rewards.py and this one
    on the same inputs and require exact equality."""
    try:
        src = subprocess.run(["git", "show", f"{ref}:python/surfgym/rewards.py"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError):          # pragma: no cover
        pytest.skip("git unavailable")
    if src.returncode != 0:                                # pragma: no cover
        pytest.skip(f"{ref} not fetched in this clone")
    import importlib.util
    import types
    mod = types.ModuleType("rewards_old")
    mod.__file__ = str(ROOT / "python" / "surfgym" / "rewards.py")
    mod.__package__ = "surfgym"
    spec = importlib.util.spec_from_loader("rewards_old", loader=None)
    mod.__spec__ = spec
    sys.modules["rewards_old"] = mod
    exec(compile(src.stdout.replace("from .core import", "from surfgym.core import"),
                 "rewards_old", "exec"), mod.__dict__)

    core_a, core_b = _FakeCore(4), _FakeCore(4)
    fa, fb = _FlatField(), _FlatField()
    kw = dict(scale=5.041e-4, time_pen=0.005, success_bonus=50.0,
              stall_ticks=1500, int_coef=0.25, int_view=8, int_speed=3)
    new = RaceReward(fa, **kw)
    old = mod.RaceReward(fb, **kw)

    class _B:
        def map_bounds(self):
            return np.array([-20000.0, -20000.0, -20000.0]), \
                   np.array([20000.0, 20000.0, 20000.0])

    for c in (core_a, core_b):
        c.map_bounds = _B().map_bounds
    new.on_reset(core_a)
    old.on_reset(core_b)
    rng = np.random.default_rng(11)
    for t in range(200):
        d = rng.uniform(0.0, 200000.0, 4)
        pos = rng.uniform(-8000.0, 8000.0, (4, 3)).astype(np.float32)
        yaw = rng.uniform(0.0, 360.0, 4).astype(np.float32)
        vel = rng.uniform(-3000.0, 3000.0, (4, 3)).astype(np.float32)
        done = (rng.random(4) < 0.02).astype(np.uint8)
        goal = (rng.random(4) < 0.01).astype(np.uint8)
        for c, f in ((core_a, fa), (core_b, fb)):
            c.states_view["origin"][:] = pos
            c.states_view["yaw"][:] = yaw
            c.states_view["velocity"][:] = vel
            c.goal_hits[:] = goal
            f.sample = lambda p, _d=d: _d.copy()           # noqa: E731
        z = np.zeros(4, np.uint8)
        ra = new(None, None, None, z.astype(np.float32), done, z, core_a)
        rb = old(None, None, None, z.astype(np.float32), done, z, core_b)
        assert np.array_equal(ra, rb), f"diverged at t={t}"
    assert new.pop_stats() == old.pop_stats()


def test_eval_honesty_order_only_matches_the_reward_and_is_default_off():
    """The scorer's global argmin credits a fall with a later stretch
    wherever the route approaches itself. --order-only scores the way the
    reward pays; the default must stay byte-identical so every number
    already in the ledger reproduces."""
    from eval_honesty import corridor_progress, corridor_progress_ordered

    # a route that doubles straight back 384 u to one side of itself - the
    # surf_src_cannonball bowl, in miniature
    a = _straight(100)
    turn = np.array([[a[-1, 0], SPACING, 0.0], [a[-1, 0], 2 * SPACING, 0.0]])
    b = a[::-1].copy()
    b[:, 1] = 3 * SPACING
    pts = np.vstack([a, turn, b]).astype(np.float32)
    # an episode that runs the FIRST stretch and then drifts sideways until
    # the RETURN stretch is nearer, without ever travelling it
    xyz = np.zeros((60, 3), np.float32)
    xyz[:40, 0] = np.linspace(0.0, 5000.0, 40)
    xyz[40:, 0] = 5000.0
    xyz[40:, 1] = np.linspace(0.0, 2.2 * SPACING, 20)
    naive, _ = corridor_progress(xyz, pts, SPACING, 1500.0)
    honest, _ = corridor_progress_ordered(xyz, pts, SPACING, 1500.0, 16)
    assert honest < naive                       # the jump is refused
    assert honest == pytest.approx(5000.0, abs=SPACING)
    # and the default path is untouched
    plain = _straight(300).astype(np.float32)
    line = np.zeros((50, 3), np.float32)
    line[:, 0] = np.linspace(0.0, 6000.0, 50)
    assert corridor_progress(line, plain, SPACING, 1500.0)[0] == \
        pytest.approx(5888.0, abs=SPACING)


def test_record_ckpt_mentions_every_new_config_key():
    """record_ckpt.py refuses to record under semantics it does not mirror;
    the three --race-arc keys must be named there."""
    import record_ckpt
    known = record_ckpt._mentioned_keys()
    for k in ("race_arc", "race_arc_corridor", "race_arc_window"):
        assert k in known or k in record_ckpt.TRAIN_ONLY
    record_ckpt.audit_cfg({"race_arc": "maps/x.route.npz",
                           "race_arc_corridor": 1500.0,
                           "race_arc_window": 16}, strict=True)
