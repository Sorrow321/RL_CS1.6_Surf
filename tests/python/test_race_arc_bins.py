"""--race-arc-bins: QUANTISE the progress coordinate into N chunks.

The idea, and why it is not a guess. Round 19 found that the geodesic's
fine-grained gradient encoded one specific path which had an interior local
optimum, and FLATTENING it is what broke the wall: xLATCH switched the
shaping term off past the frontier and 0 finishes in 234 episodes became
52/102. A chunked potential applies that same move everywhere instead of
once at the end - inside a chunk there is no gradient at all, which says
*any pathing inside this chunk is equally good* and stops the reward
dictating a specific route.

It also sweeps a density axis whose ends are already measured:

    N = 1        goal bonus only          xNOSHP: 0/9, xBIN3 stuck at 2.4%
    N = 20       this arm                 ?
    N -> inf     continuous arc length    xARC: 63/102 finishes

so the headline question is how coarse the reference may get before it stops
working - at 20 waypoints, equipping a new map is cheap.

What these tests pin:
 (a) the potential IS the chunk index - flat inside a chunk, +/-1 across a
     boundary, and nothing else;
 (b) the budget is still exactly 100 over a start -> finish run, so the arm
     does not silently also move the racing-beats-quitting constraint;
 (c) the order-only window and the corridor gate survive quantisation, which
     is what stops a global nearest-chunk assignment paying for dying;
 (d) N = 0 is bit-identical to continuous arc length.
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
N_BINS = 20


# ---------------------------------------------------------------- fixtures
class _FlatField:
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


# ------------------------------------------------ the potential IS the index
def test_the_potential_is_the_chunk_index():
    """floor(arc / (length/N)), clipped to [0, N-1] - nothing else."""
    pts = _straight(201)                       # length 25,600 u
    ap = ArcProgress(pts, SPACING, bins=N_BINS)
    assert ap.length == pytest.approx(25600.0)
    assert ap._binw == pytest.approx(1280.0)
    arc = np.array([0.0, 1279.9, 1280.0, 6400.0, 25599.9, 25600.0])
    want = np.array([0.0, 0.0, 1.0, 5.0, 19.0, 19.0])   # last bin clipped
    assert np.array_equal(ap._bin(arc), want)


def test_no_gradient_at_all_inside_a_chunk():
    """The deliberate part: any pathing within one chunk is equally good."""
    ap = ArcProgress(_straight(201), SPACING, corridor=1500.0, bins=N_BINS)
    core = _FakeCore(1)
    core.at([[6500.0, 0.0, 0.0]])              # bin 5 spans 6,400-7,680
    rr = _rr(ap, core)
    path = ((6550.0, 0.0), (6600.0, 400.0), (6650.0, -900.0),
            (6700.0, 1200.0), (6750.0, 0.0), (7679.0, 0.0))
    for x, y in path:
        core.at([[x, y, 0.0]])
        assert float(_step(rr, core)[0]) == pytest.approx(0.0, abs=1e-12)
    # ...and the continuous control pays for every one of those moves
    ap_c = ArcProgress(_straight(201), SPACING, corridor=1500.0)
    core_c = _FakeCore(1)
    core_c.at([[6500.0, 0.0, 0.0]])
    rr_c = _rr(ap_c, core_c)
    paid = 0.0
    for x, y in path[:-1]:
        core_c.at([[x, y, 0.0]])
        paid += abs(float(_step(rr_c, core_c)[0]))
    assert paid == pytest.approx(250.0, abs=1.0)     # 5 x 50u of arc


def test_a_boundary_crossing_pays_exactly_one_chunk():
    ap = ArcProgress(_straight(201), SPACING, corridor=1500.0, bins=N_BINS)
    core = _FakeCore(1)
    core.at([[1279.0, 0.0, 0.0]])              # last inch of bin 0
    rr = _rr(ap, core, arc_scale=100.0 / ap.pay_span)
    core.at([[1281.0, 0.0, 0.0]])              # first inch of bin 1
    assert float(_step(rr, core)[0]) == pytest.approx(100.0 / (N_BINS - 1))
    core.at([[1279.0, 0.0, 0.0]])              # and back: the refund
    assert float(_step(rr, core)[0]) == pytest.approx(-100.0 / (N_BINS - 1))


# ------------------------------------------------------------- the budget
def test_the_total_collectible_budget_is_still_exactly_100():
    """Matching scale = 100/d0 and arc_scale = 100/route_length. Otherwise
    the arm silently also moves the racing-beats-quitting constraint
    (shaping income must exceed time_pen; the measured cliff is
    0.0125/tick) and two things would be under test at once."""
    ap = ArcProgress.load(ROUTE, bins=N_BINS)
    assert ap.length == pytest.approx(231680.0)
    assert ap.pay_span == pytest.approx(float(N_BINS - 1))
    scale = 100.0 / ap.pay_span
    # ride the whole line: index 0 -> N-1, i.e. N-1 crossings
    span = float(ap._bin(np.array([ap.length]))[0]
                 - ap._bin(np.array([0.0]))[0])
    assert span == pytest.approx(float(N_BINS - 1))
    assert scale * span == pytest.approx(100.0)
    # the continuous control spends the same 100 over the same line
    ap_c = ArcProgress.load(ROUTE)
    assert (100.0 / ap_c.pay_span) * ap_c.length == pytest.approx(100.0)


def test_income_still_beats_the_time_penalty_over_a_run():
    """The constraint the scale exists to hold fixed: an 81 s run banks 100
    of shaping against 81 s * 100 ticks/s * 0.005 = ~40 of time cost, the
    same margin the continuous arm and the geodesic control have."""
    ticks = 81.0 * 100.0                       # an 81 s run at 100 ticks/s
    per_tick = {}
    for n in (0, N_BINS):
        ap = ArcProgress.load(ROUTE, bins=n)
        span = (ap.length if n == 0 else float(n - 1))
        per_tick[n] = (100.0 / ap.pay_span) * span / ticks
    # income per tick is identical quantised and continuous, and that is the
    # whole point of deriving the scale from pay_span
    assert per_tick[N_BINS] == pytest.approx(per_tick[0])
    assert per_tick[N_BINS] == pytest.approx(0.01235, rel=1e-3)
    # racing beats quitting: income/tick > time_pen. The measured cliff is
    # 0.0125/tick (backlog item 0), and the arm runs the default 0.005.
    assert per_tick[N_BINS] > 0.005
    assert per_tick[N_BINS] < 0.0125            # ...and the cliff is where


# ------------------------------------------- anti-farming survives binning
def test_hovering_across_a_boundary_nets_zero():
    """Sitting on a chunk edge and oscillating over it is the obvious farm.
    The telescoping (signed) form pays it exactly nothing."""
    ap = ArcProgress(_straight(201), SPACING, corridor=1500.0, bins=N_BINS)
    core = _FakeCore(1)
    core.at([[1279.0, 0.0, 0.0]])
    rr = _rr(ap, core, arc_scale=100.0 / ap.pay_span)
    total = 0.0
    for x in (1281.0, 1279.0) * 8:
        core.at([[x, 0.0, 0.0]])
        total += float(_step(rr, core)[0])
    assert abs(total) < 1e-9


def test_telescopes_to_the_index_difference_over_an_episode():
    ap = ArcProgress(_straight(201), SPACING, corridor=1500.0, bins=N_BINS)
    core = _FakeCore(1)
    core.at([[0.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    b0 = float(ap._bin(ap.arc)[0])
    total = 0.0
    for k in range(1, 200):
        core.at([[100.0 * k, 300.0 * np.sin(k / 5.0), 0.0]])
        total += float(_step(rr, core)[0])
    assert total == pytest.approx(float(ap._bin(ap.arc)[0]) - b0, abs=1e-9)


def test_off_corridor_pays_zero_and_never_a_penalty():
    ap = ArcProgress(_straight(201), SPACING, corridor=1000.0, bins=N_BINS)
    core = _FakeCore(1)
    core.at([[2000.0, 0.0, 0.0]])
    rr = _rr(ap, core, time_pen=0.005)
    anchor = float(ap.arc[0])
    for y in (1200.0, 3000.0, 8000.0):
        core.at([[2000.0, y, 0.0]])
        assert float(_step(rr, core)[0]) == pytest.approx(-0.005)
        assert float(ap.arc[0]) == pytest.approx(anchor)


def test_the_order_only_window_still_refuses_a_folded_back_chunk():
    """NOT optional, and the reason this is built on ArcProgress rather than
    a global nearest-chunk assignment: a global argmin credits a fall with
    up to 46,000 u where this route folds back, and cannonball's terminal
    fall lands only 4,703 u from the goal in straight-line terms - a LATE
    chunk, i.e. the agent would be paid for dying."""
    a = _straight(100)                    # out along y = 0
    b = a[::-1].copy()                    # ...and back 200u to one side
    b[:, 1] = 200.0
    pts = np.vstack([a, b])
    ap = ArcProgress(pts, SPACING, corridor=1500.0, window=16, bins=N_BINS)
    core = _FakeCore(1)
    core.at([[1000.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    core.at([[1000.0, 200.0, 0.0]])       # 200 u away, but ~24,000 u of arc
    r = float(_step(rr, core)[0])
    # the window caps the anchor move at 2,048 u; one chunk here is 1,274 u,
    # so at most two boundaries - never the folded-back stretch
    assert abs(r) <= np.ceil(ap.window * SPACING / ap._binw)
    # what it refuses: the GLOBAL nearest chunk at that same point is the
    # last one on the line, i.e. the fall would have been paid for
    g_arc, g_off = ap.locate(np.array([[1000.0, 200.0, 0.0]]))
    assert float(g_off[0]) < 1.0                     # exactly on the return
    assert float(ap._bin(g_arc)[0]) == float(N_BINS - 1)
    assert float(ap._bin(ap.arc)[0]) <= 2.0          # what it actually paid


def test_a_respawn_reanchors_globally_and_the_new_chunk_is_the_spawn_chunk():
    ap = ArcProgress(_straight(401), SPACING, corridor=1500.0, window=16,
                     bins=N_BINS)
    core = _FakeCore(1)
    core.at([[0.0, 0.0, 0.0]])
    rr = _rr(ap, core)
    core.at([[30000.0, 0.0, 0.0]])
    assert float(_step(rr, core, done=[1])[0]) == 0.0
    assert float(ap.arc[0]) == pytest.approx(30000.0, abs=SPACING)
    binw = ap._binw                              # 51,200 / 20 = 2,560 u
    assert float(ap._bin(ap.arc)[0]) == pytest.approx(np.floor(30000.0 / binw))
    core.at([[30100.0, 0.0, 0.0]])               # same chunk: pays nothing
    assert float(_step(rr, core)[0]) == pytest.approx(0.0, abs=1e-12)


def test_the_diagnostics_stay_in_UNITS_so_they_stay_comparable_to_xARC():
    """arc_gain / arc_reach / arc_off are read against xARC's published
    figures; quantising the PAYMENT must not quantise the measurement."""
    ap = ArcProgress(_straight(401), SPACING, corridor=1500.0, bins=N_BINS)
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


# ---------------------------------------------------- N = 0 is the control
def test_bins_zero_is_the_continuous_object_field_for_field():
    a = ArcProgress.load(ROUTE)
    b = ArcProgress.load(ROUTE, bins=0)
    assert a.bins == b.bins == 0
    assert a.pay_span == b.pay_span == a.length
    assert a.describe() == b.describe()


def test_bins_zero_is_bit_identical_to_continuous_arc_length():
    """The proof the brief asks for: N = 0 must be bit-identical to
    continuous. Same route, same corridor, same window, same positions -
    every reward float equal, not approximately equal."""
    pts = np.asarray(np.load(ROUTE)["route"], np.float64)
    rng = np.random.default_rng(3)
    n = 8
    ap0 = ArcProgress(pts, SPACING, corridor=1500.0, window=16, bins=0)
    apN = ArcProgress(pts, SPACING, corridor=1500.0, window=16)
    c0, cN = _FakeCore(n), _FakeCore(n)
    start = pts[rng.integers(0, len(pts), n)]
    c0.at(start)
    cN.at(start)
    r0 = _rr(ap0, c0, time_pen=0.005, arc_scale=4.3163e-4)
    rN = _rr(apN, cN, time_pen=0.005, arc_scale=4.3163e-4)
    idx = rng.integers(0, len(pts), n)
    out0, outN = [], []
    for t in range(300):
        idx = np.clip(idx + rng.integers(-3, 6, n), 0, len(pts) - 1)
        p = (pts[idx] + rng.normal(0.0, 400.0, (n, 3))).astype(np.float32)
        done = (rng.random(n) < 0.02).astype(np.uint8)
        c0.at(p)
        cN.at(p)
        out0.append(_step(r0, c0, done=done))
        outN.append(_step(rN, cN, done=done))
        assert np.array_equal(out0[-1], outN[-1]), f"diverged at t={t}"
    assert np.array_equal(np.concatenate(out0), np.concatenate(outN))
    sa, sb = r0.pop_stats(), rN.pop_stats()
    assert sa.keys() == sb.keys()
    for k in sa:                       # equal_nan: an arm with no finishes
        assert np.array_equal(sa[k], sb[k], equal_nan=True), k


def test_bins_zero_is_bit_identical_to_the_arclen_branch(
        ref="origin/arclen"):
    """The strong form: run the PREVIOUS revision of route.py - the one the
    xARC control actually ran - against this one with the flag off, and
    require exact equality."""
    try:
        src = subprocess.run(["git", "show", f"{ref}:python/surfgym/route.py"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.SubprocessError):          # pragma: no cover
        pytest.skip("git unavailable")
    if src.returncode != 0:                                # pragma: no cover
        pytest.skip(f"{ref} not fetched in this clone")
    import types
    mod = types.ModuleType("route_old")
    mod.__file__ = str(ROOT / "python" / "surfgym" / "route.py")
    mod.__package__ = "surfgym"
    exec(compile(src.stdout, "route_old", "exec"), mod.__dict__)

    pts = np.asarray(np.load(ROUTE)["route"], np.float64)
    new = ArcProgress(pts, SPACING, corridor=1500.0, window=16, bins=0)
    old = mod.ArcProgress(pts, SPACING, corridor=1500.0, window=16)
    rng = np.random.default_rng(5)
    n = 8
    idx = rng.integers(0, len(pts), n)
    p0 = pts[idx].astype(np.float64)
    new.reset(p0)
    old.reset(p0)
    assert np.array_equal(new.arc, old.arc)
    for t in range(400):
        idx = np.clip(idx + rng.integers(-4, 8, n), 0, len(pts) - 1)
        p = pts[idx] + rng.normal(0.0, 600.0, (n, 3))
        da, ia = new.advance(p)
        db, ib = old.advance(p)
        assert np.array_equal(da, db), f"delta diverged at t={t}"
        assert np.array_equal(ia, ib), f"corridor gate diverged at t={t}"
        assert np.array_equal(new.arc, old.arc)
        assert np.array_equal(new.idx, old.idx)


# ------------------------------------------------------------- the plumbing
def test_the_flag_exists_and_threads_through_to_the_reward():
    out = subprocess.run([sys.executable, str(ROOT / "python" / "train_fast.py"),
                          "--help"], capture_output=True, text=True,
                         timeout=300)
    assert "--race-arc-bins" in out.stdout


def test_record_ckpt_mentions_the_new_config_key():
    import record_ckpt
    known = record_ckpt._mentioned_keys()
    assert "race_arc_bins" in known or "race_arc_bins" in record_ckpt.TRAIN_ONLY
    record_ckpt.audit_cfg({"race_arc": "maps/x.route.npz",
                           "race_arc_corridor": 1500.0,
                           "race_arc_window": 16,
                           "race_arc_bins": N_BINS}, strict=True)


def test_the_real_route_at_20_bins_is_11584u_per_chunk():
    """What the arm actually runs. 231,680 u / 20 = 11,584 u a chunk, i.e.
    90.5 route vertices - against a legal ~35 u of motion per tick, so a
    chunk is ~5 s of champion-pace flying with no gradient in it at all."""
    ap = ArcProgress.load(ROUTE, bins=N_BINS)
    assert ap._binw == pytest.approx(11584.0)
    assert ap._binw / SPACING == pytest.approx(90.5)
    # the wall (route vertex ~1598 = 204,544 u, 88.3%) sits inside chunk 17
    assert int(ap._bin(np.array([1598 * SPACING]))[0]) == 17
    assert "QUANTISED into 20 chunks" in ap.describe()
