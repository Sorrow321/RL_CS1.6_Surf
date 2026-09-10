"""``dip/*``: the setback diagnostic, and the two implementations of it.

A DIP is a maximal contiguous stretch of reward calls whose
``depth = (d - b) * scale`` is positive, ``b`` being the episode's running
minimum of the geodesic distance.  It SURVIVES when a new record closes it
and FAILS when the episode ends inside it.  The point of the metric is the
one question ``race/eval_progress`` cannot ask - how much temporary loss the
policy tolerates before it dies - and it is only evidence if the number the
trainer logs and the number an offline trajectory analysis computes are the
SAME number.  So the definition lives once, in :mod:`surfgym.dipmeter`, and
this file pins:

  * the ONLINE accumulator (``DipMeter``, what ``RaceReward`` drives) equals
    the OFFLINE reference (``enumerate_dips``, what the trajectory analysis
    imports) call for call, over random walks with random episode ends;
  * the RESET semantics: spawn, reservoir respawn, demo respawn, truncation
    - every episode start restarts ``b`` at its own distance, so a respawn
    deep in the map can never fabricate a dip.  ``--respawn-frac 0.9`` means
    most episodes start mid-map, so this is the common case, not the corner;
  * the depth at TERMINATION is the depth of the episode's LAST LIVE state,
    not of the next episode's spawn (the core autoresets before the reward
    is computed);
  * ``depth`` is in REWARD UNITS - the whole map is worth 100 on any map -
    so two maps of very different length are directly comparable;
  * and that it is LOGGING ONLY: the meter on and the meter off produce
    bit-identical rewards, bit-identical stall/stagnant masks and identical
    numpy/torch RNG states.

    python -m pytest tests/python/test_dipmeter.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                        # noqa: E402
from surfgym.dipmeter import (DIP_KEYS, DipMeter,           # noqa: E402
                              dip_depths, dip_summary,
                              dips_from_episode, enumerate_dips,
                              merge_dip_raw, terminal_depth)
from surfgym.goalfield import EuclidField                   # noqa: E402
from surfgym.rewards import RaceReward                      # noqa: E402

ZONES = ROOT / "maps" / "surf_src_cannonball.zones.json"
D0 = 198380.0
SCALE = 100.0 / D0


# --------------------------------------------------------------------------
# the fake core RaceReward's own tests use
# --------------------------------------------------------------------------
class FakeCore:
    def __init__(self, n: int = 1, bounds=((-3e4, -3e4, -3e4),
                                           (3e4, 3e4, 3e4))):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self._bounds = (np.asarray(bounds[0], np.float32),
                        np.asarray(bounds[1], np.float32))

    def map_bounds(self):
        return self._bounds


def _step(rw, core, done=None, trunc=None):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    done = z if done is None else np.asarray(done, np.uint8)
    trunc = z if trunc is None else np.asarray(trunc, np.uint8)
    obs = np.zeros((n, 15), np.float32)
    return rw(obs, obs, obs, np.zeros(n, np.float32), done, trunc, core)


@pytest.fixture
def goal_box():
    if ZONES.exists():
        return json.loads(ZONES.read_text(encoding="utf-8"))["end"]
    return {"mins": [-14720.0, 7487.0, -1824.0],
            "maxs": [-8064.0, 7488.0, -352.0]}


def _race(goal_box, **kw):
    kw.setdefault("scale", SCALE)
    kw.setdefault("time_pen", 0.0)
    return RaceReward(EuclidField(goal_box), **kw)


def _at(core, d, i=0):
    """Put env ``i`` where the field reads (very close to) ``d`` units."""
    p = core.states_view["origin"].copy()
    p[i] = (-11000.0, 7488.0 + float(d), -1000.0)
    core.states_view["origin"] = p
    return core


def _place(core, ds):
    p = core.states_view["origin"].copy()
    for i, d in enumerate(ds):
        p[i] = (-11000.0, 7488.0 + float(d), -1000.0)
    core.states_view["origin"] = p
    return core


# --------------------------------------------------------------------------
# 1. the offline reference, by hand
# --------------------------------------------------------------------------
def test_the_worked_example():
    """``d 10 -> 8 -> 12 -> 11 -> 8 -> 6`` at scale 1, dt 0.04.

    Sample 0 is the spawn, so depth is 0, 0, 4, 3, 0, 0: ONE dip, samples
    2..3, depth 4, two calls = 0.08 s, closed by the record at sample 4 -
    survived.
    """
    d = [10.0, 8.0, 12.0, 11.0, 8.0, 6.0]
    assert dip_depths(d, 1.0).tolist() == [0.0, 0.0, 4.0, 3.0, 0.0, 0.0]
    dips = enumerate_dips(d, 1.0, 0.04)
    assert len(dips) == 1
    st, en, dep, secs, surv = dips[0]
    assert (st, en) == (2, 4)
    assert dep == pytest.approx(4.0)
    assert secs == pytest.approx(0.08)
    assert surv is True
    assert terminal_depth(d, 1.0) == 0.0


def test_a_dip_open_at_the_end_fails_only_if_the_episode_ended():
    d = [10.0, 8.0, 12.0, 11.0]          # never gets back to a record
    fail = enumerate_dips(d, 1.0, 0.04, ended_in_dip=True)
    assert len(fail) == 1 and fail[0][4] is False
    assert fail[0][2] == pytest.approx(4.0) and fail[0][3] == pytest.approx(0.08)
    # the same array as a CUT WINDOW: unresolved, reported as neither
    assert enumerate_dips(d, 1.0, 0.04, ended_in_dip=False) == []
    assert terminal_depth(d, 1.0) == pytest.approx(3.0)


def test_depth_is_never_negative_and_zero_at_a_record():
    rng = np.random.default_rng(0)
    d = np.cumsum(rng.normal(size=400)) * 50.0 + 5000.0
    dep = dip_depths(d, SCALE)
    assert (dep >= 0.0).all()
    assert dep[0] == 0.0
    # zero exactly where d is a new running minimum
    rec = d <= np.minimum.accumulate(d)
    assert np.array_equal(dep == 0.0, rec)


def test_depth_is_in_reward_units_so_maps_are_comparable():
    """A setback of 10% of map A's d0 and 10% of map B's d0 read the SAME,
    which is the whole reason the scale is folded in."""
    a = enumerate_dips([1000.0, 900.0, 900.0 + 0.10 * 1000.0, 500.0],
                       100.0 / 1000.0, 0.04)
    b = enumerate_dips([50000.0, 45000.0, 45000.0 + 0.10 * 50000.0, 100.0],
                       100.0 / 50000.0, 0.04)
    assert a[0][2] == pytest.approx(b[0][2]) == pytest.approx(10.0)


# --------------------------------------------------------------------------
# 2. ONLINE == OFFLINE, which is the requirement the metric rests on
# --------------------------------------------------------------------------
def _online_per_env(paths, ends, scale, dt):
    """Drive DipMeter over N env columns and return its pooled raw dict."""
    n = len(paths)
    T = len(paths[0])
    m = DipMeter(n)
    m.start(np.array([p[0] for p in paths], np.float64))
    for t in range(1, T):
        d = np.array([p[t] for p in paths], np.float64)
        e = np.array([bool(x[t]) for x in ends], bool)
        m.update(d, e, scale, dt)
    return m.pop_raw()


def _offline_per_env(paths, ends, scale, dt):
    """Split each env column into episodes exactly as the autoreset does -
    a call with ended[t] is the FIRST call of the next episode, and the
    dying episode's last sample is t-1 - and enumerate each with the
    reference implementation."""
    parts = []
    for p, e in zip(paths, ends):
        start = 0
        for t in range(1, len(p)):
            if e[t]:
                parts.append(dips_from_episode(p[start:t], scale, dt,
                                               ended_in_dip=True))
                start = t
        if start < len(p) - 1:      # the tail episode never ended: unresolved
            parts.append(dips_from_episode(p[start:], scale, dt,
                                           ended_in_dip=False))
    return merge_dip_raw(parts)


def _same(a, b):
    for k in ("surv_depth", "surv_secs", "fail_depth", "fail_secs",
              "term_depth"):
        x, y = np.sort(np.asarray(a[k])), np.sort(np.asarray(b[k]))
        assert x.shape == y.shape, f"{k}: {x.shape} vs {y.shape}"
        assert np.array_equal(x, y), f"{k}: {x[:8]} vs {y[:8]}"
    assert int(a["n_ended"]) == int(b["n_ended"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_online_accumulator_equals_the_offline_reference(seed):
    """Random walks in ``d`` with random terminals, 8 envs x 600 calls, the
    exact numbers - not a tolerance.  This is what stops the trainer's
    columns and an offline trajectory analysis from drifting apart."""
    rng = np.random.default_rng(seed)
    n, T = 8, 600
    dt = 4 * 10.0 / 1000.0
    paths, ends = [], []
    for _ in range(n):
        d = np.cumsum(rng.normal(size=T)) * 120.0 + 40000.0
        e = rng.random(T) < 0.02
        e[0] = False
        # an episode start relocates d arbitrarily (a reservoir respawn is a
        # teleport): the meter must restart its record there
        for t in np.flatnonzero(e):
            d[t:] += rng.normal() * 8000.0
        paths.append(d)
        ends.append(e)
    scale = SCALE
    _same(_online_per_env(paths, ends, scale, dt),
          _offline_per_env(paths, ends, scale, dt))


def test_summary_is_the_same_from_either_side():
    rng = np.random.default_rng(7)
    n, T = 6, 900
    dt = 4 * 7.666667 / 1000.0
    paths, ends = [], []
    for _ in range(n):
        d = np.abs(np.cumsum(rng.normal(size=T))) * 200.0 + 10000.0
        e = rng.random(T) < 0.03
        e[0] = False
        paths.append(d)
        ends.append(e)
    on = dip_summary(_online_per_env(paths, ends, SCALE, dt))
    off = dip_summary(_offline_per_env(paths, ends, SCALE, dt))
    assert set(on) == set(DIP_KEYS)
    for k in DIP_KEYS:
        assert on[k] == pytest.approx(off[k], rel=0, abs=0), k
    # the window is big enough that the test is actually testing something
    assert on["fail_frac"] > 0.0 and on["max_survived_depth"] > 0.0


# --------------------------------------------------------------------------
# 3. reset semantics: a respawn restarts b, and never fabricates a dip
# --------------------------------------------------------------------------
def test_a_respawn_deep_in_the_map_starts_a_fresh_record(goal_box):
    """``--respawn-frac 0.9`` puts most episodes mid-map.  The record must
    restart at the RESPAWN's own distance: otherwise the jump from 20,000 to
    3,000 would either read as a giant record or (jumping the other way) as
    a fabricated 17,000 u dip the policy never flew."""
    rw = _race(goal_box, dip=True)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))            # spawn, anchors b
    _step(rw, _at(core, 15000.0))            # a record
    # the episode ends and the core autoresets DEEP in the map, at 3,000
    _step(rw, _at(core, 3000.0), done=[1])
    _step(rw, _at(core, 3200.0))             # 200 u out of the NEW record
    _step(rw, _at(core, 2900.0))             # back to a record: dip closed
    raw = rw.pop_dip_raw()
    assert raw["n_ended"] == 1
    # nothing about the 12,000 u relocation is in the numbers
    assert len(raw["surv_depth"]) == 1
    assert raw["surv_depth"][0] == pytest.approx(200.0 * SCALE, rel=1e-3)
    assert len(raw["fail_depth"]) == 0
    assert raw["term_depth"][0] == pytest.approx(0.0, abs=1e-9)


def test_termination_depth_is_the_last_live_state_not_the_new_spawn(goal_box):
    """The core autoresets before the reward is computed, so on the ended
    call ``d`` is already the next episode's spawn.  The depth an episode
    DIED at is the previous call's."""
    rw = _race(goal_box, dip=True)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    _step(rw, _at(core, 10000.0))            # record
    _step(rw, _at(core, 12500.0))            # 2,500 u into a dip
    _step(rw, _at(core, 19000.0), done=[1])  # dies here; d is the new spawn
    raw = rw.pop_dip_raw()
    assert raw["n_ended"] == 1
    assert len(raw["surv_depth"]) == 0
    assert len(raw["fail_depth"]) == 1
    assert raw["fail_depth"][0] == pytest.approx(2500.0 * SCALE, rel=1e-3)
    assert raw["term_depth"][0] == pytest.approx(2500.0 * SCALE, rel=1e-3)
    s = dip_summary(raw)
    assert s["fail_frac"] == 1.0
    assert s["survived_per_ep"] == 0.0
    assert np.isnan(s["max_survived_depth"])


def test_truncation_counts_as_an_episode_end(goal_box):
    rw = _race(goal_box, dip=True)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    _step(rw, _at(core, 10000.0))
    _step(rw, _at(core, 11000.0))
    _step(rw, _at(core, 20000.0), trunc=[1])
    raw = rw.pop_dip_raw()
    assert raw["n_ended"] == 1 and len(raw["fail_depth"]) == 1


def test_duration_uses_the_runs_own_tick_base(goal_box):
    """SECONDS, from ``act_every * tick_ms``, never a hard-coded 100 Hz.
    Same call sequence at two tick bases -> two different durations."""
    def run(tick_ms, every):
        rw = _race(goal_box, dip=True, tick_ms=tick_ms, every=every)
        core = FakeCore()
        _step(rw, _at(core, 20000.0))
        _step(rw, _at(core, 10000.0))
        for _ in range(5):                     # 5 calls inside the dip
            _step(rw, _at(core, 10500.0))
        _step(rw, _at(core, 9000.0))           # closes it
        return rw.pop_dip_raw()["surv_secs"][0]

    assert run(10.0, 4) == pytest.approx(5 * 4 * 10.0 / 1000.0)
    assert run(7.666667, 4) == pytest.approx(5 * 4 * 7.666667 / 1000.0)
    assert run(10.0, 1) == pytest.approx(5 * 1 * 10.0 / 1000.0)


def test_the_window_drains(goal_box):
    rw = _race(goal_box, dip=True)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    _step(rw, _at(core, 10000.0))
    _step(rw, _at(core, 11000.0))
    _step(rw, _at(core, 9000.0))
    assert len(rw.pop_dip_raw()["surv_depth"]) == 1
    assert len(rw.pop_dip_raw()["surv_depth"]) == 0     # drained


def test_many_envs_at_once(goal_box):
    """The trainer runs 2,048 env columns through one call; the per-env
    bookkeeping must not leak across columns."""
    n = 4
    rw = _race(goal_box, dip=True)
    core = FakeCore(n)
    _place(core, [20000.0] * n)
    _step(rw, core)
    _place(core, [10000.0] * n)
    _step(rw, core)
    # env 0 dips 1,000 and recovers; env 1 dips 3,000 and dies; envs 2/3
    # keep setting records
    _place(core, [11000.0, 13000.0, 9000.0, 8000.0])
    _step(rw, core)
    _place(core, [8000.0, 20000.0, 8500.0, 7000.0])
    _step(rw, core, done=[0, 1, 0, 0])
    raw = rw.pop_dip_raw()
    assert raw["n_ended"] == 1
    assert len(raw["surv_depth"]) == 1
    assert raw["surv_depth"][0] == pytest.approx(1000.0 * SCALE, rel=1e-3)
    assert raw["fail_depth"][0] == pytest.approx(3000.0 * SCALE, rel=1e-3)


# --------------------------------------------------------------------------
# 4. LOGGING ONLY: on and off are the same run
# --------------------------------------------------------------------------
def _drive(rw, path, ends, n=1):
    core = FakeCore(n)
    out = []
    for k, d in enumerate(path):
        e = ends[k]
        out.append(_step(rw, _at(core, d),
                         done=[1] if e == 1 else [0],
                         trunc=[1] if e == 2 else [0]).copy())
    return np.concatenate(out), core


PATH = ([20000.0 - 137.0 * i for i in range(80)]
        + [6000.0 + 311.0 * i for i in range(30)]
        + [15000.0 - 97.0 * i for i in range(120)])
ENDS = [0] * len(PATH)
for _i in (23, 61, 104, 149, 188, 210):
    ENDS[_i] = 1
ENDS[137] = 2


def test_the_meter_does_not_change_the_reward(goal_box):
    """Bit for bit, over a path that backtracks, terminates and truncates -
    the requirement that makes this safe to leave ON by default."""
    a, _ = _drive(_race(goal_box, time_pen=0.005, dip=True), PATH, ENDS)
    b, _ = _drive(_race(goal_box, time_pen=0.005, dip=False), PATH, ENDS)
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a, b), "the dip meter perturbed the reward"


def test_the_meter_does_not_change_the_liveness_masks(goal_box):
    ra = _race(goal_box, time_pen=0.005, dip=True)
    rb = _race(goal_box, time_pen=0.005, dip=False)
    _drive(ra, PATH, ENDS)
    _drive(rb, PATH, ENDS)
    assert np.array_equal(ra.stagnant_mask(), rb.stagnant_mask())
    ma, mb = ra.pop_stall_mask(), rb.pop_stall_mask()
    assert (ma is None and mb is None) or np.array_equal(ma, mb)
    sa, sb = ra.pop_stats(), rb.pop_stats()
    assert set(sa) == set(sb)
    for k in sa:
        assert sa[k] == sb[k] or (sa[k] != sa[k] and sb[k] != sb[k]), k


def test_the_meter_draws_from_no_rng(goal_box):
    """numpy AND torch, global and default_rng: identical states after the
    same number of reward calls with the meter on and off.  A diagnostic
    that consumed one draw would fork every rollout after it."""
    import torch

    def states(dip):
        np.random.seed(12345)
        torch.manual_seed(12345)
        rw = _race(goal_box, time_pen=0.005, dip=dip)
        _drive(rw, PATH, ENDS)
        ns = np.random.get_state()
        return (ns[0], ns[1].copy(), ns[2], ns[3], ns[4],
                torch.get_rng_state().clone())

    a, b = states(True), states(False)
    assert a[0] == b[0] and a[2] == b[2] and a[3] == b[3] and a[4] == b[4]
    assert np.array_equal(a[1], b[1]), "numpy RNG state diverged"
    assert torch.equal(a[5], b[5]), "torch RNG state diverged"


def test_the_meter_leaves_every_piece_of_reward_state_identical(goal_box):
    """Not just the returned reward: every array the reward carries forward.
    If the meter had touched one of them the divergence would appear on a
    LATER call, which a single-call comparison would miss."""
    ra = _race(goal_box, time_pen=0.005, int_coef=0.25, dip=True)
    rb = _race(goal_box, time_pen=0.005, int_coef=0.25, dip=False)
    _drive(ra, PATH, ENDS)
    _drive(rb, PATH, ENDS)
    for name in ("_d", "_dc", "_best", "_since", "_ticks", "_s", "_rec",
                 "_latched", "_counts", "_prev_cell", "_d0"):
        x, y = getattr(ra, name), getattr(rb, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert np.asarray(x).dtype == np.asarray(y).dtype, name
            assert np.array_equal(x, y), name
    for name in ("n_success", "n_fail", "n_trunc", "int_paid"):
        assert getattr(ra, name) == getattr(rb, name), name


def test_the_meter_does_not_mutate_what_it_is_handed(goal_box):
    """DipMeter.update takes `d` and `ended` by reference; an in-place op on
    either would corrupt the reward that is computed from the same arrays."""
    m = DipMeter(3)
    d0 = np.array([100.0, 200.0, 300.0])
    m.start(d0)
    d0_before = d0.copy()
    d = np.array([90.0, 260.0, 300.0])
    e = np.array([False, False, True])
    d_before, e_before = d.copy(), e.copy()
    m.update(d, e, SCALE, 0.04)
    assert np.array_equal(d, d_before) and np.array_equal(e, e_before)
    assert np.array_equal(d0, d0_before)   # start() copied, did not alias


def test_the_hook_is_the_last_thing_call_does():
    """Structural pin for the LOGGING-ONLY claim, since the trainer is not
    run-to-run reproducible on a GPU and a whole-run diff cannot prove it:
    the meter is driven by the final statement before `return r`, so no term
    of the reward and no piece of episode state can be computed from it."""
    import ast
    src = (ROOT / "python" / "surfgym" / "rewards.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "RaceReward")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    last, prev = fn.body[-1], fn.body[-2]
    assert isinstance(last, ast.Return) and last.value.id == "r"
    assert isinstance(prev, ast.If)
    assert "self._dip" in ast.unparse(prev.test)
    # and nothing else in __call__ mentions the meter
    hits = [n for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and n.attr == "_dip"]
    assert len(hits) == 2, f"_dip referenced {len(hits)}x inside __call__"


def test_off_allocates_nothing_and_logs_nothing(goal_box):
    rw = _race(goal_box, dip=False)
    core = FakeCore()
    for d in range(20000, 100, -500):
        _step(rw, _at(core, float(d)))
    assert rw.dip is False
    assert rw._dip is None
    assert rw.pop_dip_raw() is None
    # and the summary of "no meter at all" is nine blanks, not a crash
    s = dip_summary(merge_dip_raw([]))
    assert all(np.isnan(s[k]) for k in DIP_KEYS)


def test_a_per_env_goal_potential_degrades_to_logging_nothing(goal_box):
    """``--goals`` + a euclid field re-assigns the goal WITHOUT an episode
    end, so every reassignment would read as a fabricated dip.  Refuse to
    log rather than log nonsense."""
    rw = _race(goal_box, dip=True, d0_per_env=True)
    assert rw.dip is False
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    _step(rw, _at(core, 21000.0))
    assert rw.pop_dip_raw() is None


def test_it_works_with_and_without_the_ratchet(goal_box):
    """The dip machinery generalises ``--race-ratchet``'s record but does
    not depend on the flag: the SAME dips are reported either way (the
    ratchet changes what is PAID, not what is measured)."""
    def run(ratchet):
        rw = _race(goal_box, dip=True, ratchet=ratchet)
        _drive(rw, PATH, ENDS)
        return dip_summary(rw.pop_dip_raw())

    on, off = run(True), run(False)
    for k in DIP_KEYS:
        assert (on[k] == pytest.approx(off[k])
                or (np.isnan(on[k]) and np.isnan(off[k]))), k


def test_the_arc_reward_still_meters_the_geodesic(goal_box):
    """``--race-arc`` replaces the shaping term but still samples ``d``
    every call, so the diagnostic keeps working - it is defined on the
    geodesic, not on whatever the objective happens to be."""
    class FakeArc:
        def __init__(self):
            self.arc = np.zeros(1, np.float64)

        def reset(self, pos, mask=None):
            pass

        def advance(self, pos):
            return np.zeros(1, np.float64), np.ones(1, bool)

    rw = _race(goal_box, dip=True, arc=FakeArc(), arc_scale=1.0)
    core = FakeCore()
    _step(rw, _at(core, 20000.0))
    _step(rw, _at(core, 10000.0))
    _step(rw, _at(core, 11000.0))
    _step(rw, _at(core, 9000.0))
    raw = rw.pop_dip_raw()
    assert len(raw["surv_depth"]) == 1
    assert raw["surv_depth"][0] == pytest.approx(1000.0 * SCALE, rel=1e-3)


# --------------------------------------------------------------------------
# 5. the fleet pooling and the trainer's column contract
# --------------------------------------------------------------------------
def test_fleet_pooling_is_a_concatenation_not_a_mean_of_means():
    """Two maps of very different length pool by CONCATENATING the raw
    per-dip arrays: depth is already in reward units, so a p90 over the
    pool is the p90 the columns claim to be."""
    a = {"surv_depth": np.array([1.0, 2.0, 9.0]),
         "surv_secs": np.array([0.1, 0.2, 0.9]),
         "fail_depth": np.array([4.0]), "fail_secs": np.array([0.4]),
         "term_depth": np.array([0.0, 4.0]), "n_ended": 2}
    b = {"surv_depth": np.array([3.0]), "surv_secs": np.array([0.3]),
         "fail_depth": np.zeros(0), "fail_secs": np.zeros(0),
         "term_depth": np.array([0.0]), "n_ended": 1}
    s = dip_summary(merge_dip_raw([a, b]))
    assert s["max_survived_depth"] == 9.0
    assert s["max_survived_secs"] == pytest.approx(0.9)
    assert s["survived_per_ep"] == pytest.approx(4 / 3)
    assert s["fail_frac"] == pytest.approx(1 / 3)
    assert s["fail_depth"] == pytest.approx(4.0)
    assert s["p90_survived_depth"] == pytest.approx(
        float(np.percentile([1.0, 2.0, 9.0, 3.0], 90)))
    assert s["p50_term_depth"] == pytest.approx(
        float(np.percentile([0.0, 4.0, 0.0], 50)))


def test_the_trainer_writes_exactly_these_columns():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert 'CSV_COLS += [f"dip/{k}" for k in DIP_KEYS]' in src
    assert "dip_stats = fleet.pop_dip_stats()" in src
    assert "dip=not args.no_dip_diag" in src
    assert '"dip_diag": not args.no_dip_diag,' in src
    assert DIP_KEYS == ("max_survived_depth", "p90_survived_depth",
                        "max_survived_secs", "survived_per_ep",
                        "fail_depth", "fail_secs", "fail_frac",
                        "p50_term_depth", "p90_term_depth")
