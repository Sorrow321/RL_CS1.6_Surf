"""``--demo-grow S``: a backward curriculum ANCHORED at the goal end.

The stock ``DemoCurriculum`` is Salimans & Chen's sliding window: starts
are drawn from ``[tau-D+1, tau]`` and tau retreats ONE state per advance.
Two properties of that rule are wrong for a 9,138-state spine inside a
2-hour budget - it stops sampling the end of the spine once tau has moved
(the policy can forget the part it already had), and one state per advance
against the hardcoded 20-iteration cooldown covers ~400 states, so the
curriculum can never reach the map start at all.

``--demo-grow S`` changes exactly two things and nothing else: the spawn
range becomes ``[tau, n-1]`` (anchored at the goal, widening backward), and
tau retreats S states per advance. The ADVANCE CRITERION stays local - the
finish rate is scored over the frontier band ``[tau, tau+D-1]``, never over
the widened range, which would be dominated by the easy goal-adjacent part
and would advance on its own success.

What this file pins:

1. FLAG OFF is bit-identical - same tau trajectory, same drawn indices,
   against the pre-flag rule re-implemented here;
2. ON, starts are anchored at the goal end and widen backward as tau moves;
3. ON, tau retreats by S per advance and backs off by S;
4. the advance is scored on the FRONTIER BAND only: success recorded deep
   in the already-mastered tail does NOT advance the curriculum;
5. tau saturates at 0 (the whole spine) and never runs past either end;
6. the 95/5 split - fresh_frac = 1 - respawn_frac - is exact;
7. ``region_report`` reports per-decile episodes and finish rate.

    python -m pytest tests/python/test_demo_grow.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE
from surfgym.respawn import DemoCurriculum


def _spine(n=1000):
    s = np.zeros(n, STATE_DTYPE)
    # distinct origins: DemoCurriculum keys its index table on them
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 10.0
    return s


def _start_pool(n=4):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = -1000.0
    return s


def _drive(dc, n_iter, win_rate, pool_size=512, fresh_frac=0.05, seed=0):
    """Feed the curriculum ``win_rate`` finishes from every start it draws."""
    rng = np.random.default_rng(seed)
    taus = []
    for _ in range(n_iter):
        pool = dc.build_pool(_start_pool(), pool_size=pool_size,
                             fresh_frac=fresh_frac)
        idx = dc.match(pool["origin"])
        idx = idx[idx >= 0]
        wins = rng.random(len(idx)) < win_rate
        dc.note_outcomes(idx, wins)
        taus.append(dc.tau)
    return taus


# ---------------------------------------------------------------- 1. off

def _legacy_move(dc):
    """The pre-flag ``_move`` verbatim, for the bit-identity check."""
    lo = max(0, dc.tau - dc.D + 1)
    ep = float(dc.ep[lo:dc.tau + 1].sum())
    win = float(dc.win[lo:dc.tau + 1].sum())
    r = win / max(ep, 1e-9)
    dc._cool = max(0, dc._cool - 1)
    moved = 0
    if ep >= dc.min_ep and dc._cool == 0:
        if r >= dc.rate and dc.tau > 0:
            dc.tau -= 1
            moved = -1
        elif r < dc.rate and dc.tau < dc.n - 1:
            dc.tau += 1
            moved = +1
        if moved:
            dc._cool = 20
    return max(0, dc.tau - dc.D + 1), dc.tau


def test_flag_off_is_bit_identical():
    """Default grow=0 reproduces the sliding rule exactly - tau path and
    the very indices drawn, which is the only thing that reaches the sim."""
    S = _spine()
    a = DemoCurriculum(S, window=10, rate=0.2, min_ep=50.0, seed=7)
    b = DemoCurriculum(S, window=10, rate=0.2, min_ep=50.0, seed=7)
    assert a.grow == 0
    rng = np.random.default_rng(3)
    for _ in range(400):
        # A: the shipped path
        pool = a.build_pool(_start_pool(), pool_size=256, fresh_frac=0.05)
        got = a.match(pool["origin"])
        # B: legacy _move + legacy draw, from B's own rng (seeded alike)
        lo, hi = _legacy_move(b)
        n_fresh = max(1, int(round(256 * 0.05)))
        want_idx = b.rng.integers(lo, hi + 1, 256 - n_fresh)
        b.rng.integers(0, 4, n_fresh)   # the fresh draw, same call order
        assert a.tau == b.tau
        assert np.array_equal(got[got >= 0], want_idx)
        outcome = rng.random(len(want_idx)) < 0.5
        a.note_outcomes(got[got >= 0], outcome)
        b.note_outcomes(want_idx, outcome)
    assert a.tau == b.tau


def test_flag_off_window_slides_and_leaves_the_end():
    S = _spine()
    dc = DemoCurriculum(S, window=10, rate=0.2, min_ep=50.0, seed=7)
    _drive(dc, 300, win_rate=1.0)
    assert dc.tau < len(S) - 1                      # it advanced
    lo, hi = dc._draw()
    assert hi == dc.tau and lo == max(0, dc.tau - dc.D + 1)
    assert hi < len(S) - 1                          # the END is no longer drawn


# ----------------------------------------------------------------- 2/3. on

def test_grow_anchors_at_the_goal_and_widens_backward():
    S = _spine()
    dc = DemoCurriculum(S, window=64, rate=0.2, min_ep=50.0, seed=7, grow=32)
    assert dc._draw() == (len(S) - 1, len(S) - 1)
    taus = _drive(dc, 300, win_rate=1.0)
    assert taus[-1] < taus[0]                       # retreated
    lo, hi = dc._draw()
    assert hi == len(S) - 1                         # STILL anchored at the goal
    assert lo == dc.tau
    # and the draws really cover the widened range
    pool = dc.build_pool(_start_pool(), pool_size=4096, fresh_frac=0.05)
    idx = dc.match(pool["origin"])
    idx = idx[idx >= 0]
    assert idx.min() >= dc.tau and idx.max() <= len(S) - 1
    assert idx.max() - idx.min() > 4 * 32           # not a 1-state window


def test_grow_steps_by_S_and_backs_off_by_S():
    S = _spine()
    dc = DemoCurriculum(S, window=64, rate=0.2, min_ep=50.0, seed=7, grow=32)
    t0 = dc.tau
    _drive(dc, 300, win_rate=1.0)
    steps = t0 - dc.tau
    assert steps % 32 == 0 and steps > 0            # only ever whole strides
    t1 = dc.tau
    _drive(dc, 300, win_rate=0.0)                   # frontier now fails
    assert dc.tau > t1                              # backed off
    assert (dc.tau - t1) % 32 == 0


def test_advance_is_scored_on_the_frontier_band_only():
    """Wins harvested deep in the mastered tail must not move tau: that is
    exactly the trivial-win trap (CLAUDE.md) in curriculum form."""
    S = _spine()
    dc = DemoCurriculum(S, window=64, rate=0.2, min_ep=50.0, seed=7, grow=32)
    _drive(dc, 300, win_rate=1.0)                   # get tau well off the end
    tau = dc.tau
    assert tau < len(S) - 1 - 64
    tail = np.arange(len(S) - 200, len(S))
    for _ in range(300):
        dc.build_pool(_start_pool(), pool_size=256, fresh_frac=0.05)
        dc.note_outcomes(tail, np.ones(len(tail), bool))
        band = np.arange(dc.tau, min(len(S), dc.tau + 64))
        dc.note_outcomes(band, np.zeros(len(band), bool))
    assert dc.tau >= tau                            # never advanced on the tail


def test_tau_saturates_at_zero():
    S = _spine(300)
    dc = DemoCurriculum(S, window=64, rate=0.2, min_ep=50.0, seed=7, grow=32)
    _drive(dc, 2000, win_rate=1.0)
    assert dc.tau == 0
    assert dc._draw() == (0, len(S) - 1)            # uniform over the spine
    pool = dc.build_pool(_start_pool(), pool_size=4096, fresh_frac=0.05)
    idx = dc.match(pool["origin"])
    assert idx[idx >= 0].min() >= 0


def test_fresh_share_is_the_start_spawn():
    """fresh_frac = 1 - respawn_frac: at 0.95 exactly 5% of the pool is the
    map start, which is what the arm's 95/5 split rests on."""
    S = _spine()
    dc = DemoCurriculum(S, window=64, rate=0.2, min_ep=50.0, seed=7, grow=32)
    pool = dc.build_pool(_start_pool(), pool_size=4096, fresh_frac=0.05)
    idx = dc.match(pool["origin"])
    assert int((idx < 0).sum()) == 205              # round(4096*0.05)
    assert abs(float((idx < 0).mean()) - 0.05) < 0.002


def test_region_report():
    S = _spine(1000)
    dc = DemoCurriculum(S, window=64, rate=0.2, min_ep=50.0, seed=7, grow=32)
    dc.note_outcomes(np.array([950, 951, 952]), np.array([True, True, False]))
    rep = dc.region_report(bins=10)
    assert rep.startswith("demo regions")
    assert "90%:3/67%" in rep
    assert "0%:0/0%" in rep
