"""--respawn-greedy-plateau S (2026-10-31, autoresearch uf2r2b1): the greedy-line pool also keeps
N evenly spaced states of every MID-EPISODE stall of S seconds or more of EVERY greedy episode - a
stretch over which the running best field distance did not improve by 64 u and after which it did -
a FINISHED episode included. surf_unitfarmer2's finishing greedy laps its start pit twice (~9 s with
no new best) before it climbs out; without the flag a finishing line gives the pool nothing.

(a) plateaus(): the stalls that end in a gain, at least min_ticks long; small gains (< eps) do not
    end a stall; a stall that never ends is not one; off-field ticks are ignored.
(b) The pool: a FINISHED episode with a plateau adds N evenly spaced states of it (none without the
    flag - the pool before this change); a failed one adds them beside its death window; plateaus
    shorter than S add nothing; bad S / N are refused.
(c) The recorder lists the new config keys as training-only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE                               # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool, plateaus           # noqa: E402


def test_plateaus_find_the_stalls_that_end_in_a_gain():
    # d falls 0..99, sits flat (wanders, never 64 u better) 100..399, falls again 400..499
    d = np.concatenate([10000.0 - 10.0 * np.arange(100),
                        9010.0 + 50.0 * np.sin(np.arange(300)),
                        8900.0 - 10.0 * np.arange(100)])
    out = plateaus(d, 200, 64.0)
    assert len(out) == 1
    a, b = out[0]
    assert 95 <= a <= 100 and 400 <= b <= 410
    # too short for 400 ticks: nothing
    assert plateaus(d, 400, 64.0) == []


def test_small_gains_do_not_end_a_stall_and_an_endless_stall_is_not_one():
    # a slow creep of 1 u per tick for 300 ticks: 300 u in all, but never 64 u past the anchor
    # inside 60 ticks... it IS a gain of >= 64 u after 64 ticks, so no plateau of 100 ticks
    creep = 10000.0 - np.arange(300, dtype=np.float64)
    assert plateaus(creep, 100, 64.0) == []
    # flat to the end (the line stops for good): --respawn-greedy-stuck's stretch, not this
    flat = np.concatenate([10000.0 - 10.0 * np.arange(50), np.full(500, 9510.0)])
    assert plateaus(flat, 100, 64.0) == []
    # off-field ticks (inf / nan) are not gains and do not break the scan
    d = np.concatenate([10000.0 - 10.0 * np.arange(50), np.full(200, np.inf),
                        9000.0 - np.arange(10, dtype=np.float64)])
    d[60] = np.nan
    out = plateaus(d, 150, 64.0)
    assert len(out) == 1 and out[0][1] == 250


def _ep(d):
    """States whose origin x is the field distance the test's d_fn reads; tick = index."""
    rows = np.zeros(len(d), STATE_DTYPE)
    rows["origin"][:, 0] = np.asarray(d, np.float32)
    rows["origin"][:, 1] = np.arange(len(d), dtype=np.float32)      # distinct, = tick
    rows["tick"] = np.arange(len(d)) + 5
    rows["stuck_ticks"] = 2
    return rows


def _build(pool, eps, ends, tick_ms=10.0):
    hook = pool.hook()
    t = 0
    for ep in eps:
        for i in range(len(ep)):
            hook(t, ep[i:i + 1], np.zeros(1), np.array([i == len(ep) - 1]), np.array([False]))
            t += 1
    return pool.build([{"end": e} for e in ends], tick_ms,
                      d_fn=lambda o: np.asarray(o, np.float64)[:, 0])


def _lap_line():
    """A finishing line: progress, a 600-tick lap (no gain), progress to the end."""
    return np.concatenate([10000.0 - 10.0 * np.arange(100),
                           9010.0 + 40.0 * np.sin(np.arange(600) / 20.0),
                           8900.0 - 10.0 * np.arange(300)])


def test_a_finished_episode_adds_its_plateau_only_with_the_flag():
    ep = _ep(_lap_line())
    off = GreedyLinePool(0.25, window=(0.25, 2.0), stuck=200)
    _build(off, [ep], ["done"])
    assert off.size == 0                                  # today: a finisher gives nothing
    on = GreedyLinePool(0.25, window=(0.25, 2.0), stuck=200, plateau=2.0, plateau_n=50)
    info = _build(on, [ep], ["done"])
    assert on.size == 50
    ticks = np.sort(np.asarray(on.rows["origin"][:, 1], np.int64))
    assert ticks[0] >= 95 and ticks[-1] <= 705             # inside the lap
    gaps = np.diff(ticks)
    assert gaps.max() - gaps.min() <= 1                    # evenly spaced
    assert (on.rows["tick"] == 0).all() and (on.rows["stuck_ticks"] == 0).all()
    assert len(info["plateaus_s"]) == 1
    assert "mid-episode plateaus" in on.line()


def test_a_short_plateau_adds_nothing_and_a_failed_episode_keeps_its_death_window():
    ep = _ep(_lap_line())
    on = GreedyLinePool(0.25, window=(0.25, 2.0), plateau=7.0, plateau_n=50)
    _build(on, [ep], ["done"])                             # the lap is 6 s at 10 ms
    assert on.size == 0
    base = GreedyLinePool(0.25, window=(0.25, 2.0))
    _build(base, [ep], ["fail"])
    with_pl = GreedyLinePool(0.25, window=(0.25, 2.0), plateau=2.0, plateau_n=50)
    _build(with_pl, [ep], ["fail"])
    assert base.size == 175                                # the death window 2.0..0.25 s
    assert with_pl.size == base.size + 50                  # + the lap (disjoint from the window)


def test_off_is_the_pool_as_before_on_failed_and_stuck_episodes():
    rng = np.random.default_rng(3)
    eps = [_ep(10000.0 - np.cumsum(rng.uniform(-5, 10, 900))) for _ in range(3)]
    ends = ["fail", "trunc", "done"]
    a = GreedyLinePool(0.25, window=(0.25, 2.0), stuck=100, seed=4)
    b = GreedyLinePool(0.25, window=(0.25, 2.0), stuck=100, seed=4, plateau=0.0)
    _build(a, eps, ends)
    _build(b, eps, ends)
    assert a.size == b.size and np.array_equal(a.rows, b.rows)
    assert "plateaus_s" not in b.info


def test_bad_values_are_refused():
    with pytest.raises(ValueError):
        GreedyLinePool(0.25, plateau=-1.0)
    with pytest.raises(ValueError):
        GreedyLinePool(0.25, plateau=2.0, plateau_n=0)


def test_the_recorder_lists_the_keys_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    m = re.search(r"TRAIN_ONLY = frozenset\(\{(.*?)\}\)", src, re.S)
    assert m and '"respawn_greedy_plateau"' in m.group(1) and '"respawn_greedy_plateau_n"' in m.group(1)
