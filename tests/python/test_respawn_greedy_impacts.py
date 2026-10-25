"""--respawn-greedy-impacts K (pbr4b4, 2026-10-25): the greedy-line pool also holds the window's
HI..LO seconds before each unfinished greedy episode's K largest drops of the energy height
z + |v|^2/2g. Synthetic episodes (a flight with two hard landings and a death), and the flag off
reproduces the old pool exactly.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/python/test_respawn_greedy_impacts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                                # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool, energy_drops       # noqa: E402


def _flight(n=2000, drops=((600, 300.0), (1300, 120.0), (1310, 90.0)), small=(900, 20.0)):
    """|v| 800 u/s level flight (h = 400 at z 0), the speed cut at each (tick, u of h) drop;
    a duck-sized 20 u wobble elsewhere must not count. origin x = the tick (an index tag)."""
    ep = np.zeros(n, STATE_DTYPE)
    h = np.full(n, 400.0)
    for t, dh in drops:
        h[t:] -= dh
    t, dh = small
    h[t:t + 5] -= dh
    v = np.sqrt(np.maximum(h, 0.0) * 1600.0)
    ep["velocity"][:, 0] = v
    ep["origin"][:, 0] = np.arange(n)
    return ep


def _build(p, eps, ends):
    p._eps = list(eps)
    p._cur = []
    p._par = None
    return p.build([{"end": e} for e in ends], 10.0)


def test_energy_drops_rule():
    ep = _flight()
    # the two big drops, each at the first tick of its 0.1 s window; the second landing (1300)
    # and its follow-up (1310) are < 1 s apart and only the larger counts; the 20 u wobble is
    # under the 40 u floor
    assert energy_drops(ep, 5, 40.0, 800.0, 100.0, len(ep)) == [590, 1290]
    assert energy_drops(ep, 1, 40.0, 800.0, 100.0, len(ep)) == [590]
    # nothing at or after ``before``
    assert energy_drops(ep, 5, 40.0, 800.0, 100.0, 1000) == [590]
    assert energy_drops(ep, 5, 400.0, 800.0, 100.0, len(ep)) == []


def test_impact_rows_are_the_approaches():
    ep = _flight()
    off = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0)
    on = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0, impacts=2)
    _build(off, [ep], ["fail"])
    _build(on, [ep], ["fail"])
    t_off = set(off.rows["origin"][:, 0].astype(int))
    t_on = set(on.rows["origin"][:, 0].astype(int))
    assert t_off <= t_on
    extra = t_on - t_off
    # [ti - 200, ti - 25) before each drop at 590 and 1290
    want = set(range(390, 565)) | set(range(1090, 1265))
    assert extra == want
    assert on.info["impacts_s"] == [5.9, 12.9]
    assert "energy drops" in on.line() and "energy drops" not in off.line()
    # a finished episode contributes nothing, with or without the flag
    on2 = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0, impacts=2)
    _build(on2, [ep], ["done"])
    assert on2.size == 0


def test_flag_off_is_unchanged():
    ep = _flight()
    a = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0)
    b = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0, impacts=0)
    _build(a, [ep], ["fail"])
    _build(b, [ep], ["fail"])
    assert np.array_equal(a.rows, b.rows) and "impacts_s" not in b.info
    with pytest.raises(ValueError):
        GreedyLinePool(0.2, window=(0.25, 2.0), impacts=-1)
