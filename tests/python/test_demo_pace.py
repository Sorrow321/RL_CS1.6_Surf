"""--demo-cooldown C / --demo-grow-max G (pbr2b3, 2026-10-24): the backward curriculum's pace.
DemoCurriculum with the defaults moves exactly as before (20-iteration cooldown, fixed step);
cooldown C spaces the moves C iterations apart; grow_max doubles the step on consecutive
advances (S, 2S, ... G) and a backoff moves back by S and resets it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def _spine(n=400):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32)        # distinct origins
    return s


def _drive(cur, outcomes, iters):
    """Each iteration: feed `outcomes(cur)` wins to the band's states, then build a pool
    (which runs _move). Returns tau after each iteration."""
    taus = []
    for _ in range(iters):
        lo, hi = cur._band()
        idx = np.repeat(np.arange(lo, hi + 1), 100)   # >= min_ep 50 even on a 1-state band
        cur.note_outcomes(idx, np.full(len(idx), outcomes(cur), float))
        cur.build_pool(_spine(4)[:1], pool_size=64)
        taus.append(cur.tau)
    return taus


def test_defaults_are_the_old_rule():
    from surfgym.respawn import DemoCurriculum
    a = DemoCurriculum(_spine(), window=5, grow=5, front_frac=0.67)
    b = DemoCurriculum(_spine(), window=5, grow=5, front_frac=0.67, cooldown=20, grow_max=0)
    win = lambda c: 1.0 if c.tau > 250 else 0.0
    assert _drive(a, win, 300) == _drive(b, win, 300)
    t = _drive(DemoCurriculum(_spine(), window=5, grow=5), lambda c: 1.0, 61)
    # one advance of 5 per 20 iterations
    assert t[0] == 394 and t[20] == 389 and t[40] == 384 and t[60] == 379


def test_cooldown_spaces_the_moves():
    from surfgym.respawn import DemoCurriculum
    t = _drive(DemoCurriculum(_spine(), window=5, grow=5, cooldown=4), lambda c: 1.0, 13)
    assert t[0] == 394 and t[4] == 389 and t[8] == 384 and t[12] == 379
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(), window=5, grow=5, cooldown=0)


def test_grow_max_doubles_and_resets():
    from surfgym.respawn import DemoCurriculum
    c = DemoCurriculum(_spine(), window=5, grow=5, cooldown=1, grow_max=40)
    t = _drive(c, lambda cur: 1.0, 6)
    # steps 5, 10, 20, 40, 40, 40
    assert t == [394, 384, 364, 324, 284, 244]
    # a failing band: back by the base step, and the step is reset
    t2 = _drive(c, lambda cur: 0.0, 2)
    assert t2 == [249, 254] and c._step == 5
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(), window=5, grow=5, grow_max=3)
    with pytest.raises(ValueError):
        DemoCurriculum(_spine(), window=5, grow=0, grow_max=10)
