"""--death-charge under --race-ratchet, and its window (uf2r1b1, 2026-10-30).

1. THE SPAWN GAP (a fix): on the call an episode dies its row already holds the NEXT episode's spawn,
   and the ratchet had min-ed the record with that spawn's distance before the charge read it. A
   deeper next spawn - a reservoir row, or a --respawn-backward-mix goal-anchored row 256-1,532 u
   from the finish - charged the dying episode for ground it never covered, up to the whole map.
   On surf_unitfarmer2 that put -7..-12 reward per training episode into uf2r1b1_dce / _nov. The
   charge now reads the record as it stood before the death call (`_rec_boot`).
2. --death-charge-window W: a death takes back only what the ratchet paid in its last W seconds;
   progress the episode survived past for W s is kept. 0 = the whole bank.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.rewards import RaceReward                            # noqa: E402
from test_arc_death_charge import _FakeCore, _step                # noqa: E402


class _LineField:
    """d = x: progress is moving towards x 0."""
    reach_max = 1e9
    cell = 32.0

    def sample(self, p):
        return np.asarray(p, np.float64)[:, 0].astype(np.float32)


def _rr(window=0.0, kappa=1.0, x0=10000.0, n=1):
    core = _FakeCore(n).at([x0] * n)
    rr = RaceReward(_LineField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9,
                    max_step=1e9, ratchet=True, ratchet_d0=x0, ng_d0=x0,
                    death_charge=kappa, death_charge_window=window, tick_ms=10.0)
    rr.on_reset(core)
    return rr, core


def _ride(rr, core, xs):
    """one call per x; returns the summed reward"""
    return sum(float(_step(rr, core.at([x]))[0]) for x in xs)


def _die(rr, core, x_next):
    """the death call: the row already holds the next episode's spawn at x_next"""
    return float(_step(rr, core.at([x_next]), done=1)[0])


def test_the_death_charge_takes_back_exactly_what_was_paid():
    rr, core = _rr()
    paid = _ride(rr, core, np.linspace(9990, 9000, 100))
    assert paid == pytest.approx(1000.0)
    # the next spawn is at the start again: the charge is the bank
    assert _die(rr, core, 10000.0) == pytest.approx(-1000.0)


def test_a_deeper_next_spawn_is_not_charged_to_the_dying_episode():
    rr, core = _rr()
    _ride(rr, core, np.linspace(9990, 9000, 100))
    # the next episode spawns 8,000 u deeper (a goal-anchored row): before the fix this charged
    # 9,000 for a 1,000 bank
    assert _die(rr, core, 1000.0) == pytest.approx(-1000.0)
    # and the new episode's record starts at its own spawn: nothing is paid for the jump
    assert _ride(rr, core, [1000.0, 990.0]) == pytest.approx(10.0)


def test_window_keeps_progress_the_episode_survived_past():
    rr, core = _rr(window=1.0)                       # 1 s = 100 calls at 10 ms, every 1
    assert rr._dc_n == 100
    _ride(rr, core, np.linspace(9990, 9000, 50))     # 1,000 paid, then...
    _ride(rr, core, [9000.0] * 200)                  # ...2 s alive with no progress
    _ride(rr, core, np.linspace(8990, 8800, 20))     # 200 more in the last 0.2 s
    # only the last 200 are inside the window
    assert _die(rr, core, 10000.0) == pytest.approx(-200.0)


def test_window_an_episode_younger_than_it_gives_everything_back():
    rr, core = _rr(window=1.0)
    _ride(rr, core, np.linspace(9990, 9500, 30))
    assert _die(rr, core, 10000.0) == pytest.approx(-500.0)


def test_window_restarts_with_the_next_episode():
    rr, core = _rr(window=1.0)
    _ride(rr, core, np.linspace(9990, 9000, 50))
    _ride(rr, core, [9000.0] * 200)
    assert _die(rr, core, 6000.0) == pytest.approx(0.0)    # nothing in the last second
    # the new episode, spawned at 6,000: its whole (young) bank is charged, nothing of the last
    assert _ride(rr, core, np.linspace(5990, 5900, 10)) == pytest.approx(100.0)
    assert _die(rr, core, 10000.0) == pytest.approx(-100.0)


def test_window_is_per_env():
    rr, core = _rr(window=1.0, n=2)
    for x in np.linspace(9990, 9000, 50):                  # both progress
        _step(rr, core.at([x, x]))
    for _ in range(200):                                    # both hold
        _step(rr, core.at([9000.0, 9000.0]))
    for x in np.linspace(8990, 8900, 10):                  # env 1 progresses again
        _step(rr, core.at([9000.0, x]))
    core.at([10000.0, 10000.0])
    core.goal_hits[:] = 0
    r = rr(None, None, None, np.zeros(2, np.float32), np.array([1, 1], np.uint8),
           np.zeros(2, np.uint8), core)
    assert float(r[0]) == pytest.approx(0.0)
    assert float(r[1]) == pytest.approx(-100.0)


def test_a_truncation_and_a_finish_are_not_charged():
    rr, core = _rr(window=1.0)
    _ride(rr, core, np.linspace(9990, 9500, 30))
    assert float(_step(rr, core.at([10000.0]), trunc=1)[0]) == pytest.approx(0.0)
    rr, core = _rr(window=1.0)
    _ride(rr, core, np.linspace(9990, 9500, 30))
    r = float(_step(rr, core.at([10000.0]), done=1, goal=1)[0])
    assert r == pytest.approx(rr.success_bonus)


def test_window_zero_is_the_whole_bank_and_allocates_nothing():
    rr, core = _rr(window=0.0)
    assert rr._rec_ring is None and rr._dc_n == 0
    _ride(rr, core, np.linspace(9990, 9000, 50))
    _ride(rr, core, [9000.0] * 200)
    assert _die(rr, core, 10000.0) == pytest.approx(-1000.0)


def test_refusals():
    core = _FakeCore(1).at([10000.0])
    with pytest.raises(ValueError, match="needs --death-charge"):
        RaceReward(_LineField(), scale=1.0, ratchet=True, ng_d0=1e4, death_charge_window=1.0)
    with pytest.raises(ValueError, match="race-ratchet"):
        RaceReward(_LineField(), scale=1.0, ratchet=False, ng_d0=1e4, death_charge=1.0,
                   death_charge_window=1.0)
    with pytest.raises(ValueError, match=">= 0"):
        RaceReward(_LineField(), scale=1.0, ratchet=True, ng_d0=1e4, death_charge=1.0,
                   death_charge_window=-1.0)
    del core
