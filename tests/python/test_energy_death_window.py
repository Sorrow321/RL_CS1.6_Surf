"""--energy-death-window W (zlr1b3, 2026-10-31): with --energy-pot on, a DEATH (done, not a
finish, not a truncation) takes back what the energy term paid over the episode's last W
seconds (its last N calls) when that was a net gain - the whole episode's pay when it is
younger than W. Energy strafed up on the way into a fatal fall is not kept: the
--death-charge-window rule applied to the energy term.

(a) a death after a steady gain takes back exactly the last N calls' pay; an older gain is kept;
(b) a young episode gives back all it was paid; a net loss in the window gives nothing back;
(c) a finish and a truncation give nothing back;
(d) the next episode starts with a clean slate (its own pay only);
(e) off is byte-identical to a reward that never heard of the flag;
(f) the recorder lists it as training-only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
from surfgym.core import STATE_DTYPE            # noqa: E402
from surfgym.rewards import RaceReward          # noqa: E402

G = 800.0
K = 10.0            # reward per 1000 u of energy height
N = 5               # calls in the window


class _FlatField:
    reach_max = 1e6 + 16.0
    cell = 32.0

    def sample(self, pos):
        return np.full(len(np.atleast_2d(pos)), 5000.0)


class _FakeCore:
    def __init__(self, n=1):
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=G))

    def h(self, h):
        """Set the energy height through the speed (z = 0)."""
        hh = np.atleast_1d(np.asarray(h, np.float64))
        self.states_view["velocity"][:, 0] = np.sqrt(np.maximum(2.0 * G * hh, 0.0)).astype(np.float32)
        self.states_view["velocity"][:, 1:] = 0.0
        return self


def _rr(core, n_window):
    rr = RaceReward(_FlatField(), scale=0.0, time_pen=0.0, stall_ticks=0, stall_eps=32.0,
                    success_bonus=0.0)
    rr.energy_pot = K
    rr.en_death_n = int(n_window)
    rr.on_reset(core)
    return rr


def _step(rr, core, done=0, trunc=0, goal=0):
    n = core.num_envs
    core.goal_hits[:] = goal
    return rr(None, None, None, np.zeros(n, np.float32), np.full(n, done, np.uint8),
              np.full(n, trunc, np.uint8), core)


def _episode(hs, n_window, end="done", next_h=0.0):
    """Drive one env through energy heights hs (one call each, the first is the spawn's), then
    end it the given way (the end call's state is the NEXT spawn at next_h). -> (rewards, end r)"""
    core = _FakeCore().h(hs[0])
    rr = _rr(core, n_window)
    rs = []
    for h in hs[1:]:
        core.h(h)
        rs.append(float(_step(rr, core)[0]))
    core.h(next_h)
    kw = {"done": dict(done=1), "goal": dict(done=1, goal=1), "trunc": dict(trunc=1)}[end]
    r_end = float(_step(rr, core, **kw)[0])
    return rs, r_end, rr, core


def test_a_death_after_a_steady_gain_takes_back_the_last_window():
    hs = list(np.arange(0.0, 1000.0, 50.0))          # +50 u per call, 19 paid calls
    rs, r_end, _, _ = _episode(hs, N)
    # the energy term anchors on its FIRST call (h 50), so the paid calls are 50 -> 950
    assert sum(rs) == pytest.approx(K / 1000.0 * 900.0, rel=1e-5)
    # the death call pays nothing itself and takes back the last N-1 calls' pay (the ring holds
    # the cum of N calls ago; the death call is the N-th)
    assert r_end == pytest.approx(-K / 1000.0 * 50.0 * (N - 1), rel=1e-5)


def test_a_young_episode_gives_back_everything_and_a_loss_gives_nothing():
    rs, r_end, _, _ = _episode([0.0, 100.0, 200.0], N)
    assert r_end == pytest.approx(-sum(rs), rel=1e-5)
    rs, r_end, _, _ = _episode([0.0, 400.0, 400.0, 400.0, 300.0, 200.0, 100.0, 50.0], N)
    # the window (last N-1 = 4 paid calls) is a net LOSS: nothing back
    assert r_end == pytest.approx(0.0, abs=1e-7)


def test_a_finish_and_a_truncation_give_nothing_back():
    hs = list(np.arange(0.0, 1000.0, 50.0))
    for end in ("goal", "trunc"):
        _, r_end, _, _ = _episode(hs, N, end=end)
        assert r_end == pytest.approx(0.0, abs=1e-7), end


def test_the_next_episode_starts_clean():
    hs = list(np.arange(0.0, 1000.0, 50.0))
    _, _, rr, core = _episode(hs, N, next_h=0.0)
    core.h(30.0)
    r1 = float(_step(rr, core)[0])                   # +30 u from the new spawn
    core.h(0.0)
    r_end = float(_step(rr, core, done=1)[0])        # dies at once: gives back only its own 30
    assert r1 == pytest.approx(K / 1000.0 * 30.0, rel=1e-5)
    assert r_end == pytest.approx(-r1, rel=1e-5)


def test_off_is_byte_identical():
    rng = np.random.default_rng(11)
    n = 6
    ca, cb = _FakeCore(n), _FakeCore(n)
    ra, rb = _rr(ca, 0), _rr(cb, 0)
    for t in range(500):
        h = np.abs(rng.normal(500.0, 300.0, n))
        done = (rng.random(n) < 0.02).astype(np.uint8)
        ca.h(h)
        cb.h(h)
        for c in (ca, cb):
            c.goal_hits[:] = 0
        a = ra(None, None, None, np.zeros(n, np.float32), done, np.zeros(n, np.uint8), ca)
        b = RaceReward.__call__(rb, None, None, None, np.zeros(n, np.float32), done,
                                np.zeros(n, np.uint8), cb)
        assert np.array_equal(a, b)
    assert ra._en_cum is None and rb._en_cum is None


def test_record_ckpt_lists_it_as_training_only():
    import record_ckpt
    assert "energy_death_window" in record_ckpt.TRAIN_ONLY
