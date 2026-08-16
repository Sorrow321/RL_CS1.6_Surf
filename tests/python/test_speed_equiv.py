"""--speed-equiv: speed folded into the shaping potential must telescope
(loops in position AND speed net zero) and the death refund must make
"accelerate and die" worthless. Plus the --int-speed count key."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from surfgym.core import STATE_DTYPE
from surfgym.rewards import RaceReward


class _FlatField:
    def sample(self, pos):
        return np.zeros(len(np.atleast_2d(pos)))


class _FakeCore:
    def __init__(self, n=2):
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)


def _rr(**kw):
    kw.setdefault("time_pen", 0.0)
    kw.setdefault("stall_ticks", 10**9)
    return RaceReward(_FlatField(), scale=1.0, **kw)


def _step(rr, core, done=None, trunc=None):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    done = z.copy() if done is None else np.asarray(done, np.uint8)
    trunc = z.copy() if trunc is None else np.asarray(trunc, np.uint8)
    return rr(None, None, None, np.zeros(n, np.float32), done, trunc, core)


def test_speed_loop_nets_zero():
    core = _FakeCore(1)
    rr = _rr(speed_equiv=2.0)
    _step(rr, core)                          # first call = on_reset, s=0
    total = 0.0
    for s in (1000.0, 1000.0, 0.0):          # up, hold, back down
        core.states_view["velocity"][:, 0] = s
        total += float(_step(rr, core)[0])
    assert abs(total) < 1e-4                 # the potential telescopes


def test_accelerate_and_die_nets_zero():
    core = _FakeCore(1)
    rr = _rr(speed_equiv=2.0)
    _step(rr, core)
    core.states_view["velocity"][:, 0] = 1000.0
    gain = float(_step(rr, core)[0])         # +scale*beta*1000
    assert abs(gain - 2000.0) < 1e-3
    core.states_view["velocity"][:, 0] = 0.0     # post-death spawn state
    dead = float(_step(rr, core, done=[1.0])[0])
    assert abs(dead + 2000.0) < 1e-3         # refund of the cached credit
    assert abs(gain + dead) < 1e-3           # episode speed credit = 0


def test_goal_keeps_the_credit():
    core = _FakeCore(1)
    rr = _rr(speed_equiv=2.0, success_bonus=50.0)
    _step(rr, core)
    core.states_view["velocity"][:, 0] = 1000.0
    _step(rr, core)
    core.goal_hits[:] = 1
    core.states_view["velocity"][:, 0] = 0.0
    r = float(_step(rr, core, done=[1.0])[0])
    assert abs(r - 50.0) < 1e-3              # bonus only, NO refund on goal


def test_off_is_bit_identical():
    core = _FakeCore(1)
    a, b = _rr(speed_equiv=0.0), _rr(speed_equiv=0.0)
    _step(a, core); _step(b, core)
    core.states_view["velocity"][:, 0] = 3000.0
    ra, rb = _step(a, core), _step(b, core)
    assert np.array_equal(ra, rb) and float(ra[0]) == 0.0


def test_int_speed_key_factorization():
    rr = _rr(int_coef=0.1, int_speed=4)
    rr._mins = np.zeros(3)
    rr._dims = (100, 100, 100)
    s = np.zeros(3, STATE_DTYPE)
    s["origin"][:, 0] = 1000.0
    s["velocity"][:, 0] = [0.0, 1500.0, 3900.0]   # bins 0, 1, 3 at 1000u width
    k = rr._cells(s)
    assert len(set(k.tolist())) == 3
    base = _rr(int_coef=0.1, int_speed=0)
    base._mins = np.zeros(3)
    base._dims = (100, 100, 100)
    k0 = base._cells(s[:1])
    assert k[0] == k0[0] * 4                 # bucket 0 of the same cell
