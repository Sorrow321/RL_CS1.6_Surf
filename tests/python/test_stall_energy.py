"""--stall-energy E and --eval-patience-energy E (zlr1b2, 2026-10-31): energy-height records
re-arm the stall / patience clocks.

On a map whose field says nothing while the player pumps a half-pipe for speed (surf_za_loop:
the record rides its first loop ~65 s with NEGATIVE field progress), the training stall rule
(15 s without a field record) and the eval patience rule (30 s) kill every pumping line. The
energy height h = z + |v|^2 / (2 g) is kept by free fall and frictionless surfing and raised only
by what the player adds (air acceleration, a jump), so a new record of h is progress the field
cannot see. Both rules re-arm on a CUMULATIVE rise of the episode's best h by more than E.

(a) RaceReward: E = 0 keeps no state and stalls exactly as before; a pumping env survives the
    window a static env dies in; free fall (h constant) is no re-arm; a saturated pump stalls one
    window after it stops gaining; an ended row re-anchors on the next episode's spawn; the
    per-episode re-arm count is reported only when on.
(b) The eval patience hooks (single and side-by-side): the same rule; energy_eps 0 = the old
    hook.
(c) The recorder lists both keys as training-only.
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

from surfgym.core import STATE_DTYPE          # noqa: E402
from surfgym.rewards import RaceReward        # noqa: E402

G = 800.0


class _FlatField:
    """d = 0 everywhere: the field never shows progress."""
    reach_max = 1e6
    cell = 32.0

    def sample(self, pos):
        return np.zeros(len(np.atleast_2d(pos)))


class _Core:
    def __init__(self, n=1):
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.failed = np.zeros(n, np.uint8)
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=G))

    def put(self, z, vx):
        self.states_view["origin"][:, 2] = np.asarray(z, np.float32)
        v = np.zeros((self.num_envs, 3), np.float32)
        v[:, 0] = np.asarray(vx, np.float32)
        self.states_view["velocity"][:] = v
        return self

    def force_fail(self, mask):
        self.failed |= np.asarray(mask, np.uint8)


def _rr(n, e, stall_ticks=10):
    core = _Core(n).put([0.0] * n, [0.0] * n)
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=stall_ticks,
                    stall_eps=32.0)
    rr.stall_energy = e
    rr.on_reset(core)
    return rr, core


def _step(rr, core, done=None):
    n = core.num_envs
    d = np.zeros(n, np.uint8) if done is None else np.asarray(done, np.uint8)
    return rr(None, None, None, np.zeros(n, np.float32), d, np.zeros(n, np.uint8), core)


def _vx_for(h):
    """the horizontal speed whose kinetic height is h (z = 0)"""
    return float(np.sqrt(2.0 * G * h))


# ==========================================================================
# (a) the training stall rule
# ==========================================================================
def test_zero_is_off_no_state_and_the_old_stall():
    rr, core = _rr(2, 0.0)
    kills = []
    for i in range(1, 25):
        _step(rr, core.put([0.0, 0.0], [_vx_for(10.0 * i), 0.0]))   # env 0 pumps
        m = rr.pop_stall_mask()
        kills.append(None if m is None else m.tolist())
    assert rr._sh_ref is None and rr._sh_best is None
    # both envs stall on the same call (the field rule alone): every 10 calls
    fired = [i for i, k in enumerate(kills) if k is not None]
    assert fired and all(kills[i] == [1, 1] for i in fired)
    assert "stall_en_rearms_per_ep" not in rr.pop_stats()


def test_a_pumping_env_survives_the_window_a_static_one_dies_in():
    rr, core = _rr(2, 32.0)
    dead0, dead1 = [], []
    for i in range(1, 40):
        _step(rr, core.put([0.0, 0.0], [_vx_for(10.0 * i), 0.0]))   # +10 u of h per call
        m = rr.pop_stall_mask()
        if m is not None:
            if m[0]:
                dead0.append(i)
            if m[1]:
                dead1.append(i)
    assert not dead0                     # re-armed every 4 calls (40 u > 32 u)
    assert dead1 and dead1[0] == 10      # the static env: exactly the old window


def test_a_slow_pump_still_rearms_cumulatively():
    """1 u of h per call (a weak strafe) never beats a per-call threshold, but the rule is
    cumulative: 33 calls later it has risen 33 u > 32 u and the timer re-arms."""
    rr, core = _rr(1, 32.0, stall_ticks=40)
    dead = []
    for i in range(1, 200):
        _step(rr, core.put([0.0], [_vx_for(1.0 * i)]))
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            dead.append(i)
    assert not dead


def test_free_fall_keeps_h_and_is_no_rearm():
    """GoldSrc's half-step gravity keeps z + v^2/2g in free fall: a fall is not a pump."""
    rr, core = _rr(1, 32.0, stall_ticks=50)
    z, vz, dt = 5000.0, 0.0, 0.01
    dead = []
    for i in range(1, 120):
        core.states_view["origin"][:, 2] = np.float32(z)
        core.states_view["velocity"][:] = np.asarray([[300.0, 0.0, vz]], np.float32)
        _step(rr, core)
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            dead.append(i)
        vh = vz - 0.5 * G * dt
        z += vh * dt
        vz = vh - 0.5 * G * dt
    assert dead and dead[0] == 50


def test_a_saturated_pump_stalls_one_window_after_it_stops():
    rr, core = _rr(1, 32.0, stall_ticks=10)
    dead = []
    for i in range(1, 60):
        h = 10.0 * min(i, 20)            # pumps to h 200 by call 20, then holds
        _step(rr, core.put([0.0], [_vx_for(h)]))
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            dead.append(i)
    # the last re-arm: the first call with best h > ref + 32 at or before call 20
    assert dead and 26 <= dead[0] <= 30


def test_an_ended_row_reanchors_on_the_next_spawn():
    """A new episode spawned with LESS energy than the last one's record must still be able to
    re-arm by its own pumping, and one spawned with MORE must not re-arm for free."""
    rr, core = _rr(1, 32.0, stall_ticks=10)
    for i in range(1, 9):
        _step(rr, core.put([0.0], [_vx_for(50.0 * i)]))                # h up to 400
    _step(rr, core.put([0.0], [0.0]), done=[1])                         # next spawn at rest
    assert rr._sh_ref[0] == pytest.approx(0.0, abs=1e-3)
    dead = []
    for i in range(1, 30):
        _step(rr, core.put([0.0], [_vx_for(10.0 * i)]))                 # pumps from 0
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            dead.append(i)
    assert not dead
    _step(rr, core.put([5000.0], [0.0]), done=[1])                      # a high spawn
    dead = []
    for i in range(1, 15):
        _step(rr, core.put([5000.0], [0.0]))                            # sits there
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            dead.append(i)
    assert dead and dead[0] == 10


def test_the_rearm_count_is_reported_when_on():
    rr, core = _rr(1, 32.0, stall_ticks=1000)
    for i in range(1, 41):
        _step(rr, core.put([0.0], [_vx_for(10.0 * i)]))
    _step(rr, core.put([0.0], [0.0]), done=[1])
    st = rr.pop_stats()
    # anchored on the first call (h 10): re-arms at h 50, 90, ..., 370 = 9 of them
    assert st["stall_en_rearms_per_ep"] == pytest.approx(9.0)
    assert rr.stall_en_rearms == 0


# ==========================================================================
# (b) the eval patience hooks
# ==========================================================================
def _hook_core(n):
    c = _Core(n)
    return c.put([0.0] * n, [0.0] * n)


def test_eval_hook_energy_rearms_and_zero_is_the_old_hook():
    sys.path.insert(0, str(ROOT / "python"))
    import train_fast as tf
    for en, want_cut in ((0.0, True), (32.0, False)):
        core = _hook_core(1)
        hook = tf.make_eval_patience_hook(core, _FlatField(), 50, energy_eps=en)
        for t in range(1, 400):
            core.put([0.0], [_vx_for(2.0 * t)])                         # +2 u of h per tick
            hook(t, None, None, np.zeros(1, bool), np.zeros(1, bool))
        assert bool(core.failed[0]) is want_cut, en
    # a static line is cut with the energy rule on as well
    core = _hook_core(1)
    hook = tf.make_eval_patience_hook(core, _FlatField(), 50, energy_eps=32.0)
    for t in range(1, 60):
        hook(t, None, None, np.zeros(1, bool), np.zeros(1, bool))
    assert core.failed[0] == 1 and hook.state["n"] == 1


def test_eval_hook_par_energy_rearms_per_env():
    import train_fast as tf
    core = _hook_core(3)
    hook = tf.make_eval_patience_hook_par(core, _FlatField(), 50, 3, energy_eps=32.0)
    for t in range(1, 400):
        core.put([0.0, 0.0, 0.0], [_vx_for(2.0 * t), 0.0, _vx_for(min(2.0 * t, 100.0))])
        hook(t, None, None, np.zeros(3, bool), np.zeros(3, bool))
    assert core.failed.tolist() == [0, 1, 1]
    core0 = _hook_core(2)
    hook0 = tf.make_eval_patience_hook_par(core0, _FlatField(), 50, 2)   # energy off
    for t in range(1, 60):
        core0.put([0.0, 0.0], [_vx_for(2.0 * t), 0.0])
        hook0(t, None, None, np.zeros(2, bool), np.zeros(2, bool))
    assert core0.failed.tolist() == [1, 1]


# ==========================================================================
# (c) the recorder
# ==========================================================================
def test_the_recorder_lists_both_as_training_only():
    import record_ckpt as rc
    assert "stall_energy" in rc.TRAIN_ONLY and "eval_patience_energy" in rc.TRAIN_ONLY
