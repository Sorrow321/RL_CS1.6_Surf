"""--stall-energy EPS (zlr1b3, 2026-10-31): the stall clock - the training stall kill
(--stall-secs), the reservoir's stagnant mask (the same clock) and --eval-patience - also
re-arms once the episode's best ENERGY HEIGHT h = z + |v|^2 / (2 g) has grown by EPS u since the
clock last re-armed (cumulative, any rate).

Why: on a map whose climb needs speed pumped first (surf_za_loop: ride a ramp end to end,
gaining speed, until it carries you up to the next one) the field's records come only once the
speed is there, so a field-only clock kills the episodes that are doing the right thing.

(a) RaceReward on a flat field: the control path kills every window whatever the energy does;
    EPS re-arms on a rising energy record (however slowly: cumulative), never on a flat or
    falling energy or on regaining energy below the record; an ended row re-anchors at its new
    spawn's own energy.
(b) Off (0) is byte-identical: rewards, the clock and the masks equal a RaceReward that never
    heard of the flag, on a random trajectory with episode ends.
(c) The eval-patience hooks (single and parallel): EPS keeps a line that only gains energy
    alive; 0 cuts it at the window as before; the off path equals the default call.
(d) The recorder lists the flag as training-only.
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

STALL = 20          # ticks (every=1: one call per tick)
G = 800.0


class _FlatField:
    reach_max = 1e6 + 16.0
    cell = 32.0

    def sample(self, pos):
        return np.full(len(np.atleast_2d(pos)), 5000.0)


class _LineField:
    """d = x: moving -x is progress."""
    reach_max = 1e6 + 16.0
    cell = 32.0

    def sample(self, pos):
        return np.asarray(pos, np.float64).reshape(-1, 3)[:, 0].copy()


class _FakeCore:
    def __init__(self, n=1):
        self.num_envs = n
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.failed = np.zeros(n, np.uint8)
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=G))

    def put(self, z=None, speed=None, x=None):
        if z is not None:
            self.states_view["origin"][:, 2] = np.asarray(z, np.float32)
        if x is not None:
            self.states_view["origin"][:, 0] = np.asarray(x, np.float32)
        if speed is not None:
            self.states_view["velocity"][:, 0] = np.asarray(speed, np.float32)
            self.states_view["velocity"][:, 1:] = 0.0
        return self

    def force_fail(self, mask):
        self.failed |= np.asarray(mask, np.uint8)


def _rr(core, eps, field=None):
    rr = RaceReward(field or _FlatField(), scale=1.0, time_pen=0.0, stall_ticks=STALL,
                    stall_eps=32.0)
    rr.stall_energy = float(eps)
    rr.on_reset(core)
    return rr


def _step(rr, core, done=0):
    n = core.num_envs
    d = np.full(n, done, np.uint8)
    return rr(None, None, None, np.zeros(n, np.float32), d, np.zeros(n, np.uint8), core)


def _speed_for(h, z=0.0):
    """|v| with z + |v|^2 / 2g = h."""
    return float(np.sqrt(max(0.0, 2.0 * G * (h - z))))


def _kills(eps, hs, z=0.0):
    core = _FakeCore().put(z=z, speed=_speed_for(hs[0], z))
    rr = _rr(core, eps)
    k = 0
    for h in hs[1:]:
        core.put(speed=_speed_for(h, z))
        _step(rr, core)
        if rr.pop_stall_mask() is not None:
            k += 1
    return k


# ==========================================================================
# (a) the clock
# ==========================================================================
def test_control_path_kills_every_window_whatever_the_energy_does():
    rising = list(np.arange(0.0, 1000.0, 10.0))            # +10 u of h per call
    assert _kills(0.0, rising) == (len(rising) - 1) // STALL


def test_a_rising_energy_record_rearms_even_slowly():
    rising = list(np.arange(0.0, 1000.0, 10.0))            # 32 u every 4 calls
    assert _kills(32.0, rising) == 0
    slow = list(np.arange(0.0, 200.0, 2.0))                # 32 u every 16 calls < STALL
    assert _kills(32.0, slow) == 0


def test_flat_or_falling_energy_is_still_stalled():
    flat = [500.0] * 101
    assert _kills(32.0, flat) == 5
    falling = list(np.linspace(500.0, 0.0, 101))
    assert _kills(32.0, falling) == 5


def test_regaining_energy_below_the_record_does_not_rearm():
    # up to 400 (re-arms), down to 0, back up to 380: below the record, so no re-arm
    hs = list(np.linspace(0.0, 400.0, 11)) + list(np.linspace(400.0, 0.0, 30)) \
        + list(np.linspace(0.0, 380.0, 30))
    core = _FakeCore().put(speed=0.0)
    rr = _rr(core, 32.0)
    since = []
    for h in hs[1:]:
        core.put(speed=_speed_for(h))
        _step(rr, core)
        since.append(int(rr._since[0]))
    # the last re-arm is at the 400 record (call 10 at the latest); from there it only counts
    last_zero = max(i for i, s in enumerate(since) if s == 0)
    assert last_zero <= 10
    assert since[-1] == len(since) - 1 - last_zero


def test_an_ended_row_reanchors_at_its_new_spawns_energy():
    core = _FakeCore().put(speed=_speed_for(2000.0))
    rr = _rr(core, 32.0)
    for _ in range(5):
        _step(rr, core)
    assert rr._ebest[0] == pytest.approx(2000.0, abs=1.0)
    core.put(speed=0.0)                                    # the respawn: h = 0
    _step(rr, core, done=1)
    assert rr._ebest[0] == pytest.approx(0.0, abs=1e-6)
    assert rr._eref[0] == pytest.approx(0.0, abs=1e-6)
    assert rr._since[0] == 0
    for i in range(1, 6):                                  # +40 u per call from the new spawn
        core.put(speed=_speed_for(40.0 * i))
        _step(rr, core)
        assert rr._since[0] == 0, i                        # re-arms on its OWN record


def test_height_counts_like_speed():
    # h is z + |v|^2/2g: a climb at constant speed is an energy record too
    core = _FakeCore().put(z=0.0, speed=300.0)
    rr = _rr(core, 32.0)
    k = 0
    for i in range(100):
        core.put(z=10.0 * (i + 1))
        _step(rr, core)
        k += rr.pop_stall_mask() is not None
    assert k == 0


def test_the_stagnant_mask_reads_the_same_clock():
    core = _FakeCore().put(speed=0.0)
    rr = _rr(core, 32.0)
    for i in range(1, 400):
        core.put(speed=_speed_for(5.0 * i))
        _step(rr, core)
    m = rr.stagnant_mask(ticks=10)
    assert m is not None and not m[0]


# ==========================================================================
# (b) off is byte-identical
# ==========================================================================
def test_off_is_byte_identical_to_a_reward_that_never_heard_of_it():
    rng = np.random.default_rng(3)
    n = 8
    ca, cb = _FakeCore(n), _FakeCore(n)
    for c in (ca, cb):
        c.put(x=np.full(n, 5000.0), z=np.zeros(n), speed=np.zeros(n))
    ra = RaceReward(_LineField(), scale=0.01, time_pen=0.001, stall_ticks=STALL, stall_eps=5.0,
                    ratchet=True, ratchet_d0=5000.0)
    rb = RaceReward(_LineField(), scale=0.01, time_pen=0.001, stall_ticks=STALL, stall_eps=5.0,
                    ratchet=True, ratchet_d0=5000.0)
    rb.stall_energy = 0.0                                   # set explicitly, as the trainer does
    ra.on_reset(ca)
    rb.on_reset(cb)
    x = np.full(n, 5000.0)
    for t in range(600):
        x = x + rng.normal(-3.0, 20.0, n)
        z = rng.normal(0.0, 200.0, n)
        sp = np.abs(rng.normal(600.0, 300.0, n))
        done = (rng.random(n) < 0.01).astype(np.uint8)
        for c in (ca, cb):
            c.put(x=x, z=z, speed=sp)
        r_a = ra(None, None, None, np.zeros(n, np.float32), done, np.zeros(n, np.uint8), ca)
        r_b = rb(None, None, None, np.zeros(n, np.float32), done, np.zeros(n, np.uint8), cb)
        assert np.array_equal(r_a, r_b)
        assert np.array_equal(ra._since, rb._since)
        ma, mb = ra.pop_stall_mask(), rb.pop_stall_mask()
        assert (ma is None) == (mb is None)
        if ma is not None:
            assert np.array_equal(ma, mb)
    assert rb._ebest is None and rb._eref is None


# ==========================================================================
# (c) the eval-patience hooks
# ==========================================================================
def _drive_hook(hook, core, hs, xs):
    kills = []
    for t, (h, x) in enumerate(zip(hs, xs)):
        core.put(x=x, speed=_speed_for(h))
        core.failed[:] = 0
        hook(t, None, None, np.zeros(core.num_envs, bool), np.zeros(core.num_envs, bool))
        if core.failed.any():
            kills.append(t)
    return kills


def test_eval_patience_keeps_a_line_that_only_gains_energy():
    from train_fast import make_eval_patience_hook
    hs = list(np.arange(0.0, 2000.0, 10.0))                # 200 ticks, +10 u of h per tick
    xs = [5000.0] * len(hs)                                # no field progress at all
    core = _FakeCore(1)
    assert _drive_hook(make_eval_patience_hook(core, _LineField(), 50), core, hs, xs) \
        == [50, 100, 150]
    core = _FakeCore(1)
    assert _drive_hook(make_eval_patience_hook(core, _LineField(), 50, energy_eps=32.0),
                       core, hs, xs) == []
    core = _FakeCore(1)
    flat = [700.0] * 200
    assert _drive_hook(make_eval_patience_hook(core, _LineField(), 50, energy_eps=32.0),
                       core, flat, xs) == [50, 100, 150]


def test_eval_patience_off_equals_the_default_call():
    from train_fast import make_eval_patience_hook
    rng = np.random.default_rng(5)
    hs = list(np.abs(rng.normal(500, 300, 300)))
    xs = list(5000.0 + np.cumsum(rng.normal(0.5, 10.0, 300)))
    ca, cb = _FakeCore(1), _FakeCore(1)
    a = _drive_hook(make_eval_patience_hook(ca, _LineField(), 40), ca, hs, xs)
    b = _drive_hook(make_eval_patience_hook(cb, _LineField(), 40, energy_eps=0.0), cb, hs, xs)
    assert a == b and len(a) > 0


def test_parallel_eval_patience_energy_rule():
    from train_fast import make_eval_patience_hook_par
    n = 3
    core = _FakeCore(n)
    hook = make_eval_patience_hook_par(core, _LineField(), 50, n, energy_eps=32.0)
    killed_at = {}
    for t in range(200):
        h = np.array([10.0 * t, 700.0, 10.0 * t])          # env 1 flat, 0 and 2 gaining
        x = np.array([5000.0, 5000.0, 5000.0 - 40.0 * t])   # env 2 also gains the field
        core.put(x=x, speed=[_speed_for(v) for v in h])
        core.failed[:] = 0
        hook(t, None, None, np.zeros(n, bool), np.zeros(n, bool))
        for i in np.flatnonzero(core.failed):
            killed_at.setdefault(int(i), t)
    assert killed_at == {1: 50}
    core = _FakeCore(n)
    hook = make_eval_patience_hook_par(core, _LineField(), 50, n)          # off
    killed_at = {}
    for t in range(200):
        h = np.array([10.0 * t, 700.0, 10.0 * t])
        x = np.array([5000.0, 5000.0, 5000.0 - 40.0 * t])
        core.put(x=x, speed=[_speed_for(v) for v in h])
        core.failed[:] = 0
        hook(t, None, None, np.zeros(n, bool), np.zeros(n, bool))
        for i in np.flatnonzero(core.failed):
            killed_at.setdefault(int(i), t)
    assert killed_at == {0: 50, 1: 50}


# ==========================================================================
# (d) the recorder
# ==========================================================================
def test_record_ckpt_lists_it_as_training_only():
    import record_ckpt
    assert "stall_energy" in record_ckpt.TRAIN_ONLY
