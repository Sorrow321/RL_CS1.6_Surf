"""--energy-stall EPS (zlr1b4, 2026-10-31): ENERGY HEIGHT counts as progress for the stall rules.

surf_za_loop's record rides its first loop for ~60 s with NEGATIVE field progress while its energy
height h = z + |v|^2 / (2 g) rises ~1,000 u; the training stall kill (--stall-secs 15) and the
greedy eval's --eval-patience (30 s) both watch the field alone and end such a line. With EPS > 0
both clocks also re-arm when the episode's best h has risen EPS above its value at the last
energy re-arm (cumulative, any rate).

(a) RaceReward's training stall clock: a line that keeps adding energy - however slowly, after a
    setback - is never stall-killed at EPS > 0 and is killed at EPS = 0; a line that adds no
    energy (constant speed, GoldSrc's exact free fall) is killed either way; a new episode starts
    its energy clock at its own spawn energy; EPS = 0 leaves the clock exactly as it was.
(b) The eval patience hooks (sequential and side by side): the same, per env.
(c) The recorder lists the flag as training-only and refuses to half-mirror it under
    --eval-stall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.rewards import RaceReward                       # noqa: E402
from test_arc_death_charge import _FakeCore, _FlatField, _step   # noqa: E402

G = 800.0


def _rr(eps, z0=0.0, v0=(0.0, 0.0, 0.0), stall_ticks=100, n=1):
    """A RaceReward reset with the core AT the first state (the trainer resets at the spawn)."""
    core = _FakeCore(n).at([0.0] * n)
    _put(core, z0, v0)
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=stall_ticks,
                    stall_eps=32.0)
    rr.stall_energy = eps
    rr.on_reset(core)
    return rr, core


def _put(core, z, v):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:] = np.asarray(v, np.float32).reshape(-1, 3)
    return core


def _ride(rr, core, zs, vs, ends=()):
    """One call per (z, v); the field is flat (no field progress ever). -> kill call indices."""
    kills = []
    for t, (z, v) in enumerate(zip(zs, vs)):
        _step(rr, _put(core, z, v), done=1 if t in ends else 0)
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            kills.append(t)
    return kills


def _pump(n, du=1.5, z0=-3000.0, v0=1000.0):
    """A pumping line: h rises du per call (32 u every ~21 calls) while z swings 200 u up and
    down (height traded for speed and back, the loop), with a 13-call setback of 60 u of h in
    the middle (a bad landing) that it pumps back. The first state is (z0, v0)."""
    zs, vs = [], []
    h = z0 + v0 * v0 / (2 * G)
    for t in range(n):
        if t:
            h += du if not (1000 <= t < 1013) else -60.0 / 13.0
        z = z0 + 200.0 * np.sin(t / 40.0)
        sp = np.sqrt(2 * G * (h - z))
        zs.append(z)
        vs.append((sp, 0.0, 0.0))
    return zs, vs


def _rr_at(eps, zs, vs, **kw):
    return _rr(eps, zs[0], vs[0], **kw)


# ==========================================================================
# (a) RaceReward
# ==========================================================================
def test_a_line_that_keeps_adding_energy_is_never_stall_killed():
    zs, vs = _pump(3000)
    rr, core = _rr_at(32.0, zs, vs)
    assert _ride(rr, core, zs, vs) == []
    rr0, core0 = _rr_at(0.0, zs, vs)
    assert _ride(rr0, core0, zs, vs) == [99, 199, 299, 399, 499, 599, 699, 799, 899, 999,
                                         1099, 1199, 1299, 1399, 1499, 1599, 1699, 1799, 1899,
                                         1999, 2099, 2199, 2299, 2399, 2499, 2599, 2699, 2799,
                                         2899, 2999]


def test_a_line_that_adds_no_energy_is_killed_either_way():
    # a constant speed on the level, then GoldSrc's exact free fall (half-step gravity)
    zs = [0.0] * 150
    vs = [(900.0, 0.0, 0.0)] * 150
    z, vz, dt = 5000.0, 300.0, 0.01
    for _ in range(250):
        zs.append(z)
        vs.append((700.0, 0.0, vz))
        vh = vz - 0.5 * G * dt
        z += vh * dt
        vz = vh - 0.5 * G * dt
    rr, core = _rr_at(32.0, zs, vs)
    k_on = _ride(rr, core, zs, vs)
    rr0, core0 = _rr_at(0.0, zs, vs)
    k_off = _ride(rr0, core0, zs, vs)
    assert k_off == [99, 199, 299, 399]
    # the jump in z at call 150 is a new energy level (an energy record once); after it the
    # free fall keeps h to float precision and the clock runs out again
    assert k_on and k_on[0] == 99 and k_on[-1] >= 349
    assert all(b - a >= 100 for a, b in zip(k_on, k_on[1:]))


def test_slow_gains_count_cumulatively_not_per_call():
    """0.4 u of h per call never clears 32 u in one call, but 32 u every 80 calls does."""
    zs = [0.0] * 2000
    vs = [(np.sqrt(2 * G * 0.4 * t + 400.0 ** 2), 0.0, 0.0) for t in range(2000)]
    rr, core = _rr_at(32.0, zs, vs)
    assert _ride(rr, core, zs, vs) == []
    # 0.3 u per call = 32 u every 107 calls: one window of 100 calls passes without a re-arm
    vs = [(np.sqrt(2 * G * 0.3 * t + 400.0 ** 2), 0.0, 0.0) for t in range(2000)]
    rr, core = _rr_at(32.0, zs, vs)
    assert _ride(rr, core, zs, vs) != []


def test_a_new_episode_starts_its_energy_clock_at_its_own_spawn():
    rr, core = _rr(32.0)          # spawn at rest at z 0
    # episode 0 climbs to h ~ 3000 (fast); the call at index 50 ends it, and (the core
    # autoresets in place) its state is ALREADY episode 1's spawn, at rest at z 0; episode 1
    # adds 0.5 u per call: it must be measured from ITS spawn, not from episode 0's 3000
    zs = [3000.0 * t / 50 for t in range(50)] + [0.0] * 401
    vs = [(0.0, 0.0, 0.0)] * 50 + [(np.sqrt(2 * G * 0.5 * t), 0.0, 0.0) for t in range(401)]
    assert _ride(rr, core, zs, vs, ends={50}) == []


def test_eps_zero_is_the_field_clock_exactly():
    rng = np.random.default_rng(1)
    zs = list(rng.uniform(-100, 100, 700))
    vs = [tuple(rng.normal(0, 600, 3)) for _ in range(700)]
    a, ca = _rr(0.0, zs[0], vs[0])
    b, cb = _rr(0.0, zs[0], vs[0])
    del b.stall_energy                    # the class default
    b.stall_energy = 0.0
    for z, v in zip(zs, vs):
        _step(a, _put(ca, z, v))
        _step(b, _put(cb, z, v))
        assert np.array_equal(a._since, b._since)
    assert a._se_ref is None and a._se_best is None


def test_field_progress_still_rearms_with_energy_on():
    class _LineField:
        def sample(self, pos):
            return np.asarray(np.atleast_2d(pos), np.float64)[:, 0].copy()
    core = _FakeCore(1).at([10000.0])
    rr = RaceReward(_LineField(), scale=1.0, time_pen=0.0, stall_ticks=100, stall_eps=32.0)
    rr.stall_energy = 32.0
    rr.on_reset(core)
    kills = []
    for t in range(1000):
        core.at([10000.0 - 40.0 * t])        # 40 u of field progress per call, no energy
        _step(rr, core)
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            kills.append(t)
    assert kills == []


# ==========================================================================
# (b) the eval patience hooks
# ==========================================================================
class _EvalCore:
    def __init__(self, n=1):
        from types import SimpleNamespace
        self.num_envs = n
        self.failed = np.zeros(n, np.uint8)
        self.states_view = np.zeros(n, dtype=[("origin", np.float64, (3,)),
                                              ("velocity", np.float32, (3,))])
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=800.0))

    def force_fail(self, mask):
        self.failed |= np.asarray(mask, np.uint8)


class _FlatEvalField:
    reach_max = 1e6
    cell = 32.0

    def sample(self, pts):
        return np.full(len(np.atleast_2d(pts)), 5000.0)


def _drive_seq(hook, core, zs, vs, ends=()):
    kills = []
    for t, (z, v) in enumerate(zip(zs, vs)):
        core.states_view["origin"][0, 2] = z
        core.states_view["velocity"][0] = v
        core.failed[:] = 0
        hook(t, None, None, np.array([t in ends]), np.array([False]))
        if core.failed[0]:
            kills.append(t)
    return kills


def test_eval_patience_keeps_a_pumping_line_and_cuts_a_stopped_one():
    from train_fast import make_eval_patience_hook
    zs, vs = _pump(3000)
    core = _EvalCore()
    hook = make_eval_patience_hook(core, _FlatEvalField(), 400, energy_eps=32.0)
    assert _drive_seq(hook, core, zs, vs) == []
    core = _EvalCore()
    hook = make_eval_patience_hook(core, _FlatEvalField(), 400)          # the field alone
    assert _drive_seq(hook, core, zs, vs) == [400, 800, 1200, 1600, 2000, 2400, 2800]
    # a setback longer than the window is no gain: cut, energy rule or not
    zs2 = zs[:600] + [zs[600]] * 500 + zs[600:1500]
    vs2 = vs[:600] + [(0.0, 0.0, 0.0)] * 500 + vs[600:1500]
    core = _EvalCore()
    hook = make_eval_patience_hook(core, _FlatEvalField(), 400, energy_eps=32.0)
    k = _drive_seq(hook, core, zs2, vs2)
    assert k and 600 < k[0] <= 1001
    # constant energy: cut with the energy rule on too
    core = _EvalCore()
    hook = make_eval_patience_hook(core, _FlatEvalField(), 400, energy_eps=32.0)
    assert _drive_seq(hook, core, [0.0] * 900, [(900.0, 0.0, 0.0)] * 900) == [400, 800]


def test_eval_patience_energy_clock_restarts_per_episode():
    from train_fast import make_eval_patience_hook
    core = _EvalCore()
    hook = make_eval_patience_hook(core, _FlatEvalField(), 100, energy_eps=32.0)
    # the hook sees the done flag on the episode's last tick and re-arms on the next one
    zs = [3000.0 * t / 50 for t in range(51)] + [0.0] * 400
    vs = [(0.0, 0.0, 0.0)] * 51 + [(np.sqrt(2 * G * 0.5 * t), 0.0, 0.0) for t in range(400)]
    assert _drive_seq(hook, core, zs, vs, ends={50}) == []


def test_eval_patience_parallel_matches_per_env():
    from train_fast import make_eval_patience_hook_par
    n = 3
    zp, vp = _pump(1500)
    core = _EvalCore(n)
    hook = make_eval_patience_hook_par(core, _FlatEvalField(), 400, n, energy_eps=32.0)
    kills, done = [], np.zeros(n, bool)
    for t in range(1500):
        # env 0 pumps, env 1 holds a constant speed, env 2 pumps until 600 then holds
        core.states_view["origin"][:, 2] = [zp[t], 0.0, zp[min(t, 600)]]
        core.states_view["velocity"][0] = vp[t]
        core.states_view["velocity"][1] = (900.0, 0.0, 0.0)
        core.states_view["velocity"][2] = vp[min(t, 600)]
        core.failed[:] = 0
        hook(t, None, None, done.copy(), np.zeros(n, bool))
        done = core.failed.astype(bool)       # a forced fail ends the episode on the next tick
        kills += [(t, i) for i in np.flatnonzero(core.failed)]
    assert [i for _, i in kills] == [1, 2]
    # env 2 last re-armed within ~21 calls before it froze at 600 (32 u at 1.5 u per call)
    assert kills[0][0] == 400 and 600 - 25 + 400 <= kills[1][0] <= 1001
    # without the energy rule every env is cut at the first window
    core = _EvalCore(n)
    hook = make_eval_patience_hook_par(core, _FlatEvalField(), 400, n)
    first = []
    for t in range(500):
        core.states_view["origin"][:, 2] = [zp[t], 0.0, zp[t]]
        core.failed[:] = 0
        hook(t, None, None, np.zeros(n, bool), np.zeros(n, bool))
        first += [(t, i) for i in np.flatnonzero(core.failed)]
    assert first == [(400, 0), (400, 1), (400, 2)]


# ==========================================================================
# (c) the recorder
# ==========================================================================
def test_the_recorder_lists_it_training_only_and_refuses_to_half_mirror():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    import re
    block = src[src.index("TRAIN_ONLY = frozenset({"):]
    block = block[:block.index("})")]
    assert re.search(r'"energy_stall"', block)
    assert 'cfg.get("energy_stall")' in src
