"""Reward/respawn semantics gate (docs/perf-implementation-plan.md 0.3).

These pin the properties that a perf refactor is most likely to break while
still producing plausible-looking training curves — the failure mode that
costs a night of GPU time before anyone notices:

  * potential shaping TELESCOPES (a closed loop nets ~0, backtracking
    refunds itself) — break it and the agent farms circles;
  * ended-tick rows are POST-AUTORESET (the states view already holds the new
    episode's spawn), so their shaping and their novelty bonus must be
    masked — break it and a respawn relocation pays thousands of units of
    "progress";
  * a goal crossing pays exactly ``success_bonus`` and ends the episode;
  * the stall-kill fires at ``stall_ticks`` and re-arms once consumed;
  * ``RespawnBuffer`` only harvests snapshots at least ``margin_ticks``
    before the episode ended.

Most tests run against a tiny fake core (fast, no DLL, no map). The ones that
genuinely need the C swept-segment goal test build a 1-env SurfCore on
surf_src_cannonball and are skipped when the map or the built core is absent.

    python -m pytest tests/python -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.goalfield import EuclidField                  # noqa: E402
from surfgym.respawn import RespawnBuffer                  # noqa: E402
from surfgym.rewards import RaceReward                     # noqa: E402

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
ZONES = ROOT / "maps" / "surf_src_cannonball.zones.json"


# --------------------------------------------------------------------------
# a fake core: the reward reads only states_view / goal_hits / map_bounds
# --------------------------------------------------------------------------
class FakeCore:
    def __init__(self, n: int = 1, bounds=((-2e4, -2e4, -2e4), (2e4, 2e4, 2e4))):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self._bounds = (np.asarray(bounds[0], np.float32),
                        np.asarray(bounds[1], np.float32))

    def map_bounds(self):
        return self._bounds

    def at(self, pos, vel=(0.0, 0.0, 0.0)):
        self.states_view["origin"] = np.asarray(pos, np.float32)
        self.states_view["velocity"] = np.asarray(vel, np.float32)
        return self


def _step(rw, core, done=None, trunc=None):
    """One reward call with the fake-core contract (obs args are unused by
    RaceReward — it reads the states view)."""
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    done = z if done is None else np.asarray(done, np.uint8)
    trunc = z if trunc is None else np.asarray(trunc, np.uint8)
    obs = np.zeros((n, 15), np.float32)
    return rw(obs, obs, obs, np.zeros(n, np.float32), done, trunc, core)


def _race(field, **kw):
    kw.setdefault("scale", 100.0 / 20000.0)
    kw.setdefault("time_pen", 0.0)
    kw.setdefault("success_bonus", 50.0)
    return RaceReward(field, **kw)


@pytest.fixture
def goal_box():
    if ZONES.exists():
        return json.loads(ZONES.read_text(encoding="utf-8"))["end"]
    return {"mins": [-14720.0, 7487.0, -1824.0], "maxs": [-8064.0, 7488.0, -352.0]}


# --------------------------------------------------------------------------
# (c) telescoping shaping
# --------------------------------------------------------------------------
def test_closed_loop_nets_zero_shaping(goal_box):
    """Two full circles must net ~0: potential shaping telescopes, so the
    only collectible income is net displacement toward the goal."""
    field = EuclidField(goal_box)
    rw = _race(field)
    core = FakeCore()
    centre = np.array([-11000.0, 3000.0, 2000.0])
    radius = 500.0
    core.at(centre + [radius, 0.0, 0.0])
    _step(rw, core)                               # first call primes _d
    total = 0.0
    n_pts = 128
    for k in range(1, 2 * n_pts + 1):             # two laps, back to the start
        a = 2.0 * np.pi * k / n_pts
        core.at(centre + [radius * np.cos(a), radius * np.sin(a), 0.0])
        total += float(_step(rw, core)[0])
    assert abs(total) < 0.2, f"two circles netted {total:.4f} shaping"


def test_backtracking_refunds_itself(goal_box):
    """Out-and-back along the track nets ~0 — the sign of the shaping term
    must be symmetric, not a one-way ratchet."""
    rw = _race(EuclidField(goal_box))
    core = FakeCore()
    core.at([-11000.0, 3000.0, 2000.0])
    _step(rw, core)
    out = sum(float(_step(rw, core.at([-11000.0, 3000.0 + 40.0 * k, 2000.0]))[0])
              for k in range(1, 51))
    back = sum(float(_step(rw, core.at([-11000.0, 3000.0 + 40.0 * k, 2000.0]))[0])
               for k in range(49, -1, -1))
    assert out > 0.0, "approaching the goal must pay"
    assert abs(out + back) < 1e-3, f"round trip netted {out + back:.5f}"


def test_time_penalty_is_the_only_income_when_still(goal_box):
    rw = _race(EuclidField(goal_box), time_pen=0.005)
    core = FakeCore().at([-11000.0, 3000.0, 2000.0])
    _step(rw, core)
    r = [float(_step(rw, core)[0]) for _ in range(10)]
    assert np.allclose(r, -0.005, atol=1e-6)


def test_teleport_step_is_clipped(goal_box):
    """A relocation of thousands of units must not cash shaping: the
    per-tick delta is clipped to max_step first."""
    rw = _race(EuclidField(goal_box), max_step=100.0)
    core = FakeCore().at([-11000.0, 0.0, 2000.0])
    _step(rw, core)
    r = float(_step(rw, core.at([-11000.0, 7000.0, 2000.0]))[0])
    assert r <= 100.0 * rw.scale + 1e-6, f"teleport paid {r:.4f}"


# --------------------------------------------------------------------------
# ended-tick masking (post-autoreset rows)
# --------------------------------------------------------------------------
def test_ended_tick_pays_no_shaping(goal_box):
    """On an ended tick the states view already holds the NEW episode's
    spawn — the apparent jump must pay exactly 0."""
    rw = _race(EuclidField(goal_box), time_pen=0.005)
    core = FakeCore().at([-11000.0, 7000.0, 2000.0])
    _step(rw, core)
    core.at([-14000.0, 2898.0, 10700.0])          # respawned at the start
    r = float(_step(rw, core, done=[1])[0])
    assert r == 0.0, f"ended tick paid {r:.4f}"


def test_ended_tick_pays_no_novelty(goal_box):
    """Same rule for count-based novelty: a respawn relocation enters a new
    256u cell but must not be rewarded for it."""
    rw = _race(EuclidField(goal_box), int_coef=1.0, scale=0.0)  # novelty only
    core = FakeCore().at([-11000.0, 7000.0, 2000.0])
    _step(rw, core)
    for k in range(1, 4):                          # honest novel cells pay
        assert float(_step(rw, core.at([-11000.0 + 300.0 * k, 7000.0, 2000.0]))[0]) > 0.0
    core.at([5000.0, -5000.0, 0.0])                # far away + episode ended
    assert float(_step(rw, core, done=[1])[0]) == 0.0


def test_novelty_self_anneals(goal_box):
    """Bouncing between two cells must decay toward zero, not pay forever."""
    rw = _race(EuclidField(goal_box), int_coef=1.0, scale=0.0)  # novelty only
    core = FakeCore().at([0.0, 0.0, 0.0])
    _step(rw, core)
    pays = []
    for k in range(40):
        core.at([0.0 if k % 2 else 300.0, 0.0, 0.0])
        pays.append(float(_step(rw, core)[0]))
    assert pays[0] > pays[-1] > 0.0
    assert pays[-1] < 0.25 * pays[0], f"novelty flat: {pays[0]:.3f} -> {pays[-1]:.3f}"


# --------------------------------------------------------------------------
# (b) stall-kill
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stall_ticks", [50, 300])
def test_stall_kill_fires_at_stall_ticks(goal_box, stall_ticks):
    rw = _race(EuclidField(goal_box), stall_ticks=stall_ticks)
    core = FakeCore().at([-11000.0, 3000.0, 2000.0])
    _step(rw, core)                                # priming call, no counter
    fired = None
    for k in range(1, stall_ticks + 20):
        _step(rw, core)                            # never moves -> never improves
        if rw.pop_stall_mask() is not None:
            fired = k
            break
    assert fired is not None, "stall kill never fired"
    assert abs(fired - stall_ticks) <= 2, f"fired at {fired}, want ~{stall_ticks}"
    assert rw.pop_stall_mask() is None, "kill must re-arm, not latch"


def test_progress_resets_the_stall_counter(goal_box):
    """Improving the best distance by more than stall_eps re-arms the window
    — otherwise a slow-but-advancing agent gets executed."""
    rw = _race(EuclidField(goal_box), stall_ticks=50, stall_eps=32.0)
    core = FakeCore().at([-11000.0, 3000.0, 2000.0])
    _step(rw, core)
    for lap in range(4):
        for _ in range(40):
            _step(rw, core)
            assert rw.pop_stall_mask() is None
        # a real advance toward the goal box (which sits at y ~ 7487)
        core.at([-11000.0, 3000.0 + 100.0 * (lap + 1), 2000.0])
        _step(rw, core)


# --------------------------------------------------------------------------
# (d) RespawnBuffer margin rule
# --------------------------------------------------------------------------
def test_respawn_margin_harvests_only_pre_margin_snapshots():
    """800-tick episode, margin 300, snapshots every 100 ticks -> exactly the
    snapshots at ticks 100..500 are harvested."""
    rb = RespawnBuffer(1, reservoir=100, margin_ticks=300, snap_every=100,
                       map_id="t")
    states = np.zeros(1, dtype=STATE_DTYPE)
    for t in range(1, 801):
        states["origin"] = [float(t), 0.0, 0.0]
        rb.observe(states, np.array([t == 800]))
    rb.flush_harvest()      # the trainer drains once per iteration
    assert rb.size == 5, f"harvested {rb.size}, want 5 (ticks 100..500)"
    got = sorted(float(x) for x in rb._store[:rb.size]["origin"][:, 0])
    assert got == [100.0, 200.0, 300.0, 400.0, 500.0]


def test_respawn_discards_pending_on_a_short_episode():
    """An episode shorter than the margin contributes nothing."""
    rb = RespawnBuffer(1, reservoir=100, margin_ticks=300, snap_every=100,
                       map_id="t")
    states = np.zeros(1, dtype=STATE_DTYPE)
    for t in range(1, 251):
        rb.observe(states, np.array([t == 250]))
    assert rb.size == 0


def test_respawn_skips_stagnant_states():
    rb = RespawnBuffer(1, reservoir=100, margin_ticks=100, snap_every=100,
                       map_id="t")
    states = np.zeros(1, dtype=STATE_DTYPE)
    stag = np.array([True])
    for t in range(1, 801):
        rb.observe(states, np.array([t == 800]), stagnant=stag)
    assert rb.size == 0, "stagnant states must never be snapshotted"


def test_respawn_reservoir_survives_a_checkpoint_roundtrip():
    rb = RespawnBuffer(1, reservoir=100, margin_ticks=100, snap_every=100,
                       map_id="ski")
    states = np.zeros(1, dtype=STATE_DTYPE)
    for t in range(1, 801):
        states["origin"] = [float(t), 0.0, 0.0]
        rb.observe(states, np.array([t == 800]))
    rb.flush_harvest()      # the trainer drains once per iteration
    n = rb.size
    assert n > 0
    rb2 = RespawnBuffer(1, reservoir=100, margin_ticks=100, snap_every=100,
                        map_id="ski")
    rb2.load_state_dict(rb.state_dict())
    assert rb2.size == n
    rb3 = RespawnBuffer(1, reservoir=100, margin_ticks=100, snap_every=100,
                        map_id="OTHER_MAP")
    rb3.load_state_dict(rb.state_dict())
    assert rb3.size == 0, "cross-map reservoirs must be dropped, not spawned"


# --------------------------------------------------------------------------
# (a) goal crossing — needs the C core's swept-segment goal test
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CANNONBALL.exists(), reason="surf_src_cannonball.bsp absent")
def test_goal_crossing_pays_bonus_and_ends_the_episode(goal_box):
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(
        num_envs=1, spawn_mode=2, max_episode_ticks=12000, water_fail=1,
        sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))
    core.set_goal_box(goal_box["mins"], goal_box["maxs"])
    core.reset(0)
    st = core.get_states()
    st["origin"] = [-11000.0, 7300.0, -1000.0]     # short of the finish curtain
    st["velocity"] = [0.0, 3500.0, 0.0]            # straight through it
    st["onground"] = -1
    core.set_state(0, st[0])

    rw = _race(EuclidField(goal_box), time_pen=0.005, success_bonus=50.0)
    _step(rw, core)                                # prime on the placed state
    act = np.zeros((1, 6), np.int32)
    act[0, 0] = 7                                  # yaw bin 7 = 0 deg
    act[0, 1] = 3                                  # pitch bin centre
    act[0, 2] = act[0, 3] = 1                      # no fwd/side input
    hit = None
    for t in range(20):
        _, _, done, trunc, _ = core.step(act)
        r = float(rw(np.zeros((1, 15), np.float32), np.zeros((1, 15), np.float32),
                     np.zeros((1, 15), np.float32), np.zeros(1, np.float32),
                     done, trunc, core)[0])
        if core.goal_hits[0]:
            hit = (t, r, int(done[0]))
            break
    assert hit is not None, "never crossed the goal box in 20 ticks at 3500 u/s"
    t, r, done0 = hit
    assert done0 == 1, "a goal crossing must END the episode"
    assert abs(r - 50.0) < 1e-4, (
        f"goal tick paid {r:.4f}, want exactly success_bonus (ended-tick "
        f"shaping is masked, so the bonus is the whole reward)")
