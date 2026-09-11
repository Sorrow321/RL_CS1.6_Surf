"""``--respawn-backward``: the map run BACKWARD on its own geodesic field.

Salimans-Chen's backward curriculum without a demonstration: the spawn band
is distance-to-goal in [floor, W], anchored at the goal and WIDENING toward
the map start whenever the finish rate of episodes spawned in the band's far
shell clears a threshold. Everything here is a way for that to be silently
wrong:

1.  the BAND is [floor, W] in distance and the SHELL is its FAR edge;
2.  the SPEED is an absolute prior down the field descent, so the sampler
    works with an EMPTY reservoir (the curriculum starts before the
    reservoir holds anything);
3.  the ADVANCE fires on the far shell's finish rate, needs min_ep
    episodes, moves W by step_frac x d0, clamps at d0 and turns the shell
    off there; the linear floor pushes W without successes;
4.  the POOL keeps the map-start share exactly, is all backward rows before
    the reservoir holds 2,000 states, and back_frac of the rest after;
5.  the reward's per-episode pairs carry the FINISH flag the rule reads;
6.  the frontier sampler's refactor (band / shell / speed hooks) leaves the
    frontier's own tests green - see test_respawn_frontier.py.

    python -m pytest tests/python/test_respawn_backward.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                           # noqa: E402
from surfgym.goalfield import EuclidField, GoalField           # noqa: E402
from surfgym.mapfleet import MapSlot                           # noqa: E402
from surfgym.respawn import (BackwardSpawnSampler,             # noqa: E402
                             RespawnBuffer)
from surfgym.rewards import RaceReward                         # noqa: E402

PETRUS = ROOT / "maps" / "surf_petrus_lite.bsp"
TRAIN_SRC = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")


def _field_npz():
    cands = [ROOT / "maps", Path("C:/RL_Surf/maps"), Path("/root/RL_Surf/maps")]
    env = os.environ.get("RLSURF_MAPS")
    if env:
        cands.insert(0, Path(env))
    for d in cands:
        p = d / "surf_petrus_lite.goal_32.npz"
        if p.is_file():
            return p
    return None


FIELD_NPZ = _field_npz()
needs_map = pytest.mark.skipif(not PETRUS.is_file(),
                               reason="surf_petrus_lite.bsp not present")
needs_field = pytest.mark.skipif(FIELD_NPZ is None,
                                 reason="no baked petrus goal_32 field")


@pytest.fixture(scope="module")
def real_field():
    z = np.load(FIELD_NPZ, allow_pickle=False)
    grid = z["grid"].astype(np.float32) * float(z["quant"])
    return GoalField(grid, z["mins"], float(z["cell"]), float(z["reach_max"]))


@pytest.fixture(scope="module")
def core():
    from surfgym import SurfCore, default_config
    return SurfCore(str(PETRUS), default_config(
        num_envs=1024, spawn_mode=2, max_episode_ticks=200,
        sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))


@pytest.fixture(scope="module")
def d0(core, real_field):
    from surfgym.rewards import map_spawn_pool
    return float(np.mean(real_field.sample(map_spawn_pool(core)["origin"])))


@pytest.fixture(scope="module")
def start_pool(core):
    from surfgym.rewards import map_spawn_pool
    return map_spawn_pool(core)


def _sampler(core, field, d0, **kw):
    kw.setdefault("start_frac", 0.05)
    kw.setdefault("step_frac", 0.05)
    kw.setdefault("min_ep", 20)
    kw.setdefault("seed", 5)
    return BackwardSpawnSampler(core, field, d0, **kw)


# ==========================================================================
# 1. the band and the far shell
# ==========================================================================
@needs_map
@needs_field
def test_band_is_distance_floor_to_W_and_the_shell_is_its_far_edge(
        core, real_field, d0):
    bs = _sampler(core, real_field, d0, floor=256.0)
    assert bs.W == pytest.approx(0.05 * d0)
    p, n_sh = bs._positions(256)
    assert len(p) >= 200
    d = np.asarray(real_field.sample(p), np.float64)
    assert (d >= 256.0 - 1e-6).all() and (d <= bs.W + 1e-6).all()
    assert n_sh > 0
    # shell rows come first in the concatenation and sit at d >= threshold
    thr = bs.shell_threshold()
    assert thr == pytest.approx(bs.W - 0.25 * (bs.W - 256.0))
    assert (d[:n_sh] >= thr - 1e-6).all()
    assert (d[n_sh:] <= thr + 1e-6).all()


# ==========================================================================
# 2. speed: an absolute prior, no reservoir needed
# ==========================================================================
@needs_map
@needs_field
def test_speed_is_the_absolute_prior_and_needs_no_reservoir(core, real_field,
                                                            d0):
    bs = _sampler(core, real_field, d0, speed=(1000.0, 2500.0))
    rows = bs.sample_states(128, None)
    assert len(rows) >= 100
    v = np.asarray(rows["velocity"], np.float64)
    spd = np.linalg.norm(v, axis=1)
    assert spd.min() >= 1000.0 - 1e-3 and spd.max() <= 2500.0 + 1e-3
    assert np.std(spd) > 100.0, "a range, not a constant"
    # the same draw with an empty reservoir is still a draw
    rb = RespawnBuffer(4, reservoir=100, margin_ticks=100)
    assert len(bs.sample_states(32, rb)) > 0


@needs_map
@needs_field
def test_velocity_points_down_the_field(core, real_field, d0):
    bs = _sampler(core, real_field, d0)
    rows = bs.sample_states(256, None)
    p = np.asarray(rows["origin"], np.float64)
    v = np.asarray(rows["velocity"], np.float64)
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    d_here = np.asarray(real_field.sample(p), np.float64)
    d_ahead = np.asarray(real_field.sample(p + 64.0 * u), np.float64)
    ok = d_ahead < real_field.reach_max
    assert ok.mean() > 0.9
    assert (d_ahead[ok] < d_here[ok]).mean() > 0.9


# ==========================================================================
# 3. the advance rule
# ==========================================================================
def _stub(d0=10_000.0, **kw):
    class _F:
        grid = np.zeros((2, 2, 2), np.float32)
        mins = np.zeros(3)
        cell = 32.0
        reach_max = 1e9
        _valid_max = 1e9

        def sample(self, p):
            return np.zeros(len(np.atleast_2d(p)), np.float32)
    kw.setdefault("start_frac", 0.1)
    kw.setdefault("step_frac", 0.1)
    kw.setdefault("rate", 0.2)
    kw.setdefault("min_ep", 20)
    kw.setdefault("window", 1000.0)
    return BackwardSpawnSampler(None, _F(), d0, **kw)


def test_advance_fires_on_the_far_shell_rate_and_clamps_at_d0():
    bs = _stub()
    assert bs.W == pytest.approx(1_000.0)
    thr = bs.shell_threshold()                 # 1000 - 0.25 * (1000 - 256)
    shell_d = np.full(40, thr + 10.0)
    # 10 % finish: below the 20 % rate -> no move
    bs.observe(100, shell_d, np.arange(40) < 4)
    assert bs.advance(100) == pytest.approx(1_000.0)
    assert bs.last_n == 40 and bs.last_rate == pytest.approx(0.1)
    # 30 % finish over 40 episodes -> one step of 0.1 x d0
    bs.observe(200, shell_d, np.arange(40) < 12)
    assert bs.advance(200) == pytest.approx(2_000.0)
    assert bs.n_adv == 1
    # the new shell is measured afresh: nothing counted until new episodes
    assert bs.advance(201) == pytest.approx(2_000.0) and bs.last_n == 0
    # episodes BEHIND the shell (nearer the goal) do not count
    bs.observe(300, np.full(40, 300.0), np.ones(40, bool))
    assert bs.advance(300) == pytest.approx(2_000.0)
    # too few episodes do not count even at 100 %
    bs.observe(400, np.full(10, bs.shell_threshold() + 1.0), np.ones(10, bool))
    assert bs.advance(400) == pytest.approx(2_000.0)
    # walk it to the end: W clamps at d0 and the shell turns off there
    for k in range(20):
        bs.observe(500 + k, np.full(40, bs.shell_threshold() + 1.0),
                   np.ones(40, bool))
        bs.advance(500 + k)
    assert bs.W == pytest.approx(10_000.0)
    assert not bs.shell_on
    assert bs.stats()["W_frac"] == pytest.approx(1.0)


def test_the_window_forgets_old_shell_episodes():
    bs = _stub(window=100.0)
    bs.observe(0, np.full(40, bs.shell_threshold() + 1.0), np.ones(40, bool))
    # 200 steps later the 40 winners have expired: no advance
    assert bs.advance(200) == pytest.approx(1_000.0)
    assert bs.last_n == 0


def test_the_linear_floor_pushes_W_without_successes():
    bs = _stub(sched_steps=1_000_000.0)
    assert bs.advance(0) == pytest.approx(1_000.0)
    w = bs.advance(500_000)
    assert w == pytest.approx(1_000.0 + 0.5 * 9_000.0)
    assert bs.advance(2_000_000) == pytest.approx(10_000.0)


def test_state_dict_roundtrip():
    a = _stub()
    a.W, a.n_adv = 4_321.0, 3
    b = _stub()
    b.load_state_dict(a.state_dict())
    assert b.W == pytest.approx(4_321.0) and b.n_adv == 3
    assert b.shell_on
    b.load_state_dict({"W": 1e9, "n_adv": 9})
    assert b.W == pytest.approx(10_000.0) and not b.shell_on


# ==========================================================================
# 4. the pool
# ==========================================================================
@needs_map
@needs_field
def test_pool_without_a_reservoir_is_start_share_plus_backward_rows(
        core, real_field, d0, start_pool):
    bs = _sampler(core, real_field, d0)
    pool = bs.build_pool(start_pool, None, pool_size=512, fresh_frac=0.1)
    assert len(pool) == 512
    so = np.asarray(start_pool["origin"], np.float64)
    po = np.asarray(pool["origin"], np.float64)
    # the first 51 rows ARE map-start rows
    for q in po[:51]:
        assert np.min(np.linalg.norm(so - q, axis=1)) < 1e-3
    d = np.asarray(real_field.sample(po[51:]), np.float64)
    assert (d <= bs.W + 1e-6).all() and (d >= 256.0 - 1e-6).all()
    # an EMPTY reservoir object is the same as none
    rb = RespawnBuffer(4, reservoir=100, margin_ticks=100)
    pool2 = bs.build_pool(start_pool, rb, pool_size=512, fresh_frac=0.1)
    assert len(pool2) == 512


@needs_map
@needs_field
def test_pool_with_a_reservoir_takes_back_frac_of_the_non_start_rows(
        core, real_field, d0, start_pool):
    bs = _sampler(core, real_field, d0)
    rb = RespawnBuffer(4, reservoir=4000, margin_ticks=100)
    # 2,500 reservoir rows parked far from the band (near the start)
    rows = start_pool[np.zeros(2500, int)].copy()
    rb.push_many(rows)
    assert rb.size >= 2000
    pool = bs.build_pool(start_pool, rb, pool_size=1000, fresh_frac=0.1,
                         back_frac=0.5)
    assert len(pool) == 1000
    n_fresh, n_back = 100, 450
    d = np.asarray(real_field.sample(
        np.asarray(pool["origin"], np.float64)), np.float64)
    assert (d[n_fresh:n_fresh + n_back] <= bs.W + 1e-6).all()
    assert (d[n_fresh + n_back:] > bs.W).all(), \
        "the reservoir half must be the reservoir's own rows"


# ==========================================================================
# 5. the reward's pairs carry the finish flag
# ==========================================================================
class _Phys:
    sv_gravity = 800.0


class _Cfg:
    phys = _Phys()


class _FakeCore:
    def __init__(self, n=2):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.config = _Cfg()
        self._b = (np.asarray([-3e4] * 3, np.float32),
                   np.asarray([3e4] * 3, np.float32))

    def map_bounds(self):
        return self._b


def test_reward_pairs_carry_the_finish_flag():
    box = {"mins": [-1.0, -1.0, -1.0], "maxs": [1.0, 1.0, 1.0]}
    rw = RaceReward(EuclidField(box), scale=100.0 / 20_000.0, time_pen=0.0,
                    frontier_d0=20_000.0)
    core = _FakeCore(2)

    def put(ds):
        p = core.states_view["origin"].copy()
        for i, d in enumerate(ds):
            p[i] = (0.0, float(d), 0.0)
        core.states_view["origin"] = p

    def tick(ended=None, goal=None):
        n = core.num_envs
        z = np.zeros(n, np.uint8)
        core.goal_hits[:] = 0 if goal is None else np.asarray(goal, np.uint8)
        trunc = z if ended is None else np.asarray(ended, np.uint8)
        obs = np.zeros((n, 15), np.float32)
        return rw(obs, obs, obs, np.zeros(n, np.float32), z, trunc, core)

    put([3_000.0, 3_000.0])
    tick()
    put([500.0, 2_500.0])
    tick()
    put([3_000.0, 3_000.0])                 # both end; env 0 finished
    tick(ended=[1, 1], goal=[1, 0])
    rw.pop_stats()
    sp, bd, spd, ok = rw.pop_frontier_pairs()
    assert sp.shape == (2,) and ok.dtype == bool
    assert sp == pytest.approx([3_000.0, 3_000.0], abs=1.0)
    assert list(ok) == [True, False]
    assert rw.pop_frontier_pairs()[0].shape == (0,)


# ==========================================================================
# 6. trainer plumbing
# ==========================================================================
def test_backward_flag_plumbing():
    for needle in (
            '"--respawn-backward", action="store_true", default=None',
            "args.respawn_backward = False",
            "BACKWARD = bool(args.respawn_backward)",
            "elif _s.backward is not None:",
            "_s.backward.build_pool(",
            "_sp, _bd, _spd, _ok = _s.reward_fn.pop_frontier_pairs()",
            "_s.backward.observe(global_step, _sp, _ok)",
            "_s.backward.advance(global_step)",
            'CSV_COLS += [f"back/W_frac{_sfx}", f"back/shell_rate{_sfx}",',
            "+ (back_row if back_row is not None else [])",
            '"respawn_backward": args.respawn_backward,',
            'state["respawn_backward"] = {',
            'ck.get("respawn_backward") is not None',
            "--respawn-backward and --respawn-frontier are ",
            "--respawn-backward and --respawn-random are ",
            "--respawn-backward is single-GPU",
            "if args.respawn_backward and args.respawn_binned is None:",
            "success_margin=bool(args.respawn_frontier_uniform\n"
            "                                    or args.respawn_backward)",
            "or args.respawn_backward else 0.0),"):
        assert needle in TRAIN_SRC, needle
    # the backward branch is tried BEFORE the reservoir's 2000-state gate
    assert (TRAIN_SRC.index("elif _s.backward is not None:")
            < TRAIN_SRC.index("elif _s.respawn is not None and "
                              "_s.respawn.size >= 2000:"))
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for k in ("respawn_backward", "respawn_backward_speed",
              "respawn_backward_floor"):
        assert f'"{k}"' in rc, f"{k} missing from TRAIN_ONLY"


def test_slot_has_no_backward_sampler_by_default():
    class _C:
        num_envs = 4
    sl = MapSlot("m", "m.bsp", _C(), 0, 4)
    assert sl.backward is None
    assert "backward" in MapSlot.__slots__
