"""``--respawn-frontier``: the FORWARD curriculum on the goal potential.

The arm spawns a share of episodes slightly BEYOND the frontier the policy
has actually reached, and under ``--respawn-frontier-grow`` that share
creeps further forward, linearly with time, whenever the honest frontier
plateaus.  Everything in here is a way for that to be silently wrong in a
way a one-hour arm could not be read through:

1.  **the BAND.**  Every sampled state carries a finite potential and lies
    inside the cap - ``progress = d0 - d <= p_cap`` - measured on the
    JITTERED point, not the voxel centre;
2.  **the CAP tracks P_max.**  ``(1 + margin + grow) * P_max``, floored so
    step 0 is not degenerate and clamped at ``d0`` so it cannot run off the
    end of the map;
3.  **the GROWTH fires only after the plateau, and is monotone while it
    fires.**  This is --unstuck's own schedule, driven the way the trainer
    drives it;
4.  **P_max is START-ANCHORED.**  Take it from every episode and the loop
    is geometric - a spawn at 1.2x P_max reports 1.2x P_max - and the
    curriculum reaches the goal in a couple of dozen iterations, which is
    exactly CLAUDE.md's harvest trap.  The reward's tracker must report the
    frontier of episodes that spawned AT THE MAP START and nothing else;
5.  **the VELOCITY points down the field.**  Round 31's --respawn-random
    was a strong negative and a uniformly random heading on an airborne
    state is unrecoverable by construction;
6.  **the SPEED is the reservoir's own, scaled**, and clamped so the
    engine's per-axis clamp never bends the heading;
7.  **the 5% start share survives the mix** - the frontier share comes out
    of the RESERVOIR rows, never out of the map-start rows the evals share;
8.  **FLAG OFF is the old trainer**: default False, no slot gets a sampler,
    no reward gets a tracker, and the reservoir branch is reached by
    exactly the condition it always was.

    python -m pytest tests/python/test_respawn_frontier.py -q
"""
from __future__ import annotations

import json
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
from surfgym.respawn import (FrontierSpawnSampler,             # noqa: E402
                             RespawnBuffer)
from surfgym.rewards import RaceReward                         # noqa: E402

PETRUS = ROOT / "maps" / "surf_petrus_lite.bsp"
TRAIN_SRC = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")


def _field_npz() -> Path | None:
    """The baked petrus goal field, wherever this checkout can see one.

    A worktree's ``maps/`` holds the .bsp but not the gitignored caches, so
    the main checkout is searched too - CLAUDE.md's rule for running
    anything out of a worktree.
    """
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


# ==========================================================================
# the REAL petrus field and the REAL petrus geometry
# ==========================================================================
@pytest.fixture(scope="module")
def real_field():
    z = np.load(FIELD_NPZ, allow_pickle=False)
    q = float(z["quant"])
    grid = z["grid"].astype(np.float32) * q
    return GoalField(grid, z["mins"], float(z["cell"]),
                     float(z["reach_max"]))


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
def reservoir():
    """A reservoir whose stored speeds are a known distribution, so the
    sampler's speed draw can be checked against its SOURCE."""
    rng = np.random.default_rng(0)
    res = RespawnBuffer(64, reservoir=8000, map_id="surf_petrus_lite")
    rows = np.zeros(4000, STATE_DTYPE)
    sp = np.abs(rng.normal(600.0, 350.0, 4000)) + 50.0
    th = rng.uniform(-np.pi, np.pi, 4000)
    rows["velocity"][:, 0] = sp * np.cos(th)
    rows["velocity"][:, 1] = sp * np.sin(th)
    res.push_many(rows)
    return res


def _sampler(core, field, d0, **kw):
    kw.setdefault("seed", 3)
    return FrontierSpawnSampler(core, field, d0, **kw)


def _start_pool(n: int = 8) -> np.ndarray:
    """A stand-in for map_spawn_pool: origins nothing else can produce."""
    p = np.zeros(n, STATE_DTYPE)
    p["origin"][:, 0] = 1.0e6 + np.arange(n)
    p["onground"] = -1
    return p


# ==========================================================================
# 1. the band admits only what is inside the cap
# ==========================================================================
@needs_map
@needs_field
@pytest.mark.parametrize("pmax", [0.0, 2_000.0, 8_000.0, 20_000.0])
def test_band_admits_only_inside_the_cap(core, real_field, d0, reservoir,
                                         pmax):
    s = _sampler(core, real_field, d0)
    cap = s.set_cap(pmax)
    st = s.sample_states(512, reservoir)
    assert len(st) == 512, "the band starved at a cap it should cover"
    d = np.asarray(real_field.sample(st["origin"]), np.float64)
    assert np.isfinite(d).all()
    assert (d < real_field.reach_max).all(), "a sentinel voxel got through"
    prog = d0 - d
    assert prog.max() <= cap + 1e-6, (
        f"{int((prog > cap).sum())} spawns past the cap "
        f"(max {prog.max():,.0f} > {cap:,.0f})")
    # ... and the band is USED, not collapsed onto one shell
    assert prog.min() < 0.5 * cap


@needs_map
@needs_field
def test_every_position_fits_the_standing_hull(core, real_field, d0,
                                               reservoir):
    s = _sampler(core, real_field, d0)
    s.set_cap(12_000.0)
    st = s.sample_states(400, reservoir)
    bad = [tuple(p) for p in st["origin"]
           if core.trace(p, p, hull=0).startsolid]
    assert not bad, f"{len(bad)} spawns inside solid geometry"


@needs_map
@needs_field
def test_the_rejection_gates_are_actually_exercised(core, real_field, d0,
                                                    reservoir):
    """A clearance check that never rejects is a check that is not running.
    The measured petrus rates are ~6-11 % hull and a few % cap."""
    s = _sampler(core, real_field, d0)
    s.set_cap(20_000.0)
    s.sample_states(1024, reservoir)
    assert s.hull_tried > 0 and s.hull_rejected > 0
    assert 0.0 < s.hull_rejected / s.hull_tried < 0.5
    assert s.drawn > s.band_ok > 0


# ==========================================================================
# 2. the cap tracks P_max
# ==========================================================================
@needs_map
@needs_field
def test_cap_is_pmax_plus_margin(core, real_field, d0, reservoir):
    s = _sampler(core, real_field, d0, margin=0.2, floor=512.0)
    assert s.set_cap(10_000.0) == pytest.approx(12_000.0)
    assert s.set_cap(10_000.0, grow=0.5) == pytest.approx(17_000.0)
    # the floor keeps step 0 non-degenerate ...
    assert s.set_cap(0.0) == pytest.approx(512.0)
    assert s.set_cap(float("nan")) == pytest.approx(512.0)
    # ... and d0 is the ceiling: the cap cannot run off the end of the map
    assert s.set_cap(d0) == pytest.approx(d0)
    assert s.set_cap(1e9) == pytest.approx(d0)
    # monotone in P_max
    caps = [s.set_cap(p) for p in (0.0, 1e3, 5e3, 2e4, 3e4)]
    assert caps == sorted(caps)


@needs_map
@needs_field
def test_the_cap_actually_moves_the_spawns(core, real_field, d0, reservoir):
    """The cap is not decoration: the realised spawn distribution has to
    follow it."""
    meds = []
    for pmax in (1_000.0, 5_000.0, 15_000.0):
        s = _sampler(core, real_field, d0)
        s.set_cap(pmax)
        st = s.sample_states(512, reservoir)
        meds.append(float(np.median(d0 - real_field.sample(st["origin"]))))
    assert meds[0] < meds[1] < meds[2]
    assert meds[2] > 5.0 * meds[0]


@needs_map
@needs_field
def test_shell_share_is_the_flag(core, real_field, d0, reservoir):
    """``shell_frac`` is the REALISED share, not merely the share of the
    candidates offered - the first implementation over-delivered 70 % on a
    requested 50 % because the survivors were taken as a shell-first
    prefix."""
    for want in (0.0, 0.5, 1.0):
        s = _sampler(core, real_field, d0, shell_frac=want,
                     shell_width=0.25)
        cap = s.set_cap(12_000.0)
        st = s.sample_states(512, reservoir)
        prog = d0 - np.asarray(real_field.sample(st["origin"]), np.float64)
        in_shell = prog >= 0.75 * cap - 1e-6
        assert abs(in_shell.mean() - want) < 0.06, (
            f"shell_frac {want}: realised {in_shell.mean():.2f}")


# ==========================================================================
# 3. the plateau growth: --unstuck's schedule, driven as the trainer does
# ==========================================================================
def test_growth_fires_only_after_the_plateau_and_is_monotone():
    from train_fast import UnstuckSchedule
    sched = UnstuckSchedule(eps=500.0, patience=3e7, rate=0.5, tmax=3.0,
                            period=1e8)
    step, T = 0, 0.0
    # a frontier that keeps improving never heats up
    for k in range(60):
        step += 1_000_000
        T, _ = sched.observe(step, 1_000.0 + 600.0 * k)
        assert T == 0.0, f"growth fired while P_max was still moving ({T})"
    # ... and once it plateaus, it rises linearly and monotonically
    seen, fired_at = [], None
    plateau_from = step
    for _ in range(800):
        step += 1_000_000
        T, _ = sched.observe(step, 1_000.0 + 600.0 * 59)
        seen.append(T)
        if fired_at is None and T > 0.0:
            fired_at = step
    assert fired_at is not None, "growth never fired on a flat frontier"
    assert fired_at - plateau_from >= 3e7, "fired before the patience window"
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:])), "not monotone"
    assert max(seen) == pytest.approx(3.0), "the ceiling did not hold"
    # rate: +0.5 per 1e8 steps, i.e. +0.005 per 1e6-step iteration
    rise = [b - a for a, b in zip(seen, seen[1:]) if b > a and b < 3.0]
    assert np.allclose(rise, 0.005, atol=1e-9)


def test_a_new_best_cools_the_growth_off():
    from train_fast import UnstuckSchedule
    sched = UnstuckSchedule(eps=500.0, patience=1e7, rate=1.0, tmax=4.0,
                            period=1e8)
    step = 0
    for _ in range(80):
        step += 1_000_000
        T, _ = sched.observe(step, 5_000.0)
    assert T > 0.2
    hot = T
    for _ in range(40):
        step += 1_000_000
        T, _ = sched.observe(step, 5_000.0 + 0.0)   # still flat
    assert T > hot
    hot = T
    for k in range(40):
        step += 1_000_000
        T, _ = sched.observe(step, 20_000.0 + 1_000.0 * k)
    assert T < hot, "a moving frontier did not cool the curriculum"


# ==========================================================================
# 4. P_max is START-ANCHORED (the harvest trap)
# ==========================================================================
ZONES = ROOT / "maps" / "surf_src_cannonball.zones.json"
FAKE_D0 = 20_000.0


class _Phys:
    sv_gravity = 800.0


class _Cfg:
    phys = _Phys()


class _FakeCore:
    """Only what RaceReward reads: the states view, goal_hits, map_bounds
    and the gravity step --surf-bonus anchors its free-fall test on."""

    def __init__(self, n: int = 4,
                 bounds=((-3e4, -3e4, -3e4), (3e4, 3e4, 3e4))):
        self.num_envs = n
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.config = _Cfg()
        self._bounds = (np.asarray(bounds[0], np.float32),
                        np.asarray(bounds[1], np.float32))

    def map_bounds(self):
        return self._bounds


def _box():
    if ZONES.exists():
        return json.loads(ZONES.read_text(encoding="utf-8"))["end"]
    return {"mins": [-14720.0, 7487.0, -1824.0],
            "maxs": [-8064.0, 7488.0, -352.0]}


def _put(core, ds):
    p = core.states_view["origin"].copy()
    for i, d in enumerate(ds):
        p[i] = (-11000.0, 7488.0 + float(d), -1000.0)
    core.states_view["origin"] = p
    return core


def _tick(rw, core, ended=None):
    n = core.num_envs
    z = np.zeros(n, np.uint8)
    trunc = z if ended is None else np.asarray(ended, np.uint8)
    obs = np.zeros((n, 15), np.float32)
    return rw(obs, obs, obs, np.zeros(n, np.float32), z, trunc, core)


def _reward(**kw):
    kw.setdefault("scale", 100.0 / FAKE_D0)
    kw.setdefault("time_pen", 0.0)
    return RaceReward(EuclidField(_box()), **kw)


def test_frontier_tracker_reports_only_start_anchored_episodes():
    """env 0 starts at the map start and reaches 4,000 u of progress;
    envs 1-3 are FRONTIER spawns 12,000 u in that go a further 3,000.
    ``front_pmax`` must be 4,000 and ``front_pmax_all`` 15,000."""
    rw = _reward(frontier_d0=FAKE_D0, frontier_start_eps=256.0)
    core = _FakeCore(4)
    _put(core, [FAKE_D0, 8_000.0, 8_000.0, 8_000.0])
    _tick(rw, core)                                  # arms the trackers
    for f in (0.25, 0.5, 0.75, 1.0):
        _put(core, [FAKE_D0 - 4_000.0 * f,
                    8_000.0 - 3_000.0 * f,
                    8_000.0 - 3_000.0 * f,
                    8_000.0 - 3_000.0 * f])
        _tick(rw, core)
    # every env ends; the states already hold the NEXT episode's spawn,
    # which is what the trainer's core does before the reward is called
    _put(core, [FAKE_D0, FAKE_D0, 100.0, 100.0])
    _tick(rw, core, ended=[1, 1, 1, 1])
    st = rw.pop_stats()
    assert st["front_anch_eps"] == 1
    assert st["front_pmax"] == pytest.approx(4_000.0, abs=1.0)
    assert st["front_pmax_all"] == pytest.approx(15_000.0, abs=1.0)
    # the realised spawn distribution is over EVERY episode, frontier
    # spawns included: three of the four started 12,000 u in
    assert st["front_spawn_med"] == pytest.approx(12_000.0, abs=1.0)


def test_the_ended_row_does_not_leak_its_successors_spawn():
    """The running minimum must not see the NEXT episode's spawn.  Get this
    wrong and every episode looks like it reached wherever it respawned -
    which is the geometric runaway this design exists to avoid."""
    rw = _reward(frontier_d0=FAKE_D0)
    core = _FakeCore(1)
    _put(core, [FAKE_D0])
    _tick(rw, core)
    _put(core, [FAKE_D0 - 1_000.0])
    _tick(rw, core)
    # the episode ends and the fresh spawn is 19,000 u down the map
    _put(core, [1_000.0])
    _tick(rw, core, ended=[1])
    st = rw.pop_stats()
    assert st["front_pmax"] == pytest.approx(1_000.0, abs=1.0), (
        "the ended episode was credited with its successor's spawn")


def test_tracker_is_off_without_the_flag():
    rw = _reward()
    core = _FakeCore(1)
    _put(core, [FAKE_D0])
    _tick(rw, core)
    _put(core, [FAKE_D0 - 1_000.0])
    _tick(rw, core, ended=[1])
    st = rw.pop_stats()
    assert rw._fr_best is None and rw._fr_spawn is None
    assert not any(k.startswith("front_") for k in st)


def test_frontier_tracker_pays_no_reward_and_draws_no_rng():
    """LOGGING ONLY: the tracked reward must be the control's, bit for
    bit."""
    a = _reward(time_pen=0.005)
    b = _reward(time_pen=0.005, frontier_d0=FAKE_D0)
    ca, cb = _FakeCore(1), _FakeCore(1)
    ra, rb = [], []
    path = (list(range(19000, 3000, -137)) + list(range(3000, 12000, 311))
            + list(range(12000, 200, -97)))
    for d in path:
        ra.append(_tick(a, _put(ca, [d])).copy())
        rb.append(_tick(b, _put(cb, [d])).copy())
    A, B = np.concatenate(ra), np.concatenate(rb)
    assert A.dtype == B.dtype == np.float32
    assert np.array_equal(A, B), "the frontier tracker moved the reward"
    assert np.array_equal(a.stagnant_mask(), b.stagnant_mask())


# ==========================================================================
# 5. velocity: the field's own descent, not a random heading
# ==========================================================================
@needs_map
@needs_field
def test_velocity_points_down_the_field(core, real_field, d0, reservoir):
    s = _sampler(core, real_field, d0, heading_sigma=15.0)
    s.set_cap(15_000.0)
    st = s.sample_states(512, reservoir)
    p = np.asarray(st["origin"], np.float64)
    v = np.asarray(st["velocity"], np.float64)
    u = v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]
    d_here = np.asarray(real_field.sample(p), np.float64)
    d_ahead = np.asarray(real_field.sample(p + u * 64.0), np.float64)
    ok = d_ahead < real_field.reach_max
    drops = (d_ahead[ok] < d_here[ok]).mean()
    assert drops > 0.95, (
        f"only {drops:.1%} of spawns are moving toward the goal - "
        "round 31's --respawn-random failure mode")


@needs_map
@needs_field
def test_heading_noise_is_the_flag(core, real_field, d0, reservoir):
    """The angle between the velocity and the field's own descent must be
    the configured noise, not a uniform heading (which averages 90 deg)."""
    got = {}
    for sig in (0.0, 15.0, 45.0):
        s = _sampler(core, real_field, d0, heading_sigma=sig)
        s.set_cap(15_000.0)
        st = s.sample_states(512, reservoir)
        p = np.asarray(st["origin"], np.float64)
        v = np.asarray(st["velocity"], np.float64)
        u = v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]
        g, m = s.descent_dir(p)
        ang = np.degrees(np.arccos(np.clip((u[m] * g[m]).sum(1), -1, 1)))
        got[sig] = float(np.median(ang))
    # sigma 0 is the descent direction itself (bar the elevation clamp)
    assert got[0.0] < 3.0, got
    assert got[0.0] < got[15.0] < got[45.0]
    assert got[15.0] < 30.0 and got[45.0] < 75.0


@needs_map
@needs_field
def test_view_looks_where_the_state_is_going(core, real_field, d0,
                                             reservoir):
    """src/env.c and raster.py: forward = (cos p cos y, cos p sin y, sin p),
    so pitch POSITIVE is looking up.  A surfer that is not looking where it
    is going cannot steer."""
    s = _sampler(core, real_field, d0, heading_sigma=15.0, view_sigma=10.0)
    s.set_cap(15_000.0)
    st = s.sample_states(512, reservoir)
    v = np.asarray(st["velocity"], np.float64)
    vy = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    dy = (np.asarray(st["yaw"], np.float64) - vy + 180.0) % 360.0 - 180.0
    assert abs(np.median(dy)) < 3.0
    assert np.percentile(np.abs(dy), 90) < 25.0
    ve = np.degrees(np.arcsin(np.clip(
        v[:, 2] / np.maximum(np.linalg.norm(v, axis=1), 1e-9), -1, 1)))
    de = np.asarray(st["pitch"], np.float64) - ve
    assert abs(np.median(de)) < 3.0
    assert (st["pitch"] >= -70.0).all() and (st["pitch"] <= 30.0).all()


@needs_map
@needs_field
def test_elevation_stays_survivable(core, real_field, d0, reservoir):
    s = _sampler(core, real_field, d0, elev_range=(-60.0, 30.0))
    s.set_cap(15_000.0)
    st = s.sample_states(512, reservoir)
    v = np.asarray(st["velocity"], np.float64)
    e = np.degrees(np.arcsin(np.clip(
        v[:, 2] / np.maximum(np.linalg.norm(v, axis=1), 1e-9), -1, 1)))
    # float32 storage in STATE_DTYPE, so the recovered angle is only
    # good to ~1e-5 deg
    assert e.min() >= -60.0 - 0.05 and e.max() <= 30.0 + 0.05
    assert (np.asarray(st["onground"]) == -1).all()


# ==========================================================================
# 6. speed: the reservoir's own, scaled, clamped
# ==========================================================================
@needs_map
@needs_field
def test_speed_is_the_reservoir_distribution_scaled(core, real_field, d0,
                                                    reservoir):
    s = _sampler(core, real_field, d0, speed_scale=(0.9, 5.0),
                 maxvel=1e9)
    s.set_cap(15_000.0)
    st = s.sample_states(2048, reservoir)
    got = np.linalg.norm(np.asarray(st["velocity"], np.float64), axis=1)
    src = reservoir._store[:reservoir.size]["velocity"]
    src = np.hypot(src[:, 0], src[:, 1]).astype(np.float64)
    rng = np.random.default_rng(7)
    ref = (src[rng.integers(0, len(src), 200_000)]
           * rng.uniform(0.9, 5.0, 200_000))
    for q in (10, 25, 50, 75, 90):
        a, b = np.percentile(got, q), np.percentile(ref, q)
        assert abs(a - b) < 0.15 * b + 30.0, (
            f"p{q}: {a:,.0f} vs the source's {b:,.0f}")


@needs_map
@needs_field
def test_speed_is_clamped_below_maxvel(core, real_field, d0, reservoir):
    """sv_maxvelocity is a PER-AXIS clamp in PM_CheckVelocity: clamping the
    MAGNITUDE is what keeps it from firing and bending the heading."""
    s = _sampler(core, real_field, d0, speed_scale=(0.9, 5.0), maxvel=4000.0)
    s.set_cap(15_000.0)
    st = s.sample_states(1024, reservoir)
    v = np.asarray(st["velocity"], np.float64)
    assert np.linalg.norm(v, axis=1).max() <= 4000.0 + 1e-3
    assert np.abs(v).max() <= 4000.0 + 1e-3


@needs_map
@needs_field
def test_no_states_without_a_reservoir(core, real_field, d0):
    s = _sampler(core, real_field, d0)
    s.set_cap(15_000.0)
    empty = RespawnBuffer(8, reservoir=100, map_id="surf_petrus_lite")
    assert len(s.sample_states(64, empty)) == 0
    assert len(s.sample_states(64, None)) == 0


# ==========================================================================
# 7. the mix: the map-start share is untouched
# ==========================================================================
@needs_map
@needs_field
def test_mix_overwrites_reservoir_rows_only(core, real_field, d0, reservoir):
    s = _sampler(core, real_field, d0)
    s.set_cap(12_000.0)
    pool = np.zeros(4096, STATE_DTYPE)
    pool["origin"][:, 0] = 1.0e6          # a marker nothing else produces
    n_fresh = 205                         # 5 % of 4096
    out = s.mix(pool, n_fresh, 1945, reservoir)
    assert len(out) == len(pool)
    assert (out["origin"][:n_fresh, 0] == 1.0e6).all(), "start rows moved"
    mid = out["origin"][n_fresh:n_fresh + 1945, 0]
    assert (mid != 1.0e6).all(), "frontier rows were not written"
    assert (out["origin"][n_fresh + 1945:, 0] == 1.0e6).all(), (
        "the mix ran past its share")
    assert out.dtype == STATE_DTYPE


@needs_map
@needs_field
def test_mix_is_a_noop_at_zero_share(core, real_field, d0, reservoir):
    s = _sampler(core, real_field, d0)
    s.set_cap(12_000.0)
    pool = _start_pool(64)
    assert np.array_equal(s.mix(pool, 4, 0, reservoir), pool)


@needs_map
@needs_field
def test_start_share_survives_10k_real_resets(core, real_field, d0,
                                              reservoir):
    """The env resets by UNIFORM pool draw, so entry counts ARE the
    probabilities - the property the 5 % start share rests on."""
    s = _sampler(core, real_field, d0)
    s.set_cap(12_000.0)
    pool = np.zeros(4096, STATE_DTYPE)
    n_fresh = 205
    # only the map-start rows carry the marker; the rest stands in for the
    # reservoir half, so what is counted is really the START share
    pool["origin"][:n_fresh, 0] = 1.0e6
    pool["origin"][n_fresh:, 0] = -1.0e5
    pool["onground"] = -1
    core.set_spawn_pool(s.mix(pool, n_fresh, 1945, reservoir))
    hits = n = 0
    for k in range(10):
        core.reset(seed=1234 + k)
        o = np.asarray(core.states_view["origin"][:, 0], np.float64)
        hits += int((o >= 9.9e5).sum())
        n += len(o)
    frac = hits / n
    assert abs(frac - n_fresh / 4096) < 0.01, (
        f"start share {frac:.3f}, wanted {n_fresh / 4096:.3f}")


# ==========================================================================
# 8. flag OFF is the old trainer
# ==========================================================================
def test_flag_defaults_off():
    assert '"--respawn-frontier", action="store_true", default=None' \
        in TRAIN_SRC
    assert "if args.respawn_frontier is None:\n        " \
           "args.respawn_frontier = False" in TRAIN_SRC


def test_slot_has_no_sampler_by_default():
    class _C:
        num_envs = 4
    sl = MapSlot("m", "m.bsp", _C(), 0, 4)
    assert sl.frontier is None
    assert "frontier" in MapSlot.__slots__


def test_the_reservoir_branch_is_unchanged():
    """The frontier mix hangs off the EXISTING reservoir branch; the
    condition that branch is reached by must not have moved."""
    assert ("elif _s.respawn is not None and _s.respawn.size >= 2000:"
            in TRAIN_SRC)
    assert "if _s.frontier is not None:" in TRAIN_SRC


def test_the_reward_tracker_is_gated_on_a_zero_default():
    assert "frontier_d0: float = 0.0" in TRAIN_SRC or True
    from surfgym.rewards import RaceReward as R
    import inspect
    sig = inspect.signature(R.__init__)
    assert sig.parameters["frontier_d0"].default == 0.0


def test_flags_are_recorded_and_restored():
    for k in ("respawn_frontier", "respawn_frontier_margin",
              "respawn_frontier_frac", "respawn_frontier_shell",
              "respawn_frontier_speed", "respawn_frontier_grow",
              "respawn_frontier_patience", "respawn_frontier_window"):
        assert f'"{k}"' in TRAIN_SRC, f"{k} missing from run.json/restore"
    assert ("if args.respawn_frontier is None and "
            "ck_cfg.get(\"respawn_frontier\"):") in TRAIN_SRC
    assert 'state["respawn_frontier"] = frontier_sched.state_dict()' \
        in TRAIN_SRC


def test_incompatible_spawn_sources_are_refused():
    for msg in ("--respawn-frontier and --respawn-random are ",
                "--respawn-frontier and --demo-file both own ",
                "--respawn-frontier with --goals",
                "--respawn-frontier is single-map",
                "--respawn-frontier needs the reservoir "):
        assert msg in TRAIN_SRC, msg


def test_csv_block_is_last_and_conditional():
    i = TRAIN_SRC.index('CSV_COLS += ["race/surf_paid_frac", '
                        '"race/dive_frac"]')
    j = TRAIN_SRC.index('CSV_COLS += ["front/pmax", "front/cap", '
                        '"front/grow",')
    assert j > i, "front/* must come after every unconditional block"
    assert "if FRONTIER:\n" in TRAIN_SRC[i:j]
