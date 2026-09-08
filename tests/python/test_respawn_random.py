"""``--respawn-random``: uniform reachable-state exploring starts.

The arm replaces the respawn reservoir with a spawn SOURCE that never looks
at what the policy has reached: 5% of episodes at the map's own start spawn
(exactly what the evals use), 95% at a uniformly random voxel of the goal
field that carries a finite potential, with a random view and a random
carried speed. What this file pins:

1. the MIX - 10k real C resets off a real pool land start-spawn 5% of the
   time, because the env resets by uniform pool draw and entry counts ARE
   the probabilities;
2. every sampled position has a FINITE POTENTIAL on the field it was drawn
   from, and the STANDING player hull fits there (a 32 u cell whose centre
   is free can still clip a wall);
3. the speed / yaw / pitch / vertical-velocity ranges, and the heading
   noise around the view yaw;
4. a sampled state ROUND-TRIPS through the core - ``set_state`` then read
   back gives the same origin, velocity and angles;
5. FLAG OFF is the old trainer: the default is False, no slot gets a
   sampler, and the reservoir branch is reached by exactly the condition it
   always was.

    python -m pytest tests/python/test_respawn_random.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                          # noqa: E402
from surfgym.goalfield import GoalField                       # noqa: E402
from surfgym.mapfleet import MapSlot                          # noqa: E402
from surfgym.respawn import RandomSpawnSampler                # noqa: E402

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
TRAIN_SRC = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")


def _field_npz() -> Path | None:
    """The baked cannonball goal field, wherever this checkout can see one.

    A worktree's ``maps/`` holds the .bsp but not the gitignored caches, so
    the main checkout is searched too - the same rule CLAUDE.md sets for
    running anything out of a worktree.
    """
    cands = [ROOT / "maps", Path("C:/RL_Surf/maps"), Path("/root/RL_Surf/maps")]
    env = os.environ.get("RLSURF_MAPS")
    if env:
        cands.insert(0, Path(env))
    for d in cands:
        p = d / "surf_src_cannonball.goal_32.npz"
        if p.is_file():
            return p
    return None


FIELD_NPZ = _field_npz()
needs_map = pytest.mark.skipif(not CANNONBALL.is_file(),
                               reason="surf_src_cannonball.bsp not present")
needs_field = pytest.mark.skipif(FIELD_NPZ is None,
                                 reason="no baked goal_32 field to draw from")


# ==========================================================================
# a REAL field over a REAL slice of the map
# ==========================================================================
@pytest.fixture(scope="module")
def real_field():
    """A GoalField over a sub-block of the baked cannonball field.

    Real geometry and real geodesic values, but ~1/300th of the voxels, so
    the test does not hold 2.7 GB of float32 the way the trainer does. The
    block is chosen as the densest 160-cell z-slab, i.e. where the map
    actually is.
    """
    z = np.load(FIELD_NPZ, allow_pickle=False)
    g = z["grid"]                       # uint16, quantized
    q = float(z["quant"])
    cell = float(z["cell"])
    reach = float(z["reach_max"])
    mins = np.asarray(z["mins"], np.float64)
    vq = (reach + 0.5 * cell) / q
    # densest z-slab, sampled coarsely so the scan is cheap
    step = 8
    occ = (g[::step] < vq).sum(axis=(1, 2))
    k = int(np.argmax(np.convolve(occ, np.ones(20), "same"))) * step
    lo = max(0, k - 80)
    hi = min(g.shape[0], lo + 160)
    block = g[lo:hi].astype(np.float32) * q
    del g, z
    bmins = mins + np.array([0.0, 0.0, lo * cell])
    f = GoalField(block, bmins, cell, reach)
    assert (block < f._valid_max).mean() > 0.01, "picked an empty slab"
    return f


@pytest.fixture(scope="module")
def core():
    from surfgym import SurfCore, default_config
    return SurfCore(str(CANNONBALL), default_config(
        num_envs=1024, spawn_mode=2, max_episode_ticks=200,
        sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))


@pytest.fixture(scope="module")
def sampler(core, real_field):
    return RandomSpawnSampler(core, real_field, seed=5)


def _start_pool(n: int = 8) -> np.ndarray:
    """A stand-in for map_spawn_pool: origins nothing else can produce."""
    p = np.zeros(n, STATE_DTYPE)
    p["origin"][:, 0] = 1.0e6 + np.arange(n)
    p["onground"] = -1
    return p


# ==========================================================================
# 1. the mix: 5% start / 95% random, measured on real C resets
# ==========================================================================
@needs_map
@needs_field
def test_mix_ratio_over_10k_real_resets(core, sampler):
    start = _start_pool()
    pool = sampler.build_pool(start, pool_size=4096)
    assert len(pool) == 4096
    core.set_spawn_pool(pool)
    n_start = n_tot = 0
    for s in range(10):                       # 10 x 1024 = 10,240 draws
        core.reset(seed=1234 + s)
        o = np.asarray(core.states_view["origin"], np.float64)
        n_start += int((o[:, 0] > 5.0e5).sum())
        n_tot += len(o)
    assert n_tot >= 10_000
    frac = n_start / n_tot
    # binomial sd at p=0.05, n=10240 is 0.22 pp; 1 pp is ~4.5 sd
    assert 0.04 < frac < 0.06, f"start share {frac:.4f} over {n_tot} resets"


@needs_map
@needs_field
def test_pool_entry_counts_are_the_probabilities(core, sampler):
    pool = sampler.build_pool(_start_pool(), pool_size=2000)
    n_start = int((np.asarray(pool["origin"])[:, 0] > 5.0e5).sum())
    assert n_start == 100                     # round(2000 * 0.05)


@needs_map
@needs_field
def test_start_frac_is_configurable(core, real_field):
    s = RandomSpawnSampler(core, real_field, start_frac=0.25, seed=7)
    pool = s.build_pool(_start_pool(), pool_size=800)
    n_start = int((np.asarray(pool["origin"])[:, 0] > 5.0e5).sum())
    assert n_start == 200


# ==========================================================================
# 2. every sampled position: finite potential + the hull fits
# ==========================================================================
@needs_map
@needs_field
def test_every_position_has_finite_potential_and_fits(core, sampler):
    rows = sampler.sample_states(512)
    o = np.asarray(rows["origin"], np.float64)
    assert len(o) == 512
    d = sampler.field.sample(o)
    assert np.isfinite(d).all()
    assert (d < sampler.field.reach_max - 0.5 * sampler.field.cell).all(), (
        "a sampled state has no honest potential")
    assert sampler.field.reachable(o).all()
    bad = [i for i, p in enumerate(o) if core.trace(p, p, hull=0).startsolid]
    assert not bad, f"{len(bad)} sampled states do not fit the standing hull"


@needs_map
@needs_field
def test_positions_lie_inside_the_field_box(core, sampler):
    f = sampler.field
    o = np.asarray(sampler.sample_states(256)["origin"], np.float64)
    maxs = f.mins + np.asarray(f.grid.shape[::-1]) * f.cell
    assert (o >= f.mins - 0.5 * f.cell).all()
    assert (o <= maxs + 0.5 * f.cell).all()


@needs_map
@needs_field
def test_rejection_is_actually_exercised(core, real_field):
    """The hull clearance test rejects a real, non-zero share of points -
    a 32 u cell whose centre is free can still clip a wall, and if this
    were 0 the check would be decorative."""
    s = RandomSpawnSampler(core, real_field, seed=13)
    s.sample_states(2048)
    st = s.d_stats()
    assert 0.0 < st["accept"] < 1.0
    assert 0.0 < st["hull_reject"] < 0.5, st["hull_reject"]
    assert s.hull_rejected > 0


# ==========================================================================
# 3. the ranges
# ==========================================================================
@needs_map
@needs_field
def test_speed_yaw_pitch_ranges(core, sampler):
    rows = sampler.sample_states(4096)
    v = np.asarray(rows["velocity"], np.float64)
    spd = np.hypot(v[:, 0], v[:, 1])
    assert spd.min() >= 1000.0 - 1e-2 and spd.max() <= 4000.0 + 1e-2
    assert spd.min() < 1100.0 and spd.max() > 3900.0        # covers the band
    assert (v[:, 2] == 0.0).all(), "vertical spawn velocity must be 0"

    yaw = np.asarray(rows["yaw"], np.float64)
    assert yaw.min() >= -180.0 and yaw.max() < 180.0
    assert yaw.min() < -170.0 and yaw.max() > 170.0

    pitch = np.asarray(rows["pitch"], np.float64)
    assert pitch.min() >= -30.0 - 1e-4 and pitch.max() <= 15.0 + 1e-4
    assert pitch.min() < -29.0 and pitch.max() > 14.0

    assert (np.asarray(rows["onground"]) == -1).all()
    # nothing else is set: the C reset zeroes the struct and copies the row,
    # so a stale tick/progress column would survive into the new episode
    for f in ("tick", "stuck_ticks", "ducked", "progress", "best_progress"):
        assert (np.asarray(rows[f]) == 0).all(), f


@needs_map
@needs_field
def test_heading_follows_the_view_yaw_with_30deg_noise(core, sampler):
    rows = sampler.sample_states(8192)
    v = np.asarray(rows["velocity"], np.float64)
    yaw = np.asarray(rows["yaw"], np.float64)
    head = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    err = (head - yaw + 180.0) % 360.0 - 180.0
    assert abs(err.mean()) < 1.5, "heading is biased off the view"
    assert 27.0 < err.std() < 33.0, f"heading sd {err.std():.1f}, want ~30"


@needs_map
@needs_field
def test_speed_range_is_configurable(core, real_field):
    s = RandomSpawnSampler(core, real_field, speed_range=(200.0, 300.0),
                           seed=11)
    v = np.asarray(s.sample_states(256)["velocity"], np.float64)
    spd = np.hypot(v[:, 0], v[:, 1])
    assert spd.min() >= 200.0 - 1e-3 and spd.max() <= 300.0 + 1e-3


# ==========================================================================
# 4. round-trip through the core
# ==========================================================================
@needs_map
@needs_field
def test_state_round_trips_through_the_core(core, sampler):
    rows = sampler.sample_states(64)
    for i in range(64):
        core.set_state(i, rows[i])
    got = np.asarray(core.states_view[:64]).copy()
    for f in ("origin", "velocity", "yaw", "pitch", "onground"):
        assert np.array_equal(np.asarray(got[f]), np.asarray(rows[f])), f


@needs_map
@needs_field
def test_reset_reproduces_the_pool_rows(core, sampler):
    """A pool draw is a full state restore, not just a teleport: the
    velocity and the view come back too (the C reset adds only the env's
    own yaw jitter, which is 0 in this config)."""
    rows = sampler.sample_states(512)
    core.set_spawn_pool(rows)
    core.reset(seed=99)
    o = np.asarray(core.states_view["origin"], np.float64)
    v = np.asarray(core.states_view["velocity"], np.float64)
    src_o = np.asarray(rows["origin"], np.float64)
    # every reset origin is one of the pool rows
    keys = {tuple(np.round(r, 3)) for r in src_o}
    assert all(tuple(np.round(r, 3)) in keys for r in o[:64])
    spd = np.hypot(v[:, 0], v[:, 1])
    assert spd.min() >= 1000.0 - 1e-2 and spd.max() <= 4000.0 + 1e-2


# ==========================================================================
# 5. the diagnostic that replaces reservoir min-depth
# ==========================================================================
@needs_map
@needs_field
def test_d_stats_describe_where_the_starts_landed(core, sampler):
    sampler.sample_states(1024)
    st = sampler.d_stats()
    for k in ("n", "min", "p10", "median", "p90", "max", "accept",
              "hull_reject"):
        assert k in st
    assert st["n"] == 1024
    assert st["min"] <= st["p10"] <= st["median"] <= st["p90"] <= st["max"]
    assert st["max"] < sampler.field.reach_max
    assert "randspawn" in sampler.d_line()


# ==========================================================================
# 6. FLAG OFF is the old trainer
# ==========================================================================
def test_flag_defaults_off():
    assert '"--respawn-random", action="store_true", default=None' in TRAIN_SRC
    assert ("    if args.respawn_random is None:\n"
            "        args.respawn_random = False\n") in TRAIN_SRC


def test_slot_has_no_sampler_by_default():
    s = MapSlot.__new__(MapSlot)
    s._init_fields("m", "m.bsp", _FakeCore(4), 0, 4)
    assert s.rand_spawn is None
    assert s.respawn is None


class _FakeCore:
    def __init__(self, n):
        self.num_envs = n


def test_reservoir_branch_is_reached_by_its_old_condition():
    """Flag off -> the elif is the untouched ``args.respawn_frac > 0.0``
    branch, with the same body it always had."""
    assert ("    if args.respawn_random:\n") in TRAIN_SRC
    assert ("    elif args.respawn_frac > 0.0:\n"
            "        # a reservoir per map: its states are RAW MAP COORDINATES,"
            ) in TRAIN_SRC
    # the per-iteration refresh: the random branch is tried FIRST and only
    # when a sampler exists, so with the flag off the reservoir branch is
    # evaluated exactly as before
    assert ("            elif _s.rand_spawn is not None:\n") in TRAIN_SRC
    assert ("            elif _s.respawn is not None and _s.respawn.size >= 2000:\n"
            ) in TRAIN_SRC


def test_flag_is_recorded_in_run_json_and_restored():
    assert '"respawn_random": args.respawn_random,' in TRAIN_SRC
    assert '"respawn_random_start_frac": (' in TRAIN_SRC
    assert '"respawn_random_speed": args.respawn_random_speed,' in TRAIN_SRC
    assert 'ck_cfg.get("respawn_random")' in TRAIN_SRC


def test_flag_turns_the_reservoir_off():
    """No reservoir under --respawn-random: min-depth and the stagnant mask
    have nothing to describe, which is why the ledger reports the sampled
    potential distribution instead."""
    i = TRAIN_SRC.index("    if args.respawn_random:\n")
    j = TRAIN_SRC.index("    elif args.respawn_frac > 0.0:\n", i)
    body = TRAIN_SRC[i:j]
    assert "RespawnBuffer(" not in body
    assert "respawn = None" in body
    assert "RandomSpawnSampler(" in body
