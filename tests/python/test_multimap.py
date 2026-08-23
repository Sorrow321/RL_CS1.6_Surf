"""``--maps``: one shared policy over several maps, one core per map.

Four things this pins, in the order they would hurt if they broke.

  * **``--map`` alone is bit-identical to the pre-``--maps`` trainer.** Every
    arm in flight resumes a checkpoint trained on that path, and surf is
    chaotic enough that one differing depth pixel forks the whole greedy
    trajectory (CLAUDE.md: the lidar march is not even bit-exact across GPU
    architectures, and that alone re-rolls an eval). So a one-slot
    :class:`MapFleet` is driven beside a hand-rolled reference core over
    ~300 ticks and compared on obs, base rewards, done, trunc, terminal obs,
    the shaped reward AND the torch RNG cursor - the fleet must not consume
    a single extra draw, or every sampled action after it differs.
  * **the reward scale is per map.** ``scale = 100/d0`` of THAT map, so a
    start-to-finish run is worth 100 on a 198,380 u map and on a 35,637 u
    one. A shared scale would make the short map's whole race worth 18
    points and the policy would learn to ignore it.
  * **env slicing.** Slot i owns rows [lo, hi) of the action array and its
    outputs land back in exactly those rows. Off-by-one here is silent: the
    trainer would step map A's core with map B's actions and still produce
    plausible numbers.
  * **the latch is a FRACTION of d0.** 6,996 u is 3.53% of cannonball's d0
    and 19.6% of petrus_lite's; the same absolute number is two different
    treatments, so ``--race-latch`` stays single-map and
    ``--race-latch-frac`` is what a multi-map run passes.

    python -m pytest tests/python/test_multimap.py -q
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                       # noqa: E402
from surfgym.goalfield import EuclidField                  # noqa: E402
from surfgym.mapfleet import (MapFleet, MapSlot, map_tag,  # noqa: E402
                              map_tags)
from surfgym.rewards import RaceReward                     # noqa: E402

CANNONBALL = ROOT / "maps" / "surf_src_cannonball.bsp"
SKI = ROOT / "maps" / "surf_ski_2.bsp"

# the two real d0 values, measured (docs/ideas-backlog.md item 0)
D0_CANNONBALL = 198_380.0
D0_PETRUS = 35_637.0
LATCH_ABS = 6996.0          # the absolute latch xLAT3 ran with, on cannonball


# --------------------------------------------------------------------------
# DLL-free fakes: enough of SurfCore for MapFleet's aggregation
# --------------------------------------------------------------------------
class FakeCore:
    """Records the actions it was handed and returns row-identifiable obs."""

    OBS = 15

    def __init__(self, n, base=0.0, bounds=((-1e4,) * 3, (1e4,) * 3)):
        self.num_envs = int(n)
        self.obs_dim = self.OBS
        self.base = float(base)
        self.states_view = np.zeros(n, dtype=STATE_DTYPE)
        self.goal_hits = np.zeros(n, np.uint8)
        self.seen = None            # last action block, as given
        self.reset_seed = None
        self.failed = None
        self._bounds = (np.asarray(bounds[0], np.float32),
                        np.asarray(bounds[1], np.float32))

    def map_bounds(self):
        return self._bounds

    def _rows(self, off):
        r = np.arange(self.num_envs, dtype=np.float32) + self.base + off
        return np.repeat(r[:, None], self.OBS, axis=1)

    def reset(self, seed=0):
        self.reset_seed = seed
        return self._rows(0.0)

    def step(self, actions):
        assert actions.shape == (self.num_envs, 6), actions.shape
        assert actions.flags["C_CONTIGUOUS"]
        self.seen = actions.copy()
        return (self._rows(0.1), self._rows(0.2).astype(np.float32)[:, 0],
                np.zeros(self.num_envs, np.uint8),
                np.zeros(self.num_envs, np.uint8), self._rows(0.3))

    def force_fail(self, mask):
        self.failed = np.asarray(mask).copy()


class TagReward:
    """A reward that just reports which slot ran it, per row."""

    def __init__(self, tag):
        self.tag = float(tag)

    def on_reset(self, core):
        pass

    def __call__(self, prev_obs, obs, term, base, done, trunc, core, goal=None):
        return np.full(len(done), self.tag, np.float32)


def _slots(sizes, rewards=None):
    out, lo = [], 0
    for i, n in enumerate(sizes):
        s = MapSlot(f"surf_map{i}", f"/x/surf_map{i}.bsp",
                    FakeCore(n, base=100.0 * i), lo, lo + n)
        s.reward_fn = (rewards[i] if rewards else TagReward(i + 1))
        s.map_center = np.array([1000.0 * i, 0.0, 0.0], np.float32)
        out.append(s)
        lo += n
    return out


# --------------------------------------------------------------------------
# map tags
# --------------------------------------------------------------------------
def test_map_tag_strips_the_shipped_prefixes():
    assert map_tag("surf_src_cannonball") == "cannonball"
    assert map_tag("surf_petrus_lite") == "petrus_lite"
    assert map_tag("kz_something") == "kz_something"


def test_map_tags_fall_back_to_full_stems_on_a_collision():
    # surf_x and surf_src_x would both tag as "x"; a CSV column has to stay
    # unambiguous, so the whole set reverts to stems
    assert map_tags(["surf_x", "surf_src_x"]) == ["surf_x", "surf_src_x"]
    assert map_tags(["surf_src_cannonball", "surf_petrus_lite"]) \
        == ["cannonball", "petrus_lite"]


# --------------------------------------------------------------------------
# env slicing
# --------------------------------------------------------------------------
def test_each_slot_steps_only_its_own_action_rows():
    f = MapFleet(_slots([3, 5]))
    act = np.arange(8 * 6, dtype=np.int32).reshape(8, 6)
    obs, base, done, trunc, term = f.step(act)
    a, b = f.slots
    assert np.array_equal(a.core.seen, act[0:3])
    assert np.array_equal(b.core.seen, act[3:8])
    # ... and each slot's obs lands back in its own rows (the fakes encode
    # their row index plus a 100-per-slot offset)
    assert np.allclose(obs[0:3, 0], [0.1, 1.1, 2.1])
    assert np.allclose(obs[3:8, 0], [100.1, 101.1, 102.1, 103.1, 104.1])
    assert obs.shape == (8, 15) and term.shape == (8, 15)
    assert len(base) == 8 and len(done) == 8 and len(trunc) == 8


def test_reward_rows_come_from_the_owning_slots_reward_function():
    f = MapFleet(_slots([3, 5]))
    z = np.zeros((8, 15), np.float32)
    r = f.reward(z, z, z, np.zeros(8, np.float32),
                 np.zeros(8, np.uint8), np.zeros(8, np.uint8))
    assert np.array_equal(r[:3], np.full(3, 1.0, np.float32))
    assert np.array_equal(r[3:], np.full(5, 2.0, np.float32))


def test_ranges_must_be_contiguous_and_ordered():
    a = MapSlot("a", "a.bsp", FakeCore(4), 0, 4)
    b = MapSlot("b", "b.bsp", FakeCore(4), 8, 12)     # gap
    with pytest.raises(ValueError):
        MapFleet([a, b])


def test_a_slots_range_must_match_its_cores_env_count():
    with pytest.raises(ValueError):
        MapSlot("a", "a.bsp", FakeCore(4), 0, 5)


def test_map_centers_are_per_row():
    f = MapFleet(_slots([2, 2]))
    mc = f.map_centers(np.array([0, 3]))
    assert np.array_equal(mc[0], [0.0, 0.0, 0.0])
    assert np.array_equal(mc[1], [1000.0, 0.0, 0.0])


def test_goal_hits_and_force_fail_route_per_slot():
    f = MapFleet(_slots([3, 3]))
    a, b = f.slots
    b.core.goal_hits = np.array([0, 1, 0], np.uint8)
    assert np.array_equal(f.goal_hits(), [0, 0, 0, 0, 1, 0])

    class Stalling(TagReward):
        def __init__(self, tag, mask):
            super().__init__(tag)
            self.mask = mask

        def pop_stall_mask(self):
            return self.mask
    a.reward_fn = Stalling(1, np.array([1, 0, 0], np.uint8))
    b.reward_fn = Stalling(2, None)
    f.apply_stall_kills()
    assert np.array_equal(a.core.failed, [1, 0, 0])
    assert b.core.failed is None


def test_single_slot_returns_the_cores_own_buffers_untouched():
    """The identity the bit-identity guarantee rests on: with one map the
    fleet is a pass-through, not a copy into a staging array."""
    f = MapFleet(_slots([4]))
    act = np.zeros((4, 6), np.int32)
    obs = f.step(act)[0]
    assert f.single
    # the fake hands back a fresh array each call; what matters is that the
    # fleet did not allocate staging buffers at all
    assert not hasattr(f, "_obs")
    assert obs.shape == (4, 15)
    assert np.array_equal(f.slots[0].core.seen, act)


# --------------------------------------------------------------------------
# per-map reward scale
# --------------------------------------------------------------------------
def _race(d0, box, **kw):
    return RaceReward(EuclidField(box), scale=100.0 / d0, time_pen=0.0,
                      success_bonus=0.0, stall_ticks=0, **kw)


def test_the_same_fraction_of_a_map_pays_the_same_on_both_maps():
    """`scale = 100/d0` per map is the whole point: finishing ANY map is
    worth 100. Two maps 5.6x apart in length, each walked 10% of the way in,
    must pay the same 10 reward."""
    long_box = {"mins": [0.0, 0.0, 0.0], "maxs": [1.0, 1.0, 1.0]}
    r_long = _race(D0_CANNONBALL, long_box)
    r_short = _race(D0_PETRUS, long_box)
    assert r_long.scale == pytest.approx(100.0 / D0_CANNONBALL)
    assert r_short.scale == pytest.approx(100.0 / D0_PETRUS)

    def walk(rw, d0):
        core = FakeCore(1)
        core.states_view["origin"] = np.array([[d0, 0.0, 0.0]], np.float32)
        rw.on_reset(core)
        core.states_view["origin"] = np.array([[0.9 * d0, 0.0, 0.0]],
                                              np.float32)
        z = np.zeros((1, 15), np.float32)
        return float(rw(z, z, z, np.zeros(1, np.float32),
                        np.zeros(1, np.uint8), np.zeros(1, np.uint8),
                        core)[0])

    # the clip is 100u/tick, so walk it in one call with max_step raised
    r_long.max_step = r_short.max_step = 1e9
    assert walk(r_long, D0_CANNONBALL) == pytest.approx(10.0, rel=1e-4)
    assert walk(r_short, D0_PETRUS) == pytest.approx(10.0, rel=1e-4)


def test_a_shared_scale_would_make_the_short_map_nearly_invisible():
    """The failure this guards against, stated as a number: on one scale
    (the long map's), the short map's ENTIRE race is worth 18 points."""
    shared = 100.0 / D0_CANNONBALL
    assert D0_PETRUS * shared == pytest.approx(17.965, abs=0.01)


# --------------------------------------------------------------------------
# the latch as a fraction of d0
# --------------------------------------------------------------------------
def test_latch_frac_resolves_to_a_different_distance_per_map():
    frac = LATCH_ABS / D0_CANNONBALL
    assert frac == pytest.approx(0.03527, abs=1e-5)
    assert frac * D0_CANNONBALL == pytest.approx(LATCH_ABS)
    # the same fraction on the short map is 1,257u, not 6,996u - which is
    # 19.6% of that map and would switch the shaping off a fifth of the way
    # down it
    assert frac * D0_PETRUS == pytest.approx(1257.0, abs=1.0)
    assert LATCH_ABS / D0_PETRUS == pytest.approx(0.196, abs=0.001)


def test_the_latch_flag_uses_each_slots_own_threshold():
    box = {"mins": [0.0, 0.0, 0.0], "maxs": [1.0, 1.0, 1.0]}
    frac = LATCH_ABS / D0_CANNONBALL
    slots = _slots([1, 1], rewards=[
        _race(D0_CANNONBALL, box, d_latch=frac * D0_CANNONBALL),
        _race(D0_PETRUS, box, d_latch=frac * D0_PETRUS)])
    f = MapFleet(slots)
    # 3,000u from the finish on both maps: past the LONG map's latch
    # (6,996u -> armed) and short of the SHORT map's (1,257u -> not armed)
    for s in slots:
        s.core.states_view["origin"] = np.array([[3000.0, 0.0, 0.0]],
                                                np.float32)
        s.reward_fn.on_reset(s.core)
    assert list(f.latch_flags()) == [True, False]


def test_terminal_latch_reads_each_rows_own_field_and_threshold():
    box = {"mins": [0.0, 0.0, 0.0], "maxs": [1.0, 1.0, 1.0]}
    frac = LATCH_ABS / D0_CANNONBALL
    slots = _slots([1, 1], rewards=[
        _race(D0_CANNONBALL, box, d_latch=frac * D0_CANNONBALL),
        _race(D0_PETRUS, box, d_latch=frac * D0_PETRUS)])
    for s in slots:
        s.reward_field = s.reward_fn.field
        s.core.states_view["origin"] = np.array([[1e6, 0.0, 0.0]], np.float32)
        s.reward_fn.on_reset(s.core)          # far away: nothing latched
    f = MapFleet(slots)
    pos = np.array([[3000.0, 0.0, 0.0], [3000.0, 0.0, 0.0]], np.float64)
    assert list(f.terminal_latch(np.array([0, 1]), pos)) == [True, False]


# --------------------------------------------------------------------------
# per-map episode statistics pool honestly
# --------------------------------------------------------------------------
def test_pop_stats_pools_success_rate_by_EPISODES_not_by_map():
    box = {"mins": [0.0, 0.0, 0.0], "maxs": [1.0, 1.0, 1.0]}
    slots = _slots([1, 1], rewards=[_race(1e5, box), _race(1e5, box)])
    slots[0].reward_fn.n_success, slots[0].reward_fn.n_fail = 1, 99
    slots[1].reward_fn.n_success, slots[1].reward_fn.n_fail = 9, 1
    st = MapFleet(slots).pop_stats()
    assert st["episodes"] == 110
    assert st["success_rate"] == pytest.approx(10.0 / 110.0)


# --------------------------------------------------------------------------
# CLI guards (no GPU, no run directory)
# --------------------------------------------------------------------------
def _train(*argv):
    return subprocess.run(
        [sys.executable, str(ROOT / "python" / "train_fast.py"), *argv],
        capture_output=True, text=True, timeout=600)


def test_the_two_latch_forms_are_mutually_exclusive():
    p = _train("--race-latch", "6996", "--race-latch-frac", "0.035")
    assert p.returncode != 0
    assert "same setting in two units" in (p.stdout + p.stderr)


# two maps that are in the repo, so these exercise the GUARDS rather than
# the path resolver (petrus_lite is gitignored)
TWO = "surf_src_cannonball,surf_ski_2"


def test_envs_must_divide_evenly_over_the_maps():
    p = _train("--reward", "race", "--envs", "7", "--maps", TWO)
    assert p.returncode != 0
    assert "does not divide evenly" in p.stdout + p.stderr


def test_an_absolute_latch_is_refused_on_a_multi_map_run():
    p = _train("--reward", "race", "--envs", "8", "--race-latch", "6996",
               "--maps", TWO)
    assert p.returncode != 0
    assert "--race-latch-frac" in p.stdout + p.stderr


def test_multi_map_needs_the_race_objective():
    p = _train("--envs", "8", "--maps", TWO)
    assert p.returncode != 0
    assert "needs --reward race" in p.stdout + p.stderr


def test_an_absolute_dfloor_is_refused_on_a_multi_map_run():
    p = _train("--reward", "race", "--envs", "8", "--race-dfloor", "6996",
               "--maps", TWO)
    assert p.returncode != 0
    assert "ABSOLUTE distance" in p.stdout + p.stderr


# --------------------------------------------------------------------------
# THE bit-identity test: one slot == the pre---maps trainer, on a real core
# --------------------------------------------------------------------------
def _have_dll():
    try:
        from surfgym import SurfCore, default_config
        SurfCore(str(SKI), default_config(num_envs=1, lidar_w=0, lidar_h=0))
        return True
    except Exception:
        return False


needs_core = pytest.mark.skipif(
    not (SKI.exists() and _have_dll()),
    reason="needs surf_ski_2.bsp and a built surfcore library")

BOX = {"mins": [-64.0, -64.0, -64.0], "maxs": [64.0, 64.0, 64.0]}
TICKS = 300
NENV = 16
# Two regimes, because whichever limit is lower fires for the WHOLE fleet:
# a 150-tick stall kill ends every episode as a FAIL (done), and with the
# kill off a 150-tick episode cap ends every one as a TRUNCATION. Running
# both is what puts done, trunc AND terminal_obs under the comparison.
REGIMES = [(400, 150), (150, 0)]


def _actions(n, ticks, seed=7):
    rng = np.random.default_rng(seed)
    return [np.ascontiguousarray(
        rng.integers([0, 0, 0, 0, 0, 0], [15, 7, 3, 3, 2, 2],
                     size=(n, 6)).astype(np.int32)) for _ in range(ticks)]


def _make_core(bsp, n, cap):
    from surfgym import SurfCore, default_config
    return SurfCore(str(bsp), default_config(
        num_envs=n, max_episode_ticks=cap, water_fail=1,
        lidar_w=0, lidar_h=0))


def _reward(field, stall):
    return RaceReward(field, scale=100.0 / 5000.0, time_pen=0.005,
                      success_bonus=50.0, stall_ticks=stall, int_coef=0.0)


def _drive_reference(bsp, acts, seed, n=NENV, cap=400, stall=150):
    """The trainer's inner loop as it was BEFORE --maps: one core, one
    reward function, called directly."""
    core = _make_core(bsp, n, cap)
    rw = _reward(EuclidField(BOX), stall)
    obs = core.reset(seed).copy()
    rw.on_reset(core)
    prev = obs.copy()
    log = []
    for a in acts:
        sm = rw.pop_stall_mask()
        if sm is not None:
            core.force_fail(sm)
        o2, base, done, trunc, term = core.step(a)
        r = rw(prev, o2, term, base, done, trunc, core)
        prev = o2.copy()
        log.append((o2.copy(), np.asarray(base).copy(), done.copy(),
                    trunc.copy(), term.copy(), r.copy(),
                    core.goal_hits.copy()))
    core.close()
    return obs, log


def _drive_fleet(bsps, acts, seed, cap=400, stall=150):
    """The same loop through MapFleet."""
    slots, lo = [], 0
    per = NENV // len(bsps)
    for bsp in bsps:
        c = _make_core(bsp, per, cap)
        s = MapSlot(Path(bsp).stem, str(bsp), c, lo, lo + per)
        s.reward_field = s.goal_field = EuclidField(BOX)
        s.reward_fn = _reward(s.reward_field, stall)
        mn, mx = c.map_bounds()
        s.map_center = ((mn + mx) / 2.0).astype(np.float32)
        slots.append(s)
        lo += per
    f = MapFleet(slots)
    obs = f.reset(seed).copy()
    f.on_reset()
    prev = obs.copy()
    log = []
    for a in acts:
        f.apply_stall_kills()
        o2, base, done, trunc, term = f.step(a)
        r = f.reward(prev, o2, term, base, done, trunc)
        prev = o2.copy()
        log.append((o2.copy(), np.asarray(base).copy(), done.copy(),
                    trunc.copy(), term.copy(), r.copy(),
                    f.goal_hits().copy()))
    for s in slots:
        s.core.close()
    return obs, log


def _nondegenerate(log, kind):
    """A bit-identity test over two dead trajectories proves nothing. This
    is the guard that the compared run actually surfed and ended episodes."""
    paid = sum(int((row[5] != 0).sum()) for row in log)
    n_done = sum(int(row[2].sum()) for row in log)
    n_trunc = sum(int(row[3].sum()) for row in log)
    assert paid > 100, f"only {paid} nonzero rewards - degenerate rollout"
    assert (n_done if kind == "done" else n_trunc) > 0, \
        f"no episode ended as {kind} ({n_done} done, {n_trunc} trunc)"


NAMES = ("obs", "base_reward", "done", "trunc", "terminal_obs",
         "shaped_reward", "goal_hits")


def _assert_same(ref, got):
    (o0, log0), (o1, log1) = ref, got
    assert np.array_equal(o0, o1), "reset obs differ"
    assert len(log0) == len(log1)
    for t, (a, b) in enumerate(zip(log0, log1)):
        for name, x, y in zip(NAMES, a, b):
            assert np.array_equal(x, y), f"{name} differs at tick {t}"


@needs_core
@pytest.mark.parametrize("cap,stall,kind",
                         [(400, 150, "done"), (150, 0, "trunc")])
def test_one_slot_is_bit_identical_to_the_pre_maps_single_map_path(
        cap, stall, kind):
    acts = _actions(NENV, TICKS)
    ref = _drive_reference(SKI, acts, seed=1234, cap=cap, stall=stall)
    got = _drive_fleet([SKI], acts, seed=1234, cap=cap, stall=stall)
    _nondegenerate(ref[1], kind)
    _assert_same(ref, got)


@needs_core
def test_one_slot_consumes_exactly_as_many_torch_rng_draws():
    """Not decoration: PPO samples actions from torch's global generator, so
    ONE extra draw anywhere in the rollout re-rolls every action after it and
    the resumed arm diverges from its own baseline immediately."""
    acts = _actions(NENV, 40)

    torch.manual_seed(99)
    _drive_reference(SKI, acts, seed=5)
    after_ref = torch.rand(4)

    torch.manual_seed(99)
    _drive_fleet([SKI], acts, seed=5)
    after_fleet = torch.rand(4)

    assert torch.equal(after_ref, after_fleet)


@needs_core
@pytest.mark.skipif(not CANNONBALL.exists(),
                    reason="needs surf_src_cannonball.bsp")
def test_two_slots_reproduce_two_independent_single_map_runs():
    """The slicing check with real physics: the fleet's rows must equal what
    two separate single-map trainers would have produced on the same
    actions, halves for halves."""
    half = NENV // 2
    acts = _actions(NENV, TICKS)
    got_obs, got = _drive_fleet([SKI, CANNONBALL], acts, seed=77)
    _nondegenerate(got, "done")

    # each reference core runs the fleet's own per-slot seed and its own
    # half of every action row
    ref_a = _drive_reference(SKI, [np.ascontiguousarray(a[:half])
                                   for a in acts], seed=77, n=half)
    ref_b = _drive_reference(CANNONBALL, [np.ascontiguousarray(a[half:])
                                          for a in acts],
                             seed=77 + 1013, n=half)

    assert np.array_equal(got_obs[:half], ref_a[0])
    assert np.array_equal(got_obs[half:], ref_b[0])
    for t, row in enumerate(got):
        for k, name in enumerate(NAMES):
            assert np.array_equal(row[k][:half], ref_a[1][t][k]), (t, name)
            assert np.array_equal(row[k][half:], ref_b[1][t][k]), (t, name)
