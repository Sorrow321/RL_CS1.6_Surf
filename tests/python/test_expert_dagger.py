"""DAgger relabelling (surfgym/dagger.py, tools/expert_dagger.py).

CPU only. The sim-side loops run under a stub core (N envs on a line, each
at its own speed, dying and finishing on schedule) and a stub decider (a
known action per env), so what is pinned is the BOOKKEEPING the labels
depend on:

  1. sampling picks states at the requested spacing and drops an env at
     its episode end;
  2. the window search labels a state with the FIRST decision of the
     lineage that wins - after cloning, that is the donor's decision, not
     the winning env's own; a finisher wins by time; a group with nothing
     alive is extinct; the feeds are primed per copy;
  3. the rows come out in surfgym.bc's format and BCDataset loads the
     merged file; weights follow the divergence rule and the share;
  4. the default-off path leaves the driver's round unchanged, and the
     hook is the only edit to expert_loop.py;
  5. the PHYSICS TICK is the checkpoint's, not 100 Hz: every core the phase
     opens is built through beam_tas.build_sim(..., tick=) and checked
     against the clock, the --obs-reward mirror is built at the same tick,
     and every seconds flag converts at it - so a synthetic 7.63 ms relabel
     lands on the 130.4 Hz grid and a 10 ms one is the legacy arithmetic
     bit for bit.

Plus, when the map, its caches and the exit10 round-23 checkpoint are
present, a CPU dry run of the whole phase on 2 states with a 16-env core
(the C core, the real policy, the torch lidar) - the restart is faithful
(the window's decision-0 row equals the sampled row).

    python -m pytest tests/python/test_expert_dagger.py -q
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.bc import BCDataset, N_SCALAR, save_bc_dataset   # noqa: E402
from surfgym.core import STATE_DTYPE                            # noqa: E402
from surfgym.dagger import (DAGGER_LINE_ID, SRC_GREEDY, SRC_SPINE,  # noqa: E402
                            SRC_STOCH, SampleBank, check_core_tick,
                            collect_rollout_samples, core_clock,
                            decision_grid, divergence_weights, even_subset,
                            merge_bc_datasets, nearest_distance, on_grid,
                            relabel_windows, rows_from_results,
                            summarize_results)
from surfgym.tick import TickClock                              # noqa: E402
import expert_dagger                                            # noqa: E402

MAIN_MAP = Path("C:/RL_Surf/maps/surf_src_cannonball.bsp")
EXIT_CKPT = Path("C:/RL_Surf_exit/runs/exit10/round_23/train/ckpt_final.pt")
EXIT_BC = Path("C:/RL_Surf_exit/runs/exit10/round_23/bc.npz")
EXIT_SPINE = Path("C:/RL_Surf_exit/runs/exit10/round_23/spine.npy")


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------
class FakeCore:
    """N envs on a line. Env i advances x by speed[i] per tick (speed is a
    property of the ENV INDEX, not of the state: a clone keeps its own
    speed, which is what makes the cloning test bite), dies on the core's
    death_tick[i]-th step since the reset (once; same-step autoreset to
    x=0, tick=0, like the C core) and finishes when x >= goal_x."""

    def __init__(self, n, speed, death_tick=None, goal_x=np.inf):
        self.num_envs = int(n)
        self.speed = np.asarray(speed, np.float64).reshape(n)
        self.death = (np.full(n, 10 ** 9) if death_tick is None
                      else np.asarray(death_tick, np.int64).reshape(n))
        self.goal_x = float(goal_x)
        self._st = np.zeros(n, STATE_DTYPE)
        self._obs = np.zeros((n, N_SCALAR), np.float32)
        self._hits = np.zeros(n, bool)
        self.bsp_path = "fake.bsp"
        self.nstep = 0

    @property
    def states_view(self):
        return self._st

    @property
    def goal_hits(self):
        return self._hits

    def get_states(self):
        return self._st.copy()

    def set_state(self, i, row):
        self._st[i] = np.asarray(row, STATE_DTYPE).reshape(-1)[0]

    def reset(self, seed=0):
        self._st[:] = np.zeros((), STATE_DTYPE)
        self._st["onground"] = -1
        self._hits[:] = False
        self.nstep = 0
        self._refresh()
        return self._obs

    def _refresh(self):
        self._obs[:] = 0.0
        self._obs[:, 0] = self._st["origin"][:, 0] / 1000.0
        self._obs[:, 12] = 0.5                  # the core's RAW slot 12

    def step(self, a):
        a = np.asarray(a)
        assert a.shape == (self.num_envs, 6) and a.dtype == np.int32
        n = self.num_envs
        self.nstep += 1
        self._st["origin"][:, 0] += self.speed
        self._st["tick"] += 1
        hit = self._st["origin"][:, 0] >= self.goal_x
        dead = self.death == self.nstep
        done = hit | dead
        self._hits[:] = hit
        for i in np.nonzero(done)[0]:
            self._st[i] = np.zeros((), STATE_DTYPE)
            self._st[i]["onground"] = -1
        self._refresh()
        return (self._obs, np.zeros(n, np.float32), done,
                np.zeros(n, bool), self._obs)


class LineField:
    """geodesic d = distance to the goal along x."""

    def __init__(self, goal_x):
        self.goal_x = float(goal_x)

    def sample(self, origins):
        o = np.asarray(origins, np.float64).reshape(-1, 3)
        return self.goal_x - o[:, 0]


def act_of(i, d):
    return np.array([i % 15, d % 7, 1, 0, 0, 0], np.int32)


class StubDecider:
    """Known action per (env, decision) with the act_every hold; the row it
    'saw' = the obs with slot 12 replaced by a feed value and a latch column
    appended, like the trainer's wrapper."""

    def __init__(self, n, k, feeds=(None, None), on_first=None):
        self.n, self.k = int(n), int(k)
        self._tick, self._held = 0, None
        self.feeds = feeds
        self._on_first = on_first
        self.calls = 0
        self.last_row = self.last_greedy = None

    def act(self, obs):
        if self._held is None or self._tick % self.k == 0:
            if self.calls == 0 and self._on_first is not None:
                self._on_first(self.feeds)
            d = self._tick // self.k
            a = np.stack([act_of(i, d) for i in range(self.n)]).astype(np.int32)
            row = np.zeros((self.n, N_SCALAR + 1), np.float32)
            row[:, :N_SCALAR] = np.asarray(obs, np.float32)[:, :N_SCALAR]
            row[:, 12] = 0.25
            row[:, N_SCALAR] = 1.0
            self.last_row = row
            self.last_greedy = a.copy()
            self._held = a
            self.calls += 1
        self._tick += 1
        return self._held


def line_scorer(field):
    def score(states, obs):
        d = field.sample(states["origin"])
        return -d, d
    return score


# --------------------------------------------------------------------------
# 1. sampling
# --------------------------------------------------------------------------
def test_decision_grid_is_the_requested_spacing():
    k, every = 3, 50
    d = decision_grid(1000, k, every)
    ticks = d * k
    assert ticks[0] == 0
    gaps = np.diff(ticks)
    assert gaps.min() >= every - (k - 1) and gaps.max() <= every + (k - 1)
    # the count is the number of grid points the decisions can reach
    assert len(d) == len(range(0, 1000 * k, every))
    assert all(on_grid(int(x), k, every) for x in d)
    assert not any(on_grid(int(x), k, every)
                   for x in np.setdiff1d(np.arange(1000), d)[:200])
    # k = 1: exactly every `every` ticks
    assert np.array_equal(decision_grid(301, 1, 100), [0, 100, 200, 300])
    assert len(decision_grid(0, 3, 50)) == 0


def test_even_subset_covers_the_pool():
    assert np.array_equal(even_subset(10, 20), np.arange(10))
    s = even_subset(1000, 10)
    assert len(s) == 10 and len(set(s.tolist())) == 10
    assert s[0] == 50 and s[-1] == 950          # stratum centres
    assert len(even_subset(0, 5)) == 0 and len(even_subset(5, 0)) == 0
    # with a generator: one random pick per stratum, so a pool whose
    # candidates interleave several episodes cannot alias with the stride
    r = even_subset(1000, 10, np.random.default_rng(3))
    assert len(r) == 10 and np.all(np.diff(r) > 0)
    assert all(100 * i <= x < 100 * (i + 1) for i, x in enumerate(r))
    assert not np.array_equal(r, s)
    assert np.array_equal(even_subset(1000, 10, np.random.default_rng(3)), r)


def test_rollout_sampling_spacing_and_episode_end():
    n, k, every = 4, 3, 50
    core = FakeCore(n, speed=[10, 20, 30, 40], death_tick=[10 ** 6, 10 ** 6,
                                                            75, 10 ** 6])
    obs = core.reset(0)
    dec = StubDecider(n, k)
    bank = SampleBank()
    src = np.array([SRC_GREEDY, SRC_GREEDY, SRC_STOCH, SRC_STOCH])
    info = collect_rollout_samples(core, dec, (None, None), obs, k, 300,
                                   every, src, np.arange(n), bank)
    A = bank.arrays()
    # grid ticks within 300: 0, 51, 102, 150, 201, 252 -> 6 per env, but
    # env 2 died at tick 75 and contributes only its first two
    want_ticks = [0, 51, 102, 150, 201, 252]
    for i in range(n):
        got = sorted(A["tick"][A["ep"] == i].tolist())
        assert got == (want_ticks if i != 2 else want_ticks[:2]), (i, got)
    assert info["end_tick"].tolist() == [-1, -1, 75, -1]
    assert info["alive"].tolist() == [True, True, False, True]
    assert not info["finished"].any()
    assert info["recorded"] == len(bank) == 6 * 3 + 2
    # the state is the pre-decision state (x = speed * tick), the row is
    # the one the decider built (slot 12 = the feed value, latch = 1)
    for j in range(len(bank)):
        i, t = int(A["ep"][j]), int(A["tick"][j])
        assert A["states"][j]["origin"][0] == pytest.approx(10 * (i + 1) * t)
        assert A["states"][j]["tick"] == t
        assert A["scal"][j][12] == 0.25 and A["latch"][j] == 1.0
        assert A["scal"][j][0] == pytest.approx(10 * (i + 1) * t / 1000.0)
    assert set(A["src"][A["ep"] <= 1].tolist()) == {SRC_GREEDY}
    assert set(A["src"][A["ep"] >= 2].tolist()) == {SRC_STOCH}
    # no feeds: no previous-decision internals
    assert np.isnan(A["d_prev"]).all() and (A["tick_prev"] == -1).all()
    # select / save / load round-trip
    sub = bank.select([0, 5, 7])
    assert len(sub) == 3 and sub.arrays()["tick"][1] == A["tick"][5]
    with pytest.raises(RuntimeError):
        sub.add(A["states"][0], A["scal"][0], 0, None, None, None, 0, 0, 0)
    # skip_first: decision 0 (a fresh feed's zero) is left out
    core.reset(0)
    b2 = SampleBank()
    collect_rollout_samples(core, StubDecider(n, k), (None, None), core._obs,
                            k, 120, every, SRC_GREEDY, np.arange(n), b2,
                            skip_first=True)
    assert sorted(set(b2.arrays()["tick"].tolist())) == [51, 102]


def test_rollout_sampling_records_the_feed_internals_before_the_decision():
    n, k = 2, 3
    core = FakeCore(n, speed=[10, 10])
    obs = core.reset(0)
    rf = SimpleNamespace(state={"d": None})
    lf = SimpleNamespace(state={"f": None, "tick": None})

    class FeedDecider(StubDecider):
        def act(self, obs):
            if self._held is None or self._tick % self.k == 0:
                # the real wrapper advances the feeds inside _obs
                d = self._tick // self.k
                rf.state["d"] = np.full(n, 100.0 - d)
                lf.state["f"] = np.array([d >= 2, False])
                lf.state["tick"] = np.full(n, d * k)
            return super().act(obs)

    dec = FeedDecider(n, k, (rf, lf))
    bank = SampleBank()
    collect_rollout_samples(core, dec, (rf, lf), obs, k, 40, 9, SRC_SPINE,
                            [7, 8], bank)
    A = bank.arrays()
    # grid every 9 ticks at k=3: decisions 0, 3, 6, 9, 12
    e0 = A["ep"] == 7
    assert A["tick"][e0].tolist() == [0, 9, 18, 27, 36]
    # decision 0 has no previous decision; decision 3 sees decision 2's
    # internals: d = 98, f = True (d >= 2), tick = 6
    assert np.isnan(A["d_prev"][e0][0]) and A["tick_prev"][e0][0] == -1
    assert A["d_prev"][e0][1] == 98.0 and A["f_prev"][e0][1] == 1.0
    assert A["tick_prev"][e0][1] == 6
    assert A["f_prev"][A["ep"] == 8].tolist() == [0.0] * 5
    assert set(A["src"].tolist()) == {SRC_SPINE}


# --------------------------------------------------------------------------
# 2. the window search
# --------------------------------------------------------------------------
def _bank_of(n, x0=0.0):
    bank = SampleBank()
    for s in range(n):
        st = np.zeros((), STATE_DTYPE)
        st["origin"][0] = x0
        st["tick"] = 100 + s
        scal = np.full(N_SCALAR, 0.1, np.float32)
        scal[12] = 0.7
        bank.add(st, scal, 1.0, 5.0 + s, 1.0, 90 + s, SRC_GREEDY, s, 300)
    return bank


def test_window_labels_are_the_winning_lineages_first_decisions():
    # two groups of four copies; k = 3; window 30 decisions (90 ticks),
    # cloning every 10 decisions (ticks 30 and 60); one elite per group
    # plus the protected greedy env 0 of each group
    k, copies, H, R = 3, 4, 30, 10
    speed = [50, 20, 10, 30,        # group 0: the greedy env IS the fastest
             10, 60, 20, 30]        # group 1: env 5 fastest, dies at tick 60
    death = [10 ** 6] * 8
    death[5] = 60
    field = LineField(goal_x=1e9)
    core = FakeCore(8, speed, death)
    seen = {}

    def on_first(feeds):
        rf, lf = feeds
        seen["d"] = rf.state["d"].copy()
        seen["f"] = lf.state["f"].copy()
        seen["tick"] = lf.state["tick"].copy()

    def make_feeds():
        return (SimpleNamespace(state={"d": None}),
                SimpleNamespace(state={"f": None, "tick": None}))

    def make_decider(feeds, greedy_mask):
        assert greedy_mask.tolist() == [True, False, False, False] * 2
        return StubDecider(8, k, feeds, on_first)

    samples = _bank_of(2)
    res = relabel_windows(core, make_decider, make_feeds, field,
                          line_scorer(field), samples, k, copies, H, R,
                          elite_frac=0.25, n_greedy=1, label_decisions=2)
    assert len(res) == 2 and all(r is not None for r in res)
    # the feeds were primed per copy from the samples' previous-decision
    # internals: d_prev 5 / 6, f_prev 1, tick_prev 90 / 91
    assert seen["d"].tolist() == [5.0] * 4 + [6.0] * 4
    assert seen["f"].all()
    assert seen["tick"].tolist() == [90] * 4 + [91] * 4
    # group 0: env 0 (greedy, fastest) wins outright -> its own decisions,
    # and the label agrees with the policy's greedy action
    r0 = res[0]
    assert not r0["finished"] and not r0["extinct"]
    assert r0["winner"] == 0 and r0["end_tick"] == H * k
    assert np.array_equal(r0["label_acts"][0], act_of(0, 0))
    assert np.array_equal(r0["label_acts"][1], act_of(0, 1))
    assert np.array_equal(r0["greedy_act0"], act_of(0, 0))
    assert r0["disagree"] is False and r0["greedy_alive"] is True
    assert r0["end_d"] == pytest.approx(1e9 - 50 * 90)
    assert r0["greedy_end_d"] == r0["end_d"]
    # group 1: env 5 leads, is cloned into 6 and 7 at tick 30, dies at 60;
    # env 7 (x 2700) is the elite at tick 60 and is cloned into 5 and 6;
    # env 5 (own speed 60 from x 2700) is the best at tick 90 - and its
    # history is env 5's ORIGINAL decision 0, carried through both clonings
    r1 = res[1]
    assert not r1["finished"] and not r1["extinct"]
    assert r1["winner"] == 5
    assert r1["end_d"] == pytest.approx(1e9 - (2700 + 60 * 30))
    assert np.array_equal(r1["label_acts"][0], act_of(5, 0))
    assert np.array_equal(r1["label_acts"][1], act_of(5, 1))
    # the label states: row 0 IS the sample state, row 1 the lineage's
    # state before decision 1 (env 5 at tick 3: x = 180)
    assert r1["label_states"][0]["origin"][0] == 0.0
    assert r1["label_states"][0]["tick"] == 101
    assert r1["label_states"][1]["origin"][0] == pytest.approx(180.0)
    # row 0 of the label rows is the row the decider built at decision 0
    assert r1["label_rows"].shape == (2, N_SCALAR + 1)
    assert r1["label_rows"][0][12] == 0.25 and r1["label_rows"][0][N_SCALAR] == 1.0
    assert np.array_equal(r1["greedy_act0"], act_of(4, 0))
    assert r1["disagree"] is True
    assert r1["greedy_end_d"] == pytest.approx(1e9 - 10 * 90)
    s = summarize_results(res)
    assert s["labelled"] == 2 and s["disagree"] == 1
    # the planner's gain over the greedy continuation: 0 in group 0 (the
    # greedy env won), 4500 - 900 in group 1
    assert s["gain_d_mean"] == pytest.approx((0.0 + (4500 - 900)) / 2)
    assert s["gain_positive"] == 1 and s["greedy_died"] == 0


def test_window_finisher_wins_by_time_and_extinct_groups_have_no_label():
    k, copies, H = 3, 4, 30
    # group 0 finishes: env 2 at 50/tick reaches x = 1000 at tick 20;
    # group 1 is extinct: every env dies at tick 10, before any cloning
    speed = [10, 20, 50, 30, 10, 20, 30, 40]
    death = [10 ** 6] * 4 + [10] * 4
    field = LineField(goal_x=1000.0)
    core = FakeCore(8, speed, death, goal_x=1000.0)
    samples = _bank_of(2)
    res = relabel_windows(core, lambda f, m: StubDecider(8, k, f),
                          lambda: (None, None), field, line_scorer(field),
                          samples, k, copies, H, 25, 0.25, 1,
                          label_decisions=3)
    r0, r1 = res
    assert r0["finished"] and r0["end_tick"] == 20 and r0["winner"] == 2
    assert np.array_equal(r0["label_acts"][0], act_of(2, 0))
    assert r0["label_acts"].shape == (3, 6)
    assert r0["disagree"] is True
    assert r1["extinct"] and r1["label_acts"] is None
    assert r1["alive_frac"] == 0.0
    s = summarize_results(res)
    assert s == {**s, "labelled": 1, "finished": 1, "extinct": 1,
                 "unprocessed": 0, "processed": 2, "samples": 2}
    # a budget of zero seconds after the first chunk stops the rest
    core2 = FakeCore(4, speed[:4], death[:4], goal_x=1e9)
    res2 = relabel_windows(core2, lambda f, m: StubDecider(4, k, f),
                           lambda: (None, None), field, line_scorer(field),
                           _bank_of(3), k, 4, H, 0, 0.25, 1,
                           budget_s=1e-9)
    assert res2[0] is not None and res2[1] is None and res2[2] is None
    assert summarize_results(res2)["unprocessed"] == 2


# --------------------------------------------------------------------------
# 3. weights, rows, the merged file
# --------------------------------------------------------------------------
def test_weights_follow_the_divergence_rule_and_the_share():
    dist = np.array([0.0, 128.0, 256.0, 1024.0, 1e6])
    w = divergence_weights(dist, elite_weight_sum=40000.0, share=0.25,
                           div_scale=256.0, div_cap=3.0)
    fac = np.array([1.0, 1.5, 2.0, 4.0, 4.0])
    assert np.all(np.diff(w) >= 0)                     # monotone in distance
    assert w.sum() == pytest.approx(40000.0 / 3.0)     # share / (1 - share)
    assert np.allclose(w / w[0], fac)                  # the factors, capped
    raw = divergence_weights(dist, 40000.0, 0.0, 256.0, 3.0)
    assert np.allclose(raw, fac)
    with pytest.raises(ValueError):
        divergence_weights(dist, 1.0, 1.0, 256.0, 3.0)
    assert len(divergence_weights([], 1.0, 0.25, 256.0, 3.0)) == 0
    # the divergence itself: nearest elite-line point
    ref = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], np.float64)
    pts = np.array([[0, 0, 0], [150, 0, 0], [100, 30, 40], [-300, 0, 0]])
    assert np.allclose(nearest_distance(pts, ref, chunk=2), [0, 50, 50, 300])
    assert np.isinf(nearest_distance(pts, np.zeros((0, 3)))).all()


def test_rows_and_merged_file_are_bc_format(tmp_path):
    import torch
    rng = np.random.default_rng(0)
    # an elite file of 5 rows, as plan_to_bc writes it
    e_st = np.zeros(5, STATE_DTYPE)
    e_st["origin"] = rng.normal(size=(5, 3))
    e_sc = rng.normal(size=(5, N_SCALAR)).astype(np.float32)
    elite = tmp_path / "bc.npz"
    save_bc_dataset(elite, e_st, e_sc, np.ones(5, np.float32),
                    np.zeros((5, 6), np.int64), np.full(5, 0.5, np.float32),
                    np.zeros(5, np.int32),
                    {"obs_reward": True, "n_latch": 1, "act_every": 3,
                     "lines": 1, "rows": 5, "best_s": 75.0})
    # two labelled samples (one with two label decisions), one unlabelled
    k, copies, H = 3, 4, 10
    field = LineField(goal_x=1e9)
    core = FakeCore(8, [10, 20, 50, 30, 10, 60, 20, 30])
    samples = _bank_of(3, x0=123.0)
    res = relabel_windows(core, lambda f, m: StubDecider(8, k, f),
                          lambda: (None, None), field, line_scorer(field),
                          samples, k, copies, H, 0, 0.25, 1,
                          label_decisions=2)
    res[2] = None                                     # budget-skipped
    w = np.array([1.5, 2.5, 3.5], np.float32)
    rows = rows_from_results(res, w, samples)
    assert rows is not None and len(rows["states"]) == 4
    assert rows["weights"].tolist() == [1.5, 1.5, 2.5, 2.5]
    assert rows["sample"].tolist() == [0, 0, 1, 1]
    assert rows["latch"].tolist() == [1.0] * 4
    assert rows["scal"][:, 12].tolist() == [0.25] * 4
    # the stub's decision-0 row differs from the stored sample row (0.7 vs
    # 0.25 in slot 12): the faithfulness counter sees both samples
    assert rows["row0_mismatch"] == 2
    assert rows["states"][0]["origin"][0] == 123.0
    assert rows_from_results([None, None], [1, 1]) is None
    out = tmp_path / "bc_dagger.npz"
    meta = merge_bc_datasets(elite, rows, out, {"k": 3}, n_latch=1,
                             obs_reward=True)
    assert meta["rows_elite"] == 5 and meta["rows_dagger"] == 4
    assert meta["rows"] == 9 and meta["dagger"] == {"k": 3}
    assert meta["weight_elite"] == 2.5 and meta["weight_dagger"] == 8.0
    assert meta["best_s"] == 75.0                     # the elite meta kept
    z = np.load(out, allow_pickle=False)
    assert z["line_id"].tolist() == [0] * 5 + [DAGGER_LINE_ID] * 4
    assert np.array_equal(z["states"][:5], e_st)
    assert np.array_equal(z["scal"][:5], e_sc)
    ds = BCDataset(out, torch.device("cpu"), n_latch=1, obs_reward=True)
    assert ds.n == 9
    assert ds.scal.shape == (9, N_SCALAR + 1) and ds.act.shape == (9, 6)
    assert float(ds.w[5]) == 1.5 and float(ds.scal[5, N_SCALAR]) == 1.0
    s, p, a, ww = ds.sample(4)
    assert s.shape == (4, N_SCALAR + 1) and p.shape == (4, 6)
    # a layout mismatch is refused, as BCDataset refuses it
    with pytest.raises(SystemExit):
        merge_bc_datasets(elite, rows, tmp_path / "x.npz", {}, n_latch=0,
                          obs_reward=True)
    # nothing labelled: the merged file is the elite file
    meta2 = merge_bc_datasets(elite, None, tmp_path / "y.npz", {}, 1, True)
    assert meta2["rows"] == 5 and meta2["rows_dagger"] == 0


# --------------------------------------------------------------------------
# 4. the driver hook: default off = the loop unchanged
# --------------------------------------------------------------------------
def test_default_off_leaves_the_round_unchanged():
    ap = argparse.ArgumentParser()
    expert_dagger.add_args(ap)
    a = ap.parse_args([])
    assert a.dagger_k == 0

    def run(*_a, **_k):
        raise AssertionError("the relabel phase ran with --dagger-k 0")

    def log(*_a):
        raise AssertionError("logged with --dagger-k 0")

    bc, bmeta = Path("bc.npz"), {"rows": 7}
    got = expert_dagger.maybe_relabel(a, Path("p.pt"), Path("round_0"),
                                      "m.bsp", bc, Path("spine.npy"), bmeta,
                                      None, run, log)
    assert got == (bc, bmeta) and got[0] is bc and got[1] is bmeta
    # an args object without the attribute at all (an older parser) is off
    got = expert_dagger.maybe_relabel(SimpleNamespace(), None, Path("r"),
                                      "m", bc, None, bmeta, None, run, log)
    assert got == (bc, bmeta)


def test_the_hook_is_the_only_change_to_the_loop():
    src = (ROOT / "tools" / "expert_loop.py").read_text(encoding="utf-8")
    assert "import expert_dagger" in src
    assert src.count("expert_dagger.add_args(") == 1
    assert src.count("expert_dagger.maybe_relabel(") == 1
    i_dist = src.index("= distil(")
    i_hook = src.index("expert_dagger.maybe_relabel(")
    i_train = src.index("= train(", i_dist)
    assert i_dist < i_hook < i_train, "the hook sits between distil and train"
    # the hook takes the distilled files and hands back what train() gets;
    # the driver's own run/log are what it logs through
    hook = src[src.rindex("\n", 0, i_hook) + 1:src.index(")", i_hook) + 1]
    assert hook.strip().startswith("bc, bmeta = expert_dagger.maybe_relabel(")
    for arg in ("args", "policy", "rdir", "map_path", "bc", "spine", "bmeta",
                "fh", "run", "log"):
        assert arg in hook.replace(" ", "").replace("\n", "").split("(", 1)[1]
    # the hook's flags come with the parser and default to off
    import expert_loop
    ap = expert_loop.build_parser()
    a = ap.parse_args(["seed.pt"])
    assert a.dagger_k == 0 and a.dagger_window == 3.0
    a = ap.parse_args(["seed.pt", "--dagger-k", "600"])
    assert a.dagger_k == 600


# --------------------------------------------------------------------------
# 4b. the physics tick: the checkpoint's, never a 100 Hz literal
# --------------------------------------------------------------------------
class TickCore(FakeCore):
    """A FakeCore that REPORTS a tick, the way SurfCore does (mean of the
    integer-ms pattern), and takes the three arming calls open_core makes."""

    def __init__(self, n, speed, tick_ms=None, **kw):
        super().__init__(n, speed, **kw)
        if tick_ms is not None:
            self.tick_ms = float(tick_ms)
        self.armed = {}

    def set_goal_box(self, mins, maxs):
        self.armed["goal"] = (tuple(mins), tuple(maxs))

    def set_teleport_fail(self, v):
        self.armed["teleport_fail"] = bool(v)

    def set_spawn_pool(self, pool):
        self.armed["pool"] = pool


def test_core_clock_and_check_core_tick_read_the_core_not_a_literal():
    # a core with no tick at all IS the reference: every conversion is the
    # legacy `* 100` arithmetic, which is what keeps 10 ms bit-identical
    ref = core_clock(FakeCore(2, [1.0, 1.0]))
    assert ref.is_reference and ref.ms == 10.0
    assert ref.secs_to_ticks(0.5, "round") == 50
    # the 7.63 ms core SurfCore builds for a --tick-ms checkpoint. The
    # checkpoint's clock says 7.63 (the flag); the CORE reports the pattern's
    # realised mean, 7.6667 - the two must agree on .ms, which is what every
    # conversion uses, not on .requested_ms
    T = TickClock(7.63)                       # load_bundle, from cfg
    C = core_clock(TickCore(2, [1.0, 1.0], tick_ms=23.0 / 3.0))
    assert C.pattern == T.pattern == [8, 8, 7]
    assert C.ms == pytest.approx(T.ms) and C.requested_ms != T.requested_ms
    assert T.ms == pytest.approx(7.666667, abs=1e-6)
    assert T.hz == pytest.approx(130.4348, abs=1e-3)
    # the four seconds flags of the phase, at 130.4 Hz rather than 100
    assert T.secs_to_ticks(0.5, "round") == 65        # --every
    assert T.secs_to_ticks(3.0, "round") == 391       # --window (via hz)
    assert T.secs_to_ticks(20.0, "round") == 2609     # --spine-secs
    assert T.secs_to_ticks(120.0) == 15652            # the episode cap
    # check_core_tick passes the matching core through and REFUSES a core
    # that quietly stayed at 10 ms (a build_sim that dropped tick=, or a
    # DLL with no surf_set_msec)
    ok = TickCore(2, [1.0, 1.0], tick_ms=23.0 / 3.0)
    assert check_core_tick(ok, T).ms == pytest.approx(T.ms)
    with pytest.raises(SystemExit) as e:
        check_core_tick(FakeCore(2, [1.0, 1.0]), T, "search core", "HINT.")
    msg = str(e.value)
    assert "search core" in msg and "7.63" in msg and "HINT." in msg
    # and the reverse: a patterned core where the reference was expected
    with pytest.raises(SystemExit):
        check_core_tick(ok, TickClock(10.0), "probe core")


def _sample_and_relabel_at(tick_ms, tmp_path, name):
    """The whole chain - sample, window, rows, merge - driven the way
    tools/expert_dagger.py drives it: every seconds flag converted at the
    CORE'S OWN clock. Returns the numbers a caller can compare across ticks.
    """
    k, copies, n = 3, 4, 8
    speed = [10, 20, 30, 40, 15, 25, 35, 45]
    core = TickCore(n, speed, tick_ms=tick_ms)
    T = core_clock(core)
    every_ticks = max(1, T.secs_to_ticks(0.5, "round"))       # --every 0.5
    bank = SampleBank()
    obs = core.reset(0)
    info = collect_rollout_samples(
        core, StubDecider(n, k), (None, None), obs, k,
        T.secs_to_ticks(2.0, "round"),                        # --rollout-secs
        every_ticks, SRC_GREEDY, np.arange(n), bank)
    A = bank.arrays()
    # --window 0.4 s, in DECISIONS, exactly as run_relabel computes H
    H = max(1, int(round(0.4 * T.hz / k)))
    samples = bank.select(np.arange(0, len(bank), max(1, len(bank) // 4)))
    core2 = TickCore(n, speed, tick_ms=tick_ms)
    field = LineField(goal_x=1e9)
    res = relabel_windows(core2, lambda f, m: StubDecider(n, k, f),
                          lambda: (None, None), field, line_scorer(field),
                          samples, k, copies, H, 0, 0.25, 1,
                          label_decisions=1)
    rows = rows_from_results(res, np.ones(len(samples), np.float32), samples)
    # merge onto a 3-row elite file, as the phase does
    elite = tmp_path / f"{name}_bc.npz"
    e_st = np.zeros(3, STATE_DTYPE)
    save_bc_dataset(elite, e_st, np.zeros((3, N_SCALAR), np.float32),
                    np.ones(3, np.float32), np.zeros((3, 6), np.int64),
                    np.full(3, 0.5, np.float32), np.zeros(3, np.int32),
                    {"obs_reward": True, "n_latch": 1, "act_every": k})
    out = tmp_path / f"{name}_merged.npz"
    meta = merge_bc_datasets(elite, rows, out, {"tick_ms": T.ms}, 1, True)
    return {"clock": T, "every_ticks": every_ticks,
            "n_ticks": T.secs_to_ticks(2.0, "round"), "H": H,
            "window_ticks": H * k, "recorded": info["recorded"],
            "ticks": sorted(set(A["tick"].tolist())),
            "rows": 0 if rows is None else len(rows["states"]),
            "merged": out, "meta": meta}


def test_a_synthetic_7_63_ms_relabel_round_trips_on_the_130_hz_grid(tmp_path):
    import torch
    a = _sample_and_relabel_at(23.0 / 3.0, tmp_path, "t763")   # [8, 8, 7]
    b = _sample_and_relabel_at(None, tmp_path, "t10")          # the reference

    # 10 ms: the legacy numbers, unchanged - 0.5 s = 50 ticks, 2.0 s = 200,
    # a 0.4 s window = 13 decisions, and the same grid the sampling test pins
    assert b["clock"].is_reference
    assert (b["every_ticks"], b["n_ticks"], b["H"]) == (50, 200, 13)
    assert b["ticks"] == [0, 51, 102, 150]

    # 7.63 ms: 1.304x the ticks per second, so every conversion moves and
    # NONE of them is the 100 Hz answer
    assert a["clock"].pattern == [8, 8, 7]
    assert (a["every_ticks"], a["n_ticks"], a["H"]) == (65, 261, 17)
    assert a["ticks"] == [0, 66, 132, 195]
    assert a["window_ticks"] == 51 and b["window_ticks"] == 39
    # the same wall-clock request, so the same number of sampled states and
    # the same number of relabelled rows - only the tick counts differ
    assert a["recorded"] == b["recorded"] == 32
    assert a["rows"] == b["rows"] == 4

    # round trip: the merged file loads and carries the P2 target
    for r in (a, b):
        ds = BCDataset(r["merged"], torch.device("cpu"), n_latch=1,
                       obs_reward=True)
        assert ds.n == r["meta"]["rows"] == 3 + r["rows"]
        z = np.load(r["merged"], allow_pickle=False)
        assert (z["line_id"] == DAGGER_LINE_ID).sum() == r["rows"]
        assert "probs" in z.files
        p = np.asarray(z["probs"], np.float64)
        assert p.shape[1:] == (6, 15)
        assert np.allclose(p.sum(-1), 1.0, atol=1e-5)


def test_open_core_and_make_feeds_carry_the_bundles_tick(monkeypatch):
    """The two wiring points inside the tool: every core is built through
    build_sim(..., tick=) and every feed through make_eval_feeds(tick_ms=).
    Spied rather than read out of the source, so a rename cannot pass."""
    import beam_tas
    T = TickClock(7.63)
    seen = {}

    def spy_build(cfg, map_path, num_envs, ep_cap, tick=None):
        seen["tick"] = tick
        seen["envs"] = int(num_envs)
        return TickCore(int(num_envs), np.zeros(int(num_envs)),
                        tick_ms=(None if tick is None else tick.ms))

    monkeypatch.setattr(beam_tas, "build_sim", spy_build)
    B = {"cfg": {"reward": "race"}, "map_path": "x.bsp", "ep_cap": 1000,
         "tick": T, "pool": {"origin": np.zeros((1, 3), np.float32)},
         "zones": {"end": {"mins": [0, 0, 0], "maxs": [1, 1, 1]}}}
    core = expert_dagger.open_core(B, 6)
    assert seen["tick"] is T and seen["envs"] == 6
    assert core.armed["teleport_fail"] is True and "goal" in core.armed
    # a build_sim that DROPS the tick is caught by check_core_tick, not by
    # a silently mis-timed relabel
    monkeypatch.setattr(beam_tas, "build_sim",
                        lambda cfg, m, n, c, tick=None: FakeCore(n, np.zeros(n)))
    with pytest.raises(SystemExit) as e:
        expert_dagger.open_core(B, 6)
    assert "6-env core" in str(e.value)

    # the --obs-reward mirror: the tick the FLAG asked for (7.63), which is
    # what beam_tas / plan_to_bc / record_ckpt all pass
    got = {}

    def spy_feeds(cfg, field, d0, k, tick_ms=10.0):
        got["tick_ms"] = tick_ms
        got["k"] = k
        return -1, None, None

    monkeypatch.setattr(expert_dagger, "make_eval_feeds", spy_feeds)
    expert_dagger.make_feeds({"cfg": {}, "gf": None, "d0": 1.0, "K": 4,
                              "tick": T})
    assert got["tick_ms"] == pytest.approx(7.63) and got["k"] == 4


needs_map = pytest.mark.skipif(not MAIN_MAP.exists(),
                               reason="needs the cannonball map")


@needs_map
def test_build_sim_gives_the_core_expert_dagger_checks_for():
    """The real C core at 7.63 ms, through the same call open_core makes:
    the [8, 8, 7] pattern, the pitch rate scaled per SECOND, and the yaw
    ceiling NOT scaled under --yaw-adaptive (CLAUDE.md / ecc0506)."""
    import beam_tas
    T = TickClock(7.63)
    cfg = {"maxvel": 4000.0, "yaw_adaptive": True, "pitch_rate": 1.33}
    core = beam_tas.build_sim(cfg, str(MAIN_MAP), 2, 1000, tick=T)
    assert tuple(core.tick_pattern) == (8, 8, 7)
    assert check_core_tick(core, T).ms == pytest.approx(T.ms)
    # adaptive yaw bins are the per-FRAME strafe optimum: the ceiling is a
    # clamp and the divisor of obs column 10, so it keeps the reference 10
    assert core.config.yaw_rate_max_deg == pytest.approx(10.0)
    assert core.config.pitch_rate_max_deg == pytest.approx(1.33 * T.scale)
    # fixed bins DO scale, so the deg per second is unchanged
    fixed = beam_tas.build_sim({**cfg, "yaw_adaptive": False}, str(MAIN_MAP),
                               2, 1000, tick=T)
    assert fixed.config.yaw_rate_max_deg == pytest.approx(10.0 * T.scale)
    # and the reference tick builds exactly the core it always built
    ten = beam_tas.build_sim(cfg, str(MAIN_MAP), 2, 1000, tick=TickClock(10.0))
    assert tuple(ten.tick_pattern) == (10,)
    assert ten.config.yaw_rate_max_deg == pytest.approx(10.0)
    assert ten.config.pitch_rate_max_deg == pytest.approx(1.33)
    assert check_core_tick(ten, TickClock(10.0)).is_reference


# --------------------------------------------------------------------------
# 5. CPU dry run on the real core, policy and map (2 states, 16 envs)
# --------------------------------------------------------------------------
_caches = [MAIN_MAP.with_name("surf_src_cannonball.goal_32.npz"),
           MAIN_MAP.with_name("surf_src_cannonball.sdf_32.npz")]
needs_exit = pytest.mark.skipif(
    not (MAIN_MAP.exists() and all(c.exists() for c in _caches)
         and EXIT_CKPT.exists() and EXIT_BC.exists() and EXIT_SPINE.exists()),
    reason="needs the cannonball map + caches and the exit10 round-23 files")


@needs_exit
def test_cpu_dry_run_relabels_real_states_faithfully(tmp_path):
    import torch
    out = tmp_path / "bc_dagger.npz"
    args = expert_dagger.build_parser().parse_args([
        str(EXIT_CKPT), "--bc", str(EXIT_BC), "--spine", str(EXIT_SPINE),
        "--out", str(out), "--map", str(MAIN_MAP), "--device", "cpu",
        "--k", "2", "--every", "0.3", "--episodes", "1",
        "--stoch-episodes", "1", "--rollout-secs", "0.7",
        "--spine-spawns", "1", "--spine-secs", "0.4",
        "--window", "0.3", "--envs", "16", "--copies", "16",
        "--greedy-envs", "2", "--resample", "5", "--score", "dv",
        "--seed", "3"])
    s = expert_dagger.run_relabel(args)
    assert s["k"] == 2 and s["labelled"] == 2 and s["extinct"] == 0
    assert s["rows_dagger"] == 2
    assert s["rows_total"] == s["rows_elite"] + 2
    # the restart is faithful: the window's decision-0 row (slot-12
    # mirror, latch) is the row the policy saw in its own rollout
    assert s["row0_mismatch"] == 0
    assert s["weights"]["sum"] == pytest.approx(s["weight_elite"] / 3.0)
    # a 10 ms checkpoint: the reference tick, every conversion legacy
    # (0.3 s = 30 ticks, a 0.3 s window = 10 decisions at act_every 3)
    cf = s["config"]
    assert cf["tick_ms"] == 10.0 and cf["tick_pattern_ms"] == [10]
    assert cf["every_ticks"] == 30 and cf["window_decisions"] == 10
    # P2: the population target is reported, and what the summary says
    # about the file is what the file IS. A 16-copy window on a policy this
    # peaked can collapse to the winner's one-hot at every head, which is
    # the documented degenerate case: the merged file then stays version 1
    # and --bc-target dist IS --bc-target argmax (the trainer prints so).
    assert 0 <= s["rows_nonhot"] <= s["rows_dagger"]
    assert 0.0 < s["target_top1_mean"] <= 1.0
    assert s["target_copies_mean"] == pytest.approx(16.0)
    ds = BCDataset(out, torch.device("cpu"), n_latch=1, obs_reward=True)
    assert ds.n == s["rows_total"] and ds.has_probs == s["bc_has_probs"]
    z = np.load(out, allow_pickle=False)
    assert (z["line_id"] == DAGGER_LINE_ID).sum() == 2
    if s["bc_has_probs"]:
        pr = np.asarray(z["probs"], np.float64)[z["line_id"] == DAGGER_LINE_ID]
        assert np.allclose(pr.sum(-1), 1.0, atol=1e-5)
        assert int((pr.max(-1).min(-1) < 0.99).sum()) == s["rows_nonhot"]
    else:
        assert s["rows_nonhot"] == 0 and s["target_top1_mean"] == 1.0
    assert (tmp_path / "bc_dagger_rows.npz").exists()
    assert (tmp_path / "bc_dagger_samples.npz").exists()
    assert (tmp_path / "bc_dagger_summary.json").exists()
