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
     hook is the only edit to expert_loop.py.

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
                            SRC_STOCH, SampleBank, collect_rollout_samples,
                            decision_grid, divergence_weights, even_subset,
                            merge_bc_datasets, nearest_distance, on_grid,
                            relabel_windows, rows_from_results,
                            summarize_results)
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
    ds = BCDataset(out, torch.device("cpu"), n_latch=1, obs_reward=True)
    assert ds.n == s["rows_total"]
    z = np.load(out, allow_pickle=False)
    assert (z["line_id"] == DAGGER_LINE_ID).sum() == 2
    assert (tmp_path / "bc_dagger_rows.npz").exists()
    assert (tmp_path / "bc_dagger_samples.npz").exists()
    assert (tmp_path / "bc_dagger_summary.json").exists()
