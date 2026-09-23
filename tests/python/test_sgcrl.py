"""Tests for python/train_sgcrl.py (Single-Goal Contrastive RL).

CPU only (no test touches a GPU). The end-to-end smoke and the collector
tests need the built core and the labyrinth map; they skip without them:
    SURFCORE_DLL   the core (default <repo>/build/surfcore.dll | libsurfcore.so)
    SGCRL_TEST_MAP labyrinth_left100.bsp (default: <repo>/maps_pool/, <repo>/maps/,
                   then C:/RL_Surf/maps_pool/ - maps_pool is untracked)
"""
from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "python"))

import train_sgcrl as ts  # noqa: E402

DLL = (Path(os.environ["SURFCORE_DLL"]) if os.environ.get("SURFCORE_DLL")
       else ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))


def _find_map() -> Path:
    if os.environ.get("SGCRL_TEST_MAP"):
        return Path(os.environ["SGCRL_TEST_MAP"])
    for c in (ROOT / "maps_pool" / "labyrinth_left100.bsp",
              ROOT / "maps" / "labyrinth_left100.bsp",
              Path("C:/RL_Surf/maps_pool/labyrinth_left100.bsp")):
        if c.exists():
            return c
    return ROOT / "maps_pool" / "labyrinth_left100.bsp"


MAP = _find_map()
needs_core = pytest.mark.skipif(
    not (DLL.exists() and MAP.exists()
         and MAP.with_name(MAP.stem + ".zones.json").exists()),
    reason="needs the built core (SURFCORE_DLL) and labyrinth_left100 (SGCRL_TEST_MAP)")


@pytest.fixture(autouse=True)
def _dll_env(monkeypatch):
    if DLL.exists():
        monkeypatch.setenv("SURFCORE_DLL", str(DLL))


# ==========================================================================
# action mapping
# ==========================================================================

def _grid_actions(duck: bool) -> np.ndarray:
    """One continuous action per discrete cell: bin centres of every head."""
    yaw = [-1.0 + (2 * k + 1) / ts.YAW_N for k in range(ts.YAW_N)]
    tri = [-0.8, 0.0, 0.8]
    jump = [-0.5, 0.5]
    heads = [yaw, tri, tri, jump] + ([jump] if duck else [])
    return np.asarray(list(itertools.product(*heads)), np.float32)


@pytest.mark.parametrize("duck", [False, True])
def test_every_discrete_combination_is_reachable(duck):
    a = _grid_actions(duck)
    d = ts.to_discrete(a, duck)
    assert d.dtype == np.int32 and d.shape == (len(a), 6)
    assert np.all(d[:, 1] == ts.PITCH_LEVEL)             # pitch: always level
    got = {tuple(r) for r in d[:, [0, 2, 3, 4, 5]].tolist()}
    want = set(itertools.product(range(15), range(3), range(3), range(2),
                                 range(2) if duck else [0]))
    assert got == want                                    # every combination, bijectively
    assert len(got) == len(a)


def test_mapping_bins_and_thresholds():
    third = 1.0 / 3.0
    a = np.zeros((9, 4), np.float32)
    a[:, 0] = [-1.0, 1.0, 0.0, -1.0 + 2.0 / 15 + 1e-4, -1.0 + 2.0 / 15 - 1e-4,
               0.999999, -0.999999, 1.0 / 15 - 1e-4, 1.0 / 15 + 1e-4]
    d = ts.to_discrete(a)
    assert d[:, 0].tolist() == [0, 14, 7, 1, 0, 14, 0, 7, 8]
    b = np.zeros((6, 4), np.float32)
    b[:, 1] = [third, -third, third + 1e-3, -third - 1e-3, 1.0, -1.0]
    b[:, 2] = [-1.0, 1.0, 0.0, 0.2, -0.34, 0.34]
    b[:, 3] = [0.0, 1e-3, -1e-3, 1.0, -1.0, 0.5]
    d = ts.to_discrete(b)
    assert d[:, 2].tolist() == [1, 1, 2, 0, 2, 0]       # forward: 0 back / 1 none / 2 fwd
    assert d[:, 3].tolist() == [0, 2, 1, 1, 0, 2]       # side: 0 A (left) / 2 D (right)
    assert d[:, 4].tolist() == [0, 1, 0, 1, 0, 1]       # jump: a3 > 0
    assert np.all(d[:, 5] == 0)                         # no duck unless --duck 1


def test_mapping_rejects_wrong_width():
    with pytest.raises(ValueError):
        ts.to_discrete(np.zeros((3, 5), np.float32), duck=False)
    with pytest.raises(ValueError):
        ts.to_discrete(np.zeros((3, 4), np.float32), duck=True)
    with pytest.raises(ValueError):
        ts.ActionMap("world")(np.zeros((3, 4), np.float32))


def test_rate_action_map_is_to_discrete_without_a_view():
    rng = np.random.default_rng(0)
    for duck in (False, True):
        am = ts.ActionMap("rate", duck)
        a = rng.uniform(-1, 1, (500, am.A)).astype(np.float32)
        acts, view = am(a)
        assert view is None and am.view_mode == 0
        assert np.array_equal(acts, ts.to_discrete(a, duck))


@pytest.mark.parametrize("duck", [False, True])
def test_world_yaw_is_a_heading_target(duck):
    am = ts.ActionMap("world", duck)
    assert am.A == (6 if duck else 5) and am.view_mode == 2
    deg = np.array([0.0, 45.0, 90.0, 135.0, 179.0, -179.0, -135.0, -90.0, -45.0])
    th = np.radians(deg)
    a = np.zeros((len(deg), am.A), np.float32)
    a[:, 0] = 0.7 * np.cos(th)                  # only the direction matters
    a[:, 1] = 0.7 * np.sin(th)
    a[:, 2] = 0.9
    acts, view = am(a)
    assert view.shape == (len(deg), 2) and view.dtype == np.float32
    err = (view[:, 0] - deg + 180.0) % 360.0 - 180.0
    assert np.abs(err).max() < 1e-3             # continuous across the +-180 seam
    assert (view[:, 1] == 0.0).all()            # pitch target: level
    assert (acts[:, 0] == ts.NEUTRAL[0]).all() and (acts[:, 1] == ts.PITCH_LEVEL).all()
    assert (acts[:, 2] == 2).all()
    # every key combination is reachable from the key columns
    tri, bit = [-0.8, 0.0, 0.8], [-0.5, 0.5]
    keys = np.asarray(list(itertools.product(tri, tri, bit, *([bit] if duck else []))), np.float32)
    b = np.concatenate([np.ones((len(keys), 1), np.float32), np.zeros((len(keys), 1), np.float32),
                        keys], 1)
    acts, _ = am(b)
    got = {tuple(r) for r in acts[:, 2:].tolist()}
    assert got == set(itertools.product(range(3), range(3), range(2), range(2) if duck else [0]))


# ==========================================================================
# future-offset sampler
# ==========================================================================

def _pmf_renorm(L, g):
    k = np.arange(1, L + 1)
    p = g ** (k - 1.0)
    return p / p.sum()


def test_future_offsets_truncated_geometric_renormalised():
    torch.manual_seed(0)
    L, g, n = 20, 0.8, 400_000
    k = ts.future_offsets(torch.full((n,), L), torch.rand(n), g, "renorm").numpy()
    assert k.min() >= 1 and k.max() <= L
    emp = np.bincount(k, minlength=L + 1)[1:] / n
    assert np.abs(emp - _pmf_renorm(L, g)).max() < 0.004


def test_future_offsets_clip_puts_the_tail_on_the_last_state():
    torch.manual_seed(1)
    L, g, n = 5, 0.9, 400_000
    k = ts.future_offsets(torch.full((n,), L), torch.rand(n), g, "clip").numpy()
    emp = np.bincount(k, minlength=L + 1)[1:] / n
    want = np.array([(1 - g) * g ** (j - 1) for j in range(1, L)] + [g ** (L - 1)])
    assert np.abs(emp - want).max() < 0.004


def test_future_offsets_per_sample_horizons():
    torch.manual_seed(2)
    n = 200_000
    L = torch.randint(1, 60, (n,))
    k = ts.future_offsets(L, torch.rand(n), 0.99, "renorm")
    assert bool((k >= 1).all()) and bool((k <= L).all())
    assert bool((k[L == 1] == 1).all())                   # one future step: always it
    # gamma -> 1 on a short horizon is ~uniform on 1..L
    k2 = ts.future_offsets(torch.full((n,), 4), torch.rand(n), 0.999999, "renorm").numpy()
    assert np.abs(np.bincount(k2, minlength=5)[1:] / n - 0.25).max() < 0.01


# ==========================================================================
# replay buffer: episodes, futures, wrap-around
# ==========================================================================

class _EpisodeWriter:
    """Writes synthetic episodes into a CPU Replay. Row features encode
    (episode uid, row index within the episode, env); a finished episode's
    terminal goal is (uid, episode length, env) - one past its last row."""

    def __init__(self, rb, rng, len_lo=1, len_hi=8):
        self.rb, self.rng = rb, rng
        self.lo, self.hi = len_lo, len_hi
        self.N = rb.N
        self.uid = np.arange(self.N, dtype=np.float64)
        self.next_uid = float(self.N)
        self.row = np.zeros(self.N, np.int64)
        self.len = rng.integers(len_lo, len_hi + 1, self.N)
        self.ep_of_row = {}                   # (abs, env) -> (uid, row)

    def step(self):
        rb, N = self.rb, self.N
        obs = np.zeros((N, ts.F_DIM), np.float32)
        obs[:, 0] = self.uid
        obs[:, 1] = self.row
        obs[:, 2] = np.arange(N)
        for e in range(N):
            self.ep_of_row[(rb.head, e)] = (self.uid[e], self.row[e])
        rb.write_step(torch.as_tensor(obs), torch.zeros(N, rb.act.shape[-1]))
        self.row += 1
        done = np.flatnonzero(self.row >= self.len)
        if done.size:
            term = np.stack([self.uid[done], self.len[done].astype(np.float64),
                             done.astype(np.float64)], 1).astype(np.float32)
            rb.close_episodes(done, term)
            for e in done:
                self.uid[e] = self.next_uid
                self.next_uid += 1.0
            self.row[done] = 0
            self.len[done] = self.rng.integers(self.lo, self.hi + 1, done.size)


def _check_samples(rb, n=4000, gamma=0.7, mode="renorm"):
    obs, act, goal, dbg = rb.sample(n, gamma, mode, debug=True)
    obs, goal = obs.numpy(), goal.numpy()
    a_abs, f_abs = dbg["a_abs"].numpy(), dbg["f_abs"].numpy()
    is_term, env = dbg["is_term"].numpy(), dbg["env"].numpy()
    head = rb.head
    assert (a_abs <= head - 2).all(), "the newest row is never an anchor"
    assert (a_abs >= max(0, head - rb.T)).all(), "anchors come from unoverwritten rows"
    assert (obs[:, 2] == env).all()
    assert (goal[:, 2] == env).all(), "a future goal never comes from another env"
    assert (goal[:, 0] == obs[:, 0]).all(), "a future goal never crosses an episode boundary"
    assert (goal[:, 1] > obs[:, 1]).all(), "positives are strictly in the future"
    nt = ~is_term
    assert np.array_equal(goal[nt, 1] - obs[nt, 1], (f_abs - a_abs)[nt])
    assert (f_abs[nt] <= head - 1).all()
    return obs, goal, dbg


def test_replay_futures_stay_in_episode_through_wraparound():
    rng = np.random.default_rng(0)
    rb = ts.Replay(n_envs=3, t_cap=10, a_dim=4, device="cpu")   # episodes <= 8 rows < T
    w = _EpisodeWriter(rb, rng, 1, 8)
    torch.manual_seed(0)
    seen_term = 0
    for t in range(120):                       # wraps the ring 12 times
        w.step()
        if t >= 2:
            _obs, goal, dbg = _check_samples(rb)
            seen_term += int(dbg["is_term"].sum())
    assert rb.head == 120
    assert seen_term > 0, "terminal goals are sampled"


def test_replay_terminal_goal_is_the_episode_end():
    rb = ts.Replay(n_envs=1, t_cap=16, a_dim=4, device="cpu")
    rng = np.random.default_rng(1)
    w = _EpisodeWriter(rb, rng, 5, 5)          # every episode exactly 5 rows
    for _ in range(12):
        w.step()
    torch.manual_seed(3)
    obs, goal, dbg = _check_samples(rb, n=20000, gamma=0.9)
    it = dbg["is_term"].numpy()
    assert (goal[it, 1] == 5).all(), "terminal goal = one past the last row"
    # the last row of a finished episode has exactly one future: the terminal
    last = (obs[:, 1] == 4) & (obs[:, 0] < 2)          # episodes 0, 1 are finished
    assert last.any() and it[last].all()
    # an anchor at row r of a finished 5-row episode has L = 5 - r futures, and
    # P(terminal) = gamma^(L-1) / sum_k gamma^(k-1) (renormalised)
    r0 = (obs[:, 1] == 0) & (obs[:, 0] < 2)
    want = _pmf_renorm(5, 0.9)[-1]
    assert abs(it[r0].mean() - want) < 0.03


def test_replay_running_episode_uses_rows_written_so_far():
    rb = ts.Replay(n_envs=1, t_cap=64, a_dim=4, device="cpu")
    rng = np.random.default_rng(2)
    w = _EpisodeWriter(rb, rng, 50, 50)        # one long episode, still running
    for _ in range(20):
        w.step()
    torch.manual_seed(4)
    obs, goal, dbg = _check_samples(rb, n=20000, gamma=0.8)
    assert not dbg["is_term"].any()
    assert (goal[:, 1] <= 19).all()
    L = dbg["L"].numpy()
    assert np.array_equal(L, 19 - obs[:, 1].astype(np.int64))
    # offsets follow the truncated geometric for each anchor's own horizon
    k = dbg["k"].numpy()
    sel = obs[:, 1] == 0                       # L = 19
    emp = np.bincount(k[sel], minlength=20)[1:] / sel.sum()
    assert np.abs(emp - _pmf_renorm(19, 0.8)).max() < 0.03


def test_replay_close_rejects_broken_bookkeeping():
    rb = ts.Replay(n_envs=2, t_cap=4, a_dim=4, device="cpu")
    for _ in range(3):
        rb.write_step(torch.zeros(2, ts.F_DIM), torch.zeros(2, 4))
    rb.ep_start[0] = 10                        # starts after its own last row
    with pytest.raises(RuntimeError):
        rb.close_episodes(np.array([0]), np.zeros((1, 3), np.float32))


# ==========================================================================
# losses and the learner
# ==========================================================================

def _args(**kw):
    a = dict(batch=64, gamma=0.9, future="renorm", random_goals=0.5, lse_coef=0.01,
             hidden=64, repr_dim=16, min_std=1e-6, alpha="auto", target_entropy=None,
             lr=3e-4, cuda_graph=0)
    a.update(kw)
    return SimpleNamespace(**a)


def test_critic_loss_decreases_on_a_toy_batch():
    torch.manual_seed(0)
    B = 64
    obs = torch.randn(B, ts.F_DIM)
    act = torch.rand(B, 4) * 2 - 1
    goal = torch.tanh(obs[:, :3] + 0.5 * act[:, :3])     # each anchor has its own future
    critic = ts.Critic(4, hidden=64, repr_dim=16)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    first = None
    for _ in range(300):
        loss, logits, _ = ts.critic_loss_fn(critic, obs, act, goal, 0.01)
        if first is None:
            first = float(loss.detach())
            acc0 = float((logits.argmax(1) == torch.arange(B)).float().mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss, logits, lse = ts.critic_loss_fn(critic, obs, act, goal, 0.01)
    acc = float((logits.argmax(1) == torch.arange(B)).float().mean())
    assert float(loss.detach()) < 0.5 * first
    assert acc > 0.9 and acc > acc0
    assert abs(float(lse.detach().mean())) < 3.0      # the logsumexp regulariser holds


def test_tanh_gauss_logp_matches_a_density_integral():
    # 1-D check: the density of a = tanh(u), u ~ N(0.3, 0.7), integrates to 1
    loc, std = 0.3, 0.7
    a = torch.linspace(-0.99999, 0.99999, 200001, dtype=torch.float64)
    u = torch.atanh(a)
    eps = (u - loc) / std
    lp = ts.tanh_gauss_logp(eps[:, None], u[:, None], torch.full_like(u[:, None], std))
    integral = float(torch.trapz(lp.exp(), a))
    assert abs(integral - 1.0) < 1e-3


def _filled_replay(N=8, T=64, steps=60, seed=0):
    rng = np.random.default_rng(seed)
    rb = ts.Replay(N, T, 4, "cpu")
    since = np.zeros(N, np.int64)
    ep_len = rng.integers(5, 30, N)
    for _ in range(steps):
        rb.write_step(torch.as_tensor(rng.normal(size=(N, ts.F_DIM)).astype(np.float32)),
                      torch.as_tensor(rng.uniform(-1, 1, (N, 4)).astype(np.float32)))
        since += 1
        done = np.flatnonzero(since >= ep_len)
        if done.size:
            rb.close_episodes(done, rng.normal(size=(done.size, 3)).astype(np.float32))
            since[done] = 0
            ep_len[done] = rng.integers(5, 30, done.size)
    return rb


@pytest.mark.parametrize("alpha", ["auto", "0"])
def test_learner_updates_on_cpu(alpha):
    torch.manual_seed(0)
    rb = _filled_replay()
    L = ts.Learner(rb, 4, _args(alpha=alpha), "cpu")
    before = [p.detach().clone() for p in L.actor.parameters()]
    cb = [p.detach().clone() for p in L.critic.parameters()]
    for _ in range(25):
        L.update()
    st = L.read_stats()
    assert L.n_updates == 25
    assert set(st) == set(ts.STAT_NAMES)
    assert all(np.isfinite(v) for v in st.values())
    assert any((a - b).abs().max() > 0 for a, b in zip(before, L.actor.parameters()))
    assert any((a - b).abs().max() > 0 for a, b in zip(cb, L.critic.parameters()))
    if alpha == "auto":
        assert L.target_entropy == -4.0
        assert float(L.log_alpha.detach()) != 0.0     # the temperature adapts
    else:
        assert st["alpha"] == 0.0 and L.log_alpha is None
    assert L.read_stats() == {}                       # read resets the accumulators


def test_actor_loss_does_not_touch_the_critic_gradients():
    torch.manual_seed(0)
    rb = _filled_replay()
    L = ts.Learner(rb, 4, _args(), "cpu")
    obs, act, goal = rb.sample(32, 0.9)
    loc, std = L.actor(torch.cat([obs, goal], 1))
    a_pi = torch.tanh(loc)
    q = (L.critic.phi(torch.cat([obs, a_pi], 1)) * L.critic.psi(goal)).sum(1)
    (-q.mean()).backward(inputs=L.actor_params)
    assert all(p.grad is None for p in L.critic.parameters())
    assert all(p.grad is not None for p in L.actor.parameters())


def test_act_is_in_range_and_greedy_is_deterministic():
    torch.manual_seed(0)
    rb = _filled_replay()
    L = ts.Learner(rb, 4, _args(), "cpu")
    obs = torch.randn(50, ts.F_DIM)
    g = torch.zeros(1, 3)
    a = L.act(obs, g)
    assert a.shape == (50, 4) and float(a.abs().max()) <= 1.0
    assert torch.equal(L.act(obs, g, greedy=True), L.act(obs, g, greedy=True))


# ==========================================================================
# the collector against the real core
# ==========================================================================

@needs_core
def test_collector_terminal_goals_from_the_core():
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones
    zone = load_zones(str(MAP), create=False)["end"]
    core = ts.make_core(str(MAP), 3, 3000)
    pool = map_spawn_pool(core)
    ts.arm_core(core, pool, zone)
    mn, mx = core.map_bounds()
    norm = ts.Normaliser(mn, mx)
    col = ts.Collector(core, norm, 4, False, None, zone, 64.0, mn, mx)
    core.reset(0)
    # env 0: next to the finish, walking into it; env 1: forced fail; env 2: idle
    st = core.get_states()
    s0 = st[0:1].copy()
    gx = 0.5 * (zone["mins"][0] + zone["maxs"][0])
    s0["origin"] = [[gx, zone["mins"][1] - 18.0, 40.0]]   # hull edge is 16 u out
    s0["velocity"] = [[0.0, 250.0, 0.0]]
    s0["yaw"] = 90.0
    core.set_state(0, s0)
    pre1 = np.asarray(core.states_view["origin"][1], np.float64).copy()
    col.observe()
    core.force_fail(np.array([0, 1, 0], np.uint8))  # consumed on the next tick
    a = np.zeros((3, 4), np.float32)
    a[0, 1] = 1.0                                     # env 0: forward
    col.step(ts.to_discrete(a))
    assert col.pending is not None
    envs, term = col.pending
    assert sorted(envs.tolist()) == [0, 1]
    tw = {int(e): norm.unpos(t) for e, t in zip(envs, term)}
    # goal hit: the terminal obs is the post-move origin at the box's hull edge
    assert ts.box_distance(tw[0][None], np.asarray(zone["mins"]) - [16, 16, 36],
                           np.asarray(zone["maxs"]) + [16, 16, 36])[0] < 1.0
    # fail: the pre-tick state
    assert np.allclose(tw[1], pre1, atol=0.05)
    assert col.goals == 1 and col.episodes == 2
    iv = col.interval()
    assert iv["train_success"] == 0.5 and iv["train_fail_frac"] == 0.5
    # the buffer takes the stash: rows of both episodes get their ends
    rb = ts.Replay(3, 800, 4, "cpu")
    rb.write_step(torch.as_tensor(col.feat.copy()), torch.zeros(3, 4))
    col.flush(rb)
    assert rb.ep_end[0, 0] == 0 and rb.ep_end[0, 1] == 0 and rb.ep_end[0, 2] == -1
    assert rb.ep_start.tolist() == [1, 1, 0]
    core.close()


@needs_core
@pytest.mark.parametrize("stem,dead_ys", [("labyrinth_left100", []),
                                          ("surf_edgeflow_blue050", [-1248.0] * 4)])
def test_live_spawns_drops_only_dead_on_arrival_points(stem, dead_ys):
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones
    bsp = MAP.with_name(stem + ".bsp")
    if not bsp.exists() or not bsp.with_name(stem + ".zones.json").exists():
        pytest.skip(f"{stem} not available")
    zone = load_zones(str(bsp), create=False)["end"]
    probe = ts.make_core(str(bsp), 1, 3000)
    pool = map_spawn_pool(probe)
    probe.close()
    for view_mode in (0, 2):
        live = ts.live_spawns(str(bsp), pool, zone, 3000, view_mode)
        assert live.dtype == bool and live.shape == (len(pool),)
        assert sorted(pool["origin"][~live][:, 1].tolist()) == dead_ys


@needs_core
def test_world_yaw_turns_toward_the_target_on_the_core():
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones
    zone = load_zones(str(MAP), create=False)["end"]
    am = ts.ActionMap("world")
    core = ts.make_core(str(MAP), 3, 3000, am.view_mode)
    ts.arm_core(core, map_spawn_pool(core), zone)
    mn, mx = core.map_bounds()
    col = ts.Collector(core, ts.Normaliser(mn, mx), 4, False, None, zone, 64.0, mn, mx)
    core.reset(0)
    st = core.get_states()
    for e in range(3):
        s = st[e:e + 1].copy()
        s["yaw"] = 90.0
        core.set_state(e, s)
    col.observe()
    a = np.zeros((3, 5), np.float32)
    a[0, :2] = [-1.0, 0.0]                        # heading 180: a 90 deg turn
    a[1, :2] = [0.0, 1.0]                         # heading 90: already there
    a[2, :2] = [0.0, -1.0]                        # heading -90: 180 deg away
    acts, view = am(a)
    view[1, 0] = np.nan                           # NaN = no turn (the core's contract)
    col.step(acts, view)
    yaw = np.asarray(core.states_view["yaw"], np.float64)
    assert abs(yaw[0] - 130.0) < 1e-3             # 4 ticks x 10 deg toward 180
    assert abs(yaw[1] - 90.0) < 1e-3
    assert abs((yaw[2] - 90.0 + 180.0) % 360.0 - 180.0) == pytest.approx(40.0, abs=1e-3)
    for _ in range(3):                            # 12 more ticks: reached and held
        col.observe()
        col.step(*am(a))
    yaw = np.asarray(core.states_view["yaw"], np.float64)
    assert abs(yaw[0] - 180.0) < 1e-3 and abs(yaw[1] - 90.0) < 1e-3
    core.close()


# ==========================================================================
# end to end (CPU)
# ==========================================================================

def _run_args(tmp_path, run, extra=()):
    return ts.build_parser().parse_args([
        "--map", str(MAP), "--run", run, "--runs-dir", str(tmp_path),
        "--envs", "8", "--steps", "6000", "--batch", "32", "--hidden", "64",
        "--repr-dim", "16", "--random-steps", "64", "--eval-every", "3000",
        "--eval-eps", "2", "--ep-secs", "4", "--log-secs", "0.2", "--device", "cpu",
        "--seed", "3", *extra])


REQUIRED = ["time/total_timesteps", "race/maps_finished", "race/eval_finish_s",
            "race/map_pct", "sgcrl/critic_loss", "sgcrl/critic_acc", "sgcrl/actor_loss",
            "sgcrl/alpha", "sgcrl/train_success", "sgcrl/sim_visited"]


@needs_core
def test_end_to_end_cpu_smoke(tmp_path):
    torch.set_num_threads(2)
    res = ts.train(_run_args(tmp_path, "smoke"))
    out = Path(res["out"])
    assert res["ticks"] >= 6000 and res["grad_steps"] > 0
    # progress.csv
    rows = list(csv.DictReader(open(out / "progress.csv", encoding="utf-8")))
    assert rows and all(c in rows[0] for c in REQUIRED)
    ev_rows = [r for r in rows if r["race/map_pct"] != ""]
    assert len(ev_rows) >= 3                          # step 0, 3000, 6000
    assert all(r["race/maps_finished"] in ("0", "1") for r in ev_rows)
    trained = [r for r in rows if r["sgcrl/critic_loss"] != ""]
    assert trained and all(np.isfinite(float(r["sgcrl/critic_loss"])) for r in trained)
    # run.json
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["tool"] == "train_sgcrl" and meta["finished"]
    assert meta["config"]["map"] == MAP.stem and "git" in meta
    assert meta["config"]["goal_override_TEST_ONLY"] is False
    assert meta["config"]["measure_field"]            # the cached field was used read-only
    # a greedy eval trajectory in the repo's JSONL format
    trajs = sorted(out.glob("traj_*.jsonl"))
    assert trajs
    lines = [json.loads(x) for x in trajs[-1].read_text(encoding="utf-8").splitlines()]
    heads = [x for x in lines if isinstance(x, dict) and "map" in x]
    tails = [x for x in lines if isinstance(x, dict) and "end" in x]
    body = [x for x in lines if isinstance(x, list)]
    assert len(heads) == len(tails) == 2
    assert heads[0]["map"] == MAP.stem and heads[0]["tick_ms"] == 10 and "phys" in heads[0]
    assert all(len(r) == 15 for r in body)
    assert sum(t["ticks"] for t in tails) == len(body)
    assert tails[0]["end"] in ("done", "fail", "trunc")
    # the checkpoint
    ck = torch.load(out / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    assert {"learner", "config", "counters", "arch"} <= set(ck)
    assert {"actor", "critic", "opt_a", "opt_c", "log_alpha", "opt_alpha"} <= set(ck["learner"])
    assert ck["counters"]["ticks"] == res["ticks"]
    assert (out / "evals.jsonl").exists()
    # resume continues the counters and appends to the same progress.csv
    res2 = ts.train(_run_args(tmp_path, "smoke", ["--resume", str(out / "ckpt_latest.pt"),
                                                   "--steps", "9000"]))
    assert res2["ticks"] >= 9000 and res2["grad_steps"] > res["grad_steps"]
    rows2 = list(csv.DictReader(open(out / "progress.csv", encoding="utf-8")))
    assert len(rows2) > len(rows)


@needs_core
def test_goal_point_override_is_labelled_test_only(tmp_path):
    torch.set_num_threads(2)
    res = ts.train(_run_args(tmp_path, "easy", ["--goal-point", "512,-1100,40",
                                                 "--steps", "2000", "--no-field"]))
    meta = json.loads((Path(res["out"]) / "run.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    assert cfg["goal_override_TEST_ONLY"] is True
    assert cfg["goal_world"] == [512.0, -1100.0, 40.0]
    assert cfg["goal_box"]["mins"] == [448.0, -1164.0, -24.0]
    assert cfg["measure_field"] is None
    # a test-box hit is never reported as a finished map
    rows = list(csv.DictReader(open(Path(res["out"]) / "progress.csv", encoding="utf-8")))
    assert "eval/box_hits" in rows[0]
    assert all(r["race/maps_finished"] in ("", "0") for r in rows)
    assert all(r["race/eval_finish_s"] == "" for r in rows)
    assert meta["result"]["eval_finishes"] == 0 and "eval_box_hits" in meta["result"]


@needs_core
def test_end_to_end_world_yaw(tmp_path):
    torch.set_num_threads(2)
    res = ts.train(_run_args(tmp_path, "world", ["--yaw", "world", "--steps", "3000",
                                                  "--no-field"]))
    out = Path(res["out"])
    cfg = json.loads((out / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["yaw"] == "world" and cfg["view_mode"] == 2 and cfg["act_dim"] == 5
    ck = torch.load(out / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    assert ck["arch"]["A"] == 5
    assert res["grad_steps"] > 0 and sorted(out.glob("traj_*.jsonl"))
    assert (out / "coverage.npz").exists()
