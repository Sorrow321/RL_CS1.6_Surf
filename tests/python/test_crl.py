"""--crl (python/surfgym/crl.py): single-goal contrastive RL as the PPO
advantage (CPPO, arXiv 2605.13554; SGCRL, 2408.05804; CRL, 2206.07568).

(a) the future-positive sampler: k ~ Geometric(1 - gamma) on {1, 2, ...};
    a positive is decision t + k of the SAME episode, or - once t + k runs
    past the episode's last decision - that episode's TERMINAL position;
    never a state of another episode; an anchor whose episode is still
    running at the newest decision is only resolvable while t + k is held;
    all of it across the ring's wrap-around.
(b) the critic: the InfoNCE loss falls and its top-1 accuracy rises on a
    toy random walk; infonce() matches a hand computation; dot and l2
    scores agree with their pairwise matrices.
(c) the advantage: A = Q - V is (R,), Q is the acted row's score, V the
    mean over K behaviour draws, so A is zero-mean when the acted actions
    come from the same distribution and exactly 0 for a deterministic
    policy; the sampler draws softmax(logits / temp) and N(mu, sigma).
(d) the feature blocks (state normalisation, one-hot layout, tanh z).
(e) trainer smokes (CPU, the toy scratch set): --crl trains (config keys,
    crl/* columns, vf 0, a checkpoint carrying the critic), records through
    tools/record_ckpt.py (the record gate's backend), resumes bare and
    continues; the refusals; the flag off gains no key and no column.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                                   # noqa: E402

from surfgym import crl as C                                   # noqa: E402

CPU = torch.device("cpu")
NEG = -1e30


def _gen(seed):
    g = torch.Generator(device=CPU)
    g.manual_seed(int(seed))
    return g


# ==========================================================================
# (a) the future-positive sampler
# ==========================================================================
def test_geometric_offsets_are_geometric_from_one():
    k = C.geometric_offsets(400_000, 0.9, _gen(1), CPU).numpy()
    assert k.dtype == np.int64 and k.min() == 1
    assert abs(k.mean() - 10.0) < 0.1                   # 1 / (1 - gamma)
    assert abs((k == 1).mean() - 0.1) < 0.003          # 1 - gamma
    assert abs((k > 10).mean() - 0.9 ** 10) < 0.004     # gamma^m tail
    k99 = C.geometric_offsets(200_000, 0.99, _gen(2), CPU).numpy()
    assert abs(k99.mean() - 100.0) < 1.5


def _world(n_env, n_blocks, T, p_end, seed):
    """A scripted history: every (decision g, env) row carries its own
    coordinates, pos = (g, env, 0) and a terminal marker (g + 0.5, env, 1),
    so a sampled positive says exactly which row it came from. Returns the
    blocks and, per env, the global index of each decision's episode end."""
    rng = np.random.default_rng(seed)
    blocks = []
    ends = rng.random((n_blocks * T, n_env)) < p_end
    for b in range(n_blocks):
        g = np.arange(b * T, (b + 1) * T, dtype=np.float32)
        env = np.arange(n_env, dtype=np.float32)
        gg, ee = np.meshgrid(g, env, indexing="ij")
        xs = np.stack([gg, ee], -1)                            # (T, N, 2)
        xa = np.zeros((T, n_env, 1), np.float32)
        pos = np.stack([gg, ee, np.zeros_like(gg)], -1)
        tpos = np.stack([gg + 0.5, ee, np.ones_like(gg)], -1)
        end = ends[b * T:(b + 1) * T]
        blocks.append(tuple(torch.as_tensor(np.ascontiguousarray(a))
                            for a in (xs, xa, pos, end, tpos)))
    return blocks, ends


def _truth(ends):
    """(episode id, index of the episode's last decision or -1 if still
    open at the newest decision) for every (g, env)."""
    G, N = ends.shape
    ep = np.zeros((G, N), np.int64)
    last = np.full((G, N), -1, np.int64)
    for n in range(N):
        cur, start = 0, 0
        for g in range(G):
            ep[g, n] = cur
            if ends[g, n]:
                last[start:g + 1, n] = g
                cur += 1
                start = g + 1
    return ep, last


@pytest.mark.parametrize("L,T,n_blocks", [(16, 6, 5), (64, 16, 3), (8, 8, 9)])
def test_positives_stay_in_the_episode_clip_to_its_terminal_and_wrap(L, T,
                                                                     n_blocks):
    N = 5
    ring = C.FutureRing(L, N, ds=2, da=1, device=CPU)
    blocks, ends = _world(N, n_blocks, T, p_end=0.15, seed=L + T)
    for b in blocks:
        ring.push(*b)
    G = n_blocks * T
    assert ring.n == G and ring.valid == min(G, L)
    ep_true, last_true = _truth(ends)
    newest = G - 1
    ix = ring.sample_index(40_000, 0.8, _gen(7))
    x, g, ep_a, ep_g = ring.gather(ix)
    a = x[:, 0].round().long().numpy()                  # the anchor's decision
    e = x[:, 1].round().long().numpy()
    k = ix["k"].numpy()
    ok = ix["ok"].numpy()
    in_ep = ix["in_ep"].numpy()
    term = ix["term"].numpy()
    assert np.array_equal(e, ix["e"].numpy())
    # only the last L decisions are ever anchors (the ring wrapped)
    assert a.min() >= max(0, G - L) and a.max() <= newest
    assert np.array_equal(a - (G - ring.valid), ix["p"].numpy())
    assert not (in_ep & term).any()
    # the ring's episode ids ARE the scripted ones
    assert np.array_equal(ep_a.numpy(), ep_true[a, e])
    lst = last_true[a, e]
    gx = g[:, 0].numpy()
    # (1) a positive inside the episode is decision a + k, a pos row, and
    #     the episode has not ended before it
    m = in_ep
    assert m.any()
    assert np.array_equal(gx[m], (a + k)[m].astype(np.float32))
    assert (g[:, 2].numpy()[m] == 0.0).all()
    assert ((lst[m] == -1) | (a[m] + k[m] <= lst[m])).all()
    assert (a[m] + k[m] <= newest).all()
    assert np.array_equal(ep_true[(a + k)[m], e[m]], ep_true[a[m], e[m]])
    # (2) past the episode's last decision: its TERMINAL position, marked
    #     g + 0.5 of the very decision the episode ended on
    m = term
    assert m.any()
    assert (g[:, 2].numpy()[m] == 1.0).all()
    assert np.array_equal(gx[m], (lst[m] + 0.5).astype(np.float32))
    assert ((lst[m] >= a[m]) & (a[m] + k[m] > lst[m])).all()
    # (3) never across an episode boundary
    assert np.array_equal(ep_a.numpy()[ok], ep_g.numpy()[ok])
    # (4) unresolvable = the episode is still open AND a + k is not held yet
    bad = ~ok
    assert ((lst[bad] == -1) & (a[bad] + k[bad] > newest)).all()
    assert not (((lst == -1) & (a + k > newest)) & ok).any()


def test_sample_keeps_only_resolvable_pairs_and_counts_them():
    ring = C.FutureRing(32, 4, ds=2, da=1, device=CPU)
    blocks, ends = _world(4, 4, 8, p_end=0.2, seed=3)
    for b in blocks:
        ring.push(*b)
    x, g, m, n_ok = ring.sample(256, 0.9, _gen(5), oversample=4)
    assert m == 1024 and 0 < n_ok <= m
    assert x.shape == (min(256, n_ok), 3) and g.shape == (x.shape[0], 3)
    ep_true, last_true = _truth(ends)
    a = x[:, 0].round().long().numpy()
    e = x[:, 1].round().long().numpy()
    gx, gt = g[:, 0].numpy(), g[:, 2].numpy()
    # every kept positive is a future of its anchor inside the same episode
    for ai, ei, gi, ti in zip(a, e, gx, gt):
        if ti == 0.0:
            assert gi > ai and ep_true[int(gi), ei] == ep_true[ai, ei]
        else:
            assert last_true[ai, ei] == int(gi - 0.5) >= ai


def test_push_rejects_a_block_longer_than_the_ring_and_episode_ids_count():
    ring = C.FutureRing(4, 2, ds=1, da=1, device=CPU)
    z = torch.zeros
    with pytest.raises(ValueError, match="crl-history"):
        ring.push(z(5, 2, 1), z(5, 2, 1), z(5, 2, 3),
                  torch.zeros(5, 2, dtype=torch.bool), z(5, 2, 3))
    end = torch.tensor([[1, 0], [0, 0], [1, 1]], dtype=torch.bool)
    ring.push(z(3, 2, 1), z(3, 2, 1), z(3, 2, 3), end, z(3, 2, 3))
    # the row an episode ENDS on still belongs to that episode
    assert ring.ep[:3].tolist() == [[0, 0], [1, 0], [1, 0]]
    assert ring.ep_ctr.tolist() == [2, 1]
    assert ring.next_end.tolist() == [[0, 2], [2, 2], [2, 2]]


# ==========================================================================
# (b) the critic
# ==========================================================================
def test_infonce_matches_a_hand_computation():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.0, 3.0], [0.0, 0.0, 0.0]])
    loss, ce, acc, lse = C.infonce(logits, 0.01)
    lse_h = torch.logsumexp(logits, 1)
    ce_h = float((lse_h - logits.diag()).mean())
    assert float(ce) == pytest.approx(ce_h, rel=1e-6)
    assert float(loss) == pytest.approx(ce_h + 0.01 * float((lse_h ** 2).mean()),
                                        rel=1e-6)
    assert float(acc) == pytest.approx(1.0 / 3.0)   # row 0; row 2's tie -> 0
    assert float(lse) == pytest.approx(float(lse_h.mean()), rel=1e-6)


@pytest.mark.parametrize("kind", ["dot", "l2"])
def test_pair_logits_and_rowwise_score_agree(kind):
    torch.manual_seed(0)
    c = C.ContrastiveCritic(5, 3, repr_dim=8, hidden=16, kind=kind)
    sa, g = torch.randn(6, 8), torch.randn(4, 8)
    m = c.pair_logits(sa, g)
    assert m.shape == (6, 4)
    for i in range(6):
        for j in range(4):
            assert float(m[i, j]) == pytest.approx(float(c.score(sa[i], g[j])),
                                                   rel=1e-4, abs=1e-4)
    if kind == "l2":
        assert (m <= 0).all()
    with pytest.raises(ValueError):
        C.ContrastiveCritic(5, kind="cos")


@pytest.mark.parametrize("kind", ["dot", "l2"])
def test_critic_learns_a_toy_random_walk(kind):
    """Envs random-walk on a line, the action choosing the direction; the
    critic has to pick each anchor's own future out of 128. The InfoNCE loss
    must fall well below chance (log 128 = 4.85) and the accuracy rise.
    (Measured at this seed: dot 4.81 -> 3.23 nats and 0.014 -> 0.125, l2
    4.83 -> 2.29 and 0.009 -> 0.421 over the 240 updates; the ceiling is the
    walk's own ambiguity - several anchors of a batch share a future.)"""
    rng = np.random.default_rng(0)
    N, T, L = 64, 32, 256
    frame = C.MapFrame([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    cr = C.ContrastiveRL(C.STATE_DIM + 2, C.STATE_DIM, N, frame,
                         [0.9, 0.0, 0.0], CPU, history=L, gamma=0.5,
                         repr_dim=16, hidden=64, kind=kind, lr=1e-3,
                         lse=0.01, v_samples=4, seed=1)
    x = rng.uniform(-1, 1, N)

    def rollout():
        nonlocal x
        raw = np.zeros((T, N, C.RAW_DIM), np.float32)
        act = np.zeros((T, N), np.int64)
        end = np.zeros((T, N), bool)
        tpos = np.zeros((T, N, 3), np.float32)
        for t in range(T):
            raw[t, :, 0] = x
            a = rng.integers(0, 2, N)
            act[t] = a
            x = np.clip(x + np.where(a == 1, 0.02, -0.02), -1, 1)
            e = rng.random(N) < 0.03
            end[t] = e
            tpos[t, e, 0] = x[e]
            x = np.where(e, rng.uniform(-1, 1, N), x)
        rt = torch.as_tensor(raw)
        xs = C.state_features(rt, frame)
        xa = C.action_features(torch.as_tensor(act)[..., None], (2,))
        cr.push(xs, xa, frame.norm(rt[..., 0:3]), torch.as_tensor(end),
                frame.norm(torch.as_tensor(tpos)))

    ces, accs = [], []
    for _ in range(L // T):
        rollout()
    for it in range(60):
        if it % 4 == 0:
            rollout()
        cr.train(4, 128)
        st = cr.pop_stats()
        ces.append(st["loss"])
        accs.append(st["acc"])
    first, last = np.mean(ces[:5]), np.mean(ces[-5:])
    assert first > 4.0, ces[:5]
    assert last < first - 1.0, (first, last)
    assert np.mean(accs[-5:]) > np.mean(accs[:5]) + 0.05, (accs[:5], accs[-5:])
    assert cr.n_updates == 240


# ==========================================================================
# (c) the advantage
# ==========================================================================
def _toy_policy(R, seed):
    torch.manual_seed(seed)
    sizes = (3, 2)
    logits = torch.randn(R, 2, 3) * 1.5
    logits[:, 1, 2] = NEG                                # head 1 has 2 bins
    mu = torch.randn(R, 2) * 0.5
    ls = torch.tensor([-1.0, -0.5])
    xs = torch.randn(R, 4)
    return sizes, logits, mu, ls, xs


def test_advantage_shapes_and_zero_mean_under_the_behaviour_policy():
    R = 6000
    sizes, logits, mu, ls, xs = _toy_policy(R, 0)
    frame = C.MapFrame([-1] * 3, [1] * 3)
    cr = C.ContrastiveRL(4 + 5 + 2, 4, 8, frame, [0.3, 0.2, 0.1], CPU,
                         history=16, repr_dim=8, hidden=32, v_samples=64,
                         seed=3)
    # the acted rows: one draw from the very distribution V averages over
    idx, z = C.sample_actions(logits, _gen(11), 1, None, mu, ls)
    xa = C.action_features(idx[0], sizes, z[0])
    pos = torch.rand(R, 3) * 2 - 1
    a, q, v = cr.advantage(xs, xa, logits, sizes, mu=mu, log_std=ls, pos=pos,
                           chunk=1024)
    assert a.shape == q.shape == v.shape == (R,)
    assert torch.allclose(a, q - v)
    # Q is the acted row's own score
    with torch.no_grad():
        g = cr.critic.psi(cr.goal)
        q_h = cr.critic.score(cr.critic.phi(torch.cat([xs, xa], -1)), g)
        sim_h = float(cr.critic.score(cr.critic.psi(pos), g).mean())
    assert torch.allclose(q, q_h, atol=1e-5)
    # zero mean: E[Q(a_t)] - E[mean_k Q(a_k)] with a_t, a_k ~ the same law
    sd = float(a.std())
    assert sd > 0.0
    assert abs(float(a.mean())) < 4.0 * sd / math.sqrt(R) + 1e-3
    st = cr.pop_stats()
    assert st["q_goal"] == pytest.approx(float(q.mean()), rel=1e-5)
    assert st["adv_std"] == pytest.approx(sd, rel=1e-5)
    assert st["sim_visited"] == pytest.approx(sim_h, rel=1e-4, abs=1e-5)
    assert math.isnan(st["loss"])                        # no update this time


def test_advantage_is_zero_for_a_deterministic_policy_and_temp_is_used():
    R = 500
    sizes, logits, mu, ls, xs = _toy_policy(R, 1)
    det = torch.full_like(logits, NEG)
    choice = torch.stack([torch.randint(0, 3, (R,)), torch.randint(0, 2, (R,))],
                         1)
    det.scatter_(2, choice.unsqueeze(-1), 0.0)
    frame = C.MapFrame([-1] * 3, [1] * 3)
    cr = C.ContrastiveRL(4 + 5 + 2, 4, 8, frame, [0.0, 0.0, 0.0], CPU,
                         history=16, repr_dim=8, hidden=32, v_samples=8,
                         seed=4)
    xa = C.action_features(choice, sizes, mu)
    tiny = torch.full((2,), -30.0)
    for temp in (None, torch.tensor(3.0)):
        a, q, v = cr.advantage(xs, xa, det, sizes, cat_temp=temp, mu=mu,
                               log_std=tiny)
        assert float(a.abs().max()) < 1e-4
    # the sampler's categorical law is softmax(logits / temp)
    one = logits[:1]
    n = 100_000
    for temp in (1.0, 2.5):
        idx, _ = C.sample_actions(one, _gen(9), n, torch.tensor(temp),
                                  None, None)
        f = torch.bincount(idx[:, 0, 0], minlength=3).float() / n
        want = torch.softmax(one[0, 0] / temp, 0)
        assert (f - want).abs().max() < 0.01, (temp, f, want)
        f1 = torch.bincount(idx[:, 0, 1], minlength=3).float() / n
        assert f1[2] == 0.0                              # the NEG pad
    # and its Gaussian is N(mu, exp(log_std))
    _, z = C.sample_actions(logits[:1], _gen(10), n, None, mu[:1], ls)
    assert torch.allclose(z.reshape(-1, 2).mean(0), mu[0], atol=0.01)
    assert torch.allclose(z.reshape(-1, 2).std(0), ls.exp(), rtol=0.02)


# ==========================================================================
# (d) the feature blocks
# ==========================================================================
def test_state_features_normalise_by_the_map_frame():
    frame = C.MapFrame([-1000.0, 0.0, -500.0], [3000.0, 1000.0, 500.0])
    assert frame.center.tolist() == [1000.0, 500.0, 0.0]
    assert frame.scale == 2000.0
    raw = torch.tensor([[3000.0, 500.0, 0.0, 1500.0, -250.0, 100.0,
                         90.0, -45.0, -1.0, 0.0],
                        [1000.0, 500.0, 500.0, 0.0, 0.0, 0.0,
                         0.0, 30.0, 3.0, 1.0]])
    f = C.state_features(raw, frame)
    assert f.shape == (2, C.STATE_DIM) == (2, 11)
    assert torch.allclose(f[0, 0:3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(f[1, 0:3], torch.tensor([0.0, 0.0, 0.25]))
    assert torch.allclose(f[0, 3:6], torch.tensor([1.5, -0.25, 0.1]))
    assert torch.allclose(f[0, 6:8], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(f[1, 6:8], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert f[0, 8] == pytest.approx(-0.5) and f[1, 8] == pytest.approx(1 / 3)
    assert f[:, 9].tolist() == [0.0, 1.0]               # -1 airborne
    assert f[:, 10].tolist() == [0.0, 1.0]
    assert np.allclose(frame.norm_np([1000.0, 500.0, 2000.0]), [0, 0, 1])
    states = np.zeros(3, dtype=[("origin", np.float32, (3,)),
                                ("velocity", np.float32, (3,)),
                                ("yaw", np.float32), ("pitch", np.float32),
                                ("onground", np.int32), ("ducked", np.int32)])
    states["origin"][1] = (1, 2, 3)
    states["onground"][2] = -1
    out = np.full((3, C.RAW_DIM), 7.0, np.float32)
    C.raw_from_states(states, out)
    assert out[1, 0:3].tolist() == [1, 2, 3] and out[2, 8] == -1.0
    assert out[0].tolist() == [0.0] * 10


def test_action_features_one_hot_layout_and_tanh_z():
    idx = torch.tensor([[0, 1, 2], [3, 0, 1]])
    f = C.action_features(idx, (4, 2, 3))
    assert f.tolist() == [[1, 0, 0, 0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 1, 1, 0, 0, 1, 0]]
    z = torch.tensor([[0.5, -2.0], [0.0, 1.0]])
    g = C.action_features(idx, (4, 2, 3), z)
    assert g.shape == (2, 11)
    assert torch.allclose(g[:, 9:], torch.tanh(z))
    k = C.action_features(idx.unsqueeze(0).expand(5, -1, -1), (4, 2, 3),
                          z.unsqueeze(0).expand(5, -1, -1))
    assert k.shape == (5, 2, 11) and torch.equal(k[3], g)


def test_state_dict_round_trip():
    frame = C.MapFrame([-1] * 3, [1] * 3)
    a = C.ContrastiveRL(13, 11, 4, frame, [0.1, 0.2, 0.3], CPU, history=8,
                        repr_dim=8, hidden=16, seed=5)
    b = C.ContrastiveRL(13, 11, 4, frame, [0.1, 0.2, 0.3], CPU, history=8,
                        repr_dim=8, hidden=16, seed=6)
    for n, p in a.critic.named_parameters():
        if n.endswith("weight"):                  # the biases start at zero
            assert not torch.equal(p, dict(b.critic.named_parameters())[n])
    a.n_updates = 17
    b.load_state_dict_all(a.state_dict_all())
    for n, p in a.critic.named_parameters():
        assert torch.equal(p, dict(b.critic.named_parameters())[n])
    assert b.n_updates == 17
    assert torch.equal(torch.rand(3, generator=a.gen),
                       torch.rand(3, generator=b.gen))
    # the critic's init does not move the global CPU stream
    torch.manual_seed(123)
    r0 = torch.rand(4)
    torch.manual_seed(123)
    C.ContrastiveRL(13, 11, 4, frame, [0, 0, 0], CPU, history=8, seed=9)
    assert torch.equal(torch.rand(4), r0)


# ==========================================================================
# (e) trainer smokes
# ==========================================================================
from test_unstuck import ABS, TRAIN, _csv, _run, _train     # noqa: E402
from test_view_continuous import (CANNONBALL, SMOKE_FLAGS,  # noqa: E402
                                  needs_run)

RECORD = ROOT / "tools" / "record_ckpt.py"
CRL_KEYS = ("crl", "crl_critic", "crl_history", "crl_gamma", "crl_updates",
            "crl_batch", "crl_lr", "crl_repr", "crl_hidden", "crl_lse",
            "crl_v_samples", "crl_goal", "crl_inputs")
CRL_COLS = ["crl/loss", "crl/acc", "crl/lse", "crl/valid", "crl/q_goal",
            "crl/sim_visited", "crl/adv_std"]
# --crl-gamma 0.5: the toy rollout is 8 decisions long and no episode ends in
# it, so at 0.99 almost no future is resolvable yet and an update could be
# skipped (a legitimate outcome, but the counts below want every update)
ON = ABS + ["--keys-hold", "--crl", "--crl-updates", "4", "--crl-batch", "64",
            "--crl-history", "64", "--crl-gamma", "0.5", "--ep-ticks", "400"]


def _cfg(d: Path):
    return json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_crl_trains_records_resumes_and_refuses():
    run = "crl_on"
    d = ROOT / "runs" / run
    r = _train(run, ON)
    assert "--crl: single-goal contrastive critic (dot)" in r.stdout
    assert "--crl: the value loss is OFF (--vf 0.5 -> 0)" in r.stdout
    cfg = _cfg(d)
    assert all(k in cfg for k in CRL_KEYS)
    assert cfg["crl"] == 1 and cfg["crl_critic"] == "dot"
    assert cfg["crl_history"] == 64 and cfg["crl_batch"] == 64
    assert cfg["vf"] == 0.0
    # 11 state + 7 held keys | 4+4+2+3 one-hot | 2 tanh z
    assert cfg["crl_inputs"][-2:] == ["onehot[4, 4, 2, 3]", "tanh_z2"]
    rows = _csv(run)
    assert len(rows) == 3
    head = list(rows[0])
    assert head[-len(CRL_COLS):] == CRL_COLS
    for x in rows:
        for c in CRL_COLS:
            assert x[c] != "" and np.isfinite(float(x[c])), (c, x[c])
        assert 0.0 <= float(x["crl/acc"]) <= 1.0
        assert 0.0 < float(x["crl/valid"]) <= 1.0
        assert float(x["crl/adv_std"]) > 0.0
        assert np.isfinite(float(x["train/approx_kl"]))
    assert "crl L " in r.stdout and " Asd " in r.stdout
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["crl"] == 1
    assert set(ck["crl"]) >= {"critic", "opt", "gen", "n_updates", "goal"}
    assert ck["crl"]["n_updates"] == 12                 # 3 iterations x 4
    sd = ck["crl"]["critic"]
    assert tuple(sd["phi.0.weight"].shape) == (256, 33)
    assert tuple(sd["psi.0.weight"].shape) == (256, 3)
    assert tuple(sd["phi.4.weight"].shape) == (64, 256)
    # the POLICY is the control's shape: no critic tensor leaks into it
    assert not any(k.startswith(("phi.", "psi.")) for k in ck["policy"])

    # the record gate's backend loads it (every crl_* key is TRAIN_ONLY)
    rr = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
               "--map", str(CANNONBALL), "--episodes", "1", "--ep-ticks", "200",
               "--out", str(d / "rec.jsonl")], timeout=900)
    assert rr.returncode == 0, rr.stdout[-3000:] + rr.stderr[-3000:]
    assert (d / "rec.jsonl").exists()

    # a BARE resume restores --crl and every knob, and continues the critic
    re = ROOT / "runs" / "crl_re"
    shutil.rmtree(re, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", re.name,
               "--ckpt", str(d / "ckpt_final.pt")] + SMOKE_FLAGS
              + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "crl=1" in r2.stdout and "restored the --crl critic" in r2.stdout
    c2 = _cfg(re)
    assert {k: c2[k] for k in CRL_KEYS} == {k: cfg[k] for k in CRL_KEYS}
    assert c2["vf"] == 0.0
    ck2 = torch.load(re / "ckpt_final.pt", map_location="cpu",
                     weights_only=False)
    assert ck2["crl"]["n_updates"] == 16
    assert ck2["global_step"] > ck["global_step"]

    # the refusals
    for extra, msg in (
            (["--crl", "--int-split", "--int-coef", "1.0"],
             "--crl is not implemented with --int-split"),
            (["--crl", "--race-sr"], "--crl is not implemented with --race-sr"),
            (["--crl", "--obs-reward"],
             "--crl is not implemented with --obs-reward"),
            (["--crl-gamma", "0.9"], "--crl-gamma without --crl"),
            (["--crl", "--crl-gamma", "1.0"], "--crl-gamma must be in (0, 1)"),
            (["--crl", "--crl-history", "4"], "shorter than one rollout"),
            (["--crl", "--reward", "forward"], "only --reward race")):
        rb = _run([sys.executable, "-u", str(TRAIN), "--run", "crl_bad"]
                  + SMOKE_FLAGS + ["--steps", "2048"] + extra)
        assert rb.returncode != 0, extra
        assert msg in rb.stdout + rb.stderr, (extra, rb.stdout[-2000:]
                                              + rb.stderr[-2000:])
    for p in (d, re, ROOT / "runs" / "crl_bad"):
        shutil.rmtree(p, ignore_errors=True)


@needs_run
def test_crl_bins_and_l2_critic_train():
    """The discrete bins (all six heads one-hot, no view z) and CPPO's
    -||phi - psi|| score, flag for flag the same pipeline."""
    run = "crl_bins_l2"
    d = ROOT / "runs" / run
    r = _train(run, ["--crl", "--crl-critic", "l2", "--crl-updates", "3",
                     "--crl-batch", "32", "--crl-history", "32",
                     "--crl-v-samples", "4", "--ep-ticks", "400"])
    assert "--crl: single-goal contrastive critic (l2)" in r.stdout
    cfg = _cfg(d)
    assert cfg["crl_critic"] == "l2" and cfg["crl_v_samples"] == 4
    assert cfg["crl_inputs"][-1] == "onehot[15, 7, 3, 3, 2, 2]"
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["crl"]["critic"]["phi.0.weight"].shape) == (256, 11 + 32)
    for x in _csv(run):
        assert np.isfinite(float(x["crl/adv_std"]))
        assert float(x["crl/q_goal"]) <= 0.0            # -distance
    shutil.rmtree(d, ignore_errors=True)


@needs_run
def test_flag_off_gains_no_key_and_no_column():
    run = "crl_off"
    d = ROOT / "runs" / run
    r = _train(run, ABS + ["--keys-hold", "--ep-ticks", "400"], steps="4096")
    assert "--crl" not in r.stdout
    cfg = _cfg(d)
    assert not any(k in cfg for k in CRL_KEYS)
    assert cfg["vf"] == 0.5
    assert not any(c.startswith("crl/") for c in _csv(run)[0])
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert "crl" not in ck
    shutil.rmtree(d, ignore_errors=True)
