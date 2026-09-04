"""--rnn gru: the recurrent policy option (python/train_fast.py).

Four properties, each of which the trainer silently depends on:

1. `rnn="none"` (the default) is the pre-flag Policy bit for bit - same
   modules, same state_dict keys, same tensors out of the same torch seed,
   same forward. The reference is the pre-change train_fast.py read straight
   out of git (the test_trunk.py idiom), pinned to a sha.
2. The update's SEQUENCE re-run (features -> gru_sequence over the packed
   episode segments -> heads) reproduces the rollout's stepwise
   (policy(obs_t, h) with h zeroed where the episode ended) logits and
   values, from the same weights, the same h_0 and the same done masks.
3. A reset zeroes the state at exactly the right decision: the first
   decision of the new episode sees no history, the last of the old one
   still does.
4. The truncation bootstrap reads the state LEAVING the truncated decision
   (h at s_T), not the zeroed one - checked on the Policy protocol and on
   the trainer's source order.

Plus the two plumbing pieces: widen_for_rnn (a feed-forward checkpoint
warm-starts function-identically) and the eval wrapper's carry/reset off
the core's tick counter.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (N_SCALAR, GreedyTorchPolicy, HeadPacker,   # noqa: E402
                        Policy, gru_segments, widen_for_rnn)

W, H = 16, 8
OBS = N_SCALAR + W * H
R = 16
# the last goallines commit before --rnn existed; PINNED (see test_trunk.py)
BASE_REV = "7fc5e1b"


def _policy(**kw):
    return Policy(OBS, W, H, emb=32, hidden=24, **kw)


def _load_baseline():
    try:
        src = subprocess.run(
            ["git", "show", f"{BASE_REV}:python/train_fast.py"],
            cwd=ROOT, capture_output=True, check=True).stdout
    except Exception as exc:                       # no git / shallow checkout
        pytest.skip(f"cannot read the baseline from git: {exc!r}")
    tmp = ROOT / "python" / "_train_fast_rnn_baseline_tmp.py"
    tmp.write_bytes(src)
    try:
        spec = importlib.util.spec_from_file_location(
            "_train_fast_rnn_baseline_tmp", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_train_fast_rnn_baseline_tmp"] = mod
        spec.loader.exec_module(mod)
        if "rnn" in mod.Policy.__init__.__code__.co_varnames:
            pytest.skip(f"{BASE_REV} already has --rnn; nothing to compare")
        return mod
    finally:
        tmp.unlink(missing_ok=True)
        sys.modules.pop("_train_fast_rnn_baseline_tmp", None)


# ---- 1. rnn=none is the old policy ----------------------------------------

@pytest.mark.parametrize("kw", [
    {},                                            # the shipped baseline
    {"route_dim": 6},                              # --route
    {"route_dim": 6, "route_critic_only": True},
    {"in_ch": 2},                                  # --surf-mask
    {"trunk": "resnet"},
])
def test_none_is_bit_identical_to_the_pre_flag_policy(kw):
    base = _load_baseline()
    in_ch = kw.get("in_ch", 1)
    obs = N_SCALAR + kw.get("route_dim", 0) + W * H * in_ch

    torch.manual_seed(1234)
    old = base.Policy(obs, W, H, emb=32, hidden=24, **kw)
    torch.manual_seed(1234)
    new = Policy(obs, W, H, emb=32, hidden=24, rnn="none", **kw)
    assert new.gru is None and new.rnn_size == 0

    a, b = old.state_dict(), new.state_dict()
    assert list(a) == list(b), "state_dict KEYS moved"
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} differs -> init RNG order moved"

    x = torch.randn(5, obs)
    with torch.no_grad():
        la, va = old(x)
        lb, vb = new(x)
    assert torch.equal(la, lb) and torch.equal(va, vb)
    # and the split path the update takes
    with torch.no_grad():
        lc, vc = new.forward_split(x[:, :new.scal_dim], x[:, new.scal_dim:])
    assert torch.equal(la, lc) and torch.equal(va, vc)


def test_none_is_the_default_and_takes_no_state():
    torch.manual_seed(7)
    a = _policy().state_dict()
    torch.manual_seed(7)
    b = _policy(rnn="none").state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert not any(k.startswith("gru.") for k in a)


# ---- 2. the sequence re-run reproduces the stepwise rollout ----------------

def _rollout(p, obs, done, h0):
    """The trainer's rollout protocol: one GRU step per decision, the state
    zeroed AFTER the decision whose episode ended."""
    h = h0.clone()
    logits, values, leaving = [], [], []
    for t in range(obs.shape[0]):
        lg, v, h = p(obs[t], h)
        logits.append(lg)
        values.append(v)
        leaving.append(h)
        h = h * (1.0 - done[t].float()).unsqueeze(1)
    return torch.stack(logits), torch.stack(values), torch.stack(leaving), h


def _rerun(p, obs, done, h0, envs):
    """The update's protocol for one minibatch of whole env sequences,
    time-major rows (t*B + j), from the stored h_0 and the done masks."""
    T = obs.shape[0]
    sub = obs[:, envs]                                    # (T, B, OBS)
    B = sub.shape[1]
    scal = sub[:, :, :p.scal_dim].reshape(T * B, -1)
    img = sub[:, :, p.scal_dim:].reshape(T * B, -1)
    feat = p.features(scal, img)
    seg = gru_segments(done[:, envs].numpy(), "cpu")
    g = p.gru_sequence(feat, h0[envs], seg)
    lg, v = p.heads(feat, scal, g)
    return lg.view(T, B, -1), v.view(T, B), g.view(T, B, -1)


@pytest.mark.parametrize("kw", [{}, {"route_dim": 6}])
def test_sequence_rerun_reproduces_the_stepwise_rollout(kw):
    torch.manual_seed(3)
    obs_dim = N_SCALAR + kw.get("route_dim", 0) + W * H
    p = Policy(obs_dim, W, H, emb=32, hidden=24, rnn="gru", rnn_size=R,
               **kw)
    T, N = 14, 6
    obs = torch.randn(T, N, obs_dim)
    done = torch.rand(T, N) < 0.2
    done[T - 1, 0] = True          # a cut on the last row: no segment after it
    done[0, 1] = True              # a cut on the first row
    h0 = torch.randn(N, R)
    with torch.no_grad():
        lg, v, _, _ = _rollout(p, obs, done, h0)
        # one minibatch = every env
        lg2, v2, _ = _rerun(p, obs, done, h0, torch.arange(N))
        assert torch.allclose(lg2, lg, atol=1e-6, rtol=0), \
            (lg2 - lg).abs().max()
        assert torch.allclose(v2, v, atol=1e-6, rtol=0), (v2 - v).abs().max()
        # two minibatches of whole sequences (the --minibatches semantic):
        # rows are gathered per env subset, so each must reproduce its own
        # slice of the rollout
        for envs in (torch.tensor([0, 2, 4]), torch.tensor([5, 1, 3])):
            lg3, v3, _ = _rerun(p, obs, done, h0, envs)
            assert torch.allclose(lg3, lg[:, envs], atol=1e-6, rtol=0)
            assert torch.allclose(v3, v[:, envs], atol=1e-6, rtol=0)


def test_rerun_has_gradients_through_the_recurrence():
    torch.manual_seed(4)
    p = _policy(rnn="gru", rnn_size=R)
    T, N = 6, 3
    obs = torch.randn(T, N, OBS)
    done = torch.zeros(T, N, dtype=torch.bool)
    done[2, 1] = True
    lg, v, g = _rerun(p, obs, done, torch.zeros(N, R), torch.arange(N))
    (lg.square().mean() + v.square().mean()).backward()
    for name, q in p.gru.named_parameters():
        assert q.grad is not None and float(q.grad.abs().sum()) > 0.0, name
    # the towers' GRU columns (trailing rnn_size inputs) get gradient too
    assert float(p.pi[0].weight.grad[:, -R:].abs().sum()) > 0.0
    assert float(p.vf[0].weight.grad[:, -R:].abs().sum()) > 0.0


# ---- 3. resets zero h at the right decision --------------------------------

def test_reset_zeroes_the_state_at_the_first_decision_of_the_new_episode():
    torch.manual_seed(5)
    p = _policy(rnn="gru", rnn_size=R)
    T, N, e, t_end = 10, 4, 2, 5
    obs = torch.randn(T, N, OBS)
    done = torch.zeros(T, N, dtype=torch.bool)
    done[t_end, e] = True
    h0 = torch.randn(N, R)
    with torch.no_grad():
        lg, v, leaving, _ = _rollout(p, obs, done, h0)
        lg2, v2, g = _rerun(p, obs, done, h0, torch.arange(N))
        # decision t_end+1 of env e starts from ZERO: it equals a forward
        # from a zero state, whatever came before
        fresh_lg, fresh_v, fresh_h = p(obs[t_end + 1, e:e + 1],
                                       torch.zeros(1, R))
        assert torch.allclose(lg2[t_end + 1, e], fresh_lg[0], atol=1e-6)
        assert torch.allclose(g[t_end + 1, e], fresh_h[0], atol=1e-6)
        assert torch.allclose(lg[t_end + 1, e], fresh_lg[0], atol=1e-6)
        # decision t_end itself (the LAST of the old episode) still carries
        # its history: a zero-state forward is a different function of it
        old_lg = p(obs[t_end, e:e + 1], torch.zeros(1, R))[0][0]
        assert not torch.allclose(lg2[t_end, e], old_lg, atol=1e-4)
        assert torch.allclose(lg2[t_end, e], lg[t_end, e], atol=1e-6)
        # the other envs never reset: their whole run is one segment
        # from h0 - the packed plan says so
        seg = gru_segments(done.numpy(), "cpu")
        assert seg.n_seg == N + 1
        env = seg.env.tolist()
        first = seg.first.tolist()
        lens = seg.lens.tolist()
        cuts = [(en, fi, ln) for en, fi, ln in zip(env, first, lens)]
        assert (e, True, t_end + 1) in cuts and (e, False, T - t_end - 1) in cuts
        for other in range(N):
            if other != e:
                assert (other, True, T) in cuts


def test_gru_segments_covers_every_row_exactly_once():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T, B = int(rng.integers(1, 12)), int(rng.integers(1, 7))
        done = rng.random((T, B)) < 0.3
        seg = gru_segments(done, "cpu")
        idx, inv, lens = seg.idx.numpy(), seg.inv.numpy(), seg.lens.numpy()
        L, S = idx.shape
        valid = np.arange(L)[:, None] < lens[None, :]
        rows = idx[valid]
        assert sorted(rows.tolist()) == list(range(T * B))
        # inv maps each flat row back to its padded slot
        flat = np.arange(L * S).reshape(L, S)
        assert np.array_equal(flat[valid][np.argsort(rows)], inv)
        # a segment starting at t=0 is the only one that takes h_0
        starts = (idx[0] // B)
        assert np.array_equal(seg.first.numpy(), starts == 0)


# ---- 4. the bootstrap uses h at s_T ----------------------------------------

def test_bootstrap_value_uses_the_state_leaving_the_truncated_decision():
    torch.manual_seed(6)
    p = _policy(rnn="gru", rnn_size=R)
    T, N, e, t_tr = 6, 3, 1, 3
    obs = torch.randn(T, N, OBS)
    done = torch.zeros(T, N, dtype=torch.bool)
    done[t_tr, e] = True                 # truncated during decision t_tr
    h0 = torch.zeros(N, R)
    term_obs = torch.randn(1, OBS)       # the reconstructed terminal row
    with torch.no_grad():
        _, _, leaving, _ = _rollout(p, obs, done, h0)
        # the trainer's static_h protocol: the graph writes the state
        # LEAVING decision t into static_h, the bootstrap reads static_h[e]
        # inside the tick loop, and only then is the row zeroed
        static_h = leaving[t_tr].clone()
        v_boot = p(term_obs, static_h[e:e + 1])[1]
        # which is V(s_T | h_T): the same as continuing the sequence one
        # more step from the un-zeroed state...
        v_ref = p(term_obs, leaving[t_tr, e:e + 1])[1]
        assert torch.equal(v_boot, v_ref)
        # ...and NOT the zero-state value the post-reset row would give
        v_zero = p(term_obs, torch.zeros(1, R))[1]
        assert not torch.allclose(v_boot, v_zero, atol=1e-5)
    # and the trainer does it in that order: the bootstrap forward reads
    # static_h BEFORE the end-of-decision zeroing in the rollout loop
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    i_boot = src.index("tv = policy(full, static_h[")
    i_zero = src.index("static_h.mul_((1.0 - b_done[t]).unsqueeze(1))")
    # --priv-critic reflowed this call onto two lines (it gained a priv=
    # argument), so match the stable PREFIX rather than the whole statement -
    # what is under test is the ORDER of the three sites, not their spelling
    i_last = src.index("_, last_val, _ = policy(static_obs, static_h")
    assert src.count("tv = policy(full, static_h[") == 1
    assert src.count("static_h.mul_((1.0 - b_done[t]).unsqueeze(1))") == 1
    assert i_boot < i_zero < i_last


# ---- plumbing: warm start and the eval wrapper ------------------------------

def test_widen_for_rnn_warm_starts_function_identically():
    torch.manual_seed(8)
    ff = _policy()
    opt = torch.optim.Adam(ff.parameters(), lr=1e-3)
    x = torch.randn(4, OBS)
    lg, v = ff(x)
    (lg.square().mean() + v.square().mean()).backward()
    opt.step()                       # every param now has Adam moments
    ck = {"policy": ff.state_dict(), "optimizer": opt.state_dict()}
    n_ff = len(list(ff.parameters()))

    torch.manual_seed(9)
    rp = _policy(rnn="gru", rnn_size=R)
    n = widen_for_rnn(ck, rp)
    assert n == 2 + 4 + 4            # pi.0/vf.0 weights, their 2 Adam moments each, 4 GRU tensors
    rp.load_state_dict(ck["policy"])                 # strict: no missing keys
    opt2 = torch.optim.Adam(rp.parameters(), lr=1e-3)
    opt2.load_state_dict(ck["optimizer"])            # group size now matches
    assert len(opt2.param_groups[0]["params"]) == len(list(rp.parameters()))
    # the appended params are the GRU's, at the END of the parameter list
    assert [k for k, _ in rp.named_parameters()][n_ff:] == \
        [f"gru.{k}" for k, _ in rp.gru.named_parameters()]
    # function-identical at step 0, for ANY hidden state: the GRU columns
    # of both towers are zero
    with torch.no_grad():
        lg0, v0 = ff(x)
        for h in (torch.zeros(4, R), torch.randn(4, R) * 3):
            lg1, v1, _ = rp(x, h)
            assert torch.allclose(lg1, lg0, atol=1e-6) and \
                torch.allclose(v1, v0, atol=1e-6)
        assert torch.equal(rp.pi[0].weight[:, -R:], torch.zeros(24, R))
        assert torch.equal(rp.vf[0].weight[:, -R:], torch.zeros(24, R))
    # a recurrent checkpoint is left alone
    ck2 = {"policy": rp.state_dict(), "optimizer": opt2.state_dict()}
    assert widen_for_rnn(ck2, rp) == 0
    # and a feed-forward Policy is never touched
    assert widen_for_rnn({"policy": ff.state_dict()}, ff) == 0


class _FakeCore:
    """Just enough of SurfCore for _TorchPolicyBase._net: the per-env tick
    counter reset_env zeroes."""

    def __init__(self, n):
        self.states_view = {"tick": np.zeros(n, np.int64)}


def test_eval_wrapper_carries_the_state_and_zeroes_it_on_a_tick_reset():
    torch.manual_seed(10)
    p = _policy(rnn="gru", rnn_size=R)
    K, n = 4, 3
    core = _FakeCore(n)
    w = GreedyTorchPolicy(p, HeadPacker("cpu"), "cpu", lidar=None,
                          core=core, act_every=K)
    o1, o2, o3 = (torch.randn(n, OBS) for _ in range(3))
    with torch.no_grad():
        # decision 1 at tick 0: from zero
        w.act(o1.numpy())
        _, _, h1 = p(o1, torch.zeros(n, R))
        assert torch.allclose(w._h, h1, atol=1e-6)
        for _ in range(K - 1):                    # the held ticks
            w.act(o1.numpy())
        # decision 2, K ticks later, no reset: carried
        core.states_view["tick"][:] = K
        w.act(o2.numpy())
        _, _, h2 = p(o2, h1)
        assert torch.allclose(w._h, h2, atol=1e-6)
        for _ in range(K - 1):
            w.act(o2.numpy())
        # decision 3: env 0 was reset in between (its counter is 1, not
        # 2K), the others advanced by exactly K
        core.states_view["tick"][:] = 2 * K
        core.states_view["tick"][0] = 1
        w.act(o3.numpy())
        h_in = h2.clone()
        h_in[0] = 0.0
        _, _, h3 = p(o3, h_in)
        assert torch.allclose(w._h, h3, atol=1e-6)
        # a feed-forward policy through the same wrapper keeps no state
        w_ff = GreedyTorchPolicy(_policy(), HeadPacker("cpu"), "cpu",
                                 lidar=None, core=core, act_every=K)
        w_ff.act(o1.numpy())
        assert w_ff._h is None
