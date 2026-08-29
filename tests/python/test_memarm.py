"""xMEM: --stack-strides and --act-hist (CPU only).

Two new observation blocks, and both have the same failure mode as the frame
stack: the rollout builds them one way, the PPO update rebuilds them another,
and if the two ever disagree PPO optimizes the network on inputs the policy
never saw - silently, with no shape error and no NaN.

--stack-strides adds a THIRD copy of that risk, because it changes no tensor
shape at all: a run that installed the ladder in the trainer but not in the
eval helper would train on a 450 ms window and be judged on a 120 ms one, and
race/eval_progress is the number every arm is judged by.

The rest: history must collapse at a spawn, out-of-range action slots must
read NEUTRAL_ACT (an action from before the episode began is a lie about the
AGENT, and "I had not acted yet" is the honest substitute), the act-hist
block must reach BOTH towers, and with the flags off nothing may move.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (ACT_CENTER, NACT, NEUTRAL_ACT,        # noqa: E402
                        NEUTRAL_ENC, NVEC, N_SCALAR, ActRing,
                        FrameRing, GreedyTorchPolicy, Policy, STACK_STRIDES,
                        _parse_strides, acthist_from_buffer,
                        acthist_from_ring, encode_actions, frame_offsets,
                        neutral_enc, set_stack_strides, stack_from_buffer,
                        stack_from_ring, stack_strides, widen_for_route)

XMEM = (5, 10, 15)                  # the arm's ladder: 150/300/450 ms
N, P, T = 5, 3, 24                  # envs, "pixels", decisions
M = 4                               # act-hist depth used by the parity tests


@pytest.fixture(autouse=True)
def _restore_ladder():
    """No test may leak an installed ladder into the next one."""
    yield
    set_stack_strides(None)


# --------------------------------------------------------- --stack-strides --
def test_the_default_is_untouched():
    assert stack_strides() == STACK_STRIDES == (1, 2, 4, 8)
    assert frame_offsets(4) == (0, 1, 2, 4)
    assert frame_offsets(5) == (0, 1, 2, 4, 8)


def test_parse_and_validate():
    assert _parse_strides("5,10,15") == XMEM
    assert _parse_strides(" 5 , 10 ,15 ") == XMEM
    assert _parse_strides(None) is None and _parse_strides("") is None
    assert _parse_strides([5, 10, 15]) == XMEM
    with pytest.raises(SystemExit):
        set_stack_strides((5, 5, 10))          # not strictly increasing
    with pytest.raises(SystemExit):
        set_stack_strides((10, 5))             # decreasing
    with pytest.raises(SystemExit):
        set_stack_strides((0, 5))              # offset 0 is the current frame
    with pytest.raises(SystemExit):
        set_stack_strides(())


def test_the_override_moves_the_offsets_and_the_arity():
    assert set_stack_strides(XMEM) == XMEM
    assert frame_offsets(4) == (0, 5, 10, 15)
    assert frame_offsets(2) == (0, 5) and frame_offsets(1) == (0,)
    with pytest.raises(ValueError, match="only 4 frames"):
        frame_offsets(5)                       # 3 strides + the current frame
    set_stack_strides(None)
    assert frame_offsets(4) == (0, 1, 2, 4), "the ladder did not come back"


def test_ring_depth_and_prologue_are_DERIVED_not_hardcoded():
    """The whole point of the flag. A ring sized 8 deep would silently clamp
    the 10- and 15-decision reach-backs onto the 8-decision one, and every
    'past' frame beyond 8 would be the same picture."""
    set_stack_strides(XMEM)
    r = FrameRing(4, N, P, "cpu")
    assert r.pro == 15 and r.buf.shape[0] == 16
    set_stack_strides(None)
    assert FrameRing(4, N, P, "cpu").pro == 4


def _simulate_frames(resets, k, seed=0):
    """Drive the REAL FrameRing in train_fast's rollout order."""
    rng = torch.Generator().manual_seed(seed)
    ring = FrameRing(k, N, P, "cpu")
    pro = ring.pro
    buf = torch.zeros((pro + T, N, P))
    ages = torch.zeros((T, N), dtype=torch.long)
    seen = []
    ring.push(torch.rand((N, P), generator=rng), None)
    for _ in range(pro):
        ring.push(torch.rand((N, P), generator=rng),
                  torch.zeros(N, dtype=torch.bool))
    ring.fill_prologue(buf)
    for t in range(T):
        ring.record(buf, ages, t)
        seen.append(ring.compose())
        ring.push(torch.rand((N, P), generator=rng), resets[t])
    return torch.stack(seen), buf, ages, pro


def _resets(seed=1, p=0.10):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand((T, N), generator=g) < p
    r[:, 0] = False
    r[3, 1] = r[4, 1] = True
    r[0, 2] = True
    return r


def test_the_update_gather_agrees_under_the_new_ladder():
    """THE test, re-run on 5,10,15: every sample's stacked image out of the
    flat buffer must be bit-identical to the stack the ring handed over."""
    set_stack_strides(XMEM)
    seen, buf, ages, pro = _simulate_frames(_resets(), 4)
    assert pro == 15
    idx = torch.arange(T * N)
    got = stack_from_buffer(buf.reshape((pro + T) * N, P), idx,
                            ages.reshape(-1)[idx], 4, N, pro)
    assert torch.equal(got, seen.reshape(T * N, P * 4))


def test_the_new_ladder_actually_reaches_further():
    """Not vacuous: with 5,10,15 the oldest frame must differ from the one
    the default ladder would have shown."""
    set_stack_strides(XMEM)
    wide, _, _, _ = _simulate_frames(_resets(seed=2, p=0.0), 4)
    set_stack_strides(None)
    narrow, _, _, _ = _simulate_frames(_resets(seed=2, p=0.0), 4)
    w = wide.reshape(T * N, P, 4)[:, :, 3]
    n = narrow.reshape(T * N, P, 4)[:, :, 3]
    assert not torch.equal(w, n), "the ladder did not change what was fed"


def test_the_eval_helper_uses_the_installed_ladder():
    """Train/eval parity. STACK_STRIDES is a module global, so the one
    failure that matters is the trainer and the recorder disagreeing."""
    set_stack_strides(XMEM)
    pol = GreedyTorchPolicy(None, None, "cpu", stack=4)
    pol._push_frame(torch.zeros((N, P)), np.zeros(N, np.int64))
    assert pol._ring.pro == 15, "the eval ring is on the default ladder"
    # and it composes the same window the trainer's ring does
    ticks = np.arange(N, dtype=np.int64) * 0
    got = [pol._push_frame(torch.full((N, P), float(d)), ticks + d * 3)
           for d in range(1, 20)]
    want = stack_from_ring(pol._ring.buf, pol._ring.head, pol._ring.age, 4)
    assert torch.equal(got[-1], want)
    assert got[-1].reshape(N, P, 4)[0, 0].tolist() == [19.0, 14.0, 9.0, 4.0]


# ---------------------------------------------------------------- encoding --
def test_the_action_encoding_is_centred_and_unit_scaled():
    assert ACT_CENTER == (7.0, 3.0, 1.0, 1.0, 0.5, 0.5)
    assert NEUTRAL_ENC == (0.0, 0.0, 0.0, 0.0, -1.0, -1.0)
    c = torch.tensor(ACT_CENTER)
    lo = encode_actions(torch.zeros(1, NACT, dtype=torch.long), c)
    hi = encode_actions(torch.tensor([[n - 1 for n in NVEC]]), c)
    assert torch.equal(lo, -torch.ones(1, NACT))
    assert torch.equal(hi, torch.ones(1, NACT))
    mid = encode_actions(torch.tensor([list(NEUTRAL_ACT)]), c)
    assert torch.equal(mid, neutral_enc().reshape(1, NACT))


# ------------------------------------------------ act-hist: ring vs gather --
def _acts(rng):
    r = torch.rand((N, NACT), generator=rng)
    return (r * torch.tensor(NVEC)).long().clamp_(
        torch.zeros(NACT, dtype=torch.long), torch.tensor(NVEC) - 1)


def _simulate_acts(resets, m=M, seed=0, warm=True):
    """Drive the REAL ActRing in train_fast's rollout order:
    compose (end of the previous decision) -> push the acted tuple ->
    record -> bump with the episode ends of this decision's substeps."""
    rng = torch.Generator().manual_seed(seed)
    ring = ActRing(m, N, "cpu")
    b_ah = torch.zeros((m + T, N, NACT))
    ages = torch.zeros((T, N), dtype=torch.long)
    if warm:                       # a previous iteration, so the prologue is
        for _ in range(m + 3):     # real history rather than zeros
            ring.push(_acts(rng))
            ring.bump(torch.zeros(N, dtype=torch.bool))
    ring.fill_prologue(b_ah)
    seen = []
    for t in range(T):
        seen.append(ring.compose())
        ring.push(_acts(rng))
        ring.record(b_ah, ages, t)
        ring.bump(resets[t])
    return torch.stack(seen), b_ah, ages


def test_the_update_gathers_exactly_what_the_rollout_composed():
    seen, b_ah, ages = _simulate_acts(_resets())
    idx = torch.arange(T * N)
    got = acthist_from_buffer(b_ah.reshape((M + T) * N, NACT), idx,
                              ages.reshape(-1)[idx], M, N)
    want = seen.reshape(T * N, M * NACT)
    assert torch.equal(got, want), \
        f"{int((got != want).any(dim=1).sum())} of {T * N} samples differ"


@pytest.mark.parametrize("m", [1, 2, 4, 15])
def test_the_agreement_holds_for_every_m(m):
    seen, b_ah, ages = _simulate_acts(_resets(), m=m)
    idx = torch.randperm(T * N)[:37]
    got = acthist_from_buffer(b_ah.reshape((m + T) * N, NACT), idx,
                              ages.reshape(-1)[idx], m, N)
    assert torch.equal(got, seen.reshape(T * N, m * NACT)[idx]), f"M={m}"


def test_the_act_parity_test_is_not_vacuous():
    r = _resets()
    assert r.any()
    seen, _, ages = _simulate_acts(r)
    assert (ages == 0).any() and (ages > 0).any()
    s = seen.reshape(T * N, M, NACT)
    assert (s[:, 0] != s[:, -1]).any(), "newest == oldest everywhere"
    # a corrupted age must break the match, or the check cannot fail
    _, b_ah, _ = _simulate_acts(r)
    bad = acthist_from_buffer(b_ah.reshape(-1, NACT), torch.arange(T * N),
                              ages.reshape(-1) + 1, M, N)
    assert not torch.equal(bad, seen.reshape(T * N, M * NACT))


def test_the_prologue_is_read_and_carries_real_history():
    """A t=0 sample of iteration 2 reaches back into the previous
    iteration's actions, not into zeros."""
    seen, b_ah, ages = _simulate_acts(torch.zeros((T, N), dtype=torch.bool))
    assert int(ages[0].max()) == M, "the fixture never fills the window"
    first = seen[0].reshape(N, M, NACT)
    assert torch.equal(first[:, M - 1], b_ah[0]), "oldest slot != prologue row"
    assert float(b_ah[:M].abs().sum()) > 0.0, "prologue is still zeros"


# --------------------------------------------------- neutral / reset rules --
def test_before_the_first_decision_every_slot_is_neutral():
    ring = ActRing(M, N, "cpu")
    got = ring.compose().reshape(N, M, NACT)
    want = neutral_enc().expand(N, M, NACT)
    assert torch.equal(got, want)


def test_a_respawn_collapses_the_whole_history_to_neutral():
    """Not to the oldest real action: after a respawn the agent has not
    acted in THIS episode, and the honest block says so."""
    ring = ActRing(M, N, "cpu")
    rng = torch.Generator().manual_seed(3)
    for _ in range(M + 2):                  # fill the window
        ring.push(_acts(rng))
        ring.bump(torch.zeros(N, dtype=torch.bool))
    assert int(ring.age.min()) == M
    ended = torch.zeros(N, dtype=torch.bool)
    ended[1] = True
    ring.push(_acts(rng))
    ring.bump(ended)
    got = ring.compose().reshape(N, M, NACT)
    assert torch.equal(got[1], neutral_enc().expand(M, NACT))
    assert not torch.equal(got[0], neutral_enc().expand(M, NACT))


def test_the_window_walks_out_one_decision_at_a_time():
    """Age j: slots 1..j are real, j+1..M are neutral - never a clamp onto
    the oldest real entry."""
    ring = ActRing(M, N, "cpu")
    acts = [torch.full((N, NACT), 0, dtype=torch.long) for _ in range(M + 1)]
    for i, a in enumerate(acts):
        a[:, 0] = i                          # yaw bin identifies the decision
    ended = torch.zeros(N, dtype=torch.bool)
    ended[0] = True                          # env 0 respawns at decision 0
    ring.push(acts[0])
    ring.bump(ended)
    assert torch.equal(ring.compose().reshape(N, M, NACT)[0],
                       neutral_enc().expand(M, NACT))
    for j in range(1, M + 1):
        ring.push(acts[j])
        ring.bump(torch.zeros(N, dtype=torch.bool))
        blk = ring.compose().reshape(N, M, NACT)[0]
        yaw = blk[:, 0] * ACT_CENTER[0] + ACT_CENTER[0]
        assert yaw[:j].tolist() == [float(j - s) for s in range(j)], \
            f"age {j}: {yaw.tolist()}"
        assert torch.equal(blk[j:], neutral_enc().expand(M - j, NACT))


# --------------------------------------------------- act-hist: train v eval --
class _FakeCore:
    def __init__(self, n):
        self.states_view = {"tick": np.zeros(n, np.int64)}


def test_the_eval_ring_and_the_rollout_ring_see_the_same_block():
    """The eval helper reads episode boundaries off the core's per-env tick
    counter; the rollout reads them off b_done. Same decisions in, same
    block out, or the recorded policy is not the trained one."""
    core = _FakeCore(N)
    pol = GreedyTorchPolicy(None, None, "cpu", act_every=3, act_hist=M,
                            core=core)
    ring = ActRing(M, N, "cpu")
    rng = torch.Generator().manual_seed(11)
    resets = _resets(seed=7, p=0.15)
    tick = np.zeros(N, np.int64)
    train, evald = [], []
    for t in range(T):
        evald.append(pol._acthist_block(N))
        train.append(ring.compose())
        a = _acts(rng)
        pol._push_act(a)
        ring.push(a)
        ring.bump(resets[t])
        tick = np.where(resets[t].numpy(), 0, tick + 3)
        core.states_view["tick"] = tick.copy()
    assert torch.equal(torch.stack(evald), torch.stack(train))
    assert resets.any(), "no respawns in the fixture"
    flat = torch.stack(train).reshape(-1, M, NACT)
    assert (flat != neutral_enc()).any(), "the whole fixture is neutral"


def test_a_held_tick_does_not_advance_the_history():
    """--act-every 3: three engine ticks, ONE decision. A ring pushed per
    tick would show the same action three times and span 150 ms where the
    flag promises 450."""
    core = _FakeCore(1)
    pol = GreedyTorchPolicy(None, None, "cpu", act_every=3, act_hist=M,
                            core=core)
    calls = {"n": 0}

    def _decide(_obs):
        calls["n"] += 1
        pol._acthist_block(1)
        pol._push_act(torch.zeros(1, NACT, dtype=torch.long))
        return np.zeros((1, NACT), np.int32)

    pol._decide = _decide
    for t in range(9):
        core.states_view["tick"] = np.array([t], np.int64)
        pol.act(np.zeros((1, N_SCALAR), np.float32))
    assert calls["n"] == 3, "the history moved on a held tick"
    # three decisions over nine ticks: the window advanced 3 decisions, not 9
    assert int(pol._aring.age.max()) == 2


# ------------------------------------------------------------ the network ---
W, H = 32, 16
KEYS = ["conv.0.weight", "conv.0.bias", "conv.2.weight", "conv.2.bias",
        "conv.4.weight", "conv.4.bias", "conv.8.weight", "conv.8.bias",
        "pi.0.weight", "pi.0.bias", "pi.2.weight", "pi.2.bias",
        "vf.0.weight", "vf.0.bias", "vf.2.weight", "vf.2.bias",
        "action_head.weight", "action_head.bias",
        "value_head.weight", "value_head.bias"]
AD = 3 * NACT


def _mk(act_dim=0, route_dim=0, critic_only=False, in_ch=1, seed=0):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + route_dim + act_dim + W * H * in_ch, W, H,
                  emb=32, hidden=24, in_ch=in_ch, act_dim=act_dim,
                  route_dim=route_dim, route_critic_only=critic_only).eval()


def test_act_hist_off_is_the_old_model_exactly():
    p = _mk()
    assert list(p.state_dict()) == KEYS and p.act_dim == 0
    assert p.scal_dim == N_SCALAR
    assert tuple(p.state_dict()["pi.0.weight"].shape) == (24, 10 + 32)
    torch.manual_seed(7)
    obs = torch.rand(2, N_SCALAR + W * H)
    with torch.no_grad():
        a, v = p(obs)
        b, w = p.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:])
    assert torch.equal(a, b) and torch.equal(v, w)


def test_the_block_widens_both_towers_and_nothing_else():
    p = _mk(act_dim=AD)
    sd = p.state_dict()
    assert list(sd) == KEYS, "act-hist must not add or rename parameters"
    assert p.scal_dim == N_SCALAR + AD
    base = _mk().state_dict()
    for k in KEYS:
        if k in ("pi.0.weight", "vf.0.weight"):
            assert sd[k].shape[1] == base[k].shape[1] + AD, k
        else:
            assert sd[k].shape == base[k].shape, k


def test_the_block_reaches_BOTH_the_actor_and_the_critic():
    """The one real trap. The route precedent feeds scal[:, N_SCALAR:] to the
    critic and only optionally to the actor; an act-hist the ACTOR cannot see
    cannot damp the actor's own oscillation, which is the mechanism."""
    p = _mk(act_dim=AD, seed=1)
    torch.manual_seed(2)
    obs = torch.rand(4, N_SCALAR + AD + W * H)
    alt = obs.clone()
    alt[:, N_SCALAR:N_SCALAR + AD] += 1.0        # perturb ONLY the block
    with torch.no_grad():
        a0, v0 = p(obs)
        a1, v1 = p(alt)
    assert not torch.allclose(a0, a1), "the ACTOR does not see act-hist"
    assert not torch.allclose(v0, v1), "the CRITIC does not see act-hist"


def test_it_composes_with_a_critic_only_route_block():
    """route-critic-only must keep hiding the ROUTE fan from the actor while
    the act-hist block still reaches it - the two blocks are independent."""
    rd = 5
    p = _mk(act_dim=AD, route_dim=rd, critic_only=True, seed=3)
    torch.manual_seed(4)
    obs = torch.rand(4, N_SCALAR + rd + AD + W * H)
    r_alt, a_alt = obs.clone(), obs.clone()
    r_alt[:, N_SCALAR:N_SCALAR + rd] += 1.0
    a_alt[:, N_SCALAR + rd:N_SCALAR + rd + AD] += 1.0
    with torch.no_grad():
        a0, v0 = p(obs)
        ar, vr = p(r_alt)
        aa, va = p(a_alt)
    assert torch.equal(a0, ar), "the route fan leaked into the actor"
    assert not torch.allclose(v0, vr), "the route fan left the critic"
    assert not torch.allclose(a0, aa) and not torch.allclose(v0, va)


# ------------------------------------------------------------- the surgery --
def test_the_zero_pad_is_a_no_op_on_the_first_forward():
    """The arm's licence to compare against its control: the widened network
    computes the SOURCE function until training moves the zeros.

    Checked to float32 ROUNDING, not bit-for-bit, and the distinction is
    real: the pad is exact in exact arithmetic, but a conv over 4 input
    channels and a Linear over 90 more (zero) columns are different
    reductions, and both cuDNN and cuBLAS reassociate them. Measured on the
    5090 over a full greedy episode: max |dlogit| 7.6e-6 on logits of order
    1e-2, and 0 of 1,696 decisions took a different action."""
    narrow = _mk(seed=5)
    wide = _mk(act_dim=AD, in_ch=4, seed=9)     # different draw on purpose
    ck = {"policy": {k: v.clone() for k, v in narrow.state_dict().items()}}
    n = widen_for_route(ck, wide)
    assert n == 3, f"expected conv.0 + pi.0 + vf.0, padded {n}"
    wide.load_state_dict(ck["policy"])
    torch.manual_seed(6)
    core = torch.rand(3, N_SCALAR)
    frame = torch.rand(3, W * H)
    # newest-first: channel 0 is the current frame, and whatever noise the
    # other three carry must not move the answer
    stack = torch.stack([frame, torch.rand(3, W * H), torch.rand(3, W * H),
                         torch.rand(3, W * H)], dim=-1).reshape(3, -1)
    blk = torch.rand(3, AD) * 4 - 2
    with torch.no_grad():
        a0, v0 = narrow(torch.cat([core, frame], dim=1))
        a1, v1 = wide(torch.cat([core, blk, stack], dim=1))
    assert torch.allclose(a0, a1, rtol=0, atol=1e-6), (a0 - a1).abs().max()
    assert torch.allclose(v0, v1, rtol=0, atol=1e-6), (v0 - v1).abs().max()
    # and the argmax - the only thing a greedy rollout actually consumes
    assert torch.equal(a0.argmax(-1), a1.argmax(-1))


def test_the_conv_pad_puts_the_original_filter_in_channel_zero():
    narrow = _mk(seed=5)
    wide = _mk(in_ch=4, seed=9)
    ck = {"policy": {k: v.clone() for k, v in narrow.state_dict().items()}}
    widen_for_route(ck, wide)
    w = ck["policy"]["conv.0.weight"]
    assert tuple(w.shape) == (16, 4, 5, 5)
    assert torch.equal(w[:, 0:1], narrow.state_dict()["conv.0.weight"])
    assert float(w[:, 1:].abs().sum()) == 0.0


def test_the_2d_pad_is_byte_identical_to_the_route_behaviour():
    """widen_for_route grew a rank case; the 2-D expression must not have
    moved, or every --route resume changes."""
    narrow = _mk(seed=5)
    wide = _mk(route_dim=7, seed=9)
    ck = {"policy": {k: v.clone() for k, v in narrow.state_dict().items()}}
    assert widen_for_route(ck, wide) == 2
    for k in ("pi.0.weight", "vf.0.weight"):
        got = ck["policy"][k]
        old = narrow.state_dict()[k]
        assert torch.equal(got[:, :old.shape[1]], old)
        assert float(got[:, old.shape[1]:].abs().sum()) == 0.0


def test_a_rank_mismatch_is_refused_rather_than_guessed():
    narrow = _mk(seed=5)
    wide = _mk(in_ch=4, seed=9)
    ck = {"policy": {k: v.clone() for k, v in narrow.state_dict().items()}}
    ck["policy"]["conv.0.weight"] = ck["policy"]["conv.0.weight"].reshape(16, 25)
    with pytest.raises(SystemExit, match="conv.0.weight"):
        widen_for_route(ck, wide)
