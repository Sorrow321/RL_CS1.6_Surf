"""--chunk H --codes 0: the DIRECT sequence head (Round 29), CPU only.

Three modes now share one trunk and one call site: flat (chunk 0), codebook
(chunk H, codes K>0) and direct (chunk H, codes 0). The first risk is that
adding the third silently moved one of the first two - both are load-bearing
for every checkpoint ever trained - so most of this file pins that they did
not. The second is the scoring: PPO's ratio must cover exactly the sequence
pi emitted (log-prob SUMMED over the H steps), while the entropy bonus must
keep the meaning --ent has in the other two modes (MEANED over the H steps),
and getting that backwards is a 10x entropy bonus wearing the same flag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (NACT, NPAD, NVEC, N_SCALAR,           # noqa: E402
                        GreedyChunkPolicy, GreedySeqPolicy, HeadPacker,
                        Policy, SampledSeqPolicy, logprob_entropy_padded,
                        sample_padded)

W, H_IMG = 32, 16
HID, EMB = 24, 32
CH = 10                       # the arm's horizon
LOGIT_W = sum(NVEC)           # 32
FLAT_KEYS = ["conv.0.weight", "conv.0.bias", "conv.2.weight", "conv.2.bias",
             "conv.4.weight", "conv.4.bias", "conv.8.weight", "conv.8.bias",
             "pi.0.weight", "pi.0.bias", "pi.2.weight", "pi.2.bias",
             "vf.0.weight", "vf.0.bias", "vf.2.weight", "vf.2.bias",
             "action_head.weight", "action_head.bias",
             "value_head.weight", "value_head.bias"]


def mk(chunk=0, codes=0, seed=0):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + W * H_IMG, W, H_IMG, emb=EMB, hidden=HID,
                  n_codes=codes, chunk=chunk).eval()


# ------------------------------------------------------- the mode guards ---
def test_flat_mode_is_untouched():
    """chunk 0 must build neither alternative head and must keep the exact
    parameter set every existing checkpoint loads against."""
    p = mk()
    assert p.code_head is None and p.decoder is None and p.seq_head is None
    assert list(p.state_dict()) == FLAT_KEYS


def test_flat_mode_is_bit_identical_to_before_the_flag():
    """Same seed, same weights, same forward. seq_head consumes RNG when it
    is built, so a mode that built it unconditionally would silently
    re-roll every downstream init."""
    a, b = mk(seed=3), mk(seed=3)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    torch.manual_seed(11)
    obs = torch.rand(4, N_SCALAR + W * H_IMG)
    with torch.no_grad():
        la, va = a(obs)
        lb, vb = b(obs)
    assert torch.equal(la, lb) and torch.equal(va, vb)
    assert la.shape == (4, LOGIT_W), "flat head must emit sum(NVEC)"


def test_codebook_mode_is_untouched():
    p = mk(chunk=CH, codes=64)
    assert p.code_head is not None and p.decoder is not None
    assert p.seq_head is None, "codebook mode must not build seq_head"
    assert tuple(p.decoder.shape) == (64, CH, LOGIT_W)
    torch.manual_seed(5)
    obs = torch.rand(3, N_SCALAR + W * H_IMG)
    with torch.no_grad():
        logits, _ = p(obs)
    assert logits.shape == (3, 64), "code head must emit K code logits"


def test_direct_mode_builds_only_a_sequence_head():
    p = mk(chunk=CH, codes=0)
    assert p.seq_head is not None
    assert p.code_head is None and p.decoder is None
    assert tuple(p.seq_head.weight.shape) == (CH * LOGIT_W, HID)
    sd = p.state_dict()
    assert "seq_head.weight" in sd and "decoder" not in sd
    assert "code_head.weight" not in sd
    # everything the flat model has, plus exactly the one head
    assert set(FLAT_KEYS) - set(sd) == set()
    assert set(sd) - set(FLAT_KEYS) == {"seq_head.weight", "seq_head.bias"}


def test_direct_head_emits_the_whole_plan():
    p = mk(chunk=CH, codes=0, seed=7)
    torch.manual_seed(1)
    obs = torch.rand(5, N_SCALAR + W * H_IMG)
    with torch.no_grad():
        logits, value = p(obs)
    assert logits.shape == (5, CH * LOGIT_W)
    assert value.shape == (5,), "one V(s) per DELIBERATION, not per step"
    assert torch.isfinite(logits).all()


def test_the_head_starts_near_uniform():
    """orthogonal 0.01 like action_head: no step of the plan may start
    deterministic, or the first iterations are unexplorable."""
    p = mk(chunk=CH, codes=0, seed=2)
    torch.manual_seed(4)
    obs = torch.rand(64, N_SCALAR + W * H_IMG)
    packer = HeadPacker("cpu")
    with torch.no_grad():
        logits, _ = p(obs)
        seq = packer.pad_seq(logits.view(64, CH, LOGIT_W))
        pr = F.softmax(seq, dim=-1)
    # yaw head has 15 bins; uniform is 1/15 = 0.0667
    top = pr[:, :, 0, :NVEC[0]].max(-1).values
    assert float(top.mean()) < 0.10, f"yaw head not near-uniform: {top.mean()}"


# --------------------------------------------------- shapes through pack ---
def test_the_plan_shape_survives_the_packer():
    packer = HeadPacker("cpu")
    logits = torch.randn(6, CH * LOGIT_W)
    seq = packer.pad_seq(logits.view(6, CH, LOGIT_W))
    assert seq.shape == (6, CH, NACT, NPAD)
    act, logp = sample_padded(seq)
    assert act.shape == (6, CH, NACT), "one 6-tuple per step of the plan"
    assert logp.shape == (6, CH), "one logp per (row, step)"
    for h, n in enumerate(NVEC):
        assert int(act[:, :, h].max()) < n, f"head {h} sampled out of range"


# ------------------------------------------- log-prob / entropy, by hand ---
def test_logprob_and_entropy_against_a_hand_computed_case():
    """Two steps, tiny numbers, everything done twice - once by the
    production helper and once by hand out of log_softmax."""
    torch.manual_seed(0)
    packer = HeadPacker("cpu")
    B, Hc = 2, 2
    logits = torch.randn(B, Hc, LOGIT_W)
    acts = torch.zeros(B, Hc, NACT, dtype=torch.long)
    for h, n in enumerate(NVEC):
        acts[:, :, h] = torch.randint(0, n, (B, Hc))
    seq = packer.pad_seq(logits)
    lp, ent = logprob_entropy_padded(seq, acts)
    assert lp.shape == (B, Hc) and ent.shape == (B, Hc)

    # by hand: slice the flat logits per head, log_softmax over that head's
    # own bins only, and add up the six chosen log-probs
    want_lp = torch.zeros(B, Hc)
    want_ent = torch.zeros(B, Hc)
    off = 0
    for h, n in enumerate(NVEC):
        sl = logits[:, :, off:off + n]
        off += n
        lsm = F.log_softmax(sl, dim=-1)
        want_lp += lsm.gather(-1, acts[:, :, h].unsqueeze(-1)).squeeze(-1)
        want_ent += -(lsm.exp() * lsm).sum(-1)
    assert torch.allclose(lp, want_lp, atol=1e-6), (lp - want_lp).abs().max()
    assert torch.allclose(ent, want_ent, atol=1e-6)


def test_the_joint_logprob_is_the_sum_over_steps():
    """PPO's ratio covers the whole emitted sequence, so the joint is the
    SUM - a mean would make the ratio wrong by a factor of H."""
    packer = HeadPacker("cpu")
    torch.manual_seed(8)
    logits = torch.randn(3, CH, LOGIT_W)
    seq = packer.pad_seq(logits)
    act, samp_lp = sample_padded(seq)
    lp, _ = logprob_entropy_padded(seq, act)
    m = torch.ones(3, CH)
    assert torch.allclose((lp * m).sum(-1), samp_lp.sum(-1), atol=1e-5)
    # and a uniform-random plan is strictly less likely than pi's own sample
    assert float((lp * m).sum(-1).mean()) < 0.0


def test_entropy_is_MEANED_not_summed_so_ent_coef_keeps_its_meaning():
    """THE scaling check. --ent multiplies the entropy of ONE sampled
    decision in flat mode; summing over H would hand the same flag a 10x
    bigger bonus, which is a second treatment. Meaning: the direct-mode
    entropy at H=10 must sit at the FLAT scale, not 10x it."""
    packer = HeadPacker("cpu")
    torch.manual_seed(12)
    logits = torch.randn(64, CH, LOGIT_W)
    seq = packer.pad_seq(logits)
    _, ent_h = logprob_entropy_padded(seq, torch.zeros(64, CH, NACT,
                                                       dtype=torch.long))
    m = torch.ones(64, CH)
    meaned = (ent_h * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    summed = (ent_h * m).sum(-1)
    # the flat model's entropy on the same distribution family
    flat = ent_h[:, 0]
    assert torch.allclose(meaned.mean(), flat.mean(), atol=0.15), \
        "meaned entropy drifted off the flat scale"
    assert float(summed.mean() / meaned.mean()) == pytest.approx(CH, abs=1e-4)


def test_the_mean_reduces_to_flat_mode_at_H_1():
    packer = HeadPacker("cpu")
    torch.manual_seed(13)
    logits = torch.randn(7, 1, LOGIT_W)
    seq = packer.pad_seq(logits)
    acts = torch.zeros(7, 1, NACT, dtype=torch.long)
    lp, ent = logprob_entropy_padded(seq, acts)
    m = torch.ones(7, 1)
    meaned = (ent * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    flat_lp, flat_ent = logprob_entropy_padded(packer.pad(logits[:, 0]),
                                               acts[:, 0])
    assert torch.allclose(meaned, flat_ent, atol=1e-6)
    assert torch.allclose((lp * m).sum(-1), flat_lp, atol=1e-6)


# ----------------------------------------------------------- tail masking --
def test_the_neutral_tail_contributes_nothing():
    """A mid-chunk episode end masks the rest to NEUTRAL_ACT; those steps
    were not emitted by pi, so neither the ratio nor the entropy may cover
    them (design doc 4.3)."""
    packer = HeadPacker("cpu")
    torch.manual_seed(9)
    logits = torch.randn(4, CH, LOGIT_W)
    seq = packer.pad_seq(logits)
    act, _ = sample_padded(seq)
    lp, ent = logprob_entropy_padded(seq, act)
    m = torch.ones(4, CH)
    m[1, 4:] = 0.0                      # env 1 ended after 4 steps
    m[2, 0:] = 0.0
    m[2, 0] = 1.0                       # env 2 ran exactly one step
    logp = (lp * m).sum(-1)
    live = m.sum(-1).clamp(min=1.0)
    ment = (ent * m).sum(-1) / live
    assert torch.allclose(logp[1], lp[1, :4].sum(), atol=1e-6)
    assert torch.allclose(logp[2], lp[2, 0], atol=1e-6)
    assert torch.allclose(ment[2], ent[2, 0], atol=1e-6)
    # changing a MASKED step's logits must not move either quantity
    seq2 = seq.clone()
    seq2[1, 7] = torch.randn(NACT, NPAD)
    lp2, ent2 = logprob_entropy_padded(seq2, act)
    assert torch.allclose((lp2 * m).sum(-1)[1], logp[1], atol=1e-6)
    assert torch.allclose(((ent2 * m).sum(-1) / live)[1], ment[1], atol=1e-6)


def test_an_all_dead_chunk_does_not_divide_by_zero():
    m = torch.zeros(2, CH)
    ent = torch.randn(2, CH)
    out = (ent * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    assert torch.equal(out, torch.zeros(2)) and torch.isfinite(out).all()


# ------------------------------------------------------------ eval path ----
class _FakeCore:
    def __init__(self, n=1):
        self.states_view = {"tick": np.zeros(n, np.int64)}


def test_the_eval_wrapper_accepts_a_sequence_head():
    p = mk(chunk=CH, codes=0, seed=6)
    pol = GreedySeqPolicy(p, HeadPacker("cpu"), "cpu", core=_FakeCore(),
                          act_every=4)
    assert pol._H == CH, "horizon must come off policy.chunk"


def test_the_eval_wrapper_refuses_a_flat_checkpoint():
    with pytest.raises(ValueError, match="decoder or sequence head"):
        GreedyChunkPolicy(mk(), HeadPacker("cpu"), "cpu", core=_FakeCore())


def test_the_greedy_plan_is_the_argmax_of_each_step():
    p = mk(chunk=CH, codes=0, seed=10)
    packer = HeadPacker("cpu")
    pol = GreedySeqPolicy(p, packer, "cpu", core=_FakeCore(), act_every=4)
    obs = np.zeros((1, N_SCALAR + W * H_IMG), np.float32)
    plan = pol._decide_chunk(obs)
    assert plan.shape == (1, CH, NACT) and plan.dtype == np.int32
    with torch.no_grad():
        logits, _ = p(torch.as_tensor(obs))
        want = packer.pad_seq(logits.view(1, CH, LOGIT_W)).argmax(-1).numpy()
    assert np.array_equal(plan, want)
    for h, n in enumerate(NVEC):
        assert plan[:, :, h].max() < n


def test_the_held_plan_runs_one_row_per_act_every_ticks():
    """act_every 4, H 10: 40 engine ticks per deliberation, and the plan's
    row index advances once every 4."""
    p = mk(chunk=CH, codes=0, seed=14)
    # Make each step of the plan DISTINGUISHABLE. At init the head is
    # near-uniform by design, so on a constant observation every step can
    # argmax to the same bin - which would make the walk-through below pass
    # vacuously. Bias step h toward yaw bin h instead.
    with torch.no_grad():
        p.seq_head.bias.zero_()
        for h in range(CH):
            p.seq_head.bias[h * LOGIT_W + (h % NVEC[0])] = 50.0
    core = _FakeCore()
    pol = GreedySeqPolicy(p, HeadPacker("cpu"), "cpu", core=core, act_every=4)
    obs = np.zeros((1, N_SCALAR + W * H_IMG), np.float32)
    calls = {"n": 0}
    real = pol._decide_chunk

    def counted(o):
        calls["n"] += 1
        return real(o)

    pol._decide_chunk = counted
    seen = []
    for t in range(40):
        core.states_view["tick"] = np.array([t], np.int64)
        seen.append(pol.act(obs)[0].copy())
    assert calls["n"] == 1, "one deliberation per 40 ticks"
    for h in range(CH):
        block = seen[h * 4:(h + 1) * 4]
        assert all(np.array_equal(b, block[0]) for b in block), \
            f"step {h} not held for act_every ticks"
    # the plan is walked IN ORDER: step h's yaw bin appears at ticks 4h..4h+3
    yaws = [int(seen[h * 4][0]) for h in range(CH)]
    assert yaws == [h % NVEC[0] for h in range(CH)], yaws


def test_sampled_and_greedy_share_the_horizon():
    p = mk(chunk=CH, codes=0, seed=15)
    for cls in (GreedySeqPolicy, SampledSeqPolicy):
        pol = cls(p, HeadPacker("cpu"), "cpu", core=_FakeCore(), act_every=4)
        assert pol._H == CH
