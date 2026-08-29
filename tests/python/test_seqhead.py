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


# ------------------------------------------- --seq-ratio per-step (Part D) --
# The per-step surrogate is meant to BE flat PPO over the expanded decisions,
# with the advantage shared inside a chunk and the trunk forward shared. The
# strongest possible statement of that is an invariance: at H=1 there is
# nothing to share, so it must reduce to the flat expression exactly. These
# reproduce mb_step's two branches on tensors rather than importing them,
# because mb_step closes over the trainer's argv.
CLIP = 0.2


def _pg_joint(a, ratio):
    return torch.max(-a * ratio,
                     -a * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()


def _pg_perstep(a, r_h, m):
    a_h = a.unsqueeze(-1)
    return (torch.max(-a_h * r_h,
                      -a_h * torch.clamp(r_h, 1 - CLIP, 1 + CLIP))
            * m).sum(-1).mean()


def test_per_step_at_H1_is_BIT_IDENTICAL_to_flat_ppo():
    """THE invariance guard. At --chunk 1 a chunk holds one decision, the
    shared advantage is that decision's own, and the sum over steps is a
    sum of one - so the per-step surrogate must be the flat expression, to
    the bit, or it is not 'flat PPO over the expanded decisions'."""
    torch.manual_seed(21)
    n = 512
    a = torch.randn(n)
    lp_new = torch.randn(n) * 0.3
    lp_old = torch.randn(n) * 0.3
    ratio = torch.exp(lp_new - lp_old)
    flat = _pg_joint(a, ratio)
    per = _pg_perstep(a, ratio.unsqueeze(-1), torch.ones(n, 1))
    assert torch.equal(flat, per), (flat - per).abs().max()


def test_per_step_is_not_the_joint_at_H_10():
    """Non-vacuity: the two branches must actually differ at H>1, or the
    invariance above proves nothing about the arm."""
    torch.manual_seed(22)
    n, Hc = 256, CH
    a = torch.randn(n)
    d = torch.randn(n, Hc) * 0.4
    m = torch.ones(n, Hc)
    r_h = torch.exp(d)
    r_joint = torch.exp(d.sum(-1))
    assert not torch.allclose(_pg_joint(a, r_joint), _pg_perstep(a, r_h, m))


def test_the_joint_ratio_hides_a_step_that_blows_up():
    """Why the arm exists. One step moving 4x while another compensates
    leaves the JOINT ratio at 1.0 - unclipped, invisible - while the
    per-step form clips both."""
    a = torch.tensor([1.0])
    d = torch.zeros(1, 2)
    d[0, 0] = 1.4          # e^1.4 ~ 4.05
    d[0, 1] = -1.4         # e^-1.4 ~ 0.247, product exactly 1
    m = torch.ones(1, 2)
    r_joint = torch.exp(d.sum(-1))
    assert torch.allclose(r_joint, torch.ones(1), atol=1e-6)
    # joint: ratio 1.0 is inside the clip band, so the surrogate is -A
    assert torch.allclose(_pg_joint(a, r_joint), torch.tensor(-1.0), atol=1e-6)
    # per-step: BOTH steps are outside [0.8, 1.2] and get clipped
    r_h = torch.exp(d)
    assert float(r_h[0, 0]) > 1 + CLIP and float(r_h[0, 1]) < 1 - CLIP
    per = _pg_perstep(a, r_h, m)
    assert float(per) > -2.0, "per-step failed to clip the blown-up step"


def test_the_sum_form_OVERWEIGHTS_each_decision_by_H():
    """The normalization, named for what it actually does.

    This test was originally called "matches flat gradient scale per env
    step" and asserted this same ratio of H - which is the DISCREPANCY, not
    a match. Round 29's xSEQ10PS ran with it: summing over the H steps and
    then meaning over ROWS matches the term COUNT per env (13 x 10 = 130
    against flat's 128 x 1) but divides by rows, so every decision entered
    the policy loss with H times the weight flat PPO gives it - 9.85x at the
    real minibatch shapes - while value_loss and the meaned entropy stayed
    per-deliberation. That reweighting is a second treatment and it
    invalidated the arm.

    Kept as a REGRESSION PIN on the known-wrong behavior so the fix is
    visible when it lands: dividing by H (or meaning over terms, `(x *
    m).sum() / m.sum()`) turns the ratio below into 1.0."""
    torch.manual_seed(23)
    dec = 1280                       # decisions in the comparison
    Hc = 10
    a_ch = torch.randn(dec // Hc)
    d_flat = torch.randn(dec) * 0.2
    # the advantage is SHARED inside a chunk, so the matched flat run is one
    # where every decision of a chunk carries that chunk's advantage
    a_flat = a_ch.repeat_interleave(Hc)
    flat = _pg_joint(a_flat, torch.exp(d_flat))
    per = _pg_perstep(a_ch, torch.exp(d_flat.view(-1, Hc)),
                      torch.ones(dec // Hc, Hc))
    # a row here sums H terms and there are H times fewer rows, so each
    # decision carries H times flat's weight. NOT a match - the defect.
    assert float(per / flat) == pytest.approx(Hc, rel=1e-5)
    # and this is what the corrected form gives
    fixed = _pg_perstep(a_ch, torch.exp(d_flat.view(-1, Hc)),
                        torch.ones(dec // Hc, Hc)) / Hc
    assert float(fixed / flat) == pytest.approx(1.0, rel=1e-5)


def test_masked_steps_contribute_nothing_to_the_per_step_surrogate():
    torch.manual_seed(24)
    n, Hc = 64, CH
    a = torch.randn(n)
    r_h = torch.exp(torch.randn(n, Hc) * 0.3)
    m = torch.ones(n, Hc)
    m[3, 5:] = 0.0
    base = _pg_perstep(a, r_h, m)
    r2 = r_h.clone()
    r2[3, 5:] = 99.0                 # nonsense in the dead tail
    assert torch.allclose(_pg_perstep(a, r2, m), base, atol=1e-6)


def test_per_step_kl_is_a_per_decision_quantity():
    """The reported kl must be comparable to a flat run's column, i.e. a
    mean over DECISIONS - not a sum over the 10 steps of a chunk."""
    torch.manual_seed(25)
    n, Hc = 128, CH
    old = torch.randn(n, Hc) * 0.2
    new = old + 0.05                 # a uniform 0.05 nats per step
    m = torch.ones(n, Hc)
    kl = ((old - new) * m).sum() / m.sum().clamp(min=1.0)
    assert float(kl) == pytest.approx(-0.05, abs=1e-6)
    # a masked chunk does not dilute it
    m[0, 3:] = 0.0
    kl2 = ((old - new) * m).sum() / m.sum().clamp(min=1.0)
    assert float(kl2) == pytest.approx(-0.05, abs=1e-6)
