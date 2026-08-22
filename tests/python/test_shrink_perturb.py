"""Soft shrink+perturb (Juliani & Ash, NeurIPS 2024, arXiv 2405.19153).

The paper, Appendix A, verbatim: "When the intervention is applied all
learnable parameters in the network are iterated through and scaled by alpha.
All parameters are then additively combined with newly sampled initialization
parameters which are scaled by beta", with "For all experiments alpha = 1 -
beta"; the *soft* variant is the one "applied after each step of gradient
descent instead of only at specific intervals". Table 1 pins beta = 1e-6.
The reference implementation (github.com/awjuliani/deep-rl-plasticity,
shared/modules.py::sp_module) is two lines:

    current_param.data *= shrink_factor
    current_param.data += epsilon * init_params[idx].data

driven by ``adapt_info: ['soft-sp', [[True, True, True], 0.999999, 0.000001]]``
and called from algos/ppo/trainer.py immediately after ``optimizer.step()``.

What is asserted here:
  * beta = 0 moves nothing, bit-for-bit (the control path stays the control);
  * beta = 1 replaces the parameter with a fresh draw from the initialisation
    distribution, and NOT with the value it had;
  * the interpolation is exact for an arbitrary beta, against a donor whose
    draw is pinned by seeding;
  * x_init is RE-DRAWN every call (a frozen donor would be regenerative
    regularisation, a different and losing row of the paper's table);
  * every learnable parameter is covered - weights and biases alike;
  * Adam's moments are untouched, as in the reference;
  * a warm resume is function-identical at step 0: shrink+perturb only acts
    after an optimizer step, so the resumed policy's outputs are unchanged
    until the first update.

CPU only, tiny shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (N_SCALAR, NVEC, Policy,             # noqa: E402
                        ShrinkPerturb, layer_norms)

# the paper's constant, so a typo in the trainer's default cannot pass
PAPER_BETA = 1e-6


def _tiny(**kw):
    """A Policy small enough for CPU, same architecture as the real one."""
    w, h = 8, 4
    return Policy(N_SCALAR + w * h, lidar_w=w, lidar_h=h, emb=16, hidden=12,
                  **kw)


def _snapshot(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def _pinned_draw(seed):
    """The exact parameters ShrinkPerturb.step() will pull toward if the RNG
    is at `seed` when it runs. Seeding immediately before each
    reset_parameters_() call is what makes the two streams line up - a fresh
    Policy() ALSO burns RNG in its module constructors, so comparing against
    `Policy(...)` under the same seed would be comparing different streams."""
    other = _tiny()
    torch.manual_seed(seed)
    other.reset_parameters_()
    return _snapshot(other)


# --------------------------------------------------------------------------
# the update itself
# --------------------------------------------------------------------------

def test_beta_zero_moves_nothing_bitwise():
    torch.manual_seed(0)
    p, donor = _tiny(), _tiny()
    before = _snapshot(p)
    ShrinkPerturb(p, donor, 0.0).step()
    for k, v in p.named_parameters():
        assert torch.equal(v, before[k]), k


def test_beta_one_is_a_fresh_draw_not_the_old_value():
    torch.manual_seed(1)
    p, donor = _tiny(), _tiny()
    before = _snapshot(p)
    expect = _pinned_draw(1234)           # what the donor will draw at seed 1234
    torch.manual_seed(1234)
    ShrinkPerturb(p, donor, 1.0).step()
    moved = 0
    for k, v in p.named_parameters():
        assert torch.equal(v.detach(), expect[k]), k
        if before[k].numel() and not torch.equal(before[k], expect[k]):
            moved += 1
    assert moved >= 6, "beta=1 must have replaced the trained weights"


@pytest.mark.parametrize("beta", [1e-6, 1e-3, 0.1, 0.5])
def test_interpolation_is_exact(beta):
    torch.manual_seed(2)
    p, donor = _tiny(), _tiny()
    # give the policy non-init values so shrink is observable
    with torch.no_grad():
        for v in p.parameters():
            v.add_(torch.randn_like(v))
    before = _snapshot(p)
    draw = _pinned_draw(99)
    torch.manual_seed(99)
    ShrinkPerturb(p, donor, beta).step()
    for k, v in p.named_parameters():
        want = before[k] * (1.0 - beta) + beta * draw[k]
        assert torch.allclose(v, want, atol=1e-7, rtol=1e-5), k


def test_alpha_is_one_minus_beta():
    torch.manual_seed(3)
    sp = ShrinkPerturb(_tiny(), _tiny(), PAPER_BETA)
    assert sp.alpha == 1.0 - PAPER_BETA          # the paper's alpha = 1 - beta
    assert sp.alpha == pytest.approx(0.999999, abs=0.0)   # reference yaml


def test_x_init_is_redrawn_every_call():
    """A frozen donor is a different method (regenerative regularisation)."""
    torch.manual_seed(4)
    p, donor = _tiny(), _tiny()
    sp = ShrinkPerturb(p, donor, 1.0)     # beta=1 makes the draw observable
    sp.step()
    first = _snapshot(p)
    sp.step()
    second = _snapshot(p)
    differing = sum(1 for k in first
                    if first[k].numel() and not torch.equal(first[k], second[k]))
    assert differing >= 6, "the donor was not re-initialised between steps"


def test_covers_every_learnable_parameter_including_biases():
    """"all learnable parameters in the network are iterated through" - the
    reference iterates module.parameters(), so biases are in scope too."""
    torch.manual_seed(5)
    p, donor = _tiny(), _tiny()
    sp = ShrinkPerturb(p, donor, 0.5)
    assert len(sp.pairs) == len(list(p.named_parameters()))
    assert sp.n_param == sum(v.numel() for v in p.parameters())
    # every pair's first element must BE a live parameter of the policy
    live = {id(v) for v in p.parameters()}
    assert {id(a) for a, _ in sp.pairs} == live
    # bias init is zeros_, so from an all-ones policy every bias must land on
    # 0.5*1 + 0.5*0 = 0.5 exactly. If biases were skipped they would stay 1.
    with torch.no_grad():
        for v in p.parameters():
            v.fill_(1.0)
    sp.step()
    biases = [(k, v) for k, v in p.named_parameters() if k.endswith("bias")]
    assert biases
    for k, v in biases:
        assert torch.allclose(v, torch.full_like(v, 0.5)), k


def test_donor_mismatch_is_refused():
    torch.manual_seed(6)
    p = _tiny()
    bad = _tiny(n_codes=4, chunk=2)          # extra code_head/decoder params
    with pytest.raises(SystemExit):
        ShrinkPerturb(p, bad, PAPER_BETA)


def test_chunked_policy_is_also_covered():
    torch.manual_seed(7)
    p, donor = _tiny(n_codes=4, chunk=2), _tiny(n_codes=4, chunk=2)
    assert "decoder" in dict(p.named_parameters())
    before = _snapshot(p)
    ShrinkPerturb(p, donor, 1.0).step()
    assert not torch.equal(p.decoder.detach(), before["decoder"])


# --------------------------------------------------------------------------
# it must not disturb anything else
# --------------------------------------------------------------------------

def test_adam_moments_are_untouched():
    """The reference writes .data only; Adam keeps its state across the pull."""
    torch.manual_seed(8)
    p, donor = _tiny(), _tiny()
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    obs = torch.randn(4, N_SCALAR + 8 * 4)
    logits, val = p(obs)
    (logits.square().mean() + val.square().mean()).backward()
    opt.step()
    assert opt.state, "no Adam state to check"
    moments = {id(k): (v["exp_avg"].clone(), v["exp_avg_sq"].clone())
               for k, v in opt.state.items()}
    ShrinkPerturb(p, donor, PAPER_BETA).step()
    for k, v in opt.state.items():
        m0, s0 = moments[id(k)]
        assert torch.equal(v["exp_avg"], m0)
        assert torch.equal(v["exp_avg_sq"], s0)


def test_warm_resume_is_function_identical_at_step_zero():
    """Nothing happens until an optimizer step happens, so a resumed arm
    starts exactly on the control's curve and any divergence is treatment."""
    torch.manual_seed(9)
    ref, arm, donor = _tiny(), _tiny(), _tiny()
    arm.load_state_dict(ref.state_dict())     # the shared warm resume
    ShrinkPerturb(arm, donor, PAPER_BETA)     # constructing it must be inert
    obs = torch.randn(8, N_SCALAR + 8 * 4)
    with torch.no_grad():
        la, va = arm(obs)
        lr_, vr = ref(obs)
    assert torch.equal(la, lr_) and torch.equal(va, vr)


def test_one_paper_step_is_a_1e6_relative_nudge():
    """Sanity on the magnitude: beta=1e-6 must be a nudge, not a reset - but
    it must not be a no-op in fp32 either, or the arm is measuring nothing."""
    torch.manual_seed(10)
    p, donor = _tiny(), _tiny()
    with torch.no_grad():
        for v in p.parameters():
            v.add_(torch.randn_like(v))
    before = _snapshot(p)
    ShrinkPerturb(p, donor, PAPER_BETA).step()
    w = p.conv[0].weight.detach()
    w0 = before["conv.0.weight"]
    rel = float((w - w0).norm() / w0.norm())
    assert 0.0 < rel < 1e-4, rel


# --------------------------------------------------------------------------
# re-initialisation plumbing
# --------------------------------------------------------------------------

def _assert_is_an_init_draw(p):
    """Every property of Policy's initialisation distribution: semi-orthogonal
    weights at the layer's gain, zero biases. This is what "sampled from the
    parameter initialization distribution" has to mean for THIS network."""
    gains = [(m, np.sqrt(2.0)) for m in list(p.conv) + list(p.pi) + list(p.vf)
             if isinstance(m, (nn.Linear, nn.Conv2d))]
    gains += [(p.action_head, 0.01), (p.value_head, 1.0)]
    for m, g in gains:
        w = m.weight.detach().reshape(m.weight.shape[0], -1)
        gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
        eye = torch.eye(gram.shape[0]) * (g * g)
        assert torch.allclose(gram, eye, atol=1e-4 * max(g * g, 1e-3)), m
        assert torch.count_nonzero(m.bias.detach()) == 0, m


def test_reset_parameters_redraws_the_same_distribution():
    """reset_parameters_() must produce the SAME distribution a freshly built
    Policy does - that is what makes the donor a faithful stand-in for the
    reference's per-call fresh nn.Module. (It cannot be bit-compared against
    Policy(...) under one seed: a constructor also burns RNG in nn.Linear's
    own default init before the orthogonal pass overwrites it.)"""
    torch.manual_seed(11)
    fresh = _tiny()
    _assert_is_an_init_draw(fresh)             # the constructor's output
    reused = _tiny()
    with torch.no_grad():
        for v in reused.parameters():
            v.mul_(0.0).add_(7.0)
    reused.reset_parameters_()
    _assert_is_an_init_draw(reused)            # and the re-draw's
    # and it is a DRAW, not a copy of some fixed table
    a = _pinned_draw(1010)
    b = _pinned_draw(2020)
    assert not torch.equal(a["conv.0.weight"], b["conv.0.weight"])
    assert torch.equal(_pinned_draw(1010)["conv.0.weight"], a["conv.0.weight"])


def test_reset_survives_channels_last_conv():
    """Policy holds conv channels_last; nn.init.orthogonal_ ends in view_as(),
    which throws on a non-contiguous weight. Regression guard."""
    torch.manual_seed(12)
    p = _tiny()
    assert p.conv[2].weight.is_contiguous(memory_format=torch.channels_last)
    p.reset_parameters_()                      # must not raise
    assert torch.isfinite(p.conv[2].weight).all()


# --------------------------------------------------------------------------
# the free diagnostic (Lyle et al. 2024, arXiv 2407.01800)
# --------------------------------------------------------------------------

def test_layer_norms_are_l2_per_parameter_plus_total():
    torch.manual_seed(13)
    p = _tiny()
    names, vals = layer_norms(p)
    assert names == [n for n, _ in p.named_parameters()]
    assert len(vals) == len(names) + 1
    for n, v in zip(names, vals):
        want = float(dict(p.named_parameters())[n].detach().float().norm())
        assert v == pytest.approx(want, rel=1e-6)
    total = float(torch.cat([v.detach().flatten() for v in p.parameters()]).norm())
    assert vals[-1] == pytest.approx(total, rel=1e-6)


def test_layer_norms_do_not_move_weights_or_rng():
    torch.manual_seed(14)
    p = _tiny()
    before = _snapshot(p)
    state = torch.random.get_rng_state()
    layer_norms(p)
    assert torch.equal(state, torch.random.get_rng_state())
    for k, v in p.named_parameters():
        assert torch.equal(v, before[k])
    assert all(v.grad is None for v in p.parameters())


def test_norm_growth_is_visible_under_plain_training():
    """The measurement has to be able to SHOW norm growth, else logging it
    proves nothing. Train a tiny net on a fixed target and watch it grow."""
    torch.manual_seed(15)
    p = _tiny()
    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    obs = torch.randn(16, N_SCALAR + 8 * 4)
    tgt = torch.randn(16, sum(NVEC))
    n0 = layer_norms(p)[1][-1]
    for _ in range(60):
        logits, _ = p(obs)
        opt.zero_grad()
        ((logits - tgt).square().mean()).backward()
        opt.step()
    n1 = layer_norms(p)[1][-1]
    assert n1 > n0
