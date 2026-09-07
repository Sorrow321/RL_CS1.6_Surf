"""``--critic-warmup N``: hold the ACTOR while the critic re-fits a new reward.

Round 28 (addenda 5-7, xsFANARC) measured the failure this exists for:
switching the reward on a warm checkpoint destroys the learned behaviour.
The mechanism is not mysterious - ``V(s)`` is suddenly the value function of
a DIFFERENT objective, so PPO's advantages are noise, and the first updates
point that noise straight at a policy that already works. The remedy is to
let the critic catch up first: collect rollouts with the resumed policy,
optimise only the value loss, then unfreeze.

Actor and critic SHARE the conv trunk here, so "freeze the actor" has to
mean the trunk too - otherwise the warmup is still moving the features every
head reads and the policy it protects is not the one that comes out.

The load-bearing claim is that the freeze is EXACT, not approximate:

  * a parameter whose ``.grad`` is None takes no Adam step, gains no moment
    and no weight decay - so **every actor-side tensor is bit-identical**
    across the warmup, not merely close;
  * zeroing the gradients instead would NOT do that: Adam's momentum keeps
    moving a parameter for many steps after its gradient hits zero, which is
    exactly the silent version of this bug;
  * the critic does move, and it moves under the same ``vf`` weighting it
    will have after the warmup, so its effective step size does not jump
    when the actor is let go.

    python -m pytest tests/python/test_critic_warmup.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import N_SCALAR, Policy          # noqa: E402

LW, LH = 16, 8
IMG = LW * LH
# the trainer's split, verbatim: everything else is actor or shared trunk
CRITIC_PREFIX = ("vf.", "value_head.", "priv_mlp.")


def _policy(seed=0, **kw):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + IMG, LW, LH, emb=16, hidden=12, **kw)


def _split(policy):
    frozen, trained = [], []
    for n, p in policy.named_parameters():
        (trained if n.startswith(CRITIC_PREFIX) else frozen).append((n, p))
    return frozen, trained


def _update(policy, opt, obs, ret, vf=0.5, warming=False, frozen=()):
    """One minibatch, exactly the three lines the trainer runs: backward,
    drop the frozen gradients BEFORE the clip, clip, step."""
    logits, value = policy(obs)
    vl = 0.5 * (value.reshape(-1) - ret).pow(2).mean()
    if warming:
        loss = vf * vl
    else:
        loss = logits.square().mean() + vf * vl      # a stand-in for pg+ent
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if warming:
        for _, q in frozen:
            q.grad = None
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    opt.step()
    return float(vl.detach())


# --------------------------------------------------------------------------
# 1. the freeze is exact
# --------------------------------------------------------------------------
def test_every_actor_side_parameter_is_bit_identical_after_the_warmup():
    p = _policy(seed=3)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    frozen, trained = _split(p)
    assert frozen and trained
    before = {n: q.detach().clone() for n, q in frozen}
    v_before = {n: q.detach().clone() for n, q in trained}

    torch.manual_seed(7)
    obs = torch.randn(8, N_SCALAR + IMG)
    ret = torch.randn(8)
    for _ in range(12):
        _update(p, opt, obs, ret, warming=True, frozen=frozen)

    for n, q in frozen:
        assert torch.equal(q.detach(), before[n]), \
            f"{n} moved during the critic warmup"
    moved = [n for n, q in trained if not torch.equal(q.detach(), v_before[n])]
    assert moved, "the critic did not move at all during its own warmup"


def test_the_shared_conv_trunk_is_frozen_too():
    """Actor and critic share the trunk, so an unfrozen trunk would be the
    warmup moving the features the policy reads - which is the thing it
    exists to protect."""
    p = _policy(seed=5)
    frozen, _ = _split(p)
    names = {n for n, _ in frozen}
    assert any(n.startswith("conv.") for n in names)
    assert any(n.startswith("pi.") for n in names)
    assert any(n.startswith("action_head") for n in names)
    assert not any(n.startswith(CRITIC_PREFIX) for n in names)


def test_adam_creates_no_state_for_a_parameter_with_no_grad():
    """The fact the whole implementation rests on: torch.optim.Adam SKIPS a
    parameter whose .grad is None - no step, no exp_avg, no weight decay."""
    p = _policy(seed=11)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3, weight_decay=0.01)
    frozen, _ = _split(p)
    torch.manual_seed(1)
    obs, ret = torch.randn(6, N_SCALAR + IMG), torch.randn(6)
    for _ in range(5):
        _update(p, opt, obs, ret, warming=True, frozen=frozen)
    for _, q in frozen:
        assert q not in opt.state or not opt.state[q], \
            "Adam kept state for a frozen parameter - it will step it later"


def test_zeroing_the_gradient_would_NOT_have_frozen_it():
    """Why the implementation drops the grad instead of zeroing it: Adam's
    momentum keeps moving a parameter for many steps after its gradient hits
    zero, and that is the silent version of this bug."""
    p = _policy(seed=13)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    frozen, _ = _split(p)
    torch.manual_seed(2)
    obs, ret = torch.randn(6, N_SCALAR + IMG), torch.randn(6)
    _update(p, opt, obs, ret)                      # one normal step: moments
    ref = {n: q.detach().clone() for n, q in frozen}
    for _ in range(5):                             # now ZERO them instead
        logits, value = p(obs)
        loss = logits.square().mean() + 0.5 * 0.5 * (
            value.reshape(-1) - ret).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for _, q in frozen:
            q.grad = torch.zeros_like(q)
        opt.step()
    drifted = [n for n, q in frozen if not torch.equal(q.detach(), ref[n])]
    assert drifted, ("zero gradients left every actor parameter untouched - "
                     "if that is ever true, this test is measuring nothing")


# --------------------------------------------------------------------------
# 2. and it lets go
# --------------------------------------------------------------------------
def test_the_actor_moves_again_once_the_warmup_ends():
    p = _policy(seed=17)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    frozen, _ = _split(p)
    torch.manual_seed(4)
    obs, ret = torch.randn(8, N_SCALAR + IMG), torch.randn(8)
    for _ in range(6):
        _update(p, opt, obs, ret, warming=True, frozen=frozen)
    held = {n: q.detach().clone() for n, q in frozen}
    for _ in range(3):
        _update(p, opt, obs, ret, warming=False)
    moved = [n for n, q in frozen if not torch.equal(q.detach(), held[n])]
    assert moved, "the actor stayed frozen after the warmup ended"


def test_the_value_loss_falls_while_the_actor_is_held():
    """The warmup's own diagnostic (`warmup/value_loss`): the critic has to
    be able to fit the new returns on the frozen features."""
    p = _policy(seed=19)
    opt = torch.optim.Adam(p.parameters(), lr=3e-3)
    frozen, _ = _split(p)
    torch.manual_seed(6)
    obs, ret = torch.randn(16, N_SCALAR + IMG), torch.randn(16) * 5.0
    first = _update(p, opt, obs, ret, warming=True, frozen=frozen)
    for _ in range(40):
        last = _update(p, opt, obs, ret, warming=True, frozen=frozen)
    assert last < first, (first, last)


# --------------------------------------------------------------------------
# 3. the flag surface
# --------------------------------------------------------------------------
def test_the_flag_exists_defaults_off_and_is_recorded():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--critic-warmup"' in src
    assert "args.critic_warmup = 0" in src                       # default off
    assert '"critic_warmup": int(args.critic_warmup or 0)' in src
    assert 'CRITIC_PREFIX = ("vf.", "value_head.", "priv_mlp.")' in src
    assert "warming = CW > 0 and it_no <= CW" in src
    # the grad drop sits BEFORE the clip and before the step
    i_drop = src.index("for _q in warm_frozen:")
    i_clip = src.index("nn.utils.clip_grad_norm_(policy.parameters(), 0.5)")
    i_step = src.index("opt.step()", i_clip)
    assert i_drop < i_clip < i_step
    assert "loss = args.vf * vl" in src            # value term only
    assert "warmup/value_loss" in src              # logged per update
    assert "--critic-warmup: DONE after" in src    # and a line when it ends


def test_it_refuses_the_cases_it_cannot_be_correct_in():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    # from scratch there is no learned behaviour to protect
    assert "--critic-warmup is the procedure for a REWARD" in src
    assert "if not args.ckpt:" in src
    # under DDP sync_grads() all-reduces p.grad for every parameter
    assert "--critic-warmup under DDP is refused" in src


def test_the_launcher_does_not_bake_the_procedure_in():
    sh = (ROOT / "tools" / "run_arm.sh").read_text(encoding="utf-8")
    assert "critic_warmup" not in sh and "critic-warmup" not in sh
