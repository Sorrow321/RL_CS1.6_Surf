"""``--aug-drop`` / ``--aug-noise`` / ``--aug-shift`` / ``--aug-coef``: image
augmentation in the PPO update, as DrAC regularisation.

The user (2026-10-16): the policy memorises every point of view of its map
and never learns what a ramp IS - "select some portion of pixels ... randomly
and just paint them black ... a little bit of noise ... shift ... the whole
image and on the left, where we have empty space now, just some padding".

**Where the augmentation goes - DrAC (Raileanu et al., "Automatic Data
Augmentation for Generalization in Reinforcement Learning", NeurIPS 2021).**
Feeding augmented frames straight into PPO's loss (RAD) scores the stored
action under pi(a | aug(s)) against a log-prob the rollout took under
pi(a | s), so the ratio is no longer an importance ratio; DrAC shows that
hurts PPO on several Procgen games and fixes it by leaving PPO on the CLEAN
frames and adding two regularisers that make the augmented frame give the
same answer:

    G_pi = KL( pi(. | s)  ||  pi(. | aug(s)) )      pi(. | s) a fixed target
    G_V  = ( V(aug(s)) - V(s) )^2                   V(s) a fixed target
    loss = L_PPO + coef * (G_pi + G_V)              (the paper's alpha_r = 0.1)

The rollout acts on clean frames, so the behaviour that collects the data is
the trained policy itself; only the network's sensitivity to the corruption
is trained away.

**The corruptions** (each optional, all on one draw per row):

* ``shift`` - a random integer offset in ``[-shift, shift]`` on each axis,
  the frame's edge pixels replicated into the gap (DrQ's pad + random crop).
  On the 64-wide, 120-degree camera one pixel is 1.9 degrees of yaw.
* ``noise`` - ``N(0, noise^2)`` added to every channel.
* ``drop`` - each PIXEL (all of its channels at once) set to 0 with
  probability ``drop``: "paint them black".

**The Gaussian heads.** The view's log sigma is one state-independent
parameter, so the KL between the two view Gaussians is
``(mu_s - mu_aug)^2 / (2 sigma^2)``. sigma is DETACHED there: the term is
about the means agreeing, and a live sigma would let the regulariser shrink
it by inflating the policy's noise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["augment_image", "drac_terms", "describe"]


def augment_image(img, h: int, w: int, c: int, drop: float = 0.0,
                  noise: float = 0.0, shift: int = 0):
    """``img`` (B, h*w*c) channel-fastest rows -> an augmented copy of the
    same shape and dtype (global torch RNG; one independent draw per row)."""
    b = img.shape[0]
    x = img.reshape(b, h, w, c)
    if shift > 0:
        dev = x.device
        dy = torch.randint(-shift, shift + 1, (b, 1), device=dev)
        dx = torch.randint(-shift, shift + 1, (b, 1), device=dev)
        ri = (torch.arange(h, device=dev).unsqueeze(0) + dy).clamp(0, h - 1)
        ci = (torch.arange(w, device=dev).unsqueeze(0) + dx).clamp(0, w - 1)
        bi = torch.arange(b, device=dev).view(b, 1, 1)
        x = x[bi, ri.unsqueeze(2), ci.unsqueeze(1)]
    if noise > 0.0:
        x = x + noise * torch.randn_like(x)
    if drop > 0.0:
        keep = torch.rand((b, h, w, 1), device=x.device) >= drop
        x = x * keep.to(x.dtype)
    return x.reshape(b, h * w * c)


def drac_terms(padded_s, padded_a, value_s, value_a, mu_s=None, mu_a=None,
               log_std=None):
    """DrAC's two regularisers, per-row means -> (kl, v_mse), both scalars.

    ``padded_*`` (B, heads, bins) categorical logits padded with a -1e30
    slot value (HeadPacker.pad; pass only the heads the policy acts with);
    ``value_*`` (B,) or (B, k); ``mu_*`` (B, n_z) view means and ``log_std``
    (n_z,) under --view-continuous, else None. The ``_s`` (clean) side is the
    target: detached here, so the gradient reaches the augmented side only.
    """
    ls = F.log_softmax(padded_s.detach().float(), dim=-1)
    la = F.log_softmax(padded_a.float(), dim=-1)
    # the -1e30 pad slots: p = 0 and the two log-probs are the same -1e30, so
    # the product is exactly 0 - no mask needed, no NaN possible
    kl = (ls.exp() * (ls - la)).sum(-1).sum(-1)
    if mu_s is not None:
        var = (2.0 * log_std.detach().float()).exp()
        kl = kl + ((mu_s.detach().float() - mu_a.float()).pow(2)
                   / (2.0 * var)).sum(-1)
    vm = (value_a.float() - value_s.detach().float()).pow(2)
    return kl.mean(), vm.mean()


def describe(drop: float, noise: float, shift: int, coef: float) -> str:
    parts = []
    if shift > 0:
        parts.append(f"shift +-{shift} px (edge-replicated)")
    if noise > 0.0:
        parts.append(f"noise sigma {noise:g}")
    if drop > 0.0:
        parts.append(f"{drop:.0%} of pixels zeroed")
    return (f"DrAC image augmentation in the update ({', '.join(parts)}): loss += {coef:g} x "
            "(KL(pi(s) || pi(aug s)) + (V(aug s) - V(s))^2), clean frames as the target; "
            "rollouts and evals see clean frames")
