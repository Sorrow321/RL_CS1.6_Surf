"""Self-imitation learning (Oh, Guo, Singh, Lee, ICML 2018) - the buffer and
the loss terms, as an auxiliary to PPO (train_fast.py --sil-coef).

Why here (docs/litsurvey-labyrinth.md, mechanism 5): inside a region the
policy can reach but cannot yet survive (the unitfarmer2 pit), an
on-policy rollout holds a handful of attempts and their advantages are a
thin, sign-mixed gradient. SIL keeps the transitions whose realised return
beat the critic's estimate and trains the actor toward the actions that
produced them, weighted by that gap, while pulling the critic up to the
realised return. It never pushes an action down: rows with R <= V(s)
contribute nothing (the loss clamps at 0), and rows in the buffer whose
gain has evaporated under the current critic are sampled with ~0 priority.

The buffer holds exactly what the policy's forward needs to re-score an
action: the scalar observation row, the image frame, the action
6-tuple, the continuous view's stored z (absolute view) and the privileged
critic columns when the run has them. Prioritised sampling is proportional
to max(R - V(s), 0) with V recomputed lazily: the gain of a sampled row is
refreshed with the critic that just scored it. FIFO eviction. Not
checkpointed: a resume starts empty.
"""
from __future__ import annotations

import torch


class SILBuffer:
    def __init__(self, cap: int, device, img_dtype, scal_dim: int, frame: int,
                 nact: int, nz: int = 0, priv: int = 0, seed: int = 0):
        self.cap = max(1, int(cap))
        self.device = device
        self.scal = torch.zeros((self.cap, int(scal_dim)), device=device)
        self.img = torch.zeros((self.cap, int(frame)), device=device, dtype=img_dtype)
        self.act = torch.zeros((self.cap, int(nact)), dtype=torch.long, device=device)
        self.z = torch.zeros((self.cap, int(nz)), device=device) if nz else None
        self.priv = torch.zeros((self.cap, int(priv)), device=device) if priv else None
        self.ret = torch.zeros((self.cap,), device=device)
        self.gain = torch.zeros((self.cap,), device=device)   # R - V as last seen
        self.size = 0
        self.head = 0
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))
        self.n_added = 0
        self.n_offered = 0

    # ------------------------------------------------------------- insertion
    def _slots(self, n: int) -> torch.Tensor:
        idx = (self.head + torch.arange(n, device=self.device)) % self.cap
        self.head = (self.head + n) % self.cap
        self.size = min(self.size + n, self.cap)
        return idx

    @torch.no_grad()
    def add(self, scal, img, act, ret, val, z=None, priv=None) -> int:
        """Keep every row whose return target beats the critic (R > V);
        returns how many were kept. If more than the capacity qualify, the
        largest gains win."""
        gain = ret - val
        keep = torch.nonzero(gain > 0, as_tuple=False).squeeze(-1)
        self.n_offered += int(ret.numel())
        n = int(keep.numel())
        if n == 0:
            return 0
        if n > self.cap:
            top = torch.topk(gain[keep], self.cap).indices
            keep = keep[top]
            n = self.cap
        rows = self._slots(n)
        self.scal[rows] = scal[keep].to(self.scal.dtype)
        self.img[rows] = img[keep].to(self.img.dtype)
        self.act[rows] = act[keep]
        self.ret[rows] = ret[keep].to(self.ret.dtype)
        self.gain[rows] = gain[keep].to(self.gain.dtype)
        if self.z is not None and z is not None:
            self.z[rows] = z[keep].to(self.z.dtype)
        if self.priv is not None and priv is not None:
            self.priv[rows] = priv[keep].to(self.priv.dtype)
        self.n_added += n
        return n

    # -------------------------------------------------------------- sampling
    def sample(self, n: int) -> torch.Tensor:
        """Row indices, drawn with replacement in proportion to the last
        known positive gain (a row whose gain has gone <= 0 is nearly never
        drawn again); empty buffer -> empty index."""
        if self.size == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        p = self.gain[:self.size].clamp(min=0.0) + 1e-6
        return torch.multinomial(p, int(n), replacement=True, generator=self.gen)

    @torch.no_grad()
    def refresh(self, idx: torch.Tensor, raw_gain: torch.Tensor) -> None:
        """Lazy V recompute: the rows just scored get their gain (R - V under
        the CURRENT critic) written back, so priorities follow the critic."""
        self.gain[idx] = raw_gain.to(self.gain.dtype)

    def mean_gain(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.gain[:self.size].clamp(min=0.0).mean())


def sil_loss_terms(logp: torch.Tensor, ent: torch.Tensor, value: torch.Tensor,
                   ret: torch.Tensor, ent_coef: float = 0.0):
    """Oh et al. eq. 1-2: L = -log pi(a|s) (R - V)_+  +  1/2 (R - V)_+^2,
    the policy term with the gap DETACHED, the value term with its
    gradient (it pulls V up toward R), minus ent_coef x entropy.
    -> (loss, raw R - V) ; the raw gap is what the buffer refreshes with."""
    raw = ret - value
    gain = raw.clamp(min=0.0)
    pg = (-logp * gain.detach()).mean()
    vl = 0.5 * gain.pow(2).mean()
    el = -ent.mean()
    return pg + vl + float(ent_coef) * el, raw.detach()
