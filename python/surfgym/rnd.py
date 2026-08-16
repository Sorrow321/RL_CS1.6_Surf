"""rnd.py — Random Network Distillation over the scalar observation.

The count-based novelty keys on discretized position (x yaw sector with
--int-view); RND is its continuous generalization: a frozen random target
MLP f(s) and a trained predictor f'(s), bonus = ||f - f'||^2. States the
fleet frequents are fitted (low bonus); fresh state/velocity/gaze combos
are not. Runs on the 15 scalar obs — the image is deliberately excluded
(a render-space RND would pay for wall textures; the scalars already
carry position, velocity, view and contact state).

Both nets see RMS-normalized inputs (positions are in thousands of units,
contact flags are 0/1 — raw scales would let one dimension dominate the
MSE), and the bonus is normalized by its own running RMS so --rnd-coef
has a stable meaning across training stages. The first ``warmup`` calls
pay zero while the statistics settle.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["RND"]


class _RMS:
    """Running mean/std (Welford over batches), torch tensors on device."""

    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        bm = x.mean(0)
        bv = x.var(0, unbiased=False)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc
                    + delta.pow(2) * self.count * bc / tot) / tot
        self.count = tot

    def state(self) -> dict:
        return {"mean": self.mean.cpu().numpy(),
                "var": self.var.cpu().numpy(), "count": self.count}

    def load(self, d) -> None:
        self.mean = torch.as_tensor(np.asarray(d["mean"]),
                                    device=self.mean.device,
                                    dtype=torch.float32)
        self.var = torch.as_tensor(np.asarray(d["var"]),
                                   device=self.var.device,
                                   dtype=torch.float32)
        self.count = float(d["count"])


def _mlp(in_dim, hidden, out_dim):
    m = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                      nn.Linear(hidden, hidden), nn.ReLU(),
                      nn.Linear(hidden, out_dim))
    for lin in m:
        if isinstance(lin, nn.Linear):
            nn.init.orthogonal_(lin.weight, np.sqrt(2))
            nn.init.zeros_(lin.bias)
    return m


class RND:
    def __init__(self, in_dim: int, feat: int = 64, hidden: int = 128,
                 lr: float = 1e-4, warmup: int = 50, device="cuda") -> None:
        self.device = torch.device(device)
        self.target = _mlp(in_dim, hidden, feat).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred = _mlp(in_dim, hidden, feat).to(self.device)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)
        self.obs_rms = _RMS(in_dim, self.device)
        # scalar RMS of the raw bonus — normalizes the coef's meaning
        self.bon_mean = 0.0
        self.bon_var = 1.0
        self.bon_count = 1e-4
        self.warmup = int(warmup)
        self.calls = 0

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.obs_rms.mean)
                / torch.sqrt(self.obs_rms.var + 1e-8)).clamp(-5.0, 5.0)

    @torch.no_grad()
    def bonus(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row novelty for a rollout batch. Updates the running stats;
        returns zeros during warmup and clips at 5 RMS units after."""
        x = x.float()
        self.obs_rms.update(x)
        self.calls += 1
        xn = self._norm(x)
        err = (self.target(xn) - self.pred(xn)).pow(2).mean(1)
        bm = float(err.mean())
        bv = float(err.var(unbiased=False))
        bc = x.shape[0]
        delta = bm - self.bon_mean
        tot = self.bon_count + bc
        self.bon_mean += delta * bc / tot
        self.bon_var = (self.bon_var * self.bon_count + bv * bc
                        + delta * delta * self.bon_count * bc / tot) / tot
        self.bon_count = tot
        if self.calls <= self.warmup:
            return torch.zeros_like(err)
        return (err / (self.bon_var + 1e-8) ** 0.5).clamp(0.0, 5.0)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """One predictor update on (a slice of) rollout observations.
        Returns the loss TENSOR (caller decides if/when to .item())."""
        xn = self._norm(x.float().detach())
        loss = (self.target(xn) - self.pred(xn)).pow(2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return loss.detach()

    # -- persistence --------------------------------------------------------
    def state_dict_all(self) -> dict:
        return {"target": self.target.state_dict(),
                "pred": self.pred.state_dict(),
                "opt": self.opt.state_dict(),
                "obs_rms": self.obs_rms.state(),
                "bon": (self.bon_mean, self.bon_var, self.bon_count),
                "calls": self.calls}

    def load_state_dict_all(self, d) -> None:
        # the TARGET must survive resumes byte-for-byte: a re-rolled target
        # makes every fitted state novel again (a fleet-wide windfall)
        self.target.load_state_dict(d["target"])
        self.pred.load_state_dict(d["pred"])
        self.opt.load_state_dict(d["opt"])
        self.obs_rms.load(d["obs_rms"])
        self.bon_mean, self.bon_var, self.bon_count = d["bon"]
        self.calls = int(d["calls"])
