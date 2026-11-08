"""mcts_policy.py - the proposal wrapper of tools/mcts_search.py (unr1b3).

SampledTorchPolicy (train_fast) with a PER-ENV greedy mask settable between decisions (the MCTS
gives child 0 of every expansion the policy's mode and samples the others), and the keys / view
temperatures of the trainer's own draw (sample_view's ``temp`` / ``temp_view``: the --unstuck
path), so a proposal at temperature 1 IS the trainer's behaviour policy. Only --view-continuous
checkpoints (the MCTS's held steer is a view offset)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from train_fast import N_VIEW, SampledTorchPolicy, sample_view, split_view  # noqa: E402


class MctsPolicy(SampledTorchPolicy):
    def __init__(self, *a, keys_temp: float = 1.0, view_temp: float = 1.0, **kw):
        super().__init__(*a, **kw)
        if not self.view_continuous:
            raise ValueError("MctsPolicy needs a --view-continuous policy")
        if float(keys_temp) <= 0 or float(view_temp) <= 0:
            raise ValueError("temperatures must be > 0")
        self.greedy = None                      # (N,) bool: envs that take the mode
        self._kt = None if float(keys_temp) == 1.0 else float(keys_temp)
        self._vt = (None if float(view_temp) == 1.0 and self._kt is None
                    else [float(view_temp), 1.0])

    @torch.inference_mode()
    def _decide(self, obs):
        logits, _ = self._net(self._obs(obs))
        cat, mu = split_view(logits.float())
        padded = self._mask_padded(self.packer.pad(cat), obs)
        if self._vt is None:
            act, z, _ = sample_view(padded, mu, self.policy.log_std())
        else:
            act, z, _ = sample_view(padded, mu, self.policy.log_std(), temp=self._kt,
                                    temp_view=self._vt)
        g = self.greedy
        if g is not None and np.any(g):
            gi = torch.as_tensor(np.flatnonzero(g), device=act.device)
            act[gi, N_VIEW:] = padded[gi, N_VIEW:].argmax(-1)
            z[gi] = mu[gi]
        return self._finish_view(act[:, N_VIEW:], z)
