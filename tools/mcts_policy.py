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
        self.hold = None                        # (N,) bool: envs that repeat their first draw
        self.hold_store = False                 # True on the macro's first decision
        # --mcts-labels (azr1b3): when set, the next decision keeps the SCALAR half of the row
        # the network reads (everything before the image: Policy.forward splits at scal_dim) as
        # `last_scal`, (N, scal_dim) float32 - the input row of a tree label. Off: nothing kept
        self.capture_scal = False
        self.last_scal = None
        self._h_act = self._h_z = None
        self._kt = None if float(keys_temp) == 1.0 else float(keys_temp)
        self._vt = (None if float(view_temp) == 1.0 and self._kt is None
                    else [float(view_temp), 1.0])

    def set_temps(self, keys_temp, view_temp):
        """PER-ENV temperatures ((N,) arrays; None = the constructor's scalars): the trainer's own
        broadcasting - an (N, 1, 1) block over the keys' (N, heads, NPAD) logits, an (N, n_z)
        sigma multiplier on the view heads (the pitch head untouched)."""
        if keys_temp is None:
            return
        kt = torch.as_tensor(np.asarray(keys_temp, np.float32).reshape(-1, 1, 1),
                             device=self.device)
        vt = np.ones((len(view_temp), int(getattr(self.policy, "n_z", 2))), np.float32)
        vt[:, 0] = np.asarray(view_temp, np.float32)
        self._kt = kt
        self._vt = torch.as_tensor(vt, device=self.device)

    @torch.inference_mode()
    def _decide(self, obs):
        x = self._obs(obs)
        if self.capture_scal:
            self.last_scal = x[:, :int(self.policy.scal_dim)].float().cpu().numpy()
        logits, _ = self._net(x)
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
        h = self.hold
        if h is not None and np.any(h):
            # HELD envs (the MCTS's random children): the policy-space action and the view z drawn
            # at the macro's first decision are repeated for the whole macro - coherent random
            # actions (the Nature paper's action repeat), in the policy's own action frame, so the
            # held keys the policy reads stay the keys the engine holds
            hi = torch.as_tensor(np.flatnonzero(h), device=act.device)
            if self.hold_store:
                self._h_act = act[hi, N_VIEW:].clone()
                self._h_z = z[hi].clone()
            elif self._h_act is not None:
                act[hi, N_VIEW:] = self._h_act
                z[hi] = self._h_z
        return self._finish_view(act[:, N_VIEW:], z)
