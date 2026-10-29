"""``--demo-bc C``: at the first decision after every spine spawn, C x the engine-scored NLL of the
spine's NEXT held keys (fwd, side, duck) - an online distillation of the curriculum's own line at
its own starts (pbfr1b1, 2026-10-29).

The term the trainer adds inside mb_step is checked here on its own: the target engine value e of
a held head is scored as policy bin e + 1, with KEEP and an explicit press of the held value one
class (keys_engine_logp), rows without a target (-1) contribute nothing, and a policy that already
puts its mass on the target pays ~0.

    python -m pytest tests/python/test_demo_bc.py -q
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

from surfgym import keyshold                                      # noqa: E402
from train_fast import keys_engine_logp, keys_held_from_features  # noqa: E402

NACT, NPAD = 6, 15


def _term(padded, feat, tgt):
    """mb_step's --demo-bc computation, verbatim."""
    _dm = (tgt[:, 0] >= 0).float()
    _held = keys_held_from_features(feat)
    _lt = keys_engine_logp(F.log_softmax(padded, dim=-1), _held)
    _lp = torch.zeros_like(_dm)
    for _c, _h in enumerate(keyshold.HELD_HEADS):
        _lp = _lp + _lt[:, _h, :].gather(1, (tgt[:, _c].clamp_min(0) + 1).unsqueeze(1)).squeeze(1)
    return -(_lp * _dm).sum() / _dm.sum().clamp_min(1.0)


def _feat(held_rows):
    """(B, 3) held engine keys -> the 7 keys-hold observation columns."""
    out = np.zeros((len(held_rows), keyshold.N_FEATURES), np.float32)
    for i, (f, s, d) in enumerate(held_rows):
        out[i, f] = 1.0
        out[i, 3 + s] = 1.0
        out[i, 6] = float(d)
    return torch.from_numpy(out)


def _peaked(B, bins):
    """Logits with all mass on the given policy bin of each held head (fwd 4, side 4, duck 3)."""
    p = torch.full((B, NACT, NPAD), -1e30)
    p[:, :, :2] = 0.0
    for h, n in ((2, 4), (3, 4), (5, 3), (4, 2)):
        p[:, h, :n] = 0.0
    for h, b in bins.items():
        p[:, h, :] = torch.where(torch.arange(NPAD) == b, 30.0, p[:, h, :])
    return p


def test_keep_and_the_explicit_press_of_the_held_key_are_one_class():
    held = [(1, 2, 0)]                                  # holding strafe-right
    tgt = torch.tensor([[1, 2, 0]])                     # the line keeps holding it
    on_keep = _peaked(1, {2: 0, 3: 0, 5: 0})            # the policy says KEEP on every held head
    on_expl = _peaked(1, {2: 2, 3: 3, 5: 1})            # ... or presses the held values explicitly
    assert float(_term(on_keep, _feat(held), tgt)) == pytest.approx(0.0, abs=1e-4)
    assert float(_term(on_expl, _feat(held), tgt)) == pytest.approx(0.0, abs=1e-4)


def test_the_wrong_key_pays():
    held = [(1, 2, 0)]
    tgt = torch.tensor([[1, 2, 0]])
    left = _peaked(1, {2: 0, 3: 1, 5: 0})               # strafe-LEFT (side engine 0 = policy bin 1)
    assert float(_term(left, _feat(held), tgt)) > 20.0


def test_rows_without_a_target_contribute_nothing():
    held = [(1, 2, 0), (1, 1, 0)]
    good = _peaked(2, {2: 0, 3: 0, 5: 0})
    bad_row1 = good.clone()
    bad_row1[1, 3, :] = torch.where(torch.arange(NPAD) == 1, 30.0, -1e30)   # row 1 presses left
    bad_row1[1, 3, 1:4] = torch.tensor([30.0, 0.0, 0.0])
    tgt = torch.tensor([[1, 2, 0], [-1, -1, -1]])       # row 1 is not a spine-spawn decision
    assert float(_term(bad_row1, _feat(held), tgt)) == pytest.approx(0.0, abs=1e-4)


def test_the_trainer_wires_it_off_by_default():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--demo-bc"' in src
    assert "DEMO_BC = float(args.demo_bc or 0.0)" in src
    assert 'meta["config"]["demo_bc"] = float(DEMO_BC)' in src
    assert "if DEMO_BC > 0.0\n             else None)" in src     # no buffer without the flag
    assert '"demo_bc"' in (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
