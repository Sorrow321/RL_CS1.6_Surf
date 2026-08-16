"""RND novelty: fitted states go quiet, fresh states pay, state survives
a checkpoint round-trip (CPU-only)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from surfgym.rnd import RND


def test_fitted_states_pay_less_than_fresh():
    torch.manual_seed(0)
    rnd = RND(15, warmup=5, device="cpu")
    seen = torch.randn(256, 15) * torch.tensor([1000.0] * 3 + [1.0] * 12)
    for _ in range(10):                      # settle stats past warmup
        rnd.bonus(seen)
    for _ in range(300):                     # fit the predictor on 'seen'
        rnd.train_step(seen)
    fresh = torch.randn(256, 15) * torch.tensor([1000.0] * 3 + [1.0] * 12) + 50.0
    b_seen = rnd.bonus(seen).mean()
    b_fresh = rnd.bonus(fresh).mean()
    assert b_fresh > 2.0 * b_seen            # novelty differential is real


def test_warmup_pays_zero():
    rnd = RND(15, warmup=10, device="cpu")
    for _ in range(10):
        b = rnd.bonus(torch.randn(64, 15))
        assert float(b.abs().sum()) == 0.0
    assert float(rnd.bonus(torch.randn(64, 15)).sum()) >= 0.0


def test_state_roundtrip_preserves_target_and_bonus():
    torch.manual_seed(1)
    a = RND(15, warmup=1, device="cpu")
    x = torch.randn(128, 15)
    for _ in range(20):
        a.bonus(x)
        a.train_step(x)
    payload = a.state_dict_all()
    torch.manual_seed(999)                   # different init for b
    b = RND(15, warmup=1, device="cpu")
    b.load_state_dict_all(payload)
    xa, xb = a.bonus(x), b.bonus(x)
    assert torch.allclose(xa, xb, atol=1e-5)


def test_bonus_scale_is_normalized():
    torch.manual_seed(2)
    rnd = RND(15, warmup=5, device="cpu")
    for _ in range(50):
        b = rnd.bonus(torch.randn(256, 15) * 500.0)
    assert 0.0 <= float(b.mean()) < 5.1      # RMS-normalized, clipped at 5
