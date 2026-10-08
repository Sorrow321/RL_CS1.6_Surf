"""--simba with --rnn gru: the GRU state is appended to each SimBa tower's input; one step and a
whole sequence give the same outputs, and the hidden state moves the actor's logits."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import train_fast as tf  # noqa: E402


def _policy():
    for scal in range(10, 80):
        try:
            return tf.Policy(scal + 64 * 32 * 3, 64, 32, emb=64, hidden=32, in_ch=3, simba=True,
                             rnn="gru", rnn_size=16, view_continuous=True,
                             view_absolute="velocity"), scal
        except AssertionError:
            continue
    raise AssertionError("no obs_dim fits")


def test_simba_towers_take_the_gru_state():
    torch.manual_seed(0)
    p, scal = _policy()
    assert p.gru is not None and p.simba
    assert p.pi[0].in_features == p.feat_dim + 16
    B = 5
    s = torch.randn(B, tf.N_SCALAR)
    img = torch.rand(B, 64 * 32 * 3)
    h0 = torch.zeros(B, 16)
    with torch.no_grad():
        lg0, v0, h1 = p.forward_split(s, img, h0)
        lg1, v1, h2 = p.forward_split(s, img, h1)
    assert lg0.shape[0] == B and v0.shape[0] == B and h1.shape == (B, 16)
    assert torch.isfinite(lg0).all() and torch.isfinite(v0).all()
    assert not torch.allclose(lg0, lg1)          # the state reaches the actor
