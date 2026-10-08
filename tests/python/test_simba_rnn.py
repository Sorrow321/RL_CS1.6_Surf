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


def test_view_continuous_sequence_logp_matches_the_rollout_draw():
    """--rnn + --view-continuous (2026-10-08): the update's sequence re-run (features ->
    gru_sequence over the episode cuts -> heads -> split_view -> logprob_entropy_view on the
    STORED z) scores every (act, z) with the log-prob the rollout drew it under, so the first
    minibatch's PPO ratio is 1 before any step."""
    torch.manual_seed(1)
    p, _scal = _policy()
    packer = tf.HeadPacker("cpu")
    T, N, S = 9, 4, tf.N_SCALAR
    obs_s = torch.randn(T, N, S)
    obs_i = torch.rand(T, N, 64 * 32 * 3)
    done = torch.rand(T, N) < 0.25
    done[0, 1] = True
    h = torch.randn(N, 16)
    h0 = h.clone()
    acts, zs, logps = [], [], []
    with torch.no_grad():
        for t in range(T):                    # the rollout: one GRU step per decision
            lg, _v, h = p.forward_split(obs_s[t], obs_i[t], h)
            cat, mu = tf.split_view(lg.float())
            a, z, lp = tf.sample_view(packer.pad(cat), mu, p.log_std())
            acts.append(a)
            zs.append(z)
            logps.append(lp)
            h = h * (1.0 - done[t].float()).unsqueeze(1)
        scal = obs_s.reshape(T * N, S)        # the update: time-major rows t*N + env
        feat = p.features(scal, obs_i.reshape(T * N, -1))
        g = p.gru_sequence(feat, h0, tf.gru_segments(done.numpy(), "cpu"))
        lg2, _ = p.heads(feat, scal, g)
        cat2, mu2 = tf.split_view(lg2.float())
        lp2, _ent = tf.logprob_entropy_view(packer.pad(cat2), torch.cat(acts), mu2,
                                            p.log_std(), torch.cat(zs))
    assert torch.allclose(lp2, torch.cat(logps), atol=1e-5), (lp2 - torch.cat(logps)).abs().max()


def test_trainer_scores_the_view_in_the_sequence_step():
    """The trainer's --rnn loss takes the --view-continuous branch and the update passes z."""
    src = (Path(__file__).resolve().parents[2] / "python" / "train_fast.py").read_text(
        encoding="utf-8")
    body = src[src.index("    def seq_loss("):src.index("    def mb_step_seq(")]
    assert "split_view(" in body and "logprob_entropy_view(" in body
    call = src[src.index("loss, pg, vl, el, logp = mb_step_seq("):][:600]
    assert "f_z=f_z" in call
