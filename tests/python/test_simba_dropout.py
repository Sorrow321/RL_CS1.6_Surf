"""--dropout on --simba towers (_EndDrop): the consistent mask on each tower's OUTPUT.

What could be silently wrong: the mask moves the state_dict (the recorder and the searches
build a dropout-free Policy and load strictly), the cleared network is not the dropout-free
one, a drawn mask is not applied, or two forwards under one draw disagree (then the rollout and
the update would not score the same sub-network and PPO's ratio would not start at 1)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import train_fast as tf  # noqa: E402


def _policy(dropout):
    for scal in range(10, 80):
        try:
            return tf.Policy(scal + 64 * 32 * 3, 64, 32, emb=64, hidden=32, in_ch=3, simba=True,
                             dropout=dropout, view_continuous=True, view_absolute="velocity")
        except AssertionError:
            continue
    raise AssertionError("no obs_dim fits")


def test_simba_dropout_is_a_consistent_mask_on_the_tower_output():
    torch.manual_seed(0)
    p = _policy(0.25)
    torch.manual_seed(0)
    q = _policy(0.0)
    drops = [m for m in p.modules() if isinstance(m, tf._EndDrop)]
    assert len(drops) == 2 and isinstance(p.pi[-1], tf._EndDrop) and isinstance(p.vf[-1], tf._EndDrop)
    assert not any(isinstance(m, tf._TanhDrop) for m in q.modules())
    # the same tensors under the same names: a dropout checkpoint loads strictly into q
    assert list(p.state_dict()) == list(q.state_dict())
    q.load_state_dict(p.state_dict(), strict=True)
    B = 6
    s = torch.randn(B, tf.N_SCALAR)
    img = torch.rand(B, 64 * 32 * 3)
    with torch.no_grad():
        lq, vq = q.forward_split(s, img)
        p.dropout_clear()
        lc, vc = p.forward_split(s, img)
        assert torch.equal(lc, lq) and torch.equal(vc, vq)      # cleared = the full network
        g = torch.Generator().manual_seed(1)
        p.dropout_resample(g)
        keep = (drops[0].mask > 0).float().mean().item()
        assert 0.5 < keep < 0.95                                  # p = 0.25 of 32 units
        assert torch.allclose(drops[0].mask[drops[0].mask > 0],
                              torch.full_like(drops[0].mask[drops[0].mask > 0], 1 / 0.75))
        l1, v1 = p.forward_split(s, img)
        l2, v2 = p.forward_split(s, img)
        assert torch.equal(l1, l2) and torch.equal(v1, v2)      # one draw, one sub-network
        assert not torch.allclose(l1, lc)                        # the draw reaches the logits
        p.dropout_clear()
        l3, _ = p.forward_split(s, img)
        assert torch.equal(l3, lq)


def test_the_weight_averages_keep_the_full_network():
    """--eval-ema: the averages take the policy's buffers by copy after every update; the
    dropout masks are buffers too and must stay out of that copy (the averages fly evals only),
    the trainer's DROP_MASKS rule."""
    import copy
    torch.manual_seed(0)
    p = _policy(0.25)
    ema = copy.deepcopy(p)
    drop_masks = {id(m.mask) for m in p.modules() if isinstance(m, tf._TanhDrop)}
    assert len(drop_masks) == 2
    p.dropout_resample(torch.Generator().manual_seed(3))
    with torch.no_grad():
        for _be, _b in zip(ema.buffers(), p.buffers()):     # the trainer's loop, verbatim
            if id(_b) not in drop_masks:
                _be.copy_(_b)
    for m in ema.modules():
        if isinstance(m, tf._TanhDrop):
            assert torch.equal(m.mask, torch.ones_like(m.mask))
    assert any(not torch.equal(m.mask, torch.ones_like(m.mask))
               for m in p.modules() if isinstance(m, tf._TanhDrop))
