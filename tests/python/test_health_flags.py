"""mm/health (2026-11-22): --conv-norm, --fp32-trunk, --wd-skip-norms.

* flag off: the Policy is the one the flags were added to, tensor for tensor (same seed);
* --conv-norm: the same conv / Linear draws (GroupNorms consume no RNG), a GroupNorm after each
  conv, and the image features no longer depend on the conv weights' scale - the property the
  flag exists for (the seven-map agent ran its embedding at 6.7e7x its init size);
* --fp32-trunk: a forward pass works and changes nothing on CPU (no autocast to leave);
* --wd-skip-norms: the decayed group is exactly the weight matrices.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import N_SCALAR, Policy  # noqa: E402

LW, LH, CH = 64, 32, 5


def make(**kw):
    torch.manual_seed(0)
    return Policy(N_SCALAR + LW * LH * CH, LW, LH, emb=512, hidden=448, in_ch=CH, simba=True,
                  view_continuous=True, view_absolute="velocity", **kw)


def obs(n=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCALAR + LW * LH * CH, generator=g)


def test_flags_off_is_the_old_policy():
    a, b = make(), make(conv_norm=False, fp32_trunk=False)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in a.modules())


def test_conv_norm_same_draws_and_groupnorms():
    a, b = make(), make(conv_norm=True)
    ca = [m for m in a.conv if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    cb = [m for m in b.conv if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    assert len(ca) == len(cb) == 4
    for x, y in zip(ca, cb):
        assert torch.equal(x.weight, y.weight)
    gn = [m for m in b.conv if isinstance(m, torch.nn.GroupNorm)]
    assert len(gn) == 3 and all(m.num_groups == 1 for m in gn)
    assert all(torch.equal(m.weight, torch.ones_like(m.weight)) for m in gn)
    # the towers and heads are the same draws too
    sa, sb = a.state_dict(), b.state_dict()
    for k in sa:
        if not k.startswith("conv."):
            assert torch.equal(sa[k], sb[k]), k


def test_conv_norm_makes_the_trunk_scale_free():
    b = make(conv_norm=True).eval()
    x = obs()
    with torch.no_grad():
        im = x[:, N_SCALAR:].reshape(-1, LH, LW, CH).permute(0, 3, 1, 2)
        f0 = b.conv(im)
        for m in b.conv:
            if isinstance(m, torch.nn.Conv2d):
                m.weight.mul_(1000.0)
                if m.bias is not None:
                    m.bias.mul_(1000.0)
        f1 = b.conv(im)
    assert torch.allclose(f0, f1, atol=1e-4), float((f0 - f1).abs().max())


@pytest.mark.parametrize("conv_norm,grows", [(False, True), (True, False)])
def test_activation_size_into_the_embedding(conv_norm, grows):
    """What the flag exists for: the activations entering the embedding Linear. Without it
    every conv's weight scale multiplies into them (1000^3 here, the trained agent's 1.3e5x
    at conv.4); with it they stay O(1) whatever the weights' norm."""
    p = make(conv_norm=conv_norm).eval()
    im = obs()[:, N_SCALAR:].reshape(-1, LH, LW, CH).permute(0, 3, 1, 2)
    upto = [i for i, m in enumerate(p.conv) if isinstance(m, torch.nn.Linear)][0]

    def into_embedding():
        x = im
        for m in list(p.conv)[:upto]:
            x = m(x)
        return float(x.pow(2).mean().sqrt())
    with torch.no_grad():
        r0 = into_embedding()
        for m in p.conv:
            if isinstance(m, torch.nn.Conv2d):
                m.weight.mul_(1000.0)
        r1 = into_embedding()
    if grows:
        assert r1 / r0 > 1e8, r1 / r0
    else:
        assert 0.5 < r1 / r0 < 2.0, r1 / r0


@pytest.mark.parametrize("conv_norm", [False, True])
def test_fp32_trunk_forward(conv_norm):
    a = make(conv_norm=conv_norm).eval()
    b = make(conv_norm=conv_norm, fp32_trunk=True).eval()
    x = obs()
    with torch.no_grad():
        la, va = a(x)
        lb, vb = b(x)
    assert torch.isfinite(lb).all() and torch.isfinite(vb).all()
    assert torch.allclose(la, lb) and torch.allclose(va, vb)


def test_wd_skip_norms_groups():
    p = make(conv_norm=True)
    dec = [q for q in p.parameters() if q.ndim >= 2]
    nod = [q for q in p.parameters() if q.ndim < 2]
    names = {id(q): n for n, q in p.named_parameters()}
    assert all(names[id(q)].endswith("weight") for q in dec)
    assert not any("ln" in names[id(q)] or names[id(q)].startswith("view_std") for q in dec)
    assert any(names[id(q)] == "view_std.log_std" for q in nod)
    assert len(dec) + len(nod) == len(list(p.parameters()))
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.01},
                             {"params": nod, "weight_decay": 0.0}], lr=3e-4, eps=1e-5)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.0]
