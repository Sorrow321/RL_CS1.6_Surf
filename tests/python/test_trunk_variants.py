"""--trunk plainres / softmoe (the user, 2026-10-03: residual connections, a mixture of experts).

(a) plainres: a fresh trunk IS the plain stack's function (each residual block's second conv
    starts at zero), and it learns away from it (the block convs get a gradient);
(b) softmoe: the shapes, each slot's dispatch weights sum to 1 over the tokens and each token's
    combine weights sum to 1 over the slots, every expert gets a gradient, bf16 autocast runs;
(c) a trainer smoke per trunk (CPU), beside --obs-normal + --obs-potential: the config key,
    record_ckpt rebuilds the same network from it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_obs_potential import ABS, CANNONBALL, RECORD, _run, _train  # noqa: E402
from test_view_continuous import needs_run                           # noqa: E402

FACEID = CANNONBALL.parent / "surf_src_cannonball.faceid_32.npz"
W, H, C = 64, 32, 5


def _policy(**kw):
    from train_fast import N_SCALAR, Policy
    return Policy(N_SCALAR + W * H * C, W, H, emb=512, hidden=64, in_ch=C, **kw)


def test_plainres_starts_as_the_plain_stack():
    import train_fast as tf
    torch.manual_seed(0)
    pr = _policy(trunk="plainres")
    blocks = [m for m in pr.conv.modules() if isinstance(m, tf._ResBlk)]
    assert len(blocks) == 3 and all(float(b.c2.weight.abs().max()) == 0.0 for b in blocks)
    # the plain stack with plainres's own conv / Linear weights
    convs = [m for m in pr.conv if isinstance(m, nn.Conv2d)]
    lin = [m for m in pr.conv if isinstance(m, nn.Linear)][0]
    plain = nn.Sequential(convs[0], nn.ReLU(), convs[1], nn.ReLU(), convs[2], nn.ReLU(),
                          nn.AdaptiveAvgPool2d((4, 8)), nn.Flatten(), lin, nn.ReLU())
    x = torch.randn(6, C, H, W)
    assert torch.allclose(pr.conv(x), plain(x), atol=1e-6)
    # ... and the residual paths learn: the zero convs get a gradient
    pr.conv(x).sum().backward()
    assert all(float(b.c2.weight.grad.abs().max()) > 0.0 for b in blocks)


def test_softmoe_mixes_and_routes():
    import train_fast as tf
    torch.manual_seed(0)
    p = _policy(trunk="softmoe")
    moe = [m for m in p.conv.modules() if isinstance(m, tf._SoftMoE)][0]
    assert (moe.E, moe.S, moe.d) == (8, 4, 16)
    x = torch.randn(5, C, H, W)
    out = p.conv(x)
    assert out.shape == (5, 512)
    # the dispatch / combine weights, recomputed from the conv features
    feats = p.conv[:7](x)                                    # (5, 64, 4, 8) after the pool
    tok = feats.flatten(2).transpose(1, 2)
    logits = tok @ moe.phi
    assert torch.allclose(logits.softmax(1).sum(1), torch.ones(5, 32), atol=1e-5)
    assert torch.allclose(logits.softmax(2).sum(2), torch.ones(5, 32), atol=1e-5)
    out.square().sum().backward()
    for e in range(moe.E):
        assert float(moe.w1.grad[e].abs().max()) > 0.0, e
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert p.conv(x).shape == (5, 512)
    with pytest.raises(SystemExit):
        _policy(trunk="softmoe", plan_film=1, film_dim=1)


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
@pytest.mark.parametrize("trunk", ["plainres", "softmoe"])
def test_trunk_trainer_smoke(trunk):
    run = f"cya_trunk_{trunk}"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-normal", "1", "--trunk", trunk])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    d = ROOT / "runs" / run
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["trunk"] == trunk
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    shutil.rmtree(d, ignore_errors=True)
