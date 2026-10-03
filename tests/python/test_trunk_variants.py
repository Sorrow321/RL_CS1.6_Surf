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


def test_simba_towers_and_embedding():
    import train_fast as tf
    torch.manual_seed(0)
    p = _policy(simba=True)
    assert isinstance(p.pi[0], nn.Linear) and isinstance(p.pi[-1], nn.LayerNorm)
    blocks = [m for m in p.pi if isinstance(m, tf._SimbaBlock)]
    assert len(blocks) == 1 and blocks[0].fc1.out_features == tf.SIMBA_EXPAND * 64
    assert any(isinstance(m, nn.LayerNorm) for m in p.conv)          # the embedding's LN
    deep = _policy(simba=True, tower_depth=4)
    assert len([m for m in deep.vf if isinstance(m, tf._SimbaBlock)]) == 2
    from train_fast import N_SCALAR
    x = torch.randn(4, N_SCALAR + W * H * C)
    logits, value = p(x)
    assert logits.shape[0] == 4 and value.shape == (4,)
    # every Linear inside the residual blocks got the orthogonal init (not torch's default)
    w = blocks[0].fc1.weight
    assert torch.allclose(w @ w.T, 2.0 * torch.eye(w.shape[0]), atol=1e-4) or \
        torch.allclose(w.T @ w, 2.0 * torch.eye(w.shape[1]), atol=1e-4)
    with pytest.raises(SystemExit):
        _policy(simba=True, trunk="plainres")


def test_split_trunk_separates_policy_and_value():
    torch.manual_seed(0)
    p = _policy(split_trunk=True)
    assert p.conv_v is not None
    from train_fast import N_SCALAR
    x = torch.randn(3, N_SCALAR + W * H * C)
    l0, v0 = p(x)
    with torch.no_grad():
        for prm in p.conv_v.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    l1, v1 = p(x)
    assert torch.equal(l0, l1) and not torch.allclose(v0, v1)       # value reads conv_v only
    with torch.no_grad():
        for prm in p.conv.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    l2, v2 = p(x)
    assert not torch.allclose(l1, l2) and torch.equal(v1, v2)       # policy reads conv only
    assert _policy().conv_v is None


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
@pytest.mark.parametrize("flag", ["--simba", "--split-trunk"])
def test_norm_split_trainer_smoke(flag):
    run = f"cya_{flag.strip('-').replace('-', '_')}"
    # --keys-hold: the scalar-side block the rented recipe carries (it is how --split-trunk
    # first failed on a box: a guard refused any route_dim)
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-normal", "1", "--keys-hold", flag, "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    d = ROOT / "runs" / run
    key = flag.strip("-").replace("-", "_")
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"][key] == 1
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    shutil.rmtree(d, ignore_errors=True)
