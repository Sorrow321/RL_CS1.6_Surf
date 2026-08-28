"""--trunk {plain,resnet}: the plain path must be the OLD path, bit for bit.

Every checkpoint this project has ever trained loads through `Policy`, so
adding a second image encoder is only safe if selecting `plain` reproduces
the pre-flag class exactly: the same modules, the same state_dict keys, the
same tensors out of the same torch seed (the init loop consumes the RNG, so
a re-ordering shows up as different WEIGHTS, not just different shapes).

The reference here is not a copy of the old code kept in this file - that
would rot. It is the pre-change `python/train_fast.py` read straight out of
git and imported under its own module name, so the test compares against
what actually shipped.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import N_SCALAR, Policy                      # noqa: E402

W, H = 64, 32
OBS = N_SCALAR + W * H
# The last commit before --trunk existed. PINNED to a sha on purpose: at
# "HEAD" this test would quietly start comparing the flag against itself the
# moment the flag landed, and skip forever after.
BASE_REV = "378dbb1"


def _load_baseline():
    """Import python/train_fast.py as of BASE_REV under a private name."""
    try:
        src = subprocess.run(
            ["git", "show", f"{BASE_REV}:python/train_fast.py"],
            cwd=ROOT, capture_output=True, check=True).stdout
    except Exception as exc:                       # no git / shallow checkout
        pytest.skip(f"cannot read the baseline from git: {exc!r}")
    tmp = ROOT / "python" / "_train_fast_baseline_tmp.py"
    tmp.write_bytes(src)
    try:
        spec = importlib.util.spec_from_file_location(
            "_train_fast_baseline_tmp", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_train_fast_baseline_tmp"] = mod
        spec.loader.exec_module(mod)
        if "trunk" in mod.Policy.__init__.__code__.co_varnames:
            pytest.skip(f"{BASE_REV} already has --trunk; nothing to compare")
        return mod
    finally:
        tmp.unlink(missing_ok=True)
        sys.modules.pop("_train_fast_baseline_tmp", None)


@pytest.mark.parametrize("kw", [
    {},                                            # the shipped baseline
    {"in_ch": 2},                                  # --surf-mask
    {"route_dim": 6},                              # --route
    {"n_codes": 8, "chunk": 4},                    # --chunk
])
def test_plain_trunk_is_bit_identical_to_the_pre_flag_policy(kw):
    base = _load_baseline()
    in_ch = kw.get("in_ch", 1)
    obs = N_SCALAR + kw.get("route_dim", 0) + W * H * in_ch

    torch.manual_seed(1234)
    old = base.Policy(obs, W, H, emb=32, hidden=24, **kw)
    torch.manual_seed(1234)
    new = Policy(obs, W, H, emb=32, hidden=24, trunk="plain", **kw)

    a, b = old.state_dict(), new.state_dict()
    assert list(a) == list(b), "state_dict KEYS moved"
    for k in a:
        assert a[k].shape == b[k].shape, k
        assert torch.equal(a[k], b[k]), f"{k} differs -> init RNG order moved"

    # and the same function, not just the same weights
    x = torch.randn(5, obs)
    for m in (old, new):
        m.eval()
    with torch.no_grad():
        la, va = old(x)
        lb, vb = new(x)
    assert torch.equal(la, lb) and torch.equal(va, vb)


def test_plain_is_the_default():
    torch.manual_seed(7)
    a = Policy(OBS, W, H, emb=32, hidden=24).state_dict()
    torch.manual_seed(7)
    b = Policy(OBS, W, H, emb=32, hidden=24, trunk="plain").state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_resnet_trunk_shape_and_size():
    p = Policy(OBS, W, H, emb=512, hidden=448, trunk="resnet").eval()
    n = sum(t.numel() for t in p.conv.parameters())
    assert 2.7e6 < n < 2.9e6, n            # the benchmarked 2.79M
    # the trunk must still hand the towers exactly `emb` features, and the
    # pooled grid must be 4x8 - an ImageNet stem collapses this to 1x2
    with torch.no_grad():
        im = torch.randn(3, 1, H, W)
        feats = p.conv[:-2](im)            # up to (and including) Flatten
        assert feats.shape == (3, 128 * 4 * 8)
        logits, value = p(torch.randn(3, OBS))
    assert logits.shape[0] == 3 and value.shape == (3,)


def test_resnet_has_no_batchnorm_buffers():
    """GroupNorm, deliberately: no running stats to be replayed by the
    captured rollout graph and no train()/eval() split in what the policy
    computes."""
    p = Policy(OBS, W, H, emb=64, hidden=32, trunk="resnet")
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for m in p.modules())
    assert [k for k, _ in p.conv.named_buffers()] == []
    x = torch.randn(4, OBS)
    p.train()
    with torch.no_grad():
        a = p(x)[0]
    p.eval()
    with torch.no_grad():
        b = p(x)[0]
    assert torch.equal(a, b)
