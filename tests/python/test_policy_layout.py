"""The conv trunk runs channels_last — prove that changed nothing but speed.

`Policy` holds its conv trunk in NHWC and restrides the depth image into
channels_last before the trunk (free, because depth is one channel). Three
things have to stay true, and all three are silent if they break:

  1. the forward produces the same numbers as the old NCHW path — in
     particular Flatten must still emit features in LOGICAL (C,H,W) order,
     not memory order, or every downstream weight is permuted;
  2. checkpoints stay interchangeable in BOTH directions, so an old ckpt
     warm-starts here and a ckpt written here still loads in play.py /
     record_ckpt.py / render_pov.py;
  3. the memory format survives `.to(device)` — otherwise the whole point
     is lost with no visible symptom except the speed coming back.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import N_SCALAR, Policy                     # noqa: E402

W, H = 32, 16                     # small trunk: same code path, fast test
OBS = N_SCALAR + W * H


def _nchw_forward(p: Policy, obs: torch.Tensor):
    """The pre-change forward, verbatim."""
    img = obs[:, N_SCALAR:].reshape(-1, 1, p.lidar_h, p.lidar_w)
    f = torch.cat([obs[:, p.feat_idx], p.conv(img)], dim=1)
    return p.action_head(p.pi(f)), p.value_head(p.vf(f)).squeeze(-1)


@pytest.fixture
def policy():
    torch.manual_seed(0)
    return Policy(OBS, W, H, emb=32, hidden=24).eval()


def test_forward_matches_the_nchw_path(policy):
    torch.manual_seed(1)
    obs = torch.rand(8, OBS)
    with torch.no_grad():
        a_new, v_new = policy(obs)
        # the same module, fed an NCHW-contiguous image the old way
        nchw = Policy(OBS, W, H, emb=32, hidden=24).eval()
        nchw.load_state_dict(policy.state_dict())
        nchw.conv = nchw.conv.to(memory_format=torch.contiguous_format)
        a_ref, v_ref = _nchw_forward(nchw, obs)
    assert torch.allclose(a_new, a_ref, atol=1e-5, rtol=1e-4), \
        f"logits drift {(a_new - a_ref).abs().max():.2e}"
    assert torch.allclose(v_new, v_ref, atol=1e-5, rtol=1e-4), \
        f"value drift {(v_new - v_ref).abs().max():.2e}"


def test_flatten_keeps_logical_channel_order(policy):
    """A channels_last tensor flattened in MEMORY order would permute the
    2048 trunk features under the emb Linear — same shape, garbage weights."""
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    xl = x.to(memory_format=torch.channels_last)
    assert torch.equal(nn.Flatten()(x), nn.Flatten()(xl))


def test_checkpoint_roundtrips_both_ways(policy):
    torch.manual_seed(2)
    obs = torch.rand(4, OBS)
    plain = Policy(OBS, W, H, emb=32, hidden=24).eval()
    plain.conv = plain.conv.to(memory_format=torch.contiguous_format)

    # new -> old
    plain.load_state_dict(policy.state_dict())
    with torch.no_grad():
        assert torch.allclose(policy(obs)[0], _nchw_forward(plain, obs)[0],
                              atol=1e-5, rtol=1e-4)
    # old -> new (the warm-start direction: every existing ckpt is NCHW)
    torch.manual_seed(3)
    legacy = Policy(OBS, W, H, emb=32, hidden=24).eval()
    legacy.conv = legacy.conv.to(memory_format=torch.contiguous_format)
    fresh = Policy(OBS, W, H, emb=32, hidden=24).eval()
    fresh.load_state_dict(legacy.state_dict())
    with torch.no_grad():
        assert torch.allclose(fresh(obs)[0], _nchw_forward(legacy, obs)[0],
                              atol=1e-5, rtol=1e-4)


def test_conv_weights_are_channels_last(policy):
    for m in policy.conv:
        if isinstance(m, nn.Conv2d):
            assert m.weight.is_contiguous(memory_format=torch.channels_last), \
                f"{m} weight lost its NHWC layout"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_memory_format_survives_to_device(policy):
    p = policy.to("cuda")
    for m in p.conv:
        if isinstance(m, nn.Conv2d):
            assert m.weight.is_contiguous(memory_format=torch.channels_last), \
                ".to(device) dropped channels_last — the speedup silently reverts"
    obs = torch.rand(4, OBS, device="cuda")
    with torch.no_grad():
        img = obs[:, N_SCALAR:].reshape(-1, p.lidar_h, p.lidar_w, 1) \
                               .permute(0, 3, 1, 2)
        # the image slice keeps the obs row pitch as its batch stride, so it
        # is not "contiguous" in either format — what matters is that it
        # DECLARES NHWC, so cudnn stays in NHWC and emits an NHWC result
        # instead of transposing back
        assert img.stride()[1] == 1, "channel stride must be 1 (NHWC)"
        assert p.conv[0](img).is_contiguous(memory_format=torch.channels_last), \
            "first conv fell back to NCHW — the transposes are still there"
        p(obs)
