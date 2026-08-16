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

from train_fast import (N_SCALAR, Policy,                   # noqa: E402
                        contiguous_optimizer_state,
                        relayout_optimizer_state)

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


def test_optimizer_state_is_restrided_on_resume(policy):
    """A pre-change checkpoint carries NCHW Adam moments. Fused Adam refuses
    a param/moment layout mismatch, so a bare --ckpt resume died on the
    first opt.step() until the moments were restrided on load."""
    legacy = Policy(OBS, W, H, emb=32, hidden=24)
    legacy.conv = legacy.conv.to(memory_format=torch.contiguous_format)
    lopt = torch.optim.Adam(legacy.parameters(), lr=3e-4)
    legacy(torch.rand(4, OBS)).__getitem__(0).sum().backward()
    lopt.step()                                    # materialise NCHW moments

    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    opt.load_state_dict(lopt.state_dict())
    mismatched = [k for p, st in opt.state.items() for k, v in st.items()
                  if torch.is_tensor(v) and v.shape == p.shape
                  and v.stride() != p.stride()]
    assert mismatched, "test is vacuous: no layout mismatch to fix"
    assert relayout_optimizer_state(opt) == len(mismatched)
    for p, st in opt.state.items():
        for v in st.values():
            if torch.is_tensor(v) and v.shape == p.shape:
                assert v.stride() == p.stride()
    assert relayout_optimizer_state(opt) == 0, "must be idempotent"


def test_saved_optimizer_state_loads_into_an_nchw_model(policy):
    """The direction the round-trip test used to miss: a checkpoint written
    HERE, resumed by code whose trunk is still NCHW (i.e. simply checking out
    the pre-channels_last commit). Adam's moments inherit the params' layout
    and load_state_dict never re-strides them, so an un-normalised save is
    unloadable there — fused Adam rejects the mismatch on the first step."""
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    policy(torch.rand(4, OBS))[0].sum().backward()
    opt.step()

    def row_major(shape):
        strides, acc = [1] * len(shape), 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = acc
            acc *= shape[i]
        return tuple(strides)

    assert any(v.stride() != row_major(v.shape)
               for st in opt.state.values() for v in st.values()
               if torch.is_tensor(v)), "test is vacuous: nothing was NHWC"

    saved = contiguous_optimizer_state(opt.state_dict())
    for st in saved["state"].values():
        for v in st.values():
            if torch.is_tensor(v):
                assert v.stride() == row_major(v.shape),                     "saved moments must be layout-neutral"

    legacy = Policy(OBS, W, H, emb=32, hidden=24)
    legacy.conv = legacy.conv.to(memory_format=torch.contiguous_format)
    lopt = torch.optim.Adam(legacy.parameters(), lr=3e-4)
    lopt.load_state_dict(saved)
    for p, st in lopt.state.items():
        for v in st.values():
            if torch.is_tensor(v) and v.shape == p.shape:
                assert v.stride() == p.stride(), "unloadable by the NCHW trunk"
    # and the values survived the normalisation
    a = next(iter(opt.state.values()))["exp_avg"]
    b = next(iter(lopt.state.values()))["exp_avg"]
    assert torch.equal(a, b)


def test_relayout_preserves_moment_values(policy):
    """Restriding must move VALUES, not reinterpret bytes — a wrong copy
    would silently corrupt Adam's second moment and quietly wreck training."""
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    policy(torch.rand(4, OBS))[0].sum().backward()
    opt.step()
    conv2 = policy.conv[2].weight
    st = opt.state[conv2]
    st["exp_avg"] = st["exp_avg"].contiguous(memory_format=torch.contiguous_format)
    before = st["exp_avg"].clone()
    relayout_optimizer_state(opt)
    assert torch.equal(opt.state[conv2]["exp_avg"], before)


def test_forward_split_matches_the_fused_forward(policy):
    """forward() must be exactly forward_split() on the two halves — the
    eval/record paths keep using the fused row while the update uses the
    split one, and they have to be the same network."""
    torch.manual_seed(4)
    obs = torch.rand(6, OBS)
    with torch.no_grad():
        a, v = policy(obs)
        b, w = policy.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:])
    assert torch.equal(a, b) and torch.equal(v, w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_depth_buffer_is_exact_under_autocast(policy):
    """S3 stores the depth half of the rollout buffer in bf16. That is not an
    approximation: autocast already rounds the image to bf16 on the way into
    the conv, so a pre-rounded buffer hands the update the identical tensor.
    Anything other than bitwise equality here means S3 changed the math."""
    p = policy.to("cuda")
    torch.manual_seed(5)
    obs = torch.rand(8, OBS, device="cuda")
    img16 = obs[:, N_SCALAR:].to(torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda",
                                         dtype=torch.bfloat16):
        a_fused, v_fused = p(obs)
        a_split, v_split = p.forward_split(obs[:, :N_SCALAR], img16)
    assert torch.equal(a_fused, a_split), \
        f"logits differ by {(a_fused - a_split).abs().max():.3e}"
    assert torch.equal(v_fused, v_split)


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
