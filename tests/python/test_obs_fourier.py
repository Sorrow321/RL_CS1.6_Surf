"""--obs-fourier L: a fixed positional encoding of the DEPTH channel.

The depth image is one scalar per ray, and surfgym/vision.py encodes a
ray's distance as

    enc = min(t, near)/near + 0.25*(1 - exp(-max(t - near, 0)/2500))

- a single monotone ramp bounded by ``DEPTH_ENC_MAX`` = 1.25 for any
(range, near) pair.  ``--obs-fourier L`` maps that to x = enc/1.25 in
[0, 1] and appends 2L channels

    sin(2^k pi x), cos(2^k pi x)     k = 0 .. L-1

(NeRF's gamma(x), Mildenhall 2020; the same construction as a diffusion
timestep embedding), so the trunk's first layer sees a basis in which two
depths a few thousandths apart are far apart.

The expansion lives in ``Policy.features``, in FRONT of conv[0], not in
the lidar kernel: the observation buffer, the rollout storage and every
recorded trajectory keep exactly one depth channel and only
``conv[0]``'s input width grows, in_ch -> in_ch + 2L.

COST (stated, not tested).  conv[0] on the 64x32 image is
``32x16 x 16 x 5x5 x in_ch`` = 204,800 MACs per sample at in_ch=1 and
2,662,400 at in_ch=13, i.e. +2.46M MACs against a whole-policy forward of
about 3.3M.  That is a large FRACTION of a very small number: the same
decision also costs 2,048 rays x up to 64 sphere-trace steps of lidar and
a physics tick, so the arm's launch measures the realised throughput both
ways rather than trusting this arithmetic.

    python -m pytest tests/python/test_obs_fourier.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.bc import N_SCALAR                          # noqa: E402
from train_fast import DEPTH_ENC_MAX, Policy             # noqa: E402

W, H = 8, 4                       # a tiny image; the encoding is per pixel
SEED = 1234


def _policy(**kw):
    in_ch = int(kw.pop("in_ch", 1))
    torch.manual_seed(SEED)
    return Policy(N_SCALAR + W * H * in_ch, W, H, emb=16, hidden=12,
                  in_ch=in_ch, **kw)


def _batch(n=5, in_ch=1, seed=7):
    g = torch.Generator().manual_seed(seed)
    scal = torch.randn(n, N_SCALAR, generator=g)
    # the depth channel spans its real range [0, DEPTH_ENC_MAX]
    img = torch.rand(n, W * H * in_ch, generator=g) * DEPTH_ENC_MAX
    return scal, img


class _Spy(torch.nn.Module):
    """Stands in for the conv trunk: records its input, emits emb zeros."""

    def __init__(self, emb=16):
        super().__init__()
        self.emb, self.seen = emb, None

    def forward(self, x):
        self.seen = x.detach().clone()
        return x.new_zeros(x.shape[0], self.emb)


def _trunk_input(policy, scal, img):
    """What conv[0] would have been handed for this observation."""
    spy = _Spy()
    real, policy.conv = policy.conv, spy
    try:
        with torch.no_grad():
            policy.features(scal, img)
    finally:
        policy.conv = real
    return spy.seen


# --------------------------------------------------------------------------
# 1. L = 0 is the pre-flag policy, byte for byte
# --------------------------------------------------------------------------
def test_l0_is_the_same_network_as_no_flag_at_all():
    off = _policy()                       # the call the trainer made before
    zero = _policy(obs_fourier=0)         # the flag, explicitly off
    a, b = off.state_dict(), zero.state_dict()
    assert list(a) == list(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    # same keys AND same values from the same seed: the flag drew no RNG


def test_l0_forward_is_bit_identical():
    off = _policy().eval()
    zero = _policy(obs_fourier=0).eval()
    zero.load_state_dict(off.state_dict())
    scal, img = _batch()
    with torch.no_grad():
        la, va = off(torch.cat([scal, img], 1))
        lb, vb = zero(torch.cat([scal, img], 1))
    assert torch.equal(la, lb) and torch.equal(va, vb)


def test_l0_builds_no_buffer_and_the_old_conv():
    p = _policy(obs_fourier=0)
    assert p.conv[0].in_channels == 1
    assert "fourier_freq" not in dict(p.named_buffers())
    assert p.obs_fourier == 0


# --------------------------------------------------------------------------
# 2. L = 6 shapes
# --------------------------------------------------------------------------
def test_l6_widens_only_conv0():
    off, six = _policy(), _policy(obs_fourier=6)
    assert off.conv[0].in_channels == 1
    assert six.conv[0].in_channels == 1 + 2 * 6
    a, b = off.state_dict(), six.state_dict()
    assert list(a) == list(b)                    # same keys, incl. no buffer
    for k in a:
        if k == "conv.0.weight":
            assert a[k].shape[1] == 1 and b[k].shape[1] == 13
        else:
            assert a[k].shape == b[k].shape, k


def test_l6_observation_is_unchanged_and_the_forward_runs():
    six = _policy(obs_fourier=6).eval()
    scal, img = _batch(n=3)                      # ONE depth channel, as before
    assert img.shape[1] == W * H                 # obs_dim did not grow
    with torch.no_grad():
        logits, value = six(torch.cat([scal, img], 1))
    assert value.shape == (3,)
    assert logits.shape[0] == 3
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_fourier_freq_is_2k_pi_and_not_in_the_state_dict():
    six = _policy(obs_fourier=6)
    f = six.fourier_freq
    assert f.shape == (6,)
    want = torch.tensor([math.pi * (2.0 ** k) for k in range(6)])
    assert torch.equal(f, want)
    assert "fourier_freq" not in six.state_dict()


@pytest.mark.parametrize("L", [1, 2, 6, 10])
def test_any_L_widens_by_2L(L):
    assert _policy(obs_fourier=L).conv[0].in_channels == 1 + 2 * L


def test_negative_L_is_refused():
    with pytest.raises(SystemExit):
        _policy(obs_fourier=-1)


# --------------------------------------------------------------------------
# 3. the channels ARE the closed form
# --------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 3, 6])
def test_channels_equal_the_closed_form(L):
    p = _policy(obs_fourier=L).eval()
    scal, img = _batch(n=4)
    im = _trunk_input(p, scal, img)                 # (N, 1 + 2L, H, W)
    assert im.shape == (4, 1 + 2 * L, H, W)
    depth = img.reshape(4, H, W, 1).permute(0, 3, 1, 2)
    assert torch.equal(im[:, :1], depth)            # channel 0 is untouched
    x = (depth / DEPTH_ENC_MAX).clamp(0.0, 1.0)
    for k in range(L):
        a = x * (math.pi * (2.0 ** k))
        assert torch.allclose(im[:, 1 + k:2 + k], torch.sin(a),
                              atol=1e-5), "sin band %d" % k
        assert torch.allclose(im[:, 1 + L + k:2 + L + k], torch.cos(a),
                              atol=1e-5), "cos band %d" % k


def test_x_is_clamped_into_0_1():
    """A pixel above DEPTH_ENC_MAX (no lidar makes one, but a hand-made
    observation can) must not wrap past the top of the encoding."""
    p = _policy(obs_fourier=2).eval()
    scal = torch.zeros(1, N_SCALAR)
    img = torch.full((1, W * H), 5.0)               # far above 1.25
    im = _trunk_input(p, scal, img)
    # x == 1 exactly: sin(pi) = 0, cos(pi) = -1, sin(2pi) = 0, cos(2pi) = 1
    assert torch.allclose(im[:, 1], torch.zeros_like(im[:, 1]), atol=1e-6)
    assert torch.allclose(im[:, 3], -torch.ones_like(im[:, 3]), atol=1e-6)
    assert torch.allclose(im[:, 2], torch.zeros_like(im[:, 2]), atol=1e-5)
    assert torch.allclose(im[:, 4], torch.ones_like(im[:, 4]), atol=1e-6)


# --------------------------------------------------------------------------
# 4. it composes with the other image channels (they are NOT encoded)
# --------------------------------------------------------------------------
def test_extra_image_channels_ride_through_unencoded():
    """--surf-mask 1 / --obs-potential put a second channel next to depth.
    The Fourier bands are a function of channel 0 only, and the second
    channel must arrive at the conv unchanged."""
    p = _policy(in_ch=2, obs_fourier=3).eval()
    assert p.conv[0].in_channels == 2 + 6
    scal, img = _batch(n=2, in_ch=2)
    im = _trunk_input(p, scal, img)
    nhwc = img.reshape(2, H, W, 2).permute(0, 3, 1, 2)
    assert torch.equal(im[:, :2], nhwc)
    x = (nhwc[:, :1] / DEPTH_ENC_MAX).clamp(0.0, 1.0)
    assert torch.allclose(im[:, 2:3], torch.sin(x * math.pi), atol=1e-6)
