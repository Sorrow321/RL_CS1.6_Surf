"""--route: the lookahead fan must be additive, not disruptive.

Three properties, all of which are silent if they break and all of which
decide whether a one-hour ablation measures the FEATURE or measures a
re-initialisation shock:

  1. route_dim=0 is the pre-route model exactly - same state_dict keys, same
     shapes, same numbers. Otherwise every existing checkpoint is stranded.
  2. widen_for_route + load_state_dict makes a route-widened policy compute
     the OLD function on its first forward, whatever the fan reports. The arm
     then starts ON the baseline curve (docs: xCTL 24,307 at step 0) and any
     later divergence is the treatment.
  3. The fan itself is finite, clamped, ego-framed and agrees in sign with
     the velocity scalars the policy already reads (src/env.c:247).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.route import RouteLine, resample_polyline       # noqa: E402
from train_fast import N_SCALAR, Policy, widen_for_route     # noqa: E402

LW, LH = 16, 8
IMG = LW * LH


def make_policy(route_dim=0, critic_only=False, seed=0):
    torch.manual_seed(seed)
    return Policy(N_SCALAR + route_dim + IMG, LW, LH, emb=16, hidden=12,
                  route_dim=route_dim, route_critic_only=critic_only)


def straight_route(device=None):
    leg = np.linspace(0.0, 6000.0, 300)
    xyz = np.stack([leg, np.zeros_like(leg), np.zeros_like(leg)], 1)
    pts, _ = resample_polyline(xyz, 128.0)
    return RouteLine(pts, 128.0, offsets=(0.5, 1.0, 2.0), device=device)


# ---------------------------------------------------------------- property 1
def test_route_dim_zero_is_the_old_model():
    a, b = make_policy(0, seed=3), make_policy(0, seed=3)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    assert a.scal_dim == N_SCALAR
    assert a.pi[0].in_features == a.vf[0].in_features
    obs = torch.randn(4, N_SCALAR + IMG)
    la, va = a(obs)
    lb, vb = b(obs)
    assert torch.equal(la, lb) and torch.equal(va, vb)


# ---------------------------------------------------------------- property 2
@pytest.mark.parametrize("critic_only", [False, True])
def test_widen_is_function_identical(critic_only):
    """A pre-route checkpoint, grown onto a route policy, must reproduce its
    own logits and value bit-for-bit no matter what the fan says."""
    base = make_policy(0, seed=11)
    opt = torch.optim.Adam(base.parameters(), lr=1e-4)
    # give the optimizer real (non-empty) moment tensors to pad
    obs0 = torch.randn(3, N_SCALAR + IMG)
    base(obs0)[1].sum().backward()
    opt.step()
    ck = {"policy": base.state_dict(), "optimizer": opt.state_dict()}

    R = 9
    wide = make_policy(R, critic_only=critic_only, seed=99)   # different init
    n = widen_for_route(ck, wide)
    assert n > 0, "nothing was padded - the widening never ran"
    wide.load_state_dict(ck["policy"])
    torch.optim.Adam(wide.parameters(), lr=1e-4).load_state_dict(ck["optimizer"])

    scal = torch.randn(5, N_SCALAR)
    img = torch.randn(5, IMG)
    want_l, want_v = base(torch.cat([scal, img], 1))
    for fan in (torch.zeros(5, R), torch.randn(5, R) * 3.0,
                torch.full((5, R), -4.0)):
        got_l, got_v = wide(torch.cat([scal, fan, img], 1))
        assert torch.allclose(got_l, want_l, atol=0, rtol=0), \
            "route columns are not zero: the resumed policy changed behaviour"
        assert torch.allclose(got_v, want_v, atol=0, rtol=0)


def test_widen_refuses_a_shrink():
    wide_ck = {"policy": make_policy(9).state_dict(), "optimizer": {}}
    with pytest.raises(SystemExit):
        widen_for_route(wide_ck, make_policy(0))


def test_critic_only_hides_the_fan_from_the_actor():
    p = make_policy(9, critic_only=True, seed=5)
    scal, img = torch.randn(4, N_SCALAR), torch.randn(4, IMG)
    l0, v0 = p(torch.cat([scal, torch.zeros(4, 9), img], 1))
    l1, v1 = p(torch.cat([scal, torch.randn(4, 9), img], 1))
    assert torch.equal(l0, l1), "actor saw the fan under --route-critic-only"
    assert not torch.equal(v0, v1), "critic did NOT see the fan"


def test_symmetric_feeds_both():
    p = make_policy(9, critic_only=False, seed=5)
    scal, img = torch.randn(4, N_SCALAR), torch.randn(4, IMG)
    l0, _ = p(torch.cat([scal, torch.zeros(4, 9), img], 1))
    l1, _ = p(torch.cat([scal, torch.randn(4, 9), img], 1))
    assert not torch.equal(l0, l1), "actor did not see the fan"


# ---------------------------------------------------------------- property 3
def test_fan_is_ego_framed_and_finite():
    r = straight_route()
    o = torch.tensor([[1000.0, 0.0, 0.0]])
    on = r.features(o, torch.tensor([0.0]), torch.tensor([1200.0]))[0]
    # facing along the route: forward ~1, lateral ~0 at every horizon
    assert on.shape[0] == r.n_features == 12
    for k in range(1, 4):
        assert abs(float(on[3 * k]) - 1.0) < 0.05
        assert abs(float(on[3 * k + 1])) < 0.05
    # yawed +90 deg the route sits to the RIGHT: lateral strongly negative,
    # the same sign convention o[1] = -vx*sin + vy*cos gives velocity
    right = r.features(o, torch.tensor([90.0]), torch.tensor([1200.0]))[0]
    assert float(right[4]) < -0.9 and abs(float(right[3])) < 0.05
    # nothing escapes the clamp, including nonsense states
    junk = r.features(torch.tensor([[1e7, -1e7, 1e6], [0.0, 0.0, 0.0]]),
                      torch.tensor([712.0, -9.0]), torch.tensor([1e9, 0.0]))
    assert torch.isfinite(junk).all() and float(junk.abs().max()) <= 4.0


def test_speed_floor_keeps_the_fan_open_at_rest():
    r = straight_route()
    o = torch.tensor([[1000.0, 0.0, 0.0]])
    rest = r.features(o, torch.tensor([0.0]), torch.tensor([0.0]))[0]
    # a standing agent still sees route AHEAD, not a fan collapsed onto itself
    assert float(rest[3]) > 0.5
    assert float(rest[9]) > 0.5


def test_feature_count_matches_offsets():
    for n in (1, 4, 8, 16):
        r = RouteLine(np.zeros((4, 3), np.float32), 128.0,
                      offsets=tuple(range(1, n + 1)))
        assert r.n_features == 3 * (n + 1)
