"""--quantiles: the distributional critic must be the PAPER's, and it must
warm-start onto the scalar checkpoint without moving a single value.

Four properties, all silent if they break, all of which decide whether the
one-hour ablation measures a DISTRIBUTIONAL CRITIC or measures a bug:

  1. quantile_huber_loss is QR-DQN Eq. 9-10 / IQN Eq. 3 exactly - the
     |tau - 1{u<0}| asymmetry, the Huber kappa knee, the /kappa, the
     midpoint taus, summed over i and meaned over the target atoms j.
     Checked against hand-computed numbers, not against itself.
  2. It really does regress the quantile function: fitted against a fixed
     distribution the N rows land on that distribution's tau_hat quantiles.
  3. N = 0 (flag off) and N = 1 are the scalar critic exactly - same
     state_dict shapes, bit-identical forward, and a value loss that is the
     baseline's 0.5 (v - ret)^2 in the Huber-quadratic region.
  4. quantilize_value_head makes a resumed quantile policy compute the
     checkpoint's values: the mean of the N rows IS the old scalar row, so
     the arm starts ON the baseline curve (xCTL: 24,307 at step 0) and any
     later divergence is the treatment.

Sources fetched and read for this file:
  Dabney et al. 2017, "Distributional RL with Quantile Regression"
    (arXiv 1710.10044) - Eq. 9 (Huber), Eq. 10 (rho), Alg. 1 (the sum over
    i of E_j), section 4 / Lemma 2 (the midpoints tau_hat_i), kappa = 1.
  Dabney et al. 2018, "Implicit Quantile Networks" (arXiv 1806.06923) -
    Eq. 3 (rho with the /kappa) and Eq. 4 (sum over i, 1/N' over j).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (N_SCALAR, Policy, QUANTILE_KAPPA,          # noqa: E402
                        quantile_huber_loss, quantile_midpoints,
                        quantile_value_loss, quantilize_value_head)

LW, LH = 16, 8
IMG = LW * LH
OBS = N_SCALAR + IMG


def make_policy(n_quant=0, seed=0):
    torch.manual_seed(seed)
    return Policy(OBS, LW, LH, emb=16, hidden=12, n_quant=n_quant)


def rand_obs(n=7, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, OBS, generator=g)


# --------------------------------------------------------------- 1. the loss
def test_midpoints_are_the_papers_tau_hat():
    """tau_i = i/N, tau_hat_i = (tau_{i-1} + tau_i)/2, 1 <= i <= N."""
    for n in (1, 2, 3, 8, 32, 200):
        taus = quantile_midpoints(n, dtype=torch.float64)
        grid = torch.arange(0, n + 1, dtype=torch.float64) / n      # tau_0..tau_N
        want = (grid[:-1] + grid[1:]) / 2.0
        assert torch.allclose(taus, want, atol=1e-15, rtol=0)
        assert taus.shape == (n,)
        # (2i-1)/2N, strictly inside (0, 1), symmetric about 1/2
        i = torch.arange(1, n + 1, dtype=torch.float64)
        assert torch.allclose(taus, (2 * i - 1) / (2 * n))
        assert float(taus[0]) > 0.0 and float(taus[-1]) < 1.0
        # sum tau_hat = N/2 exactly: this is why 2/N rescales the paper's
        # sum-over-i aggregation onto the scalar critic's 0.5 u^2
        assert float(taus.sum()) == pytest.approx(n / 2.0, abs=1e-12)


def test_quantile_huber_matches_hand_computed_values():
    """Four cases worked out by hand from Eq. 9-10.

    rho^k_tau(u) = |tau - 1{u<0}| * L_k(u) / k,
    L_k(u) = 0.5u^2 if |u| <= k else k(|u| - 0.5k),  u = target - theta.
    """
    # (a) N=2 (taus .25/.75), theta = [0, 0], target 0.5, kappa 1.
    #     u = +0.5 both -> quadratic: L = 0.5*0.25 = 0.125, no indicator.
    #     rho = .25*.125 + .75*.125 = 0.03125 + 0.09375 = 0.125
    q = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    t = torch.tensor([0.5], dtype=torch.float64)
    taus = quantile_midpoints(2, dtype=torch.float64)
    got = quantile_huber_loss(q, t, taus, kappa=1.0)
    assert got.shape == (1,)
    assert float(got) == pytest.approx(0.125, abs=1e-12)

    # (b) N=1 (tau .5), theta = 3, target 0, kappa 1: u = -3, LINEAR region,
    #     L = 1*(3 - 0.5) = 2.5, indicator fires -> w = |0.5 - 1| = 0.5,
    #     rho = 0.5 * 2.5 / 1 = 1.25
    q = torch.tensor([[3.0]], dtype=torch.float64)
    t = torch.tensor([0.0], dtype=torch.float64)
    taus = quantile_midpoints(1, dtype=torch.float64)
    got = quantile_huber_loss(q, t, taus, kappa=1.0)
    assert float(got) == pytest.approx(1.25, abs=1e-12)

    # (c) kappa = 2, same u = -3: L = 2*(3 - 1) = 4, /kappa -> 2.
    #     N=2: w = |0.25 - 1| = 0.75 and |0.75 - 1| = 0.25 -> 0.75*2 + 0.25*2
    q = torch.tensor([[3.0, 3.0]], dtype=torch.float64)
    taus = quantile_midpoints(2, dtype=torch.float64)
    got = quantile_huber_loss(q, t, taus, kappa=2.0)
    assert float(got) == pytest.approx(0.75 * 2.0 + 0.25 * 2.0, abs=1e-12)

    # (d) TWO target atoms -> mean over j, sum over i.
    #     theta = [0, 0], atoms {+0.5, -0.5}, kappa 1, taus .25/.75:
    #       atom +0.5: as in (a) per-i: .25*.125, .75*.125
    #       atom -0.5: indicator fires: .75*.125, .25*.125
    #     mean over j per i: .5*.125 each -> sum over i = 0.125
    q = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    t = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    got = quantile_huber_loss(q, t, taus, kappa=1.0)
    assert float(got) == pytest.approx(2 * 0.5 * 0.125, abs=1e-12)


def test_asymmetric_weighting_is_the_only_thing_ordering_the_quantiles():
    """|tau - 1{u<0}|: a HIGH tau is charged little for under-shooting and a
    lot for over-shooting, which is what drags row i to the tau_i-quantile."""
    taus = torch.tensor([0.1, 0.9], dtype=torch.float64)
    err = 0.5                                   # inside the quadratic region
    lo_under = quantile_huber_loss(torch.tensor([[-err]], dtype=torch.float64),
                                   torch.tensor([0.0], dtype=torch.float64),
                                   taus[:1], kappa=1.0)   # u > 0 -> w = tau
    lo_over = quantile_huber_loss(torch.tensor([[err]], dtype=torch.float64),
                                  torch.tensor([0.0], dtype=torch.float64),
                                  taus[:1], kappa=1.0)    # u < 0 -> w = 1-tau
    assert float(lo_under) == pytest.approx(0.1 * 0.5 * err * err)
    assert float(lo_over) == pytest.approx(0.9 * 0.5 * err * err)
    hi_under = quantile_huber_loss(torch.tensor([[-err]], dtype=torch.float64),
                                   torch.tensor([0.0], dtype=torch.float64),
                                   taus[1:], kappa=1.0)
    assert float(hi_under) == pytest.approx(0.9 * 0.5 * err * err)
    # tau = 0.5 is symmetric; the pair (tau, 1-tau) mirrors
    assert float(lo_under) == pytest.approx(
        float(quantile_huber_loss(torch.tensor([[err]], dtype=torch.float64),
                                  torch.tensor([0.0], dtype=torch.float64),
                                  torch.tensor([0.9], dtype=torch.float64),
                                  kappa=1.0)))


def test_huber_knee_is_at_kappa_and_the_gradient_is_capped():
    """Eq. 9: quadratic inside kappa, linear outside, C1 at the knee - which
    is the whole point (an outlier return cannot blow up the critic)."""
    taus = torch.tensor([0.5], dtype=torch.float64)
    zero = torch.tensor([0.0], dtype=torch.float64)

    def loss(theta, kappa=1.0):
        q = torch.tensor([[theta]], dtype=torch.float64, requires_grad=True)
        out = quantile_huber_loss(q, zero, taus, kappa=kappa)
        out.backward()
        return out.detach().item(), q.grad[0, 0].item()

    v_in, g_in = loss(0.5)                       # |u| = 0.5 < kappa
    assert v_in == pytest.approx(0.5 * 0.5 * 0.25)
    assert g_in == pytest.approx(0.5 * 0.5)      # d/dtheta = w * u ... sign
    v_knee, g_knee = loss(1.0)                   # |u| = kappa exactly
    assert v_knee == pytest.approx(0.5 * 0.5)
    v_out, g_out = loss(50.0)                    # far outside
    assert v_out == pytest.approx(0.5 * (50.0 - 0.5))
    assert abs(g_out) == pytest.approx(0.5)      # capped at w * kappa
    assert abs(g_out) == pytest.approx(abs(g_knee))
    # kappa scales the knee, not the small-error curvature (because of /kappa)
    assert loss(0.5, kappa=4.0)[0] == pytest.approx(v_in / 4.0)


def _fit_quantiles(atoms, taus, kappa, n):
    """Full-batch fit of N free quantile rows against a fixed distribution."""
    theta = torch.zeros(1, n, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    for it in range(4000):
        if it == 2000:
            for g in opt.param_groups:      # anneal: at small kappa rho is
                g["lr"] = 0.002             # piecewise LINEAR, so Adam orbits
        opt.zero_grad()                     # the optimum at radius ~lr
        quantile_huber_loss(theta, atoms, taus, kappa=kappa).mean().backward()
        opt.step()
    return theta.detach().squeeze(0)


def test_it_actually_fits_the_quantile_function():
    """The property the whole paper is about: minimising this loss against a
    distribution puts row i at that distribution's tau_hat_i quantile.

    Checked at small kappa, where rho is the pure pinball loss (the paper's
    qr-dqn-0) and the minimiser IS the quantile. At kappa = 1 (qr-dqn-1, what
    we run) the quadratic knee deliberately pulls the rows toward the centre:
    that is the Huber's stability-for-bias trade, and the second half of this
    test pins it down so nobody later reads it as a bug.
    """
    n = 8
    taus = quantile_midpoints(n, dtype=torch.float64)
    # a FIXED distribution (no sampling noise): 2000 atoms of a standard
    # normal, laid out on its own inverse CDF
    grid = torch.linspace(0.00025, 0.99975, 2000, dtype=torch.float64)
    atoms = torch.distributions.Normal(0.0, 1.0).icdf(grid).unsqueeze(0)
    want = torch.quantile(atoms.squeeze(0), taus)

    sharp = _fit_quantiles(atoms, taus, 0.01, n)
    assert torch.allclose(sharp, want, atol=0.02), f"{sharp} vs {want}"
    # ORDERED, which is what makes their mean meaningful
    assert bool((sharp[1:] > sharp[:-1]).all())
    # and the mean of the fitted rows is the distribution's mean (0 here) -
    # the number GAE actually consumes
    assert float(sharp.mean()) == pytest.approx(0.0, abs=0.01)

    huber = _fit_quantiles(atoms, taus, 1.0, n)
    assert bool((huber[1:] > huber[:-1]).all())
    assert float(huber.mean()) == pytest.approx(0.0, abs=0.01)
    assert float(huber.abs().max()) < float(sharp.abs().max())   # shrunk
    assert torch.allclose(huber, want, atol=0.35)


# ------------------------------------------- 2. the scalar critic, unchanged
def test_flag_off_is_the_scalar_model_byte_for_byte():
    p0 = make_policy(n_quant=0)
    assert p0.n_quant == 0
    assert tuple(p0.value_head.weight.shape) == (1, 12)
    assert tuple(p0.value_head.bias.shape) == (1,)
    # every existing checkpoint still loads, strictly
    p0b = make_policy(n_quant=0, seed=1)
    p0b.load_state_dict(p0.state_dict(), strict=True)
    obs = rand_obs()
    with torch.no_grad():
        logits, v = p0(obs)
    assert v.shape == (obs.shape[0],)
    # ... and it is exactly value_head(vf(f)).squeeze(-1), the shipped line
    with torch.no_grad():
        f = torch.cat([obs[:, p0.feat_idx],
                       p0.conv(obs[:, N_SCALAR:].reshape(-1, LH, LW, 1)
                               .permute(0, 3, 1, 2))], dim=1)
        ref = p0.value_head(p0.vf(f)).squeeze(-1)
    assert torch.equal(v, ref)


def test_n_equals_one_is_bit_identical_to_the_scalar_critic():
    """Same seed, N=1 vs N=0: identical shapes, identical init, identical
    numbers. The mean over a length-1 axis is the identity."""
    p0, p1 = make_policy(n_quant=0, seed=5), make_policy(n_quant=1, seed=5)
    assert tuple(p1.value_head.weight.shape) == tuple(p0.value_head.weight.shape)
    for (k0, t0), (k1, t1) in zip(p0.state_dict().items(),
                                  p1.state_dict().items()):
        assert k0 == k1 and torch.equal(t0, t1)
    obs = rand_obs()
    with torch.no_grad():
        l0, v0 = p0(obs)
        l1, v1 = p1(obs)
    assert torch.equal(l0, l1)
    assert torch.equal(v0, v1)
    with torch.no_grad():
        q1 = p1.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:],
                              quantiles=True)[1]
    assert q1.shape == (obs.shape[0], 1)
    assert torch.equal(q1.squeeze(-1), v0)


def test_value_loss_reduces_to_the_baselines_mse_term():
    """With N=1 and |u| <= kappa the arm's value loss IS 0.5 (v - ret)^2 -
    the same term, the same scale, so --vf keeps its meaning."""
    torch.manual_seed(0)
    v = torch.randn(64, 1, dtype=torch.float64) * 0.1
    ret = torch.randn(64, dtype=torch.float64) * 0.1
    taus = quantile_midpoints(1, dtype=torch.float64)
    got = quantile_value_loss(v, ret, taus, kappa=1.0)
    want = 0.5 * (v.squeeze(-1) - ret).pow(2).mean()
    assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_value_loss_is_scale_matched_for_every_n():
    """N identical rows (which is exactly how the arm STARTS) cost what the
    scalar critic costs, for any N: sum_i |tau_hat_i - 1{u<0}| = N/2."""
    for n in (1, 2, 8, 32, 200):
        taus = quantile_midpoints(n, dtype=torch.float64)
        for u in (0.03, -0.4, 0.9):
            q = torch.full((5, n), 1.0, dtype=torch.float64)
            ret = torch.full((5,), 1.0 + u, dtype=torch.float64)
            got = quantile_value_loss(q, ret, taus, kappa=1.0)
            assert float(got) == pytest.approx(0.5 * u * u, rel=1e-12)


# ---------------------------------------------- 3. the warm-start operation
def _fake_ckpt(policy):
    """A checkpoint with real Adam moments: one backward + step populates
    exp_avg/exp_avg_sq for every parameter, exactly like a resumed run."""
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    obs = rand_obs(11, seed=9)
    logits, v = policy(obs)
    (logits.square().mean() + v.square().mean()).backward()
    opt.step()
    return {"policy": policy.state_dict(), "optimizer": opt.state_dict(),
            "global_step": 3_782_737_920}


def test_warm_start_is_function_identical():
    torch.manual_seed(11)
    scalar = make_policy(n_quant=0, seed=11)
    with torch.no_grad():                        # a NON-trivial value head
        scalar.value_head.weight.normal_(0.0, 0.5)
        scalar.value_head.bias.fill_(0.37)
    ck = _fake_ckpt(scalar)
    row_w = ck["policy"]["value_head.weight"].clone()
    row_b = ck["policy"]["value_head.bias"].clone()

    nq = 32                                      # Sophy's 32
    qp = make_policy(n_quant=nq, seed=11)
    n_patched = quantilize_value_head(ck, qp)
    # weight + bias + (exp_avg, exp_avg_sq) for each of the two
    assert n_patched == 6
    qp.load_state_dict(ck["policy"], strict=True)
    opt_q = torch.optim.Adam(qp.parameters(), lr=3e-4, eps=1e-5)
    opt_q.load_state_dict(ck["optimizer"])       # shapes must line up

    # every quantile row IS the trained scalar row, bit for bit
    assert torch.equal(qp.value_head.weight, row_w.repeat(nq, 1))
    assert torch.equal(qp.value_head.bias, row_b.repeat(nq))

    obs = rand_obs(64, seed=21)
    with torch.no_grad():
        _, v_scalar = scalar(obs)
        _, v_quant = qp(obs)
        q_rows = qp.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:],
                                  quantiles=True)[1]
    assert q_rows.shape == (64, nq)
    # the MEAN of the quantiles is the checkpoint's value (float32 mean of 32
    # identical values, so exact to a couple of ULPs, not "close enough")
    assert torch.allclose(v_quant, v_scalar, rtol=1e-6, atol=1e-7)
    assert float((v_quant - v_scalar).abs().max()) < 1e-5
    # all rows equal at load: the distribution is a point mass on the old V
    assert float((q_rows - q_rows[:, :1]).abs().max()) == 0.0


def test_warm_start_moments_are_replicated_not_zeroed():
    scalar = make_policy(n_quant=0, seed=4)
    ck = _fake_ckpt(scalar)
    params = list(scalar.parameters())
    vh_idx = [i for i, p in enumerate(params)
              if p is scalar.value_head.weight or p is scalar.value_head.bias]
    before = {i: {k: ck["optimizer"]["state"][i][k].clone()
                  for k in ("exp_avg", "exp_avg_sq")} for i in vh_idx}
    assert any(float(t["exp_avg"].abs().max()) > 0 for t in before.values())

    qp = make_policy(n_quant=8, seed=4)
    quantilize_value_head(ck, qp)
    for i in vh_idx:
        for k in ("exp_avg", "exp_avg_sq"):
            got = ck["optimizer"]["state"][i][k]
            assert got.shape[0] == 8
            assert torch.equal(got, before[i][k].repeat(
                8, *([1] * (before[i][k].dim() - 1))))


def test_warm_start_is_a_noop_where_it_should_be():
    # flag off: never touches the checkpoint
    scalar = make_policy(n_quant=0, seed=6)
    ck = _fake_ckpt(scalar)
    assert quantilize_value_head(ck, make_policy(n_quant=0, seed=6)) == 0
    assert tuple(ck["policy"]["value_head.weight"].shape) == (1, 12)
    # resuming a run that is ALREADY quantile: nothing to expand
    qp = make_policy(n_quant=8, seed=6)
    ck_q = _fake_ckpt(qp)
    assert quantilize_value_head(ck_q, make_policy(n_quant=8, seed=6)) == 0
    assert tuple(ck_q["policy"]["value_head.weight"].shape) == (8, 12)


def test_warm_start_refuses_a_head_it_cannot_map():
    qp8 = make_policy(n_quant=8, seed=7)
    ck8 = _fake_ckpt(qp8)
    with pytest.raises(SystemExit):
        quantilize_value_head(ck8, make_policy(n_quant=32, seed=7))


def test_default_kappa_is_the_papers():
    assert QUANTILE_KAPPA == 1.0
