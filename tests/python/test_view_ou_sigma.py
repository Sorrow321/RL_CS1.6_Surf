"""--view-ou-sigma: temporally correlated view exploration (round 40, pnOU).

The mechanism, as implemented in train_fast.py's rollout:

  * a per-ENV pre-tanh YAW offset ``c_e ~ N(0, sigma)`` lives in a static
    buffer ``ou_c`` of shape (N, NZ) with only column 0 non-zero;
  * the EXECUTED view command is ``view_from_z_t(z + c_e)``, while the z
    STORED in the PPO buffer is the un-offset draw ``z``;
  * ``c_e`` is redrawn ONLY on the rows of ``ended_acc`` - i.e. at episode
    starts - so it is a per-EPISODE behaviour constant, not per-decision
    noise.

What is pinned here:

(a) THE IMPORTANCE-WEIGHT IDENTITY. Storing the un-offset z makes the
    rollout's recorded log-prob EXACTLY the behaviour policy's log-density
    of the executed action: log N(z + c; mu + c, sigma) == log N(z; mu,
    sigma) for every c. PPO's ratio is therefore on-policy w.r.t.
    N(mu + c_e, sigma) with no correction term - which is the whole reason
    the per-episode version was preferred over a per-decision one.
(b) THE UPDATE'S RECOMPUTATION IS UNTOUCHED. logprob_entropy_view scores
    the stored z and is oblivious to the offset, bit for bit.
(c) THE REDRAW RULE. Over a scripted done-mask sequence the offset is
    constant inside an episode and changes exactly on the rows that ended;
    the PITCH column is never offset.
(d) SIGMA 0 IS OFF. The trainer's guard is a Python constant, so the added
    op does not exist; the arithmetic identity z_exec == z is pinned here.
(e) THE FLAG EXISTS on the CLI and defaults to 0.0.
"""
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import train_fast as tf                                          # noqa: E402


NZ = 2


def _mu_ls(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(n, NZ, generator=g)
    ls = torch.tensor([-3.21, -1.20])
    return mu, ls


def _yaw_offsets(n, sigma, gen):
    c = torch.zeros(n, NZ)
    c[:, 0] = torch.randn(n, generator=gen) * sigma
    return c


# ---------------------------------------------------------------- (a)

@pytest.mark.parametrize("sigma", [0.1, 0.3, 1.0, 3.0])
def test_offset_logp_identity(sigma):
    """log N(z + c; mu + c, sigma) == log N(z; mu, sigma): the stored z is
    the sufficient statistic, so no importance weight is needed."""
    mu, ls = _mu_ls(64, seed=7)
    g = torch.Generator().manual_seed(11)
    z = mu + ls.exp() * torch.randn(mu.shape, generator=g)
    c = _yaw_offsets(mu.shape[0], sigma, g)

    stored = tf.gauss_logp(z, mu, ls)                 # what the rollout logs
    executed = tf.gauss_logp(z + c, mu + c, ls)       # the behaviour density
    assert torch.allclose(stored, executed, atol=1e-4, rtol=0)


def test_offset_actually_moves_the_command():
    """The identity above must not be achieved by the offset doing nothing:
    the EXECUTED yaw command has to differ from the un-offset one."""
    mu, ls = _mu_ls(64, seed=7)
    g = torch.Generator().manual_seed(13)
    z = mu + ls.exp() * torch.randn(mu.shape, generator=g)
    c = _yaw_offsets(mu.shape[0], 0.3, g)

    plain = tf.view_from_z_t(z, 1.33, "velocity")
    shifted = tf.view_from_z_t(z + c, 1.33, "velocity")
    assert not torch.allclose(plain[:, 0], shifted[:, 0])
    # PITCH (column 1) is untouched: the offset lives in z column 0 only
    assert torch.equal(plain[:, 1], shifted[:, 1])
    # and the shift is worth degrees, not a rounding error
    assert float((shifted[:, 0] - plain[:, 0]).abs().mean()) > 0.2


# ---------------------------------------------------------------- (b)

def test_update_recomputation_is_oblivious_to_the_offset():
    """The update recomputes from the STORED z, so PPO's epoch-0 ratio is
    exactly 1 - the offset never enters it."""
    B = 32
    mu, ls = _mu_ls(B, seed=3)
    padded = torch.randn(B, tf.NACT, max(tf.NVEC))
    g = torch.Generator().manual_seed(5)
    z = mu + ls.exp() * torch.randn(mu.shape, generator=g)
    act = torch.zeros(B, tf.NACT, dtype=torch.long)

    # the rollout's joint log-prob, assembled the way sample_view does
    logp_cat, _ = tf.logprob_entropy_padded(padded[:, tf.N_VIEW:],
                                            act[:, tf.N_VIEW:], None)
    roll = logp_cat + tf.gauss_logp(z, mu, ls)

    new, _ent = tf.logprob_entropy_view(padded, act, mu, ls, z)
    assert torch.allclose(new, roll, atol=1e-5, rtol=0)
    ratio = (new - roll).exp()
    assert torch.allclose(ratio, torch.ones(B), atol=1e-5, rtol=0)


# -------------------------------------------------------------------- (c)

def test_offset_is_a_per_episode_constant():
    """The trainer's redraw expression, replayed against a scripted done
    mask: constant inside an episode, fresh on every row that ended, pitch
    never offset."""
    N = 8
    sig = torch.zeros(NZ)
    sig[0] = 0.3
    gen = torch.Generator().manual_seed(90210)
    c = torch.randn(N, NZ, generator=gen) * sig

    # env 0 ends at t=2 and t=5; env 1 never ends; env 7 ends every step
    dones = torch.zeros(7, N)
    dones[2, 0] = dones[5, 0] = 1.0
    dones[:, 7] = 1.0
    dones[3, 3] = 1.0

    hist = [c.clone()]
    for t in range(dones.shape[0]):
        c = torch.where(dones[t].unsqueeze(1) > 0,
                        torch.randn(c.shape, generator=gen) * sig, c)
        hist.append(c.clone())

    # pitch is never offset, at any time
    for h in hist:
        assert torch.all(h[:, 1] == 0.0)

    # env 1 never ended -> one constant for the whole rollout
    col = torch.stack([h[1, 0] for h in hist])
    assert torch.all(col == col[0])

    # env 0: constant over t = 0..2, a new value from t=3 on, another from 6
    e0 = torch.stack([h[0, 0] for h in hist])
    assert e0[0] == e0[1] == e0[2]
    assert e0[3] == e0[4] == e0[5]
    assert e0[6] == e0[7]
    assert e0[0] != e0[3] and e0[3] != e0[6]

    # env 7 ended every step -> every value distinct (i.i.d. draws)
    e7 = torch.stack([h[7, 0] for h in hist])
    assert len(set(e7.tolist())) == len(e7)


def test_the_offset_is_held_long_enough_to_matter():
    """CLAUDE.md's arithmetic: the surviving petrus manoeuvre is a 27.8 deg
    mean heading offset over 12-13 consecutive decisions, which is ~46
    sigma as an i.i.d. run at sigma_z = exp(-3.21). A HELD offset reaches
    the same mean at a few sigma of the OFFSET distribution, which is the
    entire point of the flag."""
    # d(off_warp)/du at u = 0, from the warp definition itself
    slope = float(tf.off_warp_t(torch.tensor([1e-4]))[0]) / 1e-4
    assert 3.0 < slope < 4.5                       # ~3.543 deg per unit u

    sigma_z = math.exp(-3.21)                      # measured view_std yaw
    n_dec = 12
    need_u = 27.8 / slope         # near u = 0 the tanh is the identity
    iid_sigmas = need_u / (sigma_z / math.sqrt(n_dec))
    assert iid_sigmas > 40.0                       # unreachable by magnitude

    held_sigmas = need_u / 0.3                     # the flag's suggested sigma
    assert held_sigmas < 30.0
    assert held_sigmas < iid_sigmas / 2.0


# -------------------------------------------------------------- (d) + (e)

def test_sigma_zero_is_the_identity():
    mu, ls = _mu_ls(16, seed=1)
    g = torch.Generator().manual_seed(2)
    z = mu + ls.exp() * torch.randn(mu.shape, generator=g)
    c = torch.zeros_like(mu)                       # sigma 0 -> c is exactly 0
    assert torch.equal(tf.view_from_z_t(z + c, 1.33, "velocity"),
                       tf.view_from_z_t(z, 1.33, "velocity"))


def test_flag_is_on_the_cli_and_defaults_to_zero():
    out = subprocess.run(
        [sys.executable, str(ROOT / "python" / "train_fast.py"), "--help"],
        capture_output=True, text=True)
    assert "--view-ou-sigma" in out.stdout
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--view-ou-sigma", type=float, default=0.0' in src
    # the executed-vs-stored split, so a refactor that stored z_exec fails
    assert "z_exec = z if ou_c is None else z + ou_c" in src
    assert "static_z.copy_(z)" in src


# ---------------------------------------------------------------- pnOU2

def test_the_identity_does_not_depend_on_the_correlation_time():
    """--view-ou-period resamples c_e MID-EPISODE. The on-policy identity
    survives because it is PER DECISION: log N(z_t + c_t; mu_t + c_t, sigma)
    == log N(z_t; mu_t, sigma) references c only at t, never at t-1 or t+1.
    Pinned here against a c that changes on every single step."""
    T, B = 40, 16
    g = torch.Generator().manual_seed(21)
    ls = torch.tensor([-3.21, -1.20])
    for t in range(T):
        mu = torch.randn(B, NZ, generator=g)
        z = mu + ls.exp() * torch.randn(B, NZ, generator=g)
        c = _yaw_offsets(B, 0.3, g)                  # a NEW c every step
        assert torch.allclose(tf.gauss_logp(z, mu, ls),
                              tf.gauss_logp(z + c, mu + c, ls),
                              atol=1e-4, rtol=0)


def test_period_countdown_holds_then_redraws():
    """The trainer's countdown rule replayed: an offset is held for exactly
    K decisions, and an episode start restarts the window early."""
    N, K = 6, 4
    sig = torch.zeros(NZ)
    sig[0] = 0.3
    gen = torch.Generator().manual_seed(7)
    c = torch.randn(N, NZ, generator=gen) * sig
    left = torch.full((N,), K, dtype=torch.int32)
    period = torch.full((N,), K, dtype=torch.int32)

    dones = torch.zeros(10, N)
    dones[5, 2] = 1.0                       # env 2's episode ends at t=5
    hist = [c[:, 0].clone()]
    for t in range(dones.shape[0]):
        redraw = dones[t] > 0
        left -= 1
        redraw = redraw | (left <= 0)
        left = torch.where(redraw, period, left)
        c = torch.where(redraw.unsqueeze(1),
                        torch.randn(c.shape, generator=gen) * sig, c)
        hist.append(c[:, 0].clone())
    h = torch.stack(hist)                   # (T+1, N)

    # env 0 never ends: held for K, then a new value, held for K, ...
    e0 = h[:, 0]
    for start in (0, 4, 8):
        blk = e0[start:start + K]
        assert torch.all(blk == blk[0]), f"not held across t={start}..{start+K}"
    assert e0[0] != e0[4] and e0[4] != e0[8]

    # env 2's end at t=5 redraws early and RESTARTS the window there
    e2 = h[:, 2]
    assert e2[5] != e2[6]                  # the end forced a fresh draw
    assert e2[6] == e2[7] == e2[8] == e2[9]   # then held K again


def test_period_zero_is_the_episode_scale_arm():
    """K = 0 is pnOU exactly: no countdown tensor, so nothing in the redraw
    path can differ from the arm that already ran."""
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--view-ou-period", type=int, default=0' in src
    # the countdown is allocated only above zero, and the redraw rule reads
    # `if ou_left is not None` - so at 0 the expression is pnOU's, op for op
    assert "if OU_PERIOD > 0:" in src
    assert "if ou_left is not None:" in src
    assert "ou_c = ou_sig_t = ou_left = ou_period_t = None" in src


def test_period_needs_a_sigma():
    out = subprocess.run(
        [sys.executable, str(ROOT / "python" / "train_fast.py"),
         "--map", str(ROOT / "maps" / "surf_petrus_lite.bsp"),
         "--view-continuous", "--view-ou-period", "16", "--steps", "1"],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "no offset to resample" in (out.stdout + out.stderr)


def test_K16_can_host_the_manoeuvre():
    """The sizing check for pnOU2. The surviving manoeuvre is 12-13
    CONSECUTIVE decisions; with a window of K, the share of windows that can
    contain a run of n entirely is (K - n + 1)/K. At K = 16 that is 25% for
    n = 13 - low but non-zero, which is the point of stating it. An episode
    (700-800 ticks = 175-200 decisions) is ~20x the manoeuvre, which is what
    pnOU held and why its offset could not be manoeuvre-shaped."""
    K, n = 16, 13
    assert (K - n + 1) / K == pytest.approx(0.25)
    ep_decisions = 750 / 4            # petrus ep_len_mean at act_every 4
    assert ep_decisions / n > 14      # pnOU's correlation time, in manoeuvres
