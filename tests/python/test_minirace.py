"""Linesight temporal mini-race (--minirace): the return arithmetic.

The mechanism under test (github.com/Linesight-RL/linesight):

  * rollouts are scored over a 7 s WINDOW (140 actions at their 20 Hz);
  * the elapsed-time-inside-the-window is a scalar of the observation;
  * the gamma schedule ends at exactly 1.0, so inside a window there is no
    discount at all and the return is just -elapsed_time + progress;
  * the window edge is TERMINAL - no bootstrap past it;
  * stored transitions get a random horizon clock, drawn with an abs() fold
    that oversamples clocks with the full window still ahead of them.

The two properties that decide whether the arm measures the paper or a bug:
with the flag off the advantages must be BIT-identical to the control, and
inside a window the return must equal the plain undiscounted sum of rewards
to the edge.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (                                    # noqa: E402
    N_SCALAR, Policy, gae_advantages, minirace_feature, minirace_phase,
    minirace_window, widen_for_route,
)

# Linesight's own constants, config_files/config.py
LS_MS_PER_ACTION = 50
LS_WINDOW_MS = 7000
LS_WINDOW_ACTIONS = 140
LS_LONG = 40            # oversample_long_term_steps
LS_MAXT = 5             # oversample_maximum_term_steps


def _reference_gae(rew, val, done, last_val, g_eff, lam):
    """The pre-minirace loop, transcribed from train_fast.py before the
    window existed. Anything that is not bit-identical to this with the flag
    off changes the control, and then the arm is not measuring the paper."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    lastgae = torch.zeros(rew.shape[1])
    for t in reversed(range(T)):
        nextval = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + g_eff * nextval * nonterm - val[t]
        lastgae = delta + g_eff * lam * nonterm * lastgae
        adv[t] = lastgae
    return adv


# --------------------------------------------------------------------------
# the window, in seconds and in decisions
# --------------------------------------------------------------------------

def test_window_seconds_survive_the_decision_rate():
    """The horizon arithmetic that already wrecked one arm: --gamma is PER
    PHYSICS TICK and the GAE raises it to ticks-per-decision itself, so a
    horizon in seconds must not move when --act-every does."""
    # this project's baseline: act_every 3 => 3 ticks per decision
    assert minirace_window(7.0, 3) == 233           # 6.99 s
    assert abs(233 * 3 / 100.0 - 7.0) < 0.02
    # halving the decision rate doubles the DECISION count and leaves the
    # window the same length in seconds
    assert minirace_window(7.0, 6) == 117
    assert abs(117 * 6 / 100.0 - 7.0) < 0.03
    assert minirace_window(7.0, 1) == 700
    # and Linesight's own numbers reproduce: 7000 ms at 50 ms per action
    assert minirace_window(LS_WINDOW_MS / 1000.0,
                           LS_MS_PER_ACTION // 10) == LS_WINDOW_ACTIONS


def test_baseline_gamma_is_a_twenty_second_horizon():
    """Stated so the ledger's number is checked, not remembered: 0.9995 per
    tick at 100 ticks/s is 2,000 ticks = 20 s, and it does NOT depend on
    act_every. The mini-race replaces that with 7 s."""
    assert 1.0 / (1.0 - 0.9995) == pytest.approx(2000.0)
    assert 2000.0 / 100.0 == pytest.approx(20.0)
    # the survey's "cheap probe" equivalent of a 7 s window, for the record
    assert 1.0 - 1.0 / 700.0 == pytest.approx(0.998571, abs=1e-6)


def test_window_is_off_by_default():
    assert minirace_window(None, 3) == 0
    assert minirace_window(0.0, 3) == 0


# --------------------------------------------------------------------------
# the clock feature
# --------------------------------------------------------------------------

def test_clock_feature_matches_linesight_normalization():
    """float_inputs_mean[0] = float_inputs_std[0] = duration/2, so the clock
    reaches the network as 2t/H - 1."""
    H = 233
    clk = torch.tensor([0, H // 2, H - 1], dtype=torch.long)
    got = minirace_feature(clk, H)
    assert got[0].item() == pytest.approx(-1.0)
    assert got[1].item() == pytest.approx(0.0, abs=2.0 / H)
    assert got[2].item() < 1.0
    assert got[2].item() == pytest.approx(1.0 - 2.0 / H)
    # start of a window is exactly -1: what the eval rows are filled with
    assert minirace_feature(torch.zeros(4, dtype=torch.long), H).tolist() \
        == [-1.0] * 4


# --------------------------------------------------------------------------
# the random horizon clock
# --------------------------------------------------------------------------

def test_phase_draw_reproduces_linesight_support_and_fold():
    """abs(randint(-35, 145)) - 5, clipped at 0, over their constants: the
    support is 0..139 and the fold doubles the band below 31 with an 11x
    spike at 0."""
    torch.manual_seed(0)
    t = minirace_phase(400_000, LS_WINDOW_ACTIONS, LS_LONG, LS_MAXT)
    assert int(t.min()) == 0
    assert int(t.max()) == LS_WINDOW_ACTIONS - 1          # 139, never the edge
    n = t.numel()
    # exact multinomial weights of the fold: 180 equally likely draws
    p0 = (1 + 2 * LS_MAXT) / 180.0                        # 11/180 at t == 0
    pmid = 2 / 180.0                                      # doubled band
    ptail = 1 / 180.0                                     # unfolded tail
    assert float((t == 0).sum()) / n == pytest.approx(p0, rel=0.05)
    assert float((t == 10).sum()) / n == pytest.approx(pmid, rel=0.12)
    assert float((t == 100).sum()) / n == pytest.approx(ptail, rel=0.15)
    # the fold's whole point: clocks with the full window ahead are the
    # oversampled ones
    assert float((t == 0).sum()) > 5 * float((t == 100).sum())


def test_phase_draw_never_lands_on_the_edge():
    """A clock of H would be a zero-length window: a transition with no
    future and no reward, which is not a training signal."""
    torch.manual_seed(1)
    for H in (1, 2, 7, 233):
        t = minirace_phase(20_000, H, max(1, 2000 // 30), max(1, 250 // 30))
        assert int(t.min()) >= 0
        assert int(t.max()) <= H - 1


# --------------------------------------------------------------------------
# the return arithmetic - the part that decides the arm
# --------------------------------------------------------------------------

def _fixture(T=24, N=5, seed=7):
    g = torch.Generator().manual_seed(seed)
    rew = torch.randn(T, N, generator=g)
    val = torch.randn(T, N, generator=g)
    last_val = torch.randn(N, generator=g)
    done = (torch.rand(T, N, generator=g) < 0.06).float()
    return rew, val, done, last_val


def test_flag_off_is_bit_identical_to_the_control():
    rew, val, done, last_val = _fixture()
    g_eff, lam = 0.9995 ** 3, 0.95
    ref = _reference_gae(rew, val, done, last_val, g_eff, lam)
    got = gae_advantages(rew, val, done, last_val, g_eff, lam, b_wend=None)
    assert torch.equal(got, ref), "minirace off must not move the control"


def test_all_zero_window_flags_are_bit_identical_to_the_control():
    """The window edge tensor is threaded through the loop even when no
    window ever ends; multiplying by (1 - 0) must not perturb a single bit."""
    rew, val, done, last_val = _fixture()
    g_eff, lam = 0.9995 ** 3, 0.95
    ref = _reference_gae(rew, val, done, last_val, g_eff, lam)
    got = gae_advantages(rew, val, done, last_val, g_eff, lam,
                         b_wend=torch.zeros_like(done))
    assert torch.equal(got, ref)


def test_inside_a_window_the_return_is_the_undiscounted_sum_to_the_edge():
    """The paper's claim, literally: gamma 1, window edge terminal, so the
    return is -elapsed_time + progress with no discount. At lambda = 1 the
    GAE target ret = adv + V is exactly sum(rewards) to the edge, whatever
    the value function says."""
    T, N = 12, 3
    torch.manual_seed(3)
    rew = torch.randn(T, N)
    val = torch.randn(T, N) * 10.0        # deliberately wrong: must not leak
    last_val = torch.randn(N) * 10.0
    done = torch.zeros(T, N)
    wend = torch.zeros(T, N)
    wend[7, :] = 1.0                      # every env's window ends at t = 7
    adv = gae_advantages(rew, val, done, last_val, 1.0, 1.0, b_wend=wend)
    ret = adv + val
    for t in range(8):
        assert torch.allclose(ret[t], rew[t:8].sum(0), atol=1e-5), t


def test_a_window_edge_blocks_the_bootstrap():
    """Nothing past the edge may reach a transition before it. Change every
    value AFTER the edge and the returns before it must not move by one bit
    of float32."""
    T, N = 10, 4
    torch.manual_seed(4)
    rew = torch.randn(T, N)
    val = torch.randn(T, N)
    last_val = torch.randn(N)
    done = torch.zeros(T, N)
    wend = torch.zeros(T, N)
    wend[5, :] = 1.0
    a1 = gae_advantages(rew, val, done, last_val, 1.0, 0.95, b_wend=wend)
    val2 = val.clone()
    val2[6:] += 1000.0
    a2 = gae_advantages(rew, val2, done, last_val + 1000.0, 1.0, 0.95,
                        b_wend=wend)
    assert torch.equal(a1[:6], a2[:6])
    assert not torch.equal(a1[6:], a2[6:])


def test_window_edge_and_episode_end_compose():
    """An env can hit both on the same decision; the result must be one
    terminal, not a doubled one."""
    T, N = 6, 2
    rew = torch.ones(T, N)
    val = torch.full((T, N), 3.0)
    last_val = torch.full((N,), 3.0)
    done = torch.zeros(T, N)
    wend = torch.zeros(T, N)
    done[2, 0] = 1.0
    wend[2, 0] = 1.0                       # both at once
    wend[2, 1] = 1.0                       # window edge only
    adv = gae_advantages(rew, val, done, last_val, 1.0, 1.0, b_wend=wend)
    ret = adv + val
    # both envs see the same thing: reward at t=2 collected, nothing after
    assert ret[2, 0].item() == pytest.approx(1.0)
    assert ret[2, 1].item() == pytest.approx(1.0)
    assert ret[0, 0].item() == pytest.approx(3.0)


def test_the_windowed_horizon_is_shorter_than_the_baseline_discount():
    """The whole point of the arm, as a number: at gamma 0.9995 per tick a
    reward 20 s away still carries e^-1 of its weight; inside a 7 s window a
    reward past the edge carries exactly zero, and one inside carries one."""
    T, N = 800, 1
    rew = torch.zeros(T, N)
    rew[400] = 1.0                          # a reward 400 decisions = 12 s away
    val = torch.zeros(T, N)
    last_val = torch.zeros(N)
    done = torch.zeros(T, N)
    base = gae_advantages(rew, val, done, last_val, 0.9995 ** 3, 1.0)
    assert base[0, 0].item() == pytest.approx((0.9995 ** 3) ** 400, rel=1e-3)
    assert base[0, 0].item() > 0.5          # 20 s horizon still feels it
    wend = torch.zeros(T, N)
    for t in range(232, T, 233):
        wend[t] = 1.0                       # 7 s windows
    mr = gae_advantages(rew, val, done, last_val, 1.0, 1.0, b_wend=wend)
    assert mr[0, 0].item() == 0.0           # past the edge: gone entirely
    assert mr[300, 0].item() == pytest.approx(1.0)   # inside: undiscounted


# --------------------------------------------------------------------------
# the observation widening - function-identical at step 0
# --------------------------------------------------------------------------

def _mk(mr_dim, mr_actor=False, route_dim=0, w=8, h=4):
    torch.manual_seed(11)
    return Policy(N_SCALAR + route_dim + mr_dim + w * h, lidar_w=w, lidar_h=h,
                  emb=16, hidden=12, route_dim=route_dim, mr_dim=mr_dim,
                  mr_actor=mr_actor)


def test_scal_dim_and_obs_dim_account_for_the_clock():
    p = _mk(1)
    assert p.scal_dim == N_SCALAR + 1
    assert p.mr_dim == 1 and p.mr_actor is False


def test_widened_policy_is_function_identical_at_step_zero():
    """The step-0 requirement, the same one --route has to meet: zero-padding
    the first Linear's trailing columns makes the resumed policy compute the
    baseline's function exactly, so every later divergence is the window and
    not a re-initialisation shock."""
    w, h = 8, 4
    base = _mk(0, w=w, h=h)
    wide = _mk(1, w=w, h=h)
    ck = {"policy": {k: v.clone() for k, v in base.state_dict().items()},
          "optimizer": {"state": {}}}
    n = widen_for_route(ck, wide)
    # ONE tensor: vf.0.weight. The actor does not read the clock by default,
    # so its first Linear does not grow at all and the resumed POLICY is not
    # merely function-identical, it is the stuck checkpoint's weights
    # untouched.
    assert n == 1, f"expected vf.0.weight padded, got {n}"
    wide.load_state_dict(ck["policy"])
    assert float(wide.vf[0].weight[:, -1].detach().abs().sum()) == 0.0
    torch.manual_seed(5)
    core = torch.randn(6, N_SCALAR)
    img = torch.rand(6, w * h)
    o_base = torch.cat([core, img], dim=1)
    seen = []
    for clock in (-1.0, 0.0, 0.5, 0.99):
        o_wide = torch.cat([core, torch.full((6, 1), clock), img], dim=1)
        with torch.no_grad():
            lb, vb = base(o_base)
            lw, vw = wide(o_wide)
        # the ACTOR is not widened at all - its weights are the checkpoint's,
        # byte for byte - so this is exact equality, not a tolerance
        assert torch.equal(lb, lw), "logits moved at step 0"
        # the CRITIC is widened by one zero column. The zero column
        # contributes nothing, but a 27-long reduction is not summed in the
        # same order as a 26-long one, so the value differs by float32
        # rounding (measured 1e-7). What must hold is that the difference
        # does not depend on the clock: the feature itself is inert.
        assert (vb - vw).abs().max() < 1e-5, "value moved at step 0"
        seen.append(vw)
    for v in seen[1:]:
        assert torch.equal(seen[0], v), "the clock is not inert at step 0"


def test_adam_moments_are_padded_too():
    w, h = 8, 4
    base = _mk(0, w=w, h=h)
    wide = _mk(1, mr_actor=True, w=w, h=h)   # both towers widen
    params = list(base.parameters())
    names = [n for n, _ in base.named_parameters()]
    st = {}
    for i, (nm, prm) in enumerate(zip(names, params)):
        if nm in ("pi.0.weight", "vf.0.weight"):
            st[i] = {"exp_avg": torch.randn_like(prm),
                     "exp_avg_sq": torch.rand_like(prm)}
    ck = {"policy": {k: v.clone() for k, v in base.state_dict().items()},
          "optimizer": {"state": st}}
    widen_for_route(ck, wide)
    want = dict(wide.named_parameters())
    for i, s in ck["optimizer"]["state"].items():
        nm = names[i]
        for k in ("exp_avg", "exp_avg_sq"):
            assert s[k].shape == want[nm].shape
            assert float(s[k][:, -1].abs().sum()) == 0.0, "new column not zero"


def test_actor_does_not_read_the_clock_by_default():
    """Linesight's ACTING network is always fed clock 0
    (game_instance_manager.py: `floats = np.hstack((0, ...))`); the random
    clocks live only in the learner's minibatch. So by default the clock
    changes the VALUE and never the policy."""
    p = _mk(1, mr_actor=False)
    torch.manual_seed(6)
    core = torch.randn(4, N_SCALAR)
    img = torch.rand(4, 8 * 4)
    with torch.no_grad():
        l0, v0 = p(torch.cat([core, torch.full((4, 1), -1.0), img], dim=1))
        l1, v1 = p(torch.cat([core, torch.full((4, 1), 0.7), img], dim=1))
    assert torch.equal(l0, l1), "the clock must not move the actor by default"
    assert not torch.equal(v0, v1), "the clock must move the critic"


def test_actor_clock_flag_lets_the_policy_see_it():
    p = _mk(1, mr_actor=True)
    torch.manual_seed(6)
    core = torch.randn(4, N_SCALAR)
    img = torch.rand(4, 8 * 4)
    with torch.no_grad():
        l0, _ = p(torch.cat([core, torch.full((4, 1), -1.0), img], dim=1))
        l1, _ = p(torch.cat([core, torch.full((4, 1), 0.7), img], dim=1))
    assert not torch.equal(l0, l1)


def test_clock_composes_with_the_route_block():
    """[15 core | R route | 1 clock | image]: the clock is appended AFTER the
    route fan so both widenings stay trailing-column zero-pads."""
    w, h = 8, 4
    p = _mk(1, route_dim=5, w=w, h=h)
    assert p.scal_dim == N_SCALAR + 5 + 1
    # route to both towers, clock to the critic only: pi is one column short
    assert p.pi[0].in_features == p.vf[0].in_features - 1
    q = Policy(N_SCALAR + 5 + 1 + w * h, lidar_w=w, lidar_h=h, emb=16,
               hidden=12, route_dim=5, route_critic_only=True, mr_dim=1)
    # critic-only route + critic-only clock: the actor tower gets neither
    assert q.pi[0].in_features == q.vf[0].in_features - 6
    r = Policy(N_SCALAR + 5 + 1 + w * h, lidar_w=w, lidar_h=h, emb=16,
               hidden=12, route_dim=5, route_critic_only=True, mr_dim=1,
               mr_actor=True)
    # critic-only route, clock to both: the actor gets the clock alone
    assert r.pi[0].in_features == r.vf[0].in_features - 5


def test_widen_is_inert_at_production_shapes():
    """The shapes the arm actually runs: 64x32 lidar, emb 512, hidden 448,
    --obs-reward's extra slot. The step-0 value error has to be float32 noise,
    not a re-initialisation."""
    w, h = 64, 32
    torch.manual_seed(2)
    base = Policy(N_SCALAR + w * h, lidar_w=w, lidar_h=h, emb=512, hidden=448,
                  extra_feat=(12,))
    torch.manual_seed(2)
    wide = Policy(N_SCALAR + 1 + w * h, lidar_w=w, lidar_h=h, emb=512,
                  hidden=448, extra_feat=(12,), mr_dim=1)
    ck = {"policy": {k: v.clone() for k, v in base.state_dict().items()},
          "optimizer": {"state": {}}}
    assert widen_for_route(ck, wide) == 1
    wide.load_state_dict(ck["policy"])
    torch.manual_seed(8)
    core = torch.randn(4, N_SCALAR)
    img = torch.rand(4, w * h)
    with torch.no_grad():
        lb, vb = base(torch.cat([core, img], dim=1))
        lo, vo = wide(torch.cat([core, torch.full((4, 1), -1.0), img], dim=1))
        lh, vh = wide(torch.cat([core, torch.full((4, 1), 0.9), img], dim=1))
    assert torch.equal(lb, lo) and torch.equal(lb, lh)
    assert torch.equal(vo, vh), "the clock is not inert at step 0"
    assert (vb - vo).abs().max() < 1e-4


def test_remaining_horizon_distribution_matches_linesights():
    """The faithfulness check that actually matters.

    Linesight re-rolls the clock for every stored transition, so what the
    learner sees is a distribution of REMAINING horizons, H - t. PPO cannot
    re-roll after the fact, so this port re-rolls the PHASE at every window
    edge instead - the window LENGTH becomes the draw. Both end up sampling
    the same quantity, and the means must line up: at their constants the
    mean remaining horizon is 4.17 s of a 7 s window, and this port has to
    land on the same seconds at this project's 3-tick decision.
    """
    # Linesight, exactly: 140 actions of 50 ms
    torch.manual_seed(12)
    t_ls = minirace_phase(500_000, LS_WINDOW_ACTIONS, LS_LONG, LS_MAXT)
    rem_ls_s = float((LS_WINDOW_ACTIONS - t_ls).float().mean()) \
        * LS_MS_PER_ACTION / 1000.0
    assert rem_ls_s == pytest.approx(4.17, abs=0.05)

    # this port: 233 decisions of 30 ms, fold constants converted by DURATION
    H = minirace_window(7.0, 3)
    long_steps, max_term = max(1, round(2000 / 30)), max(1, round(250 / 30))
    t_us = minirace_phase(500_000, H, long_steps, max_term)
    rem_us_s = float((H - t_us).float().mean()) * 3 / 100.0
    assert rem_us_s == pytest.approx(rem_ls_s, abs=0.1), \
        f"mean remaining horizon {rem_us_s:.2f}s vs Linesight {rem_ls_s:.2f}s"
    # and it is still well inside the field's 1.4-13 s consensus band, and
    # far under this project's 20 s
    assert 3.0 < rem_us_s < 5.0


def test_window_edge_rate_is_one_over_the_mean_window():
    """What the run prints while it trains: the fraction of rollout rows that
    are window edges. It must be 1/mean-window-length, not 1/H - the phase
    re-roll makes the average window shorter than the full window."""
    H = minirace_window(7.0, 3)
    torch.manual_seed(13)
    t = minirace_phase(200_000, H, max(1, round(2000 / 30)),
                       max(1, round(250 / 30)))
    mean_len = float((H - t).float().mean())
    assert 1.0 / mean_len == pytest.approx(0.0072, abs=0.0008)
