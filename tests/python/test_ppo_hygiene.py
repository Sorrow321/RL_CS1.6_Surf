"""PPO hygiene: the four always-on read-outs, ``--eval-stall``, ``--ret-norm``.

What these pin, in the order that matters:

  * **explained variance is the SB3 formula and it is -inf-safe.** A perfect
    critic scores 1, a critic that predicts the mean scores 0, a critic worse
    than the mean scores negative, and a rollout whose returns have no
    variance to explain reports NaN rather than -inf or a division error -
    which is exactly the rollout this project keeps producing (a policy that
    dies at the same tick every episode);
  * **the three fractions share ONE denominator.** trunc / stall / crawl are
    all "share of the episodes that ENDED this iteration", counted off the
    same masks the rollout already builds. A read-out whose numerator and
    denominator come from different bookkeeping starts disagreeing with
    itself the first time an episode ends mid-decision;
  * **the crawl threshold is a MEAN over the episode**, not an instantaneous
    speed: a policy that launches at 3,000 u/s and then sits on a wall for
    14 s is crawling, and the point of the column is to tell that apart from
    dying fast;
  * **--eval-stall is a MIRROR of the training rule, not a second rule.**
    Running minimum, 32u threshold, per-CALL cadence (CLAUDE.md: at
    act_every 4 a per-tick check would quarter the effective step and kill
    legitimate flight), re-armed after a kill, reset at an episode boundary;
  * **--ret-norm 0 changes nothing.** The normalizer is identity until it is
    updated, and the trainer's OFF branch is the literal expression it was
    before - the bit-identity claim is checked end-to-end by a real GPU run
    in the report, and here at the level of the arithmetic;
  * **the round trip closes.** De-normalizing a normalized return returns it,
    which is the property every V(s) read outside the value loss depends on.

    python -m pytest tests/python/test_ppo_hygiene.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from train_fast import (CRAWL_SPEED, ReturnNorm, chain_ticks,     # noqa: E402
                        episode_hygiene, explained_var_from_sums,
                        make_eval_stall_hook)


# --------------------------------------------------------------------------
# explained variance
# --------------------------------------------------------------------------
def _ev(y, v):
    """The shipped formula, fed the sufficient statistics the trainer feeds
    it (n, sum y, sum y^2, sum e, sum e^2 with e = y - v)."""
    y = np.asarray(y, np.float64)
    v = np.asarray(v, np.float64)
    e = y - v
    return explained_var_from_sums(y.size, y.sum(), (y * y).sum(),
                                   e.sum(), (e * e).sum())


def test_perfect_critic_scores_one():
    rng = np.random.default_rng(0)
    y = rng.normal(50.0, 12.0, 4096)
    assert _ev(y, y.copy()) == pytest.approx(1.0, abs=1e-12)


def test_mean_critic_scores_zero():
    """A critic that predicts the batch mean explains none of the variance.

    This is the calibration point for reading the column: 0 is not "broken",
    it is "no better than a constant", and it is where an untrained critic
    starts.
    """
    rng = np.random.default_rng(1)
    y = rng.normal(-3.0, 7.0, 8192)
    assert _ev(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-12)


def test_a_constant_offset_is_still_explained():
    """EV is variance-based, so a critic that is right up to a CONSTANT bias
    still scores 1 - which is precisely why --ret-norm's shifting mean does
    not corrupt this column."""
    rng = np.random.default_rng(2)
    y = rng.normal(100.0, 5.0, 2048)
    assert _ev(y, y + 17.5) == pytest.approx(1.0, abs=1e-12)


def test_worse_than_the_mean_goes_negative_and_is_not_clipped():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, 4096)
    ev = _ev(y, -y)          # anti-correlated: Var(y - v) = 4 Var(y)
    assert ev == pytest.approx(-3.0, rel=1e-9)


def test_zero_variance_returns_nan_not_minus_inf():
    """The degenerate rollout: every episode identical, Var(G) = 0. The old
    formula divides by zero; this must report 'no measurement'."""
    y = np.full(512, 42.0)
    assert math.isnan(_ev(y, np.zeros_like(y)))
    assert math.isnan(_ev(y, y.copy()))          # 0/0, not 1
    assert math.isnan(explained_var_from_sums(0, 0.0, 0.0, 0.0, 0.0))


def test_non_finite_statistics_report_nan():
    assert math.isnan(explained_var_from_sums(
        10.0, float("inf"), float("inf"), 0.0, 0.0))
    assert math.isnan(explained_var_from_sums(
        10.0, 0.0, 100.0, 0.0, float("nan")))


def test_sufficient_statistics_reduce_like_a_single_batch():
    """The DDP property: summing two ranks' statistics gives the number a
    single rank holding both halves would report. A mean of per-rank EVs
    does NOT have this property, which is why the trainer reduces the
    statistics rather than the ratio."""
    rng = np.random.default_rng(4)
    y = rng.normal(20.0, 9.0, 4096)
    v = y * 0.7 + rng.normal(0.0, 2.0, 4096)
    a, b = slice(0, 1500), slice(1500, None)          # deliberately uneven

    def stats(sl):
        yy, vv = y[sl], v[sl]
        ee = yy - vv
        return np.array([yy.size, yy.sum(), (yy * yy).sum(),
                         ee.sum(), (ee * ee).sum()])

    tot = stats(a) + stats(b)
    assert explained_var_from_sums(*tot) == pytest.approx(_ev(y, v), rel=1e-9)


# --------------------------------------------------------------------------
# trunc / stall / crawl on a synthetic ended mask
# --------------------------------------------------------------------------
def test_episode_hygiene_counts_on_a_synthetic_mask():
    n = 8
    ended = np.zeros(n, bool)
    ended[[1, 3, 4, 6]] = True
    is_trunc = np.zeros(n, bool)
    is_trunc[[3, 6]] = True                  # 2 of the 4 ended are truncations
    ep_len = np.array([100, 500, 100, 1000, 200, 100, 50, 100], np.int64)
    # mean speeds, in ku/s: env 1 = 2.0 (flying), env 3 = 0.1 (crawling),
    # env 4 = 0.299 (crawling, just under), env 6 = 0.301 (just over)
    mean = np.array([9.9, 2.0, 9.9, 0.1, 0.299, 9.9, 0.301, 9.9])
    spd_sum = mean * ep_len
    n_end, n_tr, n_cr = episode_hygiene(ended, is_trunc, spd_sum, ep_len)
    assert (n_end, n_tr, n_cr) == (4, 2, 2)


def test_episode_hygiene_ignores_envs_that_did_not_end():
    """Only ENDED rows are counted - a live crawler is not an outcome yet,
    and counting it would make the fractions drift with episode length."""
    n = 6
    ended = np.zeros(n, bool)
    ended[2] = True
    is_trunc = np.ones(n, bool)              # every LIVE row looks truncated
    ep_len = np.full(n, 100, np.int64)
    spd_sum = np.zeros(n)                    # every LIVE row looks crawling
    spd_sum[2] = 100.0 * 5.0                 # ... but the ended one flew
    assert episode_hygiene(ended, is_trunc, spd_sum, ep_len) == (1, 1, 0)


def test_episode_hygiene_no_ends_is_all_zero():
    z = np.zeros(4, bool)
    assert episode_hygiene(z, z, np.zeros(4), np.zeros(4, np.int64)) == (0, 0, 0)


def test_crawl_is_the_episode_MEAN_not_the_final_speed():
    """A launch at 3,000 u/s followed by 14 s stuck on a wall IS a crawl.
    Mean over 1,500 ticks = 3,000*100/1,500 = 200 u/s."""
    ep_len = np.array([1500], np.int64)
    spd_sum = np.array([3.0 * 100])          # 100 fast ticks, then nothing
    assert episode_hygiene(np.array([True]), np.array([False]),
                           spd_sum, ep_len) == (1, 0, 1)


def test_crawl_threshold_is_300_units_per_second():
    ep_len = np.array([1000, 1000], np.int64)
    spd_sum = np.array([(CRAWL_SPEED - 1.0) / 1000.0,
                        (CRAWL_SPEED + 1.0) / 1000.0]) * 1000
    assert episode_hygiene(np.ones(2, bool), np.zeros(2, bool),
                           spd_sum, ep_len)[2] == 1


def test_zero_length_episode_does_not_divide_by_zero():
    assert episode_hygiene(np.array([True]), np.array([False]),
                           np.array([0.0]), np.array([0], np.int64)) == (1, 0, 1)


# --------------------------------------------------------------------------
# the stall-kill counter is a real count, not a boolean
# --------------------------------------------------------------------------
class _FakeCore:
    def __init__(self, n, origins=None):
        self.num_envs = n
        self.failed = np.zeros(n, np.uint8)
        self.states_view = np.zeros(n, dtype=[("origin", np.float64, (3,))])
        if origins is not None:
            self.states_view["origin"] = origins

    def force_fail(self, mask):
        self.failed |= np.asarray(mask, np.uint8)


class _FakeReward:
    def __init__(self, mask):
        self._mask = mask

    def pop_stall_mask(self):
        return self._mask


def test_apply_stall_kills_returns_the_number_it_killed():
    from surfgym.mapfleet import MapFleet

    class _Slot:
        pass

    fleet = MapFleet.__new__(MapFleet)          # no map, no sim: this method
    a, b = _Slot(), _Slot()                     # only touches slots
    a.core, a.reward_fn = _FakeCore(4), _FakeReward(
        np.array([1, 0, 1, 0], np.uint8))
    b.core, b.reward_fn = _FakeCore(4), _FakeReward(
        np.array([0, 0, 0, 1], np.uint8))
    fleet.slots = [a, b]
    assert fleet.apply_stall_kills() == 3
    assert list(a.core.failed) == [1, 0, 1, 0]
    assert list(b.core.failed) == [0, 0, 0, 1]

    # and a fleet with nothing to kill reports 0, not None: the trainer adds
    # this straight into a counter
    a.reward_fn = _FakeReward(None)
    b.reward_fn = _FakeReward(None)
    assert fleet.apply_stall_kills() == 0


# --------------------------------------------------------------------------
# --eval-stall
# --------------------------------------------------------------------------
class _LineField:
    """d = the env's x coordinate. Moving -x is progress toward the goal."""

    def sample(self, pts):
        return np.asarray(pts, np.float64)[:, 0].copy()


def _run(hook, core, xs, every=1):
    """Feed the hook one tick per entry of `xs`, stopping at the first kill."""
    for t, x in enumerate(xs):
        core.states_view["origin"][0, 0] = float(x)
        hook(t, None, None, np.zeros(core.num_envs, bool),
             np.zeros(core.num_envs, bool))
        if core.failed[0]:
            return t
    return None


def test_eval_stall_fires_after_the_window_without_improvement():
    core = _FakeCore(1)
    hook = make_eval_stall_hook(core, _LineField(), stall_ticks=100,
                               stall_eps=32.0, every=1)
    # arm on tick 0, then 100 ticks of no improvement (a 10u drift is under
    # the 32u threshold, so it does NOT re-arm the timer)
    killed = _run(hook, core, [1000.0] + [1000.0 - (i % 3) * 5.0
                                          for i in range(200)])
    assert killed is not None
    # armed at t=0; `since` reaches 100 on the 100th call after that
    assert killed == 100
    assert hook.state["n"] == 1


def test_eval_stall_does_not_fire_while_the_agent_is_improving():
    core = _FakeCore(1)
    hook = make_eval_stall_hook(core, _LineField(), stall_ticks=100,
                               stall_eps=32.0, every=1)
    # 33u closer every tick: over the threshold, so the timer re-arms
    assert _run(hook, core, 1000.0 - 33.0 * np.arange(400)) is None
    assert hook.state["n"] == 0


def test_eval_stall_needs_a_32u_step_not_just_any_progress():
    """The rule is a per-CALL threshold against a running MINIMUM, not a
    rate: 5u per call is progress and still stalls out."""
    core = _FakeCore(1)
    hook = make_eval_stall_hook(core, _LineField(), stall_ticks=100,
                               stall_eps=32.0, every=1)
    assert _run(hook, core, 1000.0 - 5.0 * np.arange(400)) == 100


def test_eval_stall_cadence_scales_with_act_every():
    """CLAUDE.md: the threshold is PER CALL. At every=4 the rule is evaluated
    once per 4 ticks and `since` advances by 4, so the WINDOW in ticks is
    unchanged - a per-tick check would quarter the effective step and kill
    legitimate flight."""
    core = _FakeCore(1)
    hook = make_eval_stall_hook(core, _LineField(), stall_ticks=100,
                               stall_eps=32.0, every=4)
    killed = _run(hook, core, np.full(400, 1000.0))
    # 4 ticks to arm, then 25 evaluated calls x 4 ticks each to reach 100
    assert killed == 4 + 25 * 4 - 1
    assert hook.state["n"] == 1


def test_eval_stall_resets_at_an_episode_boundary():
    core = _FakeCore(1)
    hook = make_eval_stall_hook(core, _LineField(), stall_ticks=100,
                               stall_eps=32.0, every=1)
    done = np.ones(1, bool)
    zero = np.zeros(1, bool)
    for t in range(90):                        # 90 ticks of nothing
        core.states_view["origin"][0, 0] = 1000.0
        hook(t, None, None, zero, zero)
    hook(90, None, None, done, zero)           # episode ends
    for t in range(91, 181):                   # 90 more in the NEXT episode
        core.states_view["origin"][0, 0] = 1000.0
        hook(t, None, None, zero, zero)
    assert not core.failed[0]                  # neither episode reached 100
    assert hook.state["n"] == 0


def test_eval_stall_rearms_after_a_kill():
    """pop_stall_mask zeroes `since` so a kill fires ONCE; the mirror does
    the same, or a single stalled episode would force_fail every tick."""
    core = _FakeCore(1)
    hook = make_eval_stall_hook(core, _LineField(), stall_ticks=10,
                               stall_eps=32.0, every=1)
    for t in range(60):
        core.states_view["origin"][0, 0] = 1000.0
        hook(t, None, None, np.zeros(1, bool), np.zeros(1, bool))
    # force_fail only ends the episode on the NEXT step, so between the kill
    # and the reset the hook keeps being called: it must not re-fire until a
    # whole fresh window has passed
    assert hook.state["n"] == 5                # 60 ticks / a 10-tick window
    # and the timer is mid-window rather than latched at the threshold
    assert 0 <= hook.state["since"] < 10


def test_eval_stall_is_off_when_the_window_is_zero():
    core = _FakeCore(1)
    hook = make_eval_stall_hook(core, _LineField(), stall_ticks=0,
                               stall_eps=32.0, every=1)
    assert _run(hook, core, np.full(5000, 1000.0)) is None


def test_chain_ticks_preserves_order_and_drops_none():
    seen = []
    assert chain_ticks(None, None) is None
    one = lambda *a: seen.append("a")          # noqa: E731
    assert chain_ticks(None, one) is one
    chained = chain_ticks(one, lambda *a: seen.append("b"), None)
    chained(0, None, None, None, None)
    assert seen == ["a", "b"]


# --------------------------------------------------------------------------
# --ret-norm
# --------------------------------------------------------------------------
def test_ret_norm_is_identity_before_any_update():
    """--ret-norm 0 never calls update(), so this is also the statement that
    the flag OFF cannot move a number: mean 0, std 1, both directions."""
    rn = ReturnNorm()
    assert (rn.mean, rn.std) == (0.0, 1.0)
    x = np.array([-3.25, 0.0, 17.5, 1e4])
    assert np.array_equal(rn.normalize(x), x)
    assert np.array_equal(rn.denormalize(x), x)


def test_de_normalization_is_an_exact_round_trip():
    rng = np.random.default_rng(7)
    rn = ReturnNorm()
    g = rng.normal(120.0, 40.0, 8192)
    rn.update(g.mean(), (g * g).mean())
    x = rng.normal(0.0, 300.0, 1000)
    assert np.allclose(rn.denormalize(rn.normalize(x)), x, rtol=0, atol=1e-9)
    assert np.allclose(rn.normalize(rn.denormalize(x)), x, rtol=0, atol=1e-9)


def test_one_update_is_debiased_to_the_batch_moments():
    """Without the 1 - beta^k debias the first iteration would divide by a
    std 1% of the truth and blow the value loss up exactly where the run can
    least absorb it. After ONE update the pair must BE the batch's."""
    rng = np.random.default_rng(8)
    g = rng.normal(-40.0, 9.0, 20000)
    rn = ReturnNorm(beta=0.99)
    mean, std = rn.update(g.mean(), (g * g).mean())
    assert mean == pytest.approx(g.mean(), rel=1e-9)
    assert std == pytest.approx(g.std(), rel=1e-6)


def test_the_ema_tracks_a_moving_level_and_is_slow():
    """beta 0.99 per ITERATION is a ~100-iteration time constant: after one
    step onto a new level the estimate has moved 1% of the way, which is the
    property 'PopArt-lite without the weight-preserving update' rests on."""
    rn = ReturnNorm(beta=0.99)
    rn.update(0.0, 1.0)                        # level 0
    first = rn.mean
    rn.update(100.0, 100.0 ** 2 + 1.0)         # level jumps to 100
    assert first == pytest.approx(0.0, abs=1e-12)
    # debiased two-step EMA: (0.99*0 + 0.01*100) / (0.99*0.01 + 0.01) ~ 50.25
    assert 45.0 < rn.mean < 55.0
    # ... and it does converge. NOTE the shape of the approach: the variance
    # is E[G^2] - E[G]^2, a difference of two ~1e4 numbers here, so a 1e-3
    # relative error in the mean is still a 1.7x error in sigma at 400
    # iterations. That is PopArt's own estimator (1602.07714 eq. 6) and it is
    # well conditioned at the ratios a real return has (~80 +- 25); it is
    # worth knowing that a huge mean over a tiny spread is where it is not.
    for _ in range(2000):
        rn.update(100.0, 100.0 ** 2 + 1.0)
    assert rn.mean == pytest.approx(100.0, rel=1e-9)
    assert rn.std == pytest.approx(1.0, rel=1e-4)


def test_sigma_is_clipped_on_a_degenerate_rollout():
    """Every return identical - a policy that dies at the same tick every
    episode, which is the regime this project keeps landing in. Var = 0, and
    sqrt(eps) = 1e-4 would hand the optimizer a 1e4x target."""
    rn = ReturnNorm()
    rn.update(7.0, 49.0)
    assert rn.std == ReturnNorm.SIGMA_MIN
    assert rn.normalize(7.0) == pytest.approx(0.0)


def test_state_dict_round_trip_restores_mean_and_std():
    rng = np.random.default_rng(9)
    a = ReturnNorm()
    for _ in range(37):
        g = rng.normal(55.0, 12.0, 4096)
        a.update(g.mean(), (g * g).mean())
    b = ReturnNorm()
    b.load_state_dict(a.state_dict())
    assert b.mean == pytest.approx(a.mean, rel=0, abs=0)
    assert b.std == pytest.approx(a.std, rel=0, abs=0)
    assert b.count == a.count
    # and it keeps EVOLVING identically, which is what a resume needs
    g = rng.normal(55.0, 12.0, 4096)
    assert a.update(g.mean(), (g * g).mean()) == b.update(g.mean(),
                                                          (g * g).mean())


def test_a_fresh_normalizer_loads_an_empty_state_as_identity():
    rn = ReturnNorm()
    rn.load_state_dict({})
    assert (rn.mean, rn.std) == (0.0, 1.0)


# --------------------------------------------------------------------------
# --ret-norm: the update the trainer actually runs
# --------------------------------------------------------------------------
def _ppo_value_step(values_raw, ret, retn=None):
    """The trainer's value half, both branches, as one function.

    OFF: the head predicts the return, the loss is 0.5*(V - G)^2.
    ON:  the head predicts (G - mu)/sigma, and every V read OUTSIDE the loss
         is de-normalized. `values_raw` is what the head emits.
    """
    import torch
    v = torch.as_tensor(values_raw, dtype=torch.float32)
    g = torch.as_tensor(ret, dtype=torch.float32)
    if retn is None:
        target = g
        v_read = v
    else:
        target = (g - retn.mean) / retn.std
        v_read = v * retn.std + retn.mean
    return float(0.5 * (v - target).pow(2).mean()), v_read


def test_ret_norm_off_is_the_untreated_value_step():
    """The OFF branch is the literal expression the trainer had before the
    flag existed: same target tensor, same loss, same V handed to GAE."""
    import torch
    rng = np.random.default_rng(11)
    v = rng.normal(0.0, 1.0, 512).astype(np.float32)
    g = rng.normal(80.0, 25.0, 512).astype(np.float32)
    loss, v_read = _ppo_value_step(v, g)
    ref = float(0.5 * (torch.as_tensor(v) - torch.as_tensor(g)).pow(2).mean())
    assert loss == ref
    assert torch.equal(v_read, torch.as_tensor(v))
    # an un-updated normalizer is the same thing again, to the bit
    loss2, v2 = _ppo_value_step(v, g, ReturnNorm())
    assert loss2 == ref and torch.equal(v2, torch.as_tensor(v))


def test_ret_norm_puts_the_target_in_unit_scale_and_reads_V_back_in_units():
    """The whole point: an ~80 +- 25 return becomes an ~N(0,1) target, while
    the value the rest of PPO sees is still in reward units."""
    rng = np.random.default_rng(12)
    g = rng.normal(80.0, 25.0, 8192).astype(np.float32)
    rn = ReturnNorm()
    rn.update(float(g.mean()), float((g.astype(np.float64) ** 2).mean()))
    target = (g - rn.mean) / rn.std
    assert abs(float(target.mean())) < 1e-3
    assert float(target.std()) == pytest.approx(1.0, rel=1e-3)
    # a critic that has learned the normalized target reads back as the raw
    # return, which is what GAE and the truncation bootstrap consume
    _, v_read = _ppo_value_step(target, g, rn)
    assert np.allclose(np.asarray(v_read), g, rtol=1e-4, atol=1e-2)


def test_ret_norm_shrinks_the_value_loss_by_sigma_squared():
    """The mechanism, stated as a number: a critic that is off by `e` reward
    units pays e^2 without the flag and (e/sigma)^2 with it. That is the
    whole treatment - the value term stops dominating the policy gradient
    when returns are large."""
    rng = np.random.default_rng(13)
    g = rng.normal(80.0, 25.0, 4096).astype(np.float32)
    rn = ReturnNorm()
    rn.update(float(g.mean()), float((g.astype(np.float64) ** 2).mean()))
    err = 5.0
    off, _ = _ppo_value_step(g + err, g)
    on, _ = _ppo_value_step((g - rn.mean) / rn.std + err / rn.std, g, rn)
    assert off == pytest.approx(0.5 * err ** 2, rel=1e-4)
    assert on == pytest.approx(off / rn.std ** 2, rel=1e-3)


def test_advantages_are_untouched_by_ret_norm():
    """--ret-norm normalizes the value TARGET. GAE runs on de-normalized V,
    so the advantages - and therefore the policy gradient - are the same
    array they were. Checked by running GAE both ways."""
    import torch
    rng = np.random.default_rng(14)
    T, N = 16, 8
    rew = torch.as_tensor(rng.normal(0.1, 0.3, (T, N)), dtype=torch.float32)
    val_raw = torch.as_tensor(rng.normal(40.0, 8.0, (T, N)),
                              dtype=torch.float32)
    done = torch.as_tensor((rng.random((T, N)) < 0.05).astype(np.float32))
    last = torch.as_tensor(rng.normal(40.0, 8.0, N), dtype=torch.float32)
    gamma, lam = 0.99, 0.95

    def gae(val, lastv):
        adv = torch.zeros_like(rew)
        lastgae = torch.zeros(N)
        for t in reversed(range(T)):
            nxt = lastv if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + gamma * nxt * nonterm - val[t]
            lastgae = delta + gamma * lam * nonterm * lastgae
            adv[t] = lastgae
        return adv

    rn = ReturnNorm()
    rn.update(40.0, 40.0 ** 2 + 64.0)
    # the head emits normalized values; the trainer de-normalizes in place
    val_norm = (val_raw - rn.mean) / rn.std
    a_off = gae(val_raw, last)
    a_on = gae(val_norm * rn.std + rn.mean,
               last * 1.0)                      # last_val de-normalized too
    assert torch.allclose(a_off, a_on, rtol=1e-4, atol=1e-3)


def test_explained_variance_is_reported_in_RAW_units_under_ret_norm():
    """The trainer de-normalizes b_val BEFORE computing the statistics, so
    the column means the same thing with the flag on and off - otherwise the
    two arms of an ablation could not be plotted on one axis."""
    rng = np.random.default_rng(15)
    g = rng.normal(80.0, 25.0, 4096)
    v = g + rng.normal(0.0, 5.0, 4096)
    rn = ReturnNorm()
    rn.update(g.mean(), (g * g).mean())
    raw = _ev(g, v)
    # same critic, expressed normalized, then de-normalized by the trainer
    v_norm = (v - rn.mean) / rn.std
    assert _ev(g, v_norm * rn.std + rn.mean) == pytest.approx(raw, rel=1e-9)
