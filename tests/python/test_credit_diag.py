"""tools/credit_diag.py - the credit arithmetic, CPU only.

No GPU, no map, no checkpoint: what is pinned here is the half of the
diagnostic that decides its verdict, i.e. that the three advantage variants
the report compares ARE what they claim to be.

  1. the empirical return recursion equals the explicit discounted sum, and
     stops at the terminal decision rather than running past it;
  2. lambda = 1 over a whole episode is exactly ``G - V`` (the telescoping
     identity that makes variant (b) "the Monte-Carlo advantage");
  3. variant (a) - the run's lambda cut at rollout-buffer boundaries -
     reproduces train_fast.py's GAE loop DECISION BY DECISION, including an
     episode that spans two buffers, checked against a literal transcription
     of that loop rather than against a restatement of the same idea. This
     is the one that carries the report: if (a) were wrong the diagnostic
     would be measuring its own bug;
  4. the buffer edge really does cut the carry (n_steps = 1 leaves the bare
     TD residual) and a buffer longer than the episode leaves no cut at all
     (variant (a) at phase 0 collapses onto variant (c));
  5. the discount table reproduces the reviewer's 0.0013-vs-0.78 contrast
     from the scratch config's own constants.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import credit_diag as cd  # noqa: E402


# ---------------------------------------------------------------------------
# a LITERAL transcription of train_fast.py's GAE loop (the block that reads
# `for t in reversed(range(T)): ... lastgae = delta + g_eff * args.gae *
# nonterm * lastgae`). Kept verbatim on purpose: the point of the test is
# that credit_diag's per-episode form agrees with the trainer's buffer form,
# not that two rewrites of the same recursion agree.
# ---------------------------------------------------------------------------
def trainer_gae(b_rew, b_val, b_done, last_val, g_eff, lam):
    T = b_rew.shape[0]
    adv = np.zeros_like(b_rew)
    lastgae = np.zeros(b_rew.shape[1])
    for t in reversed(range(T)):
        nextval = last_val if t == T - 1 else b_val[t + 1]
        nonterm = 1.0 - b_done[t]
        delta = b_rew[t] + g_eff * nextval * nonterm - b_val[t]
        lastgae = delta + g_eff * lam * nonterm * lastgae
        adv[t] = lastgae
    return adv


G = 0.9995 ** 4          # the scratch config: gamma per tick, act_every 4
LAM = 0.95


# ---------------------------------------------------------------------------
# 1. returns
# ---------------------------------------------------------------------------
def test_returns_match_the_explicit_discounted_sum():
    r = np.array([0.1, -0.2, 0.4, 50.0])
    nt = cd.nonterm_mask(len(r))
    got = cd.discounted_returns(r, G, nt)
    want = np.array([sum(G ** k * r[t + k] for k in range(len(r) - t))
                     for t in range(len(r))])
    assert np.allclose(got, want)


def test_return_stops_at_the_terminal_decision():
    """nonterm[-1] = 0, so nothing after the episode can leak into G."""
    r = np.array([1.0, 2.0, 3.0])
    nt = cd.nonterm_mask(3)
    assert cd.discounted_returns(r, G, nt)[-1] == pytest.approx(3.0)
    # and a non-terminal last row would carry (used by nothing here, but the
    # flag has to mean what it says)
    nt2 = cd.nonterm_mask(3, ended_last=False)
    assert np.all(nt2 == 1.0)


# ---------------------------------------------------------------------------
# 2. lambda = 1 is the Monte-Carlo advantage
# ---------------------------------------------------------------------------
def test_lambda_one_full_episode_is_return_minus_value():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.2, 40)
    r[-1] -= 3.0
    V = rng.normal(1.0, 0.5, 40)
    nt = cd.nonterm_mask(40)
    delta = cd.td_residuals(r, V, G, nt)
    adv = cd.gae(delta, G, 1.0, nt, n_steps=None)
    assert np.allclose(adv, cd.discounted_returns(r, G, nt) - V)


def test_td_residual_is_the_trainers_delta():
    r = np.array([0.3, -0.1, 2.0])
    V = np.array([5.0, 4.0, 1.0])
    nt = cd.nonterm_mask(3)
    d = cd.td_residuals(r, V, G, nt)
    assert d[0] == pytest.approx(0.3 + G * 4.0 - 5.0)
    assert d[1] == pytest.approx(-0.1 + G * 1.0 - 4.0)
    assert d[2] == pytest.approx(2.0 - 1.0)      # terminal: no bootstrap


# ---------------------------------------------------------------------------
# 3. variant (a) == the trainer's buffer form
# ---------------------------------------------------------------------------
def _episode_from_buffers(bufs, g, lam, ep_lo, ep_hi):
    """Run trainer_gae over a chain of buffers (each one its own backward
    pass, bootstrapping on the next buffer's first value, exactly as the
    trainer does one iteration after another) and return the advantages of
    global decisions ep_lo..ep_hi inclusive."""
    T = bufs[0][0].shape[0]
    out = []
    for i, (rew, val, done) in enumerate(bufs):
        nxt = bufs[i + 1][1][0] if i + 1 < len(bufs) else np.zeros(1)
        out.append(trainer_gae(rew, val, done, nxt, g, lam))
    flat = np.concatenate(out, axis=0)[:, 0]
    assert len(flat) == T * len(bufs)
    return flat[ep_lo:ep_hi + 1]


def test_variant_a_matches_the_trainer_loop_inside_one_buffer():
    T = 8
    rng = np.random.default_rng(1)
    rew = rng.normal(0, 0.2, (T, 1))
    val = rng.normal(1.0, 0.4, (T, 1))
    done = np.zeros((T, 1))
    done[5, 0] = 1.0                       # an episode ends at decision 5
    want = trainer_gae(rew, val, done, val[0] * 0.0, G, LAM)[:6, 0]

    r = rew[:6, 0].copy()
    V = val[:6, 0].copy()
    nt = cd.nonterm_mask(6)
    delta = cd.td_residuals(r, V, G, nt)
    got = cd.gae(delta, G, LAM, nt, n_steps=T, phase=0)
    assert np.allclose(got, want), (got, want)


def test_variant_a_matches_the_trainer_across_a_buffer_boundary():
    """The case the report turns on: an episode LONGER than the rollout is
    cut, and every decision before the cut is credited by a separate
    backward pass. phase = the global index the episode started at, mod T."""
    T = 4
    rng = np.random.default_rng(2)
    rew = rng.normal(0, 0.3, (2 * T, 1))
    val = rng.normal(2.0, 0.5, (2 * T, 1))
    done = np.zeros((2 * T, 1))
    done[6, 0] = 1.0                       # episode spans decisions 1..6
    bufs = [(rew[:T], val[:T], done[:T]), (rew[T:], val[T:], done[T:])]
    want = _episode_from_buffers(bufs, G, LAM, 1, 6)

    r = rew[1:7, 0].copy()
    V = val[1:7, 0].copy()
    nt = cd.nonterm_mask(6)
    delta = cd.td_residuals(r, V, G, nt)
    got = cd.gae(delta, G, LAM, nt, n_steps=T, phase=1 % T)
    assert np.allclose(got, want), (got, want)
    # and the cut is real: the same episode without truncation credits its
    # first decision MORE than the trainer's cut version does
    full = cd.gae(delta, G, LAM, nt, n_steps=None)
    assert not np.allclose(full, got)


# ---------------------------------------------------------------------------
# 4. what the buffer edge does
# ---------------------------------------------------------------------------
def test_n_steps_one_leaves_the_bare_td_residual():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.2, 12)
    V = rng.normal(0, 0.4, 12)
    nt = cd.nonterm_mask(12)
    delta = cd.td_residuals(r, V, G, nt)
    assert np.allclose(cd.gae(delta, G, LAM, nt, n_steps=1, phase=0), delta)


def test_a_buffer_longer_than_the_episode_is_variant_c():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.2, 20)
    V = rng.normal(0, 0.4, 20)
    nt = cd.nonterm_mask(20)
    delta = cd.td_residuals(r, V, G, nt)
    a = cd.gae(delta, G, LAM, nt, n_steps=128, phase=0)
    c = cd.gae(delta, G, LAM, nt, n_steps=None)
    assert np.allclose(a, c)


def test_phase_mean_is_bracketed_by_its_envelope():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.2, 30)
    V = rng.normal(0, 0.4, 30)
    nt = cd.nonterm_mask(30)
    delta = cd.td_residuals(r, V, G, nt)
    m, lo, hi = cd.gae_phase_mean(delta, G, LAM, nt, 8)
    assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12)
    # every phase is one of the 8 the mean averages, so the envelope must
    # contain each of them
    for p in range(8):
        a = cd.gae(delta, G, LAM, nt, n_steps=8, phase=p)
        assert np.all(a >= lo - 1e-12) and np.all(a <= hi + 1e-12)


def test_phase_matters_at_the_first_decision():
    """The credit a start state's action gets depends on where the rollout
    boundary happens to fall - which is why the report averages over it."""
    L, T = 6, 8
    r = np.zeros(L)
    r[-1] = 50.0                    # one success, at the end
    V = np.zeros(L)
    nt = cd.nonterm_mask(L)
    delta = cd.td_residuals(r, V, G, nt)
    advs = [cd.gae(delta, G, LAM, nt, n_steps=T, phase=p)[0] for p in range(T)]
    # phase 0 puts no cut inside a 6-decision episode: the full trace
    assert advs[0] == pytest.approx((G * LAM) ** 5 * 50.0)
    # phase 3 cuts at decision 4, and the success at 5 is on the far side
    assert advs[3] == pytest.approx(0.0, abs=1e-12)
    assert max(advs) > min(advs)
    # and an episode LONGER than the buffer can be cut off from its own
    # success under EVERY phase - which is the failure mode the report is
    # looking for
    L2 = 24
    r2 = np.zeros(L2)
    r2[-1] = 50.0
    nt2 = cd.nonterm_mask(L2)
    d2 = cd.td_residuals(r2, np.zeros(L2), G, nt2)
    assert max(cd.gae(d2, G, LAM, nt2, n_steps=T, phase=p)[0]
               for p in range(T)) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. the discount table
# ---------------------------------------------------------------------------
def test_discount_table_reproduces_the_reviewers_contrast():
    tbl = cd.discount_table(0.9995, 4, 0.95)
    by_k = {r["k_decisions"]: r for r in tbl}
    row = by_k[125]
    assert row["seconds"] == pytest.approx(5.0)
    assert row["ticks"] == 500
    assert row["gae_weight"] == pytest.approx(0.00128, abs=2e-5)
    assert row["discount_weight"] == pytest.approx(0.779, abs=2e-3)
    assert row["ratio"] > 500
    # monotone in k, both columns
    ws = [by_k[k]["gae_weight"] for k in (25, 50, 125, 250)]
    ds = [by_k[k]["discount_weight"] for k in (25, 50, 125, 250)]
    assert ws == sorted(ws, reverse=True)
    assert ds == sorted(ds, reverse=True)


def test_discount_table_at_lambda_one_is_the_discount():
    for r in cd.discount_table(0.9995, 4, 1.0):
        assert r["gae_weight"] == pytest.approx(r["discount_weight"])
        assert r["ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# scoring plumbing (no core needed)
# ---------------------------------------------------------------------------
def test_summarize_excludes_truncated_and_still_counts_them():
    eps = [
        {"truncated": False, "ret": 1.0, "V0": 2.0, "err0": 1.0,
         "adv_a": 0.1, "adv_b": 0.5, "adv_c": 0.2, "arc": 210000.0,
         "arc0": 200000.0, "decisions": 10, "past_wall": True,
         "finished": False},
        {"truncated": False, "ret": -1.0, "V0": 2.0, "err0": 3.0,
         "adv_a": -0.1, "adv_b": -0.5, "adv_c": -0.2, "arc": 201000.0,
         "arc0": 200000.0, "decisions": 8, "past_wall": False,
         "finished": False},
        {"truncated": True, "ret": 99.0, "V0": 0.0, "err0": -99.0,
         "adv_a": 9.0, "adv_b": 9.0, "adv_c": 9.0, "arc": 0.0,
         "arc0": 0.0, "decisions": 1, "past_wall": False, "finished": False},
    ]
    row = cd.summarize(eps, "t", 5.0, "sampled", 4)
    assert row["n"] == 2 and row["n_trunc"] == 1
    assert row["n_success"] == 1 and row["frac_success"] == pytest.approx(0.5)
    assert row["bias_mean"] == pytest.approx(2.0)          # the 99 is out
    assert row["adv_b_gap"] == pytest.approx(1.0)
    assert row["len_s_mean"] == pytest.approx(9 * 0.04)


def test_summarize_of_an_empty_batch_is_not_a_crash():
    row = cd.summarize([], "t", 0.0, "greedy", 4)
    assert row["n"] == 0
    assert cd.md_table([row]).count("\n") == 2


def test_unsupported_config_is_refused_loudly():
    with pytest.raises(SystemExit):
        cd.check_supported({"reward": "race", "rnn": "gru"})
    with pytest.raises(SystemExit):
        cd.check_supported({"reward": "coverage"})
    with pytest.raises(SystemExit):
        cd.check_supported({"reward": "race", "race_arc": "line.npz"})
    cd.check_supported({"reward": "race", "obs_reward": True,
                        "act_every": 3})       # the stuck checkpoint: fine
