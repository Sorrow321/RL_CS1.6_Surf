"""Search-derived distillation targets (P2 policy, P3 value).

The expert-iteration loop throws away everything the planner computed except
one argmax per decision. Two papers say that is the weak end of the loop:
Expert Iteration (Anthony, Tian & Barber 2017) measured **50 +/- 13 Elo**
between CAT (the argmax) and TPT (the search's own distribution over first
decisions) at **indistinguishable top-1 accuracy** (47.0% vs 47.7% error),
and AlphaZero's loss carries a `(z - v)^2` term our critic has no analogue
of even though the planner's own line supplies an exact, bootstrap-free `z`.

What is pinned here:

  1. **an old BC file is still an old BC file.** A version-1 file loads, the
     writer still emits version 1 when handed no target, and the trainer's
     BC loss on it is byte-identical to the pre-flag expression under the
     default flags (`--bc-target argmax`, `--bc-value-coef 0`);
  2. **the distribution target REDUCES to the one-hot exactly** - not
     approximately: the padded slots hold NEG = -1e30 and `0 * -1e30` is
     -0.0, so `-(onehot * log_softmax).sum()` is the gathered log-prob bit
     for bit, and `--bc-target dist` on a version-1 file is `argmax`;
  3. **the survivor distribution is the search's own**: prefix-matching
     lineages, one line in gives its one-hot back, and the population count
     inside `relabel_windows` is the truncation-selection visit count;
  4. **the return-to-go is the trainer's arithmetic** - hand-computed on a
     tiny synthetic line at 10 ms AND at 7.63 ms, where `gamma` is raised to
     `tick/10` before `act_every` and the two must not be confused;
  5. **the value term is off by default and live when asked**, reads the
     privileged block under --priv-critic, and is masked to the rows whose
     return is complete;
  6. **the loop passes the flags through and logs the diagnostics** (a CPU
     `--dry-run` of tools/expert_loop.py with `--train-extra --bc-target
     dist --bc-value-coef 0.5`).

    python -m pytest tests/python/test_search_targets.py -q
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.bc import (BC_VERSION, BCDataset, NACT, NPAD, N_SCALAR,  # noqa: E402
                        check_probs, count_probs, decision_gamma,
                        gumbel_improved_probs, load_bc_arrays,
                        make_line_reward, onehot_probs, returns_to_go,
                        save_bc_dataset, survivor_probs)
from surfgym.core import ACTION_NVEC, STATE_DTYPE                    # noqa: E402
from surfgym.tick import TickClock                                    # noqa: E402
from train_fast import (NEG, NVEC, HeadPacker, Policy,                # noqa: E402
                        logprob_entropy_padded)

MAIN_MAP = Path("C:/RL_Surf/maps/surf_src_cannonball.bsp")
CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# helpers: a BC file of either version, and the two loss expressions
# --------------------------------------------------------------------------
def _rows(n, seed=0):
    rng = np.random.default_rng(seed)
    st = np.zeros(n, STATE_DTYPE)
    st["origin"] = rng.normal(size=(n, 3)) * 100.0
    st["velocity"] = rng.normal(size=(n, 3)) * 500.0
    st["yaw"] = rng.uniform(-180, 180, n)
    st["pitch"] = rng.uniform(-30, 30, n)
    st["tick"] = np.arange(n)
    scal = rng.normal(size=(n, N_SCALAR)).astype(np.float32)
    latch = (rng.random(n) > 0.5).astype(np.float32)
    act = np.stack([rng.integers(0, v, n) for v in ACTION_NVEC], 1).astype(np.int64)
    w = (rng.random(n) + 0.1).astype(np.float32)
    return st, scal, latch, act, w


def _write(path, n=17, seed=0, probs=None, zret=None, zmask=None, meta=None):
    st, scal, latch, act, w = _rows(n, seed)
    m = {"obs_reward": True, "n_latch": 1, "act_every": 3, "lines": 1}
    m.update(meta or {})
    save_bc_dataset(path, st, scal, latch, act, w,
                    np.zeros(n, np.int32), m,
                    probs=probs, zret=zret, zmask=zmask)
    return st, scal, latch, act, w


def _old_bc_loss(padded, act, w):
    """train_fast's bc_loss_fn body BEFORE this change (the argmax NLL)."""
    logp, _ = logprob_entropy_padded(padded, act)
    return -(logp * w).sum() / w.sum().clamp_min(1e-6)


def _new_bc_loss(padded, act, w, probs, dist: bool):
    """the shipped body's policy half, both branches."""
    logp, _ = logprob_entropy_padded(padded, act)
    nll = -(logp * w).sum() / w.sum().clamp_min(1e-6)
    lsm = F.log_softmax(padded, dim=-1)
    ce_row = -(probs * lsm).sum(-1).sum(-1)
    ce = (ce_row * w).sum() / w.sum().clamp_min(1e-6)
    return (ce if dist else nll), nll, ce


# ==========================================================================
# 1. an old BC file is still an old BC file
# ==========================================================================
def test_writer_emits_version_1_when_handed_no_target(tmp_path):
    f = tmp_path / "v1.npz"
    _write(f)
    z = np.load(f, allow_pickle=False)
    assert int(z["version"]) == 1
    # exactly the keys the pre-flag writer wrote, in its order
    assert list(z.files) == ["version", "states", "scal", "latch", "actions",
                            "weights", "line_id", "meta"]
    assert BC_VERSION == 2                     # the writer CAN write 2


def test_version_1_file_loads_and_gets_the_version_1_meaning(tmp_path):
    f = tmp_path / "v1.npz"
    _st, _sc, _la, act, _w = _write(f)
    a = load_bc_arrays(f)
    assert a["version"] == 1 and not a["has_probs"] and not a["has_value"]
    assert np.array_equal(a["probs"], onehot_probs(act))
    assert not a["zret"].any() and not a["zmask"].any()
    ds = BCDataset(f, CPU, n_latch=1, obs_reward=True)
    assert ds.version == 1 and ds.probs is None and ds.value_rows == 0
    # sample_all synthesises the one-hot per BATCH, so no (n, 6, 15) tensor
    # is ever materialised for a file that has no distribution in it
    s, p, at, w, pr, z, zm, pv = ds.sample_all(6)
    assert pr.shape == (6, NACT, NPAD) and pv is None
    assert torch.equal(pr.sum(-1), torch.ones(6, NACT))
    assert torch.equal(pr.argmax(-1), at)
    assert not z.any() and not zm.any()


def test_old_file_gives_a_byte_identical_loss_under_the_defaults(tmp_path):
    torch.manual_seed(3)
    f = tmp_path / "v1.npz"
    _write(f, n=23, seed=5)
    ds = BCDataset(f, CPU, n_latch=1, obs_reward=True, seed=11)
    packer = HeadPacker(CPU)
    padded = packer.pad(torch.randn(23, sum(NVEC)))
    act, w = ds.act, ds.w
    probs = torch.as_tensor(onehot_probs(act.numpy()))
    # the loss the trainer computes with --bc-target argmax against the
    # exact expression that shipped
    old = _old_bc_loss(padded, act, w)
    new, nll, ce = _new_bc_loss(padded, act, w, probs, dist=False)
    assert torch.equal(old, new) and torch.equal(old, nll)
    # ...and the DIST branch on the same version-1 file is the same number
    assert torch.equal(old, ce)


# ==========================================================================
# 2. the distribution target reduces to the one-hot EXACTLY
# ==========================================================================
def test_dist_on_a_one_hot_target_is_the_argmax_loss_bit_for_bit():
    torch.manual_seed(7)
    packer = HeadPacker(CPU)
    B = 33
    logits = torch.randn(B, sum(NVEC)) * 4.0        # wide: big NEG gaps
    padded = packer.pad(logits)
    assert float(padded.min()) < -1e29              # the NEG slots are there
    assert float(padded.min()) == float(torch.tensor(NEG))
    act = torch.stack([torch.randint(0, n, (B,)) for n in NVEC], 1)
    w = torch.rand(B) + 0.1
    probs = torch.as_tensor(onehot_probs(act.numpy()))
    dist, nll, ce = _new_bc_loss(padded, act, w, probs, dist=True)
    assert torch.equal(dist, ce)
    assert torch.equal(ce, nll), (float(ce), float(nll))
    # the padded slots contribute EXACTLY zero, not a rounded zero: the
    # target is 0 there and 0 * (-1e30) is -0.0
    lsm = F.log_softmax(padded, -1)
    assert float((probs * lsm)[probs == 0.0].abs().max()) == 0.0
    # and the gradient is the same gradient
    lg = logits.clone().requires_grad_(True)
    _old_bc_loss(packer.pad(lg), act, w).backward()
    g0 = lg.grad.clone()
    lg2 = logits.clone().requires_grad_(True)
    _new_bc_loss(packer.pad(lg2), act, w, probs, dist=True)[0].backward()
    assert torch.equal(g0, lg2.grad)


def test_a_real_distribution_moves_the_loss_and_stays_a_distribution():
    torch.manual_seed(8)
    packer = HeadPacker(CPU)
    B = 12
    padded = packer.pad(torch.randn(B, sum(NVEC)))
    act = torch.stack([torch.randint(0, n, (B,)) for n in NVEC], 1)
    w = torch.ones(B)
    oh = onehot_probs(act.numpy())
    soft = 0.7 * oh + 0.3 / np.asarray(ACTION_NVEC, np.float32)[None, :, None] \
        * (np.arange(NPAD)[None, None, :]
           < np.asarray(ACTION_NVEC)[None, :, None])
    soft = (soft / soft.sum(-1, keepdims=True)).astype(np.float32)
    check_probs(soft, act.numpy())
    pt = torch.as_tensor(soft)
    dist, nll, ce = _new_bc_loss(padded, act, w, pt, dist=True)
    assert not torch.equal(ce, nll)
    # cross-entropy to a SOFTER target is at least the entropy of the target
    ent = float(-(soft * np.log(np.clip(soft, 1e-12, None))).sum(-1).sum(-1)
                .mean())
    assert float(ce) >= ent - 1e-4


# ==========================================================================
# 3. the survivor distribution is the search's own
# ==========================================================================
def test_survivor_probs_is_the_prefix_matched_share():
    # three lineages from one spawn. Decision 0: all three agree on head 0
    # bin 5 except line 2, which takes 6. Decision 1: lines 0 and 1 split.
    def tab(*rows):
        return np.array(rows, np.int64)

    t0 = tab([5, 0, 1, 1, 0, 0], [3, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0])
    t1 = tab([5, 0, 1, 1, 0, 0], [4, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0])
    t2 = tab([6, 0, 1, 1, 0, 0], [9, 0, 1, 1, 0, 0])
    sp = survivor_probs([t0, t1, t2], None, 0)
    assert sp.shape == (3, NACT, NPAD)
    check_probs(sp)
    # decision 0 sees all three: 2/3 on bin 5, 1/3 on bin 6
    assert np.allclose(sp[0, 0, 5], 2 / 3) and np.allclose(sp[0, 0, 6], 1 / 3)
    # decision 1 sees only the two that took bin 5 at decision 0
    assert np.allclose(sp[1, 0, 3], 0.5) and np.allclose(sp[1, 0, 4], 0.5)
    # decision 2: line 1 diverged at decision 1, so only line 0 is left ->
    # its own one-hot, which is exactly the CAT target
    assert np.array_equal(sp[2], onehot_probs(t0[2][None, :])[0])
    # every head that never varies is a one-hot at every decision
    for h in range(1, NACT):
        assert sp[:, h].max(-1).min() == 1.0


def test_survivor_probs_weights_by_the_line_weight_and_is_one_hot_alone():
    t0 = np.array([[5, 0, 1, 1, 0, 0]], np.int64)
    t1 = np.array([[6, 0, 1, 1, 0, 0]], np.int64)
    sp = survivor_probs([t0, t1], [3.0, 1.0], 0)
    assert np.allclose(sp[0, 0, 5], 0.75) and np.allclose(sp[0, 0, 6], 0.25)
    # a single line is its own one-hot - the old behaviour as a special case
    assert np.array_equal(survivor_probs([t0], None, 0),
                          onehot_probs(t0))
    assert np.array_equal(count_probs(t0), onehot_probs(t0)[0])


def _dagger_stubs():
    """The deterministic FakeCore/StubDecider fixtures of
    tests/python/test_expert_dagger.py - ONE definition of them, because
    the population target has to be checked against the same known
    cloning history that file's own label assertions use."""
    sys.path.insert(0, str(ROOT / "tests" / "python"))
    import test_expert_dagger as ted
    return ted


def test_relabel_windows_emits_the_population_distribution():
    """The DAgger label carries the SURVIVORS' first decisions, and after
    cloning that IS the truncation-selection visit count."""
    ted = _dagger_stubs()
    from surfgym.dagger import relabel_windows

    # the fixture of test_window_labels_are_the_winning_lineages_first_
    # decisions: group 0 = envs 0-3 (the greedy env 0 is fastest and is
    # cloned over all three laggards), group 1 = envs 4-7 where env 5 leads,
    # is cloned into 6 and 7, dies at tick 60, and env 7 (carrying env 5's
    # decision 0) is cloned back over 5 and 6.
    k, copies, H, R = 3, 4, 30, 10
    speed = [50, 20, 10, 30, 10, 60, 20, 30]
    death = [10 ** 6] * 8
    death[5] = 60
    field = ted.LineField(goal_x=1e9)
    core = ted.FakeCore(8, speed, death)
    res = relabel_windows(core, lambda f, m: ted.StubDecider(8, k, f),
                          lambda: (None, None), field, ted.line_scorer(field),
                          ted._bank_of(2), k, copies, H, R,
                          elite_frac=0.25, n_greedy=1, label_decisions=2)
    r0, r1 = res
    for r in (r0, r1):
        assert r["label_probs"].shape == (2, NACT, NPAD)
        check_probs(r["label_probs"])
        assert r["label_alive"] == copies
    # group 0: every laggard was cloned from the greedy env, so the whole
    # population descends from ONE first decision - the target collapses to
    # the old one-hot, which is the point: nothing is lost where the search
    # agreed with itself
    assert np.array_equal(r0["label_probs"],
                          onehot_probs(r0["label_acts"][:2]))
    # group 1: env 4 kept its own decision 0, envs 5/6/7 all carry env 5's.
    # act_of(i, d) puts i % 15 in head 0, so the target is 1/4 on bin 4 and
    # 3/4 on bin 5 - a real distribution, and exactly the survivor count.
    h0 = r1["label_probs"][0, 0]
    assert h0[4] == pytest.approx(0.25) and h0[5] == pytest.approx(0.75)
    assert h0.sum() == pytest.approx(1.0)
    assert int(r1["label_acts"][0][0]) == 5          # the winner is env 5
    # label decision 1 narrows to the winner's own descendants (5, 6, 7),
    # which agree, so it is the winner's one-hot again
    assert np.array_equal(r1["label_probs"][1],
                          onehot_probs(r1["label_acts"][1:2])[0])
    # heads nothing varies on stay one-hot at every decision
    for h in range(2, NACT):
        assert r1["label_probs"][:, h].max(-1).min() == 1.0
    s = summarize_target(res)
    assert s["target_top1_mean"] is not None


def summarize_target(res):
    from surfgym.dagger import summarize_results
    return summarize_results(res)


def test_a_finished_window_counts_the_population_at_the_crossing():
    """A goal crossing CLEARS `valid` for the finisher one tick later, so
    the population is snapshotted at the crossing itself - otherwise the
    winning lineage would be missing from its own target."""
    ted = _dagger_stubs()
    from surfgym.dagger import relabel_windows

    k, copies, H = 3, 4, 30
    # group 0 finishes at tick 20 (env 2 at 50/tick), before the first
    # cloning at tick 75 - so all four first decisions are still distinct
    speed = [10, 20, 50, 30, 10, 20, 30, 40]
    death = [10 ** 6] * 4 + [10] * 4
    field = ted.LineField(goal_x=1000.0)
    core = ted.FakeCore(8, speed, death, goal_x=1000.0)
    res = relabel_windows(core, lambda f, m: ted.StubDecider(8, k, f),
                          lambda: (None, None), field, ted.line_scorer(field),
                          ted._bank_of(2), k, copies, H, 25, 0.25, 1,
                          label_decisions=3)
    r0, r1 = res
    assert r0["finished"] and r0["winner"] == 2
    lp = r0["label_probs"]
    check_probs(lp)
    assert lp.shape[0] == len(r0["label_acts"])
    # four live copies, four distinct head-0 bins (act_of puts i % 15 there)
    assert all(lp[0, 0, i] == pytest.approx(0.25) for i in range(4))
    assert r0["label_alive"] == 4
    # the extinct group has no label and therefore no target
    assert r1["extinct"] and r1["label_probs"] is None


# ==========================================================================
# 3b. the Gumbel-improved policy (Danihelka 2022), the second mode
# ==========================================================================
def test_gumbel_improved_probs_is_softmax_of_logits_plus_sigma():
    rng = np.random.default_rng(1)
    logits = np.full((NACT, NPAD), NEG)
    for h, n in enumerate(ACTION_NVEC):
        logits[h, :n] = rng.normal(size=n)
    counts = np.zeros((NACT, NPAD))
    q = np.zeros((NACT, NPAD))
    # nothing visited -> the policy itself, unchanged (no improvement to apply)
    p0 = gumbel_improved_probs(logits, q, counts, 0.0)
    check_probs(p0[None])
    for h, n in enumerate(ACTION_NVEC):
        want = np.exp(logits[h, :n] - logits[h, :n].max())
        want /= want.sum()
        assert np.allclose(p0[h, :n], want, atol=1e-6)
    # every action visited once with a flat Q is also the policy: sigma is a
    # constant per head and a constant shift does not move a softmax
    counts[:] = (np.arange(NPAD)[None, :]
                 < np.asarray(ACTION_NVEC)[:, None]).astype(float)
    q[:] = 0.5 * counts
    p1 = gumbel_improved_probs(logits, q, counts, 0.5)
    assert np.allclose(p1[0, :ACTION_NVEC[0]], p0[0, :ACTION_NVEC[0]],
                       atol=1e-6)
    # raising ONE action's Q raises exactly that action's probability
    q2 = q.copy()
    q2[0, 3] = 1.0
    p2 = gumbel_improved_probs(logits, q2, counts, 0.5)
    assert p2[0, 3] > p1[0, 3]
    assert (p2[0, :ACTION_NVEC[0]][np.arange(ACTION_NVEC[0]) != 3]
            <= p1[0, :ACTION_NVEC[0]][np.arange(ACTION_NVEC[0]) != 3] + 1e-9
            ).all()
    # c_scale scales the strength of the improvement
    p3 = gumbel_improved_probs(logits, q2, counts, 0.5, c_scale=0.0)
    assert np.allclose(p3[0, :ACTION_NVEC[0]], p0[0, :ACTION_NVEC[0]],
                       atol=1e-6)


def test_relabel_windows_gumbel_mode_uses_the_policys_root_logits():
    """--label-target gumbel: the window's decision-0 target becomes
    `softmax(logits + sigma(completedQ))` over the same population. The
    prior comes from the decider's `last_logits`; a decider that does not
    expose one falls back to the counts (the stub in
    tests/python/test_expert_dagger.py has none, which is what keeps the
    default path unchanged)."""
    ted = _dagger_stubs()
    from surfgym.dagger import relabel_windows

    k, copies, H, R = 3, 4, 30, 10
    speed = [50, 20, 10, 30, 10, 60, 20, 30]
    death = [10 ** 6] * 8
    death[5] = 60
    field = ted.LineField(goal_x=1e9)
    rng = np.random.default_rng(3)
    logits = np.full((8, NACT, NPAD), NEG)
    for h, n in enumerate(ACTION_NVEC):
        logits[:, h, :n] = rng.normal(size=(8, n)) * 2.0

    class LogitDecider(ted.StubDecider):
        def act(self, obs):
            a = super().act(obs)
            self.last_logits = logits
            return a

    def run(target, c_scale=0.1):
        core = ted.FakeCore(8, speed, death)
        return relabel_windows(core, lambda f, m: LogitDecider(8, k, f),
                               lambda: (None, None), field,
                               ted.line_scorer(field), ted._bank_of(2), k,
                               copies, H, R, elite_frac=0.25, n_greedy=1,
                               label_decisions=2, label_target=target,
                               c_scale=c_scale)

    g = run("gumbel")[1]["label_probs"]
    c = run("count")[1]["label_probs"]
    check_probs(g)
    check_probs(c)
    # decision 0 is the improved policy, not the survivor count
    assert not np.allclose(g[0], c[0])
    # at c_scale 0 sigma vanishes and it IS the policy's own softmax
    g0 = run("gumbel", c_scale=0.0)[1]["label_probs"]
    for h, n in enumerate(ACTION_NVEC):
        want = np.exp(logits[5, h, :n] - logits[5, h, :n].max())
        want /= want.sum()
        assert np.allclose(g0[0, h, :n], want, atol=1e-6)
    # deeper label decisions stay survivor counts in every mode
    assert np.allclose(g[1], c[1])
    # an unknown mode is refused rather than silently ignored
    with pytest.raises(ValueError):
        run("visits")


# ==========================================================================
# 4. the return-to-go is the trainer's arithmetic
# ==========================================================================
def test_returns_to_go_matches_a_hand_computed_line_at_both_ticks():
    # a 4-decision line: three shaping steps and a +50 finish bonus
    rew = [0.5, -0.25, 1.5, 50.0]
    gamma, k = 0.9995, 4
    # 10 ms: gamma is DEFINED per 10 ms tick, so g = gamma**act_every
    g10 = decision_gamma(gamma, k, 10.0)
    assert g10 == gamma ** k
    z = returns_to_go(rew, g10)
    want = [0.0] * 4
    want[3] = 50.0
    want[2] = 1.5 + g10 * want[3]
    want[1] = -0.25 + g10 * want[2]
    want[0] = 0.5 + g10 * want[1]
    assert np.allclose(z, want, atol=1e-6)
    # 7.63 ms: the pattern is [8, 8, 7] = 7.6667 ms, and the SAME HORIZON in
    # seconds means gamma ** (tick/10) per tick, THEN ** act_every. Getting
    # the two exponents the wrong way round is the failure this pins.
    tk = TickClock(7.63)
    assert tk.pattern == [8, 8, 7] and abs(tk.ms - 23 / 3) < 1e-9
    g763 = decision_gamma(gamma, k, 7.63)
    assert abs(g763 - (gamma ** (tk.ms / 10.0)) ** k) < 1e-15
    assert g763 > g10                       # a shorter tick discounts less
    z2 = returns_to_go(rew, g763)
    w2 = [0.0] * 4
    w2[3] = 50.0
    w2[2] = 1.5 + g763 * w2[3]
    w2[1] = -0.25 + g763 * w2[2]
    w2[0] = 0.5 + g763 * w2[1]
    assert np.allclose(z2, w2, atol=1e-6)
    # explicitly: the 20 s horizon is unmoved, so the per-DECISION discount
    # at 131 Hz is the 100 Hz one raised to the ratio of decision durations
    assert abs(g763 - g10 ** (tk.ms / 10.0)) < 1e-12
    # a one-element line is its own reward; an empty one is empty
    assert returns_to_go([2.0], 0.5)[0] == pytest.approx(2.0)
    assert len(returns_to_go([], 0.5)) == 0


def test_make_line_reward_mirrors_the_trainers_construction():
    class F0:
        def sample(self, o):
            return np.zeros(len(np.asarray(o).reshape(-1, 3)))

    cfg = {"gamma": 0.9995, "time_pen": 0.005, "success_bonus": 50.0,
           "race_shaping": 1.0, "max_step": 100.0, "stall_secs": 15.0,
           "stall_eps": 32.0, "int_coef": 0.25}
    rf, info = make_line_reward(cfg, F0(), d0=200000.0, k=4, tick_ms=10.0)
    assert rf.scale == 100.0 / 200000.0
    assert rf.time_pen == 0.005 and rf.success_bonus == 50.0
    assert rf.every == 1                       # per TICK, like the rollout
    assert info["dropped"] == ["int_coef"] and rf.int_coef == 0.0
    assert info["gamma_decision"] == pytest.approx(0.9995 ** 4)
    # --tick-ms: the per-tick amounts scale so the per-SECOND ones do not
    rf2, i2 = make_line_reward(cfg, F0(), 200000.0, 4, tick_ms=7.63)
    tk = TickClock(7.63)
    assert rf2.time_pen == pytest.approx(0.005 * tk.ms / 10.0)
    assert i2["gamma_tick"] == pytest.approx(0.9995 ** (tk.ms / 10.0))
    # --reward-per-decision moves the cadence, exactly as the trainer does
    rf3, i3 = make_line_reward(dict(cfg, reward_per_decision=True), F0(),
                               200000.0, 4)
    assert rf3.every == 4 and i3["every"] == 4
    # the two objectives a replay cannot mirror are refused, not approximated
    with pytest.raises(SystemExit):
        make_line_reward(dict(cfg, race_arc=True), F0(), 200000.0, 4)
    with pytest.raises(SystemExit):
        make_line_reward(dict(cfg, race_kill_aware=1), F0(), 200000.0, 4)


# ==========================================================================
# 5. the value term: off by default, masked, privileged when asked
# ==========================================================================
def _value_loss(v, z, w, zmask, mu=0.0, sig=1.0):
    """the shipped body's value half."""
    vm = w * zmask
    err = (v - (z - mu) / sig).pow(2)
    return (err * vm).sum() / vm.sum().clamp_min(1e-6)


def test_value_coef_zero_is_zero_gradient_bit_for_bit():
    torch.manual_seed(2)
    p = Policy(N_SCALAR + 8 * 4, 8, 4, emb=16, hidden=12)
    obs = torch.randn(9, N_SCALAR + 32)
    act = torch.stack([torch.randint(0, n, (9,)) for n in NVEC], 1)
    w = torch.rand(9) + 0.1
    packer = HeadPacker(CPU)
    probs = torch.as_tensor(onehot_probs(act.numpy()))
    z = torch.randn(9)
    zm = (torch.rand(9) > 0.5).float()

    def run(coef):
        p.zero_grad()
        logits, value = p.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:])
        padded = packer.pad(logits.float())
        loss = _new_bc_loss(padded, act, w, probs, dist=False)[0]
        if coef > 0.0:
            loss = loss + coef * 0.5 * _value_loss(
                value.float().reshape(-1), z, w, zm)
        loss.backward()
        # the value head gets NO gradient from the policy half, so its
        # .grad is None at coef 0 - which is the point being pinned
        return [torch.zeros_like(q) if q.grad is None else q.grad.clone()
                for q in p.parameters()]

    g0, g1 = run(0.0), run(0.0)
    assert all(torch.equal(a, b) for a, b in zip(g0, g1))
    g2 = run(0.5)
    assert not all(torch.equal(a, b) for a, b in zip(g0, g2))


def test_the_value_target_is_masked_to_complete_returns():
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    z = torch.tensor([1.0, 0.0, 3.0, 0.0])
    w = torch.ones(4)
    zm = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # rows 1 and 3 have no return; a perfect fit on 0 and 2 is loss 0
    assert float(_value_loss(v, z, w, zm)) == pytest.approx(0.0)
    # no masked-in row at all -> the term is 0, not NaN
    assert float(_value_loss(v, z, w, torch.zeros(4))) == pytest.approx(0.0)
    # --ret-norm: z enters the critic's own frame
    mu, sig = 2.0, 4.0
    got = _value_loss(torch.tensor([0.0, 0.0, 0.0, 0.0]), z, w, zm, mu, sig)
    want = (((0 - (1.0 - mu) / sig) ** 2) + ((0 - (3.0 - mu) / sig) ** 2)) / 2
    assert float(got) == pytest.approx(want)


def test_version_2_file_round_trips_and_the_priv_block_is_built_once(tmp_path):
    f = tmp_path / "v2.npz"
    n = 13
    st, scal, latch, act, w = _rows(n, seed=4)
    probs = onehot_probs(act) * 0.8
    for h, nv in enumerate(ACTION_NVEC):        # spread the rest evenly
        probs[:, h, :nv] += 0.2 / nv
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    zret = np.linspace(-1.0, 40.0, n).astype(np.float32)
    zmask = (np.arange(n) % 3 != 0).astype(np.float32)
    save_bc_dataset(f, st, scal, latch, act, w, np.zeros(n, np.int32),
                    {"obs_reward": True, "n_latch": 1}, probs=probs,
                    zret=zret, zmask=zmask)
    a = load_bc_arrays(f)
    assert a["version"] == 2 and a["has_probs"] and a["has_value"]
    assert np.allclose(a["probs"], probs) and np.allclose(a["zret"], zret)

    seen = {"n": 0}

    def priv_fn(states, lat):
        seen["n"] += 1
        assert len(states) == n and len(lat) == n
        # the trainer's closure reads the STORED state and latch column
        assert np.allclose(np.asarray(lat), latch)
        return np.column_stack([states["origin"], states["velocity"],
                                np.zeros((n, 4))]).astype(np.float32)

    ds = BCDataset(f, CPU, n_latch=1, obs_reward=True, priv_fn=priv_fn)
    assert seen["n"] == 1                      # ONCE, not per minibatch
    assert ds.priv is not None and ds.priv.shape == (n, 10)
    assert ds.value_rows == int(zmask.sum())
    _s, _p, at, ww, pr, z, zm, pv = ds.sample_all(5)
    assert pv.shape == (5, 10) and pr.shape == (5, NACT, NPAD)
    assert torch.allclose(pr.sum(-1), torch.ones(5, NACT), atol=1e-5)
    # the same draw as sample(): one randint call, so the RNG stream matches
    d2 = BCDataset(f, CPU, n_latch=1, obs_reward=True, priv_fn=priv_fn)
    d3 = BCDataset(f, CPU, n_latch=1, obs_reward=True, priv_fn=priv_fn)
    s2 = d2.sample(5)
    s3 = d3.sample_all(5)
    for i in range(4):
        assert torch.equal(s2[i], s3[i])


def test_check_probs_refuses_a_target_that_is_not_a_distribution():
    act = np.zeros((2, NACT), np.int64)
    p = onehot_probs(act)
    check_probs(p, act)
    bad = p.copy()
    bad[0, 0, 0] = 0.5
    with pytest.raises(ValueError):
        check_probs(bad)
    bad2 = p.copy()
    bad2[0, 2, 5] = 1.0                        # head 2 has only 3 bins
    bad2[0, 2, 0] = 0.0
    with pytest.raises(ValueError):
        check_probs(bad2)
    with pytest.raises(ValueError):
        check_probs(p[:, :, :3])
    # a target with no mass on the stored action would be an infinite NLL
    other = np.full((2, NACT), 1, np.int64)
    with pytest.raises(ValueError):
        check_probs(p, other)


# ==========================================================================
# 6. the flag surface + the merge
# ==========================================================================
def test_the_flags_exist_are_recorded_and_are_guarded():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    for flag in ("--bc-target", "--bc-value-coef"):
        assert f'"{flag}"' in src
    for key in ("bc_target", "bc_value_coef"):
        assert f'"{key}":' in src              # lands in run.json
        assert f'ck_cfg.get("{key}")' not in src   # never auto-restored
    # both are Python constants at trace time, so the compiled graph a
    # control run traces is the one it always traced
    assert "BCV = float(args.bc_value_coef or 0.0) if args.bc_file else 0.0" in src
    assert 'BCDIST = bool(args.bc_file) and args.bc_target == "dist"' in src
    assert "if BCV > 0.0:" in src
    assert "loss = loss + bc_coef_t * _lb" in src
    # the four diagnostics are columns of progress.csv, appended LAST
    i = src.index('CSV_COLS += ["bc/ce_dist", "bc/head_acc", "bc/joint_acc", '
                  '"bc/value_mse"]')
    assert "csv_f = csv_w = None" in src[i:i + 2000]
    rec = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for key in ("bc_target", "bc_value_coef"):
        assert f'"{key}"' in rec               # TRAIN_ONLY: recordable
    pb = (ROOT / "tools" / "plan_to_bc.py").read_text(encoding="utf-8")
    for flag in ("--no-search-target", "--no-value-target"):
        assert f'"{flag}"' in pb
    ed = (ROOT / "tools" / "expert_dagger.py").read_text(encoding="utf-8")
    assert '"--label-target"' in ed and '"--dagger-label-target"' in ed


def test_merge_keeps_version_1_and_promotes_only_when_there_is_a_target(
        tmp_path):
    from surfgym.dagger import DAGGER_LINE_ID, merge_bc_datasets
    elite = tmp_path / "elite.npz"
    st, scal, latch, act, w = _write(elite, n=5, seed=9)
    rows = {"states": st[:2], "scal": scal[:2], "latch": latch[:2],
            "actions": act[:2], "weights": w[:2],
            "probs": onehot_probs(act[:2]),
            "zret": np.zeros(2, np.float32), "zmask": np.zeros(2, np.float32),
            "row0_mismatch": 0}
    out1 = tmp_path / "m1.npz"
    merge_bc_datasets(elite, rows, out1, {}, n_latch=1, obs_reward=True)
    assert int(np.load(out1)["version"]) == 1      # nothing new to store
    a1 = load_bc_arrays(out1)
    assert a1["line_id"][-1] == DAGGER_LINE_ID and len(a1["states"]) == 7
    # a REAL distribution on the dagger rows promotes the merged file
    soft = rows["probs"].copy()
    soft[0, 0] = 0.0
    soft[0, 0, :ACTION_NVEC[0]] = 1.0 / ACTION_NVEC[0]
    rows2 = dict(rows, probs=soft)
    out2 = tmp_path / "m2.npz"
    merge_bc_datasets(elite, rows2, out2, {}, n_latch=1, obs_reward=True)
    a2 = load_bc_arrays(out2)
    assert a2["version"] == 2 and a2["has_probs"]
    # the ELITE half keeps its version-1 meaning (the one-hot of its actions)
    assert np.array_equal(a2["probs"][:5], onehot_probs(act))
    assert np.allclose(a2["probs"][5], soft[0])
    # a version-2 elite file keeps its targets through the merge
    e2 = tmp_path / "elite2.npz"
    z = np.linspace(0, 9, 5).astype(np.float32)
    save_bc_dataset(e2, st, scal, latch, act, w, np.zeros(5, np.int32),
                    {"obs_reward": True, "n_latch": 1},
                    zret=z, zmask=np.ones(5, np.float32))
    out3 = tmp_path / "m3.npz"
    merge_bc_datasets(e2, rows, out3, {}, n_latch=1, obs_reward=True)
    a3 = load_bc_arrays(out3)
    assert a3["version"] == 2
    assert np.allclose(a3["zret"][:5], z) and a3["zmask"][:5].all()
    assert not a3["zmask"][5:].any()           # the relabel rows carry none


# ==========================================================================
# 7. end to end: the loop passes the flags and the trainer logs them
# ==========================================================================
DRY = ROOT / "runs" / "dryST"
dry_ok = pytest.mark.skipif(
    not (DRY / "expert_summary.jsonl").exists(),
    reason="run: python tools/expert_loop.py scratch --name dryST --dry-run "
           "--train-extra --bc-target dist --bc-value-coef 0.5")


def _csv_rows(p):
    return list(csv.DictReader(p.open(newline="", encoding="utf-8")))


@dry_ok
def test_expert_loop_dry_run_passes_the_flags_and_logs_the_diagnostics():
    """The CPU dry run completed, the trainer got the flags through
    --train-extra, the BC file carries the search target, and the four
    diagnostics are in progress.csv and bc_log.csv."""
    tdir = DRY / "round_0" / "train"
    assert tdir.exists(), sorted(p.name for p in (DRY / "round_0").iterdir())
    cfg = json.loads((tdir / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["bc_target"] == "dist"
    assert cfg["bc_value_coef"] == 0.5
    assert cfg["bc_file"]
    # the BC file the loop's own plan_to_bc built carries both targets
    a = load_bc_arrays(Path(cfg["bc_file"]))
    assert a["version"] == 2 and a["has_probs"] and a["has_value"]
    check_probs(a["probs"], a["actions"])
    meta = a["meta"]
    assert meta["target_kind"] == "distribution" and meta["value_target"]
    assert meta["value"]["gamma_decision"] == pytest.approx(
        decision_gamma(meta["value"]["gamma"], meta["act_every"],
                       meta["tick_ms_requested"]))
    # the dry run's planner window (500 ticks) ends with every lineage
    # still ALIVE, so no row carries a complete return and the value term
    # is honestly inert - warned about at startup, and blank in the CSV
    assert meta["value"]["rows_masked_in"] == 0
    rows = _csv_rows(tdir / "progress.csv")
    assert rows
    for c in ("bc/ce_dist", "bc/head_acc", "bc/joint_acc", "bc/value_mse"):
        assert c in rows[0], c
    live = [r for r in rows if r["bc/ce_dist"] != ""]
    assert live, "no iteration logged a BC diagnostic"
    for r in live:
        assert 0.0 <= float(r["bc/head_acc"]) <= 1.0
        # all six heads agreeing is at most any one head agreeing
        assert 0.0 <= float(r["bc/joint_acc"]) <= float(r["bc/head_acc"]) + 1e-9
        assert float(r["bc/ce_dist"]) >= 0.0
        assert r["bc/value_mse"] == ""          # no complete return
    bl = _csv_rows(tdir / "bc_log.csv")
    assert bl and {"bc/ce_dist", "bc/joint_acc", "bc/value_mse",
                   "bc/value_rows"} <= set(bl[0])
    # the pre-flag header is still a strict PREFIX of the new one
    head = (tdir / "bc_log.csv").read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("time/total_timesteps,bc/coef,bc/loss,bc/acc,")
    assert all(float(r["bc/value_rows"]) == 0.0 for r in bl)


VALUE_RUN = DRY / "round_0" / "trainv"
value_ok = pytest.mark.skipif(
    not (VALUE_RUN / "progress.csv").exists(),
    reason="run the dry run, then train_fast on bc_value.npz (see the "
           "docstring of this test)")


@value_ok
def test_the_value_term_fits_the_planner_return_with_a_privileged_critic():
    """End to end on the dry run's own states with `zmask` set: the value
    term is live, reads the privileged block, and the error FALLS.

    Reproduce:

        python - <<'PY'
        from surfgym.bc import load_bc_arrays, save_bc_dataset
        import numpy as np
        a = load_bc_arrays("runs/dryST/round_0/bc.npz")
        save_bc_dataset("runs/dryST/round_0/bc_value.npz", a["states"],
                        a["scal"], a["latch"], a["actions"], a["weights"],
                        a["line_id"], a["meta"], probs=a["probs"],
                        zret=a["zret"],
                        zmask=np.ones(len(a["zret"]), np.float32))
        PY
        python python/train_fast.py --ckpt runs/dryST/round_0/scratch/ckpt_final.pt \\
            --run dryST/round_0/trainv --steps 16384 --map <main>/surf_src_cannonball.bsp \\
            --bc-file runs/dryST/round_0/bc_value.npz --bc-coef 0.5 --bc-batch 64 \\
            --envs 64 --emb 64 --hidden 64 --n-steps 8 --minibatches 2 --epochs 1 \\
            --ep-ticks 3000 --bc-target dist --bc-value-coef 0.5 --priv-critic 1
    """
    cfg = json.loads((VALUE_RUN / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert cfg["bc_value_coef"] == 0.5 and cfg["priv_critic"] == 1
    rows = [r for r in _csv_rows(VALUE_RUN / "progress.csv")
            if r["bc/value_mse"] != ""]
    assert len(rows) >= 2, "the value term never ran"
    v = [float(r["bc/value_mse"]) for r in rows]
    assert all(x >= 0.0 and np.isfinite(x) for x in v)
    # the critic is being pulled toward z: the error falls monotonically
    # over the run's few iterations (it is the only term touching it here
    # beyond PPO's own, and the rollout is 64 envs of noise)
    assert v[-1] < v[0], v
    bl = _csv_rows(VALUE_RUN / "bc_log.csv")
    assert all(float(r["bc/value_rows"]) > 0.0 for r in bl)
