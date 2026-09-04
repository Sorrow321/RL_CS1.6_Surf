"""--tail-weight: TailRL's tail-likelihood advantage reweighting.

Ramasubramanian, Arora, Tajwar et al., *Tail-Likelihood Reinforcement
Learning* (arXiv 2609.02987, https://github.com/Zanette-Labs/TailRL). The
objective is J = integral_0^1 log p(x, tau) dtau over reward thresholds
rather than the mean return, and its gradient is a harmonically weighted
mixture of best-of-k gradients - which is what "beat the world record" is
and what PPO's mean is not. The algorithm is one line: multiply each
rollout's advantage by w = integral_0^r dtau / p(tau), with p estimated by
counting how many of the N rollouts for the same input exceed tau.

What is pinned here:

1. ``weights_from_u`` is the paper's recurrence VERBATIM - checked against
   an independent exact-fraction implementation of the README's
   ``gap_i / survivors_i`` line and against a hand computation.
2. Theorem 1, numerically: sum_k (1/k) grad Best-of-k == grad J_TailRL, with
   the k = 1..5 terms named individually and the tail summed to convergence.
   Both sides come from autograd on a 4-atom categorical, so nothing about
   the identity is assumed by the code under test.
3. The finite-N estimator is consistent: Monte-Carlo over 20k groups of
   N = 32 reproduces the population gradient.
4. Our transplant's own invariants - mean-1 per group (which is what makes
   group frames independent), weight 0 for the worst rollout, weight 1 for
   groups too small or too flat, ties equal, and the blend interpolating.
5. ``--tail-weight 0`` is the trainer that shipped: same config dump, same
   progress.csv on every shared column, same policy weights and same Adam
   moments after a tiny CPU scratch run against the pre-flag code from git.
6. With the flag on, the run trains, logs the tail/* diagnostics, and the
   resulting policy tensors DIFFER from the untreated control's.
7. A warm resume of the round-30 finisher checkpoint works with the flag on
   (the arch is the checkpoint's; only envs / n_steps are shrunk).
8. The grouping is the fleet's own goal-distance bin and needs no binned
   reservoir, so turning the flag on cannot move the start distribution.

    python -m pytest tests/python/test_tail_weight.py -q

CPU only, tiny nets. It needs the built core and the prebaked cannonball
goal field. Running from a git worktree, point SURF_TEST_MAPS at the main
checkout's maps/ (CLAUDE.md: a worktree copy has different mtimes and every
prebaked cache misses) and SURFCORE_DLL at its built core. The resume test
additionally needs SURF_TEST_FINISHER (default
C:/RL_Surf_base/runs/research/xENT131/ckpt_10774118400.pt) and is skipped
without it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

MAPS = Path(os.environ.get("SURF_TEST_MAPS") or (ROOT / "maps"))
CANNONBALL = MAPS / "surf_src_cannonball.bsp"
GOALFIELD = MAPS / "surf_src_cannonball.goal_32.npz"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt"
                         else "libsurfcore.so"))
FINISHER = Path(os.environ.get(
    "SURF_TEST_FINISHER",
    "C:/RL_Surf_base/runs/research/xENT131/ckpt_10774118400.pt"))

needs_core = pytest.mark.skipif(
    not (CANNONBALL.exists() and DLL.exists() and GOALFIELD.exists()),
    reason="needs the built core + cannonball + its prebaked goal field")

import torch                                                   # noqa: E402

from surfgym.tailrl import (group_u, tail_weights,             # noqa: E402
                            weights_from_u)


# ==========================================================================
# (a) the weight formula IS the paper's
# ==========================================================================
def _readme_weights(r):
    """The TailRL README's line, transcribed independently and in EXACT
    rational arithmetic:

        gap_i = r_(i) - r_(i-1)  (r_(0) := 0)
        survivors_i = N - i + 1
        A_i = N * sum_{j <= i} gap_j / survivors_j
    """
    fr = [Fraction(x).limit_denominator(10 ** 9) for x in r]
    order = sorted(range(len(fr)), key=lambda i: fr[i])
    n = len(fr)
    out = [Fraction(0)] * n
    acc, prev = Fraction(0), Fraction(0)
    for rank, i in enumerate(order, start=1):
        acc += (fr[i] - prev) / (n - rank + 1)
        prev = fr[i]
        out[i] = n * acc
    return [float(x) for x in out]


def test_weights_are_the_papers_recurrence_exactly():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 8, 32):
        u = np.sort(rng.random(n))
        got = weights_from_u(u)
        want = _readme_weights(u)
        assert np.allclose(got, want, rtol=0, atol=1e-12), (n, got, want)


def test_weights_by_hand_on_a_four_rollout_group():
    """N = 4 at u = 0, 1/3, 2/3, 1. survivors 4, 3, 2, 1; gaps 0, 1/3, 1/3,
    1/3, so A = 4 * (0, 1/9, 1/9+1/6, 1/9+1/6+1/3) = 0, 4/9, 10/9, 22/9."""
    w = weights_from_u(np.array([0.0, 1 / 3, 2 / 3, 1.0]))
    assert np.allclose(w, [0.0, 4 / 9, 10 / 9, 22 / 9], atol=1e-12)
    assert abs(float(w.mean()) - 1.0) < 1e-12


def test_binary_rewards_reproduce_the_papers_rlvr_case():
    """In TailRL's own binary-reward setting every FAILED rollout has
    gap_1 = 0 and weight exactly 0, and the successes split N between
    them. Weight 0 here means "does not vote" - not "pushed down", which is
    the one place this transplant deliberately differs (no mean-centring;
    the sign lives in the GAE advantage)."""
    u = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    w = weights_from_u(u)
    assert np.allclose(w[:6], 0.0)
    assert np.allclose(w[6:], 4.0)          # 8 * (1/2), split over 2
    assert abs(float(w.mean()) - 1.0) < 1e-12


def test_ties_get_equal_weights_whatever_the_sort_order():
    u = np.array([0.0, 0.5, 0.5, 0.5, 1.0])
    w = weights_from_u(u)
    assert np.allclose(w[1:4], w[1])
    perm = np.array([4, 2, 0, 3, 1])
    assert np.allclose(weights_from_u(u[perm]), w[perm])


def test_the_top_weight_is_bounded_by_the_group_size():
    for n in (2, 4, 16, 64):
        u = np.zeros(n)
        u[-1] = 1.0                          # the extreme case: all mass low
        w = weights_from_u(u)
        assert abs(float(w.max()) - n) < 1e-9
        assert float(w.max()) <= n + 1e-9


# ==========================================================================
# (b) Theorem 1: grad J == sum_k (1/k) grad Best-of-k
# ==========================================================================
_ATOMS = torch.tensor([0.0, 0.35, 0.7, 1.0], dtype=torch.float64)


def _survivals(theta):
    """S_m = Pr(r > tau) for tau in [atom_{m-1}, atom_m), i.e. the mass at or
    above atom m. The top atom is 1.0 so p is positive on all of [0, 1) and
    log p is integrable - J is -inf otherwise, which is the paper's
    "rewards in [0, 1]" assumption biting."""
    q = torch.softmax(theta, dim=0)
    return torch.flip(torch.cumsum(torch.flip(q, (0,)), 0), (0,))


def _seg_widths():
    return _ATOMS - torch.cat([torch.zeros(1, dtype=torch.float64),
                               _ATOMS[:-1]])


def _J(theta):
    return (_seg_widths() * torch.log(_survivals(theta))).sum()


def _best_of_k(theta, k):
    """E[max of k iid draws] = integral_0^1 Pr(max > tau) dtau
    = integral_0^1 (1 - (1 - p)^k) dtau."""
    s = _survivals(theta)
    return (_seg_widths() * (1.0 - (1.0 - s) ** k)).sum()


def _grad(fn, theta):
    t = theta.clone().requires_grad_(True)
    fn(t).backward()
    return t.grad.detach().clone()


def test_best_of_k_decomposition_holds_numerically():
    theta = torch.tensor([0.3, -0.4, 0.9, 0.1], dtype=torch.float64)
    gj = _grad(_J, theta)
    # the k = 1..5 terms, named individually (k = 1 is the plain mean return)
    terms = [_grad(lambda t, k=k: _best_of_k(t, k), theta) / k
             for k in range(1, 6)]
    assert torch.allclose(terms[0], _grad(lambda t: _best_of_k(t, 1), theta))
    # the partial sums close on grad J monotonically in the residual norm
    resid = []
    part = torch.zeros_like(gj)
    for t in terms:
        part = part + t
        resid.append(float((gj - part).norm()))
    assert all(b < a for a, b in zip(resid, resid[1:])), resid
    assert resid[-1] < 0.5 * resid[0]
    # ... and the FULL harmonic sum reproduces it. The k-th term decays like
    # (1 - p_min)^{k-1}, so a few thousand terms is machine precision here.
    full = torch.zeros_like(gj)
    for k in range(1, 4001):
        full = full + _grad(lambda t, k=k: _best_of_k(t, k), theta) / k
    assert torch.allclose(full, gj, atol=1e-9), (full, gj)
    # and it is not a trivial identity: best-of-1 alone is nowhere near it
    assert float((terms[0] - gj).norm()) > 0.1 * float(gj.norm())


def test_the_finite_n_estimator_recovers_the_population_gradient():
    """g^(N) = (1/N) sum_i w(r_i) grad log pi(z_i) is consistent (the paper's
    estimator is not unbiased at finite N; it is a plug-in of the empirical
    survival). 20k groups of 32 is enough to see it land."""
    theta = torch.tensor([0.3, -0.4, 0.9, 0.1], dtype=torch.float64)
    gj = _grad(_J, theta).numpy()
    q = torch.softmax(theta, dim=0).numpy()
    atoms = _ATOMS.numpy()
    rng = np.random.default_rng(7)
    n, groups = 32, 20_000
    draws = rng.choice(len(q), size=(groups, n), p=q)
    acc = np.zeros(len(q))
    for row in draws:
        # the PAPER's frame: the rewards are already in [0, 1], so they go
        # into the recurrence raw (this is where our per-group min-max
        # transplant departs, and it is not exercised here)
        w = weights_from_u(atoms[row])
        onehot = np.zeros((n, len(q)))
        onehot[np.arange(n), row] = 1.0
        acc += (w[:, None] * (onehot - q[None, :])).sum(0) / n
    acc /= groups
    assert np.allclose(acc, gj, atol=0.02), (acc, gj)
    # the SAME machinery with flat weights gives the MEAN's gradient, which
    # is a different vector - so the agreement above is not vacuous
    assert not np.allclose(_grad(lambda t: _best_of_k(t, 1), theta).numpy(),
                           gj, atol=0.02)


# ==========================================================================
# (c) the transplant's own invariants
# ==========================================================================
def test_every_group_averages_to_one_so_frames_are_independent():
    """Per-group min-max is what makes the batch mean 1 automatically and
    stops a group of large-return episodes out-shouting a near-goal group
    whose returns are small by construction."""
    rng = np.random.default_rng(3)
    ret = np.concatenate([rng.normal(0, 1, 40),          # group 0
                          rng.normal(500, 90, 40)])      # group 1, 500x scale
    grp = np.array([0] * 40 + [1] * 40)
    w, st = tail_weights(ret, grp, min_n=4)
    assert st["groups"] == 2 and st["n_med"] == 40
    assert abs(w[:40].mean() - 1.0) < 1e-12
    assert abs(w[40:].mean() - 1.0) < 1e-12
    # shifting or scaling ONE group leaves both groups' weights untouched
    ret2 = ret.copy()
    ret2[40:] = ret2[40:] * 17.0 - 3000.0
    w2, _ = tail_weights(ret2, grp, min_n=4)
    assert np.allclose(w, w2)


def test_small_and_flat_groups_fall_back_to_weight_one():
    ret = np.array([1.0, 2.0, 3.0,          # group 0: 3 episodes < min_n 4
                    5.0, 5.0, 5.0, 5.0,     # group 1: no spread at all
                    0.0, 1.0, 2.0, 3.0])    # group 2: weighted
    grp = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    w, st = tail_weights(ret, grp, min_n=4)
    assert np.allclose(w[:7], 1.0)
    assert st["groups"] == 1 and st["n_med"] == 4.0
    assert w[7] == 0.0 and w[10] == pytest.approx(22 / 9)


def test_blend_interpolates_toward_one_and_preserves_the_mean():
    ret = np.arange(8, dtype=float)
    grp = np.zeros(8, np.int64)
    full, _ = tail_weights(ret, grp, min_n=4, blend=1.0)
    half, _ = tail_weights(ret, grp, min_n=4, blend=0.5)
    off, _ = tail_weights(ret, grp, min_n=4, blend=0.0)
    assert np.allclose(off, 1.0)
    assert np.allclose(half, 1.0 + 0.5 * (full - 1.0))
    for w in (full, half, off):
        assert abs(w.mean() - 1.0) < 1e-12


def test_ess_and_w_max_report_the_variance_traded():
    ret = np.zeros(16)
    ret[-1] = 1.0                            # one rare winner
    w, st = tail_weights(ret, np.zeros(16, np.int64), min_n=4)
    assert st["w_max"] == pytest.approx(16.0)
    assert st["ess"] == pytest.approx(1.0 / 16.0, rel=1e-9)
    flat = np.linspace(0.0, 1.0, 16)
    _, st2 = tail_weights(flat, np.zeros(16, np.int64), min_n=4)
    assert st2["ess"] > 0.5 and st2["w_max"] < 4.0


def test_time_outcome_puts_every_finisher_above_every_non_finisher():
    ret = np.array([10.0, 30.0, 20.0, 5.0])
    fin = np.array([True, False, True, False])
    secs = np.array([74.0, 0.0, 73.0, 0.0])   # secs unread for non-finishers
    u = group_u(ret, "time", fin, secs)
    assert u[0] < u[2]                        # 74.0 s below 73.0 s
    assert min(u[fin]) > max(u[~fin])
    assert u[3] < u[1]                        # non-finishers ranked by return
    w, _ = tail_weights(ret, np.zeros(4, np.int64), mode="time",
                        finished=fin, secs=secs, min_n=4)
    assert w[2] == max(w) and abs(w.mean() - 1.0) < 1e-12
    # an all-finisher group is ranked on time alone
    fin2 = np.ones(4, bool)
    secs2 = np.array([74.0, 76.0, 73.0, 75.0])
    u2 = group_u(ret, "time", fin2, secs2)
    assert list(np.argsort(u2)) == [1, 3, 0, 2]


def test_return_outcome_is_the_episode_return_ordering():
    ret = np.array([-3.0, 8.0, 0.5, 8.0])
    u = group_u(ret, "return")
    assert u[0] == 0.0 and u[1] == 1.0 and u[1] == u[3]
    assert group_u(np.full(5, 2.0), "return") is None


def test_unknown_outcome_mode_is_refused():
    with pytest.raises(ValueError):
        group_u(np.zeros(4), "median")


# ==========================================================================
# (d) the grouping: the fleet's own goal-distance bin, no reservoir needed
# ==========================================================================
class _Field:
    """Distance field stub: d is the x coordinate."""

    _valid_max = 1e9

    def sample(self, pos):
        return np.asarray(pos, np.float32)[:, 0]


def test_depth_bins_come_off_the_race_field_not_the_reservoir():
    from surfgym.mapfleet import MapFleet

    n = 6
    sv = np.zeros(n, dtype=[("origin", np.float32, 3)])
    sv["origin"][:, 0] = [0.0, 99.0, 100.0, 550.0, 999.0, 1e9]
    slot = types.SimpleNamespace(
        sl=slice(0, n), lo=0, reward_field=_Field(), goal_field=None,
        rf_d0=1000.0, respawn=None,
        core=types.SimpleNamespace(states_view=sv))
    fleet = types.SimpleNamespace(slots=[slot])
    out = np.full(n, -7, np.int64)
    MapFleet.stash_depth_bins(fleet, np.ones(n, bool), out, 10)
    assert out.tolist() == [0, 0, 1, 5, 9, -1]
    # only the ENDED rows are written
    out2 = np.full(n, -7, np.int64)
    ended = np.zeros(n, bool)
    ended[2] = True
    MapFleet.stash_depth_bins(fleet, ended, out2, 10)
    assert out2.tolist() == [-7, -7, 1, -7, -7, -7]
    # a slot with no field leaves the ungrouped marker
    slot.reward_field = None
    out3 = np.full(n, -7, np.int64)
    MapFleet.stash_depth_bins(fleet, np.ones(n, bool), out3, 10)
    assert out3.tolist() == [-1] * n


def test_ungrouped_episodes_form_one_group():
    ret = np.arange(6, dtype=float)
    w, st = tail_weights(ret, np.full(6, -1, np.int64), min_n=4)
    assert st["groups"] == 1 and abs(w.mean() - 1.0) < 1e-12


# ==========================================================================
# (e) source-level contracts
# ==========================================================================
def test_record_ckpt_knows_the_flag_is_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for k in ("tail_weight", "tail_outcome", "tail_min_n", "tail_bins"):
        assert f'"{k}"' in src, k
    import record_ckpt
    assert {"tail_weight", "tail_outcome", "tail_min_n",
            "tail_bins"} <= record_ckpt.TRAIN_ONLY


# ==========================================================================
# the CPU scratch runs
# ==========================================================================
# --stall-secs 3 / --ep-ticks 600 so episodes actually END inside a run this
# short: with no ended episodes the weight matrix is all ones and the "on"
# arm would be vacuously identical to the control.
SMOKE_FLAGS = ["--map", str(CANNONBALL), "--reward", "race", "--envs", "64",
               "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
               "--lidar-cell", "32", "--lidar-range", "11500",
               "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
               "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
               "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95",
               "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
               "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
               "--ep-ticks", "600", "--time-pen", "0.005",
               "--success-bonus", "50", "--finish-k", "0",
               "--stall-secs", "3",
               "--race-dist", "geodesic", "--maxvel", "4000",
               "--train-stride", "1", "--yaw-adaptive",
               "--respawn-frac", "0.9", "--respawn-margin", "1",
               "--respawn-reservoir", "1000", "--int-coef", "0.25",
               "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
               "--record-every", "1e12", "--eval-eps", "1",
               "--eval-greedy-only", "--no-eval-at-start", "--seed", "7"]
TAIL_COLS = ["tail/w_max", "tail/w_p90", "tail/groups", "tail/n_med",
             "tail/ess", "tail/cov", "tail/p50", "tail/p75", "tail/p90"]


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _train(run, extra, script="train_fast.py", steps="30720",
           flags=None, timeout=3600):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(ROOT / "python" / script),
           "--run", run] + (SMOKE_FLAGS if flags is None else flags) \
        + ["--steps", steps] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=timeout,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-6000:]
    assert "baking" not in r.stdout.lower(), \
        "a cache MISSED - the maps are not restamped (CLAUDE.md)"
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


# ==========================================================================
# (f) no flag == the unpatched trainer, bit for bit
# ==========================================================================
# Config-dump keys added by OTHER opt-in features merged after the reference
# commit, with the value each takes when its flag is off. --tail-weight adds
# NOTHING when off, which is the claim, so this stays empty unless something
# else lands in between.
INERT_SINCE: dict[str, object] = {}


def _preflag_candidates():
    """Newest first-parent ancestors of HEAD, newest first: the reference for
    'no flag == pre-flag code' is the closest commit that lacks the flag,
    never an old integration branch (whose trainer does not even accept
    today's flags)."""
    try:
        r = subprocess.run(
            ["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        refs = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        refs = []
    return tuple(refs[1:]) or ("HEAD^",)


def _unpatched_trainer(dst: Path):
    for ref in _preflag_candidates():
        # BYTES, never text: this console is cp1251 and train_fast.py is
        # UTF-8 with em dashes - a locale round trip would corrupt the copy
        r = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if r.returncode == 0 and b"--tail-weight" not in r.stdout:
            dst.write_bytes(r.stdout)
            return ref
    return None


@needs_core
def test_no_flag_is_bit_identical_to_the_unpatched_trainer():
    base_py = ROOT / "python" / "_train_fast_pretail.py"
    ref = _unpatched_trainer(base_py)
    if ref is None:
        base_py.unlink(missing_ok=True)
        pytest.skip("no pre-flag train_fast.py reachable from git")
    try:
        _train("tw_bit_base", [], script=base_py.name)
        _train("tw_bit_new", [])
    finally:
        base_py.unlink(missing_ok=True)

    a, b = ROOT / "runs" / "tw_bit_base", ROOT / "runs" / "tw_bit_new"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    added = set(cb) - set(ca)
    assert not set(ca) - set(cb), set(ca) - set(cb)
    assert added <= set(INERT_SINCE), added
    for k in added:
        assert cb[k] == INERT_SINCE[k], (k, cb[k])
    assert {k: v for k, v in cb.items() if k not in added} == ca, \
        [k for k in ca if ca[k] != cb.get(k, "@")]
    # the flag writes NOTHING into the dump when off
    for k in ("tail_weight", "tail_outcome", "tail_min_n", "tail_bins"):
        assert k not in cb

    ta = (a / "progress.csv").read_text(encoding="utf-8").splitlines()
    tb = (b / "progress.csv").read_text(encoding="utf-8").splitlines()
    ha, hb = ta[0].split(","), tb[0].split(",")
    # the tail/* block is APPENDED, so the old header stays a strict prefix
    # and a resumed run's progress.csv migrates instead of breaking
    assert hb[:len(ha)] == ha
    assert hb[len(ha):] == TAIL_COLS
    assert len(ta) == len(tb) >= 2
    for ra, rb in zip(ta[1:], tb[1:]):
        fa, fb = ra.split(","), rb.split(",")
        fa[3] = fb[3] = ""          # time/fps is wall-clock
        assert fb[:len(fa)] == fa
        assert fb[len(fa):] == [""] * len(TAIL_COLS)

    sa = torch.load(a / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert sa["global_step"] == sb["global_step"]
    assert set(sb["config"]) - set(sa["config"]) <= set(INERT_SINCE)
    assert not set(sa["config"]) - set(sb["config"])
    assert {k: v for k, v in sb["config"].items()
            if k in sa["config"]} == sa["config"]
    assert set(sa["policy"]) == set(sb["policy"])
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    oa, ob = sa["optimizer"]["state"], sb["optimizer"]["state"]
    assert set(oa) == set(ob) and oa
    for k in oa:
        for f in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[k][f], ob[k][f]), (k, f)
    for r in ("tw_bit_base", "tw_bit_new"):
        shutil.rmtree(ROOT / "runs" / r, ignore_errors=True)


# ==========================================================================
# (g) the flag on: it runs, it logs, and it changes the gradient
# ==========================================================================
@needs_core
def test_flag_on_runs_logs_its_diagnostics_and_moves_the_weights():
    _train("tw_ctl", ["--tail-weight", "0"])
    _train("tw_on", ["--tail-weight", "1", "--tail-min-n", "2",
                     "--tail-bins", "8"])

    ctl, on = _csv("tw_ctl"), _csv("tw_on")
    assert all(r["tail/ess"] == "" for r in ctl), "control logged tail/*"
    weighted = [r for r in on if r["tail/groups"] not in ("", "0")]
    assert weighted, "no rollout ended an episode - the arm proved nothing"
    for r in weighted:
        n = float(r["tail/n_med"])
        assert n >= 2.0
        assert 1.0 <= float(r["tail/w_max"]) <= n + 1e-6
        assert 0.0 < float(r["tail/ess"]) <= 1.0
        assert 0.0 < float(r["tail/cov"]) <= 1.0
        assert 0.0 <= float(r["tail/p50"]) <= 1.0
        # the weights are doing something: a batch this concentrated cannot
        # come out of an untreated run, where every weight would be 1
        assert float(r["tail/w_max"]) > 1.0

    cc = json.loads((ROOT / "runs" / "tw_ctl" / "run.json")
                    .read_text(encoding="utf-8"))["config"]
    co = json.loads((ROOT / "runs" / "tw_on" / "run.json")
                    .read_text(encoding="utf-8"))["config"]
    assert "tail_weight" not in cc         # --tail-weight 0 is still OFF
    assert co["tail_weight"] == 1.0 and co["tail_outcome"] == "return"
    assert co["tail_min_n"] == 2 and co["tail_bins"] == 8

    sc = torch.load(ROOT / "runs" / "tw_ctl" / "ckpt_final.pt",
                    map_location="cpu", weights_only=False)
    so = torch.load(ROOT / "runs" / "tw_on" / "ckpt_final.pt",
                    map_location="cpu", weights_only=False)
    assert sc["global_step"] == so["global_step"]
    diff = [k for k in sc["policy"]
            if not torch.equal(sc["policy"][k], so["policy"][k])]
    assert diff, "the reweighting changed no policy tensor at all"
    for r in ("tw_ctl", "tw_on"):
        shutil.rmtree(ROOT / "runs" / r, ignore_errors=True)


# ==========================================================================
# (h) a warm resume of the round-30 finisher, with the flag on
# ==========================================================================
@needs_core
@pytest.mark.skipif(not FINISHER.exists(),
                    reason="needs the xENT131 finisher checkpoint")
def test_warm_resume_of_the_finisher_takes_the_flag():
    """Dry-run scale: envs and n_steps are shrunk, the ARCHITECTURE is not -
    emb/hidden/lidar come out of the checkpoint and the arch guard refuses
    anything else. Everything the run needs (tick 7.63, act_every 4,
    obs_reward, the race latch) is restored from the checkpoint's config."""
    run = "tw_resume"
    # --steps is the ABSOLUTE budget, and this checkpoint is at 1.08e10, so
    # a small number here would exit before the first iteration
    gs = int(torch.load(FINISHER, map_location="cpu",
                        weights_only=False)["global_step"])
    flags = ["--map", str(CANNONBALL), "--reward", "race", "--envs", "8",
             "--n-steps", "4", "--minibatches", "1", "--epochs", "1",
             "--ckpt", str(FINISHER), "--ckpt-every", "1e12",
             "--record-every", "1e12", "--eval-eps", "1",
             "--eval-greedy-only", "--no-eval-at-start", "--seed", "3"]
    r = _train(run, ["--tail-weight", "1", "--tail-min-n", "2",
                     "--tail-bins", "16"],
               steps=str(gs + 640), flags=flags, timeout=3600)
    assert "TailRL advantage reweighting" in r.stdout
    ck = json.loads((ROOT / "runs" / run / "run.json")
                    .read_text(encoding="utf-8"))["config"]
    # the checkpoint's own physics and objective came back untouched
    assert ck["tick_ms"] == 7.63 and ck["act_every"] == 4
    assert ck["emb"] == 512 and ck["hidden"] == 448
    assert ck["tail_weight"] == 1.0 and ck["tail_bins"] == 16
    rows = _csv(run)
    assert rows, "the resumed run wrote no progress row"
    assert set(TAIL_COLS) <= set(rows[0])
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
