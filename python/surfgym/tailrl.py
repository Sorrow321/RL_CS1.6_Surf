"""TailRL episode-level advantage reweighting (arXiv 2609.02987).

Ramasubramanian, Arora, Tajwar et al., *Tail-Likelihood Reinforcement
Learning* (CMU / Berkeley, 2026), https://github.com/Zanette-Labs/TailRL .

THE OBJECTIVE.  For a reward normalised to [0, 1] and a "prompt" x,

    p_theta(x, tau) := Pr_{z ~ pi_theta(.|x)} ( r(x, z) > tau )
    J_TailRL(theta; x) := integral_0^1 log p_theta(x, tau) dtau

i.e. the expected log-probability of clearing a UNIFORMLY DRAWN reward
threshold, instead of the expected reward.  Its gradient is

    grad J = integral_0^1 (1 / p_theta(x, tau)) grad p_theta(x, tau) dtau

- the same score-function gradient PPO/REINFORCE take, with each threshold
weighted by 1/p, so thresholds the policy clears RARELY count for more.
Their Theorem 1 is the reason to care here:

    grad J_TailRL = sum_{k >= 1} (1/k) grad Best-of-k(theta; x)

- a harmonically weighted mixture of best-of-k objectives, which is what
"chase the world record" is and what the PPO mean is not.  (Proof sketch,
reproduced numerically in tests/python/test_tail_weight.py: with rewards in
[0, 1], Best-of-k = integral_0^1 (1 - (1-p)^k) dtau, so
sum_k (1/k) grad(...) = integral (sum_k (1-p)^{k-1}) grad p dtau
= integral grad p / p dtau.)

THE ESTIMATOR.  With N rollouts for the same x and sorted outcomes
r_(1) <= ... <= r_(N), the empirical survival function is a step function -
exactly N - i + 1 of the N rollouts exceed any tau in [r_(i-1), r_(i)) - so
the per-rollout weight

    w(r) = integral_0^r dtau / p_hat(tau)

collapses to the paper's recurrence (their README, verbatim):

    gap_i = r_(i) - r_(i-1)   (r_(0) := 0)
    survivors_i = N - i + 1
    A_i = N * sum_{j <= i} gap_j / survivors_j          (then mean-centred)

``weights_from_u`` below is that line, and nothing else.

WHAT THIS TRANSPLANT CHANGES, AND WHY (read before using it).

1. *Mean-ONE, not mean-centred.*  TailRL is critic-free: A_i IS the
   advantage, so mean-centring is what gives failures a negative sign.
   Here w MULTIPLIES a GAE advantage that already carries its own sign, and
   centring would flip the sign of every below-average episode's advantage -
   a different algorithm.  Instead the weights are non-negative and average
   to one, so the objective is reweighted and the effective learning rate is
   not.  A weight of exactly 0 (the group's worst episode, always) zeroes
   that episode's advantage; it never flips its sign.

   Two riders on that, both about PPO's per-minibatch advantage
   STANDARDISATION, which runs after this and which the flag does not
   touch.  (i) A zero advantage is not literally "no vote": the
   standardisation subtracts the minibatch mean, so a zeroed row leaves
   with -mean/std like any other row at zero.  That is the control's
   behaviour for a zero advantage too, not something the reweighting
   introduces, but it is why the honest claim is "does not carry its own
   sign" rather than "is dropped".  (ii) The batch mean-1 normalisation the
   trainer applies is EXACTLY a no-op on the loss, because standardising
   (a - mean)/std is invariant to scaling every a by the same constant.  It
   is kept because it makes ess / w_max readable and because it would
   matter the moment advantage standardisation were turned off.  What does
   the real work is the RATIO of weights, within and across groups - which
   is what the per-group frame in (2) fixes.

2. *The [0, 1] frame is PER GROUP.*  The paper's rewards live in a common
   [0, 1] for every prompt.  Our episode returns do not: an episode that
   respawns 5,000 u from the goal can bank at most a small fraction of the
   shaping an episode from the map start can, so a globally-anchored frame
   would systematically down-weight every near-goal group and turn the
   mechanism into "train on long episodes".  Each group is therefore mapped
   onto [0, 1] by its OWN min and max.  That is not only the honest choice,
   it is the self-consistent one: the recurrence telescopes to
   sum_i A_i = N * (u_max - u_0) = N, so per-group min-max normalisation
   makes EVERY group's weights average to exactly 1 and no group can
   out-shout another by the scale of its returns.

3. *r_(0) := 0 means the group minimum.*  Under (2) the worst episode of a
   group gets weight exactly 0.  That is faithful, not an artefact: in
   TailRL's own binary-reward setting every failed rollout has gap_1 = 0 and
   gets weight 0 too.  ``blend`` (the trainer's --tail-weight) interpolates
   w -> 1 + blend * (w - 1) for anyone who wants that softened; the
   interpolation preserves the mean-1 property exactly.

4. *Groups of one, or of identical outcomes, get weight 1.*  There is no
   tail to estimate from a single sample, and ``min_n`` refuses to estimate
   one from too few (the paper's N is 8-64 per prompt).

The maximum weight a group can produce is N (all mass at the bottom, one
rollout at the top: A_N = N * (1 - 0)/1).  So the variance this adds is
bounded by the group size, and ``ess`` reports what it actually cost.

TWO FIT PROBLEMS THIS SETTING HAS AND THE PAPER'S DOES NOT.  Both are
measured, not guessed, on the round-30 finisher (xENT131: 2,048 envs,
T = 128 decisions, act_every 4, tick 7.667 ms, ep_len_mean 3,497 ticks,
~300 episodes ending per rollout, d0 = 198,380 u at ~2,900 u/s).

*(1) Choosing --tail-bins is a trade between N and a START-STATE confound.*
Episodes in one group start at DIFFERENT states inside the bin, and the
depth difference moves the outcome by itself.  Across a bin of width W:

  --tail-outcome time    the deeper start simply takes longer:
                         W / 2,900 seconds (16 bins -> 4.3 s).
  --tail-outcome return  the potential shaping PAYS for that depth and
                         partly cancels it: +100 W/d0 of shaping against
                         -W/2,900 of time penalty = W/6,289 reward, and one
                         reward is one second here (16 bins -> 2.0 s).

So ``return`` carries a **2.7x smaller** start-state confound than ``time``
on this reward - the dense shaping is doing the normalisation - which is
why it is the default and the recommended setting, even though ``time`` is
the metric the run is judged on.  Its price is the intrinsic bonus
(``--int-coef 0.25``, 0.06-0.90 per episode here), which ``time`` would not
carry.  Group size goes the other way: ~300 ends per rollout over B bins is
~300/B per group, against the paper's N = 8-64.  16 bins ~ 19 per group and
2.0 s of confound; 32 ~ 9 and 1.0 s; 64 ~ 5 and 0.5 s.  ``tail/n_med`` and
``tail/groups`` in progress.csv are the read-out - if n_med falls under
``min_n`` the bins are too fine and the groups silently stop being weighted.

*(2) Only the LAST T decisions of an episode are ever reweighted.*  TailRL
is bandit-like: one scalar outcome for a whole rollout, and the whole
rollout is reweighted by it.  Here an episode is ~874 decisions and the PPO
buffer is 128, so an episode spans ~7 buffers and its outcome only exists
in the last one - the earlier fragments were already updated at weight 1 and
cannot be revisited.  ``tail/cov`` reports the share of the buffer that
carries a real weight; expect ~7% in that configuration
(0.5 * 300 / 2,048).  The mechanism therefore reaches the final ~3.9 s of
each episode, not the whole run, and any claim about behaviour EARLY in an
episode has to survive that.  Raising it means raising ``--n-steps``
(linear, and T is a real variable with its own optimum) or holding whole
episodes in the buffer, which PPO's fixed-T rollout does not do.
"""
from __future__ import annotations

import numpy as np

# --tail-outcome time, MIXED groups only: where a group contains both
# finishers and non-finishers, the two are laid on the shared [0, 1] scale
# as two bands - non-finishers ordered by RETURN (which is the dense
# progress signal, so this is "ranked by how far they got"), finishers
# ordered by NEGATIVE FINISH TIME, every finisher above every non-finisher.
# Thirds, so the band widths and the gap between them are one stated
# constant and carry no units: the alternative (splicing seconds onto
# reward units) has no non-arbitrary scale at all.
_NF_BAND = (0.0, 1.0 / 3.0)
_FIN_BAND = (2.0 / 3.0, 1.0)
_TAUS = (0.5, 0.75, 0.9)


def _band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``x`` min-max'd onto [lo, hi]; a flat block goes to the band TOP.

    The top, not the bottom, so that a group whose best block is flat still
    reaches u = 1 and keeps the mean-1 property of the weights.
    """
    a, b = float(np.min(x)), float(np.max(x))
    if b <= a:
        return np.full(x.shape, hi, np.float64)
    return lo + (hi - lo) * (x - a) / (b - a)


def group_u(ret: np.ndarray, mode: str = "return",
            finished: np.ndarray | None = None,
            secs: np.ndarray | None = None) -> np.ndarray | None:
    """One group's outcomes on TailRL's [0, 1] reward scale, or None when the
    group carries no information to reweight on (every outcome identical).

    ``mode``:
      ``return``  the episode's undiscounted collected return, exactly as
                  ``rollout/ep_rew_mean`` sums it.
      ``time``    negative finish time for finishers, non-finishers ranked
                  below every finisher by their return.  See _NF_BAND.
    """
    ret = np.asarray(ret, np.float64)
    if mode == "return":
        return None if ret.max() <= ret.min() else _band(ret, 0.0, 1.0)
    if mode != "time":
        raise ValueError(f"unknown tail outcome {mode!r}")
    fin = np.asarray(finished, bool)
    sec = np.asarray(secs, np.float64)
    if not fin.any():
        return group_u(ret, "return")
    if fin.all():
        return None if sec.max() <= sec.min() else _band(-sec, 0.0, 1.0)
    u = np.empty(ret.shape, np.float64)
    u[~fin] = _band(ret[~fin], *_NF_BAND)
    u[fin] = _band(-sec[fin], *_FIN_BAND)
    return u


def weights_from_u(u: np.ndarray) -> np.ndarray:
    """The paper's recurrence, verbatim, on outcomes already scaled to [0, 1].

        gap_i = u_(i) - u_(i-1), u_(0) := 0
        survivors_i = N - i + 1
        w_i = N * sum_{j <= i} gap_j / survivors_j

    Ties are handled by the recurrence itself: a zero gap adds nothing, so
    equal outcomes come out with equal weights whatever order they sorted
    in.  ``sum_i w_i == N * max(u)``, so the weights average to 1 whenever
    the group reaches the top of its own scale.
    """
    u = np.asarray(u, np.float64)
    n = u.size
    order = np.argsort(u, kind="stable")
    us = u[order]
    gaps = np.empty(n, np.float64)
    gaps[0] = us[0]                       # u_(0) := 0
    gaps[1:] = np.diff(us)
    surv = np.arange(n, 0, -1, dtype=np.float64)      # N, N-1, ..., 1
    ws = np.cumsum(gaps / surv) * float(n)
    w = np.empty(n, np.float64)
    w[order] = ws
    return w


def tail_weights(ret, groups, *, mode: str = "return", finished=None,
                 secs=None, min_n: int = 4, blend: float = 1.0):
    """TailRL weights for a batch of finished episodes.

    ``ret``       (E,) undiscounted episode returns.
    ``groups``    (E,) group id per episode - the reservoir depth bin the
                  episode SPAWNED in.  Episodes sharing an id are the
                  paper's "N rollouts for the same input x"; -1 is the
                  ungrouped bucket (no binning field, or a spawn the field
                  reads as invalid) and forms one group of its own.
    ``finished``  (E,) bool, ``secs`` (E,) episode wall seconds - only read
                  under ``mode="time"``.
    ``min_n``     groups smaller than this keep weight 1: there is no tail
                  to estimate from a handful of samples.
    ``blend``     w -> 1 + blend * (w - 1); 1.0 is the paper's weight, 0.0 is
                  a no-op.  Mean-preserving.

    Returns ``(w, stats)`` with ``w`` (E,) float64 >= 0 and ``stats`` the
    diagnostics the trainer logs.
    """
    ret = np.asarray(ret, np.float64)
    groups = np.asarray(groups, np.int64)
    n_ep = int(ret.size)
    w = np.ones(n_ep, np.float64)
    stats = {"groups": 0, "n_med": 0.0, "used": 0, "eps": n_ep,
             "p": {t: float("nan") for t in _TAUS}}
    if n_ep == 0:
        return w, stats
    sizes, p_acc = [], {t: [] for t in _TAUS}
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if idx.size < max(2, int(min_n)):
            continue
        u = group_u(ret[idx], mode,
                    None if finished is None else np.asarray(finished)[idx],
                    None if secs is None else np.asarray(secs)[idx])
        if u is None:
            continue                       # no spread: nothing to reweight
        w[idx] = 1.0 + float(blend) * (weights_from_u(u) - 1.0)
        sizes.append(idx.size)
        stats["used"] += idx.size
        for t in _TAUS:
            p_acc[t].append(float(np.mean(u > t)))
    stats["groups"] = len(sizes)
    if sizes:
        stats["n_med"] = float(np.median(sizes))
        for t in _TAUS:
            stats["p"][t] = float(np.mean(p_acc[t]))
    s1, s2 = float(w.sum()), float((w * w).sum())
    # normalised effective sample size of the weights: 1.0 when every
    # episode weighs the same, 1/E when one episode carries the batch
    stats["ess"] = (s1 * s1 / (n_ep * s2)) if s2 > 0.0 else float("nan")
    stats["w_max"] = float(w.max())
    stats["w_p90"] = float(np.percentile(w, 90.0))
    return w, stats
