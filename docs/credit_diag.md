# Does a discovered success reach the actions that caused it?

`tools/credit_diag.py` - a bounded, training-free diagnostic. No arm was
run for this and nothing here is a training verdict.

## The question

The discount horizon is settled and is not what this asks about. `gamma`
is 0.9995 per PHYSICS TICK, the trainer raises it to `act_every`, and 2,000
ticks = **20.0 s** (CLAUDE.md section 5: treat the horizon question as
answered and do not shorten it).

What that does not settle is how efficiently a newly discovered success
trains the actions taken 5-10 s earlier. PPO does not propagate a reward
with `gamma` alone; it propagates it with GAE, and the **direct** weight of
a TD residual `k` decisions later is `(gamma^act_every * lambda)^k`. On the
scratch config - decisions every 0.04 s (`--act-every 4` at 10 ms), GAE
`lambda` 0.95, rollouts of `--n-steps 128` decisions = 5.12 s - that is

| k decisions | seconds | `(gamma^4 * lambda)^k` | `gamma^4^k` | ratio |
|---|---|---|---|---|
| 25 | 1.00 | 0.264 | 0.951 | 4x |
| 50 | 2.00 | 0.0696 | 0.905 | 13x |
| 125 | 5.00 | 0.00128 | 0.779 | 609x |
| 250 | 10.00 | 1.64e-06 | 0.607 | 370,765x |

(reproduced by `credit_diag.py` from the checkpoint's own constants and
pinned in `tests/python/test_credit_diag.py`).

So the 0.78 the discount keeps at 5 s does not arrive as a direct trace.
It arrives **through the critic** - the `gamma^k * V(s_{t+k})` term - and is
therefore only as good as `V` is at states the policy has barely visited.
That is the thing to measure, and it is measurable without training:

1. **Is the critic right at pre-wall states?** `V(s_t)` against the
   empirical discounted return `G_t` of complete terminal continuations.
2. **Do successful continuations exist under sampling at all?**
3. **If they do, does their FIRST action get credit?** The advantage that
   decision receives under
   * **(a)** the trainer's own GAE - this run's `lambda` and `n_steps`, cut
     at rollout-buffer boundaries exactly as `train_fast.py` cuts them;
   * **(b)** `lambda = 1` over the whole episode (the Monte-Carlo
     advantage `G_t - V_t`) - what "longer traces" would buy;
   * **(c)** this run's `lambda` over the whole episode, no truncation -
     which separates "`lambda` is too short" from "the ROLLOUT is too
     short".

If successes exist and (b) credits their first action while (a) does not,
the fix is longer traces (`lambda -> 1` with rollouts covering the
manoeuvre), not a longer horizon.

## Method

Everything is read off the checkpoint's own config - `gamma`, `gae`,
`n_steps`, `act_every`, and every reward flag - so the numbers are this
run's, not a reconstruction.

* **Start states.** The checkpoint's OWN greedy episode from the eval spawn
  is rolled to termination with every tick's FULL state kept (a trajectory
  row is lossy: `ducked` / `induck` / `duck_time` / `fuser2` /
  `oldbuttons` / `basevelocity` are not in it and a wrong hull can start an
  episode inside geometry - `tools/traj_to_spine.py`'s opening argument).
  The states at 0 / 1 / 2 / 5 / 10 s before that episode's terminal event
  are the start states, snapped down to a decision boundary. On a stuck
  checkpoint that episode IS a wall-stopping one, so these are pre-wall
  states by construction. `--from-spine` substitutes a recorded episode.
* **Continuations.** From each start state, K envs run to termination on
  the real core: one greedy batch (identical by construction) and one
  sampled batch, optionally at a widened temperature (`--temps`, the
  `diversity_bench` sigma knob).
* **Reward.** `surfgym.rewards.RaceReward`, built constant for constant
  from the config the way `train_fast.py:7328` builds it - the same
  `scale = 100/d0`, `time_pen`, `success_bonus`, `max_step`, `d_floor`,
  `int_coef` with the checkpoint's own novelty table restored. Training's
  stagnation kill is mirrored (`--no-stall-kill` turns it off): evals do
  not stall-kill, but the trainer does, and this diagnostic is about what
  TRAINING would see.
* **`--obs-reward`.** Slot 12 is fed `tanh(r_acc/0.1)` from the reward the
  decision actually earned, exactly as `train_fast.py:10442` writes it in
  the rollout - not `record_ckpt.py`'s recomputed eval mirror.
* **Per decision** (one decision = `act_every` physics ticks) the tool
  records `V(s_t)` off the value head of the row the policy acted on, the
  reward summed over the decision's ticks (the trainer's `r_acc`), and
  whether the episode ended on it.
* **`G_t`** is the exact discounted sum of the rest of that episode at
  `g = gamma^act_every`. A continuation cut by the 120 s episode cap is
  missing the trainer's bootstrap `g*V(s_T)`, so it is counted and then
  EXCLUDED from every statistic rather than quietly biasing the returns.
* **Variant (a)** walks the trainer's recursion
  `lastgae = delta + g*lambda*nonterm*lastgae` with `lastgae` reset to 0 at
  every rollout-buffer edge, over every phase the episode could have
  started at (the phase in the trainer is whatever the rollout iteration
  happened to be at), and reports the mean plus the envelope.
  `tests/python/test_credit_diag.py` checks this decision by decision
  against a literal transcription of `train_fast.py`'s GAE loop, including
  an episode that spans two buffers.
* **Self-check.** A greedy restart from the state `h` seconds before
  the probe's terminal event must reproduce the probe's tail, i.e.
  last about `h` seconds. The tool prints a WARNING when it does not -
  that would mean the state capture lost something the physics needed,
  and every number in the row would be about a different trajectory.
* **Success** is reported two ways, because on a stuck checkpoint the
  absolute one is routinely empty and an empty set answers nothing:
  corridor arc past the wall (205,440 u) or a finish - the frontier
  definition CLAUDE.md insists on - and, alongside it, arc more than one
  route vertex (128 u) past what the GREEDY continuation from the same
  state reached.

### What this cannot say

* `V` was fit on returns under the TRAINING distribution (the sampled
  policy plus the respawn reservoir), so the honest comparison is the
  SAMPLED row; the greedy row is a different policy's return and its bias
  is not a critic error.
* One checkpoint, one probe episode, one seed per row. This measures the
  credit path's arithmetic on real states, not a treatment effect.
* Corridor arc from a mid-route start is absolute (`ArcProgress` with an
  order-only window 16), so it is directly comparable to
  `tools/eval_honesty.py` numbers, but a continuation that starts at
  205,3xx u and ends there has NOT "reached 88%" of anything it earned.

## Results (2026-09-08, local 5090)

Two checkpoints, `surf_src_cannonball`, K = 64 continuations per start
state, `--temps 0,1`. Both runs take under a minute.
Artifacts: `runs/research/creditdiag/{cyPOTLC,sOBSR2}/`.

|  | cyPOTLC | sOBSR2 |
|---|---|---|
| what | scratch, view-continuous + `--view-absolute velocity`, `--obs-potential logabs --obs-potential-curtain` | the STUCK checkpoint, discrete bins, `--obs-reward` |
| step | 3,991,928,832 | 3,782,737,920 |
| `act_every` / decision | 4 / 40 ms | 3 / 30 ms |
| `lambda` / `n_steps` | 0.95 / 128 dec = **5.12 s** | 0.95 / 128 dec = **3.84 s** |
| probe (greedy from the eval spawn) | 72.98 s, corridor **205,208 u** (88.57%), ends z = -4,118 | 73.37 s, corridor **205,262 u** (88.60%), ends z = -4,248 |

Both probes are textbook wall-stops: they track to 88.6% and fall.

**The restart is faithful.** A greedy continuation from the state h seconds
before the probe's terminal event lasts h + one decision and reaches the
probe's own arc to within 1-55 u at every horizon (cyPOTLC h = 10 s:
10.04 s, 205,208 u against the probe's 205,208 u). No WARNING fired.

### 1. The critic

`V(s0)` against the empirical return `G(s0)` of the greedy continuation
(the deterministic row, so its spread is 0 by construction):

| h | cyPOTLC V / G / V-G | sOBSR2 V / G / V-G |
|---|---|---|
| 0 s | -0.074 / -0.002 / **-0.072** | +0.945 / -0.008 / **+0.952** |
| 1 s | +0.402 / +0.459 / -0.058 | +0.214 / +0.247 / -0.033 |
| 2 s | +1.165 / +1.312 / -0.147 | +0.797 / +0.794 / +0.004 |
| 5 s | +3.802 / +3.974 / -0.171 | +2.268 / +2.990 / **-0.722** |
| 10 s | +8.466 / +8.859 / -0.393 | +6.315 / +7.316 / **-1.001** |

The critic is not the broken part. Over 1-10 s both are calibrated to
within 4-13% of `G` and biased the SAFE way (pessimistic). The one gross
error is sOBSR2 at h = 0: **0.03 s before the fall becomes fatal the critic
still values the state at +0.95** while the realised return is -0.008. It
does not see the death at all. cyPOTLC, which reads the potential field as
a lidar channel, does not have that failure (-0.072).

Under sampling it is the SPREAD that grows, not the bias: cyPOTLC at
h = 2 s has `V-G` p10/p50/p90 = -2.34 / -0.20 / -0.10.

### 2. Do successful continuations exist?

Only near the wall, and only marginally. "past wall" = corridor arc >
205,440 u; "beat" = more than one route vertex (128 u) past the greedy
continuation from the same state. **No continuation finished, anywhere.**

| h | cyPOTLC sampled (T=1) | sOBSR2 sampled (T=1) |
|---|---|---|
| 1 s | 0 (0) past wall, 0 (0) beat | 0 (0), 0 (0) beat |
| 2 s | **2 (11)** past wall, 6 (29) beat; best 205,480 (205,690) | **27 (51)** past wall, 40 (55) beat; best 205,568 (205,696) |
| 5 s | 1 (8) past wall, 5 (24) beat; best 205,479 (205,696) | **0 (0)** - every draw ends 1,212 u SHORT of greedy |
| 10 s | 0 (0) past wall, 15 (0) beat; best 205,440 | **0 (0)** - 1,209 u short |

Two things follow. The wins are **real but tiny** - 230 to 490 u past a
205,2xx line, against a 231,680 u finish; this is the lip of the wall, not
the descent. And on the stuck checkpoint, from 5 s and 10 s out the
behaviour policy's OWN noise destroys the flight before it arrives
(episodes 2.52 s instead of the greedy 5.04 s). At those horizons there is
nothing for any credit rule to credit.

### 3. Does the first action get credit?

PPO standardises advantages inside every minibatch
(`train_fast.py:9153`), so a uniform scale difference between variants is
divided out. The decision-relevant number is the **beat-set d'**: (mean
advantage of the continuations that beat greedy minus the mean of those
that did not) divided by the batch's own advantage sd, at decision 0.

| ckpt | h | mode | beat | ep len | d' (a) trainer | d' (b) lambda=1 | d' (c) 0.95, no cut |
|---|---|---|---|---|---|---|---|
| cyPOTLC | 2 s | sampled | 6/64 | 55 dec | **+2.78** | +2.80 | +2.78 |
| cyPOTLC | 2 s | T=1 | 29/64 | 62 dec | +1.27 | +1.09 | +1.25 |
| cyPOTLC | **5 s** | **sampled** | **5/64** | **128 dec** | **-0.50** | **+2.18** | **-0.33** |
| cyPOTLC | 5 s | T=1 | 24/64 | 133 dec | -0.53 | +0.13 | -0.50 |
| cyPOTLC | 10 s | sampled | 15/64 | 244 dec | +0.13 | +0.18 | +0.17 |
| sOBSR2 | 2 s | sampled | 40/64 | 80 dec | **+1.06** | +0.22 | +1.05 |
| sOBSR2 | 2 s | T=1 | 55/64 | 82 dec | -0.39 | +0.00 | -0.32 |

**The credit rule inverts the sign at exactly one place, and it is the
place the reviewer named.** cyPOTLC, 5 s before the fall, under the
trainer's own behaviour policy: 5 of 64 draws beat the greedy line, their
continuations are 128 decisions long - exactly one rollout buffer - and the
trainer's GAE rates their first action **half a standard deviation WORSE
than the failures'**, while the Monte-Carlo advantage rates it **+2.18 sd
BETTER**. Everywhere else (a) and (b) agree.

**It is lambda, not `--n-steps`.** Variant (c) - this lambda with no
truncation at all - is -0.33, so 2.51 of the 2.68 d' gap between (a) and
(b) is the lambda decay and only 0.17 is the rollout cut.
`(gamma^4 * 0.95)^128 = 0.0011`: the payoff is already invisible before the
buffer edge is reached.

### 4. Which lambda

Variant (a) re-scored at other lambdas, at this run's `n_steps` and the
same phase average (`--lam-sweep`; pure arithmetic on the same rollouts):

| ckpt | h | mode | beat | 0.95 | 0.97 | **0.99** | 0.995 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| cyPOTLC | 2 s | sampled | 6/64 | 2.78 | 2.78 | 2.78 | 2.79 | 2.79 |
| cyPOTLC | **5 s** | **sampled** | 5/64 | **-0.50** | **-0.23** | **+1.32** | +1.66 | +1.89 |
| cyPOTLC | 5 s | T=1 | 24/64 | -0.53 | -0.45 | -0.27 | -0.21 | -0.14 |
| cyPOTLC | 10 s | sampled | 15/64 | 0.13 | 0.20 | 0.22 | 0.22 | 0.21 |
| sOBSR2 | 2 s | sampled | 40/64 | 1.06 | 1.02 | 0.71 | 0.59 | 0.47 |
| sOBSR2 | 2 s | T=1 | 55/64 | -0.39 | -0.18 | 0.11 | 0.14 | 0.15 |

The sign flips between **0.97 and 0.99**, and raising lambda costs nothing
where the credit already works (2 s: 2.78 -> 2.79). The one row a higher
lambda makes worse is sOBSR2's 2 s window (1.06 -> 0.47), where the payoff
is 0.8 s away and the Monte-Carlo tail is pure variance.

### Verdict

**Successful continuations DO exist from pre-wall states under the
trainer's own sampled policy, and at 5 s their initiating actions do not
get credit - they get anti-credit.** On the scratch checkpoint, 5 s before
the fall, 5 of 64 sampled continuations beat the greedy line, and the
trainer's GAE at `lambda = 0.95` scores their first action 0.50 sd BELOW
the failures' while `lambda = 1` scores it 2.18 sd above; the
no-truncation control attributes 94% of that to the lambda decay rather
than to `--n-steps`. The critic is not the culprit - calibrated to 4-13%
of `G` over 1-10 s and biased pessimistic - except on the stuck checkpoint
at the fall itself, where it values a doomed state at +0.95 against a
realised -0.008. But the win is narrow: at 10 s no variant separates
anything (d' 0.13-0.18); on the stuck checkpoint at 5-10 s sampling is
strictly destructive, so there is nothing to credit; and every "success"
here is 230-490 u past the lip of the wall, never a finish.

**Recommendation, one line: worth one hour - run a `--gae 0.99` arm on the
scratch config; it is the only change measured here that flips the sign of
the learning signal for the successes that actually exist, and it costs
nothing where the credit already works.**

Caveats: one seed, one probe episode per checkpoint, and the 5 s beat set
is 5 episodes of 64. This says the GAE credit path inverts at about 5 s; it
does not say an arm will move `race/eval_progress`, and per the RETRACTION
in CLAUDE.md a 1-hour scratch arm cannot be ranked at one seed anyway -
report which gate it clears and at what step.
