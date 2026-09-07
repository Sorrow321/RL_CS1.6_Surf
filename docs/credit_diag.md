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

## Results

*(filled in below by the run)*
