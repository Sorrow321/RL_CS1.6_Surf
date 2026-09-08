# `--race-ratchet` - pay only for new progress records

## What it does

The stock race shaping is the signed potential difference

```
r_t = scale * (d_{t-1} - d_t) - time_pen
```

with `d` the geodesic distance-to-finish. The ratchet replaces the shaping
term with the improvement of a **per-episode record** `b`:

```
b_0     = d at the episode's own start (spawn AND every reservoir respawn)
b_{t+1} = min(b_t, d_{t+1})
r_t     = scale * (b_t - b_{t+1}) - time_pen        (>= 0 before time_pen)
```

`scale` is unchanged (`100/d0 * --race-shaping`). The time penalty, the
success bonus, the fail penalty, the intrinsic term and every liveness rule
(the stall detector, the respawn `stagnant` mask) are untouched and all of
them keep the **raw** `d`.

Worked example - `d` going `10 -> 8 -> 12 -> 8 -> 6`:

| step | d | b | ratchet | stock |
|---|---|---|---|---|
| 1 | 8 | 8 | **+2** | +2 |
| 2 | 12 | 8 | **0** | -4 |
| 3 | 8 | 8 | **0** | +4 |
| 4 | 6 | 6 | **+2** | +2 |

Both collect `scale * (d_start - d_best)` over an episode. The ratchet is
the signed term with the refunds and the re-charges deleted, not a different
budget: leaving is free, and re-gaining ground you already banked is not
paid a second time.

## Why

Round 19 measured the two halves of the cannonball trap separately:

* `--race-dfloor` (xCLAMP) flattens the potential *inside* `d <= L` and
  still charges the climb back out: **0/99 finishes**;
* `--race-latch` (xLATCH) makes leaving free once the episode has been
  inside: **52/102 finishes**.

So a flat potential is not enough - *leaving must be free of shaping
charge*. The latch buys that with a threshold `L` picked off a champion
trace (6,996 u, the last tick the map pushed back). The ratchet is the same
property with **no threshold, no reference line and no map-specific
constant**: every detour anywhere on the map costs zero, not only the one
below `L`.

On cannonball specifically: route vertices 1600 -> 1680 raise `d`
6,632 -> 14,976 on the champion's own line, charged `-4.02` by the stock
term against a `+50` success bonus the stuck policy has never observed.
Turning back at vertex 1601 is locally optimal and the reward says so. Under
the ratchet that segment costs exactly 0.

## The observation column

The record is **episode history, not state**. Without it in the row, the
critic cannot tell "10 units out on the way in" (the next tick pays) from
"10 units out on the way back" (it pays nothing), and the value function is
unlearnable - the same argument `--race-latch` makes for its flag.

So the network is fed one extra column,

```
gap_t = (d_t - b_t) / d0
```

exactly `0` at a new record and positive by how far the episode has backed
off it, normalised by that map's own start geodesic. With `d` already in the
row, `gap` determines `b = d - gap*d0`, hence what the next transition pays.

Where it lives: the **last** column of the trailing scalar-side block,
`[fan | latch | act-hist/compass | T | ratchet]`. That is the only growth
direction `widen_for_obs`' zero-pad is function-identical for, so a warm
resume of a checkpoint that has never seen the column computes the
checkpoint's own function at step 0 (to ~1 ulp of fp32). A core scalar slot
would NOT do: `Policy.feat_idx` is sorted, so a new scalar lands in the
middle of the row and the zero-pad silently permutes every existing feature.

`RaceReward.ratchet_boot()` is the record one reward call ago. The
truncation bootstrap rebuilds `s_T` from outside the reward call, after the
autoreset has already restarted the live record at the next episode's spawn,
so the terminal row's column is
`(d_T - min(ratchet_boot, d_T)) / d0` (`MapFleet.terminal_ratchet`).
Evaluation rollouts get their own mirror (`_make_eval_ratchet_feed`), which
detects episode starts off the core's per-env tick counter the way
`_make_eval_latch_feed` does; under `--obs-reward` the slot-12 mirror
differences the **record**, not the raw distance, or an eval would feed a
ratchet-trained policy the signal it was never trained on.

## Rules

* `--race-ratchet` is a store-true flag, **off by default**, and off is the
  control path byte for byte (no array, no branch the control did not take;
  `tests/python/test_race_ratchet.py` pins this over a backtracking path).
* Mutually exclusive with `--race-latch` / `--race-latch-frac`,
  `--race-dfloor`, `--race-arc` and `--race-ng` - each is a different
  treatment of the same defect and composing two measures neither. The
  reward object refuses with a `ValueError`.
* Also refused with a per-env goal potential (`--goals` + a euclid field):
  untested, and the column's normaliser is a map-scale quantity.
* A **reset never generates progress reward.** The record restarts at the
  new spawn, so a reservoir respawn deep in the map neither pays for the
  jump nor leaves the fresh episode unable to earn anything - the same
  contract `--race-latch` has for arming on the spawn tick.
* Recorded in `run.json` / the checkpoint config as `race_ratchet`, and
  **restored on a resume**: dropping it would also drop an observation
  column, so the widened checkpoint would not even load.
* Multi-map safe: each slot normalises by its own `rf_d0`.

## Reading an arm

The verdict is **finishes**, from
`python tools/eval_honesty.py --route maps/surf_src_cannonball.route.npz
runs/research/<ARM>/traj_*.jsonl`. `race/eval_progress` cannot see past the
wall at all (it saturates at 191,812 u on any route-following episode), and
it has already been anti-correlated with the truth once. Controls on the
stuck checkpoint: 0 finishes. xLATCH 52/102, xARC 63/102.

`tools/reward_replay.py` replays recorded trajectories through both reward
computations and prints, per episode, the total shaping paid, the charge
over the ramp detour and what a revisit pays - the pre-flight check that the
detour has actually lost its charge.

## Switching the reward on a warm checkpoint: `--critic-warmup N`

Round 28 (ledger addenda 5-7, xsFANARC) measured what happens when the
reward changes under a trained policy: the learned behaviour is destroyed.
The mechanism is not mysterious - `V(s)` is suddenly the value function of a
DIFFERENT objective, so PPO's advantages are noise, and the first updates
point that noise straight at a policy that already works. It reproduces in a
64-env local smoke of this very arm: the stuck checkpoint's first eval
covers 53.9% of the route and the second, ~100k steps later, covers 1.5%.

`--critic-warmup N` is the remedy. For the first `N` PPO updates after a
resume the rollouts are collected with the resumed policy and **only the
value loss is optimised**:

* actor and critic share the conv trunk, so freezing the actor freezes the
  trunk too. The trainable set is `vf.*`, `value_head.*` and (under
  `--priv-critic`) `priv_mlp.*`; everything else is held;
* the hold is exact. After `loss.backward()` and **before** the clip, every
  frozen parameter's `.grad` is set to `None`, and `torch.optim.Adam` skips
  a parameter with no gradient entirely - no step, no moment, no weight
  decay. So the actor is **bit-identical** when the warmup ends, not merely
  close. Zeroing the gradients instead would NOT do that: Adam's momentum
  keeps moving a parameter for many steps after its gradient hits zero;
* `project_log_std()` is skipped while warming (`log_std` is an actor
  parameter and frozen means frozen, projection included);
* the loss is `args.vf * vl`, the same weighting it has afterwards, so the
  critic's effective step size does not jump when the actor is let go.

`N` is counted in **updates** (rollout buffers), so it covers
`N * n_steps * envs * act_every` environment steps. On the stuck checkpoint
(`T=128`, `envs=2048`, `act_every=3` = 786,432 steps per update)
**`--critic-warmup 96` is 75,497,472 steps**, ~9% of a one-hour budget.

Refused from scratch (nothing to protect, and the critic has nothing to
fit) and under DDP (`sync_grads()` all-reduces `p.grad` for every
parameter and the frozen half has none). It prints
`warmup k/N warmup/value_loss ...` per update and a line when it ends;
the value loss coming DOWN is the whole diagnostic, and if it does not the
critic cannot fit the new reward on these features and the arm is already
answered. `tests/python/test_critic_warmup.py`.

The ratchet also adds an input column the resumed checkpoint has no weight
for. That is `widen_for_obs`' trailing zero-pad, the same path `--race-latch`
took in round 19: measured on the stuck checkpoint, 6 tensors widened
(`pi.0`/`vf.0` and their Adam moments gain one ZERO column at scalar row 15)
and the resumed policy computes the checkpoint's own function at step 0.
