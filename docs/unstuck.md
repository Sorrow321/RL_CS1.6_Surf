# `--unstuck`: a plateau-driven exploration temperature, and the benchmark that gates it

Branch `contyaw-abs` (2026-09-06, evening). Default OFF; with the flag off
the trainer is byte-identical to the trainer before it (pinned, see "What
is pinned"). Everything here was measured on the local 5090 with a 2048-env
trainer sharing it (its fps did not move: 520-530k steps/s before, during
and after), nothing was rented.

## The ask

The user's words: "The more we sit at plateau (progress/reward doesn't
increase), the more we increase reward for exploration ... imagine we have
some temperature T. We start with low temperature. Low temp = low
entropy/chaos. Once our progress gets stuck, we start increasing T, e.g.
linearly. We need some mechanism that would make the following thing: the
larger T, the more diverse are trajectories ... when we get stuck, this T
starts rising, and at some point it would be more beneficial for policy to
just explore rather than grab immediate reward by just flying to the
ground. But first we need to make some benchmark to check whether T works.
We need to see if we have a method to diversify rollouts, and some metric
that measures how close they are (are they +- the same line, or
different)."

Two parts, in that order: **A**, a benchmark that dials T on a frozen
checkpoint and measures diversity and its cost; **B**, the trainer
mechanic, built on the same tempered sampling so that "the trainer's
behaviour policy at T" and "the benchmark's rollout at T" are one code
path.

The checkpoint both parts are measured on is the **wall checkpoint**:
`C:/RL_Surf_base/runs/research/cyABSV/ckpt_8002732032.pt` (absolute
velocity-frame view, 8.0B steps, 9/9 greedy episodes at the 88.8 % wall,
learned sigma 0.056 / 2.7 pre-tanh - the pitch head is capped to 0.5 at
load by this branch). Its greedy rollout from the eval's seed-0 spawn:
order-only corridor progress **205,252 u** of 231,680, dies at **68.5 s**.

## Part A - the benchmark: `tools/diversity_bench.py`

    python tools/diversity_bench.py C:/RL_Surf_base/runs/research/cyABSV/ckpt_8002732032.pt \
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp \
        --route C:/RL_Surf/maps/surf_src_cannonball.route.npz \
        --out runs/research/divbench/wall --n 64 --temps 0,0.25,0.5,1,2,4 \
        --knob sigma --knob eps

N rollouts (default 64) of ONE checkpoint from ONE fixed spawn state, at a
list of temperatures T, under a diversification knob applied at SAMPLING
time. The spawn is the eval's: a 1-env core built exactly as
`tools/record_ckpt.py` builds it (race start pool, descent yaw, pitch -10,
the core's 5-degree yaw jitter) is reset with `--seed` (default 0) and env
0's state is taken; the bench core (N envs, jitter 0) then spawns every
env from that one state through a one-entry pool, and the tool checks
that origin / velocity / yaw / pitch agree across all N. `--from-spine
spine.npy --at-tick T` starts from the spine state nearest that tick
instead (a mid-route start). The sampling noise seed (`--sample-seed`) is
the SAME for every T, so the temperature is the only thing that changes
between rows. Map and route are passed as the MAIN checkout's absolute
paths (the worktree bake trap of CLAUDE.md; a cache miss is reported as a
warning).

The knobs, all in `train_fast.TemperedTorchPolicy`, a `SampledTorchPolicy`
that calls the same two draw helpers the trainer's rollout calls:

* **sigma**: the Gaussian view heads' sigma x (1+T) and, on the
  categorical heads (the keys; the bins too on a discrete checkpoint), a
  softmax temperature logits / (1+T) - `sample_padded(padded, temp)` /
  `sample_view(padded, mu, log_std, temp)`. T = 0 is the plain
  `SampledTorchPolicy` byte for byte (temp None takes the shipped ops, no
  extra RNG draw);
* **eps**: per decision each head's sample is replaced by a uniform draw
  with probability p = min(0.5, 0.05 T) (the planner's `--eps` rule: a
  uniform bin on a categorical head, u uniform in (-1, 1) and z = atanh(u)
  on a view head);
* **both**;
* three ATTRIBUTION knobs that temper one component of `sigma` at a time
  through the same call's per-component form (`sample_view(.., temp,
  temp_view)`): **yaw** (the yaw head's sigma only), **pitch** (the pitch
  head's sigma only), **keys** (logits / (1+T) on the categorical heads
  only).

A `--greedy` reference (all N rollouts identical by construction, and
checked) is reported alongside. Metrics per (knob, T), written to
`bench.csv`, `bench.md`, `bench.png`, `spread.json` and one
`rollouts_<tag>.npz` per row:

1. **progress**: order-only corridor progress per rollout
   (`tools/eval_honesty.corridor_progress_ordered`, window 16, corridor
   1500, run batched over the rollouts - pinned equal to the per-rollout
   function) - mean, max, finishes, share past 205,440 u (the wall);
2. **spread vs time**: the rollouts are time-aligned from the one spawn;
   at every second the RMS distance of the ALIVE rollouts from their
   medoid; the curve, its values at 25 / 50 / 75 % of the median episode
   length and at 60 s (the wall entry is ~68 s);
3. **branches**: single-linkage clusters at 512 u of the end positions,
   and of the positions at 30 s and 60 s;
4. **coverage**: distinct 256 u position cells visited by the union of the
   rollouts, and NOVEL cells = cells whose count in the checkpoint's own
   `int_counts` (summed over its 8 view x 3 speed bins; the key is pinned
   against `RaceReward._cells`) is zero - the number the unstuck mechanic
   wants to raise - plus `rare` = under 100 visits;
5. **same-line share**: the fraction of rollouts within 100 u of the
   greedy trajectory for their whole life (spatially, the max over the
   life of the distance to the nearest greedy point; a time-aligned
   variant and a first-60-s variant are in the CSV).

### Results on the wall checkpoint (64 rollouts, spawn seed 0)

Full tables: `docs/unstuck/wall.md` (the design grid, both knobs),
`docs/unstuck/wall_sigma_fine.md`, `docs/unstuck/wall_eps_fine.md`,
`docs/unstuck/wall_attrib.md`; the CSVs next to them; the PNGs
`docs/unstuck/wall.png` (progress vs T, spread vs time, branches vs T,
coverage vs T), `wall_sigma_fine.png`, `wall_eps_fine.png`,
`wall_attrib.png`. Progress in route units (wall = 205,440; route =
231,680); spread in u; "alive" = rollouts still flying at 60 s.

**The `sigma` knob** (sigma x (1+T), logits / (1+T)):

| T | prog mean | prog max | past wall | len med | spread 25/50/75 % | spread 60 s (alive) | branches end/30s/60s | cells | novel | rare | same-line |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 205,252 | 205,252 | 0/64 | 68.5 s | 0/0/0 | 0 (64) | 1/1/1 | 1,112 | 0 | 0 | 100 % |
| 0 | 177,320 | 205,440 | 0/64 | 71.4 s | 389/803/1,254 | 1,309 (52) | 10/1/1 | 3,247 | 0 | 0 | 2 % |
| 0.1 | 165,997 | 205,684 | 1/64 | 72.1 s | 386/911/1,569 | 1,703 (49) | 11/2/6 | 3,480 | 0 | 2 | 8 % |
| 0.2 | 163,645 | 205,382 | 0/64 | 72.8 s | 529/1,235/2,255 | 2,481 (46) | 20/4/6 | 3,434 | 0 | 1 | 0 % |
| 0.25 | 147,054 | 205,454 | 1/64 | 72.7 s | 482/1,022/2,238 | 2,618 (41) | 18/2/7 | 3,483 | 0 | 0 | 5 % |
| 0.3 | 121,266 | 205,616 | 4/64 | 64.6 s | 610/1,384/2,174 | 3,383 (35) | 26/5/13 | 3,657 | 0 | 23 | 3 % |
| 0.4 | 100,644 | 205,696 | 1/64 | 35.6 s | 411/1,334/999 | 3,568 (23) | 38/4/12 | 3,751 | 0 | 4 | 2 % |
| 0.5 | 61,638 | 203,666 | 0/64 | 13.0 s | 241/847/323 | 2,355 (9) | 32/4/7 | 3,049 | 0 | 11 | 3 % |
| 0.75 | 20,446 | 114,347 | 0/64 | 10.5 s | 274/642/682 | - (0) | 19/3/0 | 1,245 | 0 | 7 | 2 % |
| 1 | 11,646 | 49,152 | 0/64 | 6.3 s | 158/361/596 | - (0) | 13/0/0 | 620 | 0 | 0 | 2 % |
| 2 | 4,216 | 15,001 | 0/64 | 4.7 s | 38/204/1,088 | - (0) | 7/0/0 | 169 | 0 | 0 | 0 % |
| 4 | 2,436 | 3,995 | 0/64 | 6.1 s | 65/328/705 | - (0) | 2/0/0 | 90 | 0 | 0 | 0 % |

**The `eps` knob** (p = 0.05 T per head per decision):

| T | p | prog mean | prog max | past wall | len med | spread 60 s (alive) | branches end/30s/60s | cells | novel | rare |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.001 | 126,050 | 205,460 | 1/64 | 67.0 s | 3,420 (34) | 27/3/8 | 3,614 | 0 | 1 |
| 0.05 | 0.0025 | 86,904 | 205,411 | 0/64 | 21.2 s | 4,955 (19) | 34/3/11 | 3,406 | 0 | 6 |
| 0.1 | 0.005 | 46,948 | 198,730 | 0/64 | 11.9 s | 257 (2) | 37/5/1 | 2,662 | 0 | 7 |
| 0.2 | 0.01 | 27,703 | 116,948 | 0/64 | 10.1 s | - (0) | 24/7/0 | 1,631 | 0 | 12 |
| 0.25 | 0.0125 | 19,704 | 102,044 | 0/64 | 9.2 s | - (0) | 20/3/0 | 1,332 | 0 | 3 |
| 0.5 | 0.025 | 9,132 | 50,165 | 0/64 | 6.4 s | - (0) | 15/0/0 | 583 | 0 | 31 |
| 1 | 0.05 | 5,530 | 15,446 | 0/64 | 6.5 s | - (0) | 7/0/0 | 188 | 0 | 3 |
| 4 | 0.2 | 2,530 | 3,758 | 0/64 | 7.7 s | - (0) | 1/0/0 | 84 | 0 | 0 |

**Attribution** - one component of `sigma` at a time:

| knob | T | prog mean | prog max | past wall | len med | spread 60 s (alive) | branches end/30s/60s | cells | novel | rare |
|---|---|---|---|---|---|---|---|---|---|---|
| yaw | 0.25 | 168,559 | 205,440 | 1/64 | 72.6 s | 1,872 (48) | 17/2/5 | 3,458 | 0 | 1 |
| yaw | 0.5 | 105,362 | 205,384 | 0/64 | 39.4 s | 3,102 (25) | 31/2/12 | 3,469 | 0 | 3 |
| yaw | 1 | 24,852 | 67,575 | 0/64 | 11.6 s | - (0) | 20/0/0 | 1,013 | 0 | 0 |
| yaw | 4 | 5,155 | 8,043 | 0/64 | 5.0 s | - (0) | 4/0/0 | 155 | 0 | 4 |
| pitch | 0.25 | 173,732 | 205,427 | 0/64 | 71.4 s | 1,867 (52) | 11/4/5 | 3,214 | 0 | 6 |
| pitch | 1 | 177,108 | 205,481 | 1/64 | 71.5 s | 1,379 (54) | 11/2/3 | 3,257 | 0 | 1 |
| pitch | 4 | 168,161 | 205,586 | 2/64 | 71.5 s | 1,615 (50) | 9/3/5 | 3,237 | 0 | 0 |
| keys | 0.25 | 174,027 | 205,607 | 2/64 | 71.8 s | 1,534 (53) | 10/2/3 | 3,380 | 0 | 1 |
| keys | 0.5 | 165,440 | 205,692 | 3/64 | 72.3 s | 1,909 (48) | 19/2/4 | 3,618 | **2** | 41 |
| keys | 1 | 52,991 | 204,395 | 0/64 | 7.2 s | 2,452 (10) | 23/3/6 | 3,238 | 1 | 8 |
| keys | 2 | 5,371 | 18,099 | 0/64 | 4.7 s | - (0) | 8/0/0 | 245 | 0 | 0 |

### What the tables say

1. **T diversifies, inside a narrow range.** On the `sigma` knob every
   diversity measure rises monotonically from T = 0 to T = 0.4: spread at
   60 s 1,309 -> 1,703 -> 2,481 -> 3,383 -> 3,568 u, end branches 10 ->
   11 -> 20 -> 26 -> 38, branches at 60 s 1 -> 6 -> 6 -> 13 -> 12, distinct
   cells 3,247 -> 3,751. Past T = 0.4 the rollouts die before they can
   diverge (median life 35.6 s at 0.4, 13.0 s at 0.5, 6.3 s at 1) and every
   measure falls with them. **The dial works, and its live range on this
   checkpoint is T in [0, 0.4].**
2. **What it costs.** Mean progress falls 177k -> 164k -> 147k -> 121k ->
   101k u over the same range (the mean is pulled down by the rollouts
   that die early), while the MAX stays at the wall through T = 0.4 - and
   at T = 0.3 four of 64 rollouts pass 205,440 u (the wall) against 0 of
   64 for the plain sampled policy, 0 of 64 greedy. None finishes. The
   crossings are 100-250 u past the wall, i.e. the ramp-1 departure
   delayed, not a new frontier.
3. **Which knob.** `eps` at the design schedule is far too hot: p = 0.0125
   (T = 0.25) already halves the flight to 9 s because one uniform draw
   of a view head at flight speed is a yaw target up to 180 degrees off
   and one uniform key is a wrong strafe. Its live range is p <= 0.0025
   (T <= 0.05), where it spreads MORE per unit of progress lost than
   `sigma` (T = 0.02: spread 3,420 u at 60 s, 27 end branches, mean
   126k; the closest `sigma` point in spread, T = 0.3, costs the same
   progress and keeps 35 alive against 34) - a draw. Between them,
   `sigma` is the one whose diversity grows smoothly with T and whose
   distribution PPO can score (an eps mixture has no simple log-prob), so
   the trainer uses `sigma`.
4. **Which component.** Tempering the **pitch** sigma is a free no-op:
   at T = 4 (sigma 0.5 -> 2.5, the camera target nearly uniform over
   [-70, 30] every decision) progress, life and spread are those of T = 0.
   The policy trained most of its 8B steps with that sigma at the 2.7 cap
   and does not read the camera pitch; it also buys no diversity. The
   **yaw** sigma is the fragile component (x1.5 halves the flight, x2
   kills it in 11.6 s: sigma 0.056 in z is ~0.2 deg of velocity-frame
   offset per decision and the ramp lines tolerate about that). The
   **keys** temperature buys the most per unit of progress lost: at
   logits / 1.5, 48 of 64 alive at 60 s, mean 165k (against 177k
   untempered), 3 of 64 past the wall - and it is the ONLY knob that
   reached cells the count table had never seen (2 at T = 0.5, 1 at T =
   1) and the most rare cells (41). The trainer got `--unstuck-temp-heads
   {all,keys,view,yaw}` from this.
5. **Novelty is worn out everywhere these rollouts can reach.** The
   checkpoint's table has 88,806 visited position cells and 1.36 B
   visits; the union of 64 rollouts at any T touches 3-4k cells and none
   of them is new (novel = 0 in every row but the two `keys` rows). The
   bonus `int_coef / sqrt(N + 1)` along the beaten path is ~0.0002 per
   transition against ~0.023 of shaping per decision, and the only region
   with counts in the hundreds is past the wall. **Tempering alone cannot
   raise the number the mechanic wants raised; that is what the count
   DECAY in Part B is for** (and why it is gated on T > 0).
6. **"Same line" at 100 u is not a discriminator here.** 2-8 % of
   rollouts stay inside 100 u of the greedy trajectory for their whole
   life at EVERY T including 0: sigma 0.056 in z drifts a 70-second
   flight by more than 100 u on its own. The spread curve and the branch
   count carry the information; the same-line share is reported because
   it was asked for, with a first-60-s variant that says the same thing.

## Part B - the mechanic: `train_fast.py --unstuck`

Default OFF and byte-identical when off (pinned on the bins and on the
absolute view). When on:

**The plateau signal.** Once per iteration the schedule reads the run's
progress in map units: the respawn reservoir's deepest reach, `rf_d0 -
min geodesic depth` (the step line's `mind` in units, through
`MapFleet.reservoir_min_depth`, cached on its 25-call cadence), and under
`--race-arc` the deepest order-only arc any episode reached this
iteration (`arc_reach`). A reading that beats its all-time best by more
than `--unstuck-eps` (500 u) is an improvement; `stuck_steps` is the env
steps since the last one. (A run with neither a reservoir nor an arc line
is refused: there would be nothing to watch.)

**The schedule** (`UnstuckSchedule`, pure Python, checkpointed):

    T = 0                                  while improvements keep coming
    stuck_steps > patience  ->  T += rate * ds / period   (cap tmax)
    improvement             ->  T halves every `period` steps (or T = 0 with --unstuck-reset),
                                and the patience window restarts

defaults `patience` 2e8 steps, `rate` 0.5 per `period` = 1e8 steps, `tmax`
4. So a run that stalls sits at T = 0 for 2e8 steps, then warms by 0.5 per
1e8 steps, reaches the cap after 8e8 more, cools by half per 1e8 steps
after a new best, and warms again from wherever it cooled to if the best
does not move for another 2e8. `--unstuck-period` exists so a smoke can run
the whole schedule in miniature.

**What T does** (each its own flag, all on by default):

* (a) `--unstuck-temp 1`: the SAMPLING temperature. The rollout draws
  every categorical head from softmax(logits / (1+T)) and every Gaussian
  view head from N(mu, (sigma (1+T))^2) - `sample_padded(padded, temp_t)`
  / `sample_view(padded, mu, log_std, temp_t, tempv_t)` with two STATIC
  tensors the schedule writes into between iterations (never a Python
  float: that would bake T into the captured graph). The update
  recomputes the log-prob and the entropy through `logprob_entropy_view /
  logprob_entropy_padded` with the same tensors, so `pi_new` and `pi_old`
  are the same tempered measure and the PPO ratio is a true importance
  ratio; the entropy bonus is the tempered distribution's too. The
  gradient on `log_std` is unchanged by the shift (d/d log_std of log_std
  + log(1+T) is 1); the gradient on the logits is scaled by 1/(1+T).
  `--unstuck-temp-heads {all,keys,view,yaw}` (default `all`) says which
  heads T reaches - the attribution table above is the reason it exists.
* (b) `--unstuck-ent 1`: the entropy coefficient x (1+T) (on top of
  `--ent-final`'s schedule, if any).
* (c) `--unstuck-int 1`: the intrinsic coefficient x (1+T)
  (`RaceReward.int_coef` is rewritten per iteration; the config keeps the
  base), and while T > 0 the visit counts are multiplied by
  `--unstuck-count-decay` (0.5) once per `period` (`RaceReward.decay_counts`,
  `>> 1` in place on the int64 table; a table pending from a checkpoint
  is decayed too), so novelty worn out on the beaten path pays again.
* (d) `unstuck/T` (the T THIS iteration's rollout ran at),
  `unstuck/stuck_steps`, `unstuck/best` appended LAST to `progress.csv`
  (only under the flag: the flag-off header is the one that shipped), and
  `T .. stuck ..M best ..u` on the step line, `T a->b` on an iteration
  that changed it.

Config: every knob is written to `run.json` only under the flag and
restored on resume (the flag itself too; an explicit CLI value wins). The
schedule's run state (`T`, the bests, the step they were set at, the decay
accumulator) rides in the checkpoint as `ck["unstuck"]` and is restored,
so a resume continues the plateau clock instead of granting a fresh
patience window. Refused with a message: `--rnn`, `--chunk`, `--yaw-cond`,
`--maps`, `--ez-eps`, `--spawn-burst`, DDP (the tempered draw and its
log-prob live in the flat single-map paths; the DDP count sync keys on a
base table a decay would desynchronise), a non-race reward, and
`--unstuck-temp-heads view|yaw` without `--view-continuous`.

### Flags

| flag | default | meaning |
|---|---|---|
| `--unstuck` | off | the mechanic |
| `--unstuck-eps` | 500 u | an improvement must beat the best by this |
| `--unstuck-patience` | 2e8 steps | stuck this long before T rises |
| `--unstuck-rate` | 0.5 | T per period while stuck |
| `--unstuck-max` | 4 | cap on T |
| `--unstuck-reset` | off | reset T to 0 on a best instead of halving per period |
| `--unstuck-count-decay` | 0.5 | counts x this per period while T > 0 (1 = never) |
| `--unstuck-period` | 1e8 steps | the period every rate is per (a smoke knob) |
| `--unstuck-temp` / `-ent` / `-int` | 1 / 1 / 1 | the three effects, individually |
| `--unstuck-temp-heads` | all | which heads the sampling temperature reaches |

### Validation

1. **Identity with the flag off** (`tests/python/test_unstuck.py`,
   `test_flag_off_is_bit_identical_to_the_trainer_before_unstuck[bins|abs]`):
   the toy scratch set (64 envs, 16x8 depth, 6,144 steps, CPU) on this
   trainer and on the `train_fast.py` of the last first-parent commit
   without `--unstuck`, for the bins and for `--view-continuous
   --view-absolute velocity`: identical config, identical `progress.csv`
   header and rows (fps excluded), identical eval trajectory bytes,
   weights and Adam moments. Both sampling paths were touched
   (`sample_padded` for the bins, `sample_view` for the continuous heads),
   so both are pinned.
2. **CPU smoke** (`test_unstuck_smoke_T_rises_scales_the_coefficients_and_resumes`):
   the same toy set with `--unstuck --unstuck-patience 0 --unstuck-period
   2048 --unstuck-rate 1 --unstuck-max 3`, five iterations: `unstuck/T`
   runs 0, 1, 2, 3, 3 (the T each rollout ran at, rising by one per
   2,048-step iteration once stuck, capped), the step line reads `ent
   0.0200` (0.005 x 4) at the cap, the count decay pulses, the config
   carries every knob, the checkpoint carries the state, and a flagless
   resume restores the flag, its knobs and `T 3.000` and runs its first
   iteration at T = 3.
3. **GPU smoke from the wall checkpoint** (64 envs, `--unstuck
   --unstuck-patience 0 --unstuck-period 1e5 --unstuck-rate 1`, eager
   `--no-compile`, 1.28M steps in 6.5 min at 26.9k steps/s; the full log
   is `docs/unstuck/gpu_smoke_cyABSV_64env.txt`). The step line every
   4 iterations:

   | step (after 8,002.73M) | T | ent coef | sig yaw/pitch | int / ep | kl | mind | best |
   |---|---|---|---|---|---|---|---|
   | +33k | 0.00 | 0.0050 | 0.056/0.500 | 0.57 | 88.9 | 15.86 % | 166,927 u |
   | +98k | 0.33->0.66 | 0.0066 | 0.056/0.500 | 1.12 | 5.8 | 15.86 % | 166,927 |
   | +229k | 1.64->1.97 | 0.0132 | 0.057/0.500 | 3.51 | 0.48 | 15.86 % | 166,927 |
   | +426k | 3.60->3.93 | 0.0230 | 0.059/0.495 | 12.7 | 0.19 | 15.86 % | 166,927 |
   | +459k | 3.93->3.13 | 0.0247 | 0.059/0.496 | 18.6 | 0.15 | 6.86 % | **184,769** |
   | +557k | 3.79->4.00 | 0.0239 | 0.061/0.495 | 17.4 | 0.11 | 6.86 % | 184,769 |
   | +819k | 4.00 | 0.0250 | 0.066/0.499 | 35.9 | 0.10 | 6.86 % | 184,769 |
   | +1,278k | 4.00 | 0.0250 | 0.072/0.500 | 48.3 | 0.16 | 6.86 % | 184,769 |

   T rose 0 -> 3.93 in 0.43M steps as set, the entropy coefficient
   followed (0.005 x (1+T), 0.025 at the cap), the learned yaw sigma
   climbed 0.056 -> 0.072 under the x5 entropy bonus, the intrinsic
   bonus paid per episode rose 0.57 -> 48 (int_coef x5 and 12 halvings
   of the count table - 644,881 non-zero cells after the first, 16,506
   after the twelfth), and the one "improvement" (best 166,927 ->
   184,769 u) halved T over a period (3.93 -> 3.13) before it climbed
   back. Two readings of that log that matter:
   * the first-iteration `kl 88.9` is the resume's, not the flag's: the
     same 64-env resume WITHOUT `--unstuck` reads kl 78.6 / 20.5 / 21.0
     on its first three lines (the update of a 2048-env checkpoint on
     512-row minibatches; the ledger saw kl 9.3 on the transplant's first
     update);
   * the "improvement" was the harvest margin, not progress: `mind` fell
     15.86 % -> 6.86 % because the tempered 64-env rollouts died later
     and differently and states inside the 10 s margin got harvested.
     The reservoir signal is exposed to exactly the artefact CLAUDE.md
     records for win rate; pair it with `--race-arc`'s arc reach where a
     route file exists (the schedule takes both), and read `unstuck/best`
     against the honest tools before believing a cool-down.
4. **The same code path** (`test_trainer_rollout_and_update_share_the_wrappers_tempered_helpers`
   + `test_tempered_wrapper_is_sample_view_at_the_same_temp`): the
   trainer's rollout and update are pinned at the source level to the
   `temp_t / tempv_t` calls of `sample_view / sample_padded /
   logprob_entropy_*`, and the wrapper's decision at temperature T is
   pinned to equal `sample_view(.., temp=T)` under the same seed. Part A's
   `sigma` rows ARE the trainer's behaviour policy at those T.

### What the benchmark says about the mechanic's defaults

The spec's `--unstuck-max 4` tempers far past the range in which the
frozen wall checkpoint survives (median life 6 s at T = 1 on `all`,
13 s at T = 0.5). Under training the policy can adapt to its own noise -
the smoke shows PPO keeps `kl` at 0.1-0.2 and pushes sigma up rather than
collapsing - but the benchmark cannot say whether that adaptation explores
past the wall or just learns to fly noisily, and a fleet of 2048 envs
dying at 6 s harvests nothing. For a first arm the measured live range
argues for **`--unstuck-max 0.5`** with the default heads, or
**`--unstuck-temp-heads keys --unstuck-max 1`** (the keys temperature was
the only component that reached unseen cells and kept 48/64 alive at
logits / 1.5); let the entropy and intrinsic scalings run as specified.
The count decay at the default 1e8-step period is 12 halvings per 1.2B
steps of plateau; the smoke's `int/ep` 0.57 -> 48 shows the term can come
to dominate the reward (shaping's whole budget is 100 per route), so watch
`int/ep` on the step line against `rew` and lower `--unstuck-count-decay`
towards 1 if it does.

## What is pinned (`tests/python/test_unstuck.py`, 20 tests)

(a) temp None is the shipped draw op for op; a temperature scales the
Gaussian residual by exactly temp (float == tensor) and flattens a
categorical head to softmax(logits / temp) (20k draws within 0.02); the
log-prob `sample_view` returns equals `logprob_entropy_view`'s at the same
temp and a hand computation; the per-component form reaches only its
component; (b) `TemperedTorchPolicy` at temp 1 / eps 0 is
`SampledTorchPolicy` byte for byte, at temp 3 its decision equals
`sample_view(.., 3)` under the same seed, `view_scale` / `keys_temp`
leave the other component's draw untouched, eps 1 makes every head
uniform, bad arguments are refused; (c) the schedule's sequence
(rise after patience at the rate, cap, halve after a best, reset mode,
the eps threshold, NaN readings, the decay pulse), state round trip;
(d) `decay_counts` (>> 1, floor, the pending table, DDP and range
refusals); (e) `pos_cells` == `RaceReward._cells` (and the view/speed key
// 24), the batched order-only progress == `corridor_progress_ordered` per
rollout on a bent synthetic route, single linkage / medoid / alive masks
/ spread curve on hand cases; (f) the two identity smokes, the unstuck
smoke with its resume, the `--rnn` refusal, and the source-level pin of
the trainer's calls.

## Not done / open

* No box was rented and no arm was run: this is the benchmark and the
  mechanic, smoked, not a result against the wall.
* The eps knob is eval-only (its mixture has no closed-form log-prob for
  the ratio); the trainer tempers with `sigma` only.
* The reservoir signal's harvest-margin exposure (above). A
  `--respawn-margin 2` run would move the reservoir's reach with every
  later death; `--race-arc`'s reach is the cleaner signal where a route
  exists, and the count decay is the part of the mechanic the benchmark
  could not exercise from the spawn (novel = 0 everywhere it reaches).
* `diversity_bench.py` mirrors the config keys of the absolute-view
  scratch family (lidar, view mode, yaw_adaptive, blend, side-hold,
  maxvel, act_every, tick 10 ms, int_view/int_speed) and REFUSES
  checkpoints that set `route_file`, `act_hist`, `obs_compass`,
  `priv_critic`, `chunk`, `rnn`, `frame_stack`, the masks, `yaw_cond`,
  a fixed pitch, `goals`, `race_latch`, `obs_reward`, `maps`, or another
  tick - record those with `record_ckpt.py` or extend `build_core /
  build_policy` there.
