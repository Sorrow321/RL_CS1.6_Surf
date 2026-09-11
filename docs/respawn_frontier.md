# `--respawn-frontier`: a forward curriculum on the goal potential

Branch `petrusnight` (2026-09-11). Default OFF; with the flag off the
trainer is bit-identical to the trainer before it (proved below, not
asserted). Everything here was measured locally on the 5090, nothing was
rented.

## The ask

The user's words:

> randomize the reservoir more by allowing to spawn the agent in places with
> higher potential compared to where it does. For example, if max potential
> so far is 100, allow it to respawn in points with potential up to 20% more
> than 100. The exact position is randomized. Speed taken from reservoir
> speeds, with up to 5.0 faster speed. Or rather let's do the following:
> when we get stuck, we start slowly increasing (linearly with time) the
> potential where we can respawn.

So: the spawn frontier sits slightly BEYOND what the policy has actually
reached, and when progress stalls the frontier creeps forward linearly with
time. Both halves are implemented; the plateau-driven one is the arm.

## What it is, in one paragraph

`progress = d0 - d` on the shaping field, `d0` = the geodesic distance at
the map's own start spawn (35,636.66 u on petrus). A share
(`--respawn-frontier-frac`, default 0.5) of the RESERVOIR half of the spawn
pool is replaced every iteration by random reachable points whose progress
is at most

    p_cap = (1 + margin + grow) * P_max        floored, and clamped at d0

with the position randomised inside that band, the heading taken from the
goal field's own local descent, and the speed taken from the reservoir's own
observed distribution scaled by `U(0.9, --respawn-frontier-speed)`. The
reservoir stays ON: it keeps harvesting, it keeps reporting min-depth, and
it is where the speeds come from.

## The five design decisions, and why

### 1. `P_max` is START-ANCHORED

The obvious reading of the ask - "max potential so far" over the training
fleet - is a **geometric runaway**. An episode spawned at `1.2 * P_max`
reports `1.2 * P_max` the instant it spawns, so the cap multiplies itself
every iteration and covers the whole map within a couple of dozen of them.
That is exactly CLAUDE.md's harvest trap (round 19 xPSSR: win rate
0 -> 18.46 % while the greedy frontier sat flat with 0/45 finishes, because
the reservoir had collapsed to 1,485 u from the goal).

So `P_max` is the deepest progress reached by an episode that **spawned at
the true map start**, over the last `--respawn-frontier-window` env steps
(default 2e7). No spawn of this class is a start spawn, so no spawn of this
class can inflate it. `RaceReward` tracks it (`frontier_d0 > 0`, logging
only) by recording, at every episode END, the `d` the episode spawned at and
the smallest `d` it reached; `pop_stats` then reports

* `front_pmax` - over the start-anchored episodes (drives the cap),
* `front_pmax_all` - over every episode (the curriculum's own reach),
* `front_spawn_med` / `front_spawn_p90` - the REALISED spawn progress.

The ordering matters and is pinned by a test: on the tick an episode ends,
`self._best` has already folded in the NEXT episode's spawn `d`, so the
tracker keeps its own running minimum and emits the pair BEFORE updating it.
Get that wrong and every episode looks like it reached wherever it
respawned - the runaway again, by a different route.

The reservoir's min-depth was the other candidate and is the wrong one: it
is a lagging, harvest-shaped statistic, and CLAUDE.md's whole point about
round 19 is that reading a curriculum off the harvest measures the harvest.

### 2. Where inside the cap: a flattened band plus a shell

A uniform draw over voxels puts most spawns near the start - the band always
contains the whole run-up, and that is where the voxels are. Two
corrections, both on by default:

* every draw is **bin-flattened in `d`** (`bins` equal-width bins over the
  admitted band, a bin uniformly among the non-empty ones, then a member
  uniformly), so coverage is uniform in PROGRESS rather than in voxel count;
* `--respawn-frontier-shell` (default 0.5) of the states come from the
  **frontier shell**, the deepest `--respawn-frontier-shell-width`
  (default 0.25) of the band.

Shell and body are drawn from disjoint `d` ranges and accepted in their own
target counts, so the flag is the REALISED share. (The first implementation
over-delivered - 69-73 % on a requested 50 % - because the survivors were
taken as a shell-first prefix; `test_shell_share_is_the_flag` pins it.)

Measured on petrus at cap 512 / 2,400 / 9,600 / 24,000 u, the realised
spawn progress has median = 0.75 x cap and p90 = 0.95 x cap, by
construction.

### 3. Velocity: the field's descent, not a random heading

Round 31's `--respawn-random` was a strong negative (the policy never left
the start platform) and a uniformly random heading on an AIRBORNE state is a
large part of why: such a state is unrecoverable by construction.

Here the direction is `-grad d` at the spawn point - central differences on
the trilinear field with sentinel neighbours dropped, one-sided where only
one side is valid, and the point REJECTED where the gradient is degenerate -
perturbed by `heading_sigma` (15 deg) in yaw and elevation and clamped to
an elevation band of [-60, +30] deg. The view is aimed along that same
direction with its own 10 deg noise, because a surfer that is not looking
where it is going cannot steer (`src/env.c` and `raster.py`:
`forward = (cos p cos y, cos p sin y, sin p)`, so pitch POSITIVE is up).

Measured on petrus: **99.7-100 %** of sampled states have the field distance
DECREASING 64 u along their own velocity. The median angle between velocity
and the field's descent is 1.7 deg at `heading_sigma = 0`, 13.7 deg at 15,
39.8 deg at 45 - a uniform heading would average 90.

The speed MAGNITUDE is drawn from the reservoir's own stored horizontal
speeds and scaled by `U(0.9, --respawn-frontier-speed)` - the same thing
`--respawn-speed` does to a reservoir row - then clamped to `--maxvel`.
The clamp is on the MAGNITUDE on purpose: `sv_maxvelocity` is a PER-AXIS
clamp inside `PM_CheckVelocity` (`src/pm.c`), and a per-axis clamp would
bend the heading the design just went to the trouble of choosing.

### 4. Clearance

The `RandomSpawnSampler` rule, reused: the JITTERED point (not the voxel
centre) must carry a finite trilinear potential AND the standing player hull
must fit (`core.trace(p, p, hull=0).startsolid`). Measured petrus rejection
rates: **6.0-11.1 % hull**, a few % cap (the jitter can push a point past
the band edge), and the voxel band itself accepts 0.25-2.7 % of uniform
draws depending on the cap (the whole field is only 3.2 % finite).

### 5. The map-start share is untouched

The frontier share is taken out of the RESERVOIR rows of an
already-built pool, never out of the map-start rows, so the preset's own
start share (`1 - --respawn-frac` = 10 % on `scratch_ablate`) is exactly
what the control got. Evals are untouched by construction: they run on a
separate eval core with the platform start pool, which no spawn source here
ever writes to.

## The plateau half

`--respawn-frontier-grow` reuses **`--unstuck`'s own schedule verbatim**
(`UnstuckSchedule`): the extra margin is 0 while `P_max` keeps improving by
more than `--respawn-frontier-eps` (500 u); once
`--respawn-frontier-patience` env steps pass with no improvement it rises by
`grow` per `--respawn-frontier-period` (1e8) up to `--respawn-frontier-max`;
an improvement halves it per period, so a run that starts moving again cools
off and one that stalls again heats up from wherever it cooled to. The
schedule is checkpointed and restored, like `--unstuck`'s.

It is driven by the START-ANCHORED `P_max`, not by the realised spawn depth,
which is the whole point: the curriculum is allowed to outrun real
capability only on a wall-clock rate limit, never on its own output.

## The trap guard

CLAUDE.md: `race/win_rate` is the third deceptive metric and it has fired on
this exact map. A curriculum that pushes spawns forward is precisely how it
fires. So:

* the step line prints `win`, the reservoir size and min-depth, and
  `front cap / Pmax / spawn med / spawn p90` **on one line, always**;
* `progress.csv` carries `front/pmax`, `front/cap`, `front/grow`,
  `front/pmax_all`, `front/spawn_med`, `front/spawn_p90`, all over the SAME
  window (an early version logged `front/pmax` windowed and
  `front/pmax_all` instantaneous, which made the anchored frontier read as
  deeper than the unanchored one - arithmetically impossible, and exactly
  the kind of thing that gets believed);
* the arm is judged ONLY on greedy eval corridor MAX from the true start
  spawn and on finishes (`tools/eval_honesty.py --order-only 16`).

## Flags

| flag | default | meaning |
|---|---|---|
| `--respawn-frontier` | off | the mechanism |
| `--respawn-frontier-margin` | 0.2 | the user's 20 % past `P_max` |
| `--respawn-frontier-frac` | 0.5 | share of the NON-start pool |
| `--respawn-frontier-shell` | 0.5 | share drawn from the frontier shell |
| `--respawn-frontier-shell-width` | 0.25 | the shell's thickness |
| `--respawn-frontier-speed` | 5.0 | top of the speed multiplier (bottom 0.9) |
| `--respawn-frontier-floor` | 512 u | minimum cap, so step 0 is not degenerate |
| `--respawn-frontier-grow` | 0.0 | extra margin per period while plateaued |
| `--respawn-frontier-patience` | 3e7 | steps without a `P_max` improvement |
| `--respawn-frontier-eps` | 500 u | what counts as an improvement |
| `--respawn-frontier-max` | 3.0 | ceiling on the extra margin |
| `--respawn-frontier-period` | 1e8 | the period the rate is per |
| `--respawn-frontier-window` | 2e7 | steps `P_max` is the max over |
| `--respawn-frontier-anchor` | off | the reservoir cannot outrun the start (below) |
| `--respawn-frontier-quantile` | 100 | P_max = this percentile of the start-anchored reaches, not the max (below) |
| `--respawn-frontier-uniform` | off | uniform over progress inside the cap once the map is finished (below) |

Refused combinations, each with its own message: no reservoir
(`--respawn-frac 0`), `--respawn-random`, `--demo-file`, `--goals`, and
multi-map (`progress` is `d0 - d` and `d0` differs per map, so one cap
cannot describe a fleet).

## `--respawn-frontier-anchor`: the reservoir cannot outrun the start (2026-09-11)

The user's ask, verbatim: *"we should remake reservoir so that we only put
there points that begin from START. So if we get spawn somewhere far away,
and then advance, that doesn't move our whole reservoir forward. It should
be softly limited by our progress from the beginning of the map (e.g. our
best died at x=10.0 and we're stuck there. Then we have a chance to respawn
at x=12.0. But if we start from x=12.0 and die at x=20.0, that doesn't make
reservoir points have starts around 20.0)."*

Why it was needed: `pnFRONT1` (frontier + `--respawn-margin 1`) held 37% of
its reservoir PAST the petrus bend while its start-anchored frontier never
got there, and `pnFRONT3B` reached the goal region and a 5-18% win rate at
286M through the same loop one step removed - a forward spawn's own
snapshots are harvested (at margin 1 everything but the fall is), become
spawns, fly on, are harvested deeper. The start-anchored `P_max` guarded the
CAP against that loop; nothing guarded the RESERVOIR.

Two rules, one flag:

1. **The harvest is capped at the frontier cap.** Every iteration, before
   the drained snapshots are pushed into the ring, rows with
   `progress = d0 - d > p_cap` are dropped (`FrontierSpawnSampler.
   harvest_mask`; `p_cap = (1 + margin + grow) x P_max` is the cap the same
   rollout spawned against). The "softly" is the margin and the plateau
   growth: with the best start at 10 the reservoir may hold states up to
   12, never 20. Dropped share on the step line (`anch drop`) and in
   `progress.csv` as `front/harvest_drop`.
2. **A start is a start FROM REST.** `P_max` counts an episode only if it
   spawned within 256 u of `d0` AND at <= 100 u/s
   (`RaceReward(frontier_anchor_speed=100)`). A frontier row that landed
   near the start is launched at reservoir speed x U(0.9, 5) - up to
   4,500 u/s - and can fly further than any real start; under the old
   distance-only rule it was the frontier.

Needs `--respawn-frontier`; refused under DDP (the cap is per rank).
Recorded in `run.json`, restored on resume, `TRAIN_ONLY` in
`record_ckpt.py`. Flag off: no row is masked, the tracker keeps the
distance-only rule, and the only change is one float per ended episode in
the tracker's pairs.

## After the first finish: `--respawn-frontier-quantile` and `--respawn-frontier-uniform` (2026-09-11)

The user, on `pnANCH` (the first petrus finisher from the true start, 9/9
at 32-34 s from 504M): *"once we reach the end, we start infinitely growing
reservoir. This is unwanted behavior. [...] shouldn't do it after 1st reach
as well. Technically we can start sampling across the whole path randomly
rather than at the end, because we already passed the map."*

What the columns showed: the first finish at 358M put `P_max` at `d0` by
555M, the cap opened the whole map, reservoir min-depth went 27.9% (302M)
to 0.0% (503M) and the win rate 0 to 42% with spawn p90 at 90% of the map.
The anchor filters nothing once `cap = d0`. Two things drove the pile-up:
the SHELL puts half the frontier draws in the last quarter of the map, and
a finished episode harvested its whole chain into the reservoir.

Two rules, no phase switch:

1. **`--respawn-frontier-quantile Q`** (default 100 = the max): `P_max` is
   the Q-th percentile of the per-episode start-anchored reaches over the
   window (`RaceReward.pop_frontier_reaches`), so one finish - or one lucky
   deep episode - does not move the cap; the cap follows what start
   episodes USUALLY reach, and reaches `d0` only when the map is finished
   from the start at a rate of `100 - Q` percent.
2. **`--respawn-frontier-uniform`**: inside the cap, spawn uniformly over
   progress. Reservoir draws are flattened over progress bins
   (`--respawn-binned 1` is implied unless set), the frontier shell turns
   OFF once the cap reaches `d0` (`FrontierSpawnSampler.shell_on`), and a
   finished episode is harvested with the same pre-end margin as a death
   (`RespawnBuffer(success_margin=True)`). Before the first finish nothing
   moves; after it every section of the path gets an equal share.

Both need `--respawn-frontier`; both recorded, restored and `TRAIN_ONLY`.
Flag off: the window max, the shell at every cap, the whole-chain finish
harvest - the sampler of every run before this section.

## Joint runs: one frontier per map (2026-09-11)

`--maps a,b` is no longer refused. Each slot carries its own sampler, cap,
`P_max` window, plateau clock and harvest mask; the `front/*` columns are
suffixed `.<tag>` exactly like the per-map eval columns, the step line
prints one `front[tag]` block per map, and the plateau state is
checkpointed per tag. The fleet's pooled `pop_stats` cannot carry per-map
`front_*` (the last slot's overwrote the rest), so each reward keeps its
own last block (`RaceReward._fr_last`) and the trainer reads that.

## Flag OFF is bit-identical

The trainer is NOT run-to-run reproducible on this 5090 - two runs of the
SAME code at 2,048 envs agree on iteration 1 and diverge from iteration 2 in
the GPU backward - so a whole-run CSV diff cannot carry this claim. Two
things do:

1.  **A CPU-deterministic probe.** The real petrus map, 256 envs, the real
    geodesic field, `RaceReward` with the arm's own constants, 3,000 ticks
    of a fixed pseudo-random action sequence, SHA-256 over every reward
    float plus the final origin / velocity / yaw block:

        HEAD dc641d1                     a4e35b358cd421cf...4cea783fd8ed53
        branch, flag off                 a4e35b358cd421cf...4cea783fd8ed53
        branch, frontier_d0 ARMED        a4e35b358cd421cf...4cea783fd8ed53

    The third line is the stronger one: the tracker is logging only - it
    moves no reward, takes no RNG and changes no mask even when it is ON.

2.  **The trainer's own first iteration.** `bitA` (branch, flag off),
    `bitB` (branch, flag off, repeat) and `bitC` (HEAD) all report
    `rew 1.2356 len 417.9` at step 1,048,576; from step 2 the branch-vs-HEAD
    divergence is no larger than the branch-vs-itself divergence.

Structurally: with the flag off `slot.frontier is None` (so `mix` is never
called and the pool expression is the value it always was),
`frontier_d0 = 0.0` (so `_fr_best is None` and the tracker's branch is never
taken), no `UnstuckSchedule` is built, and `CSV_COLS` gains nothing.

    python -m pytest tests/python/test_respawn_frontier.py -q
