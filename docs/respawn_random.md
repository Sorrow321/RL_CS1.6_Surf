# `--respawn-random` - uniform reachable starts

The user's brief, verbatim:

> change spawns. 5% - spawn from start, play normally, same as evals. 95% -
> spawn in random part of the map, random view direction, random speed
> (from, say, 1k to 4k). Potential spawn positions - all reachable, where we
> have some potential.

This is a spawn **SOURCE**, and it **replaces the respawn reservoir**. It is
not a curriculum and it does not look at anything the policy has done: the
distribution is a property of the MAP, fixed before the first gradient step,
and identical at step 0 and at step 10^10.

## What a start looks like

At every episode start (autoreset included - there is only one reset path):

| | |
|---|---|
| **5%** | the map's own start spawn - byte-identical to what an eval uses |
| **95%** | a random reachable state, drawn fresh every training iteration |

The random state:

* **position** - a uniformly random voxel of the goal field that holds a
  finite potential, jittered uniformly inside its own 32 u cell.
  "Reachable, where we have some potential" IS the goal field: the geodesic
  BFS marks every free voxel it can reach with a distance and everything
  else with a sentinel, so `grid < _valid_max` is exactly the set asked
  for. **Airspace is included**, deliberately - the BFS runs over free
  voxels in 3-D and most of what it reaches is open air, which is where a
  surfer spends the map. On `surf_src_cannonball` that set is **10.11%** of
  the 671,156,372 voxels of the 32 u field (67,887,227 voxels).
* **view** - yaw uniform in [-180, 180), pitch uniform in [-30, 15] deg.
* **velocity** - horizontal speed uniform in [1000, 4000] u/s, heading
  `yaw + N(0, 30 deg)` (moving roughly where it is looking, as a surfer
  does), vertical velocity 0, `onground = -1` (airborne).

Two rejections, and only two:

1. the **standing player hull** must fit at the jittered point
   (`core.trace(p, p, hull=0).startsolid` is False). A 32 u cell whose
   centre is free space can still clip a wall, and the hull is 32x32x72.
   Measured on cannonball: this rejects **1.9-2.3%** of otherwise-valid
   points.
2. the trilinear potential at that exact point must still be finite
   (`field.reachable(p)`) - the voxel test is on the centre, the spawn is
   not.

Nothing else is excluded. In particular a state inside the finish box is
not filtered out; it is ~10^-5 of the field's volume, and with no reservoir
there is nothing for a trivial win to reinforce.

## Why it is routed through the spawn pool

`RandomSpawnSampler.build_pool` returns a `STATE_DTYPE` pool and the
trainer hands it to `core.set_spawn_pool`. The C reset (`reset_env`,
`src/env.c`) zeroes the whole state struct, copies one uniformly drawn pool
row, re-zeroes `tick` / `stuck_ticks`, re-projects `progress` /
`best_progress`, and adds the env's own `yaw_jitter_deg`. **That is exactly
the mechanism a reservoir respawn uses**, so "entry counts are the
probabilities" holds (which is what makes the 5/95 split a pool
composition, not a branch), and every counter the reward and liveness logic
keys on an episode start is reset the same way it already was for a
reservoir respawn:

* `RaceReward`: `_best`, `_since` (the stall timer), `_ticks`, per-env
  `_d0`, the `d_latch` flag, the arc anchors (`arc.reset(..., mask=ended)`,
  a global search precisely because "a respawn relocates arbitrarily"), the
  novelty `_prev_cell` mask;
* the trainer: the depth-history ring (`push(ended=...)`), the episode
  hygiene counters, the truncation bootstrap.

None of that needed a change, and none of it was changed. The one thing
that DOES change is what the trainer can say about the starts (below).

The pool is rebuilt **every training iteration** (4,096 entries: 205 start
rows + 3,891 fresh random states). Cost on cannonball, measured: **17 ms per pool**
against a ~3.5 s iteration (0.5%). The sampler is two-stage on purpose -
the voxel draw is pure numpy and over-draws 12x, and only as many survivors
as are still needed reach the per-point trilinear sample and the 0.7 us
hull trace. A one-stage version cost 220 ms.

## What the core clamps

* `sv_maxvelocity` is a **per-axis** clamp inside `PM_CheckVelocity`
  (`src/pm.c:115`), applied during the move and not at reset. At the arm's
  `--maxvel 4000` a 4,000 u/s horizontal speed is never clamped: its
  largest component is at most 4,000, and only when the heading is
  axis-aligned. At the stock 2,000 it WOULD be, so this flag belongs with
  `--maxvel 4000`.
* Pitch is not clamped at reset; [-30, 15] is inside the engine's own
  [-70, 30].
* Yaw is wrapped to [0, 360), not clamped, after the env's
  `yaw_jitter_deg` is added.

## Diagnostics: the reservoir metrics are meaningless here

There is no reservoir, so **`reservoir min-depth`, the reservoir size and
the respawn "stagnant" mask describe nothing** under this flag (min-depth
reports NaN, the fleet's `observe_respawn` is a no-op, and no snapshots are
harvested). CLAUDE.md's rule that a rising win rate next to a falling
min-depth is measuring the harvest does not apply, because there is no
harvest.

What is logged instead, at the same cadence, is where the starts actually
landed on the shaping potential:

```
randspawn d: min 0  p10 6,250  median 52,908  p90 158,673  max 198,017
            (3,891 states, voxel accept 10.1%, hull reject 2.30%)
```

`RandomSpawnSampler.d_stats()` returns the same numbers as a dict.

**Read that line before reading `race/win_rate`.** The distribution is
uniform over VOLUME, not over arc: cannonball's goal-adjacent airspace is
open and large, so **16.8%** of starts land within 10,000 u of the finish
and **0.05%** land inside it (d = 0). A training win rate of a few percent
is therefore the SAMPLER, not the policy - the local 64-env smoke posted
4-9% within two million steps from scratch. The evals are start-spawn and
greedy, so the verdict (corridor MAX, finishes) is unaffected; the training
win rate simply is not a signal under this flag. Percentiles of the
sampled potential on cannonball, for reference (d0 = 198,380):

| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|---|---|---|
| 792 | 3,131 | 6,083 | 14,580 | 53,121 | 107,347 | 156,760 | 195,444 |

## Flags

| flag | default | |
|---|---|---|
| `--respawn-random` | off | turn it on; turns the reservoir off entirely |
| `--respawn-random-start-frac` | 0.05 | share of episodes at the map spawn |
| `--respawn-random-speed MIN MAX` | 1000 4000 | horizontal spawn speed, u/s |

All three are written to `run.json` and restored from a checkpoint. It
needs the geodesic goal field (`--reward race --race-dist geodesic`) and
refuses to start without one. **Evals are untouched** - they run off
`slot.plat_pool`, the map start spawn, exactly as before, which is why the
arm's verdict (corridor MAX, finishes) is still a statement about starting
at the start.

`--respawn-frac` / `--respawn-margin` / `--respawn-reservoir` are ignored
when this is on, and the launcher says so on the first line:

```
respawn RANDOM: 5% of episodes at the map start spawn (what the evals use),
the rest at uniform random reachable states; speed 1000-4000 u/s, yaw
U[-180,180), pitch U[-30,15]; the reservoir is OFF (--respawn-frac 0.9 ignored)
```

## Flag-off identity, and why it is argued rather than measured

The whole diff is **additive**. `git diff --numstat` is 87/2 on
`train_fast.py`, 188/1 on `respawn.py`, 5/1 on `mapfleet.py`, and every
"deletion" is a line that gained a name: the import, `__all__`,
`MapSlot.__slots__`, and `if args.respawn_frac > 0.0:` becoming the `elif`
of a new `if args.respawn_random:` that is False by default. Nothing that
runs with the flag off was changed.

That argument is load-bearing because **the full trainer is not
reproducible run-to-run on this hardware**, so CLAUDE.md's "bit-identity
check against the baseline code path" cannot be run end to end here. Two
runs of the SAME code, same `--seed 0`, same 40,960 steps on the local
5090, diverge as much as the old code does from the new one (first greedy
eval `fwd 5u / path 862u` vs `fwd 146u / path 1733u`; `kl 0.0279` vs
`0.0271`). torch.compile at `max-autotune`, bf16 and CUDA-graph capture are
between the seed and the arithmetic. **Do not read a differing short run as
evidence of an effect** - run the determinism control first, as this arm
did.

What IS pinned deterministically is the code path: with the flag off no
slot has a sampler, the pool-refresh `elif` chain reaches the reservoir
branch by the condition it always had, and both are asserted on the
trainer's source in `tests/python/test_respawn_random.py`.

## Tests

`tests/python/test_respawn_random.py` (17), against the real map and a real
slice of the baked cannonball field:

1. the mix over **10,240 real C resets** is 5%, and the pool's entry counts
   are the probabilities;
2. every sampled position has a finite potential on the field it was drawn
   from AND fits the standing hull; the hull rejection is non-zero, so the
   check is not decorative;
3. speed / yaw / pitch / vertical-velocity ranges, the heading noise (mean
   ~0, sd ~30 deg around the view yaw), and that no other state column is
   carried in;
4. a sampled state round-trips through `core.set_state` (origin, velocity,
   yaw, pitch, onground) and through a real `core.reset`;
5. **flag off is the old trainer**: the default is False, a fresh slot has
   no sampler, the reservoir `elif` is reached by the condition it always
   had, and the flag is in `run.json` and restored.
