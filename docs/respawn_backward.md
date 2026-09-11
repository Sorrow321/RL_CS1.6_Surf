# `--respawn-backward`: the map run BACKWARD on its own field

## The ask

The user, 2026-09-11, verbatim: *"instead of spawning in start of the map
and slowly progress forward, we go backward. What I mean by that: in the
beginning of training, we spawn agents basically at the end of the map, with
high speed. Then, as training moves, we put the spawn locations backward more
and more, so that at some point we spawn at start of the map. There are some
caveats, like how to make reservoir in this case."*

## What it is

Salimans & Chen's backward curriculum (1812.03381, the mechanism behind the
only run that has finished a map from a curriculum here - cySPINEW's
`--demo-grow`, round 34) with the map's own geodesic field standing in for
the demonstration, so it is champion-free. `BackwardSpawnSampler` subclasses
`FrontierSpawnSampler` and reuses its positions (voxels of the field with a
finite potential, bin-flattened in d, standing-hull clearance on the
jittered point), its velocity direction (the field's own descent, heading
noise 15 deg, elevation clamped) and its view. What differs:

| | `--respawn-frontier` | `--respawn-backward` |
|---|---|---|
| admitted band (distance to goal) | `[d0 - cap, d0]`, cap grows FORWARD from the start | `[floor, W]`, W grows BACKWARD from the goal |
| the shell (aggressive half, `shell_frac`) | the deepest part, nearest the goal | the FAR part, nearest the start |
| what moves the band | start-anchored P_max (+ plateau growth) | the far shell's FINISH rate |
| spawn speed | reservoir speed x U(0.9, 5) | absolute U(lo, hi), default 1000-2500 u/s |
| needs the reservoir? | yes (speeds) | no; it takes its rows once it holds 2,000 states |

## The five decisions

1. **The band is ANCHORED at the goal and WIDENS**, it does not slide. A
   sliding window drops the mastered end of the map and the policy forgets
   it; the widening band keeps every mastered section in the distribution,
   and at `W = d0` it is exactly "uniform over the whole path" - where the
   forward recipe's `--respawn-frontier-uniform` wanted to end up. The shell
   turns off there for the same reason.
2. **The advance signal is the far shell's finish rate.** Episodes spawned
   at `d >= W - shell_width x (W - floor)` are the hardest the band offers;
   when at least `--respawn-backward-min-ep` (50) of them ended within
   `--respawn-backward-window` (2e7) steps and `>= --respawn-backward-rate`
   (0.2, the paper's) finished, W grows by `--respawn-backward-step x d0`
   (5%) and the shell statistics restart. Map-start spawns (the 10% share
   the evals share) sit outside the band while `W < d0` and do not count.
3. **A linear floor is optional.** `--respawn-backward-steps S` (0 = off)
   makes W at least `w0 + (d0 - w0) x step / S`, so a stalled shell still
   reaches the start by S - the backward analogue of the frontier's plateau
   growth, and the same warning applies: it can outrun capability.
4. **Speed is an absolute prior** because the reservoir is empty when the
   curriculum starts and the user asked for "high speed". Petrus's ramps
   need ~1,550 u/s (CLAUDE.md), so the default U(1000, 2500) straddles it;
   `--respawn-backward-speed LO HI` sets it.
5. **The reservoir, the caveat the user raised.** It keeps harvesting, with
   two rules borrowed from the forward recipe: finishes are harvested with
   the same pre-end margin as deaths (`RespawnBuffer(success_margin=True)`),
   and draws are flattened over progress bins (`--respawn-binned 1` is
   implied). Backward episodes only move TOWARD the goal from their spawn,
   so their snapshots lie inside `[floor, W]`; map-start episodes add their
   own near-start snapshots. Before the reservoir holds 2,000 states the
   pool is `[map-start share | backward rows]`; after that
   `--respawn-backward-frac` (0.5) of the non-start rows are backward and
   the rest are reservoir draws. No anchor is needed: nothing can run past
   the goal.

## Flags

| flag | default | meaning |
|---|---|---|
| `--respawn-backward` | off | the mechanism |
| `--respawn-backward-start` | 0.05 | W at step 0, fraction of d0 |
| `--respawn-backward-step` | 0.05 | growth per advance, fraction of d0 |
| `--respawn-backward-rate` | 0.2 | far-shell finish rate that advances W |
| `--respawn-backward-min-ep` | 50 | shell episodes before the rate counts |
| `--respawn-backward-window` | 2e7 | steps the rate is measured over |
| `--respawn-backward-steps` | 0 | linear floor: W reaches d0 by this step (0 = off) |
| `--respawn-backward-frac` | 0.5 | backward share of the non-start pool once the reservoir has 2,000 states |
| `--respawn-backward-shell` | 0.5 | share of backward rows from the far shell |
| `--respawn-backward-shell-width` | 0.25 | the shell's thickness, fraction of the band |
| `--respawn-backward-speed` | 1000 2500 | spawn speed U(lo, hi), u/s |
| `--respawn-backward-floor` | 256 u | nearest admitted distance to the goal |

Exclusive with `--respawn-frontier`, `--respawn-random`, `--demo-file` and
`--goals`; needs the reservoir and the geodesic field; single-GPU (W is
advanced per rank); one sampler per map on a joint run. Recorded in
`run.json`, restored on resume (`W` and the advance count per map are
checkpointed), `TRAIN_ONLY` in `record_ckpt.py`.

## Read-out

Step line: `back W 12.3% shell 24%/61 adv 2` - W as a fraction of d0, the
far shell's finish rate over its episode count, advances so far. Every 100
iterations a `backward:` line with the realised spawn progress (median /
p90). `progress.csv`: `back/W_frac`, `back/shell_rate`, `back/shell_n`,
`back/n_adv`, `back/spawn_med`, `back/spawn_p90` (suffixed `.<tag>` on a
joint run). The verdict is, as ever, the greedy true-start eval: corridor
MAX, finishes, `race/eval_finish_s`.

## Flag OFF is bit-identical

`BackwardSpawnSampler` is reached only under the flag; the refactor that
made room for it (`_band`, `_shell_ranges`, `_has_speed_source`, `_speeds`
on the frontier sampler) returns the frontier's own values and makes the
same RNG calls in the same order. `tests/python/test_respawn_frontier.py`
(44 tests) stays green.
