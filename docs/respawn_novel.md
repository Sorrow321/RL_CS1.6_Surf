# `--respawn-novel`: own states moved to where the agent has never been

Branch `petrusnight` (2026-10-19). Default OFF; with the flag off the trainer
is bit-identical to the trainer before it (checked: config, `progress.csv`,
eval trajectory, weights and Adam moments of the CPU smoke).

## The ask

The user's words:

> we cannot teleport forward, but we can be respawned in any position where
> we haven't been, but the potential is no bigger than our biggest
> potential. So, for example, [...] in some of our rollouts, we could be in
> the point where our potential equals to 10. That means that we can be
> respawned in any position where the potential is also 10 or less with the
> same velocity that we had in our existing rollout. [...] maybe it would be
> a better idea to kind of make it non-uniform, so like we're spawning
> closer to frontier, [...] closer to place where we died, but just in a
> different place basically.

The motivating case on sidistic is the first ramp. The policy only has to turn
left to take it and does not, so a spawn on top of that ramp shows it what it
never tries.

`--respawn-frontier` is the opposite mechanism: it spawns AHEAD of the
best (`docs/respawn_frontier.md`).

## What a novel row is

A novel row is one of the policy's OWN reservoir states moved to a point it has
never been. It keeps that state's velocity, view, duck state and held keys.
`surfgym.respawn.NovelSpawnSampler` builds the rows as follows.

1. **Source.** Half of the rows (`--respawn-novel-shell` 0.5) come from the
   deepest quarter of the reservoir by field distance
   (`--respawn-novel-shell-width` 0.25), i.e. the states nearest where the
   agent dies. The rest are flattened over progress across the whole
   reservoir, so a missed turn near the start is reachable too.
2. **Move.** The source is moved along a uniform direction on the sphere,
   by a distance drawn from `U(256, 2048)` u (`--respawn-novel-radius`).
   Up to 8 tries are made per source (`--respawn-novel-tries`); a source
   with no survivor is dropped.
3. **Gates.** Every gate below must pass at the moved point:
   * it is on the field's reachable set (finite potential);
   * it is **no deeper than the deepest point a REAL episode has reached**
     (`d >= d_best`, i.e. potential <= the best potential);
   * it is in a cell of the visit grid (`--respawn-novel-cell` 128 u)
     whose 3 x 3 x 3 block **no REAL episode has entered recently**, so it is
     at least one cell clear of everywhere the policy goes on its own. Two
     rules make "recently" mean that:
     * **the window** (`--respawn-novel-window` 1e7 env steps). There are two
       grids, rolled every 10M steps, so "entered" means within the last
       10-20M steps. Without it, the near-random fleet of the first few
       million steps marks the whole start region for good, including the
       turn the converged policy never takes, and the gate passes nothing.
       This happened in the first launch: `sidEM2cap2surfNV` produced 0-1
       rows per pool through 53M with 1,772 cells marked, and was restarted
       as `sidEM2cap2surfNV2`.
     * **only REAL episodes mark cells.** A novel episode's flight marks
       nothing, so a blind spot stays on offer until the policy's own
       episodes go there, instead of being used up by the spawns that show
       it;
   * the standing player hull fits (`core.trace(q, q, hull=0)`, the
     frontier sampler's rule).
4. **Contact state.** The engine re-derives contact on the first tick
   (`onground = -1`). The conveyor and water state of the old position are
   cleared.

`--respawn-novel-frac` (0.5) of the RESERVOIR rows of each iteration's spawn
pool are replaced by novel rows. The map-start share
(`1 - --respawn-frac`) is untouched, so the evals and the start-anchored
statistics see what they always saw.

## Why it cannot teleport the run forward

* **Real episodes.** A REAL episode is one spawned at the map start or from
  the reservoir. Only real episodes lower `d_best` and mark the visit grid,
  once per decision.
* **No harvest.** A novel episode is never harvested. `RespawnBuffer.skip`
  points at the sampler's per-env tags, and a tagged episode's pending
  snapshots are discarded at its end. So its flight can never become anyone's
  spawn.
* **Result.** The reservoir and the ceiling move only through what the policy
  learns, never through where a novel row put it. This closes the harvest
  loop that `--respawn-frontier-anchor` had to be added for.
* **How an episode is tagged.** By its spawn ORIGIN. The C reset copies a
  pool row byte for byte (`src/env.c` `reset_env`, spawn_mode 2), and the
  post-step view of an ended env is that spawn. `note_spawns` therefore looks
  the origin up in the keys of the live and the previous pool. This was
  checked on the real engine:
  `test_the_real_reset_lands_on_the_pool_origin_byte_for_byte`.

## Read-out

* **Step line:** `novel ceil <progress>u rows <n> eps <share>`. These are
  the ceiling, this iteration's novel rows, and the share of the episodes
  that started from one.
* **Every 100 iterations, a detail line:**
  * where the rows landed (progress median / p90), how far they were moved
    and the shell share;
  * the rejection shares (off-field, past the ceiling, visited, hull).
* **`progress.csv`, LAST and only when on:** `novel/ceiling`, `novel/rows`,
  `novel/prog_med`, `novel/moved_med`, `novel/share`, `novel/visited`.
* **Checkpoint:** `respawn_novel` holds the visit grid (bit-packed) and the
  ceiling, per map. A bare resume restores the flags and the state.
  `record_ckpt.py` lists the flags as TRAIN_ONLY.

Judge it as CLAUDE.md says: by the gate cleared and the step it was cleared at,
from the greedy eval at the true start.

## Measured on sidistic before launch (analysis only)

The probe used the finished sidEM2cap2surf checkpoint's reservoir (20,000
states), a visit grid from its last six evals plus the reservoir, and the
ceiling at its 27.5% frontier.

* **Volume.** 1,843 rows in 0.03 s.
* **Where they landed.** Moved a median of 1,064 u. Progress was median 21.6%
  of the map and p90 26.1%, against the 27.5% ceiling.
* **Rejected:** 56% off the reachable set, 2% past the ceiling, 21% visited,
  16% hull.

In training the visit grid holds the whole fleet's history, so the visited
share will be higher.

## Flags

| flag | default | meaning |
|---|---|---|
| `--respawn-novel` | off | the mechanism |
| `--respawn-novel-frac` | 0.5 | share of the NON-start pool |
| `--respawn-novel-shell` | 0.5 | share of sources from the deep shell |
| `--respawn-novel-shell-width` | 0.25 | the shell, as a quantile of the reservoir's distances |
| `--respawn-novel-radius` | 256 2048 | how far a state is moved, u |
| `--respawn-novel-cell` | 128 | visit grid cell, u (one cell of clearance) |
| `--respawn-novel-tries` | 8 | moves tried per source |
| `--respawn-novel-window` | 1e7 | "visited" = entered within the last W-2W env steps (0 = ever) |

The flag is refused without the reservoir (`--respawn-frac 0`), and with
`--respawn-random`, `--respawn-frontier`, `--respawn-backward`,
`--demo-file`, `--goals` and DDP.

## Tests

`tests/python/test_respawn_novel.py`:

* every row passes every gate, and keeps its source's velocity, view and
  duck state;
* the deep shell delivers what it is asked for;
* the ceiling moves only with real envs;
* tags and the reservoir skip;
* the real reset lands on the pool origin byte for byte;
* checkpoint round trip;
* the TRAIN_ONLY keys;
* a CPU trainer smoke that resumes onto a filled reservoir, builds, logs,
  saves, restores and records;
* the refusals.
