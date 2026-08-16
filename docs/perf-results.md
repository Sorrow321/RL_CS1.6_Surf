# Perf results (measured)

Running record for `docs/perf-implementation-plan.md`. Every row is the
median of the last 30 of 40 iterations from the frozen benchmark checkpoint,
per protocol 0.2. Numbers are milliseconds per iteration (786,432 ticks).

**Rig**: rented vast.ai box, 1x RTX 5090, Ubuntu 24.04, 16 cores, torch
2.13.0+cu130, triton 3.7.1.
**Frozen inputs**: `/root/RL_Surf/runs_ckpt.pt` (= ckpt_6348079104),
`surf_src_cannonball`, maxvel 4000, respawn_frac 0.9, act_every 3, 2048 envs,
lidar 128x64 @ range 11500 / near 2000 / cell 32u.
**Command**:

    python3 -u python/train_fast.py --ckpt runs_ckpt.pt --run perfbench \
        --record-every 1e12 --steps <S0+40*786432> --timing > runs/pb_<item>.log
    python3 tools/perf_report.py runs/pb_<item>.log --vs runs/pb_baseline.log

## Table

| item | commit | total ms | ticks/s | vs baseline | notes |
|---|---|---|---|---|---|
| baseline | fcb9ad4 | 3895.8 | 201,864 | 1.000x | matches the 3.90 s/iter reference in DEPLOY.md exactly |

Run-to-run precision: the p10..p90 spread of `total` inside a run is 0.6% of
the median, so a real 2% change is resolvable.

## Baseline decomposition

| phase | ms | % of iter | what it is |
|---|---:|---:|---|
| **update** | 2785.8 | 71.5% | 4 epochs x 16 minibatches, CPU wall |
| &nbsp;&nbsp;update_gpu | 2785.2 | — | GPU span of the same region |
| &nbsp;&nbsp;mb_gpu | 2781.2 | — | sum of the 64 per-minibatch GPU spans |
| **rollout_wall** | 1101.0 | 28.3% | the 128-decision loop, CPU wall |
| &nbsp;&nbsp;sync_copy | 663.5 | 17.0% | buffer copies + the per-decision device sync (= CPU blocked on the GPU) |
| &nbsp;&nbsp;env | 266.4 | 6.8% | 384 `core.step` calls |
| &nbsp;&nbsp;reward_py | 128.2 | 3.3% | RaceReward, numpy |
| &nbsp;&nbsp;vis_cpu | 19.8 | 0.5% | pose gather + H2D + render launch |
| &nbsp;&nbsp;respawn | 12.9 | 0.3% | RespawnBuffer.observe |
| &nbsp;&nbsp;book | 2.9 | 0.1% | episode bookkeeping |
| &nbsp;&nbsp;boot | 0.6 | 0.0% | truncation V(s_T) bootstrap |
| &nbsp;&nbsp;rollout_fwd | 224.7 | 5.8% | graphed forward+sample, GPU |
| &nbsp;&nbsp;lidar | 439.4 | 11.3% | SDF march + depth copy, GPU |
| gae | 6.6 | 0.2% | |
| pool | 0.5 | 0.0% | respawn pool rebuild |
| ckpt | 0.0 | 0.0% | time-based, ~1 write per 15 iterations |
| record | 0.0 | 0.0% | 2751 ms on the one recording iteration |
| misc | 0.4 | 0.0% | csv + prints |

Derived:

* **GPU busy ~3453 ms = 88.6%** of the iteration (update 2785 + lidar 439 +
  fwd 225 + gae 4). The card is nearly saturated; there is no idle GPU to
  reclaim by moving work onto it.
* **CPU-only serial ~431 ms = 11.1%** (env + reward + respawn + book +
  vis_cpu). This is the ceiling for S5-style overlap.
* `sync_copy` 663.5 ~= `rollout_fwd` 224.7 + `lidar` 439.4 = 664.1: the
  per-decision sync spends its entire time waiting for exactly the GPU work
  it enqueued. The rollout is a strict GPU-then-CPU ping-pong, sum not max,
  exactly as the audit described.
* **43.5 ms per minibatch** (2781.2/64) at ~0.74 TFLOP/mb = ~17 TFLOPS
  effective, ~8% of the 5090's bf16 peak.

## What the measurement changes vs the audit

The audit's estimates were log-derived; five of them move materially.

| claim | audit | measured | consequence |
|---|---|---|---|
| PPO update share | ~40% | **71.5%** | the update is THE target, not one of three |
| GPU lidar share | 19-25% | **11.3%** | S7's whole-iteration ceiling is 1.13x, not 1.3-1.8x |
| C env share | 1.5% | **6.8%** | still not worth porting, but it is no longer noise |
| reward numpy + bookkeeping | 10-20% | **3.3%** | S4's throughput case is weak on its own |
| 256 per-minibatch `float()` syncs | "hidden serializer" | **4.6 ms/iter total** | S2's ceiling is 1.001x |
| eval/ckpt overhead | 11-37% of wall | **2.2%** amortized | the merged cadence change already took most of S1 |

The `float()` result is the sharpest: `mb_gpu` (2781.2) is 99.8% of the
update's CPU wall (2785.8), so the host is never the update's critical
path. The 256 syncs cost nothing because the GPU has 43 ms of queued work to
chew on at every one of them — they wait for work that has to happen anyway.
S2 therefore cannot buy throughput; it stays on the list only as a
prerequisite for putting the minibatch step inside a compiled/graphed region
(S6), where a mid-region host sync IS fatal.

Two caveats on the numbers that move the other way:

* **S1's value scales with policy quality.** The recording cost measured
  here (2751 ms for greedy+stoch, 3 episodes each) is small because this
  checkpoint's agents die early (mean episode 1500 ticks). A policy that
  survives to the 12,000-tick cap costs what the audit measured (+7 to
  +48 s), i.e. up to ~28% of wall at the current `--record-every 25e6`. S1
  is cheap insurance against a cost that grows exactly as training succeeds.
* **`env` at 6.8%** is 384 `core.step` calls at 0.69 ms each for 2048 envs =
  0.34 us/env-tick, ~2.9M env-ticks/s against the 13M/s bench figure. The
  gap is the batch-step's own bookkeeping (autoreset, obs writeback, goal
  test), not `pm_tick`.

## Revised priorities (evidence-ordered)

Ceilings, if the phase went to exactly zero:

| lever | targets | ceiling |
|---|---|---|
| PPO update (S3 + S6 + arch) | 2785.8 ms | 3.54x |
| overlap rollout with update (S8) | 1101.0 ms | 1.39x |
| lidar (S7) | 439.4 ms | 1.13x |
| hide CPU-serial work (S5) | 430.8 ms | 1.12x |
| recordings (S1) | ~86 ms amortized | 1.022x (grows with policy quality) |
| per-minibatch syncs (S2) | 4.6 ms | 1.001x |

Order actually followed, and why it departs from the plan's S1-S2-S3-S6-S7:
S1 first (unchanged — cheap, safe, and it practices the protocol on a low-risk
item), then the update stack, then S7. S2 is folded into S6 rather than run as
its own measured item, because a 1.001x item cannot be validated by a harness
with 0.6% noise.
