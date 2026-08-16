# Perf results (measured)

Running record for `docs/perf-implementation-plan.md`. Every row is the
median of the last 30 of 40 iterations from the frozen benchmark checkpoint,
per protocol 0.2. Numbers are milliseconds per iteration (786,432 ticks).

**Rigs** — three rented vast.ai boxes, each 1x RTX 5090 / Ubuntu 24.04 /
torch 2.13.0+cu130 / triton 3.7.1, but NOT otherwise comparable:

| box | host | cores | RAM | role |
|---|---|---:|---:|---|
| A | ssh9.vast.ai:12801 | 16 | 64 GB | reference baseline, learning-safety runs |
| B | 192.165.134.28:15040 | 192 | 251 GB | item benchmarks |
| C | 82.141.118.44:24704 | 64 | 440 GB | retired — 15.4% IQR, unusable for measurement |

**Every speedup is a within-box ratio.** Absolute ms never crosses boxes:
the CPU count alone changes the `env` and `reward_py` phases, and the boxes
are shared, so each one measures its own reference commit before an item.
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
| S9 channels_last | 1e9f6c9 | 3239.4 | 242,771 | **1.203x** | update 2785.8 -> 2200.4 (1.266x), rollout_fwd 224.7 -> 167.9 (1.338x) |
| S7 early-exit march | 0e7e402 | 2985.8 | 263,378 | **1.305x** | lidar 439.4 -> 190.4 (2.31x); 1.086x on top of S9 |
| S10 OMP team cap | 935ae53 | 2987.4 | 263,238 | 1.305x | **no-op on box A (16 cores) by design; 1.407x on box B (192 cores)** |
| S6 torch.compile | c408bf8 | 2771.4 | 283,779 | **1.406x** | update 2202.3 -> 1981.2 (1.112x); 60 s one-time compile |
| S3 split bf16 obs | b00b1a1 | 2756.0 | 285,365 | **1.414x** | 1.015x on top of S6; rollout buffer 8.6 -> 4.3 GB, VRAM 18.3 -> 14.6 GB |
| S1 async evals | not merged | 3018.6 | — | **0.994x** | measured REGRESSION — see below |

**Run the reference and the item back to back.** A single S7 run on box A
came back with `update` at 2398 ms against the reference's 2200 — a 9% swing
in a phase S7 does not touch. Two interleaved S9/S7 pairs put `update` at
2203.9 / 2204.5 / 2205.1 / 2204.8 (ratio 1.000) and `total` at 3242.8 /
2985.8 / 3241.4 / 2985.8, i.e. both pairs reproducing to 0.1 ms. The
one-off was the box, not the change. Boxes B and C had independently shown
`update` unchanged across S9->S7, which is what prompted the re-measurement.

Run-to-run precision: the interquartile spread of `total` inside a run is
0.4-0.6% of the median, so a real 2% change is resolvable. Report the IQR,
not a tail percentile: a rented box takes occasional host-contention blips
(4 of 30 iterations in the S9 run ran 10-20% long, with `env` and `update`
degrading together) that move a p90 without touching the median.

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

S2 is folded into S6 rather than run as its own measured item: a 1.001x change
cannot be validated by a harness with 0.4% noise.

## S9 — channels_last conv trunk (new item, not in the audit)

`tools/bench_update.py --profile` on the isolated minibatch found that the
single largest consumer of update GPU time was not arithmetic at all:

| kernel | calls / 3 steps | CUDA ms | share |
|---|---:|---:|---:|
| `convolution_backward` | 9 | 62.3 | 47.5% |
| **`nchwToNhwcKernel`** | 36 | **37.5** | **28.6%** |
| `cudnn_convolution` | 9 | 22.1 | 16.9% |
| **`nhwcToNchwKernel`** | 15 | **7.3** | **5.5%** |
| `threshold_backward` (ReLU) | 12 | 10.5 | 8.0% |
| `clamp_min` (ReLU) | 12 | 7.3 | 5.5% |
| adaptive_avg_pool2d fwd+bwd | 6 | 8.5 | 6.4% |

**34% of all update GPU time was layout conversion.** cuDNN's tensor-core
convolutions are NHWC; handed NCHW tensors it transposes in and back out
around every conv, forward and backward, on 16384 samples of 64x128.

Holding the trunk `channels_last` and restriding the depth image into NHWC
before it removes those kernels. Depth is one channel, so `(B,1,H,W)` NCHW
and NHWC hold the same bytes in the same order and the restride is a free
`view+permute`; the arithmetic and the weights are untouched.

Isolated-step A/B (`tools/bench_update.py`, ms per minibatch):

| variant | ms | x |
|---|---:|---:|
| eager (NCHW) | 43.32 | 1.000 |
| bf16obs (S3) | 42.50 | 1.019 |
| avgpool | 42.15 | 1.028 |
| compile (S6) | 40.12 | 1.080 |
| **chlast (S9)** | **33.36** | **1.298** |
| chlast+bf16obs | 32.52 | 1.332 |
| chlast+compile | 33.15 | 1.307 |
| chlast+avgpool+compile | 32.19 | 1.346 |

S9 alone captures 1.298 of the 1.346 available from that whole family, and
S3/S6/avgpool are each worth 2-8% *on top of the update only* — i.e. ~1-3%
end-to-end. That is why S9 went first.

Two things it cost, both now pinned by tests:

* **Fused Adam refuses a param/moment layout mismatch.** Every checkpoint
  predating this branch carries NCHW moments, so the first benchmark died on
  iteration 1 with "params, grads, exp_avgs, and exp_avg_sqs must have same
  dtype, device, and layout". `relayout_optimizer_state()` copies mismatched
  state into an `empty_like(param)` on resume. It must survive even a revert
  of the layout itself, or checkpoints written here stop resuming.
* **`Flatten` must keep LOGICAL (C,H,W) order**, not memory order, or every
  weight downstream of the trunk is silently permuted. It does — pinned by
  `test_flatten_keeps_logical_channel_order`.

## S7 — early-exit SDF march (the audit's premise was wrong)

The audit expected a hierarchical coarse-grid march, on the theory that in
the maxvel-4000 regime the agent flies over open air, sky rays never hit,
and each burns all 64 sphere-trace steps. `tools/bench_lidar.py` censused
16.8M real rays from mid-run poses in the checkpoint's own respawn
reservoir:

| statistic | value |
|---|---|
| mean steps per ray | **9.87** |
| median / p90 / p99 | 8 / 19 / 39 |
| rays that exhaust the 64-step loop | **0.12%** |
| rays that never hit (reach `range`) | **0.00%** |
| loop trips doing real work | **15.4%** |

There are no sky rays — the SDF pads the world boundary as solid, so every
ray terminates — and the fine march is already adaptive. The premise, and
with it the case for a coarse grid, does not survive contact.

What the census DID find is that cost tracks the loop's trip count, not its
work: render time is 0.43 ms at MAX_STEPS=8 and 5.71 ms at 64. Two causes,
each worth about 2x:

1. **No early exit.** `for _ in range(MAX_STEPS)` is a constexpr loop with no
   break, so a block pays all 64 trips even when every lane died at step 9.
   Block-max step counts average 24.5 at BLOCK=64 and 34.0 at BLOCK=256.
2. **An unroll cliff.** 1.65 ms at 32 trips vs 4.90 at 48 — 1.5x the trips
   for 3x the time. A constexpr bound is fully unrolled and the register
   pressure past ~32 costs more than the marching. A runtime bound stops it.

Kernel A/B (`tools/proto_march.py`, ms per 2048-env render, all bit-exact):

| kernel | BLOCK | warps | ms | x |
|---|---:|---:|---:|---:|
| production (constexpr, no exit) | 256 | 4 | 5.70 | 1.00 |
| runtime loop only | 256 | 8 | 3.57 | 1.60 |
| early exit | 256 | 4 | 1.86 | 3.07 |
| early exit | 128 | 4 | 1.81 | 3.15 |
| **early exit** | **64** | **2** | **1.37** | **4.15** |
| early exit | 64 | 4 | 5.15 | 1.11 |

The block/warp surface is sharp — one ray per thread (warps = BLOCK/32) is
the rule; get it wrong and the early exit is worth nothing. In situ the
`lidar` phase went 439.4 -> 190.4 (2.31x rather than 4.15x: the phase also
carries the 67 MB depth copy, and the live fleet's poses are more clustered
than reservoir samples).

Bit-exactness is not a tolerance here. `alive` is only ever `&=`'d and a
dead lane adds 0 to `t`, so leaving early cannot change an output;
`tests/python/test_lidar_march.py` asserts `max|diff| == 0` against a
verbatim copy of the old kernel on the 8 specified poses, on both the Linux
and the Windows 5090. The benchmark runs agree: rew 13.85 vs 14.09, kl
0.047 vs 0.042, ep_len 1652 vs 1651. No learning-safety run is needed for a
change that cannot alter a single number.

## S10 — cap the OpenMP team (a rental-shape trap, not a code bug)

`surf_step` is an `omp parallel for` over the envs and the rollout forks that
team 384 times per iteration; numpy's reward math shares the runtime. The
team defaults to the core count, and sizing it to EVERY core makes the worker
threads spin against the master. The failure is invisible on a small box and
severe on a big one, which is exactly backwards from what renting suggests.

Paired runs, `main` vs S10:

| box | cores | main ms | S10 ms | x | env ms |
|---|---:|---:|---:|---:|---|
| A | 16 | 2985.8 | 2987.4 | 0.999 | 243.8 -> 244.4 |
| B | 192 | 4418.9 | 3141.7 | **1.407** | 915.3 -> 144.9 |

On A the setting torch already lands on is the one S10 picks, so it is a
deliberate no-op there — which is the point: the same code now behaves on a
192-core box. S10 also *stabilises* B: its two S10 runs came back at 3141.6
and 3141.7 ms while its two `main` runs differed by 47 ms.

Note the interaction with renting. Before S10, paying for a 192-core machine
bought a 1.48x SLOWER iteration than a 16-core one (4418.9 vs 2985.8) purely
through this default. After it, core count is close to irrelevant (3141.7 vs
2987.4) — which is the correct shape, since the trainer is GPU-bound.

## S6 — torch.compile the minibatch step

`max-autotune-no-cudagraphs`, not `reduce-overhead`: 1.067x vs 1.011x on the
isolated step. The no-cudagraphs part is deliberate — the rollout already
owns a CUDA graph, and the update is 99.8% GPU-busy rather than launch-bound,
so there is nothing for a second graph capture to win.

In the trainer it measured better than in isolation (1.112x on the update,
1.076x end-to-end) because the minibatch gathers sit inside the compiled
region there. Compile costs 41-60 s once, warmed before the loop rather than
on iteration 1 so a toolchain failure can fall back to eager instead of
killing an overnight run.

Two traps worth naming:

* **`ent_coef` must go in as a 0-d tensor.** A Python float in a compiled
  signature is baked in as a constant, so an `--ent-final` schedule would
  recompile the whole region every single iteration.
* **"the curves look similar" is not a numerics gate.** PPO noise hides real
  drift. `tools/bench_update.py --verify` runs compiled and eager on the SAME
  weights and the SAME minibatch: loss agreed to 1.3e-5 relative, worst
  gradient drift 9.6e-3 relative. bf16's own relative precision is ~4e-3, so
  that is reassociation, not a change of math.

## S1 — async evals: a measured REGRESSION (0.994x), not merged

The premise is sound: the eval is 2 x n_rec batch-1 episodes with no
stall-kill, so its cost grows exactly as the policy learns to survive, and
the trainer blocks on it. The implementation — write a light snapshot, spawn
`tools/record_ckpt.py --both`, fill the eval csv columns when it exits — works
correctly. It is still slower.

Per-iteration totals make the mechanism unambiguous. S1's iterations are
cleanly bimodal:

| arm | iterations | median ms |
|---|---:|---:|
| reference, no eval running | 34 | 2757 |
| reference, eval iteration | 2 | 7055 |
| **S1, no eval running** | 16 | **2758** |
| **S1, eval subprocess running** | 20 | **3304** |

The S1 code change itself is free (2758 vs 2757). But the eval subprocess
taxes every training iteration by 546 ms while it runs, and it runs for ~10
iterations per record point. At `--record-every 12e6` that is 303 ms/iter
against the blocking version's 240 ms/iter.

**Why: two CUDA contexts time-slice badly on a GPU that is already 85%
busy.** The eval's own kernels are tiny (batch-1 renders), but they queue
behind the trainer's big ones and the context switches cost both sides. The
same recording that takes 4.3 s in-process takes ~28 s as a subprocess.
Startup is not the culprit — it is only 4.8 s total (torch 0.7, goal field
1.8, SDF 2.3).

Scaling it out does not rescue the idea. For a policy reaching the
12,000-tick cap at the default `--record-every 25e6`, blocking costs ~337
ms/iter and the subprocess ~444 ms/iter — the contention grows with the
eval's duration too, so the async version loses in the regime it was meant
to fix.

**The right fix is to make the eval cheaper, not to move it.** Running the
n_rec episodes as n_rec parallel envs in one core would cut its decisions and
its kernel count ~3x while keeping one CUDA context. That needs `record.py`
to write per-env trajectories and is left as future work; the branch
`perf/s1-async-evals` is pushed but unmerged.

## Final: 1.407x end-to-end, measured back to back

`perf/baseline-ref` (fcb9ad4) and `main` run alternately on box A, two pairs:

| phase | baseline | main | x |
|---|---:|---:|---:|
| **total** | **3930.4** | **2793.9** | **1.407** |
| update | 2784.8 | 1971.8 | 1.412 |
| &nbsp;&nbsp;mb_gpu | 2779.9 | 1967.2 | 1.413 |
| rollout_wall | 1133.9 | 797.2 | 1.422 |
| &nbsp;&nbsp;sync_copy | 663.9 | 357.4 | 1.858 |
| &nbsp;&nbsp;lidar | 440.0 | 190.4 | 2.311 |
| &nbsp;&nbsp;rollout_fwd | 224.6 | 167.5 | 1.341 |
| &nbsp;&nbsp;env | 285.8 | 257.1 | 1.112 |
| &nbsp;&nbsp;reward_py | 139.7 | 139.5 | 1.001 |

Individual runs: baseline 3928.9 / 3931.9, main 2821.6 / 2766.2.
**200,087 -> 281,482 ticks/s.** VRAM 18.3 -> 14.6 GB.

The same paired measurement on box B (192 cores, where S10 also applies):

| phase | baseline | main | x |
|---|---:|---:|---:|
| **total** | **5446.7** | **2904.4** | **1.875** |
| update | 2862.8 | 2022.9 | 1.415 |
| rollout_wall | 2546.4 | 853.8 | 2.982 |
| **env** | **1181.3** | **125.7** | **9.40** |
| lidar | 455.5 | 202.3 | 2.252 |

144,388 -> 270,770 ticks/s; runs 5488.2 / 5405.1 and 2908.6 / 2900.2.

The cross-box comparison is the useful part. **Before this pass the 192-core
box was 1.39x SLOWER than the 8-core box** (5446.7 vs 3930.4 ms) purely
through the OpenMP default. After it they are within 4% (2904.4 vs 2793.9).
Core count is now irrelevant to this trainer, which is the correct shape for
a GPU-bound workload — and a warning that the "bigger" rental was costing
money before S10.

## Where the remaining time is, and what is left

At 2793.9 ms the shape has not changed — the update still dominates:

| phase | ms | share |
|---|---:|---:|
| update | 1971.8 | 70.6% |
| rollout GPU (lidar + fwd) | 357.9 | 12.8% |
| rollout CPU-serial (env + reward + book) | ~415 | 14.9% |
| gae + pool + misc | ~6 | 0.2% |

GPU busy is ~2335 ms = 83.6%. That bounds everything left:

| lever | ceiling | why it is capped there |
|---|---|---|
| hide ALL CPU-serial work (S5/S8) | **1.18x** | the GPU is the bottleneck, so overlap can only reclaim the ~415 ms the CPU spends alone — and `env` (257 ms) is INSIDE the fwd->env->lidar chain, so the realistic figure is ~1.06x |
| further update work | small | 31 ms/minibatch is cuDNN-conv-bound; compile, layout, bf16 and pool variants are all spent |
| S4 reward on GPU | <1.0x | reward_py is 139 ms of CPU time; moving it onto an 84%-busy GPU adds to the critical resource |

**The pure-software pass is essentially done at ~1.4x.** The audit's "~2.5-3x
software" assumed the update was launch- and bandwidth-bound with easy wins;
measured, it is cuDNN convolution on a 16384x1x64x128 input, and conv1 alone
(1 input channel, so cuDNN uses its generic engine rather than an
implicit-GEMM tensor-core kernel) is the single largest layer.

Everything past this point costs something other than engineering time:

* **Science levers** (the audit's list, unchanged): lidar 64x32 is ~4x less
  conv work; a stride-4 patchify first conv removes the 1-channel generic
  engine entirely; 4 epochs -> 3 is linear. Each trades sample efficiency or
  perception and needs an A/B on learning, not on wall clock.
* **Hardware**: with the software at 1.4x and the card 84% busy, a faster GPU
  now converts almost linearly, which was not true at the start.
