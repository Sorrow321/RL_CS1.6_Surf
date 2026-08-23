# Massive multi-map DDP run - plan and component gates

Goal (user, 2026-08-23): one training run over all usable maps at once, on
multi-GPU DDP, with aggregated metrics - "average % of route covered" and
"% of maps the agent can finish". The bet is that breadth buys the
generalisation that no single-map mechanism has.

**Every component below is separately testable and has a numeric gate. Do
not integrate before its gate passes; the integration is where a silent
failure becomes uninterpretable.** Outlier maps may be dropped.

---

## What is already measured (do not re-derive)

**The map set - and the paper shortlist did NOT survive verification.**
620 maps surveyed (`tools/survey_maps.py`). After two corrections the survey
reads **69 ready / 104 stage-linked / 447 no-zones**, but entity parsing is
not proof:

| stage | count |
|---|---|
| paper "ready" (first survey) | 47 |
| **verified trainable** (`tools/verify_maps.py`) | **22 pass, 24 fail, 1 ambiguous** |
| after the `detect_zones` origin fix | 43 + `kei_luupy` |
| **distinct maps after de-duplicating `_b2`/`_b3` re-releases** | **19** |

**A 47-map fleet built from the paper list would have carried 25 silent
nulls.** `train_fast.py` hard-fails on unreachability ONLY inside the
`--race-kill-aware` branch, and `mapfleet.py` checks nothing at all - so
those maps would have trained forever at 0% while dragging the aggregate
metric down and looking like "the agent does not generalise". **Gate the
fleet on `verify_maps.py` before it ever runs. This is the single most
important prerequisite on this page.**

Why 24 failed: **22 are staged** - the finish's component is entered by a
teleport, and the spawn sits in a sealed start room whose only exit is a
teleport, which `--teleport-fail` turns into instant death. 2 have unusable
spawns, 1 is a narrow link. **15 of them come back for free** by seeding the
`spawn_mode=2` pool at that teleport's destination - no code change, and the
reachability check then passes by construction.

**A real bug was found and fixed on the way** (`zones.py`,
`tests/python/test_zone_origin_offset.py`): `detect_zones` ignored a brush
entity's `origin` key, so origin-brush maps got phantom zone boxes. 9 of 173
zoned maps were affected; `surf_sg_dash` passed all four checks against a
phantom finish and fails against its real one. **No map in `maps/` except
sidistic carries such a trigger, so every trained checkpoint's zone boxes
are unchanged** - no past result is invalidated.

**And the survey's own link rule was wrong ~half the time.** 696 of 1,491
end-ward teleports (47%) land within 512 u of a spawn - a catch net that
happens to respawn you nearer the finish is still a catch net. Requiring the
destination to be away from every spawn moved the count 47 -> 69 ready.

**The Surf Gateway service adds a SECOND, DISJOINT source of zones.** A
decompiled AMXX plugin exposes `POST http://buttons.surfcs.net/` with
`map=<name>`, returning the timer buttons' world positions. Fetched once for
all 620 maps at 1 req/s (`tools/fetch_gateway_buttons.py`, cache in
`runs/research/gateway_buttons.json`).

**It covers exactly the maps the BSP does not**: 0 of the 69 BSP-ready maps,
but **173 of the 447 no-zone maps**, 166 of which got a zone file. The two
sources are complementary, not redundant - which also means the
cross-validation was only possible on one map (`surf_floathub`, where they
agree: 0.0 u horizontal offset, 216 u of pure Z between a floor curtain and a
wall button).

**The honest funnel is 447 -> 21, not 447 -> 166:**

| stage | count |
|---|---|
| no-zone maps | 447 |
| service has buttons | 173 |
| got a zone file | 166 |
| pass `verify_maps.py` | **30** |
| ... with a race over 2,000 u | **21** |

**So the trainable set is ~19 (BSP-verified) + 21 (gateway) = ~40 maps**, not
620 and not 47.

Three properties of gateway zones that matter for the metric:
* **They are 37x smaller than BSP finishes** (median cross-section 21,904 u^2
  vs 808,960). A null on a gateway map is not evidence about the map.
* Zones are emitted with **pad 64** = the engine's `+use` radius, so the box
  is definitionally "where a human could have pressed it"; below pad 32, nine
  finish buttons have no standable point at all. The unpadded `true_aabb` is
  kept in every file.
* **Start button to spawn: median 98 u**, but **18 maps exceed 512 u** (worst
  `surf_airfrance` 6,713 u). `train_fast.py` times from spawn, so on those 18
  the clock is NOT comparable to a human record.

Zone files live in `runs/research/gateway_zones/` because
`maps_full_dataset/` is read-only; `load_zones` resolves
`<bsp>.parent/<stem>.zones.json`, so they go live when copied next to their
BSP. `load_zones` was also inverted to trust any non-`auto` source verbatim -
it previously discarded a `gateway` file and returned no zones at all.

**Goal-field cost.** 47 maps = 10.7G voxels @cell 32 = 15.9x cannonball.

| | |
|---|---|
| serial bake | **8.5 GPU-hours** |
| on 6 boxes | ~1.4 h wall-clock, **~$1.11** |
| 10 smallest maps (smoke subset) | 60 Mvoxel, **~3 min** total bake |

**Goal fields are CPU-resident numpy and this decides the architecture:**

| layout | RAM per rank |
|---|---|
| every rank loads all 47 | **21.4 GB - not viable** |
| sharded over 8 ranks | **2.7 GB - fine** |

So **maps must be sharded across ranks.** That is also the natural
data-parallel shape - gradients all-reduce, so one policy still sees every
map - but it means the run is NOT "every rank trains every map", and eval
must all-gather per-map results rather than reporting rank 0's.

**Voxel size is the biggest single lever, and the two cells are wrongly
coupled.** Voxels scale as `cell^-3`, so coarsening the goal field is
cubically cheap:

| cell | total Gvoxel | uint16 RAM | per rank /8 | serial bake | vs cell 32 |
|---|---|---|---|---|---|
| 32 (today) | 11.01 | 22.0 GB | 2.75 GB | 8.7 h | 1.0x |
| 48 | 3.31 | 6.6 GB | 0.83 GB | 2.6 h | **3.3x** |
| 64 | 1.42 | 2.8 GB | 0.35 GB | 1.1 h | **7.8x** |
| 96 | 0.43 | 0.9 GB | 0.11 GB | 0.3 h | 25.5x |

Today `train_fast.py:2001` sets `slot.cell = args.lidar_cell or
pick_cell(core)` - **ONE variable drives both the lidar SDF and the goal
field**, though they do unrelated jobs. The lidar cell is perception
fidelity (depth error is on the order of the voxel size). The goal-field
cell is only the reward's spatial resolution, and the field is a smooth
distance function read trilinearly, so it tolerates coarseness far better.
**Decoupling them lets perception stay at 32 while the field goes to 64 -
7.8x less bake and RAM for something the reward barely notices.** At cell 64
the whole 47-map field set is 2.8 GB and shards to 0.35 GB per rank, which
removes the RAM constraint entirely.

`pick_cell` already auto-coarsens (smallest power of two from 16 that fits a
700M-voxel budget), so the giant maps are partly handled; the win here is
applying it deliberately to the field rather than inheriting the lidar's
value.

**MEASURED on cannonball, 2026-08-23 - cell 48 is safe, cell 64 is NOT.**
Baked the field at 48 and 64 and compared both against the existing cell-32
field along the 1,811-vertex champion route:

| cell | bake | voxels | d0 at route v0 | monotone steps along the champion line |
|---|---|---|---|---|
| 32 | ~32 min | 671M | 198,353 | 95.6% |
| **48** | **411 s** | 202M | **198,935** (+0.3%) | **95.5%** |
| 64 | 71 s | 86M | **95,122 (HALF)** | **70.6%** |

**The failure mode is the OPPOSITE of what was predicted here, and it is the
dangerous direction.** The worry above was that coarse voxels would DILATE
geometry and wall off a corridor - a conservative failure that only makes
shaping stingy. What actually happens at cell 64 is TUNNELLING: the grid
stops resolving thin geometry, the wavefront flows through floors and walls,
and `reach_max` collapses from 199,634 to 98,833 with reachable voxels
falling 19.9M -> 8.3M. d0 halves because the geodesic found shortcuts that
do not exist, and monotonicity along the champion line drops 95.6% -> 70.6%,
i.e. the field stops being a usable progress coordinate.

That is exactly the defect `goalfield.py` samples occupancy on an 8u lattice
to prevent ("a 10u floor between two track stages reads solid instead of
letting the geodesic tunnel through it and paint a permanent reward trap").
At 64u the slab sampling can no longer catch it. **A tunnelled field is
worse than a coarse one: it is a false shortcut the shaping will actively
drive the agent into.**

**Revised numbers at the safe cell (48): 3.3x cheaper than 32, not 7.8x.**
All 47 maps = 3.31G voxels, 6.6 GB uint16 total, **0.83 GB per rank sharded
over 8**, 2.6 h of serial bake. The RAM constraint is still removed; the
saving is just smaller than cell 64 promised.

**Gate, per map and not global:** sweep 32/48/64 and keep the largest cell
where (a) `reach_max` and d0 stay within a few percent of the cell-32 value
and (b) monotonicity along a reference path does not degrade. **Checking
reachability alone is NOT sufficient - at cell 64 every one of the 1,811
route vertices still read as reachable while the field underneath them was
nonsense.** Tight maps keep a fine field; maps that need cell 32 and are
huge are drop candidates.

**Rollout VRAM - the table below was 2x TOO PESSIMISTIC.** `b_img` has been
**bf16** since the split-obs change (`train_fast.py:1883`), not float32.
Measured per rank at T=32 on a 3090: 5.06 GB at 8,192 envs, 8.29 at 16,384,
14.27 at 32,768, ~21.3 at 65,536. Keep the shape of the table, halve the
numbers:

| envs | T=128 | T=64 | T=32 | T=16 |
|---|---|---|---|---|
| 2,048 | 2.15 GB | 1.07 | 0.54 | 0.27 |
| 4,096 | 4.29 | 2.15 | 1.07 | 0.54 |
| 8,192 | 8.59 | 4.29 | 2.15 | 1.07 |
| 16,384 | **17.18** | 8.59 | 4.29 | 2.15 |

A 3090 has 24 GB and also holds the policy, optimizer, CUDA graphs and the
lidar SDF (~2-3 GB on cannonball). **N=16,384 at T=128 does not fit; at
T<=32 it does.** This composes with the `n_steps` result - small T won on
learning as well - so **large N with small T is the natural configuration**,
not a compromise.

**The physics is OpenMP-parallel** (`src/env.c:534`, `:676`), so more CPU
cores genuinely help. **But `torchrun` forces `OMP_NUM_THREADS=1`** (already
recorded in CLAUDE.md; `tools/ddp_launch.sh` exports it back). Under DDP the
ranks SHARE one machine's cores, so per-rank cores fall as ranks rise. "More
GPUs means more CPU" is true per box and false per rank - **that is the
central risk of this plan and gate A exists to measure it.**

**The blocker.** `ddp` and `multimap` are disjoint branches: `ddp` has zero
references to `--maps`, `multimap` has zero references to DDP. The
integration is the real work.

---

## ANSWERED 2026-08-23: the hardware, the ceiling, and the cost

**Buy 4x RTX 3090, one rank per GPU, 32,768 envs/rank, T=32,
`--minibatches 16`, launched with `tools/ddp_launch.sh 4`.** That is
1,301,629 steps/s = **4.69e9 steps/hour at $0.151 per 1e9 steps** - 1.44x
better than the previous best, with no code change.

| configuration | $/h | best steps/s | **$ per 1e9 steps** |
|---|---|---|---|
| 1x 3090 | 0.168 | 363,819 | **0.129** |
| **4x 3090** | 0.707 | **1,301,629** | **0.151** |
| 8x 5090 | 3.201 | 5,050,843 | 0.176 |
| 4x A100 SXM4 80G | 4.908 | 2,538,464 | 0.537 |

**Do not buy A100/H100 - 3.6x the cost per step.**

**The parallelism ceiling is 32,768 envs/rank, and it is NOT VRAM.** The
ladder on 4x3090: 1,109,409 -> 1,208,896 -> **1,301,629** -> 1,220,259 ->
dies (8k/16k/32k/65k/131k). The same peak appears on a 32 GB 5090 and an
**80 GB A100 that still has 54 GB spare at 65,536 envs and gains 1.2%**.
Buying VRAM buys nothing.

**What stops scaling is `update`** - per-sample cost falls 7.54 -> 6.61 ->
5.93 us then rises to 6.49. `env` scales linearly and **`allreduce` is
CONSTANT at 426-435 ms at every rung**, so its share falls 15.3% -> 2.1%:
bigger ranks make communication cheaper, not dearer.

**And that turnover is a minibatch-SIZE artefact, not an envs limit.**
`--minibatches` is a count, so it doubles with envs; re-running 65,536
envs/rank at `mb=32` gives **1,320,715 steps/s** with update cost back to
5.92. Not free - it doubles the all-reduce count and `mb` is part of the
pinned learning config.

**DDP does not turn over by 8 ranks**: 84.0% efficiency at 4 ranks, 81.3% at
8, 90.3% on NVLink A100s. **68% of the 4-rank penalty is CPU contention in
the physics step, not communication** - `env` 472 -> 1,527 ms while `update`
rose only 373 ms. Confirmed twice: NVLink cuts all-reduce from 428 ms to
27.9 ms and buys back only 6 efficiency points. **The rank limit is
`cores/ranks`, exactly as gate A predicted. Scale by ADDING ranks, never by
growing them.**

**Multi-map RAM is not a constraint.** RSS is ~4.3-6.1 GB per rank and envs
are nearly free (~7.5 KB/env - 32,768 envs cost 0.25 GB). Headroom is
~63 GB/rank. 161 maps at cell 48 held UNSHARDED by every rank (~22.6 GB)
fits; at cell 32 unsharded (~75 GB/rank) it does not. **So sharding is an
optimisation here, not a requirement** - which removes gate E from the
critical path.

**The tension is resolved.** Separate single 3090s remain cheaper per step
($0.129 vs $0.151) but the premium is now 1.17x, down from 1.8x - and one
3090 is **hard-capped at 363,819 steps/s** (65,536 envs is 7.5% slower,
131,072 dies). One policy on one GPU gets 1.31e9 steps/hour, about 8.1M per
map per hour over 161 maps. Take DDP.

## Component gates

### A. CPU scaling and the OMP trap  -- **DONE 2026-08-23, PASSED**

Measured on the 32-core local box, cannonball, 2048 envs, T=32:

| OMP threads | env ms | speedup | **total iter ms** | env % of iter |
|---|---|---|---|---|
| 1 (torchrun's default) | 331.4 | 1.00x | 685.4 | **48.4%** |
| 4 | 126.8 | 2.61x | 455.2 | 27.9% |
| **8** | 66.8 | 4.96x | **406.9 (best)** | 16.4% |
| 16 | 35.4 | 9.36x | 428.8 | 8.3% |

**Physics scales, but the ITERATION does not scale past 8 threads.** env keeps
falling to 16 threads (9.4x) while total time RISES 406.9 -> 428.8, because
`update` goes 231 -> 283 ms: the OMP threads contend with the python/torch
work feeding the GPU. **More cores for physics is not free.**

**The trap is real and expensive:** at `OMP_NUM_THREADS=1`, which is what
torchrun sets, physics is **48.4% of the iteration** against 16.4% at 8
threads - a ~1.7x throughput loss if a DDP launcher forgets to export it.

**Sizing rule: ~8 cores per rank.** On a 32-core box that is 4 ranks
comfortably; **8 ranks gives 4 cores each and sits below the knee.** So "more
GPUs means more CPU" holds only while cores/rank >= 8 - an 8-GPU box needs
~64 cores to avoid starving physics, and that is a box-selection criterion,
not an afterthought.

### A-old. CPU scaling and the OMP trap  (superseded)

Measure physics wall-time per step vs `OMP_NUM_THREADS` at fixed envs, then
at fixed threads vs envs. Confirm `torchrun` really does clamp it to 1 and
that the launcher's export restores it.

**Gate:** a documented curve of steps/s vs threads, and the per-rank thread
count that keeps `env` at or below its current 11% of the iteration. **If
physics does not scale with cores, more envs per box is pointless and the
whole plan should shift to more boxes with fewer ranks each.**

### B. Envs scaling  -- **DONE 2026-08-23, and it CAPS the plan's ambition**

Same box, 8 OMP threads, T=32:

| envs | ms/iter | steps/s | vs 2048 | update us/sample |
|---|---|---|---|---|
| 2,048 | 415.5 | 473,184 | 1.00x | 3.82 |
| **4,096** | 634.3 | **619,921** | **1.31x** | 2.67 |
| 8,192 | 1380.6 | 569,631 | 1.20x | 3.23 |

**Throughput PEAKS at 4,096 envs and FALLS at 8,192.** Per-sample update cost
rises 21% from 4k to 8k - the GPU is saturated, so more envs buy nothing and
cost VRAM.

**This contradicts the premise of "envs = 2048 x 2/4/8".** On a 3090-class
GPU the useful range is ~4,096 envs per rank, not 16,384. Scaling past that
must come from MORE RANKS, not bigger ranks - which is also what the goal
field RAM argues for. 16,384 was not measured (the harness timed out); it is
moot unless a much larger GPU changes the saturation point, which is exactly
what the A100/H100 pass in gate H is for.

### B-old. Envs scaling, single GPU, single map  (superseded)

`--envs 2048 / 4096 / 8192 / 16384` crossed with `--n-steps 32 / 64`. Record
steps/s, VRAM, and the TIMING phase split at each point.

**Gate:** the (envs, T) pairs that fit in 24 GB with headroom, and the
throughput curve. Expect it to bend where CPU physics saturates - **find
that knee, it sizes everything downstream.**

### C. Multi-map correctness at current scale  (no rental; local)

On `multimap`: 2-3 maps, verify per-map `eval_per_map`, per-map `scale =
100/d0`, and that trajectories carry their own map tag.

**Gate:** per-map evals differ and are attributed to the right map; a
deliberately unreachable map shows as 0 rather than contaminating the mean.

### D. DDP correctness  (rental: smallest multi-GPU box)

On `ddp`: the existing exactness gates. Confirm the 1.73x-on-4x3090 figure
reproduces and that `OMP_NUM_THREADS` is exported.

**Gate:** gradient/step equivalence gate passes; measured speedup recorded.

### E. Map sharding across ranks  (new code)

Each rank loads only its shard's cores, lidars and goal fields. Shards
should be balanced by **voxel count, not map count** - the distribution is
brutally skewed (`surf_src_epiphany` and `surf_src_driftless` are 4.5G of
the 10.7G total).

**Gate:** RAM per rank matches prediction; every map appears in exactly one
shard; a rank failing to bake its shard fails the run loudly rather than
training on a subset silently.

### F. Aggregated metrics  (new code)

All-gather per-map results each eval and report: mean % of route per map,
**fraction of maps with >=1 finish**, and the full per-map table.

**Gate:** on a rigged case - one map solved, one at zero - the aggregate
reads correctly from every rank. The metric must distinguish "learns all
maps a little" from "learns two maps well"; the transfer measurement in the
backlog (a cannonball finisher covers 0.7-0.9% of an unseen map) says that
distinction is the entire question.

### G. Integration smoke  (rental: 4x3090, 1 hour)

10 smallest ready maps (~3 min of bakes), sharded, aggregated metrics on.

**Gate:** it runs an hour without dying, the aggregate metric moves, and
per-map numbers are attributable. **This is a plumbing test, not a result -
1 h from scratch is below the ~2.5 h CLAUDE.md says is needed to conclude
anything.**

### H. Scale and price/performance  (rental)

Only after G. 8x and 16x of the cheap card, then a 30-minute A100/H100 pass
purely for the table. **Never debug on the expensive hardware** - arrive
with a config that already works.

**Gate:** steps/s per dollar-hour per configuration, and the rank count
where DDP scaling turns over.

---

## Order, and why

A and B first: they are free, local, and can kill or reshape the plan before
any rental. C and D are independent and can run in parallel. E and F are the
new code and depend on C and D passing. G integrates. H scales.

**The two most likely failure modes, stated in advance so they are not
rationalised later:** per-rank CPU starvation under torchrun making more
ranks actively worse (gate A), and the aggregate metric being dominated by a
handful of easy maps so that "it generalises" is indistinguishable from "it
learned the three short ones" (gate F).
