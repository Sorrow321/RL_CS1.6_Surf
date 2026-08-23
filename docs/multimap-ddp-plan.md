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

**The map set.** 620 maps surveyed (`tools/survey_maps.py`,
`runs/research/map_survey.json`): **47 ready**, 126 blocked by end-ward
stage-link teleports, 447 with no in-BSP timer zones.

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

**The risk, and the gate.** Coarser voxels dilate geometry - the slab
occupancy is deliberately conservative, so false walls only make shaping
less generous, never farmable. The real danger is a corridor narrower than
one cell being walled off entirely, which turns a map into
`surf_src_sidistic`: goal unreachable from spawn, trains forever at 0%.
**Gate: for each map, sweep cell 32/48/64/96 and keep the largest cell at
which the goal is still reachable from the spawn AND d0 is within a few
percent of its cell-32 value.** That is a per-map number, not a global one -
tight maps keep a fine field, open ones go coarse. Maps that only work at
cell 32 are candidates for dropping.

**Rollout VRAM** (`b_img` is `(T, N, FRAME)` float32, FRAME = 64x32):

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

## Component gates

### A. CPU scaling and the OMP trap  (no rental; local)

Measure physics wall-time per step vs `OMP_NUM_THREADS` at fixed envs, then
at fixed threads vs envs. Confirm `torchrun` really does clamp it to 1 and
that the launcher's export restores it.

**Gate:** a documented curve of steps/s vs threads, and the per-rank thread
count that keeps `env` at or below its current 11% of the iteration. **If
physics does not scale with cores, more envs per box is pointless and the
whole plan should shift to more boxes with fewer ranks each.**

### B. Envs scaling, single GPU, single map  (no rental; local)

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
