# DDP implementation plan (designed, NOT built)

Produced by a multi-agent design pass: a 42-item survey of per-process state,
three independent designs, each hostile-reviewed, then synthesised. Nothing
here is implemented. The measured numbers it rests on are in
`docs/perf-results.md`.

**Read section 1 before doing anything** — the headline is that the obvious
implementation (wrap the policy in `torch.nn.parallel.DistributedDataParallel`)
would silently never synchronise gradients in this codebase.

# DDP for `train_fast.py` — Implementation Plan

Grounded in: `python/train_fast.py` (1470 lines), `python/surfgym/rewards.py`, `python/surfgym/respawn.py`, `src/env.c:486-491`, `docs/perf-results.md` (box D = 4x RTX 5090 / 2x EPYC 9654, solo **2763.8 ms/iter = 284,542 ticks/s**).

---

## 1. VERDICT

**Build the manual flat all-reduce (design 2), not `torch.nn.parallel.DistributedDataParallel`.**

Both DDP designs scored the same on correctness; the tiebreaker is that **DDP buys nothing measurable here and costs four hazard classes.**

The whole model is **1,958,369 params = 7.83 MB fp32** (verified by arithmetic against `Policy.__init__`, 20 parameter tensors). DDP's default `bucket_cap_mb=25` puts the entire gradient set in **one bucket**, so its reducer fires only when the last gradient (`conv[0].weight`) is ready — i.e. at the end of backward, exactly where a manual post-`backward()` all-reduce fires. **DDP's overlap advantage is zero at default settings.** To get overlap you must set `bucket_cap_mb~2`, which activates Dynamo's `DDPOptimizer` and splits the compiled `mb_step` into ~4 subgraphs — putting S6's measured **1.112x** (update 2202.3 -> 1981.2) at risk to recover ~40 ms. That trade is not obviously positive and neither DDP design could resolve it without a prototype.

Against that non-benefit, DDP costs:

| hazard | evidence in this repo |
|---|---|
| Reducer never arms | `mb_step` calls `policy.forward_split(...)` (`train_fast.py:1173`), which DDP does not intercept. `nn.Module.__getattr__` does not proxy to `.module`, so `ddp.forward_split` raises; the "fix" `ddp.module.forward_split` computes correct numbers and **never all-reduces** — four divergent policies with plausible curves. |
| Graph capture | `step_compute()` calls `policy(static_obs)` (`:1100`) and is captured at `:1117`. A DDP wrapper records forward-time bookkeeping into the graph. |
| Rank-shaped forwards | truncation bootstrap `:1297` (`len(ti)` is per-rank), GAE `last_val` `:1328`, `GreedyTorchPolicy`/`SampledTorchPolicy` `_decide`. |
| Checkpoint interchange | `policy.state_dict()` `:1151` gains `module.`; breaks `tools/record_ckpt.py`, `play.py`, dashboard record buttons, single-GPU resume, and `relayout_optimizer_state`. |
| `static_graph` iteration counting | the compile warm-up at `:1204` is DDP iteration 1; a per-rank compile fallback (`:1213-1217`) desyncs the count and **hangs at iteration 2, not errors**. |

The manual path's one real debt — DDP's constructor parameter broadcast — is three lines of `dist.broadcast` plus a checksum assert. Design 2's review was right that this is "the single largest correctness debt"; the mitigation is to make the assert **unconditional and on by default**, not a debug flag.

**Grafts from the other two designs:**
- `--warm-caches` single-process pre-pass (design 3, step 4) — removes the whole cold-cache race class rather than guarding it.
- `python/surfgym/distributed.py` facade with a literal no-op path at `world_size==1` (design 3, step 1) — keeps the single-GPU path provably untouched.
- The `assert_distinct` on post-reset `core.states_view["origin"]` (design 3, step 7) — the single best idea in any of the three.
- Collective compile-fallback flag + step-0 NCCL microbench gate (design 1, steps 0 and 9).
- Diagnostics hoisted out of the inner minibatch loop (design 3, step 11).

**Corrections I am making to all three designs** (details in §4, §6):
1. **Do NOT seed the minibatch permutation identically across ranks.** Designs 1 and 3 both say "seed it identically, it costs nothing." It costs something measurable. Local index `j = t*N_local + e`; an identical permutation makes global minibatch *k* contain the **same timestep multiset from every rank**, doubling the standard deviation of each minibatch's timestep composition versus independent draws. Timesteps are not exchangeable (advantages and returns correlate with `t`). Seed it **rank-distinctly and deterministically**.
2. **Do NOT append gathered episodes in rank order.** Measured `ep_len_mean` across `runs/*/progress.csv` is **283–2684 ticks**; at 786,432 ticks/iteration that is **293–2,780 episodes finishing per iteration** against `deque(maxlen=200)` (`:1141`). The deque saturates inside a single iteration in every run in the repo, so "append in rank order" leaves `rollout/ep_rew_mean` a permanent metric over the **last rank's env block only** — strictly worse than the 1/R subsample it was fixing. Sort by `(tick_in_iteration, global_env_id)`.
3. **Sort the gathered respawn rows the same way** before pushing. This turns design 1's overclaim ("identical to single-GPU", which its review correctly rejected) into a true statement, for the cost of two int32 columns and a 2k-element argsort.
4. **Batch the advantage-moment all-reduce per epoch, not per minibatch** — 4 collectives/iteration, not 64 (design 2 got this right; design 1 and 3 did not).
5. **Compute the counts delta as `_counts - _counts_base`, not a second `np.add.at`** — the extra scatter would land inside `reward_py`, the most expensive numpy phase (204.4 ms on box D).

### Expected speedup

Projection built from box D's decomposition (update ~2010, lidar ~190, `rollout_fwd` ~170, env 101.5, `reward_py` 204.4, gae 6.6):

| phase | 1 GPU | 4 GPU | basis |
|---|---:|---:|---|
| update | 2010 | **579** | measured 9.70/33.69 = 86.8% eff |
| lidar (512 envs) | 190 | 53 | ~linear + launch floor |
| rollout_fwd (batch 512) | 170 | 90 | extrapolated left of the 62.3%@2048 point |
| env (512 envs) | 101.5 | 40 | work/4, same 384 fork/joins |
| reward_py (512 rows) | 204.4 | 95 | serial numpy, per-op overhead floor |
| gae | 6.6 | 6.6 | 128 sequential launches, N-independent |
| vis_cpu / respawn / book / loop | ~76 | ~60 | |
| **grad all-reduce (exposed)** | 0 | **38** | 64 x 7.83 MB @ 20 GB/s busbw *(assumed)* |
| adv moments + counts + reservoir + stats | 0 | 6 | 4 + 1 + 1 + 3 collectives/iter |
| **rank skew** | 0 | **25** | *unmeasured* |
| **total** | **2763.8** | **~993** | **2.78x** |

**~2.8x on 4 GPUs (band 2.5–3.0x), ~1.8–1.9x on 2 GPUs.** ~791k ticks/s. `docs/perf-results.md`'s 3.4x is correctly labelled a ceiling; **3.47x (0.868 x 4) is the update-only bound** and the update is only 72.7% of the iteration. The remaining 27% splits *worse*, not better — `reward_py` is serial numpy with a per-call floor, `env` pays 384 OpenMP fork/joins regardless of N, `gae` is 128 launches that do not shrink at all, and batch-512 lidar/forward sit left of the measured efficiency curve. The 99.6%-efficiency four-independent-runs result is the **precondition** test (host is not feeding-limited), not a prediction — those four processes never waited for each other.

**Effort: 10–12 days**, one engineer. Not 6. The gradient path is 2 days; the shared-state and ordering work is 4; the validation suite in §5 is 3–4.

**Platform note nobody raised: NCCL is Linux-only.** The multi-GPU path runs only on the rented Ubuntu boxes. On Windows dev, `world_size==1` must never touch `torch.distributed` — that is what the `enabled=False` no-op facade guarantees.

---

## 2. THE PER-MINIBATCH ADVANTAGE NORMALISATION QUESTION

`train_fast.py:1179`, inside the `torch.compile`d `mb_step`:

```python
a = f_adv[idx]
a = (a - a.mean()) / (a.std() + 1e-8)   # per-minibatch, like SB3
```

### Definitive answer: the mean and std MUST be all-reduced.

Not "should" — the naive split changes the objective, and it is invisible in every logged number.

**Why.** Today the estimator runs over the full 16,384-row minibatch. Split four ways, each rank centres **and independently rescales** its own 4,096 rows. Two distinct defects:

1. **Per-rank loss reweighting.** Rank *r*'s advantages are divided by `std_r`, which fluctuates around the global std. At n=4096 the relative sd of a sample std is `1/sqrt(2(n-1)) ~ 1.1%`. So the four shards enter the averaged gradient with weights differing by ~1.1% (1sigma) — small, but **systematic and permanent**, and there is no log line where it appears.
2. **It moves the clip boundary's effect.** `pg = max(-a*ratio, -a*clamp(ratio, 1-clip, 1+clip)).mean()`. `a`'s magnitude decides the gradient weight of clipped vs unclipped rows, so rescaling changes which samples dominate the PPO update, not just their scale.

It is also the **only cross-env coupling in the entire loss** — I verified `ratio`, `vl`, `el` are all per-row, and that GAE (`:1332-1338`) is elementwise over the env axis (`nextval`, `nonterm`, `delta`, `lastgae` are all `(N,)`), with `last_val` from this rank's own `static_obs`. Splitting envs reproduces single-GPU advantages **bit-for-bit per env**. Same for the truncation bootstrap `:1281-1298` and `ep_ret`/`ep_len` `:1139-1140`. So this one line is the whole cross-rank obligation in the loss path.

**Cost of doing it right: 4 collectives of 32 float64 values per iteration.** Batch them per *epoch*, not per minibatch — 16 minibatch moment pairs in one `(2, 16)` all-reduce. Do **not** issue 64 separate 3-scalar all-reduces (design 1 and 3): each would queue on the same stream behind the previous minibatch's ~0.6 ms gradient collective *and* be a hard data dependency for the next `mb_step`, so the honest cost is 10–20x the quoted "~3 ms".

**Three things that will trip you:**
- `torch.Tensor.std()` defaults to `correction=1`. Use `n_g - 1`, not `n_g`. At 16,384 rows the difference is a factor of 1.00003 — irrelevant to learning, **guaranteed** to fail a comparison test and cost an afternoon.
- The epsilon stays **outside** the sqrt: `/(std + 1e-8)`, matching `:1179` character for character.
- Reduce in **float64**. `sumsq - sum*mean` is cancellation-prone and two fp64 scalars cost nothing.

### The code

Hoist the moments (and the tiny `f_adv` gather — 16 KB against the 67 MB image gather, so nothing S6 bought is lost) out of the compiled region, exactly as `ent_t` already is (`:1357`).

`train_fast.py`, replacing `mb_step` (`:1171-1184`):

```python
def mb_step(f_scal, f_img, f_act, f_logp, a_loc, f_ret, idx,
            ent_coef, adv_mean, adv_std):
    with amp:
        logits, value = policy.forward_split(f_scal[idx], f_img[idx])
        logp, ent = logprob_entropy_padded(
            packer.pad(logits.float()), f_act[idx])
        value = value.float()
    ratio = torch.exp(logp - f_logp[idx])
    a = (a_loc - adv_mean) / (adv_std + 1e-8)   # per-minibatch, like SB3;
    # moments are GLOBAL across ranks (see D.adv_moments) so the estimator is
    # over the same 16384 rows a single-GPU run normalises over
    pg = torch.max(-a * ratio,
                   -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
    vl = 0.5 * (value - f_ret[idx]).pow(2).mean()
    el = -ent.mean()
    return pg + args.vf * vl + ent_coef * el, pg, vl, el, logp
```

`train_fast.py`, replacing the epoch/minibatch loop (`:1361-1367`):

```python
for _ in range(args.epochs):
    perm = torch.randperm(T * N, device=device, generator=perm_gen)
    ap = f_adv[perm].view(args.minibatches, mb)      # (M, mb), row k == f_adv[idx_k]
    if D.enabled:
        apd = ap.double()
        st = torch.stack([apd.sum(1), apd.pow(2).sum(1)])       # (2, M) f64
        dist.all_reduce(st)                                     # SUM, one per epoch
        n_g = float(mb * D.world_size)
        m64 = st[0] / n_g
        v64 = (st[1] - st[0] * m64) / (n_g - 1.0)               # ddof=1, == torch.std
        a_mean, a_std = m64.float(), v64.clamp_min(0).sqrt().float()
    else:
        # world_size==1 keeps the LITERAL current estimator so the single-GPU
        # path stays bit-identical — do not "unify" these two branches
        a_mean, a_std = ap.mean(1), ap.std(1)
    for k in range(args.minibatches):
        idx = perm[k * mb:(k + 1) * mb]
        ev_mb = tm.gpu_start("mb_gpu")
        loss, pg, vl, el, logp = mb_step(
            f_scal, f_img, f_act, f_logp, ap[k], f_ret, idx,
            ent_t, a_mean[k], a_std[k])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync_grads()                       # MUST be here — before the clip
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()
        tm.gpu_end(ev_mb)
```

`ap[k]`, `a_mean[k]`, `a_std[k]` are tensors, not Python floats — a scalar in a compiled signature is baked in as a constant and would recompile every iteration, the same reason `ent_t` exists.

**Never `.item()` these.** The collective and the dependent kernels stay on-stream; a host sync here would expose the full latency.

**Do not "fix" this by normalising the whole iteration's advantages once.** That is a different algorithm from SB3's per-minibatch normalisation and changes learning on 1 GPU too.

---

## 3. SHARED STATE

Both documented globals: **share them. Do not shard.** The cost of sharing is ~4 ms against a ~993 ms target — there is no efficiency defence for sharding either one.

### 3a. `RaceReward._counts` — count-based novelty (`rewards.py:467, 545-561`)

The docstring (`:428-433`) is explicit: *"visit counts N GLOBAL across all envs and all episodes… 2048 envs share one count table, so beaten paths wear out in minutes."* The bonus is `int_coef / sqrt(_counts[cell] + 1)` paid on each 256u cell transition.

| option | exploration consequence |
|---|---|
| **Accept sharding** | Each rank's table accumulates at 1/R the rate. For the same fleet-wide visitation, the paid bonus is **exactly sqrt(R) = 2.0x too large at 4 ranks, permanently.** The self-annealing the design rests on takes 4x longer in wall time. Worst: rank 2 grinding a corridor no longer devalues it for ranks 0/1/3 — the fleet stops being one explorer with a shared memory and becomes four smaller ones re-paying novelty on the same beaten path. Dividing `int_coef` by sqrt(R) restores the *magnitude* and nothing else; call it an approximation, never a fix. **Reject.** |
| **Sync per tick** | Exact. 384 all-reduces/iteration of 5.3 MB ~ 7% of wall. **Reject.** |
| **Share via per-iteration delta all-reduce** | **RECOMMENDED.** ~3 ms/iteration (0.3%). |

**Honest residual, stated correctly** — both DDP designs got this arithmetic wrong (they quoted 1.34x / 1.44x, which is the *four-single-entries* case, not the frontier case). For a cell entered *M* times fleet-wide inside one iteration with prior count `n?`:

- single-GPU pays `?_{j<M} c/sqrt(n?+j+1)`
- delta-synced DDP pays `R · ?_{j<M/R} c/sqrt(n?+j+1)`

For `n? = 0` and large *M* this is `2c·sqrt(M)` vs `2c·sqrt(M·R)` — a **sqrt(R) = 2.0x over-payment on freshly discovered cells, for at most one iteration each**, decaying to 1.0x as `n?` grows past *M*. This is the *same* factor as full sharding, but bounded to a single iteration instead of forever, which is what preserves the "beaten paths wear out in minutes" property (that timescale is ~30–60 iterations, not one). Note single-GPU **already has this mechanism at the 1-tick scale**: `:555` reads `self._counts[mc]` *before* `np.add.at` at `:560`, so several envs entering a fresh cell on the same tick all read the pre-increment count. DDP widens the window from 1 tick to 384. That framing is defensible; the "1.34x" framing is not.

Ship `--int-sync-every <decisions>` (default = whole iteration, i.e. 128) so the window can be tightened to 8 syncs/iteration (~2.5% of wall) if an A/B on `int/ep` and `race/success_rate` shows the frontier over-payment matters. Do not tighten it on theory.

**Implementation** (`rewards.py`, new methods on `RaceReward`):

```python
def counts_delta(self) -> np.ndarray | None:
    """Increments since the last sync, int32 (never absolutes)."""
    if self._counts is None:
        return None
    if self._counts_base is None or len(self._counts_base) != len(self._counts):
        self._counts_base = np.zeros_like(self._counts)
    return (self._counts - self._counts_base).astype(np.int32)

def apply_counts_delta(self, fleet_delta: np.ndarray) -> None:
    """fleet_delta is the ALL-RANK sum of counts_delta()."""
    np.add(self._counts_base, fleet_delta, out=self._counts, casting="unsafe")
    self._counts_base[:] = self._counts
```

`_counts_base` initialised to `None` in `__init__`, allocated in `on_reset` alongside `_counts` (and set to a copy of the restored table when `_pending_counts` is applied at `:506`, so a resume starts with a zero delta).

Trainer, once per iteration after the rollout, **on every rank unconditionally**:

```python
if D.enabled and isinstance(reward_fn, RaceReward) and reward_fn.int_coef > 0.0:
    d = reward_fn.counts_delta()
    if d is not None:
        cnt_gpu.copy_(torch.from_numpy(d))      # persistent int32 buffer, 5.3 MB
        dist.all_reduce(cnt_gpu)
        reward_fn.apply_counts_delta(cnt_gpu.cpu().numpy())
```

**Deltas, never absolutes.** That is precisely what makes the checkpoint round-trip safe: `restore_counts` (`:518`) loads the full table on **every** rank, `_counts_base` is set equal to it, and the first sync adds only new visits. Both failure modes the survey named — a 1/R history, and history multiplied by R on the first sync — are **structurally impossible**, not merely avoided.

Cannonball at 256u is ~114x110x106 ~ 1.33M cells = **5.3 MB int32**. Note the threshold in a comment: a much larger map or a smaller `--int-cell` scales this linearly, and past ~50 MB you want a sparse `(cell_id, count)` all-gather instead.

### 3b. `RespawnBuffer._store` — the Go-Explore frontier (`respawn.py:45`)

This is not a statistic, it is **the archive**. Snapshots harvested from every env feed the spawn pool for every env.

| option | exploration consequence |
|---|---|
| **Accept sharding** | Four independent frontiers: a breakthrough found by rank 2 never becomes a start state for 0/1/3. Ring fills at 1/R so it needs R times more iterations to be representative. The `respawn.size >= 2000` gate (`:1229`) is crossed R times later **and on different iterations per rank** (harvest depends on episode outcomes) — a window where one rank trains on 90% mid-run respawns while another is still on 100% fresh starts, feeding **one shared gradient**. That is exactly what the comment at `:1230-1233` says the floor exists to prevent. The 20k-state checkpoint subsample (`:122-131`) persists a quarter of the frontier and re-seeds all ranks from it at every restart. **Reject.** |
| **Share via per-iteration all-gather** | **RECOMMENDED.** ~200 KB/iteration. |

**Volume:** `STATE_DTYPE.itemsize == 108` (verified). Fleet harvest is at most a few thousand rows/iteration. Trivial.

**Why it is semantics-free:** `build_pool` runs at the **top** of the iteration (`:1229`), `observe` runs during the rollout (`:1312`). Deferring a push from mid-rollout to end-of-rollout changes nothing `build_pool` can see, on 1 GPU or N.

**Implementation** (`respawn.py`):
- `observe`'s harvest branch (`:68-74`) appends `(tick, env_id, row)` to `self._out` instead of calling `_push`.
- `drain_harvest() -> (rows, ticks, env_ids)`.
- `push_many(rows)` — a vectorised wrap-aware two-slice ring write. **Required**: with sharing, every rank pushes ALL ~2k fleet rows, and the per-row Python `_push` (`:87-91`) would cost 10–20 ms/iteration.

Trainer, once per iteration, on every rank: size-all-gather (see §6 trap 11), variable-length all-gather of the packed `(row, tick, global_env_id)` bytes, **argsort by `(tick, global_env_id)`**, then `push_many` on every rank.

That sort is the difference between "same set, same distribution" (which design 1's review correctly rejected as an overclaim) and **"`_store`, `_head`, `_size` and eviction are byte-identical to a 2048-env single-GPU run"**, which is a claim you can actually assert in a test. It costs two int32 columns and a ~2k-element argsort (~50 µs).

**Four things fall out for free:**
- `self.rng` keeps `seed=23` on every rank (`respawn.py:53`) and `build_pool` (`:94-119`) produces the **identical 4096-entry pool** everywhere -> `core.set_spawn_pool` uploads the same table and each env draws from it with its own distinct C stream = exactly the single-GPU spawn distribution.
- The `>= 2000` gate is evaluated from a rank-identical quantity, so all ranks flip on the same iteration. **No separate all-reduce needed.**
- `state_dict`/`load_state_dict` need **no collective**, so rank 0 writes and every rank loads the same 20k payload with no R-way duplication biasing `build_pool`'s uniform draw.
- **The checkpoint path stays collective-free** — which is what keeps the wall-clock `ckpt_latest` branch at `:1432` from becoming a deadlock. This is the load-bearing reason to replicate rather than have rank 0 own the state.

### 3c. Verified NOT to need treatment

Checked one by one, so nobody re-litigates: GAE (`:1326-1338`) and the truncation bootstrap (`:1281-1298`) are elementwise over the env axis; every non-race reward's trackers are per-env sized `core.num_envs` (`rewards.py:100-103, 166, 196, 247-248, 339-342`); `RaceReward._d/_best/_since/_ticks/_prev_cell` (`:463-469`) likewise; `CoverageSpeedReward._visited` (`:272`) is per-episode **and** per-env by design and must **never** be shared (the docstring at `:231-232` is explicit); C-side per-env arrays (`src/env.c:400`: `rng`, `once_used`, `goal_hit`, `pending_fail`, `st`/`pp`) are touched only inside their own slot of the `omp parallel for`. Spawn pools (`:872-885`, seeds 17/23) and `race_d0` (`:874`) are pure functions of the map — load-bearing and currently unchecked, hence the startup checksum in step 7.

---

## 4. STEP-BY-STEP

Each step is independently testable. Steps 1–9 must leave `world_size==1` behaviour **provably unchanged**.

### Step 0 — Measure the collective before writing trainer code (30 min)

The entire projection has one input I cannot derive from the repo: NCCL busbw across the SYS hop between NUMA0 (GPU0/1) and NUMA1 (GPU2/3), no NVLink.

`tools/bench_nccl.py`: under `torchrun --standalone --nproc_per_node=4`, allocate `torch.zeros(1_958_369, device=f'cuda:{local_rank}')`, warm 10, then time **64** `dist.all_reduce` calls with one `torch.cuda.synchronize()` at the end. That is exactly one iteration's gradient traffic. Also time a 32-element f64 all-reduce (the moment collective) and run `nvidia-smi topo -m`; check ACS is off (ACS on disables PCIe P2P and even the same-socket hops become host-staged).

**Decision gate:**
- **< 40 ms** -> ship §4 as written, fully exposed all-reduce.
- **40–80 ms** -> ship as written; schedule the overlap hook (step 15) as a follow-up.
- **> 120 ms** -> do the overlap hook first, or reconsider 4 ranks.

Record the number in `docs/perf-results.md` before writing code.

### Step 1 — `python/surfgym/distributed.py` (new file)

A facade so the single-GPU path never touches `torch.distributed`. `Dist` object: `rank, world_size, local_rank, device, is_main, enabled`. `init()` reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from torchrun, calls `torch.cuda.set_device(local_rank)` **first**, then `init_process_group('nccl', timeout=timedelta(minutes=45))`.

Helpers: `all_reduce_sum_`, `all_reduce_mean_`, `broadcast_`, `all_gather_var(t)` (size-all-gather -> pad -> all-gather -> slice; **never a fixed pad**), `barrier()`, `rank0_first()` contextmanager, `assert_equal(tag, buf)`, `assert_distinct(tag, buf)`.

**Every helper returns its input unchanged when `world_size == 1`.** `enabled=False` is a literal no-op path — this is what makes the §5 bit-for-bit gate possible and keeps Windows dev working (NCCL is Linux-only).

*Test:* import on Windows with no env vars; `D.enabled is False`, every helper a no-op.

### Step 2 — OMP team size (`train_fast.py:24`, `_default_omp_threads`)

Runs before the numpy/torch/surfcore import, so it can read `LOCAL_WORLD_SIZE`. Guard against double-division when the launcher already pinned:

```python
n = len(os.sched_getaffinity(0))            # respects taskset/numactl
lws = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
div = 2 * lws if n * lws <= (os.cpu_count() or n) else 2
return str(max(4, min(32, n // div)))
```

On box D (192 cores) 32 -> 24 threads/rank. On a 16-core box this is the difference between 4x8=32 threads on 16 cores — the exact oversubscription S10 exists to prevent (**5.99 vs 0.88 ms/step** measured) — and 4x4=16.

*Test:* set `LOCAL_WORLD_SIZE=4`, assert the value; set it with a narrowed `taskset`, assert no double-division.

### Step 3 — Device and fleet arithmetic (`train_fast.py:823, 829, 839, 903`)

Replace `device = torch.device("cuda" ...)` with `D = distributed.init(); device = D.device`. **This must be the first CUDA-touching line** — before `GpuLidar` (`:922`), `Policy(...).to(device)` (`:929`), every buffer (`:1056-1081`) and the graph capture (`:1106`). Today all four processes resolve to `cuda:0` and either OOM on 4x14.6 GB or run at a quarter speed while looking correct.

```python
N_GLOBAL = args.envs
if args.envs % D.world_size:
    raise SystemExit(f"--envs {args.envs} not divisible by world_size {D.world_size}")
N = args.envs // D.world_size
if (args.n_steps * N) % args.minibatches:
    raise SystemExit(f"T*N_local {args.n_steps*N} not divisible by "
                     f"--minibatches {args.minibatches}")
```

`N` (local) flows unchanged into `default_config(num_envs=N)` `:839`, `RespawnBuffer(N, ...)` `:903`, every buffer `:1056-1081`, and `MB = T*N // args.minibatches` `:1186` -> 2048/4 = 512 envs, T*N = 65,536 rows, /16 = **4,096 rows/rank/minibatch**, exactly the 86.8%-efficiency row, and still **16 minibatches x 4 epochs = 64 gradient steps**, matching single-GPU.

Update `--envs` help text to say **GLOBAL fleet**. The comment at `:477-479` is the reason: *"rew-20 at 52M steps here vs 98M at 8192 envs"*. Require an explicit `--run` when `world_size > 1` (the default at `:483` is `time.strftime` evaluated per process — four ranks launched across a minute boundary derive different run directories).

*Test:* `--envs 2047` on 4 ranks fails loudly at startup with both numbers.

### Step 4 — `--warm-caches` and the artifact races

Add a flag that runs `SurfCore` + `load_zones` + `build_goal_field` + `GpuLidar` and exits. The launcher runs it **once, single-process**, before `torchrun`. This removes the whole class: `zones.py:129` `write_text`, `vision.py:161/225/257` npz writes, `goalfield.py:244` npz write, and the **9–11 GB Bellman-Ford bake**. Four concurrent bakes on GPU 0 is a guaranteed OOM; a torn `.sdf_` npz read by another rank is **silently wrong vision for the entire run**; a torn goal field gives that rank a different `race_d0` and hence a differently-weighted contribution to the shared gradient.

In-process fallback: `rank0_first()` around `load_zones` (`:850`), `build_goal_field` (`:862`, **and pass `device=device`** — `goalfield.py:158` defaults to `'cuda'` with no rank awareness), and `GpuLidar(...)` (`:922`).

Note the barrier-timeout hazard: a cold cannonball bake is 10–30 min per DEPLOY.md against NCCL's default 10-min timeout — hence the 45-min `init_process_group` timeout, and hence pre-warming is the *mechanism*, not the fallback.

*Test:* delete the caches, run `--warm-caches`, then a 4-rank launch; assert no rank rebuilds (log a line when a cache is written).

### Step 5 — Three seed streams (`train_fast.py:1133`, `:929`, `:1362`)

Add `--seed` (int, default 0 — preserves today's hardcoded `core.reset(0)` exactly at `world_size==1`).

**(a) Env streams — rank-DISTINCT, and exactly a partition.** `core.reset(args.seed + D.rank * N)`. `src/env.c:486-491` is literally `uint64_t t = seed + (uint64_t)i; s->rng[i] = sm64(&t) | 1;` — so rank *r*'s envs get streams `sm64(seed + r*N + i)`, and the union over ranks is **bit-for-bit the same set** a single-GPU 2048-env run with the same seed draws. Not "decorrelated" — equivalent. Today every rank calls `reset(0)`, so rank 0 env 7 and rank 3 env 7 draw the **identical** uint64 stream driving the spawn-pool index (`env.c:251`), yaw jitter (`:255`), curriculum draw (`:266-269`) and map-spawn pick (`:288`).

**(b) Policy init — rank-COMMON.** `torch.manual_seed(args.seed)` before `Policy(...)` at `:929`, so `nn.init.orthogonal_` (`:120-126`) draws identical weights. Belt-and-braces with the explicit broadcast in step 6.

**(c) Action noise — rank-DISTINCT, and set BEFORE graph capture.**
`torch.cuda.manual_seed((args.seed * 1000003 + 1 + D.rank) & 0x7FFFFFFFFFFFFFFF)` immediately after `:931`. The Gumbel `torch.rand_like` in `sample_padded` (`:266`) runs inside the graph captured at `:1117`; the philox seed is captured with the graph, so seeding must happen before capture and re-seeding after has no effect on the replay.

**(d) Minibatch permutation — rank-DISTINCT, deterministic.**
```python
perm_gen = torch.Generator(device=device)
perm_gen.manual_seed((args.seed * 2654435761 + 7919 + D.rank) & 0x7FFFFFFFFFFFFFFF)
```
used **only** at `:1362`. **This is where I overrule designs 1 and 3.** They both prescribe a rank-*common* permutation seed ("it costs nothing"). It costs something: local index `j = t*N_local + e`, so identical permutations make global minibatch *k* contain the **same timestep multiset from every rank**. Timesteps are not exchangeable — advantages and returns correlate with `t` — so the coupling roughly doubles the sd of each global minibatch's timestep composition versus independent draws, partly undoing the very variance reduction stratification is supposed to buy. Rank-distinct **and** deterministically derived gives reproducibility *and* independence. There is no goal that a common seed serves.

Record `seed` in `meta["config"]` and in `save_ckpt` so a resume reproduces the fleet.

*Test:* two `world_size==1 --seed 0` runs of 3 iterations produce bit-identical policy tensors (today impossible — this is a new capability and a strong smoke test). Replay the captured graph twice from the same `static_obs` and assert the sampled actions **differ** (guards the philox-offset assumption). Dump the first 10 sampled actions per rank and assert they differ across ranks.

### Step 6 — Explicit parameter broadcast + the checksum assert

There is no DDP constructor to sync weights. This is the manual design's one debt; pay it twice.

After the `--ckpt` load and `relayout_optimizer_state` (`:967-983`) and **before** the graph capture (`:1106`):

```python
if D.enabled:
    for p in policy.parameters():
        dist.broadcast(p.data, src=0)
```

Then assert, at startup and **every 100 iterations, on by default** (not behind a debug flag):

```python
c = torch.stack([sum(p.double().sum() for p in policy.parameters())])
lo, hi = c.clone(), c.clone()
dist.all_reduce(lo, op=dist.ReduceOp.MIN); dist.all_reduce(hi, op=dist.ReduceOp.MAX)
assert torch.equal(lo, hi), f"rank params diverged at iter {it_no}"
```

It must be **exactly** zero. The chain: identical init (broadcast) -> identical grads after all-reduce (NCCL is deterministic and rank-symmetric) -> identical clip scale -> identical fused-Adam step -> identical params. Every link is checkable.

**One thing to state so the assert isn't scary the first time someone reads it:** `torch.backends.cudnn.benchmark = True` (`:824`) selects conv algorithms by timing, so ranks may pick **different algorithms** and produce different *local* forward/backward numerics. That affects the gradient **before** the all-reduce only; after it, all ranks hold identical bytes. So the summed gradient differs from a single-GPU run by kernel-level reassociation (~1e-3 relative), which is the same class the repo already accepted for S6 (`tools/bench_update.py --verify`: loss 1.3e-5 relative, worst gradient drift 9.6e-3, against bf16's own ~4e-3). Parameters never diverge.

*Test:* the gloo 2-rank CPU harness in §5; corrupt one rank's weights deliberately and assert the check fires.

### Step 7 — Startup invariants (turn silent divergence into a startup failure)

One block after construction, before the loop. `assert_equal` on a float64 vector of: policy param checksum; `hash(pool.tobytes())`; `race_d0`; `hash(goal_field grid)`; `obs_dim`, `N`, `world_size`, `args.minibatches`, `args.epochs`, `T`; `hash(meta["config"] minus run name)`.

Then, the single most valuable line in the plan:

```python
D.assert_distinct("spawn_origins", hash(core.states_view["origin"].tobytes()))
```

right after `core.reset(...)` at `:1133`. That one assertion catches a reverted `reset(0)`, a stray global `manual_seed`, **and** a future "reproducibility fix" — the failure mode where every rank simulates bit-identical trajectories, the 2048-env global batch silently becomes R copies of 512 envs, and **not one logged number changes**.

*Test:* revert `reset(base + rank*N)` to `reset(0)` and confirm startup fails.

### Step 8 — `global_step` counts the fleet (`train_fast.py:1316`)

`global_step += N_GLOBAL`, not `N`. One line, and it makes correct **and rank-identical**: the loop bound (`:1223`), `--record-every` (`:1390`), `--ckpt-every` (`:1426`), the `--ent-final` anneal fraction (`:1352-1354`), `BlendedReward.set_step` (`:1226` -> `rewards.py:639`), the `traj_%010d` filenames (`:1392`), the CSV `time/total_timesteps` column and the reported fps (`:1381`). A 4-GPU run would otherwise claim 250M steps at 1B consumed and every sample-efficiency comparison against the baseline would be meaningless. Rank-identical is what keeps every record/ckpt/exit branch from diverging.

It also disarms `BlendedReward._step`'s self-count fallback (`rewards.py:651`), which increments by `len(done)` = the LOCAL env count and would anneal R times too slowly. Add `assert reward_fn._external` for `BlendedReward` under DDP.

*Test:* 2-rank 2-iteration run; assert `global_step` after iteration *k* equals the 1-GPU value.

### Step 9 — The gradient all-reduce (`train_fast.py:1369-1370`)

Build once, after the optimizer:

```python
_flat = torch.zeros(1_958_369, device=device)
_views, off = [], 0
for p in policy.parameters():
    _views.append(torch.as_strided(_flat, p.shape, p.stride(), off))
    off += p.numel()
_inv_ws = 1.0 / D.world_size

def sync_grads():
    if not D.enabled:
        return
    grads = [p.grad for p in policy.parameters()]     # RE-READ every call
    torch._foreach_copy_(_views, grads)
    dist.all_reduce(_flat)
    _flat.mul_(_inv_ws)
    torch._foreach_copy_(grads, _views)
```

**`zero_grad(set_to_none=True)` at `:1368` rebinds `p.grad` to a fresh tensor every minibatch.** A cached grad list would sync stale storage while `opt.step()` uses fresh tensors — four divergent nets with **no error message**, caught only by the step-6 checksum. Re-read `p.grad` every call.

Correctness never depends on strides: `copy_` is a value copy respecting logical indices, so any consistent bijection works. `as_strided` with the params' own strides is chosen only so the copy is a pure memcpy (channels_last strides are a permutation of the index space, so the views are non-overlapping). If `_foreach_copy_` falls into a per-tensor slow path on strided views (20 x 2 x 64 = 2560 launches ~ 8 ms/iteration), profile it once and fall back to per-tensor `copy_` on flattened contiguous slices.

**One all-reduce, not 20.** Twenty separate calls at ~25 µs launch+latency each is 1,280 calls and ~32 ms/iteration of pure overhead on top of the bandwidth.

**Placement is not negotiable: after `loss.backward()`, before `clip_grad_norm_`.** Clipping local gradients and then averaging the clipped results is a different algorithm.

*Test:* gloo 2-rank CPU harness — the averaged gradient must equal the single-process gradient over the concatenated batch, to fp32 rounding.

### Step 10 — Novelty counts sharing

Per §3a. `rewards.py`: `_counts_base`, `counts_delta()`, `apply_counts_delta()`. Trainer: one persistent 5.3 MB int32 GPU staging buffer, one all-reduce per iteration, unconditional on every rank. `--int-sync-every` flag.

*Test:* CPU test — two fake `RaceReward`s visiting known cells, merged table equals a single-process table. Plus a resume test: `_counts.sum()` unchanged across a 2-rank save/restart cycle.

### Step 11 — Respawn reservoir sharing

Per §3b. `respawn.py`: `_out` outbox, `drain_harvest()`, `push_many(rows)`. Trainer: size-all-gather, variable-length all-gather, argsort by `(tick, global_env_id)`, `push_many` on every rank. Use the deferred path at `world_size==1` too (one code path) — it is semantically identical because `build_pool` runs at the top of the iteration.

*Test:* CPU test — `push_many(rows)` leaves `_store`/`_head`/`_size` identical to a loop of `_push(row)`. Cross-rank: `hash(_store[:size].tobytes())` equal on all ranks after 5 iterations, and equal to a 1-GPU reference run's after iteration 1.

### Step 12 — Fleet-wide metrics, all unconditional on every rank

**(a) `RaceReward.pop_stats`** (`rewards.py:596-609`). Extend the return with raw counters `n_success, n_fail, n_trunc, int_paid`, and replace `finish_ticks: list` with a running sum + count (a list cannot be all-reduced). Trainer at `:1385-1388`: call `pop_stats()` on **every** rank, all-reduce the 6-vector, then recompute `success_rate`/`finish_s`/`int_per_ep` from fleet totals. Otherwise `race/success_rate` is a win rate over 512 envs — same expectation, 2x the noise of the single-GPU baseline it will be plotted against, and any threshold comparison is comparing estimators of different variance.

**(b) `ret_hist`/`len_hist`** (`:1141`, appended at `:1303`). Accumulate `(tick_in_iteration, global_env_id, ret, len)` locally; all-gather once per iteration; **sort by `(tick, global_env_id)`**; append on every rank, `maxlen=200` unchanged. That reproduces "the last 200 finished episodes fleet-wide" *exactly* — the same 200 episodes a 1-GPU run's deque would hold.

**Do not append in rank order** (both other designs do). Measured `ep_len_mean` is 283–2684 ticks, so 293–2,780 episodes finish per iteration against `maxlen=200`: the deque saturates within one iteration and rank-ordered append makes `rollout/ep_rew_mean` a permanent metric over the **last rank's env block**, a fixed set of envs with fixed RNG streams. Strictly worse than the 1/R subsample. Also do not widen `maxlen` to `200*R` — the deque would then span R times more iterations and the headline number becomes laggier than the baseline.

**(c) Diagnostics** `kl, loss_v, loss_pi, loss_ent` (`:1374-1376`). Hoist out of the inner loop: today those four `float()` calls fire 256 times per iteration (4.6 ms measured) and only the last minibatch's values survive. Stash the last as tensors; after both loops, one `all_reduce_mean_` of a 4-vector then a single `.tolist()`. Behaviour-identical, makes the logged diagnostics fleet-wide, reclaims the 4.6 ms, and gives a free health check (a per-rank KL that disagrees wildly means the shards desynced).

**Every one of these must sit OUTSIDE any `if rank == 0` block.** A collective inside a rank guard hangs the job.

*Test:* 2-rank run; `race/success_rate` from the merged counters equals the value a 1-GPU run computes over the same episode set (feed both a scripted outcome sequence).

### Step 13 — Rank-0 guards (all collective-free by construction)

`run.json` (`:1025, :1219, :1462`) — and record `'envs': N_GLOBAL, 'envs_per_rank': N, 'world_size': R, 'seed': args.seed, 'ddp': True` so a DDP run can never be confused with the 8192-env ablation. The **entire** `progress.csv` block (`:1033` path check, `:1038-1043` in-place header migration, `:1045` open, `:1436` writerow, `:1445` flush, `:1464` close) — four append handles plus a whole-file rewrite corrupts the run's history. `save_ckpt` (`:1150-1160`) and all three call sites (`:1428, :1433, :1463`). The `eval_core` construction (`:912-920`) and the whole record block (`:1390-1423`). Every print (`:769, 875, 891, 897, 906, 972, 978, 981, 1119, 1210, 1411, 1422, 1455`).

`PhaseTimer(args.timing and D.is_main, ...)` at `:825` — `tools/perf_report.py` parses one `TIMING` line per iteration and interleaved output from four ranks **silently corrupts every perf measurement the frozen benchmark protocol depends on**.

`ckpt_latest` at `:1432` gets `if D.is_main and time.perf_counter() - last_latest_save >= 60.0`, plus a comment in capitals: **this branch is rank-DIVERGENT by construction and no collective may ever be added inside it.** Replication (steps 10–11) is what makes that safe — rank 0's tables *are* the fleet's tables, so the checkpoint needs no gather. This preserves the ~60 s cadence the last commit deliberately tuned.

*Test:* 2-rank run; exactly one `progress.csv`, one `run.json`, one `TIMING` line per iteration, `perf_report.py` parses cleanly.

### Step 14 — Graph capture, compile warm-up, and the launcher

**Graph capture** (`:1112-1118`): pass `capture_error_mode="thread_local"`. A live NCCL communicator runs a watchdog thread doing `cudaEventQuery`; under the default `"global"` mode that aborts the capture, and the `try/except` at `:1120-1122` turns the abort into a **silent `graph = None` on a race-dependent subset of ranks**. Numerics are unaffected (eager `step_compute` and the graph replay compute the same thing) but the skew and throughput are, invisibly. Log which ranks captured.

**Compile warm-up** (`:1200-1220`): keep the warm-up backward at `:1208` outside every rank guard, but it must **not** call `sync_grads()` (there is no `opt.step()` there and the grads are dropped). Make the fallback **collective**: after the try/except, `all_reduce` a success flag with `MIN` and have **all** ranks drop to eager if **any** rank's inductor failed, so the compiled region stays uniform. Set `TORCHINDUCTOR_CACHE_DIR` per rank in the launcher (four ranks autotuning into one `FileLock`'d cache either serializes the 41–60 s compile 4x or races).

**`tools/ddp_launch.sh`:**
```
python3 python/train_fast.py --warm-caches --map ... --reward race ...
torchrun --standalone --nproc-per-node=4 tools/numa_wrap.sh python/train_fast.py \
    --run <name> --envs 2048 --seed 0 ...
```
`numa_wrap.sh` execs `numactl --cpunodebind=$((LOCAL_RANK/2)) --membind=$((LOCAL_RANK/2))` (box D: GPU0/1 on NUMA0, GPU2/3 on NUMA1, SYS between). `os.sched_getaffinity(0)` at `:41` already respects the pinning, so the OMP team size self-corrects.

### Step 15 — Instrumentation, then (conditionally) overlap

Add two `PhaseTimer.FIELDS`: `allreduce` (CPU bracket around `sync_grads`) and `skew` (a `dist.barrier()` immediately before the update, **gated on `--timing`**). The 128-decision rollout (`:1241-1324`) contains **zero** collectives, so ranks free-run for ~285 ms with cost varying by episode outcomes, and the slowest paces all 64 gradient all-reduces. Unmeasured, this is the most likely way the projection misses. Leave the barrier off outside `--timing`: the wait happens anyway inside the first all-reduce, and a permanent barrier prevents a fast rank overlapping its first minibatch with another rank's tail.

**Overlap, only if step 0 or the `allreduce` field says you need it** (~25 lines, still no DDP). The parameter split is unusually favourable here:

| bucket | params | MB | ready in backward |
|---|---:|---:|---|
| `value_head`, `action_head`, `pi`, `vf` | 885,729 | 3.54 | first |
| `conv[4]` (Linear 2048->512) | 1,049,088 | 4.20 | next |
| `conv[0..2]` (the three convs) | 23,552 | 0.09 | last |

Register `p.register_post_accumulate_grad_hook` to fire bucket 1's all-reduce as soon as its last grad lands and bucket 2's right after — both then overlap with `convolution_backward`, which S9's profile puts at **47.5% of update GPU time**. That hides ~90% of the traffic without ever constructing a DDP wrapper, without a Dynamo graph split, and without touching the checkpoint format.

### Step 16 — Tests

Per §5.

---

## 5. HOW TO PROVE IT DID NOT CHANGE LEARNING

Four tiers. Tiers A–C are cheap and must all pass before any long run.

### Tier A — property tests (`tests/python/test_ddp_invariants.py`, CPU + gloo, no GPU box needed)

Extend the existing suite (`python -m pytest tests/python -q`):

1. **Advantage moments.** Synthetic advantages split into R shards; the pooled `(sum, sumsq, n)` reconstruction with `n_g - 1` must equal `torch.cat(shards).std()` to fp32 rounding. **This is the test that catches the ddof slip.**
2. **`push_many` == loop of `_push`.** `_store`, `_head`, `_size` byte-identical.
3. **Counts delta round-trip.** Two fake ranks visiting known cells -> merged `_counts` equals the single-process table. Then a resume: total visits unchanged (catches both the 1/R-history bug and the multiply-by-R bug).
4. **Gather ordering.** Given `(tick, global_env_id)` keys, the merged episode list equals the single-process append order **exactly**.
5. **`all_gather_var`** round-trips variable-length rows in rank order, including a burst that would overflow any fixed pad.
6. **gloo 2-rank harness** (`torch.multiprocessing.spawn`, CPU): the flat all-reduce + `mul_(1/ws)` gradient equals the single-process gradient over the concatenated batch; the param checksum is exactly equal after 20 steps.
7. **Checkpoint interchange.** A run's `ckpt_final.pt` loads into the single-GPU trainer, `tools/record_ckpt.py:143` (`policy.load_state_dict(ck['policy'])`, no prefix stripping) and `play.py`. Free with the no-DDP design — **pin it so nobody wraps the policy later.**

### Tier B — the update path is numerically what it was (`tools/bench_update.py`)

Add `--verify-adv`, in the shape of the existing `verify_compile` (`bench_update.py:207`), which is the repo's own standard: *"Not 'the training curves look similar' — that hides a real drift behind PPO's own noise."*

- **Eager (`--no-compile`), `world_size==1`:** the new hoisted-moment `mb_step` must be **bitwise** identical to the old in-region expression on the same frozen inputs — loss and every gradient. This is why the `D.enabled` branch keeps the literal `ap.mean(1)` / `ap.std(1)`; do not "unify" the two branches into one "mathematically equivalent" formula.
- **Compiled:** hold it to the same tolerance S6 was held to — loss <= 1e-4 relative, worst gradient drift <= 5e-2 relative.

### Tier C — the 1-GPU vs 2-GPU run

**State the honest thing first: a 1-GPU and a 2-GPU run cannot agree trajectory-wise, and must not be expected to.** Run A draws one `(2048,6,15)` Gumbel tensor per decision; run B draws two `(1024,6,15)` tensors from different streams. Even with identical action seeds the fleets diverge, because the env streams are (correctly) a partition. Any protocol that demands matching curves is asking for the impossible and will be quietly weakened until it proves nothing. So split it:

**C1 — EXACT, at t=0, before any gradient step.** Instrument a `--dump-invariants` flag that writes one JSON line and exits after `core.reset` + pool construction:

| quantity | requirement |
|---|---|
| `hash(concat_r core.states_view["origin"])` | run B (rank0?rank1) **equals** run A's 2048-env hash |
| initial spawn pool bytes | equal |
| `race_d0`, goal-field hash | equal |
| policy param checksum | equal (same frozen ckpt) |
| `obs_dim`, `MB`, `epochs x minibatches` | 4096x2 vs 8192x1... **`MB` differs by design; `epochs*minibatches == 64` must match** |

The first row is the whole seeding argument, proved rather than asserted. It is exact and it takes seconds.

**C2 — EXACT, every iteration, WITHIN the 2-GPU run.** These are the cheap asserts that actually catch the silent bugs, and they run in production, not just in the test:
- `hash(respawn._store[:size])` equal across ranks (and equal to a 1-GPU reference for iteration 1)
- `hash(reward_fn._counts)` equal across ranks
- `hash(spawn_pool)` equal across ranks at every `build_pool`
- `respawn.size` equal across ranks (proves the `>= 2000` gate flips together)
- param checksum MIN == MAX (step 6)
- `global_step` identical

**C3 — STATISTICAL, against a measured null.** This is the part every design hand-waved. You cannot compare one 2-GPU curve to one 1-GPU curve; you need the intrinsic spread first.

1. Run the **1-GPU trainer 3x with `--seed 0/1/2`**, resumed from the frozen `runs_ckpt.pt` (= `ckpt_6348079104`), for **+500M steps** = 636 iterations ~ 30 min each on one 5090 (786,432 ticks/iteration at 2763.8 ms). Config frozen per protocol 0.2: `surf_src_cannonball`, maxvel 4000, `--respawn-frac 0.9`, `--act-every 3`, `--envs 2048`, lidar 128x64.
2. That gives a per-step band for every `progress.csv` column. The ones that matter: **`race/eval_progress`** (the cleanest learning signal — it comes from `race_progress()` on the rank-0 `eval_core`, a 1-env greedy rollout on the game-authentic platform start, so it is directly comparable), `race/success_rate`, `rollout/ep_rew_mean`, `train/approx_kl`, `train/entropy_loss`.
3. Run **2 GPUs, `--envs 2048` (1024/rank), `--seed 0`**, same 500M steps.
4. **The 2-GPU trace must lie inside the 3-seed 1-GPU band at every logged step**, and in particular `train/approx_kl` must sit in the band the whole way — a KL that runs systematically high or low is the signature of a broken advantage normalisation or a mis-weighted gradient.

Total cost: about 2 GPU-hours. Cheap enough that there is no excuse for skipping it.

**One extra A/B, only if `--int-coef > 0`:** the novelty staleness (§3a) is the single deliberate approximation. Compare `int/ep` (the `race_int` field, `:1451`) and `race/success_rate` between the 2-GPU run and the 1-GPU band. If `int/ep` runs high early and converges, that is the predicted sqrt(R) frontier over-payment; decide with `--int-sync-every 16` whether it costs anything real. Do not tighten it prophylactically — 8 syncs/iteration is ~2.5% of wall.

### Tier D — perf, on the frozen protocol

`--timing` + `tools/perf_report.py --vs <baseline>`, `PhaseTimer` on rank 0 only. Measure the box's **own** 1-GPU baseline the same day — absolute ms never cross boxes. Report the IQR (0.4–0.6% of median in-run), not a tail percentile. New fields to watch: `allreduce`, `skew`, and `mb_gpu / update` — it is 99.8% today because the GPU always has 43 ms of queued work at each `float()` sync; at 9.7 ms/minibatch that queue is 4.4x shallower, so a ratio below ~0.95 means the host has become the update's critical path (step 12c's diagnostics hoist already removes the main cause).

---

## 6. WHAT NOT TO DO

1. **Do not call `torch.manual_seed(same)` on all ranks for the action noise.** Identical weights + identical action noise + the hardcoded `reset(0)` makes every rank simulate **bit-identical trajectories**. The 2048-env global batch silently becomes R copies of 512 envs and the averaged gradient equals a 512-env gradient. **No loss curve, KL, entropy or reward number changes — only sample efficiency, by ~R.** The minibatch-permutation requirement is exactly what tempts you into this. Three streams, three rank-affinities (step 5). The `assert_distinct` on post-reset origins is what makes it loud.

2. **Do not seed the minibatch permutation identically across ranks.** Designs 1 and 3 both recommend it. `j = t*N_local + e`, so an identical permutation gives global minibatch *k* the same timestep multiset from every rank, roughly doubling the sd of its timestep composition. Rank-distinct, deterministically derived, from a dedicated generator.

3. **Do not normalise advantages per-shard.** §2. Also do not "fix" it by normalising the whole iteration's advantages once — that is a different algorithm from SB3's per-minibatch normalisation and changes 1-GPU learning too.

4. **Do not reinterpret `--envs` as per-GPU.** "Each GPU gets 2048" is 8192 globally, which the repo's own ablation (`train_fast.py:477-479`) measures as **rew-20 at 98M steps vs 52M** — a sample-efficiency regression that presents as a throughput win. The VRAM headroom DDP creates (14.6 GB -> ~5–7 GB/card; `b_img` alone 4.3 -> 1.1 GB) makes it look free. Assert divisibility, say GLOBAL in `--help`, and record `envs`/`envs_per_rank`/`world_size` in `run.json`.

5. **Do not clip before syncing.** `sync_grads()` goes between `loss.backward()` and `clip_grad_norm_` (`:1369-1370`). Clipping local gradients then averaging is a different algorithm and it is the easiest thing to get backwards.

6. **Do not cache the `p.grad` list.** `zero_grad(set_to_none=True)` at `:1368` rebinds every grad each minibatch. A cached list syncs stale storage while `opt.step()` uses fresh tensors — four divergent nets, no error message, caught only by the step-6 checksum. That assert is load-bearing, not belt-and-braces.

7. **Do not put a collective inside a rank-conditional or wall-clock branch.** Specifically: `if rank == 0`, the `ckpt_latest` branch at `:1432` (decided from each process's own clock), the compile-fallback path at `:1213-1217`, and the eval/record block (`:1390-1423`, which stalls rank 0 for 2.7 s today and **28–48 s** for a policy that survives to the 12,000-tick cap). All are safe today precisely because they contain no collective, and all three are exactly where someone will later add "let me just gather X before saving". Replication is what makes that unnecessary. Make it a code-review checklist item, not a comment.

8. **Do not wrap the policy in DDP "just for the constructor broadcast."** `mb_step` calls `forward_split` (`:1173`), which DDP does not intercept; `ddp.module.forward_split` computes correct numbers and never all-reduces. Three lines of `dist.broadcast` do the same job with none of the surface. If a future engineer does wrap it, exactly one call site may touch the wrapper (`grep -n ddp_policy` must return exactly two lines) and `save_ckpt` must save `policy.module.state_dict()`.

9. **Do not sync the novelty counts per tick.** 384 all-reduces/iteration ~ 7% of wall.

10. **Do not maintain the counts delta with a second `np.add.at`.** It lands inside `reward_py`, the most expensive numpy phase (204.4 ms on box D), 768 calls/iteration. `_counts - _counts_base` is one vectorised subtract per iteration.

11. **Do not use fixed-size pads for the gathers.** Steady state is ~2k respawn rows/rank/iteration, comfortably under any cap — but an env holds up to `ep_ticks/snap_every` = 120 pending rows at `--ep-ticks 12000`, so a correlated mass-truncation burst overflows a 4096-row pad and **silently discards frontier states**. Size-all-gather first, then pad to the max. Same hole in any capped `ret_hist` gather.

12. **Do not append gathered episodes (or respawn rows) in rank order.** §1 correction 2 and step 12b. Measured `ep_len_mean` 283–2684 ticks means the 200-deque saturates inside one iteration, so rank-ordered append pins `rollout/ep_rew_mean` to the last rank's env block permanently. Sort by `(tick_in_iteration, global_env_id)`.

13. **Do not let four ranks bake the goal field or the SDF.** `build_goal_field` defaults `device='cuda'` (`goalfield.py:158`) with no rank awareness — four concurrent 9–11 GB Bellman-Ford wavefronts on GPU 0 is a guaranteed OOM, followed by four racing `np.savez_compressed` on one path. A torn `.sdf_` npz read by another rank is **silently wrong vision for the whole run**; a torn goal field gives that rank a different `race_d0` and a differently-weighted contribution to the shared gradient. `--warm-caches` once, out of band.

14. **Do not leave `capture_error_mode` at its default.** With a live NCCL communicator the watchdog's `cudaEventQuery` can abort the capture at `:1117`, and the `try/except` turns that into a silent `graph = None` on a race-dependent subset of ranks. Pass `capture_error_mode="thread_local"` and log which ranks captured.

15. **Do not let the default `--run` stand under `torchrun`.** `time.strftime("fast_%m%d_%H%M")` at `:483` is evaluated per process; four ranks launched across a minute boundary derive different run directories. Require `--run` explicitly when `world_size > 1`.

16. **Do not enable bf16 gradient compression** (or any comm hook) without a `bench_update.py --verify`-grade numerics check. It halves the 38 ms but adds ~4e-3 relative noise to the averaged gradient, and this repo's own standard is that "the curves look similar" is not a numerics gate.

17. **Do not compare absolute ms across boxes**, and do not carry the prompt's box-A decomposition (2794 ms, env 257, `reward_py` 139) into a box-D projection. Box D is 2763.8 with env 101.5 and `reward_py` 204.4 — the phase that scales worst is the one the slow EPYC clock makes expensive.

18. **Do not expect a 1-GPU and an N-GPU run to match trajectory-wise.** §5 C3. Demanding it produces a test that gets weakened until it proves nothing. Compare against a measured 3-seed band.

19. **Do not try NCCL on Windows.** The multi-GPU path is Linux-only. The `enabled=False` facade is what keeps the Windows dev loop working and keeps the single-GPU path provably identical.

20. **Do not "optimise" `CoverageSpeedReward._visited` into a shared table.** It shrinks by R under the split, which is a memory win. Sharing it would change the objective outright — `rewards.py:231-232` is explicit that novelty there is per-episode and per-env, and every episode must re-earn its route.
