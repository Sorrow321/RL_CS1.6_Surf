# Performance implementation plan (hand-off)

> **STATUS: executed. Results and the revised picture live in
> `docs/perf-results.md` — read that first.** End-to-end 1.407x on the
> reference box (3930 -> 2794 ms/iter, 200k -> 281k ticks/s), and **1.875x
> measured** on the 192-core box where S10 also applies (5447 -> 2904; see the
> cross-box table in `docs/perf-results.md`). What actually shipped, and how it
> differs from the plan below:
>
> | item | planned | outcome |
> |---|---|---|
> | Phase 0 | timer, protocol, semantics tests | done; 27 tests |
> | S1 evals off hot path | 1.15-1.6x | **0.994x, NOT merged** — two CUDA contexts contend worse than the block they remove |
> | S2 one sync per update | 1.05-1.1x | **dropped** — measured ceiling 1.001x |
> | S3 split bf16 obs | 1.1-1.2x | 1.015x (+ VRAM 18.3 -> 14.6 GB) |
> | S6 torch.compile | 1.15-1.3x | 1.076x |
> | S7 hierarchical march | 1.3-1.8x | **premise refuted**; an early exit + runtime loop bound gave 1.086x, bit-exact |
> | S9 channels_last | *not in the plan* | **1.203x** |
> | S10 OMP team cap | *not in the plan* | 1.407x on 192 cores, no-op on 16 |
>
> The two biggest wins were not on the list, and three of the listed items
> were worth a fraction of their estimate. That is what Phase 0 was for.

Executable follow-up to `docs/perf-audit-x10.md`. Scope: implement + measure
the audit's software items on the RENTED Linux box. Read the audit first for
the why; this file is the what/how/verify.

## Ground rules (read before touching anything)

1. **Work on the rented Linux box only** (`ssh -p 12801 root@ssh9.vast.ai`,
   repo at `/root/RL_Surf`, dashboard on :8000). The LOCAL Windows machine
   runs a live training (`race_respawn`) — do not stop, edit, or benchmark
   on it; reading its logs is fine. The remote `bench_remote` run may be
   stopped freely — the box exists for this work.
2. **Git flow**: local repo -> commit -> push -> `git pull` on the remote
   (GitHub works on this box; if a future box blackholes GitHub, see
   DEPLOY.md). One branch/commit per item (`perf/s1`, `perf/s2`, ...).
3. **Local command shapes** (Windows Git Bash): NO `cd ... && ... > file`
   compounds, NO `for` loops — both force manual permission prompts. Use
   straight-line commands and python heredocs; `until ...; do sleep; done`
   polls are fine. Inside an ssh remote-command string anything goes.
4. **Do-not-break invariants** (each has bitten us already):
   - Physics stays bit-exact: never touch `src/pm.c`; the C env is 1.5% of
     wall — there is nothing to win there.
   - Reward semantics: potential shaping must telescope (loops net ~0);
     ended-tick rows are post-autoreset (NEW episode) and must stay masked;
     stall-kill and respawn-harvest rules unchanged. Anything touching
     `rewards.py`/`goalfield.py` sampling must pass the semantics test
     (write it first — see Phase 0.3).
   - Depth encoding formula (near-linear + exp tail, near=2000) is
     warm-start ABI — do not alter output values, only how fast they are
     computed (S7 must be pixel-verified).
   - Checkpoint self-description: any new flag needs the ckpt-config
     save/restore dance in `train_fast.py` (grep `restored.append` for the
     pattern).
   - Cache .npz signatures embed bsp size+mtime_ns — see DEPLOY.md if
     caches mysteriously rebake.
5. After each substantive item, run an adversarial review pass over the
   diff before merging (this repo's convention; reviews have caught real
   reward-corruption bugs every time).

## Phase 0 — measurement harness (before any optimization)

0.1 **Phase timer**: add `--timing` to `train_fast.py`. Accumulate per
    iteration, print one parse-friendly line:
    `TIMING iter=N rollout_fwd=… lidar=… env=… reward_py=… sync_copy=… gae=… update=… ckpt=… record=… total=…` (ms).
    Use `time.perf_counter()` brackets; for GPU phases add
    `torch.cuda.Event(enable_timing=True)` pairs around the graphed forward,
    `lidar.render`, and the update minibatch loop (sync once per iteration
    to read events, not per phase). Insertion points: `policy_step()`,
    `fill_vision()`, the `core.step` + reward sub-loop, GAE block, the
    epochs loop, `save_ckpt`, the recording block.
0.2 **Benchmark protocol** (applies to every item):
    - Frozen inputs: `/root/RL_Surf/runs_ckpt.pt` (= ckpt_6348079104) is the
      benchmark checkpoint; never overwrite it.
    - Run: `python3 -u python/train_fast.py --ckpt <frozen> --run perfbench
      --record-every 1e12 --steps <S0+40*786432> --timing > runs/pb_<item>.log 2>&1`
      (40 iterations, no recordings, then it exits by itself).
    - Metric: median of the last 30 TIMING totals + per-phase medians.
      Baseline BEFORE any change, re-measured after every item. Keep a
      running table in `docs/perf-results.md` (item, per-phase ms, total,
      speedup vs baseline, commit hash).
    - Learning-safety check for items that could affect training math
      (S3, S6, S7): 30-minute run from the frozen ckpt, `race/eval_progress`
      must stay in the 60-95k band it currently occupies; a collapse below
      ~40k = revert.
    Reference numbers: remote baseline 3.90 s/iter non-recording
      (201k ticks/s); local Windows 5.76 (for context only).
0.3 **Semantics tests** (port to `tests/python`, pytest): (a) scripted
    goal-crossing pays success_bonus and ends the episode (place a state
    before the finish curtain at (-11000,7300,-1000) v=(0,3500,0), step);
    (b) stall-kill fires at stall_ticks +-2; (c) two scripted circles net
    |shaping| < 0.2; (d) RespawnBuffer margin rule (episode of 800 ticks,
    margin 300, snapshots at 100..800 -> exactly ticks 100..500 harvested).
    These gate S4 and any reward-adjacent refactor.

## Items, in order

**PROTOCOL (user-mandated): one improvement at a time.** Implement a single
item, benchmark it (0.2), run the ablation/learning-safety check, confirm
rewards/losses have NOT collapsed (ep_rew_mean and eval_progress in their
usual bands, losses finite and same order), record results, commit — only
then start the next item. Never stack two unmeasured changes.

**S1 — evals + checkpoints off the hot path** (expect 1.15-1.6x; hours)
- ALREADY MERGED (baseline includes it): `ckpt_latest` writes are
  time-based (>= 60 s between writes) instead of every iteration, and the
  default `--record-every` was raised 10e6 -> 25e6. What remains for S1 is
  only the recording SUBPROCESS below.
- Recordings: replace the inline `record_rollout` calls with a spawned
  subprocess (`tools/record_ckpt.py <fresh ckpt> --ep-ticks <ep_ticks>`,
  plus the `_stoch` variant), like the dashboard does. The eval csv columns
  (eval_fwd/path/speed, race/eval_progress) should then be filled lazily:
  when the expected traj file exists and is complete, compute stats next
  iteration (train_fast already has `episode_stats`/`race_progress`).
  Recording SEMANTICS must not change (same episodes count, same spawn
  pool, same ep cap) — only who waits for them.
- Verify: TIMING shows record≈0 and ckpt ~1/10th; eval columns still
  populate (allowed to lag one cycle); dashboard still lists recordings.

**S2 — one sync per update** (1.05-1.1x; hours)
- In the epochs loop, `float(vl)/float(pg)/float(el)` and the kl `float()`
  run per minibatch = 256 device syncs. Accumulate tensors
  (`kl_sum += (f_logp[idx]-logp).mean().detach()` etc.), call `.item()`
  once after the loop. csv values must match old behavior in expectation
  (they currently log the LAST minibatch's values — logging the mean is an
  acceptable, arguably better, change; note it in the commit).

**S3 — split obs buffer, image slice in bf16** (1.1-1.2x; ~1 day)
- `b_obs` is (T,N,8207) fp32 ≈ 8.6 GB. Split: scalars (T,N,15) fp32 +
  image (T,N,8192) bf16. Autocast already computes the conv in bf16, so
  numerics are unchanged in the forward; only storage/gather traffic drops.
  Touch: buffer allocs, `static_obs` (keep fusing into one fp32 row for the
  graphed rollout forward OR adapt `Policy.forward` to take two tensors —
  prefer adapting forward; the CUDA graph re-captures fine), minibatch
  gather + concat (or two-tensor forward) in the update.
- Verify: learning-safety check (0.2); measured update+copy gain.

**S6 — torch.compile the minibatch step** (1.15-1.3x; Linux only)
- Factor the minibatch forward+loss into a function of static-shaped
  tensors; `torch.compile(mode="reduce-overhead")` first,
  `max-autotune-no-cudagraphs` if stable. Keep `packer.pad` /
  `logprob_entropy_padded` inside the compiled region only if shapes stay
  static (they do: mb size is constant). Losses must match eager within
  bf16 noise over 5 iterations before trusting it.

**S7 — hierarchical SDF march** (1.3-1.8x in the current regime)
- vision.py: precompute a coarse grid, factor 8 (256u): coarse[c] =
  min(fine over the 8^3 block). For any point inside block c, fine sdf >=
  coarse[c], so marching on coarse values is conservative WITHIN the block;
  step by `max(coarse*0.9, coarse_min_step)` while `coarse > switch_thresh`
  (e.g. 2*256u), else fall through to the existing fine loop. Two-level
  loop inside `_march_kernel` (coarse steps capped ~16). Torch fallback
  path gets the same treatment.
- Verify (mandatory, this is the depth ABI): render 8 fixed poses
  (spawn area, mid-track, near glass at (-13536,3392,10688), sky-heavy
  view) with old and new kernels; max abs encoded-depth diff <= the
  encoding equivalent of one fine voxel (compute both on the same GPU).
  Then TIMING lidar delta + end-to-end.

**S4 — reward math on GPU / vectorized respawn** (only after S1-S3+S6 are
merged and stable; gate with the 0.3 semantics tests; see audit for
details). **S5/S8 (async overlap)** — out of scope for this pass; they
restructure the loop and deserve their own plan.

## Reporting

`docs/perf-results.md` table after every item + a final summary: achieved
end-to-end factor on the remote box, updated audit numbers, and which items
are worth back-porting expectations to the Windows box (all of them run
there too — nothing here is Linux-specific except S6's compile ergonomics).
