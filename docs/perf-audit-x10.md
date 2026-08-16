# RL_Surf x10 Throughput Audit (synthesized, final)

**Verdict up front.** No single villain. The C env (~1.5% of wall, 65x headroom) and the 1.96M-param network are NOT bottlenecks; the iteration is roughly 40% bandwidth-bound PPO update, ~20-25% GPU lidar, ~20-30% serialized single-thread Python glue, and 11-37% pure overhead (eval recordings + per-iteration checkpointing). x10 is achievable only as a stack: software fixes (~2.5-3x on this box) x Linux multi-GPU rental (~3-4x more) — renting alone buys <=2x (Amdahl: the Python/serial half doesn't scale with the GPU), software alone caps at ~3x.

**Measured anchors** (per-iteration wall recovered from consecutive cumulative-fps rows in runs/*/progress.csv; iteration = 786,432 ticks = 2048 envs x 128 decisions x K=3):
- race_A (cannonball, mixed spawns, default maxvel): **4.13 s/iter = ~190k ticks/s** marginal; ~172k incl. recording pauses (11-15% of wall). MEASURED.
- race_int (same + int_coef 0.25): 4.14 s/iter — **count-based intrinsic is throughput-free**. MEASURED.
- race_start2 (maxvel 4000, platform spawns, respawn OFF): 7.79 s/iter; race_respawn (same + respawn 0.9): 7.33 s/iter, live session ~8.6. **The current regime runs ~91-107k ticks/s — a 1.9x regression vs race_A that arrived with maxvel 4000 + platform spawns, NOT with respawn or intrinsic.** I verified all three run.json configs directly; the race_start2 control (respawn_frac 0.0, still 7.79s) refutes one sub-analysis's attribution of +3.3 s/iter to RespawnBuffer.observe. The reservoir is measured-free.
- Recording iterations cost +2.2s each on race_A but +7.4 to +48.6s on race_respawn (**31-37% of wall**) — long-surviving agents run eval to the full 12,000-tick cap, batch-1, twice per record point, no stall-kill in eval (record.py). MEASURED.
- In-repo anchors: ~25 ms/minibatch bf16 (train_fast.py:413-416 comment; cross-checked — 64 x 94ms fp32 = 6.0s matches eyes128_10B's measured 5.9 s/iter almost exactly); lidar ~6 ms/2048-env render (ski_2 bench — likely optimistic for cannonball); C env 13M ticks/s (tests/bench.c).

## 1. Per-iteration time budget (race_A regime, ~4.1 s; M=measured/anchored, E=estimate)

| Phase | s/iter | share | status |
|---|---|---|---|
| PPO update: 4 ep x 16 mb x ~25ms | ~1.6 | ~40% | M (in-repo bench + fp32 cross-check) |
| GPU lidar: 128 renders x ~6ms+ | 0.75-1.0 | 19-25% | E (ski_2 anchor; cannonball likely higher) |
| Graphed rollout forward+sample | ~0.15 | ~4% | E (FLOP count; CUDA-graphed) |
| C env: 384 core.step calls | ~0.06 | ~1.5% | M (bench-derived) |
| numpy reward (GoalField trilinear x384) + bookkeeping | 0.4-0.8 | 10-20% | E — profiler |
| Copies, per-decision torch.cuda.synchronize (train_fast.py:931), GAE, eager truncation-bootstrap renders | 0.6-1.0 | 15-25% | E — residual, profiler |
| save_ckpt("latest") EVERY iteration (line 1086, ~24-35MB) + csv | 0.1-0.3 | 3-7% | E |
| Eval/recording, amortized on top | +0.6 (race_A) / +2.7 (current regime) | 11-37% of wall | M |

Current regime adds +3.2-3.7 s/iter, unexplained. Best hypothesis: maxvel-4000 launches put agents high over open air, sky rays never hit and march all 64 steps (lidar ~2x), plus 40u/tick BSP traces. **This is profiler question #1 — worth up to 1.9x by itself.**

Per-decision structure (~17-18ms): fully serial ping-pong — GPU phase (graph replay ~1.2ms + lidar ~6ms + 67MB D2D copies), hard sync, then CPU phase (3 x [core.step 0.16ms + numpy reward 0.5-1ms + respawn/bookkeeping]). Sum, never max: GPU idles ~8-9ms and CPU ~7.5ms per decision. The 32-thread CPU is ~97% idle; the saturated resource is ONE Python thread.

## 2. Top bottlenecks, with evidence

1. **PPO update (~40%), bandwidth-bound not compute-bound.** 0.74 TFLOP/mb at 25ms = ~29 TFLOPS = ~14% of 5090 bf16 peak; ~20GB DRAM traffic/mb (538MB fp32 b_obs gather, eager-ReLU activation re-reads, backward) = ~12ms pure DRAM floor. Plus hidden serializers: float(kl)/float(vl) per minibatch = **256 device-wide syncs per update** (train_fast.py:1036-1038), and 13k+ small launches/iter on Windows WDDM.
2. **Serialized rollout host phase (~20-30%).** One torch.cuda.synchronize per decision; reward/respawn/bookkeeping in numpy on the critical path 3x per decision, though rewards never feed the next action — the true serial chain is only fwd(1.2) -> env(0.5) -> lidar(6) ≈ 7.7ms/decision.
3. **GPU lidar (~19-25% on race_A, likely dominant in current regime).** 16.8M rays x <=64 scattered f16 gathers over a 1.34GB SDF — gather-bandwidth-bound; sky rays pay worst case.
4. **Pure overhead (11-37% of wall, measured).** Batch-1 eval recordings without stall-kill; per-iteration torch.save.
5. **Non-bottlenecks:** C env; network parameter count (cost driver is the 8,192-pixel depth image as DATA); CPU core count.

## 3. Ranked interventions (expected end-to-end factor x cost)

**Software (do in this order):**
- **S1. Evals/checkpoints off the hot path** — recordings in a subprocess or batched into the fleet, stall-kill in eval, ckpt every ~10 iters. **1.15x race_A / 1.45-1.6x current regime. Hours.**
- **S2. Kill the 256 per-minibatch float() syncs** — accumulate stats on-device, one sync per update. **1.05-1.1x. Hours.** Prerequisite for S6.
- **S3. Compact b_obs: split buffer, image slice bf16 (or uint8 — depth is voxel-quantized).** Precision-free: autocast already feeds the conv bf16 in both rollout and update. 8.6GB -> 4.3/2.15GB; halves-to-quarters the update's dominant gather traffic and the 67MB/decision D2D copy; frees 4-6GB VRAM. **1.1-1.2x. 1 day.**
- **S4. Reward on GPU + vectorized respawn + fused K-step.** GoalField trilinear is a native GPU gather (currently ~115 numpy ops/sub-tick, goalfield.py:67-93); shaping telescopes across K sub-ticks; vectorize RespawnBuffer.observe (numpy ring buffer, not per-env lists); merge K=3 sub-steps into one C call returning per-sub-tick done masks. **1.1-1.2x + prerequisite for S5. 1-2 days.**
- **S5. Overlap rollout phases** — CUDA events instead of the per-decision device sync, reward/bookkeeping in a worker thread under the GPU phase, batch truncation bootstraps (one deferred render/forward per decision, not eager per-sub-tick). Rollout 2.3s -> ~1.1-1.3s. **1.3-1.45x. 3-5 days.**
- **S6. torch.compile (or full CUDA-graph capture) of the static-shape minibatch step** — fuses ReLU into conv epilogues, kills eager activation traffic and launch overhead. Update 1.6 -> ~0.9-1.2s. **1.15-1.3x. 2-3 days** (triton already works on this box; expect some Windows inductor friction; graph capture is the fallback).
- **S7. Hierarchical SDF march** in vision.py (coarse 128-256u grid to skip open space, fine near geometry). 2-4x on lidar in cannonball's open-air regime; **1.3-1.8x overall in the current regime** — probably erases most of the maxvel-4000 regression. Pixel-verifiable against the current kernel. **2-3 days.**
- **S8. Stale-by-one async PPO** — overlap update with next rollout (Sample Factory pattern; clipped IS absorbs 1-iteration lag). Iteration becomes max(rollout, update) instead of sum. **~1.4-1.6x after S1-S7. 3-5 days.**
- **Science-lever ablations (real speedups, NOT free — run as A/Bs):** lidar 64x32 (4x fewer rays AND 4x smaller conv input; AdaptiveAvgPool2d makes a cross-resolution warm start valid; ~1.8-2.5x), K=3->4 or 6 (+33-100% ticks/decision at coarser control), stride-4 patchify first conv (update 2-2.5x), 2 epochs (linear wall win, pays in samples — repo already measured this regression).
- **Anti-recommendations:** more envs (repo's own ablation: 8192 envs ~doubled samples-to-skill; the tax is on batch WIDTH, not on wall-clock rate); FP8; porting physics to GPU (bit-exact pm_tick parity is the project's soul; env is 1.5% of wall); bigger single GPU on unmodified code.

**Hardware / rental (after, or alongside, S1-S7):**
- **H1. Linux, whatever you rent** — removes WDDM launch/sync overhead, native inductor; build.sh already produces libsurfcore.so. 1.1-1.4x on launch-bound parts, effectively free. Needs A/B.
- **H2. The pick: one Linux node, 4x 5090/4090-class (~$0.5-1/GPU-hr) or 2x H100/H200.** Two valid topologies: (a) DDP splitting the SAME global batch (512 envs/GPU x 4 — identical learning dynamics, no width tax; 8MB grads so PCIe suffices, no NVLink needed); (b) learner/actor split (GPU0 rollout+lidar, GPU1 update+evals). ~8+ high-clock cores per GPU; core count beyond that is wasted. **2.5-4x on top of software work.**
- **H3. Single H200/B200: only after software work.** HBM bandwidth (4.8-8 TB/s vs 1.79) gives 2-4x on the update+lidar terms once they dominate; on current code <=1.3-1.5x. Worst ROI today.
- **H4. Zero-code alternative for extra GPUs:** independent seeds/hyperparameter population — immune to the width tax, often more research/dollar.

## 4. What x10 realistically requires

From the race_A baseline (190k): S1-S7 on the 5090 => ~2.5-3x (~500-600k ticks/s, unchanged PPO semantics). Plus H2 (Linux multi-GPU, DDP-same-batch or learner/actor) => **~1.5-2M ticks/s = x8-12**. One modest science concession (K=6, 64x32 lidar, or patchify) buys the margin if the hardware falls short. From the CURRENT regime (91-107k), S1+S7 alone recover ~2x before anything else. **First action either way: a 10-line phase timer** (perf_counter + torch.cuda.Event around update / lidar / forward / env / reward, printed per iteration) — it settles the four open estimates: (a) true 25ms/mb decomposition on this card, (b) actual cannonball lidar cost and the sky-ray hypothesis, (c) the 0.6-1.0s rollout glue residual, (d) the unexplained 1.9x maxvel-4000 regression. No live benchmarks were run for this audit — the 5090 is currently 97-98% occupied by an active training run; everything above is log-derived wall time, in-repo measurements, or counted-ops estimates, flagged accordingly.

## 5. Intrinsic-reward config for the reservoir+intrinsic run

**Recommendation: `--int-coef 0.2`** (defensible window 0.15-0.25), `--fail-pen 0`, respawn settings as-is. Rationale: shaping income is ~0.025/tick; a full-speed dive crosses a fresh 256u count-cell every ~8 ticks, so first-visit income ~coef/8 per tick — 0.2 makes untouched frontier pay ~breakeven with racing income as a decaying premium on top of shaping. race_int ran 0.25 stable (KL ~0.03) at zero throughput cost (measured), but respawn_frac 0.9 concentrates 90% of the fleet at the frontier, so per-episode intrinsic income runs ~2-3x higher — hence 0.2, not 0.25. Code-verified already-correct (do not "fix"): ended-tick masking prevents respawn relocation paying novelty (rewards.py ~542); counts + reservoir are checkpointed and an explicit `--int-coef 0.2` overrides the ckpt's stored 0. Watch tonight: `race/int_per_ep` should sit at 10-30% of |ep_rew_mean| (halve coef if >50% for hours); if the frontier stalls with win%=0 and flat reservoir growth, lower `--respawn-margin` to ~7s before touching the coef; judge progress by `race/eval_progress` + reservoir growth, not win% (eval spawns are unboosted). Do NOT reset the counts table mid-run; skip NoisyNets. Best cheap follow-up coupling the two systems: novelty-weighted reservoir sampling — weight stored states by 1/sqrt(counts[cell]) using the existing intrinsic table (~15 lines in respawn.py; Go-Explore's cell-prioritized archive, directly counters frontier dilution).

**Key files:** C:\RL_Surf\python\train_fast.py (rollout 922-1000, sync 931, update 1002-1038, float() syncs 1036-1038, per-iter ckpt 1086, envs-ablation note 267-270), C:\RL_Surf\python\surfgym\vision.py (triton march 41-85), C:\RL_Surf\python\surfgym\goalfield.py (67-93), C:\RL_Surf\python\surfgym\rewards.py (RaceReward ~404-600), C:\RL_Surf\python\surfgym\respawn.py (observe 57-85), C:\RL_Surf\src\env.c (498-613), C:\RL_Surf\runs\{race_A,race_int,race_start2,race_respawn}\{run.json,progress.csv}, C:\RL_Surf\python\surfgym\record.py.