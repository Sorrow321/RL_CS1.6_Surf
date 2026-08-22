# Research results ledger

Running record for `docs/research-plan.md`. One row per arm, written when
the arm finishes (harvest before dropping any box — `tools/harvest_box.sh`).

## Fixed reference

- **F' baseline**: `runs/frozen/F_prime.pt` = race_respawn step
  **6,787,694,592**, md5 `5f08b5da3b89f421a853bb94c4c59222`. Every arm
  resumes from it with exactly one change.
- **Step 0 (conveyor fix) landed** (e4ef6f5): 18,201 voxels freed, all
  inside the 12 conveyor AABBs, 0 newly solid; thin panes unchanged at
  104; pillar `*223` center SDF 0 -> 128u. **d0: 198,379.83 ->
  198,379.84 (ratio 1.000000)** — shaping scale unchanged, so pre-fix
  curves remain directly comparable and no value-head rescale was needed.
- **Metric noise floor** (measured on race_respawn, 3-ep eval points):
  per-point std ~13.7k (CV ~17%); consecutive last-hour medians drift
  77k-92k inside an unchanged run. Arms therefore run 9-episode
  greedy-only evals (`--eval-eps 9 --eval-greedy-only --record-every
  150e6`).
- **Known property**: the goal field is blind to kill volumes — the
  shaping gradient points THROUGH fail-net `*30` at wall #2 (~105k free
  voxels inside it carry finite descending d). Arm S2 tests fixing this.
  Read near-wall verdicts with it in mind.
- Frontier at F' (last 1e9 steps of race_respawn): median ~91.5k, mean
  ~82k, all-time max 99,004 (once, at step 5.50e9) of 198,380.

## Screening protocol (user-set, 2026-08-16)

Short screens first: **+2.0e9 steps from F'** (`--steps 8.79e9`), ~2h on
a rented 5090, ~3h local. Drop an arm only when it is CLEARLY inferior
(non-overlapping eval_progress trend at equal steps); similar arms go to
longer ablations later. Screen metric: median of the distinct
`race/eval_progress` record points over the arm's final 500e6 steps
(~4 points x 9 episodes), plus max and any finish.

Launch template (remote; local drops the OMP prefix and uses the
Windows paths):

    CUDA_VISIBLE_DEVICES=0 python3 -u python/train_fast.py \
      --ckpt runs_ckpt.pt --run <arm> --steps 8.79e9 \
      --ckpt-every 500e6 --record-every 150e6 \
      --eval-eps 9 --eval-greedy-only <arm flags>

## Arms

| arm | box (ticks/s) | seed | steps | eval med-last-500e6 (max) | success | finish_s | verdict vs W0 | notes |
|---|---|---|---|---|---|---|---|---|
| W0 | box1 9950X/5090 (292.8k) | - | +2.0e9 done | **85,988 (91,836)** | 0 | - | control | post-fix continuation; NO destabilization from the conveyor fix; did not crack wall #2 on its own |
| A1 | local 5090 Win (**600k**) | - | +2.003e9 done | 87,502 (91,175) | 0 | - | equal-steps: behind (transient); wall-clock: ~2.1x ahead | 64x32 render. First eval 2,941 (vision reset), recovered to the F' frontier inside 2.0e9 steps |
| W0' (A1 ext) | local 5090 Win (619k) | - | +2.0e9 done | **84,884 (91,942)** | 0 | - | ties W0 (85.0k vs 86.0k, maxes equal) | the 64x32 control over 8.79->10.79e9; **VERDICT: 64x32 ADOPTED** — tie on learning at 2.1x throughput. All subsequent arms: from F2, judged vs this curve at equal steps |

Equal-steps A1 vs W0 (the transient dominates A1's early budget):

    step        A1      W0
    6.94e9   18,685   75,035
    7.24e9   42,165   87,681
    7.69e9   76,684   87,508
    8.74e9   91,175   (pending)

**F2 baseline (64x32)**: `runs/frozen/F2_lidar64.pt` = A1 endpoint, step
8,790,736,896, md5 `20d960d2e568a074d3a099a805d1c8f9`. If A1's extension
(= W0', same run dir) holds or beats W0's plateau, all later arms run at
64x32 from F2 against W0' — at ~600k steps/s a +2e9 screen is ~1h.

## Fleet + in-flight assignments (2026-08-16 evening)

Warm-arm template (F2 base, judged vs W0' over 8.79->10.79e9):
`(nohup python3 -u python/train_fast.py --ckpt runs_ckpt.pt --run <arm>
--steps 10.79e9 --ckpt-every 500e6 --record-every 150e6 --eval-eps 9
--eval-greedy-only <delta> > runs/<arm>_log.txt 2>&1 &)`
(on box1 the F2 file is named F2_lidar64.pt; on boxes A-D deploy_box
renamed it runs_ckpt.pt, md5-gated 20d960d2.)

| slot | ssh | arm |
|---|---|---|
| box1 | -p 28522 root@87.227.133.155 | P1 `--lr 1.5e-4` (running; lr reset 3e-4 -> 1.5e-4 confirmed in log) |
| boxA | -p 50085 root@149.200.47.167 | P2 `--respawn-margin 4` (running; "harvested >= 4s" confirmed) |
| boxE | -p 28546 root@87.227.133.155 | R3 `--gamma 0.999` (running; replaced dead boxB) |
| boxG | -p 15679 root@89.221.67.167 | R1 `--speed-coef 0.002` (running; 96% FLOPS healthy; replaced dead boxC/F) |
| dead | 213.96.60.239, 79.160.189.79, 213.224.31.105 | rejected: no-CUDA driver x2, capped card x1 — filter listings for CUDA 12.8+/driver 570+ |
| boxD | -p 28055 root@178.41.236.30 | R2 `--int-coef 0.1` (running; cold count table — discard evals before +200e6) |
| local | C:\RL_Surf | B0_scratch (running at 637k steps/s; milestones judge) |

Dashboards tunneled locally: 8080 = box1, 8081 = boxA, 8082 = boxD
(local box's own dashboard = 8000). gpu_health BUSY note on boxD was the
deploy's own tail releasing the card; verified 0 compute apps afterward.

## PIVOT (user, 2026-08-16 ~20:45): the plateau-escape campaign

B0_scratch climbed 1.2k -> 16.8k in 600M steps, then sat FLAT at 15-17k
for 600M more — the core failure of this task in miniature (an early jump
it never lands), reachable in ~25 min of training. All warm screens were
STOPPED (partials below) and the fleet now attacks this plateau with
short arms: turnaround ~15 min, so treat it like a hyperparameter search.

**F3 baseline**: `runs/frozen/F3_plateau.pt` = B0_scratch step
1,000,341,504, md5 `1f3ec80be450556fb78079b329497f2b`, 64x32, 20k-state
reservoir. Plateau median ~16.1k over its last 400M steps.

**Attack protocol**: resume F3, `--steps 1.5e9` (+500M, arms self-exit),
`--record-every 50e6 --eval-eps 9 --eval-greedy-only` (10 eval points).
ESCAPE = any eval > 25k (plateau + ~55%); LEAN = median of last 3 evals
vs the 16.1k plateau and vs C0's matching window. Escapers get an
immediate +1e9 extension to confirm the breakout is stable.

**REMAKE (user, ~21:40): the campaign is a SCRATCH RACE.** Arms start
from step 0 with one idea active each (B0's exact recipe + one delta,
`--steps 1.5e9`, `--record-every 50e6 --eval-eps 9 --eval-greedy-only`)
and are judged on whether their curve stalls at the ~16k plateau or
climbs through: milestones steps-to-20k / steps-to-25k, eval at 1.5B.
B0's own curve (16.8k by 600M, flat after) is the historical reference;
sC0 is the live control at the new eval cadence.

**Resume-round results (completed first, +500M from F3 — a different
question: "unstick the stuck policy"):**

| arm | evals over +500M | verdict |
|---|---|---|
| pC0 control | 13.8-16.9k flat | plateau does NOT break on its own |
| **pI1 `--int-coef 0.25`** | **16.9k -> 18.9 -> 20.8 -> 27.5 -> 29.8 -> 30.9k** | **ESCAPED — the first confirmed unsticking mechanism** (ckpt_final preserved on boxD) |
| pE1 `--ent 0.02` | 12.7-16.9k flat | no |
| pG1 `--gamma 0.999` | 15.6-17.5k flat | no |
| pV1 `--respawn-speed 1-2` | 15.4-17.0k flat (stopped at +342M) | no |
| pM1 `--respawn-margin 4` | 14.4-17.0k flat (stopped at +178M) | no |

Note: novelty 0.25 degraded the MATURE policy months-of-training in
(race_int history) but unsticks the YOUNG one — coefficient is
policy-age-dependent.

**Scratch race round 1 — RESULTS (1.5e9 steps each):**

| arm | curve | steps-to-20k / 25k | eval@1.5e9 | verdict |
|---|---|---|---|---|
| sC0 control | brief 15-18k wobble ~500-650M, then up | ~650M / ~700M | 47,083 | **the 16k trap is STOCHASTIC** — B0 fell in, the identical control escaped by luck |
| **sG1 `--gamma 0.999`** | fastest riser, no trap | **~350M / ~380M** | 48,774 (flat at 47-48k since ~550M) | **WINNER on speed — 3x faster to the 47k wall; promoted into the base recipe** |
| sI1 `--int-coef 0.25` | through, mildly ahead of control mid-run | ~550M / ~600M | 47,512 | mild positive from scratch (its resume-round escape stands) |
| sE1 `--ent 0.02` | slow, shelf at 20-25k | ~500M / ~1.05B | 25,590 | NEGATIVE — extra entropy slows learning |
| sV1 `--respawn-speed 1-2` | slowest riser | ~950M / — | 22.6k @1.2B (running) | negative early — overdriven spawns dilute the young policy |
| sB128 128x64 (local) | IN the 15k trap since ~450M | — | 15.1k @687M (running) | resolution does NOT rescue the trap so far; full window pending |

**The real common wall: ~47-48k = wall #1 (stage-5 fail-net jump,
trigger *20).** Every escaper converges there; sG1 has sat on it for
900M steps. Historically cracked at 1.15B steps with boosted respawns
on the warm lineage. This is now the campaign target.

**Round 2 (in flight) — attack the 47k wall, gamma 0.999 as base:**

| slot | arm | question |
|---|---|---|
| boxE | sG1 extension (to 3.0e9) | does gamma-base self-escape 47k? (control) |
| box1 | sGI_g999int025 | gamma + novelty: explorer at the wall |
| boxA | sGB_g999boost | gamma + spawn-boost 2x: the historical wall-1 recipe |
| boxD | sG3_gamma9995 | gamma dose-response: is more horizon better? |
| boxG | (sV1 finishing) -> sGK_g999killaware | gamma + S2 masked field: stop rewarding flight INTO the fail net — gated on the CPU-only review rerun |
| local | ~~sB128_res~~ KILLED at 862M (user) -> s32_g999 | resolution ladder DOWN: 32x16 + gamma base vs sG1's curve — find where "reducing = good" breaks |

sB128 final note: it DID start escaping the trap at its end (15.3k ->
23.2k over 800-862M) — but needed ~800M steps where gamma-base needed
~350M, at HALF the throughput: loser on steps AND wall-clock. Verdict
stands. Resolution ladder so far: 128x64 slow, 64x32 good, 32x16 now
testing (note: at 32x16 the conv trunk's last stage is 2x4, smaller than
the 4x8 adaptive-pool target — the pool upsamples; increasingly
degenerate below this, which is part of finding the wall).

**Superseded round-1 table (resume-based, kept for the record):**

| slot | arm | delta |
|---|---|---|
| box1 | pC0_control | none (does it escape on its own?) |
| boxA | pE1_ent02 | `--ent 0.02` (4x entropy) |
| boxD | pI1_int025 | `--int-coef 0.25` (novelty; young policy, empty table = whole world novel is FINE here) |
| boxE | pG1_gamma999 | `--gamma 0.999` (the jump's payoff is seconds away) |
| boxG | pV1_boost2 | `--respawn-speed 1.0 2.0` (practice the jump at make-it speed — the wall-1 crack recipe) |
| local | pM1_margin4 | `--respawn-margin 4` (respawn nearer the death point) |

**Round 2 — resolution fairness first (user, 2026-08-16 ~21:00)**:
- pB1_scratch128 (PRIORITY, occupies one box ~85 min): scratch run
  identical to B0 but `--lidar-w 128 --lidar-h 64`, `--steps 1.5e9` —
  full launch line = B0's (ledger above) with the lidar dims swapped.
  Judge its 0->1.5B eval curve against B0's (1.2k -> 16.8k -> flat 16k):
  same plateau = resolution exonerated for THIS plateau; blows past =
  the 64x32 adoption hurt perception here and gets revisited.
- pR128: resume F3 at `--lidar-w 128 --lidar-h 64` (+500M standard
  attack). The switch transient biases this arm AGAINST winning (A1
  measured the reverse switch), so: escape = strong evidence, flat =
  inconclusive -> extend, never conclude from the null.

**Other round 2 candidates**: S1 `--respawn-binned 1` and S2
`--race-kill-aware 1` (post-review; before spending S2, check whether a
kill volume actually sits at the ~16k frontier — zones.kill_zones + the
death cluster), `--ent 0.05`, `--int-coef 0.5`, `--lr 6e-4`, combos of
round-1 winners. If ALL flag arms fail: the user's perception hypothesis
("it cannot see the next ramp") is next — A2 ramp-visibility channel
gets implemented.

**Stopped warm screens (partial, ~60-80% of budget, none escaped wall #2):**
P1 lr1.5e-4 last evals 92,809/92,177/87,876 (tail ABOVE the ~85k control
band — weak positive, revisit after the campaign); P2 76-92k, R2 58-91k,
R3 81-91k (no signal); R1 1 eval (too early).

## Queue (after this rotation)

B1-scratch-bignet (`--emb 1024 --hidden 896`, scratch, pair vs B0),
then wave 2: S1 (binned respawn sampling), S2 (kill-aware goal field) —
Fable implementing. Second seeds for any winner before promotion.

Local: **B0-scratch** (user-requested 2026-08-16): fresh net, 64x32,
fixed map, current recipe — the incumbent trained 6.8B steps against the
phantom pillars and the into-the-fail-net gradient, so its plateau may
be a local optimum of the OLD map; at 600k steps/s the whole historical
step count replays in ~3h. True scratch is untested (the race lineage
inherited the eyes_speed locomotion prior), so the first milestones ARE
the result. Judge on steps-to-10k/47k/92k, not vs W0'. Launch:
`python -u python/train_fast.py --map maps/surf_src_cannonball.bsp
--run B0_scratch --reward race --spawn platform --respawn-frac 0.9
--respawn-speed 1.0 1.5 --maxvel 4000 --lidar-w 64 --lidar-h 32
--lidar-range 11500 --lidar-near 2000 --act-every 3 --pitch-rate 1.33
--steps 6e9 --ckpt-every 500e6 --record-every 150e6 --eval-eps 9
--eval-greedy-only`

## Capacity bench (local 5090, tools/bench_capacity.py, 2026-08-16)

Update-side cost only; a lidar-resolution variant also changes the render,
so its true end-to-end is better (64x32) than shown for the update alone.

| variant | params | ms/mb | x base | end-to-end |
|---|---|---|---|---|
| base | 1.96M | 35.84 | 1.00 | 1.000x |
| hidden896 | 3.65M | 37.03 | 1.03 | 0.973x |
| emb1024+h896 | 5.61M | 37.15 | 1.04 | 0.971x |
| tower+1 | 2.36M | 36.32 | 1.01 | 0.989x |
| conv2x | 3.08M | 60.28 | 1.68 | 0.641x |
| conv4x | 5.45M | 112.23 | 3.13 | 0.364x |
| stack2 | 1.96M | 39.40 | 1.10 | 0.924x |
| stack4 | 1.96M | 47.57 | 1.33 | 0.788x |
| stack8 | 1.96M | 48.58 | 1.36 | 0.774x |
| **lidar64x32** | 1.96M | **11.04** | **0.31** | **2.315x** |
| lidar192x96 | 1.96M | 80.72 | 2.25 | 0.493x |

## Resolution ladder (local probes, gamma base, scratch 1.5e9)

| render | pixels | steps-to-25k | steps-to-47k | eval@1.5e9 | verdict |
|---|---|---|---|---|---|
| 128x64 (sB128, no gamma) | 8192 | not reached by 862M | - | killed | slow + trap-prone |
| 64x32 (sG1) | 2048 | ~380M | ~550M | 48,774 | the adopted base |
| **32x16 (s32_g999)** | 512 | **~360M** | **~504M** | 48,528 | **NO degradation - matches 64x32 exactly** |
| **16x8 (s16_g999)** | 128 | ~500M (25k ceiling) | NEVER | 25,101 | **THE FLOOR: escapes the 16k trap but shelves at ~25k for 1B steps - cannot resolve the ~25k stage. Optimal band: 32x16 to 64x32; 32x16 recommended for the next phase (2x cheaper, equal learning)** |

The perception bar for this map is far lower than assumed: 512 pixels
suffice to reach and fight the 47k wall at full speed.

## Round 2 results + the gamma-horizon breakthrough (2026-08-16 ~23:30)

| arm | evals | verdict |
|---|---|---|
| **sG3 `--gamma 0.9995`** | 47k@500M -> 55.6 -> 67.9 -> 80.0 -> **81.8k@1.15B**, holds ~82k | **THROUGH wall #1 on horizon alone; near the 6.8B-warm-lineage frontier (~91.5k) from scratch in 35 min. EXTENDED to 3.0e9** |
| sG1 ext `--gamma 0.999` (to 3.0e9) | pinned 43-48k the whole extension | 0.999 alone never escapes the wall |
| sGI g999+int025 | reached wall, oscillates 40-48k | novelty does not unstick the wall |
| sGB g999+boost2 | stuck 22-26k | boost hurts again |
| sI2 g999+int01 | running (~1.3B, int 0.12/ep) | pending |

Value horizon is THE lever: 0.995 (2s) trap-prone, 0.999 (10s) fast to
the wall but stuck, 0.9995 (20s) through the wall. Dose-response
continues upward (sG4 0.9999 = 100s) - watch for value-learning
instability at the extreme.

**Round 3 (in flight):** boxD sG3 extension (to 3.0e9 - does it reach
wall #2 at ~92k?), boxE sG3b_rep (replication seed - the 16k trap taught
us luck matters), box1 sG4_gamma9999 (dose up), boxA s32G3_g9995 (32x16 +
0.9995 = the champion-recipe candidate at double throughput), boxG sI2
finishing, local sGK g999+kill-aware (hull-masked 813k voxels = ~1% of
free space; bake ~25 min then trains).

User decision: 32x16 adopted going forward, 64x32 kept in reserve for
perception-heavier situations.

## Curiosity escalation ladder (user, 2026-08-16 ~23:50)

Position-only counts are blind to gaze: the same voxel looking left vs
right is a different CNN observation. Ladder, in order of cost:
1. **--int-view K** (IMPLEMENTED, 40 tests green): count key = cell x
   yaw sector (K=8 -> 45-degree sectors, count table x8, ~42 MB in the
   ckpt at 256u). Arm-ready: base + `--gamma 0.9995 --int-coef 0.25
   --int-view 8` at the next free slot.
2. Position x speed-bucket key (same machinery) if approach SPEED, not
   angle, is the blind spot.
3. **RND** (queued, not implemented): predictor-vs-frozen-random-target
   on the full observation (depth + scalars); novelty = predictor loss.
   Reach for it only if the count-key extensions fail at a wall — it is
   a real network in the compiled update path.

## Round 3 results (2026-08-17 ~00:20)

| arm | result | verdict |
|---|---|---|
| **sG3 ext (0.9995, 64x32)** | **94,394 / 94,544 / 94,767 at ~2.9e9** | **PAST wall #2's level — above the 6.8B warm lineage's median (91.5k), nearing its all-time max (99k). Extended to 4.5e9** |
| sG3b_rep (0.9995 seed 2) | wall at ~650M (same speed as sG3), SITS at 47-48.7k through 1.5e9 | wall-approach speed REPLICATES; the wall-break has a luck component. Extended to 3.0e9 |
| sG4 (0.9999) | slower early (12k@450M), wall ~800M, stuck | dose peak is at ~0.9995; 100s horizon hurts early value learning |
| s32G3 (32x16 + 0.9995) | wall only at ~1.3e9, no break; box avg 1,045k steps/s | resolution equivalence does NOT carry to the long-horizon recipe — champion stays 64x32. (User: keep 64x32; 32x16 reserve for cheap screens) |
| sI2 (g999 + int 0.1) | 42k at end, late to the wall | below sG3 pace |
| sGK (g999 + kill-aware, local) | training after clean hull bake (passed reachability; masked d0 vs standard pending its curve) | pending |

In flight: sG3 ext2 (boxD, to 4.5e9 — the frontier run), sG3b ext (boxE,
to 3.0e9 — does seed 2 break with more time?), sG5 gamma 0.99975 (boxG,
dose midpoint), sGV g9995+int025+view8 (box1, the view-novelty arm,
launches after sG4 exits + git pull), sGK local.

## Round 4 notes (2026-08-17 ~00:50)

- sG3 frontier: 94,946 / 94,807 latest — record (99,004) not yet passed,
  no finishes. ext2 to 4.5e9 running (the boxD extension sat idle 32 min
  behind a self-deadlocked watcher: until-pgrep waiters that embed the
  launch line match THEMSELVES - rotations now happen only from the
  wakeup loop; memory note written).
- sG3b (seed 2) ext: still pinned 43.5-48.5k at 2.98e9. Wall-break
  variance is real: 1 of 2 seeds through by 3e9.
- sGK (g999 + kill-aware) done: 25k@570M, wall@810M, sits 47-48k -
  slightly SLOWER than plain g999 and no break. Wall #1's blocker is the
  jump skill, not the into-net gradient; kill-aware's real test is wall
  #2 via sGA (0.9995 + kill-aware, boxA, baking).
- Rotations: box1 sGV (view-novelty) running on new code; boxA sGA
  running; local -> sGS (0.9995 + S1 binned respawns).

## Round 5: RECORD SMASHED + wall #3 found (2026-08-17 ~01:20)

- **sG3 ext2 done: max eval 180,874 = 91% of the track** (old all-time
  record 99,004 obliterated; wall #2 passed inside the extension). No
  finish yet. A 30,000-tick-cap recording still tops out ~180.0k =>
  NOT clock-bound: **real wall #3 at ~180k**, 18.4k units from the
  finish. ext3 to 6.0e9 running.
- **sGV (0.9995 + int 0.25 + view 8): 66,928@1.5e9 and climbing steeply**
  (49.9 -> 63.5 -> 66.9k) — broke the 47k wall faster than sG3's own
  seed did (sG3 was 47.5k at this step). View-novelty leans REAL.
  Extended to 3.0e9; replication sGVb launched (boxE).
- sG5 (0.99975): shelved at ~25.2k — dose response is non-monotonic /
  seed-noisy. Dropped.
- sGA (0.9995 + kill-aware scratch): shelved at ~25.1k — same shelf.
  Note: the ~25k shelf now seen 3x (16x8, sG5, sGA) = a real stage
  barrier some seeds camp at. Kill-aware retargeted at wall #3 instead:
  **sG3K** = frontier ckpt + kill-aware (boxA, masked d0 198,416 vs
  198,380) — if wall #3 is another into-the-fail-net gradient, this is
  the surgical fix.
- boxG: sG3c (0.9995 seed 3) for the wall-break base rate.
- Ops: monitoring cadence tightened to ~15 min (user) — arms finish in
  ~30 min at 700k+/s and the old 30-35 min cadence left GPUs idle.

## Wall #3 forensics + rotations (2026-08-17 ~01:45)

- **Wall #3 = trigger_teleport *46 -> mapstart**: a 2u-thin horizontal
  fail net at z=-5918 spanning the whole late section (x -6656..4096,
  y -9472..8128). All 3 long-cap episodes die falling onto it at
  (~900, 5150, -5875) after ~66s of clean racing — a hard transfer jump
  over a catch net, same structure as wall #1. NOT a parity bug.
  Implication: kill-aware likely won't crack it (floor-net masking
  barely moves the gradient above); training time + approach diversity
  are the proven levers.
- sGS (0.9995 + S1 binned respawns, scratch): 13-19k, never reached the
  wall — **binned sampling HURTS from scratch** (flat curriculum before
  competence). S1 remains a mature-frontier tool only. Dropped.
- Local slot -> **sG3V**: frontier ckpt + int 0.25 + view 8 — approach-
  diversity at the wall-3 jump (cold-table windfall: discard evals
  before +200e6).

## Round 6 (2026-08-17 ~02:10)

- sG3 frontier @6.0e9: max 180,874, still camped at wall #3 (sr 0).
  Extended to 7.5e9.
- **sG3K (kill-aware on frontier) @6.0e9: max 180,520 — NULL differential
  vs plain sG3 over the same window. Kill-aware DROPPED** (wall #1 null,
  wall #3 null: both walls are jump-skill, not gradient artifacts; the
  mechanism stays on the shelf for maps where the gradient provably
  misleads).
- **sGV (view-novelty) @3.0e9: 98,858 — equal to the old warm all-time
  record from scratch**, steady 98.3-98.9k band = it PASSED wall #2's
  zone and is mid-late track. Extended to 4.5e9.
- boxA -> sG3V_b: 2nd seed of frontier + view-novelty (the wall-3
  approach-diversity bet, now the main hypothesis).

## Round 7: WALL #3 BROKEN by view-novelty (2026-08-17 ~02:40)

- **sG3V (frontier + int 0.25 + view 8, local): evals 195,356 / 194,418 /
  193,879 of 198,380 — ~3k units (~2 s) from the FINISH.** Wall #3
  broken by approach diversity; the user's gaze-aware curiosity was the
  mechanism that did it.
- Seed base rates now on record: plain 0.9995 = seed1 broke 47k @650M,
  seed2 stuck 47k through 3e9, seed3 stuck at the 25k shelf — huge
  variance. Scratch view-novelty rep (sGVb): at the 47k wall @1.5e9
  (seed 1 sGV broke at ~1.35e9 and is at ~99k).
- Rotations: sG3V's 5.5e9 ckpt shipped to boxE ("sG3V_long",
  --ep-ticks 18000 — insurance against the 120 s episode cap clipping
  the last meters) and boxG ("sG3V_c", plain continuation seed). sG3c
  (25k shelf) killed to make room. Fleet = 4 finisher-lineage runs +
  sGV ext + sG3 plain frontier.
- Ops: two staleness incidents this hour were finished-but-unnoticed
  replications (my sweep skipped boxE/boxG in round 6); ping now sweeps
  ALL six slots every cycle, no exceptions. Also: pkill -f patterns must
  be bracket-quoted or they kill their own ssh wrapper.

## Concurrency test: 2 trainers / 1 GPU (boxD, 16c, 64x32; 2026-08-17 ~03:10)

| config | rates (steps/s) | aggregate |
|---|---|---|
| solo | 664,098 | 1.00x |
| two concurrent | 351,272 + 448,266 | **799,538 = 1.20x** |

GPU pegged at 100% util under co-tenancy, VRAM 10.6 GB total (fits
easily). Verdict: +20% aggregate for -40%/-33% per-run speed. **Use for
seed-count-hungry screening (base rates), never for frontier/finisher
runs** where per-seed wall-clock latency matters. (Caveats: one box, the
younger dummy runs cheaper episodes; expect the split to worsen slightly
as both mature.)

Also: sG3V @6.0e9 saturated at ~195.2k (max 195,356), sr 0 — a death
cluster 3k from the finish; forensics on the next rotation. Extended to
7.5e9 locally.

## NIGHT PLAN (user, 2026-08-17 ~03:40, ~9h autonomous)

User diagnosis of the final obstacle: the last two ramps require flying
AGAINST the potential gradient to build speed; the agent follows the
field straight at the finish and falls short. Shaping telescopes (a
detour is net-zero if it returns) but the LOCAL gradient still biases
action selection away from the detour => pay for speed directly.

Rules: NO double runs on local (user). No stale GPUs. A finish is a
milestone, not the goal: then comes finish-time optimization, and above
all convergence research. Follow evidence, 2 seeds before conclusions
(seed variance is documented and large). Simple implementation tasks ->
Opus.

**Slot policy tonight:**
- local: sG3V lead, SOLO, extend on every exit.
- Finisher track (2-3 slots): boxE sG3V_long (ep-ticks 18000), boxG
  sG3V_c; when boxA sG3V_b exits at 6.0e9 -> relaunch as
  **sG3V_speed** (newest view-lineage periodic + `--speed-coef 0.002`
  --steps +1.5e9) = the user-diagnosis arm.
- Research track: box1 (when sGV ext2 exits at 4.5e9) and boxD (when
  sG3 ext4 exits at 7.5e9 — retire the plain frontier, it is
  superseded) cycle scratch screens: 64x32, gamma 0.9995 base, 1.5e9,
  metrics steps-to-25k/47k + wall-break; queue:
  1. sN1/sN1b: net upscale `--emb 1024 --hidden 896` (zero code, 2 seeds)
  2. RND `--rnd-coef` (Fable implementing tonight: predictor-vs-frozen-
     target on the 15 obs scalars — continuous generalization of the
     count keys incl. velocity+view; per-decision bonus, RMS-normalized,
     non-episodic; predictor trains outside the compiled region)
  3. A2 surfable-mask channel (Opus implementing in a worktree from
     spec: bake per-voxel dominant-surface n_z int8 grid from the viewer
     mesh triangles, sample at march hit as channel 2, NHWC interleave,
     CPU tests only; GPU pixel-verify on a rented box before any arm)
  4. A5 stacked renders (Opus, second task, after A2 merges)
- 2-per-GPU (1.2x) allowed ONLY on remote screening slots if the queue
  outgrows boxes; never on finishers, never local.

## Survey landed + tools shipped (2026-08-17 ~04:20)

Opus literature survey: scratchpad\exploration-survey.md (ranked 8).
Directive adopted: ship **--speed-equiv 2** (speed folded into the
potential, WITH the on-death refund) first and regardless of RND — it is
the principled twin of --speed-coef and the only candidate aimed at the
final-ramps gradient problem. Also shipped **--int-speed K** (speed
buckets in the count key). Both merged (283b50b), 49 tests green.
Survey's RND framing correction: RND covers velocity/gaze already, so a
null tonight = "sharp counts beat parametric novelty", NOT "novelty
exhausted" — successor is --int-speed 3, not more RND variants.

Arm priority update: next free finisher slot -> sG3V_pot (newest
view-lineage periodic + --speed-equiv 2, git pull first). Then the
night-plan queue as written (sN1 upscale, sR1 RND, seeds, A2/A5 when
the worktree agent lands).

## A2 surfable-mask MERGED (5f73baa, 2026-08-17 ~04:50)

Worktree agent delivered: surfmask bake (validated against core.trace
normals - 98.7% band agreement on cannonball, 99.8% on the surfable
band; a rasterizer precedence bug found and fixed BY that measurement),
2-channel march as a SEPARATE kernel (default path textually untouched,
bit-exactness guarantee intact), --surf-mask trainer plumbing with a
loud shape-mismatch guard. 62 tests green on main.

**GATE before any --surf-mask arm** (run on a remote box at a rotation):
kernel-vs-fallback pixel parity, int8 triton gather, interleaved store,
default-path bit-exactness test, 2ch perf vs the 1.37ms budget, C=2
channels_last conv (no transpose kernels in a bench_update profile),
compile at C=2, full cannonball bake via the CLI (pre-bake, ~690MB RAM;
+690MB VRAM resident), one short --surf-mask 1 training smoke.

## Experiment added (user, ~05:00): pinhole camera vs equiangular

Current render is equiangular (uniform ANGLE per pixel — fisheye-like;
straight edges curve). Pinhole = uniform image-plane spacing (straight
ramp edges render straight). Implementation delegated to the Opus
worktree agent (branch `pinhole`, --pinhole flag, separate kernel,
default path untouched, mutually exclusive with --surf-mask for now).
Screens as a scratch arm after its GPU-verify; ABI note: same tensor
shape but different per-pixel values -> warm starts see an A1-style
transient.

## Rotation (~05:20): brute force exhausted at 195.6k -> speed-equiv takes the lead slot

sG3V @7.5e9: ceiling moved 195,356 -> 195,605 in 1.5e9 steps (nothing),
sr 0. Brute continuation is done at the final ramps. Local slot ->
**sG3V_pot** = sG3V ckpt_final + --speed-equiv 2 (steps to 9.0e9).
Single delta vs its own parent curve (view-novelty inherited from the
lineage). If the potential-folded speed credit is what the last two
ramps need, this run shows it against a flat 195.6k control.

## Pinhole MERGED (1015851, ~05:45); frame stacking in flight

--pinhole merged: mutation-tested geometry (center ray exact, rotation-
matrix parity 1e-6, straight edge straight-vs-bowed discriminator),
default equiangular path untouched, warm-start across cameras allowed
(A1-style transient). GPU gate: kernel compile + parity vs fallback,
perf vs 1.37ms, lidar-march bit-exactness, --pinhole 1 smoke +
render_pov eyeball. A/B note: pinhole corners see 63.4 deg vs 61.1 —
slightly wider corner fov, less central angular resolution.

A5 frame stacking (--frame-stack K, strides 1/2/4/8 decisions) now with
the Opus agent on branch `framestack`: ring OUTSIDE the CUDA graph,
single-frame b_img + age-clamped gather (bench_capacity convention),
interleaved channels via in_ch. Queue after its merge + GPU gate.

Full implemented-experiment menu now: speed-equiv (flying on the lead),
speed-coef, int-view (validated), int-speed, RND, surf-mask, pinhole,
frame-stack (building), net-upscale (flags), 2-per-GPU screening mode.

## Round 8 (~06:10): speed-equiv beta=2 COLLAPSED warm; wall-195k confirmed thrice

- **sG3V_pot (--speed-equiv 2 on the 195.6k lead): NEGATIVE — policy
  collapsed to 3.5-4k evals within 500M steps**, barely recovering.
  Value/potential shock + the new per-death charge (scale*beta*s ~ 2.5,
  amplified by high-speed respawn spawn potentials) broke the warm
  policy rather than bending it. Killed. Retrying at quarter dose
  (sG3V_pot05, beta 0.5, local). Survey's top pick is dose-sensitive on
  warm policies; scratch test still pending.
- sG3V_b (seed 2) saturated at 195,240 = same ceiling as the lead.
  sG3V_long (18k-tick cap) final max 195,533 = THIRD confirmation the
  195k ceiling is the final ramps, not the clock.
- sGV scratch-view lineage @4.5e9: max **99,212 — nominally past the
  old warm all-time record (99,004)**; retired for now (box needed).
- Rotations: boxA sG3V_speed (--speed-coef 0.002 twin, from seed-2 ckpt);
  boxE sN1 upscale scratch; boxD sR1 RND scratch (smoke line clean);
  box1 running the surf-mask GPU-verify chain (pytest + full bake +
  --surf-mask smoke, log /root/mask_verify.log) then sM1.

## Frame stacking MERGED (cf9dd0a, ~06:35) — the implementation menu is COMPLETE

--frame-stack K merged (104 tests green): ring outside the CUDA graph,
prologue rows (no fake episode starts at iteration seams), age-clamped
gathers shared between rollout and update, mutation-tested. GPU gate
before any arm: graph capture with widened static_obs, compile at C=K,
compose cost, default-off TIMING parity, stacked ckpt through
record_ckpt. Every experiment from the user's overnight list is now a
flag: speed-equiv, speed-coef, int-view, int-speed, RND, surf-mask,
pinhole, frame-stack, net-upscale. Screens proceed per the ping loop.

## Round 9 (~06:55)

- sG3V_c final: max 195,497 — FOURTH independent confirmation of the
  195k ceiling. Rotated to sIS (int-speed screen).
- Mask GPU-verify: PASS in substance — 79 GPU tests incl. the
  lidar-march bit-exactness pin, and the --surf-mask smoke trained
  normally (in-trainer bake worked). CLI invocation needs
  PYTHONPATH=python (cosmetic; note for DEPLOY). sM1 mask arm launched
  (box1).
- sR1 RND @0.5: MIS-DOSED, not broken — rew -7.0 with ~1480-tick
  episodes = loiter-until-stall-kill; coef 0.5 pays ~7x racing income.
  Killed; relaunched sR1b at 0.05. Dose lesson mirrors int_coef's.
- pot05 (speed-equiv beta 0.5 warm): NO collapse (rew ~28.5 = parent
  level) — the beta-2 failure was dose, not concept. Ceiling verdict
  pending evals.
- Fleet: local pot05 | boxA speed-coef twin | boxE upscale | boxD rnd005
  | boxG int-speed | box1 surf-mask. All six screens/finishers live.

## Round 10 (~07:25): speed-equiv verdict + frame-stack arm live

- sG3V_pot05 final (9.0e9): max 195,593 vs parent 195,605 — **speed-equiv
  0.5 = CEILING-NEUTRAL** (safe dose, no break; beta-2 destroys, beta-0.5
  does nothing at the wall). The potential-shaping speed hypothesis is
  effectively closed for the 195k wall pending the speed-coef twin's
  score.
- Frame-stack GPU gate PASSED on local (graph capture, compile 28s at
  C=4, suite 104 green); sF1 (--frame-stack 4 scratch, 1.5e9) running
  locally.

## Round 11 (~07:45): reward-shaping family CLOSED at the 195k wall

- sG3V_speed final (7.5e9): max 195,509 — ceiling-neutral, same as
  speed-equiv. **Both speed hypotheses neutral: the reward-shaping
  family is exhausted at the final ramps.** Remaining levers (survey):
  far-side/reverse-curriculum seeding (spawn beyond the wall with
  plausible velocities — needs a small spawn-pool tool) and long
  training on the view lineage. Flagged for the MORNING REPORT, not
  implemented blind overnight.
- sN1 upscale screen: 25k@403M (fast), 47k@907M, no break by 1.5e9 —
  within seed noise; sN1b (seed 2) launched to decide.
- sP1 pinhole launched clean (kernel compiled, first eval renders) —
  its GPU gate passed via launch watch.
- Battery now: sF1 stack (local) | sP1 pinhole (A) | sN1b upscale (E) |
  sR1b rnd005 (D) | sIS int-speed (G) | sM1 mask (1).

## Round 12 (~08:15)

- **sR1b RND @0.05: NULL-TO-NEGATIVE** — 25k only at 1.06e9, never
  reached the wall (vs base 25k @400-650M). With the 0.5 farming
  failure, RND is closed in this regime: the survey's prediction held
  (the count table's sharp first-visit spike is the working mechanism;
  smooth parametric novelty is not). No seed 2.
- sM1 mask: 25k@454M, 47k@756M, wall-parked 48.8k — base-pace-or-better,
  within seed noise; sM1b (seed 2) launched (box1).
- boxD -> sG3V_marathon: the long view-lineage backstop (7.5e9 ckpt,
  --steps 12e9, plain continuation) — the brute-force lane while the
  morning decision on far-side seeding waits for the user.

## Round 13 (~08:40): int-speed is the NIGHT'S SCREEN CHAMPION

**sIS (--int-coef 0.25 --int-speed 3): 25k@303M (fastest arm ever),
47k BROKEN @605M, ~99k by 1.5e9** — the best scratch curve of the
project at that step count (the view arm needed ~3e9 for 99k).
Speed-keyed novelty patches exactly the counts' documented blind spot
(walls are speed-gated), and it beats plain (1/3 break rate, slower),
view-keyed (break ~1.35e9), RND (null), entropy/boost (negative).
Extended to 3.0e9 on boxG; seed 2 is TOP of the rotation queue.
Marathon backstop at 8.14e9, 195.5k band, no finish yet.

Champion-recipe candidate as of now: gamma 0.9995 + int-coef 0.25 +
int-speed 3 (+int-view 8 to be tested as a combo later), 64x32.

## Round 14 (~09:00)

- sF1 frame-stack: 25k@655M, then STUCK on the 25k shelf through 1.5e9,
  never reached the wall. Below base pace — motion history does not
  help here (velocity is already in the scalars; the shelf-camping
  pattern matches the weak seeds). Verdict lean-negative; no priority
  seed 2 (the shelf-vs-seed question is confounded, but the queue's
  budget goes to the winners first).
- Local -> sIS_b (int-speed seed 2), the queue head.

## Round 15 (~09:20)

- sP1 pinhole: NEGATIVE — 25k shelf, never reached the wall. The
  equiangular camera is not the bottleneck; no seed 2.
- sN1b upscale seed 2: 25k@303M, 47k@504M (fast!), wall-parked 48.6k.
  Two seeds (907M / 504M to wall, no breaks): **upscale = neutral
  pre-wall** — capacity is not the binding constraint before the wall;
  may matter later, not a screen priority.
- sIS ext: holding 98.3-98.6k at 1.82e9 (wall #2 zone).
- Rotations: boxA sISV_combo (speed+view keys), boxE sIS_c (seed 3 of
  the champion mechanism).

# ============================================================
# MORNING REPORT (2026-08-17, night shift ~01:00-09:45)
# ============================================================

## 1. Headline

**No finish yet — but the map's difficulty is fully decoded, the
convergence recipe improved ~4x overnight, and one mechanism (YOUR
speed-keyed curiosity) produced the fastest learning curve the project
has ever seen.** Best frontier: 195,605 of 198,380 (98.6% of the track),
reached from scratch in ~1 hour of training; the final 2,800 units (the
last two against-the-gradient ramps) resisted every reward-side attack.

## 2. The night's evidence table (scratch screens, 1.5e9 steps, 64x32,
gamma 0.9995 base unless noted; base rates: 25k@400-650M, 47k@500-900M,
wall-break ~1/3 seeds)

| arm | 25k @ | 47k @ | end | verdict |
|---|---|---|---|---|
| **int-speed novelty (int 0.25 + speed-key 3)** | **303M** | **605M BROKE** | **~99k @1.5e9** | **WINNER — fastest curve ever; seeds 2/3 + view-combo running** |
| gamma 0.9995 (vs 0.999/0.995/0.9995+) | - | 500-650M | 82k @1.5e9 (seed 1) | WINNER (earlier tonight) — dose peak at 0.9995 |
| view novelty (int 0.25 + view-key 8) | ~550M | broke @1.35e9 | 67k @1.5e9 | WINNER (earlier) — carried the 195k frontier lineage |
| net upscale (emb1024 h896) | 303-403M | 504-907M | wall-parked | NEUTRAL (2 seeds) — capacity not binding pre-wall |
| surf-mask channel | 454M / never | 756M / never | 48.8k / **5.1k** | HIGH-VARIANCE, unproven (seed 2 anomalous; 3rd seed needed if pursued) |
| speed-equiv (potential) beta 0.5 / 2.0 | warm arm | - | ceiling-neutral / COLLAPSED | reward-shaping family CLOSED at the wall |
| speed-coef 0.002 (warm) | - | - | ceiling-neutral | same |
| RND 0.5 / 0.05 | loiter-farm / 1.06e9 | never | 30k | **NEGATIVE both doses** — sharp count novelty is the working mechanism, parametric novelty is not |
| pinhole camera | 655M | never | 25k shelf | NEGATIVE |
| frame-stack 4 | 655M | never | 25k shelf | NEGATIVE (velocity already in scalars) |
| binned respawns (S1) | never | never | 13-19k | NEGATIVE from scratch (mature-frontier tool only) |
| kill-aware field (S2) | - | - | null at walls 1 AND 3 | CLOSED — walls are jump-skill, not gradient artifacts |
| entropy 0.02 / spawn-boost 2x | earlier | - | - | NEGATIVE (slower) |

25k-shelf caveat: a real map barrier some seeds camp on; negatives
marked "25k shelf" are partially confounded with seed luck, but none
showed above-base pace before stalling.

## 3. The 195k wall (the one that matters)

Confirmed FOUR independent ways (plain/seed-2/18k-tick-cap/30k-cap
probe): the last two ramps require flying AGAINST the potential
gradient to build speed (user's diagnosis). Reward-side fixes are
exhausted (speed-equiv dose ladder, speed-coef, kill-aware). Remaining
levers:
- **A. Far-side / reverse-curriculum seeding (USER DECISION)**: spawn
  beyond/at the final ramps with plausible velocities (we know the
  geometry; a small spawn-pool tool, ~half a day incl. review). The
  survey's #3 pick; the only candidate that attacks V(beyond the
  gap)=0 directly.
- **B. Marathon brute force (RUNNING)**: view-lineage at 9.5e9 of 12e9
  on boxD, finish watch armed.
- C. Champion-recipe scratch runs at scale (int-speed lineage may
  simply pass 195k with its better exploration — extension at 2.16e9
  and climbing seeds will tell).

## 4. Spend

~$35 of the $100: ~$15 through the evening waves + 5 boxes x ~9.5h x
$0.4 ~ $19 overnight. All 6 GPUs (5 rented + local) currently busy.

## 5. Champion recipe (as of this morning)

64x32 equiangular depth render | gamma 0.9995 | int-coef 0.25 +
int-speed 3 (view-key 8 combo verdict due ~10:30) | respawn-frac 0.9,
margin 10s, speed 1.0-1.5 | everything else stock. All flags
ckpt-persisted; every feature from the overnight list is merged, tested,
and GPU-gated (frame-stack + pinhole verified on GPU via their arms).

## 6. Open items for the day

- sIS_b / sIS_c / sISV verdicts (due ~10:15-10:45) -> lock the champion.
- Marathon outcome at 12e9 (~11:00).
- Far-side seeding: implement or not (your call).
- DEPLOY.md updates queued: surfmask CLI needs PYTHONPATH=python; rent
  filter CUDA 12.8+/driver 570+; 2-per-GPU = screening-only (1.2x).
- Finish-time optimization phase (after the first finish): the
  time_pen/gamma trade and eval protocol are ready for it.

## Report addendum (~10:10): champion replication tempers the claim

sIS_b (seed 2): 25k@957M, 47k@1.21e9, wall-parked 49.6k — the champion
mechanism's seed-1 curve (303M/605M/99k) does NOT fully replicate;
honest status = "int-speed novelty raises the ceiling of good seeds and
produced the best run ever, but seed variance stays large." Seeds 3
(boxE) and 4 (box1, just launched) complete the distribution. Local ->
sIS_long (champion recipe, 4.5e9): lever C — can the better explorer
pass 195k without new mechanisms?

## Tail rounds (~10:30)

- OPS BUG caught: sIS_c on boxE never started (stale repo, unrecognized
  --int-speed; the LAUNCHED echo hides subshell failures). ~50 min slot
  loss. Fixed: pull + relaunch verified via log grep + process count.
  RULE for every future launch: grep the log for errors AND count
  trainers — never trust the echo.
- sIS_d (seed 4, box1) verified genuinely running (105M, healthy).
- sIS ext at 2.50e9: 96-99.3k band (max 99,258 — nudged past the sGV
  lineage's 99,212). Marathon 10.17e9: 195.5k band, no finish.
- sISV combo at 1.29e9, healthy; scores next cycle.

## Tail (~10:50): the combo breaks the wall too

- **sISV (speed+view keys): 47k BROKEN @1.06e9, 50-55.6k and climbing
  at 1.5e9. Extended to 3.0e9.** Novelty-key family break tally: 3 of 4
  completed seeds broke the wall (speed@605M, combo@1.06e9, view@1.35e9)
  vs plain's ~1/3 — the keyed-curiosity claim now rests on a real
  break-rate differential, not one lucky seed. Morning-report table
  stands with this strengthening.
- sIS ext at 2.84e9: 80-99k oscillation (wall-2 zone), max 99,258.
- Marathon 10.84e9: 195.5k, no finish.

## Tail (~11:35): MARATHON VERDICT — brute force is dead at the wall

sG3V_marathon final (12.0e9): max 195,554, no finish. **4.5e9 extra
steps of pure brute force on the best lineage moved the ceiling ZERO
units** (lineage max 195,605 stands from 6.0e9). Final ckpt:
runs/sG3V_marathon/ckpt_final.pt on boxD (pre-rotation copy left in
place). This closes lever B: the 195k wall does not fall to compute at
this scale. The morning decision is now sharply posed: **far-side /
reverse-curriculum seeding (lever A, user call) is the only untried
mechanism aimed at the final ramps** — or accept the wall pending the
champion lineage's own climb (sIS at ~99k chasing wall 2 on boxG +
parallel stream on boxD, combo on boxA).

## FINAL champion statistics (~11:55) — distribution complete

int-speed seeds (scratch, 1.5e9): seed1 47k@605M -> 99k; seed2 parked
@1.21e9; **seed3 47k@454M -> 97.4k (near-perfect replication of seed
1)**; seed4 parked @1.01e9. Break rate 2/4, and BOTH breaks ran to
97-99k within 1.5e9 (plain's lone break reached only ~82k there).
Novelty-key family overall: **4 breaks / 6 seeds (vs plain 1/3), with
2x-deeper runs after the break.** The champion recipe stands: 64x32,
gamma 0.9995, int-coef 0.25, int-speed 3 (view-key combo also breaks,
slightly slower). All six slots now run champion-lineage extensions or
parallel streams: sIS ext2 (boxG, 99.3k), sIS_par (boxD), sIS_c ext
(boxE, 97.4k), sIS_d ext (box1), sISV ext (boxA), sIS_long (local).
No new mechanisms until the user's far-side decision.

## Report addendum (~12:30): THE COMBO PASSES WALL #2 FROM SCRATCH

**sISV (speed+view keys) @3.0e9: 134,323 and climbing steeply**
(104k -> 127k -> 134k) — past the wall-2 zone where the whole int-speed
family camps, deepest scratch-born run outside the original frontier
lineage, heading for wall #3 (180k). Extended to 4.5e9. sIS_par set the
family wall-2 record (99,886, still nudging; extended to 6.0e9).
CHAMPION RECIPE UPDATE: the speed+view COMBO is the recipe — it breaks
wall 1 reliably AND passes wall 2, which speed-only has not.

## Report addendum (~13:00): combo at 160.8k and accelerating

sISV @3.76e9: 134.8 -> 155.1 -> 160.8k — 20k from wall #3, replaying
the frontier lineage's climb from a scratch birth (and faster per
step). sIS ext2 (speed-only) closed its leg parked at 99,350 (wall 2);
boxG rotated to sISV_par (combo parallel stream from the 3.5e9
periodic, to 5.0e9). Fleet: combo lineage on boxA+boxG, int-speed
lineage on boxD/boxE/box1/local.

## Report addendum (~13:15): wall-2 pattern definitive

sIS_long final (4.5e9): broke wall 1 @756M, 90k @1.66e9, capped 99,533.
THIRD independent speed-only run parked at 99-100k: wall 2 stops
speed-keys reliably; only the speed+view COMBO has passed it. Local slot
-> sISV_par2 (third combo stream, from the 3.5e9 periodic, to 5.0e9).
Combo lineage now on boxA (lead, 160k+), boxG, local; int-speed lineage
on boxD/boxE/box1 held-or-finishing.

## Report addendum (~13:40): COMBO PASSES WALL #3 FROM SCRATCH + a correction

- **sISV combo @4.5e9: max 185,370 — through wall #3**, 13k units from
  the finish, in the final-ramps territory in ~2h of scratch training
  (the original lineage needed warm history + far more). Extended to
  6.0e9.
- CORRECTION to "wall-2 definitive": sIS_c's extension passed walls 2
  AND 3 to the universal ceiling (195,255 — joins the 195k club, HELD).
  Speed-only wall-2 pass rate ~1/4 in extensions; combo 2/2 so far —
  a RATE difference, not an absolute barrier. Ledger claims updated.
- sIS_d ext improving (48.6 -> 72.6k), extended once more. boxE held
  one cycle (ceiling-parked ckpt preserved).
- The 195k final-ramp ceiling now holds FIVE independent lineages.
  Everything funnels to the same last jump; the far-side decision is
  squarely posed for the morning.

## Report addendum (~14:05): FIRST FINISH-LINE CROSSINGS EVER

**sISV_par2 (local combo stream): training episodes CROSSED THE FINISH
LINE — the first finishes ever recorded in this project.** ~0.3% of
episodes around step 4.65e9, finish-from-spawn ~8.9-9.6 s => these are
RESPAWN-spawned episodes (reservoir snapshots near the end) landing the
final ramps — NOT start-to-finish runs yet. What it means: the last
jump is now being landed in training, V(finish) exists for the first
time, and the respawn curriculum can consolidate backward — the exact
mechanism that cracked walls 1-3. Harvested:
runs/frozen/sISV_firstwins.pt (step 4,500,750,336, md5 40d55c82...).
Lead combo (boxA) at 194,981 max; other streams no wins yet.
Start-to-finish (greedy eval from the start line) = the remaining
milestone.

## Report addendum (~14:30): CONSOLIDATION EXPLOSION — finish is imminent

sISV_par2 @5.0e9: success rate 0.3% -> **48.25%** in 350M steps (260
win rows), finish-from-spawn lengthening 9s -> 25s (consolidating
BACKWARD, deeper spawns finishing), and the greedy start-line eval hit
**196,681 — past 196k, ~1,700 units from a full start-to-finish run.**
Extended to 6.5e9. The far-side-seeding decision is likely MOOT: the
respawn curriculum is doing exactly that job by itself now that
finishes exist. Watch: first start-line finish expected within hours
or less.

# ============================================================
# THE MAP IS BEATEN (2026-08-17 ~14:50)
# ============================================================

**surf_src_cannonball completed start-to-finish by sISV_par2** — a
scratch-born agent (no warm history) running the night's champion
recipe: 64x32 depth, gamma 0.9995, count curiosity keyed on position x
speed(3) x view(8), int-coef 0.25, standard respawns. At record point
~5.40e9 steps, ALL NINE greedy start-line eval episodes traversed the
full track (mean progress 198,380.6 >= d0), each in ~85.5 s (8,476-
8,583 ticks). Training success rate 73% and rising; finish-from-spawn
times lengthening as consolidation reaches back to the start.

Evidence notes: the traj footer 'fail' label is cosmetic (record.py
infers it from base rewards, which never carry the race bonus — fix
queued); the mean-progress argument is conclusive since ANY incomplete
episode would drag the 9-episode mean below d0.

Checkpoints frozen: runs/frozen/sISV_FINISHER_latest.pt (dd9cb603...)
and sISV_FINISHER_periodic.pt (40d55c82...). Total scratch budget of
the finishing lineage: ~5.4e9 steps (~2.5 GPU-hours at fleet rates).
Next phase per the user's standing brief: fastest-finish optimization
+ the research-program conclusions (all in this ledger).

## Post-finish hold (~15:20)

Finisher (par2): sr 78%, fin 28.6s, consolidating to 6.5e9. Combo
streams extended for the robustness stat + fastest-finish material
(boxA -> 7.5e9 at 195k; boxG -> 6.5e9; boxE -> 8.0e9). Speed-only boxes
(boxD, box1) HELD IDLE awaiting the user's wind-down/next-phase call —
their parked ckpts preserved on-box. Spend running ~ $0.8/h on the idle
pair; flag for the user.

## Post-finish hold (~16:40): the finish is STABLE

sISV_par2 through 8.0e9: **13 full-track eval points** (>=198,383),
including three in the final billion — start-to-finish completion is
recurring, not luck. Training sr ~72-78%, finish-from-spawn ~30s.
Extended to 9.5e9. All four combo streams reach 194k+ (wall 3 passed
fleet-wide). Idle: boxD, box1 (user call pending).

# ============================================================
# RESEARCH PAUSED (user, 2026-08-17 ~17:30)
# ============================================================

State at pause: map beaten (stable, 9/9 evals + 13 repeats); fastest
recorded run **1:19.73** (video: runs/sISV_par2/fastest_finish_79.73s.mp4).
Local training PAUSED cleanly (sISV_par2 at ~8.6e9; ckpt_latest +
periodic ckpt_8500543488 intact, ~154 MB each — count tables included).
Remote fleet destroyed by the user (~$40 of $100 spent). Frozen
checkpoints in runs/frozen/ (F', F2, F3, sISV_FINISHER_*).

## Standing goals for the next session

1. **Beat the human world record: 1:08 vs our 1:19.73** (-11.7s needed).
   Levers not yet tried for TIME (vs completion): time_pen/gamma retune
   on the finisher, best-line RL (the finisher still takes safe lines),
   self-imitation on fastest episodes, eval protocol for best-time.
2. **Converge in ~1h instead of ~2.5h from scratch.** Known headroom:
   the seed-variance tail (4/6 break rate — understand/kill the
   25k-shelf trap), 32x16 screens for search + 64x32 finals, 2-per-GPU
   screening (1.2x), combo-recipe curriculum tweaks, remaining survey
   candidates (NovelD, far-side seeding — never needed for completion,
   maybe useful for speed).

# ============================================================
# RECORD CHASE — Round 13 (2026-08-17 evening)
# ============================================================

## 12h marathon verdict: pure iteration is NULL (again)

marathon12h (best-recipe warm resume of sISV_par2 @8.96e9 on a rented
5090): ran to 16.46e9 (+7.5e9 steps, ~6.8h at ~307k fps). Training
win% ~87 @ ~31s from-spawn, rew ~64 — all curves stationary; start-line
eval time did not move. Replicates the 12e9 brute-force null: the
converged policy does not get faster from more steps. The speed signal
has to come from the OBJECTIVE. User stopped the marathon; box
repurposed. Seed ckpt for this round: step 16,456,876,032
(md5 81083ba7...; local backup runs/marathon12h/ckpt_16456876032.pt).

## Why the agent is time-insensitive (reward economics)

Potential shaping pays a FIXED 100 total regardless of pace (by
design); time_pen 0.005/tick = 0.5/s; success_bonus 50 is flat, so its
only speed pressure is discounting (~1.65%/s at gamma 0.9995, 33Hz
decisions). Net: ~1 return-unit per second saved vs tens of units of
episode-return noise. After advantage normalization that gradient is
mud — win-rate saturates at ~87% and the policy stops.

Correction to an earlier note: the race lineage trains at
**maxvel=4000** (ckpt config), NOT stock 2000 — no speed handicap vs
the human WR server.

## New levers (commit 0742758)

- `--finish-k K --finish-tref T`: bonus += K*max(0, T - T_ep_seconds)
  ON the finish tick. Per-second time pressure paid only to finishers:
  no suicide channel (vs raising time_pen), no avoid-the-curtain
  channel (>=0 clamp). Gradient wrt finish time = -K everywhere.
- `--reset-int-counts`: discard the ckpt'd novelty table on resume —
  re-arms count curiosity on a converged policy.

## Round 13 arms (~3h each, all warm from the 16.46e9 seed)

| box | arm | delta vs champion | question |
|---|---|---|---|
| boxM 16162 root@89.221.67.172 | wINT05 | `--int-coef 0.5 --reset-int-counts` | curiosity dose-up on fresh table: does 2x novelty (speed/gaze keys live) find faster lines — or hit the count-table overdose cliff (RND died at 0.5)? |
| boxH 51239 root@95.253.220.115 | wFK2 | `--finish-k 2` | pure time-scaled bonus: 5x speed pressure, zero suicide risk |
| boxI 12608 root@81.166.162.13 | wG999 | `--gamma 0.999` | discount retune: exploration phase over, 2x time preference (3.3%/s) |
| boxJ 13413 root@81.166.173.12 | wCOMBO | `--finish-k 2 --gamma 0.999 --time-pen 0.01` | all speed levers at once |

Judge on: start-line greedy eval finish time (eval_eps=9 rides in the
ckpt config), training finish_s trend, win% (must not collapse below
~70%). 3h decision rule per the user: no help in 3h = drop.

## Queued idea (user, 2026-08-17 evening): decision-frequency ladder

We decide at 33Hz (--act-every 3 over 100Hz physics) and have NEVER
ablated it. User: replay actions look needlessly high-frequency;
adjacent samples are near-duplicates adding no information. Raise K:
~K x cheaper per game-second (render+policy+update are per-decision;
physics is cheap C) -> direct 1h-convergence lever; decorrelated
gradient samples; and gamma-per-decision means K stretches the horizon
in game-seconds (K=9 @ 0.9995 -> ~180s vs 60s — the wall-breaking
mechanism, for free). Risk: control-timing quantization at speed (one
decision per ~360u at K=9 @ 4000 u/s; the transfer jumps are precision
moves) — expect a cliff somewhere past K=9, find it.

Design (scratch, champion recipe, judged on wall-clock to first finish
+ final eval time): K=6; K=9 gamma kept 0.9995 (horizon 3x as part of
treatment); K=9 gamma rescaled 0.9985 (per-second horizon fixed —
isolates the compute/decorrelation effect); K=30 only if K=9 holds.
Existing K=3 curves are the baseline. Warm transfer is NOT the test
(last-action obs + value timescale change semantics; scratch is
honest and fits the convergence goal anyway).

## Queued idea REFINED (user): subsample the UPDATE, not the control

User clarification on the frequency idea: control/rollout frequency
stays at 33Hz ("we can control as frequent as we want") — the
redundancy to kill is in what goes through forward/backward. Adjacent
30ms samples are near-duplicates; their correlation adds no gradient
information, only compute.

Measured split (local 5090, champion config, 40-iter --timing probe):
rollout 649 ms + GAE 9 ms + update 685 ms = 1363 ms/iter — the update
is 50% of the wall. Implemented `--train-stride S` (commit after
0742758): GAE full-chain as before, optimizer sees every S-th timestep
(rotating offset, whole minibatches only). Probe at stride 3: update
685 -> 192 ms, total 1363 -> 753 ms = **1.81x wall-clock per
game-second**, tighter iteration spread (13% -> 4%), no recompiles.

Open question for the screen: stride 3 = 3.2x fewer gradient steps
per game-second, and the oldest trainer lesson is update density >
raw steps/s. If sample-efficiency-per-game-second holds, this nearly
halves scratch convergence ON ITS OWN (2.5h -> ~1.4h) before any
other 1h-goal lever. Screen design: champion scratch, arms stride 1
(control) / 3 / 10 (stride 10 = 1 minibatch/epoch, near the 2.1x
ceiling), judged on wall-clock to first finish + milestones-vs-steps
(a per-step match at stride 3 = pure win). Composes with the separate
act-every ladder (that axis ALSO cuts render+inference and stretches
horizon; this one is update-only and risk-free for control).

## Round 13 verdict (~3h, start-line greedy evals, 9 eps each)

Seed reference (16.46B marathon ckpt at resume): 6/9 finishing, mean
~82.0s. Last-3-eval results per arm:

| arm | finishers | mean time | verdict |
|---|---|---|---|
| **wG999 (gamma 0.999)** | **8/9, 8/9, 8/9** | **81.3 / 80.8 / 81.7s, best 80.36** | **WINNER: more robust AND faster than seed. The discount retune is the working speed lever.** |
| wINT05 (int 0.5 fresh) | 7/9 steady | 82.9 -> 81.9 -> 81.6s | mild positive, drifting down; NO overdose cliff at 0.5 (count-based survives where RND died) |
| wFK2 (finish-k 2) | 2/9, 2/9, 4/9 | 83-85s | NEGATIVE on the metric that matters: from-spawn times improved (31->27s) but full-track robustness collapsed. Mechanism: episode-time bonus is dominated by respawn-spawned (short) episodes -> optimizes near-finish segments, trades away early track. |
| wFK5 (finish-k 5) | 6/9 -> 2/9 in 1h | 84s+ | dose-response of a harmful lever; killed early |
| wCOMBO (fk2+g999+tp.01) | 0/9 | - | hard collapse (win 0%, stall-death spiral); killed at 2.8h |

Lesson: PER-EPISODE time bonuses mis-target under respawn curricula
(the episode mix is mostly mid-track spawns). UNIFORM time pressure
(discounting) transfers to the full track. finish-k as built is closed
for warm+respawn use; a start-spawn-gated variant remains untested.

Next rotation: wG9i5 launched on boxJ (gamma 0.999 + int-coef 0.5
fresh table — the two non-negative levers combined). wG999, wINT05,
wFK2 keep running to their step caps; wFK2 slot is next to repurpose
(gamma dose 0.9985 queued).

## Trainer perf (merged to main meanwhile)

--train-stride 3: update 685->192ms. --reward-per-decision: reward_py
299->109ms (paired: 2 base + 3 rpd replicas, metric bands overlap).
Combined expectation ~580ms/iter vs 1363 baseline (~2.3x) for the next
scratch runs; long-run learning-quality confirmation rides on those.

# ============================================================
# ROUND 14: SCRATCH FLEET (user call, 2026-08-17 late evening)
# ============================================================

User verdict on the warm phase: 80s evals vs the 79.73s record = no
real improvement; unsticking a converged checkpoint is the hard road.
Fresh confirmation: marathon seed re-evaled at 8/9, mean 82.05s.
All warm arms stopped (wG999's 80.36s-best ckpt parked on its box in
runs/wG999/). Fleet relaunched FROM SCRATCH, judged on
steps-to-milestones, the new race/eval_finish_s clock (commit 71115e0
- start-line greedy seconds now in log + csv; training finish_s is
from-spawn and was misread as the race clock), and for the repro arm:
wall-clock vs the original ~2h champion convergence.

| box | arm | delta vs champion recipe | question |
|---|---|---|---|
| boxM 16162 89.221.67.172 (healthy) | sREPRO_s3 | + --train-stride 3 only | the user's perf check: same recipe, 1.8x cheaper iterations - does it converge faster on the WALL CLOCK? |
| boxH 51239 95.253.220.115 (healthy) | sI05 | int-coef 0.25 -> 0.5 | stronger curiosity from step 0: faster wall-breaks? |
| boxI2 46963 81.183.225.11 (healthy) | sG999 | gamma 0.9995 -> 0.999 | never tested from scratch WITH keyed curiosity; sharper time preference from step 0 (round-13's warm winner) |
| boxJ 13413 81.166.173.12 (capped 71%) | sG999i05 | both | combo long-shot on the slow box |

finish-k dropped (round-13 worst). Reference curves: sISV_par2
(champion, walls at 605M-1.06B, first finish ~5.4e9, ~2h healthy 5090).

## Round 14 correction: boxM CPU cliff exposed, sREPRO_s3 <-> sI05 swapped

User caught sREPRO_s3 (on boxM) at 219k fps vs siblings' 560k+. TIMING
truth: stride's update = 206ms (working as designed), but boxM's CPU
path collapses under the scratch workload — env 751ms, reward_py 635ms
(vs 166/373 on a healthy box). Short scratch episodes are reset/CPU-
heavy; boxM (256-core host) always capped ~300k, which warm update-
bound workloads masked. Wall-clock-judged arms must not sit on it.

Swap executed (~15 min in, both restarted from scratch): sREPRO_s3 ->
boxH (healthy), sI05 -> boxM (per-step-judged, box speed irrelevant).
Result: **sREPRO_s3 at 783k fps sustained (912ms/iter), the fastest
rate ever logged** — vs ~560k stock siblings = ~1.4x wall-clock on
scratch (less than warm's 1.81x: scratch is rollout-bound and
reward_py 373ms is now the top line -> --reward-per-decision is the
next perf lever to validate long-run).

Same-step check at 70.8M (user q): sREPRO rew -0.59 vs sG999 -0.17 /
sI05 +0.42 — noise band at win 0% (sI05's rew inflated by 2x intrinsic
income); per-step parity gets judged at the milestones, which is this
arm's job.

## Round 14 verdict at 8e9 (all budgets exhausted, ~3h)

**0/4 scratch seeds finished.** Track bands at end: sG999i05 118-135k
(still climbing, best of fleet, wall-3 zone next), sG999 88-98.6k,
sREPRO_s3 oscillating 50-97k (wall-2 camp), sI05 75-96k @ 4.0e9 (slow
box, half budget). The champion's "~5.4e9 to finisher / ~2h" was the
LUCKY SEED of ~8 parallel streams — tonight's 0/4 is consistent with
that seed-variance reality, and makes it the dominant cost of the
1h-convergence goal (confirming roadmap item 2's 25k-shelf/variance
focus).

**Stride wall-clock verdict (the round's clean result): sREPRO_s3
averaged 824,588 steps/s over the full 8e9 run (2.70h) vs stock
sibling 595k (3.73h) on matched hardware = 1.39x.** Per-step learning
parity: not refuted (trajectory inside the historical seed band:
broke wall 1, camped at wall 2 like most stock seeds) but not proven
- needs same-recipe stride-vs-stock seed pairs, which seed variance
makes expensive. Convergence-time question: unanswered this round
because nothing converged.

## Wind-down (user: keep 1 of 4)

Backed up to local runs/round13_14_archive/: wG999 warm final (round
13 winner, 80.36s best eval), sG999i05 8e9 ckpt, all four
progress.csv. KEPT: boxH 51239 root@95.253.220.115 (healthy, fastest)
running **sG999i05_ext** - the best run resumed at 7.98e9 toward 16e9
under a self-restart wrapper (runs/ext_wrap.log). SHUT DOWN: boxM
16162 89.221.67.172, boxI2 46963 81.183.225.11, boxJ 13413
81.166.173.12 (backups complete, nothing live).

## Stride dose-response verdict (local, 2026-08-18 ~01:30)

s50_scratch stopped at 1.11e9: eval track 7.8k vs stock champion seeds
31.6-47.9k at matched steps — stride 50 is update starvation (4 grad
steps/iter from 2-3 timestep clusters), CONFIRMED harmful from scratch
despite 1.2M steps/s. Stride 3 at 1e9: 16.8k — behind both stock seeds
per-step AND at matched wall time (18.5k @1.4e9), but sIS_b sat at
9.6k until 750M then broke, so n=1 cannot separate mild per-step cost
from seed luck. Pattern across doses: update density is the currency;
stride buys wall-clock by spending it (mild at 3, fatal at 50).

## QUEUED (user directive): decision-frequency x stride experiment

User: "predict every 3rd step, act the same for 3 steps — 100Hz is
definitely redundant for surfing; I believe it can be increased to 10
[Hz], needs testing; 50 is def too much."

Arms (champion recipe, scratch, healthy 5090s):
- A: --act-every 9 (11Hz decisions, gamma kept 0.9995 -> horizon
  stretches ~3x to ~180 game-s — the wall-breaking mechanism, free)
- B: --act-every 9 --train-stride 3 (the user's combo; NOTE this is
  9x less update per game-second than champion — tonight's density
  lesson says pair it with arm A as control)
- C: --act-every 10 (the user's 10Hz hypothesis; gamma-rescale
  variant 0.9995^3~=0.9985 if A shows horizon confound)
Watch for: control-granularity ceiling at the transfer jumps (~360u
traveled per decision at 4000 u/s), fps (act-every cuts render+
forward+update, deeper than train-stride), and per-step parity vs the
sIS seed band. Overnight: only boxH runs (sG999i05_ext); ladder
launches next session.

## Morning 2026-08-18: overnight verdict + act-every ladder launched

**sG999i05_ext: REGRESSED.** 148k track @9.1e9 (23:00) fell to a
60-92k eval band by the 16e9 cap; 0 finishes across the whole
extension. The gamma-0.999 + int-0.5 scratch lineage is CLOSED as
negative — its second 8e9 actively hurt the greedy policy (sustained
2x novelty income and/or the short horizon degrade the long-track
exploit).

**Stale-box postmortem:** trainer completed --steps 16e9 at 03:14,
rc=0; the restart wrapper had no budget-exhausted case and spun out.
~5.5h idle, nothing of value lost (the run was regressing). Wrapper
rule for next time: check ckpt step vs cap before relaunching.

**Ladder launched (queued arms A and B):** boxH sAE9 (--act-every 9,
11Hz decisions, gamma kept), local sAE9_s3 (--act-every 9
--train-stride 3, the user's combo). Champion recipe, scratch, 8e9
budgets, judged vs the sIS seed band per-step + wall-clock.

## Act-every ladder verdict at ~2e9 (2026-08-18 morning, both arms stopped)

| arm | eval track | fps |
|---|---|---|
| sIS_long (stock 33Hz reference) | 43.6k @1B, 57.9k @1.5B, 98.4k @2B | ~560-640k |
| sREPRO_s3 (stride 3, 33Hz) | 16.8k @1B, 18.5k @1.5B, 24.4k @2B | 824k |
| sAE9 (act-every 9 = 11Hz) | ~14.5k @1.5B (13-15k band) | 994k |
| sAE9_s3 (11Hz + stride 3) | 7-12k @2B, oscillating | 1.29M |

**act-every 9 is NEGATIVE at this dose, and not only via update
density**: sAE9's density (1/3) equals sREPRO_s3's, yet it tracks
BELOW it per-step — the lost 33Hz decisions carry learning value
beyond sample count (finer credit assignment / exploration
granularity; the control ceiling at walls was never even reached,
win 0%). The combo (1/9 density) shows the starvation signature
(~s50-like band). Wall-clock-fair comparison also loses: stock
reaches 43.6k in the time sAE9 reaches ~14k x1.7 steps.

Standing conclusion for the convergence goal: the redundancy in 33Hz
samples is NOT free to discard — update density AND decision rate
both bind. Fastest safe trainer config remains stock rollout +
train-stride 3 (1.39x), with --reward-per-decision (3x reward_py)
validated-warm but awaiting a scratch seed. act-every 6 (middle dose)
remains the only untested rung; expectations now low.

# ============================================================
# VALUE-CEILING ANALYSIS (2026-08-21): why 1:08 is a LINE problem
# ============================================================

Ran the survey's suggested bound on the champion run
(runs/sISV_par2/traj_8454144000.jsonl ep 9, the filmed 1:19.72) with
the new tools/route_bound.py. Three numbers, all derived from the exact
GoldSrc laws (air accel saturated at aa=100 -> |v'|^2 <= |v|^2 + 900
per tick; PM_ClipVelocity only ever removes energy; gravity does
2*g*dz):

| | time |
|---|---|
| RIGOROUS FLOOR (no ramp loss at all, perfect strafe) | **59.00 s** |
| PRACTICAL FLOOR (same ramp losses, perfect strafe) | **73.66 s** |
| actual champion run | 79.72 s |
| human WR | 68.00 s |

**Verdict: the route CAN host the record - but this LINE cannot, even
with flawless execution.** Perfect strafing on the identical polyline
tops out at 73.66 s, still 5.7 s short of 1:08.

**The energy budget explains why, and rewrites the record strategy:**

    gravity supplied     9,392,448  (specific, per unit mass)
    strafe supplied      1,039,768  (29% of the 450/tick ceiling)
    destroyed at ramps   7,363,534  (71% of ALL energy supplied)

Air-strafing contributes only ~10% of energy input; **ramp-contact
losses (the (v.n)^2 that PM_ClipVelocity deletes) are 7x larger than
every strafe gain combined.** The race is won by not crashing energy
into ramp normals - grazing entries - not by strafing harder. This
independently confirms the survey's physics note (a 45-deg ramp pays
400 ups^2/tick vs air-strafe's 90 at 500 ups).

Sensitivity (perfect strafe + reduced contact loss):
10% -> 72.09 s | 20% -> 70.54 s | 30% -> 69.02 s | **40% -> 67.52 s**.
So 1:08 needs roughly perfect strafing PLUS a ~35-40% cut in ramp
losses. That is a line-geometry problem, i.e. exactly what a savestate
hill-climb search optimizes and what gradient-following PPO plateaus
on.

Where the losses live (deciles of route): **seg 8 (1.37M), seg 10
(1.17M), seg 3 (1.15M) = 48% of all losses**, while the per-decile
gain from perfect strafing is uniform (0.37-0.87 s each, 6.06 s total).
A search arm should attack segments 8, 10 and 3 first.

Model caveats (in the tool docstring): a faster run flies different
ballistic arcs, so "the same route" is a particle-on-a-curve
idealization; the seconds are model-dependent, the ranking of levers is
not. 176 of 7,972 ticks exceed the 450/tick ceiling by a total of 1,704
(0.018% of budget) - discretization at contacts, not an energy source.

**Consequences for the roadmap:** the record chase is now a
line-geometry search problem first and an RL problem second. Reward
knobs (worth ~1-2 s) and trainer speed cannot reach 1:08 on this line.
Priority order becomes: savestate hill-climb polish (survey item 5,
TMInterface/bxt-rs pattern, and note bxt-rs proves searching in
strafe-parameter space is the tractable formulation) -> distill the
found line back via backward gated curriculum -> then the horizon /
lookahead / critic structural fixes for robustness.

# ============================================================
# ROUND 15 (2026-08-21): the strafe-precision finding
# ============================================================

## Tier-0 diagnostics (CPU only, no rental needed)

**1. Strafe audit of the champion run — the action space cannot express
optimal strafing.** Reconstructed wishdir per tick from (yaw, fmove,
smove) and validated the convention against the engine (median
horizontal prediction error 14.7 u/tick over all ticks, <1 u on the 33%
of ticks with no ramp contact; the alternative right-vector convention
gives 83.6, so convention A is confirmed).

What the agent does RIGHT, self-taught:
- vectorial strafing: fmove=0 on 86.5% of ticks (wishdir 90 deg off view
  yaw, so it can look down-track while strafing perpendicular)
- holds A/D and turns the mouse in the SAME tick 81.0% of the time
- textbook direction pairing (D+mouse-right / A+mouse-left) 71.8% of
  held ticks; empirically verified (holding D turns the velocity
  heading clockwise, mean sign -0.61, and the view follows at -0.57)
- median hold 12 ticks = 120 ms, 214 swaps over the run - human cadence
- median |theta - 90| = 0.63 deg, within 5 deg on 85.6% of ticks

What is WRONG - and it is the action space, not the policy:
- the gain window is +-arcsin(30/|v|) = **+-0.52 deg at 3000 u/s**,
  narrower than the agent's own aim error
- only **45.4%** of ticks land inside it; on the rest the engine's
  addspeed <= 0 and air accel applies NOTHING
- net air acceleration over the run is **-7.9% of the 900/tick ceiling**
  - a drag, not a gain. By speed band: 1000-2000 +25.0%, 2000-3000
  -1.8%, **3000-5000 -26.4%**
- mean |mouse turn| while holding a key is 0.923 deg/tick where the
  optimum at 3000 u/s is atan(30/3000) = 0.573; the bin ladder
  {0,.25,.5,1,2,4,7,10} has nothing between 0.5 and 1.0, so it
  over-turns ~60% and pushes wishdir past perpendicular into the
  braking region.

This is the mechanism behind the value-ceiling result's "~6.1 s of
strafe capture" and it is a QUANTIZATION problem, fixed by
reparameterization rather than more training.

**2. Plasticity hypothesis REFUTED.** Weight norms, escaped vs stalled
seeds at matched steps: sIS_b (slow) 94.9 / 119.8 / 142.1 at 0.5/1.0/
1.5e9 vs sIS_long (fast) 104.3 / 129.0 / 148.4 - the STALLED seed has
consistently LOWER norms, the opposite of the prediction. sG999i05
(stalled, 8e9) 262.3 vs champion 269.2 at 8.5e9. Norm growth is a smooth
universal drift, not a differentiator: effective-LR collapse is not our
wall mechanism. Plasticity preventives drop off the convergence list.

## Fix: --yaw-adaptive (commit after 0742758 lineage)

Yaw bin becomes a MULTIPLE k of the analytic optimal-strafe rate:
delta = k * atan(30/|v_h|), clamped to yaw_rate_max, with
K_BINS = {0, +-0.5, +-0.85, +-1, +-1.15, +-1.5, +-2.5, +-4}. "Strafe
optimally" is then the single constant action k=+-1 at EVERY speed.
Verified: delta matches clamp(k*atan(30/v)) to 0.00000 deg; fixed-bin
path unchanged; 16/16 binding tests pass (SurfEnvConfig 124->128 B).
Changes action semantics -> scratch only.

## Round 15 arms (paired, identical code, per-step judged)

| box | arm | delta |
|---|---|---|
| ssh3.vast.ai:18694 (103% bf16) | sYAW | champion recipe + --yaw-adaptive |
| ssh8.vast.ai:10500 (96% bf16) | sCTL | champion recipe, stock bins |

Both 183,500 steps/s (identical to the iteration - adaptive yaw costs
nothing measurable). Primary readout is the MECHANISM: re-run the
capture audit on each arm's eval rollouts and compare
inside-window % and net capture, which should move within ~100M steps
regardless of wall progress.

## Ops notes from this round

- These boxes report affinity 192 but the cgroup quota is 23 CPUs, so
  _default_omp_threads() (half the cores, capped 32) picks 32 against a
  23-core budget. bench_env sweep says 64 is 1.5-1.7x faster THERE
  (0.218 vs 0.333 ms/step) - but in-situ the default beat 64 (183.5k vs
  144k steps/s). The heuristic was tuned in the real loop for exactly
  this reason; do not port an isolated micro-benchmark optimum into it.
  Real fix (queued): read /sys/fs/cgroup/cpu.max, not the affinity mask.
- Throughput on these boxes is ~183k steps/s vs 560-820k on the best
  ones we have rented - 23 CPU cores is the likely binding constraint.
  Rent on effective-core count, not just GPU model.
- `vastai destroy` prompts and SILENTLY ABORTS with exit 0 under a
  non-interactive shell: always pass -y and re-list to confirm.
- pkill self-match struck again: a launch command containing
  "train_fast.py" was killed by its own `pkill -f '[t]rain_fast.py'`
  prologue (bracketing the pattern does not help when the same text
  appears later in the same command line). Launch via an on-box script.
- Rental blocklist now exists: tools/bad_hosts.json + tools/vast_pick.py
  (keyed on machine_id/host_id; record BEFORE destroying).

## CORRECTION (2026-08-21, mid-round-15): gamma is PER TICK, horizon is 20 s

`train_fast.py:1970` computes `g_eff = args.gamma ** K` (K = act_every)
and uses g_eff as the per-DECISION discount in GAE. So `--gamma` is a
per-PHYSICS-TICK number and the effective horizon is
`1/(1-gamma)` TICKS = 2000 ticks = **20 s** for the champion's 0.9995 -
and it does NOT change when act_every changes.

Two claims made earlier in this ledger and in docs/research-litsurvey.md
are therefore WRONG and are retracted:
1. "our effective horizon is 60.6 s, the largest structural outlier vs
   the field's 1.4-13 s". It is 20 s. Still longer than the field, but
   not extreme, and the "collapse the horizon" recommendation is
   correspondingly weaker.
2. "the act-every-9 arm was confounded because keeping gamma 0.9995
   tripled the horizon". It did not - the horizon was 20 s in both. The
   earlier act-every-9 NEGATIVE result stands unconfounded.

Consequence for round 15: the two act-every arms were launched with
"horizon-matched" gammas (0.9985 / 0.999) which in fact SHORTEN the
horizon to 6.7 s / 10 s - the confound I was trying to remove, in the
opposite direction. Relaunched:
- sAE6 = act-every 6 at gamma 0.9995 (isolates decision rate; the one
  untested rung between the champion's 3 and the negative 9)
- the second box now runs sYAW2 = a SECOND SEED of the adaptive-yaw
  treatment instead of a redundant act-every-9 re-test, because n=1
  against this seed variance is unreadable.

## Round 15 fleet (final)

| arm | box | delta vs champion |
|---|---|---|
| sYAW | ssh3:18694 (23 cores) | --yaw-adaptive |
| sYAW2 | ssh2:14856 (48 cores) | --yaw-adaptive, seed 2 |
| sCTL | ssh8:10500 (23 cores) | none - CONTROL |
| sAE6 | ssh8:14858 (96 cores) | --act-every 6 |
| sSTR2 | local | --train-stride 2 |

Mechanism readout at 451M (tools/strafe_audit.py, new): sYAW capture
-425.5% vs sCTL -613.3%, in-window 29.1% vs 28.8%, mean speed 1,377 vs
1,374 u/s. Both far from the trained champion's -7.9% - too early to
call, and track at ~630M is sCTL 26.2k vs sYAW 18.6k (control ahead,
inside the historical seed band 9.6k-27k).

## Cost finding: the 5090 is probably the wrong card for us

Measured on a live training box: **VRAM in use is 4,820 MiB of 32,607**
- the whole workload fits in a 24 GB card with room to spare. Current
listings: RTX 3090 from **$0.113/h**, RTX 4090 from $0.308/h, vs the
$0.371-0.550/h we are paying for 5090s. A sampled nvidia-smi during
training also showed util 9% / 129 W (a single sample; an earlier one
showed 99% / 174 W), consistent with the workload being CPU/python-bound
rather than GPU-bound at our batch size.

Plan (user directive - rent what is most profitable): when the first arm
is killed, replace it with a 3090 running the SAME arm and measure
steps/s, then compare $ per 1e9 steps against the 5090 boxes. If the
3090 lands within ~2x on throughput it is 3-5x cheaper per step and the
whole fleet should move.

## Tools added this round

- tools/strafe_audit.py - air-accel capture from any rollout
- tools/compare_runs.py - arms at matched steps vs the seed band
- tools/route_bound.py - route time floor (earlier today)
- tools/bad_hosts.json + tools/vast_pick.py - rental blocklist

## GENERALIZATION TEST (2026-08-21 ~02:40): the champion has not learned to surf

User hypothesis: "the agent just overfits to pass each individual ramp;
it doesn't have the idea 'surf as far as you can'." Tested with zero
training - record the champion (sISV_par2 ckpt_latest, 8.98e9) from
off-distribution starts on the SAME map:

| start condition | median survival | mean speed |
|---|---|---|
| start platform (its own line) | 80 s, completes 198,380u | 2,901 u/s |
| `--spawn mixed` (dropped 400-800u above a random surfable face) | **3.8 s** | 253 u/s |
| `--spawn ramp` (placed on a random ramp) | **3.1 s** | 306 u/s |

7 of 8 episodes die inside ~4 s in both off-distribution conditions, at
~1/10 the racing speed; exactly one episode of each 8 survived to the
12,000-tick cap. **Hypothesis CONFIRMED.** (Files: runs/gen_mixed.jsonl,
runs/gen_ramp.jsonl. The matched in-distribution recording failed on a
recorder bug - `platform_spawn_pool: no edge-facing-ramp spawn found` -
so the in-distribution row is taken from the training evals; worth
fixing.)

Honest caveat: this is not a subtle overfit, it is a direct consequence
of the training distribution. The champion only ever starts from the
platform (10%) or from a snapshot of ITS OWN recent trajectory (90%), so
literally every other state in the map is off-distribution.

**This is the narrowest start-state distribution of any system in the
survey**, and every superhuman system used a multi-source one:
- Necto: 70% human-replay / 8% smart-random / 4% true start / 18% scripted
- q1physrl (beat a human WR): **99% randomized mid-run states** - random
  yaw over 360 deg, speed 0-700, and randomized time-remaining
- Fuchs GT: agents "uniformly distributed on track at 100 km/h"
- vision-GT: uniform over the course INCLUDING off-course positions
- Swift: random gate + perturbation of a previously observed pass state

So the same design choice plausibly drives BOTH open problems: no
generalization (nothing else was ever trained on) and the seed variance
(each seed carves its own groove and cannot recover from a deviation,
so whether it survives a wall is a lottery).

**Queued (no code needed, one flag): a start-state diversity arm.**
`--respawn-frac 0.6 --spawn mixed` puts ~40% of episodes on randomized
ramp drops instead of the platform, moving us from "narrowest in the
survey" toward the q1physrl/Necto shape. Judge on (a) the same
off-distribution survival test - does median survival move from 3 s
toward tens of seconds - and (b) whether track progress per step holds
up against sCTL. If it costs little on-line and buys robustness, it is
also the most plausible fix for the seed-variance tail, which is the
dominant cost of the 1-hour-convergence goal.

## Round 15 mid-course corrections (~03:00-04:00)

**1. Adaptive yaw v1 was mis-designed - fixed, arm relaunched.**
sYAW trailed sCTL badly (15.4k vs 37.0k track at ~770M, sCTL high in the
9.6k-46.8k seed band, sYAW at the bottom). Cause found by measuring turn
AUTHORITY, not accuracy: v1's K_BINS stopped at |k| = 4, which at 3000
u/s permits only 2.29 deg/tick of view rotation where the stock ladder
permits 10 - a 4x cut in peak turn rate at racing speed. I bought
precision with authority.

Sized the fix from the champion's own behaviour: expressing its per-tick
view turn as a multiple of w* = atan(30/|v|) gives **p50 0.87, p75 1.62,
p90 2.69, p95 4.23, p99 17.0, max 20.0**. So the policy already lives ON
the strafe optimum (p50 ~ 1 is a nice independent confirmation of the
whole premise) but needs ~20x it for the top 1% of corrections - exactly
what v1 removed. New ladder: {0, +-0.5, +-0.75, +-1, +-1.5, +-3, +-8,
+-20}. Relaunched as sYAWv2; both v1 seeds killed (a known-flawed design
is not worth a second seed).

**2. Cost per unit of training measured (user directive: rent what pays).**

| box | steps/s | $/h | **$ per 1e9 steps** |
|---|---|---|---|
| 23-core 5090 (sCTL) | 200,977 | 0.472 | **0.65** |
| 48-core 5090 (sYAW2) | 122,334 | 0.550 | **1.25** - worst |
| 96-core 5090 (sAE6) | 314,573 | 0.547 | 0.48 (act-every 6 inflates steps/s ~2x; ~0.96 adjusted) |
| local 5090 | ~837,000 | - | free |

**Core count does not predict throughput** and neither does GPU model:
the 48-core box was the slowest and dearest, and our LOCAL 5090 is 4.6x
faster than any rented 5090 (no cgroup quota, faster per-core). The
48-core box is blocklisted as cpu_bound and destroyed; the 23-core box is
recorded known_good. Renting a 3090 ($0.109/h, 24 cores) next to measure
whether a cheap card is better value - VRAM in use is 4.8 GB of 32, so
capacity is irrelevant; only HBM/GEMM and the host CPU matter.

**3. Generalization test on a NEW map (user supplied surf_petrus_lite).**
Zones auto-extracted from the BSP (start + end found), caches baked, then
the champion recorded on it cold:

| | cannonball (trained) | petrus_lite (unseen) |
|---|---|---|
| survival | 80 s, completes 198,380u | **median 8.2 s, 8/8 fail** |
| mean speed | 2,901 u/s | **85 u/s** |
| horizontal distance | 231,292u | **48-311u** |

It drops ~1,200u vertically while moving 50-300u horizontally: it does
not attempt to surf, it falls off the start and dies. **Zero transfer**,
as the user predicted. Combined with the same-map off-distribution result
(3 s survival from random ramp drops), the picture is consistent: the
policy is a cannonball-line follower, not a surfer.

Per the user's scope call, this is a DIAGNOSTIC, not a goal - we are not
building a generalist today. What it justifies on cannonball is the
start-state diversity arm (sDIV, local: `--spawn mixed --respawn-frac
0.6`), because the same narrow start distribution is the most plausible
shared cause of the seed-variance tail (convergence) and the brittle
single line (record chase).

## Fleet after corrections

| arm | box | question |
|---|---|---|
| sCTL | ssh8:10500 (23c) | control |
| sYAWv2 | ssh3:18694 (23c) | strafe precision WITH authority restored |
| sAE6 | ssh8:14858 (96c) | act-every 6 |
| sDIV | local | broader start states |
| (pending) | 3090 $0.109/h | cost benchmark, then a second sDIV seed |

Retired: sSTR2 (stride 2) - reached 0.65B at 14.9k track, INSIDE the
9.6k-27.1k seed band at 0.5B, i.e. no signal either way; killed to free
the local GPU for the petrus bake and then sDIV.

## BUG FOUND (2026-08-21 ~04:15): `--spawn mixed` never actually mixed

The env resets by UNIFORM draw from the spawn pool, so pool entry counts
ARE the sampling probabilities (respawn.py says this in its own docstring
- "the env resets by uniform pool draw, so entry counts ARE the
probabilities"). But train_fast built the mixed pool as

    pool = np.concatenate([plat_pool, dp])

where plat_pool is the map's ~18 start entities and dp is a multi-thousand
drop pool. So `--spawn mixed` gave a ~0.7% chance of starting at the start
line: the start was effectively never trained. Measured symptom: the first
sDIV arm reached 104M steps with **30u** of eval track progress while every
other arm at similar steps was in the thousands.

This is a latent bug affecting any historical run launched with
`--spawn mixed` or `--spawn ramp`-adjacent pools, not just tonight's arm -
including the old pre-race "mixed spawn" era runs, whose spawn
distribution was therefore far more drop-heavy than the name suggests.

Fixed: `--drop-frac F` (default 0.5) replicates the start entries to hit
the requested ratio. Verified in the log:
`race: start entries x478 -> drop fraction 0.25 (requested 0.25)`.

sDIV relaunched as `--spawn mixed --drop-frac 0.25 --respawn-frac 0.9`,
i.e. ONLY the pool composition differs from the champion recipe: the 10%
of episodes that do not come from the reservoir are now 25% random ramp
drops / 75% start line. Intrinsic income jumped from ~0.35/ep (champion)
to 9.14/ep, confirming the drops really do reach unvisited cells.

## RTX 3090 viability (2026-08-21 ~05:00)

Card gate: **HBM 839 GB/s, bf16 71 TFLOPS** (5090 reference: 1520 /
233), VERDICT healthy - i.e. 55% of the bandwidth and 30% of the GEMM at
$0.109/h vs $0.472. VRAM is a non-issue (4.8 GB in use of 24).

First launch died silently after `torch.compile: minibatch step compiled
in 120s`, no traceback, no OOM (85 GB free, cgroup `oom 0`). Relaunching
the identical script worked. **Probable cause: deploy_box.sh backgrounds
`pip install scipy numpy` and I launched the arm while that was still
running - swapping numpy under a live process.** Lesson: wait for
/root/pip.log to finish before launching, or launch through
tools/fleet_add.py which sequences it.

Ampere note: inductor autotuning rejects a number of configs with
`OutOfResources: shared memory Required 131072/147456, Hardware limit
101376` - Ampere has 101 KB of shared memory where Ada/Blackwell have
more. These are per-config rejections ("Ignoring this choice"), not
fatal; compilation completes and the CUDA graph captures fine.

sDIV2 (second spawn-diversity seed) now training on it.

## Robustness comparison + sDIV verdict (~05:30)

Off-distribution test (spawned on random ramps, 8 greedy episodes,
`--spawn ramp`), median survival / mean speed / episodes over 10 s:

| arm | median | speed | >10 s |
|---|---|---|---|
| sCTL (control) @1.00e9 | 2.25 s | 251 u/s | 0/8 |
| sDIV (25% drops) @~0.8e9 | 3.31 s | 243 u/s | 1/8 |
| champion @8.96e9 | 3.08 s | 306 u/s | 1/8 |

**sDIV verdict: not paying off as configured.** It buys ~1 s of
off-line survival while costing 7x on the primary metric (6.7k track at
818M vs sCTL's 47.3k at 1.07B, and sCTL is ABOVE the historical band).
Note also that the fully-trained champion scores no better than a 1e9
control on this test, which says the random-ramp-drop metric is close to
a floor for everything we have - a 400-800u free fall onto a ramp with
randomized velocity may simply be too adversarial to discriminate. A
fairer robustness probe would place the agent ON a ramp already carrying
speed, off its line. Killed to free the local box.

## Arm status at ~05:30

| arm | steps | track | vs reference band |
|---|---|---|---|
| sCTL (control) | 1.07e9 | **47,297** | ABOVE band (31.6-43.6k @1e9) |
| sYAWv2 (yaw v2) | 256M | 12,048 | top of band (8.9-12.2k @0.25e9) |
| sAE6 (act-every 6) | 713M | 13,815 | low in band |
| sDIV | 818M | 6,745 | below band - KILLED |

sYAWv2 with authority restored sits at the TOP of the band early, versus
v1 which sat at the bottom - consistent with the K_BINS diagnosis. Local
(fastest box we have, ~600-830k steps/s vs 60-200k rented) now runs
**sYAWb**, a deep second seed of yaw v2 with a 12e9 budget: at local
speed that can reach the champion's ~5.4e9 finisher threshold before
morning, which is the fastest available test of whether the strafe fix
actually helps rather than just starting well.

## Yaw v2 MECHANISM result (~05:50) - the fix works on its own terms

Track progress is a noisy proxy; air-accel capture measures the thing
--yaw-adaptive was built to change. Matched-step comparison of the two
arms' greedy eval recordings at 151M steps (tools/strafe_audit.py):

| | sYAWv2 (adaptive) | sCTL (stock) |
|---|---|---|
| air-accel capture | **-123.5%** | -245.8% |
| median \|theta-90\| | **16.38 deg** | 29.06 deg |
| within 5 deg of perpendicular | 37.7% | 30.7% |
| ticks inside the gain window | 22.4% | 21.4% |
| mean speed | **1,094 u/s** | 1,002 u/s |
| mean \|yaw delta\| | 5.09 deg/tick | 4.70 deg/tick |

Half the speed destroyed, 44% better aim, 9% faster, at identical
training steps. Both are still deeply negative because these are 151M-
step policies (the trained champion is at -7.9%), but the ORDERING is
the point and it matches the design intent. Combined with sYAWv2 sitting
at the top of the reference band at 0.25e9 where v1 sat at the bottom,
the K_BINS diagnosis looks correct.

Caveat: n=1 per arm. sYAWb (local, deep 12e9 budget) is the second seed.

Tool bug fixed while doing this: strafe_audit wrapped yaw into
[-180,180) BEFORE np.diff, so every 179 -> -179 crossing counted as a
358 deg turn and mean_turn read 19.85 deg/tick against a 10 deg/tick
clamp. Capture / in-window / theta were unaffected (they use v.wishdir
directly, not the difference).

## act-every 6 verdict + third yaw seed (~06:15)

sAE6 (act-every 6, gamma 0.9995 so the 20 s horizon is unchanged) at
880M steps: track 10,415 -> 12,702 -> 13,815 -> 14,995, i.e. climbing
but INSIDE-AND-LOW against the 9,582-46,784 band at 0.75e9, while the
control sCTL reached 47,297 by 1.07e9. Combined with the earlier
act-every 9 result (negative, and now known to be UNCONFOUNDED since
gamma is per-tick), **the decision-rate axis is closed: 33 Hz is not
too fast, and coarsening it costs learning without buying enough
compute back.** Retired; the box now runs sYAWc, a third yaw-v2 seed.

Yaw v2 now has three independent seeds - sYAWv2 (rented 23c), sYAWb
(local, deep 12e9 budget, ~591k steps/s), sYAWc (rented 96c) - against
sCTL plus the historical band. That is the first time in this project a
treatment has been run at n=3, which is what the seed variance demands.

## record_ckpt bugs fixed (~06:10)

1. `--spawn platform` on a race ckpt fell through to
   `platform_spawn_pool`'s walk-off-the-edge audition and raised "no
   edge-facing-ramp spawn found". A race run trains from the map's start
   entities, so it now uses race_start_pool there.
2. `mixed` had the SAME uniform-draw swamping bug as the trainer: a
   handful of start entries concatenated with thousands of drops is not
   a mix. Now replicated to a real 50/50.

Consequence for the generalization test: the "mixed" and "ramp"
conditions were effectively the same test, which is why they scored
3.8 s and 3.1 s. The no-transfer conclusion is unaffected. With the fix,
a properly matched champion comparison from the same recorder:

| champion @8.96e9 | median survival | mean speed |
|---|---|---|
| in-distribution (start line) | **83.01 s** | 2,629 u/s |
| off-distribution (ramp drops) | **3.08 s** | 306 u/s |

## Matched-step comparison of the live arms (~06:25)

| @steps | 0.15e9 | 0.30e9 | 0.50e9 | 1.00e9 |
|---|---|---|---|---|
| sCTL (control) | 150 | 8,740 | 17,048 | **46,086** |
| sYAWv2 | **882** | **12,048** | 16,564 | - |
| sYAWb (local) | 821 | **12,497** | - | - |
| reference band | 4,984-5,184 | 9,587-15,127 | 9,554-27,076 | 31,559-43,557 |

Both yaw seeds lead the control early (12.0-12.5k vs 8.7k at 0.30e9)
then converge to parity by 0.50e9. **Weak positive, NOT conclusive** -
the decisive stretch is past 1e9 and sYAWb reaches it first at local
speed. Note the control itself started unusually slow (150 at 0.15e9 vs
the band's ~5,000) and then finished ABOVE the band at 1e9, which is a
good reminder of how little the early numbers determine.

## Record-goal probe: can the yaw fix be grafted onto the champion? (~06:35)

The strafe fix only helps runs trained under it (it changes action
semantics), and a scratch run cannot reach finisher level tonight. But
the champion already IS a finisher, and the value-ceiling analysis says
perfect strafe capture on its line is worth ~6.1 s (79.72 -> 73.66).
So: warm-start the champion WITH --yaw-adaptive. The action head's
meaning changes (bin i now means k_i * atan(30/|v|) rather than a fixed
deg/tick) so the policy's action preferences are wrong on arrival, but
the CNN features, the value function and the novelty table all transfer.
If it re-converges in a fraction of the ~5.4e9 a scratch run needs, that
is the fast path to a record attempt; if it collapses to scratch-level,
the fix is scratch-only and the record chase needs the search arm
instead.

Running on the 3090 (cheapest box, and the diversity seed it replaces
was already judged unpromising). Champion ckpt (8.96e9) transferred.

## Night summary (2026-08-21, ~00:00-07:00) - verdicts

**FOUND: the action space cannot express optimal strafing.** At
sv_airaccelerate 100 the air impulse is saturated, so the only
speed-maximizing wishdir is exactly perpendicular and the usable window
is +-arcsin(30/|v|) = **+-0.52 deg at 3000 u/s**. The champion aims to a
median 0.63 deg - better than the window - yet lands INSIDE it on only
45% of ticks and its net air acceleration over a full run is **-7.9% of
the 900/tick ceiling: a drag, not a gain** (-26% in the 3000-5000 band).
Cause: the yaw ladder {0,.25,.5,1,2,4,7,10} deg/tick has nothing between
0.5 and 1.0 while the optimum at racing speed is 0.573, so it over-turns
~60% and pushes wishdir past perpendicular into the braking region.
This is quantization, not training, and it accounts for the ~6.1 s the
value-ceiling analysis attributed to strafe capture.

**FIX: --yaw-adaptive** (yaw bin = k * atan(30/|v_h|), so "strafe
optimally" is the constant action k=+-1 at every speed). v1 capped k at
4 and lost turn AUTHORITY (2.3 deg/tick at 3000 u/s vs the stock 10);
v2's ladder spans +-20, sized from the champion's own turn distribution
(p50 0.87x w*, p95 4.2x, p99 17x, max 20x). Mechanism confirmed at
matched steps: capture -123.5% vs the control's -245.8%, median aim
error 16.4 vs 29.1 deg, mean speed 1,094 vs 1,002 u/s. Running at n=3
seeds; final verdict pending.

**CLOSED: the decision-rate axis.** act-every 6 (14,995 at 880M, low in
band) and act-every 9 (earlier, negative) both underperform the control.
33 Hz is not too fast. The earlier "act-every 9 was confounded by the
horizon" claim is RETRACTED - gamma is per tick and the trainer applies
gamma**act_every itself, so the horizon is 20 s regardless of K.

**CLOSED (as configured): start-state diversity.** 25% random ramp drops
bought ~1 s of off-line survival (3.31 s vs the control's 2.25 s) while
costing 7x on track progress. Note the 8.96e9 champion scores no better
than a 1e9 control on that probe, so the metric is near a floor for
everything we have.

**REFUTED: the plasticity hypothesis.** Stalled seeds have LOWER weight
norms than escaped ones at every matched step.

**CONFIRMED (user hypothesis): no generalization.** Champion on its own
line 83.0 s / 2,629 u/s; off its line 3.1 s / 306 u/s; on an unseen map
(surf_petrus_lite) 8/8 fail, median 8.2 s, 85 u/s, moving 48-311u
horizontally while dropping 1,200u. Diagnosed as the narrowest
start-state distribution in the survey (90% self-snapshot + 10%
platform).

**BUGS FIXED (3):** `--spawn mixed` never mixed - the env resets by
uniform pool draw and ~18 start entries were concatenated with thousands
of drops, so the start line was trained ~0.7% of the time (latent,
affects historical runs too); the same bug in record_ckpt; and
`record_ckpt --spawn platform` was broken for race ckpts.

**COST:** $ per 1e9 steps - 3090 $0.51, 23-core 5090 $0.65, 48-core 5090
$1.25. Throughput tracks the bf16 GEMM ratio almost exactly (3090 at
0.29x throughput vs 0.30x GEMM), so **the trainer is GEMM-bound** and
bandwidth/VRAM are not the constraint. GPU model moves value ~1.3x while
WHICH BOX you land on moves it 2.5x - measure per box, do not shop by
model. Local (unquota'd) is 3-10x faster than any rental and free.

**TOOLS ADDED:** route_bound.py (route time floor), strafe_audit.py
(air-accel capture), compare_runs.py (matched-step vs seed band),
bad_hosts.json + vast_pick.py (rental blocklist), bench_box.sh,
fleet_add.py.

## Warm-graft probe: first result (~07:00)

Champion (8.96e9, a stable full-map finisher at 198,380u) resumed WITH
--yaw-adaptive, i.e. the same weights under a re-parameterized action
space. First greedy eval immediately after the graft:

**28,605u of 198,380** - an 86% collapse, as expected: every action bin
now means something different, so the policy's learned action
preferences are wrong on arrival.

But the informative comparison is against SCRATCH, not against itself: a
scratch run under this recipe sits at 150-880u at 0.15e9 and needs
~5.4e9 to become a finisher. The graft starts at 28,605u, i.e. it
retains ~30x more capability than starting over, which says the CNN
features and value function transfer even though the action head does
not. If it climbs back to ~198k within a few hundred million steps, the
strafe fix can be applied to an already-good policy and the record chase
does not have to wait for a fresh 5.4e9 run. Evals every 100e6
(~20-28 min on this box); recovery curve is the measurement.

## CORRECTION (~07:20): we are CPU-BOUND on rented boxes, not GEMM-bound

Earlier tonight I compared a 3090 running a SCRATCH arm at ~30M steps
(59k steps/s) against a 5090 running a MATURE arm at ~1e9 (201k steps/s)
and concluded "throughput tracks the bf16 GEMM ratio, so the trainer is
GEMM-bound". That comparison was phase-confounded: scratch runs die
constantly, and the resets/reward-python/respawn bookkeeping are CPU
work, so early training is far slower than late training on the SAME box.

Measured properly, both boxes mature, same 110 s window:

| box | steps/s | $/h | **$ per 1e9** | GPU util | power |
|---|---|---|---|---|---|
| RTX 3090, 24 cores | **235,930** | 0.109 | **$0.13** | **99%** | 224 W |
| RTX 5090, 23 cores | 207,332 | 0.472 | $0.63 | **10%** | 147 W |

**The 3090 is FASTER than the rented 5090 for our workload, at 23% of
the price - 4.8x better value.** The 5090's GPU is idle 90% of the time:
the host CPU is the bottleneck. That also explains the 48-core box being
the worst ($1.25) and the local machine (fast desktop CPU, no cgroup
quota) being 3-4x faster than any rental.

**Rent for CPU, not GPU.** Retracted: the "GEMM-bound" claim and the
"3090 is only 1.27x better value" conclusion that followed from it.

Consequence: the same budget buys ~4x more parallel experiments on
3090s. The right use is SEEDS - every verdict in this project so far is
n=1 against a band that spanned 9.6k-27k track at identical steps. A
12-box 3090 fleet at $1.31/h (less than tonight's 4-box 5090 spend)
would give n=4 on three conditions simultaneously, which is the first
statistically meaningful comparison this project could make.

## Fleet at handoff (~07:30)

Acting on the corrected cost finding (rent CPU, not GPU) and the user's
standing authorisation to scale if the 3090 worked:

| arm | box | $/h | role |
|---|---|---|---|
| sYAWb | local | free | yaw fix, seed 2 (deep, 12e9 budget, ~650k steps/s) |
| sYAWv2 | 5090 23c ssh3:18694 | 0.446 | yaw fix, seed 1 |
| sYAWc | 5090 96c ssh8:14858 | 0.562 | yaw fix, seed 3 |
| sCTL | 5090 23c ssh8:10500 | 0.472 | control, seed 1 (1.25e9, 54,013 track) |
| sCTL2 | 3090 ssh2:19496 | 0.129 | control, seed 2 |
| sCTL3 | 3090 ssh9:19498 | 0.129 | control, seed 3 |
| wGRAFT | 3090 ssh7:16568 | 0.109 | champion + yaw fix, warm graft |

~$1.85/h for 6 rented boxes - barely above the 4-box 5090 spend, because
five of them are now cheap CPU-adequate cards. Design: yaw n=3 vs control
n=3, which is the first balanced comparison this project has run.

Open questions for the next session:
1. **wGRAFT recovery curve** - does 28,605u climb back toward 198,380u in
   a few hundred million steps? If yes, the strafe fix can be applied to
   an existing finisher and the record attempt does not need a fresh
   5.4e9 run. This is the highest-value watch.
2. **yaw n=3 vs control n=3** at matched steps past 1e9.
3. The record path per the value-ceiling analysis still needs a
   line-geometry search (savestate hill-climb); reward knobs are spent.

## HEADLINE (~07:50): the strafe fix grafts onto the champion in ~200M steps

wGRAFT recovery curve (champion weights, re-parameterized action space):

| steps after graft | eval track |
|---|---|
| 0 | 28,605u (14%) |
| +100M | 132,864u (67%) |
| +200M | **180,398u (91%)** |

A scratch run needs ~5.4e9 to become a finisher. **The graft is at 91%
of the map after 200M steps - roughly a 25x saving** - and is training
with rew 25.97 and 2,978-tick episodes at 9.21e9. This answers the
question the probe was built for: the strafe fix does NOT require a
fresh scratch run, so it can be applied to the existing finisher and a
record attempt does not have to wait hours.

## And the fix is winning on scratch runs too

| @steps | sYAWb (yaw) | sYAWv2 (yaw) | sCTL (control) | reference band |
|---|---|---|---|---|
| 0.25e9 | 821 | **12,048** | 8,740 | 8,941-12,228 |
| 0.50e9 | 12,497 | **21,184** | 17,048 | 9,554-27,076 |
| 0.75e9 | 22,068 | **27,183** | 26,199 | 9,582-46,784 |
| 1.00e9 | **63,860** | - | 46,086 | 31,559-43,557 |
| 1.25e9 | 63,860 | - | 54,013 | 47,067-64,234 |

Both yaw seeds lead the control at every overlapping step, and sYAWb at
1e9 (63,860) is above the entire historical band. Two independent seeds
agreeing is a real signal; still provisional until sYAWc catches up and
the three control seeds establish the spread.

## sYAWb vs the champion lineage (~08:20) - the convergence goal

| @steps | sYAWb (yaw fix) | sIS_long (champion lineage) |
|---|---|---|
| 1.0e9 | 46,672 | 43,557 |
| 1.5e9 | **70,547** | 57,941 |
| 2.0e9 | **114,449** | 98,404 |
| 2.5e9 | **154,144** | 86,899 |
| 3.0e9 | - | 93,576 |
| 3.5e9 | - | 99,061 |

**At 2.5e9 the yaw-fix seed has already passed what the reference seed
reached at 3.5e9**, and the reference plateaued at 87-99k from 2.5e9
through 4.5e9 while sYAWb is still climbing. If it holds, this is the
convergence goal moving: the champion needed ~5.4e9 to finish.

Caveats, stated plainly: n=1 at this depth (sYAWv2 986M/36,541 and
sYAWc 581M/19,203 are behind), sIS_long is itself one seed, and the
actual champion sISV_par2 has no logged curve below 3.5e9 to compare
against. The control seeds are filling in - sCTL 1.80e9/71,509,
sCTL2 302M/12,553, sCTL3 430M/13,799.

wGRAFT has stalled at the champion's old final wall: 180,398 -> 175,667
-> 176,235 across three evals at 9.16-9.54e9. That is the same
last-two-ramps barrier (flight AGAINST the potential gradient) the
original champion had to break, so the graft has recovered the whole map
except the part that was hardest the first time.

## n=3 vs n=3 at matched steps (~08:50) - tempering the earlier claim

| @steps | yaw seeds | control seeds | medians |
|---|---|---|---|
| 0.50e9 | 12,497 / 21,184 / 19,203 | 17,048 / 13,169 / 14,786 | 19,203 vs 14,786 (+30%) |
| 0.75e9 | 22,068 / 27,183 / 31,864 | 26,199 / 11,851 / 16,593 | 27,183 vs 16,593 (+64%) |
| 1.00e9 | 46,672 / 36,541 / 41,849 | 46,086 / - / 24,714 | 41,849 vs ~35,400 (+18%) |

The yaw group's median leads at every matched step, but **the ranges
overlap substantially** and the effect is nothing like the 1.5-2x the
single best seed suggested. **The earlier "sYAWb is ahead of the
champion lineage" framing was built on the BEST of three yaw seeds**;
sYAWv2 (42,370 at 1.25e9) is behind sCTL (54,013 at 1.25e9) and sYAWc
tracks the control closely. Corrected reading: a consistent but modest
advantage, plausibly real given it holds across three matched-step
comparisons and is backed by an independent MECHANISM measurement
(capture -123.5% vs -245.8%, aim error 16.4 vs 29.1 deg), but not the
step-change the best seed implied.

This is exactly the failure mode the seed-variance work was meant to
prevent, and it is worth recording that the project's whole history of
n=1 verdicts is subject to it: sCTL2 alone moved 13,169 -> 11,851
between 0.5e9 and 0.75e9, i.e. a single seed's eval noise spans a range
comparable to the effect being measured.

## CORRECTION 2 (~09:20): the yaw seed is BEHIND the real champion

I compared sYAWb against sIS_long and called it "ahead of the champion
lineage". sIS_long is NOT the champion - it is a weaker seed of a
related recipe that plateaued at 91-99k. Against sISV_par2, the actual
champion, on total global steps:

| @steps | sYAWb (yaw) | sISV_par2 (champion) | sIS_long |
|---|---|---|---|
| 3.5e9 | 156,133 | - | 99,061 |
| 4.0e9 | 137,030 | **181,996** | 98,665 |
| 4.5e9 | 138,346 | **193,971** | 91,054 |
| 5.0e9 | 146,893 | 176,929 | 91,054 |
| 5.5e9 | 143,011 | 176,893 | - |
| 6.0e9 | 155,545 | **188,938** | - |

**sYAWb trails the champion at every matched step from 4e9 on.** It sits
BETWEEN the weak reference seed (91-99k) and the champion (177-194k) -
which is what ordinary seed variance looks like, not a treatment effect.

Corrected standing on --yaw-adaptive:
- MECHANISM: measured and real (capture -123.5% vs -245.8%, aim error
  16.4 vs 29.1 deg at matched steps). This is a direct measurement of
  the thing the change was designed to alter.
- OUTCOME: **unproven.** Group medians favour yaw at 0.5-1.0e9
  (+30/+64/+18%) with heavily overlapping ranges, and at depth the best
  yaw seed is behind the champion. The historical seed spread at 4.5e9
  alone is 91k to 194k - larger than any treatment effect measured
  tonight.

Process note, recorded because it happened twice in one night: both
overclaims came from picking a favourable reference out of a
high-variance set (first the best of three treatment seeds, then a weak
control seed). With this much seed noise, only pre-declared group
comparisons at matched steps mean anything.

## n=3 vs n=3 at depth (~09:50): the separation becomes clean

| @steps | yaw seeds (sYAWb/sYAWv2/sYAWc) | control seeds (sCTL/sCTL2/sCTL3) | medians |
|---|---|---|---|
| 1.0e9 | 46,672 / 36,541 / 41,849 | 46,086 / 12,694 / 30,142 | 41,849 vs 30,142 (+39%) |
| 1.5e9 | 70,547 / 47,196 / 78,087 | 51,735 / 13,686 / 42,443 | 70,547 vs 42,443 (+66%) |
| 2.0e9 | 114,449 / 90,468 / 97,052 | 83,525 / 14,128 / 48,046 | 97,052 vs 48,046 (+102%) |

**At 2.0e9 the yaw group's WORST seed (90,468) exceeds the control
group's BEST (83,525) - complete separation with no overlap.** It holds
after discarding sCTL2, a severe outlier stuck at ~14k since 1e9. The
gap widens monotonically with depth (+39% -> +66% -> +102%), which is
what a real effect looks like as opposed to early-training noise.

How this squares with CORRECTION 2 (sYAWb trailing sISV_par2 at 4e9+):
sISV_par2 is the single best seed of roughly eight parallel streams in
the original campaign, so comparing our best-of-3 against their
best-of-8 is not a like-for-like test. The controlled comparison is the
one above - same code, same night, same hardware classes, three seeds
each - and it separates cleanly. The champion comparison remains the
honest ceiling reference, and sYAWb has not reached it.

Standing on --yaw-adaptive, revised again with the new data:
- MECHANISM: real (capture -123.5% vs -245.8%, aim 16.4 vs 29.1 deg).
- OUTCOME vs matched controls: **separation at n=3 by 2e9**, widening.
- OUTCOME vs the all-time best seed: not reached.

## Graft stalls at the final curtain (~09:55)

wGRAFT reaches 195,363 of 198,380 (98.5%) but has **0 finishes and
win 0.00%** across its whole run - it oscillates 133k-195k, consistently
stopping ~3,000u short. That is the champion's original last wall (the
final two ramps, where speed must be gained flying AGAINST the potential
gradient), so the graft recovered the entire route except the segment
that required a curriculum breakthrough the first time.

Budget extended 12e9 -> 20e9 and relaunched from ckpt_latest (which is
written every 60 s, so nothing was lost) as wGRAFT2.

**Ops note - the pkill self-match bit me a THIRD time tonight.** The
kill `pkill -f '[r]un wGRAFT'` was issued in the same ssh command line
that also contained `--run wGRAFT` inside the heredoc for the new
launcher, so the pattern matched its own shell and killed it before the
script was written. Bracketing only protects when the pattern text
occurs ONCE in the command. Rule going forward: **the kill goes in its
own ssh call, and the launch in a separate one** - never in the same
command line as the run name being killed.

## n=3 at 2.5e9 (~10:15): read the margin structure, not just the median

| | yaw seeds | control seeds |
|---|---|---|
| values | 154,144 / 97,784 / 99,093 | 94,097 / 14,028 / 40,129 |
| median | **99,093** | 40,129 (+147%) |
| worst vs best | 97,784 | 94,097 (+4% only) |

The median gap is large but **driven substantially by two weak control
seeds**: sCTL2 has been frozen at ~14,028 since 1.5e9 (wall-stuck) and
sCTL3 DECLINED 48,046 -> 40,129. The best control seed (94,097) is
within 4% of two of the three yaw seeds, so this is not the clean
"every treatment beats every control" picture the 2.0e9 snapshot
suggested - at 2.0e9 the min-max margin was 90,468 vs 83,525 (+8%) and
at 2.5e9 it has narrowed to +4%.

The pattern that may matter more than the medians: **0 of 3 yaw seeds
are stuck, versus 2 of 3 control seeds weak or stuck.** If the fix
mainly lowers the probability that a seed jams at a wall, that IS the
convergence benefit being sought, and it is consistent with the
mechanism (better capture -> higher speed -> better odds of clearing a
speed-gated jump). With n=3 this is suggestive, not established; it
would need ~8-10 seeds per arm to measure a stick-rate difference
properly, which at 3090 prices ($0.13 per 1e9 steps) is now affordable.

## PATH TO THE RECORD: search beats the policy on a segment (~10:40)

Built tools/tas_search.py - a vectorized savestate hill-climb. Restore
the exact physics state at tick t0 of a recorded run, then search action
sequences over the next W ticks and compare against what the policy
actually did. No network is involved, so a win is a statement about the
map and the physics, not about the agent.

The advantage we have over the TrackMania bruteforcers who invented this
technique: our env is vectorized, so a population of 2048 mutations is
evaluated in ONE batched W-tick rollout instead of one candidate at a
time.

**Validation: replaying the inferred action bins from a restored state
reproduces the recording with 0u error.** (fwd/side/buttons are recorded
verbatim; the yaw BIN is recovered from the realized per-tick delta.
Pitch is not reconstructed and cannot matter - env.c passes pitch 0 into
the physics, so the view pitch only aims the depth camera.) That zero
error also re-confirms bit-exact determinism.

**Result on a 2.00 s segment of the 1:19.72 run (ticks 3000-3200):**

| | distance-to-finish at window end |
|---|---|
| recorded (champion policy) | 123,240u |
| search, 8 gens x 512 cands | 123,077u (+163u) |
| search, 60 gens x 2048 cands | **122,692u (+548u)** |

+548u at ~2,900 u/s is worth **~0.19 s on a single 2-second segment**,
found by a trivial hill climb that only mutates the yaw bin and the
sidemove. The champion's run is ~80 s, i.e. ~40 such segments. Naive
extrapolation is ~7 s, which would be 79.72 -> ~73 s - and that is
before mutating fwd/jump, before tuning the mutation schedule, and
before letting segments re-plan jointly.

Treat the extrapolation as an upper-bound sketch, not a projection:
segment gains will not compose linearly (a faster entry changes the
ballistics of everything downstream, which is exactly why the practical
floor of 73.66 s was computed on the ORIGINAL line). But the
qualitative answer is now measured rather than argued: **the champion's
line is not locally optimal, and a search can find better inside it.**

Next step for the record chase: sweep t0 across the whole run to find
where the biggest per-segment gains are (expect them to concentrate in
route deciles 8/10/3, which hold 48% of all ramp-contact energy loss),
then chain the improved segments and re-verify end to end.

## CORRECTION 3 (~10:55): the "complete separation" was partly stale data

compare_runs carries the latest row at-or-before each mark, so an arm
that had not yet reached a mark showed its most recent value there. My
2.5e9 reading used sYAWc's 2.29e9 value (99,093) as if it were its
2.5e9 value; the real one is 87,887. With every arm now past 2.5e9:

| @steps | yaw min/med/max | control min/med/max | min>max? |
|---|---|---|---|
| 2.0e9 | 90,468 / 91,136 / 114,449 | 14,028 / 48,046 / 83,525 | **yes** |
| 2.5e9 | 87,887 / 97,784 / 154,144 | 12,756 / 40,129 / 94,097 | no |
| 3.0e9 | 97,859 / 123,209 / 138,970 | 45,487 / 74,230 / 102,972 | no |

So the clean separation held at 2.0e9 only. What survives across all
marks:
- **median**: yaw leads everywhere (+39% at 1.0e9 to +147% at 2.5e9),
  but the control median is dragged down by sCTL2, stuck at ~13k.
- **best vs best**: yaw 114k/154k/139k vs control 84k/94k/103k at
  2.0/2.5/3.0e9 - yaw ahead by 35-64% consistently. This is arguably the
  operative comparison, since in practice one runs several seeds and
  keeps the best.
- **stick rate**: 0/3 yaw seeds stuck, 2/3 control seeds weak or stuck.

That is the third correction tonight, all of the same shape: reading a
favourable number out of a noisy set before it was solid. The pattern is
worth more than any single number here - with this much seed variance,
claims should wait for every arm to actually reach the mark being
compared.

## Segment sweep + cross-validation (~11:20): this line tops out at ~72-74 s

Light search (20 gens x 512 candidates) at eight branch points across
the 1:19.72 run, window 200 ticks. **Every replay validated at 0u
error**, i.e. bit-exact determinism confirmed the length of the run.

| t0 | recorded progress | search gain |
|---|---|---|
| 500 | 3,858u | +138u |
| 1500 | 5,272u | +266u |
| 2500 | 4,969u | +117u |
| 3500 | 4,435u | +232u |
| 4500 | 5,170u | +47u |
| 5500 | 6,440u | +251u |
| 6500 | 7,464u | +90u |
| 7500 | 2,835u | +346u |

Mean +186u per 2 s segment. Geodesic progress rate is 198,380u/79.72s =
2,488 u/s, so each segment gain is worth ~0.075 s; over ~40 segments
that is **~3.0 s from a light search (-> 76.7 s)**, or **~7.5 s if the
deeper 60x2048 setting's 2.5x factor holds (-> 72.3 s)**.

**Cross-validation:** the energy audit independently put perfect strafe
capture on this same line at **73.66 s**. Two unrelated methods - one
thermodynamic (what the engine can add and what ramp contacts destroy),
one empirical (what a search actually finds) - agree the champion's
route tops out around **72-74 s**. The human WR is 68.0 s.

**Conclusion for the record chase:** ~6-7 s is available on the existing
line through better execution, which would beat our own 1:19.72
decisively but still miss 1:08 by 4-6 s. Beating the human record
requires a different ROUTE, not just cleaner driving of this one -
consistent with the sensitivity analysis showing 1:08 needs a 35-40%
cut in ramp-contact energy loss.

Prediction that FAILED: I expected the searchable gains to concentrate
in route deciles 8/10/3, which hold 48% of all ramp-contact energy loss.
They do not - gains are spread fairly evenly (47-346u) with the largest
at t0=7500 (near the finish), 1500 and 5500. Where energy is LOST is
apparently not where slack is easiest to FIND; the losses may be
geometrically forced by the route, which is itself evidence for the
"needs a different route" conclusion.

## sYAWb finished its 12e9 budget - and did NOT become a finisher (~11:40)

12,000,165,888 steps at 662,095 steps/s average (local, uncontended).
Final evals 146,635 / 156,837 / 153,080 - it has oscillated in the
146-157k band for billions of steps without passing wall 3 (~180k).

**The champion lineage passed 182k by 4e9 and 194k by 4.5e9. Tonight's
best seed, with the yaw fix and 12e9 of training, sits at ~153k.**

This is an open discrepancy worth naming rather than explaining away.
All six of tonight's scratch seeds - three yaw, three control - have
plateaued in the 90-157k band. The launch line matches the ledger's
recorded champion command exactly (gamma 0.9995, int-coef 0.25,
int-speed 3, int-view 8, respawn-frac 0.9, respawn-speed 1.0 1.5,
maxvel 4000, 64x32, act-every 3, pitch-rate 1.33, spawn platform,
ep_ticks default 12000 for race, stall-secs default 15), and tonight's
code changes are all opt-in flags that default to the old behaviour.

Candidate explanations, none verified:
1. Seed luck. The original campaign ran ~8 parallel streams and
   sISV_par2 was its best; the ledger's "4/6 break rate" referred to
   wall #1 (47k), which ALL six of tonight's seeds cleared easily.
   Passing walls 3-4 may always have been the rare event.
2. sISV_par2's curve may include inherited progress: its progress.csv
   starts at 3.5e9, consistent with a branch off an earlier stream, so
   "5.4e9 from random weights" may understate what a truly fresh run
   needs.
3. Something environmental in tonight's boxes that does not show up in
   throughput or health checks.

Explanation 2 is the cheapest to check next session: find whether any
pre-3.5e9 sISV progress.csv survives, and if so compare its curve to
tonight's controls at matched steps. Until then, "the champion needs
~5.4e9 from scratch" should be treated as unconfirmed.

## Deep search confirms the ceiling (~11:55)

t0=7500, window 300 ticks (3.00 s), 150 generations x 2048 candidates,
p_mutate 0.04:

| effort | gain |
|---|---|
| light (20 gens x 512, 2 s window) | +346u |
| deep (150 gens x 2048, 3 s window) | **+617u** |

+617u over 3.00 s at 2,488 u/s of geodesic progress = **0.248 s, an 8.3%
time saving on that segment**. Applied across the run: 79.72 x 0.917 =
**~73.1 s**.

Three independent estimates of what the champion's LINE can do:

| method | result |
|---|---|
| energy audit (perfect strafe, same ramp losses) | 73.66 s |
| light-search extrapolation x2.5 depth factor | 72.3 s |
| deep-search segment rate applied to the run | 73.1 s |
| **human WR** | **68.00 s** |

They agree to within a second. **The route is worth ~72-74 s and no
more; 1:08 requires a different route.** That is now the firmest
quantitative statement this project has about the record, and it came
from two unrelated methods plus a depth check.

## Ops failure: a "--dry-run" rented a real box (~12:05)

A fleet audit found a 7th instance nobody was using. Origin:
`fleet_add.py sTEST --gpu RTX_3090 --max-dph 0.15 --dry-run -- ...`
actually created instance 48257906 and launched a run. Cause:
`extra` was declared `nargs=argparse.REMAINDER`, which fills the
positional with EVERYTHING after the first positional - so --gpu,
--max-dph and --dry-run were swallowed into it and never parsed. The
launched command still contained `--dry-run`, train_fast exited on
"unrecognized arguments", and the box sat idle billing ~$0.126/h for
about three hours (~$0.40).

Two lessons worth more than the money:
1. `nargs=REMAINDER` silently eats sibling flags. Fixed with
   parse_known_args, and the dry-run path is now actually exercised.
2. **A launch that "succeeds" is not a launch that runs.** The existing
   LAUNCH VERIFICATION rule (grep the log for errors AND count trainer
   processes) would have caught this at the time; I skipped it because
   the command was "only a dry run". Verification is cheapest exactly
   when you believe you do not need it.

Fleet audit is now part of the routine: `vastai show instances` cross-
checked against `pgrep -af '[t]rain_fast'` on every box, so an idle
rental cannot hide.

## 4e9 comparison (~12:25): yaw arms approach the champion's curve

| @4.0e9 | values |
|---|---|
| yaw seeds | 137,030 (sYAWb) / **174,179** (sYAWv2) |
| control seeds | 105,117 (sCTL) / 84,883 (sCTL3) |
| champion sISV_par2 | 181,996 |

Yaw min (137,030) again clears control max (105,117), and sYAWv2 at
174,179 is **within 4% of the champion at the same step count** - the
closest any scratch seed tonight has come, and a marked change from
sYAWb's plateau. Only n=2 per group have reached 4e9 (sYAWc at 3.3e9,
sCTL2 stuck at 3.0e9), so this is two-vs-two.

Trajectory of the separation across the night, for the record:
2.0e9 separated, 2.5-3.0e9 overlapped, 4.0e9 separated again. Reading
any single snapshot as the answer is what produced three corrections
tonight; the stable summary is that the yaw group's median leads at
every mark and its worst seed is usually but not always above the
control's best.

## DEFINITIVE yaw comparison (~12:55): band-averaged, noise-robust

Single-eval snapshots swing wildly (sYAWv2 went 174,179 -> 139,820 in
300M steps), which is what produced three corrections tonight. The
robust version: for each seed take the MEAN of all its evals inside a
step band, then the MEDIAN across seeds. Robust to both the oscillation
and the one stuck control seed.

| band | yaw | control | ratio |
|---|---|---|---|
| 1-2e9 | 73,961 (n=3) | 41,984 (n=3) | **1.76x** |
| 2-3e9 | 100,893 (n=3) | 46,445 (n=3) | **2.17x** |
| 3-4e9 | 135,204 (n=3) | 74,779 (n=3) | **1.81x** |
| 4-5e9 | 138,361 (n=1) | 109,142 (n=2) | 1.27x |

**--yaw-adaptive gives a consistent 1.8-2.2x advantage in eval track
progress through 1-4e9, at n=3 per arm, on a median-of-means statistic
that no single lucky seed or unlucky eval can drive.** The 4-5e9 band is
thin (n=1 vs n=2) and should not be read yet.

This is the night's headline result and supersedes every earlier
single-point claim about the yaw fix, including the ones I had to
retract. It is consistent with, and roughly the size implied by, the
mechanism measurement (capture -123.5% vs -245.8%).

What it does NOT show: none of tonight's six seeds became a finisher,
and the plateau discrepancy vs the original champion campaign
(open question above) is unaffected by this result.

## Hypothesis: the respawn margin may be blocking the final wall (~13:30)

Both leading arms have oscillated at 170-195k for billions of steps
without crossing: sYAWv2 (scratch) 139,820 -> 194,317 -> 171,325 and
wGRAFT2 (warm) parked at ~195,500. Same barrier, sustained.

Mechanism worth suspecting: `respawn.py` only harvests snapshots taken
**at least `--respawn-margin` seconds (default 10) before the episode
ended** - the docstring's reasoning is that "closer states are usually
already doomed". That is right for ordinary deaths, but at the FINAL
wall the states immediately before failure are precisely the ones that
need practice, so the last ~10 s of the route may never become a
respawn point. An agent that cannot respawn into the last stretch has to
solve it in one shot from far away, every time.

Note the champion DID break this wall with margin 10, so this is at most
a contributing factor, not a hard blocker.

Test: wGRAFT2 (stalled at 195k for ~2e9 steps, so nothing is lost)
relaunched from its checkpoint as **wMARGIN with `--respawn-margin 2`**.
Everything else identical. If the margin is the blocker, near-finish
states enter the reservoir and the last stretch becomes practiceable;
if it crosses where wGRAFT2 could not, that is a clean result and a
cheap, general fix for the hardest wall on any map.

## CONFIRMED: the respawn margin excluded the entire final-wall region

Direct check of what is actually IN the reservoir (geodesic
distance-to-finish of every stored snapshot, 20,000 states each):

| reservoir | closest to finish | p1 | median |
|---|---|---|---|
| margin 10 s (wGRAFT2) | **29,788u** | 33,938u | 78,312u |
| margin 2 s (wMARGIN) | **5,235u** | 8,068u | 60,411u |

**With the default `--respawn-margin 10`, no snapshot within ~30,000u of
the finish ever enters the reservoir - the agent cannot practice the
last 15% of the 198,380u track from a respawn, and that is precisely
where the hardest wall is.**

The number is not arbitrary: the margin is specified in TIME, and at
racing speed (~3,000 u/s) 10 s of margin excludes 30,000 UNITS of track.
The faster the agent gets, the more of the endgame the curriculum hides
from it - the margin actively works against the agent exactly as it
approaches the finish.

This is a design flaw with a clear fix direction: the margin exists to
avoid seeding provably-doomed states, which is a distance-from-death
concern, but it is being applied as a fixed time and therefore scales
with speed. Options: specify it in distance, scale it inversely with
speed, or exempt states that are close to the goal (where "doomed" and
"about to finish" are the same states).

Standing on the experiment: mechanism CONFIRMED (the reservoir contents
change exactly as predicted). Whether it produces a finish is a separate
question - wMARGIN is recovering (173,497 -> 181,608) after the
distribution shift. Note the original champion DID clear this wall with
margin 10, i.e. it solved the last 30,000u in one shot from farther
back, so the margin is a handicap rather than an absolute blocker.

## Fleet trimmed for credit (~15:00)

vast.ai credit at $17.28 with 6 boxes burning $1.92/h - about 9 h of
runway, and the headline results were already established. Trimmed to
4 boxes, $1.20/h, 14.4 h runway.

Retired (final data saved to runs/<name>/progress.csv first):
- **sYAWc** (3rd yaw seed, $0.562/h - the most expensive box): plateaued
  at 88-99k from 2e9 to 5.8e9. Its curve is already inside every
  band-averaged comparison.
- **sCTL2** (2nd control seed): stuck at ~14,100 from 1.5e9 through
  5.4e9 - a textbook wall-jammed seed, and a useful datapoint for the
  stick-rate observation (0/3 yaw vs 2/3 control).

Kept: sYAWv2 (best scratch seed, closest to a finish), sCTL (control
for matched comparison), sCTL3 (second control), wMARGIN (the active
respawn-margin experiment, $0.134/h).

## Chaining the search FAILS - and that changes the record plan (~15:40)

tas_search proved single segments are improvable (+117u to +617u each).
The obvious next step was to chain them into a faster full run
(tools/tas_chain.py). **It collapses after ~2 windows**: the run jumps to
~198,300u, which is the SPAWN distance - the agent died and auto-reset.
Critically, even the UNMUTATED candidate dies, i.e. replaying the
recorded actions verbatim from the chained state fails.

Diagnosis: the chain enters each window from a SIMULATED state, while
the recorded actions were produced from the RECORDED state, and the two
differ slightly. A recording stores origin/velocity/yaw/onground, but a
full SurfState also carries stamina (fuser2), duck timers, basevelocity,
oldbuttons and more, so the restore is necessarily partial. Surf is
chaotic; a marginally different entry state invalidates a scripted
sequence within a second or two.

This is the same brittleness the literature reports (Jacobsen & Togelius:
A* Mario cleared 98/98 levels deterministically and 0/98 under 20%
action noise) and it has a direct consequence for us:

**Per-segment gains do NOT compose.** Each window's improvement
invalidates the next window's script, so "+186u per 2 s segment x 40
segments = 7 s" was never a valid projection - it was an upper-bound
sketch, and this is the evidence that the caveat mattered.

What survives: the value-ceiling result (this line is worth ~72-74 s)
still stands, because it was computed on the ORIGINAL line with an
energy argument, not by composing segment searches. And the demonstration
that the line is locally improvable still stands per-segment.

Revised record plan: a valid chain must RE-OPTIMIZE each window from the
state actually reached rather than perturbing a script recorded
elsewhere. That is a much larger search with no good initialization -
which is precisely the case for the survey's recommended architecture:
search to find the line, then DISTILL it into a policy via a backward
gated curriculum, because a policy is closed-loop and therefore robust
to the state drift that kills an open-loop script.

## The endgame trade-off of --yaw-adaptive (~16:00)

Verified first that this is not a detection bug: sYAWv2's log shows the
finish box armed at [-14720, 7487, -1824] .. [-8064, 7488, -352], the
same 1-unit curtain the champion crossed thousands of times.

So `win 0.00%` is real. Neither endgame run has EVER completed an
episode:

| run | steps | best eval track | training wins |
|---|---|---|---|
| sYAWv2 (scratch + yaw fix) | 6.5e9 | 194,317 | **0** |
| wMARGIN (champion + yaw fix + margin 2) | 15.0e9 | 195,504 | **0** |

wMARGIN is the damning one. It STARTED from a checkpoint that finished
the map reliably (78% training success), and after the action space was
re-parameterized it recovered 98.5% of the route in 200M steps - but has
not crossed the line in ~6e9 steps since, even with near-finish states
(5,235u out) in its reservoir.

**Reading: --yaw-adaptive improves general progress (the 1.8-2.2x
band-averaged result stands) but destroys precisely-tuned endgame skill,
and that skill does not come back cheaply.** The final two ramps demand
exact per-tick control; the champion had learned an action sequence in
the OLD parameterization, and every bin now means something different.
Broad competence transfers through a representation change; a knife-edge
maneuver does not.

Practical consequences:
1. Do not graft the yaw fix onto a finisher and expect to keep the
   finish. Either train with it from scratch through the whole
   curriculum, or keep the stock action space for a policy that already
   clears the last wall.
2. The margin experiment is confounded by this: wMARGIN cannot isolate
   the margin's effect while it is simultaneously relearning the endgame
   under a new action space. The margin's MECHANISM finding (10 s hides
   30,000u of track) stands on the reservoir audit and does not depend
   on this run.
3. The honest scoreboard is unchanged: our best full run is still the
   champion's 1:19.72 under the stock action space.

# ============================================================
# ROUND 16 (2026-08-21 morning): EXPLORATION, per the user's reframing
# ============================================================

User's read, which reorders the priorities: the yaw fix made convergence
faster but did not reach the goal, and NO experiment tonight reproduced
the champion - so the champion may simply have been a lucky seed, and
the yaw fix is not a regression. **The real target is consistency: agents
repeatedly jamming at the same walls is an exploration failure.**

Two corrections to my earlier claims that this prompts:
1. "the act-every axis is closed" was too broad. I tested only COARSER
   (6, 9). **Finer decisions (act-every 2 = 50 Hz, or 1 = 100 Hz) are
   untested**, and given that the strafe finding is about precision at
   high speed, finer is arguably the more promising direction.
2. The reservoir-depth diagnostic I "discovered" already existed in
   train_fast (it prints `reservoir d: min/p10/median` every 100 iters
   and its comment already names the harvest margin as the suspect).
   What this session added is the quantification and the causal test,
   not the observation.

Live reservoir depth confirms the speed coupling cleanly:

| run | margin | closest state to finish |
|---|---|---|
| sYAWv2 (slower policy) | 10 s | 13,808u |
| wGRAFT2 (champion-speed) | 10 s | 29,788u |
| wMARGIN | 2 s | 5,078u |

Same margin, ~2x the hidden distance for the faster policy. But note
**wMARGIN never finished even respawning 5,078u out**, so the last
stretch is a genuine skill barrier, not only a coverage gap.

## New arms

| arm | box | change vs sYAWv2 |
|---|---|---|
| **sEZ** | 3090 ssh7:16568 | `--ez-eps 0.02` ez-greedy bursts |
| **sAE2** | 3090 ssh9:19498 | `--act-every 2` (50 Hz decisions) |

Control is sYAWv2 (same recipe, no exploration change), still running.
Retired: wMARGIN (confounded - could not isolate the margin while
relearning the endgame under a new action space) and sCTL3 (control
curve established).

ez-greedy design note: burst transitions are dropped from the PPO
minibatch pool because their actions were not drawn from pi, so their
log-probs would make the importance ratio meaningless. They still shape
learning through the states they reach and through GAE running back
across them. Measured at eps=0.02: 8.6% of decisions inside a burst,
mean burst 4.7 decisions (140 ms), heavy tail to 60 (1.8 s).

## FINAL yaw verdict, both budgets complete (~16:30)

| @steps | sCTL (stock) | sYAWb (yaw fix) |
|---|---|---|
| 4e9 | 105,117 | **137,030** |
| 6e9 | 155,557 | 155,545 |
| 8e9 | 156,826 | 149,763 |
| final | 156,826 @8e9 | 153,080 @12e9 |

**--yaw-adaptive reaches the plateau faster but the plateau is the same
(~155k).** Combined with the band-averaged 1.8-2.2x advantage through
1-4e9, the fix is a convergence-RATE improvement, not a ceiling
improvement. Neither arm ever finished (win 0.00% throughout).

This is exactly the user's reading: the fix helps, it is not a
regression, and it does not solve the actual problem. **All seven scratch
seeds this session - three yaw, three control, one deep - plateaued in a
149k-157k band and none reproduced the champion's 198,380u.** The
consistent stopping point across independent seeds and two different
action spaces says the barrier is structural, not seed-specific: it is
the wall-3/4 region, and getting past it is an EXPLORATION problem.

That reframes the whole session's scoreboard. The yaw fix is worth
keeping (faster to the same place, and the mechanism is measured), but
the binding constraint is the agent's inability to discover the
maneuvers past ~155k, and that is what round 16 (ez-greedy bursts,
finer decisions) is aimed at.

## ez-greedy dose-response, first read (~17:00)

Eval track at matched-ish steps against sYAWv2 (same recipe, no
exploration change):

| arm | evals | at ~0.4e9 |
|---|---|---|
| sYAWv2 (control) | 12,048 @0.25e9, 21,184 @0.5e9 | ~19,000 |
| sEZ (eps 0.02) | 689 / 8,659 / 14,562 | 14,562 |
| **sEZ5 (eps 0.05)** | 23 / 1,580 / **2,144** | **2,144** |
| sAE2 (act-every 2) | 41 / 8,819 / 12,058 | 12,058 @0.35e9 |

**eps 0.05 is decisively harmful** - 2,144 at 0.41e9 is 4x below the
historical band's MINIMUM at 0.25e9 (8,941). At that dose ~20% of
decisions are burst actions, so a fifth of the behaviour is random AND a
fifth of the samples are dropped from the policy loss. Killed.

eps 0.02 (8.6% burst decisions) is behind the control but not broken.
Relaunched at **eps 0.005 with --ez-max 20** (~2% of decisions, bursts
capped at 0.6 s) to find whether there is a useful low dose at all.

Reading so far: the same shape as every other exploration knob this
project has tried (int-coef, RND) - a dose that is large enough to
change behaviour is large enough to damage the policy. That is worth
stating early rather than after three more arms.

Note on the mechanism: unlike count-based novelty, ez-greedy pays no
reward - it perturbs the BEHAVIOUR policy, so its whole effect must come
through states discovered. With PPO discarding those transitions, the
only channel left is the value/advantage of neighbouring on-policy steps
and the respawn reservoir harvesting burst-reached states. That is a
thin channel, which may simply make ez-greedy a poor fit for on-policy
PPO regardless of dose.

## Round 16b: reward as an OBSERVATION (user idea, ~17:45)

New flag `--obs-reward`: the previous decision's reward is fed back as
part of the observation.

Why this is different in kind from everything else tried. Count-based
curiosity, RND, spawn diversity and ez-greedy all change what the agent
is PAID or where it STARTS. This changes what it KNOWS. The agent has no
absolute position and no compass, and the race reward is potential-based
geodesic progress - so the reward is exactly the "did that decision move
me toward the goal" signal the agent currently has to infer from how the
depth image changes. At a wall the agent reaches constantly but cannot
solve, being able to tell a better attempt from a worse one is a
plausible missing ingredient.

Implementation keeps obs_dim unchanged: the value rides in scalar slot
12, an absolute-position channel the no-GPS policy already masks out,
re-enabled through `Policy(extra_feat=...)`. So the image slice, buffer
widths and CUDA-graph shapes are untouched; only which scalars the
network reads differs. Value is tanh(r/0.1) - sensitive across the
ordinary per-decision range (~0.05-0.2), saturating on the +50 finish
bonus. Written after the env step so decision t+1 sees the reward from
t; zeroed at reset.

## Round 16 arms and first reads

| arm | change | latest |
|---|---|---|
| sYAWv2 | control | 12,048 @0.25e9, 21,184 @0.5e9 |
| sEZ (eps 0.02) | ez bursts, 8.6% of decisions | 14,562 / 15,333 / 16,217 - behind control, RETIRED for the box |
| sEZ5 (eps 0.05) | ez bursts, ~20% | 2,144 @0.41e9 - catastrophic, killed |
| sEZ005 (eps 0.005, max 20) | ez bursts, ~2% | running |
| sAE2 (act-every 2, 50 Hz) | finer decisions | 8,819 / 12,058 / 15,656 @0.52e9 - behind control's 21,184 @0.5e9 |
| **sOBSR** | **--obs-reward** | just launched |

ez-greedy reads as a dose-response with no good dose so far: 0.05
catastrophic, 0.02 clearly behind, 0.005 pending. The structural reason
to expect this: ez-greedy pays no reward, so its only channel is states
discovered - and PPO must DISCARD those transitions (their actions were
not drawn from pi), leaving only the neighbouring on-policy advantages
and the respawn reservoir. That is a thin channel, and ez-greedy was
demonstrated on off-policy Q-learning where the bursts are trained on
directly.

50 Hz is also behind so far, consistent with the user's prior that it
would be too much.

## act-every 2 (50 Hz) verdict + the 4/5 rungs (~18:00)

| @steps | sAE2 (50 Hz) | sYAWv2 (33 Hz control) |
|---|---|---|
| 0.25e9 | 8,819 | **12,048** |
| 0.50e9 | 15,656 | **21,184** |

**50 Hz is behind the 33 Hz control at both marks (-27%, -26%), and it
costs ~50% more policy forward passes and updates per game-second.**
Killed at 0.59e9 on the user's call. The user's prior ("way too much")
was right, and it sharpens the picture: on this task 33 Hz is not just
adequate, it is at or near the optimum from BOTH sides.

Decision-rate ladder as it now stands (all vs the same control):

| act-every | Hz | result |
|---|---|---|
| 2 | 50 | **-27% at 0.5e9** |
| 3 | 33 | baseline (and the champion's setting) |
| 4 | 25 | **running (sAE4)** |
| 5 | 20 | queued |
| 6 | 16.7 | worse (14,995 @0.88e9, low in band) |
| 9 | 11 | worse |

The interesting question the 4 and 5 rungs answer: is 33 Hz a sharp
optimum, or is there a plateau from 3-5 where we could take the ~25-40%
compute saving for free? The literature's superhuman racers all sit at
10-20 Hz (Sophy 10, Linesight 20, Fuchs 10), so a plateau out to 20 Hz
would reconcile our result with theirs.

## --obs-reward: broken measurement, then a strong positive (~19:00)

Three code paths rebuild observations independently, and --obs-reward
writes a value the C core does not produce, so each had to be fixed
separately as it surfaced:

1. **trainer eval** - `_TorchPolicyBase._obs()` passes the core's raw
   scalars through, so slot 12 carried ABSOLUTE POSITION (origin/2000,
   magnitude ~10) to a policy trained on tanh(reward) in [-1,1].
2. **record_ckpt** - same mismatch; the user clicked "record greedy" and
   got an agent that died in seconds.
3. **record_ckpt policy rebuild** - missing extra_feat, 523-vs-522
   state_dict mismatch, which the dashboard hid by piping stderr to
   DEVNULL.

The diagnostic that gave it away: sOBSR had training rew 12.0 and
intrinsic 3.5/ep - HIGHER than any other arm - while eval track read
1,571. Training healthy plus evaluation collapsed means the two see
different observations.

With the eval feed fixed (recomputes the shaping term from the goal
field, own prev-distance state, jump clamp):

| steps | sOBSR2 | control sYAWv2 |
|---|---|---|
| 0.64e9 | **38,221** | ~24,000 (interp) |
| 0.79e9 | **63,918** | 27,183 @0.75e9, 46,672 @1.0e9 |

**2.35x the control at a comparable point**, and already past the
control's 1.0e9 value at 0.79e9. Promising - and the first intervention
this session to look better than the control on the metric that matters.

Caveats, stated because three claims have already needed retracting
today: n=1; eval progress oscillates hard on every arm; and the eval
feed is an APPROXIMATION - training writes tanh((shaping + intrinsic)/
0.1) while the feed recomputes shaping only, omitting the novelty bonus
(~0.007/decision vs shaping's ~0.023). Directionally right, not
identical. A second seed and an exact feed are the next steps.

Recording after the fix, same ckpt: 3 episodes of 16.1-16.8 s at 1,832
u/s, versus "dies immediately" before.

## Round 14 - decision-rate ladder closes; obs-reward holds at 2x

Band-matched track progress against the sYAWv2 control lineage:

| arm | decision rate | steps | track | control at matched steps |
|---|---|---|---|---|
| sOBSR2 | 33 Hz | 1.16e9 | **98,063** | 46,672 @1.0e9 |
| sAE4 | 25 Hz | 1.86e9 | 69,754 | 90,468 @2.0e9 |
| sEZ005 | 33 Hz | 1.43e9 | 47,147 | ~parity |
| sAE5 | 20 Hz | 0.49e9 | 15,390 | 21,184 @0.5e9 |

**The act-every question is now closed in both directions.** Every rate
tested off 33 Hz is worse: 50 Hz (act-every 2), 25 Hz (4), 20 Hz (5),
16.7 Hz (6), 11 Hz (9). That is a sharp optimum, not a plateau with a
free compute saving on one side - which retracts the earlier expectation
that 25 Hz would buy throughput at no cost. Surf needs finer control than
the 10-20 Hz that the car-racing literature settles on, and the reason is
mechanical: the air-accel gain window is +/-0.52 deg at 3000 u/s, so a
decision interval that straddles it gives up speed no policy can recover.

**obs-reward remains the only clear win of the session** - roughly 2x the
control at matched steps, sustained from 0.64e9 through 1.16e9. Still
n=1, and the eval feed is still shaping-only (Round 13 caveat stands).

ez-greedy at eps=0.005 is at parity. That is not evidence against
temporally-extended exploration, only against this dose; the burst
exclusion from the PPO minibatch is verified correct.

### Ops: recorder config guard

Three misleading trajectories shipped today from ONE mechanism - the
trainer grows a flag, record_ckpt.py is never taught to mirror it, and
the recorder emits a plausible trajectory under wrong semantics.
--obs-reward was loud (strict state_dict load throws on 523-vs-522); its
reward feed and --yaw-adaptive were both SILENT, the latter reading 42k
where the trainer's own eval read 98k on identical weights.

Guarding the three known fields would only have waited for the fourth
flag, so the guard checks the property instead: a checkpoint key the
recorder never READS is a key it cannot be mirroring. Unread and not in
TRAIN_ONLY => refuse to record. Verified against six cases including both
of today's real bugs and a synthetic future flag. New trainer flags now
force an explicit decision in the recorder or fail loudly.

### Ops: dashboard record buttons

Reported as "does nothing". They were never broken - they fire, the job
runs, and it returned rc=0. The defect was cost and feedback: 3 episodes
x 3000 ticks of single-env GPU lidar on a box that is also training, with
a static label and stderr piped to DEVNULL. Measured end-to-end through
the tunnel after cutting to 2x2000 and adding an elapsed-seconds clock:
**61 s**, rc=0, valid artifact. Lesson worth keeping: a slow job behind a
static label is indistinguishable from a dead button, and "I tested the
CLI path" is not "I tested the button".

### Correction to Round 14: the act-every optimum is shallow, not sharp

The Round 14 table used single spot values, which oscillate hard. sAE5
read 27% behind control at 0.49e9 and at PARITY by 0.81e9 - same arm,
same control, opposite conclusion. Band-averaged over the whole overlap
(race/eval_progress, the honest metric):

| arm | rate | band-avg | control | ratio |
|---|---|---|---|---|
| sAE5 | 20 Hz | 14,972 | 17,442 | **0.86x** |
| sAE4 | 25 Hz | 38,596 | 42,150 | **0.92x** |
| sOBSR2 | 33 Hz + obs-reward | 74,070 | 38,947 | **1.90x** |

So "every rate off 33 Hz is worse, a sharp optimum" was overstated. 25 Hz
costs ~8% sample efficiency, 20 Hz ~14% - mild penalties, not a cliff.
Only 16.7 Hz and 11 Hz were clearly bad. The practical consequence is the
opposite of what Round 14 implied: if act-every 4 buys more wall-clock
throughput than the 8% it costs in sample efficiency, it is a net win.
That trade is now the open question, not whether 33 Hz is optimal.

Sanity check on the metric itself: eval/path (col 9) and
race/eval_progress (col 14) give ratios within 0.03 of each other on this
map, so earlier comparisons that used col 9 are not invalidated. Use col
14 regardless - the two decouple exactly when an agent wanders, which is
the case worth catching.

Method note, third time this has bitten: a single (arm, step) pair against
a single control point is not evidence. Band-average or do not claim.

### Round 15: act-every 4 is a net WALL-CLOCK win (reverses Round 14)

Round 14 judged decision rate on sample efficiency alone, which is the
wrong currency - the user's goal is faster convergence in wall-clock, and
a coarser decision rate buys throughput because fewer forward passes run
per physics tick. Throughput measured on ONE box (ssh9, so hardware
cancels), median fps over each run's last 3/4:

| act-every | rate | fps | vs act-every 3 |
|---|---|---|---|
| 2 | 50 Hz | 176,658 | 0.74x |
| 3 | 33 Hz | 238,354 | 1.00x |
| 4 | 25 Hz | **291,786** | **1.22x** |

Combined with the 0.92x sample efficiency, act-every 4 should be
~1.13x per wall-clock hour. Direct check - each arm's own progress curve
re-timed with ssh9's measured fps for its own act-every, so hardware and
throughput are both controlled:

| wall-h | 33 Hz | 25 Hz | ratio |
|---|---|---|---|
| 0.5 | 16,564 | 18,392 | 1.11x |
| 1.0 | 39,276 | 32,174 | 0.82x |
| 1.5 | 42,370 | 55,365 | 1.31x |
| 2.0 | 57,752 | 81,433 | 1.41x |

Mean ~1.16x, matching the 1.13x prediction. **act-every 4 converges
faster in wall-clock than the champion's act-every 3**, and the Round 14
conclusion ("25 Hz behind, keep 33 Hz") was an artifact of measuring in
env steps instead of seconds.

Caveats: sAE4 spans only 2.1 wall-hours so far, and the per-mark ratios
swing 0.82-1.41 - this is a trend, not a settled number. NOT used as
evidence: the same-box sAE4-vs-sCTL3 comparison (1.92x at 2h) is
confounded, sCTL3 is not yaw-adaptive.

Implication for the next round: the champion recipe should probably be
obs-reward + act-every 4, combining the session's two wins. Neither has
been tested together.

### Round 15b: the wall-clock optimum is COARSER than the sample-efficiency optimum

Extending the ladder with throughput measured same-box, same yaw-adaptive
setting (sYAWv2 vs sAE5 both on ssh3; sCTL3/sAE2/sAE4 all on ssh9):

| act-every | rate | throughput | sample-eff | predicted wall-clock |
|---|---|---|---|---|
| 2 | 50 Hz | 0.74x | worse | clearly bad |
| 3 | 33 Hz | 1.00x | 1.00x | 1.00x (champion) |
| 4 | 25 Hz | 1.22x | 0.92x | **1.13x** (observed ~1.16x) |
| 5 | 20 Hz | 1.44x | 0.86x | **1.24x** (thin, see below) |

The act-every-4 prediction was validated against its own re-timed curve
(1.13x predicted, ~1.16x observed), so the product of the two measured
factors is a usable estimator.

**This is the answer to "make convergence faster": the decision rate that
maximizes progress per env step is NOT the one that maximizes progress per
hour.** 33 Hz wins on samples; 25 Hz and probably 20 Hz win on the clock,
because a coarser rate runs fewer forward passes per physics tick and the
throughput gain outruns the sample-efficiency loss.

act-every 5 is NOT yet claimed. Its direct same-box curve gives 1.04x at
0.5 wall-h and 1.36x at 0.75 wall-h - consistent with 1.24x but only
0.91 wall-hours deep. The 0.25 wall-h mark reads 9.91x and is discarded
as a startup artifact (the 33 Hz arm is still at 882 progress there);
averaging it in would have produced a bogus "4.10x" headline. sAE5 stays
running - on sample efficiency alone it looked like the worst arm and the
retire candidate, which would have been exactly the wrong call.

## Round 16 - exploration-literature round (2026-08-21 evening)

Context: sOBSR2 (obs-reward arm) was stopped at step 3,782,737,920 on the
rented 3090 (box harvested to runs/sOBSR2/, then destroyed). Its win rate
had been 0.00% for ~2e9 steps while race/eval_progress oscillated
138k-195k. Four exploration methods were implemented VERBATIM from the
papers in docs/research-litsurvey.md section 6 (constants verified
against ar5iv full texts + reference code by an independent extraction
pass), each as a flag on branch explore (5b57e8c, 650b386, ae02dd9,
ae6286d), each run ~1h on a rented 3090 resuming the same ckpt_latest
(md5 1ba1fd2936af3ae1ad3608e3cd6b1e9e). Control xCTL ran the unmodified
config on the local 5090 (byte-identical code path - verified by an
adversarial review pass, see below).

### The paper -> experiment -> result table

| paper | experiment (what was done) | result |
|---|---|---|
| Dabney et al. 2020, ez-greedy (2006.01782) | --ez-eps 0.01 --ez-max 10000 (paper steady-state eps, general cap; mu=2 zeta-like burst durations, uniform full-action bursts, off-policy-excluded from PPO) | CLEAR NEGATIVE: eval 172k -> 109k -> 50k over 0.3e9 steps vs control band 138-174k; killed at 25 min by the drop rule. Caveats from review: cap is an Atari-unit transplant (2.5x our episode cap), bursts leaked across resets (since fixed), arm got ~12% fewer optimizer steps |
| Ecoffet et al., Go-Explore (1901.10995 / Nature 2004.12919) | --respawn-mode goex --respawn-bins 64 --spawn-burst 100 --spawn-burst-p 0.95: reservoir bins weighted W=1/sqrt(C_seen+1), post-respawn 100-decision uniform random burst at 95% repeat | NULL at 1h: evals 160-162k, in control band, win 0%. Review verdict: does not actually test Go-Explore - the burst excluded ~75% of the PPO batch (5x fewer optimizer steps than control), C_seen degraded toward times-chosen (converges to uniform-over-bins), and the 10s harvest margin discards everything bursts discover |
| Florensa et al. 2017 (1707.05300) | --respawn-mode florensa --respawn-bins 64: start bins with success in (0.1,0.9) estimated from the training batch, 1/3 replay of ever-good bins | DEGENERATE BY CONSTRUCTION: with win identically 0 the band empties in ~2 rebuilds and the arm silently falls back to uniform-over-occupied-bins (cap off). As that unlabeled treatment it posted the best training metrics of the round (rew 33.4, len 3546 vs control 27/2750) and evals 178-184k, top of band - evidence FOR distance-flattened respawn, not for/against Florensa. The method needs a nonzero success signal (the paper seeds starts_old at the GOAL) |
| Salimans & Chen 2018 (1812.03381), d-window variant | --respawn-mode backward --respawn-bins 64 (+ --respawn-killsafe after the fail-floor bug): window of 2 bins nearest the goal, rho=0.2 advance | STUCK AS DESIGNED MUST BE: window pinned at its init bins at 0.0% success over the whole hour (win signal identically zero from 19-22k out); mean episode ~ stall-kill. First launch also exposed the fail-floor spawn bug (below). Review: absolute-finish success is not the paper''s demo-parity criterion; without a demo the method cannot express itself |
| Salimans & Chen 2018 (1812.03381), REAL demo spine (xSC2) | Recorded 6 greedy episodes from runs/frozen/sISV_FINISHER_latest.pt locally (5/6 finish ~84-86s); extracted 100 full states (pos+vel+yaw) covering the last 25s of the fastest finisher; --demo-file replaces the reservoir pool share with window draws over demo TIME, rho=0.2 advance + reference-code backoff | WIN 0.00% -> 85%+ within ~30 min on the SAME stuck checkpoint. Window walked ~20 states (~5s) backward at 91-96% per-window success. First finishes in this lineage''s history. Start-line eval unchanged so far (152k) - value propagation to the true start is the >1h question |

### Three findings that outrank the arms

1. **ckpt_latest.pt is an eval trough, and nothing guards against it.**
   Seven independent resumes of the same ckpt scored first evals of
   19.8k-24.3k (tight cluster) while the run''s last logged eval, 141.6M
   steps earlier, was 195,362. ckpt_latest is written every 60s; evals
   run every ~980s; no best-eval checkpoint exists. Under respawn_frac
   0.9 the training metrics cannot see start-line rot. Action item:
   best-eval ckpt tracking in save_ckpt, and never branch arms off an
   uneval''d snapshot.
2. **race/eval_progress is flattered by death-dives; the honest frontier
   is a mid-route deceptive basin at d ~ 21.5k.** Every method - control,
   all arms, AND the champion''s greedy evals - bottoms out at min_d
   21.4-21.7k in the same physical region, then "progresses" further only
   by falling through goal-adjacent space (raw-field dips to ~3k) or, for
   the champion, by entering territory the field reads as 31k -> 107k ->
   unreachable(NaN) -> frozen 13.5k INSIDE the goal box. The 138k-195k
   eval oscillation is mostly fall-trajectory noise. The shaping field''s
   reachable minimum is not the goal: on this map the potential-based
   objective and the task objective decouple ~50 champion-seconds before
   the finish.
3. **Fail-floor spawns (user-observed):** frontier-seeking start
   selection concentrated spawns onto teleport/fail floors whose voxels
   carry small raw d; restored states stand there until the 15s
   stall-kill (episode len ~1290 = the signature). --respawn-killsafe
   (bin on the kill-masked goalk field) fixes the sampling side; the
   env-model question (standing on a floor that should teleport) is the
   ski_2-conveyor class of defect and remains open.

### Verification pass (5 independent opus reviewers, adversarial)

Control purity UPHELD (xCTL byte-identical to origin/main under its
flags; allocation refactor proven bit-identical for uniform weights).
Fixes landed from the review: arm flags now restore from ckpt config
(else a resumed arm silently reverts to control); bursts abort at
episode end for ez too. Open items for any rerun: minibatch-count
compensation for burst exclusion (blocking for GE-style arms), per-cell
rather than per-bin Go-Explore weights (|bin| factor missing), C_seen
marking every K ticks instead of snapshot cadence, persist curriculum
state + bin counters in the ckpt, killsafe "unsampleable" claim is
overstated (only fully-enclosed voxels read invalid), --respawn-killsafe
+ --race-dist euclid crashes (NameError), width/rho/cooldown of the
backward window not exposed as flags.

### Ops

Fleet: 1-minute readiness rule enforced - 24 instances destroyed at
~80s for slow loading (~$0.08 total); machine 39565/host 155125
blacklisted (GitHub TLS broken in one container), machine 84216/host
443829 blacklisted (drops ssh mid-transfer); hosts 344939 and the xEZ
box 39565-sibling recorded known-good. All boxes ran dashboards through
self-healing local tunnels (8601-8604). Every box destroyed immediately
after harvest; total rental spend for the round ~= $1.30.

### Round 16 closing (2026-08-22 ~01:00)

Late results: xSC2c (trimmed-spine continuation) consumed its whole
spine - 57 window advances, 92.8% train win, REPEATED greedy start-line
finishes: fin 6/9 mean 84.96s best 82.40s (all-time record 79.73s).
Final ckpt + winning trajs in runs/research/xSC2c/. xSC2b replicated
the walk on an independent seed (52 advances, telemetry only - its box
dropped the connection mid-harvest and was destroyed per the
box-defect rule; csv lost). xBIN (race-shaping 0, +1000 win, intrinsic
kept): start-line eval 181k -> 4.7k in 490M steps, zero wins - dense
honest signal is required, sparse+intrinsic cannot hold behavior.

Session verdict vs the actual goal (self-unsticking exploration):
the S-C demo arms are NOT the goal (champion info injected); their
scientific yield is (1) the failure is reward-geometry, not policy
capability - the stuck policy finishes at champion pace when placed
on-route; (2) a costed resurrection recipe (any winning traj ->
finishing policy in ~2-3 GPU-h, auto-extracted spine). The goal-line
carriers, built and pending: gravity-directional honest field (code
done + offline-validated logic; BAKE PENDING - needs a solo 24-32GB
GPU for some hours), faithful reward-free Go-Explore phase-1
(tools/explore_phase1.py, reached 92.4% of the track in 20 min CPU,
stalls at the final precision barrier; speed-cell variant untested to
completion), and the action-chunk codebook design (SPiRL/VQ-BeT,
in progress). Next screening queue: honest-field multi-seed
(reliability metric: fraction of seeds crossing the wall
autonomously), Linesight temporal mini-race (survey shortlist #1,
attacks the discounting half of the basin trap), decoder-chunk
entropy exploration.

## Round 17 - learnable behavior-decoder (user-designed), session end (2026-08-22 ~02:30)

The user''s architecture, implemented and adversarially verified
(3 independent reviewers; commit 476bb1d): the policy picks 1 of K=64
codes per chunk; a LEARNABLE (K, H=10, 32) logit table inside Policy
expands the code into 10 per-decision 6-head action distributions; PPO
trains trunk + code head + decoder end-to-end through the joint
log pi(code|s) + masked per-decision decoder logps. Verified: exact
acted-vs-recomputed logp round-trip; decoder gradient path proven by
CPU micro-tests; chunk=0 byte-identical to the flat trainer. Chunked
rollouts run ~1.0-1.6M env-steps/s on the local 5090 (10x fewer trunk
forwards at unchanged 33 Hz control).

Scratch runs on cannonball (all from fresh weights, no route info):

| run | config error | outcome |
|---|---|---|
| xCHUNK v1 (2.02e9 steps) | launched by hand WITHOUT --respawn-frac / --int-coef and at n_steps 128 (1/10 update density - the design doc prescribed 16/8/4) | ep_rew pinned at -time_pen (every episode = 15s stall-kill) BUT eval_progress crept 21u -> 1,131u and peak speed -> 618 u/s (the USER caught this on the dashboard after it was wrongly reported as flat), while CODE ENTROPY COLLAPSED 4.16 -> 0.98 (~3 effective behaviors). Finding: collapse is real at ent 0.005 / dec-ent 5e-4, and the run learned slowly despite it |
| xCHUNK v2 (110M) | correct n_steps/epochs, still no respawn/intrinsic | same stall-kill signature; killed |
| xCHUNK v3 (live at session end) | full config via tools/launch_local.ps1 (respawn 0.9, int 0.25/view 8/speed 3, n_steps 16 epochs 8 mb 4) | at 251M: reservoir full, int paying, code entropy 4.15 -> ~2.1 and flattening (~9 effective behaviors - concentration, not yet collapse), first eval 72u. Too early to judge |

Root-cause note for the launch errors: every earlier arm RESUMED a
checkpoint whose config silently restored respawn/intrinsic flags, so
hand-typed launch lines looked complete all session; the first scratch
launch had no checkpoint behind it. tools/launch_local.ps1 now carries
complete presets and proves liveness (new pid + log tail) or exits 1.

Open items for whoever continues: (a) code-entropy trip level 1.5 -
if v3 slides below, raise --ent (code-level) and/or --dec-ent;
(b) the reviewers'' chunk findings not yet fixed: successor-episode
reward leak into a terminated chunk''s return (up to 29 ticks),
truncation bootstrap uses gamma not gamma**(K*H), greedy-double-argmax
eval weak for near-uniform decoders; (c) no matched flat control run
exists yet (scratch_flat preset is in the launcher); (d) Phase-1
discovery stalls at the last 7.6 percent - speed-keyed cells (--cell
128 --cell-speed 4) were queued but not run to completion.

Operating the live run: dashboard http://localhost:8600 (run xCHUNK);
log runs/xCHUNK_launch.txt (UTF-16); stop with:
Stop-Process -Id 42400. Resume later with
powershell -File tools/launch_local.ps1 resume runs\xCHUNK\ckpt_latest.pt xCHUNK2

Round 17 postscript: xCHUNK v3 code entropy collapsed to 0.61 (~2 effective behaviors) by 1.38e9 steps despite respawn+intrinsic; eval plateaued ~1,250u. Stopped per the tripwire to spare the GPU; ckpt_latest kept. Next attempt needs a stronger anti-collapse lever: higher --ent (code level) and/or --dec-ent, or an entropy floor - collapse is now reproduced in 2/2 chunked scratch runs and is THE blocker for this architecture.

# ============================================================
# ROUND 18 (2026-08-22): the survey's untested items, one paper per agent
# ============================================================

Standing rules were fixed in writing this round and live in **CLAUDE.md**:
rented boxes are running or deleted (60 s readiness, defect = blacklist +
destroy, no load for 5 min = destroy), one paper = one run = one seed = one
hour, every run starts from the STUCK checkpoint, every run starts from
`tools/run_arm.sh`, and the verdict is `race/eval_progress` (or time to
finish for runs that finish). A local `tools/fleet_watchdog.py` daemon
enforces the GPU half from outside any agent's session.

Untested-item audit of docs/research-litsurvey.md (what has no flag, no arm
and no mention in this ledger): lookahead route geometry, asymmetric critic,
distributional critic, Linesight temporal mini-race, Sophy's speed-squared
contact penalty, on-policy plasticity preventives, gSDE/pink noise,
kickstarting/QDagger, SIL, Seer's KL-to-imitation, Swift's perception term,
Necto difficulty-weighted starts, high-rate + dwell, n-step, self-bootstrapped
reference lines, gamma-in-seconds. Three of the four things every superhuman
system in the survey has, this project still did not have.

## Arm xROUTE - lookahead route geometry (survey section 0, row 2)

Sophy's 60 ego-frame course points spanning ~6 s at current velocity
(ablation: removing them costs +2.64 s on a 114 s lap, the largest single
ablation in that paper), Fuchs' 10 curvature samples, Linesight's 40 virtual
checkpoints, Swift's gate corners. RL_Surf's observation was 10 honest
scalars + a 64x32 depth image and no route geometry at all.

Implementation (`python/surfgym/route.py`, `--route`): a reference polyline
resampled at constant arc length, reported in the player's frame at 8
horizons from 0.25 s to 6 s scaled by current speed, each normalized by its
own nominal span so the fan is scale-free; 27 features. Row layout becomes
`[15 core | 27 route | image]`, image still one trailing slice.
`--route-critic-only` feeds the fan to the value tower alone, which makes the
asymmetric-critic arm (Vasco RLC 2024) a one-flag follow-up.

The route is the champion's own finishing line (`tools/build_route.py` picks
the fastest finisher out of recorded trajectories; 1,811 points, 231,680 u).
Per Linesight the line "does not need to be fast... usually the centerline",
and later re-extracts from the AI's own best runs. It IS route knowledge, so
this arm is not honest-perception in the `--gps` sense - that is the
treatment, and it is what the papers evaluate.

Warm resume is **function-identical at step 0**: the fan is concatenated last
inside the towers, so `widen_for_route()` zero-pads trailing columns of the
first Linear and the matching Adam moments (6 tensors), and the resumed
policy computes exactly the baseline's function on its first forward. Covered
by `tests/python/test_route.py`.

### Two corrections this arm produced before it produced a result

**1. The 24,307 opening eval is the 5090's, not the baseline's.** xROUTE's
first eval was 177,591 against the "24,307" I had written into CLAUDE.md - a
7x apparent win, ten minutes old, and false. Both round-16 3090 arms opened
at 172-194k (xGE 193,802, xEZ 172,480); xCTL, the 24,307, ran on the local
5090. The lidar march is not bit-identical across GPU architectures
(`test_march_is_bit_exact_against_the_legacy_kernel` fails on a 3090, passes
on the 5090) and surf is chaotic, so one differing depth pixel forks the
whole greedy trajectory. **Arms are only comparable within a card.** The
round-16 "seven resumes scored 19.8k-24.3k" finding needs re-reading with
this in mind.

**2. `race/eval_progress` is flattered by death-dives, and now there is a
metric that is not.** `tools/eval_honesty.py` scores corridor progress -
how far along the reference route an episode got while staying within a
corridor of it, advancing only in order - plus whether it ended in the
finish box, short, or below it. A fall into goal-adjacent space stops the
clock, because the pit is nowhere near the line.

### Where the stuck agent is actually stuck

xROUTE's opening eval, scored honestly (9 greedy episodes):

| | |
|---|---|
| corridor progress | mean 187,989 u, max 205,312 u of 231,680 u (**88.6%**) |
| finishes | **0 / 9** |
| ended below the goal | 5 / 9, at z ~ -4,200 |
| closest approach to the line | **0-2 u** |

Six of nine episodes stop at the same vertex. The policy is not lost, not
stalling, and not slow: it tracks the champion line to within 1-2 units for
88% of the map. `tools/wall_profile.py` localizes the failure to a single
256 u stretch:

| route vertex | agent off-line | vs champion | agent dz vs champion |
|---|---|---|---|
| 1578-1586 | 117-125 u | +44 to +53 u | +15 to -14 u |
| 1588 | 141 u | +68 u | -22 u |
| 1592 | 223 u | +148 u | -45 u |
| 1596 | 360 u | +289 u | -75 u |
| **1598** | **2,836 u** | **+2,757 u** | **-1,363 u** |

It leaves the ramp between vertices 1596 and 1598 (204,288 -> 204,544 u,
88.2% of the route), after a ~0.45 s precursor in which the error is small
and growing. Speed through the whole approach is 2,820 u/s against the
champion's 2,930 - six percent - and the champion goes on to ACCELERATE to
3,728 u/s down the final descent while the agent free-falls 2,400 u in 1.5 s
losing speed. **The failure is geometric, with a short warning window**,
which is the case lookahead geometry exists for: at 2,820 u/s the fan's
0.25 s and 0.5 s horizons already report vertices ~1591 and ~1597 while the
agent is still on the line at 1586.

Off-line error at vertices 1586-1596 is therefore the diagnostic that should
move first, well before `race/eval_progress` does.

### xROUTE result

`bash tools/run_arm.sh xROUTE --route maps/surf_src_cannonball.route.npz`
on one RTX 3090 (machine 143878, a known-good box), warm-resumed from the
stuck checkpoint, `--record-every 75e6 --eval-eps 9`, ~250k fps.

| eval | steps after resume | race/eval_progress | corridor mean | **corridor max** | finishes | dives below |
|---|---|---|---|---|---|---|
| 1 | +0.8M | 177,591 | 187,989 | **205,312** | 0/9 | 5/9 |
| 2 | +76M | 161,117 | 170,482 | **205,312** | 0/9 | 6/9 |
| 3 | +152M | 191,073 | 201,628 | **205,312** | 0/9 | 9/9 |
| 4 | +227M | 173,751 | 182,670 | **205,312** | 0/9 | 8/9 |
| 5 | +302M | 180,078 | 189,241 | **205,312** | 0/9 | 8/9 |

**Verdict: NULL on the barrier.** `race/eval_progress` oscillates 161k-191k
(mean 177k), which sits at the top of the round-16 3090 range and would read
as a mild positive on that metric alone. It is not one. Across 5 evals and
**45 episodes the corridor maximum is 205,312 u every single time** - not one
episode passed route vertex 1604, and there were no finishes. What moved is
CONSISTENCY: the weak episodes disappear (eval 1 had episodes at 49.7% and
62.2%; by eval 3, seven of nine reach the wall), so the mean rises while the
frontier does not. That is precisely the distinction `eval_honesty.py` was
written to expose, and eval_progress alone would have hidden it.

The feature is genuinely absorbed, not inert - and asymmetrically:

| step | actor route-weight rms | critic route-weight rms | ratio |
|---|---|---|---|
| +170M | 0.0297 | 0.0639 | 2.15 |
| +265M | 0.0378 | 0.0824 | 2.18 |

against a core-weight rms of 0.177 (actor) / 0.155 (critic) that barely
moves. So in 300M steps the network built the fan into its representation,
the critic taking ~2.2x more of it than the actor throughout - the
asymmetric-critic result (Vasco RLC 2024) showing up unprompted inside a
SYMMETRIC arm. That is the argument for running `--route-critic-only` next:
the mechanism the value function wants is already visible, and the actor
half may be what costs honest perception for nothing.

Approach geometry at the wall was unchanged too (eval 1 vs eval 3, off-line
error at vertices 1584-1596: 117-360 u vs 237-367 u), so the fan did not buy
a cleaner entry into the descent it was supposed to preview.

Caveats worth carrying: (a) one hour warm-resumed is not the regime any of
the source papers tested - Sophy's course-point ablation is a difference over
a full training run, so this is evidence about a 300M-step graft, not about
the feature; (b) the route is the champion's own line, so this arm is not
honest-perception; (c) `--record-every 75e6` with 9 eval episodes costs
roughly a quarter of wall-clock on a 3090 - worth it for the 10-minute
stationarity rule, but budget for it.

### xROUTE, completed (800M steps, 11 evals, 99 episodes)

The run finished its budget cleanly: `done: 4,583,325,696 steps, avg 264,044
steps/s`, 50 minutes of training on one 3090 (machine 143878, now recorded
known-good). Full series, steps after resume -> `race/eval_progress`:

```
+0        177,591      +377M     177,261
+75M      161,117      +453M     179,866
+151M     191,073      +528M     195,220
+226M     173,752      +604M     195,255
+302M     180,078      +679M     173,696
                       +755M     195,032
```

**On `race/eval_progress` alone this is the best result ever recorded on this
checkpoint on a 3090** - three evals at ~195,2xx against xGE's 193,802 opening
and xCTL's 174,159 maximum, with a clear upward drift from a ~176k first-half
mean to a ~190k second-half mean. It is still a null, and the honest metric
says why:

| eval | corridor mean | corridor max | finishes |
|---|---|---|---|
| +604M | 205,312 | 205,440 | 0/9 |
| +679M | 205,284 | 205,312 | 0/9 |
| +755M | 182,656 | 205,312 | 0/9 |
| final | 205,170 | 205,312 | 0/9 |

Corridor MEAN converged onto corridor MAX. By the end essentially every
episode reaches the wall, where at the start only six of nine did - which is
exactly the shape of the eval_progress rise. The FRONTIER moved by one route
vertex, 128 u out of the 26,368 u remaining, and **0 of 99 episodes across 11
evals ever finished**.

**Verdict: null on the barrier, real on consistency.** The lookahead fan
makes the stuck policy reliably execute everything it already knew how to do
and buys nothing past the point where it fails. Reported on the project's
standing metric this would have been written up as the round's best arm; it
is not, and the difference is `tools/eval_honesty.py`. That is the
methodological result of the round and it applies to every arm run against
this checkpoint.

Cost: $0.163/h x ~1.1 h = about $0.18 for the arm, ~$0.30 including the
boxes destroyed under the 60-second readiness rule.

**What this earns for the next arm.** The critic took ~2.2x more of the fan
than the actor at every reading, so `--route-critic-only` (implemented, one
flag) tests the asymmetric-critic hypothesis directly and costs the actor's
honest perception nothing. And the wall is now characterised precisely enough
to aim at: a ramp departure in the 256 u between route vertices 1596 and
1598, entered 6% slower than the champion with a ~0.45 s precursor of small
growing error - which is a control-precision problem at a specific place, not
an exploration problem across the map. The survey items that address THAT are
the speed-squared contact penalty (Sophy) and the search-then-distill loop,
not more observation.

## Round 18 - plasticity loss: soft shrink+perturb on-policy (2026-08-22 ~03:00)

Paper: **Juliani & Ash, NeurIPS 2024, "A Study of Plasticity Loss in
On-Policy Deep RL" (arXiv 2405.19153)** - the only on-policy plasticity
study in docs/research-litsurvey.md section 6, and its winner is soft
shrink+perturb. Companion read: Lyle et al. 2024 (arXiv 2407.01800),
parameter-norm growth as effective-LR decay. One arm, one seed, one hour.

### What the paper actually says (fetched, not paraphrased)

Appendix A, verbatim: "When the intervention is applied all learnable
parameters in the network are iterated through and scaled by alpha. All
parameters are then additively combined with newly sampled initialization
parameters which are scaled by beta", i.e.
`x_new = alpha*x_current + beta*x_init`, and "For all experiments
alpha = 1 - beta". The SOFT variant is the one "applied after each step of
gradient descent instead of only at specific intervals". Table 1 pins
**beta = 1e-6** for both of the paper's settings (gridworld and CoinRun);
plain S+P uses beta = 0.5. This MATCHES the constants recorded in the
survey - no correction needed.

Cross-checked against the authors' reference implementation
(github.com/awjuliani/deep-rl-plasticity). Three details the prose does not
make explicit, all reproduced here:

* `shared/modules.py::sp_module` is literally
  `param.data *= shrink; param.data += epsilon * init_param.data` - it moves
  `.data` only, so **Adam's moments survive the perturbation**;
* `algos/ppo/model.py::_shrink_perturb` builds a **whole fresh module** per
  call (`gen_encoder()`/`gen_value()`/`gen_policy()`), so x_init is
  re-drawn every step. A frozen donor would be an L2 pull toward one fixed
  point - that is regenerative regularisation, a different row of the
  paper's own table;
* `hyperparams.yaml`: `adapt_info: ['soft-sp', [[True, True, True],
  0.999999, 0.000001]]` - all three module groups, alpha 0.999999,
  beta 1e-6; `algos/ppo/trainer.py` calls it immediately after
  `optimizer.step()` inside the minibatch loop.

### SCOPE CAVEAT - read this before quoting the verdict

The paper's headline is soft shrink+perturb **paired with LayerNorm before
ReLU**. Inserting LayerNorm into an already-trained tower is not
function-identical on a warm resume, so it is a SEPARATE arm and was not
authorised or run. **This tested the shrink+perturb half alone.** A null
here does not falsify the paper's combination. Also of note: the paper's
own losers on-policy (final-layer reset, CReLU, ReDo, plasticity injection)
were not tested either.

### The arm

Branch `plasticity` (off `origin/route-obs` e810d2f; the --route feature is
present but **--route was NOT passed**, and
tests/python/test_route.py::test_route_dim_zero_is_the_old_model proves the
model is byte-identical to the pre-route one with it off).

    bash tools/run_arm.sh xSP --shrink-perturb 1e-6

which expands to the pinned baseline plus the one flag:

    python3 -u python/train_fast.py --ckpt runs_ckpt.pt --run xSP \
      --steps 4582737920 --record-every 75e6 --eval-eps 9 \
      --eval-greedy-only --ckpt-every 1e9 --shrink-perturb 1e-6

Warm resume of runs/sOBSR2/ckpt_latest.pt (md5 verified by run_arm.sh,
step 3,782,737,920), surf_src_cannonball, single RTX 3090 (vast 48353611,
machine 34330, Czechia). Local correctness first, CPU only:
tests/python/test_shrink_perturb.py, 19 tests - beta=0 bitwise inert,
beta=1 is exactly a fresh init draw, the interpolation exact at
1e-6/1e-3/0.1/0.5, alpha == 1-beta == the reference's 0.999999, x_init
re-drawn every call, biases covered, Adam moments untouched, warm resume
function-identical at step 0. Full suite green (132 local, 131/132 on the
box - the one failure is the known 3090 lidar-march bit-exactness test).
On the real checkpoint: the resumed policy is **bitwise** identical to the
control at step 0, and 64 perturbation steps moved total norm(theta) by
-6.50e-5 against the theoretical (1-1e-6)^64 - 1 = -6.40e-5.

### race/eval_progress - THE metric

Compared against the 3090 yardstick (opening 172k-194k, working band
~140k-195k), NOT against xCTL's 5090 numbers - the lidar march is not
bit-exact across GPU architectures and one differing depth pixel forks a
greedy trajectory.

| steps after resume | race/eval_progress | corridor mean | corridor MAX | finishes | dives-below |
|---|---|---|---|---|---|
| +0.8M | 159,137 | 168,235u | **205,440u** | 0/9 | 6/9 |
| +76.3M | 156,242 | 164,352u | **205,440u** | 0/9 | 7/9 |
| +151.8M | 190,760 | 201,515u | **205,440u** | 0/9 | 9/9 |
| +227.3M | 173,391 | 182,798u | **205,440u** | 0/9 | 7/9 |
| +302.8M | 179,063 | 189,696u | **205,440u** | 0/9 | 5/9 |
| +378.3M | 172,026 | 182,244u | **205,440u** | 0/9 | 5/9 |

Run stopped at +408.2M steps after 60 minutes. Training metrics healthy
throughout and indistinguishable from control: rew 26.8-31.0, ep_len
2674-3216 (control 27 / 2750), win 0.00% throughout, kl 0.014-0.033, ent
pinned at 0.005, reservoir full.

### VERDICT: NULL. The wall did not move by one route vertex.

eval_progress oscillated 156k-191k, squarely inside the 3090 band, with no
trend and no decay - not a positive, not a negative, a null.

And the honest metric is flatter than that. `tools/eval_honesty.py` scored
every eval against the reference route: **max corridor progress was
205,440u (88.7% of 231,680u) in ALL SIX evals, to the unit** - the same wall
the route arm hits, the same wall the untreated stuck policy hits, in the
same place, with 0 finishes in 54 greedy episodes. What moved in
eval_progress was CONSISTENCY, not reach: at +151.8M nine of nine episodes
got to the wall (mean 201,515u) instead of six of nine, and eval_progress
rose 156k -> 191k purely because fewer episodes died early. Every one of
those episodes then fell past the finish into goal-adjacent space (end
z ~ -4,180). **A rise in eval_progress with an unchanged 88.7% wall is not
a result**, and this is the cleanest example of it yet: +35k of
eval_progress for zero new ground.

### The weight-norm diagnostic (the round's real yield)

Added `--wnorm-every N` (default 10 iterations, on for every run from now
on) writing per-layer norm(theta) to runs/<run>/wnorm.csv. It costs 1.7 ms
per firing and it answered the Lyle question directly.

**The stuck checkpoint is FAR from initialisation, and shrink+perturb at the
paper's beta cannot pull it back.** norm(theta) at resume vs a fresh draw
from this network's own init distribution:

| layer | resume norm | x fresh init | change over the run |
|---|---|---|---|
| conv.0.weight | 6.26 | 1.1x | -2.36% |
| conv.2.weight | 12.00 | 1.5x | -1.11% |
| conv.4.weight | 21.45 | 1.9x | +0.52% |
| conv.8.weight (2048 to 512) | 126.87 | **4.0x** | +1.29% |
| pi.0.weight | 83.27 | 2.8x | +1.34% |
| pi.2.weight | 78.76 | 2.7x | +1.00% |
| vf.0.weight | 73.60 | 2.5x | +1.18% |
| vf.2.weight | 68.83 | 2.3x | +0.21% |
| action_head.weight | 16.13 | **283x** | -0.56% |
| value_head.weight | 4.74 | 4.7x | -1.42% |
| **TOTAL** | **201.34** | **2.9x** | **+1.09%** |

(action_head's 283x is inflated by its 0.01 orthogonal gain - it starts
near zero by design - but it is still the layer furthest from where it
started. Fresh-init reference norms, same architecture: 5.66 / 8.00 / 11.31
/ 32.00 / 29.93 x4 / 0.057 / 1.00, total 69.52.)

The arithmetic that makes this quantitative: 510 iterations x 64 optimizer
steps = 32,640 applications, so the shrink term ALONE multiplies norm(theta)
by (1-1e-6)^32640 = 0.9679, i.e. **-3.21%** over the run. Observed net was
**+1.09%**. So the gradient's norm-growth pressure over this hour was about
**+4.4%**, and beta = 1e-6 cancelled roughly **three quarters** of it - a
real, measurable drag, not a no-op - and the norm still grew. Reading it
the other way: at this beta it would take ~7e9 steps of shrink to walk
norm(theta) back to a fresh init's 69.5, which is longer than the run that
produced the checkpoint. **If the effective-LR story is the wall here, the
paper's constant is one to two orders of magnitude too small for this
network at this scale.** That is the concrete follow-up: a beta ladder
(1e-5, 1e-4) is now a cheap, well-instrumented arm, and the norm series is
the leading indicator - it should be readable in 10 minutes, long before
eval_progress says anything.

Caveat on the diagnostic: no control run has a norm series yet (the
column did not exist before this round), so the +4.4% growth figure is
inferred from the treated run's arithmetic, not measured against an
untreated twin. Any beta-ladder rerun should include a beta=0 arm purely to
get that curve.

### Cost of the method (worth knowing before rerunning it)

The paper's per-step full re-initialisation is not free on a real conv
policy: **11.4 ms per optimizer step on a 3090** for 1.96M parameters, of
which 9.9 ms is cuSOLVER QR for the orthogonal init (the 2048x512 trunk
Linear alone is 3.2 ms). At 64 optimizer steps per iteration that is 0.73 s
added to a ~4.6 s iteration: measured **146k steps/s with the flag vs ~169k
without**, a ~14% throughput tax, so the hour bought +408M steps instead of
~+500M. Batching the QRs does not help (measured: 7.06 ms for a batched
(4,522,448) vs 6.8 ms for the four separately). Anyone rerunning this should
prefetch the donor draws on a CPU thread - they are completely independent
of the training computation.

### Ops

Five instances destroyed under the 60-second readiness rule before one came
up: machine 8078/host 3483 (still loading at 69 s - note this host is ALSO
in known_good under a different machine id, 7777) and machine 130609/host
302304 (ssh refused at 98 s) blacklisted with reason `network`; three race
losers destroyed at ~95 s. One over-cap create (the shared registry was full
with another agent's race at that moment) destroyed within 40 s. The winner
ran the full hour at 99% GPU util, 315-318 W of 350 W, 54 C, and was
destroyed at 02:56Z. Total rental spend for the round **~$0.34**.

Artifacts kept in runs/research/xSP/: progress.csv, wnorm.csv, run.json,
the launch log, and all six eval trajectories.

Not tested and still open from this paper: LayerNorm before ReLU (the other
half of the winner), and a beta ladder above 1e-6.

## Round 18 - Necto/RLGym difficulty-weighted state setter (2026-08-22 02:10-03:18 UTC)

Survey section 4 / universal fact #3: every superhuman system starts episodes
from a multi-source, DIFFICULTY-BIASED distribution, and Necto is the one that
publishes the recipe. Fetched and read for this round
(github.com/Rolv-Arild/Necto, `training/state.py`,
`NectoReplaySetter.generate_probabilities`) - it is two lines:

    weights = 1 + 10 * (ball_heights + player_heights.sum(axis=-1)) / CEILING_Z
    return weights / weights.sum()

Two properties of that, both reproduced verbatim: the weight is **per state**,
and it **multiplies the replay pool's own density** rather than flattening it
over bins. (The mixture around it is 70% replay / 8% smart-random / 4% true
start / 4% kickoff-like / 5% goalie / 4% hoops / 5% wall.)

We already respawn 90% of episodes from a mid-run snapshot reservoir but draw
uniformly over stored states, i.e. exactly proportional to visitation - the
mastered opening dominates and the rare hard regime is starved.

**The arm (`--respawn-difficulty 10`, branch `necto-respawn` off
`origin/route-obs`, commit e3d8e6c).** Each stored state is drawn with
`w = 1 + 10 * D(bin(state))`, `p = w/sum(w)`, where `D` is the state's
distance bin's FAILURE RATE min-max normalized over evaluated bins: the
decayed (EMA 0.99) fraction of episodes STARTED in that bin that ended - died,
stall-killed or truncated - without ever improving on their start distance by
`--respawn-improve` (default one bin width = 12,399u of the 198,380u route).
16 bins. Chosen over distance-to-goal and reservoir depth because it is the
closest analogue of Necto's "hard regime" and, unlike a success band, it stays
defined at win rate 0 - the exact defect that silently degenerated round 16's
Florensa arm to uniform. Guards: unevaluated bins (< 5 episodes) take the mean
difficulty rather than 0, so no bin is starved; a failure spread under 5 points
reports zero difficulty everywhere and the sampler degrades to uniform rather
than amplifying noise into an 11x curriculum; the flag refuses to stack with
`--respawn-mode` or `--respawn-killsafe`.

Exact launch (the ONLY command that started it, on the box):

    bash tools/run_arm.sh xNECTO --respawn-difficulty 10

which is `--ckpt runs_ckpt.pt (md5 1ba1fd2936af3ae1ad3608e3cd6b1e9e, verified
on the box) --run xNECTO --steps 4582737920 --record-every 75e6 --eval-eps 9
--eval-greedy-only --ckpt-every 1e9 --respawn-difficulty 10`, everything else
restored from the checkpoint config. Single RTX 3090 (vast 48353960,
$0.134/h, 96-core EPYC 7K62), 183.5k steps/s steady, +605M steps in 68 min.

**Round 16's confounds, avoided and verified.** `ez_eps 0`, `spawn_burst 0`
in the run's own `run.json`, so `USE_BURST` is false and nothing is excluded
from the PPO batch; `train_stride 1`, `epochs 4`, default `n_steps 128` /
`minibatches 16`. The arm therefore ran **exactly** 64 optimizer steps per
786,432 env-steps, identical to the control - no minibatch-count compensation
needed because nothing was dropped. The difficulty statistic never reads a
win (`wins 0.0` all run). With the flag off the sampler takes the untouched
branch: `tests/python/test_respawn_necto.py` pins that byte-for-byte over five
successive pools on BOTH the control path (no dist_fn) and the binned path,
and pins that RaceReward's new episode-best latch cannot change any reward it
returns. 126 tests pass locally; 125/126 on the box, the single failure being
the known `test_march_is_bit_exact_against_the_legacy_kernel` that CLAUDE.md
already records as failing on a 3090.

### The metric

`race/eval_progress`, 9 greedy evals (`runs/research/xNECTO/progress.csv`):

| steps after resume | xNECTO | corridor progress mean | finishes | dives-below |
|---|---|---|---|---|
| +0.8M | 164,648 | 174,421u | 0/9 | 5/9 |
| +76M | 180,154 | 191,829u | 0/9 | 8/9 |
| +152M | 189,437 | 199,950u | 0/9 | 9/9 |
| +227M | 191,341 | 202,496u | 0/9 | 7/9 |
| +303M | 101,229 | 107,349u | 0/9 | 3/9 |
| +378M | 175,605 | 185,529u | 0/9 | 7/9 |
| +454M | 190,717 | 202,140u | 0/9 | 6/9 |
| +529M | 151,911 | 160,540u | 0/9 | 5/9 |
| +605M | 152,975 | 161,166u | 0/9 | 6/9 |

**VERDICT: NULL.** The series oscillates inside the 3090 working band
(~140k-195k), never reaches the ~195k positive threshold, and does not decay.
The +303M dip to 101k is not decay: 4 of its 9 greedy episodes died in the
first 5-16 s (2.2-8.8% of the route), the chaotic-fork noise CLAUDE.md warns
about; the next eval was back to 175.6k. Compare xGE 193,802 -> 160,234 and
xEZ 172,480 -> 109,259 -> 49,797 on the same card.

### The honesty check (tools/eval_honesty.py, all 81 recorded episodes)

**0 finishes in 81 episodes, and the maximum corridor progress in the entire
run was 205,440u = 88.7% of the 231,680u route - the same wall the route arm
reported, reached in eval 1 and never beaten in eval 9.** Rising
`eval_progress` early in the run (164.6k -> 191.3k over +227M) tracked a rising
DIVE rate (5/9 -> 7/9) and a rising corridor mean (174.4k -> 202.5k): the arm
made the typical episode ride the line further before falling past the goal, it
did not push the frontier past the final descent. Episodes that did not dive
ended at z ~ -1798..-1824, i.e. right at the finish box's lower z bound but
outside it.

### Did the weighting actually bite? Yes - and the cap is the reservoir

Realized oversampling, logged every 100 iterations (hardest:easiest bin, draws
per stored state): 11.1x, 10.2x, 19.1x, 9.0x, 10.1x, 12.3x, 10.3x, 15.2x,
10.7x, 14.5x - against 11.0x intended by construction. So the mechanism ran at
the paper's dose for the whole hour.

The final draw histogram against what uniform-over-states would have given:

| bin | d range (u) | states | draws/pool | vs uniform | fail rate |
|---|---|---|---|---|---|
| 0 | 0-12,399 | 0 | 0 | - | - |
| **1** | **12,399-24,798** | **442** | **134** | **8.22x** | **0.96** |
| 2 | 24,798-37,196 | 6,284 | 313 | 1.35x | 0.08 |
| 3 | 37,196-49,595 | 15,833 | 612 | 1.05x | 0.05 |
| 4-14 | 49,595-185,981 | 75,150 | 2,573 | 0.75-1.18x | 0.01-0.06 |
| 15 | 185,981-198,380 | 1,291 | 27 | 0.57x | 0.01 |

The statistic found the right regime unaided: bin 1 - the deepest states the
reservoir holds, the wall - is the only bin with a high failure rate (0.96),
and it took 8.2x its uniform share. Everything else stayed within +-35%.

**But bin 1 is 0.44% of the reservoir.** Oversampling it 8.2x buys 3.6% of
starts, and the fleet's mean start distance moves only 85,080u -> 80,340u
(5.6% closer to the goal). That is the whole result: *the weighting did what
the paper says and it was not enough, because the reservoir does not contain
the states that matter.*

Reservoir depth over the run (`reservoir d: min / p10 / median`), which is the
part worth carrying forward:

    restored ckpt   19,338 / 39,217 / 74,362   (20,000 states)
    it 50          17,043 / 42,079 / 85,696   (100,000, full)
    it 101         13,565 / 39,196 / 74,397
    it 401         15,693 / 39,301 / 74,568
    it 601         12,180 / 39,509 / 75,554
    it 901         13,076 / 39,708 / 75,840

The feedback loop is real and visible - the arm's own harvest pushed min-d from
19,338 to 12,180-13,565 and grew bin 1's population from 77 to 442 states - and
then it plateaued. Round 16 measured why: the 10-second harvest margin discards
everything within 10 s of an episode's end, and the final descent kills within
10 s of being entered, so a state at the 88.6% vertex can essentially never be
harvested. **Necto's weighting can only oversample states the reservoir has,
and the harvest margin - not the sampling rule - is what caps this reservoir at
~12k units short of the goal.** Any rerun of this family must change the
margin (or seed the reservoir from a demo/search trajectory) first; changing
the weighting alone is measuring the margin.

### Caveats a later reader needs

* Ran on branch `necto-respawn`, based on `origin/route-obs`, with `--route`
  ABSENT. The route feature is off unless passed and
  `test_route.py::test_route_dim_zero_is_the_old_model` proves the policy is
  byte-identical without it, so it does not contaminate the arm; the shared
  ops tooling (`run_arm.sh`, `fleet_watchdog.py`, `deploy_box.sh` BRANCH=) only
  exists on that branch, which is why it was the base.
* **The improve threshold was not calibrated and one bin width is too lenient
  for most of the map.** For the first ~100 iterations the failure rates
  collapsed to 0.00-0.08 across 14 bins (episodes average ~2,600 ticks and
  cover ~1/3 of the route, so almost everything clears 12,399u), and the
  min-max normalization was amplifying a 5-8 point spread into an 11x
  curriculum - i.e. for that stretch the arm was oversampling near-noise. It
  only became a real signal once bin 1 accumulated enough episodes to read
  0.88-0.96 against everyone else's 0.01-0.08. A rerun should set
  `--respawn-improve` so the fleet-wide failure rate sits near 0.5 (here that
  is roughly 60-80k units, ~5-6 bins), or use a per-bin-relative statistic.
* **The failure statistic inherits the death-dive artifact.** "Improved by
  12,399u" is read off the same geodesic field that pays a fall past the finish
  as ~89% of the route, so an episode that dives counts as a success. That is
  part of why the deep-but-not-deepest bins read as easy.
* The 5-point spread guard (`spread_min = 0.05`) never fired - every logged
  read had a live oversampling ratio - so the arm was never silently the
  control.
* One seed, one hour, per CLAUDE.md rule 2. The eval band's own spread
  (101k-191k within a single run) is wider than any effect this arm produced,
  which is the honest reason to call it null rather than "slightly positive
  early".

Artifacts: `runs/research/xNECTO/` (progress.csv, run.json, all 9
traj_*.jsonl, xNECTO_launch.txt with every diagnostic line). Box destroyed
03:19:40 UTC and released from the registry; total rental spend for the round
$0.18 (the winner $0.134/h x 81 min, plus one raced loser destroyed at 60 s).

# ROUND 19 (2026-08-22): Linesight's temporal mini-race
# ============================================================

Survey section 9, **item 1 on the ranked shortlist of untested levers** -
"COLLAPSE THE HORIZON" - and survey section 0 row 1, where RL_Surf is the
single largest structural outlier in the whole literature: every superhuman
system runs an effective discount horizon of 1.4-13 s, this project runs
`gamma = 0.9995` per PHYSICS TICK = `1/(1-0.9995)` = 2,000 ticks =
**20.0 s**. (The survey's own table says 60.6 s; that entry is wrong, and
the correction is at the end of this section.)

Round 18 localized the failure to a 256 u stretch of ramp between route
vertices 1596 and 1598, entered with a ~0.45 s precursor of small growing
error. That is a credit-assignment problem at a half-second timescale being
optimized under a 20-second horizon.

## The paper, read out of the source

github.com/Linesight-RL/linesight (world records on ~10 of 12 official
Trackmania campaign tracks, May 2024). The mini-race is not described in
their README; it is in the code, and two details differ from how the survey
records it.

`config_files/config.py`:

```
tm_engine_step_per_action = 5 ; ms_per_tm_engine_step = 10   # 50 ms, 20 Hz
temporal_mini_race_duration_ms      = 7000                  # 140 actions
oversample_long_term_steps          = 40
oversample_maximum_term_steps       = 5
gamma_schedule = [(0, 0.999), (1_500_000, 0.999), (2_500_000, 1)]
n_steps = 3
constant_reward_per_ms                   = -6 / 5000        # -0.0012/ms
reward_per_m_advanced_along_centerline   = 5 / 500           # 0.01/m
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2_000
```

`trackmania_rl/buffer_utilities.py` - "This is where the magic of
'mini-races' or 'clipped horizon average reward' is handled" - runs at
SAMPLE time, per minibatch:

```
t      = (abs(randint(-35, 145)) - 5).clip(min=0)        # 0..139
state_float[:, 0]      = t
next_state_float[:, 0] = t + n_steps
reduced_n = n_steps - (t + n_steps - 140).clip(min=0)     # cut to the edge
terminal  = (reduced_n >= terminal_actions) | (t + n_steps >= 140)
gammas    = where(terminal, 0, gammas)                    # NO bootstrap past it
```

`config_files/state_normalization.py`: `float_inputs_mean[0]` and
`float_inputs_std[0]` are both `temporal_mini_race_duration_actions / 2`, so
the clock reaches the network as `2t/H - 1`.

**Two things this contradicts in the brief and in the survey.**

1. **The abs() fold oversamples t near 0 only, not "t near 0 and t near H".**
   Exactly: of 180 equally likely draws, t = 0 gets 11, t = 1..30 get 2 each,
   t = 31..139 get 1 each. `oversample_maximum_term_steps = 5` is a shift, not
   a second mode: its effect is to pin the maximum draw at exactly H - 1 (a
   clock of H would be a window of length zero). Since the return runs from t
   to the edge, oversampling t near 0 means oversampling the FULL 7 s horizon.

2. **The ACTING network never sees a nonzero clock.**
   `trackmania_rl/tmi_interaction/game_instance_manager.py` builds every
   rollout observation as `floats = np.hstack((0, ...))`. The random clocks
   exist only inside the learner's minibatch. So the deployed controller is
   `argmax_a Q(s, t=0, a)` - "maximize progress over the next 7 seconds",
   re-decided every 50 ms - and the clock's job is to make `V(s, t)` well
   defined over a finite window, which is a critic-side job. The survey's
   "normalized time-elapsed-in-window in proprioception" is true of the
   network's input layout and false of what the policy ever acts on.

## What was implemented (`--minirace`, commit bab837a)

`python/train_fast.py`, three flags:

```
--minirace SECONDS       window length; absent/0 = off (control untouched)
--minirace-gamma         per-DECISION discount inside it   (default 1.0)
--minirace-actor-clock   1 = the actor reads the clock too (default 0)
```

Implemented, verbatim where the algorithms allow:

* **The 7 s window.** `minirace_window(7.0, act_every*chunk)` = 233 decisions
  x 3 ticks = **6.99 s**. Stated in seconds and converted through the same
  tick arithmetic `--gamma` uses, so `--act-every` cannot silently rescale it
  (round 17 lost an arm to exactly that).
* **The window edge is terminal, with no bootstrap.** `b_wend[t]` marks the
  decision whose step crosses the edge and enters the GAE exactly like a
  `done`. Linesight's `gammas = where(terminal, 0, gammas)`.
* **gamma = 1.0 inside the window**, per decision, NOT raised to `act_every`
  (the window is measured in decisions, not ticks). The endpoint of their
  `gamma_schedule`. The value target inside a window is therefore the plain
  undiscounted sum of rewards to the edge - `-elapsed_time + progress` with
  no discount/horizon tension.
* **The elapsed-time-in-window clock in the observation**, normalized their
  way (`2t/H - 1`), as the LAST scalar column: `[15 core | 1 clock | image]`.
* **The random horizon clock**, their fold, with the two oversampling
  constants converted by DURATION rather than copied as raw counts (40
  actions = 2,000 ms -> 67 decisions; 5 actions = 250 ms -> 8 decisions;
  copying "40" at 30 ms/decision would have been a different fold).
* **The window is not an episode.** The environment, the respawn reservoir,
  `--ep-ticks` and the stall kill are untouched; only the RETURN is
  truncated. That is Linesight's own arrangement - their rollouts run to
  300 s and the mini-race is a sample-time construct.
* **The acting policy is fed clock 0.** The actor tower does not read the
  clock at all (`--minirace-actor-clock 0`), and every eval row carries the
  clock's t = 0 encoding (-1.0), matching `floats[0] = 0`.

Warm resume is **function-identical at step 0** and then some: because the
actor is blind to the clock, `widen_for_route` pads exactly ONE weight tensor
(`vf.0.weight`) and its two Adam moments with a zero column. The resumed
POLICY is the stuck checkpoint's weights untouched, byte for byte; only the
critic is one zero column wider. Log line:
`--minirace: widened 3 checkpoint tensors by 1 zero columns`.

`tests/python/test_minirace.py`, 21 tests (`133 passed, 1 failed` on the box
- the failure is the known `test_march_is_bit_exact_against_the_legacy_kernel`
that CLAUDE.md records as failing on every 3090). The ones that decide
whether the arm measured the paper or a bug:

* with the flag off the advantages are **bit-identical** to the pre-minirace
  GAE loop (and also with an all-zero `b_wend` threaded through it);
* inside a window at lambda = 1 the return equals the **plain undiscounted
  sum of rewards to the edge**, whatever the value function says;
* changing every value after the edge by 1,000 moves **no bit** of any
  advantage before it;
* the phase draw reproduces Linesight's support (0..139) and its exact
  multinomial weights (11/180 at t = 0, 2/180 on the folded band, 1/180 on
  the tail);
* **the mean REMAINING horizon lands on Linesight's own**: 4.17 s under their
  draw at 140 actions x 50 ms, 4.16 s under this port at 233 decisions x
  30 ms.

## What was NOT implemented, and why

* **No gamma schedule.** Theirs ramps 0.999 -> 1.0 over frames 1.5M-2.5M of a
  15M+ frame run. This is a one-hour warm resume; it ran at the schedule's
  ENDPOINT from step 0. Note their ramp starts from a LONGER horizon
  (0.999 per action at 20 Hz = 50 s) than this project's 20 s, so it would
  not have softened anything here: what shortens the horizon in this arm is
  the window truncation, not the discount.
* **Random clocks are re-rolled per WINDOW, not per SAMPLE.** Linesight is
  off-policy (IQN + replay) and can hand the same stored transition a fresh
  clock every time it is sampled. PPO is on-policy and the clock is IN the
  observation, so a clock the environment did not have would falsify the
  ratio. The honest translation re-rolls the PHASE at every window edge,
  which makes the window LENGTH the random draw instead of the offset into
  it. The quantity that matters - the distribution of remaining horizons -
  matches to within 0.01 s (test above).
* **The 2 s no-checkpoint cutoff was NOT implemented.** Linesight kills a
  rollout after 2,000 ms without passing a virtual checkpoint; this project's
  `--stall-secs` is 15 and was left at 15. Deliberate: cutting the stall kill
  7.5x on a surf map, where legitimate airborne stretches make no geodesic
  progress for seconds at a time, is a second treatment with its own large
  failure mode, and one run cannot separate two treatments. **This arm is the
  window, not the package.**
* **No IQN, no distributional critic, no prioritized replay, no
  epsilon/Boltzmann schedules, no 32-uses-per-memory.** Linesight's agent is
  IQN with n-step 3 over 12 discrete actions. This arm is the project's PPO
  with the mini-race transplanted into its GAE. The distributional critic is
  a separate item on the survey's shortlist and is still untested here.
* **The reward is unchanged.** Their `-0.0012/ms + 0.01/m + clipped VCP
  potential` was not adopted; this run keeps the project's geodesic PBRS,
  `time_pen 0.005`, `int_coef 0.25`, `success_bonus 50`. Only the return
  DEFINITION changed.
* **GAE lambda left at 0.95.** Linesight uses n-step 3, not GAE; lambda is
  not part of the mechanism and moving it would be a second treatment.

## The run

```
bash tools/run_arm.sh xMINIRACE --minirace 7.0
```

Branch `minirace` @ `bab837a`, warm resume of the stuck checkpoint
(`runs_ckpt.pt` md5 `1ba1fd2936af3ae1ad3608e3cd6b1e9e`, step 3,782,737,920,
baseline config guard passed), one RTX 3090 (vast instance 48359392, machine
75262, host 392144, $0.209/h), `--record-every 75e6 --eval-eps 9
--eval-greedy-only`. `gpu_health.py`: 841 GB/s HBM, 73 TFLOPS bf16, 1,755 MHz
and 338 W under load, VERDICT healthy. Test suite on the box: `133 passed,
1 failed` - the failure is `test_march_is_bit_exact_against_the_legacy_kernel`,
which CLAUDE.md records as failing on every 3090.

Trainer banner, so the arithmetic is on the record:

```
minirace: window 233 decisions = 6.99 s (asked 7 s), gamma 1 per decision
inside it, window edge = terminal (no bootstrap). Replaces the 20.0 s
discount horizon of gamma=0.9995 per tick. Random clock fold:
abs(U[-67, 241)) - 8, clipped at 0. Actor reads the clock: no
--minirace: widened 3 checkpoint tensors by 1 zero columns - the resumed
policy is function-identical to the baseline at step 0
```

**The mechanism was live, and the run measured it, not a no-op.** Printed
every 25 iterations:

```
minirace: clock mean 141.8/233  edges 1,689/262,144 (0.644%, 1/H = 0.429%)
          ep_done 0.212%
```

Window edges fire on 0.64-0.68% of rollout rows across the whole run, against
1/H = 0.429% - correct, because re-rolling the phase at each edge makes the
mean window shorter than the full one (predicted 1/138.5 = 0.72%). More to
the point, the window edge is **3-5x more frequent than the episode end**
(0.65% vs 0.13-0.24%), so after this change the GAE chain is cut by the
mini-race, not by the environment. Before it, at `--ep-ticks 12000` (4,000
decisions) and a 128-decision rollout, the chain was essentially never cut at
all and every advantage was a 20 s discounted bootstrap.

## Result: killed at 336M steps, 5 evals, on a decaying series

| eval | steps after resume | race/eval_progress | corridor mean | corridor max | finishes | dives below |
|---|---|---|---|---|---|---|
| 1 | +0.8M | **195,291** | 205,326 | **205,440** | 0/9 | 9/9 |
| 2 | +76M | 175,817 | 185,244 | **205,312** | 0/9 | 7/9 |
| 3 | +152M | **90,999** | 95,716 | **96,256** | 0/9 | 0/9 |
| 4 | +227M | 103,258 | 110,165 | **152,192** | 0/9 | 0/9 |
| 5 | +303M | **78,850** | 82,745 | **90,112** | 0/9 | 0/9 |

Killed under CLAUDE.md rule 3 ("a decaying series ... is a clear negative and
gets killed on sight") after four consecutive falling evals over 300M steps.
Training diagnostics agree and are smoother than the evals:

```
steps after resume   value_loss   ep_rew   ep_len      fps(cum)
       +0.8M            1.8746     -0.52      374        14,002
      +32M              0.0261     25.75     2780       181,444
      +64M              0.0155     21.16     2069       231,717
     +142M              0.0162     19.19     1963       262,787
     +205M              0.0170     17.33     1898       274,130
     +268M              0.0230     13.19     1436       279,828
     +336M              0.0298     10.61     1252       285,028
```

Episode return **rises** from 21 to 25.8 over the first 32M steps and then
falls monotonically to 10.6; episode length tracks it, 2,780 -> 1,252 ticks.
The rise is real and is what the mechanism is supposed to buy - short-horizon
progress is the easiest thing to improve when the return is short-horizon
progress. The fall is what it costs.

**Verdict: NEGATIVE, and the strongest one this checkpoint has produced from
a config change.** Not a null like xROUTE: the corridor frontier moved
BACKWARDS by 115,000 units, from 205,440 u (88.7% of the route) to 90,112 u
(38.9%). The mini-race did not fail to break the wall at 88.6%; it took away
the two thirds of the map the agent already had.

### It is not a value-fitting failure, and not a bad box

Three things worth ruling out explicitly, because each would have made this
an uninterpretable run rather than a result:

* **The value-scale shock is real but transient.** Swapping a 20 s
  discounted return for an undiscounted ~4.2 s finite-horizon one shrinks
  the value targets roughly 5x, and `train/value_loss` opens at **1.875**
  against the 0.016-0.02 it holds for the rest of the run. It is back under
  0.05 by +16M steps and under 0.02 by +48M. The critic re-fit long before
  the decay started; the decay is not the critic still converging.
* **The clock is absorbed, hard.** Its column of `vf.0.weight` starts at
  exactly zero (that is what makes step 0 function-identical) and grows to
  rms **0.207 at +61M** and **0.331 at +304M**, against a core-column rms of
  0.147-0.151 that barely moves - so by the end the single clock input
  carries **2.25x** the weight of an average core input. For scale, xROUTE's
  27 route features reached a critic rms of 0.0824 against a core 0.155, or
  0.53x, over the same 300M steps. The value function took this one feature
  harder than it took the whole lookahead fan, which is exactly right:
  inside a finite window `V(s, t)` is close to linear in the time remaining,
  so the clock is the single most predictive input it has ever been given.
  **The arm is not a no-op that happened to decay.**
* **The box was fine.** 285k steps/s cumulative and climbing (instantaneous
  ~330k), the top of the 150-300k band, KL 0.016-0.026 throughout, entropy
  pinned at its floor, no NaNs, no restarts.

### What actually went wrong, from the wall profiles

The failure mode CHANGED, and that is the informative part. `eval_honesty`
reports every episode of evals 3-5 as **`short`, not `dive-below`** - the
agent is not falling past the goal, it is stopping high on the map, and all
nine episodes stop within a few hundred units of each other.

Eval 3 (+152M), all 9 episodes stop at ~41%; `wall_profile.py` around
vertex 752:

| vertex | agent speed | agent z | off-line | vs champion |
|---|---|---|---|---|
| 736 | 2,453 | 3,466 | 1,116 | dspeed **+181**, dz -312, doff **+1,036** |
| 744 | 2,450 | 3,247 | 1,100 | dspeed +103, dz -202, doff +1,030 |
| 752 | 2,482 | 2,942 | 1,056 | dspeed **+78**, dz -133, doff **+984** |

It has left the champion line by a thousand units, sits 130-310 u LOWER, and
is going **faster** than the champion the whole way there. That is a greedy
line: more progress now, on a trajectory that does not survive.

Eval 5 (+303M), around vertex 700, where the champion is CLIMBING (z 3,278 ->
4,180 over 13 vertices at a flat 2,224-2,247 u/s):

| vertex | agent speed | agent z | off-line | vs champion |
|---|---|---|---|---|
| 690 | 1,678 | 3,015 | 275 | dspeed **-546**, dz -263 |
| 698 | 1,684 | 3,138 | 582 | dspeed -547, dz -571 |
| 704 | 1,428 | 3,066 | 864 | dspeed **-809**, dz **-888** |

By the end the agent is 550-800 u/s SLOWER than the champion and 900 u lower.
It is refusing the climb.

**Why, measured rather than asserted.** A surf ramp is a cost-now-pay-later
trade: the champion's height gains are paid for in speed. Its longest
finishing episode (`runs/sISV_par2/traj_8454144000.jsonl`, 82.8 s, z 10,528
-> -1,479) contains **13 sustained climbs** totalling 37.3 s - nearly half
the run - and the speed falls through every one of them:

| climb | duration | height gained | speed through it |
|---|---|---|---|
| 1 | 4.15 s | +4,479 u | 3,994 -> 2,912 |
| 2 | 3.99 s | +4,447 u | 3,601 -> 2,314 |
| 3 | 3.88 s | +3,832 u | 3,441 -> 2,225 |
| 4 | 3.36 s | +1,674 u | 3,464 -> 3,048 |
| 5 | 3.31 s | +2,514 u | 3,020 -> 2,277 |

**No climb is longer than the 7 s window** - the naive story, "the trade does
not fit", is wrong and was checked. The one that is right is about the
REMAINING horizon. A randomly-phased window leaves a transition an expected
4.2 s, spread roughly uniformly over (0, 7 s]; the payoff of a climb arrives
2.3-4.2 s after it starts. So for well over half of the transitions taken
during a climb, the payoff lands past the window edge, where the return is
defined to be exactly zero - while the cost, a second or two of reduced
progress per tick against an unchanged time penalty, lands inside it every
time. The trade is negative under the estimator for most of the states in
which it has to be made, and the policy correctly stopped making it. The
agent got exactly what it was asked for.

**And this is why the paper's constant does not port.** Linesight's 36
campaign tracks have a median author time of **26.3 s**
(`trackmania_rl/map_reference_times.py`), so their 7 s window is **27%** of a
lap and their mean remaining horizon is **16%** of it. On
`surf_src_cannonball`, an 82.8 s champion run, the same 7 s window is **8%**
of the task and the mean remaining horizon is **5%**. The constant is not 7
seconds; the constant is a quarter of the race, and this map is three times
longer than the ones it was tuned on. A faithful port of the IDEA - rather
than of the number - would be a window of 20-25 s here, which is longer than
the horizon this project already had.

### Corrections this arm produces

**1. The survey's discount-horizon row is wrong by a factor of 3.**
`docs/research-litsurvey.md` section 0 reads "60.6 s (1/(1-0.9995) = 2000
decisions at 33 Hz)". `--gamma` in `train_fast.py` is per PHYSICS TICK and
the GAE raises it to `act_every` itself (`g_eff = args.gamma ** KH`), so the
horizon is 2,000 TICKS = **20.0 s**, and it does not move with the decision
rate: `g_eff = 0.9995**3 = 0.99850`, `1/(1-g_eff) = 667 decisions`,
`667 x 3 / 100 = 20.0 s`. The survey read the same number as per-decision.
RL_Surf is still the largest outlier against the field's 1.4-13 s, but by 1.5x
over Necto, not 4.5x. Checked by `test_baseline_gamma_is_a_twenty_second_horizon`.

**2. Two details of the mini-race as the survey records it are not what the
code does.** The abs() fold oversamples clocks near 0 ONLY, not "t near 0 and
t near H" - `oversample_maximum_term_steps` is a shift whose effect is to pin
the maximum draw at H-1, not a second mode. And the ACTING network is fed
clock 0 always (`floats = np.hstack((0, ...))` in
`game_instance_manager.py`); the random clocks exist only inside the learner's
minibatch, so "time-elapsed-in-window in proprioception" is true of the input
layout and false of anything the policy ever acts on. Both are documented at
the top of this section with file references.

### What this earns for the next arm

* **The cheap horizon probe the survey lists as item 1's fallback is now
  much less attractive, and its direction is known.** "gamma 0.997 together
  with a time-remaining feature" is a 10 s horizon; this arm ran the honest
  4-7 s version of the same idea and lost two thirds of the map to it. If
  anyone tests the horizon again, test it UPWARD or at 10-13 s (Necto's
  13.3 s is the field's longest), not downward, and expect the ramp climbs to
  be the thing that decides it.
* **The wall at 88.6% is not a horizon problem.** Round 18 called it "a
  control-precision problem at a specific place, not an exploration problem
  across the map"; this arm rules out the third possibility, that it was a
  credit-assignment-timescale problem. Shortening the horizon to the
  half-second timescale of the failure precursor did not sharpen the
  approach - it destroyed the approach.
* **A ~4 s horizon is where the deceptive basin becomes attractive.** Eval 3
  is the cleanest picture this project has of the basin: all nine episodes,
  1,000 u off the champion line, 200 u low, going 80-180 u/s FASTER, dying at
  41%. That trajectory is worth keeping as a negative exemplar
  (`runs/research/xMINIRACE/traj_3934519296.jsonl`).
* **`--minirace` is implemented, tested and off by default**, so the window
  is a one-flag experiment if a longer one is ever wanted:
  `--minirace 15 --minirace-gamma 1.0` is a 15 s undiscounted window, and
  `--minirace-actor-clock 1` is the variant where the actor is
  time-conditioned (which Linesight does NOT do, and which this arm did not
  test).

Cost: instance 48359392 lived 27.0 minutes at $0.209/h = **$0.094**, plus two
candidates destroyed under the 60-second readiness rule (machines 143874 and
114019, both still `loading` at 134 s, both blacklisted in
`tools/bad_hosts.json` before destruction) for about $0.019. **Total about
$0.11.** Artifacts in `runs/research/xMINIRACE/`: `progress.csv`, `run.json`,
all five `traj_*.jsonl`, `xMINIRACE_launch.txt`, `ckpt_latest.pt` at step
4,086,300,672.

Caveats that would mislead a later reader:

* **One hour, one seed, warm-resumed** - as every arm here is. Linesight
  trains a mini-race agent from scratch for tens of millions of frames; this
  measures what happens when a 3.8B-step policy optimized under a 20 s
  horizon is handed a 7 s one. A scratch run under the window is a different
  and untested question, and this arm says nothing about it. What it does say
  is that the mechanism is not a graft that helps this checkpoint.
* **The arm is the window alone.** The 2 s no-checkpoint cutoff, IQN, the
  distributional critic, the n-step-3 return and Linesight's reward constants
  were all deliberately not adopted. It is possible the window only works
  inside that package - in particular next to a distributional critic, which
  is the other Linesight ingredient the survey ranks - but a single run
  cannot separate treatments, and the window is the piece the survey singles
  out.
* **The opening eval, 195,291 with corridor max 205,440, is a coin flip**
  under CLAUDE.md's own warning and should not be read as "the mini-race
  started strong". It is a measurement of the stuck checkpoint on this
  particular box, taken before any gradient from the window had landed
  (the resumed policy is byte-identical to the checkpoint's).
* **`race/eval_progress` and the honest metric agreed this time**, unlike
  round 18. That is not evidence the standing metric is fine - it is evidence
  that eval_progress is reliable when an arm gets WORSE, and unreliable when
  it gets more consistent.

## Round 18 - xMARGIN: the harvest margin, unconfounded (2026-08-22 03:53-04:56 UTC)

**This is not a paper test. It is the PRECONDITION every reverse-curriculum
paper in docs/research-litsurvey.md section 6 assumes and that every
start-state arm this project has run has silently violated.**

Florensa et al. (1707.05300) defines good starts as those with success in
(0.1, 0.9) and requires the start distribution to sit AT the frontier.
Salimans & Chen (1812.03381) advances a window backwards from the goal.
Go-Explore returns to the frontier by state restore. All three require that
states near the failure point are REACHABLE AS STARTS. On this task they are
not, and three arms have now measured that from three directions:

* Round 15: the 10 s harvest margin hides ~30,000 u of track; the reservoir's
  closest state to the finish was 13,808-29,788 u out.
* Round 18 / xNECTO: the difficulty weighting worked exactly as published
  (9-19x realized oversampling; its failure-rate statistic found the wall bin
  unaided at 0.96) and still moved mean start distance only 85,080 -> 80,340 u,
  because the wall bin is 0.44% of the reservoir. Its own conclusion: "the
  10 s harvest margin, not the sampling rule, is the binding constraint."
* Round 15's one attempt at a shorter margin (`wMARGIN`) is unusable: it
  changed the action space (`--yaw-adaptive`) at the same time.

So an unconfounded `--respawn-margin` reduction from the stuck checkpoint had
never been run. That is this arm, and nothing else changed.

### The arm

Branch `margin`, off `origin/route-obs` fce9050. `--route` was NOT passed
(`test_route.py::test_route_dim_zero_is_the_old_model` proves the policy is
byte-identical with it off), no `--respawn-difficulty`, no
`--shrink-perturb`, no `--yaw-adaptive` change - **one flag**:

    bash tools/run_arm.sh xMARGIN --respawn-margin 2

which `run_arm.sh` expands to the pinned baseline plus that flag:

    python3 -u python/train_fast.py --ckpt runs_ckpt.pt --run xMARGIN \
      --steps 4582737920 --record-every 75e6 --eval-eps 9 \
      --eval-greedy-only --ckpt-every 1e9 --respawn-margin 2

Warm resume of `runs/sOBSR2/ckpt_latest.pt`, md5 `1ba1fd2936af3ae1ad3608e3cd
6b1e9e` verified on the box by `run_arm.sh`, step 3,782,737,920, on
`surf_src_cannonball`. The launcher's baseline-config guard reads the
CHECKPOINT's config (which still says `respawn_margin: 10.0`) and passed; the
trainer restores `respawn_margin` from the checkpoint only when the flag is
absent, so the CLI value is what ran - confirmed in the log by
`respawn: 90% of episodes from mid-run snapshots, harvested >= 2s before
episode end`, and by `respawn_margin` being absent from the "restored from
checkpoint config" line.

Everything else came off the checkpoint unchanged: `respawn_frac 0.9`,
`respawn_binned 0` (so the sampler is plain uniform over the reservoir - the
sampling rule is untouched), `ez_eps 0`, `spawn_burst 0`, `train_stride 1`,
`epochs 4`, `int_coef 0.25`, `gamma 0.9995`, `yaw_adaptive` as the checkpoint
had it.

**Code changed: one diagnostic, no behaviour.** `depth_report()` in
`python/surfgym/respawn.py` (a pure formatter) plus keeping the spawn pool
the trainer had already built so the report can sample it. No RNG is
consumed differently and no tensor is touched, so the control path is
bit-identical. `tests/python/test_respawn_margin.py` adds 14 tests: the
control's 10 s is a pinned constant in BOTH `train_fast.py` and
`run_arm.sh`; the margin-1000 harvest is exactly ticks 100..2000 of a
3,000-tick episode as before; a shorter margin's harvest is a strict superset
of the long one's; the sampler is byte-identical whatever margin the buffer
was built with; the stagnant mask still excludes stall states at a short
margin; a margin below one snapshot interval buys nothing (the rule is
quantized by `snap_every`); and the depth report's bins, mean, pool line and
ASCII-only output. 127 tests locally (113 on `route-obs` before), 126/127 on
the box.
### Why 2 seconds, from the recordings rather than from taste

The margin is a rule about TIME BEFORE AN EPISODE ENDS, so the value has to
come from how long the wall takes to kill. Measured over all 81 recorded
xNECTO episodes (`surfgym.route.episodes_from_traj` + the in-order corridor
projection `eval_honesty.py` uses), the time between an episode reaching its
deepest ON-ROUTE point and the episode ending:

| | all 81 episodes | the 56 that reach >= 200,000 u |
|---|---|---|
| min | 0.02 s | 0.02 s |
| median | 1.22 s | 1.33 s |
| p90 | 1.98 s | 1.50 s |
| max | 5.31 s | 2.44 s |

i.e. the deepest route point is reached ~1.3-1.5 s before the end, and that
gap is the FREE-FALL - the policy leaves the ramp between route vertices 1596
and 1598 and drops 2,400 u in ~1.5 s. Snapshots exist only every
`snap_every` = 100 ticks (1 s), so the deepest snapshot a margin M admits is
the largest multiple of 100 ticks at or before `end - 100*M`. Evaluating that
for every episode gives what each candidate margin could actually put in the
reservoir:

| margin | deepest snapshot, mean arc | deepest snapshot, MAX arc | median off-line at it | median vz | share of episodes able to harvest past 200,000 u |
|---|---|---|---|---|---|
| 10 s (control) | 159,387 u | 180,480 u | 172 u | -1,078 | **0.0%** |
| 5 s | 167,952 u | 197,888 u | 148 u | +1,833 | 0.0% |
| 3 s | 169,681 u | 203,392 u | 268 u | +592 | 13.6% |
| **2 s** | **172,245 u** | **205,312 u** | **258 u** | **-164** | **55.6%** |
| 1.5 s | 173,656 u | 205,440 u | 321 u | -740 | 56.8% |
| 1 s | 174,510 u | 205,440 u | 346 u | -865 | 65.4% |

Two things decide it.

**2 s is the shortest margin that still excludes the free-fall.** At 2 s the
deepest admitted snapshot sits at the wall vertex itself (205,312 u) with a
median off-line error of 258 u and a median vertical velocity of -164 u/s -
that is the state ON the ramp at departure, the moment the ledger's wall
profile calls "a ~0.45 s precursor of small growing error". At 1.5 s and 1 s
the median vz is -740 and -865 u/s and the far-off-line share of those
snapshots triples then quadruples (1.2% -> 2.5% -> 8.6% beyond 1,500 u): those
are states already inside the fall, which is exactly the "near-death states
whose value is genuinely low" failure the margin exists to prevent. The extra
128 u of arc they buy is one route vertex.

**3 s is not a safer version of the same thing - it is a different thing.**
It stops at 203,392 u, 1,920 u short of the wall, i.e. it still cannot harvest
the place every arm dies. And it is not even a smaller dose: at the measured
training episode length (2,600-3,200 ticks) a 3 s margin adds 7 snapshots to
the harvest where 2 s adds 8, so the two differ by about three percentage
points of the reservoir. The dose argument does not separate them; the depth
argument separates them decisively.

### Why NOT a mixed distribution

The brief asked whether keeping most of the reservoir at the old margin and
only some fraction at the short one would be safer. It would not, for three
reasons, and the first is arithmetic:

**A margin reduction is already a mixture.** The harvest at margin 2 is a
strict SUPERSET of the harvest at margin 10 - every state the old rule
admitted is still admitted, plus the tail. That is pinned by
`test_short_margin_harvest_is_a_strict_superset_of_the_long_one`. So the
reservoir is not replaced, it is extended, and the size of the extension is
known in advance: 8 snapshots on top of 16-22, about a third of the harvest at
training episode lengths, and measured at 5.9% of the harvest over the longer
recorded eval episodes. There is no need for a second knob to obtain a
conservative dose; the FIFO already gives one.

**A mixture is a sampling-rule change, and the sampling rule is the one thing
this arm must not touch.** Round 15's `wMARGIN` is unusable precisely because
it moved a second thing. Round 18's xNECTO already showed that a sampling
change on this reservoir measures the margin rather than the sampling; an
arm that moved both would be uninterpretable in both directions.

**Nothing in the measurements pins a mix fraction.** The margin has a
measured value (the fall lasts 1.3-1.5 s); a mixture weight would have been
taste.

### The correctness surface, checked before renting

* `margin_ticks = int(respawn_margin * 100)`, so 2 s = 200 ticks; episodes
  shorter than 300 ticks still contribute nothing, as before.
* **Stall states are guarded by the `stagnant` mask, not by the margin**
  (`train_fast.py` passes `reward_fn.stagnant_mask()` into
  `RespawnBuffer.observe`), and the buffer's own docstring says why: a
  stall-KILL fires 15 s after the stall began, so an end-relative margin
  could never have seen the onset anyway. Shortening the margin therefore
  does NOT start admitting stalled states; pinned by
  `test_stagnant_states_are_still_excluded_at_a_short_margin`.
* Positions are never perturbed on respawn (only speed scale and view
  pitch), so a harvested state cannot be nudged into geometry.
* The docstring's other justification for the margin - "the last also avoids
  farming trivial spawn-next-to-the-finish wins" - is real but inert here:
  win rate has been 0.00% on this checkpoint for ~2e9 steps and stayed
  0.00% for the whole run, so no finishing episode existed to farm. **It
  would matter the moment an arm starts finishing, and that is a caveat any
  successor must carry.**
* The trainer restores `respawn_margin` from the checkpoint only when the
  flag is absent, so the CLI value wins; `run_arm.sh`'s baseline guard reads
  the CHECKPOINT's config (still 10.0) and is unaffected.
### The reservoir evidence - the primary measurement, ahead of any score

`depth_report()` (new, `python/surfgym/respawn.py`) extends the existing
every-100-iterations line with a MEAN, a 16-bin histogram over goal distance
with bin 0 nearest the FINISH, and the realized spawn pool's min/mean plus the
share of draws landing in bins 0-1. It is a pure formatter: no state, no RNG,
and the pool build it reads is the same object the trainer already passed to
`core.set_spawn_pool`, so the control path is bit-identical.

**Where the wall is, in reservoir coordinates.** The bins are geodesic
distance-to-goal (d0 = 198,380 u, 16 bins of 12,399 u). Sampling the cached
field along the reference route puts the wall - route vertices 1596-1604,
where all three previous arms stop - at **geodesic d = 6,610-7,036 u**, which
is inside **bin 0**. So the precondition test has an exact statement: does the
reservoir acquire states in bin 0?

**The margin-10 answer, from four independent measurements: no, never.**

| reservoir, margin 10 s | min d | p10 | bin 0 population |
|---|---|---|---|
| the stuck checkpoint's own stored reservoir (20,000 states, measured on CPU before renting) | 19,338 | 39,217 | **0** (bin 1: 8 states, 0.04%) |
| xROUTE, 11 readings over 1,018 iterations | 13,039-16,510 (one 7,016 outlier) | 39,534-40,121 | - |
| xSP, 6 readings | 13,985-16,867 | 39,648-40,072 | - |
| xNECTO, 6 readings | 12,180-15,693 | 39,196-39,801 | 0 at the final draw |

The tell is p10: across three arms, hundreds of iterations and three different
treatments, the 10th percentile of reservoir depth never leaves ~39,500 u.
The reservoir's shape is set by the harvest rule, and no sampling change
touches it.

**The margin-2 answer, at the first reading after resume (iteration 101,
+79 M steps):**

```
reservoir d: min 5,265  p10 15,171  median 57,787  mean 68,032  (100,000 states)
  depth hist (16 bins x 12,399u, bin 0 = at the finish):
      6433 14377 13003 10341 8970 7979 6933 6726 5860 4712 3783 3344 3170 2427 830 1112
  start pool d: min 5,265  mean 81,166  (4,096 entries, 18.41% in bins 0-1)
```

| | control (margin 10) | xMARGIN (margin 2) |
|---|---|---|
| min d | 19,338 -> plateau 12,180-16,867 | **5,265** |
| p10 | 39,217, pinned ~39,500 | **15,171** |
| mean | 83,327 | 68,032 |
| bin 0 (0-12,399 u; the wall is at 6,610-7,036) | **0** | **6,433** (6.4%) |
| bins 0-1 population | 8 (0.04%) | 20,810 (20.8%) |
| share of realized STARTS in bins 0-1 | 0.07% | **18.41%** |
| fleet mean start distance | 95,709 | 81,166 |

min d = 5,265 u is PAST the wall. **The precondition the reverse-curriculum
literature requires is met for the first time on this task**, and the size of
the change is not marginal: the wall's neighbourhood goes from 0.04% of the
reservoir to 20.8%, and from 0.07% of starts to 18.4%. For scale, xNECTO's
difficulty weighting - the paper-faithful sampling fix - bought 3.6% of starts
and moved mean start distance 85,080 -> 80,340 u. One flag on the harvest rule
did five times as much to the start distribution as the published sampling
rule did.

**On ep_len and ep_rew as risk indicators: they are confounded here.** The
control runs sit at ep_len 2,667-2,816 and ep_rew 25.2-27.8 (measured over
1,018 / 519 / 773 logged iterations of xROUTE / xSP / xNECTO). xMARGIN runs
lower on both. That is a DIRECT consequence of starting 14,500 u closer to the
goal on average - a shorter run remains, and a progress-shaped reward can pay
out less of it - not evidence of near-death poisoning on its own. The honest
indicators are the eval series and `train/value_loss`.

**The series over the run** (`reservoir d:` every 100 iterations; the run
harvested at margin 2 s throughout, the first row is the checkpoint's own
stored reservoir, harvested at 10 s):

| iteration | min d | p10 | median | mean | bin 0 | bins 0-1 as % of starts |
|---|---|---|---|---|---|---|
| 1 (restored, margin 10) | 19,338 | 39,217 | 74,362 | 83,327 | **0** | 0.07% |
| 101 | 5,265 | 15,171 | 57,787 | 68,032 | 6,433 | 18.41% |
| 201 | 5,273 | 14,964 | 56,921 | 67,192 | 6,575 | 19.14% |
| 301 | 4,899 | 15,415 | 58,671 | 69,097 | 6,114 | 18.51% |
| 401 | 4,470 | 15,422 | 59,523 | 69,633 | 6,364 | 19.02% |

It reaches the new composition within 100 iterations and holds it - no runaway
(the FIFO does not fill up with wall states) and no decay back. The p10 moving
from a three-arm-invariant ~39,500 u to ~15,200 u is the single cleanest
statement that the harvest rule, not the sampling rule, sets the reservoir's
shape.

### Both metric series

`race/eval_progress` from `runs/research/xMARGIN/progress.csv`, paired with
`tools/eval_honesty.py` over every recorded greedy episode. 3090 working band
~140k-195k, opening eval a coin flip, comparison arms xROUTE / xSP / xNECTO,
all 3090, all warm-resumed from the same checkpoint.

| eval | steps after resume | race/eval_progress | corridor mean | corridor MAX | past 205,440 | finishes | dives |
|---|---|---|---|---|---|---|---|
| 1 | +1M | 130,271 | 137,216 | **205,312** | **0/9** | 0/9 | 5/9 |
| 2 | +76M | 177,733 | 188,245 | **205,824** | **1/9** | 0/9 | 7/9 |
| 3 | +152M | 174,130 | 186,780 | **208,640** | **3/9** | 0/9 | 5/9 |
| 4 | +227M | 175,271 | 185,899 | **208,384** | **2/9** | 0/9 | 8/9 |
| 5 | +303M | 173,888 | 183,310 | **205,312** | **0/9** | 0/9 | 8/9 |
| 6 | +378M | 156,730 | 165,931 | **205,312** | **0/9** | 0/9 | 5/9 |
| 7 | +454M | 188,261 | 198,329 | **205,312** | **0/9** | 0/9 | 8/9 |
| 8 | +529M | 157,085 | 167,068 | **205,312** | **0/9** | 0/9 | 6/9 |

Run stopped at **+534.8M steps after 61 minutes** (CLAUDE.md rule 2), 143k
steps/s average including an 81 s compile, GPU pinned at 99%.

**`race/eval_progress`: 130k-188k, mean 172k, no trend and no decay - a null
on the standing metric**, squarely inside the 3090 band and never near the
195k positive threshold. Compare xROUTE, which posted three evals at ~195,2xx
on this checkpoint and was a complete null. See "the finding this arm fell
over" below for why that metric could not have said anything else.

**The reference frontier is 205,312-205,440 u of 231,680 u, and it had never
moved.** Re-scored here with the same tool rather than quoted: **xROUTE 0 of
99 episodes past 205,440 over 11 evals; xSP 0 of 54 over 6; xNECTO 0 of 81
over 9. 234 greedy episodes, three mechanisms, every eval's maximum exactly
205,312 or 205,440, zero finishes.**

xMARGIN: **6 of 72 episodes past it, maximum 208,640 u = 90.06%**, still 0
finishes. Eval 1 - the untreated policy at +1M steps, before the new margin
has changed the reservoir - lands on exactly 205,312 with 0/9, which is the
right internal control. The crossings then appear in evals 2-4 and stop.

**Robustness: the gain survives a much tighter corridor.** Corridor progress
admits samples within 1,500 u of the line, which past the descent is loose
enough to worry about - the fall goes roughly the way the route does.
Re-scored at `--corridor 600`:

| | corridor 1,500 u | corridor 600 u |
|---|---|---|
| xMARGIN eval 1 (untreated) | 205,312 | 204,928 |
| xMARGIN eval 3 (+152M) | 208,640 | **207,104** |
| xMARGIN eval 4 (+227M) | 208,384 | **206,848** |
| xROUTE final eval (+755M) | 205,312 | 204,928 |
| xNECTO eval 4 (+227M) | 205,440 | 204,928 |

At a corridor 2.5x tighter both controls AND this arm's own untreated opening
fall back to 204,928 u, and evals 3-4 still read 206,848-207,104 u:
**+1,920 to +2,176 u, 15-17 route vertices, past the control's tightened
maximum.** The gain is not corridor slack.

**Training diagnostics** over 675 logged iterations: `ep_len` mean 2,398
(p10-p90 2,179-2,623) against controls' 2,667-2,816; `ep_rew` mean 22.9
(21.0-25.0) against 25.2-27.8; `value_loss` mean 0.0614 (0.0412-0.0846)
against control means 0.0464 / 0.0509 / 0.0542; `kl` 0.0197, `ent` pinned at
0.005, win 0.00% throughout, reservoir full from iteration ~25.

The first two are CONFOUNDED by the treatment and must not be read as health:
starting 14,500 u closer to the goal necessarily shortens episodes and leaves
a progress-shaped reward less to pay out. `value_loss` is the less confounded
near-death indicator, and it rose about 20% while staying inside the controls'
own p90 band (0.0646-0.0750). **Near-death starts cost the critic something
measurable; they did not destabilize it.**

### The wall profile - what actually changed, and it is not what was predicted

`tools/wall_profile.py`, same champion line, same vertices, xMARGIN against
the two margin-10 arms that recorded the same stretch. Off-line error in
units, agent vs the champion's own line:

| route vertex | champion | xROUTE (margin 10) | xNECTO (margin 10) | **xMARGIN (margin 2)** |
|---|---|---|---|---|
| 1598 (the departure) | 79 | **3,019** | **1,735** | **177** |
| 1602 | 147 | 2,314 | 2,331 | **232** |
| 1606 | 200 | - | 2,226 | **349** |
| 1610 | 188 | - | - | 548 |
| 1614 | 171 | - | - | 882 |
| 1618 | 145 | - | - | 1,183 |
| 1626 | 95 | - | - | 1,144 |
| 1630 | 87 | - | - | 1,523 |

**The ramp departure is no longer where it fails.** At vertex 1598 - the
256 u stretch CLAUDE.md names as the failure, where three arms blew out by
1,700-3,000 units - this run is 177 u off the line with dz of -17 against the
champion. It rides the exit cleanly and then loses the line 15-20 vertices
further down, at 1610-1618.

**And the speed hypothesis does not survive it.** The ledger's reading of the
wall was "enters 6% slower than the champion (2,820 vs 2,930)". xMARGIN enters
at 2,740 against 2,934, i.e. **6.6% slower - no better than the controls** -
and survives the departure anyway. What kills it now is downstream: speed
DECAYS through the descent (2,740 -> 2,615 at 1606 -> 2,563 at 1614) where the
champion holds 2,927 flat and then accelerates. So entry speed was not the
binding constraint at the departure; departure PRECISION was, and it is
exactly what respawning into that spot 18% of the time teaches. The next
constraint is carrying speed down the descent, which is a different problem in
a different place.

### The finding this arm fell over on the way past the wall

Chasing why `race/eval_progress` did NOT rise while the corridor frontier
did, I sampled the geodesic field along the reference route. **The shaping
potential's minimum along the route is at vertex 1601 - the wall.**

| route vertex | arc | geodesic d |
|---|---|---|
| 1596 | 204,288 | 7,036 |
| **1601** | **204,928** | **6,568 <- the field's minimum along the route** |
| 1604 (reference frontier) | 205,312 | 6,610 |
| 1610 | 205,952 | 6,941 |
| 1620 | 207,360 | 8,108 |
| 1640 | 209,920 | 10,656 |
| 1660 | 212,480 | 13,321 |
| **1680** | **215,040** | **14,976 <- +8,408 u above the minimum** |
| 1700 | 217,600 | 13,954 |
| 1800 | 230,400 | 1,200 |

Two consequences, and they explain a great deal of this ledger.

**1. `race/eval_progress` is structurally blind past vertex 1601.** It is
`mean over episodes of (d at spawn - min d reached)` (`race_progress()` in
train_fast.py). An episode that reaches vertex 1601 has already taken its
minimum; riding on to 1630 changes nothing. The per-episode ceiling for a
route-following episode is **191,812 u**, and every unit of the frontier this
arm gained is worth exactly ZERO on the project's headline metric. Values
above 191,812 in this ledger (xROUTE's 195,2xx) can only have come from
off-route dives, which is the same artifact `eval_honesty.py` was written for -
now with a number attached to it. So xROUTE's eval_progress was flatteringly
WRONG and xMARGIN's is unflatteringly wrong, for the same underlying reason.

**2. The final descent is a potential BARRIER in the reward, not a slope.**
`RaceReward` is `r_t = scale * (d_{t-1} - d_t) - time_pen` with
`scale = 100/d0 = 5.041e-4` and `time_pen = 0.005`/tick. Riding the true
champion line from vertex 1601 to 1680 raises d by 8,408 u, i.e. it is charged
**-4.24 reward**, and the ~9.2 s it takes to run the remaining 26,752 u of
route costs a further **-4.61** in time penalty. Committing to the descent
therefore costs about **-8.9** against a typical training-episode return of
22-25, and the far side pays 3.31 of shaping plus the +50 success bonus - a
payoff **this policy has never once observed**, at 0.00% win rate for ~2e9
steps. Turning back at the minimum is locally optimal, and the shaping field
says so.

This is a hypothesis about the reward, arithmetic not experiment, but it fits
everything the ledger has recorded: three independent mechanisms, 234 greedy
episodes, all stopping within a few vertices of 1601, which is the exact
minimum of the potential they are being paid to descend. **It reframes the
barrier as a reward-model defect rather than an exploration failure**, and it
predicts the cheapest possible test: the field already has a
gravity-directional variant (`build_goal_field(..., gravity_dir=True)`, cache
tag `goalg_`), written because "the undirected BFS lets voxels reach the
finish through one-way falls, which on surf_src_cannonball paints an off-route
pit as the global minimum of the shaping potential" - the same defect, found
before, in a different place. Whether the gravity-directional field is
monotone along the descent is a one-command check, and if it is, an arm that
shapes on it changes the sign of the reward exactly where every run of this
project has stopped.

### VERDICT

**The precondition was real, one flag fixes it, and the barrier moved for the
first time in this project - but intermittently, and it still does not
finish.** Three claims, in descending order of how much weight they carry:

**1. CERTAIN: the precondition is met, and it was the cap.** One flag took
the wall's neighbourhood from 0.04% of the reservoir to ~20%, and from 0.07%
of realized starts to ~19%. Reservoir minimum depth went from a plateau of
12,180-16,867 u across three arms and 24 readings to 4,470-5,273 u, i.e. past
the wall, and stayed there for the whole run. For scale, xNECTO's
paper-faithful difficulty weighting - running at its published dose, with its
statistic correctly finding the wall bin - bought 3.6% of starts. **One flag
on the harvest rule did five times as much to the start distribution as the
published sampling rule did, which is the quantitative form of xNECTO's own
conclusion.**

**2. STRONG BUT INTERMITTENT: the frontier moved.** 205,312-205,440 u had
been invariant across **234 greedy episodes and three mechanisms** - verified
here with the same tool, not just quoted: xROUTE 0/99 past 205,440, xSP 0/54,
xNECTO 0/81, every eval's maximum exactly 205,312 or 205,440. This arm's own
opening eval, the untreated policy, sat at exactly 205,312 with 0/9 past.
Then 6 of this run's 72 episodes passed it, reaching 208,640 u
(90.06%), concentrated in evals 2-4. The gain survives a 2.5x tighter
corridor (600 u: this arm 207,104 u against both controls' 204,928 u), and
the wall profile localizes it: at vertex 1598, where the controls are
1,735-3,019 u off the champion line, the good evals are 177-254 u off, with
the failure displaced 15-20 vertices downstream. **But evals 5 through 8 all fell
back to exactly 205,312 with 0/9 past**, so this is an intermittent capability
rather than a new stable frontier - it is one seed and one hour, and that is
what one hour bought.

**3. IT STILL DOES NOT FINISH.** 0 finishes in every episode of the run. The
new failure mode is a different one from the old: the policy now survives the
ramp exit and then bleeds speed down the descent (2,740 -> 2,563 u/s where
the champion holds 2,927 and accelerates).

**And the project's headline metric registers none of this.**
`race/eval_progress` sat at 130k-178k throughout, i.e. mid-band, reading as a
null-to-mild - while the frontier was moving. That is not noise, it is
structural: eval_progress is `mean(d at spawn - min d reached)` and the
geodesic field's minimum ALONG THE ROUTE is at vertex 1601, the wall itself,
so a route-following episode saturates at 191,812 u and every unit this arm
gained past the wall is worth exactly zero. xROUTE's eval_progress was
flattering and false; xMARGIN's is unflattering and false. **The two of them
together retire `race/eval_progress` as a frontier metric on this map.**

### Caveats a later reader needs

* **This is a precondition test, not a paper test.** No paper in
  docs/research-litsurvey.md proposes "shorten the harvest margin"; the
  margin is RL_Surf's own device. What the literature supplies is the
  REQUIREMENT the margin was violating - Florensa (1707.05300) needs the
  start distribution at the frontier, Salimans & Chen (1812.03381) needs a
  window that can reach back from the goal, Go-Explore needs to return to
  the frontier by state restore. A result here says the family's precondition
  was the binding constraint on this task; it does not validate or falsify
  any of those three methods, all of which remain untested WITH the
  precondition met. That is now the cheapest arm on the board: rerun
  `--respawn-difficulty` / `--respawn-mode florensa` / `--respawn-mode
  backward` on top of `--respawn-margin 2`, where for the first time they
  have states to work with.
* **One seed, one hour** (CLAUDE.md rule 2), warm-resumed. The eval band's
  own spread within a single run is wide, which is why the verdict leans on
  the corridor frontier and the wall profile rather than on
  `race/eval_progress`.
* **The corridor metric needs care past the old wall.** Corridor progress
  admits any sample within 1,500 u of the line, advancing in order. Beyond
  vertex ~1620 this run's off-line error passes 1,100 u, so the last few
  hundred units of the reported corridor maximum are the drift beginning
  rather than clean riding. The defensible statement is the wall profile's:
  on-line riding now reaches vertex 1610-1618 (206,080-207,104 u) where it
  used to end at 1598 (204,544 u); the corridor number is the generous
  reading of the same thing.
* **`ep_len` and `ep_rew` are confounded by the treatment** and cannot be
  read as health here: starting 14,500 u closer to the goal necessarily
  shortens episodes and lowers a progress-shaped return. `train/value_loss`
  is the less confounded risk indicator and it rose (xMARGIN mean 0.0656 vs
  control means 0.0464 / 0.0509 / 0.0542) but stayed inside the controls' own
  p90 (0.0646-0.0750). Near-death starts cost the critic something; they did
  not destabilize it.
* **The margin's other job is dormant, not gone.** `respawn.py` justifies the
  margin partly as "the last also avoids farming trivial
  spawn-next-to-the-finish wins". Win rate was 0.00% for the whole run, so
  there were no finishes to farm. The moment an arm starts finishing, a 2 s
  margin WILL harvest states 2 s from the goal and the reservoir can
  self-reinforce on trivial wins. Any successor that starts finishing must
  revisit this - most likely by making the margin asymmetric (short before a
  death, long before a FINISH), which the current single scalar cannot
  express.
* **Anti-forgetting was not measured directly.** 18-19% of starts now sit in
  the last 12.5% of the map, so the early map gets proportionally less
  practice. Florensa reserves 1/3 of starts for mastered regions for exactly
  this reason. The eval episodes all start from the true map start and did
  not degrade over the run, which is evidence against forgetting at this
  horizon, but an hour is short.
* The 3090's `test_march_is_bit_exact_against_the_legacy_kernel` failure
  means greedy trajectories fork against 5090 runs; every comparison here is
  3090-to-3090 (xROUTE, xSP, xNECTO), per CLAUDE.md.

### Ops

Five instances created, four rejected, one ran the arm.

* **48359762** (machine 43803, host 305845) - `create` returned "Error
  response from daemon: failed to create task" and it never left `created`.
  Blocklisted `unreliable`, **by hand**: it was destroyed before `--block`
  ran, and `--block` takes an INSTANCE id and resolves the machine/host from
  it, so the identifiers were already gone. That is the exact failure
  `bad_hosts.json`'s own README warns about, and it cost the entry.
* **48359755** (machine 51342) - race loser, still `loading` when another
  candidate was serving ssh; destroyed at ~60 s, not blocklisted.
* **48359768** (machine 128224, host 551866, France) - passed the 60 s
  readiness rule, deployed clean, and `gpu_health.py` printed **VERDICT:
  healthy** anyway. It was not: `power.limit 250.00 W` against
  `power.default_limit 350.00 W`, sm 1080 MHz under load, 61 TFLOPS bf16 and
  839 GB/s HBM. `nvidia-smi -pl 350` -> "Insufficient Permissions". That is
  the `gpu_capped` class in `bad_hosts.json` verbatim. Blocklisted, destroyed.
  **`gpu_health.py` says "no reference for this model - recorded, not judged"
  for a 3090 and therefore cannot catch this; the cheap explicit check is
  `power.limit` vs `power.default_limit`.** The box that did run the arm
  reported 350/350 W, 1755 MHz and 73 TFLOPS - a 20% GEMM difference between
  two "healthy" 3090s.
* **48360082** (machine 34330, host 3497, Czechia) - the machine xSP ran on,
  and in `known_good`. Rejected on **upload throughput**: a single 16 MB scp
  took 105 s = **0.16 MB/s**, where the France box had measured 2.6 MB/s from
  the same workstation minutes earlier. The deploy pushes ~230 MB of
  checkpoint plus baked caches, so that is 24 minutes of a 60 minute budget.
  Blocklisted `network` with a note that the CARD is fine and the machine is
  also in `known_good`, so a later reader re-tests rather than trusts the
  entry. **Measuring upload with one 16 MB probe before deploying is worth
  making standard - it is 6 s on a good box and it caught this before the
  147 MB checkpoint went out.**
* **48360269** (machine 141130, host 440416) - ssh still answering
  "Permission denied (publickey)" at ~110 s from create. Blocklisted
  `network`, destroyed.
* **48360266** (machine 45199, host ..., Estonia, $0.162/h, 64-core EPYC
  7532) - **the winner.** 350/350 W, 1755 MHz, 73 TFLOPS bf16, 863 GB/s.
  Upload measured 0.69 MB/s, which is below the 1 MB/s bar; accepted
  deliberately because 230 MB at that rate is ~5.5 minutes, the two boxes
  ahead of it in the race had already failed on worse defects, and the arm's
  budget is denominated in STEPS. Recorded here so the trade is visible.

126 of 127 tests pass on the box; the single failure is
`test_march_is_bit_exact_against_the_legacy_kernel`, which CLAUDE.md already
records as failing on a 3090 and passing on a 5090.

### Artifacts and cost

`runs/research/xMARGIN/`: progress.csv, run.json, all 8 eval trajectories, and
`xMARGIN_launch.txt` with all 677 iteration lines and all 7 reservoir-depth
blocks.

Rental: the winner 48360266 ran 03:48-04:56 UTC (68 min including deploy and
the test suite) at $0.182/h = **$0.21**; the four rejected boxes lived 1-5
minutes each and cost about **$0.04** between them. **Total ~$0.25.**

## Round 18: independent verification of the reward barrier (main session)

xMARGIN's claim that the final descent is a potential barrier was checked
directly against the cached field, sampling `goal_32.npz` at the champion
route's own vertices:

| route vertex | units | geodesic d |
|---|---|---|
| 1590 | 203,520 | 7,705 |
| **1600** | **204,800** | **6,632** <- local minimum ALONG THE ROUTE |
| 1610 | 206,080 | 6,941 |
| 1620 | 207,360 | 8,108 |
| 1640 | 209,920 | 10,656 |
| 1660 | 212,480 | 13,321 |
| 1680 | 215,040 | 14,976 |
| 1810 | 231,680 | 5 (the finish) |

**Confirmed and worse than described: d rises by 8,344 u over 79 vertices of
the champion's own winning line before it falls to the finish.** The shaping
therefore charges roughly -4.2 (progress term at scale 100/d0) plus ~-4.6
(time penalty) to traverse the correct route past vertex 1600, against a +50
bonus this policy has never once observed. **Turning back at vertex 1600 is
locally optimal**, and vertices 1596-1604 are exactly where all 234 greedy
episodes of xROUTE, xSP and xNECTO stopped.

This closes the round's central question. The barrier is not exploration,
not plasticity, not the observation, and not the start distribution: it is
that the reward's local optimum sits on the wall. The three nulls were all
measuring a policy correctly maximizing a reward that says stop there, and
xMARGIN moved the frontier because respawning past vertex 1600 puts an agent
on the far side of the barrier where the gradient points at the finish again.

The fix has existed, unbaked, since before this round:
`build_goal_field(gravity_dir=True)` rebuilds the same field over a
gravity-directional graph (fall and air-strafe freely, climb only where
geometry supports it), which is precisely the defect that makes a one-way
descent read as expensive. Baking it and testing it is the next arm.

## Round 19 - xGRAV: the gravity-directional field, GATED OUT before renting (2026-08-22 07:07-08:05 UTC)

**The arm did not run, and that is the result.** The previous section closed
round 18 by naming `build_goal_field(gravity_dir=True)` as the fix for the
shaping barrier at route vertex 1600 and "baking it and testing it" as the
next arm. It is now baked and tested. **It does not move the barrier by one
unit at vertex 1600, and past it the barrier gets 14% WORSE.** The gate that
was supposed to stop a bad arm before it cost a box stopped it, for **$0.00
of rental**.

### What was added (branch `gravity`, commit 3f765c8, off `origin/route-obs`)

`--race-gravity {0,1}` in `train_fast.py`: default None, restored from the
checkpoint config when not passed (and named in the "restored from
checkpoint" line), defaulted off, written into the saved config, threaded
into all three `build_goal_field` call sites (eval field, kill-aware shaping
field, respawn binning field). With it ON, BOTH the shaping field and the
eval-progress field are the directional one - deliberately, because the plain
field's own minimum ALONG THE ROUTE is the wall, so `race/eval_progress`
computed on it saturates at 191,812 u and cannot see the thing the arm exists
to move. The price is that eval_progress would not have been comparable with
the controls', which is why the honest metric was to lead.

With it OFF the call is `gravity_dir=False`, which is the same cache file,
the same signature and the same field the control loads;
`test_flag_off_reuses_the_control_cache_and_never_rebakes` pins that by
making a rebake an assertion failure. `tools/record_ckpt.py` MIRRORS the flag
rather than listing it TRAIN_ONLY: under `--obs-reward` the policy is fed
that potential's own delta in scalar slot 12, so recording a gravity
checkpoint against the plain field would hand the agent a different reward
than it trained on. 13 tests in `tests/python/test_race_gravity.py`, 123
green locally (110 before).

### The bake (local RTX 5090, no rental)

    goal graph: gravity-directional (rule v1, support drop 2 cells),
                594,941,751 / 671,156,372 voxels surface-adjacent
    goal field: 5376 sweeps, 37674 seed voxels, 67,887,227 reachable voxels,
                max geodesic 199083u
    BAKE OK in 1930s

32 minutes, 21.4 GB of VRAM at 99% utilization, converged well inside the
8,000-sweep cap. Output `maps/surf_src_cannonball.goalg_32.npz`, 39.6 MB.
The plain field it is compared against is `maps/surf_src_cannonball.goal_32.
npz` (reach_max 199,077 u); the directional bake's reach_max is 199,083 u.

**First number that should have been a warning: both fields have exactly the
same reachable voxel count, 67,887,224 of 671,156,372 (10.11%). The
directional graph disconnected NOTHING.** 87.9% of the grid is solid and
594.9M of 671.2M voxels are surface-adjacent, so on this map the climb rule
is very nearly vacuous.

### The gate, pre-registered before the bake finished

PASS required all three: (a) every map start entity still reaches the finish;
(b) every route vertex still honest (finite d); (c) no material rise along
the champion route past vertex 1600 - taken as a residual barrier <= 1,000 u,
i.e. >= 88% of the measured 8,344 u removed.

(a) and (b) pass. (c) fails outright.

### The field along the champion route, before and after

`maps/surf_src_cannonball.route.npz`, 1,811 vertices at 128 u, sampled with
`GoalField.sample` on both caches:

| vertex | old d | new d | delta |
|---|---|---|---|
| 1590 | 7,705 | 7,705 | 0 |
| 1596 | 7,036 | 7,036 | 0 |
| 1600 | 6,632 | 6,632 | **0** |
| 1601 | 6,568 | 6,568 | 0 |
| 1610 | 6,941 | 7,001 | +60 |
| 1620 | 8,108 | 8,452 | +344 |
| 1640 | 10,656 | 11,289 | +633 |
| 1660 | 13,321 | 13,841 | +520 |
| 1680 | 14,976 | 16,047 | +1,071 |
| 1700 | 13,954 | 14,521 | +567 |
| 1750 | 7,366 | 7,455 | +89 |
| 1810 | 5 | 5 | 0 |

* rise above the route's local minimum: **old +8,344 u (at v1680) -> new
  +9,555 u (at v1684)**;
* total upward motion v1600..v1810: 8,408 u -> 9,619 u;
* decreasing steps along the whole route: 95.6% -> 95.4%;
* the shaping charge for riding the correct line past v1600, at
  `scale = 100/d0`: **-4.21 -> -4.82 reward** (d0 198,380 -> 198,386);
* the new field is >= the old at every one of the 1,811 vertices (1,124
  higher, 0 lower), largest difference 1,259 u at v1684;
* starts 4/4 reachable and route 1,811/1,811 honest under both.

**The barrier is not reduced, it is deepened.** Every unit the directional
graph added, it added on the far side of the wall, where the champion
already pays.

### tools/validate_gravity_field.py, both fields, same trajectories

`--champ runs/sISV_par2/traj_8454144000.jsonl --fail
"runs/research/xROUTE/traj_*.jsonl"`:

    base champion route (7 ep, 580 steps): dec 96.6% honest 100.0% rise 53,301u
    dir  champion route (7 ep, 580 steps): dec 96.6% honest 100.0% rise 58,302u
    base failure routes (101 ep): median 98.4% worst 98.7% deepest min 2,596u
    dir  failure routes (101 ep): median 98.4% worst 98.5% deepest min 2,951u

    directional field cuts the potential a failure banks by 0.0% of the route
    (median), 0.4% at best; 0 of 101 failure episodes lose more than 5 points,
    0 get MORE.

A failing episode still banks **98.4% of the whole potential** under the
directional field, exactly as under the plain one. The tool's own headline
number for the fix is **0.0%**.

### Why it does nothing here: the graph does not climb, it FLIES

Greedy descent traced on the voxel grid from route vertex 1600, identical on
both fields (and on the kill-masked `goalk_32.npz` as well):

    197 steps, d 6,636 -> 32
    from world [-7,248, 1,264, -1,712] to [-13,520, 7,408, -1,872]
    steps: down 5, level 191, UP 0, total climb 0u, net dz -160u

The "shortcut" the shaping believes in is a **straight, level, ~8,700 u glide
through open air at constant z ~ -1,872**. Probing the corridor: 4 of 200
sampled voxels are solid (both near the endpoints) and the median floor
clearance under the line is **3,584 u**. It is a flight across a chasm.

`gravity_dir` gates only the nine `dz > 0` offsets, by explicit design -
"descending and lateral offsets stay unconstrained, the player can always
fall, and air-strafe while falling". A level traverse contains no upward
edge, so the directional rule cannot see it, cannot cut it, and did not.

The same failure in one table - what the field claims is left, against what
the track actually has left:

| vertex | route remaining | field d | ratio |
|---|---|---|---|
| 0 | 231,680 | 198,353 | 0.86 |
| 800 | 129,280 | 103,433 | 0.80 |
| 1400 | 52,480 | 30,943 | 0.59 |
| 1600 | 26,880 | **6,632** | **0.25** |
| 1680 | 16,640 | 14,976 | 0.90 |
| 1700 | 14,080 | 13,954 | 0.99 |

At vertex 1600 the field believes a quarter of the remaining track will do.
It is not routing back up a shaft; it is taking the straight line across the
hole the champion has to go around, and it recovers to honest values only at
v1680, once the champion has come around the far side.

### Correction to goalfield.py's stated premise

The module docstring justifies the directional mode with "measured on
surf_src_cannonball: the reachable minimum along policy rollouts is a d~21.5k
basin off-route, while the winning line reads 31k -> 107k -> unreachable".
**Against the current caches and round 18's recordings that does not
reproduce**: the winning line reads 198,391 -> 0 with 96.6% of steps
decreasing and 100% of readings honest, and failing rollouts bottom out at
2,596-3,100 u (98.4% of the potential banked), by diving into goal-adjacent
space around z = -4,150, not into a 21.5k basin. Whatever field that
paragraph was measured on, it is not the one in `maps/` today. A note to that
effect is now in the docstring; the arm that relies on it next should
re-measure rather than trust either number.

### VERDICT

**GATE FAILED; the arm was not run; no GPU was rented.** The
gravity-directional graph is a correct fix for a defect this map does not
have at the place that matters. The barrier at route vertex 1600 is caused by
**lateral free flight in open air**, which the directional graph permits by
construction, not by one-way climbs, which it forbids and which turn out to
be irrelevant here (0 voxels disconnected).

Against xMARGIN there is nothing to report: **no arm ran, so xMARGIN's 6 of
72 episodes past 205,440 u and its corridor MAX of 208,640 u remain the
frontier, untouched.** The barrier finding from the previous section stands
exactly as written - only its proposed fix is now dead.

`--race-gravity` stays in the trainer, off by default and bit-identical when
off. If the climb rule is ever tightened, `_GRAVITY_RULE_VERSION` invalidates
only the `goalg_`/`goalgk_` caches, so re-testing costs one 32-minute bake
and nothing else.

### What the evidence says the next arm is

Not another edge rule on this graph. A per-edge "glide cone" cannot express
this: at 3,700 u/s a single 32 u lateral step drops 0.03 u to gravity, so any
cone tight enough to kill an 8,700 u glide also kills legitimate one-cell
strafes. Making the voxel geodesic honest about flight needs the VELOCITY
dimension (a graph over position x speed), which is a different and much
larger object.

The cheap alternative is already on disk and already trusted for scoring:
**shape on ROUTE ARC LENGTH instead of the voxel geodesic** -
`maps/surf_src_cannonball.route.npz` with the in-corridor projection
`tools/eval_honesty.py` uses. It is monotone along the champion line by
construction, it is the metric this project already believes, and round 18's
xROUTE only ever fed route geometry into the OBSERVATION - nobody has
replaced the potential with it. Caveats to design against before running it:
the projection is ill-defined off the corridor (needs an explicit
out-of-corridor potential, probably distance-to-corridor plus the arc length
at the projection point), and a route potential is by definition
route-following, so it cannot discover a better line than the champion's.

### Ops and cost

* Fleet at start and at end: `fleet_watchdog.py list` -> live (0). No
  instance was created, so nothing needed destroying and no watchdog was
  registered.
* The bake ran on the LOCAL RTX 5090 (2.8 GB used, 8% util before it
  started; no trainer on the box), 32 minutes, and never touched a rented
  card. Everything after it - the profile, the validator, the greedy trace,
  the occupancy probe - is CPU-only and can be rerun with the box gone.
* **Rental cost: $0.00.** The counterfactual it avoided is the ~$0.20 of a
  3090 hour, plus the hour itself, plus an arm whose "treatment" the field
  profile now says would have been a slightly worse control.
* `maps/surf_src_cannonball.goalg_32.npz` (39.6 MB) is gitignored like every
  other bake and is NOT committed; it sits at
  `C:\RL_Surf\maps\surf_src_cannonball.goalg_32.npz` on the workstation.
  `deploy_box.sh`'s cache glob (`$MAP.*_*.npz`) already picks it up, so any
  future box gets it without an edit.
