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

## Round 22 - how much parallelism does ONE policy pay for? (2026-08-23 21:46-23:16 UTC)

**Question (user):** what hardware gives the most training throughput per dollar
for ONE policy, and where does parallelism stop paying? For the ~161-map
multi-map run, which needs one policy over a large batch. Hardware
characterisation, not an ablation: 5-10 minute rungs, `ddp` branch code
(this branch is `ddp` + the ops-tool fixes it predated), `tools/bench_scale.sh`
(new - run_arm.sh's SCRATCH argument set verbatim, varying only `--envs`,
`--n-steps` and the record cadence), cannonball, from scratch, 64x32 depth,
no `--obs-reward`, `--timing`, median of the LAST HALF of each rung's
iterations. Throughput = `envs * n_steps * act_every / iter_seconds`; under
DDP `--envs` is the GLOBAL fleet, so that formula already gives the aggregate.
Four boxes, ~$6.4 of GPU time, all destroyed, registry clean at 23:16 UTC.

### The answer, in three lines

**Buy 4x3090 boxes and run one rank per GPU at 32,768 envs/rank, T=32.**
Best measured **$0.151 per 1e9 steps** - a 1.44x improvement on round 21's
$0.217 champion, entirely from a cheaper box and a higher rung. **A single
3090 is still cheaper per step ($0.129) but is capped at 0.36M steps/s and
CANNOT be pushed past it**, so it cannot serve the multi-map run's
one-policy-large-batch requirement at any envs setting. **DDP scaling has NOT
turned over by 8 ranks** (81.3% at 8 ranks on 5090s, 84.0% at 4 ranks on
3090s, 90.3% at 4 ranks on NVLink A100s), and what limits it is **CPU
contention in the physics step, not the all-reduce**.

### The boxes (all health-gated with `tools/gpu_health.py --all`)

| tag | GPUs | CPU | cores eff | thr/GPU | host RAM | gpu_frac | $/h |
|---|---|---|---|---|---|---|---|
| **A** | 4x RTX 3090 24G | EPYC 7502 32c/64t | 42.7 | 10.7 | 251 G | 0.67 | 0.7067 |
| **B** | 1x RTX 3090 24G | EPYC 7282 (quarter) | 8.0 | 8.0 | 126 G | 0.25 | 0.1683 |
| **C** | 8x RTX 5090 32G | 2x EPYC 7B12, 256t | 256 | 32.0 | 472 G | 1.00 | 3.2010 |
| **D** | 4x A100 SXM4 80G | EPYC 7713 | 61.4 (quota) | 15.4 | 2003 G | 0.50 | 4.9080 |

A: 840 GB/s, 69-71 TFLOPS bf16, 290-316 W of 350. B: 842 GB/s, 71 TFLOPS,
315 W of 330. C: 1,517-1,524 GB/s, 230-238 TFLOPS. D: 1,762-1,767 GB/s,
239-270 TFLOPS at the 400 W cap; **NVLink NV12 all-to-all**, the only box in
this or the previous round with working P2P.

---

## Part 1 - the parallelism ceiling on 4x3090

### The envs ladder, 4 ranks, T=32, `--minibatches 16` (box A)

| envs/rank | GLOBAL envs | **aggregate steps/s** | per rank | iter ms | VRAM/GPU | RSS/rank | update us/sample |
|---|---|---|---|---|---|---|---|
| 8,192 | 32,768 | 1,109,409 | 277,352 | 2,835.5 | 5.06 G | 5.32 G | 7.54 |
| 16,384 | 65,536 | 1,208,896 | 302,224 | 5,204.3 | 8.29 G | 5.64 G | 6.61 |
| **32,768** | **131,072** | **1,301,629** | **325,407** | 9,667.0 | 14.27 G | 6.05 G | **5.93** |
| 65,536 | 262,144 | 1,220,259 | 305,065 | 20,623.3 | ~21.3 G | - | 6.49 |
| 131,072 | 524,288 | **DIES** | - | - | - | - | - |

**The ceiling is 32,768 envs per rank at T=32, and it is NOT the VRAM wall.**
Throughput peaks there, falls 6.3% at 65,536, and the run dies at 131,072.

**Which phase stops scaling first: the UPDATE.** Per-sample update cost falls
7.54 -> 6.61 -> 5.93 us and then RISES to 6.49. The other two candidates do
not bend:

* `env` (the OpenMP C physics step) scales linearly - 329 -> 749 -> 1,527 ->
  2,467 ms across an 8x range of envs, i.e. per-env cost flat to 6%. It stays
  11-16% of the iteration throughout.
* **`allreduce` is CONSTANT at 426-435 ms at every rung**, because it depends
  only on the gradient size. Its share therefore FALLS 15.3% -> 8.2% -> 4.4%
  -> 2.1% as envs grow. **Bigger ranks make comm cheaper, not dearer.**

**And the update's turnover is a MINIBATCH-SIZE effect, not an envs effect.**
`--minibatches` is a COUNT, so holding it at 16 doubles the minibatch every
time envs double: at 65,536 envs/rank it is 131,072 samples. Rerun with
`--minibatches 32`, putting the minibatch back to the 32,768-env value:

| 4 x 65,536, T=32 | steps/s | update us/sample | allreduce ms |
|---|---|---|---|
| `--minibatches 16` | 1,220,259 | 6.49 | 426.6 |
| `--minibatches 32` | **1,320,715** | **5.92** | 877.4 |

The regression disappears completely and it becomes the best 3090 number
measured. **It is not free**: the all-reduce count doubles with the optimizer
step count (426 -> 877 ms), and `--minibatches` is part of the learning
config CLAUDE.md says must not be "corrected" (T=32 with 16 minibatches is a
4x update density that is *part of what works*). Report it as the mechanism,
not as a recommended setting.

### Where it actually dies, and the plan's VRAM table is 2x pessimistic

**`b_img` is bfloat16, not float32.** The S3 split-obs change
(`train_fast.py:1883-1884`) stores the depth half of the rollout buffer at
the precision autocast already rounds it to, so the buffer is
`(PRO+T) * N * FRAME * 2` bytes. Every VRAM prediction in
`docs/multimap-ddp-plan.md` is therefore twice the truth. Measured, per rank,
T=32: **5.06 / 8.29 / 14.27 / ~21.3 GB** at 8,192 / 16,384 / 32,768 / 65,536
envs.

At **131,072 envs/rank, T=32** the process reaches ~20-21 GB before the
update runs, and it dies in three stages, all in the same log:

1. `CUDA graph capture failed (OutOfMemoryError... Tried to allocate 2.00
   GiB... 1.79 GiB is free) - eager fallback`
2. `torch.compile failed (... Tried to allocate 4.00 GiB ...) - eager update`
3. inductor autotune OOMs at 23.24 GiB in use, and every rank exits 1.

**Retried at T=16, per the protocol, 131,072 envs/rank RUNS** - 23.55 GB of
24.0, no graph or compile failure - and delivers **841,528 steps/s**, 35%
BELOW the T=32 optimum. So the T=32 ceiling is bounded by VRAM, but the
throughput ceiling is 65,536 and the optimum is 32,768; you never want the
rung that VRAM forbids.

Equal-buffer check (T*N = 1,048,576 samples/rank either way):

| 4 ranks | steps/s |
|---|---|
| 32,768 envs, T=32 | 1,301,629 |
| 65,536 envs, T=16 | 1,317,603 |

Within 1.2%. **What the GPU cares about is the rollout buffer's SIZE, not how
it is split between envs and rollout length** - which is convenient, because
`--n-steps` is a real learning variable (round 21) and `--envs` is not.

### The same ceiling on a 32 GB card and an 80 GB card

**It is not the card.** Both bigger cards peak at the same 32,768 envs/rank:

| 1 rank, T=32 | 32,768 envs | 65,536 envs | change | VRAM at 65,536 |
|---|---|---|---|---|
| RTX 3090 24G (box B) | 363,819 | 336,582 | **-7.5%** | 22.1 G of 24 |
| RTX 5090 32G (box C) | 776,704 | 745,363 | **-4.0%** | 28.0 G of 32 |
| A100 80G (box D) | 702,626 | 710,770 | +1.2% | 26.3 G of **80** |

The A100 has 54 GB spare at 65,536 envs and gains 1.2%. **Nothing above
~32,768 envs per rank is worth buying on any card**, and a card bought for
its VRAM will not convert that VRAM into throughput on this workload.

---

## Part 2 - the cost table

Each configuration at ITS OWN best envs setting, which is 32,768 per rank
everywhere.

| # | configuration | $/h | best steps/s | steps/hour | **$ per 1e9 steps** |
|---|---|---|---|---|---|
| 1 | **1x RTX 3090** (box B) | 0.1683 | 363,819 | 1.31e9 | **0.129** |
| 2 | **4x RTX 3090** (box A) | 0.7067 | 1,301,629 | 4.69e9 | **0.151** |
| 2b | 4x RTX 3090, `--minibatches 32` @ 65,536 | 0.7067 | 1,320,715 | 4.75e9 | 0.149 |
| 3 | **8x RTX 5090** (box C) | 3.2010 | 5,050,843 | 18.18e9 | **0.176** |
| 4 | **4x A100 SXM4 80G** (box D) | 4.9080 | 2,538,464 | 9.14e9 | **0.537** |
| - | round 21 champion (4x3090 TR 3960X) | 0.8170 | 1,043,948 | 3.76e9 | 0.217 |

Per-GPU figures for reference (each box's price divided by its GPU count, at
the measured single-rank throughput): 3090 $0.129, 5090 $0.143, A100
**$0.485**.

**Round 21's $0.217 is beaten by 1.44x and NONE of it is a code change.** It
is the same branch, one rung further up the envs ladder (32,768 vs 8,192,
worth 1.17x) on a box that costs $0.177 per GPU-hour instead of $0.204
(worth 1.15x), plus a healthier draw. **The $/h of the box remains the whole
story** - round 21's "the ordering is the ordering of $/h" survives a second
round and a second card generation.

**The A100 is 3.6x the cost per step of a 3090**, and the H100 was not rented
because that ratio settles it: at $8.5/h for 4x H100 PCIE, even at twice the
A100's throughput it would land near $0.9 per 1e9. **Datacenter cards buy
capability this workload does not need** - their VRAM converts to +1.2% (see
Part 1), and the only real difference is NVLink, which is worth 6 points of
DDP efficiency, not a different price class.

### Where DDP scaling turns over, and what causes it

**It does not turn over by 8 ranks.** All at 32,768 envs/rank, T=32, each
against its OWN box's single-rank number:

| box | ranks | aggregate steps/s | speedup | efficiency |
|---|---|---|---|---|
| A (4x3090) | 1 | 387,512 | 1.00x | 100% |
| A | 2 | 726,966 | 1.88x | 93.8% |
| A | **4** | 1,301,629 | **3.36x** | **84.0%** |
| D (4xA100, NVLink) | 1 | 702,626 | 1.00x | 100% |
| D | **4** | 2,538,464 | **3.61x** | **90.3%** |
| C (8x5090) | 1 | 776,704 | 1.00x | 100% |
| C | **8** | 5,050,843 | **6.50x** | **81.3%** |

Round 21's 2.9-3.4x at 4 ranks reproduces (3.36x). **Going 4 -> 8 ranks costs
about 3 points of efficiency on a box with 32 threads per GPU** - a slope,
not a cliff, and 8 ranks of 5090 still deliver 5.05M steps/s from one process
group.

**The mechanism is CPU contention, not the interconnect.** Decomposing box
A's 1 -> 4 rank penalty at 32,768 envs/rank (ms per iteration; `rollout_wall`
contains `env`, and `allreduce` runs inside `update` under the step-15
overlap, so the columns are not additive):

| phase | 1 rank | 2 ranks | 4 ranks | 1 -> 4 delta | share of the loss |
|---|---|---|---|---|---|
| `env` (OpenMP physics) | 472.0 | 724.1 | 1,526.5 | **+1,054.5** | **68%** |
| `rollout_wall` (incl. env) | 2,263.2 | 2,498.6 | 3,377.8 | +1,114.6 | 72% |
| `update` (incl. overlapped comm) | 5,845.9 | 6,036.6 | 6,218.5 | +372.6 | 24% |
| `allreduce` (measured, mostly hidden) | 0 | 232.9 | 428.3 | +428.3 | - |
| **total** | **8,117.8** | **8,654.4** | **9,667.0** | **+1,549.2** | |

Two independent confirmations that this reading is right:

* **NVLink.** Box D's all-reduce is **27.9 ms** against box A's 428.3 - 15x
  smaller - and its 4-rank efficiency is 90.3% against 84.0%. Deleting the
  comm cost entirely buys **6 points**; the other **10 points are the CPU**.
* **8 ranks on 256 threads.** Box C at 8 ranks has 32 threads per GPU (three
  times box A's 10.7) and still loses only 18.7%, with `allreduce` at
  345.9 ms of a 4,982.5 ms iteration (**6.9%**) and `env` up 350.0 -> 778.7 ms.

**So the rank-count limit is set by `cores / ranks`, exactly as gate A of
`docs/multimap-ddp-plan.md` predicted.** An 8-GPU box needs ~64 effective
cores; box C's 256 is why 8 ranks work. A faster fabric is worth 6 points;
more cores are worth 10.

### Host RAM per rank - the multi-map constraint

Measured RSS per rank, cannonball, goal field at cell 32:

| envs/rank | 8,192 | 16,384 | 32,768 | 65,536 |
|---|---|---|---|---|
| RSS (box B, 1 rank) | 4.33 G | 4.79 G | 4.70 G | 4.74 G |
| RSS (box A, 4 ranks) | 5.32 G | 5.64 G | 6.05 G | - |

**The env count is very nearly free: ~7.5 KB per env** (57,344 extra envs
cost 0.41 GB). Per-rank RAM is `~3 GB of process/torch/CUDA` + `the goal
field, SDF and occupancy of the maps that rank holds` + `7.5 KB x envs`. The
32,768-env optimum costs 0.25 GB of it.

Headroom per rank on the boxes measured: **62.8 GB** (box A, 251 G / 4),
**59.0 GB** (box C, 472 G / 8), 500 GB (box D), 126 GB (box B). Against the
plan's 0.83 GB/rank of cell-48 goal fields for 47 maps sharded over 8, and
~2.8 GB/rank scaling that to 161 maps, **no box in this class is RAM
constrained, sharded or not** - 161 maps at cell 48 held UNSHARDED by every
rank is ~22.6 GB/rank, which still fits box A's 62.8 GB. At cell 32 unsharded
it would be ~75 GB/rank and would NOT fit; sharding or cell 48 fixes it, and
the plan already requires both.

---

### The recommendation

**For the 161-map run: 4x RTX 3090, one rank per GPU, 32,768 envs per rank,
T=32, `--minibatches 16`, launched with `tools/ddp_launch.sh 4`.** 4.69e9
steps/hour at $0.151 per 1e9. Filter offers on `cpu_cores_effective /
num_gpus >= 8` (round 21) and prefer `gpu_frac == 1.0`; the price per
GPU-hour is the only other thing that matters.

**The tension the user named, resolved.** Four separate single 3090s are
still cheaper per step - $0.129 against $0.151, a **1.17x** premium for DDP,
down from round 21's 1.8x. But they train four SEPARATE policies, and Part 1
closes the single-GPU escape hatch: **one 3090 tops out at 363,819 steps/s
and no envs setting lifts it** (65,536 is 7.5% slower, 131,072 dies). One
policy on one GPU is capped at 1.31e9 steps/hour, which spread over 161 maps
is 8.1M steps per map per hour - about 1/500th of what one map needed to
reach the current frontier. **The multi-map run needs the batch, the batch
needs ranks, and 1.17x is a cheap price for it.** Take DDP.

**Scale by adding ranks, not by growing them.** A rank stops improving at
32,768 envs on every card tested; ranks are still 81% efficient at 8. If
4.69e9 steps/hour is not enough, the next box is 8x5090 at 18.2e9 steps/hour
for a 1.17x cost penalty ($0.176) - a fair trade when wall-clock is the
binding constraint, which for a 161-map run it may well be.

**Do not buy A100/H100 for this.** 3.6x the cost per step, and the 80 GB that
justifies the price converts into +1.2% of throughput.

---

### Ops findings, and four things that cost time

* **`torchrun` is not the only place OMP gets mis-sized - `nproc` is.** Box B
  is a `gpu_frac` 0.25 slice with a **7.68-CPU cgroup quota**, but
  `os.sched_getaffinity` returns the host's **32**, so `train_fast.py`'s own
  `_default_omp_threads` (cores/2) asked for **16 threads on 7.68 CPUs**.
  Measured at 8,192 envs, T=32: OMP 16 (the default) **280,673**, OMP 8
  **341,697**, OMP 4 306,924 - the naive default costs **21.7%**, all of it
  in `env` (487.1 -> 302.9 ms). Round 21 flagged this for `ddp_launch.sh` on
  shared multi-GPU boxes; it applies to the SINGLE-GPU default path too, and
  it is the difference between $0.157 and $0.129 per 1e9 on box B. **On any
  box with `gpu_frac < 1`, read `/sys/fs/cgroup/cpu.max` and set
  `OMP_NUM_THREADS` to about that number.** (Teaching `ddp_launch.sh` and
  `_default_omp_threads` to read `cpu.max` instead of `nproc` would fix both
  halves; not done here.)
* **An 8x5090 box ran 8 ranks perfectly and could not run 4, 2, or any
  subset.** Box C's 8-rank run is clean; **six** attempts at 4 and 2 ranks all
  aborted within 30-50 s in the NCCL watchdog with a CUDA error - across
  `NCCL_P2P_DISABLE=1`, `+NCCL_SHM_DISABLE=1`, `--ddp-overlap 0`, and both
  NUMA halves (`CUDA_VISIBLE_DEVICES=0,1,2,3` and `4,5,6,7`). The first
  attempt, with P2P left on auto, **hung** for 8 minutes at the first
  collective instead. So this round has no measured 4x5090 row: the 4x5090
  offers are literally half of this machine at exactly half the price
  ($1.601), and a 4-rank number on it could not be obtained. **Machine
  recorded known_good WITH the caveat** - it is a fine 8-rank box and a broken
  4-rank one. If a 4x5090 row is needed, rent a 4-GPU offer directly and
  expect to debug NCCL.
* **There was exactly ONE 8x3090 offer in the whole market** (m148206,
  $1.016/h, 72 cores, reliability 0.934) and it never came up: 22 minutes
  after create it was still on `Verifying Checksum` of the image pull.
  Destroyed, **not blacklisted** - a slow pull is not a defect (round 21).
  The 8-rank question was answered on 5090s instead, which is the harder test
  anyway: a 5090 rank finishes its compute ~2x faster, so the same gradient
  all-reduce is a larger fraction of its iteration.
* **Box-to-box seeding is not always possible, and the workstation uplink was
  the pole.** With another agent's 9-box bake running, `scp` from here ran at
  **~200 KB/s** and the 152 MB cache set took **14 minutes** on a $3.2/h box.
  The recommended fix (`SEED_HOST`/`SEED_PORT`) then FAILED for the A100 box -
  the 5090 box could not reach its direct SSH port at all (`connect ... timed
  out` after 2m10s), so rented boxes are not mutually routable in general.
  Pushing only the four files actually needed (`occ_32`, `slabocc_32`,
  `sdf_32`, `goal_32` = 57 MB, against the 152 MB `*.npz` glob
  `deploy_box.sh` sends) took under a minute; **md5 verified on arrival**, per
  CLAUDE.md.
* **`bench_scale.sh`'s VRAM sampler is not trustworthy on every rung** - it
  reported 1,550 MiB for a rung a direct `nvidia-smi` read showed at
  20,796-21,308 MiB. The peaks quoted above for that rung are the direct
  reads. Also note every peak includes the inductor autotune spike, which is
  larger when the card is otherwise empty: box A's 1-rank 32,768-env run
  peaked at 22.2 GB where four ranks doing the same work per rank peaked at
  14.3 GB each. **The steady-state requirement is the 4-rank number; the
  1-rank number is the allocator helping itself to free memory.**

### What this changes in `docs/multimap-ddp-plan.md`

* Gate B's envs answer is settled, and it is neither prior answer: the 5090's
  "peak at 4,096" (a local-box artefact) and round 21's "still improving at
  8,192" are both superseded by **a peak at 32,768 envs/rank at T=32 on 3090,
  5090 AND A100**. The plan's "scaling past that must come from MORE RANKS,
  not bigger ranks" was right, at a rung 4x higher than it thought.
* The plan's rollout-VRAM table is **2x too large** - `b_img` is bf16.
* Gate H is answered: **DDP scaling does not turn over at 8 ranks (81.3%)**,
  and the limiter is `cores/ranks`, which is gate A's variable, not the
  fabric.
* The goal-field RAM concern is real but not binding on any box in this
  class: **~7.5 KB per env** and ~3 GB of fixed process, against 59-63 GB of
  headroom per rank on both multi-GPU boxes measured.
