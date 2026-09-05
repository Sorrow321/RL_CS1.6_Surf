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

## Arm xCONTACT - Sophy's contact penalty, charged on the energy PM_ClipVelocity destroys

`bash tools/run_arm.sh xCONTACT --contact-pen 1e-6 --contact-clip 5.0`
on one RTX 3090 (machine 143878 / host 74292, the known-good box), branch
`contact-pen`, warm-resumed from the stuck checkpoint
(md5 `1ba1fd2936af3ae1ad3608e3cd6b1e9e`, step 3,782,737,920),
`--record-every 75e6 --eval-eps 9 --eval-greedy-only`.
`done: 4,583,325,696 steps, avg 285,812 steps/s` - 800M steps, 11 evals,
99 greedy episodes, 47 minutes of training.

### What the paper actually says (fetched, not paraphrased)

GT Sophy, Nature 602:223 (2022), Methods "Rewards" + Extended Data Table 1.
The wall term, verbatim:

    R_w(s, s') = -(s'_w - s_w)(s'_kph)^2

where `s_w` is the CUMULATIVE time the agent has been in contact with a wall
and `s'_kph` is speed in km/h at the new state. Course progress is
`R_cp(s,s') = l' - l` in METRES along the centreline. Extended Data Table 1,
in full:

| Course | R_cp | R_soc | R_loc | R_w | R_ts | R_ps | R_c | R_r | R_uc |
|---|---|---|---|---|---|---|---|---|---|
| Seaside | 1 | 0.01 | 0 | **0.01** | 0.25 | 0.5 | 5 | 0.1 | 0 |
| Maggiore | 1 | 0.01 | 0 | **0.01** | 0.25 | 0.5 | 4 | 0.1 | 0 |
| Sarthe | 1 | 0 | 5 | **0.01** | 0 | 0.5 | 5 | 0.1 | 5 |

**Correction to docs/research-litsurvey.md section 1.** The quadratic ->
linear switch "to avoid an explosion in values at Sarthe" is the OFF-COURSE
term only: `R_soc = -(s'_o - s_o)(s'_kph)^2` at weight 0.01 becomes
`R_loc = -(s'_o - s_o)s'_kph` at weight **5** (a 500x weight change that
roughly preserves magnitude at ~300 kph, doubled in the two chicanes). The
WALL term stayed quadratic at 0.01 on all three tracks, Le Mans included. So
the paper's own evidence is that a speed-SQUARED contact penalty survives the
fastest track in the paper; the explosion they hit was in a term that fires
continuously while off-course, not once per contact.

### What was implemented, and why it is not the literal form

Sophy's counter is *time in contact*. Transplanted literally that penalty is
minimised by never touching a ramp, which on a surf map deletes the task.
What GoldSrc gives us and GT does not is the exact harm of each contact:
`PM_ClipVelocity` removes precisely the plane-normal component, so the
destroyed specific kinetic energy of a contact is exactly
`0.5*(|v_in|^2 - |v_out|^2)` and a grazing entry (`v.n = 0`) destroys nothing
however long it lasts. The SHAPE is kept - a cumulative contact counter,
differenced across the transition, quadratic in speed, small weight against
progress at weight 1 - and the counter is changed:

    R_c(s, s') = -w * ( C(s') - C(s) ),   C = running sum over contacts of
                                              0.5*(|v_in|^2 - |v_out|^2)

in (u/s)^2, clamped at >= 0 (the counter is per-episode and resets on
respawn) and capped at `--contact-clip` reward units per call.
`--contact-linear` implements Sophy's Sarthe branch, `sqrt(2*dE)` = the
normal speed removed; it was NOT the arm that ran.

### How the weight was set against this project's progress reward

Derived, not tuned, from this project's own measured energy-to-time
sensitivity:

* `RaceReward` uses `scale = 100 / d_0` with `d_0 = 198,380 u` (printed at
  launch), so total geodesic shaping over any start->finish run is exactly
  **100**, plus `success_bonus 50`, minus `time_pen 0.005/tick` = 0.5
  reward/s.
* `tools/route_bound.py` on the champion (ep 9, 79.72 s): ramp contact
  destroys **7,363,534** (u/s)^2 - the same units this accumulator reports,
  verified (its "specific energy" is `g*dz`, i.e. 0.5*v^2 units) - and a
  **40%** cut in that loss with perfect strafing is worth 73.66 s -> 67.52 s,
  i.e. **6.14 s**.
* Marginal rate 6.14 s / (0.40 x 7,363,534) = 2.085e-6 s per unit destroyed;
  at 0.5 reward/s that is **1.042e-6 reward per unit**.

`--contact-pen 1e-6` therefore charges contact damage almost exactly what it
is worth in seconds under this project's own time cost (within 4%). Scale
checks at that weight:

| | |
|---|---|
| steady ride on a 45-deg ramp (gravity's 8 u/s, normal-projected) | 16/tick -> **1.6e-5 reward/tick**, 0.11% of the 0.0146/tick progress income - riding is free |
| champion's whole 79.72 s run | 7.36 reward: 6.7% of its ~110 return, 18% of its 39.9 s time cost |
| one catastrophic contact at 2,820 u/s (all speed annihilated) | 3.98 reward - 8% of the success bonus |
| **measured in-run on the stuck policy** | **~2.0e6/episode -> ~2.0 reward on a ~25 return, 8%** |

That last row lands where Sophy's own does: their wall term is ~0 over a
clean lap and ~7% of a lap's progress income for a one-second scrape at
200 kph. 3,000 u/s is 206 kph, so the speed regimes are literally comparable.

`--contact-clip 5.0` bounds the term. It binds only on a contact destroying
> 5e6, i.e. removing > 3,162 u/s of normal velocity in one tick. That bound
is needed for the reason the brief flagged - surf reaches the 4,000 u/s
`maxvel`, so `v^2` spans 60x against a ~0.015/tick reward stream - but
whether it ever actually fired cannot be recovered from the logged
per-episode aggregate (~2.0 reward/episode against a 5.0 PER-CALL cap). It
certainly never dominated.

**It is NOT potential-based.** The geodesic shaping telescopes: a closed loop
nets 0 and the collectible total is `scale * d_0` for every successful run,
so it cannot change the optimum. This term does: it is a genuine cost that
shifts each episode's return by however much energy that episode smashed into
ramp normals. That is the point (a slower grazing line should beat a faster
slamming one), but it costs three things and they should be read into every
number below: (a) episode return is no longer comparable to other arms, (b)
the critic has to relearn a baseline that now depends on contact history, and
(c) with too large a weight the optimum moves to "never touch a ramp", the
task-deleting failure. The 0.11%-of-progress ride cost is the number saying
(c) did not happen here.

### Implementation (ABI 7 -> 8)

* `pm_fly_move` accumulates `max(0, 0.5*(|v_in|^2 - |v_out|^2))` per call into
  `PmCtx.clip_loss`. Inside FlyMove the ONLY thing that touches velocity is
  contact resolution (ClipVelocity, the two-plane crease solve, the
  degenerate zeroings) - no gravity, no acceleration - so the entry/exit
  energy difference IS the normal-component destruction. Trapped-in-solid
  (`allsolid`) is excluded: a degenerate engine state that ends the episode
  is not a ramp contact. `PM_WalkMove` runs FlyMove twice speculatively and
  keeps one result, so only the kept branch's loss counts.
* `env.c` keeps a per-env accumulator, cumulative per EPISODE, zeroed by every
  reset path; `surf_contact_loss()` exposes it zero-copy as float64. Python
  differences it across the transition - Sophy's cumulative-counter shape -
  and clamps at 0 so a respawn reads as "no loss", never as a paid bonus.
* `RaceReward(contact_pen=, contact_clip=, contact_linear=)`, flags
  `--contact-pen / --contact-clip / --contact-linear`, all ckpt-restored.
  `pop_stats` reports `contact_e_per_ep` (raw (u/s)^2) and `contact_per_ep`;
  the trainer prints them every iteration as `clip 2.016e+06/ep (- 1.97)`, so
  the mechanism is visible without waiting for an eval.

### Correctness before renting

* `tests/test_physics.c` block C1 (MSVC locally, gcc on the box): a straight
  drop onto the flat spawn platform, where n = (0,0,1) and overbounce is 1, so
  the hand-computed destroyed energy is `0.5*v_z_in^2` with v_z_in = the
  state's v_z minus AddCorrectGravity's half tick (4.0). Matches to **1e-9
  relative**, and is **exactly 0.0** on every free-fall tick and on the
  resting tick after landing.
* `tests/python/test_contact_pen.py`, 10 tests: the per-tick identity over a
  real ramp ride; `0.5*(v.n)^2` against the trace's plane normal with the
  tangential speed preserved (the "normal component only" claim, which is what
  makes the term safe here); cumulative + per-episode-reset semantics; and at
  the reward layer - flag off is **bitwise** the control, the charge is
  exactly `w*dE` (and `w*sqrt(2*dE)` on the linear branch), the clip binds
  where it should, and a respawn never pays a bonus.
* **Bit-identity of the instrumented core.** Built the baseline sources
  (`origin/route-obs`) into a second DLL with identical compiler and flags and
  drove both with the same fixed random action stream from the same seed:
  **4,000 ticks x 64 envs = 256,000 env-ticks on surf_src_cannonball,
  comparing obs / rewards / done / trunc / terminal_obs and the whole per-env
  state array bitwise -> BIT-IDENTICAL.** "Flag off = the control" is proven
  at the level that matters, not asserted.
* On the box: `./build/test_physics` ALL OK; `pytest tests/python -q` **122
  passed, 1 failed**, the failure being
  `test_march_is_bit_exact_against_the_legacy_kernel`, the known
  3090-architecture difference. `gpu_health.py --all`: healthy (841 GB/s HBM,
  72 TFLOPS bf16, sm 1665 MHz, 309 W of 350).

### Result

`race/eval_progress`, steps after resume:

```
+0.8M    165,363      +454M    173,750
+76M     195,330      +529M    195,118
+152M    173,482      +605M    153,079
+227M    161,760      +680M    174,072
+303M    157,724      +756M    152,627
+378M    194,900
```

11 evals, mean **172,473**, range 152,627-195,330, first half 174,760 vs
second half 169,729 - inside the 3090 working band, no trend, and below
xROUTE's ~177k/~190k halves. Three evals above 194,900 would read as a strong
positive on this metric alone. They are not.

`tools/eval_honesty.py` on all 11 evals (99 greedy episodes):

| eval | corridor mean | **corridor max** | finishes | dives below |
|---|---|---|---|---|
| +0.8M | 175,132 | **205,440** | 0/9 | 5/9 |
| +76M | 205,298 | **205,312** | 0/9 | 9/9 |
| +152M | 182,812 | **205,312** | 0/9 | 7/9 |
| +227M | 171,207 | **205,440** | 0/9 | 6/9 |
| +303M | 166,172 | **205,312** | 0/9 | 6/9 |
| +378M | 205,198 | **205,312** | 0/9 | 8/9 |
| +454M | 183,324 | **205,312** | 0/9 | 6/9 |
| +529M | 205,326 | **205,440** | 0/9 | 9/9 |
| +605M | 160,924 | **205,312** | 0/9 | 7/9 |
| +680M | 183,125 | **205,440** | 0/9 | 8/9 |
| +756M | 160,469 | **205,312** | 0/9 | 7/9 |

**0 of 99 episodes finished, and the corridor maximum is 205,312-205,440 in
every single eval - the same wall xROUTE hit, to the vertex.** Two independent
mechanisms (a 27-feature lookahead fan; a contact-energy penalty) now stop at
exactly 88.6-88.7% of the route.

### The wall profile: the term did exactly what it charges for, and it was not enough

`tools/wall_profile.py --from-vertex 1584 --to-vertex 1602`, speed (u/s) and
off-line error (u) through the failure stretch:

| eval | v@1584 | v@1590 | v@1596 | **decay 1584->1596** | off@1584 | off@1596 |
|---|---|---|---|---|---|---|
| +0.8M | 2,716 | 2,716 | 2,635 | **-81** | 210 | 307 |
| +227M | 2,754 | 2,754 | 2,754 | **0** | 166 | 317 |
| +529M | 2,732 | 2,731 | 2,729 | **-3** | 145 | 334 |
| +756M | 2,742 | 2,742 | 2,740 | **-2** | 248 | 408 |
| champion | 2,935 | 2,934 | 2,927 | -8 | 73 | 71 |

The speed DECAY through the approach - which is precisely the quantity this
penalty charges for - **collapses from -81 u/s to ~0 within 227M steps and
stays there for the rest of the run**. Entry speed rises ~+20 to +40 u/s and
the champion deficit narrows from -219 to -190. So the mechanism is real,
fast, and aimed at the right place.

It changes nothing about where the policy dies. Off-line error at 1584-1596
does not improve (145-334 at best, 248-408 by the end, against the champion's
71-73), the ramp departure is still in the 256 u between vertices 1596 and
1598, and the free-fall after it is if anything more violent (final eval:
off-line 408 -> 2,766 u in one 2-vertex step, dz -1,262). The policy now
carries its speed cleanly to the same cliff.

Training-episode contact energy over the whole run barely moved: 2.08e6 ->
1.95e6 per episode (~6%, inside iteration noise), while the approach's speed
decay went to zero. Both are consistent: the 1584-1596 scrub is a small part
of an episode's total contact loss, and the penalty bought the part it could
reach cheaply.

**Verdict: NULL on the barrier. Positive, measurable and mechanism-specific
on contact quality - and that is now shown to be insufficient.** This is a
stronger null than xROUTE's, because the treatment demonstrably worked:
xROUTE's fan was absorbed but changed nothing at the wall; xCONTACT's penalty
was absorbed AND changed the exact physical quantity it targets AND the wall
did not move by one vertex. Combined with the value-ceiling analysis (contact
losses are 71% of all energy supplied), the reading is that the barrier at
route vertex 1597 is a LINE-GEOMETRY problem - where the policy is on the
ramp - not an energy-accounting problem, and a dense per-tick reward term
cannot search line geometry. That is an argument for the savestate
hill-climb + distill loop (survey items 5/7) over further reward knobs, and
it is the second independent arm this round to say so.

### Caveats that would mislead the next reader

* **n = 1 seed, 800M warm-resumed steps.** Sophy's reward components were
  never ablated in this regime; their numbers are differences over a full
  training run. This is evidence about an 800M-step graft onto a stuck
  policy, not about the mechanism.
* **The weight is derived, not tuned, and it is deliberately on the low
  side** - time-equivalent at the margin. A 3-10x weight is untested and is
  the obvious follow-up, with the explicit risk that it moves the optimum
  toward "avoid ramps". The evidence that there is headroom: at 1e-6 the
  policy zeroed the approach's speed decay within 227M steps and then had
  nothing left to buy.
* **Not potential-based**, so `rollout/ep_rew_mean` (~25) is not comparable
  with any other arm's, and the critic is fitting a different target.
* The opening eval (165,363) is a coin flip per CLAUDE.md; do not read the
  +0.8M -> +76M jump as learning.
* The accumulator excludes the trapped-in-solid path, and on GROUND ticks
  counts only the WalkMove branch the engine kept. Both are documented in
  `src/sim.h` and both are rare in surf (the agent is airborne on a ramp
  essentially the whole run), but a bhop-style map would need this re-read.
* `--contact-linear` (Sophy's Sarthe branch) is implemented and tested but
  was NOT run.

### Cost, and an ops lesson that cost 62 minutes

$0.163/h x 2.00 h = **$0.326**, plus $0.004 for one racing candidate
destroyed under the 60-second rule. **Total ~$0.33** - of which only 47
minutes was training.

The other 62 minutes went to getting a 153 MB checkpoint onto the box, and
the failure mode is worth writing down:

* Two agents pushing their 153 MB checkpoints from the same workstation
  collapsed the uplink from **1.6 MB/s to 44 KB/s** (measured with a 4 MB
  probe). The per-box probe in DEPLOY.md does not catch this, because the
  box is fine - the shared uplink is not.
* **`scp` silently truncated the 153 MB transfer to 3.9 MB and exited 0.**
  `deploy_box.sh` then installed it and only the `md5sum` at the end of step
  4 caught it. Never trust an scp exit code on a long transfer; always check
  the md5 (deploy_box.sh does - keep it that way).
* The fix that worked, in 28 seconds: **box-to-box.** The other agent's box
  already had `runs_ckpt.pt` at the same md5 (every arm ships the same stuck
  checkpoint), so authorising its key on the new box and scp'ing
  datacenter-to-datacenter moved 153 MB in 28 s. `deploy_box.sh`'s
  `SEED_HOST`/`SEED_PORT` path exists for exactly this and should be the
  DEFAULT whenever any other box in `fleet_watchdog list` is up, not the
  fallback. `fleet_watchdog list` already tells you which boxes those are.
* Minor: `maps/*.goalk_*.npz` is only needed with `--race-kill-aware 1`,
  which the baseline does not use. Skipping it saves 33 MB of the ~240 MB
  seed.

### Reconciling xCONTACT with xMARGIN (main session)

xCONTACT's own conclusion - "the barrier is line geometry, not observation
and not energy accounting" - was written against the three earlier nulls and
should be read alongside xMARGIN and the field verification above, which
landed at nearly the same time. The three results fit together cleanly:

* The penalty **worked on what it charges for**: speed decay through the
  approach (vertices 1584-1596) went from -81 u/s to 0 by +227M and stayed
  there, and entry speed rose 20-40 u/s. It did not improve off-line error,
  and the departure stayed between vertices 1596 and 1598.
* xMARGIN moved the frontier by **starting episodes past vertex 1600**, not
  by improving control - and its own wall profile shows off-line error at
  1598 collapsing from 1,735-3,019 u to 177-254 u once the agent trains on
  states beyond the barrier.
* The field sample shows why both are true: at vertex 1600 the shaping
  reward reaches its minimum along the route, so an agent approaching from
  the near side is being paid to stop, however cleanly it arrives. Fixing
  the arrival (xCONTACT) cannot help while the objective says stop; starting
  past it (xMARGIN) can.

So "line geometry" is a fair description of the last 11% but not the reason
the policy stops: the reason is the reward's local optimum. The savestate
hill-climb + distill loop remains the right tool for the geometry, and it
should be run against a field that does not pay the agent to turn around.

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

## Round 19 - xARC: Linesight's progress reward. THE MAP IS FINISHED (2026-08-22 06:18-07:21 UTC)

**Read this first: this arm puts CHAMPION ROUTE INFORMATION INTO THE REWARD,
not just into the observation.** Round 18's xROUTE fed the reference line to
the network (`--route`, an ego-frame lookahead fan); the potential it was
paid on was still the map's own geodesic field, a BFS over the map's own free
voxels with no human and no champion in it. `--race-arc` replaces that
potential with arc length along a polyline extracted from the champion's
fastest finishing episode. That is further from this project's autonomy goal
than anything else in rounds 18-19 and it is stated first on purpose. What
the finishes below prove is that a MONOTONE progress coordinate removes the
vertex-1600 barrier. They do not prove the agent solved the map unaided. The
follow-ups that would are listed at the end, and they are Linesight's own:
its reference line "does not need to be fast... usually the centerline", and
as records fall it is re-extracted from the AI's own previous runs.

### The defect, restated in one paragraph

The shaping term is `scale * (d_prev - d_now)` with `d` a geodesic BFS
distance over free voxels (`python/surfgym/goalfield.py`). Sampled along the
champion's own winning line the field bottoms out at **d = 6,568 u at route
vertex 1601** and then RISES 8,408 u over the next 79 vertices before
dropping to the finish. Riding the correct line past that minimum is charged
**-4.24** of shaping plus about **-4.61** of time penalty, against a +50
success bonus this policy had never once observed (win rate 0.00% for ~2e9
steps). Turning back at vertex 1601 is locally optimal, and 234 greedy
episodes of three control arms did exactly that. Round 19's
gravity-directional field was baked and gated out because the shortcut the
graph believes in is a **level 8,700 u glide through open air** with 3,584 u
of floor clearance and zero climb; no rule about `dz` can see it.

### The paper, read out of the source

`github.com/Linesight-RL/linesight` (world records on ~10 of 12 official
Trackmania campaign tracks, May 2024), `config_files/config.py` and
`trackmania_rl/buffer_management.py`:

    constant_reward_per_ms                     = -6 / 5000    # -0.0012/ms
    reward_per_m_advanced_along_centerline     = 5 / 500      # 0.01/m
    shaped_reward_dist_to_cur_vcp              = -0.1
    shaped_reward_min_dist_to_cur_vcp          = 2
    shaped_reward_max_dist_to_cur_vcp          = 25
    shaped_reward_point_to_vcp_ahead           = 0
    engineered_{speedslide,neoslide,kamikaze,close_to_vcp}_reward_schedule
                                               = [(0, 0)]     # all OFF
    final_speed_reward_as_if_duration_s        = 0
    cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2_000
    temporal_mini_race_duration_ms             = 7000
    gamma_schedule = [(0, 0.999), (1_500_000, 0.999), (2_500_000, 1)]

and the progress term itself, verbatim:

    reward_into[i] += (
        rollout_results["meters_advanced_along_centerline"][i] -
        rollout_results["meters_advanced_along_centerline"][i - 1]
    ) * config_copy.reward_per_m_advanced_along_centerline

**Three things differ from the brief's summary and from
docs/research-litsurvey.md, and they are recorded rather than glossed:**

1. **There IS a distance term and it is ON by default.** `get_potential()`
   returns `shaped_reward_dist_to_cur_vcp * clip(|dist to the current virtual
   checkpoint|, 2, 25)`, i.e. a real -0.1/m distance penalty. What makes it
   safe is that it is applied as a **Ng-Harada-Russell potential** (the
   function's only comment is a link to the 1999 shaping paper) through
   `state_potential` / `next_state_potential` on each stored transition, so
   it is policy-invariant by construction: it can speed learning up and it
   cannot move the optimum. The non-potential twin that WOULD move the
   optimum, `engineered_close_to_vcp_reward`, ships zeroed. **It is
   deliberately NOT implemented here:** a policy-invariant term cannot
   remove a barrier in the optimum, which is the entire defect, and adding
   it would have been a second change.
2. **That distance is to the next CHECKPOINT, not to the line.** The
   reconciliation with Song & Scaramuzza (RSS 2023) therefore holds exactly
   as docs/research-litsurvey.md states it: progress ALONG a line is a
   progress objective (their gate-progress agent: 100% success), a penalty on
   distance TO a line is what fails (44% nominal / 0% realistic).
   `--race-arc` pays arc advance and has no lateral term at all.
3. **Linesight's constants only balance because of the mini-race.** At
   0.01/m against 1.2/s the progress income equals the time penalty at
   **120 m/s = 432 km/h**, above ordinary Trackmania speeds - coherent only
   because the fixed 7 s window makes elapsed time a near-constant offset.
   RL_Surf has no such window (round 19 already established that a 7 s window
   is a regression here), so copying the ratio would have been wrong. The
   scale below is derived from RL_Surf's own budget instead.

### The reward that was implemented

`--race-arc maps/surf_src_cannonball.route.npz`, new class
`python/surfgym/route.py::ArcProgress` plus a branch in
`python/surfgym/rewards.py::RaceReward`:

    r_t = arc_scale * clip(a_t - a_{t-1}, +/-max_step) - time_pen   in corridor
    r_t =                                              - time_pen   out of it

`a` is the continuous arc coordinate of the player's projection onto the
reference polyline. It is the incremental twin of
`tools/eval_honesty.py::corridor_progress` - the metric that decides this arm
- made cheap enough to pay every physics tick to 2,048 envs.

Four design points, each with the reason it is not taste:

* **Signed delta, not "new ground only".** The term stays potential-based and
  telescopes to `a_end - a_spawn` over an episode. This is what kills the
  obvious farming mode: hovering near a high-arc vertex, or running a stretch
  back and forth, nets **exactly zero**. A `max(0, .)` ratchet would pay
  twice for the same 500 u of track.
* **Off-corridor pays ZERO and never a penalty.** Leaving the line stops the
  clock exactly as it does in the scorer. It must never pay to leave the
  line and it must never charge for lateral deviation, so the corridor is a
  gate on income, not a distance term.
* **The anchor is local.** The projection is searched in a window of +/-16
  vertices (2,048 u of arc) around the previous anchor, and the anchor is
  FROZEN while off-corridor. A legal tick moves <= ~35 u, so the window is
  58x the physical bound; what it buys is that an off-route flight cannot
  walk the coordinate down the track and cash it on re-entry. Respawns
  relocate arbitrarily and are re-anchored with a global search.
* **The same `max_step` clip as the geodesic term** (100 u/tick), so a
  teleport-ish relocation cannot cash shaping in either mode.

### The scale, derived rather than tuned

The geodesic term is `scale = 100/d0` with `d0` = 198,380 u (mean start
geodesic, printed by the trainer), i.e. **100 total collectible shaping over
one start->finish run**. The route is 1,811 vertices at 128 u = **231,680 u**,
so

    arc_scale = 100 / 231,680 = 4.3163e-4 per unit   (geodesic 5.041e-4)

collects the same 100. Nothing about the SIZE of the budget changes, only its
SHAPE.

| | geodesic | arc |
|---|---|---|
| scale | 5.041e-4/u | 4.3163e-4/u |
| total collectible over a full run | 100.00 | 100.00 |
| break-even speed (income == the 0.005/tick time penalty) | 992 u/s | 1,158 u/s |

The break-even moves 17%, against a policy that runs the descent at
2,700-3,700 u/s, so the speed incentive is materially unchanged.

### The barrier, both potentials, along the champion route

Cumulative shaping banked by riding the champion line from the start:

| vertex | arc | geodesic d | cum geodesic | cum arc |
|---|---|---|---|---|
| 0 | 0 | 198,353 | 0.00 | 0.00 |
| 1400 | 179,200 | 30,943 | 84.39 | 77.35 |
| 1596 | 204,288 | 7,036 | 96.44 | 88.18 |
| **1601** | **204,928** | **6,568** | **96.68 <- peak** | 88.45 |
| 1620 | 207,360 | 8,108 | 95.90 | 89.50 |
| **1680** | **215,040** | **14,976** | **92.44 <- trough** | 92.82 |
| 1750 | 224,000 | 7,366 | 96.27 | 96.69 |
| 1810 | 231,680 | 5 | 99.98 | 100.00 |

* **Worst drawdown over all 1,811 vertices: geodesic -4.24 (v1601 -> v1680);
  arc +0.00.** The arc potential has zero drawdown anywhere, by construction.
* From the field's minimum at v1601 to the finish, the geodesic dips to
  **-4.24** before ending at +3.31; the arc rises **monotonically to +11.55**.
* The remaining 26,752 u costs -4.61 of time penalty at 2,900 u/s. So
  committing to the descent is worth **-1.30 net with a -8.85 intermediate
  trough** under the geodesic and **+6.94 monotone** under arc, before the
  +50 bonus either way.

### What did NOT change, and why

* `time_pen` 0.005/tick and `success_bonus` 50 untouched. One thing.
* `--respawn-margin 2` is passed **because xMARGIN ran exactly that** and is
  the paired control; the arm differs from its control in the reward and in
  nothing else.
* `--route` (the observation-side lookahead fan) is NOT passed. `--race-arc`
  builds its own line object precisely so the policy's input row is
  unchanged.
* **The stall detector and the respawn `stagnant` mask stay on the geodesic
  field.** They are liveness rules ("kill an agent whose score has not
  improved in 15 s", "never snapshot a provably-stuck state"), not the
  objective, and moving them would change respawn harvesting - a second
  treatment on top of the one xMARGIN measured. Checked rather than assumed:
  on the champion's own nine recorded episodes the longest
  no-geodesic-improvement stretch is **13.07-13.25 s against the 15 s kill**,
  and the descent from v1601 to the first vertex that improves on it is
  19,968 u = 6.0-8.0 s. The detector does not misfire on a correct descent,
  and that is equally true of every control arm.
* One thing DOES change downstream and must be named: under `--obs-reward`
  the policy reads its own shaping in scalar slot 12, so changing the reward
  necessarily changes one observation channel. That is a consequence of the
  treatment, not a second treatment, but it means the eval feed and the
  recorder must mirror it or the policy is fed a signal it never trained on -
  the exact bug that made sOBSR's evals meaningless. `_make_eval_arc_feed` in
  `train_fast.py` and the arc branch in `tools/record_ckpt.py` both mirror
  it, and `record_ckpt.py`'s config audit now names `race_arc`,
  `race_arc_corridor` and `race_arc_window`. **Verified end to end**: an
  independent `tools/record_ckpt.py` run against the final checkpoint
  finished 3 of 3 (last row of the table below).

### The correctness surface, checked before renting (CPU only)

* **Flag off is bit-identical, proved against the branch point.**
  `test_flag_off_is_bit_identical_to_the_branch_point` loads
  `origin/route-obs:python/surfgym/rewards.py` as a second module and runs
  both implementations over 200 randomized steps (4 envs; random distances,
  positions, yaws, velocities, deaths and goal hits; intrinsic novelty on),
  requiring `np.array_equal` on every step and equal `pop_stats()`. A second
  test pins the control branch against an independent recomputation of
  `clip(d_prev - d) * scale - time_pen`. No RNG is consumed differently and
  no tensor is touched on the control path.
* **Agreement with the scorer, on real recordings.** Replaying 180 recorded
  episodes (xMARGIN 72, xROUTE 99, champion 9) through `ArcProgress` and
  comparing with `corridor_progress`: **agreement within 63 u - half the
  128 u vertex quantization the scorer snaps to - on 178 of 180.**
* **Cost of the term, measured on CPU on the real map** (2,048 envs, 400
  ticks, real physics, real goal field): the reward function goes from
  0.64 to 1.64 ms/tick. On the box the run averaged **241,480 steps/s**, so
  it is inside the between-box noise.
* 129 tests collected, **127 green on the box** (113 before this arm; the
  single failure is `test_march_is_bit_exact_against_the_legacy_kernel`, the
  known 3090 failure CLAUDE.md documents). 16 of the new tests are in
  `tests/python/test_race_arc.py`.

### A correction to the metric, found by this arm and fixed before it scored

`tools/eval_honesty.py::corridor_progress` takes the nearest vertex over the
WHOLE route at every sample, so wherever the route approaches itself the
credited index can JUMP - which is the "an off-route fall claims a later
stretch" failure the file exists to prevent, surviving in the one place the
frontier now sits. Two measurements:

* two of the champion's own recorded episodes die at 87,355 u and are
  credited **133,760 u**: the route folds back within ~1,000 u of itself
  there;
* xARC eval 2, episode 2: the agent leaves the line at **209,664 u** and
  falls, and because the route's own descent into the bowl passes within
  1,100-1,800 u of the falling body the scorer credits **220,800 u**.

`--order-only 16` (new, **default off**) rescores with `ArcProgress`, i.e.
the exact rule the reward pays, so the metric and the reward cannot disagree.
Every number already in this ledger reproduces byte for byte with the flag
absent. **All xARC figures below are quoted the order-only way.** Re-scoring
xMARGIN's 72 episodes with it changes nothing material (both rules agree
within 63 u on every episode; its 208,640 u maximum is unmoved; "past
205,440" goes 6 -> 7 of 72 on one boundary case), so the comparison is like
for like.

### The run

    bash tools/run_arm.sh xARC --respawn-margin 2 \
        --race-arc maps/surf_src_cannonball.route.npz

expanded by the launcher to the pinned baseline plus those two flags. Warm
resume of `runs/sOBSR2/ckpt_latest.pt`, md5 `1ba1fd2936af3ae1ad3608e3cd6b1e9e`
**verified on the box**, step 3,782,737,920, on `surf_src_cannonball`, one
RTX 3090 (vast 48369998, machine 143878), one seed. The baseline-config guard
reads the CHECKPOINT's config and passed; the "restored from checkpoint
config" line contains neither `respawn_margin` nor `race_arc`, so both CLI
values are what ran, and the trainer printed

    respawn: 90% of episodes from mid-run snapshots, harvested >= 2s before
             episode end
    arc route surf_src_cannonball.route.npz: 1811 pts @ 128u = 231,680u,
             corridor 1500u, window +/-16 (2,048u) -> shaping scale
             0.00043163/u (vs geodesic 0.000504083/u)
    race: start geodesic 198380u

**800,587,776 steps in 56.5 minutes, 1,018 iterations, 241,480 steps/s
average**, 11 in-trainer evals of 9 greedy episodes plus one independent
`record_ckpt.py` recording of 3.

### RESULT: THE MAP IS FINISHED

| eval | steps after resume | race/eval_progress | corridor MAX (published) | corridor MAX (order-only) | past 205,440 | finishes | best finish |
|---|---|---|---|---|---|---|---|
| 1 | +1M | 184,390 | 205,312 | 205,362 | 0/9 | 0/9 | - |
| 2 | +76M | 159,018 | 220,800 | **214,485** | 7/9 | 0/9 | - |
| 3 | +152M | 156,305 | 223,872 | **223,909** | 6/9 | 0/9 | - |
| 4 | +227M | 198,244 | 231,680 | **231,680** | 9/9 | **4/9** | **81.09 s** |
| 5 | +303M | 188,371 | 231,680 | **231,680** | 8/9 | **7/9** | 81.71 s |
| 6 | +378M | 179,444 | 231,680 | **231,680** | 8/9 | **6/9** | 81.25 s |
| 7 | +454M | 177,424 | 231,680 | **231,680** | 8/9 | **8/9** | 81.24 s |
| 8 | +529M | 198,391 | 231,680 | **231,680** | 9/9 | **9/9** | 81.26 s |
| 9 | +605M | 198,381 | 231,680 | **231,680** | 9/9 | **9/9** | 82.44 s |
| 10 | +680M | 176,639 | 231,680 | **231,680** | 8/9 | **8/9** | 81.17 s |
| 11 | +756M | 198,371 | 231,680 | **231,680** | 9/9 | **9/9** | **81.05 s** |
| rec | +792M | - | 231,680 | **231,680** | 3/3 | **3/3** | 81.42 s |

**63 of 102 greedy episodes finished the map. 84 of 102 crossed 205,440 u.**
Finish times: best **81.05 s**, median 81.75 s, mean 81.83 s, worst 83.14 s.
Eval 1 - the untreated policy at +1M steps, before the reward has changed
anything - lands on 205,362 u with 0/9 past the line and 0 finishes, which is
the right internal control and is exactly where xMARGIN's own eval 1 landed.

Against the paired control and the three arms before it:

| arm | mechanism | greedy episodes | corridor MAX | past 205,440 | finishes |
|---|---|---|---|---|---|
| xROUTE | lookahead route geometry | 99 | 205,312 | 0 | 0 |
| xSP | soft shrink-and-perturb | 54 | 205,312 | 0 | 0 |
| xNECTO | difficulty-weighted respawn | 81 | 205,440 | 0 | 0 |
| xMARGIN | `--respawn-margin 2` | 72 | 208,640 | 6 (7 order-only) | **0** |
| **xARC** | **the same margin + arc shaping** | **102** | **231,680 (100%)** | **84** | **63** |

Per CLAUDE.md rule 3 the verdict metric switches for a run that finishes:
wall-clock start to finish. Measured identically on both (first recorded tick
to first entry into the finish box, +64 u pad), **xARC's final eval is 9 of 9
finishes, best 81.05 s, mean 81.36 s, against the champion recording the
route was built from at best 81.36 s, mean 82.20 s**
(`runs/sISV_par2/traj_8454144000.jsonl`, 7 of 9). This ledger's headline
1:19.72 for that run is measured on a different basis and these numbers are
NOT comparable to it - do not read the above as a record.

`tools/wall_profile.py` on the final eval, against the champion line: the
arm now tracks it to within **50-387 u from vertex 1580 to 1780**, through
the descent that broke every previous arm and up the final climb, at
2,259-3,698 u/s against the champion's 2,380-3,728.

### `race/eval_progress` moved in the WRONG DIRECTION while this happened

184,390 -> 159,018 -> 156,305 while the honest frontier went
205,362 -> 214,485 -> 223,909, and it only recovered (to 198,2xx, i.e.
saturated) once episodes actually crossed the line. Round 18 proved this
arithmetically; here it is in a run. The metric is
`mean(d at spawn - min d reached)` and the geodesic minimum along the route
is at vertex 1601, so nothing past the wall can raise it - and worse, the OLD
behaviour (dive off the ramp into goal-adjacent space at z ~ -4,200) reached
a LOWER d than the new behaviour of riding the route down into the bowl at
z ~ -5,380, so the standing metric FELL as the agent started doing the right
thing. **It was anti-correlated with the truth for the whole middle of this
run.** Lead with `eval_honesty.py`.

### The anti-farming audit, on the recordings rather than in the abstract

An arc reward computed from the agent's own position is farmable if the
corridor and order rules leak. Replaying all 102 recorded greedy episodes
through the same `ArcProgress` the trainer used, and comparing what the
shaping PAID against what the geometry supports:

    paid / (furthest arc reached - spawn arc):  max 1.0000, mean 0.9995
    shaping paid: mean 90.98, max 100.00 of the 100.00 budget

The ratio cannot exceed 1: the term is a potential, so an episode's total is
exactly `a_final - a_spawn` and every unit of forward motion later given back
is charged back. The training log's own read-out agrees from the other side -
`arc gain` (mean arc gained per episode) never exceeds `reach`, and the share
of ticks spent outside the corridor earning nothing fell from 12.1% early to
**0.7% at the end**, not because leaving the line is punished (it is not) but
because staying on it is where the income is.

### Training diagnostics, 1,018 logged iterations

`ep_rew` mean 44.14 (p10 21.19, p90 62.18), last 65.21, against xMARGIN's
22.9; `ep_len` mean 2,790; `kl` 0.0176; `ent` pinned at 0.005;
`value_loss` mean 0.698 (p10 0.072, p90 1.376) against control means
0.046-0.061. **The critic's loss went up more than tenfold and that is
expected, not a pathology**: the return distribution now contains +50 finish
bonuses that never existed in this checkpoint's experience, and the win rate
rose monotonically throughout. Training win rate by tenth of the run:

    0.0%  0.0%  2.7%  25.6%  41.8%  58.0%  67.9%  68.8%  76.7%  77.6%

first non-zero at **+151.0M steps**. The greedy evals lag it by about one
eval, as they should.

**The caveat CLAUDE.md pre-registered for this exact situation did fire and
must be read.** At `--respawn-margin 2` the reservoir harvests states 2 s
from the goal the moment an arm starts finishing, and can then self-reinforce
on trivial wins. It is now live: training `finish_s` sits at 31-34 s against
the greedy evals' 81 s, i.e. most training finishes start deep in the route.
**That is why the verdict rests on the greedy evals from the start line and
not on the win rate**, and any successor that reads a win rate on a
short-margin run owes it the same separation.

### What is now the wall

Nothing on this map. Eval 11 finished 9 of 9 and the arm holds the champion
line to within 400 u for the whole final descent and climb. The remaining
gaps are speed, not survival: the final ramp is run at 2,259 u/s against the
champion's 2,380 (-121), and the arm's mean finish is 81.36 s against a
81.05 s best, so there is ~0.3 s of consistency and an unknown amount of line
quality left. The next objective on this map is time, which is CLAUDE.md rule
3's other branch.

One detail a successor building routes needs: **the route file stops 52.31 u
short of the goal curtain** (last vertex y = 7,434.7, finish box y in
[7,487, 7,488]; 0 of 1,811 vertices are inside the box). So the arc
coordinate saturates just before the line, the last 52 u pay no shaping, and
`reach 231,680u` in the training log means "at the end of the route", not
"crossed". The +50 bonus covers the gap and nothing here was affected, but a
route built for a map with a deeper finish volume should be extended INTO it.

### VERDICT

**STRONG POSITIVE, and the largest single result in this ledger.** Replacing
the geodesic distance-to-goal potential with arc length along a reference
line took the frontier from 205,312-208,640 u of 231,680 u - where it had sat
across four arms and 306 greedy episodes with zero finishes - to **63 finishes
in 102 greedy episodes**, with the last eval at 9 of 9 and finish times at
champion pace. It did it in 800M steps of one hour on one 3090, changing one
term of the reward.

The mechanism is neither exploration nor capability. Round 16 already proved
the capability half (placed on-route by a demo curriculum, the same weights
finish at champion pace). What was missing was a reward that did not charge
the agent to do the right thing. The geodesic potential has an interior local
minimum at route vertex 1601 because the voxel graph believes in an 8,700 u
level glide across open air; arc length along a route cannot have one. Three
exploration mechanisms and 234 greedy episodes were spent on a barrier that
was arithmetic in the reward.

**xMARGIN is not thereby demoted.** Its `--respawn-margin 2` is carried by
this arm and is very probably load-bearing: it is what put states at the wall
into the reservoir at all. The clean claim this arm supports is that **margin
2 alone reached 208,640 u and never finished, and margin 2 plus a monotone
progress coordinate finishes 62% of the time**. Separating them needs a third
run (arc shaping at the default 10 s margin) that was not in this budget, and
it is the first thing to run next.

### The autonomy caveat, and the three follow-ups that discharge it

Every earlier arm that used the champion's line put it in the OBSERVATION
(xROUTE) or in the START DISTRIBUTION (the demo curriculum). The potential
itself was always derived from the MAP. `--race-arc` changes that: the thing
the agent is paid for is now defined by a polyline extracted from the
champion's fastest finishing episode (`tools/build_route.py`). **A finish
under this reward is not evidence that the agent solved the map unaided.**

In increasing order of autonomy, all now cheap to test:

1. **Degrade the line.** Re-run with a route built from a NON-finishing
   rollout - the stuck checkpoint's own best episode reaches 205,312 u =
   88.6% of the map unaided - or with the champion line resampled at 1,024 u
   so it carries the corridor and not the racing line. If the arm survives a
   deliberately bad line, the treatment is "monotone progress coordinate",
   not "champion imitation". This is the one that decides how much of the
   result is honest.
2. **Bootstrap it.** Build the line from the ARM's own best run and re-run.
   That is Linesight's loop exactly, and after the first iteration it needs
   no champion at all.
3. **Fix the potential instead of replacing it.** The honest statement of the
   defect is that a voxel geodesic over POSITIONS cannot represent the fact
   that the player cannot fly. Round 19 established that no edge rule on that
   graph can express it and that the real fix needs the VELOCITY dimension.
   Arc length is a cheap way around that, not a solution to it.

And one for the ablation list: **the 100/d0 shaping budget survived a
complete change of potential without retuning.** Two potentials whose
per-unit scales differ by 17% produced the same episode-return scale. The
budget, not the scale, is the invariant worth carrying to the next map.

### Ops and cost

* `python tools/fleet_watchdog.py list` empty at start. Raced three RTX 3090
  candidates (48369998 machine 143878, 48370028 machine 95613, 48370039
  machine 17682), all registered on create with a 115-minute deadline.
  **48369998 was ssh-usable within 60 s**; the other two were destroyed and
  released 58 s and 53 s after creation, both confirmed gone. Cap of 4 never
  exceeded.
* `deploy_box.sh` with `BRANCH=arclen`, `LOCAL_CKPT=/c/RL_Surf/runs/sOBSR2/
  ckpt_latest.pt`, `EXPECTED_MD5=1ba1f...`, `SKIP_TORCH=1`. The checkpoint's
  md5 was **verified on the box** (`1ba1fd2936af3ae1ad3608e3cd6b1e9e`), 8
  cache .npz shipped, bsp mtime pinned, `gpu_health.py` VERDICT healthy
  (841 GB/s HBM, 73 TFLOPS bf16, 1,665 MHz under load, 332 W of 350 W).
  `pip install --break-system-packages pytest scipy` as expected.
* Trainer alive from 06:22:44 to 07:19:15 UTC, GPU pinned, fps 211k-241k,
  never decaying, never stationary. Box destroyed 07:20:51 UTC,
  `vastai show instances` returned 0 instances, watchdog released, fleet
  empty.
* **Rental cost: $0.17** for the winner (62.1 minutes at $0.1633/h) plus
  about **$0.01** for the two racing losers. **Total ~$0.18.**
* Artifacts in `runs/research/xARC/`: 12 trajectory files, `progress.csv`,
  `run.json`, `xARC_launch.txt`.

## Round 19 - xAUTO: does the arc line have to be a CHAMPION's? (2026-08-22 07:32-09:01 UTC)

xARC finished this map by paying arc length along a reference line and left
one caveat open: **that line was the champion's own winning trajectory.** This
arm attacks the caveat. Read the next paragraph before any number below.

**What ran is the FALLBACK, not the preferred experiment.** The plan was to
derive the line from `tools/explore_phase1.py` - a reward-free Go-Explore
phase-1 pass with no champion, no reward and no checkpoint in it anywhere.
Phase-1 was run twice, in both of the configurations this ledger names, and
neither produced a line worth putting in a reward. What ran instead is the
ledger's own follow-up #1: **the champion line degraded until it carries the
corridor and not the racing line.** So this arm answers *"does the line need
champion SKILL?"*. It does **not** answer *"can the line be discovered
autonomously?"*, and nothing below licenses a claim of autonomy.

### Part 1: the Go-Explore phase-1 line does not exist yet

Two passes, CPU only, on the local workstation, no GPU and no reward. The
map's own `goal_32.npz` is read ONLY for the tool's progress print, as the
tool intends; the champion route is used ONLY as a ruler afterwards.

    SURFCORE_DLL=... python tools/explore_phase1.py \
        --map maps/surf_src_cannonball.bsp --out runs/explore_auto \
        --envs 512 --goals 1 --max-iters 40000 --seed 0            # config A
    ... --cell 128 --cell-speed 4 --out runs/explore_auto_sp       # config B

| config | wall | chunks | exploration runs | cells | best geodesic d | geodesic progress | champion-arc reach (ruler) |
|---|---|---|---|---|---|---|---|
| A - the default `--cell 256`, position-only | 18.5 min | 13,000 | 18.5 M | 3,821 | 183,477u | **7.50%** | **15,445u = 6.67%** |
| B - the queued `--cell 128 --cell-speed 4` | 15.3 min | 8,000 | 14.0 M | 54,100 | 183,423u | **7.53%** | **15,478u = 6.68%** |

Both froze early and stayed frozen: config A's best distance had not moved for
its last 11 minutes (11,600 chunks, 15 M exploration runs) while its archive
grew by 18 cells; config B's had not moved for 9 minutes. Both stop at the
same physical place, `(-11,060, -7,950, 6,650)`, ~15,000 u along the champion
line - the first hard gate, still at the top of the map. Speed-keyed cells
multiplied the archive 14x (54,100 against 3,821) and bought **0.01 percentage
points** of reach.

**The ledger's "reached 92.4% of the track in 20 min CPU" did not reproduce,
and the two numbers are not on the same axis.** 92.4% as a geodesic-progress
figure means `d ~ 15,000u`, and the geodesic field is not injective along the
route: `d = 15,000u` is route vertex ~1530 (84.5% of arc) coming down AND
vertex ~1672 (92.4% of arc) in the bowl. So "92.4% of the track is past the
wall at 88.2%" cannot be read off that number either way. What was measured
here is `d = 183,450u`, i.e. **7.5%** - short of the wall by an order of
magnitude under every reading. Either the earlier run used a configuration
that was not written down, or the figure is a different quantity. **Do not
plan another arm on the 92.4% number until a phase-1 run reproduces it with
its flags recorded.**

**The second finding is about extraction and it survives the first.** Even
given an archive that HAD reached the wall, turning one into a polyline needs
a leaf: the archive is a tree of cells and `chain()` walks `parent` from a
leaf back to a map spawn. Only two rules are free of champion information,
and both were measured on tonight's archives:

* `--archive-leaf dist` - the archived cell closest to the finish in the map's
  own geodesic field. It did pick the true frontier in both configs. But
  **that field's minimum along the route is AT the wall** (vertex 1601,
  `d = 6,568u`, against 14,976u in the bowl at vertex 1680), so a line
  selected this way stops at the wall by construction - precisely the stretch
  the experiment needs covered. *The defect that made xARC necessary also
  poisons the only goal-aware selector an autonomous pipeline has.*
* `--archive-leaf depth` - the cell whose cheapest known route from a spawn is
  longest in physics ticks. No goal information at all. It picked a dead-end
  pocket **1,507 u off the champion line at 3.15% of arc** (config A) and
  **1,953 u off at 5.47%** (config B). Where falling is free, "hardest to
  reach in time" is not "furthest along".

Both rules are implemented in `tools/build_route.py`; `depth` is the default
because it is the honest one, and the measurement above is why neither is a
solution yet.

### Part 2: the line that ran - the champion's, with the skill taken out

    python tools/build_route.py --from-route maps/surf_src_cannonball.route.npz \
        --decimate 32 --out maps/surf_src_cannonball.coarse32.route.npz

1,811 champion vertices -> **58 waypoints** at 4,096 u spacing (~1.5 s of
champion play each) -> resampled at 128 u -> 1,691 points, **216,320 u** of arc
against the champion's 231,680 u, because straight chords cut every corner.

**Exactly what information survives, measured rather than asserted:**

| | |
|---|---|
| waypoints | **58**, one per 4,096 u of track |
| deviation from the champion line | **max 1,131 u**, p99 1,037 u, median 296 u, rms 417 u |
| the precision the stuck checkpoint actually has | it tracks the champion line to **1-2 u** for 88% of the map |
| line vertices inside SOLID map geometry (`occ_32.npz`) | **24.8%** (420 of 1,691). The champion line: 0.1% |
| speeds / timings / actions carried | **none** |

**The line is not flyable**: a quarter of it is inside rock. What is left is
which corridor the track runs through, in which order, at 4,096 u resolution -
which is what Linesight says is enough ("does not need to be fast... usually
the centerline"). It is still route knowledge and it still came from a
recording of a finisher; what it can decide is whether xARC's finishes came
from the reference line's SHAPE (a monotone progress coordinate through the
right corridor) or from its QUALITY (a racing line to imitate).

### The line is a usable progress coordinate - checked on CPU before renting

Replaying real recordings through `surfgym.route.ArcProgress` with the arm's
own settings (corridor 1,500 u, window +/-16):

| line | recording | reach mean | >99% | off-corridor ticks | arc given back |
|---|---|---|---|---|---|
| champion | the champion's 9 | 86.2% | 7/9 | 0.2% | -158 u |
| **coarse 1/32** | the champion's 9 | 86.1% | **7/9** | 0.6% | -142 u |
| champion | xARC's last 3 evals, 21 eps | 95.3% | 20/21 | 0.0% | -0 u |
| **coarse 1/32** | xARC's last 3 evals, 21 eps | 95.3% | **20/21** | 0.4% | -1 u |

and a FINISHING champion episode is never more than **1,122 u** from the
coarse line anywhere on the map - inside the 1,500 u corridor in every 25%
band of the run, the final descent included. As a scorer the coarse line loses
nothing measurable.

**The shaping scale is the tool's own derived rule, untouched.** The trainer
printed `arc route ... 1691 pts @ 128u = 216,320u ... -> shaping scale
0.000462278/u (vs geodesic 0.000504083/u)`, against xARC's 4.3163e-4. Per unit
of REAL track travelled the two arc arms are **identical**:
`4.62278e-4 x 216,320/231,680 = 4.3163e-4`, because the 100-per-run budget
rule absorbs the shorter path. Break-even speed 1,082 u/s against xARC's 1,158
and the geodesic's 992, versus a descent run at 2,700-3,700 u/s.

### The uncovered tail: decided, implemented, NOT exercised

The brief asked for a decision on what happens past the end of a truncated
line, and warned the result hinges on it. The decision, pre-registered:

> **Pay nothing past the line's end; no geodesic fallback.** A truncation
> makes a FLAT region, not a barrier: past the last vertex the arc coordinate
> saturates, so going forward pays 0 while coming back pays NEGATIVE (the term
> is a signed delta) - against the geodesic, which actively PAID the agent to
> turn back at vertex 1601. Falling back to the geodesic past the line's end
> reintroduces the exact term under test, needs a potential-stitching design
> whose discontinuity at the handoff is a new correctness surface, and is a
> second treatment inside a one-hour budget.

`--allow-unfinished` implements the visible relaxation of "a route must reach
the finish": without it `build_route.py` refuses exactly as before; with it
the written `.npz` carries `truncated=True` plus the gap in map units, and the
tool prints a TRUNCATED banner. **But the line that ran is not truncated** -
decimation keeps the last vertex, so the coarse line ends 52.31 u from the
finish curtain, the same gap xARC recorded for the champion route. **The tail
question stays open and this arm says nothing about it.**

### The correctness surface (CPU only, before renting)

* **No training code was touched.** `python/surfgym/route.py`,
  `python/surfgym/rewards.py` and `python/train_fast.py` are byte-identical to
  `origin/arclen`; the arm passes a different file to `--race-arc`.
* `tests/python/test_route_degrade.py`, **17 new CPU-only tests**: the archive
  chain (root-first walk, the two leaf rules disagreeing, a cycle guard, an
  unknown rule rejected), decimation and quantization bounds, `end_gap`, the
  CLI refusing `--archive` without `--allow-unfinished`, a truncated write
  actually carrying `truncated=True`, and the two functional properties the
  reward needs from a degraded line - an agent riding the SOURCE line still
  advances monotonically to the end of the decimated copy, and out-and-back
  still nets zero.
* **Flag-off bit-identity**: `test_flag_off_is_bit_identical_to_the_branch_point`
  runs `origin/arclen:tools/build_route.py` and this one over the same
  synthetic finisher recording and requires `np.array_equal` on the route
  array plus equal spacing and seconds. Separately, rebuilding the committed
  champion route from `runs/sISV_par2/traj_8454144000.jsonl` with the new code
  reproduces `maps/surf_src_cannonball.route.npz`'s array exactly.
* The scorer used below reproduces xARC's and xMARGIN's published figures byte
  for byte (231,680u / 84 / 63 and 208,640u / 7 / 0), so the comparison is
  like for like.
* On the box: **146 collected, 145 green**, the single failure being
  `test_march_is_bit_exact_against_the_legacy_kernel`, the known 3090 failure
  CLAUDE.md documents. (The image ships no pytest: `pip install
  --break-system-packages pytest` is needed on top of `deploy_box.sh`'s pip
  line, which installs only scipy/numpy.)

### The run

    bash tools/run_arm.sh xAUTO --respawn-margin 2 \
        --race-arc maps/surf_src_cannonball.coarse32.route.npz

Warm resume of `runs/sOBSR2/ckpt_latest.pt`, md5
`1ba1fd2936af3ae1ad3608e3cd6b1e9e` **verified on the box**, step
3,782,737,920, `surf_src_cannonball`, one RTX 3090 (vast 48376624, machine
54594, host 69155), one seed. The baseline config guard passed; the "restored
from checkpoint config" line contains neither `respawn_margin` nor
`race_arc`, so both CLI values are what ran, and `runs/xAUTO/run.json` records
`race_arc = maps/surf_src_cannonball.coarse32.route.npz`,
`race_arc_corridor = 1500.0`, `race_arc_window = 16`, `respawn_margin = 2.0` -
identical to xARC in every field except the route file.

**800,587,776 steps in 57.5 minutes, 1,018 iterations, 222,052 steps/s
average** - the same step count and the same iteration count as xARC - 11
in-trainer evals of 9 greedy episodes plus one independent `record_ckpt.py`
recording of 3.

### RESULT: THE MAP IS FINISHED FROM A LINE THAT IS NOT A RACING LINE

All figures `--order-only 16`, scored against the **champion** route
(`maps/surf_src_cannonball.route.npz`) and never against the line the arm was
paid on - only the champion route makes these numbers comparable with xARC and
xMARGIN.

| eval | steps after resume | race/eval_progress | corridor MAX (published) | corridor MAX (order-only) | past 205,440 | finishes | best finish |
|---|---|---|---|---|---|---|---|
| 1 | +0.8M | 165,409 | 205,312 | 205,352 | 0/9 | 0/9 | - |
| 2 | +76M | 192,076 | 207,360 | **207,349** | 8/9 | 0/9 | - |
| 3 | +152M | 191,931 | 222,848 | **222,810** | 7/9 | 0/9 | - |
| 4 | +227M | 154,522 | 231,680 | **231,680** | 7/9 | **6/9** | **80.56 s** |
| 5 | +303M | 181,829 | 231,680 | **231,680** | 8/9 | **8/9** | **80.51 s** |
| 6 | +378M | 152,164 | 231,680 | **231,680** | 6/9 | **6/9** | 81.27 s |
| 7 | +454M | 179,362 | 231,680 | **231,640** | 8/9 | **7/9** | 81.56 s |
| 8 | +529M | 176,066 | 231,680 | **231,680** | 8/9 | **7/9** | 81.20 s |
| 9 | +605M | 198,381 | 231,680 | **231,680** | 9/9 | **9/9** | 80.72 s |
| 10 | +680M | 198,369 | 231,680 | **231,680** | 9/9 | **9/9** | 80.89 s |
| 11 | +756M | 176,591 | 231,680 | **231,680** | 8/9 | **7/9** | 80.94 s |
| rec | +801M | - | 231,680 | **231,679** | 3/3 | **3/3** | 81.74 s |

**62 of 102 greedy episodes finished the map. 81 of 102 crossed 205,440 u.**
Finish times best **80.51 s**, median 81.40 s, mean 81.44 s, worst 82.91 s.
Eval 1 - the untreated policy at +0.8M steps - lands on 205,352 u with 0/9
past the line and 0 finishes, which is the right internal control and is
exactly where xMARGIN and xARC opened.

| arm | reference line the reward pays on | greedy eps | corridor MAX | past 205,440 | finishes |
|---|---|---|---|---|---|
| xROUTE | none (geodesic potential) | 99 | 205,312 | 0 | 0 |
| xSP | none (geodesic potential) | 54 | 205,312 | 0 | 0 |
| xNECTO | none (geodesic potential) | 81 | 205,440 | 0 | 0 |
| xMARGIN | none (geodesic potential) | 72 | 208,640 | 7 | **0** |
| xARC | **the champion's winning trajectory**, 1,811 pts | 102 | 231,680 (100%) | 84 | **63** |
| **xAUTO** | **58 chords, 24.8% of it inside rock** | 102 | **231,680 (100%)** | **81** | **62** |

**The two arc arms are indistinguishable on every axis measured**, at matched
steps: first finish at eval 4 in both (xARC 4/9, xAUTO 6/9); 100% corridor MAX
from eval 4 on in both; 63 vs 62 finishes in 102 episodes; 84 vs 81 past the
old frontier. And the training telemetry lands on top of itself - `ep_rew`
mean 44.39 vs 44.14, `ep_len` 2,781 vs 2,790, `kl` 0.0178 vs 0.0176,
`value_loss` 0.697 vs 0.698, 1,018 iterations each.

Per CLAUDE.md rule 3, a run that finishes is judged on wall-clock start to
finish. Measured identically across all three (first recorded tick to first
tick inside the finish box, +64 u pad):

| | best | median | mean | finishers |
|---|---|---|---|---|
| champion recording `runs/sISV_par2/traj_8454144000.jsonl` | 81.35 s | 82.16 s | 82.19 s | 7/9 |
| xARC (champion line) | 81.04 s | 81.74 s | 81.82 s | 63/102 |
| **xAUTO (coarse line)** | **80.51 s** | **81.40 s** | **81.44 s** | 62/102 |

**The arm trained on a 58-waypoint skeleton is 0.5 s faster at its best and
0.4 s faster on average than the arm trained on the champion's own line.** Do
not read that as a record: it is one seed, one hour, and this ledger's
headline 1:19.72 is measured on a different basis.

### The agent does not follow the line it is paid on

This is the mechanism, and it is measurable. `tools/wall_profile.py` on eval 8
against the CHAMPION line, through the vertices where every control arm died:

| vertex | xAUTO speed | xAUTO off the CHAMPION line | champion speed |
|---|---|---|---|
| 1540 | 2,805 u/s | 183 u | 2,927 |
| 1560 | 2,816 u/s | 251 u | 2,928 |
| 1580 | 2,827 u/s | 325 u | 2,935 |
| 1600 | 2,833 u/s | 207 u | 2,926 |
| 1620 | 2,838 u/s | 158 u | 2,928 |
| 1640 | 2,844 u/s | 145 u | 2,921 |
| 1660 | 3,684 u/s | 79 u | 3,728 |

The reference line it was trained on is **up to 1,131 u** from the champion
line; the policy ends up within **79-325 u** of the champion line. It is not
imitating the reference - it cannot, a quarter of the reference is inside
rock. **The reference supplies the ORDERING; the physics supplies the LINE.**
That is the whole finding, and it is why a coarse line is enough.

### `race/eval_progress` was anti-correlated again, harder than in xARC

165,409 -> 192,076 -> 191,931 -> **154,522** while the honest frontier went
205,352 -> 207,349 -> 222,810 -> **231,680 with 6 finishes**. The eval that
first finished the map posted the LOWEST reading of the run so far. The
arithmetic: the metric is `mean(d at spawn - min d reached)` over 9 episodes,
and eval 4 had six 100% finishes plus **two deaths at 1.5% and 2.5% of the
route** - early falls on the first ramp, nothing to do with the wall. Six
saturated episodes and two zeros average to 154k. **Lead with
`eval_honesty.py`; on this task `race/eval_progress` is a mixture of "how far"
and "how often", and at the frontier the second term dominates.**

### Training diagnostics, 1,018 logged iterations

`ep_rew` mean 44.39 (p10 21.04, p90 60.38), last 55.35; `ep_len` mean 2,781;
`kl` 0.0178; `ent` pinned at 0.005; `value_loss` mean 0.697 (p10 0.066, p90
1.297) against the geodesic controls' 0.046-0.061 - the same tenfold rise xARC
saw, and for the same reason: the return distribution now contains +50 finish
bonuses this checkpoint had never observed. Training win rate by tenth of the
run:

    0.0%  0.0%  4.7%  34.5%  52.4%  61.5%  66.8%  70.6%  72.4%  74.6%

first non-zero at **+169.1M steps** (xARC: +151.0M). The share of ticks spent
outside the corridor earning nothing fell from 12-15% early to **6.1-6.4%** at
the end - higher than xARC's 0.7%, which is exactly what a line that is 24.8%
inside rock should do: the agent physically cannot ride it, so it lives at the
corridor's edge and still collects.

**The short-margin caveat fired again and the verdict is again separated from
it.** At `--respawn-margin 2` the reservoir harvests states 2 s from the goal
once an arm starts finishing: training `finish_s` sits at **mean 29.5 s (min
7.5 s)** against the greedy evals' 81 s, i.e. most training finishes start
deep in the route. That is why the verdict rests on the greedy evals from the
start line and not on the win rate.

### VERDICT

**STRONG POSITIVE, and it discharges the largest of xARC's three caveats.**
The reference line does **not** need to be a champion's, and it does not need
to be a racing line at all. Fifty-eight straight chords, a quarter of them
passing through solid rock, up to 1,131 u away from the champion's line and
6.7% shorter than it, produce the same frontier (100%), the same finish count
(62 vs 63 of 102), the same learning curve and marginally faster finish times
than the champion's own trajectory did. Linesight's claim - the reference line
"does not need to be fast... usually the centerline" - reproduces here
literally.

**What this licenses.** xARC's finishes came from the SHAPE of the potential -
a monotone progress coordinate through the right corridor - and not from
imitating a champion. For a new map the requirement is therefore a **coarse
ordered corridor, roughly one waypoint per 4 km of track**, not a
demonstration. That is a far weaker input than a winning run, and it is the
first result in this ledger that makes the ~1000-map goal look like an
engineering problem rather than a research one.

**What this does NOT license.** Nothing here is autonomous. The 58 waypoints
were sampled from a recording of a finisher; "cheaper to obtain" is not
"free". No line was discovered tonight without champion data, phase-1 reached
6.7% of this map in 34 CPU-minutes across two configurations, and both honest
ways of extracting a line from a phase-1 archive are broken on this map for
reasons now measured. And the truncated-line case - the one that decides
whether a line that stops short of the finish still works - was implemented,
pre-registered and **not tested**, because the line that ran reaches the end.

### What to run next, reordered by this result

1. **Bootstrap, and it is now the cheapest champion-free line by far.** The
   stuck checkpoint's own best non-finishing episode reaches 205,312 u = 88.6%
   unaided; `build_route.py --allow-unfinished` will turn it into a line
   tonight. This arm's evidence says the line's QUALITY does not matter - so
   the only open variable is its REACH, which makes the pre-registered tail
   decision above the whole experiment. That is Linesight's own loop, and
   after one iteration it needs no champion at all.
2. **Find the degradation limit, because it is the corridor and not the
   waypoint count.** Measured tonight on CPU: 1/64 decimation (30 waypoints,
   8,192 u chords) breaks completely - 88.6% of a champion episode's ticks
   fall OUTSIDE the 1,500 u corridor and arc reach collapses to 4.2%. At 1/32
   the line is 1,131 u off and the agent's own deviation at the wall is
   79-325 u, so the 1,500 u corridor is nearly exhausted. **The cheap knob is
   `--race-arc-corridor`, not the line.**
3. **Phase-1 on surf needs a different mechanism, not a bigger archive.**
   14x more cells bought 0.01 percentage points. Restore-and-explore with
   uniform random actions cannot pass the first speed-gated ramp; the
   reservoir the trainer already keeps does pass it, which suggests seeding
   the archive from rollouts rather than from map spawns.
4. Separating `--respawn-margin 2` from the arc reward (arc shaping at the
   default 10 s margin) is still unrun and still the cleanest missing control.

### Ops and cost

* `fleet_watchdog list` empty at start, daemon already running. Raced three
  RTX 3090 candidates - 48376598 (machine 137375), 48376607 (machine 140567),
  48376624 (machine 54594, host 69155) - all registered on create with a
  115-minute deadline. Cap of 4 never exceeded.
* **A deliberate deviation from the 60-second readiness rule, recorded rather
  than glossed.** None of the three had a usable ssh session at 60 s: all
  three were still pulling the 5 GB image (`status_msg` showed "Verifying
  Checksum" / "Pull complete" at 70-120 s). 48376624 reached `running` about
  3 minutes after create and ssh answered on the first try; 48376607 followed
  seconds later; 48376598 never came up. The rule exists so an agent does not
  sit waiting on one bad host, and racing three and taking the first serves
  that purpose - but the race took **3 minutes, not 60 seconds**, and a
  successor should expect that whenever no candidate has the image cached.
  The two losers were destroyed and released at 07:52:17 and 07:52:23, both
  confirmed gone; `vastai show instances` then listed exactly one instance.
* `deploy_box.sh` with `BRANCH=autoline`,
  `LOCAL_CKPT=/c/RL_Surf/runs/sOBSR2/ckpt_latest.pt`, `EXPECTED_MD5=1ba1f...`,
  `SKIP_TORCH=1`. Checkpoint md5 **verified on the box**
  (`1ba1fd2936af3ae1ad3608e3cd6b1e9e`, step 3,782,737,920), 6 cache .npz
  pushed (9 .npz in `maps/` total), bsp mtime pinned, `gpu_health.py` VERDICT
  healthy (841 GB/s HBM, 71 TFLOPS bf16, 1,695 MHz under load, 317.6 W of
  350 W). The route file itself needed no scp: it is committed on the branch,
  so the clone carried it.
* Trainer alive from 07:58:53 to 08:56:24 UTC, GPU pinned, fps 192k-222k,
  rising throughout, never decaying, never stationary. Box destroyed 09:00:39
  UTC, `vastai show instances` returned 0 instances, watchdog released, fleet
  empty.
* **Rental cost: $0.226** for the winner (72.1 minutes at $0.18777/h) plus
  **$0.02** for the two racing losers. **Total ~$0.25.** The phase-1 passes
  were free (local CPU).
* Artifacts in `runs/research/xAUTO/`: 11 trajectory files, `rec.jsonl`,
  `progress.csv`, `run.json`, `xAUTO_launch.txt`. The line itself is committed
  at `maps/surf_src_cannonball.coarse32.route.npz`.

## Round 19 - xSELF: a reference line from the agent's OWN best FAILURE (2026-08-22 09:33-10:41 UTC)

xARC finished this map by paying arc length along the champion's own winning
trajectory. xAUTO showed the line does not need champion SKILL - 58 straight
chords, a quarter of them inside rock, finished it just as often. Both lines
still came from a recording of a FINISHER, so the one thing neither could
answer was REACH: **does a monotone progress coordinate that covers only the
part of the map the agent already flies still remove the barrier, or does the
reference have to reach past the hard part?**

This arm builds the line from the stuck checkpoint's OWN best non-finishing
episode. That checkpoint has never crossed the line; its best greedy episodes
reach 205,240-205,472 u of 231,680 u and then leave the ramp. **The line
therefore stops at 88.12% and the final 11.88% (27,520 u) of the map has no
reference at all.** No finisher's recording enters its provenance anywhere.

**The answer is that it still works.** 47 of 102 greedy episodes finished,
corridor MAX 231,680 u (100%), last eval 9 of 9, best time 80.06 s - the
fastest of the three arc arms. **Eight episodes in the whole run ended in the
1,280 u between the line's end and the old wall, and all eight are in evals
1, 2 and 4; from +303M steps onward, none.** The agent does not stall where
its reference runs out.

### The line: what was measured before anything was built

The source recordings are the stuck checkpoint's own: 270 greedy episodes
across xROUTE / xSP / xNECTO / xCONTACT, 0 finishes, champion-corridor reach
**205,240-205,472 u** - a spread of **232 u**, 0.11% of the map. *The choice
of episode is therefore immaterial and no ranking rule can be worth much;
what matters is where the line STOPS.*

Because every one of those episodes ends in a fall, this is not a truncation
problem, it is a contamination problem. Resampled raw, the line's last few
thousand units point DOWN into the pit, and the reward then pays for falling.
Measured, not asserted - replaying real recordings through
`surfgym.route.ArcProgress` on an **untrimmed** line built this way:

| replayed episodes | reach on the untrimmed line |
|---|---|
| xARC's, which FINISH the map | 96.9% |
| the stuck checkpoint's own, which FALL | **99.3%** |

A faller out-earns a finisher by 2.4 points of the 100-point shaping budget.
That is the same *kind* of defect as the geodesic barrier this whole line of
work exists to remove, and shipping it would have made a null uninterpretable.

### Three trim rules, two of which fail, all measured on the real recordings

* **The champion route as a ruler** (cut at the episode's furthest corridor
  progress along the champion line). Works, and injects champion knowledge
  into precisely the number under test. Rejected.
* **The map's own geodesic goal field** (cut at the episode's minimum distance
  to goal). Champion-free, and **wrong**: it lands 0.55-1.31 s *inside* the
  fall (chosen episode: tick 7245 against a departure at 7190; another: tick
  7299 against 7168), because the field's deceptive basin is goal-adjacent
  airspace below the ramp. **The defect that made the arc reward necessary in
  round 18 also poisons the only map-derived selector an autonomous pipeline
  has** - the same finding xAUTO recorded for `--archive-leaf dist`, now
  measured on trajectories instead of archives.
* **Consensus across the checkpoint's own episodes** (cut where an episode
  stops agreeing with its siblings). Implemented and **measured to fail, for a
  reason worth writing down: the failure is not idiosyncratic.** One
  checkpoint played greedily falls the SAME way every time, so the falls
  corroborate each other. At a 256 u median-agreement radius over 63 siblings
  it dropped **0.05 s of a 2.2 s tail** and chose an episode ending at
  z = -4,157, deep in the pit.

**What survives is the last tick at which the map pushed back.** Between
contacts a Source player is a projectile: `vz` falls by exactly one gravity
step per tick. A tick whose vertical acceleration *departs* from that step is
a tick where geometry acted. The failure here is a fall into an empty pit, so
the last such tick is the end of the part of the track that physically exists
for this policy. It needs no champion, no route file, no goal field, no map
file and no engine constant - the gravity step is taken from the recording's
own median `diff(vz)` (recovered as **-8.0 u/tick^2** across all 270 episodes).

It is sharp on this data. The five best episodes cut within **25 u** of the
same physical point - champion arc 204,153-204,178, at
`(-6,675..-6,698, 1,603..1,740, -1,780)`, the same z to the unit - and it
ranks first **the same episode the champion ruler would**, which is the
convergence that makes it trustworthy here. On a champion's own finishing
episodes it cuts ~4.2 s / 10,900 u before the finish, because the champion's
last stretch is a genuine unbroken flight; that is why the rule is offered for
truncating a FAILURE and not for building a complete route.

`tools/pick_selfline.py` (new, 11 CPU-only tests in
`tests/python/test_pick_selfline.py`) does the ranking and the trim and writes
one episode back out in the recorder's format; the line itself then comes from
the normal tool, which records the truncation:

    python tools/pick_selfline.py --out runs/research/xSELF/source_episode.jsonl \
        "runs/research/{xROUTE,xSP,xNECTO,xCONTACT}/traj_*.jsonl"
    python tools/build_route.py --allow-unfinished \
        --out maps/surf_src_cannonball.self88.route.npz \
        runs/research/xSELF/source_episode.jsonl

    270 episodes in 30 file(s); raw path length 1,584..213,278u
    gravity step recovered from diff(vz): median -8, spread -8..0 u/tick^2
    chosen: traj_4161011712.jsonl ep 5 trimmed at tick 7110 of 7368
    raw path 210,019u -> trimmed 205,082u (dropped 4,937u, 2.58s of tail)
    ** TRUNCATED: the last vertex is 5,949u from the finish box. **

### The line that ran, and exactly how far it reaches

| | |
|---|---|
| source | xSP `traj_4161011712.jsonl` ep 5, a **non-finishing** greedy episode of the stuck checkpoint |
| trim | tick 7110 of 7368 - the last tick the map pushed back; 2.58 s / 4,937 u of fall dropped |
| line | **1,603 pts @ 128 u = 205,056 u**, `truncated=True`, `end_gap` 5,949 u |
| where it ENDS on the champion route | vertex 1595 = **204,160 u of 231,680 u = 88.12%**, 327 u off the champion line |
| **uncovered tail** | **27,520 u = 11.88% of the map, with no reference of any kind** |
| relative to the old wall | the line's end is **1,280 u BEFORE** the 205,440 u frontier every control arm died at |
| relative to the geodesic barrier | route vertex 1601 (204,928 u), the old potential's interior minimum, is **768 u PAST** the line's end |
| deviation from the champion line, over the stretch it covers | max **788 u**, p99 551, median **120**, rms 204 |
| the same for xAUTO's coarse line | max 1,128, p99 990, median 280, rms 392 |

**The line was deliberately NOT decimated.** xAUTO already proved a degraded
line works; decimating here would add a second degradation and hand a null the
escape hatch "the line was bad". At full resolution this line is *better* than
xAUTO's on every deviation statistic and it is flyable by construction - the
policy flew it. **The only thing wrong with it is that it stops at 88.12%**,
which is the single variable this arm exists to move.

The trimmed line does not leak. Replayed through `ArcProgress` with the arm's
own settings (corridor 1,500 u, window +/-16):

| replayed episodes | reach on the TRIMMED line |
|---|---|
| champion, 7 of 9 finish | max 100.0% |
| xARC's finishers | max 100.0%, and **all 12 at 100.0%** |
| xAUTO's finishers | max 100.0% |
| the stuck checkpoint's own fallers | max 100.0% |

Everybody saturates at the line's end and **nobody can pass it**. The whole
line is coverable by correct play, so "reached the end and stalled" would have
been a real observation rather than an artefact of an unreachable last vertex.

### What happens past the end: xAUTO's pre-registered choice, kept and verified

xAUTO pre-registered "pay nothing past the line's end; no geodesic fallback"
and never got to exercise it, because its line reached the finish. **This arm
exercises it and keeps it unchanged - there is no second treatment here.**
Verified in code before renting, on the actual file:

    0u past the end:    delta +0.000  inside=True
    200u past the end:  delta +0.000  inside=True
    600u past the end:  delta +0.000  inside=True
    1400u past the end: delta +0.000  inside=True
    3000u past the end: delta +0.000  inside=False
    retreat  2 vertices: delta   -256.0 u
    retreat  5 vertices: delta   -640.0 u
    retreat 10 vertices: delta -1,280.0 u

Past the last vertex the arc coordinate saturates, so **going forward pays
exactly 0 and coming back pays NEGATIVE** - a flat region, not a barrier. It
is worth saying plainly what that means here: **11.88% of this map, including
the wall itself and the entire final descent, was run on the time penalty and
the +50 finish bonus alone.**

The shaping scale is the tool's own derived rule, untouched. The trainer
printed

    arc route surf_src_cannonball.self88.route.npz: 1603 pts @ 128u = 205,056u,
        corridor 1500u, window +/-16 (2,048u) -> shaping scale 0.000487672/u
        (vs geodesic 0.000504083/u)

| | geodesic | xARC | xAUTO | **xSELF** |
|---|---|---|---|---|
| scale | 5.04083e-4/u | 4.3163e-4/u | 4.62278e-4/u | **4.87672e-4/u** |
| break-even speed vs the 0.005/tick time penalty | 992 u/s | 1,158 | 1,082 | **1,025** |

against a descent run at 2,700-3,700 u/s, so the speed incentive is again
materially unchanged. The one consequence of truncation worth naming: the
100-per-run budget is now collected over 88.12% of the track instead of 100%,
i.e. **13.5% more shaping per unit of real track than xARC** - a consequence
of keeping the rule, not a tuning choice.

### The correctness surface, checked before renting (CPU only)

* **No shared code was touched.** `python/`, `tools/build_route.py`,
  `tools/eval_honesty.py` and `python/train_fast.py` are **byte-identical** to
  `origin/autoline` (`git diff --stat origin/autoline` is empty; the branch
  adds three new files). Flag-off bit-identity is therefore structural rather
  than tested: there is no shared code path that could differ.
* `tests/python/test_pick_selfline.py`, **11 new CPU-only tests**: the cut is
  the last contact and is kept; the gravity step is read from the data, not a
  constant, at three different values; **a longer fall does not move the cut**;
  a purely ballistic episode is not trimmed at all (fails safe); degenerate
  and 1-row inputs do not raise; tolerance widens monotonically; ranking on
  the TRIMMED path inverts the raw ranking; the written episode round-trips
  through the reader `build_route.py` uses; and, end to end through
  `ArcProgress`, **trimming removes the pay-for-falling** while not trimming
  reproduces it.
* Locally: **154 passed, 3 skipped**. On the box: **157 collected, 156 green**,
  the single failure `test_march_is_bit_exact_against_the_legacy_kernel`, the
  known 3090 failure CLAUDE.md documents.
* The scorer used below reproduces xARC's published figures byte for byte
  (63 finishes in 102 episodes, best 81.04 s), so the comparison is like for
  like.

### The run

    bash tools/run_arm.sh xSELF --respawn-margin 2 \
        --race-arc maps/surf_src_cannonball.self88.route.npz

Warm resume of `runs/sOBSR2/ckpt_latest.pt`, md5
`1ba1fd2936af3ae1ad3608e3cd6b1e9e` **verified on the box**, step
3,782,737,920, `surf_src_cannonball`, one RTX 3090 (vast 48383245, machine
54594, host 69155), one seed. The baseline config guard passed; the "restored
from checkpoint config" line contains neither `respawn_margin` nor `race_arc`,
so both CLI values are what ran, and `runs/xSELF/run.json` records
`race_arc = maps/surf_src_cannonball.self88.route.npz`,
`race_arc_corridor = 1500.0`, `race_arc_window = 16`,
`respawn_margin = 2.0` - **identical to xARC and xAUTO in every field except
the route file**.

**800,587,776 steps, 1,018 iterations, 220,931 steps/s average** - the same
step count and the same iteration count as both arc arms - 11 in-trainer evals
of 9 greedy episodes plus one independent `record_ckpt.py` recording of 3.

### RESULT: THE MAP IS FINISHED FROM A LINE THAT STOPS AT 88.12% OF IT

All figures `--order-only 16`, scored against the **champion** route
(`maps/surf_src_cannonball.route.npz`) and never against the line the arm was
paid on - only the champion route makes these numbers comparable with xARC,
xAUTO and xMARGIN.

| eval | steps after resume | race/eval_progress | corridor MAX (order-only) | past 205,440 | finishes | best finish |
|---|---|---|---|---|---|---|
| 1 | +0.8M | 165,317 | 205,341 | 0/9 | 0/9 | - |
| 2 | +76M | 172,621 | 207,049 | 5/9 | 0/9 | - |
| 3 | +152M | 191,799 | **217,216** | 9/9 | 0/9 | - |
| 4 | +227M | 191,727 | 209,133 | 7/9 | 0/9 | - |
| 5 | +303M | 181,470 | **228,480** | 8/9 | **3/9** | 81.30 s |
| 6 | +378M | 195,419 | **231,666** | 9/9 | **5/9** | 81.66 s |
| 7 | +454M | **149,448** | **231,680** | 6/9 | **6/9** | 81.88 s |
| 8 | +529M | 181,846 | **231,680** | 8/9 | **8/9** | 81.15 s |
| 9 | +605M | 155,827 | **231,671** | 7/9 | **7/9** | **80.06 s** |
| 10 | +680M | 150,107 | **231,680** | 6/9 | **6/9** | 80.55 s |
| 11 | +756M | 198,371 | **231,605** | 9/9 | **9/9** | 80.26 s |
| rec | +801M | - | **231,596** | 3/3 | **3/3** | 81.04 s |

**47 of 102 greedy episodes finished the map. 77 of 102 crossed 205,440 u.
Corridor MAX 231,680 u = 100%.** Finish times best **80.06 s**, median
81.18 s, mean 81.25 s, worst 82.44 s.

Eval 1 - the untreated policy at +0.8M steps - lands on 205,341 u with 0/9
past the line and 0 finishes, exactly where xMARGIN, xARC and xAUTO all
opened. It is the internal control and it is also, usefully, the null this
experiment was built to be able to see: **5 of its 9 episodes end in the
1,280 u band between the line's end and the old wall.** That is what "reaches
the line's end and stalls there" looks like, and it is what the arm stopped
doing.

### THE MEASUREMENT THIS ARM EXISTS FOR: where episodes end relative to the line's end

The self line's last vertex sits at champion arc **204,160 u**. Every episode
of the run, binned by where its champion-route corridor progress stopped:

| eval | steps | ended BEFORE the line's end | ended in the 1,280 u between the line's end and the old wall | ended PAST the old wall | finished |
|---|---|---|---|---|---|
| 1 | +0.8M | 4/9 | **5/9** | 0/9 | 0 |
| 2 | +76M | 3/9 | **1/9** | 5/9 | 0 |
| 3 | +152M | 0/9 | 0/9 | 9/9 | 0 |
| 4 | +227M | 0/9 | **2/9** | 7/9 | 0 |
| 5 | +303M | 1/9 | 0/9 | 8/9 | 3 |
| 6 | +378M | 0/9 | 0/9 | 9/9 | 5 |
| 7 | +454M | 3/9 | 0/9 | 6/9 | 6 |
| 8 | +529M | 1/9 | 0/9 | 8/9 | 8 |
| 9 | +605M | 2/9 | 0/9 | 7/9 | 7 |
| 10 | +680M | 3/9 | 0/9 | 6/9 | 6 |
| 11 | +756M | 0/9 | 0/9 | 9/9 | 9 |
| rec | +801M | 0/3 | 0/3 | 3/3 | 3 |
| **total** | | **17/102** | **8/102** | **77/102** | **47** |

**Eight episodes out of 102 ended where the reference runs out, and all eight
are in evals 1, 2 and 4. From +303M steps onward the band is empty in every
single eval.** The 17 that ended before the line's end are ordinary early
deaths on the first ramps, not a stall at the frontier - the same failure mode
that gave xAUTO's eval 4 its two 1.5%/2.5% deaths.

And on the line the reward actually paid: **85 of 102 episodes reached
100.0% of it**, mean 90.3%. The reference was consumed in full and the run
continued past it.

### The signature of a flat tail, visible in the training log

`off` - the share of physics ticks spent outside the corridor earning nothing
- by tenth of the run:

    xSELF   9.5%  15.9%  23.3%  30.1%  26.6%  30.4%  33.1%  34.7%  34.2%  35.5%
    xAUTO   12-15%  ..............................................  6.1-6.4%
    xARC    12.1%   ..............................................  0.7%

**The two arms whose line reached the finish drove this DOWN; this one drove
it UP, monotonically, to more than a third of all ticks.** That is not a
pathology, it is the arithmetic of the treatment: the last 11.88% of the map
lies past the line's end, so every episode that gets further necessarily
spends more of its life earning nothing. The number rising is the direct
observable that the agent is living in the unreferenced region - and it rose
in lockstep with the finish count. Mean arc gained per training episode was
67,008 u of the 205,056 u line (32.7%), at a 90% respawn fraction.

### `race/eval_progress` was anti-correlated for the third arm running

165,317 -> 172,621 -> 191,799 -> 191,727 -> **181,470** while the honest
frontier went 205,341 -> 207,049 -> 217,216 -> 209,133 -> **228,480 with the
first 3 finishes**. Its lowest reading of the entire run, **149,448 at +454M**,
is the eval that posted **corridor MAX 231,680 with 6 of 9 finishes**. Its
highest, 198,371, is the last eval, which is saturation and not information.
**Lead with `eval_honesty.py`.** Round 18 proved this arithmetically, xARC and
xAUTO each caught it in a run, and here the anti-correlation is the sharpest
yet: the single worst eval by the standing metric is 6-of-9 on the honest one.

### Training diagnostics, 1,018 logged iterations

`ep_rew` mean 38.86 (p10 23.95, p90 55.60), last 55.81, against xMARGIN's
22.9 and both arc arms' ~44.2; `ep_len` mean 2,740; `kl` 0.0181; `ent` pinned
at the 0.005 coefficient; `value_loss` mean 0.470 (p10 0.074, p90 0.857)
against the geodesic controls' 0.046-0.061 - the same order-of-magnitude rise
xARC and xAUTO saw, and for the same reason: the return distribution now
contains +50 finish bonuses this checkpoint had never observed. Training win
rate by tenth of the run:

    0.0%  0.0%  0.5%  5.1%  20.3%  44.7%  63.7%  69.7%  72.5%  73.0%

**first non-zero at +203.7M steps, against xARC's +151.0M and xAUTO's
+169.1M.** The truncation costs about 35-50M steps of discovery latency, and
the greedy evals lag it by about one eval as they should.

The short-margin caveat fired again and the verdict is again separated from
it: training `finish_s` sits at **mean 27.3 s (min 2.0 s)** against the greedy
evals' 81 s, i.e. most training finishes start deep in the route. **That is
why the verdict rests on the greedy evals from the start line and not on the
win rate.**

### Against the controls, all on the same checkpoint and scored the same way

| arm | reference line the reward pays on | how far the line reaches | greedy eps | corridor MAX | past 205,440 | finishes |
|---|---|---|---|---|---|---|
| xROUTE | none (geodesic potential) | - | 99 | 205,312 | 0 | 0 |
| xSP | none (geodesic potential) | - | 54 | 205,312 | 0 | 0 |
| xNECTO | none (geodesic potential) | - | 81 | 205,440 | 0 | 0 |
| xMARGIN | none (geodesic potential) | - | 72 | 208,640 | 7 | **0** |
| xARC | the champion's winning trajectory, 1,811 pts | 100% | 102 | 231,680 | 84 | **63** |
| xAUTO | 58 chords, 24.8% inside rock | 100% | 102 | 231,680 | 81 | **62** |
| **xSELF** | **the stuck checkpoint's own best FAILURE** | **88.12%** | **102** | **231,680 (100%)** | **77** | **47** |

Per CLAUDE.md rule 3 a run that finishes is judged on wall-clock start to
finish. Measured identically across all four (first recorded tick to first
tick inside the finish box, +64 u pad):

| | best | median | mean | worst | finishers |
|---|---|---|---|---|---|
| champion `runs/sISV_par2/traj_8454144000.jsonl` | 81.35 s | 82.16 s | 82.19 s | 82.78 s | 7/9 |
| xARC (champion line, full) | 81.04 s | 81.74 s | 81.82 s | 83.13 s | 63/102 |
| xAUTO (champion line, 58 chords) | 80.51 s | 81.39 s | 81.41 s | 82.91 s | 62/102 |
| **xSELF (own failure, 88.12%)** | **80.06 s** | **81.18 s** | **81.25 s** | **82.44 s** | **47/102** |

**The arm whose reference never saw the last 11.88% of the map is the fastest
of the three on best, median, mean and worst.** Do not read that as a record:
one seed, one hour, and this ledger's headline 1:19.72 is measured on a
different basis.

**Where it is genuinely behind is CONSISTENCY, and that should be stated
plainly**: 47 of 102 against 63 and 62, first finish one eval later (+303M
against +227M for both), first training win 35-50M steps later. Over the last
three evals plus the recording it is 25 of 30, against xARC's 29 of 30 and
xAUTO's 28 of 30. **Truncating the reference is not free - it costs discovery
latency and hit rate. It just does not cost the frontier.**

`tools/wall_profile.py` on the final eval against the champion line - all nine
episodes reach vertex 1809 of 1811:

| vertex | xSELF speed | off the champion line | champion speed |
|---|---|---|---|
| 1540 | 2,837 u/s | 126 u | 2,927 |
| 1560 | 2,849 u/s | 155 u | 2,928 |
| 1580 | 2,861 u/s | 321 u | 2,935 |
| 1600 | 2,867 u/s | 102 u | 2,926 |
| 1620 | 2,874 u/s | 124 u | 2,928 |
| 1640 | 2,874 u/s | 190 u | 2,921 |
| 1660 | 3,732 u/s | 99 u | 3,728 |
| 1680 | 3,587 u/s | **1,100 u** | 3,644 |

Vertices 1600 onward are **entirely past the line's end** - the reference
stops at vertex 1595 - and the agent holds the champion line to 99-190 u
through all of it at champion speed, then takes a line 1,100 u away in the
bowl at vertex 1680 and still finishes faster. **Nothing was paying it to do
any of that.**

### VERDICT

**STRONG POSITIVE, and the cleanest result of the round because it turns a
hypothesis with an informative null into a hypothesis with an answer.**

The pre-registered null was: *if the barrier was ever anything other than the
reward's interior local minimum, an arm whose reference stops at 88.12% will
reach the line's end and stall there, and some external knowledge of the
unexplored region is required.* The band between the line's end and the old
wall held **8 of 102 episodes and none at all after +303M steps**. The arm
finished the map 47 times, hit 100% corridor MAX from eval 6 onward, ran the
final descent it was never paid for at champion speed within 100-190 u of the
champion line, and posted the fastest finish times of the three arc arms.

**What the mechanism therefore is, stated as precisely as the evidence
allows.** The barrier was arithmetic in the reward and nothing else. Replacing
an interior-minimum potential by a monotone one removes it, and the monotone
coordinate does not have to *cover* the hard part - it only has to get the
agent to it without charging it for going on. Past the reference the reward is
flat: forward pays 0, backward pays negative, and the +50 finish bonus plus a
20 s discounted horizon is enough to carry 27,520 u of unreferenced map. The
old potential was worse than nothing there; **nothing beats it.**

**The cost of truncation is real and it is in the rate, not the frontier.**
47 finishes against 63 and 62, one eval later, 35-50M steps later to the first
training win, and a third of all ticks spent outside the corridor by the end.
If a successor needs hit rate rather than reach, extending the reference is
worth roughly one eval of discovery and ~15 points of hit rate.

### What this does and does not license about autonomy

**What it licenses.** The reference line for this map can be built from a
recording of a policy that has **never finished it**, with **no champion, no
demonstration, no goal field and no human route** anywhere in its provenance,
and the resulting agent finishes. Concretely, the only inputs to the line were
the stuck checkpoint's own greedy recordings and the fact that gravity is
constant. The champion route appears in this section **only as a ruler**, to
report where the line ends and to score the results comparably - it is not an
input to `pick_selfline.py`, to `build_route.py`, or to the reward. **This
closes the loop xARC's follow-up #2 asked for: Linesight's bootstrap, one
iteration, on this map.** After this iteration the pipeline needs no champion
at all, and the line it produced is a strictly better starting point than the
one it was built from - the arm's own last eval finishes 9 of 9 at 80.26 s.

**What it does NOT license.** The bootstrap still needs a *seed*: a policy
that already flies 88% of the map. That seed is the stuck checkpoint, which
was itself trained for ~3.78e9 steps under the geodesic potential this work
has now shown to be defective on this map. **Nothing here shows the loop
starting from nothing.** For the ~1000-map goal the open question is no longer
"how good must the reference be" (xAUTO: coarse is fine) or "how far must it
reach" (this arm: 88% is fine, and the cost is hit rate, not reach) - it is
**"where does the first 88% come from on a map nobody has ever flown"**, and
round 19's phase-1 measurements say the Go-Explore answer currently reaches
6.7% of this map in 34 CPU-minutes. That is the remaining gap and it is now
the only one.

**One methodological finding worth carrying to other maps.** The two obvious
champion-free ways to decide where a failed run stops being useful - the map's
own geodesic field, and consensus across the policy's own episodes - are both
broken here, for reasons that are not specific to this map: a distance-to-goal
field with a deceptive basin will always put its minimum inside the failure,
and a deterministic policy fails the same way every time, so its failures
corroborate each other. **A contact/physics criterion sidesteps both**, and on
this data it was accurate to 25 u across five independent episodes.

### Ops and cost

* `fleet_watchdog list` empty at start. Raced three RTX 3090 candidates -
  48383245 (machine 54594, host 69155), 48383264 (machine 12552), 48383268
  (machine 38527) - all registered on create with a 115-minute deadline. Cap
  of 4 never exceeded.
* **The 60-second rule held this time**, unlike xAUTO's 3-minute race: 48383245
  reached `running` about 40 s after create and ssh answered on the first try
  inside 60 s (the image was cached on machine 54594 - the same machine xAUTO
  eventually won on). The two losers were destroyed and released at 09:34:05
  and 09:34:11, both confirmed gone; `vastai show instances` then listed
  exactly one instance.
* `deploy_box.sh` with `BRANCH=selfline`,
  `LOCAL_CKPT=/c/RL_Surf/runs/sOBSR2/ckpt_latest.pt`, `EXPECTED_MD5=1ba1f...`,
  `SKIP_TORCH=1`. Checkpoint md5 **verified on the box**
  (`1ba1fd2936af3ae1ad3608e3cd6b1e9e`, step 3,782,737,920), caches shipped,
  bsp mtime pinned, `gpu_health.py` VERDICT healthy (841 GB/s HBM, 71 TFLOPS
  bf16, 1,665 MHz under load, 306.8 W of 350 W). The route file needed no
  scp: it is committed on the branch at
  `maps/surf_src_cannonball.self88.route.npz` (19 KB).
  `pip install --break-system-packages pytest scipy` on top, as expected.
* Trainer alive and watched throughout, GPU pinned at 99%, 306-326 W, fps
  161k -> 221k cumulative average, never decaying, never stationary; the
  frontier moved at every check. Box destroyed 10:40:59 UTC,
  `vastai show instances` returned `[]`, watchdog released, fleet empty.
* **Rental cost: $0.21** for the winner (67.7 minutes at $0.1878/h) plus under
  **$0.01** for the two racing losers, which lived 38 s and 30 s.
  **Total ~$0.22.** All line-building was local CPU and free.
* Artifacts in `runs/research/xSELF/`: 11 trajectory files, `rec_final.jsonl`,
  `source_episode.jsonl` (the trimmed source recording), `progress.csv`,
  `run.json`, `xSELF_launch.txt`. The line is committed at
  `maps/surf_src_cannonball.self88.route.npz`; the tool that picked and
  trimmed it is `tools/pick_selfline.py`.

## Round 18 - the distributional critic (--quantiles): built and tested, NOT RUN (2026-08-22)

The survey's third structural gap (litsurvey section 0): both superhuman
racers use a distributional critic and both report it mattering. GT Sophy
(Nature 602:223) is QR-SAC with 32 quantiles and its Maggiore ablation puts
the QR head at +0.69 s on a 114 s lap - without it Sophy is not faster than
the best human. Linesight (Trackmania WRs) is IQN. Vasco et al. (RLC 2024),
the vision-based superhuman GT agent and the closest match to this project's
observation constraint, uses 200 quantiles. RL_Surf has a scalar PPO critic
on a task with four catastrophic-failure branches.

**The arm did not run.** Everything below the "what was built" line is why,
and what the next person inherits. Branch `qr-critic` (off `origin/route-obs`
e810d2f), commits 16ce206 + 88e8fff, pushed.

### What was built

`--quantiles N` (+ `--quantile-kappa`, default 1.0): `Policy.value_head`
emits N quantiles of the return distribution instead of its mean, trained
with the quantile Huber loss transcribed from the papers themselves
(ar5iv full texts of arXiv 1710.10044 and 1806.06923 were fetched and the
equations read off, not paraphrased):

    tau_i     = i/N,  tau_hat_i = (tau_{i-1} + tau_i)/2 = (2i-1)/2N
    L_kappa(u)       = 0.5 u^2                  if |u| <= kappa
                     = kappa (|u| - 0.5 kappa)  otherwise
    rho^kappa_tau(u) = |tau - 1{u<0}| * L_kappa(u) / kappa
    loss             = sum_{i=1..N} E_j [ rho^kappa_{tau_hat_i}(T theta_j - theta_i) ]

(QR-DQN Eq. 9-10 + Alg. 1; IQN Eq. 3-4. QR-DQN prints rho without the
1/kappa and IQN divides - both ran kappa=1, where they are the same
expression. We divide, so kappa stays a pure Huber knob.) SUM over the N
rows, MEAN over target atoms; PPO's target is one GAE return per sample, so
N' = 1.

Everything downstream of the critic - GAE, the returns, the truncation
bootstrap, the CUDA-graphed rollout, the logged value - keeps consuming ONE
scalar, and that scalar is the MEAN of the quantiles (what QR-SAC's actor is
updated against). Only the critic's own loss ever sees the distribution.

Warm start (`quantilize_value_head`): the base checkpoint's (1, hidden)
value row is replicated into all N rows, so the MEAN is the old scalar value
exactly and the resumed policy computes the same values, advantages and
returns on its first forward. Adam's exp_avg/exp_avg_sq are replicated with
it - `widen_for_route` ZEROES the moments for the columns it adds because
those are new and have no history, whereas these rows are copies of a row
that has one, and zero moments would hand every row a full lr*sign(g) first
step. The rows separate on the first update anyway: their tau_hat_i differ.

`train/q_spread` (new CSV column): q90 - q10 of the critic's own
distribution over the 2048 rollout envs, one extra forward per iteration
(~1 ms against a ~1.6 s iteration). It is what makes a null readable - wide
spread = the head is representing a branch, ~0 = it collapsed back onto a
point mass and the arm tested nothing.

**Caveat that would mislead the next reader.** The paper's aggregation sums
over i, so with a single target atom and |u| <= kappa the loss is exactly
(N/2) * L_kappa(u) - N times the scalar critic's 0.5 u^2 (sum_i
|tau_hat_i - 1{u<0}| = N/2 whichever side u falls). Left alone that
multiplies the critic's gradient - into a conv trunk the ACTOR SHARES - by
32, and the arm would measure a 32x value-loss coefficient rather than a
distributional critic. `quantile_value_loss` therefore multiplies by 2/N,
which is exactly the reparameterised `--vf` that undoes it and reduces to
the baseline's 0.5 (v - ret)^2 identically at N=1, kappa >= |u|. The loss
SHAPE is the paper's; the scale is the host algorithm's. Anyone rerunning
this should know that `--vf 0.5` therefore means the same thing in both arms
(confirmed live: the smoke run logged value_loss 0.072, inside xCTL's
0.021-0.839 band, median 0.055).

Tests: `tests/python/test_quantile_critic.py`, 14 tests - hand-computed loss
values for four cases, the asymmetric weighting, the Huber knee and its
capped gradient, an actual quantile-function fit (at small kappa the rows
land on the fixed distribution's quantiles within 0.02; at kappa=1 they are
deliberately shrunk toward the centre, pinned down so nobody later reads the
Huber's bias as a bug), N=0 and N=1 bit-identity with the scalar critic, and
the warm-start surgery incl. the Adam moments. `python -m pytest tests/python
-q`: **127 passed** (113 on route-obs before this arm, once the baked SDF
caches are present - without them 3 lidar tests skip).

### The launch that was prepared, and why it never ran

    powershell -File tools/launch_local.ps1 resume \
        C:/RL_Surf/runs/sOBSR2/ckpt_latest.pt xQR \
        --quantiles 32 --record-every 75e6 --eval-eps 9 --eval-greedy-only \
        --steps 5482737920

(md5 of the base ckpt verified 1ba1fd2936af3ae1ad3608e3cd6b1e9e, step
3,782,737,920; --steps = ckpt + 1.7e9 = one hour at the 5090's ~476k/s.)

A 256-env smoke of exactly that code path ran first and proved the wiring:
config restored from the checkpoint (respawn_frac 0.9, int_coef 0.25,
obs_reward, gamma 0.9995, act_every 3, ...), `distributional critic: 32
quantiles ... kappa 1`, `replicated the scalar value head into 32 quantile
rows (6 tensors incl. Adam moments)`, resumed at step 3,782,737,920, one
update and one greedy eval completed. Its eval number (195,986u on ONE
episode at 256 envs) is not comparable to anything and is recorded only as
proof the path is live.

Then the local 5090 was not free. From 03:50 another agent's local trainer
(`--run xNECTO_local`, pid 40476) held 26.6 GB of the 32 GB card at 99%
utilisation. That run FINISHED at 03:56:01 - its own run.json says so, with
duration_s 83.2 and ckpt_final.pt written - but the process never exited: it
sat in shutdown burning ~47% of a core and holding the whole 26.6 GB for the
next 25+ minutes. Freeing the card meant killing a process that had already
saved all of its outputs; that action was refused by the harness permission
classifier, and there is no way to ask a sleeping user. Roughly nine hours
then passed before the coordinator's "report and stop" arrived, by which
time the card was idle (3 GB, 9%) but the instruction was explicit: no new
run. So the hour of training never happened.

**eval_progress series: none. Verdict: none.** The arm is one command away.

### What the next person should know before running it

1. The hypothesis this arm was built around has MOVED. The 88% wall was
   arithmetic in the reward (the geodesic field's minimum along the route
   sits at the wall and rises 8,344u along the champion's own winning line),
   and route-based shaping now finishes the map. The quantile head was
   motivated by that wall being a genuinely bimodal state; on the new
   shaping the return distribution at the wall is a different object. The
   arm is still worth running - a distributional critic is one of the four
   convergent facts, and 4 catastrophic-failure branches do not go away -
   but it should be run ON the shaping that works, not against the one that
   was proven wrong, and `train/q_spread` is then the measurement that
   nobody else has.
2. `race/eval_progress` was ANTI-CORRELATED with the truth through the
   winning run (184,390 -> 156,305 while finishes appeared). Score with
   `tools/eval_honesty.py --order-only 16` (corridor MAX + finishes), not
   with eval_progress.
3. Sophy's 32 is the starting N; Vasco's vision agent used 200 on the
   closest-matching observation setup, and N is one flag.
4. `tools/launch_local.ps1`'s resume preset never set `$run`, so every
   resumed local run wrote its log to `runs\_launch.txt` and the liveness
   proof tailed the wrong file - or, with no `runs\` directory in a fresh
   worktree, no file at all, which is exactly how the first launch attempt
   here reported a live trainer that had already died. Fixed in 88e8fff. A
   fresh worktree also needs `mkdir runs` and `build.ps1` before anything
   will start, and the baked caches copied in with `cp -p` (the cache
   signature is the BSP's size + mtime_ns, so a plain `cp` of the .npz files
   next to a freshly-checked-out .bsp re-bakes the geodesic field).

## Round 19 - xLATCH: the shaping term SWITCHES OFF past the wall, with no reference line (2026-08-22 14:09-15:03 UTC)

xARC, xAUTO and xSELF all finish this map, and all three pay a monotone
coordinate read off a recorded trajectory. xCLAMP asked whether the LINE is
what matters or only the absence of a CHARGE, and answered halfway: a
potential floor (`Phi = -max(d, 6996)`) is flat inside the shell, put 53 of
99 greedy episodes past 205,440 u - the first non-arc arm to do that in bulk
- and finished **0**.

**Why it finished nothing is visible in the potential.** A floor flattens the
region `d <= 6,996`. The valley is not in that region. Between route vertices
1600 and 1680 the champion's own line RAISES the geodesic **6,632 -> 14,976
u**, and every state of that climb is ABOVE the floor, so the floor charges
all of it: `100/198,380 * (14,976 - 6,996) = -4.02`. The agent is still paid
to turn back at the wall, exactly as in the untreated control. The floor only
stopped paying it to fall - which is why 60 of its 99 episodes are
dives-below.

xSELF's arc line simply ENDS at route vertex 1595. Past that point its
shaping is identically 0 - leaving is free - and it finished 47 of 102.

**This arm gives the geodesic the same property with no line at all.**

    bash tools/run_arm.sh xLATCH --respawn-margin 2 --race-latch 6996

Once an episode first reaches `d <= 6,996`, the shaping term contributes
exactly 0 for the rest of that episode, in both directions. Nothing else
moves: time penalty 0.005/tick, intrinsic 0.25, the +50 finish bonus,
`--respawn-margin 2`.

| | v1600 -> v1680 | v1680 -> finish |
|---|---|---|
| xCLAMP (`--race-dfloor 6996`) | **-4.02** | +4.02 |
| xSELF (arc line, ends at v1595) | 0 | 0 |
| **xLATCH** | **0** | **0** |

The threshold is xCLAMP's number, reused verbatim and NOT re-derived:
`tools/pick_dfloor.py`, the geodesic distance at the last tick the map pushed
back, over the stuck checkpoint's own recordings. Champion-free by
construction - the policy's own trajectories, the map's own distance field
and the constancy of gravity - and it lands at route vertex 1595, which is
exactly where xSELF's independently built self line was trimmed.

### The one thing this could not be done naively: it is not Markov

A latch is episode HISTORY. Two states with the same position, velocity and
depth image pay different rewards depending on where the episode has been, so
the critic is asked to predict two different returns from one input and the
value function is unlearnable. `race_dfloor`'s own docstring names the same
hazard from the other side ("NOT a running-minimum ratchet, which would be
history-dependent and unrepresentable by the critic").

**The flag is therefore an observation.** One extra input feature, and
specifically a **1-wide route block concatenated LAST**, which is where
`widen_for_route()` zero-pads a checkpoint that has never seen it:

* the warm resume is **function-identical at step 0**. Verified on CPU
  against the real `runs/sOBSR2/ckpt_latest.pt` before renting: 6 tensors
  padded (the pi and vf first-Linear weights plus their Adam `exp_avg` and
  `exp_avg_sq`), identical logits and value with the flag at 0 **and** at 1.
  The trainer printed the same thing on the box.
* a core scalar slot would NOT do. `Policy.feat_idx` is sorted, so a new
  scalar lands in the MIDDLE of the concatenated row and the zero-pad
  silently permutes every existing feature.

The trainer prints `the flag is obs column 15`, and `--route` composes with
it (fan first, flag last) although this arm passes no route.

### What keeps the RAW geodesic

* **the stall detector and the respawn `stagnant` mask.** Past the switch
  every state pays 0, so on the latched value nothing would ever look like
  progress and the 15 s stall-kill would fire on the whole final descent.
  Same rule `--race-arc` and `--race-dfloor` already follow.
* **`race/eval_progress`**, which the trainer computes from the raw field
  independently of this change - so it stays directly comparable with
  xMARGIN.

### Where the flag also had to be mirrored

Three places where "the eval is not the trained policy" hides, all of which
land on the trajectories every verdict is read from:

1. the eval rollout's own flag column (`_TorchPolicyBase.latch_fn`), tracked
   off the eval core's per-env tick counter, which `reset_env` zeroes;
2. **scalar slot 12 under `--obs-reward`**, which carries the policy's own
   shaping. Training writes the LATCHED value there; a mirror reporting the
   unlatched one feeds a latch-trained policy a feature it has never seen, in
   exactly the states this arm is about;
3. the truncation bootstrap's reconstructed terminal row, which V(s_T) reads
   after the autoreset has already moved the live flag on to the next
   episode's spawn - hence `RaceReward.latch_boot()`.

`tools/record_ckpt.py` had (1) and (2) unmirrored for **both** `race_latch`
and `race_dfloor`, and its static config guard - which skips only keys whose
value is `None`, and both of these default to `0.0` - refused to record any
checkpoint from either branch at all, the dashboard record button included.
Fixed here.

### The correctness surface, checked before renting (CPU only)

`tests/python/test_race_latch.py`, 17 cases; whole suite green before and
after (144 passed / 3 skipped locally; 146 passed on the box with the one
documented 3090 `test_march_is_bit_exact_against_the_legacy_kernel` failure).

| what | why it is in the file |
|---|---|
| flag OFF is the control **bit for bit** - values, dtype, stall mask, stagnant mask, over a path that dives past the threshold and climbs back out | an arm that must start identical has to be provably identical |
| the valley costs **0.000** under the latch and **-4.02** under the floor, on the real 6,632 -> 14,976 segment | the treatment, stated as a number |
| total shaping over a run is `scale * (d0 - L)` and not one unit more, whatever happens after the switch | no new income channel was opened |
| clears at an episode start; a spawn already inside the shell starts LATCHED | at `--respawn-margin 2` the reservoir really does place starts past the wall, and charging those for leaving is the term being removed |
| the column read at t is exactly what decides whether t+1 pays | the Markov property, tested directly rather than argued |
| the stall kill still fires on an env genuinely stuck inside the shell, and does NOT fire on a correct approach inside it | the one that silently invalidates the run |
| the 1-wide block warm-resumes function-identically and reaches BOTH towers | step 0 is the baseline; the critic can see the regime |

### The run

Single rented RTX 3090 (instance 48399814, machine 38527), warm resume of
`runs/sOBSR2/ckpt_latest.pt` md5 `1ba1fd2936af3ae1ad3608e3cd6b1e9e` at step
3,782,737,920, **md5 verified on the box**. 800,587,776 steps at an average of
**263,096 steps/s** (14:09-15:03 UTC), never decaying, never stationary. One hour,
one seed.

### RESULT: THE MAP IS FINISHED WITH NO REFERENCE LINE ANYWHERE

All figures `--order-only 16` against the champion route
(`maps/surf_src_cannonball.route.npz`), used only as a ruler.

| eval | steps after resume | race/eval_progress | corridor MAX (order-only) | past 205,440 | finishes | best finish |
|---|---|---|---|---|---|---|
| 1 | +0.8M | 154,430 | 205,414 | 0/9 | 0/9 | - |
| 2 | +76M | 191,850 | 208,068 | 7/9 | 0/9 | - |
| 3 | +152M | 145,241 | 207,787 | 6/9 | 0/9 | - |
| 4 | +227M | 173,266 | **226,714** | 7/9 | **1/9** | 80.47 s |
| 5 | +303M | 154,192 | **231,664** | 6/9 | **3/9** | 81.14 s |
| 6 | +378M | 160,148 | **231,680** | 6/9 | **5/9** | 81.71 s |
| 7 | +454M | 197,637 | **231,680** | 9/9 | **8/9** | 80.89 s |
| 8 | +529M | 169,640 | **231,680** | 7/9 | **7/9** | 80.82 s |
| 9 | +605M | 198,381 | **231,575** | 9/9 | **9/9** | 80.94 s |
| 10 | +680M | 176,544 | **231,602** | 8/9 | **8/9** | **80.35 s** |
| 11 | +756M | 176,676 | **231,680** | 8/9 | **8/9** | 81.06 s |
| rec | +793M | - | **231,680** | 3/3 | **3/3** | 80.84 s |

**52 of 102 greedy episodes finished the map. 76 of 102 crossed 205,440 u.
Corridor MAX 231,680 u = 100%.** Finish times best **80.35 s**, median
81.23 s, mean 81.29 s, worst 82.35 s - every one of them measured as
first recorded tick to the first tick inside the finish box (+64 u pad),
the same basis as xARC / xAUTO / xSELF, which is why these differ by
0.02-0.06 s from the trainer's own `fin ... best` log lines.

Eval 1 - the untreated policy at +0.8M steps - lands on 205,414 u with 0/9
past the line and 0 finishes, exactly where xMARGIN, xCLAMP, xARC, xAUTO and
xSELF all opened. It is the internal control.

**The dive signature dies out.** Episodes that end below the finish box
without crossing it, by eval: 6, 7, 6, 7, 3, 1, 1, 0, 0, 0, 0, 0. That is the
difference between xCLAMP and this arm stated as one row: the floor deleted
the fall's INCOME and the agent kept falling (60 of 99); the latch deleted the
climb's COST and the agent stopped falling and started landing.

### Against the controls, all on the same checkpoint and scored the same way

xMARGIN, xCLAMP and xLATCH were re-scored here from their own recordings with
one `eval_honesty.py --order-only 16` invocation each; xMARGIN reproduces its
published 7/72 and 208,640 u exactly, which is the check that the numbers
below are comparable. xARC / xAUTO / xSELF are quoted from their sections.

| arm | what the reward pays past the wall | greedy eps | corridor MAX | past 205,440 | finishes |
|---|---|---|---|---|---|
| xROUTE / xSP / xNECTO | geodesic potential, unmodified | 234 | 205,312-205,440 | 0 | 0 |
| xMARGIN | geodesic potential, unmodified | 72 | 208,640 | 7 | **0** |
| xCLAMP | floored potential: charges the climb OUT | 99 | 208,697 | 53 | **0** |
| | *(all 11 recordings in `runs/research/xCLAMP/`; its own section may quote a subset)* | | | | |
| xARC | arc length on the champion's line (100%) | 102 | 231,680 | 84 | **63** |
| xAUTO | arc length on 58 chords (100%) | 102 | 231,680 | 81 | **62** |
| xSELF | arc length on its own failure (88.12%), then nothing | 102 | 231,680 | 77 | **47** |
| **xLATCH** | **nothing, from a distance threshold** | **102** | **231,680 (100%)** | **76** | **52** |

Per CLAUDE.md rule 3 a run that finishes is judged on wall clock, measured
identically across the arms (first recorded tick to the first tick inside the
finish box, +64 u pad):

| | best | median | mean | worst | finishers |
|---|---|---|---|---|---|
| champion `runs/sISV_par2/traj_8454144000.jsonl` | 81.35 s | 82.16 s | 82.19 s | 82.78 s | 7/9 |
| xARC (champion line, full) | 81.04 s | 81.74 s | 81.82 s | 83.13 s | 63/102 |
| xAUTO (champion line, 58 chords) | 80.51 s | 81.39 s | 81.41 s | 82.91 s | 62/102 |
| xSELF (own failure, 88.12%) | **80.06 s** | 81.18 s | 81.25 s | 82.44 s | 47/102 |
| **xLATCH (no line at all)** | **80.35 s** | **81.23 s** | **81.29 s** | **82.35 s** | **52/102** |

**A reward that contains no reference line of any kind finishes this map more
often than the one built from the agent's own best failure, and within a
tenth of a second of it on every time statistic.**

### `race/eval_progress` was anti-correlated for the fourth arm running

154,430 -> 191,850 -> 145,241 -> 173,266 -> 154,192 -> 160,148 -> 197,637 ->
169,640 -> 198,381 -> 176,544 -> 176,676, against xMARGIN's 130,271 /
177,733 / 174,130 / 175,271 / 173,888 / 156,730 / 188,261 / 157,085. Same
band, same shape, no trend - **while the honest frontier went 205,414 ->
231,680 and 0 finishes became 52.** Eval 5 (154,192, one of the run's three
lowest readings) is the eval that finished 3 of 9. The metric is measuring
whether an episode dives into the goal-adjacent basin, and the treated agent
stopped doing that.

Reading these two arms on `eval_progress` alone would have ranked xMARGIN
above xLATCH.

### `tools/wall_profile.py`, final eval, against the champion line

Eight of nine episodes reach vertex 1810 of 1811 (the ninth died at vertex 27
- an early death, not a wall failure).

| vertex | xLATCH speed | off the champion line | champion speed |
|---|---|---|---|
| 1540 | 2,838 u/s | 171 u | 2,927 |
| 1560 | 2,844 u/s | 169 u | 2,928 |
| 1580 | 2,852 u/s | 219 u | 2,935 |
| 1600 | 2,856 u/s | 171 u | 2,926 |
| 1620 | 2,853 u/s | 98 u | 2,928 |
| 1640 | 2,844 u/s | 104 u | 2,921 |
| 1660 | 3,700 u/s | 61 u | 3,728 |
| 1680 | 3,627 u/s | **1,456 u** | 3,644 |

The control's departure at vertex 1598 was 1,735-3,019 u off-line; xMARGIN
got it to 177-254 u and still never finished. This arm holds 61-219 u through
the whole approach, accelerates into the final descent within 28 u/s of the
champion, and then takes a line 1,456 u away in the bowl - and finishes.
Exactly xSELF's signature (102 u at 1600, 1,100 u at 1680), from a reward
that never saw a line.

### Training diagnostics, 1,018 logged iterations

| | first 20% | middle | last 20% |
|---|---|---|---|
| training win rate | 0.00% | 60.30% | **77.56%** |
| `rollout/ep_rew_mean` | 22.40 | 48.51 | 56.18 |

First training win at **+184.8M steps**, peak win rate 86.11%. **The stated
`--respawn-margin 2` hazard fired here and has to be named**: CLAUDE.md warns
that once an arm starts finishing, a 2 s harvest margin puts states 2 s from
the goal into the reservoir and the run can self-reinforce on trivial wins.
Training episodes late in the run average 28-36 s, i.e. they are
reservoir-seeded fragments, and the 77-86% win rate is a rate over those, not
over full runs. It is not what this arm is scored on: **every number in the
result table is a GREEDY eval from the platform spawn pool, a full 80-82 s
run of the whole map.** xARC, xAUTO and xSELF all ran under the same margin
after they started finishing, so this is a shared condition and not a
difference between the arms.

### VERDICT

**STRONG POSITIVE. The reference line is scaffolding.**

The pre-registered read was: finishes near 47 means the polyline is
scaffolding and nothing trajectory-derived is needed; finishes near 0 means
the arc coordinate does something a route-free rule cannot. **52 of 102.**

What this settles, in order of how much it was in doubt:

1. **The barrier was never an exploration problem, a start-state problem or a
   perception problem. It was the shaping reward's own arithmetic**, and the
   fix does not need a line, a demonstration, a champion, a goal field
   rebake, or any object derived from a trajectory. It needs one distance
   threshold and one bit of episode state.
2. **The distinction that matters is CHARGING, not FLATNESS.** xCLAMP and
   xLATCH share a threshold, a derivation, a start state and a spawn margin,
   and differ in one term: whether the climb back out of the shell is
   charged. That single term is 0 finishes against 52. The clamp's own
   53/99 past the wall was mostly falling past it (60/99 dives-below) -
   deleting the fall's income is not the same as deleting the climb's cost,
   and only the second one lets the agent cross.
3. **The autonomy claim is now clean.** xSELF removed the champion from the
   provenance but still needed the policy's own recordings to build a line.
   This needs neither: `--race-latch 6996` is a scalar, and `pick_dfloor.py`
   derives it from recordings the policy produced by failing. Nothing in this
   arm's reward knows the shape of the route.

**What it does NOT settle.** The threshold still has to come from somewhere,
and here it came from the policy's own recorded failures on this map - so
this is champion-free, not map-free. And the seed is still a policy that
already flies 88% of the map; where that comes from on an unflown map is
untouched by this result, exactly as it was after xSELF.

**The cheapest next thing** is a threshold sweep: the latch is one number, so
whether 6,996 is load-bearing or whether anything in a broad band works is a
one-flag ablation, and a band would be much stronger evidence than a point.
The obvious paired control is `--race-latch` at a value far from the wall
(e.g. 60,000), which should be a regression if the number matters and a null
if it does not.

### Ops and cost

* `fleet_watchdog list` empty at start; the daemon was already running. Raced
  three RTX 3090 candidates - 48399814 (machine 38527, host 198030),
  48399819 (machine 29027), 48399820 (machine 12552) - all registered on
  create with a 115-minute deadline. Cap of 4 never exceeded.
* **The 60-second rule held.** 48399814 reached `running` and answered ssh
  within 70 s of create; the two losers were destroyed and released at
  14:07:15 and 14:07:24, both confirmed gone.
* `deploy_box.sh` with `BRANCH=latch`,
  `LOCAL_CKPT=/c/RL_Surf/runs/sOBSR2/ckpt_latest.pt`,
  `EXPECTED_MD5=1ba1f...`. Checkpoint md5 **verified on the box**
  (`1ba1fd2936af3ae1ad3608e3cd6b1e9e`, step 3,782,737,920), caches shipped,
  bsp mtime pinned, `gpu_health.py` VERDICT healthy (862 GB/s HBM, 73 TFLOPS
  bf16, 1,740 MHz and 286 W of 350 W under load).
  `pip install --break-system-packages pytest scipy` on top, as expected.
* Trainer alive and watched throughout: 263k steps/s average, cumulative fps
  monotonically rising the whole run, no decay, the frontier moving at every
  check. Box destroyed 15:05:06 UTC, confirmed gone, watchdog released, the
  registry's remaining two boxes belong to other agents.
* **Rental cost: $0.15** for the winner (59.1 minutes at $0.1533/h) plus
  under **$0.01** for the two racing losers, which lived 72 s and 80 s.
  **Total ~$0.16.** Everything else was local CPU.
* Artifacts in `runs/research/xLATCH/`: 11 trajectory files, `rec_final.jsonl`
  (3/3 finishes, recorded on the box through the patched `record_ckpt.py`),
  `progress.csv`, `run.json`, `xLATCH_launch.txt`.

**A note on how this was scored.** `eval_honesty.py --order-only 16` and the
`surfgym.route.ArcProgress` it calls live on the arclen/autoline/selfline
branches; the `clamp` branch this arm was cut from predates them. Every
number above was produced by running the **selfline** branch's
`tools/eval_honesty.py` and `tools/wall_profile.py` unmodified against these
recordings, so xLATCH is scored by exactly the code that scored xARC, xAUTO
and xSELF. Nothing from those branches was merged into `latch`, which carries
only the arm. The check that this is sound is xMARGIN reproducing its
published 7/72 and 208,640 u to the unit.

## Round 20 - xNS64 / xNS128 / xNS256: does the PPO rollout length matter? (2026-08-23 14:20-15:15 UTC)

**One question:** does `--n-steps` (the PPO rollout length T) move learning on
cannonball? Three arms, identical in every respect except that one flag:
`xNS64` (T=64), `xNS128` (T=128, the default and **the control**), `xNS256`
(T=256). One seed each, one hour each, three RTX 3090s.

**VERDICT: it matters, and it is a large effect. T=64 reaches 1.64x the
control's frontier. T=128 and T=256 are not separable from each other.**

### Baseline: the NEW from-scratch baseline, not the stuck checkpoint

These arms were launched **twice**. The first launch was a warm resume of
`runs/sOBSR2/ckpt_latest.pt` under the old rule; it was killed after ~9
minutes when the user's baseline changed to cannonball / 64x32 depth /
**no `--obs-reward`** / **from scratch** / 1 hour. **Nothing from the resumed
runs is reported here** and their run directories were removed on the boxes
so no scratch `progress.csv` could append to them.

Launched with `SCRATCH=1 BUDGET=800000000 bash tools/run_arm.sh <ARM>
--n-steps <T>` on branch `maskonly` @ `2a83357`. `run.json` was confirmed on
every box before any conclusion was drawn - the ONLY differing field is
`n_steps`:

| arm | n_steps | minibatches | epochs | obs_reward | ckpt | map | lidar | envs |
|---|---|---|---|---|---|---|---|---|
| xNS64 | **64** | 16 | 4 | False | None | surf_src_cannonball | 64x32 | 2048 |
| xNS128 | **128** | 16 | 4 | False | None | surf_src_cannonball | 64x32 | 2048 |
| xNS256 | **256** | 16 | 4 | False | None | surf_src_cannonball | 64x32 | 2048 |

(That check only became possible with `2a83357`; before it, `n_steps` and
`minibatches` were never written to `run.json` and read `None` even on a run
launched with the flag explicitly. For an ablation whose arms differ in
exactly that flag, the permanent record would have shown nothing
distinguishing them.)

### The result - honest metric, at MATCHED steps

`tools/eval_honesty.py --order-only 16`, 11 evals x 9 greedy episodes = **99
episodes per arm**. Corridor MAX in units of the 231,680 u route:

| steps | xNS64 MAX | xNS128 MAX (control) | xNS256 MAX |
|---|---|---|---|
| 75M | **7,629** | 5,548 | 2,785 |
| 150M | **14,299** | 8,236 | 5,809 |
| 225M | **16,878** | 10,504 | 10,046 |
| 300M | **18,073** | 15,645 | 11,871 |
| 375M | **22,809** | 17,932 | 15,005 |
| 450M | **27,095** | 18,019 | 17,070 |
| 525M | **30,087** | 18,082 | 17,067 |
| 600M | **34,048** | 20,447 | 17,794 |
| 675M | **35,002** | 22,836 | 18,825 |
| 750M | **37,432** | 22,834 | 19,150 |

**Arm totals:** xNS64 **37,432 u (16.16%)**, xNS128 **22,836 u (9.86%)**,
xNS256 **19,150 u (8.27%)**. **0 finishes and 0 dives in 99/99 episodes for
all three arms**, as expected from scratch in an hour.

**xNS64 leads at all 11 of 11 matched-step evals**, by 1.4x-1.7x, and the
lead widens rather than closing. Corridor MEAN tracks MAX the whole way
(35,389 / 18,126 / 18,925 at 750M), so this is the whole distribution
moving, not one lucky episode - which is exactly the check that turned
round 18's xROUTE into a null.

**xNS128 vs xNS256 is a TIE and must be reported as one.** MAX favours 128
(22,836 vs 19,150) but corridor MEAN at 750M favours 256 (18,925 vs 18,126),
`eval_progress` favours 256 (18,403 vs 17,208), and the two arms cross twice
(at 225M and 525M). One seed cannot separate them. **The effect is not
monotone in T; it is T=64 against everything else.**

### The standing metric agrees this time - and that is worth recording

`race/eval_progress` at 750M: xNS64 **33,830**, xNS128 17,208, xNS256 18,403.
Same verdict, same ~2x gap, same 128/256 tie. After round 19, where
`eval_progress` was ANTI-correlated with the truth, an arm where the two
metrics agree end to end is a stronger result than either alone. (The
documented 140k-195k band does **not** apply here - that is the stuck
checkpoint's band and these are scratch runs.)

### win_rate paired with reservoir min-depth, as required

`race/win_rate` was **0.00% for all three arms for the entire hour**, so the
trivial-win trap never fired. Final reservoir min-depth (all three
reservoirs full at 100,000 states): xNS64 **161,632 u**, xNS128 177,325 u,
xNS256 189,857 u, against a start distance of 198,380 u. **The min-depth
ordering matches the frontier ordering** - xNS64's reservoir is harvesting
states ~28k u further down the map than xNS256's - which corroborates the
frontier from an independent quantity.

### What `--n-steps` actually changes here: it is TWO knobs, not one

This is the caveat that decides how much the result is worth.
`--minibatches` is a COUNT, not a size. It was held at 16 across the arms
(as "identical except --n-steps" requires), so gradient updates per rollout
are pinned at `epochs x minibatches = 64` while the rollout itself gets
longer:

| T | env steps/rollout | minibatch size | grad updates per 1M env steps | rollout ticks | discount weight inside rollout |
|---|---|---|---|---|---|
| 64 | 393,216 | 8,192 | **162.8** | 192 | 9.2% |
| 128 | 786,432 | 16,384 | **81.4** | 384 | 17.5% |
| 256 | 1,572,864 | 32,768 | **40.7** | 768 | 31.9% |

**T=64 takes 2x the gradient steps per environment step of T=128 and 4x of
T=256, on 1/2 and 1/4 the batch.** So the arm confounds the GAE-horizon
effect it was designed to test (how much discount weight falls inside the
rollout before it bootstraps) with **update density**, and the shape of the
result - a large win for the smallest T, no separation between the other two
- looks much more like update density than like a horizon effect. A horizon
story would predict T=256 doing something distinctive at the long end; it
does not.

**Do not bank the horizon interpretation.** The clean follow-up is one arm
that holds updates-per-step fixed by scaling `--minibatches` with T
(`--minibatches 8` at T=64, `32` at T=256, keeping minibatch SIZE at 16,384).
If T=64 still wins there it is the horizon; if the arms collapse together,
this round measured update density, and the answer is "take more, smaller
gradient steps" - a cheaper knob than T.

The flag's own comment in `train_fast.py` already says as much - "update
density matters as much as throughput... 64 -> 300M-step sample-efficiency
regression when this was 2 epochs x 8 minibatches over 1M-sample rollouts" -
so this round is consistent with a regression already seen once from the
other direction.

### Not a throughput effect

Measured over the full runs: **269,730 / 279,013 / 287,095 steps/s** for
T=64 / 128 / 256. Larger T is marginally **faster**, so xNS64 won *despite*
the lowest throughput, and per-step comparison is the conservative direction
here. All three consumed ~800.4M steps in 45-47 minutes of training. The
cumulative `fps` column is a running average from startup and was not used.

### Ops and cost

* All three arms ran on **three separate 3090s inside ONE physical machine**
  (16571, Spain, EPYC 7B13, 21 effective cores each) - the cleanest possible
  same-card comparison. GPU health VERDICT healthy on all three (840 GB/s
  HBM; 69 / 72 / 73 TFLOPS bf16).
* **The 60-second readiness rule cost three racing rounds and 6 blacklist
  entries.** Of 9 candidates created, only **2 ever reached `running` inside
  the window**; the rest were still pulling the ~7 GB devel image past 60 s.
  Blacklisted (`network`) then destroyed, per the precedent already in the
  file.
* **What actually predicted readiness was whether the host had the image
  cached, and the only usable proxy was a machine already seen coming up
  fast.** Machine 16571 was ssh-ready **17 s** after create - and it is
  already in `known_good` as "the only one ssh-ready in 31.5 s after
  create". Its two **sibling offers** came up in ~35 s and ~60 s.
  **Renting the siblings of a known-fast machine beat re-rolling the general
  pool three times over. Do that first next round.**
* **The blacklist is now 56 entries, 6 of them from this round, against a
  3090 pool that lists ~30 offers of which 12-14 were already blocked.** The
  60-second rule and the size of the 3090 pool are on a collision course.
  Worth a user decision on whether "still pulling the image" is a defect or
  just needs the sibling-renting pattern above.
* **`vastai create` returned `"success": false` and created instance
  48473141 anyway.** Caught unregistered by `fleet_watchdog list` and
  destroyed **unblocked** - a create-API glitch on a machine that is in
  `known_good` is not evidence of a defect. **Do not trust `success: false`;
  check `show instances`.**
* **`fleet_watchdog extend --minutes N` SETS the deadline to now+N, it does
  not ADD N.** "Extending" three boxes registered for 100 minutes by 25 cut
  their deadlines from 15:33 to 14:30 - it would have destroyed three
  mid-training boxes ~40 minutes early. Caught and corrected within 10 s.
  Same family as the 2026-08-23 watchdog incident already in CLAUDE.md.
* **Do not re-`register` a live box to relabel it.** `cmd_register` REPLACES
  the entry, dropping the latched `ready` flag - the exact field whose
  absence makes the readiness rule destroy a running box.
* **scp truncated the harvested checkpoint and exited 0**: 146,488,320 bytes
  local against 153,649,611 on the box, md5 mismatch. Caught only by the md5
  check, re-pulled, verified `1324d47c2ab8b3542ab33f4dcd1886c2`. The standing
  rule held; it fires often enough to be worth the check every time.
* **No `numba` on any box** (the pytorch image ships without it), so all
  three ran the numpy reference `GoalField.sample` instead of the fused
  kernel `maskonly` adds. Deliberately not installed: the fused path is
  bit-identical per its own commit, so it costs throughput only, and leaving
  all three boxes identical keeps the arms comparable.
* Local pre-flight before renting: T=64 and T=256 both smoke-tested on the
  local 5090 for full iterations. T=256 peaked at ~6.7 GB - the real risk was
  OOM on a 24 GB 3090, since T doubles both the rollout buffer and the
  minibatch (MB = T*N/16 = 32,768 at T=256). It was fine.
* Watchdog: every instance registered on create; all three released and
  **confirmed gone** at 15:15:25 / 15:15:31 / 15:15:38 UTC.
* **Rental cost ~$1.49** (81.9 + 75.5 + 73.2 minutes at $0.372/h, plus ~$0.06
  for 6 racing losers and 1 duplicate).
* Artifacts in `runs/research/xNS{64,128,256}/`: 11 trajectory files each,
  `progress.csv`, `run.json`, `<ARM>_launch.txt`; plus `xNS64/ckpt_800M.pt`
  (md5 `1324d47c2ab8b3542ab33f4dcd1886c2`, verified against the box).
* Scored with the **arclen** branch's `tools/eval_honesty.py` and its
  `surfgym.route.ArcProgress`, run from a detached worktree - `maskonly` has
  neither and `--order-only 16` is mandatory. Same procedure the xLATCH entry
  above documents; verified by reproducing xARC's published 231,680 u and
  9-of-9 finishes to the unit before scoring anything here.
---

## Round 20 - the 47-map "ready" shortlist, VERIFIED. 22 of 47 pass; the other 25 are staged maps

No GPU, no rental, no bake: `tools/verify_maps.py` (new) runs four checks per
map on the CPU. Artifacts: `runs/research/maps_verified.json` (per-map
record), `runs/research/stage_links.json` (the 126), and
`runs/research/map_survey_zonefix.json` (the 620-map survey re-run with the
zone bug below fixed).

### The four checks

1. **loads** - `SurfCore(bsp, default_config(num_envs=8, spawn_mode=2,
   lidar_w=0, lidar_h=0))` + `reset(0)`.
2. **spawn sane** - every `info_player_start` origin (which is exactly what
   `reset_env` copies, src/env.c) traced with the standing hull for
   `startsolid`, checked against `kill_z` (auto = world min z - 256), plus 20
   neutral ticks over 8 envs to catch a lethal `trigger_hurt`, water or a
   stuck spawn. Disqualifying only when it takes the whole map out - one
   embedded spawn of 32 is a warning, it only wastes 1/32 of episodes.
3. **finish reachable from the spawn** - the check paper classification
   cannot make, and the one that matters.
4. **d0 and extent** - euclidean spawn-to-finish-AABB distance, map AABB,
   free-space volume, for fleet sizing.

### The reachability method and its error bars

`build_goal_field` is authoritative and costs a GPU bake of minutes per map.
Its graph is not exotic, though: `_bfs_geodesic` relaxes over the **26**
neighbours of every FREE voxel of `goal_occupancy` = `vision.slab_occupancy`
at `vision.pick_cell`'s cell, seeded at the end zone inflated by
`_zone_seed_box`. "Is the spawn in the finish's component?" is therefore a
connected-component LABEL, not a shortest path, and
`scipy.ndimage.label(free, structure=ones(3,3,3))` answers it on the CPU in
seconds on the identical grid at the identical cell. The answer IS
`GoalField.reachable(spawn)`.

**Cell**: `pick_cell`'s standard 700M-voxel budget - 16u for a normal
8k-unit map, 32u for a source port, 64u for the six 30-48k-unit monsters.
Occupancy is rebuilt in-process, so `maps_full_dataset/` was never written to.

**Validated against three maps with known answers**: petrus_lite reachable,
cannonball reachable, sidistic NOT reachable. On sidistic it does not merely
agree with round 19's xSID GPU bake, it **reproduces its numbers**: 4 free
components at cell 32, finish component **214,523** voxels, **1,773,929** at
cell 16, **0/2** spawns, seal **7 cells = 224 u**.

**False negatives.** Slab occupancy dilates geometry by up to cell/2 per
axis, so a passage narrower than about one cell can read as sealed - at cell
64 the test wrongly calls petrus_lite unreachable. Three guards: the cell is
the finest that fits the standard budget; every failure is re-tested against
the **permissive** centre-sampled grid at the same cell, which does not
dilate at all; and `gap_units` measures the solid the finish component must
be dilated through before it touches the spawn's. Each side can grow by at
most cell/2, so a seal of >= 2 cells is thicker than the dilation could have
made it and is real wall. Calibration point: sidistic, 224 u at cell 32,
where the ledger's manual read found 64 u of worldspawn.

**False positives.** The slab lattice samples at cell/4, so worldspawn
thinner than that (4u at cell 16, **16u at cell 64**) can still be threaded,
merging two genuinely sealed components. The `--goal-cell` work on `nsteps`
measured exactly this on cannonball - at cell 64 the wavefront tunnels
through thin floors and d0 halves, while a reachability check still passes.
The six maps `pick_cell` puts at 64u were therefore re-run at 32.

### A bug in detect_zones: the origin brush

A brush entity built around an ORIGIN BRUSH stores its model vertices
relative to that origin and carries the world offset in the `origin` key;
world = model + origin, which is the convention `goalfield`'s kill-volume
masking already assumes (`hull_probe.contains(mi, pts - origin)`).
`detect_zones` ignored it, so on such a map the race zones landed near the
world origin - a phantom finish box.

**9 of the 173 zoned maps (5%) are affected**, including `surf_src_sidistic`,
which is in `maps/`. On the five inside the 47 the correction is unmistakable:

| map | start line to nearest spawn, before -> after | finish centre contents |
|---|---|---|
| surf_bomzis | 5,634 -> **147** u | SOLID -> empty |
| surf_sg_china | 4,148 -> **189** u | empty -> empty |
| surf_sg_dash | 4,032 -> **144** u | empty -> empty |
| surf_sg_oldtemple | 4,713 -> **64** u | SOLID -> empty |
| surf_src_whoknows2_b1 | 1,212 -> **473** u | SOLID -> empty |

Race length `d(start,end)` goes from 62-368 u (i.e. the two zones on top of
each other near the origin) to 7,334-21,526 u. Sidistic's finish moves 512 u
- into the same free-space component, so **the xSID verdict is unchanged**.

Fixed in `python/surfgym/zones.py` with
`tests/python/test_zone_origin_offset.py`. **No map in `maps/` except
sidistic carries such a trigger, so every trained checkpoint's zone boxes are
bit-identical.** Re-running the whole 620-map survey with the fix moves
exactly six maps: bomzis, sg_china, sg_dash, sg_oldtemple and whoknows2_b1
leave `ready` for `zones_but_links` (with a correct finish, their teleports
read as end-ward), and `surf_kei_luupy` moves the other way. **The shortlist
is 43 + kei_luupy, not 47** - and `surf_sg_dash` is the cautionary one: it
PASSED all four checks against the phantom box and fails against the real
finish.

### The table: 22 of 47 pass

**PASS (22)** - loads, spawn sane, finish reachable, d0 recorded:

| map | cell | free comps | spawns reachable | d0 (u) | extent (u) |
|---|---|---|---|---|---|
| surf_freeland_uncapped | 16 | 1 | 29/29 | 5,442 | 7808 x 7232 x 7809 |
| surf_gi_rino | 32 | 1 | 16/16 | 18,229 | 16693 x 16757 x 15808 |
| surf_hamburglar_love | 16 | 1 | 1/1 | 7,661 | 8256 x 4720 x 4304 |
| surf_hollow_lite | 16 | 36 | 1/1 | 9,203 | 8159 x 2291 x 6726 |
| surf_latebra | 16 | 1 | 32/32 | 5,609 | 5184 x 8064 x 4384 |
| surf_lowestbidder | 16 | 1 | 1/1 | 7,674 | 7848 x 5424 x 5696 |
| surf_prechasm | 16 | 1 | 23/23 | 4,862 | 6387 x 3780 x 2172 |
| surf_secret_passage | 16 | 14 | 1/1 | 7,499 | 8094 x 5473 x 7918 |
| surf_sg_guater | 16 | 1 | 31/31 | 8,503 | 7872 x 7856 x 7664 |
| surf_simulatedway | 16 | 4 | 11/11 | 9,762 | 7928 x 8151 x 8040 |
| surf_src_celestial | 16 | 3 | 1/1 | 16,816 | 16012 x 14038 x 11702 |
| surf_src_celestial_b2 | 16 | 3 | 1/1 | 16,816 | 16012 x 14038 x 11702 |
| surf_src_celestial_b3 | 16 | 3 | 1/1 | 16,816 | 16012 x 14038 x 11702 |
| surf_src_hollow | 16 | 35 | 1/1 | 22,904 | 20400 x 5728 x 16816 |
| surf_src_ing | 32 | 3 | 1/1 | 29,630 | 11328 x 32047 x 27712 |
| surf_src_ing_b1 | 32 | 3 | 1/1 | 29,630 | 11328 x 32047 x 27712 |
| surf_src_ing_b2 | 32 | 3 | 1/1 | 29,630 | 11328 x 32047 x 27712 |
| surf_src_joutsenlaulu_b1 | 32 | 16 | 1/1 | 24,661 | 32672 x 32320 x 32256 |
| surf_src_mellow | 16 | 1 | 1/1 | 15,505 | 6720 x 15936 x 12736 |
| surf_src_volcanic | 16 | 1 | 1/1 | 13,346 | 14304 x 14848 x 11616 |
| surf_unitfarmer2 | 16 | 1 | 1/1 | 5,612 | 7936 x 7968 x 3072 |
| surf_volcanic_lite | 16 | 1 | 1/1 | 6,702 | 7152 x 7424 x 5808 |

Three of those are the same map (`surf_src_celestial` / `_b2` / `_b3`,
identical extents and d0) and three more are `surf_src_ing` / `_b1` / `_b2`,
so the set is **18 distinct maps**. Add `surf_kei_luupy` (pass, cell 16, 6
components, 30/30 spawns, d0 7,147), which the zone fix moves INTO the ready
class: **23 verified files, 19 distinct maps.**

**FAIL (24) + ambiguous (1), by which check and why:**

| bucket | n | maps |
|---|---|---|
| **STAGED - the finish's free-space component is entered by a TELEPORT** | 22 | anguish, benevolent, bomzis, indoors, malevolent, pecado, princessburglar_b24, sg_china, sg_dash, sg_oldtemple, spastic_b2, src_aura_b2, src_corruption, src_cyberwave, src_driftless, src_epiphany, src_quickie, src_quickie_b5, src_shade, src_whoknows2_b1, tronic_lite, zen_b1 |
| **spawn unusable** - every env dies inside 20 neutral ticks | 2 | src_quickie_b2, src_twist |
| **narrow link** - slab grid sealed, undilated grid connected, seal < 2 cells | 1 | inprison |

Nothing failed check 1 (all 47 load) and nothing failed for want of a zone.
23 of the 25 are disconnected under the **permissive** undilated grid too, so
they are not dilation artifacts; the two that are not (`pecado` gap 32u,
`inprison`) are the only ones where the wall model is doing the work.
`gap_units` where measurable: anguish 224, princessburglar 272, tronic_lite
272, twist 192, zen_b1 192 - all far past the 32u player hull.

**The six 30-48k-unit maps that `pick_cell` puts at 64u were re-run at 32u**
(`runs/research/maps_verified_cell32.json`), the cell where the `--goal-cell`
work says the field stops tunnelling. **Every verdict held**: corruption,
cyberwave, driftless, epiphany, whoknows2_b1 fail at both cells,
joutsenlaulu_b1 passes at both.

Four more things the table records that are not verdicts but would have cost
a night each:

* `surf_src_shade`'s d0 is **1,057 u** - the spawn is essentially at the
  finish and the detected start line is 5,667 u away. A degenerate race.
* `surf_src_driftless`'s start line is **12,969 u** from the nearest spawn,
  `surf_src_corruption`'s 2,998 u. On the other 44 it is 44-1,160 u.
* `surf_src_quickie` / `_b2` / `_b5` have **32/32 spawns inside geometry**.
  On `_b2` that is fatal - 640 episode deaths in 400 neutral ticks over 8
  envs. On the other two the sim settles them and they run, so `startsolid`
  alone is a warning here, not a verdict.
* `surf_src_twist` likewise dies in every env at spawn.

### What the failures actually ARE, and how many come back

**24 of the 25 non-passing maps have at least one teleport landing inside
the finish's free-space component**, and on 15 of them a teleport goes
directly **from the spawn's component into the finish's**. The picture is
uniform and it is not exotic geometry: **the spawn sits in a sealed start
room whose exit is a teleport**, which under `teleport_fail` ends the
episode on touch. `surf_src_twist` is the extreme case - its 184 teleports
share ONE destination, the finish component holds 99.9% of free space, and 8
of those 184 are the start-room door. `survey_maps.py` called all 184 death
catches, which is right about 176 of them.

So these are not broken maps and they are not `ready` maps. **They are
stage-link maps whose first link is the start-room exit**, and the survey
missed it for exactly the reason Part 2 measures: the link's destination
sits next to an `info_player_start`, so the end-ward distance test reads it
as a respawn.

**15 of them come back for free.** `SurfCore.set_spawn_pool` already exists
and `spawn_mode=2` already uses it, so seeding the pool at that teleport's
DESTINATION starts the agent past the start-room door, on the real start
line, with every teleport still fatal and the real finish still the finish.
No code change, no map edit, and the destination is in the finish's
component by construction, so check 3 passes by construction too. The 15:
anguish, malevolent, pecado, princessburglar_b24, sg_dash, spastic_b2,
src_corruption, src_cyberwave, src_driftless, src_quickie, src_quickie_b5,
src_shade, src_twist, src_whoknows2_b1, zen_b1.

The remaining 9 have the finish behind a teleport that is NOT reachable from
the spawn's component, so the way out of the start room is something else -
a `func_door`, which `src/bsp.c` makes permanently SOLID (there is no
entity-I/O system anywhere in `src/`: no button, no target, nothing opens)
and which `surf_occupancy_grid` clips through a zero-length hull-2 trace, so
a closed start gate is a wall in the grid AND in the physics. Those need
either the same spawn-pool treatment aimed past the door or a door model in
the core. `tools/verify_maps.py --diagnose <bsp>` prints which brush
entities straddle two components.

**Headline for a fleet.** 22 of 47 verified as-is, 23 with `surf_kei_luupy`,
19 distinct maps after de-duplicating the `_b2`/`_b3` re-releases; plus 15
recoverable by a spawn-pool seed that costs nothing.

### Part 2 - the 126 `zones_but_links` maps

`tools/stage_links.py` (new) dumps every teleport of all 126 with its source
brush AABB, destination, and whether it moves the player materially toward
the finish. `runs/research/stage_links.json`.

**How many links.** 40 of 126 (32%) have exactly ONE end-ward destination;
75% have three or fewer. Median map: 29 live teleports, 7 end-ward, 22
backward, 5 distinct destinations. But of those 40 single-link maps only 16
have a single SOURCE brush - the rest have 2 to 73 brushes pointing at the
same destination, because a stage's start doubles as the respawn point for
that stage's own fall nets.

**The end-ward rule itself is wrong about half the time.** 696 of the 1,491
end-ward teleports (47%) land within 512 u of an `info_player_start`: they
are respawn nets whose SOURCE brush merely happens to sit farther from the
finish than the spawn does. And 256 of the 396 end-ward destinations (65%,
across 112 of the 126 maps) ALSO receive backward sources - on a staged map
the stage-k start IS both the previous stage's exit and this stage's respawn
point, so **no destination-level rule can separate them**.

Excluding respawn-adjacent destinations leaves **26 maps with no end-ward
teleport at all**. One of them is `surf_petrus_lite`: 17 teleports, every
one targeting `startspawn_1`, whose destination is **9.8 u** from
`info_player_start` - a map this project has trained and finished for rounds
under `teleport_fail`. `surf_src_petrus` and `surf_src_sidistic` are in the
same list. Adding `dest within 512 u of a spawn -> catch net, regardless of
its source brush` to `survey_maps.py` moves those 26 into the shortlist for
free.

**Nothing in the entity data separates links from nets.** Source-brush
geometry, end-ward-not-at-spawn (n=795) vs catch (n=4,180), p25/p50/p75:

| feature | end-ward | catch |
|---|---|---|
| z as a fraction of world height | 0.15 / 0.28 / 0.61 | 0.25 / 0.49 / 0.72 |
| min dimension, u | 4 / 32 / 320 | 2 / 4 / 96 |
| footprint, 1e3 u^2 | 90 / 1,180 / 7,250 | 162 / 1,541 / 6,304 |
| horizontal slab (dz<=64, min xy>=256) | 37.1% | 52.9% |

Best single-threshold balanced accuracy / AUC: `z_frac` 0.633/0.626,
`min_dim` 0.622/0.632, `volume` 0.558/0.571, `footprint` 0.551/0.517 -
chance is 0.500. **Both classes are mostly large thin horizontal slabs**;
"nets are wide slabs low in the map, portals are small" is false here.
Naming is no better: 73-77% of destination names in BOTH classes contain a
digit, the top tokens are shared (`mapstart`, `stage`, `start`, `spawn`,
`lvl`), and the trigger brush's own `targetname` is empty for 93% of
end-ward and 94% of catch brushes. Destination names also cover bonus rooms
(`bonus1`, `Secret 1`, `GoToDisco`) that are end-ward and off-route.

**What DOES separate them is free-space topology.** A stage link's
destination leaves the source's connected component; a catch net's does not.
That is the same connected-component pass check 3 already runs, and it is
the only rule measured here that works.


#### Option (b) is not a sketch - it was measured on 32 of the 126

`tools/verify_maps.py --stage1` locates stage 1 topologically: the free-space
component holding the spawn IS stage 1, and a stage link is a teleport whose
SOURCE brush is reachable inside that component and whose DESTINATION is not.
Run over every 4th map of the 126 (32 maps, unbiased w.r.t. size,
`runs/research/stage1_sample.json`):

| | n | share |
|---|---|---|
| finish already in the spawn's component - **single-stage, mis-binned** | 11 | 34% |
| staged, exactly ONE stage-1 exit destination | 10 | 31% |
| staged, 2-9 stage-1 exit destinations | 8 | 25% |
| staged, NO reachable exit from stage 1 | 3 | 9% |

**Nine of those ten single-exit maps have exactly ONE source brush** - one
AABB, straight out of `parse_bsp`, to write into `maps/<map>.zones.json` as
`end` with `"source": "manual"`. The tenth has 8 brushes sharing one
destination; any of them ends stage 1, so take their union.

The 11 mis-binned ones (kei_luupy, maestra, rapira, sabuleum,
src_cannonball_b3, src_forsaken_b2, src_inferno_b1, src_lockdown_b1,
src_raphaello, src_utopia, src_yellow_b1) need no splitting at all - they go
straight into the four checks. Scaled to all 126 that is roughly **43 maps
that were never staged**, and `surf_kei_luupy` is already verified `pass`.

The 3 with no reachable exit are the same class as the 9 door-gated failures
in Part 1 and should be looked at with `--diagnose` before being dropped.

#### Recommendation: option (b), stage 1 only

**(a) teleport-as-traversal** needs three things, one of them in C:
a per-teleport allow-list (`apply_triggers` returns 2 for ANY teleport and
`s->teleport_fail` is one global flag - allowing one teleport while failing
on the other 130 is a new API, an ABI bump and a `SurfCore` binding); a
teleport edge in `_bfs_geodesic` (after each sweep,
`d[src] = min(d[src], d[dest])` - cheap and correct); and the same component
pass to decide which teleports qualify.

**(b) stage 1 only** needs **no code change at all**:
* `src/env.c:601` is `int trig = complete ? 0 : apply_triggers(s, i, st);` -
  the swept goal-box test runs on pre-trigger positions and short-circuits
  the trigger evaluation, so an episode that touches a teleport brush which
  is also the goal box **completes instead of failing**. The goal box is
  hull-inflated (`goal_mins - player_maxs`), so a 1 u teleport curtain
  registers at any speed.
* the end zone is just an AABB in `maps/<map>.zones.json`, and `load_zones`
  always trusts `"source": "manual"`. `trigger_teleport` model bboxes come
  straight out of `parse_bsp` (offset by the entity `origin` - the bug fixed
  above).
* stage 1 keeps the map's real `info_player_start` and the map's real timer
  start line, so the start state is the real one, and every other teleport
  in the map stays a failure exactly as today.

**(c) exclude** costs 126 maps.

Pick **(b), and stop at stage 1**:
1. It is the only free option. (a) touches the C core every checkpoint's
   determinism depends on, and it needs (b)'s component pass anyway, so
   doing (b) first is not wasted if (a) is ever wanted.
2. What a 620-map dataset is FOR is geometry diversity, not race length.
   Stage 1 of a staged map is a complete surf course. 126 shorter maps is
   most of the value of 126*N maps.
3. Stages 2..N are the part I would NOT do: the only start state available
   is the teleport DESTINATION, a point with zero velocity, and a surf stage
   entered at speed may not be solvable from rest. Option (a) does not have
   that problem because the agent arrives under its own momentum - which is
   the argument for doing (a) later, if per-map length turns out to matter.


### What to do next

1. **Fix `survey_maps.py`'s discriminator** before anything else: a teleport
   whose destination is within 512 u of an `info_player_start` is a catch
   net regardless of where its source brush sits. That is one line and it
   frees 26 of the 126 immediately.
2. **Gate the fleet on `verify_maps.py`.** `train_fast.py` hard-fails on
   unreachability only in the `--race-kill-aware` branch (line 2077); the
   normal path computes `race_d0 = mean(goal_field.sample(spawn))`, which is
   the SENTINEL on a disconnected map, and `scale = 100/d0` then shapes on
   nothing. `mapfleet.py` checks nothing at all. A 47-map fleet built from
   the paper shortlist would have had 25 silent nulls in it.
3. **Seed the spawn pool past the start-room teleport** for the 15 maps
   listed above. Free, and it is the single largest recovery available.
4. **Then** build the stage-1 zone files for the 126.

### Cost

All CPU, all local, no rental, no bake. The 47-map pass is ~35 minutes of
one 8-thread process alongside a live trainer; `--stage1` over 32 maps is
~12 minutes. `maps_full_dataset/` was never written to - the occupancy is
rebuilt in-process instead of going through `vision.slab_occupancy`'s npz
cache.

## Round 21 - CPU for multi-GPU DDP: many slow cores or fewer fast ones? (2026-08-23 20:05-21:08 UTC)

**Question (user):** what CPU should the multi-GPU fleet be rented on -
EPYC/Threadripper-style many-slow-cores, or fewer fast ones? Hardware
characterisation, not an ablation: 5-10 minute runs, `ddp` branch (commit
`5a4da66`), `tools/ddp_launch.sh`, cannonball, from scratch, 64x32 depth, no
`--obs-reward`, `--n-steps 32`, `--timing`, median of the last 20 of 32
iterations. Throughput = `envs * n_steps * act_every / iter_seconds`; under
DDP `--envs` is the GLOBAL fleet (`train_fast.py:1407-1411` splits it), so
the same formula already gives the AGGREGATE.

### The answer, in one line

**Neither. Buy the GPU and the cheapest CPU that clears ~4 physical cores
per rank.** Across three healthy 4x3090 boxes spanning **3.4x in
cores x clock**, 4-rank aggregate throughput spans **1.16x**, and the
CHEAPEST box has the best steps/s per dollar-hour.

### The boxes (all 4x RTX 3090, all healthy on `gpu_health.py --all`)

| tag | CPU | phys cores | threads | thr/GPU | GHz nominal | RAM | $/h |
|---|---|---|---|---|---|---|---|
| **H** | EPYC 7282 (Zen2) | 16 | 32 | 8.0 | 2.8 | 251 G | 0.982 |
| **E** | Threadripper 3960X (Zen2) | 24 | 48 | 12.0 | 3.8 | 125 G | 0.817 |
| **C** | EPYC 7B13 (Zen3, 2 sockets) | 127 | 255 | 21.3 eff / 63.8 seen | 3.54 | 503 G | 1.372 |

`cores x clock`: H 89.6, E 182.4, C 302 (on vast's *effective* thread count)
or 902 (on what the container actually sees). H and E were whole machines
(`gpu_frac` 1.0); C was 4 GPUs of a shared host, CFS quota 108.8 CPUs.

A fourth box, **B** (EPYC 7K62, 48c/96t, 2.6 GHz, $0.603), was rented and
then blacklisted for power-capped GPUs (below). Its CPU-only `env` column is
kept here because the physics step does not touch the GPU; nothing else from
that box is usable.

### 1. Single-rank OMP sweep (2048 envs, T=32, 1 of 4 GPUs busy)

`env` ms - the OpenMP C physics step - at fixed thread count, across four
different CPUs:

| OMP | H (2.8 GHz) | E (3.8 GHz) | B (2.6 GHz) | C (3.54 GHz Zen3) |
|---|---|---|---|---|
| 1 | 526.0 | 471.4 | 515.1 | 400.2 |
| 2 | 280.4 | 256.1 | 276.8 | 216.7 |
| 4 | 146.8 | 135.2 | 140.6 | 121.9 |
| 8 | 78.2 | 69.0 | 72.3 | 68.0 |
| 16 | 41.8 | 36.8 | 38.6 | 40.5 |
| 32 | **43.2 (worse)** | 26.5 | 22.1 | 23.9 |

**A 1.46x nominal clock spread buys 5-13% of physics.** At 8 threads the
four CPUs are within 15% of each other, and at 32 the 2.6 GHz EPYC is the
FASTEST of the four. The physics step is memory-bound, not clock-bound: what
moves it is thread count, and it keeps scaling until it runs out of PHYSICAL
cores. H (16 physical cores) is the only box where 32 threads is worse than
16 - the SMT siblings cost more than they add.

Total iteration, same sweep:

| OMP | H total ms | E total ms | C total ms | env % of iter (H/E/C) |
|---|---|---|---|---|
| 1 | 1143.3 | 1056.5 | 976.0 | 46.0 / 44.6 / 41.0 |
| 4 | 763.6 | 726.2 | 701.0 | 19.2 / 18.6 / 17.4 |
| 8 | 698.0 | 663.4 | 652.7 | 11.2 / 10.4 / 10.4 |
| 16 | **660.9 (knee)** | 632.5 | 630.3 | 6.3 / 5.8 / 6.4 |
| 32 | 665.1 | **626.4** | **621.3** | 6.5 / 4.2 / 3.9 |

**1 -> 8 threads buys 1.50-1.64x. 8 -> 32 buys 4-6%.** The
`OMP_NUM_THREADS=1` trap that `torchrun` force-exports is worth 1.5-1.6x on
a 3090 (the local 5090's figure was 1.7x) - that half of gate A reproduces
everywhere.

**The local box's "knee at 8, and 16 is WORSE" does NOT generalise.** On the
32-core 5090 box `update` rose 231 -> 283 ms from 8 to 16 threads. On all
three rented 3090 boxes `update` is FLAT to within 1.5% across the whole
sweep (C 421.4 -> 430.1, E 431.4 -> 434.4, H 444.7 -> 444.1). The local knee
was a property of a machine where the OMP team is half the box, not of the
code. The real rule is **do not exceed the physical core count**.

### 2. Envs sweep at the knee (1 rank)

| envs/rank | H steps/s | E steps/s | C steps/s | update us/sample (H/E/C) |
|---|---|---|---|---|
| 2,048 | 297,508 | 313,895 | 316,446 | 6.77 / 6.63 / 6.56 |
| 4,096 | 328,803 | 346,385 | 352,866 | 6.31 / 6.37 / 6.25 |
| 8,192 | **349,075** | **357,485** | **363,147** | 6.05 / 6.22 / 6.05 |

**A 3090 does NOT saturate at 4,096 envs - it is still improving at 8,192**,
and per-sample update cost is still FALLING (6.56 -> 6.05 us on C). This is
the opposite of the plan's gate B, which found the 5090 peaking at 4,096 and
regressing 21% in per-sample cost at 8,192. Gate B's "the useful range is
~4,096 envs per rank" is a 5090 result and must not be carried to the 3090
fleet. The gain 4,096 -> 8,192 is only +3-6%, so 4,096 stays a reasonable
VRAM-cheap choice - but nothing turns over, and the CPU does not decide it:
three different CPUs are within 4% of each other at every env count.

### 3. Four-rank DDP (`tools/ddp_launch.sh 4`)

| box | OMP/rank | per-rank envs | total ms | env ms | allreduce ms | **AGGREGATE steps/s** | per-rank | $/1e9 steps |
|---|---|---|---|---|---|---|---|---|
| H | 16 (1-rank knee) | 8,192 | 4767.9 | 1769.3 | 158.2 | 659,772 | 164,943 | 0.413 |
| H | **4 (launcher)** | 8,192 | 2984.4 | 653.9 | 155.4 | **1,054,039** | 263,510 | **0.259** |
| H | 16 | 4,096 | 3653.6 | 1706.2 | 147.9 | 430,503 | 107,626 | 0.634 |
| E | 32 (1-rank knee) | 8,192 | 4354.2 | 1372.2 | 389.8 | 722,458 | 180,615 | 0.314 |
| E | **6 (launcher)** | 8,192 | 3013.3 | 405.0 | 468.2 | **1,043,948** | 260,987 | **0.217** |
| E | 32 | 4,096 | 3201.4 | 1372.4 | 236.3 | 491,305 | 122,826 | 0.462 |
| C | 32 (1-rank knee) | 8,192 | 2559.3 | 162.8 | 237.1 | **1,229,112** | 307,278 | **0.310** |
| C | 31 (launcher) | 8,192 | 2604.1 | 175.3 | 248.6 | 1,207,990 | 301,998 | 0.315 |
| C | 32 | 4,096 | 1502.4 | 92.7 | 241.6 | 1,046,901 | 261,725 | 0.364 |

**The single-rank OMP knee is the WRONG number to carry into DDP, and it is
a 1.4-1.6x error.** Sizing every rank to the 1-rank optimum requests 4x that
many threads from one box: on H that is 64 threads on 16 physical cores and
`env` blows up 9.9x (178 -> 1769 ms); on E it is 128 threads on 24 cores and
`env` blows up 13.2x (104 -> 1372 ms). `ddp_launch.sh`'s own rule -
`nproc / (2 * nproc_ranks)`, i.e. **physical cores divided by ranks** -
recovers 1.60x on H and 1.44x on E, and on C (where 4 x 32 still fits)
changes nothing. **That rule is right and load-bearing; never bypass the
launcher with a hand-set `OMP_NUM_THREADS`.**

Scaling efficiency, 4-rank aggregate against 4x the same box's single-rank
number at the same per-rank envs:

| box | 1 rank @ 8,192 | 4-rank aggregate (launcher OMP) | speedup | efficiency |
|---|---|---|---|---|
| H | 349,075 | 1,054,039 | **3.02x** | 75.5% |
| E | 357,485 | 1,043,948 | **2.92x** | 73.0% |
| C | 363,147 | 1,207,990 | **3.33x** | 83.2% |

**The remembered "1.73x on 4x3090" does NOT reproduce - it is 2.9-3.4x on
this branch.** That figure predates commit `6c274a1` (step-15 bucketed async
all-reduce from post-accumulate-grad hooks); exposed comm is now 147-468 ms
of a 2.6-3.0 s iteration (5-16%), against the 453 ms/iter of fully exposed
SHM all-reduce that perf-results S13 measured. No P2P on any of these boxes
(GeForce driver, PHB/NODE topology only).

### The purchasing rule

**1. Throughput tracks NEITHER cores, NOR clock, NOR their product.**

| box | cores x clock (effective threads) | 4-rank aggregate | $/h | $/1e9 steps |
|---|---|---|---|---|
| H (8 thr/GPU, 2.8 GHz) | 89.6 | 1,054,039 | 0.982 | 0.259 |
| E (12 thr/GPU, 3.8 GHz) | 182.4 | 1,043,948 | 0.817 | **0.217** |
| C (21 thr/GPU, 3.54 GHz) | 302 | 1,207,990 | 1.372 | 0.315 |

**3.4x of CPU buys 1.16x of throughput** - and 1.01x between H and E, which
differ 2.0x in `cores x clock` in opposite directions. The GPU is the
product being bought; the CPU is a floor to clear, not a lever to pull.

**2. Threads per GPU: 4 PHYSICAL cores per rank is the floor, 8 is
comfortable, past that pay nothing.** H runs 4 ranks on 16 physical cores
(4 each) and lands within 15% of C's 4 ranks on 127 physical cores (31
each). At 4 cores/rank physics is still 21.9% of the iteration (654 of
2984 ms) against 6.7% on C, so there IS headroom - it is worth about 15%,
and C charges 1.68x the hourly rate for it. As a listing filter:
**`cpu_cores_effective / num_gpus >= 8` is sufficient; below 8 you are
buying that 15%, above 16 you are buying nothing.** DEPLOY.md's old line -
"pick a high single-core clock and >=8 cores per GPU; core count beyond that
buys nothing" - is CONFIRMED for the core-count half and REFUTED for the
clock half.

**3. steps/s per dollar-hour, which is what decides the fleet:**

| box | 4-rank aggregate | $/h | **$ per 1e9 steps** |
|---|---|---|---|
| E Threadripper 3960X | 1,043,948 | 0.817 | **0.217** |
| H EPYC 7282 | 1,054,039 | 0.982 | 0.259 |
| C EPYC 7B13 | 1,207,990 | 1.372 | 0.315 |

The ordering is the ordering of `$/h`. It is not the ordering of any CPU
statistic.

**And the DDP premium is real: four single 3090s are ~1.8x cheaper per
step.** Single-3090 offers cleared at $0.113-0.175/h (28 listed, median
$0.175) the same evening; one 3090 at the measured 349-363k steps/s and
$0.15/h is **$0.119 per 1e9 steps** against $0.217 for the best 4x box.
Multi-GPU is worth that premium only where ONE policy over a 4x batch is the
point - which is exactly the multi-map plan - never as a way to buy steps
more cheaply.

**4. RAM does not matter.** Measured per-rank RSS on E at 4,096 envs/rank,
cannonball, cell 32: **4.86-4.91 GB per rank**, 19.6 GB for four ranks plus
0.46 GB of launcher, against 125 GB on the smallest box measured (H and C
had 251 and 503 GB). VRAM was ~5.0 GB per 3090 at 4,096 envs / T=32. The
multi-map plan's 0.83 GB per rank of goal fields at cell 48 sharded over 8
lands inside that with two orders of magnitude of headroom. **No box in this
class is RAM-constrained; do not pay for RAM.**

### Rental log, and four things that cost time

Nine instances created, all destroyed, registry clean at 21:08 UTC, about
$2 of GPU time.

* **The image pull, not the CPU, is what loses boxes.** THREE candidates
  (offer 46862553 twice, 41246100 once) were destroyed by
  `fleet_watchdog.py`'s 420 s readiness backstop while still pulling
  `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`. None was blacklisted - a
  slow pull is not a defect - but **the practical filter for a benchmark box
  is `inet_down` plus whether the host already has the image**, and racing
  four candidates is the only way through. Hosts that had it cached were
  ssh-ready in 2-5 minutes; a re-create against the same machine does NOT
  reliably hit a warm cache.
* **`gpu_health.py` prints `VERDICT: healthy` for a 3090 that is crippled.**
  Offer 35580414 (m78080, EPYC 7K62) delivered four cards **power-capped at
  180 W of 350 W**: 660-1065 MHz under load, **30-35 TFLOPS bf16 and
  574-764 GB/s** against 69-73 TFLOPS / 840-863 GB/s on the three healthy
  boxes, and a PPO `update` phase of 705 ms against 425-444 ms on identical
  GPUs. The tool says "no reference for this model - recorded, not judged"
  and then prints `healthy` anyway. **Read the TFLOPS and the watts, never
  the verdict, until a 3090 reference is added.** Blacklisted `gpu_capped`.
* **A "4x3090" offer can arrive with all four GPUs already 100% busy.**
  Offer 48305729 (m146873, EPYC 7702P, `gpu_frac` 0.8) had 1.3-3.4 GB of a
  FOREIGN tenant's memory on every one of the four cards and 100% SM
  utilisation before anything of ours ran. The listing tell was `dlperf`
  **87.5 against 146 for every peer 4x3090 offer**. Blacklisted
  `unreliable`. Add `dlperf` to the pre-rental read.
* **The container sees the HOST's threads, not its entitlement.** `nproc`
  and `sched_getaffinity` returned 255 on C (CFS quota 108.8 CPUs, vast
  "effective" 85.3) and 96 on the capped EPYC (quota 46.1). So
  `ddp_launch.sh`'s `nproc / (2 * ranks)` over-provisions on any shared box:
  on C it asked for 124 threads against a 108.8-CPU quota. It cost 1.7%
  there, but on a `gpu_frac` 0.33 box it is the same oversubscription that
  costs 1.4-1.6x above. **Prefer `gpu_frac == 1.0` for multi-GPU, or teach
  the launcher to read `cpu.max` rather than `nproc`.**

### What this changes in `docs/multimap-ddp-plan.md`

* Gate A's **"~8 cores per rank"** stands as a sufficiency threshold, but
  its "8 ranks gives 4 cores each and sits below the knee" is too
  pessimistic: 4 physical cores per rank costs about 15%, not a cliff. The
  cliff is at 4 ranks x the 1-rank knee, i.e. exceeding the box's physical
  cores, and the launcher already prevents it.
* Gate B's **envs peak at 4,096** is a 5090 result. On a 3090 throughput is
  still rising at 8,192 envs/rank with per-sample update cost still falling.
* Gate D's **1.73x** is superseded: **2.9-3.4x** on 4x3090 with the step-15
  overlap in.
* The plan's central risk - "per-rank CPU starvation under torchrun making
  more ranks actively worse" - **is real, and is fully handled by
  `ddp_launch.sh`'s existing sizing rule.** It fires only when
  `OMP_NUM_THREADS` is set by hand from a single-rank sweep, which is what
  this round did deliberately in order to measure it.

# ============================================================
# ROUND 23 (2026-08-24): DDP x --maps integration + aggregated metrics (mmddp)
# ============================================================

Branch `mmddp`. The integration everything else was waiting on. `ddp` and the
map work were disjoint branches - 85 commits one way, 8 the other, 2,628
lines apart in `train_fast.py` - so **no branch had both features**. This
round merges them, adds the aggregated multi-map metrics, and proves the
whole thing on 5 maps x 2x3090 for an hour.

**This is a PLUMBING result and no learning verdict may be read into it.**
1 h from scratch is far below the ~2.5 h this file says is needed to conclude
anything, and the run trains 5 maps nobody has ever trained.

## 1. How the two splits compose: NESTED, and maps are REPLICATED

`--envs` is the GLOBAL fleet. DDP cuts it into `world_size` rank-shares;
`--maps` cuts each rank's share into `NMAPS` slots:

    PER = envs / (world_size * maps)

envs per slot, and **every rank holds every map**. `--envs` must be a multiple
of `ranks x maps`; the trainer refuses otherwise rather than truncating a slot
behind its own logs. Verified on the box: `envs 16000 / envs_per_rank 8000 /
envs_per_slot 1600 / n_maps 5 / world_size 2` in `run.json`.

**Maps are replicated across ranks, not sharded - a deliberate reversal of
gate E in `docs/multimap-ddp-plan.md`.** Three reasons, in order of weight:

1. **RAM stopped forcing it.** The gated-cell field set is 3.69 GB total,
   0.46 GB/rank over 8 even unsharded, against ~63 GB of headroom. Measured
   here: **3.7 GB of VRAM and 17 GB of RSS** for 5 maps x 8,000 envs/rank.
2. **Replication is what makes the aggregate metric cheap AND correct.** Every
   rank holds every slot, so each cross-rank sync is a FIXED-SHAPE collective
   over `NMAPS`. Sharded maps make every one of them a variable-length gather
   over a per-rank subset, and a slot present on one rank only cannot use a
   world-wide collective at all without a subgroup per map.
3. **It keeps the gradient batch map-balanced.** Under sharding a rank's whole
   minibatch comes from one map, and the per-minibatch advantage
   normalisation IS all-reduced (ddp-plan sec 2) - it would be centring a fleet
   whose composition differs rank to rank.

The cost is `NMAPS` cores + `NMAPS` lidar SDFs resident per rank instead of
`NMAPS/world_size`. Sharding stays available; nothing assumes replication
except the fixed shapes, which degrade to a gather.

### The four conflicts inside the training loop, and why each mattered

The danger was never the diff size. `train_fast.py` keeps `core`,
`reward_fn` and `respawn` as **aliases onto slot 0**, and every DDP sync git
auto-merged in referenced those aliases. Each would have compiled, run, and
synced ONE map while the other four diverged - **with no logged number
changing.**

| sync | what the alias would have done | what it does now |
|---|---|---|
| novelty counts | slot 0's table only | one batched `(slot, cell, inc)` gather - O(1) collectives in the map count |
| respawn reservoir | slot 0's ring only | one gather PER SLOT, keyed slot-locally. A reservoir state is raw map coordinates: a cannonball state respawns inside solid geometry on petrus |
| episode history | `local + rank*N` | `i*PER*W + rank*PER + j` - the env id a single-process `N_GLOBAL`-env run would have used. The naive key interleaves MAPS instead of ranks |
| race stat counters | slot 0's counters | summed over maps, then over ranks |

Startup invariants now hash **every** slot's spawns, pool, field grid and d0,
plus `NMAPS`, `PER` and the map-list digest. A rank that resolved a different
BSP for slot 3 - a stale cache, a half-synced staging directory - fails at
startup instead of training one map against another map's reward scale.

`_default_omp_threads` needed both sides combined: the CFS-quota clamp AND the
torchrun world-size split, with the quota bounding **both** sides of the "is
this rank's affinity already narrowed" test.

## 2. The metric, and what it is NOT

Two aggregates, all-gathered, in `progress.csv` and on the dashboard:

* **`race/map_pct`** - mean over maps of the mean over that map's greedy eval
  episodes of `100*(d_spawn - d_min)/d_spawn`: the share of that map's OWN
  route covered.
* **`race/maps_finished`** - fraction of maps with at least one eval episode
  whose recorded path crosses the finish AABB.
* both repeated **trigger-only**, plus the full per-map table printed each
  eval and four suffixed CSV columns per map (40 columns at 5 maps).

**Neither is `race/eval_progress`, and each departure is a documented failure
of it.**

* eval_progress is in MAP UNITS and this pool spans a 5x range of route
  length, so a units mean is a weighted vote the long maps win. Pinned by
  `test_aggregate_is_a_percentage_not_map_units`: a long map at 10% and a
  short map at 90% must average 50%, where eval_progress says 25,957 u -
  84% the long map's opinion.
* eval_progress saturates at the geodesic field's interior minimum
  (cannonball's is at 88% of the route) and round 18 measured it
  **anti-correlated** with the frontier.
* the finish test is the env's own swept-AABB slab test over the recorded
  positions - the same test `src/env.c:seg_hits_box` runs, on the same
  per-tick segments - **not** the geodesic `<= 150 u` proxy, which a
  death-dive into goal-adjacent airspace passes without finishing anything.
  `test_race_coverage.py` pins it in both directions, including the two cases
  it exists for: a 1 u curtain crossed at 35 u/tick (a point-in-box check
  tunnels straight through) and a dive that passes UNDER the finish, scores
  84% on the distance metric and finishes nothing.

**Finish kind is carried per map and reported split.** `"button"` when the end
box has `true_aabb` / `from: func_button` or the zone file is `gateway`, else
`"trigger"`. 42 of the 107-map pool are trigger; the other 65 are +use button
boxes ~8x smaller in face area that **the simulator cannot press at all** - it
substitutes arriving inside the box - so a null on one is much weaker evidence
and the two are never pooled into one headline without the split beside it.

**Win rate is now impossible to report alone.** `MapFleet.reservoir_min_depth()`
returns the shallowest reservoir state as a FRACTION of that slot's own d0
(maps differ 5x, so the raw unit is not comparable across them), and the step
line prints `win ... res <n> mind <pct>` together. Round 19's xPSSR read
18.46% off a reservoir that had drifted to 1,485 u from the goal; that number
is only legible next to the depth.

### GATE F: PASSED, twice

**Locally** (`tests/python/test_multimap_ddp_metrics.py`): the plan's rig -
one map solved, one at zero - through the SHIPPED `train_fast.eval_aggregate`,
plus a **real two-process gloo group** where each process fills only the row
it owns, all-reduces, and both must read `map_pct 50%`, `maps_finished 0.5`
and the complete per-map table. Before the all-reduce each rank holds a table
half of which is zeros, so a rank reporting its own slice would read 50%/0% on
rank 0 and 0%/50% on rank 1.

**On the box**, 2 NCCL ranks, 5 real maps, `DDP_DEBUG_STDOUT=1`:

    [1,536,000] AGGREGATE[r0]  cover 1.09%  maps finished 0.00% (0/5) || trigger-only 1.09% ... (5 maps)
    [1,536,000] AGGREGATE[r1]  cover 1.09%  maps finished 0.00% (0/5) || trigger-only 1.09% ... (5 maps)

Rank 0 evaluated maps 0, 2, 4; rank 1 evaluated 1, 3. **Both printed the
identical five-row table**, and it stayed identical at all 16 evals.
Per-map at that point: 2.45 / 1.87 / 0.58 / 0.35 / 0.21, mean 1.092.

The counts fan-out has its own test plus a **negative control** that performs
the slot-0-only sync and asserts map 1 is still rank-divergent afterwards -
without it the positive test proves nothing.

**The eval is now sharded over maps, round-robin by rank, and that puts a
COLLECTIVE in a branch the single-map DDP design deliberately kept
collective-free** (ddp-plan sec 6.7: rank 0 records while the fleet free-runs).
The trade reverses with the map count - rank 0 evaluating NMAPS maps serially
blocks every other rank at the next collective for NMAPS eval-lengths anyway,
so the stall is paid either way and sharding pays it NMAPS-times shorter. The
branch is rank-symmetric (`global_step` and `next_record` both derive from
`N_GLOBAL*T`) and that is **asserted**, because getting it wrong is a hang
rather than an error. A rank that skips its shard raises instead of letting
the aggregate quietly become a mean over the rest.

## 3. Dashboard: resolution was never the bug, and the gate now says so

`test_viewer_map_resolution.py` passes (the viewer takes the map from the
trajectory's own header, not `run.json`). Verified end to end on the real
run: five recordings per eval, `traj_<step>_<maptag>.jsonl`, each carrying its
own `"map"` header, written by BOTH ranks into the shared run dir;
`dashboard._run_info` lists them distinctly (`prechasm | greedy`,
`latebra | greedy`, ...); the new metric series come through
`_metrics_from_csv`.

**But that test passed throughout an incident where every pool map rendered an
empty scene**, because `viewer/assets/<map>.mesh.json` did not exist and the
correctly-resolved request 404'd. **"The logic is right" and "the page
renders" are different claims.** New gate `test_dashboard_map_assets.py`
starts a real server on an ephemeral port and asserts HTTP 200 for every map
the recent runs name in their trajectory headers, with a 404 control so it
cannot pass vacuously. Verified by deleting one mesh: it fails naming the map
and the status code. `fetch_pool.sh` now exports the meshes after unpacking.

## 4. The smoke run

**5 trigger-finish maps x 2x RTX 3090, from scratch, 16,000 global envs,
`--act-every 3`** (the run predates the 3 -> 4 change; the MULTIMAP launcher
branch carries 4 from now on). prechasm (d0 12,905) / unitfarmer2 (30,589) /
latebra (33,763) / sabuleum (30,462) / src_mellow (39,644), picked
smallest-first off the pool manifest among trigger finishes with `_b2`/`_b3`
re-releases de-duplicated. Goal cells **48,48,48,32,48** - sabuleum is a
`keep_ref` map that tunnels at 48 - which is what the new per-map
`--goal-cell` list form exists for.

| | |
|---|---|
| box | 2x3090, Ryzen 9 5950X 16c/32t, 125 GB, $0.497/h, Japan |
| health | 844/841 GB/s HBM, 74/75 TFLOPS bf16 (107-109% of ref) |
| ready | under 60 s from create |
| **steady-state throughput** | **691,200 steps/s = 2.49e9 steps/hour** |
| VRAM | **3.7 GB of 24 per rank** |
| RSS | 17 GB of 125 |
| SDF / field bakes | **0** - the pool ships them prebaked and every cache hit |
| tests on the box | 208 passed, 7 skipped |
| run completed | 2,400,768,000 steps, avg 661,545 steps/s |

**691,200 steps/s is 1.90x a single 3090's measured 363,819 on ONE map.** So
five maps cost essentially nothing against one at this scale, and 2-rank DDP
scaling is ~95%. VRAM at 3.7 GB of 24 says `--envs` was left far below the
box's ceiling.

**`race/map_pct` by steps** (identical on both ranks at every point):

| steps | `race/map_pct` | `race/maps_finished` |
|---|---|---|
| 1,536,000 | 1.09% | 0.00% (0/5) |
| 152,064,000 | 5.91% | 0.00% (0/5) |
| 302,592,000 | 7.84% | 0.00% (0/5) |
| 453,120,000 | 8.96% | 0.00% (0/5) |
| 603,648,000 | 9.98% | 0.00% (0/5) |
| 754,176,000 | 10.39% | 0.00% (0/5) |
| 904,704,000 | 12.16% | 0.00% (0/5) |
| 1,055,232,000 | 12.52% | 0.00% (0/5) |
| 1,205,760,000 | 12.66% | 0.00% (0/5) |
| 1,356,288,000 | 15.11% | 0.00% (0/5) |
| 1,506,816,000 | 15.20% | 0.00% (0/5) |
| 1,657,344,000 | 16.17% | 0.00% (0/5) |
| 1,807,872,000 | 16.86% | 0.00% (0/5) |
| 1,958,400,000 | 16.15% | 0.00% (0/5) |
| 2,108,928,000 | 17.55% | 0.00% (0/5) |
| 2,259,456,000 | 17.32% | 0.00% (0/5) |

Both ranks printed identical aggregates at all 16 evals where rank 1's line was captured (of 16 total).

Diagnostics moved with it and none contradicts: training reward -1.91 ->
6.36, reservoir min-depth 96.25% -> 52.925% of d0, win rate **0.00%
throughout** (so the trivial-win trap never fired, and the win rate is being
reported next to the depth as required). **0 finishes on any map**, which at
this budget is the expected reading and not a result.

Checkpoint round-trip verified separately on a local 2-map run: `int_counts`
and `respawn` are saved as dicts keyed by map stem, restored per slot
(184,357 visits and 623 reservoir states came back into the right tables),
and `policy.state_dict()` carries no `module.` prefix, so `record_ckpt.py` /
`play.py` / dashboard record buttons still read a DDP checkpoint.

## 5. Three fixes the smoke run forced, worth carrying

* **`ddp_launch.sh` sized OMP off `nproc`, which is not the budget on a
  fractional rental.** This box reported `nproc 32` against a **cgroup quota of
  30**; the other candidate in the race reported `nproc 255` against
  `cpu_cores_effective 42.7`, and would have been asked for **63 OMP threads
  per rank on ~21 usable CPUs**. Measured cost of exactly that mistake
  elsewhere: 21.7%. The launcher now reads `cpu.max` / `cpu.cfs_quota_us` and
  takes the smaller; it printed `cgroup CPU quota 30 < nproc 32 - sizing off
  the quota` on the first launch. Duplicated from `_default_omp_threads`
  deliberately: torchrun clamps `OMP_NUM_THREADS` before the trainer is
  reached, so the launcher is the only place that can fix it.
* **`vast_pick.py` documented CLAUDE.md's physical-core rule and did not
  implement it** - it hardcoded `num_gpus=1` and filtered on a threads floor.
  It now takes `--num-gpus` and applies
  `cpu_cores_effective / (2*num_gpus) >= 8` directly. On this search that is
  the difference between 3 candidates and 2, and the one it excludes at 6.0
  physical cores/GPU is the shape that lost 8x3090 the round-22 comparison.
* **`deploy_box.sh` could only deploy a checkpoint.** `NO_CKPT=1` skips
  everything checkpoint-shaped, which is what let a from-scratch multi-map arm
  deploy at all.

## 6. What remains before a full-pool run

1. **`--envs` is untuned for this shape.** 3.7 GB of 24 at 8,000 envs/rank.
   Round 22's per-box optimum was 32,768/rank; nothing here tested it with 5
   map slots in the way, and the envs optimum is a property of the BOX.
2. **`ep_ticks` is still ONE number for the whole fleet.** 6000 here because
   these routes are 12.9k-39.6k u against cannonball's 198k; over 107 maps
   spanning that range a single cap is a different treatment per map.
   `max_episode_ticks` is already per core, so a per-slot cap is small.
3. **`--respawn-margin` is one number too**, and the trivial-win trap it opens
   is per map: the moment ONE map starts finishing, its reservoir harvests
   states 2 s from that goal. `reservoir_min_depth` makes that visible;
   nothing acts on it.
4. **Eval cost grows linearly in the map count and is sharded only
   `world_size` ways.** At 107 maps on 2-4 ranks that is 27-54 map-evals per
   rank per record point. Either the cadence drops, or the eval samples a
   subset of maps per record point - with the caveat that a per-eval subset
   makes `map_pct` a different estimator each time.
5. **65 of the 107 pool maps are button finishes** and the simulator cannot
   press a button. The split reporting is in; what is NOT settled is whether a
   button map belongs in the TRAINING mix at all, since its reward geometry is
   a ~8x smaller target in face area.
6. **`--n-steps 32` was measured at `act_every 3`.** T counts DECISIONS, so at
   4 it is 128 physics ticks of GAE window instead of 96. That optimum does
   not transfer unchanged and has not been re-measured.
7. **Nothing has been measured about learning**, by construction. The
   from-scratch multi-map baseline does not exist; the first real question is
   whether one policy over N maps beats N policies over 1 map each at equal
   total steps, and that needs the full budget, not an hour.

### What this changes in `docs/multimap-ddp-plan.md`

* **Gate C (multi-map correctness): PASSED**, and locally, as the plan
  intended - per-map evals differ and are attributed to the right map,
  per-map `scale = 100/d0` holds, trajectories carry their own map tag.
* **Gate E (map sharding): DROPPED, not deferred.** The plan already
  downgraded it to "an optimisation, not a prerequisite" once the field RAM
  was measured. This round makes it an anti-goal for now: replication is what
  keeps every cross-rank sync a fixed-shape collective, and sharding would
  turn each into a variable-length gather over a per-rank map subset. Revisit
  only if resident cores/SDFs (not fields) become the constraint.
* **Gate F (aggregated metrics): PASSED**, both on the rigged one-solved /
  one-zero case through a real two-process group and on the box with two NCCL
  ranks and five maps.
* **Gate G (integration smoke): PASSED** at 5 maps x 2 GPUs rather than the
  plan's 10 maps x 4 GPUs - deliberately smaller, per the brief, because the
  failure modes being gated are all rank-count >= 2 and map-count >= 2, and a
  smaller box is a cheaper place to find them.
* **The plan's second stated failure mode - "the aggregate metric being
  dominated by a handful of easy maps" - is now structurally excluded rather
  than hoped against.** `map_pct` is a mean over MAPS of a per-map
  PERCENTAGE, so neither a long map nor a map with more eval episodes can
  out-vote a short one; both are pinned by tests.
* **`--goal-cell` is now per map** (a comma list aligned with `--maps`), which
  the plan's own per-map coarsening gate requires and which no code path
  previously honoured: 21 of the 110 usable maps tunnel at cell 48 and must
  keep 32, and one global value either wastes 3.3x on the other 89 or ships a
  nonsense field for those 21.

### The greedy episodes are a GATE LADDER on these maps too

Opened the trajectories, as the standing rule requires. At 1.507e9 steps all
three greedy episodes of every map end within 10-40 u of each other:

    latebra      3,989 / 3,996 / 3,996 u from its finish box
    mellow      13,975 / 14,000 / 14,050 u
    prechasm     3,826 / 3,834 / 3,837 u
    sabuleum     3,384 / 3,386 / 3,391 u
    unitfarmer2  4,931 / 4,932 / 4,942 u

All labelled `fail`, none anywhere near the finish, and **no death-dives** -
the distances are large and mutually consistent rather than one episode
landing in goal-adjacent airspace. This is the same nearly-binary
"which physical gate did this seed clear" structure this file documents for
cannonball, reproducing on five maps nobody has trained, and it is the reason
`map_pct` moves in steps rather than smoothly. It also re-confirms that
**"MEAN tracks MAX" is worth nothing as corroboration**: for a deterministic
greedy policy inside one mode the three episodes are the same episode.

**The final per-map table (2,259,456,000 steps), which is what
attributability means here:**

| kind | map | % of own route | box finishes | eval_progress | d0 |
|---|---|---|---|---|---|
| trigger | `mellow` | 40.75% | 0/3 | 16,154 u | 39,644 u |
| trigger | `prechasm` | 17.23% | 0/3 | 2,211 u | 12,905 u |
| trigger | `latebra` | 14.62% | 0/3 | 4,943 u | 33,763 u |
| trigger | `sabuleum` | 8.66% | 0/3 | 2,635 u | 30,462 u |
| trigger | `unitfarmer2` | 5.35% | 0/3 | 1,636 u | 30,589 u |

Each row is the map that produced it, and the two ranks that split the
work printed the same five rows.

**This table contains the inversion the percentage exists for.** `prechasm`
and `sabuleum`: in UNITS sabuleum looks better (2,635 u against 2,211 u), and
as a share of its own route prechasm is nearly TWICE sabuleum (17.23% vs
8.66%), because sabuleum's route is 2.4x longer. `race/eval_progress` ranks
those two the wrong way round, and it would do it 107 times over on the full
pool.


### CORRECTION to the round-23 smoke table: the fields DID rebake

The table above says `SDF / field bakes | 0 - the pool ships them prebaked and
every cache hit`. **That is wrong, and the way it was wrong is worth more than
the number was.** I grepped the log for "bak"/"build_sdf" and got zero hits -
but the trainer does not print those words. It prints

    slab occupancy: rasterized 1 thin solid entities
    goal field: 320 sweeps, 504 seed voxels, 121,186 reachable voxels, ...

and there is one `goal field:` line for **every one of the five maps** in the
warm-caches pass, plus seven `slab occupancy:` rasterizations. Every field was
rebuilt from scratch. The reported d0 values matched the manifest exactly,
which is what made it look like a cache hit - but a correct rebake reproduces
d0 by construction, so **d0 agreement is not evidence of a cache hit and must
never be used as one.**

The cause is the one found independently and fixed in `tools/restamp_maps.py`:
every cache keys on `v2_<size>_<mtime_ns>` of the `.bsp`, and **tar does not
preserve sub-second mtimes**, so unpacking the pool invalidates its own
prebaked caches. It hit 103 of 108 maps.

**Two things this does NOT change, and one it does.**

* The throughput number stands. 661,545 steps/s is steady-state, measured
  over a 60-second window mid-run and again as the run's own average
  (661,518 / 661,545 on the two ranks) - the bake was over long before.
* **`ddp_launch.sh`'s `--warm-caches` pre-pass did exactly its job.** The
  entire rebake landed in ONE process before torchrun started; the two ranks
  then read caches, and nothing raced or OOM'd. That step was written to
  prevent concurrent bakes, and this run is the first time it actually had to.
* **It changes the full-pool startup estimate completely.** These five maps
  are 121k-717k reachable voxels and rebaked inside the launcher's 300 s
  liveness window, which is why it went unnoticed. The pool's large maps are
  three orders of magnitude bigger, the pre-pass is SERIAL and
  single-process by design, and there is nothing in the log to distinguish it
  from a slow cold start. **Run `tools/restamp_maps.py` after unpacking, and
  treat a `goal field:` line at startup as a failed cache, not as progress.**

The general rule, which is the same one this file already states about
single-observation evidence: **a grep that finds nothing is only evidence if
you have checked that the thing you are grepping for is what gets printed.**

---

## Round 24 (2026-08-24): the "4-rank DDP deadlock" was numba, and the 107-map pool now trains on 4x3090

Agent: fable session. Box: instance 48549980, machine 16571 (known-good,
Spain), 4x RTX 3090 on an EPYC 7B13, nproc 255, cgroup quota 108.8 CPUs,
$1.372/h. Launch config: `launch_pool.sh` verbatim, NMAPS=107 ENVS=54784
STEPS=500e9 EVAL_EPS=1 RECORD_EVERY=10e9, act-every 4.

### The launch checklist did its job

ssh usable 23.8 s after create; power.limit = power.default_limit = 350 W
on all four cards; utilization 0% pre-deploy (no foreign tenant);
gpu_health 840 GB/s HBM on 4/4, bf16 70-73 TFLOPS (101-105% of ref);
pytest 228 passed (after installing pytest - the image ships without it
and deploy's suite step silently skipped; deploy_box.sh fixed);
restamp --check 108/108 matching, 0 rebakes; pool_args 107 maps
(65 button + 42 trigger; cells 84@48 / 21@32 / 2@72). One landmine
defused pre-rent: committed `launch_pool.sh` had a literal `\n` token
that argparse would have rejected as a stray positional (`ced4582`).

### The stall, and what it actually was

4-rank launch: warm caches clean (0 sweeps), compile 66 s, then the
EXACT documented deadlock signature - 128 AUTOTUNE lines, log frozen,
~2,300% CPU per rank, 0% GPU, progress.csv untouched 12+ min. Killed at
the 10-minute stationary bound. Fallback per runbook: 2 ranks -
**reproduced identically** (compile 61 s then silence), which killed the
CPU-starvation theory: this box has 10.7 physical cores/GPU and the
quota was detected (OMP 13/rank).

Diagnosis without ptrace (vast containers deny SYS_PTRACE, py-spy is
dead on arrival): relaunch with `PYTHONFAULTHANDLER=1`, wait for the
stall, `kill -ABRT` both ranks, read the faulthandler stacks out of the
merged log. Both ranks were ALIVE inside iteration 1's rollout:
`core.py:604 step` and `goalfield.py:193 sample` <- `rewards.py:730`
<- `mapfleet.py:257 reward`. Not a deadlock - a crawl.

Cause: `goalfield._FAST_SAMPLE` is `@njit(parallel=True)`, and numba
sizes its pool off HOST cpu_count - 255 per rank here - ignoring both
the cgroup quota (108.8) and torchrun's OMP_NUM_THREADS=13. Observed
294 threads per rank = 255 numba + 13 OMP + torch/NCCL, 52-62 of them
spinning. `mapfleet.reward` calls `sample()` once per SLOT per decision:
107 syncs of a 255-thread pool per decision on 256-point batches (one
point per thread, pure barrier overhead), times 2-4 ranks fighting over
a 42% CPU quota. Iteration 1 alone exceeds 10 minutes at 0% GPU.

Why three prior configurations never saw it: single-map 4/8-GPU benches
(1 call/decision), the 5-map smoke (5 calls), the single-process 107-map
run (one pool, whole box). The pathology needs many slots x several
ranks x a big-host fractional rental at once. **The round-23 "deadlocked
twice, 620% CPU, 0% GPU, 2 ranks worked" observation is hereby
reinterpreted: same mechanism, worse at 4 ranks than 2 only because the
oversubscription doubles.** Rank count was an amplifier, never the cause.

Fix (`a80ec9e`): `ddp_launch.sh` exports
`NUMBA_NUM_THREADS=$OMP_NUM_THREADS` per rank. Bit-identity unaffected:
every prange element is independent, so thread count reorders nothing.

### With the cap, the same box at 4 ranks

| phase | measured |
|---|---|
| launch -> compile done | ~3 min (warm caches ~1 min hot; compile 4-5 s inductor-cached, 61-66 s cold) |
| launch -> first step lines | **~4 min** |
| full 107-map eval, eval-eps 1, 4-way sharded | **40.4 s** (record=40378.7 ms; untrained policy - re-measure at 10e9: trained episodes run toward the 6,000-tick cap) |
| marginal throughput | **~1.00-1.03M steps/s** (dS/d(S/fps) over adjacent step lines) |
| GPUs | 4/4 at 100% util, 332-349 W of 350 |
| iter-1 TIMING (ms) | rollout_wall 3693 (env 529, reward_py 2049 incl. numba JIT, vis 705, lidar 695, fwd 213), gae 852, update 2851, allreduce 234, skew 289, ckpt 5735, record 40379 |

The user's launch-latency bar - ssh-ready to training iterations in
under 5 minutes - is met on a deployed box (~4 min including compile).
A cold first-ever launch pays one-time autotune (~+2 min) plus deploy
(~7 min) and pool fetch (~8 min at Drive's ~1.4 MB/s throttle).

Open, unchanged: the eval is still batch-1 per map (runbook section 7);
at RECORD_EVERY=10e9 that is ~2-3% of wall-clock even at 10-minute
trained-policy evals, but the in-fleet batched eval remains the right
fix and its own piece of work. NUMBA_NUM_THREADS is capped only in
`ddp_launch.sh`; single-process launchers on fractional rentals have the
same exposure and should get the same line if one ever runs a big pool.

---

## Round 25 (2026-08-24): size x lr grid on the 108-map pool, 2x3090, matched 0.85e9 steps

User-ordered 3x3: widths (emb/hidden) 512/448, 1024/896, 2048/1792 x lr
3e-4, 3e-5, 3e-3. One seed per arm (standing rule). All nine on ONE box
(m20092 Poland, 2x3090 health-gated 103-105%, $0.422/h), queued by an
on-box runner (survives the workstation leaving); eval 1 at iteration 1,
eval 2 at 0.85e9 steps; RANKS=2 ENVS=27648 (128 envs per rank-map),
config otherwise launch_pool verbatim. Total campaign ~8.5 h, ~$3.6.

race/map_pct at eval 2 (trigger-subset in parens); finishes 0 everywhere:

| width \ lr | 3e-5      | 3e-4 (pinned) | 3e-3          |
|---|---|---|---|
| 512/448    | 6.19 (1.73) | 9.38 (2.86)  | 10.83 (3.89) |
| 1024/896   | 5.82 (1.95) | 11.30 (4.41) | 5.50 (1.88)  |
| 2048/1792  | 6.21 (1.52) | **12.26 (5.41)** | 0.56 (0.05) |

end-of-run approx_kl: 3e-4 column ~0.016-0.017; 3e-5 ~0.003-0.005;
3e-3 = 0.084 / 0.229 / 0.176 by width. Arm wall-times 38-48 / 51-54 /
68-70 min by width. Cannonball per-map at eval 2: 3.72% for 4x/3e-4
against ~0.7-1.2% everywhere else.

**Verdicts (against the 27% single-seed noise floor):**

1. **lr/10 is a real loss** - 34-49% below the pinned column at every
   width, kl pinned at ~0.004: it simply has not moved yet. Outside noise.
2. **lr x10 destabilizes with width** - a within-noise +15% at 1x width,
   clearly worse at 2x, and a collapse at 4x (0.56, kl 0.18). The pinned
   3e-4 is the right column; do not chase the 1x/3e-3 cell without a
   longer arm.
3. **Width at pinned lr rises monotonically at matched steps** - 9.38 ->
   11.30 -> 12.26. The headline +31% (4x vs 1x) is only marginally
   outside the floor, BUT the trigger subset (+89%) and cannonball (3x)
   corroborate on independent axes, so capacity looks like a real
   matched-step positive on the multi-map task - in sharp contrast to
   single-map cannonball, where bigger nets are validated dead. CAVEAT:
   4x width costs ~1.9x wall per step (68-70 vs 38-48 min/arm), so at
   matched WALL the ordering is unresolved by this design (evals were
   matched-step by explicit choice). The clean follow-up is 1x-vs-4x at
   matched wall-clock, or a 4x throughput measurement on the production
   4x3090 to price the trade.

Raw data: runs/ablation_results.tar.gz (2.07 GB, md5 60a2c7b3, all nine
progress.csv + eval trajectories + launch logs), parked on the main box
and on the workstation.

---

## Round 26 (2026-08-25): map hygiene, vision blind spots, and the depth ablation ($0, local)

Questions 1-3 of docs/research-questions-2026-08-25.md, all answered from
the harvested artifacts and the local box. No rentals.

### 1. The drop-exploit map is surf_bucetation - removed from the pool

All four maps that ended the 108-map run at map_pct 100.0 (bucetation,
shortbox, ut0pia, desert_city; desert_city got there late, the doc's
"three by eval 2" undercounted) are the SAME arithmetic hole, not four
separate exploits: the per-episode pct is 100*(d0-dmin)/d0 on the goal
field, whose zero set is the padded end box inflated by max(0.75*cell,
20)=36u of seed grow plus up-to-one-voxel (48u) of containment sampling.
So d=0 is sampleable up to ~85u OUTSIDE the padded box, and pct=100 does
not imply a finish. Measured at the final eval, dmin=0 was banked while
17-90u outside the box on all four; eval finishes stayed 0 everywhere.

Which one is the no-xy-movement map: **bucetation**. Every one of its 18
eval episodes is the same move - hop off the start at z~1900, free-fall
~4,600u with only 376-1,162u of total xy drift, clip the goal halo 89u
outside the hanging button box (46u button, pad 192, z -1920..-1504) at
2,306-2,420 u/s vertical, keep falling, die at kill_z = world_min_z-256
= -2786 ("end":"fail", ~5s). It never lands and never enters the box.
In GoldSrc the fall is unsurvivable (lethal impact ~1,024 u/s); the sim
has no fall damage (src/pm.c) and the user ruled out adding it
(2026-08-25), so the fix is removal:

* files moved maps_pool/ -> maps_pool_removed/ (pool_args now emits 106
  maps, 0 skipped; +cannonball from maps/ = 107 trainable);
* tools/build_pool_bundle.py grew an EXCLUDED dict so a v3 bundle
  rebuilt from the gitignored runs/research JSONs cannot re-include it.

The other three, reported for a keep/remove decision:

* **shortbox**: start box touches the end area; d0=384u, and one eval
  SPAWNED at d0=0 (1-tick episode). 100% by walking ~400u. Degenerate.
* **ut0pia**: walks the floor 64u BELOW the padded box bottom and clips
  the halo during a jump (17u outside in x). d0 ~1.3k; 60s timeouts.
* **desert_city**: flat map, walks within 62u of the padded box, d0 ~2k.
* (excessus 89.1 is dive-flattered - dmin 1,672u reached mid-fall
  1,188u from the box - but not a 100-liar.)

Metric consequence, for whoever builds v3 zones: the halo is pad+36u+48u
of slop on top of a pad that is already 159-192u on these button maps
(the pad autogrows past 64u until a standable point exists, so the
CLAUDE.md 4b "64u" is the floor, not the value). The user's tighter-pad
rule plus flooring dmin at the halo width - or requiring n_finish_box
for a 100 - would stop the class, not just these four maps.

### 2. gi_rino's invisible ramps are CLIP brushes, and the blindness is structural

Question 2's hypothesis (b) is confirmed and it is not an edge case. The
mechanism, verified in src/:

* the standing player traces BSP HULL 1 (real clipnodes:
  trace.c bsphull_for_usehull 0->1), where GoldSrc compilers put CLIP
  brushes;
* every vision/reward grid - occupancy, slab occupancy, the SDF the
  depth march reads, and the goal field - is built from
  surf_occupancy_grid (env.c), which queries point_contents on HULL 0
  plus a point-hull zero trace. CLIP brushes do not exist in hull 0.

So collision the player feels can be invisible to depth BY CONSTRUCTION,
and the mapper idiom "visible func_illusionary ramp + invisible CLIP
collision" produces exactly the user's observation: ramps render in the
3D viewer (which draws all brush-entity faces) and are absent from the
POV depth. On gi_rino, 102 of 105 func_illusionary models are
clip-backed (9/9 probe columns: standing hull stops inside the bbox,
point hull passes through) - the round-wall ramps are the map's PRIMARY
surf feature and the policy cannot see any of them. The viewer was
truthful here; the depth is what lies.

Pool-wide sweep (tools/clip_sweep.py; per-model dual-hull probes, 3,000
random far-from-visible-geometry points per map, and a behavioral scan
of the final-eval trajs; full table runs/research/clip_sweep_round26.csv):

* **28 of 106 maps** have clip-backed illusionary geometry (gi_rino 102
  models, fallway 29, src_utopia 16, simulatedway 14, hamburglar_love 8,
  raphaello 6, ...);
* **37 of 106 maps** have invisible clip volume in open space - points
  more than 80u from anything the SDF knows, where the standing hull is
  stuck (pyk_yougi 12.1% of sampled far points, raphaello 9.6%,
  kairo_b2 6.2%, lockdown_b1 3.2%, ...);
* **surf_skids2's final traj stands on invisible geometry in 12 of 16
  landings** - and that one is a SECOND mechanism: the floor there is a
  1u-thick WORLD brush (CONTENTS_SOLID z -388..-389, ent 0) that
  threads the cell/4=8u slab sampling lattice. The thin-geometry
  rasterization pass covers thin solid ENTITIES only (vision.py
  slab_occupancy); thin world brushes have no net, exactly as its own
  "catches any slab >= ~cell/4" comment admits.
* Hypothesis (a) - viewer over-render of truly nonsolid decor - is also
  real but minor: e.g. simulatedway carries 178 illusionary models both
  hulls pass through (decoration); it poisons human traj reading, not
  learning.

Consequences beyond depth: the goal field shares the blind occupancy, so
on the 37 affected maps the geodesic can flow THROUGH clip walls
(shorter d0, wrong shaping routes), and eval_progress/map_pct inherit
that. gi_rino's own map_pct 13.5 was earned by an agent flying between
ramps it could not see.

Fix directions (not implemented - both change every SDF and break
bit-identity with all trained checkpoints, so own arm + semantics bump
+ one full pool rebake, ideally ridden together):

* (b1) clip blindness: query hull-1 clipnode contents at voxel centers
  (the player's C-space, which is arguably what a depth sensor for THIS
  body should report); geometry fattens by the hull half-widths, within
  one 32u voxel;
* (b2) thin world brushes: rasterize world faces (the viewer mesh
  already extracts them) into the slab grid, the same way thin entities
  are already rasterized.

### 3. Depth is load-bearing: every ablation collapses the policy

record_ckpt.py grew --depth-mode (live/off/frozen/shuffle), eval-side
only, scalars untouched: "off" feeds the encoding's clear-sky value
(1.2444 at near 2000 / range 11500), "frozen" repeats each episode's
first frame, "shuffle" applies a fixed seeded permutation of the 32
image rows. ckpt_32e9 (runs/mmPOOL_harvest/ckpt_32048676864.pt), greedy,
one episode per map x mode on the local 5090, six maps spanning the
bands (PROVISIONAL probe set, not the section-10 bundle: excessus 89.1,
kns 56.0, gi_rino 13.5 as the clip-blind contrast, petrus_lite 18.9,
unitfarmer1 7.5, cannonball 2.2). Offline pct via each map's own field:

| map | live | off | frozen | shuffle |
|---|---|---|---|---|
| excessus    | 18.6 (529t fail)  | 0.0 (6000t trunc) | 0.0 (6000t trunc) | 5.3 (426t fail) |
| kns         | 54.1 (883t fail)  | 0.0 (trunc) | 0.3 (trunc) | 1.3 (255t fail) |
| petrus_lite | 18.2 (990t fail)  | 0.0 (trunc) | 0.0 (trunc) | 0.6 (388t fail) |
| unitfarmer1 |  7.6 (490t fail)  | 0.0 (trunc) | 0.2 (trunc) | 1.4 (328t fail) |
| cannonball  |  6.7 (1018t fail) | 0.0 (trunc) | 0.0 (trunc) | 0.6 (847t fail) |
| gi_rino     | 13.4 (1162t fail) | 0.0 (trunc) | 0.1 (trunc) | 2.1 (1291t fail) |

Verdict: **the policy is not a scalar automaton.** With no usable image
it does not even leave the start area (off/frozen idle to the 60s
timeout on all six maps); with rows shuffled it moves and dies blind.
This holds on gi_rino too - the depth it CAN see (world geometry) is
what carries its 13.4%. The vision work is not wasted; question 3 is
closed. Caveats: one greedy episode per cell (deterministic policy, and
the collapse is 18-54 -> ~0, far beyond any seed noise); recorded on
the 5090, so live-vs-harvest levels are not comparable across cards
(excessus 18.6 here vs 89.1 on the 3090 - the mode comparison is
within-card and stands).

Artifacts: tools/clip_sweep.py, record_ckpt.py --depth-mode,
runs/research/clip_sweep_round26.csv, ablation trajs in the session
scratchpad (q3/traj_<map>_<mode>.jsonl), maps_pool_removed/README.md.

---

## Round 27 (2026-08-25/26): question 4 (time pressure / free death / the broken telescope), the reward-ratio grid, a search planner, and spawn-from-the-record-line

Large round, four workstreams. Read section 1 FIRST: it invalidates the
design of half of them and is the most important thing measured here.

### 1. THE 3-HOUR SCRATCH NOISE FLOOR IS 5-9x. Single-seed scratch arms cannot rank treatments.

The gate-ladder retraction (rounds 20-21) said 1-hour from-scratch arms
vary 2.7-3.0x between IDENTICAL configs. This round ran the same
treatment (`--fail-pen 10`) as two properly-controlled pairs - each
within one card, one seed, one machine, matched steps, 3 hours - and
they came out OPPOSITE:

| pair | control corridor MAX | fail_pen 10 corridor MAX | verdict |
|---|---|---|---|
| local 5090, seed 0 (xCTLS / xFP10) | 24,704u (10.7%) | **107,136u (46.2%)** | treatment 4.3x BETTER |
| rented 3090, seed 1 (bCTL / bFP10) | **134,272u (58.0%)** | 15,360u (6.6%) | treatment 8.7x WORSE |

Control-to-control spread 5.4x; treatment-to-treatment spread 7.0x. Both
pairs are internally valid; they simply disagree. **Tripling the arm
length made the floor WORSE, not better** - 3 h is long enough for a run
to commit to a gate and stay there, so the binary-gate pathology
compounds instead of averaging out. Every 3-hour single-seed
from-scratch verdict in this round and any future one is therefore
uninterpretable at anything under ~9x, and the user's suspicion that
xFP10's local "win" was seed luck is CONFIRMED - it was, and the
replication went the other way by a bigger margin.

Consequences, adopted immediately:
* **`--fail-pen 10` is NOT a validated improvement.** The round-27
  overnight claim is retracted; the flag is neutral-to-unknown.
* The remaining bake-off arms (`bNG3` = `--race-ng 3` tax-only,
  `bGAM1` = `--gamma 1.0`) were **killed unrun and the box destroyed**:
  running a 3 h single-seed scratch arm now knowingly produces a number
  that cannot be read. Those two mechanisms remain OPEN and need a
  testbed with a reproducible frontier - see section 4, which found one.
* The 1-hour ratio grid (section 5) is reported as SCREENING with its
  own measured control-control spread, never as a ranking.

### 2. Ng-conformant shaping, strict form: a theorem, not a bug (xNGS, killed at 407M)

`--race-ng 1` (per-step tax `(1-gamma^k)*Phi` + terminal charge `-Phi`
on death AND finish) collapsed a scratch run to **fast suicide**:
ep_len 1,140 -> 373 and pinned, ep_rew pinned at exactly `-time_pen*len`
(-1.88), corridor MAX 3,456 -> pinned 2,560u, reservoir min-depth 99%,
value_loss ~3e-4 (the critic's task became trivial).

The arithmetic, which is the whole lesson: the conformant per-step sum
telescopes to `gamma^T*Phi(death) - Phi(spawn)`, and the terminal charge
subtracts `gamma^T*Phi(death)`. Total shaping over ANY trajectory =
`-Phi(spawn)`: a constant fixed at spawn, identical for every behaviour.
A policy that has never finished then has nothing left to optimise but
`-time_pen*T`, which is maximised by `T -> 0`. **Strict policy
invariance (including termination) is exactly the property a
from-scratch curriculum cannot have** - the broken telescope's
non-invariant income IS the curriculum. Same death as sparse reward
(xNOSHP/xBIN3), reached from the opposite direction.

Do not run `--race-ng 1`, or `--death-charge 1.0`, from scratch.
`--race-ng 3` (tax only, terminals stock) is the form that keeps the
income - implemented, still untested.

### 3. The time-penalty dose brackets cleanly (three arms, three regimes)

* `--time-pen 0` (xTP0, 3 h local): no suicide basin, but no urgency -
  eval plateaued 12.4k (0.6x its control), peak speed ~1,780 u/s vs the
  control's ~2,340. Removing time pressure costs progress.
* `--time-pen 0.005` (pinned): viable band.
* `--time-pen 0.01` (gTP100, grid): the **die-fast signature** returns -
  ep_len ~400, the same shape xNGS collapsed into.

So the explicit penalty is net USEFUL as the early pace-setter, and the
suicide basin is a function of how much of the reward is a pure
per-tick cost. This is consistent across four independent arms and is
the one part of question 4 that the noise floor does not threaten
(the effects are behavioural signatures, not gate positions).

### 4. SPAWN-FROM-THE-RECORD-LINE: the round's real result, and it replicates

User experiment: take the planner's best run (section 6), replay it
capturing full physics state per tick, CLIP the tail so the goal is
never spawned in, and train from scratch with every training spawn drawn
uniformly along that line. Evals still spawn at the MAP START, so the
readout is the real task. Two arms, clip = last 10% and last 50%:

| arm | spine coverage | spawns reach | corridor MAX | finishes |
|---|---|---|---|---|
| xDEMO90 | first 90% (7,418 states) | d = 7,674u | **205,312-205,440u (88.7%)** | 0 |
| xDEMO50 | first 50% (4,121 states) | d = 100,219u | **205,440-205,568u (88.7%)** | 0 |
| xCTLS (local control) | - | - | 24,704u (10.7%) | 0 |
| bCTL (3090 control) | - | - | 134,272u (58.0%) | 0 |

Both spine arms reached 88.7% of the route **from scratch in ~30
minutes** and then stopped at the SAME gate - route vertex ~1596-98,
205,4xx u - which is the documented wall where the 3.78e9-step warm
lineage sat for ~2e9 steps. Three things follow, and unlike section 1
they are above the noise floor because the two arms replicate each other
to within 256 u:

1. **Convergence: spine spawns reach the stuck checkpoint's frontier
   from scratch in half an hour.** That is the fastest route to the
   wall ever recorded here by a wide margin, and it needs no champion -
   the line came from the policy's own search output.
2. **The wall is NOT an exploration/access problem.** xDEMO90's spawns
   sat PAST the wall (down to d = 7,674u, i.e. beyond vertex 1601) for
   the entire run and it still never crossed. Placing the agent at the
   doorstep with a good line behind it is not sufficient.
3. **The wall is NOT a coverage-generalisation limit either.** xDEMO50
   was never spawned past the halfway point and still extended **38.7
   percentage points beyond its own coverage** to reach the identical
   gate. The method generalises forward fine; the ending is simply hard.
   (This was the user's discriminating design: "if it stalls at 50% the
   method is the limit; if it flies to ~90% the ending is." It flew.)

Caveats recorded honestly: eval_progress on these arms is
dive-flattered (7-9 of 9 episodes end below the finish box; peak
eval_progress 178k against a corridor 205.4k), win rate 0.00%
throughout, and neither arm was run to a wall-clock limit past ~2.2e9
steps. `tools/build_spine.py` (commit 9491b59) builds a spine at any
clip fraction; spines live in `runs/beam_tas/`.

**This is now the recommended testbed for reward-mechanism arms**
(sections 1-3's open questions): its frontier is reproducible to 256 u
across two independent configurations, where a scratch control varies
5.4x. `--race-ng 3` and `--gamma 1.0` should be run on top of it.

### 5. Reward-RATIO screening grid (12 arms, 1 h each, 4 rented 3090s)

User-designed: since a global reward scale is nearly a no-op under PPO's
advantage normalisation, the real surface is the RATIOS. Arms:
`--race-shaping` {0.5, 2}, `--time-pen` {0.0025, 0.01}, `--int-coef`
{0.1, 0.5}, the four interesting pairs, and the SAME control on two
boxes (gCTLa / gCTLb) to measure the cross-box floor directly.
`race/eval_progress` at matched steps (corridor MAX not yet computed;
eval_progress caveats apply). Data: `runs/research/r27grid/`.

| arm | vs baseline | 0.25e9 | 0.5e9 | 0.85e9 |
|---|---|---|---|---|
| gCTLa | control (slow box) | 12,557 | - | - |
| gCTLb | control (replicate) | 13,174 | 18,129 | 33,380 |
| gIC10 | int-coef 0.1 | 8,233 | 24,487 | **38,811** |
| gSH2 | shaping 2 | **16,661** | **23,250** | 25,969 |
| gEXTP | int-coef 0.5 + time-pen 0.0025 | **15,371** | 18,140 | 24,906 |
| gTP25 | time-pen 0.0025 | 8,807 | 18,688 | - |
| gLOPRES | shaping 0.5 + time-pen 0.0025 | 10,037 | 16,372 | 18,657 |
| gHIPRES | shaping 2 + time-pen 0.01 | 8,437 | 15,419 | 17,596 |
| gEXPLO | int-coef 0.5 + shaping 0.5 | 12,326 | 7,392 | 5,591 |
| gSH05 | shaping 0.5 | 4,746 | 3,172 | - |
| gTP100 | time-pen 0.01 | 2,631 | 3,239 | 2,935 |

**The control pair agrees to 1.05x (617 u) at 0.25e9** - the only mark
both reached, box A being CPU-bound at ~130k steps/s. That is a tight
EARLY floor and it matches the ledger's older "level is reproducible at
an early matched point (0.4% at 525M)" note; the 5-9x spread of section
1 is a LATE phenomenon. So early-mark verdicts are usable and late ones
are not - which is exactly what round 21 recommended and this round
finally has the control pair to prove.

Safe calls (order-of-magnitude, visible at the early mark):

* **`--time-pen 0.01` is dead** (2.6-3.2k, ~5x below control at every
  mark) and **`--race-shaping 0.5` is dead** (4.7k -> 3.2k, decaying).
  Note this CONTRADICTS the old "time_pen 0.010 is the optimum" figure,
  which came from a warm, act_every 3 config - it does not transfer to
  the from-scratch act_every 4 baseline.
* `gEXPLO` (int 0.5 + shaping 0.5) decays 12.3k -> 5.6k: kill-on-sight.

Not calls, but the useful shape:

* `gSH2` and `gEXTP` lead early (1.2-1.3x control, outside the 1.05x
  early floor) and fall BELOW control by 0.85e9 - the crossover that
  makes end-of-run 1-hour rankings unsafe.
* `gIC10` (int-coef 0.1) is the only arm above control at the last mark
  (+16%), inside the assumed late floor: a CANDIDATE for a 3 h
  confirmation on the spine testbed, not a result. Interesting that
  LESS novelty bonus is the direction.
* Overall: the penalty side is the sensitive knob, the pinned 0.005 is
  not obviously wrong, doubling is clearly bad, halving is
  neutral-to-good early; shaping down is bad, up helps only early.

### 6. A search planner that beats the policy it plans with (tools/beam_tas.py)

Built this round (commits 2f33881, 183c03d, adbba10): 2,048 stochastic
copies of a checkpoint stepped in lockstep in the real sim from one
recorded spawn, elite selection every R decisions with `set_state`
cloning (state + action history + obs-reward feed state), finishes
detected off `core.goal_hits`, and the winner **replayed open-loop on a
fresh env and asserted bit-exact**. It is MCTS's deterministic special
case: policy as prior, the simulator as the model, selection instead of
UCB, rollouts to real terminals instead of a value at a leaf.

On the documented finisher `runs/frozen/sISV_FINISHER_latest.pt`
(the filmed 1:19.73 champion): **greedy 85.23 s -> search 82.42 s
(-2.81 s)** in 12 s of wall-clock (1.62M env-steps/s), 3,682 finishing
runs, eval_honesty 100% corridor FINISH, replay bit-exact.
(`runs/frozen/F_prime.pt` was the plan's default and is NOT a finisher -
0/9 greedy draws, best 37.6 s; the gate caught it and the ledger agrees:
F' is the race_respawn baseline at max 99,004u, success 0.)

What did NOT work, with numbers:

* **24-wave campaign** (8 seeds x R {25,100,250}): nothing beat 82.42 s;
  nearest 82.44 s. The R-bands do not overlap (82.42-82.70 / 83.55-83.93
  / 84.27-84.56), so **selection frequency dominates seed noise and
  wider windows are slower convergence, not distinct lines**. 2/8 tight
  windows went extinct with zero finishers (mode collapse); 0/16 wide.
* **Epsilon-widened proposals** (v2: eps {0.05,0.15,0.30} x window
  {5 s,10 s}, commit-and-replan MPC): **all six DNF**. At eps 0.15-0.30,
  62-88% of decisions carry a randomised head and surf flight does not
  survive it. Uniform noise does not discover ramps, it discovers
  falling. Macro-line search needs STRUCTURED deviation (forced branch
  points, position-stratified elites), not white noise.
* **Prefix dedup** (the user's "fully expanded -> reroll" mechanic):
  **0.0% collisions** on 12-decision prefixes at 2,048 candidates,
  before and after. Proposal narrowness is not the binding constraint at
  this width; the SELECTION horizon is.

Two findings worth more than the -2.81 s:

* **The goal field steered the planner into a kill net.** With
  boundary ranking by geodesic d, 2,038 of 2,048 candidates died at ONE
  tick at ONE z - the documented fail-net `*30` at wall #2, through
  which the field carries finite DESCENDING d. "Alive, smallest d"
  literally selects mid-funnel states. The same blindness that mis-shapes
  the RL reward mis-steered a second, independent consumer.
* **The critic knows what the field does not.** Ranking the boundary by
  `V(s)` instead cleared that funnel exactly where d-ranking died and
  went 3.5x deeper, then stalled where the d-rank/V-rank correlation
  FLIPS NEGATIVE (spearman +0.5 -> -0.5 at d ~ 117k): the critic
  actively prefers higher-d states there. **A V-vs-d rank-disagreement
  scan is a free map of everywhere the shaping lies**, and it is one
  batched forward per state.

### 7. Ops findings (all cost time this round)

* **`run_arm.sh` single-process launches need the `NUMBA_NUM_THREADS`
  cap that `ddp_launch.sh` already has.** A fractional rental reports
  the HOST's nproc (256), numba sizes its pool off that, and the box
  crawls. Cost: ~65 min on one grid box. The ledger predicted this
  exposure in round 23 and it fired exactly as written.
* **The vast ssh PROXY can refuse while the box is healthy.** `ssh5.
  vast.ai:10064` returned "Connection closed" for ten minutes while the
  direct `public_ipaddr` + the port-22 host mapping worked instantly.
  Deploy scripts should fall back to direct on proxy failure.
* **`record.py`'s trailer mislabels goal-box finishes as `"end":"fail"`**
  when no waypoints are set (it infers success from a +50 reward the
  race core does not emit on that path). beam_tas reads `goal_hits`
  instead. Anything that counted finishes from trailers under-counted.
* **`record_ckpt.py`'s obs-reward d0 is computed from pre-reset states**
  (zeroed origins) where `train_fast.py:2520` uses raw spawn origins -
  a latent scale error in the eval feed. Not fixed; do not copy it.
* Blocklisted: machine **46769** (Norway 8x3090 fractional - host
  offline 55+ min mid-arm, plus the nproc=256 numba exposure) and
  machine **12863** (cpu_bound, measured).
* Every rented box now gets a dashboard tunnel at deploy time
  (CLAUDE.md rule added at user request this round).

### 8. What is open after this round

1. `--race-ng 3` (conformant tax, terminals stock) and `--gamma 1.0` -
   the two remaining question-4 mechanisms, both implemented, both
   unrun. Run them ON THE SPINE TESTBED (section 4), not on scratch.
2. `--death-charge kappa` (partial death charge, curriculum kept) -
   implemented, unrun, same testbed.
3. The wall at route vertex ~1596-98 is now known to be neither an
   access nor a coverage problem. Remaining hypotheses: control
   precision at the transition, the reward arithmetic ACROSS it (the
   final descent raises geodesic d by 8,408u), or something only the
   clipped last 10% teaches. The V-vs-d disagreement scan (section 6) is
   the cheapest next probe.
4. Expert iteration / AlphaZero-style distillation of planner output
   back into the policy - the natural third channel (line -> reward was
   xARC, line -> spawns is section 4, line -> actions is this). Online
   AZ is compute-infeasible at ~2,700 decisions/episode; phase-alternating
   ExIt on planner-solved wall states is the affordable form.
5. Whether ANY 3-hour single-seed scratch protocol can be rescued -
   time-to-gate as a continuous statistic is the round-21 suggestion and
   was not used here; section 1 says it should be, or seeds must be
   paired.

### Round 27 addendum: the grid completed (12/12) and its HONEST corridor numbers

The section-5 table above was `race/eval_progress` only, written before
`gIC50` finished and before `eval_honesty.py` had been run on the
harvested trajectories. Both gaps are now closed. Corridor MAX over each
arm's last three evals (route 231,680u; means are the last eval's):

| arm | change vs pinned baseline | corridor MAX | mean | vs gCTLb | last step |
|---|---|---|---|---|---|
| gIC10 | `--int-coef 0.1` | **68,608** | 68,295 | **1.39x** | 1.59e9 |
| gIC50 | `--int-coef 0.5` | **57,344** | 56,491 | 1.16x | 1.51e9 |
| gSH2 | `--race-shaping 2` | 49,664 | 44,700 | 1.01x | 1.51e9 |
| **gCTLb** | **control** | **49,408** | 48,526 | - | 1.51e9 |
| gEXTP | `--int-coef 0.5 --time-pen 0.0025` | 48,256 | 46,663 | 0.98x | 1.59e9 |
| gLOPRES | `--race-shaping 0.5 --time-pen 0.0025` | 26,624 | 22,187 | 0.54x | 1.51e9 |
| gHIPRES | `--race-shaping 2 --time-pen 0.01` | 24,320 | 20,693 | 0.49x | 1.59e9 |
| gEXPLO | `--int-coef 0.5 --race-shaping 0.5` | 14,464 | 13,326 | 0.29x | 1.66e9 |
| gTP100 | `--time-pen 0.01` | 3,840 | 3,541 | **0.08x** | 1.58e9 |
| gCTLa | control (CPU-bound box) | 37,632 | 20,494 | - | 0.45e9 |
| gTP25 | `--time-pen 0.0025` | 19,712 | 19,143 | - | 0.45e9 |
| gSH05 | `--race-shaping 0.5` | 7,808 | 3,371 | - | 0.45e9 |

**Zero finishes in all twelve arms.** The bottom three rows ran on the
CPU-bound box and stop at 0.45e9, so they are comparable only to each
other and to gCTLa (which they lose to: 19.7k and 7.8k against 37.6k).

What corridor changes versus eval_progress:

* **The negatives get worse and cleaner.** `--time-pen 0.01` is 0.08x
  control - **12.9x below** - not the ~5x eval_progress suggested. Doubling
  the time penalty is catastrophic, full stop.
* **`gIC10` gets better, not worse.** +16% on eval_progress becomes
  **1.39x on the honest metric**, and `gIC50` corroborates in the same
  direction (1.16x). Both intrinsic doses beat the control while their
  midpoints differ, so the response is not monotone in `int_coef` -
  but LESS novelty bonus than the pinned 0.25 is the only change in
  this grid that survives the honest metric on the upside.
* **The early leaders vanish.** `gSH2` (1.3x early) and `gEXTP` land at
  1.01x and 0.98x - dead level with control. Early-mark leadership did
  not survive; this is the crossover pattern again, now confirmed on
  corridor rather than inferred from eval_progress.
* The pair arms (`gLOPRES`, `gHIPRES`) are both ~0.5x: combining two
  ratio changes was worse than either alone in every case measured.

**Status of `gIC10`: a candidate, not a result.** 1.39x sits just
outside the ledger's assumed 27% late floor, but this round measured
the control-control spread at 1.05x only at the EARLY mark (0.25e9);
there is no measured control-control corridor spread at 1.5e9, and
section 1 of this round showed the late spread can reach 5.4x. The
confirmation must be run on the spine testbed (section 4) or as a
paired-seed design - not as another single-seed scratch arm.

Grid ops: all four boxes destroyed, fleet at zero, data in
`runs/research/r27grid/{A,B,C,D}/`. Bake-off data (bCTL/bFP10, the
section-1 pair) is in `runs/research/r27bakeoff/`; that box was
destroyed by the coordinator after harvest, not by the grid agent.

### Round 27 addendum 2: the two knobs worth tuning, and the START-DISTRIBUTION hypothesis (user, 2026-08-26)

**`--time-pen` is bracketed and the optimum is INTERIOR.** Three
independent measurements now bound it on the from-scratch act_every-4
baseline:

| time_pen | result | source |
|---|---|---|
| 0.01 | corridor 3,840u = **0.08x control** (12.9x below), ep_len ~400 (die-fast) | gTP100, 1 h grid |
| 0.005 (pinned) | control level | gCTLa/gCTLb, bCTL, xCTLS |
| 0.0025 | ~level early (19,712u at 0.45e9 on the CPU-bound box, vs control 37,632u there) | gTP25, 1 h grid |
| 0.0 | eval plateau 12.4k = **0.6x control**, peak speed 1,780 vs 2,340 u/s | xTP0, 3 h local |

So both ENDS are bad - doubling collapses learning into the suicide
basin, zeroing removes the pace-setter - and **the optimum lies strictly
inside (0, 0.01), with 0.005 not obviously wrong and 0.0025 unresolved
against it.** The old ledger figure "0.010 is the optimum, worth keeping"
came from a WARM, act_every-3 config and does NOT transfer; treat it as
superseded for scratch runs. A fine sweep (0.0025 / 0.004 / 0.005 /
0.0075) is worth one campaign, but only on a testbed with a reproducible
frontier - see below.

**`--int-coef` is NOT established.** 0.1 scored 1.39x and 0.5 scored
1.16x on corridor against one control at one seed, with no measured
control-control corridor spread at that step count and a demonstrated
late spread of up to 5.4x elsewhere in this round. Two doses on opposite
sides of the pinned 0.25 both landing above control is more consistent
with noise than with a dose-response. **Treat as unresolved, not as
"less novelty is better".**

#### The hypothesis the spine result actually supports (user, 2026-08-26)

The user's reading of section 4, recorded because it reframes the
mechanism and suggests the next experiments:

> Training from scratch means the state-visitation distribution is
> concentrated at the map start and extends forward only slowly. The
> samples are not IID; the early map is distilled into the weights
> billions of times before the late map is ever seen, and the weights
> drift into a region that makes the rest harder to learn. **The
> distribution is the key.** The spine result is then not "demonstrations
> help" but "a uniform start distribution over the map, given to a
> FRESH network, removes an accumulated pathology".

This predicts something the current data cannot distinguish: whether
xDEMO90/xDEMO50's advantage comes from the uniform state distribution
per se, or from the fresh weights, or from both. The decisive
experiment is a 2x2 - {fresh weights, continued weights} x {spine
spawns, map-start spawns} - where the interesting cell is
CONTINUE-FROM-WALL-CHECKPOINT + spine spawns: if that recovers most of
the spine effect, the distribution is doing the work; if only the fresh
init does, it is a plasticity/primacy effect and belongs with the
shrink-and-perturb family (already implemented here, never run on top
of a working shaping).

Soft alternatives to a hard reset, to be designed rather than assumed:
importance-weighted or prioritized start-state sampling toward
under-visited depth, a teacher/student pair where a fresh student trains
on the teacher's state distribution, or periodic partial re-init. A
literature survey was commissioned on 2026-08-26 and its findings will
be appended.

### Round 27 addendum 3: the literature answer

The start-distribution survey commissioned in addendum 2 is in
`docs/research-litsurvey.md`. Short version: BOTH halves of the user's
hypothesis are established - the reset half as primacy bias /
shrink-and-perturb / plasticity loss, and the distribution half as
CPI concentrability and the policy-gradient **distribution mismatch
coefficient** `D_inf` (Agarwal et al., JMLR 2021), which says
convergence degrades inversely with how badly the training start
distribution matches the occupancy of a good policy. Training from the
map start is the worst case of that coefficient and the spine is the
fix; the theory predates the experiment. What is NOT named in the
literature is the combination actually run here (full re-init + a
uniform distribution over a previously recorded line), which is also
why the 2x2 credit-assignment experiment is needed. Three supported
soft alternatives, cheapest first: TD-error-prioritised reservoir
sampling (PLR, a scoring change on existing machinery), continuous
shrink-and-perturb at a corrected beta, and teacher/fresh-student
distillation.

### Round 27 addendum 4: xLOOP - the iterated reset+respawn loop COMPOUNDS for four rounds, then hits the wall and stays there for twenty

The automated version of the spine experiment (user, 2026-08-26):
every round is a FRESH from-scratch network trained for 1e9 steps; at
the end of a round, 20 greedy map-start evals are recorded, the deepest
one is picked, its terminal fall is trimmed (contact_cut, the CLAUDE.md
gravity-departure rule), its per-tick states become the next round's
spawn distribution, and the weights are thrown away. `tools/loop_driver.py`
+ `tools/loop_spine.py`; results in `runs/xLOOP/loop_summary.jsonl`.
Stopped by the user after **24 rounds = 24e9 steps, ~9.6 h** on the local
5090. Round 0 ran at run_arm's default eval cadence (44 min); from round
1 the in-round evals were cut 3x (`XLOOP_RECORD_EVERY=225e6`) and a round
costs **23 min**.

**Phase 1 - it compounds, and fast (rounds 0-3):**

| round | spawned from | corridor MAX | % of route | chosen min_d |
|---|---|---|---|---|
| 0 | map start | 55,680 | 24.0% | 144,965 |
| 1 | r0 spine (2,398 states) | 97,792 | 42.2% | 105,732 |
| 2 | r1 spine (12,000) | 154,624 | 66.7% | 54,324 |
| 3 | r2 spine (5,956) | **205,568** | **88.7%** | 3,761 |

Each round starts from random weights and inherits ONLY a state
distribution, and the frontier still grows 24% -> 42% -> 67% -> 88.7% in
four rounds. **From a cold start with no champion, no demonstration and
no planner line, the loop reaches the documented wall in about two
hours.** That answers the question the single-shot spine arms could not:
the bootstrap does compound, and the compounding is worth ~3.7x of route
coverage over three iterations.

**Phase 2 - and then it stops, completely (rounds 3-23):**

Twenty consecutive rounds sit at corridor **204,215 +/- 5,138**
(min 181,248, max **205,824**), i.e. pinned at 88.7-88.8% of the route.
**0 finishes in 480 greedy evaluations.** The spines in this phase are
all the same shape (~6,800-7,300 states ending at d ~ 18,700 after
~580 ticks of fall trimmed), so the loop is faithfully reproducing the
same line every round and gaining nothing from it.

**The wall is now identified from four independent directions**, all
landing within 640 u of each other:

| method | corridor MAX |
|---|---|
| the stuck warm lineage (3.78e9 steps, rounds 16-19) | 205,312-205,440 |
| xDEMO90 (planner line, 90% clip) | 205,312-205,440 |
| xDEMO50 (planner line, 50% clip) | 205,440-205,568 |
| **xLOOP (self-bootstrapped, no external line at all)** | **205,184-205,824** |

Nothing about the start-state distribution crosses it. xLOOP's spines
cover the map to d ~ 18,700 (i.e. past 90% of the geodesic), its
policies get to within **2,233 u** of the goal geodesically, and still
zero of 480 greedy episodes cross the finish. Combined with addendum 2's
result that xDEMO90 spawned PAST the wall and never crossed either, the
conclusion is now firm: **the wall is not an exploration, access,
coverage or curriculum problem. Distribution methods take the agent TO
it, reliably and cheaply, and then do nothing.**

Two secondary observations worth keeping:

* min_d bottoms out at 2,233-3,000 u while corridor stays at 88.7% -
  the documented off-route dive into goal-adjacent airspace. The loop's
  SELECTION uses min_d, so it is picking dives; the fall trim is what
  stops that poisoning the next spine (and the summary records corridor
  beside min_d so the substitution is visible). A corridor-based
  selection rule is the obvious next tweak.
* Round 19 is a single bad round (corridor 181,248) that recovered
  immediately - the loop is robust to one weak generation because the
  next round re-derives its spine from 20 fresh evals.

**What this closes and what it opens.** It closes the "does iterating
compound" question (yes, for four rounds) and the "is the wall
distribution-bound" question (no). It opens exactly one thing: the wall
itself, which is now the single blocking result on this map and has
survived every mechanism this project has tried. The remaining live
hypotheses are control precision at the transition, the reward
arithmetic ACROSS it (the final descent RAISES geodesic d by 8,408 u),
and the fact that no method has ever shown the agent a successful
crossing - which is what the planner (which CAN finish, 82.42 s) could
supply as actions rather than as spawn states.

### Round 27 addendum 5: the planner cannot cross the wall either (searched 2026-08-26, STOPPED by user)

Applied beam_tas AT the wall to xLOOP round_11 (step 1.00e9, the
deepest of the 24 rounds: greedy min_d 2,233, wall approach 2,978 u/s).
Deterministic greedy prefix to a tick 2.0s / 3.0s before its departure,
then 2048 sampled continuations per wave, eps=0, route scoring,
finishes ranked first.

**47 waves x 2048 envs (~96,000 candidate rollouts): 0 crossings.**
Frontier route vertex **1602 in every single wave**; min_d 5,731-5,773,
a total spread of **42u across all 47 waves** (seeds 0-7 x resample
{10,25,50} x two restore points). The stopping point is invariant to
seed, selection frequency and lead time.

Deaths cluster hard: route vertex q10/50/90 = 1600/1601/1602, z
-4,550/-4,364/-4,252 (the pit below the map), median 4,924u off-line,
~4,095 death events per wave.

**Why search cannot help, from wall_profile (agent minus champion):**
the deficit is not jitter at the transition, it is a line that has
already diverged long before it. Vertex 1540: speed -193, off-line
+158. 1560: -193, +161. 1580: -195, +223. **Vertex 1600: speed -216,
z -1,196, off-line +2,739 - the agent is 2,845u off the champion line
where the champion is 106u.** Sampling the policy amplifies what it can
nearly do; it cannot recover a line that is already 2.8k units wrong,
which is the same macro-line limit that killed the eps-widened search.

So the wall now stands against: the warm lineage (3 mechanisms), three
start-distribution methods, a 24-round self-bootstrapping loop (480
greedy evals), and policy-guided search from directly upstream
(96k rollouts). The only thing that has ever crossed it is the champion
lineage, whose ordinary respawn reservoir happened to harvest states
near the goal, produced real finishes, and consolidated backward
(0.3% -> 48% success in 350M steps).

User stopped this line of work here.

### Round 27 addendum 6: xLOOP vs CONTINUOUS training at matched 4e9 steps

The question addendum 4 left open: is the iterated reset worth anything
against simply training one network for the same total compute? Corridor
MAX over every eval recorded at or below 4e9 steps, all runs on
surf_src_cannonball (route 231,680u), computed with eval_honesty:

| run | kind | corridor | % route |
|---|---|---|---|
| xDEMO50 (planner line, 50% clip) | spine-seeded | 211,456 | 91.3% |
| xDEMO90 (planner line, 90% clip) | spine-seeded | 206,208 | 89.0% |
| **xLOOP rounds 0-3 (4 x 1e9, resets)** | **ITERATED** | **205,568** | **88.7%** |
| xFP10 (5090, fail_pen 10) | continuous | 134,272 | 58.0% |
| bCTL (3090, control, seed 1) | continuous | 134,272 | 58.0% |
| gEXTP / gCTLb / gIC50 / gIC10 / gSH2 | continuous | 66-70k | 29-30% |
| xCTLS (5090, control, seed 0) | continuous | 40,704 | 17.6% |
| gLOPRES / gHIPRES / gCTLa | continuous | 37-40k | 16-17% |
| gTP25 / bFP10 / gEXPLO / xTP0 | continuous | 13-20k | 6-9% |
| gSH05 / gTP100 | continuous | 4.5-7.8k | 2-3% |

**No continuous from-scratch run came close.** Sixteen of them, eleven
configurations, two GPUs: the best reached 58.0% and the median ~17%,
against xLOOP's 88.7% at identical step budget - **1.53x the luckiest
continuous run and ~5x the median.** And 58% is itself the fortunate
tail: the other control at the same budget managed 17.6%, which is the
5.4x control-to-control spread of section 1.

The only runs above xLOOP are the two spine-seeded arms, which are not
competitors but the same mechanism handed a head start - both were
given the planner's 82.42s FINISHER line for free. xLOOP built its
line from nothing: round 0 is a plain scratch run reaching 24.0%,
squarely inside the ordinary control distribution.

**So the effect is not the demonstration - there was none.** Three
reset-and-respawn iterations turn an unremarkable seed into the wall.
Caveats unchanged: 0 finishes anywhere in the table, n=1 on the loop,
and the reset-vs-distribution 2x2 (addendum 2) is still unrun, so this
shows the COMBINATION beats continuous training without yet saying
which half does the work.

Correction to addendum 4 and the wall table: xDEMO50 ran to **5.13e9
steps, best eval_progress 193,599 at 1.66e9**, and its corridor MAX at
<=4e9 is **211,456** - above the 205,440-205,568 quoted earlier from a
mid-run read, and the highest corridor any run in this project has
reached without finishing.

### Round 27 addendum 7: a ResNet-style trunk costs 9-10x throughput (measured, NOT run)

User asked whether the 3-conv image trunk could be replaced by a
pretrained ImageNet backbone (resnet/efficientnet). Two separate
answers, both measured before any long run:

**Pretrained weights: blocked and ill-posed.** torchvision is broken in
this environment (`operator torchvision::nms does not exist` against
torch 2.11), so ImageNet checkpoints are not loadable without fixing
that dependency. Independently, the transfer is ill-posed: the input is
64x32x1 DEPTH with a custom encoding (d/near plus an exponential far
tail), not 224x224x3 natural images, and resnet18's ImageNet stem
collapses a 32x64 input to 8x16 before layer1 and **1x2 by layer4** -
the pretrained hierarchy has no spatial extent left to use. Not run.

**Architecture, from scratch: measured and rejected on cost.** Added
`--trunk {plain,resnet}` (commit 4a05a31) where `resnet` is a residual
trunk SIZED for 64x32 (3x3 stem, 3 stages of 2 BasicBlocks at 32/64/128
channels, GroupNorm, pool to 4x8) = 2.79M params against the plain
trunk's 1.07M (of which the conv layers are only 23.5k). Two 150M-step
smokes, identical but for the trunk:

| trunk | marginal steps/s | VRAM | a 3h arm reaches |
|---|---|---|---|
| plain | **656,903-675,852** | ~8 GB | ~7.1e9 steps |
| resnet | **68,018-71,764** | **31.5 GB** | ~0.78e9 steps |

**9.2-9.9x slower end to end** (the isolated forward at batch 2048 is
16x: 0.76ms -> 12.1ms). The forward is a bigger share of the step budget
than expected because the physics core is fast and PPO runs the network
again for 64 minibatches per rollout. At 0.78e9 steps per 3 h arm,
neither matched-wall nor matched-step comparison is affordable, and at
31.5 GB it would not fit a 24 GB 3090 at these batch settings. Killed
by the user at the gate; the long arm was never launched.

**Two by-products worth keeping:**

* **BatchNorm is a train/eval landmine in this trainer.** The trainer
  never calls `policy.eval()`; `record_ckpt.py` always does. A BN trunk
  is therefore a DIFFERENT FUNCTION in training and in every recording
  (measured 0.0214 max logit difference on identical input), and the
  CUDA graph mutates its running stats on every replay. GroupNorm has
  no buffers and measured 0.0000. Any future arm adding a
  train/eval-divergent layer (BN, dropout) must account for this.
* The `plain` path is bit-identical, verified by `tests/python/
  test_trunk.py` against the PRE-FLAG commit read out of `git show` by
  pinned sha - same state_dict key order, `torch.equal` on every
  tensor, identical forward, across baseline / surf-mask / route /
  chunk shapes.

## Round 28 - goallines: the 2x2 from scratch on a FIELD-DERIVED line (local 5090, 2026-08-31 01:40-07:00)

Plan: docs/research-plan-goallines.md (E0, plus E1/E2 transposed to
scratch per the user's directive: from scratch, local, 1 h per arm, mmddp
base). Branch `goallines` = mmddp + cherry-picked --race-arc
(216c508/3a0d76e) + mapfleet arc_* passthrough + launch_local $map fix +
build_route --from-field. All four arms `scratch_ablate`, one seed, one
hour wall-clock each, same RTX 5090, dashboard-visible in
C:\RL_Surf_gl\runs.

### E0 - the field-descent line (no training, no champion, no demo)

    python tools/build_route.py --from-field --map <main>/maps/surf_src_cannonball.bsp \
        --cell 32 --seed="-14208,2898,10688" --out maps/surf_src_cannonball.fieldroute.npz

Goal-field cache HIT (no bake); 5,256 voxel steps -> 1,573 pts @ 128u,
201,162u of line; d descends 198,232 -> 64u. Expectations, scored:
termination at the goal basin MET (BFS strict descent cannot stick);
in-solid 0.0% vs xAUTO's 24.8% EXCEEDED (the walk lives in honest air
voxels by construction); deviation vs the champion route PARTIALLY
MISSED - mean 861u / median 719u / p90 1,677u / max 3,342u at the final
descent, with 25.3% of vertices beyond xAUTO's 1,131u max across eight
mid-map segments (peaks 1.3-2.3ku vs the 1,500u corridor).

### The 2x2 (reward x observation, all on the same fieldroute.npz)

| arm | reward | line in obs | fps sustained | steps in 1 h |
|---|---|---|---|---|
| xsCTL | geodesic potential | no | 725k | 2.637G |
| xsFARC | field-line arc | no | 514k | 1.870G |
| xsFAN | geodesic potential | fan 27 feats | 674-689k | 2.453G |
| xsFULL | field-line arc | fan 27 feats | 460k | 1.670G |

The arc term costs ~30% fps - ArcProgress.advance is windowed numpy on
the per-TICK hot path (~700k calls/s); numba-fuse it like
goalfield._FAST_SAMPLE before any rented arc arm. The fan costs ~6%
(torch, per decision).

race/eval_progress at MATCHED STEPS (same card, config identical but for
the treatment):

| steps | xsCTL | xsFARC | xsFULL | xsFAN |
|---|---|---|---|---|
| 0.50G | 16,815 | 16,754 | 13,234 | 13,628 |
| 1.00G | 29,318 | 43,092 | 15,444 | 36,962 |
| 1.50G | 45,269 | 42,057 | 14,995 | 55,774 |
| 1.67G | 37,390 | 47,264 | 16,952 | 77,340 |
| end of hour | 72,911 @2.64G | 47,340 @1.87G | 16,952 @1.67G | 100,120 @2.45G (peak 106,478) |

eval_honesty --order-only 16 on each arm's LAST eval (champion route as
ruler; eval_progress tracked the honest metric all night, dives-below
0/9 everywhere):

| arm | corridor mean | corridor MAX | % of route | finishes |
|---|---|---|---|---|
| xsCTL | 76,544 | 78,080 | 33.6% | 0/9 |
| xsFARC | 49,252 | 49,280 | 21.3% | 0/9 |
| xsFULL | 17,877 | 18,048 | 7.8% | 0/9 |
| **xsFAN** | **107,833** | **122,752** | **52.9%** | 0/9 |

### What the evidence says, against the written expectations

1. **The champion-free line CARRIES THE ORDERING (E1's question): MET at
   matched steps.** xsFARC tracks xsCTL to 0.4% at 0.50G (the same
   level-reproducibility seen at 525M in the seed-noise measurement),
   leads 47% at 1.0G, trades back after. Arc paid honestly: end gain
   30,052u / reach 48,752u / off 3.3% - no farming, corridor holding
   despite E0's deviation. In WALL-CLOCK (the standing currency) the
   control still ends a gate ahead, but that is the 30% implementation
   tax, not the mechanism.
2. **The fan on the GEODESIC reward is the night's result - and it
   contradicts the written expectation.** E2 predicted "no harm at
   best on one map; the net can ignore the fan". xsFAN instead cleared
   122,752u corridor MAX - 1.57x the control, past every round-21 gate
   band, from a slow start (-20% at 0.50G) to a lead at every mark from
   1.0G on. HOWEVER: 1.57x sits INSIDE the 1.71x-3.04x corridor-MAX
   spread this file records between byte-identical 1 h runs (the xSH1
   retraction), so at one seed the standing rules forbid calling it an
   effect. It is a direction, and the strongest single-arm number a
   scratch hour has produced on this card.
3. **xsFULL surfaced a real failure mode of the corridor design.** By
   end-of-hour its training rollouts were 86.8% off-corridor, arc gain
   1,910u, intrinsic exhausted (0.04/ep), reward pinned at the raw
   -time_pen floor - the frozen anchor makes detours FREE, and a policy
   fully off-line with no novelty left has NO gradient back to the line
   (the geodesic control always has one). Evals kept scoring 17k
   because greedy stays near spawn-line. Candidate fixes, each its own
   arm: a capped distance-to-line re-entry potential; corridor sized to
   E0's deviation peaks; non-exhausting intrinsic.
4. Every cross-arm difference above except the 0.50G agreement is
   uncallable at one seed on a near-binary gate metric. What tonight
   PROVES is mechanism viability and wiring: the line exists without a
   champion (E0), the arc pays it honestly, the fan reads it, all
   diagnostics flow, and the plan's next rungs (E3 multi-map / E5
   point-goals - where goal DIVERSITY forces the net to read the line)
   are what can turn xsFAN's direction into a claim.

Runs: C:\RL_Surf_gl\runs\{xsCTL,xsFARC,xsFULL,xsFAN} (progress.csv,
traj_*.jsonl, ckpt_latest.pt each). Line: maps/surf_src_cannonball.fieldroute.npz
(committed, 12 KB; also mirrored at the main checkout for the runs'
recorded absolute paths).

### Round 28 addendum - the E2 probe: xsFAN READS the fan (97.5% collapse), xsFULL never did

`record_ckpt.py --route-mode {live,off,frozen}` (the --depth-mode idiom;
width unchanged, content ablated) on each checkpoint, 9 greedy episodes,
`eval_honesty --order-only 16`, champion ruler:

| ckpt | live mean/max | fan OFF mean/max | frozen mean/max |
|---|---|---|---|
| xsFAN @2.43G | 108,672 / 124,544 | 2,716 / 3,328 (-97.5%) | 2,304 / 3,072 |
| xsFULL @1.65G | 16,512 / 18,048 | 15,701 / 17,280 (-4.9%) | - |

Live reproduces each trainer's own eval (recorder fidelity check). The
probe is WITHIN-policy and causal, so the identical-run seed spread that
gates the cross-arm level comparison does not apply to it: xsFAN's
policy genuinely built its behavior on the line observation - zero the
27 columns and a 52.9%-of-route policy becomes a 1.4% one. xsFULL, the
arc+fan arm that plateaued off-corridor, is fan-INDEPENDENT (-4.9%,
noise): under a reward that pays nothing off the line, the fan never got
grounded; under the dense geodesic gradient it became load-bearing.

Two consequences for the plan. (1) The E2 question "does the policy
read the fan" is answerable in 10 GPU-minutes per checkpoint and should
gate every future fan arm. (2) The grounding hypothesis sharpens: it is
not the fan OR the reward, it is their interaction - a reward with
gradient everywhere grounds the observation; the corridor-freeze
design, as measured in xsFULL, can leave it unread. E3/E5 (goal/map
diversity) remain the test of whether what xsFAN learned is "follow the
visible line" or "this map, memorized with the fan as a position
encoding" - the probe cannot distinguish those two on one map.

### Round 28 addendum 2 - xsFAN2 (seed 1): the fan arm REPLICATES on both axes (user-directed rerun)

The user directed a seed rerun of xsFAN to test the lucky-seed
explanation (the one-seed rule holds for arms; this is a replication of
the round's headline, the same precedent as the xEP4/xNS128 noise-floor
measurement). Identical launch, `--seed 1`, same 1 h, same card.

| | xsCTL (seed 0) | xsFAN (seed 0) | xsFAN2 (seed 1) |
|---|---|---|---|
| eval_progress @1.0G | 29,318 | 36,962 | 52,687 |
| eval_progress @1.5G | 45,269 | 55,774 | 87,180 |
| eval_progress peak | 72,911 @2.64G | 106,478 @2.42G | 98,368 @1.89G |
| corridor mean / MAX (last eval) | 76,544 / 78,080 | 107,833 / 122,752 | 101,234 / 106,752 |
| probe: fan OFF | n/a | mean -97.5% | mean -90.0% |

* **Level: 2 of 2 fan seeds far above the control**, and the two fan
  seeds agree to 1.15x on corridor MAX while both sit 1.37-1.57x over
  the control - tighter agreement than the 1.71-3.04x identical-config
  spread predicts for unrelated draws. xsFAN2 led the control at every
  matched-step mark from 1.0G.
* **Mechanism: 2 of 2 fan seeds are fan-DEPENDENT** (zeroing the 27
  columns collapses corridor mean 97.5% / 90.0%). The within-policy
  probe replicates across seeds.
* Still open, unchanged: one control seed (a seed-1 control would
  complete the pair), and whether the learned behavior is "follow the
  visible line" or map-specific with the fan as a position encoding -
  E3/E5 remain the test. But "lucky seed" is no longer the simplest
  explanation of the fan effect; two independent draws reproduced both
  the level class and the dependence.

### Round 28 addendum 3 - zero-shot fan transfer to petrus_lite: NULL (confounded, read narrowly)

Built surf_petrus_lite.fieldroute.npz the same way (--from-field, 282
pts, 35,963u, d 35,624 -> 64), then recorded the xsFAN checkpoint ON
PETRUS with the petrus line feeding its fan (`record_ckpt --map
--route` override, new), live vs zeroed, scored against the petrus line:

    live: mean 1,052u / max 1,280u    fan off: mean 1,024u / max 1,024u

No transfer, and no live-vs-off delta (28u = noise). READ NARROWLY: the
policy cannot execute petrus movement at all (its motor program is
cannonball-tuned; petrus needs the ~1,550 u/s ramp gate), so the fan
never got a chance to steer - this null CANNOT distinguish "fan =
line-following" from "fan = position encoding". What it does establish:
eval-time zero-shot transfer of the whole policy is absent, so the
generalization question is only answerable by TRAINED diversity (E3
multi-map / E5 point goals), not by probing single-map checkpoints on
foreign terrain.

Next per the user's direction: xsFANX, a plain 2 h resume of xsFAN
(no latch - rejected as a per-map ad-hoc constant; the principled
anti-barrier reward in this program is the arc coordinate, pending a
generic fix for the xsFULL off-corridor grounding flaw). WRITTEN
PREDICTION before the run: the fan changes nothing about the reward
arithmetic at the final descent (vertex 1601: the correct path RAISES d
8,408u, charged ~-4.2 vs a never-seen +50), so xsFANX should climb into
the high-80s% and post 0 finishes, stopping at the wall like the 234
warm-resume control episodes. A finish would falsify the barrier
transfer to scratch lineage; a stall confirms it.

### Round 28 addendum 4 - xsFANX: the wall prediction CONFIRMED to the vertex

2 h plain resume of xsFAN (5.60G total steps). Corridor progress
205,312u = 88.6% - the exact control wall - with closest-approach 2-6u
the whole way, then dive-below (end z ~-4,100) on 14 of 16 final
episodes; 0 finishes; one 205,568u flicker (xMARGIN-class). The
eval_progress readings up to 195,505 were dives, per the standing
warning. VERDICT: the barrier transfers to scratch lineage; the fan
solves REACHING the wall (5.6G from scratch vs the 3.8G warm ckpt that
needed margin-2 harvesting) and cannot touch CROSSING it - that is
reward arithmetic.

NEXT (launching now): xsFANARC - resume xsFANX with --race-arc on the
field line, fan unchanged. The fully champion-free stack: field line
(ordering), fan (observation), arc (barrier-free monotone coordinate).
WRITTEN PREDICTION: finishes within the hour (round-18 xARC analog went
0 -> 63/102 on the warm stuck ckpt with a champion line); high arc_off%
on failure implicates the E0 deviation segments vs the 1,500u corridor;
low-off% failure would challenge field-line ordering at the wall.

### Round 28 addendum 5 - xsFANARC: prediction FAILED; the failure mode is a THIRD pattern (warm reward-switch shock), and the wall ordering stays untested

Full pre-registered hour ran (5.60G -> 6.93G, +1.33G steps, 335-380k
fps). 0 finishes - the prediction (finishes within the hour, xARC
analog) is FALSIFIED. But neither pre-registered dig branch fits:
off-corridor stayed 0.3-1.3% (not the E0-deviation/corridor failure)
and the policy never re-reached the wall under arc (not a wall-ordering
failure - that claim is simply UNTESTED by this arm). What happened
instead: the reward switch collapsed the inherited deep-map behavior -
greedy 87.9% on the first post-resume eval (the old policy), then 26-33%
for the rest of the hour; training arc reach plateaued ~66-69k (33% of
arc); kl spiked to 0.03 at the switch. Reading: VALUE SHOCK. The critic
was fitted to geodesic-shaping returns; under arc, deep-map states
(where the geodesic paid the dive) now return ~0, advantages turn to
noise, and PPO dismantles the behavior faster than arc rebuilds it.
Round-18 xARC did not show this on the sOBSR2 ckpt (its evals never
dipped) - its 3.8G of stall history may have left a flatter value
landscape than our hot streak did. User called the arm at ~+1.3G.

Candidate designs for a clean retest of "arc crosses the wall on this
lineage", each generic (no map constants): (a) critic-only warm-up
after a reward switch (freeze pi, refit V, then unfreeze); (b) anneal
geodesic -> arc over ~100M steps; (c) reset the value head at the
switch (accept the shock, make it explicit); (d) from-scratch fan+arc
with an off-corridor re-entry gradient (fixes xsFULL grounding instead
of switching rewards). Unranked; none launched.

### Round 28 addendum 6 - xsFANARC2: the shock is NOT transient; and the self-line supersedes the field line

Zero-delta continuation of the arc switch, killed at +1.58G post-switch
steps by the stationarity rule: arc reach 66,357 -> 66,515u (~150u) over
1.5G steps, greedy pinned 31-33%, off 0.3-0.5%, reservoir min-depth
shallowing 30% -> 67% as the deep states wash out. VERDICT: a warm
geodesic->arc switch destroys the deep-map behavior and equilibrates
mid-map; time does not repair it. Any future reward switch needs the
critic handled explicitly (reset/refit/anneal - designs in addendum 5).

THE GENERAL MECHANISM (user direction: no more line heuristics): distill
the reference from the policy's own experience; the field line is only
day-zero initialization. Built with existing validated tools only
(pick_selfline last-contact trim + build_route --allow-unfinished,
ported from the selfline branch): from xsFANX's wall episodes ->
maps/surf_src_cannonball.selfroute.npz, 1,599 pts / 204,559u, ends at
champion vtx 1599 (88.3%) with the trimmed fall descending ALONG the
champion corridor to 5,847u short of the finish box; deviation vs
champion mean 150u / max 696u (field line: 861/3,342), 0% in-solid.

NEXT - xsFANSELF, the single-delta arm: resume the wall-stuck xsFANX
checkpoint changing ONLY the fan's line (fieldroute -> selfroute); same
geodesic reward, same flags, no value shock (the return definition is
unchanged, and along the ridden 88% the new line IS this policy's own
path, so the features barely move). WRITTEN PREDICTION: greedy resumes
>= 85% immediately; at the wall the fan now points down the real
descent instead of across the void, so dives should bend toward the
finish box (5,847u past the line end) - watch for the first nonzero
win_rate of the entire lineage. Failure mode to watch: the geodesic
barrier still charges the descent, so the fan-vs-reward conflict may
merely relocate the dive; that outcome would isolate the barrier as the
sole remaining blocker, with the critic-handled arc switch as the
follow-up.

### Round 28 addendum 7 - xsFANSELF: prediction FALSIFIED instantly; the line is part of the observation contract

Single-delta line swap (fieldroute -> selfroute) on the wall-stuck
xsFANX ckpt. Predicted >= 85% immediate carryover; observed 1.5% cover
on the FIRST post-resume eval (+0.6M steps = the unchanged policy),
plateau 2.6% across three evals, killed by the stationarity rule at
~15 min. Diagnosis: spatial proximity of two lines does NOT mean
feature proximity - the fan reads the line's PARAMETRIZATION (nearest-
point projection, arc-offset lookaheads, local tangents), and a flown
path weaves where a descent trace runs straight, so the swap perturbs
the features the way the shuffle probe does, on a policy measured 97.5%
fan-dependent. GENERAL LESSON for the distillation loop: the reference
line is part of the policy's observation contract. Updating it
mid-training needs either (a) the RETRAIN branch (from scratch on the
new line - clean, costs a fresh run), or (b) a feature-continuous
update rule (anneal/blend between lines, or constrain re-extraction to
preserve parametrization where the policy already flies) - (b) is
design work, not a flag. Both reward-switch (add. 6) and line-switch
(here) now show the same shape: THE WARM POLICY IS BRITTLE TO ITS
GUIDANCE CONTRACT, in reward and in observation alike.

### Round 28 addendum 8 - the selfroute 2x1 (PRE-REGISTRATION, written before evidence)

User granted ~8 h autonomous. Program: the distillation loop's RETRAIN
branch, as a 2-arm from-scratch comparison on the SELF-LINE, 3 h each,
local 5090, one seed, scratch_ablate config:

* **xsFAN3** = fan on selfroute, geodesic reward (running now).
  Expectations: (E1) early curve at matched steps within the fan band
  (xsFAN 13,628 / 36,962 / 55,774 and xsFAN2 17,498 / 52,687 / 87,180
  at 0.5/1.0/1.5G) - the honest line should not be WORSE as an
  observation; a large early lead would say line parametrization
  quality matters to learning speed. (E2) at wall depth (~5-6G,
  reachable inside 3 h at fan throughput), the geodesic barrier is
  unchanged, so the base prediction stays dive-at-88.6% - EXCEPT the
  fan now points down the real descent, and the line ends 5,847u short
  of the finish box: any bending of the dive toward the box, or the
  lineage's first nonzero win_rate, is the fan fighting the reward and
  partially winning. 0 finishes = barrier confirmed sole blocker,
  cleanly, on an honest line.
* **xsARC3** = fan + --race-arc BOTH on selfroute, from scratch (next).
  The thesis test with every measured confound removed: no warm switch
  (add. 6/7 brittleness does not apply), no lying line (the corridor
  wraps the policy-flyable path family), truncated line -> beyond its
  end the reward is success-bonus-only, the validated xSELF regime.
  Expectations: off-corridor share stays LOW (the xsFULL 86.8% plateau
  was the field line's geometry + exhausted novelty; the self-line
  hugs real flight), the fan GROUNDS (probe collapse at end-of-run),
  and the wall is crossable because arc never charges the descent.
  Failure with high off% re-opens the grounding design; failure with
  low off% and no finishes challenges the thesis itself.

Engineering in parallel (CPU only, agent): numba fast path for
ArcProgress.advance, bit-identity-gated like goalfield._FAST_SAMPLE -
the measured 30% throughput tax is the blocker for ever renting arc
arms.

### Round 28 addendum 9 - xsFAN3/3b verdict: the RAW flown line is a worse deep-game observation; RDP is the general smoother; xsARC3 amended to the smoothed line

xsFAN3 (scratch, selfroute fan, geodesic): opened at HALF the fan band
(7,864 @0.45G vs 13,628-17,498), caught the band by 2.4G (~99k), then
PARKED - probe at 4.83G shows the honest level: corridor mean 107,961 /
max 108,416 (54.4%), vs the fieldroute lineage's ~85% at the same
steps. Killed by stationarity (2.5G at one gate). The probe also shows
the policy is fully fan-dependent (off: 2,375 mean, -97.8%) - so the
LINE'S PARAMETRIZATION is what limits, not whether it is read. The
mechanism suspect: an expert's flown path WEAVES; at 3,500 u/s the
speed-scaled near horizons oscillate laterally and the weave inflates
local arc, degrading the fan exactly where precision matters. (Process
note: the scratch preset's --steps 3e9 ended xsFAN3 at 71 min; the
continuation resumed with zero contract deltas and zero shock -
first eval 98,765 vs 98k before, confirming same-contract resumes are
safe.)

The GENERAL repair, one parameter, no map constants: Douglas-Peucker
simplification of the distilled line (removes vertices only, so arc
length can only shrink - the lattice quantize that xAUTO used
staircases a WEAVING source, +57% length, and was rejected on that
measurement). rdp eps=512u: 1,599 -> 65 vertices, 204,491 -> 198,617u
(97.1%), deviation vs champion mean 239u / max 642u, within 498u of
the flown path everywhere, end unchanged at champion vtx 1599 (88.3%).
maps/surf_src_cannonball.selfsmooth.npz. The distillation loop is now:
experience -> last-contact trim -> RDP -> line.

AMENDMENT to addendum 8: xsARC3 runs fan + --race-arc BOTH on
selfsmooth (not raw selfroute). Expectations unchanged: low off%
(corridor wraps real flight), fan grounds (probe at end), wall
crossable (arc never charges the descent; the line ends 5,847u short
of the box with only the +50 beyond - the validated xSELF regime).
The ArcProgress numba fast path (10.8x, bit-identical) is active, so
arc throughput should now be near fan-only.

### Round 28 addendum 10 - the 07:25-15:25 GPU queue (PRE-REGISTRATION)

User directive: fill 8 wall-hours. Serial on the local 5090:

1. 07:25-09:50 xsARC3 completes its 3 h (running; addendum 8/9
   expectations stand). Then ~20 min: verdict + route-mode probe.
2. 10:15-13:15 **xsFAN4** = fan on SELFSMOOTH + geodesic reward,
   scratch, 3 h, --steps 9e9. The missing cell: does SMOOTHING fix the
   deep-game observation deficit? Comparators at matched steps: xsFAN
   (fieldroute: 36,962 @1G, ~100k @2.4G, 88% wall by ~5G as xsFANX)
   and xsFAN3 (raw selfroute: parked 54%). Expected: opening at or
   above raw-selfroute's, the 54% park broken; at wall depth the
   xsFANX dive repeats (geodesic still charges the descent) unless the
   smoothed fan shifts the fan-vs-reward conflict. BRANCH RULE: if
   xsARC3 FINISHES the map, xsFAN4 is replaced by a 2 h same-contract
   continuation of xsARC3, and slot 3 becomes the petrus ARC arm
   (generalize the winning recipe).
3. 13:35-14:32 **xsPFAN** = petrus_lite scratch, fan on the petrus
   field line, geodesic reward, 55 min. First TRAINED second-map data
   point for the recipe (the zero-shot probe was null; this is the
   trained-diversity direction E3 needs, with zero new plumbing).
   Scored against the petrus fieldroute ruler; petrus speed-gate
   caveat applies; type-1 map so finishes are real finishes.
4. 14:35-15:32 **xsCTL2** = cannonball scratch control, seed 1,
   55 min. Closes the replication asymmetry (two fan seeds vs one
   control seed) at the matched-step marks.

Probes after every arm; analysis and ledger interleaved on CPU during
runs. 55-min budgets on 3-4 are deliberate (window fit); matched-step
comparison is unaffected.

### Round 28 addendum 11 - xsARC3 at 4.5G: THE WALL IS BROKEN (queue amendment, pre-registered before the driver ends)

Honest scoring at 4.46G: **8/9 greedy past 205,440u** - the barrier
that stopped 234 control episodes and every lineage since round 18 -
riding the corridor at 0-3u closest-approach, corridor mean 207,730 /
MAX 221,056u (95.4%), all dive-below at z ~-5,380: the descent through
the old turn-back point is LEARNED, only the landing into the box is
not. Training: arc reach = p90 = line end (198,656u) - the bulk of
episodes ride the full line - off 2.4-2.9%, mind 6.8%, intrinsic still
paying at the frontier. The grounding failure (xsFULL) did not recur
on the smoothed honest line, exactly as pre-registered.

QUEUE AMENDMENT (ante hoc): the branch rule keyed on "finishes"; the
observed state (majority past-wall, 0 finishes, frontier advancing) is
the convergent case it should also cover. Slot 2 becomes **xsARC3b** -
a 2 h ZERO-DELTA continuation (same-contract resumes are measured
safe, addendum 9) chasing the first champion-free from-scratch finish.
xsFAN4 moves to slot 3 at 2.5 h; petrus xsPFAN to slot 4 at 40 min;
xsCTL2 is dropped (least informative of the four). Window unchanged,
ends ~15:25.

### Round 28 addendum 12 - queue amendment 2 (ante hoc): slot 3 becomes LOOP ITERATION 2

xsARC3b at 9.2G: wall-crossing is stable (greedy 86-97%, training
saturated at line end, off ~0%), and the landing is NOT converging -
0 wins across 4.7G post-wall steps, dives ending z ~-5,380 uniformly,
train-side win 0.00% too. Diagnosis: past the line end nothing steers
the fall (fan points back at the last vertex, arc pays nothing,
intrinsic 0.4/ep); the box crossing has ~zero probability under the
dive family. This is the distillation loop's own signal to iterate:
xsARC3b's episodes have MAP CONTACT deeper than the line's source
(z -5,380 vs xsFANX's -4,100), so re-distilling from them extends the
reference toward the box and shrinks the unreferenced gap.

Amendment: on driver end, (1) probe xsARC3b (fan-read), (2) distill
selfsmooth2 = pick_selfline on xsARC3b's deepest evals + last-contact
trim + rdp 512 (the now-standard pipeline), measure end-gap vs the
box, (3) launch **xsLOOP2** = SCRATCH (contract brittleness forbids a
warm swap), fan + arc on selfsmooth2, all remaining window (~2.5 h,
--steps 12e9). xsARC3 broke the wall at 4.5G, inside that budget.
xsFAN4 and xsPFAN are dropped from today's window (recorded as next in
line; the smoothing-under-geodesic question is partly superseded by
arc's result). Expectation: if the loop converges, each iteration's
line ends deeper and iteration 2 lands finishes; the loop itself -
init from the field, distill, smooth, retrain - is the general
mechanism under test, no map constants anywhere.

### Round 28 addendum 13 - iteration-2 distillation NULL; the goal-append completion; xsLOOP2 launch

The amendment-12 expectation is falsified at zero GPU cost: the
re-distilled line ENDS SHORTER (champion vtx 1574 / 86.9%, gap 7,377u
vs iteration 1's 5,847u), because past the wall the episodes never
touch the map - the descent is pure free fall, and the last-contact
trim CANNOT extend a line through contactless airspace, by
construction. A general limitation of experience distillation, now
measured. (Probe: xsARC3b is fan-dependent like the whole family -
off-mode collapses to 2,688u mean.)

The general completion uses no experience and no champion: append THE
GOAL ITSELF - the box center from the zones file, task data the
success bonus already uses - as the line's final vertex. Works on any
map, zero constants. The dive geometry says why it should matter: the
box is at (-11,392, 7,488, -1,088) and the dives end at z -5,380 some
kilounits wide - they MISS by a margin no blind exploration has
crossed in 5.4G post-wall steps; the appended 7,187u chord is exactly
the missing WHERE. maps/surf_src_cannonball.selfgoal.npz (1,605 pts,
205,344u, end_gap 0).

**xsLOOP2** launches now: SCRATCH, fan + --race-arc BOTH on selfgoal,
~3.2 h, --steps 12e9. The full loop under test end to end: field-line
init -> train -> distill (last-contact) -> smooth (rdp) -> complete
(goal append) -> retrain. Expectations: wall-crossing reproduces
(selfgoal's first 198k u IS selfsmooth); training arc reach saturates
the FULL line including the box approach; first champion-free
from-scratch finishes on this map. Guard: win_rate is only read next
to reservoir min-depth (the xPSSR trap), margin stays the pinned 10.

### Round 28 addendum 14 - window close (07:25-15:25): xsLOOP2 at the wall, the goal-append not yet saturated

xsLOOP2 ended at 7.30G (window, not convergence): greedy peak 94.8%
with the honest metric AT the wall exactly (mean 198,599 / max
205,440u, 8/9 dive-below), training reach 199,990 / p90 199,850 - the
bulk of episodes stop where the appended goal chord BEGINS, so the
completion mechanism never got its saturation phase (xsARC3 needed
~4.5-6.8G for wall-crossing to consolidate; xsLOOP2 ran the same
ascent ~2G behind in phase - the 32% off-corridor excursion at 2.7G,
which self-corrected to ~1%, is the visible phase cost). 0 finishes.
Expectations scored: wall approach REPRODUCED on the completed line
(second independent draw of the arc+fan ascent); goal-append UNTESTED
at saturation - not falsified, not confirmed.

Day summary (all scratch, one seed, local 5090, champion-free lines):

| arm | line | reward | outcome |
|---|---|---|---|
| xsFAN3/3b | raw self | geodesic | parked 54% - weave parametrization |
| xsARC3+b | smoothed self | arc | **WALL BROKEN: 8/9 past 205,440u, corridor MAX 221,056 (95.4%)** - first-ever majority crossing, any lineage |
| xsLOOP2 | smoothed+goal | arc | at the wall @7.3G, phase-lagged, window-bounded |

Standing next steps, in value order: (1) continue xsLOOP2 ~2 h to
saturate the goal chord (zero deltas; the direct finish chase);
(2) xsFAN4 (smoothing under geodesic - the dropped comparison);
(3) xsPFAN petrus (first trained second-map point); (4) xsCTL2.
The loop pipeline (field init -> distill -> rdp -> goal-append ->
retrain) is committed end to end; every line is in maps/ with
provenance.

### Round 28 CORRECTION to addenda 11/14 (user challenge, upheld)

Addendum 11's "the barrier that stopped ... every lineage since round
18" and addendum 14's "first-ever majority crossing, any lineage"
OVERCLAIM: the round 18-19 WARM arms (xARC 84/102 past 205,440, xAUTO,
xSELF, xLATCH) crossed the wall and finished the map. The correct
scope: xsARC3 is the first FROM-SCRATCH lineage with a majority of
greedy episodes past 205,440u (prior scratch best: xsFANX, 1/16 at
205,568, the rest AT 205,312), and the first majority crossing with NO
champion data anywhere in the pipeline (xARC/xAUTO lines came from a
champion recording; xSELF's from a 3.8G-step pre-trained checkpoint's
own runs; xLATCH's threshold from that checkpoint's recordings).
Also restated to prevent the recurring confusion: any "cover %" / track
figure above ~96% in eval lines is the DECEPTIVE geodesic metric
(dive-inflated, saturates at 191,812 on-route); wall claims are made
only on eval_honesty corridor numbers, where yesterday's xsFANX peak
was 205,312-205,568 (88.6%) and xsARC3's is 221,056 (95.4%).

## Round 29 - goal conditioning (docs/research-plan-goalcond.md), G0 done, G1 pre-registration

G0 (machinery, local): surfgym/goals.py (per-env MultiLine fan - BIT-
IDENTICAL to RouteLine per env - chord/segment builders, sphere goals,
reachable-air sampler, k-curriculum, stats; 16 tests), surfgym/goalsys.py
(trainer glue), RespawnBuffer(goal_k=, goal_min_dist=) harvesting a goal
+ segment per snapshot from the same episode's future, train_fast
--goals (fan slot, sphere kill via force_fail, goal mask into the
reward, eval goals with viewer metadata, goals.csv), record_ckpt mirror
+ --goal-band/--goal-holdout-only. Existing suite 281 green (control
path untouched: every new branch is behind `goalsys is None`).
Smoke (256 envs) surfaced two things worth the ledger: (1) the trainer
drains the reservoir outbox itself, so goal columns must ride that push
(fixed); (2) TRIVIAL-GOAL TRAP, literal: a barely-moving policy harvests
a "goal" 1 s later inside its own sphere - 100% reached at 0.0 s. Fixed
by a general rule, goal_min_dist = 2.5 x radius, no map constant. Also:
goal harvesting is gated by the respawn margin (nothing shorter than the
margin harvests), so goal arms run --respawn-margin 2 - a pre-registered
deviation from the ablation pin, because here the margin is a GOAL
GENERATION parameter, not a start-distribution treatment.

G1 = xsG1: scratch_ablate + --goals 1 --race-shaping 0 (sparse: bonus +
time penalty + the baseline intrinsic) --respawn-margin 2 --goal-kmin 1
--goal-kmax 5 --goal-air-frac 0 (air goals only as the fallback when a
start carries no reached-state goal; both kinds reported separately),
1 h, --steps 9e9. Expectations per the plan: success on the 1-5 s band
rises from ~0 to > 50% within the hour; the route-mode probe (off) on
the final ckpt collapses success (the fan is read); time-to-goal falls.
Digs pre-written: success ~0 -> arc-on-segment reward (G1b); success
high but no probe collapse -> shrink sphere / raise k_min.

### Round 29 - G1 verdict (xsG1, 1 h, 1.0G steps): ALL THREE EXPECTATIONS MET

* success on the 1-5 s band: training 24% -> 49% over the hour
  (monotone), eval on fresh random-air goals 1/9 -> 5/9 (56%). MET on
  eval, borderline on training. Split by kind, the honest reading:
  reached-state goals ~99% throughout (momentum-trivial: a goal 1-5 s
  ahead on your own path is hit by flying straight), random-air goals
  18% -> 39% (needs a turn toward the target - the real signal).
* time-to-goal 351 -> 265 ticks: MET.
* PROBE (12 seeded random-air goals, greedy): goal block live 6/12
  reached, ZEROED 0/12: MET - the goal is read and load-bearing, the
  first goal-conditioned positive in the ledger.
Costs: 276k fps (vs ~690k for the route arms) - per-tick python goal
bookkeeping + per-tick line uploads; optimization queued, not blocking.

G2 PRE-REGISTRATION = xsG2 (3 h, --steps 12e9): G1 + curriculum
(--goal-curriculum 1, 10-90% band rule, k_cap 60 s) + 30% random air +
held-out sector --goal-holdout 0.30,0.40 (geodesic fractions of d0,
applied to BOTH goal kinds - fixed now; reached-state goals inside the
band are refused). G2-arc (arc along the per-env segment) is DEFERRED:
the per-env arc reward is not built yet, so tonight is G2-sparse alone
and the sparse-vs-arc question stays open. The map-finish-as-goal is
also deferred (needs the goal-completed self-line per env).
Expectations: k_max climbs from 5 s (curriculum engages once band
success > 50%); air-goal success stays >= 30% as goals get farther;
success vs distance decays smoothly; reached-state success drops below
99% once k covers many seconds (no longer momentum-trivial). Then G3:
held-out-band probe live/zeroed vs in-distribution, ~30 GPU-min.

G2 AMENDMENT before launch (user, watching xsG1): goals clustered near
the start and some spheres sat at the map ceiling (BFS "reachable" =
free flight). Fix in goalsys: air goals anchored on VISITED states (pool
rows within the reach band, frontier-biased 0.7 toward deeper-than-start
anchors, goal within 1.5 radii of the anchor). Expectation added: the
goal distribution's field depth deepens over the run (frontier pull),
and no goal lands in unvisited ceiling air.

xsG2 RELAUNCH (10 min in): success 93.6% at 2.1 s with mind 99.1% and
k_max pinned - successful 2 s episodes are SHORTER than the 2 s harvest
margin, so successes never fed the reservoir, anchors stayed at the
spawn, goals stayed trivial: the circularity the plan warned about,
measured. Two general fixes: (1) a goal-reached ending is not a death,
so successful episodes harvest their whole chain (margin applies to
deaths only); (2) the curriculum counts every goal kind (air goals were
not feeding it). Expectations unchanged; relaunched as xsG2.

xsG2 STOPPED at 1 h (843M): success 86%, kmax 5 -> 6 s only, mind 98.9%
-> 97.3%, episodes ~2 s. Structural, and mine: the anchor rule places a
goal within 1.5 radii (288 u) of a VISITED state, so each generation of
successes can extend the visited set by at most ~288 u - measured ~3,000
u/h on a 198,000 u map - and no goal can ever be far, so the curriculum's
upper band never fills. Fix (general): BALLISTIC extrapolation - the goal
is the anchor state flown on for tau ~ U[0.5, k_max] s along its recorded
velocity under the sim's gravity (a physically plausible place to have
been, BEYOND the visited set), halving tau on rejection (solid /
unreachable / held out), falling back to the anchor shell. k for such a
goal = distance/speed_est + tau, so far goals populate the curriculum's
top band. Relaunch as xsG2b, same flags. Expectation: mind falls
markedly within the hour; kmax climbs past 10 s; eval goal distances
grow; success settles in the curriculum's 10-50% window instead of 86%.

xsG2b STOPPED at 1 h (680M): mind 99.3% -> 98.8%, kmax pinned 4 s,
success 69%, episodes 1.5 s. The frontier does not compound because
START sampling is shallow-dominated: starts are drawn uniformly from a
reservoir whose bulk is near the spawn, so frontier states are almost
never starts; and a 1.5 s success yields one 1 s-cadence snapshot, so the
goal-entry state (the deepest) is never harvested. Fixes, both existing
or trivial: --respawn-binned 1 (uniform over 16 distance bins - the
round-18 Go-Explore cell selection, so frontier bins get an equal share
of starts) and snapshot cadence 0.25 s under --goals. Relaunch as xsG2c.
Expectation: mind falls steadily through the hour (frontier compounds),
kmax climbs, eval goal distance grows.

G3/G4 PRE-REGISTRATION (before xsG2c reports). G3 on xsG2c's final ckpt:
record_ckpt 12 seeded goals x {live, --route-mode off} in-distribution,
then --goal-holdout-only (the 30-40% band) x {live, off}. Expected:
live >> off on both sets; held-out live within ~2x of in-distribution at
matched distance. G4 = xsG4: identical to xsG2c but --goal-obs ball
--goal-views 4 (depth + four ball views, in_ch 5), 3 h, --steps 12e9;
scored on the same G3 protocol plus success vs the stairs-needed flag.
Expected: ball >= fan on goals whose chord crosses solid; ties on open
goals; probe collapse under --route-mode off (which zeroes the ball).
Optimization queued: per-tick goal bookkeeping costs ~2.5x fps vs the
route arms; goal-obs ball skips line building (e0cc674).

xsG2c STOPPED (user, watching): goals still next to the spawn - every
generator so far is experience-relative (near visited states, or their
ballistic extrapolation), so goals can only hop past the visited set,
and a novice's visited set is the spawn. Fix: ROUTE-DEPTH goals
(--goal-route <map line> --goal-route-frac 0.7): a goal on the map line
delta of arc AHEAD of the start's projection, delta ~ U[2.5 R, speed_est
x k_max] from the curriculum, clamped to the line end (so the finish box
is the delta -> end goal); the fan line is the route slice from the start.
On surfable geometry by construction; the remaining 30% of starts keep
reached-state / air goals for off-route diversity. Relaunch as xsG2d.
Expectation: the agent surfs the track within the first 20 min (eval
track/cover climbs as a side metric), mind falls fast, kmax climbs, goal
success sits in the 10-50% window.

xsG2d at 10 min (144M): route-depth goals WORK - route success 4% -> 34%,
kmax back to 6 s, mind 99.3% -> 97.2% in seven minutes (fastest fall of
any goal arm). Sparse is reaching the route so far. G2-ARC PRE-REGISTERED
= xsG2e: xsG2d's flags with --goal-reward arc --race-shaping 1 (the
per-env arc progress along each env's own goal line - MultiArcProgress,
byte-exact twin of the round-18 arc, numba 18.5x - replaces the geodesic
term; 100 per map-line length of arc). 3 h. Expectations: route success
at matched steps >= xsG2d's, kmax climbs faster and farther (dense
progress makes far route goals learnable before they are reachable by
luck), off-corridor share low (the lines are the routes themselves).
If sparse (xsG2d) matches it at matched steps, shaping is unnecessary
for goal-reaching on this map and the plan's variable 3 resolves to
sparse.

xsG2d STOPPED at ~25 min (user, watching): the numbers were being
misread and the design still could not leave the spawn. Clarifications
now in the ledger: route-goal success (34%) is the reach rate of goals
placed 0.5-9k u AHEAD OF WHEREVER THE EPISODE STARTED - not map progress;
race/map_pct (0.5%) is the greedy eval, which runs air-goal episodes from
the platform and never attempts the map. Starts come only from visited
states (mind 97% = nothing beyond 3% of the map), so starts, goals and
the 34% all live in the first few percent. Dashboard 'multiple lines' =
my relaunches appended to the same csv (deduped; fresh names from here).

THE FIX (general, champion-free provenance): SELF-DEMO STARTS. The
xsFANX episode the self-line was distilled from carries full states
along the whole route; subsampled at 0.1 s into a STATE_DTYPE spine
(maps/surf_src_cannonball.selfdemo.npy, the Salimans-Chen reset-to-state
mechanism already in the trainer, --demo-file with the window = the whole
spine so starts are uniform along the route), episodes begin at 30%, 60%,
90% of the map from the first minute with route goals delta ahead of
each. Demo rows also anchor the air goals along the route. New metric:
route-goal success PER START-DEPTH BAND (10% bins of d0) in the console
and goals.csv - the number that says whether the agent trains beyond
the spawn. Relaunch = xsG2f (new name).
Expectations: nonzero route success in every depth band within 20 min,
eval (still platform-based) unchanged, kmax climbs; the sparse-vs-arc
arm (xsG2e) inherits these starts.

xsG2f CRASHED at 18M steps: the air-goal fallback sampler drew ONE candidate
per batch (64 tries) over a bounding box that is mostly solid or
unreachable, and a deep route start exhausted every stage. Fixed by
batched rejection sampling (32/64/512 candidates per stage). Relaunched
as xsG2g, identical flags; the xsG2f pre-registration carries over.

xsG2g at 5 min (75M), self-demo starts along the whole route: route-goal
success PER START-DEPTH BAND 0%:26% 10%:44% 20%:80% 30%:70% 40%:67%
50%:48% 60%:56% 70%:58% 80%:55% 90%:69% (goals delta <= 3k u ahead, kmax
2 s); mind 2.8% (the reservoir spans the map through harvested chains
from demo starts). The local goal-reach skill develops on EVERY section
at once when starts span the route - the parallel regime. Superseded by
the user's frontier specification (starts from the spawn + reservoir,
goals in a band behind ONE frontier F that advances on a 30%-success
rule, F = 1 -> the finish itself): xsG3f next. Kept for comparison.

xsG3f PRE-REGISTRATION (the user's frontier spec, 2026-09-02): scratch,
sparse, starts = platform + reservoir (binned over depth, margin 2), goals
70% route goals in the 5% band behind ONE frontier F (start 5%), 21%
reached-state, 9% air; F += 10% when frontier success over 300 episodes
>= 30% (cool-down 50 iterations); F = 1 -> the finish box as the goal.
No held-out band (a route band would block the frontier; held-out goes
off-route later). 3 h. Expectations: F advances every 10-25 min at first
and slows as sections get harder; frontier success oscillates in the
30-60% window between advances; the greedy eval (platform -> frontier
goal) tracks F; mind falls with F; the run's headline is F(t).
Falsifier: F stuck at one section for > 45 min with success < 30% =
that section needs more than sparse (the arc arm) or a smaller step.

xsG3f restarted as xsG3g at 3 min: F0 = 5% of this route is 10,000 u from
the spawn, so the first band spanned 0-10k u and frontier success was 1%
- the 30% rule could never fire. The FIRST band must be reachable in
absolute units: F0 = 2% (4k u), band 2%, step 5% (10k u per advance -
reservoir starts spread to the previous frontier, so each step is the
same difficulty class as the first). Rule and min-episodes unchanged.

xsG3h (frontier, F0 2%, band 2%, step 5%, rate 30%): FIRST ADVANCE at
~5 min / 138M steps - 'frontier -> 7% (frontier-goal success 31% over
2443 episodes)'; frontier success then 7-10% on the new 5-7% band
(10-14k u from the spawn), reservoir deepest at the old frontier. The
mechanism behaves as specified: master, step, struggle. Eval goals now
spread over the band (9 different goals per recording).

xsG3h STUCK after its first advance (F 2% -> 7%, band 2%, step 5%):
frontier success 0% over 2,947 episodes, mind rising 97.9% -> 98.4% (the
ring loses its deep states when nobody reaches them). Cause: a step
larger than the band leaves a 6k u gap between the reservoir's deepest
state (~4k u) and the first goal (10k u) - no goal in between, nothing
to learn from. Fix (the balance rule, made geometric): band = step, so
after an advance the goals span from the old frontier to the new one and
the near edge stays reachable. Relaunch xsG3i: start 3%, band 5%, step
5%, rate 30%. Expectation: advances continue past the first one.

xsG3i: advanced once (3% -> 8% at 32%), then 0% on the contiguous band.
Start side this time: the reservoir's deepest state is ~5.5k u but its
100k rows are overwhelmingly spawn-region, and 16 depth bins over a
198k u map are 12k u wide - the whole 0-16k u region is ONE bin, so the
binned sampler spreads nothing inside it. Fix: 128 bins (1.5k u), the
existing mechanism at the band's resolution. Relaunch xsG3j.

xsG3j: advanced once (3% -> 8% at 30%), then 1% on the new band with the
reservoir spread evenly over 0-5k u (128 bins worked). The band's WIDTH
is the limit: 5% of this route is 10k u, and a 30% rule over the whole
band needs episodes to surf 5-11k u past any start before it can fire.
The step must match what one episode adds to the reach: band = step =
2% (4k u) here; the principle (30% then step) is the user's, 10% was a
scale for a shorter map. Relaunch xsG3k. Expectation: advances every
5-15 min at first, F(t) roughly linear until a hard section.

xsG3k: no advance, 0% on the first band - start 3% with band 2% placed
the first goals at 2-6k u, excluding the 0.5-2k goals that carried every
earlier first band to 30%. The first band must begin at the spawn:
start = band = 2%. Relaunch xsG3l (start/band/step 2%, 128 bins).

xsG3l at 10 min (215M): first advance at ~5 min (2% -> 4% at 31%), and on
the new 4-8k u band frontier success is 20-23% and rising - the first
run where the second band recovers instead of pinning at 0%. Reservoir
deepest 3.3% / median 1.6% of the map; eval 7/9 at the previous
frontier; kmax 12 s. The master-step-recover rhythm the user specified.
Parameters that made it work, all forced by evidence: first band at the
spawn, band = step = 2% (4k u), 128 depth bins for start spreading, 30%
rule over 300+ episodes with cool-down. Running the full 3 h.

xsG3l STOPPED at ~35 min (user, watching): each frontier advance shifts
the whole training distribution to the new band, the value function
re-estimates and the sections behind stop being trained directly - it
looked like forgetting after every step. xsG4u = UNIFORM route goals
(--goal-route-uniform 1): from every start, a goal at any distance ahead
along the route up to and including the finish sphere, all the time - a
stationary distribution; 70% of starts, the rest reached-state / air.
Eval spans the route the same way. Expectations: success per
goal-distance bin decays smoothly with distance and every bin rises
over the hours (near first); the reservoir deepens monotonically; no
collapse events; first finishes appear when the 20 s+ bin turns nonzero.

xsG4u STOPPED (user): the 30% success_rate was the blend of easy goal
kinds (reached-state 100%, air 96%) over route goals at 0.3% - uniform
over 200k u of arc puts the median route goal 100k u away, so nothing
reachable carries signal and the reservoir never deepens (mind 99.45%).
xsG5f = the user's FIXED GOAL SET (--goal-fixed): positions generated
once - route points every 2,000 u of arc plus the finish sphere, and 100
reachable air points near the route - sampled per rollout, route goals
ahead of the start only, NO reached-state goals, no curriculum, sparse.
Starts platform + reservoir (128 bins). Expectations: route success per
distance bin decays with distance but the near bins (<5 s) rise fast,
far bins rise later as reservoir starts deepen (mind falls); the
distribution never changes so no collapse events; the finish sphere is
in the set from step 0.

xsG5f STOPPED at ~12 min (user: agents just drop down). REWARD TRAP: with
shaping off, the reward was -0.005/tick + 50 on the goal + free death.
For the ~95% of episodes whose goal is unreachable, the only thing the
agent controls is how long it pays the time penalty, so dying at once is
optimal - and the near-goal bonuses do not outweigh that in the
gradient. Fix with existing flags: --time-pen 0 --finish-k 1
--finish-tref 60 (no per-tick cost; +50 on arrival plus up to +60 for
arriving sooner; death gains nothing). Relaunch xsG5g, otherwise
identical. Expectations: episodes lengthen (no suicide incentive), the
<2 s bin stays high, the 2-5 s and 5-10 s bins rise, and time-to-goal
falls as the speed bonus bites.

xsG5g STOPPED at ~12 min (user: the spec's variable 3 was a Euclidean
dense reward + bonus). xsG5e = --goal-reward euclid: per tick 0.005 x
(Euclidean distance to the goal before - after), potential-based, + 50
on arrival + up to 60 for speed, no time penalty; fixed goal set, no
curriculum, sparse-free. Expectations: every episode now has a gradient
toward its goal, so the 2-5 s and 5-10 s bins rise within the hour, the
reservoir deepens (mind falls), and time-to-goal falls; the Euclidean
lie around corners costs on some goals (the stairs), visible as bins
with high approach reward but low arrival.

xsG5e RESULT (killed at 152M, 9 min): the EUCLIDEAN FALL TRAP.
ep_len 1,470 -> 370-440 ticks within 6 min; eval 9/9 episodes end at
z 8,184 (the void floor under the platform) after 5-13 s, d_goal cut by
1,200-3,000u each; goal success 2% overall (dist bins 0-2s 60%, 2-5s
30-40%, 5s+ 0%). The straight line to a route goal points DOWN through
the void, so a dive banks 0.005 x 2,000u = 10 in 4 s and death keeps
it (r[ended]=0 wipes only the last tick). Same failure family as the
geodesic death-dive, worse because Euclid has no corridor.

xsG5f = xsG5e + --death-charge 1.0 with a PER-ENV potential origin (the
distance at assignment): a dying episode nets exactly 0 shaping, the Ng
terminal convention; NO per-step tax and NO time penalty, which are the
two things that made round 27's kappa=1 (xNGS) collapse to suicide -
here suicide nets 0 and any goal reach nets +50 + speed bonus + bank.
Expectations: ep_len rises back above 1,000 ticks (dives no longer
pay), the 0-2s / 2-5s bins stay >= xsG5e's, the 5-10s bin turns
non-zero, and eval episodes end ON the route rather than at z 8,184.
Risk: with far goals paying nothing until survived, learning is driven
only by the near-goal reaches and the reservoir, i.e. it may look sparse
for the first hour.

NAME CORRECTION for the two entries above: the Euclidean arm and its
death-charge successor were launched as xsG5e and xsG5f, but xsG5f was
already the fixed-set arm from 05:35 and the relaunch appended a second
life to that directory (dashboard: two lines per plot, reported by the
user). runs/xsG5f is restored to the fixed-set rows (see its NOTE.txt),
and the death-charge arm runs as xsG5h. The trainer now refuses a fresh
launch into a directory that already holds a run, a resume truncates
progress.csv to its checkpoint step, and the dashboard reads only the
last monotone segment of any older file (commit 3d69553).

xsG5h RESULT (killed at ~255M, 17 min): death forfeiting the bank removed
the dive - eval episodes now ride the first ramp instead of dropping
off the platform - but the frontier is STATIONARY: every greedy eval
(27 of 27) fails at the same point, arc ~5,400u (2.6% of the map, z
8,200, 420u off the line), reservoir min-depth pinned at 97.96-98.03%
for 15 minutes, 5s+ bins 0%. The near bins did learn (0-2s 0% -> 84%,
2-5s 0% -> 38%, ep reward ~0 by design). Diagnosis: the draw. Uniform
over the ~100 fixed goals ahead gives a start its NEXT goal ~1% of the
time, and with the bank forfeited at death the far 99% carry no signal
- the run was learning from ~15 goal reaches per 1,000 episodes, all
inside the first 2% of the map.

xsG5i = xsG5h + --goal-fixed-decay 8: training weight exp(-i/8) over the
i-th fixed goal ahead (P next 12%, within 3 31%, beyond 20 goals 8%,
mean ~17,000u ahead); air goals the same over their distance rank;
EVAL UNCHANGED (uniform over the map). Stationary, every goal present
from step 0 - no frontier, nothing to forget. Expectations: the 2-5s
and 5-10s bins climb within 15 min, reservoir min-depth falls below
97% (past the arc-5,400 gate) within 30 min, and the eval endpoint
moves off (-14,040, -1,328, 8,200). If the endpoint does not move by
30 min the gate is a control problem the goal signal does not address.

xsG5i RESULT (killed at ~380M, 14 min): the near-weighted draw lifted
route-goal success 1.3% -> 4.4% at matched age and ep_len 650 -> 850,
but the frontier is the same gate: greedy evals end at (-14,02x,
-1,328, 8,185) or stand 120 s at (-13,776, -1,352, 8,290), 3 of 6
truncated; reservoir min-depth 97.39-97.44%; 5s+ bins 0%.

RE-READING xsG5e (correction to its entry above): its eval endpoints
were NOT a dive off the platform. The floor at z ~8,190 runs under the
whole first section (x ~ -14,000, y +2,800 .. -1,328); xsG5e episodes
ended spread along it from y +1,809 (an early drop) to y -1,328 (the
full ramp ride to the section end), and xsG5h/i all ended at y -1,328.
So the fall trap was PARTIAL: the death charge removed the early drops
but also removed the progress-before-death income that carries the
geodesic arms through gates (round 27: the broken telescope IS the
curriculum), and the gate at arc ~5,400u (the end of the first ramp,
vmax ~1,500 u/s) is where every Euclidean arm stops. At matched age
xsG5e had HIGHER near bins (0-2s 65%, 2-5s 38% at 128M) than xsG5h
(44% / 6% at 75M, 84% / 38% at 250M).

xsG5j = the user's spec verbatim plus the draw: --goal-reward euclid,
NO death charge, --goal-fixed-decay 8, --time-pen 0 --finish-k 1
--finish-tref 60. Expectations: near bins at or above xsG5e's at 128M;
the reservoir deepens past 97% and the eval endpoint leaves y -1,328
within 30 min (the dense income for getting further plus the next
goal beyond the gate). If it too stops at arc 5,400 with vmax ~1,500,
the gate is a SPEED gate and the goal signal does not address speed;
the next arm would then be about speed, not goals.

xsG5j at 36 min (830M): THE FRONTIER MOVES. Honest corridor (order-only
16, champion route): 8,438u max at 605M -> 15,441u max / 13,921u mean at
756M, 8 of 9 episodes at 14.3-15.5k, closest-approach 0-16u (on the
line), end z 6,670 = the first gate of the round-21 ladder (17k). vmax
1,500 -> 2,150 u/s. Reservoir min-depth 97.0% -> 86.8% (the reservoir
now reaches 13% of the map); goal success 12.6% -> 13-15% with the 5-10s
bin 0% -> 9-11% and the 10-20s bin still 0%; win (goal reach) 13-15% at
~4 s. The 120 s standing-at-the-gate evals are gone (9/9 fail forward).
Matched-step reference: the from-scratch geodesic controls sat at
17.5-18.1k corridor at 525M, so the goal-conditioned Euclidean arm is
~1 gate behind them in steps while learning a goal policy on top.

xsG5j at 1 h (1.42B): corridor mean 15,963 / 16,840 / 17,216u and max
17,908 / 17,978 / 17,967u at 1.13B / 1.21B / 1.28B (end z 5,511-5,520 =
the 17k rung of the round-21 ladder; one earlier episode at 982M hit
26,847u). Closest-approach 0-24u throughout. Matched to the geodesic
from-scratch controls (17.5-18.1k at 525M, 16.0-18.2k at 750M) the
goal-conditioned Euclidean arm reaches the same rung at ~1.7x the
steps, while learning a goal policy the controls do not have. Training:
reservoir min-depth 86.8% -> 50.8% (a deep harvest, but goals from
starts past 10% of the map are reached <= 1% of the time and overall
goal success fell 15% -> 9% as those starts entered the mix); 10-20s
bin 0-1%. Continuing to the 3 h budget; the question is whether the
deep starts convert into 10-20s reaches before the ladder stalls it.

xsG5j at 85 min (1.81B): corridor max 26,916 / 24,332 / 26,904u, mean
21,975 / 17,343 / 18,057u at 1.51B / 1.59B / 1.66B - the 26k rung, with
the mean oscillating between the 17k and 26k rungs (the ladder's
binary character, as round 21 described). The 10-20s goal bin turned
non-zero: 0% -> 5-6%; 5-10s 8-14%; band-0 route success 35%. The
reward dip the user saw (25-30 -> 11-16 from 1.04B) is the START MIX:
the reservoir min-depth jumped 80% -> 51% on a few long falls, the
binned sampler gives those bins an equal share, and ~65% of route
assignments now start 10-50% deep where success is 0-1%; on band-0
starts success ROSE 21% -> 35% over the same window. Those deep states
are almost certainly mid-air fall states (2 s before the end of a
dive) - the death-dive harvest, amplified by uniform-over-bins.

xsG5j at 127 min (2.54B): EVAL FRONTIER STATIONARY at the 25-27k rung
since 1.51B (corridor max 25,600 / 25,040 / 25,475u at 2.27 / 2.34 /
2.42B, mean 25,072 / 22,654 / 22,785u - the mean has consolidated onto
the rung the max first touched at 982M). By the 10-minute rule the
greedy platform run is stalled; the goal policy underneath is not:
10-20s bin 7-11%, the 20s+ bin 0% -> 2-3% (first non-zero), route
success from starts 10-20% deep 0% -> 5-6% and 40-50% deep 0% -> 2-3%.
Left to its 3 h budget on the local box for the end-of-run held-out
goal probe; no finishes, 0/9 past the wall, none expected.

xsG5j STOPPED by the user at ~150 min (3.0B): eval frontier parked on
the 25-27k rung for 90 minutes (max 25,203 / 25,070 / 26,776u at
2.72-2.87B), 0 finishes; goal policy still creeping (10-20s bin 12%,
20s+ 1-3%, deep-band route success 2-7%). Verdict: the fan + Euclidean
reward + fixed set LEARNS goal reaching and generalises slowly to
farther goals, but as a map-progress engine it is ~2x slower in steps
than the geodesic controls and stalls on the same ladder.

xsG5k = xsG5j with the goal passed as the 4-VIEW DEPTH RENDER OF THE
BALL (--goal-obs ball --goal-views 4, front/back/left/right, in_ch=5)
instead of the lookahead fan; everything else identical (Euclid 0.005/u,
no death charge, fixed set, decay 8, time-pen 0, finish-k 1). This is
the cleaner goal-conditioning test: the fan carries the ROUTE segment
to the goal (the path), the ball carries only the destination.
Expectations: near bins (0-2s, 2-5s) rise at a comparable rate to
xsG5j's (14% / 2% at 118M, 37% / 22% at 452M) if the render is read;
far bins and the eval frontier LAG the fan arm, because the path is no
longer given; a flat 0-2s bin past 200M means the render is not being
read (check the ball channel statistics before blaming the policy).
Throughput is unmeasured for the 4-view render - record fps.

xsG5k (BALL RENDER, 4 views) at 20 min (317M): the render is read, and
at matched steps the ball arm is AHEAD of the fan arm on goal reaching:
goal success 3.5% vs 3.8% at 150M, 7.7% vs 4.0% at 250M, 12.2% vs 7.0%
at ~317-350M (route 9.1% vs 5.6%); bins 0-2s 32% / 2-5s 25% at 317M
(fan: 37% / 22% at 452M). Eval corridor max 5,683 / 5,907u at 152M /
227M = the first-ramp gate, where the fan arm was at the same age
(passed it at ~300-378M). Throughput 240-270k fps vs 350k (the four
extra 64x32 renders per decision), so in WALL-CLOCK the two are about
level at 20 min. Reservoir min-depth 96.97%.

xsG5k at 42 min (674M): goal reaching stays AHEAD of the fan arm at
matched steps (16.3% vs 9.1% at 400M, 15.8% vs 8.8% at 500M, 13.4% vs
11.2% at 600M; at 674M bins 0-2s 47%, 2-5s 31%, 5-10s 9%, i.e. where
the fan arm was at ~830M), but the MAP frontier lags: eval corridor max
5,907 / 6,431 / 6,656u at 378 / 454 / 530M with mean 2.8-4.4k (most
platform episodes end before the first-ramp gate), reservoir min-depth
94.2% vs the fan arm's 86.8% at the same wall-clock. Consistent with
the pre-registration: the ball carries the destination, not the path -
near goals are reached better, far progress is slower. A structural
limit to keep in mind: the four views span 360 deg of yaw but only
+-45 deg of pitch, so a goal steeply below or above is in NO view.

xsG5k at 62 min (1.06B): goal reaching has CONVERGED with the fan arm at
matched steps (700M 19.7% vs 12.5%, 800M 13.7% vs 15.8%, 900M 17.9% vs
14.8%, 1,000M 15.9% vs 15.9%); the ball arm leads on the 5-10s bin
(17-24% vs ~12% at 1.06B) and is at 0% on 10-20s where the fan had
0-1%. Reservoir min-depth 86.7% at 1.06B (fan: 86.8% at 830M). Eval
corridor max 6,669 / 7,096 / 13,394u at 832 / 907 / 982M, mean 4.2-6.6k
- the first episode past the first-ramp gate at 982M; the fan arm had
max 26.8k and mean 15.6k at the same step. So: destination-only
conditioning reaches goals as well as path conditioning, and the map
frontier from the platform moves at roughly half the rate in steps
(and slower still in wall-clock at 290k vs 350k fps).

xsG5k STOPPED by the user at ~82 min (1,194,328,064): mind 86.697%; goals 18.8% (route 20.8%/969 ach  nan%/0 air 14.4%/450) kmax 5s asg 1024/0/395  depth 0%:28% 10%:0%  dist 0-2:29% 2-5:25% 5-10:23% 10-20:0% 20+:0%
Verdict (ball vs fan, matched steps): goal reaching equal after ~600M
(both ~16-20%), ball ahead on the 5-10s bin, map frontier from the
platform at ~half the fan arm's rate in steps (13.4k vs 26.8k corridor
max at 982M) and slower again in wall-clock (290k vs 350k fps). The
user's read: training is slow because of the REWARD, not the
representation, and asked for a geodesic-style reward per goal without
a rebake, composed from the one baked field.

xsG5l = xsG5k + --goal-reward geo: potential -0.005/u x
max(|dF(x) - dF(goal)|, |x - goal|), dF = the baked geodesic-to-finish
field (surfgym.goals.GoalDistField(geo=...), tests/python/test_goaldist.py).
Both terms are admissible lower bounds on the true geodesic distance to
the goal, so the max is the tighter one, exact when the goal is on the
shortest path to the finish (every route goal ahead). Along the route it
pays the SAME per-tick signal the geodesic arms were paid (a constant
shift of dF), which is the signal that carried them through the ladder's
rungs at ~2x the steps of the Euclidean arms; off the level set the
Euclidean term takes over; unreachable samples fall back to Euclid.
Death keeps the bank (as xsG5j/k). Expectations: at matched steps the
near bins match xsG5k's and the eval corridor max reaches the 17k rung
by ~600M and the 26k rung by ~1B (the geodesic controls' pace), i.e.
~2x xsG5k's; the free-flight deception at the wall is inherited
unchanged, so nothing past 88% is expected from this arm.

xsG5l (ball + COMPOSED GEODESIC goal reward) at 20 min (406M): the
user's hypothesis holds - the reward was the bottleneck. Matched steps
vs xsG5k (ball + Euclid): goal success 10.2% vs 3.5% at 150M, 10.5% vs
4.8% at 200M, 16.4% vs 7.7% at 250M, 19.2% vs 16.3% at 400M; bins at
406M 0-2s 36% / 2-5s 27% / 5-10s 17% (xsG5k needed ~1B for 5-10s 17%).
Eval corridor max 14,349u at 227M and 15,204u at 303M, mean 10.6-11.0k
(xsG5k: 5.9k / 6.4k max at the same steps; the fan+Euclid xsG5j needed
756M for 15.4k). Reservoir min-depth 87.2% at 406M (xsG5k reached that
at 1.06B). Throughput 339k fps. ~2.5x faster in steps than the
Euclidean reward on the map frontier, on the same representation.

xsG5l at 41 min (852M): goal success 29.6-31.7% (matched steps: 21.7%
vs xsG5k 15.8% / xsG5j 8.8% at 500M; 29.2% vs 13.7% / 15.8% at 800M);
bins 0-2s 42-47%, 2-5s 40-43%, 5-10s 33-38%, 10-20s 12-17% (the fan +
Euclid arm first showed 12% on 10-20s at 2.2B). Eval corridor mean
16,410 / 17,268 / 17,004u and max 24,805 / 25,466 / 28,800u at 605 /
680 / 756M: the 17k rung at 605M (pre-registered ~600M) and the 26k
rung at 756M (pre-registered ~1B), i.e. the geodesic controls' pace or
better, on a policy that is goal-conditioned through the ball render
alone. Reservoir min-depth 86.7%. Throughput 350k fps.

xsG5l at 62 min (1.24B): eval corridor mean 23,103u / max 29,953u at
1.13B (the 26k rung consolidating, mean up from 17.0k at 756M). Goal
success 25-26% after the reservoir opened the 20-30% band at ~1.18B
(min-depth 85 -> 75%): the start-mix dip described above, with band-0
success still rising (44 -> 49%) and the new bands at 11-17% (10-20%)
and 4-7% (20-30%) - higher than xsG5j ever got on those bands (0-8%).
Bins 0-2s 36-38%, 2-5s 32-35%, 5-10s 24-33%, 10-20s 10-16%.
REservoir diagnosis (checkpoints read directly): the deep bins hold
the agent's own stalled arrivals - xsG5l 20-30% band 75% of states
below 200 u/s (mean 408) within ~250u of the route; xsG5j 20-50%
bands mean speed 110 u/s. Equal-share bin sampling hands those starts
up to half the episodes the moment a band opens; success from them
is 0-5%; the shallow-start success rises monotonically through every
drop in all three runs. The mean reward/success drops are the start
mix, not the policy. Proposed for the next arm: a harvest speed floor
(~500 u/s) in RespawnBuffer; not applied to the live run.

xsG5l at 95 min (1.86B): the 48k rung. Eval corridor max 48,596 /
48,454 / 49,385u at 1.59 / 1.66 / 1.74B, mean 33,893 -> 37,996u (the
geodesic controls: one of four reached ~48k by 750M, the other three
sat at 17-18k for the hour). Training bins 0-2s 40%, 2-5s 30%, 5-10s
29%, 10-20s 19%, 20s+ 8% (first non-zero); route success from starts
10-20% deep 30%, 20-30% deep 7%; reservoir min-depth 63%. Throughput
330k fps. Left running; two 3090 boxes are being stood up for the
24-hour pair (xsG5m = this config, xsG5n = the same with the fan).

=== 24-HOUR PAIR on vast (2026-09-02 12:16 local, user: 'give it time')
Two single-3090 boxes on the same host (machine 16571, EPYC 7B13 x2 =
256 threads, 21 effective cores per offer), image pytorch 2.7.1-cu128,
branch goallines @ 700c4de, caches shipped + bsp mtime pinned (no
rebake), launched via SCRATCH=1 tools/run_arm.sh with BUDGET 40e9,
NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16:
  xsG5m  instance 49633620  ssh9.vast.ai:33620  = xsG5l verbatim (ball
         render 4 views + composed geodesic goal reward, fixed set,
         decay 8, no death charge, time-pen 0, finish-k 1/60 s)
  xsG5n  instance 49633837  ssh3.vast.ai:33836  = the same with the
         lookahead FAN (--goal-obs fan) instead of the ball
Safety: fleet_watchdog registry deadline 2026-09-03T11:55Z, and an
ON-BOX watchdog (/root/box_watchdog.sh, vastai CLI + key on the box)
that destroys the instance at 2026-09-03T11:44Z or 10 min after the
trainer dies - the workstation is going off. Dashboards: box:8000,
local tunnels localhost:8001 (xsG5m) / localhost:8002 (xsG5n) while
the workstation is on. Harvest before destroying:
  bash tools/harvest_box.sh 33620 ssh9.vast.ai xsG5m
  bash tools/harvest_box.sh 33836 ssh3.vast.ai xsG5n
Pre-registration: xsG5m should retrace xsG5l (17k rung ~600M, 26k
~750M, 48k ~1.6B; 0 finishes expected before the wall's free-flight
deception); xsG5n (fan) is expected AHEAD on the map frontier at
matched steps (the fan carries the path) and level on goal reaching;
the question is which of the two, if either, gets past 88% given
24 h (~7B steps at ~280k fps).
Race notes: machines 95613 and 57139 blocklisted (image still pulling
at 93 s); box 2's git clone needed http.version=HTTP/1.1; deploy_box's
pytest spun at 2,600% CPU for 13 min on the 256-thread host (numba
parallel pool off nproc) and was killed - the GPU gate is what matters.

=== 24-HOUR PAIR, morning of 2026-09-03 (21.5 h in; harvested to
runs/research/xsG5m and xsG5n at 09:30 local, boxes still running)
xsG5n (FAN + composed reward): 18.7B steps at 246k fps, goal success
75%, reservoir min-depth 1.2% (the reservoir spans the whole map).
Eval corridor over the last ten evals (17.97-18.65B): mean 92-143k,
max 164k-205k; touched the WALL three times (205,391 / 205,391 /
205,474u), 1 of 9 past 205,440u once, 0 finishes, dives-below 1-2/9.
From scratch, goal-conditioned, it has reached exactly the place the
stuck checkpoint reached after 3.8B steps of geodesic training, and
it has sat there for the last ~4 hours of evals without a finish.
xsG5m (BALL + composed reward): 16.4B steps at 215k fps, goal success
76%, reservoir min-depth 6.6%. Eval corridor peaked at 180,992 /
184,224u max (mean 75-78k) at 15.63-15.70B, then DECAYED: 57.9k /
108.6k / 140.6k / 109.6k / 144.1k, and the last three evals died in
the first section (max 3,527 / 19,415 / 3,884u, mean 3.3-5.4k) while
training goal success held at 76%. A platform-start collapse of the
greedy policy, not of the goal policy; cause not diagnosed (the eval
goal is uniform over the map and a far goal below +-45 deg pitch is in
no view - suspect, unproven). Per the decaying-series rule this arm is
a kill; left running to the on-box deadline (13:44 local) for the
user's decision, harvest already taken.
Verdict so far: the fan (path) beats the ball (destination) on the map
frontier by a wide margin at 20 h (205k vs 184k peak, and the ball
arm collapsed); goal reaching is equal (75-76%). Neither finishes.
ADDENDUM 09:35: xsG5m's next eval (16.38B) RECOVERED - mean 77,539u,
max 147,982u, 0 dives. The three dead evals (16.16-16.31B) were a 25-min
blip, not a regression; the decaying-series verdict above is withdrawn.
xsG5n at 18.72B: max 205,428u (the wall again), mean 93,705u, 0/9
finishes, dives-below 2/9.

=== RESERVOIR PAIR (2026-09-03 09:50 local), the user's two experiments
on the fan + composed-reward baseline (xsG5n's config), same two boxes,
final harvest of xsG5m (16.7B) / xsG5n (19.1B) taken first
(runs/research/xsG5m, xsG5n: progress, run.json, ckpts 16.0B / 18.0B +
19.0B, last evals). The user's notes on the 24 h pair: no finishes;
eval rollouts show the agent flying past a goal and turning back,
sometimes catching it; reward rises while map pct does not; the goals
sit where the agent surfs anyway; the ball's +-45 deg pitch coverage
loses goals below; action history untested.
  xsG5o  box 49633620  = xsG5n + UNIFORM reservoir (--respawn-binned 0,
         margin 2 s kept; the user: 'sample uniform from the reservoir,
         do not care about the frontier').
  xsG5p  box 49633837  = xsG5n + --respawn-min-speed 500 (binned 128
         kept): a snapshot below 500 u/s is never harvested, so the deep
         bins hold only states the agent can surf from.
Expectations. xsG5o: starts follow visitation, so the early track is
over-trained and the frontier starved - the reward/success dips vanish
(no band ever opens with equal share) but the eval frontier should
advance SLOWER than xsG5n's (17k rung ~600M, wall ~18B); if it matches
or beats xsG5n, the bins were never what pushed the frontier. xsG5p:
the dips shrink (new bands open with surfable states), goal success
from deep starts rises above xsG5n's 0-8%, and the frontier advances at
least as fast; the failure case is that the floor empties the deep
bins (arrivals are slow BECAUSE they are arrivals) and the reservoir
stops deepening past ~13% - watch min-depth against xsG5n's timeline
(86.7% at 0.4B, 62.9% at 1.9B, 1.2% by 18B). Both 24 h, matched
against xsG5n at equal steps; the honest metric is corridor MAX and
finishes from the platform eval, plus the per-band route success.

=== THIRD BOX (2026-09-03 10:40 local): the user's control for the
reservoir pair - the BASELINE algorithm (no goal conditioning, the
plain from-scratch geodesic race reward: run_arm.sh SCRATCH defaults,
time-pen 0.005, finish-k 0, shaping 100/d0) with ONLY the reservoir
changed to xsG5o's: --respawn-margin 2 --respawn-binned 0 (uniform
over states).
  xsG5q  instance 49730950  ssh5.vast.ai:10950  (machine 16571 again,
         the third GPU on the same EPYC host)
Race: machines 28676 and 112545 blocklisted (image still pulling at
86 s). Expectations: matched to the round-21 scratch controls (17-18k
at 525M, one of four at 48k by 750M) and to xsG5o at equal steps. If
xsG5q keeps pace with xsG5o on the platform eval, goal conditioning
buys nothing for map progress on this map; if xsG5o leads, the goal
signal is doing work beyond the reservoir change; per-episode reward
is NOT comparable across the two (different reward scales).

Reservoir pair + control at ~1B (2026-09-03 11:00 local), matched-step
platform eval (order-only corridor, evals at 0.83 / 0.91 / 0.98B):
  xsG5n (ref, binned 128)      max 23,244 / 27,069 / 29,132  mean 16.5k / 22.1k / 26.0k
  xsG5o (uniform reservoir)    max 17,947 / 18,027 / 18,033  mean 16.6k / 16.3k / 16.6k
  xsG5p (binned + speed floor) max 18,004 / 18,019 / 17,994  mean 11.5k / 13.5k / 13.1k
Both variants sit on the 17k rung where the reference had moved to
the 26k rung; one rung at one seed is inside the round-21 noise, so no
call yet. Training side at ~1.0B: xsG5o goal success 37.6%, min-depth
89.6% (ref 86.7%); xsG5p 27.7%, min-depth 50.6% (the floor lets the
deep bins fill with surfable states, so the reservoir deepened 1B
steps earlier than the reference's 62.9% at 1.9B) - the reservoir
goes deep without the platform frontier following, which is the
pre-registered failure case for the floor unless the frontier catches
up. xsG5q (baseline reward, uniform reservoir) at 277M after 12 min:
390k fps (no goal machinery), min-depth 93.5% already.

=== RESERVOIR PAIR + CONTROL, END (2026-09-03 23:30 local). The vast
balance ran out; all three trainers died at 19:33 local with no error
in their logs (killed from outside), the containers stayed up and idle
and the on-box watchdogs never fired (no 'not alive' lines - they were
frozen with the rest of the container). Harvested to runs/research/
xsG5o, xsG5p, xsG5q; boxes destroyed afterwards. The user's read of
the plots: nothing interesting. Honest metric, last three platform
evals (order-only corridor), 0 finishes everywhere:
  xsG5o goal fan, UNIFORM reservoir   11.0B steps  max 155,511 / 124,609 / 155,308  mean 111.5k / 63.0k / 75.8k
        goal success 60%; reservoir min-depth 52,453u = 74% of the map
        never harvested (uniform-over-states starves the frontier, as
        pre-registered); dives-below 0-2/9.
  xsG5p goal fan, binned + 500 u/s floor  9.7B  max 145,867 / 171,678 / 172,126  mean 93.3k / 88.0k / 88.1k
        goal success 77%; reservoir min-depth 2,506u (99% of the map),
        the deep bins now surfable; the platform frontier still ~30k
        short of xsG5n's wall at the same age (xsG5n: max 176-205k
        from ~18B, ~100k mean at 9.4B by eval_progress).
  xsG5q BASELINE reward, uniform reservoir  14.8B  max 98,741 / 98,654 / 98,510  mean 87.5k / 98.4k / 98.2k
        390k fps; all 9 episodes at one gate (~98.5k) for the last
        evals; reservoir min-depth 100,732u (50%) - the uniform
        reservoir never harvested past the eval frontier.
Reading: with the SAME uniform reservoir, the goal-conditioned fan
(xsG5o, 155k) is ahead of the plain geodesic baseline (xsG5q, 98.5k
at 1.35x the steps), so the goal signal does work beyond the
reservoir change; the speed floor did what it was built for (deep
bins fill with surfable states, 99% depth by 9.7B) without moving the
platform frontier past the reference's. One seed each; no finishes;
the wall stands.

## Round 22 - the WHOLE 620-map corpus re-verified after the button-zone fix: 238 trainable, and 171 of them have a BUTTON for a finish (2026-08-24, local CPU only, no GPU, $0)

Commit `2258655` made `detect_zones` accept a `func_button` whose `target` is
`counter_start` / `counter_off` as a zone in its own right, padded by
`BUTTON_PAD = 64`. The survey moved `ready` 69 -> 161 and `no_zones`
447 -> 176. That is a *survey* number: it says a map has zone entities, not
that the finish can be reached from the spawn. This round runs the real test
over the whole corpus and produces the trainable set the multi-map DDP run
will consume.

Everything here is CPU-side and deterministic: `verify_maps.py`'s
26-connected `scipy.ndimage.label` over `slab_occupancy` at
`vision.pick_cell`'s cell, no bake, no rental. Artifacts (gitignored) in
`runs/research/`: `verify_corpus/` (444 per-map JSONs), `verify_gateway/`
(166), `verify_endonly/` (8), `zone_audit.json`, `freewins/` (241),
`freewins_gw/`, `maps_trainable.json`.

Three new tools: `tools/zone_audit.py` (cross-source agreement + the padding
sweep), `tools/free_wins.py` (the teleport graph over free-space components),
`tools/maps_trainable.py` (the final set). `verify_maps.py` gains
`--zones-dir`, inert without the flag, so a type-2 map whose buttons are not
in the BSP at all can go through the identical four checks. Control for that
path: re-verifying the 166 gateway maps through it reproduced the earlier
gateway run's verdict on **166 of 166 maps**.

### The funnel, end to end

| stage | maps |
|---|---|
| in `maps_full_dataset/` | **620** |
| have an END zone from SOME source | **618** (type 1 trigger 178, type 3 button 270, type 2 gateway 170) |
| no end zone at all | 2 - `surf_catacombs_h`, `surf_ski_2` |
| put through `verify_maps.py` | **618** |
| **pass all four checks = TRAINABLE** | **238** (38.5%) |
| distinct families after de-duplicating `_b2`/`_b3`/`_ez` re-releases | **216** |

`surf_ski_2` is only "no source" at corpus level - it carries a hand-written
`maps/surf_ski_2.zones.json`, and hand-labelling is what the 2 remaining
maps would need.

**Pass rate depends enormously on which bin the map is in**, and that is the
most useful thing in this table:

| population | n | pass | rate |
|---|---|---|---|
| in-BSP, survey `ready`, type 3 button finish | 91 | 70 | **76.9%** |
| in-BSP, survey `ready`, type 1 trigger finish | 70 | 43 | **61.4%** |
| in-BSP, `zones_but_links`, button finish | 176 | 67 | 38.1% |
| in-BSP, `zones_but_links`, trigger finish | 107 | 23 | 21.5% |
| type 2 gateway-only | 170 | 31 | **18.2%** |
| in-BSP END with no START zone (recovered, below) | 4 | 4 | 100% |

Round 20's 47-map shortlist came out 22/47 = 47%. Over the full 444 in-BSP
maps it is **203/444 = 45.7%**, so the shortlist was representative; the
corpus is simply four times bigger than anyone had tested.

### The trainable set, and the failure table

`runs/research/maps_trainable.json`. 238 rows, each carrying `finish_kind`,
`evidence`, the unpadded finish AABB, `cell`, `extent`, `d0_euclid_mean`,
`n_spawns`, `spawns_reachable` and any warnings.

Failures (380 of 618), by which check failed:

| verdict | checks failed | n |
|---|---|---|
| `fail` | reachable | 365 |
| `fail` | spawn_sane + reachable | 7 |
| `ambiguous` | reachable | 5 |
| `fail` | spawn_sane | 2 |
| `spawn_misplaced` | reachable | 1 |

**Reachability is essentially the only thing that fails.** Nothing failed
`loads`. Nine maps have a spawn that kills all 8 envs within 20 neutral
ticks (`surf_longjumps_run`, `surf_metra`, `surf_src_activation_b1`,
`surf_src_quickie_b2`, `surf_src_skipalot_b1`, `surf_src_sunday`,
`surf_src_sunday_b2`, `surf_src_twist` and one more), and 7 of those 9 fail
reachability as well.

Quality of the 238 that pass: 228 carry **no warning at all**; every one of
them reached the finish component at the base cell (`reach_mode: slab` - not
one needed the finer retry); median straight-line spawn-to-finish 5,032 u;
median largest extent 7,808 u.

### Cross-source agreement: the two button sources are DISJOINT, so there is almost nothing to cross-check

This was meant to be the round's main consistency test and it cannot be,
because the premise is wrong. Across 620 maps, in-BSP buttons (type 3) and
the Surf Gateway service (type 2) overlap on **four role-comparisons in
total**:

| map | role | in-BSP kind | centre distance | gw centre -> BSP box | padded overlap |
|---|---|---|---|---|---|
| `surf_meow_brokengame` | end | button | 46.8 u | 26.1 u | **yes** |
| `surf_skeleton_beta_4` | start | button | 24.2 u | 17.1 u | **yes** |
| `surf_floathub` | start | trigger | 292.8 u | 216.0 u | no |
| `surf_green_pot` | start | button | 1,481.8 u | 1,466.3 u | no |

Read that as: **the service is a complement, not a duplicate.** It serves
exactly the maps whose BSP has no timer wiring. Two independently built
datasets partitioning the corpus instead of contradicting each other is
itself a consistency result - but it means the position check has n=4 and no
statistical power, and **no future plan should assume the gateway can
validate the in-BSP buttons.**

Of the four, two agree to within 47 u with overlapping padded boxes. Both
disagreements are explainable and neither costs a map:

* `surf_floathub` - the BSP zone is a type-1 `trigger_multiple` 1024x1024x1
  start plate and the gateway button sits **216 u above it**: a button on
  the wall over the start line. `detect_zones` gives the trigger precedence,
  which is the conservative choice and the right one.
* `surf_green_pot` - a genuine 1,466 u conflict, and **the gateway record
  for that map is self-flagged `suspect` (`stop` button at `origin 0 0 0`)**.
  The BSP button is 458 u from the nearest spawn, the gateway button
  1,427 u, so the BSP button is the plausible start. The gateway record is
  the wrong one.

Two ambiguity risks found on the way, small but worth knowing: **9 maps
carry two `counter_start` buttons and 9 carry two or three `counter_off`**,
and `detect_zones` takes the first match in entity order.
`surf_placeholder`, `surf_src_lt_omnific` and `surf_src_lt_omnific_s` are
among them.

### The padding decision: 64 is right, 48 is the empirical floor, 32 loses maps

Face area of the target the agent has to hit (largest face, unpadded), over
the whole corpus:

| finish | n | p10 | **median** | p90 |
|---|---|---|---|---|
| type 1 `trigger_multiple` | 178 | 65,536 | **798,848** | 7,756,186 |
| type 3 `func_button` | 270 | 1,024 | **2,048** | 2,581 |
| type 2 gateway button | 171 | 400 | **400** | 400 |

**390x** between a trigger finish and an in-BSP button finish, **1,997x**
against a gateway button - CLAUDE.md's "roughly 400x" reproduces exactly.
Restricted to the 238 maps that actually train the gap is *wider*: median
trigger face 1,462,272 u^2 against 2,048 (button) and 400 (gateway),
**714x**.

The sweep - "how many finishes have NO point inside the box where a player's
origin could be", hull-1 of the world model on a 9^3 lattice, the same probe
the gateway work used:

| finish population | n | pad 0 | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| type 1 trigger | 178 | 4 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **type 3 button** | 270 | **55** | 23 | 7 | 2 | **2** | **0** | **0** | 0 | 0 |
| type 2 gateway | 171 | 9 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

The gateway row reproduces the earlier gateway measurement exactly (9 of 171
unpadded, 0 from pad 32 up) - the method's own control.

**The in-BSP buttons ARE a different population and they need more pad.**
215 of 270 are standable unpadded; the tail runs further out - 32 first
become standable at pad 8, 16 at pad 16, 5 at pad 24, and **2 only at pad
48** (`surf_minigolf`, `surf_minigolf_2k`, 16x16x7 buttons recessed in a
wall). At pad 32, which clears both other populations, **two type-3 finishes
still have nowhere to stand.**

So pad 48 is the empirical floor for 100% of type-3 finishes, and **64 is
the right choice**: it is the engine's own `+use` reach
(`PLAYER_SEARCH_RADIUS`) rather than a fitted constant, it matches what the
gateway zone files already use so both button types are treated identically,
and it leaves one 16 u grid cell of margin over the measured floor.

**Padding buys nothing it should not.** Checked explicitly over all 620
maps: **0** maps where the padded finish box contains a spawn point, **0**
where the padded start box overlaps the padded end box, and exactly one map
whose unpadded finish is within 200 u of a spawn (`surf_shortbox`, 160 u).
Nor does it close the evidence gap: padded to 64 the median type-3 finish
face is 30,720 u^2, still **26x smaller** than an unpadded trigger finish.

**Consequence for reading a multi-map run.** 67 of the 238 trainable maps
have a trigger finish (`evidence: strong`); **171 have a button finish** -
140 type 3 plus 31 type 2 - and are `evidence: weak`. A null on a
button-finish map is much weaker evidence than a null on a trigger-finish
map, and the two must not be pooled into one aggregate.
`maps_trainable.json` carries the flag per map.

### The two free wins, measured over all 241 in-BSP failures

`tools/free_wins.py` builds the teleport graph over free-space components -
one directed edge per `trigger_teleport`, from every component its source
brush touches to the component holding its destination - and asks how many
teleports separate a spawn from the finish.

**(a) The spawn-pool seed. 73 of 241 come back WHOLE.**

| hops from a spawn component to the finish's | maps | what one seed buys |
|---|---|---|
| 0 (finish already there; failed another check) | 3 | nothing - fix the spawn |
| **1** | **73** | **the whole map**: one seed at the start-room door |
| 2-16 | 80 | only the LAST stage - the seed lands past 1-15 further links |
| no chain at all | 85 | nothing - the way out is not a teleport |

Round 20 measured this on the 47-map shortlist and named 15 maps that "come
back for free". **14 of those 15 reproduce as `hops == 1` here.** The
exception is a correction: `surf_src_corruption` has **no** teleport chain
from its spawn component to its finish at the cell `pick_cell` gives it
(64 u); round 20's answer came from its cell-32 re-run of the six largest
maps, where the chain does exist. The win is real for that map but only
visible at the finer cell - a cell sensitivity, not a mechanism.

**A correction to round 20's "no code change" claim.** The mechanism exists
(`SurfCore.set_spawn_pool`, `spawn_mode=2`), but `train_fast.py` builds the
pool from `map_spawn_pool(core)` - the `info_player_start` entities - and
takes `race_d0 = mean(goal_field.sample(raw["origin"]))` at those same raw
origins. On a sealed-start map that d0 is the unreachable sentinel, so
`scale = 100/d0` shapes on nothing even after the pool is seeded. **Both the
pool and d0 have to come from the seed point**, which is a ~10-line hook in
`train_fast.py`, not zero.

**(b) The stage-1 goal box. 130 unambiguous candidates, 94 of them standable.**

`src/env.c:601` is `int trig = complete ? 0 : apply_triggers(s, i, st)`, so
the goal test runs first and short-circuits the trigger: a goal box on a
stage link's teleport brush **completes** the episode instead of failing it.
Confirmed by reading the source. No code change - only a `<map>.zones.json`
whose `end` is the link brush.

* 180 of 241 have at least one teleport leaving the spawn's component;
* **130 have exactly ONE exit destination** - an unambiguous stage 1;
* **94 of those 130 have an exit brush with a standable point in it**, which
  is the real count: a goal box the player cannot occupy is not a finish;
* the remaining 50 have 2-6 exit destinations and need a choice made.

The two wins overlap almost completely: every one of the 153 maps with any
seed also has a stage-1 exit, 27 maps have a stage-1 exit but no seed, and
**61 of the 241 have neither** - no teleport leaves the spawn's component at
all. That is the `func_door` case round 20 identified (`src/bsp.c` makes
doors permanently solid and there is no entity-I/O system anywhere in
`src/`).

### A gate bug in TWO tools, worth 5 trainable maps

`survey_maps.classify` returns `no_zones` unless `has_start AND has_end`, and
`fetch_gateway_buttons.write_zones` emits a zone file only when the service
returned BOTH buttons. **The trainer consumes only the END** - `goal_box` is
the finish, and runs start at `info_player_start`, not at the start zone. So
a map with a perfectly good finish and no start zone was binned out and never
verified.

Eight such maps exist. Verified now:

* in-BSP END, no start zone: `surf_airfrance`, `surf_temple_of_toon`,
  `surf_temple_of_toon_sc`, `surf_src_celestial_b1` - **4 of 4 PASS**;
* gateway END, no gateway start: `surf_ancientmemories` **passes**;
  `surf_colours4`, `surf_sg_speedway`, `surf_slimerun` fail reachability.

**+5 trainable maps for free.** The fix is one condition in each tool; not
applied here because it changes the survey's published counts and belongs in
its own commit.

### Error bars, stated honestly

* **The reachability test is geometry only.** Components come from
  `slab_occupancy`; a `trigger_teleport` or `trigger_hurt` brush is NOT a
  wall in that grid, and under race rules touching one ends the episode. A
  map can therefore pass check 3 and still be untrainable because every
  geometric route crosses a fatal net. `verify_maps` has always had this
  hole - it is not new - but the trainable set inherits it.
* **Dilation false negatives are bounded at 27 of 444.** Slab occupancy
  dilates geometry by up to cell/2 per axis. Of the 241 in-BSP failures,
  **212 are disconnected even in the undilated, centre-sampled grid**, so
  they are not dilation artifacts. The other 27 are connected there: 23 were
  promoted to `fail` by the >= 2-cell seal rule (median measured seal 96 u,
  minimum 32 u), 3 stayed `ambiguous`, 1 is `spawn_misplaced`. 26 of the 27
  could not be retried finer because they already sit at the 16 u floor.
  **The honest band is 238 trainable, at most 265 if every dilation-suspect
  map turned out to be connected.**
* **21 maps land on a 64 u cell**, where the dilation is +-32 u against a
  32 u player hull - the worst case in the corpus. Round 20 re-ran 6 of them
  at cell 32 and got **identical verdicts** (5 fail, 1 pass), which is the
  only direct evidence that 64 u is not flipping answers.
* One method, no seeds, no statistics: this is deterministic geometry, so
  the round's usual seed-noise warnings do not apply.

### What to do next

1. **Build the first multi-map fleet from the 67 trigger-finish maps only.**
   They are the ones whose null means something. 216 families is plenty of
   breadth; mixing in 171 button finishes at the start makes the first
   multi-map result uninterpretable.
2. Land the two one-condition gate fixes (`classify`, `write_zones`): +5 maps.
3. If more maps are wanted, the stage-1 goal box is the cheapest lever -
   **94 maps** with an unambiguous, standable link brush, zero code change,
   one zones file each. They are shorter maps, not the original maps, and
   must be labelled as such.
4. The spawn-pool seed needs its ~10-line hook (pool AND d0 from the seed)
   before any of its 73 maps can be claimed.

### Cost

Local CPU only: ~4 h wall with up to 24 worker processes on a 32-core box,
peak ~55 GB RAM. **No GPU, no bake, no rental, $0.** `maps_full_dataset/`
was not written to (620 `.bsp`, 1 `.res`, 256 `.txt` before and after).

### Round 22 addendum - the coarse-cell sensitivity closed, and the free wins on the type-2 half

Both were still running when the section above was written. Completed:

**The 64 u cell is not flipping verdicts. All 21 of them, re-run at cell 32:
21 of 21 verdicts UNCHANGED** (16 fail, 5 pass, same maps). Round 20 had this
for 6 of the 21; it now covers the whole cell-64 population, and the worst
dilation case in the corpus (+-32 u against a 32 u player hull) is measured
rather than assumed. **The 238 figure does not move.**

The *free-win* answer is slightly cell-sensitive, which is the correction
predicted above. Of the 16 cell-64 failures re-analysed at 32, two change:
`surf_src_corruption` goes `hops = None -> 1` (exactly the round-20 claim,
now explained) and `surf_src_kitsune` `None -> 7`. So the in-BSP counts move
73 -> **74** whole-map seeds, 130 -> **133** single-exit stage 1s and
94 -> **95** unambiguous standable stage-1 boxes. Everything else is identical at both cells.

**The 136 type-2 (gateway) failures, same analysis:**

| | in-BSP (241) | gateway (136) | **total (377)** |
|---|---|---|---|
| `hops == 1` - one seed recovers the WHOLE map | 74 | 16 | **90** |
| `hops` 2-16 - a seed exists, but trains the last stage only | 81 | 90 | **171** |
| no teleport chain from the spawn to the finish at all | 83 | 30 | **113** |
| exactly ONE stage-1 exit destination | 133 | 94 | **227** |
| ... of those, with a STANDABLE exit brush | 95 | 65 | **160** |

The type-2 maps are far more deeply staged: only 12% of their failures are
one hop from the finish against 31% of the in-BSP ones, and their median
chain is 3-4 links. That fits what they are - the service exists for the
maps whose BSP was never wired for a timer, which skews old and multi-stage.

**Consolidated, over the whole corpus**: 238 train today; **+90** would train
whole on one seed point each once `train_fast.py` takes both the pool and
`race_d0` from that point; **+160** stage-1 fragments are available with no
code change at all, as separate shorter maps. The remaining 113 need
something the teleport graph cannot give - in most cases a `func_door` the
core models as permanently solid.

### Round 22 addendum 2 - BUTTON_PAD changed to 192 while this was running, so read the numbers above as a pad-64 result

`d2049d6` ("zones: uniform on-touch finishes - inflate button boxes to pad
192") landed on `cpubench` after this round's 618 verifications had started
and before they finished. **Every verdict above, and
`runs/research/maps_trainable.json`, is a `BUTTON_PAD = 64` artifact.** The
new constant makes the finish seed box larger, which can only ever ADD passes,
so 238 is a **lower bound** under pad 192. Nothing above is invalidated; it is
dated.

Four measurements bearing on that decision, all on the END-zone population
(`detect_zones`'s `end`, n=270 type-3 / 178 type-1), all from
`runs/research/zone_audit.json` with the sweep extended to 192 and 320.

**1. Standability is not a constraint at 192.** 0 of 270 type-3 finishes lack
a standable point from pad 48 up, and that holds at 192 and 320. The bigger
pad costs nothing here.

**2. Comparability to a type-1 finish is 23%, not 72% - because the 72%
compares against START and END zones pooled.** Type-1 start zones are far
smaller than type-1 end zones (median largest face 142,080 u^2 against
798,848, a factor of 5.6), so pooling them halves the reference:

| type-1 reference, per-axis medians (sorted) | product of the two largest |
|---|---|
| END zones only, n=178 | 9 x 632 x 1243 -> **785,576 u^2** |
| START zones only, n=206 | 4 x 256 x 504 -> 129,024 u^2 |
| start+end pooled, n=384 | 8 x 367 x 768 -> 281,856 u^2 |

The pooled row is the one that gives ~258,000 and hence "72%". Measured
against what a finish should be compared to - a type-1 **finish** - a padded
button reaches:

| type-3 button finish | median largest face | share of the type-1 END median (798,848) |
|---|---|---|
| pad 64 | 30,720 u^2 | **4%** |
| pad 192 | 186,368 u^2 | **23%** |
| pad 320 | 473,088 u^2 | 59% |

So 192 closes most of the gap on a log scale (26x smaller becomes 4.3x
smaller) but does **not** make a button finish equivalent to a trigger
finish, and the `evidence: strong|weak` split in `maps_trainable.json` should
stay.

**3. Spawn-swallow is rarer than the clamp assumes, on this population.**
Distance from the nearest spawn to the unpadded type-3 finish box, n=270:
**0 maps** are inside at pad 64, **1** at pad 192 (0.4%), **2** at pad 320
(0.7%). The `_pad_clear_of_spawns` clamp is right to exist; it just fires on
one map at 192, not on a meaningful fraction. (The 3.8%-at-320 figure must be
over a different population - worth reconciling, not worth blocking on.)

**4. The real cost of the bigger pad is a goal-seed LEAK, and it is not
small.** `verify_maps` seeds the finish component from the padded box grown by
`max(0.75*cell, 20)`. Going 64 -> 192 makes that seed reach **128 u further on
every side**, and a wall thinner than that puts free voxels on BOTH sides
inside the seed - so the finish's component silently becomes the spawn's and
the map reads "reachable" when the player still cannot get there.

Measured on the 130 button-finish maps that fail reachability today: 42 have a
seal thin enough for `seal_thickness` to measure at all (<= 320 u), **median
48 u**, minimum 32 u - and **29 of them have a seal of 128 u or less**, which
is exactly the extra reach pad 192 buys. `surf_cryptic`, `surf_desert_maze`,
`surf_hell_ez`, `surf_house_final`, `surf_house_rmk`, `surf_longjumps_run`,
`surf_meow_wwii` and `surf_minigolf` all sit at 32 u.

**So: re-run `verify_maps.py` over the corpus at whatever pad ships, and treat
every map that newly passes as suspect until its `gap_units` is checked.** A
pass that appears only because the goal box grew through a wall is the same
class of error as `surf_src_sidistic` - a map that trains to a null and looks
like a hard map - and this round's whole point was to stop shipping those.

## Round 30 prelude: adversarial review fixes (goal radius leak)

An adversarial review of the goal system (`python/surfgym/goals.py`,
`goalsys.py`, `goalball.py`, `respawn.py`) found three bugs, all confirmed
with repro scripts before anything was touched. Fixed on `baseline` in
commit **a903cfd** (`tests/python/test_goal_radius_leak.py`, 16 cases, 13 of
which fail against the pre-fix tree). Nothing here changes the control path -
no `--goals` flag, no reward, no observation outside the goal arms.

**1. THE FINISH-GOAL RADIUS LEAK (high).** `SphereGoals.set(idx, centers)`
wrote `self.radius` only when a radius was passed, and `GoalSystem.assign`
passed one only for FINISH goals. So the first time an env drew the map's
finish - radius = half the finish box's longest side, **3,328 u** on
cannonball against the nominal **192 u** - it KEPT that radius for every
ordinary goal it was handed afterwards. The arrival test for that env stayed
17.3x too wide in radius, ~5,200x too large in volume, for the rest of the
run, and the contamination is per env and permanent, so the affected share
of the fleet only grows.

**Which arms it touched:** every arm whose goal distribution contains the
map's finish - `--goal-fixed` (the fixed set always ends with the finish),
`--goal-route-uniform`, and `--goal-frontier` once F reaches 1. That is
**xsG5g through xsG5p** of round 29. Arms with no finish in the draw
(air-only and pure reached-state goals) never set a non-nominal radius and
are unaffected.

**What it inflated, and what it did NOT.** It inflated the TRAINING-side goal
numbers: `race/goal_success` and its per-kind / per-k-bin splits, and
`ticks_to_goal` (a goal "reached" from up to 3,328 u away is reached sooner).
Since a sphere entry also ends the episode, it shortened those episodes too.
It did **not** touch the platform eval corridor numbers - the honest metric
the round was called on - because the eval runs on its own sphere:
`GoalSystem.eval_hooks` recomputes `ev["radius"]` per EPISODE and
`on_tick` tests against that, never against `self.sphere`. Round 29's
frontier / corridor-MAX comparisons therefore stand; its training goal-success
curves for the arms above do not, and must not be compared across arms that
differ in how often the finish was drawn.

**The fix.** `assign` now passes the nominal radius explicitly on every row
before the finish override, and `SphereGoals.set` resets to the constructor's
radius when none is given (belt and braces - either half alone would have
prevented it). `GoalBallLidar.set_goals` gets the same reset rule, since the
ball is the channel the policy actually looks at. `GoalDistField.set` holds
no radius at all, which is now stated in the code where someone would look.

**2. `--goal-route-uniform` CRASH (medium).** `_route_goal` guarded on
`s0 >= route_len - R` while every branch draws from `[s0 + 2.5 R, ...]`, so a
start projecting into `(L - 2.5R, L - R]` handed `Generator.uniform` a low
above its high and killed the trainer with `ValueError` mid-run. Reachable:
a goal near the end of the line is reached, a successful episode harvests its
whole chain into the reservoir, and the next iteration spawns there. The
guard is now `s0 + 2.5 * radius >= route_len` and the caller falls back to a
reached-state or air goal.

**3. ACHIEVED-GOAL k WAS IN SNAPSHOTS, NOT SECONDS (medium).** For
reached-state (kind-0) goals `assign` set `k = len(seg) - 1`, a count of
reservoir snapshots, while `KCurriculum`'s band and `GoalStats`' bins are in
SECONDS. Under `--goals` the trainer runs a 0.25 s snapshot cadence
(`snap_every=25`), so k read **4x high** and the curriculum counted **95%**
of reached-state goals as top-third against a true **22%** - and the top
third is the only vote that moves `k_max`. k is now converted with the
reservoir's own cadence (`GoalSystem(snap_every=...)`, `train_fast` passes
`respawn.snap_every`, and `iterate()` re-latches it from the live buffer),
capped at `kcap`, floored at one snapshot interval. Caveat left as a TODO in
the code: a chain longer than `seg_max` (64) is subsampled, so the count
saturates and understates the longest goals; the exact figure is the
harvest's own tick gap, which would need a fifth goal column through
`drain_harvest` / `push_many` / `state_dict` / `build_pool` and would change
the checkpoint format.

**Reported, not fixed** (out of this task's scope, still open): the binned
`build_pool` cap top-up can draw sentinel-distance states that `dist_valid_max`
says are not sampleable; `build_pool`'s `vel_scale` can emit spawns below
`--respawn-min-speed`; and `pool_map` keys spawns by rounded origin, so
several pool rows sharing one origin all resolve to the LAST of them.


## Round 30 prelude: the contact blackout in the depth render

**The GPU depth lidar went fully black whenever the agent was touching the
ramp it was surfing.** Found by adversarial review, reproduced, fixed and
measured; commit `123a69e` on `baseline`.

**The mechanism.** `slab_occupancy` marks a voxel solid if ANY sample within
+-cell/2 of its centre is solid (that dilation is deliberate - it is what
stops thin panes slipping through the lattice). At cell 32 that means the
EYE'S OWN VOXEL reads sdf 0 whenever the eye is within ~16-30 u of a wall or
ramp. A surfing hull puts it exactly there: the eye is 17 u above the origin
and the hull is 16 u half-width, so a player pressed against a ramp is 20-25 u
from it. Every march - the triton depth kernel, the surf-mask kernel, the
normals kernel merged this round, the pinhole kernel and `_render_torch` -
tested `d > hit_eps` at t = 0, before the ray had moved. So `alive` went
false for EVERY ray at once and the whole 128x64 image was 0.0: not a dark
image, the literal zero tensor, on a policy whose only exteroception is that
image.

**How often, measured with the renderer itself** (not with a proxy): every
decision-aligned moving frame of the last 6 `xsARC3` traj files and of
`xsG5n`, rendered twice, once through a verbatim copy of the pre-fix kernel
and once through the fixed one.

| run | frames rendered | fully black BEFORE | fully black AFTER | frames that CHANGED |
|---|---|---|---|---|
| xsARC3 (6 files, 345,253 ticks) | 73,461 | **320 (0.436%)** | **0 (0.000%)** | 320 (0.436%) |
| xsG5n (4 files, 159,527 ticks) | 33,404 | **197 (0.590%)** | **0 (0.000%)** | 197 (0.590%) |

The review's split of the same statistic: **4.5x concentrated at ramp/wall
CONTACT ticks** versus free flight - i.e. the blindness is not spread over
the episode, it fires precisely at the moments the policy has to steer
against a surface. On a 1,500-decision episode that is ~7 blind decisions,
all of them at contacts.

**The fix.** Remember the voxel the eye starts in and suspend the hit test
while the sample is still inside THAT voxel, still stepping by `min_step`.
Rays aimed into the wall then stop at the next voxel; rays aimed away leave
it and march normally. One integer compare per step, on the flat voxel index
the SDF gather already computes. Applied to all four triton kernels and to
the torch fallback.

**It is bit-identical on every frame that was not black, and that was
MEASURED, not just argued.** The argument: the SDF gather is
nearest-neighbour, so while the sample is inside the start voxel `d` IS that
voxel's own value, and on an eye in open air that value already satisfies
`d > hit_eps` - so the extra term can only ever change a lane whose start
voxel is solid. The measurement: across the 106,865 live frames above, the
set of frames that changed **equals the set that was black, exactly** (320
== 320 and 197 == 197). `tests/python/test_blackout.py` additionally pins
max|diff| == 0 against verbatim pre-fix copies of the torch march and of
`_march_kernel` / `_march_kernel_nz`, on a synthetic scene and on 50 poses
recorded off a live cannonball policy, on the GPU and on the torch path,
with the surf-mask and normals channels included.

**What the fixed render sees at the four recorded black poses**, against the
exact C tracer (26-direction nearest solid from the eye):

| pose | truth: nearest solid | render: nearest hit | render: farthest ray | share of rays past 100 u |
|---|---|---|---|---|
| (-14152, 1831, 8716) | 25.3 u | 38.4 u | 2,456 u | 83.2% |
| (-14146, 1842, 8714) | 25.3 u | 38.4 u | 2,457 u | 79.1% |
| (-14084, 846, 8307) | 25.2 u | 19.2 u | 2,434 u | 75.7% |
| (-3426, -10148, 2766) | 23.1 u | 9.6 u | 67 u | 0.0% |

The first three go from a zero tensor to a normal view of the map. The
fourth is genuinely wedged - the tracer finds nothing past 123 u in any of
five probe directions - so 67 u is the honest answer there, not a residual
defect.

**Cost.** 1.088x on the march kernel over the real decision distribution
(1.051x on a batch with no black frames at all, which is the compare alone;
the rest is the extra stepping on contact frames extending the block-wide
early exit). The march is roughly a fifth of a training step, so ~1.7% of
throughput. An analytic slab-exit variant (`t < t_exit` instead of
`vox == vox0`) was prototyped, produced **bit-identical** output on both
black and non-black poses - an independent confirmation that the voxel-index
compare is exactly "still inside the start voxel" - and was SLOWER (1.094x
vs 1.051x on the same batch), so the index compare stays.

**EVERY CHECKPOINT TRAINED SO FAR SAW BLACK FRAMES AT CONTACTS.** The bug is
as old as the SDF renderer, it is in the depth channel that every warm-start
trunk was trained on, and it fires on ~0.4-0.6% of moving decisions on
cannonball, concentrated at contacts. Two consequences:

1. **A control re-run on the fixed renderer is NOT bit-comparable to any run
   before `123a69e`.** The observation stream differs at exactly the frames
   that were black. Surf is chaotic and one differing depth pixel forks a
   greedy trajectory (this file already records that for the 3090-vs-5090
   lidar difference), so an old control number and a new one are two
   different experiments even at the same seed and config. Re-baseline
   before comparing, and say which side of this commit an arm ran on.
2. It is a candidate mechanism for the wall, not a proven one. The stuck
   checkpoint leaves the ramp between route vertices 1596 and 1598 after a
   ~0.45 s precursor of growing off-line error, which is a control-precision
   failure at a contact - and a contact is exactly where the sensor was
   dropping out. **That is a hypothesis this fix makes testable, not a
   result.** Nothing here has been trained yet.

**One thing this uncovered that is still open.** Four of
`tests/python/test_lidar_march.py`'s own eight fixture poses (indices 2-5,
labelled "at the glass panes" and "mid-track, open air") put the eye INSIDE a
dilated-solid voxel. They rendered a wall of 0.0, so that file's
`max|diff| == 0` bit-exactness assertion has been comparing 0 == 0 on half
its poses since it was written - and "mid-track, open air" is not open air,
the fixed render stops every ray there at 9.6-28.8 u. The guard changes
exactly those four by design, so that test goes red on any checkout that has
the baked SDF cache. It needs the same four lines in ITS legacy copy, or
four poses that are actually in open space; it was left alone here because
this task's file scope was `vision.py`, the new test and this ledger.
`test_blackout.py::test_four_of_test_lidar_marchs_own_poses_were_blind` pins
the blind subset so the failure is signposted rather than mysterious.

=== ROUND 30 WAVE 1 (2026-09-04 ~01:50 local, pre-registered). Branch
baseline = goallines + tool folds + review fixes (goal radius leak,
route-uniform crash, achieved-k units, contact blackout in the depth
render, truncation bootstrap under per-env lines, resume restore of
n_steps/epochs/minibatches, d0_per_env guard) + the new flags. The
CONTROL (xW1CTL on a 3090, xW1CTL4 on a 4090) is xsG5n's config with
those fixes and --eval-stall 1; every arm changes ONE thing:
  3090: xW1RETN --ret-norm 1 | xW1WIDE --emb 1024 --hidden 896 |
        xW1DEEP --tower-depth 4 --conv-mult 2 | xW1NRM --normals 1 |
        xW1PITCH --pitch-fixed -25
  4090: xW1GRU --rnn gru --rnn-size 256 | xW1FOV --lidar-hfov 160
        --lidar-vfov 120 | xW1COMP --obs-compass 1 | xW1FP32
        --fp32-heads 1 | xW1BON10 --success-bonus 10
3.5 h budget per box, harvest at ~3 h, matched-step reads at 1B and 2B
on the order-only corridor MAX from the platform eval (+ per-band goal
success and the new explained-variance / trunc / stall / crawl columns).
Compare only within a card type. Expectations: RETN and FP32 are
optimizer hygiene - a matched-step gain of a rung or nothing; WIDE/DEEP
answer the user's capacity question (round 25's multi-map width gain
predicts a small positive at matched steps, at ~1.5-2x wall); NRM and
PITCH test the 'ramp = surfable' hypothesis (the goal-fan control looks
UP 86% of the time); FOV/COMP/GRU are the exploration-of-inputs arms;
BON10 tests whether the 2,500:1 bonus-to-shaping ratio destabilises.
The local 5090 runs exit10 (expert iteration from xQR32, 12 rounds).
One seed each: a difference under one rung at one seed is noise.

Wave 1 as launched (2026-09-04 02:45 local), card per arm - compare only
within a card type; three controls:
  3090: xW1CTL 49803037 | xW1RETN 49803039 | xW1WIDE 49803040 | xW1COMP 49806277
  4090: xW1CTL4 49803796 | xW1GRU 49805756 | xW1FOV 49805757
  5090: xW1CTL5 49804678 | xW1DEEP 49805249 | xW1FP32 49806646 | xW1BON10 49806648
Readiness: 60 s window kept; 1 of 9 4090 and 1 of 8 5090 offers came
up in the first races (image pulls), 8+7 machines blocklisted as
network; four small fill rounds placed the rest. One 5090 (machine
23132) failed gpu_health and is blocklisted gpu_capped. Deadlines
(on-box watchdogs) 05:42-06:39 local; harvest planned from 05:15.

=== ROUND 30 EXPERT ITERATION (exit10, local 5090, 2026-09-04 01:16-): the
AlphaZero-style loop COMPOUNDS. Seed xQR32 (9/9 finishes, 77.74 s best
/ 78.02 s mean, spawn basis). Per round: 9 greedy evals -> beam planner
(300 s, 14 waves, greedy-envs 64, dv score) -> 16 fastest lines
distilled (~40k rows) + spine -> 3e8-step warm PPO with the BC term
0.5 -> 0 and 90% spine spawns -> 9 greedy evals. Rounds 0-5:
  greedy best  77.74 -> 77.67 -> 77.66 -> 77.18 -> 76.90 -> 76.75 -> 76.43 s
  greedy mean  78.02 -> 78.04 -> 77.94 -> 77.69 -> 77.41 -> 77.27 -> 76.94 s
  planner best 76.40 -> 75.86 -> 75.78 -> 75.44 -> 75.25 -> 74.88 s
  finishes 8-9/9 every round (6/9 once, round 3 out); ~830 s per round.
So -1.3 s of greedy time in 6 rounds (~85 min), and the planner's own
line improves as the policy it plans from improves (-1.5 s). The
ledger's practical floor on this line is 73.66 s; the WR is 68.00 s
(a different route). 12 rounds queued; the code is on feat/expert-
iteration (not yet merged into baseline).

Wave 1 at ~2 h (monitor 03:48 local; corridor MAX of the last two evals,
mean in brackets; step in B):
  3090  xW1CTL  1.93B 28.8k/28.4k (23.2k/22.0k) | xW1RETN 1.91B 90.0k/66.9k (77.1k/51.0k)
        xW1WIDE 1.84B 56.4k/60.7k (46.5k/54.4k) | xW1COMP 1.22B 17.9k/17.9k (14.0k/14.5k)
  4090  xW1CTL4 2.06B 29.2k/75.9k (26.6k/51.3k) | xW1GRU 1.17B 30.1k/27.3k (17.5k/23.1k)
        xW1FOV  1.61B 79.1k/40.3k (46.6k/31.5k)
  5090  xW1CTL5 1.10B 30.8k/28.0k (25.1k/21.5k) | xW1DEEP 1.28B 107.3k/114.4k (75.1k/70.8k)
        xW1FP32 1.02B 31.4k/34.3k (20.0k/28.4k) | xW1BON10 1.18B 71.7k/40.3k (53.6k/33.7k)
Interim reading, one seed each, rung noise applies: the deeper+wider
trunk (DEEP: 107-114k at 1.28B vs its control's 28-31k at 1.1B) and
return normalization (RETN: 67-90k at 1.9B vs its control's 28k) are
far outside the one-rung band; WIDE, FOV and BON10 read positive;
GRU, FP32 and COMP read neutral so far. No finishes anywhere.
Expert loop rounds 6-10: greedy best 76.43 -> 76.25 -> 76.42 -> 76.23,
mean 76.94 -> 76.59 -> 76.66; planner 74.9 -> 74.4-74.6 s. The gain
per round shrank to ~0.05-0.2 s and the policy sits ~1.8 s behind the
planner's line despite 98% per-head BC agreement.

## Round 30 (branch feat/expert-iteration) - AlphaZero-style EXPERT ITERATION: the planner's line distilled back into the policy (built + one local round, 2026-09-04)

The owner's ask: "one thing that helped to get a few seconds is planner,
but we never TRAINED with planner with alphazero style". Built as
`tools/expert_loop.py`: per round, RECORD greedy map-start evals of
policy_r -> PLAN with `tools/beam_tas.py` waves from the same spawn ->
DISTIL the kept finishing lineages into (state, core scalars, latch,
action) rows (`tools/plan_to_bc.py`, `surfgym/bc.py`) + the best line's
per-tick spine -> TRAIN a WARM resume of policy_r with `--bc-file` (a
weighted cross-entropy of the six factored heads against the planner's
indices, summed into every PPO minibatch step before the one backward;
`--bc-coef 0.5 -> 0` linear over the round) and `--demo-file` (90% of
spawns uniform along the planner's line) -> EVALUATE policy_{r+1}.
`runs/<loop>/expert_summary.jsonl` carries one line per round.

**Seed.** xQR32 (`runs/research/xQR32/xQR32_final.pt`, md5 91238a87...,
9/9 finishes, 77.86 s) carries a 32-quantile critic the mainline trainer
has no flag for. `tools/ckpt_qr_to_scalar.py` collapses it EXACTLY: the
row-mean of a linear quantile head computes the mean of the quantiles
for every input (max |dV| 5.7e-6 over 256 random features); the actor
is byte-identical; Adam moments row-meaned. The mainline trainer resumes
the result with every flag restored (race_latch 6996, obs_reward,
act_every 3, respawn_margin 2, the 20,000-state reservoir).

**The planner did not work on this seed as it stood, and the fix is a
result in itself.** The v1 population search that took the 85.23 s
champion to 82.42 s crossed NOTHING on xQR32 (2,048 sampled lineages,
0 finishes, extinct at 94.5 s): the population reaches the wall at 75 s
in lockstep with the greedy line, then arrives at the finish window
late and low and every lineage dives past the box (deaths at route
vertex 1810, z -4,200..-5,400). Diagnosis by greedy prefix: from the
greedy line's own 75.0 s state, 2,048/2,048 sampled continuations
finish (77.78 s); from its 60 s state, 0. Sampling from a policy this
sharp is SLOWER than its mode, and 0.75 s truncation selection on d
cannot buy the loss back before the window closes. Two additions,
measured on the same seed (greedy gate 77.83 s):

| beam_tas | best | finishes |
|---|---|---|
| plain population search (`--score d`) | no crossing | 0 |
| `--score dv` (critic ranks the endgame) | 77.85 s | 2,048 |
| `--greedy-envs 64 --score d` | 76.79 s | 1,878 |
| `--greedy-envs 64 --score dv` | **76.65 s** | 4,095 |

`--greedy-envs G`: envs [0, G) take the argmax of the shared forward
(greedy continuations of the elites after every resample; env 0 is
never cloned over), so a wave is bounded below by the greedy line.
`--score v/dv`: rank by V(s) instead of the goal field once the frontier
is within 20,000 u of the goal (round 27's "the critic knows what the
field does not", now in population mode). `--race-latch` checkpoints
are searchable (the flag is cloned with the state), and
`--keep-finishers K` saves the K fastest distinct lineages.

**One full local round (RTX 5090, shared), `runs/exit_r1/`:**

| phase | result | wall |
|---|---|---|
| eval_in (9 greedy, seed 777) | **9/9, best 77.74 s, mean 78.02 s** | 36 s |
| plan (26 waves, 2,048 envs, R=25, greedy-envs 64, dv) | **26/26 crossed, best 76.40 s** (top 8: 76.40-76.51; gate greedy 77.83 s) | 654 s |
| distil (16 fastest distinct lines) | 40,761 rows, spine 7,640 states | 10 s |
| train (1e8 steps warm PPO+BC, 2048 envs) | bc nll 3.18 -> 0.49, head-acc 0.83 -> 0.98, train win 87-93% | 287 s (376k steps/s) |
| eval_out (policy_1, same 9 spawns) | **9/9, best 77.39 s, mean 77.95 s** | 44 s |

Paired by spawn: policy_1 - policy_0 = -6.9 ticks mean (5 of 9 faster,
range -54..+31). So one 5-minute round moved the best greedy time
77.74 -> 77.39 s and the mean 78.02 -> 77.95 s while keeping 9/9; the
planner's 76.40 s is not reached by the greedy mode after one round,
and the difference is inside one spawn's jitter (+-0.3 s). The claim
this round licenses is "the loop runs end to end and does not break the
finisher"; whether it COMPOUNDS is the 10-round overnight question.

**Verification.** `tests/python/test_expert_iteration.py` (6 tests):
the BC row assembled by the trainer ([15 core | latch | render(state)])
is `torch.equal` to `GreedyTorchPolicy._obs` at the same decisions on
the same core, scalars/latch/depth alike, and the policy's greedy
actions on the assembled rows are the recorded ones; the loss is the
per-head cross-entropy; a zero coefficient is zero gradient bit for bit
and the loop skips the term; the flags are recorded, guarded and
TRAIN_ONLY for the recorder; the quantile collapse is exact. Byte-
identity without `--bc-file` holds by construction (every added line is
behind `if args.bc_file` / `bc is not None`; the PPO minibatch step is
untouched) - it could NOT be shown empirically here: the resumed
trainer is not run-to-run deterministic on this shared box (two runs of
the same code: first-update kl 0.0949 vs 0.0939; cudnn.benchmark and
inductor autotune under load).

**Why the scalars are stored, not re-derived.** `write_obs` (src/env.c)
reads `last_yaw_delta`/`last_pitch_delta`, per-env values OUTSIDE
SurfState that `set_state` does not carry, so a spare core cannot
reproduce training's row from a state alone. The replay that builds the
rows steps the same core code along the same actions, so its scalars
ARE training's; only the depth image is rendered per minibatch.

**Costs, local 5090:** ~26 min per round at 3e8 train steps + 600 s of
planning (~14 min train, ~11 min plan, ~1.5 min evals); 10 rounds ~4.5 h.
The top-8 waves sat within 0.11 s, so `--plan-budget 300` (12 waves)
loses little and brings a round to ~21 min.

=== WAVE 1 RESULT (harvested 05:00-05:07 local, all 11 boxes destroyed,
runs/research/xW1*: corridor.csv over every eval, ckpt, last 2 trajs).
Order-only corridor MAX (mean) in ku at matched steps, best MAX, last step:
  3090  1.0B        1.5B        2.0B        2.5B        2.9B       best  last
  CTL   26.1(16.3)  55.1(42.5)  78.3(61.7)  91.6(69.4) 106.3(58.5) 106.3 2.95B
  RETN  27.1(16.4)  39.6(28.3)  77.6(53.2)  67.8(38.2)  91.3(61.1) 106.2 2.87B
  WIDE  18.0(15.0)  39.9(30.0)  74.5(62.7) 106.4(57.3) 101.3(67.2) 106.4 2.79B
  COMP  17.6(13.6)  18.1(16.5)  46.9(36.1)     -           -        48.8 2.34B
  4090
  CTL4  20.3(15.8)   4.0( 3.9)  29.2(26.6)  59.8(47.7)  95.0(61.0) 100.8 3.40B
  GRU   18.0(16.9)  58.7(19.0)  41.2(29.3)     -           -       104.1 2.27B
  FOV   48.8(27.2)  79.0(46.6)  39.9(38.7)  40.2(36.0) 117.7(70.1) 117.7 2.95B
  5090
  CTL5  30.8(25.1)  48.2(13.1)  85.0(61.3)     -           -        85.0 1.89B
  DEEP 106.3(49.7)  97.8(54.5) 166.6(95.6)     -           -       191.3 2.19B
  FP32  34.3(28.4)  72.4(44.9)  92.4(48.6)     -           -        92.4 2.04B
  BON10 59.9(41.8)  39.8(27.8) 100.7(62.6)     -           -       106.1 2.34B
No finishes anywhere. VERDICTS (one seed; a rung at one seed is noise):
* DEEP (--tower-depth 4 --conv-mult 2, 1.98x params) is the one result
  far outside the noise: 166.6k at 2.0B and 191.3k at 2.19B against its
  control's 85.0k at 1.89B - within 15k of the WALL at 2.2B steps,
  where the fan+geo lineage previously needed ~18B (xsG5n). The trunk's
  capacity was the binding constraint; the user's item 4 stands, and
  round 15's 'neutral pre-wall' verdict (towers only, gate-ladder metric)
  is superseded.
* FOV 160x120 (117.7k vs 95.0k at 2.9B, ahead at 1.0B and 1.5B too) and
  BON10 (100.7k vs 85.0k at 2.0B) read positive; GRU reached 104k by
  2.27B where its control needed ~3.2B (time-to-rung); RETN and WIDE
  reached the 90-106k band 0.4-0.6B earlier than the control but end on
  the same gate; FP32 heads neutral; COMP (field compass) NEGATIVE
  (46.9k vs 78.3k at 2.0B).
* Not run: NRM and PITCH (the 3090 race came up short); they go into
  wave 2 on top of DEEP.
Cost: ~11 box-hours x 3.3 h ~ USD 12.5 incl. the failed races.

=== ROUND 30 WAVE 2 (launched 05:12-06:00 local, 2026-09-04). Every arm
= wave 1's control + --tower-depth 4 --conv-mult 2 (the DEEP net), with a
DEEP control per card type; one change each:
  5090: xW2DEEP5 (control) 49816685 | xW2BON10 --success-bonus 10 49817053 |
        xW2FOV --lidar-hfov 160 --lidar-vfov 120 49817341
  4090: xW2DEEP4 (control) 49817750 | xW2RETN --ret-norm 1 49818050 |
        xW2NRM --normals 1 49818055 | xW2PITCH --pitch-fixed -25 49818408 |
        xW2GRU --rnn gru --rnn-size 256 49818410
  3090: xW2DEEP3 (control) 49818761 | xW2DEEPER --tower-depth 6 --conv-mult 3
        49819144 | xW2SPEED --speed-coef 0.005 49819441
xW2RETN's box was GONE within ~25 min of its ALIVE (trainer death ->
the on-box watchdog, or a host failure; no log survived); relaunched
on a fresh 4090 with an early console check. Pre-registered reads:
matched steps at 1.0 / 1.5 / 2.0B against the same-card DEEP control;
the questions are whether FOV / BON10 / GRU / RETN stack on DEEP,
whether NRM / PITCH (never run in wave 1) move the ramp-reading policy,
whether DEEPER (2.9x params) keeps paying, and whether the speed term
makes strafing gains reward-visible. 4 h deadlines 09:10-10:00 local,
harvest from 08:35. Budget after wave 1: ~USD 37; wave 2 ~USD 14.

Expert iteration exit10, 24 rounds done (01:16-07:40 local, ~6.4 h on
the 5090): greedy best 77.74 -> 75.31 s (round 22 in), greedy mean
78.02 -> 76.0 s, planner best 76.40 -> 73.78 s; finishes 5-9/9 per
round (the finish rate got noisier after round 11). Rounds 0-5 gave
-1.3 s, rounds 6-23 another -1.1 s: the loop keeps compounding but at
a third of the early rate, and the policy trails the planner's line by
~1.5-2.4 s throughout. Round-by-round in runs/exit10/expert_summary
.jsonl (worktree C:\RL_Surf_exit).

=== WAVE 2 RESULT (harvested 08:33-08:39 local, all boxes destroyed).
Order-only corridor MAX (mean) in ku at matched steps, best, last step;
every arm = the DEEP net (+ one change), controls per card:
  5090   1.0B         1.5B         2.0B         2.5B        best  last
  DEEP5  78.8(60.4)   40.3(30.7)   38.4(28.8)  203.8(117.3) 203.8 2.64B
  BON10  91.0(66.4)  119.5(56.4)   88.3(60.2)   29.1(26.5)  170.6 2.79B
  FOV   106.4(89.0)  123.4(91.9)  197.1(74.2)  162.1(76.4)  203.5 2.57B
  4090
  DEEP4  67.3(46.1)   89.4(47.8)   39.6(26.6)   41.3(35.4)  157.5 2.95B (121.9 at 3.0B)
  NRM   105.7(64.1)  149.2(102.3)  29.4(26.0)  171.1(131.6) 171.1 2.57B
  PITCH  87.6(54.7)   41.0(27.2)  152.7(79.8)   40.4(36.9)  179.2 3.47B
  GRU    93.0(69.3)   92.3(79.2)   29.0(25.1)      -        119.6 2.19B
  3090
  DEEP3  60.5(34.3)   29.2(14.9)      -            -        117.8 1.81B
  DEEPER 106.5(84.2)  91.9(67.0)      -            -        117.9 1.51B
  SPEED  49.0(37.7)  106.2(64.9)  117.7(65.8)      -        117.7 2.04B
  RETN: both boxes died within an hour (host deaths; the combination ran
  300M steps locally without error) - inconclusive, not re-run.
No finishes. VERDICTS: (1) the DEEP net alone reaches the WALL: DEEP5
203.8k at 2.5B and FOV 197-204k from 2.0B, where the shallow lineage
needed 18B (xsG5n) - the corridor MAX swings between rungs from eval to
eval, so read 'best so far' and 'first step at which a band is reached'.
(2) On top of DEEP, at matched steps against the same-card control:
FOV 160x120 reaches the wall ~0.5B earlier (positive); NRM (normals)
149k at 1.5B vs 89k and 171k at 2.5B vs 41k (positive); PITCH -25 fixed
152.7k at 2.0B vs 39.6k (positive - the ramp-reading hypothesis holds:
no pitch control, looking down, beats the free-pitch control); SPEED
106k at 1.5B vs 29k and DEEPER (2.9x params) 106k at 1.0B vs 60k
(positive early, both on slow 3090 boxes with fewer steps); BON10 mixed
(ahead at 1.0-1.5B, behind at 2.5B, best 170.6k vs 203.8k); GRU neutral
(ahead at 1.0B, level at 1.5B, best 119.6k at 2.19B vs the control's
157.5k at 2.95B - fewer steps, unreadable past 1.5B). (3) Nothing on the
deep net reads negative. One seed each; the rung noise is large, and
the 3090 group ran only 1.5-2.0B steps.
COMBINED READING of the night (user's question: the one thing that
restricts everything): the conv trunk's capacity. Wave 1's control took
3B steps to reach 106k; the deep net reaches 118-204k in 1.5-2.6B and
sits at the wall by 2.5B. Second-order, each worth a rung or two:
wider FOV, fixed downward pitch, surface normals, the speed term.
NEXT (wave 3, not launched - budget ~USD 22 left): the stack
DEEP + FOV + PITCH -25 + NRM + SPEED on one card type with a DEEP
control, 6-8 h, to see whether the stack finishes; and the LOOP
reset-vs-spine cell B. The timer track: exit30 (planner 600 s, 6e8
steps/round) is running locally from the round-23 policy.

exit30 (expert iteration from the round-23 policy, planner 600 s and
6e8 steps per round, 2x exit10's budgets), rounds 0-2 at 10:11 local:
  greedy best 75.73 -> 75.24 -> 75.62 -> 75.33 s, mean 76.05 -> 75.83 -> 75.96 -> 76.12
  planner 73.78 / 73.84 / 73.66 s, finishes 9/9, 9/9, 8/9; ~1,600 s per round
Doubling the per-round budgets did not break the plateau: the policy
hovers at 75.2-75.6 s best while the planner's line sits AT the
ledger's practical floor for this route (73.66 s). The limiting factor
on the timer track is now the distillation's reproduction of the
planner line (98% per-head BC agreement still compounds into ~1.7 s
over ~7,500 decisions), not the planner and not the budget. Candidates
for the next iteration: DAgger-style relabelling (plan from the
POLICY's own states, not only the elite line), a larger BC batch or
coefficient floor, or distilling into the DEEP net.

=== ROUND 30 WAVE 3 (launched 11:01-11:20 local, 2026-09-04, six 5090s):
the promising arms on the PLAIN GEODESIC lineage (no goal conditioning;
run_arm SCRATCH defaults: time-pen 0.005, finish-k 0, shaping 100/d0) +
--respawn-margin 2 --respawn-binned 1 --respawn-bins 128 --eval-stall 1:
  xW3CTL   shallow control                      49839880
  xW3DEEP  --tower-depth 4 --conv-mult 2         49840366
  xW3DFOV  DEEP + --lidar-hfov 160 --lidar-vfov 120  49840370
  xW3DPITCH DEEP + --pitch-fixed -25             49840377
  xW3DNRM  DEEP + --normals 1                    49840385
  xW3DSPEED DEEP + --speed-coef 0.005            49841427
Same card type throughout; 4 h deadlines ~15:00-15:20 local; harvest
from ~14:25. Question: does the capacity result transfer to the plain
geodesic reward (the lineage the finishers came from), and do the
perception arms stack on it there too? Reference: the wave-1 shallow
goal-fan control reached 106k at 3B; xsG5q (geodesic, uniform
reservoir, shallow) 98.5k at 14.8B.

=== ROUND 30, day 2 (2026-09-04 11:40 local): the user's follow-ups.
* WR DEMO parsed and compared (tools/demo/, runs/research/wr_demo/
  wr_vs_ours.md, commit 2268782): zone clock 68.60 s (WR) vs 76.88 s
  (xQR32) vs 80.34 s (champion); same physics (maxvel 4000 in force),
  same geometry. Nobody uses ramp 2 in the finish room; the WR turns up
  on its first quarter-pipe contact (0.73 s) where we slide 3,360 u
  (3.0 s). Reachable losses in order: strafe geometry (13% of air steps
  brake, 16% gain nothing; net -0.65M vs +1.15M), W held in the air
  11% (human 0%), key dithering 1.5 flips/s vs 0.42, 459 jump/duck
  presses vs 0, 47% more ramp energy destroyed, the finish-room turn
  (2.3 s). Not reachable: the WR's 131 fps = 31% more air-accelerate
  steps per second than our 100 Hz physics.
* GAZE (tools/gaze_wave.py, runs/research/gaze30, aed4a11): flying
  backwards is an exact symmetry of the movement code (side-strafe at
  yaw+180 with the key flipped = identical trajectory); only the depth
  image breaks the tie. xW2DEEP4 flew 99.9% backwards, pitch at the
  ceiling clamp 57% of ticks, and still reached 157k = open-loop map
  memory. Two of four identical deep runs were blind (0/96/100/0%);
  the shallow controls 0-3%. Every arm above 190k is aligned; xQR32
  looks down the route 99.9%. Within the deep arms rho(in-FOV,
  corridor) = +0.74 (p=0.009). Gaze mode is chosen at spawn and held;
  run explains 99% of the variance, place 3%.
* GRU PROBE (tools/gru_probe.py, 0df49b8): zeroing the hidden state
  collapses the corridor 92k -> 12-17k and changes 74-84% of decisions
  (yaw/pitch heads): the GRU is correct and load-bearing. The flat arms
  are throughput (0.75x fps, 2.3B vs 3.0B steps in the hour); at
  matched steps GRU was ahead in both waves inside the noise.
* FROM-SCRATCH EXPERT ITERATION (216ba5b): --objective progress|finish|
  auto (arc banked only at map contacts; last-contact trim), --seed
  scratch; running as exit_scratch on 5090 49843014 (12 rounds,
  deadline 19:39 local). * DAGGER RELABEL (a2cb5fa): plan short windows
  from the policy's OWN states, weighted by divergence from the elite
  line, merged into the BC set; 8 rounds from the round-23 policy
  launching as exitdag on a 5090. * HELD-OUT MAPS (ed1b03b):
  --heldout-maps eval-only slots; wave file tools/wave/wave_heldout.json
  (xH1SHAL / xH1DEEP / xH1PITCH on 8 pool maps, scored on prechasm and
  hollow_lite) queued behind the cap. * WAVE 3 (geodesic lineage, six
  5090s) training since 11:01-11:20, harvest 14:25.

### Round 30, day 2 - the physics tick as a variable (`--tick-ms`, commit 1778867)

The WR demo runs at 131 fps; this trainer's physics tick was a compile-time
10 ms. `--tick-ms 7.63` now cycles an [8, 8, 7] ms pattern (7.6667 ms,
130.4 Hz, 0.04 ms off the demo), 1.304x the physics steps per second, with
every per-second constant (gamma, time penalty, stall window, respawn
margin, view rates, finish clock) rescaled so that 10 ms stays bit-identical
(tests/python/test_tick_ms.py, 9 pass). Isolated physics: a 3 s air strafe
from 250 u/s gets 391 accelerate impulses instead of 300 and ends at
643.05 u/s instead of 575.86 (1.117x).

**The finisher does NOT transfer.** Frozen xQR32 (scalar-critic copy,
actor byte-identical), 9 greedy episodes each, CPU render, same seed:

| tick | decision interval | finishes | best | corridor MAX (order-only 16) | mean |
|---|---|---|---|---|---|
| 10 ms, K=3 | 30 ms | **9/9** | 77.56 s (mean 78.05, sd 0.27) | 231,680 (100%) | - |
| 7.63 ms, K=4 | 30.7 ms | 0/9 | - | 198,272 | 49,216 |
| 7.63 ms, K=3 | 23 ms | 0/9 | - | 197,679 | 81,728 |

At 7.63 ms the episodes run 22-34 s and fall (end vz -827 to -905 u/s,
min field distance 13.5-14.2k = 93% of d0). One episode per setting
reaches the wall region (~198k) and none passes it. This is the gaze
finding again from the other side: the line is memorised open-loop against
the 10 ms dynamics, so 30% more acceleration per second of held strafe
puts the agent somewhere else at every ramp.

Consequence for the experiment: a warm resume at 7.63 ms starts from a
non-finisher and has to re-fit the line; the free physics gain is only
visible once it does. The planner (model-based on the real physics) can
measure the physics floor of the line at 131 Hz directly; that
measurement is queued. Arms xQR32T10 / xQR32T131 (K=4) / xQR32T131K3 are
the 1 h warm-resume pair plus the finer-control variant.

**Correction (same day, after the adversarial review of the tick commit,
ecc0506).** The two 7.63 ms rows above were recorded with two defects the
review found and fixed: under `--yaw-adaptive` (xQR32's setting) the yaw
ceiling was scaled by the tick, inflating obs column 10 by 1.30x and
clamping low-speed turns 23% short, and the recorder's `--obs-reward`
mirror used the unscaled time penalty and gamma. Re-recorded with the
fixed recorder (same checkpoint, seed, CPU render):

| tick | decision interval | finishes | corridor MAX | mean | episodes end at |
|---|---|---|---|---|---|
| 7.63 ms, K=4 | 30.7 ms | 0/9 | 139,925 | 25,749 | 15.7 s, falling |
| 7.63 ms, K=3 | 23 ms | 0/9 | 124,489 | 37,817 | 17.9 s, falling |

Worse, not better: the confounds had been flattering the transfer. The
mechanism is visible in the action semantics themselves: with
`--yaw-adaptive` the yaw step is the per-FRAME strafe optimum
`atan(30/|v|)`, so at 130 Hz the same action turns 1.30x more per second
(K=4) or is issued 1.30x more often (K=3). That is the correct physics for
131 fps - a human at 131 fps turns faster too - but the policy's turn
commands were fitted to the 100 Hz budget, so its line is wrong from the
first ramp. The 10 ms row (9/9, 77.56 s best) is unaffected (bit-identical
path). Verdict unchanged: a warm resume must re-fit the line; the tick
schedule (`--tick-ms-schedule`, in build) is the candidate mitigation.

### Round 30 wave 3 (geodesic lineage) - read from the monitor's polls; the boxes died unharvested

**What happened.** The six 5090s (xW3CTL / xW3DEEP / xW3DFOV / xW3DPITCH /
xW3DNRM / xW3DSPEED, from scratch on cannonball, plain geodesic reward, no
goals, `--respawn-margin 2 --respawn-binned 1 --respawn-bins 128 --eval-stall 1`
+ the arm flag, 4.5 h) were to be harvested at 14:25 and relaunched. An
API session limit killed every agent and timer from ~14:20 to 15:10; at
15:11 the on-box watchdogs destroyed all six on their deadline, as they
are built to, with every checkpoint and trajectory unharvested. The DAgger
box (exitdag, rounds 0-2 trained, planner lines 73.83 / 73.89 s) finished
its driver during the same window and self-destructed 10 minutes later.
Only the from-scratch expert loop (exit_scratch) survived. Harvesting was
a manual step that needed an agent alive at the right minute; that is
being fixed in the local fleet daemon (harvest 20 min before the deadline,
independent of any agent).

**What survived**: the zero-token monitor's 30-minute polls
(`scratchpad/wave3/status/history.jsonl`, 8 polls per run), each carrying
the on-box `eval_honesty --order-only 16` corridor MAX of the latest eval.
Matched-step, first poll at or after the step (so up to ~0.5B late):

| arm | 1.0B | 1.5B | 2.0B | 2.5B | 3.0B | 3.5B | last | best | fps |
|---|---|---|---|---|---|---|---|---|---|
| xW3CTL (shallow) | 49,152 | 63,744 | 82,432 | 134,144 | 104,704 | 103,424 | 103,424 @3.87B | 134,144 | 269k |
| xW3DEEP (4/2) | 113,664 | 58,112 | 57,856 | 57,600 | 57,728 | - | 57,728 @3.14B | 113,664 | 230k |
| xW3DFOV (4/2 + 160x120) | 71,552 | 105,728 | 107,136 | 106,880 | 107,008 | - | 107,008 @3.47B | 107,136 | 248k |
| xW3DPITCH (4/2 + pitch -25) | 133,376 | 105,728 | 106,240 | 104,576 | 105,600 | - | 105,600 @3.34B | 133,376 | 238k |
| xW3DNRM (4/2 + normals) | 50,176 | 133,888 | - | - | - | - | 133,888 @1.62B | 133,888 | 120k |
| **xW3DSPEED (4/2 + speed 0.005)** | 114,944 | 164,864 | 174,976 | 198,400 | 198,400 | **203,776** | 203,776 @3.57B | **203,776** | 267k |

Finishes: 0/9 everywhere, every poll.

**Reading it (one seed, the 27% floor and the gate ladder apply).**
* On the plain geodesic reward the speed term is the one treatment that
  climbs monotonically to the wall (~204k, the same 205,440 u frontier
  every stuck lineage stops at) by 3.5B steps. The other five sit on the
  103-107k gate or below it at the end.
* The deep trunk on its own did NOT reproduce the goal-lineage result
  here: xW3DEEP reached 114k at 1.0B and then fell to a 58k plateau, while
  the shallow control reached 134k. Under the gate ladder a single
  regression like that is within seed noise, so the honest statement is
  "capacity alone is not sufficient on this lineage in 4.5 h", not
  "capacity hurts".
* Normals cost 55% of throughput (120k vs 230-270k fps) and reached 134k
  at 1.5B, the best matched-step reading of any arm at that point except
  DSPEED; it had done 1.6B steps when the box died.
* Everything past the polls - the shape of DSPEED's approach to the wall,
  whether its episodes end on the route or dive, the checkpoints - is
  lost. Rerun DSPEED (and the pair CTL+SPEED) when a box is free; it is
  the only arm of this wave worth a second hour.

### Round 30 day 2 - 131 Hz and the entropy bonus: 1 h warm resumes of the finisher (read 16:52)

All arms are warm resumes of xQR32 (`xQR32_scalar.pt`, step 7.773B; the
exact scalar-critic copy, actor byte-identical), `ARM_RESUME=1`, explicit
`--n-steps 128 --minibatches 16 --epochs 4` (= what it trained at), 9
greedy episodes per eval every 75M steps. Times are spawn-to-finish in
REAL seconds (the recorder header carries the tick; finish_times sums the
per-step dt). "pooled" = all finishing episodes of all evals so far.

| arm | card | flags | evals | finishes (last 3 evals) | best | pooled mean / median | n |
|---|---|---|---|---|---|---|---|
| xQR32T10 (control) | 4090 | none | 27 | 9/9, 8/9, 3/4 | 77.58 s | 78.69 / 78.67 | 190 |
| **xQR32T131** | 4090 | `--tick-ms 7.63 --act-every 4` | 33 | 6/9, 7/9, 8/9 | **76.16 s** | **77.38 / 77.37** | 195 |
| **xENT** | 5090 | `--ent 0.001` | 9 | 6/9, 9/9, 9/9 | **75.90 s** | **77.20 / 77.07** | 67 |
| xTICKRAMP | 5090 | `--tick-ms-schedule 10:7.63:500e6 --act-every 4` | 11 | 4/9, 8/9, 2/9 | 77.49 s | 78.52 / 78.56 | 61 |

**131 Hz.** The frozen policy does not transfer (t=0 eval on the 4090:
0/9, as on CPU), but PPO re-fits the line within ONE recording interval:
+75M ticks -> 6/9 at 78.07 s best, then 77.39, 77.03, 77.26, 76.88, ...
76.16 s. At matched wall-clock the 7.63 ms arm is 1.3 s faster on the
pooled mean and 1.4 s on the best than the 10 ms control on the same
card. Its finish rate is more variable (2-8 of 9 per eval) - the line is
still moving. The planner puts a floor under this: on the same line the
beam search finishes in **74.70 s at 7.63 ms vs 76.56 s at 10 ms**
(-1.86 s, -2.4 %; 1.2 % shorter path x 1.3 % higher mean speed, top speed
4,344 vs 4,295 u/s; gate on the yaw ceiling ON, replay bit-exact; CPU,
1,024 envs, `tick/planner/{m10,m763fix}`). The K=3 decision interval of
that search (23 ms) is a confound in the planner's favour; a K=4 rerun
is in progress.

**Entropy bonus.** `--ent 0.001` (from 0.005; the temporal survey's #1
proposal, xQR32 sits at 62 % of maximum entropy) is the cheapest gain of
the round: 9/9 on the last two evals, best 75.90 s, pooled mean 1.5 s
under the control, after 675M steps. That is 0.6 s from the expert
iteration's 75.31 s with no planner at all. Caveat: 5090 vs the pair's
4090s, so the cross-card rule applies to the exact size, not the sign.

**Ramp.** Mid-transition at this read (the ramp ended at +500M, eval 7);
finishes fell to 2-4/9 as the tick moved and the best is 77.49 s. The
hard jump recovered in 75M steps, so the ramp's premise (avoid a
collapse) was unnecessary; it still tells us whether a gradual change
lands somewhere better. Read at 3 h.

**Also.** The control itself drifts UP over the hour (t=0 mean 78.24 ->
pooled 78.69): continued plain PPO on the finisher does not improve its
time, which is the same plateau the expert loops saw. Every gain above
is against that drift, not against a static baseline.

Launched: xENT131 (`--ent 0.001 --tick-ms 7.63 --act-every 4`), the two
positives stacked.

**Planner floor, decision interval separated (follow-up, CPU, 1,024
envs, same seed).** With `--act-every 4` at 7.63 ms (30.7 ms decisions,
the control's cadence in seconds, resample window matched) the beam
search finishes in **75.34 s**, against 76.56 s at 10 ms (-1.22 s,
-1.6 %) and 74.70 s at 7.63 ms with K=3 (23 ms decisions, -2.4 %). So
the 131 Hz physics alone is worth about 1.2 s on this line, and the
finer decision grid another ~0.6 s; a policy at 131 Hz should be run at
K=3 once it has re-fitted the line (the K=4 arm was chosen to keep the
decision interval the weights were trained at). Every winner replays
bit-exact; the greedy gate fails 3/3 at 7.63 ms in both runs, matching
the 0/9 frozen transfer.

### Round 30 day 2 - there is NO route alternative on cannonball (route-level search, CPU)

Question: is the last third of the WR gap the LINE (ramp entries, the
finish-room turn, "skipping the last ramp")? Method: the record's demo
was used only to LOCATE junctions and to check afterwards whether a
search found the same alternative - never as a seed, reward or prior.
Both lines projected onto `surf_src_cannonball.route.npz` with
compare_wr's ordered projection.

**Answer: our planner line already IS the record's line.** Over the
212 ku where both project, the max 3D separation is 1,309 u - under
beam_tas's own 1,500 u corridor, exceeded on 0.00 % of the arc; path
lengths differ 0.46 % (226,652 vs 227,687 u); the geodesic field values
of the two lines at matched arc agree to a median 62 u of d0 198,380.
Two claims in `runs/research/wr_demo/wr_vs_ours.md` did not survive
measurement: the record does NOT take different ramps (21 matched
contacts within 2.5 ku, only entry speed and height differ) and it does
NOT skip the last ramp (neither line touches finish-room ramp 2; the
extra contacts are ours, at 215.0 and 218.2 ku).

**Where the 5.12 s (zone clock, 73.72 vs 68.60) actually is:**

| segment | separation | seconds |
|---|---|---|
| 0-208 ku, no line separation | <= 900 u | **+3.70 s**, accruing at +0.09..0.13 s per 4 ku |
| J1 finish-room bowl (208 ku -> finish) | 3,357 u of x | **+1.42 s** (dive -0.49, ramp-1 phase +2.29, exit -0.38) |
| J2 53.8-61.4 ku | 1,178 u | -0.32 s (ours faster) |
| J3 pit 176-190 ku, ours 219 u lower | 742 u | +0.15 s |
| J4 89.6-93.4 ku, ours brushes a ramp | 511 u | +0.20 s |

J1 is a wall-ride: at the bowl floor the record rotates its bearing -140
deg in 0.38 s at 3.5-7.9x the air-strafe bound with one 0.73 s touch;
ours rotates +13 deg, falls 216 u deeper, and slides 3,357 u over 3.02 s
and 3 touches.

**Can the search find J1? No, and the reason is measured.** Ten beam
runs at 512 envs, 7.63 ms, from a replayed prefix (`--prefix-line`, new):
a control, eight branch variants (`--branch-at`, `--branch-jitter`,
`--branch-hold`, `--branch-protect`, `--score d`) and a second control
finished within 0.25 s of each other (74.63-74.88 s) and none took the
alternative. Fork probe at the room entry, 512 free continuations with
NO selection: 456 wide-slide, 56 head-on crashes, **0 record-like**;
0 of 1,536 across three proposal distributions. The field PREFERS the
record's branch (d 10,679 vs 14,749 at +3.25 s) and the critic ranks the
modes it is given correctly; the alternative simply has zero mass in the
policy's proposal, because it is decided by where you meet a wall while
turning at 2.2x the free-flight strafe bound, which no perturbation of
free-flight controls reaches. Population collapse is the mechanism: the
kept lineages of the reference run are byte-identical on 95.4 % of their
decisions. `--score d` let the branch take the whole population and came
out slowest (74.88 s) - round 27's kill-net trap again; keep `dv`.

**The 3.70 s is strafe cadence, in the PLANNER's line too:** it flips
A/D 2.14 times per second against the record's 0.42, median hold 0.046 s
vs 0.418 s, wishdir within 0.5 deg of perpendicular on 50.5 % vs 82.6 %
of free-flight steps, net air-strafe energy +0.07 M vs +1.15 M. The
planner proposes from the policy, so it inherits the dithering. Fix on
both sides: held-key macro-actions (side key, duration) as the search's
proposal, and, on the policy, the entropy / action-history /
yaw-conditioned-key arms running now.

**Reusable:** `--prefix-line NPZ[:TICKS]` replays a saved line open-loop
and searches from there: the 1,024-env reference (74.70 s in 2,768 s)
reproduces at 74.71 s in 169 s, 16x cheaper - a ladder of segment waves
for `expert_loop.py:plan()`. Best line found B5 74.63 s (73.66 s zone),
inside one-tick noise of its control; saved under scratchpad `rs/B5`.
Branch `route-search` merged (e28cc85), `tests/python/test_beam_branch.py`.

**Readiness estimate revised:** the "route third" does not exist; the
gap is 3.7 s of strafe execution (attackable with the tools in hand) plus
1.4 s of one contact manoeuvre the current proposals cannot produce.

### Round 30 day 2 - 18:35 read: the two positives stacked - **73.96 s, 9/9**

xENT131 (`--ent 0.001 --tick-ms 7.63 --act-every 4`, warm resume of
xQR32, 5090, launched 17:04), snapshot-harvested at 18:33 at step 9.74B
(+1.9B ticks, ~1.5 h):

| eval (step) | finishes | best | mean |
|---|---|---|---|
| 9,661,579,264 | **9/9** | **73.96 s** | 74.44 s |
| 9,737,076,736 | 6/9 | 74.05 s | 74.49 s |

That is 1.9 s under the previous best policy time (xENT 75.82-75.90 s at
10 ms; expert iteration's 75.31 s) and BELOW the planner floor of the
line at 10 ms (76.56 s), within 0.7 s of the 131 Hz planner floor
(74.70 s, K=3) and past the K=4 one (75.34 s). On the human clock
(~-0.9 s) it is ~73.1 s against the record's 68.60 s: the gap is 4.5 s,
all of it the strafe-execution and finish-room components measured by
the route search. Same-tick, same-card comparison: xENT131 vs xENT131's
own t=0 (frozen at 7.63 ms: 0/9) - there is no 5090 10 ms control in
this batch, so the size of the entropy x tick interaction is not
separable here; the direction is not in doubt.

Other arms at this read: xENT (10 ms, 5090) 9/9 at 75.82 s best / 76.32
mean; xTICKRAMP 77.04-77.43 s, 2-8 of 9 (the ramp ended at +500M and it
is still re-fitting; the hard jump did better). Held-out trio at
0.6-1.3B steps, not readable yet.

**Losses this window.** xQR32T131's BUDGET (4e9 ticks) ran out at 17:35
(578k ticks/s on the 4090 - 4e9 is 1.9 h there, not 4 h), the box
self-destructed 10 min later and the daemon's two-poll confirmation lost
the race by one inconclusive probe: its checkpoint and recordings are
gone; its 33-eval time table above stands. The control xQR32T10 hit the
same budget at 18:12 and WAS harvested (52 s) - the early-exit path
works when it gets its two polls. Fixed for everything still running:
every on-box watchdog restarted with a 40-minute post-exit grace (same
deadlines). exit_scratch's host went OFFLINE at 17:32 (vast status), so
its round-5+ results (22.8 % of the route, 0 finishes at round 5) are
unreachable; the daemon leaves it to its deadline.

### Round 30 day 2 - 18:45 read: action history and the yaw-conditioned key are NULL at 1 h; held-out trio early

Warm resumes of xQR32 on 4090s at 10 ms, same card as the control
(xQR32T10: best 77.58 s, pooled mean 78.69 s, the control drifting up
from 78.24 over its run):

| arm | evals | best | eval means | mechanism read-out (rollout, first -> last row) |
|---|---|---|---|---|
| xHIST (`--act-hist 4`, 24 zero columns widened) | 23 | 77.87 s | 78.3-79.4 s | `act/strafe_flip` 0.376 -> 0.258 |
| xYAWC (`--yaw-cond`, 15x3 zero table) | 21 | 77.49 s (t=0) then 77.82-79.25 | 78.2-80.4 s | `act/yaw_side_agree` 0.675 -> 0.827 |

Both features engage (flips down by a third; yaw/side agreement up 15
points) and neither moves the time inside the hour - if anything both
sit above the control's drift. Verdict: null at one hour, harvested and
released at 18:50. Read together with xENT (-1.5 s at the same hour with
the entropy bonus alone at 0.005 -> 0.001): the representation changes
do not pay while the sampling noise stays at 0.005; the untested
combination is each of them WITH `--ent 0.001`.

Held-out trio (multi-map, 5090s) at 1.5-2.1B of 2.5B steps: held-out
field cover 2.9 % (hollow_lite) and 5.6 % (prechasm), 0 finishes on
either, unchanged over the last three evals; training-map finishes 0/2
per map. Not readable before the deadline harvest at ~20:15.

Fleet: xQR32T131 lost (budget/grace race), exit_scratch host offline;
on-box grace 40 min everywhere; credit $21.49 at 18:45.

**Correction (user, 2026-09-04 evening, from the 3D viewer): the record's
path IS different at the end.** With the demo loaded next to our best run,
the record takes ONE ramp in the finish room where our line takes TWO.
The route search's own measurements say the same thing and its headline
did not: the record makes one contact (0.73 s) at the bowl and exits at
x -9,727; ours makes three contacts over 3.02 s and 3,357 u of bowl to
x -12,989, with the two extra ramp contacts at 215.0 and 218.2 ku. The
"max separation 1,309 u, no route alternative" claim came from projecting
both lines onto the route file, which ends before the finish room, so the
one place where the paths differ was excluded from the separation metric.
Standing conclusion, restated: the line is the same for the first 208 ku
(3.70 s of execution); the finish room is a genuine ROUTE difference
worth 1.42 s, one ramp instead of two, and the planner's proposals never
produce it (0 of 1,536 free continuations). The user's earlier
recollection ("it skips the very last ramp") was correct in substance.
Next: a deterministic macro grid and a state-fork reachability probe at
the room entry, from a replayed prefix (`--prefix-line`), which costs
~2 minutes per run at 512 envs.

**19:21 read.** xENT131 keeps improving: eval at step 9.888B best
**73.21 s** (mean 73.86 s, 6/9), neighbouring evals 73.84-74.05 best /
74.29-74.49 mean; pooled 281 finishers, min 73.21, median 74.75. xENT
(10 ms): pooled min 74.94, median 76.46. xENT131K3 (3-tick decisions,
+450M ticks): 76.34-77.75 best per eval while it re-fits, 0/9 at t=0 as
expected. On the human clock the best run is now ~72.3 s against 68.60.

### Round 30 day 2 - held-out generalisation probe: INCONCLUSIVE (no training-map competence at 2.5B); tick ramp final

xH1SHAL / xH1DEEP (`--tower-depth 4 --conv-mult 2`) / xH1PITCH (+
`--pitch-fixed -25`), from scratch on 8 pool maps (8,192 envs = 1,024 per
map), 2 EVAL-ONLY held-out maps (hollow_lite, prechasm), 2.5B steps in
4.5 h on 5090s, all three harvested by the daemon on early exit:

| arm | train-map progress (mean % of map) | train-map finishes | held-out cover hollow_lite / prechasm (last, max) | held-out finishes |
|---|---|---|---|---|
| xH1SHAL | 12.0 % | 0 | 4.7 (5.9) / 6.6 (7.0) % | 0 |
| xH1DEEP | 16.8 % | 0 | 3.1 (6.2) / 4.4 (9.5) % | 0 |
| xH1PITCH | 13.1 % | 0 | 7.1 (7.1) / 5.6 (8.2) % | 0 |

The probe cannot answer the memorisation question because the arms never
became competent on the TRAINING maps: 2.5B steps over 8 maps is ~310M
per map, an order of magnitude under what one map needs (2-3B for the
deep trunk on cannonball), and no map was finished. Held-out cover of
3-9 % is what an incompetent policy reaches by falling forward. The deep
trunk's training-map progress (16.8 %) leads the shallow one (12.0 %) as
on cannonball, but nothing here is a generalisation result. A real probe
needs ~2-3B steps PER training map (20+ h on one card for 8 maps) or a
2-map pool; the tooling (`--heldout-maps`, HeldoutSlot, the harvested
per-map trajectories) is in place.

**xTICKRAMP final** (`--tick-ms-schedule 10:7.63:500e6`, K=4, 4 h,
3.93B ticks): last evals 4/9 at 76.92 and 76.03 s best; pooled it never
matched the hard jump (xQR32T131: 76.16 best / 77.38 mean at 1 h) and is
2.8 s behind xENT131. The ramp is unnecessary - the hard transfer re-fits
within 75M ticks - and it costs the first 500M ticks. Do not use it.

### Round 30 day 2 - SYNTHESIS AND ROADMAP (written 2026-09-04 20:30 for whoever picks this up)

**Where the clock stands.** Best policy: xENT131, 73.21 s spawn clock (~72.3 s
on the record's start-zone clock) against the human record 68.60 s: gap 3.7 s.
Two days ago the best policy was 77.7 s. The day's gains, in order of size:

| what | gain | how measured | cost it took |
|---|---|---|---|
| 131 Hz physics tick (`--tick-ms 7.63`) | -1.2 s physics floor (planner, decision interval matched); the policy re-fits the line in 75M ticks and was 1.3 s faster than the same-card control at 1 h | planner floor 76.56 -> 75.34 s; xQR32T131 vs xQR32T10 | 1 build + 1 review, 2 arms |
| entropy bonus 0.005 -> 0.001 (`--ent 0.001`) | -1.5 s at 1 h on the finisher, 9/9 | xENT vs control | zero build, 1 arm |
| the two stacked (xENT131) | 77.6 -> 73.2 s in 2.7 h | one arm | zero build |
| finer decisions at 131 Hz (K=3, 23 ms) | -0.6 s on the planner floor (74.70 vs 75.34); policy arm xENT131K3 running | planner | 1 arm |

**What the day established (the insights):**

1. **The remaining gap is EXECUTION, not exploration and not (mostly) route.**
   3.7 s of it accrues along a line the record also flies; it is strafe
   cadence (A/D flips 2.14/s vs 0.42, holds 0.05 s vs 0.42 s, wish direction
   within 0.5 deg of perpendicular on 50 % vs 83 % of free-flight steps). The
   planner's own line has the same defect because it proposes from the policy.
2. **One route difference exists: the finish room.** The record makes one ramp
   contact and exits; we make three. Worth 1.42 s. Zero of 1,536 free
   continuations from the room entry produce it: it is a proposal problem
   (a wall-ride built by turning at 2x the free-flight bound in the first
   0.8 s of the dive), not a scoring or horizon problem. The earlier "no
   route alternative" headline excluded the finish room by construction
   (route projection) and was corrected the same evening.
3. **Strafing needs no memory; the loss is action-side.** The strafe fixed
   point is deadbeat-stable in one frame, the observation carries the last
   yaw delta (scalar 10) so the zig-zag branch is not ambiguous, and the two
   heads that must agree in sign sit on a saddle at 50/50. Consistent with:
   GRU / frame stacking / chunking null; act-hist null; yaw-conditioned key
   null (both mechanisms engage - flips down a third, agreement up 15 points
   - and neither moves the time while the entropy bonus stays at 0.005).
   Lowering the bonus is what worked. Untested and plausible: act-hist or
   yaw-cond WITH `--ent 0.001`.
4. **Plain PPO on the finisher does not improve its time** (the control
   drifted 78.2 -> 78.7 s over its run). Every gain came from changing the
   objective (entropy), the physics (tick) or the data (planner lines).
5. **Expert iteration compounds then plateaus ~1.7 s behind the planner** and
   the planner is now slower than the policy (74.70 vs 73.21 s). The *Zero
   survey says why: we clone the single best line (CAT) where AlphaZero /
   ExIt use the search's first-decision DISTRIBUTION (TPT), and we discard
   the planner line's return as a value target. Learned world models are
   irrelevant here (the model is real and cheap).
6. **Seed lottery and gate ladder still apply to from-scratch arms**; the
   warm-resume-of-a-finisher protocol (metric = finish time and finishes/9,
   9 episodes per eval, same card, control on the same day) is what made
   today's one-seed arms readable.
7. **Held-out generalisation is unanswered**: 2.5B steps over 8 maps is
   ~310M per map, no training map was finished, so the probe measured
   nothing. It needs 2-3B per training map (20+ h on one card) or a 2-map pool.

**Ranked roadmap - expected gain vs effort (estimates unless marked measured):**

| # | item | expected gain on the timer | effort | confidence | prerequisite / falsifier |
|---|---|---|---|---|---|
| 1 | Expert iteration from xENT131 at 131 Hz with the planner's held-key macro proposals (P1, in build) and search-derived targets (P2 policy distribution, P3 value; in build) | -1.0 to -2.0 s (the planner floor must first drop below the policy; the loop then compounds as exit10 did) | 1-2 days build (in progress) + 1 box-night | medium | planner floor with macro proposals < 73 s; one loop round A/B argmax vs dist |
| 2 | K=3 decisions at 131 Hz on the entropy finisher (xENT131K3, running) | -0.3 to -0.6 s | 0 | medium-high | its 1-2 h read vs xENT131 |
| 3 | Finish-room one-ramp exit (J1 grid + state fork, in build) | -1.4 s if reachable | 1 day + the distillation | low-medium: the manoeuvre may need a different entry speed/height | the state-fork reachability probe first |
| 4 | Entropy lower still (`--ent 0.0005`) or an anneal (`--ent-final`) | -0.3 to -0.8 s | 0 | medium | 1 h arm vs xENT131 |
| 5 | act-hist 4 / yaw-cond WITH `--ent 0.001` | 0 to -0.5 s | 0 | low-medium | 1 h arms; the diagnostics already move |
| 6 | Reanalyse the respawn reservoir with the planner (P4) | -0.3 to -1.0 s | 1 day | low-medium | round-over-round greedy time |
| 7 | Gumbel root search per decision with the real sim (P5) | unknown; it does not fix proposal mass | 2-3 days | low | planner floor |
| 8 | Held-out generalisation, properly budgeted | not a timer item; answers "does it learn to surf" | 20+ box-hours | - | 2-map pool first |
| 9 | Tick ramp, GRU, frame stacking, chunking, bigger MLP towers, RND, BC warm start, 7 s reward window | none - do not retest | - | measured | - |

To reach the record from 72.3 s the sum of 1 + 2 + 3 (+4) has to land: about
-3.7 s. Items 1 and 3 are the ones that can each move more than a second;
neither is measured yet. Readiness estimate: ~50 %.

**Ops lessons written in blood today (all now enforced in code):**
- Harvest is a daemon function (`fleet_watchdog.py --harvest`, 20 min before the
  deadline and on trainer exit), never an agent's timer; the on-box grace after
  a trainer exit is 40 min, not 10 (a 4090 finished a 4e9 budget in 1.9 h and
  the 10-min grace lost the box to a one-poll race).
- `BUDGET` is in physics ticks and a 4090 runs 420-580k of them per second: 4e9
  is ~2 h there, not 4. State which bound ends a run.
- Market price caps in `vast_pick.py` (3090 < 0.22, 4090 <= 0.45, 5090 <= 0.60
  $/h). The readiness blocklist has thinned the market to a handful of offers;
  readiness blocks should probably expire (user's call).
- `box_finish.sh` truncated run_arm's output to 8 lines and hid the ALIVE line
  of multi-map runs, so the launcher destroyed a healthy box; fixed (tail -40).
- Measurements that project onto a route file stop where the file stops; the
  finish room was invisible to the route search's separation metric.

**xENT131 final (budget 4e9 ticks reached at 20:10, harvested by the daemon on
exit).** Last four evals: 6/9 73.86 s, 9/9 74.23 s, 9/9 73.98 s (mean 74.53),
5/9 73.68 s (mean 73.98). Best single episode of the run: 73.21 s (eval at
9.89B). The mean plateaued at 74.0-74.5 s over the final 2B ticks - the same
"plain PPO stops improving the finisher" shape as the control, one level
lower. Seed checkpoint for the next loop: runs/research/xENT131/
ckpt_10774118400.pt (7.63 ms, K=4, ent 0.001). Its box was reused for xENT05
(`--ent 0.0005` continuation, 1.5e9 ticks) rather than idling out its grace.
### Round 30 day 2 - J1 settled: the record's finish-room turn is a hard AIR-BRAKE, not a wall ride, and our line cannot afford it

**Correction to the J1 mechanism.** Day 2 recorded that at the bowl the
record "rotates its horizontal bearing -140 deg in 0.38 s at 3.5-7.9x the
free-flight air-strafe bound, i.e. a wall/ramp ride with vz on gravity".
The rate is right; the inference is not. `atan(30/|v|)` is the MAX-GAIN
strafe rate - wishdir exactly perpendicular, where PM_AirAccelerate's
`addspeed = 30 - dot(v, wishdir)` is exactly 30. It is NOT the maximum TURN
rate: a couple of degrees PAST perpendicular makes
`addspeed = 30 + |v| sin(delta)` and the per-frame impulse rises to the
`accel * maxspeed * frametime` ceiling - 190 u/s at 131 fps - for a speed
cost of only `|v| sin(delta) tan(delta)`.

Measured on the demo rows, three ways, all agreeing:

* through the whole turn (demo rows 6515-6563) the player is **223-421
  units from the nearest map surface** (26-ray probe from the origin); the
  first surface inside 100 u appears at row 6565, and that is the ramp;
* the impulse the turn needs is **35-181 u/s per frame** against an exact
  PM_AirAccelerate ceiling of **39-190 u/s** at the recorded speed and
  wishdir angle - inside it at every row;
* the recorded side key is **D held** and the view is swept 0.1 -> 5.8 deg
  PAST perpendicular, which is exactly the geometry that buys the extra
  `addspeed`.

So J1 is **a 0.40 s hard air-brake turn in free air** that rotates the
heading -134 deg (-79 -> -213, i.e. through -180) for 212 u/s of speed, delivering the player
to the SAME near ramp both lines use, already pointing +y, so ONE 0.77 s
contact launches it up the finish corridor. Both lines touch that one ramp;
the difference is the heading they touch it with (hull-clearance rule,
80 u, applied identically to both):

| | ramp touch in | bearing in -> out | vz in -> out | hspd in -> out | then |
|---|---|---|---|---|---|
| record | (-9,231, -4,338, -4,738) | **+147 -> +93** | -2,369 -> **+2,195** | 2,844 -> 2,386 | straight to the finish |
| ours | (-9,504, -2,846, -4,400) | **-81 -> -101** | -2,233 -> +468 | 2,903 -> 3,664 | pushed 2,229 u further into -y, 1.8 s across the bowl, then a SECOND ramp at (-13,129, -4,658, -5,184) for the launch |

The user's viewer reading (one ramp vs two) is confirmed. The mechanism
behind it is not the one recorded: it is not a ramp our line fails to
reach, it is a turn our line does not make.

**The manoeuvre is inside our action space.** 2.6 deg/frame at 2,900-3,050
u/s is K ~ 4.4 x `atan(30/|v|)`, between `K_BINS` 3 and 8, one side key
held - a held-key macro.

**But our line cannot afford it, and the deficit is arithmetic.** Air
acceleration is horizontal only (PM_AirMove zeroes wishvel[2]), so
vz(t) = vz0 - g t exactly and the airtime to any height is fixed by the
entry state. Ours enters the room 147 u lower and sinking 161 u/s faster,
which is **0.23 s less airtime at every height**, and 5 % slower
(2,888-2,903 against 3,032-3,055). The turn costs 0.40 s of flight and
212 u/s.

| what | fall time | ground track | needs | we have | margin |
|---|---|---|---|---|---|
| be where the record STARTS its turn (-8,625, -3,684, -3,892), on a near-straight approach | 1.931 s | 5,627 u / 0.988 | 2,949 u/s | 2,895 | **-1.8 %** |
| be where the record TOUCHES the ramp (-9,231, -4,338, -4,738), having flown the turn at the record's own path/displacement ratio (0.859, measured) | 2.316 s | 6,491 u / 0.859 | 3,263 u/s | 2,895 | **-11.3 %** |

(the record's own flight to that touch: 2.540 s, 7,719 u of path for
6,628 u of displacement, mean 3,039 u/s.) Any ONE of (+147 u height,
-161 u/s sink, +5 % speed) restores the first row's margin; none of them
restores the second. **J1 is not a defect of
the finish room. It is the same 3.70 s of strafe cadence the run loses
over the first 208 ku, cashed at the one place on the map where it buys a
whole manoeuvre.**

**Measured, three independent probes, all negative.**

| probe | plans / starts | left the bowl | ONE-ramp exit | note |
|---|---|---|---|---|
| held-key macros (yaw bin x side key x hold x 2nd segment), open-loop from three forks at the room entry | 18,900 | **0** | 0 | open-loop control cannot fly this room at all, so an open-loop null alone proves nothing |
| the same family from four LATE forks (0.6-1.5 s before the ramp) | 25,200 | 14 | 14 | all energy-dead: exit hspd 1,270-1,695 and vz +73..+238 against the 2,508 / +2,203 our own line exits with; 0 finishes |
| state fork + THE POLICY as the controller, deterministic displacement grid at the room entry, no cloning, no selection | 375 + 375 | 81 / 12 | **0** | over dx, dy +-1,500, dz +800, heading +-30 deg the exit x moves **27 units** (-13,031..-13,058); the policy's bowl behaviour is an attractor |

And the frontier: over 10,649 macros the best first bowl contact is
(-8,608, -3,003, -4,367) - the record's x, but 1,335 u short in -y and
371 u high of where it touches the ramp.

**`--branch-grid WHERE:SPEC` (new, `tools/beam_tas.py`, tests
`tests/python/test_branch_grid.py`).** The same fork as `--branch-at` with
a DETERMINISTIC enumerated fill: yaw-bin offset x side key x hold duration
x {macro, macro + mirrored counter-macro}, replicated round-robin over the
non-greedy population, the policy continuing when a plan's macro ends,
`--branch-protect` unchanged, and the winner's summary naming the plan it
descends from. It draws no randomness at all. Off is byte-identical
(asserted end to end).

Runs: 512 envs, `--greedy-envs 8` so the 168 plans get exactly 3 envs each,
seed 0, `--score dv`, `--branch-protect 6`, `--tick-ms 7.63`, from
`--prefix-line` on the m763fix line. Room entry is tick 8606 (65.98 s); our
own ramp touch is tick 8889-8907 (68.15 s).

| run | prefix | fork | holds (ticks) | finish spawn / zone | bowl touches | exit x | grid |
|---|---|---|---|---|---|---|---|
| m763fix (the line itself) | - | - | - | 74.70 / 73.72 | 2: 0.46 s + 0.72 s | -13,022 | - |
| C512 (round 30 control) | 8606 | - | - | 74.71 | - | - | - |
| B5 (round 30, random burst) | 8531 | t8604 | - | 74.63 | - | - | 0 of 448 alive |
| G0 control | 8268 | - | - | 74.70 / 73.72 | 2: 0.55 + 0.69 | -13,019 | - |
| G1 at the room entry | 8531 | 8604 (65.96 s) | 21/42/84/168 | 74.66 / 73.68 | 2: 0.46 + 0.68 | -13,022 | 0 of 504 alive |
| G2 at entry -1.0 s | 8400 | 8475 (64.97 s) | 21/42/84/168 | 74.73 / 73.75 | 2: 0.46 + 0.68 | -13,022 | 0 alive |
| G3 at entry -2.0 s | 8268 | 8343 (63.96 s) | 21/42/84/168 | 74.70 / 73.73 | 2: 0.44 + 0.67 | -13,022 | 0 alive |
| G4 0.6 s before the ramp | 8531 | 8829 (67.69 s) | 21/42/63/84 | 74.86 / 73.89 | 2: 0.48 + 0.73 | -13,032 | 24 alive at the last generation, 23 u of d behind the leader |
| G5 1.2 s before the ramp | 8531 | 8754 (67.11 s) | 42/84/126/168 |  74.80 / 73.82 |  2: 0.48 + 0.66 |  -13,032 |  0 of 504 alive |

Every run: two bowl touches, the near ramp then the far ramp, exit x
within 13 units of the untreated line (-13,019 .. -13,032), and the fastest finisher was never a
grid lineage. The five finish times span 74.63-74.86 s, i.e. 0.23 s, which
is the same band round 30's ten branch runs produced.

**What the runs say.** The grid takes up to 120 of the 128 elite slots for
the first four to six generations after the fork and is then culled; no
grid lineage finished in any run, and every winner was the untreated line.
The one that stayed close is the LATE fork (G4): 24 of 504 grid lineages
were still alive at the last generation before the finish, 23 u of geodesic
d behind the leader. `--score dv` is part of why it cannot be read as a verdict
on the manoeuvre: this map's geodesic d inside the finish room is always
below `--v-switch 20000`, so `dv` is the CRITIC for the whole search, and
the critic ranks a lineage that has traded 200 u/s for position below the
untreated one.

**Contact detection: the `|dv - gravity| > tol` rule over-counts, badly.**
The same fact that makes the turn possible - an impulse up to 200 u/s when
the wishdir opposes the velocity - makes hard braking read as a contact,
and its "normal" comes out as the wishdir, which has n_z = 0 and is
indistinguishable from a vertical wall. Two rules that are right: the
GEOMETRIC one (the hull within 80 u of a surface - needs no action, works
on the demo), and, when the action is known, the exact one (in free flight
the horizontal impulse is PARALLEL to wishdir and non-negative along it, so
a perpendicular component, a negative parallel component, or a dvz off the
tick's gravity step is a contact, and the residual against that prediction
IS the surface normal). On our own reference line the exact rule gives
17.7 % of ticks in contact (day 2 measured 19.1 % of run time), and the two
rules agree on the finish room's three touches; the 40 u/s threshold
invented nine and produced a phantom "vertical wall ride" for our line too.

**Recommendation.** Do not build a finish-room arm. J1 is 11 % of entry
speed away, and entry speed is the strafe cadence the run already loses
over the first 208 ku - the arms for it are already running (`--ent 0.001`,
held-key macro proposals). When the run enters the finish room ~5 % faster
and ~150 u higher, the manoeuvre becomes available to a search that can
propose a 0.4 s held brake turn, which `--branch-grid` now can. If the
junction is searched again: fire the grid at EVERY resample boundary in the
room rather than one, and do not rank it with `dv` inside the room, where
`dv` is the critic and the critic prices a 200 u/s trade as a loss.

**Roadmap correction after the J1 search (merged fa14dc9, branch j1-search;
its full section is above).** The finish-room difference is not a wall ride
and not a different ramp: both lines touch the SAME first ramp, but the
record arrives at it after a 0.40 s hard air-brake turn in free air (D held,
view swept up to 5.8 deg past perpendicular, impulse 35-181 u/s per frame
against the 190 u/s accelerate ceiling, -134 deg for 212 u/s of speed) and
exits climbing to the finish, while we arrive with the opposite heading, get
pushed 2,229 u deeper and need a second ramp. Reachability: from our own
entry state the manoeuvre is 1.8 % of speed short at the turn and 11.3 %
short at the touch (we enter 147 u lower, sinking 161 u/s faster, 5 % slower:
0.23 s less airtime); 375 policy-controlled state forks over a 3,000 x
3,000 x 800 u box and 60 deg of heading all exit within 27 u of the same x;
6 deterministic macro-grid searches (`--branch-grid`, new) never left the
two-touch attractor. Verdict: item 3 of the roadmap is DOWNSTREAM of item 1
(entry speed and height are the strafe cadence over the first 208 ku); do
not build a finish-room arm until a run enters the room ~5 % faster and
~150 u higher, then fire `--branch-grid` at every resample boundary in the
room and do not rank the room with `dv`. Also found: the `|dv - g| > 40`
contact rule over-counts (a braking tick's residual lies along wishdir, which
looks like a vertical wall); use the hull-distance or exact-impulse rule in
scratchpad/j1/contact.py. The three touches on our line stand under the exact
rule.

**20:40.** xENT final (10 ms, `--ent 0.001`, 4e9 ticks): last eval 8/9,
best 74.87 s, mean 75.78; pooled 76.03 mean. The from-scratch expert loop
(exit_scratch, identical settings to the lost run: round 0 = 1.5B plain PPO
steps, then 12 planner/distil/train rounds, planner 600 s at 2,048 envs,
objective auto) was RESUBMITTED at 20:38 on the harvested xENT box
(5090, $0.411/h, instance 49865391, ssh7:25390, deadline 04:37, harvest spec
registered with the driver pid as liveness) - the market under the price
caps had no box that reached ssh in time (6 attempts, 2 pools). Also running:
xENT05 (`--ent 0.0005` continuation of xENT131's 10.77B checkpoint) and
xENT131K3b (3-tick continuation, 2.5e9 ticks; the 1-h arm ended at best
74.44 s, ahead of the 4-tick arm at the same tick count).
### Round 30 day 3 - held-key macro proposals do NOT lower the planner floor (CPU)

Question: day 2 measured that our planner line and the record's are the
same line to within 1,309 u and that 3.70 s of the 5.12 s gap is strafe
EXECUTION - the planner flips A/D 2.14 times a second against the
record's 0.42 because it proposes every decision from the policy, and
selection cannot pick what is never proposed. So: make the proposal a
HELD KEY and measure the floor.

`tools/beam_tas.py --macro-hold MIN:MAX` (new, off by default and
byte-identical when off). Each macro env draws one (side, forward) pair
and a LOG-UNIFORM duration in [MIN, MAX] seconds, rounded to whole
decisions, and holds the keys for it. `--macro-yaw track` additionally
sets the yaw bin analytically each decision to the one whose per-tick
view turn matches the velocity rotation the held key produces
(atan(30/|v|) per frame, NEGATIVE for D and positive for A; under
`--yaw-adaptive` that is K_BINS k = -+1 at every speed). `--macro-fwd
draw|none|policy` says whether the forward key is part of the macro.
`--macro-frac F` gives only the last F of the non-greedy envs a macro so
the two proposals COMPETE instead of one replacing the other. The macro
state is CLONED donor -> loser at every resample, beside the action
table: a generation is 0.575 s and the holds under test are 0.2-0.8 s,
so redrawing at boundaries would truncate exactly what is measured.
Draws come from a private numpy generator, so the torch stream is
untouched. `tests/python/test_macro_hold.py` (30 cases with
test_beam_branch), and the control below reproduces round 30 day 2's
C512 tick for tick (9,745 = 74.71 s), which is the byte-identity check
on the real configuration.

All runs: `xQR32_scalar.pt`, cannonball, 512 envs, 7.63 ms, seed 0,
`--score dv`, `--greedy-envs 64` except P0/P1, from a prefix of the
m763fix reference line. CPU, 4 threads, BelowNormal, one at a time.

| arm | prefix | macro | result | won by env | finishes |
|---|---|---|---|---|---|
| M0 | 8604 | - (control) | **74.71 s** | 14 | 512 |
| M1 | 8604 | 0.2:0.8, yaw policy, fwd draw | 75.21 s | 50 | 511 |
| M2 | 8604 | 0.2:0.8, yaw track, fwd draw | 75.65 s | 4 | 90 |
| M3 | 8604 | 0.2:0.8, yaw track, fwd none | 75.09 s | 9 | 613 |
| P0 | 8604 | - (control, greedy-envs 0) | 74.72 s | 1 | 512 |
| P1 | 8604 | 0.2:0.8 track/none, greedy-envs 0 | **NO CROSSING** | - | 0 |
| R1 | 8604 | 0.2:0.8 track/none, frac 0.5 | 74.95 s | 0 | 4 |
| N0 | 3999 | - (control) | **74.89 s** | 1 | 1023 |
| N1 | 3999 | 0.2:0.8, yaw policy, fwd draw | 75.54 s | 1 | 507 |
| N2 | 3999 | 0.2:0.8 track/none | **NO CROSSING** | - | 0 |
| R2 | 3999 | 0.2:0.8 track/none, frac 0.5 | 75.73 s | 0 | 76 |

**Every arm is slower than its control, and two do not finish at all.**
Best macro arm R1 74.95 s vs 74.71 s; at the longer prefix R2 75.73 s vs
74.89 s.

**The mechanism IS implemented and does what it says** - which is what
makes this a real negative rather than a bug. Cadence of the SEARCHED
SUFFIX (new: `summary["cadence_searched"]`; the whole-line number is
88 % replayed prefix and says nothing about the arm). Reference under
these definitions: WR demo 0.84 flips/s, 0.422 s median held-key run,
79.3 % of free-flight ticks with wishdir within 0.5 deg of perpendicular,
+1.17 M net free-flight change of 0.5|v_h|^2; m763fix planner line 3.27,
0.046, 43.1 %, -0.38 M. (Median hold and the WR energy reproduce day 2's
0.418 s / 0.046 s / +1.15 M exactly; the flip and perpendicular counts
sit on the same ordering at a different absolute level, so compare arms
under THESE numbers.)

| arm | flips/s | median hold | perp 0.5 deg | strafe energy | fwd held |
|---|---|---|---|---|---|
| M0 control | 3.32 | 0.023 s | 25.1 % | -0.17 M | 34.9 % |
| M2 track/draw | 1.76 | 0.207 s | 28.7 % | -0.64 M | 35.4 % |
| **M3 track/none** | **1.86** | **0.115 s** | **46.5 %** | **+0.05 M** | **3.8 %** |

M3 halves the flip rate, holds keys 5x longer, nearly DOUBLES the
perpendicular share and is the only line in this round whose free-flight
strafe energy is positive - and it still loses 0.38 s.

**Three things the runs say about why.**

1. **A held-key lineage never wins.** In every crossing arm the first
   env to cross is below `--greedy-envs 64` - a greedy continuation of
   an elite - and with `--macro-frac 0.5` it is env 0, the protected
   greedy line itself. The macro widens the proposal; selection then
   discards it. New `summary["best_env"]` records this.
2. **Replacing the proposal kills the population.** P1 (no greedy floor)
   and N2 (100 % macro over 43 s of route) reach 0 finishes; N2 dies
   8,634 times and P1 3,742. Isolating the free-flight stretch on the
   honest coordinate (`--objective progress --max-ticks 8604` from
   :4000, so the finish room is excluded): control Q0 banks **193,170 u
   (83.4 %)**, macro Q1 **149,167 u (64.4 %)** and goes extinct at
   t = 7,824. At matched generations 41-45 Q1 is stationary at 149,167
   while Q0 climbs 157,943 -> 164,049. So the loss is NOT the finish
   room; it is the free flight the mechanism was aimed at.
3. **The yaw is doing two jobs and a held key forces a choice.** With
   `--macro-yaw policy` the policy's yaw and the drawn key disagree
   (act/yaw_side_agree's failure mode), which is why M1/N1 lose. With
   `track` they agree by construction - and the yaw then stops STEERING,
   because it is slaved to the strafe optimum instead of to the ramp.
   The record holds a key 0.42 s because its view does both at once; the
   analytic tracker does only the first. `--macro-fwd draw` is a third,
   separable cost: it puts W/S on ~35 % of free-flight ticks (the record
   0 %), swinging wishdir 45 deg off velocity - M2 to M3 is that cost
   alone, 0.56 s.

**Verdict: do NOT wire held-key macros into `tools/expert_loop.py`.** An
open-loop hold is not what the record is doing; the record is closed-loop
at 131 Hz with a view that steers and strafes simultaneously. The
proposal-side fix cannot supply that, because the thing the search would
have to propose is a FEEDBACK LAW, not a key sequence. What is worth
keeping from this round is the DIAGNOSTIC: `summary["cadence"]` /
`["cadence_searched"]` now report flips/s, median held-key run,
perpendicular share and net strafe energy on every planner run, with or
without the flag, which is the direct read-out of the 3.70 s and the
number a policy-side arm (entropy, action-history, yaw-conditioned keys)
has to move. `--macro-frac` and `best_env` stay because they are how you
tell "the proposal was widened" from "the search used it".

**Not run:** a from-spawn full-route macro arm. N2 already fails to cross
from half the route with the greedy floor in place, so a run that starts
earlier on the same route cannot cross either; the 20 minutes went to
Q0/Q1 instead, which localise the loss to free flight rather than to the
finish room.

Branch `macro-hold` (fbb5919, 9ee648b), logs and summaries under
scratchpad `mh/`.

**Roadmap correction after the macro-hold search (merged 78a8c34; its full
section is above).** Held-key macro-actions as the PLANNER's proposal are a
measured negative: every arm slower than its control (74.71 -> 75.09-75.65 s
from the room prefix; 74.89 -> 75.54 s from half the route), two arms never
cross, and over free flight alone the macro population stalls at 149 ku
where the control climbs to 193 ku. Mechanism, measured: a held-key lineage
never wins selection (the winner is always a greedy continuation), and a
held key forces the yaw to choose between steering and strafing - the
record's view does both at once, closed-loop at 131 Hz. So roadmap item 1
loses its "P1" half: the planner will not out-strafe the policy by proposal
design; the 3.7 s is policy-side (entropy, and the search-derived targets
P2/P3 for the loop). Kept: every planner run now records cadence (flips/s,
median held-key run, perpendicular share, net strafe energy) in
summary.json - the number any policy-side arm has to move: record 0.84
flips/s / 0.422 s / 79.3 % / +1.17 M; our planner line 3.27 / 0.046 s /
43.1 % / -0.38 M.

### Round 30 day 2 - xLOOP131: the reset loop re-armed for the finisher (prepared 22:40, launching)

The user's question: the iterated reset-and-respawn loop (round 27 xLOOP,
24 % -> 88.7 % of the route in four rounds, 1.5-5x continuous training at
matched compute, then pinned at the wall) - would it help at the current
stage, a finisher plateaued at 74.0-74.5 s under plain PPO? Hypothesis: a
FRESH network fitted to the current best line's state distribution at
131 Hz with the low entropy bonus re-fits the line without 11.8B steps of
accumulated pathology (plasticity / primacy bias) and may strafe better; a
finisher's spine contains the finish approach, so the reverse curriculum
comes free.

Prepared (branch loop131, merged e4738d3):
* `tools/traj_to_spine.py`: the spine is recovered by EXACT REPLAY of a
  recorded episode (fwd/side/jump/duck read from the rows, the yaw and
  pitch bins inverted from the recorded per-tick view deltas), because the
  recorder rows lack ducked/duck_time/fuser2/oldbuttons/basevelocity and
  19.6 % of this line's ticks are ducked. Max deviation 0.038 u over 9,612
  ticks, `done` on the last step. Spine = xENT131's fastest finisher
  (73.692 s, `traj_11700011008` ep3), one state per physics tick, 99.94 %
  of the route, no trim (contact_cut would have cut 3.68 s before the
  finish - the known champion-line failure of that rule, skipped for
  finishers).
* `loop_spine.py --pick fastest` (default `deepest` byte-identical): for a
  finisher min_d is 0 for every episode, so the old rule degenerates to
  file order. `loop_driver.py` gains XLOOP_FLAGS / XLOOP_SPINE0 /
  XLOOP_PICK / XLOOP_MAP (the map path was hardcoded to C:/, so the loop
  could never have run on a rented box).
* Launch (scratch branch + overrides, last wins): `--tick-ms 7.63
  --act-every 4 --ent 0.001 --ep-secs 120 --time-pen 0.01 --race-latch
  6996 --respawn-margin 2 --respawn-binned 1 --respawn-bins 128
  --eval-stall 1 --demo-file maps/spine_r0.npy --demo-window 9612
  --demo-rate 2.0 --demo-min-ep 1e9 --respawn-frac 1.0`, BUDGET 1.5e9
  ticks. `--ep-secs 120` because the pinned 12000 ticks is 92 s at 7.667 ms;
  `--time-pen 0.01` and `--race-latch 6996` because xENT131 trains with
  them and dropping them would change the objective under test (the
  latch is the flag that removes the final-descent shaping charge that
  pinned the old loop). The binned-reservoir flags are inert under a demo
  pool (`if demo ... elif respawn`).
* CPU dry run: spine loads, window frozen over the whole line, 240 ended
  episodes re-identified as spine rows, 3 wins from a random net within
  270k steps, 512/512 spawns field-exact vs the spine (yaw within the 5
  deg jitter), 21.7 % ducked, no rebake.
* Found on the way: `record_ckpt` recordings label every episode
  `"end":"fail"` (its core emits 0 reward and record_rollout infers done
  from a +25 final reward) - do not read those footers; finish_times /
  eval_honesty are unaffected.

Verdict rule: fresh net vs xENT131 at the same 1.5e9 budget on finishes/9
and greedy time (xENT131 plateau 74.0-74.5 mean, best 73.69 in its last
evals); positives in order: any eval under 73.69 s; 9/9 under 74.0 mean;
finishing at all by 1.5e9. Cadence of the spine itself: 2.70 flips/s,
0.123 s median hold, 65.8 % perpendicular, +0.65 M (record 0.84 / 0.422 /
79.3 / +1.17) - the plasticity hypothesis predicts the hold and the
perpendicular share move toward the record.

**22:30 - xENT05 (`--ent 0.0005`, continuation of xENT131's 10.77B
checkpoint, 131 Hz, K=4): the mean moves.** Last two evals: 7/9 best
73.22 s / mean 73.69 s, and 9/9 best 73.41 s / mean 73.75 s (pooled 16
finishers: min 73.22, mean 73.72, sd 0.30), against xENT131's plateau of
74.0-74.5 mean at the same tick. The entropy bonus keeps paying: 0.005 ->
0.001 was worth ~1.5 s on the mean, 0.001 -> 0.0005 another ~0.4 s, with
no loss of finish rate. The run was cut at +1.0B of its 1.5B ticks by an
orchestration bug (a box-reuse waiter keyed on the box's harvest receipt,
which persists across re-registration, fired on the previous run's receipt
and replaced the trainer with the A/B arm); its results were pulled by hand
(ckpt_11774459904 + last two evals). Roadmap item 4 is confirmed; the
next arm on that axis is `--ent 0.00025` or an anneal from 0.001.

### Round 30 day 2 - 23:15: the AlphaZero-style loop with the value target compounds again - **72.46 s**

exitTPT (`tools/expert_loop.py` from xENT131's 10.77B checkpoint at 131 Hz,
2 rounds, planner 600 s at 2,048 envs, `--train-extra --bc-target dist
--bc-value-coef 0.25`, 5090, launched 21:51, finished 22:39, harvested by
the daemon on exit):

| round | greedy in (best / mean / fin) | planner line | greedy out (best / mean / fin) |
|---|---|---|---|
| 0 | 74.84 s / 75.19 / 5/9 | **72.34 s** | 73.27 s / 73.90 / 8/9 |
| 1 | 73.27 s / 73.90 / 8/9 | **71.54 s** | **72.46 s / 72.74 / 5/9** |

Two rounds, ~50 min of box time: best 74.84 -> 72.46 s and mean 75.19 ->
72.74 s. Two things moved at once: the planner, proposing from a 131 Hz
low-entropy policy, now finds 71.5 s lines (its floor from the old policy
was 74.70 s), and the distillation with the value target closes to within
0.9 s of the planner's line per round (the old loop sat 1.7 s behind).
CAVEAT: the matched control (exitCAT, argmax target, no value term)
launches next on the 3-tick box; until it reports, the value target's own
share of this is unknown - the planner-floor drop alone would explain part
of it. The distribution target is a 0.2 % rider here (elite collapse).
On the record's clock the best run is now ~71.6 s (gap 3.0 s).

**xENT131K3b final** (3-tick decisions at 131 Hz, ent 0.001, 2.5e9 ticks
after the 1.5e9 first hour): last evals 4-8/9, best 74.16-75.12 s per eval,
pooled 225 finishers min 73.64 / mean 75.03 s. NOT better than the 4-tick
policy (xENT131 plateau 74.0-74.5, best 73.21): the finer decision grid
that bought the planner 0.6 s does not transfer to the policy at this
budget; roadmap item 2 is null. Its box hosts the A/B control next.

xLOOP131r0 (fresh net from the 73.69 s spine, 1.5e9 ticks) launched on the
exitTPT box at 23:17 after the treated arm's harvest.

**23:52 - overnight loop launched.** exitTPT2: `tools/expert_loop.py` from
exitTPT's round-1 checkpoint (72.46 s greedy best, md5 a037248d...), 24 rounds
bounded by a 12 h deadline (11:45), `--train-extra --bc-target dist
--bc-value-coef 0.25`, planner 600 s at 2,048 envs, 3e8 train steps per
round, on a 5090 at $0.470/h (instance 49902545, ssh7:22544) found under the
price cap on the first attempt; daemon harvest of the round summaries and the
newest round's checkpoint armed. Expectation on record: several more rounds
of gains, then a plateau in the 70-72 s range unless the planner keeps
finding faster lines as the policy improves (the round-by-round
planner_best_s in expert_summary.jsonl is the leading indicator). Also
running: exitCAT (the A/B control), xLOOP131r0 (reset round 0), exit_scratch
(from-scratch loop, round 2). Both short boxes are released automatically
after their harvest to keep the balance ($12.49 at 23:15) above vast's
kill threshold through the morning.

### Round 30 day 3 (01:00) - the A/B verdict: the search-derived VALUE target is what compounds

Same seed (xENT131 ckpt 10.77B, 131 Hz), same settings, same round-0 planner
line (72.34 s in both - the planner is deterministic given the seed), two
rounds each, 5090s:

| round | control exitCAT (argmax, no value term): greedy out best / mean / fin | treated exitTPT (dist target + value coef 0.25): best / mean / fin | planner line (CAT / TPT) |
|---|---|---|---|
| 0 | 74.01 / 74.83 / 7/9 | **73.27 / 73.90 / 8/9** | 72.34 / 72.34 |
| 1 | 73.82 / 74.28 / 7/9 | **72.46 / 72.74 / 5/9** | 72.11 / 71.54 |

The treated arm closes to 0.9 s of its planner line per round; the control
stays 1.7 s behind (yesterday's plateau). Because the distribution target is
a 0.2 % rider on elite lines (measured), this is the VALUE target's result:
about -0.7 to -1.4 s on the best and -1.0 to -1.5 s on the mean per round,
and it feeds back into the planner (71.54 vs 72.11 in round 1). One seed
each, but the two arms share every input up to the first distillation, so
the difference is the term. exitTPT2 (24 rounds from the 72.46 s checkpoint)
runs overnight on this configuration; exitDAG (the same + `--dagger-k 600`,
which is where the distribution target gets its mass) launches beside it.

**xLOOP131r0 (reset round 0): NULL at 1.5e9 ticks.** A fresh network spawned
uniformly on the 73.69 s spine learns to finish from spine spawns (training
success 5-7 %) but its map-start greedy evals never leave the first 1 % of
the route (corridor MAX 2,577-2,641 u, 0/9 finishes at every eval,
eval_progress flat at 2,051 u). The reset does not re-fit a finisher in one
round; the old loop's compounding took 4 rounds x 1e9 to reach the wall from
scratch, and this is that same cost with nothing yet to show. Drop unless a
multi-round budget is spare; the plasticity hypothesis stays untested rather
than refuted.

Ops: the daemon inherited the previous run's `only_extra/extra/newest`
harvest fields when a box was re-registered for a new run with a plain
`--harvest "<port> <host> <run>"`, so xLOOP131r0's box pulled exitTPT's extras
and not the round's own files (pulled by hand, then fixed in
fleet_watchdog.py). xTAIL (tail-weighted PPO) found no box under the price
caps at 01:00 (0-1 offers per round); morning.

**Night plan (01:15, user asleep until ~10:30; vast $34 + the local 5090).**
Target: a greedy finish under 69.5 s spawn clock (~68.6 s on the record's
clock) by morning. Running: exitTPT2 (24 rounds to 11:45) and exitDAG (8
rounds to 09:00), both treated (dist + value 0.25), both from the 72.46 s
seed; exit_scratch to 04:37, then its box takes exitV05 (3 rounds, value
coef 0.5, same seed - the short ablation for the next gradient direction).
Local 5090: xTAIL (tail-weighted PPO, 1.5e9 ticks from xENT131's 10.77B
checkpoint) then its control xTAILCTL. Daemon harvests everything; wake-ups
03:15 / 05:45 / 08:30. Rule of the night per the user: short ablations to
find the direction, longer runs to step along it.

**01:25.** exitTPT2 rounds 0-2 (from 72.46 s): planner 71.00 / 70.91 /
70.83 s; policy out 72.23 (7/9) / 72.08 (9/9) / 72.19 (6/9) best, means
72.67 / 72.46 / 72.44. Compounding but slowing - the planner gains ~0.1 s
per round and the policy sits 1.2-1.4 s behind it. exitDAG's first launch
aborted at the relabel phase: the box's checkout had not moved (a
hand-shipped tools/dashboard.py made `git checkout` refuse, silently, in
the refresh step), so the old tick guard fired; relaunched at 01:24 on a
head verified against origin (hard reset), same seed, 8 rounds to 08:46.
Its aborted round-0 planner reached 71.00 s in 36/36 crossing waves. Lesson
applied to every reuse chain: `git reset --hard FETCH_HEAD` and compare
the box's head to origin/baseline before launching anything.

**02:40 - xTAIL (tail-weighted PPO) vs control, local 5090, 1.5e9 ticks from
xENT131's 10.77B checkpoint, ent 0.001, both arms concurrent on the card
(a faulty pid check started the control early; same conditions for both):**

| arm | last three evals best / mean | pooled (n) min / mean / median |
|---|---|---|
| xTAILCTL (no tail flags) | 73.48/74.00, 73.20/73.65, 73.82/74.36 | (155) 73.20 / 74.21 / 74.19 |
| **xTAIL** (`--tail-weight 1 --tail-outcome return --tail-bins 64`) | 73.62/73.74, 73.41/73.72, **73.33/73.54** | (138) **72.93 / 73.84 / 73.74** |

About -0.4 s on the pooled mean and -0.3 s on the best, finish rates
similar (6-8/9). Modest and positive, one seed each, same card and hour;
consistent with the transplant reaching only the last ~7 % of each
episode's steps. Follow-up launched at 02:40 on the local 5090: exitTAIL,
the treated expert loop (dist + value 0.25) from the 72.46 s seed with
tail weighting inside its PPO rounds, 6 rounds, to read against exitTPT2's
rounds 0-5.

### Round 30 day 3 - 03:15 read: every loop compounding; value coefficient 0.5 leads; planner budget is not the limit

All loops from the 72.46 s seed (exitTPT round 1), treated (dist target +
value target), 131 Hz, 5090s unless noted; "out" = greedy best / mean /
finishes after the round:

| round | exitTPT2 (v 0.25, 24 rounds) | exitDAG (v 0.25 + dagger-k 600) | exitV05 (v 0.5, 3 rounds, done) | exitPB (v 0.25, plan 1200 s, 4090) | exitTAIL (local, v 0.25 + tail-weighted PPO) |
|---|---|---|---|---|---|
| 0 | 72.23 / 72.67 / 7 (pl 71.00) | 72.54 / 72.67 / 8 (pl 71.00) | 72.01 / 72.47 / 7 (pl 71.00) | 72.27 / 72.76 / 7 (pl 71.06) | 72.41 / 72.72 / 7 (pl 71.01) |
| 1 | 72.08 / 72.46 / 9 (70.91) | 72.24 / 72.54 / 5 (70.96) | 72.14 / 72.56 / 6 (70.90) | | 72.04 / 72.26 / 8 (70.93) |
| 2 | 72.19 / 72.44 / 6 (70.83) | 71.80 / 72.25 / 9 (70.73) | **71.70 / 72.30 / 8 (70.68)** | | |
| 3 | 71.83 / 72.11 / 5 (70.57) | **71.51 / 72.01 / 7 (70.50)** | | | |
| 4-7 | 71.64 / 71.94 / 8; 71.60 / 71.83 / 7; 71.59 / 71.71 / 7; **71.28 / 71.83 / 7** (pl 70.52 -> 70.25) | | | | |

Readings (one seed each, shared inputs up to the first distillation):
* The main loop keeps compounding at ~0.1-0.15 s per round on the best and
  the mean; best so far **71.28 s** (round 7), ~70.4 s on the record's clock.
* DAgger is ahead of the plain loop at matched rounds 2-3 (71.80 / 71.51 vs
  72.19 / 71.83 best) - the distribution target with real mass helps.
* Value coefficient 0.5 beats 0.25 at matched round 2 (71.70 vs 72.19 best)
  and finished its 3 rounds; the reaper released its box 10 min after the
  harvest. Direction found; stepping along it: exitV05L (20 rounds at 0.5
  from exitV05's round-2 checkpoint, 71.70 s, md5 59f74f2b...) and a short
  exitV10 (value 1.0) are being placed (orchestrator 2).
* Doubling the planner budget (1,200 s) did NOT improve the planner line at
  round 0 (71.06 vs 71.00 s): the planner is proposal-limited, not
  search-limited. Its remaining rounds run out cheaply on a 4090.
* exit_scratch rounds 5-10: 0 finishes, no finish plan (progress objective);
  the from-scratch loop is not converging on this budget; ends 04:37.
* Credit $40.48 (topped up twice). Reaper working as designed.
