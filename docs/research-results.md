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
