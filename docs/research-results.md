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
