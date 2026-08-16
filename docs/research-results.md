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
| 16x8 (s16_g999) | 128 | running | | | trunk heavily degenerate here (conv3 out 2x1 vs 4x8 pool) - finding the floor |

The perception bar for this map is far lower than assumed: 512 pixels
suffice to reach and fight the 47k wall at full speed.
