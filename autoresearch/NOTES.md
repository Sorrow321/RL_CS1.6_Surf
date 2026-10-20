# Lineage notes - surf_src_sidistic

Append-only, newest at the bottom, a dated heading per entry (time from `date`). This file is
the part of the workspace that carries what a lineage KNOWS from one round to the next. If your
branch wins, the next four forks read what you leave here, so write facts with numbers.

## Round 0 - inherited state (setup session, 2026-10-20)

### The map

* `maps/surf_src_sidistic.bsp`, 7 stages. The WR is 115.99 s by a human; the demo is analysis
  only and NEVER training (CLAUDE.md section 0).
* The goal field at the spawn is d0 = 97,768 u. 1% of progress = 978 u.
* The 2026-08-23 "unreachable" verdict was a voxel artifact (thin panes with holes filled as
  their AABB). It was fixed on 2026-10-18 (`vision.pane_tag`) and the caches were re-baked.
  The finish box is `zones.json` end, z 176..184.
* Checkpoint curtains sgcp_1..7 are in the bsp. `autoresearch/score.py` reports the furthest
  one entered (CP).

### The gates seen so far

Greedy eval from the true start, best episode, % of d0:

1. **5.6-5.7%: the first-ramp side branch.**
   * Every run so far parks here for 200M-1.1B steps.
   * The policy rides a side branch off the first ramp. The way on is a turn onto the ramp
     itself ("it just has to turn left", the user).
   * Exits, all coinciding with the unstuck temperature T rising past ~1.3-2: 529M
     (compass + pot-rate), 755M (novel respawn), 1,133M (surf flag + bonus), 1,510M (cap-2
     base).
2. **20.8%: the wall hole ("the window").**
   * A ~110x130 u hole in a wall at x ~1,150, z 13,198..13,314, 350-430 u above the agent's
     line, with a fail plate on its sill.
   * The human climbs to it by cutting back into an A-frame ramp before it ends.
   * Passed (CP1, 22.9%) at 755M by compass + pot-rate, 1,359M by surf, 1,661M by cap-2.
     The novel-respawn run stalled there from ~1.0B to 1.41B.
3. **27.4-27.5%: the CP2 wall. Never crossed.**
   * The agent dives (|v| ~2,000, vz ~-1,370) into a fail at x ~3,050, z ~11,200 at ~25.4 s.
     It surfs only 3-8% of its last 3 s.
   * The human passes the same spot ~250 u higher and level (vz -141) at 27.4 s, after turning
     EAST onto a ramp at t 24.6-26.8 s.
   * That turn is a 1,958 u dip in the field (2.0 reward units at 100/d0): the field charges
     for the detour and pays for the dive.
   * The surf run spent 1.5B steps at T = 2 here without one crossing.
   * `--obs-pot-rate` reads -1.7..-1.9 during the human's turn and +1.5..+1.7 along the
     agent's dive.
* Along the human line the field has 7 such dips of 1.4-3.0 reward units, so expect more
  walls of this kind past CP2.

### Runs on this map

All local RTX 5090, 1024 envs, 400-470k fps, from scratch, one seed each.

| run | recipe | trap exit | window | best | stopped |
|---|---|---|---|---|---|
| sidEM2cap2 | base: --unstuck --unstuck-reach eval --respawn-margin 2 --unstuck-max 2 | 1,510M | 1,661M | 27.4% | 2.9B |
| sidEM2cap2surf | base + --obs-surf 1 --surf-bonus 0.3 | 1,133M | 1,359M | 27.5% (CP2 wall from ~1.8B to 4.0B) | 4.0B, done |
| sidEM2cap2surfGR | surf + --obs-compass 1 --obs-pot-rate 1 | **529M** | **755M** | 27.2% at 755M | 758M, by the user |
| sidEM2cap2surfNV2 | surf + --respawn-novel | 755M | not by 1.41B (stuck at 20.8%) | 20.9% | 1.41B, session ended |

Recipe flags are listed as "base" or "surf" plus the extras shown.

* Every one of them also carries the scratch preset: --view-continuous --view-absolute velocity,
  --keys-hold, --obs-potential norm + curtain, --obs-normal 1, --simba 1, --ep-ticks 30000
  (300 s), --respawn-frac 0.9, --int-coef 0.25 --int-view 8 --int-speed 3, --act-every 4,
  gamma 0.9995, --n-steps 128.
* T (the unstuck temperature) is the run-state that matters most for the trap exits. It rises
  0.5 per 1e8 steps after 2e8 steps without a 500 u eval-frontier gain, and is capped at 2.
* The compass + pot-rate lead is ONE seed. The early-mark agreement between identical configs
  was measured at ~0.4% (CLAUDE.md), so a 600M-step lead at two marks is large, but it is
  still one draw.

### The incumbent handed to round 1

* Recipe: `config.incumbent_recipe` (sidEM2cap2surfGR's flags). `box.py launch` prepends it.
* Checkpoint: `$AR_HOME/incumbent/ckpt.pt`, sidEM2cap2surfGR at 749.7M steps. That is 6M
  steps before its window-passing eval (755M: 27.2%, CP1, 7/9 through the hole). Resume it
  with `box.py deploy --ckpt` + `box.py launch --resume`.
* Matched-step references:
  * `$AR_HOME/incumbent/r0/progress.csv` (GR, to 758M);
  * `$AR_HOME/incumbent/ref_surf/progress.csv` (the surf run, to 4.0B, with the CP2 wall).
* Score to beat: 27.5 (the best eval anyone has, surf at 3,775M). GR's best is 27.22.

### Mechanisms already in the trainer

The docs say how each one works and what it did.

* `--unstuck` (docs/unstuck.md);
* `--respawn-frontier` / `--respawn-backward` / `--respawn-random`
  (docs/respawn_frontier.md, respawn_backward.md, respawn_random.md);
* `--respawn-novel` (docs/respawn_novel.md);
* `--obs-compass` / `--obs-pot-rate` (docs/obs_potential.md);
* `--obs-surf` / `--surf-bonus`;
* `--race-ratchet` (docs/race_ratchet.md);
* `--goal-planner` and the ramp search (docs/planner-design.md, docs/ramp-search-design.md,
  paused by the user on 10-18);
* the archive / go-explore family (docs/edge_archive.md).

`docs/research-results.md` is the full ledger; search it before re-running an idea.

### Throughput off the 5090

The 3090 is the reference card in CLAUDE.md (~0.75-0.9e9 steps/h on the older, smaller
config). With SimBa + normals + the potential channel at 1024 envs, expect roughly 150-250k
steps/s. Measure it on your box in the first minute and size `--steps` to the time.

## Round 1, r1b4 - 2026-10-20 01:15 UTC: the CP2 wall's anatomy, and two exploration arms

Box: vast 56816052, RTX 3090 at 0.187 $/h (machine 143360, 32 threads). The trainer is
GPU-bound there (99%), at about 150k steps/s on the incumbent recipe, so 30 min is about
0.27B steps. Offers were empty under the caps from 00:45 to 01:00 UTC; a 3090 at 0.187
appeared at 01:01.

### What the CP2 wall is (ANALYSIS ONLY: the agent's own evals, plus the core's trigger test)

Tools: scratchpad `failprobe.py` / `failtop.py` place a still player hull on a 32 u grid with
`--teleport-fail` on, step one tick and paint where it dies. The fail-teleport AABBs are
misleading: the agent's surviving lines pass through `*99`'s box.

* **The agent's line** (incumbent 755M and surf 3,775M evals, both lanes; table rows are
  the agent's state at that time):

  | t (s) | x | z | what happens |
  |---|---|---|---|
  | 21.3-22.4 | 1,750 -> 3,000 | low point 11,650-11,740 at x 2,300-2,500 | surfs the U-shaped ramp east (contact every tick) |
  | 22.4-22.5 | ~3,000 | 11,900 | leaves the ramp lip at vz +800..+860, \|vh\| 1,250-1,450 |
  | 22.5-23.5 | to 4,500-4,640 | apex 12,360-12,380 | pure ballistic flight from here (vz falls exactly 8 u/s per tick, no contact); air-strafes a 180 deg U-turn inside a room bounded at x ~3,400 / ~4,650 |
  | 25.2-25.4 | ~3,000-3,100 | ~11,200 | dives west at vz -1,000..-1,370 into the slanted fail sheet; dead |

* **The open way** (free space between fail volumes, from the probe):
  * A lower room at z ~11,000-11,850, x 3,000-5,800, under the fail plates `*84` (z 12,256)
    and `*45` (z 11,904, x 4,194-5,344).
  * At its east end, a D-shaped curved wall (z ~11,000-11,400, x 4,000-5,800) over the fail
    floor at z ~10,800.
  * A horizontal SLOT back west at z ~11,350-11,550 (x 2,600-3,600), between the slanted
    lower fail sheet and an upper fail sheet. Then openings in a thin wall at x ~2,450.
  * This matches the human description in round 0: east to (5,554, 454, 11,079) at 25.5 s,
    surfing, then back west level at z 11,469.
* **So the missing behaviour is a chain:**
  1. leave the U-ramp LOW (vz about +300, not +850), so as to pass under `*45`;
  2. do NOT air-turn: fly east about 1.6 s to the D-wall;
  3. surf the D-wall round;
  4. come back west level through the slot.
* **Field values** (goal_32): the agent's death 70,910 (27.47%); D-wall apex 73,914
  (24.40%); slot middle 70,859; wall gap at x 2,400 70,296 (28.1%); CP2 67,558 (30.9%).
  Passing the slot is worth about +3.5% at once.

### Won't or can't (CPU record_ckpt, incumbent checkpoint, `--spawn-states`)

* 10 hand-built lower-room starts heading east (x 3,800-5,200, z 11,300-11,750, 1,400-1,550 u/s),
  14 greedy episodes. Every one air-turns back west within ~1.2 s and dies in the pit, at
  26.3-27.4%.
* Only the start nearest the D-wall (5,200, -300, 11,300) touches it: 37.7% surf contact,
  survives 3.1 s, then dies on the fail floor.
* **So the route is not in the policy's repertoire.** It has to be PRACTISED, and the field
  charges for each of steps 2-3 (the dive pays about +1.5 units before it dies, while the
  D-wall leg costs about -1.9).
* The incumbent reservoir (20k states) holds 0 states in the lower room. 15 states sit in
  x 3,200-4,700, all of them on the upper flight (z 12,000-12,400).
* CPU record_ckpt works on this host: about 2.3 min for a 25 s greedy episode.

### Arm A: r1b4_nv = warm incumbent + `--respawn-novel` (defaults), launched 01:08 UTC

* Why: an offline probe of NovelSpawnSampler, run on the incumbent's reservoir with a proxy
  visit grid (reservoir plus eval lines, +-96 u jitter), puts about 12% of its novel rows in
  the lower room and 1.5% in the slot band, with velocities copied from the policy's own
  states. Those are exactly the starts needed to practise steps 2-4, as a backward chain.
* r1b1 runs `--death-charge 1.0` warm at the same time, so this arm does not repeat it.

### New code: `--spawn-burst` under `--view-continuous` + `--unstuck` (commit 10d6cf6)

* Go-Explore's explore-after-return: random HELD keys plus a random held view command
  after a respawn, out of the PPO pool via b_ez.
* `--spawn-burst-frac P`: the share of respawns that get a burst. `--spawn-burst-uniform`:
  burst length U{1..K}.
* Flag off is bit-identical to the parent trainer (`tests/python/test_spawn_burst_viewc.py`,
  run on sidistic because cannonball's goal field is not baked on this host).
* This is the physically-consistent counterpart of `--respawn-novel`: bursts reach the
  lower room through real physics (a coast from the U-ramp at vz ~+300 lands on the D-wall
  at z ~11,150), and those states can be harvested.

### 01:45 UTC: Arm A result, two more probes, and Go-Explore from the frontier

* **Arm A, r1b4_nv** (warm incumbent + `--respawn-novel`, 0.30B steps, 01:08-01:37):
  * evals 27.22 / 27.35 / **27.44** / 27.39 / 27.39 (750M-1,052M);
  * flat at the wall, CP2 never crossed;
  * 41-47% of episodes started from novel rows; T 0.08-0.2 (still inside patience).
  * Stopped by exact pid at 1.05B. Best score 27.437 (901M).
* **Probe 2, D-ramp exits** (CPU, incumbent): 12 starts at the Go-Explore archive's own
  west-heading exits from the D-ramp (z 11,280-11,650, 400-1,000 u/s). All die on the fail
  floor within 1.2-2.3 s. They are too slow to carry to the slot; the human leaves the
  D-ramp at about 1,800 u/s.
* **Probe 3, the one D-ramp contact** (lower14 ep 2, start (5,200, -300, 11,300) heading
  east at 1,471 u/s): the policy hits the D-ramp HEAD-ON at 0.3 s. Speed falls 1,552 -> 662
  u/s in 0.1 s; it slides west at 500-800 u/s and falls at x ~4,000.
* **So the missing skill is a tangential, speed-keeping ride round the curved D-ramp.**
  Random actions cannot do it either: they lose speed.
* **Go-Explore phase 1 from the frontier** (`tools/explore_phase1.py --roots-spine`, roots =
  463 states from the incumbent's own reservoir between the CP1 hole and the CP2 wall; cell
  128 u x speed 4; 5 CPU-min here, 0.22M ticks/s, 862k runs):
  * 13,340 cells: 6,236 in the lower room, 2,182 on the D-ramp, 604 of those heading west;
  * best field d 70,731 (27.65%), 42 cells past the old frontier, none below 70,500;
  * the slot is not threaded.
  * Its own "best" printout is a different distance; read archives with the goal field
    (scratchpad `gechain.py`).
  * This is the cheapest map-free instrument found so far for asking "what is reachable
    from where the policy dies".
* **Arm B, r1b4_sb** (warm incumbent + `--spawn-burst 50 --spawn-burst-p 0.98
  --spawn-burst-uniform --spawn-burst-deep 0.03`), launched 01:38:
  * 0.2% of decisions are burst rows: about 3.7 frontier bursts/s;
  * deep threshold d <= 75-76k (the CP1 -> CP2 stretch).
* **Go-Explore on the box's spare CPU:** a 20-min run started 01:42 (`runs/ge2`, cell 128 x
  speed 8 x view 8, 8 threads, nice 10) while Arm B trains on the GPU.

### 01:55 UTC: Go-Explore at scale, the trigger calibration, a negative generator

* **Do not run CPU-heavy tools on the training box.**
  * `tools/explore_phase1.py` with 8 threads on the 32-thread 3090 box cut Arm B from
    ~185k to ~31k steps/s. It was killed after 1.5 min.
  * A background job there ignores SIGINT (bash sets it for background jobs), so kill it
    by exact pid with SIGTERM. The snapshot (`--snapshot-every`) is what survives.
* **The box archive in those 1.5 min** (0.94M ticks/s, 1.4M runs, cell 128 x speed 8 x
  view 8):
  * 178,742 cells: 85k in the lower room, 32k on the D-ramp, 9.6k of those heading west;
  * best d 70,755, nothing below 70,500;
  * fast west-heading lower-room states exist (6.5k above 1,300 u/s) but sit low (median z
    11,219) and descend (median vz -489);
  * only 16 are near a slot approach, and those are too slow.
* **Trigger calibration** (scratchpad `calib.py`):
  * a STILL hull at the agent's death point (3,027, -937, 11,224) does not fail, but the
    same point moving at (-1,400, 0, -1,300) fails on its first tick. The slanted sheet is
    within about 14 u, so the 32 u probe maps are right to within a cell;
  * a westward flight at z 11,450 from x 3,500 to 2,500 is free.
  * Free gap: at x 2,600, z ~11,310-11,620; at x 3,000, ~11,200-11,780.
  * **To pass, arrive at x ~3,600 with vz >= ~0 at z 11,450-11,650 and >= ~1,400 u/s
    horizontal.** No ballistic air U-turn can do that (its apex would have to lie below
    the launch height), so a surface has to redirect the fall, which is the D-ramp.
* **NEGATIVE: "physics-novel" spawns by random held bursts** (scratchpad `physnovel.py`):
  * 600 sources (deepest 3% of the reservoir) x 48 bursts;
  * 13% alive at the burst end; 501 kept as alive + novel + under the ceiling;
  * **median speed 279 u/s.** Random actions that end alive and novel are stalled states:
    useless as approach states.
* **What survives that filter is Go-Explore's archive**, because it keeps the FIRST state
  that reaches a cell, including fast in-transit ones. Selecting
  `novel & d >= d_best & speed >= the policy's own frontier median (746 u/s)` from the box
  archive gives 67,061 states: 43k in the lower room, 12.9k heading east, 16k west
  (scratchpad `gefast.py` -> `ge2_fastnovel.npy`, md5 3d75f45f...).
  Provenance: roots = the incumbent's own reservoir states; random macro search in the
  simulator; no demo; no map box (field d, the policy's visits and its speed are the only
  inputs).

### 02:05 UTC: Arm B result; Arm C launched

* **Arm B, r1b4_sb** (warm + `--spawn-burst 50 --spawn-burst-p 0.98 --spawn-burst-uniform
  --spawn-burst-deep 0.03`, 01:38-02:01, 0.23B steps):
  * evals 26.79 / 27.40 / 27.39 / **27.49** (750M-976M);
  * reservoir min-depth 74.4% throughout, so no training episode passed the slot;
  * stopped by exact pid. Best score 27.489 (976M), this branch's best so far.
* **Probe 4** (CPU, incumbent): 14 starts at the Go-Explore archive's fast east-heading
  lower-room states near the D-ramp (x 4,550-4,970, z 11,066-11,340, vx 1,127-1,455).
  * The policy touches the ramp (contact up to 62%), but every episode loses its
    horizontal speed and falls to the fail floor (z ~10,790) within 0.4-3.3 s.
  * The D-ramp ride is a skill to PRACTISE, not a decision to flip.
* **Arm C, r1b4_gs**, launched 02:02 by box.py ssh with `SELF_STATES=1` (box.py launch
  cannot pass it):
  * warm incumbent + `--spawn-states runs/ge2_fastnovel.npy --spawn-states-frac 0.3`;
  * that is 30% of spawns from the 67k fast novel states of the policy-rooted Go-Explore
    archive (provenance above, no demo).
  * Watch the reservoir min-depth: it falls below 72.5% only when a training episode
    passes the slot and lives 2 s more. Then the eval frontier.

### 02:30 UTC: Arm C at 1.03B (25 min) - first sign of learning in the D-ramp region; extended

* **Evals** (frontier u / score): 25,009 / 25.58, 26,737 / 27.35, 26,768 / 27.38,
  26,800 / 27.41 (750M-976M). Still at the wall.
* **Reservoir:** min-depth 74.28% -> 73.92%. The checkpoint's 20k-state sample holds 125
  lower-room states (98 heading west, 0 east); the deepest is d 72,432. No training episode
  has passed the slot.
* **Probe 4 rerun on Arm C's 1,027.6M checkpoint** (same 14 fast east-heading starts near
  the D-ramp):

  | | incumbent | Arm C |
  |---|---|---|
  | episodes that come back west | 0/14 | 6/14 (after 15-29% ramp contact) |
  | best d reached | 72,134 | 71,428-71,850 (x ~3,470-3,780) |
  | how it ends | fail floor | still the fail floor |

  * Arm C arrives too low (z ~11,000; it needs ~11,450) and too slow (~900 u/s).
  * Mean d_min over the 14: 72,670 (incumbent) -> 72,290 (Arm C).
* **Decision:** extend to the full 1e9-step budget (stops at 1.75B, ~03:30 UTC).
  * The criterion is a sub-skill that is moving, not a gate yet.
  * T is starting to rise (patience used up at 1.03B), so exploration increases.
  * Box deadline extended to 04:29 UTC.
* **Ops:** `box.py pull --ckpt` truncated twice today (r1b2 and r1b4): the trainer rewrites
  ckpt_latest.pt mid-copy. `cp` it to a snapshot on the box first, then scp the snapshot and
  check its md5.

### 03:05 UTC: Arm C at 1.39B (T 1.88) - evals flat, but the D-ramp sub-skill keeps improving

* **Evals** 1,052M-1,354M, scores 27.44 / 27.45 / **27.48** / 27.47 / 27.41:
  * frontier 26,798-26,868 u; no CP2;
  * the eval median fell to 20.6% at 1,354M as T rose to 1.88 (the high-T oscillation the
    round-0 lineage saw).
* **Reservoir** min-depth 73.79%: no training episode passed the slot.
* **Probe 4 again** (same 14 fast east-heading D-ramp starts):

  | checkpoint | comes back west | mean survival | mean d_min |
  |---|---|---|---|
  | incumbent | 0/14 | 1.5 s | 72,670 |
  | Arm C 1.03B | 6/14 | 1.5 s | 72,290 |
  | Arm C 1.39B | 11/14 | 2.2 s (max 4.2 s) | 71,900 |

  * All 14 still die on the fail floor at x 3,500-4,300, z ~10,800-11,000: at the slot's
    east end, about 450 u too low.
  * The gradient is real: a better ride gets further west before dying, which pays field
    progress, and that leads to the slot.
* **Other branches** (status.md):
  * r1b2 has 27.90 from `--noise-tau 0.5` (AR(1)-correlated sampling noise). Its CP2 death
    point moved 500 u west and 190 u up, near the slot's west end, without crossing.
  * r1b1 and r1b3 are at 27.44-27.46.
* **Plan:** when Arm C ends (1.75B, ~03:28) resume it from ckpt_final with the same flags
  (Arm C2, +1e9). Promotion criterion: the probe's mean d_min keeps falling, or the
  reservoir's min-depth moves below 72.5% (a slot pass in training).

### 03:16 UTC: Go-Explore rooted at Arm C's own states, a higher-dive variant, the high-T wobble

* **GE3** (local, 6 CPU-min, roots = Arm C's deepest 3% of its reservoir, cell 128 x speed 8
  x view 8):
  * 182,934 cells; best d **70,483 (27.91%)**, deeper than the incumbent-rooted archives
    (70,731-70,755); 1,956 cells past the old frontier.
  * The best chain is NOT the D-ramp route. It is a HIGHER DIVE: from the upper room it
    passes x 3,000 at z ~11,600 (the agent's line is at ~11,224 there), reaches (2,653,
    -883, 11,330) at vz -988, and dies on the lower sheet at the wedge's narrow end, 20-50 u
    too low.
  * This matches r1b2's noise-tau result (27.90: the death point 500 u west and 190 u up).
* **The geometry past the slot** (core probe, x 400-4,000):
  * a free corridor at z ~10,900-11,550 for x 1,500-2,400, under the `*75` plate (z 11,552),
    then a descent to CP2 (x 768-816);
  * fail plates on the corridor floor (z ~10,690-10,960) and the slanted `*70`.
  * A steep dive through the slot would have to clear those as well.
* **Physics of the higher dive:** in free flight only horizontal speed is controllable.
  From (3,039, 11,598) with vz -740, clearing z 11,350 at x 2,600 needs ~1,500 u/s; the
  archive state has 1,230.
* **GE4** (focused search, 3,000 roots at d 70,483-70,978, cell 64, 2 s bursts, 3 min,
  1.77M runs): EVERY run died. Nothing below 70,500. Random actions cannot clear the narrow
  end; stopped.
* **Arm C's high-T wobble:** at T = 2 the greedy line oscillates. Evals 1,429M / 1,505M /
  1,580M = 26,802 / **15,692** / 26,871 u frontier.
* **Next, Arm C2:**
  * resume Arm C's final checkpoint (1.75B) with `--unstuck-max 1` (on resume T is clamped
    to the new cap at the next stuck update) and `--spawn-states runs/ge23_fastnovel.npy
    --spawn-states-frac 0.4`;
  * that file is GE2 + GE3, 139,622 states, all d >= 70,883: no spawn past the policy's
    own best, the respawn-novel rule; md5 b807ba4f...;
  * it includes the higher-dive approach states up to the frontier.

### 03:30 UTC: Arm C final (1.75B); Arm C2 launched

* **Arm C, r1b4_gs**, 02:02-03:26, 1.0B steps at 198k steps/s average:
  * evals 1,429M-1,731M scored 27.41 / **16.05** / 27.48 / 27.47 / 27.48;
  * the frontier never left 26,8xx u; no CP2 crossing;
  * the policy's own view sigma fell to 0.028 / 0.007 (the noise collapse r1b2 also saw).
  * Its best score is 27.484, under Arm B's 27.489.
* **Arm C2, r1b4_gs2**, launched 03:28 via box.py ssh + `SELF_STATES=1`:
  * resume of `runs/r1b4_gs/ckpt_final.pt` with `--spawn-states runs/ge23_fastnovel.npy
    --spawn-states-frac 0.4 --unstuck-max 1`;
  * T clamped 2.0 -> 1.00 at the first update; +1e9 steps, so it stops at 2.75B, about
    04:55;
  * box deadline extended to 05:29 UTC.

### 04:03 UTC: Arm C2 at 2.08B - best score 27.523; the D-ramp ride still improving; an outer loop

* **Evals** 1,750M-2,052M scored 27.41 / 27.43 / 27.47 / **27.52** / 27.49. The 1,977M eval
  (frontier 26,909 u) is this branch's best: 27.523. No CP2. Reservoir min-depth 73.85%;
  T 1.00, as capped.
* **Probe 4 on C2's 2.09B snapshot** (md5 a6594840...):

  | checkpoint | comes back west | mean survival | mean d_min | best d_min |
  |---|---|---|---|---|
  | Arm C 1.39B | 11/14 | 2.2 s | 71,900 | - |
  | Arm C2 2.09B | 13/14 | 2.8 s | 71,790 | 71,340 at (3,308, 537, 11,099) |

  * Its exits now arrive ~100-400 u below the slot's lower edge at x 3,300-3,900.
  * Steady but slowing: ~100-400 u of d_min per 0.7B steps.
* **Method from here: alternate the two halves of policy-guided Go-Explore.**
  * (a) Grow the archive from the policy's NEWEST deepest reservoir states (its improved
    D-ramp exits are better starting points than random ones).
  * (b) Train with spawns from that archive, filtered to d >= the policy's own best
    (never a forward teleport).
  * The trainer re-reads `--spawn-states` when the file's mtime changes (every 20
    iterations; designed for an outer loop), so (b) is refreshed in place by an atomic
    `mv` on the box, without a restart.

### 04:14 UTC: outer loop, iteration 1 - archive refreshed from C2's own states, swapped in live

* **GE5** (local, 8 CPU-min, roots = Arm C2's deepest 3%, d <= 73,549; cell 128 x speed 8 x
  view 8):
  * 156,714 cells; best d 70,461 (27.93%);
  * **767 slot-approach candidates** (west-heading, near-level, z 11,350-11,650,
    x 3,300-4,300; the incumbent-rooted archive had 16), because the roots are now the
    policy's own improved D-ramp exits;
  * still nothing below 70,000.
* **The swap:** fast-novel filter (d >= 70,883, novel vs the incumbent's visits,
  >= 746 u/s) gives 57,000 states; merged with the previous file, 196,622 states, md5
  88c04377....
  * On the box it was `mv`'d over `runs/ge23_fastnovel.npy`; the original is kept as
    `ge23_fastnovel.orig.npy`.
  * The trainer logged `--spawn-states: re-read 196,622 own states ... at step
    2,284,322,816`.
  * NOTE: from that step on, r1b4_gs2's run.json names the old path, but the content is
    the merged file.
* **Evals since the last entry:** 2,128M frontier 26,907 u; 2,203M frontier 26,886 u
  (about 27.50-27.52). Still the wall.

### 04:45 UTC: the slot from the archive's best approaches - still too low; aiming C3 at the higher dive

* **C2 evals** 2,354M-2,581M: frontier 26,844-26,890 u, flat. Reservoir min-depth 73.85%.
* **Probe 5** (C2's 2.6B snapshot, md5 39469b96...):
  * 14 starts at GE5's fastest slot-approach states: z 11,351-11,377, heading west at
    ~1,229 u/s, vz -214..+89, x 3,550-4,060.
  * All die in 0.4-0.8 s on the lower sheet at x ~3,050-3,180, z ~11,140-11,200 (d_min
    70,953-71,121).
  * They are ~250 u too low and too slow. The human crosses x 2,882 at z 11,469, level at
    1,836 u/s, having left the D-ramp at ~1,800 u/s while climbing (vz +260..+422).
  * The policy's own D-ramp exits run ~900 u/s and ~500 u lower. That speed-and-height gap
    will not close this round at the rate it is closing now.
* **C3 aims at the reachable gain instead: the HIGHER DIVE** (GE3/GE5 best chains, 27.91-27.93%;
  r1b2's 27.90).
  * Spawn file `c3_spawns.npy` (md5 7a1bcdba...): the 196,622 broad states plus the chains
    to the 300 deepest cells of GE3 and GE5, kept at d >= 70,883 (1,041 approach states,
    weighted x30, 13.7% of the file).
  * The selection is generic: approaches to the deepest points the policy-rooted search
    reached, never past the policy's own best.
  * Swapped into C2 live at step 2,640,838,656. C3 resumes C2's final checkpoint with the
    same file.

### 04:56 UTC: C2 final (2.75B), best score 27.544; C3 launched (the higher-dive spawns)

* **Arm C2, r1b4_gs2**, 03:28-04:52, 1.0B steps at 198k steps/s:
  * 14 evals between 26,822 and 26,909 u frontier;
  * the best score is **27.544 (2,279M)**, this branch's best, scored the orchestrator's
    way (`best_in` over every run dir): 37 valid evals;
  * no CP2. ckpt_final md5 77a42ab2... was pulled.
* **Arm C3, r1b4_gs3**, launched 04:53:
  * resume of C2's ckpt_final with `--spawn-states runs/c3_spawns.npy --spawn-states-frac
    0.4 --unstuck-max 1`;
  * +1e9 steps, so it stops at 3.75B, about 06:20;
  * box deadline 06:53.

### 05:10 UTC: outer loop, iteration 2 - archive from C2's final states, swapped into C3

* **GE6** (local, 8 CPU-min, roots = C2-final's deepest 3%, d <= 73,524; 1.45M runs):
  * 188,092 cells; best d **70,431 (27.96%)**; 2,358 cells past the old frontier, 81 below
    70,500; none below 70,000.
  * Each iteration rooted at the improved policy reaches slightly deeper: 70,755 / 70,483 /
    70,461 / 70,431 (GE2 incumbent-rooted, GE3 Arm C, GE5 C2 at 2.1B, GE6 C2 final).
* **The swap:** `tools/ge_spawns.py spawns` over GE2+GE3+GE5+GE6, with the incumbent's
  visits / speed / d_best as the reference and chains to each archive's 300 deepest cells
  x30. That gives 336,807 rows (19% deep-chain approaches), md5 d3ea144b....
  * `mv`'d over C3's `runs/c3_spawns.npy`; the original is kept as `c3_spawns.orig.npy`.
  * Re-read at step 2,917,662,720.
* **C3 evals:** 2,751M / 2,826M / 2,902M = 26,841 / 26,903 / 26,879 u frontier. No change in
  the greedy dive yet.

### 05:30 UTC: C3 stopped (flat); fast entries are rare; C4 = C3 + self-imitation (SIL)

* **Arm C3, r1b4_gs3**, 04:53-05:27, 2.75B -> 3.14B:
  * the evals' frontier stayed at 26,841-26,903 u, all 9 episodes at the wall each time;
  * best score 27.517 (2,826M);
  * the higher-dive approach states (deep chains, live since 2.64B and 19% of the file since
    2.92B) did not move the greedy dive in ~500M steps.
  * Stopped by exact pid at 3,141.5M; the latest checkpoint was snapshotted on the box as
    `runs/gs3_last.pt`.
* **Fast lower-room entries are rare** (CPU, scratchpad `entries.py`):
  * random held bursts from C2-final's own CP2-region reservoir states (200 sources x 32);
  * only 6 of 6,400 enter the lower room heading east above 900 u/s, all at x ~5,000,
    1,150-1,255 u/s.
  * So bursts would not supply the fast, policy-speed D-ramp entries either.
* **SIL smoke** (CPU, sidistic; continuous view + keys-hold + potential + unstuck +
  spawn-states + `--sil-coef 0.1`, with SELF_STATES=1 for the trainer's own section-0
  guard): runs.
* **Arm C4, r1b4_gs4**, launched 05:27:
  * resume of `runs/gs3_last.pt` with the same spawn file (the c4 content), `--unstuck-max 1`
    and **`--sil-coef 0.1`**.
  * Why: SIL learns from transitions whose return beat the critic. The best D-ramp
    attempts (further west before dying) and the higher dives are exactly the near-misses
    whose on-policy advantage is a thin gradient.
  * +1.5e9 steps, so it stops at 4.64B, about 07:35. Box deadline 07:58.

### 06:00 UTC: CP2 IS PASSABLE WITH THE RATCHET (r1b1 and r1b2, public status.md); C4 stopped; C5 = my ramp-practised lineage + `--race-ratchet`

* **Other branches' public results** (status.md / RESULT.json, read 05:56):
  * **r1b1, 28.553**: FROM SCRATCH, incumbent recipe + `--respawn-novel` + `--race-ratchet`.
    At 1,633M its greedy line flies the human's CP2 line (drop under the plate, east to the
    +y ramp, turnaround, climbing exit, level slot at z ~11,490).
  * **r1b2, 28.492**: warm, `--race-ratchet` + correlated noise on 25% of envs + unstuck T
    heated to 2. Its greedy policy flies the same detour from 1.20B.
  * The common factor is **`--race-ratchet`**: only new progress records pay, so the D-ramp
    detour (1,958 u uphill in the field) is no longer charged step by step.
* **My mistake, for the record:** I discounted the ratchet. I argued that a failed detour
  still nets less than the dive, which is true in total return. But under the stock
  potential every eastward step has a NEGATIVE reward, so GAE drives the eastward actions'
  advantages negative and the policy is pushed away from the detour before it can ever
  succeed. My Arms C-C4 practised the D-ramp under that charge for ~2.4B steps and only
  crept closer.
* **Arm C4, r1b4_gs4** (SIL 0.1), 05:27-05:56, 3.14B -> 3.47B:
  * evals at frontier 26,851-26,904 u, flat;
  * stopped by exact pid; snapshot `runs/gs4_last.pt`.
* **Arm C5, r1b4_gr5**, launched 05:57:
  * resume of `runs/gs4_last.pt` (3.47B) with `--race-ratchet` (record-gap column 27 zero-
    widened into 6 tensors, function-identical at step 0), the same Go-Explore spawn file at
    0.4, `--unstuck-max 2` and `--sil-coef 0`;
  * +2e9 steps, so it stops at 5.47B, about 08:50;
  * box deadline 08:58.
  * The bet: a policy that already practised the D-ramp ride for 2.4B steps, now with the
    detour uncharged, passes the slot and goes further than 28.55%.
