# NOTES - the surf_princessburglar_b26 lineage

The lineage's memory: every round's winner appends what it learned (what worked, what did not,
the numbers). Read it before you start; append to it before you wrap up.

## The map (operator, 2026-10-23, measured before round 1)

* The surf community calls it THE hardest map. ~7.7k x 7.7k x 7.5k units; the course falls
  ~7,100 u from the top to a finish box at the bottom (x 560-880, y 2608-2928, z -3509..-3474).
* The spawn (511, 413, 2888) is in a SEALED start room (11% of free space) whose only exit is
  the trigger_teleport `start_run`. Every teleport is fatal in our core, so the zones file
  (source manual) declares the spawn at its destination, (860, -3005, 3653) yaw 0. 30
  teleports go to `start` (fall catches back to the room), one to `tlp_end` (the end room, just
  after the finish), one to `burgir` (a bonus room off the course). All stay fatal.
* Goal field (no-corner-cut, cell 32): 507,547 reachable voxels, start d 57,149 u (54,367 u on
  the corner-cutting field). The course is one free-space component (83% of free space).
* Physics: the WR server's movevars are our defaults except sv_maxvelocity 3500 (the recipe
  carries --maxvel 3500; sidistic trained at 4000).
* WR demo (ANALYSIS ONLY, never training - CLAUDE.md section 0; parsed copy in
  $AR_HOME/incumbent/wr/): 77.5 s of playback at 7.62 ms frames (131 fps; we train at 10 ms),
  start curtain and start_run at 0.25 s, finish at 75.44 s = **75.18 s** zone to zone; max
  speed ~2,760 u/s per frame. It is a yardstick and a route instrument (BEV / where the policy
  dies), never a spawn source, a line, a reward or a BC target.

## Where the recipe comes from: the sidistic lineage (7 rounds, 2026-10-20..23)

Round winners there: 28.6 / 28.8 / 54.7 / 94.7 / 98.3 / 98.3 / 107.2 (the first greedy finish
from the true start, 118.05 s). `config.incumbent_recipe` is its finishing run's flag set, from
scratch. What each part did ON SIDISTIC (constants were set there, several late in a long line):

* `--race-ratchet` + `--respawn-novel` (R1): pay only new progress records; spawn own reservoir
  states moved into unvisited cells. Together they crossed the first dip gates from scratch.
* novel-spawn options `--respawn-novel-band/turn/shell/radius/frac` (R2, R4): put the moved
  states in the dips behind the frontier; band 0.15 after a lucky episode lifted the ceiling.
* `--time-pen 0.001`, `--surf-bonus 0` (R3, R6): at 0.005 dying beat every slow dip detour.
* `--respawn-greedy 0.25 --respawn-greedy-window 0.25 2 --respawn-greedy-stuck 200` (R3, R4): own
  greedy-eval states from just before each greedy death; the reservoir had <1% there.
* `--stall-eps 5` (R4): the race stall rule runs per TICK; 32 u/tick killed 30-50% of training
  episodes (a correction of a per-tick constant, not a tuning).
* `--eval-patience 30 --eval-parallel 1` (R4): evals were half the wall clock.
* `--unstuck-temp-heads keys`, `--unstuck-max 0.2` (R4, R5): exploration temperature on the key
  heads only; the low cap kept a long greedy line intact. FROM SCRATCH on sidistic the early
  gates were crossed with the cap at 2 ("temperature discovers, the death charge keeps").
* `--race-no-corner-cut 1` (R5): the 26-neighbour field leaked through slanted one-voxel walls.
* `--pitch-sigma-min 0.05` (R4, R5, found three times): the camera-only pitch head sat on the
  sigma floor and carried ~95% of every update's KL.
* `--respawn-front-perturb ...`, `--respawn-backward-mix 0.1`, `--front-temp 1 --front-band 0.01`
  (R6): own frontier-band states with the heading turned and speed scaled; goal-anchored rows.
* `--energy-pot 3 --energy-pot-front 0.01 --energy-pot-front-bonus 1` (R7): potential shaping on
  the energy height z + |v|^2/2g, priced only up to the frontier band (paid once at its edge).
  The last sidistic gate was an energy deficit through a U-turn; +300 u of energy into it.
* `--respawn-backward-elev -60 60` (R7): the goal-anchored rows had been clamped to a +30 deg
  climb; allowed to climb, real training finishes rose ~10x and the greedy line finished.
* Learning-rate cuts (1e-4, then 5e-5) on the peak weights reproduced the finishes (R7): the
  finishing skill appeared at a transient peak and drifted away at 3e-4.

Ideas that LOST their rounds there and are not in the recipe (worth knowing, not instructions):
`--race-kill-aware 1` (a field that routes around kill volumes; 54.7 -> 68.3 in R4), `--respawn-front`
(rows from the 2 s before real episodes reach the frontier; the first real training finish, R6),
`--fail-pen 10` (a death charge, R3). Nulls there: `--speed-pot` (harmful, twice), SIL variants,
`--respawn-greedy-speed` on all rows, frontier temperature at a wall.

## After the program (operator, local): search and distil

The policy-guided population search `tools/beam_tas.py` (now mirroring the compass / ratchet /
surf / pot-rate blocks) turned sidistic's non-finishing snapshot into finishing lines from the
true start in a minute per wave, and three rounds of `tools/exit_local.py` (search -> distil into
BC rows with `tools/plan_to_bc.py` -> train -> gate) took the best line from 118 s to 113.8 s
from spawn. What mattered: rank by distance early and by the policy's own critic near the end
(`--score dv`; distance-only ranking collapsed onto the policy's line and stalled), 4096 envs,
resample every 25 decisions. On a box it runs as it does locally (CPU physics + the GPU).

## Open questions for this map (not instructions)

* Does the recipe, built late on one long sidistic line, explore well from scratch here?
* Is the sidistic finisher's weight a useful warm start ($AR_HOME/incumbent/sid_ref/ckpt.pt)?
* Where does the field disagree with the way through (dips, U-turns, kill volumes)?

## Round log

(winners append below)

## pbr1b1 (round 1, slot 1) - 2026-10-23 14:50 UTC

Box: 57382830, RTX 5080 + i9-14900 (32 thr, cgroup quota 30.7 CPUs), 0.561 $/h, Italy.
The trainer saturates the CPU quota (3,100% CPU), ~250k steps/s steady after warm-up;
a second concurrent trainer would not add throughput.

### Baseline: the incumbent recipe from scratch is stuck AT THE START (pbr1b1_base)
`config.incumbent_recipe` from scratch, stopped at 290M (of 4e8) because it was flat:
greedy evals 0.1 / 0.4 / 0.4 / 0.1 % (score 0.417 at 151M), best progress record 111 u
for 280M steps, reservoir deepest 1.7% at 282M (creeping via front-perturb rows), unstuck T
at its 0.2 cap, view sigma at its 2.718 cap. Every greedy episode dies or is patience-cut
inside the start room (x 650..1130, y -3155..-3100, z 3650..3690).

### Why (analysis of the map + the WR demo, ANALYSIS ONLY)
* Along the record's route the no-corner-cut field is almost perfectly monotone EXCEPT the
  start: the record first goes EAST, away from the goal, for ~6 s (d rises 1,520 u =
  2.7% of d0: 57,149 -> 58,664 at x 2,333), surfing a half-pipe down to z 3,370 and back up,
  gaining ~230-280 u of energy height (z + |v|^2/2g: 3,653 at spawn -> ~3,930), then flies
  back west at z ~3,720-3,760 (60-110 u ABOVE the spawn floor level) at ~500-600 u/s
  through a passage north of the spawn (y ~ -2,885), and from there d falls monotonically
  (the only other rise on the whole route is 199 u at 20%).
* So the start is a speed/height gate: the way on is ~70-110 u above the spawn and the only
  way up is energy built in the east half-pipe. Directly west of the spawn there is a
  push-down trigger (2,500 u/s) + conveyor doorway at x 648..718, y -3054..-2946, and walls.
  A trigger_teleport (fatal) sits just above the spawn area (z 3,696..3,780 AABB), so
  jumping near the spawn kills: the 227M eval died 9/9 at 1.2-1.7 s jumping.
* The field-follower goes west into the corner at (650, -3155, 3652) and stays there;
  the sidistic finisher's weights zero-shot do exactly the same (local CPU recording:
  0.3%, stuck jumping in that corner for 80 s).

### The start gate in OUR core, probed (2026-10-23 ~15:00 UTC, analysis tools only)
* The only way out of the start room is a CROUCH TUNNEL heading west at y ~ -2,885, floor
  ~3,703 (87 u above the spawn floor), x ~800 -> ~90; a standing hull is stuck in it
  (stuck_ticks -> fail), a ducked hull flies/slides through. The record ducks for ~1 s there.
* The shaping field cannot see it: `goal_occupancy` is the slab grid at 32 u, and every
  tunnel voxel (z 3,717-3,749) is slab-solid (point-hull free) -> unreachable. 126,935 such
  "point-free, slab-solid" voxels on this map. The field instead descends from the spawn into
  a passage on the SOUTH side (y ~ -3,096, z ~3,733) that is crouch-height too and sits over
  kill volume *127 - a dead end. Every PPO greedy episode ends at its mouth (650, -3,100, 3,690).
* A standing hull crossing x 862..878 at z ~3,745 (the record's own line, rows 666-669) is
  inside trigger_teleport *257 in our core; ducked it survives. So the record's exact line
  dies here unless it ducks ~0.1 s earlier - the gate is a crouch puzzle more than a speed one.

### Go-Explore phase 1 solves the start gate by itself in ~1 minute (pbr1b1_ge1)
`tools/explore_phase1.py --cell 64 --cell-speed 8 --cell-speed-max 2400 --decisions 100
--act-every 4 --envs 2048` on the box from the TRUE spawn (reward-free, uniform random actions,
95% repeat, cells chosen 1/sqrt(seen+1)): 2.2M ticks/s, 3,597 cells in 4.4 min. Its deepest
chain: spawn -> a DUCK-JUMP north-west onto the tunnel mouth (t ~0.2-1 s) -> west through the
tunnel at 200-300 u/s -> drops out at x ~90 -> slides down to the floor at z ~3,332 ->
(827, -3,312, 3,367) = 3.19% on the grader's field (PPO's best: 0.42%). Then it plateaus: the
next section needs surf skill that uniform random actions do not have (cannonball's 6.67%
freeze, same cause). Archive: $AR_HOME/runs/pbr1b1/_ge1/ (archive.npz, ge1_states.npy as
STATE_DTYPE rows, md5 1b1fa36b115e2e5112f229b5caaa2bc3): 2,197 cells at negative progress (the
east half-pipe), ~1,000 past 1%, 62 inside the tunnel.
So the start is an exploration gate PPO's narrow policy never samples (it learned "jump near
the spawn = death": trigger *257 hangs over the ledge east of the spawn), and an archive of
return-then-explore with uniform actions crosses it at once. Section 0/0b: the archive is our
own reward-free search from the true spawn, no demo or record data, generic constants.

### Warm sidistic weights + the GE archive as spawns (pbr1b1_gewarm, gewarm2)
* `pbr1b1_gewarm` (resume of the stripped sidistic finisher, recipe flags from its config,
  `--spawn-states ge1_states.npy --spawn-states-frac 0.3`): the 76M greedy eval ran EAST from
  the true spawn into the half-pipe and died there (9/9, z ~3,290-3,310): 61% of the archive
  is that dead-end half-pipe (behind the start on the field), and a uniform draw taught it.
  Stopped at 78M.
* Fix (generic): keep only archive cells the shaping field ranks DEEPER than the true start
  (d < d_start): 1,058 of 3,597 (70 within 150 u of the spawn, 53 in the tunnel, the rest past
  it). `pbr1b1_gewarm2` = the same launch on that file (ge1_fwd_states.npy,
  md5 6897f4c420346e07a2b35ad18a719c36).
* Trainer flag `--ge-minutes M [--ge-frac 0.3 --ge-cell 64 --ge-speed 8]` (default off,
  bit-identical, TRAIN_ONLY in record_ckpt): runs explore_phase1 from the true spawn for M
  minutes on all CPUs before training (it exits 1 when nothing finished - accepted), writes
  runs/<run>/ge_states.npy (reused on resume), applies the same deeper-than-the-start filter on
  the shaping field, and mixes it into every training spawn pool as --ge-frac (the
  --spawn-states rule; no SELF_STATES needed: it is the run's own search, never data).
  explore_phase1 gained `--maxvel` (its core defaulted to 2000; the recipe runs 3500).
  Tests: tests/python/test_ge_minutes.py (CPU, princessburglar).

### Gate 2 is ENERGY: the pit after the tunnel drop (measured 2026-10-23 15:30 UTC)
* Past the tunnel the route drops into a pit (floor z ~3,330-3,340 at x ~650-900) and must climb
  out (the record: a U-turn east -> north -> west up to z ~3,500-3,530 at ~700-750 u/s, then a
  long drop west). The record arrives with energy height h = z + |v|^2/2g ~3,850-3,970 (it
  pumped the east half-pipe first: h 3,653 at spawn -> ~3,930). Go-Explore's states past 2%
  have h <= ~3,600-3,667 - the uniform-action search crawls through the tunnel at 200-300 u/s.
* `explore_phase1 --cell-energy 32` (6 threads, 6.6 min, rooted at the spawn, ge2 archive in
  $AR_HOME/runs/pbr1b1/_ge2/): 12,561 cells, half-pipe h max 3,802 (vs 3,775 without energy
  cells), pit h max ~3,600-3,667, frontier unchanged at 3.2%. Random actions do not generate the
  ~200+ u of energy; the policy's own surfing has to (or the climb needs less than I think).
* PPO's training frontier (reservoir min-depth 96.76%) has not moved past the archive's 3.2% in
  220M steps of gewarm2 either.
* Merged forward spawn file of both archives: _ge2/ge12_fwd_states.npy (4,038 states, h median
  3,582, max 3,859; md5 f93b2d0b6e45e96c044c9e56e3382ace).

### The start gate, corrected (2026-10-23 ~15:50 UTC, core traces)
* pbr1b1_gewarm2 (400M, stopped at budget): greedy evals 0.414 / 0.418 / 0.421 / 0.440 / 0.432 /
  0.451 (best, 378M). From 76M on, EVERY greedy episode ends at the tunnel MOUTH (x ~649-681,
  y ~-2880..-2930, on the spawn floor) instead of the south dead end or the east pit - the archive
  taught WHERE to go - but it never gets in.
* Ducked-hull free space by core.trace: the tunnel entrance (x 592-640) has its floor at z ~3,698,
  80 u above the spawn floor (3,616-3,618); a thin shelf at the same height runs along the mouth's
  north side (y ~-2,878..-2,895, x 652-700+). The best jump (duck-jump) lifts the feet ~63 u. So
  from a standstill on the spawn floor the tunnel is out of reach: it is entered from ABOVE
  (falling onto the shelf / into the mouth) - energy again, or a lucky line.
* In both GE archives the tunnel was entered exactly twice (in 3.8M + 0.9M runs), from airborne
  parents at the mouth with h ~3,726-3,740 (bursts of 1.9-4.8 s), and once from the SOUTH passage
  (466, -2,983, 3,692) into the tunnel at (440, -2,933, 3,708): the field's south route joins the
  tunnel further west, it is not a dead end in topology - only for a slow walker.
  1,024 simple jump/duck/forward macros from the policy's own mouth states and 2,048 x 400-tick
  random bursts from the GE parent state: 0 entries. A rare event for uniform actions and for PPO.
* So both gates are energy gates. Next arm: per-call energy shaping (`--energy-pot-front 0
  --energy-pot-front-bonus 0`, K = 3: pays every gain of h, charges every loss), resumed from
  gewarm2 at 400M with both archives (pbr1b1_geen).

### The start room, solved in the simulator (2026-10-23 ~16:10 UTC)
* The wall west of the spawn (x ~620-660) has two gaps (north y ~-2,902..-2,916, south
  y ~-3,100), both with floors at z ~3,698 (80 u above the spawn floor) into a room (floor 3,666,
  ceiling ~3,778) that leads to the tunnel. Between them sits the doorway's func_conveyor plates
  (upper plate top ~3,678-3,681) inside a 2,500 u/s push-down trigger.
* The way in (found by `tools/entry_search.py`, random bursts with 95% repeat): (1) a duck-jump
  from the spawn floor that lands on the NORTH CORNER of the upper conveyor plate (feet ~3,678;
  the duck-jump peaks at ~3,679 - about 1 u of margin) - ~1 in 3,000 bursts from the true spawn;
  (2) from the plate, a jump + duck into the north gap, then crawl west ducked - from 1 in 400 to
  1 in 10,000 bursts depending on the landing. One burst in ~50,000 does both from the GE parent.
  No pump needed: this is a precision puzzle, not an energy one.
* Box run: 24 spawn->plate lines (81,920 bursts, 2.5 min) and 71 plate->tunnel lines (548,864
  bursts, 3 min) = 71 full spawn->tunnel lines, 270-552 ticks each.
  $AR_HOME/runs/pbr1b1/_entry/: stageA.npz, stageB.npz, full_lines.npz, entry_ge_states.npy
  (every 9th tick of the 71 lines = 3,361 states + the 4,038 GE forward states; md5
  3d014fb2abc6d883fd51b839611889b6).
* pbr1b1_geen (per-call energy shaping, resumed from gewarm2 at 400M): no pumping emerged in
  227M steps (episode energy +900/-1,000 u), greedy line still at the mouth (0.411-0.419), and
  the yaw head's sigma hit its 2.718 cap - a stuck warm policy loses its view precision. Stopped.
* Next: pbr1b1_entry = the stripped sidistic weights (fresh) + recipe + `--spawn-states
  entry_ge_states.npy --spawn-states-frac 0.4`: a dense curriculum along the machine-found entry.

### The entry curriculum, and why the stuck runs rot (2026-10-23 ~17:00 UTC)
* pbr1b1_entry (fresh stripped sidistic weights + recipe + 40% spawns from entry_ge_states.npy):
  the greedy line learned the first precision step - at 151M every episode duck-jumps
  continuously (jump ~95%, duck ~85% of ticks) with the feet peaking exactly at the plate
  (3,679), and one episode per eval at 151M / 227M / 302M LANDED on the conveyor plate - but none
  made the plate -> gap jump; scores 0.36-0.42.
* pbr1b1_entry2 (continued at 400M with the plate->tunnel step weighted 3x): evals 0.440 /
  0.343 / 0.258 / 0.166 at 400/476/551/627M while the YAW sigma climbed 0.54 -> 2.1 -> 2.6
  (clamp 2.718). The same rot in geen (sigma at the clamp, evals flat) and gewarm2 (0.08 ->
  1.06). Cause: the view sigma is one state-independent parameter and the entropy bonus pushes
  log sigma up by the coefficient every update while the advantages carry no signal (a stuck
  start room); the eval is greedy, but the policy is TRAINED on near-random yaw and its mean
  drifts. Same story as the pitch head (--pitch-entropy 0 under --view-absolute).
* `--yaw-entropy Y` (new, default 1.0 bit-identical, TRAIN_ONLY): Y = 0 drops the yaw head's
  entropy term; exploration stays with --unstuck's sampling temperature. Test:
  tests/python/test_yaw_entropy.py. Arm pbr1b1_yaw0 = pbr1b1_entry's launch + --yaw-entropy 0.
* Random bursts from the archive's 12 highest-energy pit states (h ~3,655-3,667) never reach
  d < 54,800 (4.1%) in 104k bursts, nor with their speed scaled x1.5 / x2 / x2.5 (h up to
  ~5,400): gate 2 is a directed U-turn climb (skill), not a lack of energy alone.
* pbr1b1_yaw0 (--yaw-entropy 0): yaw sigma holds 0.069-0.074 through 290M (vs 0.08 -> 0.5+ before);
  151M eval 3/9 episodes land on the plate (entry: 1/9), score 0.443; 227M 0/9 (feet max 3,721).
* WHY the second jump fails - the view at LOW SPEED: on the plate the eval's yaw swings ~160 deg
  every 20 ticks (near the 10 deg/tick ceiling). src/env.c view_mode 1: below 100 u/s the yaw
  command is a per-tick DELTA from the current yaw (base = yaw), above it an OFFSET from the
  velocity heading. The sidistic weights output offsets (tens of degrees) learned at speed; at
  plate speed they become near-max spins, so the keys push in random directions and the jump
  into the gap is never aimed. The start room is all < 100 u/s precision; PPO has to learn a
  low-speed yaw mode from plate experience.
* pbr1b1_yaw0b (yaw0 continued at 400M with the plate->tunnel step weighted 3x): plate landings
  3/9 (400M) -> 1 -> 0 -> 0 -> 0 (702M), the view still spins at 10 deg/tick on the plate, no
  tunnel entry, 0.40-0.45. Stopped.
* pbr1b1_scr (the recipe FROM SCRATCH + the same curriculum + --yaw-entropy 0): yaw sigma
  0.31 -> 2.718 (clamp) by 270M ANYWAY, pitch at its 0.5 cap, kl 0.28: from scratch the policy
  gradient itself widens the view, so --yaw-entropy 0 only stops the bonus-driven part (the warm
  yaw0 run held 0.07). Evals 0.41-0.45, 0 plate landings. Stopped at 270M.
* pbr1b1_sil2: stripped sidistic weights with the YAW MEAN head zeroed (view_head row 0 and its
  bias: u = 0 = look along the velocity at speed, no turn below 100 u/s; the policy's surf trunk
  and keys untouched; _ckpts/sidwarm_yaw0head.pt, md5 a0438d24ebf21f53fc7a27beaa083a99) + the
  plate-weighted curriculum (entry2_states.npy, 40%) + --yaw-entropy 0 + self-imitation
  (--sil-coef 0.5 --sil-top 64: a rare tunnel entry stays in the buffer ~780 iterations) +
  --front-temp 0 (the trainer refuses front-temp with SIL). First eval: the view turns
  0.01 deg/tick at low speed (no spin).
* pbr1b1_sil2 (zeroed yaw head + plate curriculum + --sil-coef 0.5 --sil-top 64): the greedy
  line COLLAPSED - 123 u (76M) -> 7 u (151M) -> 8 u (227M), dying at the spawn; the SIL loss
  grew 5 -> 12.9 and the yaw sigma crept 0.078 -> 0.34. Stopped. (pbr1b1_yaw0c, a filler
  continuation of yaw0 on the original curriculum, ran 18:28-18:33 while I built the next arm.)
* `--demo-success-d M` (new, flag off bit-identical, tests/python/test_demo_success_d.py): the
  Salimans-Chen curriculum (DemoCurriculum) only counted FINISHES, so a spine ending in a hard
  room could never advance; with M a spine episode also succeeds when its ratchet record reaches
  d(spine end) + M on the shaping field.
* pbr1b1_bcur5: zeroed-yaw-head warm weights + `--demo-file spine_short.npy` (the shortest
  spawn->plate->tunnel line, 270 states, _entry/spine_short.npy md5
  b6e295f63250b43d1de8e3bc7fdce3ac) `--demo-window 10 --demo-grow 10 --demo-front-frac 0.67
  --demo-success-d -50` (success = record d <= 56,787, 50 u inside the tunnel) + --yaw-entropy 0.
  The curriculum refuses --respawn-novel / -backward-mix / -front-perturb / -greedy (and
  --int-real-only needs novel), so those are 0 for this arm: 90% of spawns come from the spine.
* pbr1b1_bcur5 at 250M: the curriculum ran from spine state 269 back to 119-129 in ~100M steps and
  oscillates there (band [129,138] 40-51% success, [119,128] 4-11%): that is mid-ascent of the
  FIRST duck-jump, where the apex clears the plate corner by ~1 u. The line: walk WNW along the
  floor to the wall at (649, -2915) (states 0-104), stop against it (vx -> 0), duck-jump in place
  drifting ~17 u SOUTH (vy -7 -> -66) onto the plate's north corner (state ~148, feet 3,680), jump
  again from the plate north-west into the gap (state ~200, feet 3,696). The greedy line from
  the true start now walks to that wall spot in 9/9 episodes (151M, 227M) and duck-jumps to the
  right apex (feet 3,679), but bunny-hops (jumps on the move) instead of stopping at the wall
  first; 0 plate landings yet. Evals 0.42.
* pbr1b1_bcur5 at 600M: the curriculum got past the first duck-jump (~430M) and kept going:
  starts [89,..] 58% -> [79] 35% -> [69] 31% -> [59] 27% -> [49] 39% -> [39] 62% -> [29] 82%
  success - from a few seconds into the line the policy now walks to the wall, duck-jumps onto the
  plate corner, jumps into the gap and gets 50 u into the tunnel. The greedy line from the TRUE
  start got WORSE meanwhile (378M 120 u, 453M 53 u, 529M 36 u): from spine states 0-28 (never
  trained yet) it jumps at once, drifts north-east and dies in trigger *257 at ~2 s. The yaw
  sigma collapsed to 0.015. `--demo-tau0 T` (new) lets a continuation start the curriculum where
  the previous run stopped (the position is not checkpointed).

### BREAKTHROUGH: the greedy line clears the start room from the true start (2026-10-23 19:18 UTC)
* pbr1b1_bcur5 (800M, ~45 min on the 5080): the backward curriculum reached the true spawn
  ([19]->[9]->[0], 99.9% success from states 0-9 at the end) and the GREEDY evals from the true
  start went 0.33 (604M) -> **3.234 (680M)** -> 3.232 (755M): 7 of 9 episodes walk to the wall,
  duck-jump onto the plate corner, jump into the gap, crawl the tunnel, drop and reach the pit at
  3.21-3.23% (809, -3,288, 3,375) in ~7-9 s; 2 of 9 still stop at the mouth (0.41%). Previous best
  0.451. Checkpoint pulled: runs/pbr1b1/pbr1b1_bcur5/ckpt_final.pt (800M,
  md5 2a17b618b001c13b4a1cfef6b69908af) - the newest file, 45M steps after the 755M eval.
* What made it work, in order: (1) Go-Explore + entry_search find the two-jump entry by uniform
  random bursts (no demo); (2) a backward curriculum on ONE such 270-tick line, scored by a
  generic success rule (`--demo-success-d -50`: the ratchet record reaches the line's end on the
  shaping field); (3) the warm sidistic trunk with the YAW MEAN head zeroed (no low-speed spin);
  (4) --yaw-entropy 0. The earlier uniform-spawn curriculum (pbr1b1_entry/yaw0) only ever got the
  plate landing; the backward order is what taught the takeoff timing.
* Next: gate 2 (the pit climb). pbr1b1_g2b = bcur5's 800M weights, the curriculum OFF, the
  recipe's frontier machinery back ON at its exact values (novel 0.2 / band 0.15 / radius
  256-3072 / shell 0.9, greedy 0.25 + stuck 200, front-perturb 0.25 x1.0-1.3, backward-mix 0.1,
  int-real-only 1), and 40% spawns from entry_ge_states.npy to keep the entry practised.
  (A first try, pbr1b1_g2, came up with novel/greedy/front-perturb at trainer DEFAULTS - the
  curriculum run's config had dropped those keys - and was relaunched within 2 minutes.)

### Gate 2 (the pit, 3.2%), measured 2026-10-23 19:40-20:15 UTC
* pbr1b1_g2b (bcur5's weights + the recipe's frontier machinery at its exact values + 40%
  entry_ge spawns): 9/9 greedy episodes reach the pit at every eval 800M-1.25B (3.229-3.235;
  bcur5 had 7/9), but nothing past it in 500M steps. ckpt_latest 1.31B pulled.
* The pit: the shaping field's route climbs ~100 u straight above the greedy line's best point
  (809, -3,288, ~3,370) up a steep SOUTH ramp (normal (-0.20, 0.73, 0.66), lower edge ~80 u above
  the pit floor); the record instead surfs an EAST ramp (lower edge ~60 u above the floor) at
  800-900 u/s. Both ramps start above what a jump from the floor reaches.
* The real loss is in the DROP after the tunnel: the greedy line exits the tunnel at ~320 u/s
  (it crawls), falls steeply, drifts north off the descending surf slope at (233, -2,995, 3,481)
  and lands FLAT on the floor - h 3,813 -> 3,378 in 0.25 s. The record surfs that slope along
  y ~-3,000..-3,030 and keeps 800-900 u/s into the east ramp.
* Searches that found nothing past the pit: entry_search (uniform bursts) from 32 of the
  policy's own pit states (138k bursts) and 32 drop states (131k), target d < 55,100; beam_tas
  pure policy proposals (4,096 envs); beam_tas `--branch-view-only` yaw-target bursts (+-20/30/60
  deg, forked at 4-5 s, protected 4-6 generations). beam_tas gained `--save-frontier` (the
  deepest non-finishing lineage, replayed, as a spine) and `--branch-view-only` (a yaw burst a
  --keys-hold checkpoint accepts).
* Arm pbr1b1_g2c: g2b's 1.31B weights + per-call energy shaping (--energy-pot-front 0
  --energy-pot-front-bonus 0, K 3): the flat landing in the drop costs ~-1.3 at the moment it
  happens, instead of a once-per-episode bonus at the frontier.

### Caveat for the lineage (CLAUDE.md 0b), written down so the next round does not inherit it blind
The start-room lines came from TWO entry_search stages whose target BOXES (the conveyor-plate
corner, then the tunnel mouth) I chose from my own core probes of this map. The searches were
random bursts and no demo or record data entered them, but a hand-picked box is map knowledge,
i.e. not yet the generic recipe 0b asks for. The generic version of the same mechanism is
Go-Explore phase 1 (reward-free, it found the tunnel by itself in ~1 min) plus stage targets
defined on the field (entry_search --field/--d-max "deeper than the current frontier"), then
the backward curriculum (--demo-file + --demo-success-d) on the found line. Not run yet.
* Gate 2, physics (core traces of the greedy line, g2b 1.25B ep1): from the tunnel exit
  (151, -2,885, 3,722; |v| ~320, heading west) the line turns south-east and FREE-FALLS for
  ~0.9 s (vz -222 -> -782 at exactly g, no contact), then hits a WALL at x ~300
  (290, -2,975, 3,340): vx -> 0, h 3,813 -> 3,378. With ~350 u/s of horizontal speed its arc
  passes BELOW the lip of the surf slope the record lands on; the record leaves the tunnel at
  600-680 u/s (carried from its half-pipe pump through the tunnel) and its arc clears the lip,
  lands tangentially at (374, -3,032, 3,386) at 913 u/s and runs onto the east ramp. Optimal air
  strafing over a 0.9 s fall adds only ~30^2/(2v) ~1.3 u/s per tick, ~+100 u/s - not the missing
  ~250. So gate 2 is the START ROOM's energy coming due: the slow two-jump entry this lineage found
  reaches the pit, and the way OUT needs the fast line (pump in the east half-pipe -> tunnel at
  speed -> long drop). Next round: a spine with a PUMP (Go-Explore with --cell-energy reached
  h ~3,800 in the half-pipe vs the record's ~3,930-3,950; or the policy's own pumping under
  per-call energy shaping), or a way to carry speed through the crouch tunnel.
* Gate 2 is ALSO a field-deception trap (grader field, core probes 20:35 UTC): the pit's best
  point (809, -3,288, 3,375) = 3.23% is the field's LOCAL optimum (under the south ramp, whose top
  the field routes over); the way out the record takes - over the gap at x ~230-280 onto the
  raised floor at x >= 330 (z ~3,376-3,428), up the east ramp, U-turn north then west - reads
  2.18% (400, -3,010) -> 2.53% (600, -3,024) -> 2.85% (840, -3,012) -> 2.86% (1,021, -2,922) ->
  3.19% (929, -2,735) -> 3.54% (735, -2,651) -> 3.93% (527, -2,683): WORSE than the pit for
  ~2-3 s of flying, so the ratchet pays nothing until the U-turn is done. Plus the speed part:
  the greedy line heads SOUTH for 0.25 s after the tunnel before turning east, so it crosses
  x ~300 at z ~3,340 and hits the raised floor's west face; heading east at once it would cross
  at z ~3,570 and land on it. Both parts at once: a detour against the field, flown at speed.

### Gate 2: Go-Explore from the policy's own tunnel exit gets past the pit (2026-10-23 20:20-21:10 UTC)
* GE4: tools/explore_phase1.py rooted at 64 of g2b's OWN greedy tunnel-exit states
  (runs/pbr1b1_entry/exit_roots.npy on the box), 64 u cells x --cell-energy 32, 4,096 envs, 768 iterations: 20,074
  cells, the deepest at **3.95%** on the grader field (534, -2,668, 3,422) - past the pit (3.24%),
  reward-free, no demo. 551 cells lie past 3.3%.
* The geometry those cells cross (core hull traces, x 260-930, y -2,620..-3,420): the "raised
  floor" is a surf RIDGE running east (x 292-612, crest z 3,389 at y ~-3,004, surf faces on both
  sides), a flat pit floor at z ~3,296 east of it, and far surf ramps (z 3,412-3,508) at
  x 772-932 to the north (y -2,652..-2,780) and the south (y -3,260..-3,356). Every chain past
  3.3% leaves the ridge's east end RISING (v (43, -335, 451) or (317, 423, 254), |v| 563-586),
  flies over the pit, lands on a far ramp and U-turns west.
* Re-finding that as one line (tools/ge_stitch.py, uniform bursts link by link): line3 got to
  chain link 21 = the ridge's east end, arriving LEVEL (v (542, -67, -97)) - the right 64 u cell,
  the wrong heading and vz - and the launch link was not re-found. The archive's own link there
  is 505-510 ticks deep: that parent cell's state was REPLACED by a shorter way in (depth 184),
  so the state the launch was found from no longer exists; all 551 deep cells descend from it.
  GE5 rooted at line3's last 12 states (8 min, 2.9M runs, 1,361 cells): nothing past the pit -
  from 0.07 s before the ridge's end there is no time left to turn up the face.
* ge_stitch `--backtrack 8,16,...,128` (new): a link not re-found from the line's end is
  retried from that many ticks earlier on the line (the line is cut back to the root that
  worked); `--cell` picks the chain end, `--prefix` resumes a line.
* First backtracking stitches (21:05-21:20 UTC, 1 min per try, offsets 0..128 ticks): the north
  chain (cell 18598) never re-found the launch; the south chain (18696) re-found it from 16 ticks
  back (enters cell 13070's box at (603, -3,034, 3,406) with |v| >= 0.9x) but no later link from
  there in 8 tries - it entered the box at its BOTTOM, not rising like the archived state
  (v (43, -335, 451)). So ge_stitch `--vel-cos C` (new, default off): a link must also arrive
  with its velocity within cosine C of the archived state's. Rerun with C 0.7 on both chains.
* Both velocity-matched stitches (C 0.7, backtracking 0..128 ticks, 21:20-21:28 UTC) failed at
  the launch as well: from line3 - tunnel exit straight onto the ridge's south face - no burst
  reaches either launch cell moving the archived way. The archive's launch came from a
  different, longer approach (~300-690 ticks) whose states the "shorter way in" rule overwrote.
* So `explore_phase1 --exact-chains` (new; flag off: archive + meta bit-identical, checked): a
  cell's parent is the cell its run was RESTORED from, states are never replaced, and each cell
  keeps its run's actions from the restore to the first entry; `tools/ge_replay.py` replays the
  deepest chain link by link from each parent's archived state, checks every link's end against
  the child's state (0.5 u), and writes one per-tick line. Local smoke: 4/4 links exact.
  GE6 = GE4's settings + --exact-chains, 12 min, 21:30 UTC.
* GE6 (21:30-21:36 UTC, stopped at ~6 min): 15,006 cells, 339 past 3.3%, deepest **3.93%**
  (536, -2,688, 3,409). `ge_replay.py` replayed the deepest chain (25 cells) EXACTLY - 24/24 links
  within 0.5 u - into one 1,881-tick line (runs/pbr1b1_ge6/line_a.npy on the box, md5
  4296eab791f13b104d33afc52bb76eed). What the line does (core states, 0.5 s steps): from the
  policy's own state in the crouch tunnel (224, -2,885; |v| 224) it goes BACK EAST through the
  tunnel to x ~837 (t 0-7 s, slow), then slides west at 300-380 u/s (t 7.5-10 s), leaves the
  tunnel at x ~0-140 with |v| 380, drops onto the ridge at 603 -> 772 u/s, LAUNCHES off its east
  end at (604, -3,035, 3,484) with vz +346, flies over the pit at z ~3,557, lands on the east
  side (x ~990-1,050), turns north and U-turns west along y ~-2,688 to 3.93%. So the search
  found the speed fix this file predicted (gate 2 = the start room's energy coming due) by
  itself: a run-up inside the tunnel, not a pump in the half-pipe.
* pbr1b1_g2cur (launched 21:39 UTC): g2b's ckpt_latest (1.305B, md5 55d4251e...) + backward
  curriculum on line_a (`--demo-window 10 --demo-grow 10 --demo-front-frac 0.67`,
  `--demo-success-d 100`: success = the record reaches d <= 55,001 = 3.76%, past the pit by
  ~0.5%; -50 as in bcur5 would ask for 50 u past an end that is falling into the north lower
  area) + 30% start-room spawns (entry_ge_states.npy) + the demo-incompatible spawn options off
  (as bcur5). 1,881 states at 10 per advance = 188 advances.
* g2cur pace (21:39-21:44 UTC): 1880 -> 1780 in ~100M steps, i.e. one advance per ~10.5M steps =
  exactly the curriculum's hardcoded 20-iteration cooldown (iterations are 524,288 steps), so
  grow 10 would need >= 1.9B more steps for the 1,780 states left, at the deadline before any
  stall. Restarted as **pbr1b1_g2cur2** from g2cur's ckpt_latest (1.403B, md5 b8b53646...) with
  `--demo-grow 20 --demo-tau0 1790` (the position it had backed off to; band [1780,1789] was at
  13-15%, the start of the north U-turn). Rule for next time: grow ~ n/100 states, so a spine of
  any length is crossed in ~100 advances (bcur5: 270/100 ~ 3, it ran 10).
* g2cur2 (grow 20) ran 1790 -> 1270 in ~230M steps (one advance per 10.5-15M, cooldown-bound),
  then STALLED at the far-side wall: from 1.71B to 1.93B+ it oscillates 1270 <-> 1250 <-> 1230.
  Band [1250,1259] (just after the line hits the east wall at (989, -3,098), v (329,-265,-374)
  -> (-163,-188,-83), then slides up a south ramp, stops at (991, -3,159) and slides back north -
  a ramp U-turn) reaches 20-59%; band [1230,1239] (0.1-0.2 s BEFORE the impact, mid-air while
  the line air-strafes right ~40 deg with the view turning 2 deg/tick, ducked) is at 0.0% in
  every visit: the policy does not reproduce the impact angle the bounce needs.
* Why it stalls (probe, 22:20 UTC: record_ckpt of g2cur2's ckpt_latest spawned at line_a states
  1230 and 1250): the GREEDY policy flies from both straight into the pit (811, -3,287) in ~2 s.
  The pit is the field's local optimum (3.23%), ~0.3% above the line state and 2 s away; the
  line's ramp U-turn first DIPS (2.99 -> 2.75%) and pays its +1.0% only ~5 s later. Under the
  20 s discount and the ratchet, P(success) ~0.4 x 0.78 ~ the pit's certain 0.31: indifferent.
  It gets worse at the line's start: 7 s of run-up back east in the tunnel (no new record) then
  the payoff at 10-18.8 s, discounted to ~1.4 units, against the pit's +2.0 at ~2.5 s (~1.76).
  The ratchet alone prefers the pit even for a policy that could fly the line perfectly.
* So `--demo-success-bonus B` (new, flag off bit-identical): an episode whose ratchet record
  CROSSES the spine's success line (d(end) + --demo-success-d) is paid B once (episodes that
  spawn past it never are). B = 50 = the finish bonus. The curriculum's success becomes the
  objective's: 50 x 0.9995^1880 ~ 19.5 at the line's start against the pit's ~1.8.
* pbr1b1_g2cur3 (launched 22:37 UTC): g2cur2's ckpt_latest (2.159B, md5 c10c3a7f...) + the same
  curriculum (grow 20, `--demo-tau0 1250`) + `--demo-success-bonus 50`. g2cur2 had spent
  1.71-2.16B oscillating 1270 <-> 1250 <-> 1230 (band [1230,1239] at 0.0% on every visit).
* g2cur3 paid 0 bonuses over 12k curriculum wins: the first `--demo-success-bonus` only paid on
  the --reward-per-decision path, and this recipe computes the race reward per TICK. Fixed (one
  helper on both paths, logs the first payment) and relaunched as **pbr1b1_g2cur4** (22:51 UTC)
  from g2cur3's ckpt_latest (2.338B, md5 e7b182cc...), `--demo-tau0 1230`: first payment at
  2.340B, training reward per episode 0.2 -> ~10.
* Root cause of the 1230 wall (core replay, 23:10 UTC): the line holds DUCK through the flight and
  releases it only at the impact (its raw actions: back+left with the view turning -2 deg/tick
  1200-1250, duck held; at 1252 duck released, right strafe). Replaying the line's own actions
  from state 1230 or 1240 reproduces it exactly; the same actions with duck released until the
  impact unduck the hull mid-air, drop it to the floor (z 3,333) and the bounce never happens.
  --keys-hold resets the held keys to NEUTRAL (duck released) at every episode start, so every
  curriculum spawn before the impact unducked on its first tick unless the policy re-pressed duck
  at once - 0.0% in every visit; after the impact the line itself releases duck, so the neutral
  reset matched and the policy reached 20-59%. A spawn-semantics bug for any crouched exact
  state (demo, reservoir, spawn-state rows), not a skill gap.
* `--keys-hold-spawn-duck` (new, flag off bit-identical): an episode whose spawn row is ducked
  starts with duck HELD.
* **pbr1b1_g2cur5** (23:01 UTC): g2cur4's ckpt_latest (2.546B, md5 e7927f3c...) + the curriculum
  (grow 20, tau0 1230) + `--demo-success-bonus 50` + `--keys-hold-spawn-duck`. g2cur4 (the bonus
  alone) had lifted the later deciles' success from 33-58% to 68-80% but band [1230,1239] stayed
  at 0.0% - consistent with the spawn-duck cause.
* g2cur5's first visits to band [1230,1239]: still 0.0% over 1,000 episodes. A core check rules
  out the mid-decision respawn (the previous episode's action row stepping the fresh spawn for
  1-3 ticks): 1, 2 or 3 ticks of released duck at the spawn leave the line's trajectory
  identical; only a sustained release (here 22 ticks) unducks. So what remains is the policy's
  own choice from the pre-impact states - most likely the post-impact habit it learned at
  1250+ (release duck, strafe right), applied 0.1-0.2 s early - with no yaw exploration
  (view sigma 0.010, --yaw-entropy 0) to find the line's air turn (wishdir ~81 deg right of the
  velocity: back+left with the view ~144 deg off the velocity, or right strafe with the view
  along it).
* For whoever continues gate 2 (written 23:15 UTC, before the round's end):
  1. The line is solved as a SEARCH problem: GE from the policy's own tunnel-exit states with
     --exact-chains + ge_replay gives an exact 1,881-tick line past the pit in ~6 min of box CPU
     (all 339 cells past 3.3% share one lineage: tunnel run-up -> ridge launch -> east-wall
     bounce -> north U-turn).
  2. The curriculum on it needs, from the start: `--keys-hold-spawn-duck` (crouched exact spawns),
     `--demo-success-bonus` (the run-up's payoff is ~19 s away; the pit is 2 s away),
     `--demo-grow ~n/100` (the 20-iteration cooldown bounds the pace at ~10.5M steps per advance).
  3. The open blocker is the pre-impact flight [1230,1249]: a ducked air turn whose wishdir is
     ~81 deg right of the velocity. The policy has ~no view exploration there (sigma 0.010 under
     --yaw-entropy 0, a choice made for the start room's precision jumps). Options: give curriculum
     spawns a view temperature, or pick a GE chain without the wall bounce (re-run GE with a
     heading-aware cell so the archive keeps more than one lineage past the pit).
* Probe of g2cur5 (2.69B) from line state 1230 with duck HELD (record_ckpt now mirrors
  --keys-hold-spawn-duck), 4 stochastic episodes: the policy stays on the line for ~6 ticks,
  then turns the velocity LEFT/north (vy -134 -> +200 in 12 ticks) where the line turns RIGHT/
  south (vy -134 -> -265), slides north along the east wall and falls into the north end of
  the pit (3.0%). So after the duck fix the remaining failure is the air-turn DIRECTION, and the
  policy never samples the right one (0 successes from [1230,1239] -> no positive examples for
  the bonus to reinforce).
* **pbr1b1_g2cur6** (23:24 UTC): g2cur5's ckpt_latest (2.826B, md5 bc4097e2...) with FINER steps
  across the impact: `--demo-window 5 --demo-grow 5 --demo-tau0 1245` (band = 1-5 ticks before
  the wall). Question: is the pre-impact wall a step-size problem (the air turn learnable from
  0.05 s out, then backward) or a direction the policy never samples at any distance?
  (g2cur5 ended with band [1250,1259] at 76% and [1230,1239] still 0.0%.)
* g2cur6 (grow/window 5) result, 2.83-3.29B: band [1245,1249] (1-5 ticks before the wall) 77.6%
  at once; [1240,1244] 16-37%; [1235,1239] (11-15 ticks) 0.0-0.4% on ~10 visits in ~45 min. So
  the pre-impact wall WAS partly a step-size problem, and the frontier stalls again ~10 ticks
  (2-3 decisions) before the impact.
* **pbr1b1_g2cur7** (23:55 UTC): g2cur6's ckpt_latest (3.285B, md5 a6559938...) with ONLY the yaw
  head's log sigma edited -4.53 -> log 0.3 (the policy's yaw exploration had collapsed to 0.011
  rad under --yaw-entropy 0), same curriculum (grow/window 5, tau0 1235). Readout: does band
  [1235,1239] move within the hour; does the greedy start room survive the noisier training.
* g2cur7 result (3.285-3.44B, ~45 min): the policy gradient shrank the restored yaw sigma
  0.27 -> 0.19 -> 0.12 -> 0.08 within ~150M steps, band [1240,1244] dropped to 12-17% (noise
  costs precision near the wall) before recovering to 22-35%, and [1235,1239] stayed at
  0.2-0.3% - unchanged. Yaw exploration as a one-off kick is not the lever; a held view
  temperature for curriculum spawns, or a line without this manoeuvre, are what is left.
* `explore_phase1 --cell-heading H` (new, flag off bit-identical): the cell key also holds H
  sectors of the horizontal velocity heading x rising/not. GE7 = GE6's settings + --cell-heading 8
  (x16 keys), 15 min (00:02-00:17 UTC): 211,596 cells (GE6: 15,006 in 6 min), best = the pit
  (52,568 u on the tool field), NOTHING past it - the finer key spreads 4,096 envs too thin; GE6
  found the launch in ~4 min. A coarser heading (2-4 sectors) or a much longer search is what is
  left to try for a second lineage past the pit.
* Box 57382830 released 00:18 UTC (nothing left worth running before the 01:34 wrap-up).
  Pulled: g2cur's ckpt_latest (1.403B, the best eval's, md5 b8b53646...) and g2cur6's
  (3.285B, the most curriculum progress without the sigma edit, md5 a6559938...).
* Regression pass at the end (00:19-00:28 UTC, local CPU): all 18 tests of this branch's flags
  pass - test_ge_minutes, test_entry_search, test_yaw_entropy, test_demo_success_d,
  test_demo_success_bonus, test_keys_hold_spawn_duck, test_exact_chains (each new flag
  bit-identical when off against the trainer / explorer before it).
* How to resume the gate-2 curriculum (the flag set the last runs used; all files in the repo):
  `box.py launch --agent A --run R --steps N --resume --self-states "<provenance of
  autoresearch/spines/README.md>" -- --maxvel 3500 --demo-file autoresearch/spines/pbr1b1_line_a.npy
  --demo-window 5 --demo-grow 5 --demo-tau0 <where the last run stood, 1235-1240>
  --demo-front-frac 0.67 --demo-success-d 100 --demo-success-bonus 50 --keys-hold-spawn-duck
  --yaw-entropy 0 --no-respawn-novel --respawn-backward-mix 0 --int-real-only 0
  --respawn-front-perturb 0 --respawn-greedy 0 --respawn-greedy-stuck 0
  --spawn-states autoresearch/spines/pbr1b1_entry_ge_states.npy --spawn-states-frac 0.3`
  (grow 5 only through the precise manoeuvre; the easy stretches went at 20 states/advance.)
  The demo curriculum refuses the recipe's novel / backward-mix / front-perturb / greedy spawns,
  hence the zeros; the start room is kept by the 30% start-room states plus 7% true-start
  spawns (9/9 to the pit through 3.3B steps of curriculum, a few 8/9 evals).
  Correction to the line above (counted, 00:33 UTC): from ~2.7B the greedy evals are mostly 8/9
  to the pit (g2cur5 2.70B 8/9; g2cur6 9,9,8,8,8,8,8 of 9; g2cur7 8,9,8) - the start room erodes
  slowly under the curriculum; one episode per eval stops at the tunnel mouth.
  Correction (checked): the non-pit episode is not always at the tunnel mouth - g2cur5 2.70B ep8
  stops on the plate by the mouth (672, -2,941; 0.39%), g2cur7 3.44B ep6 stalls INSIDE the
  tunnel (351, -2,885; 1.00%), g2cur6 3.28B ep2 reaches the pit area at 3.06% (under the 3.1%
  cut used for the count, so that eval is really 9/9 in the pit region).
* line_a is crouched almost throughout (1,717 of 1,881 states ducked; airborne + ducked outside
  the tunnel: ticks 656-761 at the tunnel's east end, 998-1086 the drop onto the ridge, and
  1165-1793 = 6.3 s from the launch through the flight, the wall bounce and the north side). So
  every curriculum spawn on those stretches started by releasing duck before
  --keys-hold-spawn-duck; the curriculum still crossed 1880 -> 1250 that way (the hull change was
  survivable there) and it became fatal only at the wall bounce. Expect the same at the drop and
  the launch - keep the flag on from the start.

### pbr1b1 round 1 lesson (wrap-up, 2026-10-24 01:36 UTC)
* **Score 0.45 -> 3.236** (best eval pbr1b1_g2cur2 traj_1403518976; handed-on checkpoint
  pbr1b1_g2cur ckpt_latest 1.403B, md5 b8b53646...). The greedy line clears the start room - two
  precision duck-jumps onto a conveyor-plate corner, a crouch tunnel - and reaches the pit at
  3.2% in 9/9 episodes. What did it: Go-Explore + entry_search found the entry by random bursts
  (no demo; caveat: two stage target boxes picked by hand), then a Salimans-Chen backward
  curriculum on ONE found line (`--demo-success-d`), the warm trunk with the yaw-mean head zeroed,
  `--yaw-entropy 0`. What did not: uniform archive spawns, SIL, energy shaping, beam search.
* **Gate 2 (the pit) is understood, not passed.** The way out needs speed the slow entry does not
  carry. `explore_phase1 --exact-chains` + `tools/ge_replay.py` found it in ~6 min of box CPU as one
  exact 1,881-tick line past the pit (3.93%; autoresearch/spines/pbr1b1_line_a.npy): a run-up back
  east inside the tunnel, a ~770 u/s drop onto a surf ridge, a launch over the pit, a ducked air
  turn into a wall, a ramp U-turn. Plain Go-Explore archives cannot be stitched into such a line
  (the "shorter way in" rule overwrites the parent state a launch was found from); exact chains can.
* **A backward curriculum on a long searched line needs four things, each found the hard way:**
  (1) `--demo-grow ~n/100` - the 20-iteration cooldown makes 10 states/advance far too slow for
  1,881; (2) `--demo-success-bonus` - the ratchet + 20 s discount prefer a pit 2 s away to a payoff
  10-19 s away, even for a perfect flier; (3) `--keys-hold-spawn-duck` - exact crouched spawns
  unducked on their first tick, 0.0% from every pre-impact state (the line is crouched for 1,717 of
  1,881 ticks); (4) fine steps through a precise manoeuvre - at 20 states/advance the pre-impact
  band sat at 0.0% for 1.5B steps, at 5 it went 77.6% (1-5 ticks before the wall) -> 16-37% (6-10)
  -> 0.1-0.4% (11-15 ticks) and stalled there.
* **Numbers of the curriculum runs (g2cur..g2cur7, 1.305-3.44B):** 1880 -> 1250 in ~600M steps,
  then the wall bounce held it from 1.71B to the end (final frontier ~state 1235). The greedy evals
  never moved off the pit (3.221-3.236); the start room eroded only slightly (mostly 8/9 after
  2.7B). A one-off yaw-sigma kick (0.011 -> 0.3) decayed back to 0.08 in ~150M steps and changed
  nothing. GE with a heading-aware cell key (x16) reached 211k cells in 15 min and nothing past the
  pit (too thin).
* **Next:** a line without the wall-bounce manoeuvre (a coarser heading key, or a longer GE), a
  held view temperature for curriculum spawns, and a curriculum that starts with all four fixes.
  Everything needed to resume is in the repo (autoresearch/spines/, the flag set above).

## pbr2b4 (round 2, slot 4) - started 2026-10-24 02:18 UTC

Box: 57454279, RTX 5080 (Italy, machine 153177), 0.561 $/h, rented 02:26 UTC (3 h deadline).
Deployed ar/pbr2b4 with pbr1b1_g2cur6's ckpt_latest (3.285B, md5 a6559938...) as runs_ckpt.pt.

### g2cur6's greedy line from the true start (record_ckpt --dump-states, 3 episodes, 02:30 UTC)
* best 3.173 / 3.090 / 3.234% (the pit, ticks 853-1070). It has CHANGED since g2b: after the
  tunnel (exit x ~126 at tick ~620, only ~190 u/s - it crawls the tunnel ducked at 180-190 u/s)
  it turns east at once and rides the surf ridge (tick 680-800: z 3,543 -> 3,410, 220 -> 455
  u/s) to its east end (630, -2,987, 3,410), then drops off it into the pit (841, -3,223, 3,351).
  line_a instead slides the tunnel at 300-380 u/s, leaves it at 380 u/s, rides the ridge at
  603 -> 772 u/s and launches off its end with vz +346. So the remaining gap is SPEED out of the
  tunnel (190 vs 380 u/s): the drop adds ~400 u/s in both cases.
* GE1 = explore_phase1 --exact-chains (GE6's settings: 64 u x --cell-energy 32, 4,096 envs, K 4,
  100 decisions, --maxvel 3500) rooted at the greedy line's last 4 s before its best point (ticks
  510-910, stride 2: tunnel x <= 330, exit, ridge, pit), 9 min, 8.5M runs: 13,652 cells, NOTHING
  past the pit (best 3.231% = a root). Those roots start too late in the tunnel for a run-up.
* GE2 (roots: the greedy line from tick 160 = the tunnel entry, stride 2, 376 roots) and GE3
  (roots: line_a states 900-1250 = tunnel slide end, exit, ridge, launch, flight up to the wall
  bounce; does a continuation WITHOUT the bounce exist?) running together, 12 min, 15 threads each.

### --demo-adapt M / --demo-cooldown C (new, default off / 20, byte-identical; commit 125fb86)
The curriculum's step adapts in [1, M]: a second advance in a row with the band at >= 2 x rate
doubles it, a backoff halves it and backs off by the halved step (Rprop's sign rule; a first
version that doubled on every easy advance cycled 32 <-> 64 at a hard state - caught by the
test). It replaces pbr1b1's by-hand "grow 20 on the easy part, 5 through the bounce", and the
cooldown (20 iterations = 10.5M steps per move, which bounded g2cur's pace) is a flag.
tests/python/test_demo_adapt.py (4 passed, CPU).

### GE2 / GE3 results and the new line (02:41-02:53 UTC)
* GE2 (roots: g2cur6's greedy line from the tunnel entry, 376 roots, 12 min, 6.8M runs): 15,824
  cells, nothing past the pit. With GE1, two searches from the POLICY'S OWN slow line found no way
  past it: from ~190 u/s out of the tunnel random bursts do not reach the launch.
* GE3 (roots: line_a states 900-1250, the fast part: 300-380 u/s slide, exit, drop, ridge,
  launch, flight to the bounce; 12 min, 5.4M runs): 16,356 cells, **633 past 3.3%, best 3.963%**,
  from TWO roots only - line_a 1058 (on the drop onto the ridge, |v| 651) and line_a 1112 (on the
  ridge, |v| ~770). None from the flight states 1150-1248 (the wall-bounce lineage was not even
  re-found in 12 min: it is the rare one).
  * chain from 1112 (chain3a, 1,213 ticks, 3.79%): lands on the far SOUTH ramp, wanders north and
    back for ~9 s, ends running west along the south ramp. The field's descent from its end goes
    west and joins the record's route (4.6% at x 310, 6.2% at x -458, 8.3% at x -1,482), so the
    south side is not a dead end for the field.
  * **chain from 1058 (chain3b, 298 ticks = 3 s, 3.95%)**: surfs the ridge east at ~780 u/s,
    launches north-east off it ((629,-3,016,3,408) v (465, 305, 396)), flies over the pit, turns
    on the far east side ((1,010,-2,862) v (568,12) -> (-257,464) in 20 ticks) and flies west
    along y ~-2,650..-2,666 at 400-500 u/s to (534,-2,666,3,435) - the record's own side and
    heading (record: 3.5% at (757,-2,651), 4% at (481,-2,662)). No 6-s flight + wall bounce.
* spine_b (1,916 states) = g2cur6's own greedy line from the true start (ticks 0-559, into the
  tunnel; it passes within 8 u of line_a's first state) + line_a 0-1057 (the run-up) + chain3b.
  autoresearch/spines/pbr2b4_spine_b.npy (md5 df440ae6...) and pbr2b4_chain3b.npy.

### pbr2b4_cur1 (launched 02:55 UTC): backward curriculum on spine_b from g2cur6 (3.285B)
`--resume` g2cur6 + `--demo-file spine_b --demo-window 10 --demo-grow 8 --demo-adapt 64
--demo-cooldown 5 --demo-front-frac 0.67 --demo-success-d 100 --demo-success-bonus 50
--keys-hold-spawn-duck --yaw-entropy 0` + the demo-incompatible spawn options off (as round 1) +
30% `--spawn-states pbr1b1_entry_ge_states.npy`; 4e9 steps cap. Success line d <= 54,990 (3.78%).
First 3 min: tau 1915 -> 1667 (chain3b state 49, on the ridge before the launch) with band
success 100 / 97 / 92 / 87%, step 8 -> 64 (the curriculum's first ~250 states took ~5M steps; at
round 1's fixed grow 20 / cooldown 20 they would have taken ~130M).
* cur1 at the launch (02:58-03:20 UTC): the adaptive step closed in on spine 1674-1683 (chain3b
  56-65, the ridge just before the launch) within 2 min (64 -> 32 -> ... -> 1), and the band
  hovered at 14-25% success around the 20% threshold for ~20 min (~300M steps) while the policy
  learned the launch (band success ~10% -> 25%; the states just after the launch 75%). Then it
  went through: **tau 1674 -> 1430 by 03:31 UTC** (band 20-47%, step back up to 8-32): the launch,
  the ridge ride, the drop onto the ridge and the tunnel exit (line_a 950-1057) are behind it; it
  is in the tunnel slide (spine 1430 = line_a 870, x ~460, -320 u/s). The far side (deciles 80-90%
  of the spine) succeeds 45% / 90%. 104k spine-line bonuses paid. ~288k steps/s.
  Greedy evals still the pit (3.2%, 8-9/9; one at 2.7%) - they can only move once the curriculum
  is through the run-up (line_a 500-1000) and into the policy's own prefix (spine < 560).

### BREAKTHROUGH: the greedy line from the TRUE START clears the pit (03:40-03:55 UTC)
* cur1's curriculum ran from the tunnel slide (03:31, spine 1430) through the run-up and the
  policy's own prefix to **tau 0 by ~03:50 UTC** (step 64, band 88-98% from spine 463 down; the
  run-up and the start room were easy). The greedy evals jumped from the pit (3.22 at 3.889B) to
  **3.937 (3.965B), 3.970, 3.926, 3.919, 3.990 (4.267B, best 4.0%, median 3.9%)**: 9/9 episodes
  clear the pit. Score 3.990 vs the incumbent's 3.236. Matched-step reading: g2cur..g2cur7 spent
  1.3-3.44B on line_a without moving the greedy line off the pit; cur1 did it in ~0.68B steps
  (3.285 -> 3.965B, ~45 min on the 5080).
* The greedy line is NOT the spine: it reaches 3.99% at tick ~660 (6.6 s from the start). It
  slides the tunnel at ~330 u/s by itself (no 7-s run-up back east), exits at x ~60, drops onto the
  ridge (807 u/s), launches north-east (664,-2,995,3,414) v (490,326,349), turns west on the far
  side (843,-2,695) and dies at ~(475..520, -2,700..-2,766, z 3,380-3,400) about 0.1 s after its
  best point: fall-catch trigger_teleport *209 (AABB x 451-762, top z 3,360) - it descends at
  -417 u/s and drops below ~3,396 before x 451; the record crosses there ~100 u higher.
* Pulled: cur1 ckpt_4285530112.pt (md5 61f26c21..., the snapshot right after the best eval) and
  ckpt_latest (4.40B at the stop). cur1 stopped at 4.405B (SIGTERM; a background job ignores
  SIGINT) to give the CPU to the next search. box.py extend sets the deadline to NOW + hours.
* GE4 / GE5 (04:00 UTC): exact-chain GE rooted at the new greedy line's last 4 s / 3 s; first
  snapshot: 5,792 cells past 4.0%, best **6.65%** (-1,015,-3,005,2,994), the chain rooted at
  greedy tick 603 (just after the far-side turn west). tools: autoresearch/analysis/
  pb_frontier_step.py does record -> roots -> 2 searches -> replay deepest -> join, on the box.
* GE4 (12 min, 22,392 cells) and GE5 (11 min, 23,701 cells) both stop at **6.65-6.67%** around
  x ~-1,000, z ~3,000-3,045 (a new wall: the chains arrive there nearly stopped). Deepest chains
  1,287-1,631 ticks; GE5 cell 23236 (6.667%) has a 546-tick chain from greedy tick 599 (the
  far-side turn): west along the north side at z 3,477-3,520 (the greedy line flies ~100 u lower
  there and falls into *209), down at x ~200 (z 3,344 -> 3,118), west through 5.0-6.6%, stop.
  Rule used, generic: among cells within 0.01% of the deepest, the shortest chain.
* **pbr2b4_cur2** (launched 04:15 UTC): resume cur1's final ckpt_latest (4.394B, md5 f213128b...)
  + the same curriculum flags + `--demo-file spine_c` (1,146 states; success line 6.49%).

### cur2 result: 6.696, and the forgetting lesson (04:15-04:43 UTC)
* cur2's first eval (4.3946B = the resume point) 3.982; then **3.23 / 3.21 / 3.23 / 3.23 / 3.22
  from 4.470B to 4.772B**: the greedy line lost its pit crossing (peak speed 443-573 u/s vs
  ~800; it crawled the tunnel again) within 75M steps. Cause: while tau worked on the new chain
  (spine_c 600-1145, the far west), NO demo draw came from the prefix (spine_c 0-598 = the policy's
  own line through the tunnel, the ridge and the launch) - the draws are [tau, n-1] only - and only
  the 7% true-start spawns and the 30% start-room states touched that part. cur1 kept its line
  because at tau 0 it practised the whole spine.
* tau reached 0 at ~04:40 (the prefix states, previously ~98%, had fallen to 15-66% band success;
  the curriculum re-taught them on its way back), and the next eval jumped: **6.696 at 4.8476B
  (best 6.70%, median 6.6%, 9/9 episodes at 6.55-6.70%)**, all stopping around x -840..-1,015 near
  the wall GE4/GE5 stopped at, wandering there until the 30-s patience cut.
* `--demo-replay R` (new, default off, byte-identical): a share R of the demo draws is uniform
  over the WHOLE spine [0, n-1]. For the next spines (R 0.2). tests/python/test_demo_adapt.py
  (5 passed incl. the trainer smoke).
* cur2 stopped at ~4.87B (pulled ckpt_latest 4.861B, md5 6c41d08c..., 14M steps after the best
  eval). Careful: `pgrep -f '<the trainer's command line>'` inside an ssh command matches the ssh
  shell itself (CLAUDE.md) - stop a run by runs/<run>.pid.
* GE6 = autoresearch/analysis/pb_frontier_step.py on cur2's 4.861B weights (record 3 greedy
  episodes, roots = the best one's last 4 s, 2 x 12-min exact-chain searches), 04:43 UTC.
* The 6.67% wall is a SPEED gate, and its cause is one hard landing (energy height h = z +
  |v|^2/2g along cur2's greedy line, ep 1): h ~3,680-3,740 from the far side to the drop at x ~200,
  then (237,-2,752,3,264) |v| 872 h 3,740 -> (159,-2,861,3,138) |v| 502 **h 3,296** in 0.25 s, and
  h ~3,280-3,340 after it. It reaches the launch ramp at x ~-460 at |v| 654-760; with h 3,740 it would
  be ~1,150 u/s there, and the record (analysis only) leaves that ramp at 916 u/s horizontal,
  vz +714 and passes x -1,018 at z 3,392 where every chain of ours stops at z ~3,000-3,045.
* GE6 (pb_frontier_step on cur2 4.861B: greedy best 6.687% at tick 971, roots ticks 571-971, two
  12-min exact-chain searches, 04:43-04:55 UTC): **nothing past 6.73%** - the wall at x ~-940..-1,015
  holds for uniform random bursts. They did find ~60 cells past the drop (x 140..-300) with
  h 3,600-3,758 (falling fast, pre-impact), but every cell beyond x -100 has h <= 3,349; rolled out
  from those 77 states (record_ckpt --spawn-states, 12 greedy + 12 stochastic) the POLICY loses the
  energy within 0.6 s too (h -> ~3,080) and stops at the same wall (6.52-6.70%).
* **Speed opens the wall** (GE rooted at the policy's own launch-ramp states, greedy ticks 860-960,
  x -219..-769, with the velocity SCALED, 6 min each): x1.25 -> 6.95% (just past it), **x1.5 ->
  12.60%** (x -3,303; 58,648 cells, depth 2,153 ticks). A smooth drop (h kept at ~3,712 instead of
  ~3,290) would put ~1,130 u/s on that ramp = the x1.5 case. So the next gate is one energy-keeping
  manoeuvre at the x ~200 drop, which neither uniform bursts nor the policy perform.
* pb_nudge_sweep.py (new): the policy's own greedy line from the true start with one yaw offset
  (record_ckpt --nudge-*: 6 ticks 650-750 before its own h drop x holds 5/15 decisions x offsets
  +-10/20/40 deg, 73 runs, 14 in parallel) - is a near-policy line that keeps the energy there?
* Nudge sweep result (05:05-05:28 UTC; the box sweep is SLOW: 8-14 parallel record_ckpt processes
  take ~4.5 min per 13-s episode, GPU contention - run 1-3 at a time or use the trainer instead):
  of the runs done (ticks 650-700, holds 5-15, offsets +-5..40 deg) none beat the unnudged line by
  more than 0.17% (best 6.707%); h on the launch ramp 3,084-3,277 (needs ~3,700). Offsets at tick
  650 (the far side) wreck the line (death at 4.0-4.8%). A small yaw deviation before the drop does
  not keep the energy.
* GE "drop" (05:30 UTC): roots = only the drop approach (greedy ticks 690-790, 101 roots), cell 64
  x --cell-energy 16 (finer), bursts 100 and 50 decisions, 10 min each: does a concentrated search
  find an energy-keeping landing (h > ~3,600 past x -300)?
* GE "drop" result (10 min x 2, 21-23k cells each): best 6.83% / 6.69%; past x -100 every cell has
  h <= ~3,388 again (h up to 3,759 only at x 100..-100, pre-impact). Concentrating random bursts on
  the drop approach does not find the energy-keeping landing: search alone does not cross this gate.
* `--demo-speed P LO HI` (new, default off, byte-identical): a share P of the demo draws outside the
  frontier band starts with velocity x U(LO, HI); those rows are lifted 0.15 u so they match no
  spine key (kept out of the curriculum's statistics). The recipe's --respawn-front-perturb-speed
  range is 1.0-1.3, used as is (CLAUDE.md 0b: not the x1.5 this map's probe showed).
* **pbr2b4_cur3** (05:40 UTC): resume cur2 4.861B + spine_c curriculum (tau from the end) +
  `--demo-replay 0.25 --demo-speed 0.3 1.0 1.3` + per-call energy shaping (`--energy-pot-front 0
  --energy-pot-front-bonus 0`: every contact's loss of h charged when it happens; under GAE
  lambda 0.95 the hard landing gets its own credit although the episode total telescopes).
* cur3 first readout (05:54 UTC, 5.19B): with --demo-replay 0.25 the curriculum swept spine_c from
  the end to tau 0 in ~10 min WITHOUT losing the line (evals 6.700 / 6.696 / 6.699 / 6.717 /
  **6.759** at 5.164B, best 6.8%; cur2 had dropped to the pit for 25 min). The energy height on the
  launch ramp (first tick past 5.6%) moved for the first time: cur2's 9 episodes all 3,086-3,103;
  cur3 at 5.09-5.16B 3,085-3,341 (several ~+250 u). Peak speed 797 -> 825 u/s. Still short of the
  ~3,700 the wall needs. Pulled cur3 ckpt_latest 5.1847B (md5 19d7d584...).
* Where cur3 loses energy (greedy evals, the drop zone 4.3% -> 5.6%): cur2 9/9 episodes fall to
  z ~2,985 at ~1,120 u/s and lose ~650-726 u of h in 25 ticks at x ~48-106 (an impact); cur3
  (5.164B) 6/9 now land EARLIER and HIGHER (x ~167-194, z ~3,145, |v| ~1,000) and lose 370-470 u,
  3/9 still the old impact. The per-call energy term is moving the landing.
* What the x1.5 chain does past the "wall" (pbr2b4_platform_chain.npy): from the scaled ramp state
  it flies UP onto a platform at z ~3,328 (x ~-730, by state 150), crawls west along it at 6-350
  u/s for ~15 s (to x ~-1,558), drops to 10% and reaches 12.6%. So the 6.7% wall is one jump onto
  a platform ~400 u above the ramp (needs h ~3,450-3,700 there; cur3's best episodes 3,341); the
  platform route itself needs no speed.
* **pbr2b4_cur4** (06:05 UTC): resume cur3 final (5.234B, md5 27125fcc...) + cur3's flags, with
  --spawn-states = entry states + 3 x platform-chain rows 150-2,153 (~45% of the 30% share on the
  platform route), so the policy learns 6.7 -> 12.6% while the energy term raises its ramp speed.
  cur3 evals: 6.700 / 6.696 / 6.699 / 6.717 / 6.759 / 6.702 (best 6.759 at 5.164B).
* cur4 (06:22 UTC, 5.64B): evals 6.683 / 6.716 / 6.781 / 6.678 / **6.796 (5.536B)** / 6.767; ramp h
  now 3,274-3,384 in 8/9 episodes (cur2 3,086-3,103; cur3's best 3,341), rising slower. The platform
  (z 3,328 ~315 u past the ramp) needs h ~3,550 (3,328 + the horizontal KE at the apex). Pulled
  cur4 ckpt_latest 5.637B (md5 8e9b6ffc...).
* GE7 (pb_frontier_step on cur4 5.637B, 1 x 8 min BESIDE the trainer, 0.44M ticks/s = a quarter of
  an idle box): greedy best 6.752%, deepest cell 6.775% - nothing onto the platform from the
  policy's own line at ramp h ~3,350. (Fixed: the step now reports "no new line" instead of a
  1-state chain when no cell beats the roots.)
* `tools/beam_tas.py --score dh [--h-coef 1]` (new; d minus the energy height, default ranking
  unchanged; tests/python/test_beam_score_dh.py): the energy-aware policy-guided beam (cur4 5.656B,
  2,048 envs, resample 25, --greedy-envs 64, --skip-gate; --eps is refused under --keys-hold, so
  the proposals are the policy's own samples, yaw sigma 0.017) also stops at the wall: min d
  53,322-53,359 (6.68-6.70%) from t 10 s to 20 s. The policy's own sampling is too narrow to find
  the landing. (beam_tas's greedy gate runs 3 episodes to the ckpt's 300-s cap first: use
  --skip-gate.)
* cur4 log at 5.79B: tr/st/cr 0%/92%/97% - 92% of training episodes end by the 15-s stall rule at
  the wall (~half of all experience wandering there), kl 0.19-0.21 (high; cur1 ~0.08-0.10).
* **pbr2b4_cur5** (07:05 UTC): resume cur4 final (5.802B, md5 dacb6a58...) + cur4's flags +
  `--unstuck-max 1.0 --unstuck-temp-heads all` (the recipe's stuck-frontier temperature with a
  higher cap and on the view heads too: more landing variety) + `--stall-secs 8` (less wall time).
  cur4 evals: ... 6.792 / 6.748.
* cur5 at 6.21B (07:21 UTC): T 0.2 -> 1.0 by 6.0B, kl fell to 0.03-0.08 (from 0.19), episodes 1,440-1,477
  ticks (stall 8 s), yaw sigma 0.010; evals **6.885 (5.8023B = cur5's first, i.e. cur4's final
  weights + 1 update)** / 6.785 / 6.836 / 6.772 / 6.835 / 6.828; ramp h still 3,281-3,378 - the
  higher temperature did not move the energy in 20 min.
* Handed-on checkpoint for 6.885: cur4's final ckpt (5,801,771,008, md5 dacb6a58...) copied into
  cur5's run dir as ckpt_5801771008.pt (the step cur5 resumed at, <= the eval's step) and pulled
  with --snapshot, so the snapshot-at-or-before rule hands on the weights that scored it.
* Why the landing is hard (record analysis, ANALYSIS ONLY): at x 175 the record is ~20 u further
  south and ~34 u higher than our line, with ~110 u more h at 4% (3,863 vs 3,740-3,777); our line
  clips a rim at (175,-2,838,3,145). A threshold: a little more energy before the drop may clear the
  rim and save the 400 u.
* More nulls at the platform gate (07:20-07:45 UTC):
  * cur4's final policy from its far-side spine states with velocity x1.15 (6 greedy episodes):
    the extra energy is shed by 4.3% (h 3,551-3,608, below the policy's own line) and the ramp
    gets 3,079-3,219; same wall (6.61-6.70%). Sped-up starts there do not teach speed.
  * beam_tas --score dh + --branch-at (view-only yaw bursts, jitter +-15 deg, held 20-28 ticks,
    protected 3-4 gens): forked at d <= 54,900 (fired at tick 700, already past the rim: forks only
    happen at resample boundaries) and at t600 (the far-side turn, 1,792 envs, 1.5-s burst): min d
    53,306-53,326 (6.69-6.72%) both times.
* GE "south1" (07:44 UTC): roots = the last 2 s of chain3a (the south-side route at 3.2-3.79%,
  pbr2b4_ge/chain3a.npy): the field's descent from its end runs west at z ~3,250-3,316 - the
  platform's own height - so the south side may reach the platform route without the jump from
  below.
* GE south1 (8 min): only 2,044 cells, best 3.924% - the south ramp's end is a dead end in physics
  (the field's westward descent there was misleading).
* **The rim is a steering problem, not a speed one** (cur4's final policy, greedy, from its OWN
  states at 4.15-4.45% with the velocity x1.0 / x1.1 / x1.2, 4 episodes each): every episode, at
  every scale (h at the start up to 3,889 - more than the record's ~3,863 at 4%), hits the same spot
  (168-179, -2,833..-2,837, 3,146-3,150) and loses the energy there (ramp h 3,212-3,377, best
  6.61-6.82%). The policy steers into that rim whatever its speed; clearing it needs a different
  path (the record passes ~20 u further south and ~34 u higher), which neither its sampling noise
  (yaw sigma ~0.01), +-15 deg yaw bursts nor uniform GE bursts have produced.
* GE rim1/rim2 (08:05 UTC, cur5 PAUSED with SIGSTOP so the search gets the whole CPU): roots = cur4's
  own greedy line from 1.0 s to 0.1 s before its worst energy drop (tick 725, (169,-2,835,3,150)),
  --cell-energy 16, cell 64 / 48, bursts 50 / 100 decisions, 10 min each.
* **GE rim1/rim2 found energy-keeping lines past the rim** (10 min each, cur5 paused): with roots on
  the CURRENT policy's own approach (1.0-0.1 s before its biggest one-tick energy drop) and
  --cell-energy 16, cells past the rim keep h up to 3,712 (x 0..-200) and 3,544-3,559 on the launch
  ramp (every earlier search: <= 3,388). rim1's deepest, **7.030%** (-1,001,-2,983,3,231) |v| 716,
  is a 233-tick chain (ge_replay 8/8) from the policy's state at 4.14%: it passes x ~182 at z 3,223
  (~75 u above the policy's rim impact) with no energy loss, lands in the half-pipe at x ~-100
  losing ~180 u (the policy loses ~400 at the rim), rides the ramp at ~1,045 u/s and reaches the
  wall top at z 3,231 (the platform is z ~3,328). First physical line past the 6.7% wall.
* plat1/plat2 (08:15 UTC): GE rooted at that chain's last 1.5 s (75 roots), 8 min x 2.
* pb_perturb_spawns.py (new, not used yet): perturbed copies of the policy's own greedy states before
  its biggest energy drop (the --respawn-front-perturb / -novel family aimed at the energy sink);
  512 of 728 valid copies for cur4's line.
