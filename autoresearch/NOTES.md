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

## pbr2b3 (round 2, slot 3) - 2026-10-24 02:18 UTC start

Box: 57454337, RTX 5080, 0.60 $/h, Florida (machine 118632), 32 threads.

### Harness issue: the round-1 checkpoint was not handed on
`AR_BRANCH.json` incumbent `ckpt: null` ("no complete checkpoint in pbr1b1_g2cur2"): the winning
eval is g2cur2's first (1.4035B), but pbr1b1 pulled the PARENT run's checkpoint
(pbr1b1_g2cur/ckpt_latest.pt, 1.403B, md5 b8b53646...), which the orchestrator's
handon_ckpt does not look for. This branch uses the two checkpoints the winner's RESULT.json and
NOTES name (md5-verified, read by exact path only, nothing else of that directory):
g2cur6 ckpt_latest 3.285B (md5 a6559938..., the most gate-2 curriculum progress) and g2cur
ckpt_latest 1.403B (md5 b8b53646..., the 3.236 eval's weights), copied to
$AR_HOME/runs/pbr2b3/_ckpts/. Operator: a winner's best eval made 0.5M steps into a resume has
its weights in the parent run; handon_ckpt could fall back to the run's `--ckpt` source.

### Probes (02:30-02:45 UTC, box)
* Uniform random bursts (entry_search, 95% repeat, 700 ticks) from line_a states 1215, 1225,
  1230, 1235, 1240, 1245 to the curriculum's success line (d <= 55,001 on the grader field):
  0 hits in 57,344 bursts - even from 1245, where the policy succeeds ~77%. The line past the
  bounce (south ramp, stop, north along the east side, U-turn west) is skill, not luck.
* record_ckpt-based benches are too slow for this (one env, ~1 min per 1,000-tick episode on a
  loaded box): tools/band_bench.py exists but use the trainer's own curriculum log instead.

### --demo-front-temp T (new, flag off bit-identical; tests/python/test_demo_front_temp.py)
An env whose episode started on the backward curriculum's frontier band [tau, tau+D-1] samples
the keys at 1 + T and the yaw sigma x --demo-front-view (default 1 + T) for its first
--demo-front-ticks ticks; the camera-only pitch head stays cold; PPO scores each row at its own
temperature (the --front-temp slots). Exploration where the curriculum is stuck, the learned rest
of the line kept cold. TRAIN_ONLY in record_ckpt.
* pbr2b3_dft1 (02:45 UTC): g2cur6's 3.285B + pbr1b1's gate-2 curriculum flag set (tau0 1240,
  window/grow 5) + --demo-front-temp 2 --demo-front-ticks 40 --demo-front-view 20 (yaw sigma
  0.011 -> 0.22 rad), 450M steps.
* pbr2b3_dft1 result (3.285-3.364B, stopped 02:51 UTC): band [1240,1244] 78% on the first visit
  (round 1's g2cur6 log: 16-37%), [1235,1239] 0.0 / 0.0 / 0.3 / 0.0% over four visits of 500
  episodes each (round 1: 0.0-0.4%). Evals 3.223 / 3.192 (pit). Keys at T 3 + yaw sigma x20 for
  the first 40 ticks of a band episode do not crack it: temperature is not the missing piece.
* WHY (measured on the spine and the incumbent eval): the greedy policy looks within +-7 deg of
  its velocity at speed (incumbent traj_1403518976, 9,974 rows with |v_h| > 200: offset p5/p95
  -7.2 / +7.1 deg - the yaw-mean head was zeroed in round 1), while line_a's view is a median
  176 deg off the velocity through the flight [1150,1250) and 139 deg at the band (88 deg on the
  ridge ride [1000,1150), 101 deg on the north side [1500,1881); only the tunnel slide
  [700,1000) looks ahead). The explorer's uniform yaw actions left it there. The core turns the
  view at most atan(30/|v|) per tick (~4 deg/tick at 420 u/s): a spawn at 1235 needs ~34 ticks
  to bring the view round and hits the wall in ~14. With the view 139 deg off, the policy's
  habitual forward/left keys turn the velocity LEFT (round 1's probe: it turns north and slides
  into the pit's north end); the line's right turn is back+left with that view. A spawn-semantics
  defect of the --keys-hold-spawn-duck family: the state carries an artifact of the explorer.
* `tools/spine_canon_view.py` (new): the same spine with yaw = the velocity heading on every row
  with |v_h| >= 100 (the core's absolute-view regime); slower rows and every other field
  unchanged -> autoresearch/spines/pbr2b3_line_a_canon.npy (md5 8de7e89f...).
* pbr2b3_canon1 (02:53 UTC): dft1's launch on the canonical spine, NO temperature (attribution).
* pbr2b3_canon1 result (3.285-3.381B, 02:53-02:59 UTC): with the canonical view the policy LOST
  the post-bounce skill it had learned on the line's own views: [1240,1244] 0.0% (dft1: 78%),
  [1245,1249] 0.2%, then oscillating [1250] 22-60% <-> [1255] 13%. Evals 3.235 / 3.142.
  Round 1 trained 1880 -> 1250 on the explorer's views, and the policy's post-impact behaviour
  is tuned to them.
* Local CPU probes (record_ckpt from single line states, greedy, g2cur6 3.285B; 03:00 UTC): from
  1236 and 1242, with either view, the policy holds RIGHT strafe with its view ~1.5 deg right of
  the velocity (recorded offset +4 = the target set a tick earlier, the heading having turned
  5.6 deg since): that turns the velocity ~5.6 deg/tick (air accel: 30 - |v| cos(91.5 deg) ~ 41
  u/s per tick at 420 u/s). The line turns ~2 deg/tick (-31 -> -55 deg over 1236-1248) into the
  wall bounce. Canonical view: the policy over-turns (heading -96 at +12 ticks vs the line's -55),
  passes south of the corner, falls into the pit (3.17-3.21%). Line view: the view first swings
  ~140 deg at the 10 deg/tick ceiling (14 ticks) with no effective wishdir, the right turn starts
  after the line's impact tick and it flies past the corner (3.03-3.06%). So the band needs a
  WEAKER turn: right strafe with the view ~+2 deg LEFT of the velocity (accel ~15 u/s/tick), a
  3.5 deg change of the yaw command - ~6 sigma at the policy's yaw sigma 0.011 rad (0.6 deg).
  dft1's x20 (12.6 deg sigma) on the line's view was far too wide for that and could not beat
  the swing anyway.
* pbr2b3_canon2 (03:00 UTC): resume canon1's ckpt_latest (3.381B, md5 acfe1d80...) on the
  canonical spine, tau0 1255, + --demo-front-temp 0.5 --demo-front-view 6 --demo-front-ticks 40
  (yaw sigma ~3.8 deg, keys logits / 1.5 for a band spawn's first 0.4 s).
* `--demo-cooldown C` / `--demo-grow-max G` (new, off = the old rule; tests/python/test_demo_pace.py):
  the curriculum's 20-iteration cooldown made every move of tau cost >= 10.5M steps (n / grow x
  10.5M for a whole spine: line_a at grow 5 = ~4B steps even if every band passed at once).
  C spaces the moves C iterations apart; G doubles the step on consecutive advances (S, 2S, ...
  G) and a backoff moves back by S and resets it, so easy stretches cross in a few moves and the
  precise ones still go at S.

### The wall bounce falls (2026-10-24 03:14 UTC): canonical view + band yaw temperature
* pbr2b3_canon2 (canon1's 3.381B weights, canonical spine, --demo-front-temp 0.5
  --demo-front-view 6 --demo-front-ticks 40), 3.381-3.568B: the curriculum re-learned the
  post-bounce bands on the canonical views ([1255] 97/71%, [1250] 17/26%), then crossed the
  pre-impact flight that held round 1 at 0.0-0.4% for 1.5B+ steps: [1245] 26% -> ... ->
  [1230] 49% -> [1225] 56% -> [1220..1215] 57% -> [1210..1175] 69-77%, every advance at the
  20-iteration cooldown (~10.5M steps each, ~17 advances in 186M steps). Greedy evals still the
  pit (3.223 / 3.217 / 3.222): the curriculum is at state ~1170 of 1,881 (the flight after the
  ridge launch).
* What it took (one run, so the two are not separated): the spine's view along the velocity (the
  policy's own view regime) AND a 3.8 deg yaw sigma on band spawns for 0.4 s (the weak turn is a
  ~3.5 deg change of the yaw command). Temperature on the explorer's views (dft1) did nothing.
* pbr2b3_canon3 (03:16 UTC): canon2's ckpt_latest (3.568B, md5 59d2dce4..., pulled) + the same
  flags, tau0 1170, + --demo-cooldown 5 --demo-grow-max 40 (the bands pass at 70%+: the
  cooldown was the pace), 600M steps.
* pbr2b3_canon3 (3.568-3.721B, 03:16-03:28 UTC, --demo-cooldown 5 --demo-grow-max 40): the
  curriculum crossed the WHOLE line in ~150M steps: 1170 -> 1095 (four moves) -> 725 -> ... ->
  285 (bands 51-94%, step 40) -> backoffs at the run-up (290-295: 2.4/11.8%) -> 0 by 3.72B.
  Deciles at the end (success per tenth of the spine): 58 / 57 / 64 / 71 / 44 / 25 / 60 / 99 /
  95 / 89% - the step-40 advances skipped states; the tunnel exit + drop + ridge + launch decile
  [940,1128) is the weakest (25%).
* canon3's 3.644B greedy eval (score 3.233, all 9 still end in the pit region): the line from the
  true start CHANGED - start room, tunnel west at ~240 u/s, exit at (151, -2,888) 255 u/s, the
  air U-turn south-east, the ridge ride east at 546 u/s - but it rides 40 u NORTH of the crest
  (y -2,961 vs the line's -3,035 on the south face) and leaves the ridge's east end LEVEL (vz +25
  vs the line's +350..440 at z ~3,445-3,484), so it lands on the pit floor's east part (899,
  -3,153, 3,314) at ~8 s and wanders there (3.0-3.23%).
* The greedy line joins the spine at the tunnel's east part heading west (line_a state ~780-810);
  states 0-779 are the explorer's run-up back east, which the greedy never flies but which held
  67% of the band draws at tau 0-300. -> autoresearch/spines/pbr2b3_line_a_canon_780.npy (the
  canonical spine from 780).
* pbr2b3_canon4 (03:29 UTC): canon3's ckpt_latest (3.721B, md5 6b1335d2...) on the 780-spine,
  tau0 350 (line 1130, just before the launch), --demo-rate 0.5 (a band must reach 50% before
  tau moves earlier: consolidation, not coverage), the rest as canon3; 600M steps.

### GATE 2 PASSED from the true start: score 3.995 (2026-10-24 03:41 UTC)
* pbr2b3_canon4's first greedy eval (traj_3721920512, 0.5M steps after resuming canon3's final
  3.721B weights): **score 3.995** (incumbent 3.236), best 4.00%, median 3.99%, 9/9 episodes past
  the pit (3.79-4.00%): start room -> tunnel -> air U-turn -> ridge -> launch -> flight -> wall
  bounce -> ramp U-turn -> north side -> west. Then 3.964 (3.797B) and 3.946 (3.873B): faster
  (episodes 7.4-7.7 s instead of 8.7-10.6 s) with the same end.
* Handed-on checkpoint: canon3's ckpt_latest (3,721,396,224, md5 6b1335d2..., the weights canon4
  resumed and evaluated 0.5M steps later), placed as pbr2b3_canon4/ckpt_3721396224.pt so the
  orchestrator's handon_ckpt (newest milestone at or before the best eval) finds it (checked).
* Gate 3 (where all 18 episodes of those evals die): falling west off the north side at
  (480-606, -2,680..-2,706, z ~3,380-3,396), 350-640 u/s, vz -380..-436, into trigger_teleport *209
  (target start = a fall catch; x 451-762, y -3,393..-2,605, z 3,207-3,360: the pit's west part
  under the north path). line_a itself ended there (its last cell, 3.93%, is the deepest GE6 got
  before the same fall). The record crosses here at z ~3,500-3,530 and 700-750 u/s ("a U-turn
  north -> west, then a long drop west"): a speed/height gate, the recipe's kind.
* TRAINING is already past it: canon4's reservoir deepest 5.07% (3.92B) -> 5.42% (3.94B) - spine
  episodes that cross the success line keep flying on the ratchet, and some clear the teleport.
* canon4 later evals: 3.964 / 3.946 / 3.975 / 3.978 (3.80-4.02B): 9/9 past the pit every time,
  9/9 dying at gate 3; the training reservoir's deepest oscillated 5.1-5.4%.
* `--respawn-greedy` and `--respawn-front-perturb` now WORK WITH --demo-file (they were refused as
  "not wired"; no existing run changes): the greedy-death rows take the tail of the curriculum
  pool's non-start rows, the perturbed frontier rows the stretch before it, both out of the demo
  rows' uniform part (after the frontier band's draws). The curriculum's success bonus stays on,
  which matters: without it the ratchet + time-pen hardly separate the gate-2 line from the pit
  (+0.77 progress reward for ~8 s more at 0.001/tick, about the same).
* pbr2b3_canon5 (03:51 UTC): canon4's ckpt_latest (4.017B, md5 2027d47c...) + the curriculum at
  tau 0 (--demo-front-frac 0.4) + --respawn-greedy 0.2 (window 0.25-2 s, stuck 200) +
  --respawn-front-perturb 0.2 (band 0.005, speed x1.0-1.3, turn 90), the recipe's values.
  First iterations: greedy pool 1,575 states from 9/9 gate-3 deaths (7.1-8.5 s), 23% of envs in
  the frontier band, reservoir deepest 5.92%.
* canon5 evals: 3.993 / 3.993 / **4.002** (4.169B; new best, 9/9 to gate 3, deaths at 7.0-7.3 s);
  training's reservoir deepest 5.9% -> 6.7% by 4.23B (frontier-perturb rows explore past it).
* Gate 3 against the record (ANALYSIS ONLY): the record crosses the teleport's west edge (x 451)
  at about (481, -2,662, 3,501) at 729 u/s (energy height z + |v|^2/2g ~3,840) and goes on down
  to (75, -2,869, 3,068) -> (-268, -2,698, 2,839) at 1,239 u/s (a surf drop) -> 7.6% by 10.3 s.
  line_a reaches the north side at ~3,550 (state 1800: z 3,482, |v| 337). Ballistics from
  line_a's end (586, -2,688, 3,475) with vz -250: vx -600 lands at z ~3,387 at x 435 (inside
  the catch, z <= 3,396 with the hull), vx -700 at ~3,402 (clears). A modest energy deficit; the
  ratchet pays +0.6-2.8% within 1-3 s past it, and canon5's greedy-death spawns sit right there.

### Gate 3 by search: the policy's own beam line to 6.6% (2026-10-24 04:20 UTC)
* canon5 evals 3.993 / 3.993 / 4.002 / **4.055** (4.244B) / 4.016 (4.320B): 9/9 die at gate 3.
  At 4.24B several episodes end with vx = 0 at x ~465, z 3,380-3,400: they hit a WALL there and
  drop into the catch - gate 3 is "arrive higher than that wall", not only faster.
* `tools/beam_tas.py` from the true start on canon5's 4.29B weights (4,096 stochastic copies,
  `--greedy-envs 32`, `--score d`, resample every 25 decisions, `--allow-nonfinisher
  --save-frontier`; it refuses `--greedy-prefix` on a --keys-hold checkpoint, and needs
  `--route-file none` off cannonball or it crashes before saving): min d 55,532 (gen 5) -> 53,387
  (gen 10) -> 53,320 = **6.70%** (gen 15) and flat after, 102 s. The frontier lineage, replayed
  exactly: the same start room / tunnel / ridge, but after the launch a FAST east-side U-turn
  (617-639 u/s; no slow wall bounce) onto the north side at z 3,493 and 610 u/s, over the wall at
  x ~465 at z 3,453, a long drop west and down to (8, -2,795, 2,931), a surf at 722 u/s, 6.55% at
  (-766, -2,892, 3,080), then it stops in a hollow at x ~-1,015, z ~2,960-3,020 (the record passes
  there at z ~3,400 and 750+ u/s: gate 4 looks like energy again).
* Spine `autoresearch/spines/pbr2b3_beam_g3b.npy` = that line cut at tick 1,065 (6.62%, its best
  before the stall). The policy's own actions, so its views fit (no canonicalisation needed).
* pbr2b3_beam1 (04:21 UTC): canon5's ckpt_latest (4.317B, md5 ab10e8d0...) + the curriculum on the
  beam spine from its end (success = d(end) + 100 = 6.45%, bonus 50) + the same pace / band
  temperature / greedy / front-perturb flags, --spawn-states-frac 0.2.
* pbr2b3_beam1 evals: 4.076 (4.318B, its resume) / 4.007 (4.393B) / **6.668** (4.469B): 4 of 9
  greedy episodes follow the beam line to the gate-4 hollow (6.50-6.67%, x ~-940..-983, then
  wander there until the eval's end), 2 die at 4.81% in the drop after gate 3 ((165-182, -2,781..
  -2,795, z ~3,160-3,190) falling at vz -650..-690), 1 dies at gate 3 (3.98%), and 2 stay in the
  pit (3.17-3.21%): with the success line moved out to 6.45%, gate 2 is no longer paid by itself
  and erodes a little. The curriculum walked back 1065 -> 590 by 4.54B (deciles 50-90%: 74 / 77 /
  89 / 99 / 100% success), now consolidating the east-side U-turn after the launch.
* Gate 4 against the record (ANALYSIS ONLY): the record surfs the long drop down to z 2,839 at
  1,239 u/s (-268, -2,698), then climbs to z ~3,407 at x -1,083 and goes on west at z ~3,380-
  3,400. The beam line reaches the bottom at 722 u/s, climbs only to z ~3,080 and falls into the
  hollow at x ~-1,000: speed at the bottom of the drop again (surfing the slope, not falling).
* beam1 4.544B eval: **6.680**, median 6.65% (most of 9 reach the gate-4 hollow).
* A second beam (beam1's 4.548B weights, `--skip-gate`: with an episode cap of 300 s the greedy
  warm-up episodes no longer die at 7 s and took >10 min) stalls at the same place: min d 53,323
  (6.69%) from gen 15 to 25. Gate 4 holds for the policy's own proposals.
* Gate 4 is the energy the slow entry never had: energy height h = z + |v|^2/2g at the gate-3 exit
  (x ~366): the beam line ~3,528, the record ~3,884 (analysis only); at the bottom of the drop
  (x ~-300): ~3,186 vs ~3,798 (722 vs 1,239 u/s). The record's surplus comes from the east
  half-pipe pump at the start that the precision entry skips (round 1's analysis).
* GE_g4 (04:53 UTC): explore_phase1 --exact-chains, 64 u x energy 32 cells, 4,096 envs, 16
  threads, rooted at 24 states of the beam line from the north side to the hollow (ticks 600-1100,
  stride 20), 10 min.

### The method so far (pbr2b3, for whoever continues)
1. A searched line (GE --exact-chains + ge_replay, or beam_tas --save-frontier) -> spine.
2. `tools/spine_canon_view.py` when the line came from the uniform explorer (its views are
   arbitrary; a beam line is the policy's own and needs nothing).
3. Backward curriculum on it: `--demo-file S --demo-window 5 --demo-grow 5 --demo-grow-max 40
   --demo-cooldown 5 --demo-front-frac 0.4..0.67 --demo-success-d 100 --demo-success-bonus 50
   --keys-hold-spawn-duck --yaw-entropy 0 --demo-front-temp 0.5 --demo-front-ticks 40
   --demo-front-view 6` (+ `--demo-rate 0.5` to consolidate), with `--respawn-greedy 0.2
   --respawn-front-perturb 0.2` (recipe values) at the gate after the spine's end.
4. When the greedy line reaches the spine's end, beam_tas from the TRUE START with the newest
   weights (`--envs 4096 --resample-every 25 --score d --greedy-envs 32 --allow-nonfinisher
   --save-frontier --skip-gate --route-file none`, ~2 min) -> cut the frontier line at its best
   before the stall -> the next spine (step 3).
Gate 2 needed steps 1-3 (GE line + canonical views + band temperature), gate 3 steps 3-4 (one
beam found the policy's own way over the wall in 100 s).
* beam1 final evals: 6.668 / 6.680 / **6.684** (4.620B, median 6.65%); stopped at 4.666B (ckpt
  md5 627bdc6b..., pulled; the curriculum was at ~370-375, the tunnel exit, after consolidating the
  launch at 34-68% bands).
* GE_g4 result (10 min, 1.23M runs, 25,034 cells): deepest on the grader field d 53,110 = **7.07%**
  at (-1,002, -2,994, 3,240). ge_replay (7/7 links exact): from the beam line's tick 720 (on the
  drop, |v| ~880) it SURFS the drop - 1,058 u/s at the bottom (z 2,878) where the beam line had
  722 - climbs at ~730 u/s to z 3,246 and hits a wall at x ~-1,000 (horizontal speed 733 -> 335 at
  7.07%; the record passes there at z ~3,400). Spine = beam line [0, 720) + that chain, views along
  the velocity -> autoresearch/spines/pbr2b3_beam_ge_g4.npy (909 states).
* Box 57454337 extended to 08:07 UTC (05:08; the 3 h rent was about to expire).
* pbr2b3_beam2 (05:09 UTC): beam1's 4.666B weights + the curriculum on pbr2b3_beam_ge_g4.npy from
  its end, same flags as beam1.
* GE_g5 (12 min, rooted at the new spine's north side -> drop -> climb, 640-909, stride 15):
  deepest 7.079% at (-1,015, -3,001, 3,238) - the same wall at x ~-1,000..-1,015, z ~3,240 that
  ends the GE_g4 chain. The highest energy height west of x -900 in the archive is ~3,631
  (record ~3,775 there, z 3,407 at x -1,083).
* Energy height h = z + |v|^2/2g along the route, policy line vs record (ANALYSIS ONLY): tunnel
  exit ~3,814 vs ~3,978; ridge ~3,787 vs ~3,907; north side ~3,734 vs ~3,842; bottom of the drop
  3,592 (GE line; beam line 3,186) vs ~3,799. A 110-200 u deficit carried from the start: the
  record pumps the east half-pipe first (+230-280 u, round 1's analysis), the precision entry
  starts from a standstill. With this energy the route past x -1,015 is closed to every search so
  far (2 beams, 2 GE runs, ~3M runs); 7.07% looks like this entry's ceiling.
* beam2 (4.666-4.784B): the curriculum came back from the spine's end to [803,807] (the bottom of
  the drop, the start of the climb) and sat there 15 min: that band 40 -> 44% against --demo-rate
  0.5, the next one 70-76%; deciles 80% (727-818) 24%, 90% 91%. Evals 6.686 / 6.658 (the greedy
  still takes the beam line's slow drop into the hollow). Stopped; pbr2b3_beam3 (05:43 UTC) = its
  4.784B weights (md5 18befd7e...), --demo-tau0 805 --demo-rate 0.3, the rest unchanged.
* GE from the TRUE SPAWN with energy cells (13 min of 20, 12,422 cells, 8 threads): no fast
  tunnel entry - the best tunnel cell has h 3,824 at 329 u/s (record ~500-600 u/s, h ~3,950),
  the east half-pipe tops out at h 3,804 (record ~3,930). As in round 1, uniform actions do not
  pump; that route to more energy stays closed to search.
* The x ~-1,015 wall (core traces, ducked hull, west from x -950): solid below z ~3,240 (a wall
  facing east at x -1,015), a rim at z 3,270-3,300 (x -950..-1,000), free above z ~3,330. The GE
  line meets it at z 3,240 with h ~3,583: the energy to clear z 3,330 is there, the trajectory is
  too flat (it reaches the wall moving 730 u/s horizontally). Gate 5 is a launch-angle problem,
  not only energy: about +90 u of height at the wall, or ~+65 u/s at the bottom of the drop.
* beam3 (rate 0.3): evals 6.667 / 6.632 / **7.168** (4.935B) / 7.107 (5.011B, median 7.0%). The
  greedy line now takes the GE chain's surf over the drop: in the 7.168 eval 2 of 9 reach the wall
  at x ~-1,007 at z 3,278-3,291 (the GE line met it at 3,240; the top is ~3,330) and fall back
  into the hollow, 3 reach the hollow (6.55-6.59%), 1 dies in the drop (4.87%), 2 at gate 3, 1
  stalls on the north side. The curriculum is back at the east-side U-turn (~575-590, 25-52%).
* beam_g5a (06:00 UTC): a beam from beam3's ~5.02B weights - can the policy's own sampling clear
  the last 40 u now?
* beam3 final evals 7.107 / 7.156 (5.086B, median 7.1%); stopped at 5.128B (ckpt md5 a312f0da...,
  pulled; the handed-on checkpoint for the 7.168 eval). Curriculum at ~320 (the tunnel exit).
* GE_g6 (rooted at beam_g5a's own climb to the wall, 10 min, 1.23M runs): deepest 7.138% at
  (-1,008, -3,000, 3,274), NO cell west of x -1,030. beam_g5a (d-ranked) frontier 7.13% at 9 s;
  beam_g5dh (`--score dh`, new: rank by -d + K x energy height, K 1) 7.145% at 9 s - the same.
* WHERE the energy goes (beam_g5a / g5dh frontier lines, h = z + |v|^2/2g): ~3,790-3,800 all the
  way down the drop (ticks 700-760; the record has ~3,800 there too), then -114 u in 0.2 s at the
  bottom ((50, -2,809, 2,967) vz -933 -> (-93, -2,704, 2,833) vz -366) and -46 more: the line
  LANDS on the bottom of the drop and the clip eats its vertical component; the record arrives
  there tangentially (vz +31 at 1,239 u/s, h ~3,798). Kept, those ~160 u clear the wall (top
  ~3,330; the line meets it at ~3,270 with h ~3,640). So gate 5 is a landing, not an energy
  deficit from the start.
* pbr2b3_beam4 (06:22 UTC): beam3's 5.128B weights + the same curriculum (tau0 320) +
  **per-call energy shaping** (`--energy-pot 3 --energy-pot-front 0 --energy-pot-front-bonus 0`:
  every loss of h charged when it happens - the landing's -160 u costs ~-0.48 at that tick -
  instead of the recipe's once-at-the-frontier kinetic bonus, which never sees the landing).
* beam4 (per-call energy shaping, 5.128-5.42B): the TRAINING frontier jumped - reservoir deepest
  7.17% -> **12.2%** within ~250M steps - but the greedy evals stayed at the wall (7.113 / 7.171 /
  7.168 / 7.164, medians 7.1-7.2%), and beams from its weights (`--score d`, and `--score dh
  --score-energy 5`; `--eps` is refused on --keys-hold) still stop at 7.1-7.19%. The jump comes
  from the frontier-perturb rows (speed x1.0-1.3 at the frontier band, which sat AT the wall):
  boosted copies of the wall states clear it and the policy has learned the stretch past it from
  there, so the greedy line needs only the passage itself.
* Landing energy across evals (h after the drop's bottom, max z reached at x < -950): beam3
  3,620-3,655 / 3,257-3,291 -> beam4 3,650-3,669 / 3,290-3,296 (wall top ~3,330): per-call
  shaping buys ~+20-30 u of h in ~250M steps; ~40 u to go. The greedy pool already spawns 20% of
  episodes on the greedy line's own drop / landing / climb (2..0.25 s before its best-progress
  stop at ~9.2 s). No searched state (3 GE archives, 4 beams) keeps more than ~3,700 through the
  landing (record ~3,798).
* beam4 later evals: 7.164 / **7.193** (5.431B: max z 3,309 at the wall, the closest yet) / 7.111 /
  7.116 / 7.130; h after the landing went 3,674-3,688 at 5.43B then back to 3,639-3,669 - per-call
  shaping alone plateaus ~20-60 u short. Stopped at 5.694B (ckpt md5 cd33a0d5..., pulled).
* pbr2b3_beam5 (07:09 UTC): beam4's weights + the same flags + `--respawn-greedy-speed 1.0 1.05`
  (the greedy-line rows on the drop / landing / climb get up to +5% speed, so training sees
  passages over the wall from the line's own states).
* `--respawn-greedy-temp T [--respawn-greedy-temp-ticks K --respawn-greedy-temp-view V]` (new,
  off = bit-identical; tests/python/test_respawn_greedy_temp.py): an env whose episode started on a
  greedy-line row samples the keys at 1 + T and the yaw sigma x V for its first K ticks - the
  --demo-front-temp rule (which cracked the gate-2 bounce) on the greedy line's own last seconds
  before it stalls. `GreedyLinePool.is_row` tells those spawns from the rest by origin.
* beam5 (5.694-5.776B, the greedy speed nudge alone): evals 7.144 / 7.157. Replaced after 80M
  steps by pbr2b3_beam6 (07:16 UTC): beam5's weights (md5 97781bd7...) + the speed nudge +
  `--respawn-greedy-temp 1 --respawn-greedy-temp-ticks 60 --respawn-greedy-temp-view 6` (keys at
  2, yaw sigma 0.02 -> 0.12 rad for 0.6 s after a greedy-row spawn on the drop / landing / climb),
  1.2B steps.
* CORRECTION to "gate 5 is a landing" (07:40 UTC): with the vertical speed included, the record's
  h at the top of the landing (8.74 s, (75, -2,869, 3,068), vz -858) is ~3,894 and it also loses
  ~100-140 u over the landing and the curved ride (3,798 at 9.04 s, 3,753 at 9.35 s, ~3,780 at the
  wall). Our newest beam line reaches nearly the same state at tick 714 ((98, -2,874, 3,077), vz
  -842) with h ~3,793 and loses ~165 over the same stretch. So the deficit IS upstream (~100 u:
  the record's start-room pump), and the landing difference is only ~30-60 u. To clear the wall
  (top ~3,330 with ~750 u/s still horizontal) the line needs h ~3,700-3,730 there; the best greedy
  has 3,673-3,690, the record 3,780.
* The policy's line gains +146 u of h from the spawn to the drop by air strafing (the record +240,
  mostly from the pump). One second of optimal air strafing at 700 u/s adds ~60 u of h; with
  --energy-pot 3 that pays ~0.18 against ~1.2 of ratchet progress for the same second.
* beam6 (greedy-row temperature + speed nudge on top of per-call shaping), evals 7.156 / 7.067 /
  7.088 / 7.147: h at the top of the landing unchanged (3,785-3,795), at the wall 3,615-3,673
  (beam4/5: up to 3,690) - not better.
* GE_g7 (07:36 UTC): 32 u cells x energy 16, 29 roots on the newest beam line's north side ->
  descent (ticks 640-780, stride 5), 12 min.
* GE_g7 (32 u x energy 16 cells, 12 min, 270,929 cells, 1.38M runs): deepest **7.202%** at
  (-1,008, -3,000, 3,315) sliding up the wall face (15 u under its top), still NO cell west of
  x -1,030; best h after the landing 3,722. No search of this round gets over the wall.
* The tunnel (greedy beam6 evals): the line enters at x 645 at ~70 u/s (h 3,735; the precision
  entry from the conveyor plate) and leaves at x ~80 at ~390 u/s (h ~3,809) after 2.1 s - ~150
  u/s per second, the optimal air-strafe gain at those speeds (30^2/(2v) per tick); the record
  enters at ~500-600 from its pump. The ~100-160 u energy deficit that ends this route at the x
  -1,015 wall is the start room's: closing it means a pumped start (east half-pipe, then into the
  tunnel at speed) - the long detour against the field that no search found (round 1 and the
  spawn GE above), the planner question of CLAUDE.md 0c.
* beam6 evals 7.156 / 7.067 / 7.088 / 7.147 / 7.121; stopped at 6.124B (ckpt md5 4a2869c2...,
  pulled). pbr2b3_beam7 (07:53 UTC): its weights + beam4's flag set with `--energy-pot 8` (was 3):
  a stronger price on energy so the policy strafes for it in the flight / U-turn / north side,
  where its line now only holds h level.
* beam7 (--energy-pot 8): evals 7.121 / **7.198** (6.200B; one episode at z 3,321 at the wall, 9 u
  under its top) / 7.183 / 7.185 / 7.132. The energy upstream did not move (tunnel exit h
  ~3,806-3,812, top of the landing ~3,800, as with K 3): the policy already gains what this entry
  allows. A beam from its 6.46B weights (64 greedy envs, another torch seed) stops at 7.18%.
* GE_pump2 (08:20 UTC): rooted at the spawn GE's 24 highest-energy states in the east half-pipe
  (x ~2,070-2,170, z ~3,290, ~900 u/s, h ~3,802), 64 u x energy 16 cells, 15 min: can the
  half-pipe give more energy, and does any of it reach the tunnel fast?
* GE_pump2 (15 min, 6.9M runs from the 24 highest-energy half-pipe states): half-pipe h 3,804 ->
  3,839; best tunnel states h ~3,828-3,830 at 340-465 u/s (the greedy leaves the tunnel at
  ~3,809): +20, the record ~3,950. The h ~3,940-3,974 cells next to the spawn are the doorway's
  trigger_push (basevelocity (0, 0, -2,500): pushed DOWN onto the conveyor), not usable energy.
  The pump stays closed to uniform-action search; it is a policy skill (surfing a half-pipe for
  speed) on a long detour against the field.
* The wall's top across y (ducked hull traced west from x -880): at y -3,020..-2,980 (where the
  line meets it, a V notch at x -988..-1,008) open from z 3,330; at y -2,920..-2,860 and
  -3,120..-3,080 open from ~3,320 but solid further east at lower z. No gap: the line is 10-30 u
  short, which matches the energy deficit.

### The x -1,015 wall falls to per-call energy shaping at K 8: score 12.216 (2026-10-24 09:06 UTC)
* beam7's evals (`--energy-pot 8 --energy-pot-front 0 --energy-pot-front-bonus 0`, beam4's
  flags otherwise): 7.121 / 7.198 / 7.183 / 7.185 / 7.132 / 7.163 / 7.197 / 7.184 / 7.197 / 7.184 /
  then **12.216** at 6.955B (median 12.0%): 8 of 9 greedy episodes clear the wall and fly on to
  11.3-12.2% (x -2,900..-3,200, z 2,490-2,690, 12-14 s), the stretch training had learned from
  the frontier-perturb rows' boosted states. ~830M steps of per-call shaping (K 3 then 8) before
  the greedy line found the last ~10-30 u. Checkpoint: beam7's ckpt_latest 7.007B (md5
  a97d0d24..., pulled; the handon pick for that eval) and a copy of the 6.975B one in _ckpts/.
* beam_g8a (beam_tas from beam7's 6.98B weights): frontier **12.52%** at 24 s, stalled from gen
  15. The line loses ~380 u of h in the drop right after the wall ((-1,355, -3,049, 2,736) h 3,635
  -> (-1,939, -2,888, 2,295) h 3,256; the record ~110 there) and reaches the next turn - the
  record's U-turn climb at x ~-3,250..-3,490 up to z ~3,000-3,080 at ~1,000 u/s, h ~3,660-3,710 -
  with h ~3,160-3,200; it stalls at (-3,400, -2,610, 2,712). Gate 6: energy again (~500 u).
* Spine autoresearch/spines/pbr2b3_beam_g8a.npy (1,338 states to 12.34%). pbr2b3_beam8 (09:14 UTC):
  beam7's 7.007B weights + the curriculum on it from its end + the beam7 flag set (energy K 8).
* beam8 (the curriculum on the 12.34% beam line, energy K 8): evals 11.820 / 12.264 / 12.368 /
  12.264 / **13.527** (7.310B, median 12.8%): the greedy line now gets through the 12.5% U-turn
  and every episode dies at z ~2,933 (x -2,989..-3,425) still gaining - a fall catch under the
  record's climb. Stopped at 7.352B (ckpt md5 635d7cf4..., pulled; the handon pick).
* beam_g9a (beam_tas from beam8's 7.33B weights): frontier **15.66%** at 15 s, the population
  extinct at 16 s. The line climbs the U-turn with h ~3,470-3,540 (beam_g8a's had ~3,200) to z
  ~3,120, runs east at 800-900 u/s to 15.36% (14.4 s), hits something at (-1,965, -2,516, 2,991)
  (900 -> 255 u/s; the record passes there at 1,046 u/s) and dies. Spine
  autoresearch/spines/pbr2b3_beam_g9a.npy (1,441 states, cut 5 ticks before that impact, 15.36%).
* pbr2b3_beam9 (09:42 UTC): beam8's 7.352B weights + the curriculum on it, the beam7 flag set.
* beam9 (the curriculum on the 15.36% line): evals 13.815 / 15.660 / 15.706 / **16.056** (7.579B,
  median 15.7%) / 15.912; the curriculum reached tau 0 at 93-98%; training frontier 17.5%.
  Stopped at 7.687B (ckpt md5 e25d10cf..., pulled; the handon pick). One update logged kl 1.66.
* beam_g10a (beam_tas from beam9's 7.67B weights): frontier **18.57%** at 18.5 s, extinct by 28 s.
  The line is fragile past 16%: an impact at (-1,669, -2,362, 2,950) costs ~520 u of h (958 -> 222
  u/s), it crawls east at 300-600 u/s to 18.1%, hits again at (-401, -2,351, 2,626) and creeps to
  18.57% at ~120 u/s. Spine autoresearch/spines/pbr2b3_beam_g10a.npy (1,736 states, cut at 18.26%,
  618 u/s). pbr2b3_beam10 (10:10 UTC): beam9's 7.687B weights + the curriculum on it.
* beam10 (beam9's weights, the curriculum on the 18.26% line): evals 15.946 / 15.423 / 15.339 /
  15.383 / 15.324, medians 15.9 -> 14.8 - a REGRESSION: every greedy episode dies at z ~2,932-2,938
  (the fall catch under the eastward run, x -2,050..-2,250, earlier than beam9's -1,720..-2,000),
  while the curriculum sat at the impact before 16% ([1445] 13%, [1440] 99%). With the success line
  out at 18.1% no episode the greedy can fly is paid the bonus any more, and the earlier route
  erodes (the same effect as the 2/9 pit episodes when the line first moved to 6.45%).
  Lesson: move the success line by one gate at a time, not to the far end of a fragile line.
* pbr2b3_beam11 (10:35 UTC): back to beam9's 7.687B weights (md5 e25d10cf...) with that line cut at
  tick 1,560 (16.68%, just past the catch: autoresearch/spines/pbr2b3_beam_g10a_1560.npy).
* beam11 (beam9's weights, the line cut at 16.68%): evals 15.946 / 15.550 / 15.869 / 15.995 /
  15.944 - no regression, no gain yet; the curriculum sits at the impact before 16% ([1445] 16%,
  [1440] 99%). kl reads 0.2-0.8 on the late runs (it was 0.05-0.15 earlier).

### pbr2b3 round 2 lesson (drafted 10:57 UTC, finalised at wrap-up)
* **Score 3.236 -> 20.414** (best eval pbr2b3_beam12 traj_9837740032, median 20.4%: all 9 greedy
  episodes from the true start reach 19.85-20.41% and die in the next fall catch at z ~2,380). The greedy line: start room -> tunnel ->
  ridge launch -> fast east-side U-turn -> north side -> over the gate-3 wall -> the drop surf ->
  over the x -1,015 wall -> the U-turn climb at x ~-3,300 -> east to the fall catch at ~16%.
* **What did it, in order:**
  1. Gate 2 (round 1's 1,235 wall, held 1.5B+ steps): a GE line's VIEW is the uniform explorer's
     (up to 176 deg off the velocity) - `tools/spine_canon_view.py` puts it along the velocity -
     and the bounce needs a ~3.5 deg yaw change - `--demo-front-temp 0.5 --demo-front-view 6
     --demo-front-ticks 40` (yaw sigma x6 for band spawns' first 0.4 s). 186M steps.
  2. `--demo-cooldown 5 --demo-grow-max 40`: the rest of a 1,881-state line in ~150M steps.
  3. Cut the spine where the greedy line joins it (the explorer's run-up is never flown).
  4. The search -> curriculum LOOP: when the greedy line reaches the spine's end, `beam_tas` from
     the true start on the newest weights (4,096 copies, 32 greedy envs, d-ranked, `--skip-gate
     --save-frontier --route-file none`, ~2 min) -> the policy's own line to the next frontier ->
     cut at its best before the stall -> the next `--demo-file`. Gates 3, 6 (12.5% U-turn) and the
     13% catch fell this way; GE `--exact-chains` from the line's own states supplied the drop surf
     (gate 4, 6.7 -> 7.07%) when the beam stalled.
  5. Per-call energy shaping (`--energy-pot 8 --energy-pot-front 0 --energy-pot-front-bonus 0`)
     carried the line over the energy-limited x -1,015 wall (7.2% -> 12.2%), with the recipe's
     frontier-perturb rows (speed x1.0-1.3 at the training frontier) teaching the stretch beyond it.
  6. `--respawn-greedy` / `--respawn-front-perturb` work with the curriculum now (they keep the
     gate after the spine's end trained).
* **What did not:** a band temperature on the explorer's own views (dft1); canonical views without
  re-learning the post-bounce part (canon1); `--respawn-greedy-temp`, `--respawn-greedy-speed`,
  `--score dh` beams, finer-cell GE at the walls (all at the x -1,015 wall); the spawn / half-pipe
  pump GE (the record's start-room pump, +100-160 u of energy, stays unfound); moving the success
  line to the far end of a fragile line (beam10: the route erodes; move it one gate at a time).
* **Where it stands:** every gate from 7% on has been ENERGY - the record carries ~100-250 u more
  energy height (z + |v|^2/2g) from its start-room pump. The next one is the fall catch at ~16%
  (x -1,700..-2,250, z ~2,900): the line runs east 60-120 u above it at ~900 u/s (record ~1,050).
* The ~16% impact (analysis; both beam lines g10a and the energy-weighted g11dh `--score dh K 2`):
  both reach x ~-1,700 (y ~-2,350, z ~2,940) and lose 500-600 u of h there (g11dh carries ~100 u
  more into it, 3,702 vs 3,594 at 15%, and loses it all). The record (ANALYSIS ONLY) arrives
  higher and further north ((-1,859, -2,404, 3,013) at 1,049 u/s, vz -350, where our line is at
  (-1,879, -2,519, 2,930) at 950) and lands tangentially on a descending ramp at (-1,652, -2,366,
  2,926), surfing it to 1,249 u/s at x -1,184. Ours hits the ramp's edge: a landing geometry again.
* beam11 final evals 16.029 / 16.012 (8.37-8.44B, medians 16.0%): steady, the band before the 16%
  impact at 17% the whole run. Stopped at 8.478B (ckpt md5 d3fb3b96..., pulled).
* GE_g16 (rooted at 25 of the g10a line's states before the 16% impact, 32 u x energy 16 cells,
  12 min): 1.27M cells, deepest **18.86%** (-123, -2,431, 2,510); past the ramp edge it holds h up
  to 3,432 (the beam lines ~3,000). The deepest chain (15/15 links) starts just after the impact
  (h 2,972), goes back and forth on the ramps there for ~6.8 s gaining h to ~3,220 (a small pump -
  the first energy GAIN any uniform search of this round found), then flies east at 640-1,078 u/s.
  Spine autoresearch/spines/pbr2b3_beam_ge_g16.npy (2,381 states: g10a to tick 1,455 + that chain).
* pbr2b3_beam12 (11:44 UTC): beam11's 8.478B weights + the curriculum on it, same flags.
* beam12 (the curriculum on beam line + GE_g16 chain): evals **18.499** (8.479B, 0.5M steps after
  resuming beam11's final weights) / 18.750 / **19.534** (8.630B) / 15.730 / 15.991. In the best
  ones ONE of 9 episodes clears the 16% catch and flies on east (19.53%: (253, -2,308, 2,433) at
  17.2 s, ~470 u below the record's line there), the other 8 die at the catch (15.8-16.0%). The
  curriculum came back through the GE pump stretch and the impact and stalls just before it
  ([1335] 30%, [1340] ~1%); the later evals regressed to the catch.
* Handoff for the 19.534 eval: beam11's final weights (8,478,261,248, md5 d3fb3b96..., the weights
  beam12 resumed and evaluated at 18.499 0.5M steps later) placed as pbr2b3_beam12/ckpt_8478261248.pt,
  the snapshot at or before the eval that handon_ckpt picks (checked) over beam12's regressed
  later ckpt_latest.
* beam12 later evals: 15.9 x5, 18.868 (9.158B), then **20.151 / 20.230 / 20.282 (9.385B) / 20.255 /
  20.194** - from ~9.2B on ALL 9 greedy episodes clear the 16% catch and reach 19.85-20.28%, dying
  at z ~2,380-2,400 (x 477-682): the next fall catch, gate 8. The curriculum's band before the 16%
  impact had sat at ~1% for ~700M steps before the greedy line got through.
* Handoff for 20.282: beam12's ckpt_latest (9.597B, md5 82f90e5b..., pulled; evals around it
  20.19-20.26). The earlier placed 8.478B snapshot was REMOVED: once the run's newest checkpoint
  passes ~9.84B, handon_ckpt's 2x rule would have preferred that worse snapshot.

### pbr2b3 wrap-up (2026-10-24 13:40 UTC)
* Final: **20.414** (pbr2b3_beam12 traj_9837740032, 9.838B; beam12's last evals 20.27-20.41). 108
  greedy evals this round, all valid. Handed-on checkpoint: pbr2b3_beam12/ckpt_9478602752.pt
  (md5 4bbcb8f9..., the trainer's milestone at or before the best eval, pulled; evals around it
  ~20.2) - beam12's ckpt_latest (10.157B, md5 9240e464..., evals 20.36-20.41) is pulled beside it.
* The flag set of the last run (beam12, a resume - every shape-changing flag comes from the
  checkpoint): `--demo-file autoresearch/spines/pbr2b3_beam_ge_g16.npy --demo-window 5 --demo-grow 5
  --demo-grow-max 40 --demo-cooldown 5 --demo-rate 0.3 --demo-front-frac 0.4 --demo-success-d 100
  --demo-success-bonus 50 --keys-hold-spawn-duck --yaw-entropy 0 --demo-front-temp 0.5
  --demo-front-ticks 40 --demo-front-view 6 --respawn-greedy 0.2 --respawn-greedy-window 0.25 2.0
  --respawn-greedy-stuck 200 --respawn-front-perturb 0.2 --respawn-front-perturb-band 0.005
  --respawn-front-perturb-speed 1.0 1.3 --respawn-front-perturb-turn 90 --energy-pot 8
  --energy-pot-front 0 --energy-pot-front-bonus 0 --no-respawn-novel --respawn-backward-mix 0
  --int-real-only 0 --spawn-states autoresearch/spines/pbr1b1_entry_ge_states.npy
  --spawn-states-frac 0.2 --maxvel 3500` (box.py launch --resume --self-states "<the spine README's
  provenance>").
* To continue (the loop of this round, one gate per cycle): beam_tas from the true start on the
  newest weights -> the frontier line past the 20.4% catch -> cut at its best before the stall ->
  the next --demo-file (move the success line ONE gate at a time); if the beam stalls, Go-Explore
  --exact-chains (32 u x energy 16 cells) rooted at the line's own states before the stall and
  splice the deepest chain in. Training's own frontier was ~20.8% at the end.
* Box 57454337 (RTX 5080, 0.60 $/h) released at wrap-up.

## pbr3b2 (round 3, slot 2) - 2026-10-24 19:40 UTC start

### Budget: no box at the start
* `box.py rent` refused on the budget pace: credit $18.46, reserve $10, 48.6 h left -> 0.174 $/h
  allowed, and the only offers passing the query were 5080s at 0.56-0.60 $/h. After a top-up
  (~20:00 UTC, credit $43.45) the pace is 0.69 $/h = ONE box fleet-wide, and another branch
  had it. Retried at every wakeup.
* Local CPU (4 vCPU, shared with the other branches): beam_tas runs at ~820 env-steps/s with 64
  envs - a 4,096-env beam from the true start would take hours here; on the box it is ~2 min.

### --auto-spine: round 2's search -> curriculum loop inside one launch (new, flag off = bit-identical)
Round 2 crossed six gates with one hand-driven loop (beam_tas from the true start on the newest
weights -> cut the frontier line -> the next --demo-file); each cycle waited for an agent wakeup
and a relaunch. `--auto-spine` (python/surfgym/autospine.py, train_fast.py) does it in the trainer:
* after every greedy eval: when `--auto-spine-quorum` (5 of 9) episodes reach the spine's success
  line (min d on the shaping field <= d(spine end) + --demo-success-d), a beam is wanted;
* next iteration: `save_ckpt("autobeam")` and tools/beam_tas.py as a SUBPROCESS (4,096 copies,
  32 greedy, d-ranked, resample 25, `--skip-gate --save-frontier --route-file none`, cap = 2x the
  eval's longest episode + 20 s) - the trainer keeps training;
* when it ends: the frontier line (exact replay) is cut at its minimum d, clocks zeroed, view along
  the velocity (spine_canon_view's rule) -> runs/<run>/autospine_<k>.npy + a provenance sidecar;
  accepted only if d(cut) + M is past both the line in force and the eval's quorum-th best (else
  retry after `--auto-spine-retry` 4 evals with the next proposal seed);
* the swap: a new DemoCurriculum on it (same window/grow/pace kwargs, tau at its end), the success
  line moved, the bonus re-armed, in-flight demo indices dropped, config demo_file = the new spine
  (so a resume continues on it). TRAIN_ONLY in record_ckpt.
* Generic (0b): the policy's own sampling from the true start + the shaping field; no map input.
* The ~20.4% catch (analysis, the record is ANALYSIS ONLY): the incumbent's line tracks the record
  within ~70 u to ~19.9% (x ~180-390, y ~-2,330, z ~2,750-2,800, ~1,200 u/s); there the record turns
  hard SOUTH (vy -717 -> -1,113 at x ~380-440) and surfs down to z 2,414 at (550, -2,976) before a
  U-turn up a south ramp; the policy turns gently (vy -224 -> -838), drifts east to x ~650-700 and
  falls into the catch at z ~2,385. A steering gate, the kind a policy-guided beam has found before.
* Additions (20:25-20:40 UTC): the Go-Explore fallback (`--auto-spine-ge-minutes 10`, default on
  with --auto-spine): a beam line that is not deep enough gets explore_phase1 --exact-chains (32 u x
  energy 16 cells, 4,096 envs, half the CPUs) rooted at 25 states of its last 5 s before its best,
  ge_replay's deepest chain spliced onto it (round 2's GE_g4 / GE_g16 by hand); `--auto-spine-stall
  24`: after 24 evals without the quorum a cycle runs anyway and only has to reach past the greedy's
  own level, so a spine the greedy cannot be carried through is re-cut on the current weights
  (round 2's beam10 -> beam11 by hand). Every hook is guarded: an exception turns the loop off and
  the run goes on with its spine.
* Tests (tests/python/test_auto_spine.py, all pass): 6 unit tests (the cut, the eval reader, accept /
  no-gain backoff / failed beam, the GE splice, the stall re-cut) and 3 CPU trainer tests on
  princessburglar - the curriculum swaps onto a stub beam's line and the checkpoint names it
  (19 min on the loaded droplet), the refusal without --demo-success-d, flag off bit-identical to the
  trainer before the change (config, progress.csv, weights). A real GE + ge_replay smoke through
  AutoSpine (8 roots on round 2's 16.68% line, 20 s, 2 threads): 67,636 cells, chain replayed 2/2
  links, spliced and accepted at 17.77%.
* Launch plan (scratchpad go.sh): rent -> deploy ar/pbr3b2 with the incumbent ckpt (9.479B, md5
  4bbcb8f9...) -> `box.py launch --resume` with round 2's beam12 flag set (NOTES above) +
  `--auto-spine`. The initial spine (pbr2b3_beam_ge_g16, success line ~18.7%) is already passed by
  9/9 greedy episodes, so the first eval wants beam 1 at once.

### Harness issue: the budget pace leaves room for ONE box across four branches (pbr3b2, 21:10 UTC)
`box.py`'s pace rule (burn + new box <= 1.1 x (credit - reserve) / hours left) at credit ~$43,
reserve $10 and ~47 h left allows 0.69 $/h, while every offer that passes `config.offer_query` costs
0.56-0.60 $/h (5080s; no 3090 / 4090 / 5090 passed the query tonight). So exactly one branch can
hold a box at a time and the other three wait (this branch has been blocked since 19:40 UTC; first
$18.46 of credit allowed no box at all). A second box would need credit >= ~$59 at this pace, or a
shorter budget_end_hours. Written down for the operator; nothing is worked around.
* The record gate's audit accepts the new `auto_spine` config key (checked on the incumbent's config
  with the key added) and still refuses unknown keys.

### Gate 8 (the ~20.4% catch), measured locally 22:10-22:40 UTC
* The incumbent's greedy line, dumped as exact states (beam_tas, 2 greedy envs, resample 5:
  runs/pbr3b2/_greedy_inc/beam_frontier.npz, 1,680 states): its LAST CONTACT with geometry is at
  tick 1544 (-940, -2,425, 2,806) = 17.44%; from there it flies 1.35 s in free fall (dvz = -8.0 every
  tick) east to x ~650 and dies in the catch at z ~2,385 (19.9% at tick 1679, 20.4% best eval).
* The record (ANALYSIS ONLY) leaves the same ramp at the same place (last contact (-928, -2,431,
  2,822), 17.47%), flies 1.27 s turning hard SOUTH (|dv_h| 65-100 u/s per 10 ms row, the policy's line
  <= 76) about 50-95 u higher, and lands on a south-going ramp at (480, -2,726, 2,496) = 20.31%, surfs
  it south to (453, -3,257, 2,529), U-turns and climbs (20.0% -> 21.0% by row 2230). The record's line
  stays below the greedy's death depth (20.41%) for ~1 s through that U-turn: a small field deception.
* Gate 8 = a precise 1.3 s air manoeuvre from the last ramp to a landing, steering + height.
* Go-Explore exact chains rooted at 25 states of the greedy line's last 5 s (15 min, 2 threads, 1,024
  envs, 1.2M cells, runs/pbr3b2/_ge_inc/): deepest cell 20.19% at (528, -2,520, 2,401) - still the fall
  into the catch. Nothing past it.
* That run exposed a flaw in the first auto-spine cut (minimum d): it ACCEPTED that line, whose end is
  inside a fatal fall - its success bonus would pay diving deeper into the catch. The cut is now the
  DEEPEST CONTACT (minimum d among states where the map pushed back, pick_selfline's contact rule;
  gravity step = the median per-tick dvz, -8.0 on this line where 68% of ticks are ballistic): the
  greedy line cuts at 1544 (17.44%), the GE splice is rejected, and the GE fallback's roots now lie on
  the approach to the last ramp instead of inside the doomed flight.
* beam_tas `--view-temp V` (new, 1 = off): the sampled envs draw the yaw with sigma x V. The
  incumbent's yaw sigma is 0.013 rad (0.75 deg; --yaw-entropy 0 collapsed it), so the population search
  proposes almost one view per state - the record's turn may need a different view offset. Local test
  running: 256 copies from the true start, V 10.
* Local beam with `--view-temp 10` (256 copies of the incumbent from the true start, 32 greedy, yaw sigma
  0.75 -> 7.5 deg on the sampled envs; 22 min at 331 env-steps/s on the droplet): the population went
  EXTINCT at t = 1700, exactly where the greedy dies (frontier 18.93% at the 16 s resample; med_d within
  ~60 u of min_d all the way - one tight bundle). View noise in the search does not crack gate 8.
* WHY - gate 8 is ENERGY at the takeoff (the record is ANALYSIS ONLY): at the shared last ramp contact
  the record leaves with vz +475, |v| 1,219, h 3,751; the policy with vz +336, |v| 1,133, h 3,609. The
  140 u/s of vz is ~70 u of apex height - the 50-95 u the record flies higher. The deficit was there on
  the ramp before (x ~-1,200: h 3,791 vs 3,646).
* NOT the air strafing: over the 9 greedy episodes' ballistic ticks (62% of all) the policy holds a pure
  side key 98% of the time (no forward), its view within 2.2 deg (median) of the velocity, and gains
  +0.48 u/s of horizontal speed per tick = 78% of the 30^2/(2v) optimum; the record's 10 ms rows show
  the same pattern (side 98%, view 1.0 deg off, +0.42 u/s per row). The ~145 u deficit is carried from
  upstream - mostly the start room (round 2: tunnel exit h ~3,814 vs the record's ~3,978; the record
  pumps the east half-pipe first, the precision entry starts from a standstill) plus landing losses.
  So the remaining route will keep presenting energy gates until the start carries more energy.
* Go-Explore exact chains rooted where the auto loop now roots them (25 states of the greedy line's ticks
  [1044, 1544], the approach to its last ramp contact), 30 min on 1 thread (nice 19), 1.6M cells,
  runs/pbr3b2/_ge_inc2/: deepest cell (101, -2,440, 2,500) = 19.24%, a fall; the deepest CONTACT stays
  1544 (17.44%) -> rejected (correctly). With the earlier 15 min run and the view-temp beam: three
  searches from the policy's own line, nothing lands past the catch. Consistent with the energy
  diagnosis: from the policy's energy the landing is out of reach, so search alone cannot pass gate 8;
  training must find the energy (round 2's x -1,015 wall took ~830M steps of per-call energy shaping)
  or the start must carry more (round 2's GE_pump2 reached the tunnel at h ~3,830 / 340-465 u/s from
  the east half-pipe vs the greedy's h ~3,735 / ~70 u/s entry: +95 u, about what gate 8 lacks).
* Auto loop backoff (new): consecutive failed cycles wait retry x 1, 2, 4, 8 evals before the next beam
  (an accepted spine resets it), so at a training-bound gate the search costs a few % of box time.
* 23:40 UTC: the operator's mid-round harness update (commit 1208385, user 2026-10-25): a flat fleet
  burn cap of 1.2 $/h replaces the pacer, and cheaper cards joined tier 0 under lower caps (3080 / 4070 /
  5070 / 60 Ti class, 0.20-0.30 $/h). A second branch rented at once (burn 0.964 $/h); the only offers
  passing the query at 23:46 were 5080s at 0.621 (storage included), so 0.236 $/h of room fits no card
  yet. The watcher now dry-runs the rent every ~16 min to catch a cheap card.
* The CPU trainer swap test passes on the final auto-spine code (deepest-contact cut, stall, backoff).
* The frame-rate question (00:55 UTC, numbers from the measurements above; the record is ANALYSIS ONLY):
  GoldSrc's air acceleration saturates at 30 u/s per FRAME, and the record was played at 131 fps
  (7.62 ms frames), so its physics allows ~31% more strafe gain per second than our 10 ms tick
  (CLAUDE.md --tick-ms). But measured per 10 ms of flight the policy already gains +0.48 u/s against
  the record's +0.42 (the record's ~51% of its own per-frame optimum, the policy's 78% of ours), so
  the frame rate is not what separates them today: the ~145 u at gate 8 is the start room's pump plus
  landing losses. `--tick-ms-schedule 10:7.63:N` would still raise every flight's gain by up to
  ~31% (an untested lever for ALL the energy gates ahead), at a transfer risk: the start room's
  duck-jump lands on the conveyor plate with ~1 u of margin at 10 ms. Physics parity with how the game
  is played, not a reward term - but a change of the problem the lineage would have to own.
