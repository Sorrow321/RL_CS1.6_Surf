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

## pbr3b4 (round 3, slot 4) - 2026-10-24 19:40 UTC start

Box: 57557523, RTX 5080, 0.563 $/h, machine 153177 (Italy), 32 threads. Rented 19:57 UTC by a
background watcher (`box.py rent` every 3 min) the moment the credit top-up ($18.46 -> $43.45) let
the budget pace (0.69 $/h) admit one box; until then every rent was refused (0.174 $/h allowed).

### The 20.4% catch (gate 8), measured 19:45-20:10 UTC
* The incumbent eval (traj_9837740032): every episode's LAST MAP CONTACT is at ~15.3 s, 17.4%
  ((-930..-968, -2,434..-2,476, z 2,810-2,873), ~1,150 u/s); the 17.4 -> 20.4% is a 1.7 s ballistic
  flight that ends in the fall catch at z ~2,383 (x 530-706). The grader's 20.4 is a fall.
* The record (ANALYSIS ONLY) flies the same gap ~75-145 u higher (z 2,932 at x 0, 2,853 at x 215;
  ours 2,857 / 2,709) and turns from east to SOUTH at x 215-440 by sliding along a curved wall
  (20.75-21.05 s: |v_h| constant at ~1,160, vz falling at exactly g, h ~3,800 held), then lands
  on a south-running ramp at (550, -2,976, 2,414) at 1,450 u/s, U-turns at (442, -3,317) and climbs
  east. Our line arrives lower (h ~3,747 vs ~3,800) and misses the wall.
* beam_tas from the true start on the incumbent's 9.479B weights (4,096 copies, 32 greedy, d-ranked,
  35 s cap, 19 s on the 5080): frontier 20.74% - the survivors crash onto a LEDGE at (661, -2,526,
  2,411) at 16.5 s (1,289 -> 301 u/s, h 3,747 -> 2,467) and hop there to 32 s. A dead end.
* Go-Explore --exact-chains (32 u x energy 16 cells, 4,096 envs, 30 threads, 12 min) rooted at 25
  of that beam line's states over 13.0-16.1 s (before the impact): 2.5M cells, 11.3M runs, best
  only ~530 u past its roots (a 25 s creeping chain). Nothing past the gate.

### --auto-spine: the search -> curriculum loop run by the trainer (new, commit a0166fb)
* `tools/auto_spine.py` = round 2's manual cycle as a rule: beam_tas from the true start on the
  newest weights -> the frontier line TRIMMED AT ITS LAST MAP CONTACT (pick_selfline.contact_cut:
  a d-ranked search's frontier is often a fatal fall; the incumbent's own 20.4 is one) and cut at
  its best d -> if that beats the greedy frontier by --gain: the spine; else Go-Explore
  --exact-chains rooted at the line's last --roots-secs before it STALLS (first tick within
  --stall-margin 500 u of its best d - the ledge crawl would otherwise root GE on a dead end),
  ge_replay of the deepest chain, spliced at its root, views along the velocity, same cut.
* `train_fast --auto-spine` (default off, bit-identical off - tests/python/test_auto_spine.py,
  TRAIN_ONLY in record_ckpt): after a greedy eval with >= --auto-spine-quorum (5) episodes past the
  spine's success line (or --auto-spine-patience steps), it saves ckpt_latest, runs auto_spine.py
  (the trainer waits; ~20 s beam + up to --auto-spine-ge-minutes GE), and a spine that beats the
  greedy frontier (median episode's deepest d) by --auto-spine-gain replaces the curriculum's
  (DemoCurriculum.reset_spine: restart at the new end) and its success line. --auto-spine-delay /
  --auto-spine-retry space the searches. Generic: every constant is a field margin, a time or a
  search budget; the same flags on any map (CLAUDE.md 0b).
* Run pbr3b4_as1 (20:13 UTC): the incumbent's 9.479B weights + round 2's last flag set (spine
  pbr2b3_beam_ge_g16.npy, energy K 8, greedy-death + front-perturb rows) + `--auto-spine
  --auto-spine-delay 4e8 --auto-spine-retry 4e8 --auto-spine-gain 500 --auto-spine-ge-minutes 8`,
  1.5B steps. Gain 500 because the ledge line (20.70%) beats the greedy median (~20.37%) by ~190 u.

### Gate 8 is not energy alone: boosted spawns, tempered beams (20:20-20:33 UTC)
* WON'T vs CAN'T, local CPU: the incumbent greedy policy spawned at its own beam-line state at
  14.0 s (on the U-turn run east, before the ramp ride) with the velocity x1.00 / 1.03 / 1.06 /
  1.10 (h 3,720 -> 3,856, above the record's ~3,800): all four die in the same catch at 20.06-20.33%;
  at x 200 they are at z 2,674-2,726 whatever the boost (the record: 2,853). More energy does not
  change the line's shape: the launch off the ramp at x ~-940 (vz ~+300 for us, ~+430 for the
  record) and the flight under the curved wall stay the same.
* beam_tas `--view-temp V` / `--keys-temp T` (new, off = byte-identical): the --greedy-envs
  proposal copies draw the yaw head at sigma x V (the policy's own is 0.013-0.016 rad) / the keys
  at logits / T. From as1's 9.59B weights: V 6 -> population extinct at 18 s, frontier 20.08% (a
  fall); V 20 -> 18.3% at 20 s. Yaw noise on every decision only adds deaths.
* Go-Explore #2 (64 u x energy 32 cells, 32 roots over the beam line's 11.6-16.15 s, i.e. the
  approach, the ramp ride and the launch; 16 threads beside the trainer): nothing past the gate
  at 6.5 min.
* `--respawn-greedy-contact` (new, off = bit-identical, tests/python/test_respawn_greedy_contact.py,
  TRAIN_ONLY): the greedy-death rows of a failed greedy episode that ENDS IN A BALLISTIC FLIGHT
  (longer than the window's LO) are the HI-LO seconds up to its LAST MAP CONTACT (the launch),
  not the seconds before its death - with the default window all 20% of those spawns sat in the
  doomed 1.7 s fall into the catch, and the front-perturb band sits on the dead-end ledge. The
  contact is read off the vz steps (gravity = the median step of the final fall; no constant).
  On the incumbent's eval: launches at 15.24-15.44 s on 9/9.
* Run pbr3b4_as2 (20:31 UTC): as1's 9.613B weights (md5 fec72708...) + as1's flags +
  `--respawn-greedy-contact --respawn-greedy-temp 0.5 --respawn-greedy-temp-ticks 100
  --respawn-greedy-temp-view 6` (keys at 1.5, yaw sigma x6 for 1 s after a greedy-row spawn: the
  --demo-front-temp rule that cracked gate 2, on the launch window), --auto-spine as before (first
  search at 10.013B). First eval: 8/9 fatal flights anchored at 15.2-15.4 s.
* as1 evals: 20.156 (9.479B, the incumbent's weights) / 20.290 (9.555B).

### Gate 8 geometry (core probes, 20:33-20:40 UTC; the record is ANALYSIS ONLY)
* The room after the 17.4% launch: open x 0-800, y -2,150..-2,600 for z 2,420-2,850, a pillar at
  (320-360, -2,390..-2,430), a slanted ceiling over x >= 320 at z ~2,880-2,900 (normal (-0.23, 0,
  -0.97)), the south wall at y ~-2,630 for x < 400, and the way on a SOUTH opening at x 400-800
  (y -2,630..-3,070) whose surf ramp starts at about (500, -2,770, 2,470). Below z ~2,390 is the
  fall catch; a dead-end ledge sits at (742, -2,700, 2,421).
* The record turns from east to south around the pillar's east side at x 215-444 in 0.3 s with no
  contact (hull traces 48-200 u clear on every side): optimal speed-preserving air strafing turns
  ~60/|v| rad per frame (wishdir 91.5 deg off the velocity, a = 60 u/s), 2.96 deg/frame at 1,160
  u/s - 116 deg in 0.3 s at its 131 fps, ~89 deg at our 100 Hz (the --tick-ms caveat of CLAUDE.md:
  a 100 Hz turn has a ~30% larger radius).
* Our greedy line turns ~0.3 s LATE: its velocity swings south only at x 500-700 (16.5-16.75 s),
  so it reaches the south opening ~150 u too low (z 2,383 at (706, -2,613)) and falls into the
  catch short of the ramp. The launch is also flatter (vz ~+300 vs ~+430 off the ramp at x -940),
  but the speed-boost probe shows energy alone does not move the line.

### Gate 8 needs the record's energy: boosted Go-Explore (20:42-21:05 UTC)
* Branched beam (beam_tas --branch-at t1450:2048 --branch-burst 200 --branch-hold 50
  --branch-jitter 45 --branch-protect 3 --branch-view-only, as2's 9.59B weights): 1,953 forked
  lineages alive at 16 s, 1 at 17 s, extinct at 18 s. View offsets through the launch and the
  turn do not find the corridor ramp.
* Core probe: a hard right air strafe from our own post-launch states either turns too early and
  hits the room's south wall (y ~-2,580, x < 400) or too late and falls into the catch; at 100 Hz
  the turn radius is ~220-440 u at 1,150 u/s.
* Go-Explore #3 (64 u x energy 32 cells, 96 roots = 24 of the beam line's states over 12.5-15.4 s
  x velocity 1.00 / 1.04 / 1.08 / 1.12 - the front-perturb rule; 15 min beside the trainer, 3.8M
  runs): ONLY the x1.12 root just after the launch ((-440, -2,314, 2,885), 1,296 u/s vs the
  policy's 1,157) passes: a sharper right turn at x 464-552, into the corridor at z 2,425 heading
  south, a HARD landing on the corridor's west surf ramp (1,476 -> 962 u/s at (551, -2,781, 2,413)),
  a bounce, and a slow ballistic exit east (570-700 u/s) to 21.26%; its x1.12 sibling at the launch
  point reaches 20.85% in the corridor. x1.08 best 20.26%, x1.04 / x1.00 <= ~20.0%.
* That chain is a dead end too: coasting from its post-ramp flight dies in the next catch at
  z ~2,400 (x 800-1,104) within 0.5-1.3 s. The record exits that ramp at ~1,100 u/s climbing to
  z ~2,900. The policy's energy height runs ~110-200 u under the record's from the start room on
  (round 2's measurements; here 3,651 vs 3,792 at the launch): gate 8 needs it.
* The south corridor (core traces): floor at feet z 2,360 = the catch everywhere; the way on is a
  53 deg surf ramp along its WEST side (x 420-540, feet z 2,390-2,530) from y -2,650 to -3,350.
  The field cannot tell a ramp landing (20.40%) from the catch fall (20.41%): a success line has
  to sit past ~20.6%, in the flight after the ramp.

### Gate 8 by search at x1.04: the spine past the catch (21:02-21:20 UTC)
* Go-Explore #4 with the trainer STOPPED (all 30 threads, 1.8M ticks/s, 15 min, 24M runs; 64 u x
  energy 32 cells; 66 roots = 33 of the beam line's states over 14.0-15.6 s x velocity 1.00 /
  1.04): the x1.04 root at the launch ((-868, -2,456, 2,843), 15.0 s, 1,219 u/s) reaches **21.54%**
  the record's way (ANALYSIS: same manoeuvre, ~100 u lower): a sharper right turn at x 443-549,
  a TANGENTIAL landing on the corridor's west surf ramp at (549, -2,759, 2,403) at 1,391 u/s, south
  along it, a U-turn at (484, -3,443), a climb north-east and east at ~1,000 u/s to (1,103, -2,852,
  2,833). The x1.04 root at x -225: 21.38%. The x1.00 roots: <= 20.58%, every one crashing at
  (540-590, -2,600..-2,700) and crawling (a dead end like the ledge). So the gate needs ~+4% speed
  (~+40 u of energy height) at the launch - not the record's full +140 u.
* That line's end is gate 9: coasting from it hits a wall at x ~1,104 just under its top and
  slides into the next catch at z ~2,405; the record clears it ~100 u higher (z ~2,930 at x 1,208).
* Spine autoresearch/spines/pbr3b4_ge_r3d.npy (1,814 states: the beam line to 15.0 s + that chain,
  tools/ge_spine.py; success line d <= 44,939 = 21.37%). New tools: tools/ge_roots_best.py (per-root
  depth of an exact-chains archive), tools/ge_spine.py.
* Lesson for the GE: focused roots (the 1.6 s around the launch) and the whole CPU found in 2 min
  what 15-min runs with 96 roots spread over 3 s beside the trainer (0.47M ticks/s) never found;
  auto_spine's GE defaults now follow it (64 u x energy 32, roots over the last 2.5 s before the
  stall, x1.0 / 1.04), and the trainer waits while it searches.
* auto_spine `--cut best` is the default now: both lines that pass this gate make their LAST
  CONTACT at the ramp landing (20.40-20.43%, level with the catch fall's 20.41%), and all that makes
  them pass (the U-turn climb to 21.4-21.5%) is ballistic after it - the contact trim would have
  cut them at the landing. The --gain margin (500 u past the greedy median) is what keeps a fall
  from counting.
* as2 evals: 20.336 / 20.366 / 20.279 (9.61-9.76B; stopped at 9.812B, ckpt md5 f929449b...).
* **pbr3b4_as3** (21:20 UTC): as2's 9.812B weights + the curriculum on pbr3b4_ge_r3d.npy (its end)
  + as2's flags (greedy-contact rows, greedy temp, front-perturb, energy K 8, success bonus 50)
  + --auto-spine (delay 4e8, retry 4e8, patience 6e8, gain 500, GE 8 min), 2B steps.

### GATE 8 PASSED by the greedy line: score 21.525 (2026-10-24 21:45 UTC)
* pbr3b4_as3 evals: 20.316 (9.81B, resume) / 20.358 / **21.451** (9.96B, ~150M steps in) / 21.451 /
  21.516 / **21.525** (10.19B): from 9.96B on 8-9 of 9 greedy episodes from the true start fly the
  found manoeuvre (turn into the corridor, ramp landing, U-turn, climb east) and die at 18.5-19.4 s
  at a WALL at x ~1,104 (z 2,777-2,829 there; the record crosses x 1,208 at ~2,930), falling back
  into the catch at (770-800, -2,950..-3,070, 2,385). Gate 9 = clear that wall: ~100 u of height.
* The curriculum crossed the whole chain in ~20 min and then sat exactly at the boosted root:
  band [1498,1502] (the x1.04 root's first states) 56-57%, band [1493,1497] (the policy's own
  unboosted states just before it) 0.0-0.1% for 300M+ steps - and yet the greedy line passed. The
  greedy arrives at the launch with a slightly different (better) state than the stochastic beam
  line the spine was cut from; a spawn on the beam line's own states lacks the energy, the
  deployed policy's line does not.
* Handoff weights pulled: pbr3b4_as3/ckpt_latest.pt at 10,265,559,040 (md5 dda2825f..., 75M after
  the 21.525 eval; evals 9.96-10.19B all 21.45-21.53).
* The trainer's own --auto-spine fired for the first time at 10.266B (9/9 past the 21.37% line):
  beam (11 s, frontier 44,919 = no better than the greedy's 44,879), the branched beam (fork at
  17.02 s = 0.5 s before the greedy's median launch), then Go-Explore (8 min, trainer waiting).

### The first live --auto-spine search, and gate 9 (21:46-21:56 UTC)
* auto-spine search 1 (10.266B, trainer waiting 499 s): beam frontier d 44,926 (no better than the
  greedy median 44,879); the branched beam died on an argparse error (a float --branch-jitter;
  fixed in 99b0ed9 with a test that checks the flags against beam_tas's parser); Go-Explore stage
  (8 min, 40 roots over the beam line's 15.0-17.5 s x 1.0 / 1.04, 7.9M runs): deepest 44,819 at
  (1,104, -2,783, 2,831) - AT the gate-9 wall, not over it. NO new spine; retry at 10.666B.
* Gate 9 geometry (core point probes, x 1,000-1,560 at y -2,780..-2,920): a thin wall at x
  ~1,120-1,160 whose top is at z ~2,860, a floor at z ~2,690 beyond it. A standing hull must
  cross with its origin at z >= ~2,896 (ducked ~2,878); every greedy episode meets it at z
  2,777-2,829 (18.5 s), bounces back and falls into the catch at (770-800, -2,950..-3,070, 2,385).
  The record crosses x 1,208 at ~2,934.
* Gate 8's lesson for the search: the GE needed x1.04 to find the route, the trained greedy then
  flew it with its own energy - the boost buys the uniform explorer the energy its random actions
  waste. So auto_spine's GE is now an ESCALATION LADDER (--ge-boost '1.0,1.04;1.08,1.12', default):
  the second stage of speed factors runs only when the first finds no line past the greedy
  frontier (--ge-minutes each). Test: test_ge_ladder_escalates_until_a_stage_passes.

### The loop runs itself: auto-spine search 2 finds gate 9 with the BRANCHED beam (22:19 UTC)
* auto-spine search 2 (10.719B, 46 s, no GE needed): the plain beam's frontier 44,876 (no better
  than the greedy's 44,854); the branched beam (fork 2,048 of 4,096 copies at 17.0 s = 0.5 s before
  the greedy line's median launch, held view offsets U(-45, 45) deg for 0.5 s each over 2 s, the
  policy's own keys, protected 3 generations) - 324 forked lineages alive at 20 s and already at
  d 43,511 (23.86%) - frontier 43,326 = **24.19%** at 34 s. NEW SPINE (k02, 3,348 states, md5
  5cf733e9...), success line d <= 43,426 (24.0%); the curriculum restarted at its end.
* The k02 line: the corridor U-turn climbs higher than the greedy's (z 2,869 at x 1,015), over the
  gate-9 wall by 19 s ((1,748, -2,870, 2,920) at 912 u/s), then a HARD landing at x ~2,370 (912 ->
  548 u/s, energy height 3,441 -> 2,902) and 13 s of slow hopping on a floor at (2,600-2,820,
  -2,760..-2,960, z ~2,750), 23.9-24.2%: gate 10 (the record surfs a ramp up to z 3,290 at x
  ~3,200 there, ANALYSIS ONLY). No speed boost anywhere: the policy's own proposals from the true
  start, a different view for 2 s at the commitment point.
* Before the swap the old spine's curriculum had reached tau 0 (88% at [0,...]): the band before
  the boosted root, 0.0-0.1% for 300M steps, did open up.
* Box extended to ~02:20 UTC.

### Score 24.291 - gate 9 by the loop; then the beam10 erosion, and milestones (22:28-23:04 UTC)
* as3 evals after the k02 swap: 24.083 (10.794B) / **24.291** (10.870B, median 24.2%): 5-6 of 9
  greedy episodes over the gate-9 wall into the crawl region (24.16-24.29%), 2 at the gate-8
  catch, 1 at the wall, 1 stalled at 5.4%. ~150M steps after the spine swap, no agent in the loop.
  Handoff: as3's ckpt_latest at exactly 10,869,538,816 (the trainer saves it before a search), md5
  302b1a3d...; copied as pbr3b4_as3/ckpt_10869538816.pt so a later ckpt_latest pull cannot move
  the handoff off it (handon_ckpt: a milestone at or before the best eval, checked).
* auto-spine search 3 (10.870B, 1,017 s): beam frontier 24.54% (no gain); GE stage 1 (x1.0/1.04)
  24.76% (no gain); GE stage 2 (x1.08/1.12) -> a chain from the beam line's state at 17.2 s x1.12
  that catches the ramp at x ~2,444 the record's way and surfs on north to 26.66% (k03, success
  line 26.49%). Within 75M steps the greedy evals fell 24.29 -> 21.51 / 21.52 / 21.49: no greedy
  line over the wall any more, and the curriculum stuck INSIDE the boosted chain (bands 0-30% at
  ~1,885). The success line had moved 2.5% past the greedy frontier and the line the greedy
  crossed (24.0%) stopped paying: round 2's beam10 erosion, reproduced.
* Fixes (4807a0b): `--auto-spine-milestones` (on a swap the old success line stays a MILESTONE
  that keeps paying the bonus once per episode on crossing) and `--demo-milestones D,...` (seed
  milestones on a relaunch); train_fast.milestone_pay + tests/python/test_auto_spine.py
  test_milestone_pay_rule. auto_spine's GE ladder is now 1.0,1.04 ; 1.08 ; 1.12 (one factor per
  stage above 1.04: a mixed stage's deepest chain is its most boosted one).
* **pbr3b4_as4** (23:03 UTC): the 24.291 weights (10.8695B, pushed md5-verified) + spine k02
  (runs/spine_k02.npy, success line 43,426 = 24.0%) + --demo-milestones 44939 (k01's line) +
  --auto-spine --auto-spine-milestones (delay 3e8, retry 4e8, patience 6e8, gain 500, GE 8 min) +
  as3's other flags, 2.5B steps. First eval 6/9 past 24.0%; first milestone payment at once.

### as4: milestones hold the route; 24.538 (23:05-23:35 UTC)
* as4 evals (the 24.291 weights + spine k02 + milestone 44,939): 24.272 / 24.350 / 24.405 / 24.197 /
  **24.538** (11.172B, median 24.4%, 8/9 past the 24.0% line). No erosion; one dip (11.10B, 3/9).
  Handoff: as4's ckpt_latest at exactly 11,172,052,992 (saved before the search), md5 0455e4df...,
  pinned as pbr3b4_as4/ckpt_11172052992.pt.
* The milestone smoke test and the flag-off bit-identity test pass (CPU, 6.7 min).
* A flaw found in the trainer's branch tick: an eval episode cut by --eval-patience while hopping
  ends "mid-flight", so its last hop counted as a launch into a fatal flight and as4's search 1
  forked its branched beam at 49.7 s (useless). Fixed in 0835494 (for the next launch): an episode
  whose best progress came > 3 s before its end STALLED and commits at its stall tick (first tick
  within 500 u of its best d); on as4's 11.17B eval the fork moves to 19.07 s, ~1 s before the
  crawl landing.
* as4 search 1 (11.172B): beam frontier 43,214 (24.38%) = no gain; GE stage 1 (x1.0/1.04) no
  line past the frontier; stage 2 (x1.08) running, stage 3 (x1.12) after it if needed.

### as4 search 1: no line past the crawl; as5 with the fixed fork (23:47-23:50 UTC)
* as4 search 1 (1,538 s): beam 43,216 / branched beam (forked at the useless 49.7 s) 43,216 / GE
  x1.0-1.04 43,021 / x1.08 43,024 / x1.12 43,018 - every stage ~160 u past the greedy (43,181),
  short of the 500 u gain. Roots were the beam line's 17.6-20.1 s; as3's x1.12 chain that passed
  gate 10 rooted at 17.2 s (before the gate-9 wall), 2.9 s before that line's stall.
* auto_spine GE roots now span the last 4.0 s before the stall (48 roots), b4e1103.
* The operator's mid-round harness (2b4bfeb): cheaper cards allowed, a flat fleet cap of 1.2 $/h,
  no extend into the $10 reserve. No change to this branch's runs.
* **pbr3b4_as5** (23:49 UTC): as4's 11.191B weights (md5 e71cecf4...) + the same flags, trainer
  0835494+ (the stall-aware branch fork) and auto_spine b4e1103, --auto-spine-delay 1e8, 3B steps.

### Gate 10 (the crawl at 24.2%): a landing, and the k03 curriculum under milestones (00:20-00:37 UTC)
* as5 evals: 24.427 / 24.511 / 24.420 (9/9 greedy episodes past the 24.0% line every time).
  as5 search 1 (1,537 s): beam 43,240 / branched beam (asked 19.1 s, forked at the next resample
  boundary 20.0 s = the crawl landing itself) 43,245 / GE x1.0-1.04 43,006 / x1.08 42,954 / x1.12
  42,964 - no line 500 u past the greedy (43,235).
* Manual branched beams on as5's weights forking EARLIER: at 17 s 43,261, at 18 s 43,133-43,183
  (1,836 forked lineages alive at 20 s, none deeper). auto_spine now runs branched beams at the fork
  tick and 1 s and 2 s before it (--branch-extra-secs 1,2; ~20 s each), aa20918, with a test.
* Gate 10 is a LANDING: the greedy line flies over the gate-9 wall at y ~-2,870 and drifts north,
  landing hard on a floor at (2,372, -2,752) (912 -> 548 u/s, energy height -540 u) and crawling;
  the x1.12 chain (k03) and the record stay ~150-250 u further south (y ~-3,000..-3,115) and catch
  a ramp at x ~2,444 rising to z ~3,234 (vz +649 there), then north at x ~3,600.
* **pbr3b4_as6** (00:36 UTC): as5's 11.377B weights (md5 d14fc257...) + spine k03 (success line
  26.49%) + --demo-milestones 44939,43426 (the k01 and k02 lines: the greedy's current crossings
  keep paying, so the beam10 erosion of as3's k03 phase cannot repeat) + --auto-spine
  --auto-spine-milestones (delay 6e8, patience 6e8), 3B steps. Question: does the curriculum get
  through the boosted chain's ramp-catch band (as3: 0-30% at ~1,885 for ~150M steps) when the
  route is protected, and does the greedy follow?

### as6: milestones stop the erosion; the k03 ramp-catch band learning (00:37-01:06 UTC)
* as6 evals (k03 spine, milestones 44,939 / 43,426): 24.528 / 24.419 / 24.465 / 24.429 / 24.416 /
  24.438 / 24.422 (11.38-11.83B): 9/9 greedy episodes reach the crawl every time - NO erosion,
  where as3's unprotected k03 phase fell to 21.5 within 75M steps. Not past the crawl yet.
* The curriculum walked k03 back to the boosted chain's ramp-catch approach (spine 1,893-1,897,
  ~19.0 s) and that band is rising toward the 30% advance rate: 23.9 -> 25.7 -> 26.1 -> 28.9%.
* pbr3b4_as7 (01:05 UTC): as6's 11.868B weights (md5 b30c3ef6...) + the same flags with
  --auto-spine-patience 1.2e9 (was 6e8): a patience search costs ~25 min while it finds nothing
  and could swap k03 (whose band is about to open) for a similar boosted line; searches still
  fire on the quorum (5/9 greedy episodes past 26.49%).

### as7: 24.555; the ramp-catch band plateaus; a 25-min focused GE (01:05-01:35 UTC)
* as7 evals: 24.539 / 24.548 / 24.555 (12.32B, new best by a hair: the crawl reaches a little
  further) - 9/9 to the crawl every time. Handoff: as7's ckpt_latest at 12,340,166,656 (md5
  158e4e1b..., 19M after the eval; handon_ckpt picks it).
* The k03 ramp-catch band (spine 1,893-1,897) plateaued: 23.6-26.6% over ~500M steps (peak 28.9%
  in as6), under the 30% --demo-rate. Even past it, the greedy would meet the x1.12 root.
* Stopped as7 at 12.340B; a manual auto_spine.py on its weights with ONE long GE stage (25 min,
  x1.0 / 1.04 / 1.08 roots over the last 4 s before the beam line's stall, no branched beam,
  full CPU) - the trainer's 8-min stages never had that budget for an unboosted line.

### The long GE finds nothing past gate 10 below x1.12; as8 (01:35-02:02 UTC)
* Manual auto_spine.py on as7's 12.34B weights, one 25-min GE stage (144 roots = 48 of the beam
  line's states over 15.3-19.3 s x 1.00 / 1.04 / 1.08, full CPU): best 42,988 (24.78%), 235 u past
  the greedy - no ramp catch. Gate 10 has only ever been passed with a x1.12 root (as3 search 3);
  every gate since 7% has pointed at the same thing: the policy's line runs ~100-200 u of energy
  height under the record's from the start room on (the record's half-pipe pump), and the uniform
  explorer needs a boost to find what the record does with that energy.
* **pbr3b4_as8** (02:01 UTC): as7's 12.34B weights + spine k03 + milestones 44,939 / 43,426 +
  --demo-rate 0.2 (Salimans-Chen's rho; the ramp-catch band sat at 24-29% under 0.3) + the rest of
  as7's flags, 5B steps (to the round's end). The milestones keep the score's route paid; the
  question is whether the policy carries the boosted chain's skill back to its own energy, as it
  did at gate 8 (x1.04).

### Where the energy deficit comes from (ANALYSIS ONLY, 02:05 UTC)
Energy height h = z + |v|^2/1600 at the FIRST arrival at each progress level on the grader field,
as7's 24.555 eval (ep0) vs the WR demo:

| progress | t greedy | t record | h greedy | h record | deficit |
|---|---|---|---|---|---|
| 1% | 2.97 | 7.59 | 3,789 | 3,945 | 156 |
| 5% | 7.21 | 11.39 | 3,816 | 3,895 | 79 |
| 10% | 11.05 | 15.25 | 3,624 | 3,651 | 27 |
| 15% | 14.03 | 18.42 | 3,657 | 3,756 | 98 |
| 20% | 16.68 | 20.97 | 3,707 | 3,796 | 89 |
| 21% | 18.16 | 22.29 | 3,354 | 3,583 | 229 |
| 24% | 20.34 | 24.00 | 3,301 | 3,600 | 299 |

* The greedy line is 4.3-4.6 s AHEAD of the record all the way (it skips the half-pipe pump) and
  recovers most of the start room's 156 u by 10% (per-call energy shaping works: 27-47 u at
  10-12%).
* Gate 10's deficit (~220-300 u) is created at GATE 8's corridor landing: from 20% to 21% the
  greedy loses ~353 u (3,707 -> 3,354), the record ~213 (3,796 -> 3,583). Our line lands on the
  corridor's west ramp the way the x1.04 GE chain did (hard: 1,476 -> 962 u/s); the record lands
  tangentially (1,450 u/s, vz -98) and keeps ~140 u more through the U-turn. A smoother landing
  there is worth more for gate 10 than any start-room pump.

### as8: the k03 curriculum reaches its x1.12 root (02:31 UTC)
* With --demo-rate 0.2 the curriculum passed the ramp-catch band and the whole boosted chain and
  now sits at its root: band [1718,1722] (the x1.12 root's states) 62.8%, band [1713,1717] (the
  policy's own unboosted states just before it) 0.0% - gate 8's picture (x1.04 there). Greedy evals
  24.455 / 24.477 / 24.527 / 24.481 / 24.420 / 24.413 / 24.354 (12.34-12.79B): the route holds
  under the milestones; no gate-10 pass yet.

### pbr3b4 round 3 lesson (DRAFT, finalised at wrap-up)
* **Score 20.414 -> 24.555** (as7 12.32B; 9/9 greedy episodes from the true start through gate 8
  (the 20.4% catch) and gate 9 (the 21.5% wall) to the crawl at 24.4-24.6%).
* **What worked:**
  1. Gate 8 by SEARCH with a speed boost: Go-Explore --exact-chains rooted at the policy's own
     launch states x1.04 (the trainer stopped, the whole CPU, roots focused on the 1.6 s around the
     launch) found the record's corridor manoeuvre in 2 min; spine + curriculum; the greedy flew it
     WITHOUT the boost ~150M steps later although the band before the boosted root stayed at 0%.
     The boost buys the uniform explorer the energy its random actions waste.
  2. Gate 9 by THE TRAINER ITSELF: train_fast --auto-spine (tools/auto_spine.py) searched after the
     greedy crossed the gate-8 success line; its branched beam (half the copies forked 0.5 s before
     the greedy's launch, held view offsets for 2 s) found the line over the wall in 46 s and
     swapped it in; the greedy followed in ~150M steps. No agent in that loop.
  3. Milestones (--auto-spine-milestones / --demo-milestones): a spine whose success line jumped
     2.5% (a x1.12 GE root) eroded the greedy from 24.29 to 21.5 in 75M steps - round 2's beam10;
     with the old success lines kept paying, the same spine held the route for 1.5B+ steps.
* **What did not:** beams with yaw temperature (--view-temp 6/20: more deaths), branched beams
  forking too late (at the commitment point itself), GE at x1.00 for gates 8 and 10, the 8-min GE
  ladder stages for gate 10 (x1.0-1.12), one 25-min GE at x1.0-1.08 for gate 10, speed-boosted
  greedy spawns as a fix for gate 8 (energy alone did not move the line), the first auto_spine cut
  rule (trim at the last contact: it cut both passing gate-8 lines at the landing).
* **Where it stands:** gate 10 is a LANDING ~250 u north of the ramp the record catches at x
  ~2,444; only a x1.12 GE root has passed it (spine pbr3b4_auto_k03.npy); the deficit (~250 u of
  energy height vs the record) is made at gate 8's corridor landing (-353 u for us, -213 for the
  record), not in the start room (the greedy recovers that by 10%).

### Gate 10 without a boost: the explorer had found the ramp - pick by energy, not depth (02:34 UTC)
* The 25-min GE archive (runs/_auto_r3f/ge on the box) has UNBOOSTED cells on the ramp the record
  surfs: cell 195108 at (2,854, -2,981, 3,004), 850 u/s, energy height 3,455 (the greedy has ~3,300
  at 24%), whose 93-tick chain starts at the beam line's OWN state at 19.3 s ((2,093, -2,934, 2,831),
  1,002 u/s - after the gate-9 wall, before the crawl landing) and rides UP the ramp (z 2,831 ->
  3,004 at y ~-2,960..-3,060) keeping h ~3,430-3,455 instead of landing on the floor.
* The deepest-cell pick cannot see it: on the ramp at 24.45% the field reads like the crawl
  (24.4-24.6%); only later, up the ramp and north, does the ramp line's progress pull away. Among
  the archive's cells at or past the greedy frontier the right pick was the max-ENERGY one (the
  highest-progress ones there were slow states on a ledge by the crawl).
* GE #7 (02:33 UTC): rooted at 18 of that chain's ramp states, x1.0, 64 u x energy 32, 10 min, 16
  threads beside as8 - to extend the line up the ramp until its progress clears the crawl's.

### Gate 10 needs ~+220 u at the corridor U-turn exit; as9 tests per-call energy K 12 (02:34-02:47 UTC)
* GE #7 (rooted at the unboosted ramp chain's own ramp states, x1.0, 10 min at 16 threads): best
  24.59% at (2,843, -2,774, 3,000) - no way UP the ramp from there. The record climbs it steeply
  (vz +726 at x ~2,480, z 3,290 by x 3,194) with ~150 u more energy than that chain: gate 10 is the
  ramp CLIMB, and it is an energy gate.
* Energy height at the U-turn exit after gate 8's corridor ramp ((625, -3,271, 2,740), where the
  x1.12 root of pbr3b4_auto_k03.npy sits): the beam line it was cut from 3,430, the x1.12 root
  3,607, the record ~3,570-3,610 (ANALYSIS ONLY), the greedy line 3,302-3,383 (as8, 9 episodes).
  Median over evals (autoresearch/analysis/pb_uturn_energy.py, a measurement point only): 3,359
  (as3 10.87B) -> 3,369 (as7 12.32B) -> 3,386 (as8 12.87B): +27 u in ~2B steps at K 8.
* as8 (k03 curriculum, rate 0.2): band at the x1.12 root 62.8%, the band before it 0.0%; evals
  24.35-24.53 (milestones hold the route). Stopped at 12.938B (md5 84f76349...).
* **pbr3b4_as9** (02:46 UTC): as8's weights + the same flags with --energy-pot 12 (was 8; round 2:
  K 3 -> 8 carried the x -1,015 wall after ~800M steps), auto-spine delay 1e9 / patience 1.5e9,
  5B steps (to the round's end). Read-out: the U-turn-exit energy per eval, and any gate-10 pass.

### USER INSIGHT (relayed by the operator, 2026-10-25 ~03:20 UTC): the FIRST ramp
* "At the very start, after start_run, the policy ignores the FIRST ramp - it runs against the
  geodesic field - and jumps straight onto the second ramp (the first ramp the policy uses today).
  A human surfs that first ramp first to gain momentum and only then goes to the second ramp,
  arriving with more speed, and the user expects that extra energy to break some of the gates the
  policy faces later." The user asks for a way to make the policy use that first ramp, under the
  standing rules: CLAUDE.md 0b - nothing hand-placed for this ramp (no reward term, spawn rule,
  target box or spine aimed at it); the mechanism must be generic, one that would also find a
  field-opposed momentum ramp on a map nobody has opened - a search or exploration that finds the
  line on its own, then the usual line-to-curriculum loop; and section 0: the WR demo stays
  analysis only.
* What this lineage already knew about it: the true spawn faces yaw 0 = EAST, toward that ramp (the
  east half-pipe); the record pumps it for ~6 s (energy height 3,653 -> ~3,930-3,950, d rising
  ~1,520 u = 2.7% against the field) and enters the crouch tunnel at ~570 u/s, where our line
  enters at ~70 u/s from a standstill (two precision duck-jumps). Uniform-action Go-Explore from
  the spawn never pumped (round 2: half-pipe h max 3,804 -> 3,839, tunnel at <= 465 u/s).
* Measured this round (energy at matched progress, ANALYSIS ONLY): our line is 156 u of energy
  height under the record at 1%, but only 27-47 u at 10-12% (the record's own h falls 3,945 ->
  3,651 by 10%); the gate-10 deficit (~250 u) is made at gate 8's corridor landing. So the pump's
  energy does not simply carry for the record either - but a pumped start also changes every
  line downstream (more speed into gates 2-8), which is the user's point and is worth testing.
* Generic mechanism tried (below): an ENERGY-ranked, branched beam from the true start - half or
  more of the policy's own copies forked at tick 1 with random view commands (some keep facing
  east; the keys stay the policy's own), the population ranked by -d + K x energy height
  (beam_tas --score dh) instead of -d alone, so lineages that gain energy AGAINST the field are
  not culled. Nothing in it names the ramp: on any map it asks "is there a start that buys
  energy the field-follower never collects?"

### The first ramp: what the generic searches did (03:23-03:52 UTC)
* Energy-ranked branched beams from the true start (beam_tas --score dh --score-energy 3/8/15,
  3,072 of 4,096 copies forked with random view commands, protected up to 5 s): the fork lands on
  the first resample boundary (1.0 s at --resample-every 25, by then every copy has turned west),
  and at --resample-every 5 (fork at 0.2 s) the forked copies still walk the normal west line -
  view-only branching cannot turn a keys-hold policy around (its keys and habit dominate).
* `beam_tas --start-state NPY[:I]` (new, off = byte-identical): the population starts from any
  state - the explorer finds WHERE, the policy's own proposals find HOW. `tools/ge_energy_picks.py`
  (new): an exact-chains archive's VIABLE cells (some cell's parent - energy alone picked states
  plunging into a death or riding the spawn doorway's push-down trigger) BEHIND the start on the
  field (d above the spawn's), ranked by energy height, with their chains.
* GE from the true spawn (64 u x energy 32, 6 min, 16 threads): the field-opposed side's best
  viable energy is only +136..+160 u over the spawn (3,789-3,813) - uniform actions do not pump
  (as in rounds 1-2) - and from every half-pipe pick ALL 4,096 policy copies died within 1 s (the
  policy has never surfed there; round 1's half-pipe deaths at z ~3,290-3,310). The near-spawn pick
  only replays the normal line, later.
* So the pump needs the POLICY to practise the half-pipe (surfing it for speed is a skill, not a
  search result). **pbr3b4_as11** (03:52 UTC): as10's 13.661B weights + the k02 curriculum +
  milestone 44,939 + --energy-pot 12 (per-call: every energy gain in the half-pipe pays) +
  --spawn-states runs/spawn_both.npy at 0.25 = the start-room entry states + the 1,891 viable
  explorer cells BEHIND the start (both directions, nothing named) - does the greedy start
  pumping, and does it come back with more energy? (Round 1 saw half-pipe spawns teach a weak
  surfer to go east and die there; this policy surfs at 1,000+ u/s.)
* as10 evals: 24.737 (13.500B = as9's 13.4999B weights, pinned as pbr3b4_as10/ckpt_13499891712.pt)
  / 24.548 / 24.733; U-turn-exit energy 3,348-3,386 (flat under K 12).

### as11: no pumping; the explorer's half-pipe states were doomed; as12 (04:21-04:32 UTC)
* as11 (both-direction explorer spawns 0.25, K 12; 13.66-14.21B): greedy evals 24.535-24.623, and
  0/9 episodes ever go east of the spawn (pb_start_energy.py), tunnel exit h 3,807-3,824 at
  3.6-3.9 s (the record ~3,969 at 8.24 s, ANALYSIS ONLY) - the start did not change.
* Why: the "viable" field-opposed cells were mostly DOOMED - from each half-pipe pick 0 / 1,024
  runs survive 1 s with neutral or uniform random actions (falling into what looks like a kill
  volume at the half-pipe's bottom, ~100 u under the record's line there), so those spawns were
  instant deaths. A cell with children is not a survivable cell: a dying run still visits new
  cells on the way down.
* Generic survivability filter: keep the behind-the-start cells from which >= 25% of 32 uniform
  random rollouts are alive after 1 s - 1,917 of 8,922 cells, x 863-2,380 (the half-pipe region),
  energy median 3,649 / max 3,815. runs/spawn_both2.npy = the start-room entry states + those.
* **pbr3b4_as12** (04:31 UTC): as11's 14.217B weights (md5 bf1b76a5...) + the same flags with
  --spawn-states runs/spawn_both2.npy (0.25), 4B steps (to the round's end).

### The first ramp: where it stands at the end of round 3 (05:04 UTC)
* as12 (survivable both-direction explorer spawns 0.25, K 12; 14.22-14.72B): greedy evals
  24.49-24.60, still 0/9 east of the spawn, tunnel exit h 3,812-3,820 at 3.6-3.9 s.
* Skill probe (beam_tas --start-state from the 3 most energetic survivable half-pipe cells, as12's
  14.72B weights, dh K 8, 8 s): the policy now SURVIVES there (253-1,809 of 4,096 copies alive at
  6-7 s, where as11's weights died to the last copy within 1 s), but it does not pump: the lines
  lose energy (3,793 -> 3,541 and similar) climbing slowly back up toward the spawn (70-409 u/s).
* Why the greedy will not take the detour under this objective (an estimate): a ~6 s detour
  against the field delays everything after it - the ratchet's progress, and the milestone
  bonuses at 21.37% / 24.0% (+50 each at ~18-20 s, ~5 units of discounted value each lost to a 6 s
  delay under gamma 0.9995/tick) - for ~3-4 units of energy shaping during the pump, much of it
  charged back at later landings; and the only gate the extra energy could unlock (gate 10, ~20 s
  in) sits at the edge of the 20 s discount horizon.
* So, for whoever continues the user's first-ramp idea (all generic, nothing aimed at the ramp):
  1. the SKILL - spawns on the explorer's own SURVIVABLE states behind the start (the as12 file's
     construction, tools/ge_energy_picks.py's rollout filter) with per-call energy shaping, until
     a --start-state beam from those states shows lineages GAINING energy;
  2. a LINE on which the pump is necessary - a search from the true start whose early ranking is
     energy (beam_tas --score dh, or GE picked by energy) carried on past gate 10, so that the
     curriculum's success line (past gate 10) is reachable only with the pumped energy; then the
     usual curriculum, with --demo-milestones keeping the current route paid;
  3. view-only branching cannot turn a keys-hold policy around at the spawn (tried: K 3/8/15,
     forks at 0.2 s and 1 s) - forcing the first decisions needs the keys too, or a spawn-side
     exploration like (1).
* as12 later evals (14.82-15.20B): 20.446 / 20.538 (two evals back at the gate-8 catch - a transient
  the milestones pulled back) / 24.551 / 24.644 / 24.688 / 24.591; still 0/9 east of the spawn,
  tunnel exit h ~3,815. The half-pipe practice disturbs the route a little and has not yet
  changed the start.
* Final skill probe (as12's 15.82B weights, the same three survivable half-pipe states, dh K 8,
  8 s): from one of them the policy now flies back WEST out of the half-pipe and reaches the crouch
  tunnel's mouth ((629, -2,879, 3,734) at 7 s, 0.52% past the start on the field) keeping its energy
  (h 3,772 -> 3,758; our greedy enters that tunnel at ~3,735, the record at ~3,926-3,943) but slowly
  (199 u/s); from the other two it still loses energy climbing the west wall. ~1.6B steps of
  survivable field-opposed spawns taught it to survive the half-pipe and to come back onto its
  route; pumping (gaining energy there) has not appeared yet.
* as12 evals to 15.80B: 24.52-24.70, 0/9 east of the spawn, tunnel exit h 3,812-3,816.

### pbr3b4 round 3 lesson (wrap-up, 2026-10-25 07:01 UTC)
* **Score 20.414 -> 24.737** (best eval pbr3b4_as10 traj_13500416000, median 24.5%; 108 valid
  greedy evals this round). Handed-on checkpoint: pbr3b4_as10/ckpt_13499891712.pt (md5
  a3a21331..., the exact weights that eval ran - as9's ckpt_latest, resumed by as10; handon_ckpt
  checked); as10's ckpt_latest (13.661B, md5 e988a96f...) pulled beside it. All 9 greedy episodes
  from the true start pass gate 8 (the 20.4% catch) and gate 9 (the 21.5% wall) and crawl on a
  floor at 24.4-24.7% (gate 10).
* **What worked, in order:**
  1. Gate 8 by SEARCH with a speed boost: Go-Explore --exact-chains rooted at the policy's own
     launch states x1.04, the trainer stopped and the CPU whole, roots focused on 1.6 s around the
     launch (2 min to the record's corridor manoeuvre; 15-min runs with 96 roots spread wide beside
     the trainer never found it). Spine + curriculum: the greedy flew it WITHOUT the boost ~150M
     steps later. The boost buys the uniform explorer the energy its random actions waste.
  2. Gate 9 by THE TRAINER ITSELF: train_fast --auto-spine (tools/auto_spine.py: beam ->
     branched beams -> Go-Explore escalation ladder -> spine swap, run after a quorum of greedy
     episodes crosses the spine's success line): its branched beam found the line over the wall in
     46 s and swapped it in; the greedy followed in ~150M steps.
  3. Milestones (--auto-spine-milestones / --demo-milestones): a spine whose success line jumped
     2.5% (a x1.12 root) eroded the greedy 24.29 -> 21.5 in 75M steps (round 2's beam10 again);
     with the old success lines paying, the same spine held the route for 3B+ steps.
  4. Cheap generic fixes found by running: the branched beam's fork at an episode's stall when
     the patience cut ended it mid-hop; branched beams at the fork tick and 1 s / 2 s before it;
     auto_spine's --cut best (the contact trim cut both passing gate-8 lines at the landing); GE
     picks by energy and by rollout SURVIVABILITY (a cell with children can still be doomed).
* **What did not:** yaw temperature on beam proposals; branched beams forking at the commitment
  point itself; GE at x1.00 for gates 8 and 10, and every GE stage up to x1.08 (8 and 25 min) for
  gate 10; speed-boosted greedy spawns at gate 8; K 12 energy shaping (no gain at the gate-8 U-turn
  exit in ~1.5B steps); the boosted k03 curriculum (its band before the x1.12 root stays at 0%).
* **Gate 10** (where it stands): a ramp CLIMB at x ~2,444-3,103 after the crawl landing. The greedy
  reaches the corridor U-turn exit with energy height ~3,340-3,390; the x1.12 chain that passes
  gate 10 had 3,607, the record ~3,600 (ANALYSIS). The deficit is made at gate 8's corridor
  landing (-353 u for us, -213 for the record), not in the start room (the greedy recovers the
  start room's 156 u by 10%). An unboosted GE chain does reach the ramp (picked by energy, not
  depth - on the ramp the field reads like the crawl) but cannot climb it.
* **The user's first-ramp insight** (2026-10-25): generic attempts and where they stand are in the
  sections above - the policy now survives the half-pipe and can fly back to the tunnel from it
  (survivable field-opposed explorer spawns, ~1.6B steps), but it does not pump, and under this
  objective a 6 s field-opposed detour does not pay unless its energy unlocks a gate within the
  ~20 s horizon. Next: the skill (spawns there until a --start-state beam GAINS energy), then a
  line where the pumped energy is necessary past gate 10, distilled with milestones on.
* **To continue** (the flag set of the best run, a resume - shape flags come from the checkpoint):
  `box.py launch --resume --self-states "<spines README provenance>" -- --maxvel 3500
  --demo-file autoresearch/spines/pbr3b4_auto_k02.npy --demo-window 5 --demo-grow 5
  --demo-grow-max 40 --demo-cooldown 5 --demo-rate 0.3 --demo-front-frac 0.4 --demo-success-d 100
  --demo-success-bonus 50 --demo-milestones 44939 --keys-hold-spawn-duck --yaw-entropy 0
  --demo-front-temp 0.5 --demo-front-ticks 40 --demo-front-view 6 --respawn-greedy 0.2
  --respawn-greedy-window 0.25 2.0 --respawn-greedy-stuck 200 --respawn-greedy-contact
  --respawn-greedy-temp 0.5 --respawn-greedy-temp-ticks 100 --respawn-greedy-temp-view 6
  --respawn-front-perturb 0.2 --respawn-front-perturb-band 0.005 --respawn-front-perturb-speed
  1.0 1.3 --respawn-front-perturb-turn 90 --energy-pot 12 --energy-pot-front 0
  --energy-pot-front-bonus 0 --no-respawn-novel --respawn-backward-mix 0 --int-real-only 0
  --spawn-states autoresearch/spines/pbr1b1_entry_ge_states.npy --spawn-states-frac 0.2`
  (+ `--auto-spine --auto-spine-milestones --auto-spine-gain 500 --auto-spine-ge-minutes 8` to let
  the trainer search; the as10 run had it off). Box 57557523 (RTX 5080, 0.56 $/h), released at
  wrap-up.

## pbr4b3 (round 4, slot 3) - 2026-10-25 07:43 UTC start

Box: 57632609, RTX 5080 + Threadripper 3970X (32 thr), 0.363 $/h, Ireland (machine 145348).
Incumbent: pbr3b4_as10 traj_13500416000 (24.737), weights 13.4999B (md5 a3a21331...).

### Where the gate-10 energy goes, tick by tick (incumbent eval ep0, 24.74%; ANALYSIS of our own line)
* 16.95 -> 17.00 s: the gate-8 corridor ramp landing, ONE impact: h 3,710 -> 3,553 (-157 u),
  v (207, -1,090, -850) -> (451, -1,130, -532) - it meets the ramp falling at vz -850.
* 17.00 -> 17.60 s: riding the ramp south and U-turning at the corridor's end: a steady -3..-5 u
  per tick, 3,553 -> 3,364 (-190 u) - a curved surface, the hull losing a little at every facet.
* 18.0 -> 19.45 s (U-turn exit -> over the gate-9 wall): the policy GAINS +94 u by air strafing
  (3,382 -> 3,483); the x1.12 GE chain of spine k03 LOSES 67 u over the same stretch (uniform
  random actions waste energy: the boost paid for the explorer's waste, not for a lack in the
  policy's flying).
* 19.45 -> 19.90 s: the gate-10 ramp catch: -68 u in one tick (vz -523 -> -278) then -68 more;
  k03 meets the same surface at 18.5 s rising (vz +65) and loses 38 u. ep0 climbs to z 3,084 at
  x 2,835 (k03: 3,187 there) and falls back (24.74%).
* So the line needs ~+60-100 u at 23% AND a smoother ramp catch; the biggest single sink is the
  gate-8 corridor landing + U-turn (-347 u in 0.65 s; the record loses ~213 over 20-21%).

### GATE 10 IS IN THE POLICY'S OWN DISTRIBUTION: a plain beam on the incumbent climbs the ramp (08:00 UTC)
* beam_tas from the true start on the incumbent's 13.4999B weights (4,096 stochastic copies, 32
  greedy, d-ranked, resample every 25 decisions, half the population forked at 16.0 s with held
  view offsets - every forked lineage died by 20 s, so the line below is the policy's own
  unbranched proposals; 26 s cap, 208 s on this box with the trainer paused): frontier **d 42,078
  = 26.36%** at 24 s, replayed exactly. The same beam ranked by -d + 3 x energy height (`--score dh
  --score-energy 3`): 42,074 at 25 s (more survivors, 2,696 vs 1,481 of 4,096) - the ranking is
  not what was missing.
* The line: the usual route to the gate-8 corridor (h 3,709 at 16.6 s; landing -166 u at 16.8 s;
  the corridor U-turn to h 3,384 at 17.4 s), strafing back up to h 3,486 by 19.0 s, then a
  SMOOTH ramp catch at (2,141, -3,115, 2,750) (v (1,019, -40, -62): -84 u; the greedy ep0 met the
  same surface at vz -523 and lost 136) and UP the ramp: z 2,785 -> 3,204 over 19.4-20.8 s, to
  (3,041, -2,903, 3,191) at 640 u/s = 25.06%; a hard landing on the floor beyond (z 3,197, h 3,447
  -> 3,284, onground), then walking / hopping north at 100-440 u/s to (3,468, -2,331) = 26.36%.
* So round 3's last searches (as4/as5 on 11.17-11.38B weights: beams 43,2xx) were BEFORE the
  policy learned to fly this; the 2B steps of energy K 12 + the k03 curriculum since then moved
  the policy's distribution to the ramp, and nobody ran a beam on the newer weights (as10-as12
  had --auto-spine off). Energy at 23% on this line ~3,480 - the greedy's best episodes have the
  same; the difference is the catch angle, a skill the stochastic policy now has.
* Spine autoresearch/spines/pbr4b3_beam_g10.npy (2,400 states, md5 7fa685aa...; success line
  d <= 42,182 = 26.19%).
* **pbr4b3_g10a** (08:03 UTC): the incumbent's weights + the curriculum on that spine from its end
  + `--demo-milestones 44939,43426` (the k01 / k02 lines keep paying) + round 3's flag set (energy
  K 12, greedy-contact rows with temperature, front-perturb) + `--auto-spine --auto-spine-milestones
  --auto-spine-gain 500 --auto-spine-ge-minutes 8 --auto-spine-retry 4e8 --auto-spine-patience
  1.2e9` (the trainer searches the next gate itself once 5/9 greedy episodes cross 26.19%). 4B
  steps. pbr4b3_c0 (the plain continuation) was stopped before its first eval.

### A 64-thread host under a 30.7-CPU quota ran the trainer at 7k steps/s: numba's pool (08:16-08:25 UTC)
* pbr4b3_g10a ran at **~7,000 steps/s** (expected ~250k): 5.8M steps in 13.5 min, load average
  ~100, 60 runnable threads at ~46% each, GPU 0% busy. The box shows 64 CPUs (Threadripper 3970X,
  SMT) but its cgroup quota is 30.72; the trainer's OMP team is quota-aware (omp=15), but numba
  sizes its parallel pool to every VISIBLE core: the goal field's fused prange sampler
  (surfgym/goalfield.py, called on every env every tick for the race reward) ran a 64-thread
  spinning OpenMP team (numba's omppool, its own libgomp) beside the core's 15 and torch's 15, and
  the quota throttled them all. `taskset` to 30 cores did not help (the teams still spin). The
  beams of 07:55-08:02 ran at 51-78k env-steps/s for the same reason (round 3's 5080 did ~750k).
* Fix (ad5f905): `import surfgym` defaults NUMBA_NUM_THREADS to OMP_NUM_THREADS (train_fast sets
  it first), else half the quota-aware CPUs (4..32), before goalfield imports numba. The sampler
  is element-wise: results unchanged (tests/python/test_numba_threads.py: 1 vs 4 threads
  bit-identical, the env default, an explicit setting kept). On round 3's 32-thread boxes numba
  had 32 threads - over-subscribed too, so this may be worth some throughput there as well.
* g10a was stopped (6M steps, the curriculum had moved to [2384,2399] at 100%) and relaunched
  unchanged as **pbr4b3_g10b** (08:25 UTC) on the fixed commit.

### Harness issue: offers whose visible threads exceed the container's CPU quota
`config.offer_query`'s `cpu_cores_effective>=32` admits a 64-thread host whose container quota is
30.7 CPUs (box 57632609). Anything that sizes a thread pool from the visible core count (numba,
torch's default intra-op pool if OMP_NUM_THREADS is unset, a tool's own `os.cpu_count()`) then
over-subscribes the quota. This branch's surfgym now caps numba; the operator may want a
`vast_pick`/`gpu_health` probe that runs the trainer's first iterations and reads steps/s.

### GATE 10 PASSED by the greedy line: 26.357 -> 26.362 (2026-10-25 08:41 UTC)
* pbr4b3_g10b evals (each 75M steps, ~4 min at 350k steps/s): 24.667 / 24.746 / 24.575 / **26.357**
  (13.727B, ~230M steps after the resume: 2/9 greedy episodes climb the ramp to 26.36%) / 26.323 /
  26.338 / **26.362** (13.953B) / 26.342 (14.029B, 3/9). The other episodes still crawl at gate 10
  (24.1-24.8%). The curriculum walked the spine back 2399 -> ~1800 (18.0 s, the corridor U-turn exit)
  in ~150M steps and oscillates there (band [1799,1803] 16-24% against the 0.3 rate).
* Every passing episode - and the beam line itself - DIES at (3,470, -2,340, z 3,197) at 26.34-26.36%:
  gate 11. trigger_teleport *35 (target start = a fall catch) is a thin sheet at z 3,161-3,169 over
  x 3,111-3,840, y -2,481..-1,277, flush with the floor (standing origin 3,197, feet 3,161): walking
  north off the floor's south part is death. Hull traces (standing, downward): a flat floor at origin
  3,197 everywhere, an L-shaped surf ridge (steep faces) rising to ~3,330 along y ~-3,000 (x 3,240-3,420)
  and running north at x ~3,480-3,600 to y ~-2,650, solid blocks at x >= 3,540, y -2,600..-2,500 and
  x <= 3,120, y >= -2,600.
* The record (ANALYSIS ONLY): flies off the gate-10 ramp top at (3,024, -2,902, 3,269) at 811 u/s
  (energy height 3,680), lands tangentially on the ridge's west face at x ~3,454-3,510, surfs it north
  at ~900 u/s (contact) and launches off its north end at (3,521, -2,508, 3,214) with vz +320..+490 in a
  ~1.5 s arc over the sheet (apex z 3,418) to y ~-1,700. Our line leaves the ramp top 80 u lower at
  640-700 u/s (h ~3,510), lands FLAT on the floor at x ~3,140 (-160 u) and walks north at ~250 u/s
  (h ~3,236 at 26%, the record 3,702): gate 11 is energy again, ~170 u short at the ramp top.
* Searches on g10b's 13.87B weights (beam_tas 4,096 copies, 30-32 s caps, ~25 s each now):
  d-ranked 42,081 / dh K 3 42,080 / dh K 8 42,085 - all at the sheet, energy profiles within 20-60 u
  of each other (10% 3,619-3,655, 21% 3,454-3,460, 25% 3,361-3,532): energy-weighted ranking does not
  buy energy, the policy's proposals all fly the same way. Branched beams (half forked at 19, 20, 21 s,
  held view offsets +-45 / +-90 deg for 0.5 s over 2 s): 42,081-42,084, every forked lineage dead.
* GE11 (08:52 UTC): explore_phase1 --exact-chains rooted at 24 of the newest beam line's
  states over the gate-10 ramp ride (19.0-21.0 s) x speed 1.0 / 1.1 / 1.2 / 1.3, 64 u x energy 32 cells,
  10 min at 12 threads beside the trainer: which boost crosses the sheet?
* GE11 result (10 min, 3.9M runs, 125,908 cells): best progress per speed factor of the ramp roots -
  x1.0 **26.07%** (nothing over the sheet), x1.1 26.54%, x1.2 26.95%, x1.3 **27.42%**. The x1.3 chain
  (ge_replay 9/9 exact) from the ramp's foot (2,445, -3,025, 2,881) at 1,196 u/s (h 3,775): higher off
  the ramp top (z 3,423), onto the ridge at (3,539, -2,852), surfs it north, launches at y -2,452
  (z 3,334, vz +286) and flies over the sheet to (3,392, -1,764, 3,188) - still ~200 u short of the
  way on.
* Gate-11 room (hull traces from below the 3,677 ceiling): x 3,180-3,780, y -2,550..-1,350, its floor
  at origin 3,197 lethal north of y -2,481 (the sheet), a pillar at x 3,180-3,420, y ~-2,220, and the
  ONLY way on a hole in the floor at x 3,240-3,660, y -1,425..-1,575 (drops to 2,612) - ~1,000 u of
  flight from the ridge's north end. Ballistics: launching at z ~3,330 with vz +500 gives ~1.5 s of air
  under the ceiling, so ~680 u/s horizontal (|v| ~840 at the launch). The policy reaches the ridge
  walking; it needs ~+250-350 u of energy height at the ramp (24%: ours 3,405-3,437, x1.3 root 3,775,
  the record ~3,600 with a better line).
* Where that much energy can come from (a hypothesis, being tested): our policy air-strafes at the
  per-frame optimum (+0.56 u of energy height per tick in free flight, measured on ep0 18.0-19.45 s:
  +94 u in 1.45 s) - and that optimum is PER FRAME (PM_AirAccelerate adds 30 u/s perpendicular: |v|^2
  grows by 900 per frame whatever the speed). The record runs 7.62 ms frames (131 fps): 74 u/s of
  energy height per second of strafing against our 56 at 100 Hz. Over the ~10-15 s of flight from 10%
  to 24% that is ~+180-250 u - the size of the deficit, and the reason a better LINE alone (the dh
  beams: +20-60 u) cannot close it. CLAUDE.md documents --tick-ms / --tick-ms-schedule for exactly
  this (cannonball's WR also ran 131 fps).

### The tick (frame rate) test: the frozen policy's beam at 10 / 8.75 / 7.67 ms (09:05 UTC)
* The same d-ranked beam (4,096 copies, g10b's 14.0B weights trained at 10 ms) with beam_tas
  `--tick-ms`: 10 ms 42,089 (26.35%, the sheet); **8.8 ms (realised [9,9,9,8] = 8.75 ms, 114 Hz):
  41,931 = 26.62%**, the line carrying +38..+62 u of energy height at every level (5% 3,878 vs 3,821;
  20% 3,770 vs 3,732; 24% 3,476 vs 3,422; 25% 3,581 vs 3,519) and - with it - flying off the ramp top
  higher and onto the ridge at x ~3,520 and SURFING IT NORTH over the sheet's edge (the record's way)
  to (3,522, -2,161, 3,209); 7.63 ms ([8,8,7]): 18.27%, extinct at the gate-8 corridor - a memorised
  line breaks at a 30% faster air acceleration (CLAUDE.md's xQR32 result), while a 14% change keeps it.
* g10b evals to 14.33B: 26.300 / 24.581 / 26.333 / 26.330 (1-3/9 over gate 10, never more).
* **pbr4b3_t1** (09:12 UTC): g10b's 14.365B weights (md5 a5339175...) + g10b's flags +
  `--tick-ms-schedule 10:7.63:800e6` (the tick moves linearly 10 -> 7.63 ms over 800M steps, then
  holds; every per-second constant follows) + `--ckpt-every 2e8` (milestones for the handoff), 2B
  steps. Question: does the route survive the ramp (the curriculum, the milestones and the start-room
  states re-teach it), and does the energy at 24-26% rise toward the ~+250 u gate 11 needs?
  Generic: the tick is the player's frame rate (the WR's 131 fps), map-independent; the lineage
  already takes sv_maxvelocity 3500 from the same demo's movevars. A checkpoint carries its ramp, so a
  next-round resume of this lineage CONTINUES at 7.63 ms (the trainer prints the tick).
* t1 evals (tick ramp 10 -> 7.63 ms over 800M steps from 14.365B): 24.770 (14.365B, the resume) /
  26.316 / **26.374** (14.516B, tick ~9.55 ms: **8/9 greedy episodes over gate 10**, 26.34-26.37%; at
  10 ms g10b never had more than 3/9). The extra air acceleration carries the ramp climb. All 8 still
  die on the gate-11 sheet.
* The 8/9 quorum fired the trainer's auto-spine search 1 at 14.516B: beam 42,139 (no gain), three
  branched beams no gain, then the GE ladder - which runs at 10 ms (explore_phase1 has no tick option)
  and per GE11 needs ~x1.3 at this gate; killed after 9 min so training resumes (next try 14.916B).
* Beams on t1's 14.52B weights (tick ~9.5 ms): at its own tick 42,261 (26.05%); at 7.67 ms ([8,8,7])
  **42,059 = 26.40%** - after 150M steps of ramp the population survives the record's tick (the 14.0B
  weights went extinct at the gate-8 corridor there). Energy height at 24/25%: 10 ms 3,422 / 3,519,
  8.75 ms 3,476 / 3,581, 7.67 ms 3,521 / 3,628 (the record ~3,600 / 3,680) - but at 7.67 ms its line still
  lands on the floor after the ramp (26%: 3,236) instead of riding the ridge.
* `explore_phase1 --tick-ms N` (whole ms; flag off bit-identical, tests/python/test_ge_tick.py),
  ge_replay steps its core at the meta's tick, and auto_spine.py reads the checkpoint's realised tick
  (config tick_ms_eff), gives the explorer its whole-ms rounding and converts its seconds at that tick
  (ed636e1; pulled onto the box, so t1's next search uses it without a restart).
* t1 evals 26.366 / 26.359 (14.59-14.67B, tick ~9.2-9.0 ms): 7-8/9 over gate 10, all dying on the sheet.
  Curriculum back at ~1,695 (17 s). Next auto-spine search at 14.916B (tick ~8.4 ms).
* t1 evals 14.74-14.97B (tick ~8.9 -> 8.3 ms): 26.359 / 26.366 / 26.363 / **26.370** - from 14.89B
  **9/9 greedy episodes over gate 10** (all die on the sheet at 26.32-26.37%).
* auto-spine search 2 (14.969B, ckpt tick ~8.25 ms -> the explorer at 8 ms): beam 42,227, branched beams
  42,057 / 42,059 / 42,054, GE stage 1 (x1.0 / 1.04 at 8 ms) deepest 41,807 = 26.85% at (3,402, -2,087,
  3,189), over the sheet and falling (10 ms GE11 x1.0: 26.07%) - short of the 41,588 the gain rule needs.
  The hole that leads on is at 27.6-27.8% (d 41,379 at (3,481, -1,552, 3,142)). Stages x1.08 / x1.12 next.
* GE12 (beside it, 14 threads, 12 min): the newest branched-beam line's ramp ride x1.16 / 1.24 / 1.32 at
  8 ms - is the hole reachable at this tick, and with how much boost?
* search 2 stages x1.08 / x1.12 (8 ms): 41,717 (27.0%) / 39,187 (tool field) - no spine (1,535 s, the
  trainer waiting). GE12 (x1.16 / 1.24 / 1.32 at 8 ms): **26.42 / 27.02 / 30.48%** - the x1.32 root at
  mid-ramp (2,541, -2,975, 2,998), |v| 923 -> 1,218, h 3,530 -> 3,926 (the record ~3,600 here, the
  uniform explorer wastes what the policy would strafe in) flies the record's way: high off the ramp
  top, the ridge north, the launch, over the sheet and DOWN THE HOLE at 27.9%, a hard landing (-400 u),
  west and north to a crash at (2,199, -909, 2,647) and a crawl to 30.48%.
* Spine autoresearch/spines/pbr4b3_ge12_x132.npy (2,841 states: search 2's branched-beam line to its ramp
  + that chain, cut after the hole's landing at 28.41%; success line d <= 41,015 = 28.23%).
* **pbr4b3_t2** (10:16 UTC): t1's 14.990B weights (md5 ab8d9f51..., tick 8.2 ms; t1's last evals 26.363 /
  26.370 with 9/9 over gate 10) + the curriculum on pbr4b3_ge12_x132.npy from its end + `--demo-milestones
  44939,43426,42182` (the k01 / k02 / g10 success lines keep paying) + t1's other flags; the tick ramp
  continues from the checkpoint (8.2 ms now, 7.63 ms from 15.165B; no --tick-ms-schedule on a resume -
  the flag would REPLACE the ramp). 3B steps. Question: does the curriculum carry the policy over the
  sheet from its own (unboosted) states, as round 3's x1.12 k03 spine eventually did for gate 10?

### GATE 11 HALF-CROSSED: 27.287 (2026-10-25 10:40 UTC)
* t2 evals (curriculum on the x1.32 spine; the tick reached 7.63 ms at 15.165B): 26.358 / 26.372 /
  26.384 / 26.363 (14.99-15.22B) then **27.228** (15.293B) / 27.163 / **27.287** (15.444B, median 27.18%):
  8/9 greedy episodes now launch off the ridge's north end and fly over the sheet at 850-940 u/s,
  landing ON it 120-500 u short of the hole's south edge (ends at (3,538-3,594, -2,081..-1,693, z ~3,190);
  the hole spans y -1,425..-1,575). Training's reservoir already reaches 28.46%.
* The curriculum walked the spine back 2840 -> 2295 in ~150M steps and sits at the x1.32 root: band
  [2300,...] (the boosted chain) 96%, band [2295,2299] (the line's own unboosted states) 0.0% - round 3's
  k03 picture - yet the greedy line follows the chain most of the way with its own (7.63 ms) energy.
* Handoff snapshot pulled: pbr4b3_t2/ckpt_15390998528.pt (the milestone at or before the 27.287 eval).
* t2's last eval **27.409** (15.519B, median 27.19%). Beams on t2's 15.47B weights at their own 7.67 ms:
  **40,393 = 29.31%** (seed 4) / 29.19% / 28.96% (a branched one): the POLICY'S OWN population now flies
  gate 11 - the ridge north, the launch, the lethal floor, DOWN THE HOLE at 28.0%, on to 29.31% where it
  falls toward trigger_teleport *128 (a catch sheet at z ~2,577 over x -1,344..3,840, y -1,824..2,064):
  gate 12. Its energy height at 20 / 25 / 26%: 3,896 / 3,742 / 3,759 - above the record's (~3,796 /
  3,680 / 3,702, ANALYSIS ONLY) and ~+500 u over the 10 ms line at 26%. The frame rate was the deficit.
* Spine autoresearch/spines/pbr4b3_beam_g11.npy (that line, 2,900 states, success line d <= 40,497 =
  29.14%). **pbr4b3_t3** (10:46 UTC): t2's 15.551B weights (md5 ec14781b...) + that spine + milestones
  44939,43426,42182,41015 + the same flags (tick held at 7.63 ms by the checkpoint's ramp), 3B steps.
  t2's ckpt_latest (the 27.409 handoff) pulled.

### GATE 11 PASSED by the greedy line: 29.271 (2026-10-25 11:08 UTC)
* t3 evals (curriculum on the policy's own 7.67 ms gate-11 line): **29.271** (15.551B, the first eval,
  0.5M steps after resuming t2's 15.551B weights: one episode down the hole to 29.27%) / 28.627 / 28.728
  (15.702B: 9/9 past the gate-10 line, median 28.67%) / 28.824 / 28.892 / 28.818 (15.93B). From 15.70B
  the greedy goes over the sheet and DOWN THE HOLE in 8-9/9 episodes and dies at 28.5-28.9%.
  Handoff for 29.271: t2's ckpt_latest (the exact weights, md5 ec14781b...) placed as
  pbr4b3_t3/ckpt_15550906368.pt (the snapshot at or before the eval).
* Gate 12 (where every passing episode dies): after the hole the line falls NORTH-WEST (vz ~-1,100) into
  trigger_teleport *128 (a catch sheet at z ~2,577) at (2,900-3,000, -1,300..-1,380). The record
  (ANALYSIS ONLY) drops SOUTH-WEST out of the hole (3,412, -1,602) -> (3,278, -1,694, 2,863), lands on
  a surf ramp at ~(3,169, -1,661, 2,806) (contact) and rides it west to (2,528, -1,527, 2,900) at 29%.
* The curriculum sits right before the hole drop (bands [2749,2753] 29-31% / [2754,...] 87%); the
  spine's success line (29.14%) lies in the catch fall, so the greedy (28.7-28.9%) rarely crosses it
  and the trainer's quorum will not fire: searches run by hand (GE13 at 8 ms over the flight and the
  drop x1.0 / 1.04; branched beams forked at 2600 / 2700 / 2800 ticks, before the hole).
* Searches for gate 12 (11:09-11:22 UTC, beside t3): branched beams on t3's 15.99B weights forked at 2600 /
  2700 / 2800 ticks (before the hole, held view offsets +-60 deg): 40,391 / 40,532 / 40,778 (the catch fall),
  every forked lineage dead; plain 40,898. **GE13** (8 ms, 64 u x energy 32, 30 roots over the gate-11
  spine's flight and drop x1.0 / 1.04, 12 min, 10 threads): deepest **30.44%** - from the x1.04 root at
  spine state 2690 (over the sheet): down the hole heading SOUTH-WEST (the record's way), a hard landing on
  the ramp at (2,880, -1,683, 2,781) (h 3,835 -> 3,445), west then north at 820-900 u/s, falling toward the
  catch at (2,259, -827, 2,596). Several x1.0 roots reach 30.0-30.3%. (ge_roots_best's "group" =
  archive index // N is unreliable once two roots share a cell: match the chain's first state to the spine.)
* t3's later evals: 28.938 / 28.976 / 29.066 / **29.207** (16.31B, median 29.12%, 9/9 at 28.8-29.2% - the
  greedy creeps further along the catch fall; 5/9 past the 29.14% success line at the last eval).
* Spine autoresearch/spines/pbr4b3_ge13.npy (3,054 states, success line d <= 39,852 = 30.27%).
  **pbr4b3_t4** (11:27 UTC): t3's 16.316B weights (md5 0c61903f...) + that spine + milestones
  44939,43426,42182,41015,40497 + the same flags (7.67 ms held), 3B steps.
* t4 evals (curriculum on the GE13 x1.04 spine): 29.257 / 29.226 / 29.235 / 29.237 / 29.254 / 29.222
  (16.32-16.69B): flat - the greedy still falls NORTH-WEST into the catch after the hole (median 29.2%),
  and 3/9 fell back to gate 10 (24.1%) in the last eval. Training's reservoir meanwhile reaches **37.3%**
  (spawns on the chain fly far past). The curriculum sits at spine 2848 = the chain 1.6 s after its root,
  just before its hard landing on the south-west ramp (band [2848] 13%, [2853+] 95%).
* Beams on t4's 16.75B weights from the true start: 40,500-40,528 (the catch fall; a branched one too).
  beam_tas --start-state from the policy's own spine states over the sheet (2690-2780): every lineage dead
  within 0.8-1.5 s - that start path does not reproduce the trainer's spawn (held keys / duck); not pursued.
* GE13's x1.0 roots never make the south-west turn (the one from the policy's own state in the hole lands
  on a ledge at (3,077, -1,111, 2,964) and crawls); both turning chains come from x1.04 roots over the sheet.
  GE14 (11:55 UTC, 15 min, 12 threads): 40 spine states over the sheet flight (2650-2800) x1.0 / 1.02.
* t4 later evals 29.225 / 29.258 / 29.233 / 29.264 / **29.279** (17.147B; handoff milestone ckpt_17116954624
  pulled): flat at the catch fall; the curriculum band before the chain's ramp landing 13-21%.
* **GE14 (unboosted roots, 15 min): 31.79%** from root spine state 2750 at x1.000 - the policy's OWN state
  just before the gate-11 hole: a bounce off the room's north side turns it south-west into the hole, the ramp
  at (2,759, -1,751, 2,950), north along x ~2,400-2,560 at 875-1,045 u/s, a hard landing at (2,338, -511, 2,619)
  (h 3,298 -> 2,884) and on north to (2,383, -23, 2,661). Other x1.0 roots: 30.30 / 30.21 / 29.96%. With more
  roots and time the uniform explorer finds the turn without a boost (GE13's 30 roots x 2 boosts in 12 min did
  not). Spine autoresearch/spines/pbr4b3_ge14.npy (3,230 states, success line d <= 39,084 = 31.61%).
* **pbr4b3_t5** (12:15 UTC): t4's 17.223B weights (md5 a0e38643...) + that spine + milestones
  44939,43426,42182,41015,40497,39852 + the same flags, 3B steps.

### Gate 12's band: the policy turns north too early on the ramp (12:35-12:42 UTC)
* t5 evals 29.265 / **29.292** (17.299B) / 29.272 / 29.270 / 29.250 / 29.283 / 29.280: flat. The curriculum on the
  unboosted GE14 spine walked 3229 -> 2884 at once and stopped: band [2879,2883] at 2.2-4.0% for 25 min,
  band [2884,...] at 91-94%. States 2862-2886 of the spine SURF a ramp (func_wall *96, x 2,789-3,538,
  y -1,827..-1,659) that pushes the line up and around from west to south-west (vz -358 -> +393).
* record_ckpt --spawn-states from spine state 2879 (t5 17.69B, greedy + 4 stochastic): the policy holds RIGHT
  strafe and turns north at once (v_y -276 -> +689 in 18 ticks), leaving the ramp before it has pushed it up
  (vz +159 -> -25), so it stays ~50-100 u lower and falls into the catch (30.1-30.2%). The spine first keeps
  turning LEFT along the ramp face for ~6 ticks (yaw 197 -> 209) and only then right. From state 2884 (vz +335
  already) the same right turn works: **32.9-33.0%**, past the spine's own end (31.79%).
* So the band needs a ~20-30 deg leftward view (or the left strafe) for ~6 ticks - far outside the band
  temperature's 6 x 0.016 rad (~5 deg) yaw sigma. **pbr4b3_t6** (12:42 UTC): t5's 17.729B weights (md5
  67975c22...) + the same spine with `--demo-tau0 2884` + `--demo-front-temp 1.0 --demo-front-view 25
  --demo-front-ticks 60` (band spawns: keys at T 2, yaw sigma ~0.4 rad, for 0.46 s), the rest unchanged.
  t5's handoff: t4's ckpt_latest (the weights t5 resumed) placed as pbr4b3_t5/ckpt_17222860800.pt.
* t6 (band keys at T 2, yaw sigma x25 for 60 ticks): band [2879] 1.4-1.7% (worse), band [2884] 67-70% (was
  92%); evals 29.272 / 29.276 / 29.265 / **29.303** (17.956B) / 29.285 / 29.272 / 29.273 / 29.265. Per-decision
  noise does not find the manoeuvre and costs the easy band.
* More probes (record_ckpt from spine states 2800 / 2830 / 2850 / 2865 / 2875, greedy + 4 stochastic each):
  28.8-29.8% from all of them - the policy surfs the ramp from no state of the drop; only from 2884 on.
  beam_tas --start-state 2879 (works here: 4,096 alive at 0.77 s): extinct by 1.5 s; branched from tick 20
  (resample every 5) with held view offsets +-45 / +-90 deg: extinct by 1.8 s, frontier 29.9%. Key bursts
  are refused on a --keys-hold checkpoint (view-only branching).
* The explorer's raw actions through the ramp contact (spine 2869-2885): yaw bin 11 (turning left), FORWARD
  held, side bin 0, jump - a wishdir relative to the explorer's own arbitrary view; the canonical spine view
  hides it, and the policy (view along the velocity) holds right strafe instead. Round 2's gate-2 picture
  (the explorer's view vs the policy's). PPO's signal at the band is ~3 successes per iteration.
* **pbr4b3_t7** (13:10 UTC): t6's 18.274B weights (md5 9053241a...) + `--demo-front-frac 0.8` (twice the band's
  share of the demo draws) with t5's band temperature (0.5 / 6 / 40), tau0 2884, the rest unchanged.
  GE15 beside it: 50 unboosted roots over the gate-11 flight, hole and drop (spine 2600-2880), 15 min - other
  ways onto the ramp, to probe with the policy.
* The explorer's view on the gate-12 ramp (GE14's line before canon_view): a steady **+46 deg LEFT of the
  velocity**, keys forward + side bin 0: wishdir ~91 deg left of the velocity - an optimal LEFT air strafe
  while riding the ramp. The policy (view along the velocity) holds right strafe there. Under --keys-hold a
  single left sample persists unless the policy overrides it, which fits the band's 2-4%.
* t7 (`--demo-front-frac 0.8`, t5's band temperature): band [2879] 8.1-8.3% in its first minutes (t5: 3-4%).
* `explore_phase1 --view-follow DEG,...` (new; flag off bit-identical, tests/python/test_ge_view_follow.py):
  each burst holds a view OFFSET from the velocity heading drawn from the list and the view is steered to it
  every tick above 100 u/s - the search acts in the policy's view frame, so a chain's keys mean what they
  would mean to the policy. GE16 (13:16 UTC): GE15's 50 unboosted roots (spine 2600-2880) with
  `--view-follow=-90,-60,-30,-15,0,15,30,60,90`, 15 min (GE15, the uniform-view twin, stopped at 4 min:
  30.46%, GE14's kind of chains).
* t7 (band share 0.8): band [2879] 7-9% (t5 3-4%) but no higher; evals 29.285 / 29.301 / **29.317** (18.501B) /
  29.285 / 29.272 / 29.304. Handoff milestone ckpt_18474336256 pulled.
* **GE16 (view-follow) 32.05%** from the policy's own unboosted state 2743: out of the hole it holds FORWARD with
  the view ~89 deg left of the velocity (= a left strafe in the policy's frame), the ramp at (3,202, -1,673,
  2,826), north to 30.0% at (2,437, -1,024, 2,766), a hard landing at (2,371, -873, 2,614) (h 3,286 -> 2,672) and
  then 25 s of crawling near the catch (z 2,594-2,830, 13-590 u/s) up to (2,279, 101, 2,596). The record passes
  there ~230 u higher with ~250 u more energy (h ~3,530 vs ~3,290 after the ramp landing, ANALYSIS ONLY): past
  30% gate 13 is energy again.
* **pbr4b3_t8** (13:36 UTC): t7's 18.771B weights (md5 fce2c1c7...) + spine autoresearch/spines/pbr4b3_ge16vf.npy
  (the g11 line to 2743 + GE16's chain to its 30.28% landing, the explorer's own views - no canon_view;
  success line d <= 39,943 = 30.11%) + milestones 44939,43426,42182,41015,40497 + band share 0.8, the rest as t7.
* t8 (the view-follow spine, explorer's views kept): the curriculum walked 3073 -> 2873 and stalled at the same
  ramp: band [2873,2877] 0.0-0.2%, [2878,...] 45-49%; evals 29.301 / **29.318** (18.847B) / 29.307 / 29.286 /
  29.285 / 29.316 / 29.310. GE17 (view-follow, rooted on GE16's flight north before its hard landing):
  **33.14%** from (2,491, -1,151, 2,860) - the next gate's material.
* Why the band stays shut, again: under --keys-hold the movement keys are HELD STATE the policy observes, and a
  spine spawn resets them to neutral; the explorer held forward (with its view ~89 deg left of the velocity)
  through the ramp, so every band spawn started from a different state than the line's own - the
  --keys-hold-spawn-duck defect for every held key. And keeping the explorer's view does not help a policy that
  re-aims its view at ~0 offset within ~9 ticks.
* Two generic fixes (ba7088f, tests/python/test_demo_keys.py: flag off bit-identical, a bad file refused;
  test_spine_canon_keys.py): `tools/spine_canon_keys.py` puts every fast row's view along the velocity AND
  translates its held keys so the wish direction survives (forward at +89 deg -> left strafe at 0 deg), and
  `train_fast --demo-keys K.npy` (TRAIN_ONLY) starts a spine spawn holding its row's keys. Spine
  autoresearch/spines/pbr4b3_ge16ck.npy + .keys.npy: on the ramp band the spawn now holds LEFT strafe with the
  view along the velocity - exactly what the policy has to learn to keep.
* **pbr4b3_t9** (14:04 UTC): t8's 19.296B weights (md5 d717d97d...) + that spine + `--demo-keys` + t8's other flags.
  t8's handoff: t7's ckpt_latest (the weights t8 resumed) placed as pbr4b3_t8/ckpt_18770558976.pt.

### GATE 12 OPENS WITH THE HELD KEYS, AND THE LOOP RUNS: 33.376 -> 36.211 (2026-10-25 14:20-14:32 UTC)
* t9 (spine pbr4b3_ge16ck = the canonical view + translated held keys, `--demo-keys`): evals 29.295 / 29.283 /
  29.317 / 29.227 then **33.376** (19.598B, 8/9 greedy episodes at 33.0-33.4%) - the ramp band that held t4-t8
  for 2.5 hours (0-9%) passed as soon as a spawn there held the left strafe the line was flown with.
* The 8/9 quorum fired the trainer's own auto-spine: search 1 (64 s) - the branched beam past gate 12 to 34.99%
  (k01, 5,100 states); search 2 at 19.674B (43 s) - the PLAIN beam to **41.07%** (k02, 5,400 states, success
  line 40.9%). Evals **35.830** (19.674B) / **36.211** (19.749B, median 35.58%). The curriculum walks k02 back at
  74-84% band success, 40 states a move. Handoff milestone ckpt_19696451584 pulled.
* The swap dropped --demo-keys (the old trainer had no keys for an auto spine): fixed for the next launch
  (4c8f7cb: auto_spine writes spine.keys.npy - a beam line's from its own action table, a GE splice's from the
  archive's link actions translated with the view - and --demo-keys loads it on a swap). t9 runs on unchanged
  (the beam lines are the policy's own, and their bands pass anyway).
* t9 evals 38.765 / 37.487 / **42.963** / 43.211 / **44.025** (20.127B, median 43.82%): the trainer's auto-spine
  ran itself - search 3 (19.976B, 127 s): branched beam to 43.73% (k03, 6,200 states); search 4 (20.127B, 74 s):
  branched beam to 45.18% (k04, 4,700 states). Four swaps in 30 min, milestones accumulating (44939 ... 32260).
  Handoff milestone ckpt_20097007616 pulled (and t9's ckpt_latest 20.189B).
* k04's band [4629,4633] sat at 2-4% against [4624,...] 99% (a cliff like the gate-12 ramp's), with the keys
  dropped on the swap. **pbr4b3_t10** (14:56 UTC): t9's 20.189B weights (md5 a228de09...) + k04 with its held keys
  (autoresearch/spines/pbr4b3_auto_t9k04.*, `--demo-keys`, `--demo-tau0 4629`) + all nine milestones + the trainer
  that keeps keys across auto-spine swaps (4c8f7cb), the rest as t9.

### 28-44% is flown above the record's energy; gate 14 is a crash at 44.2% (15:15-15:22 UTC)
* t10 (k04 + its held keys, all nine milestones): evals 44.127 / 44.327 / 44.276 / **44.373** (20.416B) / 44.322;
  the band at k04's end [4629] rose 2-4% -> 7.5% with the keys - it sits on a hop's landing in a slow
  bunny-hop crawl at 44.8-45.2% (k04 ends there). Handoff milestone ckpt_20389560320 pulled.
* Beams on t10's 20.54B weights (d-ranked x2, dh K 3): 31,175 / 31,191 / 31,163 = 45.45% - the same crawl.
* The beam line (s51) against the record (ANALYSIS ONLY), first arrival at each level: 28% 1,279 u/s h 3,920
  (record 1,043 / 3,533); 32% 1,315 / 3,819 (1,131 / 3,552); 36% 1,347 / 3,886 (1,187 / 3,638); 40% 1,453 / 3,869
  (1,279 / 3,602); 44% 1,350 / 3,908 (1,213 / 3,674) - on the record's own route within ~50 u, FASTER and ~250 u
  higher in energy all the way: the 7.63 ms tick did what the per-frame physics said.
* Gate 14: at 28.46 s the line, falling at vz -686, hits a floor + wall corner at (-964, -1,395, 2,612): |v| 1,419 ->
  341, h 3,915 -> 2,696 in two ticks, then crawls (45% only 8 s later). The record passes there 150-200 u higher
  (z 2,755-2,837) and keeps 1,071+ u/s. A line-shape gate (too steep), not energy.
* GE18 (15:22 UTC, view-follow, 8 ms, 12 min): 40 roots on the s51 line over 26.4-28.4 s (before the crash), x1.0.
* t10 later evals 44.361 / 44.352 / 44.345 / 44.429 / **44.505** (20.869B); the k04 band at its hop landing 13%.
  Handoff milestone ckpt_20790116352 pulled (and t10's ckpt_latest 20.947B).
* **GE18 (view-follow, unboosted): 47.18%** from the s51 line's state at 42.08% (-1,094, -51, 2,924): the chain
  meets the 44.2% corner falling less steeply, REBOUNDS (h 3,869 -> 3,351 but still 1,082 u/s) instead of
  stopping dead, and flies west along the record's corridor at ~780 u/s to (-2,405, -1,241, 3,096). Spine
  autoresearch/spines/pbr4b3_ge18ck.npy + .keys.npy (s51 to 3,557 + the chain, canonical views, translated keys;
  success line d <= 30,286 = 47.00%).
* **pbr4b3_t11** (15:36 UTC): t10's 20.947B weights (md5 6a247ef4...) + that spine + `--demo-keys` + ten
  milestones (..., 32260, 31429), the rest as t10.

### GATE 14 PASSED by the greedy line: 47.253 (2026-10-25 15:57 UTC)
* t11 (keyed spine pbr4b3_ge18ck): evals 44.545 / 44.519 / 44.493 / 45.101 (one episode through) then **47.253**
  (21.250B, 9/9 greedy episodes at 47.09-47.25%) ~300M steps after the launch: the greedy rebounds off the 44.2%
  corner and flies the corridor west to the spine's end. The 9/9 quorum fired the trainer's auto-spine search 1
  at once (running). Handoff: the EXACT weights of the eval (ckpt_latest saved by the search at 21,249,916,928)
  pinned as pbr4b3_t11/ckpt_21249916928.pt and pulled (md5 32bc5715...).
* t11's auto-spine search 1 (21.250B): beam / three branched beams 30,118-30,147 (47.3%), GE stage 1 (x1.0 / 1.04,
  the uniform explorer) 30,132 at (-2,403, -1,233, 3,150) - no spine; stopped before the boosted stages (687 s of
  trainer wait). Gate 15 sits at (-2,400, -1,240, ~3,100-3,150), 47.2-47.3%; the record (ANALYSIS ONLY) goes on
  north from there ((-2,320, 255, 3,316) at 50%).
* auto_spine's Go-Explore stages now run `explore_phase1 --view-follow` by default (`--ge-view-follow`, 'none' =
  the old explorer): on the box for t11's next search at 21.65B.
* GE19 (16:07 UTC, view-follow, 12 min beside the trainer): 48 roots over the last 4 s before the 21.25B beam
  line's stall (its tick 3,927 of 4,100).
* t11 evals 47.204 / 47.174 / 47.188 / 47.201 / 47.169 (21.33-21.63B): 9/9 stop at gate 15 (47.1-47.2%). GE19
  (view-follow, 48 roots over the 4 s before the stall): best 47.37% - nothing past it either.
* Gate 15 against the record (ANALYSIS ONLY): the record surfs at (-2,286..-2,443, -1,706..-1,598), launches with
  vz +617 and flies NORTH at z ~3,200-3,240 (47.45% at (-2,450, -1,183, 3,239), 661 u/s, h ~3,510); our lines
  reach the same x at z ~3,100 with h ~3,300-3,330 and turn east. The ~190 u of energy is gate 14's rebound
  (-518 u at the 44.2% corner, where the record keeps ~3,550): gate 15 is energy again, created one gate back.
* t11's auto-spine search 2 (21.703B, 1,647 s with the trainer waiting): beams 30,142 (no gain); GE stage 1
  (view-follow, x1.0 / 1.04) 29,990 = 47.52% at (-2,411, -1,167, 3,212) - level with the record there, 208 u past
  the greedy (< the 500 u gain); stage 2 (x1.08) 30,107; **stage 3 (x1.12) 28,718 = 49.75%** at (-2,406, 104,
  3,219), north along the record's route. NEW SPINE k02 (4,569 states, success line d <= 28,818 = 49.57%), its
  held keys loaded from k02/spine.keys.npy on the swap (4c8f7cb, first live use). Gate 15 needs ~x1.12 of the
  explorer's energy at its root (the rebound's loss, one gate back).

### Gate 15 is a LAUNCH, and k02's curriculum is stuck on its first 5 ticks (17:15-17:50 UTC)
* t11 on k02 (21.70-22.33B): evals 47.14-47.22%, 9/9 at the gate. The curriculum oscillates every ~500 band
  episodes between tau 3898 (band [3898,3902]: 0.0-0.4% success) and 3903 (band [3903,3907]: 53-57%) for 600M+
  steps; the reservoir's deepest state is d 26,634 (53.4%, from k02's late spawns).
* What happens at 3899-3905 (k02 = the policy's beam line to 3781 + the x1.12 GE chain from 3782): the chain
  releases the ramp (keys forward+right -> left at 3899) at the ramp's lip and the velocity is deflected UP for
  6 ticks (vz 200 -> 659, horizontal 1,144 -> 686 u/s, -300 u of h), orbits ~140 deg against the wall, then flies
  north at z 3,060 -> 3,258 and lands on the floor at (-2,320, -954, 3,207) at 4,060, then hops north to y +63
  (49.75%). Spawns from 3903 on are already mid-launch.
* The greedy (22.16B, ep 8) at the same lip: h ~3,650 (MORE than the chain's ~3,500 after its launch), but its
  launch is soft (vz 256 -> 456 over 15 ticks, horizontal 1,062 -> 959): it reaches (-2,420, -1,256, 3,080), ~180 u
  below the chain at that y, circles a full turn in the air and falls back south-east. Energetically it could make
  the 3,207 floor with ~800 u/s to spare; what it lacks is the CONVERSION of speed into height at the lip, and the
  chain gets that from a faster (x1.12), 48 u higher approach that the policy's own line does not have.
* So k02 is a dead end for the greedy as built: the band before the launch is ~0%, and even a learned launch would
  start from a state 12% faster than the policy's own line. The next lever is upstream and generic: search gate 14
  (44.2%) for a line that keeps its energy (round 3's "pick by energy, not depth"), or search the lip from the
  policy's own (unboosted) states for a chain that launches high; GE stage 1 (x1.04) already reached 47.52% at
  (-2,411, -1,167, 3,212), level with the record, ~200 u short of the floor.
* **The exact chain on the box (t11/auto/k02/ge3 link actions, 17:25 UTC):** from 3890, 3896, 3898 and 3902 the
  chain's own actions launch (vz max 659, z max 3,270, d 29,770 after 200 ticks) - at 8 ms, and at [8,8,7] too
  (vz 641-653; the run then dies later in the hops, except from 3902). The SAME actions with **duck held**: vz max
  368-372, z max 3,069-3,071, d 30,650 - the greedy's failure to the unit (the greedy holds jump+duck, buttons 6,
  through the lip; the chain launches unducked and ducks only at 3920). In the policy's frame (view along the
  velocity, the chain's raw keys / the spine's translated keys): vz 200-320, no launch - the 8-direction key
  translation cannot express this contact, so the k02 band spawns cannot even be replayed open-loop.
* **And from the GREEDY's own approach (22.16B ep 8, rows 3990-4013, local [8,8,7] sweep of 9 held key pairs x 13
  view offsets x duck for 60 ticks): z max is capped at 3,092 unducked / 3,110 ducked whatever is held** - a ceiling
  at z ~3,128 above the place the greedy launches (vz up to 676 there, straight into it). The chain reaches the lip
  48 u higher and 12% faster and launches further west, past the ceiling. So gate 15 is the APPROACH line (set at
  gate 14 and the ramp before the lip), not a missing input at the lip: no curriculum on the lip alone can fix it,
  and un-ducking alone does not either.
* Next round, generically: an energy-ranked search from roots before gate 14 whose chains are scored at the lip
  (height + speed on arrival, not depth), and a policy-frame explorer for contacts like this one (a continuous
  view offset, not 8 key directions, if the spine is to carry it).

### GATE 15 PASSED by the trainer's own loop: 47.253 -> 51.093 (2026-10-25 18:27 UTC)
* (The analysis above was right about k02 and wrong about the loop.) t11's auto-spine search 3 fired on PATIENCE at
  22.910B (17:52 UTC, 1.2e9 steps after k02 with no quorum; ~25 min of trainer wait): beam / three branched beams
  30,183 (no gain); GE stage 1 (x1.0 / 1.04) 29,997; **stage 2 (x1.08) 28,683 = 49.81%** from root 3,819 of the
  policy's own beam line (a LESS boosted chain than k02's x1.12, from a later root) -> spine k03 (4,438 states,
  keys translated; autoresearch/spines/pbr4b3_auto_t11k03). The greedy followed it ~150M steps after the swap:
  eval 22.986B 47.132, **23.062B 50.720** (7/9 at 50.5-50.7%) -> the 9/9-style quorum fired search 4 at once
  (37 s): the PLAIN beam on t11's own weights reached d 26,901 = 52.93% -> spine k04 (4,700 states, the beam's own
  keys; pbr4b3_auto_t11k04). Evals **23.137B 51.093** (9/9 at 47.5-51.1%, median 50.7%; best), 23.213B 50.887.
* Handoff: the snapshot before the 51.093 eval, ckpt_22950182912 (187M steps before), predates the breakthrough
  (its own eval 22.986B: 47.13), so ckpt_latest was pulled at 18:29 UTC = step 23,203,414,016 (66M after the
  eval, > 2x nearer; md5 ca202f62...) and is the RESULT.json ckpt; ckpt_23150460928 (13M after) pulled as well.
* So the loop beat my diagnosis: a x1.08 chain from a later root was learnable where the x1.12 chain from root
  3,782 stalled at its launch - the escalation ladder's FIRST sufficient boost, re-searched on newer weights, is
  what passed. Patience (1.2e9) mattered: the quorum would never have fired on k02.
* Gate 16 (50.7-51.1%): 8/9 episodes end with "fail" AT their best row, at (-2,240..-2,410, 340..520, 2,900-3,095),
  700-1,075 u/s - a fail trigger at the north end, reached lower than the record's line there (ANALYSIS ONLY: the
  record is at z ~3,316 by y 255). The k04 curriculum was at [4,189, 4,699] at 30-34% success at 18:31 UTC.

### GATE 16 PASSED, the loop again: 51.093 -> 52.930 (2026-10-25 18:57 UTC)
* On k04 (the policy's own beam to 52.93%): evals 23.288B 52.911, 23.364B 52.923 (quorum -> search 5, 638 s: beam
  to 27,135 then an UNBOOSTED view-follow GE stage-1 chain from root 4,344 to d 25,021 = 56.22% -> spine k05,
  autoresearch/spines/pbr4b3_auto_t11k05), 23.439B 52.910, **23.515B 52.930** (9/9 at 52.83-52.93%; best).
* Wrap-up (19:03 UTC): one more eval, **23.590B 52.937** (9/9 at 52.8-52.94%, median 52.85%; the round's best),
  then 23.666B 52.920. Handoff: ckpt_23551016960.pt (39M steps before the 52.937 eval, md5 7fbcd458...); the
  wrap-up's ckpt_latest is step 23,665,836,032 (75M after, md5 d6d9edd0...), not twice as near.

### pbr4b3 round 4 lesson (wrap-up, 2026-10-25)
* **Score 24.737 -> 52.937** (pbr4b3_t11 23.590B, 9/9 greedy episodes at 52.8-52.94% from the true start, median
  52.85%; 0 finishes). Gates 10, 11, 12, 13 (the auto-spine's 35/41/43.7/45.2% lines), 14 (47.25%), 15 (50.7-51.1%) and 16
  (52.9%, the policy's own beam line k04) passed; the loop's k05 (an unboosted GE chain to 56.2%) was being learned
  at the end.
* **What did it, in order of effect:**
  1. **Throughput, 50x on a quota-limited box** (ad5f905): numba sized its pool to 64 visible cores under a 30.7-CPU
     quota and the goal field's prange team throttled the trainer to ~7k steps/s; surfgym now defaults
     NUMBA_NUM_THREADS to the OMP count. ~350k steps/s after, and the beams ran 10x faster (25 s each).
  2. **A plain beam on the incumbent's own weights** passed gate 10 (round 3's last searches were on older weights).
  3. **The physics tick (frame rate): `--tick-ms-schedule 10:7.63:800e6`.** Air-strafe gain is per FRAME (|v|^2 grows
     900 per optimally strafed frame), the policy strafes at that optimum, and the WR runs 131 fps: at 100 Hz the
     line was 100-300 u of energy height short of the record at gates 10-11. The ramp alone took gate 10 from 1-3/9
     to 9/9; by 28-44% the policy's beam line flies the record's route ~250 u ABOVE its energy, faster. (A checkpoint
     carries its ramp: a resume continues at 7.63 ms; never pass the flag to a resume.)
  4. **Held keys at spine spawns (`--demo-keys`, tools/spine_canon_keys.py, auto_spine spine.keys.npy).** Under
     --keys-hold the keys are state the policy observes; every spine spawn reset them to neutral. A searched
     manoeuvre made with held keys (gate 12's ramp: forward + a view 46-89 deg left of the velocity = a left strafe)
     then starts from another state than the line's own: its band sat at 0-9% for 2.5 h over five runs (wider
     temperature, more band share, explorer views kept - none helped); with the canonical view, the keys translated
     to keep the wish direction, and spawns holding them, it opened at once and the greedy went 29.3 -> 33.4%.
  5. **The view-follow explorer** (`explore_phase1 --view-follow`, auto_spine's default now): bursts hold an offset
     from the velocity heading, so chains are found in the policy's own frame - gates 12 and 14 fell to UNBOOSTED
     chains from the policy's own states (the uniform yaw-rate explorer needed x1.04-1.32 there).
  6. **The trainer's own loop (`--auto-spine`, quorum 5/9 or patience 1.2e9 steps)**: once the keys were right, t9
     swapped in four of the policy's beam lines in 30 min (35 -> 41 -> 43.7 -> 45.2%); in t11 the loop's third search
     (on PATIENCE, 1.2e9 steps after a x1.12 spine that stalled at gate 15's launch) found a x1.08 chain from a later
     root, the greedy passed it ~150M steps after the swap (47.13 -> 50.72), and the quorum's search 4 swapped in the
     policy's own beam to 52.9% in 37 s. No agent in the loop for either gate; milestones kept every crossed line
     paid.
* **What did not:** energy-ranked beams (`--score dh`: +20-60 u, the same lines), branched beams at gates 11-12,
  start-state beams from mid-line states (all lineages died - beam_tas's start path is not the trainer's spawn),
  wider band temperature (T 2, yaw x25: worse), boosted GE spines from early roots (x1.32 at gate 11, x1.12 at gate
  15: the curriculum stalls on the part the policy cannot reach at its own speed - k02's band before the launch sat
  at 0.0-0.4% for 1.2e9 steps), my own reading of gate 15 (I called it an approach problem the loop could not
  solve; a less boosted chain from a later root solved it - measure before concluding, and let patience run).
* **Numbers:** t11 (resume of t10 20.947B, 3e9 budget): 47.253 at 21.250B (gate 14), 47.1-47.2 for 1.7e9 steps on
  k02, 50.720 at 23.062B, 51.093 at 23.137B, 50.887 at 23.213B, 52.911 / 52.923 / 52.910 at 23.29-23.44B,
  52.930 at 23.515B, **52.937 at 23.590B**, 52.920 at 23.666B. ~330k steps/s on the 3090 box between searches;
  each search stage is ~8 min of trainer wait (search 2: 1,647 s; search 3: ~25 min; search 4: 37 s).
* **To continue (next round):** resume pbr4b3_t11's ckpt_23551016960.pt (the RESULT.json ckpt) with t11's flags
  and the last spine: `ARM_RESUME=1 SELF_STATES=1 bash tools/run_arm.sh <run> --envs 1024 --maxvel 3500
  --ckpt-every 2e8 --demo-file autoresearch/spines/pbr4b3_auto_t11k05.npy --demo-keys
  autoresearch/spines/pbr4b3_auto_t11k05.keys.npy --demo-window 5 --demo-grow 5 --demo-grow-max 40
  --demo-cooldown 5 --demo-rate 0.3 --demo-front-frac 0.8 --demo-success-d 100 --demo-success-bonus 50
  --demo-milestones 44939,43426,42182,41015,40497,39943,37251,33777,32260,31429,30286,28818,28783,27001
  --keys-hold-spawn-duck --yaw-entropy 0 --demo-front-temp 0.5 --demo-front-ticks 40 --demo-front-view 6
  --respawn-greedy 0.2 --respawn-greedy-window 0.25 2.0 --respawn-greedy-stuck 200 --respawn-greedy-contact
  --respawn-greedy-temp 0.5 --respawn-greedy-temp-ticks 100 --respawn-greedy-temp-view 6 --respawn-front-perturb
  0.2 --respawn-front-perturb-band 0.005 --respawn-front-perturb-speed 1.0 1.3 --respawn-front-perturb-turn 90
  --energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0 --no-respawn-novel --respawn-backward-mix 0
  --int-real-only 0 --spawn-states autoresearch/spines/pbr1b1_entry_ge_states.npy --spawn-states-frac 0.2
  --auto-spine --auto-spine-milestones --auto-spine-gain 500 --auto-spine-ge-minutes 8 --auto-spine-retry 4e8
  --auto-spine-patience 1.2e9`. The checkpoint carries the 7.63 ms tick (never pass --tick-ms-schedule to it) and
  the keys-hold / potential / view shapes. On a quota-limited box check `nproc` vs the cgroup quota first (the numba
  fix is in surfgym; the CPU count the trainer sizes from is the quota).
* **Where it stands:** the greedy at 52.8-52.9% (d ~26,950, the end of k04); k05 (unboosted, to 56.2%) swapped in at
  23.364B and being learned when the round ended. Gate 16 (~51%, a fail trigger at the north end of the corridor)
  fell to the policy's own beam line within ~150M steps. The loop is running: let its patience fire before
  intervening.

## pbr5b2 (round 5, slot 2) - 2026-10-25 19:47 UTC start

Box: 57714456, RTX 5080 + Ryzen 9 5900XT (32 thr, cgroup quota 30.72 CPUs), 0.621 $/h, Florida
(machine 118632, round 2's box; the 0.363 $/h Threadripper offer was taken by another agent within a
minute). Incumbent: pbr4b3_t11 ckpt_23551016960 (md5 7fbcd458...), eval 52.937 at 23.590B.

### c1: the incumbent's loop, continued (19:53 UTC)
* **pbr5b2_c1** = resume of the incumbent's weights with round 4's last flag set (NOTES "To continue")
  on spine k05 (the trainer's own search 5: beam to 4,344 + an unboosted view-follow GE chain to
  56.22%) + its held keys + all 14 milestones + --auto-spine (quorum 5/9, patience 1.2e9, gain 500,
  GE 8 min per stage), 4e9 steps. The curriculum restarts at k05's end (its position is not
  checkpointed). The first eval of the resumed weights (23.5515B): frontier 29,822 u on the trainer's
  field, one episode dead at 22 s.
* What lies past k05 (ANALYSIS ONLY, the WR demo on the grader field): the record reaches 52.9% at
  ~44.8 s (our greedy at ~36 s: ~9 s ahead), drops from z ~2,600 to ~1,250 at 1,693 u/s (46 s, 54.8%)
  and climbs back to z ~2,200-2,500 (47-49 s, 56.7-58%) with energy height ~2,800-2,850 (z +
  |v|^2/1600). k05's line ends at 56.22% at (-2,395, 2,472, 1,965) at 212 u/s, h ~1,990: ~800 u of
  energy height under the record there. The gate after k05 will probably be energy again.

### The gate after k05 is an energy climb; auto_spine gets energy-anchored GE roots (20:00-20:15 UTC)
* Past k05's end (56.22%, z 1,965) the grader field's path CLIMBS to z ~2,357-2,421 by 56.6-57.0% and
  runs at z ~2,390 to 61% (greedy descent on the field grid). k05's line arrives with energy height
  h ~1,994 (212 u/s): it cannot reach z 2,390 at all. The k05 GE chain (from state 4,344, 50.86%) loses
  ~1,650 u of h in crashes over 50-54% (single-tick losses -254 at 50.2%, -140 at 51.2%, -126 / -120 at
  51.6-51.7%, -335 at a floor landing at 52.7%, -348 at 54.0%); the record (ANALYSIS ONLY) loses ~520
  there: it surfs a zig-zag descent from z ~2,600 to ~1,250 at up to 1,712 u/s (h 3,039 at the U's
  bottom, 54.8%) and swoops up (vz +1,243) to z 2,213+ at 56.7% with h ~2,800. k05 reaches the bottom at
  ~1,150 u/s (h ~2,130): the climb needs >= ~1,350 u/s there even with zero speed at the top.
* So the loop will probably stall at ~56.2%: auto_spine's GE roots sit in the last 4 s before the stall,
  all post-crash, and a x1.12 boost of a 200-700 u/s state adds < 60 u of h.
* **auto_spine `--ge-boost` E stages** (ef2d3db, pulled onto the box: the running trainer's next search
  uses it): the default ladder is now `1.0,1.04;E1.0,1.04;1.08;1.12`; an E stage roots the explorer in
  the 1 s before each of the 3 largest losses of h (over 0.1 s windows, >= 0.5 s apart) in the line's
  last 12 s before its stall (`energy_roots`; --energy-roots-secs / --energy-drops / --energy-lead).
  Generic: the policy's own line and energy height, no map constant. On k05 itself it picks the losses
  at 4,254 / 4,551 / 4,731. Tests: test_energy_roots_sit_before_the_largest_losses,
  test_ge_ladder_energy_stage_gets_its_roots (+ the old ladder / branched / cut tests pass).
* c1 curriculum so far: k05 walked back 5001 -> ~4436 in ~25 min (bands 37-75% except two stuck spots:
  [4576-4586] 0.0-0.1% for ~5 min (the chain leaves a floor and turns 150 deg left in the air), then
  [4421-4436] 0.1-1.2% (the chain's own impact on a surface at 51.7%)). Greedy evals 52.92-52.94%.
* Local probe (record_ckpt, the incumbent's weights, single k05 states at 54% / 54.6% with the velocity
  x1.0 / 1.2 / 1.4): every episode dies in the fall catch at z ~1,190 under the U's bottom; x1.0 from
  4,780 reaches 56.15% then falls; with more speed the policy flies another line and dies sooner. More
  energy alone does nothing for weights not trained there (a WON'T, not a CAN'T).

### c1 reaches k05's end: 56.341 (24.0045B); search 1 and the 56.6% shaft (20:20-20:45 UTC)
* c1 evals: 52.92 / 52.86 / 52.90 / 52.85 / 52.88 / 52.86 (23.55-23.93B, while the curriculum walked k05
  back to ~4,370 = 51.2% at ~300k steps/s), then **56.341** (24.0045B, a few episodes through) and
  **56.252** (24.080B, median 56.12%, 6/9 past k05's success line 56.05%) -> the trainer's quorum fired
  auto-spine search 1 at 24.080B (ckpt_latest saved there, md5 adb0febb..., pulled with the snapshot
  ckpt_23951572992, md5 be54db67..., the hand-on rule's pick for the 56.341 eval).
* Where they die: 6/9 rise at (-2,260..-2,350, 2,470-2,540, z ~1,997-1,999) into trigger_teleport *192 /
  *210 (fall catches) under a ceiling; 2/9 still die at the 52.8% floor, 1 at 47.4%. Core probes
  (hull traces + play_step teleport tests over xy slices): the way up is a SHAFT ~64-96 u wide at
  x ~-2,476..-2,412 through solid at z ~2,000-2,100 into an upper room (z ~2,150-2,250, x -2,476..-2,220),
  then another rise to z ~2,350 (free x -2,316..-2,220), which the field's path takes at 57.0%. The record
  (ANALYSIS ONLY) goes up the shaft at (-2,411, 2,568, 2,020) at 1,103 u/s and on to z 2,446-2,486 with
  h ~2,800; the greedy arrives with h ~2,290 at 56.1% (it can rise to ~2,290 at most).
* Greedy energy vs the record, first arrival (ANALYSIS ONLY): 50% 3,680 vs 3,562 (ours higher), then three
  hard impacts the record does not have - 50.2% -191, 51.7-51.95% -687 (record -317), 52.8% -490 (a floor
  landing at (-2,346, 1,166, 2,206); the record flies over that floor ~100-250 u higher and ~300 u further
  north) - 56.1% 2,288 vs 2,797. ~630 u lost to impacts over 50-56% beyond the record's.
* Search 1 (24.080B, greedy median d 25,079, gain 500 -> needs d < 24,579): plain beam 24,807 (56.59%);
  three --branch-tick beams 24,807-24,810; the two NEW energy forks (0.5 s before the plain line's largest
  losses, ticks ~4,500) 24,810-24,812 with ALL 2,048 forked lineages dead (the whole population is at
  203/4,096 valid 3.8 s after the fork: the descent is narrow at 1,000+ u/s, a +-45 deg held view kills);
  GE stage 1 (last 4 s, x1.0 / 1.04) deepest 24,798 = 56.61% IN the upper room (-2,316, 2,466, 2,155): its
  archive has 320 cells in the upper room with h <= 2,395, none at z >= 2,300 - the energy wall.

### Search 1's verdict, the beam probes, and what the energy problem really is (20:55-21:15 UTC)
* Search 1 (24.080B, 2,241 s of trainer wait): NO spine. GE stages, deepest d (needs < 24,579 = 57.0%):
  x1.0/1.04 24,795 (56.61%) / E (energy roots) 24,819 / x1.08 24,748 (56.70%) / x1.12 24,725 (56.74%) - all in
  the upper room. The E stage left the default ladder (c70f542); the energy forks stay (cheap).
* c1 stopped at 24.0805B (ckpt_latest = the search's 24.080B save, md5 adb0febb...). Its curriculum had sat at
  band [4371,4375] (k05's GE chain at 51.2%, 7-15%) for ~30 min: most curriculum spawns were on the uniform
  explorer's crash-heavy chain, which the greedy no longer follows (it reaches 56.1% on its own line).
* Beam probes on c1's 24.08B weights, from the true start, the box idle (47 s each): `--score dh --score-energy 2`
  -> 56.59%, h at 56% 2,337; the same + `--keys-temp 1.5` -> 56.58%, h 2,201. The plain beam had 56.59%, h 2,345.
  The same three crashes in every line (50.1-50.2% -200..-346, 51.6-51.8% -310..-497, 52.8% -353..-370): the
  policy's population has no crash-free descent; energy-weighted ranking cannot keep what is not there.
* The archives DO hold energy the policy's line lacks: the E-stage archive (roots before the crashes) has cells
  at y 1,200-1,500 (53%) with h 3,022 (the beam ~2,441 there) and at the U-bottom h 2,725-2,741 in stage 1's
  archive (the beam ~2,380-2,412); but every explorer chain loses it again before the climb (max h in the climb
  region 2,343-2,427). And the policy does not keep it either: rolled out (greedy, box) from the U-bottom state
  with h 2,741 it hits the U's exit walls (-114, -160, -282) and reaches 56.0% with h 2,021 - with more speed it
  flies a line it never learned. (A local probe print showed only the start energy; corrected here.)
* So the gate needs a new SKILL line: the descent flown high along the walls (the record stays at z ~2,550-
  2,600 from y ~600 to ~1,450, ANALYSIS ONLY) and the shaft climb with that energy. Searches alone have not
  produced it.
* **pbr5b2_c2** (21:00 UTC): c1's 24.080B weights + c1's flags + `--respawn-greedy-energy 3` (the greedy pool now
  6,512 states incl. the windows before the descent's losses of -316..-733 u at 34-36.6 s and gate 14's corner)
  + `--demo-tau0 4340` (the curriculum restarts just before k05's explorer chain) + `--auto-spine-delay 4e8`.
  Evals 56.107 / 56.142 (24.08 / 24.156B). Paused at 24.16B (21:05) for a focused GE:
* GE-C (21:06 UTC, full CPU, 10 min): roots = chain A's last second (beam line to 4,372 + stage 1's exact chain
  to the U-bottom cell 156792, h 2,741, replayed 13/13 -> runs/_stitch/lineA.npy on the box) + the policy's own
  greedy climb from it (rollB, 0.9 s before its first wall), x1.0 / 1.04 / 1.08, view-follow, 8 ms.
* GE-C (10 min, 102 roots): deepest 24,656 = 56.86% (needs < 24,579 = 57.0%); 1,153 cells in the upper room with
  h up to 2,657 (search 1's archives: <= 2,395), 6 cells at z >= 2,300 (h <= 2,346). GE-D (8 min, rooted at GE-C's
  48 most energetic upper-room cells, x1.0): nothing past 56.86%, no cell above z 2,300.
* The geometry past the upper room (core hull + teleport slices, 32 u grid): from the upper room (z 2,150-2,295) the
  only way up is a column at x -2,320..-2,224 (z ~2,300) into a high area (z 2,380-2,450, x -2,480..-2,224,
  y 2,366-2,686), and from there narrow corridors run SOUTH at z ~2,390 (x -2,480..-2,448 and -2,256..-2,224, y
  1,982-2,334) - the field's 57.0-58% path. The record (ANALYSIS ONLY) does the whole climb in ONE ballistic rise:
  up the U's north wall (vz +1,243 at z 1,484), into the shaft at (-2,411, 2,568, 2,020) with vz +874, through the
  upper room and the column to z ~2,490 (2,020 + 874^2/1,600 = 2,497), then south down the corridor. Free air cannot
  turn horizontal speed into height, so every explorer chain that enters the upper room sideways is stuck under the
  column whatever its energy: the gate needs the U exit to throw the speed straight up the shaft (vz ~+900 at z
  ~2,000; ours: ~+350..+470 at z ~1,940, max rise ~2,070).
* A cell-key gap: GE's 64 u x energy cell does not tell a state rising at vz +900 in the shaft from one crossing it
  sideways with the same |v|, and under --exact-chains the first one found keeps the cell. GE-E (21:32 UTC, 10 min)
  = 147 roots (chain A's last 1.3 s at the energetic U-bottom + the policy's U exit from it, x1.0 / 1.04 / 1.08) with
  `--cell-heading 1` (the existing heading key with one sector: only the rising / not split, x2 keys).
* **GE-E (rising split) passes the climb: 62.25%.** 241,622 cells, 3,627 on the upper level (z >= 2,300; GE-C 6,
  GE-D 0), deepest d 21,577 = 62.25% (the 56.86% wall) - but from ONE root only: root 139 = the policy's own U-exit
  state rollB[64] x1.08 (|v| 1,385 -> 1,495); every x1.0 / x1.04 root stopped at 56.2-56.84%. The chain (ge_replay
  46/46, 6,292 ticks) goes up the shaft in one rise to z ~2,378 at 56.9%, then CRAWLS the upper level at 100-260 u/s
  (energy height ~2,400) for ~40 s to 60.5%, then 61-62.25% at ~600 u/s. The record (ANALYSIS ONLY) crosses 57-60.6%
  in ~5 s at 650-850 u/s with h ~2,800: the line is energy-starved past the climb.
* Spine `autoresearch/spines/pbr5b2_e1.npy` + keys (10,952 rows; tools/stitch_spine.py, provenance in the spines
  README): c1-search-1 beam line to 4,372 (51.2%) + stage-1 GE chain A (root x1.04) to the most energetic U-bottom
  cell (h 2,741) + the policy's own greedy rollout from it (64 ticks) + GE-E's chain (root x1.08) to 62.25%. Energy
  along it: 52% 3,097 / 53% 3,055 / 54.5% 2,742 (the beam line: 2,770 / 2,379 / 2,380).
* **pbr5b2_c3** (21:42 UTC): c2's 24.162B weights (md5 e72262ca...) + spine e1 + its keys + milestones: the 14 + 25,121
  (k05's line, crossed by the greedy) + 24,479 / 23,400 / 22,500 (57.2 / 59.05 / 60.6% along e1: the climb and two
  partial payments, so the success line at 62.07% is not the only one) + c2's other flags (--respawn-greedy-energy 3,
  --auto-spine-delay 4e8), 4e9 steps.
* Lesson so far (generic): (1) Go-Explore's cell needs the vertical-velocity sign at a CLIMB - a rising state and a
  sideways one share a 64 u x energy cell, and the first found keeps it; `--cell-heading 1` (rising / not) found
  the shaft rise where four ladder stages and two focused runs did not. (2) "Pick by energy, then hand over": the
  stage-1 archive held a U-bottom state with +360 u of energy over the policy's line; the policy rolled out from it
  and GE rooted on that rollout carried it up the shaft. Neither the policy nor the explorer did it alone.

### c3 stalls in e1's crawl; c4 on the cut spine (21:44-22:12 UTC)
* c3 (spine e1 to 62.25%): the curriculum walked 10,951 -> 10,436 in a few minutes and then sat for ~20 min at band
  [10,431,10,435] (0.0-1.4%; [10,436] 41-47%): an explorer impact at 61.5% (-185 u), far inside the 40 s crawl and
  5% past anything the greedy flies; every band before it has to pass it to reach the 62.07% success line. Greedy
  evals 56.1% median (d 25,082-25,094) all along, best 56.25 / 56.14. Stopped at 24.626B.
* Spine `pbr5b2_e1c` = e1 cut at row 5,004 (57.20%, d 24,459, just past the climb on the upper level; success line
  57.02%): the next gate only (round 2's lesson: move the success line one gate at a time). Its x1.08 root is row
  4,661 (row 4,660 = the policy's own U-exit state), 345 rows before the end.
* **pbr5b2_c4** (22:06 UTC): c3's 24.626B weights (md5 3a2d4848...) + e1c + keys + the 16 milestones (..., 25,121,
  24,479) + c3's other flags, 4e9 steps. Expect a stall at band [4,656,4,660] (unboosted, before the x1.08 root);
  round 4's gate 15 was passed by the greedy ~150M steps after an x1.08 swap.

### c4: the greedy finds the shaft, not the energy; the descent needs teaching for ENERGY (22:06-22:36 UTC)
* c4 (spine e1c to 57.2%) evals 56.118 / 56.146 / 56.193 / 56.222 / **56.579** (24.929B, median 56.24%) / 56.233 /
  56.216: 2/9 episodes now go UP THE SHAFT into the upper room (56.58%, then wander there until the patience cut;
  another reached the shaft at 56.35%). Handoff for 56.579: c4's snapshot ckpt_24826609664 (pulled, md5
  8e092b45...; c4's ckpt_latest 25.172B pulled too). The curriculum reached the x1.08 root within ~15 min and sat
  there: band [4,664,...] (boosted) 73%, band [4,659,...] (incl. the policy's own U-exit rows 4,659-4,660) 2.4-2.9%.
* Physics of those episodes: energy height at the U-bottom 2,301-2,333 (record ~3,040, ANALYSIS ONLY), into the shaft
  at z 2,000 with vz +349 (record +874), top z ~2,139; the column to the high area needs z >= ~2,380, i.e. >= ~800 u/s
  vertical at the shaft: ~200-450 u of energy the policy's line does not have.
* Where it is lost (beams, probes): every policy line has three impacts over 50-53% - a wall at 50.2% (-200..-346),
  the slanted func_wall/conveyor contact at 51.7% (-310..-497; the record loses ~300 there too) and a floor landing
  at 52.8% (-353..-370; the record flies over it). Forked policy beams 0.5-1 s before the 50.2% wall with small held
  view offsets (+-10 / +-20 deg, protected 4-5 generations, dh K 2-3 ranking): every forked lineage dead, the same
  three impacts in the frontier line - the policy's descent is a narrow tube.
* No booster: the only pushes near the climb are 30-50 u/s sheets (trigger_push *212 / *213); the record's one-frame
  +1,905 vz at 51.7% is a demo artifact that cancels the next frame.
* The searched cleaner descent: in search 1's stage-1 archive the energetic U-bottom states (h 2,512-2,741) all come
  from x1.04 roots at the beam line's rows 4,370-4,372 (51.2%: +4% speed = ~+70 u there); x1.0 roots reach <= 2,389
  (GE-F, unboosted roots over 49.5-51.6% with the rising split, 12 threads beside c4: <= 2,017 / 2,604 at x1.04, null).
  Chain A (x1.04) keeps ~600 u more than the policy's line through 51.7-54.5% (h at 53%: 3,055 vs 2,379).
* **`train_fast --demo-success-energy M`** (0739d2d; train_fast.energy_gate; tests/python/test_demo_success_energy.py:
  the gate, print/dump/refusals, flag off bit-identical to the trainer at 10456fb): a spine episode succeeds and is
  paid the bonus only if it crosses the success line with energy height >= the spine end's - M. A spine found for its
  energy and no deeper than the greedy is otherwise a trivial success. Generic (the spine's own end state).
* **pbr5b2_c5** (22:36 UTC): c4's 25.172B weights (md5 bb43ff40...) + spine `pbr5b2_e1d` (e1 to chain A's end, 4,597
  rows, end h 2,741; success line d(end)+100 at 54.4%) + `--demo-success-energy 200` (h >= 2,541 at the crossing; the
  greedy has ~2,300) + the 16 milestones + c4's other flags, no --auto-spine (refused with the energy rule). 30-min test:
  does the curriculum walk chain A back to its x1.04 root, and does the greedy's U-bottom energy rise?

### c5: the energy-gated curriculum teaches the descent - 56.711 (2026-10-25 23:11 UTC)
* c5 evals 56.165 / 56.177 / 56.161 (25.55-25.70B, energy at 53% ~2,330 as before), then **56.711** (25.776B, median
  56.51%) / 56.709 (25.852B): the greedy now flies chain A's descent from its own state - energy height at first arrival,
  median of the episodes: 53% 2,331-2,352 -> **2,864-2,897**, 54.5% ~2,310 -> 2,661-2,672, 56% ~2,300 -> 2,591-2,593 (+~300 u
  into the climb). 7/9 episodes of the 56.711 eval reach the upper room (56.47-56.71%, z up to ~2,264; before ~2,130);
  they top out at x ~-2,363..-2,392, 40-70 u WEST of the column (x -2,320..-2,224) that leads up. Two episodes fell back to
  the 24% crawl (gate 10) in that eval: some erosion.
* The curriculum: chain A's end (4,596) -> its edge-pass bands (4,466 / 4,461: 25 -> 39%) -> the key change at 4,451
  (0.1% for a while) -> past chain A's x1.04 root (4,372) -> the beam rows at 38-83% (row 3,721 at 25.86B), in ~35 min.
  The x1.04 root band did NOT stall (an energy-gated success rewards arriving with the energy, and the policy found it
  from its own unboosted state).
* Handoff for 56.711: c5's snapshot ckpt_25772425216 (3.7M steps before the eval; md5 25a4f93c..., pulled).
* So: **a spine found for its energy needs a success rule that asks for the energy** (`--demo-success-energy`): the
  depth rule would have counted every crossing (the greedy already flies past chain A's end), and a boosted depth spine
  (c4's e1c) stalled at its root. Generic: the margin is relative to the spine's own end state.
* 23:13 UTC: c5 paused at 25.876B (ckpt md5 4728837e...); auto_spine.py by hand on its weights (beam, branched beams at
  4,970 / -1 s / -2 s + energy forks, GE stage 1 x1.0 / 1.04 with the rising split, 8 min): is the column climb now in
  reach of the policy's own (energetic) line?
* **The search on c5's weights: the PLAIN beam reaches 62.34%** (d 21,524 at tick 6,398 = 49 s; 23:13 UTC, ~1 min, no
  branched beam or GE needed): with the descent learned the policy's own population climbs the shaft unboosted -
  energy height 53% 2,962 / 54.5% 2,794 / 56% 2,709, up the shaft into the high area by 56.9% (z 2,378), then over the
  upper level at 270-350 u/s to 62% in 11 s (GE-E's x1.08 chain crawled it in ~40 s). Spine
  `autoresearch/spines/pbr5b2_s6.npy` + keys (the beam's own). The energy curriculum unlocked the climb for the loop.
* **pbr5b2_c6** (launched 23:13-23:15 UTC): c5's 25.876B weights (md5 4728837e...) + spine s6 + its keys (success line d 21,624 =
  62.16%) + milestones (the 16 + 23,400 / 22,500 along s6) + --auto-spine (quorum / patience 1.2e9; GE with the rising
  split by default now) + --respawn-greedy-energy 3, no energy rule (s6 is a depth spine).

### GATE 17 PASSED by the greedy line: 61.566 (2026-10-25 23:36 UTC)
* c6 evals (spine s6 = the policy's own 62.34% beam line): 56.176 (25.877B, the resume) / **61.505** (25.952B, ~75M steps
  after the launch) / 61.487 / **61.566** (26.103B, median 61.41%) / 61.500 / 61.448: 9/9 greedy episodes up the shaft,
  over the upper level and south along a floor strip; energy height at 56% 2,690-2,758 (c4: ~2,300). Handoff for 61.566:
  c6's snapshot ckpt_26076512256 (27M steps before the eval; md5 9a679e41..., pulled).
* Gate 18 (where all 9 die, 61.3-61.6%): bunny-hopping south on a floor strip at z ~2,377 (feet 2,341), the greedy drifts
  40-60 u east of the beam line (x ~-1,980 vs -2,020..-2,060) off the strip's edge onto trigger_teleport *211 (a fall
  catch, z 2,317-2,343). The curriculum sits right there (band [6,158,6,162] 0.6-0.7%, [6,163,...] 56-59%). The beam line
  itself ends in a fall from 61.9% (z 2,341 at (-1,990, 2,238)) to 62.34% (z 2,145, heading west): the gate after.
* The chain of this gate, for the record: (1) search: Go-Explore with the rising / not cell split found the climb but only
  from an x1.08 root, and a cleaner descent only from an x1.04 root; (2) a depth curriculum on the stitched boosted line
  (e1c) taught the policy to AIM at the shaft (2/9 into the upper room, 56.579) but stalled at the x1.08 root; (3) an
  ENERGY-gated curriculum on the descent alone (e1d) taught the energy in ~20 min; (4) on those weights the policy's own
  plain beam made the climb unboosted (62.34%); (5) the usual depth curriculum on that line -> 9/9 within ~75M steps.
* Gate 18, probed (record_ckpt, c6's 26.76B weights, single s6 rows): from rows 6,150 / 6,158 the policy dies within
  0.1-0.2 s ON the beam line's x (-2,019..-2,029) at y ~2,444, z 2,363 (the teleport sheet under the strip); from row
  6,165 it goes on to 62.29%. The strip has a gap at y ~2,450: the beam lands at y ~2,470 (rows 6,160-6,170: vz -208 ->
  +194, duck released) and hops over it; the greedy and every band spawn before the landing do not jump - a timing
  gate, not a drift. c6's band [6,158,6,162] at 2.2% for ~25 min; evals flat at 61.4% median (26.4-26.7B).
* c6's own patience search 1 (27.085B, 38 s): the plain beam on c6's weights to d 21,536 (62.32%) - a faster, more
  energetic line of the policy's own (61% at row 5,542 vs s6's 6,042; energy height at 57% 2,729 vs 2,475) that crosses
  the gap stretch at x ~-1,993..-2,003 -> spine k01 swapped in (success line 21,636; s6's 21,624 became a milestone).
  c6 evals 61.41-61.53 (26.86-27.16B), median 61.3-61.4%; the curriculum on k01 at the gap band (row 5,674: 27.8%).

### pbr5b2 round 5 lesson (DRAFT, 00:25 UTC; finalised at wrap-up)
* **Score 52.937 -> 62.367** (c6 29.048B, median 62.34%; 81 valid evals). Gates passed: k05's descent to 56.2% (c1,
  the inherited loop: 56.341), the 56.6% shaft climb (gate 17: 56.579 -> 56.711 -> 61.566), the 61.5% gap hop (gate
  18, by the loop's own k01/k02 lines and the curriculum: 61.731 -> 62.264 -> 62.367). Stopped at gate 19 (62.3%).
* **What worked, in order of effect:**
  1. **An energy-gated curriculum (`train_fast --demo-success-energy M`)** for a line found for its ENERGY, not depth:
     Go-Explore's chain A flew the 51.7-54.5% descent ~600 u of energy height cleaner than the policy's line but no
     deeper; with the depth rule every crossing counts (the greedy already flies past it), with M = 200 a spine episode
     succeeds only if it arrives with the spine end's energy. ~20 min / ~400M steps: the greedy's energy at 53% went
     2,330 -> 2,880, and it walked past the chain's x1.04 root, where every boosted DEPTH spine of this lineage stalled.
  2. **The rising / not split in Go-Explore's cell key** (`explore_phase1 --cell-heading 1`, now auto_spine's default
     `--ge-cell-heading 1`): a state rising up the 56.3% shaft and one crossing it sideways shared a 64 u x energy cell;
     four ladder stages and two focused runs stopped under the climb; with the split GE-E went up it (62.25%).
  3. **Energy first, then the policy's own search:** on the weights that had learned the descent, the PLAIN beam made
     the climb unboosted (62.34%); a depth curriculum on that line put 9/9 greedy episodes past the climb within ~75M
     steps (56.71 -> 61.51). Boosted GE spines only taught the AIM (c4: up the shaft into the upper room).
  4. Tools: `tools/stitch_spine.py` (beam / GE chain / policy rollout segments -> one spine + keys; e1, e1c, e1d),
     `--respawn-greedy-energy N` (greedy-line spawns before the line's largest energy losses; no measurable effect on
     its own here), energy forks and E stages in auto_spine (nulls at their first use; the E stage left the default).
* **What did not:** energy-weighted beams (`--score dh`, with or without keys temperature: the same three impacts);
  branched beams forked before the impacts (every forked lineage dead: the descent is a narrow tube for the policy);
  Go-Explore from pre-crash roots (the uniform explorer adds its own crashes); a curriculum on the full stitched line
  e1 (stalled at once on an explorer impact 5% past the greedy); a boosted depth curriculum (e1c: stalled at its
  x1.08 root, as round 4's k02 did).
* **Generic rule learned:** when the gate is energy, find WHERE the energy is lost (the line's largest one-tick losses
  of z + |v|^2/2g vs a line that keeps it), find a line that keeps it (the archive's most energetic non-plunging state
  at matched depth, often from a slightly boosted root), and teach THAT with an energy success rule; then let the
  policy's own search take the next gate.
* c6 evals 27.39-27.61B: 61.447 / **61.731** (27.462B: ep0 / ep5 hop the 61.5% gap and die further along the strip at
  y ~2,350, 61.72-61.73%) / 61.395 / 61.444; medians 61.27-61.32% (the rest still at the gap). Handoff for 61.731: the
  snapshot at or before it, ckpt_27278180352 (md5 600c77a8..., pulled; ckpt_27478458368 16M after it pulled as well).
  The k01 curriculum's gap band [5,669,...] at 28.4-29.9% (the 30% rate).
* c6 evals 27.69-28.14B: 61.561 / 61.489 / 61.443 / **62.257** (27.915B) / 61.470 / **62.264** (28.066B: 3/9 hop the
  gap and ride k01 to its end, falling at 62.25-62.26% into a catch at z ~2,052, x ~-2,155) / 61.536; medians
  61.3-61.4%. Handoff for 62.264: the snapshot at or before it, ckpt_27879014400 (md5 7675bd94..., pulled;
  ckpt_28079292416 13M after it pulled too). The k01 gap band passing at 31-32% (row 5,664).
* Gate 19 (the fall at 62.3%, ANALYSIS of the record only as a yardstick): the record leaves (-2,013, 2,234, 2,290) at
  ~900-1,000 u/s heading west with energy height ~2,800 and is at (-2,853, 2,477, 2,082) 1 s later (63.7%); our line
  leaves the same spot at 400-720 u/s with h ~2,515 and drops into a catch. Energy again (~300 u).
* c6's patience search 2 (28.293B, 82 s): the branched beam to d 21,550 (62.29%) -> spine k02, no deeper than k01 (the
  greedy median had moved to 22,052, so +502 u passed the gain). Every policy line ends in the 62.3% fall: gate 19 holds
  for the plain and branched beams. Energy along k02: 57% 2,730 (680 u/s) / 59% 2,514 / 61% 2,438 (228 u/s) / 62% 2,491;
  the record (ANALYSIS ONLY) ~2,830 throughout 57-61%: our line hops the upper-level corridor and loses ~300 u to the
  landings where the record surfs its walls.
* The gate-17 recipe again: GE-G (01:34 UTC, 12 min, 12 threads beside c6, rising split, view-follow, 8 ms) rooted at
  k02's own upper-level states over 57-61.5% (47 x1.0 / 1.04) - an energetic corridor line to teach with
  --demo-success-energy, then the policy's own search for the fall.
* c6 evals 28.44-28.82B: **62.314** / 61.484 / 61.495 / 61.581 / **62.358** (28.746B) / 62.294; medians 61.25-61.54%.
  Handoff for 62.358: ckpt_28680126464 (md5 854471c7..., pulled).
* GE-G (12 min, 12 threads, roots on k02's 57-61.5% x1.0 / 1.04): deepest 61.69%; its most energetic non-plunging cells
  beat k02's line by only +70..+160 u over 58.5-61.5% (e.g. 2,617-2,633 vs ~2,480). GE-H (10 min, roots on k02's last 4 s
  before its end x1.0 / 1.04 / 1.08 - which collapsed into 48 cells: a 4-8% boost at 500-700 u/s stays in the same 32 u
  energy bucket, so the boosted copies were dropped as duplicates): deepest 62.32%, every deep cell falling into the
  catch at z ~2,050.
* Gate 19 (the fall at 62.3%): the record (ANALYSIS ONLY) leaves the strip's end at vx ~-720 and catches a west-running
  surf ramp at (-2,234, 2,273, ~2,097) (62.5%) at ~1,040 u/s; our lines leave at 330-400 u/s horizontal and drop short
  into the catch, ~80-100 u east of the ramp. Where the energy goes (k02, 57% -> its end): -1,388 u of losses (13 hop
  landings / wall hits of 42-160 u, vz ~-280 -> 0 on the corridor floor, and ~600 u of small ones) against +1,168 of gains
  (strafes, jumps): the policy lands and bunny-hops at 300-400 u/s; the record stays AIRBORNE above the corridor floor
  (z 2,415-2,576) at 650-850 u/s. The missing skill is staying in the air through the corridor.
* GE-I (01:56 UTC, 15 min, 12 threads beside c6): roots on k02's 57-59.5% (43 states, still at h ~2,700) x1.0 / 1.08 (x1.04
  collapses), --cell-energy 16 + the rising split: an airborne, energy-keeping corridor line to teach with
  --demo-success-energy.
* c6 evals 28.90-29.05B: 62.295 / 62.359 / **62.367** (29.048B, median 62.34%: 7/9 at k02's end) -> the quorum fired
  search 3. Handoff for 62.367: c6's ckpt_latest saved for that search at exactly 29,047,652,352 (the eval's step),
  pinned as runs/pbr5b2_c6/ckpt_29047652352.pt (md5 2411cd53..., pulled).
* GE-I (15 min, roots k02 57-59.5% x1.0 / 1.08, --cell-energy 16 + rising split, 12 threads): a null - deepest 61.61%,
  its unboosted cells within +0..90 u of k02's energy.
* **Why the corridor is slow: the core runs CS 1.6's stock bunny-hop cap** (src/pm.c pm_prevent_mega_bunny_jumping,
  `enable_bhop_cap` 1, train_fast --bhop-cap default 1): a jump from the GROUND above 1.2 x 250 = 300 u/s rescales the
  velocity to 0.8 x 300 = 240 u/s. Every hop of the policy's corridor / strip line pays it (hence 240-400 u/s there);
  the record (ANALYSIS ONLY) never takes off from that floor - it stays airborne above it. The WR server's setting is
  unknown (not in its movevars; CS 1.6 default on), so the stock cap stays: the gate needs a line across the upper
  level without ground jumps. (Not a reward or map constant: a physics fact worth knowing for every later gate.)
* Search 3 (29.048B): plain and branched beams 21,485-21,496 (62.40%); GE stage 1 61.68%; x1.08 / x1.12 stages next.
* Search 3 (29.048B, 1,729 s): NO spine. Beams 21,488-21,500; GE x1.0/1.04 21,898, x1.08 21,550, x1.12 21,506 (62.4%)
  - every stage rooted at line row 5,679 (61.5%): the stall rule (first tick within 500 u of the best d) fires early
  on a line that CRAWLS to its end, so the roots (4 s before the stall) cover 58-61.5% and miss the strip's end and
  the drop entirely. A generic auto_spine flaw (fix below).
* GE-G's "energetic" corridor cells (h 2,616-2,641 at 60.5-60.9%) are slow chains (e.g. 1,665 ticks, 12.8 s, from
  59.5% to 60.88% where the line takes ~5 s): their energy is HEIGHT gathered while wandering, ~100 u above the floor.
  Restricted to chains no slower than the line, nothing beats the line's energy. A stitched spine to one of them
  (runs/_stitch/spine_g.npy on the box, gone) was not trained.
* c6 last evals 62.343 / 62.354 / 62.363 (29.12-29.27B), medians 62.30-62.33%; the training frontier flat (front band
  >60.8%, front-perturb best 35,640 u on the trainer's field the whole last hour). **Box 57714456 released 02:45 UTC**:
  the branch's best (62.367, c6 29.048B, hand-on ckpt_29047652352 pinned + pulled) stands, every search ends at the
  62.3% drop, and the program's credit above the reserve (~$5.5) is what round 6 runs on.

### Gate 19, offline (02:49 UTC, no box): two lanes; and two loop fixes
* The record (ANALYSIS ONLY) does NOT use the west floor strip at 61.2-62% that every policy line hops (x -1,980..-2,060,
  with the 61.5% gap): from (-1,944, 2,945, 2,433) at 60.64% it flies SOUTH down an EAST lane at x ~-1,830..-1,944,
  airborne over the fall-catch sheet (trigger_teleport *211, x -2,150..-1,793, z 2,317-2,343) at ~670 u/s, descending
  2,442 -> 2,290, and curves west to (-2,013, 2,234) and onto the west-running ramp at (-2,234, 2,273, ~2,097). GE-G's
  one fast cell at 61.0-61.5% (583 u/s at (-1,865, 2,698, 2,404), an x1.04 root) sat in that lane; its chain went no
  further. Gate 19 = the airborne east lane at ~600-700 u/s, which needs the corridor before it flown fast (the policy
  slows from ~500 u/s to ~230 in the corridor's last turn at 60.5-61%, and the bhop cap makes the strip's hops cost speed).
* Two generic flaws of the trainer's own search, FIXED (87f3097; tests/python/test_ge_roots_keep_all.py,
  test_auto_spine.py): (1) `explore_phase1 --roots-keep-all` (auto_spine passes it for stages with more than one
  --ge-boost factor): boosted copies of a state usually share their original's 64 u x energy cell and were dropped
  before exploring - a mixed stage ran unboosted (GE-H 144 roots -> 48 cells); (2) `auto_spine.tail_roots`: a line that
  crawls on past its stall dies at its END (search 3's three stages all rooted at 58-61.5% while it fell at 62.3%,
  1.7 s past the 'stall') - half the roots now go before the stall and half before the end.
* **To continue (next round)** - resume the 62.367 weights (runs/pbr5b2/pbr5b2_c6/ckpt_29047652352.pt, the hand-on,
  tick 7.67 ms carried) with c6's flags. c6's last spines k01 / k02 lived in the box's runs/pbr5b2_c6/auto/ (box.py
  pull does not fetch it) and died with the box: start from `autoresearch/spines/pbr5b2_s6.npy` + `.keys.npy` (the
  policy's own 62.34% line, success line 62.16%) - the greedy is already past it, so the first eval's quorum fires
  the trainer's own search, which makes the next spine:
  `box.py launch --resume --self-states "<the spines README provenance>" -- --maxvel 3500 --ckpt-every 2e8
  --demo-file autoresearch/spines/pbr5b2_s6.npy --demo-keys autoresearch/spines/pbr5b2_s6.keys.npy --demo-window 5 --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5
  --demo-rate 0.3 --demo-front-frac 0.8 --demo-success-d 100 --demo-success-bonus 50 --demo-milestones
  44939,43426,42182,41015,40497,39943,37251,33777,32260,31429,30286,28818,28783,27001,25121,24479,23400,22500
  --keys-hold-spawn-duck --yaw-entropy 0 --demo-front-temp 0.5 --demo-front-ticks 40 --demo-front-view 6
  --respawn-greedy 0.2 --respawn-greedy-window 0.25 2.0 --respawn-greedy-stuck 200 --respawn-greedy-contact
  --respawn-greedy-energy 3 --respawn-greedy-temp 0.5 --respawn-greedy-temp-ticks 100 --respawn-greedy-temp-view 6
  --respawn-front-perturb 0.2 --respawn-front-perturb-band 0.005 --respawn-front-perturb-speed 1.0 1.3
  --respawn-front-perturb-turn 90 --energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0 --no-respawn-novel
  --respawn-backward-mix 0 --int-real-only 0 --spawn-states autoresearch/spines/pbr1b1_entry_ge_states.npy
  --spawn-states-frac 0.2 --auto-spine --auto-spine-milestones --auto-spine-gain 500 --auto-spine-ge-minutes 8
  --auto-spine-retry 4e8 --auto-spine-patience 1.2e9`. For gate 19 by the gate-17 recipe: a GE (rising split, x1.0 /
  1.04 with --roots-keep-all) rooted at the policy's corridor states from ~59.5% that keeps >= ~550 u/s through the
  corridor's last turn into the EAST lane; then a spine to that chain with `--demo-success-energy` (M below the gap
  between the chain's end energy and the greedy's), then the policy's own beam.

### Playbook: an ENERGY gate by the gate-17 recipe (pbr5b2; all tools in the repo)
1. Diagnose: the greedy's energy height h = z + |v|^2/1600 at first arrival per progress level
   (autoresearch NOTES scripts: per-eval h at matched progress) and the policy beam line's largest one-tick losses;
   the gate is energy when the line arrives where it dies with too little h (a climb it cannot make, a gap it cannot
   clear) and loses it to impacts upstream.
2. Search with energy in mind: `explore_phase1 --exact-chains --cell 64 --cell-energy 32 --cell-heading 1
   --view-follow=-90,-60,-30,-15,0,15,30,60,90 --tick-ms 8 --roots-keep-all` rooted at the policy line's own states
   BEFORE its largest losses, x1.0 / 1.04 (10-15 min, the whole CPU if you can pause the trainer).
3. Pick: `tools/energy_pick.py --archive A --line <beam npz or spine> --field F --lo L --hi H --out pick.json
   --replay chain.npy --map M` (the most energy over the line at matched progress, chain not slower than the line).
4. Stitch: `tools/stitch_spine.py --out spine.npy --field F --no-cut beam:<beam npz> ge:A:chain.npy` (+ keys).
5. Teach the ENERGY: resume with `--demo-file spine.npy --demo-keys spine.keys.npy --demo-success-energy M` (M below
   the gap between the chain end's h and the greedy's there; 200 for a ~400-600 u gap) and NO --auto-spine (refused
   with the energy rule). Watch the greedy's h at the chain's end per eval; round 5: ~400M steps (~20 min).
6. Then the policy's OWN search on the new weights (tools/auto_spine.py by hand, or relaunch with --auto-spine): at
   gate 17 the plain beam made the climb unboosted the first time; a normal depth curriculum on that line followed.
Things that did NOT work at energy gates (round 5): energy-weighted beams (--score dh), branched beams forked before
the impacts (the descent was a narrow tube: every fork died), GE from pre-crash roots with the energy STAGE as a
depth search (the explorer adds its own crashes), a depth curriculum on a boosted stitched line (stalls at the root).

### Harness issue: `box.py pull` does not fetch a run's auto-spine spines
`box.py pull --run R` fetches run.json, progress.csv, the launch log, the greedy evals and checkpoints, not
`runs/R/auto/k*/spine.npy` / `spine.keys.npy` / `spine.json`. With `--auto-spine` the run's CURRENT spine and its keys
live only there, so a released box takes them along (pbr5b2 lost c6's k01 / k02 this way; the next round has to
rebuild them with one search). Workaround on the branch side: scp `runs/R/auto/` (or `auto/current.json` + the
current spine) before `box.py release`. The operator may want `pull` to fetch `auto/*/spine*` and `auto/current.json`.
* Regression pass (02:58 UTC, local CPU, 5 min): 34/34 - test_auto_spine (incl. the trainer smoke + flag-off
  bit-identity + milestone tests), test_respawn_greedy_energy, test_demo_success_energy, test_stitch_spine,
  test_ge_roots_keep_all, test_energy_pick, test_respawn_greedy_contact.

### Gate 19 needs ~+300 u of energy at the strip's end, not ~+50; GE --select-energy (03:57-04:14 UTC, no box)
* `box.py rent` REFUSED at 04:04 (and 04:13): the fleet burns 0.863 $/h of the 1.2 cap (3 boxes) and the cheapest
  passing offer is a 5080 at 0.387 / a 4080S at 0.410 $/h. Retrying each wakeup; offline work meanwhile.
* **Fix (3a7d523): `auto_spine.tail_roots` never fired on its own case.** At the checkpoint's 7.67 ms tick the 4 s
  root window is 522 ticks and the rule's threshold (half of it) 2.0 s; search 3's line crawled 1.7 s past its
  stall. Now a quarter of the window (1.0 s); test at the real tick. (The keep-all fix only ever mattered for the
  mixed x1.0,1.04 stage: search 3's x1.08 / x1.12 stages were boosted properly and found nothing either.)
* **Probe (local CPU, record_ckpt, c6's 62.367 hand-on ckpt_29047652352, single s6 rows, horizontal velocity
  scaled; a diagnostic bench, never training):** from row 6,342 (the strip's edge, z 2,335, still turning:
  v (-51, 345, -368)) x1.0 / 1.25 / 1.5 / 1.8 all fall into the catch within 0.5 s (62.26-62.35%); from row 6,354
  (heading west, airborne, (-2,015, 2,261, 2,298), 360 u/s horizontal, h 2,501): x1.0 62.35%, x1.25 62.39%, x1.5
  62.57%, x1.8 (626 u/s, h 2,683) 62.61% - past the ramp's x at z 2,049, below it, into the catch. The record
  (ANALYSIS ONLY, as a yardstick) passes the same point (-2,013, 2,234, 2,290) at ~900-1,000 u/s, h ~2,800. So the
  drop needs ~+300 u of energy height where our line has 2,500 - my earlier "~480 u/s" (+50 u) was far too low -
  and no strip-end search at the default x1.0-1.12 ladder (+0..+30 u there) can cross it.
* Where s6 (c5's beam, the line c6 learned) loses it (h = z + |v|^2/1600 at first arrival): 55.5% 2,740
  (1,325 u/s up the shaft) / 56.0% 2,709 / 56.5% 2,559 / **57.0% 2,483** (418 u/s; the largest one-tick loss of
  56.5-61%, -101 u, at row 4,803 = 56.88%) / 58-61% 2,429-2,505 (bunny hops: -953 u of losses, +838 of gains over
  10 s). The record (ANALYSIS ONLY) holds ~2,830 over 57-61%. **Gate 19 is lost at the top of the 56.3% shaft and
  in the upper corridor, 5-6% of the route before the drop**: the line must leave the shaft without the impact
  and cross the corridor without landing (the bhop cap), i.e. keep the shaft's ~2,700.
* **New: `explore_phase1 --select-energy K`** (7d63e22; `auto_spine --ge-select-energy K`, both default 0 =
  bit-identical, tests/python/test_ge_select_energy.py: the weights, the explorer flag-off archive + meta identical
  to 3a7d523 and the flag live, ge_replay exact; the explorer's selftest 99/99): each place's (64 u, every energy /
  heading bucket) most energetic cell is drawn (1 + K) x as often as W = 1/sqrt(C_seen+1). Count-based selection
  expanded GE-G's one 583 u/s east-lane cell no more often than the slow cells around it, and its chain went no
  further. Untested at a gate.
* The run for gate 19 when a box is free (search only, ~25 min): two explorers side by side, 16 threads each,
  --select-energy 3 vs 0 (the control), roots = s6 rows 4,600-4,860 (55.2-57%, the shaft top, h 2,740 -> 2,483)
  x1.0 + x1.04 with --roots-keep-all, --cell 64 --cell-energy 32 --cell-heading 1 --view-follow, 8 ms, 20 min;
  read with tools/energy_pick.py against s6 over 58-62% (cells keeping h >= the line + 150 u, chains not slower
  than it). Then, if one arm finds such a line: stitch + --demo-success-energy (the playbook below), then the
  policy's own beam.
* **--select-energy A/B, local CPU** (04:16-04:24 UTC; explore_phase1 4 min x 2 threads x 256 envs each, same seed,
  96 roots = s6 rows 4,600-4,860 x1.0 / 1.04, --roots-keep-all --cell 64 --cell-energy 32 --cell-heading 1
  --view-follow, 8 ms; read on the scorer's field against s6): K = 3 vs 0 -> deepest **61.41% vs 59.42%**, cells past
  57 / 59 / 61%: **2,782 / 867 / 125 vs 1,324 / 105 / 0**; on-route, not-slower cells with >= +100 / +200 u of energy
  over s6: 511 / 76 vs 92 / 11 (max +256 vs +242, all at 56-58%: the shaft exit, e.g. h 2,725 at 734 u/s at
  (-2,505, 2,457, 2,388) where s6 has 2,476). Past 58% neither arm beats the line (gains median -21..-58, max
  +27..+75) and every deep chain is 3-5x slower than it (e.g. 3,452 vs 852 ticks): the explorer crawls the corridor
  as the policy does. One run each: the flag reaches much further in matched compute; it did not find the
  corridor's energy here.
* c6's own greedy (the 62.367 eval, h / |v| at first arrival): 55.5% 2,790-2,805 / 1,360; 57% ~2,660 / 650-670;
  58% 2,625-2,704 / 608-703; **59% ~2,505 / 440-470; 60% ~2,472 / 310-320**; 61% ~2,465 / 280-315. So c6 (k01 / k02's
  lines) already exits the shaft with the energy s6 lost (s6: 2,483 at 57%), and loses ~200 u at **58-60%** instead.
  The corridor search must root at c6's own states at ~57.5-58.5% (h ~2,680), not at s6's.
* **How the record crosses the corridor (ANALYSIS ONLY, CLAUDE.md section 0; contacts = frames whose vz departs from
  free fall by > 30 u/s):** over 56.5-61.5% (5.7 s) it keeps h 2,773-2,852 at 640-840 u/s horizontal and touches
  something on only 44 of 569 frames, ALL at z 2,391-2,455 - 27-91 u above the flat floor our lines land on (z 2,364):
  short rides up the corridor's sloped sides that turn speed upward (57.7%: hspd 789 -> 728, vz -1 -> +246; 58.6-58.9%,
  19 frames: 840 -> 718, vz -165 -> +384), h nearly kept through each. A half-pipe ridden on its sides; the flat floor
  (and with it the bhop cap) is never touched. Our lines land in the middle and hop (s6 / c6: every large one-tick
  loss over 57.5-60.5% is at z 2,364). The missing skill: ride the sides.
* c6's own corridor states, exactly (local CPU; the GPU line does not reproduce on CPU from the true start - it fell
  at the shaft, 56.36% - nor from s6's 55.5% state, which exits the shaft like s6): record_ckpt --dump-states of c6's
  hand-on weights spawned at its OWN eval row at 56.0% (ep2 of the 62.367 eval; position / velocity / view from the
  .jsonl on an s6 state 38 u away as the template for the bookkeeping fields) -> h / |v| 2,681 / 689 at 57%,
  2,713 / 690 at 57.5%, 2,683 / 685 at 58%, then **2,501 / 427 at 58.5%**, 2,477 / 322 at 60% - the GPU evals'
  profile (died at 61.22%, gate 18, on CPU).
* **Corridor search from c6's own states, A/B (local CPU, 04:32-04:49 UTC; 8 min x 2 threads x 256 envs each, same
  seed, 96 roots = the c6 CPU line's rows 108-300 (57.0-58.3%, h ~2,680-2,713, ~690 u/s) x1.0 / 1.04
  --roots-keep-all, the s6 A/B's other settings; read against the c6 line):** --select-energy 3 vs 0: deepest
  61.41% vs 61.52%; cells past 57 / 59 / 61%: 4,581 / 994 / 161 vs 4,470 / 1,001 / 189; best energy over the line
  at 58-58.5%: +200 vs +167 (12 vs 5 cells >= +100), at 58.5-59%: +26 vs -10, past 59%: nothing either way. The
  best cell takes c6's 58.0-58.2% corner (where c6's velocity turns (-437, -498) -> (-202, +199) and h drops
  2,683 -> 2,473 in 0.34 s) WIDE, on the floor (-2,729, 1,924, z 2,360, vz 0) at 694 u/s - and loses it in the next
  stretch: friction and the hop cap again, not the record's side riding. **Verdict on --select-energy: not
  established** - a large reach effect from the shaft-top roots (61.41 vs 59.42%), none from the corridor roots, a
  small energy effect; one run each, 1/20-1/60 of a box-scale search. Default stays off.
* **For the next round (gate 19, generic):** the corridor 57.5-61% must be flown on its sloped sides, corner by
  corner (c6 loses ~200 u at the 58.0-58.2% corner, ~40 u more by 60%). (1) A box-scale Go-Explore (16-32 threads,
  20-30 min) from c6's OWN corridor states (record_ckpt --dump-states of the hand-on weights spawned at their own
  56% eval row - the procedure above; the CPU line reproduces the GPU profile through 61%), then energy_pick +
  stitch_spine + --demo-success-energy per corner (the gate-17 playbook). (2) Better: automate the playbook in the
  trainer's loop - when the depth ladder finds nothing and the line's largest energy losses lie upstream, an energy
  stage (GE before the losses, picked by energy_pick, handed back as a spine with the energy success rule; lift the
  --demo-success-energy + --auto-spine refusal for it). Every piece exists and is tested; the glue does not.

### The loop's ENERGY stage: `--auto-spine-energy G` (04:55-05:14 UTC, local; c9dd964 + 4c3b20d)
The gate-17 playbook (find the loss, search for energy, pick by energy, teach with the energy rule) was manual; gates
17 and 19 are both energy gates, and a recipe that needs an operator at every energy gate is not one recipe. Now the
trainer's own loop does it, behind one flag (default off; flag-off bit-identity: the auto_spine and
demo_success_energy suites' trainer-before-the-flag tests still pass):
* `tools/auto_spine.py --energy-gain G`: when the depth ladder finds no deeper line, an ENERGY stage runs Go-Explore
  (`--energy-boost` 1.0,1.04, --roots-keep-all) rooted before the line's largest energy losses (energy_roots, as the
  E stage), and `energy_cell` picks the DEEPEST archive cell keeping >= G u more energy height than the line at the
  same progress (energy_pick's filters: on the route, not plunging, not slower than the line; the deepest, not the
  largest gain, because a large gain is often a state a few ticks before the place where the line loses its own -
  the corridor analysis above). Its chain is replayed exactly (ge_replay --cell), spliced onto the line, and the
  spine is marked `energy` (energy_gain, energy_h, energy_window in spine.json).
* `train_fast --auto-spine-energy G` passes `--energy-gain G`; on an energy spine the success rule becomes the
  --demo-success-energy rule with floor = the spine end's h - G/2 (`spine_energy_rule`), the quorum counts only
  greedy crossings that carry it (`crossed_with_energy`: the energy at the first tick past the line - an energy
  spine is no deeper than the greedy, so a plain crossing proves nothing), current.json records `energy_h_min`;
  a depth spine goes back to the depth rule. Refused without --auto-spine, without --demo-success-bonus, or with
  --reward-per-decision; TRAIN_ONLY in record_ckpt.
* Tests: tests/python/test_auto_spine_energy.py (the pick rule; go_explore replays the picked cell and nothing on
  None; the stage runs only when asked and only after the ladder failed; the trainer's two rules; a trainer smoke
  run with the flag - its notice, config key and `--energy-gain` on the search line - and both refusals). Regression
  39/39 (auto_spine, auto_spine_energy, demo_success_energy, ge_select_energy, ge_roots_keep_all, energy_pick,
  stitch_spine, respawn_greedy_energy). End to end on CPU (a smoke trainer, the depth ladder forced to fail with
  --auto-spine-gain 1e6, G = 1): "an ENERGY spine (+7 u over the policy's line): success only with energy height
  >= 3,720" swapped in, then "energy spine - 0/2 greedy episodes cross ... with energy height >= 3,720".
* **Untested at a gate.** The next round's gate-19 run: the to-continue launch (c6's 62.367 hand-on, spine s6) plus
  `--auto-spine-energy 150` (G set once, a physical margin: ~13% of the speed at 700 u/s; the local corridor
  searches found +167..+200 u through the 58% corner in 8 min x 2 threads). Expect the first search's depth ladder
  to fail at the drop and its energy stage to root at c6's corridor losses (58.0-58.2%); the curriculum then
  teaches that corner's energy, and the next quorum (energetic crossings) searches again from the new weights.

### c7: the energy stage at gate 19 (05:41-05:50 UTC, box 57771742)
* The fleet's burn fell to 0.527 $/h at 05:41 and a box fit: RTX 4070 Ti, 0.229 $/h, 32 threads, deadline 07:25
  (`extend --hours` is from NOW, not added: 0.5 h would have shortened it). Deployed aec7ef7 + c6's hand-on
  ckpt_29047652352 (md5 2411cd53... on the box).
* **c7 = resume of the 62.367 weights with the to-continue flags + `--auto-spine-energy 150`,
  `--auto-spine-ge-minutes 5`** (8 in c6: the depth ladder is expected to fail at the drop, as search 3's did, and the
  first test of an idea is <= 30 min of box time), --steps 1.2e9; spines s6 + keys and pbr1b1_entry_ge_states as in c6
  (provenance in autoresearch/spines/README.md; no demo). A WARM resume measuring AT the wall: does the loop's own
  energy stage find and teach gate 19's lost energy? Record gate passed (4 recordings in 24 s).
* First eval (29.048B): 7/9 greedy past s6's success line (d <= 21,624; median deepest 21,536) -> search 1 at once,
  with `--energy-gain 150` on its command line.
* **c7 search 1 (05:49-06:14 UTC, 1,498 s): the loop's energy stage FOUND gate 19's lost energy and swapped it in.**
  Depth: plain beam 21,490 (62.40%), five branched beams (incl. energy forks at ticks 4,072 / 4,230) 21,490-21,497,
  GE x1.0,1.04 / x1.08 / x1.12 21,483 / 21,483 / 21,476 - the tail roots now fire ("the line crawls 1.6 s past its
  stall (tick 5,689) to its end (5,899): roots before both") and still nothing is 500 u deeper. Energy stage: 48 roots
  before the beam line's largest losses (-337 u at tick 4,138, -233 at 4,296, -136 at 4,892), window 48.75-62.40%,
  10,063 candidates; the deepest keeping >= 150 u: cell 251,579 at **59.55%, h 2,701 (+217 u over the line)**, a
  328-tick chain from an x1.04 root at line row 4,814 -> spine k01 (5,142 states + keys), success = crossing d <=
  23,216 with h >= 2,626 (the end's - 75). The policy's line has ~2,480-2,505 there (c6's evals: 59% ~2,505).
  The trainer resumed at 227k steps/s (GPU 99%; progress.csv's time/fps averages over the search's pause).
* **c7's energy spine taught the 58% corner in 75M steps (~5.5 min at 227k steps/s).** Greedy h / horizontal speed
  at first arrival (scorer's field), the eval at the resume (29.048B) vs the first eval on the energy spine
  (29.124B): at 58.2% 2,461-2,519 (one 2,573) / 268-333 u/s -> **2,683-2,751 in 9/9 / 557-665 u/s**; at the success
  line (59.4%) 2,473-2,500 -> 2,633-2,783 in 7/9 (the quorum: "energy spine - 7/9 greedy episodes cross d <= 23,216
  with energy height >= 2,626 (9 cross it)" -> search 2 at once). Then it is lost at the NEXT corner: by 60% every
  episode is back at 2,455-2,496 / 256-364 u/s (the corridor's north end, (-2,690, 2,683) -> (-2,530, 2,891)). One
  corner per energy spine - the loop now walks the corridor corner by corner by itself.
* **Score 62.369** (29.124B, median 62.4%; scorer: best 62.4%, 0/9 finishes) - c6's 62.367 by a hair, same gate (all
  9 still fall at the 62.3% drop). Hand-on: ckpt_latest saved for search 2 at exactly the eval's step (global_step
  29,123,674,112), pulled and pinned as runs/pbr5b2/pbr5b2_c7/ckpt_29123674112.pt (md5 afb44185...). These weights
  carry the corner's energy: the better start for the next round.
* **c7 search 2 (06:20-06:39 UTC, 1,169 s) on the weights that learned the corner: the FIRST LINE PAST THE 62.3% DROP.**
  Plain beam 21,611, branched 21,601-21,614, GE x1.0,1.04 21,490, x1.08 21,454, **x1.12 20,489 (64.15%)** - an 807-tick
  chain from line row 5,017 (~60%, the next corner, where the greedy now loses the energy) with the root's velocity
  x1.12: 1,022 u past the greedy frontier -> depth spine k02 (5,824 states + keys; committed as
  autoresearch/spines/pbr5b2_c7k02 with provenance), success line d <= 20,589. Exactly gate 17's sequence, now by the
  loop alone: the energy first (k01, one corner), then the search passes the gate (here with a x1.12 root).
* The caveat for the next round: k02's root is boosted x1.12 at the 60% corner, and a DEPTH curriculum stalls at a
  boosted root (round 4's k02, c4's x1.08 at gate 17) unless the policy learns to arrive there with the speed - which is
  what the energy rule teaches. A generic fix worth building: when the depth ladder's spine starts from a boosted root
  (x > 1.04), teach the stretch up to the root with the energy rule first (h at the root - G/2), then the depth spine.
* c7 on k02 (06:39-07:07 UTC): evals 29.199 / 29.275 / 29.350 / 29.426B = 62.362 / 62.326 / 62.366 / 62.357, 0/9 past
  k02's success line each time (medians 21,516-21,559); the curriculum band went [5,473 ...] -> [5,423, 5,823] at
  7-47% success, still ~3.5 s of line short of the x1.12 root (row 5,017). Pulled at wrap-up: 6 evals, ckpt_latest
  (29.46B, md5 a17970d6...), snapshots ckpt_29247930368 (md5 6fa55d5e...) and ckpt_29448208384 (md5 ce3d39f0...), the
  hand-on ckpt_29123674112 (62.369, md5 afb44185...), auto/k01 + k02 spines (also in autoresearch/spines/) and
  current_final.json. Box 57771742 released at wrap-up.

### pbr5b2 round 5 lesson (wrap-up, 2026-10-26 07:09 UTC) - supersedes the DRAFT above
* **Score 52.937 -> 62.369** (c7 29.124B, median 62.36%; c6's 62.367 at 29.048B before it). Gates passed: k05's descent (c1), gate 17 = the 56.6% shaft climb (c4-c6),
  gate 18 = the 61.5% gap hop (c6's loop). Gate 19 = the 62.3% drop: the loop's own search found the first line past it
  (k02, 64.15%) 27 min before wrap-up; the greedy had not followed it by wrap-up (k02's curriculum band at row ~5,423 of 5,823, 3.5 s short of its x1.12 root at row 5,017; evals 62.33-62.37%).
* **What worked, in order of effect:**
  1. **ENERGY gates handled by the trainer's own loop** (`train_fast --auto-spine-energy 150`, `auto_spine --energy-gain`,
     new this round, default off): when no deeper line exists, Go-Explore before the line's largest energy losses, the
     deepest cell keeping >= 150 u more energy height than the policy's line, taught with the energy success rule (h >= the
     end's - 75) with an energy quorum. At gate 19, first use: +217 u at 59.55% found in one search (25 min), learned by the
     greedy in 75M steps (h at 58.2% 2,46x-2,51x -> 2,683-2,751, 9/9); the NEXT search then passed the drop by depth
     (x1.12 GE from the next corner, 64.15%). Gate 17's sequence (energy first, then the search passes) without an operator.
  2. **The energy-gated curriculum itself** (`--demo-success-energy`, round 5's first new piece): gate 17's cleaner descent
     taught in ~20 min where every boosted DEPTH spine of the lineage had stalled at its root.
  3. **The rising / not split in GE's cell key** (`--cell-heading 1`, auto_spine's default): the shaft climb.
  4. Loop fixes: tail roots (a crawling line's end gets roots; the threshold now fires at 7.67 ms) and --roots-keep-all
     (boosted copies are explored, not dropped as duplicates).
  5. Diagnostics that decided where to look: a probe of the policy at its own states with scaled speed (gate 19's drop needs
     ~+300 u: x1.8 at the strip's edge still falls short) and the per-progress energy profile of the greedy (c6 loses
     ~200 u at the 58% corner, s6 at the shaft top) - and, as ANALYSIS ONLY, the record rides the corridor's sloped sides
     while our policy lands on its floor under CS's bunny-hop cap.
* **What did not:** energy-weighted beams; branched beams before the impacts; GE from pre-crash roots; a depth curriculum on
  a boosted stitched line (stalls at the root); strip-end searches at gate 19 (the energy is lost 3-4% earlier);
  `--select-energy` (GE selection by energy: a big reach gain in one local A/B, none in the other - unproven, default off).
* **Generic rules learned:** (1) when a gate is energy, find where it is lost and teach a line that keeps it with an energy
  success rule, then let the policy's own search take the gate - now one flag; (2) energy is lost corner by corner (flat
  floors, the hop cap): one energy spine per corner, ~25 min of search + ~6 min of training each; (3) a depth spine from a
  boosted root asks for speed the policy does not have: teach the stretch up to the root for energy first (not built).
* **Next round:** resume c7's hand-on (runs/pbr5b2/pbr5b2_c7/ckpt_29123674112.pt, 62.369, md5 afb44185...; or ckpt_latest at 29.46B, md5 a17970d6..., mid-way through k02's curriculum) with c7's flags (the to-continue block + `--auto-spine-energy 150
  --auto-spine-ge-minutes 5`) on spine `autoresearch/spines/pbr5b2_c7k02` (+ keys): the loop's current spine, past the drop.

## pbr6b4 (round 6, slot 4) - 2026-10-26 07:53 UTC start

Incumbent: pbr5b2_c7 ckpt_29123674112 (md5 afb44185...), eval 62.369 at 29.124B. Credit at the start: $11.74, i.e.
$1.74 above the $10 reserve for the whole fleet - one well-chosen run, not a sweep. Box 57784754: RTX 4080S, 0.322 $/h,
Germany (machine 56783), rented 08:05 UTC for 2 h.

### Where the greedy loses gate 19's energy: ONE corner, a landing + the bunny-hop cap (08:00 UTC, local, ANALYSIS of our own lines)
* The 62.369 eval (29.124B), h = z + |v|^2/1600 / horizontal speed at first arrival, 9 episodes: 59.5% ~2,720-2,790 /
  640-730 u/s (two episodes already ~2,470); **60.0% 2,455-2,504 / 256-364** (every episode); 61.5% ~2,440 / 205-340.
  The k02 chain (the loop's line past the drop) at the same levels: 2,871 / 842, **2,783 / 685**, 2,721 / 545.
* What happens in the corner (ep0, ticks 5,341-5,350, at (-2,569, 2,894)): the line descends onto the corridor floor
  at z 2,360 (vz -458 -> 0: -140 u), then JUMPS off the ground at 701 u/s and CS's bunny-hop cap (src/pm.c: a ground
  jump above 300 u/s is rescaled to 240) cuts it to 244 u/s (-207 u). ~350 u in 0.03 s; the rest of the corridor,
  the strip and the drop are flown on what is left.
* What k02's chain does there (rows 5,106-5,300): it brushes the corridor's SLOPED side at (-2,745, 2,800, 2,391)
  (vz -223 -> -70, -46 u), rides it up and round (vz +29 -> +365 at y 2,880-2,935) and crosses east at z 2,446-2,489,
  never touching the floor (h 2,783 held from 59.8% to 60.6%); the same side ride at (-1,954, 2,913) throws it up
  (vz +424) into the east lane over the catch sheet. The record's "ride the sides" (round 5's analysis), found by
  the loop's own search from an x1.12 root at row 5,017 (58.61%, |v| 778 -> 873, h 2,770 -> 2,867: +97 u).
* So the skill k02's curriculum has to teach is the side ride at the corridor's north end; the boost at the root is
  +97 u, smaller than the 350 u the greedy throws away in the corner. Plan: continue c7's k02 curriculum (c7's
  ckpt_latest 29.46B, md5 a17970d6..., had walked it back 5,823 -> 5,423 in 28 min) with the loop on and a shorter
  patience (4e8, ~30 min), so the trainer's own search runs on the weights that have learned the corridor.

### c8: c7's k02 curriculum continued, the loop on (08:11-08:15 UTC, box 57784754)
* **pbr6b4_c8** = resume of c7's ckpt_latest (29.499B, md5 a17970d6...; c7's evals 62.33-62.37 there) with c7's flag set
  on spine `autoresearch/spines/pbr5b2_c7k02` + keys, `--demo-tau0 5423` (where c7's curriculum stood at wrap-up), the
  20 milestones of c7's current_final.json (the 18 + s6's 21,624 + k01's 23,216), `--auto-spine-energy 150
  --auto-spine-ge-minutes 5`, and `--auto-spine-patience 4e8` (c7: 1.2e9; ~30 min here, so the loop's own search runs
  once the curriculum has walked k02's chain back toward its x1.12 root at row 5,017), --steps 1.2e9. A WARM resume
  measuring AT the wall (gate 19). Record gate passed (4 recordings in 16 s).
* First minutes: the band at [5,423, 5,428] (61.6%, the east lane's south part) oscillates 1.6-53% per visit.
* c8 evals: 62.366 (29.4996B, the resume) / 62.338 / 62.340 / (29.726B); 9/9 still fall at the 62.3% drop. Its curriculum
  sat ~15 min at band [5,423, 5,427] (12-16%; the band after it 55-58%): k02's chain LANDS there on a surf slope at
  (-1,895, 2,276, z 2,386) falling at vz -484 (-172 u), 0.05-0.08 s after those spawns; then it passed and walked the
  east lane back 5,428 -> 5,368 at 29-40% in ~8 min. Stopped at 29.806B (ckpt_latest md5 d9ffb9c7...): its patience
  search was due at the next eval, and on c7's record (its x1.12 stage) it would have swapped in another k02-like
  chain and restarted the curriculum at that spine's end.
* Local GE (8 min x 2 threads x 256 envs, 20 roots = k02's own beam rows 4,940-5,016 at 58.0-58.6%, x1.0, 64 u x energy
  32 x rising split, view-follow, 8 ms): 106,657 cells, deepest 61.46%, and NO side ride - past 60% its best energy is
  2,527 (the greedy ~2,480, the chain 2,783): +20..+43 u over the greedy's line (energy_pick on a reference line = k02's
  beam rows + the greedy's own continuation). From the policy's unboosted states the corner trick was not found at this
  compute; the x1.12 chain enters the corner at 842 u/s horizontal, the greedy at ~690.

### The starting spine as an ENERGY spine; c9 = k02 cut before its stuck landing (08:20-08:31 UTC)
* `train_fast --demo-success-energy M` was refused with `--auto-spine` (the loop swaps in depth spines). With
  `--auto-spine-energy G` the loop already teaches its own energy spines with that rule and counts only energetic
  crossings for its quorum, so the pair is now allowed there (8a1cccc): the starting spine is an energy spine, a swap
  sets the rule anew; without --auto-spine-energy it stays refused. Test test_an_energy_starting_spine_runs_the_loop
  (7/7 of test_auto_spine_energy pass, CPU 72 s).
* Spine `autoresearch/spines/pbr6b4_k02e` = k02 cut at row 5,420 (61.65%, h 2,721), before the landing band its depth
  curriculum stalled on; success line row 5,399 (61.49%, d <= 22,014) with h >= 2,571 (M 150). The greedy crosses that
  line on the west strip with h ~2,440, so it fails the rule; the chain crosses it in the east lane with ~2,721. The
  curriculum now asks for the corner (59.9%) and the east lane flown with energy, not for the landing and the drop.
* **pbr6b4_c9** (08:30 UTC): c8's 29.806B weights + k02e + keys, `--demo-tau0 5368` (c8's position),
  `--demo-success-energy 150`, milestones + k02's own line 20,589 (paid once per episode on crossing 64%), the loop on
  with `--auto-spine-patience 0` (quorum only: 5/9 greedy crossings WITH the energy) and `--auto-spine-ge-minutes 3`.
  The first launch's trainer died silently after its first eval (log cut mid-line, no traceback, GPU free after);
  the identical relaunch ran. First minutes: band [5,333, 5,420] 99.6% (spawns past the line with the chain's energy
  succeed at once), step 40.

### GATE 19 PASSED by the greedy line: 64.692 (2026-10-26 08:43 UTC)
* c9 (the energy spine k02e) evals: 62.344 (29.807B, the resume) / 62.288 / 62.366 (29.958B) - and at 29.958B the
  energy quorum fired: "energy spine - 6/9 greedy episodes cross d <= 22,014 with energy height >= 2,571 (9 cross
  it)", ~150M steps (~9 min) after the launch. The curriculum had walked k02e back 5,368 -> ~5,270 (the east lane's
  entry, 60.7%) at 30-40% band success.
* **The loop's search 1 (46 s): the PLAIN BEAM on those weights went past the drop**: d 20,132 = 64.77%, no fork, no
  boost, no explorer chain -> depth spine k01 (5,700 states + the beam's own keys; committed as
  autoresearch/spines/pbr6b4_c9k01), success line d <= 20,232. Its line: rides the corridor's side at the 59.9% corner
  ((-2,741, 2,884, z 2,375), vz +220), crosses at z ~2,543 with h 2,816 (the greedy used to land at z 2,360 and keep
  ~2,480), the east lane at (-1,876, 2,796) h 2,791, (-1,931, 2,245) h 2,847 at 61.8%, over the drop at (-2,513, 2,158,
  2,111) 930 u/s (62.85%), on to (-3,004, 2,504) at 64%.
* **Eval 30.033B: score 64.692** (best 64.69%, median 62.25%): 2/9 greedy episodes past the drop to the k01 line's
  end region, ~75M steps after the swap. Hand-on snapshot ckpt_30006575104 (26.7M steps before the eval, md5
  51111d5a..., pulled).
* So gate 19 went exactly as gate 17 did, and this time with the loop doing every search: an ENERGY lesson on the
  stretch where the greedy loses it (here a depth spine's own chain, cut before the band its depth curriculum stalled
  on) -> the energetic quorum -> the policy's own plain beam passes the gate unboosted -> a depth curriculum on that.
  The x1.12 boost of k02's root was never learned around: the energy rule made the corner's side ride pay for itself.
* Budget: $1.03 above the reserve at 1.01 $/h of fleet burn, so no extension can move the 10:09 UTC deadline; c9 runs
  to it and is harvested before.
* c9 on k01 (the policy's own 64.77% line): evals 30.109B **65.391** (best 65.39%, median 62.3%) / 30.184B 64.818 (median
  64.3%) / 30.260B 64.782 (**7/9 past the drop**, median 64.7%) -> the depth quorum fired the loop's search 2 at 30.260B
  (ckpt_latest saved for it, md5 55384f10..., pulled). The k01 curriculum had walked back 5,699 -> ~4,540 (57.5%, the
  corridor before the corner) in ~25 min at 18-53% bands, step 40 on the easy stretches.
* Search 2 (08:56 UTC on): plain beam d 19,937 (65.1%), branched beams 19,937 / 19,732 (65.47%) / ... / 19,949 - the gain
  rule needs d < 19,683 (500 u past the greedy median 20,183); the GE ladder (3 min stages) runs next.
* Search 2's result (427 s): GE stage 1 (x1.0 / 1.04, 3 min) - a chain from the branched beam line's row 5,976 (65.47%) x1.04,
  1,352 ticks, to d 16,970 = **70.31%** -> depth spine k02 (7,328 states, autoresearch/spines/pbr6b4_c9k02), success line
  70.13%. The chain walks a floor north at x -3,256 at 210-260 u/s (65.5 -> 66.4%), hops slowly for ~400 rows at
  (-3,306, 3,560, 2,360-2,385) (66.5%), then drops to z ~2,000 and surfs east at 600-740 u/s to 70.3%. The prefix (the
  policy's branched beam) passes gate 20 itself (the ramp climb at 64.7-65.4%, where 7-9/9 greedy episodes die).
* c9 evals on k02: 64.813 / 64.829 / 64.827 / 64.835 (30.34-30.56B, medians 64.7-64.8%): the drop is now reliable,
  gate 20 holds. The curriculum oscillated at k02's [6,702 / 6,707] (21-55%) - inside the explorer's slow hopping,
  ~730 rows after the chain's root.
* **pbr6b4_c10** (09:16 UTC): c9's ckpt_latest (30.574B, md5 5db99696...) + spine `pbr6b4_c9k02c` = k02 cut at row 6,300
  (66.42%, success line 66.25%: one gate past the greedy, before the hopping) + keys, the depth rule, the 24 milestones
  (+ k02e's 22,014, k01's 20,232, k02's 17,070), NO --auto-spine (a search now would only stop the last evals before the
  10:09 deadline). Question: does the greedy take gate 20 (the climb) within the box's last ~45 min?
* c10 evals (k02c, the depth rule, no loop; 30.574-31.178B, 9 evals): 64.764 / 64.830 / 64.773 / 64.836 / 64.779 / 64.752 /
  64.898 / 64.769 / 64.832, medians 64.7-64.8%: 9/9 past the drop in most evals, every one stopping at gate 20 (the ledge
  catch at 64.8%: the greedy rises to z ~2,240-2,300 at (-3,385, 2,640), the branched beam line to z 2,324 with vz +297
  and lands on a ledge at z ~2,377). The curriculum sat on the chain's floor walk at [6,095 / 6,100] (6.7-31%) the whole
  ~30 min - the walk at exactly 250 u/s (ground speed) on the floor at x -3,256, 65.8% - and never reached the climb.
* **Box 57784754 released 09:47 UTC** (deadline 10:05; the credit at $10.01, i.e. AT the reserve, so no extension).
  Pulled: c8 / c9 / c10 evals and logs, c9's ckpt_30006575104 (the hand-on for the 65.391 eval, md5 51111d5a...) and its
  search-2 save ckpt_latest (30.2598B, md5 55384f10...), c10's ckpt_latest (31.1815B, md5 6955707b...: the newest weights,
  the drop reliable), the auto spines k01 / k02 (also in autoresearch/spines/).

### pbr6b4 round 6 lesson (DRAFT, 09:50 UTC; finalised at wrap-up)
* **Score 62.369 -> 65.391** (c9 30.109B; best episode 65.39%; from 30.26B on 7-9/9 greedy episodes past the 62.3% drop,
  medians 64.7-64.8% over 12 evals). Gate 19 (the 62.3% drop, which held the lineage from 28.07B to 29.96B) passed by the
  greedy line ~150M steps after the change that opened it; gate 20 (a ledge catch at 64.8%) holds. ~1.6 h of box time
  in all (the program's credit hit the reserve at 09:47).
* **What did it, in order:**
  1. **Measure where the energy goes, tick by tick, on the greedy's own eval** (not the beam's, not the spine's): the whole
     gate-19 deficit was ONE corner at 59.9% - a floor landing at vz -458 (-140 u) and a ground jump at 701 u/s that CS's
     bunny-hop cap cut to 244 u/s (-207 u). The loop's k02 chain rides the corridor's sloped side there instead.
  2. **A depth curriculum stalled far from where the greedy leaves the line -> cut the spine before the stalled band
     and teach that stretch with the energy rule.** k02's curriculum sat at a landing band 2% past the corner; k02 cut
     before it (k02e) has a success line the greedy already crosses, so the depth rule would be trivial - with
     `--demo-success-energy 150` it asks for the corner's energy. 150M steps: 6/9 greedy episodes crossed with it.
  3. **`--demo-success-energy` with `--auto-spine-energy`** (new, 8a1cccc): the starting spine can be an energy spine and the
     loop's energetic quorum fires its own search. That search's PLAIN BEAM passed the drop unboosted in 46 s (k01, 64.77%):
     gate 17's sequence (energy first, then the policy's own search), the third time in this lineage.
  4. The depth curriculum on the policy's own line (k01) made the drop reliable (2/9 -> 7/9 in ~225M steps) and fired
     search 2 by quorum, which found the next line (k02, 70.3%, x1.04 GE root at 65.47%).
* **What did not:** a local unboosted GE from the policy's states before the corner (8 min x 2 threads: no side ride,
  +20-43 u); c10's cut of search 2's spine at 66.4% (depth rule): its curriculum sat ~30 min on the explorer chain's slow
  floor walk (6.7-31%) and never reached gate 20's climb. One silent trainer death right after a resume's first eval (no
  traceback; the identical relaunch ran).
* **Generic rules learned:** (1) the gate is where the GREEDY loses energy, measured on its own eval; (2) when a depth
  curriculum stalls at a band past where the greedy diverges, move the success line back before that band - with the
  energy rule if the cut line is no deeper than the greedy, the depth rule if it is deeper (round 2's "one gate at a time"
  plus round 5's energy rule); (3) the x1.12 boost of k02's root never had to be learned around: once the corner's
  energy was there, the policy's own population found its own way past the drop.

### `--demo-stall-cut STEPS`: the two cuts of this round, by the trainer (10:10 UTC, local, no box)
* Today's two interventions were one generic rule applied by hand: a backward curriculum that STALLS on a band (tau
  reaches no new minimum) past where the greedy leaves the line -> cut the spine just before that band; the cut's
  success line is either already crossed by the greedy (k02e: then the depth rule would be trivial, so the ENERGY rule)
  or deeper than it (k02c: the depth rule). `train_fast --demo-stall-cut STEPS` (default 0 = off, nothing runs or is
  dumped; requires --auto-spine and --auto-spine-energy G) watches the curriculum's lowest tau; after STEPS steps without
  a new minimum it cuts the spine --demo-window rows before it, compares the cut's line with the last greedy eval's median
  deepest d (train_fast.stall_cut_plan), sets the energy floor (cut end's h - G) or the depth rule, keeps the old line as
  a milestone, restarts the curriculum at the cut and writes runs/<run>/auto/cutNN.json. The loop's quorum then fires
  its search exactly as it did after k02e (energetic crossings) and k02c (depth crossings).
* Tests (CPU): tests/python/test_demo_stall_cut.py (the rule's two branches and its None cases; a princessburglar
  trainer run that cuts once at row 34 of a 40-state spine; the refusal without --auto-spine-energy; TRAIN_ONLY) +
  test_auto_spine_energy + test_demo_success_energy: 13/13. Untested at a gate. A patience of ~2e8 steps matches what
  I waited by hand (c8's landing band ~15 min at ~300k steps/s before I cut it).
* **To continue (next round, if any):** resume c10's ckpt_latest (runs/pbr6b4/pbr6b4_c10/ckpt_latest.pt, 31.1815B, md5
  6955707b...; or c9's search-2 save ckpt_latest 30.2598B, md5 55384f10...) on the loop's own spine past gate 20,
  `autoresearch/spines/pbr6b4_c9k02` + keys (success line 17,070 = 70.13%), with c9's flag set (the c8 block above with
  `--demo-file autoresearch/spines/pbr6b4_c9k02.npy --demo-keys autoresearch/spines/pbr6b4_c9k02.keys.npy`, no tau0, no
  --demo-success-energy, milestones `44939,43426,42182,41015,40497,39943,37251,33777,32260,31429,30286,28818,28783,27001,
  25121,24479,23400,22500,21624.107,23215.873,20589.457,22013.855,20231.945`, `--auto-spine --auto-spine-milestones
  --auto-spine-gain 500 --auto-spine-ge-minutes 5 --auto-spine-retry 4e8 --auto-spine-patience 0 --auto-spine-energy 150`)
  + `--demo-stall-cut 2e8` - the trainer then makes the k02c-style cut itself when k02's curriculum stalls in the
  explorer chain's slow floor walk (66.5%), and gate 20 (the 64.8% ledge catch, in the branched-beam prefix) is taught
  from the cut. The checkpoint carries the 7.67 ms tick and the keys-hold / potential / view shapes; never pass shape or
  tick flags to it.
* Gate 20 (the 64.8% ledge), offline (10:40 UTC): the greedy (c10's last eval) and the branched-beam prefix of k02 are the
  SAME line to within ~30 u from 63% to 64.7% (h 2,545-2,565, 520-600 u/s horizontal, the same positions); at 64.89% the
  beam line strikes the ledge's edge at (-3,411, 2,654, 2,360) (|v| 599 -> 328, -140 u) and lands on it at z 2,377, the
  greedy meets it a few units differently and falls into the catch at z ~2,066. A precision catch like gate 18's gap hop,
  not energy: rows ~5,700-5,760 of pbr6b4_c9k02 are the band to teach (c10's curriculum never got back past 6,095).
  Also in the regression pass: test_auto_spine 16/16 with --demo-stall-cut in the trainer (flag-off identity holds).

### Gate 20 is TWO gates: the ledge catch, then a one-hull walkway over a catch sheet (11:37-12:00 UTC, local, no box; ANALYSIS of our own states)
* Credit $9.80 at 11:37 UTC (below the $10 reserve), fleet burn 0: no box can be rented. Offline bench instead:
  `tools/record_ckpt.py --spawn-states` with c10's ckpt_latest (31.18B, md5 6955707b...), greedy, CPU, from windows of
  the loop's own spine `pbr6b4_c9k02` (no keys at spawn, so not exactly the curriculum's spawns):
  * rows 5,700-5,740 (the branched beam's climb to the ledge, 64.65-64.89%): **8/9 CATCH the ledge** (on to 65.32-65.42%),
    1/9 falls short into the catch below (trigger_teleport *210, z 2,010-2,028: the greedy evals' 64.8% death);
  * rows 5,770-5,800 (on the landing platform, 64.92-65.06%): **9/9 die at 65.43-65.50%**, at (-3,331..-3,342,
    2,904-2,948, z 2,369);
  * rows 6,090-6,104 (the explorer chain walking north at 250 u/s, 65.81-65.85%): **6/6 die at 65.91-66.25%**, x -3,278..
    -3,320, z 2,369-2,383.
* The geometry (map entities and the field, measurement only): north of the landing platform the only floor is a strip one
  hull wide along a wall at x ~-3,256 (the explorer chain walks it at exactly x -3,256 from y 2,944 to 3,626), and just
  below the strip's level lies a fatal catch sheet (*186: x -3,577..-3,240, y 2,682..3,751, z 2,324-2,350) with a hole near
  its north end (the chain drops through at (-3,351, 3,513) to z ~2,000 and surfs east to 70.3%). The trigger is invisible
  to the lidar (the policy sees the floor at z ~2,000 through it). The field is defined in the air above the sheet (z
  2,389, y 3,150: 65.85-65.88% for every x from -3,256 to -3,420; the sheet's own cells unreachable) and descends north at
  ~1 u per u of y with a faint westward slope, so the policy strafe-jumps north-north-west off the strip and sinks onto
  the sheet: free flight through open space again (CLAUDE.md's gravity-dir note), here at walking speed.
* **The 65.391 eval's best episode (65.39%) is that death**: it caught the ledge and fell onto the sheet. A greedy that
  learns only the catch gains ~0.1; the next real gain (the hole, then the surf to 70.3%) needs the walk along the wall.
* **c10's stall at [6,095 / 6,100] (6.7-31%) was a death-dive at the line's margin**: k02c's line (d <= 19,291 = 66.244%)
  is where those dives end (66.22-66.25% from 65.81-65.85% spawns), so its "successes" were dives that crossed the line
  and died on the sheet. The walk itself was never paid apart from the dive.
* **What that means for `--demo-stall-cut`:** a cut inside the walkway puts the success line within a dive's reach (the
  bench's dives gain +0.1..+0.42% before they die), and the curriculum would walk back on dives. The generic guard is
  `--demo-success-alive SECS` (new, below): a crossing counts only if the episode survives it.

### `--demo-success-alive SECS`: a crossing counts only if the episode survives it (11:45-12:05 UTC, local, no box)
* `train_fast --demo-success-alive SECS` (default 0 = off; needs --demo-success-d and --demo-success-bonus, the per-tick
  reward path): a crossing of the spine's success line (with the energy under --demo-success-energy) counts for the
  curriculum's outcome, the bonus and --auto-spine's quorum only once its episode has stayed alive SECS past it, or it
  finishes; the bonus is paid then, not at the crossing. An episode that spawns past the line counts from its spawn
  (unpaid), so a band whose spawns die within SECS fails honestly. The quorum counts the eval episodes that stay alive
  SECS (rows = physics ticks) past their first crossing, or end at the finish. Generic: dying right after the line is not
  the skill on any map; it closes the death-dive door that a near line (a stall cut, a short spine) opens.
  (train_fast.alive_gate / crossed_and_alive; the state resets on every spine swap and stall cut.)
* Tests (CPU): tests/python/test_demo_success_alive.py 6/6 (147 s + 20 s) - the gate (a survived crossing is paid once, a
  dive that dies within the ticks fails, a spawn past the line succeeds unpaid), the quorum rule (survival, energy,
  finish), a princessburglar trainer run (notice, config key, the refusals), one with --auto-spine whose eval hook counts the survived crossings for the quorum, **flag off bit-identical to the trainer
  before the change** (config, every progress.csv number but time/*, final weights), TRAIN_ONLY. Regression: test_demo_stall_cut + test_auto_spine_energy + test_demo_success_energy + test_demo_success_d 15/15 (1 skip: the old --demo-success-d identity test finds no pre-change trainer in its 200-commit window). Untested at a gate.
* **To continue, revised (supersedes the c11 line of the block above):** c10's ckpt_latest (31.18B) on the FULL loop
  spine `pbr6b4_c9k02` + keys (its line 70.13% lies past the walkway's hole, so no dive off the strip can reach it), c9's
  flag set, the milestones and the loop exactly as in that block, plus `--demo-success-alive 1` (a guard; it changes
  nothing while the line is that far), and **without `--demo-stall-cut`**: on this spine a stall would come on the
  walkway (c9 sat at [6,702 / 6,707] near the hole, c10 at [6,095 / 6,100]), and a cut there - even with the survival
  rule - would only move the line onto the strip and stall again 5 rows back. What the policy has to learn there is to
  stay on the strip (east to the wall after the catch, then north along it to the hole), taught backward from the hole.

### The walkway is not in the policy's own sampling either (12:28-12:33 UTC, local, no box)
* The same bench with `--stochastic` (the training distribution: keys sampled, yaw sigma 0.017), 18 episodes per band:
  **0/18 survive from the strip band** (rows 6,090-6,104: every one dies at 65.92-66.26% within 36-104 ticks, x -3,278..
  -3,322) and **0/18 from the platform band** (rows 5,770-5,800: 65.03-65.45%, 22-112 ticks). The dive is the policy's
  only behaviour there; the curriculum's front temperature (keys / 1.5, yaw sigma x 6) lasts 40 ticks, and the strip
  has to be held for ~130-260 ticks (1-2 s) before the hole. So no backward curriculum on these weights can be expected
  to find the walk by sampling; the searches did (the branched beam took the platform to the wall, the reward-free GE
  chain walked the strip). This is CLAUDE.md 0c's case for macro exploration: a 25 Hz policy that cannot commit to
  "along the wall" for a second against the field's pull. (`--bc-file` would distil the chain, but it is refused with
  --keys-hold, which these weights carry.) c11 as planned is still the honest next run; its curriculum should be
  expected to sit on the walkway, and a band at ~0% for 2e8 steps there is the signal to stop it.

### Credit topped up ($39.78 at 13:32 UTC); a flaw in `--demo-success-alive`; c11 without it
* **`--demo-success-alive` must NOT be used on a spine whose end is doomed**, which the loop's GE chains usually are: a
  chain ends at its deepest cell, often mid-fall (pbr6b4_c9k02's row 7,325: z 1,872, vz -502, right above the catch
  sheet *184 at z 1,663-1,842). The first bands spawn past the success line (100 u before the end), and with the rule
  they must then survive SECS - they cannot, so the curriculum could never leave the spine's end. The CPU tests used a
  safe 40-state walk and could not show it. The right requirement is min(SECS, the spine's own ticks from its crossing
  of the line to its end) - derived from the spine, so a doomed end shrinks it - not yet built. Until then the flag is
  only for lines cut inside a survivable stretch (a --demo-stall-cut line), and c11 runs without it: its line (70.13%)
  lies past the walkway's hole, out of any dive's reach.
* Box 57820687: RTX 4080, 0.299 $/h, United Kingdom (machine 151788), rented 13:34 UTC for 3 h (credit $39.78, the fleet
  at 0.22 $/h before it). Deployed ar/pbr6b4 946a19e with c10's ckpt_latest (31.1815B, md5 6955707b...).
* **pbr6b4_c11** (launched 13:37 UTC, pid 3508): resume of c10's ckpt_latest (31.1815B) on the FULL loop spine
  `pbr6b4_c9k02` + keys (7,328 states, success line d <= 17,070 = 70.13%, past the walkway's hole), c9's flag set without
  tau0 or the energy rule, the 23 milestones (c10's 24 minus k02's own line), the loop as in the to-continue block
  (`--auto-spine --auto-spine-milestones --auto-spine-gain 500 --auto-spine-ge-minutes 5 --auto-spine-retry 4e8
  --auto-spine-patience 0 --auto-spine-energy 150`), no --demo-stall-cut, no --demo-success-alive; --steps 5e9 (stop at
  36.18B; it will be stopped before the 19:12 UTC wrap-up). Record gate passed (4 recordings in 16 s). A warm resume
  measuring AT the wall (gate 20: the catch and the walkway). The question: does the backward curriculum teach the
  walkway from the hole back, an increment at a time, where a cold policy never stays on it?
* **Fixed (13:45 UTC):** `--demo-success-alive SECS` now asks min(SECS, the spine's own ticks from its crossing of the
  line - or from the spawn row, for a spawn past it - to its end) (train_fast.alive_need / spine_cross_row; the quorum
  uses the spine's own ticks past its line). A spine ending in a fall asks no more survival than it flew itself; a spawn
  at its last rows succeeds as before; a dive that dies sooner after the line than the spine did still fails.
  test_demo_success_alive 7/7 (flag off still bit-identical). c11 keeps running without the flag.
* c11, first 18 min (13:37-13:55 UTC, 332k steps/s): the curriculum walked k02 back from its end 7,327 -> 6,717 (the hops
  before the walkway's hole, where c9 sat) in ~5 min at 100% / 71% bands, then through the hole region onto the strip:
  at 13:55 it oscillates [6,312 / 6,317] (66.42%, the chain's slow point on the strip at (-3,256, 3,480)), the newest
  states' success rising 21.4 -> 29.5% over six visits, the band behind them 54.7 -> 64.3%. So the walkway IS being
  taught backward from the hole, an increment at a time - what a cold policy (0/36 on the strip) never showed. Evals
  64.844 (31.182B, the resume) / 64.773 / 64.896 / 64.818 / 65.132 (31.484B, best 65.1%: one episode on the platform);
  0/9 greedy past the 70.13% line.

### GATE 20 PASSED by the greedy line: 71.270 (2026-10-26 14:19 UTC, c11)
* c11's curriculum walked the full loop spine back 7,327 -> 6,717 (5 min) -> 6,312 (the strip's slow point, 18 min) ->
  5,667 (the climb before the ledge, 42 min): the walkway, the platform and the catch were each learned backward from
  the hole, an increment at a time, at 30-64% bands (the newest states at 5-31%).
* Evals (9 greedy episodes from the true start): 64.844 (31.182B, the resume) / 64.773 / 64.896 / 64.818 / 65.132 /
  65.095 / 64.856 / 65.352 / 65.215 / 65.227 / **71.270 (31.937B)**: 8/9 catch the ledge (65.06-65.37%; 7 of them then
  die on the platform's north edge at z ~2,369, on the catch sheet), **1/9 walks the strip, drops through the hole and
  surfs on to 71.27%** (past k02's own end; it dies at (-1,113, 3,168, 1,723), above the next catch sheet *184), 1/9 dies
  early at 55.0%. The loop: "1/9 greedy episodes past the success line d <= 17,070; greedy frontier 19,887" (5/9 fires
  search 3).
* Hand-on snapshot ckpt_31782338560 (154.7M steps before the eval, md5 71d7ab34..., pulled); the eval's own checkpoint
  is between it and ckpt_31982616576. Box 57820687 extended to 19:28 UTC (the run is the first to clear gate 20).
* So the plan held without any cut: on the FULL spine (its line past the death-dive's reach), the backward curriculum
  taught a stretch that neither the greedy (0/18) nor the policy's own sampling (0/36) ever flew - the increments were
  small enough. The lesson of c10 (hand cut at 66.4%, line within a dive's reach, stuck 30 min) and c11 together: keep
  the success line where only the skill reaches it, and let the curriculum make the increments.
* c11 evals 31.937-32.465B: 71.270 / 71.266 / 71.325 / **71.381 (32.163B, the best)** / 71.290 / 71.158 / 71.189 / 71.132;
  past-the-line counts 1, 1, 4, 2, 3, 1, **5**/9. At 32.390B (median 70.45%) the quorum fired the loop's search (its 1st in
  c11): **the PLAIN BEAM on those weights to d 15,471 = 72.93% in 55 s** -> depth spine c11/auto/k01 (9,000 states, md5
  392147eb..., success line d <= 15,571 = 72.75%), the 70.13% line kept as a milestone; its curriculum walked back from
  8,999 at 100% bands, step 40 (8,244 by 14:45). That eval's episodes: 5/9 past the walkway, dying at 70.45-71.19% at
  (-979..-1,176, 3,215-3,675, z 1,718-1,880), onto the next catch sheet (*184, z 1,663-1,842) in the surf section after the
  hole (the next gate, ~71%); 2/9 on the walkway (65.0-66.0%), 2/9 short of the catch (64.7-64.9%).
* Hand-on for 71.381: ckpt_31982616576 (180.9M before the eval, md5 dca6920f..., pulled).
* 15:11 UTC: c11 on k01 (the loop's 72.93% beam line): evals 32.541-32.918B 71.288 / 71.357 / 71.127 / 71.268 / 71.192 /
  **71.499 (32.918B, median 71.4%: most greedy episodes now pass the walkway and stop at the ~71.5% gate)**; the
  curriculum sits at k01's [6,904 / 6,909] (at that gate, 19-43% bands, rising), 0/9 past its 72.75% line. Hand-on for
  71.499: ckpt_32783728640 (134.7M before the eval, md5 91badb50..., pulled).
* 15:36 UTC: c11 evals 32.994-33.371B 71.372 / 71.273 / 71.355 / 71.456 / 71.396 / 71.420 (medians 70.8-71.4%; the best
  stays 71.499), 0/9 past k01's 72.75% line. **The ~71.4% gate is a landing + a climb** (k01 = the loop's beam line, fetched
  from the box: auto/ via a base64 tar, md5 392147eb...): at 71.31% (row ~6,640) the beam lands on a small brush-entity
  platform (ground entity 87, z ~1,845) just above the catch sheet *184 (z 1,663-1,842), then climbs out with slow
  duck-jumps (z 1,845 -> 2,115, 30-300 u/s, ~18 s to row 9,000 = 72.93%) and walks south at y 3,200 -> 2,426. The greedy
  misses the platform and dies on the sheet at z ~1,720. The curriculum is teaching the climb backward (k01's
  [6,819 / 6,844], 78-82% bands, 6,904 -> 6,819 in 25 min); the landing at ~6,640 is next.

### c11's curriculum stopped on a bookkeeping bug, not on the skill; `--demo-match-band`; c12 (16:04-16:16 UTC)
* From 33.06B (~15:20 UTC) c11's curriculum never moved again: the band [6,819 / 6,823] showed "success 0.0% over 0 eps"
  for 50 min, the spine's per-decile episode counts froze, and `demohot` (envs in a band spawn's hot window) dropped from
  0.5-1.0% to exactly 0.0% at the move. Cause: k01's rows ~6,818-6,825 are the beam STANDING STILL at one origin
  (-1,075.2, 3,190.1, 1,898.7), and DemoCurriculum re-identifies spawns by rounded origin with a key that keeps only the
  LAST row of a repeated origin (6,825, outside the band) - every band spawn was credited outside the band, which never
  reached its 50-episode minimum. Any spine that stands still at a band boundary can freeze a curriculum this way.
* `train_fast --demo-match-band` (3654fc4; default off = byte-identical matches): a repeated origin is credited to its
  copy inside the frontier band when there is one, else as before. tests/python/test_demo_match_band.py 4/4 (the stuck
  case reproduced: a run of 8 copies past the band's end - 0 outcomes in the band off, 45.3 on), and the trainer's
  flag-off identity test still passes.
* c11 stopped 16:13 UTC at 34.105B (ckpt_latest md5 8b4619dc..., pulled; 39 evals, best 71.499). Box re-deployed on
  3654fc4 (the checkout is fetched, runs/ kept) and **pbr6b4_c12** launched 16:16 UTC (pid 12964): resume of that
  checkpoint on the loop's spine committed as `autoresearch/spines/pbr6b4_c11k01` (+keys) with `--demo-tau0 6819` (where
  c11's curriculum stood) and `--demo-match-band`, the 24 milestones (+ k02's 17,070), the loop flags as in c11; --steps
  5e9. Record gate passed.
* c12's first minute: the same band that sat at "0 eps" for 50 min - "tau <- earlier [6,814] (success 69.0% over 64
  eps)", then [6,804] 49.8%, [6,784] 55.0%, a backoff to [6,789]; demohot back at 0.4-0.5%. The fix is the whole story.
* 16:38 UTC: c12's curriculum walked k01 back 6,819 -> 6,459 in 22 min (the climb and the platform hops learned) and
  oscillates [6,459 / 6,464] (33-37% / the newest states 6-8%): the LANDING - the beam arrives at ~900 u/s (row 6,440,
  70.93%, h 2,520) and brakes onto the platform area by row 6,480 (71.25%, horizontal 104 u/s, h 1,935), a precision stop
  above the catch sheet. Evals 34.106-34.483B 71.489 / 71.436 / 71.389 / 71.351 / 71.453 / 71.401 (medians 71.2-71.3%,
  the greedy dies at that landing); the best stays 71.499.
* **17:05 UTC: 73.472** (c12, eval 34.785B): one episode lands on the platform, climbs out and reaches 73.47% (past k01's
  end; it dies on the catch sheet *184 at (-875, 2,040, z 1,719)); 6/9 still die at the landing (70.9-71.2%), 1/9 at the
  ledge catch (64.8%), 1/9 at 11.8%. Hand-on ckpt_34706292736 (79.2M before the eval, md5 79b56f8d..., pulled). c12's
  curriculum is past the landing band, at k01's [6,339 / 6,354] (the fast flight before it, 30-44% bands). Other c12 evals
  71.20-71.47 (35.012B: median 61.4%, an unsteady one).
* 17:34 UTC: c12 evals 35.087-35.540B 71.317 / 71.209 / 71.330 / **73.488** / 71.168 / **73.499 (35.465B, the best)** /
  71.274: in the two high ones one episode lands and climbs to 73.5% (the next stop, the same catch sheet further on);
  medians 71.1-71.2% (the landing still kills most). The curriculum oscillates [6,314 / 6,319] (31-34% / 14-19%), the
  fast flight before the landing. Hand-on for 73.499: ckpt_35307126784 (157.8M before, md5 f2781ad5..., pulled).
* 18:01 UTC: c12 evals 35.616-36.069B 73.475 / 73.532 / 73.580 / 73.556 / **73.590 (35.918B, the best)** / 71.284 / 71.282:
  five evals in a row with one episode past the landing to 73.5-73.6%, medians 71.2%. The curriculum's newest states at
  [6,314] rose 14-19% -> 24-28% (the advance threshold is 30%). Hand-on for 73.590: ckpt_35907960832 (10.0M before the
  eval, md5 160da9eb..., pulled).
* 18:28 UTC: **at 36.295B 6/9 greedy episodes past k01's 72.75% line** (median 73.44%; the landing is learned) - the loop's
  search 1 of c12 fired (on c12's ckpt_latest). Evals 36.069-36.295B 71.282 / 73.489 / 73.528 / 73.503; every episode
  past the landing stops at ~73.5% (the catch sheet *184 again, at (-875, 2,040)); the best stays 73.590. The curriculum
  had reached [6,304] (30.8%).
* 18:53 UTC: c12's search 1 (fired at 36.295B on 6/9) took 1,661 s - beam, branches and the GE ladder found no line
  deeper than the greedy's 73.5%, and the ENERGY stage returned a line keeping +253 u of energy height over the policy's
  (6,416 states, end d 16,352, taught with the energy rule h >= 2,100); its curriculum walked back 6,415 -> 4,630 in ~10
  min, and at 36.446B 7/9 greedy episodes crossed its line with the energy -> search 2 started (running at the
  wrap-up). Evals 36.295-36.446B 73.503 / 71.475 (median 61.5%) / 73.518; the best stays 73.590.

### pbr6b4 round 6 lesson (wrap-up, 2026-10-26 19:15 UTC) - supersedes the DRAFT above
* **Score 62.369 -> 73.590** (pbr6b4_c12, eval 35.917922304B; best episode 73.59%; hand-on ckpt_35907960832, 10.0M before
  the eval, md5 160da9eb..., pulled). Three gates passed by the greedy line from the true start this round: gate 19 (the
  62.3% drop: 65.391, c9, 08:59 UTC), gate 20 (the 64.9% ledge catch + the walkway over a catch sheet: 71.270 -> 71.499,
  c11, 14:19-15:11 UTC) and the 71.3% landing + climb (73.472 -> 73.590, c12, 17:05-18:01 UTC); at 36.295B 6/9 and at
  36.446B 7/9 greedy episodes pass the landing (medians 73.4%), all stopping at ~73.5% (the same catch sheet further on).
  Box time ~7.4 h on two cards (RTX 4080S 08:05-09:47; RTX 4080 13:34-19:12; the credit sat at the $10 reserve 09:47-13:32).
* **What did it, in order:**
  1. **Gate 19: measure where the GREEDY loses energy on its own eval** (one corner at 59.9%: a floor landing and a ground
     jump cut by CS's bunny-hop cap), **cut the stalled depth spine before its stuck band and teach that stretch with the
     energy rule** (k02e, `--demo-success-energy 150`, newly allowed with `--auto-spine-energy`): the energetic quorum fired
     the loop's own search and its plain beam passed the drop unboosted.
  2. **Gate 20: the FULL loop spine, no cut** (c11): its success line lay past the walkway's hole, out of a death-dive's
     reach, and the backward curriculum taught the walkway from the hole back, then the platform, then the catch, in
     ~45 min - a stretch neither the greedy (0/18) nor the policy's own sampling (0/36) ever stayed on. c10's hand cut
     inside the walkway (a line within a dive's reach) had sat 30 min on fake successes.
  3. **The loop kept going by itself**: 5/9 past the walkway fired its search, whose plain beam found the line to 72.93%;
     6/9 past that line fired the next, whose energy stage found a +253 u line; 7/9 energetic crossings fired a third.
  4. **A curriculum bookkeeping bug** (c11 froze 50 min: a band inside a stood-still run of identical origins was never
     credited, because the origin key kept only the last copy) - fixed by `--demo-match-band`; the relaunch (c12) moved
     at once and learned the 71.3% precision landing (arrive at ~900 u/s, stop on a small platform above a catch sheet).
* **What did not:** a local unboosted GE before the gate-19 corner; c10's hand cut at 66.4% (stuck on the walkway's dive
  band); `--demo-success-alive` as first written (it would have deadlocked a spine ending mid-fall - capped since at the
  spine's own survival); one silent trainer death after a resume's first eval (the identical relaunch ran).
* **New, generic, default off and bit-identical (CPU-tested):** `--demo-stall-cut STEPS` (a770eda, never run at a gate),
  `--demo-success-alive SECS` (17b60a8 + 6bdb0ed, never run at a gate) and `--demo-match-band` (3654fc4, run in c12: the fix
  that unfroze the curriculum).
* **Generic rules learned:** (1) the gate is where the greedy loses energy, measured on its own eval; (2) when a depth
  curriculum stalls past where the greedy diverges, move the line back before that band - energy rule if no deeper than
  the greedy, depth rule if deeper; (3) keep the success line where only the skill reaches it (past a death-dive's
  reach) and let the curriculum make the increments - a stretch no cold or sampled policy ever flew was learned that way;
  (4) a curriculum that stops moving may be a bookkeeping failure, not a learning one: check that its band still records
  outcomes (`demo starts ... over N eps`, `demohot`) before reading a stall as difficulty.
* **Next (if the program continues):** resume c12's ckpt_latest (36.6B, md5 d802b3e8..., pulled) on its auto/k02 spine
  (search 2's result at the wrap-up: beam + GE, 6,852 states to d 14,673 = 74.33%, md5 0aced692...; in
  runs/pbr6b4/pbr6b4_c12/auto/k02, pulled) with the loop on and `--demo-match-band`; the next stop is the catch sheet at
  ~73.5% after the climb. Last evals 73.503 / 71.475 / 73.518 / 73.586 / 73.443 (medians 61.5-73.4%).

## pbr7b2 (round 7, slot 2) - 2026-10-26 20:26 UTC start

Box: 57867173, RTX 4080S, 0.41 $/h, Ontario CA (machine 141481), rented 20:33 UTC for 3 h. Incumbent: pbr6b4_c12
ckpt_35907960832 (md5 160da9eb...), eval 73.590 at 35.918B. Credit at the start: $34.25, fleet burn 0.

### c1: c12's loop, continued on its last spine (20:41 UTC)
* **pbr7b2_c1** = resume of c12's ckpt_latest (step 36,659,265,536, md5 d802b3e8...; round 6's last evals 73.44-73.59 with
  6-7/9 past the 71.3% landing) on the loop's own search-2 spine, committed as `autoresearch/spines/pbr6b4_c12k02` (+keys;
  6,852 rows, the policy's plain beam to row 6,387 + an x1.04 GE chain to 74.33%, success line d <= 14,773 = 74.15%),
  c12's flag set (the round-5 to-continue block + --demo-match-band, --auto-spine-patience 0, --auto-spine-ge-minutes 5,
  --auto-spine-energy 150), the 26 milestones of c12's auto/current.json, --steps 6e9. Record gate passed (4 recordings).
  A warm resume measuring AT the wall (the ~73.5% catch sheet after the climb) - the inherited loop, run as it was.
* c1 evals: **73.567** (36.660B = the resumed weights: 4/9 past the 71.3% landing, 5/9 die at it) / **73.671** (36.735B:
  7/9 past the landing at 73.34-73.67%, median 73.42% - the branch's first best over the incumbent's 73.590). Every
  passing episode climbs off the platform slowly (~20 s: 71.5% at ~54 s, 73.5% at 70-78 s) and dies on catch sheet *184
  at (-850..-1,130, 1,780-2,080, z ~1,720). Handoff candidate: c1's ckpt_latest 36.777B (md5 3aabb185..., pulled).
* Throughput: 218k steps/s measured over 60 s (EPYC 7K62 host, cgroup quota 30.72 CPUs; the trainer uses ~6.8 of them,
  GPU ~15% busy - ~23 CPUs of the quota idle beside it).
* The k02 spine vs the greedy (ANALYSIS of our own lines): the spine's GE chain (x1.04 root at row 6,387, 70.79%) flies
  WEST of the platform without landing on it - 72.2% at h ~2,520 (vh ~760), 73.35% at (-1,267, 1,922, 1,855) h 2,321,
  74.32% at (-907, 1,563, 2,067) h 2,258 at 52.5 s - where the greedy lands on the platform (h 2,593 -> 1,995) and
  reaches 73.5% ~20 s later with h ~1,930. The record (yardstick only) is at 74% with h ~2,707 at 61 s. c12's search-2
  beam line itself (its frontier, cut at 6,387 for the GE root) had stayed high at z ~2,120 over 72-73% and crawled
  (50-90 u/s) to 73.68% at 55.8 s.
* The curriculum: 6,851 -> 6,496 in ~4 min (bands 48-100%), then backed off to [6,531] and oscillates [6,526] <-> [6,531]
  (band [6,531,6,535] 25 -> 43%, band [6,526,6,530] 4 -> 9%): the chain clips something at rows 6,532-6,535 ((-1,081,
  2,761, 2,122): horizontal 792 -> 652 u/s, -124 u) and the spawns just before it rarely make the line.
* c1 evals 3-4: 73.466 (36.811B) / 73.601 (36.886B: 6/9 past the landing; 3 die at it, 1 at 45%); the curriculum
  passed the 72.25% impact band (~10 min) and walked to [6,436 / 6,441] (71.5%, 49 rows after the chain's x1.04 root
  at 6,387): band [6,441,...] 60-70%, band [6,436,...] 10-15%.

### `--auto-spine-async`: the trainer's search beside the training (20:55-21:05 UTC, local CPU; 45ff5c2)
* Why: on this box the trainer uses ~6.8 of its 30.72-CPU quota and the GPU ~15% (it is single-thread bound), and a
  blocking search held it for up to ~28 min per gate (pbr6b4_c12's search 1: 1,661 s; the explorer inherits the
  trainer's OMP_NUM_THREADS=15, half the quota). With the flag the search runs on a copy of ckpt_latest
  (auto/kNN/ckpt.pt; ckpt_latest is rewritten every minute and the search loads it once per beam), its log goes to
  auto/kNN/auto_spine.log, the trainer polls it every iteration and applies its outcome (swap or retry) at the first
  iteration after it ends; no second search while one runs; a run that ends kills it. Generic: no constant.
* Tests (CPU): tests/python/test_auto_spine_async.py 3/3 - TRAIN_ONLY in record_ckpt; a smoke run in which search 1
  starts, ends while the training goes on (step later than the start) and its outcome is applied, the copy and its
  path in the search's log, the refusal without --auto-spine (232 s); **flag off bit-identical** to the trainer
  before the change with a blocking search firing in both (config, every progress.csv number but time/*; 125 s).
* **pbr7b2_c2** (21:07 UTC): c1 stopped at 36.947B, its ckpt_latest (36.945B, md5 242f1a2a...) staged as runs_ckpt.pt on
  the box, the box re-deployed on 45ff5c2, launched with c1's flags + `--demo-tau0 6436` + `--auto-spine-async`. Record
  gate passed (4 recordings). ~4 min of training lost to the relaunch.
* c2 evals 73.509 / 73.567 / 73.584 / 73.644 (36.95-37.17B, medians 73.3-73.5%): steady, every passing episode on the
  73.5% catch sheet. The curriculum sits at k02's platform hop: rows 6,426-6,437 fall onto the platform's edge (ground
  entity 87 at (-1,042, 3,193, 1,896), 71.37%) and the chain JUMPS off it on the landing tick (vz 0 -> +440, horizontal
  931 -> 756 u/s, h 2,645 -> 2,509), where the greedy lands and stops (h 2,593 -> 1,995) and climbs out for ~20 s. Band
  [6,436,...] 40-42%, band [6,431,...] 21-26% (rate 0.3): a bunny-hop timing band.
* **Look-ahead GE 1** (21:09-21:24 UTC, beside the trainer, 12 threads nice 10, 0.62M ticks/s; analysis, never training):
  explore_phase1 with auto_spine's GE settings (64 u x energy 32, rising split, view-follow, 8 ms, exact chains) rooted
  at 32 rows of k02's chain end (rows 6,600-6,851, 72.2-74.32%) x1.0 / 1.04: 37,253 cells, 5.6M runs, deepest **74.42%**
  at (-860, 1,535, 2,090) - nothing past it. The chain's end is an energy dead end for the explorer: there it has h
  ~2,258-2,390 at 500-660 u/s, and the record (yardstick only) passes the same point (74.34%: (-928, 1,553, 2,158))
  at 950 u/s with h ~2,723, then loops over catch volume *158 (x -793..-514, y 1,135..1,699, z 1,670..1,984) at
  ~950-1,000 u/s and dives east to 76%. Our line is ~465 u of energy height short there; the greedy (platform line)
  ~770 u (h ~1,930 at 73.5%). The next gate is energy again - the loop's energy stage's kind.
* Look-ahead GE 2 (21:31 UTC, the same 32 roots x1.08 / 1.16 / 1.24, 15 min): how much boost passes 74.42%?
* **Look-ahead GE 2** (x1.08 / 1.16 / 1.24 on the same 32 roots, 15 min, 92,960 cells): per root group (root cell ->
  boost by velocity ratio) - x1.08: nothing past 74.45%; **x1.16: 74.61-75.23%** (spine rows 6,746-6,762, ~73.6%);
  **x1.24: 75.8-78.43%** (rows 6,730-6,770; the deepest (1,137, 1,482, 1,251) at 114 u/s, h 1,301 - a long descent
  east). So the way past 74.4% needs ~+100-150 u of energy height at ~73.6% (x1.16 at |v| ~670 = +96 u, x1.24 = +149 u)
  - beyond the loop's depth ladder (x1.0-1.12): its energy stage (+150 u) is the one that should find it.
* **c2 eval 37.474B: score 74.287** (best 74.29%, median 73.5%): ep2 makes k02's touch-and-go hop at 71.4% (h 2,277
  at 505 u/s at 71.5%, where the platform-line episodes have ~1,910-2,070 at 30-320 u/s), flies the chain at 840-870
  u/s over 72.5-73% and reaches 74.29% at 56.5 s (the platform line reaches 73.5% at 70-80 s), then falls into the
  catch at (-1,054, 1,514, 1,722). 8/9 others: the platform line (73.4-73.5%, one at 17.8%). Handoff: ckpt_latest
  37.5296B (55.6M after the eval, md5 2643d415..., > 2x nearer than snapshot ckpt_37345558528, md5 0acedfd7..., both
  pulled). The curriculum: through the hop band to [6,416 / 6,421] (bands 17-30%).
* Look-ahead GE 3 (21:51 UTC): roots = k02 rows 6,400-6,600 (after the x1.04 root, through the hop and the 72.25%
  impact) x1.0 / 1.04 with --select-energy 3, 15 min: an impact-free line that keeps the energy and goes past 74.42%?
* The x1.24 look-ahead chain to 78.43%, replayed exactly (ge_replay 18/18 links; analysis): from k02 row 6,746 x1.24
  ((-1,300, 1,679, 1,992), 73.53%, h 2,449) it rises NORTH-EAST to (-979, 1,682, 2,153) at 74.16% (vz +143), loops
  south-east over catch volume *158 ((-698, 1,528, 2,129) 74.7%, (-590, 1,229, 1,921) 75.15%), dives east to
  (-254, 1,209, 1,529) at 75.96% (vz -1,009), lands hard at ~76.3% (h 2,509 -> 1,938), runs east at ~820 u/s to 78.0%
  and crashes at ~78.2% (h 1,944 -> 1,316), crawling to 78.43%. The record's own route there (yardstick: 74.34% at
  (-928, 1,553, 2,158), 75.20% at (-573, 1,204, 1,866), 76.47% at (25, 1,231, 1,415)) is the same loop, flown with
  h ~2,720-2,740. So the gate is the climb north-east at 74.2% (to z ~2,150): it needs h >= ~2,400-2,450 at ~73.5%
  (k02's chain has 2,283, the platform line ~1,930).
* c2 later evals: 73.500 / 73.656 / 73.544 / **74.287** / 73.569 / 74.228 (37.625B, median 21.4%: unsteady) / 74.283 /
  73.592 (to 37.776B); 1-2/9 on k02's chain each time, the curriculum at the hop band [6,421 / 6,426] (27-35%).
* **Look-ahead GE 3** (21:51-22:06 UTC; k02 rows 6,400-6,600 x1.0 / 1.04, --select-energy 3, 15 min, 58,187 cells):
  deepest 74.35% (still nothing past 74.45%), but the archive keeps far more energy than k02's chain at matched
  progress: max h over 72.5-74.25% +220..+286 u (73.5%: 2,506 vs 2,286).
* `tools/energy_splice.py` (new, 58bb644): the loop's energy-stage pick (auto_spine.energy_cell: the DEEPEST cell with
  >= G u more than the line, on the route, not plunging, not slower) + ge_replay + splice at the root + canonical views +
  translated keys, on an archive searched by hand. On look-ahead 3 with G 150: cell 48,251 at 74.30%, h 2,407 (+153 u),
  an UNBOOSTED root at k02 row 6,468 (71.86%, after the hop), 345 states (k02 needs 375 ticks to the same progress) ->
  spine **`autoresearch/spines/pbr7b2_e1`** (6,813 rows, k02 rows 0-6,468 unchanged + the chain). The chain flies LOW
  along x ~-1,230..-1,340 at z 1,780-1,850 and ~1,000 u/s over 72.25-73.5% (h 2,430-2,440; k02's chain 2,330-2,520
  higher up at ~800 u/s), climbs to (-1,166, 1,514, 2,027) at 74%, and heads east-north-east at 741 u/s to 74.30% -
  the record's route through 72-74.2% (yardstick: 72% (-1,256, 2,730, 1,798), 73% (-1,249, 2,156, 1,809), 74.21%
  (-1,026, 1,503, 2,133)), found reward-free from the policy's own state, with ~300 u less energy than the record.
* **pbr7b2_c3** (22:15 UTC): c2 stopped at 37.81B (curriculum at [6,421 / 6,426], the hop band, in e1's shared prefix),
  its ckpt_latest (37.810B, md5 5f865cd9...) resumed on e1 + keys with `--demo-success-energy 75` (success = cross
  d <= 14,789 (74.13%) with h >= 2,332; the loop's own G/2 rule) and k02's line 14,773 added as the 27th milestone
  (crossing it still pays), the loop as in c2 (async, energy 150: its quorum counts energetic crossings). Why not
  wait for the loop: its quorum needed 5/9 greedy episodes on k02's chain (1-2/9 after 50 min) and then a ~30-min
  search whose energy stage would do this same pick with a 5-min explorer; the e1 prefix keeps what the curriculum
  had learned. First eval (c2's weights): 0/9 cross 74.13% with the energy (the chain-flying greedy has ~2,300 there).
* c3 (e1, energy rule) first 20 min: the curriculum walked e1's new chain back 6,812 -> 6,457 in ~2 min (bands 62-81%,
  step 40: the low route is easy from its own states) and is back on the hop band [6,422 / 6,427] (22-34%). Evals 73.517
  / 73.555 / 73.812 / 73.345 (38.037B: median 71.5%, 4/9 die at the landing again, 1/9 makes the hop and dies at 73.0%):
  the policy is between the platform line and the hop.
* **Look-ahead GE 4** (e1 rows 6,690-6,812 = 73.4-74.3% x1.0 / 1.04 / 1.08, 15 min): 74.35 / 74.35 / 74.38% - every chain
  ends at (-870, 1,500-1,550, z 2,047-2,080). The +153 u of e1 does not open it for the explorer.
* What the gate is (the record as a yardstick only): the way on is a long rising arc from a ramp LAUNCH at ~73.3-73.5%
  ((-1,246, 1,876, 1,854), vz +411 -> +623 at 1,160 u/s) to an apex z ~2,174 at (-854, 1,638) at 74.36%, over the next
  catch. e1 launches at the same place (vz +389-411) with |v| ~910 and peaks at z ~2,057 - ~100 u short; the x1.24 k02
  root (h 2,449) peaked at 2,153. In free flight horizontal speed cannot become height: the launch's energy sets the
  apex. e1's h at the launch ~2,360-2,436, the record's ~2,700-2,745: the ~300 u deficit is made upstream (66% walkway
  walked at 250 u/s: h 2,428 vs the record's 2,745; 68-71%: -200..-250; the hop -136).
* Look-ahead GE 5 (22:35 UTC): e1 rows 6,690-6,812 x1.12 / 1.2 / 1.28, 12 min - the boost the climb needs from e1.
* **Look-ahead GE 5** (e1 rows 6,690-6,812 x1.12 / 1.2 / 1.28, 12 min): 74.39 / 74.43 / **78.17%** (x1.28 only, roots at
  rows 6,721-6,737 = the launch). From e1's low route the climb needs ~+300 u at the launch (h ~2,680), more than
  k02's slightly higher line (x1.24 = h ~2,449 at row 6,746, heading east-south-east (580, -342, 527)): the launch
  state (place, direction, energy), not energy alone. Look-ahead 3's archive holds ~1,000 cells at 73.2-73.8% with
  h >= 2,400, some within 30 u of k02's launch point, all heading south / south-west - none passed.
* **Look-ahead GE 6** (k02 rows 6,280-6,400 = 69.4-70.9%, before the platform, x1.0 / 1.04, --select-energy 3, 8
  threads, 20 min): deepest 74.24%, but max h over 71.25-73.0% 2,550-2,734 vs k02's 2,332-2,585 (+150..+235 u; the
  explorer avoids the hop / the 72.25% impact), 73.5% 2,451 (+165). Unboosted energy now exists at the launch.
* c3 evals 74.247 (38.113B) / 74.269 / 73.474 / **74.352** (38.339B: ep7 crosses e1's line with the energy - h 2,381 at
  71.5% (the hop kept), 2,401 at 73.0% at 947 u/s, 2,400 at 74.3%; the quorum line reads "1/9 greedy episodes cross
  d <= 14,789 with energy height >= 2,332"). Handoff: c3's ckpt_latest 38.397B (md5 b30831e3..., 58M after the eval,
  > 2x nearer than ckpt_38210633728, both pulled). The curriculum at [6,347 / 6,352] (70.3%, before the hop).
* Look-ahead GE 7 (23:00 UTC): rooted at look-ahead 6's 64 most energetic non-plunging launch-zone states (72.8-73.55%,
  h 2,406-2,463, one per 48 u x 30 deg cell), x1.0, 15 min - "pick by energy, then hand over" (round 5's gate 17).
* **Look-ahead GE 7** (rooted at look-ahead 6's 64 most energetic non-plunging launch-zone states, h 2,406-2,463,
  x1.0, 15 min): deepest 74.31% - unboosted energy at that level, in the explorer's hands, does not climb.
* **The policy's own beam on c3's 38.629B weights** (beam_tas from the true start, 4,096 copies, d-ranked, the loop's
  settings, 151 s beside the trainer): frontier 74.36% (d 14,652). It flies e1's low route with MORE energy than e1
  (h 2,472 at 73.0%, 2,452 at 73.5%, 2,483 at 74.3%; e1 2,436 / 2,359 / 2,400), but leaves the launch ramp at 73.5%
  ((-1,281, 1,734, 1,932)) with vz 445 at 912 u/s and peaks at z ~2,061 - e1's apex. The record (yardstick) leaves the
  same spot (within 25 u) at vz 623 and 1,114 u/s (apex ~2,161): ~200 u/s more and a steeper launch. Two branched beams
  forked 0.5-0.9 s before the launch (held view offsets +-45 deg, 2,048 copies, the loop's settings): 74.36% both -
  no forked lineage beats the plain line.
* Where the speed goes (beam line vs record, |v|): 71.2% 1,000 vs ~1,056; 72.0% 819 vs 1,223; 72.5-73.0% ~1,010 vs
  ~1,230. From 71% the record surfs DOWNHILL along x ~-1,300 (z 2,065 -> 1,798 by 72%) and gains speed; every line of
  ours detours east to the platform (x ~-1,000), stays high (z ~2,090 at 72%), hops, and drops back west later.
* Look-ahead GE 8 (23:30 UTC): the beam line's own rows 6,610-6,790 (72.5-74.2%) x1.0 / 1.08 / 1.16 - the boost the
  policy's current best line needs at the launch (if x1.08, the loop's own ladder finds it at the quorum).
* **What stops every line at 74.4% (core hull traces, measurement only):** a block across the corridor (y ~1,500-1,700)
  with its west face at x ~-853 and its top at standing-hull origin z ~2,137-2,150 (x -853..-800); west of it the pit
  (a fall lands at z ~1,714 = the level where every greedy episode dies), east of it a floor at z 2,017. Every chain
  meets the face at z 2,047-2,080 and falls back. Clearing it needs an apex >= ~2,155 from the ~73.5% launch ramp:
  vz ~600 there (ours 445 at 912 u/s; the record, yardstick, 623 at 1,114 u/s).
* **Look-ahead GE 8** (the c3 beam line's own rows 6,610-6,790 x1.0 / 1.08 / 1.16): 74.37 / 74.38 / **78.18%** - from the
  policy's current line x1.16 passes (from e1 it took x1.28). The deepest x1.16 chain (ge_replay 21/21) roots at beam row
  6,645 (72.9%: the beam's own state at 1,008 -> 1,169 u/s, about the record's speed there) and ends at (1,004, 1,369,
  1,247) = 78.18%; stitched with the beam line -> d1 (7,275 rows, md5 828108c8...; on the box, not trained). The x1.16
  root asks ~+160 u/s (~+220 u of energy height) at 72.9%.
* The policy's own lines at 72.9%: plain beam h 2,468 / 1,007 u/s, two branched beams forked before the launch 2,429, two
  energy-ranked (`--score dh --score-energy 3`) branched beams forked at 70.0 / 70.4% 2,444-2,449 - all within ~40 u:
  its population does not carry the energy, and none passes (74.27-74.36%).
* **The fork (the record as a yardstick):** at 70.5% the record and the policy's beam line are at the SAME point
  ((-1,299, 3,633-3,635, 2,110-2,117)); the record heads more south (v (502, -874), 1,009 u/s) and within 0.12 s turns
  to south-south-west ((-255, -984) at 70.8%) onto a downhill surf along x ~-1,300 (z 2,093 -> 1,798 by 72%, 1,030 ->
  1,223 u/s, h ~2,750 held); the beam (858 u/s) keeps heading south-east ((578, -641) at 70.8%) to the platform at
  x ~-1,000, hops it (h 2,610 -> 2,497) and reaches 72% at 819 u/s. A speed-preserving air turn of ~45 deg at 70.5% is
  the difference.
* Look-ahead GE 9 (23:41 UTC): the beam's own rows 6,330-6,385 (70.3-70.8%) with the velocity turned 0 / -20 / -40 /
  -60 deg (clockwise, |v| kept: the recipe's front-perturb heading turn, here as search roots), 15 min.
* **Look-ahead GE 9** (turned roots at the 70.5% fork): 0 deg 73.70%, -20 deg 74.22%, **-40 / -60 deg die by 71.7-71.9%**:
  with the policy's ~850-900 u/s the record's westward line is not viable - the record carries ~1,009 u/s and h ~2,746
  there (ours 2,577): the deficit exists before the platform.
* **Where the energy deficit is made (c3's beam line vs the record, energy height h at first arrival, yardstick):**
  56-65.5% the policy is AT the record's energy (gap -90..+82 u; at 65.75% its h 2,773 is above the record's 2,733);
  65.75 -> 65.90% it LANDS on gate 20's one-hull walkway strip at x -3,256 (h 2,773 -> 2,610 in 20 ticks, 794 -> 587 u/s),
  walks it (540-600 u/s), drops through the hole at ~880 u/s and reaches 67% with h ~2,516 (gap -246); the gap holds at
  -160..-200 to 71% and the platform hop adds ~-70..-110. At 65.6% the two are in nearly the SAME state ((-3,328..-3,348,
  3,008-3,013, 2,420-2,442), vz -161..-205); in the next 0.15 s the record contacts something that throws it up and WEST
  (vz +193 at (-3,441, 3,122, 2,426)), flies over the catch sheet *186 at z ~2,390-2,450 and ~690-775 u/s to the hole and
  drops through at 983-1,111 u/s with h ~2,760 held. So the 74.4% block is paid for 8% upstream, at the walkway that
  round 6's curriculum taught the policy to WALK (the only behaviour it ever found there).
* Look-ahead GE 10 (00:00 UTC): the beam's rows 5,680-5,775 (64.9-65.7%, before the strip landing) x1.0 / 1.04,
  --select-energy 3, 15 min: a line over the sheet that keeps the energy (the gate-17 / gate-19 recipe, one corner).
* c3 later evals 74.309 / 74.321 / 74.286 / 74.315 / 74.334 / 74.275 (39.09-39.47B; at 39.547B "3/9 past e1's line"):
  steady at 1-3/9 on e1's route, all stopped by the 74.4% block. Stopped at 39.547B (ckpt_latest 39.543B, md5 75891496...).
* **Look-ahead GE 10** (the c3 beam's rows 5,680-5,775 x1.0 / 1.04, --select-energy 3, 15 min, 14 threads, 10.1M runs):
  the explorer gets OVER the catch sheet west of the strip (x ~-3,430..-3,561, z ~2,390-2,445, the record's side) keeping
  +142 u over the beam line at 66.0% and +211 u at 66.5% (h 2,736); through the hole and the surf after it the gain falls
  to +71..+94 (67-68%) and below the beam past 68.5% (the explorer's clumsy drop, not the policy's).
* **Energy spine `autoresearch/spines/pbr7b2_e3`** (tools/energy_splice.py, G 150 = the loop's own): the policy's own beam
  (c3 38.629B) to row 5,711 (65.18%) + a 173-state chain from that state x1.04 to cell 60,145 at 66.59%, h 2,684 (+153 u
  over the beam line there; 172 ticks against the beam's 280 to the same progress), airborne all the way at 690-775 u/s.
* **pbr7b2_c4** (00:21 UTC): c3's ckpt_latest (39.543B) on e3 + keys with `--demo-success-energy 75` (cross d <= 19,194 =
  66.41% with h >= 2,609; the beam line has ~2,574 there) + the 27 milestones + e1's line 14,789 (crossing 74.13% keeps
  paying), the loop as in c3. First eval: "0/9 cross with the energy (9 cross it)". The plan: the walkway's energy (+150 u
  at 66.5%) carried to the 74.4% block, whose launch lacks ~+180-220 u (x1.16 passes from the policy's line); then the
  loop's own search (its energetic quorum on e3) on the more energetic weights.
* **Look-ahead GE 11** (e3 rows 5,790-5,883 x1.0 / 1.04, --select-energy 3, 15 min): +242 u over the beam at 66.5% (h 2,767),
  +118 at 67.0%, +45 at 67.5%, +24 at 68-69% - the explorer's own hole drop sheds it; whether the policy keeps it is
  what c4 measures.
* c4 evals 73.527 / 73.398 / 74.247 / 74.254 / 74.271 (39.54-39.85B): "0/9 cross d <= 19,194 with h >= 2,609 (7-8 cross
  it)". The GREEDY's walkway (39.846B eval): 65.5-66.4% at 232-291 u/s with h ~2,423-2,442 - it WALKS the strip at the
  bunny-hop-capped ground speed (~250 u/s), ~320 u under the record and ~150 u under its own best stochastic copy (the
  beam flew 65.0-66.35% at 540-794 u/s, h 2,574-2,773); through the hole it gains back to ~2,500 at 67-68%. The
  curriculum walked e3 back past its x1.04 root (5,711) and the wall slide (66.2%, ~0.3% for a few minutes) to the beam's
  ledge approach [5,613 / 5,618] (bands ~50% / 7%) in ~20 min.
* c4 evals to 40.148B: 73.396 / 74.281 / 74.308 / 74.273, but per-episode the route ERODES - some evals have 4-5 of 9 die at
  5.9-9.3 / 27 / 46 / 56.5-62.5% (c3's evals were nearly all at 73-74%): 99% of the curriculum's episodes sit in e3's last
  decile (61-66.6%) and nothing in the spine covers 67-74% any more. Probe (record_ckpt, c4's 40.246B weights, greedy,
  spawned at e3's last rows, h 2,684): 3/4 crash in the hole drop at ~67.6% (h 2,226-2,251 at 67%), 1/4 falls back to the
  platform line to 73.62% (h 2,102 at 72%) - from e3's end the greedy does not yet know the hole.
* **But the policy's POPULATION learned e3** (beam_tas from the true start on c4's 40.246B weights, the loop's settings):
  frontier 74.33%; it flies west over the sheet (x -3,539..-3,561 at 65.5-66%) and carries +83..+136 u over c3's beam line
  through the hole and the surf - h 2,694 at 66% / 2,652 at 67% (954 u/s) / 2,654 at 70% / **2,704 at 71.2% (1,066 u/s)** -
  then the PLATFORM HOP takes it (2,586 at 72%, 2,517 at 72.9%, 2,486 at 73.5%: +34 at the launch). One corner learned,
  the next one (71.2-72.9%, -190 u) exposed - round 5's corridor picture ("one energy spine per corner").
* Look-ahead GE 12 (01:12 UTC): that beam line's rows 6,260-6,343 (70.3-71.35%, before the hop) x1.0 / 1.04,
  --select-energy 3, 15 min: a hop-free line that keeps the energy to the launch - and over the block?
* **The loop's own energetic quorum fired on e3** (c4): energetic crossings 0/9 for 50 min, then 4/9 (40.374B) and **9/9**
  (40.450B: "9/9 greedy episodes cross d <= 19,194 with energy height >= 2,609") - the GREEDY learned the walkway corner in
  ~900M steps (~60 min). Search 1 ran BESIDE the training (`--auto-spine-async`, first live use: launched at 40.450B on a
  copy of ckpt_latest, ended at 40.519B after 298 s, the trainer at ~256k steps/s throughout) and swapped in the plain beam
  as a depth spine k01 (6,600 states to 74.29%, success line 74.11%; saved in runs/pbr7b2/pbr7b2_c4/auto_k01/).
* **Then c4 collapsed**: evals 61.505 (40.525B) / 62.261 (40.601B), medians 61.2-61.5% - every episode dies at gate 18 (the
  61.5% gap hop), a skill held for two rounds. One update at 40.468B logged **kl 0.92** (the run's usual 0.05-0.3); episode
  length fell ~3,000 -> ~1,800, reward ~300 -> ~210. The eval right before (40.450B: 74.267, 9/9 energetic crossings) was
  healthy and the swap came only 6M steps before the first collapsed eval. Cause not established: the search ran in its own
  process (training numerics untouched; GPU / CPU only shared), and a kl spike also followed the first wave of success
  bonuses (9/9 energetic crossings -> many +50 payments at once). Watch the next async search for a repeat.
* **Look-ahead GE 12** (the e3-trained beam's rows 6,260-6,343 = 70.3-71.35%, x1.0 / 1.04, --select-energy 3): the explorer
  keeps +146..+251 u over that beam through 71.5-73.0% (e.g. h 2,764 at 1,091 u/s at 72.75%, on the west side, no hop) and
  +129..+178 at 73.25-74.0%; deepest 74.38% (its energetic cells near 73.5% lie east of the launch ramp).
* **Energy spine `autoresearch/spines/pbr7b2_e5`** (energy_splice G 150): that beam (c4 40.246B) to row 6,320 (71.08%) + a
  106-state chain from it x1.04, airborne past the platform (no hop), to cell 57,109 at 72.32%, h 2,735 (+217 u over the
  beam there, 105 ticks against 197), ~1,030-1,080 u/s.
* **pbr7b2_c5** (01:33 UTC): resumed from the search's own copy of c4's weights at 40.4499B (auto/k01/ckpt.pt, md5
  20f467f8...: the 9/9-energetic, pre-collapse weights) on e5 + keys, `--demo-success-energy 75` (cross d <= 15,920 = 72.14%
  with h >= 2,660), 30 milestones (+ e3's 19,194 and k01's 14,794), the loop as in c4. First eval: 0/9 with the energy, 8/9
  cross the line.
* **Look-ahead GE 13** (e5 rows 6,370-6,425 = 71.9-72.32% x1.0 / 1.04, 15 min): the explorer reaches the RECORD's energy at
  the launch zone (h 2,804-2,827 over 72.5-73.75%) and still stops at 74.37% - its energetic cells drift EAST of the launch
  ramp line (x -1,122..-1,214 at 73.3-73.5%). How the x1.16 chain (d1) clears the block: it reaches the ramp line (x
  ~-1,280..-1,320, y 2,060-1,700) at 1,165 u/s (c3's line 1,007), rides it to vz 560 at z 1,957 (73.41%), peaks at z ~2,153
  at 74.36% and crosses the block's face (x -861) DUCKED at z 2,151 (a ducked hull clears ~18 u lower than a standing one).
  So energy is necessary, not sufficient: the launch must start on the ramp line at ~1,160+ u/s. e5 ends ON that line high
  (z 2,074, h 2,735): dropping to the ramp would give ~1,190 u/s.
* c5 evals: 74.262 / 74.242 / **74.370** / **74.374** (40.677B, the branch's best by a hair) / 74.346. The GREEDY now flies e3's
  walkway (h 2,630-2,791 at 66% where c3 walked at ~2,425; ~2,600-2,670 at 71.2%), still lands on the platform at 72% (8/9);
  0/9 cross e5's line with h >= 2,660 yet. Handoff for 74.374: ckpt_40650145792 (26.7M before the eval, md5 4ee0add1...,
  pulled; ckpt_latest 40.790B pulled too).
* **Probe from e5's end** (record_ckpt, c5's 40.807B weights, greedy, 6 episodes spawned at e5's last 26 rows): the policy
  CARRIES the energy into the launch zone - h 2,743-2,779 at 72.5-73.3% at 1,038-1,180 u/s (the x1.16 chain's speed) -
  but reaches 73.3% at x -1,161..-1,240, EAST of the ramp line (-1,280..-1,320), loses ~400 u there (h -> 2,196-2,393 at
  73.5%) instead of launching, and dies at 74.19-74.27%. The energy is now reachable; the missing skill is entering the
  launch ramp on its line at that speed - after e5's end, so taught only by the episodes that continue past it.
* Look-ahead GE 14 (01:57 UTC): e5 rows 6,385-6,425 x1.0 / 1.08 / 1.16 - the boost the explorer needs from there.
* **Look-ahead GE 14** (e5's end, rows 6,385-6,425, x1.0 / 1.08 / 1.16): 74.21 / 74.30 / 74.40% - even x1.16 from e5's end
  does not clear the block: e5 passes the platform HIGH (z ~2,070 at 72.3%) and comes into the launch ramp from the wrong
  place; the boosted chains that pass start LOW on the ramp line (z ~1,840 at 72.9%), the record's west downhill line.
* **Look-ahead GE 15** (the e3-trained beam's own rows 6,252-6,307 = 70.2-70.9%, h 2,663-2,693, |v| 958-1,013, velocity
  turned 0 / -20 / -35 / -50 deg): every turn now survives (with c3's energy -40 / -60 died by 71.9%) but none clears the
  block (74.43 / 74.00 / 74.15 / 74.27%) and none keeps the record's energy on the west line (2,420-2,520 at 72.9-73.5%).
* c5's population (beam on its 41.2B weights): h 2,629 at 66% (over the sheet), 2,696 at 71.2% (1,062 u/s), then WEST of the
  platform at 72.0% (x -1,323: e5's skip) but at 2,542, 2,469 at 72.5%, 2,475 / 1,019 u/s at 72.9%; frontier 74.31%.
  **Look-ahead GE 16** from its ramp approach (72.4-73.45%) x1.08 / x1.12: 74.38 / 74.43% - no pass; the block needs ~1,160+
  u/s on the ramp line (x1.16 of c3's 1,007 passed), ~+15% over the population's ~1,020. The next loss: 71.2 -> 72.0%
  -154 u where the record (yardstick) surfs DOWNHILL (z 2,027 -> 1,798) gaining 1,090 -> 1,223 u/s and losing ~38 u.
* c5 evals 74.215 / 74.291 / 74.318 / 74.329 / ... (to 41.58B, medians 73.1-73.2%): 0/9 cross e5's line with h >= 2,660
  in 2 h.
* Look-ahead GE 17 (02:48 UTC): c5's population line rows 6,263-6,323 (70.6-71.35%, before that descent) x1.0 / 1.04,
  --select-energy 3, 15 min; the energy pick to be restricted to the ramp-entry window (72.6-73.0%).
* **Look-ahead GE 17** (c5's population line rows 6,263-6,323 = 70.6-71.35%, x1.0 / 1.04, --select-energy 3): +108..+211 u over
  that line over 71.5-73.25% (72.5%: h 2,680 at 1,063 u/s at (-1,283, 2,453, 1,975), ON the launch ramp's approach line);
  near the ramp entry (72.9%) its fastest state is 1,062 u/s; deepest 74.32%.
* c5's last evals 74.279 / **74.374** (41.734B, median 74.3%: most episodes now reach the block) / 74.358 / 74.318; its curriculum
  sat 90 min at e5's band [6,345 / 6,350] (20-25% / 85%), 0/9 energetic crossings of e5's line (h >= 2,660 at 72.14%).
* **Energy spine `autoresearch/spines/pbr7b2_e6`** (energy_splice G 150, window 72.2-72.8% = the ramp's approach): c5's
  population line (beam on ~41.2B) to row 6,294 (71.01%) + a 137-state chain from it x1.04, airborne over the platform and
  back onto the ramp line, to cell 86,087 at 72.47% ((-1,255, 2,521, 1,993) heading south at 1,045 u/s, h 2,675 = +205 u). A
  clean drop from there onto the ramp at 72.9% gives ~1,158 u/s - the speed the passing x1.16 chains had.
* **pbr7b2_c6** (03:09 UTC): c5's ckpt_latest (41.936B, md5 30fea41a...) on e6 + keys, `--demo-success-energy 75` (cross
  d <= 15,831 = 72.30% with h >= 2,600), 31 milestones (+ e5's 15,920), the loop as before. First eval: 0/9 with the energy,
  9/9 cross the line.
* **Probe from e6's end** (record_ckpt, c6's starting weights 41.936B, greedy, 6 episodes at e6's last 21 rows): the speed is
  there - 72.9% at 1,137-1,144 u/s, h ~2,697 - but all 6 die at 73.04-73.06% within ~0.7 s: they descend at vz -128 -> -570
  and hit the launch ramp's FRONT EDGE at y ~2,100, z ~1,785 (vy -> 0, then a slide down the face to the death level), ~50 u
  under its top; the x1.16 chain meets the ramp LEVEL (z ~1,837, vz ~0 at y ~2,060) and rides it up to vz 560. e6 ends ~160 u
  too high for a clean entry. Look-ahead GE 18 (03:11 UTC): e6 rows 6,380-6,430 x1.0 / 1.04 - a clean ramp entry from there?
* **Look-ahead GE 18** (e6's end, rows 6,380-6,430 x1.0 / 1.04): deepest 74.34%; its fastest state RISING on the ramp does 955
  u/s. Across look-aheads 6 / 12 / 17 / 18 the low on-line states at the ramp approach (72.3-72.95%, z 1,800-1,900, x within 70
  u of -1,280, heading south) top out at **~1,013-1,025 u/s** - the same ceiling as the policy's best greedy episodes (960-1,056
  u/s at 72.9% in c5's 41.73-41.81B evals); the record (yardstick) does ~1,223 there. The greedy's 71.2% energy is now the
  record's (h 2,673-2,762 at 1,068-1,126 u/s) and is lost by 72.5% (-150..-290 u): the record falls 312 u along x ~-1,300 from
  the 70.5% fork with no contact loss (|v| 1,009 -> 1,223, h held); every line of ours first heads toward the platform
  (x ~-1,000 at 71.2%) and comes back west. Kept to 72.9% at z ~1,850 the greedy's 71.2% energy would be ~1,187 u/s - enough.
* **New best 74.399** (c6's first eval, 41.936B = c5's final weights, median 74.3%: most episodes now reach the block); c5 also
  had 74.396 at 41.205B. Handoff: c5's ckpt_latest (41.9357B, md5 30fea41a...) placed as pbr7b2_c6/ckpt_41935699968.pt (the
  eval's exact weights). c6 evals after it 74.341 / 74.340 / 74.329 / 74.337; 1/9 energetic crossing of e6's line at 42.163B.
* **Energy-ranked branched beams on c6's ~42.2B weights** (beam_tas --score dh --score-energy 3, half the copies forked at the
  population line's 70.25% / 70.55% ticks with held view offsets +-45 deg): both give the same frontier line, the most
  energetic this round - h 2,791 at 70.5% (1,078 u/s), **2,818 at 71.2% (1,163 u/s)**, 2,706 at 72.0% (east of the platform,
  x -877), 2,691 at 72.5% (1,113 u/s), **2,637 at 72.9% at 1,124 u/s on the ramp line (x -1,295)**, frontier 74.26%: within ~4%
  of the x1.16 chain's 1,165 u/s at the ramp, and still not over the block. Saved as runs/pbr7b2_r5/b4.npy + keys on the box.
* Look-ahead GE 19 (03:40 UTC): that line's ramp approach (rows 6,307-6,393 = 72.4-73.45%) x1.0 / 1.04 / 1.08, 12 min - is x1.04
  (the mildest boost, learned without the boost by round 3's greedy at gate 8) now enough?
* **Look-ahead GE 19** (the energetic dh line's ramp approach, 72.4-73.45%, x1.0 / 1.04 / 1.08): 74.35 / 74.34 / 74.44% - even
  x1.08 (~1,214 u/s at the ramp) does not clear the block from it. Speed is not the whole threshold: the x1.16 chain that passes
  enters the ramp heading straight SOUTH (v (1, -1,165, 44)) along x ~-1,275, level at z ~1,833; this line swings back west from
  the platform side (x -877 at 72.0%) and enters at an angle. c6's greedy: 2/9 then **4/9** energetic crossings of e6's line
  (42.39 / 42.46B), then 1/9 - its quorum is close.

### pbr7b2 round 7 lesson (DRAFT, 03:56 UTC; finalised at wrap-up)
* **Score 73.590 -> 74.399** (c6 41.936B = c5's final weights, median 74.3%). The 71.3% landing and the platform climb fell to the
  inherited loop (c1/c2: k02's touch-and-go hop, 73.67 -> 74.29); everything since stops at one wall at 74.4%.
* **The 74.4% wall, fully characterised** (hull traces, look-aheads 1-19, the record as a yardstick): a block across the corridor
  (top z ~2,150, death pit before it) that is cleared only by a launch-ramp entry at ~73.0% heading straight south along x ~-1,275,
  level at z ~1,835, at ~1,160+ u/s, then a DUCKED crossing (x1.16 of the policy's line passes to 78.2%; x1.0-1.12 and every
  energetic explorer state at the record's energy, coming in from another direction or height, fail). The record does it with
  energy kept from the 70.5% fork (a 312 u contact-free fall along x ~-1,300 to 1,223 u/s); our lines detour toward the 71.4%
  platform and come back west, and both the policy and the uniform explorer top out at ~1,020-1,060 u/s on the low ramp approach.
* **Where the energy goes, measured** (`autoresearch/analysis/pb_energy_profile.py`): the policy matches the record's energy from
  56% to 65.75%; it lost ~250 u at gate 20's walkway (landing on the one-hull strip and walking it at the bunny-hop-capped 250 u/s)
  - **fixed this round** by energy spine e3 (an over-the-sheet line found by a reward-free search from the policy's own states,
  picked and taught with the loop's own energy rule; the greedy learned it 9/9 in ~60 min); then -150..-290 u over 71.2-72.5%
  (the platform detour) - e5 / e6 taught part of it (the greedy now carries the record's energy at 71.2%, h ~2,700-2,760).
* **New, generic, default off and bit-identical:** `--auto-spine-async` (the loop's search beside the training; its first live
  search ran 298 s with the trainer at full speed - a blocking search held it up to ~28 min); `tools/energy_splice.py` (the loop's
  energy-stage pick + splice on an archive searched by hand, so an energy spine can be built without waiting for the quorum);
  `stitch_spine.py` spine segments.
* **What did not work / cost time:** e1 (the low route: energy after the launch, not before it); e5 (passes the platform high and
  meets the ramp from the wrong place); turned roots at the fork (survive, keep no energy); a kl 0.92 update right after the
  first energetic quorum collapsed c4 to 61.5% (restarted from the search's own copy of the pre-collapse weights).
* **Look-ahead GE 20: THE BLOCK FALLS TO THE APPROACH DIRECTION, not to more energy.** Roots = the energetic dh line's own states
  at 72.55-73.05% (rows 6,322-6,358, heading SSE at ~1,1xx u/s, vz -228..-462) with the horizontal velocity TURNED to head
  straight south, speed kept, x1.0 and x1.04: one x1.0 root (row 6,356, 73.0%) goes **over the block to 77.95%** (x1.04: 74.45%
  - a precise, chaotic entry). The energy the policy now carries is enough; what it lacks is a ~20 deg heading change toward
  straight south in the last ~0.05-0.3 s before the launch ramp (an air turn of ~3 deg/tick at 1,100 u/s - in its reach).
* **Depth spine `autoresearch/spines/pbr7b2_d2`** (stitch_spine: the dh beam line to row 6,356 + that chain, 8,520 rows to 77.95%;
  ge_replay 25/25 exact): it rides the ramp (vz +183 -> +529 over 73.24-73.44%), crosses the block DUCKED at z ~2,086-2,132, lands
  on the floor beyond at ~74.4% and then crawls (explorer-style, ~0-260 u/s) for ~8 s to 75.7%, drops, and crashes at 77.6%.
  `pbr7b2_d2c` = d2 cut at row 7,010 (74.80%, on the floor past the block; success line d <= 14,499 = 74.63%, out of reach of any
  line that does not clear the block).
* **pbr7b2_c7** (04:14 UTC): c6's ckpt_latest (42.928B, md5 eced6274...) on d2c + keys, the depth rule, 32 milestones (+ e6's
  15,831), the loop as before. First eval: 0/9 past 74.63%, greedy median 14,707 (74.27%).
* (A process note: `pgrep -f auto_spine.py` inside a `box.py ssh` command matches its own shell - CLAUDE.md's pkill warning - and
  read as a leftover search; `ps ... | grep -v grep` showed none. And Python's atexit does not run on SIGTERM, so a trainer killed
  that way while an async search runs would orphan the search - kill it by hand, or SIGINT the trainer.)
* **c7 collapsed too, and this time the cause is visible.** Its curriculum walked d2c back at 100% bands and got PAST the turned
  root (row 6,356) within ~10 min - from its own SSE heading the policy learned the turn and the block crossing (bands at
  71.9% 35-66% by 04:28) - but the greedy evals went 74.304 -> **56.364 / 56.232 / 55.637** (medians ~55.5%: every episode dies
  at the 56% shaft). From the first iteration the value loss was 275 and the episode reward 480, and the updates ran at kl
  0.4-1.35 with one at **kl 4.5** (43.003B) just before the first collapsed eval. Spawns on d2c's chain cross a PILE of bonuses
  within a second: the success line (74.63%) and three milestones within 21 u of each other (k01 14,794 / e1 14,789 / k02
  14,773 at 74.11-74.15%), + e5 / e6's two lines 89 u apart - ~200 u of reward the value head had never seen, huge advantages,
  destructive updates. c4's collapse (kl 0.92) also followed a sudden wave of bonuses (its first 9/9 energetic quorum).
* **pbr7b2_c8** (04:31 UTC): c6's pre-collapse ckpt_latest (42.928B, md5 eced6274...) on d2c again with the milestones
  DE-CLUSTERED (one line at 74.15%, one of e5 / e6's; 29 in all) and `--lr 1e-4` (was 3e-4; the sidistic lineage's late lr cuts
  held a fragile skill the same way). Generic: milestones closer than the bonus they each pay are one line; the lr is a
  stability knob, not a map constant.

### THE 74.4% BLOCK FALLS: 74.399 -> 77.042 (2026-10-27 05:20 UTC, c8)
* c8 (d2c, lr 1e-4, 29 de-clustered milestones): stable (update kl median 0.12), evals 74.30-74.36 for ~55 min while the
  curriculum worked the block crossing and the launch band (rows 6,400-6,404 at 0.3-0.4% for ~15 min), then **75.422** (43.533B,
  1/9 over the block) and **76.864** (43.608B: 7/9 over it, median 75.8%) - at 74.4% the episodes pass (-823..-846, 1,630-1,653) at
  z 2,125-2,138, over the block's top. 7/9 past d2c's line fired the loop's async search 1 (158 s beside the training): the PLAIN
  beam on those weights to d 12,997 = 77.26% -> depth spine k01 (6,773 states, success line 77.08%). Next eval **77.042**
  (43.684B, 8/9 over the block at 75.2-77.04%, median 76.32%): every passing episode dies at z ~1,261-1,265 between x -335 and
  432 (75.8-77.0%) - the next catch level, gate 22.
* Handoff: c8's ckpt_latest at 43,706,744,832 (23M after the 77.042 eval, > 2x nearer than ckpt_43608178688; md5 recorded by
  box.py), pulled and pinned as pbr7b2_c8/ckpt_43706744832.pt; ckpt_43608178688.pt (the exact weights of the 76.864 eval, the
  search's own copy) and ckpt_43529011200.pt pulled too.
* What did it, in order: (1) energy at the right place - gate 20's walkway (e3) and the 71-72.5% descent (e5/e6), each found by
  a reward-free search from the policy's own states and picked by the loop's energy rule; (2) the APPROACH, not more energy -
  look-ahead 20's heading-turned root (the recipe's front-perturb turn as a search root, no speed change) found the only way
  over; (3) stability - c7 learned the same line in ~10 min at lr 3e-4 and collapsed (stacked milestone bonuses -> kl 4.5);
  c8 at lr 1e-4 with one line per crossing took ~55 min and held; (4) the loop then ran itself (quorum 7/9 -> async search ->
  the policy's own beam line as the next spine).
* c8 after the block: evals 77.469 / 77.688 / 77.638 / **78.219** (43.986B, median 77.6%) - the loop runs itself: search 2 fired at
  43.835B (7/9 past k01's 77.08% line), ran 192 s beside the training and swapped in the BRANCHED beam (7,100 states to 78.30%,
  success line 78.13%); 1/9 past it at 43.986B. Handoff for 78.219: ckpt_43929567232 (56M before, md5 75e0993a..., pulled);
  c8's ckpt_latest 44.061B pulled too. Update kl median 0.13 (spikes to 1.5 still occur; no collapse).
* **78.619** (44.137B) and **78.858** (44.212B, median 78.46%): the curriculum got through k02's ramp kick (rows 6,944-6,954, where
  the spine lands on the ramp at x ~0-50, z ~1,412 and vz goes -189 -> +297; bands there at 13-41% for ~40 min) and walked back
  to row 6,229 within minutes; 7/9 past k02's line fired search 3 at 44.212B. Handoff for 78.858: the search's own copy of the
  eval's weights (auto/k03/ckpt.pt, global_step 44,212,158,464, md5 665370e8...), pulled as pbr7b2_c8/ckpt_44212158464.pt.
* Where k02 ends (geometry, hull traces): the corridor past 77% is free from z 1,260 (the floor with the fall-catch teleport *183
  lying on it, x -514..1,984) to 1,660 (ceiling). A surf ramp rises SOUTHWARD along y ~1,150-1,330 from x ~1,100 (hull-centre top
  1,412 at y 1,300, 1,524 at y 1,200) and climbs eastward (+50-60 u by x 1,400). k02's end (1,070, 1,378, 1,273, vz -573) is
  170 u north of it and ~150 u too low: a dead end ~4-7 ticks above the catch sheet. The record (ANALYSIS ONLY) keeps z ~1,500
  along the south wall at y ~1,190-1,250 and surfs that ramp up (z 1,462 at 78.6%, 1,542 at 79%, 1,687 at 80%).
* Look-ahead GE 21 (05:53 UTC, beside c8): k02 rows 6,880-7,095 (76.0-78.2%), x1.0 / x1.04 / turned -20 / +20 deg, 12 min.
* Search 3 (44.212B, 236 s beside the training): the PLAIN beam on the eval's weights to d 11,534 = **79.82%** -> spine k03 (6,900
  states, success line d <= 11,634 = 79.64%), swapped in at 44.267B; milestones + 12,500 (k02's line). k03 stays higher than k02
  (z 1,419-1,455 at x 274-624), meets the south ramp's north edge at x ~1,200-1,250 (rows ~6,760-6,780: vz -373 -> +368, |v|
  1,228 -> 804, a hard landing) and rides it up to z 1,539 at x 1,670, ending at (1,880, 1,366, 1,496) descending. The
  curriculum works that landing (bands at rows 6,749-6,764, 24-31%; 86% from 6,759) as it worked k02's kick.
* **78.906** (44.439B, median 78.64%; 0/9 past k03's line yet). Handoff: ckpt_44330123264 (108M before; pulled). From 06:08 UTC a
  copier on the box (runs/pbr7b2_snapcopy.py) keeps ckpt_latest as ckpt_<step>.pt once a minute, so later bests have a snapshot
  within ~16M steps before them (the trainer's own cadence is 2e8).
* **Look-ahead GE 21** (k02 rows 6,880-7,095, x1.0 / 1.04 / turned +-20 deg, 12 min, 9.7M runs, 75,959 cells): deepest **80.75%**
  on the grader's field, but CRAWLING (|v| ~150-300, h ~1,520-1,560, at (2,220-2,350, 1,590-1,690, z ~1,460-1,510)); 2,151 cells
  past k03's 79.82%. The loop's own k03 line is the better target (it keeps ~700-780 u/s); left as is.
* **80.623** (44.514B) and **80.644** (44.665B, median 79.21%; 3/9 past k03's line 79.64%): the curriculum is through k03's ramp
  landing (bands at rows 6,634-6,639 now, 9% vs 86-89%); 3/9 episodes ride the south ramp past k03's own end (79.82%) and die at
  z ~1,459-1,461 at (2,173-2,402, 1,424-1,557) - the fall-catch sheet *265 (x 1,040-2,928, top ~1,440) - while turning north;
  4/9 still miss the ramp (z ~1,243 on *183 at x 1,204-1,440) and 2/9 die at z ~1,321-1,334 at x ~1,611 (79.2%). The record
  (ANALYSIS ONLY) is ~230 u higher there (z 1,687 at 80%, 1,823 at 82% at (2,445, 2,408)). Handoff for 80.644: the copier's
  ckpt_44658851840 (6.3M before the eval, md5 f8de566d...), pulled.
* **80.856** (44.816B, median 79.58%; handoff the copier's ckpt_44809846784, 6.3M before, md5 c2cc9b8a..., pulled) and 80.686
  (44.892B, median 80.4%: 5+/9 past k03's line) -> **search 4** (136 s beside the training): the PLAIN beam on the 44.892B weights
  runs to **d 4,265 = 92.54%** (9,200 states; beam_tas 4,096 envs, resample 25, its horizon = the eval's longest episode + 15 s)
  -> spine k04, success line d <= 4,365, swapped in at 44.923B; the curriculum walks it back at 99-100% bands, 40 rows a step.
  k04 (rows ~ ticks at 7.67 ms): climbs from 80.8% (2,400, 1,696, 1,630, vz +403) to z 1,814 at x 2,647, crosses a platform at z
  1,565 at ~50-240 u/s (rows 7,500-7,900, 82.2-83.3%: a walk), drops to z ~800-900 at 85-90% and ends at (637, 2,918, 470) falling
  at 1,017 u/s, t 70.5 s. The record (ANALYSIS ONLY) runs the same corridor (82% (2,445, 2,408, 1,823), 86% (894, 3,563, 838),
  92% (637, 2,892, 739)). What stops the greedy is the turn north at ~80.4% (the *265 sheet), where the sampled beam survives.
* c8 after k04's swap: evals 80.796 / 80.780 / 80.731 (44.97-45.12B, medians 80.3-80.6%): every greedy still dies at the 80.4% turn
  north, while k04's curriculum, walking back from 92.5%, sat at rows ~8,180 (84.5%, the drop to the lower level) at 27-36% - ~1,200
  rows before the turn with ~55 min left. So **c9** (06:55 UTC): c8's ckpt_latest (45,186,285,568, md5 90410697...) on
  **k04c** = k04 cut at row 7,450 (82.04%; success line d <= 10,366 = 81.86%, past the turn and the climb to z ~1,814, out of a
  dive's reach), the 33 milestones c8 carried (+ 14,499 / 13,097 / 12,500 / 11,634), every other flag as c8. The loop stays on: 5/9
  past 81.86% fires search 5. First eval 80.543 (median 78.0%). A second copier keeps c9's per-minute snapshots.
* **c9 passes the turn: 80.952 (45.489B), 82.652 (45.564B), 82.477 (45.640B)** (medians 80.5-80.6%). c9's curriculum walked k04c
  back from 82.04% to the turn (rows ~6,945-6,990, 36-51%) in ~8 min and on to 78.9% in ~15 (a 5-row window on the south ramp,
  rows 6,810-6,815: 3% -> 51% by 07:22); from 45.56B one greedy episode per eval clears the turn north and the climb (z ~1,814)
  and dies further north on the same sheet (*265, z ~1,461 at (2,285, 2,627): 82.65%); the rest still die at the turn (80.5-80.9%).
  Handoff for 82.652: the copier's ckpt_45558005760 (6.3M before the eval, md5 6dec286c...), pulled.

### pbr7b2 round 7 lesson (wrap-up, 2026-10-27 07:45 UTC) - supersedes the DRAFT above
* **Score 73.590 -> 82.652** (pbr7b2_c9, eval 45.564297216B; best episode 82.65%, median 80.56%; hand-on ckpt_45558005760, 6.3M before the eval, md5 6dec286ccb0d92de94a797ea61eb2d89, pulled). Gates passed by
  the greedy line from the true start this round: the 71.3% landing + the platform climb (73.67 -> 74.40, c1-c6), **the 74.4%
  block** (74.399 -> 77.042, c8, 05:20 UTC), then three more catch levels in ~1.5 h with no hand work at all - the loop's own
  searches (k01 77.26%, k02 78.30%, k03 79.82%) carried the greedy over gate 22's catch sheet (*183, z ~1,226) onto the south
  ramp and past k03's end to the turn north at ~80.4% (the *265 sheet), 77.042 -> 80.856 (c8 44.816B); then c9, on
  the loop's search-4 line cut at 82.04% (k04c), learned the turn: 80.952 -> **82.652** (45.564B; 1-2/9 greedy past it per eval). Box: one RTX 4080S
  (57867173, 0.41 $/h) for the whole round, ~11 h.
* **What did it, in order:**
  1. **Energy at the right place**, each found by a reward-free search from the policy's own states and picked/taught with
     the loop's energy rule: e3 (gate 20's walkway, +~250 u), e5 / e6 (the 71-72.5% descent). Measured, not guessed:
     `autoresearch/analysis/pb_energy_profile.py` (energy, speed, position at first arrival per progress level).
  2. **The APPROACH, not more energy, at a block**: 19 look-aheads characterised the 74.4% block; the 20th rooted the search at
     the policy line's own states with the horizontal velocity TURNED (the recipe's front-perturb turn as a search root, no
     boost) and found the only way over (d2 -> d2c).
  3. **Stability**: c4 (kl 0.92 after the first energetic quorum) and c7 (kl 4.5: three milestones within 21 u of each other
     plus the success line, crossed within a second by every spawn on the new chain) both collapsed the greedy (61.5%, 56%).
     c8 at `--lr 1e-4` with milestones de-clustered (lines closer than the bonus they pay are one line) learned the same
     crossing in ~55 min and held (update kl median 0.12-0.19 for 2.7 h).
  4. **The loop runs itself once the policy is stable**: `--auto-spine-async` (the search beside the training, 136-236 s,
     the trainer at full speed) fired four times in 75 min (quorum 7/9, 7/9, 7/9, 5+/9) - plain beam 77.26%, branched beam
     78.30%, plain beam 79.82%, plain beam **92.54%** (9,200 states) - each swapped in at the next iteration; the backward
     curriculum learned each new segment's one hard tick-window (k02's ramp kick at rows 6,944-6,954, k03's ramp landing at
     6,750-6,765) in 15-40 min.
* **What did not / open:** e1, e5's route and turned roots at the fork (survive, keep no energy); look-ahead 21 (a hand-run GE
  from k02's rows, 80.75% but crawling at ~200 u/s) - the loop's own beam was the better line; k04 in full: its curriculum,
  walking back from 92.5%, sat at the drop to the lower level (84.5%, 27-36%) ~1,200 rows before where the greedy dies (fixed by
  the cut: c9 on k04c walked back to the turn in ~8 min, to 78.9% in ~15, worked a 5-row window on the south ramp (rows
  6,810-6,815, 3% -> 51%) and its greedy cleared the turn ~25 min after the relaunch). Still open: most greedy episodes (7/9)
  still die AT the turn; the one that clears it dies further north on the same sheet (*265, z ~1,461 at (2,285, 2,627)).
* **New, generic, default off and bit-identical (CPU-tested):** `--auto-spine-async` (45ff5c2; TRAIN_ONLY in record_ckpt;
  tests/python/test_auto_spine_async.py incl. bit-identity against the pre-change trainer); `tools/energy_splice.py` (58bb644:
  the loop's energy-stage pick + splice for an archive searched by hand); `tools/stitch_spine.py` spine segments (c3c87fb).
  Ops: a per-minute ckpt_latest copier on the box (runs/pbr7b2_snapcopy.py) so a best eval's hand-on snapshot is within ~16M
  steps of it (the trainer's own cadence is 2e8).
* **Generic rules learned:** (1) at a block the policy reaches with enough energy, try the approach DIRECTION before more energy:
  turned roots are a cheap generic search perturbation; (2) bonuses that land together are one bonus: de-cluster milestones,
  and cut the lr for a fragile skill rather than let one update undo it; (3) never stop the training for the search; (4) a
  beam line ends where its horizon ends, sometimes a few ticks above a catch sheet (k02) - harmless, the next quorum's search
  from the policy's own weights replaces it; (5) a spine much longer than the gap to the greedy's failure point spends the
  curriculum's time where the greedy never goes - cut it just past that point (k04c) when time is short.
* **Next (if the program continues):** resume c9's ckpt_latest (46,023,049,216, md5 50c3a56e3d70db7bc66b59bbfac2caa9, pulled) on k04c (autoresearch/spines/pbr7b2_k04c, md5 814044ec...) or on
  the loop's k04 in full (runs/pbr7b2/pbr7b2_c8/auto_k04, md5 f3f74722..., to 92.54%), with the c8/c9 flags (lr 1e-4, the 33
  milestones, the loop async); the next stops are the 5-row window on the south ramp at 78.9% (rows 6,810-6,815) and the turn
  north at ~80.4% (*265), then the drop to the lower level at ~84.5%.

## pbr8b3 (round 8, slot 3) - 2026-10-27 08:33 UTC start

Box: 57938261, RTX 3090, 0.377 $/h, Missouri US (machine 153789), rented 08:35 UTC for 3 h. Incumbent: pbr7b2_c9
ckpt_45558005760 (md5 6dec286c...), eval 82.652 at 45.564B. Credit at the start: $22.03, fleet burn 0.

### c1: c9's loop, continued from its last weights (08:40 UTC)
* c9's later evals (scored here): 82.477 (45.640B) / 82.152 / 82.454 / 80.717 / 80.644 / **82.564 (46.017B: 6/9 past the
  80.4% turn at 80.7-82.56%, median 81.9%, 5/9 past k04c's success line 81.86% = the loop's quorum)**. Every passing
  episode dies on the *265 sheet (z ~1,460) at (2,325-2,533, 2,256-2,601); 2/9 at 73.5-74.3% (the old block).
* **pbr8b3_c1** = resume of c9's ckpt_latest (46,023,049,216, md5 50c3a56e..., 6M after that eval) on k04c + keys with
  c9's flag set reconstructed from its run.json (the round-5 to-continue block + --demo-match-band, --auto-spine-patience 0,
  --auto-spine-ge-minutes 5, --auto-spine-energy 150, --auto-spine-async, --lr 1e-4, the 33 milestones), --steps 6e9.
  A warm resume pushing the frontier: the quorum is met, so search 5 (the policy's own beam on these weights) should
  fire at the first eval and give the next spine past the *265 sheet.
* `tools/snapcopy.py` (new, ops): the per-minute ckpt_latest -> ckpt_<step>.pt copier of round 7, now in the repo, with
  pruning (keeps the last 30 min, and the newest snapshot at or before every eval).
* c1 first evals: **82.359** (46.0236B = the resumed weights: 2/9 past 81.86%, 4/9 die at the 80.4% turn, 3/9 at the old
  74.4% block region 73.4-73.5% - the greedy is chaotic: the same weights 6M steps earlier gave 6/9 past the turn) /
  82.425 (46.099B, median 80.5%). Record gate passed (4 recordings, POV ok); config = c9's but --steps; 209k steps/s on
  the 3090 (GPU ~70%). The first updates after the resume logged kl 7.58 / 3.40 / 2.22 (c9's own start: 8.58 / 4.53
  at its 6th-7th updates, and c9 still made 82.652) - the curriculum's restart at the spine's end; watch for erosion.
  The curriculum walked k04c back from 7,450 to [6,645 / 6,650] (78.6%) in ~25 min (bands 71-78% at step 40, then
  6.8% at 6,650).
* Where the passing greedy episodes die vs the k04 line (the loop's own search-4 beam, 9,200 rows to 92.54%; rows ~
  ticks at 7.67 ms): after the turn north k04 climbs to z 1,814 at (2,647, 1,935) (81.08%), goes EAST to (2,912, 2,245)
  and lands at z 1,547 (81.27%), climbs north to (2,735, 2,663, 1,588) (81.84%) and walks a platform at z 1,565 west
  from (2,611, 2,817) at 50-240 u/s (82.2-83.3%), then drops to the lower level at 84.5-85.9% (z 1,571 -> 834) and runs a
  loop at ~900 u/s to 91.5% (846 z). The greedy episodes that pass the turn head NORTH-WEST directly (x 2,325-2,533 at y
  2,256-2,601) and fall onto the *265 sheet (z ~1,460) - they cut the corner that k04 takes east to x ~2,912.
* c1 evals 3-5: 82.512 (46.175B, median 82.2%: 6/9 past the turn, all on the *265 sheet at 82.1-82.5%; 1/9 at 24.2%) /
  82.302 / 82.528 (46.326B, median 82.3%). The quorum (6/9 past 81.86%) fired **search 1** at 46.175B (442 s beside the
  training): plain beam 9,700 (83.03%, beats the greedy frontier 10,191 by 491 < the gain 500), branched beams 9,695 / 9,701,
  then the third branched beam (forked at tick 6,951: 2 s before the first fork at 7,212, which is 0.5 s before the greedy's median fatal launch) to **d 3,091
  = 94.59%** (9,000 states) -> swapped in at 46.261B, success line d <= 3,191 - and the curriculum restarted at 94.6% while
  the greedy dies at 82%: the k04 situation of round 7 again, by the loop's own hand.
* Search 1's line (runs/pbr8b3/pbr8b3_c1/auto_k01, analysis of our own line): after the turn it flies NORTH directly like the
  greedy but HIGH (z 1,794 at (2,500, 1,999) 81.31%, 1,730 at (2,572, 2,626) 82.0%; the greedy is at z ~1,460 there, on the
  sheet), lands on the platform at (2,464, 2,920, 1,565) 82.5%, walks it west at 250 u/s (the bunny-hop cap) to 83.8%, drops
  at 84.1-85.25% (z 1,604 -> 1,232 -> 802), runs a loop at 960-1,027 u/s (86-91.4%: (212, 3,391) -> (-71, 2,727) -> (997,
  2,003) -> (1,521, 2,497) -> (923, 2,862)) and drops into the finish shaft: 92.11% (520, 2,657, 608), 93.05% (986, 2,794,
  207), 94.59% (650, 2,816, -539) at |v| 1,344, vz -1,169, t 69.0 s. The beam ran 9,467 ticks; its 85 surviving copies all
  died within ~100 ticks after tick 9,000 (gen 90: 85 valid, min d 3,085; 0 finishes).
* The finish shaft (the record as a YARDSTICK only): from 92% the record spirals down a ~400 x 400 u column (x 531-917, y
  2,583-2,989), its horizontal velocity reversing every ~0.2 s, 1,400 -> 2,200 u/s, z 720 at 92.07% -> -531 at 94.56% ->
  -1,637 at 96.5% -> -2,961 at 99.2% -> the box (z -3,509..-3,474) at 75.44 s. Not a free fall: ~4,200 u of descent ridden
  along the shaft's walls. The last 8% may be the hardest stretch, and a beam's horizon must reach it.
* **pbr8b3_c2** (09:08 UTC): c1 stopped (SIGTERM: the nohup'd trainer IGNORES SIGINT; no search was running) at ckpt_latest
  46,383,759,360 (md5 d8dce63d...), box re-deployed on 87d095c, resumed on **`autoresearch/spines/pbr8b3_k01c`** (search 1's
  line cut by the cut-ahead rule at row 7,739 = 83.92%, d 9,188, on the platform walk; success line d <= 9,288.5 = 83.75%)
  with c1's flags + `--auto-spine-cut-ahead 1000` and the 34 milestones the swap had made (+ k04c's 10,366). Record gate
  passed. Generic: G = 2 x the loop's own gain (500), a field-distance margin, no map constant.
* **c2's first eval: 83.103** (46.3843B = c1's final weights after one update; ep5 to 83.10% at (2,076, 2,746, 1,459), 7/9
  past the turn, median 82.42%) - the branch's first best over the incumbent (82.652). Hand-on: the copier's
  ckpt_46384283648 (the eval's own step, md5 7af70f27...), pulled. Second eval 82.597. c2's curriculum walked k01c back
  from 7,739 to [7,114 / 7,119] (~82.0%, the flight north over the *265 sheet to the 82.5% platform) in ~10 min and works
  it there (4.7-32% bands) - exactly where the greedy falls.
* **The finish shaft, look-ahead (09:12-09:17 UTC, box CPU/GPU beside the trainer, ANALYSIS ONLY, never training):**
  1. The policy's own beam (beam_tas --start-state on c2's 46.43B weights, 4,096 copies) from search 1's line at row 8,800
     (92.11%, (520, 2,657, 608)) spirals down the shaft to **d 632 = 98.9%** ((649, 2,765, -2,788), vz -1,817) in 3.1 s,
     then the whole population dies within 0.8 s, 0 finishes; from row 8,700 (91.42%): d 3,166.
  2. Why (core probe, runs/pbr8b3/pbr8b3_la/shaft_probe.py, + zones.hull_probe of the brushes): every copy dies in
     trigger_teleport *132 (target start), a TWISTING central blade above the finish box - its deadly standing-hull region
     is a diamond at z -2,800, a block x 576-896 / y 2,624-2,784 at -2,900, other rotated shapes down to -3,370 - inside
     an AABB (416..1,024, 2,464..3,072, -3,374..-1,679) that covers the whole column. The finish trigger *123 (z -3,509..
     -3,474) and the fatal tlp_end teleport *148 (z -3,532..-3,497) sit right below; the core tests the goal box (swept
     segment) BEFORE triggers, so tlp_end cannot pre-empt a finish. The shaft above is a spiral func_wall (*43, z -2,688..
     -1,408) with trigger_push volumes (*195 / *196 / *48) and catch teleports (*181, small *254-*256).
  3. **Go-Explore (the loop's GE settings, 12 threads, roots = the 98.9% lineage's own states at z -1,028..-2,720) found
     the FINISH in 0.1 min** (2 iterations, 67k runs, 1 goal hit, d 4): from (565, 2,746, -2,558) at vz -1,713 it lands on a
     ledge at the spiral's bottom (z ~-2,613), drifts east around the blade at 230-340 u/s and drops at x 720-855, y
     2,814-2,850 into the box ((855, 2,815, -3,431)), 184 states (ge_replay links 2/2). Saved in runs/pbr8b3/pbr8b3_la/.
  So the end is in reach of the loop's own GE stage from the policy's states; nothing map-specific is needed. The route
  from 83.9% to the shaft (the platform walk, the 84.5% drop, the 86-91% loop) is the policy's own (search 1's line).
  4. The policy's own beam on c2's ~46.47B weights from search 1's line at row 7,800 (84.12%, on the platform's west end):
     d 2,990 (94.8%); from 8,000 (85.25%, mid-drop): d 648 (98.9%, the blade); from 8,400 (88.53%): d 2,812. Past the 83.8%
     platform the POPULATION already flies the rest down to the blade; what is left is the greedy's reliability over
     82-84% (c2's curriculum band now) and the blade (a GE stage's find).
* c2 evals 82.597 / 82.712 / 82.343 / 82.891 / 82.951 / 82.921 (to 46.837B, medians 82.1-82.5%; 8/9 past the turn, 0/9
  past k01c's success line 83.75%). The curriculum got through the 82% band (the flight to the platform) in ~25 min and
  walked back to [6,869] (79.4%, bands 29-48%). Where the greedy still fails, vs k01c (ANALYSIS of our own lines): at
  80.6-81.2% the spine RIDES the ramp up (|vh| 933 -> 816, vz +304..+449, apex z 1,825 at 81.69%) and flies straight north
  at x ~2,550 (823 u/s) onto the platform (z 1,565 at (2,581, 2,844)); the greedy carries MORE energy (h 2,251-2,293 vs
  2,228-2,248) but keeps it as horizontal speed (|vh| ~970-1,000), peaks at z 1,676 and drifts WEST (vx -330..-600), so
  at y 2,628 it is at x 2,246, z 1,523 - below and west of the platform - and drops onto the *265 sheet.
* Box extended to 12:42 UTC (09:42; credit $20.9, fleet burn 1.06 $/h with 3 boxes).

### THE PLATFORM FALLS: 83.103 -> 92.699 (2026-10-27 ~09:50 UTC, c2)
* c2 evals after the curriculum passed the 81% ramp: **90.697** (46.913B), 84.726, **92.699** (47.064B), 90.875 (47.139B),
  84.158 (47.215B). At 47.064B ep5 rides the ramp onto the platform, walks it, drops, flies the 86-91% loop and dies in the
  finish shaft at (647, 2,829, z 280) at 80.4 s (92.70%: the top of catch *181, whose hull covers ~59% of the column at
  z 200); the others: 84.1-84.4% (2), 82.2-82.8% (2), the 74.4% block region 73.5-74.4% (3), 12.0% (1). Hand-on for 92.699:
  the copier's ckpt_47061139456 (2.6M before the eval, md5 b8e5c78f0eb54f2ea3914f1ee6c97bbe), pulled.
* The stops now (47.14-47.21B): **84.1%** - 6/9 at 47.215B fall off the platform's WEST END at (1,654-1,670, 3,153-3,172)
  onto a sheet at z 1,479 (search 1's line leaves the platform airborne there: (1,646, 3,201, 1,604) ducked at 84.12%,
  then (1,462, 3,315, 1,586) and the drop to z 1,232 at 85.25%); **90.7%** - at 47.139B 3/9 flew the loop and CRAWLED at
  its exit (x 1,171-1,325, y 2,668-2,936, z 709-844) until the eval's 30 s patience; **92.7%** - the shaft's top catch.
* **The loop under the cut rule (first live use):** c2's search 1 fired at ~47.215B (quorum: 5+/9 past 83.75%), ran 397 s
  beside the training, and its PLAIN beam (7,845 states) was cut by `--auto-spine-cut-ahead 1000` at d 8,049 = 85.9%
  (greedy frontier 9,052 - 1,000) -> swapped in at 47.224B, success line d <= 8,149 (85.74%); the curriculum walked it
  back at 100% bands, 40 rows a step.
* c2 evals 84.472 / 90.672 / **92.570** / 90.780 / 92.437 (47.29-47.59B; medians 84.2-84.4%): 4/9 past the new success line
  (85.74%) at 47.517B and 47.592B - one short of the quorum. Where they stop (47.517 / 47.592B): 84.4% (falls off the
  platform's end, 2-3/9), **90.6-90.8% (4-5/9: they fly the 86-90% loop and then CRAWL ~30 s at (1,190-1,320, 2,500-2,920,
  z ~710-750) until the eval patience)**, 92.4% (1/9: into the shaft, dies on catch *181's top at (515, 2,772, 280) - search
  1's line passes it on the EAST side at (986, 2,794, 207)), 82.5-82.8% (1-2), 72.9% (1), 3.6% (1). The policy's own
  line from search 1 crosses the 90.7% spot ~150 u higher (z 870-940), flying WEST over it into the shaft. The training
  stall fraction rose 33% -> 66% (crawl 70-83%) over 47.41-47.60B: the training episodes stall there too.
* c2 evals 92.606 / 92.342 / 92.364 / 90.837 (47.67-47.89B): **the median greedy episode now reaches 90.7%** (d ~5,323;
  6/9 past 85.74% at 47.819B). The quorum fired **search 2 at 47.668B, and it could not reach the greedy**: its plain
  beam from the true start lost its leaders at the 80.4% turn / the *265 sheet (gen 70, t 7,000: 172 of 4,096 copies
  valid, min d 10,419) and its best line died at the platform's end (d 8,860 = 84.5%, t 55.2 s); the population then
  re-cloned from copies stuck at the 74.4% block (gen 75: 4,096 valid, min d 15,585). The greedy floor (32 envs) is ONE
  trajectory - every search draws spawn seed 0, and that draw's greedy line dies at 84.5% where the eval's nine greedy
  episodes pass 5-6 times out of 9. Its branched beams forked after that (ticks 8,647-8,778) and gave the same line; it
  went on to GE stages from 84.5% that cannot beat a 90.7% frontier by 500 u. The stochastic population is much weaker
  than the greedy at the gates the greedy has learned.
* **pbr8b3_c3** (10:59 UTC): c2 stopped at 47.936B (search 2's process tree killed by hand first: SIGTERM skips the
  trainer's atexit), resumed on **`autoresearch/spines/pbr8b3_k01d`** = search 1's c1 line cut by the cut-ahead rule
  against c2's frontier (5,323): row 8,836, d 4,322 = 92.44%, in the upper shaft; success line 92.26%. 36 milestones
  (+ k01c's 9,288.5 and c2-k01's 8,149). Every other flag as c2. First eval 92.660 (median 90.7%, 1/9 past 92.26%).
* **`--auto-spine-from-spine S`** (new, generic, default off; tools/auto_spine.py --start-spine / --start-keys /
  --start-row): the search's beams start from the CURRENT spine S seconds before the row where it reaches the greedy
  frontier (before its end if it never does), and the new line = the spine's rows before that + the beam's own states
  (+ keys). The beam then only has to fly from just before the greedy's failure point, not the whole route at
  temperature 1. Look-ahead evidence for it: from search 1's line at 84.12-88.5% the policy's beam reached 94.8-98.9%.
* c3 evals 92.660 / 92.550 (47.94 / 48.01B, medians 90.7%, 1-2/9 past 92.26%); its curriculum walked k01d back through
  the shaft entry and the 86-90% loop at 80-89% bands in ~10 min to the 84.5% drop band (rows 7,946-7,951, 8-17%).
  Tests: --auto-spine-from-spine smoke + bit-identity, --auto-spine-cut-ahead bit-identity re-run, 5/5 (143 s, CPU).
* **pbr8b3_c4** (11:02 UTC): c3 stopped at 48.015B (no search running), box re-deployed on 58da965 (a first deploy
  failed: the k01c files I had written on the box before committing them were untracked there and blocked the checkout;
  removed, same md5 as the commit), resumed on k01d with `--demo-tau0 7951` (c3's last band) and
  **`--auto-spine-from-spine 4`** (S = the loop's own --roots-secs), every other flag and the 36 milestones as c3.

### The loop starts on its own spine: 92.699 -> 94.852 (2026-10-27 11:10-11:25 UTC, c4)
* c4 evals 90.698 (48.016B = c3's weights; median 72.9%) / 92.925 / **94.852** (48.167B) / 93.637 (median 92.7%) / 92.860.
  At 48.167B ep7 reached **94.85% at (663, 2,771, z -710)** in the finish shaft at 73.3 s; the others: 92.5-93.1% (3, the
  shaft's top catch *181), 90.85% (1, the crawl), 82.6-82.9% (2), 68.4% (1), 61.4% (1). Hand-on: the copier's
  ckpt_48162144256 (4.7M before the eval, md5 9cf130ab26140376eeb39f8b42869fa8), pulled.
* **The first live `--auto-spine-from-spine` search** (c4's search 1, quorum at ~48.24B): `--start-row 8313` (k01d's row at
  (-58, 2,635, 858) in the 86-91% loop, 4 s before the greedy frontier 4,184), beam 1,300 ticks x 4,096 in **10 s**,
  frontier **d 542 = 99.05% (the blade)**; the line (9,413 rows) cut by `--auto-spine-cut-ahead 1000` at row 8,967 (d
  3,182, 94.4%), success line d <= 3,282; the whole search 19 s (c2's search 2 from the true start: 1,100+ s, no line).
  The curriculum then worked the shaft's top section (rows 8,722-8,727: 3-28%).
* Box extended to 14:24 UTC (11:24; credit $19.09, fleet burn 1.06 $/h, 8.6 h to the reserve).
* c4 evals 93.451 / 94.669 / 93.655 / 94.775 / 94.685 (48.39-48.70B, medians 90.7-93.0%; 0-1/9 past 94.26%). The shaft is
  a stack of catch blades: at 48.62B 6/9 enter the shaft and die in catch *181's top part at z ~262 (92.4-93.0%), 1/9
  goes on to z -647 (94.78%) and dies a tick later at 1,350 u/s; at 48.70B 4/9 crawl at the 90.7% entry gate again.
  The curriculum works the entry approach (rows 8,557-8,562 of c4's k01, 20-43%). Stable: update kl median 0.28 (p90
  0.52; the 7.51 max is the resume spike), stall fraction 19-26% (c2: 66%), 257k steps/s.
* Endgame dry run (11:56 UTC, box CPU/GPU beside the trainer, ANALYSIS): auto_spine.py as the loop would call it near the
  end - `--start-spine` c4's k01 at row 8,700 (91.6%), a pretend greedy frontier d 900, --cut-ahead 1000, energy 150,
  GE 5 min/stage - to check the path beam-from-spine (stops at the blade, d ~540) -> GE stages -> a line to the finish.
* **The endgame dry run reaches the FINISH by the loop's own rule** (one auto_spine call, ~6 min, analysis only): beam from
  c4's spine row 8,700 -> d 1,358 (97.6%); the energy-fork branched beam (fork at tick 8,946, before a -1,518 u loss) ->
  d 420 (99.3%); GE stage 1 (tail roots: that line crawls 1.6 s past its stall) -> **d 0**: a 9,488-row line from the
  true start into the finish box at (861, 2,754, -3,435), the GE chain 125 states from an UNBOOSTED root at row 9,363,
  ~72.7 s from the start (the record: 75.18 s). Saved as runs/pbr8b3/pbr8b3_la/dry1/spine.npy (+keys, md5 560a4cba...).
  Kept in reserve: the loop will build its own when the greedy reaches the blade; a full spine now would put the
  curriculum at the finish, ~6 shaft blades ahead of where the greedy dies (round 7's k04 lesson).
* c4 evals 94.990 / **95.383** (48.922B: ep3 to (685, 2,744, z -1,033), ep2 94.61% at z -603; 5/9 die on *181's top at z
  ~262, 1/9 at 82.0%, 1/9 at 72.4%) / 94.825 (median 94.7%) / 94.812. Hand-on for 95.383: the copier's ckpt_48912924672
  (8.9M before the eval, md5 d8173269cea1dbcc294a8f93d50fa887), pulled. The loop's **search 2** (from the spine, quorum
  ~48.997B) ended at 49.063B after 259 s: NEW SPINE from **beam+ge** (the GE stage beat the beam), cut by the rule at d
  2,016 (96.5%), success line d <= 2,116 (96.3%); greedy frontier then 3,027 (94.7%).
* c4 evals 94.847 / 94.712 / 94.882 / 94.701 (49.15-49.37B): **8/9 greedy episodes now reach 94.5-94.7%** and die at the
  same blade, z -575..-625 at (564-682, 2,838-2,941) - 150-180 u WEST of where c4's k02 crosses that height ((745, 2,906,
  -559) at row 8,880, then (688, 2,711, -1,007) at 95.31%); 0/9 past k02's line (96.3%). The curriculum's band is that very
  crossing (rows 8,882-8,887, 12-52%): the right window, learning. Box extended to 15:33 UTC (credit $17.84, fleet burn
  1.06 $/h, 7.4 h to the reserve).
* c4 evals 94.827 / 94.881 / 94.641 / 94.845 (49.53-49.75B): still at the z -600 blade, 0/9 past 96.3%; the curriculum
  oscillated at rows 8,862 (8% success) <-> 8,867 (64%) for ~40 min. **Why**: k02 (search 2's line) is the plain beam only
  to row 8,738 (92.2%); from there it is the GE stage-1 chain from a **x1.04 boosted root** (uniform explorer actions,
  606 states, uncut to the finish, d 0), so the curriculum asks the greedy to copy an explorer's precise path at 4% more
  speed than it has - round 5's lesson ("a depth spine from a boosted root asks for speed the policy does not have"). A
  probe confirms it: from k02's own rows 8,760 / 8,820 / 8,850 the policy's whole population (4,096, c4's 49.8B weights)
  dies at the blade (frontier d 3,161-3,356, extinct within 200-400 ticks).
* **pbr8b3_c5** (13:10 UTC): c4 stopped at 49.813B (no search running), resumed on **`autoresearch/spines/pbr8b3_k03p`**:
  c4's search-1 spine rows 0-8,699 + the PLAIN beam of the policy's own population from row 8,700 (the endgame dry run's,
  c4's 48.7B weights: no fork, no boost) to 97.6%, cut by the cut-ahead rule against c4's frontier 3,128 at row 8,997
  (96.28%), success line d <= 2,228; it crosses the blade at x ~720-790 like k02, on the policy's own line. Every other
  flag as c4 (from-spine 4, cut-ahead 1000), 38 milestones. Generic lesson: the loop's GE stage may hand back a chain from
  a boosted root; when the curriculum stalls inside such a chain, a plain beam of the policy's own population through
  the same gate (when one exists) is the better spine.
* c5 evals 94.819 (49.814B = c4's weights) / 94.864 / 94.534 / 94.553 / 94.667 (to 50.116B, medians 94.4-94.6%): 0/9
  past 96.28%. Its curriculum walked k03p back from 8,997 to rows 8,857-8,867 (z -60..-108, ~0.5 s before the z -600
  blade) - the same rows c4 stalled at on k02 - and is LEARNING there: band 8,867 42-49%, 8,862 19-48%, 8,857 3.6 -> 4.4
  -> 6.4 -> 9.9 -> 13.9% over ~15 min (the advance threshold is 30%). The blade is hard for the policy on its own line
  too; the approach (x ~720-790 at z -600, the greedy comes in at 564-682) is set ~0.5 s before it.
* c5 evals 94.844 / 94.989 / 94.864 / 94.655 (50.27-50.49B; 0/9 past 96.28%). The curriculum passed the 8,857 band and
  walked back to [8,827 / 8,832] (z ~150-200, the shaft's top catch approach; 4-33%) by 13:43 - it is retracing the
  shaft toward where the greedy's approach first leaves the spine's line. Box extended to 16:43 UTC; the credit was
  topped up ($41.6 at 13:44, 29.9 h to the reserve).
* **Where the greedy loses the blade (13:55-14:05 UTC, box, ANALYSIS of our own lines):** three greedy recordings of
  c5's 50.53B weights (record_ckpt --dump-states, seeds 0-2) all die at 94.56-94.59% (x 533-547, z -588..-597). Aligned
  by field d, the greedy tracks k03p within 20-80 u through the whole 86-91% loop (~250 rows / 2 s later in time: it
  walks the platform slower) up to the shaft entry (spine row 8,700 (783, 2,687, 789) vs greedy (764, 2,672, 826)), and
  LEAVES it at the entry (spine rows 8,740-8,780): the spine turns north-west (v (-592, 676) at (510, 2,662, 631)), the
  greedy heads north (v (-50, 956) at (531, 2,727, 660)), then reverses early on the way down and meets the -600 blade
  ~150 u west. Probes (beam_tas --start-state on the greedy's OWN dumped states, 4,096 copies): from its states after
  the entry (z 600 / 300 / 0) the whole population dies at the blade within 100-200 ticks; from its state at the loop's
  start (row 8,310, (1,221, 3,441, 897)) the population flies the loop, enters like the spine and passes the blade to
  **d 1,778 = 96.9%**. So the gate is the shaft-entry maneuver (~0.3 s), ~50-90 rows before c5's curriculum band (8,827).
  (A first, wrong reading - a "shortcut" to the shaft's north lip - came from matching the greedy and the spine by
  height instead of by d: the greedy's first z 900 crossing is the loop's start, the spine's the shaft entry.)

### The shaft-entry maneuver learned: 95.383 -> 96.582 (2026-10-27 ~14:05 UTC, c5)
* c5 evals 95.370 / **96.500** (50.795B) / **96.582** (50.871B: 6/9 at 96.44-96.58%, all dying together at the next
  blade, z ~-1,635..-1,653 at x 648-749) / 96.536 (50.946B, the same 6/9). The curriculum on k03p (the policy's own beam
  line) taught the entry maneuver in ~55 min. Hand-on for 96.582: the copier's ckpt_50862227456 (8.4M before the eval,
  md5 b30fef4a0180681fbba2a1ddd50be98b), pulled.
* c5's search 1 (from the spine, start row 8,474, 213 s) handed back **another x1.04-boosted GE chain**: its beam reached
  only d 2,311 (96.0%) - less than the greedy (2,034) - and GE stage 1 (factors [1.0, 1.04] in ONE stage: the deepest
  chain wins and is usually the boosted one) rooted a 659-state chain at row 8,737, before the shaft entry, uncut to the
  finish; cut by the rule at 98.2% (success line 98.04%). The curriculum now walks an explorer's route through the whole
  shaft - the k02 trap again.
* Probes (box, ANALYSIS): in the shaft the policy's stochastic population is WEAKER than its greedy - from the greedy's
  own dumped states 1-4 s before its 96.5% death (record_ckpt --dump-states, the 96.582 weights, episode 5 of seeds
  100-108: 96.51%) the beam's frontier is d 2,755-2,916 or extinct; so a beam cannot extend the greedy here and the
  loop's search falls to GE, whose stage 1 mixes boosted roots.
* So (14:30 UTC): auto_spine.py by hand with the GREEDY's own recorded line as `--start-spine` (episode 5 + its held
  keys from the .jsonl, `--start-row 9425` = just before its death, so the line IS the greedy's), `--greedy-d 1997`,
  and `--ge-boost "1.0;1.04;1.08"` (UNBOOSTED first, a stage of its own): GE roots in the greedy's own last 4 s.
* **98.484** (c5, 51.022B: ep3 98.48% and ep0 98.35% past the z -1,640 blade to z ~-2,580 at (522-560, 2,777-2,831),
  the next blade; 5/9 at 96.5%). Hand-on: the copier's ckpt_51014795264 (6.8M before the eval, md5
  f027b8f1f6de9686c4170a2c4b472000), pulled. The next eval (51.097B) eroded: 5/9 back at the 74.4% block region, the
  boosted-chain band at 0.0%.
* The greedy-rooted search (auto_spine.py by hand, `--start-spine` = the greedy's own recorded line, episode 5 of the
  96.582 weights, + its keys, `--start-row 9170` = 2 s before its death; `--ge-boost "1.0;1.04;1.08"`): the plain beam
  2,752; the ENERGY-FORK branched beam (forked at 9,172, right at the start) to **d 1,069 = 98.13%** - beats the greedy by
  928 u with no Go-Explore -> spine **`pbr8b3_g2`** (the greedy's own line + the policy's own branched beam). (A first try
  with `--start-row 9425`, a few ticks before the death, gave no line at all: the beam went extinct without a frontier,
  and auto_spine skips every later stage when the plain beam returns nothing.)
* `--auto-spine-ge-boost SPEC` (new, generic, default off; c70f589): the loop's Go-Explore ladder for auto_spine
  --ge-boost; tests/python/test_auto_spine_ge_boost.py 3/3 incl. flag-off bit-identity (157 s, CPU).
* **pbr8b3_c6** (14:28 UTC): the 98.484 weights (c5's ckpt_51014795264, before the erosion) on g2 + keys, every c5 flag +
  `--auto-spine-ge-boost "1.0;1.04;1.08;1.12"` (unboosted roots first, a stage of their own), 39 milestones.

### THE FIRST GREEDY FINISH FROM THE TRUE START: 109.846 (2026-10-27 ~14:55 UTC, c6)
* c6 (the 98.484 weights on g2, unboosted-first GE ladder) evals: 96.565 (51.015B = c5's weights) / 96.589 / 98.341 /
  98.461 / 98.452 (51.317B, median 98.4%) / **109.846 (51.393B): 2/9 greedy episodes FINISH** - ep8 in **71.28 s** (the
  score's time; episode 71.3 s), ep3 72.5 s - both dropping into the box at (785-858, 2,703-2,732, z -3,422..-3,448)
  at ~2,300 u/s; 5/9 die at the ledge-level blade (98.23-98.50%, z ~-2,575..-2,604), 1 at 95.35%, 1 at 94.55%, 1 at
  78.2%. **The human world record on our zones is 75.18 s**; the policy's best finish is 3.9 s faster. Next evals
  98.501 / 99.052 (median 98.5%; 7/9 past g2's success line 97.95%).
* Hand-on for 109.846: the copier's ckpt_51386515456 (6.3M before the eval, md5 153e975f0e669867dec84d0715c33cdf),
  pulled.
* c6 evals after the first finish: 98.501 / 99.052 / 99.185 / **110.969** (51.770B: 3/9 finish, best 70.94 s) /
  **112.071** (51.846B: 4/9 finish, best 71.22 s, median 99.7%) / 99.430 / 108.657 (51.997B: 1/9, 73.63 s). Hand-on for
  112.071: the copier's ckpt_51837927424 (7.9M before the eval, md5 0456576922a3e79de68ef4e450eef01b), pulled.
* c6's search 2 (from-spine on g2, retried at ~51.72B with the patched auto_spine) took 625 s: its beam from the spine
  reached only d 4,677; GE stage 1 (x1.0) d 926 - short of the greedy (867) - and stage 2 (x1.04) rooted a 755-state
  chain at row 9,061 (the shaft entry) into the box -> spine k02 to d 0, success line d <= 100. Its curriculum: **0.0%**
  at rows 9,750-9,815 (0.5 s before the box) - spawns on a boosted explorer chain fail even there (finishes DO count as
  successes: the outcome ORs fleet.goal_hits()). Same trap a third time.
* A hand search rooted at g2's END (auto_spine.py, `--start-spine` g2, `--start-row 9669` = the ledge at the spiral's
  bottom; the first try from row 9,400 put GE's roots above the ledge and found only d 1,961 / 903) - plain beam d 1,047,
  then **GE stage 1 UNBOOSTED -> d 1.1 (the finish box)**: a 386-state chain from row 9,415 -> spine
  **`pbr8b3_f1`** (the greedy's own line + its branched beam + a short unboosted chain into the box).
* The greedy recorder does not reproduce the trainer's eval: record_ckpt on the 112.071 weights (seeds 300-308) gives 0/9
  finishes and mostly early deaths (56-82%; one 320 s truncation), where the trainer's evals of those weights finish
  4/9. Spawn draws differ; the policy is specialised to the eval's spawn. So the greedy's own FINISHING line could not be
  dumped.
* (A process slip: `pkill -f 'runs/pbr8b3_la/gs3'` over box.py ssh matched its own shell - CLAUDE.md's self-match
  warning - and killed the session before the next command; the trainer was untouched.)
* **pbr8b3_c7** (15:41 UTC): the 112.071 weights (c6's ckpt_51837927424) on f1 + keys (success line d <= 101), c6's flags,
  40 milestones (+ g2's 1,169). c6 stopped at ~52.0B.
