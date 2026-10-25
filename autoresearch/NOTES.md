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

## pbr5b4 (round 5, slot 4) - 2026-10-25 19:48 UTC start

Incumbent: pbr4b3_t11 traj_23590338560 (52.937, 9/9 greedy at 52.8-52.94%), weights ckpt_23551016960
(md5 7fbcd458..., = $AR_HOME/incumbent/r4/ckpt.pt; tick 7.667 ms, spine k05 swapped in at 23.364B).

### No box at the start (19:50-20:05 UTC)
* `box.py rent` refused: two boxes already hold 0.984 of the fleet's 1.2 $/h cap and the only offer that
  passes the query is a 0.62 $/h 5080 (no 3080 / 4070 / 60 Ti class offer with >= 32 threads at the
  time). A background watcher retries the rent every 60 s; deploy + launch are scripted to follow it.

### Where the incumbent's line loses its energy (ANALYSIS of our own lines; the record is ANALYSIS ONLY)
* The incumbent greedy dies at 52.9% falling at vz -903 into a fail volume at z ~2,200 (-2,434, 1,100),
  36.5 s in; its energy height h = z + |v|^2/1600 is ~3,680 at 50%, ~3,520 at 51%, ~2,810 at 52% (the
  record: 3,565 / 3,614 / 3,193).
* Spine k05 (the beam to 4,344 + an unboosted view-follow GE chain to 56.22%) is a chain of crashes
  from 50% on: single-tick losses of 254 (wall, 50.2%), ~245 (51.0-51.2%), ~330 (51.6-51.7%), 335 (a
  landing at 52.7%), 348 (54.0%), 171 (56.1%) u of h. From 50% to 56% it loses ~1,620 u (3,772 ->
  2,151); the record ~790 (3,565 -> 2,772). k05 ends at 56.22% at 209 u/s (h ~1,993) at the top of a
  climb (z 1,966); the record's route climbs on to z ~2,508 by 58% with h ~2,845. Prediction: once the
  greedy flies k05, the next gate needs ~+600-800 u of energy height, beyond the GE ladder's x1.12.
* `--auto-spine-pick-energy` (new, 42e46d9; off = byte-identical; TRAIN_ONLY): a GE stage's chain ends
  at the archive cell past the spine's bar (greedy frontier - gain) with the highest training
  potential -100 d / d0 + K h / 1000 (K = the run's --energy-pot, the objective's own two terms) that
  survives >= 25% of 32 uniform-random 1 s rollouts (a fall into a fatal volume keeps its h until it
  hits), instead of the deepest cell; the spine ends at the pick. tools/ge_replay.py --pick-energy /
  --pick-max-d / --pick-d0, tools/auto_spine.py --ge-pick-energy / --d0. Generic: the reward's own
  potential, no map constant. Tests: test_auto_spine.py (pick order, plumbing, the cut, survival).
* Local check of the pick (20:07-20:15 UTC; explore_phase1 --exact-chains --view-follow --tick-ms 8,
  3 min x 2 threads on this machine, 12 roots = k05 rows 4290-4344, 72,816 cells): the DEEPEST cell
  (56.15%, h 2,105) is a dead end (no run lived past it); the highest-potential cells past a 53.7%
  bar are all free-falling states at 54.2-54.3% (vz -1,000..-1,200, h ~2,400 kept until the impact),
  and 1-s random rollouts barely filter them (2 of 64 at >= 25%). So viability now comes from the
  explorer's own tree: a candidate needs a subtree reaching >= 1 s past it (ge_replay.ticks_ahead;
  0d4e7ee). 1,514 of 28,980 candidates pass; the pick is 55.95% at h 2,160 with a 1.4-6 s
  continuation. Every viable chain past 53.7% has h <= ~2,330: on this route the energy is gone by
  the 52-54% landings (the record: h 3,193-3,234 at 52-53%). If 56% is an energy wall, the fix is
  upstream (a smoother line through 50-54%), not a deeper pick.
