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

## pbr1b2 (round 1) - working notes

### 2026-10-23 14:34 UTC - start

* Box: vast 3090 machine 144293 at 0.375 $/h (rented 14:32 after ~5 min of "no offer passes":
  the >=32-thread filter leaves very few offers once three branches hold boxes).
* WR line on the goalc field (ANALYSIS ONLY): d is monotone along the record's line except
  the first ~4 s - the record goes EAST from the spawn (d 57,232 -> 58,664, +1,430 u), turns
  and comes back west past the spawn at ~6.5 s - and one 50 u bump at ~21 s. So apart from
  the start the field is a good guide along the line a human takes (the deception risk is
  free flight across voids, not a field that points the wrong way).
* NEW tools/transfer_ckpt.py: a cross-map warm start. A plain resume of another map's
  checkpoint would restore its greedy-death pool (raw sidistic coordinates, restored whatever
  the map), its --unstuck best reading (a sidistic distance) and its config map (the record
  gate then looks for a map the box does not have). The tool keeps the weights + Adam, sets
  the target map, drops all run state and restarts the step at 0 (generic: nothing in it
  reads the target map). sid_ref -> runs/pbr1b2/xfer/sid2pb_ckpt.pt.

### 2026-10-23 15:16 UTC - THE START GATE (analysis only: BSP entities + scripted physics)

The first gate of this map is the start itself, and it is an ENERGY gate the field does not show:

* The spawn platform (x ~650..1150, y -3155..-2845, floor z 3617) is walled north, south and
  west (scripted walking/jumping/duck-jumping west stops at x ~650-730). Its west edge carries two
  func_conveyor steps under two trigger_push volumes that push DOWN at 2500 u/s (angles 90 0 0):
  a trap, not a booster. Its north edge has the "healingdoor" triggers/doors (dmg -10000).
* Jumping over the platform's north and south strips (x 856..1144, y > -2906 or < -3098, hull
  top above ~3700) is a fatal trigger_teleport (start, bbox x 865-1151 z 3696-3780); the centre
  band y -3098..-2906 and anything at x < 856 are safe (killmap: one-tick teleport test).
* The only exit is EAST into a U-trough (x ~1150..2400) whose floor is a fatal teleport (start,
  bbox x 192-2396 z 3236-3667): walking east off the platform dies at 2.5 s, z 3312.
* The WR (analysis only) pumps that trough: east down the ramp to 1,150 u/s, up the far side to
  x 2361 z 3635, back west to x 1126 at z 3713 (~800 u/s), then flies WEST over the platform's
  centre band and the west wall at z ~3710-3734 and drops at x ~70-300. d rises +1,430 u on the
  way out and only falls below the spawn's after ~6.5 s.
* So: a ramp speed-up (energy gain z + v^2/2g of ~+430 u) before any field progress is possible,
  under a ratchet that pays nothing until it is done, beside two kill zones that punish the
  random jumps an exploring policy makes. Exactly the CLAUDE.md 0b case ("speedup on ramps that
  are not shown by reward vector field") - so the fix must be generic, not a trough term.

### 2026-10-23 15:16 UTC - arm 1: pbr1b2_xfer (sidistic finisher -> princess, full recipe) - NULL

* zero-shot (step 0, 9 greedy eps on the box): 0.394 - it walks the platform at ~250 u/s for the
  whole 300 s, never drops into the trough.
* training (warm, recipe, 117-127k steps/s - GPU-bound on this map, 99% util, so ~120k is this
  map's 3090 rate, not a bad box): evals 0.406 (0.5M) -> 0.066 (76M) -> 0.088 (151M). The
  reservoir's deepest state stayed at 99.36-99.37% of d0 (~360 u, the platform's west edge).
  By 151M training episodes were ~2 s long: it learned to SUICIDE into the trough (stalling
  15 s costs 1.5 of time penalty, dying costs nothing, the pump pays nothing). Stopped at 155M.
* the transfer prior does not help at this gate; the warm surfer is not pulled into the trough
  because nothing pays for energy behind the frontier (--energy-pot-front-bonus pays only when
  the record ENTERS the frontier band, i.e. after a successful pump).

### arm 2: pbr1b2_xen - the same warm start with --energy-pot-front 0 --energy-pot-front-bonus 0

The recipe's energy potential (K 3, Phi = K h / 1000, h = z + |v|^2 / 2g) paid on every call
instead of only once at the band edge: a pump in the trough is paid as it happens (~+1.3 for the
WR's +434 u), so the gate is shaped by a physical invariant, no map constant. Risk: energy
farming in the trough instead of exiting (sidistic's "dip lock" reason for the front form).

### 2026-10-23 15:53 UTC - arms 2 and 3 (warm, sid2pb at step 0, 4e8 budget each)

* arm 2 pbr1b2_xen (+ --energy-pot-front 0 --energy-pot-front-bonus 0, plain energy potential):
  evals 0.388 / 0.331 / 0.096 (0.5M / 76M / 151M). Same suicide as arm 1, now the fastest one:
  at 151M all 9 greedy episodes JUMP INTO THE NORTH KILL STRIP within 1.4-2.4 s. Note the plain
  energy term itself pays a jump-suicide (+0.13: the jump's energy is paid, a death keeps it, a
  safe landing would charge it back) - the "no death refund" rule makes dying mid-air the best
  end of a jump. Stopped at 155M.
* arm 3 pbr1b2_xen0 (arm 2 + --time-pen 0): evals 0.408 / 0.425 / 0.454 - no suicide (episodes
  ~8 s, stall kills 54%), energy churn +1,006/-1,079 u per episode, but still on the platform
  (reservoir deepest 99.38% of d0). Stopped at 153M. So the time penalty IS what drives the
  suicide here; removing it is necessary, not sufficient.
* LESSON: the time penalty + a dead-end start = a suicide attractor (0.1/s of living cost against
  0 for dying; the platform's kill strips make dying a 1.4 s action).

### arm 4: pbr1b2_xsp = arm 3 resumed at 151.5M + --respawn-speed 1.0 1.6

The reservoir rows (most of the spawn pool) spawn with their speed x U(1.0, 1.6) instead of
U(0.9, 1.1) (an existing flag: "practices speed-gated jumps"). Generic: "what if I were faster
here". A trough state falling at ~660 u/s gets up to ~1,060 u/s, which is about the energy the
WR carries out of the trough (+430..590 u of z + v^2/2g); the ratchet pays a trough spawn for
every new record it sets on the way back west (its b_0 is its own spawn d), so the critic can
learn what energy is worth there before the policy can make it.

### 2026-10-23 16:40 UTC - arm 4 null; the gate is a CLIMB at the platform's NW corner, not a pump

* arm 4 pbr1b2_xsp (arm 3 + --respawn-speed 1.0 1.6, from 151.5M): 0.444 (152M) / 0.436 (227M),
  reservoir deepest unchanged, unstuck T rose to 0.15. Stopped. All four RL arms stall with the
  frontier at the platform's NW corner (x ~650-730, y ~-2900): exactly where the way on starts.
* NEW beam_tas --score rec_e (elites by record, then energy height) + --dump-elite-states: with the
  RL policy or the sidistic finisher as the proposal, 4096 lineages x 22-89 generations never
  gained more than ~+300 u of energy height and never left the platform; --eps and --macro-hold
  are refused for keys-hold checkpoints. The sidistic finisher spawned ON the start ridge (bench,
  constructed states) rides it for 15 s but adds no energy: it never learned to pump.
* Go-Explore phase 1 (tools/explore_phase1.py, random 95%-repeat bursts from the map start, 128 u
  cells x 8 speed bins, NEW --maxvel 3500) left the platform in 2.5 minutes (3M runs): best cell
  3.19% of the grader's d0, under the platform at z ~3330 - the region the WR crosses at 8.5-9.9 s.
  Its chain: walk to the NW corner -> onto the NORTH LEDGE (z ~3714 origin, 68 u above the
  platform floor) -> west along the ledge to x ~90 -> drop -> back east UNDER the platform. The
  WR's trough pump is only the FAST way onto that ledge; the slow way is a climb.
* The climb (bursts from the archived corner state): 1 in 16,384 random 8 s bursts - a duck-jump
  onto the edge of the west conveyor step at y ~-2945 (the trigger_push above the conveyors
  pushes down, but not at their north edge), then west along the step tops onto the ledge.
  175 scripted single/double jump and duck-jump patterns all failed. A ~1e-4 event for random
  keys is why PPO never found it; the ratchet pays the moment it is on the ledge (d falls west).
* Stopped at 3.2% after 8 min (487 cells): the next obstacle is below the platform.
* NEW explore_phase1 --view-velocity: explore in the trainer's view space (absolute yaw target in
  the velocity frame, off_warp prior), so a held burst is a constant air strafe, not a spin.
  Run 2: 64 u cells (so the climb is split into archived steps), 8 speed bins, 30 min.
* NEW tools/archive_to_spawn.py: archive rows ahead of the start -> --spawn-states.
* Plan (arm 5): the recipe + --time-pen 0 + --spawn-states <archive> (Go-Explore from the map
  start: champion-free, no demo; CLAUDE.md 0b lists archives / return-then-explore and planning
  in the simulator as generic), warm from the sidistic transfer at step 0.
