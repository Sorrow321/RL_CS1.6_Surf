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

### 2026-10-23 16:58 UTC - arm 5 (Go-Explore spawn states) and the SECOND gate (the pocket)

* arm 5 pbr1b2_ge5 (sidistic transfer at step 0 + recipe + --time-pen 0 + --spawn-states: 650
  Go-Explore archive rows ahead of the start, 30% of the pool): evals 0.402 / 0.421 (0.5M / 76M).
  The reservoir jumped to 3.2% within 5M steps (from the spawn rows) and stayed there; the greedy
  line stays at the NW corner jumping at y -2850..-2910 (the climb is at y ~-2945). Stopped 115M.
* Go-Explore run 4, rooted at the archive's 64 deepest states (3.0-3.2%): 360 cells, ~100 deaths
  in 1.6M runs - the deep cells are an ENCLOSED POCKET under the platform (floor z ~3300). The
  field counts it 3.2% because free space continues above it. The WR passes through it only
  because it lands there at ~1,000 u/s off the ledge drop and surfs straight up a ramp east
  (z 3317 -> 3522), curves north and west along an upper level at z ~3500, and drops into the
  lower chamber at x ~200, z ~3000 (11 s, 4.8%). A random walker drops in at ~600 u/s and is
  stuck. So the start holds TWO hidden gates: the corner climb (or the trough pump), then speed
  carried off the ledge into the pocket's ramp.
* NEW explore_phase1 --dump-paths (the burst states that first reached each new cell; a precise
  move happens inside one cell, and the archive keeps one state per cell) and archive_to_spawn
  --paths / --bin-width / --per-bin (rows balanced along field progress: a uniform draw over an
  archive starves narrow passages). Run 3 (64 u, view-velocity, paths): 3,654 cells, 58,187 path
  states, best 3.84%; balanced file 6,371 rows (530 at the climb, 1,208 on the ledge).
* arm 6 pbr1b2_ge6 launched: arm 5's 109M weights + recipe + --time-pen 0 + --spawn-states
  runs/pbr1b2_ge3/spawn_bal.npy (frac 0.3). Box 57383391 (3090, 0.375 $/h, deadline 19:01 UTC).

### 2026-10-23 17:33 UTC - arm 6 null; the climb is a 9 u exploit; arm 7 on the dense chain

* arm 6 pbr1b2_ge6 (arm 5's 109M + balanced Go-Explore spawns, frac 0.3): 0.419 / 0.413 (110M /
  185M). Bench (record_ckpt --spawn-states, arm 6 weights at ~205M): from mid-climb states 0/5
  reach the ledge, from corner-floor states 0/5. The spawn mix had put the reservoir's frontier
  band in the POCKET (3.6%), so the recipe's frontier machinery (novel / front-perturb / greedy
  rows) trained the dead end, not the climb the eval needs first.
* HARNESS-ADJACENT LESSON: a second CPU job on the box (Go-Explore at 10 threads, nice 5, or a
  record_ckpt bench) cut the trainer from ~150k to 30-45k steps/s (OMP pools oversubscribed).
  Search and train SEQUENTIALLY on one box.
* The climb, decoded: the conveyors at the platform's west edge sit under trigger_push volumes
  that push DOWN at 2,500 u/s; a duck-jump at y ~-2939 lands the hull on conveyor 2's north edge
  with ~9 u of overlap just outside conveyor 1's push zone, then a walk west along that edge onto
  the ledge. The map's push-down traps exist to stop exactly this; the intended way off is the
  pump (the WR). A 9 u tightrope is a poor thing to ask PPO to find.
* Go-Explore run 7 (64 u x 16 bins of |v| - NEW --cell-speed-3d - from the start): no deeper than
  the pocket. Energy reached: trough h <= 3,899, over the platform <= 3,964, ledge <= 3,827, pocket
  <= 3,894; the WR carries ~3,950-3,980 into the pocket and leaves it at ~3,920 up the east ramp:
  the pocket gate is ~+100 u of energy brought in from upstream.
* The DENSE CHAIN (runs/pbr1b2_probe/dense_chain.py): for the deepest cell, the burst states
  that first reached each cell on its parent chain, concatenated - 787 states, one per decision,
  root -> 3.84%. It climbs on the SOUTH side ((778,-3067) -> (629,-3111, z 3716)), runs a south
  ledge west to x ~150, drops at x ~30-50 and comes back east into the deepest region south of
  the pocket. NEW explore_phase1 --dump-paths now also saves the decision actions and views.
* arm 7 pbr1b2_ge7 launched 17:32: arm 6's 208M + recipe + --time-pen 0 + --spawn-states
  chain_dense.npy at frac 0.5 (the spine-testbed construction on a self-found line).

### 2026-10-23 17:47 UTC - arm 7 null (uniform spine); arm 8 = Go-Explore phase 2 (backward curriculum)

* arm 7 pbr1b2_ge7 (arm 6's 208M + the dense chain as --spawn-states at frac 0.5, the
  spine-testbed construction): 0.402 / 0.411 (209M / 284M), unchanged at 323M. A uniform draw
  over the whole chain spreads the practice; the precise climb gets a ~4% share and is not
  learned in 115M steps.
* NEW --demo-reach R (train_fast; surfgym.respawn.demo_reach_wins; default off, not dumped):
  Salimans-Chen's curriculum advances on the finish rate of its frontier window, so it could not
  run on a partial spine. With R, a spine episode is a win when its lowest field d comes within
  R u of the spine's last row (or it finishes). The window then starts at the chain's end and
  only moves earlier once its band succeeds 20% of the time - the training piles up right before
  the hardest move instead of spreading over the chain (Go-Explore's phase 2, Ecoffet et al.).
* --demo-file is not wired with --respawn-novel / --respawn-greedy / --int-real-only 1; on a
  resume the restored ones are turned off explicitly: --no-respawn-novel --int-real-only 0
  --respawn-greedy 0 --respawn-greedy-stuck 0 --respawn-backward-mix 0 --respawn-front-perturb 0
  --spawn-states ''.
* arm 8 pbr1b2_ge8: arm 7's latest + recipe + --time-pen 0 + the switches above + --demo-file
  runs/pbr1b2_ge3/chain_dense.npy --demo-reach 128 --demo-grow 8 --demo-window 16
  --demo-front-frac 0.67.

### 2026-10-23 18:18 UTC - arms 8/9: the backward curriculum works on the easy part, stalls at a pocket move

* arm 8 pbr1b2_ge8 (--demo-grow 8): tau 786 -> 682 in 136M steps, every band won 98.7-99.9%
  (a spine episode from the chain's end region reaches the win line almost always), but ~10M
  steps per 8-state advance: far too slow to reach the climb (rows ~159-186) in budget. Evals
  0.402 / 0.417 (the greedy line from the true start is only trained once tau reaches 0).
* arm 9 pbr1b2_ge9 (--demo-grow 32, from arm 8's 466M): tau 786 -> 594 fast, then OSCILLATES:
  band [594,609] 0.0% success (1,600 eps), band [626,...] 96.8-97.6%. Rows 589-620 are a move
  inside the pocket ((653,-3030,3441) -> (864,-3165,3476), 2.65% -> 3.15%) that Go-Explore's
  random bursts made once and the policy cannot repeat from those states. Evals 0.427 / 0.405.
* The chain is STITCHED, not continuous: e.g. rows 162 -> 163 jump from a burst rising at z 3685
  to another burst's stored state ducked at z 3751 (a cell's stored state is replaced when a
  shorter route is found; the dumped path is the first visit's). A curriculum only resets into
  the rows, so that is sound for spawns; it is not an open-loop replay.
* arm 10 pbr1b2_ge10: arm 9's 617.6M + the spine cut at row 593 (2.7%, chain_dense_593.npy):
  the curriculum can now walk back through the drop, the ledge and the climb to the true start.

### 2026-10-23 18:41 UTC - arm 10 stalls at the next stitch; the warm lineage degraded; arm 11

* arm 10 pbr1b2_ge10 (spine rows 0..593, from arm 9's 617.6M): tau 593 -> 433 fast, then the same
  oscillation one segment earlier: band [433,448] 0.0%, band [465,...] 100% - rows ~434-465 are
  the drop off the south ledge ((29,-3045,3490) -> (153,-2997,3591)), another random-burst move
  the policy does not repeat. Evals 0.413 / 0.406 / 0.417.
* The chained warm runs DEGRADED the policy: the yaw sigma rose 0.15 (sidistic) -> 0.4 (arm 3)
  -> 0.75 (arm 4) -> 1.18 (arm 9) -> 2.72 (arm 10, the cap: a random view), and arm 10's training
  episodes were 100% stall kills. With --time-pen 0 and dead-end spawns the reward barely
  depends on the view, so ent 0.005 wins. Lesson: restart from clean weights, lower ent.
* LESSON on Go-Explore chains as curricula: each random-burst move the policy cannot repeat is a
  wall the backward curriculum oscillates at (0% vs ~100% on the two sides). A chain built from
  the POLICY's own proposals (beam search) would not have them; a random-burst chain needs its
  unrepeatable moves cut out or the spine truncated before them.
* arm 11 pbr1b2_ge11: the clean sidistic transfer (step 0) + recipe + --time-pen 0 --ent 0.002 +
  the spine cut at row 378 (start -> corner climb -> south ledge, 1.44%), --demo-grow 16.

### 2026-10-23 19:03 UTC - arm 11 -> arm 12 (continuation)

* arm 11 pbr1b2_ge11 (clean sidistic transfer, --ent 0.002, spine rows 0..378): the yaw sigma
  stayed at 0.13 (ent 0.002 fixed the drift); tau 378 -> 282 in 217M steps (bands 61-100%), the
  reservoir reached 3.2% (the policy carries on past the spine's end by itself). Evals 0.396 /
  0.416 / 0.428 (tau far from 0: the true start is still only 10% of the pool). ~33M steps per
  16-state advance - spine episodes run until the stall kill (~18 s), so few episodes end per
  iteration.
* NEW --demo-tau-start N (DemoCurriculum tau0; tau is not checkpointed).
* arm 12 pbr1b2_ge12: arm 11's 217M weights, same flags + --demo-tau-start 282, budget 1.5e9.
  Box extended to 23:01 UTC.

### 2026-10-23 19:34 UTC - arm 12 stalled on a chain bug (fixed); arm 13

* arm 12 pbr1b2_ge12 (continuation at tau 282): the curriculum did NOT MOVE in 280M steps - band
  [282,297] logged 0 episodes. Cause: my dense chain REPEATED segments (a burst that enters
  several chain cells has its prefix in each of their dumped paths; 42 duplicated origins in 379
  rows), and DemoCurriculum.match() attributes a state to the LAST row with its origin, so the
  band's spawns were counted at rows 314-329. Evals 0.405 / 0.402 / 0.401 / 0.402.
* NEW tools/archive_chain.py (dense root -> cell chain, each state kept at its first occurrence;
  test). The deduplicated spine: 379 -> 333 rows (old 282 -> 269, the climb old 155-186 -> 143-174).
  Arms 8-10's 0% bands did log episodes, so those stalls were real; arm 12's was the bug.
* arm 13 pbr1b2_ge13: arm 12's 508M weights + chain_dedup_378.npy + --demo-tau-start 269.

### 2026-10-23 20:06 UTC - BREAKTHROUGH: 3.234, the first greedy line off the start platform

* arm 13 pbr1b2_ge13 (deduplicated spine, --demo-tau-start 269): tau 269 -> 141 in ~90M steps
  (bands 68-99%), then the climb's run-up band [125,140] (rows of the duck-jump onto the
  conveyor edge) oscillates with the band after it (92%): its win rate at each advance attempt
  rose 20.7% -> 25.0% -> 30.6% -> 39.4% over ~200M steps - PPO learning the precise move from
  concentrated practice. Greedy evals FROM THE TRUE START: 0.421 / 0.406 / 0.404 (508-660M), then
  **3.234 at 735M** (2/9 episodes at 3.22-3.23%) and **3.231 at 810M (6/9 episodes at 3.23%)**.
  Every deep episode stops at the same place, (808,-3286,3375): the south part of the pocket
  under the platform. The policy does the corner climb, the ledge, the drop and the pocket entry
  on its own, beyond the spine's end (1.44%).
* Snapshots pulled for the handoff: ckpt_708313088.pt (27M before the 735M eval),
  ckpt_808452096.pt (2M before the 810M eval), ckpt_latest.pt.
* HARNESS ISSUE (box.py pull --snapshot): the trainer names milestones ZERO-PADDED
  (ckpt_0708313088.pt) and box.py builds f"ckpt_{s}.pt" from the int (ckpt_708313088.pt), finds
  no file, and skips it SILENTLY. Workaround used: on the box `ln ckpt_0708313088.pt
  ckpt_708313088.pt`, then `box.py pull --snapshot 708313088` (md5-verified). The orchestrator's
  milestone regex accepts the unpadded local name.
* What made it work, in order: (1) Go-Explore phase 1 from the map start finds the route
  (random held-key bursts, --view-velocity, 64 u x speed cells); (2) the dense chain with each
  state once (tools/archive_chain.py); (3) Salimans-Chen's backward curriculum turned into
  Go-Explore phase 2 by --demo-reach (win = reach the spine's end region); (4) the spine cut
  before moves the policy cannot repeat (a band at 0% vs ~100% next to it marks one); (5)
  clean weights + --ent 0.002 (the yaw sigma drifted to its cap in the chained warm runs);
  (6) --time-pen 0 (the platform's kill strips make suicide the attractor otherwise).

### 2026-10-23 20:14 UTC - arm 13 stopped at tau 45 (past the climb); Go-Explore from the pocket

* arm 13's curriculum went 125 -> 45 between ~810M and ~890M: the climb's run-up is learned and
  the window was walking the platform back toward the true start (row 0) when it was stopped
  (its ckpt_latest + --demo-tau-start 45 continues it). The spines are now in the repo:
  autoresearch/analysis/pb_chain/ (chain_dedup_378.npy = arm 13's; chain_dedup_full.npy = 714
  rows to 3.84%) with a README (provenance; how to resume without the file: --demo-file '').
* The score only moves with deeper progress: every deep greedy episode stops at (808,-3286,3375),
  the south part of the pocket. Go-Explore run ge14: rooted at 885 of the policy's OWN pocket
  states (every 20th row with progress >= 2.9% of the 810M eval), 64 u cells x 16 bins of |v|,
  --view-velocity 1, --dump-paths 1, 10 min at full CPU (training paused).

### 2026-10-23 20:26 UTC - 3.235 with median 3.2% (the route is reliable); arm 14 explores the pocket

* arm 14 pbr1b2_ge14 (arm 13's final 853.0M weights; the curriculum OFF, the recipe's frontier
  machinery back ON - novel / front-perturb / greedy rows / backward-mix - plus --respawn-speed 1.0
  1.5, 15% --spawn-states from the 333-row spine, --time-pen 0 --ent 0.002, --demo-file ''
  --demo-front-frac 0 --demo-grow 0 to switch the restored curriculum off): evals 3.235 (853.5M,
  one iteration after the resume: median 3.2%) and 3.234 (929M, median 3.2%). Best = 3.235;
  arm 13's 853.0M weights copied into the run as ckpt_853016576.pt for the handoff (pulled).
* The pocket frontier: front-perturb rows from 27 frontier states (d <= 55,311), novel ceiling
  1,896 u, no progress past 3.24% yet at 978M; yaw sigma drifting up again (0.13 -> 0.46).
* Go-Explore ge14 rooted at 885 of the policy's own pocket states (64 u x 16 |v| bins): 684
  cells, ~1,400 deaths in 1.3M runs, best = the roots: the pocket is enclosed for random bursts
  (as run 4). The WR leaves it east up a ramp at ~1,000 u/s, carried in from the north-ledge drop;
  the policy enters off the south ledge at ~650 u/s and stops at (808,-3286,3375), a local field
  minimum ~200 u BELOW the WR's d through the pocket (55,300 vs ~55,500): the next gate is a
  field dip plus an energy deficit, the sidistic recipe's own last gate.

### 2026-10-23 20:29 UTC - the pocket gate is an ENERGY deficit; arm 15 (plain energy potential)

* Energy trace of a deep greedy episode (929M eval, h = z + |v|^2/1600): the policy WALKS the
  south ledge at 60-140 u/s (h ~3,725; the WR runs its ledge at ~600 u/s, h ~3,950), then
  CRASH-LANDS three times on the way down (3,740 -> 3,532, 3,634 -> 3,503, 3,522 -> 3,426) and
  ends at h ~3,407 in the pocket. The WR keeps h ~3,980 into the pocket by landing on the slope
  and leaves up the east ramp (needs ~3,920+). Deficit: ~250 u from walking + ~300 u lost to flat
  landings.
* arm 15 pbr1b2_ge15: arm 14's 994M weights + --energy-pot-front 0 --energy-pot-front-bonus 0
  (the recipe's K 3 energy potential paid/charged on every call: a gained strafe speed is paid,
  a crash landing is charged ~-1 per 300 u lost) with everything else as arm 14. Generic: a
  physical invariant, no map constant; with --time-pen 0 arm 3 showed no suicide.

### 2026-10-23 20:52 UTC - arm 15 (plain energy pot) and the policy-guided search at the pocket; arm 16

* arm 15 pbr1b2_ge15 (994M -> 1,160M): evals 3.232 / 3.232 / 3.231 (median 3.2% each), energy
  term active (+860/-1,033 u of h per episode), frontier unchanged (reservoir deepest 3.235%).
* beam_tas --score rec_e from the TRUE START with arm 15's weights (4096 lineages, 2 s
  generations, 60 s): the policy's own proposals now run the whole route. With --rec-quant 1024
  the population parked at the drop landing (d ~56,170 shares a bin with the pocket's ~55,300 and
  has more energy there - coarse bins make the energy tie-break myopic); with --rec-quant 256 the
  whole population was in the pocket by 12 s and then stuck: h max 3,453 for 40 s. Neither the
  policy's proposals nor random bursts ADD energy in the pocket; it has to be brought in.
* arm 16 pbr1b2_ge16: arm 15's 1,160M weights, same flags + --respawn-front-perturb-speed 1.0 3.0
  (the frontier-band rows' speed scale, a generic width): the agent experiences the pocket with
  the speed the exit needs, so the critic can learn what speed is worth there, while the plain
  energy potential pays bringing it in. Box extended to the round's end.

### 2026-10-23 21:18 UTC - arm 16 at 1.39B; the pump route is still unfound

* arm 16 pbr1b2_ge16 (front-perturb speed x1.0-3.0 + plain energy pot): evals 3.230 / 3.233 /
  3.235 / 3.231 (median 3.2%); training frontier inching: front-perturb best 1,850 -> 1,874 u,
  novel ceiling 1,906 u (3.34%). Kept running (cheap) as the pocket probe.
* The highest-energy states Go-Explore found near the platform (run 7, h up to 3,964) are
  ARTIFACTS: all have vz -450..-640 over the west conveyors - the trigger_push's 2,500 u/s
  impulse, not usable energy. No archived state flies west over the platform at speed (the WR's
  pump exit), so random bursts never found the pump; rooting there would not help.

### Recommendations for the next round (pbr1b2's view, 21:18 UTC)

1. Start from the handed-on checkpoint (3.235, median 3.2%); a resume needs `--demo-file ''`
   (or the spine file from autoresearch/analysis/pb_chain/) - see that README.
2. Keep --time-pen 0 and --ent 0.002 on this map: the time penalty made suicide (the platform's
   kill strips, 1.4 s) the attractor in four arms; ent 0.005 drove the yaw sigma to its cap
   whenever the reward stopped depending on the view.
3. The next gate is the pocket: the policy arrives with h ~3,400 (walks the south ledge at
   ~100 u/s and crash-lands three times); the exit ramp needs ~3,920+. Ideas not yet tried: a
   Go-Explore cell key on energy that the PUSH TRIGGERS cannot inflate (e.g. horizontal speed
   only on the ledge and pocket, or excluding trigger_push volumes), rooted at the policy's
   ledge states; a backward curriculum over a spine that BRINGS energy (a north-ledge route);
   energy-aware beam search ranking (record, then h) with the selection delayed until after the
   landing.
4. The method that worked (Go-Explore phase 1 -> dense deduplicated chain -> --demo-reach
   backward curriculum, spine cut before unrepeatable moves) is generic: nothing in it reads the
   map beyond its goal field; it is how to cross the NEXT hidden gate too, once Go-Explore can
   reach past it.

### 2026-10-23 21:52 UTC - the descent loses the energy (bench); box released

* arm 16 stopped at 1.68B: evals 3.227-3.235 (median 3.2%), one TIE at 3.235 (1,311.8M; its
  milestone ckpt_1260388352.pt pulled next to it so either tied eval hands on a checkpoint).
* Bench (record_ckpt --spawn-states, arm 16 weights): the policy spawned on the SOUTH LEDGE at
  450-650 u/s (h 3,840-4,000, WR-level ledge energy; ducked like the states it came from - a
  standing hull at a ducked height fails as stuck in 0.1 s, the first try's mistake) still ends
  at the same pocket point (3.23%) with only h 3,620-3,670 left down there: it sheds ~350 u on
  the descent whatever it starts with. So ledge speed alone is not enough: the descent must KEEP
  the energy (the WR flies over the floor the policy lands on and lands on the pocket's slope,
  off the NORTH ledge). That is the skill to find next.
* Box 57383391 released at 21:51 (no further idea had a credible chance in the 3.7 h left;
  the fleet was over its spending pace).
* config.json incumbent_recipe (the lineage key): --time-pen 0.001 -> 0 and --ent 0.002 added
  (default 0.005). Evidence: warm arms 1-3 (suicide attractor at 0.001) and arms 4-10 (yaw sigma
  0.15 -> 2.72 at ent 0.005); arms 11-16 ran with both. NOT tested from scratch on this map.
* Correction to the 21:18 note: the core applies a non-once trigger_push through basevelocity,
  so run 7's top-energy states (vz -450..-640 by the west conveyors) are not push-inflated - they
  are FALLING down the shaft there. A 3D-speed cell key credits a fall as energy; on this map
  the horizontal-speed key (explore_phase1's default) is the better energy proxy.
* Next idea I would build (needs a box; the fleet was over its pace from 21:52): a POLICY-GUIDED
  Go-Explore / MAP-Elites over (position cell x horizontal-speed bin) - the population search's
  proposals are the policy's own (so every found move is repeatable, unlike random bursts), and a
  cell keeps its best-ENERGY representative, so lineages are compared only against lineages at
  the same place: the rec_e beam ranking failed because it compares a lineage that already
  crash-landed in the pocket with one still on the ledge. The deepest high-energy chain it finds
  is the next --demo-reach spine.
* NEW beam_tas --elite-cells C (with --score rec_e; default off; unit-tested, NOT yet run on a
  box): the highest-energy lineage of every occupied C-u cell is an elite first. Suggested first
  run: the handed-on checkpoint, from the true start, --envs 4096 --resample-every 50
  --elite-frac 0.25 --score rec_e --rec-quant 256 --elite-cells 128 --max-ticks 6000
  --allow-nonfinisher --skip-gate --route-file none --dump-elite-states <npy>; look for elite
  rows in the pocket with h > ~3,900 and anything past 3.24%.

### 2026-10-23 22:56 UTC - --elite-cells run (5080 box 57433644, 7 min, released)

* beam_tas --score rec_e --rec-quant 256 --elite-cells 128 from the TRUE START, arm 16's
  1,260M weights as the proposal, 4096 lineages, 2 s generations, 60 s: the population is in the
  pocket by 12 s; per-cell energy elites raise the best pocket energy from 3,453 (plain rec_e) to
  3,635 - far below the ~3,920 the exit needs - and nothing passes 3.23%. The pocket has TWO
  mirror stop points, (808,-3289,3377) and (810,-2711,3377) (the pocket is symmetric about
  y ~-3000). Verdict: the policy's own proposals contain no energy-keeping descent, so no
  selection rule over them finds one; the next round needs a proposal that can (random
  view-velocity bursts did not either) or a new skill learned first (ledge speed + slope landing).
