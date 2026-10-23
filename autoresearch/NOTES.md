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

## pbr1b3 (round 1), 2026-10-23 14:49 UTC - the start gate, measured

Box 57382729: RTX 3090 on an AMD EPYC 7502 (64 threads, 2.5 GHz cap), 0.35 $/h. The incumbent
recipe trains at ~120k steps/s instantaneous there (trainer at 5200% CPU, GPU 10% busy).

* **The sidistic finisher's weights zero-shot** (`record_ckpt.py --map surf_princessburglar_b26
  --maxvel 3500`, CPU, greedy): walk ~200 u west-south-west and press into the spawn room's SW
  corner (650, -3155, 3654) for 84 s; best progress 0.32% (d 56,968 of 57,149). The scratch
  baseline's first eval stops at the same corner (0.3%, score 0.34).
* **Why (analysis of the WR line, CLAUDE.md section 0 - analysis only):** the spawn floor is at
  hull-centre z 3652 in a room x 650-1150, y -3100..-2900. Its west wall (x ~600) has a WINDOW
  at the north side (y ~ -2884, hull z ~3700-3760), i.e. 50-110 u above the floor; the north
  ledge is +87 u, the south ledge +132 u - all above a standing jump. The record does not walk
  out: it drops EAST into a halfpipe (x 1180-2400, floor z ~3360), pumps once (east wall up to
  z 3654, back down, up the west slope to z 3763 at 4.8 s), flies west over the room at z ~3720
  and ~590 u/s, through the window, and drops at x ~50. Along its line the no-corner-cut field
  RISES by up to 1,520 u for the first 4.8 s (d 57,149 -> 58,661 -> 56,459 at 5.7 s); after that
  the field falls monotonically along the record to the finish (no other rise > 300 u).
* So the first gate is an ENERGY gate behind a potential plateau: the field (dynamics-free)
  points west/up into the wall; the way is "go away, gain energy, come back high".
* A weights-only warm start of sid_ref was prepared (policy + optimizer, steps 0, config map
  set to princess and maxvel 3500; no reservoir / greedy rows / novelty counts / unstuck clock,
  all of which would otherwise be restored with sidistic coordinates - the greedy rows and the
  unstuck best are NOT map-keyed in the trainer's resume path).

### pbr1b3, 2026-10-23 14:56 UTC - the scratch baseline cannot even run here; two arms side by side

* `pbr1b3_base` (incumbent recipe from scratch, 102M steps, then stopped): greedy frontier 215 u
  (0.5M) -> 250 u (76M), score 0.34 -> 0.408; greedy peak speed 98 -> 121 u/s (it never runs,
  let alone pumps). Training: novel ceiling 432 u, 21 visited 128-u cells in 53M steps.
* **The yaw (view) sigma hits the clamp**: `sig` 0.30 at 1M -> 0.95 (53M) -> 1.80 (80M) ->
  2.72 = e^LOG_STD_MAX (100M). The start room pays almost nothing (the ratchet's whole budget
  there is ~250 u = 0.44 reward; novelty at 0.25 is spent in 21 cells), so the entropy bonus
  alone drives the yaw head to its ceiling; under the velocity-frame absolute view a
  saturated tanh(N(mu, 2.7)) is a bang-bang look direction, i.e. random movement. On a map
  whose start pays progress at once (a ramp, a drop) this never shows; here the first dense
  signal has to come from somewhere else.
* `box.py launch --env KEY=VALUE` added on this branch (Harness issue below) so two arms can
  share the box at OMP_NUM_THREADS=15 each (the trainer uses ~52 of 61 CPUs at omp 30 with the
  GPU 10% busy, so two half-width trainers should give more total steps than one).
* Arms now running together (launched 14:53-14:56): `pbr1b3_sidwarm` = the sidistic finisher's weights
  (weights + optimizer only, see above) on the recipe as restored from its config + --map
  princess --maxvel 3500; its yaw sigma starts at 0.077. `pbr1b3_nov4` = the recipe from scratch
  with count-based novelty x4 (`--int-coef 1.0`, the uf2 evidence: novelty at 4-10x was the
  only champion-free mechanism that took that map's start dip).

### Harness issue (pbr1b3)

`box.py launch` cannot pass environment variables to run_arm.sh / the trainer, so two arms
cannot share one box without oversubscribing (each trainer defaults to omp = half the CPUs).
This branch adds `--env KEY=VALUE` (repeatable; upper-case keys; refuses the launch-mode keys
SELF_STATES/CKPT/ARM_RESUME/SCRATCH). Nothing else changes in box.py.
