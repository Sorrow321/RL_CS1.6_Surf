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

### pbr1b3, 2026-10-23 15:07 UTC - two trainers on one box do NOT work; omp 15 beats omp 30 alone

* Two trainers side by side at OMP_NUM_THREADS=15 each (61-CPU quota, EPYC 7502): load
  average 97, ~3,200% CPU each, and NEITHER advanced a single 524k-step iteration in 40 s; the
  first greedy eval took 8 min instead of ~1. Spin-waiting thread teams (OMP + numba's
  parallel kernels default to every core) fight each other. One trainer per box.
* Alone, the same trainer at OMP_NUM_THREADS=15 runs **136k steps/s** instantaneous, against
  ~122k at the default 30 on this box: half-the-cores is too many on a 64-thread Zen 2 part.
* `pbr1b3_sidwarm` (warm start) was stopped after its first eval: frontier 237 u, peak 235 u/s
  (it walks; the scratch policy at 76M peaked at 121 u/s), patience-cut 8/9, score 0.410. Its
  yaw sigma started at 0.078 (a converged policy). Not continued: one box, and 4x novelty from
  scratch has the better prior for a start gate.
* `pbr1b3_nov4` (recipe + `--int-coef 1.0`, scratch) continues alone (launched 14:56).

### pbr1b3, 2026-10-23 15:34 UTC - novelty x4 is a null; the yaw entropy bonus is mis-specified

* `pbr1b3_nov4` (recipe + `--int-coef 1.0`, scratch, 152M steps, stopped): frontier 175 u (0.5M)
  -> 248 u (76M) -> 208 u (152M), peak 131-132 u/s, ~25 visited 128-u cells, yaw sigma at the
  2.718 clamp by 125M - the baseline's curve exactly. Every greedy episode of both runs stays in
  the WEST half of the spawn room (x 649-936, mean speed ~50 u/s); none goes east to the door
  at x ~1150 that leads to the halfpipe. Score 0.403.
* **Root cause found in the code:** `logprob_entropy_view` paid the GAUSSIAN entropy of the
  view heads, 0.5 + 0.5 log 2pi + log sigma, which grows without bound; the core receives
  u = tanh(z), whose entropy peaks near sigma ~0.7 (mu = 0) and falls beyond it. Where the
  reward carries no signal the bonus alone pushes the yaw sigma to LOG_STD_MAX (2.718), and
  under `--view-absolute velocity` the warp maps |u| ~ 1 to a +-180 deg offset: the agent looks
  BACKWARDS from its velocity half the time and can never build speed. (The pitch head hit the
  same clamp on cannonball by 3-8B steps; that was fixed by zeroing its share, --pitch-entropy.)
* `--view-ent-squash 1` (commit b47f38c): the squashed action's entropy, H(z) + E[log(1 -
  tanh(z)^2)] with one reparameterised draw per row; the pitch share follows --pitch-entropy.
  A correction of the objective, right on every map (where the reward does carry a yaw signal
  it binds rarely). Tested: quadrature match, gradient on log sigma +1 -> negative at the clamp,
  per-row sigma, CPU smoke, and flag-off bit-identity against the base trainer (config,
  progress.csv, eval bytes, weights).
* `pbr1b3_enall` (recipe + `--energy-pot-front 0 --energy-pot-front-bonus 0`: the energy
  potential paid everywhere) runs since 15:28; next: the recipe + `--view-ent-squash 1`.
* Harness note: `box.py deploy` onto a box whose GPU is busy with a running trainer fails
  deploy_box.sh's GPU spec check ("below spec ... SW Power Cap") AFTER the git checkout: the
  code is deployed, restamp is skipped (harmless when the maps did not change). The GPU is
  fine (it passed the same check idle at 14:29); not blocklisted.

### pbr1b3, 2026-10-23 16:13 UTC - the start room is a trap box; Go-Explore with ENERGY cells gets out

* `pbr1b3_vesq` (recipe + `--view-ent-squash 1`, scratch, 150M, stopped): the yaw sigma
  plateaued at ~1.0 (0.69 @51M, 0.90 @76M, 1.03 @141M - the squashed entropy's optimum is
  sigma ~0.9-1.05 for |mu| <= 1) instead of 2.718, as designed; but the greedy line is the same
  (238-245 u, peak ~105 u/s, ~24 visited cells). Necessary, not sufficient: keep it on.
* **What the room really is** (the core's own triggers, analysis): a `trigger_push` straight
  DOWN at 2,500 u/s at the floor's west end (x 648-718, y -3054..-2946) into teleport volumes
  below - the SW corner every policy parks in is beside it, and standing in it kills in 4
  ticks; a `trigger_teleport` strip above the north ledge's east part (x 865-1151, z ~3700);
  a door east into the halfpipe only below z ~3720 (y ~-3000, 64 u wide); ledges at +87 u
  (north) / +132 u (south) above the floor. Random actions die in <4 s 88% of the time.
* **Go-Explore phase 1** (`tools/explore_phase1.py`, random actions, 95% repeat, state restore,
  1/sqrt(seen+1); on the box at ~1.6M ticks/s):
  - position-only 128 u cells (local probe): 160 cells and stuck in the room;
  - 128 u x 15 speed buckets (100 u/s): stuck at ~1,150 cells for 3 min, then a burst climbed
    out at the SW corner (onto z 3739 at x ~600, +87 u) and the archive reached goalc d 55,319
    = **3.2% of the field at 10.5 s** (west along the raised section, drop at x ~100, east
    through the lower level at 600-700 u/s, north) - the record's own lower-level line after
    its north window;
  - 96 u x **24 u energy-height bins** (new `--cell-energy`): the same depth in **0.8 min**.
  The record pumps for its energy; the search ratchets it bin by bin, which the speed bucket
  (100 u/s ~ 150 u of height at 800 u/s) could not.
* Next: Go-Explore phase 2 - train the recipe (+ squash) with `--spawn-states` = the parent
  chains of the archive's deepest cells (`tools/gx_spawns.py`), re-written while the search
  goes on (the trainer re-reads the file on mtime). Provenance: the search's own states, rooted
  at the true start, random actions; no demo, no record.

### pbr1b3, 2026-10-23 16:26 UTC - the search finds the record's window by itself; phase 2 launched

* The energy-keyed search (`gx_pb4`: 96 u cells x 24 u energy bins, seed 1, 12 min on the box
  at ~1.3M ticks/s, 9,272 cells) reached **3.82%** of the goalc field: spawn -> NW corner ->
  onto the north ledge (+62 u, a duck-jump height) -> west along it at z ~3714-3728 through the
  window -> drop at x ~100 -> the lower level east at ~700 u/s -> south -> back up toward z
  ~3450 on both the south (620,-3320) and north (629,-2637) sides. That is the record's own line
  through the window (the record goes on west from (420,-2636,3478)), found with random actions.
  Saved: $AR_HOME/runs/pbr1b3/gx_pb4/ (archive, log, the 184-state spawn file).
* **A search and a trainer cannot share the box either**: the phase-2 trainer next to the
  search (20 OMP threads) sat in its first eval for 5 min (`pbr1b3_gx1`, killed; its record
  gate failed cleanly, it does not count). Time-share instead (one at a time).
* `pbr1b3_gx2` (16:23): recipe + `--view-ent-squash 1` + `--spawn-states runs/gx_spawns.npy
  --spawn-states-frac 0.5`, scratch, alone, OMP 15. Provenance (passed to launch --self-states):
  the search's own states, rooted at the true start, random actions; no demo, no record, no
  policy. Evals are from the true start as always.

### pbr1b3, 2026-10-23 17:20 UTC - the start gate is climb + TIGHTROPE + drop; phase 2 is slow to learn it

* `pbr1b3_gx2` at 350M (recipe + squash + 50% spawns from the search's chains): training
  episodes reach the search frontier (reservoir min-depth 96.4% = 3.6% progress), but every
  greedy eval from the true start stops at ~250 u (0.5M ... 302M). At 151M, 2 of 9 greedy
  episodes went north onto the NW step (z ~3722) and stopped.
* Diagnostics (record_ckpt --spawn-states from the search's own states, on the box):
  - from the 11 pre-climb states (floor / mid-air ducked at the sill): 0/16 climbs (greedy and
    stochastic); the policy hops onto the room's west step (z 3714, x ~700-730) and stands there,
    or reverses its westward motion;
  - from 16 sill / window states: 7/8 greedy episodes DIE within 13-62 ticks just past the
    window, falling from z ~3724 toward ~3705 at y -2900..-2917.
* Why (the core, analysis): the whole west section (x 200-600) has a KILL LAYER just above its
  floor - a ducked hull dies within 0-20 ticks even at rest anywhere there. The only safe line
  is a narrow ledge along the north wall, y ~-2884..-2892, ducked at z ~3714-3730, for ~500 u
  (the record slides it at ~590 u/s, y -2880..-2885). So the gate = duck-jump onto the sill ->
  a ~500 u tightrope ducked -> drop at x ~90 into the lower level. The map is symmetric about
  y ~-3005; the first search also found the south twin (600,-3127,3739).
* Spawn file re-weighted live (the trainer re-reads it): start / climb (progress < 300 u) x15,
  300-700 u x4, 700-1000 u x2, the rest x1, plus every fine-archive cell on the tightrope (288)
  and around the climb (259) x3 -> 2,910 rows: 45% start/climb, 25% tightrope, 30% deeper.
  A fine search (32 u x 24 u energy, 4 min, 118k cells, trainer paused with SIGSTOP/SIGCONT)
  supplied the dense states.
* 17:40 diagnostics with the 517M checkpoint: from tightrope states near its WEST end (x 87-171)
  the policy now survives, drops and crosses the lower level to 3.1-3.2% (1,745-1,835 u) - the
  downstream part is learned; from mid-tightrope / sill states it still dies in 15-80 ticks,
  drifting 10-15 u south off the ledge (the safe band is y ~-2896..-2881, ~15 u; 281/288 of the
  search's ledge states are mid-hop, ducked, many with southward velocity). Greedy evals 378M:
  155 u, 453M: 190 u (worse than the corner). Spawn file rebuilt: 70% climb / sill / tightrope
  (progress 120-800 u), 10% start room, 20% downstream.
* Harness note: box 57382729 (Thailand) uploads at ~0.022 MB/s to the agent host, with the
  trainer paused or not (network path, not CPU): a 63 MB checkpoint takes ~50 min to pull.
  Start checkpoint pulls long before a deadline.

### pbr1b3, 2026-10-23 17:51 UTC - hand-weighted spawns replaced by a generic reverse curriculum

* **Correction (CLAUDE.md 0b):** from 240M to 604M `pbr1b3_gx2` drew its spawn file with
  weights I set after looking at where the policy dies (progress bands x15 / x4 / x2, and the
  fine archive's cells inside a box around the climb and the tightrope). That is a map-specific
  spawn rule: those 360M steps are ANALYSIS, not a recipe result. Greedy evals over them: 155,
  190, 168 u (worse than the corner's ~250).
* **The generic replacement** (commit bb7fe35): `--spawn-states-curric G` (surfgym/ss_curric.py,
  Florensa et al. 2017): every --spawn-states row keeps decayed counts of the episodes started
  from it and of those whose ratchet record got G x d0 past the row's own field distance; rows
  are drawn with weight max(p (1 - p), 0.02). G = 0.01 x d0 on every map. Tested: weights,
  outcome scoring, counts kept across a re-read, draw frequencies, CPU smokes (32 episodes
  noted), flag-off bit-identity against the base trainer.
* Spawn set, chosen generically: the union of the parent chains of the 256 deepest cells of
  the fine search (32 u x 24 u energy) and the 64 deepest of the coarse one (96 u) - 1,061
  states, no region selection.
* `pbr1b3_gx3` (17:52): ARM_RESUME of gx2's 604.5M checkpoint (its own; md5 6d0395f5...) +
  `--spawn-states runs/gx_spawns_cur.npy --spawn-states-frac 0.5 --spawn-states-curric 0.01`
  (squash and the recipe restored from the checkpoint).
