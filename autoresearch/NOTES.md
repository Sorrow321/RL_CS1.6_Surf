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

## pbr1b4 (round 1) - 2026-10-23 14:39 UTC: start

* **The start, measured on the record line (ANALYSIS ONLY).** The goal field (goalc_32) falls
  almost monotonically along the WR line from 0.27 s to the finish: after the start the largest
  rise above the running minimum is ~200 u (at ~21.7 s). The one exception is the first ~6 s:
  the record runs EAST from the spawn into a half-pipe (d 57,149 -> 58,643, +1,500 u), turns,
  comes back ~100 u above the spawn and flies WEST over the roof of the chamber beside the
  spawn, dropping in at x ~ 0 (WR ~8.3 s). In the field the chamber is DOWNHILL from the spawn
  (d ~55-56.5k): the field already points the right way, and the half-pipe is the record's
  speed gain (~900 u/s into the chamber), not a hidden route. So the first gate should be
  "enter the chamber slow and survive it", not a deceptive dip.
* **New flag `--init-weights CKPT`** (default off; tests/python/test_init_weights.py): a
  FRESH run whose policy starts from CKPT's weights and nothing else (the optimizer, the step
  counter, every reservoir, the novelty table and the schedules start as scratch's). A plain
  --ckpt resume onto another map would also restore sidistic's reservoir, greedy-death states,
  novelty table and --unstuck reading. The sidistic finisher (sid_ref, 25.39B steps) loads
  strictly under this map's full recipe (37 tensors; checked locally on CPU).
* **Harness issue (market):** at 14:28-14:38 UTC no offer passed the filters: the cheapest
  allowed card was a 5080 at 0.667 $/h (cap 0.65), 4090s from 0.669 (cap 0.45), 5090s from
  0.936 (cap 0.60), two 3090s at 1.33+. `rent` was refused four times.

### 14:44 UTC - box 57384171 (RTX 5090, 0.535 $/h, Ryzen 9 7950X 32 thr, KR), run pbr1b4_iw

* The rent retry loop got a box at 14:42 (first attempt after the loop started). Deployed
  ar/pbr1b4 (b7c17fa) with sid_ref pushed as runs_ckpt.pt (md5 390bf508...).
* `pbr1b4_iw` = the incumbent recipe from scratch + `--init-weights runs_ckpt.pt` (the sidistic
  finisher, 25.39B steps; all 37 policy tensors loaded, everything else fresh), budget 3e9 with
  a 30-min review (no step-budget schedule in the recipe, so a promising run need not restart).
  First minutes: half the training episodes stall-killed, greedy eval 250 u (0.4%).

### 15:04 UTC - pbr1b4_iw: NULL, and THE START IS A TRAP (measured with the core's traces)

* `pbr1b4_iw` (recipe + --init-weights sid_ref) stopped at 345M steps: greedy best 250 u
  (0.4%, the FIRST eval, the sidistic weights almost untouched; score 0.403), then 0.0-0.1%;
  furthest training state 557 u; novelty paid 0.01/ep; T hit its 0.2 cap at ~250M; training
  episodes shrank to ~1.5 s (the policy learned to die: with the ratchet + time-pen 0.001 and
  no progress available, the fastest death is the best return).
* **The start, from the core (`SurfCore.trace`, standing hull):** the spawn platform (floor
  3616, origin 3653) is closed to the WEST by solid up to >= z 3800 at x <= 640 (every
  y -3150..-2870). The one gap west is a slot at y -3054..-2946 between two func_conveyor
  plates holding two trigger_push volumes (x 648-718, z 3657-3698, pitch 90 = straight DOWN,
  speed 2500): a death chute into the catch teleport. The exit is a hidden upper corridor on
  the NORTH side: at origin z 3740, y -2884 a hull trace runs west from x 860 to x 66 (a
  ledge at ~3700, ~85 u above the platform floor; a standing jump is ~45 u, a duck-jump ~63).
  The record gets there by running EAST into the half-pipe (x 1177-2396, y -3136..-2864,
  z 3296-3568), surfing it end to end and back (~6 s), and climbing out of its west end at
  ~530 u/s to z ~3749, then flying west along the corridor and dropping into the chamber at
  x ~0. The goal field (free-space BFS, no gravity) points UP into the corridor from the spawn
  and through the death chute: an ENERGY DEFICIT at step one, +1,500 u of d the wrong way.
* What the sidistic policy did here: the first eval (weights untouched) walked west and pressed
  against the x ~649 wall for 35-50 s; by 76M steps PPO had it running east into the half-pipe
  and falling off its sides (y -3181 / -2829, z ~3300) in 3-9 s; by 150M it died in 1.5 s by
  touching a teleport brush at (900, -3098, 3686).
* Generic reading: three recipe constants set late on sidistic work against a from-scratch
  trap start: the time penalty (dying is free, living costs), the 0.2 temperature cap
  ("temperature discovers" ran at cap 2 from scratch on sidistic), and energy priced only on
  entering the frontier band (the band is AT the spawn, so pumping speed in the half-pipe pays
  nothing).
* `pbr1b4_esc` launched ~15:03 UTC: recipe + --init-weights sid_ref + `--time-pen 0
  --unstuck-max 2 --energy-pot-front 0 --energy-pot-front-bonus 0` (energy paid everywhere).
  Read: does any episode reach the corridor (best > ~1,000 u = 1.75%)?

### 15:38 UTC - the start, solved by search: Go-Explore phase 1 passes the trap in 30 s

* `pbr1b4_esc` (+ --time-pen 0 --unstuck-max 2 --energy-pot-front 0): NULL at 567M steps, best
  eval 0.466 (0.4%). Episodes lengthened (1.5 s -> 9 s), T reached 1.83 and the yaw sigma blew
  up to 2.7, but the greedy line only learned to perch at (675, -2934, 3732) on the conveyor
  plate north of the death chute, bouncing for 30 s, ~40 u from the way out.
* **The way out in OUR core** (core traces; the record is analysis only): the corridor is a
  CRAWL SPACE here. Standing, its cross-section is z 3732-3740 x y -2910..-2886 (a slot a few u
  tall); DUCKED it is z 3716-3756 x y -2918..-2866. The record runs it STANDING at z ~3722 -
  inside our solid by ~10 u. That is systematic, not local: 805 of the record's 7,723 rows put
  its hull inside our geometry, in ~dozen segments along the whole course (6.7-8 s, 15, 17, 23,
  25, 29, 32, 49, 53, 60, 64 s), each needing a ~12 u lift to be free (the core's hull vs sloped
  faces). Harmless on open ramps, decisive in a tight slot: our start needs a duck the server's
  does not.
* **Go-Explore phase 1 (`tools/explore_phase1.py`, unchanged; cell 32, 100-decision bursts at
  act-every 4, 95% repeat) from the TRUE START** reached the corridor in 40 iterations (~30 s,
  local CPU, 0.12M ticks/s) and the chamber below it (best goalc d 55,315 = 3.21%) within a
  minute; on the box (4,096 envs, 28 threads) 3.06M ticks/s, the same plateau in 30 s, then
  ever-shorter chains (best route 7.07 s deep). It stalls at 3.2% (the chamber: a skill gate,
  where RL and the field should take over).
* **New tool `tools/archive_spawns.py`**: the archive's root -> best-goal-field cell chain as a
  STATE_DTYPE .npy for --spawn-states (no map constant: the field's own frontier, the
  tools/archive_route.py rule). Provenance: reward-free random bursts from the true start; no
  demo, no policy, no reward.
* `pbr1b4_ge` launched ~15:37 UTC = recipe + --init-weights sid_ref + `--time-pen 0
  --spawn-states runs/ge_pb1/chain.npy --spawn-states-frac 0.2` (73 chain states, 3.21%).
  Go-Explore's "robustify" half done by RL from states along the found route. Read: does the
  GREEDY line from the true start learn the crawl (best > 1.5%), and does RL push past 3.2%?

### 16:07 UTC - pbr1b4_ge: the chain teaches the tunnel but not the CRUX; spawn files swapped live

* pbr1b4_ge at 470M steps (uniform spawns over the 73-state chain): the reservoir's deepest
  state sits at the chain's end (mind 96.77% = 3.23%), but the GREEDY line from the true start
  is unchanged (0.4%, score <= 0.429): it now jumps at the west wall under the NORTH tunnel
  (649-653, -2905..-2923, z ~3690), ~20 u too low for the ducked window (z >= 3716), and never
  climbs onto the conveyor plate the route needs. 80%+ of training episodes are stall-killed.
* The route the search found (the box archive's chain, 73 states, 7.07 s to 3.21%): walk west,
  duck-jump onto the UPPER CONVEYOR PLATE (x 648-718, y -3054..-2946, top z 3681, inside the
  2,500 u/s down-push: the chain stands on it at origin 3696), jump west from the plate into the
  SOUTH tunnel (y ~ -3100, the mirror of the north one), crawl ducked to x ~150, drop into the
  chamber (a U-turn: east along its south side, north, back west), end at (811, -2716, 3365).
* **The crux, measured with the batched core and the searcher's own random bursts (95% repeat,
  2.4 s):** from chain state 13 (inside the tunnel) 255/512 bursts get through; from the plate
  states 8 / 10 / 11 / 12: 1/512, 0/512, 0/512, 2/512 (and 3/2048 from 8). The plate -> tunnel
  entry is a ~0.2% event; open-loop constant holds (19 yaws x 3 strafes x 5 jump times x duck)
  succeed 0/570. Note the chain is STITCHED (each row is the best state of its cell; its
  children may have been found from another state in the cell), so a stored row is not
  guaranteed to continue the way the chain does - another reason the crux rows are hard starts.
* --race-arc along the found route would NOT fix the crux: the greedy's north-wall spot projects
  onto the route's slot vertex (~170 u away, inside the 1,500 u corridor), so it already earns
  the local arc; the crux is a motor skill, not a deceptive coordinate.
* Spawn files swapped LIVE (the trainer re-reads --spawn-states on mtime change, every 20
  iterations): 15:55 UTC -> the trap part only (chain states 0-28; re-read at 345.5M), 16:07 UTC
  -> the crux only (states 2-13: floor -> plate -> tunnel mouth; ~575M). Backward-algorithm
  logic without the code: put the starts where the policy fails, ~17 starts per state per
  iteration instead of ~3.
* Also measured: Go-Explore with speed in the cell key (--cell-speed 4), rooted at the tunnel
  exit, plateaus at the same chamber cell in 5 min (random actions do not make speed; the
  chamber's way out is the record's upper path at z ~3500, ~150 u above the search's floor).

### 16:27 UTC - the yaw sigma hits the clamp on a stuck start; --yaw-sigma-max; pbr1b4_ys

* Box 57384171 extended to 19:19 UTC (orchestrator notice).
* pbr1b4_ge stopped at 923M steps: best 0.429, the greedy unchanged after both live swaps.
  **Diagnosis: the yaw head's sigma was at the global clamp (2.718)** - 0.08 at the sidistic
  weights, 0.9 by 345M in pbr1b4_iw, 2.7 by 567M in pbr1b4_esc, 2.718 by ~800M here. With the
  absolute view in the velocity frame, the movement direction follows the yaw, so every training
  rollout walked and jumped in a random direction: a 0.2% crux can never be practised that way.
  While a run is stuck the entropy bonus is the only force on that sigma (the pitch head had the
  same pathology; --pitch-entropy 0 and its log-0.5 cap fixed it).
* The archive also holds the NORTH tunnel (105 cells; south 107): its shallowest cell is 1.95 s
  deep - walk WNW to (758, -2944) at ~250 u/s, jump, duck in the air, fly into the mouth at
  (664, -2915, 3696), land ducked inside at (633, -2916, 3714). No conveyor plate. The greedy
  line already presses at that wall (slow, no running duck-jump). The field and the compass point
  into the north tunnel, so this route agrees with the shaping.
* **New flag `--yaw-sigma-max S`** (commit 425b849; tests/python/test_yaw_sigma_max.py, 4 pass
  incl. a CPU trainer run): the yaw head(s)' log sigma ceiling at log S instead of 1.0 under
  --view-absolute; projected after every optimizer step like the pitch cap; config key only when
  set; mirrored in record_ckpt and beam_tas. Generic: one constant on every map (0.5 = the
  pitch cap).
* `pbr1b4_ys` launched 16:26 UTC = recipe + --init-weights sid_ref + --time-pen 0
  + `--yaw-sigma-max 0.5 --spawn-states runs/ge_pb1/spawn_v2.npy --spawn-states-frac 0.3`
  (148 rows: the 25 crux states [south chain 2-13 + the north-tunnel chain] x3, + the 73-state
  chain). Read: does the greedy enter a tunnel (best > 1.5%)?

### 17:31 UTC - the crux measured properly (a counting bug corrected), entry bursts, spawn v4

* **CORRECTION (my bug, 16:15-17:20 UTC):** the batched-core probes counted a burst as "in the
  corridor" when the origin read x < 560 after the step - but an env that DIED that tick (any
  catch teleport) is autoreset to the map's info_player_start (511, 413, 2888), x = 511. So
  deaths counted as successes. Wrong numbers, now void: "2.6% of random bursts from the true
  start reach a tunnel", the time-before-entry ladder (70% / 41% / 14% / 8% / 5%), and the
  plate-state rates (1/512 etc.). The 17:21 spawn swap (5,519 "success-bundle" states) was in
  fact a bundle of states BEFORE DEATHS - replaced at 17:31. Test with `alive &= ~ended` FIRST.
* **Corrected (alive, x < 560, z > 3600, 3-4 s random held-key bursts):** from the true start
  0/4096; from the chain's plate/mouth states (south and north) 0/1024 each; from inside the
  south tunnel (chain 13) 260/1024. The stitched chain's crux rows are not on a successful
  trajectory (the archive found the entry from another state in those cells).
* **`tools/explore_phase1.py --goal-progress P`** (default off; self-test 99/99): an ALIVE burst
  P units down the goal field (the tool's goal_32) is a win, dumped as win_k.traj.jsonl /
  win_k.spine.npy - the first PHYSICALLY CONTINUOUS crossing of a gate, not only the finish.
  On goal_32 here: walls under the mouths 247-248 u, mouths 239, inside at x 600: 324, x 400:
  ~510. P 400 -> 20 wins in 6 iterations but all from cells already inside; P 265 -> 30 wins
  in 0.1 min with the real entries (e.g. north: 122 ticks from (695, -2929, 3680) to the mouth).
* The NORTH entry, from the bursts: stand on the conveyor plate's NW corner (650, -2940, origin
  3696 ducked / 3714 standing), hop NORTH-UP ducked into the slot (z 3719-3731, y -2929 ->
  -2923), crawl west. The greedy line jumps from the FLOOR 20-40 u north of that corner.
  Random continuations from states along the real bursts: north 0-3.6%, south mouth footholds
  (647, -3088, 3714 ducked on ground) 6-10%.
* Spawn file v4 (live in pbr1b4_ys at 991M, 17:31): the last 10 spine rows of all 50 wins
  (ge_gp1 P 400 + ge_gp2 P 265: the entry bursts and their stitched approach) x3 + chain[13:]
  (inside the tunnel -> the chamber). 1,560 rows, 30% of the spawn pool.

### 17:59 UTC - the backward curriculum is working one step at a time: the last step is learned

* Probe (record_ckpt --stochastic, 20 x 4 s episodes from the entry-burst states, pbr1b4_ys
  ckpt at ~1.40B): **11/20 enter the corridor alive and crawl to x 112-241** (most of the
  tunnel), from both mouths (e.g. (649, -2929, 3719), (648, -3087, 3716), (686, -3112, 3717)).
  The 9 failures all start ONE STEP EARLIER - on the plate edge / floor at z 3686-3696, before
  the hop up into the mouth (x 665-759). Random continuations from the same states: 1-10%. So
  RL has learned the final step from spawns and is one step short; the greedy line from the true
  start (still at the south wall, floor level, 0.4%) needs two more steps: floor -> plate, plate
  edge -> mouth.
* Spawn v5 live at 1.44B (17:59 UTC): the FULL spines of all 50 search wins (1,130 rows, 567 of
  them on the approach east of x 700) + the entry rows x2 + chain[13:] = 2,190 rows.

### 19:16 UTC - POLICY-GUIDED SEARCH FINDS THE WAY: lines from the true start to ~5% (expert iteration)

* RL alone did not chain the curriculum back: from the plate/floor approach states the policy
  went 0/20 into the corridor (18:35 probe; it steps off the plate and jumps from the floor,
  max z 3697). pbr1b4_ys2 (resume of ys at 1.89B, spawn v6 = approach states x4 + entry rows +
  chain tail, frac 0.5, --sil-coef 0.1 --sil-top 256 --front-temp 0): greedy still 0.4% at
  2.12B.
* **tools/beam_tas.py (policy-guided population search) from the TRUE START with the ys
  checkpoint (1.89B), `--objective progress --route-file self_route.npz`**: with the policy's own
  proposals only (keys-hold refused --eps/--macro-hold) the population stays at the wall (best
  arc 343 u). Patched beam_tas so --eps works under --keys-hold (uniform over the POLICY heads'
  bins). **With --eps 0.15: the replayed best line enters the south tunnel at ~17-19 s (20 s
  cap); with a 30 s cap, waves 2 and 3 (torch seeds 2, 3) reach arc 2,981 / 3,010 u (5.2-5.3%
  of the route) at ~27 s, min goalc d ~56,221 u**, replay exact. The population does what the
  policy alone cannot: it keeps the rare lineage that made the hop and clones it.
* `tools/self_route.py`: the route = the box archive's chain (73 states: start -> plate -> south
  tunnel -> chamber) + the goalc field's steepest descent from the chain's end to the finish
  basin (448 pts @ 128 u, 57,236 u, d_end 64). Champion-free: a reward-free search + the field.
* Next (running): tools/plan_to_bc.py over the 3 waves (24 kept lines) -> BC rows (policy-space
  targets) + spine, then a warm resume with --bc-file (expert iteration, the sidistic operator's
  exit_local loop) so the GREEDY line imitates the found line.

### 19:30 UTC - expert iteration launched: pbr1b4_bc (BC on the search lines + spine spawns)

* tools/plan_to_bc.py over the 3 search waves: **17,899 BC rows from 24 lines, 0 dropped**,
  best arc 3,010 u; spine = the best line's 2,970 per-tick states from the TRUE START (the
  final 31-tick fall trimmed) - a physically continuous line, unlike the Go-Explore chain. The
  distil is CPU-heavy (17 cores for ~20 min while a trainer shared the box; minutes alone).
* pbr1b4_ys2 stopped at 2.143B (best 0.429; greedy unchanged). `pbr1b4_bc` = resume of its
  checkpoint + `--bc-file runs/beam_pb3/bc.npz --bc-coef 0.5 --bc-coef-final 0 --bc-steps 1e9
  --bc-target dist --bc-value-coef 0.25 --spawn-states runs/beam_pb3/spine.npy
  --spawn-states-frac 0.3 --sil-coef 0 --sil-top 0 --front-temp 1 --front-band 0.01` (the
  recipe's frontier temperature back; SIL off - a resume restores --sil-top, which refuses
  --sil-coef 0 unless --sil-top 0 is passed too). BC NLL 47.9 -> 7.8 in the first iterations.
* Box 57384171 extended to 21:20 UTC (18:20).

### 19:47 UTC - BC fits the lines (99.9%) but the greedy does not reach them: distribution shift -> DAgger

* pbr1b4_bc (2.143B -> 2.435B, BC coef 0.5 -> 0.36): BC head-acc 0.999, joint-acc 0.997,
  view-mse 0.009 on the 17,899 planner rows - and the greedy evals unchanged (0.416-0.442).
  The greedy reaches the floor SOUTH of the conveyor plate by 1.2 s (714, -3078) and bounces at
  z 3652-3687 for 30 s; it never mounts the plate.
* The found lines climb the same way (waves 2 and 3, ~17.5-19.5 s): STAND on the plate's SW
  corner (649-654, -3059..-3066, origin 3714 = feet on the plate), jump-spam there (z 3721-
  3746), drift SW while high into the slot (630, -3088, 3742) ducking, crawl. But they WANDER
  ~16 s on the floor first (the population needed that long to stumble on the climb), so ~90%
  of the BC rows teach wandering and the climb is a handful of decisions whose preconditions the
  greedy never reproduces (plus the search used ONE spawn yaw; evals jitter it by 8 deg).
* Next: expert iteration done properly - more waves with --eps 0.25, --resample-every 15 (keep
  the first hop sooner), a different spawn seed per wave (the yaw jitter), 4 waves with the ys
  1.89B proposer + 2 waves with the BC checkpoint itself as the proposer (DAgger: the lines start
  where the greedy actually goes); distil per spawn and merge; resume BC from bc_latest.

### 20:16 UTC - expert-iteration round 2: more lines, three spawns, climb-weighted spawns (pbr1b4_bc2)

* Search waves (2048 envs, 30 s cap, progress objective along the self route): with the ys
  1.89B proposer, --eps 0.25 --resample-every 15, spawn seeds 1-4 -> 3 of 4 FAILED (arc 343;
  faster cloning culls lineages before they climb), seed 4 crossed at 15.65 s (arc 3,014 u).
  With the BC checkpoint (2.43B) as the proposer (DAgger), seeds 5-6 -> seed 5 crossed at
  20.7 s (arc 3,013 u), seed 6 failed. With the earlier --eps 0.15 --resample-every 25: 2 of 3
  crossed. So far ~4 of 10 waves find the way; resample 25 looks better than 15.
* The climb (seed 4 line, the earliest): from the floor's SW corner (649, -3098, 3652) jump
  NORTH while rising -> land ON the plate's SW corner (649, -3068, 3714 = feet on the plate)
  -> jump SOUTH -> land on the south mouth's ledge (643, -3083, 3714) -> jump W/SW at z ~3740
  -> crawl. Three hops, each from a standing foothold.
* plan_to_bc per spawn seed (5,965 + 5,931 rows, 0 dropped) + the round-1 file -> bc_merged.npz
  (~29.8k rows); spawn_climb.npy = the 3 replayed best spines + their 6 s windows around the
  tunnel entry (500 ticks before .. 100 after) x4 = 16,141 rows. A distil costs ~300-350
  CPU-minutes regardless of the line count; two in parallel ~22 min; never next to a trainer.
* `pbr1b4_bc2` 20:15 UTC = resume of bc_latest (2.43B) + `--bc-file bc_merged.npz --bc-coef 0.5
  --bc-coef-final 0 --bc-steps 1e9 --bc-target dist --bc-value-coef 0.25 --spawn-states
  spawn_climb.npy --spawn-states-frac 0.4 --sil-coef 0 --sil-top 0 --front-temp 1
  --front-band 0.01`.

### 20:56 UTC - THE START TRAP IS PASSED BY THE GREEDY LINE: 3.229 (pbr1b4_bc2, expert iteration)

* pbr1b4_bc2 greedy evals: 0.43 / 0.44 / 0.42 / 0.42 / 0.42 (2.43B-2.73B), then **3.181 at
  2.804B and 3.229 at 2.879B**: 3 of 9 greedy episodes from the TRUE START mount the plate, hop
  into the south tunnel, crawl it and drop into the chamber (best 3.23%, far point
  (808, -3290, 3370), the chamber floor - the same place every search line ends). Median still
  0.41% (6/9 stay at the start). One eval earlier (2.577B) 1/9 had already mounted the plate.
* Checkpoint: the trainer keeps only ckpt_latest (+ every 1e9 steps), which moved 18.9M steps
  between the eval and my copy -> pulled as ckpt_2898264064.pt (its TRUE step; RESULT.json
  says so). Since 20:52 a watcher on the box (/root/ckwatch.sh <run> <protect>) copies
  ckpt_latest to ckpt_<step>.pt every 60 s (newest 15 kept): pull the one at/before a best eval.
* Round-3 DAgger waves (proposer = BC round-1 checkpoint, eps 0.15, resample 25, spawn seeds
  7-10): 3 of 4 crossed (arcs 2,962-3,008 u); distilled with OMP 8 threads sequentially beside
  the trainer (5,987 / 5,941 / 5,950 rows) - not merged yet.
* **The score is a MAX over 9 episodes: reliability at the start does not move it until
  something finishes. Next gate = the chamber (an energy gate: the record leaves on an upper
  path at z ~3500, ~150 u above the floor where every line stops).** Running: beam_tas waves
  from the true start with the bc2 checkpoint (it passes the start itself), 60 s cap, eps 0.15,
  seeds 11-13 - can the population find the way out of the chamber?
