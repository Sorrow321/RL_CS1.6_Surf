# NOTES - the unified-recipe lineage on surf_src_sidistic

The lineage's memory: every round's winner appends what it learned (what worked, what did not,
the numbers, the ladder). Read it before you start; append to it before you wrap up.

## Round 0 (operator, 2026-11-08)

* **Why this lineage exists.** After four maps solved by fork-and-flush (sidistic,
  princessburglar, unitfarmer2, za_loop), the user read the meta-analysis
  (`/home/steam/meta_analysis/four_maps_one_recipe.html`, reports in
  `/home/steam/meta_analysis/reports/`) and called the result a "Frankenstein": a different
  mix of methods per map and per wall. This program looks for ONE recipe, frozen at launch,
  that clears sidistic's walls by itself (PROTOCOL.md: the goal, the one-recipe rule, the
  ladder, your slot). The candidate: a Go-Explore archive whose explore step runs from fully
  random to fully policy-guided on one knob, feeding a backward curriculum.
* **The code** is `op/zl-expert` (796c56f7): the za_loop lineage's best branch (ar/zlr3b2) plus
  the operator's expert-loop fixes. It carries every earlier lineage's tools. Earlier notes are
  in git history: `git show 5096b65c:autoresearch/NOTES.md` (za_loop rounds 1-3),
  `git show b93a7151:autoresearch/NOTES.md` (unitfarmer2), `git show
  8a825803:autoresearch/NOTES.md` (princessburglar). The October sidistic lineage's own notes:
  `/home/steam/ar/archive/r7/r7b2/NOTES.md` (read-only; its runs are under `/home/steam/ar/runs/`).
* **The recipe** (`config.incumbent_recipe`) is the October finisher's (r7b2_bel,
  `/home/steam/ar/runs/r7b2/r7b2_bel/run.json`) from scratch at that lineage's physics: 10 ms,
  `--maxvel 4000`, `--ep-ticks 30000`. run_arm.sh's SCRATCH preset adds the continuous
  velocity-frame view, held keys, the potential channel + curtain, the novelty bonus (0.25, 8
  view sectors, 3 speeds) and the rest.
* **The grader** (`score.py`) reads `maps/surf_src_sidistic.goalc_32.npz` with start_d
  99,675 u, the October grader's field. Checked 2026-11-08: the October finisher's eval
  traj_25448939520 scores 107.176 here (1/9 finishes, 118.05 s from spawn, a real trigger
  finish).

### The ladder (autoresearch/ladder.json), and what October did at each wall

The lineage: r2b2_sband (from scratch) -> sbandx -> sdef -> sband2 -> r3b3 -> r4b3 -> r5b4 ->
r6b2 -> r7b2. Steps are its cumulative checkpoint counter (1024 envs x 10 ms ticks). The flag
changes are for INSIGHT ONLY: rebuilding one of them as a per-wall fix is exactly what this
program rules out.

| wall | October past | change made just before (warm resume) | from scratch |
|---|---|---|---|
| first ramp (>8%) | 0.30B | none (scratch root) | 0.18-1.51B; 4 runs never |
| window (sgcp_1) | 1.04B | novel sampler band -> default at 0.89B | r3b2 0.45B, r1b1 0.58B |
| pit dive (>27.5%) | 1.94B | band rows back on at 1.34B | r3b2 0.98B, r1b1 1.63B |
| cavity (sgcp_2) | 4.14B | --respawn-greedy 0.15, --int-real-only at 3.34B; --time-pen 0 --surf-bonus 0 at 3.74B | r3b2 1.28B |
| post-CP2 block (>33%) | 6.97B | --respawn-greedy 0.3, window 0.25-2 s at 6.24B | r3b2 1.59B |
| CP3 | 7.32B | none | r3b2 1.91B |
| CP4 | 9.39B | --eval-patience 30, --eval-parallel 1, --unstuck-temp-heads keys, --stall-eps 5 | - |
| CP5 | 11.78B | --respawn-greedy-stuck 200, --respawn-novel-band 0.15 | - |
| CP6 | 14.26B | none | - |
| CP7 | 15.23B | none | - |
| chamber entry | ~19.35B | T cap 0.2, pitch-sigma floor, no-corner-cut field, backward mix, --respawn-front-perturb | - |
| finish | 25.45B | --time-pen 0.001, perturb x1.0-1.3, --energy-pot (K 1 -> 3, front bonus), --respawn-backward-elev -60 60, lr 1e-4 -> 5e-5 | - |

* Throughput in October: 135-277k steps/s on rented 3090 / 4090 / 5080 boxes (0.5-1.0B per
  hour). The lineage's 25.45B steps to the first finish were ~42 trainer-hours.
* Every first-past step can be late by one eval interval (75M early, 3-25M later). The counter
  leaves out discarded backtracks and the three losing branches of every round: the October
  numbers are a best-of-4 per round.

### The Nature Go-Explore paper (Ecoffet, Huizinga, Lehman, Stanley, Clune 2021)

Read from the authors' full version with Extended Data and Supplementary Information
(https://adrien.ecoffet.com/files/go-explore-nature.pdf; the arXiv copy leaves out the SI
ablations, sections 3.1-3.10). "~" = read off a figure.

* Restore-based exploration (SI Alg. 1): sample 100 cells with probability proportional to W,
  restore each, explore. W = 1/sqrt(C_seen + 1) (Methods eq. 4), C_seen = explore rollouts that
  touched the cell. Explore = uniform random actions for 100 steps (Atari, 4 frames each; 30 in
  robotics) or until the episode ends, the previous action repeated with p 0.95 (0.90 in
  robotics). A cell's state is replaced by a new arrival with a higher score, or the same score
  and a shorter trajectory; one state per cell, paths never spliced. Cells: downscaled frames
  whose parameters a random search re-tunes during the run, or domain-knowledge cells (room,
  x, y; level, keys). 2B frames per Atari run (1B with domain cells), 20M in robotics.
* Policy-based Go-Explore: PPO + self-imitation (16 of 256 actors replay archive trajectories);
  reward +1 per cell reached along the archived cell trajectory (soft window 10), +3 at the
  final cell, plus the clipped game reward. Start cell W = 1/(0.5 C_steps + 1). After reaching
  the cell the actor commits 50/50 for the whole explore step: random actions (not trained on)
  or policy sampling toward goals (10% an adjacent cell not in the archive, 22.5% any adjacent
  cell, 67.5% an archive cell by W; a new goal when reached or after 100 steps), with the
  softmax temperature 1 + (0.01 max(0, t - 50))^2, t = steps since the last new cell; the
  episode stops after 1,000 steps without a new cell. Cells found (ED Fig. 7): Montezuma at 12B
  frames ~8,723 policy / ~4,911 return / ~2,595 random (3.4x; 1.5x at 0.6B), Pitfall at 10B
  ~5,416 / ~3,004 / ~1,394 (3.9x). The return path alone found 1.9-2.2x random's cells.
* Robustification: the backward algorithm with PPO + SIL (coefficient 0.1), 10 demonstrations
  (one drawn per episode), reward scaled by 10 / the demos' mean absolute discounted return;
  robotics 198 of 200 runs succeeded; 1 vs 10 demonstrations was NEVER compared; the
  policy-based variant needs no robustification.
* Ablations (SI 3.x): no action repeat is worse in most games and robotics never reaches 100%
  in 5x the budget; fixed (Montezuma-tuned) downscaling fails "often catastrophically" elsewhere;
  generic vs domain selection weights: slower (~65K at 2B vs ~157K at 1B) but it finishes;
  robustification without SIL: no change on Atari, robotics 0% vs 96.5%; policy-based without
  SIL: Montezuma ~1K vs ~20K, Pitfall ~1K vs ~6.5K; a policy given only the goal cell (no cell
  trajectory) earns nothing in Pitfall. NOT in the paper: exploring without returning, uniform
  selection, restore vs policy return at a matched budget, entropy-schedule ablations, MCTS
  (future work; earlier restore-based planners scored 540 and 2,500 on Montezuma).
* The paper's own caveats: random exploration "assumes that random actions have a sufficiently
  high probability of discovering new cells; more complex problems may require policy-based
  exploration"; two paths into one cell can be very different states ("jumping over a gap" vs
  "falling into it").

### Operator's control and harness notes

* **Control** `unop_ctl`: the incumbent recipe from scratch with NO search, on its own box
  (59663649, RTX 5060 Ti, 64 threads, ~125k steps/s at the start, 12 h deadline, launched
  2026-11-08 04:25 UTC). Its evals and ladder are mirrored to `$AR_HOME/control/` every ~30
  min. A first box for it (59662180) had another tenant on its GPU (100% busy, 150 W, nothing
  of ours running): blocklisted and released.
* `box.py launch` returned exit 255 ("Broken pipe") while the run had in fact started: check
  `ps` on the box and `runs/<run>_launch.txt` before launching again.

## Round log

### unr1b2 (slot 2: the policy-guided beam as Go-Explore's explore step) - 2026-11-08

**What was built** (commits 57395473, e97d7667; `tools/ge_beam.py`, `train_fast --ge-beam`,
`tests/python/test_ge_beam.py`, `tools/ge_beam_report.py`):

* An archive of EXACT states: cells = 64 u position x kinetic-energy height bucket x velocity
  heading sector x rising (explore_phase1.CellHash), selection W = 1/sqrt(1 + runs that touched
  the cell) (Nature) plus a 25% share drawn from the 64 deepest cells on the goal field. A cell
  keeps its first arrival unless a later one is 0.25 s sooner (a NEW node: chains through the
  old one stay exact).
* The explore step restores the selected cells through the core's OWN reset (spawn pool = the
  selected states, yaw jitter 0), so a 1-env replay of the recorded rows from the same state is
  bit-identical; a copy that dies is re-seeded at another selected cell mid-hold (its first row
  lasts h0 ticks) - every applied engine row is recorded. EVERY new cell any copy reaches is
  archived with its link (parent cell + rows); no culling.
* The knob, two forms: `--random-frac` = share of runs that are Nature bursts (uniform keys,
  uniform view offset in {-90..90} deg from the velocity, 95% repeat), and `--burst B` = every
  other run starts with B decisions of such held random rows, then the policy (keys at T 1.5,
  a held yaw-z offset N(0, 0.3) redrawn every 25 decisions).
* Lines: the deepest VIABLE cell (a run from it lived >= 1 s) is replayed link by link (each
  link's end checked against the archived state to 1e-3 u) into per-tick states from the true
  start; exported when 100 u deeper than the last. The trainer adopts a line once it beats the
  latest greedy eval's best episode by 300 u, as a DemoCurriculum (the --finish-line constants:
  0.5 s band, 0.25 -> 4 s steps, rho 0.2, cooldown 5) on 20% of every spawn pool; success = the
  paid ratchet record <= the line's end d + 200 u (or a finish). A deeper line replaces it when
  the greedy crosses the current one, the curriculum reaches the line's start, or 3e8 steps.
* Box 59667641 (RTX 4080, Threadripper PRO 3955WX 32 threads, 0.31 $/h).

**Explorer alone, the knob as the share of random runs** (2026-11-08 05:05-05:16 UTC, box
59667641, October checkpoint r0 = sidEM2cap2surfGR at 749.7M, whose greedy passes the window and
stalls at the CP2 wall 26.9-27.1%; three explorers in parallel, 512 envs, 64 cells x 150
decisions per step, cells 64 u x 128 u energy x 4 headings, 10 min each, ~540 steps, ~3M runs;
scored with tools/ge_beam_report.py on the grader's goalc field, start_d 99,675):

| random-frac | deepest | deepest viable (line) | max CP | cells | cells found by random runs |
|---|---|---|---|---|---|
| 0 (policy beam) | 27.44% | 26.24% (25.8 s line, CP1) | 1 | 1.88M | - |
| 0.5 | 27.21% | 25.52% | 1 | 2.15M | 54% (top-200 deepest: 74%) |
| 1 (Nature step) | 5.30% | 5.29% | 0 | 0.31M | 100% |

* The policy beam reaches the policy's own frontier in < 30 s of explorer time (8 lines exported
  in the first 30 s) and stays there: 27.44% is the greedy's own wall (pit dive = 27.5%), not
  past it. Random bursts from restored cells cannot fly the track (5.3% in 10 min) - the
  policy's competence is what moves the archive along the map.
* No setting passed the pit dive in 10 minutes. The archive grows ~5k cells per step at this
  cell size (2.5M nodes, 2.3 GB RSS in 10 min) - far too many for hours, and the count weight
  spreads over millions of cells. Next: coarser cells and the burst form of the knob.

**Explorer alone, the burst form of the knob and coarser cells** (05:24-05:34 UTC, same box,
checkpoint and protocol; cells 128 u x 256 u energy x 4 headings; goalc progress):

| explorer | minutes | deepest | deepest viable (line) | max CP | cells |
|---|---|---|---|---|---|
| policy beam (burst 0) | 9.9 | 27.48% | 26.33% (25.9 s) | 1 | 355k |
| burst 10 decisions on EVERY run | (1.5) | 9.7% (r0's field) | - | - | - |
| burst 25 decisions on every run | (1.5) | 5.7% (r0's field) | - | - | - |
| policy beam, frontier band 2% (frontier share 0.5) | 7.8 | 27.41% | 26.29% | 1 | 307k |
| burst 10 on 25% of runs | 7.6 | 27.38% | 26.48% (27.1 s) | 1 | 352k |

* A 0.4 s random burst at the start of EVERY run wrecks the flight (9.7% vs 27.8% after 1.5
  min; stopped): the same lesson as per-decision epsilon at speed - the policy cannot recover
  from a random start state at surf speed often enough to carry the frontier.
* On a quarter of the runs it is neutral for the frontier and the burst runs found 48% of the
  200 deepest cells; the frontier band changed nothing measurable in 8 min.
* Coarse cells reach the same frontier with 5x fewer cells (355k vs 1.88M in 10 min).
* Verdict for the explorer ALONE: the policy beam returns to the policy's frontier (through
  the window, CP1) within a minute from the true start, and NO setting of the knob passes the
  pit-dive wall (27.5%) in 8-10 min of 1/3 of a 4080 box. Random exploration alone cannot even
  fly the track. Where the knob matters is breadth at the frontier, not reaching it.
* The loop launched with: policy beam, 10-decision bursts on 25% of runs, cells 128/256/4, 512
  envs, 8 threads, duty 0.3.

**Run unr1b2_geb** (launched 05:35 UTC, record gate passed 05:37; box 59667641; commit
d342b843): `box.py launch --run unr1b2_geb --steps 2e10 -- --ge-beam --ge-beam-burst 10
--ge-beam-burst-frac 0.25 --ge-beam-cell 128 256 4 --ge-beam-envs 512 --ge-beam-threads 8
--ge-beam-duty 0.3` on top of config.incumbent_recipe, from step 0, one recipe.

**unr1b2_geb at 529M (06:07 UTC)** - first ramp at 453.5M (ON TIME, deadline 1.0B), window
CP1 at 529.0M (ON TIME; October 1,036.5M, r3b2 0.45B, r1b1 0.58B); best eval 24.404 (24.4%,
CP1). Throughput ~280k steps/s on the 4080 with the explorer at duty 0.3. The greedy sat at
1.7% from 76M to 302M; the explorer's lines (from the true start, through its archive) were
adopted as they passed it - line 10 (3.3%) at 291M, line 14 (5.1%) at 342M -> greedy 5.7% at
378M; line 19 (10.4%) at 438M, line 21 (15.2%) at 453M -> greedy 15.8% at 453.5M (first ramp);
then 23.5% (CP1) at 529M by itself, past the explorer's frontier (15.8%, duty 0.3 lags the
policy). The control unop_ctl (the same recipe without the explorer, a 5060 Ti) FAILED the
first ramp: no greedy eval past 8% by 1.06B. One seed each (the first ramp varied 0.18-1.51B
in October), but the lead-and-follow sequence is the mechanism working as designed.

**unr1b2_geb stopped at 1.06B (06:37 UTC), ON TIME, best 27.107 (982M; 27.080 at 680M)** - the
greedy reached the CP2 wall at 604M (27.0%) and sat there 27.0-27.1% for 450M steps: 8 of 9
greedy episodes die ~25.5 s in at 2,000 u/s at (3,050, +-900, 11,200), their best progress at
the death point. The explorer sat at the same wall (deepest 27.42%, viable 25.92%) and its
frontier rule was broken there: the 64 deepest cells (27.20-27.27%) had been restored 12-19 times
each and no run from any of them lived 0.2 s - the fatal flight itself - so a quarter of every
explore step went into it; the deepest cell a run lived 1 s from was at 25.9% (4,098, -886,
11,918), ~1.3 s before the death point. Fix (generic): `--ge-beam-front-doomed` - the frontier
share skips cells restored >= 2 times from which no run lived 1 s. Stopped instead of riding
to the pit-dive deadline (3.88B) because the run was not testing the explore step at a wall.
Harvest: runs/unr1b2/unr1b2_geb (15 evals, ckpt_latest, ckpt_1000341504, ge/ lines + logs).

**Run unr1b2_geb2** (commit d26c84ee): the same recipe + `--ge-beam-front 0.5 0.01
--ge-beam-front-doomed --ge-beam-duty 0.5` (half the selections from the deepest 1% band of
non-doomed cells, count-weighted; the explorer runs half the time).

**unr1b2_geb2 at 418M (07:02 UTC)** - first ramp at 302.5M (ON TIME; geb 453.5M; October
302.5M), best 15.736 at 378M; 287k steps/s with the explorer at duty 0.5 (no measurable cost
over duty 0.3). The same lead-and-follow: lines adopted at 1.3/1.7/2.8/3.2/3.5/4.9/6.0/11.7%
(12M..369M), greedy 1.7 -> 2.7 -> 4.3 -> 10.7 -> 15.7% (76M..378M), no 225M-step stall at 1.7%
this time. Control unop_ctl: still no first ramp at 1.36B.

**geb2 at 851M (07:27 UTC)**: window CP1 at 529M (ON TIME), best 26.914 (680M); at the CP2 wall
again from 604M. Explorer: deepest 27.24%, deepest viable 26.42% (geb: 25.92% - the doomed-cell
skip moved the viable frontier ~500 u deeper), 272k cells, 40 lines; no line adopted since 369M
(none beats the greedy). The curriculum's line-26 success fell 96% -> 60% while 4-5 of 9 greedy
episodes now stop at 15.8% (the policy changed its line there).

**The CP2 wall's anatomy (measurement only, CLAUDE.md 0b)**: the greedy dies inside a
trigger_teleport (-> sidstart) box x 2,592..6,240, y +-1,152, z 10,752..11,332; another one sits
above, x 2,528..3,680, z 11,552..11,972 - the way on is the ~220 u gap between them (and thin
"no_jump" teleports at z 11,392 near y +-900). The explorer HAS 1,379 cells in that gap
(x 2,400..3,800, z 11,332..11,552), every one falling at vz ~ -950..-980 u/s, none viable (no
run from them lived 1 s): all are mid-dive into the lower box. Nothing arrives level or rising.
So the wall is not a missing cell but a missing APPROACH: the line must enter the gap with ~zero
vertical speed, which the policy's dive never produces and a 0.4 s burst at 2,000 u/s does not
fix. A generic explore step that could: branching much earlier on the chain (seconds before the
dive, where the height/speed is still adjustable), more coherent (held) perturbation there, or
a cell key that separates level from falling (vz bins) so a level arrival counts as new.

**Explorer alone AT the CP2 wall** (07:31-07:46 UTC, beside geb2 on the same box; checkpoint
unr1b2_geb/ckpt_1000341504.pt = this lineage's own policy stuck at 27.1%; 256 envs, 4 threads,
15 min, cells 128/256/4, band 1%, doomed skip; `--chain-frac 0.5 --chain-secs 5` = half the
selections from the deepest viable line's own chain in its last 5 s):

| explorer | deepest | deepest viable | max CP |
|---|---|---|---|
| geb2's live explorer (no chain roots), ~1 h at the wall | 27.24% | 26.42% | 1 |
| W2: chain roots, 10-decision bursts on 25% of runs | 27.48% | 26.89% | 1 |
| W1: chain roots, 25-decision bursts on 50% of runs | 27.50% | 26.97% | 1 |

Chain roots move the viable frontier ~0.5% deeper (to ~1 s short of the death point) but no
variant finds a viable state past 27.5% - the approach into the gap is not produced. Next test
(60 min, 08:48 UTC): W1 + `--cell-vz 250` (rising / level / falling cells, so a level arrival
is kept instead of losing its cell to the dive that got there first), and the same with 12 s
runs and 10 s of chain roots. geb2 keeps training meanwhile (on time; pit-dive deadline 3.88B).

**What the CP2 wall needs, and whether the explorer reaches it** (08:15 UTC; insight from the
October notes, /home/steam/ar/archive/r7/r7b2/NOTES.md "WON'T vs CAN'T" and the 1,633M
milestone - analysis only): the crossing line drops under a thin fail plate (z ~11,905) just
west of its edge after the U-turn, flies EAST through the corridor under it, turns round on a
ramp at its east end, climbs west and crosses the gap LEVEL (z ~11,470) - ~2.8 s away from the
goal (the field's 2.0-unit dip). October: "perturbing the policy's actions at the wall is the
wrong lever; putting it into the unvisited states next to its frontier is the right one".
Measured in the explorers' archives (corridor = x 4,194..5,600, z 11,332..11,850):

| archive | corridor cells | heading east | east & viable | east-ramp cells (x 5,000..6,300, z 10,700..11,400) | viable |
|---|---|---|---|---|---|
| geb2 live (band, doomed skip; ~2 h) | 131 | 50 | 0 | 0 | 0 |
| X1 (+ chain roots 5 s, 25-decision bursts on 50%, vz cells; 40 min) | 10,613 | 2,386 | 37 | 14,882 | 149 |
| X2 (+ 12 s runs, 10 s chain; 26 min) | 1,656 | 287 | 3 | 5,260 | 21 |

So chain roots + long bursts DO reach the detour's states (the east corridor and the ramp) -
what is missing is chaining them into a line: those cells are BEHIND the distance frontier
(higher d), so only the small global count share ever restores them. Next test Z1 (30 min):
+ `--recent-frac 0.25` (a quarter of the restores from the cells found in the last 50 explore
steps - the novelty frontier, not the distance frontier).

**The explorer at the wall, 50 min (X1) - the return leg exists** (08:37 UTC): X1's archive has
2,190 cells heading west and climbing under the plate after the ramp turnaround (81 viable) and
2,104 cells heading west level-ish in the gap (105 viable); the furthest reach x ~3,276 at
vz ~ -240 u/s, ~700 u short of the pit floor's west edge (x 2,592) - they sink into the pit box
after ~0.7 s. The human crosses at z ~11,469, vz -141. So the explorer with chain roots + 1 s
bursts + vz cells finds the whole detour (drop under the plate, east corridor, ramp turnaround,
west climb, level in the gap) and lacks only a flatter / higher crossing. Z1 (+ recent share, 20
min): the same states, fewer (893 level-west cells, 19 viable); deepest 27.52% (not viable).

**unr1b2_geb2 stopped at 1.86B (08:38 UTC)**, on time (pit-dive deadline 3.88B), best 26.914; its
explorer had no chain roots (0 cells on the east ramp after ~2 h). Harvest:
runs/unr1b2/unr1b2_geb2 (evals, ge/ lines + logs).

**Run unr1b2_geb3** (commit 39b2820b, launched 08:39 UTC): the incumbent recipe + `--ge-beam
--ge-beam-burst 25 --ge-beam-burst-frac 0.5 --ge-beam-cell 128 256 4 --ge-beam-cell-vz 250
--ge-beam-envs 512 --ge-beam-threads 8 --ge-beam-duty 0.5 --ge-beam-front 0.5 0.01
--ge-beam-front-doomed --ge-beam-chain 0.5 5` (X1's explorer at twice its envs and threads).

**geb3 at 424M (09:03 UTC)**: first ramp at 227.0M, window CP1 at 378.0M (both ON TIME; the
fastest of the three runs - geb 453.5M / 529.0M, geb2 302.5M / 529.0M; October 302.5M /
1,036.5M); best 25.527 at 378M; 288k steps/s. The control unop_ctl has FAILED the first ramp AND
the window (2.64B, no greedy eval past 8%).

**geb3 stopped at ~0.86B (09:30 UTC)**, on time (first ramp 227M, CP1 378M), best 27.107 (680M),
at the CP2wall since 529M. Its explorer was useless there: its frontier sat at 15.75% from
explore step 408 to 795 (~25 min) while the policy flew to 27% - the archive had been built by
earlier, weaker weights, the chain roots (half the restores) kept branching from that old line's
dead end, and nothing re-ran the CURRENT policy from the start. (The X1 wall test never showed
this: its archive was built by a policy already at the wall.) Fix: `--ge-beam-trace 20 1500` -
every 20th explore step every env starts at the true start and the current weights fly 1,500
decisions (60 s) with no bursts (Go-Explore's return with the policy).

**Run unr1b2_geb4** (commit e580109e, launched 09:31 UTC): geb3's flags + `--ge-beam-trace 20
1500`.

**geb4 at 898M (10:23 UTC)**: first ramp 302.5M, CP1 529.0M (ON TIME), at the CP2 wall from 604M
(26.8-26.9%), best 26.893. The trace steps work: the explorer's frontier followed the policy
past 15.75% to the wall (deepest 27.27%, viable 26.22% at explore step 602). Its archive (10:21)
has no detour cells yet (49 east-corridor cells, none viable; nothing on the ramp or the return
leg) after ~10 min of explorer time at the wall; X1 needed ~40 min at full duty. Measured with
scratchpad detour.py (analysis only).

### Round 1 so far, in one place (10:25 UTC)

| run | explorer | first ramp | CP1 | at CP2 wall from | best | stopped |
|---|---|---|---|---|---|---|
| control unop_ctl (operator) | none | FAILED (none by 3.6B) | FAILED | - | ~6% | - |
| geb (d342b843) | beam + 10-dec bursts on 25%, duty 0.3 | 453.5M | 529.0M | 604M | **27.122** | 1.06B (frontier rule wasted on doomed cells) |
| geb2 (d26c84ee) | + doomed skip, band 1%, duty 0.5 | 302.5M | 529.0M | 604M | 26.914 | 1.86B (explorer never reached the detour) |
| geb3 (39b2820b) | + chain roots 5 s, 25-dec bursts on 50%, vz cells | 227.0M | 378.0M | 529M | 27.107 | 0.86B (explorer stuck behind its own policy) |
| geb4 (e580109e) | + trace steps from the true start | 302.5M | 529.0M | 604M | 26.893 | running |

* Every explorer run clears the first two walls on time - the explorer's lines lead the greedy
  through the first ramp (adopted at 3.3/5.1/10.4/15.2% in geb, the greedy following within 40M
  steps) - while the same recipe without it never clears the first ramp. One seed per run, but
  four of four with the explorer against zero of one without.
* No run passes the CP2 wall (27.5%). Its line is a 2.8 s detour away from the goal under a fail
  plate and back level through a 220 u gap. Alone at the wall, the explorer with chain roots +
  1 s bursts + level/falling cells reaches every piece of that detour in 40-50 min (X1), but has
  not chained it into a line past the gap's west edge.

**geb4 at 1.36B (10:50 UTC)**: best 26.988, at the CP2 wall (26.8-26.9% every eval since 604M).
Its explorer's archive (10:41): 310k cells; east corridor 83 (1 viable), ramp 23, level-west in
the gap 8 (min x 4,182), none past the west edge - ~1/3 of X1's explorer work at the wall so
far (duty 0.5 beside the trainer, a third of it spent on trace steps). Pit-dive deadline 3.88B
(~13:15 UTC).
**X1b** (10:51, analysis only): X1's own archive (geb's 1B checkpoint, 410k cells) resumed for
90 more minutes with the same flags, at low priority beside geb4 - does more search complete
the detour into a line past 27.5%? (A line from it cannot enter any training run: the
one-recipe rule.)

**X1b at 11:11 (80 min of search in all at the wall, analysis)**: the detour keeps filling in -
east corridor 3,681 cells (131 viable), east ramp 19,509 (404), level-west in the gap 2,733
(222) - but the furthest west in the gap is STILL x 3,276 (as at 08:37): ~700 u short of the pit
floor's west edge, arriving at vz ~ -240 u/s and sinking into the pit box. The explore step
covers the detour and stalls at its last stretch.

**geb4 stopped at 1.87B (11:24 UTC)**, on time, best 26.988, at the CP2 wall since 604M (T pinned
at the --unstuck-max 0.2 cap since 900M; --front-temp samples ~15% of envs at T 2 near the
front). Its explorer did follow the policy to the wall (trace steps) and was building the detour
(318 ramp cells by 11:11) - but the trainer only ever saw DEPTH-improving lines, and the detour
sits at about the greedy's depth (a 2.0-unit field dip), so none of it reached the training.

**Run unr1b2_geb5** (commit 8955493b, launched 11:25 UTC): geb4's flags + `--ge-beam-spawn
0.15` - 15% of every training spawn pool from the explorer's viable cells in the deepest 10% of
its viable cells (rewritten every 20 explore steps): the exact detour states, next to the
frontier, where October's notes say the CP2 wall is cracked ("put it into the unvisited states
next to its frontier").

**geb5 at 813M (12:17 UTC)**: first ramp 378.0M, CP1 529.0M (ON TIME), at the CP2 wall from 529M
(27.0-27.1%), best 27.077. Its explorer is at the wall (deepest 27.46%, viable 26.43%). The spawn
file is live: 103 viable frontier-band cells (18.5-26.4%, median 23.0%), rewritten every 20
explore steps; 3 of them on the detour (2 east-ramp, 1 level-west in the gap) - 15% of the pool
over 103 rows is ~6 spawns per row per pool rebuild, so the detour rows already get ~18 spawns an
iteration. Pit-dive deadline 3.88B ~ 15:35 UTC: geb5 runs to the wrap-up.

**geb5 at 1.29B (12:45 UTC)**: greedy still 27.0-27.1% (best 27.082). With X1b finished its
explorer fills the detour fast: east corridor 640 cells (1 viable), ramp 5,602 (37), level-west
in the gap 495 (6; furthest west x 3,776). The spawn file has 263 rows, 55 of them (21%) on the
detour (4 corridor, 46 ramp, 5 level-west): ~3% of all training spawns now start on the detour.
Whether the policy learns to commit to it from the true start is what the next evals show.

## Round 1 extended (operator note, 2026-11-08 15:10 UTC): one round to 2026-11-09 12:35 UTC

The user's decision: round 1 is the LAST round (wrap-up 2026-11-09 11:55 UTC); an ON TIME run keeps
training, a run that misses a wall deadline is stopped and a NEW recipe may start from scratch with
`--steps 60e9`; boxes extend <= 12 h at a time; wake every 45-60 min while a run trains; the
verdict is the ladder (which recipe got furthest, and when).

**geb5 FAILED the pit dive (stopped 15:20 UTC at 3.95B)**: no greedy eval past 27.5% by the 3.88B
deadline (3.851B: 27.098, 3.926B: 27.093; best 27.109 at 3.247B). Its explorer had the whole
detour (east corridor 4,325 cells / 110 viable, ramp 20,523 / 225, level-west 4,739 / 246,
furthest west x 3,106 at 14:38) and 596 frontier spawns, ~21% of them on the detour, for 3.4B steps
- and the greedy never once entered the corridor. So the states were there and the policy did
not take them: the explore step is no longer the bottleneck at this wall.

**Next recipe, unr1b2_geb6** (15:21 UTC, commit 8955493b): geb5's flags + `--unstuck-max 2`. Why:
every run here sat at CP2 with the plateau temperature pinned at the recipe's cap 0.2 since
~900M; that cap is October's BASEMENT-era value (round 5), never re-run from scratch (the
meta-analysis says so), while October's from-scratch CP2 crossing (r1b1 ratnovx, 1,633M: ratchet +
novel spawns) ran at T cap 2, and October's r2b3_t0 lost the CP2 line without the temperature.
One constant for the whole run, set at launch; T still rises only on a plateau (0.5 per 100M after
200M without 500 u) and halves on a new best, on the key heads only.

### BREAKTHROUGH (16:59 UTC): geb6 clears the pit dive and CP2

| wall | geb6 | October lineage | deadline |
|---|---|---|---|
| first ramp | 302.5M | 302.5M | 1.00B |
| window (CP1) | 453.5M | 1,036.5M | 2.07B |
| pit dive (>27.5%) | **1,208.5M** | 1,939.9M | 3.88B |
| cavity (CP2) | **1,661.5M** | 4,136.1M | 8.27B |

Best eval 31.38 (31.4%, CP2) at 1,661.5M. The greedy sat at 27.0% from 529M; the plateau
temperature started rising ~730M and reached the new cap 2 ~1.13B; 27.9% at 1,208M, 28.4% at
1,359M, 29.6% at 1,435M, 31.4% (CP2) at 1,661M; T fell back to 0.21 on the new bests. Its
explorer had the detour and now 7,319 cells PAST the gap's west edge (430 viable; deepest viable
30.35%). One seed: the cap and the explorer's spawns are not separated, but five runs with cap 0.2
and the same or a weaker explorer all sat at 27.1% for 1-3.4B steps, and this one left within
~80M steps of T reaching 2.

### 17:57 UTC: geb6 clears the post-CP2 block and CP3 (best 54.05 at 2,642.9M)

| wall | geb6 | October lineage | deadline |
|---|---|---|---|
| post-CP2 block (>33%) | **2,039.0M** | 6,965.7M | 13.93B |
| CP3 | **2,265.4M** | 7,318.0M | 14.64B |

Evals: 33.3% at 2,039M, 34.0% 2,114M, 36.1% 2,190M, 42.2% (CP3) 2,265M, 46.5% 2,492M, 50.3%
2,567M, **54.05% at 2,642.9M** (5 of 9 greedy episodes at 54.0%; they cross 31.4% at 36 s, 42% at
49 s, 50% at 59 s). Box extended to 2026-11-09 05:58 UTC (run ON TIME, ~287k steps/s).

**The explorer is now BEHIND the policy.** Its deepest cell has sat at 31.45% since explorer step
1,481 (2.16B trainer steps, 126 min): no trace copy (1,500 decisions = 60 s from the true start,
the current weights at keys T 1.5 + held yaw offset) and no frontier copy has passed 31.45%, while
the greedy flies to 54%. Its line (31.08%) still gets the GEL share (20% of the pool: no newer line
beats the greedy, and the adoption rule only REPLACES a line) and its spawns 15% - so 35% of the
pool starts at or behind 31% while the frontier is at 54%. The run still advances (frozen recipe,
not changed). Two generic defects for the next recipe: (1) a trace step should fly the policy at
its EVAL temperature (greedy), or the archive cannot follow a line the noisy copies cannot hold;
(2) the line share should lapse when the greedy has crossed the line's end and no newer line
beats it (the curriculum's job is done; the pool returns to the base recipe).

### 18:56 UTC: geb6 clears CP4 and CP5 (best 69.56 at 3,624.4M)

| wall | geb6 | October lineage | deadline |
|---|---|---|---|
| CP4 | **3,020.4M** | 9,392.6M | 18.79B |
| CP5 | **3,397.9M** | 11,783.4M | 23.57B |

Evals: 54.0-54.2% from 2.64B to 2.94B, 56.2% (CP4) 3,020M, 59.7% 3,171M, 61.5% 3,247M, 63.4% (CP5)
3,398M, 66.0% 3,473M, 69.5% 3,549M, **69.56% at 3,624.4M** (7 of 9 greedy episodes at 69.0-69.6%;
they cross 54% at 62 s and 63% at 76 s, and all end in a fail at z ~5,245 at 80-81 s - one place,
like every earlier wall). The episode cap (300 s) is far away. At ~286k steps/s the run reaches
~21B by the wrap-up; CP6's deadline is 28.5B, so the round's clock, not the ladder, ends it.
Training: stall 19% / crash 28% of rollout episodes (from 10% / 7% at 2.7B) as the frontier
spawns move deeper. The explorer's deepest cell: 31.48% (still behind).

### 19:54 UTC: geb6 at 82.45% (CP5) at 4,001.9M - at October's "83%" stop before CP6

Evals: 61.2% at 3,700M (one dip), 72.2% 3,775M, 76.1% 3,851M, 82.2% 3,926M, **82.45% at 4,001.9M**,
then 80.9-82.4% to 4,606M (0.6B flat). October's CP6 note: "stops at 72%, 83% and a creep 84-86%";
its CP6 came at 14.26B (deadline 28.5B). Training ent rose 0.005 -> 0.015 and int to 4.65/ep:
the plateau temperature is up again. Explorer still at 31.48%.

### 20:53 UTC: geb6 on the 81-82% plateau (5.66B; best still 82.45 at 4.0B)

Evals 4.76-5.66B: 80.6-82.3% (one 61.5% dip at 4.76B). The deep episodes peak at x -5,200..-6,600,
z ~4,400 at 91-93 s and fall (October: "stops at 72%, 83% and a creep 84-86%" before CP6 at
14.26B; deadline 28.5B). The early crossing has become less reliable: at 5.66B 5 of 9 greedy
episodes fall at 27.8% (x ~2,290, z ~11,400, the pit-dive edge) - medians 27.8% at 5.06B, 5.36B
and 5.66B against 0-2 of 9 at 3.6-4.1B. Training: stall 10% / crash 34%, ent 0.015 (the plateau
temperature is up), int 4.9/ep. Explorer unchanged at 31.48%. ON TIME, keeps training; box
extended to 2026-11-09 08:53 UTC.

### Attribution (21:05 UTC): the RL recipe or the explorer? (one seed each)

* **0-27% (first ramp, window): the explorer probably helped.** In every explorer run its line
  crossed the first ramp before the greedy (geb: lines adopted at 3.3/5.1/10.4/15.2%, the greedy
  following within 40M steps); 6 of 6 explorer runs cleared the ramp by 0.45B and the window by
  0.53B, while the same recipe without it (unop_ctl) never cleared the ramp in 7.6B (best ~6%).
  But that is ONE control seed, the ramp's from-scratch seed noise is 0.18-1.51B, and October
  cleared it at the same 302.5M with no explorer.
* **27-31% (pit dive, CP2): the temperature cap.** Five explorer runs at cap 0.2 sat at 27.1% for
  up to 3.4B steps - geb5 with the whole detour in its archive and in 21% of its spawns; geb6 left
  ~80M steps after T reached 2. October's r1b1 ratnovx crossed CP2 at 1,633M at cap 2 with no
  explorer; geb6 at 1,661M. The explorer's share of the CP2 crossing is not shown.
* **31-82% (CP3-CP5): the base recipe with cap 2 alone.** The explorer's archive has sat at 31.48%
  since 2.16B; its line + spawns (35% of the starts) sit at or behind 31%.
* "Ahead of October" is not explorer credit: October's lineage acquired its flags wall by wall
  (the ladder's notes), geb6 had the final recipe from step 0.
* The test that would settle it: the incumbent recipe + `--unstuck-max 2`, no explorer, from
  scratch.

### 21:52 UTC: geb6 still on the 81-82% plateau at 6.66B (best 82.47 at 6,417.8M, +0.02)

Evals 5.74-6.64B: best-of-9 80.8-82.5% except three dips (56.0% 5.81B, 57.1% 6.12B, 49.9% 6.19B);
medians are erratic (10.6% at 6.42B, 17.4% at 6.49B): the greedy loses its early crossings in
some episodes while the deep ones still reach 82%. Training crash share 39% (34% at 5.66B), int
5.8/ep, ent 0.015. Explorer 31.48%. ON TIME (CP6 deadline 28.5B); keeps training.

### Lesson for the next explorer (user's design note, 2026-11-08 ~22:10 UTC)

The user's intended design was ONE stream: the explorer searches, PPO trains on the explorer's
backward curriculum, and the explorer's explore step INTERPOLATES between the greedy policy
(T -> 0) and Nature's random rollouts (T -> inf), temperature (or beam / MCTS) being the knob.
geb6 did not do that: every copy ran at keys T 1.5 + yaw at the policy's own sigma + a held yaw
offset - about the noise of PPO's own training rollouts - and no copy was near-greedy. The 82% is
the DETERMINISTIC policy's; the sampled policy cannot hold the line for 90 s, so the archive
stalled at 31.48% (from-start traces also stop at 60 s, the greedy needs 92 s to reach 82%).
Next explorer: (1) draw each copy's temperature from a ladder that includes near-greedy (Ape-X,
Horgan et al. 2018: per-actor eps_i = 0.4^(1 + 7 i/(N-1))), (2) from-start traces as long as the
greedy line, (3) first CHECK that a zero-noise explorer copy reproduces the greedy eval line
(not verified in this round: a setup mismatch in the explorer's observation would look the same
as too much noise). With the explorer stuck behind, a curriculum-only PPO would have had no
starts past 31%: in geb6 the late walls came from the base recipe's own frontier spawns.

### 22:51 UTC: geb6 82.48 at 7.47B (plateau); the explorer moved 31.48% -> 54.2%

Evals 6.72-7.63B: best-of-9 80.7-82.5% (dips 57.1% 6.80B, 60.1% 7.40B), **82.475 at 7,474.8M**
(median 82.3% there, 27.8-34% in several others). The explorer left 31.48% at explorer step
4,712 (21:57 UTC, weights from 6.73B) and reached 54.24% (viable 54.21%, policy-mode copies,
line 95 = 66.9 s, 21 links) within 15 min - then stopped there again. 54% is where the greedy
passes at 62 s, i.e. where a 60 s from-start trace ends, and where the greedy itself sat from
2.64B to 2.94B. Reading: the policy's tolerance to noise improved (the trainer's key temperature
has been up on the plateau since ~4B), so T 1.5 copies now hold the line for a full trace, and
the trace length caps the rest. No adoption (54% is behind the greedy's 82%); its frontier cells
(15% of starts) now span 34-54% (3,872 cells, d 45,643..65,526 at 7.57B) instead of ~31%.

### 23:51 UTC: geb6 still 80-82.5% at 8.66B; box extended to 11:55 UTC

Evals 7.70-8.61B (13): best-of-9 80.0-82.5% in 12 of them (82.455 at 8.53B, just under the
82.475 best at 7.47B), one CP4 dip (60.8% at 8.46B); medians 27.8-81.0%. Training crash share
40%, ent 0.015, 283k steps/s. The explorer is still at 54.24% (explorer step 6,071, 510 min; no
new depth since ~22:12 UTC, line 53 still the curriculum). ON TIME (CP6 deadline 28.5B; at this
rate the round clock ends the run at ~21B). Box extended 12 h to 11:55 UTC; one more extension
after 00:40 UTC covers the wrap-up (11:55) and the round end (12:35).

### 00:53 UTC: geb6 82.52 at 8.91B (still CP5); box to 12:58 UTC

Evals 8.68-9.66B (14): best-of-9 80.7-82.5% in 13, **82.515 at 8,909.2M** (new best, +0.04
over 7.47B - the same plateau, not a new wall), one dip 71.0% at 8.76B; medians 14.6-81.2%.
Explorer still 54.24% (step 6,821, 572 min). Box extended 12 h to 12:57:51 UTC (covers the
wrap-up at 11:55 and the round end at 12:35). ON TIME; keeps training.

### 01:54 UTC: geb6 81-82.5% at 10.72B (best still 82.52 at 8.91B)

Evals 9.74-10.72B (14): best-of-9 81.0-82.5% in 13 (82.468 at 10.27B), one CP4 dip (60.0% at
10.65B); medians 20.3-81.6%. Explorer still 54.24% (step 7,545, 633 min). For the wrap-up: the
milestone snapshot at or before the best eval, ckpt_8002732032.pt, is pulled (80 MB, md5
c08c41fc..., 70 s); snapshots are every ~1.0B (ckpt_1000341504 .. ckpt_10003415040). ON TIME.

### 02:57 UTC: geb6 81.0-82.4% at 11.78B (best still 82.52 at 8.91B)

Evals 10.80-11.78B (14): all CP5, best-of-9 81.0-82.4% (no CP4 dip this hour), medians 32-81%.
The explorer inched to 54.71% (viable 54.54%, line 97; 75% of its 100 deepest cells now come
from burst copies). ON TIME; keeps training.

### 03:58 UTC: geb6 leaves the 82% plateau - 84.2-84.6% from 12.2B, best 84.60 at 12.53B

* Evals 11.85-12.91B: 82.3 (11.85B), **29.6% at 11.93B** (all nine episodes at the same 29.6%,
  a one-eval collapse), 83.8, 81.1, 82.2, then **84.2-84.6% in 9 of 9 evals from 12.23B**
  (84.604 at 12,533.1M, median 84.5% there). Still CP5; October's CP6 note reads "stops at 72%,
  83% and a creep 84-86%" - geb6 is in the same creep. Best moved in RESULT.json.
* What changed just before: the trainer ADOPTED the explorer's line 97 at 11,929.1M (68.0 s,
  25 links, end d 45,380 = 54.47%) - its comparison is against the LATEST eval's best d, and the
  29.6% collapse eval (best d 70,143) put the 54% line "past the greedy". The curriculum's 20% of
  starts moved from line 53 (all of 0-30%, half of them on its first 0.5 s, i.e. the map start)
  to the last 0.75-1.5 s of line 97 (54.5%), success 52-54%. Reading (one seed, not shown): the
  shift puts a fifth of the training episodes 30 s further down the route, nearer the frontier;
  the 84% level appeared 0.3B steps after it. It may also be coincidence: the plateau had held
  for 7.9B steps and October crept through the same 83-86% band without an explorer.
* Explorer 55.62% (line 100, 55.55%). Snapshot ckpt_12004098048.pt pulled (md5 a14dc9f1...,
  81 s) - the newest milestone at or before the best eval. ON TIME; keeps training.

### 05:02 UTC: the explorer catches the greedy - 55.7% -> 83.7% in 3 minutes; geb6 84.3-84.6%

* Evals 12.99-13.82B (12): all 84.3-84.6% best-of-9 (**84.626 at 13,212.6M**, new best, +0.02:
  the same 84% level), medians 27.8-84.4%. Still CP5 (October: "a creep 84-86%" before CP6).
* The explorer's line went 55.74% (line 101, explorer step 9,339, 04:23 UTC) -> 83.73% (line
  119, step 9,380, 04:26 UTC) in 41 explorer steps, every link from POLICY-mode copies (mode 0),
  24 -> 32 links, 68 s -> 100 s; then 84.15-84.45% (lines 121-123, 104.9 s, 04:29-04:55 UTC);
  deepest cell 84.56%. Weights at line 101: 13.18-13.21B, i.e. the policy that holds 84%. So
  the explorer, at a fixed T 1.5 + yaw noise, follows the policy once the policy is robust
  enough for its noise - it trails the greedy, it does not lead it (consistent with the
  explorer-noise lesson above). Not adopted: line 123 does not beat the latest eval's best d
  by the gain; the trainer's curriculum is still line 97 (54.5%), success 40.6%.
* Snapshot ckpt_13004439552.pt pulled (md5 27c71a7c..., 90 s). RESULT.json best moved. ON TIME.

### 06:07 UTC: geb6 clears CP6 (14,496.0M) and CP7 (14,873.5M) - best 97.61; the explorer led

Timeline (trainer step, eval best-of-9; explorer lines from ge/ge_beam_0.log):
* 13,892.1M: eval collapse (all nine at 10.4%); line 123 ADOPTED (patience, line 97 had run 2B
  steps): end 84.45% = the greedy's own frontier, so 20% of the starts sit at the frontier.
* 13.97-14.42B: the greedy creeps 84.8 -> 84.9 -> 86.0 -> 85.8 -> 86.1 -> 87.6 -> 87.6%.
* 05:37 UTC: the explorer's line jumps 84.74% -> 89.71% (line 126), then 90.0-91.5% (127-132),
  92.29% (133), 93.8-96.7% (134-144, 05:45-05:59), 97.94% (147, 124.9 s, 41 links, 06:06);
  deepest cell 98.04% (October's basement stall is 98.0-98.3%, chamber entry 98.5%).
* 14,477.2M: line 126 ADOPTED (89.71%; the greedy, 87.6%, had crossed line 123) -> eval
  14,496.0M **91.3% = CP6** (October 14.26B).
* 14,614.0M: line 133 ADOPTED (92.29%; greedy 91.4%) -> 14,722.5M 96.5%.
* 14,841.0M: line 144 ADOPTED (96.73%; greedy 96.4%) -> 14,873.5M **97.61% = CP7** (October
  15.23B). Then 88.6% (14.95B).
Reading (one seed): from 84% on, the explorer and the greedy ride the same improving policy;
each explorer line was 1-2% ahead of the greedy when adopted and the greedy passed it within one
eval. Both plateau exits (82% -> 84% after line 97, 84-86% -> CP6 after line 123) came 0.3-0.6B
steps after an adoption that put a fifth of the starts at or near the frontier. Not shown: what
the base recipe's own frontier spawns would have done in the same steps (they had 8B steps at
82% and did not). Snapshot ckpt_14004781056.pt pulled (md5 fe9bbd20..., 96 s). ON TIME; at
~1.02B steps/h the round clock ends the run at ~20.9B (October: chamber entry 19.35B).

### 07:02 UTC: geb6 best 97.96 (CP7) at 15.02B, then the greedy falls back to the pit (26-28%)

* Evals 15.02-15.40B: 97.96 (**best, 15,024.5M**, 0.5% short of the chamber entry at 98.5%),
  97.9, 97.9, 86.4, 97.9, 97.9; 15.48-15.55B 96.5, 96.5; then **26.1 (15.63B), 79.3, 28.2,
  27.8, 26.4, 26.1 (16.01B)**: 5 of the last 6 evals die at the old CP2 pit (26-28%), all nine
  episodes alike. The two earlier one-eval collapses (29.6% at 11.93B, 10.4% at 13.89B)
  recovered at the next eval; this one has not, for 0.4B steps so far.
* Training at 15.7-16.0B: crash share 47-55% (40% at 8.7B), kl 0.06-0.25 (0.05 before), ent at
  its 0.015 floor, T 2.0. "win 6-10% @ 2.2-2.5 s" with reservoir min-depth 1.69%: harvest wins
  from starts next to the goal, not policy finishes (evals 0/9 finishes throughout). A third
  of the envs are past 97.3% at any decision (fronttemp 30-37%: curriculum line 148 at 98.09%
  with success 3.6-3.9%, the explorer's frontier cells, the recipe's frontier spawns), so the
  early route gets fewer episodes - the likely cause of the regression at the pit (not shown).
* Explorer 98.36% (line 150, 98.31%; 0 finishes). Line 148 adopted at 15,251.0M (patience;
  greedy 86.4% in that eval). ckpt_15005122560.pt pulled (md5 78aafe3a..., 19M steps before
  the best eval) and set as best.ckpt. ON TIME (chamber entry deadline 38.7B): keeps training;
  the walls already cleared stand on the ladder.

### 07:57 UTC: geb6's greedy still stuck at the pit (26.4-29.5%) for 1.2B steps

Evals 16.08-16.84B (11): every one 26.4-29.5% (CP1; medians 22-29%), 0 finishes - the greedy
no longer crosses CP2 from the start, 1.2B steps after its last deep eval (96.5% at 15.55B).
Training: crash 50%, kl 0.13, ent 0.015, win 9.6% @ 2.3 s (harvest next to the goal, reservoir
min-depth 1.70%), the frontier tracker "stuck 2,069.9M" at 95,625 u. The recipe resets 90% of
episodes from its spawn sources (respawn_frac 0.9) - reservoir, frontier, novel band, the
greedy window (0.25: the last 0.25-2 s of the greedy's eval line, which now ends at the pit) -
and the curriculum (line 148, 98.09%, success 2.7-3.2%). The explorer is at 98.40% (0 finishes) and
cannot be adopted (line 150 beats line 148 by 219 u < the 300 u gain). ON TIME (chamber
entry deadline 38.7B); the frozen recipe keeps training, CP6/CP7 stand on the ladder, and the
best eval's checkpoint (15.0B) is local.

### 09:05 UTC: geb6 at 18.1B - the greedy crosses CP2 again in 4 of 15 evals, no deeper

Evals 16.91-17.97B (15): 26.1-32.2%, 0 finishes; four cross CP2 (30.9% at 17.36B, 32.2% at
17.52B, 31.5% at 17.67B, 31.2% at 17.82B), the other eleven stop at CP1 (26.1-30.2%). Best
stays 97.96 (15.02B). Training at 17.97-18.12B: crash 50-57%, kl 0.10-0.33 with one 0.81
(0.13 at 16.9B, ~0.05 before 15B), ent 0.015, harvest wins 6.4-10.1% @ 2.2-2.6 s (reservoir
min-depth 1.71%), fronttemp 25-34% past 97.3%; curriculum line 148 (98.09%) success 3.1-3.6%.
Explorer 98.40% (viable 98.35%, line 150 at 98.31% not adoptable, 0 finishes; 30.3M runs, 2.1M
cells). ON TIME (chamber entry deadline 38.7B) - keeps training to the round end (~21B at
11:55). Budget: credit $22.25, burn 0.91 $/h, reserve $10.

### 10:10 UTC: geb6 at 19.2B - greedy 26-32%; the PPO update size has grown ~8x since the regression

Evals 18.04-19.10B (15): 26.1-32.1%, 0 finishes; 8 cross CP2 (31.1-32.1%), 7 stop at CP1
(26.1-28.0%). Best stays 97.96 (15.02B); the trainer's frontier tracker has not moved its best
(95,625 u) since 14.87B ("stuck 4,230M"). **approx_kl, median per 0.25B of steps** (lr 3e-4
constant, 4 epochs x 16 minibatches, clip 0.2): 0.054-0.081 from 12.75B to 15.25B, then 0.101
(15.5B - the greedy regressed at 15.63B), 0.121 (16.0B), 0.143 (16.5B), 0.150 (17.5B), 0.202
(18.0B), 0.288 (18.5B), 0.411 (19.0B, p90 0.76). The policy now moves several times further per
update than it did while it was clearing CP6/CP7; the measured entropy sits at its 0.015 floor
throughout. Training: crash 49-57%, harvest wins 6.3-9.9% @ 2.4 s (reservoir min-depth 1.69%).
Explorer 98.44% (viable 98.37%, 0 finishes; 32.1M runs, 2.17M cells); line 150 (98.31%) still
not adoptable. ON TIME (chamber entry deadline 38.7B) - the frozen recipe keeps training.
Budget: credit $21.38, burn 0.91 $/h.

### 11:10 UTC: geb6 at 20.1B - greedy 27-31%; line 151 (98.41%) adopted on patience

Evals 19.18-20.08B (13): 26.8-31.1%, 0 finishes (5 cross CP2 at 30.8-31.1%). Best stays 97.96
(15.02B). Line 151 ADOPTED (patience) at 19,615.2M: 127.5 s, 51 links, end d 1,582 (98.41%),
the greedy's best d in the latest eval 69,706 (30.1%); success on its band 1.6%. approx_kl median per
0.25B: 0.42 (19.0B), 0.52 (19.25B), 0.47, 0.51, 0.42 (20.0B; p90 0.76-1.13) - level since
19B at ~6-8x the CP6/CP7 period's 0.05-0.08; crash share 52-56%. Explorer 98.45% (viable
98.42%, 0 finishes, 33.7M runs, 2.22M cells). ON TIME (chamber entry deadline 38.7B). Budget:
credit $20.49, burn 0.91 $/h. Next: the wrap-up at 11:55.
