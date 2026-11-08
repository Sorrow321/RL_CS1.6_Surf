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

## Round 1, unr1b1 (slot 1: the Nature paper's random explore step)

### 2026-11-08 05:00 UTC - the explorer ALONE, rooted where October policies stalled

Box 59665725 (RTX 5060 Ti, Threadripper PRO 3975WX, 64 threads, quota 30.7 CPUs, $0.177/h).
`tools/explore_phase1.py` with the slot's settings: uniform random actions, 100 decisions of
4 ticks, 95% repeat, views held at offsets from the velocity (`--view-follow
-90..90`), cells 64 u x energy height 32 u x rising/not (`--cell-heading 1`),
`--select-energy 1`, `--maxvel 4000`, 2048 envs on 14 threads (~0.75M ticks/s). Roots: the
last 4 s before the stall of the best episode of an October greedy eval (the POLICY'S OWN
line; analysis only, never training), 48 states, approximate STATE_DTYPE rows from the traj
rows (origin, velocity, yaw, pitch, onground).

| test | roots | 6 min of explorer (grader field goalc_32) | wall passed? |
|---|---|---|---|
| first ramp | r2b2_sband@151.5M, stalls 4.48% | max 5.71% (757k cells, 4.5M bursts) | NO (8%) |
| window | r2b2_sbandx@526.4M, stalls 20.44% | max 20.75% (353k cells, 3.4M bursts) | NO (sgcp_1 never entered) |
| pit dive | r2b2_sdef@1263M, stalls 26.97% | max 27.48% (1.89M cells, 2.7M bursts), a fast fall | NO (27.5%) |
| true spawn | the map spawn | ~5.7% after 14 min (812k cells, 9.9M bursts) | NO |

* First ramp: the October line that clears 8% (r2b2_sband@302.5M) climbs a +y ramp at ~1,200
  u/s and U-turns at its top (~(13,285, 1,008, 13,707)); the explorer reaches the climb
  ((12,203, 900, 13,473)) at 660-740 u/s and never makes the turn. A speed-gated manoeuvre:
  random bursts spend the energy it needs.
* Window: the passing line (r2b2_sdef@1036.5M) arrives at the x = 1,200 wall at z ~13,150-13,330
  climbing at ~760 u/s and goes through near y ~40; the stalled line arrives at z ~12,890 and
  the explorer's deepest cells slide along the wall at z ~13,192 (20.75%): close, not through.
* 96-97% of bursts die; the archive grows 100-500k cells a minute, nearly all ballistic air.
* `--cell-contact` (archive only states the map holds) at the window: 57k cells instead of 353k,
  but max 20.44% (worse than 20.75%): the hole needs a flight, and contact roots do not find it.
* So the Nature explore step ALONE does not get past walls 1-3 in minutes from where a policy
  stops (the pit within 0.02% of its threshold, on a fall). It still leads a from-scratch policy
  between walls (the CPU smoke: a random policy's 0.01% line -> 2.5% in 20 s of explorer).

### 2026-11-08 05:15 UTC - the loop: `--ge-line` (commit 2450a18a), run unr1b1_gel

* `train_fast.py --ge-line FRAC`: after every greedy eval, while no search runs, the best
  greedy episode (exact states, the `--respawn-greedy` capture; `GreedyLinePool.last_eps`) goes
  to `tools/auto_spine.py --line` BESIDE the training: no beam, no policy, the uniform explorer
  above (`--ge-boost 1.0`: no velocity boosts, `--ge-select-energy 1`, exact chains) rooted on
  the line's last 4 s before its stall for `--ge-line-minutes`. The deepest chain (goalc d) is
  replayed exactly and spliced onto the line. It is ADOPTED when its end beats the best greedy
  episode's deepest d by `--ge-line-gain` (200 u), and the current line's end by as much unless
  the greedy has crossed the current line's success line. The adopted line is a
  DemoCurriculum on FRAC of every training spawn pool (the `--finish-line` composition: the
  recipe's reservoir / novel / greedy / front-perturb rows are kept): half the draws on the
  frontier band (0.5 s), half uniform behind it to the line's end; the start widens 0.25 s per
  advance (doubling to 4 s, cooldown 5 iterations, min 50 episodes) while the band's episodes
  reach the end (ratchet record within `--ge-line-success` 100 u of its d, or a finish) >= 20%.
* Tests (`tests/python/test_ge_line.py`, CPU, sidistic): flag off bit-identical to the trainer
  before the change (config + every progress.csv number; 2 passed, 11:43); TRAIN_ONLY; a smoke
  where 6 searches ran beside the training, 3 lines were adopted, a shallower one was refused.
* Run `unr1b1_gel`: the incumbent recipe + `--ge-line 0.2 --ge-line-minutes 8
  --ge-line-threads 12`, from scratch, 6e9 steps budget. `box.py launch` returned exit 255
  (broken pipe) but the trainer (pid 5119) and run_arm.sh (its record gate) kept running; the
  box record therefore does not list the run - it is `runs/unr1b1_gel` on box 59665725.
* The control at matched steps (05:06 mirror): 1.7% at 76-378M, first ramp not cleared, 174k
  steps/s.

### 2026-11-08 06:05 UTC - unr1b1_gel at 459M: first ramp 227M, WINDOW 378M; the control 1.7%

| step | unr1b1_gel best (median) | max_cp | control unop_ctl best |
|---|---|---|---|
| 0.5M | 0.8% (0.4%) | 0 | 1.4% |
| 76M | 1.7% (1.7%) | 0 | 1.7% |
| 151.5M | 6.7% (5.8%) | 0 | 1.7% |
| 227M | 15.6% (12.7%) - first ramp | 0 | 1.7% |
| 302.5M | 15.8% (15.8%) | 0 | 1.6% |
| 378M | 26.8% (23.5%) - window (sgcp_1) | 1 | 1.7% |
| 453.5M | 27.0% (26.9%) - at the pit wall | 1 | 1.7% (to 680M) |

* Ladder: first ramp 227.0M (October 302.5M, deadline 1.0B) ON TIME; window 378.0M (October
  1,036.5M, deadline 2.07B) ON TIME. fps 172k with the search running, the control's 174k.
* What the explorer did: search 1 (from the 0.5M eval's best episode, d 98,824) found a 30.9 s
  line to d 94,179 (5.5%) in 495 s, adopted at 89M; the band succeeded 35% and walked back to
  row 641 of 3092; the greedy crossed its success line by 151.5M (6.7%). Search 2 (from 6.7%)
  found nothing 200 u past the greedy (92,957 vs 92,984); the POLICY then cleared the first
  ramp (227M) and the window (378M) by itself. Search 3 (from 15.8%) found 83,628 vs 83,888 at
  its start, but the greedy was at 72,995 when it ended: not adopted (the latest-greedy rule).
* So: the control sits at 1.7% for 600M steps; this run left 1.7% within 75M steps of the
  first line. One seed each - October from scratch cleared the first ramp in 0.18-1.51B and 4
  runs never did - so the line is the likely cause, not a proof.
* A flaw to fix in the next version: once the greedy has crossed the adopted line's success line
  the line keeps 20% of the spawn pool (here the map's first 5.5%) until a deeper line is found.
  Its band also stopped receiving outcomes at row 641 ("0 eps"): probably repeated origins in
  the line's slow start (the `--demo-match-band` defect) - harmless here, the greedy was past it.

### Harness issue (unr1b1, 2026-11-08 05:45): a path-MTU black hole from the box to the droplet

`box.py pull` returned an empty tar ("empty file") and scp stalled at 0 bytes: from box
59665725 (193.181.215.114:33140, behind vast's NAT 10.66.0.1) to the droplet every TCP segment
over ~1,400 bytes is dropped (`head -c 1000` arrives in 1 s, `head -c 1400` never does; the
droplet's route cache says PMTU 1420; the box's /proc/sys is read-only, so no MTU probing).
Workaround on this branch, nothing system-wide: `tools/mss_proxy.py` is an ssh ProxyCommand that
sets TCP_MAXSEG 1200 on its own socket, and `tools/pull_run_mss.py --agent A --run R [--ckpt]
[--snapshot STEP]` runs box.py pull's exact tar stream (and md5-checked checkpoint copies)
through it into `$AR_HOME/runs/<agent>/<run>/` (200 KB in 1 s). The fleet watchdog's own
harvest of this box would fail the same way: pull before every deadline.
