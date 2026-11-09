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

### 2026-11-08 06:35 UTC - 784M: the pit wall; the control FAILED the first ramp

* The control `unop_ctl` (the incumbent recipe, no search) has FAILED wall 1: 1.7% at every
  eval to 1.057B (deadline 1.0B), ladder "STOP: a wall was missed". This run passed it at 227M.
* unr1b1_gel: 26.9-27.0% at every eval 453.5M-755.5M (the pit wall, threshold 27.5%).
* Search 4 (from the 27.0% line) found a 26.8 s line to d 72,393 (27.37%), adopted at 540.5M.
  Its chain (209 states from the greedy line's row 2466) makes the greedy's own turn at
  (4,635, 37, 12,347) but holds ~550 u more energy height on the descent (E 13,139 vs the
  greedy's ~12,585: ~450 u higher at the same x), and falls into the pit at 27.57%.
* The line curriculum found the hard spot and sits on it: the band from row 2499 (1.75 s before
  the end) succeeds ~41%, from row 2474 (2.0 s) ~7-11%; tau oscillates 2474 <-> 2499 every few
  iterations since ~600M. The policy cannot yet keep that height through the turn.
* Search 5 found a line to d 72,198 (27.57%, past the wall's threshold) and was NOT adopted: it
  beat the current line's end by 195 u, the rule asks 200 u (the greedy had not crossed line 4's
  success line). Search 6 runs from the 755.5M eval.
* Training: unstuck temperature at its 0.2 cap ("T 0.20", stuck 405M), novel ceiling 27.6%,
  fps 173.6k.

### 2026-11-08 06:40 UTC - v2 flags for the next run (commit 0ac44ba3; unr1b1_gel keeps 2450a18a)

* `--ge-line-retire`: once the MEDIAN greedy episode crosses the adopted line's success line,
  the line's pool share goes back to the recipe's own rows until the next line is adopted.
* `--ge-line-match-band`: the line curriculum credits a repeated-origin spawn to its copy in
  the frontier band (the `--demo-match-band` rule).
* The trainer deletes a finished search's exact-chain archive (0.3-1.3 GB each; ~70 searches
  in 12 h would fill a 60 GB box disk). For the running 2450a18a run I delete them by hand:
  `find runs/unr1b1_gel/gel -name archive.npz -mmin +15 -delete` (the trainer never reads them).
* Tests: v2 config dump + refusals pass; flag off still bit-identical to the base trainer
  (re-run on 0ac44ba3: 1 passed).
* Searches 5 and 6 both ended at d 72,198 (27.57%, a fall into the pit) and were refused by
  the current-line rule (195 u past line 4's end, 200 u asked).

### 2026-11-08 06:57 UTC - 1.06B: still 26.9%, the line curriculum walks back one notch

* Evals 906M-1,057M: 26.9% best, 26.8% median (pit wall). Search 7 ended at d 72,257, not adopted.
* Line 4's curriculum now oscillates between rows 2449 and 2474 (the band from 2474, 2.0 s
  before the line's end, rose from 7-11% to ~20% success; from 2449, 2.25 s, ~5%): the policy is
  learning the turn, 0.25 s per ~450M steps. The chain starts at row 2466; below it the bands are
  the greedy's own states.
* fps 174k; disk on the box 2.8 GB after deleting the finished searches' archives.

### 2026-11-08 07:30 UTC - 1.34B: what the pit wall is (insight, from this run's own lines)

* Evals to 1.284B: 26.9% (pit wall since 453.5M). Searches 8-9 ended at d 72,206 / 72,305 (not
  adopted). Line 4's curriculum still oscillates between rows 2449 (5% success) and 2474 (20-24%).
* Line 4's chain vs the 1.28B greedy's best episode at the same turn ((4,615, -106) -> (4,636,
  38) -> heading -x): the SAME energy height (13,710-13,750 u) but the chain flies the turn 50-70
  u higher (z 12,361 vs 12,304 at the apex, (4,316, -326, 12,402) vs (4,424, -320, 12,348) before
  it). At (4,124, 832) the chain LANDS on a slab at z ~11,939 (vz -818 -> -36, energy -457 u) and
  slides along it to the pit; the greedy passes ~40 u under that slab's level and keeps falling
  (vz -1,101 at (3,679, 800, 11,603)), ending at 26.9%. The skill is height at the turn, not speed.
  Line 4 is ducked at row 2440 (unducked by 2450).
* Box deadline extended to 13:23 UTC; disk 3.4 GB after the cleanup.

### 2026-11-08 07:52 UTC - 1.64B: line 11 adopted, the greedy still at the pit wall

* Evals 1.36B-1.586B: 26.7-26.9%. Search 11 (from the 1.51B greedy) found a 26.3 s line to
  d 72,152 (27.61%), 241 u past line 4: adopted at 1.597B. Its curriculum went straight to the
  same kind of spot: rows 2450 (22.8%) <-> 2475 (5.8%) of 2625, 1.5-1.75 s before its end.
* The run has sat at the pit wall for 1.13B steps (since 453.5M); the deadline is 3.88B.
* v2.1 `--ge-line-front-temp T` (+ `--ge-line-front-ticks`, default 100): an env whose episode
  started on the line curriculum's frontier band samples the keys at 1 + T and the yaw sigma x
  (1 + T) for its first ticks (the `--demo-front-temp` rule through the same per-env temperature
  slots; PPO scores each row at its own temperature; `gelhot X%` on the step line). Aimed at the
  pit's band (6-10% success just before the chain leaves the policy's line, a ~10 deg steer the
  policy's view sigma ~0.07 rarely samples). Verified on CPU: refusals, config dump, flag off
  bit-identical (4 passed), and an instrumented smoke where a line was adopted and line spawns
  were matched (126/126 respawns; map-start spawns are the line's row 0) and the band ran hot
  (gelhot 1.6-20.3%). A smoke shorter than the first search + an episode never respawns on the
  line: give it ~40k steps with --ep-ticks 300.

### 2026-11-08 08:17 UTC - 1.92B: the line-11 curriculum walks back onto the policy's own states

* Evals 1.66B-1.89B: 26.8-27.0% (best score 27.027 at 1.737B). Search 13 ended at d 72,226,
  not adopted (line 11 ends at 72,152).
* Line 11's curriculum moved from rows 2450/2475 to 2400/2425 (the band from 2425 succeeded
  20.9%, from 2400 1.3%): 0.7-1.0 s before the chain's branch point (row 2496), so the bands are
  now the greedy's OWN states and the policy is learning to steer onto the slab from its own
  approach. Box: fps 175k.

### 2026-11-08 08:46 UTC - PIT DIVE cleared at 2.114B (ON TIME); a 28.2% line follows

* traj_2114453504: best 27.6% (score 27.574), the pit-dive wall's first crossing (October
  1.94B, deadline 3.88B). The next eval (2.19B) fell back to 26.9%: one episode of nine, so not
  yet reliable.
* How: line 11's curriculum walked from rows 2450/2475 (1.5 s before its end) to 2400/2425 by
  1.92B - onto the greedy's own states before the chain's branch point - and the greedy learned
  to steer onto the slab the explorer's lines land on. Searches 12-14 found nothing 200 u past
  line 11.
* Search 15 (from the 2.114B greedy, best d 72,836) found a 30.6 s line to d 71,583 (28.18%),
  adopted at 2.20B; its band succeeds 89% / 73% and the curriculum is doubling back fast
  (steps 200, 400).
* Line 15's chain (546 states from row 2513, (4,283, 291, 12,001)) is a DETOUR the field does not
  pay: +x away from the goal (progress 26.0% -> 24.2%) through (5,647, 242, 11,010), a turn at
  (5,408, -475), back -x along y ~ -900..-1,040, ending at (2,222, -903, 10,990) falling toward the
  cavity curtain sgcp_2 (x 768-816, z 10,060-11,300) at 28.18%. The random explorer found the
  loop in one 8 min search.

### 2026-11-08 09:13 UTC - 2.50B: the pit crossing was one episode; line 15's loop is the new hard part

* Evals 2.19B-2.49B: 26.8-26.9% (the 27.6% at 2.114B was one episode of nine; the ladder keeps
  the first crossing). Search 16 ended at d 72,172 (not adopted; line 15 ends at 71,583).
* Line 15's curriculum walked back fast (89%, 73% success at its first bands: steps 200, 400)
  to rows 2783/2808 of 3058 and now oscillates there (29.7% from 2808, ~2% from 2783): the return
  leg of the chain's +x loop ((5,408, -475) -> (4,719, -734)), ~2.5 s before the line's end.
* Box deadline extended to 15:12 UTC; credit $49.89, fleet burn $1.27/h.

### 2026-11-08 09:45 UTC - 2.81B: the greedy is now one deterministic line dying at 27.05%

* Evals 2.64B-2.79B: best and median 27.1% (score 27.09). All nine episodes of the 2.79B eval
  are the same line: the turn at (4,528, 45), -x along y ~890, death falling at (3,040, 885,
  11,195), 25.4-25.9 s - ~500 u (+y) of the slab the explorer lines land on (y ~400).
* Line 15's curriculum still oscillates 2783 (30%) <-> 2808 (0.5%), the loop's return leg.
  Searches 18-19 ended at d 72,268 / 72,253: nothing 200 u past line 15 (71,583).
* Training's crawl share ("cr" in tr/st/cr) jumped 5-10% -> 30-34% at ~2.6B and stays there;
  the greedy does not crawl (it dies falling).

### 2026-11-08 10:15 UTC - 3.13B: the spawn-duck defect (a recommendation for the next recipe)

* Evals 2.87B-3.10B: 27.0-27.1% (one deterministic line). Searches 20-21: d 72,181 / 72,292.
* Line 15's hard window (rows 2783 -> 2808: 0.2% vs 37% band success) is a short DUCKED flight
  (rows 2776-2802: vz changing by the gravity step) onto a ramp that carries the line up from
  (4,840, -695, 11,146) (vz rising 8-11 u/s a tick).
* Under `--keys-hold` every spawn starts with the keys NEUTRAL, so a spawn on a ducked row
  unducks on its first tick (the hull's feet drop ~36 u in the air) unless the policy re-presses
  duck at once. The incumbent recipe does not set the existing generic fix
  `--keys-hold-spawn-duck` ("an episode that spawns crouched starts with duck held"). Ducked
  rows: 39-50% of the policy's own greedy lines (g004 49.3%, g011 49.8%, g015 39.4% of their
  prefixes) and 22-72% of the explorer's chains (random bursts press duck half the time). So
  every reservoir / greedy / novel / line spawn on such a row starts in another state than the
  one it was harvested from - plausibly why the curriculum stalls in ducked flights.
* RECOMMENDATION for any new run of this recipe: add `--keys-hold-spawn-duck` (an existing,
  tested flag; one generic rule, no map constant). A fuller fix for the line curriculum is
  spawning with the line's own held keys (the `--demo-keys` rule): the greedy prefix's keys are
  in the eval rows (fwd, side, buttons), the chain's in the archive's link actions
  (auto_spine.chain_keys) - not built.

### 2026-11-08 10:48 UTC - unr1b1_gel STOPPED at 3.50B (on time but stuck); next: unr1b1_v2

* Why stop an ON-TIME run: since 2.6B the greedy is one deterministic line dying at 27.0-27.1%;
  line 15's curriculum has sat on the ducked flight (2783: 0.2%, 2808: 37%) for ~0.9B steps; 9
  searches in a row found nothing 200 u past line 15. With 5 h left, a corrected recipe from
  scratch has more upside than 3B more steps of a stuck line, and the round keeps unr1b1_gel's
  best eval (27.574 at 2.114B) either way.
* Harvested to $AR_HOME/runs/unr1b1/unr1b1_gel/: 47 evals, progress.csv, run.json, the launch
  log, ckpt_latest.pt (the final weights, ~3.50B, md5 57119706...), snapshots ckpt_1000341504 /
  ckpt_2000683008 (the newest at or before the best eval) / ckpt_3001024512, gel_lines.tgz (every
  search's line, spine, roots, logs; archives deleted).
* unr1b1_v2 = the same loop + four generic fixes for defects this run showed (commit ea5ba4c7):
  `--keys-hold-spawn-duck` (ducked spawns hold duck), `--ge-line-retire` (a learned line gives
  its 20% back), `--ge-line-match-band` (band credit on repeated origins), `--ge-line-front-temp
  1` (band spawns sample keys at 2 and yaw sigma x2 for 1 s). From scratch, step 0.
* unr1b1_v2 launched 10:48 UTC on box 59665725 (commit 173fb6af). `box.py launch` broke on the
  path-MTU hole again (exit 255); the trainer kept running, run_arm.sh's record gate died after
  its greedy recording, and the gate re-run by hand PASSED (greedy, stoch, mixed, POV). The first
  eval (0.5M) is unr1b1_gel's to the digit (same seed 0, d 98,824).

### 2026-11-08 11:16 UTC - unr1b1_v2 at 281M: first ramp 227M (as v1); line 1 learned and RETIRED

* Evals: 0.8% (0.5M), 1.7% (76M), 5.8% (151.5M), 9.6% (227M; first ramp, v1 had 15.6% there).
  The same first line as v1 (seed 0: 30.9 s to d 94,179), adopted at 88M; `--ge-line-retire`
  fired at 151.5M (median greedy d 93,941 <= its success line 94,279): v1 kept that line on 20% of
  the pool until 540M. Search 2 (from 151.5M) ended at 92,962, behind the greedy's 90,129 by the
  time it ended. fps 176.7k.
* Cosmetic: after a retirement the step line's `gelhot` keeps its last value (the share is only
  recomputed while a line is drawn); it is not a live share.

### 2026-11-08 11:42 UTC - unr1b1_v2 at 558M: window 453.5M (ON TIME), at the pit wall

| step | v2 best (median) | v1 (unr1b1_gel) best |
|---|---|---|
| 227M | 9.6% (9.5%) | 15.6% |
| 302.5M | 9.6% (9.6%) | 15.8% |
| 378M | 17.9% (15.8%) | 26.8% (window) |
| 453.5M | 26.6% (25.3%) - window (sgcp_1) | 27.0% |
| 529M | 26.9% (26.8%) | 27.0% |

* Search 4 (from the 453.5M greedy) found a 26.1 s pit line to d 72,468 (27.30%), adopted at
  540.5M - the step v1 adopted its pit line at. Its curriculum went straight to rows 2387/2412
  (2.0-2.25 s before the end: 23.9% / 0.3%), the turn before the slab again; `gelhot` 2.2% live.

### 2026-11-08 12:10 UTC - unr1b1_v2 at 843M: on a pit line past the threshold

* Evals 604M-831M: 26.8% (the pit wall). Search 5 found a line to d 72,196 (27.57%) and it WAS
  adopted at 691.5M (272 u past line 4's 72,468; v1's equal line was refused by 5 u). Its
  curriculum: the band from 2434 (1.5 s before the end) 48%, from 2409 12.7% (v1 at the same
  steps: 41% / 7-11% on line 4). Search 6: 72,315, not adopted. fps 176.7k.
* Box deadline extended to ~16:16 UTC (the wrap-up pull at 15:55 needs the box).

### 2026-11-08 12:38 UTC - unr1b1_v2 clears the PIT DIVE at 906.5M (v1: 2.114B)

* Evals 906.5M / 982M / 1,057.5M: best 27.6% / 27.6% / 27.5% (the wall in three evals running;
  v1 crossed once), 1,133M back to 26.8%. Best score 27.581 (traj_0906493952) - the branch's best.
* Ladder v2: first ramp 227.0M, window 453.5M, pit dive 906.5M (October 0.30 / 1.04 / 1.94B;
  v1 0.227 / 0.378 / 2.114B). One seed each: v2's 2.3x at the pit is large but not a proof; what
  changed is the four fixes together (spawn-duck, retire, match-band, band temperature).
* The pit line v2 learned was search 5's (27.57%), adopted because it beat search 4's line by
  272 u; v1's equivalent was refused by 5 u. Line curriculum now at rows 2384/2409 of 2584
  (27.6% / 13.8%). Search 7 found nothing 200 u past the greedy.

### 2026-11-08 13:05 UTC - unr1b1_v2 at 1.43B: the pit crossed in 5 of 7 evals; curriculum 4 s back

* Evals 906.5M-1,435M: best 27.6 / 27.6 / 27.5 / 26.8 / 27.6 / 27.5 / 26.9 / 27.6% (median
  26.5-26.9%); best score 27.611 (traj_1434976256).
* Line 5's curriculum walked from rows 2384/2409 to 2159-2209 of 2584 (3.8-4.3 s before its end;
  20-21% at the band, a 3.8% backoff at 2184). Searches 7-9 from 27.5-27.6% greedy lines: nothing
  200 u past the greedy (deepest 72,118-72,243). fps 176.5k.

### 2026-11-08 13:34 UTC - unr1b1_v2 at 1.73B: the pit every eval; the explorer's root window is the limit

* Evals 1.43B-1.66B: best 27.6 / 27.6 / 27.5 / 27.5% (median 26.8-26.9%); best score 27.636
  (traj_1510473728). Line 5's curriculum: rows 2159/2184/2209 (20-34% / 3.7%).
* Searches 10-11 from 27.6% greedy lines: deepest 72,085-72,107 vs the greedy's 72,154 - nothing
  200 u deeper. In v1 the cavity loop (line 15, 28.18%) branched off at the turn ~(4,283, 291),
  and v1's greedy stalled AT the turn, so its roots ("the last 4 s before the stall") covered the
  branch. v2's greedy now stalls ~4 s later, in the pit, so the turn is outside the root window.
* A generic fix for the next version: roots over a longer stretch of the line (e.g. half the
  roots in the last 4 s, half spread over the 4-12 s before; or the root window grows each time a
  search returns nothing), so a detour that branches seconds before the stall stays reachable.
* Built (default off, for the next version): `--ge-line-roots-grow` - each search that ends
  without an adopted line doubles the next search's root window (auto_spine --roots-secs 4 -> 8
  -> 16 s, the cap); an adopted line resets it to 4 s. Tests: refusal, config dump, flag off
  bit-identical (4 passed).

### 2026-11-08 14:01 UTC - unr1b1_v2 at 2.03B: 28.2% best, median 27.6%; line 5 learned and retired

* Evals 1.737B-1.963B: 27.7 / 27.6 / 26.9 / 28.2% best; the 1.963B eval's MEDIAN is 27.6% (past
  the pit) - `--ge-line-retire` fired (median d 72,148 <= line 5's success line 72,296). Best
  score 28.235 (traj_1963458560), above v1's 27.574.
* Search 14 runs from the 1.963B eval's 28.2% episode. Searches 11-13: nothing 200 u past the
  greedy.
* The 1.963B greedy now flies the LOOP v1's explorer found (line 15): a swing to (5,821, 114) at
  26.5 s, -x along y ~ -900 at ~1,700 u/s, all nine episodes dying between x 2,180 and 2,670 (z
  10,740-11,430); the best two reach (2,183, -870, 10,980) = 28.24% and 28.08%. The cavity curtain
  sgcp_2 is ~1,400 u further -x (x 768-816, z 10,060-11,300), under the slab.

### 2026-11-08 14:28 UTC - unr1b1_v2 at 2.31B: 29.9% (median 29.6%), a 30% line in training

* Evals 2.04B / 2.11B / 2.19B / 2.27B: best 28.5 / 29.4 / 29.6 / 29.9%, median 28.2 / 29.3 /
  29.6 / 29.6% - the loop learned and extended. Best score 29.886 (traj_2265448448).
* Search 14 (from 28.2%): deepest 71,416, not 200 u past. Search 15 (from the 2.11B greedy)
  found a 33.0 s line to d 69,791 (29.98%), adopted at 2.20B; its band: rows 3075 (38.8%) / 3100
  (5%) of 3300. The cavity curtain (sgcp_2) not yet entered by any eval.

### 2026-11-08 14:56 UTC - unr1b1_v2 clears the CAVITY (sgcp_2) at 2.341B (October 4.14B)

* Evals 2.341B / 2.416B / 2.492B / 2.567B: best 31.5 / 31.8 / 32.1 / 32.2%, median 31.4 / 31.6 /
  31.7 / 31.8%, every one entering sgcp_2. Best score 32.158 (traj_2567438336).
* Ladder v2: first ramp 0.227B, window 0.454B, pit dive 0.906B, cavity 2.341B - October 0.30 /
  1.04 / 1.94 / 4.14B (deadlines 1.0 / 2.07 / 3.88 / 8.27B). Next: the post-CP2 block (33%,
  October 6.97B).
* The cavity fell to the curriculum on search 15's line (2.20B: 33.0 s to d 69,791, 29.98%) -
  the greedy went 29.9% -> 31.5% within 140M steps of its adoption. Searches 16-17: nothing 200 u
  past the greedy (deepest 68,012).

### 2026-11-08 15:30 UTC - the round is extended to 2026-11-09 12:35 UTC (the last round)

* Operator note (the user's decision, 15:10 UTC): round 1 ends 2026-11-09 12:35 UTC (wrap-up
  11:55) and is the LAST round. An ON-TIME run keeps training (no stopping it for a variant); a
  run that misses a wall deadline is stopped and a NEW recipe may start from scratch; new runs use
  --steps 60e9 and a run that reaches its cap is resumed from its own checkpoint with the same
  flags and a higher cap; box extensions <= 12 h; wake every 45-60 min while a run trains.
* unr1b1_v2 (launched with --steps 6e9) reaches its cap ~20:15 UTC. Then: copy its launch log and
  progress.csv aside on the box, and resume it from its own checkpoint with the SAME flags -
  `MAP=maps/surf_src_sidistic.bsp BUDGET=<60e9 - ckpt step> RECORD_EVERY=75e6
  CKPT=runs/unr1b1_v2/<its ckpt> ARM_RESUME=1 bash tools/run_arm.sh unr1b1_v2 --envs 1024
  --ge-line 0.2 --ge-line-minutes 8 --ge-line-threads 12 --ge-line-retire --ge-line-match-band
  --ge-line-front-temp 1 --keys-hold-spawn-duck` (box.py launch --resume deploys the incumbent's
  checkpoint, not a run's own) - and diff the new run.json config against the old one. The
  --ge-line curriculum state is not in the checkpoint: a resumed run starts without a line until
  its first search adopts one.
* 15:25 UTC: 2.93B, evals 32.3 / 32.1 / 32.4 / 32.3% (median 31.8-32.1%), best score 32.374
  (traj_2793930752). Box deadline extended to 2026-11-09 03:26 UTC.

### 2026-11-08 16:20 UTC - unr1b1_v2 at 3.49B: 32.4% since 2.8B, no line since line 15's retirement

* Evals 2.87B-3.47B: best 32.3-32.4% every time (median 31.4-32.4%; one eval's median 21.9%).
  Best score 32.418 (traj_3246915584). The post-CP2 block is at 33% (d 66,782); the greedy's best
  reaches d ~67,384.
* Line 15 was retired at 2.34B (median crossed its success line); searches 16-24 found nothing
  200 u past the greedy (deepest 67,300). The run trains on the recipe's own spawn rules alone -
  the case `--ge-line-roots-grow` (commit 0bb08d22, after v2's launch) was built for; v2 stays
  as launched (an ON-TIME run keeps training).
* Cosmetic: after a retirement the periodic line still counts "line episodes": map-start spawns
  match the line's row 0 (the true spawn).
* Pace 176.5k steps/s = 0.635B/h: the 6e9 cap ~20:15 UTC; the post-CP2 deadline (13.93B)
  ~08:45 UTC tomorrow.

### 2026-11-08 17:40 UTC - unr1b1_v2 at 4.21B; what the post-CP2 block is; the cap resume is armed

* Evals 3.55B-4.08B: best 32.2-32.4%, best score 32.445 (traj_3850895360). Searches 25-28: nothing
  200 u past the greedy (deepest chain ends d 67,296-67,341; the greedy's best d ~67,340). Still ON
  TIME (cavity 2.341B); the post-CP2 deadline is 13.93B.
* What the block is (MEASUREMENT only - map geometry from the goal field and the greedy
  trajectories, never a recipe input; CLAUDE.md 0b): ~33.5 s in, a ramp kick (t ~30.5 s,
  vz +1,150) throws the player to an apex z ~11,000 near x ~-200, and the route then goes -x
  through a SLOT about 200 u tall over a ledge: at x ~-912..-944 the field is free only for
  z ~10,340-10,532 (solid below and above), and it drops away past x ~-976. v2's greedy episodes
  mostly turn back short of it (min x -700..-830) and fall to the kill floor at z ~9,715; in the
  last 14 evals only 6 of 126 episodes got to x <= -880, all of them 180-570 u BELOW the slot
  (z 9,774-10,124, falling at 1,100-1,400 u/s, horizontal speed only 480-610 u/s). From the apex
  at ~850 u/s the slot is reachable only by flying nearly straight at -x: a height-precision wall,
  like the window (CP1), not a speed wall.
* The search roots (the last 4 s before the stall: line states 2890-3290 = 28.9-32.9 s) already
  include the ramp approach, and 12 searches of uniform random bursts from them never threaded the
  slot. Whether longer roots (--ge-line-roots-grow, 8/16 s) would is open; v2 cannot use it
  (frozen, ON TIME).
* Cap resume, verified ahead of time: a throwaway 1-iteration resume of a COPY of v2's checkpoint
  under a scratch run name, with exactly the resume flags, built a config identical to v2's
  run.json in 283 of 284 keys (`steps` alone differs); the scratch run was deleted. Two scripts on
  the box: `runs/resume_v2.sh` (copies the launch log, progress.csv and run.json aside, moves
  `gel/` to `gel_seg<HHMM>` - the resumed trainer counts searches from 1 again and would overwrite
  g001.. - then `ARM_RESUME=1 bash tools/run_arm.sh unr1b1_v2 ...` with the same flags, cap 60e9,
  pre-launch and post-launch record gates) and a one-shot watcher `runs/autoresume_v2.sh` (started
  17:22 UTC, log `runs/unr1b1_v2_autoresume.txt`) that runs it only if the trainer ended AT its
  cap (a `done:` line, ckpt_final.pt, its step >= 6e9) and `runs/unr1b1_v2.no_autoresume` is
  absent. A crash or a stop is left alone.
* Box deadline extended to 2026-11-09 05:23 UTC (17:22 UTC, 12 h). The box's container is capped
  at 30.7 CPUs (cpu.max); trainer ~14 + search 12, so nothing else runs on it.

### 2026-11-08 18:30 UTC - explorer alone at the post-CP2 slot with 16 s roots: nothing either

* unr1b1_v2 at 4.83B: the last 11 evals (4.08-4.83B) are 32.43-32.46% best, 32.42-32.44% median -
  one deterministic line that ends at the slot every time. Best score 32.456 (traj_4681367552).
  Searches 29-32: nothing 200 u past the greedy (deepest chain ends d 67,287-67,356).
* Test (analysis only, nothing changed in the run): the trainer's own search command
  (`tools/auto_spine.py --line`, same flags) on search 28's line, but rooted in the last 16 s
  before the stall (48 roots, line states 1690-3290) instead of 4 s; 24 min on 4 threads at
  nice 19 on the box beside the training (the trainer's cumulative fps kept rising through it,
  176,766 -> 176,848). Result: 623 iterations, 5.96M runs (a trainer search does 4.45-4.65M in its
  8 min), 3.88M cells (5x a search's 0.74-0.80M: the wider roots spread it out), 0 lines; best
  chain end d 67,365, no deeper than the greedy (67,347). So `--ge-line-roots-grow` would not
  have rescued v2 here, at least not at 16 s and one search's budget.
* Why the uniform step cannot do this wall (reading the trajectories): the slot is ~700 u of -x
  from the apex, and the apex velocity points ~45 deg off -x (-727, 547). Threading it needs a
  coordinated in-air turn (strafe + yaw sweep) that keeps the speed, then a straight glide that
  arrives within a 200 u height band. A 95%-repeat uniform burst with view-follow offsets rarely
  produces a coordinated turn, and the archive's cells (64 u x 32 u energy x heading) do not
  reward partial turns, so there is no stepping stone. The cavity loop (line 15) was a
  different kind of wall: an open +x detour that random bursts DO find.
* A local run of the same test was killed by its own RSS guard at 4 min (the archive passed
  2.6 GB on the 7 GB droplet): explorer tests belong on a box.

### 2026-11-08 20:40 UTC - unr1b1_v2 reached its 6e9 cap and resumed itself to 60e9

* The first segment ended at its cap: `done: 6,000,476,160 steps, avg 177,074 steps/s` (20:13:25
  UTC). The on-box watcher saw the trainer exit at 20:13:28, found the `done:` line and
  ckpt_final.pt at 6,000,476,160 >= 6e9, and ran `runs/resume_v2.sh` at 20:14:15. Pre-launch record
  gate on ckpt_final.pt PASSED (greedy + POV, 12 s); trainer pid 24267 alive; post-launch gate on
  the first ckpt_latest.pt PASSED (greedy, stoch, mixed + POV, 28 s). run.json vs the first
  segment's (kept as `run_seg2014.json`): 284 config keys, one differs - `steps` 6e9 -> 60e9.
* Restored from the checkpoint: novelty counts (19.5M visits), the respawn reservoir (20,000
  states, min d 67,358), the --unstuck state, the --respawn-novel state (ceiling d 67,271) and the
  greedy-line pool. NOT restored: the --ge-line curriculum ("no line yet") and its search counter
  (searches count from 1 again; the first segment's 40 searches are in `gel_seg2014/`). That is
  equivalent here: line 15 was RETIRED at 2,340,945,920 (the median greedy crossed its success
  line, the same eval that cleared the cavity), and a retired line adds no spawns and no band
  temperature (train_fast 173fb6af: the pool share and the front-temp band are both gated on
  `not GELS["retired"]`), so the training distribution has had no line rows for 3.66B steps
  either way; the adoption rule (beat the best greedy by 200 u) is the same with no line.
* The first segment's logs are kept on the box and locally with matching md5
  (`unr1b1_v2_launch_seg2014.txt`, `run_seg2014.json`, the resume and watcher logs);
  progress.csv continues in place.
* Evals 5.51-6.15B: best 32.42-32.47%, best score 32.472 (traj_5511839744); search 1 of the new
  segment: nothing (deepest chain end d 67,321, greedy 67,348). Post-CP2 block deadline 13.93B
  ~ 2026-11-09 08:40 UTC at 177k steps/s. Box deadline extended to 2026-11-09 07:26 UTC (19:26,
  12 h); one more extension after 00:35 UTC covers the round end.

### 2026-11-08 23:40 UTC - one 3 h explorer archive at the post-CP2 slot: saturated, 24 u

* unr1b1_v2 at 8.14B (178.9k steps/s): evals 6.23-8.11B all best 32.45-32.48%, median 32.4% - the
  same deterministic line ending at the slot. Best score 32.476 (traj_6302990336). Searches 1-14 of
  the second segment: nothing 200 u past the greedy (search 14's deepest chain end d 67,311; 15
  running). Still ON TIME (cavity 2.341B); the post-CP2
  deadline 13,931,380,736 falls ~2026-11-09 08:35 UTC. v2 already carries `--respawn-greedy 0.25`
  (window 0.25-2 s, stuck 200) from the base recipe - almost the mechanism October used to pass this
  block at 6.97B (0.3) - so that alone does not explain October's pass.
* Diagnostic (no action - the run is ON TIME): PPO approx_kl, median per 0.2B, 0.02 up to 2.5B,
  0.04-0.06 at 3.5-5.5B, then 0.07 (6.0B), 0.11 (6.4B), 0.35 (7.0B), 1.0-1.4 (7.4-7.8B), 2.1 (8.0B),
  p90 up to 8. It grows smoothly through the 6.0B resume (no step there; the trainer has no LR or
  entropy schedule on `--steps`, and the optimizer state is restored), explained variance stays
  0.996, value loss flat, evals unchanged. The view sigmas: yaw 0.05, pitch walked up to its 0.5 cap
  by ~7.6B. The trainer's own notes record the same symptom from a view head (approx_kl 1-22 per
  update, r5b4, `--pitch-sigma-min`); which head carries it here was not measured.
* Test x4L (analysis only, nothing changed in the run): the trainer's own search command
  (`tools/auto_spine.py --line`, same flags, the trainer's field copy) on the second segment's
  search-2 line, the trainer's 4 s roots, but ONE archive for 180 min (seed 1017, 4 threads, nice 19,
  20:35-23:35 UTC) instead of fresh 8-min archives. Result: 7,850 iterations, 79.9M runs (~17x a
  search's 4.5-4.7M), 878,872 cells (867.7k at 115 min: ~170 new cells a minute, saturated); the
  explorer's best (its goal_32 field) 65,624 u from minute 3 to minute 180; the replayed chain's
  end on the grader's field d 67,287 vs the greedy's 67,311 = 24 u, under the 200 u adoption gain.
  Peak RSS 2.40 GB; the trainer's fps unchanged (178.9-180.6k).
* Where the archive stops (snapshot at it 5000, 351 MB, read on the box) against the slot's shape
  (MEASUREMENT only, from the goalc field; never a recipe input, CLAUDE.md 0b): a ramp floor rising
  from x +200 (z 10,336) to x -152 (z 10,592) kicks the player to an apex ~11,000; from x -184 a
  shelf (solid z 10,528-10,592), open above it, a tunnel below it (z 9,952-10,528); a hole in the
  shelf near y ~0 at x ~-440..-696; above 10,528 an end wall from x ~-776; in the tunnel a ledge
  at x ~-920 with its top at 10,336-10,340 (192 u under the tunnel ceiling), floor 10,112 at x
  -952, open past x -984. The archive: no cell past x -904; at the ledge face (x -884..-916) its
  highest cell is z 10,257, 83 u under the ledge top; its max z per 32 u of x drops from 10,910
  (x -756..-788) to 10,394 (x -788..-820) at the end wall; its states that drop through the hole
  fall at 600-740 u/s with only ~400-600 u/s of horizontal speed, so they meet the ledge face.
* Reading: clearing the ledge needs a flight that enters the tunnel nearly level (a low apex, or a
  drop through the hole's far edge) with more -x speed than any random burst keeps - one coordinated
  manoeuvre, scored only at its end. 80M uniform 95%-repeat bursts and the archive's cells
  (64 u x 32 u energy x heading) give it no stepping stone, and the archive saturates, so more
  compute for the same explore step is not the missing piece. With x16 (16 s roots, 24 min) this
  closes the explorer-alone question at this wall for slot 1's step.

### 2026-11-09 00:40 UTC - unr1b1_v2 at 8.81B; box extended to 12:37 UTC; harvest snapshot pulled

* Evals 8.19-8.79B: best 32.45-32.46%, median 32.4-32.5%; best score still 32.476 (traj_6302990336).
  Searches 15-19 of the second segment: nothing (deepest chain ends d 67,285-67,346). ON TIME; the
  post-CP2 deadline 13.93B is ~08:35 UTC at 179k steps/s.
* `box.py extend --hours 12` at 00:37: box 59665725 now runs to 2026-11-09 12:37:56 UTC, past the
  round end (12:35).
* The snapshot at or before the best eval, ckpt_final.pt (6,000,476,160 - the first segment's
  end), pulled to the run dir now (md5 166c386574f325998b442a266140801e, box and local), so the
  round-end harvest needs only ckpt_latest.pt. Box-side snapshots: 1-5B, final (6.0B), 7B, 8B.
* Plan: if the ladder calls the post-CP2 block FAILED at 13.93B, stop v2 (ladder rule 2), pull
  its ckpt_latest.pt (copied aside, md5) and the gel records, and release the box: a new recipe
  launched at ~08:40 has ~3.2 h (~2.1B steps) before the wrap-up, where v2 itself took 2.8B to
  reach this wall, so it could not change the ladder's verdict.

### 2026-11-09 05:40 UTC - the explore step's repeat constant is not the barrier (xR90); v2 at 12.07B

* unr1b1_v2: evals 11.36-11.97B best 32.45-32.47%, median 32.4-32.5%; the best score is still
  32.476 (traj_6302990336). Searches 20-40 of the second segment: nothing (32-40: deepest chain
  ends d 67,289-67,329 against greedy bests 67,317-67,327). 65 searches since the last line
  (first segment's search 15, 2.20B), 80 in all. ON TIME; post-CP2 deadline 13.93B, first eval
  past it ~14.00B, ~08:40 UTC.
* Test xR90 (analysis only, nothing changed in the run): x4L's explorer
  (`tools/explore_phase1.py`, the flags a v2 search runs: 64 u x 32 u energy x rising/not cells,
  --select-energy 1, view-follow -90..90, 100-decision bursts, exact chains, the same 47 roots
  from v2's own greedy line, seed 1017, 4 threads, nice 19) with ONE constant changed:
  `--repeat-p 0.90` (x4L: the default 0.95; the Nature paper's robotics value). 60 min
  (04:33-05:34 UTC): 2,634 iterations, 27.1M runs, 825,763 cells; best (its goal_32 field)
  65,640 u at minute 4, 65,596 u from minute 44 (x4L 65,624 u). Replayed on the grader's field:
  chain end d 67,281 (5 cells, 136 states) vs the greedy's 67,311 and x4L's 67,287.
* Envelope at the slot (arch_slot.py; geometry as in the 23:40 entry, MEASUREMENT only): no cell
  past x -904 (x4L: -904); at the ledge face (x -884..-916) the highest cell z 10,256 (x4L
  10,257), 84 u under the ledge top; cells at z >= 10,340 end at x -852 (the most -x one, x -832
  z 10,362, moves (-408, 155, -652) u/s); max z per 32 u of x drops from 10,910 (x -756..-788) to
  10,388 (x -788..-820) at the end wall, as in x4L. Peak RSS 1.34 GB; trainer fps 178,975-178,981.
* Reading: a 2x shorter mean hold (20 decisions -> 10) moves the envelope by 1 u, so the repeat
  constant is not what keeps the uniform step out of the slot.
* Next (running, analysis only): xH8 = x4L's explorer with `--cell-heading 8` instead of 1 (8
  sectors of the horizontal VELOCITY heading x rising/not, an existing generic option of the
  explorer). With --exact-chains a cell keeps the FIRST state that keys into it, so a 64 u x
  energy x rising/not key keeps whichever heading arrived first, and a fast -x approach to the
  hole may never get a cell to return to. 60 min from 05:40 UTC, guard 10 GB, max-cells 6M.

### 2026-11-09 06:46 UTC - a direction-aware cell key does not get in either (xH8); the slot is a speed gate

* unr1b1_v2 at 12.74B: searches 41-45 nothing (42-45: deepest chain ends d 67,302-67,324); 70
  searches since the last line, 85 in all. ON TIME; fps 178,956.
* Test xH8 (analysis only): x4L's explorer with `--cell-heading 8` (8 sectors of the horizontal
  velocity heading x rising/not) instead of 1, repeat-p back at 0.95, the same 47 roots, seed 1017,
  60 min (05:40-06:40 UTC). The key space is 8x larger and the archive grows ~6x: 5,117,476 cells
  (under the 6M cap; peak RSS 4.93 GB), 1,277 iterations at 0.29M ticks/s (half xR90's rate:
  each iteration is slower with the larger archive), 13.2M runs. Best (goal_32) 65,651 u (xR90
  65,596, x4L 65,624); replayed chain end d 67,323, NOT past the greedy's 67,311. Envelope: no cell
  past x -904; the highest cell at the ledge face (x -884..-916) z 10,217 (x4L 10,257); cells at z
  >= 10,340 end at x -785. Null: the direction-blind key is not what keeps the step out.
* Why (v2's own greedy evals, MEASUREMENT only - never a recipe input, CLAUDE.md 0b; 193 episodes
  of 25 evals from 2.8B to 11.97B, `slot_approach.py`): the line leaves the kicker at |v_h|
  850-1,078 u/s (median 928) heading -x, peaks at z 10,911-11,006 (median 10,984) at x ~-228,
  weaves (S-turns, yaw 90-250 deg, |v_h| constant) over the shelf, reaches the hole at its FAR end,
  hits the end wall (x ~-704, z ~10,620: vx -> 0), slides down into the tunnel at vz -790..-860,
  then heads for the ledge at 500-700 u/s falling ~1,000 u/s and is down at z 10,100-10,300 by x
  -830..-900, short of or against the ledge face (40-240 u under its top). A ballistic estimate on the field's geometry (hole x ~-440..-720 in a 64 u
  shelf 10,528-10,592; ledge top 10,340 at x -912..-944; ducked hull): getting under the shelf
  before the end wall AND over the ledge in one flight needs >= ~1,150-1,200 u/s of -x speed
  (window vz -400..-430 there), ~1,450-1,700 u/s for a usable window (vz -520..-730, an apex
  ~170-330 u above the shelf ~0.7-0.9 s before the hole). The policy brings ~930 u/s from an apex
  ~390 u above the shelf only ~230 u before the hole: too slow and too high, too late.
* Reading: the post-CP2 block is a hidden SPEED gate - the potential field shows the hole and the
  slot but not that they need ~1.3-1.8x the speed the line carries, gained upstream on the ramps
  before the kicker. The uniform explore step from the line's last 4-16 s cannot add that speed
  (air strafing at ~930 u/s gains little; the speed comes from the surf ramps many seconds back),
  and neither repeat-p 0.90 nor a direction-aware key changes it. This is the CLAUDE.md 0b case
  ("speedup on ramps that are not shown by reward vector field"): the recipe's exploration has to
  value speed/energy itself well upstream of a wall, not only explore at it.

### 2026-11-09 06:59 UTC - CORRECTION: the post-CP2 slot is a precision gate with a speed floor, not a plain speed gate

The 06:46 entry's "hidden speed gate" rested on the 32 u goal field's picture (a hole through
the shelf). The simulator's own collision says otherwise. Measured locally with the core (duck
hull, `autoresearch/analysis/unr1b1_slot_*.py`; MEASUREMENT only, CLAUDE.md 0b):
* Geometry (ducked origin): shelf top ~10,590 (origin >= 10,608-10,612), the tunnel under it
  (origin <= 10,524-10,560 by y); a SLIT through the shelf only at |y| <~ 40, x -484..-704; a thin
  canopy plate (~10,754-10,766) over the slit (|y| <~ 80); a beam (origin 10,524-10,820) at x
  -420..-464 closes the slit's +x end; at x -708 the tunnel ceiling steps down to origin 10,524
  (the end wall above); the ledge top 10,322 (origin >= 10,340) at x -912..-948. A flight that
  clears the ledge and lands on the tunnel floor beyond is teleported back to the map start
  (sanity flights from inside the tunnel), so the route past the ledge is not the floor.
* bench2: 328 of v2's own approach positions (6 greedy episodes, kicker apex to end wall) x
  horizontal speed K 0.6-2.5 x heading +-60 deg x vz +-400 = 132,840 hands-off ducked flights:
  0 pass. The policy's own velocities: deepest x -768 (the end wall).
* bench3: synthetic starts over / in the slit (x -500..-650, |y| <= 80, z 10,580-10,700; -x
  speed 400-1,500, vy +-300, vz -600..+200): 45 of 9,600 pass, ALL at y 0 (in the slit), z
  10,580-10,620 (shelf height), x -500..-600, -x speed >= 1,000 u/s (0 at 400-800), vz -200..-600.
* Reading: the line must skim the shelf (<= ~10,620, i.e. inside the slit's near half, entered
  from the side under the canopy) with >= ~1,000 u/s toward -x. v2's line crosses that x range
  at 10,750-10,850 with ~900 u/s of -x speed, reaches the slit at its far end and hits the step.
  Speed is necessary but the altitude and the alignment are what it lacks most: a narrow
  window (0.5% of a box already centred on it), invisible to the potential field, which only
  says "go down here". That a uniform explore step from the line's last 4-16 s never finds it
  (x16, x4L, xR90, xH8) fits a needle of this size; the 06:46 "speed gate" sentence is withdrawn.

### 2026-11-09 08:06 UTC - the explore step from the line's own states never brings the conjunction (slot_reach); v2 at 13.57B

* v2: 13.57B at 07:59 UTC (178,980 steps/s); 181 evals, newest 13.55B at 32.45-32.47 (best
  32.476 at 6.30B); search 50 nothing (deepest chain end d 67,335), search 51 running: 75 searches
  since search 15's line, 90 in all. Post-CP2 deadline 13.93B: its last eval before (13.93B) lands
  ~08:33 UTC, the first eval past it (14.00B) ~08:40 UTC.
* `autoresearch/analysis/unr1b1_slot_reach.py` (local core, MEASUREMENT only, CLAUDE.md 0b): the
  explore step exactly as explore_phase1 draws it with v2's search flags (uniform over the whole
  action set incl. jump and duck, repeat 0.95 per 4-tick decision, view held at an offset of
  -90..90 deg from the velocity heading), 4 s bursts from 30 of v2's own greedy states (3 eval
  episodes of traj_13550747648, 0.15-2.5 s before the end wall), 3,000 bursts each = 90,000:
  0 pass the ledge. 17,919 cross x -560 (the slit's near half); the fastest of them at 988 u/s
  toward -x. 596 cross inside the slit's width (|y| < 48): from the starts 0.25-2.5 s out all at
  z >= 10,790 (over the canopy, ~200 u above the shelf); from the 0.15 s starts (already beside
  the slit) a few fall back across it under the shelf. 540 cross at z <= 10,700, all beside the
  slit. Exactly 1 of 90,000 is in the slit at shelf height (z 10,560-10,640) - at 189 u/s.
  88,575 bursts die within the 4 s (teleport floor, falls).
* Reading: the passing envelope is a conjunction - inside the slit's ~80 u width, at shelf
  height, >= ~1,000 u/s toward -x - and a random burst from the line reaches each coordinate on
  its own, never two of them with the third. From 2-2.5 s out the bursts mostly lose the
  approach altogether (0-545 of 3,000 even reach the slit's x). That fits 80M explorer runs and
  90 searches saturating on one envelope: the cells that would lead in are never generated, so a
  count- or energy-based selection has nothing to pick, whatever its constants (repeat 0.90 /
  0.95, 4 s / 16 s roots, heading key). What it says for a generic recipe: at a precision gate
  the search has to change the approach seconds upstream and keep the conjunction together; a
  uniform explore step bolted to the policy's own line does not.

### 2026-11-09 08:52 UTC - unr1b1_v2 FAILED the post-CP2 block; stopped at 14.10B, harvested, box released

* ladder.py on the pull at 08:46 UTC: post-CP2 block (33%) FAILED - deadline 13,931,380,736
  (October 6,965,690,368); the last eval before it (13,928,235,008) 32.463, the first past it
  (14,003,732,480) 32.434, both CP2, 0/9 finishes. First ramp 227M, window (CP1) 453.5M, pit
  dive 906.5M, cavity (CP2) 2.341B stand, all ON TIME. Best eval unchanged: 32.476 at 6.30B.
* Stopped per operator rule (2) at 08:48:53 UTC: `runs/unr1b1_v2.no_autoresume` touched, trainer
  pid 24267 killed (no signal handler: ckpt_latest.pt is the last periodic save), last progress
  row 14,099,152,896; no search or explorer left (search 54 had ended at 14.09B: nothing, d
  67,318). 94 searches in all (40 in the 0-6B segment, 54 after the cap resume); the last line
  came from search 15 at 2.20B, so 79 searches since found none.
* Harvest (md5-checked): `ckpt_latest.pt` (global_step 14,093,385,728; md5
  4cca3059230cfb3328eecd7cdf2a9941) and `ckpt_5001707520.pt` (the newest ckpt_<step>.pt at or
  before the best eval; md5 fe8116f1dc74c313a9967fd3be470aef; ckpt_final.pt = 6.0B was already
  local, md5 166c3865...) with 188 eval trajectories, progress.csv and run.json
  (`tools/pull_run_mss.py`); the search records without the field / archive copies
  (`runs/unr1b1/xtest_box/unr1b1_v2_records.tgz`, 878 entries, md5 c3e5e97f...) and the explorer
  tests (`unr1b1_xtest.tgz`, 43 entries, md5 7bfc9468...). The harvest script's own md5 compare
  also matched its `ls` line and refused both tarballs; re-checked by hand, both match the box.
  Newest eval 14,079,229,952: 32.462.
* Box released at 08:51:37 UTC (`box.py status`: no box). No new recipe is launched: ~1.9B steps
  were left before the 11:55 wrap-up, and v2 itself needed 2.34B to reach the cavity, so no recipe
  started now could get past v2's rung; what is left of the round goes to local analysis.

### 2026-11-09 09:32 UTC - an optimizer over the explore step's actions cannot reach the slot from the line's last 4 s either

`autoresearch/analysis/unr1b1_slot_cem.py` (local core, MEASUREMENT only, CLAUDE.md 0b): a
cross-entropy search (3,000 rollouts x 20 iterations per start, 4% elite, smoothing 0.6) over the
explore step's own action set, per 4-tick decision, from v2's greedy states (ep 0 of
traj_13550747648) T s before the end wall. Target: the passing envelope the hands-off bench
measured (at the x -560 crossing: |y| < 30, z 10,580-10,620, >= 1,000 u/s toward -x) or a pass
(alive past the ledge). 384 iterations, 1.15M rollouts in all: 0 passes, 0 crossings in the slit
at shelf height (|y| < 48, z <= 10,640).
* The view held at an offset from the velocity heading (explore_phase1's), uniform start: from
  0.5-2 s out the best crossing is in the slit's width (y 22-56) but 170-245 u too high (z
  10,789-10,864); from 3-6 s out no rollout even gets back to x -560 (random keys fall off the
  surf ramps).
* The LINE'S OWN per-tick view + a held delta (+-3..30 deg) and its keys (80% at the policy's
  choice to start): replaying the line's actions open loop, aligned to the policy's decision
  phase (its keys change on tick % 4 == 0), reproduces the line from 1-3 s out (it hits the end
  wall at x -704/-707) but not from 6 s. Best per start: 1 s y -93 z 10,753 858 u/s; 2 s y 30
  z 10,790 854 u/s; 3 s y -185 z 10,621 743 u/s; 4 s y -269 z 10,600 677 u/s; 5 s loses the line.
  The search finds the slit's width only 130-170 u too high, or shelf height only 180-270 u
  beside the slit and 250-320 u/s short - never the two together, let alone with the speed.
* (Two first variants - a depth score, and a key-seeded start - ran before the phase fix; both
  0 passes, stuck on the end wall at x -754..-770.)
* Reading: the slot cannot be reached by changing the line within its last ~4 s, by random
  bursts (90,000) or by an optimiser (1.15M rollouts) over the explore step's actions. Its entry
  is set up further upstream, where an open-loop search loses the line, so a search rooted in the
  last 4-16 s before the stall (the --ge-line roots) has nothing to find there. Generic consequence
  for the next recipe: the search must branch where the line still has the freedom (seconds
  before the stall) and carry a closed-loop controller through the surf in between - the
  explorer's cells only remember open-loop chains.
