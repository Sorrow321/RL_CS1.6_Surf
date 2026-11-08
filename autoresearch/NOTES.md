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

### unr1b3 (round 1, slot 3: policy-prior MCTS over macro-actions) - 2026-11-08 05:16 UTC

**Box** 59668027 (RTX 5080, 32 threads, $0.322/h, Minnesota), rented 05:09 UTC.

**What was built (commits cf8bb9ce, 7b69c1f2 on ar/unr1b3)**
* `tools/mcts_search.py` + `tools/mcts_policy.py`, driven by `tools/beam_tas.py --mcts` (beam_tas
  builds the checkpoint-faithful core, lidar, wrappers and side feeds; the MCTS reuses them, and
  its best line goes through beam_tas's own `--save-frontier` open-loop replay).
  * Node = exact sim state + EVERY per-env inference state a wrapper reads off past decisions
    (held keys + their tick, compass d0 + prev tick, surf tracker, ratchet record, obs-reward feed,
    latch), stored row-wise (RowStore / SideState) and restored into any env slot. Restore check:
    on CPU the greedy children repeat the greedy pass's engine rows 24/24 with 0.0000 deg; on the
    5080 339/504 (max 3.8 deg) - a 1-env vs 4096-env batch gives the GPU forward different float
    results, so a near-tie argmax flips (lines stay exact: they are replayed open-loop).
  * Edge = macro of M = 12 decisions (0.48 s): child 0 the policy's mode, the others samples at
    keys temperature 1.5 with a held yaw offset ~ N(0, 20 deg) (view targets are in the velocity
    frame, so a held offset is a held steer), kept from the parent with p 0.5 (coherent steering).
  * Selection: PUCT, P = 1/children (sampled children: Sampled MuZero's uniform prior over policy
    samples), Q = max backup of CONTACT-BANKED progress (best d at a tick the map pushed back;
    a capped-speed fall is not contact), min-max normalised; the exploration term weighted by
    Go-Explore's 1/sqrt(1 + visits) of the child's end cell RELATIVE among siblings; batched
    with virtual visits, a subtree whose leaves are all pending is blocked for the wave.
  * Roots: the policy's own greedy line (1-env pass from the true spawn, stopped at death or 10 s
    without a 32 u gain), a root every 12 decisions over its last 20 s.
* Trainer: `--auto-spine-scratch` (the first spine is the map spawn, 2 rows, written to
  runs/<run>/auto/spine0.npy: the loop runs from step 0 with no picked file), `--demo-share F`
  (the spine curriculum's draws take F of every pool on top of the recipe's own sources, so it runs
  beside --respawn-novel / --respawn-backward-mix, which --demo-file refused), `--auto-spine-explorer
  mcts` + `--auto-spine-mcts-minutes`. All default off, TRAIN_ONLY in record_ckpt.
  **Flag-off bit-identity**: the full SCRATCH preset + incumbent recipe on CPU, 3 iterations, the
  base commit's trainer vs this one: run.json config identical, progress.csv identical (fps aside).

**First explorer-alone test (old selection, 05:11-05:20 UTC)**: r0 (October incumbent r0, greedy
from seed 0 dies at 23.75 s, banked 23.9%): 500 waves, 2.05M nodes, only 4,318 cells, best 24.99%,
26-217 leaves per wave instead of 256. Two design faults: (1) a batch kept descending into the
deepest subtree and failing once its leaves were pending (fixed: per-wave blocking); (2) the cell
weight multiplied the exploration term by 1/sqrt(1 + visits), and with ~49k cell visits per wave it
went to ~0: pure exploitation (fixed: relative among siblings). c_puct 1 -> 2.

**Explorer-alone, round 2 (05:20-05:34 UTC, box 59668027, all lines graded with score.py's field
via autoresearch/analysis/mcts_line_report.py; the walls: pit dive > 27.5%, cavity = sgcp_2)**

| explorer | ckpt | search | best line (score.py) | CP | note |
|---|---|---|---|---|---|
| greedy (seed-0 spawn) | r0 | - | 23.9% (banked, beam_tas field) | 1 | dies at 23.75 s; r0's eval best 26.9% |
| MCTS v1 (old selection) | r0 | 8 min, 3.8M nodes, 4,478 cells | 24.63% | 1 | 52.6 s crawl |
| MCTS v2 (blocking, relative W, c 2) | r0 | 6 min, 5.7M nodes, 11,358 cells | 25.98% | 1 | plateau after ~200 waves |
| plain beam (auto_spine's: 4096, R 25, greedy 32) | r0 | 11 s | **27.52%** | 1 | just past the pit dive |
| plain beam | r1 | 140 s | 28.12% | 1 | r1's eval best 28.2%; no sgcp_2 |

* The MCTS as configured LOSES to the plain beam at the pit dive: 6 min for 25.98% vs 11 s for
  27.52%. The tell: only 15-80% of children survive one 0.48 s macro - keys temperature 1.5 plus a
  held 20 deg steer on EVERY sampled child is too disruptive at speed (the documented "per-decision
  epsilon kills beams at speed" lesson, here as a held perturbation on all children). Fix under test:
  --mcts-perturb-frac 0.5 (half the sampled children are the policy's own samples at temperature 1,
  the paper's 50/50 random-vs-policy commitment) + --mcts-novelty 1.
* r1's greedy pass first stopped at 13 s (the policy idles at the spawn > 10 s): the stall window is
  now 30 s (the recipe's eval patience).

**Explorer-alone, round 3 (05:34-05:50 UTC)**
* The beam's "27.52%" on r0 is MID-AIR: its line reaches min d on its last tick, 1.16 s after its
  last map contact; BANKED (best d at a contact tick) it is 25.83%. r1's beam: 28.12% raw, 26.10%
  banked. On banked progress the MCTS v2 line (25.98%) matches the beam; nothing banked a landing
  past the pit dive. (score.py grades raw min d, so a fall counts there.)
* r0, 6 min each: --mcts-novelty 1 -> 26.21% banked, 13,459 cells (the most); --mcts-perturb-frac
  0.5 + novelty 1 -> 26.18%, 9,980 cells. All variants plateau by wave ~200; inside noise.
* r1 from the seed-0 spawn IDLES at the start (greedy best 1.68% after 33 s), so its test became a
  first-ramp test: MCTS (pf 0.5, nov 1) 6.71% (the first ramp is 8%), the beam from the same start
  26.10% banked. Allocation is the tell: ~93% of children die per wave, and 256 leaves spread over
  the tree leave few survivors at the deepest frontier, where the beam puts all 4,096 copies.
* c_puct is the lever: r1 from the idle start, 4 min, pf 0.5 + nov 1: c_puct 2 -> 6.71% (depth 12,
  5,947 cells), 128 children per leaf -> 5.86% (worse: concentration alone does not help),
  **c_puct 0.25 -> 15.57% (depth 25, 14,875 cells)**, past the first ramp. With c 2 the exploration
  term swamps the small progress differences between siblings and the tree goes broad and shallow;
  the beam's truncation selection is the c -> 0 end of the same knob. Still below the beam's 26.1%
  banked from the same start, and every variant plateaus after ~300 waves.

### unr1b3_mcts1 - the from-scratch run (launched 2026-11-08 06:00 UTC, box 59668027, commit 1524fea3)
`box.py launch --run unr1b3_mcts1 --steps 2e10 --` the incumbent recipe +
`--auto-spine --auto-spine-scratch --auto-spine-async --auto-spine-recheck --auto-spine-milestones
--auto-spine-explorer mcts --auto-spine-mcts-minutes 5 --auto-spine-envs 2048 --auto-spine-ge-minutes 0
--auto-spine-patience 3e8 --auto-spine-retry 3e8 --demo-share 0.25 --demo-success-d 100
--demo-success-bonus 50 --demo-window 5 --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5
--demo-rate 0.2 --demo-min-ep 50 --demo-front-frac 0.5` (the curriculum constants are za_loop E2's /
uf2r1b1 A13's; the MCTS runs at beam_tas's --mcts defaults of this commit: M 12, K 16, c_puct 0.25,
novelty 1, perturb-frac 0.5, keys temp 1.5, yaw sigma 20 deg, hold-p 0.5, cells 128 u x 256 u energy,
roots every 12 decisions over the greedy line's last 20 s). Nothing changes after launch: every
search, cut and adoption is the trainer's rule on its own state. Provenance of every spawn file:
the map spawn (spine0.npy) and the run's own searches on its own weights. Record gate passed (23 s).
Search 1 fired at the first eval (524k steps, 9/9 past the spawn line, greedy frontier d 98,903).
* **Search 1** (524k-102M steps, 305 s beside training): the untrained greedy idles at the spawn
  (stall at 34 s, best 0.28%); 15 roots in its first 7 s; 620 waves, 1.27M nodes but only **164
  cells**, best 1.77% at 47 s (depth 91 - deep in time, not in space), replay exact. auto_spine
  adopted it against the frontier the search started from (98,903 -> line 97,913, gain 990 u), the
  trainer's --auto-spine-recheck discarded it: the policy itself had reached 97,964 meanwhile. Next
  try at 402M. An explorer whose children are samples of an untrained policy explores like that
  policy: it idles. The trainer: ~281k steps/s on the 5080 (with the search beside it).

### 06:30-06:40 UTC: both from-scratch runs stall at the spawn; the first ramp is the battleground
* **The control FAILED the first ramp** (1.7% at 1.06B, $AR_HOME/control LADDER: STOP). unr1b3_mcts1
  sits at the same 1.6-1.7% through 0.68B (9 evals). Its 9 eval episodes are IDENTICAL: straight
  east off the spawn platform (7837, 0, 15042) -> fall -> death at 3.0 s at (8,933, 252, 13,836),
  1.61%. The policy learned the nearest field gradient into a kill; the first ramp needs a left turn.
* Search 2 (402-549M): no spine (MCTS 97,910 vs greedy 98,083: 173 u < the 200 u gain).
* Explorer-alone on the run's OWN 0.7B checkpoint, in a separate worktree on the box (/root/t2) so
  the run's checkout (and so its searches' code) stays at the launch commit: --mcts-random-frac 0.5
  (random children: first draw at temperature 5 on every head, held for the macro) -> 1.85%, 981
  cells, depth 5 after 230 waves: the 2.9 s greedy line gives only 5 roots (0.5-2.4 s) and most
  children fall off the platform.
* Explorer-alone on the run's 0.7B checkpoint, 4 min each (2048 envs, beside the training):
  beam 1.78% (dies with the policy); MCTS as in the run 1.77%; MCTS --mcts-random-frac 0.5
  (M 12) **5.08%** (3,210 cells, still rising); random 1.0 + M 25 (1 s) 1.77% (2,187 cells);
  random 0.5 + M 25 3.93%. The first separation from the beam: at a deadly start the policy's
  samples die with it, random held children (mixed 50/50 with policy samples) keep exploring;
  all-random is worse than the mix, and 0.48 s macros beat 1 s ones. None reached 8% in 4 min.
* **unr1b3_mcts1 cleared the first ramp at 906M: 10.6% (best AND median), ON TIME (deadline 1.0B)**
  - the control failed this wall (1.7% at 1.06B). NOT the MCTS's credit: no search spine had been
  adopted (search 1 stale, search 2 below the gain), so the run differs from the control only by
  --demo-share 0.25 of spawns at the map spawn (the spawn spine), the searches' GPU contention, a
  5080 instead of a 5060 Ti and the seed. Search 3 started at 906M from a 10.6% greedy frontier.
* Search 3 (906-986M): greedy pass dies at 19.9 s (banked 9.5%); 29 roots, 1,087 waves, 2.2M nodes,
  8,887 cells, best 12.70% (d 87,021) vs the frontier it started from (89,131): +2,110 u - and STALE
  on arrival: the policy's 982M eval was already at 83,957 (15.8%). While the policy improves this
  fast (1.7% -> 10.6% -> 15.8% in 150M steps) a 5-min search cannot lead it; the search's job is
  the plateau. Restore check on the 5080: 345/348 decisions.
