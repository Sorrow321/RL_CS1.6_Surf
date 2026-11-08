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

### unr1b4 (round 1, slot 4: the knob as one rule) - the plan (2026-11-08 04:59 UTC)

* **Why not in-trainer bursts.** The cheapest form of slot 4 (the trainer's own envs as the
  explorer: a coherent per-episode coin between a held random burst and the policy after every
  own-state respawn) is October's ez-greedy with a coin, and October measured ez bursts on this
  map three times (r1b1_ez2 everywhere, r1b1_novezf and r3b3_ezf at the frontier): nulls -
  "bursts kill episodes", "the reservoir never keeps them", "a burst cannot change altitude in
  the air" (`/home/steam/ar/archive/r7/r7b2/NOTES.md`). What they lacked is what Go-Explore
  adds: an archive that KEEPS every new place with the exact actions that reached it, and a
  curriculum that teaches the line. So the knob lives in a real archive search.
* **The explorer** (`tools/knob_ge.py`, run by `tools/beam_tas.py --ge-knob` on the
  checkpoint-faithful wrappers, called by `tools/auto_spine.py --knob`): roots = the policy's
  own greedy line from the true start, its last 4 s before it stalls (first tick within 500 u of
  its best d); cells = 64 u position x 64 u kinetic-energy buckets, CONTACT states only (ground
  or a surf face: a ballistic apex is not a place to continue from); W = 1/sqrt(1 + chosen);
  restore exactly (clock zeroed, the scalar obs row stored with the cell); runs of 100
  decisions (4 s), each committed to RANDOM (uniform engine keys + a yaw target drawn in the
  policy's velocity frame through its own off_warp, held, redrawn with p 0.05 per decision) or
  POLICY (its own samples, T = 1). **The knob** = the random share lambda = probability
  matching on the new-cell yield per decision of each mode (decayed over ~one population of
  run ends), 0.5 until 256 ends per mode, clipped to [0.05, 0.95]. The deepest cell's chain
  (each link = the run's engine + view rows) is replayed open-loop from its root and checked at
  every junction (origin within 1 u), then spliced onto the greedy line.
* **The loop in the trainer** (one recipe, frozen): `--auto-spine --auto-spine-scratch` (NEW:
  no starting spine; the first search after `--auto-spine-patience` steps from the first eval;
  nothing drawn, paid or matched before) `--auto-spine-knob` (NEW: the search is the knob
  explorer) `--demo-share 0.25` (NEW: the spine curriculum takes 25% of the spawn pool and the
  recipe's own pool - reservoir, novel, greedy, perturbed rows - keeps the rest, instead of the
  default rule that REPLACES the whole reservoir share; so the run differs from the control by
  the search-fed share and nothing else) + the pb/zl curriculum constants (`--demo-window 5
  --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5 --demo-front-frac 0.67 --demo-success-d
  100 --demo-success-bonus 50 --auto-spine-milestones --auto-spine-async --auto-spine-recheck`).
* Flags off = bit-identical by construction (DEMO_ON = demo_file, DEMO_LIVE true, DEMO_SHARE
  None -> the old paths); config keys dumped only when on; TRAIN_ONLY in record_ckpt.
  tests/python/test_knob_ge.py (11 pass with test_auto_spine_recheck), test_auto_spine.py and
  test_demo_success_d.py still pass. Commit caa7f81b.
* Next: a box; smoke the explorer alone on an early scratch checkpoint (does it run, does the
  knob move, does the chain replay exactly, does it find cells past the greedy line), smoke the
  scratch bootstrap (a search fires on patience and its spine goes live), then the run.
* 05:08 UTC: box **59667168** (RTX 3090, 0.28 $/h, 3 h deadline). Smoke `unr1b4_smoke` (the
  full recipe with `--auto-spine-patience 2e7 --auto-spine-retry 2e7 --auto-spine-ge-minutes 2`,
  evals every 10M): the record gate passes with the new flags (greedy, stoch, mixed, POV). First
  launch refused: `--respawn-novel` and `--respawn-backward-mix` refused any spine curriculum;
  under `--demo-share` the recipe's own pool is still built, so they stay wired (c8a0e0bd).
* **Smoke result (05:05-05:13 UTC, `unr1b4_smoke`, analysis of the mechanics only).** The first
  search fired on patience at 21.5M (the greedy dies on the start platform at 4.6 s, best d
  97,748 of 97,768): in 129 s (2 min budget) the explorer built **6,402 contact cells from 37
  roots over 368,608 runs** (2,048 envs; ~27 decisions per run, most die fast), cells by mode
  policy 2,750 / random 3,615 over 4.0M / 5.8M decisions (yield 6.9e-4 vs 6.2e-4 per decision:
  a tie); best cell d **96,845 = 903 u past the greedy line**, chain of 5 links (modes random,
  random, policy, random, policy) from root 97, **replayed exactly (0 u junction error)**. The
  trainer adopted it ("NEW SPINE from knob ... the FIRST spine is live"), the curriculum drew
  from it at once (band success 100%, spine-line bonuses paid), and the recipe's own pool stayed
  intact beside the 25% share (novel 737 rows, backward-mix 369, greedy 922). Training 181k
  steps/s on this 3090 after the search. **Defect fixed**: the knob's yield sums decayed over ONE
  population of run ends, and lambda swung 0.55 -> 0.18 -> 0.52 -> 0.95 in 2 min while the two
  modes' totals stayed within 10%: now 16 populations (98b702cf).
* **The run: `unr1b4_knob`, launched 05:14 UTC from step 0, commit 98b702cf**, box 59667168. Extra
  flags after `config.incumbent_recipe`: `--auto-spine --auto-spine-scratch --auto-spine-knob
  --auto-spine-async --auto-spine-recheck --auto-spine-milestones --auto-spine-patience 3e8
  --auto-spine-envs 2048 --auto-spine-ge-minutes 10 --demo-share 0.25 --demo-window 5
  --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5 --demo-front-frac 0.67 --demo-success-d 100
  --demo-success-bonus 50 --ckpt-every 2.5e8`, `--steps 1e10`, evals every 75M. Nothing changes
  after launch; the first search comes ~300M steps after the first eval.
* **05:46 UTC, unr1b4_knob search 1 (the first real one, at the 302.5M eval, 611 s).** Evals 76M ..
  378M all read 1.7% (the control `unop_ctl` too, through 680M: the start-platform wall, before
  the first ramp). The explorer's greedy line: 332 ticks (3.3 s), best d 97,972 (1.7%), stall at
  296; 75 roots -> 36 root cells. In 10 min: **15,674 cells from 1.35M runs, deepest d 94,144 =
  5.55%** (the line 1.7%), reached by 61 s at 94,780 and flat from ~7 min (yield per decision
  0.0021-0.0026 at 1 min, ~1e-4 from 6 min on: the archive saturates in reach of these roots).
  **The knob stayed at 0.52-0.64** the whole search (random found cells ~1.2-1.5x faster per
  decision; cells born policy 4,054 / random 11,584). The chain: 13 links, modes R P P R P P P P
  R R P P R (8 policy, 5 random), replayed exactly (0 u). Adopted at 393M: the first spine (894
  states, end d 94,115, success line 94,215). Training ran 251-254k steps/s beside the search
  on this 3090 (GPU 97%, explorer 1.8 GB): the search is nearly free here.
* 05:44 UTC: the explorer-only knob ablation on search 1's checkpoint (`runs/unr1b4_abl/` on the
  box: the rule, lambda fixed 0 = all policy, lambda fixed 1 = all random; 10 min each, in
  parallel so they share the contention; same roots and seeds).
* **05:55 UTC - THE KNOB ABLATION (explorer only, no training; search 1's checkpoint at 302.5M,
  the 1.7% start-platform wall; the same greedy line, roots, seeds and 10 min budget, the three
  arms in parallel so they shared the contention; ~5.7-6.3M decisions each):**

  | arm | cells | deepest d | progress | cells / 1k decisions (by mode) | best chain |
  |---|---|---|---|---|---|
  | **the rule** (lambda 0.52-0.62 throughout) | **11,888** | **94,760** | **4.93%** | policy 1.54, random 2.34 | 11 links: R P P R P P P P R R P |
  | lambda 0 (all policy) | 3,018 | 97,908 | 1.77% | policy 0.50 | 10 policy links |
  | lambda 1 (all random) | 6,440 | 96,924 | 2.76% (flat from ~2 min) | random 1.01 | 2 random links |

  (the greedy line: 1.70%; the in-run search 1, same rule with ~3x less contention: 5.55%.)
  **The middle of the knob beats both ends, and by a lot: 3.9x / 1.8x the cells, 3.2 / 2.2
  points more progress at matched decisions.** It is a synergy, not an average: inside one
  archive the policy's yield per decision is 3.1x its yield alone (it continues FORWARD from
  the new contact states the random runs reach sideways) and the random runs' is 2.3x theirs
  alone (they branch from the places the policy carried them to); the deepest chains alternate
  the modes. Alone, the policy repeats its own fall and random saturates within 2 minutes.
  The yield rule kept lambda in the middle by itself (the two modes' yields stayed within
  1.5x), which is where it should be here. One seed per arm, one wall: a probe, not a law.
  Cost noted: three explorers beside the trainer cut it from ~250k to ~25k steps/s for 10 min
  (CPU threads oversubscribed); one at a time from now on.
* **06:03 UTC - FIRST RAMP CLEARED at 529.0M (score 8.354: best 8.3%, median 7.4%), on time
  (deadline 1.0B).** The control `unop_ctl` (same recipe, no search) still reads 1.7% at 604M
  and 680M. The loop did it in two moves: search 1 (302.5M) took the archive from the greedy's
  1.7% to 5.55%; the spine went live at 393M; the curriculum's band walked it back from row 893
  to 713 by 453M (greedy 2.6%) and to 368 by ~505M (30% band success, 68.7k spine-line bonuses,
  training reward 6 -> 30/ep); at 529M the greedy flew the spine and past it (7.4-8.3%), the
  quorum fired (search 2 at 529M, from the new line). So the knob explorer + curriculum broke the
  wall the bare recipe has sat at since 76M. One seed; but the control has been flat for 600M.
* v2 is ready and NOT running (adccf082, `--auto-spine-knob-archive`: one archive for the whole
  run instead of a fresh one per search; unit-tested). Built while v1 looked slow; v1 is on time,
  so it keeps the box.
* **06:15 UTC - search 2 (529M weights, quorum-triggered) was STALE.** Its greedy line: 9.7 s,
  best d 91,558 (8.1%), stall 882, 89 root cells. In 10 min: 43,001 cells over 880,657 runs;
  the deepest d 90,117 (9.6%) by 33 s and 90,088 at the end (flat from ~1 min; yields 1.1e-2 ->
  3e-4 per decision); knob 0.58-0.66 (random ~1.5x the yield; cells policy 14,386 / random
  28,526 over 14.2M / 19.4M decisions); chain 12 links, 5 random + 7 policy, exact replay. But by
  604M the greedy alone was at 11.9% median (12.4% cover) - past the spine's 9.6% - so
  `--auto-spine-recheck` discarded it (next search >= 916.6M, the 3e8 retry). Past the start wall
  the recipe's own RL outruns a 10-min search; the search pays where the policy sits still
  (the 1.7% wall: 450M steps).
