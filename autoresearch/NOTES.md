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
* **06:13 UTC - search 2 (529M weights, quorum-triggered) was STALE.** Its greedy line: 9.7 s,
  best d 91,558 (8.1%), stall 882, 89 root cells. In 10 min: 43,001 cells over 880,657 runs;
  the deepest d 90,117 (9.6%) by 33 s and 90,088 at the end (flat from ~1 min; yields 1.1e-2 ->
  3e-4 per decision); knob 0.58-0.66 (random ~1.5x the yield; cells policy 14,386 / random
  28,526 over 14.2M / 19.4M decisions); chain 12 links, 5 random + 7 policy, exact replay. But by
  604M the greedy alone was at 11.9% median (12.4% cover) - past the spine's 9.6% - so
  `--auto-spine-recheck` discarded it (next search >= 916.6M, the 3e8 retry). Past the start wall
  the recipe's own RL outruns a 10-min search; the search pays where the policy sits still
  (the 1.7% wall: 450M steps).
* **06:26 UTC - evals since the first ramp** (score.py best / median): 604M 12.8 / 11.9, 680M
  15.8 / 15.7, 755M 17.3 / 14.9, **831M 18.76 / 15.3**. The control `unop_ctl` (the same recipe
  without the search) **FAILED the first ramp**: 1.6-1.7% at every eval to 1,057M (deadline 1.0B).
  At matched steps the difference is the whole of it: 8.3% vs 1.7% at 529M, 18.8% vs 1.7% at 831M.
* **06:46 UTC - search 3 (982M, patience) -> spine 2.** The greedy stalled at ~18.7% for three
  evals (831M 18.8, 906M 18.7, 982M ~18.7; frontier 18,605-18,681 u). The explorer's OWN greedy
  rollout of the 982M weights died at 6.6 s (2.0%) while the trainer's 9 eval episodes flew
  15.8-18.6%: the recorder/beam_tas run the policy in fp32 ("the recorder never enters autocast"),
  the trainer's rollout in bf16, and a chaotic map turns that into another line within seconds
  (searches 1-2 happened to agree: 1.7% / 8.1%). So its roots sat on the start platform. The
  archive got to 20.3% within 32 s anyway, by cascading policy runs along the policy's own route
  (98,709 cells, 787k runs; knob 0.53-0.57, random ~1.2x the yield; cells policy 41,829 / random
  56,836). The deepest chain: 10 links, ALL POLICY, 2,408 ticks from root 236, d 79,296 (20.4%),
  exact replay. Adopted at 1,071M: spine 2 (2,083 states, end d 79,137 = 20.6%, success line
  79,237; greedy frontier 83,966). Where the policy can already fly a stretch, policy chains
  carry the depth and random adds breadth; at the 1.7% wall the depth needed both (above).
  Defect to fix in a later recipe (not this run): roots should come from the trainer's own eval
  states (the greedy-line pool keeps them exactly), not from a fp32 re-rollout.
* 1,057M eval: frontier 20,167 u (~20.2%, moving again). 06:46: the explorer-only arms on search
  3's checkpoint, ONE AT A TIME beside the trainer (lambda 0 then 1, 10 min each, same roots and
  seeds; the rule arm is search 3 itself, which ran alone beside the trainer too).
* CORRECTION to the search-3 note: the trainer's greedy eval ALSO runs `GreedyTorchPolicy` in fp32
  (EVAL_GREEDY, train_fast.py ~13894; 9 envs on its eval core), so "bf16 vs fp32" is not the
  cause. What differs is the batch (9 eval envs vs the explorer's 1-env rollout core) and so
  the GPU kernels' numerics; on a chaotic map that is enough for another line. The fix stands:
  root the search on the eval's own states.
* **07:11 UTC - the knob ablation, second checkpoint (search 3's, 982M; roots on the start
  platform because its re-rolled greedy died at 2.0%; 10 min each, each explorer alone beside
  the trainer):** the rule (search 3 itself) 98,709 cells, deepest 20.4%; **lambda 0 (all
  policy) 74,690 cells, the SAME 20.4%** (d 79,296, a 10-link all-policy chain); lambda 1 (all
  random) 6,728 cells, 2.8%. So over the two checkpoints the rule's mix is never worse than the
  better pure end and, at the wall where the policy itself was stuck (1.7%), 2.8x deeper than it:
  where the policy can fly a stretch the policy chains carry the depth (the mix keeps ~45% of
  the runs on them and adds breadth: 1.3x the cells), and where it cannot, random branches give
  the policy somewhere new to continue from. Pure random never gets past ~3% at speed. Answer
  to the user's question so far: the policy-guided side is not useless - it is what goes deep
  - but alone it repeats its own fall; the knob belongs in the middle, and the yield rule puts
  it there (0.52-0.66 in all 4 searches) without being told.
* **07:13 UTC - search 4 (1,208M, quorum: the greedy flew spine 2 to 20.6%) -> spine 3.** Its
  re-rolled greedy matched the eval this time (20.6 s, best d 79,130 = 20.6%, stall 1,901, roots
  [1,501, 1,901]). 21,040 cells, 674,719 runs, knob 0.56-0.64 (cells policy 6,190 / random 14,773
  over 9.6M / 13.4M decisions: random 1.6x the yield). **The deepest chain: 14 links, 11 RANDOM
  + 3 policy**, 1,368 ticks from root 1,889, d 76,076 = **23.7%** (the October window stall sat
  at 20.7-21%), exact replay. Adopted at 1,269M (3,256 states, success line 76,173). The pattern
  over the four searches: where the policy flies (search 3) the chain is all policy; at a wall
  that wants a detour (the start platform, the window) the chain is a mix weighted to random
  links - and the yield rule never had to be told which case it was in.
* **07:31 UTC - v1 STOPPED at 1,473M; v2 launched.** v1's evals 1,133-1,435M all read 20.6-20.7%
  (best **20.692 at 1,284M**; pulled: ckpt_latest, ckpt_1000341504, ckpt_1250426880 - the hand-on
  for 20.692 is ckpt_1250426880). **Why it stalled - a recipe bug, not the explorer:** spine 3
  passes the window curtain sgcp_1 (rows 1,995-2,001; the greedy stalls ~row 1,900), then wanders
  slowly in a low-energy pocket and STANDS STILL on the ground at rows 2,815-2,835 (|v| = 0, one
  origin). DemoCurriculum matches a spawn to its spine row by origin and keeps the LAST row of a
  repeated origin, so every spawn on the band [2,820, 2,824] was credited to row 2,835: the band
  showed "0 eps" for 100+ iterations and tau froze at 2,820 - the curriculum never spawned the
  stretch between the greedy's stall and the window. The fix exists (`--demo-match-band`,
  pbr6b4); random explore links stop the player more often than a beam does, so a knob spine
  needs it more. v1's next search could not come before ~1.59B (patience), the window deadline
  is 2.07B, and the same defect would recur at every slow stretch (the cavity), so the run that
  round 2 may continue should carry the fix.
* **v2 = `unr1b4_knob2`, launched 07:29:48 UTC from step 0, commit 9c8be647** - v1's flags plus
  four generic fixes, nothing tuned on the map: `--demo-match-band` (the standstill bug);
  `--auto-spine-knob-evalroots` (NEW: the search roots on the exact states of the eval's median
  episode as the greedy-line pool captured them, instead of a 1-env re-rollout whose GPU numerics
  differ from the eval's batch: v1's search 3 re-rolled a line that died at 2.0% while the eval
  flew 15.8-18.6%); `--auto-spine-knob-archive` (NEW: ONE archive for the whole run - cells,
  their root lines, the runs that reached them - so a hard wall gets the sum of several searches,
  as Go-Explore intends); `--auto-spine-deeper` (NEW: a spine still being taught is replaced only
  by a deeper one, or a persistent archive handing back its same best line would restart the
  walk-back). The v2 explorer paths were smoked on the box from a separate worktree beside v1
  (an eval-style line as roots; a second search loaded the first's archive: 2,290 cells, 1 line,
  1,119 runs -> saved 2,339 cells, 2 lines). Record gate passed.
* **08:16 UTC - v2 so far.** To 302.5M it is v1 bit for bit (the evals at 76/151/227/302M read
  the same fwd/path/track/frontier to the unit: the fixes are off-path until a spine exists), so
  from there it is a clean A/B on the fixes. Search 1 (302.5M, rooted on eval episode 4: 332
  ticks, the same line v1 re-rolled) -> 5.5%, spine 1 live at 387M (v1: 393M). **453M: 6.85
  (v1 2.6% at 453M)**; quorum -> search 2 at 453M (v1: 529M). **First ramp at 529M (8.131), on
  time**, like v1. Search 2: roots on the eval's own line (7.3 s, d 93,823), the archive carried
  over (15,130 cells, 1 line, 7,310 runs) + 74 new root cells -> saved 37,226 cells over 2 lines;
  best d 90,065 (9.6%), chain 7 links P P R P R R R from root 552 of line 2, exact replay;
  ADOPTED at 544M as spine 2 (v1's search 2 at 529M came back stale). Every v2 path has now run
  inside the trainer.
* **08:46 UTC - v2 STOPPED at 880M (15.66 at 831M); v3 launched.** v2's spines 1-3 came quickly
  (387M, 544M, 698M; v1 had one by 698M) and the evals tracked v1 (604M 9.9 vs 12.8, 680M 15.1
  vs 15.8, 755M 15.6 vs 17.3, 831M 15.7 vs 18.8). Then **search 4 (755M) found NOTHING**: no
  cell deeper than the eval line's own best (d 84,126). Its log: 170 s before the first report,
  and only 7.6k new cells in 10 min (fresh-archive searches add 20-90k). **The persistent archive
  is a null with count-based selection:** W = 1/sqrt(1 + chosen) over the whole archive gave the
  86k carried cells (chosen ~37x each, W ~0.16: total ~14k) about 99% of the selections and the
  75 new roots at the frontier (W = 1: total 75) about 1% - the archive drowned its own frontier,
  and it grows with every search. Go-Explore's archive persists, but with Atari-sized archives
  and a frontier that is new cells; here every search re-roots at the deployed line, which the
  old cells do not know. (A fix would select by recency or a frontier band; not built - v1's
  per-search archive is what cracked the start platform and the window.)
* **v3 = `unr1b4_knob3`, launched 08:45:26 UTC from step 0, commit 9c8be647 (flags only):** v2
  without `--auto-spine-knob-archive`, i.e. v1's explorer (a fresh archive per search) + the
  three fixes: `--demo-match-band`, `--auto-spine-knob-evalroots`, `--auto-spine-deeper`. Record
  gate passed. It should replay v1/v2 to 302M (the fixes are off-path until a spine exists).
* **09:58 UTC - v3 so far** (`unr1b4_knob3`, 972M): replayed v1/v2 to 302M; search 1 -> spine 1
  (the same 796-state line as v2's) live at 391M; 453M 6.2%; search 2 (529M, a fresh archive on
  the eval line) STALE like v1's; **first ramp at 604M (10.1%), on time**; 680M 15.15, 755M
  15.75, 831M 17.27, 906M 17.96 - on v1's curve. Search 3 is due at the 982M eval, then the
  window, where the fixes (match-band, eval-line roots) should first make the difference.
* **10:09 UTC - v3 search 3 (982M, patience) -> spine 2 THROUGH THE WINDOW.** Rooted on the eval
  line itself (2,055 ticks = 20.6 s, best d 81,075 = 18.7%, stall 1,647; roots [1,247, 1,647];
  v1's search 3 at the same step had re-rolled a line that died at 2.0%), a fresh archive: 38,217
  cells, 902,755 runs, knob 0.53-0.6 (cells policy 14,546 / random 23,582 over 15.7M / 19.9M
  decisions). Deepest cell d 75,294 = **24.5%**, chain 10 links P P R P P R R P R P from root
  1,339, exact replay. **The spine passes the window curtain sgcp_1 (rows 2,104-2,110) and has NO
  repeated-origin rows** (v1's spine 3 had a 21-row standstill). Adopted at 1,074M (2,487 states,
  success line 75,394). In ONE search from the deployed line, v3 got what v1 needed two searches
  for (20.6%, then 23.7%) - and without the standstill.
* **10:25 UTC - v3 CLEARED THE WINDOW at 1,284M: score 26.727 (best 26.73%, median 26.69%,
  CP1), on time (deadline 2.07B).** The curriculum on spine 2 walked back without freezing (band
  rows 2,486 -> 2,261 -> 1,766 -> 1,276 in ~140M steps; the window at rows 2,104-2,110), the 1,208M
  eval dipped to 15.7% (the policy in flux), and at 1,284M all 9 greedy episodes went through the
  window to 26.7% - past spine 2's end (24.5%), at the October CP2 wall (26.9-27.1%). The quorum
  fired search 4 at 1,284M. Pulled the hand-on: `unr1b4_knob3/ckpt_1250426880.pt`. Ladder: first
  ramp 604M, window 1,284M - both ON TIME. (October's lineage: 0.30B / 1.04B with recipe changes
  between walls; v1 of this branch: 529M / never - frozen at 20.7%.)
* **10:34 UTC - v3 search 4 (1,284M, quorum) found NOTHING at the CP2 pit wall.** The eval line:
  25.7 s, best d 73,071 (26.7%) at its LAST tick (2,568: the line dies at its best point, a fatal
  fall over the pit), stall 2,530, 95 root cells. The archive's deepest CONTACT cell: d 73,562
  (26.2%) by 30 s and never better; 19,852 cells, yield 3.4e-3 -> 1e-4 per decision within ~3 min
  (saturated); knob 0.59-0.74 (random ~1.5-2x the yield). Two generic limits show here: (1)
  contact-only cells cannot beat a line whose best d is a fall point; (2) the October notes say
  this wall is deceptive in the field (the field's route is closed; the human line dips ~2,000 u
  under the pit slab into a cavity) - a "deepest d" output cannot hand on a detour whose d is
  worse until a chain reaches the far side, and from these roots none did in 10 min. October
  cracked it with RL mechanisms that are all in the recipe (greedy-death spawns, no time charge),
  so v3 runs on: the pit dive is due by 3.88B, the next search at >= 1.68B (retry). 1,359M eval:
  frontier 26.5%, median 24.6%.
* **10:43 UTC - the knob over all 12 searches** (`autoresearch/analysis/knob_table.py` on the
  pulled search logs; "line best" = the line the search rooted on; v1 search 3 re-rolled a line
  that died at 2.0%):

  | run / search | step | line best | deepest cell | knob range | cells / 1k decisions, policy / random | deepest chain | outcome |
  |---|---|---|---|---|---|---|---|
  | v1 / 1 | 303M | 1.7% | 5.5% | 0.46-0.64 | 0.32 / 0.49 | RPPRPPPPRRPPR | adopted |
  | v1 / 2 | 529M | 8.1% | 9.6% | 0.51-0.66 | 1.01 / 1.47 | RPPRPPRRPRPP | stale |
  | v1 / 3 | 982M | 2.0% | 20.4% | 0.53-0.61 | 2.67 / 3.36 | PPPPPPPPPP | adopted |
  | v1 / 4 | 1,208M | 20.6% | 23.7% | 0.53-0.76 | 0.64 / 1.11 | RPRRRRRRRPPRRR | adopted (froze) |
  | v2 / 1 | 303M | 1.7% | 5.6% | 0.31-0.73 | 0.35 / 0.52 | RPRRPPRRPPP | adopted |
  | v2 / 2 | 454M | 5.9% | 9.6% | 0.54-0.70 | 0.49 / 0.75 | PPRPRRR | adopted |
  | v2 / 3 | 605M | 9.9% | 15.3% | 0.56-0.66 | 1.28 / 1.63 | PPRRPPRRRPPP | adopted |
  | v2 / 4 | 755M | 15.6% | 15.4% | 0.42-0.66 | 0.35 / 0.29 | - | none (drowned) |
  | v3 / 1 | 303M | 1.7% | 5.6% | 0.32-0.69 | 0.34 / 0.49 | RPRRPPRRPPP | adopted |
  | v3 / 2 | 529M | 7.0% | 9.6% | 0.52-0.71 | 0.60 / 1.03 | PRPRRRP | stale |
  | v3 / 3 | 982M | 18.7% | 24.5% | 0.50-0.65 | 0.92 / 1.19 | PPRPPRRPRP | adopted -> window |
  | v3 / 4 | 1,284M | 26.7% | 26.2% | 0.54-0.74 | 0.42 / 0.77 | - | none (CP2 pit) |

  Read-out: the yield rule kept the random share inside 0.31-0.76 in every search, never near an
  end; random runs found 1.2-1.8x the policy's new cells per decision in 11 of 12; and the
  deepest chain was a MIX of both modes in 9 of the 10 searches that found one (the exception,
  all policy, is the stretch the policy already flew). Together with the two explorer-only
  ablations (all policy 1.77% / all random 2.76% / the rule 4.93% at the 1.7% wall), the answer
  to "at which point is the policy-guided search more useful": neither end is enough; the
  policy carries the depth, random runs supply the branches it continues from, and one
  probability-matching rule on the archive's own yield finds the middle without being told.
* **11:08 UTC - v3 at the CP2 pit since 1,284M.** Evals 1,359-1,812M: 26.46-26.92 (best **26.919 at
  1,435M**, hand-on still ckpt_1250426880), medians 22.7-26.8%. Search 5 (1,737M, retry) found
  nothing either (26,232 cells, knob 0.55-0.70, no cell past the line's 73,455). Next search >=
  2.13B; the pit dive (27.5%) is due by 3.88B (~13:35 UTC at ~218k steps/s). Probe (analysis, v3
  untouched): the same explorer on search 5's checkpoint and eval line with the roots spread over
  the line's last 12 s instead of 4 s - if the decision that matters here comes seconds before
  the stall (October: the record drops to a lower level ~2-3 s before the pit), a 4 s window
  never branches from it.
* **11:18 UTC - the roots-window probe at the CP2 pit: NULL.** Search 5's checkpoint and eval line
  (26.1 s, best d 73,455 = 26.3% at 2,571, stall 2,531), roots over the line's last 12 s (301
  roots -> 245 root cells, vs 95 in a 4 s window): 35,471 cells (search 5: 26,232), knob
  0.63-0.72, the deepest contact cell d 73,917 (25.8%), never past the line. A wider root window
  alone does not crack this wall in 10 min; not adopted for anything.
* **11:40 UTC - two more explorer-only probes at the CP2 pit (search 5's checkpoint and line, 10 min
  each): both NULL.** Runs of 300 decisions (12 s): 24,396 cells, deepest contact d 73,903; cells
  of 128 u: 7,080 cells, deepest 73,910. With the roots-12 probe (73,917) and searches 4-5
  (73,562 / ~73,9xx), every explorer variant saturates at the same deepest contact state, d
  ~73,900 (25.85%), short of the line's own best (a fatal-fall point, 26.3-26.7%). Next probe
  saves its archive to see whether the explorer reaches the record's lower level at all
  (between the pit kill's top and the plate's level, October's notes) - i.e. whether the limit
  is reach or the "deepest d" output.
* **11:52 UTC - WHY the explorer cannot hand on a line past the CP2 pit (analysis of a saved
  archive, search 5's checkpoint and eval line, roots 12 s; 35,033 cells).** The deployed line
  goes east to (4,613, -460, 12,363) at 25.1% (the U-turn level), turns back WEST and dives into
  the pit kill (it dies at (3,688, 1,075, 10,961)); its best d (26.2-26.7%) is taken MID-DIVE.
  The explorer DOES reach the record's lower level: **230 contact cells between the pit kill's
  top and the plate's level** (x 3,978-6,063, z 11,332-11,879; October's notes put the record
  at (4,041, 661, 11,814) and (5,554, 454, 11,079) there), and the archive's deepest cell of all
  is one of them ((4,343, -906, 11,346), d 73,917 = 25.84%, a random-mode cell). But the field
  rates that detour no better than the policy's fatal dive (73,917 vs 73,455), and both gates
  compare a CONTACT cell with a FALL point - knob_ge's "beats the line's own best d" and the
  trainer's gain test against the greedy's median deepest d. **So at a deceptive wall the
  limit is the output rule, not reach.** Generic fix for a later recipe (not built): measure the
  greedy frontier on CONTACT states too - the deepest d at or before each eval episode's last
  map contact (auto_spine's `--cut contact` rule, already written) - so a supported state on the
  detour beats a frontier that is a dive; the curriculum then teaches the lower level and the
  next search roots there. The pit's far side (back west over the pit top, then CP2 at x ~800)
  was not reached in any probe (min cell x 885 is the approach).
* **11:58 UTC - v3 STOPPED at 2,259M (best 26.919 at 1,435M; evals 1,284-2,190M all 26.5-26.9,
  the CP2 pit); v4 launched.** Pulled: unr1b4_knob3/ckpt_latest.pt (md5 1070b4a6...; for a round 2
  that would CONTINUE v3) and ckpt_1250426880.pt (the hand-on for 26.919). Why stop an on-time
  run (pit dive due 3.88B): 1.0B steps flat, two searches and four probes null, and the archive
  analysis above names the cause - the policy is PAID for its fatal dive (the ratchet pays the
  falling d and the recipe charges nothing for dying), so the dive is its local optimum, and every
  gate compares the explorer's supported states with that dive.
* **v4 = `unr1b4_knob4`, launched 11:58:15 UTC from step 0, commit 105960e0** = v3's flags +
  `--ratchet-contact 1` (an existing generic reward correction: the ratchet pays a record that
  moves only on supported ticks, so an apex over open air - or a dive into a kill - pays at the
  landing that reaches it, or never; the curriculum's success line and bonus read the same
  contact record) + `--auto-spine-contact` (NEW: the greedy frontier, the explorer's gate and the
  spine's cut on supported states). Record gate passed; run.json: ratchet_contact 1,
  auto_spine_contact 1, evalroots 1, deeper 1 (demo_match_band is on - it is in the launch line -
  but its config dump is gated on --demo-file; cosmetic). It has ~3.9 h: at v3's pace the CP2
  pit comes at ~1.3B (~13:45), leaving ~2 h there. If --ratchet-contact slows the early walls,
  that is a result too.
* **12:07 UTC - v4 was broken; v5 = `unr1b4_knob5` relaunched 12:05:56 UTC from step 0, commit
  5635aa00.** v4's auto-spine hook raised `IndexError: index 9 ... size 8` at every eval (the
  trainer caught it and trained on with NO search): `surfgym.route.episodes_from_traj` keeps only
  [tick, x, y, z, vx, vy, vz, yaw], and my contact frontier read a ground flag in column 9. Fixed
  to vz alone (ground cancels gravity every tick, so it reads as supported like a surf face) and
  tested on REAL episodes_from_traj output; checked on v3's 1,435M pit eval: contact frontier
  23.6% (median d 76,171) vs all-state 26.6% (73,136) - the explorer's lower-level cells (25.84%)
  clear it by ~2,250 u. v4 stopped at 95M. v5's flags = v4's (v3 + `--ratchet-contact 1
  --auto-spine-contact`); its first eval's hook ran clean (frontier 99,277); run.json now records
  demo_match_band and the other curriculum keys (13415521).
* **13:21 UTC - v5 FAILED THE FIRST RAMP; v3 CONTINUED.** v5 (`unr1b4_knob5`, v3 +
  `--ratchet-contact 1 --auto-spine-contact`): 1.7% to 378M, spine 1 (to 5.5%) live at 392M - the
  same search as v3's - but the greedy needed until 680M to fly it (v3: ~453M), and search 2 (680M)
  found no supported cell more than 58 u past the line's contact best (10,753 cells; its roots
  [508, 908] sat before the line's crawl to its best at 1,297); evals 680-982M all 5.5-5.6%
  (982M: 5.595) -> no eval past 8% by 1.0B: FAILED at wall 1. **Read-out: paying only supported
  progress slows the early map** - the platform's launch onto the first ramp is airborne, unpaid
  until it lands - and it was never tested at the pit it was meant for. A recipe that wants the
  contact ratchet for deceptive walls would need it from where the dives are, not from step 0 -
  which a frozen recipe cannot say; or a frontier on contact (the half that is free of this cost)
  without the contact ratchet. Not tried.
* **`unr1b4_knob3c` = v3 CONTINUED** from its own newest checkpoint (box
  runs/unr1b4_knob3/ckpt_latest.pt, md5 9ac57e6e..., step 2,258,632,704 - written at its 11:57
  stop, after my 11:57:12 pull), on v3's commit 9c8be647 (checked out on the box) with v3's exact
  extra flags; the trainer restored the recipe; run.json config diffed against v3's: 292 keys, 1
  differs (steps). The interruption is treated as a crash-resume: the reservoir, the curriculum
  and the spine are not in a checkpoint, so it starts with no spine and searches after 3e8
  steps. The pit dive is due at 3.88B (~15:30 UTC at ~215k steps/s).

### unr1b4 round 1 - THE LESSON (written 13:50 UTC; final numbers appended at wrap-up)

**What slot 4 built** (all default-off, bit-identical when off, TRAIN_ONLY for the recorder, tests in
tests/python/test_knob_ge.py): `tools/knob_ge.py` + `beam_tas.py --ge-knob` (Go-Explore over exact
CONTACT states - 64 u cells x 64 u kinetic-energy buckets - W = 1/sqrt(1 + chosen); each explore
run of 100 decisions commits to held random actions in the policy's own velocity frame (repeat
0.95) or to the policy's samples; THE KNOB = the random share, set by probability matching on each
mode's new-cell yield per decision over 16 populations of run ends; the deepest chain replayed
exactly and spliced onto its root line), `auto_spine.py --knob`, and in the trainer
`--auto-spine-scratch` (the loop from step 0 with no spine), `--auto-spine-knob`, `--demo-share`
(the spine curriculum as a 25% pool share, the recipe's own pool kept),
`--auto-spine-knob-evalroots` (root on the eval's own exact states), `--auto-spine-deeper`,
`--auto-spine-knob-archive` (null), `--auto-spine-contact` (frontier/gate/cut on supported states).

**The ladder (one seed each; the control = the same recipe, no search, `unop_ctl`):**

| run | recipe | first ramp | window (CP1) | best | why it stopped |
|---|---|---|---|---|---|
| control | incumbent, no search | FAILED (1.7% to 5.96B) | FAILED | 1.72 | never left the start platform |
| v1 `unr1b4_knob` | + the knob loop | **529M** | - | 20.69 | curriculum frozen by a spine standstill (origin matching) |
| v2 `unr1b4_knob2` | v1 + match-band, eval roots, deeper, ONE archive | 529M | - | 15.66 | the run-long archive drowned the frontier (null) |
| **v3 `unr1b4_knob3`** | **v1 + match-band, eval roots, deeper** | **604M** | **1,284M** | **26.919** | flat at the CP2 pit 1.28B+ (continued as knob3c) |
| v5 `unr1b4_knob5` | v3 + `--ratchet-contact 1 --auto-spine-contact` | FAILED (5.6% at 982M) | - | 5.60 | paying only supported progress slows the start |

**The knob (12 searches + 2 explorer-only ablations): the policy-guided side is what goes deep,
random runs supply the branches it continues from, and neither alone is enough.** The yield rule
held the random share at 0.31-0.76 in every search without being told; random runs found 1.2-1.8x
the policy's new cells per decision; the deepest chain mixed both modes in 9 of 10. At the 1.7%
wall (where the policy itself was stuck) the mix reached 4.93% vs all-policy 1.77% / all-random
2.76% at matched decisions; where the policy already flew a stretch, mix = all-policy (20.4%),
all-random 2.8%.

**The three recipe bugs the loop exposed (all generic, all fixed in v3):** (1) spines with a
standstill froze the curriculum - `--demo-match-band`; (2) the search re-rolled the greedy in its
own 1-env numerics and flew another line - root on the eval's exact states; (3) a persistent
archive with count-based selection gave ~99% of the picks to old cells - per-search archives.

**What stops v3 at the CP2 pit, and what to try next (round 2):** the policy is PAID for its
fatal dive (the ratchet pays the falling d; nothing charges a death), and every gate compares the
explorer's supported states with that dive - so a line onto the record's lower level (which the
explorer reaches: 230 contact cells under the plate, above the pit kill) is never handed on. The
contact ratchet removes the payment but slows the start (v5 failed wall 1). Generic candidates,
untested: (a) the contact FRONTIER alone (`--auto-spine-contact`, built) without the contact
ratchet, plus a success rule that counts only a SUPPORTED crossing of the spine's line (needed:
the all-state record lets a dive cross it; `--demo-success-alive` degenerates for a spine that
ends at its deepest state - alive_need caps at the spine's own ticks past the line); (b) a death
charge (Grzes 2017's correction for potential shaping under termination: the trainer has
`--death-charge`), which takes the dive's paid progress back at the death without touching the
start. Either would be ONE generic term, set at step 0 and frozen.
* CORRECTION to (b) above: `--death-charge` exists but the trainer REFUSES it with `--unstuck`
  (train_fast.py ~11222: "--unstuck is not implemented with --death-charge"), which is in the
  incumbent recipe - so (b) needs that combination built (the tempered draw's log-prob path) before
  it can be one frozen term of this recipe.
* **14:12 UTC - v3 (continued as unr1b4_knob3c) set a new best: 27.474 at 2,939M** (best episode
  27.47%, median 26.8%, CP1) - at the pit-dive line (27.5%, due 3.88B) after ~1.6B steps flat at
  26.5-26.9 (evals 2,335-2,863M: 26.77-26.87). Search 1 of the continuation (2,561M, patience)
  found nothing (25,418 cells, deepest contact cell d 73,450 = 26.3%, the line's dive 72,952).
  Hand-on pulled: unr1b4_knob3c/ckpt_2758803456.pt. ladder.py on the knob3c directory alone reads
  LATE / STOP because its first eval is at 2.26B; its walls are in unr1b4_knob3 (604M, 1,284M).
* **14:30 UTC - PIT DIVE CLEARED at 3,165M (score 27.536: best 27.54%, median 26.8%), on time
  (deadline 3.88B).** v3's third wall, by the recipe's own RL after ~1.9B steps at the CP2 pit
  (the continuation's searches 1-2 at 2,561M / 3,014M found nothing: 25,418 / 32,077 cells,
  deepest contact cell d ~73,450); evals 2,939M 27.47 and 3,165M 27.54 are the first past the
  pit's 26.9 plateau. Hand-on: unr1b4_knob3c/ckpt_3008888832.pt (pulled). Ladder for v3: first
  ramp 604M, window 1,284M, pit dive 3,165M - all ON TIME; next the cavity (CP2), due 8.27B.
* **14:59 UTC - search 3 of the continuation (3,467M) handed on a line PAST THE PIT: spine at
  3,558M, end d 72,036 = 27.73%** (greedy frontier 72,900 = 26.9%: +864 u). Rooted on the eval
  line (26.0 s; its best is its last tick, the dive), 10 links R P P R R P P R P P (4 random, 6
  policy) from root 2,248, exact replay. Its path: east and LOW (4,951, -717, 11,171), then back
  WEST over the pit top (3,713, -908, 11,593) to (2,511, -881, 11,410) - past the pit's west edge -
  the shape October's notes give for the record's line ("under the plate, east, low ... then a
  ramp, and back west over the pit top at z ~11,469"). Searches 4-5 of v3 and 1-2 of the
  continuation, from a less trained policy, never reached it in 10 min; the policy's own approach
  (3.4B steps) brought the detour into reach. So the all-state gate did let it through once a
  supported cell on the far side existed: the limit at the pit was reach on the far side, which
  the policy's progress provided - the output-rule analysis above stands for the lower level
  itself, whose cells were not deeper than the dive. The curriculum is teaching it now.
* **15:13 UTC - 28.164 at 3,769M: all 9 greedy episodes at 28.2%** - the policy flies the detour
  past the CP2 pit that search 3 of the continuation found (adopted 3,558M; the curriculum walked
  its band back from row 2,978 to 2,283 by ~3.73B, near the chain's root at 2,248). 3,543-3,694M
  evals 27.55-27.63 on the way. The quorum fired search 4 at 3,769M from the new line.
* **15:35 UTC - OPERATOR NOTE (the user's decision, 15:10 UTC): round 1 is the LAST round, now
  ending 2026-11-09 12:35 UTC (wrap-up 11:55).** On-time runs keep training; a run that misses a
  wall deadline is stopped (then a NEW recipe may start from scratch); a run that reaches its step
  cap is resumed from its own checkpoint with the same flags and a higher cap (new runs:
  --steps 60e9); boxes extended <= 12 h at a time; wake every 45-60 min while a run just trains;
  the verdict is the ladder. So v3 (on time: pit dive at 3,165M; cavity CP2 due 8.27B) keeps
  training: its continuation's cap (4,258,632,704) will be resumed from its own ckpt_final.pt on
  the same commit with the same flags and --steps raised. Box extended to 2026-11-09 03:35 UTC.
  3,844-3,996M evals: 28.20 / 28.17 / 28.21 (9/9 past the pit); search 4 (3,769M) found nothing
  deeper (27,724 cells).
* **15:53 UTC - v3 reached its cap and continues as `unr1b4_knob3d`.** unr1b4_knob3c stopped at its
  cap, 4,258,791,424 steps (ckpt_final.pt md5 4b693f1f...; evals to 4,222M at 27.9-28.2, best
  **28.212 at 3,996M**); resumed from that ckpt_final on the same commit 9c8be647 with the same
  flags and --steps 56e9 (stop at 60.26B), per the operator note. The trainer's exit killed
  auto_spine.py but NOT its beam_tas child (pid 36675, search 5): killed by hand - a known gap in
  `_auto_spine_kill` (it kills the auto_spine process, not its process group). Each resume drops
  the curriculum and the spine (not in a checkpoint): the first search comes 3e8 steps after the
  first eval.
* **16:25 UTC - v3 (knob3d) climbs past the pit: 30.106 at 4,637M** (9/9 at 30.1%, CP1). Evals
  4,259M 28.20, 4,335M 29.32, 4,410M 29.96, 4,486M 29.90, 4,561M 30.07, 4,637M 30.11. Search 1 of
  knob3d (4,561M) adopted a spine at 4,661M ending at d 69,434 = 30.3% (frontier 69,716). Hand-on
  pulled: unr1b4_knob3d/ckpt_4508876800.pt. Next wall: the cavity (curtain sgcp_2), due 8.27B.
* **16:47 UTC - CAVITY (CP2) CLEARED at 4,712M (score 30.676, CP2), on time (deadline 8.27B;
  October 4.14B).** v3's ladder: first ramp 604M, window 1,284M, pit dive 3,165M, cavity 4,712M -
  all on time, one recipe from step 0 (two cap/stop resumes on the same commit and flags). Best
  **30.681 at 4,788M** (hand-on unr1b4_knob3d/ckpt_4758962176.pt, pulled); evals 4,712-4,939M
  30.59-30.68. Search 2 of knob3d (4,712M) found nothing past d 69,328. Next: the post-CP2 block
  (>33%, due 13.93B), CP3 (14.64B).
* **17:39 UTC - v3 (knob3d) at 5.68B, a plateau at 30.5-30.7%** (CP2; evals 4,712-5,618M 30.52-30.72,
  one dip 27.57 at 5,543M; best **30.723 at 5,467M**, hand-on pulled). Searches 2-3 of knob3d
  (4,712M / 5,185M) found no cell past the frontier (d ~69,250; search 3 only 8,559 new cells).
  Next wall: the post-CP2 block (>33%, due 13.93B; October stalled at 32.3-32.5%).
* **18:45 UTC - v3 (knob3d) at 6.52B: 30.5-30.9%, new best 30.849 at 6,222M** (CP2; evals
  5,693-6,524M 30.58-30.86, one dip 9.3% episode; hand-on snapshot unr1b4_knob3d/ckpt_6009389056.pt
  pulled). Searches 4-5 of knob3d (5,618M / 6,071M) found nothing past the line (k05: 10,003 cells,
  1.25M runs, yield ~0 per decision by 8 min, deepest cell d 69,055 vs the line's 69,052).
  **The plateau is a fatal dive again**: in the 6,222M and 6,524M evals, 16 of 18 episodes end
  `fail` at 32.1-32.5 s with their best d ON THE DEATH TICK (falling, vz -136..-335, horizontal
  speed 430-820, z 10,260-10,370, spread over ~300 x 400 u) - the CP2-pit signature one wall later.
  The ratchet pays the fall, nothing charges the death.
  Box extended to 2026-11-09 06:42 UTC (`extend --hours N` sets now+N, it does not add).
* CORRECTION to the 13:50 "CORRECTION to (b)": the refusal at train_fast.py ~11222 is
  `--curiosity-cond is not implemented with --death-charge` (that block lists --unstuck AND
  --death-charge as incompatible with --curiosity-cond). Nothing refuses `--death-charge` with
  `--unstuck`; `--death-charge` + `--race-ratchet` (+ `--death-charge-window`) is wired
  (rewards.py: the bank = what the ratchet paid in the last W s, read from `_rec_boot`), and both
  flags are in record_ckpt.py. So (b) is ONE flag pair on this recipe, no code.
* **Plan if v3 misses the post-CP2 block (13.93B, ~03:40 UTC at 229k steps/s):** stop it (ladder)
  and launch v6 = the v3 flags + `--death-charge 1 --death-charge-window 2` from scratch,
  `--steps 60e9`. The constants are not read off this map: kappa 1 and W 2 s are uf2r1b1's
  windowed charge (W = the recipe's own `--respawn-margin 2`, the doomed-tail margin), the lineage
  that finished unitfarmer2. It cannot overtake v3 in the ~8 h left; it measures whether the one
  generic term moves the early walls, the CP2-pit dive above all (v3: 1,284M -> 3,165M stuck there).
* **18:48 UTC - search 6 of knob3d (6,524M) handed on a line PAST the 30.8% dive: spine at 6,630M,
  end d 67,501 = 32.28%** (greedy frontier 69,105 = 30.67%: +1,604 u). Rooted on the eval line at
  tick 3,020 (of 3,218; its best is the death tick), 18 links R P P R P R P R + ten R (13 random,
  5 policy), exact replay (0 u). The deepest cell appeared at ~450 s of the 600 s search (yield
  had been ~0 per decision from 180 s): best d 69,477 -> 67,588. Its path: from (1,194, 512,
  10,115) west and UP - vz +1,162 at (899, -519, 10,104), 10,867 at x 88, i.e. ~500 u above where
  the policy's dives die (x 140-435, z ~10,300) - then west to x -703, ending in a fall at
  (-628, 67, 10,342). Searches 2-5 (10 min each, same rule) had not reached it. The curriculum
  restarts at the spine's end; the post-CP2 block needs > 33% (d < 66,782), due 13.93B.
* **19:22 UTC - v3 at 7.11B, best 30.920 at 7,052M** (hand-on unr1b4_knob3d/ckpt_7009730560.pt pulled;
  6,524M scored 30.918). The greedy still dies at ~30.8% (evals 6,600-7,052M 30.73-30.92, one 27.6
  dip); the curriculum on search 6's spine oscillates at starts [3,038-3,043, 3,413] (success
  6.7-31.3% over ~500 episodes, the junction with the policy's own line at row ~3,020). Search 7
  (6,977M) found a line to d 67,302, kept out by the 200 u gain rule (the spine being taught ends at
  67,501).
* **19:46 UTC - v3 FLIES SEARCH 6's LINE: 32.268 at 7,354M, 32.427 at 7,430M (median 32.3%, 7/9
  past 31.6%)** - ~0.8B steps after the spine (6,630M), the curriculum walked back from row 3,038 to
  ~1,320 by 7.4B. Hand-on unr1b4_knob3d/ckpt_7259815936.pt pulled. The 7,430M episodes reach their
  best d (67,353-67,469) at x -690..-800, z 10,220-10,290, falling at vz ~-1,000+, then land
  ~3-8 s later on a lower floor at z ~9,715 (y +-300, d ~67,950 - behind their best) and end
  `fail` there: past the dive, the next wall is getting out of that lower level. Search 8
  (7,430M) roots on the new line. The post-CP2 block needs > 33% (d < 66,782), due 13.93B (~03:35
  UTC); October stalled here at 32.3-32.5% and cleared it at 6.97B.
* **20:40 UTC - v3 COLLAPSED at ~7.6B: evals 7,656-8,110M score 1.1-2.2 (one 15.2 episode at
  7,808M), from 32.37 at 7,581M.** The greedy now stands on the spawn platform until eval patience
  (2/9) or drops off it to z ~13,830 and dies (7/9) - the control's picture. The training log has a
  runaway PPO step size before it: median approx_kl per 250M bucket 0.018-0.024 over 0-4B,
  0.03-0.05 over 4.25-6.5B, 0.063 (6.75B), 0.103 (7.0B), 0.19 (7.25B), **0.64-0.77 (7.5-8.0B)**, with
  single updates at 13.4 / 22.0 (7,592-7,597M, where the training reward fell 41 -> 13) and up to
  45.6; value loss 1.8-2.1 -> 5-6. View sigma 0.05-0.07 (yaw) / 0.49 (pitch, capped). The incumbent
  runs PPO at lr 3e-4 with no KL guard (`--target-kl` unset). October saw the same class
  (docs/research-results.md 2026-10-07: first-minibatch KL 3-30 with a collapsed sigma at lr 3e-4;
  `--target-kl 0.02` did not stop it, `--lr 1e-4` did; a later scan found KL spikes do not
  predict collapses). The run is ON TIME by the ladder (its walls stay cleared; the post-CP2 block
  is due 13.93B), so per the operator note it keeps training; the best stays 32.427 (7,430M,
  hand-on ckpt_7259815936 from before the collapse). Search 9 (7,883M) rooted on a collapsed eval
  (925 ticks) and found a line to 67,388 - kept out by the gain rule. **Lesson for the next
  generation: the late-run instability is a recipe property (lr 3e-4, no step-size guard, a
  near-deterministic view head); a frozen recipe needs its stabiliser at step 0.**
* **21:40 UTC - v3 at 9.03B: no recovery.** Evals 8,185-8,940M score 1.0-3.4 (one 15.7 episode at
  8,412M); median approx_kl per 250M bucket 0.81 (8.0B), 0.95 (8.25B), 1.10 (8.5B), **2.78
  (8.75B)**, single updates to 91.6; the box log at 9.02B prints kl 1.5-7.0, ent 0.006. Training
  episodes still fly from the curriculum's starts (rows 748-3,413 of search 6's spine, success
  56.6% over 336 episodes; training reward 33-38), so what broke is the true-start greedy, not
  the spine's skill. Ladder (chain dir): all four walls on time, post-CP2 due 13.93B (~03:40 UTC).
  CORRECTION to the 20:40 entry: the view sigma did NOT collapse. Checkpoint view_std (yaw /
  pitch): 0.064/0.089 (1.25B), 0.075/0.104 (3.5B), 0.068/0.211 (5.26B), 0.068/0.358 (6.0B),
  0.066/0.479 (7.0B), 0.070/0.496 (7.26B) - yaw flat for 6B steps. What grew is the weights:
  L2 over the policy state_dict 195 (1.25B) -> 310 (3.5B) -> 379 (5.26B) -> 402 (6.0B) -> 431
  (7.0B) -> 439 (7.26B); view_head.weight 0.30 -> 0.43 -> 0.72 -> 0.88 -> 1.32 -> 1.29. A fixed
  sigma under a 4.4x more sensitive mean makes the same lr a larger step in KL (KL ~ (dmu/sigma)^2:
  1.5x the view_head norm from 6.0B to 7.0B predicts 2.2x the KL; measured 0.045 -> 0.103). So the
  October "collapsed sigma" picture does not fit here; norm growth at a fixed lr does. Generic
  levers, in order of evidence on this repo: --lr 1e-4 (stopped October's), AdamW --wd (bites on
  the norm - 0.67x at 700M, efWD_blue050 - never tested as a stabiliser), a KL-adaptive lr
  (rl_games' schedule; not in the trainer). Not for v6: its ~6.7B steps end before v3's runaway
  began, and a slower lr would cost the early walls it is meant to measure.
* **22:45 UTC - v3 RECOVERED by itself: 32.301 at 9,016M, new best 32.447 at 9,695M** (median 32.3%
  at 9,469 / 9,544 / 9,695M; two relapses to ~2% at 9,242-9,317M and one 27.6 at 9,620M). Hand-on
  unr1b4_knob3d/ckpt_9510584320.pt pulled. The KL did NOT come down (median per 250M bucket 3.9
  (9.0B), 4.4 (9.25B), 5.4 (9.5B), p90 16-20, max 115) while the greedy came back, so the KL's size
  does not track the collapse; part of it may sit in a head with no physics effect (pitch: no
  entropy term, sigma at its 0.5 cap) - per-head KL is not logged. Same picture as 7,430M: all 7
  good episodes reach their best (d 67,333-67,503) at x -630..-820, z 10,200-10,340, falling at vz
  ~-1,050, then land on the lower floor (z 9,715-9,880) and end there. Searches 10-12 (8,433 /
  8,879 / 9,326M) found lines to d 67,384 / 67,373 / 67,499 - none 200 u deeper than the spine
  being taught (67,501), so no new spine since 6,630M. The curriculum is back at the spine's first
  rows (starts [13-23, 3,413], success 0-1.2% over ~240 episodes: the true-start region is what
  it is re-learning). Post-CP2 (>33%, d < 66,782) due 13.93B (~03:40 UTC).
* **23:45 UTC - v3 at 10.54B: a second relapse, to the CP2-pit dive.** Evals 9,771-9,997M 32.36-32.45
  (median 32.3-32.4%), then 10,073-10,526M **26.5-27.0% (CP1), 9/9 episodes alike**: each ends `fail`
  at 26.2-26.4 s with its best d on the death tick (vz ~-1,400, hspd ~1,300, x ~3,170, z ~11,160)
  - the pit dive v3 first got past at 3,165M. The training side turned at 10.0B: reward per
  125M bucket 37-42 (9.5-9.875B) -> 11.8 (10.0B) -> 9.2-9.7; value loss 4.1-4.9 -> 1.6 -> 0.14-0.17;
  the KL fell back (median 9.6 at 9.75B -> 0.46-1.46 from 10.125B), so the relapse is not a KL
  spike either. The box log shows the energy front bonus paying +0 u/ep (it paid ~+150 at 9.0B)
  and the curriculum at spine rows 315-325 (success 5-40%). Search 14 (10,073M, rooted on the
  relapsed eval line) reached d 69,193 - kept out by the gain rule; search 15 (10,526M) running.
  The walls stay cleared, so the run is ON TIME: post-CP2 due 13.93B (~03:45 UTC). Best stays
  32.447 (9,695M; 32.446 at 9,997M).
* **00:30 UTC - v3 recovered again (32.482 at 10,752M, new best; 10,601-11,054M 31.9-32.48, median
  32.4% from 10,752M) and search 16 handed on a line PAST the post-CP2 line: NEW SPINE at 11,087M,
  end d 66,539 = 33.25% (the wall needs d < 66,782).** Hand-on unr1b4_knob3d/ckpt_10510925824.pt
  pulled (md5 5dda60e8...). Search 16 (10,979M): rooted on the eval line (4,632 ticks, best d
  67,370 at tick 4,234; 101 roots over ticks 3,758-4,158), 90 root cells; 12,614 cells, 1.26M
  runs, lambda 0.50-0.77 (0.77 at the end); yield ~0 per decision from 150 s, then the deepest
  cell at ~470 s (best d 67,409 -> 66,542) - search 6's shape again (a find late in a flat
  search). Chain 12 links from root tick 3,774, modes [0,0,0,1,1,1,0,1,1,0,1,0] (6 + 6), exact
  replay (0 u). Its path: where the greedy falls from x ~-700 (z ~10,200, vz ~-1,050) to the
  lower floor at z ~9,715, the line is still CLIMBING at (-477, -185, 10,256) (vz +422), lands
  on a floor further west at (-1,375, -214, 9,911) and crawls west along it to (-1,452, -42,
  9,958) at the end. The curriculum restarts at its end; the wall is due 13.93B (~2.8B steps, ~3.4
  h); search 6's line took ~0.8B steps to reach the greedy.
* **01:25 UTC - v3 at 11.94B: a THIRD collapse, a different one - the yaw sigma blew up to the
  clamp right after search 16's spine.** Evals 11,129 / 11,205M 32.41 / 32.36, then 11,281-11,885M
  1.3-4.3. Box log (every 30th step line): at the adoption (11,087M) crawl 33% -> 58-66%, stall
  0 -> 5-55%, kl 7.96 / 6.60, reward ~45 -> ~23; then the yaw sigma climbs exponentially from 0.053
  (11.09B): 0.060 (11.12B), 0.104 (11.19B), 0.266 (11.23B), 0.486 (11.28B), 0.989 (11.36B), 1.306
  (11.39B) ... **2.718 = exp(LOG_STD_MAX) by 11.9B** (entropy loss -1.9 -> -6.5, value loss 4-5 ->
  14-19, kl ~0.07). The curriculum sits at that spine's rows 4,334-4,339 (success 13-34%,
  oscillating), i.e. on its tail: a slow crawl along the floor at x -1,375..-1,452 (hspd 25-100)
  where yaw barely matters, so the entropy bonus can win on the one state-independent yaw sigma.
  The trainer already has the generic fix in this branch's history: `--yaw-entropy 0`
  (c416c4a8; its help cites princessburglar 0.08 -> 2.718 in ~500M steps), and the uf2r1b1 dcw arm
  ran it from scratch with the windowed death charge (ledger: the lineage that finished
  unitfarmer2). The incumbent recipe does not carry it. Trade-off: with no bonus the yaw sigma is
  left to the policy gradient and may sit LOWER, which is the side of failure 1 (KL ~ 1/sigma^2).
  So v3 has now shown three late failure modes on one frozen recipe: KL runaway with weight-norm
  growth (7.6B), a reward/value relapse with no KL signal (10.0B), and the yaw sigma at the clamp
  after a crawl-tailed spine (11.1B). Still ON TIME (post-CP2 due 13.93B; the first eval past it
  is 13,998M, ~03:51 UTC). v6 stays as planned (v3 flags + `--death-charge 1
  --death-charge-window 2`; box code = 9c8be647, v3's): all three failures came after ~7.6B and v6
  has ~6.5B steps; `--yaw-entropy 0` goes to the next generation's list with the trade-off.
* Process slip, 01:20 UTC: a `git log --all -S--yaw-entropy` also listed a commit subject from
  another branch's ref (a commit not in this branch's history). The protocol forbids reading other
  branches' refs; nothing from it is used, and searches here stay on HEAD's history.
* **04:00 UTC - v3 FAILED the post-CP2 block and is STOPPED; v6 launched.** Evals 12,791-14,074M
  0.9-2.1 (yaw sigma still 2.713-2.718); the evals past the deadline, 13,998M and 14,074M, scored
  1.97 and 1.58. `ladder.py` on the chain (188 evals): "STOP: a wall was missed" - first ramp
  604,504,064 / window 1,283,981,312 / pit dive 3,165,126,656 / cavity 4,712,300,544 ON TIME,
  post-CP2 block FAILED (deadline 13,931,380,736). Trainer pid 37326 sent TERM at 14,126M (no
  search was running; nothing left on the GPU). v3's final: best **32.482 at 10,752M** (hand-on
  ckpt_10510925824.pt, md5 5dda60e8...), the 33.25% line of search 16 never reached the greedy.
  Box extended to 2026-11-09 16:00 UTC (12 h; released at wrap-up 11:55).
  **v6 = `unr1b4_dc6`**: the box's code is v3's commit 9c8be647 (clean), the incumbent recipe +
  v3's exact extra flags (from boxes/unr1b4.json, run unr1b4_knob3) + `--death-charge 1
  --death-charge-window 2`, from scratch, `--steps 60e9`. Record gate passed (greedy, stoch,
  mixed + POV in 36 s); the trainer prints DEATH CHARGE kappa=1 and the 2 s window (200 calls)
  under the ratchet. The death-charge next-spawn fix ee29bffb is an ancestor of 9c8be647. Question:
  does charging back a fatal fall's last 2 s of ratchet pay shorten the fatal-dive plateaus (v3:
  window 1,284M -> pit dive 3,165M; the 30.8% dive 4.7-7.3B)? It has ~6.5B steps to 11:55 UTC;
  matched-step comparison against v3's ladder.
* **04:35 UTC - v6 (`unr1b4_dc6`) at 0.42B: 5.75% from 151M on** (evals 151 / 227 / 303 / 378M:
  5.75 / 5.76 / 5.76 / 5.75, 9/9 alike) against v3 at the same steps 1.70 / 1.72 / 1.71 / 1.62 (v3
  left the platform at 454M, 6.28). First ramp (> 8%) pending, deadline 1.00B (v3 604M). Search 1
  (from 0.30B) found no new spine (21,818 cells, 0.98M runs, lambda 0.68). 207k steps/s (v3
  ~225k).
* **05:30 UTC - v6 at 1.16B: FIRST RAMP at 604M (12.65; v3 also 604M), ON TIME; 20.5% from 831M**
  (9/9 alike). Matched steps v6 / v3: 680M 15.21 / 15.15, 755M 15.78 / 15.75, 831M 20.58 / 17.27,
  906M 20.49 / 17.96, 982M 20.52 / 18.77, 1,057M 20.48 / 18.25 - a gate ahead, not a wall (v3 took
  the window, 26.7%, at 1,284M). No spine yet: search 2 (755M) reached d 79,324, STALE against the
  greedy frontier 79,174 that moved under it; next search at 1,138M. 220k steps/s.
* **06:25 UTC - v6 at 1.81B: WINDOW (CP1) at 1,435M (26.60), ON TIME (deadline 2.07B; v3 1,284M).**
  Evals 1,133-1,359M 20.46-20.51, 1,435-1,812M 26.43-26.83 (median 26.0-26.4). Now at the CP2-pit
  dive like v3 at the same steps: at 1,812M 8/9 episodes end `fail` at 28.3-29.2 s with their best
  d on (or 0.5 s before) the death tick at x 3,230-3,810, z 10,930-11,130 (v3 at 1,812M: 25.1-26.1
  s, same place). One difference in how they die: v6's last ticks fall at vz -200..-1,030 (v3
  -1,460..-1,500). Search 3 (1,313M) adopted a spine to d 75,275 (24.5%); the greedy passed it.
  Pit dive (> 27.5%) due 3.88B (v3 3,165M) - the death charge's test.
* **07:15 UTC - v6 PIT DIVE at 1,888M (27.98), ON TIME - 1.28B steps before v3 (3,165M), and
  ahead of October (1.94B).** Evals 1,888-2,416M 26.69-28.19 (median 26.0-27.9; 9/9 at 27.8-27.9%
  from 2,265M). It came from the policy, not a spine: search 3's spine ended at 24.5% and the greedy
  was already past it; between the 1,812M and 1,888M evals the greedy frontier moved d 73,429 ->
  72,109 with no new spine (search 5 from 1,888M found none, 40,399 cells). v3 sat at this same
  fatal dive (best d on the death tick at x ~3,300, z ~11,100) from the window (1,284M) to 3,165M =
  1.88B steps; v6 from 1,435M to 1,888M = 0.45B. One seed and a gate-like wall, so read it as a
  large lead at the wall the charge was built for, not a measured effect size. Search 6 (2,341M)
  adopted a spine to d 70,513 (29.3%); its curriculum starts at rows 3,313-3,388 (success 97.9%),
  then 3,073 (33.7%). Training healthy: kl 0.011, yaw sigma 0.073, 217k steps/s. Next: the cavity
  (CP2), due 8.27B (v3 4,712M) - at 217k steps/s v6 reaches ~6.4B by the 11:55 wrap-up.
* **07:55 UTC - v6 CAVITY (CP2) at 2,643M (30.77), ON TIME - v3 4,712M, October 4.14B.** v6's
  ladder: first ramp 604M / window 1,435M / pit dive 1,888M / cavity 2,643M (v3: 604M / 1,284M /
  3,165M / 4,712M). Evals 2,492-2,945M 29.43-31.14, best **31.135 at 2,718M** (v3's best 32.482
  stays the RESULT best until v6 passes it). Searches 6, 7, 8 each adopted a spine: d 70,513
  (29.3%, 2,441M), 69,521 (30.3%, 2,586M), **67,404 (32.4%, 2,954M)**. At 2,945M 6/9 episodes reach
  30.3-30.8% at x 330-870, z 10,200-10,450, slow (hspd 60-380) and end falling (vz -280..-680) -
  v3's 30.8% plateau region (v3 sat there 4.7-7.3B, until search 6's spine at 6.63B to 32.28%); 3/9
  fall earlier at 27.8% (x ~2,285, z 11,415). Post-CP2 (> 33%, d < 66,782) due 13.93B; v6 reaches
  ~6.1B by the wrap-up.
* **08:50 UTC - v6 at 3.71B: 31.0-32.4%, best 32.363 at 3,247M** (32.351 / 32.360 at 3,473 / 3,549M;
  median 30.7-31.7%) - v3 first reached 32.27 at 7,354M. v6's best is 0.12 under v3's 32.482. At
  3,549M 8/9 episodes end `fail` at ~38 s falling (vz -310..-930) at 31.3-31.8%, x -460..+80, z
  10,450-10,820; one reaches 32.36% at x -708 and lands on the lower floor (z 9,717) like v3's
  32.4% episodes. The curriculum is at the END of search 8's spine (rows 3,426-4,071 of 4,072,
  success 19-64%) - it has not walked back yet; search 9 (3,408M) reached 67,378, kept out by the
  gain rule. Healthy: kl 0.022, yaw sigma 0.063, reward 47.7, 216k steps/s. ~6.2B by 11:55.
* **09:05 UTC - v6 search 10 (from ~3.7B) handed on a spine PAST the post-CP2 line: end d 64,776 =
  35.0% at 3,869M** (the wall needs d < 66,782; v3's deepest line ever was 66,539 at 11.09B).
  Rooted at eval-line tick 3,300; chain 16 links (modes [1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0], 5 + 11),
  exact replay (0 u); best d 66,540 at 480-540 s, then 64,776 by 600 s - a late find again. Path:
  (2,422, -531, 10,363) falling -> a lower level (1,176, 499, 9,723) at 1,600 u/s -> climbing at
  (-505, -101, 10,245) (vz +454) -> west (-929, -38, 10,374), (-1,383, 289, 9,972), (-1,678, -343,
  10,068), (-1,969, -40, 9,852) -> ends mid-fall at (-1,616, -303, 9,015) (vz -358, hspd 1,002).
  Unlike v3's search-16 line it has no crawl tail (hspd >= 320 throughout), so the curriculum will
  not sit on states where yaw is irrelevant. The curriculum restarts at its end; v6 is healthy
  (kl 0.025, yaw sigma 0.066). ~1.9-2.2B steps remain to the 11:55 wrap-up.
* **09:40 UTC - v6 at 4.34B: search 11 (4,312M) replaced search 10's spine with a deeper one, end d
  63,075 = 36.7%** (the deeper-only rule: 1,701 u past 64,776). Greedy still on the 31.4-32.4%
  plateau (evals 3,700-4,304M 31.35-32.36, median 31.3-31.5%; frontier 68,314). The curriculum
  walks back fast on the new spine: starts 4,596 -> 4,576 -> 4,536 -> 4,496 (success 71.5-83.0%,
  step 40). Healthy: kl 0.035, yaw sigma 0.065, reward 39.7. ~1.8B steps to the 11:55 wrap-up
  (v3 needed ~0.72B from search 6's spine to the greedy flying it).
* **10:15 UTC - v6 clears the POST-CP2 BLOCK and CP3 at 4,681M: 38.965 (9/9 at 37.2-39.0%, CP3) -
  NEW BEST, ON TIME, and ahead of October (6.97B / 7.32B).** v6's ladder: first ramp 604M / window
  1,435M / pit dive 1,888M / cavity 2,643M / post-CP2 4,681M / CP3 4,681M; October: 0.30 / 1.04 /
  1.94 / 4.14 / 6.97 / 7.32B; v3 (no death charge): 604M / 1,284M / 3,165M / 4,712M / FAILED.
  Evals 4,379-4,606M 31.39-31.58, then 4,681M 38.97 and 4,757M 37.37 (median 37.3%). The 4,681M
  episodes end `fail` at 48-52 s with their best d on the death tick, falling (vz ~-900) at
  (650, -180, 8,740) - 8/9 alike - and one at (2,192, 272, 8,670) for 38.97%: a lower level
  (z ~8,500-8,750) below the post-CP2 area. It came after the curriculum on search 11's spine
  (63,075 = 36.7%, adopted 4,312M) walked back; search 12 (4,767M) has a spine to d 58,361 =
  41.4% from the new frontier 62,503. Hand-on unr1b4_dc6/ckpt_4501536768.pt pulled (md5
  c6ca5864...). RESULT.json best -> v6.
* **10:42 UTC - v6 at 5.07B: best 41.924 at 5,059M (41.920 at 4,983M; 8/9 at 41.9%), CP3; CP4 next
  (due 18.79B, October 9.39B).** Evals 4,832 / 4,908M 37.38 / 37.34, then 4,983 / 5,059M 41.92.
  The 41.9% episodes reach their best at (5,290-5,296, 531-540, 8,528-8,573) STANDING on a ledge
  (vz 0, hspd 62-195) at ~51 s, then drop to z ~7,770 and die (vz ~-1,100) about 1.5 s later; one
  episode falls early at 25.9%. Search 12's spine (to 41.4%) was passed by the greedy; search 13
  (5,066M) adopted a spine to d 57,547 = 42.3% from the frontier 57,916. Hand-on
  unr1b4_dc6/ckpt_5001707520.pt pulled (md5 962c8bff...); RESULT best -> 41.924.
* **11:12 UTC - v6 at 5.46B: best 43.843 at 5,436M**, CP3 (evals 5,134-5,436M 42.29-43.84; medians
  42.2-42.3% at 5,134 / 5,210 / 5,361M, 15.8% at 5,285 / 5,436M). At 5,436M 4/9 episodes reach
  42.3-43.8% (the best at (6,826, 305, 8,575) at 53 s, falling, then z 7,776), and 4/9 now die
  EARLY at 15.8% (~15.5 s, (5,818, 50, 13,030), falling) - an early-line instability while the
  curriculum walks search 13's spine back (starts 2,291-2,296 of 5,262, success 0-59.5%). Healthy:
  kl 0.019, yaw sigma 0.068, 212k steps/s. Hand-on unr1b4_dc6/ckpt_5251792896.pt pulled (md5
  e3ff0554...); RESULT best -> 43.843.
