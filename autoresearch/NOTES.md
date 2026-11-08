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
