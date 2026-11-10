# NOTES - the one-search series on surf_src_sidistic

Each slot agent appends what it learned under a dated heading at the END of this file (what it
built, the numbers, the ladder, what it means for the user's question). Read the operator's
section first. Everything below it is the unified-recipe program's log as its slot 3 left it
(the branch this program was forked from): the history of run 5, kept for reference.

## The one-search series (operator, 2026-11-09)

**What this is.** The user asked whether ONE thread (a search that also teaches the network,
"the way alpha 0 was trained") would do better than the hybrid that finished sidistic (PPO
beside a search). PROTOCOL.md has the user's words, the five parts of the hybrid, the four
slots and the rules. This section is the evidence you start from.

**Where things are.**

* This program: `AR_HOME=/home/steam/ar_az`, checkout `/home/steam/RL_Surf_az` (branch
  `az/main`), agents `azr1b1..4` on branches `ar/azr1b1..4`, tmux session `ar_az`.
* Run 5, read-only: `$AR_HOME/reference/run5/` (run.json, progress.csv, launch log, every eval,
  `search_ticks.json`, `auto_bundle.tgz` with all 51 searches' logs and summaries, 18
  checkpoint snapshots). Its own log is below in this file ("Round 1 FINAL lesson").
* The last program's other slots: `git show origin/ar/unr1b4:autoresearch/NOTES.md` (the knob
  as a rule, `tools/knob_ge.py`), `origin/ar/unr1b2` (the policy-guided beam), `origin/ar/unr1b1`
  (the Nature paper's random bursts). Their runs: `/home/steam/ar_uni/runs/` (read-only).

**Run 5 against the others (one seed each, the same recipe apart from the explore step).**

| slot | explore step | how far | when |
|---|---|---|---|
| b3 run 5 | MCTS, sampled children all random-held | finish, later 9/9 at 118.28 s | first finish 8.23B steps |
| b3 run 4 | MCTS, 0.5 random / 0.25 perturbed / 0.25 policy | 32.5%, failed the post-CP2 block | flat from 2.7B to 14.16B |
| b2 geb6 | policy-guided beam, `--unstuck-max 2` | CP7 (best 97.96), then its greedy fell back to the CP2 pit | CP7 at 14.87B |
| b4 v6 | the yield-rule knob + a windowed death charge | 44.1%, 6 walls on time | 5.96B |
| b1 v2 | random bursts (the Nature paper) | 32.5%, failed the post-CP2 block | 14.10B |
| control | no search | 1.72% | never left the first platform |

**What the search did in run 5.**

* A search line opened 4 of the 12 walls: the spawn exit (first ramp), the approach to the pit,
  the loop before CP6 (search 15; learned by PPO in 0.35B steps after a 1.8B creep) and the
  finish (search 20 found the rim and the shaft in 207 s; learned in 0.23B). The policy found
  the other 8 itself while the curriculum trained on the newest line. That includes the
  post-CP2 block, which 15 minutes of search alone did not open.
* 20 searches had started by the first finish: 2.57B simulator ticks beside 8.23B of rollouts
  (24% of the total). All 51 searches of the run: 3.99B ticks.
* One search: 2,048 game copies for 6 minutes at ~262k ticks/s (~92M ticks), ~930 waves, ~1.9M
  nodes, a few hundred archive cells, 16 roots on the last 20 s of the greedy line, macro of 12
  decisions (0.48 s), 16 children, c_puct 0.25, novelty weight 3, random temperature 5. The
  best line is the policy's own flight plus a tail of search macros (median 18 macros, 8.6 s).
* The tree's value is a node's best BANKED progress in its subtree (progress at a tick the map
  pushed back), min-max normalised. It is not a discounted return and the network's value
  function plays no part in it. The prior over children is uniform.

**The random-or-policy question so far.**

* In the trainer: run 4 (half random) failed where run 5 (all random) passed. One pair, one
  seed.
* Explorer alone (b4, 10 min each, same roots and seeds): at a wall where the policy was stuck,
  the yield rule reached 4.93%, all-policy 1.77%, all-random 2.76%; where the policy could fly,
  the rule 20.4%, all-policy 20.4%, all-random 2.8%. The rule sat at 52-66% random by itself.
* These two do not agree, and they measure different things (a tree against a chain explorer,
  a whole run against 10 minutes). Slot 2 exists to settle it inside the tree.
* b4's v2 kept one archive for the whole run and it was a null: old cells took ~99% of the
  selections.

**Known weak points of run 5's loop (observed, never fixed: the recipe was frozen).**

* Of the 51 searches, 33 gave a new line, 14 found nothing and 4 lines were discarded as stale
  (the policy had passed them while the search ran). The search roots on ONE greedy replay, and
  slot 3 of the last program traced 12 of those 18 misses to a replay that died early.
* Search 22 adopted a line far behind the policy because the frontier it was checked against
  came from one bad eval.
* After the first finish the trainer keeps firing a search every ~0.4B steps and re-adopts a
  finishing line each time, restarting the backward curriculum.
* The training win rate measures the spawns near the goal, never the policy. Only the greedy
  evals from the start count.

**The operator's view, to be tested and not believed.**

* PPO may be the hybrid's second explorer: it found 8 of 12 walls. A tree-only learner has to
  find those through the tree, or through its own stochastic play.
* A policy trained only on tree labels is trained on states the tree visits. The grader flies
  the greedy policy with no tree. The gap between the two is the classic imitation failure
  (DAgger); roots on the policy's own lines are the standard answer.
* An all-random tree teaches "hold this one action for half a second" at each labelled state.
  Whether that is a usable policy target is an open question; a tree with policy children gives
  closed-loop labels.

**Trainer facts checked for this program (train_fast.py on this branch).**

* `--auto-spine-scratch` writes `auto/spine0.npy` = the map's spawn state as the first "line",
  so the curriculum machinery runs from step 0 and every later line is the run's own search.
* `--demo-share S` puts the curriculum's window on a share S of every spawn pool "on top of
  the recipe's own spawn sources". Without it the reservoir share of the pool is replaced by
  curriculum-window states (`demo.build_pool(..., fresh_frac=1 - respawn_frac)`), and the
  trainer then REFUSES `--respawn-novel` and `--respawn-backward-mix`.
* A search starts on the quorum (enough greedy episodes past the line's success mark) or after
  `--auto-spine-patience` steps without a new line; after a search that found nothing it waits
  `--auto-spine-retry`. Launch-log lines: `[<step>] auto-spine search N: ...`, `... NEW SPINE
  from beam (...)`, `... STALE spine discarded ...`.
* The MCTS flags (`--mcts-*`) are defined in `tools/beam_tas.py`; the child modes are drawn in
  `tools/mcts_search.py` ("the knob"). The trainer passes extra ones through
  `--auto-spine-mcts-args "<one string>"`.
* Imitation machinery that exists: `--bc-file`, `--bc-coef`, `--bc-target argmax|dist`,
  `--bc-value-coef`; `tools/plan_to_bc.py`, `tools/expert_dagger.py`, `tools/exit_local.py` (the
  expert loop: search, distil, gate, in rounds). There is no supervised-only switch in the
  trainer. CLAUDE.md says `--keys-hold` is refused with `--bc-file`; check whether that still
  holds before you plan around it.

**Boxes, from the last round** (run 5's recipe): RTX 5080 ~365k steps/s, RTX 4080 ~280k, RTX
3090 ~215k, RTX 5060 Ti 125-180k. The trainer is bound by one CPU thread, so the host matters
as much as the card.

**Things that cost time last round.**

* `box.py pull --ckpt` of `ckpt_latest.pt` fails its md5 when the trainer rewrites the file
  during the copy. Pull a `ckpt_<step>.pt` snapshot instead, or pull again.
* Agents rented boxes whose deadline fell before the round's end. Check it on every wakeup.
* `box.py pull` does not bring `auto/`. Run 5's search bundle was tarred on the box and fetched
  by hand before the release; the per-search checkpoint copies (5.6 GB) were left behind.
* If Claude Code shows a "Teach auto mode about your environment?" dialog in your window,
  nobody but the user can dismiss it. Do not idle at a prompt: always end a turn with a
  scheduled wakeup.

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

### 07:14 UTC: window ON TIME, plateau at the pit-dive wall
* **Window (CP1) cleared at 1.057B, ON TIME** (deadline 2.07B; October 1.04B). Evals since 1.13B:
  27.0 / 27.1 / 27.1 / 27.1 / 27.1 / 27.1% (CP1) - the October pit-dive stall (26.9-27.1%).
* The plateau's episodes (1.51B eval): 8/9 last touch the map at 22.2 s at (3,030, 615, 11,870),
  BANKED 23.57%, then fly 3 s at ~1,500 u/s and die at 25.1 s at (3,010, 840, 11,210) - where raw d
  reads 27.1%. The greedy "frontier" is a fatal dive.
* Search 4 (1.36-1.45B): greedy pass dies at 24.85 s (raw 27.0%, banked 25.1%); 42 roots, 1,218
  waves, 2.5M nodes, 13,314 cells; best BANKED 25.86% - 2.3% deeper on contact than the greedy -
  rejected: the trainer compares it with the eval's RAW frontier (72,677 = 27.1%). The banked /
  raw asymmetry I flagged at launch now blocks the loop at exactly the wall it was built for.
  Fix for a next run (cannot touch this one): --auto-spine-banked, the greedy frontier and the
  quorum read the eval episodes' contact-banked depth.
* **Why the pit wall holds** (the goal field around sgcp_2, probed at y 0): ABOVE the pit slab
  (z 12-13k, x 800-2,000) the field reads 20-22% progress, BELOW it (z 10.5-11k) 28-31%: the way
  on is under the slab, into the cavity and through sgcp_2 (x 792, z 10,060-11,300; 30.5%). The
  policy's last contact is on the slab region at (3,030, 615, 11,870); it then flies west and dies
  at (3,010, 840, 11,210), 2,200 u short of the curtain. The MCTS tree (pit test, 3.0M nodes) has
  LIVE nodes as far west as x 600 - all at z >= 12,242, over the slab, where the field is behind.
* Explorer probes on the run's 1.6B plateau checkpoint (beside training, /root/t2):
  | explorer | cells | banked | raw best (live tick) |
  |---|---|---|---|
  | greedy | - | 23.57% (eval) / 25.1% (seed-1 pass) | 27.0-27.1% (dies) |
  | in-run search 4 (MCTS, random 0, nov 1) | 13,314 | 25.86% | - |
  | MCTS random 0.5, nov 1, 6 min | 17,196 | 25.86% | 27.08% |
  | MCTS random 0.5, **nov 3**, 6 min | **26,542** | **26.02%** | **27.57%** |
  | beam (policy samples, raw d) | - | - | 26.86% (dies) |
  Novelty buys coverage (1.5x cells) and a little depth; nothing banks a landing under the slab.
* For a next run (code ready, CPU-smoked): --auto-spine-banked (compare like with like), and the
  MCTS defaults random-frac 0.5 (first-ramp trap: 5.08% vs 1.77%) + novelty 3.

### 08:10 UTC: the pit wall is a PRECISION gate (map triggers read with surfgym.zones.parse_bsp)
* The plateau's flight (2.42B eval, ep 0): over the pit going east at 21.5 s (2,430, 632, 11,668),
  up the far side to (4,645, 335, 12,309) at 23.5 s, a turn, then west and down - dead at
  (3,100, 850, 11,200) at ~24.8 s, inside the pit-floor trigger_teleport -> sidstart (x 2,592-6,240,
  y +-1,152, z 10,752-11,332; --teleport-fail makes it a death). The cavity under the slab is
  bounded below by another teleport (x 1,312-2,400, z 9,536-10,632); entering it from the pit means
  crossing x = 2,592 inside a ~220 u window (z 11,332-11,552, under the teleport layer of x
  2,528-3,680) at ~2,000 u/s, clear of a thin 'no_jump' teleport at y 832-960. The policy's
  westward fall was heading for the cavity's height but crossed the pit floor 120+ u too low.
  (AABBs of trigger models are upper bounds - the real brushes can be thinner.)
* MCTS with roots through the fatal flight (--mcts-roots-flight 1, 8 s, 16 roots, random 0.5,
  nov 3, 6 min): banked 26.22%, raw 27.59%, 20,498 cells - no live branch through the window.
  The 20 deg held offsets are coarse for a 220 u window; testing 5 deg held offsets next.
* Fine perturbations at the pit gate (5 deg held offsets on every sampled child, keys temp 1, roots
  through the flight, nov 3, 6 min): banked **26.61%** (best of all variants; 20 deg: 26.22%), raw
  27.54%, 11,727 cells; nearly every child dies at the gate (11/2048 alive at wave 200).

### 08:20 UTC: run 1 STOPPED at 2.829B (pit-dive deadline 3.88B), run 2 launched
* unr1b3_mcts1 sat at 27.0-27.1% from 1.13B to 2.83B (1.7B steps, 23 evals). Its loop could not
  act at this wall by construction: searches 4-6 each banked 25.9-26.5% (deeper than the greedy's
  banked 23.6%) and each was compared with the greedy's RAW 27.1% fatal dive. Stopped ~40 min
  before its deadline so run 2 gets the time. Harvested: ckpt_latest (md5 200041b7...),
  ckpt_1000341504 (the snapshot at/before its best eval 27.126 @ 1.28B; md5 f2a0f547...).
  **Run 1 final ladder: first ramp 0.906B ON TIME, window 1.057B ON TIME, pit dive not cleared.**
* **Run 2 `unr1b3_mcts2`** (launched 08:21 UTC, commit 29202df0, same box): the incumbent recipe +
  run 1's extra flags + `--ratchet-contact 1 --auto-spine-banked --auto-spine-mcts-minutes 6` and
  the new MCTS defaults of this commit (random-frac 0.5, novelty 3, roots through a fatal flight,
  steer scale log-uniform 2-20 deg). One semantics everywhere: progress counts where the map pushes
  back - the reward (--ratchet-contact: a ballistic apex or a fatal dive pays nothing until a
  landing), the curriculum's success and bonus (paid_record), the quorum and frontier
  (--auto-spine-banked) and the MCTS's own score. It is a generic objective correction (no map
  constant) and it removes the reward run 1's policy collected for diving deeper into the pit.
* Run 2 at 0.52B: the same spawn trap as run 1 and the control (1.6-1.7%, evals 76-453M).
  Search 1 (0.5M-107M): stale by 59 u (the policy moved during the 6 min). **Search 2 (453-557M):
  the FIRST ADOPTED SPINE of this program** - greedy pass dies at 3.12 s (banked 1.43%, raw 1.59%),
  7 roots over 0.0-2.9 s (the spawn itself included), 1,419 waves, 2.9M nodes, 2,501 cells, best
  banked 2.86% (raw 4.24%) at 7.95 s, replay exact; vs the eval's BANKED frontier 98,258 -> +1,433 u
  -> NEW SPINE (795 states, end d 96,826, success line d <= 96,926). The banked comparison and the
  random children are what let the loop act at the trap where run 1's searches were stale or short.

### 09:15 UTC: run 2 FAILED the first ramp; a dead-end spine plus its bonus; run 3 without the bonus
* Run 2 after the adoption: the greedy learned the spine (2.8% from 0.68B, 9/9 past its success
  line, the curriculum walked back to tau 0 at 97% band success) - and stayed there: 2.84-2.85%
  through 1.06B -> **FAILED the first ramp** (deadline 1.0B). Search 3 (680-783M) found the same
  2.86% point (no gain); the next would have been 1.08B. Stopped at 1.139B (ckpt_latest pulled).
* Why: the spine is a DEAD END. It leaves the platform east like the dying greedy but drifts
  south (y -178 at 3 s: it survives the drop the greedy dies in), then runs south-east to
  (9,870, -990, 13,850) and stops at 2.86% with no speed. Run 1's own first-ramp route (its 0.906B
  eval) leaves the same way but lands NORTH (y +195), keeps ~1,330 u/s north-east to (9,878, 974)
  at 4 s and on east to the ramps at x 12,000-13,400 (8% at 9 s). The MCTS's banked score preferred
  the south branch (2.86% banked vs the north route's 2.80% at 4 s - it only pulls ahead at 4.5 s).
* And the bonus made the dead end the best-paid place on the map: --demo-success-bonus 50 for
  crossing its success line (paid ~841k times by 1.06B), kept forever by --auto-spine-milestones.
  Run 1 never adopted a spine, so nothing anchored its policy and it found the ramp by itself.
* **Run 3 `unr1b3_mcts3`** (09:15 UTC, commit 29202df0): run 2's recipe WITHOUT
  --demo-success-bonus and --auto-spine-milestones. The search's lines stay a spawn curriculum
  (25% of the pool, the backward walk on the success line) but reaching one is not paid, so a
  search line that is a local optimum cannot outbid the field. Everything else as run 2.
* Next explorer change (code 1f08b064, NOT in run 3, which runs 29202df0): --mcts-energy W - the
  search's value = banked progress + W x lambda x (energy height at the banked point - the
  spawn's) / d0, lambda = the checkpoint's own --energy-pot K x d0 / 1e5 (the trainer pays 100 per
  d0 of progress and K per 1000 u of h: 1 u of h ~ 3 u of progress at K 3); the line handed on is
  still the DEEPEST banked node (what the trainer's adoption compares). Why: run 2's dead end
  (2.86% banked, no speed) outscored the speed-keeping north route in a progress-only value.
  Testing on run 1's 0.7B trap checkpoint (W 1 vs W 0, 4 min each, beside run 3).
* Trap test (run 1's 0.7B checkpoint, the east-dive policy; new defaults: random 0.5, nov 3, flight
  roots, steer 2-20 deg; 4 min each beside run 3): **--mcts-energy 1: 5.55% banked at 7.98 s,
  8,952 cells (364 waves); energy 0: 5.55% at 13.53 s, 4,108 cells (870 waves)**. BOTH lines take
  run 1's NORTH route (land at y +93, (9,894, 1,027) at 4 s, east to the ramp complex at x ~12,200)
  - the route the policy found by itself at 0.9B - and stop at 5.55% where that route winds around
  the complex. The energy value keeps the speed (1,483 vs 1,227 u/s at 6 s, 5.5 s faster to the
  same point) and doubles the coverage per wave. Run 3 (no energy term) kept running: the gain is
  real but modest, not worth a restart; it goes into the next recipe.

### 09:56 UTC: run 3 stopped (first ramp failing), run 4 launched
* Run 3 (no bonus, contact ratchet): search 2 (378-485M) adopted a BETTER spine than run 2's -
  877 states, 5.0% banked: a south landing (y -574 at 4 s) then east to the ramp complex, ending
  STOPPED on a ledge at (12,180, 0, 13,081) at 8.8 s. The curriculum never got off its end
  (tau 791-796 of 876, band success 19-36% around the 20% advance rate) and the greedy stayed at
  1.8-1.9% (dies at 3.64 s at (9,299, -231, 13,853)). The policy degenerated: yaw sigma 1.28 at
  0.79B (runs 1/2: 0.17-0.26 at that point), 57% of training episodes crashing. Stopped at 0.827B
  (2 evals before the 1.0B deadline, the verdict clear).
* Reading across runs 1-3 at the first ramp (one seed each):
  | run | ratchet | spine adopted | bonus | first ramp |
  |---|---|---|---|---|
  | control | raw | - (no search) | - | FAILED (1.7% at 1.06B) |
  | 1 | raw | none (stale / short) | yes | **0.906B** |
  | 2 | contact | 2.86% dead end (south) | yes | FAILED (2.85%) |
  | 3 | contact | 5.0% ledge (south) | no | FAILED (1.8% at 0.79B) |
  Both contact-ratchet runs failed; the contact ratchet pays the platform drop (~1.5 s airborne)
  only on landing, and run 3's policy went entropic. Adopted spines ended in places the policy
  could not learn to leave (a dead end, a ledge).
* **Run 4 `unr1b3_mcts4`** (09:56 UTC, commit 5b59a646): run 1's recipe (RAW ratchet, spawn spine,
  demo-share 0.25) + --auto-spine-banked (adopt by banked depth) WITHOUT the success bonus /
  milestones, + the MCTS defaults of this commit (random 0.5, novelty 3, flight roots, steer 2-20
  deg, --mcts-energy 1). Its first iterations reproduce run 1's numbers EXACTLY (step 1.57M: kl
  0.0081, ev -0.099, best 1,164 u): the trainer is deterministic here, so until run 4 adopts a
  spine it should track run 1 - a controlled pair for what adoption does.

### 10:25 UTC: RUN 4 - the MCTS -> curriculum loop clears the first ramp 2.4x sooner (controlled pair)
* **Run 4 cleared the first ramp at 0.378B (14.9% best / 10.9% median), ON TIME; 18.3% at 0.454B.**
  Run 1 needed 0.906B; the control never (1.7% at 1.06B).
* The pair: runs 1 and 4 share the seed, the box and the trainer, which is deterministic here -
  their 0.5M evals are identical (1.166) and both sit at 1.7% through 0.15B. **Their search 1 found
  the SAME line** (end d 97,913 in run 1, 97,921 in run 4; 1.76% banked, 12 s): run 1 compared it
  with the eval's RAW frontier (97,964: the fatal dive) and discarded it as stale; run 4 compared it
  with the BANKED frontier (99,207) and adopted it at 106M. From there:
  | step | run 1 | run 4 | run 4's loop |
  |---|---|---|---|
  | 0.106B | - | - | search 1 adopted: 1.76% banked (12.1 s, 645 cells) |
  | 0.227B | 1.6% | **4.5%** | (9/9 past spine 1's line) |
  | 0.335B | - | - | search 2 adopted: 5.86% banked (7.76 s, 6,334 cells) |
  | 0.378B | 1.7% | **14.9%** | first ramp cleared |
  | 0.454B | 1.6% | **18.3%** | |
  | 0.486B | - | - | search 3 adopted: 15.13% banked (13.86 s, 27,814 cells) |
  (Run 4 also differs in the MCTS defaults - energy, random children, novelty 3, flight roots - and
  has no success bonus; but search 1, the adoption that split the pair, found the same point in
  both runs, so the split is the adoption rule.) One seed; the effect is 2.4x on a wall whose
  from-scratch spread is 0.18-1.51B, so it is suggestive, not proof - but the pair is controlled
  up to the first adoption, which the October noise floor is not.

### 10:55 UTC: run 4 - window ON TIME at 0.605B, then the pit wall
* **Window (CP1) cleared at 0.605B, ON TIME** (run 1: 1.057B; October 1.04B). 26.9% at 0.755B, then
  26.6-26.8% through 1.13B: the pit-dive wall again. Run 4's best eval 26.855 (< run 1's 27.126).
* Search 4 (0.64-0.71B) adopted a 33 s line, end d 72,827 (26.9% banked) vs the greedy's banked
  76,306 (23.4%); 2.78M nodes, **36,546 cells**, raw best 27.40%. The line loops over the pit twice,
  LANDS IN ITS EAST PART at z ~11,000 (x 4,400-5,200: inside the pit-floor teleport's AABB, so its
  real brushes are thinner than the box) and surfs back west at y ~ -900 toward the cavity - a
  candidate route under the slab, not the policy's fatal west dive.
* The curriculum walked back from its end to ~28.8 s on it (the pit landing) and stalls there at
  0% band success over ~500 eps per window; backing off 5 states per cooldown it needs ~200M steps
  to return to the end. The greedy stays at its banked 23.6% (0/9 past the 26.9% line), so searches
  now fire on patience (1.057B, 1.133B).

### 11:20 UTC: run 4's MCTS threads the pit gate (in search; the policy still learning it)
* Searches 5 (1.06-1.13B) and 6 (1.59-1.62B), both patience-triggered, adopted lines ending at d
  ~72,042 = **27.72% BANKED - past the pit-dive threshold (27.5%)**; 2.9M nodes, 30-31.5k cells,
  raw live best 28.4%. Spine 6 (29 s): over the pit, a landing in its east part (5,038, 563,
  11,015) at 25.5 s, a loop back west at y ~ -880, z 11,400-11,800, a contact at (3,109, -896,
  11,720) at 28.5 s, and **across the gate plane x = 2,592 at z 11,412** - inside the ~220 u window
  between the pit-floor teleport (top 11,332) and the layer above (11,552), at the other y (the
  'no_jump' trap is at y +832..+960) - alive at (2,518, -882, 11,412). The fine end of the steer
  mixture plus the patience searches found what the 6-min explorer-alone tests on run 1's policy
  did not.
* The policy has not followed yet: evals 26.7-26.9% (the greedy still dives at 26.8% raw, banked
  23.6%), 0/9 past the 27.7% line. The curriculum walked back to ~26.8 s on spine 6 (61.8% band
  success), then 3.2% at the last ~2 s - the gate itself. Pit-dive deadline 3.88B (~13:05 UTC).

### 11:48 UTC: **RUN 4 CLEARS THE PIT DIVE at 2.341B (27.872, ON TIME; deadline 3.88B)** - new best
* Evals 2.04-2.27B: 27.0 / 27.1 / 27.1 / 27.1%, then **27.9% best / 27.8% median at 2.341B**. The
  curriculum on the searched gate line had walked back to tau 0 (the true start, 80% band
  success); the greedy's banked frontier moved 76,170 -> 73,944 (23.6% -> 25.8%).
* The 2.341B eval: 7/9 episodes pass the gate window (x < 2,592 at z 11,332-11,552) into the
  cavity and reach 27.6-27.9% raw; 2/9 bank a contact inside it (27.80-27.82%) and live to 31-50 s.
  Most die on the cavity's floor teleport (x 2,208-2,528, z ~10,690) or the kill below it. The
  policy's own variant goes through at y ~ +650..+700 (the searched line went at y -880): the
  curriculum taught WHERE, not the exact line.
* The chain that did it, all by the run's own rules: patience search 5 (1.06B) / 6 (1.59B) - MCTS
  with flight roots, the 2-20 deg steer mixture, random children, novelty 3, energy value - found
  a 27.72%-banked line through the gate; the banked adoption rule took it; the backward curriculum
  (25% of spawns, no bonus) taught it from its end back to the start; the policy then flew it from
  the true start ~0.7B steps later. Run 1 (the same recipe without banked adoption) sat at this wall
  1.7B steps with its searches rejected.
* Hand-on: ckpt_latest at 2.378B (md5 checked by box.py pull), 37M after the eval (the 2.0B
  snapshot is 340M before it, so the newest is more than twice as near).
* Ladder, run 4: first ramp 0.378B / window 0.605B / pit dive 2.341B - all ON TIME. Next: the cavity
  (sgcp_2 at x ~792; deadline 8.27B - beyond this round's ~7.5B reach).

### 12:16 UTC: **run 4 clears the CAVITY (sgcp_2) at 2.718B - ON TIME (deadline 8.27B; October 4.14B)**
* Evals: 29.9% (2.57B, 2.64B), then **CP2 at 2.718B (31.6%)**, 31.7% best at 2.794B (score
  **31.743**, the new best), 31.6-31.7% through 2.94B - at the post-CP2 block (33%; October's
  stall 32.3-32.5%, deadline 13.9B). The quorum is met again (6-8/9 past the 70,669 line), so the
  searches keep firing. Hand-on: ckpt_latest at 3.0015B (md5 33d4ce94...), 208M after the best eval
  (the 2.0B snapshot is 794M before it).
* Run 4's ladder so far: first ramp 0.378B, window 0.605B, pit dive 2.341B, cavity 2.718B - every
  wall ON TIME, and walls 3-4 sooner than the October lineage (1.94B / 4.14B). Run 1 (raw adoption)
  never got past the pit; the control never past the first ramp.
* 12:45 UTC: run 4 at 3.61B, 31.6-32.3% (CP2) - at the post-CP2 block (33%, deadline 13.9B).
  Best eval **32.338 at 3.171B** (hand-on: ckpt_3001024512, 170M before it, md5 1208da42...).
  Search 10 (3.2B) adopted a line banking 30.2% (+232 u over the greedy's banked frontier);
  search 11 fired on patience at 3.55B.
* 13:12 UTC: run 4 at 4.23B, flat at 31.6-32.1% (CP2) since 2.72B - the post-CP2 block (33%).
  The plateau's episodes (4.15B eval): 7/9 pass sgcp_2 (x 792), last touch the map at ~31.5 s
  (banked ~30.0%), fly ~2 s west and die at (-400, ~150, 10,600) at 31.6-31.8% raw. sgcp_3 is at
  x ~0, z 8,940-9,508, y +-336: the line must DROP ~1,100 u more before x ~0 - another narrow
  window. Searches 11 (3.66B) and 12 (4.11B) adopted lines banking only 30.2-30.3% (+200..350 u
  over the greedy's banked frontier): at this wall the 6-min MCTS finds no deeper landing yet.
* 13:45 UTC: run 4 at 4.91B: median up to 32.4% (32.1-32.4% evals), best **32.431 at 4.681B**.
  Searches 13 (4.56B) / 14 (4.71B) adopted lines banking 32.1% / 32.4% (+2,100 / +280 u over the
  greedy's banked frontier); the greedy's banked frontier followed them, 30.0% -> 32.1%.
* 14:13 UTC: run 4 at 5.55B, 32.4% (median 32.4%) since 4.6B - below the 33% block. Search 15
  (5.16B) adopted another 32.4%-banked line; search 16 fired on patience at 5.51B. Box deadline
  extended to 16:42 UTC (past round end: a box that dies before the scoring is a known hazard).

## Round 1 lesson (unr1b3, slot 3: policy-prior MCTS) - the recipe that worked and why

**The recipe (run 4, `unr1b3_mcts4`, commit 5b59a646, from step 0, nothing changed after launch):**
the incumbent recipe + `--auto-spine --auto-spine-scratch --auto-spine-async --auto-spine-recheck
--auto-spine-banked --auto-spine-explorer mcts --auto-spine-mcts-minutes 6 --auto-spine-envs 2048
--auto-spine-ge-minutes 0 --auto-spine-patience 3e8 --auto-spine-retry 3e8 --demo-share 0.25
--demo-success-d 100 --demo-window 5 --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5
--demo-rate 0.2 --demo-min-ep 50 --demo-front-frac 0.5`, the MCTS at beam_tas's --mcts defaults of
that commit (M 12, K 16, c_puct 0.25, novelty 3, random-frac 0.5, perturb-frac 0.5, keys temp 1.5,
steer log-uniform 2-20 deg held with p 0.5, energy value 1, roots through a fatal flight, cells
128 u x 256 u energy). **To continue it in round 2: deploy its newest checkpoint and launch with
--resume and exactly these extra flags; diff run.json's config.**

**What decided it (in order of evidence):**
1. **Compare like with like.** A search line ends where it last BANKED progress (a contact); the
   eval's greedy frontier counts a fatal dive's raw depth. Comparing the two (run 1) rejected every
   useful line at the spawn trap and at the pit gate. `--auto-spine-banked` is the one-line fix, and
   it split two otherwise identical runs: same first search line, run 1 stale vs run 4 adopted ->
   first ramp at 0.906B vs 0.378B, and only run 4 ever got past the pit.
2. **Do not PAY for reaching a search line.** With --demo-success-bonus 50 (+ milestones), run 2's
   first spine - a south dead end at 2.86% - became the best-paid place on the map (paid ~841k
   times) and the run failed the first ramp. Without the bonus the line is only a spawn curriculum
   and a local optimum cannot outbid the field.
3. **Keep the raw ratchet.** Both --ratchet-contact runs (2, 3) failed the first ramp; run 3's
   yaw sigma blew up to 1.28 with 57% crashes (the ~1.5 s platform drop pays nothing until landing).
4. **What the MCTS needed** (explorer-alone tests on the October r0/r1 and runs' own checkpoints):
   c_puct 0.25, not 2 (with 2 the tree goes broad and shallow: 6.7% vs 15.6% from an idle start);
   half the sampled children RANDOM (first draw at temperature 5, held for the macro: 5.1% vs the
   beam's 1.8% at the spawn trap where the policy's own samples die with it); per-wave blocking of
   pending subtrees and a sibling-relative cell weight (else the batch collapses / exploration
   vanishes); roots THROUGH a fatal flight and a 2-20 deg steer mixture (fine offsets threaded the
   pit gate's ~220 u window: the line that cleared it banked 27.72%, found by patience searches 5/6).
5. **The curriculum taught WHERE, not the exact line**: the policy crossed the pit gate at y +650
   where the searched line went at y -880; the backward walk (25% of spawns) reached tau 0 first.

**Where it stops**: the post-CP2 block (33%): episodes pass sgcp_2 and die at (-400, ~150, 10,600);
sgcp_3 needs a ~1,100 u lower line near x ~0. Searches 13-15 found +0.3% at best there.
**Open**: the 100 u --demo-success-d margin stalls a backward walk at a hard stretch (run 4's spine
4: 0% band success at the pit landing for ~200M steps); --mcts-energy's value is shown only on one
trap (same depth, 5.5 s faster, 2.2x cells); one seed throughout.

### 15:37 UTC: OPERATOR NOTE - round 1 extended to 2026-11-09 12:35 UTC (wrap-up 11:55), the LAST round
* The user's decision (AR_BRANCH.json operator_note): a run ON TIME keeps training (no stopping it
  for a variant); a run that misses a wall deadline is stopped and a NEW recipe may start from
  scratch; --steps is only a stopping point (new runs --steps 60e9; a capped run is RESUMED from its
  own checkpoint with the same flags and a higher cap); boxes extended <= 12 h at a time; wake every
  45-60 min while a run just trains. The verdict is the ladder: which recipe got furthest, and when.
* So: **run 4 keeps training** (ON TIME on all 4 walls so far). Box extended to 2026-11-09 03:36 UTC.
  Its next deadline: the post-CP2 block at 13.93B (~20:40 UTC at 362k steps/s); its --steps 2e10
  cap arrives ~01:15 UTC -> resume with the same extra flags and --steps 60e9.
* 16:00 UTC, explorer-alone at the post-CP2 block (run 4's ~7.4B checkpoint, run 4's own MCTS
  defaults, **15 min** instead of the run's 6): greedy pass dies at 34.15 s (raw 32.43%, banked
  31.84%); 42 roots over 14.4-34.1 s; 2,911 waves, 5.96M nodes, **40,817 cells** (saturated by wave
  ~400); best banked **32.37%** at depth 4, raw best 32.45%. A 2.5x longer search does not open
  this wall: the tree covers the region and finds no landing lower down toward sgcp_3. Not the
  fallback recipe if run 4 misses its 13.93B deadline.
* 17:50 UTC, explorer-alone at the post-CP2 block with roots over the WHOLE greedy line
  (--mcts-roots-secs 40: 64 roots over 0-34.1 s, 10 min): 1,589 waves, 3.25M nodes, 41,286 cells,
  best banked **32.16%** (raw 32.47%) at depth 3 - no better than 20 s roots (32.37% in 15 min).
  Neither a longer search nor earlier roots opens this wall from run 4's policy: the tree saturates
  the region (31-41k cells) and every landing past ~32.4% is out of reach of the policy's samples,
  random held actions and 2-20 deg steers within the macro horizon. If run 4 misses 13.93B, the
  next recipe cannot be "the same search, longer".
* **The post-CP2 block is a kill-sheet maze** (map triggers, insight only): a 4 u teleport->start
  floor at z 10,576-10,580 over x -784..-144, y +-736 (run 4's episodes die ON it at (-400, ~150,
  10,600)); a func_wall block x -720..-464 wrapped in teleport skins (z 10,584-10,769) with a
  'no_jump' sheet above it (10,800); a teleport slab x -144..864, z 10,080-10,452 under the cavity
  exit; more thin sheets at z 10,320 (x ~-930), 10,096-10,100 (x -1,388..-944) and 9,680-9,696
  (x -1,780..-176); sgcp_3 at x 0..16, y +-320, z 8,976-9,472. The goal field: 32.0-32.4% around
  (-400..-800, z 10,000-10,400), 35.5% at (-800, z 9,600), 36.5% at (0, z 9,200) - the way on goes
  DOWN between sheets to the west and back east under them. Every searched branch that leaves the
  32% shelf dies on a sheet within a macro or two, so the trees stay 3-4 macros deep.
* 18:33 UTC, run 4 at 11.03B: still 32.4-32.5% (searches 26-28: one more 32.4% line, one null).
  **Plan at its 13.93B post-CP2 deadline (~20:45 UTC) if it misses: stop it (harvest first) and
  launch run 5 from scratch = run 4's recipe + `--auto-spine-mcts-args "--mcts-random-frac 1.0
  --mcts-perturb-frac 0"` (commit f5421ec9; --steps 60e9 per the operator note)** - the MCTS knob
  at the RANDOM end (15 of 16 children random-held, the 16th the policy's mode), everything else
  identical. It answers the program's question with the one thing that differs: does the
  policy-guided half of run 4's explorer matter (run 4 = the 0.5 mix: first ramp 0.378B, window
  0.605B, pit 2.341B, cavity 2.718B)? New flag --auto-spine-mcts-args puts the knob in the flag
  set (byte-identical when absent; unit test; CPU-smoked). The search code is unchanged since run
  4's commit (git diff 5b59a646..f5421ec9 of beam_tas/mcts_search/mcts_policy is empty).

### 20:52 UTC: run 4 FAILED the post-CP2 block (deadline 13.93B); run 5 launched (the knob at the random end)
* Run 4's last eval at or before the deadline (13.892B): 32.459 (best episode 32.5%, CP2, 0/9
  finishes). `ladder.py`: "STOP: a wall was missed" - **post-CP2 block FAILED** (deadline
  13,931,380,736). 188 evals; best **32.499 at 12.458B** (traj_12457607168; hand-on checkpoint
  ckpt_12004098048). Flat at 32.4-32.5% from ~4.6B to 14.16B: ~9.5B steps at one wall while the
  loop kept adopting 32.4% lines (search 33 at 13.17B: greedy 67,756 u, its line 67,316 u).
* Final ladder of run 4: first ramp 0.378B, window 0.605B, pit dive 2.341B, cavity 2.718B (all ON
  TIME; October 0.30 / 1.04 / 1.94 / 4.14B), post-CP2 block FAILED. Stopped at 14.161B (20:51 UTC),
  harvest pulled; box checkout -> 9eda7009 (code f5421ec9).
* **Run 5 (`unr1b3_mcts5`) launched 20:52 UTC from step 0**: run 4's recipe + `--auto-spine-mcts-args
  "--mcts-random-frac 1.0 --mcts-perturb-frac 0"`, `--steps 60e9`; record gate passed; run.json
  config carries the knob string intact. Box extended to 2026-11-09 08:54 UTC (one more extension
  after 00:45 covers the round end). Same seed and trainer as run 4, so the two should match until
  the first adopted search line - the first divergence is the explorer's own result.

### 21:21 UTC: run 5 (knob at the random end) - first ramp ON TIME at 0.529B
* **Search 1 on the IDENTICAL policy** (both runs' step-524k weights; runs 4 and 5 match bit for
  bit in every progress.csv training column until the first adopted line): run 5's MCTS (random
  1.0, perturb 0) 934 waves, 1.91M nodes, 446 cells, banked **1.771%** (raw 2.12%), line 1,454
  ticks; run 4's (0.5 / 0.5) 1,275 waves, 2.61M nodes, 645 cells, banked **1.764%**, 1,207 ticks.
  Same depth at the spawn trap; the all-random tree is slower per wave and covers 31% fewer cells.
* Then the runs part: run 4's greedy passed its line by 227M and search 2 (335M) gave a 5.9% line
  -> first ramp 0.378B. Run 5's greedy sat at 1.6-1.7% until 0.38B, then 4.57% (0.454B), **13.91%
  (0.529B: first ramp, ON TIME)**, 15.77% (0.605B). Its search 2 (started at 95,810) came back with
  a 93,949 line after the policy had reached 85,992 by itself: discarded STALE, next search at
  0.859B. So this ramp was the policy's own progress from the search-1 curriculum.
* First ramp so far: run 4 0.378B, run 5 0.529B, run 1 0.906B (raw adoption), the control never
  (0-1.7% to 1.06B). Run 4 and run 5 are both inside the first ramp's seed spread (0.18-1.51B);
  the matched-step comparison continues at the window (run 4 0.605B).

### 22:08 UTC: run 5 - window ON TIME at 0.680B; at the pit wall its searches do not thread the gate
* Ladder run 5: first ramp 0.529B, **window (CP1) 0.680B** (ON TIME; run 4 0.605B). Since 0.83B
  the evals sit at 26.78-26.91% (pit dive = 27.5%) - the same wall run 4 sat at 0.68-2.34B.
* The child mix (mcts_search.py): of the 15 sampled children per expansion, independent draws
  `explore = U < perturb_frac`, `rnd = U < random_frac`, `explore &= ~rnd`. Run 4 (0.5 / 0.5):
  50% random-held (temperature 5, first draw held for the macro), 25% perturbed (keys temp 1.5 +
  held steer log-uniform 2-20 deg), 25% the policy's own samples at temperature 1. Run 5
  (random 1.0): 100% random-held - NO perturbed and no plain policy children (perturb-frac is moot
  at random 1.0; the RNG stream is the same).
* Run 5's pit searches: k03 (0.94B) banked **25.861%** (raw 27.496%, 36,738 cells), k04 (1.43B)
  banked **25.866%** (raw 27.550%, 34,304 cells). Run 4's at the same wall: k04 26.683% (raw
  27.403%), then **k05 27.729% / k06 27.733%** (raw 28.4%) - the lines that threaded the ~220 u
  gate came from the PERTURBED children's fine steers. The random end has none: so far it reaches
  the gate (raw 27.5%) but banks nothing past it. Deadline for run 5's pit dive: 3.88B (~23:50 UTC).
* If run 5 fails the pit: run 6 = run 4's recipe + `--auto-spine-mcts-args "--mcts-random-frac 0"`
  (the POLICY end of the same knob: 50% perturbed, 50% plain samples, no random-held children),
  which completes a 3-point sweep of random-frac (0 / 0.5 / 1.0) with everything else fixed.

### 22:56 UTC: **RUN 5 CLEARS THE POST-CP2 BLOCK (2.341B) AND CP3 (2.643B) - six walls ON TIME; new best 36.914**
* Ladder run 5 (random end of the knob): first ramp 0.529B, window 0.680B, **pit dive 1.812B**,
  **cavity 2.114B**, **post-CP2 block 2.341B**, **CP3 2.643B** - all ON TIME (October 0.30 / 1.04
  / 1.94 / 4.14 / 6.97 / 7.32B; deadlines 1.00 / 2.07 / 3.88 / 8.27 / 13.93 / 14.64B). Evals: 27.87%
  (2.04B), 30.71% (2.11B), 34.96% (2.34B), 36.46% (2.42B), 36.80% (2.64B, sgcp_3), **36.914 at
  2.718B** (best; hand-on = ckpt_latest at exactly 2,718,433,280, snapshotted on the box to
  runs/unr1b3_mcts5_snaps/ and pulled). Run 4 (0.5 mix) never got past 32.5% in 14.16B steps.
* The pit: run 5's searches k05 (banked 26.94%) and k06 (30.04%, 59,643 cells) came after its
  policy had started through; the 22:08 worry (no perturbed children -> no fine steers) was
  wrong for this run: the policy threaded the pit itself at 1.81B from the k03/k04 curriculum.
* **How run 5 passes the post-CP2 block** (eval 2.718B, episode 2 vs run 4's best eval 12.46B):
  both reach (-750 .. -800, z ~9,750) after the cavity exit's launch. Run 4's episodes then turn
  back east and bounce on the 32% shelf (x -300 .. -850, z 9,770-10,320) until they fail. Run 5's
  go WEST, away from the goal: (-721, z 10,471) -> (-1,379, 10,331) -> (-1,953, 10,162) ->
  (-2,020, 10,113) -> down at (-1,961, 9,630) -> (-1,775, 9,197) -> (-1,669, 9,084) -> back EAST
  under the sheets: (-978, 9,397) ... toward sgcp_3 (x 0..16, z 8,976-9,472). That is around the
  west end of the 9,680-9,696 kill sheet (x -1,780 .. -176) - the detour the goal field implied
  (insight only). Run 5's searches did not find it first (k07 at 2.27B: banked 32.41%, then
  STALE): the POLICY found the detour itself between 2.27B and 2.34B.
* Two runs, one seed each, identical until the first adopted line: the difference is a sample of
  the path, not yet a law about the knob. What is measured: at the random end the loop reached
  every wall sooner from the pit on (1.81 vs 2.34B, cavity 2.11 vs 2.72B) and passed the block
  run 4 never passed.

### 23:50 UTC: run 5 clears CP4 (3.624B) and CP5 (3.851B) - eight walls ON TIME; best 70.676
* Ladder run 5: ... CP3 2.643B, **CP4 3.624B**, **CP5 3.851B** (October 9.39B / 11.78B; deadlines
  18.79B / 23.57B). Evals: 42.06% (3.02B), 46.18% (3.17B), 53.70% (3.47B), 55.79% (3.62B, sgcp_4),
  one 27.93% dip (3.70B, all 9 at the pit), 57.29% (3.78B, median 27.6%), **70.676% (3.851B,
  sgcp_5, median 69.3%)**. Searches 9-10 (3.28B, 3.58B) gave lines to 48.97% / 53.08% banked; the
  greedy kept passing them (search 11 launched at 3.85B from a 31,809 frontier).
* Checkpoint keeping: ckpt_latest is saved on a ~60 s clock (train_fast.py: "ckpt_latest is for
  crash recovery ... ~1-min cadence"), not at evals - except that an eval which TRIGGERS a search
  saves it at that step. A harvest helper on the box (runs/unr1b3_mcts5_snaps/best_keeper.py;
  reads only, never touches the trainer) copies every new ckpt_latest into a ring and, when an eval
  beats every earlier one, keeps the ring copy at or before it (PROTOCOL's hand-on rule, the copy
  after it only if more than twice as near). Kept so far: ckpt_2718433280 (36.914, exact),
  ckpt_3787456512 (57.288 at 3,775,397,888, +12.1M), ckpt_3850895360 (70.676, exact).

### 00:54 UTC: run 5 to 84.383 (4.908B), then the 84-86% creep before CP6; box extended to 12:47
* Box 59668027 extended at 00:47 to 2026-11-09T12:47:36Z (covers the 12:35 round end). Run 5 at
  5.13B, ~364k steps/s, all 8 walls ON TIME; CP6 deadline 28.52B is past the round's reach
  (~19.6B by 11:55), so it trains to the end (operator rule 1).
* Evals since CP5: 78.74 (3.93B), 79.64, 81.41, 82.53, **84.077 (4.228B)**, then a plateau:
  83.66 / 83.74 / 83.93 / 84.374 (4.530B, exact ckpt) / 84.29 / ... / **84.383 (4.908B, median
  84.3%)** / 84.18 / 84.21 (5.059B). Kept by the keeper and pulled (md5 verified):
  ckpt_4900519936.pt (-7,340,032 steps before the best eval: the hand-on checkpoint) and
  ckpt_4530372608.pt (exact, 84.374).
* Where it stops (MEASUREMENT only, insight for the notes - nothing enters the run): the greedy
  line flies EAST from x -9,600 (z 3,430), peaks at z ~4,770 near x -5,400, sinks ~1,000 u over
  the next ~2,100 u of x and is at (-3,300, y ~90, z 3,780) at t ~92 s, sinking 1,250 u/s. The goal
  field's route from there runs east through a gap in a solid block: at x -2,800 the passable cells
  start at z 4,072 (x -2,400: z 3,624+; the field's descent passes (-2,500, 154, 4,004) and
  (-1,700, 218, 3,748) on to sgcp_6 at x -200). The policy arrives ~300 u under the gap's floor,
  hits the face and either dies there or bounces back west and falls to the z ~3,044 floor. This
  is the October lineage's own stop ("stops at 72%, 83% and a creep 84-86%").
* The search saturates there too: k12 (4.41B) and k13 (4.86B) both end at d ~15,000 (84.9%;
  15,124 / 15,037). k13: 41 roots over ticks 7,296-9,216 (the last 19 s of the line), 1,371
  waves, 2.81M nodes, 46,737 cells with ~0 new per 10 waves for its last ~1,300 waves; its deepest
  raw d was 85.5% (inside the gap) but no line survived past 84.92%. So random-held macros from
  the policy's own line, rooted up to 19 s upstream, do not find the higher approach in 6 min.
  The curriculum's success line (d <= 15,137) is met by 84-93% of the last-decile spawns, so the
  loop is not stuck on the line it has - it has no line past the gap.

### 01:54 UTC: run 5 clears CP6 at 6.116B (nine walls ON TIME); best 91.524 at 6.493B - the search's loop taught it
* Ladder run 5: ... CP5 3.851B, **CP6 6.116B** (October 14.26B, deadline 28.52B). Evals: the creep
  held 84.08-84.35% from 4.23B to 5.96B (19 evals), then 85.00% (6.040B), **89.937 (6.116B, sgcp_6,
  median 89.8%)**, 90.91, 91.14, 91.06 (median 22.3%: early deaths), 91.22, **91.524 (6.493B,
  median 91.3%, exact checkpoint)**. Pulled, md5 verified: ckpt_6493306880.pt (the hand-on one)
  and ckpt_6115819520.pt (first CP6 eval).
* **How the gap was passed (measurement only).** Searches 13 and 14 (4.86B, 5.32B) ended at
  the gap's mouth ((-2,736, 98, 4,158) and (-2,890, 105, 4,068)) with no eastward speed. Search 15
  (5.77B; 42 roots over 77-97 s, 2.90M nodes, 41,941 cells) went further. Its line takes the
  first approach, falls to z ~3,100-3,400, turns BACK WEST
  (vx -1,980), circles at x -5,700 .. -5,830, comes east again climbing on a surface (vz +1,033 at
  (-4,274, -37, 3,886)), peaks at (-3,370, 100, 4,548) and enters the gap at (-2,795, 114, 4,291)
  moving east at 938 u/s (end d 14,827 = 85.1%). Its success line (d <= 14,939) was the first
  one inside the gap. 0.35B steps of curriculum later the policy's greedy flies the same loop:
  eval 6.493B ep0: first approach to (-3,565, 262, 4,033), back west to (-4,774, -689, 3,130) and
  (-5,656, 460, 3,330), east climbing (-3,935, 342, 3,273; vz +1,154), (-3,101, -537, 4,161),
  through the gap at (-2,272, 251, 4,256), past sgcp_6 (x ~ -200), down to 91.44% at
  (-1,136, 159, 2,807). That is the explore step doing its job: a recovery loop AWAY from the
  goal that the field's descent would not suggest, found by random-held macros, then learned.
* Search 16 (6.116B) returned NO line, and the reason is a failure mode of the loop: the search
  tool's own greedy replay of the checkpoint died at 12.2 s (d 97,924) while the trainer's eval
  of the same weights reached d 10,242, so its 26 roots sat in the first 12 s (best 15.35%). The
  trainer keeps search 15's line (5-7 of 9 greedy episodes past its success line) and retries
  after --auto-spine-retry 3e8 steps. A run whose greedy has early-death modes (the 6.342B eval:
  median 22.3%) can lose a search to one unlucky replay.

### 02:53 UTC: run 5 clears CP7 at 6.720B (ten walls ON TIME); best 98.01 at 7.022B; the basement stall
* Ladder run 5: ... CP6 6.116B, **CP7 6.720B** (October 15.23B, deadline 30.47B). Evals: 92.67
  (6.569B), 96.51 (6.644B), **97.012 (6.720B, sgcp_7)**, 97.98 (6.795B), then flat 97.84-98.01
  through 7.78B (best **98.010 at 7.022B, median 97.3%, exact checkpoint**; 98.007 at 7.324B).
  Pulled, md5 verified: ckpt_7021789184.pt (hand-on) and ckpt_6719799296.pt (first CP7 eval).
* Searches after CP6: k17 (6.569B; 42 roots over 86.9-106.6 s, 2.97M nodes but only 15,280
  cells, best 92.56%) came back STALE - the policy had already passed its line (greedy frontier
  d 6,298 vs the line's 7,416), so CP7 is the policy's own. k18 (7.022B) and k19 (7.475B)
  returned NO line: the search tool's greedy replay died at 27.2% (49 s, the pit) and stalled at
  22.6% (56 s), so their roots sat there (281 and 400 cells). The run's greedy has early-death
  modes: 4 of the 9 episodes of the 7.022B eval died at exactly 27.79% at (2,280, 735, 11,414),
  and 7 evals since 6.72B have medians of 29-87%. One search in three now goes to a replay
  that dies early. That is a property of the recipe: the search roots on a single greedy replay.
* Where it stops (MEASUREMENT only): best points (2,380-2,640, -620..470, z 1,000-1,350) at
  ~115-118 s, then a fall to the basement floor (z 693-711), which the field marks unreachable
  (d sentinel at (2,416, 396, 720)), and the eval's 30 s patience. The field's route from
  (2,637, 337, 1,165) climbs to (2,733, 209, 1,421), crosses at (2,925, 17, 1,357) and drops down
  a shaft at (2,893, -79) to the finish slab (x 2,872-3,048, y -88..88, z 176-184). The policy
  arrives under that rim. October: "basement stall 98.0-98.3%", chamber entry ~19.35B, "over
  the rim at 25,065M", first finish 25.45B.

### 03:52 UTC: RUN 5 FINISHES SIDISTIC FROM SCRATCH at 8.230B steps - all 12 walls ON TIME; best 113.759
* Ladder run 5 (one recipe, from step 0, launched 2026-11-08 20:52 UTC): first ramp 0.529B,
  window 0.680B, pit dive 1.812B, cavity 2.114B, post-CP2 block 2.341B, CP3 2.643B, CP4 3.624B,
  CP5 3.851B, CP6 6.116B, CP7 6.720B, **chamber entry 8.230B, finish 8.230B** (the eval at
  ~03:09 UTC, ~6.3 h after launch). The October lineage, with recipe changes between runs, took
  0.30 / 1.04 / 1.94 / 4.14 / 6.97 / 7.32 / 9.39 / 11.78 / 14.26 / 15.23 / 19.35 / 25.45B.
* Evals (score.py, goal hull None = plain-box trigger, box_only 0 everywhere): basement stall
  97.86-98.01 to 8.154B, then 108.182 (8.230B, 2/9 finishes, 121.21 s), 111.524 (8.305B, 5/9,
  exact checkpoint), 109.31 (3/9), 108.20 (2/9), **113.759 (8.532B, 7/9 finishes, 120.57 s, median
  100%)**, 110.42 (4/9), 108.22, 108.23, 111.56 (5/9), 111.54, 110.41, 112.68 (9.060B, 6/9, 119.51 s),
  112.65 (9.136B, 6/9). Every eval since 8.23B finishes 2-7 of 9; finish times 119.5-121.4 s
  (October's first greedy finish: 118.05 s).
* Hand-on checkpoint (PROTOCOL): ckpt_8506572800.pt, the newest at or before the best eval
  (-25,165,824 steps, ~70 s: the keeper's ring had nothing closer), md5 3aef0257... Also pulled
  and verified: ckpt_8305246208.pt (exact, 111.524, 5/9) and ckpt_8217690112.pt (-12.1M before
  the first finishing eval).
* **The finish came from the search.** Search 20 (launched at the 7.928B eval; its own greedy replay died
  at 117.98 s, d 2,046): 41 roots over 98.4-117.6 s, 843 waves, 1.73M nodes, 45,171 cells, a
  FINISHING line at 119.41 s found in 207 s (depth 22). The curriculum restarted at its end
  (success line d <= 100) and the greedy finished 0.23B steps later (8.230B). Searches 21 and 23
  then got the policy's own finishing replay as their line (0 s of search).
* The finishing line (eval 8.532B, 120.57 s; MEASUREMENT only): past sgcp_7 it flies low over the
  shaft area ((2,716, 470, 888)), out east to (3,981, -2, 950), back west at z ~770-880, climbs
  on a surface (vz +872 at (2,584, -462, 1,203)) to (3,080, -13, 1,675) above the rim, kills its
  horizontal speed over the shaft ((2,946, -9, 1,611), v ~(-30, -71)) and drops straight down
  (x 2,961, y ~0) to the finish slab at z ~180. Like the CP6 gap: a loop that comes back higher.
* A curriculum regression, recovered: search 22 (8.758B) started from a greedy replay that stalled
  at 40 s (1.44%) and adopted a line to 27.70% (d 72,085), because the frontier it had to beat was
  the median of an eval whose median episode died at the start (97,952); the curriculum
  restarted at 27.7% for ~42M steps until search 23 put the finishing line back (8.842B). The
  trainer's "N/9 past the success line" count stays 0/9 on finishing evals (its d check does not
  count a finished episode), so patience keeps launching searches beside the training.
* The early-death modes persist (medians 1.8-29% on some evals: the start, the pit at 27.8%).
  More finishes per eval is where the score still grows (10 points for 9/9), plus ~0.03 per second.

### 04:53 UTC: run 5 best 116.011 at 10.344B - 9/9 greedy finishes (119.68 s)
* Evals since 9.2B (score.py; box_only 0): finishes per eval 4, 0 (9.287B, 79.04), 6, 7, 2, 2, 4,
  7, 3, 4, 7, 5 (118.87 s, the fastest single finish so far, 10.042B), 4, 2, 5, **9/9 at 10.344B
  (116.011, best 119.68 s, median 100%)**, 5. Reliability is still the spread: 2-9 of 9.
* Hand-on checkpoint: ckpt_10321657856.pt (-22,020,096 steps before the eval; the keeper's ring
  had nothing closer), md5 b4e9d8f1..., pulled and verified.
* Searches 24-26 (9.32B, 9.77B, 10.23B) all came back with NO line: the trainer counts no
  finished episode as past the success line, so patience fires a search every ~0.45B steps, and
  each one replayed a greedy run that died early (frontiers 42,651 / 76,072 / 50,833) and rooted
  there. Cost: ~6 min at ~80% PPO throughput each; no regression, the curriculum keeps search
  23's finishing line.

### 05:53 UTC: run 5 at 11.91B - best still 116.011; every eval finishes (1-8 of 9)

- 20 evals since the best (10.42-11.85B): finishes 5,2,4,7,7,8,4,8,8,4,3,6,1,7,6,1,7,5,6,5 of 9
  (104/180 = 58% of greedy episodes); none 9/9 again, so 116.011 (10.344B) stands. Fastest single
  finish 118.36 s (11.174B, 3/9). Local score.py agrees with the keeper (no new best since 04:43).
- Searches 27-30 (patience: the trainer's success count never includes finished episodes, see
  04:53) all returned finishing lines (end d 0-3) in 21-23 s, because the greedy root they branch
  from already finishes; each one restarts the backward curriculum at its end. Searches 24-26 had
  found nothing (their greedy root died early). Since ~10.6B the loop re-adopts a finishing line
  every ~0.38B steps; the finish rate per eval did not trend up across them (2-8 of 9 throughout).
- Training: ~370k steps/s, entropy 0.006, sigma 0.065 (near-deterministic). The training win rate
  (60-73% at 16-23 s episodes) is the curriculum and reservoir spawning near the goal (reservoir
  min depth 1.16%), not the true-start rate; only the greedy evals measure the policy.
- Box 59668027 (deadline 12:47 UTC) covers the 11:55 wrap-up; no extension needed.

### 06:53 UTC: run 5 at 13.27B - best still 116.011; where the non-finishers stop

- 18 evals 11.93-13.21B: finishes 4,7,8,3,7,4,5,5,6,5,7,8,8,7,2,5,2,4 of 9 (97/162 = 60%), no
  9/9, fastest single finish 118.49 s (12.231B); 116.011 (10.344B) stands, keeper agrees.
- Searches 31 (finishing line, 23 s), 32 and 33 (NO line, 368-370 s): the root replay died early
  (beam frontier d 13,899 / 15,377 against a greedy frontier of ~2,200).
- Measurement over all 38 evals since the best (342 greedy episodes, 201 finished = 59%): the 141
  failures stop at 0-5% (7), 5-20% (12), 20-25% (17), 25-30% (29, 13 of them at 27.75-27.78% =
  the pit dive), 30-35% (18), 40-45% (18), 45-50% (8), 55-60% (14), 60-90% (16), and NONE past
  90%. So the late walls (CP6 gap, CP7 slit, chamber, finish) are solved; the misses are early
  and mid-route control slips, 45% of them in the 20-35% band around the pit dive and the
  cavity. Not a missing skill: the same checkpoint passes each of these places in most episodes.

### 07:54 UTC: run 5 best 116.035 at 14.572B - 9/9 greedy finishes again (118.96 s)

- New best 116.035 (local score.py agrees): 9/9 finishes, best 118.96 s, mean 119.28 s. Its
  checkpoint is EXACT (ckpt_latest saved at the eval step, offset 0), kept by the keeper, pulled,
  md5 ba314e7b86d4b069bd51598d07dea378 verified locally; RESULT.json best points at it.
- 19 evals 13.29-14.65B: finishes 7,7,5,5,2,4,7,5,8,8,5,6,7,2,8,7,4,9,8 of 9 (114/171 = 67%),
  up from 58% (10.42-11.85B) and 60% (11.93-13.21B). Searches 34 and 36 returned finishing lines
  (21-22 s), 35 nothing (root died early). Still ~371k steps/s, entropy 0.006.

### 08:54 UTC: run 5 at 16.0B - finish share keeps rising (71%), best still 116.035

- 17 evals 14.72-15.93B: finishes 7,7,8,5,9,7,7,9,5,1,7,5,4,7,7,7,6 of 9 (108/153 = 71%). The
  share of greedy episodes that finish has risen every stretch: 58% (10.42-11.85B), 60%
  (11.93-13.21B), 67% (13.29-14.65B), 71% (14.72-15.93B). Two more 9/9 evals (116.029 at 15.025B,
  119.12 s; 116.011 at 15.251B, 119.67 s), both slower than the best, so 116.035 (14.572B) stands.
- Searches: 38 a finishing line (22 s); 39 STALE (its root replay died early, end d 72,068); 40 no
  line. Trainer and keeper alive, ~372k steps/s, disk 22G/60G.

### 09:55 UTC: run 5 best 116.052 at 16.459B - 9/9 finishes, 118.45 s

- New best 116.052 (local score.py agrees): 9/9 finishes, best 118.45 s, mean 119.33 s. Hand-on
  checkpoint ckpt_16454254592.pt (ckpt_latest 4,718,592 steps = ~13 s before the eval), kept by
  the keeper, pulled, md5 fd2cd41d51f79189ec6cfd9d98ec1aee verified; RESULT.json best points at it.
- 19 evals 16.01-17.36B: finishes 6,6,7,4,4,8,9,5,3,5,7,7,8,1,2,9,7,5,7 of 9 (110/171 = 64%), so
  the share did NOT keep rising (58, 60, 67, 71, now 64%): it fluctuates around 60-70% with
  single bad evals (1/9 at 16.987B, 2/9 at 17.063B). 9/9 twice (16.459B, 17.138B). Fastest single
  finish 118.12 s (17.365B, 7/9). Searches 41 and 43 returned finishing lines, 42 nothing.

### 11:00 UTC: run 5 best 116.057 at 17.440B - 9/9 finishes, 118.28 s

- New best 116.057 (local score.py agrees): 9/9 finishes, best 118.28 s. Hand-on checkpoint
  ckpt_17420517376.pt (ckpt_latest 19,922,944 steps = ~54 s before the eval), kept by the keeper,
  pulled, md5 e71220cd95caa040f48a9e1137564b07 verified; RESULT.json best points at it.
- 21 evals 17.44-18.95B: finishes 9,8,7,4,4,5,8,4,4,5,6,5,5,6,8,6,6,7,8,6,4 of 9 (125/189 = 66%),
  and for the first time no eval below 4/9 (earlier stretches had 1/9 and 2/9 evals). Mean finish
  times are drifting down (118.45-119.3 s against 119-120 s at 10-12B). Searches 45-47 all
  returned finishing lines (21-23 s). Trainer and keeper alive; disk 24G/60G.

## Round 1 FINAL lesson (unr1b3, slot 3) - run 5 finished sidistic from scratch at 8.23B steps

This supersedes the run-4 lesson above (written before the extension). Run 4's notes there
still hold for what decided the first walls; this section is the round's verdict.

**The recipe (run 5, `unr1b3_mcts5`, commit 9eda7009 / code f5421ec9, from step 0 at
2026-11-08 20:52 UTC on box 59668027 (RTX 5080), nothing changed after launch):** run 4's recipe
(the incumbent flags + the auto-spine MCTS loop listed above) + ONE knob,
`--auto-spine-mcts-args "--mcts-random-frac 1.0 --mcts-perturb-frac 0"`: of the 16 children per
MCTS expansion, child 0 is the policy's greedy closed-loop macro and children 1-15 are
random-held (first decision at temperature 5 on every head, held for the 12-decision macro,
open loop). The full flag list is RESULT.json best.recipe.

**The ladder (ladder.py on the pulled evals; every wall ON TIME):**

| wall | run 5 | deadline | October |
|---|---|---|---|
| first ramp | 0.529B | 1.000B | 0.303B |
| window (CP1) | 0.680B | 2.073B | 1.037B |
| pit dive | 1.812B | 3.880B | 1.940B |
| cavity (CP2) | 2.114B | 8.272B | 4.136B |
| post-CP2 block | 2.341B | 13.931B | 6.966B |
| CP3 | 2.643B | 14.636B | 7.318B |
| CP4 | 3.624B | 18.785B | 9.393B |
| CP5 | 3.851B | 23.567B | 11.783B |
| CP6 | 6.116B | 28.523B | 14.262B |
| CP7 | 6.720B | 30.468B | 15.234B |
| chamber entry | 8.230B | 38.690B | 19.345B |
| finish | 8.230B | 50.898B | 25.449B |

First finish at 8.230B steps, ~6.3 h after launch (October's lineage, with recipe changes along
the way: 25.45B). Run 4 (same recipe, knob at 0.5 random / 0.25 perturbed / 0.25 policy samples)
cleared 4 walls ON TIME and FAILED the post-CP2 block (flat 32.4-32.5% from 2.7B to 14.16B).

**Best: 116.057 at 17.440B** (9/9 finishes, best 118.28 s); 9/9 in 9 evals from 10.344B on; fastest single finish 117.94 s (20.385B). Hand-on checkpoint ckpt_17420517376.pt (~54 s before
the eval, md5 e71220cd95caa040f48a9e1137564b07); the run's final state ckpt_latest.pt (step 20,459,814,912,
md5 3255478bc1e9b565a4474aa5f54c20fe) and the milestone ckpt_17005805568.pt were also pulled.

**What decided it (in order of evidence):**
1. **The explorer's children at the RANDOM end.** Runs 4 and 5 differ only in that knob and
   agree to the first adopted line; run 5 cleared the post-CP2 block at 2.341B, where run 4 sat at
   32.4-32.5% from 2.7B until it was stopped at 14.16B (deadline 13.93B). Policy samples and small perturbations of the policy near a wall repeat
   the policy's own mistake; random-held macros leave the policy's manifold, and the tree (c_puct
   0.25, max-backup, sibling-relative cell novelty) keeps the ones that bank progress. One pair,
   one seed: a paired observation, not a statistic.
2. **Two walls fell to search lines the policy then learned** through the backward curriculum
   (25% of spawns on the line, window grown from the line's end): the gap before CP6 (search 15's
   loop back west and the climb into the gap, learned in 0.35B after a 1.8B creep at 84-86%) and
   the finish (search 20 found the rim and the shaft in 207 s; learned in 0.23B, first finish
   8.230B). The other walls fell to the policy itself while the curriculum trained on the newest
   line.
3. **After the first finish the run kept improving without any change**: finishing share of greedy
   episodes 58% (10.4-11.9B) -> 60% -> 67% -> 71% -> 64% -> 66% (17.4-19.0B) -> 66%
   (19.0-20.4B), 9/9 evals from 10.34B on, best 9/9 time 119.68 s -> 118.28 s; fastest single finish
   117.94 s (20.385B).

**What did not work / recipe properties (observed, not fixed - the recipe is frozen):**
- The search roots on ONE greedy replay. When that replay dies early, the search finds nothing
  or a STALE line (searches 16, 18, 19, 24-26, 32, 33, 35, 39, 40, 42). Rooting on several evals'
  episodes (or the deepest one) would remove this.
- Search 22 adopted a 27.7% line (a regression of ~42M steps) because the frontier it was checked
  against came from one bad eval's median.
- The trainer's success counter does not count FINISHED episodes (they end at d ~2,200 before the
  d <= 100 line), so after the first finish a patience search fires every ~0.38-0.45B steps and
  re-adopts a finishing line (searches 27-47, 21-23 s each), restarting the backward curriculum
  each time. It cost ~6 min at ~80% throughput per search and did not stop the improvement.
- Non-finishers after the finish are control slips, not a missing skill: of 141 failed episodes in
  10.4-13.2B, 45% stop at 20-35% (pit dive, cavity) and none past 90%.
- Training win rate (60-73% at 16-27 s) measures the curriculum/reservoir harvest near the goal
  (reservoir min depth 1.16%), never the policy; only the greedy evals from the start do.

**Caveats:** one seed, one map. CLAUDE.md 0b calls a winner only when the SAME recipe repeats on
a second map with no constant changed; this recipe has not been run anywhere else. The MCTS
constants are generic (no map geometry, no record) but were chosen from explorer-alone tests on
this map's early walls in runs 1-4. Box: RTX 5080, ~364-372k steps/s while training.

**Wrap-up (12:00 UTC):** the first `box.py pull --ckpt` failed its md5 (ckpt_latest.pt is rewritten every
~60 s and the 80 MB scp took longer), so the trainer was paused (SIGSTOP by pid) 6 s after a save and
the pull repeated: ckpt_latest.pt at 20,459,814,912 verified. 271 evals pulled; score.py --best and
ladder.py on them give the table and the best above. Box 59668027 released.
Also pulled before the release (md5 166391c47a867e6c8efa23d33e3c9110 verified):
`runs/unr1b3/unr1b3_mcts5/auto_bundle.tgz` - every search's spine.npy, auto_spine.log, beam.log,
beam/summary.json and greedy_baseline.jsonl, plus the keeper's log and state. Not in it: the
spines' spine.keys.npy / spine.json and the per-search ckpt copies (5.6 GB), lost with the box.
`box.py status --agent unr1b3`: no box (12:32 UTC).

## azr1b2 (slot 2: a balanced tree) - 2026-11-09 20:56 UTC

**What this slot changes in run 5: part 2 only, the tree's child mix.** Run 5 fixed it at the
random end: of each expansion's 16 children, child 0 plays the policy's best action and children
1-15 are random-held (`--mcts-random-frac 1.0 --mcts-perturb-frac 0`). Run 4 (fixed 0.5 random /
0.25 perturbed / 0.25 policy samples) failed the post-CP2 block that run 5 passed. Here ONE rule
sets the shares of the three kinds of sampled child from the search's own results, inside every
search, from equal shares at its start.

**The rule (`--mcts-mix-rule sib`, tools/mcts_search.py "THE CHILD MIX", commit 96f81053).**
* Three kinds of sampled child: **P** = a plain policy sample (temperature 1), **E** = a
  perturbed policy sample (keys temperature 1.5 + a held steer, log-uniform 2-20 deg), **R** =
  random-held (first decision at temperature 5 on every head, held for the 12-decision macro).
  Child 0 (**G**, the policy's mode) stays as it is in every expansion.
* **What the yield counts: the child that beat all its siblings.** After every wave, in each
  expansion the best child earns its kind one point: best on the tree's OWN value (the
  backed-up banked-progress score, the number the tree selects on) when some child improved on
  its parent's value; when nobody went deeper (a wall), best on the new archive cells a LIVE
  child opened (the tree's own novelty measure; a child that died cannot be continued, so its
  cells do not count). Ties split the point; a win by child 0 is nobody's point (a sampled
  kind earns credit only by doing better than the policy's own move AND the other samples).
  Why this and not new cells alone: the tree's job is depth (the line it hands on is judged on
  banked progress) and new cells in this tree include cells a falling child opens on its way
  to dying; new cells decide only where depth cannot (at a wall nobody banks deeper).
* **The shares**: probability matching on each kind's points PER CHILD, decayed over 16 waves
  (`--mcts-mix-horizon 16`; a 6-min search runs ~1,400 waves, ~128 expansions each), with a
  floor of 5% per kind (`--mcts-mix-floor 0.05`, so a dominant kind can reach 90%), 2 warm
  waves at equal shares (`--mcts-mix-warm 2`). share_k = 0.05 + 0.85 x y_k / sum(y).
* Every constant is a search budget (waves, a floor): nothing names the map, a place, a speed
  or a progress value. Per search, not per run (each search starts at equal shares; the last
  program's slot 4 found a run-long archive a null).
* `--mcts-mix-rule log` = the fixed knob draw for draw with the rule's bookkeeping beside it
  (which kind wins, cells per kind, survival) - the bench's run-5 arm uses it.
* Off (the default) = the legacy draw: `legacy_child_kinds` is the old two-draw code verbatim;
  tests/python/test_mcts_mix.py checks it consumes the same random stream (and 22 other
  checks: shares, floor, tournament, ties, the tree's per-node kind). No trainer flag: the rule
  reaches the search through run 5's `--auto-spine-mcts-args` string, so record_ckpt is
  untouched.

**The run, as planned before launch:** run 5's recipe (`config.incumbent_recipe`, 126
elements) with ONE value replaced: `--auto-spine-mcts-args "--mcts-mix-rule sib
--mcts-mix-floor 0.05 --mcts-mix-horizon 16 --mcts-mix-warm 2"` (it replaces run 5's
`"--mcts-random-frac 1.0 --mcts-perturb-frac 0"`, which the rule makes moot). Search cadence,
budget (6 min, 2,048 copies), roots, macro, c_puct, novelty, energy value, curriculum and both
spawn pools exactly as run 5. `--steps 60e9`. From step 0.

**Box** 59951396 (RTX 5080, 32 threads, $0.42/h, Ohio), rented 20:53 UTC, deadline 12 h.

### 21:15 UTC: identity check and the paired bench at the CP6 creep (measurement only)

* **Flag-off identity on the GPU** (box 59951396, run 5's ckpt_4900519936, seed 13, 15 waves):
  the base commit's search (09099d2a) vs this branch with `--mcts-mix-rule off` and with `log`:
  30,762 nodes each, every array of mcts_tree.npz, the summary and the replayed line equal (log
  only adds its own summary keys). The rule changes nothing unless it is switched on.
* **The pair** (run 5's 4.90B checkpoint, inside its 84-86% creep before CP6; the command line
  of run 5's own searches, seed 13, max-ticks 13,771, 6 min each, alone on the 5080 at ~550-580k
  ticks/s, so ~2,000 waves against ~1,400 beside training):

  | arm | waves | cells | best banked | raw best | best line's macros |
  |---|---|---|---|---|---|
  | run 5's tree (15 random-held + child 0) | 1,968 | 34,168 | 84.54% | 84.61% | RGRGRGRGRRGRR |
  | the rule, memory 16 waves | 2,077 | 39,680 | **84.83%** | 84.95% | RGREEEPRRPRRE |

  The rule went 285 u deeper and covered 16% more cells; neither found the loop back west that
  run 5's search 15 found from the 5.66B weights (85.13%), so neither crosses into the gap here.
  Mean shares of the rule P 0.29 / E 0.35 / R 0.36. Wins per child over the search: child 0
  0.0047, P 0.0006, E 0.0016, R 0.0023 (run 5's arm: child 0 0.0060, R 0.0014). The best line
  mixes all three kinds (G = child 0, P sample, E perturbed, R random-held).
* **A flaw, fixed before launch.** Both trees saturate after ~200 waves (cells +0..+30 per 10
  waves) and wins become rare (~2 per wave against tens early on). With a memory of 16 WAVES a
  handful of tail wins swung the shares (P 0.2 -> 0.88 -> 0.2 within 100 waves). The tail is where
  run 5's wall-opening lines came from (search 15's loop at wave ~392 of 1,415, search 20's finish
  at ~832 of 843; searches 4-6 and 13 peaked by wave ~50-110), so the tail's mix has to follow
  evidence. Commit 8cb13c12: the memory is counted in WINS (`--mcts-mix-horizon 256`: each win
  decays the old points and children by exp(-1/256); a wave without a win changes nothing, and
  children drawn in proportion to the shares leave the ratios where they were). Test: shares
  hold through 400 winless waves and move on the next burst. Re-check arm running.
* **The re-check** (the win-counted memory, same checkpoint, seed and budget): 1,690 waves,
  28,383 cells, best **84.28%** banked (depth 5, from wave ~60 on), steady shares P 0.20 / E
  0.37 / R 0.43 the whole search. Three single-seed trees from one checkpoint: run 5's 84.54%,
  the rule with a 16-wave memory 84.83%, with the win memory 84.28%. The spread between arms is
  as large as any difference, so 6-minute trees at one seed cannot rank these variants (the same
  binary-gate pathology CLAUDE.md documents for evals). What the bench does say: the rule runs,
  its tree reaches about the depth of run 5's tree at this wall, it keeps all three kinds alive
  (wins per child: R > E > P > 0), and the best lines mix all kinds. The win-counted memory is
  kept for the run: it is the version whose shares follow evidence, and the tail is where run 5's
  lines came from. Bench box time: ~27 min (identity 1, three arms 6.3 each).

### 21:21 UTC: run `azr1b2_sib` launched from step 0 (commit 932167fe, box 59951396)

`python autoresearch/box.py launch --agent azr1b2 --run azr1b2_sib --steps 60e9 --
--auto-spine-mcts-args "--mcts-mix-rule sib --mcts-mix-floor 0.05 --mcts-mix-horizon 256
--mcts-mix-warm 2"` = run 5's 126 flags + that one value flag (the trainer keeps the last, so it
replaces run 5's `"--mcts-random-frac 1.0 --mcts-perturb-frac 0"`; run.json's
`auto_spine_mcts_args` reads the rule string). Search cadence, budget, roots, curriculum and both
spawn pools are run 5's. Record gate passed (4 recordings, 34 s). Search 1 fired at step 524,288
as in run 5, its beam.log reads `child mix rule sib (floor 0.05, horizon 256 wins, warm 2)`.
Provenance of every spawn file: the map spawn (auto/spine0.npy) and the run's own searches on its
own weights; no demo, no file passed by hand. Box deadline extended to 2026-11-10 09:13 UTC (the
watchdog's extend is now+12 h, monotone: extend again on later wakeups until it covers 22:40).

### 21:36 UTC: azr1b2_sib at 0.21B - throughput, search 1

* **Box speed**: i9-14900KF (32 threads) + RTX 5080. ~295k steps/s with no search beside the
  trainer (sampled 202.9M -> 207.6M in 16 s), 250k average over the first 13.5 min with search 1
  beside it. Run 5's 5080 did 364-372k: this box is ~20% slower, well above the 200k floor. The
  search beside training runs ~168k ticks/s (run 5's: ~262k). At ~260k average the finish
  deadline (16.46B) falls near 15:00 UTC on 2026-11-10, before the credit cutoff (~19:00 UTC,
  operator note 2).
* **Search 1** (524k -> 92.8M steps, 358 s, beside training): the untrained greedy dies at once
  (0.17%), 6 roots, 613 waves, 1.26M nodes, only 302 cells (the spawn platform), best banked
  1.77% (d 97,912, depth 51 = 24 s of macros) - the same point run 5's search 1 found (1.771%).
  ADOPTED at 92.8M (run 5's was adopted at ~106M in run 4 / its own search 1). The rule: mean
  shares P 0.26 / E 0.25 / R 0.49; wins per child R 0.0149, child 0 0.0123, P 0.0071, E 0.0065
  (all on value: at the spawn every expansion's best child is the one that banks deeper); the
  line's 51 macros: RPRRPRRRRPPPRPRRERRRRRPRPRERRGGRERRRRERRPPRRGRRPPR (R 31, P 13, E 3, G 3).
* Evals 0.5M / 76M / 151M: 1.68% best (CP0), the spawn trap, as run 5 (1.6-1.7% to 0.38B).

### 22:05 UTC: azr1b2_sib at 0.69B - off the spawn platform (4.4%), first ramp pending

* Evals: 1.6-1.7% to 0.53B (the spawn trap, as run 5 to 0.38B), then **4.44% at 0.604B and
  0.680B** (median 4.4%). First ramp (8%) not yet: run 5 0.529B, deadline 1.600B. Throughput
  282k steps/s average since launch (searches included).
* Searches (`autoresearch/analysis/mix_searches.py` on the pulled auto/ logs):

  | k | start | waves | cells | greedy | best banked | mean P/E/R | wins/child G P E R | line | outcome |
  |---|---|---|---|---|---|---|---|---|---|
  | 1 | 0.5M | 613 | 302 | 0.17% | 1.77% | 0.26/0.25/0.49 | .0123 .0071 .0065 .0149 | G3 P12 E4 R31 | adopted 92.8M |
  | 2 | 453M (patience) | 747 | 678 | 1.46% | 2.33% | 0.31/0.33/0.36 | .0095 .0069 .0073 .0082 | E3 R2 | adopted 538M |
  | 3 | 680M (quorum) | running | | 3.67% median | 5.77% at wave 230 | 0.35/0.36/0.30 at wave 230 | | | |

  The policy learned search 2's line within one eval (4.4% at 604M). At the spawn the rule leaned
  random (R 0.49); one step further on, the kinds' win rates were close and the shares near equal.

### 22:32 UTC: azr1b2_sib at 1.13B - first ramp and window ON TIME; at the pit-dive wall

* **Ladder (steps)**: first ramp **0.755B** ON TIME (run 5 0.529B, deadline 1.600B); window
  (CP1) **0.906B** ON TIME (run 5 0.680B, deadline 1.600B). Evals 0.83B 20.7%, 0.91B 26.8%
  (CP1), then 26.9 / 26.8 / 26.9% to 1.13B: the pit-dive wall (27.5%; run 5 1.812B, deadline
  3.625B). The 1.057B eval's median was 15.8% (early-death episodes, as in run 5).
* Search 3 (680M, quorum on search 2's line) found 5.77% banked and came back STALE: the policy
  had flown to 14.2% by itself in the 6 min (as run 5's search 2). So the first ramp and the
  window were the policy's own progress from the search-2 curriculum, exactly the pattern of run
  5. Search 3's rule: mean P 0.38 / E 0.37 / R 0.26, final 0.43 / 0.46 / 0.11; wins per child G
  0.0022, P 0.0010, E 0.0011, R 0.0017.
* Search 4 (1.133B, quorum) at the pit: 26.63% banked by wave ~550 (the greedy's banked frontier
  ~23.6%), shares P 0.15 / E 0.35 / R 0.50. Run 5's pit searches banked 25.86 / 25.87 / 26.94%
  (k03-k05) before its policy threaded the gate itself at 1.81B; run 4's mixed tree threaded it
  in search (27.72%, k05/k06).

### 23:04 UTC: azr1b2_sib at 1.74B - at the pit gate, 27.1% raw (wall 27.5%), two pit lines adopted

* Evals 1.21-1.74B: 26.3, 26.9, 26.9, then 27.1% x5 (medians 26.1-27.1%), CP1. Pit dive (27.5%)
  not yet; run 5's policy threaded it at 1.812B; deadline 3.625B (~1.8 h at this box's speed).
* Searches at the pit (both adopted: they bank deeper than the greedy's banked 23.58%):

  | k | start | waves | cells | best banked | raw | mean P/E/R | wins/child G P E R | line |
  |---|---|---|---|---|---|---|---|---|
  | 4 | 1.133B quorum | 786 | 27,907 | 26.63% | 27.59% | 0.18/0.33/0.49 | .0091 .0010 .0024 .0038 | P1 E2 |
  | 5 | 1.586B patience | 894 | 28,152 | 26.58% | 27.56% | 0.19/0.31/0.50 | .0077 .0008 .0023 .0038 | G2 P1 E1 R8 |

  Run 5's pit searches (all random-held) banked 25.86 / 25.87 / 26.94% (k03-k05), run 4's mixed
  tree 26.68 / 27.72 / 27.73% (k04-k06, its fine perturbed steers threaded the gate). The rule
  here: ~half random-held, a third perturbed, P near 0.2; per child R wins most, E next, P least.
  Neither of my pit lines is through the gate (27.5% = d 72,264; lines end at d 73,141 / 73,192).

### 23:46 UTC: **PIT DIVE ON TIME at 2.492B** - the rule's tree threaded the gate (perturbed children)

* Ladder (steps): first ramp 0.755B, window 0.906B, **pit dive 2.492B** (27.55%, median 27.5%) -
  all ON TIME (run 5 0.529 / 0.680 / 1.812B; deadlines 1.600 / 1.600 / 3.625B). Evals 1.81-2.42B sat
  at 27.0% (the fatal dive at the gate) until search 6's line was learned.
* **Search 6** (2.039B, patience; adopted 2.122B): 796 waves, 35,840 cells, best **27.73% banked**
  (raw 28.30%) by wave ~64, i.e. THROUGH the gate (27.5% = d 72,264; line end d 72,049). Mean
  shares P 0.19 / E 0.31 / R 0.50. Its 12 macros: **E P R G P E G E G R E E** (E5, P2, R2, child
  0 x3) - the gate was threaded by PERTURBED children (held fine steers, log-uniform 2-20 deg, keys
  temperature 1.5), ending on two of them. The policy learned it in 0.37B steps (2.122 -> 2.492B).
* Against the earlier runs at this gate: run 4 (fixed 0.5 R / 0.25 E / 0.25 P) threaded it in
  search at 27.72% (k05/k06) and its policy passed at 2.341B; run 5 (all random-held, no E
  children) never threaded it in search (25.86-26.94% banked) and its policy passed by itself at
  1.812B. This run passed later than both (0.68B after run 5, inside the 2x deadline) and by the
  same route as run 4. The rule kept E at ~0.31 although R won more per child here (R 0.0057, E
  0.0036, P 0.0014 per child): the floor never bound; probability matching left E a third.
* Total unit (steps + search ticks; search_ticks.json: 7 searches, 525M ticks by 2.49B): first
  ramp 0.962B, window 1.113B, pit dive 3.017B - ON TIME (run 5 0.763 / 0.914 / 2.448B; deadlines
  2.000 / 2.000 / 4.897B). A search here spends ~60-80M ticks (6 min at 170-220k ticks/s beside
  the trainer on this box) against ~92M in run 5.

### 00:35 UTC: cavity ON TIME at 2.718B; at the post-CP2 block (run 4's wall) since 2.87B

* Ladder (steps): ... pit dive 2.492B, **cavity (CP2) 2.718B** ON TIME (run 5 2.114B, deadline
  4.229B). Evals since: 31.6, 31.6, **32.45 (2.869B)**, 32.2, 31.7, 32.1, 32.2, 32.2% (medians
  31.3-31.8%), CP2 - under the post-CP2 block (33%; run 5 2.341B; deadline 4.682B, ~1.4 h at
  ~290k steps/s). Run 4 sat at 32.4-32.5% here from 4.6B to 14.16B and failed; run 5's POLICY
  found the westward detour itself (its search 7 was stale).
* Searches 7-9:

  | k | start | waves | cells | greedy | best banked | raw | mean P/E/R | wins/child G P E R | outcome |
  |---|---|---|---|---|---|---|---|---|---|
  | 7 | 2.492B quorum | 741 | 45,401 | 25.77% | 27.71% | 29.34% | 0.18/0.32/0.50 | .0148 .0015 .0047 .0063 | adopted 2.572B (the gate line again, G1 P1 E3 R1) |
  | 8 | 2.718B quorum | 825 | 28,326 | 29.87% | 29.97% | 31.16% | 0.14/0.32/0.54 | .0072 .0006 .0023 .0036 | no line (+64 u over the greedy's banked 30.0%) |
  | 9 | 3.171B quorum | 786 | 40,949 | 25.73% | 29.30% | 29.30% | 0.18/0.32/0.50 | .0115 .0013 .0038 .0056 | no line: its own greedy replay died at the pit (25.7% banked) while the eval's median banked 30.0% |

  Search 9 is the known failure mode of run 5's loop (one greedy replay as the roots; NOTES
  "Known weak points"); not part of this slot's change. The curriculum still trains on search 7's
  line (6-9 of 9 greedy episodes pass its success line, so searches keep firing on the quorum).
* The rule's shares have been steady across the pit and the block: R 0.50-0.54, E 0.31-0.32, P
  0.14-0.19; per child R > E > P every time, child 0 above all.
* Other slots (status.md 00:29): slot 1 (one pool) walls 1-7 ON TIME, CP4 at 3.55B; slot 3 at
  28.2% before CP2; slot 4 v1 failed the first ramp.
* **Planned next recipe, written before any launch (used ONLY if azr1b2_sib fails a wall):**
  one pool + the balanced tree = slot 1's flag list (its RESULT.json, commit da7981e5: run 5's
  126 flags without `--respawn-novel*`, `--respawn-greedy*`, `--respawn-front-perturb*`,
  `--respawn-backward-mix`, `--respawn-backward-elev`, `--demo-share`, `--int-real-only`; 85
  elements, launched with `--no-recipe`) with its last value replaced by this slot's rule string
  `--auto-spine-mcts-args "--mcts-mix-rule sib --mcts-mix-floor 0.05 --mcts-mix-horizon 256
  --mcts-mix-warm 2"`. Taken from slot 1 (azr1b1_pool1, walls 1-7 ON TIME by 3.55B). It is the
  protocol's own example of a combined second recipe.

### 01:15 UTC: azr1b2_sib at 4.00B - flat under the post-CP2 block (31.6-32.4%), deadline 4.682B

* Evals 3.32-4.00B: 32.3, 32.3, 32.4, 31.9, 32.2, 32.2, 32.2, 31.7, 32.2, 32.4% (medians
  31.6-32.1%); best still 32.449 at 2.869B. Search 10 (3.624B, quorum): no line (+63 u over the
  greedy's banked 69,780 = 30.0%). Searches 8-10 at this block found nothing deeper; run 4's
  searches here found +200-350 u (3.2-4.1B) and 32.1-32.4% lines (4.56-4.71B), none past the block.
* Where the 4.00B eval's episodes stop (MEASUREMENT only): 5/9 pass the cavity and die at x -300
  .. -900, z 10,400-10,800 (ep 3 the deepest, (-889, 301, 9,714) at 34.9 s) - run 4's plateau
  ((-400, ~150, 10,600), the 4 u teleport floor); 3/9 reach x ~880 then turn back east and fall
  into the pit floor (x ~5,400, z ~10,780); 1/9 dies at 15 s over the pit. Run 5's policy went
  further WEST (x -1,400 .. -2,000) and came back east under the sheets.
* Pulled (md5 verified): ckpt_2000683008.pt and ckpt_3001024512.pt (the milestones around the
  best eval at 2.869B; the 3.0B one is 0.13B after it, the 2.0B one 0.87B before).
* The last eval that can clear this wall is the one AT the deadline (4,681,891,840 is an eval
  step); if it misses, the run is stopped, harvested, and the planned second recipe launched.
