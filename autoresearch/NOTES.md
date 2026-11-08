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
