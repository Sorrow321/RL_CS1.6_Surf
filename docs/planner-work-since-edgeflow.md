# Planner / executor work since the EDGEFLOW / LABYRINTH detour benchmark (2026-09-23 .. 2026-09-25)

Compiled 2026-09-25 14:55-15:10 local time (+0200) from the repository (branch `petrusnight`) for an
independent verification pass. Everything below was read from the repo: the ledger, the commits, the
code, the run directories. Nothing was trained, launched or tested for this document; the only
computation was reading `progress.csv` files and launch logs with a short script. The repo moved
while this was written: HEAD went from `7ca92ea` (14:35) to `e48d76b` (14:52, MCTS, section 3.21)
and then to `7a40522` (15:05, MCTS knobs); HEAD at 15:08 was 7a40522.

Addendum (main session, after this document was written): ledger commit `930ddfa` (heading
"2026-09-25 15:08") now records the tracking metric, the override ablation, flat_b050 through
905M and the MCTS comparison of section 3.21, so the "no ledger entry" notes in 1.4, 2.7 and 3.21
are out of date for those items. `runs/mcts_eval/b025_mctsfast.log` finished after 3.21 was
written: `--plan-mcts 32 --plan-mcts-k 8 --plan-mcts-depth 5 --plan-mcts-time --plan-mcts-gamma
0.7 --plan-mcts-uniform 0.5` on blue025 = 9/9, 10.1 s mean (fidelity 34/34, median 1 u). The
ledger inconsistencies listed in section 7 are real and have not been corrected yet.

Conventions

* `L:NNNNN` = line NNNNN of `docs/research-results.md` (the ledger; append-only, 27,353 lines, not
  modified after 7ca92ea; the last entry is "2026-09-25 11:34 (machine clock) - fleet empty" at
  L:27347).
* `file:NNN` = a line of that file AT COMMIT 7ca92ea, except in section 3.21 (at e48d76b). Four files
  changed after 7ca92ea (in e48d76b and 7a40522): python/surfgym/goalprimplan.py (lines >= 921 moved), python/surfgym/
  goalsearch.py (rewritten), tools/record_ckpt.py (lines >= 620 moved), tests/python/
  test_goal_primlearn.py (lines >= 178 moved). For those, read the cited lines with
  `git show 7ca92ea:<path>`.
* Commit hashes are on `petrusnight`; commit times are author times, local +0200
  (`git show -s --format=%ad a56ae10` -> `... 14:35:16 2026 +0200`).
* Timeline rows give the ledger heading's own time and, in brackets, the commit that introduced it
  (and code commits); section 7.1 lists the headings whose time is later than that commit.
* "k/9 greedy" = the trainer's own greedy eval from the map start, 9 episodes, planner greedy +
  executor greedy. A run's launch log prints it as `plan-eval finish k/9`; primlearn runs also write
  `plan/eval_finish` = k/9 to `progress.csv`. Labyrinth jump-planner and new-maze runs used 3.
* [checked] = the number was re-read for this document from the run's `progress.csv` or launch log,
  not only from the ledger. UNVERIFIED = could not be confirmed from the repo.

## 0. Scope

* The start date given for this work (2026-09-22/23) is the start of the PLANNER program. The ledger
  dates the benchmark maps earlier: the edgeflow dip ladder `surf_edgeflow_blue025/050/100/200` on
  "2026-09-15 23:55" (L:22975) and the labyrinth ladder `labyrinth_left025..200` on
  "2026-09-20 16:15" (L:23557). Covered in detail here: from the first 2026-09-23 entry (L:24423,
  commit 4b33be0 00:22) to the end of the ledger, plus five later commits (e3ea977 14:07, a56ae10
  14:35, 7ca92ea 14:35, e48d76b 14:52, 7a40522 15:05), the run started at 14:35 (`runs/flat_b050`) and the MCTS
  evaluation in `runs/mcts_eval` (14:51-15:05). None of these has a ledger entry. Section 2.0 lists
  the 09-15..09-21 waves only as context.

## 1. Summary

### 1.1 Goal

CLAUDE.md section 0c (CLAUDE.md:109, added in eaa8436 09-24 00:50) and docs/planner-design.md
sections 1-2: the planner/executor split exists for MACRO exploration. On maps whose potential field
does not show the path, the 25 Hz policy cannot commit to one side long enough to find out that
something new is there; a planner that decides every 1 to a few seconds can commit and look ahead.
Requirements (planner-design.md section 2): generic for mazes and surf with the same constants
(CLAUDE.md 0b), no top-down view, no fixed context window, no predefined trajectory set in the long
run, velocity part of the state.

### 1.2 Headline results (one seed per arm, everywhere)

Labyrinths (walking maps, `maps_pool/labyrinth_*`):

| what | result | run | source |
|---|---|---|---|
| deterministic BFS planner + fan executor trained on lab100 (stage 1) | lab100 9/9 from 202M (15.52 s); lab200 (never trained) 9/9 from 202M (23.62 s) | plLAB100a | L:25285 [checked: runs/plLAB100a_launch.txt] |
| same targets, destination-only executor (flat control) | lab100 9/9; lab200 0/9 zero-shot | plLAB100ball | L:25624 |
| learned vocabulary planner `--plan-r-ok 0` over the frozen plLAB100a executor | 9/9 on lab100 AND on the unseen lab200 at every eval from +100M, shortest routes | plLRN100b, plLRN200b | L:25644, L:25934 [checked] |
| + strict judge, sharp-turn shapes, Euclidean progress 0.5 (the new default) | 9/9 on both from +25M; chosen plans 23% off the walkable graph (vocabulary base ~80%) | plLRN100h | L:26121 [checked] |
| user's mazes, learned planner over the frozen plHARDa executor | easy01 3/3 from +25M; medium01 3/3 from +150M; hard01 0/3 at all 17 evals | plLRNeasy / plLRNmed / plLRNhard | L:26208 |
| jump-point planner, no network, depth 12, over plHARDa | hard01: 2,865 of 2,865 training episodes from the start finished; greedy 3,3,2,3 of 3 | jH2e12 | L:26249 [checked] |

Edgeflow with a GEOMETRY-graph planner (2026-09-24: deterministic BFS on `--plan-graph ride` or
`tight`, a trained fan executor):

| map | best greedy from the map start | run | source |
|---|---|---|---|
| blue025 | 6/9 at 907M, from scratch (0/9 through 806M) | srR025f | L:26341 [checked] |
| blue050 | 8/9 at +450M warm from efCTLnp_blue050; 9/9 at 1.108B and 1.310B from scratch (unseen blue025 8/9) | srW050f; srR050f | L:26374, L:26508 [checked] |
| blue100 | 9/9 at +400M, warm from srW050f | srW100f | L:26410 [checked] |
| blue200 | 9/9 at +350M warm from srW100f; 9/9 at 202M on the tight graph with every plan to the finish; 9/9 at 303M on varied plans + unseen blue100 8/9 | srW200f; srFT200; rpCTL | L:26410, L:26634, L:26758 [checked] |

Caveats written in the same entries: the ride graph connects start and finish only on the edgeflow
maps, not on cannonball, celestial, unitfarmer2, petrus_lite or utopia (L:26474, CORRECTION at
L:26498); the `tight` graph's constants were set by looking at blue200 plots and the ledger labels it
"A DIAGNOSTIC, not a recipe candidate (CLAUDE.md 0b)" (L:26567-26569). srFT200, efTGT200 and the four
rp* arms ran on `tight` (L:26541, L:26640-26643, L:26728).

Edgeflow with the LEARNED primitive planner (2026-09-25: `--goal-planner primlearn`, no graph, an
egocentric input):

| map | best greedy from the map start | run, status | source |
|---|---|---|---|
| blue025 | 9/9 at 904.9M (13.09 s); 8/9 at 1.207B and 1.308B | rec_b025, THE RECIPE from step 1 | L:27212 (its table lists only 502M 0/9, 603M 0/9, 1.207B 8/9); the 9/9 is in runs/research/rec_b025/progress.csv and runs/research/gate_bench/rec_b025_launch_final.txt [checked] |
| blue050 | 9/9 at 1.207B (16.0 s) and 1.912B (12.99 s); 7/9 at 1.006B | rec_b050, THE RECIPE from step 1 | L:27275, L:27330 [checked] |
| blue100 | 4/9 at 2.262B (21.53 s); 2-4/9 from 2.161B on | ret_b100L, the recipe flags on a chain of checkpoints | L:27236 [checked] |
| blue200 | 0/9 at every eval | ret_b200sL (the recipe; continues ret_b200s2 @ 1.811B, the fresh-planner chain from step 1), ret_b200f (+ `--respawn-frac 0.7`, continues ret_b200w2 @ ~2.2B), ret_b200m (+ `--respawn-frac 0.7 --plan-mu-bound 1.5`, continues ret_b200sL @ 2.748B) | L:27253-27256, L:27270-27273, L:27304-27306, L:27347 [checked] |

Before the recipe existed, prim2f_b025 finished blue025 6/9 at 690M and 8/9 at 791M and 891M, with
the held-out blue050 at 1/9 at 690M and 791M (L:26991; runs/prim2f_b025/progress.csv [checked]).

### 1.3 The recipe and its one-recipe check

* THE RECIPE (L:27317-27322): `--goal-planner primlearn --plan-shaping refund --plan-cover 0.3
  --plan-return 1 --ep-secs 30 --int-coef 0`, with primlearn's defaults (`--exec-cut 1`,
  `--plan-uniform 0.5` and the rest of `PRIMLEARN_DEFAULTS`, python/surfgym/goalprimplan.py:86-90),
  warm-started from step 1's executor `runs/prim1_b025/ckpt_0501219328.pt` with a FRESH planner.
* Exact launches: runs/research/gate_bench/box_rec_b025.txt:1 (source checkpoint + flags) and
  runs/research/gate_bench/research_rec_b050_launcher.txt:12 (`python3 -u python/train_fast.py
  --ckpt runs/src_ckpt.pt --run rec_b050 --steps 2001219328 --record-every 100e6 --eval-eps 9
  --eval-greedy-only --ckpt-every 1e9 --map maps_pool/surf_edgeflow_blue050.bsp --goal-planner
  primlearn --plan-shaping refund --plan-cover 0.3 --plan-return 1 --ep-secs 30 --int-coef 0
  --ckpt-every 250e6`). Both logs print `resumed runs/src_ckpt.pt at step 501,219,328` and
  `planner: FRESH` (runs/research/rec_b025/rec_b025_launch.txt:16,22;
  runs/research/gate_bench/rec_b050_launch.txt:17,23).
* Carried from the checkpoint (run.json `config` of both runs): continuous absolute view in the
  velocity frame, keys-hold, `goal_obs fan` with `goal_fan_offsets` 0.25..2.0 s, `goal_reward arc`
  (launch log: corridor 384 u, "goal arc shaping scale 0.0666667/u (100 per 1,500u)"), `time_pen`
  0.005, `success_bonus` 50, `stall_secs` 30, `respawn_frac` 0.9, `respawn_margin` 1, blue200 held
  out, 2,048 envs, `n_steps` 128, seed 0.
* One-recipe check (CLAUDE.md 0b): the same flags from the same executor passed blue025 and blue050
  (L:27248-27251, L:27319-27320). Limits visible in the sources: rec_b025 started with the
  checkpoint's 20,000-state reservoir (same map) and rec_b050 with none (cross-map resume;
  rec_b025_launch.txt:15, rec_b050_launch.txt:15-16); step 1's executor trained on blue025 only;
  blue100 passed from a chain (p2_b100 -> ret_b100w -> ret_b100L, L:27321-27322); blue200 did not
  pass; no run of this recipe on unitfarmer2, petrus, cannonball or celestial appears in the ledger.
* The checkpoint rec_b025 kept (runs/research/rec_b025/rec_b025_keep.pt, md5 8492d2e4f23e...) matches
  the ledger's "1.302B checkpoint (md5 8492d2e4...)" (L:27224-27225) [checked].

### 1.4 State at ~15:05 (no ledger entry for any of this)

* `flat_b050` runs locally since 14:35: the recipe plus `--prim-flat 1` on blue050, resumed from
  prim1_b025 @ 501M (runs/flat_b050/run.json: `prim_flat 1`, `plan_shaping refund`, `plan_cover
  0.3`, `plan_return 1`, `ep_secs 30`, `int_coef 0`). runs/flat_b050_launch.txt shows 0/9 greedy at
  502M, 603M, 704M, 804M and 905M. That is the state at the time of writing, not a result.
* MCTS over primitives exists since e48d76b (14:52) as an EVAL-TIME search in tools/record_ckpt.py
  (`--plan-mcts N`), not in training (section 3.21). runs/mcts_eval/summary.txt (15:00) compares
  plain greedy / `--plan-search 8` / `--plan-mcts 16` with one checkpoint per map (listed in 3.21), 9
  recorded episodes each: blue025 9 / 9 / 9, blue050 8 / 9 / 9, blue100 1 / 7 / 7, blue200 0 / 0 / 0
  finishes. Work in progress: more variants were being recorded at 15:05, and 7a40522 (15:05)
  added MCTS knobs.

## 2. Timeline

### 2.0 Context before 2026-09-23 (summary only)

| ledger heading | what | result | source |
|---|---|---|---|
| 2026-09-15 23:55 | edgeflow dip ladder; dips 0 / 148 / 532 / 2,204 u (0 / 5.54 / 19.91 / 82.49 reward) by tools/dip_probe.py | benchmark built | L:22975 |
| 2026-09-16 06:15 .. 2026-09-20 00:55 | edgeflow waves 1-5 (ratchet, curiosity, death charge, full-episode rollouts, regularisation, ...) | only efRAT finished blue025 (7.58 s at 504M); nothing crossed blue050 | L:23025 .. L:23337 |
| 2026-09-20 01:25 | Go-Explore phase 1 (tools/explore_phase1.py), no policy | reached blue050's finish after 28.4 min of search | L:23390 |
| 2026-09-20 17:15 | labyrinth, geodesic vs Euclidean shaping | geodesic finishes every rung; Euclid stops at the first wall from rung 100 up | L:23647 |
| 2026-09-20 19:50 .. 2026-09-21 01:45 | Sibling Rivalry `--race-sr --sr-select` | passes lab100 under Euclid, nothing else; closed | L:23936, L:24149 |
| 2026-09-20 23:10 / 23:35 | the user's MACRO vs MICRO reframing; robotics hierarchy survey | direction | L:24049, L:24087 |
| 2026-09-21 02:00 .. 04:05 | certified potential fields (tools/certify_field.py, `--goal-field-file`) | null | L:24170 .. L:24345 |
| 2026-09-21 04:10 | shutdown; the agreed direction (expert iteration) and the user's open questions for a line planner | - | L:24378 |

### 2.1 2026-09-23 00:22 - 04:40: detour-benchmark arms without a planner

| heading [commit] | what | result | source |
|---|---|---|---|
| "2026-09-23 - the Discord thread" [4b33be0 00:22] | SEE, CRL, SGCRL, the SGCRL mechanism paper, CPPO read | SGCRL named the one new lead | L:24423 |
| 01:45 [5a943e2 01:35, 94a55ad 01:44] | reverse curriculum from the goal (Florensa 2017) | blue050's goal-rooted pool never leaves the finish platform | L:24464 |
| 02:25 [74a5df6 02:22] | CPPO `--crl` (772041f, 8bd69c7, merged f759324) and SGCRL (932b42d..35a35d0, merged bb3ff64) launched | 4 of blue050's 16 spawns dead on arrival | L:24587, L:24709 |
| 02:40 [8bd8b07 02:36] | reverse curriculum result | 0 finishes from the true start on lab100 / lab200 | L:24737 |
| 03:20 [1c4a977 02:53] | OpenAI Five App. O.2; GRU pair and ladder randomisation launched | - | L:24809 |
| 03:30 [1c4a977 02:53] | CPPO crlLAB200 | training episodes finished lab200 (86M-~300M), then lost; no greedy finish | L:24919 |
| 03:45 [a52efac 03:32] | xsG6n: cannonball goal fan with the modern action space | 119,188 u order-only max at 1.28B, 0 finishes; stopped at 1.32B | L:24969, L:25212 |
| 04:00 [4574289 03:55] | ladder randomisation labLADDERb; GRU; SGCRL; CPPO blue050 | lab100 9/9 from 303M, lab200 8/9 at 504M then intermittent; GRU, SGCRL, CPPO-EF050 null | L:25033 |
| 04:25 [3808e67 04:17] | direction: PLANNER + EXECUTOR, deterministic first | - | L:25157 |
| 04:40 [79a87d3 04:25], "User addition (04:50)" [76ac373 04:29] | the night's program; simplify goals not maps; the planner must discover executability | - | L:25212, L:25252 |

### 2.2 2026-09-23 05:20 - 22:30: the labyrinth planner stack

| heading [commit] | what | result | source |
|---|---|---|---|
| 05:20 [8bb92ab 05:19; code 53dcc37, merged 4c8ffb1] | stage 1: `--goal-planner bfs` + fan executor | plLAB100a 9/9 on lab100 and unseen lab200 from 202M | L:25285 |
| 05:50 [187957f 05:47; code a3e8f17 + f40e87b merged 88b6d54, c435642] | stage 3: `--goal-planner learned`, `--freeze-policy` | built | L:25324 |
| 06:15 [c4832a5 05:59], 06:30, 07:10 [2785a35 07:08] | plLRN100a / plLRN200a (+0.7 / -0.3 reward) | lab100 up to 9/9 with routes ~3x the graph path, evals swing 0 <-> 9/9; lab200 up to 7/9 | L:25387, L:25428, L:25480 |
| 07:19 [f41b648; code 5aa62ca merged d2517d6] | surf stack: `--plan-vocab surf`, `--goal-planner vocab` | built | L:25518 |
| 07:30 [82ff5b1 07:29] | flat control plLAB100ball | lab200 0/9 zero-shot | L:25624 |
| 08:05 [9e1580d 08:02; code a10c90d 07:11] | `--plan-r-ok 0` | plLRN100b 9/9 on both maps at every eval | L:25644 |
| 08:30 [1709d25 08:14] | surf stage b interim, psEF050v | own segments completed 67-79%, fixed shapes 0% | L:25701 |
| 09:10 [a9b004b; code 6b83f06 merged 97d609f] | `--plan-vocab proposals` | built, not launched | L:25742 |
| 09:20 [6ac83a2 09:17] | surf status | psEF050v never left the platform; psEF050L null | L:25877 |
| 09:50 [23acff6] | plLRN200b | 9/9 on both maps at every eval | L:25934 |
| 14:04-19:03 [4af766a, dcf3d0b, d0645ea, fcd1a4c, 38f04eb] | `record_ckpt --dump-plans`, `--bfs-replan-len`, viz_planner*.py | tools | commits |
| "(afternoon)" [7e89023 15:00, e5019c1 15:51; code 6d4d0cb 15:00, 3109d2c 15:50] | `--plan-corridor`, `--plan-lturn`, `--plan-strict` | judge loophole found and closed | L:25972 |
| 18:10 [bbf3b62] | wave 8-10 result; new learned-planner default | plLRN100h 9/9 on both from +25M | L:26121 |
| 20:45 [77d7ce7 20:40] | the user's mazes, zero-shot | the lab100 pipeline finds none | L:26168 |
| 22:30 [b275fb0 22:27] | the mazes trained | easy +25M, medium +100M, hard never in 400M | L:26208 |

### 2.3 2026-09-24 00:50 - 12:36: design doc, jump planner, surf graph

| heading [commit] | what | result | source |
|---|---|---|---|
| [eaa8436 00:50, c63d52c 02:54, 623d49b 03:11, a0437bc 04:04, 0b6294a 11:28] | docs/planner-design.md sections 1-8, CLAUDE.md 0c | design (MCTS over jump points proposed, section 7) | docs/planner-design.md |
| 07:20 [a83e170 07:04; code 969710b, 3001742, a5e7f2a] | `--goal-planner jump`, `--jump-u episodic`, FIN_BONUS fix | hard01 solved at depth 12 at step 0 | L:26249 |
| 07:20 [a83e170 07:04; code 3d05515, 4c8c17e] | `--plan-graph ride`, `--goal-obs fanline` | ride connects all four edgeflow maps | L:26315 |
| 08:40 [ff06584 08:39] | stage 1 on blue025 | srR025f 6/9 at 907M; srR025L (fanline) 6/9 at 907M, 10.2 s vs 14.1 s | L:26341 |
| 09:25 [dfaffcb] | srW050f | first ever blue050 finishes, 8/9 | L:26374 |
| 10:50 [f15d896 10:48] | srW100f, srW200f; jump planner over frozen executors | 9/9 on both | L:26410 |
| 11:30 [19b0b2e 11:25; code 14221b5 11:08] | jump planner, 1,500 u jumps | jS100L up to 7/9, jS200L up to 8/9 | L:26445 |
| 11:40 [eb69019 11:27], CORRECTION 11:50 [09e562d 11:28] | ride graph on the benchmark maps | only edgeflow is connected | L:26474, L:26498 |
| 12:36 [5147a8d] | srR050f from scratch | 9/9 at 1.108B; unseen blue025 8/9 | L:26508 |

### 2.4 2026-09-24 18:48 - 23:15: blue200, spawn fix, throughput, representation study

| heading [commit] | what | result | source |
|---|---|---|---|
| 19:35 [corrected to 18:58, 84b2fc9; code 2177f19, 40fc135] | blue200: route as a reward vs through the fan; `--plan-graph tight` | efPLN200 0/9 (old route trap); efTGT200 6/9 at 202M; srF200f 6/9 at 202M; srR200f 0/9 through 806M | L:26528 |
| CORRECTION "20:00" [19:11], FIX "20:40" [19:26, code 0e6b320] | embedded spawns lifted clear | blue200 map_pct ceiling ~77% explained | L:26605-26632 |
| 19:45 [9ae1122 19:41; code 6ffa469] | tight plan through the fan vs as a reward; `--timing` | srFT200 9/9 at 202M vs efTGT200 6/9 | L:26634 |
| CORRECTION "20:35" [62688a0 20:12; code 079dee2] | compiled Douglas-Peucker | 255k -> 475k steps/s | L:26698 |
| [f158133 20:55], [1ebef43 21:49] | plan-representation survey; `--plan-film` | - | docs/plan-representation-survey.md |
| 22:20 [267fe9e] / 23:15 [500a19a 23:11] | plan-representation study on 4 x RTX 4090 | 8-point fan = 30-point fan; FiLM harmful | L:26722, L:26758 |

### 2.5 2026-09-25 03:24 - 05:32: motion primitives

| heading [commit] | what | result | source |
|---|---|---|---|
| [06c9461 03:24, d25ffd3 04:12] | tools/viz_primitives.py, tools/viz_curve_sliders.py | visualisation | commits |
| "2026-09-25 (machine clock)" [4c2cf6d 04:21, 104008c 04:33] | world-record primitive coverage, ANALYSIS ONLY | ~6 numbers per 2-s route window | L:26808 |
| 05:32 [19d52c8; code 79b303c 04:44, af0d772 04:47, 27efd45 05:30] | step 1 `--goal-planner prim`; step 2 `--goal-planner primlearn` launched | prim1_b025 results; prim2_b025 launched | L:26869 |

### 2.6 2026-09-25 05:36 - 11:34: primlearn rewards and the recipe (details in 3.12)

| heading [commit] | what | result | source |
|---|---|---|---|
| 05:45 [corrected 05:37] | completion farm | fix f3d3acb | L:26925 |
| 06:25 [corrected 06:20] | reward separation, four failures (a18c98f, 1b96bb6, 18c4d13, 8a33b2b) | prim2f first training finishes | L:26947 |
| 06:30 [corrected 06:24] | blue025 passed | prim2f 6/9 at 690M; the executor does not follow literally | L:26991 |
| [eef6365 06:32] | search `record_ckpt --plan-search M` | - | commit |
| 07:05 [corrected 06:45; code 271fc31, 069a53c] | bigger maps 0/9; `--plan-obey` inconclusive; `--plan-cover` | - | L:27019 |
| 07:55 [corrected 07:10; code 61fc4ba, 9610526] | coverage farm; count-weighted fix; refund bias; exact shaping | recipe v3 | L:27053 |
| 07:12 CORRECTION [f2d24fc 07:11] | heading-time correction table | - | L:27087 |
| 07:22 [28665ef; code e541126] | exact shaping removed the forward pull; `--plan-shaping refund` | - | L:27103 |
| 07:30 [6aab0e2; code 28304d2 07:23] | exact shaping wanders; `--plan-return` everywhere | - | L:27126 |
| 07:40 [2630079] | return finds the finish in training on all four maps | greedy still 0/9 | L:27147 |
| 08:01 [ca44b26; code 82c955a 07:44] | blue050 passed; search copies the held keys | - | L:27168 |
| 08:28, 08:53, 09:18, 09:19, 09:42 | blue050 5/9; recipe passes blue025 and blue050; blue100 4/9; blue200 start | see 1.2 | L:27192 .. L:27275 |
| 10:06 [d37b629; code 2358501 10:05] | `--plan-mu-bound` | ret_b200m 0/9 | L:27294 |
| 10:27, 11:13, 11:34 | end-of-night state; rec_b050 9/9; fleet empty | blue200 0/9 | L:27308 .. L:27347 |

### 2.7 After the last ledger entry (no ledger text)

| commit | what |
|---|---|
| e3ea977 14:07 | every primitive written into the recording's trailer; the viewer shows the ACTIVE primitive (3.17) |
| a56ae10 14:35 | `exec/track_strict` / `exec/track_lenient`; `record_ckpt --plan-override`; `--prim-flat` (3.16, 3.18, 3.19) |
| 7ca92ea 14:35 | viewer/runs.js: an unescaped quote in the exec/track description broke the file |
| (no commit) 14:35 | `flat_b050` launched (1.4) |
| e48d76b 14:52 | `goalsearch.PrimMCTS`, `record_ckpt.py --plan-mcts N` (+ `-k`, `-depth`, `-c`); `--plan-search` now clones the real executor wrapper and observation (3.15, 3.21) |
| (no commit) 14:51-15:05 | runs/mcts_eval: greedy vs search vs MCTS recordings on the four maps (3.21) |
| 7a40522 15:05 | goalsearch.py / record_ckpt.py: `--plan-mcts-time`, `--plan-mcts-gamma`, `--plan-mcts-uniform`, `goalsearch.unsquash` (3.21) |

## 3. Components

### 3.1 Detour-benchmark arms without a planner (2026-09-23 night)

* Reverse curriculum (5a943e2): `tools/explore_phase1.py --roots-goal`, tools/goal_curriculum_pool.py,
  `--demo-front-frac`; tests/python/test_goal_curriculum.py (15 tests). Result: rcLAB100 / rcLAB100e /
  rcLAB200 0 finishes from the start; with Euclidean shaping the frontier stopped on the band where
  the Euclidean distance rises along the path (L:24737-24783); the blue050 pool never left the finish
  platform, so rcEF050 was not run (L:24504-24515, L:24781-24783). Runs: runs/research/rcLAB100*.
* CPPO `--crl` (python/surfgym/crl.py; flags python/train_fast.py:5672-5715; tests/python/test_crl.py,
  16 tests): A = Q(s,a,g*) - V(s,g*), contrastive critic, no reward in the gradient. crlLAB100 never
  left the start room (L:24791); crlLAB200 finished in TRAINING between 86M and ~300M and lost it
  (L:24919); crlEF050 null (L:25108). Runs: runs/crlLAB100, runs/crlLAB200, runs/crlEF050.
* SGCRL (python/train_sgcrl.py; `--yaw world`, `--alpha 0`, `--drop-dead-spawns` default 1;
  tests/python/test_sgcrl.py, 23 test functions): 0 goal hits in ~1.4M training episodes over lab100,
  lab200 and blue050 (L:25096-25127).
* Ladder randomisation labLADDERb (OpenAI Five App. O.2): lab100 9/9 from 303M, lab200 at two evals
  (L:25039-25083). The user then ruled out map simplification: "simplify not the map itself, but
  rather the goals" (L:25235-25239). GRU `--rnn gru`: null (L:25085-25094).

### 3.2 BFS planner and the stage-1 executor

* Code: python/surfgym/goalplan.py (module docstring :1-56; `BFSPlanner` :647; `make_plan_hooks`
  :952), GoalSystem kind 3 "planned" (python/surfgym/goalsys.py `_planned_goal` :608). Commit
  53dcc37, merged 4c8ffb1.
* Walk graph: a node is a free cell with solid within 64 u below (`PLAN_SUPPORT_U`) above the kill
  ceiling; 26-neighbour edges without corner cutting; 256 random targets + the finish box, one
  Dijkstra field each; a plan = steepest descent of the target's field, lifted to the start's height,
  Douglas-Peucker at one cell, resampled at 128 u (goalplan.py docstring).
* Flags (python/train_fast.py:4623-4652, :4821-4842): `--goal-planner bfs`, `--goal-plan-targets`
  256, `--goal-plan-finish` 0.2, `--goal-plan-dmin` 256, `--goal-plan-dmax` 4096, `--goal-reward
  plan|arc`, `--goal-fan-offsets`. Recorder: `--plan-target finish|random` (tools/record_ckpt.py:627),
  `--bfs-replan-len` (:605, fcd1a4c).
* Tests: tests/python/test_goal_planner.py (15 test functions; commit 53dcc37 says 18 tests).
* Measured: plLAB100a (L:25285), the flat control plLAB100ball (L:25624), plHARDa on hard01 7/9
  at 403M with the unseen easy01 8/9 and medium01 9/9 (L:26212-26217) [checked].

### 3.3 The learned vocabulary planner (goallearn.py) and the maze reward farm

* Code: python/surfgym/goallearn.py (docstring :1-95; constants :103-175). Commits a3e8f17, f40e87b
  (merged 88b6d54), a10c90d, 6d4d0cb, 3109d2c.
* Design: a categorical over 80 shapes (16 world headings x 5 turns 0 / +-45 / +-90 deg, 8 x 100 u =
  800 u), anchored at the agent, nothing filters a shape; input = a world-aligned 32 x 32 patch of
  64 u cells (walkable fraction + this episode's visits, capped at 4) + 9 scalars (Euclidean finish
  direction and log distance, velocity, yaw); a plan closes at arc >= 0.9 of its length (corridor =
  goal radius 192 u), at its budget (length / 250 u/s x 1.5) or at episode end; PPO gamma 0.95 and
  lambda 0.95 per plan, clip 0.2, entropy 0.01, 512 plans per update, 4 epochs (goallearn.py:137-175);
  `--freeze-policy 1` keeps the executor and its Adam state bit-identical.
* Reward per plan (`PLAN_DEFAULTS`, goallearn.py:168-175): +0.7 completed / -0.3 not (AMIGo), +10
  on the finish, novelty 0.5/sqrt(n) over global 128 u cells at the plan's end (none on a death),
  optional Euclidean progress (default 0).
* THE MAZE REWARD FARM (L:25644-25651): +0.7 per completed plan with gamma 0.95 per plan makes an
  endless stream of completed plans worth 0.7 / 0.05 = 14, more than the +10 finish. plLRN100a /
  plLRN200a wandered (routes ~3x the shortest path) and their greedy evals swung between 0/9 and 9/9
  (L:25428-25513). Fix a10c90d: `--plan-r-ok 0` (defaults stay 0.7 / -0.3). plLRN100b / plLRN200b:
  9/9 at every eval on shortest routes (L:25644, L:25934) [checked].
* THE JUDGE LOOPHOLE (L:26049-26077): the arc tracker's +-16-vertex window spans a whole 7-vertex
  plan, so "90% of the arc" meant arriving near the plan's end by any route. `--plan-strict`
  (3109d2c): window 2, fail after 0.5 s outside the corridor or after 1 s with < 32 u of progress
  (goallearn.py:155-160). `--plan-lturn` adds 16 x 12 sharp-turn shapes (272 in all); `--plan-corridor
  U` sets the judge's tolerance; `--plan-progress` pays Euclidean progress per 1,000 u.
* New default after waves 8-10 (L:26164-26166): `--plan-strict --plan-corridor 64 --plan-r-fail -1.0
  --plan-lturn --plan-progress 0.5 --plan-r-ok 0`, 200M. Wall crossing (share of the chosen plans'
  length off the walkable graph, measured, never rewarded) fell 55% -> 22-23% (L:26129-26136).
* Tests: tests/python/test_goal_learned.py (14 test functions; the ledger says 16/16, L:25366).

### 3.4 Surf plan diet and trajectory proposals (built 2026-09-23, not in the final stack)

* `--goal-planner vocab --plan-vocab surf` (python/surfgym/goalsurf.py, 5aa62ca; constants
  goalsurf.py:98-119): 144 shapes (16 headings x 3 turns x 3 constant descents), length clamp(3 s x
  max(|v_xy|, 500 u/s), 800, 6000) u; the executor trains on uniform shapes plus hindsight segments of
  its own flights (`--plan-hindsight` 0.5). tests/python/test_goal_surf.py (14 test functions).
  psEF050v completed its own segments (up to 79%) and 0% of the fixed shapes; its saved reservoir
  lies entirely on the start platform; psEF050L (learned planner over it) null (L:25701-25740,
  L:25847-25895).
* `--plan-vocab proposals` (python/surfgym/goalprop.py, 6b83f06 merged 97d609f; constants
  goalprop.py:110-141): K = 32 candidates per decision (hindsight, perturbations earlier / later /
  higher / flatter / left / right, uninformed shapes), eps-NMS at 64 u, a pointer head.
  tests/python/test_goal_proposals.py (21 test functions; the commit says 23/23). Never launched
  (L:25897-25900).

### 3.5 The jump-point planner (goaljump.py)

* Code: python/surfgym/goaljump.py (docstring :1-23; constants :38-55). Commits 969710b, 3001742,
  a5e7f2a, 14221b5. Flags `--jump-depth` (3), `--jump-u euclid|novelty|episodic|euclid+episodic`
  (euclid), `--jump-t` (0.05), `--jump-len` (750) (python/train_fast.py:4667-4684).
* Options = the distinct places one jump away on the graph (8 compass probes, dropped under 0.3 of a
  jump, ends within 200 u merged, a reachable finish always an option); a depth-limited search
  values each option by the best node of its subtree (U minus 0.1 per 1,000 u walked; a reachable
  finish is `FIN_BONUS` 10,000 minus 10 per 1,000 u); training draws softmax(value / T), the eval
  takes the argmax. No network: results are decided at step 0 (L:26249-26265).
* FIN_BONUS bug (L:26308-26313, a5e7f2a): the first value of 100 went negative on routes > 10,000 u.
  The rerun jH3e16 (depth 16) and jH3e12 / jH3x12 show greedy 3/3, 3/3, 2/3, 3/3 in
  runs/jH3*_launch.txt [checked]; the ledger does not report them.
* Tests: none found (no file under tests/ references goaljump or `--jump-depth`).
* Edgeflow over frozen executors: jS050W 9,8,8,7 of 9; jS200L 6,5,8,6 (L:26445-26458) [checked].

### 3.6 Graphs: walk / ride / tight

* `--plan-graph walk|ride|tight` (python/train_fast.py:4656-4664; constants goalplan.py:64-96;
  `ride_mask` goalplan.py:193; `BFSPlanner` graph kinds goalplan.py:672-770).
* ride (3d05515): free cells above the kill ceiling with solid within 256 u below (`RIDE_SUPPORT_U`)
  or one 128 u hop (`RIDE_HOP_U`) from such a cell (tools/dip_probe.py's route model). Start->finish
  2,732 / 3,058 / 4,871 / 8,583 u on blue025/050/100/200 (L:26317-26322).
* tight (2177f19, 40fc135): ride's cells and edges; a step costs length x the mean height above the
  surface below (cells) + `TIGHT_EDGE_W` 3 per cell closer than `TIGHT_EDGE_U` 128 u to the rim; the
  random-target band reads geometric length. A DIAGNOSTIC by the ledger (L:26567-26569).
  tests/python/test_goal_planner_tight.py (2 tests).
* Limit (L:26474-26506): no start->finish connection on cannonball, celestial, unitfarmer2,
  petrus_lite, utopia ("momentum is the next step").
* Tools (2177f19): tools/plan_route.py (a map's start->finish plan as a `--race-arc` route, geometry
  only), tools/plan_progress.py (greedy evals scored along it), tools/viz_run_vs_plan.py. Route files:
  runs/research/planroute/surf_edgeflow_blue200.plan_{ride,tight}.npz (L:26575).

### 3.7 The fan and the plan-representation study

* The fan (`goals.MultiLine`, python/surfgym/goals.py:141-): per env the nearest plan point plus one
  point per horizon at arc offset speed x horizon (speed floored at 500 u/s), each as (forward, left,
  up) in the yaw frame: 3 x (1 + horizons) scalars. Default horizons 0.25, 0.5, 1, 1.5, 2, 3, 4.5,
  6 s (python/surfgym/route.py:66); the primlearn runs use 0.25..2.0 s (27 scalars; run.json
  `goal_fan_offsets`, launch log "8 horizons 0.25-2s -> 27 features").
* `--goal-obs fanline` (4c8c17e; `PlanLineLidar`, python/surfgym/goalball.py:202): the next 12 plan
  vertices as 1.5 px dots in an extra depth channel. srR025L vs srR025f: same first-finish step;
  10.2 s vs 14.1 s; training goals 84.1% vs 64.5%; 150-236k vs 256-306k steps/s (L:26347-26366).
* `--plan-film` (1ebef43; python/train_fast.py:4398, :711-726; tests/python/test_plan_film.py, 4
  tests): an MLP over the fan emits FiLM (gamma, beta) for the last conv block, zero-initialised.
* Survey: docs/plan-representation-survey.md (f158133; its "Result" section came with 500a19a).
* Study (L:26722-26806; runs/research/rp*; logs runs/research/gate_bench/rp*_launch.txt) [checked]:
  from scratch on blue200, varied plans (80% random targets, 20% finish), tight graph, blue100 held
  out, 1B each, on RTX 4090s (CLAUDE.md asks for 3090s; L:26739-26744).

| arm | fan | fusion | blue200 / unseen blue100 greedy |
|---|---|---|---|
| rpCTL | 8 points, 0.25-2 s | concat | 9/9 and 8/9 at 303M, 9/9 and 9/9 from 404M |
| rpDENSE | 30 points, 0.2-6 s | concat | 7/9 and 9/9 at 303M, 9/9 and 9/9 from 404M |
| rpFILM | 8 points | + FiLM | 0/9 at every eval to 907M |
| rpBOTH | 30 points | + FiLM | 0/9 until 605M; 4/9 and 6/9 at 706M; 9/9 and 9/9 at 806M |

  Verdict (L:26802-26806): the plain concatenated fan stays; a denser fan is neutral; FiLM hurts.

### 3.8 The spawn fix

* `rewards.map_spawn_pool` (python/surfgym/rewards.py:2273-2317; `SPAWN_UNSTICK_U` = 64 u at :2270),
  commit 0e6b320: a spawn whose standing hull starts inside solid is moved to the nearest clear height
  within 64 u in 1 u steps, up before down, and dropped only when none is clear.
  `SURFGYM_LEGACY_SPAWNS=1` returns the raw entities for old bit-identity tests (6ffa469).
* Why: the core fails a player trapped in solid for 5 ticks; blue200's 4 front-row spawns (y =
  -1,248) were embedded, so 2 of 9 eval episodes died at once and `race/map_pct` topped out at ~77%
  (CORRECTION L:26605-26612, which also corrects the "greedy policy stands still" reading of
  2026-09-16).
* Fixed maps (L:26620-26627): all four edgeflow maps (4 of 16 spawns, up 4 u), surf_ph_restyle,
  surf_hopee_v2, surf_pyk_yougi, surf_src_sidistic (1 of 2 dropped). tests/python/test_spawn_pool_solid.py
  (3 tests: blue200's 4 spawns move exactly 4 u up; sidistic drops 1; cannonball unchanged).
* Earlier the SGCRL trainer got its own rule, `--drop-dead-spawns` (35a35d0; L:24709-24735).

### 3.9 The throughput fix

* 6ffa469: the reservoir no longer computes reached-state goal segments under `--goal-planner bfs /
  jump` (their O(snapshots^2) Python loop).
* 079dee2: `goals._rdp_fast` (python/surfgym/goals.py:439-514), the numba form of the Douglas-Peucker
  in `_rdp`, used only by `BFSPlanner.line_from` (goalplan.py:905-912, `fast=True`); one nearest-node
  query per reset. tests/python/test_rdp_fast.py (1 test, random, lattice and degenerate polylines).
* Measured on blue200, srFT200's early config (L:26698-26720): 255k steps/s before any fix -> 293k
  with the segments off -> 475k with the compiled simplification; the flat arm runs 604k. The ledger
  first blamed the segments (L:26670-26687, 1,130 of 1,196 ms/iter) and corrected itself: 69% of
  `goalsys.assign` was `goals._rdp` (L:26698-26707).

### 3.10 Motion primitives (step 1) and the world-record coverage analysis

* ANALYSIS ONLY (CLAUDE.md section 0): tools/primitive_coverage.py (4c2cf6d, 104008c) fits turn-rate
  families to world-record flights (cannonball, unitfarmer2, petrus) after removing the air-strafe
  weave (sigma 0.25 s). A 2-s window needs ~6 numbers (quadratic sideways and vertical rates):
  85-88% of route windows within 128 u, 98-99% within 256 u; 8 numbers 97-100% (L:26826-26845).
  On our own edgeflow finishers 6 numbers redraw 100% of 2-s windows within 64 u (L:26849-26867).
  Reports: runs/research/primitives/, primitives_smooth/, primitives_edgeflow/.
* Step 1, `--goal-planner prim` (python/surfgym/goalprim.py; 79b303c, af0d772): a primitive leaves
  along the velocity (the view below 50 u/s) and is fixed by the sideways and vertical turn rates at
  K = 3 knots joined by the polynomial through them, traced at max(speed, 300 u/s) for 2 s
  (`PRIM_DEFAULTS` goalprim.py:23-24; `curve` :41-69). Each spawn draws the 6 numbers uniformly
  (sideways +-180 deg/s, vertical -120..+90 deg/s); success = entering the 192 u sphere at the end; a
  draw whose end sphere can be entered early is redrawn (af0d772, goalprim.py:126-141). Flags
  `--prim-secs`, `--prim-knots`, `--prim-side`, `--prim-down`, `--prim-up`, `--prim-floor`
  (python/train_fast.py:4605-4622). tests/python/test_goal_prim.py (7 tests).
* prim1_b025 (blue025 from scratch, blue200 held out, 4-s episodes): training success ~41% at ~440M;
  climbing primitives 0-7.5% flyable (L:26877-26899). Prim-evals (seeded random primitives) 1/9, 4/9,
  6/9, 7/9, 3/9 at 1M-404M, held-out blue200 2, 5, 5, 7, 3 [checked: runs/prim1_b025_launch.txt, which
  also has 4/9 and 4/9 at 504M].
* tools/viz_primitives.py (06c9461) draws a different, earlier family (delay + smoothstep ramp-in
  over a fixed grid, viz_primitives.py:1-26); tools/viz_curve_sliders.py (d25ffd3) the 4-number
  family. Neither is the family goalprim.py trains.

### 3.11 primlearn: the learned primitive planner (step 2)

* Code: python/surfgym/goalprimplan.py (27efd45 and every later primlearn commit); trainer wiring
  python/train_fast.py:7365-7456 (flag resolution), :12515-12547 (construction), goalsys.replan
  (python/surfgym/goalsys.py:864-891), `_on_step_learned` (:939-). Refused under DDP
  (train_fast.py:12519-12521). tests/python/test_goal_primlearn.py (9 tests at 7ca92ea, 10 at e48d76b).
* Network `PrimPlannerNet` (goalprimplan.py:211-256): 2 x 256 tanh MLP; a 3-component Gaussian
  mixture over the 6 pre-squash numbers (log std clamped to [-3, 0.5]), tanh-squashed into goalprim's
  ranges (`squash` :288-297); components start apart on the sideways knots; a value head.
* Input (`observe`, :177-208): 24 point traces from the agent (8 azimuths x elevations -30 / 0 / +20
  deg around the direction of motion, 4,000 u, log-scaled, through the simulator's own trace) + 8
  scalars: the finish in the motion frame (3), log distance, speed, horizontal and vertical speed, and
  the progress banked in the episode (the 8th scalar). No map, no top-down view.
* Lifetime (`on_tick`, :464-635): a primitive closes when arc >= 0.9 of its length inside the 192 u
  corridor (`MultiArcProgress`, window 16, :335-336), after 1.5 x its duration (300 ticks at 10 ms),
  or when the episode ends; the next one is chosen at the next executor decision (`plan`, :648-692).
  `--plan-uniform` 0.5: that share of EPISODES opens with step 1's uniform draw instead of a planner
  choice (:658); those are not planner transitions and are logged as `exec/complete_unif`.
* PPO (`update`, :698-758): once 2,048 closed primitives are in (`plan_batch`), 4 epochs, minibatch
  max(256, M/8), clip 0.2, value coef 0.5, entropy 0.01, grad clip 0.5, Adam 3e-4; advantages from
  `goallearn.plan_gae` with gamma 0.95 and lambda 0.95 PER PRIMITIVE regardless of its duration
  (goallearn.py:679-696).
* `PRIMLEARN_DEFAULTS` (:86-90): plan_lr 3e-4, plan_ent 0.01, plan_batch 2048, plan_epochs 4,
  plan_novelty 0.5, plan_progress 1.0, plan_finish_bonus 10, plan_r_ok 0, plan_r_fail 0,
  plan_uniform 0.5, plan_obey 0, plan_cover 0, plan_shaping "pbrs", plan_mu_bound 0.
* The executor keeps training on the chosen primitives: arc progress along the current primitive
  (corridor 384 u) + the race reward's time penalty and +50 finish bonus (L:26966-26967;
  python/surfgym/rewards.py:1550 `r[goal] += self.success_bonus` with the finish mask from
  goalsys `_on_step_learned`).
* Greedy eval (`make_primlearn_hooks`, :869-1016), shared by the trainer and tools/record_ckpt.py
  (:1320-1357): the heaviest component's mean from the map start to the finish box.

### 3.12 primlearn reward variants, in order (blue025 unless stated; warm from prim1_b025 @ 501M)

| # | run | code | what changed | what broke / happened | source |
|---|---|---|---|---|---|
| 1 | prim2_b025 | 27efd45 | planner: progress + 0.5 completed / -0.5 not + 10 finish + end novelty; executor return crosses primitives (20-s episodes) | COMPLETION FARM: entropy 5.19 -> 0.40, planner primitives completed 26% -> 83%, time-outs 2% -> 91%, 0 finishes; stopped at 545M | L:26925-26939; run.json plan_r_ok 0.5, plan_r_fail -0.5 |
| 2 | prim2b_b025 | f3d3acb | r_ok 0; -0.5 per failure; a dying primitive's progress clipped at 0 | SUICIDE DIVE: the greedy planner's 2nd primitive was a steep dive off platform 2 (a death ends the -0.5 stream) | L:26959; a18c98f message |
| 3 | prim2c_b025 | old code (no run directory found) | task-only planner, executor return still crossing primitives | EXECUTOR FARM: 100% of episodes ran out the 20 s, executor reward ~250 per episode, greedy covered 8.9% | L:26960 |
| 4 | prim2d_b025 | a18c98f, 1b96bb6 | `--exec-cut 1`, `--int-coef 0`; planner paid for the task only; a death pays no progress | the planner aims at the finish (+120-160 u planned per primitive) and 40% of its primitives die: over the void is the straight line | L:26961; 18c4d13 message |
| 5 | prim2e_b025 | 18c4d13 | a DEATH charges back the progress banked in the episode | crosses the gap, falls at the ramp entrance (6/6 greedy); a time-out keeps its bank ("reach platform 2 and wait") | L:26962; 8a33b2b message |
| 6 | prim2f_b025 (from prim2e @ 588M) | 8a33b2b | a TIME-OUT charges the bank back too (the rule later named `refund`) | first training finishes, 0.7% of episodes by 623M; greedy 6/9 at 690M | L:26963, L:26991 [checked] |
| 7 | p2_b050, p2_b100, p2_b200b (boxes, 06:18) | as 6, `--ep-secs 30` | the blue025 recipe on the bigger maps | 0/9 at every eval; blue050's greedy agent turns left onto the bottom row, then bounces there for up to 25 s | L:27030-27037, L:27063-27071 [checked: runs/research/gate_bench/p2_*_launch_final.txt] |
| 8 | prim3_b025 | 271fc31 | `--plan-obey 1` | stopped at 600M to free the GPU; inconclusive | L:27043-27046 |
| 9 | cov_b050 | 069a53c | `--plan-cover 0.1`, episodic only | COVERAGE FARM: the greedy agent walked the start platform for 30 s, 5.8-7.8% of the route | L:27055-27061 [checked: map_pct 5.84 / 7.79 / 7.01] |
| 10 | cov2_b050 | 61fc4ba | count-weighted C / sqrt(1 + N), C 0.3, old refund | ran ~15 min, replaced | L:27084-27085 |
| 11 | v3_b050 (from step 1); v3_b050w / v3_b100w / v3_b200w (continuing p2) | 9610526 | EXACT potential-based shaping (Phi = bank, alive gamma Phi' - Phi, ANY end -Phi) + cover 0.3 | WANDER: v3_b050 at 603M covered 3.3% of the route; the box runs' planned progress fell to +1 / -32 / -65 u per primitive | L:27103-27115, L:27126-27135 [checked: v3_b050 map_pct 3.3] |
| 12 | ref_b050w (from p2_b050 @ 1.019B) | e541126 | `--plan-shaping refund` + cover 0.3 | keeps going forward: 34% of the route at 1.121B, 0/9 | L:27117-27132 [checked: 33.6%] |
| 13 | ret_b050w, ret_b100w, ret_b200w, ret_b200s | 28304d2 | + `--plan-return 1` | training finishes on all four maps within ~10 min (10.5-29.8% of episodes); greedy 0/9 at first; ret_b050w then 2/9 at 1.537B .. 5/9 at 2.041B | L:27147-27166, L:27192-27197 [checked] |
| 14 | rec_b025, rec_b050 | same flags, from step 1 | THE RECIPE | see 1.2 | L:27212, L:27236, L:27330 [checked] |
| 15 | ret_b200f | - | recipe + `--respawn-frac 0.7` (30% map-start spawns), a labelled VARIANT | 0/9 to 3.648B | L:27258-27274, L:27347 [checked] |
| 16 | ret_b200m | 2358501 | recipe + `--respawn-frac 0.7 --plan-mu-bound 1.5` | executor completion 7.0% (from ~1%), training finishes 18% -> 0% then back to 12.1% at 3.957B; 0/9 to 4.159B | L:27324-27327, L:27341-27352 [checked] |

The survey written during this sequence (docs/hrl-reward-survey.md, 271fc31) summarises it as "four
of the five reward failures on 2026-09-25 were one end priced differently from the others"
(hrl-reward-survey.md section 0, point 7) and pitfalls 1-3 (section 8).

### 3.13 `--exec-cut`: the executor's return ends with each primitive

* python/train_fast.py:4780-4786 (flag), :7526-7533 (default 1 under primlearn, 0 otherwise; refused
  without a plan-driven fleet), :11920-11922 (`b_cut`), :14641-14649 (set for the envs `goalsys.replan()`
  just re-planned), :14693-14697 (GAE: `nonterm = 1 - max(done, cut)`). Commit 1b96bb6.
* Why: without it "chaining primitives paid ~40 each without bound ... and finishing ENDED the paid
  stream" (1b96bb6 message; prim2c in 3.12). docs/hrl-reward-survey.md section 0 point 3 calls it the
  h-DQN / HAC / HiTS form. No dedicated test; test_trainer_and_recorder_run_primlearn asserts
  `exec_cut == 1` in run.json (tests/python/test_goal_primlearn.py:159).

### 3.14 Coverage, return, obedience, shaping, mean bound

| flag (default) | commit | code | rule | test | measured |
|---|---|---|---|---|---|
| `--plan-cover C` (0) | 069a53c, 61fc4ba | goalprimplan.py:371-390, 488-500, 577-583 | per env, the 128 u cells visited ALIVE this episode; a planner primitive that ends alive earns C / sqrt(1 + N) per cell it added, N = episodes that covered the cell before; bitmap budget `COVER_MAX_BITS` 4e8 | test_episodic_coverage_pays_new_ground_to_survivors, test_coverage_is_count_weighted_across_episodes | episodic-only version farmed (3.12 #9) |
| `--plan-return 1` (0) | 28304d2 | goalprimplan.py:760-768; respawn.py:210-213, 735-739; train_fast.py:12535-12539 | reservoir states drawn by 1 / sqrt(1 + N) of their 128 u cell's coverage count; refused without `--plan-cover` > 0 (train_fast.py:7452-7454) | test_return_weights_steer_the_reservoir_draw | 3.12 #13 |
| `--plan-obey 1` (0) | 271fc31 | goalprimplan.py:532-541 | credit = min(p, f x p), f = min(1, arc covered / 0.9); backward progress in full; `plan/credit_frac` | test_obedience_gate_credits_what_was_flown | 3.12 #8, inconclusive |
| `--plan-shaping pbrs` (default) | 9610526, e541126 | goalprimplan.py:542-558, 616-624 | alive: 0.95 x (bank + credit) - bank; ANY end: -bank | test_shaping_telescopes_to_zero_over_an_episode | 3.12 #11, wanders |
| `--plan-shaping refund` | 8a33b2b (as the rule), e541126 (as a flag) | same lines | progress paid as it comes; a FAILED end (death, cap) pays -max(bank, 0) instead of its own progress; the finish keeps the bank | (no dedicated test found) | 3.12 #6, #12-#14; THE RECIPE |
| `--plan-mu-bound B` (0) | 2358501 | goalprimplan.py:218-223, 252-253; mirrored by record_ckpt.py:1340-1346 | pre-squash means B x tanh(raw / B) | none found | 3.12 #16 |

### 3.15 Search: `record_ckpt.py --plan-search M` (python/surfgym/goalsearch.py)

* `PrimSearch` (goalsearch.py:36-188; eef6365, 82c955a): M candidates = the mixture's heaviest mean +
  M-1 samples; each is flown from the env's exact state (teleported with `core.set_state` into a
  scratch core of M envs, one neutral tick) by the checkpoint's own greedy executor for the primitive's
  budget; score = finish: bonus + progress; death: -progress_coef x max(bank, 0); alive: progress +
  0.95 x V_planner(end) (+ end novelty only with `explore=True`) (:163-168); the best is committed.
  Depth 1: one primitive is simulated per candidate.
* Used only in tools/record_ckpt.py (:616-621 flag, :1349-1357, :2109-2135, report :2198-2205);
  the trainer never builds one (grep for PrimSearch). The goalsearch.py docstring (:18-20) also
  describes SCOUT envs in training; no such code exists (section 7).
* 82c955a: the first version started every slot from RELEASED keys while the real executor held some
  (`--keys-hold`) and paid novelty in an eval; fixed by copying the real executor's `KeysHold` state
  (:115-120) and `explore=False`.
* e48d76b (after 7ca92ea): the simulated executor now continues the real wrapper exactly - held
  action and view command, held keys and their episode-start detector, the decision phase - and
  reads the real env's current observation instead of spending a neutral tick (`_row_of`,
  `_policy`, `_start_obs`: `git show e48d76b:python/surfgym/goalsearch.py`, lines 187-236);
  `--plan-search` uses the same clone. Its message: over one primitive the simulated end moved
  from a median 182 u off the real one to 15 u (p90 58 u) on blue050.
* Default: `--plan-search 0` = the plain greedy planner (tools/record_ckpt.py:616).
* Measured: blue025 prim2f @ ~800M 6/6 with and without search (eef6365 message); ret_b050w 1.384B
  plain 0/9 vs search 2/9, 109 of 296 candidates died in simulation (82c955a message, L:27177-27181);
  1.895B plain 4/9 vs search 3/9 (L:27196-27197); blue100 and both blue200 runs 0/9 with and without
  (L:27199-27203); blue200 @ 2.28B 0/9, 89 of 192 candidates died (L:27290). Recording files
  runs/ret_b050w_es_plain.jsonl, runs/ret_b050w_es_search.jsonl, runs/ret_b200sL_es_plain.jsonl,
  runs/ret_b200sL_es_search.jsonl exist; which checkpoints they were made from is UNVERIFIED.

### 3.16 Metrics

* progress.csv columns `PRIMLEARN_COLS` (goalprimplan.py:93-103), appended last and only under
  primlearn; computed in `note_and_row` (:771-829):
  * executor: `exec/complete` (share of the planner's primitives completed), `exec/arc_frac`,
    `exec/complete_unif` (the uniform openers), and since a56ae10 `exec/track_strict` /
    `exec/track_lenient`: per tick exp(-d / 64 u) with d = distance to where the curve wants the agent
    at that tick, and exp(-d / 256 u) with d = distance to the nearest point of the path (sampled every
    10 ticks), averaged over each primitive (:103-110, :475-483; eval :895-903, :985-991);
  * planner: `plan/adv_plan` (progress the primitive promises), `plan/adv_real` (made; never positive
    on a death), `plan/plan_fwd`, `plan/death`, `plan/ep_prog`, `plan/ep_prog_start`, `plan/finish`,
    `plan/finish_start`, `plan/eval_finish` (the verdict), `plan/credit_frac`, `plan/cover_ep`, and
    its PPO statistics (`plan/entropy`, `plan/loss_pi`, `plan/loss_v`, `plan/kl`, `plan/updates`,
    `plan/cover`, ...).
* Dashboard: viewer/runs.js puts the primlearn charts first (`PREFERRED`, :16-26) and describes each
  column (`GROUP_DESC` exec / plan :52-53, `DESC` :57-79). Added in 1b96bb6, extended by 069a53c,
  61fc4ba, 271fc31, a56ae10; quoting fixed in 7ca92ea.
* The eval note (goalsys.py:1268-1280) prints "plan-eval finish k/n (prim planner greedy: P
  primitives, track s/l strict/lenient, cmpl X%; start D u from the finish, T s)".

### 3.17 Viewer: the planner's plan lines

* e3ea977: the eval hooks attach `episode_end(ep)` to `episode_meta` (goalprimplan.py:1006-1015);
  `record_rollout` merges its return into the episode's trailer (python/surfgym/record.py:221-226):
  `plans: [{t, numbers, track, line}]`, t = the episode tick the primitive starts on.
* viewer/app.js:427-489 (`buildGoalOverlay`, `updatePlanOverlay`), :1066, :1090-1098: the active
  primitive is a solid line, earlier ones dashed and dimmed, later ones hidden, and the overlay text
  reads "planner primitive k/N  (solid = active, dashed = earlier)" (line style and text, not colour
  alone). Before this commit a recording carried only the first primitive (the header's line).

### 3.18 The planner-override ablation: `record_ckpt.py --plan-override straight|random|frozen`

* a56ae10; tools/record_ckpt.py:622-626 (default: no override); goalprimplan.py:912-923 (straight =
  all-zero numbers, a straight line along the motion; random = step 1's uniform draw with a fixed
  seed) and :999-1003 (frozen = the episode's first primitive, no re-plan).
* Result, source = the a56ae10 commit message only: rec_b050 @ 1.754B, planner 8/9 vs straight 0/9,
  random 0/9, frozen 0/9; tracking (strict/lenient) planner 0.16/0.54, straight 0.23/0.72, random
  0.33/0.73 - "the executor DECODES the planner's up-to-the-sky curves; without them it falls". The
  recordings behind these numbers were not found under runs/ (UNVERIFIED).

### 3.19 Flat primitives: `--prim-flat 1`

* a56ae10; python/train_fast.py:4616-4619; goalprim.py:54-63 (`curve(..., flat=True)`: initial
  pitch 0 and vertical rate 0 - the curve bends sideways only, at its start height); mirrored by
  tools/record_ckpt.py:1334-1335. The planner's action keeps all 2 x knots numbers
  (`PrimitivePlanner.n_numbers`, goalprim.py:98-100); the three vertical ones are then ignored by
  the curve.
* Only run: flat_b050 (1.4). Observed in its log, not a result: the step line's planner entropy rose
  from 4.25 at ~544M to 11.71 at ~689M, and at the 704M eval the executor tracked the flat primitives
  at 0.48/0.85 (strict/lenient) with 70% completed (runs/flat_b050_launch.txt).

### 3.20 Other tools, docs and infrastructure of the period

* Visualisation: tools/viz_planner.py (4af766a, with `record_ckpt --dump-plans`,
  tools/record_ckpt.py:612-615), viz_planner_surf.py (dcf3d0b), viz_planner_bfs.py (d0645ea),
  viz_planner_input.py (38f04eb), viz_jump_planner.py (3dfa53a), viz_jump_probes.py (5e4bd5d,
  e1aafda), viz_plan_repr.py and viz_surf_algo.py (6778c69), viz_primitives.py, viz_curve_sliders.py.
* Docs: docs/litsurvey-planner-executor.md (aab9b7d; section 7 is the staged recipe the ledger
  cites as 7.1-7.6), docs/planner-design.md, docs/plan-representation-survey.md,
  docs/hrl-reward-survey.md.
* Infrastructure: c435642 (a bare map stem resolves under maps/ and maps_pool/), 6cb998b (the
  dashboard serves pool maps' zones.json), a52efac (run_arm.sh gates `--goal-route` under rule 0),
  267fe9e (deploy_box waits for the dpkg lock and fails without a built core).

### 3.21 MCTS over primitives (e48d76b, 14:52; eval-time only; no ledger entry)

Line numbers in this section are at e48d76b (`git show e48d76b:<path>`).

* Code: `_Edge` (python/surfgym/goalsearch.py:242-272) and `PrimMCTS(PrimSearch)` (:275-460: docstring
  :276-290, `_expand` :315-412, `choose` :415-460). Wired only into tools/record_ckpt.py:
  `--plan-mcts N` expansions per decision (default 0 = off), `--plan-mcts-k` 6 children per
  expansion, `--plan-mcts-depth` 4 primitives, `--plan-mcts-c` 1.25 (PUCT) (tools/record_ckpt.py:
  622-632); `make_primlearn_hooks` passes the real env's observation and logs the simulation's
  predicted end against the real one (goalprimplan.py diff in e48d76b). The trainer does not use it.
* Algorithm (docstring and code above): a node is an exact simulator state + the executor wrapper's
  state + the episode's banked progress. Expanding a node draws K primitives from the planner's
  mixture AT THAT STATE (the heaviest mean + K-1 samples) and flies them in one batch with the
  checkpoint's greedy executor on a K-env scratch core until each closes as in the real eval (arc >=
  0.9 or the 300-tick budget, at the executor's next decision tick), dies or finishes. Edge reward =
  the planner's (progress per 1,000 u; + the finish bonus; a failed end pays -max(bank, 0)); leaf =
  the planner's value head; MAX backup (`_Edge.q` = r + 0.95 x the best child); PUCT over
  min-max-normalised values with a uniform prior; the most-visited root edge is committed (ties: the
  higher value) and the tree is rebuilt at the next decision.
* Tests (e48d76b): `test_mcts_edges_back_up_by_max`, and an MCTS recording inside
  `test_trainer_and_recorder_run_primlearn` (`--plan-mcts 3 --plan-mcts-k 3 --plan-mcts-depth 2`,
  asserting "MCTS: 3 expansions", "mcts: " and "mcts fidelity" in the output).
* Measured - runs/mcts_eval/summary.txt (written 15:00; per-row logs runs/mcts_eval/b*_*.log and
  recordings traj_b*_*.jsonl; 9 recorded greedy episodes per row; finishes as printed by
  record_ckpt.py, "goals: k/9 reached"):

| map, checkpoint | plain greedy | `--plan-search 8` | `--plan-mcts 16 --plan-mcts-k 6 --plan-mcts-depth 4` | MCTS fidelity: ending matched; alive-close error median / p90 |
|---|---|---|---|---|
| blue025, runs/research/rec_b025/rec_b025_keep.pt | 9/9, 11.5 s | 9/9, 10.4 s | 9/9, 10.4 s | 35/35; 0 u / 67 u |
| blue050, runs/research/rec_b050/ckpt_1754267648.pt | 8/9, 13.4 s | 9/9, 13.7 s | 9/9, 13.1 s | 42/42; 1 u / 37 u |
| blue100, runs/ret_b100L/ckpt_2357198848.pt | 1/9, 24.0 s | 7/9, 20.7 s | 7/9, 20.8 s | 64/64; 0 u / 48 u |
| blue200, runs/ret_b200m/ckpt_final.pt | 0/9 | 0/9 (107 of 200 candidates died in simulation) | 0/9 (0 finishes seen below the root) | 34/34; 0 u / 31 u |

* The blue050 greedy row (8/9; tracking strict 0.164 / lenient 0.541) is the same checkpoint as the
  `--plan-override` ablation of a56ae10 ("planner 8/9 ... tracking planner 0.16/0.54", 3.18), and
  agrees with it.
* 7a40522 (15:05; seen as uncommitted edits at 15:04, then committed): `unsquash`, `_Edge.disc` and
  three `PrimMCTS` options with recorder flags - `--plan-mcts-time` (gamma per nominal primitive
  duration of flight instead of per primitive), `--plan-mcts-gamma` (0.95), `--plan-mcts-uniform F`
  (that share of each expansion's sampled children drawn uniformly from the primitive ranges). Two recordings in runs/mcts_eval used
  them: b025_mctstime.log (9/9, 10.5 s) and b200_mctswide.log (48 expansions, 8 children of which 4
  uniform, depth 6: 0/9; 195 of 328 candidates died; 0 finishes seen below the root; ending matched
  40/41). A b025_mctsfast.log was being written (empty) at 15:05.

## 4. What failed, and why

| what | measurement | the reason given | source |
|---|---|---|---|
| reverse curriculum (lab100 / lab200) | 0 finishes from the start | the curriculum moved the start to the detour, it did not teach it | L:24766-24783 |
| CPPO `--crl` | lab100 never leaves the start room; lab200 finished in training, then lost it; blue050 null | approx_kl 0.05-0.36 (spike 1.28) on lab100 and 0.06-0.64 on lab200 vs ~0.01 in reward-driven cells; view sigma 0.014 (lab100), 2.07 (blue050) | L:24791-24806, L:24919-24967, L:25108-25113 |
| SGCRL | 0 goal hits in ~1.4M training episodes on 3 maps | - | L:25096-25127 |
| GRU | at the first wall on both rungs | memory cannot remember a detour no episode contains | L:25085-25094 |
| learned planner with +0.7 per completed plan | routes ~3x the shortest path, evals 0 <-> 9/9 | completion stream worth 14 > the +10 finish | L:25644-25651 |
| lenient completion judge | wall crossing not reduced by a tighter corridor | tracker window spans a 7-vertex plan | L:26049-26077 |
| fixed-vocabulary surf executor / planner (psEF050v, psEF050L) | 0% of fixed shapes completed; never left the platform; planner null | constant-descent shapes are not flyable; nothing paid for leaving the platform | L:25877-25900 |
| learned planner on hard01 | 0 of 32,768 training episodes finished | Euclidean progress term, a local visit window, the finish never seen | L:26228-26247 |
| jump planner at depth <= 8 on hard01 | loops | memoryless + deterministic + horizon shorter than the detour | L:26288-26306 |
| jump planner at depth 16 | 0.2% | FIN_BONUS 100 went negative on long routes | L:26308-26313 |
| stage 1 from scratch on blue200 (srR200f) | 0/9 through 806M | its mix and/or its 10x weaker per-unit reward, not the fan (rpCTL later converged) | L:26537, L:26578-26586, L:26784-26788 |
| the old ride route as a flat reward (efPLN200) | 0/9 through 504M | its start segment pays leaving the platform into the void | L:26538, L:26549-26556 |
| FiLM gating (rpFILM, rpBOTH) | never / ~400M later | [inference in the ledger] plan-dependent image features before the policy reads either | L:26792-26800 |
| primlearn variants | see 3.12 #1-#5, #9, #11 | each is an episode end or a stream priced differently from the others | L:26947-26989, L:27053-27135 |
| blue200 under the recipe and its variants | 0/9 at every eval to 4.159B | the start is under-trained; the planner's means pinned at the action bounds | L:27258-27352 |
| first search version | 0/9 with search vs 2/9 plain | released keys in simulation; novelty in an eval score | 82c955a message |
| p2_b200 launch | died at startup | a resume restores `heldout_maps`, and step 1 had blue200 held out | L:27039-27041 |
| the ride graph on the benchmark maps | cannonball, celestial, unitfarmer2, petrus_lite, utopia not connected | flights between ramps longer than one 128 u hop; momentum not modelled | L:26474-26506 |

## 5. Open problems

### 5.1 blue200's start (planner saturation)
Six greedy episodes of ret_b200s2 @ 1.811B all die in 2.0-3.6 s leaving the start platform
diagonally (L:27260-27263); under the recipe only 10% of spawns are at the map start and half of
those open with a uniform primitive, so the spawn decision is barely trained (L:27263-27268). The
greedy primitives from the spawn sit at the action bounds, e.g. [-180, 167, 171 | 90, 90, 90]
(L:27282-27288); every sample then saturates too and PPO never tries the straight first move
(L:27300-27303). `--plan-mu-bound 1.5` made the primitives more followable (executor completion 7.0%)
but erased the planner's old codes (training finishes 18% -> 0%) and never produced a greedy finish
(L:27324-27327, L:27341-27352). Not in the ledger but in the logs: several blue200-trained runs
finished the HELD-OUT blue025 at some evals, e.g. ret_b200s 4/9 at 905M
(runs/research/gate_bench/ret_b200s_launch_final.txt) and ret_b200m 1/9 at 2.95B and 3.15B
(runs/ret_b200m_launch.txt:233, :435) [checked].

### 5.2 The executor does not follow the planner literally
prim2f @ 691M: `exec/complete` 0.74%, `exec/arc_frac` 30.7%, planned +65.5 u vs real +166.5 u per
primitive (runs/prim2f_b025/progress.csv [checked]; L:27006-27017: "a SIGNALLING CODE the executor
responds to, not a geometry it follows"). The recipe runs keep it: rec_b050 @ 1.006B `exec/complete`
0.17%, planned -68.9 u vs real +87.7 u; rec_b025 @ 1.303B 0.92%, +32.1 vs +263.8 u
(progress.csv [checked]). The override ablation (3.18) says the executor needs the planner's curves
anyway. `--plan-obey` was the one attempt to make followability count and was inconclusive (3.12 #8).
Recommendations in docs/hrl-reward-survey.md section 7 that are not implemented at 7a40522: remove the
executor's +50 map-finish bonus (still paid, rewards.py:1550), a pull toward the plan off the
corridor, an SMDP discount by primitive duration (plan_gae uses 0.95 per primitive), a time-left
scalar. Also noted (not measured): the primitive completion tracker uses window 16
(goalprimplan.py:335-336), the setting whose loophole `--plan-strict` closed on the maze (3.3).

### 5.3 Search is depth 1; search of any depth is eval-time only
`PrimSearch` simulates one primitive per candidate and scores its end with the planner's value
(3.15). The ledger records it helping once on blue050 and never on blue100 / blue200 (L:27168-27203,
L:27290); the later runs/mcts_eval table (3.21) shows `--plan-search 8` taking blue100's checkpoint
from 1/9 to 7/9 and blue200 still at 0/9. Neither `PrimSearch` nor `PrimMCTS` is used in training
(grep of python/train_fast.py), so nothing the search finds flows back into the planner or the
executor.

### 5.4 MCTS
Until 14:52 MCTS existed only as a proposal: the ledger "(afternoon)" entry of 2026-09-23 ("The
planner as a SEARCH (MCTS-like) ... For the future", L:26044-26047) and docs/planner-design.md
sections 6-7 (MCTS over jump points, pi from U, later learned from the search's visit counts). At
14:39 no MCTS code was in the repo. e48d76b (14:52) added `PrimMCTS`, an eval-time MCTS over
primitives with exact simulator states as nodes (3.21); 7a40522 (15:05) added three knobs, and
its first measurements exist only in runs/mcts_eval (no ledger entry): MCTS matched `--plan-search 8`
on blue025 / 050 / 100 and found no finish anywhere in its trees on blue200. Not implemented at
7a40522: MCTS in training, a prior learned from visit counts (planner-design.md section 7, step 2),
tree reuse between decisions (the tree is rebuilt, goalsearch.py:288-289 at e48d76b). The jump planner
(3.5) is an exhaustive depth-limited search, not MCTS.

### 5.5 The flat-primitive run
flat_b050 is running, 0/9 at its first five evals (502M-905M), no ledger entry (1.4, 3.19).

### 5.6 Other open items named in the sources
* Momentum: the ride graph does not connect the real benchmark maps (L:26488-26496,
  planner-design.md section 8); primlearn has no graph, but no primlearn run exists on those maps.
* primlearn under DDP is refused (train_fast.py:12519-12521).
* The jump planner has no tests (3.5); `--plan-override`, `--prim-flat`, `--plan-mu-bound` and the
  track metrics have no dedicated tests (grep of tests/).
* srTGT200 (stage 1 on the tight graph with srR200f's mix) was "running" at L:26541 and is never
  reported again; runs/srTGT200 does not exist.

## 6. How to verify

Commands (CPU only; a GPU trainer is running on this machine):

    git log --since=2026-09-22T20:00 --format="%h %ad %s" --date=format:"%m-%d %H:%M" petrusnight
    git show -s <hash>                    # full commit message
    git log --reverse -S"<heading text>" --format="%h %ad" -- docs/research-results.md   # who added a heading
    grep "plan-eval finish" <launch log>  # the per-eval greedy result lines

`progress.csv` columns used above: `time/total_timesteps`, `plan/eval_finish`, `race/eval_finish_s`,
`race/map_pct`, `plan/finish`, `plan/finish_start`, `exec/complete`, `exec/arc_frac`,
`plan/adv_plan`, `plan/adv_real`, `plan/death`, `race/heldout_finishes.<map>`.

| headline | where |
|---|---|
| rec_b025 0,0,1,2,9,5,6,8,8 of 9 at 502M..1.308B | runs/research/rec_b025/progress.csv; runs/research/gate_bench/rec_b025_launch_final.txt; L:27212 |
| rec_b050 0,0,1,2,2,7,8,9,4,8,5,8,8,8,9 of 9 at 502M..1.912B | runs/research/rec_b050/progress.csv; runs/research/gate_bench/rec_b050_launch.txt; L:27330-27337 |
| ret_b050w 0,0,0,0,2,1,3,4,0,5 at 1.135B..2.041B | runs/ret_b050w/progress.csv; L:27194-27196 |
| ret_b100L 0 x4, 1,2,0,0,2,4,3,2 at 1.356B..2.463B | runs/ret_b100L/progress.csv; L:27241-27243 |
| blue200 all 0/9 | runs/ret_b200sL, runs/ret_b200m, runs/research/ret_b200f, runs/research/ret_b200s2 (progress.csv); runs/research/gate_bench/ret_b200*_launch*.txt |
| prim2f 6,8,8 of 9 at 690M, 791M, 891M | runs/prim2f_b025/progress.csv |
| p2 baselines 0/9 | runs/research/gate_bench/p2_b050_launch_final.txt, p2_b100_launch_final.txt, p2_b200b_launch_final.txt |
| stage-1 surf and jump runs | runs/<run>_launch.txt for srR025f, srW050f, srW100f, srW200f, srR050f, srFT200, jS050W, jS200L |
| labyrinth planner runs | runs/<run>_launch.txt for plLAB100a, plLRN100b, plLRN200b, plLRN100h, plHARDa, jH2e12 (3-episode evals) |
| plan-representation study | runs/research/gate_bench/rp{CTL,DENSE,FILM,BOTH}_launch.txt |
| recipe flags and provenance | runs/research/gate_bench/box_rec_b025.txt:1; research_rec_b050_launcher.txt:12; run.json of both runs |
| training rates quoted in the ledger (prim2f 39.1% / 26.8% at 691M, rec_b025 45.7% / 29.1% at 1.303B) | the progress.csv row nearest that step (checked: 0.3915 / 0.268; 0.4571 / 0.2908) |
| MCTS code and its first measurements | `git show e48d76b`, `git show 7a40522`; runs/mcts_eval/summary.txt and runs/mcts_eval/*.log |
| flat_b050 state | runs/flat_b050_launch.txt (`grep "plan-eval finish"`), runs/flat_b050/run.json |

Tests (not run for this document). PowerShell: `$env:CUDA_VISIBLE_DEVICES='-1'` (an empty value is
deleted on Windows and leaves the GPU visible), then e.g.
`python -m pytest tests/python/test_goal_primlearn.py tests/python/test_goal_prim.py -q`. Relevant
files: test_goal_primlearn.py (10 at e48d76b), test_goal_prim.py (7), test_goal_planner.py (15),
test_goal_planner_tight.py (2), test_goal_learned.py (14), test_goal_surf.py (14),
test_goal_proposals.py (21), test_spawn_pool_solid.py (3), test_plan_film.py (4), test_rdp_fast.py (1),
test_crl.py (16), test_sgcrl.py (23), test_goal_curriculum.py (15) - counts are `def test_` functions,
parametrisation adds cases. They need build/surfcore.dll and the maps_pool maps;
test_trainer_and_recorder_run_primlearn launches the trainer (CUDA_VISIBLE_DEVICES=-1 in its own
subprocess env, test_goal_primlearn.py:120-125) and writes, then deletes, runs/primlearn_smoke.

Caution: a recording's trailer field `end` is computed from the core's step reward
(python/surfgym/record.py: `end = "done" if r0 >= _DONE_BONUS_MIN else "fail"`); in the primlearn
recordings under runs/ (e.g. runs/research/rec_b050/traj_1754267648_rec_greedy_plans.jsonl) every
episode reads "fail", so do not count finishes from it (why was not investigated).

## 7. Discrepancies found, and UNVERIFIED items

### 7.1 Heading times later than the commit that introduced them (uncorrected in the ledger)
Found with `git log --reverse -S"<heading>"`. The ledger corrects 09-24 18:58/19:11/19:26 (L:26636)
and five 09-25 headings (L:27087); these are not corrected:

| heading as written | introduced by | line |
|---|---|---|
| 2026-09-23 03:20 and 03:30 | 1c4a977 02:53 | L:24809, L:24919 |
| 2026-09-23 03:45 | a52efac 03:32 | L:24969 |
| 2026-09-23 04:25 | 3808e67 04:17 | L:25157 |
| 2026-09-23 04:40; "User addition (04:50)" | 79a87d3 04:25; 76ac373 04:29 | L:25212, L:25252 |
| 2026-09-23 06:15 | c4832a5 05:59 | L:25387 |
| 2026-09-23 08:30 | 1709d25 08:14 | L:25701 |
| 2026-09-24 07:20 (both entries) | a83e170 07:04 | L:26249, L:26315 |
| 2026-09-24 11:40; "CORRECTION (11:50, same day)" | eb69019 11:27; 09e562d 11:28 | L:26474, L:26498 |
| "CORRECTION (2026-09-24 20:35, machine clock)" | 62688a0 20:12 | L:26698 |

### 7.2 Ledger vs data
* rec_b025: L:27216-27222 lists 502M 0/9, 603M 0/9, 1.207B 8/9; the run also had 703M 1/9, 804M 2/9,
  904.9M 9/9, 1.006B 5/9, 1.106B 6/9, 1.308B 8/9 (progress.csv and launch log). The end-of-night table
  (L:27312) gives blue025 as 8/9; the best eval was 9/9.
* ret_b050w: L:27170-27172 calls 1/9 at 1.638B the first finish "after 0/9 at every eval of every
  earlier blue050 arm"; the same run was 2/9 at 1.537B (progress.csv; the ledger's own L:27195).
* L:26999 "603M (prim2e) 0/9 ... 44%": prim2e_b025's progress.csv has one eval (502M, map_pct 38.2%)
  and ends at 591M; a 603M eval at 44.1% is in prim2d_b025's progress.csv. Possibly mislabelled.
* L:26875 says prim1_b025 was stopped at 516M; its launch log's last step lines are at ~521M.
* Held-out blue025 finishes of blue200-trained runs (5.1) and the jH3* results (3.5) are in the logs
  but not in the ledger.

### 7.3 Code vs ledger / docs
* goalprimplan.py's module docstring (:27-45) presents exact potential-based shaping as the
  progress term, and `plan_shaping` defaults to "pbrs" (goalprimplan.py:89, train_fast.py:7444-7445),
  while THE RECIPE runs `--plan-shaping refund` (L:27133-27137, L:27317). A primlearn run without the
  flag gets pbrs.
* e541126's message and the comment at goalprimplan.py:543-546 date the refund rule "06:50-07:10";
  by commit times it was the default from 8a33b2b (06:16) to 9610526 (07:07).
* goalsearch.py:18-20 (also at e48d76b) describes SCOUT envs using search in training; no code in
  python/train_fast.py builds a `PrimSearch` or `PrimMCTS` (3.15).
* rewards.py:2270 sets `SPAWN_UNSTICK_U` = 64 u; the ledger's fix table says sidistic had "no clear
  height within 128 u" (L:26626). The comment at rewards.py:2288 says the edgeflow spawns were "3 u
  too low"; the test asserts a 4 u lift (tests/python/test_spawn_pool_solid.py:33).
* viewer/runs.js:76 describes `plan/reward` as progress + finish bonus + end novelty (no coverage, no
  charge-back); `PrimLearnedPlanner.describe()` (goalprimplan.py:430) always prints "a death charges
  the bank back", whatever `--plan-shaping` is.
* Commit-message test counts differ from the `def test_` counts (test_goal_planner 18 vs 15,
  test_goal_learned 16 vs 14, test_goal_proposals 23 vs 21, test_goal_surf 16 vs 14); parametrisation
  may explain it (UNVERIFIED; tests not run).

### 7.4 UNVERIFIED
* The `--plan-override` straight / random / frozen numbers (3.18): commit message only; their
  recordings were not found (the planner row agrees with runs/mcts_eval/summary.txt, 3.21).
* Which checkpoints the runs/*_es_*.jsonl search recordings came from (3.15).
* Whether the runs/mcts_eval series was finished (in progress at 15:05); no ledger entry exists for
  MCTS, flat_b050 or the override ablation.
* Any result of flat_b050 (still running at 15:05).
* srTGT200's result (never reported; no run directory).
* Ledger-only numbers not re-derived here: the reverse curriculum, CPPO, SGCRL, ladder, GRU and
  xsG6n figures; psEF050v / psEF050L; the world-record coverage percentages; the search results of
  06:32 and 08:10; the blue200 saturated knot values; the p2_b050 bottom-row description; the timing
  breakdowns. Each is cited to its ledger line above.
