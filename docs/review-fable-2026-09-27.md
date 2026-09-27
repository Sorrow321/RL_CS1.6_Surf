# Third-judge review: the planner/executor hierarchy on edgeflow blue200 (Fable, 2026-09-27)

Read-only review. Sources: the ledger (docs/research-results.md, 2026-09-19 to 2026-09-27
03:16), docs/planner-design.md, the code (python/train_fast.py, python/surfgym/goalprimplan.py,
goalprim.py, goalarc.py, tools/edge_archive.py, tools/contingent_archive.py) and the run files
(runs/research/ch3v3LC_b200, archive_*, contingent_b200_f4k, runs/efTGT200, runs/research/rpCTL,
runs/arch2_b200). No GPU, no training, no simulator runs. Every number below is quoted from those
files or computed from them on the CPU (pandas over progress.csv / json). One seed per arm
everywhere, as the ledger says; I only lean on gaps far outside the 27% noise floor.

## 0. Verdict in one paragraph

blue200 is not hard. It was solved in this repository three days before the planner program
started, twice, in about six minutes of GPU each: a FLAT 25 Hz policy paid arc progress along a
fixed route line (efTGT200: 79% of training episodes finish at 199M steps, greedy 6/9 from the
start at 202M, 7/7 playable spawns at 706M, at 609k steps/s = 5.5 minutes for 200M) and the
same policy with the route drawn into its fan (srFT200: 9/9 greedy at 202M; rpCTL: 9/9 from
303M and 8-9/9 ZERO-SHOT on the never-trained blue100). Ledger 2026-09-24 19:35, 19:45, 23:15.
The only thing those runs took from outside the agent was the route, and tonight's edge archive
produces a route from the agent's own states in 2-8 seconds on 6 of 6 seeds (ledger 2026-09-27
02:01). The five days in between tried to make a PPO "planner" DISCOVER the route through a
reward that is binary until the first finish, on top of an executor that was deliberately
trained to ignore everything past the next two seconds. That combination cannot work on a
10-13 plan route and the ledger's own numbers say why (section 2). The learned online planner
should be dropped. Macro exploration is needed and is already solved by the archive; the
memory the policy needs is the route line; the rest is the flat recipe that already exists.

## 1. The facts that decide it (all from the ledger and run files)

| what | route source | executor objective | blue200 greedy from the start | steps | ledger |
|---|---|---|---|---|---|
| efTGT200, flat policy, `--race-arc` line, no planner, no fan | tight graph (map geometry) | arc progress along the WHOLE route, 20 s horizon | 6/9 at 202M, 7/7 playable at 706M | 200-700M (6-20 min local) | 09-24 19:35 |
| srFT200, fan + arc | tight graph | same | 9/9 at 202M, 9/9 at 303M, 404M | 200M | 09-24 19:45 |
| rpCTL, fan + arc, varied plans | tight graph | same | 9/9 from 303M; blue100 zero-shot 8-9/9 | 303M | 09-24 23:15 |
| efPLN200, flat, `--race-arc` on the RIDE route (air shortcut at the start) | ride graph | same | 0/9 to 504M | - | 09-24 19:35 |
| primlearn recipe, every variant (v1ri, cov, az, sil, straight, replan, 2x2 rays) | learned planner, refund_i | arc along a 2 s plan, return CUT at every re-plan | 0/9 in every eval of every arm, to 2B | 5 days | 09-25 .. 09-27 |
| + decision-time MCTS / archive / contingent archive | search in the simulator | same executor | 1/9, 2/18, 0/9; 2/9; 1/18 | - | 09-26 18:33, 09-27 02:41, 03:16 |
| edge archive (discovery only) | the agent's own states | the trained executor as the move | finishing chain in 2-8 s, 6/6 seeds | 0 training | 09-27 02:01 |

Three things follow before any code is read. (1) The executor / perception stack finishes this
map reliably when it is given a route (rows 1-3). (2) The route's SHAPE decides whether it
works as a reward (row 4: the ride route's air shortcut trapped the flat policy; a chain the
executor actually flew has no air shortcut by construction). (3) Finding a route on this map
is a solved search problem (last row). Everything the hierarchy failed at sits between (1) and
(3): learning, by PPO on a near-binary reward, what the archive finds in seconds and what the
flat policy then flies in minutes.

## 2. Critical flaws, ranked by how much they matter

### 2.1 The executor was trained to be reckless, on purpose (`--exec-cut 1`)

`train_fast.py:15126-15131`: under `--exec-cut 1` a re-plan is treated as an episode end in
GAE (`nonterm = 1 - max(done, cut)`), so the executor's return is the arc progress of the
CURRENT 2 s plan and nothing else. Its death costs it at most the remainder of that plan. The
recipe's own diagnostics show what that trains:

* `runs/research/ch3v3LC_b200/progress.csv`, after 800M: `plan/death` = 0.356 (36% of all
  primitives end in a death), `rollout/ep_len_mean` = 321 ticks = 3.2 s (the mean TRAINING
  episode, 90% of them from mid-route reservoir spawns, is one and a half plans long),
  `plan/finish_start` mean 0.0006.
* Survival per plan 0.64. A route of 10-13 plans therefore has P(finish from the start) of
  0.64^10 = 1.2% to 0.64^13 = 0.3% even if every CHOICE is right. This one number explains
  "greedy 0/9 from the start in every arm" and "training finishes 20-35% only from near-goal
  spawns" simultaneously.
* The closed-loop archive log (`runs/research/archive_cl2_b200/closed_loop.json`) measures the
  same thing per decision from REAL states: the survival of the best move is 1.0 at the spawn,
  0.06-0.6 at the corner (t = 2-4 s), 0.03-0.5 at the ramp turn (t = 8 s). The contingent
  planner's own Q values (`contingent_b200_f4k/episodes.jsonl`) are 0.10-0.47 for the best
  first move and drop to 0.000 for every move after one or two flights in 13 of 18 episodes.

The ledger read this as "the executor's reliability is the limit, not the choice of moves"
(09-27 02:41, 03:16). Correct, but it is not a mystery about skill: the objective never asked
for reliability. The flat policy in efTGT200 has the same network, the same observation, the
same physics, a 20 s horizon and a global monotone progress coordinate, and it learned the
corner and the ramp turn in 200M steps because dying forfeits everything after it.

`--exec-cut 0` was tried and farmed (v2 cells, 09-27 00:44: 0.1-0.7% deaths, every episode to
the 30 s cap). That farm is not an argument for the cut; it is a symptom of paying arc
progress PER PLAN. A per-plan reward can be collected forever by re-planning on a safe platform;
a global arc coordinate along one route is bounded by the route's length and cannot be farmed.
The "middle ground" the ledger discusses (bootstrap with the planner's value at the plan's end)
does not help either: under refund_i that value is ~0 everywhere before the first finishes
(section 2.2), so it would give the executor the same zero survival incentive.

### 2.2 refund_i is a binary reward, and a binary reward cannot teach a 10-13 step chain whose links each fail a third of the time

`goalprimplan.py:1364-1398`: under refund_i every primitive is paid its Euclidean progress, the
bank compounds at 1/gamma, and any non-finish end pays `-bank` (line 1383), so a failed episode
nets exactly 0 discounted. The ledger states the consequence itself: "V ~ P(finish) x ~12 minus
the bank at risk" (09-26 08:34), and from the corner P(finish) is ~0 (section 2.1). So the
planner's gradient at the start and along the bottom corridor is zero, the chain has to be
learned BACKWARD from reservoir finishes, and it stalled at the corridor because the corridor's
right answer is two or three "straight" plans with no reward of their own (09-26 11:42, 12:45).
Every arm from 09-26 09:07 to 09-27 01:51 (refund_i, pbrs, coverage 1.0, SIL, SIL-uniform,
HIRO relabelling, keep-going component, AlphaZero, expert iteration, --plan-return 2/3, the
3-choice head, the ray head, the judge family, 0.5 s replans) was a way of pushing a zero
signal around. The maze round says the same in the clean case: refund_i passes left200 where
the executor is near-deterministic (P_can ~ 1) and fails medium01/hard01, while the GEODESIC
reference passes every maze in 100M (09-25 19:31): what was missing was a dense, honest
progress coordinate, not a better bank rule. On surf the archive supplies that coordinate.

### 2.3 The plan alphabet is not a route; a fixed line through the fan is

Rays are "level straight lines at 0 / +-45 deg from the horizontal velocity's heading", falling
back to the VIEW yaw below 50 u/s of horizontal speed (`goalprimplan.py:462-476`,
`goalprim.py:32`). "Left" therefore names a different line from every slightly different
state, which is exactly why the archive's chain of choices replays open-loop 0-1/32 from the
start (09-27 02:01) although each edge survives 27-32/32 from its exact parent (seed 5:
fidelity product 0.77). The earlier 6-number primitives were worse: the planner pinned them at
the bounds and the executor "decoded the up-to-the-sky curves as a code" (override ablation,
09-25 15:08: the planner's primitives 8/9, straight 0/9, random 0/9, tracked LESS than
meaningless ones). Both alphabets are memoryless: nothing carries the route from one plan to
the next except the state's velocity.

Compare the fan of a FIXED route line: 8 points 0.25-2 s ahead of a polyline anchored in the
map. It carries the whole route, it is the same object on every map, and rpCTL's executor
trained on it generalises zero-shot to an unseen map. Design-doc section 10's list (frame,
replan more often, show the previous plan, replan on deviation) is an attempt to give the ray
alphabet the continuity that a line has for free.

### 2.4 The 2x2 "judge family" measured nothing, and the "frontier moved" reading is a pit dive

* The "strict" executor corridor was 384 u in 3D around a LEVEL ray at the plan origin's
  height. blue200's start platform is at z 592-608 (`maps_pool/surf_edgeflow_blue200.zones.json`)
  and the kill floor is at z ~275 (every greedy eval episode in
  `runs/research/ch3v3LC_b200/traj_1206910976.jsonl` ends at z 274.6-279.7): a fall of ~325 u,
  inside the 384 u corridor. From the bottom corridor (z ~405-435) the floor is 130-160 u below.
  So on this map a falling executor NEVER left the strict corridor and strict = lenient; the
  ledger's "the factors barely separate" (09-27 01:51) is the expected result of a null
  manipulation, not a finding about judges. (`goalarc.py:517-518`: delta is 0 only OUTSIDE the
  corridor.)
* "The map-start frontier moved from 16-21% to 30-37%" (09-27 01:26) is `plan/ep_prog_start`,
  a Euclidean progress fraction. All nine greedy episodes of the final eval end at
  x 860-2,293, y -140..-390, z ~276, after 2.9-5.0 s: north of the start, in the pit, on the
  straight line at the finish. 30-37% of 3,016 u is how far a diving agent gets toward the
  finish before it hits the floor. CLAUDE.md section 3 already warns that death-dives flatter
  `race/eval_progress`; the same warning applies to `plan/ep_prog_start` and was not applied.
  The 3-ray head did not move a frontier; it made the dive longer.

### 2.5 The greedy policy is not the policy that was trained, and every verdict is on the greedy one

`train_fast.py:2668-2680` (`GreedyTorchPolicy`): argmax on the categorical heads, including the
`--keys-hold` heads whose bin 0 is "keep", and z = mu on the view. The code's own docstring on
`SampledTorchPolicy` (line 2695) says the argmax mode "can be much weaker". Measured tonight:
the greedy archive died 17 of its first 24 flights and exhausted itself at depth 2
(`runs/research/archive_greedy_b200/summary.json`), while the sampled executor survives the
same first plan from the spawn 78-100% of the time in all nine closed-loop episodes
(`archive_cl2_b200/closed_loop.json`, t = 0). On blue050 the two agree (6/9 vs 6/9, 09-26
08:04), so this is not universal, but on blue200 every "0/9" is a measurement of a policy
nobody optimised. This is not the root cause (the sampled closed loop is still only 2/9) but
it means (a) all planner evals understate the trained policy by an unknown amount, and (b) any
distillation, archive or search that uses the greedy executor as the "deployable" policy is
using the wrong object. A cheap fix exists in the code (`TemperedTorchPolicy`, low temperature)
and should be the eval policy alongside greedy, reported together.

### 2.6 Perception is NOT the bottleneck on blue200 (answers question 1b)

The same 64x32 depth render, velocity and held keys, plus the fan, take the corner and the
ramp turn 9/9 in rpCTL / srFT200 and even without the fan in efTGT200 (7/7 playable). What the
executor lacks in the hierarchy is not pixels; it is (i) a plan that tells it where the turn
is before it arrives (a 2 s ray relative to its own velocity does not), and (ii) a reason to
arrive alive. A higher-resolution render would not change 2.1-2.3.

### 2.7 Smaller issues, for the record

* The discovery archive zeroes `tick` and `stuck_ticks` on restore (`edge_archive.py:208-210`),
  so a deep node never inherits the cap; the closed loop fixed this with `keep_clock`. Fine as
  a discovery tool, wrong for anything that claims real-clock finishes.
* Edge "fidelity" counted survival, not landing in the child's key (Codex's point; the
  analysis shows key hits 0-13/32 on the early edges, `archive_an_s5/edge_analysis.json`).
  So a "chain" is a set of individually reachable states, not an option sequence. That is
  exactly what a ROUTE LINE needs and exactly what an OPTION POLICY cannot use.
* The planner is non-Markov for its own reward (coverage state and remaining time unobserved,
  bank clipped to +-5, `goalprimplan.py:297`); the categorical head's entropy collapsed to
  0.13 of 1.10 under refund_i because failures are flat 0 and novelty/coverage become the
  objective (Codex 09-27 00:02). Real, but second-order next to 2.1-2.2.
* `--plan-replan 0.25` was judged "worse" with the credit window confounded (the ledger says
  so). Not a result about planning frequency.
* The hierarchy costs throughput: 200k steps/s (ch3v3LC) against 609k for the flat route
  policy on the same card. Every hierarchy hour bought a third of the samples.

## 3. Is the planner needed? (question 2)

The learned online planner is not, and the design doc's own mission statement says why.
Section 1 asks for "macro exploration: commit to a direction for seconds and find out that
something new is there". A PPO policy over 3 rays with a zero reward until the first finish does
not do that; it draws from a softmax at 0.5 Hz. The archive does exactly that: it holds every
state it reached, returns to the least-visited one, and extends it by one committed plan. On
blue200 it found the detour in 2-8 s on every seed where 2B steps of planner training found it
0 times from the start. The thing the design doc wanted is built and works; it is
`tools/edge_archive.py`, not `goalprimplan.py`.

What the planner DID buy, to be fair: relative to a flat policy on the deceptive Euclidean
potential, it passed blue025/050/100 (ctl_b050 0/9 vs rec_b050 9/9). But the ride-graph
planner + fan also passed those (srW050f 8/9, srR050f 9/9 from scratch, 09-24), and neither
comparison involved a self-found route. Against "flat policy + a route line" the learned
planner has no win anywhere in the ledger, and against the archive as an explorer it has no
win either.

What replaces its roles:
* macro exploration (find the route): the archive over exact simulator states, count-only
  selection, moves = whatever the current policy can do (plus random bursts, as
  `explore_phase1.py` does; that crossed blue050 in 28 min with no policy at all). Ecoffet et
  al.'s policy-based Go-Explore, already 90% built here.
* memory / commitment (keep the route): the route line, as the arc-progress coordinate and/or
  the fan. It is fixed in the map, so "commit" is automatic and the 25 Hz policy never has to
  hold a decision for seconds.
* lookahead in the head: the simulator, at discovery time only. At deployment nothing but the
  flat policy runs; no MCTS per decision, no 4,000 flights per move.

The elaborate machinery of design-doc sections 4-7 (P_can, energy pruning, a learned jump
model, MCTS over jump points, AlphaZero on the planner) models what the exact simulator already
gives for free. Do not build a learned model of a simulator you can fork.

## 4. Why humans find this map easy (question 3)

* A human SEES the loop once and keeps it: a high-resolution wide view, head turning, and a
  persistent mental map. The agent has 64x32 depth at 120x90 deg, no recurrence (`rnn: none`),
  no frame stack: it cannot represent "the corridor I saw 3 s ago". Generic substitute: the
  archive (memory in the simulator) plus the route line (memory in the observation). Both
  exist.
* A human PLANS ahead offline: "ramps on the left go somewhere, the finish is up there". The
  agent must either discover the route by reward (hopeless when the potential points the wrong
  way for 3,900 u) or by search (seconds). Generic substitute: search in the simulator. Exists.
* A human PRACTISES with memory of failures: restart near the corner, try the turn again. That
  is return-to-state plus spawn curricula, both in the trainer (`--respawn-frac`, the reservoir,
  `--demo-file` with the agent's own states).
* A human COMMITS: once the route is decided, the corner is not re-decided every 2 s from
  scratch. Generic substitute: the fixed line. Exists.
* A human has an intuitive momentum model ("I need speed for that ramp"). The agent learns it
  only through reward; the archive's speed bins at least make speed part of the explored state.
* A human's motor skill is CONSISTENT once learned. PPO's policy is stochastic (view sigma
  ~0.3, held keys with a keep bin) and the greedy mode is a different policy (2.5). Generic
  substitute: train longer on a fixed route with a full-episode horizon; evaluate the sampled
  or low-temperature policy; both cheap.
* What cannot be supplied generically: semantic priors about ramps. Not needed on edgeflow.

The honest summary: humans have a map and a route; the agent was asked to grow one out of a
binary reward. Give it the map's route (from its own search) and the gap closes, as the
09-24 runs showed.

## 5. The simplest design, and ONE decisive experiment (question 4)

### The design

Same flags and constants on every map, no demo, no map constant:

1. DISCOVER: `tools/edge_archive.py`-style search from the true start over exact simulator
   states, count-only selection, moves from the current policy (sampling) plus random key
   bursts; stop at the first finishing chain. Constants: 128 u cells, the speed/azimuth bins,
   1/sqrt(1+n). Output: a chain of the agent's own states and the flown path.
2. LINE: resample the chain's flown path at 128 u (`route.resample_polyline`, the format
   `--race-arc` already loads: an npz with `route (N,3)` and `spacing`). Provenance: the
   agent's own states (CLAUDE.md 0, SELF_STATES).
3. TRAIN: the existing flat recipe with `--race-arc <line>` (corridor 1,500 u, window 16,
   gamma 0.9995, +50 finish, the reservoir at 0.9), optionally the fan
   (`--goal-obs fan`, srFT200's form) and the chain's states as a self-spine
   (`--demo-file chain_states.npy --demo-window 10`, as arch2 did, but WITHOUT the planner).
   Evaluate greedy AND sampled from the true start.
4. ITERATE: the trained policy is the archive's move operator on the next map or on a stuck
   map (it is also what rpCTL showed generalises zero-shot with the fan).

This is Go-Explore phase 1 + phase 2 with the route as the coordinate instead of the potential,
which is the one thing wave 6 (efGEwin_blue050, 09-20) did not have when it was stopped half
way down the spine under the deceptive geodesic reward.

### The experiment (local 5090, ~15 minutes, no rental)

Build the line from `runs/research/archive_b200_s5/chain.json` (the flown `path` points of
every node, concatenated; seed 5's edges survive 27-32/32 so the line is physically flyable
end to end) and launch efTGT200's exact configuration with that line in place of the tight
route: `SCRATCH=1 POT=off SELF_STATES=1 bash tools/run_arm.sh <arm> --race-arc <chain
line.npz>` on blue200, 500M steps, evals every 100M from the true start, greedy and sampled.
Its control is efTGT200 itself (same flags, the map-geometry line), already on disk.

* CONFIRMS the design: >= 6/9 greedy from the start by 300M (efTGT200: 6/9 at 202M), training
  finish rate above ~50% by 300M. Then the planner program is closed on edgeflow and the same
  three steps are run unchanged on blue025/050/100 (trivial) and on a real map (section 6).
* KILLS it: 0/9 greedy and 0/9 sampled at 500M with training finishes under 20%. Then read
  where the episodes die against the line: if it is the chain's corner, the line's shape is
  the problem (rebuild from the highest-fidelity chain or the union of several); if it is
  everywhere, the arc coordinate itself does not carry this policy, which would contradict
  three earlier runs and be worth knowing.

### Attacking your own candidate (drop the planner, archive route, xSELF-style flat training)

It holds up, with three caveats that are real and one that is not:

* NOT inherited: the reliability problem. It was manufactured by the 2 s return cut and the
  per-plan reward (2.1). The flat policy's return spans the episode and its progress
  coordinate is global; the same executor stack learned this map's corner in 200M steps
  under exactly that objective. The chain's per-edge unreliability is a property of the
  option executor from drifted states, not of the line.
* Caveat 1, the line's shape: a chain flown at 500-900 u/s by a reckless sampled executor is
  jerkier than the tight route. The 1,500 u corridor and the order-only window give the policy
  room to find its own path near the line; efPLN200 shows the one shape that does trap (an air
  shortcut), which a flown chain cannot contain. If the corner segment is ugly, use the
  best-fidelity chain or several chains; do not hand-edit it.
* Caveat 2, the move operator on a NEW map: tonight's archive used an executor trained 1.25B
  steps on blue200. For the recipe to be generic the operator must exist before map-specific
  training: random bursts (28 min on blue050, fails blue025's ramps in 30 min), the generic
  step-1 primitive follower (prim1_b025 followed 35-40% of random primitives on the held-out
  blue200), or the flat policy trained on the potential until it stalls, then used as the
  operator with its own moves plus noise. This is the SECOND experiment, not the first, and it
  is the one that decides generality: re-run step 1 on blue200 with an operator that has never
  seen blue200. The archive tolerates a weak operator (it keeps every survivor); the question
  is minutes versus hours.
* Caveat 3, the deployed artifact is a per-map policy trained by one recipe, not one universal
  policy. That matches CLAUDE.md 0b ("one recipe ... same flags and constants"), and the fan
  version (rpCTL) is the path to a route-conditioned universal executor if that is wanted.
* Not a caveat: "the route is map-specific data". So is the reservoir. The rule bans map
  CONSTANTS and human demos; a line computed by the pipeline from the agent's own states is
  neither.

## 6. What this means for the real maps

The benchmark that matters is unitfarmer2's pit and cannonball's wall, and there the same split
applies: discovery is a search problem and following is a dense-reward problem. The archive's
keys already include speed, so "fast at the bottom of the pit" is a distinct cell that a
count-based search will keep and extend; the operator there should be the map's own stuck
policy plus bursts, not a ray executor. The flat arc mechanism already worked on cannonball
with a self line (xSELF 47/102 finishes, CLAUDE.md section 3). What is unproven is the archive
finding the pit exit with a weak operator; that is where the next hard night belongs, and it
costs CPU minutes per attempt, not GPU hours.

## 7. What I would stop doing

* Any further arm on the primlearn hierarchy, on blue200 or elsewhere, until the section 5
  experiment has run. Every knob tried in the last two days moved a zero signal.
* Reporting `plan/ep_prog_start`, `race/eval_progress` or `race/map_pct` on edgeflow as a
  frontier. On these maps they measure how far into the pit a dive gets.
* Reporting greedy-only evals. Report greedy and sampled (or tempered) side by side.
* Building learned models of the simulator (jump models, P_can heads, AlphaZero targets)
  while the exact simulator is forkable at 200k steps/s.
