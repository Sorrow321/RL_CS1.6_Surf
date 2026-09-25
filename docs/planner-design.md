# The planner: why it exists, what it must do, and the proposed redesign

Living design note, started 2026-09-24 from the user's requirements after the
new mazes (ledger 2026-09-23 22:30). Sections 1-3 are the user's and are
settled; section 4 is a PROPOSAL until the user agrees.

## 1. Mission (user, 2026-09-24)

**The planner exists for macro exploration.** On maps where the potential
field does not describe the correct path, the surf agent does not finish,
because it cannot explore: the policy acts 25 times a second, and nothing
in it can make a long-term commitment to one side, go there, and find out
that something new is there. The labyrinth benchmark exposed this even at
its mild rungs.

The split into a planner (macro) and an executor (micro, the policy) exists
to make exploration possible and convenient at the macro level, and to get
an agent that does not know where to go unstuck. The planner decides an
order of magnitude less often (once every 1 to a few seconds), which lets it
commit to a direction and look ahead by seconds. Everything else about it
serves that goal.

Evidence so far: 34 flat PPO cells never left the corridor (2026-09-21);
with the planner, labyrinth 100/200, easy01 (+25M) and medium01 (+100M) are
solved; hard01 is not (0 of 32,768 training episodes ever finished; the
planner is trapped in the region nearest the finish in a straight line).

## 2. Requirements (user, 2026-09-24)

* **Generic**: one planner for the labyrinths and for surf, same constants
  on every map (CLAUDE.md 0b).
* **No top-down view**: surf maps can be multi-storey; a bird's-eye image is
  ill-defined there.
* **No fixed context window**: the hard maze showed what a 2,048 u window
  costs (a branch left behind looks unexplored again).
* **No predefined trajectory set** in the long run: a fixed vocabulary of
  800 u shapes crosses walls 60-97% of the time in 160-200 u corridors.
* **Velocity is part of the state** and must be part of the plan: on surf,
  what is reachable depends on speed.

## 3. The user's formulation (2026-09-24)

* A trajectory is a **sequence of states**: positions in the maze, position
  plus velocity on surf.
* The planner is a distribution Q(tau | s; theta) over trajectories from the
  current state, concentrated on trajectories that lead toward the goal and
  that the executor can do (or nearly - a moving target, so the executor
  keeps getting slightly harder useful plans).
* Only SAMPLING is needed, not densities (implicit models are fine), with a
  temperature raised when the agent is stuck.
* Open questions he named: how long a trajectory is; whether it must reach
  the goal or only start in a good direction; whether its probability should
  reflect the CURRENT executor's ability.

## 4. Proposal (assistant, 2026-09-24) - not decided

**Factor the distribution.**

    Q(tau | s)  proportional to  P_can(tau | s) * exp( U(end of tau) / T )

* **P_can** - can the current executor do tau from s. Learned by supervised
  learning from outcomes: every executed plan is a labelled example (the
  strict judge already measures it), and every trajectory the executor
  actually produced is a valid, physically possible example for free. No
  physics model: the simulator already enforced physics on everything the
  executor did. This answers the third open question: yes, the current
  executor. The "slightly harder" moving target comes from the tails
  (sampling at temperature, noisy rollouts): stretches that work get
  weighted up, the executor trains on them, P_can widens.
* **U** - how useful the END of the segment is. Before the finish has ever
  been reached, the only honest signal is NOVELTY against a GLOBAL archive
  of the states reached so far (3-D position cell + a coarse velocity bin -
  coarse on purpose: 3,072 keys per cell farmed in place, CLAUDE.md 4).
  After the first finish, a learned time-to-goal value. Never the
  straight-line distance: that is what trapped hard01.
* **T** - the user's temperature, raised automatically when the archive
  stops growing.
* This is the known optimum of KL-regularised RL ("control as inference",
  Levine 2018): the best distribution that stays close to what you can do is
  that prior re-weighted by exp(return / T). Training theta is weighted
  maximum likelihood on the executor's own rollouts (reward/advantage-
  weighted regression) - supervised, no adversarial game. Sampling is draw
  K candidates, re-weight, pick (what MPPI does) - no MCMC.

**Answers to the other open questions.**
* Length: a SEGMENT of a few seconds (the commitment horizon), plus the
  score of where it ends; the score (novelty or value) carries the rest of
  the route. Infinite trajectories are neither needed nor learnable.
* Memory lives in the SCORE (the global archive, queried at decision time),
  not in the network's image - that also removes the top-down view.

**v1, concretely.**
* Candidates: segments of the executor's OWN past motion (a few seconds of
  states with velocity, in its own frame), clustered - the vocabulary is
  learned from its motion; on surf it will hold real flight arcs with
  speeds.
* Network: from the executor's egocentric observation (depth render,
  velocity, finish direction), predict P_can per candidate (and the value).
* Choice: P_can x exp(U / T). No PPO on the planner.
* Executor: also reads the plan's velocities (the fix for "flies past the
  goal and turns back"); trained on progress along the plan, with its own
  segments as plans (hindsight), as today.
* First test: labyrinth_hard01, the case that fails today. Then edgeflow,
  then unitfarmer2.

**Decision still open**: whether the planner may also search with forked
simulators (exact, costs compute); v1 does not need it.

## 5. Revision after the user's review (2026-09-24)

* **U is pluggable (user).** Any generic proxy for "how good is where this
  segment ends" is a candidate term to test: Euclidean distance, geodesic
  distance, the potential, novelty, a learned value. Section 4's "never
  straight-line distance" is withdrawn. The hard maze shows only that the
  Euclidean proxy under a GREEDY choice traps; the temperature is what has
  to get it out.
* **The generator must not be the executor's own rollouts (user).** (1) At
  a place it has never been there is nothing to select from; (2) it would
  require the policy to explore before the planner can, when exploring is
  the planner's job; (3) there is no gap between planner and policy: if
  the policy presses W 99.9% of the time at some state, all its own futures
  start with W. The correction: P_can measures the executor's CAPABILITY
  (does it succeed when ASKED to follow tau), not its HABIT (what it does
  unprompted). The executor is plan-conditioned, so what it can follow is
  wider than what it does on its own; P_can is trained on its attempts at
  REQUESTED plans.
* **The generator is the map's geometry (the user's voxel graph).**
  Candidates are paths from the current position through free space,
  expanded on the graph out to the edge of a ball the segment's length -
  available anywhere, including places never visited, and independent of
  the policy.
* **Velocity is not predicted; energy carries it.** Node = (voxel, energy
  E = v^2/2 + g z of the current state). A voxel above the energy ceiling
  (z > E/g, plus a margin for strafe gains) is pruned, and the speed along a
  path follows from v^2 = 2(E - g z). Generic physics, no map constant.
  Energy is necessary, not sufficient (no mid-air turns; efCERT's stitching
  fallacy is energy-feasible): P_can learns the rest, conditioned on the
  current velocity.
* On the maze the walkable graph makes P_can close to 1, so the maze tests
  U and T (exploration); surf tests P_can and the energy prior.

## 6. Planning in jumps (user, 2026-09-24)

**The user's requirements.** (1) Consecutive plans should be continuous - today
each re-plan is an unrelated shape; ideally the planner plans to the end of
the map, like MCTS / AlphaZero, with a proxy (the potential, Euclidean
distance, novelty) at the leaves instead of a trained value. (2) A tree at
the engine's decision rate is impossible: 60 s x 25-30 decisions/s = ~1,800
levels. The planner must be able to SKIP time - predict where the agent will
be ~2 s from now if it starts now and does something.

**Proposal.**
* The executor following one plan segment is an OPTION (Sutton, Precup and
  Singh 1999): a temporally extended action. One tree edge = one segment,
  2-4 s of play. A 60 s map is ~20 edges deep instead of ~1,800.
* The jump from a node to its child can be computed two ways:
  * exact: fork the simulator (`SurfCore.get_states` / `set_state`) and run
    the frozen executor on the segment - exact physics, ~100-300 ticks of
    compute per edge;
  * learned: a JUMP MODEL (an option model, Precup and Sutton 1998):
    (start state, local geometry, segment) -> (P_success, end position, end
    velocity, duration). Supervised on every executed segment; one forward
    pass per edge. This is where the user's "predict the velocity" belongs:
    as the model's prediction of how a segment ENDS, not as part of the
    request. P_can of section 4 is its success head.
* Only the FIRST edge is executed for real (receding horizon), so model
  errors never accumulate in the world, and every executed edge is a new
  training example exactly where the planner goes. The geometry input must be
  map-derived (a 3-D voxel patch around the node's position), because an
  imagined node has no render.
* Leaves: U (pluggable proxy) plus the elapsed time. A learned value trained
  on the search's own results is the optional later step (AlphaZero's reason
  for one: a proxy is myopic past the tree's depth).
* Continuity: keep the tree between decisions (AlphaZero re-roots on the
  child it played; MPC warm-starts from the shifted previous solution), so
  the next plan is the continuation of the previous best branch unless new
  information changes it.
* Jump-model uncertainty (ensemble disagreement) is itself a generic
  novelty signal: "I don't know what happens if I try this" is worth trying.

## 7. One picture (2026-09-24, proposed; supersedes the v1 of section 4)

The user asked for a single picture and a start-simple plan. The pieces of
sections 4-6 fit as follows.

* **A trajectory is a sequence of JUMP POINTS**: full states (position, and on
  surf velocity) every ~2-3 s: s0, s1, ..., sn. The path between two jump
  points is whatever the executor does to get there; it is not the planning
  object. Voxels are an INDEX (for merging duplicates and for novelty), not
  the representation. Same object on the maze and on surf; velocity kept;
  a 60 s map is ~20-30 jumps.
* **The model is a Markov chain over jump points** (the user's HMM-like
  picture): Q(tau) = prod_t pi(s_{t+1} | s_t). Sampling is ancestral, one
  jump at a time, with temperature pi^(1/T).
* **Generation - the children of a node are the DISTINCT places (and speeds)
  the agent can be in 2-3 s.** Get them by probing: try K simple generic
  intents from s_t (e.g. 8 directions), see where each ENDS, and merge the
  ones that end in the same state (position within ~one cell, similar
  velocity). The number of distinct outcomes IS the ambiguity: in a
  corridor or mid-air most probes end in the same state (1-2 children); at
  a junction or a take-off point they split (left / right, earlier /
  later). Merging is what prevents the exponential blow-up.
* **How a probe's end is found** (the jump): the walkable graph on the maze
  (exact, no simulation); a forked simulator running the executor on surf
  (exact); a learned jump model later (section 6), checked against the
  exact ones.
* **The guide is pi** (AlphaZero's policy prior over a node's children).
  Start with pi proportional to exp(U(child) / T), the user's pluggable proxy
  - no learning. Then learn pi from the search's own visit counts
  (AlphaZero), so fewer simulations are needed; pi is the Q(tau | theta) of
  section 3.
* **Search** = MCTS over jump points, U at the leaves, only the first jump
  executed, tree re-rooted after it (continuity).

Where the earlier pieces went: P_can = which probes succeed and where they end
(the jump); exp(U/T) = the prior; the voxel graph = the maze's exact jump;
energy = a pruning check on surf probes and on the learned jump model;
memory / novelty = one choice of U.

**Plan, adding complexity only while it fails:**
1. Maze, no learning in the planner: jump points on the walkable graph,
   probes = distinct reachable places ~3 s away, exact jumps, pi from U,
   MCTS; the executor plHARDa follows the graph path to the chosen jump
   point. Test labyrinth_hard01 with U = Euclidean and U = novelty at a few
   depths (U = geodesic is the sanity check: it must equal the BFS planner).
2. Learn pi from the search (AlphaZero) and a jump model; check both against
   step 1's exact answers.
3. Surf: probes executed in forked simulators with a surf executor; outcomes
   merged by position and velocity; energy check.

## 8. Status after the night of 2026-09-23/24 (ledger for the numbers)

* **Step 1 is built** (`--goal-planner jump`, surfgym/goaljump.py): options =
  distinct places one jump away on a graph, a depth-limited search, U
  pluggable (euclid / novelty / episodic / euclid+episodic), a draw from
  softmax(value / T); the executor gets the chosen jump's path through the
  fan, as before. Nothing learns; results are decided at step 0.
* **labyrinth_hard01 solved** at depth 12 (100% of training episodes from
  the start, greedy 2-3/3, ~51 s vs 45 s ideal). Why it looped at depth
  <= 8: memoryless + deterministic + a horizon shorter than the detour.
* **The surf graph** (`--plan-graph ride`: surfaces within 256 u below or
  one 128 u hop) connects all four edgeflow maps; stage 1 on it trains
  executors that FINISH blue025 (from scratch), blue050 (first ever), 100
  and 200 (warm up the ladder), and the jump planner drives those frozen
  executors to the finish on all four (1,500 u jumps better than 750 u).
* **The plan drawn into the camera** (`--goal-obs fanline`) helps
  modestly at one seed (finishes 28% faster, 84% vs 65% training goals).
* **The limit:** the ride graph does not connect cannonball, celestial,
  unitfarmer2, petrus or utopia - their flights between ramps are longer
  than a hop. Momentum (section 5's energy nodes, or simulator probes / a
  learned jump model - section 6) is the next step, and the place where
  the planner stops being a geometry exercise.
* **Honest scope:** the search on these maps always sees the finish (a
  12-jump horizon covers the routes), so what is shown is planning with a
  map, not yet exploration without one; one seed per arm.

## 9. Candidate for later: planner + executor as ONE policy, ONE reward (user, 2026-09-25)

**The user's idea, verbatim in substance:** treat the planner and the policy as one agent and
train them jointly with a single global reward (Euclidean, geodesic or binary). PPO needs
pi(a | s, theta); let the action be a = {plan, sequence of policy actions}. Then

    pi(a | s, theta) = pi_planner(plan | s, theta) * pi_policy(a_1..a_T | s, theta, plan)
    log pi(a | s)    = log pi_planner(plan | s) + sum_t log pi_policy(a_t | s_t, plan)

and train both factors jointly on the one reward. "The policy performs the plan" stays IMPLICIT:
the executor uses the plan however it pays under the global reward. Status (2026-09-25 18:00):
BUILT as `--plan-joint 1` (merge 4c297d1). The executor trains on the race reward toward the
finish (no arc pay, no --exec-cut). The planner trains on the same reward summed over each
primitive, discounted on the executor's clock, with the cap bootstrapped. Each factor keeps its
own critic; the planner's copy is scaled by 0.01. A shared critic was rejected: the executor's
critic sees the plan, so it is Q(s, plan), and its TD error gives the planner no signal. First
runs are on the 2D mazes (ledger).

Notes for whoever builds it (assistant, 2026-09-25):
* Closest published form: HiPPO (Li, Florensa, Clavera, Abbeel, "Sub-policy adaptation for
  hierarchical RL", ICLR 2020): a manager choosing a latent every p steps plus skills, trained
  jointly with PPO on the environment reward through this factorisation. Related:
  option-critic (Bacon, Harb, Precup 2017) and DAC (Zhang & Whiteson 2019), both end to end on
  the task reward.
* The executor ALREADY treats the plan as a code. The override ablation (ledger 15:08): planner
  8/9, straight / random / frozen 0/9. The executor tracks meaningless curves better than the
  planner's. Joint training makes that explicit, and drops the arc-following reward and its
  farming.
* **Risk 1: collapse.** The executor can learn to ignore the plan (a flat agent with a dead
  input). Then the planner's choice stops mattering and its gradient vanishes. HiPPO holds the
  latent for a random duration to keep the skills plan-dependent. Measure plan-dependence
  directly: the override ablation, run periodically.
* **Risk 2: the reward is the flat agent's.** A Euclidean or geodesic reward is the deceptive
  signal the planner exists to escape (review 2.2: cannonball's fold-back, unitfarmer2's pit).
  The binary reward is honest but sparse. Either way, exploration still has to come from
  somewhere (coverage, returns, the planner's commitment).
* **Minimal arm in this code base:**
  - the executor on the global reward: no arc pay, no `--exec-cut`, its return running across
    re-plans;
  - the planner on the same reward, summed over its primitive (SMDP-discounted);
  - one advantage signal for both factors, or two critics on one reward as a first step;
  - everything else the recipe's, with the no-planner control (`--plan-fixed straight`) beside
    it.

## 10. Continuity: the planner's frame, its memory and when it re-plans (user, 2026-09-26)

After watching `pl1_b050` (plain Euclidean planner reward, squashed-action entropy): the planner's
ideas are reasonable once the agent is on the left surf row, but there is a GAP between planner
and executor. The user's diagnosis: the planner solves a needlessly hard problem because of the
way the plan is handed to it - it plans discontinuously (one curve, then a timer, then an
unrelated curve), in a frame that moves with the full 3D velocity, with no memory of what it
asked for last time. The analogy is the policy's own view control: deltas -> absolute values +
continuous control + the previous yaw / pitch visible to the policy was what made it work
(section 2 of CLAUDE.md, `--view-continuous --view-absolute velocity`). "We need to make the
planning idea as simple as possible."

In the user's order:

1. **The frame (first test, running as `pl2_b050`).** Do not lay the curve on the 3D velocity:
   whether the agent is climbing or falling should not change what the numbers draw. Lay it on
   the velocity's projection onto the horizontal plane (`--prim-frame level`, commit 15b39fe:
   leaves level along the horizontal heading, traced at the horizontal speed, the vertical rates
   bend it from level). The planner's inputs were already in that horizontal heading frame
   (`motion_frame`). Further option to try: an ABSOLUTE frame fixed to the map, not related to
   the player at all.
2. **Re-plan more often**, instead of once per ~2 s primitive.
3. **Memory of the previous plan.** Show the next planner the plan it made last time - its
   numbers, or an image of it, or both - so it can correct "a little bit left, a little bit
   right" instead of drawing an unrelated curve (with absolute numbers this is the view-control
   recipe again). Or condition the next curve on the previous one's end.
4. **When to re-plan: not a fixed timer.** Today the curves are about the same length and are
   re-chosen on a clock, so for the current speed a curve can be too short or too long.
   Candidates: re-plan when the agent is too far from the current plan, or when it has flown
   half of it (overlapping plans, as receding-horizon control does).

Status: 1 (level frame) is the running test; 2-4 and the absolute frame are the backlog, to be
taken in that order unless the user says otherwise.
