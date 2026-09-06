# Literature survey: the MACRO / MICRO split - hierarchy, gates, and search over macro-actions (2026-09-06)

Commissioned against the ledger entry "2026-09-06 01:35 - the user's direction for the formulation": MACRO control
(which ramps, where to take off - a sparse sequence of gates from the map's own geometry, a geodesic field or an
exploration archive, never a human demo) split from MICRO control (a goal-conditioned controller that strafes to the
next gate), with the macro level a SEARCH over gate sequences executed by the controller on real physics rather than a
second learner. Setting: PPO, depth images, no memory, a deterministic C simulator at 131 Hz with native state
restore, ~1M env-steps/s on CPU; an expert-iteration loop (beam over tick-level actions seeded by the policy ->
distillation -> PPO) already yields a searched line at 68.54 s, 0.06 s under the human record, but it only perturbs
the current line and has never found the one-ramp-touch finish the record uses. ASCII only. Numbers are from the cited
paper unless marked "(ours)" = measured in this project's ledger; "NOT FOUND" = looked for, absent from the source.
Companions: `research-litsurvey.md`, `research-plan-goalcond.md`, `research-results.md`.

## 0. Executive summary - the shape to build
1. Macro = a **SEARCH over a gate graph whose edges are executed by a FROZEN goal-conditioned controller on real
   physics with state restore**, not a learned manager: Nachum 2019 measured that most of HRL's benefit is
   *exploration*, and flat agents with temporally extended exploration matched HIRO on AntMaze/AntPush/AntBlock.
2. Closest working precedent in our domain: MCTS at 1 Hz over discrete waypoints driving a learned 50 Hz controller
   (Head-to-Head Autonomous Racing, >90% win rate vs end-to-end MARL).
3. Closest formulation to the sketch is **DC-MCTS** (AND/OR tree over subgoals, edges run by a goal-conditioned
   policy); its dominant failure, "delusional" plans from a wrong value oracle, we delete by running edges instead of
   predicting them. **TTGS** is the case for the frozen half: Dijkstra over dataset states plus an UNCHANGED policy
   took HIQL 1.4% -> 78.6%.
4. **Gates must carry a SPEED, not only a position.** Ours: at the finish-room entry a 0.2% velocity change kills 95%
   of open-loop replays while +-4 u of position is tolerated 70% of the time. `--goals` conditions on a position
   sphere only - the concrete gap.
5. **Gate spacing must sit inside the controller's measured reach.** PRM-RL's chaining law: success =
   (per-edge)^(waypoints), 0.85^6.05 = 37% predicted vs 38% observed. Ours: ~99% at 1-5 s on the agent's own path,
   5-11% at 10-20 s, 2-3% past 20 s. So gates every ~1-2 s.
6. **The executor must be steerable or the search has zero variance.** Ours: forking at the room entry and running the
   CURRENT policy over a +-1,500 u grid moved the exit x by 27 units over 750 starts.
7. Do NOT learn termination (option-critic: 100% per-step termination without a deliberation cost) and do not mine
   options from policy confidence (OptionZero: ~75% of learned options are plain action-repeat - commitment, not
   alternatives).
8. Use the gate plan as a heuristic for the tick-level beam, not a script (RoGuE: 2 collision-free executions
   following the roadmap directly, 100% using it as a cost-to-go wavefront for a search in the true state space).
9. Gates from contact and takeoff events in the agent's own archive, not the geodesic field (ours: mid-route minimum
   at the wall; the BFS believes in free lateral flight).
10. One-day validation is a steerability measurement in the finish room, not a training arm (final section).

# PART A - hierarchy at scale: what was stable and why

## 1. Options / semi-MDPs (Sutton, Precup, Singh 1999) and Option-Critic (Bacon, Harb, Precup 2017)
An option is (initiation set, internal policy, termination); a set of options induces a semi-MDP, and if primitive
one-step options are retained the optimum stays representable - the formal statement of the macro trade. Option-Critic
learns internal policies and terminations from task reward alone: Atari, 8 options, conv 32/64/64 + 512, lr 0.00025,
entropy 0.01, termination regularizer xi = 0.01, beating DQN on Asterix, Seaquest and Zaxxon within 200 episodes. Its
own defect: "the termination gradient tends to shrink options over time... since in theory primitive actions are
sufficient". Harb et al. 2018 quantifies it: without a deliberation cost termination hits "100% very quickly, meaning
each option only lasts a single time-step"; with eta = 0.020, Amidar 880 vs 512, Hero 20,100 vs 2,625.
HERE: keep the semi-MDP framing and the primitives (our tick beam), discard learned termination. We already know a
macro decision's granularity - a contact, a takeoff, a room entry - and the BSP gives it free; learning it reproduces
the collapse and we then hand-tune a stickiness constant to undo it.

## 2. FeUdal Networks (Vezhnevets et al. 2017)
The Manager emits a goal that is a DIRECTION in latent space, "because it is more feasible for the Worker to be able
to reliably cause directional shifts in the latent state than it is to assume that the Worker can take us to
(potentially) arbitrary new absolute locations". Horizon c = 10; transition policy gradient A^M * grad d_cos(s_{t+c} -
s_t, g_t), crediting the Manager for the direction the world moved, never for the Worker's actions; dilated LSTM
radius 10; Worker intrinsic reward = mean cosine similarity of the last c state deltas to the goals issued. Montezuma:
first room (400) under 200 epochs vs LSTM-A3C's 300+, ~2600 where the LSTM stagnates at 400; Asterix ~20x
Option-Critic; Frostbite ~7x LSTM.
HERE: the directional argument transfers exactly and argues for gates as ego-frame offsets plus a target speed - half
of which `--goal-obs fan` already is. FuN reports no instability because it never backpropagates Worker into Manager:
the same instinct as freezing the executor.

## 3. HIRO (Nachum, Gu, Lee, Levine 2018) - the non-stationarity fix
Goals are raw state-space directional deltas; the high level acts every c = 10 steps with a fixed goal transition
between; low-level reward -||s_t + g_t - s_{t+1}||_2. The off-policy correction relabels a stored high-level
transition with the g~ maximizing log mu_lo(a_{t:t+c-1}|s, g~) over 10 candidates: 8 from a Gaussian centred on
s_{t+c} - s_t, plus g_t, plus s_{t+c} - s_t. At 10M steps: Ant Gather 3.02 +- 1.49 (FuN 0.01, SNN4HRL 1.92, VIME
1.42), Ant Maze 0.99 (others 0.0), Ant Push 0.92 (<= 0.02), Ant Fall 0.66 (0.0). Quoted: "Since the lower-level policy
is changing underneath the higher-level policy, a sample observed for a certain high-level action in the past may not
yield the same low-level behavior in the future... a non-stationary problem for the higher-level policy."
HERE: HIRO's central engineering effort exists only because both levels learn. If the macro level re-executes each
edge on real physics at plan time the correction is unnecessary by construction - the edge cost is measured, not
remembered. Strongest single argument for search-over-gates in this project.

## 4. HAC (Levy, Konidaris, Platt, Saenko 2019)
Three relabelings. Hindsight ACTION transitions replace the proposed subgoal with the state actually reached, so the
upper level trains "as if the optimal lower level policy hierarchy had been used". Hindsight GOAL transitions are HER
per level. Subgoal TESTING runs the lower level noise-free a fraction lambda of the time and penalizes an unreached
subgoal by -H. H = 10 per level for 3 levels, H in [20,30] for 2; 3-level > 2-level > flat DDPG+HER on inverted
pendulum, UR5 reacher and ant four-rooms (figures, not a table). Quoted: "the transition function at each level above
ground level will continue to change as long as the policies below that level continue to be updated".
HERE: subgoal testing is a feasibility filter on the gate proposal distribution, and in a search it is free - an edge
the controller fails scores dead and the beam drops it. Hindsight action relabeling maps to recording the gate the
controller ACTUALLY hit on every executed edge and adding it to the graph: that is how the graph grows without a
generator.

## 5. h-DQN (Kulkarni, Narasimhan, Saeedi, Tenenbaum 2016)
A meta-controller picks a subgoal from hand-specified entities (key, four doors, ladders, the player) from a custom
object detector; the controller gets intrinsic reward for reaching it. Montezuma ~+400/episode after ~4.3M steps (2.3M
pretrain + 2M joint) vs DQN 0 and Gorila DQN 4.16. Stated limitation: "several key ingredients are missing such as
automatic discovery of objects".
HERE: the honest statement that hierarchy works once someone supplies the discrete vocabulary and stalls without one.
Our "key and doors" - ramp faces, takeoff points, room entries - are already in the BSP and partly extracted
(`surfmask` rideability, `beam_tas`'s contact detector, `explore_phase1.py` cells). We are in the lucky half, which
argues for geometric gates.

## 6. Director (Hafner, Lee, Fischer, Abbeel 2022)
Both levels train entirely on imagined rollouts inside a learned world model. A goal autoencoder maps states to
discrete categorical codes; the manager's action is a code, rechosen every K = 8 steps; the worker maximizes a
max-cosine similarity between its state features and the decoded goal. Two manager critics, extrinsic and exploration
r_expl = log p(z|s) - log p(z). Solves Egocentric Ant Maze at all four sizes and is the only method to reach the goal
reliably in the larger ones; also Visual Pin Pad, DMLab, Atari, humanoid walking without demos. Ablation: remove the
goal autoencoder and goals become uninterpretable and the agent fails.
HERE: the best-behaved *learned* hierarchy here, and the reason is instructive - both levels see one frozen shared
model, so neither chases the other through the real environment. We have something stronger, the exact simulator with
restore. Keep the goal-autoencoder lesson: the high level's action space must be a small discrete set of *realizable*
states, i.e. quantize gates into archive contact cells rather than proposing arbitrary points in R^3.

## 7. Why Does Hierarchy (Sometimes) Work So Well in RL? (Nachum et al. 2019)
Isolates three claimed benefits: temporally extended training, semantic high-level action spaces, temporally extended
exploration. "Shadow" flat agents trained on HIRO's own collected experience with multi-step rewards (c_rew = 3)
matched HRL on AntMaze, AntPush, AntBlock, AntBlockMaze. Two flat substitutes - "Explore & Exploit" (a
reward-maximizing and a random-goal agent switched every c_switch steps at 0.2/0.8) and a "Switching Ensemble" of five
agents - matched HIRO on most benchmarks. Quoted: "the success of HRL on these tasks is largely due to better
exploration... as opposed to discovering high-level representations which make policy and value function training
easier."
HERE: the paper that decides the shape. If the payoff is temporally extended exploration, a simulator with state
restore buys it directly and far more cheaply - which is what `explore_phase1.py` and the respawn reservoir already
are. It also predicts a learned-manager arm here: no better than a flat agent with the same start-state machinery,
which is what the frontier arms showed (ours: xsG3h..xsG3l each advanced once, then stalled).

# PART B - goal-conditioned controllers with a planner on top

## 8. Hierarchical Control for Head-to-Head Autonomous Racing (Thakkar et al. 2022, 2202.12861)
The high level discretizes racing (velocity bands, tyre wear, position as lane ID + checkpoint index) and runs MCTS at
1 Hz over ~8 checkpoints (~120 m), emitting "a series of waypoints in the form of target lane IDs... and the target
velocities at each of the checkpoints". Two low levels consume those at 50 Hz: a PPO-trained MARL policy (waypoints +
opponent state + lidar) or an LQNG Nash solver. MCTS-RL won "over 90% of head-to-head races" with the second-best
safety score, beating end-to-end MARL, fixed-trajectory RL and LQNG. Argument: encoding discrete rules inside a
continuous optimization is intractable in real time.
HERE: our proposed architecture already working in a racing game, and *target velocity per waypoint* is exactly the
field our gates lack. Note the 50:1 decision-rate ratio: at 131 Hz with `act_every 4` (30.7 ms/decision) the analogous
gate layer runs at ~1-2 Hz, which also matches our controller's measured reliable reach.

## 9. ANYmal Parkour (Hoeller, Rudin, Sako, Hutter 2023, 2306.14874)
Five skills (walk, jump, climb-up, climb-down, crouch) trained individually, then FROZEN. The navigation policy emits
a hybrid action at 5 Hz: a categorical over the 5 skill indices plus a Gaussian over local position r*, heading psi*
and a **TIME command t***; skills run at 50 Hz. The skills' position-tracking reward is **sparse - activated only on
the last timestep** - so a skill may take a safer path inside its budget rather than greedily reduce distance. Gaps
and boxes to 1 m, crouch 0.4 m, up to 2 m/s; learned vs hand-coded navigation over ~1000 rollouts: 98.2 vs 95.3, 96.3
vs 60.9, 97.6 vs 75.3 percent. Perception reconstructs occluded geometry - and hallucinates (invents a staircase
behind boxes). Eight networks in total.
HERE: the only parkour system whose high level can be swapped for a planner without retraining, because the skills are
frozen and the interface is an explicit timed pose. (skill, local pose, time budget) is a better gate signature than a
position ball, and the sparse-at-deadline reward is the published form of our finding that arc/geodesic shaping
charges the agent for the manoeuvre it must make.

## 10. Extreme Parkour (Cheng, Shi, Agarwal, Pathak 2023, 2309.14341)
The direction reward is `min(<v, d_hat_w>, v_cmd)` with d_hat_w the unit vector to the next waypoint - projected
velocity capped at commanded speed - in the WORLD frame, because the base frame lets the robot "exploit the reward and
learn the unintended behavior of turning around the obstacle". Two phases: PPO with privileged scandots, then DAgger
into a depth ConvNet+GRU with a "Mixture of Teacher-Student" feeding the student's own predicted heading only while
|theta_pred - d_hat| < 0.6 rad, since "directly using predicted heading could result in catastrophic distribution
drift". Jumps 0.5 m, gaps 0.8 m, ramps 37 deg, under 20 h on one 3090. Removing the inner-product term collapses Step
terrain to 0.14 +- 0.00 from 0.99.
HERE: `min(<v, d_hat>, v_cmd)` is the cheapest gate-following reward that pays direction and speed together without
paying for instantaneous distance reduction - exactly what our geodesic potential does wrong at the wall - and the cap
is how a speed target enters a dense reward without a second term. The teacher-student gate is the recipe for
distilling a searched gate plan without the drift behind our 0.6 s policy-to-line gap.

## 11. Robot Parkour Learning (Zhuang et al. 2023) and WASABI (Li et al. 2022)
Zhuang makes obstacles penetrable in pre-training and *penalizes* penetration rather than preventing it,
`-sum_p (a5*1[p] + a6*d(p)) * v_x` (the v_x factor stops the agent farming the penalty by standing still), on an
automatic curriculum (s += 0.05 past a threshold); then obstacles go solid and the policy is fine-tuned. Five skills,
then DAgger into one 48x64-depth policy: climb 86%, leap 80%, crawl 100%, tilt 73%. **Without soft dynamics the agent
"cannot make any learning progress on climbing and leaping"**; blind = 0%; an MLP without the GRU = 0% climb, 1% leap,
because the obstacle leaves the FOV before commitment. WASABI matches base-only, dynamically infeasible mocap demos
with a Wasserstein GAN-GP discriminator, specifically because an LSGAN critic saturates when the demo is beyond reach.
HERE: the soft-then-hard curriculum is the general recipe for "a wall gives zero gradient" - our finish-room ramp is
that, and a temporarily softened contact would give a dense signal where open-loop probing gives 0/18,900 (ours). The
memory ablation is a direct warning: we are memoryless and depth-only, so a gate below or behind the camera is
invisible at the moment of commitment - keep `--goal-views 4`.

## 12. Graph search over states with a frozen policy: SoRB, SPTM, SGM, TTGS, LEAP
SoRB (Eysenbach et al. 2019) trains goal-conditioned with r = -1 per step so V(s,g) = -d_shortest_path; vertices are
replay-buffer states, edges longer than MaxDist (= 3) are deleted, Dijkstra gives a waypoint to the unchanged flat
policy. Two devices are load-bearing: distributional RL with a catch-all final bin ("the true value for a state and
goal that are unreachable is -inf, which cannot be represented") and independently weighted Q ensembles - sharing conv
layers produced "highly-correlated distant predictions that exhibited the 'wormhole' problem". SUNCG held-out houses
~80% at 10-step goals vs <20% flat, ~40% vs ~0% at 20 steps; 1,000 nodes, cutting to 100 gave "no discernible drop";
hindsight relabeling "hurt performance of the non-search policy". SPTM (Savinov et al. 2018) builds the graph from ~5
min of random exploration with a retrieval net (pairs <= 20 steps apart), a median filter over +-10 frames before an
edge is believed, and picks the furthest vertex on the Dijkstra path still reachable with confidence > 0.95: 100% on
all seven test mazes vs 33.9% best baseline; 85/55/50% without shortcuts. SGM (Emmons et al. 2020) merges nodes only
under **two-way consistency** and adds an **edge-cleanup pass deleting any edge the policy fails**: ViZDoom 20,100 ->
2,087 nodes, PointEnv SoRB 28.0% -> 100%, ViZDoom 39.3% -> 60.7%, SafetyGym 68.2% -> 92.9% (cleanup ablation 38.2% ->
92.9% vs one-way 31.2% -> 64.6%); residual 26% at long range. TTGS (2510.07257) samples 4,000 dataset states, maps
values to distances, uses a soft trust region (beyond tau, weight x*1000^(x/tau)) and hands the FROZEN policy the
farthest waypoint inside a step budget: HIQL 0% -> 80.9% pointmaze-giant-stitch, 4.4% -> 78.1% humanoidmaze-giant,
1.4% -> 78.6% antmaze-giant, GCIQL 0% -> 98.0%; it helps when "(i) the task requires long-horizon stitching, (ii) the
base policy is locally reliable, (iii) the dataset covers intermediate states", gains 0-2 points on manipulation, and
its "largest hop ratio" rho is 0.04-0.22 where it helps vs 0.46-0.62 where data is missing. LEAP (Nasiriany et al.
2019) is the continuous version - K latent subgoals scored by the value function's feasibility per hop (K, optimizer,
per-task numbers NOT FOUND).
HERE: TTGS is the strongest evidence for our exact split - add search over subgoals to a frozen goal-conditioned
policy, change nothing else, get an order-of-magnitude result on long-horizon stitching
- and rho transfers directly as a pre-flight diagnostic on the largest gate hop the archive forces on us. Copy
verbatim: a MaxDist cap expressed in *seconds of flight*, temporal-consistency filtering before an edge is believed,
and edge cleanup (nearly free for us - we can just run the edge). Treat the wormhole warning as ours: the geodesic BFS
believes the player can fly laterally across a void, and our beam already shows the pathology (`--score dv` ranked a
lineage that traded 200 u/s for position below the untreated line).

## 13. Roadmaps validated by the controller: PRM-RL, RoGuE, RL-RRT
PRM-RL (Faust et al. 2018) accepts a roadmap edge **only if the trained RL policy traverses it in >= 85% of 20 Monte
Carlo attempts** (max edge 10 m): indoor navigation 43-56% vs 8-18% for straight-line PRM in buildings 12-52x larger
than training; longest run 216 m / 400 s / 45 waypoints. The law that binds us: success decays geometrically in
waypoint count, 0.85^6.05 = 37% predicted vs 38% observed; named failure, "the RL agent does not require the robot to
come to rest at the goal region, therefore the robot experiences some inertia when the waypoint is switched". RoGuE
(2310.03239) trains a goal-conditioned SAC+HER controller in an obstacle-free world, validates edges by rolling it
into a ball of eps = 0.5, records that it does not land on the node ("gaps"), and then does NOT execute the roadmap: a
backward wavefront over it is a cost-to-go heuristic for a tree planner in the true state space. Quadrotor 100% vs
SAC+HER 5% and hierarchical SAC+HER 13%; following the roadmap directly, "only 2 such executions returned
collision-free trajectories". RL-RRT amortizes the same test into a learned obstacle-aware time-to-reach estimator
(74-80% accuracy).
HERE: the 85%-of-20-rollouts test is the cheapest correct definition of a gate edge for us; the chaining law bounds
how many gates a plan may contain; and RoGuE's "heuristic, never a script" is what reconciles a gate plan with the
tick beam we already have - the gate sequence biases the beam's population rather than replacing it.

## 14. Go-Explore (Ecoffet, Huizinga, Lehman, Stanley, Clune, Nature 2021)
Archive of cells; return by restoring simulator state; explore with ~100 random steps at 95% action-repeat; selection
weight W = 1/sqrt(C_seen + 1); then a robustify phase converting the brittle trajectory into a policy with the
Salimans-Chen backward algorithm. Montezuma 43,791 without domain knowledge and 1,731,645 with it (human WR 1.2M);
Pitfall 6,954 / 102,571; 2B exploration frames plus 10B robustification, and only 11 of 57 games were robustified
because it is expensive. Named pathologies: detachment, derailment, sensitivity to cell resolution. Within a cell it
keeps only the best trajectory.
HERE: we already have return-by-restore, and `explore_phase1.py` is a Go-Explore. Robustify is not optional - the
68.54 s line is exactly the brittle artifact it describes (ours: one decision delayed by one tick anywhere kills 97%
of replays), so the 0.6 s policy-to-line gap is the price of robustness, not a distillation defect. And "one best
trajectory per cell" is why the archive will not enumerate route topologies by itself.

# PART C - search over macro-actions in games

## 15. MuZero (1911.08265), Sampled MuZero (2104.06303), Gumbel MuZero, OptionZero (2502.16634)
MuZero plans with a learned latent model (800 sims/move on board games, 50 on Atari; Atari-57 median 204.1% / mean
4999.2% at 20B frames). Its ONLY temporal abstraction is standard frameskip: "50 simulations every fourth time-step,
and then repeating the chosen action four times"; Atari plateaus near 100 sims, attributed to model error. Sampled
MuZero expands only K sampled actions per node and corrects with the empirical sample distribution pi_beta_hat =
(beta_hat/beta) * pi (K = 20 for DMC including a 56-D humanoid; K = 50 of 362 Go actions "closely approaches" the full
baseline); using plain pi "gives unstable results because of the f/beta term". Gumbel MuZero gives a guaranteed policy
improvement at very small simulation counts via Gumbel-Top-k plus Sequential Halving. OptionZero adds an option head
predicting the greedy continuation's cumulative probability (dominant option = longest prefix with prod P > 0.5) and a
multi-step dynamics network so one edge jumps l steps: 26 Atari at 50 sims, MuZero mean 922.72% / median 328.40% vs l
= 3 at 1054.30% / 391.69% (the abstract's "131.58% improvement" is a percentage-POINT delta, ~+14.3% relative),
improving 20/26 games; in play 37.62% of executed actions were options, mean length 1.69 steps, and **~75% of learned
options are a repeated primitive action**.
HERE: MuZero argues for a learned model we do not need. Keep Sampled MuZero's correction (weight sampled gate
proposals by the empirical draw, not the proposal density) and Gumbel's budget allocation - the right primitive when
each gate expansion costs seconds of physics and we can afford tens per node. OptionZero is the clearest warning in
the survey: an option vocabulary mined from the policy's own confidence finds *commitment*, not *alternatives*. Our
tick beam has exactly that pathology, and a learned-from-the-policy abstraction will not cure it - gates must come
from the map.

## 16. Subgoal MCTS (Gabor et al., IJCAI 2019) and Automatic Goal Discovery (CoG 2021)
Change only the EXPANSION step: to expand a node, roll out random primitive actions until a subgoal predicate fires
and make that sequence one macro edge - so branching depends on the number of *reachable subgoals*, not on the action
space. Per-decision compute S-MCTS 0.0045 s, hierarchical 0.0021 s, flat MCTS 0.0724 s: 16-34x cheaper at competitive
return on gridworld and Tetris. Predicates are cheap and geometric ("obstacles on opposing sides next to the agent" =
a doorway). The 2021 follow-up replaces the hand predicate with a quality-diversity search at each leaf (novelty =
distance to the k nearest existing subgoals, score = 0.5 novelty + 0.5 reward, descriptor = position), beating both
vanilla and hand-predicate MCTS on PTSP over 10 maps - but "struggled on constrained environments, particularly maps
with narrow corridors".
HERE: the cheapest implementation of the user's sketch and the one needing least new machinery - the gate predicate is
`core.set_goal_box` plus a contact test, and expansion is a rollout until it fires. The QD variant is how a gate set
could be discovered rather than enumerated; its documented weakness (corridors) is a caution for a tight finish room.

## 17. DC-MCTS - Divide-and-Conquer MCTS (2004.11410)
An AND/OR tree over SUBGOALS instead of a sequential action tree: an OR node is a task (s, s''), an AND node splits it
by inserting an intermediate s' and recursing on both halves. It assumes a goal-conditioned policy plus a value oracle
v_pi(s, s'') giving that policy's success probability on the sub-task - i.e. "execute this edge with the controller" -
with pUCT traversal and a learned subgoal proposal prior trained by behavioural cloning on search results plus HER.
21x21 procedural mazes and MuJoCo quadruped mazes at 200 low-level oracle calls per maze; beat sequential MCTS at
every budget on the hard MuJoCo mazes. Stated failures: it needs a *universal* value oracle, approximation error
yields "delusional" plans, the subgoal action space is the whole state space so branching is huge, and if the
low-level policy cannot reach a state no planning fixes it.
HERE: the published version of the user's formulation with its failure list pre-written. Two of those we delete by
executing edges instead of predicting them; the others - huge branching and an unreachable low level - are what a
geometric gate vocabulary and the final section's steerability test are for. The divide-and-conquer ordering (insert a
midpoint, recurse) is also the right shape for refining an existing line, which is half the user's requirement.

## 18. MAGIC - Macro-Action Generator-Critic (Lee, Cai, Hsu, RSS 2021)
Learn the macro-action SET offline, optimized for the planner that consumes it: a generator maps (belief, context) to
a candidate set, Macro-DESPOT plans with it, and a critic net is a differentiable surrogate for the planner's value so
the generator is trained by gradient ascent through it. Macro-actions are **geometric and hand-parameterized in form,
learned in value**: 2D cubic Bezier curves, 3 control points = 6 reals, 8 candidates, 8 steps. Total reward at 0.1 s
per step: Light-Dark 54.1 vs handcrafted macros -30.9 vs primitive DESPOT -96.1; Crowd-Driving 58.6 vs 13.1 vs 14.8;
Puck-Push 87.9 vs 34.0 vs 53.0. Degrades at both length 2 and length 8.
HERE: the demonstration that a *parameterized geometric* macro beats both a hand-picked set and primitive planning by
a wide margin. Our gate is the same object with different parameters - (contact face, entry speed, entry heading)
instead of Bezier control points - and MAGIC says the parameterization can later be tuned by the planner's own value
rather than guessed.

## 19. Expert Iteration (Anthony, Tian, Barber 2017)
Self-play generates states, MCTS computes an imitation target (normalised visit counts), a new apprentice is trained
on the pairs and plugged back as a UCT prior with w_a = 100. Hex 9x9, 10,000 sims/move, ~7.4M positions distributed:
75.3% vs MoHex at 10K iterations, 59.3% at 100K, 55.6% at 4 s/move; the apprentice alone beat its own expert 87/162.
The two load-bearing sentences: "The learning rate of EXIT is controlled by two factors: the size of the performance
gap between the apprentice policy and the improved expert, and how close the performance of the new apprentice is to
the performance of the expert it learns from"; and "By employing search, we can find strong move sequences potentially
far away from the apprentice policy".
HERE: this indicts our loop directly. Its learning rate IS the expert-apprentice gap, and a tick beam seeded by the
policy's own proposals has a gap that shrinks to zero as the policy becomes self-consistent. Ours confirms it:
identical recipes finished 0.32 s apart on the line over 16 rounds with line improvements a rare event, and forcing
entropy into the proposals (eps 0.15/0.30) gave ZERO finishes in eight runs. A gate-level search restores the gap
without noise.

## 20. PTSP macro-action MCTS (Perez et al., IEEE TCIAIG 6(1), 2014)
A ship in a 2D maze under physics, 40 ms per decision. A physics-aware TSP solver fixes the waypoint ORDER once; MCTS
then steers over macro-actions defined as "execute one of the six primitive actions for T time steps", scored by
waypoints collected, progress to the current target, and speed. T = 15 was best and won both 2012 PTSP competitions;
the search space over a 2000-step run is ~10^103 at T = 15 vs ~10^1556 at T = 1. They tried variable-length
macro-actions and **rejected them**: "different actions took different lengths of time to execute" made the search
systematically favour longer macros. Named failure: the myopic evaluator makes the ship "wait near the waypoint, or
spiral around it... but not actually passing through it".
HERE: the most transferable operational warning. Gate edges have different durations by construction, so a gate search
must score by TIME, or it will prefer long edges for the wrong reason. The two-tier structure - fix the gate ORDER
with a cheap global solver, then search only the execution - is also a cheaper first version than searching both
jointly.

## 21. TMInterface bruteforce and speedrun routing (arXiv 2106.01182)
TMInterface is stochastic local search on a base replay with savestate rewind (`bf_modify_prob = 0.1`, bounded
steering deltas), justified because exhaustive steering over 4 ticks at 1000 attempts/s would take "about 9.353
billion years". Sectioning is standard and explicit: `bf_target = checkpoint` with `bf_target_cp`, input modification
restricted to a time range, and the evaluation and modification windows need not overlap; practitioners keep windows
under ~6 s. Its documented failures are ours verbatim: local optima ("we can't progress because of something
specific... that would result in a short-term worse run") and the butterfly effect ("changing earlier inputs will have
an increasingly significant effect on the rest of the run"). The routing survey names the split we want - operational
routing (inputs at frames) vs strategic routing (traversal between events) - formalises the latter as a weighted
game-event digraph, and reports that in every documented instance the route graph was built MANUALLY.
HERE: our beam is TMInterface's algorithm and inherits its documented ceiling; the community never fixed it with a
better mutation operator, only with shorter windows. The routing survey is the literature's name for the user's
proposal, and its honest finding - route graphs are hand-built - is why deriving gates from BSP geometry is the
differentiator here.

## 22. Homotopy-class path planning (Bhattacharya, Kumar, Likhachev 2010/2012; 2311.00853)
Compute an h-signature for a path (2D: a Cauchy/residue integral of an obstacle-marker function along it; 3D: a
Biot-Savart analogue), then search an augmented graph whose nodes are (position, h-signature). A*/Dijkstra on it
separates topologically distinct paths by construction, and the K best paths in K distinct classes come from
repeatedly blocking classes already found: "Two paths share the same H-signature if and only if they belong to the
same topology". Cost grows with obstacle density. A 2023 alternative drops signatures via locally shortest paths on a
tangent graph and reports 320 topologically distinct paths in 100 ms where signature methods take seconds.
HERE: the only literature that formalises "one ramp touch instead of two" as a first-class object. The cheap version
for us is not an integral: tag each search node with the **sequence of contact-face ids** it has touched and forbid
merging or pruning plans with different tags. That alone would stop the beam culling a one-touch lineage against a
two-touch leader, which our G0-G5 branch-grid runs measured - the grid took up to 120 of 128 elite slots for four to
six generations, was then culled, and no grid lineage ever finished.

# PART D - gates from geometry, goal supervision, reachability curricula

## 23. Hindsight Experience Replay (Andrychowicz et al. 2017)
Store each transition twice, once with the original goal and once with a substitute from the same episode, recomputing
r = -[f_g(s) = 0]. The "future" strategy - "k random states which come from the same episode... and were observed
after it" - with k = 4 or 8 is best and "the only strategy which is able to solve the sliding task almost perfectly".
Bit-flipping: DQN alone solves n <= 13, DQN+HER up to n = 50. The ablation that matters (Sec 4.4), over shaped rewards
lambda|g - s_obj|^p - |g - s'_obj|^p: "Surprisingly neither DDPG, nor DDPG+HER was able to successfully solve any of
the tasks with any of these reward functions", for two given reasons - "a huge discrepancy between what we optimize...
and the success condition", and "shaped rewards penalize for inappropriate behaviour... which may hinder exploration".
HERE: 4.4 is the published statement of our cannonball barrier - the shaping field's minimum is not the success
condition, so the shaping optimum is not the task optimum, and the final descent is charged -4.24 for a manoeuvre the
task requires. It argues for a gate reward that is sparse-at-the-gate plus a time penalty (`--goal-reward sparse`)
over a distance potential, and for relabeling the goal *distribution* (which the reservoir already does), since
on-policy PPO cannot replay transitions.

## 24. GCSL (Ghosh et al. 2019) and the skill-prior family (SPiRL, SkiLD, PlanGAN)
GCSL relabels every trajectory as an optimal demonstration of the goal it reached and behavior-clones: D = {(s_t, a_t,
g = s_{t+h}, h)}, O(T^2) points per trajectory, no value function; Theorem 3.1 makes it a lower bound, J >= J_GCSL -
4T(T-1)alpha^2 + C. Comparable to TD3-HER on 2D navigation, better by a large margin on pushing and the claw, much
less hyperparameter-sensitive; limits: it explores only through policy stochasticity and optimizes reaching the FINAL
state rather than the shortest path, so it "can learn round-about trajectories". SPiRL defines a skill as H = 10
primitive actions in a 10-d latent with a learned prior, the high level acting once every H steps under SAC with
entropy replaced by -alpha KL(pi || prior): it solves a maze 4x larger than training where SAC, BC+SAC and
skills-without-prior all fail, on 85,000 offline trajectories. SkiLD adds a demonstration-support discriminator and
switches regularizer by D(s); on a misaligned task it held ~0.8 where SPiRL collapsed to ~0.2. PlanGAN plans with an
ensemble of conditional Wasserstein GANs, "around 4-8 times more sample efficient" than HER on four tasks.
HERE: GCSL is the cheapest way to turn our non-finishing episodes into gate supervision and composes with `--bc-file`;
its quadratic-in-T bound is another argument for short gate horizons. SPiRL is the learned-vocabulary alternative to
geometric gates and its honest cost is tens of thousands of offline trajectories. SkiLD's support discriminator is
what we would want if a gate plan leads the controller off its training manifold - the one-touch case exactly. PlanGAN
is a negative here (a learned model where we have the true one), but its ensemble-agreement scoring is right in
another form: score a gate edge under a perturbation ensemble, not one replay.

## 25. What the three superhuman racers actually feed as "gates"
None feeds a distance scalar; all feed raw ego-frame geometry. Linesight (Trackmania) stores a reference line at
`distance_between_checkpoints = 0.5` m from a human replay ("the authors usually drive along the centerline") and
feeds `n_zone_centers_in_inputs = 40` points taken `one_every_n_zone_centers_in_inputs = 20` - 40 points at 10 m
spacing, 400 m of lookahead, 3 x 40 = 120 floats rotated into the car frame, extrapolated past the map ends. Its stall
detector is `cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2000` - an INDEX-advance test, not a
distance-epsilon test. Swift (Nature 2023) feeds only the NEXT gate as "the relative position of the four gate corners
with respect to the vehicle, resulting in a vector in R^12", inside a 31-d observation; reward = progress to the gate
centre + a perception term keeping the optical axis on it - 5.0 for a crash; PPO, 100 agents, 1e8 interactions in 50
minutes on one workstation (lambda coefficients NOT FOUND). GT Sophy feeds 60 equally spaced 3D points per track edge
and centre line, spanning "approximately the next 6 seconds of travel" as a function of current velocity.
HERE: three actionable points. Swift's four-corner gate in R^12 is a *pose*, not a sphere, and is the right shape for
"touch this ramp face" - our sphere loses the surface normal that separates a ramp catch from a wall hit. Sophy's
lookahead is in SECONDS and stretches with speed, which our fan does not do and which matters at 3,800 u/s.
Linesight's index-advance stall rule is strictly better than our per-call `stall_eps`, which we know misfires at
`act_every 1`.

## 26. Reachability curricula: Florensa, GoalGAN, Skew-Fit / MEGA, BaRC, kinodynamic RRT*
Florensa grows starts backward from the goal by Brownian rollouts (T_B = 50, Sigma = I, collect M = 10,000, subsample
N_new = 200 and append N_old = 100 old starts - the stated 1/3, against catastrophic forgetting), keeping only "good
starts" with 0.1 < R < 0.9: point-mass maze ~95% vs ~40% for uniform sampling. GoalGAN replaces this with an LSGAN
over goals labelled by the GOID band (R_min = 0.1, R_max = 0.9, claimed robust over (0,0.25) and (0.75,1)), two thirds
of goals from the GAN and one third uniform from the buffer: Free Ant coverage ~0.86 vs ~0.4, Maze Ant ~0.71 vs ~0.3,
holding ~0.7 as goal dimension grows where uniform collapses to ~0.1 at N = 6. Skew-Fit reweights achieved goals by
q(s)^alpha (alpha = -1 typical, -2.5 on Ant) and provably converges to uniform over reachable states; MEGA instead
picks the **minimum-density achieved goal** (KDE bandwidth 0.1, 100 candidates, filtered by a Q cutoff), reaching 90%
on PointMaze in ~2,000 episodes where PPO+SR needed ~7.5M steps. **BaRC** is Florensa with the Brownian expansion
replaced by a Hamilton-Jacobi backward reachable set, because random actions from a medium-success state "can break
down for highly dynamic or unstable systems" and for underactuated systems with drift the expansion "may never train
from starts that belong to the true initial state distribution": 5D car >95% in under 30 iterations vs 20-40% for the
Florensa-style curriculum, and on a 6D planar quadrotor BaRC learns in 4-8 iterations while the random curriculum
"never learns any non-colliding behavior". Kinodynamic RRT* makes the same point classically: "straight-line
connections between pairs of states are typically not valid trajectories due to the system's differential
constraints".
HERE: we already have most of the first half (`--goal-curriculum` is the 10-90% band, `--goal-frontier` the advance,
the reservoir the achieved-goal buffer) and it is the half that stalled. New information: MEGA's argmin-density
frontier rule, strictly better than uniform-over-bins - which our own reading of xsG5j blamed - and BaRC, the
published statement of why a position-only gate is under-specified in a momentum-dominated task and why the geodesic
BFS (voxel adjacency, not dynamics) cannot order gates. We cannot afford HJ PDEs and do not need to: our reachable set
is measurable by simulation, and PRM-RL's 20-rollout/85% test is BaRC's idea paid for empirically.

# Risks and what is known to fail

1. **Two learners chasing each other.** HIRO and HAC both state it; HIRO's off-policy correction and HAC's three
   relabelings exist only to patch it. Avoided if the macro level searches.
2. **Learned termination collapses to primitives** (option-critic: 100% per-step without a deliberation cost). Do not
   learn where a gate ends.
3. **Options mined from policy confidence are action-repeat** (OptionZero, ~75%): commitment, not alternatives - our
   exact current failure.
4. **A non-steerable executor gives a zero-variance search.** Ours: 27 u of exit spread over a +-1,500 u grid, 750
   forks. Measure steerability BEFORE building a planner.
5. **Learned edge distances lie, worst at long range** (SoRB wormholes, SGM's residual 26%, SPTM aliasing). Ours: the
   geodesic BFS believes in free lateral flight.
6. **Chaining decay**: (per-edge success)^(waypoints); PRM-RL 0.85^6.05 = 37% vs 38% observed. With our 5-10 s goal
   success of 8-14% a 6-gate plan is dead on arrival; at the 1-5 s ~99% band it is fine. Gate spacing is not a free
   parameter.
7. **Position-only gates are under-specified here.** Ours: 0.2% velocity change at the room entry kills 95% of replays
   while +-4 u of position is tolerated 70%. Gates need a speed band and preferably a face normal (Swift's R^12 pose,
   not a sphere).
8. **Executing the plan literally is the wrong use of it.** RoGuE: 2 collision-free executions following the roadmap
   vs 100% using it as a heuristic. Ours: open-loop macros left the bowl 0 times in 18,900.
9. **Variable-duration edges bias the search** (PTSP rejected variable-length macros). Score gate edges in TIME.
10. **The expert-apprentice gap is the loop's learning rate** (ExIt). Ours: identical recipes 0.32 s apart on the line
    over 16 rounds; eps-uniform proposals at 0.15/0.30 gave zero finishes in eight runs. Noise does not restore the
    gap; a different action space might.
11. **Shaped rewards can be anti-correlated with the success condition** (HER 4.4: no task solved under any shaped
    variant). Prefer sparse-at-the-gate plus a time penalty, and never add a tracking reward to a searched line.
12. **Archives keep one trajectory per cell**, so Go-Explore covers state space, not behaviour space. Tag plans by
    contact sequence (section 22) or the beam culls the one-touch lineage against the two-touch leader - which our
    G0-G5 grid runs measured directly.
13. **Trivial-gate self-reinforcement.** Ours: win rate 0 -> 18.46% while the greedy frontier stayed flat and
    reservoir min-depth fell to 1,485 u. Report gate depth beside gate success.
14. **No memory.** Robot Parkour Learning: an MLP without a GRU scored 0% climb / 1% leap because the obstacle leaves
    the FOV before commitment. We are depth-only and memoryless.
15. **Metric noise.** Ours: 2.7-3.04x between byte-identical configs, a 27% seed floor at 750M, late 3-hour marks
    worse than early ones. Judge a gate arm on the spine testbed (replicates to 256 u) or on a time-to-event, never on
    an end-of-run mean.
16. **Throughput and API shape.** The per-tick Python goal bookkeeping costs 276k fps against ~690k, and
    `core.set_goal_box` is per-CORE, not per-env - a vectorized gate search must group envs by gate or pay the Python
    test.

# A minimal experiment that could validate the split on this map's finish room in one day

Do not train an arm first; **measure steerability**, because the whole formulation dies if the controller cannot be
commanded and no training arm would reveal that. Step 1 (local CPU, ~2 h): build a gate set for the finish room from
material we already have - the room-entry window of `runs/research/exitLONG2/spine_r8ep3.npy` (ticks 7950-8280), the
ramp-1 and ramp-2 faces from the BSP via `surfmask` rideability, and the contact states in the `exitROOMARCH` archive
- quantized to 40-80 gates, each stored as a `set_goal_box` AABB plus a speed band and an approach heading, with the
contact-face id kept as a topology tag. Step 2 (one 3090/4090, ~4-6 h, one seed): fine-tune the existing
goal-conditioned controller on the room window only -
`--demo-file spine_r8ep3.npy --demo-window 2000 --respawn-frac 1.0 --goals 1 --goal-obs ball --goal-views 4 --goal-reward sparse --goal-kmin 0.5 --goal-kmax 2.5 --tick-ms 7.63`
- with gates drawn from step 1 rather than from the reservoir; this is the same shape and cost as the exitROOM* arms
already placed, so the budget is known. Step 3 (local CPU, ~1 h): the measurement. From ONE restored room-entry state,
command each gate in turn and record first-contact position, exit speed and contact-face id over 20 rollouts per gate.
**Pass criteria, fixed in advance:** (a) first-contact spread across commanded gates >> 27 u, the current policy's
attractor width - if not, the split is refuted here and cheaply; (b) at least 30% of gates reached at >= 85% over 20
rollouts, the PRM-RL edge threshold, so a graph exists at all; (c) at least one commanded gate produces a one-touch
contact tag, which 44,100 tick-level probes have never produced. Step 4, only if (a)-(c) hold (~1 h): a depth-3 beam
over gate sequences (entry -> gate_i -> gate_j -> curtain), each edge executed by the frozen controller from a
restored state, scored by arrival TIME with dead edges dropped, plans deduplicated by topology tag rather than by
proximity, and the surviving sequence handed to `beam_tas` as a `--prefix-line` bias rather than executed - the RoGuE
rule. Deliverable either way: a yes/no on whether a one-ramp-touch exit exists under a goal-commanded controller,
which is the question the tick-level search has failed to answer, plus a measured steerability number that says
whether to build the macro layer at all.
