# Planner + executor: how to train a plan-emitting brain over a learned controller, jointly and stably

Literature survey, 2026-09-23. Web research only; no code was run, no GPU was touched,
and nothing in the repo changed except this file. ASCII only.

Commission (user, 2026-09-23, condensed from the verbatim request): "we need the
planner ... something that can produce some potential movement directions, like a
polyline ... and then the executor that actually executes this plan ... how to train
them such that first it is stable because I know that hierarchical reinforcement
learning is unstable ... we could penalize the executor for not executing the path
that the planner gave it. And also penalize the planner for giving such hard or
impossible trajectories that cannot be executed ... an egg and a chicken problem ...
the planner would produce just a polyline in 2D ... maximum like 600 to 1000 units ...
the brain should be inside the planner ... the plan shouldn't be too much different
from what the agent can do ... if you randomize the trajectories and take only the
ones that are different enough, these are your trajectory proposals ... We could
actually start with deterministic planner ... selecting some point in the map and
drawing the path towards that point using BFS."

**What this file does NOT repeat.** Four documents already cover much of the
neighbourhood. Their entries are only POINTED TO here, together with any number I read
that they did not have:

* [NAV] `docs/litsurvey-robot-navigation-hierarchy.md` (2026-09-20, 78 refs): command
  interfaces, ANYmal and legged navigation stacks, HIRO, HAC, FuN, option-critic, HRAC,
  HIGL, Director, LEAP, SoRB, HIQL, Nachum 2019, RL-RRT, MoPA-RL, MPPI, PEG, LGE,
  Florensa, VIN, ViKiNG, Roth 2025 FDM, Skill-Nav, Rudin 2022 IROS.
* [DET] `docs/litsurvey-detour-navigation.md` (2026-09-20, 80 papers): shaping
  theory, exploration bonuses, Go-Explore, learned distances (Laplacian, QRL, CRL),
  SoRB / SPTM / SGM, Sibling Rivalry, MEGA, GoalGAN, asymmetric self-play, AMIGo,
  ez-greedy, HER.
* [HIER] `docs/research-litsurvey-hier.md` (2026-09-06): option-critic and the
  deliberation cost, FuN, HIRO, HAC subgoal testing, Director, Nachum 2019,
  Head-to-Head racing (Thakkar 2022), ANYmal / Extreme / Robot Parkour, SoRB / SPTM /
  SGM / TTGS / LEAP, PRM-RL / RoGuE / RL-RRT, MuZero family, DC-MCTS, MAGIC, ExIt,
  PTSP, TMInterface, GCSL / SPiRL / SkiLD, the racers' gate inputs, BaRC.
* The ledger (`docs/research-results.md`), entries of 2026-09-21 and 2026-09-23: the
  certified-field waves 8-9 (the "stitching fallacy" and the zero-margin skim), the
  Discord papers (SEE, CRL, SGCRL, Bastankhah 2025, CPPO), OpenAI Five App. O.2 and
  the ladder-randomisation result.

Everything else below (about 80 papers) is new to the repo. Numbers are from the cited
paper unless marked "(ours)", which means measured in this project's ledger. "figure
only" means the paper reports the number only as a plot, so no digit is quoted. "NOT
FOUND" means I looked for it and it is absent from the source. Most entries were read
from the arXiv HTML or ar5iv renderings; where only an abstract could be read, the
entry says so.

---

## 0. The answer in one page

### The recommended recipe (details, constants and failure modes in section 7)

1. **Stage (a): labyrinth, a deterministic planner, and an executor trained on plans to
   RANDOM targets.**
   * The planner runs BFS on the walkable grid, from the agent's CURRENT position to a
     target drawn at random from the reachable floor. The distance band widens with
     measured success.
   * The plan is the next 2-4 s of that path (600-1000 u at walking speed). It reaches
     the executor through the existing 27-scalar fan on the per-env line, and is
     re-planned on a randomised clock and whenever the agent leaves it.
   * The executor's reward is the existing per-env arc progress along the CURRENT plan
     (`--goal-reward arc`), plus a bonus for reaching the plan's end, the time penalty
     and the death charge. There is no global potential.
   * A third of training plans are deliberately suboptimal or perturbed (Haro et al.
     2026), so the executor learns that a plan is a hint.
   * Verdict: finishes from the true start on lab100 and lab200 with the planner's
     target set to the finish. Then a held-out rung.
2. **Stage (b): surf, where free-space plans are wrong. Make feasibility a
   MEASUREMENT.**
   * Train the same executor on a mixed plan diet: BFS plans to random targets,
     segments of its own flights, and perturbations of those (earlier or later
     take-off, higher-slower or flatter-faster). Log every plan's outcome.
   * Certify a candidate plan by forking the simulator and running the frozen executor
     on the WHOLE plan from the real state. Never certify by chaining edges: that is
     the ledger's stitching fallacy.
   * Require a clearance margin equal to the executor's measured tracking error
     (FaSTrack, RTD). The ledger's zero-margin skim died 16 u below its plan.
   * Selection works as in PDM-Closed: simulate every candidate through the controller
     and take the best surviving one.
   * A capability judge trained on the logged outcomes (Kim 2022, Yang 2021, Guzzi
     2020, ABS) is the cheap amortisation, added second.
3. **Stage (c): a learned planner, trained in an order that cannot collapse.**
   * Proposals come from a vocabulary of the executor's own executed segments,
     deduplicated by epsilon-cover or NMS, plus an uninformed half (Ichter 2018:
     lambda = 0.5).
   * Scoring is realised progress in simulation under multiplicative feasibility gates
     (PDM-Closed, Plan-R1). The scores are distilled into a planner network with one
     head per score term (Hydra-MDP).
   * The planner is then fine-tuned by PPO on the task reward with the executor FROZEN
     (MLSH's warm-up, ANYmal Parkour).
   * Only after that do the two alternate. The executor's plan diet keeps at least 1/3
     random plans, and the planner's commitment is randomised, p ~ U{5,15} (HiPPO).
   * Exploration lives in plan space: novelty over plan outcomes, and the frontier of
     an archive of the states that plans reached.

### The findings that decide the design

* **F1. Pay the executor for PROGRESS along a crude plan, not for tracking it.**
  * On the same drone and track, MPC tracking a time-optimal trajectory succeeded 44%
    on the nominal model and 0% on a realistic one. RL paid progress to the next gate
    succeeded 100% on both (Song 2023). Their stated reason: the planning/control
    decomposition "limits the range of behaviors that can be expressed by the
    controller".
  * Progress along NON-flyable gate-centre segments came within 5.2% of time-optimal
    (Song 2021).
  * ETH's path-conditioned local planner has no path-adherence term at all and keeps
    its success under degraded paths (Haro 2026).
  * This is our own xAUTO result, "the ordering, not the line", in five other labs.
* **F2. Train the executor on plans wider than any planner will ever send, including
  wrong ones.** Examples: random commands resampled every 10 s (legged_gym),
  deliberately suboptimal and perturbed paths (Haro 2026), a throwaway high level used
  only to make the low level controllable (Heess 2016), and plan switches mid-motion
  (SNN4HRL's Ant falls over at every skill switch it was not trained on).
* **F3. Order the training; do not co-train from zero.**
  * MLSH, verbatim: "It is important that phi only be updated when theta is at a
    near-optimal level". Without the warm-up, both sub-policies learned to go to the
    midpoint of the destinations.
  * Every robot stack trains the controller first and freezes it (ANYmal Parkour
    96.3% vs 60.9% for manual navigation).
  * Joint on-policy PPO does work, but only after the skills exist (HiPPO).
* **F4. Feasibility is measured on the executor, for whole plans.**
  * Examples: DHRL's one-reachable-hop execution (40.1% vs 0.0% for fixed-period HRAC
    on a 56x56 maze), SSE's stop-on-failure, PDM-Closed's simulation through the
    controller (CLS-R 92 vs 72 for the best learned planner), and Kim 2022's
    capability model trained on the controller's rollouts (83.2% vs 45.2% success).
  * Chaining edges instead of certifying whole plans is what led the agent into the void
    over blue050's pit (ours, efCERT_blue050: 61 of 90 eval episodes turned left, none
    reached the ramp, all died following the graph's false descent).
* **F5. A plan needs a margin and a time.** FaSTrack and RTD inflate the plan by the
  controller's measured tracking error. HiTS shows that a plan without an arrival time
  becomes non-stationary as the executor gets faster: HAC "deteriorates quickly".
* **F6. Proposals are a vocabulary built from the executor's own motion, plus a
  scorer.**
  * Driving converged on this: PDM's 15, MTR's 64 -> 6, CoverNet's epsilon-cover,
    VADv2's 4,096, Hydra-MDP's 8,192.
  * Racing did too: HRHC's library of the car's own equilibria.
  * Setter-Solver is the user's design published whole: validity (goals the solver has
    achieved), feasibility (a judge of solver success) and coverage (entropy).
* **F7. Charge infeasible AND trivial plans with one term.** AMIGo's teacher gets +0.7
  only if the student succeeds and needed at least t* steps, and -0.3 otherwise.
  PAIRED's regret is zero for an impossible task. Learning progress (ALP-GMM) is zero
  for both mastered and hopeless tasks.
* **F8. Hierarchy pays in transfer, not on one task.**
  * DAC+PPO equalled plain PPO on 3 of 4 single tasks and won in transfer.
  * Nachum 2019 and Barkour say the same ([NAV], [HIER]).
  * So the executor must be trained across targets and maps, and every arm needs a flat
    control.

---

## 1. Why two-level training is unstable, and the published fix for each cause

| cause | what it looks like | published fixes | strongest evidence | ports to on-policy PPO? |
|---|---|---|---|---|
| non-stationary low level | the planner's experience ("plan X leads to Y") stops being true as the executor changes; HiTS: the executor getting FASTER shrinks the transition time | HIRO relabelling; HAC hindsight actions; HiTS timed subgoals; freeze the executor; MLSH warm-up; HiPPO on-policy joint gradient; Director's shared world model | HiTS Platforms ~86% vs HAC ~40%; MLSH without warm-up: both sub-policies collapse to one goal | freezing, warm-up, timed plans, HiPPO: yes. Relabelling: needs replay |
| infeasible subgoals | the planner asks for what the executor cannot do, and both levels learn from noise | HAC subgoal testing (lambda 0.3, penalty -H); HRAC adjacency (k = 10, eta = 20); BrHPO two-sided reachability; SSE stop-on-failure; DHRL one-hop execution; PRM-RL / SGM / Guzzi edge tests | DHRL 40.1% vs HRAC 0.0% at every period on 56x56; BrHPO ablations all worse | yes: supervised bookkeeping or reward terms |
| collapse to trivial subgoals | the planner asks for "stay here" or "the goal itself" (RLA's two degenerate solutions), because trivial plans always succeed or cost nothing | pay the planner on realised task progress; AMIGo's t* bar (+0.7 / -0.3); HIGL's pull toward a landmark; RLA's margins; learning-progress sampling | AMIGo KeyCorridor 0.54 vs RND 0.23; ALP-GMM 39.6% vs random 18.2% mastered | yes |
| degenerate options | with learned termination, options shrink to one step, one option dominates, or termination saturates at 0 or 1 | deliberation cost (sensitive), termination critic, interest functions, diversity bonus; OR no learned termination at all: a fixed or randomised clock | Harb 2018: termination -> 100% without a switching cost [HIER]; PPOC: one option used only at the start of each episode; HiPPO random p better on 6 of 8 | the clock ports trivially |
| proposal mode collapse | all samples of the generator converge to one plan | fixed anchors or vocabulary; NMS or DPP; a coverage (entropy) loss | DiffusionDrive diversity 11% -> 74% with anchors; MTR mAP 0.4164 with 64 k-means intention points + NMS vs 0.3245 for its 6-query end-to-end variant; Setter-Solver coverage ablation | yes |
| stitching in a learned graph | edges joined across different trajectories, or across states that differ in unobserved velocity, form chains no single rollout can fly | certify whole plans with one rollout; two-way consistency and edge clean-up (SGM); failure-aware edge costs (SSE, c_dist = 5); velocity in every node | ours: the burst-certified graph field (efCERT_blue050) sent the agent into the void; SGM clean-up 38.2% -> 92.9% [HIER] | yes |
| executor drift | the executor, fine-tuned on the planner's plans, forgets how to follow other plans | keep random plans in the diet; a KL anchor (RIS alpha = 0.1, GPS epsilon); HESS-style anchoring of well-fitted outputs | Ichter lambda = 0.5; Florensa 2:1 new:old; GoalGAN 2/3 : 1/3 [HIER] | yes |

---

## 2. Training the two levels jointly and stably

### 2.0 Already covered elsewhere: the one-line versions, so this section stands alone

| method | the stability mechanism | the number worth remembering | where |
|---|---|---|---|
| HIRO (Nachum et al., NeurIPS 2018) | off-policy correction: relabel a stored high-level action with the candidate subgoal (of 10: 8 Gaussian around s_{t+c} - s_t, the original, the achieved delta) that best explains the low level's actions under its CURRENT policy; c = 10 | Ant Maze 0.99, Push 0.92, Fall 0.66 at 10M steps, vs <= 0.02 for FuN / SNN4HRL / VIME | [HIER] 3, [NAV] 3.1 |
| HAC (Levy et al., ICLR 2019) | hindsight ACTION transitions (replace the proposed subgoal with the state actually reached), hindsight goal transitions, subgoal TESTING with a -H penalty | 3 levels > 2 levels > flat on ant four-rooms | [HIER] 4; constants in 2.2 below |
| FeUdal Networks (Vezhnevets et al., ICML 2017) | the manager emits a DIRECTION; a transition policy gradient credits it for the direction the world moved; the worker's gradient never flows into the manager; c = 10 | Montezuma ~2,600 vs an LSTM stagnating at 400 | [HIER] 2 |
| Option-critic + deliberation cost (Bacon 2017; Harb 2018) | learned termination; without a switching cost, options shrink to one step ("termination hits 100% very quickly"); eta = 0.020 fixes it | Amidar 880 vs 512, Hero 20,100 vs 2,625 | [HIER] 1 |
| Director (Hafner et al., NeurIPS 2022) | both levels train on imagined rollouts of ONE shared world model; goals are discrete codes of a goal autoencoder, chosen every K = 8 steps; the manager gets an exploration reward | the only method that reliably solves the larger egocentric Ant mazes | [NAV] 3.7, [HIER] 6 |
| SoRB / SGM / TTGS | plan over replay states with value-as-distance; SGM adds two-way consistency and edge clean-up; TTGS pairs a frozen policy with a graph | SGM ViZDoom 39.3% -> 60.7%; TTGS HIQL 1.4% -> 78.6% on antmaze-giant | [HIER] 12, [DET] 3 |
| Nachum et al. 2019 | the benefit of hierarchy is mostly exploration; flat agents with temporally extended exploration match HIRO | "Explore & Exploit" and switching ensembles match HIRO on most Ant tasks | [NAV] 3.13, [HIER] 7 |
| robotics freeze-then-train (ANYmal Parkour, Lee et al. 2024, ViNL) | train the controller on RANDOM commands, freeze it, train the navigator on top | parkour navigation 96.3% vs 60.9% manual, course B | [NAV] 2.2; new details in 4.2 |

### 2.1 Freeze, warm up, alternate: the schedules and their numbers

#### SNN4HRL: Florensa, Duan, Abbeel, "Stochastic Neural Networks for Hierarchical Reinforcement Learning", ICLR 2017, arXiv:1704.03012
* Mechanism:
  * Pretrain a stochastic-latent skill policy on a PROXY reward ("proportional to the
    magnitude of the speed of the robot, without constraining the direction of
    movement") plus a mutual-information bonus.
  * Then "we leverage the provided skills by freezing them and training a high-level
    policy". It picks one of K = 6 skills and holds it for a fixed switch time T.
* Evidence:
  * "80% of the trained SNNs with bilinear integration acquire at least forward and
    backward motion ... improved to 100% by increasing alpha_H".
  * The hierarchy "shines much more" once the horizon grows to 5,000 steps, and
    "frozen low-level policies are already sufficient".
  * Downstream numbers are figure only.
* Constants: TRPO step 0.01, gamma 0.99. T = 500 on the mazes and 10 on food gather;
  "the switch time T does not critically affect the performance for these static
  tasks".
* Stated failure: "When switching to a new skill, the robot finds itself in a region of
  the state-space that might be unknown to the new skill ... Ant ... falling over". The
  proposed fix is "learning a transition policy or integrating switching in the
  pretrain task".
* For RL_Surf: that failure is ours when a plan changes mid-ramp. The executor must be
  trained on plan SWITCHES (a new plan arriving at an arbitrary moment), not only on
  whole plans from a standstill. Random re-planning during executor training provides
  this for free.

#### Heess, Wayne, Tassa, Lillicrap, Riedmiller, Silver, "Learning and Transfer of Modulated Locomotor Controllers", arXiv:1610.05182, 2016
* Mechanism:
  * The low level (a "spinal" module that sees proprioception only) is trained
    "jointly with a provisional high-level controller which provides task-specific
    information during pre-training and hence ensures controllability".
  * Then "the provisional high-level controller is discarded and the weights of
    low-level controller are frozen. For each transfer task a new high-level controller
    is trained".
  * The high level acts every K = 10 steps. Its exploration noise is held constant over
    each interval, which "induces spatially and temporally correlated stochasticity".
* Evidence: "Where a monolithic end-to-end architecture fails completely, learning with a
  pre-trained spinal module succeeds". The soccer attempts "from scratch do not succeed".
  Returns are not given as numbers.
* Stated failures: "not all low-level controllers were equally suitable for solving the
  transfer task". The humanoid showed "a much stronger sensitivity to the learning
  parameters".
* For RL_Surf:
  * This is the user's "start with a deterministic planner" in its original form. A
    provisional, throwaway high level (for us, BFS to random targets) exists only to
    make the low level controllable, and is replaced later.
  * Holding the noise over each interval is the cheap version of "exploration in line
    space": explore by perturbing the PLAN and holding it, not by jittering keys.

#### MLSH: Frans, Ho, Chen, Abbeel, Schulman, "Meta Learning Shared Hierarchies", ICLR 2018, arXiv:1710.09767
* Mechanism: for each task, reset the master. For W iterations update ONLY the master
  (warm-up), then for U iterations update the master and the sub-policies jointly. The
  master period is N; both levels use PPO.
* Why the warm-up, verbatim: "It is important that phi only be updated when theta is at
  a near-optimal level ... If theta is random, the optimal sub-policies both lead the
  agent to the midpoint of the destinations." Without it, "both sub-policies moving to
  the same goal point".
* Constants:
  * Four rooms: W = 20, U = 30, master actions last 25 steps.
  * Physics tasks: W = 20, U = 40, "master policy actions last 200".
  * Learning rate 0.01 for the master, 0.0003 for the sub-policies.
* Evidence: Walk/Crawl transfer 14,333 vs 6,055 for a shared policy and -643 for a
  single policy. Ant obstacle 193 vs 0.
* Stated limits: "we do not optimize towards the true objective"; "no gradient signal
  being passed between the master and sub-policies".
* For RL_Surf:
  * The named failure is exactly the chicken-and-egg the user describes. An executor
    trained against a random planner learns to satisfy the AVERAGE plan.
  * MLSH's answer is an order: never update the low level while the high level is still
    random.
  * Our version: train the executor against a FIXED, sensible planner (BFS, stage a)
    before any learned planner exists. Warm a learned planner up against a FROZEN
    executor before the two are ever updated together.

#### HiPPO: Li, Florensa, Clavera, Abbeel, "Sub-policy Adaptation for Hierarchical Reinforcement Learning", ICLR 2020, arXiv:1906.05862
* Mechanism:
  * An on-policy hierarchical policy gradient trains BOTH levels at the same time with
    PPO. When skills are well separated, it treats the skill latent as part of the
    observation (Lemma 1). The neglected term is O(nH eps^(p-1)); with eps ~ 0.1 and
    p ~ 10 it is ~1e-10, and the cosine to the full gradient is 0.94-0.98.
  * Each level gets an unbiased baseline.
  * The manager's commitment length is RESAMPLED at every decision, p ~ Uniform{5, 15}.
* Evidence:
  * Jointly fine-tuning the skills "achieves higher final performance than fixing the
    sub-policies" (figure only).
  * From scratch: "faster learning and better performance than flat PPO"; option-critic
    "fails to learn"; MLSH "does not provide any benefits".
  * Random p beats fixed p on 6 of 8 robustness cases (e.g. a Snake mass change costs
    20% vs 25%).
* Constants: lr 3e-3, clip 0.1, 10 updates per iteration, gamma 0.999, 6 sub-policies,
  batch 50k-100k.
* For RL_Surf: the only published evidence that BOTH levels can be trained jointly by
  on-policy PPO, i.e. by our trainer. Two pieces port:
  * the randomised commitment p ~ U{5,15}: a free regulariser, and an answer to "how
    often should the planner be called";
  * the fact that joint adaptation beat freezing only AFTER the skills existed (they were
    pretrained with DIAYN or SNN4HRL first).

#### DAC: Zhang, Whiteson, "The Double Actor-Critic Architecture for Learning Options", NeurIPS 2019, arXiv:1904.12691
* Mechanism: the option SMDP is rewritten as two augmented MDPs. The high MDP holds the
  master and the terminations; the low MDP holds the intra-option policies. An
  off-the-shelf PPO then trains both. In practice both levels update at the same time on
  the same data, "though theory suggests alternating updates for stationarity".
* Evidence: on single-task MuJoCo, DAC+PPO matched plain PPO on 3 of 4 tasks: "learning
  the additional structure, the options, may be overhead". The clear wins were in
  TRANSFER (6 DMC task pairs, 4 options), over OC, IOPG, PPOC and AHP.
* For RL_Surf: a second independent measurement, after Nachum 2019 and Barkour, that a
  hierarchy buys nothing on ONE task and pays off when the low level is reused. For us
  the "tasks" are targets and maps. That is why the executor must be trained across many
  targets and the ladder / pool, not on one route.

#### Jain, Iscen, Caluwaerts, "Hierarchical Reinforcement Learning for Quadruped Locomotion", IROS 2019, arXiv:1905.08926
* Mechanism:
  * The high level outputs a latent command AND a duration: 100-700 low-level steps,
    i.e. 0.6-4.2 s.
  * Both levels are trained JOINTLY end-to-end with Augmented Random Search, on the
    path-following reward r = dist(x(t-1), goal) - dist(x(t), goal).
  * Transfer = freeze the low level and re-initialise the high level.
* Evidence: the frozen low level transferred to 22 different paths; the best latent size
  was 4; it ran on a real Minitaur.
* For RL_Surf: a 4-dimensional command is enough for a legged body to follow arbitrary
  2D paths. And joint training works with a derivative-free optimiser: the instability
  belongs to gradient-coupled levels, not to hierarchy as such.

#### RLA: Yang Yu, "Reinforcement Learning with Anticipation: A Hierarchical Approach for Long-Horizon Tasks", arXiv:2509.05545, 2025
* Mechanism: a goal-conditioned low level, plus a deterministic "anticipation model" that
  proposes a subgoal s_hat on the shortest path. It is trained by value-geometric
  consistency, V*(s0, sg) = V*(s0, s_hat) + V*(s_hat, sg).
* The two degenerate solutions it names are the user's "collapse":
  * the trivial START subgoal (s_hat ~ current state);
  * the trivial GOAL subgoal (s_hat ~ final goal).
* Fix: margin regularisers L_prog and L_non_trivial (c_prog = 1 step), plus a WARM-UP in
  which "only the low-level actor-critic" trains "with random subgoals" before joint
  training.
* Evidence: NONE. The preprint is theoretical (tabular convergence proofs, no
  experiments).
* For RL_Surf: useful only as a clean statement of the two collapse modes and of the
  warm-up everyone converges on. Do not cite it as evidence.

### 2.2 Deadlines, testing and failure-aware credit

#### HAC subgoal testing, the constants: Levy, Konidaris, Platt, Saenko, ICLR 2019, arXiv:1712.00948
* Subgoal testing rate lambda = 0.3. In that fraction of cases "the current lower level
  policy hierarchy must be followed exactly" (no noise).
* A missed tested subgoal costs "penalty = -H, or the negative of the maximum horizon of a
  subgoal", with gamma_i = 0 on that transition. H = 10 for 3 levels; [20, 30] for 2
  levels.
* The ablation (numbers figure only):
  * Without testing, levels "would always learn to set unrealistic subgoals that could
    not be achieved within H actions".
  * Penalising EVERY miss, even under the noisy exploring lower level, "performed
    significantly worse" ("overly conservative subgoals").
* For RL_Surf: this is the user's "penalize the planner for impossible trajectories",
  with one crucial detail. Charge the penalty only when the executor ran WITHOUT
  exploration noise, or the planner learns timid plans. In our trainer: certify a plan
  with a greedy executor rollout (the recorder's deterministic mode), and charge the
  planner only for those failures.

#### HiTS: Guertler, Buechler, Martius, "Hierarchical Reinforcement Learning with Timed Subgoals", NeurIPS 2021, arXiv:2112.03100
* Mechanism:
  * The high level emits a subgoal PLUS "the desired time until achievement". The time
    counts down, and when it runs out "the higher level is queried for a new subgoal".
  * Fixing the transition time makes hindsight-relabelled transitions "consistent with a
    completely stationary SMDP" (Prop. 2).
  * The diagnosis of HAC: "This transition time tau will shrink during the course of
    training as the lower level tries to optimize a shortest path objective". The
    executor getting FASTER is itself a non-stationarity for the planner.
* Evidence (30 seeds; no HIRO comparison):
  * Platforms: ~86% vs HAC ~40% asymptotic.
  * Tennis2D: ~40% in 20M steps, while HAC's "performance decreases ... after about 1.5
    million time steps as the lower level has become too fast".
* Constants:
  * SAC at both levels.
  * The low level is paid 1 only if the subgoal is achieved AT the deadline, and the
    deadline is terminal (no bootstrap).
  * 5% of timed subgoals are drawn uniformly.
  * A per-subgoal emission penalty -c (value NOT FOUND).
* For RL_Surf: the decisive entry for the plan's representation.
  * By HiTS's argument, a surf plan without a TIME or speed profile is non-stationary: as
    the executor learns to fly faster, the same polyline means something different to
    the planner.
  * So put an arrival time or a target speed on the polyline's vertices.
  * Our ledger already says "gates must carry a speed": a 0.2% velocity change at the
    room entry kills 95% of replays ([HIER] 0.4).

#### SSE: Hwang, Lee, Kim, Han, "Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning", arXiv:2506.21039 (v2 2026)
* Mechanism:
  * The high level gets a transition with return ONLY if the low level actually reaches
    the subgoal within a reachability threshold.
  * An unreachable subgoal TERMINATES the episode with zero return ("stop-on-failure").
    A partial success records the last reliably reached waypoint.
  * Graph edges get failure-aware costs, d~(v1 -> v2) = d(v1 -> v2) * max(1, c_dist *
    ratio_fail(v2)).
* Evidence: beats HIGL, DHRL, HIRO, HRAC, NGTE, BEAG and PIG on 5 AntMaze variants, 2
  KeyChest and 2 Reacher tasks (figure only, 5 seeds). Solves AntDoubleKeyChest with 3
  high-level steps.
* Constants: c_dist = 5 ("optimal balance"); exploration ratio eta = 0.2.
* For RL_Surf: two directly usable rules.
  * Credit the planner for REALISED plan outcomes only, and end the planner's episode
    segment when a plan fails. Do not relabel a failed plan as a success: that is what
    HER-style relabelling does, and it hides infeasibility.
  * The failure-aware edge cost is the generic version of our certified field: multiply
    an edge's length by the observed failure ratio of its target node.

### 2.3 Feasibility constraints on the high level: adjacency, landmarks, decoupled horizons

#### HRAC, the details: Zhang, Guo, Tan, Hu, Chen, "Generating Adjacency-Constrained Subgoals in Hierarchical Reinforcement Learning", NeurIPS 2020, arXiv:2006.11485 (TPAMI extension arXiv:2111.00213)
* Mechanism:
  * The high level may propose only subgoals inside the k-step adjacent region,
    G_A(s, k) = {g : d_st(s, phi^-1(g)) <= k}.
  * An embedding psi is trained CONTRASTIVELY on the agent's own trajectories: "When the
    temporal distance between two states in one trajectory is not larger than k, the
    corresponding element in the adjacency matrix will be labeled to 1". The hinge is
    l * max(||psi(g_i) - psi(g_j)|| - eps_k, 0) + (1 - l) * max(eps_k + delta -
    ||psi(g_i) - psi(g_j)||, 0).
  * The high level's loss adds eta * max(||psi(phi(s)) - psi(g)|| - eps_k, 0).
  * The adjacency is deliberately tied to the CURRENT low level: "we exploit the fact
    that the low-level policy itself changes over time during the training procedure".
* Constants:
  * k = 10 (both the subgoal period and the adjacency radius), eps_k = 1.0, delta = 0.2,
    eta = 20.
  * The adjacency net: lr 2e-4, batch 64, embedding 32. It is pretrained 50,000 steps on
    random-policy trajectories, then refit every 50,000 steps for 25 epochs.
  * TD3 at both levels. The goal space is discretised "to 1x1 grids for adjacency
    learning".
* Evidence:
  * In HLPS's table (5M steps, 10 seeds): Ant Maze HRAC 0.90 vs HIRO 0.71; Ant Push 0.01
    vs 0.00.
  * DHRL's Table 1 shows how much HRAC depends on its fixed period c_h:
    * 12x12 map: 43.0 / 88.4 / 78.3 / 4.5% at c_h = 5 / 10 / 30 / 50;
    * 24x24 map: 18.0 / 48.9 / 57.4 / 16.4%;
    * 56x56 map: 0.0% at every c_h.
* Stated limit: "The construction of an adjacency matrix limits our method to tasks with
  tabular state spaces".
* For RL_Surf:
  * The part that ports to PPO is the reachability net: supervised contrastive learning
    on the EXECUTOR'S OWN rollouts, refit on a schedule, used as a hinge penalty on the
    planner.
  * For surf, the k-step labels must be computed on (position, velocity) states and from
    ONE trajectory. Joining pairs across trajectories is exactly the ledger's stitching
    fallacy.

#### HIGL, the details: Kim, Seo, Shin, "Landmark-Guided Subgoal Generation in Hierarchical Reinforcement Learning", NeurIPS 2021, arXiv:2110.13625
* Mechanism: HRAC plus a DIRECTION.
  * Landmarks come from visited states: a coverage set chosen by farthest-point sampling,
    plus a novelty queue scored by RND.
  * A graph over them is weighted by the low-level value and cut above gamma_dist.
  * The high level's subgoal is pulled toward a pseudo-landmark, g_pseudo = g_cur +
    delta_pseudo (g_sel - g_cur) / ||g_sel - g_cur||, which is "located near the current
    state but also directed toward the selected landmark".
  * The adjacency term keeps subgoals feasible; the pull stops them collapsing onto
    trivial nearby points.
* Evidence:
  * "HIGL achieves a success rate of 65.1%, whereas HRAC performs about 17.6% at
    timesteps 10x10^5 in Ant Maze (U-shape, sparse)". That is 1e6 steps; [DET] 4 says
    10M, which is a misreading.
  * DHRL's sparse table, Antmaze at 1.0M: HIGL 60.3%, HRAC 48.9%, HIRO 0.8%. All three
    score 0.0% on the Bottleneck and Complex mazes.
  * Cost: 13 h vs HRAC's 6 h per 1M steps.
* Constants:
  * M_cov = 20 and M_nov = 20 landmarks (40 on the W-shape; one table read says 60,
    unresolved).
  * Landmark loss eta = 20.
  * delta_pseudo = 0.5 (point maze), 2.0 (ant maze) or 1.0 (arms); it is 0 for the first
    60K steps.
  * gamma_dist = 38.0 (mazes) or 15.0 (arms). Adjacency k = 5-7; subgoal period 10.
* For RL_Surf: delta_pseudo and gamma_dist are PER-ENVIRONMENT constants, which makes
  them map-specific under our rule 0b. The portable idea is the pull itself: the plan
  must point at a frontier landmark but end within the executor's reach.

#### DHRL: Lee, Kim, Jang, Kim, "DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical Reinforcement Learning", NeurIPS 2022 (oral), arXiv:2210.05150
* Mechanism: DECOUPLE the two horizons: "unlike the previous HRL methods which have the
  relationship of h_high x h_low = H, our algorithm does not have such relations".
  * Graph nodes are chosen by farthest-point sampling over replay states.
  * An edge exists when the temporal distance Dist(s -> g) = log_gamma(1 + (1 - gamma)
    Q_lo(s, pi(s, g) | g)) is below a cutoff.
  * The high level emits a far subgoal every c_h steps; the graph converts it into
    waypoints; the low level chases each waypoint for Dist(wp_{i-1} -> wp_i) steps. The
    executor is therefore only ever asked to cover ONE learned-reachable edge, however
    far the high level looks.
  * Failures "cause underestimation of Q-values ... overestimation of temporal distance
    ... and spoils the graph". The fix is two Q-networks with different HER proportions,
    plus a "gradual penalty" that separates achieved, near-but-failed and far-from-graph
    cases.
* Evidence:
  * DHRL reaches 95.1 / 91.1 / 40.1% on the 12x12 / 24x24 / 56x56 maps, where HRAC's
    best fixed c_h gives 88.4 / 57.4 / 0.0%.
  * Sparse Antmaze at 1.0M: 91.1% vs HIGL 60.3, HRAC 48.9, HIRO 0.8.
  * Bottleneck: 38.7% vs 0.0. Complex (4.0M): 40.1% vs 0.0.
* Constants: 300-500 landmarks; 75 initial episodes without graph planning;
  gradual-penalty rate 0.2; c_h 10-50. "Only a few hyperparameters have been changed
  across various locomotion experiments: the number of nodes, penalty, and c_h".
* For RL_Surf: the strongest single result in this section, and a direct answer to the
  user's "maximum like 600 to 1000 units".
  * The PLAN may be long: the planner looks far.
  * But the segment the executor is held to must be one certified-reachable hop.
  * The 56x56 result (40.1% vs 0.0% for fixed-period HRAC at every period) is what
    happens when the executor is asked for more than it can reach.

#### BrHPO: Luo, Sun, Ji, Zhan, "Bidirectional-Reachable Hierarchical Reinforcement Learning with Mutually Responsive Policies", arXiv:2406.18053, 2024
* Mechanism: the literal form of the user's two penalties in one algorithm.
  * The reachability of a subtask is R_i = E[D(psi(s_(i+1)k), g_(i+1)k) / D(psi(s_ik),
    g_(i+1)k)]: the ratio of the final to the initial distance to the subgoal, computed
    from the low-level reward. Lower is better.
  * The MANAGER is regularised: argmin KL(pi_h || exp(Q - V)) + lambda_1 R_i. This
    penalises the planner for plans that are not reached.
  * The WORKER is penalised: r_hat_l = r_l - lambda_2 R_i. This penalises the executor
    for not reaching them.
* Evidence:
  * Tasks: AntMaze, AntBigMaze, AntPush, AntFall, Reacher3D, Pusher. BrHPO beats HIRO,
    HIGL, RIS, CHER and flat SAC, on learning curves only (no table; an earlier read of
    approximate curve values is deliberately not quoted).
  * Ablations removing the manager term (NoReg), the worker term (NoBonus) or both
    (Vanilla) are all worse: "BrHPO outperforms all three variants by a significant
    margin".
  * It claims "at least a 2x improvement in training efficiency" over the
    adjacency-matrix and graph methods (HRAC, HIGL).
* Constants: lambda_1 and lambda_2 swept over {0.1, 0.5, 1, 2, 5}; subtask length k in
  {5, 10, 20, 50}; SAC, lr 3e-4, batch 256, buffer 1e6.
* Stated limitation, verbatim: "a main challenge is to design an appropriate low-level
  reward to compute the subgoal reachability, thus limiting the application in sparse
  low-level reward settings".
* For RL_Surf: our executor reward (arc progress to the plan's end) supplies exactly the
  dense low-level distance BrHPO needs, so the limitation does not bite. The caveats are
  an x,y goal space and off-policy SAC; the two terms themselves are scalars and port to
  PPO unchanged.

#### GCMR: Wang, Tang, Yang, Sun, Wang, Zhang, Chen, "Guided Cooperation in Hierarchical Reinforcement Learning via Model-based Rollout", arXiv:2309.13508 (IEEE TNNLS 2024 per listings)
* Mechanism:
  * (a) A HIRO-style correction that re-rolls states through a learned dynamics
    ensemble, with weights decaying at rho = 0.95 toward short rollouts. "Soft"
    relabelling moves a stored subgoal only delta_sg toward its relabel.
  * (b) A gradient penalty on ||grad_a Q_lo|| above a model-inferred Lipschitz bound.
  * (c) One-step rollout planning, in which the HIGH-level critic guides the LOW-level
    actor (L_osrp = -lambda_osrp E[Q_hi(...)]).
* Evidence: faster convergence than the other methods "in all Ant Maze tasks" (curves
  only, 5 seeds); second on Reacher and Pusher, behind PIG and DHRL.
* Constants: lambda_gp = 1.0; lambda_osrp in [5e-5, 5e-4]; delta_sg = 20 (small maze) or
  30 (large). Landmark counts across the family: HIGL 40, ACLG 120, DHRL 300, PIG 400.
* Stated limit: "the scope of applicability is off-policy goal-conditioned HRL".
* For RL_Surf: only (b) ports. Limit how sharply the executor's value can change with its
  action while the planner is learning, i.e. a smoothness regulariser on the low level
  during joint phases. Low priority.

### 2.4 Stable subgoal REPRESENTATIONS (relevant only if the plan space is learned)

* **LESSON**: Li, Zheng, Wang, Zhang, "Learning Subgoal Representations with Slow
  Dynamics", ICLR 2021 (OpenReview wxRwhSdORKG; the paper text was not readable here).
  * A slowness triplet loss pulls consecutive states together and pushes states c steps
    apart beyond a margin (as restated in HESS). The code default is c = 50.
  * In HLPS's table: Ant Maze 0.89, Ant Push 0.74, vs HIRO 0.71 / 0.00.
* **HESS**: Li, Zhang, Wang, Yu, Zhang, "Active Hierarchical Exploration with Stable
  Subgoal Representation Learning", ICLR 2022, arXiv:2105.14750.
  * The stability term L_s = E[lambda(s) ||phi(s) - phi_old(s)||^2] anchors the
    representation of the best-fitted 30% of states (lambda_0 = 0.1) to its previous
    value. With it "there is almost no change for the red features from 0.25M steps to
    the end"; without it "all the features are changing dramatically".
  * Subgoals are chosen among VISITED states within a radius r_g = 20, by novelty minus
    alpha = 0.03 times a reachability potential. They are reachable by construction.
  * HLPS table (stochastic, 10M steps): Ant Push sparse 0.77 vs LESSON 0.71 and HRAC
    0.08; Ant Fall sparse 0.29 vs LESSON 0.54.
* **HLPS**: Wang, Wang, Yang, Kamarainen, Pajarinen, "Probabilistic Subgoal
  Representations for Hierarchical Reinforcement Learning", ICML 2024, arXiv:2406.16707.
  * A Gaussian-process prior (Matern kernel) sits on the latent subgoal space; the
    posterior mean is the representation. k = 50.
  * Stochastic, 10M steps: Ant Push sparse 0.91 (HESS 0.77, LESSON 0.71, HRAC 0.08,
    TD3 0.00); Ant Fall sparse 0.79 (0.29 / 0.54 / 0.24 / 0.00).
  * Deterministic, 5M steps: Ant Maze 0.96 (0.91 / 0.89 / 0.90, HIRO 0.71); Ant Push
    0.90 (0.80 / 0.74 / 0.01, HIRO 0.00).
* **HIDI**: Wang, Wang, Pajarinen, "Hierarchical Reinforcement Learning with
  Uncertainty-Guided Diffusional Subgoals", ICML 2025, arXiv:2505.21750.
  * The high level is a conditional DIFFUSION model (N = 5 steps), regularised by a
    sparse GP (psi = 1e-3) and a Q term (eta = 5). It uses the GP mean with probability
    0.1.
  * Motivation, verbatim: "a changing low-level policy ... causes past experiences in
    reaching previously achievable subgoals to become invalid".
  * Qualitative evidence: a "small divergence between the generated and reached
    subgoals", while "subgoals generated by HIRO are often unachievable". Numbers figure
    only.
* Verdict for RL_Surf:
  * All four fix drift of a LEARNED subgoal space, off-policy (SAC or TD3). A plan
    expressed as a world-frame or ego-frame polyline does not drift, which is one more
    reason to keep the plan geometric.
  * HESS's anchor term is the one generic stabiliser worth remembering if the plan
    vocabulary itself is ever learned (stage c): anchor the well-fitted part of the
    vocabulary to its previous value.

### 2.5 Planning over subgoals with value-as-distance, at training time or at test time

#### LEAP, the details [NAV] 3.8 and [HIER] 12 did not have: Nasiriany, Pong, Lin, Levine, NeurIPS 2019, arXiv:1911.08453
* Mechanism:
  * Subgoals are chosen by minimising the l-infinity norm of a "feasibility vector" of
    TDM values along start -> g1 -> ... -> gK -> goal, minus lambda * sum log p(z_k)
    under a VAE prior.
  * The search is CEM over VAE latents, with replanning after each subgoal using K - 1
    subgoals.
* Constants:
  * Image tasks: Tmax = 100, K = 3 subgoals 25 steps apart.
  * Ant: Tmax = 600, K = 11 subgoals 50 steps apart.
  * CEM: 15 iterations x 1,000 sequences (Ant: 50 x 10,000); elites the top 25%, then the
    top 1%.
  * "l-infinity-norm outperformed the l1-norm". Prior weight lambda = 0.1 (navigation,
    Ant) or 0.001 (push). VAE latent 8 (Ant).
* Evidence:
  * Ant: "LEAP is the only method that successfully navigates the ant to the goal. HIRO,
    HER, HER+ don't attempt to go around the wall at all".
  * Push-and-reach: under 10 cm at 400k steps, 5x fewer samples than TDM-100.
* For RL_Surf: two details transfer.
  * The l-infinity aggregation scores a plan by its WORST hop. That is the right
    aggregation for a chain whose success is a product (PRM-RL's chaining law).
  * The log-prior term keeps the subgoals on the manifold of states the agent has
    actually produced. It is the user's "the plan shouldn't be too much different from
    what the agent can do", implemented as a likelihood penalty.

#### C-Planning: Zhang, Eysenbach, Salakhutdinov, Levine, Gonzalez, "An Automatic Curriculum for Learning Goal-Reaching Tasks", ICLR 2022, arXiv:2110.12080
* Mechanism: EM. "the E-step corresponds to planning an optimal sequence of waypoints
  using graph search, while the M-step aims to learn a goal-conditioned policy to reach
  those waypoints". At test time "we directly set the agent's goal to the final goal
  state without commanding any intermediate states".
* Evidence:
  * Maze-11x11: "only our method is able to make any learning progress".
  * "C-Planning performs the same at test-time with and without SoRB-based planning".
  * 1.74x-4.71x cheaper at deployment than planning over 4-16 waypoints.
  * Success numbers figure only.
* Constants: "Maximum Steps Reaching Goal" 20 (forces a waypoint change if the waypoint
  is not reached); 5 waypoints (MetaWorld) or 8 (2D maze); relabel next 0.3, future 0.2.
* For RL_Surf: the planner as a TRAINING-TIME curriculum that can be thrown away. If the
  flat policy trained on planned waypoints ends up not needing them, that is the simplest
  possible recipe, and the standing rules prefer it. Measure it: distil the planner away
  ([NAV] 7's Barkour control).

#### RIS: Chane-Sane, Schmid, Laptev, "Goal-Conditioned Reinforcement Learning with Imagined Subgoals", ICML 2021, arXiv:2107.00541
* Mechanism:
  * A high-level policy, trained AT THE SAME TIME as the policy and its critic, predicts a
    subgoal halfway to the goal by minimising C(s_g | s, g) = max(|V(s, s_g)|,
    |V(s_g, g)|).
  * The subgoals define a PRIOR policy, pi_prior(a | s, g) = E_{s_g}[pi(a | s, s_g)]. The
    policy is updated by argmax E[Q - alpha KL(pi || pi_prior)].
  * Subgoals are never commanded at test time.
* Constants: alpha = 0.1; high-level temperature lambda = 0.1; I = 10 subgoal samples;
  batch 2048.
* Evidence: large margins over LEAP, SAC+HER and HIRO on U-, S-, Pi- and omega-shaped Ant
  mazes and on image-based manipulation (figure only).
* For RL_Surf: the softest possible coupling. The planner does not command the executor at
  all; it regularises it. That is the natural shape for stage (c)'s "keep the executor
  close to what it already does on nearby plans", and alpha = 0.1 is a published starting
  weight.

### 2.6 Planner / policy CONSENSUS: the published form of the user's two penalties

#### Guided Policy Search (BADMM): Levine, Finn, Darrell, Abbeel, "End-to-End Training of Deep Visuomotor Policies", JMLR 17(39), 2016, arXiv:1504.00702
* Mechanism:
  * Minimise over (p, theta) of E_p[l(tau)], subject to p(u_t | x_t) = pi_theta(u_t | x_t),
    solved by Bregman ADMM with KL-divergence Lagrangian terms.
  * The trajectory optimiser (the PLANNER) is penalised for trajectories the policy (the
    EXECUTOR) does not reproduce, and the policy is trained to imitate the trajectories.
  * The trajectory update is bounded by D_KL(p || p_hat) <= epsilon.
  * The reason, verbatim: "The training data must come from the policy's own state
    distribution."
* Evidence (real PR2, end-to-end vs pose features vs pose prediction): shape cube 96.3% vs
  70.4% vs 0%; coat hanger 100% vs 88.9% vs 55.6%; bottle cap 88.9% vs 55.6%.
* For RL_Surf: THIS is the user's chicken-and-egg with both penalties, solved.
  * A Lagrangian charges the planner for plans the executor cannot follow and trains the
    executor toward the plans, converging to agreement.
  * It needs a model-based trajectory optimiser. For us, the "trajectory optimiser" is a
    search over proposals in the real simulator, and the consensus term becomes "prefer
    proposals close to what the executor did" (LEAP's prior, RIS's KL).

#### Mordatch, Todorov, "Combining the benefits of function approximation and trajectory optimization", RSS 2014, doi:10.15607/RSS.2014.X.052
* Mechanism (abstract level only; the PDF could not be parsed here): the two learning
  problems are coupled by ADMM. "the trajectory optimizer is given an augmented cost to
  find solutions that resemble the current output of the neural network", so it acts "as
  a teacher gradually guiding the network towards better solutions".
* For RL_Surf: the earliest statement of the design principle. The planner is not free: it
  pays for distance from what the executor currently does. Details NOT FOUND (full text
  not readable).

### 2.7 Learned options: the degenerate cases, and why a fixed or randomised clock wins

* **The Termination Critic**: Harutyunyan, Dabney, Borsa, Heess, Munos, Precup, AISTATS
  2019, arXiv:1902.09996.
  * Failure: "in later stages of training, the options tend to collapse to single-action
    primitives ... due to using the advantage function as a training objective of the
    termination condition".
  * The deliberation cost "is highly sensitive to the associated cost parameter".
  * Fix: terminate where the option's final-state distribution is predictable.
  * Evidence (four rooms) is qualitative: A2OC with the deliberation cost "tends to
    saturate on constant zero or constant one termination probability".
* **Options of Interest**: Khetarpal, Klissarov, Chevalier-Boisvert, Bacon, Precup, AAAI
  2020, arXiv:2001.00271.
  * Names the degeneracy "only one option being active all the time, or options
    switching often".
  * Learned interest functions help "to some extent": about 100 steps better than OC on
    four rooms, 2x faster convergence on TMaze.
* **Diversity-Enriched Option-Critic**: Kamat, Precup, arXiv:2011.02565, 2020.
  * Failure: "multiple options adopting very similar behavior, or a shrinking set of task
    relevant options".
  * The fix is a mutual-information bonus weighted tau = 0.0-0.7 PER ENVIRONMENT, which
    is a per-map constant under our rules.
* **PPOC**: Klissarov, Bacon, Harb, Precup, arXiv:1712.00004, 2017. Option-critic with
  PPO. In MuJoCo, "one option is used to gain momentum at the start of the episode and is
  never used thereafter". Options only helped on HopperIceBlock.
* Verdict for RL_Surf:
  * Every learned-termination fix adds a sensitive, per-task knob.
  * A fixed or randomised commitment (T = 10-500, K = 10, p ~ U{5,15}) is stable and
    insensitive on static tasks; learned termination degenerates by default.
  * So call the planner on a randomised clock plus a measured event (deviation, or plan
    consumed). Never use a learned termination head.

### 2.8 What section 2 says for RL_Surf

1. **Order the training.** Every stable recipe does:
   * first the executor, on random commands with a throwaway high level (Heess,
     SNN4HRL, the robot stacks, RLA);
   * second the high level, against a frozen executor (MLSH's warm-up, ANYmal, Lee 2024);
   * joint training only third, and only behind a shield: HiPPO's on-policy gradient
     with random commitment, HiTS's deadlines, and a random-plan diet for the executor.
2. **The plan must carry time** (HiTS), and **the executor must be held to one reachable
   hop** (DHRL).
3. **Feasibility is a two-sided term.** BrHPO puts lambda_1 on the planner and lambda_2 on
   the executor. HAC charges -H, but only under a noise-free executor.
4. **Never learn termination.** Use a randomised clock (p ~ U{5,15}) plus measured
   events.
5. **GPS/ADMM is the formal version of the user's two penalties.** The practical version
   here:
   * the planner's proposals come from the executor's own motion, which gives consensus
     by construction;
   * the executor is trained on the planner's plans mixed with random ones.

---

## 3. Feasibility, and the user's two penalties

### 3.1 Penalty 1, on the EXECUTOR: what the plan-following reward should be

The evidence in one table first:

| what the low level is paid for | paper | outcome |
|---|---|---|
| tracking a time-optimal trajectory (MPC) | Song et al. 2023 | 44% success on the nominal model, 0% on a realistic one; "crashes the drone immediately after launch" on hardware |
| progress along the reference with a contour penalty (MPCC) | Song 2023; Liniger 2015 | 76% / 20% success (Song); laps 8.9-9.2 s vs 9.5-9.8 s for the two-level HRHC (Liniger) |
| progress to the next gate (RL) | Song 2023 | 100% / 100%, 5.14 / 5.26 s laps |
| progress along NON-flyable gate-centre segments | Song et al. 2021 | within 5.2% of time-optimal; 97.4% success on 1,000 random tracks |
| centreline-projection progress, minus a wall penalty | Fuchs et al. 2021 | beat the fastest of 52,303 human players (1:15.913 vs 1:16.062) |
| a position + time command, paid only at the end | Zhang et al. 2024; Rudin 2022 IROS [NAV] | "far outperform[s]" velocity tracking, which falls below 90% after level 4 |
| the path as a hint, with no adherence term | Haro et al. 2026 | +7.02% SPL; success unchanged under degraded paths |
| normalised waypoints vs velocities | GNM 2023 | 0.95 vs 0.54 success in moderate environments |
| a velocity-command kernel exp(-e^2 / 0.25) | legged_gym | the de-facto standard for velocity commands; walking in 4-20 min |
| strict tracking of an optimiser's footholds and base | DTC 2024 | wins on sparse stepping terrain where the task reward is too sparse to find; "not ... more sample efficient" |

#### Song, Romero, Mueller, Koltun, Scaramuzza, "Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning", Science Robotics 2023, arXiv:2310.10943
* The argument, verbatim: "OC is limited by its decomposition of the problem into planning
  and control with an explicit intermediate representation, such as a trajectory, that
  serves as an interface. This decomposition limits the range of behaviors that can be
  expressed by the controller, leading to inferior control performance when facing
  unmodeled effects." A time-optimal trajectory uses the most of the available actuator
  power at all times, so under model mismatch there is no control authority left.
* Evidence (Split-S; lap time and success):

  | controller | nominal model | realistic model |
  |---|---|---|
  | trajectory tracking | 4.92 +- 0.10 s, 44.0% | failed, 0.0% |
  | MPCC | 5.03 +- 0.18 s, 76.0% | 5.34 +- 0.27 s, 20.0% |
  | RL | 5.14 +- 0.09 s, 100% | 5.26 +- 0.32 s, 100% |

  * Real world: tracking "crashes the drone immediately after launch"; MPC 5.8 s; RL
    5.35 s. Top speed 108 km/h, 12.58 g.
* Reward: r(k) = ||g_k - p_{k-1}|| - ||g_k - p_k|| - b ||omega_k||, with b = 0.01;
  collision -10; finish +10; 100 Hz; trained "in only ten minutes".
* For RL_Surf: the strongest argument against making the executor TRACK a plan at the
  performance limit, and surf at 3,000+ u/s is at the performance limit. The planner's
  line must be a corridor and an ordering; the executor must own the speed.

#### Song, Steinweg, Kaufmann, Scaramuzza, "Autonomous Drone Racing with Deep Reinforcement Learning", IROS 2021, arXiv:2103.08624
* Reward: progress r_p = s(p_t) - s(p_{t-1}) along "straight line segments connecting
  adjacent gate centers", plus a safety term near gate edges. The path is not flyable:
  "no additional effort is required for computing the reference path".
* Evidence (lap times in s, AlphaPilot / Split-S / AirSim):
  * polynomial trajectory 12.23 / 15.13 / 23.82;
  * time-optimal (CPC) 8.06 / 6.18 / 11.40;
  * RL 8.14 / 6.50 / 11.82, "within 5.2% of the theoretical limit";
  * 97.4% success on 1,000 random tracks.
* For RL_Surf: a crude, unflyable line is enough as the progress coordinate. It is the
  published twin of our xAUTO result: 58 chords, 24.8% of the vertices inside solid
  geometry, and it matched the full line.

#### Fuchs, Song, Kaufmann, Scaramuzza, Duerr, "Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning", RA-L 2021, arXiv:2008.07971
* Setup: SAC; r_t = centreline-projection progress - c_w ||v||^2 on wall contact, with
  c_w = 5e-4; gamma 0.98-0.982; 10 Hz in training, 60 Hz in evaluation; observations are
  13 range finders plus 10 curvature samples 0.2 s apart.
* Evidence (agent vs the fastest human):
  * track 1: 1:15.913 vs 1:16.062, the best of 52,303 players;
  * track 2: 1:39.408 vs 1:39.445;
  * track 3: 2:06.701 vs 2:07.319.
  * Training took 56-73 h on 4 PlayStations.
* Failure modes:
  * Without the wall term, policies "did not brake and simply grinded along the track's
    walls".
  * With a FIXED penalty, the agent "did not react" or chose "full braking and standing
    still".
  * It could not finish unseen tracks.
* For RL_Surf: a penalty must scale with the thing being avoided (here speed squared); a
  constant penalty teaches standing still. If the executor gets an off-plan charge at
  all, it must be proportional, not flat.

#### legged_gym: Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep RL", CoRL 2021 (PMLR 164), arXiv:2109.11978, and the repository config
* Reward:
  * Kernel exp(-||e||^2 / 0.25), from the config: "tracking_sigma = 0.25 # tracking
    reward = exp(-error^2/sigma)".
  * Scales: tracking_lin_vel 1.0, tracking_ang_vel 0.5.
  * Penalties: lin_vel_z -2.0, ang_vel_xy -0.05, torques -1e-5, dof_acc -2.5e-7,
    action_rate -0.01, collision -1; feet_air_time +1.0.
  * Every scale is multiplied by dt, and only_positive_rewards = True.
* Commands: vx, vy and yaw rate in [-1, 1], heading in [-pi, pi]; "resampling_time = 10."
  (10 s); commands with norm < 0.2 are zeroed; episode 20 s.
* Curricula:
  * Terrain: move up if distance > env_length / 2, down if distance < |cmd| x
    episode_length x 0.5. Solving the last level sends the robot to a random level.
  * Command: widen lin_vel_x by 0.5 once tracking exceeds 0.8 of its scale.
* PPO: 24 steps per env, gamma 0.99, lambda 0.95, lr 1e-3, 5 epochs, 4 minibatches,
  entropy 0.01, clip 0.2. It "struggles when we provide fewer than 25 consecutive steps,
  corresponding to 0.5 s".
* For RL_Surf: copy two things for random targets. Resample the command every 10 s WITHIN
  an episode, and promote or demote on MEASURED distance. Both are generic.

#### ABS: He, Zhang, Xiao, He, Liu, Shi, "Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion", RSS 2024, arXiv:2401.17583
* The agile policy is trained on GOAL REACHING, not velocity tracking:
  * r_task = 60 r_possoft + 60 r_postight + 30 r_heading - 10 r_stand + 10 r_agile -
    20 r_stall.
  * r_track = 1/(1 + ||e/sigma||^2) x 1(t > T - T_r)/T_r, with sigma_soft = 2 m (T_r =
    2 s), sigma_tight = 0.5 m (T_r = 1 s) and sigma_heading = 1 rad (T_r = 2 s).
  * r_agile rewards forward speed toward a cap that "cannot be reached" (4.5 m/s).
  * Goals x ~ U(1.5, 7.5) m, y ~ U(-2, 2) m; episodes U(7, 9) s.
* A reach-avoid value network is trained on "200k episodes" of the agile policy's own
  rollouts (gamma_RA = 0.999999).
  * When V > V_threshold = -0.05, a recovery policy takes over. It tracks a twist chosen
    by 5 gradient steps of "arg min d_goal_future s.t. V_hat < V_threshold".
  * The threshold is set "to compensate for learning errors without causing
    over-conservative shielding".
* Evidence:
  * ABS 79.1 +- 4.4% success and 5.7% collisions, vs the agile policy alone at 77.3% /
    21.7% and a Lagrangian baseline at 77.4% / 9.1%.
  * Real world: 9-10 of 10 in every environment; peak 3.1 m/s.
* Stated limit: "when the obstacles are too dense and form a local minimum, our policy can
  easily fail".
* For RL_Surf: two templates.
  * The executor's end-of-plan reward: two kernels, soft and tight, paid only in the last
    seconds.
  * A capability model trained on the executor's own rollouts, used as a certificate.

#### Zhang, Rudin, Hoeller, Hutter, "Learning Agile Locomotion on Risky Terrains", IROS 2024, arXiv:2311.10484
* The command is a target position, a heading and a time. The task reward is paid only in
  the last seconds, because "inappropriate velocity commands can be misguiding".
* Evidence:
  * "The settings based on the navigation formulation far outperform the velocity tracking
    one". Velocity tracking falls below 90% after level 4 (~40% sparsity).
  * Real stepping stones 8/10; balance beams 5/5; >= 2.5 m/s.
* Constants:
  * Position term 1/(1 + ||p - p*||^2), with weight 10 over the last 2 s or 25 over the
    last 4 s. Heading weights 5-12.
  * Termination -200.
  * "Don't wait": -(1 - delta_p(1)) x 1(||v|| < 0.2).
  * "Move in direction": cos(v, p* - p), weight 1 "for first 150 iter".
  * RND weight 1.0. Targets 1.5-4.9 m; episodes 5-7 s.
* For RL_Surf: note the "move in direction" bootstrap: a dense directional term used only
  for the first 150 iterations, and removed once the terminal reward is being found.

#### DTC: Jenelten, He, Farshidian, Hutter, "DTC: Deep Tracking Control", Science Robotics 2024, arXiv:2309.15462
* Mechanism:
  * A trajectory optimiser (TAMOLS) produces footholds and a base trajectory, and a 50 Hz
    PPO policy is rewarded for TRACKING them.
  * Base terms: exp(-sigma ||e||^2) with sigma = 1200 / 90 / 10 / 1 / 0.05 / 0.005 for
    position / rotation / linear velocity / angular velocity / linear acceleration /
    angular acceleration, weight 1 each.
  * Footholds: -ln(||p* - p||^2 + 1e-5), weight 6, paid once per gait cycle.
  * The base-pose reference is deliberately NOT observed, "to reduce sensitivity to
    mapping errors".
* Evidence:
  * Higher success than pure RL over 120 terrains (figure only); "outperformed
    baseline-rl-2 by a huge margin" on sparse terrain.
  * Parkour 20% faster than TAMOLS alone; footholds tracked to 2.3 cm.
  * More than 4,000 robots for two weeks, and "not ... more sample efficient than similar
    unifying RL approaches".
* For RL_Surf: the case FOR tracking. When the task reward is too sparse to find
  (stepping stones), a planner's dense targets are what the RL executor needs, which is
  our labyrinth situation. But there the planner is dynamically exact. Our free-space BFS
  is not, so the tracking must stay loose (a corridor and progress), not tight (DTC's
  sigma = 1200).

#### RLOC: Gangapurwala, Geisert, Orsolino, Fallon, Havoutis, "RLOC: Terrain-Aware Legged Locomotion using Reinforcement Learning and Optimal Control", T-RO 2022, arXiv:2012.03094
* Stack: an RL footstep planner, queried once per stride; a 400 Hz whole-body controller;
  an RL domain-adaptive tracker; an RL recovery policy. "The footstep planner is trained
  to adapt to the behavior of the motion controller".
* Evidence: under 32 h of total training vs ~454 h for prior work; zero-shot transfer
  from ANYmal B to C.
* Failure: without a "strategic velocity command generator", training hit "reward
  collapse".
* For RL_Surf: another instance of a planner trained WITH the controller in the loop, so
  it learns the controller's envelope rather than assuming it.

#### Meyer, Robinson, Rasheed, San, "Taming an Autonomous Surface Vehicle for Path Following and Collision Avoidance Using Deep Reinforcement Learning", IEEE Access 8, 2020, arXiv:1912.08578
* Reward: r_pf = -1 + (sqrt(u^2 + v^2) / U_max * cos(chi_tilde) + 1) * (exp(-gamma_e
  |y_e|) + 1).
  * gamma_e = 0.05 per metre. The course error chi_tilde is measured to a look-ahead
    point 100 m down the path.
  * The exponential was chosen over a quadratic for its "fatter tails", so the agent is
    still paid for small improvements when it is far off the path.
* A trade-off weight lambda between path following and collision avoidance is sampled PER
  EPISODE (Gamma(1, 2)) and given to the policy.
* Evidence (PPO, 100 random scenarios): lambda = 1 gives 87% success and 22.12 m mean
  cross-track error; lambda = 1e-5 gives 100% success and 66.8 m.
* For RL_Surf:
  * The multiplicative "speed along the path x cross-track kernel" form pays progress only
    while near the line, so it cannot be farmed by flying fast in the wrong place.
  * The per-episode lambda input is the published form of a TIGHTNESS command. The
    planner can later ask for "follow exactly" or "use the line as a hint", and the
    executor has been trained on both.

#### AutoRL: Chiang, Faust, Fiser, Francis, "Learning Navigation Behaviors End-to-End With AutoRL", RA-L 2019, arXiv:1809.10124
* Path following (PF):
  * A PRM path is interpolated at spacing d_ws, and the policy sees "a partial path
    consisting of the N_partial un-reached waypoints".
  * A waypoint is reached "iff 1) the previous waypoint w_{i-1} is already reached and 2)
    the robot is within d_wr of w_i".
  * The reward has step, distance-to-next-waypoint, collision and clearance terms. The
    weights AutoRL's search found were [-0.0351, -0.9098, -34.158, -4.769], in that
    order.
* Evidence:
  * PF: 98.7% average success across three large buildings, vs 80% for the best
    non-learned baseline (PRM-DWA).
  * The point-to-point (P2P) policy, trained on random start/goal pairs 5-10 m apart,
    "struggle[s] with large-scale local minima, such as moving from one room to another,
    which it was not designed to do".
  * Cost: "12 days to train 1000 agents".
* For RL_Surf: the cleanest published version of our labyrinth result. A policy trained on
  random local goals stands at the wall between rooms, like our Euclid cells (34 of
  them); the same kind of policy fed a planner's path does not. The ordered-waypoint rule
  (a waypoint counts only after its predecessor) is the arc-window rule our `goalarc.py`
  already implements.

#### DD-PPO: Wijmans, Kadian, Morcos, Lee, Essa, Parikh, Savva, Batra, "Decentralized Distributed PPO", ICLR 2020, arXiv:1911.00357
* Task: PointGoal navigation with RANDOM start/goal pairs in 3D house scans. Reward:
  r_t = -delta geodesic_distance - 0.01 per step, plus a terminal 2.5 x SPL.
* Evidence:
  * 99.9% success, SPL 0.969 (val, RGB-D + GPS+Compass) after 2.5 billion frames.
  * A BLIND agent (GPS+Compass, no camera): 97.3% success, SPL 0.729.
  * "error recovery, including several well executed backtracks".
* For RL_Surf:
  * The strongest random-target navigator in the literature was trained on a GEODESIC
    shaping reward, the same "pre-solved" reward our labyrinth needed.
  * Its lesson is not that geodesic shaping is avoidable. It is that a policy trained on
    random targets under a correct per-target field becomes a general navigator.
  * On the labyrinth, arc progress along a BFS plan to a random target IS that per-target
    geodesic field, delivered through the observation.

#### MPCC and HRHC: Liniger, Domahidi, Morari, "Optimization-Based Autonomous Racing of 1:43 Scale RC Cars", Optimal Control Applications and Methods 2015, arXiv:1711.07300
* MPCC (one level): cost = contouring (lateral) error + lag error - gamma x progress along
  the centre line. Maximising progress "is closely related to a time optimality
  criterion", and lets the controller schedule its own speed instead of tracking a timed
  reference.
* HRHC (two levels): a path planner ENUMERATES a finite trajectory library generated from
  the car's own stationary motions ("95 stationary points, out of which 26 correspond to
  drifting equilibria") over a 0.35 s horizon (N = 14), picks the best feasible one, and
  an MPC tracks it.
* Evidence:
  * Laps on an 18.43 m track: HRHC 9.5-9.8 s vs MPCC 8.9-9.2 s.
  * Compute: HRHC 2.93 ms mean planning + 5.10 ms QP, deadline missed 0.07% of the time;
    MPCC 15.03 ms and 4.4%.
* For RL_Surf: both halves matter.
  * MPCC is the classical form of the executor reward we want: progress along the line
    with a contour penalty and no time parameterisation.
  * HRHC is the classical form of the user's proposal set: a library built from the
    BODY'S OWN feasible motions (drifts included), enumerated and selected every cycle.
  * The numbers state the price: the two-level version was ~6% slower than the
    single-level progress maximiser.

#### Arena-Rosnav: Kastner et al., IROS 2021, arXiv:2104.03616
* Mechanism:
  * The DRL local planner chases a subgoal on the global path "within the range of
    look-ahead distance", d_ahead = 1.55 m.
  * The global path is re-planned when "the robot is off-course for a certain distance
    from the global path or if a time limit t_lim is reached without movement", with
    t_lim = 4 s.
  * Reward: +15 at the goal, a progress term, -10 on collision, -0.15 in the danger zone,
    -0.01 per step.
* Evidence (20 obstacles at 0.3 m/s): 81% success vs MPC 70.5%, TEB 68%, DWA 56%.
* For RL_Surf: a published instance of the replanning rule the user asked about: a fixed
  look-ahead subgoal, plus a replan on deviation or on a stall timer.

#### Haro, Richter, Yang, Cadena, Hutter, "Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation", arXiv:2603.13888, 2026
* Mechanism:
  * The local RL planner is conditioned on N = 15 reference waypoints in the robot frame
    (direction + log-compressed distance, with self- and cross-attention).
  * TRAINING PATHS ARE A MIX: optimal A* routes on probabilistic roadmaps, AND
    deliberately suboptimal Greedy-Best-First paths with detours. Every waypoint is
    perturbed by up to 1 m.
  * The reward has NO path-adherence term: goal proximity, smoothness, collision and tilt,
    plus a "shortcut reward" when the agent skips unnecessary waypoints.
* Evidence: +7.02% SPL over the baseline with optimal paths. Under degraded paths, success
  is unchanged (0.8276 vs 0.8320).
* For RL_Surf: the direct precedent for stage (a), and a warning about penalty 1. ETH let
  the path be a HINT the policy can overrule, and trained that by showing it wrong paths on
  purpose. That is what makes the executor safe to drive with an imperfect planner, and on
  surf every early plan is imperfect.

**Verdict for 3.1: what the executor should be paid.**
* Arc progress along the CURRENT plan: the projection gain, windowed and gated by a
  corridor. This is what `goalarc.py` already computes.
* A bounded bonus for being at the plan's end when its time runs out: ABS's and Zhang's
  kernel 1/(1 + (d/sigma)^2), in a soft and a tight version, paid in the last 1-2 s.
* The time penalty and the death charge.
* No cross-track penalty beyond the corridor gate at first. If one proves necessary, make
  it a bounded kernel that multiplies the progress (Meyer), never a flat charge (Fuchs).

The user's "penalize the executor for not executing the path" is then implicit:
off-corridor flight earns nothing, and failing to reach the plan's end forfeits the end
bonus. That is loose enough for Song's and Haro's argument, and tight enough to stop the
executor ignoring the plan.

### 3.2 Penalty 2, on the PLANNER: constraining or penalising infeasible plans

HAC's subgoal testing (lambda = 0.3, -H), SSE's stop-on-failure, HRAC's adjacency hinge
(eta = 20) and DHRL's one-hop execution are in section 2. The rest:

#### Setter-Solver: Racaniere, Lampinen, Santoro, Reichert, Firoiu, Lillicrap, "Automated Curricula Through Setter-Solver Interactions", ICLR 2020, arXiv:1909.12892
* Mechanism: a goal SETTER trained with three losses, L_setter = L_val + L_feas + L_cov.
  * Validity: E_{g achieved by the solver}[-log p(S^-1(g + xi, f))]. Maximise the
    likelihood of goals the SOLVER HAS ACTUALLY ACHIEVED.
  * Feasibility: E_{z,f}[(J(S(z, f)) - sigma^-1(f))^2]. A JUDGE network J, trained by
    cross-entropy to predict the solver's success, must agree with a requested
    feasibility f ~ U[0, 1]. The judge's weights are frozen during the setter update.
  * Coverage: E[log p(S(z, f))]. Maximise the entropy of the goals.
* Evidence:
  * Grid alchemy task: "validity and coverage losses are necessary, while the feasibility
    loss is not necessary, but does improve consistency".
  * 3D colour-pair finding: "Removing any of the setter losses results in substantially
    worse performance".
  * Beats Goal GAN.
* Stated limit: "Learning conditional generative models ... can be challenging if the valid
  goals are not trivially observable".
* For RL_Surf: **this is the user's design, published.**
  * Validity = "proposals from the executor's repertoire": the setter is a density model
    of achieved outcomes.
  * Feasibility = "penalise impossible plans": a judge of executor success. Difficulty is
    also a KNOB f the planner can dial, not only a penalty.
  * Coverage = "take only the ones that are different enough".
  * The ablation says which part is indispensable: validity and coverage always,
    feasibility for consistency.

#### AMIGo, the constants: Campero, Raileanu, Kuttler, Tenenbaum, Rocktaschel, Grefenstette, ICLR 2021, arXiv:2006.12122
* Teacher reward: +alpha = 0.7 if the student reaches the goal having taken >= t* steps;
  -beta = 0.3 otherwise (failed OR too easy). t* += 1 when the student succeeds in more
  than t* steps 10 consecutive times.
* Student reward: 1 - 0.9 t / t_max on arrival.
* Evidence: KeyCorridor (hard) 0.54 vs RND 0.23, RIDE 0.19, ICM 0.00. ObstructedMaze
  (hard) 0.17 vs 0.00 for all three.
* Limits: absolute (x, y) goals, in fully observable grids.
* For RL_Surf: the published answer to the user's "penalise lines the follower cannot
  execute WITHOUT rewarding stationary ones". One reward is negative for BOTH the impossible
  plan and the trivial plan, and the triviality bar t* rises by itself. [DET] lists AMIGo;
  these are the numbers.

#### PAIRED: Dennis, Jaques, Vinitsky, Bayen, Russell, Critch, Levine, "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design", NeurIPS 2020, arXiv:2012.02096
* Mechanism: the proposer (an environment designer) is paid REGRET = U(antagonist) -
  U(protagonist), approximated as the max over antagonist episodes minus the protagonist's
  mean. Verbatim: "If the adversary generates unsolvable environments, the antagonist and
  protagonist would perform the same and the adversary would get a score of zero".
* Evidence: zero-shot transfer to hand-designed mazes. PAIRED is the only method with
  non-trivial success on the Labyrinth and Maze test levels, where domain randomisation
  and minimax score about zero (figure only). All three agents use PPO.
* For RL_Surf: an impossible-plan filter that needs no judge network. A plan is worth
  proposing only if SOME executor rollout (the best of K noisy rollouts, the "antagonist")
  completes it while the greedy executor does not yet. That is: feasible, and not yet
  mastered. Cheap for us, because the K rollouts are forks of the real simulator.

#### PLR: Jiang, Grefenstette, Rocktaschel, "Prioritized Level Replay", ICML 2021, arXiv:2010.03934
* Mechanism: score each level (for us, a target or a plan family) by its mean |GAE| (the
  L1 value loss). Sample by rank with temperature beta = 0.1, mixed with a staleness term
  rho = 0.1.
* Evidence (abstract): matches the previous state of the art on Procgen. Combined with
  UCB-DrAC, "over 76% improvement in test return relative to standard RL baselines".
* For RL_Surf: the generic replacement for a hand-written target curriculum in stage (a).
  Targets whose returns the critic still mispredicts are replayed; mastered and hopeless
  ones decay. No map constant.

#### ALP-GMM: Portelas, Colas, Hofmann, Oudeyer, "Teacher Algorithms for Curriculum Learning of Deep RL in Continuously Parameterized Environments", CoRL 2019, arXiv:1910.07224
* Mechanism:
  * Absolute learning progress, alp = |r_new - r_old|, measured against the nearest
    previously sampled task parameter.
  * GMMs with 2..10 components are refit every 250 episodes on the last 250 (parameter,
    ALP) pairs, chosen by AIC.
  * 20% of tasks are drawn uniformly (p_rnd).
* Evidence:
  * Stump Tracks (20M steps): 39.6 +- 9.6% of test tasks mastered, vs Covar-GMM 33.3,
    RIAC 32.1, random 18.2.
  * Hexagon Tracks (80M): 80%, vs 68% for a hand-designed oracle curriculum.
* For RL_Surf: learning progress is zero for BOTH mastered and impossible tasks, so a
  planner or target sampler that chases it cannot collapse onto either. That is the
  property the user asked for, with a 20% uniform floor as the published insurance.

#### Plan-R1: Tang, Kan, Shan, Chen, "Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling", arXiv:2505.17659, 2025
* Mechanism:
  * A trajectory planner is pretrained by next-token prediction on EXPERT DRIVING LOGS
    (forbidden as a source here), then fine-tuned by RL: GRPO with group size G = 4, and
    "variance-decoupled" advantage normalisation so that rare safety violations keep their
    weight.
  * The reward's SAFETY terms are MULTIPLICATIVE gates: collision-free and inside the
    drivable area, or the reward is zero.
  * The soft terms are weighted: comfort 2, time-to-collision 5, speed limit 2, progress 1.
* Evidence (nuPlan reactive closed-loop score, with post-processing; Val14 / Test14-hard /
  Test14-random): 93.54 / 81.70 / 93.71, vs PDM-Closed 92.12 / 75.19 / 91.64.
* For RL_Surf: the multiplicative gate is the clean form of "an infeasible plan earns
  nothing, however much progress it promises". A subtracted penalty is worse, because a
  large enough promised progress can out-vote it.

#### FaSTrack: Herbert, Chen, Han, Bansal, Fisac, Tomlin, "FaSTrack: a Modular Framework for Fast and Guaranteed Safe Motion Planning", CDC 2017, arXiv:1703.07373
* Mechanism:
  * Plan with a simple model and track with a high-fidelity one.
  * Offline, Hamilton-Jacobi reachability on the relative pursuit-evasion game gives a
    TRACKING ERROR BOUND (TEB) and a safety controller.
  * Obstacles are Minkowski-inflated by the TEB ("a 'safety bubble' around the planning
    system"). The safety controller takes over near the bound; any controller can run
    inside it.
* Evidence (a 10D quadrotor tracking a 3D RRT planner): TEB "a box of V = 0.81 m in each
  direction" for a 0.5 m/s planner; each online iteration ~25 ms.
* For RL_Surf: the formal answer to the ledger's zero-margin skim. Measure the executor's
  worst-case deviation from plans of a given family, then plan against hazards inflated by
  that amount. We cannot solve the HJ game, but we can MEASURE the bound, because every
  logged plan gives a cross-track error.

#### RTD: Kousik, Vaskov, Bu, Johnson-Roberson, Vasudevan, "Bridging the Gap Between Safety and Real-Time Performance in Receding-Horizon Trajectory Design for Mobile Robots", IJRR 2020, arXiv:1809.06746
* Mechanism:
  * An "offline Forward Reachable Set (FRS) computation of a robot's motion when tracking
    parameterized trajectories; the FRS provably bounds tracking error". The bound is
    obtained by sums-of-squares, with tracking error a polynomial in time.
  * Online, obstacles map to unsafe plan PARAMETERS; a nonlinear program picks a safe
    parameter vector, with braking as the fallback.
  * "Persistent feasibility is achieved by prescribing a minimum sensor horizon and a
    minimum duration for the planned trajectories".
* Evidence: "safe and persistently feasible across thousands of simulations and dozens of
  real-world hardware demos". Replans every 0.5 s (Segway) and 0.375 s (Rover); Segway
  horizon 1 s at 1.25 m/s. The comparison success rates were not readable (NOT FOUND).
* For RL_Surf: the planner chooses PARAMETERS of a trajectory family whose tracking error
  was characterised offline. "Take off earlier or later, higher or flatter" is exactly
  such a parameter vector.

**Verdict for 3.2.** Three layers, cheapest first:
1. **Validity by construction.** The planner may only propose plans drawn from the
   executor's own repertoire (Setter-Solver's validity, NoMaD and RECON sampling), plus an
   uninformed share.
2. **Feasibility by measurement.** Certify with a noise-free executor rollout (HAC's
   testing, PDM-Closed), inflate hazards by the measured tracking error (FaSTrack, RTD),
   and gate the planner's reward multiplicatively (Plan-R1, SSE's stop-on-failure).
3. **Difficulty by learning progress.** Reward the planner for plans that are feasible but
   not yet mastered (AMIGo's t*, PAIRED's regret, ALP). This is the published way to
   punish both impossible and trivial plans with a single term.

### 3.3 Learned reachability used to prune plans: capability models from the controller's own rollouts

Already covered: Roth et al. 2025 FDM + MPPI (+27% navigation success, [NAV] 2.1),
PRM-RL (edges accepted at >= 85% of 20 rollouts), RL-RRT, SoRB, SGM and TTGS ([HIER]
12-13). New:

#### Kim, Kim, Hwangbo, "Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation", RSS 2022, arXiv:2204.08647
* Mechanism:
  * The controller is an RL command-tracking locomotion policy (ANYmal C, in simulation).
  * A forward dynamics model (FDM), an LSTM over lidar, state history and a 12-step
    command sequence, predicts future x,y and collision probability. It is trained on the
    controller's own rollouts under random command sequences.
  * A sampling MPC draws candidates from a time-correlated random sampler plus a CVAE
    "informed trajectory sampler" trained on 160K rollouts (the released test config: 100
    CVAE samples, CVAE latent 16; 729 samples in the configuration without the CVAE).
  * Score: R_track = exp(-DTW(global path, predicted xy) / tau), plus R_safety = the mean
    of (1 - p_collision).
  * Horizon 12 steps x 0.5 s = 6 s; planning at 2 Hz; the global path (BIT*) is truncated
    4.8 m ahead.
* Evidence:
  * 83.2% success vs 45.2% for a PD waypoint follower at 0.43 obstacles/m.
  * Ablations: approximate kinematic model 76.5%; random sampler only 73.6%; informed
    sampler only 63.2%.
  * 98.2% vs 94.6% at 0.2 obstacles/m.
  * FDM collision accuracy 94.6%, 0.1 m per-step position error, more than 20,000x faster
    than full simulation.
* Stated limit: simulation only.
* For RL_Surf: the closest single paper to stages (b)-(c).
  * A capability model learned from the executor's own random-command rollouts.
  * A LEARNED proposal sampler trained on the executor's own trajectories, MIXED with an
    uninformed sampler. The ablation shows the mix beats either alone: 83.2 vs 73.6 vs
    63.2.
  * Scoring by progress along a global path plus predicted failure.

#### Yang, Wellhausen, Miki, Liu, Hutter, "Real-time Optimal Navigation Planning Using Learned Motion Costs", ICRA 2021
* Mechanism: a CNN+MLP maps a 2 x 2 m height scan and a motion (dx, dy, dpsi, psi_a) to
  the RL locomotion controller's normalised energy, time and FAILURE RISK. These become
  edge costs in a GPU roadmap searched with A*.
  * Edges with risk above R_max = 0.5 are pruned.
  * c(e) = w_E c_E + w_T c_T + w_R c_R, with w_E = 5, w_T = 5, w_R = 100.
  * Trained on 370k samples, each averaged over 12 repeated attempts (R = the empirical
    failure probability).
* Evidence: on rough terrain, the optimised path cost is 30.32 vs RRT*'s 62.75, found in
  0.52 s vs 141.2 s ("three orders of magnitude faster").
* Stated limit: "probabilistically incomplete and can fail in narrow environments due to
  aliasing".
* For RL_Surf: repeat each labelled motion (12x) so that the label is a PROBABILITY, not a
  coin flip. That matters for a chaotic simulator, where one depth pixel forks a
  trajectory (ours, the cross-card bit-exactness note in CLAUDE.md).

#### Guzzi, Chavez-Garcia, Nava, Gambardella, Giusti, "Path Planning With Local Motion Estimations", RA-L 5(2), 2020
* Mechanism: a CNN over (heightmap patch, relative target) predicts the success
  probability y_S and duration y_T of a local motion. RRT* / SST planners assign each edge
  the risk -log y_S and discard edges below a threshold tau (0.5 for ANYmal, 0.98 for
  Thymio). Path risk is the sum of edge risks.
* Evidence:
  * AUC 0.889 (ANYmal) and 0.972 (Thymio). At tau = 0.98, precision 99.7% and recall 96.1%.
  * Executed ANYmal paths arrived safely ~80% of the time.
  * One path succeeded 16/30 = 0.53 against a predicted 0.57, i.e. calibrated.
* Data: ~90K samples of the robot's own controller in Gazebo (76% successes); Thymio 104K
  real samples in 120 min.
* Stated limits: it ignores "the sequential nature of following a path", and durations are
  "systematically underestimated".
* For RL_Surf: -log(success) summed along a plan is the right path cost for a chain
  (PRM-RL's product law, in log space). The stated limit is our stitching fallacy again:
  per-edge estimates ignore the state the previous edge leaves you in.

#### Chavez-Garcia, Guzzi, Gambardella, Giusti, "Learning Ground Traversability from Simulations", RA-L 3(3), 2018, arXiv:1709.05368
* Mechanism: a simulated robot drives on procedurally generated heightmaps. A patch is
  labelled traversable if the robot advances more than d = 0.12 m within T = 1 s. There are
  450k samples (~27 km) on 30 terrains.
* Evidence: ACC/AUC 0.926 / 0.970 on synthetic terrain; 0.819 / 0.840 on a real quarry,
  vs 0.723 / 0.762 for a feature-based baseline. Evaluating a map is ~1000x faster than
  simulating it (350 ms vs 320 s).
* For RL_Surf: the labelling rule is "did the controller make progress within T". That is
  a physics-scale, map-free definition of an edge, set once.

#### WVN: Frey, Mattamala, Chebrolu, Cadena, Fallon, Hutter, "Fast Traversability Estimation for Wild Visual Navigation", RSS 2023, arXiv:2305.08510
* Mechanism: an online self-supervised label, sigmoid(-k(v_error - v_thr)), from the gap
  between the commanded and the achieved velocity. The label is reprojected into past
  images, and a small MLP on DINO-ViT features learns it.
* Evidence: under 5 min of in-field training; accuracy 81.05 / 82.45 / 78.21% (hilly,
  forest, grass); 8/8 goals reached.
* For RL_Surf: the label is "did the controller achieve what it was commanded", which for
  us means "did the executor complete the plan". It is learnable online, while training.

#### Value Function Spaces: Shah, Xu, Lu, Xiao, Toshev, Levine, Ichter, ICLR 2022, arXiv:2111.03189
* Mechanism: the high level's state is Z(s) = [V_o1(s), ..., V_ok(s)], the value
  functions of its k skills. On top of it runs Double DQN, or random-shooting MPC with a
  one-step model.
* Evidence:
  * MultiRoom 2 / 4 / 6 / 10 success: 0.98 / 0.92 / 0.83 / 0.77, vs 0.64 / 0.46 / 0.42 /
    0.29 on raw observations.
  * Zero-shot on MR4 / MR10: 0.87 / 0.67 vs 0.47 / 0.20.
* For RL_Surf: the executor's own critic, evaluated on the candidate plans, is a natural
  input for the planner ("how well would I follow plan k from here"). It needs no extra
  training, but it is reliable only on plans the executor was trained on (SoRB's
  wormholes).

#### SayCan: Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", arXiv:2204.01691, 2022
* Mechanism: choose the skill maximising p(success | s, skill) x p(skill useful |
  instruction), i.e. usefulness MULTIPLIED by the skill's value function.
* Evidence: over 101 instructions, 84% planning and 74% execution success. Without the
  value functions, planning falls to 67%.
* For RL_Surf: MULTIPLY the planner's preference by the executor's success probability;
  do not add them.

#### GNM: Shah, Sridhar, Bhorkar, Hirose, Levine, "GNM: A General Navigation Model to Drive Any Robot", ICRA 2023, arXiv:2210.03370
* Mechanism: a goal-conditioned policy predicts tau = 5 NORMALISED future waypoints
  (scaled by each robot's top speed) plus a temporal-distance head. The temporal distance
  is "a measure of traversability": graph "edges are computed using the temporal distance
  estimates", and routes are found with Dijkstra.
* Evidence:
  * One policy across robots: LoCoBot 0.96, Tello 0.99, Vizbot 0.93, Jackal 0.94, vs
    0.26-0.79 when trained on a single dataset.
  * In moderate environments, VELOCITY actions scored 0.54 and normalised WAYPOINTS 0.95.
* Data: ~60 h of teleoperated AND autonomous driving. All labels are hindsight (positives
  from the same trajectory), so the executor's own rollouts would do.
* For RL_Surf: evidence for the interface (waypoints over velocities), and a recipe for a
  reachability head trained from the executor's own flights.

#### ViNT: Shah, Sridhar, Dashora, Stachowicz, Black, Hirose, Levine, "ViNT: A Foundation Model for Visual Navigation", CoRL 2023, arXiv:2306.14846
* Mechanism:
  * An image diffusion model proposes K subgoal images.
  * Each is "grounded" by ViNT's temporal distance and action rollouts.
  * An A*-like frontier ranks them by f = d_M(o_t, s-) + d_pred(s-, s) + h(s, G).
  * On invalid samples: "Samples from the diffusion model may be invalid subgoals, but ViNT
    is robust to such proposals".
* Evidence: indoor 0.94 vs 0.81 for ViNT-R (random dataset images as the candidates), 0.72
  for end-to-end BC, 0.19 for RECON. Outdoor 1.00 vs 0.61 / 0.44.
* Constants: 5 action waypoints; context 5; DDIM 200 steps; max distance 20; lambda 0.01.
* For RL_Surf: the published pattern of generate -> ground with the executor's temporal
  distance -> search. The ViNT-R ablation (0.94 vs 0.81) measures what a generative
  proposer adds over proposing random states from the buffer.

#### RECON: Shah, Eysenbach, Rhinehart, Levine, "Rapid Exploration for Open-World Navigation with Latent Goal Models", CoRL 2021, arXiv:2104.05859
* Mechanism:
  * A latent goal model with an information bottleneck (beta = 1.0) is CONDITIONED ON THE
    CURRENT OBSERVATION, so a sample z ~ N(0, I) from the prior is a RELATIVE, reachable
    goal ("the representation ... only encodes information about relative location of the
    goal from the context").
  * Exploration samples subgoals from that prior.
  * A topological graph stores predicted temporal distances (thresholds delta_1 = 4,
    delta_2 = 15).
* Evidence: in 8 unseen environments, goals up to 80 m away were found in 9 min 54 s on
  average, 50% faster than the best baseline, then reached in 26 s; >30% higher weighted
  success.
* Data: "over 5000 self-supervised trajectories" in 9 environments over 18 months,
  collected by a time-correlated random walk, NOT teleoperation. This is the kind of data
  permitted here.
* For RL_Surf: the cleanest recipe for "proposals from the repertoire". A generative model
  of where the executor can get to FROM HERE, trained on its own trajectories, sampled for
  exploration and scored by a learned temporal distance.

#### Feasibility-Guided Planning over Multi-Specialized Locomotion Policies: Luo, Wang, Mandala, Chou, Christmann, Chen, Chan, Lee, Chen, ICRA 2026, arXiv:2602.07932
* Abstract only: each terrain-specific locomotion policy is "paired with a learned
  predictor that generates feasibility tensors from elevation maps and task information,
  enabling classical planners to derive optimal paths". Training data and numbers NOT
  FOUND.
* For RL_Surf: the 2026 state of practice is still "one capability predictor per
  controller, consumed by a classical planner".

#### Alonso, Peter, Goumard, Romoff (Ubisoft La Forge), "Deep Reinforcement Learning for Navigation in AAA Video Games", arXiv:2011.04764, 2020
* Why navigation meshes fail: movement abilities (grapples, jetpacks, double jumps) need
  hand-placed "links" that are "expensive to build" and do not correspond to traversable
  space.
* Method: a SAC agent with 3D occupancy, depth and the goal. Random goals are drawn inside
  a radius that GROWS by curriculum until it covers the map. Dense distance-to-goal reward
  plus a time penalty.
* Evidence: 100% success on a 120 x 120 x 30 m map, 90% on 300 x 300 x 100 m. It does not
  generalise to unseen maps (retrained per map).
* For RL_Surf:
  * The closest domain in the literature: a game character whose movement abilities make
    free-space planning wrong. The answer there is the same: the controller learns the
    reachability the geometric planner cannot represent.
  * It also shows the limit: per-map training.

**Summary of how robotics learns what the controller can do.**

| paper | inputs | predicts | data (all from the controller itself) | used as |
|---|---|---|---|---|
| Kim 2022 | lidar, state history, 12-step command | future xy, collision probability | random-command rollouts; 160K for the sampler | MPPI-style scoring at 2 Hz |
| Yang 2021 | 2 x 2 m height scan, motion | energy, time, failure risk | 370k motions x 12 repeats | A* edge cost; prune at R > 0.5 |
| Guzzi 2020 | heightmap patch, relative target | success probability, duration | ~90K own motions (sim), 104K (real) | -log p edge cost; prune at tau |
| Chavez-Garcia 2018 | oriented patch | traversable in 1 s | 450k simulated traversals | map + Pareto paths |
| WVN 2023 | image features | traversability | online, velocity-tracking error | cost map, learned in < 5 min |
| ABS 2024 | state, obstacles | reach-avoid value | 200k own episodes | switch to recovery at V > -0.05 |
| Roth 2025 [NAV] | geometry, history, command sequence | future state, failure probability | years of sim rollouts | zero-shot MPPI |
| ours, as proposed | executor observation + plan | completes / time / clearance / cross-track | executor rollouts under random plans | certificate + margin + planner input |

### 3.4 Trajectory proposals from the executor's own repertoire, filtered for diversity

#### 3.4.1 Vocabulary + scorer: the planners of autonomous driving

* **MultiPath**: Chai, Sapp, Bansal, Anguelov, CoRL 2019, arXiv:1910.05449.
  * A FIXED set of anchor trajectories obtained by k-means ("we used the k-means algorithm
    as a simple approximation"): K = 16 (mu, Sigma) or 64 (mu). The network outputs a
    softmax over anchors plus a Gaussian offset per anchor.
  * Where k-means gives "highly redundant clusters", they use hand-uniform anchors instead:
    64 = 16 orientations x 4 distances.
  * Horizon 6 s at 5 Hz. Motivation: learned mixtures "suffer from issues of mode
    collapse".
  * Waymo internal: MultiPath minADE5 0.63, vs 1.41 for plain regression.
* **CoverNet**: Phan-Minh, Grigore, Boulton, Beijbom, Wolff, CVPR 2020, arXiv:1911.10298.
  * Prediction is CLASSIFICATION over a trajectory set, built in one of two ways.
  * A FIXED set: 20,000 training trajectories reduced by greedy set cover, so every
    trajectory lies within eps of a member under the max point-wise distance.
  * A DYNAMIC set: integrate a vehicle model from the current state over "a diverse set of
    constant lateral and longitudinal accelerations". Every member is feasible by
    construction.
  * Set sizes for 6 s trajectories: eps = 8 / 5 / 4 / 3 / 2 m -> 64 / 232 / 415 / 844 /
    2,206; dynamic eps = 3 m -> 357; hybrid 774. Coverage "under 2 meters ... with fewer
    than 2,000 elements".
  * HitRate5 at 2 m: fixed eps = 2 m 0.24, dynamic 0.33, vs MultiPath-64 0.10.
  * Limit: metrics "plateau, or even decrease at around 500-1,000 modes".
* **TNT**: Zhao et al., CoRL 2020, arXiv:2008.08294.
  * Over-sample goal TARGETS from the map (N = 1000, e.g. one per metre of lane), keep the
    top M = 50, regress one trajectory per target, then select K = 6 greedily: keep a
    trajectory "if [it] is distant enough from all the selected trajectories".
  * Target spacing 5.0 / 2.0 / 1.0 / 0.5 m gives minFDE6 1.55 / 1.31 / 1.29 / 1.29, so
    ~1 m is enough.
  * The NMS threshold is NOT FOUND.
* **MTR**: Shi, Jiang, Dai, Schiele, NeurIPS 2022, arXiv:2209.13508.
  * "64 motion query pairs where their intention points are generated by conducting k-means
    clustering algorithm on the training set", then NMS: "select top 6 predictions from 64
    predicted trajectories ... the distance threshold is set as 2.5 m".
  * Validation mAP 0.4164, vs 0.3245 for MTR-e2e (6 learned queries, no NMS); the paper's
    Fig. 4 credits the static intention points with "much better mAP and miss rate".
* **Wayformer**: Nayakanti, Al-Rfou, Zhou, Goel, Refaat, Sapp, ICRA 2023, arXiv:2207.05844.
  * 64 mixture modes reduced to 6 by greedy covering: "select the fewest centroid modes
    such that all output modes are within a final distance D away", then 3 refinement
    iterations; D = 2.3 on WOMD.
  * WOMD minADE 0.545, mAP 0.419.
* **MotionLM**: Seff et al., ICCV 2023, arXiv:2309.16534.
  * Sample discrete motion tokens autoregressively (169 tokens at 2 Hz), up to 512 rollouts
    per replica, then aggregate by NMS clustering.
  * mAP by rollout count: 1 -> 0.0578, 8 -> 0.1324, 64 -> 0.1585, 512 -> 0.1687. Many
    samples, then clustering, is what works.
* **PRIME**: Song, Luan, Ding, Wang, Chen, CoRL 2021, arXiv:2103.04027.
  * A MODEL-BASED generator builds candidates as longitudinal quartic x lateral quintic
    polynomials in the Frenet frame: 35 terminal speeds x 9 lateral offsets, filtered by
    kinematics (v <= 33.33 m/s, |a| <= 8 m/s^2, kappa <= 0.33) and collision, giving "484
    feasible trajectories on average".
  * A learned evaluator scores them; greedy NMS reduces to K = 6.
  * Infeasibility 0.00% vs 16.52% for LaneGCN, at a worse minADE (1.22 vs 0.87).
* **PDM-Closed**: Dauner, Hallgarten, Geiger, Chitta, "Parting with Misconceptions about
  Learning-based Vehicle Motion Planning", CoRL 2023, arXiv:2306.07962.
  * Proposals: "IDM policies at five distinct target speeds, namely, {20%,40%,60%,80%,100%}
    of the designated speed limit ... three lateral centerline offsets (+-1m and 0m),
    thereby producing N=15 proposals".
  * Each is simulated "4 seconds at a 10Hz" THROUGH THE CONTROLLER (LQR + kinematic
    bicycle), which lets the planner "correct drift that may arise when the controller
    fails to accurately track the intended trajectory".
  * Scoring: multiplicative (at-fault collision, drivable area, direction) x weighted
    (5 TTC + 2 comfort + 5 progress)/12. Progress is normalised by the best proposal free
    of multiplicative infractions.
  * nuPlan CLS-R: 92, vs 72 for the best LEARNED planner (PlanCNN) and 77 for IDM alone.
  * Ablation: removing the lateral offsets gives 89, removing the speed variants 88.
  * A rule-based planner, no training data.
* **VADv2**: Jiang, Chen, Gao, Liao, Zhang, Liu, Wang, arXiv:2402.13243 ("Accepted to ICLR
  2026").
  * A 4,096-entry vocabulary chosen by "furthest trajectory sampling" from demonstrations;
    the network outputs one probability per entry.
  * A "conflict loss" treats entries that hit road boundaries or other agents as
    negatives. Why a distribution: regression may "produce an intermediate action" when
    "the feasible solution space is non-convex".
  * Vocabulary 256 vs 4096: collision@3s 0.057 vs 0.039.
* **Hydra-MDP**: Li et al., arXiv:2406.06978, 2024 (1st place, CVPR 2024 challenge).
  * A k-means vocabulary of 4,096 or 8,192 trajectories. A teacher runs OFFLINE SIMULATION
    of every vocabulary entry to get one score per metric, and the network learns one BCE
    head per metric.
  * navtest PDMS: 83.0 (8,192) vs 80.2 when distilling the single total score; 86.5 in the
    best variant; PDM-Closed with ground-truth perception 89.1.
* **DiffusionDrive**: Liao et al., CVPR 2025, arXiv:2411.15139.
  * 20 k-means anchors plus TRUNCATED diffusion (2 denoising steps from an anchored
    Gaussian).
  * Vanilla diffusion: "different random noises converge to similar trajectories", with
    diversity 11%. Truncated and anchored: 74%.
  * PDMS 88.1 at 45 FPS, vs 84.6 at 7 FPS for vanilla diffusion.
* **Werling, Ziegler, Kammel, Thrun**, "Optimal Trajectory Generation for Dynamic Street
  Scenarios in a Frenet Frame", ICRA 2010.
  * Sample end states (lateral offset d_i x duration T_j, or target speed x T_j) and
    connect them with jerk-optimal polynomials. Cost C = k_j J + k_t T + k_d d_1^2.
  * Filter out candidates that exceed acceleration or curvature limits; send the cheapest
    survivor to the tracking controller.
  * Replan every 100 ms, starting from the previous plan so that "no discontinuities
    occur". Numeric weights NOT FOUND.

#### 3.4.2 Generative proposals (diffusion, masked goals, and infeasibility detection)

* **NoMaD**: Sridhar, Shah, Glossop, Levine, "NoMaD: Goal Masked Diffusion Policies for
  Navigation and Exploration", ICRA 2024, arXiv:2310.07896.
  * A diffusion head over the future action sequence, with the goal token dropped at p_m =
    0.5. With the goal masked it samples the robot's own undirected behaviour: "a bimodal
    distribution of collision-free actions in the absence of a goal". 10 denoising steps.
  * Exploration success 98% with 0.2 collisions, vs subgoal diffusion (ViNT) 77% / 1.7 and
    random subgoals 70% / 2.7. Goal-conditioned navigation 90%.
  * For RL_Surf: goal masking is literally "what I usually do from here", the undirected
    half of a proposal set.
* **Diffuser**: Janner, Du, Tenenbaum, Levine, ICML 2022, arXiv:2205.09991.
  * Diffuses states and actions jointly; goals by inpainting; receding-horizon execution.
    It generalises by "stitching together in-distribution subsequences".
  * Maze2D U / M / L: 113.9 / 121.5 / 123.0 vs IQL 47.4 / 34.9 / 58.6. Its Maze2D data is
    "undirected - ... a controller navigating to and from randomly selected locations":
    the nearest published analogue of an executor's own random-target rollouts.
  * Diffuser plans can pass through walls (shown by the restoration-gap paper).
* **Decision Diffuser**: Ajay, Du, Gupta, Tenenbaum, Jaakkola, Agrawal, ICLR 2023,
  arXiv:2211.15657.
  * Diffuses STATES only, and executes them with an inverse-dynamics model a_t = f(s_t,
    s_t+1). Actions "tend to be more high-frequency and less smooth".
  * Hopper-Medium: states-only 79.3 vs states+actions 66.3 vs Diffuser 58.5. That is a
    planner(states) / executor(actions) split, measured.
  * Stated limit: it is "circumventing the need for exploration".
* **Hierarchical Diffuser**: Chen, Deng, Kawaguchi, Gulcehre, Ahn, "Simple Hierarchical
  Planning with Diffusion", ICLR 2024, arXiv:2401.02644.
  * A sparse diffuser generates every K-th state (K = 15 for long horizons); a low-level
    diffuser fills each segment between clamped endpoints.
  * Maze2D-Large 155.8 vs Diffuser 123.0; planning 3.3 s vs 9.9 s. Compositional
    out-of-distribution test 100% vs 0%.
  * K is "a task-dependent hyper-parameter": set it once for all maps, or it is a map
    constant.
* **HDMI**: Li, Wang, Jin, Zha, "Hierarchical Diffusion for Offline Decision Making", ICML
  2023 (PMLR 202). A reward-conditional subgoal diffuser plus a goal-conditional trajectory
  diffuser. The body was not readable here; numbers are secondary (Maze2D 120.1 / 121.8 /
  128.6, from HD's table).
* **Restoration gap (RGG)**: Lee, Kim, Choi, "Refining Diffusion Planner for Reliable
  Behavior Synthesis by Automatic Detection of Infeasible Plans", NeurIPS 2023,
  arXiv:2310.19427.
  * gap = E ||tau - restore(perturb(tau))||: noise a finished plan to diffusion time t-hat
    = 0.9, denoise it, and measure how far it moved. Infeasible artifacts ("passing through
    walls") move more.
  * A gap predictor trained on the model's own samples guides generation away from them.
  * Maze2D-Large 123.5 -> 135.4 (RGG) -> 143.9 (RGG+).
  * For RL_Surf: an infeasibility detector that needs NO feasibility labels. It needs
    "transition data that uniformly covers the state-action space", which the executor's
    rollouts do not.

#### 3.4.3 Latent skill spaces searched by CEM or beam search

* **OPAL**: Ajay, Kumar, Agrawal, Levine, Nachum, ICLR 2021, arXiv:2010.13611.
  * A VAE over c = 10-step sub-trajectories (latent 8, KL weight beta = 0.1); a task policy
    picks the latent.
  * AntMaze large-diverse: CQL 14.9 -> 70.3 with OPAL. With c = 1 there is no gain.
* **Play-LMP**: Lynch, Khansari, Xiao, Kumar, Tompson, Levine, Sermanet, CoRL 2019,
  arXiv:1903.01973.
  * A latent plan space with a plan-proposal prior and a closed-loop decoder that replans
    at ~1 Hz. 85.5% on 18 tasks from states.
  * Its data is human VR teleoperation: FORBIDDEN here. Only the architecture transfers.
* **SkiMo**: Shi, Lim, Lee, CoRL 2022, arXiv:2207.07560.
  * An H = 10-step skill VAE plus a skill-DYNAMICS model; CEM over skills (512 samples, 64
    elites, 6 iterations).
  * H = 1-5 gives too little abstraction; 15-20 is "much worse in Maze". 5x fewer
    interactions than SPiRL on Kitchen.
* **LSP**: Xie, Bharadhwaj, Hafner, Garg, Shkurti, ICLR 2021, arXiv:2011.13897.
  * CEM over 3-dimensional skills held for K = 10 steps inside a Dreamer model (G = 16
    candidates, top M = 4), with a mutual-information term keeping skills distinct.
  * In transfer, it reached a return of 500 in 70k steps vs Dreamer's 130k.
* **TAP**: Jiang, Zhang, Janner, Li, Rocktaschel, Grefenstette, Tian, ICLR 2023,
  arXiv:2208.10291.
  * VQ-VAE codes over L = 3-step chunks, and beam search under an autoregressive prior with
    score = return + alpha ln(min(p(z | s), beta^M)).
  * The prior floor beta = 0.05 is an out-of-distribution filter: plans the agent "has not
    done" are rejected. Performance is flat over beta in [0.002, 0.1] and -30% at 1e-5.
* Verdict: about 10-step chunks won repeatedly (OPAL, SkiMo, LSP). The TAP prior floor is
  a label-free "have I done this" test.

#### 3.4.4 Diverse NEAR-OPTIMAL behaviour: "take off earlier or later, higher or flatter"

* **SMERL**: Kumar, Kumar, Levine, Finn, NeurIPS 2020, arXiv:2010.14484.
  * A DIAYN diversity reward added ONLY when a latent's return is within epsilon of
    optimal: R + alpha 1[R >= R* - eps] sum r~. Constants: alpha = 10, eps = 0.1 R*, |Z| = 5
    latents; at test time, each latent runs 1 episode and the best is kept.
  * The best training policy can be among the worst under perturbation (-424.1 vs -222.7).
    It finds "distinct paths to the goal" in 2D navigation.
* **DOMiNO**: Zahavy et al., ICLR 2023, arXiv:2205.13521.
  * Maximise diversity (successor-feature distance to the nearest other policy) subject to
    each policy keeping >= alpha of the optimal value (alpha = 0.9, sets of 10). A Van der
    Waals-style reward stops pushing once policies are l_0 apart.
  * 1.5-2.0x better than the baseline under large perturbations.
* **DSF**: Yuan, Kitani, "Diverse Trajectory Forecasting with Determinantal Point
  Processes", ICLR 2020, arXiv:1907.04967.
  * A DPP over N latents of a frozen CVAE, with a trajectory-space similarity exp(-k d^2),
    and a greedy DPP MAP choosing "the most diverse subset".
  * ADE / ASD 0.182 / 0.147 vs 0.262 / 0.022 for plain CVAE sampling. Diversity measured
    in the LATENT space was worse.
* For RL_Surf:
  * The user's "take off earlier or later, fly higher or flatter but faster" is a set of
    NEAR-OPTIMAL alternatives: SMERL's eps-optimality gate and DOMiNO's alpha-optimality
    constraint formalise "reasonable".
  * DSF says to measure diversity in the space of the PLANS (polylines), not in a latent
    space.

#### 3.4.5 Learned samplers and retrieval

* **Ichter, Harrison, Pavone**, "Learning Sampling Distributions for Robot Motion Planning",
  ICRA 2018, arXiv:1709.05448.
  * A CVAE conditioned on (start, goal, obstacles) learns where successful plans went and
    proposes samples there. Learned and uniform samples are mixed at lambda = 0.5 ("a
    satisfactory balance"); the uniform half is kept because completeness and asymptotic
    optimality depend on it.
  * "approximately an order of magnitude improvement in success rate and cost" over uniform
    sampling.
  * Failures: poor transfer to harder problems than trained on; learned samples can "cut
    corners".
  * For RL_Surf: lambda = 0.5 is the published default for how much of a proposal set may
    come from the learned prior. The rest must stay uninformed, or the search can never
    find a line the executor has not flown before, which is the whole problem at
    unitfarmer2's pit.
* **Learned Motion Matching**: Holden, Kanoun, Perepichka, Popa, ACM TOG (SIGGRAPH) 2020.
  * Games already do "proposals from the repertoire" at frame rate: motion matching
    retrieves the recorded segment whose current state and desired future trajectory best
    match the request. The learned version keeps motion matching's behaviour with "no need
    to store animation data or additional matching meta-data in memory" (abstract only).
  * The database there is motion capture (forbidden as a source). The mechanism,
    nearest-neighbour retrieval of executed segments keyed by (state, desired future),
    applies unchanged to the executor's own flights.
* **HRHC's library** (Liniger 2015, section 3.1): the same idea from control. The
  vocabulary is the body's own equilibria, including 26 drifting ones.

#### 3.4.6 Synthesis: a proposal machine trained ONLY on the executor's own rollouts

1. **Corpus.** Every executed segment of the executor, with its start state (position AND
   velocity) and its outcome. Every generator above is supervised in hindsight (future
   positions, frames, temporal distances), so executor rollouts can replace its dataset.
   Diffuser's Maze2D data, "a controller navigating to and from randomly selected
   locations", is exactly what stage (a) produces.
2. **Vocabulary.** Two routes:
   * cover or cluster the corpus in the ego frame, per speed band: k-means (MultiPath K =
     16-64, MTR 64, DiffusionDrive 20, Hydra 4,096-8,192), an eps-cover (CoverNet: 2 m ->
     2,206 entries) or farthest-point sampling (VADv2);
   * or GENERATE variants at the current state from the dynamics (CoverNet's dynamic set,
     PRIME's 35 x 9, PDM's 5 x 3, HRHC's equilibria). For us the "dynamics" is the
     executor running in the simulator.
3. **Candidates per decision.** Top vocabulary entries, plus perturbations (the user's
   take-off / height / speed axes, which are PDM's speed x lateral axes, both measured to
   matter: 92 -> 88 / 89 without one), plus an UNINFORMED half (Ichter lambda = 0.5; Kim
   2022's mix beat either sampler alone). Reduce to K by NMS or a DPP measured in plan
   space (MTR 64 -> 6 at 2.5 m; DSF).
4. **Score.** Simulate each candidate through the executor (PDM-Closed: 4 s at 10 Hz
   through the real controller). Score = multiplicative gates (survived; clearance >= the
   measured tracking error) x weighted soft terms (progress toward the target, normalised
   by the best surviving candidate, minus time).
5. **Learn.** Distil the per-term simulated scores into one network head per term
   (Hydra-MDP: 83.0 vs 80.2 when distilling the single total). Keep the vocabulary FIXED,
   because anchors are what prevent mode collapse (DiffusionDrive 11% -> 74% diversity).
6. **Hard limit.** Generators only recombine their data: Diffuser "stitch[es] ...
   in-distribution subsequences", RGG needs data that "uniformly covers the state-action
   space", and DD is "circumventing the need for exploration". A self-trained generator
   cannot propose a take-off the executor has never flown. Novelty must come from the
   uninformed half and from exploration at the plan level.

---

## 4. Robotics practice: deterministic planners over learned controllers

### 4.1 The layered stack, and why the tracker is separate

* **Paden, Cap, Yong, Yershov, Frazzoli**, "A Survey of Motion Planning and Control
  Techniques for Self-driving Urban Vehicles", IEEE T-IV 2016, arXiv:1604.07446.
  * The canonical layers: route planner -> "behavioral layer ... generates a motion
    specification" -> "A motion planner then solves for a feasible motion" -> "A feedback
    control adjusts actuation variables to correct errors in executing the reference path".
  * Why the tracker is separate: "The tracking errors generated during the execution of a
    planned motion are due in part to the inaccuracies of the vehicle model".
  * On planner models: a high-fidelity model "may complicate the planning and control
    problems", while a kinematic model "permits instantaneous steering angle changes which
    can be problematic if the motion planning module generates solutions with such
    instantaneous changes".
  * The planner is re-run "every time the model of the world is updated".
* **Fan et al.**, "Baidu Apollo EM Motion Planner", arXiv:1807.08048, 2018.
  * Path (DP over sampled rows joined by quintic polynomials, then a spline QP inside the
    resulting "feasible tunnel") and speed (DP, then QP) are solved separately in the
    Frenet frame. Each lane is planned in parallel, and a "cross-lane trajectory decider"
    picks one.
  * Feasibility enters as QP constraints on curvature, curvature rate, acceleration and
    jerk.
  * Horizon "at least an eight second or two hundred meter"; "less than 100 ms on average";
    3,380 hours and ~68,000 km of driving.
* **Werling 2010** (section 3.4.1): replan every 100 ms from the previous plan's state, "so
  that no discontinuities occur". This is the standard answer to replanning jitter.
* **PDM-Closed 2023** (section 3.4.1): a deterministic proposal-and-score planner beat
  every learned planner in closed loop (CLS-R 92 vs 72). The user's intuition that "the
  planner is actually deterministic" in robotics is the measured state of the art in
  driving.

### 4.2 Planners over LEARNED controllers: what is new beyond [NAV] 2

#### ANYmal Parkour, details [NAV] 2.2 and [HIER] 9 did not have: Hoeller, Rudin, Sako, Hutter, Science Robotics 9(88), 2024, arXiv:2306.14874
* Each skill is trained on a position + heading + TIME command, with "the distance-to-goal
  penalty is only activated on the last time-step". Termination penalties apply for falls
  and for contact forces > 2,500 N.
* The navigation policy runs at 5 Hz. It sees the perception latent, the goal position,
  the REMAINING TIME, the base velocity and the orientation. Its goal reward is paid on the
  last time step of the episode.
* The skills stay frozen. Stated limitation: "modifying one requires retraining the
  others".
* The navigator learns the skills' limits by rollout: boxes too high for the climb skill
  make it choose another route. Learned vs manual: 98.2% vs 95.3% (A), 96.3% vs 60.9% (B),
  97.6% vs 75.3% (C).

#### The other stacks, in one line each
* **DTC** (section 3.1): an optimiser plans and an RL tracker follows. It wins where the
  task reward is too sparse to find.
* **RLOC** (section 3.1): the footstep planner is trained with the controller in the loop.
* **ABS** (section 3.1): a capability model from the controller's own 200k episodes, used
  as a switch.
* **Kim 2022** (section 3.3): a capability model + a learned sampler + an uninformed
  sampler + scoring.
* **Haro 2026** (section 3.1): the A* path as a HINT to an RL local planner.
* **Lee 2024** [NAV]: Dijkstra -> 10 Hz learned navigator -> 50 Hz frozen locomotion, with
  novelty at the navigation level.
* **Skill-Nav** [NAV]: A* on walls only -> 2D waypoints -> a waypoint-conditioned policy.

### 4.3 The command interface: what works and why

| interface | best evidence | when it fails |
|---|---|---|
| velocity twist | legged_gym (the standard); Hwangbo 2019, Lee 2024 [NAV] | agile or sparse terrain: Zhang 2024's navigation formulation "far outperform[s]" it; GNM velocities 0.54 vs waypoints 0.95 |
| target position + time, reward at the end | Rudin 2022 IROS (gaps 0.15 -> 1.2 m) [NAV]; ANYmal Parkour; ABS; Zhang 2024 | the long route: it is a LOCAL interface and needs a planner above it |
| waypoints / short polyline | GNM (5), ViNT (5), Haro 2026 (15), AutoRL PF (98.7%), Skill-Nav [NAV], Kim 2022 (4.8 m of global path) | when the path is treated as a TRACK at the performance limit (Song 2023) |
| progress along a crude line | Song 2021 (within 5.2% of optimal on unflyable segments), Fuchs 2021, Linesight / Sophy [HIER], our xAUTO | does not by itself choose the route: the line has to exist |
| a full timed trajectory to track | DTC (with an exact optimiser) | model mismatch at the limit: 0% (Song 2023) |

**For RL_Surf:** a short polyline carrying time or speed, fed as waypoints (the fan), and
paid as progress plus an end-of-plan reward. That combines the rows that work and avoids
the row that fails at the limit.

### 4.4 How the planner learns what the controller can do: the recurring pattern

1. Freeze the controller.
2. Roll it out in randomised simulation under random commands or targets, repeating each
   one if the simulator is chaotic (Yang: 12x).
3. Label outcomes automatically: progress within T (Chavez-Garcia), success and time
   (Guzzi), energy, time and risk (Yang), future xy and collision (Kim), reach-avoid value
   (ABS), velocity-tracking error (WVN).
4. Regress from local geometry + the command to those labels. That takes 1e5-5e5 samples:
   Kim 160K, Guzzi 90K, Yang 370k, Chavez-Garcia 450k, ABS 200k episodes.
5. Plan with it, in one of three ways: as an edge cost (-log p, or a weighted sum), as a
   prune threshold (R_max = 0.5, tau = 0.5-0.98), or as a certificate that hands over to a
   recovery controller (V_threshold = -0.05).
6. Refit when the controller changes (Roth 2025; HRAC refits every 50k steps).

Our executor and simulator make step 2 cheaper than in any of these papers (600k env
steps/s, exact state restore), which is why certifying by direct simulation comes FIRST
here and the learned model second.

---

## 5. The user's design, sentence by sentence, against the literature

| the user's sentence | the literature's name for it | evidence | what to do |
|---|---|---|---|
| "a planner ... produce some potential movement directions, like a polyline" | waypoint / path-conditioned local policy | GNM waypoints 0.95 vs velocities 0.54; Haro 2026 (15 waypoints); AutoRL PF 98.7% | keep the fan on a per-env line; it is already 3D (fwd, left, up), so height is expressible |
| "the executor that actually executes this plan" | a command-conditioned controller trained on RANDOM commands | legged_gym (resampled every 10 s); ABS; Zhang 2024; Heess 2016 | stage (a) |
| "hierarchical reinforcement learning is unstable ... make it not collapse" | non-stationarity, infeasible subgoals, trivial subgoals, degenerate options | section 1 | order the training (F3); a clock, not learned termination |
| "penalize the executor for not executing the path" | the plan-tracking reward | tracking 44% / 0% vs RL progress 100% (Song 2023); no adherence term (Haro 2026) | pay progress along the plan + an end-of-plan bonus; off-corridor earns nothing |
| "penalize the planner for giving such hard or impossible trajectories" | subgoal testing, adjacency, two-sided reachability, a feasibility judge | HAC (lambda 0.3, -H); HRAC (eta 20); BrHPO (lambda_1); Setter-Solver; SSE; Plan-R1's gate | certify with a NOISE-FREE executor rollout; gate the planner's reward multiplicatively |
| "an egg and a chicken problem" | planner/policy consensus; warm-up order | GPS/BADMM; Mordatch-Todorov ADMM; MLSH warm-up; RLA | a throwaway planner first, a frozen executor second; consensus by construction (proposals from the executor's repertoire) |
| "progressively improve the trajectories such that they lead us towards the end goal" | curricula over goals and plans; expert iteration | ALP-GMM, PLR, PAIRED, Setter-Solver; ExIt [HIER] 19 | search -> distil -> search again (stage c) |
| "a polyline in 2D ... maximum like 600 to 1000 units" | plan horizon and representation | DHRL (look far, execute one hop); HiTS (timed subgoals); Kim 2022 6 s horizon; PDM 4 s; Sophy ~6 s [HIER] | set the length in SECONDS (2-4 s). 600-1000 u is exactly that at walking speed (242-345 u/s, ours) but only 0.2-0.5 s at 2,000-3,000 u/s. Add a time or speed per vertex |
| "the brain should be inside the planner" | frozen skills, with the navigator learning their envelopes | ANYmal Parkour: learned 96.3% vs manual 60.9% | keep the executor generic; route decisions live only in the planner |
| "the plan shouldn't be too much different from what the agent can do" | consensus / a likelihood prior / a tracking-error bound | the GPS constraint; LEAP's log-prior (lambda 0.1); RIS's KL (alpha 0.1); TAP's prior floor (beta 0.05); FaSTrack's TEB | draw proposals from the executor's own segments; margin = the measured tracking error |
| "randomize the trajectories and take only the ones that are different enough" | a vocabulary + NMS / eps-cover / DPP | CoverNet (eps 2 m -> 2,206 entries); MTR (64 -> 6 at 2.5 m); DSF; PDM's 15 | exactly this, with an uninformed half (Ichter lambda 0.5) |
| "take off earlier, later, higher, flatter but faster ... reasonable actions" | parametric trajectory families; near-optimal diversity | Werling / PRIME lattices (35 speeds x 9 offsets); PDM (either axis removed costs 3-4 points); HRHC's library; SMERL (eps 0.1 R*, 5 latents); DOMiNO (alpha 0.9) | perturbation operators on executed segments: shift the take-off along the arc, scale the speed profile, offset laterally |
| "a humanoid robot ... the planner tells it where to go" | the layered legged stack | [NAV] 2; DTC; ANYmal Parkour | - |
| "for robotics ... the planner is actually deterministic" | rule-based proposal + simulation scoring | PDM-Closed CLS-R 92 vs 72 for the best learned; Apollo EM; Skill-Nav A* [NAV] | stages (a)-(b) use a deterministic planner; learning comes last |
| "selecting some point in the map and drawing the path towards that point using BFS" | random targets + a global planner + a local policy | DD-PPO 99.9% on random point goals; AutoRL (P2P fails room-to-room, PF 98.7%); Haro 2026; Ubisoft's growing radius | stage (a), literally |

---

## 6. What does NOT transfer, stated plainly

* **Off-policy machinery.**
  * HIRO relabelling, HAC hindsight, HER, and every HRL method in section 2 except HiPPO,
    MLSH, DAC and SNN4HRL run on SAC or TD3 with replay.
  * What ports to our on-policy PPO: the SUPERVISED parts (reachability nets, graphs,
    judges, vocabularies) and the reward and penalty terms.
  * What does not: the relabelling that keeps an off-policy buffer consistent. For us,
    freezing and ordering replace it.
* **x,y goal spaces that ignore velocity.** Every Ant benchmark here hands the algorithm
  the torso's position. Surf reachability depends on speed, so every reachability label
  and graph node here must include velocity.
* **Per-environment constants.** HIGL (delta_pseudo, gamma_dist), DHRL ("the number of
  nodes, penalty, and c_h"), GCMR (delta_sg), DEOC (tau 0.0-0.7), Hierarchical Diffuser (K
  "task-dependent") and E3B ([DET]) were re-tuned per task. Under rule 0b each gets ONE
  value for all maps, or is not used.
* **Human data.** Driving logs trained MultiPath, CoverNet's fixed set, TNT, MTR,
  Wayformer, MotionLM, PRIME's evaluator, VADv2, Hydra-MDP's imitation head, DiffusionDrive
  and Plan-R1's pretraining. Teleoperation trained parts of GNM / ViNT / NoMaD, Play-LMP,
  and SkiMo's Kitchen and CALVIN tasks. Motion capture trained DSF and motion matching.
  Only their ARCHITECTURES transfer. Every corpus here must be the executor's own rollouts
  (rule 0). The hindsight-labelled ones (GNM, ViNT, NoMaD, Diffuser, OPAL) need nothing
  else.
* **Generators cannot invent.** Diffusion and VAE proposers recombine what they were
  trained on. On unitfarmer2 the executor has never flown the pit line, so no generator
  trained on its flights will propose it. The uninformed half and plan-level exploration
  are not optional.
* **Model-based trajectory optimisers.** GPS, DTC's TAMOLS, FaSTrack's HJ solve and RTD's
  sums-of-squares all need an analytic model. We have the exact simulator instead, which is
  better for certification and useless for gradients.
* **Time scales.** Legged and driving stacks plan at 2-10 Hz for bodies moving at 1-30
  m/s. At 3,000 u/s a surf player covers the length of a legged planner's whole horizon in
  a fraction of a second. Horizons must be expressed in seconds of travel, never in units.
* **One task.** DAC, Nachum 2019 and Barkour agree that hierarchy is roughly a flat agent
  on a single task. A planner that only ever solves one map proves nothing under rule 0b.
* **Figure-only evidence.** BrHPO, SSE, HIDI, GCMR, RIS, PAIRED and several others report
  learning curves only, over 5-10 seeds. Our program runs one seed. Treat their rankings
  as direction, not magnitude.

---

## 7. Recipe for RL_Surf

### 7.0 The shape, and what already exists in the code

**Executor.** The current PPO policy, trained from scratch in the default action space
(continuous absolute view + keys-hold), with the potential channel off (it is refused with
`--goals` anyway).
* **Input**, which exists: the lookahead fan on a per-env line (`--goals`,
  `surfgym.goals.MultiLine`, 27 scalars: 9 points x (fwd, left, up) in the ego frame, at
  geometric offsets from 125 to 3,000 u).
* **Reward**, which exists: `--goal-reward arc` (`surfgym/goalarc.py`). Signed arc progress
  along each env's own line, with a local window, a corridor gate (leaving the line
  freezes the coordinate instead of charging) and telescoping, so hovering nets zero.
* **Also existing:** `--goal-curriculum` (the 10-90% band), goal horizons in seconds
  (`--goal-kmin/kmax`), state restore in the core, `tools/certify_field.py`,
  `tools/explore_phase1.py`, and the recorder's deterministic mode.

**Missing** (a literature survey does not build them; this is only what the recipe
needs):
1. a per-env PLANNER that writes each env's line: BFS on the walkable grid to a target,
   re-run on a clock and on deviation;
2. a plan-outcome LOGGER;
3. a CERTIFICATION harness: fork the core, run the frozen executor greedily on K candidate
   plans, return outcomes;
4. a VOCABULARY builder from the logged segments;
5. later, a planner head.

**Rule compliance.**
* Plans come from map GEOMETRY (BFS) and from the policy's OWN flights. No human data
  (rule 0), and each ledger entry states the provenance.
* Every constant below is in seconds, hull widths, fractions or counts, and is set once for
  all maps (rule 0b).
* The labyrinth ladder is a TEST BED for the mechanism, never part of the recipe.

### 7.1 Stage (a): deterministic BFS planner, executor on plans to RANDOM targets (walking labyrinth)

**What.**
* **Planner.** BFS/Dijkstra on the walkable grid, from the agent's CURRENT cell to the
  target. The plan is the next L_plan of the path: L_plan = max(600 u, 3 s x current
  speed), i.e. about 750-1000 u at walking speed, matching the user's 600-1000 u.
  * Re-plan on a randomised clock, p ~ U{5,15} decisions (HiPPO; 0.2-0.6 s at 25 Hz).
  * Also re-plan on deviation: leaving the corridor, or 4 s without progress
    (Arena-Rosnav).
  * Because BFS runs from the agent's position, recovery from any deviation is automatic.
* **Targets.**
  * Uniform over reachable floor cells whose PATH distance lies in a band [d_lo, d_hi]
    measured in seconds of walking. d_hi widens by the existing 10-90% success-band rule
    (Florensa / GoalGAN [HIER]; legged_gym promotes and demotes on measured distance). PLR
    (beta 0.1, rho 0.1) or ALP-GMM are the generic alternatives.
  * 20% of targets are uniform over the whole map (ALP-GMM's p_rnd): the exploration
    floor.
  * Resample the target on arrival AND every ~10 s (legged_gym's resampling_time). Each
    episode then contains several plans and plan SWITCHES (SNN4HRL's failure).
* **Plan-quality mix.** 2/3 of plans are the shortest BFS path. 1/3 are deliberately
  suboptimal (a detour through a random intermediate cell, like Haro 2026's GBFS paths) or
  perturbed (vertices jittered by up to a few hull widths; Haro used 1 m). This teaches
  "the plan is a hint".
* **Reward.**
  * goalarc progress along the CURRENT plan (re-anchored on every re-plan), with the
    corridor set in hull widths;
  * plus an end-of-plan bonus when the target is reached. Either the existing arrival
    bonus, or ABS's two kernels 1/(1 + (d/sigma)^2) paid in the last T_r s, soft sigma and
    tight sigma, with sigma set in hull widths;
  * plus the time penalty and the death charge.
  * NO global potential.
* **Training.** Executor only (the planner is BFS), from scratch, one seed, same card as
  the controls.
* **Evaluation.** The planner's target is the finish, with shortest-path plans only, greedy
  episodes from the true start.

**Arms.**
* (i) lab100 alone and (ii) lab200 alone.
* (iii) 025 + 050 + 100 trained, 200 held out. This is the generalisation test the ladder
  arm could not do, because a new map has no easier siblings. Here the same map supplies
  easy sub-problems (near targets) automatically, which is the generic form of the ladder.
* The controls already exist: Euclid (stands at the wall), geodesic shaping (finishes), and
  ladder randomisation (finishes intermittently).

**Expected, and what falsifies it.**
* The executor + BFS planner should finish lab100 and lab200 from the start within the
  budget geodesic shaping needed (100-300M steps), and finish the held-out rung.
* If not, check first whether the plan is being read at all: zero the fan at eval; success
  should collapse.
* If the fan is read but finishes fail, look at the tracking statistics (arc completion,
  cross-track distribution, success against plan curvature). The executor, not the plan,
  is the bottleneck.

**Why first.**
* On a walking labyrinth free space IS feasible (walking is reversible; ours: the
  goal-rooted search covered every floor cell in a minute). So the planner is exact, and
  any failure belongs to the executor or to the interface.
* AutoRL, DD-PPO and Haro 2026 all predict success.
* The byproduct is the first CAPABILITY LOG: plan geometry, start (position, velocity),
  outcome.

### 7.2 Stage (b): surf, where free-space plans are infeasible: feasibility from the executor's own rollouts

**b1. The executor on surf maps** (the edgeflow ladder first, then the benchmarks), with the
same reward and a MIXED plan diet from three sources:
* (i) BFS to random targets. These are free-space plans, many infeasible; they become the
  NEGATIVE labels.
* (ii) HINDSIGHT plans: segments of the executor's own recent flights, re-issued from
  states near their start (HAC's hindsight actions, GCSL [HIER], Setter-Solver's validity).
  These are feasible by construction.
* (iii) PERTURBATIONS of (ii) along the user's axes: take-off shifted earlier or later
  along the arc, the speed profile scaled (higher-slower or flatter-faster), a lateral
  offset. These are the Werling / PRIME / PDM lattice axes, applied to the executor's own
  motion instead of a car model.

Mix learned (ii + iii) and uninformed (i) half and half (Ichter lambda = 0.5). The
perturbation magnitudes are proposals of this survey, not published values: set them once
in seconds and hull widths.

**b2. Log every plan's outcome.** Completed (reached the plan's end within its time),
time, minimum clearance to the kill plane and walls, maximum cross-track error, death. The
cross-track distribution per speed band IS the executor's tracking error bound (FaSTrack's
TEB, RTD's FRS), measured instead of solved.

**b3. The certification harness.**
* For a candidate plan at a state: fork the core (state restore), run the FROZEN executor
  GREEDILY (HAC's testing rule: no exploration noise), and record the same outcome.
* WHOLE plans only, from the REAL state. Never chain certified edges: that is the ledger's
  stitching fallacy, confirmed by training in efCERT_blue050.
* For robustness, add 5-20 rollouts from slightly perturbed states (PRM-RL: 20 rollouts at
  >= 85%; Yang: 12 repeats). The simulator is deterministic but chaotic, so one rollout is
  exact for one state and says nothing about its neighbours.

**b4. Selection, done the PDM-Closed way.**
* Candidates: top vocabulary segments near the current (position, velocity), plus their
  perturbations, plus BFS toward the target, reduced to K = 16-64 by eps-NMS in plan space.
* Simulate each for its length (2-4 s).
* Score = 1[survived] x 1[clearance >= q95 of the measured tracking error] x (progress
  toward the target / the best surviving candidate's progress) - time.
* Execute the best. Re-plan on the clock or on deviation, and keep the current plan unless
  a new one wins by a margin (hysteresis; Werling's plan stitching).
* **Targets:** the finish when a certified chain reaches it. Otherwise the FRONTIER of an
  archive of plan-reached states (Go-Explore in plan space, with MEGA's min-density or
  HIGL's coverage + novelty rule [DET]); BFS toward it is one proposal source among three.

**b5. The capability judge, SECOND.**
* A network J(observation, plan) -> P(complete), E[time], minimum clearance, trained by
  BCE and regression on the b2 and b3 outcomes. The literature used 1e5-5e5 labelled
  samples.
* Refit after every executor phase (HRAC every 50k steps; Roth 2025).
* Check calibration the way Guzzi did (16/30 = 0.53 observed vs 0.57 predicted).
* The judge amortises b3. It never replaces b3 for the final choice when the budget allows.

**Measure before any training arm** ([HIER]'s steerability rule).
* From a fixed state at blue050's decision point, do different candidate plans produce
  different outcomes, with a spread far beyond the policy's attractor width?
* If the executor cannot be steered, stage (c) is pointless.

**Verdict for stage (b).**
* On blue050, does sample-and-certify with the frozen executor find a finishing plan
  sequence from the true start, WITH margin?
* The references are ours: random bursts crossed in 28 min with zero margin; the clearance
  search did not cross in 40 min.
* This is the user's "search needs a learned prior" (ledger 2026-09-21 04:05), with the
  executor itself as the prior.

### 7.3 Stage (c): the learned planner, with joint training that cannot collapse

**c1. Distil the search.**
* The planner network sees the executor's observation + the target information + the
  executor's own critic values for the top candidates (VFS).
* It outputs a categorical over the candidate set (a vocabulary classifier, as in CoverNet
  and VADv2), with ONE HEAD PER SCORE TERM (Hydra-MDP).
* It is trained to predict b4's simulated per-term scores. This is ExIt's apprentice
  [HIER] 19 for plans.

**c2. Fine-tune the planner by PPO, with the executor FROZEN** (MLSH's warm-up; ANYmal's
navigator at 5 Hz over frozen skills).
* The planner acts at the plan rate, p ~ U{5,15}, as a semi-MDP. It is paid REALISED task
  reward: the finish bonus, time, and death forfeiting the bank (the Grzes death charge).
* A plan that fails certification gets a multiplicative zero (Plan-R1, SSE).
* A triviality bar: a plan must require >= t* seconds of progress (AMIGo: +0.7 / -0.3, t*
  +1 after 10 successes). ALP-weighted target sampling is the alternative.
* A coverage floor: keep the planner's entropy above a floor and monitor the number of
  distinct plans chosen (Setter-Solver's coverage).

**c3. Alternate, never co-train from zero.**
* Executor phase: fine-tune on a diet of 1/2 planner plans + 1/2 random / hindsight /
  perturbed plans. Use the same plan-following reward, which is a stationary objective
  whatever the planner's quality. Use a lower LR, PPO clipping, and a KL anchor to the
  previous executor on the random plans (RIS alpha = 0.1; GPS epsilon).
* Planner phase: re-certify the vocabulary with the new executor (capabilities changed;
  Roth 2025's refresh), re-distil, then PPO.
* Two timescales: the planner every iteration, the executor every N iterations.
* Keep plan TIMES fixed across an executor update, so the planner's data stays stationary
  (HiTS).
* HiPPO is the evidence that a joint PPO phase is safe once both levels exist. If a joint
  phase is tried, resample the commitment p ~ U{5,15}.

**c4. Exploration at the plan level.**
* Novelty or count bonuses over plan OUTCOMES (end states in position x speed), not over
  actions (Lee 2024 [NAV]; Nachum 2019).
* NoMaD-style goal-masked sampling (p_m = 0.5) as the undirected proposal, i.e. "what I
  usually do from here".
* Half the proposals stay uninformed.
* This is where unitfarmer2's pit line has to come from, because no generator trained on
  the executor's flights will propose it.

**c5. Try to distil the hierarchy away** (C-Planning; Barkour [NAV]). Train a flat policy on
the finished system's flights. If it matches, the flat policy is the recipe: simpler, and
preferred under the standing rules.

### 7.4 Constants: the published values, and a proposal set ONCE for every map

| decision | published values | proposed for RL_Surf |
|---|---|---|
| planner call rate | ANYmal Parkour 5 Hz over 50 Hz skills; Lee 2024 10 / 50 Hz; Kim 2022 2 Hz; RTD every 0.5 s / 0.375 s; Werling and FaSTrack 100 ms; PDM 10 Hz; HIRO, FuN, Heess c = 10; Director K = 8; HiPPO p ~ U{5,15} | clock p ~ U{5,15} executor decisions (0.2-0.6 s), + re-plan on leaving the corridor or 4 s without progress (Arena-Rosnav), + hysteresis |
| plan length / horizon | GNM, ViNT 5 waypoints; Haro 15; Kim 2022 6 s, global path 4.8 m ahead; PDM 4 s simulated, 8 s extended; VADv2 3 s / 6 waypoints; MultiPath 6 s; HRHC 0.35 s at 50 Hz; Sophy ~6 s [HIER] | 2-4 s of travel at the current speed, >= 600 u; the executor held to the first certified hop (DHRL); fan offsets unchanged (125-3,000 u) |
| plan content | HiTS: time per subgoal; Thakkar: target speed per checkpoint [HIER]; Swift: gate pose [HIER] | 3D ego points (the fan already has fwd/left/up) + a time or speed per vertex |
| executor reward | Song 2023: progress to the gate (b = 0.01 body rate); Fuchs: projection progress - 5e-4 v^2 on wall contact; ABS: 1/(1+(e/sigma)^2) in the last T_r (2 m / 2 s soft, 0.5 m / 1 s tight); Zhang: weight 10-25 in the last 2-4 s; legged_gym exp(-e^2/0.25); Meyer exp(-0.05 e) x speed along the path; MPCC progress - contour | goalarc progress along the current plan (corridor-gated) + an end-of-plan bonus (soft + tight kernel in the last 1-2 s) + time penalty + death charge; no flat off-plan penalty (Fuchs) |
| target sampling | legged_gym resamples every 10 s, promotes/demotes on distance; ABS goals 1.5-7.5 m, episodes 7-9 s; Zhang 1.5-4.9 m; AutoRL P2P 5-10 m; Ubisoft growing radius; Florensa / GoalGAN band 0.1-0.9 | a band on path distance (seconds), widened by the 10-90% rule; resample on arrival and every ~10 s; 20% uniform (ALP-GMM); PLR beta 0.1, rho 0.1 as the alternative |
| plan diet mix | Ichter lambda 0.5 learned:uniform; Florensa 2:1 new:old; GoalGAN 2/3 : 1/3; OpenAI Five 80/20 [ledger]; Haro optimal + suboptimal (ratio NOT FOUND) | executor: >= 1/3 random or perturbed plans at every stage; planner proposals: 1/2 uninformed |
| feasibility test | PRM-RL >= 85% of 20 rollouts [HIER]; Yang R_max 0.5 over 12 repeats; Guzzi tau 0.5 / 0.98; ABS V_threshold -0.05; HAC tests 0.3 of subgoals, penalty -H | a greedy executor rollout from the real state, + 5-20 perturbed-state rollouts; accept at >= 85% |
| margin | FaSTrack TEB 0.81 m for a 0.5 m/s planner; RTD tracking-error polynomial | q95 of the executor's measured cross-track error in the speed band, added to the kill plane and walls |
| proposal set | PDM 5 x 3 = 15; PRIME 35 x 9 -> 484 feasible; MTR 64 -> 6 by NMS at 2.5 m; MultiPath K 16-64; CoverNet eps 2-8 m -> 2,206-64; VADv2 4,096; Hydra 4,096-8,192; DiffusionDrive 20 anchors; NoMaD p_m 0.5, 10 denoising steps; Kim 100 CVAE samples (729 without the CVAE) | 16-64 candidates per decision after eps-deduplication; a vocabulary of 256-4,096 executed segments per speed band |
| planner penalties | BrHPO lambda_1, lambda_2 in {0.1..5}; HRAC eta 20; AMIGo +0.7 / -0.3, t* +1 per 10 successes; SSE c_dist 5, eta 0.2; Plan-R1 multiplicative gates, G = 4 | a multiplicative gate (zero reward for a plan that fails certification) + the AMIGo triviality bar; lambda_1 = 1 if a subtracted term is ever used |
| KL / anchors | RIS alpha 0.1; LEAP prior weight 0.1; TAP prior floor beta 0.05; HESS lambda_0 0.1 on the best-fitted 30% | executor fine-tune: PPO clip + KL to the previous executor at alpha 0.1, on random plans |
| capability data | Kim 160K; Yang 370k (x12); Guzzi ~90K; Chavez-Garcia 450k; ABS 200k episodes; HRAC refit every 50k steps | ~1e5 logged plans before the judge is trusted; refit after every executor phase |
| warm-up | MLSH W = 20 master-only iterations; DHRL 75 episodes before the graph is used; HIGL's pull off for 60K steps; RLA low-level-only warm-up | the executor trained to plateau on stage (a)/(b) plans before any learned planner; the planner trained against the frozen executor before any joint phase |
| chunk / skill length (only if a latent plan space is used) | OPAL c = 10; SkiMo H = 10; LSP K = 10 | 10 decisions (0.4 s) |

### 7.5 Failure modes to watch

| # | failure | its signature in our logs | literature | guard / monitor |
|---|---|---|---|---|
| 1 | the two learners chase each other | the planner's success on its own plans falls after each executor update | HIRO; HiTS ("HAC deteriorates quickly"); MLSH ("both sub-policies moving to the same goal point") | alternate phases; fixed plan times; the random-plan diet |
| 2 | trivial plans | realised progress per plan shrinks toward 0 while planner "success" -> 100% | RLA's degenerate solutions; option collapse | pay realised task progress; AMIGo's t*; ALP |
| 3 | impossible plans | the certification rate falls; executor deaths on planner plans | HAC ("unrealistic subgoals"); HRAC; BrHPO | a multiplicative gate; charge only under a NOISE-FREE executor (HAC: charging noisy misses gives "overly conservative subgoals") |
| 4 | proposal mode collapse | the K candidates lie within eps of each other | DiffusionDrive (11% diversity); MTR-e2e | a fixed vocabulary / anchors; NMS; coverage loss |
| 5 | stitching | a certified chain that no single rollout flies | ours (efCERT); SGM; Diffuser's "stitching together in-distribution subsequences" | certify WHOLE plans from the real state; velocity in every key |
| 6 | zero margin | the executor reproduces the plan a few units lower and dies | ours (efCERTt: 16 u lower); FaSTrack; RTD | margin = the measured tracking error |
| 7 | chaining decay | success = (per-plan success)^(plans per episode) | PRM-RL: 0.85^6.05 = 37% predicted vs 38% observed [HIER] | plans inside the executor's reliable reach (ours: ~99% at 1-5 s [HIER]); re-plan |
| 8 | a stale capability model | the judge says feasible and the rollout fails, more often after executor updates | Roth 2025 [NAV]; WVN learns online | refit after each executor phase; certification is the ground truth |
| 9 | re-plan dithering | plan identity changes on every call; speed is lost | the deliberation cost [HIER]; Werling's stitching; PTSP [HIER] | a hysteresis margin; random commitment |
| 10 | the executor overfits to the planner | success on RANDOM plans falls | SNN4HRL's switching failure; Barkour [NAV] | keep >= 1/3 random plans; evaluate on random plans every phase |
| 11 | a wrong plan followed faithfully | the executor follows BFS into the pit | Song 2023 (tracking 0%); Haro 2026 (hint) | pay progress, not tracking; suboptimal plans in training; the end-of-plan reward |
| 12 | the generator cannot propose the unflown | the detour never appears among the candidates | the Diffuser / RGG / DD data limits | 1/2 uninformed proposals; plan-level exploration |
| 13 | throughput | forked rollouts dominate wall clock | PDM: 15 x 4 s; Kim: FDM 20,000x faster than full simulation | certify on a subset of envs; amortise with the judge |
| 14 | the metric lies | eval_progress rises with no finish | ours (xROUTE, xARC, xPSSR) | finishes from the true start; time-to-event |

### 7.6 Controls and metrics

* **The flat control (Nachum 2019).** The same executor, trained with the same
  random-target curriculum but with no plan in the observation: the fan is zeroed and the
  target is given only as a relative position, DD-PPO style. If it matches, the plan adds
  nothing, and the result is reported that way.
* **The distillation control** (Barkour; C-Planning): a flat policy trained on the finished
  system's flights.
* **Metrics.**
  * Finishes from the true start, and the STEP at which the first finish appears (the gate
    ladder rule), never an end-of-run mean.
  * Executor tracking statistics.
  * Certification rate.
  * The planner's entropy and the number of distinct plans it uses.
  * The distribution of plan lengths and realised progress per plan.
* **Protocol.** One seed per arm, a same-card control, SCRATCH for the executor. The ledger
  entry states the plans' provenance: map geometry and the policy's own flights, no
  record.

---

## 8. References

Entries marked [NAV], [DET] or [HIER] were already covered there; they are listed here only
where this file adds details. All others are new to the repo.

**Joint training, stability, subgoal feasibility**
1. Florensa, Duan, Abbeel. Stochastic Neural Networks for Hierarchical Reinforcement Learning (SNN4HRL). ICLR 2017. arXiv:1704.03012.
2. Heess, Wayne, Tassa, Lillicrap, Riedmiller, Silver. Learning and Transfer of Modulated Locomotor Controllers. arXiv:1610.05182, 2016.
3. Frans, Ho, Chen, Abbeel, Schulman. Meta Learning Shared Hierarchies (MLSH). ICLR 2018. arXiv:1710.09767.
4. Li, Florensa, Clavera, Abbeel. Sub-policy Adaptation for Hierarchical Reinforcement Learning (HiPPO). ICLR 2020. arXiv:1906.05862.
5. Zhang, Whiteson. DAC: The Double Actor-Critic Architecture for Learning Options. NeurIPS 2019. arXiv:1904.12691.
6. Jain, Iscen, Caluwaerts. Hierarchical Reinforcement Learning for Quadruped Locomotion. IROS 2019. arXiv:1905.08926.
7. Yu. Reinforcement Learning with Anticipation: A Hierarchical Approach for Long-Horizon Tasks (RLA). arXiv:2509.05545, 2025.
8. Guertler, Buechler, Martius. Hierarchical Reinforcement Learning with Timed Subgoals (HiTS). NeurIPS 2021. arXiv:2112.03100.
9. Hwang, Lee, Kim, Han. Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning (SSE). arXiv:2506.21039, 2025-2026.
10. Levy, Konidaris, Platt, Saenko. Learning Multi-Level Hierarchies with Hindsight (HAC). ICLR 2019. arXiv:1712.00948. [HIER, constants]
11. Zhang, Guo, Tan, Hu, Chen. Generating Adjacency-Constrained Subgoals in HRL (HRAC). NeurIPS 2020. arXiv:2006.11485; TPAMI extension arXiv:2111.00213. [NAV, details]
12. Kim, Seo, Shin. Landmark-Guided Subgoal Generation in HRL (HIGL). NeurIPS 2021. arXiv:2110.13625. [NAV, DET, details]
13. Lee, Kim, Jang, Kim. DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical RL. NeurIPS 2022. arXiv:2210.05150.
14. Luo, Sun, Ji, Zhan. Bidirectional-Reachable Hierarchical RL with Mutually Responsive Policies (BrHPO). arXiv:2406.18053, 2024.
15. Wang, Tang, Yang, Sun, Wang, Zhang, Chen. Guided Cooperation in HRL via Model-based Rollout (GCMR). arXiv:2309.13508; IEEE TNNLS 2024.
16. Li, Zheng, Wang, Zhang. Learning Subgoal Representations with Slow Dynamics (LESSON). ICLR 2021.
17. Li, Zhang, Wang, Yu, Zhang. Active Hierarchical Exploration with Stable Subgoal Representation Learning (HESS). ICLR 2022. arXiv:2105.14750.
18. Wang, Wang, Yang, Kamarainen, Pajarinen. Probabilistic Subgoal Representations for HRL (HLPS). ICML 2024. arXiv:2406.16707.
19. Wang, Wang, Pajarinen. HRL with Uncertainty-Guided Diffusional Subgoals (HIDI). ICML 2025. arXiv:2505.21750.
20. Nasiriany, Pong, Lin, Levine. Planning with Goal-Conditioned Policies (LEAP). NeurIPS 2019. arXiv:1911.08453. [NAV, HIER, details]
21. Zhang, Eysenbach, Salakhutdinov, Levine, Gonzalez. C-Planning: An Automatic Curriculum for Learning Goal-Reaching Tasks. ICLR 2022. arXiv:2110.12080.
22. Chane-Sane, Schmid, Laptev. Goal-Conditioned RL with Imagined Subgoals (RIS). ICML 2021. arXiv:2107.00541.
23. Levine, Finn, Darrell, Abbeel. End-to-End Training of Deep Visuomotor Policies (guided policy search, BADMM). JMLR 17(39), 2016. arXiv:1504.00702.
24. Mordatch, Todorov. Combining the benefits of function approximation and trajectory optimization. RSS 2014. doi:10.15607/RSS.2014.X.052.
25. Harutyunyan, Dabney, Borsa, Heess, Munos, Precup. The Termination Critic. AISTATS 2019. arXiv:1902.09996.
26. Khetarpal, Klissarov, Chevalier-Boisvert, Bacon, Precup. Options of Interest: Temporal Abstraction with Interest Functions. AAAI 2020. arXiv:2001.00271.
27. Kamat, Precup. Diversity-Enriched Option-Critic. arXiv:2011.02565, 2020.
28. Klissarov, Bacon, Harb, Precup. Learnings Options End-to-End for Continuous Action Tasks (PPOC). arXiv:1712.00004, 2017.

**The planner's penalties, curricula over plans and goals, safety margins**
29. Racaniere, Lampinen, Santoro, Reichert, Firoiu, Lillicrap. Automated Curricula Through Setter-Solver Interactions. ICLR 2020. arXiv:1909.12892.
30. Campero, Raileanu, Kuttler, Tenenbaum, Rocktaschel, Grefenstette. Learning with AMIGo: Adversarially Motivated Intrinsic Goals. ICLR 2021. arXiv:2006.12122. [DET, constants]
31. Dennis, Jaques, Vinitsky, Bayen, Russell, Critch, Levine. Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED). NeurIPS 2020. arXiv:2012.02096.
32. Jiang, Grefenstette, Rocktaschel. Prioritized Level Replay. ICML 2021. arXiv:2010.03934.
33. Portelas, Colas, Hofmann, Oudeyer. Teacher Algorithms for Curriculum Learning of Deep RL in Continuously Parameterized Environments (ALP-GMM). CoRL 2019. arXiv:1910.07224.
34. Tang, Kan, Shan, Chen. Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling. arXiv:2505.17659, 2025.
35. Herbert, Chen, Han, Bansal, Fisac, Tomlin. FaSTrack: a Modular Framework for Fast and Guaranteed Safe Motion Planning. CDC 2017. arXiv:1703.07373.
36. Kousik, Vaskov, Bu, Johnson-Roberson, Vasudevan. Bridging the Gap Between Safety and Real-Time Performance in Receding-Horizon Trajectory Design for Mobile Robots (RTD). IJRR 2020. arXiv:1809.06746.

**The executor's reward: tracking vs progress**
37. Song, Romero, Mueller, Koltun, Scaramuzza. Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning. Science Robotics 2023. arXiv:2310.10943.
38. Song, Steinweg, Kaufmann, Scaramuzza. Autonomous Drone Racing with Deep Reinforcement Learning. IROS 2021. arXiv:2103.08624.
39. Fuchs, Song, Kaufmann, Scaramuzza, Duerr. Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning. RA-L 2021. arXiv:2008.07971.
40. Rudin, Hoeller, Reist, Hutter. Learning to Walk in Minutes Using Massively Parallel Deep RL. CoRL 2021 (PMLR 164). arXiv:2109.11978; github.com/leggedrobotics/legged_gym. [NAV, constants]
41. He, Zhang, Xiao, He, Liu, Shi. Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion (ABS). RSS 2024. arXiv:2401.17583.
42. Zhang, Rudin, Hoeller, Hutter. Learning Agile Locomotion on Risky Terrains. IROS 2024. arXiv:2311.10484.
43. Jenelten, He, Farshidian, Hutter. DTC: Deep Tracking Control. Science Robotics 2024. arXiv:2309.15462.
44. Gangapurwala, Geisert, Orsolino, Fallon, Havoutis. RLOC: Terrain-Aware Legged Locomotion using RL and Optimal Control. T-RO 2022. arXiv:2012.03094.
45. Meyer, Robinson, Rasheed, San. Taming an Autonomous Surface Vehicle for Path Following and Collision Avoidance Using Deep RL. IEEE Access 8, 2020. arXiv:1912.08578.
46. Chiang, Faust, Fiser, Francis. Learning Navigation Behaviors End-to-End With AutoRL. RA-L 2019. arXiv:1809.10124.
47. Wijmans, Kadian, Morcos, Lee, Essa, Parikh, Savva, Batra. DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames. ICLR 2020. arXiv:1911.00357.
48. Liniger, Domahidi, Morari. Optimization-Based Autonomous Racing of 1:43 Scale RC Cars (MPCC, HRHC). Optimal Control Applications and Methods 2015. arXiv:1711.07300.
49. Kastner et al. Arena-Rosnav: Towards Deployment of DRL-Based Obstacle Avoidance into Conventional Autonomous Navigation Systems. IROS 2021. arXiv:2104.03616.
50. Haro, Richter, Yang, Cadena, Hutter. Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation. arXiv:2603.13888, 2026.

**Capability models learned from the controller's own rollouts**
51. Kim, Kim, Hwangbo. Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation. RSS 2022. arXiv:2204.08647.
52. Yang, Wellhausen, Miki, Liu, Hutter. Real-time Optimal Navigation Planning Using Learned Motion Costs. ICRA 2021.
53. Guzzi, Chavez-Garcia, Nava, Gambardella, Giusti. Path Planning With Local Motion Estimations. RA-L 5(2), 2020.
54. Chavez-Garcia, Guzzi, Gambardella, Giusti. Learning Ground Traversability from Simulations. RA-L 3(3), 2018. arXiv:1709.05368.
55. Frey, Mattamala, Chebrolu, Cadena, Fallon, Hutter. Fast Traversability Estimation for Wild Visual Navigation (WVN). RSS 2023. arXiv:2305.08510.
56. Shah, Xu, Lu, Xiao, Toshev, Levine, Ichter. Value Function Spaces: Skill-Centric State Abstractions for Long-Horizon Reasoning. ICLR 2022. arXiv:2111.03189.
57. Ahn et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan). arXiv:2204.01691, 2022.
58. Shah, Sridhar, Bhorkar, Hirose, Levine. GNM: A General Navigation Model to Drive Any Robot. ICRA 2023. arXiv:2210.03370.
59. Shah, Sridhar, Dashora, Stachowicz, Black, Hirose, Levine. ViNT: A Foundation Model for Visual Navigation. CoRL 2023. arXiv:2306.14846.
60. Shah, Eysenbach, Rhinehart, Levine. Rapid Exploration for Open-World Navigation with Latent Goal Models (RECON). CoRL 2021. arXiv:2104.05859.
61. Luo, Wang, Mandala, Chou, Christmann, Chen, Chan, Lee, Chen. Feasibility-Guided Planning over Multi-Specialized Locomotion Policies. ICRA 2026. arXiv:2602.07932.
62. Alonso, Peter, Goumard, Romoff. Deep Reinforcement Learning for Navigation in AAA Video Games. arXiv:2011.04764, 2020.
63. Hoeller, Rudin, Sako, Hutter. ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots. Science Robotics 9(88), 2024. arXiv:2306.14874. [NAV, HIER, details]

**Proposal sets: driving**
64. Chai, Sapp, Bansal, Anguelov. MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction. CoRL 2019. arXiv:1910.05449.
65. Phan-Minh, Grigore, Boulton, Beijbom, Wolff. CoverNet: Multimodal Behavior Prediction using Trajectory Sets. CVPR 2020. arXiv:1911.10298.
66. Zhao et al. TNT: Target-driveN Trajectory Prediction. CoRL 2020. arXiv:2008.08294.
67. Shi, Jiang, Dai, Schiele. Motion Transformer with Global Intention Localization and Local Movement Refinement (MTR). NeurIPS 2022. arXiv:2209.13508.
68. Nayakanti, Al-Rfou, Zhou, Goel, Refaat, Sapp. Wayformer: Motion Forecasting via Simple & Efficient Attention Networks. ICRA 2023. arXiv:2207.05844.
69. Seff, Cera, Chen, Ng, Zhou, Nayakanti, Refaat, Al-Rfou, Sapp. MotionLM: Multi-Agent Motion Forecasting as Language Modeling. ICCV 2023. arXiv:2309.16534.
70. Song, Luan, Ding, Wang, Chen. Learning to Predict Vehicle Trajectories with Model-based Planning (PRIME). CoRL 2021. arXiv:2103.04027.
71. Dauner, Hallgarten, Geiger, Chitta. Parting with Misconceptions about Learning-based Vehicle Motion Planning (PDM-Closed). CoRL 2023. arXiv:2306.07962.
72. Jiang, Chen, Gao, Liao, Zhang, Liu, Wang. VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning. arXiv:2402.13243 (ICLR 2026).
73. Li et al. Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation. arXiv:2406.06978, 2024.
74. Liao et al. DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving. CVPR 2025. arXiv:2411.15139.
75. Werling, Ziegler, Kammel, Thrun. Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame. ICRA 2010.
76. Paden, Cap, Yong, Yershov, Frazzoli. A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles. IEEE T-IV 2016. arXiv:1604.07446.
77. Fan et al. Baidu Apollo EM Motion Planner. arXiv:1807.08048, 2018.

**Proposal sets: generative, latent-skill, diverse**
78. Sridhar, Shah, Glossop, Levine. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration. ICRA 2024. arXiv:2310.07896.
79. Janner, Du, Tenenbaum, Levine. Planning with Diffusion for Flexible Behavior Synthesis (Diffuser). ICML 2022. arXiv:2205.09991.
80. Ajay, Du, Gupta, Tenenbaum, Jaakkola, Agrawal. Is Conditional Generative Modeling all you need for Decision-Making? (Decision Diffuser). ICLR 2023. arXiv:2211.15657.
81. Chen, Deng, Kawaguchi, Gulcehre, Ahn. Simple Hierarchical Planning with Diffusion. ICLR 2024. arXiv:2401.02644.
82. Li, Wang, Jin, Zha. Hierarchical Diffusion for Offline Decision Making (HDMI). ICML 2023, PMLR 202.
83. Lee, Kim, Choi. Refining Diffusion Planner for Reliable Behavior Synthesis by Automatic Detection of Infeasible Plans (restoration gap). NeurIPS 2023. arXiv:2310.19427.
84. Ajay, Kumar, Agrawal, Levine, Nachum. OPAL: Offline Primitive Discovery for Accelerating Offline RL. ICLR 2021. arXiv:2010.13611.
85. Lynch, Khansari, Xiao, Kumar, Tompson, Levine, Sermanet. Learning Latent Plans from Play. CoRL 2019. arXiv:1903.01973. [human teleoperation data: architecture only]
86. Shi, Lim, Lee. Skill-based Model-based Reinforcement Learning (SkiMo). CoRL 2022. arXiv:2207.07560.
87. Xie, Bharadhwaj, Hafner, Garg, Shkurti. Latent Skill Planning for Exploration and Transfer (LSP). ICLR 2021. arXiv:2011.13897.
88. Jiang, Zhang, Janner, Li, Rocktaschel, Grefenstette, Tian. Efficient Planning in a Compact Latent Action Space (TAP). ICLR 2023. arXiv:2208.10291.
89. Kumar, Kumar, Levine, Finn. One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL (SMERL). NeurIPS 2020. arXiv:2010.14484.
90. Zahavy, Schroecker, Behbahani, Baumli, Flennerhag, Hou, Singh. Discovering Policies with DOMiNO. ICLR 2023. arXiv:2205.13521.
91. Yuan, Kitani. Diverse Trajectory Forecasting with Determinantal Point Processes (DSF). ICLR 2020. arXiv:1907.04967.
92. Ichter, Harrison, Pavone. Learning Sampling Distributions for Robot Motion Planning. ICRA 2018. arXiv:1709.05448.
93. Holden, Kanoun, Perepichka, Popa. Learned Motion Matching. ACM TOG (SIGGRAPH) 2020. [motion-capture data: mechanism only]
