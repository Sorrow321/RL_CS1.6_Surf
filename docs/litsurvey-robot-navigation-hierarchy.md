# Macro / micro: what robotics does about the layer above a good low-level controller

Literature survey, 2026-09-20. Web research only; no code was run and nothing in the
repo was changed except this file. ASCII only.

Commissioned question (user, 2026-09-20): our surf agent has a MICRO policy that is
excellent (25 Hz decisions over 100 Hz physics: strafing, turns, take-offs,
ramp-riding, superhuman when the path is defined) and essentially NO macro policy
(which way to go, which ramp to take). This is the legged-robot situation: a machine
with many motors that must be coordinated every instant (micro) plus something that
says "go there, turn left, walk around that" (macro). What are roboticists doing?

---

## 0. The one-sentence answer, and why it is exactly our bug

Every deployed learned-legged-locomotion system in the last seven years has the same
shape as ours - a fast low-level policy conditioned on a short command, plus a slower
thing that emits commands - and the entire research effort of the last four years has
moved to the SECOND layer. The single most repeated finding, stated almost verbatim
in Fu et al. 2022 (VP-Nav), Hoeller et al. 2024 (ANYmal Parkour), Roth et al. 2025
(perceptive FDM) and Lee et al. 2024 (wheeled-legged city navigation), is:

> the high-level layer must be aware of what the low-level layer can actually do,
> and a planner built on free space alone is not.

Our geodesic BFS field is precisely "a planner built on free space alone". It is a
shortest-path computation over voxels the player can OCCUPY, with no notion of
whether the player can GET from one voxel to the next at the speeds and in the
directions surf physics allows. Our documented pathology - the field says "straight"
across a pit that the player cannot fly across, the potential dips 2,840 u into it,
the policy tracks the dip and dies - is the textbook failure that the robotics
literature calls a non-traversability-aware cost map, and the field has three
standard fixes for it, all of which are legal under our constraints (no human demos,
no per-map constants; a planner that reads geometry or runs the simulator is fine):

1. **Certify the edges with the controller.** Replace "is this voxel free" with "can
   MY policy get from A to B", measured by rolling the policy in the simulator
   (Roth 2025 FDM + MPPI; Wellhausen 2021 reachability planning; ArtPlanner 2023) or
   by a learned goal-conditioned value function used as a graph metric (Eysenbach
   2019 SoRB). The pit then has no outgoing edge and the potential stops lying.
2. **Change the interface from a field to a WAYPOINT with a time budget.** Rudin
   et al. 2022 (IROS) showed that swapping a velocity-command interface for a
   "reach this local position within this time" interface, with the reward paid only
   in the last second, multiplies what the same robot can traverse by 5-8x
   (gaps 0.15 m -> 1.2 m; climbs 0.1 m -> 0.95 m) with no new skill engineering. The
   macro then owns WHERE and the micro owns HOW, and a wrong global field can no
   longer capture the micro.
3. **Put an RL policy on top of a FROZEN micro, in the same simulator, on a sparse
   goal reward.** This is ANYmal Parkour (nav at 5 Hz over skills at 50 Hz),
   Lee 2024 (nav at 10 Hz over locomotion at 50 Hz), ViNL, RoM-Nav. The macro acts
   in a 3-4 dimensional command space, which is small enough that ordinary
   exploration covers it - which is the real reason hierarchy helps
   (Nachum et al. 2019).

The skeptical counterweight, which must be read before building anything: Nachum,
Tang, Lu, Gu, Lee, Levine, "Why Does Hierarchy (Sometimes) Work So Well in
Reinforcement Learning?" (arXiv:1909.10618, 2019) isolates the benefit of hierarchy
across locomotion, navigation and manipulation and concludes that "most of the
observed benefits of hierarchy can be attributed to improved exploration, as opposed
to easier policy learning or imposed hierarchical structures", and then builds
NON-hierarchical exploration baselines that match hierarchical RL. Translated to our
problem: if a macro layer helps us, it will be because it made exploration happen in
a 3-dimensional waypoint space instead of a 6-dimensional action space, not because
two networks are better than one. Any macro design should be run against a flat
control that explores in the same small space.

---

## 1. Legged-robot locomotion controllers and the COMMAND INTERFACE they expose

This section answers "what exactly is the wire between the two layers". The answer is
remarkably uniform, and there are only four interfaces in the entire literature:
(a) a velocity twist, (b) a position/waypoint plus a time budget, (c) a gait/style
vector, (d) a discrete skill index or a latent skill code.

### 1.1 Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter, "Learning agile and dynamic motor skills for legged robots", Science Robotics 4(26):eaau5872, 2019

* Robot/task: ANYmal quadruped; command-following locomotion, recovery from a fall.
* Interface: **velocity twist** - commanded forward/lateral velocity and yaw rate, up
  to ~1.5 m/s, as extra observation columns. The policy runs at 200 Hz (joint-level);
  the command is whatever a human or a planner writes into those columns.
* Training: PPO in a rigid-body sim with a learned actuator network; one level only.
  Command robustness comes from sampling commands uniformly during training.
* Macro layer: none - a human joystick.
* Detour result: n/a.
* Applicability: this is the ancestor of the entire interface question, and it is the
  "our micro" analogue. The relevant lesson is only that the command is an OBSERVATION
  COLUMN, sampled uniformly at train time so the micro is robust to ANY command,
  including bad ones. Our `--race-arc`-style conditioning would live in the same place.

### 1.2 Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning quadrupedal locomotion over challenging terrain", Science Robotics 5(47):eabc5986, 2020

* Robot/task: ANYmal, blind locomotion over rubble, mud, snow, streams.
* Interface: **velocity twist**, again. The contribution is teacher/student privileged
  learning (teacher sees terrain and contact state, student sees proprioception only)
  and a temporal convolutional encoder over proprioceptive history.
* Macro layer: none.
* Applicability: the privileged-teacher trick matters to us for a different reason -
  it is how you legitimately move information that exists in the SIMULATOR (and not
  at deployment) into a policy without demonstrations. If we ever want a policy that
  behaves as if it knew the reachability structure, teacher/student on simulator-
  privileged information is the sanctioned route, and it uses no human data.

### 1.3 Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning robust perceptive locomotion for quadrupedal robots in the wild", Science Robotics 7(62):eabk2822, 2022

* Robot/task: ANYmal with LiDAR-derived local elevation map; hour-long alpine hike at
  human pace.
* Interface: **velocity twist**, plus an exteroceptive local height map as an
  observation. An attention-based recurrent belief encoder fuses proprioception and
  exteroception and learns when to distrust the map.
* Macro layer: still a human or an external planner writing the twist.
* Detour behaviour: none at the policy level - the policy handles terrain, not routes.
* Applicability: the belief encoder is the canonical answer to "the exteroceptive map
  is sometimes wrong" - it learns to fall back on proprioception. Our analogue: the
  potential channel (`--obs-potential`) is our exteroceptive map, and it is sometimes
  wrong in exactly the way Miki's height map is wrong. A learned gate on the potential
  channel is a generic mechanism, though Miki's version needs a proprioceptive signal
  that CONTRADICTS the map, which in our case only arrives after the fall.

### 1.4 Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning", CoRL 2022, arXiv:2109.11978 (Legged Gym / Isaac Gym)

* Robot/task: ANYmal and others on rough terrain; the reference open-source stack.
* Interface: **velocity twist**, with a "game-inspired" terrain curriculum: robots are
  promoted to harder terrain rows when they track the command well and demoted when
  they do not.
* Training: PPO, thousands of parallel environments on one GPU, minutes of wall clock.
* Applicability: directly comparable to our 2,048-env / 600k-step-per-second setup;
  the curriculum promotion rule is a generic mechanism (promote on measured success,
  not on a hand-set schedule) and is a legitimate template for a command curriculum
  if we adopt a waypoint interface. Note the curriculum is over TERRAIN DIFFICULTY,
  which in our case would be over WAYPOINT DISTANCE, not over map regions - so it
  stays map-agnostic.

### 1.5 Kumar, Fu, Pathak, Malik, "RMA: Rapid Motor Adaptation for Legged Robots", RSS 2021, arXiv:2107.04034

* Robot/task: A1 quadruped, unseen terrains/payloads.
* Interface: **velocity twist** plus an internally estimated latent "extrinsics"
  vector; a base policy conditioned on extrinsics, and an adaptation module that
  regresses extrinsics from proprioceptive history.
* Training: two phases, base policy with privileged environment parameters, then the
  adaptation module by supervised regression in sim.
* Applicability: low for the macro question; high as the pattern "the low level
  estimates what it needs from its own history rather than being told".

### 1.6 Margolis, Agrawal, "Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior", CoRL 2022, arXiv:2212.03238

* Robot/task: Unitree Go1; a single policy that spans a structured family of gaits.
* Interface: a **wide command vector** - velocity twist PLUS gait timing offsets, foot
  swing height, body height, pitch, stance width. The macro can therefore select not
  only where to go but HOW, in a continuous, semantically meaningful space.
* Training: one-stage RL with all command dimensions sampled during training.
* Detour result: n/a, but the paper's point is that behaviour choice can be deferred
  to deployment time without retraining.
* Applicability: **high and under-appreciated for us.** The surf analogue of "gait
  parameters" is a style/aggression vector (target speed band, strafe aggressiveness,
  how early to leave a ramp). If the micro is trained conditioned on such a vector,
  the macro gains a second channel that is not "where" but "how", and choosing it is
  a 3-4 dimensional search the macro can actually do. Crucially the vector is defined
  on generic quantities (speed, turn rate), not on any map's geometry.

### 1.7 Siekmann, Godse, Fern, Hurst, "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition", ICRA 2021, arXiv:2011.01387; and Li, Cheng, Peng, Abbeel, Levine, Berseth, Sreenath, "Reinforcement Learning for Robust Parameterized Locomotion Control of Bipedal Robots", ICRA 2021, arXiv:2103.14295

* Robot/task: Cassie biped.
* Interface: a **parameterized gait command** - walking/hopping/running/skipping
  selected by periodic reward coefficients (Siekmann), or (walking speed, step
  length, walking height, turning yaw rate) (Li).
* Training: one level, RL in sim, sim-to-real.
* Applicability: same lesson as Walk These Ways; a command interface can be much
  richer than a twist without costing anything, and it is the macro's action space.

### 1.8 Radosavovic, Xiao, Zhang, Darrell, Malik, Sreenath, "Humanoid Locomotion as Next Token Prediction", NeurIPS 2024, arXiv:2402.19469 (and "Real-World Humanoid Locomotion with Reinforcement Learning", Science Robotics 2024)

* Robot/task: Digit humanoid walking in San Francisco, zero-shot.
* Interface: **velocity twist**; a causal transformer over sensorimotor tokens
  predicts the next action token. Trained on a mixed corpus: a prior neural policy,
  a model-based controller, motion capture, and YouTube videos of humans.
* Applicability judgment: **the training corpus is human data and is out of bounds for
  us as a training source.** The architectural point that DOES transfer is that the
  micro can be a sequence model over its own history with the command as a token; the
  data point does not.

### 1.9 Berkeley Humanoid (Liao, Zhang, Sreenath et al., arXiv:2407.21781, 2024) and the Unitree H1/G1 line

* Interface: **velocity twist** again, trained with Legged-Gym-style massively
  parallel PPO and heavy domain randomization.
* Applicability: confirms the twist is the de-facto standard interface for humanoids
  too. Nothing new for the macro question; included for coverage of the humanoid arm
  of the request.

### 1.10 How the low level is made robust to ARBITRARY commands

Across all of the above the recipe is identical and worth stating because it is the
part we would have to copy:

* the command is sampled **uniformly over its full range** every episode (and often
  resampled mid-episode), so the micro never sees a command distribution that matches
  any particular macro;
* a **command curriculum** widens the range as tracking improves (Rudin 2022);
* **domain randomization** over dynamics parameters so the micro does not overfit to
  an exact physics;
* the reward is a **tracking** reward (exponential in command error) plus regularizers
  (torque, joint acceleration, collision), with the task reward sparse or absent.

The point for us: if the macro is going to emit waypoints, the micro must have been
trained on waypoints drawn from a distribution WIDER than what any macro will produce,
including waypoints that are unreachable. This is the generic mechanism; it contains
no map constant.

---

## 2. NAVIGATION ON TOP OF LEARNED LOCOMOTION

This is the closest literature to the user's question. Group the systems by what the
macro is.

### 2.1 MACRO = classical planner on a cost map, MICRO = learned controller

#### Fu, Kumar, Agarwal, Qi, Malik, Pathak, "Coupling Vision and Proprioception for Navigation of Legged Robots" (VP-Nav), CVPR 2022, arXiv:2112.02094

* Robot/task: A1 quadruped, point-goal navigation in cluttered indoor/outdoor scenes.
* Architecture: onboard depth -> occupancy map -> cost map -> **fast marching planner**
  -> a velocity-command generator -> RMA locomotion policy at joint level. A **safety
  advisor** module watches proprioception, writes obstacles the camera missed (glass
  walls, invisible ledges) into the occupancy map, and imposes an environment-
  determined **speed limit** on the command generator.
* Macro output: a **velocity twist**, at the map update rate; the path is recomputed
  from the cost map.
* Training: the micro is RL+RMA in sim; the macro is CLASSICAL (fast marching), not
  learned. The safety advisor is a learned fall/slip predictor.
* Needs: a map it builds online; a simulator for the micro; no demos.
* Detour behaviour: this is exactly where the detour comes from - the safety advisor
  writes the impassable thing into the map, the planner re-plans AROUND it, and the
  robot goes around a glass wall it cannot see.
* Applicability to us: **very high, and it is the cleanest statement of our bug.** The
  abstract's own framing - make the high-level path planner "aware of the walking
  capabilities of the locomotion policy in varying environments" - is our missing
  piece. Our version of the safety advisor is: the policy fell / lost all speed /
  failed to advance from cell c toward cell c', so mark that EDGE as non-traversable
  and rebuild the potential. Generic; no map constant; uses only the policy's own
  experience.

#### Wellhausen, Hutter, "Rough Terrain Navigation for Legged Robots using Reachability Planning and Template Learning", IROS 2021

* Macro: a **reachability-based** planner - poses are valid only if a learned model
  says a feasible body pose/footholds exist there. So the free-space test is replaced
  by a capability test.
* Applicability: the conceptual ancestor of point 1 in section 0. Its "template"
  learning is quadruped-specific, but the pattern (node validity = learned capability,
  not geometry) is fully generic.

#### Frey, Roth, Cadena, Hutter et al., "ArtPlanner: Robust Legged Robot Navigation in the Field", arXiv:2303.01420 (Field Robotics 2023)

* Macro: sampling-based planner over a **reachability abstraction** with learned
  foothold scores restricting where the robot may step; deployed in the DARPA SubT
  aftermath and on real missions.
* Applicability: same lesson; a production-grade instance of "the planner must be told
  what the controller can do".

#### Roth, Frey, Cadena, Hutter, "Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation", RSS 2025, arXiv:2504.19322

* Robot/task: ANYmal, Barry, ANYmal-on-Wheels; local navigation in rough terrain.
* Architecture: a **learned forward dynamics model** predicts the robot's future state
  AND a failure probability, conditioned on surrounding geometry and proprioceptive
  history; it is trained on "multiple years" of simulated navigation experience
  including deliberately high-risk manoeuvres, plus some real interactions. The model
  is then dropped into a zero-shot **MPPI** planner: sample command sequences, roll
  them through the FDM, score by progress and failure probability, execute the first.
* Macro output: a velocity command sequence at the MPPI rate; the micro is the
  existing frozen locomotion policy.
* Training order: micro first (frozen), then the FDM is fit to rollouts of THAT micro,
  then MPPI uses it with no further learning.
* Needs: a simulator, a frozen controller, no demos, no human labels.
* Result: +41% position-prediction accuracy over baselines, **+27% navigation success
  rate** in rough simulated environments; a separate FDM per platform captures each
  platform's own limits.
* Applicability: **the single most directly transferable paper in this survey.** It is
  "run the simulator to learn what the policy can do, then plan with that" - which is
  exactly the permitted category (a planner that runs the simulator). Our version does
  not even need a learned FDM at first: with 600k env steps/s we can roll the real
  physics. The learned FDM is the cheap amortization of that.

### 2.2 MACRO = learned navigation policy, MICRO = learned controller (frozen)

#### Hoeller, Rudin, Sako, Hutter, "ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots", Science Robotics 9(88):eadi7566, 2024; arXiv:2306.14874

Read closely, as requested. This is the architecture the user's framing describes.

* Robot/task: ANYmal D; parkour-like courses with boxes, gaps, low overhangs; speeds
  up to 2 m/s on hardware.
* **Skill set (the micro layer), five policies**: walking (irregular terrain, stairs,
  slopes), jumping (gaps up to ~1 m), climbing up (boxes up to ~1 m), climbing down,
  crouching (under ~0.4 m clearance). Each is trained separately with RL on a
  **position-based command with a time constraint** (reach this pose within this time)
  plus a terrain curriculum. Reported per-skill success exceeding 90% up to each
  skill's maximum trained difficulty.
* **Macro (the navigation policy)**: a **hybrid action space** - a Gaussian over
  continuous local commands (target position, heading, and a TIME) plus a Categorical
  over the discrete skill index. It runs at **5 Hz**; the skills run at **50 Hz**; the
  perception module at **30 Hz**.
* Macro training: PPO on a **sparse global-goal reward**, with the skills FROZEN. The
  navigation policy therefore learns the skills' competence envelopes by trial: it
  discovers which skill works from which relative pose, and it will refuse a jump it
  cannot make and pick a climb instead.
* **Perception**: a two-scale occupancy reconstruction from highly occluded, noisy
  point clouds (coarse net ~4 m range at 12.5 cm; refiner ~2 m at 6.25 cm), trained
  unsupervised from ~2,000 simulated trajectories with a binary cross-entropy
  occupancy loss.
* Needs: a simulator; no expert demonstrations; no offline computation; no a-priori
  map of the environment at deployment.
* Reported result: the learned navigation policy reached ~96.3% success in complex
  scenarios against ~60.9% for hand-designed/manual command trajectories.
* Detour behaviour: the paper's claim is precisely skill-awareness - "the navigation
  policy is aware of the capabilities of each skill, and it will adapt its behavior
  depending on the scenario". The demonstrated behaviour is choosing a different SKILL
  for an obstacle rather than a long spatial detour around it; I did not find an
  experiment showing a large route-level detour around an impassable region.
* Applicability judgment: **the template for our macro**, with two caveats. (a) The
  five skills are a designer's taxonomy matched to the benchmark's obstacle types -
  that is the part that smells map-specific and we should not copy it; our equivalent
  should be a CONTINUOUS command, not a skill index, or a skill set discovered rather
  than named. (b) The macro is trained on a sparse goal reward in randomized courses -
  which is only tractable because the macro's action space is 4-5 dimensional at 5 Hz.
  That is the generic gain, and it is Nachum 2019's exploration argument in hardware.

#### Rudin, Hoeller, Bjelonic, Hutter, "Advanced Skills by Learning Locomotion and Local Navigation End-to-End", IROS 2022, arXiv:2209.12827

Not hierarchical at all, and that is why it matters: it shows how much of the macro
problem is really an INTERFACE problem.

* Robot/task: ANYmal over gaps, pits, stairs, boxes.
* Interface: instead of tracking a stream of velocity commands, the policy is given a
  **target position in the base frame plus the remaining time**, with a **6 s** episode.
* Reward: **sparse**, paid only in the final **1 s** window, as
  `(1/T_r) * 1/(1 + ||x_b - x_b*||^2)`, plus the usual continuous penalties (joint
  acceleration, torque, collisions, action rate, foot acceleration). The sparsity is
  deliberate: it leaves the robot free to accelerate, decelerate, or move laterally
  if that is what gets it there in time.
* Terrain curriculum in three stages, promotion by success rate.
* Result vs a velocity-tracking baseline on the same robot: gaps **1.2 m vs 0.15 m**,
  pit/climb **0.95 m vs 0.1 m**, stairs **0.4 m vs 0.22 m**; on hardware, 0.6 m gap
  jumps and 0.55 m box climbs.
* Applicability judgment: **the highest value-per-unit-risk change available to us.**
  The "reward only in the last second" trick is a generic instrument for exactly our
  disease: a continuous distance-shaping reward (our potential) tells the agent what
  to do at every instant and therefore forbids a detour that temporarily increases the
  distance; a terminal, time-boxed reward does not. This is a reward-structure change,
  not a map-specific term, and it is one flag's worth of work if we adopt a waypoint
  interface.

#### Lee, Bjelonic, Reske, Wellhausen, Miki, Hutter, "Learning robust autonomous navigation and locomotion for wheeled-legged robots", Science Robotics 9(89):eadi9641, 2024; arXiv:2405.01792

* Robot/task: ANYmal-on-Wheels; kilometre-scale autonomous missions in Zurich and
  Seville.
* Three layers: a **global** path planner (Dijkstra on a navigation graph built offline
  from LiDAR reality capture), a **learned high-level controller (HLC)** at **10 Hz**
  that emits a velocity twist (vx in [-1.0, 2.0] m/s, vy in [-0.75, 0.75] m/s,
  wz in [-1.25, 1.25] rad/s), and a **learned low-level controller (LLC)** at **50 Hz**
  that outputs 12 joint positions + 4 wheel velocities.
* Training order: LLC first with teacher/student privileged learning on random velocity
  targets; then the LLC is **frozen** and the HLC is trained with PPO on a dense
  waypoint-following reward **plus an exploration bonus for visiting novel locations**,
  in procedurally generated worlds (Wave Function Collapse) with navigation graphs.
* Results: 8.3 km autonomous mission at 1.68 m/s average; peak 5.0 m/s; mechanical cost
  of transport 0.16.
* Applicability: the **procedurally generated training worlds** are the answer to
  "how do you get a macro that is not tuned to one map" - train it over a distribution
  of worlds. We have 108+ maps in `maps_pool/` and a map generator is not needed; a
  macro trained across the pool with the same constants is exactly the deliverable the
  standing rules demand. The **novelty bonus inside the macro's reward** is also
  notable: they needed it at the macro level, not at the joint level - which is the
  mirror image of our `--int-coef / --int-view / --int-speed` novelty living in the
  micro's action space where it farms in place.

#### Kareer, Yokoyama, Batra, Ha, Truong, "ViNL: Visual Navigation and Locomotion Over Obstacles", ICRA 2023, arXiv:2210.14791

* Robot/task: A1 in unseen apartment scans; reach a goal coordinate while stepping over
  clutter.
* Two policies trained in **two different simulators** and never co-trained: a visual
  navigation policy in Habitat that outputs **linear and angular velocity commands**,
  and a visual locomotion policy in Isaac that follows velocity commands while lifting
  feet over obstacles (trained with a foot-contact "lifting" reward and a privileged
  obstacle map distilled to egocentric depth).
* Co-deployment is **zero-shot** - the navigator's twist is simply fed to the locomotor.
* Applicability: proves the interface is a genuine abstraction boundary - you can
  develop the two layers independently and glue them at deployment. For us this means
  a macro can be prototyped against a FROZEN current checkpoint without retraining it,
  which is cheap and reversible.

#### Truong, Yarats, Li, Meier, Chernova, Batra, Rai, "Learning Navigation Skills for Legged Robots with Learned Robot Embeddings", IROS 2021

* Macro: a learned navigation policy that outputs velocity commands, conditioned on a
  learned **robot embedding** that encodes the low-level dynamics (max speed, slipping,
  contact behaviour) of whichever robot it is driving.
* Applicability: the embedding is the explicit form of "the macro must know what the
  micro can do". In our setting the analogous conditioning variable is not a robot id
  but the micro's own competence - e.g. a learned scalar per (cell, direction) that
  says "this policy clears this transition with probability p".

#### Hoeller, Wellhausen, Farshidian, Hutter, "Learning a State Representation and Navigation in Cluttered and Dynamic Environments", RA-L 6(3):5081-5088, 2021; arXiv:2103.04351

* Macro: a local navigation policy trained with RL that consumes a **learned latent
  world state** built by fusing a sequence of depth frames with the camera trajectory
  (state representation learning), and emits velocity commands. No explicit mapping.
* Detour behaviour: avoids static and moving obstacles reactively; local, not route-
  level.
* Applicability: moderate. It is the "no map, learn the representation" end of the
  spectrum; our map IS available, so we should exploit it rather than re-learn it.

#### Sorokin, Tan, Liu, Ha, "Learning to Navigate Sidewalks in Outdoor Environments", RA-L 7:3906-3913, 2022; arXiv:2109.05603

* Macro: a route plan from **public map services** (a map read, not a demo) plus a
  learned policy that keeps the robot on the sidewalk and avoids pedestrians.
* Training: a teacher in an abstract privileged world, then **behaviour cloning into a
  student** with realistic sensors. The BC source is the teacher AGENT, not a human -
  which under our section-0 rules is the allowed kind of distillation.
* Applicability: the structure "a global route from a geometric source + a learned
  local policy" is precisely what our geodesic field was supposed to be. The paper is
  a reminder that the global source is allowed to be crude as long as the local policy
  is free to deviate.

#### Seo, Gupta, Zhu, Skoutnev, Sentis, Zhu, "Learning to Walk by Steering: Perceptive Quadrupedal Locomotion in Dynamic Environments" (PRELUDE), ICRA 2023, arXiv:2209.09233

* Two levels: a high-level navigation controller predicting navigation commands, a
  low-level RL gait controller realizing them.
* **The high level is trained by imitation learning on HUMAN demonstrations collected
  on a steerable cart.**
* Applicability judgment: **architecturally on-target, methodologically forbidden for
  us.** Listed explicitly as the negative example: this is the shape of system we want
  with exactly the ingredient (human demonstrations of the macro) that our section 0
  prohibits. Worth citing in the ledger as "the macro-by-imitation branch, closed".

### 2.3 MACRO and MICRO fused into one policy, with a direction command

#### Cheng, Shi, Agarwal, Pathak, "Extreme Parkour with Legged Robots", ICRA 2024, arXiv:2309.14341

* Robot/task: a low-cost A1/Go1; high jumps on obstacles 2x the robot's height, long
  jumps across gaps 2x its length, handstands, tilted ramps.
* Architecture: a **single** vision-to-action policy. A privileged "oracle" phase gives
  the policy a scandot terrain representation and a target direction; then **direction
  distillation** trains a depth-based student that also predicts the yaw it should be
  heading at, so at deployment a human (or a planner) can steer it with a coarse
  direction and the policy resolves the precise heading itself.
* Applicability: the relevant idea is that the macro output can be **coarse and
  approximate** (a direction), with the micro correcting it - which matches our own
  xAUTO finding that a 58-chord decimation of a reference line, with 1,131 u maximum
  deviation and a quarter of its vertices inside solid geometry, performed as well as
  the full line. The ORDERING is what the macro must supply, not the geometry.

#### Zhuang, Fu, Wang, Atkeson, Schwertfeger, Finn, Zhao, "Robot Parkour Learning", CoRL 2023, arXiv:2309.05665

* Two-stage RL: pre-train each specialized skill with **soft dynamics constraints**
  (the robot may penetrate obstacles, with a penalty), then fine-tune with hard
  dynamics and a simple forward-progress + energy reward; then **DAgger-distill** the
  skills into one vision-based policy.
* Applicability: the **soft dynamics constraint** is an elegant, generic exploration
  device that we should note: let the agent temporarily violate a physical constraint
  so it can discover the shape of the solution, then anneal the violation away. Our
  surf equivalent would be a temporary relaxation (e.g. allowing a short "impossible"
  glide) that is annealed to zero - and, critically, it contains no map constant, only
  a schedule on a physics parameter. The DAgger here is from the system's OWN RL
  experts, which is allowed.

#### Zhuang, Yao, Zhao, "Humanoid Parkour Learning", CoRL 2024, arXiv:2406.10759

* End-to-end vision-based whole-body humanoid parkour with no motion prior; 0.42 m
  platform jumps, 0.8 m gaps, 1.8 m/s running.
* Applicability: confirms the pattern scales to humanoids without demonstrations.

#### Agarwal, Kumar, Malik, Pathak, "Legged Locomotion in Challenging Terrains using Egocentric Vision", CoRL 2022, arXiv:2211.07638; Yang, Yang, Wang, "Neural Volumetric Memory for Visual Locomotion Control", CVPR 2023, arXiv:2304.01201

* Both: end-to-end depth-to-action, no elevation map, no foothold planner. NVM adds an
  SE(3)-equivariant volumetric memory so the robot remembers terrain that has left the
  camera's view.
* Applicability: moderate. These strengthen the MICRO under partial observability; our
  micro is not the bottleneck. NVM's memory idea is relevant only if we later want the
  macro to reason about geometry it can no longer see.

### 2.4 Learned local planners trained WITHOUT any demonstrations

#### Yang, Wang, Cadena, Hutter, "iPlanner: Imperative Path Planning", RSS 2023; and Roth, Nubert, Yang, Mittal, Hutter, "ViPlanner: Visual Semantic Imperative Learning for Local Navigation", ICRA 2024, arXiv:2310.00982

* Architecture: a network maps a depth (iPlanner) or depth+semantics (ViPlanner) image
  and a goal to a **key-point path** with an associated collision probability. Trained
  by **imperative learning**: a bilevel optimization where the lower level is a
  trajectory optimizer / cost map and the upper level backpropagates through it. No
  labelled paths, no expert demonstrations, no RL.
* Applicability: **high, and under-used in the RL community.** This is the "learn a
  planner by differentiating through a cost" recipe; the supervision is the map
  geometry itself, which is legal for us. An iPlanner-style macro would output a short
  key-point path in our map and be trained by differentiating a traversability cost -
  no rewards, no episodes, no demos. Our difficulty is that surf traversability is not
  a static cost field; it depends on speed, which is the whole problem. That said,
  a velocity-augmented cost (cost of entering cell c with speed v) is a generic object.

#### Frey, Mattamala, Roth, Cadena, Fallon, Hutter et al., "Resilient Legged Local Navigation: Learning to Traverse with Compromised Perception End-to-End", ICRA 2024 (Best Paper finalist), arXiv:2310.03581

* Models perception failures as **invisible obstacles and pits**, trains an RL local
  navigation policy to reconstruct the true environment in latent space from corrupted
  perception and react end-to-end; +30% success under perception failure vs heuristic
  reactive planners; <10 ms CPU inference on ANYmal.
* Applicability: "invisible pits" is our exact geometry - a place the map says is fine
  and is not. The mechanism (train under a randomized corruption of the exteroceptive
  channel so the policy learns to distrust it) is fully generic and cheap: randomly
  corrupt the `--obs-potential` channel during training so the policy must cross-check
  it against depth. This is a real candidate and it is one training flag.

### 2.5 Waypoint interfaces and skill switching (2025-2026)

#### "Skill-Nav: Enhanced Navigation with Versatile Quadrupedal Locomotion via Waypoint Interface", arXiv:2506.21853 (2025)

* Interface: **2D waypoints in the robot's base frame**. The low level is a
  waypoint-guided locomotion policy that autonomously chooses its own gait/skill
  (jump, climb, avoid, cross a gap) to reach the waypoint; the robot advances to the
  next waypoint after dwelling ~2 s at the current one.
* Low-level rewards: a reaching term `r_reach = n_p / (t + eps)` (number of waypoints
  reached per unit time), a "stay" term, and a cosine-similarity tracking term that
  penalizes backward motion.
* Training curriculum: **WP-Fixed** (predetermined waypoint sequences over fixed terrain
  rows) then **WP-Random** (waypoints sampled from the robot's own position and yaw,
  irregularly distributed over a 10x10 grid). Domain randomization and inflated virtual
  obstacles during distillation.
* High level: interchangeable - **A\*/Dijkstra on a wall-only occupancy map**, or GPT-4
  prompted with the terrain description.
* Result: omni-traverse with obstacles 89% success, 8.2 m average travel, 11.0 s.
* Applicability: **the clearest modern statement of "waypoints are a better interface
  than velocity commands"**, with the argument we care about: waypoints are sparse,
  easy to generate, and let the low level decide HOW. The A\*-on-walls-only high level
  is notable - the macro can be crude because the micro is capable.

#### "Discovery of skill switching criteria for learning agile quadruped locomotion", arXiv:2502.06676 (2025, Frontiers in Robotics and AI 2026)

* Gait-specific low-level policies (trot, bound, gallop) learned from contact-pattern
  rewards alone; a **high-level policy generates blending weights** over them for
  goal-tracking; the DISTANCES at which skills switch are not hand-set but discovered
  by an **outer optimization loop** that updates them as learning progresses.
* Applicability: directly addresses the "no hand-tuned thresholds" requirement - the
  switching criterion is optimized, not chosen. If we ever adopt a skill index, this
  is how to avoid writing a per-map threshold.

#### Caluwaerts et al. (Google DeepMind), "Barkour: Benchmarking Animal-level Agility with Quadruped Robots", arXiv:2305.14654 (2023)

* Two solutions to the same obstacle course: (a) **specialist skills + a high-level
  navigation controller** that switches between them, and (b) a single Transformer
  "Locomotion-Transformer" **distilled from the specialists**. Scores were comparable;
  the generalist had slightly lower average score but smoother transitions.
* Robot completes the course at about half a dog's speed.
* Applicability: the honest read is that a monolithic distilled policy can match a
  switching hierarchy on a benchmark of this size - i.e. the hierarchy bought
  smoothness, not capability. Another data point for Nachum 2019's scepticism.

#### "Learning Safe Humanoid Navigation from Reduced Order Models" (RoM-Nav), arXiv:2609.19272 (2026, under review ICRA 2027)

* Explicitly states that "a standard single-stage RL navigation pipeline struggles to
  scale to multi-level and multi-story terrain". Decomposition: first train a policy on
  **reduced-order dynamics** (a simple model of the robot) with full 3D LiDAR, so
  navigation knowledge is learned cheaply; then **kickstart** a full-order policy from
  it with the locomotion policy **frozen** in the loop. A Poisson safety filter on the
  navigation output restores safety against out-of-distribution obstacles.
* Result: Unitree G1, mapless multi-floor navigation, >10 m vertical displacement,
  >100 m path length.
* Applicability: **the "reduced-order model" step is an idea we can use cheaply.** Our
  reduced-order surf model is a point mass with a speed cap and a turn-rate cap moving
  on ramps - a model in which a macro can be trained or searched in seconds, then used
  to kickstart or to supply waypoints to the real policy. It contains no map constant
  (the caps come from the physics engine, not the map), it reads map geometry, and it
  is a strictly better free-space abstraction than BFS-over-voxels.

---

## 3. HIERARCHICAL RL WHERE THE LOW LEVEL IS LOCOMOTION AND THE HIGH LEVEL NAVIGATES

This is the Ant/Humanoid-maze literature. Read it with one eye on what the AntMaze
benchmark actually gives the algorithm, because most of the detour results depend on
it: a **low-dimensional state** with the torso's global (x, y) available, and a
subgoal space that is literally those two coordinates. That is a hand-picked, perfect
abstraction, and it is the reason the U-shape and W-shape mazes get solved.

### 3.1 Nachum, Gu, Lee, Levine, "Data-Efficient Hierarchical Reinforcement Learning" (HIRO), NeurIPS 2018, arXiv:1805.08296

* Task: Ant Maze (U-shape), Ant Push, Ant Fall - a simulated quadruped in a maze with
  a 29-D state whose first dimensions are the torso xyz.
* Architecture: a high level proposing a **subgoal in the raw state space** (in
  practice the position coordinates) every **c steps** (c = 10 in the published
  configuration); a low level trained on a **parameterized distance reward** to that
  subgoal; a goal transition function h that shifts the subgoal with the agent so it
  stays relative.
* Key mechanism: the **off-policy correction** - because the low level changes, old
  high-level transitions are invalid, so a stored subgoal is relabelled with whichever
  of ~10 candidate subgoals makes the actually-observed low-level action sequence most
  likely under the CURRENT low-level policy. This is what makes off-policy HRL work.
* Both levels trained **concurrently, off-policy (TD3)**, from scratch.
* Result: substantially beats FuN, SNN4HRL and VIME on all three tasks; solves Ant Maze
  where flat agents fail.
* Applicability judgment: the mechanism is sound and generic, but (a) it is **off-
  policy**; our stack is on-policy PPO at 600k steps/s, and HIRO's sample efficiency
  argument is worth much less at our throughput; (b) the subgoal space being raw
  position is a privileged choice, though in our case a 3-D position subgoal is equally
  natural and equally generic; (c) Ant Maze's detour is a corridor, not a pit whose
  shortest-path field lies.

### 3.2 Levy, Konidaris, Platt, Saenko, "Learning Multi-Level Hierarchies with Hindsight" (HAC), ICLR 2019, arXiv:1712.00948

* Trains every level **in parallel** by pretending the level below is already optimal:
  **hindsight action transitions** (relabel the subgoal with the state the lower level
  actually reached) and **hindsight goal transitions** (HER lifted to the hierarchy).
  First framework to learn 3-level hierarchies in parallel in continuous spaces.
* Applicability: hindsight relabelling is off-policy machinery; on-policy PPO cannot
  use it directly. Its conceptual contribution for us is the diagnosis: the reason
  training a macro on a moving micro is hard is non-stationarity, and there are exactly
  two cures - relabel (HAC/HIRO) or FREEZE THE MICRO (the robotics answer, section 2.2).
  **Freezing is far simpler and is what every hardware system does.**

### 3.3 Vezhnevets, Osindero, Schaul, Heess, Jaderberg, Silver, Kavukcuoglu, "FeUdal Networks for Hierarchical Reinforcement Learning", ICML 2017, arXiv:1703.01161

* Manager emits a **direction in a learned latent space** every c steps; the worker
  maximizes **cosine similarity** to that direction; the manager is trained with a
  transition policy gradient rather than as a standard RL agent over the worker.
* Applicability: the "subgoal as a DIRECTION, not a point" idea is relevant to surf,
  where an absolute waypoint may be unreachable but a direction always exists. Weak
  empirically against HIRO on mazes.

### 3.4 Bacon, Harb, Precup, "The Option-Critic Architecture", AAAI 2017, arXiv:1609.05140; Bagaria, Konidaris, "Option Discovery using Deep Skill Chaining", ICLR 2020; Bagaria, Senthil, Konidaris, "Skill Discovery for Exploration and Planning using Deep Skill Graphs", ICML 2021

* Option-critic learns intra-option policies AND termination conditions end-to-end with
  no subgoals.
* **Deep skill chaining** is the interesting one for us: create an option that reliably
  reaches the goal from its neighbourhood, then create another option whose goal is to
  reach the first option's initiation set, and chain BACKWARD until the start state is
  covered. **Deep skill graphs** generalizes chains to a graph and uses it for planning.
* Applicability judgment: **skill chaining is the principled version of what our own
  reservoir/window curriculum is doing by hand.** Our spine/window machinery cuts a
  window near the frontier and trains there; skill chaining formalizes "learn the last
  segment first, then learn to reach its initiation set". It is champion-free and
  map-free by construction when the chain is grown from the policy's OWN reachable
  set. The published results are on low-dimensional continuous mazes, and the
  initiation-set classifiers are the fragile part at our observation dimensionality.

### 3.5 Gehring, Synnaeve, Krause, Usunier, "Hierarchical Skills for Efficient Exploration" (HSD-3), NeurIPS 2021, arXiv:2110.10809

* Task: sparse-reward bipedal/humanoid benchmark (GoalWall, Stairs, Hurdles, Limbo,
  Gaps) with up to a 21-joint Humanoid.
* Idea: pre-trained skills trade generality against specificity. HSD-3 builds a
  **hierarchy over GOAL SPACES** - which subset of proprioceptive features the high
  level is allowed to control (e.g. just x-position, or x plus torso height, or the
  full set) - and lets the agent select the goal space as well as the goal, with a
  third option of falling back to native actions.
* Result: matches or beats single-goal-space hierarchies across the suite, with large
  gains on GoalWall and Stairs.
* Applicability: **the "which features does the macro control" question is exactly our
  open question** (a waypoint? a heading? a target speed? all three?), and HSD-3's
  answer - do not choose, learn to choose, and keep native actions as an escape hatch -
  is generic and does not touch the map. The pre-training is unsupervised and demo-free.

### 3.6 Eysenbach, Gupta, Ibarz, Levine, "Diversity is All You Need" (DIAYN), ICLR 2019, arXiv:1802.06070; Sharma, Gu, Levine, Kumar, Hausman, "Dynamics-Aware Unsupervised Discovery of Skills" (DADS), ICLR 2020, arXiv:1907.01657

* DIAYN: learn skills by maximizing mutual information between skill index and visited
  states, with no reward; skills can then be composed by a high level.
* DADS: learn skills AND their dynamics model simultaneously, so that **model-predictive
  control can plan IN SKILL SPACE zero-shot**. Demonstrated for an Ant navigating with
  diverse locomotion primitives; reported to outperform standard MBRL, model-free
  goal-conditioned RL and prior unsupervised-skill HRL on downstream navigation.
* Applicability judgment: DADS is architecturally the most appealing of the skill-
  discovery line for us because the planner is MPC over a learned skill-transition
  model - i.e. the macro is a planner, not a policy, and it plans in a space the agent
  discovered itself. The caution is that DADS's skills on Ant are essentially
  "move in direction theta at speed v", which in surf is not a stable abstraction
  (what a held input does depends violently on whether you are on a ramp).
  Also, the local-minimum problem does not disappear: MPC over skills with a distance-
  to-goal cost inherits the same shortcut-through-the-pit failure unless the skill
  dynamics model has learned that the pit kills you - which it will, if the skills were
  rolled in the real simulator.

### 3.7 Hafner, Lee, Fischer, Abbeel, "Deep Hierarchical Planning from Pixels" (Director), NeurIPS 2022, arXiv:2206.04114

* Architecture: learn a world model (DreamerV2-style); a **goal autoencoder** compresses
  model states into small **discrete codes**; the **manager** picks a code every K steps
  and the decoder turns it into a model state that becomes the **worker's** goal; the
  worker is rewarded by a **max-cosine** similarity to the goal (matching both direction
  and magnitude); the manager maximizes task reward plus an **exploration reward**
  `log p(z|s_t) - log p(z)`. Both levels are trained inside imagined rollouts of the
  world model.
* Results: solves visual control, Atari, DMLab; notably an **egocentric 3D maze with a
  quadruped** from camera + proprioception, where flat exploration methods fail. Same
  hyperparameters across all domains and no domain knowledge.
* Applicability judgment: **the most relevant pure-RL hierarchy for us**, because the
  subgoal space is LEARNED (no privileged xy), the goals are interpretable (decodable to
  images), and the exploration bonus lives at the MANAGER level where it can actually
  move the agent to a different region, rather than at the action level where ours
  currently farms in place. The costs are large: it requires a world model over our
  depth+scalar observation, which is a different training stack from our on-policy PPO,
  and its mazes are much smaller than a surf map.

### 3.8 Nasiriany, Pong, Lin, Levine, "Planning with Goal-Conditioned Policies" (LEAP), NeurIPS 2019, arXiv:1911.08453

* A VAE over valid states defines a latent subgoal space; at TEST time, gradient descent
  optimizes a SEQUENCE of latent subgoals so that each hop is within the reach of a
  goal-conditioned policy whose value function (TDM) measures reachability.
* Applicability: "optimize the subgoal chain at test time using the policy's own value
  function as the feasibility metric" is a clean generic macro. Our version would
  optimize a chain of waypoints under a learned "can my policy get from A to B" model -
  which is the same object section 2.1 recommends building.

### 3.9 Eysenbach, Salakhutdinov, Levine, "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning" (SoRB), NeurIPS 2019, arXiv:1906.05253

* Build a graph whose NODES are observations already in the replay buffer and whose
  EDGE WEIGHTS are the goal-conditioned value function's predicted distance; run
  Dijkstra to get a sequence of subgoals; the policy only ever has to do short hops.
* Solves sparse-reward tasks over 100+ steps, including image observations.
* Applicability judgment: **top-tier for us.** It is the "the agent's own experience IS
  the map" construction, it uses no demonstrations, its nodes come from the policy's own
  states (which our section-0 rules explicitly permit), and the edge metric is learned
  reachability rather than geometric distance - the exact quantity our BFS field lacks.
  Our Go-Explore archive already IS this node set; only the edge metric is missing.

### 3.10 Zhang, Guo, Tan, Hu, Chen, "Generating Adjacency-Constrained Subgoals in Hierarchical Reinforcement Learning" (HRAC), NeurIPS 2020 spotlight, arXiv:2006.11485; Kim, Lee, Shin, "Landmark-Guided Subgoal Generation in Hierarchical Reinforcement Learning" (HIGL), NeurIPS 2021, arXiv:2110.13625

* HRAC learns an **adjacency network** so the high level only proposes subgoals the low
  level can reach within c steps; reported ~96% on AntMaze.
* HIGL samples **landmarks** by two criteria - coverage (dispersion of visited states)
  and **novelty** (prediction error) - and pulls the high level's subgoal toward the
  selected landmark, with an adjacency constraint for reachability. On AntMaze U-shape
  sparse it reports **65.1% success at 1e6 steps against HRAC's 17.6%**.
* Applicability judgment: HIGL is the closest published thing to "our Go-Explore archive
  should become the macro's goal proposal distribution". The caveat is loud: these
  numbers are on a 2-D goal space in a low-dimensional maze with an oracle position
  readout, and the adjacency network is trained on a dense visitation graph that a
  29-D Ant produces in a 10 m maze - not on a 200,000-unit surf route.

### 3.11 Park, Ghosh, Eysenbach, Levine, "HIQL: Offline Goal-Conditioned RL with Latent States as Actions", NeurIPS 2023, arXiv:2307.11949

* ONE action-free value function yields both a high-level policy that outputs a latent
  **subgoal STATE** and a low-level policy that acts toward it. The argument is that the
  value function is inaccurate for faraway goals but the DIRECTION of the gradient
  toward a nearby subgoal is still right, so hierarchy makes the method robust to value
  noise.
* Applicability: the insight "the value function's ordering is reliable locally and
  unreliable globally" is precisely our potential field's failure profile, stated in RL
  terms. It argues for using our geodesic potential ONLY over a short horizon to a
  macro-supplied subgoal, which is design #2 below.

### 3.12 Haarnoja, Hartikainen, Abbeel, Levine, "Latent Space Policies for Hierarchical Reinforcement Learning", ICML 2018

* Each layer is a max-entropy policy with latent variables; a higher layer acts by
  setting the lower layer's latents. Layers are trained bottom-up and each solves the
  task directly, so no layer is crippled.
* Applicability: the "every layer can solve the task, the hierarchy only reparameterizes
  the action space" framing is a safe way to add a macro without risking a regression -
  which matters given our 27% seed-noise floor and our history of null arms.

### 3.13 The skeptical anchor: Nachum, Tang, Lu, Gu, Lee, Levine, "Why Does Hierarchy (Sometimes) Work So Well in Reinforcement Learning?", arXiv:1909.10618, 2019

* Isolates three candidate benefits - temporally extended credit assignment, a
  semantically meaningful action space, and exploration - across locomotion, navigation
  and manipulation, and finds **exploration** is essentially the whole effect. They then
  construct non-hierarchical exploration schemes (switching the exploration noise in a
  goal/representation space rather than the action space) that match hierarchical RL.
* Applicability judgment: **read this before any macro arm is launched.** It supplies the
  mandatory control: whatever macro we build, also run the flat version that explores in
  the same space (e.g. correlated, temporally extended noise in a waypoint-shaped space)
  and see whether the hierarchy adds anything. This is cheap and it is exactly the sort
  of control our ledger has been burned for omitting.

---

## 4. PLANNING IN THE SIMULATOR OR A MODEL, WITH A LEARNED TRACKING CONTROLLER

"Plan then track". We have an advantage almost nobody in robotics has: the simulator IS
the environment, it is cheap, and it is resettable. Robotics papers pay dearly for a
model; we get the true one for free.

### 4.1 Chiang, Hsu, Fiser, Tapia, Faust, "RL-RRT: Kinodynamic Motion Planning via Learning Reachability Estimators from RL Policies", RA-L 2019, arXiv:1907.04799

* Train an obstacle-avoiding RL policy; use it as the RRT's **local steering function**
  AND train a **reachability estimator** (predicted time-to-reach between two states
  under that policy) to guide node selection; the same policy executes the plan.
* Needs: a simulator, no demos. Macro = a sampling-based planner; micro = the RL policy.
* Applicability judgment: **this is the canonical "plan with the controller in the loop"
  paper and it maps onto our problem with almost no translation.** Our nodes are
  (position, velocity) states from the policy's own rollouts; the steering function is
  the current policy driven toward a waypoint; the reachability estimator is the thing
  our BFS field should have been.

### 4.2 Yamada, Lee, Salhotra, Pertsch, Pflueger, Sukhatme, Lim, Englert, "Motion Planner Augmented Reinforcement Learning for Robot Manipulation in Obstructed Environments" (MoPA-RL), CoRL 2020, arXiv:2010.11940

* The RL agent's action space is **augmented**: a small action is executed directly, a
  large action is handed to a motion planner that realizes it collision-free. The agent
  learns when to call the planner.
* Applicability judgment: **strong and directly implementable for us.** Surf version: the
  policy's action space gains one extra "macro" dimension whose magnitude decides whether
  this decision is a normal 25 Hz key/view action or a request to a planner ("take me to
  waypoint w"). The planner runs in our simulator. There is no map constant anywhere;
  the switching threshold is on ACTION MAGNITUDE, which the agent learns.

### 4.3 Williams et al. MPPI, and the learned-prior variants: "Policy-Guided MPPI", "Residual-MPPI: Online Policy Customization for Continuous Control" (arXiv:2407.00898), "ProxPI: Proximal Prior Injection for Sampling-Based MPC under Learned-Prior Mismatch" (arXiv:2609.00941), "RGB: RL Guided Whole-Body MPPI for Humanoid Control" (arXiv:2606.25123)

* Sampling-based MPC: perturb a control sequence, roll it through a model, importance-
  weight by cost. The learned-prior family centres the sampling distribution on a policy
  so the search is cheap, and then has to solve the problem that when the policy is
  out-of-distribution the prior HURTS: ProxPI's answer is to keep nominal-centred
  sampling and attach the policy as a soft proximity cost, so the optimizer can escape
  a bad prior and fall back to vanilla MPPI performance.
* Applicability judgment: **the out-of-distribution failure mode ProxPI addresses is
  precisely ours at the pit**, where the policy's prior is confidently wrong. An MPPI
  layer over our own simulator, with the policy as a soft prior rather than a hard
  centre, would sample branches the policy would never take - including the left turn -
  and would cost only rollouts, which we have in abundance.

### 4.4 Hubert, Schrittwieser, Antonoglou, Barekatain, Schmitt, Silver, "Learning and Planning in Complex Action Spaces" (Sampled MuZero), ICML 2021, arXiv:2104.06303

* Extends MuZero's MCTS to continuous/high-dimensional action spaces by planning over a
  SAMPLED subset of actions, with a principled correction so policy evaluation and
  improvement remain unbiased. Evaluated on Go, DM Control Suite and Real-World RL.
* Applicability: an honest "algorithm redesign of the search space" candidate in the
  user's sense. The barrier is our observation (depth image) and rollout cost inside a
  tree; but with a resettable simulator and 600k steps/s, tree search over TEMPORALLY
  EXTENDED actions (see 4.6) is not absurd.

### 4.5 Hansen, Su, Wang, "TD-MPC2: Scalable, Robust World Models for Continuous Control", ICLR 2024, arXiv:2310.16828

* Local trajectory optimization (MPPI/CEM) in the latent space of a learned decoder-free
  world model, with a learned policy prior seeding the samples; 104 tasks, one
  hyperparameter set, scales to a 317M-parameter 80-task agent.
* Applicability: the "one hyperparameter set across 104 tasks" claim is the exact
  property our standing rules demand of a recipe. Worth noting as the strongest existing
  evidence that a plan-then-act method can be constant-free across domains.

### 4.6 Action chunking / macro-actions as a search-space redesign: "Reinforcement Learning with Action Chunking" (Q-chunking), arXiv:2507.07969, ICML 2025

* The policy predicts a SEQUENCE of actions over a fixed horizon and executes it
  open-loop; TD backups then skip timesteps, and exploration becomes temporally
  coherent by construction.
* Applicability judgment: **this is the cheapest instance of the user's "algorithm
  redesign / different search space".** Our `--chunk` flag already exists. The reported
  mechanism - temporally coherent exploration plus faster credit propagation - is
  exactly what a 25 Hz agent needs to discover a 2-second detour. And it introduces one
  constant (the chunk length) that is set once and carried across maps.

### 4.7 Ecoffet, Huizinga, Lehman, Stanley, Clune, "First return, then explore" (Go-Explore), Nature 590:580-586, 2021; arXiv:2004.12919

* Archive of cells; select a cell, **return** to it deterministically (in a resettable
  simulator, by restoring the state), then **explore** from it; robustify the resulting
  trajectory afterwards. Beats the state of the art on all 11 hard-exploration Atari
  games, and the paper's own robotics example is a simulated Fetch robot placing an
  object on one of four shelves, where PPO sees zero reward in 1e9 frames.
* Note on provenance: the robustification phase is imitation of the algorithm's **own**
  archived trajectories, which under our section-0 rules is self-imitation, not
  demonstration learning.
* Applicability judgment: we have already reproduced the good half of this - a
  reward-free archive search found our detour in 28 minutes. The literature's message is
  that the missing half is the RETURN-AND-CONTINUE loop with a policy, not the search.

### 4.8 Gallouedec, Dellandrea, "Cell-Free Latent Go-Explore" (LGE), ICML 2023, arXiv:2208.14928

* Removes the hand-designed cell partition - the single biggest source of domain
  knowledge in Go-Explore - by building the archive over a **learned latent
  representation** (inverse dynamics, forward dynamics, or autoencoding), and selecting
  goals by density in that latent space.
* Applicability judgment: **this is how our archive stops being map-specific.** Our
  current keys (position cell x view sector x speed rung x climb x heading) are a
  hand-designed partition, and CLAUDE.md already records that `--int-rare-speed 1200`
  was read off a map. LGE replaces the whole key with a learned latent and a density
  criterion - one mechanism, same constants on every map.

### 4.9 Hu, Chang, Rybkin, Jayaraman, "Planning Goals for Exploration" (PEG), ICLR 2023 spotlight, arXiv:2303.13002

* Learn a world model; then **plan the GOAL COMMAND** (using sampling-based planning in
  the model, e.g. CEM) so that the goal-conditioned policy, at its CURRENT skill level,
  ends up in the state with the highest expected exploration value; then run an
  exploration policy from there. A Go-phase/Explore-phase loop where the Go-phase target
  is optimized rather than sampled from the archive.
* Environments: Point Maze, Walker, **Ant Maze (long maze)**, 3-block Stack. Reported to
  train goal-conditioned policies far more efficiently than Go-Explore-style baselines
  and ablations.
* Applicability judgment: **the best-matched single paper to our problem statement in
  this whole survey.** Our documented facts are (a) an archive search finds the detour,
  (b) a curriculum from that search's own states was climbing it. PEG is the principled
  closing of that loop, and its goal-selection criterion - maximize the exploration value
  of where the policy WILL END UP, not where the goal is - is exactly the fix for our
  measured failures where "both champion-free ways of picking a leaf out of a Go-Explore
  archive are broken on this map" (the geodesic leaf stops at the wall by construction;
  the deepest-node leaf picks a dead-end pocket). PEG picks neither: it picks the goal
  whose PURSUIT maximizes downstream novelty.

### 4.10 Mendonca, Rybkin, Daniilidis, Hafner, Pathak, "Discovering and Achieving Goals via World Models" (LEXA), NeurIPS 2021, arXiv:2110.09514

* An **explorer** policy plans to reach states the model finds surprising (foresight,
  not retrospective novelty) and an **achiever** policy learns to reach them; after the
  unsupervised phase, tasks specified as goal images are solved zero-shot.
* Applicability: the explorer/achiever split is the same idea as PEG with a different
  goal-selection rule. Relevant mainly as the second data point that the goal-proposal
  mechanism is where the leverage is.

### 4.11 Florensa, Held, Wulfmeier, Zhang, Abbeel, "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017, arXiv:1707.05300

* Grow the START-STATE distribution backwards from the goal by random walks from
  existing starts, keeping the starts whose expected return falls inside a
  [R_min, R_max] band so the agent is always trained at the edge of its competence.
* Applicability: already partly implemented in our stack (the reservoir, the window, the
  `--respawn-margin` work). The literature contribution we have NOT used is the
  **return band as the selection rule** - starts are kept because the measured return
  sits in a band, never because of where they are on the map. That rule is fully generic
  and it is the honest replacement for any hand-placed window.

### 4.12 Value Iteration Networks and the differentiable-planner line: Tamar, Wu, Thomas, Levine, Abbeel, "Value Iteration Networks", NIPS 2016 best paper, arXiv:1602.02867; Lee, Parisotto, Chaplot, Xing, Salakhutdinov, "Gated Path Planning Networks", ICML 2018

* A CNN whose recurrent application IS value iteration over a learned reward/transition
  map; trainable end-to-end; **generalizes to unseen maps** far better than a plain CNN
  policy because the planning computation is structural rather than memorized.
* Applicability judgment: the reason to care is the generalization claim - a planning
  module learns a MAP-READING function, not a map. If we want a macro that transfers
  across `maps_pool/` with identical weights, a differentiable planner over a local
  geometry patch is the architecture with the best published evidence for exactly that.
  The caution is that VIN's demonstrated domains are 2-D grids; a surf macro would need
  a 3-D (or 2.5-D) patch and a velocity dimension, and nobody has published that.

### 4.13 Savinov, Dosovitskiy, Koltun, "Semi-parametric Topological Memory for Navigation", ICLR 2018, arXiv:1803.00653; Shah, Levine, "ViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints", RSS 2022, arXiv:2202.11271

* SPTM: a non-parametric graph of observations plus a learned retrieval network; no
  metric information, only connectivity; the agent builds it while exploring.
* ViKiNG: a learned local traversability/temporal-distance model, a topological graph
  built incrementally in a NEW environment, and an overhead-map or satellite heuristic
  used only as a **planning heuristic** whose accuracy is not trusted; navigates to goals
  up to 3 km away with no geometric reconstruction.
* Applicability judgment: **ViKiNG's structure is the honest description of what our
  geodesic field should be** - a HEURISTIC for search, not a reward. In ViKiNG the
  overhead map can be wrong (a building where there is none) and the system still gets
  there because the map only orders the frontier; the local model decides what is
  actually traversable. We use ours as a shaping reward, which is the one usage that
  makes a wrong heuristic fatal.

---

## 5. PAPERS WHOSE EXPLICIT POINT IS "THE LOW LEVEL IS GOOD, THE HIGH LEVEL IS MISSING"

Collected here because these are the ones to read first if only a few can be read.

1. **Fu et al. 2022 (VP-Nav)** - states the thesis: make the high-level planner aware of
   the low-level's walking capabilities; adds a safety advisor that writes what the
   controller discovered back into the planner's map, producing detours around things
   vision cannot see. Detours emerge FROM THE HIERARCHY, not from a locomotion reward.
2. **Hoeller et al. 2024 (ANYmal Parkour)** - takes five separately trained skills and
   adds nothing but a 5 Hz policy over (position, heading, time, skill index) trained on
   a sparse goal reward; 96.3% vs 60.9% for manual command sequences.
3. **Rudin et al. 2022 (IROS)** - the same robot, the same RL, only the COMMAND INTERFACE
   changed from velocity to position+time with a terminal reward: 8x the gap, 9.5x the
   climb. The clearest evidence that the interface, not the capacity, is the binding
   constraint.
4. **Lee et al. 2024 (Science Robotics)** - freezes an excellent locomotion controller
   and trains a 10 Hz navigation policy above it with PPO, a waypoint reward and a
   NOVELTY BONUS AT THE NAVIGATION LEVEL; 8.3 km of city.
5. **RoM-Nav 2026 (arXiv:2609.19272)** - says outright that a single-stage RL navigation
   pipeline does not scale to multi-level terrain, and fixes it by learning the macro on
   a reduced-order model first and then kickstarting the full-order policy with the
   locomotion policy frozen.
6. **Roth et al. 2025 (FDM, RSS)** - replaces the planner's assumptions about the
   controller with a model LEARNED FROM THE CONTROLLER'S OWN ROLLOUTS, and gets +27%
   navigation success. The most mechanical statement of "run the simulator to find out
   what your policy can do, then plan with that".
7. **Nachum et al. 2019 (arXiv:1909.10618)** - and the reason none of the above should be
   believed uncritically: the benefit is exploration, and a flat agent exploring in the
   same abstract space can match it.

---

## 6. What is NOT transferable, stated plainly

Being specific about this matters more than the positive list, because our ledger is
full of mechanisms that worked in their paper's setting and were null in ours.

* **AntMaze detours depend on a privileged 2-D goal space.** HIRO, HRAC, HIGL, HAC and
  most of section 3 solve U-shape and W-shape mazes with the torso's (x, y) handed to
  the algorithm as the subgoal space. That abstraction is exactly right for a maze and
  it is doing most of the work. Our analogue (a 3-D position waypoint) is defensible and
  generic, but the published success rates do not transfer; they are a statement about
  the abstraction, not about the algorithm.
* **Low-dimensional state.** Almost every section-3 result uses proprioceptive state
  vectors of tens of dimensions. Director and LEXA are the exceptions (pixels), and
  their mazes are small.
* **Off-policy machinery.** HIRO's off-policy correction, HAC's hindsight relabelling,
  HIQL's offline value learning, SoRB's replay-buffer graph all assume off-policy
  learning with a replay buffer. Our stack is on-policy PPO at very high throughput.
  Some of these port (SoRB's graph can be built from any stored states); some do not
  (hindsight action relabelling).
* **Human data.** Radosavovic 2024's corpus includes MoCap and YouTube human video.
  Hansen 2024's Puppeteer pretrains its tracker on 836 CMU MoCap clips. PRELUDE trains
  its high level on human cart demonstrations. ASE (Peng 2022) learns its skill
  embedding adversarially from a motion-capture dataset. **All four are forbidden as
  training sources under our section 0; their ARCHITECTURES are not.**
* **Designer-chosen skill taxonomies.** ANYmal Parkour's five skills, Barkour's
  specialists, and the gait families in Walk These Ways and Siekmann 2021 are chosen by
  a human who looked at the obstacle set. Copying a taxonomy chosen for our benchmark
  map would be exactly the ad-hoc method our section 0b forbids. A continuous command,
  or skills discovered without supervision (DIAYN/DADS/LGE), keeps us clean.
* **Hand-set switching thresholds.** The 2025 skill-switching paper is the only one that
  OPTIMIZES the switch distance rather than setting it; everywhere else the threshold is
  a constant chosen for the benchmark.
* **Classical planners assume a static traversability cost.** iPlanner, ViPlanner,
  ArtPlanner and VP-Nav all build a cost over geometry that does not depend on the
  robot's velocity. Surf traversability depends almost entirely on speed and direction
  of travel: the same voxel is passable at 2,900 u/s and lethal at 2,820 u/s (our own
  measurement at the cannonball ramp). Any cost map we build must carry at least a
  velocity coordinate, which is the single biggest engineering difference between our
  setting and theirs.
* **Racing is the honest comparator and it has no macro.** Wurman et al., "Outracing
  champion Gran Turismo drivers with deep reinforcement learning", Nature 602:223-228,
  2022, trains QR-SAC from experience with no demonstrations and beats the best human
  drivers - but the macro problem is REMOVED by construction: the observation contains
  course-progress features (a sequence of points along the track centreline ahead of the
  car), so the agent is told where the track goes and only has to solve the micro
  problem. That is the same asymmetry we have, and it is why surf's macro is the
  genuinely open part rather than a solved one we have failed to copy.

---

## 7. Architectures to try first, mapped to our code

Ranked. Each entry states what the macro emits, at what rate, how it is computed or
trained, how the micro is conditioned, the paper constants, why it should find the left
turn the potential field hides, and what would falsify it.

### #1. Reachability-certified potential: rebuild the macro field on edges the POLICY can actually traverse

**Papers:** Roth et al. 2025 (FDM + MPPI, RSS); Eysenbach et al. 2019 (SoRB); Wellhausen
& Hutter 2021; Chiang et al. 2019 (RL-RRT); Fu et al. 2022 (safety advisor).

**What the macro is:** not a policy at all - a graph, and a potential computed on it.
Nodes are (position cell, coarse velocity bin) states drawn from the POLICY'S OWN
rollouts (our reservoir and our `tools/explore_phase1.py` archive already hold them).
An edge (a -> b) exists only if a rollout of the CURRENT policy, spawned at a and
commanded toward b, actually reaches b within a time budget. The macro field is then
Dijkstra on that graph, and it replaces `build_goal_field`'s BFS-over-free-space output
in `maps/<map>.goalg_32.npz`.

**Rate:** the field is static per refresh; refresh every N million steps as the policy
improves (RL-RRT and SoRB both re-estimate reachability as the policy changes).

**How computed:** by running our own simulator. With 600k env steps/s, certifying ~1e5
candidate edges at ~2 s each is roughly 20-30 minutes of GPU time - the same order as
the 28 minutes our reward-free archive search already took. A learned FDM (Roth 2025:
predict future state + failure probability from local geometry and proprioceptive
history) is the amortized version to build second, not first.

**How the micro is conditioned:** unchanged. Same `--obs-potential norm` channel, same
shaping. Only the NUMBERS in the field change. This is the crucial property: it is a
one-file change with no new flag semantics, no new head, no tensor-shape change, and
therefore no checkpoint incompatibility.

**Why it should find the left turn:** the pit's dip exists only because BFS believes a
lateral glide across open air is an edge. Our own measurement already proved this - "a
greedy trace on the grid from vertex 1600 is 191 level steps, 5 down, 0 up, a straight
~8,700 u level glide through open air with 3,584 u of floor clearance", and gravity-
gating `dz > 0` disconnected exactly zero voxels because the deception is free flight,
not a climb. A policy-certified edge set deletes that glide outright, because no rollout
of any policy completes it. The potential then descends along the only route that
exists, which is the left turn - and our measured fact (1) says that when the field's
descent IS the route, the micro learns it in 100M-1B steps.

**Constants:** the time budget per edge and the success threshold. Both are set once in
seconds and in a success fraction, carried unchanged across all four benchmark maps and
`maps_pool/`. Nothing is read off a map.

**Falsification:** if the certified field on unitfarmer2 still dips into the pit, the
edge test is too permissive (widen the velocity binning or lengthen the budget). If the
field becomes disconnected at the start, the policy is too weak to certify anything and
the graph must be seeded from the archive's reward-free rollouts instead of the policy's.

**Risk:** this is the lowest-risk, highest-information arm in the list, and it is the
one that turns a reward defect into a geometry computation, which is where CLAUDE.md
already says the cannonball wall belonged ("cannonball's was reward arithmetic").

### #2. Waypoint interface with a terminal, time-boxed reward: the macro emits (local position, time), the micro owns everything else

**Papers:** Rudin et al. 2022 (IROS, position+time command, reward only in the last 1 s of
a 6 s episode - gaps 0.15 m -> 1.2 m, climbs 0.1 m -> 0.95 m); Skill-Nav 2025 (2-D
base-frame waypoints, r_reach = n_p/(t+eps), WP-Fixed then WP-Random curriculum on a
10x10 grid, A\*-on-walls high level, 89% omni-traverse); Hoeller et al. 2024 (5 Hz
hybrid position/heading/time command over 50 Hz skills); HIQL (value is reliable
locally, unreliable globally).

**What the macro emits:** a waypoint in the agent's own frame plus a time budget -
concretely (dx, dy, dz, T), one every 1-2 s of game time, which at `--act-every 4` and
10 ms ticks is one macro decision per 25-50 micro decisions.

**How the micro is conditioned:** retrain the micro ONCE, from scratch, as a
waypoint-conditioned policy: append (dx, dy, dz, T_remaining) to the scalar
observations, and pay the reward as Rudin does - **nothing until the last second of the
window, then `1/(1 + ||x - x*||^2)`** - plus our existing regularizers. During micro
training waypoints are sampled UNIFORMLY over a distance/direction range (Hwangbo's and
Rudin's rule), never from any map's route, with a promotion curriculum on distance.
Our geodesic potential, if kept at all, is applied only over the short horizon to the
current waypoint (HIQL's argument), so its global mistakes cannot capture the agent.

**How the macro is computed:** start with the crudest planner that works (Skill-Nav uses
A\* on a walls-only occupancy map), i.e. our existing geodesic field used ONLY to order
waypoints, never as a reward. Then, when #1 exists, use the certified graph. Then, if
needed, replace the planner with a PPO macro on a sparse finish reward plus a novelty
bonus at the macro level (Lee et al. 2024's recipe).

**Why it should find the left turn:** two independent reasons. (a) The terminal,
time-boxed reward removes the per-instant obligation to decrease distance, which is the
mechanism that makes our detour locally irrational - our own ledger already measured
this as a POTENTIAL BARRIER ("from vertex 1601 to 1680 the champion's own line RAISES
geodesic d by 8,408 u, charged at -4.24 reward... turning back at vertex 1601 is locally
optimal and the reward says so"). A reward paid only at the end of a 2 s window prices
the detour correctly. (b) The macro's action space is 4-dimensional at ~0.5 Hz, so
ordinary count-based novelty over MACRO actions covers the branch point in minutes,
whereas our `--int-view 8 / --int-speed 3` keys currently farm 3,072 keys per cell in
the micro's space and never change the route.

**Costs, honestly:** this changes tensor shapes (new scalar columns), so it is
SCRATCH-ONLY under the section-2 rules, and it needs a new micro trained from zero -
which our own evidence says is 0.75-1.5B steps to the kill-floor gate. Budget one full
arm, not an hour.

**Constants:** the waypoint sampling range (in units), the window length (in seconds),
the reward-window fraction. Set once; identical on every map. Rudin's published values -
6 s episode, 1 s reward window - are a defensible starting point converted to our tick.

### #3. PEG-style goal planning over our own archive, closing the loop we already opened

**Papers:** Hu et al. 2023 (PEG, ICLR spotlight - plan the GOAL COMMAND so the
goal-conditioned policy ends up where exploration value is highest, then explore);
Gallouedec & Dellandrea 2023 (LGE - cell-free latent archive, removes the hand-designed
cell partition); Ecoffet et al. 2021 (Go-Explore return-then-explore); Florensa et al.
2017 (reverse curriculum with a return band); Kim et al. 2021 (HIGL - landmarks by
coverage + novelty).

**What the macro emits:** a GOAL STATE for the current episode, drawn from the archive,
chosen by PEG's criterion - maximize the expected exploration value of where the
goal-conditioned policy will ACTUALLY END UP when pursuing that goal, not the novelty of
the goal itself. The episode then runs Go-phase (pursue the goal) then Explore-phase
(temperature up, novelty reward on).

**Why this specifically:** our ledger records that BOTH champion-free ways of picking a
leaf out of the archive are broken on cannonball - "the geodesic one stops a line AT the
wall by construction (that field's minimum is there), and the deepest-node one picks a
dead-end pocket at 3-5% of arc". PEG's selection rule is neither of those. It is
explicitly designed for the case where the goal you want is not the goal you should
command, because the policy cannot execute the former yet.

**How the micro is conditioned:** goal-conditioned on the archive state (or its latent),
which is the same waypoint conditioning as #2 - so #2 and #3 share one micro and can be
run as a 2x2.

**Provenance:** every state is from the policy's or the archive search's OWN rollouts;
nothing touches a human recording. LGE's latent archive additionally removes our
hand-designed key (position cell x view sector x speed rung x climb x heading) and with
it `--int-rare-speed 1200`, which CLAUDE.md already flags as a constant read off this
map's speed rung.

**Why it should find the left turn:** our measured fact (3) is that the reward-free
archive ALREADY finds the detour in 28 minutes and a curriculum from its own states was
climbing it. The missing piece is not discovery; it is the return-and-consolidate loop
with a selection rule that does not collapse onto the wall or into a pocket. That is
exactly what PEG supplies, and it is the smallest delta from where we already are.

**Falsification:** if PEG's goal selection still collapses onto the wall, the exploration
value estimate is the problem, not the selection; fall back to HIGL's two-criterion
landmark sampling (coverage dispersion + model prediction error) which is simpler and
has published AntMaze numbers (65.1% vs 17.6%).

### #4. MPPI / tree search in our own simulator with the policy as a SOFT prior

**Papers:** Roth et al. 2025 (zero-shot MPPI over a learned FDM); ProxPI 2026 (keep
nominal-centred sampling, attach the policy as a soft proximity cost so the optimizer can
escape a wrong prior); Residual-MPPI 2024; Yamada et al. 2020 (MoPA-RL - the agent learns
when to call the planner); Hubert et al. 2021 (Sampled MuZero); Hansen et al. 2024
(TD-MPC2, one hyperparameter set across 104 tasks); Q-chunking 2025 (temporally coherent
action sequences).

**What the macro is:** at each macro tick, fork the simulator, sample K action-chunk
sequences (chunk length ~0.5-1 s), roll them with the real physics, score by
(certified-graph potential at the end) + (failure probability) - (time), and execute the
first chunk of the best. The policy supplies the sampling distribution, but as a SOFT
prior (ProxPI's correction), so branches the policy would never take are still sampled.

**Why it should find the left turn:** this is the only design in the list that does not
need the field to be right at all - it evaluates branches by running them. The pit
branch terminates in a fall within the horizon and is scored accordingly. It is also the
design that most directly matches the user's "algorithm redesign with optimal search
space", and it exploits the one asset robotics papers do not have: a cheap, resettable,
exact simulator.

**Cost and risk:** by far the most expensive per environment step, and it changes the
data-collection loop rather than a flag. The honest staging is: build #1 first (the
scoring function this search needs), run #2 (the interface this search commands), and
only then attempt #4. Q-chunking (`--chunk`, already in our code) is the cheap
down-payment: it buys temporally coherent exploration with none of the search machinery,
and it is the thing to measure before committing to a tree.

### Mandatory controls, from the literature's own scepticism

* **Nachum et al. 2019's flat control.** For whichever of #2/#3 is run, also run a FLAT
  agent that explores with temporally extended, correlated noise in the same abstract
  space and no second network. If the flat control matches, the hierarchy is a
  reparameterization and should be reported as such.
* **Barkour's distillation control.** A single policy distilled from the hierarchy
  matched the switching hierarchy on that benchmark. If a macro works, check whether it
  can be distilled away - a distilled single policy would be a simpler recipe and our
  standing rules prefer it.
* **Our own 27% seed-noise floor and the binary-gate pathology.** None of these arms can
  be ranked on end-of-run `race/eval_progress`. Report the GATE CLEARED and the STEP at
  which it cleared, as CLAUDE.md's retraction section requires.

---

## 8. Full reference list (56 entries)

Locomotion controllers and command interfaces
1. Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26):eaau5872, 2019.
2. Lee, Hwangbo, Wellhausen, Koltun, Hutter. Learning quadrupedal locomotion over challenging terrain. Science Robotics 5(47):eabc5986, 2020.
3. Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter. Learning robust perceptive locomotion for quadrupedal robots in the wild. Science Robotics 7(62):eabk2822, 2022.
4. Rudin, Hoeller, Reist, Hutter. Learning to Walk in Minutes Using Massively Parallel Deep RL. CoRL 2022. arXiv:2109.11978.
5. Kumar, Fu, Pathak, Malik. RMA: Rapid Motor Adaptation for Legged Robots. RSS 2021. arXiv:2107.04034.
6. Margolis, Agrawal. Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior. CoRL 2022. arXiv:2212.03238.
7. Margolis, Yang, Paigwar, Chen, Agrawal. Rapid Locomotion via Reinforcement Learning. RSS 2022 / IJRR 2024.
8. Siekmann, Godse, Fern, Hurst. Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition. ICRA 2021. arXiv:2011.01387.
9. Li, Cheng, Peng, Abbeel, Levine, Berseth, Sreenath. RL for Robust Parameterized Locomotion Control of Bipedal Robots. ICRA 2021. arXiv:2103.14295.
10. Radosavovic, Xiao, Zhang, Darrell, Malik, Sreenath. Humanoid Locomotion as Next Token Prediction. NeurIPS 2024. arXiv:2402.19469.
11. Radosavovic et al. Real-World Humanoid Locomotion with Reinforcement Learning. Science Robotics, 2024.
12. Liao, Zhang, Sreenath et al. Berkeley Humanoid: A Research Platform for Learning-based Control. arXiv:2407.21781, 2024.

Navigation on top of learned locomotion
13. Fu, Kumar, Agarwal, Qi, Malik, Pathak. Coupling Vision and Proprioception for Navigation of Legged Robots (VP-Nav). CVPR 2022. arXiv:2112.02094.
14. Hoeller, Rudin, Sako, Hutter. ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots. Science Robotics 9(88):eadi7566, 2024. arXiv:2306.14874.
15. Rudin, Hoeller, Bjelonic, Hutter. Advanced Skills by Learning Locomotion and Local Navigation End-to-End. IROS 2022. arXiv:2209.12827.
16. Lee, Bjelonic, Reske, Wellhausen, Miki, Hutter. Learning robust autonomous navigation and locomotion for wheeled-legged robots. Science Robotics 9(89):eadi9641, 2024. arXiv:2405.01792.
17. Kareer, Yokoyama, Batra, Ha, Truong. ViNL: Visual Navigation and Locomotion Over Obstacles. ICRA 2023. arXiv:2210.14791.
18. Hoeller, Wellhausen, Farshidian, Hutter. Learning a State Representation and Navigation in Cluttered and Dynamic Environments. RA-L 6(3):5081-5088, 2021. arXiv:2103.04351.
19. Truong, Yarats, Li, Meier, Chernova, Batra, Rai. Learning Navigation Skills for Legged Robots with Learned Robot Embeddings. IROS 2021.
20. Sorokin, Tan, Liu, Ha. Learning to Navigate Sidewalks in Outdoor Environments. RA-L 7:3906-3913, 2022. arXiv:2109.05603.
21. Seo, Gupta, Zhu, Skoutnev, Sentis, Zhu. Learning to Walk by Steering (PRELUDE). ICRA 2023. arXiv:2209.09233. [human demos - forbidden as a source]
22. Caluwaerts et al. Barkour: Benchmarking Animal-level Agility with Quadruped Robots. arXiv:2305.14654, 2023.
23. Cheng, Shi, Agarwal, Pathak. Extreme Parkour with Legged Robots. ICRA 2024. arXiv:2309.14341.
24. Zhuang, Fu, Wang, Atkeson, Schwertfeger, Finn, Zhao. Robot Parkour Learning. CoRL 2023. arXiv:2309.05665.
25. Zhuang, Yao, Zhao. Humanoid Parkour Learning. CoRL 2024. arXiv:2406.10759.
26. Agarwal, Kumar, Malik, Pathak. Legged Locomotion in Challenging Terrains using Egocentric Vision. CoRL 2022. arXiv:2211.07638.
27. Yang, Yang, Wang. Neural Volumetric Memory for Visual Locomotion Control. CVPR 2023. arXiv:2304.01201.
28. Wellhausen, Hutter. Rough Terrain Navigation for Legged Robots using Reachability Planning and Template Learning. IROS 2021.
29. Frey, Roth, Cadena, Hutter et al. ArtPlanner: Robust Legged Robot Navigation in the Field. arXiv:2303.01420, 2023.
30. Roth, Frey, Cadena, Hutter. Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation. RSS 2025. arXiv:2504.19322.
31. Yang, Wang, Cadena, Hutter. iPlanner: Imperative Path Planning. RSS 2023.
32. Roth, Nubert, Yang, Mittal, Hutter. ViPlanner: Visual Semantic Imperative Learning for Local Navigation. ICRA 2024. arXiv:2310.00982.
33. Frey, Mattamala, Roth, Cadena, Fallon, Hutter et al. Resilient Legged Local Navigation. ICRA 2024. arXiv:2310.03581.
34. Skill-Nav: Enhanced Navigation with Versatile Quadrupedal Locomotion via Waypoint Interface. arXiv:2506.21853, 2025.
35. Discovery of skill switching criteria for learning agile quadruped locomotion. arXiv:2502.06676, 2025 / Frontiers in Robotics and AI, 2026.
36. Learning Safe Humanoid Navigation from Reduced Order Models (RoM-Nav). arXiv:2609.19272, 2026.
37. RL with Data Bootstrapping for Dynamic Subgoal Pursuit in Humanoid Robot Navigation. arXiv:2506.02206, 2025.

Hierarchical RL: locomotion below, navigation above
38. Nachum, Gu, Lee, Levine. Data-Efficient Hierarchical Reinforcement Learning (HIRO). NeurIPS 2018. arXiv:1805.08296.
39. Nachum, Tang, Lu, Gu, Lee, Levine. Why Does Hierarchy (Sometimes) Work So Well in RL? arXiv:1909.10618, 2019.
40. Nachum, Gu, Lee, Levine. Near-Optimal Representation Learning for Hierarchical RL. ICLR 2019. arXiv:1810.01257.
41. Levy, Konidaris, Platt, Saenko. Learning Multi-Level Hierarchies with Hindsight (HAC). ICLR 2019. arXiv:1712.00948.
42. Vezhnevets, Osindero, Schaul, Heess, Jaderberg, Silver, Kavukcuoglu. FeUdal Networks for HRL. ICML 2017. arXiv:1703.01161.
43. Bacon, Harb, Precup. The Option-Critic Architecture. AAAI 2017. arXiv:1609.05140.
44. Bagaria, Konidaris. Option Discovery using Deep Skill Chaining. ICLR 2020.
45. Bagaria, Senthil, Konidaris. Skill Discovery for Exploration and Planning using Deep Skill Graphs. ICML 2021.
46. Gehring, Synnaeve, Krause, Usunier. Hierarchical Skills for Efficient Exploration (HSD-3). NeurIPS 2021. arXiv:2110.10809.
47. Eysenbach, Gupta, Ibarz, Levine. Diversity is All You Need (DIAYN). ICLR 2019. arXiv:1802.06070.
48. Sharma, Gu, Levine, Kumar, Hausman. Dynamics-Aware Unsupervised Discovery of Skills (DADS). ICLR 2020. arXiv:1907.01657.
49. Hafner, Lee, Fischer, Abbeel. Deep Hierarchical Planning from Pixels (Director). NeurIPS 2022. arXiv:2206.04114.
50. Haarnoja, Hartikainen, Abbeel, Levine. Latent Space Policies for Hierarchical RL. ICML 2018.
51. Nasiriany, Pong, Lin, Levine. Planning with Goal-Conditioned Policies (LEAP). NeurIPS 2019. arXiv:1911.08453.
52. Zhang, Guo, Tan, Hu, Chen. Generating Adjacency-Constrained Subgoals in HRL (HRAC). NeurIPS 2020. arXiv:2006.11485.
53. Kim, Lee, Shin. Landmark-Guided Subgoal Generation in HRL (HIGL). NeurIPS 2021. arXiv:2110.13625.
54. Park, Ghosh, Eysenbach, Levine. HIQL: Offline Goal-Conditioned RL with Latent States as Actions. NeurIPS 2023. arXiv:2307.11949.
55. Hansen, Su, Wang et al. Hierarchical World Models as Visual Whole-Body Humanoid Controllers (Puppeteer). arXiv:2405.18418, 2024. [MoCap-pretrained tracker]
56. Peng, Guo, Halper, Levine, Fidler. ASE: Large-Scale Reusable Adversarial Skill Embeddings. ACM TOG 41(4):94, 2022. [MoCap dataset]

Planning, search and exploration
57. Chiang, Hsu, Fiser, Tapia, Faust. RL-RRT: Kinodynamic Motion Planning via Learning Reachability Estimators from RL Policies. RA-L 2019. arXiv:1907.04799.
58. Yamada, Lee, Salhotra, Pertsch, Pflueger, Sukhatme, Lim, Englert. Motion Planner Augmented RL (MoPA-RL). CoRL 2020. arXiv:2010.11940.
59. Hubert, Schrittwieser, Antonoglou, Barekatain, Schmitt, Silver. Learning and Planning in Complex Action Spaces (Sampled MuZero). ICML 2021. arXiv:2104.06303.
60. Hansen, Su, Wang. TD-MPC2: Scalable, Robust World Models for Continuous Control. ICLR 2024. arXiv:2310.16828.
61. ProxPI: Proximal Prior Injection for Sampling-Based MPC under Learned-Prior Mismatch. arXiv:2609.00941, 2026.
62. Residual-MPPI: Online Policy Customization for Continuous Control. arXiv:2407.00898, 2024.
63. Ecoffet, Huizinga, Lehman, Stanley, Clune. First return, then explore (Go-Explore). Nature 590:580-586, 2021. arXiv:2004.12919.
64. Gallouedec, Dellandrea. Cell-Free Latent Go-Explore (LGE). ICML 2023. arXiv:2208.14928.
65. Hu, Chang, Rybkin, Jayaraman. Planning Goals for Exploration (PEG). ICLR 2023 spotlight. arXiv:2303.13002.
66. Mendonca, Rybkin, Daniilidis, Hafner, Pathak. Discovering and Achieving Goals via World Models (LEXA). NeurIPS 2021. arXiv:2110.09514.
67. Florensa, Held, Wulfmeier, Zhang, Abbeel. Reverse Curriculum Generation for RL. CoRL 2017. arXiv:1707.05300.
68. Eysenbach, Salakhutdinov, Levine. Search on the Replay Buffer (SoRB). NeurIPS 2019. arXiv:1906.05253.
69. Savinov, Dosovitskiy, Koltun. Semi-parametric Topological Memory for Navigation. ICLR 2018. arXiv:1803.00653.
70. Shah, Levine. ViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints. RSS 2022. arXiv:2202.11271.
71. Tamar, Wu, Thomas, Levine, Abbeel. Value Iteration Networks. NIPS 2016 (best paper). arXiv:1602.02867.
72. Lee, Parisotto, Chaplot, Xing, Salakhutdinov. Gated Path Planning Networks. ICML 2018.
73. Wohlke, Schmitt, van Hoof. Hierarchies of Planning and Reinforcement Learning for Robot Navigation. ICRA 2021. arXiv:2109.11178.
74. Reinforcement Learning with Action Chunking (Q-chunking). ICML 2025. arXiv:2507.07969.
75. Trott, Zheng, Xiong, Socher. Keeping Your Distance: Solving Sparse Reward Tasks Using Self-Balancing Shaped Rewards (Sibling Rivalry). NeurIPS 2019. arXiv:1911.01417.
76. Ha, Xu, Tan, Levine, Tan. Learning to Walk in the Real World with Minimal Human Effort. CoRL 2020. arXiv:2002.08550.
77. Rimon, Koditschek. Exact robot navigation using artificial potential functions. IEEE T-RO 8(5):501-518, 1992.
78. Wurman et al. Outracing champion Gran Turismo drivers with deep reinforcement learning. Nature 602:223-228, 2022.

---

## Appendix: the two things worth writing on the whiteboard

**A. Nobody in robotics rewards distance-to-goal at every instant on a field computed
over free space.** They either (i) compute the field over a CAPABILITY graph, (ii) pay
the reward only at the end of a short, time-boxed window, or (iii) both. Our shaping is
the one design that makes a wrong heuristic fatal instead of merely inefficient. That is
the difference between our setting and theirs, and it is one file plus one flag wide.

**B. The macro layer's value is that it makes the exploration problem small.** Nachum
2019 proved it, Lee 2024 relied on it (the novelty bonus lives in the NAVIGATION policy,
not the joints), and our own 34 null exploration cells are the negative proof: novelty in
a 6-dimensional 25 Hz action space farms in place; novelty in a 4-dimensional 0.5 Hz
waypoint space would have to move the agent somewhere new. Whatever macro we build, the
metric that says it is working is not the reward - it is whether the branch point at the
pit gets visited by both branches within the first hundred million steps.
