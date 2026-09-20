# Literature survey: detour navigation under a deceptive distance reward

Compiled 2026-09-20. Web research only; no code was run and no other file in
this repo was touched. ASCII only.

## 0. The problem, restated in the literature's vocabulary

The measured failure is this: with a EUCLIDEAN potential the agent walks to the
first wall and shuffles there for the whole episode; the detour that solves the
map temporarily RAISES the potential by 195 u (about 9.6 reward, ~5% of the
episode's total shaping) on one rung and by 823 u (~40% of the total shaping) on
another. With a GEODESIC potential (a BFS through free space) the same agent
finishes every rung in 100-200M steps.

Three separate names exist for this in the literature, and they are not the same
thing. Keeping them apart is what decides which papers are relevant.

1. **Deception** (Lehman and Stanley 2011). The objective function's gradient
   points into a cul-de-sac. The search is not short of coverage, it is being
   actively pulled the wrong way. This is the correct primary diagnosis here:
   the repo's own Go-Explore phase 1 found the detour in 28 minutes of
   simulator time with NO reward at all, which proves the state is reachable by
   undirected search and that the blocker is the objective, not the geometry.

2. **Hard exploration** (Montezuma, MyWayHome, DM-HARD-8). The rewarding state
   is hard to reach at all by any undirected process. Most of the exploration
   literature (RND, ICM, counts, NGU, BYOL-Explore) targets THIS, which is
   probably why count-based novelty and RND were null here: they are designed to
   solve a problem this task does not have.

3. **Credit assignment / local policy search**. PPO is a local search in policy
   space. Ng, Harada and Russell (1999) guarantee that potential-based shaping
   preserves the OPTIMAL policy; it guarantees nothing about what a local search
   finds, and Wiewiora (2003) makes the reason concrete: potential-based shaping
   is EXACTLY equivalent to initializing Q with the potential. A Euclidean
   potential is therefore a wrong PRIOR that the policy must un-learn, not a
   bias that vanishes.

Two arithmetic checks follow from (3) and should be done before spending a GPU
hour on anything in this survey, because they are free and they are generic:

* **Is the shaping term `gamma * Phi(s') - Phi(s)`, or just `Phi(s') - Phi(s)`?**
  Ng 1999's invariance requires the gamma factor. Without it the shaping is not
  potential-based at all, and the error is not neutral: it systematically
  charges long trajectories, i.e. it charges detours twice. This is the single
  cheapest thing to verify.
* **Is the discounted terminal bonus actually larger than the detour deficit?**
  On the 823 u rung the deficit is ~40 reward units paid immediately, against a
  +50 bonus arriving hundreds of decisions later. At gamma = 0.9995 per physics
  tick and act_every 4, 200 decisions of discount is a factor of about 0.67, so
  +50 is worth about +33 at the moment of choice. If the shaping is implemented
  without the gamma factor (above), the optimal policy under the shaped
  objective may genuinely BE to refuse the detour, and no exploration method can
  fix a reward whose argmax is wrong. Grzes (2017) is the matching correction on
  the termination side and the repo already has it as the "death charge".

The rest of this file is grouped as requested. Every entry states what the
method needs, what it reported on the most detour-like task available, and a
one-line verdict against the constraints (no demos, no map constants, PPO is the
trainer but off-policy parts are allowed).

---

## 1. Classic framings: deception, novelty search, quality-diversity, shaping theory

### 1.1 Deception and novelty search

**Abandoning Objectives: Evolution Through the Search for Novelty Alone.**
Joel Lehman, Kenneth O. Stanley. Evolutionary Computation 19(2):189-223, 2011.
(Also Lehman and Stanley, GECCO 2011, "Novelty search and the problem with
objectives".)
*Task/reward:* the deceptive "hard maze" - a 2D maze whose cul-de-sacs lie
closer to the goal in straight-line distance than the true path. Fitness =
final Euclidean distance to goal. This is literally the repo's reward function.
*Mechanism:* discard the objective. Score each individual by the average
distance, in a behaviour space (here: final (x,y)), to its k nearest neighbours
in an archive of past behaviours. Select on that alone.
*Needs:* a population, a behaviour descriptor, an archive. No map, no demos, no
model.
*Result:* objective-based search solved the hard maze in 3/40 runs; novelty
search in 39/40. The paper's own framing - "the fitness gradient is no more
helpful than random search" - is the canonical statement of this failure mode.
*Verdict:* the correct diagnosis, the wrong algorithm class for 2048-env PPO.
Its descendants (NS-ES, MAP-Elites, QD-RL) are the usable forms. Important
caveat: the behaviour descriptor (final x,y) IS a map-aware choice; position
cells are the same choice the repo already made and found null as a BONUS -
the difference is that novelty search uses it as the ONLY objective, not as an
additive bonus that a dense reward can out-vote.

**Improving Exploration in Evolution Strategies for Deep RL via a Population of
Novelty-Seeking Agents (NS-ES, NSR-ES, NSRA-ES).** Edoardo Conti, Vashisht
Madhavan, Felipe Petroski Such, Joel Lehman, Kenneth Stanley, Jeff Clune.
NeurIPS 2018. arXiv:1712.06560.
*Task/reward:* Atari and a Humanoid locomotion task with a DECEPTIVE TRAP - a
three-sided enclosure whose exit requires moving away from the reward gradient.
Dense reward plus the trap.
*Mechanism:* run ES with a meta-population of M agents; the gradient estimate is
a weighted combination of the reward gradient and a novelty gradient over a
behaviour characteristic. NSRA-ES adapts the weight online, increasing the
novelty weight when reward stagnates.
*Needs:* population of M = 5 agents, behaviour archive, massive parallelism
(which matches this repo's 2048 envs and 600k steps/s better than most).
*Result:* plain ES stays in the trap indefinitely; NSRA-ES escapes and reaches
higher final reward.
*Verdict:* generic, no map constants, and the ADAPTIVE weight (raise novelty
only while reward is flat) is the single most transferable idea in the paper.
But ES is a different trainer, not a PPO add-on, and the behaviour
characteristic is again a position descriptor.

**Illuminating search spaces by mapping elites (MAP-Elites).** Jean-Baptiste
Mouret, Jeff Clune. arXiv:1504.04909, 2015.
**Policy Gradient Assisted MAP-Elites (PGA-MAP-Elites).** Olle Nilsson,
Antoine Cully. GECCO 2021.
**QD-RL: Efficient Mixing of Quality and Diversity in RL.** Geoffrey Cideron,
Thomas Pierrot, Nicolas Perrin, Karim Beguir, Olivier Sigaud. arXiv:2006.08505,
2020.
*Task/reward:* QD-RL explicitly targets point-maze and ant-maze with deceptive
reward gradients.
*Mechanism:* maintain a grid ("archive") of elites indexed by a behaviour
descriptor; each cell keeps the highest-performing solution with that behaviour.
PGA-MAP-Elites and QD-RL add a policy-gradient variation operator (a TD3 critic)
so that large neural controllers can be evolved rather than small ones.
*Needs:* archive, behaviour descriptor, population of policies, off-policy critic.
*Result:* QD-RL solves point-maze and ant-maze where the deceptive gradient traps
policy gradient alone; the papers' summary is that "optimizing for diversity is
required to overcome the deceptive nature of the reward" while the quality term
supplies asymptotic performance.
*Verdict:* this is the "population search / algorithm redesign" option the user
named. Genuinely a different search space, generic, no map constants. Cost: a
second optimizer plus an archive; and the behaviour descriptor is again final
position, about which this repo already has negative evidence in bonus form.

### 1.2 Potential-based shaping theory: what it does and does not promise

**Policy Invariance Under Reward Transformations: Theory and Application to
Reward Shaping.** Andrew Y. Ng, Daishi Harada, Stuart Russell. ICML 1999.
*Statement:* `F(s,a,s') = gamma*Phi(s') - Phi(s)` is necessary and sufficient
for the optimal policy to be unchanged. The paper's own worked example is a
shaping "bug" produced by a non-potential shaping term.
*Verdict for this problem:* the theorem is about the OPTIMUM and is silent about
the optimization path. It also means the Euclidean potential cannot be blamed
for changing the answer - only for changing the landscape. Use it as the
correctness check named in section 0, not as a defence of the current reward.

**Potential-Based Shaping and Q-Value Initialization are Equivalent.**
Eric Wiewiora. JAIR 19:205-208, 2003. arXiv:1106.5267. (See also Wiewiora,
Cottrell and Elkan, ICML 2003, "Principled methods for advising RL agents",
which extends this to state-ACTION advice and shows look-ahead advice is the
form that does not break invariance.)
*Statement:* a learner with `Q_0(s,a) = Phi(s)` makes identical updates to a
learner receiving potential-based shaping.
*Verdict:* the cleanest way to see why the Euclidean potential is fatal to a
local search - it is a confident wrong initialization of the value function
along the straight line, and PPO must pay real samples to overwrite it.

**Dynamic Potential-Based Reward Shaping.** Sam Devlin, Daniel Kudenko. AAMAS
2012.
*Statement:* invariance survives a potential that CHANGES during learning,
`F = gamma*Phi(s',t') - Phi(s,t)`, provided transitions are consumed with the
potential that was current when they were generated.
*Verdict:* this is the licence to replace the hand-built potential with a LEARNED
distance (section 3) while keeping the guarantee - and note the guarantee does
NOT transfer to off-policy replay with stale reward labels. On-policy PPO is
exactly the setting where dynamic PBRS is sound. That is a rare structural
advantage of this repo's trainer and it should be used.

**Reward Shaping in Episodic Reinforcement Learning.** Marek Grzes. AAMAS 2017.
*Statement:* in the episodic case the `gamma^n Phi(s_n)` term at termination
breaks invariance unless the terminal potential is forced to zero.
*Verdict:* already implemented here as the death charge. Keep it; it is the
reason a policy that dies at the wall cannot bank the potential it accumulated
on the way in.

**Learning to Utilize Shaping Rewards: A New Approach of Reward Shaping.**
Yujing Hu, Weixun Wang, Hangtian Jia, Yixiang Wang, Yingfeng Chen, Jianye Hao,
Feng Wu, Changjie Fan. NeurIPS 2020. arXiv:2011.02669.
*Task/reward:* sparse-reward cartpole and MuJoCo with imperfect hand-written
shaping.
*Mechanism:* bi-level optimization. The lower level optimizes the policy on
`r + z_phi(s) * f(s)`; the upper level optimizes the shaping WEIGHT function
z_phi by the gradient of the TRUE (unshaped) return with respect to z.
*Needs:* an extra weight network and a second gradient. No map, no demos.
*Result:* "fully exploit beneficial shaping rewards, and meanwhile ignore
unbeneficial shaping rewards or even transform them into beneficial ones."
*Verdict:* the principled version of "turn the shaping off where it lies". The
true-return gradient is estimated from episodes that reach the goal, which this
agent never does, so it cannot bootstrap alone - but as a SECOND stage after any
section 6 mechanism produces a first finish, it is the generic way to stop the
Euclidean potential being re-learned.

**Efficient Potential-based Exploration in RL using Inverse Dynamic Bisimulation
Metric (LIBERTY).** Yiming Wang, Ming Yang, Renzhi Dong, Binbin Sun, Furui Liu,
Leong Hou U. NeurIPS 2023.
*Mechanism:* build the exploration bonus itself as a POTENTIAL over a learned
inverse-dynamics bisimulation metric, so the bonus is policy-invariant by
construction.
*Needs:* metric network; standard deep RL otherwise.
*Result:* MuJoCo and Atari. No maze / detour result.
*Verdict:* the right SHAPE for any bonus added to this repo - a potential
difference cannot be farmed by oscillating in place, whereas an additive bonus
can - but no detour evidence. Treat as a design constraint, not a candidate.

**Potential-based reward shaping in Sokoban.** Zhao Yang, Mike Preuss, Aske
Plaat. arXiv:2109.05022, 2021.
*Verdict:* a small, honest data point that a hand-designed potential helps on a
puzzle domain only when it happens to be aligned with the solution - which is
the general statement of this repo's geodesic-versus-Euclidean result.

---

## 2. Exploration methods evaluated on maze / obstacle-detour tasks

Ordered from "already validated dead here" to "not yet tried here".

### 2.1 Already null in this repo (do not retest)

**Unifying Count-Based Exploration and Intrinsic Motivation.** Marc Bellemare,
Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Remi Munos.
NeurIPS 2016. arXiv:1606.01868. (And **#Exploration: A Study of Count-Based
Exploration for Deep RL**, Haoran Tang, Rein Houthooft, Davis Foote, Adam
Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel,
NeurIPS 2017, arXiv:1611.04717 - the SimHash version the repo's position-cell
counts reproduce.)
*Verdict here:* null, and the literature explains why. A per-step bonus
`beta / sqrt(N(s))` has to out-bid a SUSTAINED shaping deficit (9.6 reward held
over the whole detour leg on one rung, ~40 on the other), and the bonus DECAYS
exactly as the agent begins to commit, because the first cells of the detour
become visited. The payoff profile is backwards. It also pays for wiggling: a
shuffle at the wall that touches new cells scores the same per-cell bonus as a
committed excursion, which is the "farming in place" already logged here.

**Curiosity-driven Exploration by Self-supervised Prediction (ICM).** Deepak
Pathak, Pulkit Agrawal, Alexei Efros, Trevor Darrell. ICML 2017.
arXiv:1705.05363.
*Task:* VizDoom MyWayHome sparse and "very sparse" - a STATIC 9-room maze, first
person, on-policy A3C/PPO. The closest published environment to this repo.
*Verdict:* solves MyWayHome-sparse, but is the weakest member of the episodic
family, and Trott et al. (2019) report PPO+ICM discovering the goal in only 2 of
5 runs on a deceptive point maze. Counted as dead here already.

**Exploration by Random Network Distillation (RND).** Yuri Burda, Harrison
Edwards, Amos Storkey, Oleg Klimov. ICLR 2019. arXiv:1810.12894.
*Verdict:* null here. RND is a GLOBAL (lifetime) novelty signal. After 100M
steps at the wall, neither the wall nor the first step of the detour is novel -
both were sampled and discarded thousands of times. The bonus is gone before the
behaviour that needs it can be reinforced.

**State Entropy Maximization with Random Encoders (RE3).** Younggyo Seo, Lili
Chen, Jinwoo Shin, Honglak Lee, Pieter Abbeel, Kimin Lee. ICML 2021.
arXiv:2102.09430. **APT / Behavior From the Void.** Hao Liu, Pieter Abbeel.
NeurIPS 2021. arXiv:2103.04551. **Proto-RL.** Denis Yarats, Rob Fergus,
Alessandro Lazaric, Lerrel Pinto. ICML 2021. arXiv:2102.11271.
*Mechanism:* a k-NN particle estimate of state entropy in a random (RE3) or
learned (APT, Proto-RL) encoder space, used as an intrinsic reward or for
reward-free pre-training.
*Result:* DMControl and MiniGrid sample-efficiency gains; no deceptive-detour
result in any of the three.
*Verdict:* the same family as the repo's null count bonus - coverage pressure,
which is not the missing ingredient. Listed so the conclusion is not re-derived.

### 2.2 Bonuses measured in STEPS rather than in cells (the important subfamily)

This is the distinction that matters most here. A bonus over POSITION cells pays
for any new cell, including the ones a wall-shuffle generates. A bonus over
TEMPORAL distance pays only for states that are many environment steps from
everything seen this episode - which is what a detour is, and what a shuffle
is not.

**Episodic Curiosity through Reachability.** Nikolay Savinov, Anton Raichuk,
Raphael Marinier, Damien Vincent, Marc Pollefeys, Timothy Lillicrap, Sylvain
Gelly. ICLR 2019. arXiv:1810.02274.
*Task/reward:* VizDoom static (SINGLETON) maze, sparse and very sparse reward,
first-person pixels; DMLab sparse navigation. The paper's own motivation is that
the agent must take a DETOUR - observations that are close in appearance or in
Euclidean terms but many steps apart in reachability.
*Mechanism:* an R-network (siamese embedding E plus comparator C) is trained to
classify whether two observations are reachable within k environment steps. An
episodic memory M holds this episode's embeddings. The bonus is
`b = alpha * (beta - F(C(M, e)))` where F aggregates the comparisons over
memory; an observation joins memory only if judged novel.
*Constants:* memory K = 200; aggregation F = 90th percentile; reachability
threshold k = 5 actions; beta = 0.5 for fixed-duration episodes and 1.0 for
variable duration; R-network pre-trained on 2.5M steps of random-policy data (at
600k steps/s that is about 4 seconds of wall clock here); reported overhead
1.84x over plain PPO.
*Needs:* one extra network (ResNet-18 in the paper; far smaller would do at
64x32 depth) and an episodic buffer. No map, no demos, no policy replay buffer.
*Result:* at least 22x faster convergence than ICM on the VizDoom static maze, to
100% success; beats ICM on all three DMLab sparse tasks; with no extrinsic reward
at all, covers at least 44x more area than ICM.
*Verdict:* **the single most on-point published paper for this repo.** On-policy
PPO, first-person, singleton maze, detours, no map, no demos, and its constants
are expressed in STEPS (k = 5, K = 200) and are therefore map-independent by
construction. Skeptical notes: the R-network is bootstrapped from random-policy
data, so on a map where a random policy cannot leave the first room it will be
badly calibrated far from the start (use the "variable duration" variant, which
trains it online); and "22x faster than ICM" is a sample-efficiency claim on a
maze whose goal IS reachable by random walk - Savinov's tasks are sparse, not
deceptive, so this is evidence for the bonus, not evidence that it beats a
competing dense reward.

**Episodic Novelty Through Temporal Distance (ETD).** Yuhua Jiang et al.
ICLR 2025. arXiv:2501.15418.
*Mechanism:* the 2025 replacement for the R-network. Temporal distance (expected
number of steps between two states) is estimated by contrastive learning with a
successor-distance parameterization (energy = a potential network minus a
quasimetric network, which forces the estimate to behave as a quasimetric). The
intrinsic reward is the aggregated MINIMUM temporal distance between the current
state and episodic memory.
*Needs:* contrastive distance network, episodic memory. No map, no demos.
*Result:* beats NovelD, E3B and EC on MiniGrid, Crafter and MiniWorld;
near-optimal on ObstructedMaze-Full within 20M steps. Argues explicitly that
temporal distance is invariant to state representation, which removes the
noisy-TV failure mode.
*Verdict:* strictly the better-engineered Savinov 2019, sharing every property
that makes Savinov relevant here. Younger, so less independently replicated. Its
distance network is directly reusable as the learned potential of section 3 - one
network, two uses.

**Exploration via Elliptical Episodic Bonuses (E3B).** Mikael Henaff, Roberta
Raileanu, Minqi Jiang, Tim Rocktaschel. NeurIPS 2022. arXiv:2210.05805.
**A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs.**
Mikael Henaff, Minqi Jiang, Roberta Raileanu. ICML 2023. arXiv:2306.03236.
*Mechanism:* `b(s_t) = phi(s_t)^T C_{t-1}^{-1} phi(s_t)` with
`C_{t-1} = sum_i phi(s_i) phi(s_i)^T + lambda I`, reset at every episode
boundary; phi is trained by INVERSE DYNAMICS (predict a_t from phi(s_t) and
phi(s_{t+1})), which discards everything the agent cannot control. This is a
continuous-state generalization of an episodic count.
*Constants:* lambda = 0.1 (swept over 0.01 / 0.1 / 1.0); intrinsic coefficient
beta = 1.0 on MiniHack but 3e-6 on VizDoom (a seven-order-of-magnitude gap - this
constant is NOT map-independent, which is a problem under this repo's rules);
feature dim 256-1024; IMPALA lr 1e-4; DD-PPO on Habitat.
*Result:* state of the art on 16 MiniHack tasks, 9 of them navigation; matches
existing methods on singleton VizDoom.
*Verdict:* the inverse-dynamics feature space is the part worth stealing - it is
exactly the fix for the logged failure where a velocity-plus-view novelty key
farmed in place, because inverse dynamics would have discarded the uncontrollable
components. But E3B is built for CONTEXTUAL MDPs where the layout changes every
episode, and Henaff's own 2023 study concludes episodic bonuses dominate in CMDPs
while GLOBAL bonuses dominate in singletons. This map is a singleton. Lower
prior for that reason.

**NovelD: A Simple yet Effective Exploration Criterion.** Tianjun Zhang, Huazhe
Xu, Xiaolong Wang, Yi Wu, Kurt Keutzer, Joseph Gonzalez, Yuandong Tian.
NeurIPS 2021.
*Mechanism:* reward the POSITIVE DIFFERENCE of RND novelty between consecutive
states, `max(N(s_t) - alpha * N(s_{t-1}), 0)`, gated by an episodic first-visit
indicator. Constant: alpha = 0.5.
*Result:* solves all static MiniGrid tasks, including the hardest procedurally
generated mazes.
*Verdict:* the bonus is a novelty POTENTIAL DIFFERENCE, so it cannot be farmed by
oscillating in place - a step back cancels the step forward. That is precisely
the pathology logged in this repo, and it is a cheap modification of the existing
bonus. Its weakness is RND's: after 100M steps at the wall there is no global
novelty gradient left to differentiate.

**DEIR: Efficient and Robust Exploration through Discriminative-Model-Based
Episodic Intrinsic Rewards.** Shanchuan Wan, Yujin Tang, Yingtao Tian, Tomoyuki
Kaneko. IJCAI 2023. arXiv:2304.10770.
*Mechanism:* a conditional-mutual-information intrinsic reward implemented with a
discriminative forward model, so the bonus scales with the novelty the AGENT
CAUSED rather than the novelty the environment produced. Built on PPO
(stable-baselines3).
*Result:* beats baselines on MiniGrid, including reduced view size, noisy
observations and invisible obstacles; generalizes on ProcGen.
*Verdict:* PPO-native, which most of this family is not. Same singleton caveat
as E3B.

**Improving Intrinsic Exploration by Creating Stationary Objectives (SOFE).**
Roger Creus Castanyer, Joshua Romoff, Glen Berseth. ICLR 2024. arXiv:2310.18144.
*Mechanism:* a diagnosis, not a new bonus. Count bonuses, pseudo-counts and
state-entropy bonuses are NON-STATIONARY rewards, which a value function cannot
fit - the critic is asked to predict a reward whose distribution shifts under it.
SOFE augments the OBSERVATION with the sufficient statistics of the bonus (an
encoding of the visitation state), making the intrinsic objective stationary in
the augmented MDP.
*Result:* improves count bonuses, pseudo-counts and state entropy; demonstrated on
sparse-reward, pixel-based, 3D navigation and procedurally generated tasks.
*Verdict:* **if any additive bonus is kept here, this is the fix that makes it
learnable by a PPO critic.** It is also a plausible partial explanation for the
null count-bonus result across a 10x coefficient sweep: a coefficient sweep
cannot repair a non-stationary target.

### 2.3 Global / large-scale exploration agents

**Never Give Up: Learning Directed Exploration Strategies (NGU).** Adria
Puigdomenech Badia, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Bilal Piot,
Steven Kapturowski, Olivier Tieleman, Martin Arjovsky, Alexander Pritzel, Andrew
Bolt, Charles Blundell. ICLR 2020. arXiv:2002.06038.
**Agent57: Outperforming the Atari Human Benchmark.** Badia, Piot, Kapturowski,
Sprechmann, Vitvitskyi, Guo, Blundell. ICML 2020. arXiv:2003.13350.
*Mechanism:* intrinsic reward = episodic novelty (k-NN over an episodic memory of
inverse-dynamics embeddings) MULTIPLIED by a lifelong novelty modulator (RND),
with a family of policies conditioned on different exploration weights beta_i;
Agent57 adds a bandit that selects among them.
*Needs:* R2D2-scale distributed off-policy learner, recurrent networks, a
population of exploration weights.
*Result:* first agent above the human benchmark on all 57 Atari games.
*Verdict:* the "episodic x lifelong" product is the right structure, and the
beta-conditioned family is the same idea as the repo's T-conditioned families.
But adopting NGU means replacing the trainer.

**BYOL-Explore: Exploration by Bootstrapped Prediction.** Zhaohan Daniel Guo,
Shantanu Thakoor, Miruna Pislar, Bernardo Avila Pires, Florent Altche, Corentin
Tallec, Alaa Saade, Daniele Calandriello, Jean-Bastien Grill, Yunhao Tang, Michal
Valko, Remi Munos, Mohammad Gheshlaghi Azar, Bilal Piot. NeurIPS 2022.
arXiv:2206.08332.
*Task:* DM-HARD-8 - 3D, first-person, continuous action, partially observed, hard
exploration. The closest benchmark in scale to this repo.
*Mechanism:* one loss. A latent world model is trained with a BYOL-style
bootstrapped multi-step prediction objective, and the same prediction error is
the intrinsic reward.
*Result:* solves the majority of DM-HARD-8 from intrinsic reward alone, where
"prior work could only get off the ground with human demonstrations" - a directly
relevant precedent for a no-demo constraint. Superhuman on the ten hardest Atari
exploration games.
*Verdict:* strong, generic, no map constants, but at heart a prediction-error
(curiosity) bonus, and this repo's problem is deception rather than
unreachability. High cost (a world model), uncertain fit.

### 2.4 Archive / return-then-explore

**Go-Explore: a New Approach for Hard-Exploration Problems.** Adrien Ecoffet,
Joost Huizinga, Joel Lehman, Kenneth Stanley, Jeff Clune. arXiv:1901.10995, 2019.
**First return, then explore.** Same authors. Nature 590:580-586, 2021.
arXiv:2004.12919.
*Mechanism:* (1) an archive of cells (downscaled frames, or domain-knowledge
features); (2) select a promising cell and RETURN to it WITHOUT exploring - by
restoring the simulator state, or in the policy-based variant by running a
goal-conditioned policy; (3) explore from there; (4) robustify by self-imitation
on the discovered trajectory.
*Needs:* archive; a resettable simulator OR a goal-conditioned return policy;
self-imitation.
*Result:* Montezuma's Revenge and Pitfall records. The Nature version shows the
goal-conditioned policy both improves exploration efficiency AND handles
stochasticity throughout training, and demonstrates sparse-reward robotic
pick-and-place.
*Verdict:* **this repo has already run phase 1 and it WORKED** - the archive found
the detour in 28 minutes. The stated blocker, "random macro-actions fail on maps
that need skill", is precisely what the Nature paper's policy-based RETURN and
its robustification phase exist to fix. Stopping at phase 1 is stopping before the
half of the algorithm that addresses the objection. Honest caveat: the headline
Montezuma numbers use domain-knowledge cells; the no-domain-knowledge variant is
weaker, and a hand-chosen cell definition is exactly the kind of map constant this
repo forbids - hence the next entry.

**Cell-Free Latent Go-Explore (LGE).** Quentin Gallouedec, Emmanuel Dellandrea.
ICML 2023. arXiv:2208.14928.
*Mechanism:* removes the cell partition entirely. Learn a latent representation
(inverse dynamics, forward dynamics or VQ-VAE), estimate density in the latent
space, and select goals in low-density regions.
*Result:* simpler than Go-Explore and more robust across several hard-exploration
environments, including Montezuma.
*Verdict:* the version that satisfies "no map constants". Encoder and density
estimator are generic and set once.

**Intelligent Go-Explore (IGE).** Cong Lu, Shengran Hu, Jeff Clune.
arXiv:2405.15143, 2024.
*Mechanism:* replace the hand-designed interestingness heuristic with a
foundation model that judges which archived states to return to and which
serendipitous discoveries to keep.
*Result:* TextWorld games; beats ReAct and Reflexion.
*Verdict:* listed for completeness. Text-domain results only; a depth image at
25 Hz is not a setting where an LLM judge is affordable or grounded.

---

## 3. Learning the geodesic structure from the agent's own experience

This section answers the user's framing directly: instead of being HANDED a BFS
field, learn a distance from the policy's own transitions and use it as the
potential. Under Devlin and Kudenko (2012) an on-policy learner may use a
potential that changes during training without losing invariance, so this is
legal and, unusually, it is legal specifically because the trainer is on-policy.

**The one caveat that applies to this whole section, stated once:** a distance
learned from the agent's own data is only correct on the SUPPORT of that data.
While the agent has never gone around the wall, the learned distance from the
wall to the goal is either undefined or optimistically short - i.e. the same
deception. None of these methods can produce the FIRST crossing. What they can do
is (a) stop the potential actively punishing the detour once it has been sampled
even once, and (b) make the potential correct on the next rung and the next map.
Pair them with something from section 5 or 6.

**The Laplacian in RL: Learning Representations with Efficient Approximations.**
Yifan Wu, George Tucker, Ofir Nachum. ICLR 2019. arXiv:1810.04586.
*Task/reward:* goal-achieving tasks in grid mazes and continuous control; the
reward is built as the L2 distance in the learned embedding.
*Mechanism:* the smallest eigenvectors of the graph Laplacian of the transition
process encode the geometry of the state space. The paper gives a scalable
model-free objective (a graph-drawing objective with a repulsive term) that
approximates them with a neural network from sampled transitions.
*Needs:* transitions only; no map, no demos, no model. Works alongside any RL
algorithm.
*Result:* L2 distance in the learned embedding is a better shaping reward than
raw Euclidean distance on maze goal-reaching.
*Verdict:* the canonical "learn the geodesic from experience" result and exactly
the mechanism the user is asking about. Constants are dimensionless (embedding
dimension d, typically 20-50).

**Reachability-Aware Laplacian Representation in RL (RA-LapRep).** Kaixin Wang,
Kuangqi Zhou, Jiashi Feng, Bryan Hooi, Xinchao Wang. ICML 2023.
arXiv:2210.13153.
*Statement, and it is a warning:* the widely repeated claim that L2 distance in
the Laplacian embedding reflects reachability is FALSE in general - two states
with a small LapRep distance can be far apart in the environment. RA-LapRep
rescales each dimension to fix it, and then its distance decreases monotonically
as the agent approaches the goal, where plain LapRep's does not.
*Verdict:* if the Laplacian route is taken, take this version. A non-monotone
learned potential would reproduce the exact failure being escaped.

**Proper Laplacian Representation Learning.** Diego Gomez, Michael Bowling,
Marlos C. Machado. ICLR 2024. arXiv:2310.10833.
*Mechanism:* an augmented-Lagrangian objective whose optimum is the true
eigenvectors (with correct ordering and rotation invariance removed), fixing the
hyperparameter sensitivity of Wu et al. 2019.
*Verdict:* the current recommended implementation of the Laplacian route.

**Optimal Goal-Reaching RL via Quasimetric Learning (QRL).** Tongzhou Wang,
Antonio Torralba, Phillip Isola, Amy Zhang. ICML 2023. arXiv:2304.01203.
*Mechanism:* the optimal goal-conditioned value function IS a quasimetric (an
asymmetric distance obeying the triangle inequality). QRL parameterizes the value
with a quasimetric network (IQE / MRN) and trains it with an objective that
maximizes distances subject to a local one-step constraint - which recovers
shortest-path structure rather than averaging over the behaviour policy.
*Needs:* off-policy or offline data; a quasimetric architecture.
*Result:* offline maze2d and online goal-conditional benchmarks, state and image
based; improved sample efficiency over standard GCRL.
*Verdict:* the asymmetry matters in this domain - a surf drop is one-way, and a
symmetric metric cannot represent that. Requires an off-policy critic alongside
PPO, which the constraints permit.

**Contrastive Learning as Goal-Conditioned RL.** Benjamin Eysenbach, Tianjun
Zhang, Sergey Levine, Ruslan Salakhutdinov. NeurIPS 2022. arXiv:2206.07568.
**Learning Temporal Distances: Contrastive Successor Features Can Provide a
Metric Structure for Decision-Making.** Vivek Myers, Chongyi Zheng, Anca Dragan,
Sergey Levine, Benjamin Eysenbach. ICML 2024. arXiv:2406.17098.
*Mechanism:* the contrastive critic that discriminates "s_future came from s" IS
a goal-conditioned value function; parameterizing its energy as a potential minus
a quasimetric yields a "successor distance" that obeys the triangle inequality
even in stochastic environments.
*Needs:* transitions plus a geometric future-state sampler (gamma = 0.99 style).
No map, no demos, no reward.
*Verdict:* the cheapest learned distance to bolt onto an existing pipeline - it
trains off the same rollouts PPO already produces, and it is the same network ETD
uses for its episodic bonus. Strong candidate as the replacement potential.

**HIQL: Offline Goal-Conditioned RL with Latent States as Actions.** Seohong
Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine. NeurIPS 2023.
arXiv:2307.11949.
*Mechanism:* one goal-conditioned value function yields three things - a
representation, a HIGH-level policy that emits a subgoal, and a LOW-level policy
that reaches it. Splitting the policy is what rescues a value function that is
too noisy over long horizons to give an action-level signal.
*Needs:* offline dataset, IQL-style value learning.
*Result:* large gains on long-horizon AntMaze and Kitchen; 62% average on unseen
Roboverse manipulation from images.
*Verdict:* the most relevant statement it makes for this repo is the diagnostic
one - a long-horizon goal-conditioned value is accurate in DIRECTION over short
spans but not over the whole map, so use it to pick subgoals, not actions.

**Search on the Replay Buffer (SoRB).** Benjamin Eysenbach, Ruslan Salakhutdinov,
Sergey Levine. NeurIPS 2019. arXiv:1906.05253.
*Task/reward:* sparse-reward visual navigation over 100+ steps in ViZDoom-style
3D mazes and 2D mazes with walls.
*Mechanism:* build a graph whose NODES are observations already in the replay
buffer and whose EDGE WEIGHTS are the goal-conditioned value function's predicted
distance; run Dijkstra on it to produce a sequence of subgoals; a
distributional value function with a pessimistic aggregation clips unreliable
long edges.
*Needs:* off-policy goal-conditioned value function, a replay buffer used as the
node set, graph search. No map, no demos.
*Result:* solves sparse-reward tasks over one hundred steps and generalizes
substantially better than flat RL.
*Verdict:* **this is the learned, experience-only analogue of the BFS field that
already works here**, and it is the most direct answer to "learn the geodesic
instead of being given it". It is also exactly the structure the repo's own
reservoir already half-implements: the reservoir is the node set, and the missing
piece is the pairwise distance and the search. Caveat: SoRB's value function is
trained on data that ALREADY covers the maze; on an unflown map the graph has no
node past the wall, so SoRB consolidates and generalizes a frontier, it does not
create one.

**Semi-parametric Topological Memory for Navigation (SPTM).** Nikolay Savinov,
Alexey Dosovitskiy, Vladlen Koltun. ICLR 2018. arXiv:1803.00653.
*Mechanism:* a non-parametric graph of observations plus a parametric retrieval
network (the same R-network idea as EC) and a locomotion network; localize, plan
over the graph, walk to the next node.
*Result:* in previously unseen 3D mazes, 3x the success rate of the best
baseline - but from FIVE MINUTES OF PROVIDED EXPLORATION FOOTAGE of that maze.
*Verdict:* important negative note for this repo: SPTM's headline result depends
on a provided traversal of the maze. Under rule 0 that footage would have to come
from the policy's own rollouts, which is possible but changes the result.

**Sparse Graphical Memory for Robust Planning (SGM).** Scott Emmons, Ajay Jain,
Misha Laskin, Thanard Kurutach, Pieter Abbeel, Deepak Pathak. NeurIPS 2020.
arXiv:2003.06417.
*Mechanism:* the same graph idea, but sparsified by a TWO-WAY CONSISTENCY test -
two states are merged only when they are interchangeable both as goals and as
starting states. Plus k-nearest filtration and walk-throughs to delete
untraversable edges.
*Result:* substantially outperforms prior methods on long-horizon, sparse-reward
VISUAL navigation; shortest-path inflation from merging scales only linearly in
the merge threshold.
*Verdict:* the practical fix for the failure mode the repo already saw in
Go-Explore leaf selection ("the deepest-node rule picks a dead-end pocket") -
sparsification by consistency, not by depth.

**Mapping State Space using Landmarks for Universal Goal Reaching.** Zhiao Huang,
Fangchen Liu, Hao Su. NeurIPS 2019. arXiv:1908.05451.
*Mechanism:* landmarks chosen from past experience by FARTHEST POINT SAMPLING
(explicitly reported as better exploration than uniform sampling), a
landmark-level map for long-range planning, and the UVFA for local control.
*Verdict:* farthest-point sampling over the reservoir is a one-line, map-free
replacement for the repo's two broken leaf-selection rules, and it is the rule
that does not collapse onto a deterministic policy's repeated falls.

**Value Iteration Networks.** Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine,
Pieter Abbeel. NeurIPS 2016 (best paper). arXiv:1602.02867.
*Mechanism:* embed a differentiable planner - value iteration unrolled as
alternating convolutions and channel-wise max-pooling - inside the policy
network, so the network LEARNS to plan on a learned reward/transition map.
*Result:* generalizes to new grid-world layouts far better than a CNN policy.
*Verdict:* the "different state abstraction / planning module" option. Its
published form assumes a 2D top-down grid, which this task does not have from a
first-person depth image; the hierarchical variants that plan on a coarse grid
under a continuous controller are closer, but this is a large redesign with no
detour-specific evidence.

**Learning the Minimum Action Distance.** Lorenzo Steccanella, Joshua B.
Evans, Ozgur Simsek, Anders Jonsson. arXiv:2506.09276, 2025. (See also
Steccanella and Jonsson, "Asymmetric Norms to Approximate the Minimum Action
Distance", arXiv:2312.10276.)
*Mechanism:* learn the minimum number of actions between two states from state
TRAJECTORIES ONLY (no actions, no rewards), with a quasimetric embedding, and
introduce benchmark domains where distance learning is hard.
*Verdict:* the lightest-weight learned distance in this survey and the one with
the fewest requirements. Very new; treat as a candidate implementation rather
than a validated result.

---

## 4. Hierarchical / subgoal planning and frontier methods

**Data-Efficient Hierarchical RL (HIRO).** Ofir Nachum, Shixiang Gu, Honglak Lee,
Sergey Levine. NeurIPS 2018. arXiv:1805.08296.
*Task:* Ant Maze (U-shaped, requires a detour around a wall), Ant Push, Ant Fall.
*Mechanism:* a high-level policy emits a GOAL IN RAW OBSERVATION SPACE every c
steps; the low-level policy is rewarded by the distance it closes to that goal.
An off-policy correction relabels stored high-level goals so old low-level data
stays valid.
*Needs:* off-policy (TD3), a goal space carved out of the observation.
*Result:* 0.99 / 0.92 / 0.66 success on Ant Maze / Push / Fall, where flat and
prior HRL baselines make no progress, within a few million samples.
*Verdict:* the strongest classic HRL result on a detour maze. The catch that
matters here: the goal space is the low-dimensional (x, y) of the ant. From a
first-person depth image there is no such coordinate without either giving the
agent the map or learning a representation (which is section 3).

**Learning Multi-Level Hierarchies with Hindsight (HAC).** Andrew Levy, George
Konidaris, Robert Platt, Kate Saenko. ICLR 2019. arXiv:1712.00948.
*Mechanism:* train all levels of the hierarchy in parallel using hindsight action
and hindsight goal transitions, so each level can learn as if the level below it
were already optimal.
*Result:* solves multi-level maze and robotics tasks; large speedups over flat
agents.
*Verdict:* same goal-space requirement as HIRO.

**Landmark-Guided Subgoal Generation in HRL (HIGL).** Junsu Kim, Younggyo Seo,
Jinwoo Shin. NeurIPS 2021. arXiv:2110.13625.
*Mechanism:* sample landmarks by two criteria - COVERAGE (dispersion of visited
states) and NOVELTY (prediction error) - build a graph over visited states plus
landmarks, run shortest-path planning to pick the most urgent landmark, and
shrink the high-level action space to a neighbourhood of it.
*Result:* 65.1% on sparse U-shaped Ant Maze at 10M steps versus 17.6% for HRAC.
*Verdict:* the cleanest published statement that frontier selection should combine
coverage AND novelty, which is a direct fix for the repo's broken leaf-selection
rules. Off-policy.

**Deep Hierarchical Planning from Pixels (Director).** Danijar Hafner,
Kuang-Huei Lee, Ian Fischer, Pieter Abbeel. NeurIPS 2022. arXiv:2206.04114.
*Task:* Egocentric Ant Maze - a 3D maze traversed by a quadruped from an
EGOCENTRIC CAMERA and proprioception, with no global position and no top-down
view. This is the closest published observation setting to this repo.
*Mechanism:* learn a world model; a manager selects GOALS IN THE MODEL'S LATENT
SPACE (discretized by a goal autoencoder) every K steps, maximizing task plus
exploration reward; a worker learns to reach them.
*Result:* "in these larger mazes, Director is the only method to find and reliably
reach the goal", outperforming exploration baselines on very sparse rewards.
*Verdict:* **the strongest published evidence that a first-person agent can solve
a detour maze without a map, demos or a given coordinate**, and it does it by
subgoal selection in a learned latent space rather than by an exploration bonus.
Cost: a full world model and a two-level policy; it is an algorithm redesign, not
an add-on.

**Discovering and Achieving Goals via World Models (LEXA).** Russell Mendonca,
Oleh Rybkin, Kostas Daniilidis, Danijar Hafner, Deepak Pathak. NeurIPS 2021.
arXiv:2110.09514. (Built on **Plan2Explore**, Ramanan Sekar, Oleh Rybkin, Kostas
Daniilidis, Pieter Abbeel, Danijar Hafner, Deepak Pathak, ICML 2020,
arXiv:2005.05960.)
*Mechanism:* train one world model; an EXPLORER policy plans, in imagination, to
states of high model disagreement (i.e. it seeks states it has NOT seen, rather
than returning to seen ones), and the states it finds become the ACHIEVER's goal
distribution.
*Result:* beats prior unsupervised goal-reaching on 40 test tasks over four
robotic and locomotion domains; zero-shot goal images after the unsupervised
phase.
*Verdict:* the "explore by foresight rather than by revisiting" idea is the part
that distinguishes it from Go-Explore and is the right answer to "the archive
keeps returning to a dead-end pocket". Manipulation and locomotion results only.

**Planning Goals for Exploration (PEG).** Edward S. Hu, Richard Chang, Oleh
Rybkin, Dinesh Jayaraman. ICLR 2023 (spotlight). arXiv:2303.13002.
*Mechanism:* a Go-Explore structure with a LEARNED choice of where to go. Each
episode: choose a goal command by planning (CEM over a world model) so as to
maximize the exploration value of the state the goal-conditioned policy will
actually end up in, then hand over to an undirected explore policy. Crucially the
commanded goal may be a state never seen, or even a physically impossible one.
*Needs:* world model, goal-conditioned policy, sampling-based planner.
*Result:* an ant navigating a long maze and a robot arm stacking three blocks;
more efficient than Go-Explore-style baselines.
*Verdict:* the most modern synthesis of "archive + goal-conditioned return +
exploration" and the best-argued rule for WHICH state to return to. Requires a
world model.

**Planning with Goal-Conditioned Policies (LEAP).** Soroush Nasiriany, Vitchyr
Pong, Steven Lin, Sergey Levine. NeurIPS 2019. arXiv:1911.08453.
*Mechanism:* optimize a sequence of SUBGOALS directly in a learned latent space
(VAE) using the goal-conditioned value function as the feasibility measure; then
have the policy chase them one at a time and replan.
*Verdict:* the continuous-optimization alternative to graph search (SoRB). Same
support caveat as section 3.

**Sub-Goal Trees - a Framework for Goal-Based RL.** Tom Jurgenson, Or Avner,
Edward Groshev, Aviv Tamar. ICML 2020. arXiv:1906.05329.
*Mechanism:* replace the Bellman recursion with the all-pairs-shortest-path
recursion: predict a MIDPOINT between start and goal, then recurse on both halves.
This gives a policy-gradient method whose credit assignment is logarithmic in
horizon rather than linear.
*Verdict:* a genuine "different search space" candidate - the horizon over which
credit must travel is the thing that is broken here, and a divide-and-conquer
recursion attacks it directly. Demonstrated on 7-DoF motion planning among
obstacles, not on exploration from scratch.

**Learning to Explore using Active Neural SLAM.** Devendra Singh Chaplot, Dhiraj
Gandhi, Saurabh Gupta, Abhinav Gupta, Ruslan Salakhutdinov. ICLR 2020.
arXiv:2004.05155. (Frontier-based exploration originally: Brian Yamauchi, "A
frontier-based approach for autonomous exploration", IEEE CIRA 1997.)
*Mechanism:* the robotics answer. Build an explicit occupancy map online (a
learned SLAM module), have a GLOBAL policy choose a long-term goal on that map,
convert it to a short-term goal with an ANALYTICAL path planner, and have a LOCAL
policy walk there. Frontier-based exploration is the classical global rule: go to
the boundary between known-free and unknown.
*Result:* explicit maps plus analytical planning give significantly better
exploration and sample complexity than end-to-end RL in 3D scenes.
*Verdict:* an honest and uncomfortable comparison point. The "learned map plus
planner" architecture is what actually works in 3D navigation, and this repo's
geodesic BFS field is the same idea with the map given instead of built. Building
the map online from the agent's own depth images is NOT giving the answer and
would satisfy the constraints - but it is a big build, and it assumes a
navigable-surface abstraction that a surf map (airborne ramp riding) violates.

**METRA: Scalable Unsupervised RL with Metric-Aware Abstraction.** Seohong Park,
Oleh Rybkin, Sergey Levine. ICLR 2024. arXiv:2310.08887.
*Mechanism:* unsupervised skill discovery where skills are made maximally
distinguishable under a TEMPORAL-distance (Wasserstein) metric rather than a
metric-agnostic mutual information, so the learned skill set approximately covers
the state space; the abstraction also gives zero-shot goal reaching.
*Verdict:* the modern skill-discovery entry. Relevant as a pre-training phase
that would give a detour-capable macro-action set, but no detour-maze result and
a large redesign.

---

## 5. Methods specifically about temporarily going AWAY from the goal

This is the section that maps one-to-one onto the measured failure.

**Keeping Your Distance: Solving Sparse Reward Tasks Using Self-Balancing Shaped
Rewards ("Sibling Rivalry").** Alexander Trott, Stephan Zheng, Caiming Xiong,
Richard Socher. NeurIPS 2019. arXiv:1911.01417.
*Task/reward:* EXACTLY this problem. "Simple distance-to-goal reward shaping
often fails, as it renders learning vulnerable to local optima." Point Maze
(10x10, walls), U-shaped Ant Maze, a 2D "bit-flipping" task, and 3D construction
in Minecraft.
*Mechanism:* run TWO rollouts from the same (s0, g) - "siblings". Each takes the
OTHER's terminal state as an ANTI-GOAL and its reward becomes

    r'(s, g, gbar) = 1                              if d(s,g) <= delta
                   = min[0, -d(s,g) + d(s,gbar)]    otherwise

The `min[0, .]` keeps the shaped reward non-positive, so the agent can never farm
proximity - only reaching the goal pays. The anti-goal term makes the local
optimum REPEL: if both siblings end at the wall, each is punished for being near
where the other ended, and the gradient at the wall flips sign. As the policy
starts reaching the goal, the two terms cancel and the reward converges to the
sparse indicator, so no NEW stable optimum is introduced. The farther sibling is
always kept; the closer sibling enters the update only if it reached the goal
(`d(s_T^c, g) < delta`) or the two terminal states are close
(`d(s_T^f, s_T^c) <= epsilon`), which is what prevents the pair from collapsing.
*Constants:* Point Maze delta = 0.15, epsilon = 5.0; Ant Maze delta = 1.0,
epsilon = 10.0; bit flipping delta = 0, epsilon = infinity. Learning rate 0.001,
entropy coefficient 0.025, 4 rollouts per update, 4 epochs.
*Needs:* **PPO (on-policy), full-episode rollouts, control over (s0, g).
Explicitly NO map, NO demonstrations, NO replay buffer, NO extra network.**
*Result:* "PPO with the naive distance-to-goal reward never succeeds." PPO+SR
discovers the goal in 5/5 point-maze runs, versus 2/5 for PPO+ICM and 1/5 for
DDPG+HER. It also solves U-maze Ant and a Minecraft construction task where naive
distance shaping fails.
*Verdict:* **the closest published match to this repo's problem, its trainer and
its constraints, by a wide margin.** The one constraint friction: epsilon is
given in WORLD UNITS and differs per task (5.0 vs 10.0), which is a map constant
under this repo's rules. The generic fix is to express it dimensionlessly, e.g.
epsilon = c * d(s0, g) with one c carried unchanged across all maps; delta is
already fixed by the finish box. Honest limitation: Pitis et al. (2020) report
PPO+SR needing about 7.5M steps to 90% on PointMaze where O/MEGA needs about 100x
fewer - SR is not sample-efficient. At 600k steps/s that is roughly 12 seconds of
wall clock here, so the objection is irrelevant at this repo's throughput. It has
not been demonstrated on image observations or at 25 Hz control, which is the
real risk.

**Episodic Curiosity through Reachability.** Savinov et al. 2019 (full entry in
2.2). Listed again here because its stated motivation IS the detour: "the agent
must take a detour" to reach a state that is Euclidean-near but reachability-far.

**Maximum Entropy Gain Exploration (MEGA / OMEGA).** Silviu Pitis, Harris Chan,
Stephen Zhao, Bradly Stadie, Jimmy Ba. ICML 2020. arXiv:2007.02832. (ALA 2020
best paper.)
*Task:* PointMaze and AntMaze with walls and dead-ends; block stacking.
*Mechanism:* when the desired goal is too far to give a learning signal, set your
OWN goals to maximize the entropy of the historical ACHIEVED-goal distribution -
in practice, `g = argmin_g p_hat(g)` over candidates drawn from the buffer, i.e.
always aim at the least-visited achieved goal, which is the FRONTIER of what you
can already do. OMEGA anneals from this unsupervised objective to the real goal
once the KL to the desired distribution becomes tractable.
*Constants:* KDE bandwidth 0.1; N = 100 candidate goals per episode; density
refit every step on 10,000 normalized samples; cutoff init -3.0, decreased by 1
when intrinsic success exceeds 70% over 10 episodes; alpha = 1/max(b + KL, 1)
with b = -3; DDPG with HER, batch 2000, action noise 0.1, random-action epsilon
0.1, gamma 0.98-0.99, Polyak 0.05.
*Result:* about 100x faster than the prior state of the art (PPO+SR) on PointMaze
and about 10x faster than hierarchical PPO+SR on AntMaze.
*Verdict:* the strongest quantitative maze result in this survey, and its
"pursue the least-dense achieved goal" rule is exactly the frontier rule the
repo's reservoir lacks. But it is off-policy DDPG+HER and needs a
LOW-DIMENSIONAL, semantically meaningful goal space (6 dimensions or fewer in the
paper). From a depth image, that goal space has to be learned - which is section
3 again. The clean reading: MEGA's SELECTION RULE is portable to the repo's
reservoir today; MEGA's algorithm is not.

**Reverse Curriculum Generation for Reinforcement Learning.** Carlos Florensa,
David Held, Markus Wulfmeier, Michael Zhang, Pieter Abbeel. CoRL 2017.
arXiv:1707.05300.
*Task:* point-mass maze and Ant maze (G-shaped and U-shaped), plus ring-on-peg
and key insertion. Reward is the sparse goal indicator - no distance shaping at
all.
*Mechanism:* start training from states NEAR the goal and expand the start
distribution outward. New start states are generated by "Brownian motion" -
short random-action rollouts from states already in the set - and then FILTERED
to the states the current policy solves sometimes but not always,
`R_min < R(pi_i, s0) < R_max`.
*Constants:* Brownian horizon T_B = 50; action noise Sigma = I; pool M = 10,000
states; N_new = 200 new starts and N_old = 100 replayed per iteration;
R_min = 0.1, R_max = 0.9; episode horizon 500 (2000 for Ant); gamma = 0.998;
TRPO with a (64,64) MLP, 50,000-step batches.
*Needs:* (i) the ability to RESET INTO AN ARBITRARY STATE - which this repo
already has; (ii) ONE goal state - which the map's own finish box provides, and
which is not a demonstration; (iii) dynamics whose random-action Markov chain
forms a communicating class. Explicitly: **no demonstrations, no reward shaping,
no dynamics model.**
*Result:* solves mazes where uniform start sampling shows "considerable slow-down"
or fails outright, with far lower variance.
*Verdict:* the legal, demo-free version of the "window / spine" curriculum this
repo already knows works. Its R_min/R_max filter is also the documented fix for
the trivial-win trap logged here (a win rate that rises while reservoir min-depth
falls): a start the policy already solves is REMOVED from the curriculum by
construction. The known failure mode, stated honestly: on a surf map the backward
Brownian walk from the finish will not generate the high-speed airborne states on
the real approach, so the curriculum stalls where backward reachability stops.
On a walking labyrinth it should not.
*Related:* **BaRC: Backward Reachability Curriculum** (Boris Ivanovic, James
Harrison, Apoorva Sharma, Mo Chen, Marco Pavone, arXiv:1806.06161, 2018)
replaces the Brownian walk with an approximate backward-reachable-set computation
given a dynamics model; **Reverse Forward Curriculum Learning** (ICLR 2024)
revisits the same idea with demonstrations, which rule 0 forbids here.

**Automatic Goal Generation for Reinforcement Learning Agents (Goal GAN).**
Carlos Florensa, David Held, Xinyang Geng, Pieter Abbeel. ICML 2018.
arXiv:1705.06366.
*Mechanism:* a GAN generates goals of intermediate difficulty ("Goals of
Intermediate Difficulty": success probability between R_min and R_max), trained
adversarially against a label from the agent's own success rate.
*Verdict:* the forward-direction twin of the reverse curriculum. Needs a goal
space it can generate in, which from images is again section 3.

**Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play.**
Sainbayar Sukhbaatar, Zeming Lin, Ilya Kostrikov, Gabriel Synnaeve, Arthur Szlam,
Rob Fergus. ICLR 2018. arXiv:1703.05407.
*Mechanism:* two copies of the agent. Alice acts for a while and stops; Bob must
either UNDO what Alice did (reversible environments) or REPEAT it (resettable
environments). Alice is rewarded for tasks that are easy for her but hard for
Bob, which automatically produces a curriculum of increasing reach with no
extrinsic reward at all.
*Needs:* reset or reversibility; two policies.
*Verdict:* fully generic, demo-free, no map constants, and the "resettable"
variant matches this simulator. It is the self-supervised way to get an agent
that can reach anywhere before the task reward is ever applied. Demonstrated on
small discrete and simple continuous tasks; no 3D first-person result.

**Learning with AMIGo: Adversarially Motivated Intrinsic Goals.** Andres Campero,
Roberta Raileanu, Heinrich Kuttler, Joshua Tenenbaum, Tim Rocktaschel, Edward
Grefenstette. ICLR 2021. arXiv:2006.12122.
*Mechanism:* a goal-generating TEACHER network proposes intrinsic goals to a
goal-conditioned STUDENT; the teacher is rewarded for goals the student reaches
but only after at least t* steps, with t* increasing as the student improves.
*Result:* solves the hardest MiniGrid exploration tasks (KeyCorridor,
ObstructedMaze) where count and curiosity baselines fail.
*Verdict:* the teacher's reward - "propose goals that take the student at least
t* steps" - is a TEMPORAL difficulty measure, which is again the right currency,
and the paper shows it produces exactly the escalating reach a detour needs. Grid
world with a symbolic goal space; porting the teacher's output space to depth
images is the open work.

**Deep Reinforcement Learning with New-Field Exploration for Navigation in Detour
Environment.** (NFE.) IEEE ICARM 2021.
*Verdict:* one of the very few papers with "detour" in the title. An internal
reward for entering unexplored fields, for a mobile robot with laser scans in
detour environments. Small-scale robotics venue, no public code found, results
not comparable to the benchmarks above. Listed for completeness, not
recommended.

**Temporally-Extended epsilon-Greedy Exploration (ez-greedy).** Will Dabney,
Georg Ostrovski, Andre Barreto. ICLR 2021. arXiv:2006.01782.
*Statement:* the stated limitation of epsilon-greedy is "its lack of temporal
persistence, which limits its ability to ESCAPE LOCAL OPTIMA". ez-greedy repeats
a single sampled action for a random duration drawn from a heavy-tailed zeta
distribution.
*Constants:* duration ~ zeta(mu) with **mu = 2** (best in their sweeps and the
value that matches animal foraging); epsilon as usual.
*Verdict:* the cheapest experiment in this entire survey - one distribution and
one constant, and mu = 2 is carried unchanged everywhere, which satisfies the
"same constants on every map" rule trivially. Note carefully what is DIFFERENT
from the OU / colored noise already tried here: colored noise is a zero-mean
perturbation of a continuous action, which does not produce a committed 200-step
deviation; a zeta-distributed action REPEAT holds one discrete decision (e.g.
"strafe left") for an unbounded duration, which is exactly the shape of a detour.
*Related:* **Deep Laplacian-based Options for Temporally-Extended Exploration**,
Martin Klissarov, Marlos C. Machado, ICML 2023, arXiv:2301.11181 - the same
persistence idea but with the repeated behaviour being a learned Laplacian
eigenoption instead of a repeated primitive, with no separate option-discovery
phase.

**Smooth Exploration for Robotic RL (gSDE).** Antonin Raffin, Jens Kober, Freek
Stulp. CoRL 2021. arXiv:2005.05719. (And **Pink Noise Is All You Need: Colored
Noise Exploration in Deep RL**, Onno Eberhard, Jakob Hollenstein, Cristina
Pinneri, Georg Martius, ICLR 2023; **Colored Noise in PPO**, arXiv:2312.11091.)
*Verdict:* included because the repo has already tried OU / colored noise. These
papers make state-dependent rather than step-dependent noise, resampled on an
interval; they improve smoothness and modestly improve exploration, but none of
them claims to escape a deceptive dense reward. Consistent with the null result
here; not worth a second arm.

---

## 6. Learning distance / reaching goals with binary rewards (the user's own proposal)

The user proposes replacing the dense potential with a binary 0/1 reward plus
intrinsic novelty. The literature's verdict on that specific combination:

**Hindsight Experience Replay (HER).** Marcin Andrychowicz, Filip Wolski, Alex
Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter
Abbeel, Wojciech Zaremba. NeurIPS 2017. arXiv:1707.01495.
*Mechanism:* relabel a failed trajectory with a goal it DID achieve, so a binary
reward becomes informative. "Avoid the need for complicated reward engineering."
*Needs:* **an off-policy algorithm and a replay buffer** - relabelling makes the
data off-policy by construction. This is the blocking constraint for a PPO repo.
*Verdict:* the right idea, the wrong trainer. See next.

**Hindsight Policy Gradients (HPG).** Paulo Rauber, Avinash Ummadisingu, Filipe
Mutz, Jurgen Schmidhuber. ICLR 2019. arXiv:1711.06006.
*Mechanism:* brings hindsight to POLICY GRADIENT methods using importance
sampling - evaluate a trajectory collected for goal g under an alternative goal
g'.
*Result:* "remarkable increase in sample efficiency" across sparse-reward
environments.
*Verdict:* the on-policy route to hindsight, and therefore the one compatible
with this trainer. Skeptical note: the importance weights are the product over a
whole trajectory and are notoriously high-variance; the paper's environments are
small and discrete (bit flipping, grid worlds, a simple Fetch task), nothing like
1,500-step 25 Hz episodes. Expect the estimator to be unusable at this horizon
without truncation or clipping, which reintroduces bias.

**Skew-Fit: State-Covering Self-Supervised RL.** Vitchyr Pong, Murtaza Dalal,
Steven Lin, Ashvin Nair, Shikhar Bahl, Sergey Levine. ICML 2020.
arXiv:1903.03698.
*Mechanism:* prove that maximizing state coverage equals maximizing goal-reaching
performance plus the ENTROPY OF THE GOAL DISTRIBUTION; then iteratively train a
generative model on collected data with rare states UP-WEIGHTED, which provably
converges to uniform over reachable states.
*Verdict:* the theoretical backbone for MEGA-style frontier goal selection.
Off-policy, image-based, robotics results.

**Bottom line on the binary-reward proposal.** Trott et al.'s own baselines are
the most honest available evidence: on a deceptive point maze, PPO+ICM (i.e.
binary reward plus a curiosity bonus) discovered the goal in 2 of 5 runs and
DDPG+HER in 1 of 5, while PPO+SR (dense distance reward plus the anti-goal
correction) got 5 of 5. In other words, on the task that most resembles this one,
DELETING the dense reward and adding novelty was measurably WORSE than KEEPING
the dense reward and correcting it. Savinov 2019 is the counter-example where
sparse plus an episodic REACHABILITY bonus does work - on a static first-person
maze, which is the right setting - so the proposal is worth one arm, but the
novelty must be temporal-distance based, not count-over-cells, or it reproduces
the null already on the ledger.

---

## 7. What to try first

Ranked for THIS problem: on-policy PPO, 2048 envs at 600k steps/s, first-person
depth, a resettable simulator, a known finish box, no demos, no map constants,
and a mechanism that must carry the same constants to every map.

### 1. Sibling Rivalry (Trott et al. 2019, arXiv:1911.01417)

*Why first:* it is the only paper found that matches the problem (distance-to-goal
shaping creating a local optimum at a wall), the trainer (on-policy PPO), and
every constraint (no map, no demos, no replay buffer, no extra network) at the
same time. It is a ~50-line change to the reward function plus pairing the envs.

*Concrete recipe:* pair the 2048 envs into 1024 sibling pairs sharing (s0, g).
Reward `min[0, -d(s,g) + d(s, gbar)]` with gbar = the sibling's terminal state,
and `+1` inside the finish box. Always keep the farther sibling; include the
closer one only if it finished or if `d(s_T^f, s_T^c) <= epsilon`. Paper
constants: lr 1e-3, entropy 0.025, 4 epochs; delta = the finish-box radius;
epsilon = 5.0 (point maze) / 10.0 (ant maze). **Set epsilon = c * d(s0, g) with
one dimensionless c fixed across all four maps** - the paper's absolute values are
map constants and would violate the standing rule.

*Why it should beat count-novelty, precisely:* count novelty tries to OUT-BID a
sustained 9.6-to-40 reward deficit with a bonus that decays exactly when the
agent commits. Sibling Rivalry does not bid at all - it CANCELS the pull. The
anti-goal is the local optimum itself (both siblings end at the wall), so the
gradient at the wall changes sign instead of being slightly offset, and the
`min[0, .]` clip removes the reason to sit at the potential minimum in the first
place. It also introduces no new stable optimum, because as soon as episodes
reach the goal the two distance terms cancel and the objective collapses to the
sparse reward - which is the property every additive bonus lacks.

*Risks to state in the ledger:* never demonstrated on image observations or at
25 Hz; MEGA reports it is ~100x less sample-efficient than frontier goal
selection on PointMaze (irrelevant at 600k steps/s, but it means "no effect in
50M steps" is not yet a null); and the pairing halves the number of distinct
(s0, g) pairs per update.

### 2. Reverse curriculum from the map's own finish box (Florensa et al. 2017, arXiv:1707.05300)

*Why second:* it is the only mechanism here that makes the +50 terminal bonus
ACTUALLY OBSERVED, and no exploration bonus does that. Once the value function has
seen the goal, the deceptive shaping is competing against a true value gradient
instead of against nothing. It reuses the spawn-state machinery this repo already
has, and its provenance is the map's own goal definition plus the policy's own
random walks, so it is clean under rule 0 (the ledger entry must say exactly
that).

*Concrete recipe, paper constants unchanged:* seed the start set with the finish
box. Each iteration, pick seeds from the current set, run T_B = 50 random-action
steps (Sigma = I) to produce nearby states, keep a pool of M = 10,000, sample
N_new = 200 new starts plus N_old = 100 replayed old ones, and KEEP only starts
whose current success rate lies in (R_min, R_max) = (0.1, 0.9). Evals still start
at the true map start, always.

*Why it should beat count-novelty:* it changes the START DISTRIBUTION rather than
the reward, so there is nothing for the dense potential to out-vote. The
(0.1, 0.9) filter is also the documented cure for the trivial-win trap already on
this ledger - a start the policy solves reliably is deleted from the curriculum,
so the win rate cannot be inflated by harvesting states next to the goal.

*Risks:* Florensa explicitly assumes the random-action chain is a communicating
class. On the walking labyrinth that is plausible; on a surf map a backward
Brownian walk from the finish will not synthesize the 2,800 u/s airborne approach,
so expect the curriculum to stall at the point where backward reachability stops,
and report where that is - it is itself a useful measurement.

### 3. Episodic reachability bonus, in STEPS (Savinov et al. 2019, arXiv:1810.02274; modern form: ETD, arXiv:2501.15418)

*Why third:* it is the exploration bonus whose units are right. The logged failure
here is that novelty over position cells (and over velocity/view keys) is farmed
in place at the wall. A reachability bonus scores a state by how many STEPS it is
from everything already in this episode's memory, so a shuffle - whose states are
1-5 steps from memory - earns approximately zero no matter how many new cells it
touches, while the far leg of a detour earns the full bonus. Its constants
(k = 5 steps, K = 200 memory entries, 90th percentile, beta = 0.5) are in units
of STEPS and episodes, never world units, so they carry across maps unchanged by
construction.

*Concrete recipe:* R-network = small siamese CNN on the existing 64x32 depth
image, trained online (the "variable duration" variant, beta = 1.0) to classify
reachable-within-k = 5 decisions versus not; episodic memory K = 200 embeddings,
admission gated by the same novelty test; bonus
`alpha * (beta - percentile_90(C(M, e)))`. Budget the reported 1.84x throughput
cost. Run it BOTH on top of the existing Euclidean shaping and on a binary
0/1 reward - the sparse arm is the user's own proposal, done with the right
novelty measure.

*Risks:* Savinov's tasks are SPARSE, not DECEPTIVE - the bonus is proven to find
a goal a random walk could find, not proven to beat a competing dense reward.
Pair with a shaping coefficient sweep or with the anti-goal correction of item 1.
Second, the R-network trained on early data will be badly calibrated far from the
start.

### 4. Replace the potential with a LEARNED temporal distance (contrastive successor distance, arXiv:2406.17098 / arXiv:2501.15418; Laplacian route: arXiv:1810.04586 with the RA-LapRep fix arXiv:2210.13153)

*Why fourth:* this is the user's own stated target - learn the geodesic structure
from the agent's own experience instead of being handed a BFS. It trains on the
rollouts PPO already produces, needs no map, no demos and no extra data
collection, and on-policy PPO is exactly the setting where Devlin and Kudenko's
dynamic-PBRS invariance holds, so the potential may be updated during training
without losing the guarantee. If item 3 is run, the SAME network serves both
purposes.

*Concrete recipe:* train a contrastive critic on (s, s_future) pairs with the
future sampled geometrically (gamma ~ 0.99), energy parameterized as a potential
network minus a quasimetric network so the estimate obeys the triangle
inequality; use `Phi(s) = -d_learned(s, goal)` as the shaping potential with the
gamma factor and a zero terminal potential. If using the Laplacian instead, use
the proper augmented-Lagrangian objective (arXiv:2310.10833) and the
reachability-aware rescaling (arXiv:2210.13153), d = 20-50 dimensions.

*Why it should beat count-novelty:* it is not a bonus at all. It removes the
CAUSE - a potential that says the detour is wrong - rather than paying the agent
to ignore it.

*The honest limit, and it is decisive for ranking:* the learned distance is only
correct where the agent has been. Before the first crossing it will be as
deceptive as the Euclidean one. So this is a CONSOLIDATOR, not a discoverer: it
should be run stacked on item 1, 2 or 3, and judged on whether it stops the
frontier regressing and whether it transfers to the next rung with no constant
changed. RA-LapRep exists precisely because plain LapRep's L2 distance does not
reflect reachability - do not skip that fix.

### 5. Finish Go-Explore: policy-based return plus self-imitation (Ecoffet et al., Nature 2021, arXiv:2004.12919; cell-free variant arXiv:2208.14928; goal selection arXiv:2303.13002 / arXiv:2007.02832)

*Why fifth, and why not lower:* the repo's phase 1 already SOLVED the hard part -
a reward-free archive found the detour in 28 minutes. The stated reason for
abandoning it ("random macro-actions fail on maps that need skill") is the exact
problem the Nature paper's second half exists to solve: stop replaying macro
actions, and instead (a) train a GOAL-CONDITIONED POLICY to return to archive
entries, so returning is a learned skill, and (b) SELF-IMITATE the successful
trajectory to fold the discovery into the main policy. Dropping the method one
step before its own fix is the weakest link in the current evidence.

*Concrete recipe:* keep the archive but drop hand-chosen cells in favour of LGE's
learned latent plus density (inverse-dynamics encoder), which also removes the
last map constant. Replace both broken leaf-selection rules with either
farthest-point sampling over the archive (Huang et al. 2019), HIGL's
coverage-plus-novelty criterion (arXiv:2110.13625), or MEGA's
`argmin_g p_hat(g)` least-density rule (arXiv:2007.02832) - all three are
map-free, and MEGA's is the one with the strongest maze numbers. Return with a
goal-conditioned policy; explore from the returned state with ez-greedy
(zeta, mu = 2) rather than uniform random actions.

*Risks:* the largest build in this list; the Nature paper's headline numbers used
domain-knowledge cells; and the archive is another moving part whose failure mode
(returning repeatedly to a dead-end pocket) this repo has already observed.

### Honourable mentions, cheap and worth stacking

* **ez-greedy with zeta(mu = 2)** (arXiv:2006.01782) - one constant, minutes to
  implement, and it is temporally persistent in a way the OU / colored noise
  already tried here is not. Run it inside whichever of the above is chosen.
* **SOFE** (arXiv:2310.18144) - if ANY additive bonus is retained, augment the
  observation with the bonus's sufficient statistics so the PPO critic can fit a
  stationary target. A coefficient sweep cannot fix a non-stationary reward, and
  that is a live candidate explanation for the null count-bonus sweep.
* **NovelD's difference form** (NeurIPS 2021), alpha = 0.5 - makes any bonus a
  potential DIFFERENCE, so it cannot be farmed by oscillating at the wall.
* **Inverse-dynamics features for the novelty key** (E3B, arXiv:2210.05805) - the
  principled version of "do not put the view angle in the key", learned rather
  than chosen.
* **Bi-level shaping weights** (arXiv:2011.02669) - after the first finishes
  exist, the generic way to stop the Euclidean potential being re-learned.

### What the evidence does NOT support

No paper found solves a SINGLETON (non-procedural) 3D first-person detour maze,
from scratch, under a deceptive dense distance reward, without doing at least one
of: (a) resetting into states near the goal, (b) keeping an archive of the
agent's own states, (c) removing or correcting the dense reward, or (d) learning
a world model and selecting subgoals in its latent space (Director). Every
top-5 item above is one of those four. Any plan that is none of them is, on the
current literature, unsupported.

---

## Appendix: full citation list

Deception, novelty and quality-diversity
1. Lehman, Stanley. Abandoning Objectives: Evolution Through the Search for
   Novelty Alone. Evolutionary Computation 19(2), 2011.
2. Conti, Madhavan, Such, Lehman, Stanley, Clune. Improving Exploration in
   Evolution Strategies for Deep RL via a Population of Novelty-Seeking Agents.
   NeurIPS 2018. arXiv:1712.06560.
3. Mouret, Clune. Illuminating search spaces by mapping elites.
   arXiv:1504.04909, 2015.
4. Nilsson, Cully. Policy gradient assisted MAP-Elites. GECCO 2021.
5. Cideron, Pierrot, Perrin, Beguir, Sigaud. QD-RL: Efficient Mixing of Quality
   and Diversity in RL. arXiv:2006.08505, 2020.

Shaping theory
6. Ng, Harada, Russell. Policy Invariance Under Reward Transformations. ICML 1999.
7. Wiewiora. Potential-Based Shaping and Q-Value Initialization are Equivalent.
   JAIR 19, 2003. arXiv:1106.5267.
8. Wiewiora, Cottrell, Elkan. Principled Methods for Advising RL Agents. ICML 2003.
9. Devlin, Kudenko. Dynamic Potential-Based Reward Shaping. AAMAS 2012.
10. Grzes. Reward Shaping in Episodic Reinforcement Learning. AAMAS 2017.
11. Hu, Wang, Jia, Wang, Chen, Hao, Wu, Fan. Learning to Utilize Shaping Rewards.
    NeurIPS 2020. arXiv:2011.02669.
12. Wang, Yang, Dong, Sun, Liu, U. Efficient Potential-based Exploration in RL
    using Inverse Dynamic Bisimulation Metric. NeurIPS 2023.
13. Yang, Preuss, Plaat. Potential-based reward shaping in Sokoban.
    arXiv:2109.05022, 2021.

Exploration bonuses
14. Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos. Unifying Count-Based
    Exploration and Intrinsic Motivation. NeurIPS 2016. arXiv:1606.01868.
15. Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, De Turck, Abbeel.
    #Exploration. NeurIPS 2017. arXiv:1611.04717.
16. Pathak, Agrawal, Efros, Darrell. Curiosity-driven Exploration by
    Self-supervised Prediction (ICM). ICML 2017. arXiv:1705.05363.
17. Burda, Edwards, Storkey, Klimov. Exploration by Random Network Distillation.
    ICLR 2019. arXiv:1810.12894.
18. Savinov, Raichuk, Marinier, Vincent, Pollefeys, Lillicrap, Gelly. Episodic
    Curiosity through Reachability. ICLR 2019. arXiv:1810.02274.
19. Badia, Sprechmann, Vitvitskyi, Guo, Piot, Kapturowski, Tieleman, Arjovsky,
    Pritzel, Bolt, Blundell. Never Give Up. ICLR 2020. arXiv:2002.06038.
20. Badia, Piot, Kapturowski, Sprechmann, Vitvitskyi, Guo, Blundell. Agent57.
    ICML 2020. arXiv:2003.13350.
21. Zhang, Xu, Wang, Wu, Keutzer, Gonzalez, Tian. NovelD. NeurIPS 2021.
22. Seo, Chen, Shin, Lee, Abbeel, Lee. RE3. ICML 2021. arXiv:2102.09430.
23. Liu, Abbeel. Behavior From the Void (APT). NeurIPS 2021. arXiv:2103.04551.
24. Yarats, Fergus, Lazaric, Pinto. Proto-RL. ICML 2021. arXiv:2102.11271.
25. Henaff, Raileanu, Jiang, Rocktaschel. Exploration via Elliptical Episodic
    Bonuses (E3B). NeurIPS 2022. arXiv:2210.05805.
26. Henaff, Jiang, Raileanu. A Study of Global and Episodic Bonuses for
    Exploration in Contextual MDPs. ICML 2023. arXiv:2306.03236.
27. Guo, Thakoor, Pislar, Avila Pires, Altche, Tallec, Saade, Calandriello,
    Grill, Tang, Valko, Munos, Azar, Piot. BYOL-Explore. NeurIPS 2022.
    arXiv:2206.08332.
28. Wan, Tang, Tian, Kaneko. DEIR. IJCAI 2023. arXiv:2304.10770.
29. Creus Castanyer, Romoff, Berseth. Improving Intrinsic Exploration by Creating
    Stationary Objectives (SOFE). ICLR 2024. arXiv:2310.18144.
30. Jiang et al. Episodic Novelty Through Temporal Distance (ETD). ICLR 2025.
    arXiv:2501.15418.

Archives and return-then-explore
31. Ecoffet, Huizinga, Lehman, Stanley, Clune. Go-Explore. arXiv:1901.10995, 2019.
32. Ecoffet, Huizinga, Lehman, Stanley, Clune. First return, then explore.
    Nature 590:580-586, 2021. arXiv:2004.12919.
33. Gallouedec, Dellandrea. Cell-Free Latent Go-Explore. ICML 2023.
    arXiv:2208.14928.
34. Lu, Hu, Clune. Intelligent Go-Explore. arXiv:2405.15143, 2024.

Learned distances, goal-conditioned RL and graphs
35. Wu, Tucker, Nachum. The Laplacian in RL. ICLR 2019. arXiv:1810.04586.
36. Wang, Zhou, Feng, Hooi, Wang. Reachability-Aware Laplacian Representation.
    ICML 2023. arXiv:2210.13153.
37. Gomez, Bowling, Machado. Proper Laplacian Representation Learning. ICLR 2024.
    arXiv:2310.10833.
38. Machado, Bellemare, Bowling. A Laplacian Framework for Option Discovery.
    ICML 2017. arXiv:1703.00956.
39. Machado, Rosenbaum, Guo, Liu, Tesauro, Campbell. Eigenoption Discovery
    through the Deep Successor Representation. ICLR 2018. arXiv:1710.11089.
40. Machado, Bellemare, Bowling. Count-Based Exploration with the Successor
    Representation. AAAI 2020. arXiv:1807.11622.
41. Jinnai, Park, Abel, Konidaris. Discovering Options for Exploration by
    Minimizing Cover Time. ICML 2019. arXiv:1903.00606.
42. Klissarov, Machado. Deep Laplacian-based Options for Temporally-Extended
    Exploration. ICML 2023. arXiv:2301.11181.
43. Eysenbach, Salakhutdinov, Levine. Search on the Replay Buffer. NeurIPS 2019.
    arXiv:1906.05253.
44. Savinov, Dosovitskiy, Koltun. Semi-parametric Topological Memory for
    Navigation. ICLR 2018. arXiv:1803.00653.
45. Emmons, Jain, Laskin, Kurutach, Abbeel, Pathak. Sparse Graphical Memory for
    Robust Planning. NeurIPS 2020. arXiv:2003.06417.
46. Huang, Liu, Su. Mapping State Space using Landmarks for Universal Goal
    Reaching. NeurIPS 2019. arXiv:1908.05451.
47. Wang, Torralba, Isola, Zhang. Optimal Goal-Reaching RL via Quasimetric
    Learning (QRL). ICML 2023. arXiv:2304.01203.
48. Eysenbach, Zhang, Levine, Salakhutdinov. Contrastive Learning as
    Goal-Conditioned RL. NeurIPS 2022. arXiv:2206.07568.
49. Myers, Zheng, Dragan, Levine, Eysenbach. Learning Temporal Distances:
    Contrastive Successor Features. ICML 2024. arXiv:2406.17098.
50. Park, Ghosh, Eysenbach, Levine. HIQL. NeurIPS 2023. arXiv:2307.11949.
51. Steccanella, Evans, Simsek, Jonsson. Learning the Minimum Action Distance.
    arXiv:2506.09276, 2025.
52. Tamar, Wu, Thomas, Levine, Abbeel. Value Iteration Networks. NeurIPS 2016.
    arXiv:1602.02867.

Hierarchy, subgoals, frontier
53. Nachum, Gu, Lee, Levine. Data-Efficient Hierarchical RL (HIRO). NeurIPS 2018.
    arXiv:1805.08296.
54. Levy, Konidaris, Platt, Saenko. Learning Multi-Level Hierarchies with
    Hindsight (HAC). ICLR 2019. arXiv:1712.00948.
55. Kim, Seo, Shin. Landmark-Guided Subgoal Generation in HRL (HIGL).
    NeurIPS 2021. arXiv:2110.13625.
56. Hafner, Lee, Fischer, Abbeel. Deep Hierarchical Planning from Pixels
    (Director). NeurIPS 2022. arXiv:2206.04114.
57. Mendonca, Rybkin, Daniilidis, Hafner, Pathak. Discovering and Achieving Goals
    via World Models (LEXA). NeurIPS 2021. arXiv:2110.09514.
58. Sekar, Rybkin, Daniilidis, Abbeel, Hafner, Pathak. Planning to Explore via
    Self-Supervised World Models (Plan2Explore). ICML 2020. arXiv:2005.05960.
59. Hu, Chang, Rybkin, Jayaraman. Planning Goals for Exploration (PEG). ICLR 2023.
    arXiv:2303.13002.
60. Nasiriany, Pong, Lin, Levine. Planning with Goal-Conditioned Policies (LEAP).
    NeurIPS 2019. arXiv:1911.08453.
61. Jurgenson, Avner, Groshev, Tamar. Sub-Goal Trees. ICML 2020. arXiv:1906.05329.
62. Chaplot, Gandhi, Gupta, Gupta, Salakhutdinov. Learning to Explore using
    Active Neural SLAM. ICLR 2020. arXiv:2004.05155.
63. Yamauchi. A frontier-based approach for autonomous exploration. IEEE CIRA 1997.
64. Park, Rybkin, Levine. METRA. ICLR 2024. arXiv:2310.08887.

Going away from the goal, curricula, hindsight
65. Trott, Zheng, Xiong, Socher. Keeping Your Distance: Solving Sparse Reward
    Tasks Using Self-Balancing Shaped Rewards (Sibling Rivalry). NeurIPS 2019.
    arXiv:1911.01417.
66. Pitis, Chan, Zhao, Stadie, Ba. Maximum Entropy Gain Exploration for Long
    Horizon Multi-goal RL (MEGA/OMEGA). ICML 2020. arXiv:2007.02832.
67. Florensa, Held, Wulfmeier, Zhang, Abbeel. Reverse Curriculum Generation for
    RL. CoRL 2017. arXiv:1707.05300.
68. Florensa, Held, Geng, Abbeel. Automatic Goal Generation for RL Agents
    (Goal GAN). ICML 2018. arXiv:1705.06366.
69. Ivanovic, Harrison, Sharma, Chen, Pavone. BaRC: Backward Reachability
    Curriculum. arXiv:1806.06161, 2018.
70. Sukhbaatar, Lin, Kostrikov, Synnaeve, Szlam, Fergus. Intrinsic Motivation and
    Automatic Curricula via Asymmetric Self-Play. ICLR 2018. arXiv:1703.05407.
71. Campero, Raileanu, Kuttler, Tenenbaum, Rocktaschel, Grefenstette. Learning
    with AMIGo. ICLR 2021. arXiv:2006.12122.
72. Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel,
    Zaremba. Hindsight Experience Replay. NeurIPS 2017. arXiv:1707.01495.
73. Rauber, Ummadisingu, Mutz, Schmidhuber. Hindsight Policy Gradients. ICLR 2019.
    arXiv:1711.06006.
74. Pong, Dalal, Lin, Nair, Bahl, Levine. Skew-Fit. ICML 2020. arXiv:1903.03698.
75. Deep Reinforcement Learning with New-Field Exploration for Navigation in
    Detour Environment. IEEE ICARM 2021.

Temporally extended exploration
76. Dabney, Ostrovski, Barreto. Temporally-Extended epsilon-Greedy Exploration.
    ICLR 2021. arXiv:2006.01782.
77. Raffin, Kober, Stulp. Smooth Exploration for Robotic RL (gSDE). CoRL 2021.
    arXiv:2005.05719.
78. Eberhard, Hollenstein, Pinneri, Martius. Pink Noise Is All You Need.
    ICLR 2023.
79. Li, Zheng, Levine et al. Reinforcement Learning with Action Chunking
    (Q-chunking). NeurIPS 2025. arXiv:2507.07969. - temporally extended ACTION
    SPACE rather than temporally extended noise; the "different search space"
    option, currently demonstrated offline-to-online with a behavioural
    constraint, so it needs data this repo would have to generate itself.

Surveys
80. Aubret, Matignon, Hassas. An Information-Theoretic Perspective on Intrinsic
    Motivation in RL: A Survey. Entropy 25(2):327, 2023. arXiv:2209.08890.
