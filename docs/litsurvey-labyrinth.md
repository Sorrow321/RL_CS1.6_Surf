# Literature survey: the labyrinth problem (unitfarmer2's start gate)

Commissioned 2026-09-14 against `docs/uf2-exploration-review.md`, `docs/cross-review-guide.md`
and the ledger's unitfarmer2 entries (2026-09-13/14). Read-only: no training, no GPU, no
rentals. ASCII only.

The racing-domain survey (Sophy, Swift, Linesight, Necto), the start-distribution/plasticity
theory and the TAS strand are already in `docs/research-litsurvey.md` and are not repeated.
This covers only the labyrinth question: how does an on-policy agent come to prefer a route
its own value function correctly scores as worse.

**Rule 0b (user, 2026-09-14) applies to every mechanism below:** unitfarmer2
is a BENCHMARK, not the goal; the recipe must be map-agnostic (same flags,
same constants on every map) or an algorithm redesign with a better search
space. A mechanism whose constant or existence comes from looking at this
map (a pit-speed bonus, a climb bonus, a pit-tuned speed gate) is ad hoc and
excluded, whatever the literature says about the class it belongs to. See
`CLAUDE.md` section 0b.

---

## (a) Summary

No published method makes a policy-gradient agent prefer a route it has correctly measured as
bad, and that is structural rather than a gap in the reading: every optimism method is
optimistic about EPISTEMIC uncertainty and is built to stop being optimistic once a branch is
well sampled, which our pit is (847k visits in uf2NOV1). Four independent lines agree the
problem is not the bonus. (1) The reward is defective with a theorem attached: potential-based
shaping preserves the optimal policy only if the potential of the state where a trajectory
STOPS is zero (Grzes, AAMAS 2017); ours is not, so "bank potential then die" provably changes
the optimum and the critic is right to take the north slide. (2) On-policy PG has a formal
lower bound: gradients can be exponentially small when the visitation distribution does not
cover the states that matter (Agarwal, Kakade, Lee, Mahajan, JMLR 2021, Prop. 4.1), so gap 1
must be fixed by injecting visitation and cannot be waited out at any step budget, and gap 2
cannot pay until gap 1 has non-negligible success. (3) The one paper that is exactly our
"death is free while exploring" question is RND's dual-value-head trick with a NON-EPISODIC
intrinsic return (Burda et al., ICLR 2019, sec. 2.3); its sec. 3.7, "dancing with skulls", is
exactly our camping. (4) The untried family here is not another bonus or spawn rule but "more
gradient per visit": self-imitation of the best near-miss (Oh et al., ICML 2018) and a
self-generated monotone progress coordinate, which is this project's own largest measured win
(round 18, 0 -> 63/102 finishes). Treat everything that sounds like optimism with suspicion:
it is the most cited and least applicable part of this file.

---

## (b) Findings per question

### Q1. Dead-end / secure exploration

* **Fatemi, Sharma, Van Seijen, Kahou, "Dead-ends and Secure Exploration in Reinforcement
  Learning", ICML 2019 (PMLR v97, 1873-1881).** A dead-end is a state from which every action
  sequence still ends in the undesired terminal event. A second MDP with the same dynamics
  pays -1 on that terminal and 0 elsewhere; its optimal value V~*(s) is minus the probability
  that even optimal play ends in catastrophe, and V~*(s) = -1 marks a dead-end. An action is
  "secure" if it does not worsen the state's own worst case, and that auxiliary value CAPS any
  exploration policy. They name our map's shape: the "Bridge Effect", many dead-ends plus a
  distant positive reward. Code `github.com/Maluuba/srw` (archived; Bridge + Montezuma only).
  The follow-ups (NeurIPS 2021 medical dead-ends; DistDeD, TMLR 2023) are offline flagging,
  never exploration.
* **A death head as a positive exploration bonus for on-policy PG: no published instance, in
  either direction.** Neighbours all use the signal to shrink the action set or penalise risk
  - Eysenbach, Gu, Ibarz, Levine, "Leave no Trace", ICLR 2018; Grinsztajn, Ferret, Pietquin,
  Preux, Geist, "There Is No Turning Back", NeurIPS 2021 (code
  `github.com/nathangrinsztajn/NoTurningBack`), which pays a NEGATIVE bonus for apparently
  irreversible actions.
* **Verdict for us, and it is negative.** The construction cannot discriminate at
  unitfarmer2's start: both branches terminate in death with probability 1, so V~ is -1
  everywhere near the platform and the cap is vacuous. Worse, with gamma < 1 the auxiliary
  value degenerates into a ranking by TIME TO DEATH, and the north slide survives 6.3 s
  against the pit's 2 s, so a discounted dead-end head would actively prefer north. It becomes
  discriminative only after something survives, i.e. after gap 1.

### Q2. Optimism and uncertainty in policy gradient

* **The deep-exploration line is value-based and replay-bound:** Bootstrapped DQN (Osband,
  Blundell, Pritzel, Van Roy, NIPS 2016), Randomized Prior Functions (Osband, Aslanides,
  Cassirer, NeurIPS 2018), RLSVI (Osband, Van Roy, Wen, ICML 2016), and for
  information-directed sampling, Russo and Van Roy (NeurIPS 2014) plus Nikolov, Kirschner,
  Berkenkamp, Krause, ICLR 2019 (Atari only, no code found). The real claim is solid: per-step
  dithering explores exponentially worse than committing to one sampled hypothesis for a whole
  episode. But the machinery (bootstrap masks over replay, episode-long commitment to a Q
  head, posterior by perturbed regression) is Q-learning. No validated ensemble-critic PPO for
  exploration exists - a genuine gap in the literature, not a miss in the search.
* **Optimistic actor-critic is SAC/TD3 machinery.** Ciosek, Vuong, Loftin, Hofmann, NeurIPS
  2019 (code `github.com/microsoft/oac-explore`) shifts the BEHAVIOUR policy to `mu_Q +
  beta*sigma_Q` from twin critics under a KL ball while still TRAINING on the pessimistic
  bound; also SUNRISE (Lee et al., ICML 2021). Moskovitz et al., "Tactical Optimism and
  Pessimism", NeurIPS 2021, is the counterweight: the right amount of optimism varies by task
  and by training time.
* **The one PG-native optimism method:** O'Donoghue, "Efficient Exploration via
  Epistemic-Risk-Seeking Policy Optimization" (ERSAC), ICML 2023 (arXiv 2302.09339): a
  risk-seeking utility over the critic's epistemic uncertainty, solved as a two-player game
  with one learned temperature, with a Bayesian regret bound under function approximation.
  DeepSea depth 250 and Atari at ALE scale. No public code, no reproduction, nothing at 1B
  steps or in continuous pixel control.
* **Why the on-policy trap is formal.** Mei, Dai, Xiao, Szepesvari, Schuurmans, NeurIPS 2021
  (arXiv 2110.15572): with on-policy samples and no oracle separating optimal from suboptimal
  actions, an algorithm either converges almost surely at a rate no better than O(1/t), or
  converges faster and FAILS to reach the optimum with positive probability; their own remedy
  is an ensemble over initializations. Bhandari and Russo (arXiv 1906.08383) give the
  mechanism: an action's gradient scales with its own probability, so a branch pushed to zero
  stops producing the signal that could bring it back. Bolland, Lambrechts, Ernst (arXiv
  2402.00162) caution that entropy terms mostly SMOOTH the objective; they are not optimism.
* **Does optimism help when the route is genuinely negative rather than merely unvisited? No,
  by construction.** Every method above is optimistic in a quantity that DECREASES with data
  about that state, and our pit is sampled and returns -1 because a skill is missing.
  Mavor-Parker, Young, Barry, Griffin, ICML 2022, is the precise statement of what you get if
  you force it anyway.

### Q3. Deceptive rewards and detour gates

* **Ng, Harada, Russell, ICML 1999** proves `F = gamma*Phi(s') - Phi(s)` preserves the optimal
  policy; the guarantee is asymptotic and says nothing about learning dynamics. **Wiewiora,
  JAIR 2003** makes it concrete: our shaping is identical to initializing Q with "north is
  worth +9", an adversarial initialization at exactly the states we need explored.
* **The theorem we violate: Grzes, "Reward Shaping in Episodic Reinforcement Learning", AAMAS
  2017 (IFAAMAS proceedings p. 565), sec. 4.1-4.2.** "In finite horizon reinforcement
  learning, the potential function has to be set to zero for a state, sN, at which a
  particular learning trajectory stops"; if the potential was rising before sN, "all the high
  potentials accumulated before visiting sN are neutralised" by Phi(sN) = 0; and, against Ng
  et al., "the potential function of the goal states is a more important property because it
  can alter the optimal policy". **That is the death charge, and it is a correctness fix, not
  an arm.** Its Theorem 1 also helps: shaping preserves optimism in PAC-MDP learning if
  unknown states have Phi >= 0 and terminal states have Phi = 0, and "a rank order of unknown
  states is sufficient".
* **Novelty search.** Lehman and Stanley, "Abandoning Objectives", Evol. Comput. 19(2), 2011.
  Their deceptive maze is our shape, but the mechanism is population selection on a behaviour
  archive, not a scalar reward, and the claim is about REACHING a region, not learning a motor
  skill once there. Our result (10x enters then camps; 4x surfs ramp 1 and dies at ramp 2) is
  consistent with what they claim, not a refutation. Quality-diversity (MAP-Elites, arXiv
  1504.04909; Cully et al., Nature 521, 2015; PGA-ME, GECCO 2021; QD-PG, GECCO 2022) adds the
  property we want - local per-bin competition, so a pit trajectory never competes with a
  north one - but all results are low-dimensional state-based control and it replaces the
  training loop, not a term.
* **Go-Explore.** Ecoffet, Huizinga, Lehman, Stanley, Clune, arXiv 1901.10995 (2019) and
  "First return, then explore", Nature 590, 580-586 (2021). Domain-agnostic cell: a grayscale
  frame downsampled to 11x8, requantized to 8 levels; selection favours less-visited,
  less-often-chosen, more-recently-discovered cells (Appendix A.5; this repo's survey records
  `W = 1/sqrt(C_seen + 1)` and a `times_chosen_since_new` counter); post-return exploration is
  random actions with 95 percent repeat. On Pitfall with domain-knowledge cells (x, y, room,
  not a demonstration) it finds all 255 rooms and scores about 70,264, the first algorithm
  above zero there. Two things to carry: the paper has NO concept of a costly detour, because
  return is exact and free, which is the assumption we lack; and Uber's "busy highway problem"
  warning that archive-and-replay locks in the shortest risky crossing rather than the robust
  one, which matters because this trainer is not bit-reproducible.
* **Closest to our literal reward shape:** Trott, Zheng, Xiong, Socher, "Keeping Your
  Distance: Solving Sparse Reward Tasks Using Self-Balancing Shaped Rewards", NeurIPS 2019
  (arXiv 1911.01417) - explicitly about distance-to-goal shaping creating a local optimum,
  fixed with PAIRED rollouts from the same start that penalise converging twice on the same
  suboptimal behaviour. Read at abstract depth only; pairing spawn buckets across 2048 envs
  would be the port. That the phenomenon bites deep RL at all is documented by "Deceptive
  Games" (Anderson et al., EvoApplications 2018).

### Q4. The two-skill bootstrap

* **The formal statement that gap 1 comes first.** Kakade and Langford, CPI, ICML 2002 (the
  restart distribution mu and its concentrability cost), and Agarwal, Kakade, Lee, Mahajan,
  JMLR 22(98), 2021: Definition 3.1 is the distribution mismatch coefficient `||d^pi_rho /
  mu||_inf`, and sec. 4.3 / Prop. 4.1 proves that without a condition tying the training
  distribution to the states a good policy visits, gradients can be EXPONENTIALLY SMALL at
  highly suboptimal policies. While essentially every pit rollout dies, the gradient
  separating a better pit attempt from a worse one can be vanishingly small at any budget.
  Visitation must be injected; it cannot be waited for.
* **Reverse curricula.** Florensa, Held, Wulfmeier, Zhang, Abbeel, CoRL 2017 (code
  `github.com/florensacc/rllab-curriculum`): keep a start state only if the current policy's
  success rate from it is in (0.1, 0.9); generate candidates with 50 steps of Gaussian random
  actions from existing good starts. The success window is reusable; the Brownian generator is
  not, since isotropic action noise will not produce a ramp exit. Salimans and Chen (arXiv
  1812.03381) and Backplay (Resnick et al., arXiv 1807.06919) are backward curricula over one
  trajectory; neither requires that trajectory be HUMAN, so the self-generated version is
  legal here, but the catch is circular since gap 1 is "produce the first success".
* **Automatic curricula (ALP-GMM, CoRL 2019; Prioritized Level Replay, ICML 2021; PAIRED,
  NeurIPS 2020) are over parameterised LEVELS, not start states.** Their transferable part is
  the SCORING signal (positive value loss / regret) applied to spawn entries, which is close
  to the Necto difficulty weighting already run here and found null - not a fresh lever.
* **Self-imitation: the least-tried, most on-point item here.** Oh, Guo, Lee, Lewis, Singh,
  "Self-Imitation Learning", ICML 2018 (code `github.com/junhyukoh/self-imitation-learning`):
  store past episodes and add `-log pi(a|s) * max(R - V(s), 0)` plus `0.5*max(R - V(s), 0)^2`.
  It never pushes down. Published as A2C plus replay, but the loss has no importance ratio and
  bolts onto PPO as an auxiliary term. Why it matters: when the archive bench spawns 8
  episodes from fast pit states and all 8 die, on-policy advantage over 8 samples is a very
  thin signal, while SIL keeps pulling toward whichever attempt got furthest without any
  succeeding. The follow-up (Ferret, Pietquin, Geist, AAMAS 2021) is value-based, not a
  drop-in.
* **Hindsight.** Andrychowicz et al., HER, NIPS 2017: their own finding is that HER with
  SHAPED rewards is worse than with the sparse binary reward - a mismatch between what is
  optimised and the success condition, plus shaped rewards penalising the exploratory
  behaviour relabelling wants to keep. Rauber et al., ICLR 2019, is the on-policy version.
  Hindsight fixes SPARSITY, ours is deception, so it could help gap 1 (relabel every pit
  attempt as "reached where I ended up") and does nothing for gap 2 alone. Harutyunyan et al.,
  "Hindsight Credit Assignment", NeurIPS 2019, is the literature match to
  `docs/credit_diag.md` (TD/GAE-lambda smears credit by temporal proximity, not causal
  relevance) but has no large-scale PPO integration: a diagnosis, not a tool.

### Q5. Domain analogues

* **Linesight (TrackMania), pb4git and Agade, `github.com/Linesight-RL/linesight`, no paper.**
  Shipped config: `reward_per_m_advanced_along_centerline = 5/500` on virtual checkpoints
  about 10 m apart, `constant_reward_per_ms = -6/5000`, and the only instantaneous-distance
  term, `shaped_reward_dist_to_cur_vcp = -0.1`, HARD-CLIPPED between 2 and 25. Progress is a
  monotone checkpoint index and the distance term is bounded, so a shortcut pays a bounded
  per-step cost and banks the full uncapped index reward on rejoining ahead - independent
  confirmation of xAUTO's "the reference line supplies the ORDERING". Our ratchet is the
  stronger form of their clip, since a detour costs literally zero. Their line is driven by a
  human near the centerline, so `xSELF` is the compliant analogue and is ahead of the
  published art. The "7 second mini-race window" attributed to Linesight in round 18 could NOT
  be re-derived from the current public code; treat that attribution as uncertain.
* **GT Sophy (Wurman, Barrett, Kawamoto, MacGlashan, Subramanian et al., Nature 602, 223-228,
  2022)** confirms only the curriculum half: mixed-scenario training spawns the agent "in each
  possible position within the track-specific start configurations", short rollouts. No claim
  that a slow-in/fast-out corner defeats a naive progress reward was found. **Drone racing**
  (Song et al., IROS 2021; Swift, Nature 620, 2023) keys progress to the CURRENT GATE only,
  index-based like Linesight, and never had our dip: the time-optimal path between two gates
  in open air always makes local progress.
* **Pitfall and Montezuma.** Structural match: many actions lead to small negative rewards, so
  most algorithms learn not to move. RND (Burda, Edwards, Storkey, Klimov, ICLR 2019) scores
  -3 on Pitfall against PPO's 0 and human 6,464, and says so. Never Give Up (Badia et al.,
  ICLR 2020) was the first positive on Pitfall WITHOUT domain knowledge, using an episodic
  k-NN bonus times a lifelong RND modulator plus a bandit over a family of exploration
  weights; Agent57 (ICML 2020) extends it. Both are R2D2/replay, not portable here; exact
  scores unconfirmed.
* **Best-sourced concrete trick in the survey: Hoeller, Rudin, Choi, Hutter et al., "ANYmal
  parkour", Science Robotics 9 (2024) (arXiv 2306.14874).** The position-tracking reward is
  TIME-GATED: `1[t* < 1] * (1 - 0.5*||r_xy - r_xy*||)` at weight 10, firing only in the last
  second of the episode, heading gated the same way. The only always-on dense term is a small
  action-rate penalty, and no term penalises moving backwards, downwards, or temporarily away
  from the goal: a reward shape where a dip costs nothing by construction, and
  demonstration-free.
* **Zhuang, Fu, Wang, Atkeson, Schwertfeger, Finn, Zhao, "Robot Parkour Learning", CoRL 2023**
  (code `github.com/ZiwenZhuang/parkour`): forward progress plus energy, no reference motion;
  obstacles are semi-permeable early (a soft penetration penalty instead of a hard block),
  annealed to zero. Generalisable form: temporarily relax the cost that makes the correct
  manoeuvre look bad, then anneal it away. Speedrun RL is a non-result: "Gotta Learn Fast"
  (Nichol et al., arXiv 1804.03720) uses plain rightward progress and never faced a required
  detour.

### Q6. Death-aware value functions

* **Risk-seeking distributional RL is the only family attacking the "V is a mean" root
  cause.** Dabney, Ostrovski, Silver, Munos, IQN, ICML 2018: sampling the quantile fraction
  from a distorted distribution at action time yields risk-sensitive policies from one
  network, including risk-SEEKING (upper-tail) ones; predecessor QR-DQN (AAAI 2018),
  deep-scale actor-critic version DSAC (arXiv 2004.14547). Decisive caveat: an upper quantile
  AMPLIFIES observed upside, and if no rollout ever survived the pit the return distribution
  is a point mass with nothing to be optimistic about - a post-gap-1 mechanism. The
  exponential-utility line (Mihatsch and Neuneier 2002; Noorani and Baras, CDC 2021) is the
  PG-side analogue but has only convergence analysis and small demos.
* **PPO-native two-critic architectures already exist in safety RL.** Achiam, Held, Tamar,
  Abbeel, CPO, ICML 2017, and Ray, Achiam, Amodei, "Benchmarking Safe Exploration in Deep RL",
  2019, whose PPO-Lagrangian is exactly "PPO plus a separate cost critic plus a multiplier
  scaling the cost advantage" (code `github.com/openai/safety-starter-agents`). Every
  published use points the multiplier at suppression; no account of inverting it was found.
* **The directly usable death-aware result: Burda, Edwards, Storkey, Klimov, RND, ICLR 2019,
  sec. 2.3 and 3.2.** Their motivating paragraph is our problem verbatim: "Because the
  maneuver is tricky the chance of a game over is high, but the payoff to Alice's curiosity
  will be high if she succeeds. If Alice is modelled as an episodic reinforcement learning
  agent, then her future return will be exactly zero if she gets a game over, which might make
  her overly risk averse." Fix: treat the INTRINSIC return as NON-EPISODIC (not truncated at
  game over) while the extrinsic stays episodic, which needs two value heads, `V = V_E + V_I`,
  each fit on its own returns with its own discount (gamma_E = 0.999, gamma_I = 0.99; raising
  gamma_I to 0.999 hurt). They warn a non-episodic EXTRINSIC stream would be farmed by
  deliberately dying. Sec. 3.7, "Dancing with skulls", is our camping: once the agent has all
  the extrinsic reward it can reliably get, it settles into interacting with dangerous
  objects, because dangerous states are rare and therefore novel. Honest caveat: their own
  ablation found two heads gave no benefit when both streams were episodic - the gain is the
  non-episodic return.

---

## (c) Ranked mechanisms

By (evidence) x (fit) / (cost). Gap 1 = the exit skill inside the pit; gap 2 = the entry
decision at the platform.

**1. Zero the terminal potential (the death charge), no time penalty.** Charge back the banked
ratchet at death, `kappa * scale * max(rec_spawn - rec_death, 0)` - built, in flight as uf2DC
/ uf2DCk5. Why: Grzes shows a nonzero terminal potential ALTERS THE OPTIMAL POLICY, so the
north slide is genuinely optimal today and no exploration mechanism can repair an optimality
defect. Necessary condition for gap 2. Cost: done (`--death-charge 1.0 --time-pen 0`).
Falsified by: the greedy line still runs north at the same rate at BOTH kappa 1.0 and 0.5; or
the run collapses (ep_len at the floor, eval near zero) because every dead episode nets
exactly 0 and no gradient remains - that second outcome is the predictable failure of kappa =
1.0 and the reason kappa 0.5 may be worth more.

**2. Terminal, survival-gated progress payment (the ANYmal shape).** Stop paying potential per
tick; pay once at episode end for the deepest record the episode reached AND survived by the
`--unstuck-hold` window. Why: Hoeller et al. gate the position reward to the last second
precisely so no per-step arithmetic penalises a detour; with the survival gate this is a
strictly stronger form of ratchet + death charge and removes the dip from the objective by
construction rather than by cancellation. Gaps 1 and 2. Cost: about a day in `RaceReward`;
AliveReach already computes survived depth. Falsified by: the start line is unchanged AND
`ep_len_mean` does not rise, meaning the per-tick payment was never what selected the slide.

**3. Two value heads with a NON-EPISODIC intrinsic return.** Split race and novelty into
separate streams, fit `V_E` and `V_I` as two outputs of `value_head`, run GAE twice with
separate gammas, and do NOT cut the intrinsic return at death. Why: Burda sec. 2.3 is our
exact situation - today the bonus rides inside the episodic stream, so a dive that dies
forfeits the novelty it would have collected and the agent is risk-averse for the exploration
reason too. With mechanism 1 this gives "death expensive for the racer, free for the
explorer". Gap 2. Cost: about a day; `value_head` is already widened for `--priv-critic`, so
the shape change has precedent. Falsified by: pit contact from the true start does not rise
against the same `--int-coef` with one episodic head; or the agent starts dying on purpose to
farm novelty (Burda's own warning) - watch death cadence and `ep_len_mean`.

**4. A self-generated monotone arc coordinate over the policy's own pit descents.** Take the
recordings that entered fast (uf2NOV1 at 1,616 u/s; uf2EDGEsp10's archive at p50 1,263 u/s),
cut one episode with `tools/pick_selfline.py` (trim at the last tick the map pushed back),
build the .npz with `tools/build_route.py`, train with `--race-arc`. Why: arc length is
monotone by construction and cannot have the dip; this is the project's largest measured
effect (round 18 xARC/xAUTO/xSELF: corridor 231,680 and 47-63 finishes of 102, against 0 for
every field-shaped arm), and Linesight independently confirms the line need only supply
ORDERING. Champion-free: the line is the policy's own. Gaps 1 and 2 at once, because inside
the pit the arc pays for carrying speed toward ramp 2 where the field pays nothing. Cost: the
tools exist; the work is the composite, since a self-line ends at the pit bottom and leaves
most of the map unshaped, so `RaceReward` needs "arc on the prefix, geodesic ratchet past the
line's end" (about half a day). Falsified by: the greedy line still turns north with the arc
reward in place (the dip was never binding); or the trimmed line still descends at its tail,
in which case it pays for falling and the arm is a second treatment - xSELF's trimming rule
exists to prevent exactly that, so check the tail first.

**5. Self-imitation of the best near-miss.** Auxiliary PPO loss `-log pi(a|s) * max(R - V(s),
0)` plus `0.5*max(R - V(s), 0)^2` over a small buffer of the best recent episodes (Oh et al.,
ICML 2018). Why: the archive bench already puts the policy in the right place and all 8
episodes die, so the binding problem is that 8 samples of on-policy advantage is a thin
gradient; SIL extracts signal from whichever attempt got furthest with none succeeding, and
never pushes down. Nothing from this family has been tried here. Gap 1. Cost: about a day, no
new spawn machinery. Falsified by: with SIL on and a demonstrably non-empty buffer, the in-pit
speed rung and the depth at death do not improve across 1B.

**6. Temporally correlated exploration at manoeuvre scale, not per-step temperature.**
`--view-ou-sigma` with `--view-ou-period` at roughly 40-75 decisions (1.6-3.0 s at
`--act-every 4`): a held heading bias across the whole entry window. Why: the entry is a 2.5 s
commitment and iid per-step noise averages out at high control rates (Tallec et al., ICML
2019), which is exactly why temperature on ALL heads cost control (KL 0.056) and scored 0/12
with no entries. The OU offset is implemented, is exactly on-policy with no importance weight,
and has never been run on unitfarmer2. Note `--ez-eps` and `--spawn-burst` are the same idea
but are REFUSED under `--view-continuous` (train_fast.py, the VIEWC block near line 6690), so
the OU offset is the family's only member available on this action space - and uf2BURST could
not have run as configured. Gap 2. Cost: two flags. Falsified by: at matched KL against the
keys-temperature control, pit contact from the true start does not rise.

**7. Go-Explore on POSITION cells, plus a decoupled search for the exit.** (i) Rekey
`--respawn-mode goex` from distance bins to position cells with `W = 1/sqrt(C_seen + 1)` and
the `times_chosen_since_new` counter: the current mode weights DEPTH BINS, which cannot
separate pit from slide because both are shallow (Skew-Fit is the principled version of the
same weighting). (ii) Re-score `tools/tas_search.py` from geodesic progress to EXIT SPEED or
depth past the pit box, mutate a 2-6 s window inside the pit, and feed what it finds to the
existing METHOD (window + keys T, then T = 0 consolidation). Why: Go-Explore's point is
decoupling route discovery from the reward gradient, and the batched (1+lambda) hill climber
already evaluates 2048 candidates per rollout here. Gap 1. Cost: a few lines in the scorer
plus the existing method. Two flags: the busy-highway warning (a replayed exact line can be
fragile when the trainer is not bit-reproducible), and a RULES QUESTION - a simulator search
uses no human data but it is not the policy discovering the line, so it needs a ruling before
it counts as a pass.

**8. Late and optional: a risk-seeking critic.** An IQN-style quantile head read at an
upper-tail distortion, or O'Donoghue's ERSAC objective. Why: it attacks the root cause (V is a
mean) instead of adding a bonus. Why last: it amplifies observed upside and there is none -
with 0/8 survivals the return distribution at the platform is a point mass. Run only after 4,
5 or 7 produce survivors. Cost: high. Falsified by: the entry rate is unchanged even though
the critic's upper quantile at the platform is already above its mean.

---

## (d) What NOT to try

* **A bigger novelty coefficient, or any prediction-error curiosity.** Taiga, Fedus, Machado,
  Courville, Bellemare, ICLR 2020: bonus methods give no meaningful gain over epsilon-greedy
  on 5 of 6 hard-exploration games, and hyperparameters tuned on Montezuma do not transfer.
  RND is validated dead in this repo and scores -3 on Pitfall; Burda sec. 3.7 predicts our
  camping. Raising `--int-coef` is not a new experiment.
* **Episodic-only novelty as the anti-camping fix.** Henaff, Jiang, Raileanu, ICML 2023:
  episodic bonuses win when episodes share LITTLE structure (procedural levels), global
  bonuses when structure is shared. One fixed map is the maximal-shared-structure case, so the
  global count table is the right family.
* **Dead-end / secure-exploration masking as a stage-1 mechanism.** See Q1: both branches are
  fatal so the cap is vacuous, and with gamma < 1 the auxiliary value ranks by time-to-death
  and prefers the 6.3 s north slide.
* **Reversibility penalties** (Eysenbach et al. 2018; Grinsztajn et al. 2021): right
  mechanism, wrong sign - they push away from the pit. **HER on top of a shaped reward:** the
  HER paper reports it is worse than the sparse binary reward and explains why; hindsight has
  to replace the shaping, not ride on it.
* **Porting Bootstrapped DQN / OAC / SUNRISE / IDS.** All need a persistent replay buffer and
  a revisitable critic to define uncertainty at all, and no validated PPO instance exists.
  Architecture change, zero evidence at scale.
* **PLR / ALP-GMM / PAIRED as written, and MAP-Elites / QD as a training loop.** The first
  three are curricula over parameterised LEVELS and we have one map; their reusable part
  (score spawn states by value loss) is close to the Necto difficulty weighting already found
  null. QD is the cleanest structural answer to "temporarily lose reward" but has no
  pixel-scale evidence and is a loop replacement, not a term.
* **A time penalty together with a death charge.** Already measured here to collapse into
  suicide; Grzes explains the arithmetic - once the bank is charged back the clock is the only
  non-zero term and dying early maximises it.
* **Ranking two 1B nulls, or adding seeds to resolve one.** CLAUDE.md's gate ladder (2.7x
  between byte-identical configs) applies to every number here. Report which rung an arm
  cleared and at which step; prefer the early matched-step mark.
* **Shortening the horizon or re-running a 7 s mini-race window.** Settled in CLAUDE.md
  section 5; nothing in this survey changes it.

---

## Open items and honest gaps

* The Linesight "7 second mini-race window" is not in that project's current public config;
  the round-18 verdict does not depend on it, but the attribution should be corrected or
  dropped.
* NGU's and Agent57's exact Pitfall scores were not confirmed - only the qualitative claim,
  and only from secondary sources.
* Trott et al. (self-balancing shaped rewards) was read at abstract depth only. It is the
  closest paper to our literal reward shape and deserves a full read before mechanism 5 is
  designed.
* No paper flips a death-probability head into a positive exploration bonus, in any direction.
  Building one would be without precedent, which argues for a small falsifiable arm rather
  than a fleet batch.
* Two 2026 preprints were verified to exist but are single-paper and small-scale: Pham, Vaze,
  Chin, "Optimistic Policy Regularization" (arXiv 2603.06793, Atari at 10M steps, prevents
  FORGETTING a good trajectory rather than discovering a bad-looking one) and Nishimori,
  Parmas, "Retry Policy Gradients in Continuous Action Spaces" (arXiv 2606.05888, best-of-K
  retry objectives that raise entropy without an explicit bonus). Neither is recommended yet;
  the second is the more interesting for gap 2 if the first seven fail.
