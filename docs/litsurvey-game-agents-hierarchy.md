# Game agents, fast control and the planner question: how the real-time game agents got their long horizon

Literature survey, 2026-09-27. Web research plus this repo's own ledger. No code was run, no
GPU was touched, and nothing in the repo changed except this file. ASCII only.

**Commission (user, 2026-09-27, condensed).** "The best thing that I know is OpenAI Five,
but they don't have any planner there. Maybe the whole planner thing isn't needed." The
questions were:

1. How do successful agents in fast real-time 3D and continuous-control games combine
   high-frequency control with long-horizon navigation or strategy?
2. Where did explicit hierarchies help, and where did flat agents win?
3. What does the exploration literature say about long, deceptive routes with momentum?
   What turns a trajectory that was found but is fragile into a reliable policy?
4. Why are some tasks easy for humans and hard for RL?
5. What does all of that mean for RL_Surf under its rules: no demos, one recipe, and a fast
   exact simulator with state restore?

**What this file does NOT repeat.** The following are only pointed to, together with any
number that was read for this file and that they did not have:

* [LIT] `docs/research-litsurvey.md`: GT Sophy, Swift, Linesight, Rocket League, q1physrl,
  Pearce & Zhu's mouse grid, Go-Explore's phase-1 constants, Salimans & Chen, Backplay,
  Florensa, the DeepMimic citation, TAS search.
* [TIME] `docs/research-litsurvey-timing.md`: AlphaStar's delay head, OpenAI Five's frame
  skip, Sophy at 10 Hz, Linesight's latch, FiGAR / TempoRL / ez-greedy.
* [TEMP] `docs/research-litsurvey-temporal.md`: the memory, frame-stacking and chunking
  nulls, and VPT's autoregressive heads.
* [HIER] `docs/research-litsurvey-hier.md`: options and option-critic, FeUdal, HIRO, HAC,
  h-DQN, Director, Nachum 2019, Thakkar 2022 head-to-head racing, parkour,
  SoRB / SPTM / SGM / TTGS, PRM-RL / RoGuE, the MuZero family, DC-MCTS, MAGIC, ExIt, PTSP,
  TMInterface, the racers' gate inputs, BaRC.
* [NAV] `docs/litsurvey-robot-navigation-hierarchy.md`: legged-robot command interfaces and
  navigation stacks. Its section 6 already states "racing is the honest comparator and it
  has no macro".
* [DET] `docs/litsurvey-detour-navigation.md`: deception, shaping theory, count and
  episodic bonuses, Go-Explore and LGE, learned distances, Sibling Rivalry, reverse
  curricula.
* [PE] `docs/litsurvey-planner-executor.md`: joint two-level training, feasibility, plan
  proposals, robotics stacks.
* [LAB] `docs/litsurvey-labyrinth.md`, [HRLR] `docs/hrl-reward-survey.md`,
  [REP] `docs/plan-representation-survey.md`.
* The ledger (`docs/research-results.md`), in particular:
  * 2026-09-23 03:20: OpenAI Five read in full (the shaped-reward table, the LSTM for fog
    of war, the horizon, App. O.2 randomisation);
  * 2026-09-24 19:35 and 19:45: efPLN200 / efTGT200 / srFT200, a flat policy given a route
    on blue200;
  * 2026-09-27 02:01 to 03:16: the edge archive and the contingent archive.

**Conventions.**

* Numbers come from the cited source unless marked "(ours)", which means they were measured
  in this project's ledger.
* "NOT FOUND": I looked for it and the source does not have it.
* "[unverified]": taken from a secondary source or from memory, not re-read for this file.
* "figure only": the paper gives the number only in a plot.

---

## 0. The answer in one page

**On "OpenAI Five has no planner": correct.** Neither does any other superhuman real-time
game agent found for this survey. OpenAI Five, AlphaStar, FTW (Quake III CTF), GT Sophy and
its vision successor, Swift, Linesight, the Rocket League bots, VPT and DreamerV3 are all
flat policies. At most they have a recurrent core, and they were trained by model-free RL,
by RL in a world model, or by imitation plus RL. None of them learned its macro structure
from scratch with a planner.

None of them faced RL_Surf's macro problem either. In every one of them the macro problem
was removed before the flat learner ever saw it, in one of four ways:

| how the macro problem was removed | who | available to RL_Surf? |
|---|---|---|
| **The route is given** as a centreline, a reference line, checkpoints, gates or milestones, and progress is measured along it | Sophy, GT7 vision, Linesight, TMRL, Mario Kart Wii, Swift; DreamerV3 and VPT (12 milestones / per-item rewards along the tech tree) | Not on a surf map. The geodesic potential is our only generic route proxy, and it lies on detour maps. |
| **A dense shaped reward** that points uphill almost everywhere | OpenAI Five (about 20 hand-weighted terms); FTW (13 game events, weights evolved by PBT; "does not learn anything without reward shaping") | Map-specific shaping is forbidden (CLAUDE.md 0b). |
| **Human data** supplies the strategy | AlphaStar (Elo 149 without human data vs 1,540 with); VPT (random init "fails to achieve almost any reward"); Rocket League (70% human-replay start states); MineRL winners; the MOBA macro-strategy model | Forbidden (CLAUDE.md 0). |
| **Search with state restore** finds the route, then RL robustifies it | Go-Explore (Montezuma, Pitfall, robotics) | **Yes.** The edge archive (ours) found blue200's detour from the map start in 2-8 s, on six of six seeds. |

So OpenAI Five shows that a flat learner is enough **once the reward, or a route, points
the right way**. It is no evidence that a flat learner can find a route the reward hides.
AlphaStar's paper states that gap in one sentence: "It is highly improbable that naive
exploration will execute a precise sequence of instructions, over thousands of steps...".
It then closes the gap with human data.

**What that means here.** On blue200 the project's macro problem is now in the fourth row:
the archive finds the route in seconds. What is left is flying that route reliably from the
true start. That is exactly the problem the racing agents solve, with three things:

* a flat policy;
* progress measured along a known route;
* start states spread along the route and around it.

The ledger already has the proof of concept, on blue200 (ours):

* With a **feasible** route to the finish, a flat policy paid for progress along it
  finished **6/9 greedy at 202M steps** (efTGT200).
* The same route **passed to the policy through the fan** finished **9/9 at 202M**
  (srFT200).
* A route that crossed open air trapped the same flat policy (efPLN200: 0/9 through 504M).

The two good routes were hand-tuned on blue200: a diagnostic, not a recipe (CLAUDE.md 0b).
The archive's chain is the generic replacement, because it is a route the executor
actually flew.

**Ranked recommendation.** Section 5 gives the reasoning, the constants and the first
experiments.

1. **Go-Explore, with a robustify phase done the racing way.**
   * The archive finds the route, as it is built today.
   * The chain's flown path becomes a self-derived route (SELF_STATES provenance).
   * The FLAT recipe policy is trained on progress along that route, with spawns spread
     along the chain and around its landing clouds, from several chains rather than one.
   * Consolidate from the true start.
   * Rebuild the route from the policy's own runs (as Linesight does), and repeat.
   * Evidence: Go-Explore, Salimans-Chen and DTSIL; every racing agent; our own efTGT200,
     srFT200 and xSELF.
2. **Targeted practice at the hard edges, inside (1).** Prioritise starts at the edges
   whose survival from the states a real flight actually reaches is low (Swift's bounded
   perturbation, DART, PLR). This is Codex's step 2. It belongs in (1)'s start
   distribution, not in a planner.
3. **Decision-time search on top, only after (1).**
   * MCTS and the contingent archive lift a policy that is already locally reliable
     (blue100: 4/9 -> 8/9 (ours)).
   * They are capped by the executor (blue200: 1/18 (ours)).
4. **Environment randomisation over a map pool**, for the multi-map recipe (OpenAI Five
   App. O.2; our ladder, lab100 9/9).
5. **A learned re-planning planner, or a two-timescale recurrent core.** This has the
   weakest evidence for single-map macro discovery in games. Keep the executor's plan
   interface in two roles:
   * as the archive's move operator;
   * as a channel that conditions the policy on a whole route (AlphaStar's z, DTSIL),
     not as a policy that decides every 2 s.

---

## 1. Q1: How successful real-time game agents combine fast control with long horizons

### 1.1 Summary table

Details and sources are in 1.2-1.10.

| agent (game) | decision rate | planner or hierarchy | reward | how the long horizon is explored | memory | imitation | scale |
|---|---|---|---|---|---|---|---|
| OpenAI Five (Dota 2) | every 4th frame at 30 fps = 7.5 Hz (0.133 s) | none; items, skills and courier hand-SCRIPTED | ~20 shaped terms, zero-sum | entropy 0.01, randomised environment, self-play 80/20 | 4096-unit LSTM | none | 770 PFlops/s-days over 10 months |
| AlphaStar (StarCraft II) | agent-chosen delay, mean 370 ms; at most 22 non-duplicate actions per 5 s | no module; the policy is conditioned on a strategy statistic z | win/loss, plus z-following pseudo-rewards active 25% of the time | human data (z, supervised init, KL to it) and league exploiters | LSTM core | 971k replays | 12 agents x 32 TPUv3 x 44 days |
| FTW (Quake III CTF) | 15 Hz [unverified] | two-timescale recurrent core (slow LSTM every tau = 5-20 steps) | weights on 13 game events, evolved by PBT | population self-play | 2 LSTMs + DNC | none | 30 agents, 450K games |
| Pearce & Zhu (CS:GO) | 16 Hz | none | none (behaviour cloning) | none | ConvLSTM | 95 h scraped + 3.3 h expert | one gaming GPU at run time |
| Arnold (ViZDoom) | frame skip 4 | two networks, switched by a hand rule: navigation / action | pickups, health, ammo; navigation paid per distance travelled | epsilon 1 -> 0.1 | DRQN | none | 1M-frame replay |
| DFP / IntelAct (ViZDoom) | frame skip 4 | none | future measurements at 1-32 steps, weighted by a goal vector | epsilon | none | none | 51-128M steps |
| GT Sophy (GT Sport) | 10 Hz | none | course progress plus penalties, no finish bonus | mixed scenarios, random starts | MLP | none | days on PS4s [LIT] |
| GT7 vision agent | 10 Hz over 60 Hz | none | centreline progress plus 7 other terms | random start points, 0-19 opponents, car specs +-25% | GRU (removing it = "complete failure" at overtaking) | none | 20 PS4 workers |
| Swift (drone racing) | onboard inference 8 ms; control rate NOT FOUND | none | progress to the next gate, perception, smoothness, crash -5.0 | random gate, bounded perturbation of observed passing states | MLP | none | 1e8 steps in 50 min, 100 agents |
| Linesight (TrackMania) | 20 Hz latch over 100 Hz [TIME] | none | time penalty plus progress along a reference line (points every 0.5 m) | epsilon / Boltzmann; stop after 2 s without progress | last 5 actions | the reference line comes from a replay that "does not need to be fast" | one PC, ~80 h on one track [unverified] |
| Mario Kart Wii (BTR) | frame skip 4 | none | +1 per checkpoint (100 per lap), low-speed penalty, +10 finish | Rainbow-style | frame input | none | ~160M frames |
| VPT (Minecraft) | 20 Hz, raw mouse and keyboard | none | per-item shaped rewards along the tech tree | KL to the behaviour-cloning prior | transformer | 70k h of YouTube, labelled by an IDM | 720 V100 x 9 days, then 1.4e10 RL frames |
| DreamerV3 (Minecraft) | 20 Hz, 25 categorical actions | none | +1 at each of 12 milestones | world-model imagination; block breaking sped up | RSSM | none | 100M steps, 17 GPU-days |
| Sonic baselines (Retro) | sticky frame skip 4 | none | max-x progress plus a completion bonus | PPO / Rainbow; joint pretraining on many levels | none | none | 1M steps per test level |
| **RL_Surf (ours)** | 25 Hz over 100 Hz physics | planner + executor (under test) | potential progress plus a finish bonus | reservoir, archive, temperatures | none by default | none (rule 0) | ~0.7B steps/h on one GPU |

### 1.2 OpenAI Five

Berner et al., "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv:1912.06680, 2019.

The ledger (2026-09-23 03:20) read the whole paper. What matters for the planner question:

**The game and the time scale.** Games run "at 30 frames per second for approximately 45
minutes. OpenAI Five selects an action every fourth frame, yielding approximately 20,000
steps per episode".

**The network.**
* One 4096-unit LSTM (84% of 159M parameters).
* Truncated BPTT over 16-step windows.
* A semantic observation, no pixels.

**No hierarchy anywhere in the method.** Hard parts were scripted instead: "the order in
which heroes purchase items and abilities, control of the unique courier unit, and which
items heroes keep in reserve".

**The reward.**
* About 20 hand-weighted terms (gold, XP, health, kills, towers ...).
* The sparse ablation (0-1 loss, horizon 1 hour, gamma 0.99996) "succeeds at reaching
  TrueSkill 155; for reference, a hand-coded scripted agent reaches TrueSkill 100". That
  is far below the shaped run (the baseline curve is figure only).

**The horizon.**
* Defined as H = 0.133 s / (1 - gamma). The GAE horizon was grown from 60 s to 840 s over
  training.
* Resuming a skilled agent with a longer horizon raised its win rate. Their comment: "we
  expect long-horizon planning to be present in highly-skilled agents, but not from-scratch
  agents".

**Exploration.** Three levers: an entropy bonus (0.01 best), team spirit, and environment
randomisation. App. O.2 randomised Roshan's health so that a long, specific sequence was
sometimes short.

**Scale.**
* 770 +- 50 PFlops/s-days.
* Batches of 1-3M timesteps.
* "Over twenty surgeries" (network changes carried across) over 10 months.

**For RL_Surf.**
* Five's long horizon rode on three things:
  * a reward that points uphill almost everywhere;
  * scripted hard parts;
  * randomisation that makes a hard sequence sometimes easy.
* The first two are forbidden here (0b, 0).
* The third was reproduced: the labyrinth ladder, lab100 9/9 from 303M (ours).
* The LSTM was for fog of war. Our GRU arms on the labyrinth were null (ours).

### 1.3 AlphaStar

Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement
learning", Nature 575:350-354, 2019. Read here from DeepMind's unformatted PDF.

**Architecture.**
* An LSTM core, a transformer over units, autoregressive action heads and a pointer
  network.
* 139M weights, 55M of them used at inference.

**Timing.**
* About 110 ms from observation to action.
* The agent chooses when it observes next: 370 ms on average, "but possibly multiple
  seconds".
* A monitoring layer allows at most 22 non-duplicate actions per five-second window.

**The APM ablation (Fig. 3G), Elo:**

| APM limit | 0% | 10% | 25% | 50% | 100% | 200% | none |
|---|---|---|---|---|---|---|---|
| Elo | 0 | 1,145 | 1,419 | 1,536 | 1,540 | 1,411 | 1,392 |

More actions per minute HURT, "possibly because the agent spends more effort refining
micro-tactics than learning diverse strategies".

**How strategy was explored.** The paper's own statement of the problem: "Consider a
policy that has learned to build and utilize the micro-tactics of ground units. Any
deviation that builds and naively uses air units will reduce performance. It is highly
improbable that naive exploration will execute a precise sequence of instructions, over
thousands of steps, that constructs air units and effectively utilizes their
micro-tactics."

The answer was human data:
* Supervised learning on 971,000 replays from players with MMR > 3500 (the top 22%).
* A strategy statistic z: the first 20 buildings and units, plus cumulative statistics.
  z is zeroed 10% of the time in supervised learning.
* In RL, the policy is initialised from the supervised policy and a KL to it is kept.
* Pseudo-rewards for following z: the edit distance of the build order and the Hamming
  distance of the statistics. Each is active with probability 25% and has its own value
  function.

**The human-data ablation (Fig. 3E), Elo.** For reference, the built-in Elite bot is 603
and the Very Easy bot 418.

| no human data | supervised only | human init | + supervised KL | + statistics z |
|---|---|---|---|---|
| **149** | 936 | 1,020 | 1,400 | 1,540 |

**The league.**

| | main agents | + main exploiters | + league exploiters |
|---|---|---|---|
| Elo | 1,540 | 1,693 | 1,824 |
| relative population performance | 6% | 35% | 62% |

**RL components.** Average win rate against fixed opponents: V-trace 49%, + TD(lambda) 73%,
+ UPGO 82%. UPGO is a self-imitation update toward better-than-average returns. Giving the
value function the opponent's observations (training only) raised the win rate from 22%
to 82%.

**Scale and result.**
* Each agent trained on 32 TPUv3 for 44 days, with 16,000 concurrent matches, and its
  learner ran about 50,000 agent steps/s.
* Final ratings: 6,275 / 6,048 / 5,835 MMR, above 99.8% of ranked players.
* Supervised learning alone reached 3,699 (above 84%).

**For RL_Surf.**
* 149 vs 1,540 is the strongest single number in this survey against "flat RL from
  scratch finds the strategy". RL with no human data scored below the Very Easy bot.
* **z is a plan without a planner.** A strategy is drawn once per game from a library. The
  policy is paid for following it only part of the time, and the unconditional policy is
  trained too. Under rule 0 the generic analogue is a route drawn from our own archive
  (section 5, rank 1b).
* UPGO is worth 9 points: the same family as the repo's SIL.

### 1.4 FTW (Quake III Capture the Flag)

Jaderberg et al., "Human-level performance in 3D multiplayer games with population-based
reinforcement learning", Science 364(6443):859-865, 2019; arXiv:1807.01281.

**Setting.** 2v2 CTF on procedurally generated maps, from pixels and game points. "Agents
do not have access to models of the environment, other agents, or human policy priors."
There is no planner and no map.

**Architecture.**
* "A hierarchical RNN consisting of two recurrent networks (LSTMs) operating at different
  timescales."
* The slow LSTM "produces an updated hidden state every tau steps". It sets a prior over
  the fast core's latent, and a KL term regularises the fast core toward it.
* tau is sampled from [5, 20) and evolved by PBT.
* A shared DNC memory.
* 540 actions from six independent action dimensions.
* 84x84 input at 15 Hz [unverified; this is how Pearce & Zhu 2021 describe FTW].

**Reward.**
* The internal reward is a weight on each of 13 game-point events, evolved by PBT.
* Ablation: "game winning reward only does not learn anything without reward shaping"
  (Fig. 2, figure only).

**Training.**
* A population of 30 agents over about 450K games, in 1,920 concurrent arenas.
* PBT exploits when the estimated win probability is below 70%, and perturbs a value by
  +-20% with probability 5%.

**The temporal hierarchy.**
* FTW without it has a lower Elo (figure only).
* "Waiting in the opponents base for a flag to reappear (opponent base camping) which we
  only observed in FTW agents with a temporal hierarchy."

**Against humans.**
* Reaction time 258 ms vs 559 ms; tagging accuracy 80% vs 48%.
* With the reaction time slowed or the accuracy lowered to human levels, the agents still
  won.
* Human testers won 25% of games (6.3% draws) after 12 hours of practice.

**Follow-up.** PPR (Stooke, Dalibard, Jayakumar, Czarnecki, Jaderberg, arXiv:2006.15223,
2020):
* One slow core and three fast cores (perception, prediction, reaction).
* An auxiliary loss makes the three cores' policies agree.
* It improves over LSTM agents on the memory tasks of DMLab-30 and on CTF (figure only).

**For RL_Surf.**
* FTW is the only superhuman game agent with a built-in two-timescale structure. That
  structure lives INSIDE the recurrent core and is trained end to end; it is not a
  separate planner.
* Its long horizon still needed dense learned shaping.
* The temporal hierarchy bought a tactic (camping) in a partially observed multi-agent
  game. It did not discover a route the reward hides.

### 1.5 Counter-Strike

Pearce & Zhu, "Counter-Strike Deathmatch with Large-Scale Behavioural Cloning",
arXiv:2104.04258. NeurIPS 2021 Offline RL workshop; IEEE CoG 2022.

**Data.**
* 5.5M frames (95 h) scraped from public servers.
* Clean expert data: 190k frames (3.3 h) of Dust2 deathmatch and 45k frames of aim
  training.

**Agent.**
* Acts at 16 Hz: an EfficientNetB0 + ConvLSTM on 280x150 input.
* The mouse is on a non-uniform grid of 19 x 13 options, finer near zero.
* Each action head has its own loss.

**Result.** 3.72 kills per minute and a K/D of 2.09 against medium bots. The medium bot
itself scores 2.41 / 1.00 and a casual human 3.51 / 2.48: the agent "matches the
performance of the medium difficulty built-in AI". Limitations: poor vertical aim, and it
"quickly forgets" enemies behind cover.

**Why no RL.** The game has no API and runs in real time only, "precluding many
reinforcement learning algorithms".

**Follow-ups are supervised, not RL.**
* Durst et al., "Learning to Move Like Professional Counter-Strike Players" (SCA 2024,
  arXiv:2408.13934): a transformer movement model from 123 h of professional play, under
  0.5 ms per step for all players.
* DECOY (Wang, Ustun, McGroarty, Winter Simulation Conference 2025, arXiv:2509.06355): a
  discretised CS:GO simulator with movement on a waypoint graph, for strategic planning
  research. It reports no RL results.

No published RL agent that plays full CS, or surf or bhop, was found. The nearest RL on the
same movement physics is q1physrl on Quake [LIT] 5.

**For RL_Surf.** CS research uses imitation because the game cannot be simulated fast. We
are in the opposite position: an exact simulator at ~200k steps/s, and no demos. The only
transferable observation is that when the CS community studies strategy, it abstracts
movement into a WAYPOINT GRAPH.

### 1.6 ViZDoom: Arnold and DFP

**Arnold.** Lample & Chaplot, "Playing FPS Games with Deep Reinforcement Learning", AAAI
2017, arXiv:1609.05521.
* A DRQN with an auxiliary head that predicts game features (for example, an enemy in
  view), which "dramatically improve[s]" training.
* TWO networks, switched by a hand rule: "If no enemies are detected in the current frame,
  or if the agent does not have any ammo left, the navigation network is called to decide
  the next action. Otherwise, the decision is given to the action network."
* The navigation network has 3 actions. It is paid for item pickups and "a small positive
  reward proportional to the distance it travelled".
* Frame skip 4.

| K/D | Arnold | comparison |
|---|---|---|
| vs built-in bots (limited deathmatch) | 5.52 with the navigation network | 4.64 without it |
| vs humans (single player) | 5.12 | humans 1.52 |

It placed second in both tracks of the 2016 ViZDoom competition, with the highest K/D.

**DFP / IntelAct.** Dosovitskiy & Koltun, "Learning to Act by Predicting the Future", ICLR
2017, arXiv:1611.01779.
* Supervised prediction of future measurements (ammo, health, frags) at offsets of 1, 2,
  4, 8, 16 and 32 steps. The action is the argmax of the goal-weighted predictions.
* Frame skip 4.
* It won the 2016 Full Deathmatch track, "outperforming the second best submission by more
  than 50%".
* D3 scenario: 33.5 frags vs A3C's 5.6.
* Ablation: three measurements at six offsets give 22.6 frags; frags only at one offset
  gives 5.0.

**For RL_Surf.**
* The hierarchy that helped in Doom was a HAND-DESIGNED phase switch (navigate or fight),
  and navigation was paid for distance travelled: an exploration reward.
* DFP's prediction at several horizons is the value-side cousin of a two-timescale agent.
  The fast policy sees consequences 32 steps ahead without any planner.

### 1.7 Racing: GT Sophy, the GT7 vision agent, TrackMania, Mario Kart, Swift

**GT Sophy** (Wurman et al., Nature 602:223-228, 2022). [LIT] 1 has the details. What
matters here:
* a flat QR-SAC policy at 10 Hz;
* progress along the course;
* no racing line supplied;
* the course's lookahead points in the observation.

**The GT7 vision agent.** Lee, Seno, Tai, Subramanian, Kawamoto, Stone, Wurman, "A
Champion-level Vision-based Reinforcement Learning Agent for Competitive Racing in Gran
Turismo 7", RA-L 2025, arXiv:2504.09021.
* An ego camera plus onboard sensors; 10 Hz over a 60 Hz game.
* A GRU actor: "completely removing the RNN resulted in complete failure, with the agent
  unable to overtake any opponents".
* An asymmetric critic that sees 177 3D track-edge points and a grid of opponents.
* Progress is "r_t^p = p_t - p_{t-1}", with p the position projected onto the centreline.
* Starts: "randomly sampled points around the track", with 0-19 opponents and car specs
  randomised by +-25%.
* It beats a human champion.
* No planner and no reference trajectory in the actor.

**TrackMania.**
* Linesight ([LIT] 3, [TIME] 1.4, [HIER] 25).
  * The reference line is "a trajectory along which the agent's progress is measured".
  * "Points are sampled every 50cm."
  * "This reference trajectory does not need to be fast. The authors usually drive along
    the centerline of the map" (Linesight docs, "custom training").
  * It beat ten of twelve official campaign world records.
* The other TrackMania agents, from a secondary history (hallofdreams.org) [unverified]:
  * TMRL (Bouteiller & Geze): LIDAR input, reward counted along a recorded reference
    trajectory.
  * PedroAI: DQN over 100+ maps, reward = speed along the reference trajectory; it
    "plateaued" after 169 days.
  * Yosh: "training wheels" (reward a technique, then remove the reward).
  * Every one of them measures progress along a line. None of them plans.

**Mario Kart Wii.** Clark, Towers, Evers, Hare, "Beyond The Rainbow", ICML 2025,
arXiv:2411.03820.
* Rainbow Road at 150cc, with items.
* 4 actions: accelerate, drift left, drift right, use item. Frame skip 4.
* Reward:
  * +1 per checkpoint, "100 in total per lap";
  * -0.01 per frame below 65 km/h;
  * -10 and termination after 80 slow frames;
  * +10 for finishing, plus a position bonus.
* About 160M frames to consistent completion.
* Where the checkpoints came from: NOT FOUND.

**Swift.** Kaufmann et al., "Champion-level drone racing using deep reinforcement
learning", Nature 620:982-987, 2023.
* Reward: r = r_prog + r_perc + r_cmd - r_crash.
  * r_prog = lambda1 (d_{t-1} - d_t), where d is the distance to the NEXT gate's centre.
  * The crash term is 5.0.
  * The lambda values are NOT FOUND in the main text (they are in Extended Data Table 1a).
* A 31-d observation that includes the next gate's four corners (R^12). A 2 x 128 MLP.
* 100 agents; 1e8 steps in 50 minutes.
* Starts: "every agent is initialized at a random gate on the track, with bounded
  perturbation around a state previously observed" there.
* It won 5/9, 4/7 and 6/9 races against three champions, and set the fastest time.

**For RL_Surf.**
* Every racer is flat.
* Every racer is told the ROUTE: a centreline, a reference line, checkpoints or the gate
  order.
* None is told the LINE: the racing line always emerges from progress.
* [NAV] 6 made the same point for Sophy. It holds across the whole genre, including the
  vision agent and the kart game.
* The start distributions are spread along the route, with perturbation (Swift, GT7). That
  is precisely what a route that was found but is fragile needs (section 3.7).

### 1.8 Minecraft

**VPT.** Baker et al., NeurIPS 2022, arXiv:2206.11795.
* An IDM (0.5B parameters), trained on ~2k h of contractor play, labels ~70k h of YouTube
  video.
* The behaviour-cloning foundation model: 0.5B parameters, 9 days on 720 V100, native
  20 Hz mouse and keyboard.
* RL fine-tuning (PPG) keeps a KL to the frozen prior and adds per-item shaped rewards,
  over 1.3M episodes (1.4e10 frames).

| item, share of 10-minute episodes | diamond pickaxe | diamonds | iron pickaxe |
|---|---|---|---|
| VPT after RL fine-tuning | 2.5% | ~20% | >80% |

RL from a random initialisation "fails to achieve almost any reward" and "never learns to
reliably collect logs".

**DreamerV3.** Hafner, Pasukonis, Ba, Lillicrap, arXiv:2301.04104; Nature 2025.
* A flat actor-critic inside a world model, with fixed hyperparameters across domains.
* Minecraft Diamond: 20 Hz, 25 categorical actions, +1 at each of 12 milestones (once per
  episode), 36,000-step (30 min) episodes.
* The environment was changed. Block breaking was sped up (following Kanitscheider et al.
  2021) "because a stochastic policy is unlikely to sample the same action often enough in
  a row to break blocks without regressing its progress by sampling a different action".
  The jump key is held for 200 ms in the background.
* Results. The two versions differ, and the Nature text was not re-read:

| version | runs | result |
|---|---|---|
| arXiv v1 | 40 seeds x 100M steps, 17 GPU-days | diamonds in 50 episodes in total; the first after 29M steps; 24/40 seeds found at least one |
| later version | - | "All the Dreamer agents we trained on Minecraft discover diamonds in 100M environment steps" |

**Director** (Hafner et al., NeurIPS 2022, arXiv:2206.04114) [HIER] 6.
* The manager picks a goal every K = 8 steps.
* It solves sparse Ant mazes where flat Dreamer fails.
* On standard benchmarks it needed task reward to the worker to close the gap to
  DreamerV2.
* Its own stated limits: the fixed K, and tasks "that require precise timing".
* The diamonds were later obtained by the FLAT DreamerV3, not by Director.

**MineRL competitions.**
* 2019 (Milani et al., arXiv:2003.05012):
  * the budget was 8M environment samples and 144 h;
  * NO team obtained a diamond;
  * of the top nine teams, 67% used hierarchical RL, 78% used human data and 100% reduced
    the action space;
  * the winner (CDS, 61.61) used a hierarchical DQN over expert demonstrations.
* 2021 winner JueWu-MC (Lin et al., arXiv:2112.04907): a high-level controller over
  options plus a low-level worker per sub-task, with ensemble behaviour cloning and
  discriminator-based self-imitation.

**LLM-planned agents** (Voyager, Plan4MC, DEPS, GITM) [not read]. These are explicit
planners that take the tech tree from a language model's human knowledge. There is no
analogue for an unseen surf map.

**For RL_Surf.**
* In Minecraft, hierarchy appears exactly where humans supply the sub-task vocabulary
  (demos, milestones, an LLM).
* The one success from scratch (DreamerV3) is flat, and it was paid at 12 human-chosen
  milestones along the route.
* It also needed the environment changed, because a stochastic policy cannot hold a key
  long enough. That is the commitment problem that `--keys-hold` solved here. It is also
  the reason a per-decision sampling executor cannot hold a line through a turn.

### 1.9 A momentum game with a deceptive reward: Sonic the Hedgehog

Nichol, Pfau, Hesse, Klimov, Schulman, "Gotta Learn Fast: A New Benchmark for
Generalization in RL", arXiv:1804.03720, 2018.

**Reward.** Cumulative reward is proportional to the horizontal offset, normalised to 9,000
at the level's end. A completion bonus of 1,000 decays to 0 at 4,500 steps (about 5
minutes). Sticky frame skip 4.

**The deception is named.** "The immediate rewards can be deceptive; it is often necessary
to go backwards for prolonged amounts of time." In Figure 2 (Labyrinth Zone, Act 2): "For
an average player, it takes 20 to 30 seconds to get through the red and orange segments."

**The baselines' fix.** The wrapper "rewards the agent based on deltas in the maximum
x-position... not punished for backtracking". It "gives a sizable performance boost" but
"still gives no information about when or how an agent should go backwards".

**Scores at 1M steps per test level:**

| humans | PPO | Rainbow | JERK | joint PPO | joint Rainbow |
|---|---|---|---|---|---|
| 7,438 | 1,489 | 2,749 | 1,904 | 3,128 | 2,969 |

JERK is a replay of the best action sequences, related to The Brute (3.2). It beat PPO,
"likely due to JERK's perfect memory and its tailored exploration strategy".

**For RL_Surf.**
* This is the closest published analogue of a surf detour with momentum.
* Max-x progress is our ratchet.
* Beyond human priors, the one big lever was TRANSFER from many training levels: joint PPO
  scored 2.1x plain PPO. That is the map-pool argument (section 5, rank 4).
* A memory-based replay of the best sequence beat a learned policy at this budget.

### 1.10 Other flat real-time agents, one line each

* Rocket League bots [LIT] 4: flat PPO; 70% of start states come from human replays; hard
  mechanics are shaped at goal magnitude.
* q1physrl [LIT] 5: flat PPO at 72 Hz with a 0.3 s key dwell beat the human record on a
  straight strafe map. It has no macro problem.
* Super Smash Bros. Melee (Firoiu, Whitney, Tenenbaum, arXiv:1702.06230): flat deep RL
  agents "competitive against and even surpass human professionals" [details not read].

### 1.11 What the agents have in common

1. **Decision rates are 7.5-20 Hz**, with action repeat or a latch over a faster engine.
   No one gets the long horizon from a slower decision rate. AlphaStar is the exception
   that proves it: its agent-chosen delay (mean 370 ms) is a LEARNED commitment,
   supervised from human data. RL_Surf's 25 Hz over 100 Hz is inside the band.
2. **The long horizon rides on a reward that points uphill almost everywhere** (Five, FTW,
   every racer, Minecraft's milestones). The discount horizon grows with skill (Five:
   60 s -> 840 s). Where the reward did not point uphill, human data did the exploring
   (AlphaStar, VPT, MineRL, Rocket League).
3. **Memory is for partial observability**, not for exploration: Five's fog of war, GT7's
   opponents off screen, FTW's base entrances. Our GRU arms were null on the labyrinth
   (ours).
4. **When hierarchy is present, it is one of three things:**
   * inside the recurrent core (FTW);
   * a hand-designed phase switch (Arnold);
   * a vocabulary supplied by humans (MineRL, StarCraft macros, the MOBA macro model;
     section 2.3).
5. **Scale is large, but not out of reach for the control half.** Swift trained in 1e8
   steps (50 minutes) and Mario Kart in 160M frames. We run ~0.7B steps per hour on one
   GPU (ours).

---

## 2. Q2: High-frequency vs low-frequency decisions

### 2.1 Action repeat and decision rate

[TIME] covers the mechanisms: AlphaStar's delay head, FiGAR, TempoRL and ez-greedy.

New numbers in this file:
* DreamerV3's environment edit and its background jump hold (1.8);
* AlphaStar's APM ablation, where more actions per minute hurt (1.3);
* FTW's slow timescale, tau in [5, 20) (1.4).

Ours (ledger 2026-09-27 01:51): planning every 0.5 s was worse than every 2 s at every
early mark.

| own steps | map-start progress, plan every 2 s | map-start progress, plan every 0.5 s |
|---|---|---|
| +50M | 17.6% | 8.8% |
| +175M | 29.9% | 18.3% |

A choice redrawn every 0.5 s is not a commitment.

### 2.2 Two-timescale architectures, and what they were for

* **FTW** (1.4): a slow LSTM every tau = 5-20 steps sets a KL prior over the fast latent.
  It bought a tactic, not a route.
* **PPR** (1.4): the same idea with three fast cores. It helped memory tasks.
* **FeUdal, HIRO, HAC, option-critic, Director**: [HIER] 1-6 and [NAV] 3 have the
  constants and the failure modes.
* **"Flattening Hierarchies with Policy Bootstrapping"** (Zhou & Kao, NeurIPS 2025
  spotlight, arXiv:2505.14975). A FLAT goal-conditioned policy, trained by bootstrapping on
  subgoal-conditioned policies with advantage-weighted importance sampling, "matches or
  surpasses" hierarchical offline GCRL on long-horizon OGBench mazes (the numbers were not
  read). So the hierarchy's benefit can be moved into TRAINING and dropped at deployment.

### 2.3 Where explicit hierarchies clearly helped in complex games, and what supplied the high level

| where | the high level | where its vocabulary came from | evidence |
|---|---|---|---|
| Montezuma (h-DQN) | picks a subgoal | hand-specified objects | ~400 per episode vs DQN 0 [HIER] 5 |
| Montezuma and others (FuN) | a latent direction every c = 10 steps | learned | ~2,600 vs LSTM 400 [HIER] 2 |
| ViZDoom (Arnold) | switches between navigating and fighting | a hand rule (an enemy detector) | K/D 5.52 vs 4.64 |
| StarCraft II (Pang et al., AAAI 2019, arXiv:1809.09095) | a controller over macro-actions | "automatically extracted from expert's trajectories" | ">93%" vs the level-7 built-in AI (Protoss) within two days |
| StarCraft II (TStarBots, Sun et al., arXiv:1809.07193) | hand macro-actions (flat RL over them), or a hard-coded hierarchy | hand-designed | beats built-in levels 1-10 (levels 8-10 cheat) |
| Honor of Kings (Wu et al., AAAI 2019, arXiv:1812.07887) | a macro strategy: "where to go on the map" and the game phase | human replays | 48% win rate vs top-1% human teams |
| MineRL winners, 2019 and 2021 | a sub-task controller | demonstrations | the best scores; no diamond in 2019 |
| Quake III CTF (FTW) | the slow LSTM | learned end to end | camping appears only with it; an Elo gap (figure only) |
| 3D exploration (Active Neural SLAM, Chaplot et al., ICLR 2020, arXiv:2004.05155) | "The Global policy samples a new goal every 25 timesteps" on a learned top-down map, then an analytic planner and a local policy | a learned map with frontiers | see below |
| Racing tactics (Thakkar et al. 2022) | MCTS at 1 Hz over waypoints | a discretised track | >90% of head-to-head races [HIER] 8 |

**Active Neural SLAM, in detail** (coverage in m^2 / fraction explored):

| | full system | global policy replaced by a frontier heuristic | best end-to-end baseline |
|---|---|---|---|
| coverage | 32.70 / 0.948 | 30.98 / 0.925 | 24.86 / 0.789 |

It also won the CVPR 2019 Habitat PointNav challenge (0.950 success, 0.846 SPL).

**The common thread: a high level works when its VOCABULARY is supplied** (objects, phases,
macros from demonstrations, a map with frontiers, a discretised track). When the high level
is itself learned from scratch in a rich game, it helps modestly (FuN, FTW), and never on a
problem as deceptive as ours. Active Neural SLAM is the cleanest ablation. Most of its gain
comes from the explicit map and the planner. A LEARNED global policy adds about 5% over a
frontier rule.

### 2.4 Where flat agents with scale and shaping won

* Every superhuman system in 1.1.
* Minecraft: the diamonds were obtained by flat DreamerV3, not by Director.
* Sonic: flat joint pretraining is the best non-human result.
* Nachum et al. 2019 [HIER] 7: flat agents with the same temporally extended exploration
  match HIRO on AntMaze, AntPush and AntBlock.

### 2.5 What went wrong in hierarchical attempts

**In the literature.**
* Options collapse to one step without a deliberation cost [HIER] 1.
* The low level is non-stationary under the high level [HIER] 3-4.
* Learned goal spaces are uninterpretable without an autoencoder [HIER] 6.
* The high level asks for infeasible subgoals [PE] 1.
* Co-training from zero collapses: MLSH's warm-up rule [PE] F3.
* Director's fixed K is weak on tasks that need precise timing.
* The hierarchies that win need a human vocabulary (MineRL, StarCraft).

**Ours (ledger).**
* **Joint training.** Joint planner x executor training matched the no-plan baseline "to
  the metre" on medium01, and failed easy01 (2026-09-25 21:52).
* **The plan is not followed as a curve.** The executor completes 0.4% of the planner's
  curves, so the planner's 6 numbers act as a latent code (review 2026-09-25, 2.3).
* **The planner's reward.** The refund reward is the Euclidean potential, one level up
  (review 2.2).
* **The habit on blue200.** Every planner, warm or fresh, learned "turn toward the goal"
  on blue200's corridor (2026-09-26 12:45).
* **The 3-choice planner** plateaued at 30-37% of the map from the start, with 0/9 greedy
  (2026-09-27 01:51).
* **Re-planning more often** (every 0.5 s) was worse than every 2 s.

**What DID work above the executor was SEARCH** (ours):
* the jump planner (no learning) solved labyrinth_hard01 at depth 12;
* MCTS at decision time took blue100 from 4/9 to 8/9;
* the edge archive found blue200's route in 2-8 s.

The pattern matches the literature. Search over an exact model works, and supplied
vocabularies work. A learned high level under a deceptive reward learns the deception at a
coarser time scale.

---

## 3. Q3: Exploration on long, deceptive routes with momentum, and making a found route reliable

### 3.1 The failure, as the big systems state it

* **AlphaStar:** "It is highly improbable that naive exploration will execute a precise
  sequence of instructions, over thousands of steps..." (1.3).
* **OpenAI Five, App. O.2** (ledger 2026-09-23): "If a long and very specific series of
  actions is necessary to be taken by the agent in order to randomly stumble on a reward,
  and any deviation from that sequence will result in negative advantage, then the longer
  this series, the less likely is agent to explore this skill thoroughly".
* **DreamerV3:** "a stochastic policy is unlikely to sample the same action often enough in
  a row..." (1.8).
* **Sonic:** "it is often necessary to go backwards for prolonged amounts of time" (1.9).

All four describe the same structure. Per-step stochastic exploration does not find a long,
precise sequence that runs against the reward's gradient, or that has no gradient. The
fixes were:
* human data (AlphaStar);
* a randomised environment in which the sequence is sometimes short (Five);
* a changed environment (DreamerV3);
* memory of the maximum progress reached (Sonic, a modest gain).

None of them used a learned planner.

Mountain Car is the textbook momentum version: drive away from the goal to gain energy.
Count-based exploration over (position, velocity) is the classic remedy [general knowledge,
not re-read]. Our archive keys on position, speed and heading, i.e. on phase space. That is
the principled reason it can find a pit dive that the potential hides.

### 3.2 Go-Explore and robustification

Numbers are new to the repo unless marked.

**Ecoffet et al., "Go-Explore: a New Approach for Hard-Exploration Problems",
arXiv:1901.10995, 2019.**

*Phase 1, exploration.*
* Cells:
  * without domain knowledge: frames downscaled to 11x8 with 8 intensity levels;
  * with domain knowledge: room, position on a 16x16-pixel grid, level and keys.
* Explore step: random actions for 100 frames, repeating the previous action with
  probability 95%.
* Montezuma level 1 took 640M frames without domain knowledge, and 57.6M with it.

*Phase 2, robustification.*
* Salimans & Chen's backward algorithm on PPO, trained with random no-ops (up to 30) and
  sticky actions.
* **"Only 40% of our attempts at robustifying trajectories ... were successful when using
  a single demonstration"**; with 10 demonstrations, "all 5 robustification runs were
  successful".
* Cost: 4.35B frames (2.4 days) per robustification run.
* Robustified scores: 43,763 without domain knowledge, 666,474 with it.

**Ecoffet et al., "First return, then explore", Nature 590:580-586, 2021;
arXiv:2004.12919.**
* 11 hard games were robustified, each from "10 + 1 virtual" demonstrations, all under
  sticky actions. Montezuma 43,791; Pitfall 6,954.
* **Policy-based Go-Explore** returns with a goal-conditioned policy instead of a restore.
  * It follows the archived trajectory with a soft window, N_w = 10.
  * Goals: 10% an adjacent cell not yet in the archive, 22.5% any adjacent cell, 67.5% a
    cell from the archive.
  * It removes the need for robustification.
  * Montezuma 97,728; Pitfall 20,093.
  * Sampling the policy finds more than 4x as many cells as random actions.
* **Robotics.** In a shelf task where PPO finds no reward in 1B frames, the found
  trajectories are "exceptionally diverse", and robustifying them "produces robust and
  reliable policies in 99% of cases".
* **Definitions that describe our failures.**
  * Detachment: the algorithm stops visiting frontier regions too early.
  * Derailment: "exploratory mechanisms ... prevent it from returning".
  * The fix: return with minimal exploration, then explore.

**Go-Explore in a large 3D game.** Lu, Georgescu, Verwey, "Go-Explore Complex 3D Game
Environments for Automated Reachability Testing", arXiv:2209.00570, 2022. Cells come from
the game's navmesh; return is by checkpoint restore. It "can fully cover a vast 1.5km x
1.5km game world within 10 hours on a single machine", and beats intrinsic-curiosity RL
([LIT] 6 has the coverage numbers).

**The Brute and sticky actions.** Machado et al., JAIR 61:523-562, 2018;
arXiv:1709.06009.
* The Brute is an open-loop trajectory-tree search that exploits determinism.
* Under the 2015 protocol it beat the best learning method of the time on 45 of 55 Atari
  games.
* With sticky actions at 0.25, Asterix fell from 6,909 to 308, while DQN went from 3,501 to
  3,123.
* Determinism makes a found sequence replayable. Any execution noise breaks it.

### 3.3 The backward algorithm and reverse curricula

**Salimans & Chen, "Learning Montezuma's Revenge from a Single Demonstration",
arXiv:1812.03381, 2018.**
* Episodes start from a window {tau-D..tau} of demo states. tau moves back when the success
  rate is >= rho = 20%.
* 1,024 workers (128 GPUs x 8), about 50B frames, 2 weeks.
* Score 74,500 against the demo's 71,500.
* Robustness:

| execution condition | score |
|---|---|
| clean (as trained) | 74,500 |
| random no-ops | held near the demo level |
| sticky actions at 0.25 | 10,000 |
| random actions at 1% | 8,400 |

  The policy was robust to where it started, not to execution noise it never trained with.
* It failed on Gravitar and Pitfall: "unable to find hyperparameters that worked".

**DTSIL.** Guo et al., "Memory Based Trajectory-conditioned Policies for Learning from
Sparse Rewards", NeurIPS 2020, arXiv:1907.10247.
* A buffer of DIVERSE trajectories from the agent's own experience.
* A trajectory-conditioned policy imitates a sampled trajectory (imitation reward 0.1 per
  matched step), then explores past its end.
* Which trajectory to follow is chosen by count, 1/sqrt(n).
* No demonstrations and no resets, under sticky actions. State of the art at the time on
  Montezuma and Pitfall, in under 5B frames.
* Caveat: its state embedding uses the ground-truth room and position, which is domain
  knowledge.
* The design argument: "we provide the agent with the full trajectory leading to the goal
  state", for "richer intermediate information and denser rewards" than the goal alone.

**DeepMimic.** Peng, Abbeel, Levine, van de Panne, SIGGRAPH 2018, arXiv:1804.02717.
Reference state initialisation (RSI) and early termination (ET), normalised return:

| skill | RSI + ET | ET only | RSI only |
|---|---|---|---|
| backflip | 0.791 | 0.730 | 0.379 |

"For the backflip, without RSI, the policy never learns to perform a full mid-air flip":
skills with a flight phase need starts spread along the motion.

**Florensa, BaRC, Backplay and GoalGAN:** [HIER] 26, [LIT] 6, [DET] 7.2.

**StreetLearn.** Mirowski et al., "Learning to Navigate in Cities Without a Map", NeurIPS
2018, arXiv:1804.00168.
* A flat LSTM agent navigates a city at kilometre scale from RGB. It has no map and no
  planner.
* Curriculum on GOAL distance: goals within 500 m first, then growing to the whole graph
  (3.5-5 km).
* "Early rewards" within 200 m of a goal whose radius is 100 m.
* 2-8B steps.
* The curriculum shrank the goals, not the map: the user's rule "simplify goals, not maps".

### 3.4 Reference-line and checkpoint progress in racing

| agent | what progress is measured along | where the line comes from | what the policy sees of it |
|---|---|---|---|
| GT Sophy | course progress | the track | 60 points per edge, ~6 s ahead [LIT] |
| GT7 vision | p_t - p_{t-1}, the projection on the centreline | the track | the critic sees 177 edge points; the actor sees none |
| Linesight | virtual checkpoints every 0.5 m | any replay, "usually the centerline"; it "does not need to be fast" | 40 points 10 m apart, in the ego frame [HIER] 25 |
| TMRL / PedroAI | points of a recorded trajectory / speed along it | a human drive [unverified] | LIDAR |
| Mario Kart Wii (BTR) | 100 checkpoints per lap, +1 each | the course | pixels |
| Swift | distance to the next gate's centre | the gate order | the next gate's 4 corners |
| Song et al. 2021 [PE] F1 | progress along straight gate-centre segments that cannot be flown | the gate order | - |
| Sonic | maximum horizontal offset | the level's x axis | pixels |

**Rules that transfer.**
1. **The line supplies the ORDERING, not the path.** Linesight's line need not be fast.
   Song 2021 came within 5.2% of time-optimal on segments that cannot be flown. Our xAUTO
   showed the same.
2. **Never TRACK the line.** Song 2023: tracking succeeded 44% on the nominal model and 0%
   on a realistic one; progress succeeded 100% [PE] F1.
3. **Mask progress off the course** (Sophy).
4. **Stop on a failed index advance**, not on a distance epsilon (Linesight: 2 s).
5. **A line through the void pays the agent for dying.** Two routes on blue200 (ours):
   * efPLN200: its start was bridged through open air; 0/9 through 504M;
   * efTGT200: it rides the ramps; 6/9 at 202M.

   The line's ORDERING must be flyable.

### 3.5 Topological and episodic memory

* SPTM, SoRB, SGM and TTGS: [HIER] 12, [DET] 3. NGU and Agent57: [DET] 2.3.
* New here: Active Neural SLAM (2.3) and StreetLearn (3.3).
* In games, the memory that helped exploration was an explicit external structure: a map, a
  graph, an archive. The recurrent agents (Five, FTW, GT7) used their memory for partial
  observability.

### 3.6 Return-then-explore

* Go-Explore (3.2), policy-based Go-Explore, LGE and PEG: [DET] 2.4 and [NAV] 4.7-4.10.
* Our edge archive is the restore-based version. Its explore step is the executor's
  options, where Go-Explore used "95% action repeat". A 2 s committed option is a better
  explore step for a 25 Hz policy. That is the planner work's one confirmed contribution to
  exploration.

### 3.7 What turns a discovered-but-fragile trajectory into a reliable policy

This is the synthesis. Each point has the evidence that backs it.

1. **Train with RL from the trajectory's states. Do not clone its actions.**
   * The backward algorithm, Go-Explore's phase 2 and DeepMimic's RSI all do RL from
     states along the trajectory.
   * Cloning a fragile line inherits its fragility: DAgger's compounding error; A* Mario
     cleared 0/98 levels under noise [LIT] 7.
2. **Train on the landing cloud, not only on the exact states.**
   * Swift uses bounded perturbation around observed passing states. GT7 uses random start
     points and randomised car specs.
   * DART (Laskey, Lee, Fox, Dragan, Goldberg, CoRL 2017, arXiv:1703.09327) injects noise
     into the supervisor, sized "to approximate the error of the robot's trained policy".
     It improved on behaviour cloning by ~62% in grasping and was up to 3x faster than
     DAgger.
   * Ours: the archive's best chains pass through states where the two hard turns succeed
     88-100% of the time; the states a real flight reaches are not those (2026-09-27
     02:41).
3. **Use several diverse trajectories, not one.**
   * Go-Explore: robustification succeeded 40% of the time from one demonstration and in
     5/5 runs from ten; 99% from diverse robotics trajectories.
   * One chain is one tube. The archive can supply many.
4. **Pay progress along the trajectory's ORDERING. Never track it** (3.4).
5. **Train under the noise you will be judged under.**
   * Salimans & Chen trained without sticky actions and fell to 10,000 under them.
     Go-Explore trained with them and held.
   * Here the verdict is greedy, in a deterministic simulator. So the noise that matters is
     the policy's own imprecision, plus the per-card render differences recorded in
     CLAUDE.md.
6. **Build funnels.** Each segment's region of success must contain the previous segment's
   landing cloud. This is sequential composition (Burridge, Rizzi, Koditschek, IJRR
   18(6):534-555, 1999) and LQR-trees (Tedrake et al., IJRR 29(8):1038-1052, 2010)
   [classic; not re-read].
   * Measure it as edge survival from REACHED states, not from exact states.
   * The archive's closed-loop runs measured this (ours, 2026-09-27 02:41): at the corner,
     the viable share of the best move is 0.0-0.6.
7. **The chaining law.** A plan of n edges succeeds with probability p^n: 0.85^6.05 = 37%
   predicted vs 38% observed in PRM-RL [HIER] 13.
   * With ten edges at 0.9 the route succeeds 35% of the time.
   * Each hard edge must therefore be pushed close to 1, and that needs targeted practice:
     PLR, Necto's difficulty weighting, Florensa's 0.1-0.9 band.
8. **Consolidate from the true start at zero temperature.** Ours: celestial was finished
   from scratch by a window of the policy's own pre-fork states, a key temperature, and
   then a T=0 consolidation (gsCELunstuck4, 43.39 s greedy).
   * That lineage is not in CLAUDE.md section 0's contaminated list.
   * The cannonball analogue (gbCAN*) IS in that list, and is not cited here.
9. **Iterate.** Rebuild the line from the policy's own best runs (Linesight; our xSELF,
   47/102 finishes on cannonball from a self line). Search again with the better policy as
   the move operator (ExIt [HIER] 19).

---

## 4. Q4: Why some tasks are easy for humans and hard for RL

**Dubey, Agrawal, Pathak, Griffiths, Efros, "Investigating Human Priors for Playing Video
Games", ICML 2018, arXiv:1802.10217.** The authors re-rendered one platformer with the
human priors masked, one at a time. Human performance (time, deaths, states visited):

| version | time (min) | deaths | states |
|---|---|---|---|
| original | 1.8 | 3.3 | 3,011 |
| masked semantics | 4.3 | 11.1 | 7,205 |
| masked affordances | 4.7 | 10.7 | 7,031 |
| masked objects + distractors | 7.7 | 20.2 | 12,232 |
| all object priors masked | 20 | 40 | ~27,000 |

* A curiosity-driven RL agent was roughly unaffected by the semantic and affordance masks.
  It needed about twice the interactions only when visual similarity was masked.
* The authors rank the priors: the notion of objects first, then visual similarity, then
  semantics and affordances.
* With every prior masked, human exploration went from systematic to nearly random.

**Tsividis et al., "Human-Level Reinforcement Learning through Theory-Based Modeling,
Exploration, and Planning" (EMPA), arXiv:2107.12544, 2021.** An agent with object, agent and
physics priors, Bayesian model learning, directed exploration and planning "closely matches
human learning efficiency" on 90 Atari-style games. The comparison numbers against DDQN
were not read.

**Humans practise surf with a state restore.**
* Surf servers have a practice mode: `!saveloc` / `!tele` / `!teleprev` save a location and
  teleport back to it (SurfTimer; cs2-surf.com; ksf.surf).
* A human who learns a hard section restores to just before it and repeats it. That is
  Go-Explore's return step plus the backward algorithm's segment practice.
* Humans also watch world-record demos, which rule 0 forbids to the agent.

**Where RL is already better than humans: control.**
* FTW: reaction 258 ms vs 559 ms, accuracy 80% vs 48%.
* Sophy: a lap-time spread of 0.061 s over 200 laps [LIT].
* AlphaStar: extra actions per minute HURT, because micro-tactics crowd out strategy.
* The gap is macro, not micro. [NAV] makes the same point for our own agent: "superhuman
  when the path is defined".

**What humans bring to a surf map, and what can be supplied generically:**

| what humans bring | can it be supplied generically? | how |
|---|---|---|
| They SEE the map: ramps and stage layout at long range | partly | the potential channel today; ego-frame route points (Sophy-style) once a route exists |
| A prior that slanted walls are rideable | yes | the BSP-derived ride graph, the energy bound (planner-design.md 5): map geometry, no constant read off a map |
| Systematic exploration, with memory of what was tried | yes | the archive; episodic novelty |
| Practice with savelocs | yes, and stronger than a human's | exact state restore, already in the trainer |
| Commitment: no dithering on keys | yes | `--keys-hold`, latches, options |
| Watching other players' demos | no | forbidden (rule 0) |

---

## 5. What this means for RL_Surf

### 5.1 The reframing

"Planner or no planner" is the wrong axis. The right question is **where the route comes
from**.
* In every real-time game system that reached top human level, the route came from the
  track, from dense shaping, from human data, or from search (section 0).
* Our rules remove dense map-specific shaping and human data.
* A surf map has no centreline.
* That leaves search, and our search works: the edge archive found blue200's detour from
  the map start in 2-8 s on six of six seeds, where 2B steps of the recipe never did
  (ours).

The remaining failure is the same problem as the racing agents' micro problem: flying a
known route reliably. The ledger measured it directly:
* the found chain's choice sequence, replayed open-loop from the start, survives 0-1 of 32
  times;
* the archive used as a closed-loop planner finishes 0-2/9;
* the contingent archive finishes 1/18;
* all three are limited by the executor at two turns (2026-09-27 02:41 and 03:16).

### 5.2 Is an explicit learned planner justified?

**For single-map macro discovery: no, on current evidence.**
* In games, a high level works when its vocabulary is supplied (2.3).
* Where a high level is learned from scratch, it helps modestly, never on a deceptive
  route.
* Nachum 2019 measured that the benefit is exploration, and a flat agent with the same
  exploration matches it.
* Our learned planners learned the deception at 2 s granularity (2.5).
* The archive does the macro exploration better and cheaper.

**What the planner work leaves that IS justified:**
* **The executor's options as the archive's move operator** (3.6). They are the reason the
  archive finds routes in seconds.
* **A plan channel that conditions the policy on a whole route.** In ours, srFT200 (the
  route through the fan) converged faster than efTGT200 (the route as a reward only):

  | step | srFT200 training finishes | efTGT200 training finishes |
  |---|---|---|
  | 175M | 94% | 61% |

  | | srFT200 | efTGT200 |
  |---|---|---|
  | greedy at 202M | 9/9 | 6/9 |

  AlphaStar's z and DTSIL are the published forms. The route is chosen once per episode
  from a library (here, the archive) and is not re-planned every 2 s.
* **Amortised search, for multi-map generalisation.** A learned proposal prior trained on
  the archive's own choices is how AlphaZero and ExIt amortise search. It is a later step.
  Its value appears across maps (the transfer argument, [PE] F8), not on one map.

### 5.3 Ranked recommendation

The constants are set once, in seconds or as dimensionless ratios, and carried unchanged to
every map (CLAUDE.md 0b).

**Rank 1. Go-Explore with a racing-style robustify phase.**

*Pipeline.*
* (a) **Find routes.** Run the edge archive from the map start (count-only selection
  1/sqrt(1 + n); the executor's options as moves; exact restores). Keep running after the
  first finish until K distinct finishing chains exist.
* (b) **Make the line.** Take each chain's FLOWN per-tick path, which is flyable by
  construction, as a self-derived line (SELF_STATES=1). Its provenance is the archive seed
  and the executor checkpoint.
* (c) **Train a flat policy on it:**
  * the recipe, with `--race-arc` progress along the line, masked off the corridor;
  * spawns spread uniformly along the chain in WINDOWS, plus bounded perturbations of the
    chain's states (Swift);
  * a map-start share;
  * optionally, the line passed through the fan: rank 1b.
* (d) **Consolidate** at T=0 with a large map-start share.
* (e) **Iterate.** Rebuild the line from the policy's own best runs (the xSELF / Linesight
  loop). Re-run the archive with the new policy as the move operator.

*Why rank 1.*
* It is the only combination backed at once by:
  * the exploration literature (Go-Explore, Salimans-Chen, DTSIL);
  * the whole racing genre (1.7);
  * our own measurements:
    * efTGT200 / srFT200: a feasible route gave 6/9 and 9/9 at 202M;
    * xSELF: a self line gave 47/102 finishes on cannonball;
    * the edge archive: routes found in seconds.
* It needs no new learning machinery. Every piece exists: `tools/edge_archive.py`,
  `--race-arc`, `--demo-file` windows with SELF_STATES, `--goal-obs fan`, T=0
  consolidation.

*Why the earlier chain arm failed.* arch2_b200 (2026-09-27 02:41) spawned on the chain but
kept the planner recipe's refund_i reward. Refund_i pays nothing from states that never
finish, so the first half of the route, where no episode finished, got no signal. Rank 1
changes the reward to progress along the chain. That is exactly what efTGT200 had.

*Constants that transfer, with their sources:*

| constant | value | source |
|---|---|---|
| archive selection weight | 1/sqrt(1 + n) | Go-Explore |
| number of finishing chains to robustify | K = 10 (40% success from 1 demo vs 5/5 from 10) | Go-Explore |
| spawns | a window {tau-D..tau}, never a point | Salimans & Chen |
| backward schedule, if used | advance at >= 20% success | Salimans & Chen |
| start set | random position on the route + bounded perturbation of observed passing states; random start points on the track | Swift; GT7 |
| hard-edge weighting (rank 2) | keep starts in the 0.1-0.9 success band; reserve 1/3 for mastered regions | Florensa |
| progress | along the route's ordering, masked off the corridor | Sophy, Linesight, GT7 |
| tracking term | none | Song 2023 |
| early stop | no index advance within ~2 s | Linesight |
| finish bonus | none needed (neither Sophy nor Linesight has one); ours may stay | Sophy, Linesight |
| discount horizon | grow it once the route is flown (60 s -> 840 s) | OpenAI Five |
| our 20 s horizon | a fair start; [LIT] says do not shorten it | [LIT] |

**Rank 1b. Condition the policy on the chosen route.** Pass the whole route to the policy
through the fan (srFT200's mode, every plan to the finish). Train it both conditioned and
unconditioned, with the following pay active part of the time: AlphaStar's 25% and its 10%
zeroing. This is the AlphaStar-z / DTSIL form of "planning". At test time the archive
chooses the route once, from the start, in seconds.

**Rank 2. Targeted edge practice** (Codex's step 2; inside rank 1's start distribution).
* Spawn from the REAL landing clouds just upstream of the edges whose survival from reached
  states is low.
* Weight those starts by low mastery x high downstream solvability. This is PLR and Necto's
  difficulty weighting, measured with the contingent archive's own statistics.
* Size the perturbation to the policy's own error, as DART does.
* It fixes exactly what the contingent archive diagnosed. It does not need a planner.
  Freezing the planner during this phase is fine, but rank 1 does not need the planner at
  all.

**Rank 3. Decision-time search on top, after ranks 1-2.**
* MCTS or the contingent archive over the robustified policy. Distil it back (ExIt; Guo et
  al., NIPS 2014: UCT distilled into a real-time CNN, with DAgger-style interleaving, beat
  DQN).
* The ceiling is the executor. Ours: blue100 4/9 -> 8/9; blue200 1/9-2/18 and 1/18.
* Use the determinism, as The Brute does: run the search with a GREEDY executor, or record
  the sampling seed of every edge, so that a found chain replays exactly. Then an open-loop
  replay from the start tells whether the route itself is solved, separately from whether a
  policy can fly it.
* If the executor stays stochastic, treat its outcomes as chance nodes. The contingent
  archive already does, as Stochastic MuZero does (Antonoglou et al., ICLR 2022
  [not re-read]).

**Rank 4. Environment randomisation over a map pool.** This is for the multi-map recipe,
not for one map.
* OpenAI Five App. O.2.
* Sonic: joint pretraining gave 3,128 vs PPO's 1,489 on unseen levels.
* Ours: the ladder, lab100 9/9 from 303M; lab200 intermittent.
* A brand-new map has no easier siblings, so this is the recipe's transfer mechanism, not
  its exploration mechanism.

**Rank 5. A learned planner that re-plans every ~2 s, or a two-timescale recurrent
core.**
* The planner: see 5.2.
* FTW's two timescales bought a tactic in a partially observed game with dense learned
  rewards. There is no evidence that they discover routes, and our GRU was null.
* Revisit memory only if partial observability becomes the bottleneck. For example: a
  multi-storey map where the route leaves the camera's view before the agent commits (Robot
  Parkour Learning's MLP-without-GRU result, [HIER] 11).

### 5.4 First experiments

Each is pre-registered and local-first, and each uses the same flags on every map. Every
verdict is greedy finishes from the true map start, step-matched, one seed. The gate-ladder
caveat in CLAUDE.md applies.

**E1 (decisive, cheap). blue200: efTGT200's flags, with the tight route replaced by the
archive chain's flown path.**
* Line: seed 5 or seed 3 of the 2026-09-27 archive runs; SELF_STATES=1.
* Spawns: 50% in windows along the chain, 50% at the map start.
* Reference: efTGT200, 6/9 at 202M and 7/7 playable spawns at 706M.
* **Pass:** >= 6/9 by 300M.
* **If it fails where efTGT200 passed**, the chain's shape is the defect: lucky edges, or
  air segments like efPLN200's. Then run E3 before anything else.

**E2. E1 plus the route through the fan (rank 1b).** Reference: srFT200, 9/9 at 202M.

**E3. Several chains.**
* Use the union of the six seeds' chains as windows, with the line taken from the chain
  whose edges survive best from reached states.
* Measure the survival at the two turns from REACHED states, before and after training.
  Robustification should raise it toward the 88-100% the archive's best states have.

**E4. Generality: the archive alone, same constants, from the true start.** Maps: blue025,
blue050, blue100, the labyrinths, celestial, cannonball and **unitfarmer2**. Report the time
to the first finish and the number of distinct finishing chains.
* unitfarmer2 decides whether the exploration half is generic. A velocity-keyed archive
  should reach the pit line if the move operator can produce the dive.
* If it cannot, the limit is the move operator (the three rays), not the method. Give the
  archive a richer, still generic, move set: rays at several angles and pitches, or the
  flat recipe policy sampled at temperature.
* Use a move operator trained without the target map (step 1's executor, trained on random
  primitives), so the recipe does not depend on the map it is solving.

**E5. Iterate on the best of E1-E3.** Rebuild the line from the policy's own finishes
(`tools/pick_selfline.py` trim rule), re-run the archive with it as the move operator, and
compare finish times.

### 5.5 Where the evidence is weak, or a claim could not be verified

* **Secondary or unread sources:**
  * FTW's 15 Hz and 84x84 come from Pearce & Zhu's description, not from the FTW paper.
  * Swift's control rate: NOT FOUND.
  * DreamerV3's results differ between arXiv v1 and the later version; the Nature text was
    not re-read.
  * The TrackMania numbers other than Linesight's come from a secondary blog.
  * JueWu-MC's score, the Sonic contest results (the blog returned 403), the LLM Minecraft
    planners and the numbers in "Flattening Hierarchies" were not read.
  * The Mountain Car, LQR-tree, sequential-composition and Stochastic MuZero statements are
    classic results cited from memory.
* **One seed.** Every in-project number is one seed. The 2.7x spread between identical
  configurations (CLAUDE.md, the gate-ladder retraction) applies to any comparison of two
  arms.
* **The archive has only been shown on one map** (blue200, a ~20 s route).
  * Its keys (128 u cells, 6 speed bins, 9 azimuth bins) were set there. They must be
    frozen and carried unchanged; LGE's learned density is the constant-free alternative
    [DET].
  * Its scaling to cannonball-sized maps and 60-80 s routes is untested. For scale,
    Go-Explore's Montezuma phase 1 took 57.6-640M frames, 5-55 minutes at our speed.
  * Its move operator on blue200 was an executor trained on blue200 by the planner recipe.
    The generic version needs a move operator trained elsewhere (E4).
* **Nothing in the literature solves our exact setting** (a deceptive momentum route in a
  real-time 3D game, from scratch, with no demos, no dense aligned shaping and no state
  restore). Every success used one of the four removals in section 0. Rank 1 extrapolates
  Go-Explore plus racing practice to our setting. It is the extrapolation with the most
  support, not a demonstrated result.
* **The strongest counter-evidence to rank 1 is ours.** The chain's first half got 0%
  finishes in arch2_b200 even from the exact chain states, although those edges survive
  84-100% from the exact states. Section 5.3 attributes that failure to the reward. If E1
  also fails, the attribution is wrong, and the executor's reliability, not the reward, is
  the binding constraint.

---

## References

Read for this file unless marked [pointer] (already in the named survey) or
[not re-read].

**Games and game agents**
* Berner et al. (OpenAI), "Dota 2 with Large Scale Deep Reinforcement Learning",
  arXiv:1912.06680, 2019. [PDF in scratchpad; the ledger read it in full on 2026-09-23]
* Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement
  learning", Nature 575:350-354, 2019 (DeepMind unformatted PDF).
* Jaderberg et al., "Human-level performance in 3D multiplayer games with population-based
  reinforcement learning", Science 364(6443):859-865, 2019; arXiv:1807.01281.
* Stooke, Dalibard, Jayakumar, Czarnecki, Jaderberg, "Perception-Prediction-Reaction Agents
  for Deep Reinforcement Learning", arXiv:2006.15223, 2020. [abstract]
* Pearce & Zhu, "Counter-Strike Deathmatch with Large-Scale Behavioural Cloning",
  arXiv:2104.04258 (NeurIPS 2021 Offline RL workshop; IEEE CoG 2022).
* Durst et al., "Learning to Move Like Professional Counter-Strike Players", SCA 2024,
  arXiv:2408.13934. [search summary]
* Wang, Ustun, McGroarty, "A Data-Driven Discretized CS:GO Simulation Environment to
  Facilitate Strategic Multi-Agent Planning Research" (DECOY), WSC 2025, arXiv:2509.06355.
  [abstract]
* Lample & Chaplot, "Playing FPS Games with Deep Reinforcement Learning", AAAI 2017,
  arXiv:1609.05521.
* Dosovitskiy & Koltun, "Learning to Act by Predicting the Future", ICLR 2017,
  arXiv:1611.01779.
* Wurman et al., "Outracing champion Gran Turismo drivers with deep reinforcement
  learning", Nature 602:223-228, 2022. [pointer: LIT 1]
* Lee, Seno, Tai, Subramanian, Kawamoto, Stone, Wurman, "A Champion-level Vision-based
  Reinforcement Learning Agent for Competitive Racing in Gran Turismo 7", RA-L 2025,
  arXiv:2504.09021.
* Linesight (Trackmania), github.com/Linesight-RL/linesight and
  linesight-rl.github.io/linesight (docs, "custom training"). [pointer: LIT 3, TIME 1.4]
* "Trackmania I - The History of Machine Learning in Trackmania", hallofdreams.org
  (secondary). [unverified numbers]
* Clark, Towers, Evers, Hare, "Beyond The Rainbow: High Performance Deep Reinforcement
  Learning on a Desktop PC", ICML 2025, arXiv:2411.03820.
* Kaufmann et al., "Champion-level drone racing using deep reinforcement learning", Nature
  620:982-987, 2023 (PMC10468397).
* Baker et al., "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online
  Videos", NeurIPS 2022, arXiv:2206.11795.
* Hafner, Pasukonis, Ba, Lillicrap, "Mastering Diverse Domains through World Models"
  (DreamerV3), arXiv:2301.04104 (v1 and v2 read); Nature 2025 [not re-read].
* Hafner, Lee, Fischer, Abbeel, "Deep Hierarchical Planning from Pixels" (Director),
  NeurIPS 2022, arXiv:2206.04114.
* Milani et al., "Retrospective Analysis of the 2019 MineRL Competition on Sample Efficient
  Reinforcement Learning", arXiv:2003.05012.
* Lin et al., "JueWu-MC: Playing Minecraft with Sample-efficient Hierarchical Reinforcement
  Learning", arXiv:2112.04907. [abstract]
* Nichol, Pfau, Hesse, Klimov, Schulman, "Gotta Learn Fast: A New Benchmark for
  Generalization in RL", arXiv:1804.03720, 2018.
* Firoiu, Whitney, Tenenbaum, "Beating the World's Best at Super Smash Bros. with Deep
  Reinforcement Learning", arXiv:1702.06230, 2017. [abstract]
* Pang et al., "On Reinforcement Learning for Full-length Game of StarCraft", AAAI 2019,
  arXiv:1809.09095. [abstract]
* Sun et al., "TStarBots: Defeating the Cheating Level Builtin AI in StarCraft II in the
  Full Game", arXiv:1809.07193. [abstract]
* Wu et al., "Hierarchical Macro Strategy Model for MOBA Game AI", AAAI 2019,
  arXiv:1812.07887. [abstract]
* Xu et al., "Agents Play Thousands of 3D Video Games" (PORTAL), arXiv:2503.13356, 2025.
  [abstract; LLM-generated behaviour trees, noted only]

**Hierarchy, navigation and memory**
* Chaplot, Gandhi, Gupta, Gupta, Salakhutdinov, "Learning to Explore using Active Neural
  SLAM", ICLR 2020, arXiv:2004.05155.
* Mirowski et al., "Learning to Navigate in Cities Without a Map", NeurIPS 2018,
  arXiv:1804.00168.
* Zhou & Kao, "Flattening Hierarchies with Policy Bootstrapping", NeurIPS 2025,
  arXiv:2505.14975. [abstract]
* Nachum et al., "Why Does Hierarchy (Sometimes) Work So Well in Reinforcement Learning?",
  arXiv:1909.10618, 2019. [pointer: HIER 7]
* Thakkar et al., "Hierarchical Control for Head-to-Head Autonomous Racing",
  arXiv:2202.12861, 2022. [pointer: HIER 8]
* FeUdal, HIRO, HAC, h-DQN, option-critic, SoRB, SPTM, SGM, TTGS, PRM-RL, RoGuE, ExIt.
  [pointers: HIER]

**Exploration and robustification**
* Ecoffet, Huizinga, Lehman, Stanley, Clune, "Go-Explore: a New Approach for
  Hard-Exploration Problems", arXiv:1901.10995, 2019.
* Ecoffet, Huizinga, Lehman, Stanley, Clune, "First return, then explore", Nature
  590:580-586, 2021; arXiv:2004.12919.
* Lu, Georgescu, Verwey, "Go-Explore Complex 3D Game Environments for Automated
  Reachability Testing", arXiv:2209.00570, 2022. [abstract]
* Salimans & Chen, "Learning Montezuma's Revenge from a Single Demonstration",
  arXiv:1812.03381, 2018.
* Guo et al., "Memory Based Trajectory-conditioned Policies for Learning from Sparse
  Rewards" (DTSIL), NeurIPS 2020, arXiv:1907.10247.
* Machado, Bellemare, Talvitie, Veness, Hausknecht, Bowling, "Revisiting the Arcade
  Learning Environment", JAIR 61:523-562, 2018; arXiv:1709.06009.
* Peng, Abbeel, Levine, van de Panne, "DeepMimic", ACM TOG 37(4) (SIGGRAPH 2018),
  arXiv:1804.02717.
* Laskey, Lee, Fox, Dragan, Goldberg, "DART: Noise Injection for Robust Imitation
  Learning", CoRL 2017, arXiv:1703.09327. [abstract]
* Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning" (DAgger), AISTATS 2011. [not re-read]
* Guo, Singh, Lee, Lewis, Wang, "Deep Learning for Real-Time Atari Game Play Using Offline
  Monte-Carlo Tree Search Planning", NIPS 2014. [abstract]
* Burridge, Rizzi, Koditschek, "Sequential Composition of Dynamically Dexterous Robot
  Behaviors", IJRR 18(6), 1999; Tedrake, Manchester, Tobenkin, Roberts, "LQR-trees",
  IJRR 29(8), 2010. [not re-read]
* Antonoglou, Schrittwieser, Ozair, Hubert, Silver, "Planning in Stochastic Environments
  with a Learned Model" (Stochastic MuZero), ICLR 2022. [not re-read]

**Human priors**
* Dubey, Agrawal, Pathak, Griffiths, Efros, "Investigating Human Priors for Playing Video
  Games", ICML 2018, arXiv:1802.10217.
* Tsividis et al., "Human-Level Reinforcement Learning through Theory-Based Modeling,
  Exploration, and Planning", arXiv:2107.12544, 2021. [abstract]
* SurfTimer practice commands (`!saveloc`, `!tele`): cs2-surf.com (article 7), ksf.surf
  (commands). [domain source]

**In-project (ours), docs/research-results.md**
* 2026-09-23 03:20 and 04:00: OpenAI Five read; the ladder (lab100 9/9), the GRU null.
* 2026-09-24 19:35 and 19:45: efPLN200 0/9, efTGT200 6/9 at 202M, srFT200 9/9 at 202M.
* 2026-09-25 21:52: joint training equals the no-plan baseline.
* 2026-09-26 12:45: planners learn "turn toward the goal" on blue200's corridor.
* 2026-09-26 18:33 and 18:58: one recipe + search finishes all four edgeflow maps (blue200
  1/9, 2/18).
* 2026-09-27 01:51: 3-choice verdicts; planning every 0.5 s worse than 2 s.
* 2026-09-27 02:01, 02:41 and 03:16: the edge archive (2-8 s, 6/6 seeds), open-loop
  replay 0-1/32, closed-loop 0-2/9, arch2_b200 0/9, the contingent archive 1/18.
* CLAUDE.md section 3: xSELF 47/102 finishes on cannonball from a self line.
* The celestial finish from scratch, gsCELunstuck4 (the memory file's round-30 log, and the
  ledger 8f48a1d).
