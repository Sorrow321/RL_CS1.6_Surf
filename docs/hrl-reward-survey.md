# HRL reward assignment: who is paid for what, and over which horizon

Literature survey, 2026-09-25 (web reading plus the repo; no code run, no GPU). ASCII only. How should
`--goal-planner primlearn` pay its levels (EXECUTOR: PPO at 25 Hz, arc progress along the current 2-s
motion primitive; PLANNER: PPO once per primitive, progress to the finish + finish bonus + end-cell
novelty)? The user's proposal: a BIG reward (progress) for the planner, a SMALL one (how well the
primitive was flown) for the executor, a gate so each trains only its own level, maybe min/max, and
one reward instead of two if that is sound.

Neighbours, not repeated: [PE] `docs/litsurvey-planner-executor.md` (schedules, feasibility constants),
[HIER] `docs/research-litsurvey-hier.md`, [DET] `docs/litsurvey-detour-navigation.md`. This file: which
reward each level gets, whether the low level's return crosses a subgoal boundary, how the levels are
kept from colluding. The code and ledger moved while it was written (commits 1b96bb6 `--exec-cut`,
18c4d13, 8a33b2b; ledger 2026-09-25 06:30: prim2f_b025 passed blue025 6/9 greedy with a planner that
learned a SIGNALLING CODE; section 6 answers the ledger's open question). Quotes are verbatim; "ours" =
this project's ledger or code.

## 0. The answer in seven points

1. The gate the user proposes is the oldest rule in HRL, Dayan and Hinton's REWARD HIDING (1993).
   h-DQN, HIRO, HAC, HiTS, Director, CHER and Setter-Solver all pay the low level ONLY for doing what
   it was told and the high level ONLY from the environment. FuN states the gradient gate outright:
   "No gradients are propagated between Worker and Manager". It is the standard, not a complication.
2. It is one DESIGNED reward, not two: the task reward goes to the planner; the executor's reward is
   the definition of the plan interface ("follow this curve"), identical on every map. Big vs small
   is meaningless across two networks: each normalises its own advantages.
3. The executor's return must END with its subgoal (terminal there, or bootstrapped under the SAME
   subgoal). Every published system that pays per subgoal does one of the two; ours did neither
   (failure 3). `--exec-cut` is the h-DQN / HAC / HiTS form. Keep it.
4. Never pay the planner for the executor's success. CHER, AMIGo and prim2_b025 show the collapse.
5. But DO gate the planner's credit on obedience, Dayan and Hinton's other constraint: "super-managers
   should only learn from the fruits of the honest labour of sub-managers". Without it the planner
   learns the executor's response function, not a geometry (HAC: "unrealistic subgoals"; ours:
   prim2f). The sound form is the user's min, on the PLANNER: credit = min(p, f x p).
6. No min/max on the executor: min charges it for the planner's bad plan (no detours), max lets it
   ignore plans (the flat agent again).
7. Every end an agent can reach (death, time cap, subgoal close, finish) must be priced by ONE
   potential, and a clock that ends a return must be visible (Grzes 2017, Pardo 2018). Four of the
   five reward failures on 2026-09-25 were one end priced differently from the others.

## 1. The systems

| system | low-level reward | high-level reward | low-level horizon across subgoals | collusion / feasibility guard |
|---|---|---|---|---|
| Feudal RL (Dayan, Hinton 1993) | paid for "doing their bidding whether or not this satisfies the commands of the super-managers" | the level above's reward; top level = task | control returns "when the state changes at the managerial level"; value per command | managers learn "only if they obey their managers" (honest labour); a time-out for impossible commands |
| h-DQN (Kulkarni 2016) | internal critic, +1 on reaching g; no extrinsic | extrinsic F summed while g runs | run ends at g or episode end; stores ({s,g},a,r,{s',g}): same g only | none (hand-given objects); controller pretrained first |
| FuN (Vezhnevets 2017) | A^D = R_t + alpha R^I_t - V^D, r^I = (1/c) sum_i d_cos(s_t - s_{t-i}, g_{t-i}), alpha ~ U(0,1) | environment only, via A^M grad d_cos(s_{t+c} - s_t, g_t) | CROSSES (continuing A3C, c = 10); worker discount shorter (fixed 0.95 on Atari, manager's varied; 0.99 vs 0.999 on Montezuma) | directional goals; no gradient between levels; the goal is scored by its alignment with the realised move |
| HIRO (Nachum 2018) | -\|\|s_t + g_t - s_{t+1}\|\|_2 only | sum of env reward over c = 10 | bootstraps with h(s,g,s') = s + g - s' (same target) even at the boundary; reward <= 0 | off-policy goal relabelling; none for feasibility |
| HAC (Levy 2019) | -1 per step, 0 at the goal; task only at the top | task goal (-1/0) | H actions per subgoal; gamma 0 on success; same goal otherwise; Q in [-H, 0] | subgoal testing lambda 0.3, tested miss -H with gamma 0; hindsight action transitions |
| HiTS (Guertler 2021) | 1 iff at the subgoal AT the deadline, else 0 | env reward minus c per subgoal emitted | deadline terminal ("we do not bootstrap and set gamma^0 = 0") | timed subgoals keep the SMDP stationary; tests with the mean action |
| Option-critic (Bacon 2017), Harb 2018 | task reward | task reward | continuing, shared | xi = 0.01 margin; deliberation cost eta (0.020 best of 0-0.03) |
| MLSH (Frans 2018) | task reward, master's choice as an observation | task reward over N steps | continuing | master-only warm-up (W = 20) |
| HiPPO (Li 2020), DAC (Zhang 2019) | task reward (joint PPO) | task reward | continuing | HiPPO: pretrained skills, random commitment p ~ U{5,15} |
| HAAR (Li 2019) | r_l = A_h(s^h_t, a^h_t) / k, replacing the env reward | env reward | the k-step segment's advantage shared by every step | pretrained skills; monotone-improvement argument |
| Director (Hafner 2022) | max-cosine to the goal only (variant: goal 1.0 + task 0.5) | extrinsic + exploration (goal-AE reconstruction error), separate critics, w 1.0 / 0.1 | rollouts cut into K = 8 pieces; same-goal critic beyond K | goals = codes of seen states; exploration to manager only |
| AMIGo (Campero 2021) | r^g = 1 - 0.9 t/t_max per goal reached, plus extrinsic | teacher +0.7 if reached with t+ >= t*, else -0.3 | CROSSES (new goal on every success) | t* +1 after 10 successes in a row slower than t* |
| Setter-Solver (Racaniere 2020) | 1 if achieved within the time limit, else 0 | setter losses: validity, feasibility, coverage | one goal per episode | judge-conditioned feasibility f ~ U(0,1); entropy |
| CHER (Kreidieh 2019) | -\|\|g - s'\|\| only | J_m + lambda J_w | HIRO-style | lambda 0.005-0.01, or a Lagrange multiplier |
| HRAC / HIGL / DHRL (2020-22) | goal distance | env reward + adjacency hinge (eta 20) | per subgoal | k-step adjacency; landmarks; one reachable hop ([PE] 2.3) |
| BrHPO (Luo 2024) | r_l - lambda_2 R_i (reachability ratio) | KL-regularised + lambda_1 R_i | per subtask | two-sided reachability |
| SSE (Hwang 2025) | reach reward | a transition only if the subgoal is reached; failure ends with 0 | per subgoal | stop-on-failure; failure-aware edge cost c_dist 5 |
| ours, primlearn (prim2f, 2026-09-25) | arc along the current primitive (0.0667/u, ~40 per 600 u) - 0.005/tick, +50 on the map finish | Euclidean progress / 1000 u + 10 finish + end-cell novelty; bank charged back at every non-finish end | `--exec-cut`: terminal at every re-plan | none; half of episodes open with a uniform primitive. Result: 6/9 greedy finishes, primitives completed 0.7% |

## 2. Does the low level's return cross a subgoal boundary? (question 2)

Three families, by what the low-level critic bootstraps from when a subgoal ends:
* TERMINAL at the subgoal (the return is one subgoal's pay): h-DQN (the controller runs "while not
  (s is terminal or goal g reached)" and stores "({s,g},a,r,{s',g})"); HAC ("gamma_i is set to 0 if a
  subgoal is tested and missed or if an action achieves the goal", critics bounded "to the range
  [-H,0] using a negative sigmoid"); HiTS (the deadline is terminal); Setter-Solver (one goal, 1 or 0).
* SAME-SUBGOAL bootstrap (the return runs past the switch only as if the old goal were still chased):
  HIRO stores "(s_t, g_t, a_t, r_t, s_{t+1}, h(s_t, g_t, s_{t+1}))", the same absolute target also at
  the c-step boundary (CORRECTION to "HIRO bootstraps across relabelled goals": never into the NEXT
  goal). Director: "we cut the imagined rollouts into distinct trajectories of length K within which
  the goal is constant. The state-critic estimates goal rewards beyond this horizon under the same
  goal". HAC at its H-action limit.
* CROSSING: FuN (a continuing worker; its fix is a shorter discount, "the Worker to be more greedy");
  AMIGo ("The teacher proposes a new goal ... whenever the student reaches the intrinsic goal"; the
  TEACHER's t* stops easy-goal farming, not the student's horizon); the shared-reward systems, which
  have no following reward to farm.

Farming needs three things at once: a positive pay per subgoal, a return that crosses subgoals, and an
end the agent controls. The first two families remove the crossing; HIRO and HAC also make the pay
non-positive. Our executor before `--exec-cut` had all three: 0.0667 per u x ~3 u per tick at
300 u/s = 0.2 per tick, worth 0.2 / (1 - 0.9995) = 400 as an endless stream (1,333 at 1,000 u/s),
against +50 for finishing, which ENDED the stream (prim2c_b025: every episode ran out its 20 s,
executor reward ~250 an episode). The classic case, Randlov and Alstrom (1998): "we rewarded the agent
for driving towards the goal but did not punish it for driving away from it. Consequently the agent
drove in circles with a radius of 20-50 meters around the starting point."

The terminal form has one condition: a subgoal that ends by DEADLINE is terminal only if the deadline
is visible. Pardo et al. 2018: "A notion of the remaining time should be included as part of the
agent's input to avoid violation of the Markov property". HiTS gives its lower level the time left.
From the fan's description (8 look-ahead points by time horizon) our executor sees how much CURVE is
left, not how much of its 1.5x time BUDGET, so its primitive time-outs are aliased.

## 3. Is the low level given any task reward? (question 3)

* None: Feudal RL, h-DQN, HIRO, HAC, HiTS, Director by default ("The worker maximizes only the goal
  rewards", and it still solves egocentric Ant mazes, DMLab, Crafter, Atari), CHER, Setter-Solver.
* All of it, plus a following term: FuN, A^D_t = R_t + alpha R^I_t - V^D_t. CORRECTION to the
  commission's summary: alpha weighs the INTRINSIC reward; the worker gets the full extrinsic return.
  AMIGo: r_t = r^g_t + r^e_t.
* A share: Director's variant (goal 1.0 + task 0.5) "completely closes the gap to the
  state-of-the-art DreamerV2 agent" on standard dense benchmarks; still, "providing task reward to the
  worker is not as important as expected".
* Only the task (nothing to follow): option-critic, MLSH ("We treat the master policy as an extension
  of the environment"), HiPPO, DAC. Derived from above: HAAR, r_l = [r^h_t + gamma_h V_h(s^h_{t+k}) -
  V_h(s^h_t)] / k.

What a task term does: the worker can "fill in low-level details" the manager is too coarse to give
(Director), and it can overrule the manager. Dayan and Hinton name the price and pay it on purpose:
"if a sub-manager only knew the super-task of its super-manager then it could bypass its manager with
advantage". Director's reading of FuN: both levels "maximize task reward and the lower level is
regularized by a goal reward but did not demonstrate clear benefits over an LSTM baseline". FuN's own
ablation includes a non-feudal variant (goals trained through the worker, no intrinsic reward, so task
reward only): "although alternatives do work to some degree their performance is significantly
inferior". For a planner that exists to force commitments the potential field does not show
(CLAUDE.md 0c), the override IS the failure: an executor paid for task progress refuses the detour.

## 4. Combining a task reward and a following reward (question 4)

| form | where | effect |
|---|---|---|
| additive, worker gets both | FuN, AMIGo, Director variant | obedience traded for task reward at a fixed rate |
| high-level advantage as the low reward | HAAR | the low level optimises the task inside a skill; no plan to obey |
| AND gate inside the low reward | HiTS (right place AND time); Meyer 2020 [PE] (speed along the path x cross-track kernel) | following pays only on the line; cannot be farmed off it |
| gate on the HIGH level's credit | Dayan-Hinton (honest labour); SSE (failure ends the segment with 0); PDM-Closed and Plan-R1 (safety gates x weighted progress) [PE] | a failed or unfollowed plan earns nothing, whatever it promised |
| goal scored by alignment with the realised move | FuN, grad = A^M grad d_cos(s_{t+c} - s_t, g_t) | a goal is pushed toward what actually happened, so it keeps its geometric meaning |
| constraint with a multiplier | CHER (lambda as a Lagrangian on the worker's return); BrHPO (lambda_1 manager, lambda_2 worker) | feasibility pressure that adapts, not a fixed bonus |
| min or max of the two | none found in the HRL literature searched | - |

The closest published statement of "the executor is paid for following, the planner for progress, and
neither is charged for the other's mistake" is three HAC/Feudal pieces together: reward hiding
(executor side); hindsight ACTION transitions, which replace the proposed subgoal with the state
reached so the upper level trains "as if the optimal lower level policy hierarchy had been used"
([HIER] 4); and subgoal TESTING, which charges -H only when a NOISE-FREE lower level misses (the plan
was infeasible, not badly flown). Charging every miss instead made agents "output overly conservative
subgoals"; dropping the tests, "levels would always learn to set unrealistic subgoals that could not
be achieved within H actions".

## 5. Keeping the high level from easy and from impossible subgoals (question 5)

Collusion (useless plans the executor always completes):
* CHER, J_m' = J_m + lambda J_w: "excessive cooperation may disincentivize an agent from making
  forward progress ... assigning goals that match the worker's current state". Working lambda
  0.005-0.01, or a multiplier enforcing J_w >= delta (delta 25-75% of the worker's return).
* AMIGo: a teacher paid for success "is rewarded for all easy-to-reach states, even late in the
  training process"; hence r^T = +0.7 if t+ >= t*, -0.3 otherwise.
* Setter-Solver: coverage E[log p(S(z,f))] and feasibility (J(S(z,f)) - sigma^-1(f))^2; "all three of
  our setter losses are necessary in complex environments".
* Asymmetric self-play: R_A = gamma max(0, t_B - t_A), R_B = -gamma t_B; "Alice's optimal behavior is
  to find the simplest tasks that Bob cannot complete". GoalGAN: goals with success in [0.1, 0.9].
Infeasibility, and its twin the signalling code: HAC and HiTS testing (noise-free lower level only),
HRAC's adjacency hinge, BrHPO, SSE, DHRL ([PE] 2.2-2.3); S3 (Srivastava, Jerath, arXiv:2607.19232,
July 2026, abstract only) pays the high level for low coarse-dynamics uncertainty, reporting
"risk-averse subgoal selection". Paying the high level for realised outcomes ALONE is not a guard:
Director gets away with it ("often learns to choose the most distant goals that the worker is able to
achieve") because its max-cosine reward has no dead zone, so even an unreachable goal still dictates
the worker's motion (toward it). Our arc reward is corridor-gated: once an unflyable primitive leaves
the 384-u corridor nothing constrains the executor, its habit becomes the message (section 6).
Shared-reward degeneracy: without a switching cost "the options eventually learn to terminate at every
step" (Harb 2018); MLSH without its warm-up sends both sub-policies to "the midpoint" ([PE] 2.1).

## 6. The user's proposal, answered

IS THE SEPARATED GATE STANDARD? Yes, verbatim from 1993: "Managers must reward sub-managers for doing
their bidding whether or not this satisfies the commands of the super-managers ... if a sub-manager
achieves the sub-goal it is given it is rewarded, even if this does not lead to satisfaction of the
manager's own goal. This allows the sub-manager to learn to achieve sub-goals even when the manager
was mistaken in setting these sub-goals." The same paper cites our architecture from 1992: Jameson's
high level gave "direct commands (like reference trajectories) to a low level agent - which learned to
obey it based on reinforcement proportional to the square trajectory error". The gradient gate needs
no mechanism: separate networks and losses, the plan reaches the executor as a sampled input.

IS THERE A SOUND SINGLE REWARD? Three published forms, each sound for its purpose and wrong for ours:
the task reward to both levels (option-critic, MLSH, HiPPO, DAC: the plan becomes a latent the executor
may ignore, and it needs deliberation costs, warm-ups or pretrained skills not to degenerate); HAAR's
high-level advantage as the low reward (the executor optimises the task, not the plan); and the plan
as potential-based advice on the task reward, F = gamma Phi_plan(s') - Phi_plan(s), which is
policy-invariant (Ng, Harada, Russell 1999), so the executor's optimum ignores the plan. All three
dissolve the commitment the planner exists to impose. The honest reading of "one reward": ONE designed
reward (the planner's); the executor's is the plan contract, generic on every map. PPO normalises each
level's advantages, so only ratios WITHIN a level matter (progress vs finish vs novelty; arc vs time).

MIN/MAX. On the executor both break reward hiding: min(follow, progress) charges a perfectly flown
backward plan and teaches refusal of the detours macro exploration needs; max(follow, progress) pays it
to leave any plan worse than the potential gradient, and hides the planner's bad plans from the planner.
On the PLANNER a min is the sound way to write the gate the literature prescribes (next paragraph).

THE SIGNALLING CODE (ledger 06:30) AND "SHOULD FOLLOWABILITY ENTER THE PLANNER'S OBJECTIVE?" prim2f
passed blue025, but its primitives complete 0.7% (arc 30.7%) while realising +167 u each against +65
planned: steep climbs with a hard left that the executor answers by surfing forward. Both halves were
predicted. Dayan and Hinton: sub-managers "need not initially understand their managers' commands"; a
command means whatever response it elicits, which is why their managers update "only if they obey their
managers". HAC without tests: "unrealistic subgoals". FuN's transition gradient scores a goal by its
cosine with the REALISED move, which ties its meaning to geometry. Our planner's credit ignores
obedience, and the executor's off-plan behaviour is unconstrained (no reward outside the corridor)
or paid (the +50 finish bonus), so the code is, as the ledger says, a legitimate optimum. The answer
is yes, as a GATE and never as a bonus: credit_k = min(p_k, f_k x p_k), p_k the realised progress, f_k
in [0, 1] how much of the primitive was flown. Forward progress counts only as far as the plan was
flown; backward progress always counts in full. It can never exceed plain progress, so nothing in it
can be farmed (the prim2_b025 farm needed a POSITIVE pay for completing). Charging backward progress in
full closes the cycle a plain product f x p would open (an unflown backward primitive then a flown
forward one would net positive with zero displacement, the non-potential cycle of Ng 1999); the
episode's sum stays <= d_spawn, reached only by flying every primitive. An additive coverage bonus,
even inside the bank, is weaker: it pays for more primitives, so it rewards delaying the finish.

## 7. Recommendation for RL_Surf

EXECUTOR: reward hiding. `--exec-cut` is right; keep it.
* Reward: arc progress along the CURRENT primitive only, inside the corridor, capped at its length,
  plus the small time penalty. No distance-to-finish term, no plan-quality term, no min/max.
* Remove the +50 race success bonus it still gets at the map finish (`rewards.py` `r[goal] +=
  self.success_bonus`; `goalsys._on_step_learned` returns the finish mask). It is the last task leak,
  worth more than a whole primitive, and it pays exactly the off-plan behaviour the code rides on.
* No dead zone: keep a pull toward the plan off the corridor too (Director's max-cosine; Meyer 2020's
  "fatter tails" kernel [PE]), so an unflyable primitive still dictates what the executor TRIES and
  shows up to the planner as no progress instead of working as a signal.
* Horizon: terminal at completion and death (h-DQN, HAC, HiTS). At a primitive TIME-OUT, either show
  the fraction of its budget left (one scalar; a new input shape, so the next scratch executor only)
  or bootstrap V under the SAME primitive there (Director, HIRO).
* Diet: keep plan_uniform 0.5 so it can still fly what the planner has not asked for yet (HiTS draws
  5% of timed subgoals uniformly; [PE] F2). Death: no charge beyond the primitive's lost pay; if it
  learns to finish primitives in doomed states, a small death charge is the one sanctioned exception.

PLANNER: the task, gated by obedience.
* Reward: credit_k = min(p_k, f_k x p_k) per 1,000 u, f_k = min(1, covered arc fraction / 0.9)
  (soft; SSE's strict form is f = 1[completed]), + 10 at the finish + end-cell novelty (none on a
  death), all inside the bank. plan_r_ok = plan_r_fail = 0 for good. HAC's caution: charging every
  miss of a NOISY executor breeds "overly conservative subgoals"; the soft fraction, or f measured on a
  greedy-executor test subset (~0.3 of primitives, left out of the executor's PPO batch), avoids it.
* Terminations, one potential for every end (Grzes). Two consistent versions:
  (A) Today (prim2f, the version that passed blue025): the bank charged back at EVERY non-finish end,
      death and the cap. It makes the cap part of the task, so Pardo applies: add the time left as a
      9th scalar. On long routes most reservoir states cannot finish within the cap, so late progress
      is worth nothing and early progress nearly face value, invisibly: one constant, different
      meanings per map, the hidden map dependence CLAUDE.md 0b warns about.
  (B) Ng's exact form, for the multi-map recipe: F = gamma_k Phi(s') - Phi(s) with Phi = the bank,
      charged back at a DEATH only; at the cap bootstrap V (Pardo: the cap is a training artefact).
      The gamma factor charges rent (1 - gamma_k) x bank per primitive, so "reach platform 2 and
      wait" (prim2e_b025) decays to 0 instead of being locked in, and a never-finishing trajectory is
      worth 0 with no clock anywhere. Run (B) against (A) where (A) stalls.
* Discount: SMDP, gamma_k = 0.95^(duration_k / 2 s) (Sutton, Precup, Singh 1999: Q(s,o) <- Q(s,o) +
  alpha [r + gamma^k max_o' Q(s',o') - Q(s,o)], k = the option's length), GAE lambda likewise; today a
  3-s time-out and a 1.5-s completion are discounted alike.
* Novelty: its own value head and normalised return, ~0.1 of the task stream (Director w_extr 1.0 /
  w_expl 0.1; RND's two heads); to the planner only (Director: worker exploration hurt).

GUARDS: nothing the executor's success alone can earn goes to the planner (a cooperation term only as a
Lagrange constraint, CHER); any charge stays <= the death charge, so dying never beats failing;
collusion alarm = planner entropy halving within ~20 updates while completion rises and progress does
not (prim2_b025); code alarm = realised progress per primitive far above planned with completion near
0 (prim2f: +167 vs +65 u, 0.7%).

LOG (per window, planner-chosen vs uniform primitives separately)
* Executor: completion, arc fraction, return per primitive (never above its pay), time-outs, deaths
  inside a primitive and within 0.5 s after a completed one.
* Planner: entropy and KL per update; realised vs planned progress, and credit, per primitive by
  outcome (completed / timed out / died); episode ends split finish / death / cap with the bank at the
  end; finishes from the true start; cells covered; novelty; primitive duration.
* The user's principle as a 2x2, (followed?) x (advanced?), with each level's mean reward per cell:
  the executor's must not depend on "advanced"; with the gate, the planner's "advanced but not
  followed" cell must fall toward zero.

## 8. The three pitfalls, with the evidence

1. PAYING THE PLANNER FOR WHAT THE EXECUTOR FINDS EASY. prim2_b025 (+0.5 per completion), in ~40
   updates: entropy 5.19 -> 0.40, completion 26% -> 83%, time-outs 2% -> 91%, novelty 0.008, 0
   finishes. CHER: over-cooperation gives "goals that match the worker's current state and prevent
   forward movement". AMIGo: "rewarded for all easy-to-reach states".
2. AN END PRICED DIFFERENTLY FROM THE OTHERS.
   * Positive stream + crossing return: our executor before the cut (prim2c: every episode ran out the
     clock, ~250 an episode; finishing ENDED the pay); Randlov-Alstrom's circles; Grzes: "whenever a
     positive reward was given in one area of the state space, it was profitable to revisit the same
     area many times when departures were not penalised"; Kostrikov 2019: a strictly positive reward
     "prevents the agent from solving tasks in a minimal number of steps".
   * Negative stream + free death: prim2b_b025 (-0.5 per failed primitive) dived off the platform as
     its second primitive in every greedy episode; Kostrikov: "it is common for learned agents to
     finish an episode earlier (to avoid additional negative penalty)".
   * Partial charges: prim2d_b025 (a death merely paid no progress) sent 40% of its primitives to death
     on the straight line over the void; prim2e_b025 (only a death charged the bank) learned "reach
     platform 2 and wait for the clock". Grzes: Phi(s_N) = 0 wherever a learning trajectory stops, so
     "all the high potentials accumulated before visiting s_N are 'neutralised'".
3. ONE LEVEL DOING THE OTHER'S JOB. An executor paid for the task (or max) bypasses its manager
   (Dayan-Hinton) and FuN-style hierarchies showed no clear gain over a flat LSTM (Director); one
   charged for bad plans (min) refuses detours. A planner not gated on obedience learns the executor's
   response function (prim2f: 0.7% completion; HAC's "unrealistic subgoals"); one charged for the
   executor's exploration noise turns timid (HAC's "overly conservative subgoals").

## 9. References (read for this file unless marked [PE]/[HIER]/[DET])

* Dayan, Hinton, "Feudal Reinforcement Learning", NIPS 5 (1993). cs.toronto.edu/~fritz/absps/dh93.pdf
* h-DQN (Kulkarni et al.) NIPS 2016, arXiv:1604.06057; FeUdal Networks (Vezhnevets et al.) ICML 2017,
  arXiv:1703.01161; HIRO (Nachum et al.) NeurIPS 2018, arXiv:1805.08296
* HAC (Levy et al.) ICLR 2019, arXiv:1712.00948; HiTS (Guertler et al.) NeurIPS 2021, arXiv:2112.03100
* Option-Critic (Bacon et al.) AAAI 2017, arXiv:1609.05140; deliberation cost (Harb et al.) AAAI 2018,
  arXiv:1709.04571; MLSH (Frans et al.) ICLR 2018, arXiv:1710.09767; HiPPO, DAC [PE]
* HAAR (Li, Wang, Tang, Zhang) NeurIPS 2019, arXiv:1910.04450; Director (Hafner et al.) NeurIPS 2022,
  arXiv:2206.04114; CHER (Kreidieh et al.) arXiv:1912.02368
* AMIGo (Campero et al.) ICLR 2021, arXiv:2006.12122; Setter-Solver (Racaniere et al.) ICLR 2020,
  arXiv:1909.12892; asymmetric self-play (Sukhbaatar et al.) ICLR 2018, arXiv:1703.05407
* HRAC arXiv:2006.11485, HIGL arXiv:2110.13625, DHRL arXiv:2210.05150, BrHPO arXiv:2406.18053,
  SSE arXiv:2506.21039, PDM-Closed arXiv:2306.07962, Plan-R1 arXiv:2505.17659, Meyer 2020
  arXiv:1912.08578 [PE]; GoalGAN arXiv:1705.06366 [DET]
* S3 (Srivastava, Jerath) arXiv:2607.19232 (July 2026, abstract only); Sutton, Precup, Singh,
  "Between MDPs and semi-MDPs", Artificial Intelligence 112, 1999
* Ng, Harada, Russell, "Policy invariance under reward transformations", ICML 1999; Randlov, Alstrom,
  "Learning to drive a bicycle using RL and shaping", ICML 1998 (quoted in U. Toronto CSC2542 notes)
* Grzes, "Reward Shaping in Episodic Reinforcement Learning", AAMAS 2017; Kostrikov et al.,
  Discriminator-Actor-Critic (reward bias), ICLR 2019, arXiv:1809.02925
* Pardo et al., "Time Limits in Reinforcement Learning", ICML 2018, arXiv:1712.00378; Burda et al.,
  RND, ICLR 2019, arXiv:1810.12894
