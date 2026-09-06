# Literature survey: action cadence, commitment and smoothing (2026-09-06)

For three measured defects of the round-30 expert-iteration loop on
`surf_src_cannonball` (131 Hz physics, `--act-every 4` = 30.7 ms = 32.6 Hz decisions,
six discrete heads `NVEC = (15, 7, 3, 3, 2, 2)` = yaw bin as a multiple of the analytic
strafe rate `atan(30/|v|)`, pitch, forward, side A/D, jump, duck):

* **D1 DITHER.** The policy flips A/D ~2.5 times/s, median held-key run 0.18 s
  (~6 decisions); the record flips 0.84/s and holds 0.42 s (~14). The planner inherits
  it: 2.14 flips/s, median side hold 0.046 s, wishdir inside 0.5 deg of perpendicular on
  50.5% of free-flight steps against the record's 82.6%.
* **D2 GAP.** The policy trails its own planner's line by a constant ~0.6 s (70.52 vs
  69.92 spawn; 0.64 s at the round-30 tail), across every loop variant tried.
* **D3 ROUTE.** The search never proposes a different finish route (one ramp touch, not
  two). 46 champion-free probes all landed in 70.00-70.02 s; per-decision entropy made
  it worse (eps 0.05 -> 69.56-69.60 vs the 69.506 line) or killed every lineage
  (eps 0.15/0.30 -> 0 finishes in 8 runs).

Not duplicated here: `docs/research-litsurvey-temporal.md` (stacking / GRU / online
chunking nulls, the deadbeat-strafe analysis, the factored-head saddle) and
`docs/research-litsurvey.md`. This file covers only WHEN the policy acts and HOW LONG it
commits.

## 0. Executive summary - the three mechanisms most likely to pay

1. **An action-conditioned HOLD head on the side key** (FiGAR + TempoRL; 2.1, 2.3).
   `log pi_hold` in the joint log-prob keeps the PPO ratio exact; W = 1..12 decisions =
   0.03-0.37 s brackets our 0.18 s and the record's 0.42 s. For D1, and FiGAR is the
   ONLY on-policy discrete policy-gradient version published.
2. **Chunked DISTILLATION of the planner** (ACT / Diffusion Policy, OPAL discipline;
   3.1): a FROZEN decoder fitted to the loop's own ~40,761 (state, planner action)
   rows/round, PPO over the code head, chunk 5-10 decisions per Q-chunking. For D2 - a
   compounding-error gap, which is what ACT's 1% -> 44% ablation measures.
3. **Held-key macros as the SEARCH's proposal with heavy-tailed durations** (4.4, 4.2).
   `--macro-hold` already exists in `beam_tas.py`; the probes used a FIXED hold grid.
   For D3 and D1 together, and the cheapest of the three.

Rank 4 down: a symmetric-KL smoothness term in the LOSS, never the reward (5.1);
`--act-hist 4`, implemented and never run; a learned delay head. What decides any of
them is free - the cadence block `beam_tas.py` writes into `summary.json`.

## 1. Decision rate and delay in the systems that beat humans

### 1.1 AlphaStar - the delay head (Vinyals et al., Nature 575:350-354, 2019)

**Mechanism, numbers.** Autoregressive over six heads: action-type -> **delay** ->
queued -> selected units -> target unit -> target location. `delay_logits` has **128
values, one per requested delay in game steps**, and - uniquely - **no temperature is
applied** before the multinomial (0.8 elsewhere). The agent chooses when it will next
OBSERVE, supervised from human replays with its own cross-entropy term. Agents wait
~370 ms on average between observations, sometimes seconds; observation-to-action delay
~350 ms; APM capped to a human envelope. The paper states the agent may react late to
unexpected situations - the price of committing.

**Mapping.** The most general hold head, but a *deliberation* rate, not a *commitment*
rate: AlphaStar does not act during the delay, whereas our engine consumes an action
every physics tick. What transfers is the ARGUMENT ORDER - delay is sampled second,
conditioned on the action type, TempoRL's result (2.3) two years early: condition our
hold head on the sampled side-key bin, not the state alone. The temperature-free
128-wide head also says the duration distribution is expected to be heavy-tailed and
should not be sharpened by whatever sharpens the action.

### 1.2 OpenAI Five - fixed frame skip (arXiv 1912.06680)

**Mechanism, numbers.** Dota 2 runs at 30 fps; Five acts on **every 4th frame**, a
timestep of **0.133 s**, ~20,000 steps per 45-minute episode, end-to-end reaction
~217 ms. No learned repetition, no sticky actions, heads computed independently.

**Mapping.** Our current design (fixed `--act-every 4`, 30.7 ms) and the existence proof
that a fixed skip plus independent heads can win a long-horizon game - so D1 is not
automatically a design flaw. But Dota has no 0.6-degree gain window: our impulse fires
only while wishdir stays within `atan(30/|v|)` of perpendicular and a key flip costs ~3
dead frames. Five licenses the fixed skip, not the iid noise on top of it.

### 1.3 GT Sophy (Wurman et al., Nature 602:223-228, 2022)

**Mechanism, numbers.** QR-SAC, **10 Hz** (100 ms), continuous action space = steering
plus a combined throttle/brake scalar, no pixels, 180 ego-frame track-edge points
spanning ~6 s at current velocity. Reward = course progress per decision cycle,
penalties for wall contact, off-course and tyre slip; NO lap-time term, NO finish bonus.
Maggiore 114.249 s vs 114.466 s human best, 200-lap std 0.061 s. Ablations: no QR head
+0.69 s, 1-step +1.48 s, **no course points +2.64 s**. Mixed scenarios with randomised
positions and speeds; the racing line was never supplied. "No substantial gains acting
more frequently than 10 Hz" over a 5-60 Hz sweep is **published with no figure or table
- do not cite it as a measured result**.

**Mapping.** (a) The racing line EMERGED from a progress reward, our `--race-arc` result
reproduced: D3 is a *proposal distribution* problem in our search, not a reward-shape
problem. (b) The related vision agent (Vasco et al., RLC 2024; arXiv 2406.12563) starts
episodes with position **uniformly sampled from in-course and off-course areas within 5%
of track width**, launch speed 0-105 km/h - the shape of our `tools/state_cloud.py`
finish-room arm, and published support for jittered arrival clouds as the instrument for
a LOCAL manoeuvre. (c) It feeds a **3-step steering history plus 2 steering deltas** and
pays two reward penalties for inconsistent steering (5.3). What does NOT transfer is
10 Hz: our strafe would lose the impulse entirely.

### 1.4 Linesight (TrackMania Nations Forever; project, not a peer-reviewed paper)

**Mechanism, numbers, mapping.** IQN over discrete key combinations; the action is
**latched for 5 engine ticks**, one decision per 50 ms (**20 Hz**) over a 10 ms engine
step. Observation: one greyscale frame, ~20 scalars, **the previous 5 actions**, 40
ego-frame virtual checkpoints; gamma = 1 inside a ~7 s mini-race window. First agent to
beat official-campaign world records (May 2024); median lap ~26.3 s, so the window is
27% of a race; no published cadence analysis. (The 50 ms / 5-tick figures come from the
project's config as recorded in `research-litsurvey-temporal.md` - project-sourced, not
paper-sourced.) Its decision interval is a **latch over a faster engine**, exactly our
`--act-every`, holding 5 ticks against our 4: we are already in the published band and
the rate itself is not the defect. It feeds **5 previous actions**, i.e. `--act-hist 5`
here, implemented in `python/surfgym/obsaux.py` and never run. Warning: its 7 s window
was reproduced faithfully in round 18 and was the round's only regression (-115,328 u).

## 2. Learned action repetition

### 2.1 FiGAR (Sharma, Lakshminarayanan, Ravindran; ICLR 2017; arXiv 1702.06054)

**Mechanism, numbers.** Factor the policy into an action part and a **repetition** part,
sampled independently from the same state, trained with the JOINT log-prob:
`L = (log f_a(a|s) + log f_x(x|s)) * A(s, a, x)`. Policy dimension grows as `|A| + |W|`,
not `|A| * |W|`; the action then executes for x steps, open loop, no abort. FiGAR-A3C
(Atari) W = {1..30}: **beats A3C on 26 of 33 games**, Enduro >900x, Atlantis >35x.
FiGAR-TRPO (MuJoCo) W = {1..30}: better on 3 of 5. FiGAR-DDPG (TORCS) W = {1..15}:
**557,929.68 vs DDPG's 59,519.70**, and the paper states FiGAR-DDPG learned policies
that were **smoother** than DDPG's. Discarding the repetition head at evaluation
degrades 24 of 33 games - load-bearing at execution, not only in training.

**Mapping - the rank-1 proposal.** The only published repetition method that is
simultaneously **on-policy, policy-gradient and discrete**, and its loss form is why:
`log pi_hold` in the joint log-prob leaves the PPO ratio exact, with neither defect that
killed `--ez-eps`. Concretely a seventh head over W = {1, 2, 3, 4, 6, 8, 12} decisions
(0.031-0.37 s at K=4 / 7.667 ms) on the SIDE key ONLY, leaving yaw free every decision -
the yaw bin is a RATE re-evaluated against live speed (`src/env.c surf_yaw_delta` under
`--yaw-adaptive`), so holding the view costs strafe efficiency while holding the key
does not. Predicted observable: median hold 0.18 -> 0.3-0.4 s, flips 2.5/s -> ~1/s.
TORCS is the closest published domain to ours, where FiGAR's margin was largest and the
smoothness claim was volunteered. ~1 day plus a bit-identity test at W = {1}.

### 2.2 Dynamic frame skip / Dynamic Action Repetition (arXiv 1605.05365 -> AAAI 2017, 31(1))

**Mechanism, numbers.** Do not learn a distribution - **double the action space**: `a_k`
and `a_(k + |A|)` are the same primitive at two frame-skip rates. **r1 = 4 and r2 = 20,
the same for every game.** DFDQN-1024 vs DQN-1024 at fixed skip 4: Seaquest 10,458 vs
5,450; Space Invaders 2,796 vs 1,189; Alien 3,114 vs 2,085. Epsilon annealed over 2M
steps instead of DQN's 1M, because the doubled action space needs more exploration.

**Mapping.** The cheap version of 2.1: widen the SIDE head from 3 bins to 6 (A/D/neutral
short and long) where "long" latches the key for L decisions in `src/env.c`. Nothing
else changes - the head is already categorical, PPO is untouched, and a 3-bin checkpoint
warm-starts by zero-padding the head as `widen_for_obs` does for observations. r2/r1 = 5
maps to L = 5 decisions = 0.153 s, inside the 0.18-0.42 s band. The caveat transfers
literally: budget MORE exploration after widening, not less.

### 2.3 TempoRL (Biedenkapp, Rajan, Hutter, Lindauer; ICML 2021, PMLR 139:914-924; arXiv 2106.05262)

**Mechanism, numbers.** A behaviour policy pi(a|s) plus a **skip policy conditioned on
the chosen action**, pi_J(j | s, a), over a skip-MDP whose transition is the j-fold
product and whose reward is the j-step discounted sum; one skip of length j yields
**j(j+1)/2 skip-transitions** because every shorter prefix is also observed. Max skip
J = 7 (gridworld), 10 (MountainCar), 4 (LunarLander), 10 (Atari), sweep
J in {2,4,6,8,10,14,20} on Pendulum. Tabular speedups **13.6x and 12.4x** on the cliff
task; Atari (5 games) splits three ways, best case Freeway ~34 vs DQN's ~25. Against
FiGAR on Pendulum with two skip values: **normalized AUC 0.76 vs 0.92**, diagnosed as
FiGAR's unconditioned head learning only "which repetition length works well **on
average** for all actions".

**Mapping.** One design instruction for 2.1: **condition the hold on the sampled side
bin**. Our left/right symmetry means a state-only hold head learns one duration for both
A and D, while the duration that pays depends on which side of the ramp we are on - the
averaging failure TempoRL names. The `j(j+1)/2` trick is value-based and does not
transfer to PPO. The no-abort problem is worse here (a 12-decision hold beginning 3
decisions before a ramp contact cannot be interrupted, and our episodes end in falls):
cap W at ~8 decisions, or adopt TAAC's closed-loop form.

### 2.4 TAAC (NeurIPS 2021; arXiv 2104.06521) and SDAR (ICLR 2025; arXiv 2502.06919)

**Mechanism, numbers.** TAAC is the **closed-loop** alternative: at EVERY step a
second-stage binary switch chooses between repeating the previous action and taking the
actor's new one, conditioned on the actually sampled action rather than the actor's
expected behaviour, with a "compare-through" Q operator for the multi-step backup. SDAR
generalises it per action DIMENSION, regenerating only the dimensions that chose "act".
TAAC beats strong baselines on 14 continuous-control tasks and mines substantial
repetition even where the task appears to repel it. SDAR: normalized AUC 1.0 vs TAAC's
0.90; **Action Fluctuation Rate 0.208 vs SAC's 0.245, Action Persistence Rate 3.75 vs
SAC's 1.00** - 3.75x longer holds at LOWER fluctuation, simultaneously.

**Mapping.** SDAR's two metrics are D1 restated: our APR is ~6 decisions (0.18 s), the
record's ~14 (0.42 s), and its contribution is that PER-DIMENSION repetition raises APR
without raising AFR - our case exactly, since yaw should keep deciding every 30.7 ms
while the side key holds. Build the hold head per-head (side only). TAAC's switch is
better in principle and worse to try first: off-policy (compare-through Q has no PPO
analogue), and a per-step switch on the previous action is PIC's mixture policy.

### 2.5 Action persistence theory (Metelli et al.; ICML 2020, PMLR 119; arXiv 2002.06836)

**Mechanism, numbers, mapping.** Persistence k is a control-frequency knob, and the loss
from acting at k is bounded by a **Time-Lipschitz constant** - how far the system moves
per unit time. PFQI learns the value function at a given persistence; a heuristic picks
k. "Persistence 1 rarely leads to the best performance" - Cartpole k = 4 scores 239.5 vs
k = 1's 169.9, the heuristic picking the optimal k ~ 4 with zero loss - yet "excessively
increasing persistence prevents the control at all". So an interior optimum must exist
and the theory says where: our Time-Lipschitz constant is large (3,000 u/s through a
0.6 deg window), so the optimum is SHORT. That is the case for capping W at 8-12
decisions rather than reaching for 0.42 s, and for a distribution over holds instead of
a longer `--act-every`, which raises persistence AND lowers the yaw decision rate - the
confound the round-14/15 ladder could not split.

### 2.6 Frame skip in Atari (Braylan et al., AAAI-W 2015; Machado et al., JAIR 61, 2018)

**Mechanism, numbers, mapping.** Braylan, Hollenbeck, Meyerson and Miikkulainen sweep
frame skip and find it among the highest-leverage hyperparameters in ALE, the best value
varying by an order of magnitude across games, with high-skip agents learning faster per
training episode. (The PDF would not convert in this pass - treat the "20-180" per-game
figures circulating for it as UNVERIFIED.) Machado et al. introduce **sticky actions**:
with probability **0.25** the environment repeats the previous action, now the ALE
v0/v5 default. Sticky actions are the cheapest possible D1 experiment and cut both ways
- a regulariser forcing robustness to the agent's own holds, and the discrete analogue
of correlated noise - but note the evidence's direction: they were introduced to make
Atari HARDER, and what they measurably do is break open-loop memorised action sequences,
which is what our 68.54 s planner line IS. Policy side only, never in `beam_tas.py`.

### 2.7 Elastic time steps - SEAC / MOSEAC (arXiv 2402.14961 v4)

**Mechanism, numbers, mapping.** The control interval is part of the action, with reward
terms trading task performance against elapsed time plus an energy penalty tied to the
NUMBER of actions (MOSEAC caps the time reward with alpha_max). Realised frequency
ranges **5-30 Hz**; best lap **43.2 s** vs CTCO 48.5, SEAC 46.1, SAC 47.6, using **~691
steps vs SAC's 957** (28% fewer decisions), in TrackMania. The closest published setting
in which the interval itself was learnable, and a racing game; below 2.1 because it is
SAC (off-policy, continuous) against baselines it chose. What transfers is the
**explicit price on deliberation**: option-critic's termination gradient shrinks options
to one step without a deliberation cost (xi = 0.01) and Harb et al. (AAAI 2018) use
eta in [0.005, 0.030]. If a hold head collapses to W = 1 that is the predicted failure,
and the published fix is a small per-decision cost in the LOSS, not the reward.

## 3. Action chunking and temporal ensembling

### 3.1 ACT (Zhao, Kumar, Levine, Finn; RSS 2023; arXiv 2304.13705)

**Mechanism, numbers.** A CVAE transformer predicts a **chunk of k future actions** from
one observation. Chunks OVERLAP - a new one every step - and the actions proposed for
the current timestep by all live chunks are combined by **temporal ensembling** with
exponential weights `w_i = exp(-m * i)`, w_0 the oldest, m controlling how fast new
observations are incorporated (smaller m = faster). 50 Hz control, **k = 100** (2.0 s).
Ablation: **1% success at k = 1, 44% at k = 100**, tapering at k = 200 and 400. Temporal
ensembling adds 3.3% (ACT) / 4% (BC-ConvMLP). 50 demonstrations per task, ~10 minutes of
data. Stated purposes: cut compounding error, model non-Markovian human pauses, remove
the jerk from adopting a new chunk every k steps.

**Mapping - the rank-2 proposal, with the honest translation.** D2 is an imitation gap
against a demonstrator whose action sequence is NOT a function of state alone - an
open-loop plan found by search - which is ACT's "non-Markovian demonstrations" case, and
1% -> 44% is the largest single effect in this survey. Two adaptations. (i) **Temporal
ensembling over discrete heads must average the LOGITS (or probabilities), not the
argmaxes** - the mean of bin 0 and bin 2 is bin 1, a different key; ensembling the
per-head categoricals with `w_i = exp(-m i)` then taking the argmax is well defined and
keeps the property that one dissenting chunk cannot flip the key. (ii) **Chunk length
comes from Q-chunking, not ACT**: Q-chunking (Li, Zhou, Levine; NeurIPS 2025) reports
h = 5 default, useful to 10, **h = 50 at zero success** in ONLINE RL, warning that
chunking "may perform poorly in settings where a high-frequency control feedback loop is
essential" - us. h = 5-10 decisions is **0.15-0.31 s**, where the record's 0.42 s and our
0.18 s both live. This is NOT round 17's `--chunk` arm, which learned a codebook ONLINE
from scratch under a max-entropy bonus and collapsed 2 of 2 (entropy 4.16 -> 0.98); every
published recipe fits the decoder to demonstrations first and freezes it, and
`tools/plan_to_bc.py` already writes ~40,761 rows per round.

### 3.2 Diffusion Policy (Chi et al.; RSS 2023; arXiv 2303.04137)

**Mechanism, numbers.** Predict an action SEQUENCE by denoising, execute it under
receding-horizon control with three horizons: observation To, prediction Tp, execution
Ta - predict Tp, execute the first Ta, re-plan. Defaults **To = 2, Tp = 16, Ta = 8**
across 15 tasks in four benchmarks; **average improvement 46.9%**. The horizon ablation
states "too long a horizon reduces performance due to slow reaction time". Its stated
reason for sequence prediction is temporal consistency: independently sampling each step
from a multimodal action distribution gives "jittery actions that alternate between the
two valid trajectories".

**Mapping.** That last sentence is D1 written three years early: our strafe has exactly
two valid modes, the heads are conditionally independent given the state
(`sample_padded` draws six Gumbel-argmaxes), and the policy sits at 62% of uniform
entropy, so per-decision sampling alternates between them. Its contribution beyond ACT
is the **Tp/Ta split** - predict 16 decisions, execute 8, re-plan - a chunk of 16 at 50%
overlap, and the cheapest way to get ACT-style ensembling without keeping k chunks alive.
`--chunk` already has the decoder plumbing and a `--codebook-init` path: a re-use.

## 4. Temporally correlated exploration

### 4.1 Pink noise (Eberhard, Hollenstein, Pinneri, Martius; ICLR 2023, notable-top-25%)

**Mechanism, numbers.** Draw action noise from a sequence with power spectral density
proportional to `f^-beta`: beta = 0 white (iid), beta = 2 red/Brownian (the OU limit),
**beta = 1 pink**, halfway; generated per episode and indexed by timestep. Evaluated on
**MPO and SAC**; pink significantly outperforms white, OU and the other colours across a
wide range of environments, and **in 80% of cases pink is not outperformed by any other
noise type**, the general finding being that intermediate correlation (0 < beta < 2)
beats both ends. (Neither the PDF nor OpenReview would convert in this pass.)

**Mapping.** Both endpoints have already fired here. Our per-decision iid draw is
beta = 0 and produces the dither; `--ez-eps` (uniform random 6-tuple bursts) sat closer
to beta = 2 and off-manifold, a clear negative. The argument for the AR(1) Gumbel
construction in `research-litsurvey-temporal.md` section 2: replace the uniform `u`
inside `argmax(logits - log(-log u))` with a per-(env, head, bin) sequence whose marginal
is exactly Uniform(0,1) but correlated across decisions. Every step then samples exactly
`Categorical(softmax(logits_t))`, `logp` is exact, nothing is dropped and the PPO ratio
is untouched. Caveat: the marginal is preserved unconditionally, not given `s_t`.

### 4.2 Colored noise in PPO (Hollenstein, Martius, Piater; AAAI 2024, 38(11); arXiv 2312.11091)

**Mechanism, numbers.** The same colored sequence inside PPO's reparameterised Gaussian
`a = mu + sigma * eps`, sequence length 1000. beta swept over
{-1, 0, 0.2, 0.5, 0.75, 1, 1.25, 2}, **16 environments, 20 seeds**; **beta = 0.5 wins**
and the paper recommends changing PPO's default from beta = 0 to beta = 0.5. The optimal
beta RISES with the number of parallel environments. The on-policy justification is
purely marginal: because eps stays Gaussian, the data "viewed at each individual step,
remains asymptotically on-policy".

**Mapping.** The constant to use. We run 2,048 parallel environments - where this paper
says MORE correlation is affordable - so beta = 0.5 is a floor, not a ceiling. ~30 lines:
one persistent `(N, NACT, NPAD)` fp32 buffer inside the captured CUDA graph, same pattern
as the GRU's `static_h`, zeroed at reset beside `ez_left`, behind `--noise-beta`
defaulting to 0 (bit-identical). Free pre-test: log the realised key-hold distribution
over 1e7 rollout steps. Known risk, from Deep Coherent Exploration (Zhang & van Hoof,
ICML 2021): correlated policies "tend to satisfy the KL constraint in fewer update
steps, which leads to slower learning" - watch the minibatch count before the early stop.

### 4.3 Ornstein-Uhlenbeck noise (Lillicrap et al.; ICLR 2016; arXiv 1509.02971)

**Mechanism, numbers, mapping.** DDPG's original exploration: an OU process
`dx = theta (mu - x) dt + sigma dW` added to the deterministic action, theta = 0.15,
sigma = 0.2, "to explore well in physical environments that have momentum". Superseded -
4.1 measures OU (beta = 2) as worse than pink across a wide range of environments, and
TD3/SAC dropped it for plain Gaussian noise. Historical anchor and warning label: OU is
the maximally correlated end, where `--ez-eps` sat. Do not reach for beta = 2 because
our symptom "looks like under-correlation" - 4.1 and 4.2 both put the optimum at 0.5-1.

### 4.4 ez-greedy / temporally-extended epsilon-greedy (Dabney, Ostrovski, Barreto; ICLR 2021; arXiv 2006.01782)

**Mechanism, numbers.** With probability eps, sample a UNIFORM RANDOM action and repeat
it for a duration `n ~ zeta(mu)` - heavy-tailed. **mu = 2.0 fixed for all Atari
results**, duration capped at 100 for Rainbow; R2D2 + ez-greedy median **17.77 vs
16.44** on Atari-57, improving hard exploration with little loss elsewhere, where
intrinsic motivation gains more on the hard games at large cost on the rest. Its own
listed failure mode: long exploratory trajectories can "waste time (e.g. running into a
wall for thousands of steps)". Entirely value-based; no policy-gradient experiment and
no importance-sampling analysis.

**Mapping - the rank-3 proposal, aimed at D3.** The failure mode it names is what a
random 6-tuple does in surf, and what `--ez-eps` and the round-30 proposal-entropy probe
both measured. But the mechanism was never tested where it belongs: **the planner's
proposal distribution**. `beam_tas.py --macro-hold MIN:MAX` already draws one
`(side, forward)` pair per non-greedy env and holds it for a **log-uniform** duration in
[MIN, MAX] s, with `--macro-yaw track` setting yaw analytically to the bin that tracks
the velocity rotation the held key produces. Log-uniform is not zeta - scale-free over a
BOUNDED range where zeta is heavy-tailed - and the finish-room probes used a FIXED grid
of holds (21/42/84/168 ticks) rather than either. `--macro-hold 0.15:1.5` with a
zeta(mu = 2) duration and `--macro-yaw track` is a ~2 hour change to an existing flag
and the only mechanism here that addresses D3: selection cannot pick what is never
proposed, and two-touch -> one-touch is one long held key.

### 4.5 The discretization theory (Tallec, Blier, Ollivier; ICML 2019; arXiv 1901.09732)

**Mechanism, numbers, mapping.** Theorem 4: per-step iid exploration collapses to a
deterministic drift as dt -> 0 and the action-value gap vanishes linearly in dt; the
discrete remedy, verbatim, is to discretize an |A|-dimensional OU process and set
`pi_explore(s, z) = argmax_a (A(s, a) + z[a])`, off-policy only (stated twice), not
PPO/TRPO/A3C. Park, Kim & Kim (NeurIPS 2021) prove the complement - policy-gradient
variance can diverge as the step shrinks - fixed by Safe Action Repetition. This is the
theoretical reason D1 is not a tuning problem: at 32.6 Hz decisions over 131 Hz physics
we are in the small-dt regime where iid noise is provably degenerate. It also names the
confound in our `--act-every` ladder, which varied the RATE with the noise held iid:
reopen rate and commitment jointly (a hold head at fixed K = 4) or not at all.

## 5. Smoothing and low-pass filtering of actions in high-rate control

### 5.1 CAPS (Mysore, Mabsout, Mancuso, Saenko; ICRA 2021)

**Mechanism, numbers.** Two regularisers in the policy LOSS, not the reward: temporal
`-lambda_T ||pi(s_t) - pi(s_{t+1})||` and spatial `-lambda_S ||pi(s) - pi(s_bar)||` with
`s_bar ~ N(s, sigma)`. On a real quadrotor, current draw **22.87 A -> 4.86 A**,
smoothness metric down ~96%, and every CAPS agent was flight-worthy where the baseline
needed cherry-picking; both terms were required. **The lambda and sigma values are not
printed in the paper** - the circulating `lambda_T = 20` is a third-party restatement.

**Mapping.** Right mechanism, Euclidean metric, which a factored categorical lacks. The
translation is a symmetrised KL between consecutive decisions' head distributions,
inside the sequence minibatch `--rnn` already builds (`train_fast.py::seq_loss`), on the
SIDE head only. Rank 4 because it makes the policy commit to whatever it is already
doing, lowering the flip rate without necessarily improving the strafe - it could
equally lock in the 13.4% of ticks spent on the dead branch. Pair it with the gain-band
share from `compare_wr.py`, never the flip rate alone.

### 5.2 LipsNet (Song et al.; ICML 2023) and low-pass filters in locomotion

**Mechanism, numbers.** LipsNet is a network with a learned, state-dependent Lipschitz
bound on its output: lambda = 1e-3, K_init = 5, DMC Reacher action fluctuation **2.41 ->
0.04 at equal return**; only the LOCAL variant is safe. Separately, Butterworth low-pass
filters on the policy output are near-standard in learned locomotion, cutoffs **4-5 Hz
on target joint angles** and ~10 Hz on filtered observations; removing the filter still
learns but with worse converged return and visible jitter, and a One-Euro filter sits
between "no filter" (highest performance) and "low-pass" (fewest falls). Stated
rationale: RL is prone to bang-bang control and a small discretization step "can induce
artificially high-frequency controller input" - the empirical form of 4.5.

**Mapping.** Both are continuous-output constructions with no categorical form, and a
low-pass filter is undefined on a key press: A/D is binary. The one axis where filtering
could apply is **yaw**, a rate quantised into 15 bins spanning -20x to +20x the analytic
strafe optimum - but that loop is deadbeat stable, a 10 deg error returning to 0.009 deg
in ONE physics tick. What DOES transfer is LipsNet's negative result: it reproduced the
finding that a **consecutive-action reward penalty INCREASES fluctuation under sparse
reward** (independently found by PIC, AAAI 2021). Our bonus is +50 over ~2,600 decisions.

### 5.3 The two racers that pay for smoothness in the REWARD

**Mechanism, numbers, mapping.** Swift (Kaufmann et al., Nature 620:982-987, 2023) adds
`r_cmd = lambda4 ||a_omega|| + lambda5 ||a_t - a_{t-1}||^2` (weights not verified here).
The vision-based GT agent (Vasco et al., RLC 2024) adds TWO: a steering-change penalty
on `|theta_t - theta_{t-1}|` and a steering-history sigmoid on
`|delta_t| + |delta_{t-1}|`, "to discourage the agent to make inconsistent decisions in
a short period of time". Both won. These are the counter-examples to 5.2's warning, and
the distinction is reward density: both optimise DENSE progress, so a small smoothness
rent is a small perturbation, whereas PIC and LipsNet found it backfires under SPARSE
reward. Our shaping is dense but the verdict metric is a terminal event, and round 18
measured how sensitive this policy is to a shaping rent (the -4.24 potential barrier at
the final descent). The mechanism belongs in the loss, not the reward.

## 6. Action commitment as part of the action space: the comparison

| system | commitment mechanism | granularity | on-policy? | discrete? |
|---|---|---|---|---|
| AlphaStar (Nature 2019) | delay head, 128 bins, no temperature | when to next OBSERVE | yes | yes |
| OpenAI Five (1912.06680) | fixed frame skip 4 (0.133 s) | none learned | yes | yes |
| GT Sophy (Nature 2022) | fixed 10 Hz | none learned | no (QR-SAC) | no |
| Linesight (TMNF) | fixed 5-tick latch (20 Hz) | none learned | no (IQN) | yes |
| FiGAR (ICLR 2017) | repetition head, W = 1..30 | open loop, no abort | **yes** | **yes** |
| DAR (AAAI 2017) | doubled action space, r1=4 r2=20 | open loop | either | **yes** |
| TempoRL (ICML 2021) | skip head conditioned on the action, J <= 10 | open loop | no | yes |
| TAAC (NeurIPS 2021) | per-step act-or-repeat switch | **closed loop** | no | no |
| SDAR (ICLR 2025) | per-DIMENSION act-or-repeat | closed loop | no | no |
| MOSEAC (2402.14961) | the interval itself is an action, 5-30 Hz | continuous | no | no |
| ACT / Diffusion Policy | fixed chunk k / Tp-Ta split | open loop, re-planned | BC | no |
| q1physrl (Quake) | `key_press_delay = 0.3 s`, environment-level | hard constraint | yes | yes |
| **RL_Surf today** | fixed `--act-every 4` (30.7 ms) | none learned | yes | yes |

Two readings. **Only two entries are simultaneously on-policy and discrete - FiGAR and
DAR** - which is why they rank 1 and 2 and the elegant closed-loop methods do not. And
`q1physrl` is the closest relative here: Quake air-strafe physics at 72 Hz, beat the
human record, and it did not LEARN a hold - it imposed one, a 0.3 s minimum between
consecutive presses of the same key (plus `smooth_keys`, a half impulse on the first
frame after a press), within a factor of two of the surf record's 0.42 s. A one-line
environment constraint and the cheapest test of the dither hypothesis.

## 7. Validated dead, or unlikely for us

* **A per-step reward penalty for switching actions.** PIC (AAAI 2021) and LipsNet
  (ICML 2023) independently report it INCREASES fluctuation under sparse reward, our
  regime. Put smoothness in the loss (5.1) or nowhere.
* **A shorter discount horizon or a temporal mini-race window.** Closed in round 18
  (-115,328 u). MOSEAC and the cadence literature are about the ACTION interval.
* **`--ez-eps` as implemented** - uniform random 6-tuple bursts with the held transitions
  dropped from the PPO loss. Structurally dead (broken ratio, up to 75% of the batch
  discarded). Do not retune eps; rebuild it as 4.1/4.4. Likewise **raw per-decision
  proposal entropy in the planner**: eps 0.05 -> 69.56-69.60 s (worse than the 69.506
  line), eps 0.15 and 0.30 -> zero finishes across 8 runs (2026-09-06). The published
  alternative in both cases is a correlated or macro proposal, not more eps.
* **Online, from-scratch chunk codebooks under a max-entropy bonus.** 2 of 2 collapses
  here (round 17) and no paper publishes that variant. ACT, OPAL, SPiRL, Diffusion
  Policy and VQ-BeT all fit the decoder to demonstrations first.
* **Chunk or hold lengths above ~10-16 decisions in online RL.** Q-chunking measures
  h = 50 at zero success; Diffusion Policy states long horizons cost reaction time;
  Metelli bounds the loss by the Time-Lipschitz constant, large here.
* **Low-pass filtering the key presses.** Undefined on a binary actuator, and the yaw
  axis it could apply to is deadbeat-stable within one physics tick.
* **OU / beta = 2 correlation**, superseded by pink (ICLR 2023) and beta = 0.5 for PPO
  (AAAI 2024); and **Sophy's "no gains above 10 Hz"**, published with no figure or table
  and not citable as a measured rate optimum.
* **Frame stacking and a GRU as fixes for dither.** Both already null here, and both are
  observation-side treatments for an action-side defect. `--act-hist 4` is the cheap
  remaining observation-side test and is still unrun.

## Appendix: what to measure, and the falsification

Every mechanism above predicts a move in the SAME numbers, which `beam_tas.py` and
`tools/demo/compare_wr.py` already write into `summary.json` under `cadence`: **A/D
flips per second** (policy 2.5, planner 2.14, record 0.84; target ~1.0-1.5), **median
held-key run** (policy 0.18 s, planner 0.046 s, record 0.42 s; target 0.3-0.4 s), and
the **share of free-flight steps whose wishdir is within 0.5 deg of perpendicular to
velocity** (planner 50.5%, record 82.6%). An arm that moves the flip rate without moving
that last share did not work for the stated reason: it locked in the dead branch. Per
CLAUDE.md section 3, an arm on a checkpoint that FINISHES is judged on wall-clock time
from start to finish plus finishes-out-of-9, never `race/eval_progress`. All three
ranked mechanisms are warm-startable on the round-30 finisher (exitLONG2 r8, 70.166 s
spawn / 69.18 s record clock), one seed, same card, matched control - and 1 and 3 have a
free CPU-only pre-test before any rental: 1 by logging the realised hold distribution
over 1e7 rollout steps, 3 by running `beam_tas.py --macro-hold` on the local CPU, where
the finish-room probe series already runs.
