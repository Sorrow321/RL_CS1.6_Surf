# The *Zero family against this task: what transfers when the model is already perfect (2026-09-04)

Commissioned by the owner, who listed the AlphaZero/MuZero line and asked
whether something there is "very suitable for us". Companions:
`docs/research-litsurvey.md`, `docs/research-litsurvey-temporal.md`,
`docs/research-results.md` round 30. Every paper was read this session from
arXiv HTML/ar5iv, the arXiv ancillary pseudocode, the `mctx` source, or (for
Gumbel MuZero, whose OpenReview PDF is gated) Danihelka's UCL thesis
chapter 5, which is that paper verbatim. No GPU, no rentals, $0.

## 0. The three facts that decide every verdict below

**(1) We already have what MuZero exists to obtain.** `src/` is a bit-exact,
deterministic, save/restore-able C model at **1.62e6 env-steps/s in-search**
(round 27's beam measurement; 6.35e5 amortised over a 26-wave plan phase).
The depth observation is a pure function of `(origin, yaw, pitch, ducked)`,
so an imagined state is *rendered*, not predicted. Every paper whose
contribution is a learned latent dynamics function is contributing the one
component we would delete.

**(2) Two corrections to the brief.** The joint action space is
`ACTION_NVEC = (15, 7, 3, 3, 2, 2)` = **3,780**, not 11,340
(`python/surfgym/core.py:61`). And `beam_tas.make_scorer` has **no speed
term**: elites rank by `-d` (geodesic), by `V(s)` under `--score v`, or by
route index; `--score dv` switches d -> V inside `--v-switch` 20,000 u.

**(3) The two open problems are measured.** Best policy 73.21 s spawn-clock
against the record's 68.60 s. The route search splits the gap into **3.70 s
of strafe execution** over the first 208 ku where the lines are within 900 u
(A/D flips 2.14/s vs 0.42; median key hold 0.046 s vs 0.418 s; wishdir
within 0.5 deg of perpendicular on 50.5% vs 82.6% of free-flight steps) and
**1.42 s at the finish-room junction J1**, a one-contact wall-ride with
**zero mass in the planner's proposal** (0 of 1,536 free continuations).

---

## 1. The papers

**1.1 AlphaZero (Silver et al. 2017, arXiv 1712.01815).** One net
`(p, v) = f(s)`; MCTS at **800 simulations per move** (40/80/200 ms
chess/shogi/Go) evaluating leaves with `v` instead of rollouts; loss
`l = (z - v)^2 - pi^T log p + c||theta||^2` with `pi` the **root
visit-count distribution** and `z` the game outcome; Dirichlet root noise
`alpha = {0.3, 0.15, 0.03}` scaled inverse to branching. Methods state the
model assumption outright: "AlphaZero is provided with perfect knowledge of
the game rules... used during MCTS, to simulate the positions resulting from
a sequence of moves."

*Fit: the closest paper here, and the only one whose model assumption is
ours.* We have the perfect deterministic model and none of the three things
that make the loop work - a tree, a value target from search, or a
visit-count policy target. Our BC term regresses six heads onto **one action
index**, the `tau -> 0` limit of AlphaZero's target, and the critic never
sees a search-derived value at all.

**1.2 MuZero (Schrittwieser et al. 2019, arXiv 1911.08265).** Replace the
simulator with `h`/`g`/`f`, unrolled `K = 5` steps and trained *only* to
predict reward, value and policy: "There is no direct constraint or
requirement for the hidden state to capture all information necessary to
reconstruct the original observation... nor any requirement for the hidden
state to match the unknown, true state of the environment." 800 sims for
board games, **50 for Atari**; search target
`N(a)^{1/T}/sum_b N(b)^{1/T}`.

*Fit: "AlphaZero plus a reward head and n-step bootstrapping" - take the
scaffolding, discard `g`.* Value equivalence buys an imperfect model we do
not need. What survives is that `pi` is a *distribution*, which is what
makes the loop robust to the search being wrong sometimes.

**1.3 MuZero Reanalyse / Unplugged (2021, arXiv 2104.06294).** Re-run MCTS
with the **latest** net on **old stored trajectories**; the fresh
visit-count policy and search value become the new targets. **With total
computation held constant**, the Reanalyse fraction alone moves the data
budget three orders of magnitude: 99.5% -> 20M frames (126.6% median HNS),
95% -> 200M (1006.4%), 50% -> 2000M (1331.7%). At 100% (fully offline) it
beats CRR 265.3% vs 155.6% median with no BC regulariser and no policy
constraint. Constants (2019 App. H): fresh policy on **80% of updates**,
value-target weight **0.25** against 1.0 for policy/reward, n-step **5**,
**2.0 samples per state instead of 0.1**. The pseudocode ships
`MostRecentBuffer`, `HighestRewardBuffer` and `DemonstrationBuffer`.

*Fit: the most applicable paper after AlphaZero.* We have a replay buffer
nobody reanalyses - the respawn reservoir (100k default, 128 goal-distance
bins). `tools/expert_dagger.py` is 70% of the machinery (600 sampled states
x 256 window copies x 3.0 s) but labels with an **argmax** and emits **no
value target**. Their three-buffer split maps onto elite lines vs reservoir.
PPO is on-policy, so reanalysed targets enter the auxiliary term - where
`--bc-file` already lives.

**1.4 Sampled MuZero (Hubert et al. 2021, arXiv 2104.06303).** Policy
improvement when you can only look at K of |A| actions: draw K from a
proposal `beta` (= `pi`, temperature-modulated, **with replacement**) and
define `Ihat_beta pi = (beta_hat/beta) f(s, a, Zhat_beta)` - consistent, not
finite-K unbiased. Correcting the visit counts afterwards is explicitly
rejected as numerically unstable; instead put
`pihat_beta = (beta_hat/beta) pi` **into the PUCT prior** and use the counts
unchanged. K: Go (|A| = 362) K in {15,25,50,100}, 50 near the full-action
baseline; Atari (18) K = 3; DM Control **K = 20 at 50 sims**; on 21-dim
humanoid.run **K = 3 suffices, no gain past 10**. Continuous actions use a
**factored categorical, 7 bins per dimension**, precisely "to avoid the
exponential explosion".

*Fit: the paper that licenses our action space, and its K's are small.* Our
policy already *is* the factored categorical it recommends, over 3,780 joint
actions; a search needs single-digit-to-20 samples plus the importance
correction, not thousands. The catch: `beam_tas` draws its K candidates
**i.i.d. per decision**, applying the sampled-MuZero proposal to a
*sequence* - that is where 2.14 flips/s comes from, and the fix is to change
what `beta` samples, not to sample more.

**1.5 EfficientZero (Ye et al. 2021, arXiv 2111.00210).** Three additions
for the 100k-step Atari regime: a **SimSiam consistency loss along the
dynamics** because "none of the reward, value and policy losses can provide
enough training signals to learn the environment model"; an **end-to-end
value prefix from a 512-unit LSTM** to fix **state aliasing** (from an
aliased latent you know the reward is coming, not at which step); and a
**model-based off-policy correction** shortening n-step for stale data.
Atari 100k mean **1.943** at 50 sims; removing the consistency loss costs
**55% of the mean and 69% of the median** - it *is* the paper.

*Fit: no, and the ablation is why.* Its dominant component stops a learned
latent drifting from the encoder; the LSTM fixes reward timing under latent
aliasing, where our reward is an exact function of state computed by the
trainer's own code. **Value-equivalence and consistency add nothing when the
model is real.**

**1.6 EfficientZero V2 (Wang et al. 2024, arXiv 2403.00564).** Keeps those
and replaces the search and the value target. (a) **Sampling-based Gumbel
search**: sample `AS = [AS1, AS2]` with `AS1` from the policy and `AS2` from
a **flattened** version of it, Sequential-Halve at the root, and sample
**only from the policy and fewer actions at non-root nodes** so the budget
buys depth. (b) **Search-Based Value Estimation**: the value target is the
mean over the tree's own imagined rollouts, riding the existing reanalyse
pass at no extra cost. (c) A **mixed target** that falls back to multi-step
TD while the model is bad or the data is fresh. **N_sim = 32 (16 on Atari),
K = 16 sampled actions (8 on Atari)**. Two numbers to keep: **S-Gumbel at
n = 8 beats Sampled MuZero at n = 50**, and TD-MPC2's MPPI evaluates **9,216
latent states per decision against EZ-V2's 32** for a 2.5% score edge.

*Fit: the search half yes, the model half no.* "Sample K, sequential-halve,
fewer samples deeper" is directly implementable against the real simulator.
Its two-source sample set is also the principled form of the `--eps`
widening that DNF'd **6 of 6** of round 27's MPC arms: widen the *prior you
sample from*, never randomise the *action you take*. With an exact model
SVE's error bound collapses to rollout variance alone.

**1.7 ReZero (Xuan et al. 2024, arXiv 2404.16364).** (a) **Backward-view
reuse**: reanalyse a trajectory in *reverse*, so when searching `S^t` you
already hold `S^{t+1}`'s root value; the root score for the logged action
becomes `r^t + gamma*m^{t+1}` and **if that action is selected the
simulation terminates immediately** - the subtree whose value you already
know is never expanded. On Pong: **54% fewer tree-search calls**, sample
efficiency unchanged. (b) **Entire-buffer reanalyse**: MCTS is removed from
data collection entirely (actions straight from the policy net, no
significant evaluation loss) and the whole buffer is refreshed once per
epoch instead of per minibatch. Net **2-4x less wall-clock per 100k steps**.
Bases: MuZero+SSL, EfficientZero, Sampled MuZero - **not** Gumbel MuZero.

*Fit: we already have its pipeline shape and none of its content.* Our
rollouts sample straight from the policy with no search and all search lives
in an offline batched phase - ReZero (b), arrived at by accident. What we
lack is targets coming out of that phase, and the backward view:
`expert_dagger` samples every 0.5 s and searches 3.0 s ahead, so its windows
overlap ~6-fold and reversing the order is the same trick. Cost, not
capability.

**1.8-1.11 The latent-world-model line: UniZero (2406.10667), ScaleZero
(2509.07945), ObjectZero (2601.06604), PriorZero (2605.12289).** All four
are LightZero-family MuZero descendants; in all four the contribution is
*learning* the model, and the search is inherited unchanged. **UniZero**
replaces MuZero's recurrent latent with a transformer over `(latent,
action)` tokens that disentangles the latent state from the implicit
history, because the unrolled latent "becomes tightly coupled with
historical information... fundamentally incompatible with the SSL loss".
**ScaleZero** adds MoE blocks (8 routed + 1 shared expert, top-1), a ViT
encoder and staged LoRA growth in which solved tasks stop receiving
gradient; it *rejects* gradient surgery (MoCo-style re-weighting cost ~40%
per step for marginal benefit) and handles action spaces with **per-task
heads**. **ObjectZero** puts frozen slot-attention encoders (DINOSAUR, 5
slots) and a C-SWM-style GNN dynamics model under **EfficientZero V2,
verbatim**. **PriorZero** mixes an LLM prior into the PUCT prior **at the
root only** (`alpha = 0.5`), fine-tuning that LLM by clipped PPO whose
advantage `Q_n(z,a) - v(z)` comes from the world model's own value function.

*Fit: no, all four, and their own numbers help.* UniZero wins where history
is needed (VisualMatch, Pong at `stack=1`, multitask) and on plain Atari
100k sits at or below its own controlled MuZero reproduction (mean 0.39 vs
0.44); its decoder-regularisation ablation reports "negligible impact", so a
*better generative* model buys nothing. ScaleZero's premise is heterogeneous
multi-task - one map, one task leaves MoE nothing to route - and its result
is parity anyway (Atari26 median 0.16 vs 0.21). ObjectZero ties its own
monolithic baseline, and a 64x32 depth raster of static level geometry has
no objects for slots to bind. PriorZero needs an enumerable, nameable action
set (Jericho's `valid_actions`, 10-50 strings) and a language-describable
observation; 2,500 decisions would be 2,500 chain-of-thought calls per
episode. Two things are worth extracting: PriorZero's **root-only prior
mixture** as a domain-agnostic skeleton, and ScaleZero's **dormant-neuron
ratio + effective rank + latent norm** as a free plasticity probe on the
documented 2.9x weight norm.

**1.12 Gumbel MuZero (Danihelka et al. 2022, ICLR spotlight).** Draw m
actions by **Gumbel top-k without replacement**, `argtop(g + logits, m)`,
and **reuse the same g** in the final selection ("we avoid a double-counting
bias"). Spend the n simulations by **Sequential Halving** over
`ceil(log2 m)` phases, each surviving action visited equally often, then
select `argmax(g + logits + sigma(qhat))` - which is `>= q` of a plain
Gumbel-Max sample from `pi` for **any** g and any `n >= 1`, because sigma is
monotone. Hence guaranteed improvement at **n = 2**.
`sigma(qhat) = (c_visit + max_b N(b)) * c_scale * qhat`, with **`c_visit =
50`, `c_scale = 1.0` on Go/chess but `0.1` on Atari** (0.1 is the shipped
`mctx` default). Unvisited actions are **completed** with `v_mix`. The
learning target is **`pi' = softmax(logits + sigma(completedQ))`, distilled
by `KL(pi', pi)`** - "This loss trains all actions, not only the action
`A_{n+1}`." `m = min(n, 16)`; **no Dirichlet noise and no temperature at
all**. 9x9 Go training-step speedup over MuZero at n = 200: **5.9x at
n = 32, 11.3x at n = 16**, where MuZero fails to learn from 16 or fewer.

*Fit: the most relevant paper after AlphaZero, on three counts.* (1) The
guarantee holds at simulation counts affordable *per decision along a
2,500-decision episode*. (2) Sequential Halving is **batch-shaped** - each
phase evaluates a set in parallel, exactly what a 2,048-env lockstep
simulator wants, where UCT's descent is sequential and would use one env at
a time. (3) Gumbel top-k without replacement is the principled version of
`beam_tas`'s ad-hoc `--dedup` prefix rehash, and `sample_padded`
(`train_fast.py:1199`) is already `argmax(logits - log(-log u))`. Import
`c_scale` carefully: our Q's are the geodesic field, `V(s)` or an arc
coordinate on unrelated scales, so the [0,1] normalisation has to be built.

**1.13 Expert Iteration (Anthony, Tian & Barber 2017, arXiv 1705.08439).**
Our loop's lineage: expert = MCTS with the apprentice as prior, apprentice =
imitation of the expert, and "removing the expert improvement step from
online ExIt reduces it to DAGGER". The result that matters: **CAT**
(`-log pi(a*|s)`, `a* = argmax_a n(s,a)`) versus **TPT**
(`-sum_a (n(s,a)/n(s)) log pi(a|s)`, the root visit distribution over 10,000
sims). Move-prediction accuracy is **indistinguishable** - top-1 error 47.0%
(CAT) vs 47.7% (TPT) - yet **the TPT network is 50 +/- 13 Elo stronger**.
Verbatim: "TPT is cost-sensitive: when MCTS is less certain between two
moves... it induces the IL agent to trade off accuracy on less important
decisions for greater accuracy on critical decisions", and "Accurate
evaluations of the relative strength of actions never made by the current
expert are still important, since future experts will use the evaluations of
all available moves to guide their search." (Note for citers: the contrast
is TPT vs **CAT**; "Chunked Rollout Targets" is not in this paper.)

*Fit: our diagnosis, written in 2017.* `expert_loop.py` is online ExIt with
a population beam instead of MCTS, and our BC target is CAT - a point mass
on the planner's action index. Our reported quality metric is
`hit = (padded.argmax(-1) == act).mean()`: **the accuracy metric that was
blind to a 50-Elo gap in the paper that ran the experiment.** The second
quote bites too - the planner proposes *from the policy*, so a policy that
never learns which non-elite actions were nearly-as-good hands the next
round's expert a worse prior: the "planner is now slower than the policy it
proposes from" symptom.

**1.14 MCTS as regularized policy optimization (Grill et al. 2020,
arXiv 2007.12509).** What the visit-count policy *is*. With
`pihat = (1 + n_a)/(|A| + sum_b n_b)` and
`lambda_N = c*sqrt(sum_b n_b)/(|A| + sum_b n_b)`, AlphaZero's selection is
greedy ascent on `q^T y - lambda_N * KL[pi_theta, y]`, maximised by
`pibar = lambda_N * pi_theta/(alpha - q)`; visit counts track it at
`||pibar - pihat||_inf <= (|A|-1)/(|A|+N)`. Three named reasons `pihat` is
poor at low N: a new high-value leaf takes many simulations to appear in the
counts while `pibar` reflects it instantly; `pihat` is a ratio of small
integers with limited expressiveness; and the prior is only ever improved
for actions sampled at least once. The decisive ablation (Ms Pacman, 8
seeds): at `N_sim = 5` the gain is almost entirely **LEARN** (distil
`pibar`, not `pihat`), at `N_sim = 50` almost entirely **SEARCH**, and the
two converge once `N_sim >= 24`.

*Fit: it says which end to fix first, and it is the target end.* Our "visit
count" is a single elite lineage - `N_sim` effectively 1 in target space,
the extreme of the low-budget regime where their answer is unambiguous.
`lambda_N` also supplies the temperature we lack: a KL toward a point mass
is the `lambda -> 0` limit, policy improvement with no trust region. Their
third failure mode is literally J1 - actions the search never samples get no
gradient, ever, so a manoeuvre with zero proposal mass can never acquire any.

---

## 2. Synthesis

### (a) What our loop is, in *Zero terms

`tools/expert_loop.py` is **online Expert Iteration with a truncation-
selection population search in place of MCTS, and CAT in place of TPT.**

| AlphaZero / ExIt component | ours | status |
|---|---|---|
| perfect model for search | `src/` C core + `set_state` | **have it, better than theirs** |
| expert = search with the apprentice as prior | `beam_tas.py`: 2,048 lockstep lineages, proposals i.i.d. per decision, elites cloned every R = 25 decisions at `--elite-frac 0.25` | a **flat, open-loop** expert |
| leaf evaluation by the value net | `--score dv`: `V(s)` inside 20,000 u | **partial** - V ranks, never backs up |
| policy target from search | the best line's action index | **CAT, not TPT** |
| value target from search / outcome | none | **absent** |
| root exploration of the prior | `--eps` (killed 6/6 arms), `--dedup` | absent in usable form |
| reanalyse old data with the current net | `expert_dagger.py`, argmax labels | **partial** |
| tree, node reuse, value backup | none (the population re-centres only under `--commit`) | absent |

Three missing parts, in the order the literature ranks them.

**1. The target, not the search** (Grill's Fig. 4 at low budget; ExIt's
50 Elo at identical accuracy). We discard everything the search computed
except one argmax per decision. That is also the cleanest account of the
measured plateau: the loop compounds 77.7 -> 75.3 s then sits **1.5-2.4 s
behind its own planner at "98% per-head BC agreement"** - and
`0.98^6 = 0.886` per decision, i.e. ~285 of ~2,500 decisions differ from the
planner every run, with no signal about which differences were cheap.

**2. A value target from the search.** The critic is trained only by GAE on
its own rollouts, yet it is the object `--score dv` ranks the endgame with -
the region where the geodesic field is known to lie. AlphaZero's `(z - v)^2`
has no analogue in our loss, and the exact return-to-go along a *completed
planner line* is free.

**3. The proposal distribution.** Grill's third failure mode and the
measured 0/1,536 at J1 are the same statement; Sampled MuZero and Gumbel
MuZero both answer it with a small, well-chosen sampled *set*.

### (b) Ranked proposals

Costs are engineer-hours. Every falsification is one arm against a same-card
control, judged on **wall-clock finish time and finishes-out-of-9** (the
checkpoint finishes, so CLAUDE.md section 3 applies) or on **the planner's
floor** (74.70 s at K=3 / 75.34 s at K=4 at 7.63 ms; 76.56 s at 10 ms).

**P1 - Held-key macro-actions as the search's sampled set** (Sampled
MuZero's `beta`, Gumbel's `argtop`; brief item iii). *Code:* replace the
i.i.d. draw in `beam_tas.SampledTorchPolicy._sample`
(`tools/beam_tas.py:428`) with a per-env macro - draw `(side, duration)`,
`duration` in {4, 8, 16, 32} decisions, hold the side key and the yaw sign
for it, re-draw the other heads per decision - using a countdown buffer
beside the existing `hist`. Prior over macros from the policy's own
marginals plus a flattened second source as EZ-V2 does; **never uniform
noise** (round 27's `--eps` DNF'd 6/6: "uniform noise does not discover
ramps, it discovers falling"). *Why:* the record holds keys **0.418 s =
13.7 decisions** at K=4/7.63 ms; our policy holds ~4.5 in expectation, so an
i.i.d. proposal produces one 14-decision coordinated hold with probability
`0.78^13 = 0.040` before the yaw-agreement factor, and a run needs ~30 in
sequence. That is how the planner inherits 2.14 flips/s from the policy it
proposes with. *Effect:* on the 3.70 s execution gap directly; plausibly on
J1, since a 0.4-0.7 s held turn is the class of perturbation "no
perturbation of free-flight controls reaches". *Falsification:* free - the
J1 fork probe already scored **0 of 1,536**; rerun it with macro proposals
(~2 CPU-minutes), then a full-line wave against 74.70 s. *Cost:* **4-8 h.**

**P2 - Search-derived policy target (TPT / Gumbel `pi'`) instead of CAT**
(brief item i, policy half). *Code:* (a) *TPT, cheap*: in
`surfgym/dagger.py::relabel_windows`, `hist[0]` already holds every copy's
first decision and the elite ranking already exists - emit the
survivor-weighted empirical distribution over first decisions per head
instead of the winner's index. (b) *Gumbel*: draw `m = 16` root macros by
`argtop(g + logits, 16)`, split the 256 copies over them by Sequential
Halving, emit `pi' = softmax(logits + sigma(completedQ))` with
`c_visit = 50` and `c_scale` starting at **0.1** (the `mctx` default, not
Go's 1.0) after normalising the window score to [0,1]. Then
`save_bc_dataset` (`python/surfgym/bc.py:338`) gains a `probs` array,
`BC_VERSION` -> 2, and `bc_loss_fn` (`train_fast.py:7445`) swaps its
`gather` for `-(target * log_softmax(padded)).sum(-1).sum(-1)`. *Effect:* on
the 1.5-2.4 s distillation plateau, which exit30's own note calls the
binding constraint on the timer track. Report the **target entropy and joint
per-decision agreement**, never `head-acc`. *Falsification:* one
`expert_loop` round with `--dagger-k 600`, argmax vs distribution, same
train budget. *Cost:* **6-10 h** for (a), +6 h for (b).

**P3 - AlphaZero's `z`: a value target on the planner's line** (brief item
i, value half; and the honest answer to item vii). *Code:*
`tools/plan_to_bc.py` already replays every kept line open-loop through the
same core - accumulate the exact per-tick reward and store the discounted
return-to-go (`gamma_eff` from `surfgym.tick`, `--time-pen`, shaping scale
`100/d0`, `+50`). `bc_loss_fn` already computes the value and **discards
it** (`logits, _ = policy.forward_split(...)`), so the term is
`0.5 * w * (v - z)^2` into the same backward. *Why not redundant with GAE:*
on a finishing line the return is exact and bootstrap-free over ~2,500
decisions. *Effect:* indirect but on both problems; `--score dv` ranks the
endgame with `V` and J1 sits inside `v_switch`. *Falsification:* same-round
A/B plus the free `V`-vs-`d` rank-disagreement scan (round 27). *Cost:*
**3-5 h.**

**P4 - Reanalyse the reservoir** (brief item iv: MuZero Reanalyse + ReZero
entire-buffer). *Code:* dump B = 2,048 reservoir states per round and feed
them to `expert_dagger` as a fourth source beside `SRC_GREEDY / SRC_STOCH /
SRC_SPINE`, emitting P2's and P3's targets. Follow the pseudocode's split -
elite lines are the `DemonstrationBuffer`, the reservoir the
`MostRecentBuffer` - and weight the value target down to **0.25** against
1.0 for policy. ReZero's **backward view** applies: the windows overlap
~6-fold, so reversing the order and terminating any lineage that re-takes
the logged action should recover something like its 54% of tree-search
calls. *Effect:* covers the states the elite line never visits - the stated
cause of the plateau. *Cost:* **8-12 h** (after P2/P3; needs a reservoir
dump path in `train_fast.py`).

**P5 - Gumbel root search with the real simulator, commit-1** (brief item
ii). *Code:* `beam_tas.commit_search` is already receding-horizon with
`--commit H --commit-frac`; set C = 1, put a Gumbel top-k root over P1's
macros in front of it, Sequential-Halve instead of one boundary rank, and
bootstrap with `V` at the leaf. *Affordability:* a beam wave is 2,048 x
~7,800 = **1.60e7 env-ticks** (~11 s at 1.62e6/s). A per-decision search
costs `n*D*K` ticks per decision; over 2,500 decisions that is **0.32 waves
at n = 32, D = 16 decisions (0.49 s of lookahead)**, **1.3 waves at n = 64,
D = 33 (1.0 s)**, **5.3 waves at n = 256**, and 16.5 waves at AlphaZero's
n = 800. The plan phase currently runs **26** waves, so per-decision search
at these widths is affordable outright. The catch is **batch shape**: n
copies fill only n of 2,048 envs, so one trajectory wastes 8-64x unless
several independent roots run in lockstep - the shape `expert_dagger`
already uses (256 copies x 8 states). *Where a tree beats the population
beam:* on the 3.70 s execution gap, where the beam's kept lineages are
**byte-identical on 95.4% of decisions** - the population has collapsed to
one lineage and the 2,048-fold width is wasted. *Where it does not:* at J1,
where the failure is zero *proposal* mass and a tree samples from the same
prior. *Cost:* **2-3 days**; last because its advantage over truncation
selection is unmeasured here.

**Runners-up.** ReZero's backward view alone (fold into P4 - cost, not
capability). PriorZero's root-only prior mixture with a non-language prior,
e.g. an `hlstrafe` `MaxAccelTheta` controller at `alpha = 0.5` into the root
proposal only. Grill's SEARCH variant, which their own ablation says pays
only at `N_sim >= 24` and which presupposes P5. ScaleZero's plasticity triad
on the 2.9x-norm checkpoint.

### (c) Do not do

* **Do not learn a latent world model, and do not add a consistency /
  reconstruction / value-equivalence auxiliary** (MuZero `g`,
  EfficientZero's SimSiam term, UniZero's transformer, ObjectZero's GNN,
  ScaleZero's MoE). We have a bit-exact model at 1.62e6 env-steps/s with
  save/restore, and imagined observations are *rendered* exactly.
  EfficientZero's own ablation puts 55-69% of its result in the component
  that repairs a defect we do not have, and UniZero reports "negligible
  impact" for decoder regularisation even where a learned model exists. This
  is the answer to brief item v: **value-equivalence ideas add nothing when
  the model is real.**
* **Do not run full-width UCT over 3,780 joint actions**, or enumerate the
  joint space anywhere. Sampled or Gumbel subsets only: K = 3 sufficed on
  21-dim humanoid, K = 50 on 362-action Go, m = 16 is Gumbel's default.
* **Do not widen the proposal with uniform noise.** Round 27 ran it (`--eps`
  0.05/0.15/0.30 x 5 s/10 s windows, commit-and-replan MPC) and **all six
  arms DNF'd**. Widen `beta`, not the executed action.
* **Do not adopt UniZero / ScaleZero / ObjectZero / PriorZero** (brief item
  vi): one map, one task, no objects, no language, and an observation the
  temporal survey already showed is Markov for the strafe problem.
* **Do not replace PPO with a MuZero-style off-policy learner.** Thirty
  rounds of calibration sit on this trainer and the single-seed noise floor
  is 27% at 750M steps; the swap would be unmeasurable. Everything above
  enters through the existing `--bc-file` auxiliary, already proved
  byte-identical when absent.
* **Do not use raw visit/survivor counts as the target at tiny budgets.**
  Grill's three reasons bite hardest at low N; prefer
  `softmax(logits + sigma(Q))`.
* **Do not report `head-acc` as evidence about a distillation change.** ExIt
  measured 47.0% vs 47.7% top-1 error across a 50-Elo gap.

### (d) The first experiment

**P1, with a free pre-check before it costs a GPU-minute.** The
pre-registered number to beat exists and is zero.

1. *Free, CPU, ~2 minutes per run.* Re-run the J1 fork probe with macro
   proposals from the replayed prefix (`--macro-hold` is the flag P1 adds;
   everything else on this line exists today):

   ```
   python tools/beam_tas.py <best xENT131 ckpt> \
       --map C:/RL_Surf/maps/surf_src_cannonball.bsp \
       --prefix-line <B5 line>.npz:<J1 entry tick> --envs 512 \
       --tick-ms 7.63 --act-every 4 --score dv \
       --macro-hold 4,8,16,32 --out-dir scratchpad/zero/j1_macro
   ```

   Read-out: record-like continuations out of 1,536 (three seeds x 512)
   against the measured **0 of 1,536**, plus the kept lineages' A/D flip
   rate and median key hold via `tools/demo/compare_wr.py` (targets 0.42/s
   and 0.418 s).
2. *If non-zero, or if the cadence moves at all:* one full-line wave from
   the map start, `--envs 2048 --greedy-envs 64`, scored on the **planner
   floor** (74.70 s at K=3 / 75.34 s at K=4).
3. *Only then* an hour of GPU: one `expert_loop` round with the macro
   planner, warm from the current best finisher, scored on greedy best
   finish time and finishes-out-of-9 against the same-card control - which
   **drifts up** (78.24 -> 78.69 s pooled), so the baseline is moving.

Why this and not P2 or P3: it is the only proposal with a free,
pre-registered falsification and a zero to beat, it attacks the larger gap
(3.70 s against 1.42 s), and P2/P3 improve only *how well we copy the
planner* - worth at most the 1.5-2.4 s we trail it by, while the planner's
own line is itself ~2.1 s behind the record for the reason P1 addresses. Fix
the expert first: ExIt can only distil what the expert finds.

---

## Appendix: arithmetic

Wave = 2,048 x ~7,800 = 1.60e7 env-ticks; plan phase 26 waves in 654 s =
6.35e5/s amortised, 1.62e6/s in-search. Search = `n*D*K` x 2,500 decisions:
n=32/D=16/K=4 -> 5.1e6 (0.32 waves); n=64/D=33 -> 2.1e7 (1.3); n=256 ->
8.4e7 (5.3); n=800 -> 2.6e8 (16.5). Macro cadence 0.418 s / 30.7 ms = 13.7
decisions; `0.78^13 = 0.040`. BC `0.98^6 = 0.886` per decision -> ~285 of
2,500 differ. Discounting (gamma 0.9995/tick over 7,800 ticks): the `+50`
bonus is worth **1.01** at spawn against ~100 undiscounted of shaping and
9.80 discounted of time penalty, and finishing 1 s earlier multiplies the
remaining discounted shaping by 1.051 - time *is* visible in the return,
which is why brief item (vii) resolves to **nothing in the *Zero family
beats the geodesic shaping for long-horizon credit here, except an exact
bootstrap-free `z` on completed lines (P3).**
