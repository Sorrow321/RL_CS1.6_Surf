# Temporal structure in the policy: why memory, frame stacking and action chunking all did nothing, and what to do instead (2026-09-04)

Commissioned by the owner: *"One thing that really bothers me is time
consistency. We tried to give multiple stacked images input, tried to predict
seq of actions, tried to use GRU, and nothing showed improvement. I wonder
why. Currently single timestamp solution makes it very hard for agent to
strafe. It requires to learn quite high freq vector field, because agent
doesn't have any memory."*

Literature verified against primary sources (arXiv, PMLR, JAIR, Nature,
published code), plus this project's ledger, plus new measurements made for
this document on trajectories already in the repo (CPU only, no training, no
GPU, no rentals). Companions: `docs/research-litsurvey.md`,
`docs/research-results.md`, `docs/action-chunks-design.md`,
`runs/research/{wr_demo,gru_probe,gaze30}/`.

New here: a closed-form analysis of what the strafe control problem is at our
decision rate (1.1-1.2), and a measurement of *coordination* between the yaw
head and the strafe-key head in our finisher against the world record (1.4).

---

## 1. Diagnosis

### 1.1 The strafe is a first-order, fully observed, DEADBEAT-stable problem

`src/pm.c::pm_air_accelerate` at `sv_airaccelerate 100`, `maxspeed 250`,
`frametime 0.01`: `accelspeed = min(100*250*0.01, 30 - c)`, `c = v . wishdir`.
The first term is 250, two orders above the 30 u/s air cap, so

    v' = v + (30 - c) * wishdir,   |v'|^2 = |v|^2 + 900 - c^2

Gain is maximal at exactly `theta = 90 deg` and positive on `|c| < 30`, i.e.
within `+- arcsin(30/|v|)` of perpendicular - 3.44 deg at 500 u/s, 0.573 deg at
3000. The required view turn is `atan(30/|v|)`, numerically the same numbers,
and the speed gain is 90 u/s per second at 500 falling to 15 at 3000. This is
bit-for-bit the closed form the GoldSrc TAS community uses: `hlstrafe`'s
`MaxAccelTheta` returns `acos((L - accelspeed)/||v||)` with `L = 30` in air and
returns exactly `pi/2` once `accelspeed >= L` (`HLTAS/hlstrafe`); jwchong's
*Half-Life Physics Reference* ch. 6.3 eq. (6.5) has the same case split.

The decisive part is the **error dynamics**. Simulating the exact update (view
turning at `atan(30/|v|)` per tick, one strafe key held, v = 3000 u/s) from an
error `e` off perpendicular, on the side where the impulse still fires:

    e = +0.5 deg -> -0.00002 after ONE tick
    e = +2.0 deg -> -0.00035 after ONE tick
    e = +10 deg  -> -0.00884 after ONE tick, 0.0 after two

The impulse rotates the velocity by exactly the amount that restores
perpendicularity in a single physics frame, from an arbitrary error, at any
speed. **The strafe fixed point is deadbeat stable** - no memory, no phase, no
accumulating state. Under `--yaw-adaptive` (pinned in the scratch baseline
since round 15, `src/env.c` `K_BINS`) the optimal action is not even a function
of speed: it is the constant `k = +-1`.

So the function a memoryless policy must represent for optimal strafing is *a
constant*, not a high-frequency vector field. And because the yaw action is a
RATE re-evaluated against live speed every tick (`src/env.c::surf_yaw_delta`),
holding a decision for K = 3 or 4 ticks costs nothing in strafe efficiency.
(Without `--yaw-adaptive`, at v = 3000 the 0.5 deg bin settles at a 0.073 deg
error and keeps 98% of the gain while the 1.0 bin settles at 0.427 and keeps
44% - the survey's "up to 17% lost to rounding". The flag removes it. Not the
open problem.)

### 1.2 The one asymmetry that matters: the dead branch

Same simulation with the error on the other side, where `c > 30` and
`addspeed <= 0`:

    e = -10 deg, v = 3000:
      error -9.43, -8.85, -8.28, -7.71, -7.14 ...
      speed  3000.0, 3000.0, 3000.0, 3000.0, 3000.0

No impulse fires. Speed is exactly constant and the error decays only at the
free view-turn rate. **On that side the physics gives zero feedback and zero
gain; recovery is pure open-loop turning.** From 85 deg back to the window edge
takes 7.7 ticks = 2.6 decisions at K = 3 and v = 3000 (4.8 ticks at 2000, 1.9
at 1000), and the yaw sign must hold for all of them.

**Escaping the dead branch costs two to three CONSECUTIVE decisions with a
consistent sign, and our policy's median key hold is two decisions.**

### 1.3 The Markov state is already in the observation

`src/env.c::write_obs` scalars 0-11: ego-frame velocity (`o[0]` along yaw,
`o[1]` left), horizontal speed, `onground`, ducked, jump-held, `sin/cos(yaw)`,
pitch, **previous yaw delta `o[10]` and previous pitch delta `o[11]`**.
`atan2(o[1], o[0])` IS the strafe error at float precision, every decision. No
actuation latency, no hidden phase.

The only genuinely hidden state is termination bookkeeping: the stall
detector's running minimum (`rewards.py:735-737`) and the remaining episode
budget. q1physrl puts `time_remaining` in its 6-float state and Linesight
carries elapsed-time-in-window; both are cheap observation columns here.

### 1.4 New measurement: it is COORDINATION, not memory

Definition: airborne ticks with horizontal speed > 200 u/s; `wishdir` rebuilt
from recorded yaw and key bins with `right = (sin yaw, -cos yaw)`
(`src/pm.c::angle_vectors` at roll 0, `pm_air_move` zeroing z); "disagree"
means the sign of the realised per-tick yaw change turns the view AWAY from the
held key's wish direction.

| | WR demo (human) | xQR32 greedy ep 0 |
|---|---|---|
| fast airborne ticks | 6,838 | 7,718 |
| yaw sign vs strafe-key sign DISAGREE | **2.7%** | **12.7%** |
| angle in 85-90.5 deg (gain band) | 87.9% | 78.8% |
| angle < 85 deg (dead branch, no impulse) | **0.3%** | **13.4%** |
| (forward, side) hold, median | 19 ticks (0.19 s) | **6 ticks (0.06 s)** |
| (forward, side) hold, mean | 73.5 ticks (0.74 s) | 20.3 ticks (0.20 s) |
| key changes over the run | 93 | **366** |

Files: `runs/research/wr_demo/wr_cannonball.jsonl` ep 0 (10 ms grid from
`tools/demo/parse_hldemo.py`) and
`C:/RL_Surf/runs/research/xQR32/traj_7405830144.jsonl` ep 0. The ledger's own
`compare_wr.py` uses a narrower band and the timed window and reads 82.6% vs
53.4% inside 89.5-90.5 deg and 0.42 vs 0.09 s median strafe hold - same
direction, same ratio.

Three consequences.

1. **This is a GREEDY eval** (`train_fast.py:1249` takes the per-head argmax),
   so the dithering is the learned mode, not sampling noise. Pearce & Zhu's
   Counter-Strike BC agent (IEEE CoG 2022) hit the same wall from the other
   side: "Selecting movement keys and mouse movement probabilistically produced
   jerky, unnatural movement, so are selected via argmax." We are *already*
   argmax and still dither, so the fix has to be in the training objective.
2. **The failure is a sign mismatch between two heads.** `sample_padded`
   (`train_fast.py:933`) draws six heads independently by Gumbel-argmax; they
   are conditionally independent given the state by construction. Correct
   strafing needs `sign(yaw rate)` and `sign(sidemove)` to agree. A factored
   distribution wanting "either (+,+) or (-,-), never (+,-)" cannot express it,
   and at the symmetric point the gradient for one head is proportional to
   `(2 p_other - 1)`, which is ZERO when the other head is undecided. **The
   left/right symmetry is a saddle for a factored policy.** VPT (Baker et al.,
   NeurIPS 2022) abandoned a factored Minecraft action space for exactly this
   reason; AlphaStar and Metz et al. (arXiv 1705.05035) use autoregressive
   heads. Honest counterpoint: OpenAI Five (2019) states "Action heads are
   computed independently" and won anyway - a design fork, not a law.
3. **The policy is nowhere near deterministic.** xQR32's last logged
   `train/entropy_loss` is -5.14 and `el = -ent.mean()` in `mb_step`, so
   H = 5.14 nats against a joint maximum of `ln15 + ln7 + 2 ln3 + 2 ln2 = 8.24`
   - 62% of uniform at 7.77e9 steps, with advantages normalised to unit std per
   minibatch so `--ent 0.005` pulls against a unit-scale signal. Spread evenly
   that is a 3-way head near p = 0.78, a mean hold of 4.5 decisions.

The gaze audit corroborates this: every run LOCKS onto one of three strafe
equilibria (offset ~0, ~90, ~180 deg) and holds it for the whole flight; RUN
identity explains 99.1% of the variance in backwards flying, PLACE on the map
2.9%. A symmetry broken once, globally, by luck.

### 1.5 Reconciling the three nulls

**Frame stacking** (sF1, round 10): "25k@655M, then STUCK ... NEGATIVE
(velocity already in scalars)". Expected. In POPGym (Morad et al., ICLR 2023,
13 memory baselines under PPO) frame stacking ranks 7th of 13 (MMER 0.190) and
a plain MLP beats every memory model on the near-MDP navigation subset -
tellingly, the MLP's aggregate score falls from 0.067 to **-0.010** once those
tasks are removed. Hausknecht & Stone (2015): "recurrency is a viable
alternative to stacking a history of frames ... [it] confers no systematic
advantage."

**GRU** (round 30 waves 1-2, `runs/research/gru_probe/`): the probe is sound
and "load-bearing" is right - zeroing `h` changes 74-84% of decisions and
collapses the corridor 92k -> 12-17k. But **load-bearing is not informative.**
A recurrent net dropped into an MDP still routes computation through `h`;
ablating it destroys a learned function without showing the function needed
history. Two controls separate them: a parameter-matched memoryless net (DEEP
191.3k at 2.19B vs GRU 104.1k at 2.27B in wave 1; 157.5k vs 119.6k on matched
cards in wave 2) and an explicit minimal memory (5.2). Siekmann et al. (RSS
2020) ran the matched-capacity version on Cassie - LSTM 2x128 vs feedforward
2x300 - and the RNN wins in simulation but not on the real robot unless the
unobserved variable it was meant to infer is actually randomised. A memory
needs something to remember. The GRU arms also ran at 0.75x throughput.

**Chunking** (round 17): code entropy collapsed 4.16 -> 0.98 and 4.15 -> 0.61
in 2 of 2 scratch runs, eval plateaued at ~1,250 u. The predicted failure of
the one variant nobody publishes - a discrete codebook learned ONLINE from
scratch under a max-entropy bonus, no pretrained decoder, no prior. OPAL learns
the primitive space first and keeps it "fixed"; SPiRL replaces max-entropy with
a KL to a learned skill prior; ACT, Diffusion Policy and VQ-BeT fit the decoder
to demonstrations.

### 1.6 Where memory would add information, and the verdict

Redundant: the strafe cycle (1.1-1.3), velocity and its derivative, the
previous view deltas, latency. Genuinely hidden: **ramp geometry outside the
view.** Pitch aims the lidar and nothing else (`src/surfcore.h`), FOV is
120 deg, and in the offset~90 equilibrium the camera points at the wall beside
the flight path. That is the one thing recurrence could carry - and round 30
shows it is cheaper to supply directly: FOV 160x120 positive in both waves
(117.7k vs 95.0k at 2.9B), fixed downward pitch positive (152.7k vs 39.6k at
2.0B), and the deeper conv trunk is the round's largest single effect. Second
and small: time-remaining and the stall margin.

**Verdict.** Right about the symptom, wrong about the cause. There is no
high-frequency vector field and no missing memory. There is (a) a factored
action distribution that must coordinate two signs across conditionally
independent heads, with a saddle at the symmetric point; (b) per-decision iid
noise making a coherent multi-decision strafe exponentially unlikely - GPM
(Zhang, Xu, Yu, ICLR 2022) states it exactly, single-step perturbation gives
"consistent movement [that] decays exponentially with the number of exploration
steps"; and (c) an entropy coefficient leaving the policy at 62% of uniform
entropy after 7.8e9 steps. All three are ACTION-SIDE. Two of the three
treatments (stacking, GRU) worked the observation side; the third (chunking)
ran the configuration the literature says will collapse.

---

## 2. Temporally-correlated exploration for DISCRETE actions

**ez-greedy** (Dabney, Ostrovski, Barreto, ICLR 2021 / arXiv 2006.01782): with
prob eps repeat a *uniform random* action for `n ~ zeta(mu)`, `mu = 2.0`, cap
100 for Rainbow. R2D2+ez-greedy median 17.77 vs 16.44 on Atari-57. **Entirely
value-based - no policy-gradient experiment and no mention of importance
sampling anywhere in the paper.** Their own listed failure mode: "obstacles and
dynamics in the MDP can cause long exploratory trajectories to waste time (e.g.
running into a wall for thousands of steps)" - a literal description of what a
random 6-tuple does in surf. That is defect one in our arm
(`train_fast.py:6346-6356`); defect two is that burst transitions are **dropped
from the PPO loss** (~20% of the batch at eps 0.05, ~75% under
`--spawn-burst 100`). Amin et al.'s survey (arXiv 2109.00157) names the
underlying problem - "Correlating the perturbations over several time steps ...
complicates the calculation of log-ratio policy gradients, as the policy is no
longer Markov" - and **no paper was found that masks held transitions out of
the loss or analyses that bias**, so our workaround is unvalidated.

**What the on-policy literature does instead.** Hollenstein, Martius & Piater,
*Colored Noise in PPO* (AAAI 2024, arXiv 2312.11091): draw the reparameterised
`eps_t` from a colored sequence with PSD `~ f^-beta` (length 1000); sweep
`beta in {-1, 0, 0.2, 0.5, 0.75, 1, 1.25, 2}` over 16 environments, 20 seeds.
**beta = 0.5 wins** - strictly between white and pink - and they "recommend
switching the default from temporally uncorrelated noise beta = 0 to temporally
correlated noise beta = 0.5". More parallel environments make *more*
correlation affordable (optimal beta rises with N_env); we run 2,048. Their
ratio treatment is purely a **marginal** argument: "because we modify the eps
in mu + sigma . eps and eps_t remains Gaussian, the data collection, viewed at
each individual step, remains asymptotically on-policy." Eberhard et al.'s
pink-noise paper (ICLR 2023) recommends beta = 1 but evaluated **MPO and SAC,
not PPO**; gSDE (Raffin et al., CoRL 2021; PPO `sde_sample_freq = 4`,
`log_std_init = -2`) works for the same reason. Both **require a Gaussian
policy** and do not transfer to a categorical head as written.

**The discrete version is published.** Tallec, Blier & Ollivier (ICML 2019)
prove per-step iid exploration collapses to a *deterministic drift* as
dt -> 0 (Thm 4) and give the discrete remedy verbatim: "take `z_dt` to be a
discretization of an (#A)-dimensional continuous OU process, and set
`pi_explore(s, z) := argmax_a (A(s,a) + z[a])`". Their caveat, stated twice:
the study is off-policy and "we do not study the time discretization invariance
of on-policy methods (A3C, PPO, TRPO...)".

**The construction to build.** `sample_padded` already computes
`argmax(logits - log(-log u))`, `u ~ Uniform` - Gumbel-max. Replace `u` with a
per-(env, head, bin) sequence whose **marginal is still exactly Uniform(0,1)**
but correlated across decisions (AR(1) Gaussian copula
`z_t = rho z_{t-1} + sqrt(1-rho^2) eps_t`, `u = Phi(z)`; or Eberhard's
generator). Every step then samples exactly `Categorical(softmax(logits_t))`,
`logp` is exact, **nothing is dropped and the PPO ratio is untouched.** That is
Tallec's argmax-over-perturbed-scores with the perturbation chosen so the
policy marginal survives, which his advantage-perturbation version does not do.
Ledger caveat: the marginal is preserved unconditionally but not conditionally
on `s_t` - exactly the approximation AAAI 2024 makes.

**Two risks**, from Deep Coherent Exploration (Zhang & van Hoof, ICML 2021),
the one paper computing the history-conditioned marginal exactly under PPO
(beta = 0.01 best): **Coherent-PPO underperforms plain PPO on Walker2d**, and
correlated exploration carries a PPO-specific cost independent of the ratio -
such policies "tend to satisfy the KL constraint in fewer update steps, which
leads to slower learning".

**Mapping.** `sample_padded` runs inside the captured CUDA graph
(`step_compute`, `train_fast.py:5532-5558`), where the GRU's `static_h` is
already a persistent in-place tensor - same pattern, one `(N, NACT, NPAD)` fp32
buffer (737 KB at 2048 envs), zeroed at reset beside `ez_left`. Flag
`--noise-beta`, default 0 = bit-identical. ~30 lines.

---

## 3. Temporal abstraction on the ACTION side

**Learned repetition.** FiGAR (Sharma, Lakshminarayanan, Ravindran, ICLR 2017)
adds an independent repetition head over `W = {1..30}`; the loss becomes
`[log pi_a + log pi_x] * advantage`, so the ratio is exact by construction and
**it was demonstrated with A3C on Atari - on-policy and discrete**, beating A3C
on 26 of 33 games (and, per their own ablation, worse on 24 of 33 if `pi_x` is
discarded at evaluation, so the head is load-bearing at execution). Its
limitation is no way to abort a committed repeat; TempoRL (Biedenkapp et al.,
ICML 2021) fixes that by conditioning the skip head on the *chosen action* (max
skip 10 on Atari) and reports FiGAR-DDPG "struggle[s] quite a lot ... already
with only two possible skip-values" without it. Metelli et al. (ICML 2020) give
the theory: "persistence 1 rarely leads to the best performance" (Cartpole
k=4: 239.5 vs k=1: 169.9) yet "excessively increasing persistence prevents the
control at all", with the cost bounded by a **Time-Lipschitz constant** - how
fast the system moves per frame. Surf at 3,000 u/s is a fast system, which is
the theoretical reason long persistence should hurt here.

**PIC** (Chen, Tang, Hao, Liu, Meng, AAAI 2021) is the one paper aimed at our
symptom in a **discrete** action space. Its metric is exactly section 1.4's,
`xi(pi) = E[(1/T) sum_t (1 - I{a_t = a_{t-1}})]`; its mechanism is a mixture
`pi(.|s_t,a_{t-1}) = mu delta(a_{t-1}) + (1-mu) pi_core(.|s_t)` with a learned,
last-action-conditioned `mu`.

**Chunking.** Every system found an interior optimum for the same stated
reason. ACT (Zhao et al., RSS 2023): `k = 100` at 50 Hz, ablation
1% -> 44% -> taper past 200. Diffusion Policy (Chi et al., RSS 2023):
`To = 2, Ta = 8, Tp = 16`; "too long a horizon reduces performance due to slow
reaction time". VQ-BeT (Lee et al., ICML 2024): chunk length **1** on four of
seven benchmarks. Q-chunking (Li, Zhou, Levine, NeurIPS 2025) puts it in online
RL with an unbiased chunked backup needing **no importance sampling**: `h = 5`
default, helps to 10, **h = 50 achieves zero success**; chunking "may perform
poorly in settings where a high-frequency control feedback loop is essential".

**What this says about `--chunk`.** Not evidence against chunking. The fix the
literature prescribes now exists and did not in round 17: the expert-iteration
loop produces **40,761 rows of (state, planner action) per round**
(`tools/plan_to_bc.py`, `surfgym/bc.py`). Fit the decoder to those, freeze it,
run PPO over the code head. OPAL's recipe with our planner as the demonstrator.

**A strafe CONTROLLER as a primitive.** `hlstrafe` exposes `MaxAccelTheta`,
`MaxAngleTheta`, `ConstSpeedTheta` and `MaxAccelIntoYawTheta`; `bxt-rs`
hill-climbs over frame bulks, not per-tick inputs. `--yaw-adaptive` is a third
of the way there. The full primitive - one action meaning "auto-strafe
left/right", under which the engine sets `sidemove` and
`yaw_delta = atan(30/|v|)` with matching signs for the K ticks - deletes the
saddle by construction. **The decisive caveat, and why it ranks fifth:**
jwchong ch. 6.3.1 says greedy per-frame maximisation happens to be globally
optimal for pure air-strafing, "perhaps owing to good luck"; ch. 8.4 says the
opposite on a **surf ramp** - "In isolation, as it turns out, it does not give
us a global optimum. This serves to illustrate the danger of thinking in
per-frame terms as is common in pure strafing." Gate it on `onground == -1`,
free flight only (79.5% of our run), and keep the manual bins.

---

## 4. Action smoothness regularisers

CAPS (Mysore, Mabsout, Mancuso, Saenko, ICRA 2021):
`J = J_pi - lambda_T ||pi(s_t) - pi(s_{t+1})|| - lambda_S ||pi(s) - pi(s_bar)||`,
`s_bar ~ N(s, sigma)`. Real quadrotor: current 22.87 A -> 4.86 A, smoothness
metric down ~96%, and every CAPS agent was flight-worthy where the baseline
needed cherry-picking; both terms are needed. **The lambda and sigma values are
not printed in the paper** - the circulating `lambda_T = 20` is a third-party
restatement. LipsNet (Song et al., ICML 2023) is the architectural alternative
(`lambda = 1e-3`, `K_init = 5`; DMC Reacher fluctuation 2.41 -> 0.04 at equal
return), and only its LOCAL variant is safe.

**Neither applies to a factored categorical head** - both are Euclidean
distances on continuous actuators. A targeted search found **no published
action-smoothness regulariser for categorical policies**; searching for
`KL(pi(.|s_t) || pi(.|s_{t+1}))` returns only trust-region methods, which
constrain KL between successive policy *iterates* - an unrelated object.

**Two independent papers warn against the obvious version.** PIC tested a
`-0.05` per-switch reward and found it "counterproductive" on one of four
driving tasks because it "violate[s] the original reward structure (sparse
reward)"; LipsNet reproduced the same on DMC Cartpole - **a consecutive-action
reward penalty in a sparse-reward environment increases action fluctuation.**
Our reward is a +50 terminal bonus over ~2,600 decisions. **Do not add a
switching penalty to the reward.** Put it in the LOSS, as a symmetrised KL
between consecutive decisions' head distributions, inside the sequence
minibatch `--rnn` already builds (`train_fast.py::seq_loss`, time-major
`idx = t*N + env`).

Counterpoint: the two *continuous* racers that do pay for smoothness in the
reward both won. Swift (Nature 2023):
`r^cmd = lambda4 ||a^omega|| + lambda5 ||a_t - a_{t-1}||^2` (weights
unverified). Vasco et al. (RLC 2024) have **two** terms - a steering-change
penalty `-|theta_t - theta_{t-1}|` and a steering-history sigmoid on
`|delta_t| + |delta_{t-1}|` - added "to discourage the agent to make
inconsistent decisions in a short period of time". Their reward is dense
progress, not our sparse bonus, which is exactly the distinction PIC and
LipsNet draw.

---

## 5. Memory done right, and the cheapest arm we have never run

### 5.1 What the field does

**The superhuman vision-based Gran Turismo agent has no memory.** Vasco, Seno,
Kawamoto, Subramanian, Wurman & Stone (RLC 2024): one 64x64 RGB frame plus 17
proprioceptive floats at 10 Hz; the critic gets 531 privileged course points
(0.1-6 s ahead), because "providing the critic with global features during
training is fundamental ... as it mitigates the partial observability of the
environment". Precision: the paper never states "frame stack 1" or "no RNN" -
it describes a single image and no recurrence and names RNNs as *future work*.
Cite it as absence of mechanism, not as an ablation. Its 17 dims include **the
current steering/throttle/brake plus a three-step steering-angle history and
two steering deltas.** That pattern is universal:

| system | frame stack | previous actions in obs | RNN | decision Hz |
|---|---|---|---|---|
| GT Sophy (Nature 2022) | no | measured steer/throttle/brake | no | 10 |
| Fuchs (RA-L 2021) | no | yes, `delta_{t-1}` | no | 10 train, **60 eval** |
| Vasco (RLC 2024) | 1 image | 3-step steering history + 2 deltas | no | 10 |
| Linesight (TMNF) | 1 grey frame | **yes, 5 previous actions, 20 scalars** | no | 20 (5-tick hold) |
| Swift (Nature 2023) | no | **yes, `a_{t-1}`, R^4 of 31** | no | not stated |
| q1physrl (beat human WR) | no | no (yaw is integrated) | no | 72 |
| Pearce & Zhu CS BC | **tried, rejected** | no | conv-LSTM | 16 |
| **RL_Surf today** | no | **only `o[10]`, `o[11]`** | optional GRU | 25-33 |

Five of eight feed previous actions; one uses recurrence; none stacks frames.
Sophy's "no substantial performance gains from acting more frequently than
10 Hz" over a 5-60 Hz sweep is **an unquantified claim with no figure or table
behind it** - do not lean on it. Pearce & Zhu is the one case where stacking
was tried and rejected in favour of recurrence: it "was successful in aim
training mode, but caused issues when navigating, with the agent often getting
stuck in doors and corners." q1physrl is the closest relative of all - 72 Hz on
Quake air-strafe physics, beat the human record - and it replaces frame skip
with an environment-level constraint: `key_press_delay`, "minimum time in
seconds between consecutive presses of the same key", **default 0.3 s**, plus
`smooth_keys` (half impulse on the first frame after a press). 0.3 s is 10
decisions at 33 Hz, within a factor of two of the WR's 0.19 s median key hold
in 1.4 - the single most directly transferable constant in this document.

### 5.2 `--act-hist K`

Implemented (`python/surfgym/obsaux.py`, commit 64bc359), mirrored in
`record_ckpt.py`, warm-starts by trailing zero-pad so it is function-identical
at step 0, ~0 throughput cost. Its docstring states the hypothesis this
document was commissioned to test: *"air-strafing needs the strafe key and the
yaw change PHASE-LOCKED, and a memoryless policy cannot see the phase it is
in."* **There is no ledger entry for it.** It encodes the last K decisions as
6K signed scalars in [-1,1]; slot 0 at K = 1 duplicates `o[10]`, so K > 1
extends a feature the policy already reads.

DeepMimic makes the same substitution explicit: in the strike task a hand-built
binary flag acts "as a memory for the state of the target ... The memory state
h can be removed by training a recurrent policy, but our simple solution avoids
the complexities of training recurrent networks while still attaining good
performance."

If `--act-hist 4` (0.16 s at 25 Hz, about the WR's median key hold) is null AND
the GRU is null, the GRU's content is not the action history, and the remaining
candidate is out-of-view geometry - which FOV and DEEP address more cheaply. If
it is positive, we have the effect at 1/100 of the GRU's throughput cost.

### 5.3 If recurrence is retried

Ni, Eysenbach & Salakhutdinov (ICML 2022) recommend **separate actor and critic
RNN encoders** ("a shared encoder increases the gradient norm and hinders
learning ... some implementations use the (inferior) shared encoder"), context
length 64, and previous action/reward as inputs. Ours shares one GRU
(`train_fast.py:276-300`) and feeds no previous action. POPGym (PPO, 13
baselines) ranks GRU first (0.349) over LSTM (0.255) and linear transformers
(0.112-0.138), so the GRU choice itself is right. R2D2's burn-in is irrelevant
here: it repairs a hidden state produced by *older parameters* pulled from
replay, and on-policy PPO has no such lag - Pleines et al. (arXiv 2205.11104)
implemented the on-policy analogue and found refreshing "does not provide a
gain in performance that satisfies the gain in computational cost". Given
1.1-1.3 and the DEEP result, expect low value; run 5.2 instead.

---

## 6. Phase priors do not apply; the decision rate is the wrong axis

**Phase.** DeepMimic (Peng et al., SIGGRAPH 2018) feeds `phi in [0,1]` to a
plain 1024/512 MLP at 30 Hz over 1.2 kHz physics, and states why: **"the target
poses from the reference motions vary with time."** Siekmann et al. (ICRA 2021)
generalise it: "To prevent the reinforcement learning environment from becoming
a non-stationary Markov decision process, some information about the periodic
reward function must be present in the state." **A clock restores the Markov
property when the REWARD is periodic.** Ours is not - the geodesic and arc
terms are functions of state - so a phase input would index nothing, and 1.1
shows there is no hidden oscillator phase to recover. DeepPhase (Starke et al.,
SIGGRAPH 2022) is supervised motion synthesis; CPG-RL (Bellegarda & Ijspeert,
RA-L 2022) *ties* end-to-end joint control on tracking (0.494 vs 0.486 m/s) and
claims robustness. Air-strafing is not a gait - the alternation is driven by
ramp geometry, and the WR alternates only 93 times in 68 s. **Do not build a
phase input or an oscillator prior.**

**Rate vs commitment.** Three axes the ledger conflates: physics rate
(`--tick-ms`, more impulses is strictly better, and a 10 ms finisher does not
transfer to 7.63 ms, 0/9); decision rate (`--act-every`, 33 Hz on samples,
25 Hz on wall-clock, "sharp" corrected to "shallow" in round 14); and
**commitment**, which is currently pinned to the decision rate and should not
be. Both halves of the discretization literature say commitment is the knob -
Tallec et al. (ICML 2019): the action-value gap vanishes linearly in dt; Park,
Kim & Kim (NeurIPS 2021): "the variance of the PG estimator can diverge to
infinity" as delta -> 0, fixed by Safe Action Repetition. **Our act-every
ladder varied the rate with the noise held iid, so it confounds rate with
commitment and cannot answer the commitment question at all.**

Every system in 5.1 is already two-timescale - a slow learned level over a fast
hand-written one: Swift emits thrust and body rates to a Betaflight controller;
Vasco's game "linearly interpolates the steering angle between steps"; Fuchs
trains at 10 Hz and **evaluates at 60 Hz with the same weights**; Linesight
latches a discrete action for 5 engine ticks; CPG-RL sets oscillator setpoints
at 100 Hz over a 1 kHz solver, the closest published match to the
strafe-executor idea. Both options papers report the same failure mode for such
designs - option-critic's "termination gradient tends to shrink options over
time ... since in theory primitive actions are sufficient for solving any MDP"
(fixed by `xi = 0.01`), Harb et al. (AAAI 2018) "without a deliberation cost,
the options eventually learn to terminate at every step" (fixed by
`eta in [0.005, 0.030]`). **A slow level needs an explicit term paying it to
stay slow.**

---

## 7. Ranked top five

| # | proposal | one-line rationale | cost |
|---|---|---|---|
| 1 | **Anneal the entropy coefficient** (`--ent-final 5e-4`) | The finisher sits at H = 5.14 of 8.24 nats after 7.8e9 steps; a near-indifferent categorical has an argmax that flips every two decisions, which is exactly the measured dithering - and the eval is already argmax, so the fix must be in the objective. | **0 h** - flag exists (`train_fast.py:1981`, schedule at 6892), never used in any arm |
| 2 | **`--act-hist 4`** | Five racing systems feed previous actions and none uses an RNN; Linesight feeds 5, Vasco 3 steering steps, Swift `a_{t-1}`. Implemented here, written for exactly this hypothesis, never run. Settles what the GRU could not, at ~0 throughput. | **0 h** code, 1 GPU-h |
| 3 | **Condition the movement-key heads on the sampled yaw bin** (autoregressive, zero-init) | Deletes the left/right saddle. `log p(yaw) + log p(side\|yaw)` is exact so PPO is unchanged, and zero-init conditioning is function-identical at step 0, so it runs as a WARM arm on the finisher. | **4-6 h** - `Policy.heads`, `sample_padded`, `logprob_entropy_padded`, greedy path |
| 4 | **Temporally correlated Gumbel noise** (`--noise-beta 0.5`) | The on-policy-exact version of ez-greedy: marginals stay exactly categorical, so nothing is dropped and the ratio is untouched - the two defects that killed `--ez-eps`. AAAI 2024 recommends beta = 0.5 as PPO's new default, and 2,048 envs is the regime where more correlation is affordable. | **3-5 h** - one persistent buffer inside the captured graph, same pattern as `static_h` |
| 5 | **Analytic strafe executor gated to free flight** (`--auto-strafe`) | Removes the yaw/key coordination by construction using the closed form `hlstrafe` ships; free flight is 79.5% of the run. Gate on `onground == -1` - jwchong shows greedy per-frame optimality FAILS on a ramp. | **1-2 days** - `src/env.c` semantics + scratch run; CPU pre-test is free |

Runners-up: TempoRL-style action-conditioned repetition head (3); chunking
redone OPAL-style with the planner's 40k BC rows as a frozen decoder (3); a
CAPS temporal term as a KL in the LOSS, never in the reward (4); the two
missing observation columns, time-remaining and stall margin (1.3).

**Falsification.** Arms 1-3 are warm-startable on
`C:/RL_Surf_exit/runs/exit_seed/xQR32_scalar.pt`, and that checkpoint FINISHES,
so per CLAUDE.md section 3 the metric is **wall-clock time from start to
finish** plus finishes-out-of-9, not `race/eval_progress`. One hour, one seed,
same card, matched control. The mechanism check is `tools/demo/compare_wr.py`
on the final trajectory (sign-disagreement, gain-band share, key-hold length) -
it is what makes a one-seed arm interpretable: an arm that moves the finish
time without moving the switch rate did not work for the stated reason. Arms 4
and 5 change exploration or action semantics, so use `SCRATCH=1
tools/run_arm.sh` with a same-card control, scored on the gate ladder (which
gate, at what step) per CLAUDE.md's RETRACTION section. Arm 4 has a free
pre-test: log the realised key-hold distribution over 1e7 rollout steps before
spending an hour, and watch the minibatch steps before PPO's early stop.

## 8. Do not retest

* **Frame stacking.** POPGym ranks it 7/13 under PPO; Hausknecht & Stone find
  no systematic benefit over recurrence; our velocity is already a scalar.
* **ez-greedy as implemented** - uniform random 6-tuple bursts with the
  transitions dropped. Dead structurally (2); do not retune eps.
* **A 7 s temporal mini-race window / a shorter discount horizon.** Already
  dead; nothing here reopens it.
* **A phase input or an oscillator prior.** Dead on the physics (6).
* **A per-step reward penalty for switching actions.** PIC and LipsNet
  independently report it backfires under sparse reward - our regime.
* **Chunking with an online, from-scratch codebook under max-entropy.** 2/2
  collapses here, zero published support for that variant.

## 9. Where the literature disagrees with the ledger

1. **"Stacking, GRU and chunking all failed, so temporal structure does not
   help."** Three different hypotheses, none of which tested action-side
   temporal correlation. The whole action-side family - FiGAR (on-policy,
   discrete), TempoRL, PIC, Q-chunking, colored noise - is untested here.
2. **"ez-greedy is a clear negative, so temporally-extended exploration does
   not fit on-policy PPO."** AAAI 2024 recommends correlated noise as PPO's
   *default*. Our two failure modes - off-manifold uniform bursts, up to 75% of
   the batch discarded - are both avoidable, and **masking held transitions out
   of the loss has no published precedent or bias analysis at all.**
3. **"Every rate off 33 Hz is worse; a sharp optimum."** Already softened to
   "shallow"; more importantly the ladder held the noise iid while varying the
   rate, which is precisely Tallec's confound. Reopen rate and commitment
   jointly or not at all. (Sophy's 5-60 Hz sweep is published with no numbers -
   do not cite it as a measured result.)
4. **The gru_probe's framing.** "Load-bearing" is shown; "carries information
   the memoryless net lacked" is not, and the probe cannot show it. Siekmann et
   al. (RSS 2020) is the matched-capacity control we are missing.
5. **"Bigger nets: validated dead (Nexto)"** in `research-litsurvey.md` s.8 is
   overturned by round 30 - the deeper conv trunk is the largest effect
   measured. Update the dead list.
6. **`--act-hist` is missing from every plan** despite being implemented, free,
   and the single most common temporal feature in the racing literature.

---

## Appendix: how the 1.4 numbers were produced

CPU only, no checkpoint, no map, no GPU. Read the two JSONL trajectories
(`[t, x, y, z, vx, vy, vz, yaw, buttons, onground, progress, reward, pitch,
fwd, side]`, `python/surfgym/record.py`), keep rows with `onground == 0` and
`hypot(vx, vy) > 200`, build `forward = (cos yaw, sin yaw)`,
`right = (sin yaw, -cos yaw)`,
`wishdir = normalize(forward*(fwd-1) + right*(side-1))`, and report
`acos(v . wishdir / |v|)` in degrees plus the sign of
`wrap180(yaw[t+1] - yaw[t])` against the sign the held key requires (`side > 1`
requires a negative yaw change). Holds are runs of constant `(fwd, side)`. The
simulation in 1.1-1.2 iterates
`v <- v + max(0, 30 - v . wishdir) * wishdir` with the view turned by
`k * atan(30/|v|)` per tick - `src/pm.c::pm_air_accelerate` and
`src/env.c::surf_yaw_delta` at `sv_airaccelerate 100`, `maxspeed 250`,
`frametime 0.01`.
