# Literature survey: RL for high-speed navigation/racing (2026-08-21)

Commissioned survey (run on a research subagent) of every system that
reached or beat top-human performance on long-horizon continuous-motion
tasks, mapped onto RL_Surf. Full agent report preserved below verbatim
(ASCII). Companion distillation and decisions live in
research-results.md.

---

## 0. THE HEADLINE: four convergent facts, and where we sit

Every superhuman system independently converged on four things. We have
two; we are an outlier on the other two.

| | Field consensus | RL_Surf today |
|---|---|---|
| Effective discount horizon | 1.4 - 13 s (q1physrl 1.4 s, Fuchs 5 s, Sophy 9.6 s, vision-GT 10 s, Necto 13.3 s, Linesight gamma=1 over a 7 s window) | 60.6 s (1/(1-0.9995) = 2000 decisions at 33 Hz). Largest single structural outlier in the whole survey. |
| Explicit lookahead geometry in the observation | Universal. Sophy: 60 ego-frame points per track edge spanning ~6 s scaled by current velocity. Fuchs: 10 curvature samples at 0.2 s intervals. Linesight: 40 virtual checkpoints in ego frame. Swift: next gate's 4 corners. Vision-GT: 531-dim course points to the critic only. | Absent. 64x32 depth image only. Sophy's ablation: removing the course-point sequence costs +2.64 s on a 114 s lap - the largest single ablation in that paper. |
| Multi-source, difficulty-biased start-state distribution | Universal, never a single source. Necto 70% human-replay / 8% smart-random / 4% true start / 18% hand-built scenarios. Swift: random gate + bounded perturbation of a previously observed passing state. q1physrl: 1% true starts. | 90% self-snapshot / 10% true start. Right idea, single source, uniform-ish weighting. |
| Distributional critic | Both superhuman racers (Sophy QR-SAC 32 quantiles, Linesight IQN). Sony: without the QR head Sophy is not faster than the best human on Maggiore (+0.69 s ablation). | Scalar PPO critic, on a task with 4 catastrophic-failure branches. |

Two things we already have right and should not touch: potential-based
shaping on a progress coordinate (Song/Scaramuzza: RL beats optimal
control precisely because it optimizes task progress rather than
tracking a reference line - 100% vs 0% success under a realistic
model), and state-restore respawns (Go-Explore: save-state restore is
~45x faster than replaying actions).

## 1. Gran Turismo Sophy (Nature 602:223, 2022)

https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/nature22.pdf

Superhuman time trial (Maggiore 114.249 vs 114.466 human best; 200-lap
std 0.061 s). QR-SAC, n-step 7, gamma 0.9896 at 10 Hz (9.6 s horizon),
2048x4 net. Reward: course progress per 0.1 s (w 1), wall contact
-(dt)*(kph)^2 (w 0.01), off-course same (w 0.01, progress MASKED while
off course), tyre slip (w 0.25), collision (w 4-5). NO lap-time term,
NO finish bonus - time falls out of discounted progress. Penalties
quadratic in speed (switched to linear on Sarthe "to avoid an explosion
in values"). Obs: no pixels - proprioception + 180 ego-frame track-edge
points spanning ~6 s at current velocity. Ablations (Maggiore): no QR
head +0.69 s; 1-step +1.48 s; no course points +2.64 s. "No substantial
gains acting above 10 Hz." 10 PS4s x 20 cars for time trial; 1 h to
lap, ~8 h to top-10% human, 8-10 more days to superhuman.

Fuchs et al. GT Sport (RA-L 2021, arxiv 2008.07971): superhuman vs
52,303 humans with a TWO-TERM reward: r = delta(course progress) -
5e-4*|v|^2 on wall contact. SAC, gamma 0.98 at 10 Hz (5 s horizon),
2x256 net (600k params), 13 rangefinders + 10 curvature samples ahead.
Start: uniformly distributed on track at 100 km/h. NEGATIVE result:
behaviour-cloning warm start does not improve final performance -
scratch overtakes it within an hour.

Vasco et al. (RLC 2024, arxiv 2406.12563) - vision-based superhuman GT,
closest match to our observation constraint. Actor: 64x64 RGB + 17
floats, frame stack 1, NO RNN. Critic: same PLUS 531-dim privileged
course points (0.1-6 s ahead). Ablations: no image = "unable to drive";
SYMMETRIC critic (critic on images too) = significantly degraded;
"providing the critic with global features during training is
fundamental." Start states: uniform over track incl. slightly
off-course, random launch speed 0-104 km/h. gamma 0.99, n-step 3, 200
quantiles. See also asymmetric actor-critic (Pinto, arxiv 1710.06542).

## 2. Swift / drone racing (Nature 620:982, 2023)

https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/

Beat 3 human champions 15/25 races. PPO, 100 agents, 1e8 steps in ~50
min on one workstation, 2x128 MLPs. Reward: gate-progress delta +
PERCEPTION term (exp of angle between camera axis and next gate - keeps
the observation informative; our aimable camera has the same failure
mode) + action smoothness - crash 5.0. No time penalty, no finish
bonus, no demos, no reference line. Start states: "random gate, bounded
perturbation around a state previously observed when passing this gate"
= our respawn reservoir, independently invented.

Song/Scaramuzza (Science Robotics 2023, arxiv 2310.10943): RL tracking
a precomputed time-optimal line: 44% success nominal, 0% realistic;
same RL on gate progress: 100%, beats professionals. "RL outperforms OC
because it optimizes a better objective." NEVER replace the geodesic
potential with reference-line tracking. Their start scheme: uniform
over all segments first, successful-state buffer second. Their reward:
r = ||g-p_{t-1}|| - ||g-p_t|| - 0.01*||omega||, collision -10, done +10.

## 3. Linesight / Trackmania (github.com/Linesight-RL/linesight)

Beat world records on ~10 of 12 official campaign tracks (May 2024).
100 Hz physics, 20 Hz decisions. IQN (distributional), 12 discrete
actions. Reward constants: -0.0012/ms time penalty + 0.01/m centerline
progress + clipped potential to next virtual checkpoint. NO finish
bonus. THE KEY TRICK - "temporal mini-race": rollouts truncated to a
7 s window (140 actions), elapsed-time-in-window is IN the observation,
gamma schedule ends at exactly 1.0 - return = -elapsed_time + progress,
no discount/horizon tension. Stored transitions get random horizon
clocks at sample time (abs() fold oversampling t~0 and t~H). Obs
includes 40 ego-frame virtual checkpoints (~400 m lookahead) from a
reference line that "does not need to be fast... usually the
centerline"; later reference lines are the AI's own previous runs
(self-bootstrapping). Early termination: 2 s without passing a
checkpoint (ours: 15 s). "Training wheels": physics-derived skill-
quality rewards (speedslide_quality = fraction of physically maximal
gain) shipped as schedules annealed to zero once mastered. Epsilon +
Boltzmann exploration schedules; each memory used 32x before discard.
Prior attempts for calibration: LIDAR-only agents "could plan for the
next turn, not two turns ahead"; multi-map generalization brutally
expensive vs single-map records.

## 4. Rocket League ecosystem (Necto/Nexto, Seer, Opti, RLGym)

Necto state setter: 70% human-replay states / 8% smart-random / 4%
true start / 18% scenarios, and replay states weighted 1 +
10*(heights)/CEILING = ceiling states 11-21x oversampled: the answer to
"visit the rare hard regime" is oversample snapshots by a difficulty
statistic. Hard mechanics (flip resets) were reward-shaped at
goal-magnitude, gated on tight preconditions - "most mechanics aren't
discovered on their own." Anti-farming idioms: cross-weighting,
min(a,b) of independent conditions, decaying event multipliers,
KRC geometric-mean AND. Distance potentials are exp(-d/max_speed) =
time-to-go, not distance-to-go. Per-skill horizons: Opti sets gamma
per skill via half-life seconds (recovery 2 s ... general play 16 s);
whole ecosystem parameterizes gamma in SECONDS. Seer BC-from-replays:
pure BC cannot play; as init it helps then REGRESSES (human-prior local
optima); fix = KL penalty toward imitation policy with decaying coef.
Nexto (bigger net) lost 300-4 to Necto at matched steps. 2% of episodes
get infinite boost (explore unaffordable maneuvers, transfer back).
Symmetry augmentation at reset = free samples.

## 5. Source/GoldSrc movement RL + the air-accel math

No public RL agent for Source/GoldSrc surf or bhop found (RL_Surf
appears to be first; caveat: thorough not exhaustive). Closest:
q1physrl (github.com/matthewearl/q1physrl) - Quake 1 physics reimpl +
PPO BEAT THE HUMAN WR on the 100m strafe map, 150M steps, 1 day CPU.
gamma 0.99 at 72 Hz = 1.4 s horizon. Acts EVERY physics tick; instead
of frameskip, a 0.3 s minimum key-dwell constraint. 99% of episodes
start from randomized mid-run states (yaw 0-360, speed 0-700, and
RANDOMIZED TIME-REMAINING); 1% true starts. Reward = dt * v_y.
Squashed (CDF) action distribution, not clipped Gaussian.

Pearce & Zhu CS deathmatch BC (arxiv 2104.04258): mouse discretized on
a NON-UNIFORM grid, finer near zero - classification beat regression.

Air-accel math for OUR config (aa=100, maxspeed 250, msec 10):
gamma1 = 250 >> L = 30 - permanently saturated regime. Optimal wish
direction EXACTLY perpendicular (c=0); gain window |c| < 30/nu
(+-3.44 deg at 500 ups, +-0.86 at 2000); speed grows as
sqrt(nu0^2 + 90000 t); required yaw rate omega* = arctan(30/nu) =
0.86-3.44 deg/TICK over the whole speed range. Our YAW_BINS
{0,.25,.5,1,2,4,7,10} cover that band with 3 ratio-2 bins - up to ~17%
of the air-accel term lost to rounding, speed-dependent. Fixes: (a)
geometric rebin inside 0.3-4.6; (b) better: reparameterize omega =
k * arctan(30/nu), k in [-2,2] - "strafe optimally" becomes the
CONSTANT action k=+-1. Add proprioception: omega*/rate_max, gain-window
coordinate nu*cos(theta)/30, realized delta-nu last tick. NOTE: our
action is a yaw RATE integrated per tick (env.c:518), so held actions
keep turning - act_every=3 costs almost nothing in strafe efficiency
(the act-every ladder measured reaction latency, not strafing). BUT
strafing is not where the 15% lives: a 45-deg ramp pays 400 ups^2 vs
air-strafe's 90 ups^2 at 500 ups (4.4x, 8.9x at 1000) - the dominant
terms are grazing entries (PM_ClipVelocity destroys (v.n)^2), staying
low/steep, and entry/exit geometry. Consider a speed-squared contact
penalty -c*delta(0.5 v^2) at Sophy's relative scale (~0.01).

## 6. Exploration/initialization literature

Go-Explore (arxiv 1901.10995, Nature 2004.12919): final cell-selection
weight is just W = 1/sqrt(C_seen + 1) (domain-knowledge weights added
nothing); +1 floor prevents starvation; times_chosen_since_new resets
on discovery (productivity counter); frontier term outweighed all count
terms in the tuned config. Post-return exploration: random actions with
95% REPEAT probability. 3D-game variant (arxiv 2209.00570): >90%
navmesh coverage vs RND <10% vs random <5%. Pathologies = our symptoms:
detachment (intrinsic reward is consumable) and derailment (long
precise sequences derailed by stochasticity) - the 4 fail-nets are
textbook derailment.

Reverse curricula: Salimans&Chen (1812.03381) - reset window {tau-D..tau},
advance at 20% success; beat its own demo; COLLAPSES 8x under 1% action
noise if point-start (always sample a window). Backplay (1807.06919):
suboptimal demos BEAT optimal demos (54% vs 31%); discovers strategies
absent from the demo; fails if advanced too fast. Florensa (1707.05300):
good starts = success in (0.1, 0.9); reserve 1/3 of starts for mastered
regions (anti-forgetting - plausible hidden contributor to our seed
variance). DeepMimic: the widely-miscited ablation - EARLY TERMINATION
carries the return (0.379 vs 0.791 without it), RSI matters
qualitatively for flight-phase skills. Audit: death must be genuinely
zero-value-absorbing.

Why RND failed (validation): Taiga ICLR 2020 (bonus methods = eps-greedy
on 5/6 hard-exploration games); EIPO (P(RND>PPO) = 0.49 over 61 games);
RC-GVF (no bonus under visual aliasing - a repetitive ramp with no
absolute position is RND's worst case); DRND (error is not a count).
Count hash strictly better where a hash exists. Do not revisit.

Temporally-correlated exploration: Tallec ICML 2019 - at high control
rates iid noise AVERAGES OUT; remedy is correlated noise, never a
coarser control rate. gSDE (2005.05719): resample state-dependent noise
every n steps; biggest published gains ON PPO (Hopper 2508 vs 1622);
log_std_init=-2, sde_sample_freq=4. Pink noise (ICLR 2023): beta=0.5
best for PPO; more parallel envs -> stronger correlation helps; CAVEAT
our yaw action is a rate so white noise on deltas is already beta=2 on
angle - log the realized PSD. ez-greedy (2006.01782): with prob eps,
repeat a random action for n ~ zeta(2) capped ~100; apply to movement
keys ONLY, keep aim closed-loop.

Seed variance/plasticity: Lyle 2024 (2407.01800) - parameter-norm
growth = effective-LR decay; stalled seeds may be unable to move far
enough in parameter space (complete exploration-free explanation of
walls). Test: log per-layer ||theta|| escaped vs stalled. Juliani&Ash
NeurIPS 2024 (2405.19153), the only on-policy study: WINNER = soft
shrink-and-perturb every step (beta 1e-6) + LayerNorm before ReLU;
final-layer reset/CReLU/ReDo FAIL on-policy. Calibrated Partial Resets:
only method avoiding collapse across 15 seeds under PPO at 400M steps.
On-policy parallelism study (2506.03404): more envs beats longer
rollouts on every health metric; but 2048 envs piled at one wall =
one env of diversity - walls are state-distribution collapse.
Kickstarting (1803.03835): distill stalled/converged seeds into fresh
nets (fresh Adam!) - 6.9x speedup, student exceeds teacher +43%.
QDagger: beats teacher in 75% of runs at 40x smaller budget.
Statistics: 0/4 seeds is compatible with per-seed escape rates 0-45%
(use IQM + performance profiles; rliable).

## 7. Speedrun/TAS search

Naive enumeration dead (52 frames/CPU-year). TMInterface bruteforce =
(1+1) hill-climber: savestate every tick, mutate events in a window,
rewind to earliest mutation, accept iff strictly faster, branch-and-
bound abort on falling behind splits. Practitioner lore: mutation
windows <= 6 s; decouple mutation window from evaluation window
(mutate a ramp, score exit speed). At our 783k fps: 1000-5000
candidates/s/core realistic - 1-2 orders above the TM community.
smb-opt: IDA* over a reimplemented movement model (10-byte states,
1e9 in RAM) solves SECTIONS, never full levels. Brittleness: A* Mario
under 20% action noise cleared 0/98 levels - search lines are
razor-thin; robustify via backward curriculum, NEVER a tracking reward.
Yosh's Trackmania A01 (23.64 vs human 23.79): RL + analytic low-level
helper + segmented training + separate brute-force glitch hunt (found
a track seam RL never did). Go-Explore Skiing lesson: reward clipping
made the agent ignore gates (one-shot event vs many-instance time
reward scale). Sonic benchmark footnote: end-of-episode bonuses
discounted away - our finish_k at gamma 0.9995 over 2600 decisions
(factor 0.27) is exactly that shape. Racing-line trajectory optimization
(min-curvature QP etc.) does NOT transfer: in surf the path GENERATES
the speed. If a planner: MPPI (forward rollouts only, tolerates clamps
and contact branches; ~1920 rollouts is the knee); Residual-MPPI ran
on top of frozen GT Sophy.

## 8. Our measured failures vs the literature

- RND: strongly validated dead. Do not revisit.
- Higher curiosity coef late-regression: validated; fix is a constraint
  (EIPO) or an episodic non-decaying table alongside the global one -
  not a different constant.
- Frame stacking: validated (superhuman vision-GT uses stack=1, no RNN;
  temporal info belongs in proprioception).
- Bigger nets: validated (Nexto); retry only after LayerNorm + norm
  anchoring.
- Per-episode time bonuses: validated (termination-correlated bonuses
  must be >= 0 / potential-based; ours was respawn-mistargeted).
- Strided rollout subsets: validated (data diversity per update is the
  binding resource).
- 11 Hz decision rate: validated as control-rate harm BUT confounded -
  our arm silently TRIPLED the discount horizon (kept gamma 0.9995;
  should have been 0.9985) and the literature's remedy is correlated
  noise or dwell constraints at HIGH rate (q1physrl: 72 Hz + 0.3 s key
  dwell). Re-run as "high rate + dwell constraint" if revisited.
- Warm gamma 0.999 helps slightly: right direction; field says go much
  further (1.4-13 s horizons).

## 9. Ranked shortlist (not yet tried)

1. COLLAPSE THE HORIZON (small-medium; goal 1 then 2). Linesight
   temporal mini-race: fixed 4-7 s windows, gamma=1 inside, normalized
   time-elapsed-in-window in proprioception, window-end = terminal,
   random window offsets at sample time. PBRS telescopes so the
   geodesic still guides long-range. Cheapest probe: gamma 0.997
   (10 s half-life... reference: 10 s = 0.997902, 20 s = 0.998950,
   30 s = 0.999300 at 33 Hz) TOGETHER WITH a time-remaining feature.
   Move time pressure into per-step penalty; shorten stall-kill 15 s
   -> ~2 s.
2. YAW ACTION REPARAMETERIZATION + gain-window proprioception (small;
   goal 1). omega = k*arctan(30/nu); optional annealed training-wheels
   shaping (realized_gain/900 - 1) on air ticks; audit for vectorial
   strafing; speed-squared contact penalty at ~0.01 relative scale.
3. ASYMMETRIC CRITIC with privileged geometry (small-medium; both).
   Critic gets exact geodesic value, absolute position, 20-40 ego-frame
   ramp points spanning 4-6 s at current speed. Actor unchanged (honest
   deployment). Consider small ego-frame lookahead for the ACTOR too,
   and a Swift-style gaze-alignment term for the aimable camera.
4. RESPAWN DISTRIBUTION OVERHAUL (small; both). W = 1/sqrt(n_chosen+1)
   per (arc-bin x speed-bucket) - we already compute this exponent;
   frontier bonus; Florensa competence band (0.1,0.9) with 1/3 reserved
   for mastered bins; multi-source (difficulty-biased snapshot weights,
   hand-authored transfer-jump entry states, q1physrl synthetic
   randomization incl. random time-remaining); post-respawn burst with
   90-95% action repeat; four per-jump reverse curricula on the 20%
   rule, windows never points; per-jump clear rate as primary metric.
5. SAVESTATE HILL-CLIMBING SEARCH ARM robustified into PPO
   (medium-large; goal 1). TMInterface-style (1+1) climber seeded from
   the 1:19.73 rollout; ~200 lines; 1000-5000 candidates/s/core; then
   backward-curriculum robustification from 4-10 perturbed variants
   (NEVER a tracking reward); closed loop policy->search->policy; plus
   an undirected glitch hunt from archived states.

Runners-up: 6. distributional (quantile) critic on PPO; 7. gSDE or
pink noise (one line, PPO-native); 8. SIL/SAIL ((R-V)_+ filter is the
method); 9. plasticity preventives on every scratch run (LayerNorm +
shrink-and-perturb beta=1e-6; NOT final-layer reset/ReDo);
10. kickstart new seeds from stalled ones (fresh Adam).

Free diagnostics first: (a) per-layer weight norms escaped vs stalled
seed; (b) entropy at wall-arrival; (c) confirm truncation bootstraps
value (we do - verified in trainer).

Expectation: Sophy spent 9 days closing <1.5%; we are trying to close
15%. Structural fixes + search, not more seeds.

Flagged unverified: TMInterface rates, Nexto hour-count folklore,
Yosh press numbers, Swift lambda values, Sophy exact lrs, "no prior
surf RL" (thorough, not exhaustive).

---

# Addendum: speedrun/TAS strand (second research stream)

New material beyond the main report:

## The GoldSrc TAS ecosystem already solved our action-space problem

**bxt-rs** (github.com/YaLTeR/bxt-rs) is a working TAS optimizer for
GoldSrc ITSELF: hill-climbing over mutations of hltas "frame bulks"
(strafe-type + target yaw + frame counts), with movement simulated via
**hlstrafe** (github.com/HLTAS/hlstrafe) - an exact analytic strafe
library implementing MaxAccelTheta / MaxAngleTheta / ConstSpeedTheta /
point-at-target-while-strafing. The critical design choice: **search
happens in autostrafe-parameter space, not per-tick input space** -
the optimal-strafe inner loop collapses the branching factor by orders
of magnitude. Author reports it "finds real skips and real
improvements" on HL runs.

Engine law verified from Valve pm_shared.c: at aa=100/wishspeed 250,
accelspeed=250 >> cap 30, so add = 30 - v*cos(theta) and
|v'|^2 = v^2 + 900 - (v cos theta)^2: **optimum exactly perpendicular,
|v'| = sqrt(v^2+900) per tick**. Lateral accel ~3000 u/s^2 independent
of speed = surf's "friction circle"; min turn radius R ~ v^2/3000.
Our +-10 deg/tick view cap is never binding - physics is.

**Action-space altitude idea (new lever):** a micro-strafe EXECUTOR
between 33Hz decisions - the decision emits a target direction, an
exact inner loop emits per-tick optimal wishdir (fully determined by
(v, target) at aa=100). This removes the act-every-K precision penalty
in principle and is how every serious GoldSrc TAS tool operates.
Strafe-efficiency auxiliary: eta = (|v'|-|v|)/(sqrt(v^2+900)-|v|),
computable per tick from state (diagnostic first, annealed reward
maybe).

## The search-then-robustify closed loop, with named precedents

PPO best run -> deterministic hill-climb polish (TMInterface/bxt-rs
pattern, frame-bulk altitude) -> polished trajectory becomes (a) the
demo spine for a Salimans-Chen backward gated curriculum (window
starts near finish, moves back only when >=20% of workers succeed)
and (b) a refreshed reference line for progress reward (Linesight
re-extracts its line from the AI's own best runs as records fall) ->
PPO robustifies and improves -> repeat. Every arrow has a verified
precedent.

Reconciliation with Song/Scaramuzza's "never track a reference line":
Linesight pays progress ALONG a line (a progress coordinate, policy
free to deviate) - that is still a progress objective. What fails is
penalizing DISTANCE TO the line. Progress-along-own-best-line also
fixes the deeper bias: the geodesic shortest path is not the fastest
line when speed must be farmed on ramps.

## Calibration datapoints

- HappyLee's SMB random bot (savestate resets + random inputs): 4.5%
  behind the crafted TAS - random search + resets closes all but the
  last ~5%; the last percent needs structure. Mirrors our 1:19.7 vs
  1:08.
- q1physrl detail: squashed-Gaussian (CDF) action head specifically
  because clipped Gaussians push probability mass outside bounds.
- No credible record-beating RL for Celeste/Dustforce/N++/QWOP -
  treat claims as folklore.
- jwchong: greedy per-frame speed maximization is NOT globally optimal
  while surfing (future ramp geometry matters) - any analytic line
  needs engine verification; argument for search/RL over myopic
  analytic policies.

Ranked recs from this strand: (1) reference-line progress from own
best run + gamma -> 1 anneal; (2) hill-climb polish at frame-bulk
altitude; (3) backward success-gated curriculum over best-run states;
(4) analytic strafe executor layer / BC-pretrain from scripted
MaxAccel controller; (5) MLTO/MPPI line generation only after 1-3.

## Late additions (second pass of the TAS strand)

- **Value-ceiling bound**: the closed forms (optimal strafe gain
  sqrt(v^2+900)/tick + ramp-slide g*tau*sin(2*alpha)/tick) can be
  integrated along ANY candidate route to lower-bound its achievable
  time. Principled answer to "does 1:08 need a different route or just
  cleaner execution of the current line?" - compute the bound for the
  champion's route before spending compute improving it.
- hlstrafe's `MaxAccelIntoYawTheta` = steer toward a target yaw at max
  accel - the exact primitive for an analytic executor under a
  policy-chosen target heading.
- Sonic benchmark transfer finding: joint pretraining + per-map
  fine-tuning roughly doubles low-budget scores - relevant to the
  convergence goal via ramp-primitive / multi-map pretraining.
