# `--curiosity-cond`: one network, a family of exploration weights

Branch `contyaw-abs` (2026-09-06, night). Default OFF; with the flag off the
trainer is byte-identical to the trainer before it (pinned, see "What is
pinned"). Everything here was smoked on the CPU with the local GPU hidden;
nothing was rented and no arm has run yet.

## The ask

The user's words: "What if T becomes input to agent? So we basically make
some agents less curious and some more. We can train such agents just by
changing the reward. Basically for agents with low T, the reward they get is
just the geodesic, while for highest T it's almost completely curiosity."

That is Agent57's design (Badia et al. 2020): one network conditioned on the
exploration weight, actors at different weights, the intrinsic reward mixed
in proportion. Memory is not needed here - T is an observation feature
present at every step - and unlike the `--unstuck` schedule
(`docs/unstuck.md`, a single T for the whole fleet that rises on a plateau)
the fleet carries the WHOLE range at once, every iteration, so the
exploit member and the explorers train the same weights from the same
rollout buffer.

## What it does

### 1. The observation: one column, the critic sees it

Each env's T enters the network as ONE extra scalar-side column,
`t = log1p(T) / log1p(T_max)` in [0, 1] (`surfgym.rewards.cc_encode`;
0 at T = 0, 1 at T_max, concave so the log-uniform continuum is spread over
the unit interval). It is the LAST column of the scalar-side block -
`[15 core | fan | latch | aux | T]`, `CC_COL = N_SCALAR + N_ROUTE - 1` -
exactly where `--race-latch`'s flag sits, so:

* it reaches both towers (`Policy.heads` concatenates `scal[:, N_SCALAR:]`
  into the pi AND the vf input): the critic knows which reward mix it is
  predicting, or the value of one state would have as many targets as
  there are members;
* a plain checkpoint resumed with the flag is widened by `widen_for_obs`'s
  trailing zero-pad (6 tensors: `pi.0`/`vf.0` and their Adam moments), so
  at step 0 it computes the checkpoint's own function at EVERY T and the
  column grows from zero (smoked: "widened 6 tensors ... at scalar-row
  15..15");
* every consumer that assembles a row feeds it: the rollout
  (`fill_vision`, a pinned staging row like the latch's), the truncation
  bootstrap's reconstructed terminal row (`RaceReward.cc_obs_boot` - the T
  of the episode that just ENDED, since the reward has already redrawn the
  live one), the in-trainer evals, `record_ckpt`, `beam_tas`,
  `diversity_bench` (`_TorchPolicyBase(cc_fn=...)`, `train_fast.make_cc_feed`)
  and the BC loader (`BCDataset(n_cc=1)`: a planner line's rows are imitated
  as the T = 0 member, the column synthesised at 0).

The checkpoint config carries `curiosity_cond`, `cc_p0`, `cc_tmin`,
`cc_tmax`, `cc_buckets`, `cc_temp_scale`, `cc_temp_gain` (written only
under the flag; restored on a flagless resume; `record_ckpt` mirrors the
first, `cc_tmax` and the two temperature knobs and lists the rest as
TRAIN_ONLY).

### 2. Per-env T, redrawn at every episode start

`RaceReward(cc_tmax > 0)` owns a T vector. At `on_reset` and at every
episode end inside `__call__` (autoreset, reservoir respawn, stall kill,
truncation - the same place the latch and the arc anchor are reset) the
ended envs draw a fresh T from the mixture

    T = 0                       with probability --cc-p0      (default 0.5)
    T ~ log-uniform [tmin, tmax] otherwise                     (0.05, 2.0)

The draw is a HASH of `(seed, env, episode index)` (`cc_draw`: SplitMix64
over the three integers, two independent uniforms per draw), not a stream:
env i's k-th episode gets the same T whatever the other envs did, so a run
is reproducible and any tool can re-derive an env's T from three numbers.
Pinned: the T = 0 share is p0 to 1 %, the mean of log T is the midpoint of
the log range, a draw in a batch equals the same draw alone. The episode
counters are not checkpointed: a resume restarts every env at episode 0
(the reservoir and the counts table are restored as before).

The step line reads `cc T0 52% Tm 0.21 len 64/64/64/64` (the share of
envs at T = 0, the mean T over the fleet, the mean episode length per
bucket of the episodes that ended this iteration) and `progress.csv`
gains, LAST and only under the flag, `cc/frac0`, `cc/T_mean` and per
bucket `cc/n_b{k}`, `cc/len_b{k}`, `cc/rew_b{k}` (count, mean length in
decisions, mean return of the episodes that ended, keyed on the T the
episode RAN at).

### 3. Per-env reward

With `w = T / T_max`, `RaceReward.__call__` pays env i

    (1 - w) * [shaping]  +  T * int_coef / sqrt(N(cell) + 1)  +  [outcome]

where `[shaping]` is everything the control computes before the ended-row
zeroing - the potential difference (geodesic or `--race-arc`), the time
penalty, the `--race-ng` tax and the speed terms - and `[outcome]` is the
success bonus, the fail penalty, the finish-time bonus and the terminal
charges, paid in full to every member. So the T = 0 member is the plain
race reward with NO novelty (the control has `int_coef 0.25` always on; the
family's exploit member does not), and the T_max member is paid nothing
for progress and nothing for time - only novelty and the outcome terms.
The count table is still the fleet's (whoever visited a cell wears it out
for everyone). `RaceReward.set_cc_T` pins the vector for the eval
mirrors and the tools. Pinned against a hand computation for T in
{0, 0.5, 1, 2} (`tests/python/test_curiosity_cond.py`), with the flag off
bit-identical to the control over a dive-and-climb trajectory.

Two consequences to keep in mind when reading a run:

* the stall detector and the respawn `stagnant` mask keep the RAW geodesic
  (they are liveness rules, as for the latch and the arc): an explorer that
  wanders without improving its best geodesic by 32 u for 15 s is killed
  like anyone else. Exploration is therefore bounded to "keeps finding new
  depth every 15 s"; a member that hunts novelty sideways dies. This is
  the same rule every arm runs under and changing it would be a second
  treatment - but it is the first thing to look at if the high-T buckets'
  `cc/len` collapses;
* `race/int_per_ep` on the step line (`int ../ep`) is now pooled over a
  mixture; the per-bucket `cc/rew_b{k}` columns are the honest read.

### 4. Advantage normalisation per T bucket (`--cc-buckets`, default 4)

PPO normalises advantages per minibatch, `(a - mean) / (std + 1e-8)`. A
T = 0 member's advantages are in shaping units (~0.02-0.2 per decision), a
high-T member's in novelty units (`int_coef / sqrt(N)`, and the returns
of the two families differ in scale and sign); one shared std would hand
whichever family has the larger spread the whole gradient scale of the
other, and the mixture's mean would shift every member's baseline. So the
moments are taken per T bucket inside the minibatch
(`train_fast.cc_bucket_normalize`: one `(mb, B)` one-hot, torch.std's
unbiased estimator per bucket, a one-row bucket normalises to 0; static
shapes, so it compiles like the estimator it replaces). Bucket 0 is the
T = 0 family; the other B - 1 cut the log-uniform continuum into
equal-probability slices (`cc_bucket_edges`, in t; rows are bucketed off
the very column the policy saw, `torch.bucketize(f_scal[:, CC_COL])`).
The smoke's `cc/rew_b0 = -0.315` (64 ticks of time penalty, no novelty)
against `cc/rew_b3` in [-0.02, +0.29] is the scale gap the buckets exist
for. `--cc-buckets 1` is the shipped estimator.

### 5. Sampling temperature on the KEYS heads (`--cc-temp-scale`, default 1)

The unstuck benchmark measured (docs/unstuck.md) that tempering the yaw
sigma kills the flight and that the keys temperature is the cheapest
diversity per unit of progress lost - and the only knob that reached
unseen cells. So a member at T samples its four categorical (keys) heads
at temperature `1 + --cc-temp-gain x T` (0.25: 1.5 at T = 2) and its
Gaussian view heads untempered. The mechanics are the unstuck work's: the
per-env temperature is a STATIC `(N, 1, 1)` tensor (`(N, NACT, 1)` on the
bins, with the two view heads pinned at 1 - `cc_temp_block`) written from
the reward's T vector before every decision and read inside the captured
rollout graph by the same `sample_padded` / `sample_view` calls; the
update records the block each decision was drawn under (`b_cct`) and
scores every row at ITS temperature through `logprob_entropy_*`
(`f_temp = f_cct[idx]`), so `pi_new` and `pi_old` are the same measure and
the ratio is a true importance ratio. `--cc-temp-scale 0` leaves every
draw at temperature 1. `--unstuck` is refused with the flag: two
temperatures on the same heads.

### 6. Evals and tools

* The in-trainer evals (greedy and stochastic) run the **T = 0 member**:
  the exploit policy is the one the arm is judged on. `race/eval_progress`
  and the trajectories are that member's.
* `python tools/record_ckpt.py <ckpt> --map ... --cc-T 1.0` records the
  member at T = 1 (greedy); `--stochastic` samples its keys at the trained
  temperature `1 + gain x T`; the trajectory header carries `cc_T`. A plain
  checkpoint refuses `--cc-T`.
* `python tools/diversity_bench.py <ckpt> --map ... --route ... --cc-T 0.5
  [--cc-trained-temp]` rolls the member at T = 0.5 through the whole
  knob/temps grid (the `cc_T` column of `bench.csv`); `--temps 0 --knob
  sigma` is that member's own sampled policy, `--cc-trained-temp`
  multiplies the trained keys temperature into every row. So the family's
  diversity at each T is measured with the SAME metrics as the
  sampling-temperature benchmark and can be put in the same table.
* `python tools/beam_tas.py <ckpt> ... --cc-T mix` (or a number) runs the
  proposal envs as members drawn from the training mixture, one T per env
  slot for the whole search (5/16 at T = 0 in the smoke), with the trained
  keys temperature per env in the plain sampled branch (`--cc-temp off` to
  leave it at 1; under `--eps`/`--dedup`/`--greedy-envs` the proposals
  draw at 1 and a note says so); the greedy gate/replay core is the T = 0
  member. `beam_best.npz` and `summary.json` carry `cc_T`. The replayed
  line is open-loop, so a line stands whatever member proposed it.
* `tools/plan_to_bc.py` needs nothing (it replays lines through the core;
  no policy forward); `BCDataset(n_cc=1)` imitates the rows as the T = 0
  member. `tools/expert_dagger.py` and `tools/transplant_view.py` REFUSE
  a family checkpoint with the reason (the sample bank reads column 15 as
  the latch and stores no T per row; the transplant rolls without the
  column) - run the expert loop with `--dagger-k 0` on a family seed.

## Flags

| flag | default | meaning |
|---|---|---|
| `--curiosity-cond` | off | the family |
| `--cc-p0` | 0.5 | probability an episode is a T = 0 member |
| `--cc-tmin` | 0.05 | floor of the log-uniform range |
| `--cc-tmax` | 2.0 | T_max: the ceiling, and the T at which the shaping weight (1 - T/T_max) is 0; also the column's encoding |
| `--cc-buckets` | 4 | advantage-normalisation buckets by t (1 = the shipped estimator) |
| `--cc-temp-scale` | 1 | keys heads sample at 1 + gain x T |
| `--cc-temp-gain` | 0.25 | that gain |
| `record_ckpt --cc-T` | 0 | the member to record |
| `diversity_bench --cc-T`, `--cc-trained-temp` | 0, off | the member to roll; multiply in its trained keys temperature |
| `beam_tas --cc-T`, `--cc-temp` | 0, trained | the proposal envs' member(s) (`mix` = the training mixture); their keys temperature |

Refused (a message, not a silent misread): `--reward` other than race,
`--int-coef 0` (the T > 0 members would be paid a scaled-down race reward
and nothing else), `--unstuck`, `--maps`, DDP, `--obs-reward` (its slot-12
mirror would need the per-env weight), `--rnn`, `--chunk`/`--codebook`,
`--yaw-cond`, `--ez-eps`, `--spawn-burst`, `--frame-stack`, `--goals`,
`--race-ng`, `--death-charge`, `--speed-equiv`, `--speed-coef` (the
reward-side extras have no measured place in the (1 - w) mix).

## How to launch it on the testbed

    SCRATCH=1 bash tools/run_arm.sh cyCC --curiosity-cond          # the family
    SCRATCH=1 bash tools/run_arm.sh cyCTL                          # the control

Both are the from-scratch cannonball baseline in the absolute velocity-frame
view (`VIEW=abs`, CLAUDE.md section 2). The two differ in the reward ONLY
through the flag: the control pays every env `shaping + 0.25 x novelty`,
the family pays the mixture above. Compare them on the exploit member's
evals (`race/eval_progress`, and `tools/eval_honesty.py` corridor MAX and
finishes on the `traj_*.jsonl`, which are the T = 0 member's), report the
gate cleared and the step it was cleared at, and read the per-bucket
`cc/len` / `cc/rew` columns for what the explorers are doing. `--cc-p0`,
`--cc-tmax` and `--cc-temp-gain` are the knobs a second arm would move.
A warm start from a plain checkpoint (`--ckpt <plain.pt> --curiosity-cond`)
is supported: the column is widened onto it and the policy starts as the
checkpoint's own function at every T.

Locally: `powershell -File tools\launch_local.ps1 scratch_ablate cyCC
--curiosity-cond` (the scratch presets pass trailing flags through).

## What to expect

Nothing is measured yet. What the design predicts, so the first run can be
read against it:

* `cc/frac0` sits at p0 +- the binomial noise of 2048 envs (~1 %) and
  moves every iteration (T is redrawn per episode); `cc/T_mean` ~0.25 at
  the defaults (half the envs at 0, the other half log-uniform on
  [0.05, 2], mean ~0.53).
* the T = 0 bucket's `cc/rew_b0` is the race reward proper; the top
  bucket's `cc/rew_b3` is dominated by novelty and shrinks as the count
  table wears the map out (the benchmark's "novelty is worn out
  everywhere a rollout can reach" applies to every member equally).
* the exploit member's evals should track a control's early on (same
  reward at T = 0, minus the 0.25 novelty the control has) and the
  question the arm answers is whether the explorers' rollouts, trained
  into the same weights, move the exploit member's frontier - the gate
  ladder of CLAUDE.md, reported as the gate and the step, not a mean.
* if the high-T buckets' `cc/len` collapses to the stall window (~375
  decisions at act_every 4) the explorers are being killed for not
  descending; that is the liveness rule above, and the honest reading is
  "the family cannot explore sideways on this map under the stall kill",
  not "curiosity failed".

## What is pinned (`tests/python/test_curiosity_cond.py`, 15 tests)

(a) the draw (mixture statistics, purity in (seed, env, episode), the
encoding and its inverse); (b) the buckets (equal probability, bucket 0 =
T = 0, numpy == torch.bucketize, p0 = 0 and B = 1 / 2); (c) the flag OFF
is the control bit for bit over a dive-and-climb, and the per-env mix
against a hand computation - shaping x (1 - w), bonus x T, the success
bonus and the fail penalty in full, the constructor's refusals; (d) T is
redrawn for the ended envs only (done and truncation), `cc_boot` keeps
the old one, `cc_obs` / `cc_obs_boot` encode; (e) the eval wrappers append
the column LAST with the encoded value and a one-column-wider policy reads
it, per-env feeds, the feed's range check; (f) `cc_bucket_normalize`
against per-bucket torch.std moments, a one-row bucket, B = 1 == the
shipped estimator; (g) `cc_temp_block` pins the view bins, the tempered
log-prob with a per-row block equals the per-row scalar computation, the
`TemperedTorchPolicy` at a per-env `keys_temp` is `sample_view` at that
block under the same seed with the Gaussian heads untouched; (h) the
source-level pin of the rollout's static temperature, the per-row record
and the per-row scoring, the bucketize on `CC_COL`, and the unstuck pins
still holding; (i) the trainer with the flag OFF is the trainer of the
last commit without it, bit for bit (config, progress.csv header and rows
minus fps, eval trajectory bytes, weights, Adam moments) on the bins and
on the absolute view; a flag-ON smoke with 64-tick episodes (finite
losses, |kl| < 0.05, frac0 in [0.25, 0.75] and moving, every bucket
seeing episodes, bucket 0 paid the 64-tick time penalty and the top
bucket paid more, the config and the checkpoint carrying the flag, the
T = 0 eval); `record_ckpt --cc-T 1.0` (header `cc_T`), `--cc-T 1.5
--stochastic` at keys temperature 1.375, a plain checkpoint refusing
`--cc-T`; a flagless resume restoring every knob; a plain checkpoint
widened (6 tensors); `beam_tas --cc-T mix` planning with the mixture and
the trained per-env temperatures, replay exact, `cc_T` in the npz and the
summary, refused on a plain checkpoint; `expert_dagger` refusing with the
reason; `BCDataset(n_cc=1)`; the trainer's refusals.

## Not done / open

* No arm has run: this is the mechanism, smoked on the CPU. The numbers in
  the ledger are smoke numbers.
* The `--obs-reward` slot-12 mirror does not know the per-env weight, so
  the flag is refused with it; the scratch testbed does not use it.
* `expert_dagger` / `transplant_view` refuse the family; the expert loop's
  DAgger phase therefore needs `--dagger-k 0` on a family seed. The
  planner's trained per-env temperature reaches only the plain sampled
  branch (not `--eps`/`--dedup`/`--greedy-envs`).
* The per-env T is not checkpointed (a resume restarts the episode
  counters), and under `--reward-per-decision` the bootstrap reads the
  live vector (`cc_obs_boot(live=True)`) because the reward is called after
  it - both documented in the code, neither exercised by an arm.
* Agent57 also trains SEPARATE value heads for the intrinsic and extrinsic
  returns and a meta-controller over the members; here there is one value
  head that reads t, and the members are drawn uniformly (in log T) with
  no bandit over them. Both are candidate second arms, not part of this
  one.
