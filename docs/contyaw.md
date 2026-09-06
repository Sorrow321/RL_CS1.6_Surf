# `--view-continuous`: continuous yaw and pitch

Branch `contyaw` (2026-09-06). Default OFF; with the flag off every layer is
byte-identical to before (see "What is pinned" below).

## Why

The discrete policy steers with two categorical heads, 15 yaw bins and 7
pitch bins. To a softmax those are separate classes: "+0.5 is the same
distance from +0.6 as it is from -10.0", every neighbourhood has to be
learned. Measured (`tools/demo/wr_scan.py`, ledger "14:10 - the record
through the policy's eyes"): the human record's turn rate falls between two
of our bins 42% of the time; our best line changes yaw bin on 36% of
decisions with 8% A-B-A alternations against 23% / 2% for the record
quantised to our bins; the held bin costs 0.44 u/s per decision against
per-tick mouse control while the physics reproduces the record to 0.10 u.

## What changed, layer by layer

### Core (`src/env.c`, `src/surfcore.h`, ABI 7 -> 8)

A new export, `surf_step_view(sim, acts, view, ...)`: the same batch step
with a float32 `(N, 2)` row per env, `(yaw command k, pitch deg/tick)`,
replacing the yaw/pitch BINS of the action row (columns 0/1 of `acts` are
ignored, the other four keep their meaning). The formulas are the bins' own
with the table replaced by the command, so a command equal to a bin's table
value reproduces the bin bit for bit:

| | bins | continuous |
|---|---|---|
| yaw, `--yaw-adaptive` | `K_BINS[b] * atan(30/v_h)` deg, clamp `+-yaw_rate_max_deg` | `k * atan(30/v_h)`, same clamp |
| yaw, fixed bins | `YAW_BINS[b] * (rate/10)` | `(k * 0.5) * (rate/10)`, clamp `+-10 * (rate/10)` (k in units where +-20 is the outermost bin) |
| pitch | `PITCH_BINS[b] * (pitch_rate/10)` | `clamp(cmd, +-10 * (pitch_rate/10))` |

`--yaw-blend` and `--side-hold` apply on both paths (they act on the
applied delta and on the side bin, after the decode). The action echo in
obs slots 10/11 is the applied delta as before. NaN commands apply 0.
`surf_step` is untouched and shares the loop (`step_impl(view=NULL)`).
`surf_pm_step_single` (the single-step API) stays discrete.

Python: `SurfCore.step(acts, view=None)` (`surfgym/core.py`, `validate_view`,
`VIEW_DIM`); `MapFleet.step(acts, view=None)`; `record_rollout` reads
`policy.view` and passes it. `surfgym/view.py` holds the numpy helpers.

### The warp (`surfgym/view.py`, `train_fast.warp_t`)

The policy draws a pre-tanh `z ~ N(mu(s), sigma)` per head, `u = tanh(z)`,
and the env applies

    K     = 20 * sign(u) * (exp(alpha |u|) - 1) / (exp(alpha) - 1),  alpha = 2 ln 19
    pitch = u * pitch_rate_max_deg

so `u = +-0.5 -> K = +-1` (the analytic optimal-strafe multiple) and
`u = +-1 -> K = +-20` (the old outermost bin). Resolution: `dK/du = 0.33` at
zero, `4.6` at K = 1, `>100` at the ceiling. **Note that consequence**: the
warp is 14x steeper at the strafe optimum than at zero, so a z error that is
harmless at K = 0 is a whole bin at K = 1 (this bit the transplant, below).

### Policy (`python/train_fast.py`)

* `Policy(view_continuous=True)` adds `view_head = Linear(hidden, 2)` (the
  means, over the pi tower like `action_head`) and `view_std.log_std`
  (a state-independent log sigma per head, init `log 0.3`, clamped to
  `[-5, 1]` in `Policy.log_std()`), both registered LAST so every existing
  parameter keeps its Adam index. `action_head` keeps its full `sum(NVEC)`
  width (its yaw/pitch logits are dead outputs under the flag), so a
  discrete checkpoint loads key for key and the transplant only fits the
  new tensors. The flat output is `[sum(NVEC) logits | 2 means]`;
  `split_view(logits)` takes it apart. An unaware caller that pads the
  34-wide tensor fails loudly (shape mismatch) rather than sampling a dead
  bin.
* z is THE action: the rollout stores it in `b_z (T, N, 2)`, the log-prob
  is the Gaussian log-density of z plus the four categorical terms, the
  entropy the categorical sum plus the Gaussian entropy (no Jacobian: tanh,
  warp and clamp are the environment's deterministic response, as the bin
  table was). `sample_view` / `logprob_entropy_view` next to the discrete
  `sample_padded` / `logprob_entropy_padded`; the update's `mb_step` takes
  `f_z`. The env receives `view = (warp(tanh z_yaw), tanh z_pitch *
  pitch_rate_max_deg)` computed inside the CUDA graph (`static_view`) and
  copied through one more pinned row; the int action row keeps
  `NEUTRAL_ACT` in columns 0/1.
* Greedy = `z = mu`. `GreedyTorchPolicy` / `SampledTorchPolicy` publish
  `.view` (`(N, 2)` float32, `None` on a discrete policy); every driver that
  steps a core (`record_rollout`, `beam_tas.run_episode`, `dagger.
  collect_rollout_samples`, `relabel_windows`) reads it.
* Action masks (`--mask-forward-air`, `--jump-cooldown`, `--duck-air-mask`)
  work unchanged - they touch heads 2..5.
* BC (`--bc-file`): the four categorical heads as before on their slice;
  the view heads by the Gaussian NLL of the row's executed z (`--bc-target
  argmax`) or the Gaussian cross-entropy to the elite copies' moment-matched
  `N(zmu, zsd)` (`--bc-target dist`): `log s + log sqrt(2 pi) + (zsd^2 +
  (zmu - mu)^2) / (2 s^2)` - at a FIXED reference sigma `s = BC_VIEW_SIGMA
  = 0.3` (the init), i.e. an MSE on mu with a fixed scale; the policy's own
  sigma is left to PPO. Measured why: at the live sigma (0.05 after the
  transplant) the term's gradient on mu is 1/sigma^2 = 400x, it dragged mu
  by a sigma per iteration and approx_kl read 1.46 / 0.62 on the first two
  iterations of the smoke; it would also fit sigma to the 0.03 z residual
  and collapse it. A row a single line survived at has `zsd = 0` and the
  two targets coincide. `bc_log.csv` APPENDS a `bc/view_mse` column under
  the flag; the console line adds `view-mse`.
* `progress.csv` is unchanged; the `step` line adds `sig y/p` (the two
  learned sigmas). `act/yaw_side_agree` reads the sign of z instead of the
  bin.
* Config: `view_continuous: 1` written ONLY when set; restored from a
  checkpoint on resume; `tools/record_ckpt.py` mirrors it (and lists the
  transplant's provenance block `view_transplant` as TRAIN_ONLY).
* A DISCRETE checkpoint cannot be resumed with the flag ("transplant it
  first"); a continuous one cannot be resumed without it.

### Planner and loop

* `tools/beam_tas.py`: the population's action history grows a `(D, N, 2)`
  float twin (`hist_v`), cloned, captured and saved beside it; proposals
  are samples of z from the policy (`EpsSampledTorchPolicy._sample_view`:
  eps-uniform per categorical head, and with probability eps a uniform
  `u in (-1, 1)` per view head; `MixedTorchPolicy`: greedy envs take
  `z = mu`); `--dedup` hashes z quantised to 0.05 (12 bits per head above
  the six bins); `--macro-yaw track` writes the analytic K directly
  (`macro_yaw_k`: `+-1` under `--yaw-adaptive`, else the analytic angle in
  the fixed path's units), `MacroHold.decide_view`; `--branch-at` draws a
  yaw COMMAND (jitter 0: `warp(u)`, `u` uniform; jitter J: an offset in K
  units); `--branch-grid` yaw offsets are floats in K units (default
  `-3,-2,-1,0,1,2,3`); `--prefix-line` replays a continuous line with its
  view, or a discrete line through the bins' own view values (bit-exact);
  `--robust` and `replay_arc` step the view alongside; commit mode carries
  the view blocks. `beam_best.npz` gains `view (D, 2)`, `view_all
  (n, D, 2)` and `view_continuous`; `summary.json` gains `view_continuous`.
* `surfgym/bc.py`: `replay_line(view=..., views_out=...)`; BC files gain
  three OPTIONAL columns, `view (R, 2)` (the executed physical view),
  `view_zmu` / `view_zsd (R, 2)` (the moment-matched z target);
  `save_bc_dataset` / `load_bc_arrays` carry them; `BCDataset(view_continuous
  =True, yaw_adaptive=, pitch_rate_max_deg=)` reads them or DERIVES them from
  a discrete file (`bin_to_view` for the point target, `bin_view_moments`
  over `probs` for the moments) and says which in `describe()`;
  `sample_all(n, view=True)` returns `(vz, vzmu, vzsd)` after the eight
  usual columns. `rank_lineages` dedups on bins + view.
* `tools/plan_to_bc.py`: reads `view`/`view_all`, refuses a plan/checkpoint
  mismatch of the flag, replays with the view, writes the executed view per
  row and the survivor-weighted z moments (`survivor_view_moments`, the
  twin of `survivor_probs`: prefix agreement on bins AND views).
* `surfgym/dagger.py` / `tools/expert_dagger.py`: `RowPolicy` publishes a
  view (greedy `z = mu`, tempered `mu + temp * sigma * eps` elsewhere);
  `relabel_windows` keeps a view twin of the window history, clones it,
  and each result carries `label_view` and the population's
  `label_view_zmu` / `label_view_zsd` (`population_view_moments`);
  `rows_from_results` / `merge_bc_datasets` carry the columns.
  `--label-target gumbel` is refused under the flag (it needs discrete root
  logits).
* `tools/expert_loop.py` needs no change: the flag is restored from the seed
  checkpoint on every resume, `record_ckpt` / `beam_tas` / `plan_to_bc` read
  it from the checkpoint and the plan. For a SCRATCH seed pass it through
  `--train-extra --view-continuous` (that tail reaches both the scratch and
  the resume trainer).
* `tools/line_fragility.py` replays a continuous line's view; its `bin`
  perturbation becomes `+-0.25 K`.
* `tools/record_ckpt.py`: unchanged trajectory row format (pitch is a state
  there; the view command is not recorded).

### The transplant (`tools/transplant_view.py`)

    python tools/transplant_view.py <discrete.pt> <out.pt> \
        --map C:/RL_Surf/maps/surf_src_cannonball.bsp \
        --finetune-steps 8000 --finetune-lr 5e-4 --finetune-batch 8192 \
        --kl-coef 0.2 --k-coef 8 --dagger-rounds 4

Rolls the discrete checkpoint on 256 envs (64 greedy, the rest sampling)
from a pool mixing the map start with its own respawn reservoir, keeps the
trunk output and the six heads' logits per decision, and targets per row
`z_yaw = atanh(warp_inv(E[K]))`, `z_pitch = atanh(E[pitch] / rate)` (the
categorical means) with `log sigma` seeded from the categorical spread in z
(floored at -3). Then, in order of what actually worked on the reference
checkpoint (all at the same 9 greedy episodes, the discrete source finishes
7/9, best 70.1 s):

| variant | fit | greedy eval |
|---|---|---|
| linear readout on the frozen tower (the design as specified) | R^2 0.80 / 0.71 in z, greedy-bin agreement 32% | dies 3.5-8 s after spawn, 9/9; corridor mean 6.7k u |
| + `--finetune-steps 4000` (pi + action_head + view_head, z-MSE + KL) | R^2 0.95 / 0.99, agreement 53% | dies 3.5-7.7 s, 9/9 |
| + K-space term, `--dagger-rounds 3` | agreement 34% on the union | corridor mean 15.3k, max 43k (18.6%), 0/9 |
| + argmax-K target, `--kl-coef 0.2`, 8000 steps, 4 rounds (**shipped**) | agreement 50% on the union | corridor mean 41.9k, max 176.9k (76.4%, one dive-below), 0/9 |

The diagnosis behind the deviation (the design said "copy every shared
weight and FIT the two new heads"): with the continuous policy's KEYS and
the discrete bins' VIEW the episode finishes (70.77 s); with the discrete
keys and the continuous mean VIEW it dies at 4.75 s. The argmax of 15
logits that are linear in the tower output is piecewise constant in it,
no single Linear fits that staircase, and the warp's slope at K = +-1 turns
a 0.13 z error into 0.6 K, i.e. one to two bins, on a line that
`tools/line_fragility.py` measured dying on a 1 u offset 59% of the time.
The fine-tune therefore also moves the pi tower and `action_head` (trunk,
value tower, value head and the seeded sigma stay bit-identical; the KL term
keeps fwd/side/jump/duck at 97.7-99.5% argmax agreement), and DAgger rounds
label the student's own drifted states with the teacher. **The transplant
is a seed for PPO+BC, not a finisher** - report it as such.

## How to launch an arm

The flag rides in the checkpoint, so every launcher works unchanged once
the seed is transplanted:

* local: `powershell -File tools\launch_local.ps1 resume runs\contyaw\ref_cont_dg2.pt cyARM`
  (add `--view-continuous` after the run name to be explicit; a scratch
  preset takes it as an extra flag);
* rented, plain PPO: `CKPT=runs_ckpt.pt EXPECT_MD5=<md5 of the transplant>
  ARM_RESUME=1 bash tools/run_arm.sh cyARM` (ARM_RESUME skips the stuck-
  checkpoint md5 gate and the pinned-config guard, whose `act_every: 3`
  the reference does not match anyway; say so in the ledger);
* rented, expert iteration: `bash tools/wave/run_exit_ab.sh <port> <host>
  <instance> cyEXIT <hours> /c/RL_Surf_cy/runs/contyaw/ref_cont_dg2.pt
  --bc-target dist --bc-value-coef 0.25` - the trailing flags reach the
  trainer verbatim; `EXTRA_LOOP_FLAGS='--dagger-k 600'` for the loop's own
  flags. Ship the ABI-8 core: the box must build from this branch
  (`build.sh`), an ABI-7 DLL refuses to load.

## What is pinned (`tests/python/test_view_continuous.py`)

(a) `surf_step` on the ABI-8 core reproduces the ABI-7 core's trajectories
byte for byte (`tests/python/data/view_golden_abi7.npz`, generated with the
pre-change DLL, four configs); (b) the continuous path fed the bins' own
values reproduces the bins bit for bit; (c) the warp anchors and inverse,
torch == numpy; (d) the mixed distribution's log-prob and entropy against
a hand computation, the update recomputes the rollout's log-prob from the
stored z; (e) the Policy's three new tensors come last and the flag off
draws no RNG; (f) the transplant's fit and optimizer-group growth; (g) BC
targets and BCDataset's derive-or-read; (h) a continuous planner line
round-trips through `replay_line`; (i) the trainer with the flag OFF is the
pre-flag trainer bit for bit (config, progress.csv, eval trajectory,
weights, Adam moments - the same comparison was made by hand against
runs captured BEFORE the trainer was touched), and a tiny flag-on run trains.

## Not supported under the flag (refused with a message)

`--yaw-cond` (the side key conditions on a yaw BIN), `--chunk` /
`--codebook` (a code decodes into bin distributions), `--act-hist` (the
history feature reads the yaw bin), `--frame-stack`, `--rnn`, `--ez-eps` /
`--spawn-burst` (bursts draw whole action rows outside the policy),
`--label-target gumbel` in expert_dagger. `tools/demo/wr_scan.py` is a
discrete-only diagnostic (it pads the raw logits). `surf_pm_step_single`
and the play client stay discrete.

## Absolute targets (`--view-absolute {velocity,world}`, branch `contyaw-abs`)

The user's follow-up: "I would run another box on the absolute predictions,
not deltas, just to compare. The only thing to consider: yaw is cycled
(-pi = pi), so we need to think how to work around it. Maybe apply cos/sin."
Branch `contyaw-abs` off `origin/contyaw` (2026-09-06). Default absent =
the delta path above, byte for byte (see "What is pinned" below).

### What the core does (ABI 8 -> 9, `SurfEnvConfig.view_mode`, LAST field)

`surf_step_view` keeps its `(N, 2)` float row; `cfg.view_mode` says how it
is read. 0 (default) is the delta command above. 1 (`velocity`) and 2
(`world`) read it as **`(yaw target deg, pitch target deg)`** and, EVERY
tick, from the live state:

| | mode 1 `velocity` | mode 2 `world` |
|---|---|---|
| base | `heading = atan2(vy, vx)` in deg (the ego frame `write_obs` uses: x' = (cos yaw, sin yaw)) when `|v_h| >= 100 u/s`, else the CURRENT yaw | 0 |
| yaw target | `base + cmd` | `cmd` |
| yaw delta | `wrap180(target - yaw)`, clamped to `+-yaw_rate_max_deg` (10) | same |
| pitch | target clamped to `[-70, 30]` first, `delta = clamp(target - pitch, +-pitch_rate_max_deg)`, then the existing `[-70, 30]` clamp | same |

`wrap180` takes the short way round: yaw 170 with target -170 turns +20,
never -340. The delta is applied exactly where the delta path applies its
delta, so **`--yaw-blend` still filters the applied delta and obs slots
10/11 echo `applied delta / ceiling`** (a target that is reached echoes
0). NaN / infinite targets apply 0. The low-speed fallback makes the
velocity-frame command a bounded delta on the start platform (a +37
command turns +10 per tick until the player moves), and the switch to the
velocity frame happens at 100 u/s, below walking speed (250).

**Why the velocity frame recomputes every tick.** The policy decides once
per `act_every` (4 ticks) but the core re-derives the target from the
CURRENT heading on every tick, so the view tracks a rotating velocity
frame continuously between decisions; a world target held for 4 ticks
would staircase during a turn. And note what the zero action is: at
`sv_airaccelerate 100` the strafe optimum holds the wishdir perpendicular
to the velocity, i.e. the view ALONG the velocity - offset 0. In the delta
parameterisation that optimum was the constant `k = +-1`, which the warp
places on its steep part (`dK/du = 4.6`); here it is the offset the warp
resolves finest (`3.5 deg` per unit `u` at zero, so sigma 0.3 in z is
about 1 deg of offset), and the observation carries the current offset
directly: slots 0/1 are the velocity in the ego frame, whose angle IS
`heading - yaw`.

`surf_step`, `surf_pm_step_single` and the play client ignore the field.
An out-of-range `view_mode` is refused at `surf_create`.

### The policy (`train_fast.py`)

`--view-absolute {velocity,world}` requires `--view-continuous`; config
key `view_absolute` (written only when set, restored from a checkpoint on
resume, refused if a resumed checkpoint carries a different mode - the
same weights would write a row the core now READS differently, and world
mode has a third head). z is still THE action PPO scores; only the
environment's deterministic map of it changes (`view_from_z_t(z,
pitch_max, absolute)`, numpy twin `surfgym.view.view_from_z_abs`):

* **velocity**: `z = (z_yaw, z_pitch)`, `u = tanh z`, yaw target =
  `off_warp(u) = 180 sign(u) (e^{b|u|} - 1) / (e^b - 1)` with `b = 2 ln
  17`, so `u = +-0.5 -> +-10 deg`, `u = +-1 -> +-180 deg`; `d off/du` is
  3.5 deg at zero, 60 at +-10 deg, >1000 at the ceiling. Inverse: `u =
  sign(off) ln(1 + |off|/180 (e^b - 1)) / b` (`off_warp_inv`). Same
  tensor shapes as the delta policy (`view_head (2, hidden)`, `log_std
  (2,)`), same init draws.
* **world**: `z = (z_c, z_s, z_pitch)`, `(c, s) = tanh(z_c, z_s)`, yaw
  target = `atan2(s, c)` in degrees - the cos/sin reading the user
  suggested; the seam disappears because the head never emits an angle.
  **The norm of `(c, s)` is ignored**, which has a consequence: with a
  fixed sigma in z the effective ANGULAR noise is roughly `sigma (1 -
  u^2) / |u|`, so it shrinks as the mean vector grows and the policy can
  quieten its own exploration without touching `log_std`. `view_head (3,
  hidden)`, `log_std (3,)`, still registered last. **With the scratch
  baseline's `gps` off the policy's scalars carry NO absolute heading
  (slots 7/8 are dropped, `SCALAR_NOGPS`)**, so a world target has to be
  inferred from the depth image; the velocity mode has no such dependency.
  Say so when the two are compared.
* **pitch (both)**: one Gaussian, target = `-20 + 50 tanh z`, the core's
  `[-70, 30]`.

`sample_view` / `logprob_entropy_view` / `gauss_*` are generic in the
number of heads (`Policy.n_z`, 2 or 3); the rollout's `b_z` / `static_z`
are `(.., n_z)` while the row the core gets stays `(N, 2)`. Greedy =
`z = mu`; the eval wrappers read `policy.view_absolute` and publish the
target row, so the trainer's evals and `tools/record_ckpt.py` (which
mirrors the key and builds its core with the matching `view_mode`) run
unchanged. The `step` line's `sig` lists every head. `act/yaw_side_agree`
keeps its rule in velocity mode (z > 0 = the view leads the velocity to
the left, read against the A key) and reports 0/0 in world mode (a target
angle has no turn direction in the buffers).

**Refused** (a clear message, not a silent misread): `--view-absolute`
without `--view-continuous`; a resume under another mode; a BC file, a
planner line or a prefix line whose view rows are of another mode (see
"The absolute mode through the tools" below - the planner, plan_to_bc,
expert_dagger, line_fragility and `--bc-file` all work in the mode since
2026-09-06 evening; before that they refused it).

### How to launch it (it is the DEFAULT since 2026-09-06)

The user made the velocity-frame mode the default action space of every
run that starts from nothing (CLAUDE.md section 2; the numbers are there).
`tools/run_arm.sh` reads `VIEW=abs|delta|bins` (default `abs`) in its
SCRATCH and MULTIMAP branches and appends the flags before the trailing
`"$@"`; `tools/launch_local.ps1` reads the same variable for every scratch
preset (`scratch_chunk` excepted - `--chunk` codes decode into bin
distributions and the flag refuses them); `tools/wave/run_exit_ab.sh` and
`rent_expert_box6.sh` export it into the loop's environment on the box for
a scratch seed. The RESUME paths pass no view flag at all: the trainer
restores `view_continuous` / `view_absolute` from the checkpoint and
refuses a mismatch, so a resumed checkpoint keeps whatever mode it carries.

    SCRATCH=1 bash tools/run_arm.sh cyABSV                 # abs, the default
    VIEW=bins SCRATCH=1 bash tools/run_arm.sh cyCTL        # the old bins
    VIEW=delta SCRATCH=1 bash tools/run_arm.sh cyDELTA     # per-tick deltas
    SCRATCH=1 bash tools/run_arm.sh cyABSW --view-absolute world   # trailing flags win

`tools/wave/box_finish.sh` / `box_relaunch.sh` pass `$*` -> `run_arm.sh`
and inherit the default (`ARM_ENV="SCRATCH=1 ... VIEW=bins"` to opt out).
The box has to build the ABI-9 core from this branch (`build.sh`; an
ABI-8 DLL refuses to load). Locally: `powershell -File
tools\launch_local.ps1 scratch_ablate cyABSV` (`$env:VIEW = "bins"` first
for a control).

### What is pinned (`tests/python/test_view_absolute.py`)

(a) the core: targets reached within the clamp and held (37 deg in 4
ticks), the seam (170 -> -170 is +20; 10 -> 350 is -20; 370 is 10), the
heading convention against the observation's ego frame (offset 0 leaves
`obs[1] = 0`), a velocity rotating 5 deg/tick under a CONSTANT row tracked
to 1e-3 deg for 39 ticks and lagging at exactly the clamp when it rotates
15 deg/tick, the low-speed fallback (50 u/s: +37 turns +10/tick, +3 turns
+3/tick, -200 turns +160 -> +10; 100 u/s is the frame, 99 is not), pitch
bounds and rate for both modes (30 in 23 ticks at 1.33, -80 settles at -70
with a 0 echo), NaN/inf apply 0, `--yaw-blend 0.5` gives 5 then 7.5 deg
with slot 10 = 0.5 / 0.75, mode 0 is the delta path, view_mode 3 refused,
ABI 9; (b) `off_warp` anchors / monotone / odd / inverse / resolution,
torch == numpy for both modes and the delta default unchanged, the norm
ignored (`(2, 2)` and `(0.5, 0.5)` are both 45 deg), the mixed
distribution's log-prob and entropy on the 2- and 3-head layouts against a
hand computation and the update's recomputation, the Policy's tensors per
mode (velocity == delta tensor for tensor; world adds the third head last),
the eval wrappers publish in-range targets with greedy = mean; (c) a CPU
scratch smoke per mode on the toy scratch argument set (finite losses,
|kl| < 0.05, eval trajectory, config + checkpoint carry the mode,
`record_ckpt` records it and reports the core's `view_mode`, a flagless
resume restores the mode, a wrong-mode resume is refused), and the two
refusals. The existing `test_view_continuous.py` (delta path golden files,
flag-off trainer identity, flag-on delta smoke) passes on the ABI-9 DLL,
and the delta mode of this trainer was compared by hand against
`origin/contyaw`'s trainer on the same smoke: identical.

## Pitch head discipline (`--pitch-entropy`, the log sigma cap; 2026-09-06 evening)

**The defect.** Pitch has no physics effect in the air (`pm.c` projects
it out of the wishdir; it only aims the depth camera), so PPO has next to
no gradient on the pitch head and the entropy bonus wins by default.
Measured on two seeds of the absolute mode: the pitch head's sigma
climbed to the `LOG_STD_MAX` clamp - **2.718 pre-tanh by 3-8B steps** -
so the camera target `-20 + 50 tanh z` was drawn nearly uniformly over
`[-70, 30]` every decision, the policy looking at the floor or the sky at
random while the yaw head worked.

**The fix, absolute mode only** (`PITCH_LOG_STD_MAX_ABS` in
`train_fast.py`; the delta mode and the bins are untouched op for op,
pinned below):

* `--pitch-entropy W` scales the PITCH head's (the last Gaussian's) share
  of the entropy bonus: `logprob_entropy_view(.., pitch_ent=W)` returns
  `ent_c + H(yaw heads) + W * H(pitch)`; at `W = 1.0` it returns the
  pre-flag expression op for op (the same tensor), at `0.0` the pitch
  term is simply absent. The log-prob is the same joint whatever `W`.
  **Default 0.0 under `--view-absolute`, 1.0 everywhere else**; refused
  without `--view-continuous`. Written to the config (`pitch_entropy`)
  ONLY under the mode and restored on resume (`record_ckpt` lists it as
  TRAIN_ONLY). `train/entropy_loss` in the mode therefore no longer
  contains the pitch term.
* The pitch head's log sigma is **capped at log 0.5** (`sigma <= 0.5`,
  i.e. the target's spread stays inside about +-25 deg of the mean):
  `Policy.log_std()` clamps through a per-head ceiling
  `view_std.log_std_hi = [LOG_STD_MAX, .., log 0.5]`, a NON-persistent
  buffer - the state_dict keys are the delta policy's and a checkpoint
  from before the cap loads key for key. `Policy.project_log_std()` pulls
  the RAW parameter under every head's ceiling after each optimizer step
  and once on resume (printed when it moved something): a raw value
  parked above a clamp has a zero gradient through it, so without the
  projection PPO could never shrink the pitch sigma again. The clamp is
  `torch.clamp(max=)`, not `torch.minimum` - at the tie `minimum` splits
  the gradient 0.5/0.5 between its equal inputs, `clamp` passes it whole.
  (In the mode the projection also keeps the yaw head's raw value at or
  under `LOG_STD_MAX`; in the delta mode nothing is projected, as before.)
* Resuming a checkpoint trained before this (the rented cyABSV boxes)
  gets the default `pitch_entropy 0` and its pitch sigma projected from
  2.718 to 0.5 at load, with a line in the log saying so.

**Measured (CPU smokes, the toy scratch set, `--ent 0.5 --lr 1e-2`,
24 Adam steps; the ledger 2026-09-06 evening has the numbers):** with
`--pitch-entropy 1` both sigmas climb; by default the yaw sigma climbs
the same way and the pitch sigma ends within noise of its start; a
longer run at `lr 2e-2` with the term ON drives the pitch sigma INTO the
cap - `sig ../0.500` on the step line and the checkpoint's raw pitch log
sigma AT log 0.5, never above it.

## The absolute mode through the tools (planner, distil, DAgger, replay, fragility)

One rule everywhere: **a view row means what the checkpoint's
`view_absolute` says** - `(K, pitch deg/tick)` for a delta checkpoint,
`(yaw offset deg in the velocity frame, pitch target deg)` for
`velocity`, `(yaw target deg, pitch target deg)` for `world`
(`surfgym.view.view_desc`) - and every core a tool builds carries the
matching `view_mode` (`beam_tas.build_sim`, hence `plan_to_bc`,
`expert_dagger.open_core` and `line_fragility` too). The mode-generic
helpers are `surfgym.view.view_from_z_any` / `z_from_view_any`
(dispatching to `view_from_z` / `z_from_view` or `view_from_z_abs` /
`z_from_view_abs`), `wrap180`, `yaw_limit` (20 K or 180 deg).

* **`tools/beam_tas.py`**: the Policy is built with the mode, the wrappers
  publish `view_from_z_t(z, .., mode)` (proposals sample z from the
  absolute heads, greedy envs take `z = mu`), `hist_v` / `view` /
  `view_all` carry the executed TARGETS per decision and the npz gains
  **`view_mode`** (0 delta / 1 velocity / 2 world; `summary.json` gains
  `view_absolute`). `--dedup` hashes 2 or 3 z heads (12 bits each from
  bit 24; the 2-head packing is unchanged). `--macro-yaw track` writes
  the analytic TARGET (`macro_yaw_abs`): offset 0 for either held key in
  the velocity frame (the strafe optimum is the view along the velocity
  and the core re-derives the frame every tick), the current heading of
  `v_h` in world mode (NaN below the core's 100 u/s frame floor).
  `--branch-at` draws a target (jitter 0: `off_warp(u)`, u uniform, in
  velocity mode - dense near "look along v"; a uniform heading in world
  mode) or an offset of J DEGREES (jitter J), wrapped to [-180, 180) on
  apply; `--branch-grid` yaw offsets are DEGREES on the policy's own
  target, default `-30,-15,-5,0,5,15,30` (a first guess - the 3 K/tick
  grid has no target-space twin - not a measured optimum). `--prefix-line`
  checks the line's `view_mode` against the checkpoint's and refuses a
  discrete line under an absolute checkpoint (bins are per-tick deltas,
  not targets). `--robust`, `replay_arc`, `run_episode`, `Playback` and
  commit mode step the rows through the core unchanged.
* **`tools/plan_to_bc.py`**: reads `view_mode` (absent = 0), refuses a mix
  of modes among the plan files and a plan of another mode than the
  checkpoint; the z targets are `z_from_view_any` (velocity: exact
  inverse warp; world: the radius-0.9 preimage - ONE of many, since the
  norm of `(c, s)` is free); `survivor_view_moments` is generic in the z
  width (3 in world mode); the BC file's meta carries `view_absolute` /
  `view_mode` and `view_zmu` / `view_zsd` are `(R, n_z)`.
* **`surfgym.bc.BCDataset(.., view_absolute=)`**: refuses a file of
  another mode ("its view rows are ... but this trainer's heads write
  ...") and a discrete file under an absolute mode; `--bc-file` is now
  ALLOWED under `--view-absolute` (the trainer passes its mode). The BC
  view term is the delta mode's: the Gaussian NLL / cross-entropy of the
  rows' z at the fixed `BC_VIEW_SIGMA`.
* **`tools/expert_dagger.py` / `surfgym.dagger`**: the decider publishes
  targets, `relabel_windows(.., view_absolute=)` maps the copies' views
  through `z_from_view_any` (`population_view_moments` generic in the z
  width), the rows' point z is filled in the mode, the dagger file's meta
  carries `view_absolute` and `merge_bc_datasets` refuses an elite file
  of another mode. `--label-target gumbel` stays refused under the flag.
* **`tools/line_fragility.py`**: replays on a core of the checkpoint's
  mode, checks the line's `view_mode`, and its `bin` perturbation is
  **+-1 deg on the yaw target** (about what +-0.25 K turns in one
  decision at flight speed).
* **`tools/expert_loop.py`** needs nothing: every phase reads the mode from
  the checkpoint / the plan, and a scratch seed gets the launcher's
  default.

Smoked end to end on the CPU (the ledger has the numbers): a tiny
velocity-mode scratch policy -> `beam_tas` waves (lines with `view` /
`view_mode 1`, replay exact) -> `plan_to_bc` (rows with mode-tagged
moments) -> `BCDataset` -> a PPO+BC resume -> `record_ckpt`, and one
`expert_loop` round.

**Pinned (`tests/python/test_view_absolute.py` (d) and (e)):** the
entropy split and its gradient (0 on the pitch log sigma at `W = 0`, the
old tensor at `W = 1`); the cap buffer (no state_dict key, log 0.5,
projection, a live gradient at the cap, an old checkpoint loading); the
two-arm sigma smoke and the cap smoke with a resume; the delta mode of
this trainer bit-identical to the trainer of the commit before the cap
(config, progress.csv, trajectory, weights, Adam moments); an absolute
planner line round-tripping through `replay_line` on a view_mode-1 core
and `plan_to_bc.load_plans` refusing a mix; `BCDataset` on absolute files
(velocity, world with 3-wide moments, point targets) and its refusals; the
planner's edits in degrees (draws, apply, grids, `macro_yaw_abs`,
`MacroHold.decide_view`), the 2/3-head hash and `build_sim`'s view_mode.
