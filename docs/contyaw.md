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
  `N(zmu, zsd)` (`--bc-target dist`): `log sigma + log sqrt(2 pi) + (zsd^2 +
  (zmu - mu)^2) / (2 sigma^2)`. A row a single line survived at has
  `zsd = 0` and the two coincide. `bc_log.csv` APPENDS a `bc/view_mse`
  column under the flag; the console line adds `view-mse`.
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
