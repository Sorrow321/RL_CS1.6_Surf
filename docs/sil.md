# `--sil-coef`: self-imitation learning as an auxiliary PPO loss

Built 2026-09-14 from `docs/litsurvey-labyrinth.md`, mechanism 5 (Oh, Guo,
Singh, Lee, "Self-Imitation Learning", ICML 2018). Default off; with the flag
absent the trainer is byte-identical (config dump, CSV header, every
per-iteration number - pinned by `tests/python/test_sil.py`).

## Why

Inside a region the policy can reach but cannot yet survive (unitfarmer2's
pit), a rollout holds a handful of attempts and their on-policy advantages
are a thin, sign-mixed gradient: the archive bench put the policy in the
right place hundreds of thousands of times and all attempts still died at
the second ramp. SIL keeps every transition whose realised return target
beat the critic's own estimate and trains the actor toward the actions that
produced it, weighted by the gap, while pulling the critic up to the realised
return. It never pushes an action down: rows with `R <= V(s)` contribute
nothing. So whichever attempt got furthest is learned from even when none
succeeded.

## What it does (`python/surfgym/sil.py`, wired in `train_fast.py`)

* After every rollout's GAE, every transition with `R > V(s)` - `R` the
  bootstrapped return target in the units the value loss uses (normalised
  under `--ret-norm`), `V` the rollout's own value prediction - goes into a
  FIFO buffer of `--sil-buffer` rows (default 50,000): the scalar row, the
  image frame (the buffer's own dtype), the action 6-tuple, the stored view
  `z` (absolute view) and the privileged critic columns when present. If more
  than the capacity qualify in one rollout the largest gains win.
* In every PPO epoch the first `--sil-batches` (default 4) minibatches also
  draw `--sil-batch-size` (default 512) buffer rows with priority
  proportional to `(R - V)_+` and add
  `coef x [ -log pi(a|s) (R - V)_+ + 1/2 (R - V)_+^2 - ent x H ]`
  to the loss before the one backward. The rows just scored get their gain
  refreshed under the current critic (lazy V), so a row the critic has
  caught up with stops being drawn.
* The log-probs come from the same padded / continuous-view helpers the PPO
  ratio uses (`logprob_entropy_padded`, `logprob_entropy_view`), untempered
  (there is no ratio to match), through `policy.forward_split` with the
  privileged columns when the critic has them.
* Logged as `sil/buffer` (rows held), `sil/mean_gain` (mean `(R - V)_+` over
  the buffer), `sil/loss` (the last SIL minibatch's loss); the step line
  prints `sil <rows> +<kept> gain <g> loss <l>`. The buffer is not
  checkpointed (a resume starts empty); the five knobs ride in the config
  and a flagless resume restores them.
* Refused with `--rnn`, `--chunk`, `--ddp`, `--frame-stack > 1`, the action
  masks, `--yaw-cond` and the out-of-policy bursts (`--ez-eps`,
  `--spawn-burst`): the SIL step re-scores stored rows through the flat
  single-map path and none of those change it. Skipped during
  `--critic-warmup`.

## Flags

| flag | default | meaning |
|---|---|---|
| `--sil-coef` | 0 (off) | the loss coefficient |
| `--sil-buffer` | 50000 | buffer capacity in transitions |
| `--sil-batches` | 4 | SIL minibatches per PPO epoch |
| `--sil-batch-size` | 512 | rows per SIL minibatch |
| `--sil-ent` | 0 | entropy coefficient inside the SIL term |

Recommended first arm on unitfarmer2 (with the working explorer, cell
novelty 4x with decay, ratchet, keys T): `--sil-coef 0.1`. Oh et al. used
`beta^sil = 0.1` on Atari with the same units (value-loss weight 0.5 vs
SIL's implicit 0.5 on the squared gap), and our value loss coefficient is
also 0.5, so 0.1 keeps the SIL term at a tenth of the PPO surrogate's scale;
`--sil-batches 4 --sil-batch-size 512` is 2,048 rows per epoch against the
262k-row rollout. Watch `sil/mean_gain`: it must come DOWN over a run as the
critic catches up (the mechanism working) - a flat or rising gain with an
unchanged start line says the imitated rows are not the ones that matter.

## Pinned (`tests/python/test_sil.py`, 6 tests)

The buffer keeps only rows with `R > V`, keeps the top gains when over
capacity, evicts FIFO, samples by gain, refreshes gains; the loss is zero
below the critic, detaches the gap in the policy term and its value term
raises `V`; the CPU smoke runs, logs the columns, prints the note and
resumes; and the flag-off trainer reproduces the parent commit's trainer on
the toy set (config dump, CSV header, every non-timing column).

## `--sil-temper 1`: score the replay at the rollout's temperature (2026-10-21, r3b2)

Under `--unstuck` the rollout draws the view heads from `N(mu, (sigma (1+T))^2)` and the
keys from `softmax(logits / (1+T))`, and PPO's ratio scores every row under that tempered
measure (docs/unstuck.md). The SIL step above scores its replayed rows UNTEMPERED, so a row
drawn at `sigma (1+T)` reads as a `(1+T)`-sigma outlier: per row the imitation gradient on
`log_std` is `-((z - mu)^2 / sigma^2 - 1)`, which averages `-((1+T)^2 - 1)` (-3 at T = 1).
Gradient descent therefore raises sigma on every SIL minibatch, whatever the rows' content.

`--sil-temper 1` hands the SAME temperature tensors (`temp_t` / `tempv_t`, the ones
`mb_step` reads) to `logprob_entropy_view` / `logprob_entropy_padded` inside `sil_step`, so
a replayed row is scored under the distribution that drew it: no net push on `log_std` from
sampling noise; the mean is still pulled toward the imitated action, `(1+T)^2` times more
gently per row. The temperature is the CURRENT one (rows drawn a few iterations earlier at
a slightly different T are scored at today's), which is exact whenever T sits at its cap.

* Default 0 = the shipped call (`SIL_TEMP is None`); the config key is only written when on;
  a flagless resume restores it; `--sil-temper 1` without `--sil-coef` is refused, and so is
  `--cc-temp-scale` (its temperature is per env; a buffer row has none). Without an
  `--unstuck` temperature it is the untempered SIL (the trainer says so).
* TRAIN_ONLY in `tools/record_ckpt.py`.
* Pinned (`tests/python/test_sil_temper.py`): rows drawn at temperature `1+T` push `log_std`
  up by `(1+T)^2 - 1` untempered and by ~0 tempered (T = 0.5, 1, 2), both pull `mu` toward
  the imitated z (tempered `(1+T)^2` x weaker); the sidistic CPU smoke with `--unstuck`
  forced hot runs, prints the note, dumps `sil_temper` and resumes it; the flag is refused
  without `--sil-coef` and absent from the dump when off; `--sil-coef` without the flag is
  bit-identical to the pre-flag trainer (config, every non-timing progress.csv column).
