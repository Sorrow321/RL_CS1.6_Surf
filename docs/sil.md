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
