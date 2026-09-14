# `--int-split`: a two-head critic with a NON-EPISODIC intrinsic return

Mechanism 3 of `docs/litsurvey-labyrinth.md`, for gap 2. Source: Burda,
Edwards, Storkey, Klimov, "Exploration by random network distillation",
ICLR 2019, sec. 2.3 (the value split and the non-episodic intrinsic
return; sec. 3.7, "dancing with skulls", is the failure mode to watch).

Default OFF, byte-identical: with the flag absent the config dump gains no
key, the model has no extra tensor, no RNG draw moves, and every number
the trainer writes (progress.csv minus fps, the eval trajectory, the
weights, the Adam moments) is the parent commit's
(`tests/python/test_int_split.py`, which runs the parent commit's whole
`python/` tree - trainer AND reward - against the flag-off trainer).

## The problem it is for

Today the count bonus rides INSIDE the episodic race reward. GAE cuts the
return at every episode end, so a dive that dies forfeits every unit of
novelty it was heading for: the critic learns that the pit is worth
nothing, and the agent is risk-averse for the exploration reason on top of
the racing one. Burda's motivating paragraph is this situation verbatim
("her future return will be exactly zero if she gets a game over, which
might make her overly risk averse"), and their fix is to treat the
INTRINSIC return as non-episodic while the extrinsic one stays episodic -
which needs two value heads, each fitted on its own return with its own
discount. Their own ablation says the gain is the NON-EPISODIC return, not
the second head as such: two heads with both streams episodic bought
nothing.

With `--death-charge` (mechanism 1) this is "death expensive for the
racer, free for the explorer".

## What the flag does

Four things, and nothing else moves:

1. **The reward** (`surfgym/rewards.py`, `RaceReward(int_split=True)`).
   The count bonus - `int_coef / sqrt(count + 1)` in cell or edge mode,
   times T under `--curiosity-cond`, times the `--unstuck-int`
   multiplier, whatever it is - is written to `reward.int_r` (a per-call
   `(N,)` float32 vector, zero on ended rows, rebuilt from zero at every
   call) instead of into `r`. The count table, the sqrt law,
   `int_paid` (the `int x/ep` on the step line), `rare_entry` and the
   archive are unchanged. Bit for bit: `r_split + int_r == r_control` on
   every call (test (a)).
2. **The critic** (`train_fast.Policy(int_split=True)`). `int_head =
   Linear(hidden + priv_hidden, 1)`, the same input block as `value_head`
   (the vf tower output, plus the privileged block under `--priv-critic`)
   and the same init (orthogonal gain 1, zero bias), REGISTERED LAST so
   every existing parameter keeps its index in `policy.parameters()` and
   its draw comes off the same RNG state as before. `_value` returns
   `(B, 2) = [V_E, V_I]`; `heads()`' `squeeze(-1)` is a no-op on a 2-wide
   row, so each consumer picks its column: the rollout writes
   `static_val` / `static_vint`, the truncation bootstrap takes column 0,
   GAE takes both.
3. **The returns.** The extrinsic stream is exactly what it was: `gamma ^
   (act_every x chunk)`, cut by `nonterm = 1 - done`, bootstrapped through
   truncations, `--ret-norm` on V_E, `ret = adv_E + V_E` is V_E's target
   and the explained variance. The intrinsic stream runs its own GAE over
   `b_rint` / `b_vint` with **no `nonterm` mask**: a death or a truncation
   is an ordinary transition into the next episode's spawn, and
   `V_I(spawn)` is what bootstraps it. Its discount is `--int-gamma`
   (below), its lambda is `--gae`, V_I is fitted RAW (never normalised).
   The policy gradient's advantage is `A_E + int_adv_coef x A_I`, and it
   enters the SAME per-minibatch normalisation the extrinsic advantage
   alone used to (and the `--cc-buckets` one, and `--tail-weight`'s
   reweighting).
4. **The loss.** `vl = 0.5 MSE(V_E, R_E) + (int_vf / vf) x 0.5 MSE(V_I,
   R_I)`, so `args.vf x vl` in the joint loss IS `vf x MSE_E + int_vf x
   MSE_I`; `train/value_loss` reports that combined term (in the units the
   loss weights it), and `--critic-warmup` optimises both heads
   (`int_head.` is a critic prefix).

### The discount convention

`--gamma` is PER PHYSICS TICK and the trainer raises it to the decision
(`gamma ** (act_every x chunk)`, CLAUDE.md). **`--int-gamma` is PER
DECISION.** Reason: Burda's `gamma_I = 0.99` is per agent step at their
frame skip of 4, which is exactly this trainer's decision at `act_every 4`;
raised to the tick it would be `0.99 ^ 4 = 0.96` per decision, a 25-decision
(1 s) horizon, too short to reach back over a death to the novelty behind
it. `0.99` per decision is a 100-decision horizon = 400 ticks = 4 s at
`act_every 4 / 10 ms` (the trainer prints the number). Burda found raising
it to 0.999 HURT; the extrinsic 20 s horizon is untouched.

## Flags

| flag | default | meaning |
|---|---|---|
| `--int-split` | off | the mechanism; needs `--reward race` and `--int-coef > 0` |
| `--int-gamma` | 0.99 | intrinsic discount PER DECISION |
| `--int-vf` | 0.5 | coefficient of `0.5 MSE(V_I, R_I)` in the joint loss |
| `--int-adv-coef` | 1.0 | `A = A_E + coef x A_I` before normalisation |

Config keys, written ONLY when on, next to the `int_rare` keys:
`int_split: 1, int_gamma, int_vf, int_adv_coef`.

`progress.csv`, only when on (after the `cc/*` block, before `dip/*`):

| column | what |
|---|---|
| `int/ret_mean` | mean intrinsic return `R_I` over the rollout buffer |
| `int/v_mean` | mean `V_I` over the buffer |
| `int/adv_abs` | mean `|A_I|` (before `--int-adv-coef`) |
| `int/ev` | V_I's explained variance of `R_I` (1 = perfect, <= 0 = no fit) |

`rollout/ep_rew_mean` is the EPISODIC reward alone under the flag (the
novelty is no longer in it); the step line's `int x/ep` still reports the
novelty paid per episode.

## Resume rules, refusals

* The critic is 2-wide, so a resume has to agree with the checkpoint in
  BOTH directions: a checkpoint carrying `int_head` resumed WITHOUT the
  flag is refused ("this checkpoint was trained with --int-split ..."),
  and a ONE-HEAD checkpoint resumed WITH the flag is refused ("cannot
  warm-start a ONE-HEAD checkpoint") - a fresh V_I would start from noise
  against a trained V_E. It is a SCRATCH recipe.
* A 2-wide checkpoint resumed WITH the flag continues; `int_gamma`,
  `int_vf`, `int_adv_coef` come back from the config when not given.
* `tools/record_ckpt.py` MIRRORS `int_split` (a strict load needs the
  tensor) and declares the three constants TRAIN_ONLY, so the record gate
  passes. Other tools that rebuild a `Policy` from a config
  (`beam_tas.py`, `credit_diag.py`, `diversity_bench.py`,
  `expert_dagger.py`, `transplant_view.py`) do NOT mirror it yet and will
  fail the strict load on a 2-wide checkpoint - add
  `int_split=bool(cfg.get("int_split"))` when one of them is needed.
* Refused with `--rnn`, `--chunk` / `--codebook`, `--ddp`, `--rnd-coef`,
  `--bc-file` (the 2-wide value output, the second GAE stream and the
  intrinsic value loss live in the flat single-process rollout, GAE and
  `mb_step`; the RND bonus would stay episodic; the BC value term reads
  V(s) as one number), with `--vf 0`, and with `--priv-critic` widening a
  plain checkpoint. `--priv-critic` from scratch works (int_head takes
  the same privileged block). `--reward-per-decision` is exact (one bonus
  per decision) and allowed. `--maps` is wired (per-slot `int_r` gathered
  in slot order) but untested.

## Where it goes on unitfarmer2

The batch-17/18 base (`SCRATCH=1 MAP=maps_pool/surf_unitfarmer2.bsp`
through `tools/run_arm.sh`, `--goal-cell 48 --respawn-margin 1
--gate-boxes docs/gate_boxes.json --ckpt-every 250e6 --seed 0`, 1B
steps) plus:

```
--race-ratchet --int-mode cell --int-coef 1.0 --respawn-frac 0.7 \
  --unstuck --unstuck-patience 2e7 --unstuck-period 2e7 --unstuck-max 1 \
  --unstuck-temp 1 --unstuck-temp-heads keys --unstuck-reach alive \
  --unstuck-reach-start-only \
  --int-split                                   # uf2SPLIT
```

and the pairing the survey names (mechanism 1 + 3, "death expensive for
the racer, free for the explorer"):

```
  ... --int-split --death-charge 1.0 --time-pen 0    # uf2SPLITDC
```

Both against batch 17's `uf2DC` / batch 16's cell-4x control at the same
seed. Defaults (`0.99 / 0.5 / 1.0`) first; `--int-adv-coef 2` is the one
knob to move if `int/ev` fits but the entry does not change.

**Falsified by** (survey): pit contact from the true start does not rise
against the same `--int-coef` with one episodic head; or the agent starts
dying on purpose to farm novelty (Burda's own warning) - watch the death
cadence and `ep_len_mean`, and `int/ev`: a V_I that never fits
(`int/ev <= 0` after the first ~100M steps) means the intrinsic advantage
is noise and the arm is answered before the entry is.

## Tests

`tests/python/test_int_split.py` (CPU only, the cannonball toy set):
(a) the reward split identity in both novelty modes and under
`--curiosity-cond`, the ended-row zero, `int_paid`, the `int_coef > 0`
refusal; (b) the intrinsic GAE recursion on a hand case where an episode
ends inside the rollout (the extrinsic stream is cut, the intrinsic one
carries the post-death novelty back over the death) and a source pin that
the trainer's loop has no `nonterm`; (c) `int_head` last, the shared
tensors bit-identical to the flag-off model under the same seed, the
`(B, 2)` output with column 0 the flag-off value, `--priv-critic`; (d) the
flag-on smoke with 48-tick episodes (config, columns, a 2-wide checkpoint
with Adam moments, `record_ckpt` loading it), the flag-on resume, the two
refused resumes, two flag refusals, and the flag-off bit-identity against
the parent commit's `python/` tree on the bins and the absolute view.
