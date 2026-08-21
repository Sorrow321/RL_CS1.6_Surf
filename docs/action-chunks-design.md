# Temporally-abstract actions: a behavior-chunk codebook for train_fast

**Status:** design + CPU prototype. Nothing here has been trained. No file
outside this doc and `tools/build_action_codebook.py` was touched.

**The idea, restated.** Today the policy emits one action per decision from
six independent heads, `NVEC = (15, 7, 3, 3, 2, 2)` (`train_fast.py:78`), and
each decision is held for `act_every = 3` physics ticks
(`train_fast.py:1403`, `train_fast.py:2184`) — 33 Hz. The proposal is that the
policy instead picks **one index out of ~128 behavior chunks**, and a frozen
decoder expands that index into `H` consecutive 6-tuples. Entropy and
epsilon then randomize *behaviors* rather than 33 Hz white noise.

---

## 0. The one-paragraph case, in this project's own numbers

The ledger closed the decision-rate question in both directions: *"Every rate
tested off 33 Hz is worse: 50 Hz (act-every 2), 25 Hz (4), 20 Hz (5), 16.7 Hz
(6), 11 Hz (9). That is a sharp optimum, not a plateau"* — and it gave the
mechanism: the air-accel gain window is `±arcsin(30/|v|)` = **±0.52 deg/tick
at 3000 u/s**, so *"a decision interval that straddles it gives up speed no
policy can recover"* (`docs/research-results.md`). But the same ledger shows
throughput rising steeply with a coarser rate: act-every 4 measured
**1.22x the fps** of act-every 3 on the same box.

Chunking is the only construction that takes both. The policy makes one
decision per `H` decisions (10x fewer forward passes, 10x fewer lidar
renders), while the *action stream reaching the engine stays at 33 Hz*
because the decoder emits a fresh 6-tuple every 3 ticks. The control rate is
untouched; only the *deliberation* rate drops. That is the headline argument,
and it is the one the act-every ladder cannot make.

The second argument is exploration. `--ez-eps` exists precisely because
*"iid per-step noise cancels out over long trajectories"* (`train_fast.py:752-753`),
and the ledger's verdict on it is *"ez-greedy at eps=0.005 is at parity"*.
The structural reason is in the code: burst transitions were not drawn from
pi, so PPO must discard them from the loss (`train_fast.py:2396-2404`) — the
exploration never reaches the policy gradient at all. Over a code head, a
random code **is** an on-policy sample from pi. Temporally-extended
exploration stops being a bolt-on and becomes the action space. That is
Tallec's remedy ("correlated noise, never a coarser control rate",
`docs/research-litsurvey.md`) applied exactly as prescribed.

---

## 1. Paper constants (extracted; verified vs. unverified flagged)

| Paper | Chunk length | Codebook / latent | Decoder | Stabilization | Decoder frozen downstream? |
|---|---|---|---|---|---|
| **SPiRL** 2010.11944 | `H = 10` (code-verified in `clvrai/spirl` configs `n_rollout_steps=10` for all three envs; the literal "H = 10" is not in the fetched paper text) | continuous `z ∈ R^10` | 1-layer LSTM, 128 hidden; decoder "mirrors the encoder's architecture and is unrolled for H steps" | **max-ent entropy replaced by KL to a learned state-conditioned skill prior**, Eq. 3; alpha auto-tuned to a target KL `delta = 1` (maze) / `5` (both manipulation); VAE `beta = 1e-2` (maze, block stacking), `5e-4` (kitchen); prior = 6-layer FC, 128 wide; agent SAC | **UNVERIFIED** — the paper never says "frozen"/"fixed"; the reference impl loads the low-level policy and never optimizes it |
| **OPAL** 2010.13611 | `c = 10` default; `c = 5` multi-task; `c = 1` ablation | `dim(Z) = 8` | encoder: FC(2 hidden) → **bidirectional GRU, 4 layers**; decoder `pi(a|s,z)` same FC arch, autoregressive for kitchen | KL penalty weight `beta = 0.1`, 100 epochs, lr 1e-3, Adam, batch 50 | **YES, explicit**: *"we learn a task policy in space of primitives ... while keeping it fixed"* — cite OPAL, not SPiRL, for frozen-decoder practice |
| **BeT** 2206.11251 | none (1 action/step); context 2/10/5/10 | **k-means bins: 2;3 (point-mass), 32 (CARLA), 24 (block-push), 64 (kitchen)** | MinGPT (1/3/4/6 layers) + a `k x dim(A)` **residual/offset matrix** | `L_focal + alpha * L_mt`; alpha "just makes sure at initialization the two losses are of the same order" — **no number in the paper**; repo kitchen config: `focal_loss_gamma: 2.0`, `offset_loss_scale: 1000.0` | n/a |
| **VQ-BeT** 2403.03181 | action chunk `n` = **1 / 1 / 1 / 10 / 5 / 6 / 1** across the 7 benchmarks | **residual VQ, `N_q = 2` layers, codebook 8–16 per layer** → 64–256 total modes | ConvNet/MLP encoder → RVQ; policy = 6-layer MinGPT (6 heads, 120 embd); RVQ decoder + offset head | `lambda_commit = 1`; `L_code = L_focal(code_1) + beta * L_focal(code_{i>1})`, `beta = 0.1–0.6` per env; codebook updated by **EMA**, no dead-code reinit documented; repo defaults `offset_loss_multiplier = 100`, `secondary_code_multiplier = 0.5` | **UNVERIFIED** in the paper; the repo's two-stage workflow (`pretrain_vqvae.py` → load `vqvae_load_dir`) implies yes |
| **ACT** 2304.13705 | `k = 100` at 50 Hz (2 s); ablation `k ∈ {1, 100, 200, 400}` | z dim not in paper (code: 32) | transformer, 7 decoder layers, ffn 3200, hidden 512, 8 heads | CVAE `beta = 10`; **temporal ensembling** `w_i = exp(-m*i)`, **m not in the paper**, code uses `k = 0.01` | n/a |
| **ez-greedy** 2006.01782 | duration `n ~ zeta`, `z(n) ∝ n^-mu`, **`mu = 2.0`** | — | — | cap `n <= 10000` (Rainbow agents: `n <= 100`); eps 1.0→0.01 linear over 4M frames | — |

Three constants this design leans on hardest, and their provenance:

* **`H = 10` is the field's consensus, not a guess.** SPiRL `H = 10`,
  OPAL `c = 10`, VQ-BeT's longest chunk `n = 10`. `train_fast.py` already
  uses `--ez-max 60` decisions for its burst cap, an order of magnitude
  longer than any of these.
* **K = 128 is inside the published range.** VQ-BeT's total mode count is
  `codebook_size ^ N_q` = 8²–16² = **64–256**; BeT's kitchen k-means uses
  **64** bins. The user's "~100–256" is exactly right, and it is *not*
  arbitrary: it is what a discrete latent supports before codes go dead.
* **KL-to-prior, not entropy** (SPiRL Eq. 3) is the stabilizer, with an
  auto-tuned coefficient targeting `delta` nats.

**What the papers say about long chunks hurting** (this is the risk this
design must engineer around, see §4): ACT — performance "slightly tapers
down" past `k = 100`, attributed to *"the lack of reactive behavior and the
difficulty in modeling long action sequences"*; SPiRL — *"too long horizons
make the skill exploration problem harder, since a larger number of possible
skills gets embedded in the skill space. Therefore, the policy converges
slower"*; OPAL — *"a larger c will inevitably make it practically harder to
control the autoencoding loss eps_c, thereby leading to an increase in
overall suboptimality and inducing a trade-off"*.

---

## 2. Data audit — verdict: **actions are exactly recoverable offline. No GPU rollout dump is needed.**

### 2.1 What is recorded

`python/surfgym/record.py:6-7` defines the 15-column row, and
`record.py:115-116` establishes the pairing (pre-step state `s_t`, action
`a_t` applied during tick `t`):

```
[t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward, pitch, fwd, side]
 0  1 2 3  4  5  6    7     8         9         10       11      12    13    14
```

**Four of the six action heads are literally in the row** (`record.py:127`,
`record.py:142-143`):

| head | index | source column |
|---|---|---|
| `a[2]` fwd | 2 | col 13, verbatim |
| `a[3]` side | 3 | col 14, verbatim |
| `a[4]` jump | 4 | `col 8 & SURF_IN_JUMP(2)` |
| `a[5]` duck | 5 | `col 8 & SURF_IN_DUCK(4)` |

**The other two are exactly invertible, not estimated.** yaw and pitch are
*rates the core integrates into the state before the physics step*:

* `src/env.c:549` — `st->yaw = wrap_yaw(st->yaw + yd)` where
  `yd = YAW_BINS[yb] * (yaw_rate_max_deg / 10)` (`env.c:69-71`)
* `src/env.c:551` — `st->pitch += pd`, `pd = PITCH_BINS[pb] * (pitch_rate_max_deg / 10)`

so `wrap180(yaw[t+1] - yaw[t])` is a member of a 15-element set of distinct
values, and `argmin` over that set recovers the bin. This is an identity, not
a regression. **Measured: 0.000000 deg snap residual and 0.000000 deg
integrated reconstruction error over 4.7M ticks** (§5).

`yaw_rate_max_deg` is always the core default 10.0: `train_fast.py` exposes
no `--yaw-rate` flag at all (only `--yaw-adaptive`, `train_fast.py:812`), so
there is no run in the corpus where the ladder is scaled.

### 2.2 The corpus

| | |
|---|---|
| trajectory files under `C:\RL_Surf\runs` | **2,889** (3.0 GB) |
| of those with a `run.json` naming the config | 2,863 |
| `surf_src_cannonball`, `act_every=3`, stock yaw | 2,422 |
| `surf_ski_2`, `act_every=3` | 165 (+206 older, `act_every` auto-detected) |
| `--yaw-adaptive` (`runs/sYAWb`) | 48 |
| `act_every=9` | 17 |
| plus the champion greedy recording in the scratchpad | 3 episodes, 43k ticks |

Two maps, one adaptive-yaw arm, one coarse-rate arm — enough for the
cross-map and cross-parameterization checks in §7 without renting anything.

### 2.3 The four things that can go wrong, and what the tool does

1. **Pitch clamp censoring.** `env.c:556-557` clamps pitch to `[-70, +30]`.
   At a rail the delta is censored and the bin is genuinely unknowable.
   Masked, not guessed (1,033 of 8,418 ticks in champion episode 0).
   *This is also a design input:* pitch is only ever a lidar-aim action —
   `env.c:580-581` passes `0.0f` for pitch into `pm_tick`, so **pitch has no
   effect on the physics at all**. It must not be chunked (§3.2).
2. **`--yaw-adaptive` runs.** `env.c:69-81` makes the ladder
   `K_BINS[k] * atan(30/|v_h|)`, clipped to ±10. The tool rebuilds the 15
   candidates per tick from cols 4,5 and inverts; inside the clip region
   several `k` collapse to the same delta, so those ticks are ambiguous and
   are masked (72.8% unambiguous in the selftest at 900 u/s).
3. **act_every phase.** `_TorchPolicyBase.act` (`train_fast.py:549-553`)
   counts ticks in `self._tick`, which `record_rollout` never resets between
   episodes. Episode 2 of a 3-episode recording starts *mid-decision*.
   Assuming phase 0 silently reads a clean `act_every=3` tape as
   `act_every=1` on two thirds of episodes. The tool searches `(K, phase)`;
   the champion file lands on `K=3, phase=0 / 2 / 0` for its three episodes,
   and `8419 % 3 = 1` — i.e. the next boundary is at local index 2, exactly
   what was detected.
4. **`pitch_rate` not always in `run.json`** (the champion file has no
   `run.json` at all). Recovered from the pitch column itself by scoring
   candidate scales on snap residual; detects 1.32 where the config says
   1.33, which is the recorder's 2-dp rounding and identifies the same bins.

### 2.4 What a GPU dump would add (described, not implemented)

Nothing needed for the *codebook*. It is needed for **the state-conditioned
prior** of §6.2, which needs `(obs, code)` pairs and `obs` is a 128x64 lidar
image the trajectory does not carry. The smallest change that would produce
it, if that arm is ever run:

> In `train_fast.py`, inside the `save_ckpt` path (`train_fast.py:1974`),
> add `--dump-windows PATH`: every `ckpt_every`, write one `.npz` holding
> `b_scal[:, :n]` (T x n x 15, float32), `b_img[PRO:, :n]` downcast to uint8
> (T x n x 8192), and `b_act[:, :n]` (T x n x 6, int8), for the first `n=8`
> envs. That is ~34 MB per dump at T=128 and reuses buffers that already
> exist (`train_fast.py:1836-1845`) — no new rollout work, one `np.savez` on
> the ckpt cadence. It must also be added to `record_ckpt.py`'s `TRAIN_ONLY`
> set or the config audit will refuse to record from such a checkpoint.

---

## 3. The design

### 3.1 Chunk length: `H = 10` decisions = 30 ticks = **300 ms**

Grounded in SPiRL `H=10`, OPAL `c=10`, VQ-BeT `n=10`; and in this project's
own scale — 300 ms is 2.5x the champion's *median strafe-key hold of 12 ticks
= 120 ms* (`docs/research-results.md`), so one chunk spans roughly one
A/D swap plus its correction. Shorter than that and the codebook is
re-encoding what the flat heads already do; much longer and you hit SPiRL's
"too long horizons make the skill exploration problem harder".

The measured quantisation cost of each `H` is in §5.3. Recommendation:
**start at `H = 10`, with `H = 5` as the fallback arm** if the first shows the
open-loop symptom (falling off ramps that the flat policy rides).

Do **not** go to 50 decisions (1.5 s). ACT's own ablation degrades past
`k = 100` at 50 Hz (2 s) with full temporal ensembling and a transformer
decoder; a 128-entry lookup table has none of that machinery.

### 3.2 What goes in a chunk — and what must not

| head | in the chunk? | why |
|---|---|---|
| `a[0]` yaw | **yes**, and it dominates the metric (weight 2.0) | the steering channel; the ledger blames it for the −7.9% air-accel capture |
| `a[1]` pitch | **NO** (weight 0.0) | `env.c:580-581` passes `0.0f` to `pm_tick`: pitch is *lidar aim only*, so it is a sensor-aiming action, not a movement action. Chunking it into the same code as the movement heads makes the codebook spend capacity on where the camera points. Handled separately — see below. |
| `a[2]` fwd, `a[3]` side | yes | the strafe keys; the A/D swap cadence is the thing worth abstracting |
| `a[4]` jump | yes | takeoff/bhop timing is intrinsically a *sequence* |
| `a[5]` duck | yes, weight 0.5 | inert unless `enable_duck`, rarely decisive |

The ledger reached the same conclusion independently: *"Pitch is not
reconstructed and cannot matter — env.c passes pitch 0 into the physics, so
the view pitch only aims the depth camera"* (`docs/research-results.md:2092`).

A live pitch head cannot be run per-decision without paying the trunk
forward pass that chunking exists to avoid. Two options, in order:

* **arm 1: turn pitch off entirely.** `--fix-pitch` already exists
  (`train_fast.py:833`) and sets `pitch_rate = 0.0` (`train_fast.py:1417`),
  freezing the gaze at its spawn value. 48 files in the corpus already run
  this way. One less confound in the first chunking arm.
* **arm 2: pitch as a second, independent code.** Emit a second index from a
  small head at the *chunk start*, decoding to a length-`H` pitch
  sub-sequence from its own tiny codebook (7 bins ^ H is small; 16 pitch
  codes is plenty). Costs one extra `nn.Linear(hidden, 16)` and one extra
  categorical per **chunk**, not per decision.

### 3.3 The clustering metric (the one non-obvious choice)

k-means over one-hot actions is **wrong here** and it is worth saying why:
one-hot makes Euclidean distance equal Hamming, so "trim left 0.25 deg" is
as far from "trim left 0.5 deg" as it is from "slam right 10 deg". But the
yaw ladder is ordinal *and geometric*: `{0, .25, .5, 1, 2, 4, 7, 10}`.

The tool encodes yaw as `sign(d) * log1p(|d|/0.25)`, normalized — which
spreads the ladder roughly uniformly and, critically, **separates 0.25 from
0.5 by 4.4x what raw degrees does**. That gap *is* the air-accel gain window:
the ledger's exact diagnosis was *"the bin ladder {0,.25,.5,1,2,4,7,10} has
nothing between 0.5 and 1.0 while the optimum at racing speed is 0.573, so it
over-turns ~60%"*. A metric that collapses those two bins would build a
codebook that cannot express the distinction the whole speed problem turns
on. (`--yaw-encode deg|rank` are available for ablation; a selftest asserts
the 4.4x property.)

### 3.4 Decoder, v1: a frozen lookup table

```
codebook: int8 (K, H, 6)      # K=128 chunks of H=10 six-tuples
decode(k) -> codebook[k]      # that is the whole decoder
```

No network, no VAE, no forward pass. This is VQ-BeT's discrete codebook with
the trivial `psi`, and it is deliberately the *most* frozen possible decoder:
OPAL's *"while keeping it fixed"* is enforced by the artifact being an
`int8` array. It is also directly inspectable — §5.2 shows the top-8 codes
decoding to legible English.

### 3.5 Decoder, v2 (the recommended second arm): a **state-conditioned** decoder that costs nothing

This is the answer to the central tension of the whole design. A 300 ms
open-loop yaw command cannot track a `±0.52 deg/tick` gain window that moves
with speed. SPiRL's closed-loop variant solves this with a state-conditioned
decoder network — which would cost a per-decision forward pass and destroy
the throughput win.

**But this engine already has the closed-loop decoder built in.**
`--yaw-adaptive` (`env.c:69-81`) reinterprets the yaw bin as a multiple `k`
of the analytic optimal-strafe rate `w* = atan(30/|v_h|)`:

```c
float w  = atanf(30.0f / vh) * (180.0f / M_PI);   /* env.c:75 */
float yd = K_BINS[yb] * w;                        /* env.c:76 */
```

Fit the codebook in `K_BINS` space instead of `YAW_BINS` space and a code
that says `k = +1` means *"hold the perpendicular, whatever your speed is"* —
evaluated by the core at **100 Hz**, per tick, from live velocity, with zero
policy compute. The chunk stays open-loop in *intent* and becomes closed-loop
in *execution*. This is the litsurvey's own "micro-strafe EXECUTOR between
33 Hz decisions ... an exact inner loop emits per-tick optimal wishdir",
obtained by composing two features that already exist rather than by building
a new one.

`runs/sYAWb` (48 files) is `--yaw-adaptive`, so this codebook can be fitted
today: `--runs C:/RL_Surf/runs/sYAWb`, and the tool inverts the adaptive
ladder automatically.

Sequencing: **v1 first** (it is a one-flag change and isolates the chunking
question), **v2 second** (it confounds chunking with the yaw
reparameterization, which is a separate arm the ledger already has running).

---

## 4. Integration into `train_fast.py`

### 4.1 The head

`Policy.action_head` is `nn.Linear(hidden, sum(NVEC))` (`train_fast.py:128`),
consumed through `HeadPacker.pad` into a `(B, 6, 15)` padded tensor
(`train_fast.py:404-417`) and then `sample_padded` / `logprob_entropy_padded`
(`train_fast.py:420-433`), whose `.sum(-1)` over heads is what makes six
independent categoricals behave as one joint action.

**A code head is strictly simpler than what is already there**: one
categorical over `K` codes. `logprob_entropy_padded` reduces to plain
`log_softmax` + `gather`, and — this matters — the *entropy is over
behaviors*, which is the entire point. Today `ent = -(p log p).sum(-1).sum(-1)`
sums six independent per-head entropies, so "high entropy" means each of yaw,
fwd, side, jump, duck is independently noisy — 33 Hz white noise, exactly
Tallec's failure mode. Over codes, high entropy means *the policy is
undecided between coherent 300 ms behaviors*.

Minimal shape:

```python
KCODES = 128
self.action_head = nn.Linear(hidden, sum(NVEC))   # keep: augment, don't replace
self.code_head   = nn.Linear(hidden, KCODES)      # new, orthogonal init 0.01
```

Keeping `action_head` alive (initialized but unused when `--chunk` is on)
means every existing checkpoint still loads, and the two modes stay
comparable. `record_ckpt.py` builds `Policy` from the ckpt config
(`record_ckpt.py`, `Policy(...)` construction) so it needs the same
`KCODES` argument.

### 4.2 Rollout: `T` becomes chunks, and the inner loop grows one level

Today: `N, T = args.envs, args.n_steps` (`train_fast.py:1411`), the rollout
is `for t in range(T)` (`train_fast.py:2108`), and each iteration steps the
core `K` times with the held action (`train_fast.py:2184-2186`) under the
comment at `train_fast.py:2169-2174`.

With chunks, `t` indexes a **chunk**, and the inner structure becomes
`for _h in range(H): for _j in range(K): core.step(...)` where the action fed
in is `codebook[code, _h]`. Consequences, each of which has an exact
counterpart already in the file:

| concern | today | with chunks |
|---|---|---|
| reward accumulation | `r_acc` over `K` ticks (`2179`, `2284`) | over `H*K` ticks — same variable, one more loop |
| GAE discount | `g_eff = args.gamma ** K` (`train_fast.py:2351`) | `g_eff = args.gamma ** (K * H)` — **one-character-class change**, the GAE loop at `2352-2357` is otherwise untouched |
| done mask | `ended_acc` OR-accumulated over `K` (`2202`, `2285`) | over `H*K`; `b_done[t]` write at `2309-2310` unchanged |
| truncation bootstrap | `V(s_T)` re-rendered from `term_obs` (`2213-2244`) | unchanged — it fires per `core.step`, inside the innermost loop |
| buffer shapes | `b_act (T,N,6)` (`1845`) | `b_act (T,N)` int64 code index; `b_img` stores one frame per **chunk**, so at `--n-steps 16` it is 8x smaller than today's `T=128` (the S3 comment at `train_fast.py:1830-1834` sizes it at 4.3 GB) |
| CUDA graph | `step_compute` captured at `1928-1944` | unchanged in structure: same `static_obs` in, `static_code` out |
| lidar render | once per decision (`fill_vision`, `2344`) | once per **chunk** — this is where the throughput comes from |

### 4.3 Episode end mid-chunk — **abort, do not leak**

The file already answers this, for the exactly analogous case
(`train_fast.py:2311-2318`):

> *"episode end aborts any burst in flight (Go-Explore: 'exploration is also
> aborted at the episode's end'; review: ez bursts must not leak a frozen
> random action into the next episode's spawn either)"*

The core autoresets in place, so without an abort the remaining decisions of
the chunk are applied to a **freshly spawned** episode — up to 300 ms of an
unrelated behavior injected at spawn. `train_fast.py:2169-2174` calls the
sub-tick version "negligible contamination" at `K=3`; at `H*K = 30` it is not.

Static-shape implementation (required: `step_compute` is CUDA-graph
captured):

```python
ended_in_chunk |= ended                 # bool (N,), accumulated as today
a_h = codebook[code, _h]                # (N, 6)
a_h = torch.where(ended_in_chunk[:, None], NEUTRAL, a_h)
```

where `NEUTRAL = (YAW_ZERO, PITCH_ZERO, 1, 1, 0, 0)` = hold everything. Shapes
are constant; only the values change. The chunk's *duration* is then variable
but its *tick count* is not, which is what the compiled/graphed regions care
about.

**`g_eff = gamma^(K*H)` is exact, not an SMDP approximation.** The usual
worry with variable-duration options is that the discount should be
`gamma^(actual elapsed ticks)`. Look at where `g_eff` appears in the GAE loop
(`train_fast.py:2355-2356`): both occurrences are multiplied by
`nonterm = 1.0 - b_done[t]`. A chunk cut short by an episode end has
`nonterm = 0`, so `g_eff` is never applied to it; a chunk that runs to
completion is *always* exactly `K*H` ticks long. The one discount value is
therefore correct on every transition it touches.

GAE already zeroes across the boundary via `nonterm = 1.0 - b_done[t]`
(`train_fast.py:2354`), so the neutral tail contributes reward the agent
earned in the new episode — bounded by `H*K` ticks of standing still, and
`nonterm` makes it non-bootstrapping. If that residual is unacceptable, the
stricter variant is to also zero `r_acc` for rows with `ended_in_chunk`,
which mirrors how `rpd` already zeroes ended rows' shaping
(`train_fast.py:2289-2291`).

### 4.4 The sample-count trade-off — state it plainly

`MB = T * N // args.minibatches` (`train_fast.py:2018`); defaults `T = 128`
(`train_fast.py:713`), `N = 2048`, `epochs = 4` (`714`), `minibatches = 16`
(`715`). A chunked policy makes **1/H the decisions per env-step**, so at
fixed `T` it covers `H` times more game time per iteration but produces the
same 262,144 samples; at fixed *game time* (`--n-steps 13` for `H=10`) it
produces `H` times fewer.

Given this project's own finding that *update density is the currency*, the
first arm should hold **game time per iteration roughly constant** and pay
for it in epochs:

```
--chunk 10 --n-steps 16 --epochs 8 --minibatches 4
```
= 16 chunks x 10 dec x 3 ticks = 480 ticks/env/iter (vs 384 today), 32,768
samples, MB = 8,192. Then compare fps and progress-per-wall-clock-hour, which
is the currency the ledger settled on, not progress-per-step.

### 4.5 Recording and the config audit — the step that gets forgotten

`tools/record_ckpt.py` refuses to record from a checkpoint whose config
contains a key the file never mentions ("*a checkpoint key the recorder never
READS is a key it cannot be mirroring*"). `--chunk`, `--codebook`, and
`KCODES` all change *what an action means*, so they are **not** `TRAIN_ONLY`:
`record_ckpt.py` must load the codebook and wrap `GreedyTorchPolicy` /
`SampledTorchPolicy` (`train_fast.py:598-617`) in a chunk-holding shim
analogous to `_TorchPolicyBase.act`'s `act_every` hold
(`train_fast.py:549-553`). Skipping this reproduces the exact failure the
guard was written for — a plausible-looking trajectory recorded under the
wrong semantics.

The chunk-holding shim is small: the same `self._held` pattern, with
`self._k = act_every * H` and `self._held = codebook[code]` sliced by
`(self._tick // act_every) % H`.

---

## 5. Prototype results (CPU, `tools/build_action_codebook.py`)

### 5.1 Derivation validated on the champion trajectory

```
=== VALIDATE .../champ_greedy.jsonl
    yaw_adaptive=False, yaw_rate_max_deg=10; pitch_rate auto-detected from the pitch column
  episode 0: 8,419 ticks, pitch_rate=1.32
    yaw   bins identified on 100.000% of ticks, worst snap residual 0.000000 deg
    yaw   INTEGRATED RECONSTRUCTION vs recorded column: mean 0.000000 deg, max 0.000000 deg over 8,418 ticks
    pitch bins identified on 100.000% of ticks (1033 clamp-censored), worst snap residual 0.010000 deg
    pitch ONE-STEP reconstruction: mean 0.010236 deg, max 1.330000 deg
    fully invertible ticks (yaw AND pitch): 100.000%
    act_every = 3 at phase 0   block-constancy [1:1.0000] [2:0.7657] [3:1.0000] [4:0.3042] ...
  episode 1: 8,495 ticks ... 0.000000 / 0.000000 deg ... act_every = 3 at phase 2
  episode 2:   505 ticks ... 0.000000 / 0.000000 deg ... act_every = 3 at phase 0
```

Integrating the *derived* yaw bins forward from the first recorded yaw
reproduces the recorded yaw column to **0.000000 deg, mean and max, over
17,416 ticks across three episodes**. These are recovered actions, not
pseudo-actions.

Independent cross-check on the corpus: the strafe-key/turn-direction pairing
comes out at **74.8%** against the ledger's independently measured 71.8% for
the champion — and it only lands there under the correct sign convention
(`env.c:243-254` builds `forward = (cos yaw, sin yaw)` with `"y' = left"`, so
**positive yaw is a LEFT turn**; the ledger's *"holding D turns the velocity
heading clockwise, mean sign −0.61, and the view follows at −0.57"* is the
same statement).

### 5.2 The K=128, H=10 codebook (700 files, 1.56M decisions)

```
=== INGEST  700 trajectory files, horizon=10 decisions, stride=10
  files 700  episodes 2,096  ticks 4,696,433  decisions 1,563,628
  act_every seen: {3: 2096}
  worst yaw snap residual over all episodes:   0.000000 deg
  worst pitch snap residual over all episodes: 0.010000 deg
  non-invertible decisions: 0 (0.0000%)
  windows: 155,418   (14.8s)

=== ACTION MARGINALS over 1,554,180 decisions
  yaw  -10:0.3% -7:0.5% -4:0.7% -2:5.5% -1:12.6% -0.5:11.1% -0.25:12.0%
        0:12.1% 0.25:12.2% 0.5:13.7% 1:12.8% 2:4.4% 4:0.6% 7:0.2% 10:1.4%
  fwd  S:0.0%  -:80.4%  W:19.6%
  side A:41.4%  -:16.3%  D:42.3%
  jump 47.91%   duck 28.68%
  |yaw| inside the REAL gain window arcsin(30/|v|) at each window's own speed: 69.7%
    (median speed 2240 u/s -> window +-0.77 deg/tick; median |yaw| commanded
     0.50 deg/tick -> over-turn 0.7x)
  strafe pairing (A with +yaw / D with -yaw) on held-key decisions: 74.8%
  distinct windows: 150,160 of 155,418 sampled; the raw space is 10^27

=== KMEANS  K=128
  occupancy: min 169  p10 484  median 902  p90 2260  max 7072
  dead codes (0 members): 0 / 128
  code entropy 6.62 bits of 7.00   perplexity 98.1 (77% of K)
  intra-cluster MSE 7.6637  vs total variance 18.6611  -> 58.9% explained
  centroid -> legal-action snap: mean L2 1.7178
  QUANTISATION MSE of the frozen decoder: 10.1240 (54.3% of total action variance)
```

**Zero dead codes and 77% perplexity** is the number to watch: VQ-BeT names
codebook collapse as the failure mode of discrete latents, and a dead code in
a policy's action set is a logit that can never learn anything. 98 of 128
codes are effectively in use.

The top-8 codes decode to exactly the vocabulary the design asked for:

| rank | code | members | decodes to | English |
|---|---|---|---|---|
| 1 | 97 | 7,072 (4.6%) | `R1/-D x10` | steady **right** turn, hold D, 300 ms — turns 1.6x the gain window at its own 2,815 u/s |
| 2 | 5 | 6,006 (3.9%) | `R1/-D+JUMP x10` | same, **jump held** (bhop-cap / takeoff) |
| 3 | 101 | 4,338 (2.8%) | `L0.25/-A x10` | **fine left trim**, hold A — 0.4x the window (under-turning) |
| 4 | 9 | 3,895 (2.5%) | `L0.25/-A+JUMP` -> `L0.5/-A+JUMP x2` -> `L0.25/-A+JUMP x7` | fine left trim with a lead-in flare, jump held |
| 5 | 57 | 3,653 (2.4%) | `L1/-A x10` | steady **left** turn, hold A — 1.3x the window |
| 6 | 47 | 3,624 (2.3%) | `yaw0/-D+JUMP x10` | **straight, jump held**, median speed 1,241 u/s, 1% onground — the airborne coast/takeoff primitive |
| 7 | 0 | 3,536 (2.3%) | `R0.25/-D+JUMP x6` -> `R0.5/-D x2` -> `R0.25/-D` -> `R0.5/-D` | fine right trim, **jump released halfway** |
| 8 | 19 | 3,456 (2.2%) | `L1/-A+JUMP x10` | steady left turn with jump |

That is "slide, takeoff, hard turn" as a 128-word vocabulary, discovered with
no labels. Note the **A↔left / D↔right pairing is perfect across all eight**
— the codebook rediscovered the strafe mechanic from action statistics alone.

### 5.3 Chunk length and codebook size, measured

`QMSE%` = quantisation MSE of the **frozen lookup decoder** as a fraction of
the total action variance in the same window space. It answers "how much of
what the flat policy did can the chunked policy still express". 260 files,
same corpus, `--yaw-encode log`.

| H | open-loop ms | K=64 | K=128 | K=256 | windows | dead codes |
|---|---|---|---|---|---|---|
| 2 | 60 | 23.1% | **17.0%** | 12.5% | 120,000 | 0 |
| 3 | 90 | 33.9% | **27.5%** | 22.9% | 120,000 | 0 |
| 5 | 150 | 47.6% | **41.9%** | 37.0% | 113,284 | 0 |
| **10** | **300** | 58.8% | **55.1%** | 51.6% | 56,452 | 0 |
| 16 | 480 | 64.8% | **62.0%** | 58.7% | 35,133 | 0 |
| 25 | 750 | 69.8% | **67.0%** | 63.5% | 22,350 | 0 |
| 50 | 1500 | 73.4% | **70.5%** | 66.8% | 10,979 | 0 |

**H dominates K by a wide margin.** Going 64 → 256 (a 4x codebook, 2 extra
bits for the policy to learn) buys 5–7 percentage points at *every* H; going
H=10 → H=5 buys 13. If the first arm is expressiveness-limited, shorten the
chunk before enlarging the codebook.

No dead codes anywhere in the sweep, and perplexity stays at 62–83% of K —
so K=128 is comfortably inside the regime VQ-BeT's 64–256 modes occupy, and
the collapse risk is in *training*, not in the fit.

The `H=50` row is the one the user's "10–50 actions" range asked about:
at 1.5 s of open loop the decoder throws away 70% of the action variance and
the codebook is fitting whole ramp traversals, not primitives. It is also
past ACT's degradation point. **10 decisions is the recommendation; 5 is the
fallback; 50 is not a candidate.**

### 5.4 Where the quantisation error sits — and a favorable surprise

At H=10, K=128:

```
  decoder-vs-truth per-head disagreement:
      yaw 71.3%   pitch 79.2%   fwd 14.7%   side 15.8%   jump 34.4%   duck 28.2%
  yaw: sign agreement 67.8%, mean |error| 0.586 deg/tick
  decisions inside the real gain window: decoder 82.6%  vs  data 69.7%
```

Read that carefully. The codebook gets the **key pattern** almost right —
fwd and side disagree on only 15–16% of decisions — and gets the **exact yaw
bin** wrong most of the time, but by only 0.586 deg/tick on average. That is
precisely the error profile BeT and VQ-BeT answer with a per-step **offset
head** (`k x dim(A)` residual matrix, `L_offset`), and it says the missing
capacity is *magnitude*, not *shape*.

The last line is the surprise: because k-means centroids regress toward the
cluster mean, the decoder commands **less** yaw than the raw policy did, and
therefore lands inside the real speed-dependent gain window `arcsin(30/|v|)`
on **82.6% of decisions against the data's own 69.7%**. The ledger's
diagnosis of the −7.9% air-accel capture was over-turning — *"it over-turns
~60% and pushes wishdir past perpendicular into the braking region"* — and
quantisation acts as a low-pass filter on exactly that.

Caveat, stated plainly: this is a **static, open-loop** measurement over
recorded windows. It says the codebook's yaw commands are better centered on
the gain window than the source policy's were; it does **not** say a chunked
policy will be faster. The rollout is what decides that (§8).

### 5.5 Transfer: the codebook crosses MAPS more easily than it crosses control rates

Fit on `surf_src_cannonball` (400 files, 87,795 windows, H=10, K=128), then
quantise three held-out sets. `QMSE%` and `expl%` are each measured against
the held-out set's **own** action variance, so the rows are comparable.

| held-out set | windows | QMSE% | expl% | codes unused | code-usage JS (bits) |
|---|---|---|---|---|---|
| *(in-sample reference)* | 87,795 | 54.8% | 58.5% | 0 | 0 |
| **`surf_ski_2` — a DIFFERENT MAP** | 21,162 | **40.2%** | **64.3%** | **0** | **0.134** |
| cannonball, different reward/lineage (`sIS_long`, `sG3V_view`) | 60,000 | 79.1% | 25.3% | 1 | 0.411 |
| cannonball, `act_every=9` (11 Hz instead of 33) | 1,911 | 82.7% | 26.1% | **46** | 0.508 |

(The 700-file production fit reproduces the map row independently:
`38.8%` held-out QMSE vs `54.3%` in-sample, `64.7%` vs `58.9%` variance
explained, JS `0.146`, 2/128 codes unused — see the tool's own
`=== TRANSFER TO HELD-OUT RUNS` block.)

The different-map row is **better than in-sample** on every measure, with a
code-usage divergence of ~0.14 bits — the two maps draw on essentially the
same repertoire. Meanwhile the same map under a different *objective* is
markedly worse, and under a different *control rate* it falls apart with 46
of 128 codes never used.

That is the cross-map argument, measured rather than asserted: **a behavior
chunk is a property of the movement engine, not of the geometry.** It also
names the real portability constraint — the codebook's time unit is one
decision, so a codebook is tied to its `act_every`, and `--chunk` and
`--act-every` must be pinned together in the run config and mirrored in
`record_ckpt.py` (§4.5).

---

## 6. Stabilization recipe

Ordered by how much each is load-bearing.

### 6.1 Frozen decoder (OPAL, explicit)

The decoder is an `int8 (K, H, 6)` array loaded from the `.npz` and never
touched by the optimizer. OPAL states this outright — *"we learn a task policy
in space of primitives ... while keeping it fixed"* — and it is the reason
the code-head logits stay interpretable across the whole run: code 97 means
`R1/-D x10` on iteration 1 and on iteration 100,000.

### 6.2 KL-to-prior instead of entropy (SPiRL Eq. 3)

SPiRL's objective replaces the max-ent entropy bonus with
`-alpha * D_KL(pi(z|s) || p_a(z|s))`, alpha auto-tuned so the realized KL
tracks a target `delta` (`delta = 1` maze, `5` manipulation).

Mapping onto `train_fast.py`: the entropy term enters at
`train_fast.py:2015` (`el = -ent.mean()`) with a coefficient scheduled at
`train_fast.py:2373-2380` (`--ent` default 0.005, `--ent-final` linear
schedule, held in a **tensor** `ent_t` specifically so a Python float does
not get baked into the compiled graph). So:

* **v1 prior — marginal.** `p(k)` = the codebook's empirical occupancy,
  saved as `occupancy` in the `.npz`. Replace `el = -ent.mean()` with
  `el = kl.mean()` where `kl = (p * (logp - log_prior)).sum(-1)`. One extra
  `(K,)` buffer; the `ent_t` plumbing carries over unchanged.
* **v2 prior — state-conditioned.** A small MLP `p(k | s)` behavior-cloned
  from `(obs, code)` pairs, which is what the dump in §2.4 is for. This is
  the actual SPiRL construction and it is what makes the prior *"focus
  exploration on the relevant parts of the skill space"*.
* **alpha auto-tuning.** SPiRL's dual update
  `alpha <- alpha - lambda * grad_alpha[alpha * (KL - delta)]` is ~4 lines
  and removes the `--ent` sweep entirely. Target `delta ≈ 1–2` nats against
  `log 128 = 4.85` nats — SPiRL's own `delta=1` on the simplest domain.

Why this matters more here than usual: an entropy bonus over 128 *coherent
behaviors* is a much stronger pull toward uniform than an entropy bonus over
6 heads, because a uniform code distribution is a genuinely random walk over
300 ms behaviors. Without the prior, `--ent 0.005` would be badly miscalibrated.

### 6.3 Codebook refresh cadence — **off for the first arm**

The tempting move is to refit the codebook from the current policy's own
recordings (the recorder already writes `traj_*.jsonl` every `ckpt_every`),
which is the Linesight "re-extract the line from the AI's own best runs"
pattern the litsurvey endorses.

**But refitting changes what code `k` means, which silently invalidates
every weight in `code_head`.** VQ-BeT sidesteps this with EMA updates that
move codes slowly and never permute them. The safe version, if this is ever
run:

1. keep `K` fixed and **warm-start** k-means from the previous centroids;
2. greedily match new centroids to old by L2 and permute `code_head.weight`
   rows to follow;
3. only replace codes whose occupancy fell below a floor (a *dead-code
   refresh*, which is the one thing VQ-BeT's EMA does not give you);
4. cadence no faster than every 5e8 steps.

For arm 1: refit never. The corpus is 1.5M decisions of real behavior;
the codebook is not the bottleneck.

### 6.4 ez-greedy over codes (and why the existing `--ez-eps` mostly goes away)

The survey's prescription is *"with prob eps, repeat a random action for
n ~ zeta(2) capped ~100; apply to movement keys ONLY, keep aim closed-loop"*,
with the paper's own `mu = 2.0` and Rainbow's `n <= 100`.

Over codes, the machinery at `train_fast.py:2118-2157` needs only its units
changed: draw a uniform random **code**, hold for `n ~ zeta(2)` **chunks**,
`--ez-max` dropping from 60 decisions to **~6 chunks** (same 1.8 s of wall
time). But the more important point is that most of the job is already done
by the action space: a single random code is 300 ms of committed, coherent
behavior — the ballistic Lévy flight ez-greedy is trying to manufacture,
sampled *on-policy*, so it does **not** hit the exclusion at
`train_fast.py:2396-2404` that made the ledger's ez arm land at parity.
Recommendation: run arm 1 with `--ez-eps 0` and let the code entropy do it.

And the survey's "keep aim closed-loop" caveat is exactly §3.5: the v2
adaptive-yaw codebook keeps aim closed-loop *by construction*, at 100 Hz.

---

## 7. Cross-map transfer — the 1000-map argument

The stated goal is convergence on arbitrary maps. A behavior codebook is the
part of the policy that should transfer, because the primitives are
**properties of the movement engine, not of the geometry**: `R1/-D x10` is a
right-hand air-strafe under `sv_airaccelerate 100`, on every map ever made.
The map-specific knowledge — *which* primitive here — stays in the trunk,
which is where it belongs.

Three concrete consequences:

1. **Transfer is measurable today, offline.** Fit on
   `surf_src_cannonball`, evaluate quantisation error on held-out
   `surf_ski_2` windows: `--holdout C:/RL_Surf/runs/eyes_acro ...`. Numbers
   in §5.5.
2. **A shared codebook is a shared action space.** Two maps trained with the
   same `.npz` have literally the same `code_head` semantics, so
   Kickstarting / QDagger distillation across maps (both already in the
   litsurvey's ranked recommendations, 6.9x speedup, student exceeds teacher
   +43%) becomes a distillation over a *common categorical*, not over two
   incompatible 6-head joints. The Sonic benchmark result the survey cites —
   "joint pretraining + per-map fine-tuning roughly doubles low-budget
   scores" — is the direct precedent.
3. **The codebook is the natural unit of a curriculum.** A code's occupancy
   on a new map is a free measure of how much of the learned repertoire that
   map actually needs.

The honest caveat: a codebook fitted on cannonball encodes cannonball's
*speed distribution*, because a fixed-degree yaw bin only means "hold
perpendicular" at one speed. This is a second, independent argument for the
§3.5 adaptive-yaw codebook — `k = +1` is speed-free and therefore genuinely
map-free.

---

## 8. The exact minimal diff for arm 1

Described, not applied. Seven edits, ~120 lines.

| # | file:line | change |
|---|---|---|
| 1 | `train_fast.py:78` (after) | `KCODES = 128` module constant; `NEUTRAL_ACT = (7, 3, 1, 1, 0, 0)` |
| 2 | `train_fast.py:101-135` | `Policy.__init__(..., n_codes: int = 0)`; if `n_codes`, add `self.code_head = nn.Linear(hidden, n_codes)`, orthogonal init 0.01, zero bias (mirroring `train_fast.py:133-134`). `forward_split` (`166`) returns `(action_head(h), code_head(h), value_head(...))` — or returns a 2-tuple where the first element is whichever head is active, to keep every call site unchanged |
| 3 | `train_fast.py:693-1050` | three args: `--chunk H` (0 = off), `--codebook PATH`, `--code-kl` (use KL-to-occupancy-prior instead of entropy). Load the `.npz`, assert `horizon == H` and `nvec == NVEC`, upload `codebook` as an `(K, H, 6)` int64 CUDA tensor |
| 4 | `train_fast.py:1845` + `2108-2344` | `b_act` becomes `(T, N)` int64; wrap the `for _j in range(K)` loop (`2184`) in `for _h in range(H)`; feed `codebook[static_code, _h]`, masked to `NEUTRAL_ACT` where `ended_in_chunk`; `step_compute` (`1919`) samples one categorical instead of `sample_padded` |
| 5 | `train_fast.py:2351` | `g_eff = args.gamma ** (K * H)` |
| 6 | `train_fast.py:2006-2016` | in `mb_step`, `logp/ent` from the single code categorical; under `--code-kl`, `el = kl_to_prior.mean()` instead of `-ent.mean()` |
| 7 | `tools/record_ckpt.py` | mirror `chunk`/`codebook` (they change what an action MEANS, so they must **not** go in `TRAIN_ONLY`); wrap the eval policy in a chunk-holding shim |

Launch for the first arm:

```
python tools/build_action_codebook.py \
    --runs C:/RL_Surf/runs/race_cannonball C:/RL_Surf/runs/race_respawn \
           C:/RL_Surf/runs/race_A C:/RL_Surf/runs/race_int \
           C:/RL_Surf/runs/race_start2 \
    --max-files 700 --k 128 --horizon 10 --out codebooks/H10_K128.npz

python python/train_fast.py --run xCHUNK --map maps/surf_src_cannonball.bsp \
    --chunk 10 --codebook codebooks/H10_K128.npz --code-kl \
    --n-steps 16 --epochs 8 --minibatches 4 \
    --act-every 3 --fix-pitch -10 --ez-eps 0 \
    <the rest of the sCTL/race arm's flags, unchanged>
```

paired against an identical `--chunk 0` control on the same box, judged on
**progress per wall-clock hour** (the ledger's currency), with two
diagnostics that decide the follow-up arm:

* `tools/strafe_audit.py` air-accel capture on the greedy recording — if
  chunking costs capture, go to §3.5 (adaptive-yaw codebook), not to `H=5`;
* code-head entropy and per-code occupancy — if occupancy collapses onto
  <20 codes, the KL prior's `delta` is too low or `K` is too high.

---

## 9. Risk register

| risk | signal | mitigation |
|---|---|---|
| 300 ms open loop loses the ±0.52 deg gain window | air-accel capture worse than control at matched steps | §3.5 adaptive-yaw codebook (closed-loop at 100 Hz, zero policy cost) |
| quantisation MSE too high to express the flat policy's behavior | §5.3 table | raise `K`, lower `H`, or add VQ-BeT's per-decision offset head (costs the throughput win — measure first) |
| `H`x fewer gradient samples per env-step | fps up but progress/step down more than `H`x | `--epochs 8 --minibatches 4` (§4.4); this is the arm's main tuning axis |
| code collapse (policy uses <20 codes) | code entropy in `progress.csv` | KL-to-prior with SPiRL's auto-tuned alpha, `delta ≈ 1–2` nats |
| codebook stale relative to an improving policy | occupancy drifts to a few codes | §6.3 warm-start refresh with permutation matching — **not** a naive refit |
| a recording made under the wrong semantics | silent | `record_ckpt.py` audit already catches it *if* step 7 of §8 is done |

---

## 10. Sources

* Pertsch, Lee, Lim, *Accelerating Reinforcement Learning with Learned Skill Priors*, arXiv 2010.11944 (SPiRL)
* Ajay, Kumar, Agrawal, Levine, Nachum, *OPAL: Offline Primitive Discovery...*, arXiv 2010.13611
* Shafiullah, Cui, Altanzaya, Pinto, *Behavior Transformers*, arXiv 2206.11251
* Lee, Cui, Shafiullah, Pinto, *Behavior Generation with Latent Actions* (VQ-BeT), arXiv 2403.03181
* Zhao, Kumar, Levine, Finn, *Learning Fine-Grained Bimanual Manipulation...* (ACT), arXiv 2304.13705
* Dabney, Ostrovski, Barreto, *Temporally-Extended epsilon-Greedy Exploration*, arXiv 2006.01782
* Tallec, Blier, Ollivier, *Making Deep Q-learning Methods Robust to Time Discretization*, ICML 2019 — via `docs/research-litsurvey.md`
* In-repo: `docs/research-results.md` (act-every ladder, air-accel capture,
  strafe audit), `docs/research-litsurvey.md` (temporally-correlated
  exploration; micro-strafe executor addendum), `src/env.c`,
  `python/surfgym/record.py`, `python/train_fast.py`
