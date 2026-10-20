# Lineage notes - surf_src_sidistic

Append-only, newest at the bottom, a dated heading per entry (time from `date`). This file is
the part of the workspace that carries what a lineage KNOWS from one round to the next. If your
branch wins, the next four forks read what you leave here, so write facts with numbers.

## Round 0 - inherited state (setup session, 2026-10-20)

### The map

* `maps/surf_src_sidistic.bsp`, 7 stages. The WR is 115.99 s by a human; the demo is analysis
  only and NEVER training (CLAUDE.md section 0).
* The goal field at the spawn is d0 = 97,768 u. 1% of progress = 978 u.
* The 2026-08-23 "unreachable" verdict was a voxel artifact (thin panes with holes filled as
  their AABB). It was fixed on 2026-10-18 (`vision.pane_tag`) and the caches were re-baked.
  The finish box is `zones.json` end, z 176..184.
* Checkpoint curtains sgcp_1..7 are in the bsp. `autoresearch/score.py` reports the furthest
  one entered (CP).

### The gates seen so far

Greedy eval from the true start, best episode, % of d0:

1. **5.6-5.7%: the first-ramp side branch.**
   * Every run so far parks here for 200M-1.1B steps.
   * The policy rides a side branch off the first ramp. The way on is a turn onto the ramp
     itself ("it just has to turn left", the user).
   * Exits, all coinciding with the unstuck temperature T rising past ~1.3-2: 529M
     (compass + pot-rate), 755M (novel respawn), 1,133M (surf flag + bonus), 1,510M (cap-2
     base).
2. **20.8%: the wall hole ("the window").**
   * A ~110x130 u hole in a wall at x ~1,150, z 13,198..13,314, 350-430 u above the agent's
     line, with a fail plate on its sill.
   * The human climbs to it by cutting back into an A-frame ramp before it ends.
   * Passed (CP1, 22.9%) at 755M by compass + pot-rate, 1,359M by surf, 1,661M by cap-2.
     The novel-respawn run stalled there from ~1.0B to 1.41B.
3. **27.4-27.5%: the CP2 wall. Never crossed.**
   * The agent dives (|v| ~2,000, vz ~-1,370) into a fail at x ~3,050, z ~11,200 at ~25.4 s.
     It surfs only 3-8% of its last 3 s.
   * The human passes the same spot ~250 u higher and level (vz -141) at 27.4 s, after turning
     EAST onto a ramp at t 24.6-26.8 s.
   * That turn is a 1,958 u dip in the field (2.0 reward units at 100/d0): the field charges
     for the detour and pays for the dive.
   * The surf run spent 1.5B steps at T = 2 here without one crossing.
   * `--obs-pot-rate` reads -1.7..-1.9 during the human's turn and +1.5..+1.7 along the
     agent's dive.
* Along the human line the field has 7 such dips of 1.4-3.0 reward units, so expect more
  walls of this kind past CP2.

### Runs on this map

All local RTX 5090, 1024 envs, 400-470k fps, from scratch, one seed each.

| run | recipe | trap exit | window | best | stopped |
|---|---|---|---|---|---|
| sidEM2cap2 | base: --unstuck --unstuck-reach eval --respawn-margin 2 --unstuck-max 2 | 1,510M | 1,661M | 27.4% | 2.9B |
| sidEM2cap2surf | base + --obs-surf 1 --surf-bonus 0.3 | 1,133M | 1,359M | 27.5% (CP2 wall from ~1.8B to 4.0B) | 4.0B, done |
| sidEM2cap2surfGR | surf + --obs-compass 1 --obs-pot-rate 1 | **529M** | **755M** | 27.2% at 755M | 758M, by the user |
| sidEM2cap2surfNV2 | surf + --respawn-novel | 755M | not by 1.41B (stuck at 20.8%) | 20.9% | 1.41B, session ended |

Recipe flags are listed as "base" or "surf" plus the extras shown.

* Every one of them also carries the scratch preset: --view-continuous --view-absolute velocity,
  --keys-hold, --obs-potential norm + curtain, --obs-normal 1, --simba 1, --ep-ticks 30000
  (300 s), --respawn-frac 0.9, --int-coef 0.25 --int-view 8 --int-speed 3, --act-every 4,
  gamma 0.9995, --n-steps 128.
* T (the unstuck temperature) is the run-state that matters most for the trap exits. It rises
  0.5 per 1e8 steps after 2e8 steps without a 500 u eval-frontier gain, and is capped at 2.
* The compass + pot-rate lead is ONE seed. The early-mark agreement between identical configs
  was measured at ~0.4% (CLAUDE.md), so a 600M-step lead at two marks is large, but it is
  still one draw.

### The incumbent handed to round 1

* Recipe: `config.incumbent_recipe` (sidEM2cap2surfGR's flags). `box.py launch` prepends it.
* Checkpoint: `$AR_HOME/incumbent/ckpt.pt`, sidEM2cap2surfGR at 749.7M steps. That is 6M
  steps before its window-passing eval (755M: 27.2%, CP1, 7/9 through the hole). Resume it
  with `box.py deploy --ckpt` + `box.py launch --resume`.
* Matched-step references:
  * `$AR_HOME/incumbent/r0/progress.csv` (GR, to 758M);
  * `$AR_HOME/incumbent/ref_surf/progress.csv` (the surf run, to 4.0B, with the CP2 wall).
* Score to beat: 27.5 (the best eval anyone has, surf at 3,775M). GR's best is 27.22.

### Mechanisms already in the trainer

The docs say how each one works and what it did.

* `--unstuck` (docs/unstuck.md);
* `--respawn-frontier` / `--respawn-backward` / `--respawn-random`
  (docs/respawn_frontier.md, respawn_backward.md, respawn_random.md);
* `--respawn-novel` (docs/respawn_novel.md);
* `--obs-compass` / `--obs-pot-rate` (docs/obs_potential.md);
* `--obs-surf` / `--surf-bonus`;
* `--race-ratchet` (docs/race_ratchet.md);
* `--goal-planner` and the ramp search (docs/planner-design.md, docs/ramp-search-design.md,
  paused by the user on 10-18);
* the archive / go-explore family (docs/edge_archive.md).

`docs/research-results.md` is the full ledger; search it before re-running an idea.

### Throughput off the 5090

The 3090 is the reference card in CLAUDE.md (~0.75-0.9e9 steps/h on the older, smaller
config). With SimBa + normals + the potential channel at 1024 envs, expect roughly 150-250k
steps/s. Measure it on your box in the first minute and size `--steps` to the time.

## 2026-10-20 01:07 UTC - r1b2 (round 1): what the CP2 death IS, and --noise-tau

### The CP2 wall is a deliberate air U-turn (ANALYSIS of our own evals, no demo used)

Read from `ref_surf/traj_3775397888.jsonl` (9/9 episodes die at 25.2-25.7 s, x 3,007-3,113,
z 11,17x-11,22x) with a per-tick dump (scratchpad tool `uturn.py`; the row format is
`record.py`'s: [t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward, pitch, fwd, side]):

* 22.0-23.0 s: the agent flies EAST (heading +8 to +60 deg) at ~1,450-1,800 u/s, rising.
* **23.0-24.5 s: it holds D (side bin "D") on every tick with the view at +1.2 deg off the
  velocity (= atan(30/|v|), the per-frame strafe optimum under --yaw-adaptive): a maximal
  right air-strafe turn, heading +60 -> -178 deg (~240 deg in 1.5 s), vz +431 -> -769.**
* 24.5-25.3 s: west, falling, into the fail at x ~3,050. The field d falls the whole way
  (72,890 at 24.0 s -> 70,947 at 25.2 s): the U-turn is exactly what the field gradient asks for.
* The human (round 0 notes) instead keeps going east to a ramp at x ~5,500 and turns ON the ramp.
  So the missing behaviour is "do NOT turn for ~1 s" = ~25 consecutive off-mode key decisions.
  Under i.i.d. sampling with P(D) ~ 0.9 that is ~0.1^25 = 1e-25 per attempt; temperature cannot
  fix it (it scales per-step noise, not its duration). That is why 1.5-2.3B steps at T = 2-4 never
  sampled the turn east.
* `dip/max_survived_depth` at the CP2 wall (surf run, 1.8-4.0B) is 0.2-0.4 reward units
  (once 1.58); the detour needs ~2.0. The FIRST-trap exit coincided with max_survived_depth
  jumping to 7.27 (GR at 508M, surf at 1,119M): that exit was a big dip survived.

### --noise-tau SECONDS (new, commit 12aede7, default off, bit-identical)

Every head's rollout sampling noise becomes a stationary AR(1) process, rho = exp(-decision_s /
tau) per decision: the view heads' Gaussian eps directly, the key heads' Gumbel through
g = -log(-log Phi(e)) (Gaussian copula). Each decision's action has EXACTLY the policy's
(tempered) distribution and the stored log-prob is unchanged, so the PPO update is untouched;
only the duration of a deviation changes (~tau s instead of 40 ms). Published basis: colored
noise in PPO (Hollenstein et al., AAAI 2024), gSDE (Raffin et al., CoRL 2021). Generic: one
constant in seconds, nothing from the map. Test: `tests/python/test_noise_tau.py` (marginals =
softmax / N(mu, sigma), log-prob = the update's recomputation, persistence, 25-decision off-mode
runs at P(mode) 0.9, flag-off trainer bit-identical to 877a2a6 on a CPU sidistic smoke).
Composes with --unstuck (the temperature still scales the logits / sigma).
Refused with --chunk, --yaw-cond, --crl, --tick-ms-schedule.

### Boxes

* 00:44-00:59 nothing under the caps (3090 0.38, 4090 0.455, 5080 0.56 $/h); another branch had
  the one cheap 5080.
* 00:59 got 56815694 (RTX 3090 Ti, 0.242 $/h, machine 68144): deploy step 6/6 gpu_health
  UNHEALTHY (power limit 300 of 450 W, sm 1,380 MHz under load, 60 TFLOPS bf16). Blocked
  (`gpu_capped`) and released at 01:04 per CLAUDE.md section 1. box.py prints the tail of
  stdout+stderr, so a deploy failure shows only ssh banners: run gpu_health by hand to see it.

### 01:35 UTC - arm 1 launched: r1b2_nt05 = warm incumbent + --noise-tau 0.5 (measures AT the CP2 wall)

* Box 56818837, RTX 3090 0.218 $/h, machine 148800, `ssh -p 40783 root@71.218.106.223`.
  **CPU: dual Xeon E5-2699 v3 (Haswell 2014), 72 threads -> only ~70k steps/s** (GPU 6% busy,
  trainer 5,700% CPU, omp 32). This is the slow-CPU class CLAUDE.md warns about; it passed the
  offer filter (>= 32 threads, >= 2.5 GHz). 30 min = ~126M steps here, not 3-4e8.
* Command: `box.py launch --agent r1b2 --run r1b2_nt05 --resume --steps 4e8 -- --noise-tau 0.5`
  (incumbent ckpt 749.7M md5 ede4be9a..., stop at 1,149.7M, evals every 75M). Restored unstuck
  state: T 0.274, best 20,546u (the first eval after the resume, 750.3M, read 25,464u).
* Log line: `rho 0.9231 per 40.0 ms decision (lag-1 s correlation 0.135)`, CUDA graph captured.
* Watch: CP2 in greedy evals; training-side `dip` (max survived setback, reward units; ~0.5 at
  resume, the CP2 detour needs ~2) and `mind` (reservoir's deepest state, 74.28% of d0 at
  resume; a training episode that survives 2 s past the death point d ~70.9k would push it
  under ~72.5%).

### 02:25 UTC - arm 1 result (r1b2_nt05, warm + --noise-tau 0.5, 750M -> 925M) and arm 2 launched

| eval | score | best | median | CP | deepest end |
|---|---|---|---|---|---|
| 750.3M (resume) | 26.05 | 26.04% | 24.94% | 1 | 24.4 s at (4265, 768, 11932) |
| 825.8M | 27.28 | 27.28% | 27.16% | 1 | 25.5 s at (3199, -899, 11165): the CP2 dive |
| **901.3M** | **27.90** | **27.90%** | 27.29% | 1 | **26.3 s at (2654, -909, 11341)**: 500 u further west, 190 u higher |

* 27.90 beats the incumbent (27.22) and the best eval anyone had (27.5), but it is a CREEP of the
  CP2 death point, not a crossing (CP still 1). sidGEOuEM2 crept the same way to 28.0% (3,681M).
* Training side: `dip/max_survived_depth` spikes of 2.0-2.65 reward units (809M, 840M, 872M,
  877M) and 7.12 (916M) against ~0.4-0.6 in the incumbent run; the reservoir's deepest state did
  NOT move (mind 74.3-74.6%), so no training episode got past the CP2 death point.
* **Cost: the policy collapses its own noise against the correlated draw.** entropy_loss
  -1.13 -> +0.67 (about 1.8 nats lost), yaw sigma 0.041 -> 0.021 in 126M steps, approx_kl climbing
  0.03 -> 0.07 -> 0.13 -> 0.19 -> 0.22 -> 0.39 (incumbent run ~0.05). Likely cause: a
  sustained deviation that loses ground is charged the field dip on every one of its ~12
  correlated decisions, so "noise is costly" arrives ~tau/40ms times stronger than under i.i.d.
  noise and the entropy bonus (0.005 x (1+T), T 0.27 -> 0.11) cannot hold it.
* Weights: `runs/r1b2/r1b2_nt05/ckpt_latest.pt` = step 925,368,320 (md5 29b902d0...), 24M steps
  AFTER the 27.90 eval (ckpt_latest is rewritten every 60 s, so the 901M weights are gone).
  Pull lesson: `box.py pull --ckpt` while the trainer runs can grab a half-written ckpt_latest.pt
  (md5 taken, then the file rewritten): stop the trainer first, or pull a milestone ckpt.

**Arm 2: r1b2_ntr = arm 1's 925.4M weights + --race-ratchet (noise_tau 0.5 restored from the
ckpt) + --ckpt-every 75e6** (milestone ckpts aligned with the 75M evals, so the best eval's weights
exist). Why the ratchet now: it makes lost ground free, which is exactly the charge that (a) blocks
learning from a detour that only half works and (b) drives the noise collapse above. Ratchet
column zero-padded on resume (6 tensors widened, function-identical at step 0). Restored unstuck T
0.081, best 27,280u. Stop at 1,325M; judged at its 30-min mark (~02:56 UTC). One mechanism added on
top of arm 1, so the attribution is "ratchet given noise-tau".

### 02:52 UTC - arm 2 first read (r1b2_ntr: arm 1's 925M + --race-ratchet, noise_tau 0.5)

| eval | score | best | median | CP | note |
|---|---|---|---|---|---|
| 925.9M (resume = arm 1's weights) | 27.18 | 27.18% | 16.76% | 1 | 3 of 9 episodes die early |
| 1,001.4M | 27.32 | 27.32% | 27.22% | 1 | 9/9 at the CP2 dive, x 3,16x-3,22x, z ~11,150 |

* approx_kl 0.2-0.6 is mostly a UNITS effect, not instability: the Gaussian head's KL scales
  as (d mu / sigma)^2 and sigma halved (0.043 -> 0.021), so the same mean steps read ~4x larger.
  The evals are steady.
* T 0.05, stuck 105M at 1,007M: the recipe's own schedule starts heating at ~1.10B (patience 2e8)
  and reaches ~1.1 by the run's end (1,325M). Hypothesis being watched: the CP2 deviation needs
  correlated noise AND a temperature that makes "not D" likely per decision; at T 2 the side
  head's mode loses ~2/3 of its logit gap.
* Milestone ckpts land ONE iteration before the eval they shadow (ckpt_1000865792.pt vs
  traj_1001390080: --ckpt-every counts from 0, --record-every from the resume's first eval);
  one PPO update apart, close enough to hand on.
* --noise-tau-frac F (65e340c): AR(1) on round(N*F) evenly spread envs, i.i.d. on the rest, to
  dilute the entropy-collapse pressure ~1/F. Built and smoked, not yet run on a box.

### 03:18 UTC - arm 2 stopped (KL runaway); arm 3 = incumbent + ratchet + noise-tau 0.5 on 25% of envs

**Arm 2 verdict (r1b2_ntr, 925M -> 1,128M, 203M steps):** evals 27.18 / 27.32 / 27.20, all 9/9 at the
CP2 dive (x ~3,2xx, z ~11,15x); no crossing, no creep. **approx_kl ran away**: iteration means
0.26 -> 0.39 -> 0.52 -> 0.97 -> 1.41 -> 1.64, single updates up to 10.6 (the i.i.d. recipe runs
0.02-0.05). Sigma had stopped shrinking (0.021 -> 0.025), so this part is NOT the sigma-units
effect: the trust region was gone. Stopped before a collapse. Weights kept on the box:
ckpt_1000865792.pt / ckpt_1076363264.pt (one update before the 1,001M / 1,077M evals).
**Reading:** over ~370M steps (arms 1+2) the all-env AR(1) draw with the MARGINAL log-prob drove the
policy steadily into a high-KL regime. The mechanism is biased: each decision of a sustained
deviation is credited with the whole deviation (score e_t/sigma instead of the innovation's).

**Arm 3 (r1b2_ntf), from the HEALTHY incumbent (749.7M, kl ~0.05):** `--race-ratchet --noise-tau 0.5
--noise-tau-frac 0.25 --ckpt-every 75e6` (256 of 1,024 envs AR(1), the rest i.i.d.). Box redeployed on
c330cdc with the incumbent ckpt (md5 ede4be9a). Stop at 1,149.7M. Watch: KL staying near the i.i.d.
level, the window re-pass, CP2.

### 03:27 UTC - --noise-tau-cond (2265c67): the exact-credit version of the correlated draw

The bias in --noise-tau's update: the stored log-prob is the MARGINAL one, so the score of each
decision is e_t / sigma = (rho e_prev + sqrt(1-rho^2) xi_t) / sigma - it credits the decision with
the part of the deviation that EARLIER decisions' noise caused. A sustained deviation is credited
~1/(1-rho) times (13x at tau 0.5 s); arms 1-2 ran the KL away.
Fix: read the policy as pi(a | s, e_prev), with the AR(1) latent as exogenous state.
* View heads: z ~ N(mu + sigma rho e_prev, sigma^2 (1 - rho^2)).
* Key heads: ONE latent per head, u = Phi(e), and the action is the bin whose CDF interval
  holds u (inverse-CDF sampling, so the marginal is still softmax). Its probability is
  Phi((ndtri(C_k) - rho e_prev)/s) - Phi((ndtri(C_(k-1)) - rho e_prev)/s), with s = sqrt(1-rho^2).
* Only the decision whose innovation STARTS a deviation is credited with it. rho = 0 (the i.i.d.
  envs of --noise-tau-frac) is exactly the plain log-prob, and the entropy bonus stays the
  marginal's.
* Cost: in correlated envs most decisions have conditional P ~ 1 (the latent sits deep inside
  its interval), so their policy gradient is ~0. The i.i.d. envs and the boundary decisions
  carry the learning, while the correlated envs mainly explore. That is why it pairs with
  --noise-tau-frac.
* Tests (12 in test_noise_tau.py) include a Monte Carlo check of the conditional against the
  draw, and the rollout/update log-prob identity.

### 03:57 UTC - arm 3 verdict (r1b2_ntf, incumbent + ratchet + noise-tau 0.5 on 25% of envs) and arm 4

| eval | score | best | median | CP | note |
|---|---|---|---|---|---|
| 750.3M (resume) | 24.80 | 24.80% | 22.27% | 1 | |
| 825.8M | 27.33 | 27.33% | 27.26% | 1 | 9/9 at the CP2 dive; same as arm 1 at 825M (27.28) |
| 901.3M | 27.39 | 27.39% | ~24.5% mean | 1 | frontier 26,783u |

* **Training health: frac 0.25 fixes the runaway at low T.** approx_kl stays at 0.055-0.062
  (iteration max 0.08-0.13) for all 150M steps, the i.i.d. level, against arm 1's 0.03 -> 0.39
  over the same span. Entropy drifts -1.28 -> -0.88 (arm 1: -1.13 -> +0.67), about 4.5x slower,
  as the 1/frac dilution predicts.
* No CP2 crossing; T cooled to 0.10 after the window re-pass (825M, a new frontier best), so the
  wall was explored at near-greedy temperature the whole time.

**Arm 4 (r1b2_nth) = arm 3's 901.3M weights + `--unstuck-patience 5e7 --unstuck-rate 1.0
--ckpt-every 75e6`** (ratchet, noise_tau 0.5, frac 0.25 restored). This is a warm-measurement
setting, not a recipe constant. The schedule counts STEPS, and on this 65k fps box the default 2e8
patience is ~55 min of wall clock; the question is what correlated noise does at the recipe's T cap
of 2 AT the wall. T rises 1.0 per 1e8 from 0.096: ~1 at ~990M, 2 at ~1.09B. Stop at 1,301M.

### 04:25 UTC - arm 4 at 1.02B (T 1.28): the reservoir moves for the first time

* T 0.16 -> 1.28 over 915M -> 1,019M; approx_kl 0.041-0.049 (iteration max <= 0.09): healthy
  under heat, with 25% correlated envs.
* `dip/max_survived_depth` chunk means 0.27-0.29 at T 0.2-0.4 -> 1.1-4.0 at T 0.9-1.2.
* **Reservoir min-depth 74.50% -> 73.68% -> 73.57% of d0 once T passed ~1.1** (deepest
  harvested state d ~71,930u). Never moved in arms 1-3 (74.3-74.6%). A state enters the reservoir
  only if its episode lived >= 2 s after it, so training episodes now SURVIVE deeper at the wall.
  The CP2 death point is d ~70,900-71,300 (72.5-72.9%), not yet passed.
* Evals: 901.8M 27.40 (median 27.38), 977.3M 27.43 (median 27.33; the episodes switched to the
  y ~ +840 lane). No crossing yet.

**04:30 reservoir probe (`tools/res_probe.py runs/r1b2_nth/ckpt_latest.pt`, run on the box; it
prints the deepest harvested states of a ckpt's reservoir):**
* At 1,023M, 3,146 of the newest 20,000 reservoir rows are deeper than d 72,800 (the old
  floor). They span x 3,693-4,793, y 390-1,136, z 11,019-12,301.
* The deepest 25 rows sit at x 3,700-4,130, y 580-970, z 11,450-11,830: heading south
  (-30..-150 deg), falling (vz -500..-1,260), 1,000-1,360 u/s horizontal.
* That is near where the human is at 24.5 s, (4,041, 661, 11,814), and ~600 u BELOW the
  agent's habitual U-turn (z ~12,390). The exploration under heat reaches the human's line, and
  the 977M greedy eval switched to the same north lane.

### 04:48 UTC - arm 4 at T 2 (1.12B): training episodes FLY THE DETOUR REGION; no return yet

Reservoir late-map rows (d < 74.5k) by x band (box-side probe `res_region.py`, newest 20k rows):

| x band | arm 3 @ 900.7M (T 0.10) | arm 4 @ 1,120M (T 2.0) |
|---|---|---|
| 3,500-4,000 | 2,161 rows, z 12,163-12,481 | 1,539 rows, z 11,799-12,491 |
| 4,000-4,500 | 60 rows, z >= 12,374, all heading east | 237 rows, z 11,119-12,494 |
| 4,500-5,000 | 0 | 114 rows, z 10,978-12,379, 32% heading east |
| 5,000-5,500 | 0 | 46 rows, z 11,012-11,633, 46% heading east |
| 5,500-6,500 | 0 | 24 rows, z 10,984-11,475, 54% heading east |

* The human's detour is (4,041, 661, 11,814) at 24.5 s -> (5,554, 454, 11,079) at 25.5 s, then a
  ramp ride back west. Under heat with correlated noise, training episodes now reach x 4,500-6,500
  at z ~11,000-11,600 and SURVIVE >= 2 s there (harvested), half of them already heading back west.
  At low T nothing east of x 4,500 and nothing below z 12,160 was ever harvested.
* No episode has come back west past the death point yet (reservoir floor 73.48-73.63%, d ~71,840;
  the death point is d ~70,900), and the greedy line still dives (1,052.8M: 27.43).
* T 2.00 from ~1.09B, approx_kl 0.04-0.08: the 25%-correlated draw stays healthy at the cap.
* Extended (box to 08:47 UTC): this is the first arm whose training reaches the region the CP2
  detour needs. Plan: let it end at 1,301M, then continue from ckpt_final with the same flags.

### 05:15 UTC - **arm 4 LEARNS THE CP2 DETOUR in the greedy line (1,203.8M, T 2)**

Eval 1,203.8M: score 27.64, top episodes die at 28.4-28.7 s instead of 25.2 s. Episode 4, every
0.3 s (record.py rows):
* 23.5-24.1 s: the usual right air U-turn (D held), to (4262, -649, 12086) heading -178 deg.
* **24.4-25.6 s: back EAST**, descending, accelerating 1,307 -> 1,878 u/s, to (5,633, -301, 11,038)
  = the human's ramp, which the human reaches at (5,554, 454, 11,079) at 25.5 s.
* **25.6-26.2 s: A held, a left turn ON THE RAMP** (heading 52 -> 108 -> 160 deg) at z 11,038-11,112
  with no speed loss (1,850 u/s).
* **26.5-27.7 s: back WEST climbing** (vz +277..+679), (3,846, 869, 11,469) at 27.1 s and
  (2,976, 591, 11,588) at 27.7 s. That is past the old death point (x ~3,050, fail plane z ~11,200)
  ~400 u above the plane, higher than the human (z 11,469 at x 2,882, 27.4 s).
* 28.0-28.7 s: then it turns south (heading -85..-89 deg), curls back east, descends into the same
  pit and dies at (3,188, -975, 11,153). The part AFTER the detour is unexplored: this is the new
  frontier.
* Training side: reservoir floor 73.6% -> **71.80% (d ~70,200)**, past the death point
  (d ~70,900). Training episodes get beyond it and live >= 2 s more.

**What did it:** correlated sampling noise (`--noise-tau 0.5`, 25% of envs) + the ratchet + the
recipe's unstuck temperature at its cap of 2, from the incumbent's own weights (no demo, nothing
map-specific). At T <= 0.3 the same noise only crept (arms 1-3); at T 2 the deviations last long
enough to reach the ramp (04:48 probe), and ~110M steps later the greedy policy flies the
detour. Timeline at the wall: T reached 2 at ~1.09B, the detour region was harvested by 1.12B,
the greedy detour appeared at 1.20B.

### 05:48 UTC - arm 4 final: **28.49 at 1,279.3M** (new best of the program); continuation r1b2_nth2

* 1,279.3M eval: score 28.49, best 28.49% at 29.2 s ending (2265, 856, 10719); second 28.2% at
  29.9 s (2615, 489, 9574); median 27.07%. The detour line now clears the old death point and dies
  ~1,000 u further west and lower. Still CP1: sgcp_2 is ~6 s further (the human crosses it at 35.1 s).
* The 1,279M frontier (27,856u) beat the best (26,720 + 500): T is cooling (2.0 -> 1.72 at 1,301M).
  Novelty income jumped to 2.3/episode (was ~0.3): training episodes are in new cells.
* Arm 4 ended at 1,301.3M (69k steps/s average). **Continuation r1b2_nth2** = ckpt_final, same
  restored flags (ratchet, noise_tau 0.5, frac 0.25, unstuck patience 5e7 / rate 1.0), --ckpt-every
  75e6, stop at 2.5B (it will be stopped before the wrap-up).
* **Hand-on note for the wrap-up:** the orchestrator takes the next incumbent's ckpt from the BEST
  EVAL'S run directory (ckpt_final.pt, else ckpt_latest.pt) and ignores RESULT.json's ckpt. The
  milestone ckpt_<step>.pt one update before the best eval must therefore be placed there.
* This box's upload is ~35 KB/s right now: an 80 MB ckpt takes ~40 min. Pull checkpoints early.

### 06:15 UTC - pulling checkpoints over a throttled link

* `box.py pull --ckpt` timed out at its 30-min scp limit with 56 of 80 MB (31 KB/s) and left a
  truncated ckpt_final.pt (removed). The throttle is PER CONNECTION: splitting the file on the box
  (`split -b 8M -d ckpt part_`) and fetching the 10 parts in parallel (`rsync --partial`, retries)
  took 297 s with matching md5.
* `runs/r1b2/r1b2_nth/ckpt_latest.pt` is now a copy of ckpt_1278738432.pt (md5 9dc75cd9...), the
  weights one update before the 28.49 eval (CKPT_NOTE.txt says so). That is what the orchestrator
  hands on from this directory.
