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
