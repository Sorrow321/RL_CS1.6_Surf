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

## Round 1, branch r1b1 - 2026-10-20 01:06 UTC (box 56813988, RTX 5080 + 7950X3D, $0.388/h)

### The box

* vast instance 56813988, RTX 5080 on a Ryzen 9 7950X3D (32 threads, quota 30.7 CPUs), $0.388/h.
  The incumbent recipe resumed at 1024 envs runs **250-265k steps/s** there (GPU 20% busy,
  CPU 88%): 75M steps per ~5 min, so 350M is ~25 min.
* `tools/gpu_health.py` had no RTX 5080 reference, so its power-cap heuristic (250 W of 360 W)
  failed every 5080 at deploy. Measured on this box: bf16 GEMM 112 TFLOPS (the card's spec) and
  816 GB/s copy (85% of 960, the 5090 reference's ratio). Added as the 5080 reference (7281e2f).

### The incumbent checkpoint is already AT the CP2 wall

* Its 755M eval: 2/9 episodes die in the CP2 pit (x 3,243-3,293, z 11,111-11,149, 25.2-25.6 s,
  27.1-27.2%); the rest die earlier (14.9-24.4 s). So a warm resume measures at CP2 at once.
* The agent's line near CP2 (surf run 3,775M, GR 755M, the same in both): east along y ~ 0 at
  23.5 s (4,273, 28, 12,381, vx +1,383), then a hard right U-turn on a curved surface
  (24.0 s: v (-186, -1,468); 24.5 s: v (-1,487, -60)), then west and down into the pit.
  The human (ledger) is at (4,041, 661, 11,814) at 24.5 s - the +y side, lower - and goes
  on EAST to (5,554, 454, 11,079) before coming back west over the pit.

### r1b1_dc1: warm + `--death-charge 1.0` - NEGATIVE

* Resume of the incumbent (749.7M) with the existing Grzes-2017 charge: a death gives back
  Phi(last state), the progress since the MAP start (stock arithmetic, not the spawn).
* 350M budget, stopped at 1.01B (by pid). Evals (score.py): 750M 26.0, 825M 24.7, 901M 24.7,
  977M 24.7 (best episode). 0 finishes.
* **What it did:** within ~75M steps the greedy line changed. All 9 episodes now end at the
  SAME point, (3,664, 1,136, ~12,003) at 24.0-24.6 s: at 23.5 s the agent turns NORTH
  (v (0, 1,027, 53)), stops dead against a wall at y 1,136 and falls into a fail. The pit
  U-turn is gone, replaced by an earlier death.
* **Why:** with kappa 1 the shaping nets ~0 over any episode that dies, so with the time
  penalty kept the objective prefers the EARLIER death (24.2 s vs 25.4 s saves ~0.6 units).
  The charge did move the route at the decision point (KL 0.09 on the first updates), but
  toward suicide, not toward the detour. This is the round-27 xNGS collapse seen locally.
  Training reward -12.8 at the resume -> +0.6 by 1.0B as the critic absorbed the charge;
  reservoir min-depth 74.3% -> 76.1% (the deep states stopped being harvested).
* Verdict: do not run `--death-charge 1.0` with the time penalty on this map. Untested:
  kappa 1 with `--time-pen 0` (the discount then favours LATE deaths).

### NEW: `--ez-eps` under `--view-continuous` and `--unstuck` (cba79e8)

* ez-greedy (Dabney et al. 2021) existed for the bins and was refused under the continuous
  view and under --unstuck. Now: a burst starts with prob eps per decision, lasts
  n ~ zeta(mu) decisions (cap --ez-max), holds a uniform draw of the YAW z (tanh(z) uniform
  on (-1, 1), so the offset is warped like the policy's: |off| < 10 deg half the time,
  up to 180) and of the four key heads; the pitch stays the policy's. Burst rows are dropped
  from the PPO pool (b_ez), as on the bins. The step line prints `ez X%`, the dropped share.
* Generic: no map constant; one rate, one zeta exponent, one cap.
* Tests (tests/python/test_ez_viewc.py, CPU, sidistic): the abs-view + keys-hold + unstuck
  smoke bursts and resumes (ez_eps restored); flag off is bit-identical to the pre-change
  trainer (config, progress.csv, eval bytes, weights).
* Why it is the next arm: 1.5B steps at T = 2 never sampled the human's continuation east.
  The policy's yaw sigma is 0.05-0.1 rad per 40 ms decision and its draws are independent,
  so a committed ~1 s deviation from the U-turn has probability ~0. A burst holds one for
  up to 2.4 s.

### r1b1_ez2: warm + `--ez-eps 0.02` (uniform bursts everywhere) - no crossing (2026-10-20 01:40 UTC)

* 400M from the incumbent (749.7M -> 1.15B), ~7% of decisions burst rows (dropped from PPO).
* Evals (score.py best / median): 750M 27.0/24.7, 825M 27.2/25.4, 901M 27.2/25.7,
  977M 27.0/21.1, 1,052M **27.3**/24.4, 1,128M 24.6/22.5. Best 27.304 (beats the incumbent's
  27.222 by the pit's width, no more). 0 finishes, no CP2 crossing.
* Training ep_len fell ~1,100 -> ~470 decisions: the bursts kill episodes, and the greedy
  line got LESS consistent (median 21-25% vs the control's 27.1% at 825M, below).
* Novelty counts of the final checkpoint (256 u cells, summed over view x speed), against
  the incumbent's: 0 visits past CP2 (x -500..700, z 9,000-11,400) in both; the lower
  corridor under kill plate *45 (x 4,200-5,600, z 10,900-11,850) 21 -> 204 env-ticks. So
  bursts DO reach the human's lower level, ~2 s of agent time per ~100M steps: far too rare,
  and the reservoir (2 s pre-end margin, uniform over the main line) never keeps them.

### Geometry of the CP2 wall (analysis only, kill volumes from the bsp)

* The pit is kill *86 (AABB x 2,592-6,240, y +-1,152, z 10,752-11,332). The field's minimum
  above it sits just over the top (d 70,847 at (3,050, -900, 11,400)); nothing below is
  reachable. The field does NOT route through the fail: a kill-masked field
  (`--race-kill-aware`) would not change the gradient here.
* A thin fail plate, kill *45 (x 4,194-5,344, y +-1,152, z ~11,905), lies under the
  agent's U-turn level (~12,000-12,400). Several ez2 greedy episodes die on it.
* The human (ledger) is at (4,041, 661, 11,814) at 24.5 s: WEST of the plate's edge and
  BELOW its level; then (5,554, 454, 11,079) at 25.5 s: under the plate, east, low. Then a
  ramp, and back west over the pit top at z ~11,469 (137 u above *86's top).
* So the missing line drops to a lower level west of the plate around 23.5-24 s and flies
  east beneath it. The agent never leaves the upper level.

### r1b1_ctl: plain warm control (incumbent recipe, no new flag)

* 825M eval: best 27.3%, MEAN 27.1% across 9: the plain resume consolidates at the pit in
  ~75M steps, where the DC arm read 24.6% and the burst arm 23.9% at the same step. Both
  mechanisms cost consistency before the wall.
* Used as the wall checkpoint for the next arms (every later warm arm branches from it).

### NEW: `--ez-front B` (62086bc)

* ez bursts may START only in envs whose progress (start geodesic - d) is within the top B
  fraction of the reservoir's deepest progress (re-read per iteration). Go-Explore's "explore
  from the frontier": the yield at the wall rises ~1/B-fold for the same eps, and the line
  before the band runs undisturbed. Generic: one band constant B, no map quantity.
* Tests: the band is live in a CPU smoke (step line `ez X% front>Y%`), the config carries
  ez_front, a resume restores it, `--ez-front` without `--ez-eps` is refused; flag off is
  bit-identical to the trainer before both ez changes.

### WON'T vs CAN'T at the CP2 pit: a nudge probe says CAN'T for short yaw deviations (01:50 UTC)

* Wall checkpoint = r1b1_ctl's ckpt_latest, step 905,445,376, md5 f7642e59...; its 901M eval
  is 9/9 at the pit (best/median 27.4%, score 27.392). Kept at
  `/home/steam/ar/runs/r1b1/r1b1_ctl/ckpt_latest.pt`.
* Probe (scratchpad nudge_sweep.py around `record_ckpt.py --nudge-*`, ANALYSIS ONLY): one
  greedy episode per config, the yaw held at a velocity-frame offset (abs 0/+-45/+-90/135)
  or the policy's own offset shifted (+-30/+-60) for 12 or 25 decisions (0.48/1.0 s) from
  tick 2250, 2325 or 2400 (22.5-24.0 s), then the policy flies free. 60 nudges + the base.
* **No nudge crosses.** The base reaches 27.37% (pit, 25.2 s); the best nudges equal it
  (abs+0 = "keep straight": the U-turn comes from the curved surface, not the aim). The rest
  die EARLIER on the fail volumes around: kill *99 (x 2,528-3,680, z 11,552-11,972) at
  23.4-23.5 s, a fail at z ~12,280-12,300 at 24.0 s, plate *45 (z ~11,905) at 24.5 s.
* **The geometry that explains it** (kill AABBs): the line past the pit is a SLOT between
  the pit floor *86 (top z 11,332) and the ceilings *99 (bottom 11,552) / *45 (11,905). The
  human flies it level (vz -141 at z ~11,469) after a ramp turnaround EAST under *45; the
  agent enters it from above, diving from ~12,300 after its U-turn, and free fall cannot
  arrest a 800+ u descent. So the crossing needs being UNDER the plate heading east, or in
  the slot heading west and level - states the policy has never been in. Short
  perturbations of its own line (nudges, ez bursts) cannot produce that; they die on the
  surrounding fail volumes first.
* Consequence for mechanisms: perturbing the policy's actions at the wall is the wrong
  lever; putting it into the unvisited states next to its frontier (with plausible
  velocities) is the right one. That is what the user's `--respawn-novel` does (own
  frontier states moved to unvisited cells, velocity kept), never tried at CP2 (NV2 stalled
  at the window). Next arm: r1b1_nov = wall checkpoint + `--respawn-novel` (defaults).

### r1b1_nov: wall checkpoint + `--respawn-novel` (defaults) - explores the right region, no crossing (02:20 UTC)

* 905M -> 1,305M (400M, ~24 min at 282k steps/s). Novel rows 1,816-1,843 per pool, 43-46%
  of episodes; ceiling 27,284-27,330 u (27.9-28.0%).
* Evals (score.py best / median): 906M 27.5/27.3, 981M 27.4/27.4, 1,057M 27.5/27.4,
  1,132M **27.5**/27.4 (score 27.462, this branch's best), 1,208M 27.4/16.1, 1,283M 27.4/27.4.
  The greedy line stays at the pit, consistently; 0 finishes.
* Visit counts (256 u cells), control at 905M -> nov at 1,305M:
  lower corridor under plate *45 (x 4,200-5,600, z 10,900-11,850) 9,005 visits / 8 cells
  -> **181,358 / 252 of 280 cells**; east ramp area (x 5,300-6,200, z 10,700-11,400)
  63 -> **26,993 / 160 of 200**; past CP2 (x -500..700, z 9,000-11,400) **0 -> 0**.
* Reading: novel spawns put the agent where the human's line goes (under the plate, at the
  ramp) 20-400x more often than anything else tried, but in 400M steps no episode turned the
  ramp into a level flight west through the slot. Two candidate reasons: (a) the stock
  shaping charges the eastward leg in the corridor (the dip), so the policy learns to turn
  west at once into the pit floor; (b) the ramp turnaround is a skill still to learn.
  `--race-ratchet` would remove (a), but it adds an observation column (widenable, yet
  PROTOCOL forbids shape flags on a resume), so it can only be tested from scratch.
* The unstuck T was rising through this run (no +500 u eval best since 825M): ~1.5 at 1.3B.

### r1b1_novezf: nov checkpoint + `--respawn-novel --ez-eps 0.04 --ez-front 0.15` - worse line, no crossing (02:55 UTC)

* 1,305M -> 1,705M. Bursts 2.1-2.4% of decisions (band from 22.0-22.2% progress), novel
  43-44% of episodes; the unstuck T rose 1.47 -> 2.0 (cap; no +500 u eval best since 825M).
* Evals (best / median): 1,306M 26.8/25.7, 1,381M 27.1/16.1, 1,457M 25.6/16.1,
  1,532M 26.2/24.8, 1,608M 26.9/22.2, 1,683M 27.3/26.9. No crossing; the greedy line got
  LESS consistent (T 2 + bursts + novel). Not worth repeating.

### Corridor bench: from the corridor heading east, the policy turns WEST at once and falls

* ANALYSIS ONLY (scratchpad bench/corridor_bench.py + corridor_east.npy): 64 of the nov
  checkpoint's OWN deep reservoir states heading east (vx > 800, progress > 22.5%) moved,
  velocity kept, to random reachable points in the corridor under plate *45
  (x 3,750-5,200, y +-900, z 11,450-11,850) - the kind of row --respawn-novel makes.
  `record_ckpt --spawn-states`, 16 stochastic + 4 greedy, 15 s cap.
* 17/20 die in 1.2-2.8 s (median 1.8 s) in the pit floor (*86, z 10,790-11,280), ending
  WEST of where they started (x 2,840-4,280): the policy reverses toward the field's
  descent and drops. 3/20 reach the ramp area (x > 5,300, z < 11,600); one survives the
  whole 15 s there (at z ~10,850, below *86's top: the ramp lies outside the kill brush).
  **0/20 come back west through the slot.** Best progress 27.66% (an episode that flew
  west at slot level and fell 250 u short of the floor's west edge at x 2,592).
* So the gap is two things: the policy turns toward the field instead of going to the
  ramp (the stock shaping CHARGES the eastward leg: the dip), and it has no ramp-turnaround
  skill. `--race-ratchet` removes the charge (detours free) - but it adds an obs column,
  so it is tested FROM SCRATCH: r1b1_ratnov = incumbent recipe + `--respawn-novel`
  + `--race-ratchet`, 500M (30 min) first test against GR's gates (trap exit 529M,
  window 755M). PROTOCOL: extend only if it clears a gate sooner.
* The bench needs the box idle (record_ckpt is CPU-bound): ~2 min for 20 episodes idle,
  ~4 min for 3 episodes beside a trainer.

### r1b1_ratnov: FROM SCRATCH, incumbent recipe + `--respawn-novel` + `--race-ratchet` - AHEAD of GR; EXTENDED (03:25 UTC)

* Launched 02:53 on the 5080 box, 500M steps (30 min at 285k steps/s), commit 99b20f4-era code
  (no new flag: both mechanisms exist). Obs: ratchet gap column 27, surf 28, pot-rate 29.
* Matched-step evals (score.py best-episode progress; GR = the incumbent's own scratch run):

  | step | ratnov | GR |
  |---|---|---|
  | 302M | 5.8% (in the first-ramp trap) | ~4.6% |
  | 378M | **15.2%** (out of the trap) | ~4.6% (exits at 529M) |
  | 453M | **20.8%** (at the window wall, median 19.0%) | ~4.6% (19.4% mean only at 680M) |

* **It left the first-ramp trap with the unstuck T at ~0** (T 0.03, stuck 42M at 496M), where
  every earlier run exited only once T had risen past ~1.3 (GR at 529M with T 1.26; surf,
  cap-2, NV2 alike). The trap was the field paying a fall off the first ramp's side branch;
  the ratchet stops charging the turn back onto the ramp and the novel rows show the ramp.
  One seed, but ~150-225M steps ahead at two consecutive marks.
* **EXTENDED** (PROTOCOL: clears a gate sooner than the incumbent at matched steps):
  r1b1_ratnovx = bare resume of ratnov's ckpt_final (500,170,752; md5 48d9cee1...), the
  ratchet and novel spawns restored from the checkpoint (no flag passed), 2e9 more steps
  (to 2.5B, ~2 h). Purpose: reach the CP2 wall and test ratchet + novel spawns there - the
  combination the corridor bench pointed at (spawns in the dip + no charge for the eastward
  leg to the ramp).

### The ramp turnaround EXISTS on the +y side; the exit is too low (analysis, 04:30 UTC)

* Traced the corridor-bench episodes that reached x > 5,300 (nov checkpoint):
  * **-y side** (start (4,669, -6, 11,515), v (1,029, -793, 243)): hits a wall at x ~5,650
    at t = 1.0 s, loses all speed, crawls there 14 s sinking from z 11,400 to 10,930. Not a ramp.
  * **+y side** (start (4,769, 799, 11,501), v (1,676, -139, 177)): the ramp at x ~5,480,
    y ~970 turns it round in ~0.7 s - v (753, -1,401, -223) at 0.5 s, then (-1,483, 443, -623)
    at 1.0 s and (-1,488, -622, -162) at 1.5 s at (4,182, 611, 11,094): heading WEST at
    1,490 u/s. It is too LOW and descending and dies at x 4,047, z 10,914 (2.1 s).
* The human (ledger) is on the +y side too: (4,041, 661, 11,814) at 24.5 s,
  (5,554, 454, 11,079) at 25.5 s, then back west CLIMBING (vz +260..+422) to cross the slot
  at z ~11,469. So the turnaround is the map's geometry, not a precision trick; the missing
  skill is the exit: leave the ramp climbing, high enough for the slot under *99
  (z 11,332-11,552 at x 2,528-3,680).
* There is a smooth reward gradient toward it: after the turnaround, the further west the
  agent flies before the pit floor takes it, the more progress (both stock and ratchet).
  Best seen so far: x 2,840 / 27.66% (250 u short of the floor's west edge at 2,592).
* So the rate-limiting parts are (1) how often an episode starts or arrives in the +y
  corridor heading east (novel rows from east-heading sources, ~23-23.5 s of the line) and
  (2) the main line's run-up: the human drops under the plate's level WEST of its edge
  (x < 4,194) at ~24 s on the +y side; the agent climbs over to its U-turn instead.
* Bench v2 (scratchpad bench/corridor_east_py.npy: 64 own east-heading states moved to the
  +y corridor x 3,750-5,000, y 300-1,000, z 11,450-11,850; `xret` = how far west it comes
  back after the ramp) tracks the exit skill across checkpoints.
