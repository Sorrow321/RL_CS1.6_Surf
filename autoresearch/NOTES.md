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

### MILESTONE (04:50 UTC): ratchet + novel spawns, from scratch, FLY THE HUMAN'S CP2 LINE at 1,633M

* r1b1_ratnovx eval 1,633M (score.py 28.553, best 28.55%, median 27.9%): 5 of 9 greedy
  episodes reach the pit floor's west edge (x 2,587-2,608, z 11,349-11,436, 28.0 s, 27.95-27.98%)
  and episode 6 passes it to x 2,224 (28.55%) before stopping dead and falling at 29.0 s.
* Episode 6, against the human (ledger):
  * 23.5-24.0 s: the usual U-turn at (4,673, -234, 12,344) -> heading west;
  * 24.5 s: (4,121, -945, 11,666), v (1,162, 426, -1,078): it DROPS under the plate's level just
    west of its edge (x < 4,194) and turns east again;
  * 25.0-25.5 s: east through the corridor under *45 to (5,557, -350, 10,989);
  * 26.0 s: (5,512, 466, 11,044), v (-1,376, 1,199, 34): the +y ramp turnaround
    (human (5,554, 454, 11,079) at 25.5 s);
  * 26.5-27.0 s: west CLIMBING, vz +305, +444 (human +260..+422);
  * 27.5 s: (3,048, 838, 11,492), vz +44: LEVEL in the slot (human z ~11,469, vz -141);
  * 28.0 s: (2,242, 886, 11,415): past the pit floor's west edge; 28.5 s: horizontal speed ~0
    at x ~2,229 (an obstacle), falls, dies at (2,224, 815, 10,715).
* The unstuck T was 1.76 at 1.40B and halved after this +1,029 u best (0.80 at 1.89B).
* The CP2 curtain is still ahead (sgcp_2 at x 768-816, z 10,060-11,300); the next obstacle is
  whatever stops the flight at x ~2,229.
* How it came: ~1B steps at the wall with novel rows in the corridor (the turnaround is the
  +y ramp's geometry) and the ratchet (no charge for the eastward leg). None of the warm arms
  without the ratchet (nov, novezf: 800M at the wall between them) produced it.

### r1b1_ratnovx end (2.5B) and continuation r1b1_ratnovx2 (05:25 UTC)

* Evals after the milestone (best / median): 1,708M 27.5/27.0, 1,784M 27.5/27.4, 1,860M 27.5/27.5,
  1,935M 27.2/27.1, 2,011M 27.5/22.6, 2,086M 27.4/26.4, **2,162M 28.0/27.9**, 2,237M 27.3/27.2,
  **2,313M 28.3/27.4**, 2,388M 27.4/25.6. The greedy line flips between the old pit dive and the
  human's line; T back at the cap 2 by 2.4B (no +500 u best after 1,633M).
* Training side: reservoir min-depth 72.165% at 2.42B (deepest harvested state 27.8% progress,
  past the old pit death at 27.5%): training episodes now survive past the pit edge for >= 2 s.
* The line is flown on BOTH lanes (the map is mirror-symmetric in y): 2,162M episodes die at
  the floor's west edge on the -y side too (x 2,576-2,635, y -859..-959, z 11,345-11,415).
* **The next obstacle:** the 2,313M best episode flies the slot level (z 11,543-11,562 at x
  3,445-3,096, vz ~0) and at 28.5 s hits something facing east at x ~2,224 (y ~700-900,
  z ~11,320-11,450): vx -1,399 -> 0, deflected to -y, falls. The human's dip 2 (27.9-31.2 s,
  1,978 u) starts right here, and CP2 (sgcp_2, x 768-816, z 10,060-11,300) is 7 s further.
* Checkpoint kept: /home/steam/ar/runs/r1b1/r1b1_ratnovx/ckpt_final.pt (2,500,329,472,
  md5 f845951c...). ckpt_1500512256.pt is on the box.
* **Continuation r1b1_ratnovx2**: bare resume of that checkpoint, same recipe, 2.6e9 steps
  (to 5.1B, ~08:00 UTC; the idea's 6 h cap from 02:53 ends 08:53). Watching for: CP2 crossings,
  the frontier past x 2,224, and whether the greedy line consolidates on the new line.

### 06:25 UTC: ratnovx2 at 3.48B; the obstacle after the slot is a DESCENT (analysis only)

* Evals 2.5-3.4B (best % / mean %): 27.2-28.3 / 23-28; 3,331M has mean 28.0% (most episodes
  fly the slot), best 28.3%. No new best past 1,633M's 28.55%; T at the cap 2 since ~2.4B.
  Reservoir min-depth 71.714% (deepest harvested state 28.3% progress).
* Geometry at the slot's west end (y +800 lane, hull traces + the goal field):
  * a hull trace west at the slot height (z 11,450) stops at x 2,224 on a wall facing +x -
    exactly where the episodes die (vx -1,400 -> 0 at 28.5 s);
  * the pit's kill floor *86 ends at x 2,592, so below the slot level west of it is open;
  * the field descends DOWN then west there: d 70,142-70,213 at z 11,400 (x 2,200-2,400),
    ~69,957 at z 11,200, 69,210 at (2,000, 800, 10,600), 69,124-69,910 at x 1,200-1,800,
    z 10,600-11,200 (x 2,000 is solid above z ~10,700).
* So the next skill is to leave the slot's level once past the floor's west edge and drop
  ~400-800 u before x ~2,224. The kill floor is a trigger (not drawn in the depth image), so
  the policy must learn where it ends from experience; the novel rows' ceiling (28.8%) now
  reaches past it. Under the ratchet the drop is paid (d falls), so it is a gradient, not a dip.

### 07:25 UTC: incumbent_recipe updated; what the next round inherits

* `autoresearch/config.json` `incumbent_recipe` now ends with `--respawn-novel --race-ratchet`
  (the r1b1_ratnov recipe). box.py prepends it to every SCRATCH launch; a resume restores both
  from the checkpoint (the ratchet adds obs column 27 - never pass it to a resume of a
  checkpoint without it, PROTOCOL's shape rule; a checkpoint WITH it restores it bare).
  Nothing else in config.json changed.
* The orchestrator takes the next incumbent checkpoint from the best eval's run directory
  (`ckpt_final.pt`, else `ckpt_latest.pt`). The best eval is in r1b1_ratnovx2, so its final
  checkpoint (5.1B) is pulled there at the end of the run.
* State of the lineage at 4.46B: the greedy line crosses the CP2 pit by the human's route in
  most evals (means 25-28%), and stops at the slot's west wall (x 2,224) - frontier 27,919 u
  (28.56%) unchanged since 1,633M, T at the cap 2 since ~2.4B.

### 08:55 UTC: the ratchet + novel recipe at its 6 h cap (5.9B); ablation launched

* r1b1_ratnovx2 (2.5-5.1B) best eval 4,917M score **28.628** (best 28.63%, median 27.7%): the
  episode crosses the pit and the slot, hits the x 2,224 wall at 28.25 s, bounces and falls
  (dies at (2,139, 922, 10,980), 28.8 s). r1b1_ratnovx3 (5.1-5.9B, the last of the idea's 6 h):
  best 28.467 (5,554M), means 23-28%. The frontier has not moved by +500 u since 1,633M; T at
  the cap 2 since ~2.4B. Branch score (orchestrator's best_in over every pulled run): 28.628,
  105/105 valid evals; the next incumbent checkpoint is r1b1_ratnovx2/ckpt_final.pt
  (5,100,797,952; md5 3cc658ed...), local.
* Ablation **r1b1_ratonly** (08:52): the OLD incumbent recipe + `--race-ratchet` only (no
  `--respawn-novel`), from scratch, 500M (`box.py launch --no-recipe` + the old flags). Question:
  does the ratchet alone give ratnov's early gates (first-ramp trap exit 378M, window ~500M;
  GR 529M / 755M)? One seed; the 30-min early mark is the comparable one (CLAUDE.md).

### Ablation 1 (09:20 UTC): `--race-ratchet` ALONE does not give the early gates

* r1b1_ratonly = the old incumbent recipe (GR) + `--race-ratchet`, no novel spawns, from scratch,
  500M (306k steps/s). Best-episode progress per eval: 76M 4.4%, 151M 4.6%, 227M 4.6%,
  302M 4.6%, 378M 4.7%, 453M 4.6%. Still at the first gate at 453M, like GR (4.6% to 454M,
  trap exit 529M) - and nowhere near ratnov (ratchet + novel): 15.2% at 378M, 20.8% at 453M.
* So the early speedup comes from `--respawn-novel` (or needs both). One seed.
* Ablation 2 launched 09:21: r1b1_novonly = GR recipe + `--respawn-novel`, no ratchet, 500M.

### Ablation 2 (09:52 UTC): `--respawn-novel` ALONE gives the early gates

* r1b1_novonly = the old incumbent recipe (GR) + `--respawn-novel`, no ratchet, from scratch,
  500M (294k steps/s). Best-episode progress (mean): 76M 4.4% (4.3), 151M 4.6% (4.5),
  227M 4.6% (4.6), **302M 10.1% (7.9) - out of the first-ramp trap**, 378M 15.8% (15.0),
  453M 17.6% (16.2).
* Matched marks (best %): 302M - GR 4.6 / ratchet-only 4.6 / novel-only **10.1** / both 5.8;
  378M - 4.6 / 4.7 / **15.8** / 15.2; 453M - 4.6 / 4.6 / 17.6 / **20.8**.
* Reading (one seed each): the novel spawns are what moves the first gates; the ratchet adds
  nothing there. Whether the ratchet is needed for the CP2 pit crossing is the open question,
  so the novel-only run is EXTENDED (it clears the trap sooner than GR at matched steps):
  r1b1_novonlyx = bare resume to 2.1B (~11:25 UTC). The ratchet + novel lineage crossed the
  pit at 1,633M (wall reached at ~651M).

### Ablation 2 extended (r1b1_novonlyx, stopped 10:23 UTC): novel spawns WITHOUT the ratchet stall at the window

* 500M -> 1.02B: every eval 20.3-20.8% (best) from 500M on - the window wall, exactly where
  NV2 (surf + novel, no ratchet) stalled from ~1.0B to 1.41B. ratnov (ratchet + novel) passed
  the window at 576M with T ~0; GR (neither) at 755M. Behind GR at matched steps from 755M ->
  stopped (PROTOCOL). The box was released at 10:24 UTC (~9.7 h, ~$3.8).
* Reading (one seed per arm): novel spawns move the first gate; WITHOUT the ratchet they stall
  at the window (2 runs); WITH it the window falls at 576M and the CP2 pit at 1,633M. The
  ratchet alone does nothing early. The combination is the recipe.

### The obstacle past the CP2 slot, for the next round (analysis only)

* Two thin fail plates below the slot's west end: kill *17 (x 1,952-2,208, y 832-960,
  z ~10,950; the 4,917M best episode died on it) and kill *21 (x 2,208-2,528, full width,
  z ~10,690; the 2,312M best died on it).
* The field's steepest-descent route from the slot's end ((2,400, +-850, 11,450), d ~70,240):
  diagonally down and toward the centre - (2,208, 850, 11,258), (2,048, 818, 11,066),
  (1,856, 722, 10,874), (1,760, 530, 10,682) - then west along y ~0 at z ~10,400 through
  x 1,632-864 and into the CP2 curtain (sgcp_2, x 768-816, z 10,060-11,300) at d ~67,500-67,750
  (~31%). Mirror-identical from the -y lane.
* So the wall at x 2,224 is an obstacle only above z ~11,300 (the route passes x 2,208 at
  11,258), and the agent arrives level at 11,450-11,550. The next skill: start descending at
  the pit floor's west edge (x 2,592; its top is 11,332) so as to pass x ~2,224 below
  ~11,300, then steer to the centre between the plates. In free fall from level flight a
  368 u approach loses only ~30 u, so it needs a downward velocity set up before the edge -
  the human's dip 2 (27.9-31.2 s) is probably that set-up. Under the ratchet the dip is free.

## ROUND 1 LESSON (r1b1) - written 10:30 UTC

**Result:** best greedy eval from the true start **28.628** (r1b1_ratnovx2 at 4,917M; incumbent
27.222, best before the round 27.5). 0 finishes. The CP2 pit - the wall every run since 10-18
died at (27.5%) - is crossed by the human's own line, champion-free, from scratch.

**The recipe that did it** (now `config.incumbent_recipe`): the round-0 incumbent recipe
(GR: unstuck eval-frontier, margin 2, cap 2, surf flag + bonus, compass, pot-rate)
**+ `--respawn-novel` + `--race-ratchet`**, from scratch, defaults, no new code:
* first-ramp trap exit 378M (GR 529M), at T ~0; window 576M (GR 755M); CP2 pit wall 651M
  (GR ~850M, surf ~1.9B); after ~1B steps at the wall the greedy line flies the human's
  route at 1,633M (drop under plate *45, east to the +y ramp, turnaround, climbing exit,
  level through the slot between kill *86 and *99) - 28.55%, then 28.0-28.6% at most evals
  from 2.1B to 5.9B.
* Ablations (one seed each, from scratch, 500M): ratchet ALONE = GR (stuck at 4.6% to 453M);
  novel ALONE = fast first gate (10.1% at 302M) but stalls at the window (20.8% to 1.02B,
  as NV2 did). Neither half alone; the pair.

**What did not work (warm, at the CP2 pit, 300-400M each):** `--death-charge 1.0` (prefers an
earlier death - north wall, 24.7%); ez-greedy bursts everywhere (`--ez-eps 0.02`, new code
for the continuous view: no crossing, line noisier); `--respawn-novel` alone from the wall
(visits the corridor/ramp 20-400x more, no crossing); novel + frontier-only bursts
(`--ez-front`, new) at T 2 (noisier, no crossing).

**Diagnostics worth reusing** (`autoresearch/analysis/`): the nudge sweep (WON'T vs CAN'T: no
short yaw deviation crosses CP2) and the corridor bench (from the corridor heading east the
policy turned west into the pit floor; the +y ramp turns it round by geometry; the exit was
too low). Both pointed at the ratchet: the stock field charged the eastward leg.

**Where the lineage stops:** the slot's west end (28.6%): a wall at x 2,224 above z ~11,300,
fail plates *17 / *21 below; the field route dives diagonally to the centre and runs west at
z ~10,400 into CP2 (~31%). The frontier has not moved by +500 u since 1,633M (T at the cap 2
since ~2.4B). Next round: start from the incumbent checkpoint (5.1B, has the ratchet column -
a bare resume restores both flags) or from scratch with the new recipe (~0.65B to the wall,
~1.6B to the pit crossing on this box).

**Ops:** the RTX 5080 + 7950X3D box ran the recipe at 270-306k steps/s (vs ~194k on a 3090 +
5950X); `tools/gpu_health.py` needed a 5080 reference (added). `score.py --best <agent dir>`
does not recurse into runs; the orchestrator passes each run dir (fine). The next incumbent
checkpoint is taken from the BEST EVAL's run dir, so pull that run's ckpt_final.

## Wrap-up (r1b1) - 2026-10-20 11:52 UTC

* Final: RESULT.json `"final": true`, score **28.628** (r1b1_ratnovx2, eval traj_4916772864,
  re-scored at wrap-up: 9 episodes, best 28.63%, median 27.72%, 0 finishes). Hand-on
  checkpoint: `runs/r1b1/r1b1_ratnovx2/ckpt_final.pt` (step 5,100,797,952, md5
  3cc658eda8fb8bafe3b5f2a26418dc52), pulled before the box was released and md5-checked again.
* The whole best lineage (ratnov -> ratnovx -> ratnovx2) was trained at commit 2b845b3, which
  is code-identical to this head (only `autoresearch/config.json` differs).
* Box 56813988 (RTX 5080 + 7950X3D, $0.388/h) ran 00:41-10:24 UTC, about 9.7 h / ~$3.8,
  released at 10:24; no box at wrap-up.

## Round 2, branch r2b2 - 2026-10-20 13:25 UTC: the 28.6% wall is the asymptote of an infeasible field route

All analysis only (CLAUDE.md sections 0 / 0b); scripts in the session scratchpad, numbers here.
No box could be rented from 12:55 to 13:25 UTC (every offer under config.max_dph was either
blocklisted or a cent over the cap), so this was done on the local CPU.

### Geometry past the CP2 slot (hull traces, kill AABBs + hull_probe, the goal field)

* The slot (between the *86 kill layer below and the *99 kill ceiling above) ends at a THIN WALL at
  x 2,512-2,576. Its openings (ducked-hull origin):
  * the **upper windows**, |y| 850-938, z ~11,400-11,490, with the no_jump kill plates *80 / *64 on
    their sills (raw z 11,392-11,394);
  * a **lower opening**, |y| < 110, z ~10,770-11,100, into a closed CAVITY under the pit's slab.
* West of the wall: a **shaft** x 2,208-2,496 (hull centre), full width in y, closed on top
  (solid at z ~11,520), kill floor *21 (raw z 10,688). Its only west exits are two **tunnels**
  under a solid block (block bottom z ~11,216-11,232): |y| 850-938, ducked origin z
  10,964-11,180, 256 u long (x 1,952-2,208), floored by the no_jump kill plates *17 / *113 (raw
  z 10,944-10,958). Beyond: a room (x 1,168-1,936) whose floor is kill *20 (raw z ~10,432, x
  848-1,772), a pillar (x 848-1,168, z 10,400-11,232), and the CP2 area (x 200-830) reachable only
  UNDER *20 / the pillar (z < 10,396) - the field's route: tunnel -> room -> drop east of x 1,772
  -> west at z ~10,300-10,400 -> CP2 (~31%) -> 39% further on.
* *86 is NOT a box: it is a thin kill layer on a sloped slab (from (2,600, 11,350) down to
  (3,400, 11,050), then flat at 10,776 to x 5,700); the slab is solid below it east of x 3,032.
* The cavity (x 2,580-3,036 at y ~0, z 9,564-11,100) is sealed except through the lower opening:
  *70's kill ramp on the west (thin solid, kill on its upper-west face, its top end meets the
  shaft floor), vertical walls east, kill floor *90 (z ~9,560). It holds one non-kill platform
  (x ~2,220-2,412 at y 0, top z ~10,050). Field progress in it 25.6-27.3%, i.e. up to ~2,000 u
  of d above the window - the size of the human's dip 2 (1,978 u, 27.9-31.2 s).

### The tunnel cannot be entered from the upper window by free flight

* Window (ducked origin z <= ~11,490) -> tunnel ceiling (<= 11,180 at x 2,208) needs >= 310 u
  of drop in 384 u of x; the tunnel then allows <= 204 u of drop in 288 u. With constant
  horizontal speed, gravity makes "drop >= 310 in the first 4/7 of the path, <= 514 over all
  of it" infeasible for ANY entry vz (the quadratic constraints have no solution).
* Brute force with the C core, 12,090 straight ducked flights from x 2,600, y 895 over z
  11,380-11,500, vz -2,400..+600, |v| 300-3,200: **best 28.79%, none past x 1,984**; with the
  vz the slot allows (>= -600): **best 28.74%** (z 11,470, vz -600, |v| 950 - slower and
  steeper than the agent's 1,570 u/s, -337).
* Random macro-action search (yaw offset, keys, duck held 1-10 decisions) from the policy's own
  8 deepest reservoir states: 13% pass the window, 2e-4 get under the block, 0 of 35k exit the
  tunnel. Iterated Go-Explore (cells 64 u x speed bucket x vz sign, count-based selection,
  400 x 512 rollouts, 69k cells): best 28.67%; every archived state inside the tunnel's x range
  falls at 800-900 u/s; no state rises anywhere in the shaft.
* **So the lineage's wall at 28.6% is the asymptote of the field's route, which is physically
  closed: the best any policy can score on it is ~28.74-28.8% (dying on *17's west end; the
  training ceiling is 28.82%).** The incumbent's 28.63% episode enters the window at 1,570 u/s,
  vz -337. The real route is something else (the human's dip 2 suggests the cavity; I found no
  exit from it).

### What novel rows look like at this wall (the incumbent checkpoint's own sampler state)

* d_best 69,594.5 (ceiling 28.82%). Of 28k sampled rows only 1.5% land at >= 27% progress;
  those in the room / cavity carry EASTWARD velocities (median vx +885..+1,163: their sources
  are the corridor-east states under *45), i.e. pointed away from the way on.
* The reservoir (20k states) has median progress 19%, p90 25.6%, max 28.0%: with the 2 s
  harvest margin it never holds states in the last 2 s before the window.

### 13:45 UTC: the human's route past the wall (WR demo 47250, ANALYSIS ONLY - section 0)

Downloaded from speedrun.eu (/demos/47250/download, surf_src_sidistic.dem, 8.6 MB) and parsed with
tools/demo/parse_hldemo.py; nothing from it enters a run (no states, lines, boxes, constants). Same
physics as the sim (gravity 800, airaccelerate 100) except maxvelocity 3,500.

| t (s) | position | velocity | what |
|---|---|---|---|
| 27.50 | (2,882, -910, 11,469) | (-1,833, -94, -141) | slot, -y lane, heading west |
| 27.75 | (2,429, -872, 11,408) | (-1,640, 814, -342) | through the -y upper window |
| 28.00 | (2,236, -521, 11,300) | (-30, 1,717, -537) | slides NORTH along the block face |
| 28.25 | (2,353, -117, 11,140) | (1,220, 1,211, -738) | turns EAST in the shaft |
| 28.50 | (2,705, 122, 10,929) | (1,012, 1,389, -938) | through the LOWER opening into the cavity |
| 29.00 | (2,450, 799, 10,354) | (-1,713, 163, -1,340) | falling under the shaft floor |
| 29.30-29.85 | (2,087, 484, 9,926) -> (2,152, -490, 10,002) | (-1,057, -1,590, -1,234) -> (1,020, -1,027, 1,190) | SURFS a ramp at the cavity's bottom: normal (-0.759, 0, 0.651), the west face of a block, x ~1,950-2,150, z ~9,790-10,050; lands moving south at ~2,000 u/s and strafes up it |
| 30.75 | (2,524, 20, 10,893) | (-853, 1,011, 629) | back WEST through the lower opening, RISING |
| 31.50 | (2,198, 908, 11,137) | (-1,208, 599, 33) | enters the +y tunnel LEVEL |
| 31.75-33.0 | room: (1,865, 939, 11,120) ... (1,428, -167, 10,736) | | surfs the room's east-facing ramp at y ~0 |
| 33.75-34.0 | (1,845, 660, 10,372) -> (1,501, 791, 10,155) | | drops east of *20's end, flies west under it |
| 34.50 | (1,112, 236, 9,730) | vz -112 -> +1,186 | another ramp at the void's bottom |
| ~35.1 | (~780, -540, ~10,200) | | CP2 |

* So the route IS a dip: ~2.4 s inside the cavity, d up to ~2,000 u above the window's (the
  ledger's dip 2), with a surf-ramp turnaround that converts the fall into the rise that makes the
  tunnel reachable level. The free-flight bound above is right; the ramp is what changes vz.
* My random / Go-Explore search reached the cavity's bottom region (3,805 cells below z 10,000) but
  never produced a single rising state there: hitting that ramp moving ALONG it (south, ~2,000 u/s)
  and turning up it is the hidden skill.
* It is the same kind of gate as the CP2 pit (dip into a space the field calls worse, ramp
  turnaround, climbing exit), which ratchet + novel spawns solved in round 1 after ~1B steps at the
  wall. Here the defaults put ~0.2% of novel rows near the cavity: frontier-focused settings
  (`--respawn-novel-shell 0.9 --respawn-novel-shell-width 0.05`) put 20.6% of novel rows in the
  cavity, and a 3,072 u radius also reaches the ramp's area (2.2% of rows west of it vs 0.03%).
  Measured offline on the incumbent checkpoint's own sampler state (no training).

### 14:10 UTC: box 56901034 (RTX 4090 + Ryzen 9 5950X, $0.401/h, Argentina) and arm 1

* No box passed box.py's filters from 12:55 to 13:54 (every offer under config.max_dph was over the
  cap by ~$0.001 or blocklisted). A background loop retrying `box.py rent` every 3 min got this one.
* **Harness issue (tools/gpu_health.py):** no RTX 4090 reference, so the power-cap heuristic (350 W
  of a 500 W default) failed the deploy. Measured on the box: bf16 GEMM 153 TFLOPS (93% of the 4090's
  165 dense spec), copy 918 GB/s, 2,400 MHz under load at 190 W (the cap never binds). Added a 4090
  reference on this branch (165 TFLOPS / 857 GB/s / 2,520 MHz, the 5080 entry's method); the
  operator's copy will not have it next round.
* **NEW `--respawn-novel-turn T`** (commit 4626387): a novel row's horizontal velocity and view yaw
  turned together by U(-T, T) deg (speed, vz, view offset kept); its own RNG, so the rows' positions
  are exactly the T = 0 rows; 0 = off (no draw, config dump unchanged). 6 CPU unit tests
  (tests/python/test_respawn_novel_turn.py). Why: the cavity's novel rows inherit +x headings from
  their sources (the corridor-east states), and the bench below shows the ramp needs an approach
  along it.
* Bench (measurement only, BENCH_ONLY states from the WR demo, record_ckpt --spawn-states, CPU):
  placed at the human's ramp-approach state (2,129, 551, 9,992) v (-903, -1,451, -1,541), the
  incumbent policy surfs the ramp in 1 of 3 episodes (vy -2,209, then turns up to vz +464..+604,
  heading east) but stays low and dies on *90 after 2.4 s; 2 of 3 fall off it at once. The skill is
  partly there; it has never practised in the cavity.
* **Arm 1, r2b2_nfturn** (14:06 UTC): WARM resume of the incumbent (5,100,797,952) + frontier-focused
  novel rows `--respawn-novel-shell 0.9 --respawn-novel-shell-width 0.05 --respawn-novel-radius 256
  3072` + `--respawn-novel-turn 90`, 450M steps, evals every 25M. It measures AT the wall: does
  practice in the cavity (now ~20% of novel rows) produce the ramp turnaround and a ceiling past
  28.82%? First readings: ~780 novel rows per pool, 17-22% of episodes from novel rows.

### 15:00 UTC: arm 1 (r2b2_nfturn) - no frontier move; bench; the band; arm 2b

* **Arm 1 result** (5,101M -> 5,551M, 450M steps, avg 206k steps/s on the 4090 + 5950X box; the
  instantaneous rate is ~300k, evals with 300 s truncated episodes eat the rest): 18 evals, best
  **28.308** (5,252M), the rest 27.3-28.3 plus ONE collapse at 5,403M (9/9 episodes turn back east
  at ~10.3 s, 10.7%; the next eval recovered). The training ceiling moved 28,174 -> 28,184 u (+10).
  At matched steps the incumbent's own continuation (r1b1_ratnovx3, 5.1-5.9B) had a best of 28.467,
  so the arm is not better. In training the frontier-focused rows had a progress median of ~17%
  (p90 26%): on this folded map the unvisited air near the CP2 sources is mostly EARLIER track.
* **Cavity bench** (cavity_bench.py on the box, MEASUREMENT ONLY: record_ckpt --spawn-states from
  WR-demo states, 8 stochastic + 3 greedy per set; "rose" = vz > 300 in the cavity, "back" = west
  through the lower opening, "lane" = in the tunnel lane):

  | start set | incumbent best / rose / back / lane | arm 1 final |
  |---|---|---|
  | shaft (27.85-28.15 s) | 28.26% / 0 / 0 / 0 | 28.26% / 0 / 0 / 0 |
  | lower opening (28.45-28.6 s) | 27.54% / 0 / 0 / 0 | 27.54% / 0 / 0 / 0 |
  | ramp approach (29.15-29.25 s) | 26.77% / 2 / 0 / 0 | 26.51% / 6 / 0 / 0 |
  | climb (30.0-30.5 s) | **28.84% / 11 / 1 / 1** | 27.53% / 11 / 0 / 0 |

  So from the human's CLIMB states the incumbent already reaches the tunnel lane once in 11; the
  missing piece is the strong turn-up on the ramp (vz +1,200 for the human, +480..+680 for both
  policies). Arm 1's cavity practice made rising off the ramp more frequent (6/11 vs 2/11).
* **NEW `--respawn-novel-band B`** (commit c45ebfe): novel rows only within B x d0 of the ceiling in
  potential (d_best <= d <= d_best + B d0); 0 = off. With B = 0.05 the rows' progress median is
  26.2% and ~85% land in the cavity. In-sampler the band cut the yield to ~200 rows per pool at
  tries 16 (3-4% of episodes, the same cavity dose as arm 1), so:
* **Sampler speed-up** (commit 15716d6): the 27-cell visit lookup ran on every candidate (0.48 of
  0.79 s per call); now only on those that passed the cheap gates. Byte-identical rows, stats and
  log line versus the original respawn.py (3 iterations x 2 configs on the incumbent's state);
  tries 64 is now ~0.3 s per call.
* r2b2_band (band 0.05, tries 16) ran 44M steps (evals 25.8 / 27.4) and was replaced by
  **r2b2_band64** (14:58 UTC): resume of arm 1's final (5,551,161,344, md5 d38eb2f7...) +
  `--respawn-novel-band 0.05 --respawn-novel-tries 64` (shell 0.9 / width 0.05 / radius 256-3,072 /
  turn 90 restored), 450M steps. First readings: 530-730 rows per pool, 10-16% of episodes novel,
  progress median 26.2%: ~3x arm 1's cavity dose.

### 15:40 UTC: arm 2b (r2b2_band64) - the cavity return leg is LEARNED (bench); extended

* r2b2_band64 = arm 1's final (5,551M) + `--respawn-novel-band 0.05 --respawn-novel-tries 64`, to
  6,001.5M (avg 217k steps/s). Novel rows 530-1,840 per pool, 10-46% of episodes (rising as the
  sampler found more cells), progress median 26.0%: most of them in the cavity.
* Evals: 27.4-28.5, best **28.523** (5,879M), three collapses to ~15% (5,652M, 5,702M, 5,727M: the
  greedy line breaks at 13-15%, recovered by the next eval). Training ceiling 28,184 -> 28,187 u.
* The greedy line CHANGED: at 5,853M three episodes go through the -y window and slide north along
  the block face toward the shaft's centre (the human's move at 28.0 s), then fall to the shaft
  floor (*21) ~160 u west of the lower opening instead of turning east into it. The incumbent dove
  into the tunnel lane instead.
* Cavity bench, arm 2b's final vs before (MEASUREMENT ONLY, demo states):
  * ramp approach: rose 10/11 (incumbent 2, arm 1 6), vz max 845, median life 2.60 s (0.55 / 1.02);
  * climb: **back 7/11, tunnel lane 6/11, west of the tunnel 5/11, best 29.92%** (incumbent 1/1/0,
    arm 1 0/0/0): from the human's climb states the policy now flies the return leg into the room;
  * shaft and lower opening: still 0 (the entry into the cavity and the drop to the ramp).
* So cavity practice from frontier-band novel rows teaches the dip's return leg in ~450M steps.
  What is still missing is the entry and the strong turn-up on the ramp (vz +845 vs the human's
  +1,190). **Extended** (PROTOCOL: a gate the incumbent never cleared, measured by the bench):
  r2b2_band64x = bare resume of arm 2b's final (6,001,524,736, md5 28e03931...), all novel settings
  restored (band 0.05, tries 64, turn 90, shell 0.9/0.05, radius 256-3,072), +1.2e9 steps (to
  7.2B, ~95 min), evals every 25M. Watching: the training ceiling (real episodes past 28.82%), the
  greedy line entering the lower opening, and evals past 28.628.

### 16:15 UTC: r2b2_band64x at 6.35B

* Evals 6.0-6.35B: best **28.551** (6,027M); 3 consecutive collapses to ~15% (6,102-6,153M). The
  collapse is a ramp at x ~6,800 (12.75-13.25 s): the agent turns UP so hard its horizontal speed
  drops to ~0 and it stalls, where good evals keep 1,250-1,700 u/s through it. Plausibly
  interference from the strong ramp turn-ups the cavity practice trains. Training ceiling flat.
* The greedy line is closing on the cavity entry: at 6,354M episode 4 slides north along the block
  face from the -y window and hits the thin wall at (2,512, 83, 10,715) - inside the lower
  opening's y range, ~55 u below its bottom (ducked origin z >= ~10,770). Episodes 2 and 6 slide on
  past the centre at z ~10,800.
* Bench at 6.35B (ckpt_latest): ramp approach rose 9/11, **vz max +1,180** (6.0B: +845, the
  human's +1,190), median life 3.66 s, best 27.43%; climb back/lane/west 3/3/3 (6.0B 7/6/5, 11
  episodes each, noisy). The shaft and lower-opening sets give identical numbers for every
  checkpoint (death in 0.6-0.9 s whatever the action: those demo states need an immediate turn), so
  they are not informative; the greedy line is the entry's measure.

### 16:40 UTC: r2b2_band64x stopped at 6.73B; r2b2_t1 (T cap 1) launched

* band64x 6.0-6.73B: 29 evals, best **28.551** (6,027M); 5 collapses to ~15% (the x ~6,800 ramp
  stall) and one at 25.1%. Training ceiling 28,187 -> 28,205 u: no real episode entered the cavity.
* The greedy line DRIFTS: at 6,455M 8 of 9 episodes pass the window into the shaft (best 28.54%,
  several end on the shaft floor at x 2,512, y -337..-541, i.e. near the lower opening's wall); at
  6,631-6,706M 7-8 of 9 die in the CP2 pit again at ~24.5 s (27.3%).
* Diagnosis so far: the cavity's return leg is learned from novel rows, the ramp turn-up is nearly
  the human's, but the remaining pieces (ramp -> climb, the entry through the lower opening) are
  precision skills, and the unstuck T has been pinned at its cap 2 since ~2.4B (the eval reach has
  not improved by 500 u). At T 2 the keys are sampled from logits/3 and the yaw sigma is x3.
* **r2b2_t1** (16:38): bare resume of band64x's ckpt_latest (6,725,042,176, md5 0f30157f...) with
  `--unstuck-max 1` (T 2 -> 1; everything else restored, incl. the novel band settings), +1.5e9
  steps (to 8.2B, ~2 h), evals every 25M. Box extended to 19:37 UTC.

### 17:10 UTC: r2b2_t1 (T cap 1) - 28.841 and the first real episode past the tunnel

* T dropped 2 -> 1 at the resume (6,725M); approx_kl 0.26-0.37 for the first updates, ~0.1 after.
* **Eval 6,751M: score 28.841** (best 28.84%, median 28.2%): above the incumbent's 28.628. Episode 8
  goes through the -y window, turns in the shaft and dives into the -y tunnel heading west at
  1,156 u/s, dying on *113 at its far west end (1,966, -904, 10,981) at 28.24 s - the field route's
  physical limit (~28.8%), reached for the first time.
* **Training ceiling 28,205 -> 29,027 u (29.69%) at ~6,751M**: a REAL episode (map start or
  reservoir) got past the tunnel, the first ever on this lineage. Flat since.
* But the greedy line then settled into the CP2 pit: 13 consecutive evals 6,776-7,078M at 27.4-27.5%
  (death at ~24.5 s on the -y side, x ~3,100), then one collapse to 14.8% (7,103M). Novel share
  40-48% of episodes.
* Checkpoint pulled: runs/r2b2/r2b2_t1/ckpt_7128219648.pt (md5 7686c36f...), a stable copy taken
  on the box (box.py pull --ckpt failed: ckpt_latest is rewritten every 25M steps, faster than an
  80 MB copy from Argentina).
* r2b2_t1 stopped at 7,176M (19 evals: 28.841 once, then 27.4-27.5 x13, one 14.8). Its end
  checkpoint is local: runs/r2b2/r2b2_t1/ckpt_latest.pt (7,176,454,144, md5 92b7ddd2...).
* **r2b2_t1f3** (17:12): resume of that checkpoint + `--respawn-novel-frac 0.3` (novel rows capped at
  30% of the non-start pool: share ~25% instead of 40-48%), T cap 1 and the band settings restored,
  +1.5e9 steps (to 8.68B). Purpose: give the start-anchored line more practice so the greedy line
  crosses the CP2 pit again, while the cavity practice continues at ~60% of the dose.

### 17:45 UTC: r2b2_t1f3 at 7.57B

* Evals 7.18-7.55B (16): 24.2-28.0, mostly 27.4-27.5. The greedy line has lost round 1's
  corridor-east detour: it U-turns at ~23.5 s and keeps diving west into the CP2 pit (dies at
  ~24.5 s, x ~3,076, -y side), where the incumbent turns east again at ~24.4 s, drops under *45 and
  uses the +y ramp. Lowering the novel share to ~25% did not bring it back in ~400M steps.
* Training ceiling 29,027 -> 29,098 u (29.76%): stochastic real episodes keep getting past the tunnel
  now and then. Under T 1 (t1 + t1f3, ~850M steps) the ceiling moved +893 u; under T 2 (band64x,
  730M) +18 u.
* Reading: the remaining path to a score above the field-route limit (~28.84) is the cavity route in
  the GREEDY line, and it bootstraps through the reservoir (a real episode entering the cavity is
  harvested >= 2 s before its end, i.e. inside the cavity, and becomes a real spawn there). Kept
  running unchanged.

### 18:10 UTC: r2b2_t1f3 stopped at 7.95B; r2b2_t05 (T cap 0.5)

* t1f3 7.18-7.95B (31 evals): 27.3-28.4 (one 28.367), the greedy line dying in the CP2 pit in
  almost every eval; training ceiling flat at 29,098 u since 7.27B (~670M steps).
* The only change that moved anything recently was lowering the unstuck temperature cap: 2 -> 1
  gave the 28.841 eval within 25M steps and +893 u of ceiling. So one more step:
  **r2b2_t05** (18:10): resume of t1f3's ckpt_latest (7,949,778,944, md5 3da04170...) +
  `--unstuck-max 0.5`; novel frac 0.3 and the band settings restored; +1.5e9 steps; evals every 25M.
  Box extended to ~21:10 UTC.

### 18:35 UTC: r2b2_t05 null; restart from the 6.354B checkpoint (r2b2_r635)

* r2b2_t05 (T cap 0.5, 7.95-8.26B): 13 evals, all 27.3-27.5% (greedy line in the CP2 pit), training
  ceiling flat at 29,098 u. Lowering T further did not free the line; no transient this time.
* Since ~6.6B every checkpoint of this lineage (band64x -> t1 -> t1f3 -> t05) has a greedy line
  locked into the pit dive, whatever T (2, 1, 0.5) and novel share (45%, 25%).
* Restarted from the last checkpoint taken BEFORE the drift: band64x at 6,354,370,560 (the one the
  16:05 bench measured: ramp vz max +1,180; its eval had 7/9 past the pit and one episode hitting the
  thin wall 55 u below the lower opening). **r2b2_r635** (18:35) = that checkpoint (md5 16fe0cd0...,
  T 2 at save) + `--unstuck-max 1 --respawn-novel-frac 0.3` (the T 2 -> 1 switch that produced the
  28.841 eval and the ceiling jump, and the lower novel share), +1e9 steps, evals every 25M.

### 19:00 UTC: r2b2_r635 null; from-scratch test of the band recipe (r2b2_sband)

* r2b2_r635 (6.354B + T cap 1 + novel frac 0.3, to 6.66B): first eval 28.374 (the 6.354B policy
  itself), then 11 of 12 evals at 27.3-27.9% - under T <= 1 the greedy line goes to the pit dive
  within 25M steps from this checkpoint too (the 28.841 transient was not reproduced).
* Conclusion for the warm lineage this round: since ~6.6B (and at once under T <= 1) the greedy line
  commits to the CP2 pit dive while the stochastic policy keeps crossing (training ceiling 29.76%).
  Max over evals so far: 28.841 (r2b2_t1, the field route's tunnel dive at its limit).
* **r2b2_sband** (19:02): FROM SCRATCH, the incumbent recipe (ratchet + novel) +
  `--respawn-novel-shell 0.9 --respawn-novel-shell-width 0.05 --respawn-novel-radius 256 3072
  --respawn-novel-turn 90 --respawn-novel-band 0.05 --respawn-novel-tries 64`, 450M steps, evals
  every 75M (round 1's marks). Question: do frontier-band novel rows clear the early gates sooner
  than ratnov (first-ramp trap exit 378M, window 576M, CP2 wall 651M)? One seed.

### 19:30 UTC: from scratch, the band recipe leaves the first-ramp trap ~75M steps sooner (one seed)

* r2b2_sband (0 -> 450M, ~250k steps/s): best-episode progress per greedy eval at round 1's marks:

  | step | r2b2_sband (ratchet + novel + band/turn/shell/radius/tries) | ratnov (ratchet + novel) | novel only |
  |---|---|---|---|
  | 76M | 4.3% | - | 4.4% |
  | 151M | 4.6% | - | 4.6% |
  | 227M | 6.5% | ~4.6% | 4.6% |
  | 302M | **12.2%** | 5.8% | 10.1% |
  | 378M | 15.9% | 15.2% | 15.8% |

  Training ceiling 18,183 u (18.6%) at 415M. The novel rows' progress median follows the frontier
  (1.1k u at 20M -> 13.8k u at 415M); 1,843 rows per pool (the full request) early on.
* Extended (cleared the trap sooner): **r2b2_sbandx** = bare resume of sband's final (450,363,392,
  md5 ad134ffb...) to 1.05B, evals every 75M. Next gates: the window (ratnov passed at 576M; novel-only
  stalled there) and the CP2 wall (ratnov 651M).

### 19:56 UTC: ...but the band recipe STALLS at the window from scratch; is the band the blocker?

* r2b2_sbandx (450M -> 885M): every eval 20.8% (best) / 18.9-20.8% (median) from 451M to 828M; the
  training ceiling 20,718 u (21.2%) flat since 493M; T rising (0 -> 1.17). ratnov (ratchet +
  default novel) was at 20.8% at 453M and PASSED the window at 576M; round 1's novel-only (no
  ratchet) stalled here from 500M to 1.02B. So with the frontier band the window behaves like it
  did without the ratchet. One seed each. Stopped (behind ratnov at matched steps).
* Hypothesis: the window is reached by cutting back into the A-frame ramp before it ends - a decision
  seconds BEFORE the frontier - and the default sampler's flattened half (rows spread over the whole
  map) is what showed it; the band keeps rows within 5% of d0 of the frontier only.
* Test, no code: **r2b2_sdef** (19:56) = sbandx's checkpoint (884,998,144, md5 f18d4c98...) resumed
  with the DEFAULT sampler (`--respawn-novel-band 0 --respawn-novel-turn 0 --respawn-novel-shell 0.5
  --respawn-novel-shell-width 0.25 --respawn-novel-radius 256 2048 --respawn-novel-tries 8`), 450M
  steps. If it passes the window soon, the band is the blocker at this kind of gate.

### 20:25 UTC: the default sampler unblocks the window at once (confounded by T); band again at the CP2 wall

* **r2b2_sdef** (885M -> 1,335M, default sampler): training ceiling 20,718 -> 23,971 u at ~990M
  (~105M steps after the switch), then 26,233 (1,053M), 27,116 (1,116M), 27,318 (1,284M). Greedy
  evals: 20.8% (885M), 20.9% (961M), **25.0%** (1,037M, CP1), 26.0% (1,112M), **27.3%** (1,188M, the
  CP2 wall), 27.4% (1,263M).
* Confound: the unstuck T was rising when the switch happened (1.17 -> 1.90 at 1,032M, then halving
  on the new bests), and earlier window passages coincided with T > 1.3. So "the band blocks gates
  whose decision lies before the frontier" is likely, not proven.
* Combined scratch lineage so far (one seed): trap exit ~230-300M (ratnov 302-378M), window ~990M
  (ratnov 576M; 435M of it stalled under the band), CP2 wall ~1.19B (ratnov 651M).
* **r2b2_sband2** (20:24): sdef's final (1,335,361,536, md5 25a0396d...) resumed WITH the band again
  (`--respawn-novel-band 0.05 --respawn-novel-turn 90 --respawn-novel-shell 0.9
  --respawn-novel-shell-width 0.05 --respawn-novel-radius 256 3072 --respawn-novel-tries 64`), 2e9
  steps. Question: does the band cross the CP2 pit (a dip just behind the frontier, like the cavity)
  faster than the default sampler did in round 1 (ratnov: ~1B steps at the wall, 651M -> 1,633M)?
* 20:55 UTC, r2b2_sband2 at 1.79B (~460M steps of band at the CP2 wall, ~600M since reaching it):
  evals 27.2-27.4% (one 24.6%), training ceiling flat at 27,318 u (27.94%), T back at its cap 2.
  No pit crossing yet - not faster than round 1's default sampler so far (ratnov: ~1B steps at the
  wall). Runs to ~3.3B (~22:40 UTC).
* 21:22 UTC, r2b2_sband2 at 2.27B: the band lineage CROSSED THE CP2 PIT. Eval 1,940M: 28.228 (best
  28.2%, median 28.0% - most greedy episodes past the pit); training ceiling 27,318 -> 27,998 u
  (28.64%). That is ~750M steps after reaching the wall (~1.19B) vs ~980M for round 1's default
  sampler (ratnov, 651M -> 1,633M): consistent with the band helping dip-behind-the-frontier gates,
  one seed. The following evals (2.0-2.24B) fell back to 27.2-27.4%; T 1.14 and rising.

### 22:40 UTC: from-scratch band lineage reaches 28.71%; per-run scores

* r2b2_sband2 (1,335M -> 3,335M, band on): evals after the 1,940M pit crossing: 2,770M **28.555**,
  3,148M **28.714** (the tunnel dive), 3,299M 28.507; others 24.7-27.7 (one 10.2 collapse). Training
  ceiling 28,140 u (28.78%) by 3.19B. Final checkpoint local: runs/r2b2/r2b2_sband2/ckpt_final.pt
  (3,335,520,256, md5 36256794...). Its greedy line crosses the CP2 pit (healthy), unlike the
  warm lineage's after ~6.6B.
* Whole scratch lineage vs round 1's ratnov (one seed each): trap exit ~230-300M (ratnov
  302-378M), window ~990M (576M; 435M lost stalled under the band), CP2 wall ~1.19B (651M), pit
  crossing ~1.94B (1,633M; ~750M at the wall vs ~980M), tunnel-dive 28.71% at 3.15B (ratnov 28.55%
  at 1.63B, 28.63% at 4.92B).
* Best eval per run (score.py --best): nfturn 28.308, band 27.385, band64 28.523, band64x 28.551,
  **t1 28.841**, t1f3 28.367, t05 27.538, r635 28.374, sband 15.909, sbandx 20.838, sdef 27.352,
  sband2 28.714.
* Last run, **r2b2_sband3** (22:37): bare resume of sband2's final, evals every 25M, until wrap-up -
  more greedy evals from the healthy scratch lineage (an eval above 28.841 would also make its
  checkpoint the hand-on one).
* 23:41 UTC: r2b2_sband3 stopped at ~4.12B (32 evals, best 28.753 = the 3,336M resume eval; then
  28.55-28.58 x3, the greedy line drifting into the pit (27.2-27.5) and back (28.2-28.58 at the
  end)). Box 56901034 released at 23:41 (~9.7 h, ~$3.9).

## ROUND 2 LESSON (r2b2) - written 2026-10-20 23:55 UTC

**Result:** best greedy eval from the true start **28.841** (r2b2_t1 at 6,751M; incumbent 28.628). 0 finishes.
The best episode dives through the -y tunnel to its far west end (1,966, -904, 10,981): the physical
limit of the field's route (~28.8%). Hand-on checkpoint: runs/r2b2/r2b2_t1/ckpt_latest.pt
(7,176,454,144, md5 92b7ddd2...) - its GREEDY line dies in the CP2 pit (27.4%) although stochastic
training episodes reach 29.76%; read its own first eval before comparing.

**What the wall at 28.6% is** (analysis only; see the 13:25 / 13:45 entries):
* The field's route past the CP2 slot (upper window x 2,544 -> 320 u shaft -> tunnel under a block,
  floored by kill plates *17 / *113) is physically CLOSED: for any entry vz and any non-increasing
  horizontal speed the window -> tunnel transition misses by >= 28 u (ducked hull); 12,090
  brute-force straight flights top out at 28.79%; ~280k random / Go-Explore rollouts found nothing.
* The human (WR demo 47250, analysis only) dips ~2,000 u into a CAVITY under the pit slab through a
  LOWER opening (y ~0, z ~10,770-11,100), surfs a ramp at its bottom (normal (-0.759, 0, 0.651),
  x ~1,950-2,150, z ~9,790-10,050: arrive moving south ~2,000 u/s, strafe up it), climbs back out
  through the lower opening and enters the tunnel LEVEL (31.5 s), then the room, under *20, CP2 at
  ~35.1 s. Same gate type as the CP2 pit: a dip the field calls worse plus a ramp turnaround.

**What worked:**
* NEW generic novel-sampler options (default off, byte-identical when off, 10 unit tests):
  `--respawn-novel-band B` (novel rows only within B x d0 of the ceiling in potential) and
  `--respawn-novel-turn T` (heading + view yaw turned by U(-T, T)); plus a byte-identical sampler
  speed-up (visit lookup only on gate survivors, 0.8 -> 0.2 s per call), which makes tries 64
  affordable. With band 0.05 ~85% of novel rows land in the cavity (defaults: ~4%).
* Bench (demo states, measurement only): after ~450M steps of band practice the policy flies the
  cavity's return leg from the human's climb states into the room 5/11 (best 29.92%; incumbent 0/11)
  and rises off the ramp 9-10/11 (incumbent 2/11), vz +1,180 (human +1,190).
* Lowering the unstuck T cap 2 -> 1 on that checkpoint gave, within 25M steps, the 28.841 eval and
  the first real episodes past the tunnel (training ceiling 28,205 -> 29,027 -> 29,098 u).
* From scratch the band recipe left the first-ramp trap ~75M sooner than round 1 (12.2% vs 5.8% at
  302M) and crossed the CP2 pit after ~750M steps at the wall (ratnov ~980M).

**What did not work:**
* Every warm checkpoint after ~6.6B has a greedy line locked into the CP2 pit dive (it drops round
  1's corridor-east detour), under T 2, 1 and 0.5 and novel shares 45% / 25%; restarting from the
  pre-drift 6.354B checkpoint with T 1 relocked within 25M steps. The stochastic policy keeps
  crossing - a greedy-vs-stochastic gap, likely a mean between two modes of the yaw Gaussian.
* From scratch the band STALLED at the window (20.8%, 435M steps) where round 1 passed at 576M; the
  default sampler got through within ~105M steps (confounded by T rising). The window's decision lies
  seconds BEFORE the frontier, outside the band. Net from scratch: tunnel dive 28.71% at 3.15B vs
  ratnov 28.55% at 1.63B.
* Frontier-focused shell/radius without the band (arm 1): rows at a median 17% of progress (the map
  folds: the air near the CP2 sources is earlier track); null.

**For the next round:**
* The way on is known (the cavity dip); the band teaches its pieces. Use the band for dips just
  behind the frontier, NOT as a global default (it blocks gates whose decision is earlier); a generic
  blend (band on the deep-shell rows only, the flattened half unbanded) is the untested middle.
* The greedy line is the bottleneck: training reaches 29.76% stochastically, evals sit at 27.4%.
  Something that consolidates the stochastic successes into the mean (self-imitation with the
  TEMPERED log-prob - the shipped --sil-coef scores T > 0 samples untempered and would inflate sigma)
  or a policy that can hold two yaw modes is the next lever.
* Diagnostics worth reusing: the cavity bench (record_ckpt --spawn-states from demo states, measurement
  only; its shaft / lower-opening sets die whatever the action - only ramp / climb inform), the
  free-flight bound, the hull-trace / flood-fill passage checks.

**Ops:** no box passed box.py's filters for an hour (12:55-13:54); a background `box.py rent` retry
loop got box 56901034 (RTX 4090 + 5950X, $0.401/h, ~210-300k steps/s), released 23:41 (~9.7 h,
~$3.9). gpu_health needed an RTX 4090 reference (harness issue; branch-local fix). `box.py pull
--ckpt` fails on a running run (ckpt_latest is rewritten faster than an 80 MB copy): copy it to a
stable ckpt_<step>.pt on the box first. Never `pkill -f` a pattern that appears in your own command
line (it killed my local shell twice).

## Wrap-up (r2b2) - 2026-10-20 23:55 UTC

* Final: RESULT.json `"final": true`, score **28.841** (r2b2_t1, eval traj_6750732288, re-scored at
  wrap-up: 9 episodes, best 28.84%, median 28.21%, 0 finishes, CP1, 18/19 valid evals in the run).
* Hand-on checkpoint: runs/r2b2/r2b2_t1/ckpt_latest.pt (7,176,454,144, md5 92b7ddd202e0b09629f93e4ac2afdbe1),
  pulled while the run was stopped and md5-checked again at wrap-up; also local:
  r2b2_t1/ckpt_7128219648.pt and r2b2_sband2/ckpt_final.pt (from-scratch lineage, 3,335,520,256).
* Box 56901034 released at 23:41 UTC; no box at wrap-up.

## Round 3, branch r3b2 - 2026-10-21 00:53 UTC: the time penalty makes every dip worth less than a death

### The arithmetic (no run needed)

* Reward units: shaping 100/d0, so 1% of progress = 1 unit; `--time-pen 0.005` per tick = **0.5 units
  per second**; `--surf-bonus 0.3` = 0.3/s while supported; novelty ~1.3 per episode (the incumbent's
  step line). At the incumbent's end an episode averages rew ~5 = shaping ~+17, time ~-16, surf ~+2.6,
  int ~+1.3: the time penalty is the second-largest term.
* Death ends the time penalty (terminal value 0, no death charge). So from any state, flying on is
  worth more than dying only if it earns more than 0.5% of d0 per second it lasts.
* **CP2 pit (round 1's gate):** the pit dive dies at ~24.5 s with 27.4%; the east detour (round 1's
  1,633M line) dies at ~29 s with 27.98-28.55%. +0.6..1.2 units of progress for ~4 s = -2.0 units of
  time: **the detour is worth ~1 unit LESS than the pit dive** under this reward, even before its risk.
  The warm lineage's greedy line went back to the pit dive (r2b2: every checkpoint after ~6.6B, under
  T 2/1/0.5); the classifier below shows it switching within 25M steps (6,751M: 9/9 detour; 6,776M:
  9/9 pit dive).
* **Cavity (the way past 28.8%):** window -> cavity -> ramp -> back through the lower opening -> tunnel
  -> room is ~4 s at zero progress under the ratchet (-2.0 units) for +2.2% to CP2 (~31%). Worth less
  than the tunnel dive (+0.4 in 0.5 s) unless the agent flies well past CP2. Same sign as the pit.
* Without the time penalty (and without the surf bonus, which would then be farmable - the trainer
  refuses a bonus at or above the penalty), every per-step reward under the ratchet is >= 0, so
  surviving is never worse than dying, and both detours beat their quick deaths whenever they reach
  further. Urgency is left to gamma (0.9995/tick: a reward 1 s later is worth 0.95).
* Prior evidence (ledger): xTP0 (`--time-pen 0`, cannonball, from scratch) lost early pace (0.6x of its
  control, peak speed 1,780 vs 2,340 u/s). Never tested warm at a dip gate, which is where the
  penalty bites.

### Tool: `autoresearch/analysis/sid_eval_route.py` (measurement only)

Tags each greedy eval episode with the route stages it reached after t = 22 s: E (east under plate
*45: the detour), S (slot past the pit floor), W (shaft), C (cavity), R (rising in the cavity),
T (tunnel), M (room). Incumbent evals: 6,751M = 9/9 E, 5 W, 1 C, 1 T (the 28.84% tunnel dive);
6,776M = 0/9 E (all pit dives at 24.3 s).

### The arithmetic on the incumbent's own greedy episodes (01:05 UTC, local CPU)

Ratchet shaping recomputed from the goal field along each greedy episode from t = 22 s (the eval
reward column is not recorded), discounted at 0.9995/tick, against 0.5/s of time penalty:

| eval (r2b2_t1) | route | ratchet shaping from 22 s (disc.) | time from 22 s | net with the penalty |
|---|---|---|---|---|
| 6,751M (9/9 detour) | E + slot/shaft, 27.8-28.8% | +2.7..+3.7 | 5.5-6.9 s = -2.8..-3.5 | **-0.1..+0.6** |
| 6,776M (9/9 pit dive) | dive, 27.3-27.4% | +2.5..+2.6 | 2.3-2.5 s = -1.2 | **+1.3..+1.4** |

So under the reward the incumbent trained on, its pit dive is worth ~1 unit MORE than the detour it
had flown 25M steps earlier: the switch was the optimiser doing its job. Without the time penalty the
detour is worth +0.2..+1.2 more than the dive.

### `--sil-temper` (new, default off)

* SIL scored its replayed rows untempered; at T = 1 that is a mean push of -3 per row on d(loss)/d(log_std),
  i.e. sigma inflation on every SIL minibatch. `--sil-temper 1` scores the rows with the same temperature
  tensors PPO's ratio uses. docs/sil.md; tests/python/test_sil_temper.py (math at T = 0.5/1/2, sidistic
  CPU smoke + resume, refusal, flag-off bit identity vs the pre-flag trainer). TRAIN_ONLY in record_ckpt.

### Ops: no box (00:54-01:05 UTC)

`box.py rent` refused from 00:54 ("no offer passes (whitelist, caps, blocklist)"); a background loop
retries every 3 min for up to 3 h (session scratchpad rent_loop.sh).

### 01:06 UTC: box 56976711 (RTX 3090 + Threadripper PRO 5955WX, 32 thr, $0.187/h, Alabama); arm r3b2_tp0

* Rented 01:02 by the retry loop; deployed ar/r3b2 @ 08b1be0 with the incumbent checkpoint
  (r2b2_t1 ckpt_latest, 7,176,454,144, md5 92b7ddd2...). Watchdog deadline 03:02 UTC.
* **r3b2_tp0** (01:05): WARM resume of the incumbent + `--time-pen 0 --surf-bonus 0`, everything else
  restored (ratchet, novel band 0.05 / tries 64 / turn 90 / shell 0.9-0.05 / radius 256-3,072 / frac 0.5,
  unstuck T cap 1 (at the cap: stuck 5.5B steps), int 0.25), 400M steps, evals every 25M. Measures AT
  the wall: does the greedy line go back to the detour (classifier E) and on to the cavity (C/R/T/M)?
  First step lines: rew 5 -> 8-14 per episode (no penalty), len ~790 decisions, T 1.00.

### 01:20 UTC: r3b2_tp0 - the time penalty WAS the CP2 lock: the greedy line flips back in 50M steps

Greedy evals (classifier: E detour, S slot, W shaft, C cavity; best/median %):

| step | +M | score | median | routes (9 episodes) |
|---|---|---|---|---|
| 7,177M | 0 (the incumbent itself) | 27.493 | 27.46 | 9 pit dives at 24.4 s |
| 7,202M | 25 | 27.405 | 27.25 | 6 pit dives, 3 earlier deaths |
| 7,227M | 50 | 28.266 | 27.96 | **7 E**, 1 ESW |
| 7,252M | 75 | 28.241 | 28.00 | 8 E, 3 ESW, **1 ESWC** (cavity entry) |
| 7,277M | 100 | **28.361** | 28.25 | **9 E, 8 ESW** |

* The incumbent lineage at matched steps (r2b2_t1f3, 7.18-7.95B, same checkpoint, time penalty on):
  27.3-28.4 with the greedy line in the pit at almost every eval. Here the detour is back in 50M steps
  and the shaft is reached by 8/9 at 100M. Training: reservoir min-depth 74.69% -> 71.79% (deeper
  harvests), novel ceiling 29,027 -> 29,204 u (new training best, 29.87%), int 1.3 -> 1.9/ep.
* One eval episode lived 255.7 s (26.4%): with no time penalty a greedy episode can loiter; evals never
  stall-kill (CLAUDE.md), training does (15 s).

### 01:40 UTC: r3b2_tp0 at 7.42B - the tunnel-dive limit, then greedy cavity entries

| step | score | median | routes |
|---|---|---|---|
| 7,303M | 27.533 | 26.66 | 9 E, dying at 26.5-27.5 (a weak eval) |
| 7,328M | **28.839** | 28.23 | 8 ESW, **1 ESWT = 28.84%** (the tunnel dive at its physical limit; incumbent 28.841) |
| 7,353M | 28.581 | 28.18 | 6 ESW, 3 early deaths |
| 7,378M | 28.812 | 28.21 | 6 ESW, 2 ESWT |
| 7,403M | 28.497 | 28.27 | 5 ESW, **2 ESWC** (cavity entries) |

* The two cavity entries fly the human's move: -y window at ~27.3/28.25 s, slide NORTH along the
  block face (v (221..508, 1,164..1,369, -333..-445)), turn east through the LOWER opening at
  y ~ -120..+16, z 10,822-10,910 (human: (2,705, 122, 10,929) at 28.5 s). Then both turn back west
  falling: one lands on the cavity's platform (x 2,180-2,350, z ~10,050-10,120), loses all speed
  and hovers there 230 s (an eval never stall-kills); the other slides west off it at 600 u/s and
  falls past the ramp region (x 1,955 at z 10,005) to the cavity floor (*90). The human swings NORTH
  (y +800 at 29.0 s) around the platform and meets the ramp moving south at ~2,000 u/s. So the
  entry is learned; the ramp approach is the missing piece.
* Loitering: with no time penalty 1-2 greedy episodes per eval live 66-300 s (cavity platform,
  first ramp); evals cost ~44 s each (~27% of wall clock at a 25M cadence). Training stall share
  7% -> 15% (the 15 s kill), no other cost seen.
* SIL is NOT the next lever: its one run (uf2DCSIL, ledger 2026-09-14) "consolidates the entry it
  already had" - here that would be the platform crawl. Kept built (`--sil-temper`) but parked.
* Next: bare continuation r3b2_tp0x (+1.5B), and a stochastic bench from the policy's own greedy
  window/shaft states (measurement: how often does the T 1 policy complete the cavity loop?).

### 01:58 UTC: r3b2_tp0 end (7.58B): the policy now PREFERS the cavity to the tunnel dive

| step | score | routes (9 episodes) |
|---|---|---|
| 7,429M | 28.295 | **5 ESWC**, 3 ESW, 1 E |
| 7,454M | 28.181 | 1 ESW, 3 E, 5 early deaths (a weak eval) |
| 7,479M | 28.332 | 3 ESWC, 4 ESW; two 300 s loiterers near the start (1.6%, 4.7%) |
| 7,504M | 28.316 | 3 ESWC, 4 ESW |
| 7,529M | 28.249 | 2 ESWC, 5 ESW |
| 7,554M | 28.321 | 2 ESWC, 4 ESW |

* No greedy episode rises in the cavity, comes back west or reaches a tunnel (no R/B/T/M after C);
  the tunnel dive (28.8%) has gone from the greedy line, so evals sit at 28.2-28.3. Several cavity
  entries hover on the platform for the full 300 s.
* tp0 = 400M steps, 01:05-01:55 (~50 min with startup and evals). Final checkpoint
  runs/r3b2/r3b2_tp0/ckpt_final.pt (7,576,485,888, md5 5e79bf98...), local.
* **r3b2_tp0x** (01:55): bare resume of it (time_pen 0 / surf_bonus 0 restored, everything else as
  tp0), +1.5e9 steps, evals every 25M. Extended because the arm cleared the CP2 pit gate in 50M steps
  where the incumbent's own continuation (r2b2_t1f3 / t05, 0.9B steps) never re-crossed it, and it
  enters the cavity, a gate no greedy eval of this lineage had entered more than once.

### 02:20 UTC: shaft bench (measurement only) - the loop fails for want of SPEED; arm r3b2_sp2

**Bench** (`autoresearch/analysis/sid_shaft_bench.py`, local CPU, tp0 final 7,576M): a greedy recording
with full states (9/9 reach the shaft, 1 enters the cavity), 324 states cut from 1.0 s before to 0.8 s
after each shaft entry (the policy's OWN greedy line), then 40 stochastic episodes of 8 s each:

| sampling | cavity | rose (vz > 300 under the shaft floor) | back west / tunnel / room / > 28.84% | best vz | entry speed (median) |
|---|---|---|---|---|---|
| T 1 (`--exec-temp 2`, the trainer's) | 17/40 | 3 | **0 / 0 / 0 / 0** | +1,063 | 811 u/s |
| T 0 (untempered) | 12/40 | 1 | 0 / 0 / 0 / 0 | +1,044 | 1,008 u/s |

* Risers hit the ramp's ends (x ~2,130-2,190, y +-700, z ~10,000) and come down on the platform: from z
  ~10,000 the lower opening (ducked origin >= ~10,770) needs vz >= ~1,150 (vz^2 / 2g, g 800). The
  human leaves the ramp at +1,190. Temperature is not the lever (T 0 is no better).
* **Speed is the deficit** (horizontal u/s, agent ep 4 of the 7,403M eval vs the WR demo, analysis
  only): slot 1,693 vs 1,835; through the -y window 1,259 (loses 434 on the frame) vs 1,831 (already
  turning north); sliding north along the block face 1,185-1,148 vs 1,717; lower opening 999 vs 1,718;
  after the reversal in the cavity 865 vs 1,721. r2b2's bench from the demo's own ramp-approach states
  (~1,900 u/s) had the band-trained policy rise to +1,180, so the turn-up exists when the approach is
  fast; the approach is not. Nothing pays for keeping speed into the cavity (a dip under the ratchet)
  until the loop completes, which never happens from a slow entry.
* Reservoir check (tp0 final's saved 20k sample): only 0.9% of states lie past 26.5% progress (slot,
  window, shaft, cavity), 22 on the platform (no dead-end flood); `respawn_binned` is 0 (uniform over
  states), and 16 distance bins would lump 25-31% into one bin, so binning is not a targeted fix.
* tp0x (7,576-7,740M): evals 28.22-28.54 (best **28.542**), 2-3 cavity entries per eval, 0 loops,
  training ceiling flat at 29,204 u; 1-2 greedy episodes per eval hover for 300 s (cavity platform,
  first ramp at 1.7-4.5%). Stopped at 7,740,063,744 (ckpt kept on the box as
  runs/r3b2_tp0x/ckpt_7740063744.pt, md5 23dd3425...).

**r3b2_sp2** (02:18): that checkpoint + `--speed-pot 2` (existing generic term: Phi = K |v_xy| / 1000 paid
as its change per call, so a speed loss at a wall or in a turn is charged when it happens; the user's
skate_laby arm skLABY14, K 10, took 79.70 -> 75.96 s). K 2, not 10: the term is not invariant under
termination (the last speed is kept at death), and at the incumbent's CP2 speeds (pit dive dies at
|v_xy| ~1,450, the detour at ~1,000) K 10 would pay the pit dive ~4.5 more than the detour, re-locking
it; at K 2 the detour still wins by ~0.3-0.9. Question: does the greedy line keep speed into the
cavity (entry speed, ramp vz) and complete the loop? 30 min first.

### 02:45 UTC: r3b2_sp2 (speed-pot 2) at +170M - the window is flown at the human's speed; the turn into the cavity is lost

| step | score | routes |
|---|---|---|
| 7,741M | 27.377 | 7 deaths at ~24.3%, 2 pit dives (the first update under the new term) |
| 7,766-7,816M | 27.96-27.97 | 7-8 E, no shaft |
| 7,841M | 28.402 | **9/9 ESW**: window at |v_xy| 1,786-1,826, min after it 1,637-1,787 (tp0: 1,259 then ~1,000) |
| 7,866M | 28.520 | 2 ESW, 2 ES, 5 earlier deaths |
| 7,892M | 28.267 | 1 ESWC, 1 ES, 7 E |

* In 100M steps the term taught what the shaft bench said was missing: the slot -> window -> slide north
  is now flown at 1,750-1,800 u/s (the human's 1,717-1,835). But the 7,841M episodes slide north
  across the WHOLE shaft (past the lower opening at y ~ 0) and fall onto the shaft floor *21 at
  y +830..+1,003, |v_xy| ~1,750: the slow turn east into the opening was charged as a speed loss and
  dropped; the fast one is not found yet. Cavity entries 0 in 5 of 6 evals (tp0x: 2-5 per eval).
* The as-built term keeps the last speed at death (no refund, unlike the trainer's own speed_equiv
  term, whose comment calls the refund "load-bearing: without it accelerate-and-die farms the speed
  credit"), so a fast floor death is free and a platform landing (losing ~1,700 u/s) costs ~3.4.
* Box extended to 05:39 UTC.

### 02:50 UTC: sp2 verdict (speed-pot 2) - NEGATIVE for the cavity; arm r3b2_gt (greedy-tail spawns)

* sp2, 7,740-8,010M (270M steps): evals 27.4-28.52 (best **28.520**); from 7,917M on 8/9 greedy
  episodes fly slot -> window -> slide north at ~1,750 u/s and die on the shaft floor (28.4%); 1 cavity
  entry in 11 evals. Bench on its 7,909M weights (same method as tp0's): only 2/9 greedy episodes reach
  the shaft, and from their 72 states **1/40** stochastic episodes enters the cavity (tp0: 17/40), median
  life 0.93 s. The term made the speed but priced the cavity out: the slow turn into the opening is a
  charged speed loss, and the fast floor death keeps its potential (free).
* NEW `--speed-pot-charge 1` (c621d1d, default off): Phi = 0 at a failure (the potential still held is
  charged on the fatal call), the refund speed_equiv already makes. With it a failed episode's speed
  term totals -K v0 whatever was flown (unit-tested), so a fast death is no longer free; what is left is
  the discount's (1 - gamma) K |v| residual, i.e. a mild "alive and fast" preference. Not run yet: by the
  arithmetic it makes the floor death and the cavity entry roughly even (the floor death keeps +0.2 of
  ratchet progress), so it does not by itself create a reason to enter the cavity.
* **r3b2_gt** (02:48): tp0x's 7,740,063,744 checkpoint (no speed term) + `--spawn-states
  runs/gt_states_r3b2_gt.npy --spawn-states-frac 0.15`: 15% of every spawn pool from the policy's OWN
  greedy line, the last 3 s (to 0.25 s) before each failed episode's end (tools/greedy_tail_states.py,
  first file from tp0's final: 440 states from 8 episodes, CP2 ramp turnaround -> slot -> window ->
  shaft), refreshed every 20 min from this run's ckpt_latest by
  autoresearch/analysis/greedy_tail_loop.sh (local record_ckpt --dump-states, atomic replace; the trainer
  re-reads on mtime). Why: the reservoir holds only ~0.9% of its states past 26.5%, so the entry
  maneuver is practised by the 10% start episodes and little else; this puts ~13% of all episodes at the
  deployed policy's own failure point. Generic: no map quantity (a tail length, a share, a cadence).

### 03:25 UTC: r3b2_gt - greedy-tail spawns move the greedy line INTO the cavity at speed (7/9), 220 u short

* gt (7,740-7,972M, 230M steps), evals: 28.26, 28.355, 28.273, 28.456, 28.221, 28.185, 27.945,
  **28.762** (7,917M: a 28.76% tunnel dive), then 7,942M **6/9 ESWC** (one rising) and 7,967M **7/9
  ESWC**: the most cavity entries of any greedy eval so far (tp0x 2-5, sp2 0-1).
* Those entries are the human's entry at nearly the human's speed, with NO speed term: through the
  lower opening at |v_xy| 1,530-1,660 (human 1,718; tp0 ~1,000), under the shaft floor at 1,150-1,540,
  onto the ramp's SOUTH end (x 1,940-2,220, y ~ -720, z 9,750-10,070), rising at vz +485..+977; their
  highest point after the ramp is z 9,963-10,547 against the lower opening's ~10,770: the best (7,967M
  ep 4, vz +977, z 10,547) is ~220 u short. Episodes live 31-35 s (3-6 s in the cavity). The last
  missing piece is the turn-up strength (~+1,150 needed; the human +1,190, sweeping the ramp's middle
  from y +484 to -490 where these hit its south end).
* Side effect found and fixed: training crawl share (episodes with mean |v_xy| < 300) 9% -> 25%. Not the
  tail spawns themselves (a local bench of 24 from the file: 1 crawl, lives 0.3-2.8 s) but what they
  feed the reservoir: states past 26.5% progress 179 -> 1,696 (good) of which 31% slow (bad) - agents
  hovering in the lower opening (393 states, x ~2,600, z ~10,810) and on the cavity platform (137),
  harvested again and again until the 15 s stall kill and respawned from. 94% of them are < 300 u/s in
  3-D too, so the existing `--respawn-min-speed` (its help: "the deep bins fill with stalled arrivals
  otherwise") catches them.
* **r3b2_gt2** (03:16): gt's 7,971,799,040 checkpoint (md5 e20344d1...) + the same greedy-tail spawns +
  `--respawn-min-speed 300`; the tail file also drops < 300 u/s states (tools/greedy_tail_states.py
  --min-speed, the same rule); refresh loop restarted for gt2.

### 03:45 UTC: r3b2_gt2 at 8.12B - the greedy line flies the loop's skeleton; the apex misses the opening

* Evals 7,972-8,098M: **7-9 of 9 greedy episodes enter the cavity in every eval** (scores 28.23-28.30:
  the tunnel dive is no longer flown, and a cavity attempt that fails dies at ~28.2%). Training crawl
  17% (gt 25%), stall 18-23%.
* The best attempts (7,972M ep 3/4, 8,098M ep 1) fly the human's loop: opening at 28.2-28.4 s ->
  west under the shaft floor at ~1,320 u/s -> onto the ramp moving SOUTH at 1,909 u/s (the human's
  ~1,900) -> surf up and out east at vz +928..980 -> curve north -> **apex z 10,793 at (2,742, 142)**
  (7,972M ep 3): above the lower opening's ~10,770 and at its y, but ~200 u EAST of it and moving
  north, so it misses and falls along the cavity's north wall. The human reaches the opening at
  (2,524, 20, 10,893) moving WEST and still climbing (+629), which carries it to the +y tunnel's
  height (z ~11,137) by 31.5 s. Missing: a stronger take-off (~+1,190) and turning west at the apex.
* Greedy-tail spawns in the cavity now carry their own ratchet record (the spawn's d), so climbing
  from the ramp exit (26.3%) toward the opening (27.4%) and into the shaft is PAID for them: the dense
  gradient for exactly the missing piece. Kept running.

### 04:00 UTC: r3b2_gt2 at 8.32B - THE TRAINING CEILING MOVES: 29,204 -> 30,571 u (31.27%)

* `novel/ceiling` (the deepest d any training episode reached) was 29,204 u (29.87%) from tp0 (~7.3B)
  to 8.28B, then 29,871 u at 8,281M and **30,571 u (31.27%) at 8,306M**: stochastic training episodes
  now complete the cavity loop and get past the field route's 28.8% limit (the room / under *20 /
  toward CP2 at ~31%). The greedy line has not followed yet: evals 8,123-8,299M are 27.99-28.55 (best
  28.549 at 8,174M), with 6/9 rising cavity attempts at 8,224M and some evals falling back to E-only.
* Box extended to ~07:30 UTC.

### 04:15 UTC: r3b2_gt2 at 8.44B - ramp take-off reaches the human's; apexes now above the opening, misaligned

* Training ceiling 30,571 -> **31,000 u (31.71%)**.
* Greedy risers per eval (vz max off the ramp / best apex): 8,324M 2 (996 / 10,413), 8,350M 5 (1,025 /
  10,492), 8,375M 3 (1,132 / 10,675), 8,400M 6 (1,166 / **10,943** at (2,744, 626)), 8,425M 2
  (**1,266** / 10,868 at (2,843, -1,011), moving WEST at 1,370 u/s). The human leaves the ramp at +1,190.
  So the take-off strength is solved within ~400M steps of greedy-tail practice; the last miss is the
  y-alignment of the apex with the lower opening (|y| < 110 at x 2,512-2,576). No change to the run.

### 04:35 UTC: MILESTONE - the greedy line flies the cavity loop and passes CP2: score 32.384 (incumbent 28.841)

| step | score | median | episodes past the tunnel (T M flags) |
|---|---|---|---|
| 8,476M | **32.123** | 28.50 | 2 (31.97, 32.12%) |
| 8,501M | 28.850 | 28.41 | 1 |
| 8,526M | **32.384** | 28.78 | **5** (32.27-32.38%), CP2 entered |
| 8,551M | 32.383 | 32.26 | 7 |
| 8,576M | 32.323 | 32.26 | 7 |

* The route (8,526M ep 7, the human's in every stage): lower opening WEST at 30.5 s (2,641, 52, 10,825)
  -> up into the -y tunnel level at 31.25-31.5 s (z 11,154-11,164) -> the room (x 1,723 -> 1,436) ->
  surf, drop east of *20 and west under it (z 10,400 -> 9,995) -> the ramp at the void's bottom
  (1,103, 243, 9,743) at 34.25 s, out at vz +1,204 -> west through CP2 (x ~800) at ~35 s -> dies at
  (-482, 76, 10,632) at 37.4 s. All five loop finishers die within (-660..-480, 76..143, z 10,630-10,840):
  the next gate.
* The unstuck T started cooling on the new eval best (best 31,406 u; T 1.00 -> 0.45 by 8,592M); training
  ceiling 32,118 u (32.85%); reservoir min-depth 71.7% -> 67.2%.
* Safety copy: runs/r3b2/r3b2_gt2/ckpt_8599896064.pt (md5 c98339c7...), local; the same file on the box.
* The recipe that did it, from the incumbent checkpoint: `--time-pen 0 --surf-bonus 0` (the objective),
  then 15% of spawns from the policy's own greedy tail (last 3 s before each failure, refreshed every
  20 min; tools/greedy_tail_states.py + autoresearch/analysis/greedy_tail_loop.sh) with
  `--respawn-min-speed 300`; 1.35B steps from 7.18B (tp0 0.4B, tp0x 0.16B, gt 0.23B, gt2 0.55B so far).

### 04:52 UTC: gt2 -> gt3 with the in-trainer `--greedy-tail-frac` (82dda58)

* gt2 (7,972-8,714M) evals after the milestone: 8,601M **32.799** (best; one episode loitered at 32.80%
  for 300 s), 8,627M 32.356 with **9/9** through the loop, 8,652M 32.541, 8,677M 32.550, 8,702M 32.548:
  the greedy line now dies at the next gate just past CP2 (x -480..-660, z ~10,630-10,840, ~37 s).
  T 0.28 by 8,662M (cooling on the new bests), training ceiling 32,143 u. gt2's final checkpoint is local:
  runs/r3b2/r3b2_gt2/ckpt_latest.pt (8,714,190,848, md5 fa2db453...) - the best eval's run.
* NEW `--greedy-tail-frac F` (82dda58, docs/greedy_tail.md, tests/python/test_greedy_tail.py): the same
  mechanism inside the trainer - a record_rollout hook keeps the full states of the trainer's OWN
  greedy eval, and the last 3 s (to 0.25 s) of each failed episode, every 5 ticks, >= 300 u/s, replace F
  of every spawn pool until the next eval (every 25M steps instead of the external loop's 20 min, and
  no dependence on this session). Box redeployed in place (28 s; runs/ kept).
* **r3b2_gt3** (04:48): gt2's final + `--greedy-tail-frac 0.15 --spawn-states ''` (clears the file
  restored from gt2's config); everything else restored (time_pen 0, surf 0, respawn-min-speed 300, T).
  First eval: `gtail 495 rows/9 fails`, mean progress 31.5%. Budget +2e9.

### The gate after CP2 (analysis, 04:56 UTC)

* The loop finishers fly west past CP2 LEVEL at z ~10,835 and ~1,170 u/s and stop dead on a wall at
  x ~ -720 (|v_xy| 1,169 -> 0 at 36.65 s, 8,702M ep 5), then fall and die at z ~10,610 (32.3-32.55%).
* The goal field's steepest descent from that point goes WEST AND DOWN under the wall: (-978, -65,
  10,530) 32.97%, (-1,449, 10,123) 33.59%, (-1,815, 9,821) 34.59%, then back east low at z ~9,520
  (x -1,724 -> +115, 35.0-37.0%). So this gate is "start descending before the wall" - a gradient under
  the ratchet (d falls as the agent drops), not a dip; the greedy tails (34.5-37 s of the line) and the
  novel band (rows within 5% of d0 behind the 32.9% ceiling) both sit on it.

### 05:15 UTC: r3b2_gt3 at 8.92B - the loop is the greedy line's default; the wall after CP2 holds

* 9 evals (8,715-8,916M): 32.32-32.57 (best **32.567**), 7-9/9 episodes through the loop in every eval,
  all dying at the wall after CP2 (~37 s). The in-trainer tail re-cuts 475-495 rows from 9 failed
  episodes at every eval.
* Training ceiling 32,856 u (33.6%): stochastic episodes already get under that wall (the field's descent
  path passes 33.6% at ~(-1,450, 10,120)). The eval frontier has not beaten 32,067 + 500 u for 313M
  steps, so T is warming again (0.28 -> 0.67).

### 05:42 UTC: r3b2_gt3 at 9.14B - the wall after CP2 holds (~700M steps)

* Evals 8,941-9,117M: 32.567, 28.267, 32.258, 28.602, 32.200, 32.547, 32.760, **32.774** (gt3 best; the
  round's best stays gt2's 32.799). Two evals (8,966M, 9,017M) fell back to deaths before the window and
  recovered. T back at its cap 1.00 (stuck 540M); training ceiling flat at 32,856 u (33.6%).
* Why the wall is hard (geometry from the line): off the void-bottom ramp at vz ~+1,200 the agent peaks at
  z ~11,000 near x -330 and flies level; the field's way on is below ~10,600 at x -720..-980. From level
  flight at ~1,100 u/s free fall loses only ~50 u over those 0.35 s, so the descent has to start well
  before the apex (a lower take-off or an early dive): a trajectory choice ~1-2 s upstream of the
  failure, inside the 3 s tail window.

### 06:13 UTC: hover farming after CP2 -> r3b2_gt4 with `--int-move-gate`

* gt3 9.14-9.32B: evals 31.7-32.57; more and more greedy episodes reach 31.1-31.9% at ~36 s, turn back EAST
  and park for the rest of the 300 s on the void-bottom ramp's face (x ~880-990, z ~9,770-9,990, |v_xy|
  ~30 u/s, airborne). Training crawl share rose with T: T 0.1 -> 1.0 took novelty per episode 0.7 ->
  1.0-2.5 and crawl 9-13% -> 17-26%. Reading: with no time penalty, a parked agent that turns its view
  collects count novelty on fresh (cell x view x speed) keys, which the count decay at T > 0 keeps
  refreshing, and that can out-earn the ~0.4 units of progress for trying the wall. (Correlation, not
  proof.)
* **r3b2_gt4** (06:12): gt3's 9,318,694,912 checkpoint (md5 9eabbf6c...) + `--int-move-gate` (existing,
  generic: novelty paid only when the POSITION cell changes, "anti-farming"); greedy tail 0.15, speed floor
  300, time_pen 0, T restored. gt3 ran 8,714-9,319M, best 32.774.

### 06:40 UTC: r3b2_gt4 (move gate) - hover farming gone, new best 32.839

* gt4 9,319-9,521M (10 evals): 31.81, 32.80, 32.57, 32.30, 32.78, 32.44, **32.839** (9,470M, the round's
  best; two episodes at 32.73/32.84% loitering 80-85 s after the wall), 32.34, 32.48. Training crawl share
  24% -> 6%, novelty 1.4 -> 0.49 per episode: the move gate removed the farm. The wall after CP2 holds
  (T at its cap 1, stuck 922M; training ceiling 32,856 u).

### 06:50 UTC: wall bench (measurement) -> gt4 stopped; FROM-SCRATCH test of the round's recipe (r3b2_sfull)

* Bench (gt4 9,533M, local CPU): greedy recording 9/9 through the loop (32.26-32.54%); 495 tail states;
  48 stochastic T 1 episodes from them: **5 get under the wall** (x < -740 below z 10,650) but at only
  ~500-620 u/s west while falling ~-1,000 u/s, and land/die at z ~9,720 (or hover on a ledge at ~10,000)
  within ~1 s; best 32.82%, none past 32.9%. The field route goes on WEST at z 10,100-10,500 to x ~-1,800
  first. So the gate needs a fast, shallow dive under the wall - a new multi-part skill, like the cavity.
* gt4 stopped at 9,599,188,992 (12 evals, best 32.839); its checkpoint, the best eval's run, is local:
  runs/r3b2/r3b2_gt4/ckpt_latest.pt (md5 bc10bd85...) - the hand-on candidate.
* **r3b2_sfull** (06:49, from scratch, 2.5e9 budget, evals every 25M): config.json's incumbent_recipe +
  `--respawn-novel --race-ratchet` (round 1's core, lost from config.json in the round-2 harness restore)
  + this round's `--time-pen 0 --surf-bonus 0 --greedy-tail-frac 0.15 --respawn-min-speed 300
  --int-move-gate`; no band, T cap 2. Matched-step references (best-episode %): ratnov (round 1, ratchet +
  novel, time penalty on): 5.8 / 15.2 / 20.8 at 302 / 378 / 453M, window 576M, CP2 wall 651M, CP2 pit
  crossing 1,633M; r2b2_sband: 12.2 at 302M, window ~990M, pit crossing ~1.94B. Question: does the round's
  recipe cross the gates sooner from scratch (in particular the CP2 pit, where the warm test flipped in
  50M steps once the penalty was gone)?

### 07:23 UTC: r3b2_sfull (from scratch) leaves the first-ramp trap at ~200M - ~175M steps before round 1

| step | r3b2_sfull | ratnov (r1, ratchet + novel) | r2b2_sband (band) |
|---|---|---|---|
| 76M | 4.4% | - | 4.3% |
| 151M | 7.1% | - | 4.6% |
| 177M | 10.4% | - | - |
| 202M | **15.3%** | - | - |
| 227M | 15.5% | ~4.6% | 6.5% |
| 277M | 15.6% | - | - |
| 302M | | 5.8% | 12.2% |
| 378M | | 15.2% | 15.9% |

* Out of the trap with the unstuck T still at 0.00 (round 0's exits all needed T > 1.3; round 1's ratnov
  left at ~T 0 too, at 302-378M). Training: rew ~4.9, ceiling 15,696 u, reservoir min-depth 84.3%.
  Next gates to watch: the window (ratnov 20.8% at 453M, through at 576M), the CP2 wall (651M) and the
  CP2 pit (1,633M).

### 07:50 UTC: r3b2_sfull at 489M - window at 453M, CP2 wall at 479M (round 1: 576M / 651M)

| step | sfull best (median) | ratnov | sband |
|---|---|---|---|
| 302M | 15.7% | 5.8% | 12.2% |
| 353M | 18.6% | - | - |
| 378M | 20.7% (16.1) | 15.2% | 15.9% |
| 403-428M | 20.9-21.0% (the window wall) | - | - |
| **453M** | **22.4%, CP1** (through the window) | 20.8% (window at 576M) | stalled at 20.8% to 885M |
| **479M** | **27.3% (25.5)**: at the CP2 wall | CP2 wall at 651M | ~1.19B |

* Every gate ~125-175M steps before round 1's best scratch run, with the unstuck T still at 0.00 (no
  plateau long enough to warm it). Next: the CP2 pit crossing (ratnov 1,633M; the warm tp0 test flipped
  the detour in 50M once the time penalty was gone).

### 08:18 UTC: r3b2_sfull at 766M - at the CP2 wall since 479M (pit dive, no detour yet)

* 12 evals 479-755M: 27.30-27.42% best, medians 25.5-27.4%: every greedy episode dives into the CP2 pit at
  ~25.5 s (no E flag). T warming (0.44, stuck 287M); training ceiling 27,346 u (27.97%).
* Expected: the warm tp0 test flipped in 50M steps because its policy already had round 1's detour (the
  penalty only made it worse than the dive); from scratch the detour has to be discovered first. ratnov
  needed ~1B steps at this wall (651M -> 1,633M); at that pace sfull would cross at ~1.5B (~10:00 UTC).

### 08:51 UTC: r3b2_sfull FROM SCRATCH crosses the CP2 pit at ~980M and enters the cavity at ~1.03B

| gate (best greedy episode) | r3b2_sfull (this round's recipe) | ratnov (r1, ratchet + novel) | r2b2 scratch (band) |
|---|---|---|---|
| first-ramp trap exit | ~200M (15.3%) | 302-378M | ~230-300M |
| window / CP1 | 453M (22.4%) | 576M | ~990M |
| CP2 wall | 479M (27.3%) | 651M | ~1.19B |
| east detour in the greedy line | 931M | - | - |
| **CP2 pit crossed** | **982M** (28.41%, 5/9 E, 2 to the shaft) | 1,633M | ~1.94B |
| **greedy cavity entries / rising** | **1,032M / 1,032-1,057M** | never | never |
| training ceiling past 28.8% | 29,034 u (29.70%) by 1.08B | - | 28.78% max |

* Evals 931-1,057M: 27.40, 27.35, 28.41, 28.22, 28.23, 28.26; T rose to 0.76 at the wall (stuck ~400M)
  and the crossing came with it, as the old exits did. ~500M steps at the CP2 wall vs ratnov's ~980M.
* So the round's recipe (ratchet + novel + no time penalty / surf bonus + greedy tails + speed floor +
  move gate) is not a warm-only effect: from scratch it reaches the cavity in ~1.0B steps, which the
  lineage's warm checkpoints needed 7.3B for.

### 08:55 UTC: incumbent_recipe updated (config.json, the lineage key)

* New scratch default: the old recipe with `--surf-bonus 0`, plus `--respawn-novel --race-ratchet` (round
  1's core) and this round's `--time-pen 0 --greedy-tail-frac 0.15 --respawn-min-speed 300
  --int-move-gate` - exactly r3b2_sfull's flags (box.py launch prepends it; run_arm.sh's scratch
  `--time-pen 0.005` comes before it and loses). Evidence: r3b2_sfull from scratch, every gate sooner
  than the lineage's best scratch run, CP2 pit at 982M vs 1,633M, cavity entries at 1.03B.
* **Harness issue:** the round-2 harness restore (commit c14ffb1, "the operator's harness at ... over
  7da4860e78") left config.json's incumbent_recipe WITHOUT r1b1's `--respawn-novel --race-ratchet`
  (ca6bb63), although incumbent_recipe is a lineage key that should travel with the winner; round 2's and
  round 3's scratch launches through box.py therefore ran without the lineage's two core mechanisms
  unless the agent passed them by hand. Worth checking that the orchestrator carries lineage_keys over.

### 09:18 UTC: r3b2_sfull FROM SCRATCH flies the cavity loop at 1.26B: 30.244, then 31.901

* Evals 1,082-1,233M: 28.25-28.59, 5-8/9 greedy cavity entries rising off the ramp every eval (the warm
  lineage's state at ~8.3B); **1,259M: 30.244** (median 29.93%: most episodes through the loop, the tunnel
  and the room), **1,284M: 31.901** (one at 31.90%). Training ceiling 31,903 u (32.6%); T 0.54.
* The full chain from nothing: first-ramp trap 200M, window 453M, CP2 wall 479M, CP2 pit 982M, cavity
  entries 1.03B, the cavity loop in the greedy line 1.26B. The lineage got there warm after 7.18B (round
  1-2) + 1.30B (this round) steps; round 1's scratch recipe crossed only the CP2 pit, at 1,633M.

### 09:46 UTC: r3b2_sfull from scratch reaches the warm frontier at 1.5B: best 32.824 (warm best 32.839)

* Evals 1,309-1,536M: 32.28, 31.82, 32.32, 32.47, 32.59, 32.58, 32.44, 32.52, 32.60, **32.824** (1,536M):
  7-9/9 greedy episodes through the loop, the room and CP2 in every eval, dying at the same wall after
  CP2 as the warm runs. Training ceiling 33,000 u (33.75%); T 0.10.
* From nothing to the warm lineage's best frontier in 1.54B steps (the warm line took 7.18B + 1.29B). For
  the next round: the recipe now in config.json reproduces this round's result from scratch on one 3090
  in ~3 h; the warm checkpoint and a scratch one are both at the wall after CP2.
