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

## Round 3, branch r3b3 - 2026-10-21 01:05 UTC: greedy-line spawns (--respawn-greedy)

### The idea

* The score is the GREEDY eval, and the greedy policy (mean yaw, argmax keys) is never executed in
  training: the behaviour policy samples at the unstuck temperature (T 1-2: sigma x2-3, logits/2-3)
  and the reservoir is harvested from those stochastic episodes. So the states the greedy line
  reaches just before it dies can be states training seldom visits with that position AND
  velocity. Round 2's warm lineage is the case: stochastic episodes crossed the CP2 pit (training
  ceiling 29.76%) while ~60 greedy evals over 1.5B steps dived into it (27.4%), whatever T or the
  novel share. DAgger (Ross et al. 2011): train on the states the DEPLOYED policy visits.
* NEW `--respawn-greedy F [--respawn-greedy-window LO HI]` (commit 5baf825,
  `python/surfgym/greedy_spawn.py`): every greedy eval's episodes are captured through the eval's
  on_tick hook (the eval core's pre-step STATE_DTYPE snapshots, what record_ckpt --dump-states
  writes); for each FAILED episode the states from HI to LO s before its last tick are kept
  (default 5.0..0.5 s); each iteration F of the pool's non-start rows (the reservoir TAIL: the
  start rows and the novel rows after them are kept) are replaced by uniform draws from them,
  clocks zeroed. Rebuilt at every eval, so it follows the greedy line to its next death; an eval
  with no failed episode empties it. Episodes from these rows are REAL (harvested, move the novel
  ceiling): the states were reached from the map start by the policy itself.
* Generic: policy-derived states (CLAUDE.md section 0 lists "windows cut from its own greedy
  line" as allowed), two constants in seconds, no map quantity (section 0b).
* Tests (tests/python/test_respawn_greedy.py, 17 pass on CPU): the window per failed episode in
  seconds, done/trunc add nothing, a new eval replaces the pool, the mix touches only the tail,
  checkpoint round trip, refusals, the capture equals record_rollout's trajectory row for row on
  the real core, a sidistic CPU smoke (resume onto a filled reservoir: pool built at every eval,
  mixed, saved, restored by a bare resume, record_ckpt reads the checkpoint), and flag off is
  bit-identical to the pre-change trainer (config, csv, eval bytes, weights).

### Box

* 00:48 UTC: the only offer under the caps (a 5090 at $0.567, New Jersey, machine 149861) never
  reached `running` in 330 s; box.py blocklisted and destroyed it. Since then no offer passes
  (cheapest 4090 $0.455 vs cap 0.45; 3090 $0.535 vs 0.30; 5090 $0.669 vs 0.60). A background
  `box.py rent` retry loop runs every 3 min.

### 01:20 UTC: box 56978070 (RTX 4090 + Ryzen 9 5950X, $0.375/h, machine 29449); arm 1 launched

* Rent loop: a 4090 at $0.429 (Texas, machine 136954) reached `running` but no ssh in 90 s ->
  blocklisted and destroyed by box.py (01:11); box 56978070 READY at 01:16.
* Local check before launch (record_ckpt on the incumbent ckpt, 2 greedy episodes, CPU): one dies
  in the CP2 pit at 24.55 s (27.36%), one STOPS at 17.4% at 17.36 s and crawls to the 300 s cap
  (evals never stall-kill). So the pool also keeps the 5 s before an episode's best-progress tick
  when that lies > 5 s before its end (commit 4b13bb5).
* **r3b3_g15** (01:20): warm resume of the incumbent ckpt (r2b2_t1 end, 7,176,454,144, md5
  92b7ddd2...; T cap 1, band novel rows and everything else restored) + `--respawn-greedy 0.15`,
  400M steps, evals every 25M. It tests the greedy line AT the CP2 pit: the incumbent's own
  continuations (round 2's t1f3 / t05 / r635, ~1.4B steps) kept the greedy line in the pit at
  27.3-27.5% in ~60 evals (one 28.367). First eval (the checkpoint itself, 7,177M): 9/9 fail,
  deaths at 21.5-24.4 s (8 of 9 at the pit) and one at 102.6 s that stopped at 21.9 s; pool 4,101
  states, 553 of the 3,686 non-start rows per pool.

### 01:30 UTC: r3b3_g15 first evals; a probe from the greedy line's own states

* Evals (score.py best / median, deaths): 7,177M 27.495 / 27.5 (the checkpoint itself: 8/9 at the
  pit at 24.3-24.4 s); 7,202M 27.344 / 27.3 (9/9 at the pit); **7,227M 28.380 / 27.8 (8/9 past
  the pit: deaths at 27.3-28.5 s)**; 7,252M back in the pit (deaths 24.1-24.3 s, two at
  19.5/20.2 s). So the greedy line flips modes at the pit; whether greedy-line practice
  consolidates the crossing is what the rest of the 400M decides.
* Probe (local CPU, ANALYSIS ONLY, `record_ckpt --spawn-states --stochastic --exec-temp 2` = the
  training temperature at T 1): from the incumbent ckpt's OWN greedy-line states at 20.0, 21.0,
  22.0, 22.5, 23.0, 23.5 s (episode that dies in the pit at 24.55 s), 30 episodes: **0 cross**
  (best 27.24-27.39%, all fail). So before any training the crossing is not reachable by
  sampling noise from the greedy line itself; the 7,227M flip came from the policy changing at
  those states within ~50M steps.

### 01:40 UTC: r3b3_g15 stopped (null at the pit); arm 2 r3b3_sgir on the scratch band checkpoint

* **r3b3_g15** (incumbent ckpt + `--respawn-greedy 0.15`, 7,176M -> 7,353M, 8 evals): 27.495,
  27.344, **28.380** (7,227M, 8/9 past the pit), 27.414, 27.435, 27.461, 27.411, 27.466 (all 9 in
  the pit at 24.2-24.6 s). The one crossing did not repeat in 125M steps -> stopped. Reading, with
  the probe above: from the greedy line's own pre-death states the policy's sampling (T 1) never
  produces the east turn (0/30), so practising exactly there has no successes to learn from; the
  pit-locked greedy line is a basin, not an under-trained state. Greedy-line spawns need a
  frontier where the stochastic policy sometimes succeeds from the greedy states.
* NEW `--int-real-only 1` (commit 21f9053): the count-based novelty bonus counted EVERY env, so
  the cavity's novelty (85% of banded novel rows land there) was used up by novel-row episodes
  and a real episode finding the way in from the main line earned almost nothing. With the flag
  RaceReward reads the novel sampler's per-env tag as `int_skip`: novel-row episodes are neither
  paid nor counted (entry and dwell). The novel sampler's own rule for its visit grid. 7 tests.
* **r3b3_sgir** (01:37): warm resume of **r2b2_sband2/ckpt_final.pt** (round 2's from-scratch
  band lineage, 3,335,520,256, md5 36256794...; T cap 2, band rows etc. restored) +
  `--respawn-greedy 0.15 --int-real-only 1`, 400M steps, evals every 25M. Why this checkpoint
  (the lineage file r2b2's RESULT.json names as the healthier warm start): its greedy line crosses
  the pit and dives the tunnel, so the greedy-line pool sits at the window / tunnel and the
  novelty pull is tested at the cavity. Control = round 2's r2b2_sband3 (bare resume of the same
  file, ~0.8B steps): best 28.753 (its resume eval), then 28.2-28.58 and pit drifts; training
  ceiling 28.78%.
* Its first eval (the checkpoint itself, 3,336M): **28.846** (best 28.85%, median 28.4%; 7/9 die
  at the window / tunnel at 28.5-29.1 s, the best at the -y tunnel's far west end (1,966, -893,
  10,985), the field route's physical limit). +0.005 over the incumbent: eval noise, not
  progress, but it is this branch's best eval for now.

### 01:45 UTC: arm 2 at 3.43B - one real episode past the tunnel at 3,354M; NEW --sil-tempered

* **r3b3_sgir training ceiling 28,202 -> 29,301 u (29.97%) at ~3,354M, 18M steps after the
  resume** - beyond round 2's best ceiling (29,098 u, warm lineage under T 1). The field's
  d = 68,467 isosurface lies at x ~944-1,552, z ~9,670-10,340: under the room, past *20's end,
  on the way to CP2 (x ~790). So a REAL episode (start, reservoir or greedy row - never a novel
  row) flew the cavity route and beyond. The control (round 2's sband3, bare resume of the same
  checkpoint, ~0.8B steps) never moved the ceiling past 28.78%. One event so far (flat to 3.43B).
* Greedy evals 3,361M / 3,386M / 3,411M: frontier 27,651 / 28,142 / 28,193 u; deaths at the window
  / tunnel 28.1-29.0 s. The unstuck T decays (1.94 -> 1.02: the resume eval was a +500 u best).
  `int` per episode 1.16 -> 0.33 (novel episodes no longer paid). Reservoir min depth 71.25% ->
  71.80% (the deepest harvested state got shallower).
* Why a cavity route is hard to adopt even when it is found (arithmetic, not measured): under the
  ratchet the dip has no shaping charge, but the time penalty (0.5 / s) charges ~1.75 for its
  ~3.5 s against +0.3 for the tunnel dive; the critic values the route only if it has seen
  progress AFTER it, which real episodes almost never reach. The real-only novelty pull (0.25 /
  sqrt(N + 1) per new cell x yaw x speed key, x(1+T) under unstuck_int) is what offsets it.
* NEW `--sil-tempered 1` (commit 9dc81ee): the shipped `--sil-coef` scores T > 0 samples under the
  untempered policy (minimising that NLL inflates sigma toward the tempered spread and flattens the
  keys); with the flag the buffer is scored under the current temp_t / tempv_t (exact for the
  rows of this iteration - a 50k buffer holds about one iteration's R > V rows). 3 tests; flag off
  bit-identical.

### 01:50 UTC: MILESTONE - a greedy eval past the tunnel: 29.823 (r3b3_sgir at 3,436M)

* Eval 3,436,707,840 (100M steps after the resume): **score 29.823** (best 29.82%, median 27.4%).
  Episode 1, against the field route's physical limit (~28.8%):
  * 23.0-26.0 s: the corridor-east detour (U-turn 23.5 s, under *45, +y ramp turnaround at
    (5,576-5,687, -309..157, ~11,000) at 25.5-25.75 s), west climbing, level in the slot at
    27.25 s (3,196, 877, 11,473);
  * 27.75 s: through the +y upper window (2,319, 863, 11,411), v (-1,622, 639, -324);
  * 28.00 s: (2,406, 1,046, 11,305), v (735, -1,128, -524) - redirected EAST by the shaft's wall;
  * 28.25 s: (2,197, 886, 11,149), v (-1,243, -34, -724) - redirected west again, now INSIDE the
    +y tunnel's height range (ducked origin z 10,964-11,180);
  * 28.50 s: (1,890, 844, 10,943) - out of the tunnel's west end (x 1,952) into the room;
  * 29.00 s: (1,812, 316, 10,381) - below *20's level EAST of its end (x 1,772), as the human does
    at 33.75 s; 29.25 s: (1,585, 268, 10,026) flying west under *20;
  * dies at 29.49 s at (1,373, 290, 9,637), vz -1,716 (the human uses a ramp at the void's bottom
    near (1,112, 236, 9,730) at 34.5 s to turn the fall into a climb).
* So the shaft is NOT closed: two wall redirections inside it (~0.5 s) lose ~260 u of height while
  staying in the shaft, which a straight flight cannot (round 2's bound and its 12k brute-force
  flights were straight). Not the human's cavity route, ~3 s faster to the room.
* The other 8 episodes of that eval died in the CP2 pit (27.4%); 3,461M: 28.798 (median 27.4%).
  The greedy line is unstable between the pit dive and the crossing.
* PROTOCOL: a gate the incumbent never cleared -> the arm is extended when its 400M end (3,735M)
  is reached (bare resume restores --respawn-greedy and --int-real-only).
* Caveat on attribution (01:50): the novelty count table is RESTORED from the checkpoint, and in
  round 2 the novel rows counted, so on this warm start the cavity's novelty is already spent:
  `--int-real-only` here mostly stops paying novel-row episodes (int/ep 1.16 -> 0.26), it does not
  re-open the cavity's novelty. The greedy-line pool (15% of spawns at the window / tunnel deaths)
  is the more likely cause; the unstuck T decay (1.94 -> 0.65 after the resume eval's +500 u best)
  is a possible confound, though round 2's sband3 resumed the same file with the same tracker.
  An ablation (greedy spawns alone) would settle it.

### 02:12 UTC: arm 2 end - the greedy line collapsed into the pit; the time penalty is the likely cause; arm 3

* **r3b3_sgir 3.34-3.74B, 16 evals** (score.py): 28.846, 28.283, 28.784, 28.837, **29.823**, 28.798,
  28.823, 28.799, 28.811, 28.792 (medians 28.2-28.6 until 3.41B, then 27.4), then 6 evals all 9/9 in
  the CP2 pit (27.45-27.49) from 3,587M. Training ceiling 29,400 u (30.07%). Final ckpt
  /home/steam/ar/runs/r3b3/r3b3_sgir/ckpt_final.pt (3,735,552,000, md5 8a883449...). The weights
  behind the 29.823 eval (3,436M) were NOT kept (ckpt_latest moves on; --ckpt-every was 1e9):
  arm 3 saves every 1e8.
* Training curves: `dip/max_survived_depth` ~2.2 units (the corridor-east detour is a 2-unit dip)
  until ~3.66B, then 0.46-0.9: the policy stopped flying the detour. The unstuck T decayed 1.94 ->
  0.26 (3.63B) after the resume eval's +500 u best, then rose again; the eval MEDIAN fell into the
  pit at ~3.44B (T ~1.0), the best episode at ~3.58B (T ~0.4). surf share 30% -> 23%, dive 70% -> 77%.
* **The time-penalty arithmetic at this wall** (reward units: 1% of progress = 1, time penalty
  0.005/tick = 0.5/s): the pit dive reaches 27.4% at 24.5 s and dies. The crossing reaches 27.4% only
  at ~27.3 s (the detour is a slow dip: 2.8 s x 0.5 = -1.4), then +1.4 to the tunnel dive's end at
  28.5 s (-0.6 time) -> net **-0.6 vs the pit dive**; crossing + the 29.8% zig-zag: about -0.1. So
  with nothing learned beyond the tunnel, dying early in the pit pays at least as much as crossing:
  the policy is indifferent-to-negative about the crossing, and the greedy line drifts between modes
  (round 2's warm lineage, arm 1, arm 2 alike). The cavity (~3.5 s of dip) costs -1.75 the same way.
  Ledger, cannonball from scratch: `--time-pen 0` had no suicide basin but less urgency (0.6x its
  control); this is the warm regime at a slow detour, a different question.
* **r3b3_tp0** (02:11): resume of r3b3_sgir/ckpt_final.pt (pit-locked) + `--time-pen 0 --surf-bonus 0
  --critic-warmup 20 --ckpt-every 1e8` (everything else restored: greedy spawns 0.15,
  int-real-only, band novel rows, ratchet, unstuck). `--surf-bonus 0` because the trainer refuses a
  surf bonus >= the time penalty (farmable by parking on a ramp; the bonus paid ~0.08/s on
  average). 400M steps. Discriminating: round 2's T / novel-share / restart tweaks never pulled a
  pit-locked greedy line out in ~1.5B steps; if removing the per-second charge does, the charge is
  the lock. Generic: no per-second penalty while nothing finishes (the discount sets the pace),
  one constant for every map.

### 02:26 UTC: arm 3 (time-pen 0) - the greedy line leaves the pit at once and ENTERS THE CAVITY

* r3b3_tp0 evals (score.py best / median): 3,736M 27.421 / 27.3 (the pit-locked checkpoint itself);
  **3,761M 27.958 / 27.0 (7/9 past the pit, 25M steps after the 10.5M-step critic warmup)**;
  3,786M 27.979; 3,811M 26.770 (4 episodes alive at the 300 s cap on the corridor / +y ramp area:
  no time charge, no eval stall rule); 3,836M 28.224 / 28.2; 3,862M 28.246 / 28.2.
* 3,836M, all 9 greedy episodes: pit crossing -> +y window -> best at (2,250-2,270, ~740, ~11,420)
  at 27.8-28.1 s inside the shaft -> down to the thin wall (x 2,512) at shaft-floor height; **3 of
  9 go through the LOWER OPENING into the cavity** and die on its kill floor *90 at (2,596-2,673,
  -77..-103, ~9,567) after 31.2-43.5 s; one ends at the opening's bottom edge (2,511, -113, 10,717).
  The human's route (WR demo, analysis only): window -> shaft -> lower opening -> cavity -> a surf
  ramp at its bottom (x ~1,950-2,150, z ~9,790-10,050) -> back out through the opening rising ->
  tunnel level at 31.5 s. No greedy eval of this lineage had entered the opening before (round 2's
  closest hit the wall 55 u below it).
* Training: `dip/max_survived_depth` back at 2.23 (the detour), p90 0.82 (more dips survived);
  ep_len 600 -> 871 decisions; int/ep 0.3 -> 1.3 (real episodes reach new keys again); T 0.74 ->
  1.37 (no new eval best yet, rising).
* So the per-second charge WAS the lock: with it the slow detour and the cavity dip were worth
  less than an early death; without it the same weights cross the pit within 25M steps and walk
  into the dip. Generic reading: a time penalty with termination and no death cost prices any
  slow stretch below dying; while nothing finishes, the discount alone should set the pace.
* EXTENDED (a gate the incumbent never cleared: the greedy line in the cavity). What is missing:
  the ramp at the cavity's bottom and the rising exit - the pieces round 2's bench showed the band
  practice teaches from the human's states.

### 02:52 UTC: MILESTONE - 30.348, the greedy MEDIAN past 30% by the human's cavity route

* r3b3_tp0 (time-pen 0) evals 3.89-4.04B: 28.22-28.26 (the greedy line in the shaft / cavity,
  deaths at 32-37 s - inside the dip progress is BELOW the tunnel's 28.8%); 4,063M 28.569; 4,088M
  29.562 (one episode out of the tunnel to (1,758, -528, 10,480) at 33.6 s); **4,113M 30.348 (best
  30.35%, median 30.2%, 6/9 past 30%)**.
* 4,113M, the episodes past 30%: they reach the void under the room - best points (880-1,150,
  -178..-577, 9,824-9,999) at 34.7-35.0 s (two alive at the 300 s cap there) - the human's 34.5 s
  position is (1,112, 236, 9,730), where a ramp at the void's bottom turns his fall into a climb to
  CP2 (sgcp_2, x 768-816, z 10,060-11,300) at ~35.1 s. Episode 8 hangs at (880, -587, 9,976), ~70 u
  short of the curtain in x and ~80 u under its bottom.
* Training ceiling 29,400 -> 30,375 u (31.07%) by 4.11B; reservoir min depth 70.33% (and 67,813 u
  = 30.6% at the extension's start); T back at the cap 2; ep_len 1,011 decisions; stall share 28%
  (episodes that stop progressing are killed at 15 s in training).
* So in ~380M steps after the warmup, with the per-second charge gone, the greedy line went from
  pit-locked (27.4%) to the human's whole CP2 sequence: corridor-east detour, +y ramp turnaround,
  slot, window, shaft, lower opening, cavity, back out, tunnel, room, under *20. The band novel
  rows (85% in the cavity), the greedy-line pool (now at the cavity / void deaths) and the
  reservoir all kept training at the frontier.
* Final ckpt /home/steam/ar/runs/r3b3/r3b3_tp0/ckpt_final.pt (4,135,583,744, md5 aae269ee...),
  pulled; snapshots ckpt_3835691008 / 3935830016 / 4035969024 on the box.
* **r3b3_tp0x** (02:52): bare resume of that checkpoint (every flag restored), +1.5e9 steps,
  evals every 25M, snapshots every 1e8. Next gate: the ramp at the void's bottom up to CP2 (~31%),
  then the human's 39% further on.

### 03:20 UTC: r3b3_tp0x - every greedy eval enters CP2; best 32.556; the next wall at x ~-700

* Evals 4,136M-4,463M (score.py): 31.788 (the 4,135M checkpoint itself, CP2), then 32.348, 32.518,
  32.307, 32.550, 32.522, 32.532, 32.324, 32.538, 32.518, 32.518, 32.322, **32.556** (4,438M,
  median 32.4%), 32.512; all CP2, medians 31.2-32.4%. (traj_4488429568 was still being written:
  32.76 on 8 episodes.) Training ceiling 32,141 u (32.9%); reservoir min depth 67.24%; T 0.99.
* The line past the void (4,438M episode 0): 33.75 s (1,089, -263, 9,824) at the void's bottom ->
  34.0-34.5 s the void-bottom ramp turns it up (vz +56 -> +993 -> +1,121; the human's at 34.5 s) ->
  ~34.8 s through CP2 (sgcp_2, x 768-816) climbing -> west at z ~10,900 (35.75-36.25 s) -> **stops
  at x ~-700 at 36.5 s (v ~0 horizontally) and falls to a kill at z ~10,613**. 8 of 9 episodes die
  there (x -671..-711, |y| < 145).

### 03:50 UTC: r3b3_tp0x regressed when the unstuck T climbed past ~1.5; restart at T cap 1 (r3b3_t1)

* r3b3_tp0x evals 4,463-4,815M: 32.512, **32.760** (4,488M, best 32.76%, CP2 - the branch's best),
  32.518, 32.537, 32.546, 32.212, 32.292 (median 28.0), then **27.924, 27.662, 27.245, 24.276,
  27.313, 31.222, 28.238, 28.347** (4,639-4,815M: the greedy line back at the CP2 approach / pit).
* The unstuck T over the run (progress.csv): 1.72 at the resume -> 0.39 (4,350M) after the 4.15-4.20B
  bests, then rising (no +500 u eval best after 31,626 u at ~4.20B: 32.76% was only +200 u) ->
  1.05 (4,500M), 1.30 (4,550M), 1.55 (4,600M), 1.80 (4,650M), 2.00 from 4,700M. The evals were
  32.3-32.8 while T was 0.39-1.3 and collapsed when it passed ~1.5-1.8.
* So with the time penalty gone, LOW T no longer breaks the line (arm 2's collapse at T ~0.4 was the
  time-penalty basin); HIGH T does (the noise destroys the precise CP2 sequence in the training
  distribution). Round 2 also had its best line at T cap 1.
* **r3b3_t1** (03:49): resume of r3b3_tp0x/ckpt_4536139776.pt (md5 378e14a3..., inside the good
  window: evals 32.52-32.55 around it; restored T 1.23) + `--unstuck-max 1 --ckpt-every 1e8`,
  +1.5e9 steps. Everything else restored (time-pen 0, surf-bonus 0, greedy spawns 0.15,
  int-real-only, band novel rows, ratchet). Question: at T <= 1 does the line hold and get past the
  x ~-700 wall? r3b3_tp0x was stopped at ~4,840M (its last snapshot ckpt_4836556800 is on the box).

### 04:15 UTC: r3b3_t1 (T cap 1) holds the CP2 line; the obstacle at x ~-700 is a kill-wrapped block

* r3b3_t1 evals 4,587-4,763M: frontier 31,567-31,815 u (32.3-32.5%) every eval, deaths at 37-38 s
  (one wobble at 4,788M: 31,036, six deaths at 28 s). T held at 1.00. So the cap keeps the line;
  no eval past the wall yet; training ceiling flat at 32,141 u since ~4.48B.
* Geometry (bsp entities, ANALYSIS ONLY): at x -720..-464, |y| <= 128 a block (func_wall *144, z
  10,580-10,764) wrapped in kill teleports - *124 (its volume), *34 / *35 (no_jump) plates on its top
  at z 10,768-10,802 (|y| <= 160), *66 under it; a kill floor *112 at z ~10,578 over x -784..-144,
  |y| <= 736; further west a thin wall *145 (x ~-928, z 10,136-10,288) with a kill plate *104 on top
  (z ~10,320) and a kill plate *43 at z ~10,100 (x -1,388..-944). The greedy line flies west at
  |y| < 145, z ~10,800-10,900, straight into the block's kill wrap.
* The training ceiling's d (65,627) sits past the block: x -850..-1,170, z 9,770-10,500 (lower,
  west). So a real training episode has already found the way around and down; every earlier gate
  of this lineage went ceiling-first, greedy line 100-400M steps later.

### 04:45 UTC: r3b3_t1 holds but does not pass the block; frontier-only bursts (r3b3_ezf)

* r3b3_t1 (T cap 1) 4,536-5,115M, ~23 evals: frontier 30,943-31,827 u (31.6-32.6%) every eval,
  deaths at 37-41 s at the kill-wrapped block; the training ceiling flat at 32,141 u since ~4.48B
  (~640M steps). Stopped at ~5.12B; snapshots ckpt_4636278784 ... ckpt_5036834816 on the box.
* Reading: at T <= 1 the stochastic policy no longer generates the ~200 u sideways / upward offset
  the block needs (one real episode did, at ~4.48B under T ~1); raising T for the whole policy
  breaks the precise earlier line (T > 1.5, above). What is needed is exploration only at the
  frontier.
* **r3b3_ezf** (04:44): resume of r3b3_t1/ckpt_5036834816.pt (md5 e820b0fc...) + round 1's
  `--ez-eps 0.05 --ez-front 0.04` (ez-greedy bursts - a uniform yaw z and random keys held for
  n ~ zeta(2) decisions, cap 60 - that may START only where progress >= (1 - 0.04) x the
  reservoir's deepest progress, i.e. past ~31.3% here = the last ~1 s before the block; burst rows
  are dropped from PPO) + `--ckpt-every 1e8`; everything else restored (T cap 1, time-pen 0,
  greedy spawns, int-real-only, band). 400M steps first. Step line: `ez 1.2-1.5% front>31.3%`;
  approx_kl 0.29-0.46 on the first updates (watch).

### 05:10 UTC: r3b3_ezf null so far; the block, measured with hull traces (analysis only)

* r3b3_ezf 5,037-5,289M (10 evals): frontier 30,655-31,843 u, deaths 37-38 s at the block, training
  ceiling flat at 32,141 u. Bursts 0.9-1.5% of decisions (front > 31.3-31.4%); approx_kl 0.29-0.46
  per update throughout (round 1's ez arms also logged high kl; the dropped burst rows are probably
  still in the kl estimate). The band starts AFTER the void-bottom ramp, and a burst cannot change
  altitude in the air (pitch has no physics effect), so it cannot fix a line that is too low/high.
* Westward hull traces (standing and ducked) at y 0 / +-300: a WALL at x ~-768 (hull centre)
  spanning z 10,600-10,950 at every y; at z 10,500-10,540 a wall at x -128 (the space under the kill
  floor *112 is closed from the east). The greedy line descends to z ~10,773 by x -700 (36.5 s), so
  its feet (z ~10,737) clip the block's top (func_wall *144, top z 10,764) and it dies on the block's
  kill plates (*35 at z 10,800, |y| <= 160) or falls to the kill floor *112 (z 10,578). The field's
  route drops from (-624, 16, 10,856) to (-784, -48, 10,472): down WEST of the block, under the
  wall's lower edge, without touching *112 - a precision drop.
* In the 5,136M snapshot's greedy pool several episodes STOP at 36.1-36.5 s and live on for
  40-220 s (the line sometimes ends somewhere safe at the block instead of dying).

### 05:25 UTC: probe - at the training temperature the policy cannot even reach the block; T cap 0.5 (r3b3_t05)

* Probe (local CPU, ANALYSIS ONLY): 24 of the policy's OWN pre-block greedy states (the 5,136M
  snapshot's greedy-line pool, x 250-660 after CP2, progress 30.5-32.4%), `record_ckpt
  --spawn-states --stochastic --exec-temp 2` (= training at T 1): **0/24 reach the block** (best
  31.75%; most die at x -124..+211, z 10,350-10,600) - while the greedy line from the same states
  reaches it (32.5%). So under T 1 the precise post-CP2 line breaks before the gate and training
  hardly practises the gate at all; the noise that helped earlier gates now hides this one.
* r3b3_ezf stopped at ~5.34B (snapshot ckpt_5337251840, md5 4c3efd22...): 12 evals 30.7-31.8k u,
  ceiling flat.
* **r3b3_t05** (05:12): resume of that snapshot + `--unstuck-max 0.5 --ez-eps 0 --ez-front 0
  --ckpt-every 1e8`, 600M steps. Evals 5,337-5,413M: frontier 30,968-31,797 u (the line holds);
  several greedy episodes now live to 40-43 s at the block; ceiling 32,141 -> 32,154 u.
* ANOMALY: approx_kl 0.6-14 per update in r3b3_t05 (0.04-0.24 in t1 / ezf, 0.03-0.15 in tp0x)
  without visible damage to the greedy line or the returns. The reported kl is (logp_old - logp_new)
  on the last minibatch after the whole update; a likely source is the pitch head (sigma 0.007, no
  physics effect, no entropy term) at the sharper temp 1.5 - not verified. Watch the evals, not kl.

### 05:52 UTC: r3b3_t05 - 32.775; the greedy line starts to get past the block

* Evals 5,438-5,715M: 32.545, ..., 32.634 (5,589M), **32.775** (5,639M, median 32.2%), 32.581
  (5,715M); every eval CP2. Episode 2 of 5,639M: best 32.78% at 37.7 s at (-745, -39, 10,286) - WEST
  of the block and BELOW the kill floor's level (*112 at z 10,578), the field's way (it lives to
  143 s and ends at (-192, 297, 9,715)).
* New behaviour: many greedy episodes now die at ~40.5 s instead of ~37 s because at ~36.3 s they
  turn back EAST at the block (v (828, -815) at 36.5 s), fall back to the void-bottom ramp (38.25 s,
  z 9,689), climb it again and make a second westward approach (block again at ~40.3 s). With no
  time penalty a retry is free.
* Training ceiling 32,154 u (flat since the restart). approx_kl still 0.9-5.6 per update with the
  line intact. Decision: extend r3b3_t05 (the gate is moving) instead of the planned from-scratch
  test.

### 06:14 UTC: r3b3_t05 end (5.94B) and its extension r3b3_t05x

* r3b3_t05 5,337-5,937M (24 evals): best **32.775** (5,639M); the rest 30.9-32.6 (frontier
  30,947-32,044 u), every eval CP2; training ceiling 32,141 -> 32,177 u. Final ckpt
  runs/r3b3_t05/ckpt_final.pt (5,937,561,600, md5 42721461...); snapshots every 1e8 on the box.
* **r3b3_t05x** (06:13): bare resume of that final (T cap 0.5, time-pen 0, surf-bonus 0, greedy 0.15,
  int-real-only, ez off - all restored), +1.5e9 steps, evals every 25M. The block gate is moving
  (one eval episode under it at 5,639M; retries at ~40 s); the planned from-scratch test of the
  recipe waits until it stalls.

### 06:45 UTC: r3b3_t05x stalls at the block; concentrated greedy-line practice (r3b3_gw)

* r3b3_t05x 5,937-6,240M (12 evals): frontier 31,041-31,862 u, no eval above 32.775; ceiling flat at
  32,177 u. Stopped; snapshot ckpt_6237978624 (md5 6add4a8b...).
* Why the gate is slow (from the probe): at the training temperature most attempts from the CP2
  area die before the block, and the default greedy window (5..0.5 s before the death) puts only
  ~1/9 of its rows within ~1 s of the block -> ~2% of spawns practise the gate itself.
* **r3b3_gw** (06:44): resume of that snapshot + `--respawn-greedy 0.3 --respawn-greedy-window 0.25
  2.0` (30% of the non-start rows from 2..0.25 s before each greedy death / best-progress stop) +
  `--ckpt-every 1e8`; everything else restored (T cap 0.5, time-pen 0, ...). 400M first.
* Its resume eval (the snapshot, 6,238M): deaths at 32-290 s - the line now often SURVIVES at the
  block for 47-290 s (best-progress stops at 36.1-41.8 s) instead of dying there.

### 07:12 UTC: r3b3_gw - the training ceiling moves past the block; best 32.814

* r3b3_gw (30% greedy rows, 2..0.25 s window) 6,238-6,465M: evals 32.814 (the resume eval of the
  6,238M snapshot - the branch's best), 31.747, 31.948, 32.473, 32.427, 31.981, 32.654 (median 22.4:
  5 episodes died at 21.2-21.5 s near the window / CP1, x ~1,320-1,460, z ~11,700-11,830 - a one-eval
  wobble, the next evals are back), 31.805, 31.992, 32.425.
* **Training ceiling 32,177 -> 33,096 u (33.85%)** within ~230M steps: real training episodes now
  get ~900 u past the block. Every earlier gate went ceiling first, greedy line next.

### 07:40 UTC: r3b3_gw end; extension r3b3_gwx

* r3b3_gw 6,238-6,638M (17 evals): best 32.814 (its resume eval), the rest 31.7-32.7 (frontier
  31,038-32,007 u); training ceiling 33,096 u (33.85%) since ~6.46B; avg 141k steps/s (the 30% greedy
  rows make short episodes; long eval episodes at the block cost time too). Final ckpt
  runs/r3b3_gw/ckpt_final.pt (6,638,010,368, md5 aef2aa51...).
* **r3b3_gwx** (07:38): bare resume of that final (greedy 0.3 / window 0.25-2.0, T cap 0.5, time-pen 0,
  int-real-only restored), +1.5e9 steps. If the greedy line has not followed the ceiling by ~08:30,
  the next arm adds `--sil-coef 0.1 --sil-tempered 1 --sil-top 256` to consolidate the training
  passes. Box deadline 10:37 UTC.

### 08:08 UTC: r3b3_gwx - the greedy line gets past the block's west side; best 32.823

* Evals 6,638-6,840M: **32.823** (6,638M, the gw final's own eval), 32.582, 31.990, 32.597, 32.798,
  32.793, 32.795, 32.806, 32.816 - the 32.8% band is now the typical best episode (was ~32.5%).
  Training ceiling 33,096 -> 33,204 u (33.96%); 125k steps/s.
* 6,814M episode 0: best 32.81% at 41.3 s at (-789, 35, 10,304) - west of the block and down, the
  field's way - then it drifts back east under the kill floor *112 and stays alive to the 300 s cap
  (ends at (-476, -97, 9,933)). Next obstacle: the passage further west - the thin wall *145 (x ~-928,
  z 10,136-10,288) with the kill plate *104 on its top (z ~10,320) and the kill plate *43 under it
  (z ~10,100, x -1,388..-944); the field's route continues at z ~10,050-10,120 to x ~-1,940.

### Harness issue (r3b3): the lineage recipe lost round 1's novel + ratchet

* `autoresearch/config.json` `incumbent_recipe` (a lineage key) at the start of round 3 was the
  round-0 GR recipe WITHOUT `--respawn-novel --race-ratchet`, although r1b1 added them in round 1
  (its 07:25 UTC note) and they are the validated scratch pair. Presumably round 2's winner did not
  change the key, so the operator's value came back. A `box.py launch` without `--resume` therefore
  ran the old GR recipe. Restored on this branch (08:10 UTC): the recipe now ends with
  `--respawn-novel --race-ratchet`. My warm-validated flags (time-pen 0, greedy spawns, int-real-only,
  T cap) are NOT in it - they are untested from scratch.

### 08:36 UTC: MILESTONE - 35.443; past the block and the thin wall, down the field's route

* r3b3_gwx evals 6,865-7,092M: 31.2-32.9 until 6.94B, then **33.655 (6,966M), 33.678, 35.443
  (7,016M), 34.531, 32.580, 34.558 (7,092M)**; medians ~32% (most episodes still die at the block).
* 7,016M episode 8: best 35.44% at 45.7 s at (-1,575, -298, 9,006) - west past the block, past the
  thin wall *145 (x ~-928), west to x ~-1,900, then down the field's descending route (the route at
  (-1,616, -304, 9,352) is 35.25%); episode 5 reaches (-2,021, -272, 10,060) at 42.3 s (the route's
  western turn). Training ceiling 33,204 -> 35,021 u (35.82%); reservoir min depth 64.75% (35.25%
  progress); T 0.5 -> 0.21 (the new eval bests halve it).
* The concentrated greedy-line pool (30%, 2..0.25 s before each greedy death) did what the default
  window did not: ~550M steps after it started the gate fell, ceiling first (6.46B), greedy line
  after (6.97B).

### 09:04 UTC: MILESTONE - 39.435, CP3

* r3b3_gwx evals 7,117-7,419M: 34.521, 35.353, 35.337, 34.533, 31.954, 34.535, 35.812, 34.522,
  **37.370 (7,318M, CP3)**, 38.190, **39.396**, 38.603 (median 37.8%), **39.435 (7,419M)** - CP3
  (sgcp_3) in every eval since 7.32B. Training ceiling 35,021 -> 41,587 u (42.5%); reservoir min
  depth 60.76% (39.2% progress); T 0.21-0.25.
* After ~1.5B steps at the block, the gates past it fell in ~400M: the thin wall, the descending
  zig-zag, CP3. The field's route continues to ~46% at (5,232, -752, 8,520) in my steepest-descent
  trace.

### 09:32 UTC: 42.6% - the greedy median follows

* r3b3_gwx evals 7,444-7,645M: 39.439, **41.998**, **42.081**, 39.450, 39.453, 39.406, 39.434,
  **42.613 (7,620M)**, **42.497 (7,645M, median 42.4%)**; all CP3. Training ceiling 41,961 u (42.9%);
  reservoir min depth 57.23% (42.8% progress); T 0.04 (the eval bests keep halving it).

### 10:02 UTC: 42.804; the greedy line consistent to 42.6%

* r3b3_gwx evals 7,670-7,872M: 42.640, 42.640, 42.580, 42.587, 42.630, 42.630, 42.654, 42.610,
  **42.804 (7,871M)**; medians 42.5-42.6% in most evals. Training ceiling 46,048 u (47.1%); reservoir
  min depth 55.64%; T 0.04 -> 0.39 (no +500 u eval best since ~7.62B).
* Where the 7,871M episodes stop (all on the field route's descending eastward legs past CP3):
  (-2,003, ~190, 9,763) at 42.6 s (34.4%), (2,269, -217, 8,505) at 47.7 s (39.4%), and the best at
  (5,296, -560, 8,854) at 56.8 s (42.8%, the route point (5,296, -560, 8,904)); then a fall.

### 10:32 UTC: 47.906 - median 47.7%; last continuation r3b3_gwx2

* r3b3_gwx 7,897-8,123M: 42.672, **47.534 (7,922M)**, 46.808, 46.515, 46.895, 47.005, 32.109 (one
  CP2 wobble), 47.649, 47.888, **47.906 (8,123M, median 47.7%)**; all CP3. Training ceiling 46,934 u
  (48.0%); reservoir min depth 54.46%. r3b3_gwx ended at 8,138,522,624 (avg 151k steps/s); its
  ckpt_final (md5 0798b81f...) is pulled to /home/steam/ar/runs/r3b3/r3b3_gwx/ckpt_final.pt.
* So from 32.8% (6.6B) the concentrated greedy-line recipe went 35.4 (7.02B), 39.4 (7.37B), 42.6
  (7.62B), 47.5 (7.92B): one gate per ~150-300M steps once the block fell.
* **r3b3_gwx2** (10:31): bare resume of that final, +7e8 steps (sized to end before the 11:58
  wrap-up), evals every 25M.

### 11:04 UTC: 50.827

* r3b3_gwx2 (bare resume of gwx's final) evals 8,139-8,441M: 47.793, 47.715, 47.775, 48.913, 49.811,
  48.775, 49.808, 50.125, 50.613, 47.630, 50.637, 50.598, **50.827 (8,441M)**; all CP3; medians
  16-49.5% (the line wobbles while the frontier moves). Training ceiling 53,121 u (54.3%); reservoir
  min depth 47.35%; T 0.09; 190k steps/s.

### 11:57 UTC: the last continuation - 54.674

* r3b3_gwx2 8,466-8,819M: 50.571, 54.040, 44.516, 50.574, 50.496, 54.166, 53.778, 48.863, 54.378, 54.321,
  **54.674 (8,718M, median 47.4%)**, 54.585, 54.636, 54.591, 54.468; all CP3. Ended at 8,838,971,392 (avg
  168k steps/s); ckpt_final pulled by a direct scp (box.py pull --ckpt hung twice, 10 min and 6.5 min, with no
  scp child; a plain scp took 12 s) and md5-checked: d3d06fb18e15b261bba72d0740bfdab3.

## ROUND 3 LESSON (r3b3) - written at wrap-up

**Result:** best greedy eval from the true start **54.674** (r3b3_gwx2 at 8,717,860,864; incumbent 28.841): CP3,
the greedy line past the human's whole CP2 sequence, a kill-wrapped block and a thin wall past CP2, CP3 and the
descending legs beyond. 0 finishes. Hand-on checkpoint: /home/steam/ar/runs/r3b3/r3b3_gwx2/ckpt_final.pt (8,838,971,392, md5 d3d06fb1...).

**The lineage of the best run** (warm, from round 2's from-scratch band checkpoint r2b2_sband2, 3.336B; T cap 2):
sgir (+ --respawn-greedy 0.15 --int-real-only 1; 29.823 once, then pit-locked) -> tp0 (+ --time-pen 0
--surf-bonus 0 --critic-warmup 20) -> tp0x -> t1 (T cap 1, from the 4,536M snapshot) -> ezf (+ frontier ez bursts,
null) -> t05 (T cap 0.5, bursts off) -> t05x -> gw (--respawn-greedy 0.3 --respawn-greedy-window 0.25 2.0) -> gwx ->
gwx2 (bare resumes).

**What worked (in order of effect):**
1. **`--time-pen 0 --surf-bonus 0`** (warm, with a 20-update critic warmup). Under the 0.5/s time penalty a slow
   stretch is worth less than dying early: the CP2 detour (2.8 s at no progress) netted -0.6 vs the pit dive, the
   cavity dip (~3.5 s) -1.75. The greedy line drifted between modes and locked into the pit in every warm lineage
   (rounds 1-2, arm 2). Removing the charge on the pit-locked checkpoint: the greedy line crossed the pit 25M steps
   after the warmup, entered the cavity by ~100M, flew the human's cavity route at ~380M (30.35%), CP2 at ~400M
   (31.8%). The trainer refuses a surf bonus >= the time penalty (farmable), hence --surf-bonus 0.
2. **The unstuck T capped low (0.5).** Without the time penalty the line held at T 0.4-1.3 and broke when T climbed
   past ~1.5 (4.6B: evals 32.5 -> 24-28). At T 1 the stochastic policy could not even reach the post-CP2 block from
   its own pre-block greedy states (probe: 0/24), so training hardly practised the gate.
3. **Greedy-line spawns concentrated at the greedy line's failure** (NEW --respawn-greedy F
   --respawn-greedy-window LO HI): F of the non-start spawn rows are the policy's own greedy-eval states from HI to LO s
   before each failed greedy episode's end (or its best-progress stop), rebuilt every eval. With 0.15 / 5..0.5 s the
   block did not move in ~1.7B steps; with 0.3 / 2..0.25 s the training ceiling moved past it in ~230M steps and the
   greedy line followed after ~550M - then the gates past it fell one per ~150-300M steps (32.8 -> 35.4 -> 39.4 ->
   42.6 -> 47.9 -> 50.8 -> 54.7).

**What did not work / caveats:**
* Greedy-line spawns where the policy's sampling never succeeds from the greedy states are useless (arm 1, the CP2
  pit under the time penalty: 0/30 crossings in a probe; one flip in 8 evals). They need a frontier the stochastic
  policy can sometimes pass.
* Frontier-only ez bursts (--ez-eps 0.05 --ez-front 0.04) at the block: null in 250M (a burst cannot change altitude
  in the air; the band starts after the ramp that sets it).
* --int-real-only (NEW): on a warm start the restored count table already holds the novel rows' visits, so it mostly
  stopped paying novel-row episodes (int/ep 1.16 -> 0.26); its effect is not separated from the greedy spawns.
* approx_kl read 0.6-14 per update at T 0.5 (vs 0.03-0.24 at T 1) with the line and returns intact - not explained.
* --sil-tempered / --sil-top (NEW, CPU-tested) never ran on the box.

**For the next round:**
* Start from the hand-on checkpoint (bare resume restores every flag above). Watch T: keep the cap at 0.5; a T rise
  past ~1.5 broke the line. Keep evals every 25M and --ckpt-every 1e8 (good weights get overwritten otherwise).
* The open science: does the recipe work FROM SCRATCH? Untested: --time-pen 0 --surf-bonus 0 + T cap 0.5-1 +
  --respawn-greedy 0.3 --respawn-greedy-window 0.25 2.0 on top of the lineage's scratch recipe (now
  `config.incumbent_recipe` = GR + --respawn-novel --race-ratchet, restored this round - harness note above). The
  ledger's cannonball result (--time-pen 0 from scratch was slower, 0.6x) is a different regime.
* Finishes will need speed again: the time term of the score only exists after a finish; reintroduce a time cost
  then (generic: per-second penalty only once the policy finishes).

**Ops:** box 56978070 (RTX 4090 + Ryzen 9 5950X, $0.375/h, machine 29449), 01:16-12:06 UTC (~10.8 h, ~$4); 140-240k
steps/s depending on the greedy-row share and eval lengths. Two offers failed readiness and were blocklisted by
box.py (machines 149861, 136954). Local CPU probes with record_ckpt --spawn-states from the policy's own states
(greedy-pool rows in ck["respawn_greedy"]) were the decisive diagnostics (~5 min each).

## Wrap-up (r3b3) - 2026-10-21 12:06 UTC

* Final: RESULT.json `"final": true`, score **54.674** (r3b3_gwx2, eval traj_8717860864: 9 episodes, best 54.67%,
  median 47.36%, 0 finishes, CP3). Per-run bests (score.py): g15 28.380, sgir 29.823, tp0 30.348, tp0x 32.760,
  t1 32.554, ezf 32.651, t05 32.775, t05x 32.589, gw 32.814, gwx 47.906, **gwx2 54.674**.
* Hand-on checkpoint: /home/steam/ar/runs/r3b3/r3b3_gwx2/ckpt_final.pt (8,838,971,392, md5
  d3d06fb18e15b261bba72d0740bfdab3); also local: r3b3_gwx/ckpt_final.pt (8,138,522,624, md5 0798b81f...),
  r3b3_tp0/ckpt_final.pt, r3b3_sgir/ckpt_final.pt.
* Branch head: code commit c24b11b (--respawn-greedy, --int-real-only, --sil-tempered, --sil-top) + notes,
  config.json incumbent_recipe with --respawn-novel --race-ratchet restored.
* Box 56978070 released at wrap-up.

## Round 4, branch r4b3 - 2026-10-21 13:15 UTC

### Box

* 12:50-13:15 UTC: no offer under the caps (cheapest 4090 $0.51 vs 0.45, 5090 $0.75 vs 0.60, 3090
  $0.53 outside the old Xeons vs 0.30); a `box.py rent` retry loop (every 3 min) got **box 57057785,
  RTX 3090 on a 32-thread 5.1 GHz host, $0.268/h, machine 151065 (Missouri)**, READY at 13:15.
  Deploy (branch + incumbent ckpt, md5 d3d06fb1...) took 2 min.

### Evals were HALF of round 3's wall time (measured)

* The incumbent run r3b3_gwx2 logs a cumulative fps; elapsed = steps / fps per row gives the
  per-iteration time: median 1.60 s per 524,288-step iteration = **328k steps/s while training**,
  but the 30 greedy evals took **2,003 of 4,168 s (48%)**: 67 s each on average, 105-143 s late in
  the run. So the run averaged 168k steps/s.
* Why: with `--time-pen 0` the greedy line often stops somewhere safe and lives to the 300 s cap.
  The 54.674 eval (traj_8717860864) holds 174,626 ticks: 5 of its 9 episodes ran the full 300 s
  (their best progress came at 13.5-127 s). An eval runs its 9 episodes one after another on a
  1-env core.

### The training stall rule kills legitimate flight on this config (measured, not changed yet)

* `--reward-per-decision` is off, so RaceReward runs EVERY TICK with `every = 1`: the 15 s stall
  timer re-arms only on a tick whose field distance beats the running best by > 32 u in that ONE
  10 ms tick, i.e. **~3,200 u/s straight down the field** (maxvelocity 4,000). CLAUDE.md documents
  the threshold as per DECISION ("scale it with the decision rate"); per tick it should be ~8.
* Offline on the 54.674 eval (the training rule replayed on each greedy episode): the greedy line
  has only 3-9 such ticks in its first 42 s, and the rule would kill **every** episode at
  42.4-43.1 s / **34.3%** (two at 15.0 s). At 16, 10.7, 8 or 4 u per tick it kills nothing before
  each episode's own best. Training's `race/stall_frac` was 0.35-0.49 (that share of training
  episodes ends by this kill).
* So training episodes live at most ~15 s past their last near-max-speed dive; from-start training
  episodes die at ~34%, and the greedy-line pool (eval states, evals never stall-kill) is what feeds
  training past that point. Candidate fix: `--stall-eps 8` (= 32 u per 4-tick decision). It changes
  the training objective (no artificial termination of slow but correct flight), so it is an arm,
  tested against a control at matched steps.
* `--eval-stall 1` (the existing mirror of that rule for evals) would therefore cut the 54.674 eval
  to **34.45** - never use it on this config.

### NEW `--eval-patience SECS` (commit 6eaeecf)

* Ends a greedy EVAL episode once its best field progress has not grown by 32 u over SECS seconds -
  CUMULATIVE (best vs the best at the last re-arm), so a line that is still gaining at any rate,
  retries included, is never cut. Offline on the 54.674 eval: 10 / 15 / 20 / 30 s keep every
  episode's best (54.67) at 32 / 34 / 35 / **39%** of the ticks. The greedy-line pool reads a cut
  episode as the truncated one it would have been (only its best-progress anchor; as_truncated).
* Default off, config key written only when on, restored on resume, TRAIN_ONLY for record_ckpt.
  11 tests (tests/python/test_eval_patience.py), incl. a CPU smoke on sidistic and flag-off
  bit-identity against the pre-change trainer (config, csv, eval bytes, weights).

### r4b3_ctl (launched 13:20 UTC): the incumbent's own continuation, cheap evals

* Bare resume of the incumbent ckpt (r3b3_gwx2 final, 8,838,971,392) + `--eval-patience 30
  --ckpt-every 1e8`, 600M steps, evals every 25M. Everything else restored (greedy rows 0.3 /
  0.25-2.0 s, T cap 0.5, time-pen 0, band novel rows, ratchet, int-real-only). It is the control
  for the stall-eps arm and banks evals meanwhile. Reservoir at the resume: min d 44,325 (54.7%).

### Even with patience, evals are ~half the wall time on this box: NEW `--eval-parallel 1`

* r4b3_ctl on box 57057785: **226k steps/s while training**, but the evals with `--eval-patience 30`
  still took 72-116 s each (patience cut 4 of 9 episodes; the rest ran 40-80 s): average 105k
  steps/s over the first 75M. An eval tick costs ~1.6 ms here: one batch-1 GPU forward per
  decision (kernel-launch latency, not arithmetic) plus the Python loop, and the 9 episodes run
  one after another on a 1-env core.
* `--eval-parallel 1`: the eval core gets --eval-eps envs and the episodes run SIDE BY SIDE
  (surfgym.record.record_rollout_parallel; env i's FIRST episode is episode i; one batched
  forward per decision for all of them), the file is written afterwards in the same format.
  The eval then costs its longest episode, not the sum. Same start distribution (every env
  starts at the map's own spawn points, the eval core's pool). The greedy-line pool and the
  patience rule have side-by-side versions (GreedyLinePool.hook(n), make_eval_patience_hook_par).
  Single map, greedy-only evals; refused with --eval-stall / --eval-mirror / --goal-planner /
  --maps. Header and config say `eval_parallel: 1`. Tests (tests/python/test_eval_parallel.py):
  one episode on a 1-env core is record_rollout BYTE FOR BYTE; N side by side each from a map
  spawn, first episode only; the pool's capture = each episode's rows; per-env patience; the
  trainer smoke (header, config, resume restore, refusals). Flag off: the eval-patience
  bit-identity test (trainer before both changes vs after) covers it.

### 13:53 UTC: r4b3_ctl at 9.08B - 54.793 (CP3); the ceiling moved when T reached 0.3-0.5

* Evals 8,839-9,066M (score.py best): 50.581 (the resume eval), 54.637, 54.565, 50.593, 54.536,
  54.668, **54.793 (8,990M, median 47.7%)**, 54.582, 54.329, 50.711. The 54.6-54.8% episodes stop
  at ~63 s near (-700..-814, +-50, 8,220-8,320) heading north at ~900 u/s, then wander and die
  7-35 s later at z ~7,260-7,620.
* Training ceiling 54,406 u (55.65%, flat since round 3) -> 55,206 (+144M) -> **56,388 u (57.68%,
  +176M)**, while the unstuck T climbed 0.06 -> 0.29 -> 0.5 (cap): the stuck counter passed 200M
  at ~+72M. Same pattern as round 3: gates open when T rises, and the 0.5 cap keeps the line.
* Throughput: 226k steps/s training-only, 121k average (evals with patience: 18-116 s each).

### 14:03 UTC: r4b3_ctl stopped at 9.14B; the main line now runs side-by-side evals (r4b3_par)

* r4b3_ctl's last evals 9,066-9,141M: 50.711, 50.893, 49.969, 47.534 (best episode), medians 47-49%:
  the 54.6-54.8% episodes disappeared while T sat at its 0.5 cap (since ~9.02B) and the training
  ceiling moved to 57.7%. Round 3 saw the same wobble while a gate opened ("medians 16-49.5%
  (the line wobbles while the frontier moves)"). Branch best stays 54.793 (8,990M).
  ckpt_latest (9,141,485,568, md5 e413d903...) pulled to runs/r4b3/r4b3_ctl/.
* `box.py deploy` on a box whose GPU is training fails at tools/gpu_health.py (it measures GEMM
  TFLOPS on the busy card and calls it capped) - the checkout is already updated by then
  (git fetch + checkout -B); the health check passed on the idle card at the first deploy.
  Harness issue, not a bad box: never blocklist on a deploy-time health failure while training.
* **r4b3_par** (14:02): resume of that checkpoint + `--eval-parallel 1 --ckpt-every 1e8 --timing`
  (eval-patience 30 restored), 300M steps. First evals: 11-20 s each (was 72-116 s).
* `--timing` profile (median of 30 iterations, 2,344 ms per 524,288-step iteration): update (GPU)
  965 ms; rollout 1,272 ms = reward_py **487** (RaceReward runs every TICK: 512 calls/iter, ~0.95 ms
  each for 1,024 envs), vis_cpu 306, sync_copy 315, lidar 204, rollout_fwd 148, env (physics) 126,
  pool 77, respawn 27. The Python reward is now the biggest CPU phase.

### 14:06 UTC: the per-tick stall rule is a KNOWN pathology (ledger 2026-10-03, skate_laby) - arm = `--stall-eps 5`

* docs/research-results.md, "skate_laby: the STALL RULE killed every healthy training episode at
  15 s (it is checked per TICK). --stall-eps 5; skLABY10": same cause as above (without
  --reward-per-decision the detector is called every tick, so it needs > 3,200 u/s of field progress
  in one tick). There `--stall-eps 5` (500 u/s per tick, "the speed floor's rate") on a warm resume
  took training ep_len 1,580 -> 4,700-7,300 ticks, the pool's deepest state 63% -> 98.5% of the
  route and the training win rate 0% -> 1.92% within 18M steps. "On cannonball the rule rarely
  fires ... slow or flat maps are where it bites."
* sidistic's training stall_frac is 0.29-0.49 (r4b3_ctl), i.e. it bites here too. The arm uses the
  ledger's constant 5 (one constant for every map, CLAUDE.md 0b; on the 54.67 eval any value 4-16
  kills nothing before an episode's own best).
* Also in that ledger area: `--reward-per-decision` cuts reward_py ~3x (validated warm, not from
  scratch); here it would save ~330 of 2,344 ms per iteration and make the stall rule per decision.
  A later throughput lever (it changes the novelty dwell counting and the diagnostics' cadence).

### 14:22 UTC: arm A (r4b3_par, control with side-by-side evals) - the greedy line switched to a 50% route at T 0.5

* r4b3_par = r4b3_ctl's 9,141,485,568 checkpoint + `--eval-parallel 1`, 186.6M steps (stopped).
  Throughput 180k steps/s average, 225k training-only: evals now 13-28 s each (were 72-116 s).
* Evals (score.py best): 50.572, **54.396** (9,167M), 50.056, 50.558, 50.734, 50.623, 50.308, 52.503.
  Training ceiling flat at 56,388 u (57.68%) the whole segment; T at its 0.5 cap (stuck counter
  423-604M); novel rows per pool swinging 152-1,654 (share 4-43%); stall_frac 0.24-0.48.
* What changed in the greedy line: at 9,167M two episodes still reach 54.4% at ~63.9 s via
  (-650, -45, 8,113); from 9,217M 7 of 9 episodes take ANOTHER route and fall to a floor at
  (2,500-2,900, +700..+1,100, z ~6,930) at ~61.5 s = 50.0-50.6%. A mode switch, not a general
  decay; it began within ~50M steps of T reaching 0.5 on ALL heads (yaw sigma x1.5).
* docs/unstuck.md (cannonball measurements): the yaw sigma is the fragile component (x1.5 halves the
  flight, x2 kills it); the keys temperature buys the most diversity per unit of progress lost and
  is the only knob that reached never-seen cells. The lineage runs `--unstuck-temp-heads all`.

### 14:20 UTC: arm C = r4b3_keys - `--unstuck-temp-heads keys` from the same 9.141B checkpoint

* Bare resume + `--eval-parallel 1 --ckpt-every 1e8 --unstuck-temp-heads keys`, 200M steps. T 0.5
  (restored) now tempers the four key heads only; the yaw / pitch sigma stay the learned ones.

### 14:24 UTC: the "50% route" is the +y MIRROR LANE of the same line (analysis)

* r4b3_par 9,167M ep 0 (54.4%) vs 9,217M ep 0 (50.1%), positions every second: identical line to
  ~52 s (within 180-560 u), then they split into the map's two y-mirror lanes at (6,945, -386) vs
  (6,666, +63): the -y episode is at (2,141, -305, 7,432) at 61 s and reaches 54% at 63 s; the +y
  episode is at (2,920, 909, 7,071) at 61 s - ~360 u lower, ~780 u further east - and falls onto the
  floor at z ~6,930 (50.0%). The map is y-symmetric (round 1 saw the CP2 line flown on both lanes),
  so this is a precision gap between lanes, not a route choice: which lane an episode drifts into at
  ~52 s decides 50% vs 54%. A policy that is not mirror-equivariant learns each lane separately.

### 14:51 UTC: arm C result; two trainers on one box do NOT work; arm B2 = keys-only + `--stall-eps 5`

* **C (r4b3_keys)**, 9,141.5M -> 9,341.8M (200M, avg 187k steps/s): evals 47.680, **54.594**, 54.358,
  48.785, 54.431, 54.387, 50.840, 50.806 - 4 of 8 at 54.4-54.6% where the control A (all heads at
  T 0.5) had 1 of 8 over the same steps from the same checkpoint (medians 44.6-53.7 vs 46.3-50.0).
  The training ceiling stayed at 56,388 u in both. The learned yaw sigma drifted 0.047 -> 0.071 under
  C (the unstuck entropy bonus x(1+T) still acts on it). One seed: C keeps the line better, it does
  not move the gate. Kept as the main line's setting (keys-only tempering, as docs/unstuck.md
  measured).
* The gate: from the frontier (-650, 100, 8,200, 54.4%) the field's route runs west through a slot
  at z ~8,400-8,520 between kill plate *1 (z 8,512, x -1,182..-1,058) / a func_wall *147 with a kill
  strip *136 on its east face (x ~-1,080, z 8,512-8,704) and the kill brushes *100 / *95 / *92 below
  (tops z 8,386 / 8,066), out to x ~-3,114 (57.45% - the training ceiling is 57.68%), then back east
  lower (59-61%) and on to ~65% by x -4,550. The greedy episodes reach the slot's mouth at ~64 s
  heading north/south at ~900 u/s and never turn into it. No training episode has passed 57.68%
  since ~9.02B.
* Tried two trainers at once on the box (C2 = C's bare continuation and B2 below, both from C's
  final): load average 66 on 32 threads, GPU 3%, ~2-22k steps/s each - each trainer keeps ~26
  threads busy (16 OpenMP + numpy/torch). Killed C2 after ~12 min; never run two trainers on one box.
* **B2 (r4b3_kst5)** (14:39): C's final (9,341,763,584, md5 8688df4f...) + `--stall-eps 5
  --critic-warmup 20 --ckpt-every 1e8` (keys-only T, side-by-side evals, patience restored), 200M.
  Reference: C's own 200M (the same config without the stall change, from 9.14B).

### 14:57 UTC: arm B2 (keys-only + `--stall-eps 5`) MOVES the frontier - 55.065, CP4

* r4b3_kst5 (from r4b3_keys' final 9,341,763,584), first 77M steps: evals (score.py) 54.400 (the
  checkpoint itself), 54.632 (median 54.5%), 54.995 (CP4, median 54.6%), **55.065 (9,418M, CP4)** -
  the line enters CP4 (sgcp_4) for the first time on this lineage.
* Training side: stall_frac 0.35-0.50 -> 0.14-0.19, ep_len 900-1,200 -> 1,150-1,550 decisions,
  reward per episode 6-7 -> 7.5-11. **The reservoir's deepest state moved 54.7% -> 56.9% of d0
  (min depth 45.3% -> 43.1%) in 77M steps** - it sat at 45.3-45.5% through A's and C's 200M each -
  and the training ceiling 56,388 -> 56,575 u. Same mechanism as the ledger's skate fix (pool's
  deepest 63% -> 98.5%): training episodes past the frontier are no longer killed 15 s after their
  last 3,200 u/s tick, so the 2 s harvest margin keeps states further down the line.
* Matched-step summary (one seed per arm, from 9.14B / 9.34B): A (all heads T 0.5) ceiling flat,
  evals <= 54.4 (1 of 8 at 54.4); C (keys only) ceiling flat, evals <= 54.6 (4 of 8); B2 = C +
  stall-eps 5: deepest reservoir state +2.2% of d0 and CP4 within 77M. Kept as the main line.

### 15:11 UTC: B2 end - 57.720 (CP4), median 56.9%; extended as the main line (r4b3_kst5x)

* r4b3_kst5 9,342-9,542M (200M; avg 120k steps/s - the first ~12 min shared the CPU with the
  killed C2 trainer): evals 54.400, 54.632, 54.995, 55.065, 54.587, 54.933, **57.720 (9,493M)**,
  57.188 (9,518M, median 56.9%: 7 of 9 episodes past the old 54.6% stop). Training ceiling 57,282 u
  (58.6%), reservoir min depth 42.16% (deepest state 57.8%), T 0.5 -> 0.36 (the eval bests halve it).
* ckpt_final.pt (9,542,041,600, md5 f6bed432...) copied by scp and md5-checked: box.py pull --ckpt
  took ckpt_latest.pt (md5 9347a05a..., the 60 s save before the final).
* **r4b3_kst5x** (15:09): bare resume of that final (stall_eps 5, keys-only T cap 0.5, patience,
  side-by-side evals, greedy rows 0.3 restored - checked in the log), 2e9 steps, evals every 25M,
  snapshots every 1e8.
* --front-temp: the trainer smoke ran clean but its step-line note sat inside the --ez-eps block
  (never printed without bursts); moved out, test rerun pending.

### 15:22 UTC: r4b3_kst5x - 60.253; --front-temp built and tested

* r4b3_kst5x (bare resume of kst5's final) evals 9,543-9,668M: 57.752 (the resume eval), **60.130**
  (9,568M), 56.935, 59.941, 59.956, **60.253 (9,668M)**; all CP4. Training ceiling 59,244 u (60.6%),
  reservoir min depth 39.53% (deepest 60.5%), T 0.15 (decaying on the new eval bests), 175k steps/s.
  The line passed the corridor's west end (kill plate *92) and the drop at x ~-3,100.
* NEW `--front-temp T_F --front-band B` (commit 4b71ca4): a per-env sampling temperature - every
  head at 1 + T_F inside the frontier band (progress >= (1 - B) x the reservoir's deepest, the
  --ez-front band), the global --unstuck T on its heads elsewhere; PPO scores each row at its own
  temperature (per-row records b_ftk / b_ftv; logprob_entropy_view takes a per-row view
  temperature). 10 tests incl. flag-off bit-identity. Not deployed yet: the next arm when the main
  line stalls (yaw exploration only where the line ends, which keys-only T no longer gives).

### 15:50 UTC: the main line stops at 60.57%; arm F = `--front-temp 1 --front-band 0.03` (r4b3_ft1)

* r4b3_kst5x 9,668-9,970M: evals 60.411, 60.538, 60.569, 60.561, 60.558, (33.538), 60.560, 60.576,
  (57.747), 60.559, **60.584 (9,945M, median 60.57%)**, 60.532 - from ~9.74B every eval's best and
  often its median end at 60.56-60.58%: the 9 greedy episodes stop at ~75-78 s at (-629..-667,
  -42..-55, 6,990-7,016) moving -y at 1,270-1,466 u/s, then wander and die at 81-108 s. The field
  route from there drops steeply (z 6,950 -> 6,566 by x -836, -> 5,766 by x -1,444, kill *24 /
  *53 / *49 alongside). Training ceiling 59,244 -> 60,907 u (62.3%): training episodes pass it.
  The +500 u eval rule saw no new best since 58,788 u (9,568M), so T went back to the 0.5 cap.
* Stopped at 9,995,550,720 (ckpt kept on the box as runs_ckpt_main.pt, md5 9b1e687d...).
* **F = r4b3_ft1** (15:50): that checkpoint + `--front-temp 1 --front-band 0.03` (every head at
  temperature 2 for envs at >= 97% of the reservoir's deepest progress = 58.8% here; the others keep
  keys-only T 0.5), 300M steps. Step line: `fronttemp 8-9% >58.8%`. Question: does yaw exploration
  at the stop (the 25 Hz policy's own sigma is 0.07) get the greedy line down the drop sooner than
  the main line, whose ceiling already leads by 1.7%?

### 16:21 UTC: arm F (--front-temp 1, band 0.03) - NULL at the 60.57% stop; why the reservoir cannot pass it; arm G = `--respawn-margin 1`

* **F = r4b3_ft1** (9,995.6M -> 10,296.0M, 300M, 179k steps/s): 12 evals, every one 60.54-60.57%
  (57.8 once), medians 56.7-60.5% - the line held (front-temp did not break it) but never passed.
  Training ceiling flat at 60,907 u (62.3%), reservoir min depth flat at 39.427% (deepest state
  exactly the greedy stop, 60.57%), T 0.5 (keys). `fronttemp` 8-26% of envs in the band.
* The stop, measured (bsp entities, ANALYSIS ONLY): the agent rides a ramp (func_wall *148, kill
  wrap *24, x -952..-296, z 6,656-7,264) to (-650, -50, 7,000) at ~75 s. Under it lie kill plates
  *53 (x -960..-416, |y| <= 960) and *36 (x -1,056..-960) at z ~6,464, then a kill box *49
  (x -1,216..-576, |y| <= 508, z 5,632-6,050) and the kill floor *31 at z 5,600; the field's route
  drops through the plates' x range (it ignores triggers) to a level at z ~5,766 going west through
  CP5 (*137, x -1,712, |y| <= 624). Training episodes reach 62.3% (z ~5,900 at x ~-1,316) - i.e.
  they drop ~1,200 u (~1.7 s of free fall) and die.
* **Why nothing past the stop enters the reservoir:** the reservoir snapshots each episode once per
  second and keeps only snapshots >= `--respawn-margin` (2 s) before the episode's end
  (surfgym/respawn.py observe). An episode that passes the stop and dies ~1.7 s later contributes
  nothing past it, so the deepest state stays at the stop (39.427% for 600M+ steps), the spawns
  never start inside the drop, and the critic only ever sees "drop -> death within 2 s".
* **G = r4b3_m1** (16:19): F's final (10,295,967,744, md5 cf6621c7...) + `--respawn-margin 1` (all
  else restored, front-temp kept so G differs from F in one knob), 300M. Read-out: the reservoir's
  deepest state below 39.43% (states inside the drop), then the greedy line past 60.57%.
  Generic: one constant in seconds (the round-18 lesson: the margin, not the sampling rule, capped
  the reservoir); the caveat is the trivial-win harvest near a goal, which needs finishes.

### 16:49 UTC: arm G (--respawn-margin 1) - NULL; main line back on the validated config (r4b3_main2)

* **G = r4b3_m1** (10,296.0M -> 10,596.4M, 300M, 184k steps/s): every eval 60.54-60.57% again;
  reservoir min depth 39.428% (unchanged to the third decimal), ceiling 60,907 u (unchanged since
  ~9.97B). So in this window no training episode passes the stop AND lives > 1 s past it: the
  margin was not the binding constraint, the passes themselves are (too) rare now.
* The greedy episodes do not die after the stop: they drift west at the ramp's height (z ~7,100,
  to x ~-1,600) and are cut by the patience rule 30 s later - they never take the drop.
* F + G: 600M steps at this gate with no change in any frontier measure; neither hurt the line.
  Total at the 60.57% stop: ~850M steps (since ~9.74B).
* **r4b3_main2** (16:46): G's final (10,596,384,768, md5 b2707815...) + `--front-temp 0
  --respawn-margin 2` - back to r4b3_kst5x's validated config (stall_eps 5, keys-only T cap 0.5,
  patience, side-by-side evals, greedy rows 0.3), 2e9 steps.
* Probe running locally (ANALYSIS ONLY): record_ckpt from G's own greedy-pool states (1,809 states,
  59.7-60.6% progress, the 2 s before each greedy stop) - greedy x8, sampled at temperature 1 x16
  and 2 x16, 8 s each: does any rollout from the stop survive the drop?

### 17:04 UTC: the 60.57% gate, measured in the simulator (ANALYSIS ONLY: own states, no demo)

Scripts: session scratchpad probe/ (macro_search.py, stage_search.py, lane_drop.py); states = G's
own greedy-line pool (1,809 STATE_DTYPE rows, 59.7-60.6% progress, the 2 s before each greedy stop).
* **Policy probe** (record_ckpt --spawn-states, 8 s each): greedy 0/8, sampled at temperature 1
  0/16 and at temperature 2 0/16 get past 60.6%; all 40 are ALIVE at 8 s, drifting west / down to
  x -1,050..-1,630, z 6,500-7,400. A WON'T, not a CAN'T-survive.
* **Random held macros on the raw core** (yaw offset U(-120, 120) from the velocity heading, random
  fwd / side / duck / 20% jump, 3 macros of 0.4-2 s, 6 s): 1,536 rollouts, best 60.59%, and every
  one past 60.57% dies within 0.1 s; most die within ~0.4 s. The stop is a local maximum of the
  field ringed by kill volumes (the ramp's wrap *24, plates *53 / *36 at z 6,464 under x
  -1,056..-416).
* **Five-stage survivor search** (each stage restarts from the previous stage's survivors at 6 s):
  survivors gather on a ledge just west of *36 at (-1,070, -60, ~6,500-6,600), 59.97%; nothing past
  60.6% in 5 stages. Below the ledge is kill box *49 (x -1,216..-576, |y| <= 508, z 5,632-6,050).
* **Hand-placed drops over the side lanes** (x -1,100..-1,550, |y| 560-600, z 6,300, moving west at
  300-900 u/s, forward held): **62.5-63.6% of progress** before dying on the kill floor *31 (z 5,600)
  at ~1.3 s; the westward ones cross the CP5 curtain (*137, x -1,712, |y| <= 624, z 5,600-6,432) on
  the way. The field's "level route" at z ~5,766 is air 166 u above that floor.
* So the next 3% of score is a two-part move the 25 Hz policy never strings together: get west of
  x ~-1,216 above a side lane (|y| ~560-620; the greedy line does drift up there after the stop),
  THEN dive. The ratchet pays the dive even though it dies. The rest of the stage (not dying on
  *31 after CP5) is a further skill - presumably a surf surface lower down, not found here.
* Novel rows ARE in the region: prog_med ~60,000 u (61.4%), 1,843 per pool, 40-50% of episodes;
  ceiling 60,907 -> 61,025 u (62.4%). The missing piece is a multi-second commitment, which no
  per-decision noise produced here (keys T 0.5, all heads at temperature 2 in F, the probe at
  temperature 2) - CLAUDE.md 0c's case for a slower decision layer.

### 17:13 UTC: NEW `--respawn-greedy-stuck N` (commit 3ea2710); main line r4b3_stuck

* The greedy-line pool keeps the 2 s before each greedy DEATH and before each best-progress STOP,
  never what the deployed line does after it stops gaining. At the 60.57% stop that is exactly
  where it wanders: west above the side lanes for 30 s (then --eval-patience cuts it), the place a
  dive gains ~3%. With N > 0 each episode whose best-progress tick lies before its last HI s also
  adds N evenly spaced states from that tick to its end: return-then-explore on the greedy line.
  Default 0; restored on resume; config only when on; TRAIN_ONLY for record_ckpt; refused without
  --respawn-greedy. Tests: tests/python/test_respawn_greedy_stuck.py (pool contents, refusals,
  trainer wiring + resume). Flag off: test_eval_patience's bit-identity test - the trainer before
  this round (584a7ad) vs this one with every new flag off - passes (config, csv, eval bytes,
  weights), which covers all four of this round's flags.
* r4b3_main2 (10,596-10,863M) evals all 60.55-60.58%; stopped. **r4b3_stuck** (17:12): its latest
  (10,862,723,072, md5 0098433e...) + `--respawn-greedy-stuck 200`, 2e9 steps: the pool went
  1,783 -> 3,375 states (stalled stretches from the stops at 73.7-77.7 s). Read-out: reservoir
  deepest below 39.43%, ceiling past 61,025 u, evals past 60.6%.

### 17:37 UTC: r4b3_stuck +269M - the training ceiling moves to 63.49% (the lane dive)

* r4b3_stuck 10,863-11,132M: training ceiling 61,025 -> **62,068 u (63.49%)** - flat at 60,907-61,025
  for ~870M steps under F, G and main2 - i.e. a real training episode now dives into a side lane to
  the value the simulator predicted for it (62.5-63.6% before the kill floor). Reservoir min depth
  39.430 -> 39.421% (the dives die within ~1.3 s, inside the 2 s margin, as expected). Greedy evals
  still 60.56-60.57% (9 evals). Running on (2e9 budget); the lineage's gates went ceiling first,
  greedy line 100-550M steps later.

### 18:03 UTC: r4b3_stuck - TRAINING breaks the 60.57% gate (reservoir deepest 68.6%, ceiling 70.3%)

* r4b3_stuck at 11,428M (+566M): reservoir min depth 39.42% -> **31.43%** (the deepest harvested
  state is now at 68.6% progress, past CP5), training ceiling 62,068 -> **68,703 u (70.27%)**, train
  ep_len 2,383 decisions, reward 15.2 per episode, int 3.4 / episode; novel rows fell to 91 per pool
  (4% of episodes: the band moved past the cells the sampler had). Greedy evals still 60.55-60.57%
  (19 evals to 11,417M) - the line has not followed yet. Gates of this lineage went ceiling first,
  greedy line 100-550M steps later.

### 18:03 UTC: CORRECTION to the 17:37 entry (attribution)

* r4b3_stuck's own progress.csv starts at ceiling 62,068 u (its first row, +0.5M): the move
  61,025 -> 62,068 happened in r4b3_main2's last ~90M steps (10,772-10,863M, before
  --respawn-greedy-stuck), not under the stuck pool as written at 17:37. Under the stuck pool the
  ceiling sat at 62,068 until +549.5M, then 63,629 and 68,703 within half a million steps (the
  breakthrough), with the reservoir's deepest state 60.6% -> 68.6% at the same time. ~1.6B steps at
  this gate in all (9.74B -> 11.41B: kst5x, F, G, main2, stuck); whether the stalled-stretch pool
  caused it is NOT shown (one seed, no control over the same steps).

### 18:27 UTC: the breakthrough did not hold; one lucky episode DISABLED the band novel rows; band 0.15

* r4b3_stuck to 11,682M: the reservoir's deepest state went back 31.43% -> 39.42% (the 68.6% states
  were replaced within ~250M steps, nothing re-harvested them); ceiling stays 68,703 u (70.27%);
  greedy evals 60.55-60.57% throughout (29 evals since 10.86B).
* **A weakness of `--respawn-novel-band B`**: the band is anchored to the all-time CEILING (rows only
  at d_best <= d <= d_best + B d0). One stochastic episode put the ceiling at 70.27%, so with B 0.05
  the band was 65.3-70.3% - ~5% of d0 AHEAD of the reservoir's real frontier (60.6%) - and moved
  states from frontier sources (radius <= 3,072 u) almost never land there: novel rows fell from
  1,843 per pool (40-50% of episodes) to 39-91 (0-4%). The novel practice around the gate stopped
  exactly when a breakthrough had just been seen.
* **r4b3_band15** (18:25): r4b3_stuck's latest (11,682,185,216, md5 b02a81b4...) + `--respawn-novel-band
  0.15` (everything else restored, incl. --respawn-greedy-stuck 200): band 55.3-70.3%, covering the
  real frontier again; rows back to 1,843 per pool, 36-48% of episodes within the first iterations.
  For the lineage: a band anchored to the reservoir's deepest state (or a decaying ceiling) would
  not have this failure mode - an untested design note, not code.

### 18:53 UTC: MILESTONE - the greedy line passes the 60.57% gate: 72.615 (CP5), median 72.1%

* r4b3_band15 (from 11,682M, band 0.15): evals 60.558, **63.356 (11,783M, CP5)**, **71.131**, 71.267,
  70.352, 71.506, **72.615 (11,909M)**, 72.091 (median 71.0%), 72.097 (11,959M, **median 72.1%**: all 9
  greedy episodes past the old stop). Reservoir min depth 38.69% -> 27.48% (deepest 72.5%),
  ceiling 71,043 u, T 0.5 -> 0.13 (new eval bests), 180k steps/s.
* So ~100M steps after the novel rows came back (1,843 per pool, 41% of episodes) the greedy line
  went 60.57 -> 63.4 -> 71.1 -> 72.6. Attribution, one seed: the band was the visible change at this
  restart; the stalled-stretch pool (--respawn-greedy-stuck 200) had run ~820M steps before it, and
  the gate had absorbed ~1.6B steps in total. The band-starvation mechanism itself is measured (rows
  1,843 -> 39-91 when the ceiling jumped ahead, back to 1,843 at band 0.15).

### 19:20 UTC: 82.674 - the line runs on past the gate (one gate per ~50-100M steps)

* r4b3_band15 12,035-12,261M: evals 72.2 (median 72.1), **79.277** (12,110M, median 78.1), 81.262,
  81.399, 81.874, 82.366, 82.145, **82.674 (12,261M, median 82.1%)**; all CP5, all 9 episodes end
  "fail" now (no stalls). Reservoir min depth 17.93% (deepest 82.1%), ceiling 82,465 u (84.3%),
  T 0.02, 183k steps/s.

### 19:48 UTC: 82.851; the next stop at 82.7% (92.5 s), measured

* r4b3_band15 12,287-12,564M: evals 80.8-81.0k u every time (82.7-82.85%), best **82.851 (12,437M)**,
  medians 53-82%; T back at the 0.5 cap (no +500 u eval best for ~300M), reservoir deepest 82.06%,
  ceiling 82,493 u (84.4%).
* The stop (bsp entities, ANALYSIS ONLY): 8 of 9 greedy episodes die at ~92.5 s at (-5,110..-5,180,
  -540..-570, 4,430-4,530) moving (+1,150, -600, -550): they surf over func_wall *152 (x -5,876..-4,976,
  z 4,477-4,792) at y ~-560 - inside the y range of the small kill block *85 (x -5,056..-4,928,
  y -704..-448, z 4,544-4,672) - and fall onto kill *122 (x -5,856..-5,024, top z 4,456) before its
  east end. The field's route goes on east at |y| < 448 between *85 and its mirror *51 and drops to
  a level at z ~4,244 (83.7-85.3%, x -4,490..-2,890).

### 20:17 UTC: the 82.7% stop falls after ~400M steps - 84.244

* r4b3_band15 12,589-12,866M: 82.0k u at 12,689M (83.9%), then **84.244 (12,840M, median 82.9%)** and
  84.234; reservoir deepest 84.6% (min depth 15.38%), ceiling 83,029 u (84.9%), T 0.5 -> 0.12.
  Box extended to 23:17 UTC.

### 20:45 UTC: a wobble at T 0.5 (the line fell back to 57.8% for ~150M steps, then returned)

* r4b3_band15 12,891-13,193M: evals at 56.5k u (57.8%, the corridor-drop stop of 10.0B) from 12,966M
  to 13,117M, then 67,479 (13,142M) and 80.8k u (82.6%) again at 13,167-13,193M. The reservoir's
  deepest state stayed at 82.1% (min depth 17.90%) throughout - training's frontier did not move
  back, only the greedy line. T at the 0.5 cap since the last +500 u eval gain (82,015 u, 12,689M).
  Branch best unchanged (84.244, 61 valid evals in this run). Same pattern as round 3's wobbles at
  the cap.

### 21:30 UTC: r4b3_band15 end (13.68B) - 84.546; continuation r4b3_band15x (evals every 12.5M)

* r4b3_band15 13,218-13,682M: evals back at 80.8-82.7k u after the wobble, **84.546 (13,645M, median
  84.2%)**; 80 valid evals in the run (11,682-13,682M, 185k steps/s). ckpt_final (13,682,343,936, md5
  d78738c7...) pulled by scp and md5-checked: the hand-on checkpoint if no later eval beats this one.
* **r4b3_band15x** (21:28): bare resume of that final, `--record-every 12.5e6` (twice the evals: a fresher
  greedy-line pool and more samples of the max, for ~15% of the steps), 1.35e9 steps (ends ~23:45,
  before the 00:09 wrap-up).

### 21:59 UTC: 85.021 (r4b3_band15x)

* r4b3_band15x 13,683-13,947M (evals every 12.5M; 151k steps/s average vs 226k training-only - the
  greedy episodes now fly ~95 s + the 30 s patience window, so a side-by-side eval costs ~25-30 s):
  frontier 82.6-83.1k u every eval, **85.021 (13,922M, median 84.5%)**. Reservoir deepest 85.1% (min
  depth 14.89%), ceiling 83,253 u (85.2%), T 0.5 (no +500 u eval best since 82,659 u at 13,645M).

### 22:26 UTC: 85.430 (median 85.4%)

* r4b3_band15x 13,960-14,199M: frontier 83.1-83.5k u, **85.430 (14,186M, median 85.4%)**; training
  ceiling 86,692 u (88.7%), reservoir deepest 85.4%, T 0.16; 153k steps/s with evals every 12.5M.

### 22:54 UTC: 91.458 - CP6

* r4b3_band15x 14,212-14,463M: **91.458 (14,287M, CP6, median 86.6%)**, then 90.3-91.4% in several evals
  (88,327 / 89,134 / 89,329 u); the unstuck tracker's best 89,417 u. Reservoir deepest 91.6% (min
  depth 8.43%), ceiling 89,568 u (91.6%), T 0.03. ~8.5% of the map left.

### 23:22 UTC: 92.150

* r4b3_band15x 14,475-14,715M: evals 89.5-90.1k u (91.5-92.2%) in most, **92.150 (14,715M, CP6, median
  87.9%)**; reservoir deepest 91.8%, ceiling 90,806 u (92.9%), T 0.36. The run's budget ends ~23:55.

## ROUND 4 LESSON (r4b3) - written at wrap-up, 2026-10-22 00:11 UTC

**Result:** best greedy eval from the true start **94.668** (r4b3_band15x at 14,991,491,072; incumbent 54.674): CP6,
94.67% of d0, 0 finishes. Hand-on checkpoint: /home/steam/ar/runs/r4b3/r4b3_band15x/ckpt_final.pt (15,032,385,536,
md5 cb1dbce40147741433b89728ed69fbd0).

**Per-run bests (score.py):** ctl 54.793, par 54.396, keys 54.594, kst5 57.720, kst5x 60.584, ft1 60.565, m1 (evals not
pulled; all ~60.55%), main2 60.581, stuck 60.573, band15 84.546, **band15x 94.668**.

**Lineage of the best run** (warm, from the incumbent r3b3_gwx2 final, 8,838,971,392): r4b3_ctl (+ --eval-patience
30) -> r4b3_keys (from ctl's 9.141B, + --eval-parallel 1 --unstuck-temp-heads keys) -> r4b3_kst5 (+ --stall-eps 5
--critic-warmup 20) -> r4b3_kst5x -> r4b3_ft1 (+ --front-temp 1 --front-band 0.03, null) -> r4b3_m1 (+ --respawn-margin
1, null) -> r4b3_main2 (--front-temp 0 --respawn-margin 2) -> r4b3_stuck (+ --respawn-greedy-stuck 200) -> r4b3_band15
(+ --respawn-novel-band 0.15) -> r4b3_band15x (--record-every 12.5e6). A bare resume of the hand-on checkpoint restores
every flag (stall_eps 5, keys-only T cap 0.5, patience, side-by-side evals, greedy rows 0.3 + stuck 200, band 0.15,
ratchet, time-pen 0).

**What worked (in order of effect):**
1. **`--stall-eps 5`.** Without --reward-per-decision the race reward and its stall detector run every TICK, so the
   15 s timer re-arms only on ONE 10 ms tick beating the running best by 32 u (~3,200 u/s down the field). It ended
   30-50% of training episodes, killing slow but correct flight 15 s after the last fast dive - the ledger's
   skate_laby pathology (2026-10-03), same fix. With it (on keys-only T) the reservoir's deepest state went 54.7 ->
   56.9% in 77M steps (flat for 400M before) and the greedy line 54.6 -> 60.6% in ~600M.
2. **Eval speedups (NEW, eval-only):** `--eval-patience 30` (end an eval episode whose best stopped gaining 32 u in
   30 s, cumulative - never cuts a gaining line) and `--eval-parallel 1` (the eval's episodes side by side on an
   N-env eval core, one batched forward per decision). Evals were 48% of round 3's wall time (5 of 9 greedy episodes
   lived to the 300 s cap without the time penalty); now ~15-30%: 121k -> ~185k steps/s on a 3090. NEVER use
   `--eval-stall 1` on this config: it mirrors the per-tick rule and would cut the 54.7% eval at 34.3%.
3. **`--unstuck-temp-heads keys`.** T 0.5 on all heads (yaw sigma x1.5) switched the greedy line into the +y mirror
   lane that dies at 50%; keys-only kept the 54% line in 4/8 evals vs 1/8 (docs/unstuck.md: the yaw sigma is the
   fragile head, the keys explore best).
4. **The 60.57% gate (~1.6B steps, 9.74B -> 11.41B).** A ramp ends where the field's straight route drops through
   kill plates; the way on is a dive in a side lane (|y| ~560-620) west of x -1,216 (simulator: 62.5-63.6% before the
   kill floor, via CP5). Probes from the policy's own pre-stop states: greedy 0/8, temperature 1 0/16, 2 0/16 past
   60.6% - all alive and drifting (a WON'T); 1,536 random held macros and a 5-stage survivor search found nothing
   past 60.6%. What moved it: a lucky training episode set the ceiling to 70.3%, which STARVED the band novel rows
   (the band hangs off the all-time ceiling: rows 1,843 -> 39-91 per pool); `--respawn-novel-band 0.15` restored them
   and the greedy line passed ~100M steps later (60.6 -> 63.4 -> 71.1 -> 72.6), with `--respawn-greedy-stuck 200`
   (NEW: the greedy pool also keeps N states of each stalled stretch) running for ~820M steps before. One seed: the
   band is the visible trigger; the stuck pool's share is unknown.
5. **After the gate:** 72.6 -> 82.7 -> 84.5 -> 85.4 -> 91.5 (CP6) -> 94.7 in ~3.3B steps; gates fell every
   ~100-400M steps, with wobbles at T 0.5 (the greedy line once fell back to 57.8% for ~150M; training's frontier
   never did).

**What did not work:** `--front-temp 1 --front-band 0.03` (NEW: every head at temperature 2 only for envs near the
frontier, PPO scored per row) - null at the 60.57% stop in 300M, did not hurt the line. `--respawn-margin 1` - null
(the passes were too rare, not the margin). Two trainers on one box - load 66 on 32 threads, ~2-22k steps/s each.

**For the next round:**
* Bare-resume the hand-on checkpoint; ~5.3% of the map is left, and once it finishes the score's time term needs a
  time cost again (time-pen is 0 in this lineage).
* The band novel rows are anchored to the all-time ceiling; anchored to the reservoir's deepest state (or a decaying
  ceiling) they would not starve after one lucky episode - an untested design note.
* Open: the recipe from scratch. The lineage's scratch recipe (config.json `incumbent_recipe`) gained only the
  eval-only `--eval-patience 30 --eval-parallel 1`; --stall-eps 5, keys-only T, greedy rows + stuck, band 0.15 are
  validated warm only.

**Ops:** box 57057785 (RTX 3090, 32-thread 5.1 GHz host, $0.268/h, machine 151065), 13:15 -> 00:01 UTC (~10.8 h,
~$2.9); 226k steps/s training-only. `box.py deploy` on a busy box fails its GPU health check AFTER updating the
checkout (not a bad box - never blocklist on it). py-spy cannot attach in the container (no ptrace). Filter `ps` to your
own worktree: other branches run tests on the same host. A WR-demo download for analysis was refused by the session's
safety classifier; simulator probes from the policy's own states answered the same question.

## Wrap-up (r4b3) - 2026-10-22 00:11 UTC

* Final: RESULT.json `"final": true`, score **94.668** (r4b3_band15x, eval traj_14991491072: 9 episodes, best 94.67%,
  median 60.56%, CP6, 0 finishes; 108/108 valid evals in the run).
* Hand-on checkpoint: /home/steam/ar/runs/r4b3/r4b3_band15x/ckpt_final.pt (15,032,385,536, md5
  cb1dbce40147741433b89728ed69fbd0), pulled by scp and md5-checked before the box was released. Also local:
  r4b3_band15/ckpt_final.pt (13,682,343,936, md5 d78738c7...), r4b3_kst5/ckpt_final.pt, r4b3_ctl/ckpt_latest.pt,
  r4b3_m1/ckpt_final.pt.
* Code: the best lineage ran at commit 3ea2710 (--eval-patience, --eval-parallel, --front-temp,
  --respawn-greedy-stuck; flag-off bit-identical to the round's start); later commits are notes and config.
* Box 57057785 released at 00:01 UTC (`box.py status`: no box).

## Round 5, branch r5b4 - 2026-10-22 01:28 UTC

### Box

* 01:00-01:20 UTC: no offer under the caps (cheapest passing-filter 3090s at $0.375-0.42 vs the 0.30 cap; 0
  boxes live fleet-wide). A background `box.py rent` retry loop was refused by the session's permission
  classifier, so the rent was retried by hand. At ~01:17 the operator raised the tier-1 caps (3090 / 3090 Ti /
  4080 / 5080) to $0.65 in config.json (an uncommitted harness edit in this worktree - not mine, not committed).
* **Box 57138676: RTX 3090, $0.281/h, machine 151065 (Missouri)** - the machine round 4 trained on (226k
  steps/s training-only). READY 01:22, deployed 01:26 (branch + incumbent ckpt, md5 cb1dbce4...).

### The incumbent's state (analysis, no box)

* Its best eval (traj_14991491072, 94.668): ONE episode reaches 94.67% (dies at 110.2 s at (22, -698, 1,689)),
  one 91.7% (stuck against a wall at x -1,340 until the patience cut), FOUR stop at the old 60.56% gate, two
  die at 34.4% (x ~-2,000, z ~9,760), one stays at the first ramp (1.6%). All 9 start at (7,837, 0, 15,042)
  with v = 0; only the spawn yaw differs (347-355 deg, the 8 deg jitter) - chaos does the rest.
* Its last 10M training steps logged `race/success_rate` 0.004-0.008 with finish times 2.6-5.1 s: training
  episodes spawned near the goal (reservoir / greedy / novel rows) started finishing. Novel ceiling d_best
  4,716 u (95.18%); reservoir deepest 94.54% (both y lanes, x ~0-200, z ~1,900-2,070); unstuck T 0.15.
* **The last stretch** (goal field + kill-teleport AABBs, ANALYSIS ONLY): the frontier states sit in a slot
  between kill plate *28 above (z ~2,065, x -224..1,248) and kill plate *39 below (z ~1,666, x -1,840..224).
  The field's value at x 200 jumps from 5,491 (z 1,700) to 4,459 (z 1,600): a lower corridor (z ~1,400-1,600)
  runs east from x ~0 to CP7 (sgcp_7, x 1,632-1,648, z 1,312-1,600) at 95.3% -> 97.0%. At y -600 the slot is
  closed by solid from x ~300-400 (z 1,700-2,000). So the way on is: east in the slot past *39's end (x 224),
  then down into the lower corridor before the wall - a ~100 u window. Then CP7, kill plates *50 (z ~1,313,
  x 1,632-2,432), *61 (z ~1,201, x 2,464-3,424), *3 (z ~660, x 1,983-4,208, |y| <= 704) and the finish box
  (x 2,872-3,048, |y| <= 88, z 176-185; arrival = the ORIGIN's segment crossing it). The field (BFS over
  solids, triggers ignored) cuts straight through *28, *10 and *3 here.
* Probe (local CPU, record_ckpt --spawn-states, the reservoir's 43 states >= 93%): greedy 0/6 past 94.7%
  (two fall onto *39 at x ~40-50 within 1-3 s - too far west; four hover until the 12 s cap).

### r5b4_main (01:27 UTC): bare resume of the incumbent

* `box.py launch --resume --record-every 12.5e6 -- --ckpt-every 1e8`, 5e9 steps budget; every flag restored
  (stall_eps 5, keys-only T cap 0.5, patience 30, side-by-side evals, greedy rows 0.3 + stuck 200, band 0.15,
  ratchet, time-pen 0, surf-bonus 0, int-real-only). Purpose: push the frontier to the first finish.

### 01:40 UTC: the final stretch, measured with the sim's own hull traces and kill hulls (ANALYSIS ONLY)

AABBs of the kill triggers are broadphase only (zones.kill_zones' own warning); these numbers use
`core.trace(p, p, hull)` for solids and `zones.hull_probe` (the trigger's hull-1 clipnodes) for kills.
* **The pocket** (where the frontier states sit, ~94.5%): x 20-200, |y| ~500-700 (the two mirror lanes),
  z ~1,720-2,100, solid walls west / east, a KILL FLOOR at z ~1,700. Its only exit is UP.
* **The wall slit:** east of the pocket a block (x ~220-1,100, z ~1,580-2,080, full width) with a kill layer
  on its top (z ~2,100); on the block a wall at x 480-540 (z 2,120-2,280, full width |y| <= ~700). Over it the
  free origin gap is **z 2,310-2,345 standing (35 u), 2,290-2,365 ducked (75 u)**; the space above the block
  is z 2,100-2,345. The 94.81% greedy episode (15,033M ep 4) climbed out of the pocket at vz +400-600 to an
  apex of ~2,150, hit this wall at x ~480 (vx +994 -> -293) and fell onto the kill layer.
* **The chimney:** past the wall the free region z ~1,840-2,080 (x 840-1,640) connects to the LOWER CORRIDOR
  (z ~1,240-1,560, x 0..2,440) only through a chimney at x ~1,080-1,240 near y 0 (z 1,600-1,800), between kill
  plates (z ~1,800 at x 800-1,050 and 1,300-1,700) and under block *157 (z ~1,900-2,000, kill on top).
* **The corridor and the shaft:** the corridor runs east under the slab past CP7 (x 1,632-1,648) above a kill
  plate (z ~1,300); at x ~2,480+ an open space leads to a funnel (walls at x ~2,760 / 3,080) narrowing to a
  ~120 u SHAFT at x ~2,880-2,960, y ~0, open from z ~1,560 down to the finish (x 2,872-3,048, |y| <= 88, z
  176-185; finish = the origin's segment crossing the hull-inflated box). The kill plates at z ~1,200 and
  ~650-700 have their holes exactly over the shaft.
* The goal field (BFS over solids) follows this route (pocket -> over the wall at z ~2,280 -> x ~840 ->
  chimney at y ~-200 -> corridor -> shaft) and only grazes kill layers (block top at z 2,100).
* Probes (local CPU, record_ckpt --spawn-states from the incumbent's 43 deepest reservoir states, all inside
  the pocket, nearly at rest): greedy 0/6 and stochastic (keys temp 1.15, = training) 0/16 get out - they
  hover or fall onto the kill floor. The exit has to arrive with speed from the ramp before the pocket
  (x ~-1,100..-400, ~1-1.5 s before the wall); the pocket states themselves are dead ends.
* **Training finishes in the incumbent's last steps came from NOVEL rows** (trivial wins): the novel sampler's
  ceiling d_best is the min d over REAL envs only (respawn.py observe), and it stood at 4,716 u (95.18%), so no
  real episode had gone deeper; the 2.6-5.1 s finishes were novel-row episodes placed in the open space above
  the shaft (d >= d_best is allowed anywhere in the field's reachable set).

### 01:37 UTC: r5b4_main - the training ceiling moves past the wall and the chimney

* Evals (score.py best / median): 15,033M 94.806 / 93.0 (the checkpoint), 15,045M 94.732, 15,058M 87.637,
  **15,071M 94.893** / 93.1, 15,083M 94.551 / **94.5** (all 9 reach the pocket), 15,096M 94.488.
* Training ceiling 93,052 -> **94,320 u (96.47%)** at ~15.10B: d 3,448 is the lower corridor at x ~1,200 - a
  REAL episode got over the slit and down the chimney. stall_frac 0.2 -> 0.29-0.41, T 0.09, evals every ~80 s
  (12.5M steps; ~156k steps/s average, ~226k training-only).

### 02:05 UTC: r5b4_main - the greedy line passes the wall slit and the chimney: 98.123, CP7

* Evals 15,108-15,297M (score.py best / median): 94.527, **95.924** (15,121M), 96.132, 96.496, 96.494 (median
  96.3), 96.865, 96.662, 92.347, 96.836 (median 96.6), 96.649, **98.093 (15,234M, CP7)**, **98.123 (15,247M,
  median 96.6%)**, 98.088, 98.119, 98.059, 96.718. So ~130M steps after the training ceiling passed the slit
  (15.10B) the greedy line did too, and ~120M later it was past the chimney, the corridor and CP7.
* Training: ceiling 96,091 u (98.29%), T 0.02, stall 0.15-0.22, `race/success_rate` 0.19-0.26 with finish times
  2.5-2.7 s - NOVEL-row trivial wins: the band admits d >= d_best (1,677 u) anywhere in the field's reachable
  set, which includes the open room high above the finish funnel; a row there falls down the shaft. They teach
  the shaft descent; they are not the route (no real episode has gone past 98.29%).
* **Where the 98.1% episodes stop** (15,247M ep 0 / ep 6): at (2,445-2,469, 370-387, 1,053-1,074) at 116.5-117 s,
  then they fall onto the basement's kill floor (z ~708). Geometry (hull traces, ANALYSIS ONLY): the corridor
  (z 1,240-1,560) ends at a wall at x ~2,440 that spans z ~1,160-1,640; under it a passage leads down into a
  BASEMENT (z ~720-1,120, x 2,000-3,240, kill floor ~700, sloped surfaces at |y| ~300-700). The finish SHAFT
  (x 2,880-2,960, |y| <= ~40, z 120-1,240) is walled off from the basement; it is entered from a FUNNEL ROOM
  above (mouth at z ~1,560: x 2,760-3,120, |y| <= 200, narrowing to the shaft at z ~1,240). The field's route
  from the stop: (2,544, 368, 1,128) -> (2,640, 336, 1,224) -> (2,736, 240, 1,320) -> (2,800, 144, 1,384) ->
  (2,896, 48, 1,352) -> straight down the shaft: a ~310 u CLIMB out of the basement through an opening near
  x 2,500-2,650, |y| ~350, into the funnel room. From the corridor end the route is a U: dip from z ~1,400 to
  ~1,160 under the wall, climb back to ~1,400. Free flight cannot climb, so a basement surface has to turn the
  fall into a rise - the CP2-cavity kind of gate (dip + ramp turnaround), which took ~380M steps once the time
  penalty was gone (round 3).
* Local copy: runs/r5b4/r5b4_main/ckpt_15232663552.pt (md5 c39c37bd...), 14M steps before the first 98.1% eval.

### 03:00 UTC: r5b4_main stopped at 15.79B - the greedy line decayed once T climbed to 0.5; restart at T cap 0.2 (r5b4_t02)

* r5b4_main 15,310-15,813M: best 98.231 (15,448M, median 98.1%, 5 of 9 in the basement, one episode climbing
  out to z ~1,220 at x 2,704 before it died), then the evals decayed: medians 86.5% from ~15.5B and 60.3% by
  15.79B (round 4's old gate), best episodes still 98.1-98.2% now and then. Training ceiling 96,176 -> 96,361 u
  (98.56%: the funnel room - a real episode made the basement climb) and flat after 15.46B.
* The unstuck T: decaying 0.15 -> 0.01 through 15.43B while the line improved; the stuck counter (no +500 u eval
  best since 95,904 u at ~15.25B) passed 200M at 15.44B and T rose 0.14 / 0.26 / 0.39 / 0.5 (cap) by 15.54B -
  the eval medians fell with it. The lineage saw the same (round 3: the line broke at T > ~1.5; round 4 wobbles
  at T 0.5) and lower T caps consolidated lines in rounds 2 and 3. A vicious circle here: the +500 u rule needs
  an eval episode out of the basement (d < 1,364), and T 0.5 keeps the greedy line from getting there.
* approx_kl 0.07-0.4 until ~15.37B, then 1.2-14.7 per update (before T rose, so not T alone). Same anomaly as
  rounds 3/4. Suspect the pitch head: sigma 0.007 with no entropy term, so a pitch-mean shift of 0.03 (z units)
  alone is ~9 nats of log-ratio - and the pitch aims the depth camera. Not verified, not changed.
* **r5b4_t02** (02:57): resume of r5b4_main's snapshot ckpt_15432941568.pt (md5 4485efac..., the weights just
  before the 98.231 eval, T 0.012 there) + `--unstuck-max 0.2 --ckpt-every 1e8`, evals every 12.5M, 5e9 budget.
  Everything else restored. Question: does a T cap of 0.2 hold the 98% line and let it consolidate the basement
  climb that training already makes? Reference: r5b4_main from the same weights at T -> 0.5 (above).
* Ops: `box.py pull` re-tars every eval file each call (1.1 GB by 15.8B, ~60 files of 15-20 MB): replaced by an
  incremental scp of the new files (scratchpad ipull.sh). `pkill -f <pattern>` in a command line that contains the
  pattern killed my own shell once more (CLAUDE.md) - kill by pid.

### 03:36 UTC: r5b4_t02 (T cap 0.2) - 98.236, still no basement exit; the KL anomaly is the PITCH head (measured)

* r5b4_t02 (15,433M -> 15,752M, T 0.2): evals 98.1-98.2% best in most, medians 28-98% (still wobbly), best
  **98.236** (15,698M, median 98.1%). No eval left the basement. T held the line no better than 0.5 did.
* **What the greedy line does in the basement** (15,584-15,597M evals, 98.1% episodes): corridor (x 1,700-2,280,
  z ~1,520 -> 1,320) -> down the passage under the last wall (x ~2,280-2,420, heading +x) -> lands on a ramp at the
  basement floor (z ~770 at x ~2,900, vz -950 -> +74) -> east to x ~3,800-4,150 climbing to z ~1,000-1,100 ->
  U-turn -> back west along y ~-450 climbing (vz +500..+840) -> passes x ~2,600 at z ~880-980 -> climbs back UP THE
  PASSAGE into the corridor (x ~2,000-2,300, z 1,150-1,310) and dies there (~121.5 s). The field's way out is an
  opening at x ~2,450-2,700, |y| ~250-400, z ~1,150-1,350 into the room above the funnel: the return leg of the loop
  passes it ~250 u too low and ~100-200 u too far out in y.
* **The approx_kl anomaly (rounds 3-5: 0.6-22 per update, "not explained")**: the pitch head's sigma sits on the
  global floor exp(LOG_STD_MIN) = 0.0067 (no entropy term; pitch target = -20 + 50 tanh z deg, so 0.3 deg), and the
  joint PPO log-ratio includes its Gaussian: a ~1 deg move of the pitch mean in one update is ~5 nats. NEW
  `--pitch-sigma-min S` (commit b6eb04a; default off, bit-identical when off - tests/python/test_pitch_sigma_min.py,
  7 tests incl. the trainer smoke, a resume restore, the record_ckpt audit and flag-off bit-identity vs the
  pre-change trainer). **Measured on the box: resuming t02's 15,733M snapshot with `--pitch-sigma-min 0.05` took
  approx_kl from 1.1-5.7 (t02's last updates) to 0.020-0.031 on the first updates.** So the reported KL - and the
  clipping of the joint ratio PPO did on every row - was the pitch head. Whether learning gets faster is the arm.
* **r5b4_ps05** (03:33): resume of r5b4_t02/ckpt_15733358592.pt (md5 e3998f36...) + `--pitch-sigma-min 0.05
  --ckpt-every 1e8` (T cap 0.2 and everything else restored), evals every 12.5M. 30-min first test against t02's
  own record from 15.43-15.75B (best 98.1-98.2, medians 28-98, no exit).

### 04:10 UTC: r5b4_ps05 (pitch floor) - KL sane, line more consistent, still no exit; the FINISH FUNNEL and a field leak

* r5b4_ps05 15,733-16,033M (~300M steps): approx_kl 0.019-0.031 throughout (was 1-22); best **98.246** (15,797M,
  median 98.1%); 4 of the last 7 evals had a median >= 96.5% (t02's medians: 28-98%) - the greedy line is more
  consistent, one seed. No eval left the basement; novel ceiling 96,341 -> 96,354 u.
* **The finish geometry** (plan-view hull maps, ANALYSIS ONLY): the shaft (x ~2,910-3,000, y ~-70..50) sits inside a
  tapering FUNNEL whose walls rise to z ~1,550 (inner opening x 2,820-3,150, |y| <= 190 at z 1,500, narrowing to the
  shaft at z ~1,200). Around it a room (x 2,460-3,240+, z 1,180-1,650+) whose FLOOR is a kill layer (z ~1,200), with
  holes down to the basement at x 2,490-2,670, |y| >= 370. So the way in is: rise through a hole from the basement,
  keep rising above the funnel rim (z > ~1,560), drop into the funnel. The greedy loop's return leg passes under the
  holes at z ~880-980 with vz +500-840: its apex would be ~1,420, ~140 u short of the rim, and it heads west.
* **The goal field leaks through the funnel wall**: the walls are one voxel thick and slanted, and the 26-neighbour
  BFS slips diagonally between two solid voxels that touch along an edge. Raw field: just outside the wall at z ~1,416,
  d 1,300; just inside 1,212 (88 u for a wall the player cannot cross); above the rim 1,388. A crop BFS from the
  slab occupancy (old rule reproduces the field to ~30 u) with NO CORNER CUTTING: outside the wall 1,333 -> 1,634,
  (2,800, 180, 1,380) 1,350 -> 1,697, (2,736, 240, 1,320) 1,461 -> 1,742; inside the funnel and above the rim
  unchanged; the basement / corridor +240-290 (a near-constant offset upstream). So the shipped shaping has a false
  minimum against the OUTER funnel wall ~100 u better than the real way over the rim - exactly where training's
  ceiling (98.56%, the room) has sat for ~500M steps.
* NEW (commit 144f9e8, default off): `build_goal_field(..., no_corner_cut=True)` / `_bfs_geodesic(no_corner_cut=)`
  (a diagonal step needs every intermediate voxel of its box free; own cache goalc_, own sig c1_) +
  `tools/bake_nocut_field.py` (bakes it, refuses a disconnected start, compares it with the shipped field along eval
  trajectories). 4 tests (tests/python/test_goal_nocut.py) + the fast-path tests pass. The trainer reads it through
  the existing `--goal-field-file` (the shaping, the potential channel, compass and pot-rate follow it - all
  offset-invariant, so only leak points change). Generic: one grid rule, no map constant.
* `box.py deploy` on the busy box (to get 144f9e8) failed only at gpu_health (GPU busy) after updating the checkout,
  as round 4 noted; the trainer kept running. Bake started on the box GPU beside ps05 (runs/bake_nocut.log).

### 04:38 UTC: the no-corner-cut field baked; r5b4_nc trains on it

* Bake on the box GPU (tools/bake_nocut_field.py, ~3 min beside a trainer): maps/surf_src_sidistic.goalc_32.npz
  (3.6 MB). Start geodesic 97,768 -> **99,675 u** (+1,906, reachable). Along the 98.2% greedy episode (16.00B eval)
  the difference new - old falls in steps from ~1,906 at the start to ~230 at the end (25-35 s, 55 s, 75-80 s: -600,
  105-115 s); 5-second progress ratios new/old 1.00-1.03 on most of the route, 0.85 / 1.07 / 1.13 / 1.08 on four
  segments; the ratchet's cumulative pay along the route 96,027 (old) vs 97,661 u (new) - the same per unit of d0.
  So the corner-cut leaks are a handful of places (slanted one-voxel walls), the funnel among them.
* **r5b4_nc** (04:32): resume of r5b4_ps05/ckpt_16133914624.pt (md5 a7cb65be...) + `--goal-field-file
  maps/surf_src_sidistic.goalc_32.npz --critic-warmup 20 --ckpt-every 1e8` (pitch floor 0.05, T cap 0.2 restored).
* **The observations follow the field too** (the potential channel, compass and pot-rate read slot.reward_field;
  --goal-field-file's help says the obs channel keeps the free-space field - stale): the first two evals, on the
  checkpoint's FROZEN weights (critic warmup), read 86.5 / 86.6% (CP5; the same weights scored 98.1% under the
  old field). ~15M steps after the warmup the line was back at 98.18% (CP7). A transient, not a loss.
* record_ckpt mirrored nothing of --goal-field-file (TRAIN_ONLY) - a recording of such a checkpoint rendered the
  free-space field. Fixed (commit db4823e): the observation channels read the named field, a missing file refuses.
  NOTE for any later resume of an r5b4_nc checkpoint on a NEW box: the field file is not in the maps tarball -
  rebuild it first: `python3 tools/bake_nocut_field.py --map maps/surf_src_sidistic.bsp` (~3 min on the GPU), or
  the trainer refuses ("--goal-field-file: ... not found").

### 04:46 UTC: NEW --race-no-corner-cut 1; r5b4_ncx continues r5b4_nc on it

* `--race-no-corner-cut 1` (commit 686f2ed, default off): bakes / loads `<map>.goalc_<cell>.npz`
  (build_goal_field(no_corner_cut=True), ~3 min on a GPU, then cached) and shapes on it the way --race-kill-aware
  does; refused with --goal-field-file / --race-ground / euclid; config only when on; restored on resume; record_ckpt
  mirrors it. 3 tests (tests/python/test_race_no_corner_cut.py: trainer + resume + recorder, refusal, flag-off
  bit-identity vs the pre-change trainer). Unlike --goal-field-file, a checkpoint carrying it resumes on any box.
* r5b4_nc (16,134-16,210M, --goal-field-file): evals 86.5, 86.6 (frozen weights), then 98.18, 98.14, 98.14, 98.12,
  96.71 - back at the basement; no finish.
* **r5b4_ncx** (04:44): resume of r5b4_nc's checkpoint at 16,209,936,384 (md5 76006b45..., kept on the box as
  runs/r5b4_nc/ckpt_stop.pt) + `--goal-field-file "" --race-no-corner-cut 1 --ckpt-every 1e8` - the SAME field (cache
  hit, start geodesic 99,675 u), now under the robust flag. Pitch floor 0.05 and T cap 0.2 restored.

### 04:52 UTC: the basement climb from the greedy line's own states - random macros never make it (ANALYSIS ONLY)

* States: 1,634 basement states (x > 2,280, z 700-1,250) cut from r5b4_nc / r5b4_ps05 greedy eval episodes (full
  STATE_DTYPE from a pool-row template + the row's origin / velocity / yaw / duck); flying ones |v_h| > 600: 410
  (|v_h| median ~1,110, max 1,538). Raw core, 1,024 envs, view_mode 1, the eval header's physics, kill teleports
  on, finish box armed; each env holds 2-4 random macros (yaw offset U(-110, 110) deg from the velocity heading,
  random strafe / forward keys, 10% jump, 30% duck) for 4-6 s (scratchpad macro/macro_search*.py).
* Result over 3,072 rollouts (seeds 1-3; |v_h| > 600 and > 1,100): **0 finishes, 0 above the funnel rim, 0 with the
  corrected-field d < 1,400**; the highest any rollout got in the funnel's x range was z 1,254-1,340 (the room's kill
  floor is at ~1,200, the rim at ~1,560); best d 1,799-1,915 (the hole itself is ~2,000). So from where the greedy
  line is now, the climb is not reachable by naive steering: it needs a precise surf UP a basement ramp (energy-wise
  |v| 1,500 at z 1,000 could reach z ~2,400) - or a faster line into the basement. Training's ceiling (the room at
  z ~1,320, 98.56% old units) says real episodes rarely get partway.
* For the next round: this is the gate. Ideas not tried: carry more speed through the corridor into the basement
  (nothing pays speed with time-pen 0); keys temperature higher only for envs in the frontier band (--front-temp on
  keys only); the WR demo shows the human line here (analysis only - round 4's download was refused by the session
  classifier).

### 05:48 UTC: the basement climb is a CAN'T for the current policy, speed or not; r5b4_t06 (keys T cap 0.6)

* r5b4_ncx 16,210-16,764M (~550M steps on the no-corner-cut field): evals 95.8-98.24% best, medians 22-98%, no
  basement exit; the novel ceiling never moved (no real episode above the funnel rim).
* **The holes into the funnel room** (hull maps): two ~180 x 300 u openings in the room's floor at x 2,490-2,670,
  |y| 400-680, against the room's WEST wall (x ~2,460); clear vertical passage from z 1,050 to 1,350. The greedy loop's
  return leg passes under the -y hole heading WEST (into that wall); the first pass crosses under the +y hole falling.
* Random macros from the basement states with velocities x1.2 / x1.4 / x1.6: max z in the funnel's x range exactly
  1,166 every time (the basement ceiling) - the rollouts hit the ceiling instead of finding the holes.
* **The POLICY itself** (r5b4_ncx 16.61B snapshot, record_ckpt --spawn-states, stochastic at the training keys
  temperature 1.2) from 350 flying basement states: x1.0 speed 0/16 finishes (best 98.24%), x1.4 speed 0/16 (best
  98.15%); many episodes stop on basement ledges (v ~0 at z ~850-1,040). Speed is not the missing piece; the climb
  (aim at a hole with vz up, heading east / inward) is a skill the policy does not have. `--respawn-speed` (built for
  speed-gated jumps) is therefore not the arm.
* **r5b4_t06** (05:52): resume of r5b4_ncx at 16,764,108,800 (md5 10d7ed86...; snapshot ckpt_16710631424 also on the
  box) + `--unstuck-max 0.6 --ckpt-every 1e8` (keys-only temperature cap 0.2 -> 0.6; T rises to it within ~80M steps
  since the stuck counter is far past 200M). The score is the max over evals and the 98.2% episodes survived T 0.5
  in r5b4_main, so more key exploration in the basement costs little; with the pitch floor the KL stays sane.

### 06:16 UTC: T 0.6 hurts; Go-Explore from own states tops out in the room; the room's EAST WALL IS A RAMP

* **r5b4_t06** (keys T cap 0.6, 16,764-16,990M): within ~230M steps the best eval episodes fell from 98.2% (CP7,
  the basement) to 96.7-97.4% (CP6 - the greedy line stops in the corridor). Stopped; the lineage's high-T lesson
  again. **r5b4_ncy** (06:13): bare resume of r5b4_ncx at 16,764,108,800 (T cap 0.2, nocut field, pitch floor).
* **Go-Explore in the simulator** (scratchpad macro/goexplore.py; archive cells 64 u x vz sign x |v|/400, rooted at
  1,634 of the policy's own basement states, count + frontier-biased selection, random held macros, 200 iterations x
  1,024 x 1.6 s, 25,764 cells): no finish, no rim crossing; best corrected-field d 1,990 -> 1,745 (plateau by
  iteration ~20), highest state in the room z 1,353. Same place as training's ceiling.
* **The room's geometry, completed** (hull maps x 2,400-5,160): the room (above the basement, x 2,460-3,420, |y| <=
  680, floor = kill layer at ~1,200 except the two holes at x 2,490-2,670, |y| 400-680 and the funnel) is bounded on
  the EAST by a SLOPED wall: x ~3,240 at z 1,250 -> ~3,300 at 1,300 -> ~3,360 at 1,350 -> ~3,420 at z 1,400-1,550,
  i.e. a ~40 deg surface facing west-up = a surf ramp. Likely route (not verified): up through a hole heading EAST,
  across the room at z ~1,300, up the east ramp past the funnel rim (z ~1,560), curve back west over the funnel
  (x 2,760-3,150, |y| <= 190) and drop in. That explains why naive steering never climbs: the height comes from
  surfing the east ramp, and both the greedy loop's passes under the holes head the wrong way (the first pass falls
  heading east, the return leg rises heading west into the room's west wall).

### 06:44 UTC: r5b4_ncy - 98.349, a greedy episode up through the basement hole; the surf-macro search also stalls

* r5b4_ncy (T cap 0.2 again, 16,764-16,984M): the line is consistent (medians 98.1% in most evals); **98.349**
  (16,903M) - episode 5 rose through the -y hole to (2,631, -351, 1,220) at 123.7 s, above the room's floor, and died
  on its kill layer at the hole's edge; 98.272 (16,878M). First greedy episode out of the basement.
* Surf-macro Go-Explore (small yaw offsets 0-30 deg with the matching strafe key, 75%; free turns 25%), seeded with the
  first archive's 4,000 best cells, 220 iterations: no finish; best d 1,699; the highest room states (z ~1,353) sit
  right above the -y hole at ~100 u/s (they rose slowly through it and stalled); one state ever got east of x 3,100
  (z 1,222). Naive simulator search does not produce the fast eastward rise across the room to its east ramp.

### 07:31 UTC: CEM over macro sequences from the greedy line's corridor state - no route; r5b4_ncz (evals every 6.25M)

* CEM (scratchpad macro/cem_route.py): 10 held macros (yaw offset, side, fwd, duck, duration), population 1,024, 64
  elites, 40 iterations, 7 s rollouts on the raw core from the greedy line's own state in the corridor ~1 s before
  the basement ((1,852, 431, 1,514), v (444, -1,116, -26), corrected d 3,047): best d 1,991 - it never got into the
  room (the Go-Explore archive reached 1,699 from basement states). With the two Go-Explore runs and the macro and
  policy probes, ~10k rollouts from the policy's own states have found no way over the funnel rim.
* r5b4_ncy 16,764-17,419M: evals 96.7-98.35, best 98.349 (16,903M), no further hole passage. The line is consistent;
  what moves the score now is sampling it: **r5b4_ncz** (07:31) = bare resume of r5b4_ncy at 17,418,944,512 (md5
  7256a1cf...) with `--record-every 6.25e6` (~60% more evals for ~20% less training).

### 08:06 UTC: the east ramp is a SPEED-GATED route (feasibility, analysis only); r5b4_rs15 (--respawn-speed 1.0 1.5)

* Hand-placed ANALYSIS-ONLY states (never trained on): rising out of a basement hole (x 2,520-2,640, |y| 420-640,
  z 1,230, vz +200..+700) heading east within +-35 deg, 1,024 random surf-like macro rollouts per speed:
  900 u/s: 0 above the rim, 0 finishes; 1,300: 149 above the rim (z > 1,560), 0 finishes; 1,700: 416 above,
  1 finish; 2,100: 535 above, 4 finishes (max z 2,357). So the room's east ramp lifts an agent over the funnel rim
  when it leaves a hole eastward at ~1,700+ u/s, and the drop into the funnel is then a precision step (random
  steering finishes 0.1-0.4%). The greedy line's basement speeds are ~800-1,540 u/s and its passes under the holes
  head the wrong way.
* That is what `--respawn-speed` exists for (respawn.py: "speed-gated jumps can be practiced at make-it speed before
  the policy has learned to CARRY that speed - the value of the boosted states then pulls the upstream line
  faster"): it scales the reservoir rows' velocity by U(lo, hi) (default 0.9-1.1; real episodes, so a boosted
  episode that clears the rim moves the novel ceiling). Generic: one range, no map quantity.
* **r5b4_rs15** (08:05): resume of r5b4_ncz at 17,633,902,592 (md5 837794dd...) + `--respawn-speed 1.0 1.5`, evals
  every 6.25M, everything else restored (T cap 0.2, nocut field, pitch floor). Watch: the novel ceiling (d_best below
  1,415 = a real episode over the rim), the greedy line's 98% reach (boosted training could disturb the earlier
  precision gates).

### 09:39 UTC: per-run bests (score.py --best); r5b4_rs15 so far

* main 98.231 | t02 98.236 | ps05 98.249 | nc 98.176 | ncx 98.238 | **ncy 98.349** (traj_16903045120) | ncz 98.261 |
  t06 98.218 | rs15 98.344 (traj_17728798720, ~18.19B). 0 finishes anywhere.
* r5b4_rs15 (`--respawn-speed 1.0 1.5`, 17.63-18.19B, ~560M steps): the greedy line kept its 98% reach (evals 96.7-98.34,
  medians up to 98.2%); the novel ceiling never moved (no real episode over the funnel rim). Boosted respawns alone
  did not open the speed-gated climb in this window.

## ROUND 5 LESSON (r5b4) - written at wrap-up, 2026-10-22 12:24 UTC

**Result:** best greedy eval from the true start **98.349** (r5b4_ncy at 16,903,045,120; incumbent 94.668): CP7, the
basement under the last wall, one episode up through a basement hole into the room above the finish funnel. 0 finishes.
Hand-on checkpoint: /home/steam/ar/runs/r5b4/r5b4_ncy/ckpt_latest.pt (17,418,944,512, md5 7256a1cf...); its config
carries race_no_corner_cut=1 (the trainer bakes maps/surf_src_sidistic.goalc_32.npz itself on a new box, ~3 min on the
GPU), pitch_sigma_min=0.05 and unstuck_max=0.2 - a bare resume restores all three.

**Per-run bests:** main 98.231 | t02 98.236 | ps05 98.249 | nc 98.176 | ncx 98.238 | **ncy 98.349** | ncz 98.261 | t06 98.218
| rs15 98.344 (to 19.29B).

**What worked:**
1. A bare resume of the incumbent: the greedy line passed the wall slit (35 u standing gap over a wall above a kill
   layer) and the chimney ~130-200M steps after training's ceiling did, then CP7 and the corridor (98.23% by 15.45B).
2. `--unstuck-max 0.2` after the T 0.5 cap had decayed the line (medians 98 -> 60%); T 0.6 later pulled the line back
   to 96.7% within ~230M steps - keep T low at this end of the map.
3. **NEW `--pitch-sigma-min S`** (default off, bit-identical; record_ckpt mirrors it): the rounds-3-5 "unexplained"
   approx_kl of 0.6-22 per update was the pitch head - its sigma sat on the global floor 0.0067 with no entropy term,
   so a ~1 deg move of the camera-aim mean was several nats of the joint PPO ratio. Floored at 0.05: kl 1-6 -> 0.02-0.03
   at once, and the greedy line's medians were steadier (4 of 7 evals >= 96.5% vs 28-98% before). Keep it.
4. **NEW `--race-no-corner-cut 1`** (+ `build_goal_field(no_corner_cut=True)`, default off, bit-identical; record_ckpt
   mirrors it, and now also mirrors `--goal-field-file`, which it rendered wrong before): the 26-neighbour goal-field
   BFS slips diagonally through one-voxel-thick slanted walls; at the finish funnel that made the outside of its wall
   ~300 u "closer" than the real way over its rim. The corrected field changes the learned route's shaping by a few
   percent on four segments only (5-s progress ratios 0.85-1.13, cumulative pay 96,027 vs 97,661 u); its
   observations transient cost 2 evals (86.5%), recovered in ~15M steps. The best eval (98.349) came on it.

**The last gate, measured (analysis only - hull traces, true kill hulls, simulator probes):** after CP7 the corridor
ends at a wall; the way on drops into a BASEMENT (kill floor ~700) and must rise again through one of two holes
(x 2,490-2,670, |y| 400-680) into a ROOM whose floor is a kill layer (~1,200); the finish SHAFT sits inside a tapering
FUNNEL whose rim is at z ~1,560, and the room's EAST wall is a ~40 deg ramp (x 3,240 at z 1,250 -> 3,420 at 1,400).
From analysis states leaving a hole eastward, the ramp lifts an agent over the rim at >= ~1,300 u/s and finishes
appear at ~1,700+ u/s (1 / 1,024 random-macro rollouts at 1,700, 4 at 2,100). The greedy line's basement speeds are
800-1,540 u/s and both its passes under the holes head the wrong way (east while falling, west while rising).

**What did not work:** ~10k simulator rollouts from the policy's own states (random macros incl. x1.2-1.6 speed,
two Go-Explore runs with 25k-cell archives, surf-like macros, CEM over 10-macro sequences from the corridor) - none
over the rim (best: into the room at z 1,353); the policy itself from its basement states at x1.0 / x1.4 speed:
0/32; keys T cap 0.6 (line regressed); `--respawn-speed 1.0 1.5` (built for speed-gated jumps; ~1.66B steps, line
held at 98.2-98.34%, ceiling never moved).

**For the next round:** the finish is one speed-gated maneuver away. Ideas not tried: carry speed - nothing pays it
with time-pen 0 (a small time cost once the line is reliable, or a speed-aware term that is generic); SIL restricted
to REAL episodes (the shipped buffer would be flooded by the 20-30% novel-row trivial wins above the funnel); keys-only
frontier temperature; more steps at T 0.2 (each earlier gate took 100M-1.6B steps). Evals every 6.25-12.5M sample the
max well; the line is consistent at ~98.2% (medians 98.1%).

**Ops:** box 57138676 (RTX 3090, machine 151065 Missouri, $0.281/h), 01:22-12:2x UTC (~11 h, ~$3.1); 145-150k steps/s
with evals every 12.5M (110k at 6.25M). The operator raised the tier-1 price caps to $0.65 at ~01:17 (config.json,
uncommitted, not mine); before that no offer passed and a background rent loop was refused by the session classifier.
`box.py pull` re-tars every eval each call (1+ GB late in a run) - pull new evals incrementally by scp. `box.py deploy`
on a busy box fails only at the GPU health check after updating the checkout (not a bad box). Never `pkill -f` a
pattern that is in your own command line.

## Wrap-up (r5b4) - 2026-10-22 12:24 UTC

* Final: RESULT.json `"final": true`, score **98.349** (r5b4_ncy, eval traj_16903045120: 9 episodes, best 98.35%, median
  98.1%, 0 finishes, CP7; re-scored at wrap-up).
* Hand-on checkpoint: /home/steam/ar/runs/r5b4/r5b4_ncy/ckpt_latest.pt (17,418,944,512, md5
  7256a1cf0844cb05c6b3ecda512e8a62, md5-checked against the box at wrap-up). Also local: r5b4_ps05/ckpt_latest.pt
  (16,186,867,712), r5b4_main/ckpt_15232663552.pt.
* Code: commit 686f2ed (--pitch-sigma-min, --race-no-corner-cut, record_ckpt mirrors) is the last code change; later
  commits are notes. Box 57138676 released at wrap-up.

## Round 6, branch r6b4 - 2026-10-22 14:05 UTC

### Box

* 13:41-14:05 UTC: no single-GPU offer passes box.py (cheapest 4090 $0.72 vs cap 0.45, 5090 $0.87 vs
  0.60, 3090 $1.33; the one 5080 under its cap, machine 150677 at $0.40, is blocklisted for a foreign
  tenant). A background `box.py rent` retry loop (every 3 min, scratchpad work/rent_loop.sh) runs.

### The human's finish, from the WR demo (47250; ANALYSIS ONLY - CLAUDE.md section 0)

Parsed with tools/demo/parse_hldemo.py (a local copy of the public file; nothing from it enters a run).
Demo seconds (the timer starts ~2 s in); positions are hull centres like SurfState.origin.

| t (s) | position | velocity | what |
|---|---|---|---|
| 112.50 | (2,544, -393, 969) | (957, -590, -922) | down the passage into the basement |
| 113.00 | (3,218, -586, 764) | (1,537, -76, -81) | basement, east, LOW |
| 113.50-114.25 | (3,923, -394, 883) -> (4,044, -59, 912) -> (3,694, 550, 818) | \|v_h\| 1,470-1,495 | a WIDE, LOW U-turn at the east end |
| 114.50-114.75 | (3,315, 588, 763) -> (2,931, 570, 781) | (-1,543, 43, -45) | west on the +y OUTER lane, z ~765, 1,545 u/s, strafe-left held |
| 114.65-114.99 | (3,066, 583, 767) -> (2,709, 407, 954) | -> (-206, -955, +1,011) | a curved KICK RAMP: the velocity rotates west -> south and up in 0.35 s |
| 115.02-115.39 | (2,700, 372, 992) -> (2,590, 554, 1,300) | -> (553, 731, 700) | slides round a curved surface (dv 50-120 per 7.6 ms frame, beyond air-accel), turning NORTH into the +y hole, rising |
| 115.50-116.25 | (2,676, 594, 1,370) -> (2,958, 53, 1,603) | | across the room east then south, over the funnel rim (z 1,575 at 116.0) |
| 116.50-118.25 | (2,959, -49, 1,581) -> (2,932, 72, 24) | \|v_h\| ~70 | stops dead over the funnel and falls down the shaft; finish |

The incumbent's greedy line (traj_16903045120, eps 1-2), same stretch: east leg at 1,420-1,460 u/s
(human 1,538), then turns ~360 u EARLIER (x 3,655-3,728 vs 4,044) and CLIMBS a surface to z 1,136-1,146
(human 883-912) losing ~300 u/s; west leg INSIDE (y 350-540 vs 570-590) and HIGHER (z 880-1,016 at
x ~2,800 vs ~780) at 1,200-1,300 u/s; it misses the kick ramp, gets its kick at x ~2,130-2,500 (past
the hole) and rises into the passage instead (z <= 1,241 at x ~2,400). So the gap is energy and line:
a wide, low east-end turn that keeps ~1,500 u/s, the outer-low west lane onto the kick ramp, then the
in-room turn to the funnel and the stop.

### The greedy-line pool is mostly basement hovering (measured on the incumbent checkpoint)

* ck["respawn_greedy"]: 3,847 rows from 9 failed episodes (best-progress stops at 114.8, 115.1, 115.4,
  115.7, 123.7, 136.7, 139.8 s; two early failures at 25.2 / 53.0 s). 49% lie in the basement, but
  their MEDIAN horizontal speed is 63 u/s: they are `--respawn-greedy-stuck 200`'s stalled stretches
  (greedy episodes hovering on basement ledges for the 30 s until --eval-patience cuts them). Moving
  basement rows: east leg 6%, west leg 7% (|v_h| median 966, p90 1,395), U-turn 3%.

### NEW `--respawn-greedy-speed LO HI` (default off)

* Each greedy-line spawn row's velocity x U(LO, HI) when it is mixed into the pool - what
  `--respawn-speed` does to the RESERVOIR rows. The greedy rows are mixed in after the reservoir's
  build_pool, so round 5's rs15 (`--respawn-speed 1.0 1.5`, ~1.66B steps, null) never boosted them:
  it boosted reservoir rows spread over the whole map (a few % of them in the basement).
* Own RNG (seed + 7): the rows drawn are exactly the rows drawn without the flag. Config key only
  when on; restored on resume; refused without --respawn-greedy; TRAIN_ONLY in record_ckpt.
  Tests: tests/python/test_respawn_greedy_speed.py (pool: same rows, velocity scaled in [LO, HI]
  per row, direction kept, off = no draw; bad ranges refused; trainer wiring + bare-resume restore +
  refusal; flag-off bit-identity vs the trainer before the change).
* Generic: one range, no map quantity; the policy's own states.
* Planned arm 1 (warm, at the wall): incumbent + `--respawn-greedy-speed 1.0 1.5
  --respawn-greedy-stuck 0` (the stalled stretches are at rest, a boost does nothing to them, and
  dropping them roughly doubles the share of moving approach states in the pool). Two knobs: the
  attribution is to the pair. Control = round 5's own continuations of this lineage at T 0.2
  (~2B steps: evals 98.2-98.35, novel ceiling flat).

### 14:45 UTC: the human's finish replays EXACTLY in our simulator; the basement is a field dip

* `SurfCore.play_step` with the demo's own usercmds (yaw, fwd/side, buttons, msec per frame) from the
  demo state at 114.6 s reproduces the trajectory to 0.1-0.2 u for 1 s and goes down the finish shaft;
  from 112.0 / 113.5 s the error stays <= 3-4 u to the finish. So the whole last stretch is plain
  physics in our core (the sharp turn into the hole at 115.03-115.40 s is air-strafing with the wish
  direction against the velocity, not a booster - the bsp has no trigger_push). The kick surface is in
  our bsp where the human rides it: a 43 deg plane, normal (0, +0.73, +0.69), touched from 114.78 to
  114.94 s; the human strafes INTO it (yaw rotating -178 -> -104 deg with strafe-left held) and is
  launched off its top edge at (2,709, 407, 954) with vz +1,011.
* Old-field progress along the lines (% of d0): the basement is a DIP - entry 98.02%, the human's
  east-end U-turn 96.62% (d 3,306), the policy's 96.94-97.07%; the human's kick launch 98.06%, the
  hole 98.23%, over the funnel 98.47-98.62%. The policy's best (98.35%, ep 5 of the incumbent eval)
  came on its SECOND basement loop, up the -y hole.
* The incumbent's ep 5 rides the human's outer-low lane ((3,119, 585, 765) at 116.4 s - the human's
  point at 114.6 s) but heading EAST (it enters on the +y side); its U-turn (x <= 3,811, climbing to
  z 1,054) brings it back west INSIDE and HIGH on the -y side (y -390..-423, z 937-994). The mirror of
  the human's line would come back on the -y outer-low lane (y ~-585, z ~765).

### 14:58 UTC: box 57226494 and arm 1 (r6b4_gs15)

* Box 57226494: RTX 3090 on an AMD EPYC 7502 (64 threads, quota 61.4 CPUs), $0.35/h, machine 141910
  (Thailand); the rent loop got it at 14:41 after ~1 h of no passing offers. Deploy 2 min (ckpt md5
  7256a1cf... verified).
* **Ops: the first launch's record gate TIMED OUT (420 s)**: the incumbent checkpoint carries
  race_no_corner_cut=1, so record_ckpt bakes maps/surf_src_sidistic.goalc_32.npz on a new box before
  its first recording, and that bake ran past the gate's timeout. Bake it first on a fresh box:
  `python3 tools/bake_nocut_field.py --map maps/surf_src_sidistic.bsp` (cached after; here the gate's
  own bake had just finished), then launch.
* **r6b4_gs15** (14:50): warm resume of the incumbent (17,418,944,512) + `--respawn-greedy-speed 1.0
  1.5 --respawn-greedy-stuck 0 --ckpt-every 1e8`, evals every 6.25M, budget 3e9. Everything else
  restored (T cap 0.2 keys, nocut field, pitch floor, greedy 0.3 / 0.25-2.0 s, novel band 0.15, ...).
  Greedy pool 2,256 states (was 3,847 with the stalled stretches), 1,106 rows per pool. ~130k steps/s
  after startup. Read-outs: the reservoir's min depth (2.013% = d 2,006: states in the hole / room
  harvested would lower it), the novel ceiling (98,260 = d 1,414: a real episode over the rim), real
  finishes, and the greedy evals.

### 15:25 UTC: the bench - the policy can finish from the hole; it lacks the kick-and-turn (MEASUREMENT ONLY)

record_ckpt --spawn-states from the WR demo's states (scratchpad work/end_bench.py; the nocut field copied
from the box into maps/ - gitignored, same bsp signature), incumbent checkpoint, 7 s episodes, greedy x3 +
stochastic (keys temp 1.2 = training) x6 per set:

| start set (demo s) | greedy finish | stochastic finish | what it does |
|---|---|---|---|
| over the funnel rim (115.95-116.25) | 2/3 | 5/6 | drops in (the novel-row trivial wins taught it) |
| in the +y hole, rising (115.2-115.3; vz 810-900) | **3/3** | **6/6** | flies the room to the funnel, finishes in ~2.8 s |
| just after the kick (114.95-115.05; vz ~1,000, heading SW) | 0/3 | 0/6 | never turns north into the hole: hits the basement ceiling (z 1,147-1,171) |
| west leg, outer-low lane (114.45-114.8) | 0/3 | 0/6 | no kick |

* From the post-kick states, more noise does not find the turn either: keys temperature 2 and 3, yaw sigma x2:
  0/30 into the hole. The human's turn there is a held strafe-right with the view ~4 deg off the velocity
  (wishdir ~94 deg from it: up to ~100 u/s per frame of lateral acceleration) - a small action the policy's
  yaw mean simply does not choose there.
* But the policy's OWN line already flies the move's shape: the incumbent's 98.349 episode (ep 5), second
  basement loop at 122.4-123.6 s, climbs the kick ramp inward (vy +170..+560, vz up to +589 at x ~2,360),
  air-strafes back outward and rises to the -y hole's inner edge, apex z 1,226 (dies on the room floor's
  kill layer). Its west-leg speed is ~1,000 u/s against the human's ~1,545, so its launch vz is ~590
  instead of ~1,010 (apex ~1,226 instead of ~1,600; the rim is ~1,560). Energy, not shape.

### 15:25 UTC: arm 1 (r6b4_gs15) - the greedy line degraded; stopped

* Evals (score.py, on the box): 98.222, 98.236, 98.161, 98.234, 98.126, 98.137, 98.051 (to 17,457M), then
  96.838, 87.728, 96.804, 97.153, 96.907, 98.090, 96.729 (to 17,501M): new failures upstream - greedy deaths
  at ~98 s (CP5, 87.7%) and best-progress stops at ~110 s (CP6, 96.8%). Reservoir min depth 2.013 -> 1.906%
  (briefly) -> 2.06%; novel ceiling flat. Best 98.236 (< incumbent 98.349).
* Likely cause: the boost hit EVERY greedy row, including the windows cut before the line's early failures
  (25-75 s); those are the DAgger corrections that keep the upstream line, now practised 0-50% faster. So:
* NEW `--respawn-greedy-speed-band B` (default 0 = every row): only the rows within B x d0 of the pool's
  DEEPEST row are scaled (d0 = the field at the eval episodes' own start; the mask is rebuilt with the pool
  and saved with it). The early-failure windows stay plain DAgger rows. Tests added to
  tests/python/test_respawn_greedy_speed.py.
