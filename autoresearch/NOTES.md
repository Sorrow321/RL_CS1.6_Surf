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

## Round 2, branch r2b3 - 2026-10-20 13:01 UTC (box 56893794, RTX 5080, $0.388/h, machine 150677)

### The frontier, precisely (analysis only: voxel grid, kill AABBs from the bsp, the field)

* The incumbent's best eval (4,917M, ep 1, 28.63%) does NOT stop at a plain wall: past the pit
  floor's west edge it drops into a 256 u shaft (x 2,240-2,496, open from z ~10,700 to the
  slot), hits the wall at (2,224, 1,002, 11,360) at 28.3 s, bounces and dies at
  (2,139, 922, 10,980) - INSIDE a window in that wall, on its sill plate.
* The wall at x ~1,950-2,210 is solid from z 10,656 to 11,520 at every y except two windows,
  y +-(864..960), z ~10,976-11,168 (~96 x 192 u in the 32 u voxels). Their sills are kill
  plates *17 / *113 (x 1,952-2,208, z 10,944-10,958). For a standing hull the origin window is
  ~64 u (y) x ~120 u (z). Kill plates *80 / *64 (x 2,528-2,592, y +-(832-960), z 11,392) cap
  the column at the shaft's east edge - ep 3/4/7/8 of that eval die there or on the pit floor
  (*86) just east of it. The shaft's bottom is kill *21 (z 10,688).
* Past the windows: a room x ~1,216-1,952, y +-1,120, z 10,632-11,168 over a kill floor
  (*70, top z 10,632). Its west wall carries a ~45 deg ramp face (y +-352, x 1,440 at z 10,660
  -> x ~1,300 at z 10,800). CP2 (x 768-816) is beyond that wall; the passage the field takes is
  low (z ~10,100-10,370 under the wall block, x 864-1,280).
* Field progress: window 28.58%, room 28.8-29.2%, CP2 30.9%. The novel ceiling of the
  incumbent (deepest REAL training episode) is 28,174 u = 28.82%: training episodes DO get
  through a window into the room, and die there (within ~0.2-0.5 s).
* So the wall is a precision maneuver (shed ~350 u of height in the 368 u between the column
  and the wall, then thread a ~64 x 120 u window) followed by a room whose floor is an
  invisible kill. The respawn margin (2 s) keeps every state within 2 s of a death out of the
  reservoir, so no training episode ever STARTS at the window or in the room.
* The incumbent run at its end: T at the cap 2 since ~2.4B, unstuck best 27,916 u (eval),
  novel ceiling 28,174 u, reservoir (20k states) d min 70,427 / p10 72,728 / median 78,979.
  Control for every warm arm here: r1b1's own continuation ratnovx3 (5.1-5.9B): best 28.467,
  means 23-28%, no window crossing in ~800M steps.

### r2b3_sil: warm + `--sil-coef 0.1` (Oh et al. 2018, existing flag) - launched 12:58 UTC

* Resume of the incumbent ckpt (5,100,797,952, md5 3cc658ed...), 450M steps (to 5.55B), all
  else restored. Measuring AT the wall: do the rare window-threading training episodes (the
  ones that beat V) become the greedy line?
* Caveat seen at once: with 1024 envs x 128 decisions per rollout, ~131k rows qualify and the
  50k buffer is REFILLED every iteration (`sil 50,000 rows +50000`), so this is
  advantage-filtered imitation of the latest rollout, not a long memory of rare successes.

### r2b3_sil result (13:08 UTC): NEGATIVE, stopped at 5.20B (99M steps)

* Evals (score.py): 5,101M (the checkpoint's own first eval) 28.200 (best 28.2%, median
  27.6%); 5,177M **27.529** (best 27.5%, median 25.9%, three greedy episodes now die at
  15-17 s on the first half of the map).
* Training side, monotone in the wrong direction within 75M steps: episode length ~950 ->
  ~610 decisions, reservoir min-depth 72.03% -> 73.35% (the deepest harvested state got
  SHALLOWER), the policy's own yaw sigma 0.032 -> 0.046. SIL scores the stored actions under
  the UNTEMPERED policy while they were drawn at T = 2 (sigma x 3), so its log-likelihood term
  pulls sigma up; and the 50k buffer is refilled by every rollout (top 50k of ~131k qualifying
  rows), so it is short-memory advantage-filtered imitation, not a replay of rare successes.
* Not worth repeating as is under --unstuck T = 2. A frontier-filtered, temperature-aware SIL
  would be a different mechanism.

### r2b3_m05: warm + `--respawn-margin 0.5` - launched 13:07 UTC

* Same resume, 450M steps (to 5.55B). The reservoir's snapshots are 1 s apart, so with the
  2 s margin the deepest harvested state is 2-3 s before a death (the slot at 26-27 s for an
  episode that dies at the window); with 0.5 s it is 0.5-1.5 s before (the slot's end, right
  before the shaft). Question: do respawns at the maneuver make the greedy line thread the
  window, and do training episodes go deeper into the room (novel ceiling > 28,174 u)?

### r2b3_m05 result (13:31 UTC): no window crossing, stopped at 5.44B (337M steps)

* Evals (best / median %): 5,177M 27.4/27.4, 5,252M 28.2/27.9, 5,328M 27.4/27.4 (most
  episodes back on the OLD pit dive, -y lane, 24.7 s), 5,403M 27.9/27.3. No episode passes
  the window; the novel ceiling stayed 28,174 u (no real episode went deeper).
* Training side healthy (len 900-980, sigma 0.031); the reservoir's deepest state moved
  72.03% -> 71.70% of d0 (progress 28.3% - the shaft): the reservoir now holds the maneuver's
  start, and it still did not produce a crossing in ~300M steps. Margin alone: no.

### CORRECTION: the room past the window is not a kill floor (kill shapes from the hull-1 clip tree)

* The kill AABBs are broadphase boxes; `zones.hull_probe` gives the real shapes. *70 is only a
  plate at z ~10,000-10,080; *20 is a plate at z ~10,400-10,470 spanning x 864-1,756 (the
  room's floor west of x ~1,756, over a thin solid slab at z 10,404).
* Under the shaft and the wall block a long surf ramp descends WEST: (2,464, 10,608) ->
  (1,760, 9,968) (32 u voxels, both lanes and y = 0), under the room's floor slab, into an open
  region (z ~9,970-10,380) that runs west under the wall block to the CP2 curtain
  (x 768-816, z 10,060-11,300).
* So the line past the slot is: shed height in the shaft (its bottom is kill *21 at z 10,690),
  thread a window (z ~10,976-11,168, y +-(864-960)), drop steeply in the room's EAST part
  (x 1,760-1,890, where *20 does not reach) onto the ramp, surf it down west, fly west under
  the room to CP2 (30.9%). Training episodes reach 28.82% (x ~1,900 just inside the window)
  and die - most likely flying too far west over *20 before they are low enough.
  Renders: scratchpad sideview2.py (hull-1 kill containment), analysis only.

### r2b3_t0: warm + `--unstuck-max 0` (the exploration temperature off) - launched 13:30 UTC

* Same resume. T restored at 2 is capped to 0: sampling untempered, entropy coef 0.005 (from
  0.015), intrinsic coef 0.25 (from 0.75), no count decay. Question: is T = 2 noise what keeps
  the line from the precise drop/window (precision), or is it what found the CP2 line
  (exploration)? First updates: approx_kl 1.0-3.1 (the untempered heads are ~3x sharper than
  what the rollouts sampled for 2.7B steps).

### r2b3_t0 result (13:57 UTC): NEGATIVE - without the temperature the CP2 line is lost

* 450M steps (5.10 -> 5.55B, 280k steps/s). Evals (best / median %): 5,177M 27.4/27.2, then
  5,252M, 5,328M, 5,403M, 5,478M all 27.4/27.4: every greedy episode back on the OLD pit dive.
  approx_kl 0.4-3.1 per update for the first ~100M steps (the untempered heads are ~3x sharper
  than what T = 2 sampled), the policy's own yaw sigma 0.032 -> 0.069, intrinsic 3 -> 0.6 /ep.
* So T = 2 (unstuck's cap) is what keeps the human CP2 line alive in this lineage; precision
  is not bought by turning it off. The box was released at 13:59 (56893794, ~1.1 h, ~$0.43).

### The window is a DEAD END from the slot (collision traces + 21k random rollouts, analysis only)

* Window clearance by `core.trace` (startsolid): origin y 852-940, z 10,992-11,152 standing
  (10,976-11,168 ducked), and the tunnel through the wall is 256 u long (x 1,952-2,208). The
  slot's exit to the shaft is a gap in a thin wall at x ~2,530-2,590, y +-(860-960), whose sill
  carries kill *80/*64 at z 11,392: the hull must cross it above z ~11,412 (ducked).
* Free fall cannot connect the two: leaving the gap at z >= 11,412 and reaching the tunnel mouth
  288 u later at z <= 11,168 needs a mean slope >= 0.85, and then the 288 u through the tunnel
  drop >= 244 u more against the 192 u the tunnel allows - for any horizontal speed, even
  ducked (the parabola only makes the second half worse). Nothing in the shaft can lift the
  player (vertical walls only clip).
* Random shooting (scratchpad `autoresearch/analysis/sid_slot_shoot.py`, held usercmd
  primitives through `SurfCore.play_step` with the live triggers, from the incumbent's OWN
  greedy states): 12,000 rollouts from the slot (t 27.2-28.1 s, 3 s) and 9,000 from the CP2
  ramp turnaround on (25.8-27.0 s, 5 s): **0 through the window, 0 below z 10,600, best
  28.72%**, every deep one dying on the window's sill at its mouth (z ~10,971). The greedy
  evals' 28.2-28.6% are exactly that death.
* So the goal field routes through a gap that physics cannot fly from where the agent arrives
  (the "free flight through open space" deception again), and the field-greedy line is stuck
  on it. The human leaves the slot on a 3.3 s detour (dip 2, d +1,978 u) and needs 7.7 s from
  the slot to CP2 against the field's ~3,200 u.
* Visit counts of the incumbent (256 u cells, all envs incl. novel rows): the cavity east of
  the slot's thin wall, the ramp under the room and the passage west toward CP2 (x ~1,000) are
  all visited - by NOVEL episodes, which are never harvested and never move the ceiling. Real
  episodes never get there. The only reachable cells never visited at all are past CP2.

### Next: `--respawn-rare F` (new, generic): Go-Explore's return over the reservoir, per CELL

* The reservoir draw is uniform over states, so it is spent on the main line; Go-Explore picks
  the CELL to return to by 1/sqrt(1 + N). Here N = entries of REAL episodes into a 256 u
  position cell (novel episodes not counted), a share F of the draw goes to cells by that weight
  split evenly over the cell's states (no density bias), the rest stays uniform. Every spawn
  is a state a real episode reached. Purpose: real episodes that wander off the field's line
  (where the detour must start) get returned to and explored from, instead of drowned by the
  main line.

### r2b3_rare: warm + `--respawn-rare 0.5` (new, aafa685) - launched 14:21 UTC (box 56903990, RTX 5080, $0.388/h)

* Code: `surfgym.respawn.RareCellReturn` + trainer wiring, docs/respawn_rare.md,
  tests/python/test_respawn_rare.py (3 unit tests; a CPU smoke with novel rows on in which the
  draw is live, logs, checkpoints and a flagless resume restores the share and the counts; the
  conflicting reservoir draws are refused; flag off bit-identical to the parent trainer: config,
  progress.csv, eval bytes, weights - 4 min on this machine).
* Resume of the incumbent (5,100,797,952), 450M steps. Counts start at zero (not in the
  incumbent ckpt). First lines: `rare eff 140/586 N 576` - the draw spreads over ~140 effective
  cells of the ~590 the reservoir stands in.
* Expected shape (computed on the incumbent's own 20k reservoir with its novelty counts as a
  stand-in for N): the rare half of the draw takes progress 5-10% from 1% to 16%, 10-15% from
  0.2% to 5%, 15-20% from 38% to 10%, 25-27% from 19% to 25%; the reservoir holds almost
  nothing at 27-28% (3 of 20k states: the 2 s margin keeps the slot out).

### The human's way past the slot (WR demo 47250, ANALYSIS ONLY - nothing from it enters training)

Re-downloaded from speedrun.eu (`/demos/47250/download`), parsed with tools/demo/parse_hldemo.py
into the session scratchpad. Field progress in brackets.

| t (s) | where | what |
|---|---|---|
| 27.4-27.6 | slot, y -905, z ~11,460, v (-1,835, 0, -65..-225) | the agent's line exactly (other lane) [27.5-27.9%] |
| 27.8-28.0 | (2,349, -818, 11,388) -> (2,236, -504, 11,295) | enters the shaft over the gap's kill cap, HITS the window wall and slides along it toward y = 0, v (2, +1,717, -545) |
| 28.2-28.4 | (2,310, -171, 11,169) -> (2,591, 14, 11,012) | strafe-turns EAST and goes through a CENTRE GAP in the slot's thin end wall (x ~2,530, y ~-60..130, z ~10,800-11,000) into the cavity behind it [28.24% -> 27.6%] |
| 28.6-29.4 | cavity -> (2,001, 276, 9,826) | falls through the cavity and turns west, down to z ~9,800 under the shaft [26.3%] |
| 29.6-30.8 | (1,983, -123, 9,805) -> (2,481, 80, 10,927) | surfs a ramp there UP (vz +390..+1,225), east, then back west through the same centre gap into the shaft, climbing |
| 31.0-31.6 | (2,412, 332, 11,027) -> (2,054, 944, 11,136) | climbs across the shaft and goes through the WINDOW nearly level (vz -55, vx -1,344) [28.6%] |
| 31.8-33.0 | room -> (1,435, -250, 10,737) | crosses the room, rides its west wall's 45 deg ramp [29.2%] |
| 33.2-34.6 | -> (1,085, 53, 9,742) | down under the room [30.1%] |
| 34.8-35.6 | -> (352, -505, 10,656) | climbs west through CP2 (35.14 s) [31.2%] |

* So the window is entered only from BELOW, climbing, after a 3.5 s detour through the cavity
  and a ramp under the shaft: exactly the dip the field charges (27.6 -> 26.3% and back). It
  confirms the free-fall analysis: from above, no line threads it.
* The pieces the agent lacks are the first ones, in REAL episodes: wall-slide toward the centre,
  strafe east through the centre gap. Its novel episodes have flown the cavity, the region under
  the shaft and the passage toward CP2 (visit counts above), so the later pieces may be partly
  learned already; nothing has ever brought a real episode from the slot into the cavity.

### r2b3_rare result (14:33 UTC): flat, stopped at 5.28B (180M steps)

* Evals (best / median %): 5,177M 27.3/27.3, 5,252M 27.9/27.4. Training: crawl share 17% ->
  33-41%, reservoir min-depth 72.03% -> 72.35% (shallower), reservoir spread over ~1,200 cells
  (from 456), draw over ~210 effective cells.
* Why it cannot work at this wall: it only re-weights the RESERVOIR, and the reservoir holds no
  state in the shaft or near the window (every real episode dies there < 2 s after arriving).

### A Go-Explore probe in the simulator finds the cavity (analysis only, 14:25 UTC)

* `autoresearch/analysis/sid_goexplore_probe.py`: archive of (128 u cell x 4 headings x 3 speed x
  climb) holding the deepest state per cell, returns by 1/sqrt(1 + chosen), 1 s of random held
  usercmd primitives per return, live triggers; seeded with the incumbent's OWN greedy states at
  26.0-28.0 s (ep 1 of its 4,917M eval). 30,000 returns (1.7M ticks, 2.7 min of CPU):
  12,146 cells, the first cells below z 10,600 after ~10,000 returns, 292 by the end - the
  cavity and the space under the shaft, i.e. the human's first piece. Best progress stayed
  28.59%: random actions do not surf the ramp below back up to the window.
* So: returns that can START inside the shaft find the way into the cavity cheaply; the trainer
  never returns there because of the reservoir's margin.

### r2b3_arch: warm + `--respawn-rare 0.5 --respawn-rare-arch 1` (c2062d5) - launched 14:34 UTC

* The archive mode of --respawn-rare: per 256 u cell a ring of 4 states at which REAL envs FLEW
  into it (a > 512 u jump between decisions is a respawn and is not archived), no pre-death
  margin; 50% of the non-start, non-novel spawn rows come from it (~22% of all spawns), cells by
  1/sqrt(1 + N), half of them among the deepest quarter of archived cells by field distance.
  Returns land in their cell, so N grows with each return and a doomed cell stops being drawn
  (what replaces the margin). Tests: archive unit test (real-only capture, jump filter, tail
  rows, shell, round trip) and a CPU trainer smoke with resume (3 min).
* First lines: archive draw over ~447 effective cells of ~617, mean N ~19.
* Leading indicator to read off its checkpoint: archive cells in the cavity (x 2,580-2,990,
  z 10,700-11,250) or below the shaft (z < 10,600, x 1,700-2,900) = real episodes on the
  detour.

### r2b3_arch at 5.40B (14:53 UTC): real episodes on the detour for the first time - EXTENDED

* Evals (best / median %): 5,177M 28.0/27.5, 5,252M 27.4/27.4, 5,328M 28.3/27.9 - the greedy line
  is the incumbent's (window mouth), no new gate yet.
* The archive (real envs only, `autoresearch/analysis/sid_arch_regions.py` on the box's
  ckpt_latest): 3,658 cells, 461k archive spawns so far.

  | region (measurement boxes only) | 5,126M (25M in) | 5,400M (300M in) |
  |---|---|---|
  | shaft lower (x 2,210-2,520, z 10,700-11,150) | 41 cells | 37 cells, 25k entries |
  | centre gap area | 4 cells | 6 cells, 6.4k entries |
  | cavity behind the slot's end wall | 8 cells | 12 cells, 3.7k entries |
  | below the shaft (z 9,500-10,650) | 180 cells | 212 cells, 131k entries |
  | room | 0 | 0 |

  Deepest archived real states: 28.75-28.81% INSIDE both windows' tunnels (x ~2,000, z
  10,976-10,990, the sill height) - the incumbent's novel ceiling was the same 28.82%.
* So within 25M steps the per-cell archive returns brought REAL episodes into the cavity and
  below the shaft, which no real episode had reached in the incumbent's 3.5B steps at this wall
  (its ceiling and visit counts above). The missing piece is the human's climb back up and level
  entry into the window. PROTOCOL: a qualitatively new reach at the wall, so extended:
  continuation r2b3_archx (bare resume of r2b3_arch's ckpt_final, archive restored), ~2B steps,
  box deadline 17:54 UTC.
* Flag-off identity re-tested after the archive edits: passed (6:49 on this machine).

### r2b3_archx at 5.95B (15:31 UTC): the detour region is explored, the climb is not found yet

* Evals since the continuation started (best / median %): 5,552M 27.4/27.4, 5,627M 28.0/26.4,
  5,703M 27.5/26.4, 5,778M 27.4/27.4, 5,854M 27.4/27.3, 5,929M 28.0/27.9 - the greedy line
  flips between the window mouth and the old pit dive, as in the incumbent's late evals.
* Archive at 5,965M: 3,879 cells, 48M real entries; below the shaft 212 cells (338k entries),
  cavity 13, centre gap 7, room 0; deepest real state still 28.81% (window tunnel, sill
  height); novel ceiling 28,174 -> 28,190 u.
* Why no climb yet (archived states, `arch_speeds.py` on the box): below the shaft |v| median
  1,106 u/s (p90 1,730), in the box where the human rides the curved surface under the cavity
  median 937 u/s (max 1,907); the ~2,000 u/s states are falling past it at the shaft's sides
  (y ~+-1,000); 81 climbing states, the highest at z ~10,400 with |v| ~700. The human enters
  that surface at ~2,200 u/s from the cavity dive and rides ~270 deg around and 1,100 u up to the
  centre gap. So the archive returns the agent to the right PLACE, not yet with the speed / line
  the ride needs.
* Kept running (bare continuation to 7.55B, ~16:50 UTC); idea cap 6 h from 14:34.

### r2b3_archx stopped at 6.22B (15:47 UTC): the archive stage stalled

* Evals 5.55-6.16B (best %): 27.4, 28.0, 27.5, 27.4, 27.4, 28.0, 28.2, 27.2, 27.4 - no new gate.
  Archive at 6.20B: 3,929 cells, 62M entries; below the shaft still 212 cells (417k entries),
  cavity 13, centre gap 6, room 0; deepest real state 28.81% since 5.40B; novel ceiling
  28,193 u. ~1.1B steps in archive mode: the detour region is visited heavily, the climb never.
* The human's climb (demo, analysis only) is a helix: from the cavity down half a turn around
  a centre near (2,370, 100) at r ~400-450 u to z ~9,800, then up the other half to the centre
  gap at z ~10,930 and across the shaft into the window - entered at ~2,200 u/s.

### r2b3_archs: + `--respawn-rare-speed 3` (d67d1cb), warm from r2b3_archx's ckpt_latest (6.22B) - launched 15:49 UTC

* Archive cells keyed by (256 u position, horizontal-speed bucket of 1,333 u/s - the recipe's own
  --int-speed 3 binning): a fast state in a cell no longer shares the ring and the count with
  the slow ones that dominate it (archived states in the helix box were median 937 u/s). The
  archive restarts empty (other keys). 450M steps.

### WON'T vs CAN'T on the helix (bench from the human's states - MEASUREMENT ONLY, 16:00 UTC)

`record_ckpt --spawn-states` with single demo states (scratchpad helix/h_*.npy), policy =
r2b3_arch ckpt_final (5.55B), 6 stochastic + 2 greedy episodes, 8 s cap:

| spawned at the human's state at | outcome |
|---|---|
| 28.4 s (through the centre gap into the cavity) | 8/8 fall and die at the bottom kill (z ~9,577) in 1.1 s |
| 29.4 s (helix bottom) | climb a little (z ~10,350), fall back, die (z ~9,565) in 1.5-3 s |
| 30.2 s (climbing the helix) | reach the centre gap's height but east of it (x ~2,840), fall, die |
| **30.8 s (through the gap, climbing into the shaft)** | **8/8 go through the window, cross the room, go under it toward CP2: 29.5-30.1%, 2 alive at 8 s near x 1,100** |

* So the policy already flies everything after the centre gap (learned from novel rows and
  the archive returns). The one missing skill is the ~2.4 s helix ride from the cavity down
  and back up to the gap.
* Random archive-and-return exploration seeded at the human's own 28.4/28.6 s states (probe,
  20,000 returns, 0.8M ticks): 4,278 cells, 0 states climbing back into the shaft. The ride
  needs surfing skill, and the reward has no signal along it: under the ratchet the helix
  (27.6 -> 26.3 -> 27.5%) pays nothing until the episode is back past its record (the window,
  28.6%), ~3 s later.

### r2b3_archs result (16:24 UTC): speed-keyed cells change nothing at this wall

* 6.22 -> 6.67B (450M). Evals (best %): 27.2, 28.0, 27.4, 27.9, 28.0, 28.2. Archive (6,445
  keys): below the shaft 366 keys, cavity 24, centre gap 14, room 0; deepest real state
  28.85% (window tunnel, sill height) - the same reach as the position-only archive.
* ckpt_final (6,672,089,088) left on the box for a later warm continuation.

### r2b3_scrarch: FROM SCRATCH, incumbent recipe + `--respawn-rare 0.5 --respawn-rare-arch 1` - launched 16:25 UTC

* 450M steps (30 min). Matched-step reference = r1b1's ratnov (the same recipe without the
  archive): best-episode progress 5.8% at 302M, 15.2% at 378M (trap exit), 20.8% at 453M
  (window wall), window passed 576M, CP2 wall 651M, CP2 pit crossed 1,633M.

### r2b3_scrarch result (16:53 UTC): from scratch the archive does NOT speed up the first gate

* 450M steps at 284k steps/s. Evals (best %): 1M 1.1, 76M 4.4, 151M 4.5, 227M 4.6, 302M 4.7,
  378M 4.7 - still in the first-ramp trap at 378M, where ratnov (the same recipe without the
  archive) was out of it (15.2%) and at the window wall by 453M (20.8%). One seed and the trap
  exit is a coin flip on this lineage (GR 529M, ratnov 378M, novel-only 302M), so not a verdict,
  but no acceleration. A likely cost: from scratch everything is rare, so 22% of the spawns
  spread over the whole explored map instead of the main line.
* So the archive is a WALL tool for now (warm, at a detour the reservoir cannot reach), not a
  scratch default. Not extended; config.incumbent_recipe unchanged.

### r2b3_archsx: warm continuation of r2b3_archs (6.67B -> up to 9.67B) - launched 16:55 UTC

* Bare resume (speed-keyed archive restored: 6,524 keys, 26M entries). Why keep training this
  idea (cap 6 h from 14:34): the archive keeps real episodes on the detour (cavity, below the
  shaft, the window tunnel at 28.85%), and the policy already flies everything after the centre
  gap (bench above); the helix ride is the one missing link, learnable only from those returns.
  Box deadline extended to 20:26 UTC.

### r2b3_archsx (6.67 -> 7.21B, stopped 17:28 UTC): eval 28.643 at 7,126M - the branch best

* Evals (best %): 27.9, 27.4, 28.5, 27.9, 27.5, 28.2, **28.6 (28.643)**, 27.4. The 28.643
  episode dies on the window's sill 30 u deeper than the incumbent's best (x 2,149 vs 2,139..2,181):
  the same failure, marginally deeper, so it beats 28.628 by 0.015 without a new gate.
* Archive at 7.13B: below the shaft 416 keys, cavity 30, centre gap 11, room 0; deepest real
  state 28.85%. Checkpoint pulled: runs/r2b3/r2b3_archsx/ckpt_latest.pt (step ~7.21B,
  md5 19059be9...).

### The archive / novel-row conflict, and `--respawn-rare-own 1` (80a105c)

* The novel rows may only go where no REAL episode went in the last 10-20M steps. Archive
  returns are real, so once the archive works the cavity / under-shaft region, the novel rows
  are pushed out of it - and the novel rows are what taught the incumbent policy everything
  after the centre gap (the bench). Under the ratchet a novel row on the helix is also PAID for
  climbing (its record starts at its own spawn d), which no archive return in the cavity is.
* `--respawn-rare-own 1`: an episode started from an archive row (tagged by spawn origin, the
  novel sampler's own rule) does not mark the novel visit grid; it still lowers the ceiling.
  Tests: unit (tagging, own-flight mask) + CPU trainer smoke with resume (3 min).

### r2b3_own: warm from r2b3_archsx (7.21B) + `--respawn-rare-own 1` - launched 17:30 UTC

* 450M steps. Watch: novel rows landing in the cavity / helix again (step line `novel ... rows`),
  archive cells climbing in the shaft from below, the room.

### r2b3_own result (7.21 -> 7.66B, 18:03 UTC) and the long continuation r2b3_ownx

* Evals (best %): 27.6, 27.9, 27.4, 27.4, 28.0, 27.9 - no gain. The flag works as intended:
  the novel visit grid (128 u) in the helix box went from 74% visited (r2b3_archsx's ckpt) to
  19%, below the shaft 63% -> 11%, cavity 45% -> 21%. But the novel sampler spreads its rows
  over the whole map (half flattened over progress): row progress median ~12,900 u (13%), p90
  ~23,000 u (23%), so few rows land on the helix. Archive: below the shaft 434 keys, cavity 40,
  room 0; deepest real state 28.85%.
* r2b3_ownx: bare continuation from r2b3_own's ckpt_final (7.66B), launched 18:04, to be
  stopped ~23:15 for the harvest (box deadline 23:40). Why: every eval of these warm archive
  runs is a chance at a deeper sill death (the archive's returns to the deepest cells - the
  window tunnel at 28.85% - are what produced 28.643), and the helix may still be found.

### 19:00 UTC: r2b3_ownx eval 8,114M = 28.766 - a BOUNCE ENTRY into the window tunnel (branch best)

* The best episode (ep 0) enters the shaft over the gap at 27.6 s (2,463, 841, 11,435), hits the
  window wall and the shaft's sides (27.7-28.1 s: v (-1,482, 724) -> (548, 1,345) -> (1,148,
  -564) -> (-551, -1,026) -> (-1,055, 169)), i.e. it sheds height while the bounces turn its
  velocity back WEST, then flies into the window tunnel at window height (28.2 s: (2,203, 892,
  11,129), v (-1,015, 176, -749)) and dies on the sill ~180 u in (2,023, 863, 10,981): 28.77%.
* This corrects the "dead end" analysis above: that proof assumed ONE ballistic arc from the
  gap. Wall contacts in the shaft break it. Crossing needs, at the tunnel mouth, roughly
  (|vz| t + 400 t^2) <= z_entry - 10,994 with t = 256 / |vx| (standing hull), e.g. |vx| >=
  ~1,250 at the top edge with |vz| <= ~645. The episode had |vx| ~1,030, vz ~-750 at z 11,129.
* The archive's tunnel states (17 real states, `tunnel_states.py` on the box): inside the tunnel
  at x ~1,970-2,010, |vx| 950-1,190, vz ~-900: ballistic exit 10,900-10,951, i.e. 45-95 u short
  of the sill. The ratchet pays every unit deeper into the tunnel, so the pressure is toward
  higher / faster entries; one crossing reaches the room, which the policy already flies
  (bench above).
* Checkpoint pulled as insurance (r2b3_ownx/ckpt_latest.pt at 19:00, md5 c3856fb1...); the run
  continues.
* Feasibility (analysis only, `sid_slot_shoot.py` from the 8,114M ep 0's OWN states at 27.6 /
  27.8 / 28.0 / 28.1 s, 3,000 random held-primitive rollouts each, 2 s): 0 cross the tunnel;
  best 28.79% (x ~2,003 on the sill). At 28.1 s (2,302, 864, 11,200), v (-1,055, 169, -669) the
  entry is already too steep (exit ~10,950); wall contacts can turn the horizontal velocity but
  never undo the fall. So the bounce entry is a near-miss that buys a deeper sill death (up to
  ~28.9% at the tunnel's far end), probably not a crossing; the helix stays the way on.

### r2b3_ownx stopped at ~9.48B (19:56 UTC); the archive idea at its cap

* 7.66 -> 9.48B (1.8B steps). Evals (best %) after the 8,114M 28.766: 27.4-28.0 everywhere
  (19 evals); archive reach unchanged (room 0, deepest real state 28.85%). The run's newest
  weights pulled: r2b3_ownx/ckpt_latest.pt (md5 11bf38c0...) - the checkpoint of the branch's
  best eval's run.
* The archive idea (14:34 -> ~20:00, ~5.5 h of box time across arch / archx / archs / archsx /
  own / ownx) is at its cap.

### r2b3_scrown: FROM SCRATCH, recipe + `--respawn-rare 0.5 --respawn-rare-arch 1 --respawn-rare-own 1` - launched 19:57 UTC

* Hypothesis for the first scratch test's slow start (4.7% at 378M vs ratnov's 15.2%): archive
  returns marked the cells around the first-ramp trap as visited and pushed out the novel rows,
  which are what drive the early gates (novel-only: 10.1% at 302M). With the own-flight flag
  the novel rows work as without the archive. 450M steps; compare with ratnov (5.8% at 302M,
  15.2% at 378M, 20.8% at 453M) and r2b3_scrarch (4.7% at 302M and 378M).

### r2b3_scrown result (20:24 UTC): the own-flight flag does not fix the scratch slow start

* 450M at 288k steps/s. Evals (best %): 1M 1.1, 76M 4.1, 151M 4.6, 227M 4.6, 302M 4.6, 378M 4.7 -
  the first-ramp trap through 378M, like r2b3_scrarch (4.7%) and unlike ratnov (15.2% at 378M).
  Two scratch runs with the archive both behind at matched steps (one seed each; the trap exit
  is late on other seeds too: GR 529M) -> stopped (PROTOCOL). Verdict for the lineage: the
  archive is a wall tool for warm runs, not a scratch default; incumbent_recipe unchanged.
* r2b3_ownx2: resume of r2b3_ownx's newest weights (9,472,835,584) for the rest of the box time
  (to ~23:15), the archive settings restored. Not a new idea: evals of the branch's best policy
  and a last chance at the helix.

### r2b3_ownx2 (9.47 -> 10.88B) stopped 21:55 UTC; box released 21:57

* Evals (best %): 27.5, 28.5, 27.4, 27.4, 27.5, 28.6, 28.5, 27.9, 27.5, 28.0, 27.4, 27.5, 27.3,
  27.4, 28.0, 27.4, 27.4, 27.4, 27.4 - from ~10.2B mostly back on the old pit dive, approx_kl up
  to ~0.2. Nothing above the 28.766 of r2b3_ownx.
* Box 56903990 (RTX 5080, $0.388/h) ran 14:19-21:57 (~7.6 h, ~$3.0); box 56893794 12:55-13:59
  (~$0.4). All runs pulled to /home/steam/ar/runs/r2b3/ (107 valid evals). Branch best by
  score.py: r2b3_ownx/traj_8113881088.jsonl, 28.766; the run's newest weights:
  r2b3_ownx/ckpt_latest.pt (step 9,472,835,584, md5 11bf38c0195ca4664c5689dbec6dbd2d). It carries
  the archive state and the --respawn-rare* flags (a bare resume restores them).

## ROUND 2 LESSON (r2b3) - written 21:58 UTC

**Result:** best greedy eval from the true start **28.766** (r2b3_ownx at 8,114M; incumbent
28.628). 0 finishes. The gain is a different death, not a new gate: the best episode bounces
between the shaft's walls and flies INTO the window tunnel at window height, dying on its sill
~180 u in, where every earlier best died at the tunnel's mouth.

**What the wall at 28.6% is** (analysis only - traces, random rollouts, the WR demo):
* The field routes the slot's exit through a 256 u window tunnel (origin box y 852-940,
  z 10,992-11,152 standing) whose sill is a kill plate. From the slot (the gap over the kill cap
  at z >= ~11,412) a single ballistic arc cannot thread it (proof in the notes); wall bounces in
  the shaft get INTO it (the 28.766 episode) but, in 12,000 random continuations of that
  episode, never through it. The greedy evals of every warm run die at the tunnel's mouth or sill.
* The human does not use that entry: from the shaft he goes EAST through a centre gap in the
  slot's end wall (y ~0, z ~11,000) into a cavity, rides a helix down and back up, comes back
  through the gap climbing and crosses the window nearly level (vz -55), then the room, under it,
  CP2 at 35.1 s (31%). That 3.5 s detour is the field's "dip 2".
* A bench from the human's states (measurement only): the policy already flies EVERYTHING after
  the centre gap (window -> room -> under the room -> ~30.1%, alive at 8 s); from the cavity and
  from the helix it falls and dies. The one missing skill is the ~2.4 s helix ride, and the reward
  has no signal along it (the ratchet pays nothing until the episode is back past its record).

**What worked:** `--respawn-rare 0.5 --respawn-rare-arch 1` (new, generic: Go-Explore's per-cell
archive of the states at which REAL episodes entered a cell, no pre-death margin, returns by
1/sqrt(1 + N), half among the deepest quarter by field distance). Within 25M steps of a warm
resume it put real episodes into the cavity and below the shaft for the first time (the
incumbent's 3.5B steps there never had), and its returns to the deepest cells (the tunnel) are
what produced the 28.6-28.77 evals. `--respawn-rare-own 1` (archive returns do not use up the
novel rows) and `--respawn-rare-speed 3` (speed-keyed cells) are refinements; neither changed
the reach.

**What did not work (warm at the wall unless noted, 100M-2B steps each):** SIL 0.1 (sigma
inflated by imitating T = 2 actions, line worse); `--respawn-margin 0.5` (no change);
`--unstuck-max 0` (the CP2 line is lost - T = 2 keeps it); reservoir-weighted `--respawn-rare`
(the reservoir holds nothing in the shaft); the archive FROM SCRATCH, with and without the
own-flight flag (both still in the first-ramp trap at 378M where ratnov was at 15.2%: a wall
tool, not a scratch default; one seed each).

**For the next round:** the helix ride. The policy has every later piece; it needs practice
states on the helix with a tangential velocity and some credit for the climb. Candidates: novel
rows near the frontier with their velocity projected onto the nearest surface (surf-ready rows,
generic), a reverse curriculum from the policy's own states once any real episode gets partway
up the helix (the archive keeps them), or a search-based explorer (planning in the simulator
from the archive's cavity states with the policy as a prior). The incumbent checkpoint
(r2b3_ownx) already carries the archive state (`--respawn-rare*` restored on a bare resume).
