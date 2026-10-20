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

## Round 2, branch r2b4 - 2026-10-20 12:58 UTC (box 56893447, RTX 5080, $0.309/h)

### Where the incumbent stops, read off its own eval (analysis only)

* Incumbent eval traj_4916772864 (score 28.628): 5 of 9 greedy episodes die at 27.45-27.95% in
  the pit's kill-coated ramp just EAST of its west edge (x 2,600-2,810, z 11,280-11,370) after a
  high arc out of the +y-ramp climb (apex z 11,800-11,860 at x ~3,500, then vz -600..-880);
  ep1 (28.63%) flies a LOWER arc (apex z ~11,520), passes over the edge at z ~11,450, hits the
  wall face at x ~2,237 (vx -1,574 -> +528), drops into the pocket behind it and dies on kill
  *17 (2,139, 922, 10,980); 2 episodes die earlier (24.8 s, the old north/pit lines).
* Geometry of the slot exit (stand-hull point tests + hull-1 kill tests, x-z sections at
  y = 0 / 500 / 880): a pillar x 2,528-2,592 tops out at hull-centre z ~11,425 under a ceiling
  that starts at hull-centre z ~11,488 for x <= 2,592 - a ~60 u window over the pillar; behind
  it a pocket x 2,240-2,496 open down to the kill sheet *21 (z ~10,690); a wall x <= 2,208
  from z ~11,170 up; under the wall a ~140 u tunnel (hull-centre z ~10,990-11,140) exists only
  in the |y| ~830-960 lanes, floored by the kill strips *17 / *113. So the next gate is a
  precise hop-drop-tunnel sequence (or a route I have not found), not a long flight.
* **The reservoir never spawns there.** The incumbent checkpoint's reservoir (20,000 states):
  progress percentiles p50 19.2%, p90 25.6%, p99 26.3%, max 27.97%; only **3 states >= 27%**
  (0.015%). Cause: snapshots are taken once per second of an episode and only those >= 2 s
  (`--respawn-margin`) before its end are harvested, so an episode that dies at 28% leaves
  nothing deeper than ~26%. The decisive last 1-2 s before the frontier is practised only
  by episodes that fly the whole run-up at T = 2 (the unstuck T at its cap since ~2.4B).
* Novel rows do not fill it either: `novel/prog_med` 12-17% (half of the sources are flattened
  over the whole reservoir), ceiling 28.1-28.2%.

### NEW: `--respawn-snap S` and `--respawn-deep F` / `--respawn-deep-q Q`

* `--respawn-snap`: seconds between two reservoir snapshots of an episode (absent = the shipped
  1 s / 0.25 s under --goals). `--respawn-deep F`: F of the reservoir draws come uniformly from
  its deepest Q quantile by goal distance (default Q 0.05), the rest uniform - Go-Explore's
  "return to the frontier" over the policy's OWN harvested states (no moved or invented state).
  Generic: three constants, no map quantity. Both restored on resume, TRAIN_ONLY in record_ckpt.
* Tests: tests/python/test_respawn_deep.py (weights realise the share; a RespawnBuffer draw lands
  it; CPU smoke + flagless resume restores; flag off bit-identical to the trainer before).

### r2b4_deep1: warm incumbent + `--respawn-snap 0.25 --respawn-margin 0.5 --respawn-deep 0.5 --respawn-deep-q 0.05` (launched 13:01 UTC)

* Box 56893447 (RTX 5080, Utah, $0.309/h): **314k steps/s** instantaneous on the resumed recipe.
* WARM, measuring AT the wall: resume of `$AR_HOME/incumbent/ckpt.pt` (5,100,797,952; md5
  3cc658ed...), +450M steps (to 5.55B, ~24 min). Commit cc88240. All recipe flags restored from
  the checkpoint (T = 2 at its cap, novel spawns, ratchet).
* Question: does putting ~25% of all spawns on the deepest 5% of a 0.25 s / 0.5 s-margin
  reservoir (the slot exit, 27-28.6%) let the policy find the pillar-pocket-tunnel line that
  3.5B steps of run-up-only practice did not? Reference = the incumbent's own continuation:
  ratnovx2/ratnovx3 2.5-5.9B, best 28.0-28.6%, frontier frozen at 27,916 u since 1,633M.
  Promising = an eval past 28.63% (x < 2,208, z < 11,140 or beyond), or the reservoir's min
  depth moving past 72.03% of d0 (27.97% progress).

### r2b4_deep1 result (5.10B -> 5.55B, 13:01-13:28 UTC): NEGATIVE, stopped at its budget

* Evals (score.py best / median): 5,101M 28.200/27.7, 5,177M 27.332/27.2, 5,252M 28.207/27.5,
  5,328M **28.297**/27.3, 5,403M 27.925/22.4, 5,479M 28.089/27.6. 0 finishes; below the
  incumbent's 28.628 and in the band of the incumbent's own continuation (ratnovx3 28.0-28.5).
* The reservoir's deepest state moved 27.97% -> 28.25-28.33%, but the novel ceiling (deepest
  REAL episode) stayed at 28,174 u for all 450M steps: with ~24% of all spawns on the deepest 5%,
  no training episode went deeper. `race/crawl_frac` 8-13% -> 50-60%: the deep shell is the
  pocket behind the x 2,224 wall face - states that hit the wall and fall at |v_h| < 300 (an
  air-strafer from rest gains ~30 u/s along its wish direction, so they are doomed).
* Lesson: "deepest by field distance" selects doomed states whenever the field's descent runs
  into a dead end beside the route. Do not run --respawn-deep at this wall again.

### CAN'T, not WON'T: random-action Go-Explore from the policy's own frontier (analysis only)

* Tool: `autoresearch/analysis/sid_go_search.py` (new): Go-Explore's explore phase with RANDOM
  held macro-actions (velocity-frame yaw targets off_warp(u), keys, duck, jump; 4-40 ticks each) in
  the real core (teleport_fail on), rooted at a checkpoint's own reservoir states, archive of
  128 u x 500 u/s cells (`--fine`: x heading octant x vz sign), count-weighted selection.
  On the 32-thread box: 8,192 envs x 200 ticks x 300 rounds in 63 s.
* Runs, roots = the incumbent's reservoir: local 2,048 x 300 ticks x 60 rounds (12k cells);
  box 8,192 x 200 x 300 (22.5k cells, roots >= 20%); box `--fine` 8,192 x 150 x 600 (200k+
  cells, roots >= 22%). **Deepest live state 28.66-28.73%, always at the mouth of the wall hole
  (x 2,086-2,225, |y| ~870-900, z ~10,940-11,030) falling at vz -850..-930.**
* Geometry that explains it (stand-hull point tests, hull-1 kill tests): the wall x <= 2,216 is
  solid z 10,608-11,568 except two holes, hull-centre |y| 848-940 x z 10,996-11,156, 280 u long
  (x 2,216 -> 1,936), floored by kill strips *17 / *113. The pillar x 2,520-2,600 in those lanes
  carries kill plates *80 / *64 on top under a ceiling: a 32 u window (hull-centre z
  11,444-11,476). A ballistic line from the pillar window to the hole's top corner continues
  ~2 u BELOW the hole's far bottom corner even ducked, so no dive from the slot threads it; the
  hole has to be entered near the apex of an arc (|vz| < ~300). The rest of the middle level is
  sealed: the pocket floor is kill *21, the pillar's other hole (y ~0, z 10,832-11,072) leads to
  a chamber under the pit ramp that drops into a lower corridor, and that corridor is cut off from
  the west by a 45-degree slab (normal (-0.71, 0, 0.71): walkable, not surfable) - the field
  jumps 71,500 -> 68,950 across it.
* Downstream is easy: own states moved just WEST of the hole (heading west at ~1,330 u/s) fly to
  29.0-30.3% with the incumbent policy (16 stoch + 4 greedy, spawn_bench). So passing the hole is
  worth ~+1.5% at once, ~+2.4% at CP2.

### The human's line past the slot, from the WR demo (ANALYSIS ONLY - section 0; nothing enters training)

* Source: speedrun.eu demo 47250 (krs, 115.99 s), re-downloaded to the session scratchpad and parsed
  with tools/demo/parse_hldemo.py; times are demo seconds (10 ms rows). It is a 7.5 s LOOP, not a dive:
  * 27.0-27.6 s: the -y lane slot (y ~ -910), z 11,440-11,480, level, 1,834 u/s west.
  * 27.7 s: over the pillar in the lane window (2,509, -903, 11,423).
  * 27.8-28.3 s: a wall-slide in the pocket turns the velocity 180 degrees at ~1,720 u/s
    (west -> north along the x 2,224 face -> east), dropping to z ~11,100.
  * 28.4 s: EAST through the pillar's centre hole (2,591, 14, 11,012).
  * 28.5-29.0 s: a U-turn in the chamber under the pit ramp, falling (vz -950..-1,345).
  * 29.1-29.9 s: falls to z ~9,790 at (1,969, 74) and SURFS a curved low ramp: vz -1,425 -> +1,293
    in 0.7 s, |v_h| up to 2,112, heading turned south then east (side key held, the view 1-2 deg
    ahead of the velocity - the plain surf input).
  * 30.0-30.4 s: rises east back into the chamber; 30.5-31.0 s: curves north and passes WEST
    through the pillar's centre hole again, still rising.
  * 31.5 s: the APEX (vz +25) at (2,185, 914, 11,138) - inside the +y wall hole - then through it:
    (1,920, 943, 11,126) at 31.7 s.
  * 32.8 s: lands on the block top (1,435, -250, 10,737), U-turns, drops off under the block, surfs
    another low ramp (vz -1,208 -> +1,190 at 34.3-35.0 s) and rises into CP2 at 35.14 s.
* Field progress along it: 28.24% at 27.9 s, DOWN to 26.25% at 29.4 s (the "dip 2"), back to 28.2%
  at 31.2 s, 30.4% at CP2. The field shows none of the climb: it is CLAUDE.md 0b's case of a hidden
  ramp skill.
* So my ballistic finding stands: the hole is entered at the apex of an arc from BELOW, and the arc
  comes from a surf in the low region the policy only ever falls through. Random actions (with or
  without a surf primitive: `--surf-p 0.5`, 421k cells) never chain the loop.
* Which generic mechanism this points at: the policy CAN surf; it never practises in the rare cells
  of the detour (the chamber, the low region), because the reservoir draw mirrors visitation and the
  2 s margin drops the last seconds of every fall. Go-Explore's return over its own states, with
  count-weighted cells instead of depth, is the textbook fix - `--respawn-cells` below.

### NEW: `--respawn-cells F` (+ `--respawn-cells-size`, `--respawn-cells-band B`) - Go-Explore's return

* `surfgym.respawn.CellReturn`: F of the reservoir draws pick a cell (128 u position cube x velocity
  class: |v_h| in 500 u/s buckets, heading octant, vertical sign) with weight
  1/sqrt(1 + harvested + chosen), then a uniform member state; the rest stay uniform. Counters ride
  in the reservoir's checkpoint payload. `--respawn-cells-band B`: only states within B x d0 of the
  reservoir's own deepest progress take part (Go-Explore's furthest-level preference). Default off,
  byte-identical (tests/python/test_respawn_cells.py: unit, smoke + flagless resume, flag off vs the
  trainer before - all pass). TRAIN_ONLY in record_ckpt.

### Detour states in the reservoirs (scratchpad lowregion.py, the checkpoint's newest 20k states)

| reservoir | pocket | pillar centre hole (rising) | chamber | low region (rising) |
|---|---|---|---|---|
| incumbent (2 s margin, 1 s snaps) | 0 | 0 | 0 | 0 |
| deep1 (0.5 s, 0.25 s, deepest-5% spawns) | 9 | 1,171 (114) | 2,705 | 109 (1) |
| cells1 (0.5 s, 0.25 s, whole-map cells 0.5) | 4 | 0 | 0 | 0 |

* So the policy DOES go east through the pillar's centre hole and down the chamber when it is spawned
  in the pocket (deep1) - the first two steps of the human's loop come for free. The missing step is
  the surf on the low block's face (normal z ~0.53, i.e. surfable) that turns the fall into a climb.

### r2b4_cells1: warm incumbent + `--respawn-cells 0.5 --respawn-snap 0.25 --respawn-margin 0.5` (14:10-14:38 UTC)

* Evals (best / median): 5,101M 28.221/27.9, 5,177M 27.335/27.2, 5,252M 28.234/28.2, 5,328M
  27.350/27.3, 5,403M 28.298/27.7, 5,479M **28.588**/27.7. 0 finishes; novel ceiling unchanged.
* Whole-map cells dilute: 107k cells ever harvested, 72-83% of the cell draws from cells harvested
  <= 10 times - side falls all over the map - and the detour got NO returns (table above).
* Next: r2b4_cells2 = `--respawn-cells 0.8 --respawn-cells-band 0.05` (cells within 5% of d0 of the
  reservoir's frontier: the pocket, the pillar hole, the chamber, the low region, the run-up).

### Skill bench from the human's line (DIAGNOSTIC ONLY - section 0 allows measuring from demo states, never training)

* Incumbent checkpoint, 1 greedy + 6 stochastic episodes (record_ckpt --spawn-states, 8 s cap, no yaw
  jitter) from single WR-line states. best progress / 'A' = reached the antechamber west of the wall:

  | WR state | what it is | result |
  |---|---|---|
  | 28.4 s | pillar centre hole, heading east | 7/7 die in ~1.1 s, 27.57% |
  | 28.9 s | chamber, falling west (-1,452, 926, -1,265) | 7/7 die in 0.6 s: no turn, ballistic into the y 1,136 side, down to the floor kill |
  | 29.2 s | bowl entry | 7/7 die in 0.3-0.6 s |
  | 29.5 s | bowl bottom (-100, -2,003, -117) | SURFS: vz up to +1,190..+1,250, 2-4 s, best 26.6-27.5% |
  | 30.0 s | bowl exit, rising east | climbs too far east (x ~2,820), misses the pillar hole, 27.3-27.5% |
  | 30.5 s | chamber, rising north | greedy **29.83% A**; stochastic 1/7 past 28.3% |
  | 31.0 s | pocket, rising north | **7/7 A**, 29.43-29.76% |
  | 31.4 s | approaching the wall hole | **7/7 A**, 29.32-29.97% |

* So the policy already owns the bowl surf and the finish (rising pocket -> wall hole apex -> antechamber);
  it lacks two steering links inside the detour: chamber fall -> bowl (a strafe into the surface the human
  catches at ~28.9 s) and bowl exit -> rising chamber (turn north-west earlier and lower). That is the
  textbook case for returning to its own rare detour states (cells) and letting RL fix the links.

### r2b4_cells2: warm incumbent + `--respawn-cells 0.8 --respawn-cells-band 0.05` + 0.25 s / 0.5 s (14:40-15:08 UTC)

* Evals (best / median): 5,101M 28.199/27.7, 5,177M 27.365/27.1, 5,252M 27.402/27.2, 5,328M 27.970/27.4,
  5,403M 27.425/26.5, 5,479M 28.050/27.6. 0 finishes. No gain.
* Reservoir (newest 20k): pocket 8, pillar centre hole 7 (2 rising), chamber 0, low region **574 (36
  rising)** - but the "rising" ones are hops on the flat top of the low block (z ~10,080-10,100, |v_h|
  < 80, vz +160..+240): a dead end, not the bowl surf. The band moves returns into the detour, into the
  wrong states.
* Link A is NOT a surface (WR line at 20 ms, 28.84-29.20 s: vz falls by exactly gravity, |v_h| ~1,720
  constant while the heading swings 132 -> 234 degrees in 0.36 s): a sharp AIR turn - strafe key held,
  view 1-2 degrees ahead of the velocity (wishdir ~91 degrees from v, so addspeed > 30 and the push is
  ~60 u/s per frame). The 10 ms core allows up to ~8 degrees per tick at ~17 u/s of speed per tick, so it
  is available to the policy; it is the same input as the bowl surf.
* Ops: the box -> workstation path degraded at ~15:00 (10 KB/s, long streams dropped; the box's own
  internet link 146 MB/s down / 6.5 MB/s up, the workstation's 84 MB/s). `box.py pull` fails with
  EOFError; scratchpad chunkpull.sh (xz on the box, 64 KB slices over separate ssh calls, md5-checked)
  works. Score and analyse on the box when possible.

### r2b4_dc1: deep shell (0.5, q 0.05) + band cells (0.4, band 0.05) + 0.25 s / 0.5 s, warm incumbent (15:15 UTC)

* deep1 put the pocket on spawn and the policy went east through the pillar hole on its own (3,985
  pillar-hole/chamber states); band cells return to rare detour states. Together they should put the
  policy in the chamber falling west - the start of link A - often.

### r2b4_dc1 result (15:15-15:44 UTC): no gain

* Evals (best / median): 5,101M 28.247/27.7, 5,177M 27.982/27.2, 5,252M 27.371/27.3, 5,328M 27.964/27.8,
  5,403M 27.597/26.5, 5,479M 28.315/27.4. Reservoir: pillar hole 1,466 (127 rising), chamber 2,786, low
  region 109 (0 rising) - deep1's picture: the policy reaches the chamber and never does link A.

### Temperature is not the blocker (diagnostic bench at the training temperature, `--exec-temp 3` = 1 + T)

* At T = 2 the finish still holds: from the 31.0 s WR state 7/8 reach the antechamber, from 31.4 s 8/8;
  the bowl surf from 29.5 s is unchanged (vz +990..+1,170). r2b4_lowT (dc1 + `--unstuck-max 0.25`) was
  stopped at 5.18B after two evals (28.294, 27.264; kl 0.26 while it re-adapted).

### THE TIME PENALTY MAKES THE LOOP NET-NEGATIVE (the WON'T is in the objective)

* `--time-pen 0.005` per tick = 0.5 reward units per second; 1 unit = 1% of d0 (scale 100/d0). Any stretch
  whose best route progresses slower than **0.5% of the map per second** loses reward. The human's loop from
  the slot (27.6 s, 27.9%) to CP2 (35.14 s, 30.4%) is +2.5 units for -3.75 units of time; to the
  antechamber it is +0.8 for -2.0. Under ratchet + time penalty, DYING at the wall is better than the loop,
  and every partial detour (link A + the bowl surf: ~+0.4 for ~-1.5) is worse still. No spawn rule fixes
  an objective that prefers the death.
* Generic, map-free correction: no per-second terms until the map has been finished - the time penalty's
  job (finish faster) only starts once something finishes. `--surf-bonus` (0.3/s) has to go with it: the
  trainer refuses a bonus at or above the time penalty (farmable by parking on a ramp).
* Arm r2b4_tp0 (launched 15:59): dc1's spawns + `--time-pen 0 --surf-bonus 0 --critic-warmup 40` (actor frozen
  for the first 21M steps while the critic re-fits). One change against dc1.

### r2b4_tp0 result (15:54-16:22 UTC): NEGATIVE - the main line degrades without the per-second terms

* dc1's spawns + `--time-pen 0 --surf-bonus 0 --critic-warmup 40`. Evals (best / median): 5,101M 28.220/28.2,
  5,177M 28.074/27.4, 5,252M 28.028/26.6, 5,328M 27.112/26.4, 5,403M 26.396/26.3, 5,479M 26.517/26.4.
  Stall kills 25% (dc1 9%): with nothing charging time, episodes linger. Novel ceiling unchanged.
* So the time-penalty arithmetic above is right about WHY partial detours are punished, but zeroing the
  penalty is the wrong fix on a warm policy (it un-learns its flight). Not repeated.

### Go-Explore phase 2 from the phase-1 archive: r2b4_arch1 (launched 16:20 UTC)

* The surf-primitive archive (gs_surf, 427k cells, rooted at the incumbent's own reservoir >= 22%) holds
  1,746 rising chamber/pocket states and 774 bowl surfs (vz up to +1,203) - the states from which the
  WR-state bench shows the policy flies the finish. Selection rule (generic): every archive state within
  5% of d0 of the archive frontier whose CellReturn cell (128 u x speed x heading x vertical sign) the
  policy's own harvest never filled (tp0's counters): 323,334 states, all 1,746 + 774 among them
  (scratchpad build_archive_spawns2.py; the region counts are for the notes, not a selection rule).
* Arm: incumbent recipe (restored) + `--spawn-states runs/archive_spawns_cells.npy --spawn-states-frac 0.3
  --respawn-snap 0.25 --respawn-margin 0.5`, original reward, SELF_STATES declared through box.py
  --self-states with that provenance. Watch: the novel ceiling (deepest REAL episode) and reservoir states
  west of the wall; a finish from an archive state is harvested (0.5 s margin) into the reservoir.
