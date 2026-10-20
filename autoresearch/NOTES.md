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

## Round 1, branch r1b3 - 2026-10-20 00:49 UTC (first entry)

### The CP2 wall, measured on the incumbent's own evals (ANALYSIS ONLY, nothing enters training)

* ref_surf 3,775M ep 1 (27.50%), the line after the window: the hole at x ~1,030 (19.75 s), a
  turn south and back east along y ~-800, a ramp launch at x ~2,400 (22.1 s, vz -311 -> +476),
  free flight up to z 12,360 at x 4,251 (23.5 s), then a MID-AIR U-turn (yaw 345 -> 201 in
  0.75 s) at x ~4,600 and a straight dive west into the fail at (3,007, -909, 11,219), 25.19 s,
  vz -1,352. All 9 episodes end there (x 3,007-3,113, z 11,172-11,224).
* GR's 755M eval (the incumbent): the same wall (27.22%), plus earlier deaths at 22.5%
  (x ~1,600, z ~11,600) and 25-26% (x 4,191-4,979, z 11,932-12,296).
* CP2 (sgcp_2) is the plane x 768..816, z 10,060..11,300: 2,200 u WEST of the death point.
* Hull traces (scratchpad geom.py, measurement only): the ramps the human turns around on
  (ledger: (5,554, 454, 11,079) at 25.5 s) are at x 4,600-5,800, y about +-600, z 11,000-11,330,
  normal_z 0.68. That is ~1,000 u BELOW the agent's U-turn altitude; above them (z 11,900-12,350)
  x 4,900-5,800 is wall. So the crossing is a different line from ~22 s on (lower, to the side),
  not a later U-turn. Several seconds of different behaviour.

### What that rules out (reasoned from the ledger's numbers)

* **Per-decision temperature.** The policy's view sigma is 0.02-0.045 in pre-tanh z, ~1 deg of
  yaw offset; T = 2 doubles it. The surf run's 1.5B steps at T = 2 and the cap-4 lineage's 1.8B
  at T = 4 had essentially no view exploration; only the keys were tempered.
* **--death-charge.** With the recipe's 0.5/s time penalty, an episode that progresses at ~0.85
  shaping units/s nets (0.85 (1 - kappa) - 0.5) per second, which is negative for kappa > ~0.4
  (fast suicide, xNGS). At kappa 0.3 the dive at CP2 still beats 15 s of wandering into a stall
  kill. Not tried.
* **Spawning past the wall** (searched states, --respawn-frontier) is teleporting forward, which
  the user ruled out for --respawn-novel. Not tried.

### The plan

1. Warm resume of the incumbent + `--view-ou-sigma 0.8 --view-ou-period 40` (existing code:
   a per-env yaw offset c ~ N(0, 0.8) in pre-tanh z, held 40 decisions = 1.6 s, added to the
   EXECUTED z only, so PPO stays on-policy). These are the constants gbCELou/gsCELou ran on
   celestial, so nothing here is read off this map. At mu = 0, 0.8 is ~25 deg of held offset,
   25x what T = 2 gives the view. Question: does committed view exploration at the wall put
   training episodes past CP2 (reservoir reach, training frontier), and does the greedy line
   follow, as it did at the window?
2. `--view-ou-temp` (new): the OU sigma scaled by the unstuck temperature, sigma x T / T_max,
   so committed exploration is OFF while the frontier moves and full at the cap. That is the
   generic, from-scratch form of (1) if (1) shows anything.
3. No box at 00:49: every offer under the caps is blocklisted or above the cap (3090s 0.381,
   4090s 0.455). Retrying every wakeup.

### 2026-10-20 01:14 UTC - `--view-ou-temp` built (commit cb0d58e), box trouble

* **What:** under `--unstuck`, the held yaw offset of `--view-ou-sigma` gets sigma
  `--view-ou-sigma x min(T, --unstuck-max) / --unstuck-max`. T = 0 (frontier moving): no offset;
  T at the cap: the full sigma. `set_ou_temp` runs wherever the schedule moves T; a held offset
  keeps its draw until its own redraw. The step line prints `ou <sigma>` after the unstuck T.
  Refused without `--view-ou-sigma > 0` or without `--unstuck`. ckpt restores; record_ckpt
  TRAIN_ONLY.
* **Tests:** `tests/python/test_view_ou_temp.py`, 6/6 on CPU (5.6 min): the formula, the refusals,
  a sidistic CPU smoke (T 0 -> 0.5 -> 1.0 -> 1.5 and `ou` 0 -> 0.10 -> 0.20 -> 0.30 = 0.8 T/4), the
  config dump, a flagless resume restoring it, and flag-off bit-identity against the parent
  trainer with `--view-ou-sigma 0.8 --view-ou-period 4` on (config, every progress.csv value,
  the greedy eval trajectory bytes).
* `tests/python/test_view_ou_sigma.py`: 14/15. `test_period_needs_a_sigma` fails at HEAD too: it
  launches the default 2048-env trainer on CPU, which this 8 GB host OOM-kills (exit 137) before
  the guard it tests. Environmental, not touched.
* **Boxes:** 01:06 a 3090 Ti at 0.242 $/h came up power-capped (300 of 450 W, 59 bf16 TFLOPS,
  gpu_health UNHEALTHY); blocklisted (machine 68144) and released per CLAUDE.md. Since then no
  offer under the caps: the 3090s at 0.269 / 0.381 are blocklisted hosts, 4090s start at 0.455.
  A background waiter retries every 3 min and deploys on success.

### 2026-10-20 01:47 UTC - box up; r1b3_ouw (ungated OU) is a clear NEGATIVE; control running; --view-ou-front built

* **Box:** vast 56818185, RTX 4080 16 GB, 0.267 $/h, 32 threads, ssh -p 40030 root@46.231.69.36.
  Healthy. Resume runs ~230-250k steps/s (GPU 99% busy: this card is GPU-bound, unlike the
  CPU-bound 3090s). 4e8 steps = ~27-30 min.
* **r1b3_ouw** = warm resume of the incumbent (749.7M) + `--view-ou-sigma 0.8 --view-ou-period 40`
  (celestial's constants), every env, every window. Training episodes fell from ~1,070 to ~300
  decisions and ep reward from ~7 to ~0.3 immediately. First greedy eval (825.75M): **4.4%**
  (frontier 4,378 u, dying at the first ramp), from 25.4% at 750M. The reservoir lost its deep
  states (mind 74.3% -> 76.0%) as short perturbed episodes refilled it. Stopped at 847M. Same
  verdict as pnOU (petrus) and gbCELou (celestial): a held 10-60 deg view error on every env
  wrecks a precise line. Scores 25.42 (750M), 4.48 (825M).
* **r1b3_ctl** = the same resume with the recipe unchanged (the matched control for every warm
  arm; nobody had run GR past 758M). Evals 750M 27.18, 825M **27.335** (CP1, at the CP2 wall).
  Training rew ~7.2, len ~1,090, mind 74.24%, T 0.14 and falling.
* **Built for the next arm** (commits 6f9a5d3, 3cf6aa4; default off, flag-off bit-identity
  re-proved, 17 tests in tests/python/test_view_ou_temp.py):
  * `--view-ou-frac p`: a redraw gives an offset with probability p, else exactly 0 for that window.
  * `--view-ou-front F`: a redraw gives an offset only if the env's current d is within F x d0 of
    the reservoir's deepest state (re-read every rollout). The line before the wall flies clean;
    the committed exploration lands where the policy dies. Step line: `ouF <edge>u in <share>`.
    No map constant: the zone is the policy's own frontier plus a round 5% of d0.
* **Next arm r1b3_ouf:** warm, `--view-ou-sigma 0.8 --view-ou-period 40 --view-ou-front 0.05
  --view-ou-frac 0.5`, 4e8 steps, against r1b3_ctl at matched steps. Training-side signal: mind
  below ~72.5% (states harvested past the pit); then the eval frontier past 27.5% / CP2.

### 2026-10-20 02:55 UTC - gated OU null; Go-Explore search from the policy's own line stalls AT the pit; velocity-direction respawns launched

* **r1b3_ctl final** (recipe unchanged, 750M -> 1.15B, 268k steps/s): every eval 27.2-27.4% (best
  **27.443** at 1,052M), 9/9 die at x 3,100-3,180, y ~-900, z ~11,170 at 24.7-25.2 s. T rose to 1.06.
  **r1b3_ctl2** (the same, continued 1.15 -> 1.34B, T reached the cap 2.0): 27.33 / 26.42 / 27.35,
  reservoir mind 74.15% -> 73.47%, no crossing. GR at T = 2 behaves like the surf run at T = 2.
* **r1b3_oufo** (`--view-ou-sigma 0.8 --view-ou-period 40 --view-ou-front 0.05 --view-ou-frac 0.5
  --view-ou-offpol 1`, warm, 400M): training line intact (rew ~6.3, len ~990, matching the control),
  ~25% of decisions inside the zone, ~9% offset and dropped. Evals 27.29-27.33%; mind 74.38%. NULL:
  ~22M held-offset decisions at the wall, not one training episode past the pit.
* **Go-Explore search (ANALYSIS ONLY; tools/explore_phase1.py, nothing enters training):** roots = 650
  of the control policy's own greedy-line states at 18.0-24.5 s (record_ckpt --dump-states), 512 envs,
  bursts of 100 decisions at 95% repeat, 256 u position cells. The archive saturated at 1,262 cells
  in ~8 min / 16M exploration runs. Deepest state on the goal_32 field: d 70,484 (27.91%) at
  (2,647, -997, 11,379), FALLING at vz -987. Every deep cell is a falling state at the pit. It explored
  the whole room east (x to 6,224, z 10,780-13,445), including the human's turn-around ramps, but
  only ever falling through them. Note: its progress print uses the stale goal_16.npz (Aug 23,
  pre-pane-fix: "start distance 7447u"); score its archive with goal_32 instead.
* **Why (measured, analysis only):** the archive keeps the FIRST state that enters a 256 u position
  cell, and the key has no vertical velocity, so the corridor cells are owned by states diving
  through them. The field (goal_32) along the human's line: its 24.5 s point 72,062; the turn-around
  ramp (5,554, 454, 11,079) 73,630 (a 1,580 u dip); its level pass (2,882, 0, 11,469) 71,352, still
  WORSE than the agent's death point 70,882; then (2,200, 0, 11,400) 70,828, the lower passage
  (1,700, 0, 10,900) 69,214 and CP2 (792, 0, 10,700) 67,560. The shaping pays for the dive at
  every tick until the detour is ~3 s along. The lethal surface the agent dies on is a floor at
  z ~11,100-11,170 (point probes, scratchpad trigprobe.py); the brushes' AABBs are not the lethal
  volumes.
* **New: `--respawn-vel-rot P` (commit 31bdab4)** - a share P of the spawn pool's reservoir rows that
  lie within `--respawn-vel-rot-front` F x d0 (0.05) of the reservoir's deepest state get a fresh
  velocity DIRECTION: uniform heading, elevation U(-45, 45) deg, speed kept, view yaw turned with it.
  The position is the policy's own (never past its real frontier). Florensa-style start-state
  perturbation of HOW the agent moves where it already is. Default off; flag-off bit-identity proved
  (6 tests). Launched as **r1b3_vrot** (P 0.5, F 0.05), warm from the incumbent, 400M, against
  r1b3_ctl. Only ~2% of the reservoir rows fall in the zone x P (the uniform reservoir holds ~4% of
  its states there), ~9 rotated episodes per iteration.

### 2026-10-20 03:28 UTC - vel-rot NULL; vz-keyed search saturates at the pit too; cell-uniform frontier respawns launched

* **r1b3_vrot** (`--respawn-vel-rot 0.5 --respawn-vel-rot-front 0.05`, warm, stopped at 1.033B):
  the rotated share grew 2% -> 14% of reservoir rows as the zone filled; evals 27.0 / 27.4 / 27.4 /
  27.4% (best 27.427); mind 74.15-74.51% throughout. NULL: ~280M steps of random velocity
  directions at the policy's own frontier positions, no training episode past the pit.
  (It ran at ~94k steps/s for ~15 min while a CPU search shared the box: on this 4080 box a
  concurrent 8-thread search cuts the trainer ~3x. Do not share the CPUs.)
* **Search, vz-keyed** (`explore_phase1 --cell-vz 6`, new, commit 4bf7dc5; 8 threads, 20 min, 5.5M
  runs, same 650 roots): 4,566 cells (vs 1,262 position-only), saturated by ~3 min (4,005) ->
  4,288 at 19 min. Deepest d 70,453 (27.94%), again every deep state FALLING (vz ~-1,000) at
  x ~2,630. Random macro-actions do not find a level corridor entry even when level states own
  their own cells.
* **The plates (bsp AABBs + point probes, ANALYSIS ONLY):** thin lethal plates *84 (z 12,256,
  x 3,686-4,476) and *45 (z 11,904, x 4,194-5,344) lie under the agent's apex and U-turn; the
  human's 24.5 s point (4,041, 661, 11,814) is BELOW *45, i.e. the human reaches the turn-around
  ramps (x 4,600-5,800, z 11,000-11,330) under the plates. So the line splits at the ~22 s launch
  ramp (the agent launches up to z 12,360), ~3 s before the death. A short perturbation at the
  U-turn cannot produce it, which fits every null so far.
* **New: `--respawn-cells S` (commit 7b954d7)** - Go-Explore SELECTION over the policy's own states:
  a share S of the reservoir rows drawn uniformly over cells (256 u x 6 vz x 4 horizontal speed)
  among reservoir states within F x d0 (`--respawn-cells-front`, 0.05) of the deepest one. The
  policy is the explorer. Default off; 11 tests incl. flag-off identity. Launched **r1b3_cells**
  (S 0.5, F 0.05, warm, 400M): 1,843 rows per pool over 83-89 zone cells.

### 2026-10-20 04:00 UTC - cells NULL; kill-aware field keeps the dip; from-scratch recipe test launched

* **r1b3_cells** (`--respawn-cells 0.5 --respawn-cells-front 0.05`, warm, 400M): 1,843 rows per pool
  drawn uniformly over 207-227 zone cells; ~45% of episodes start at the wall (train rew ~4.5, len
  ~700). Evals 25.8 / 27.3 / 27.4 / 27.44 / 27.41 / 27.41%; mind 74.0-74.4%. NULL.
* **Kill-aware field (ANALYSIS, `build_goal_field(mask_kill=True)`, 131 kill volumes masked, baked on
  the box in 258 s, cached as maps/surf_src_sidistic.goalk_32.npz there):** every distance shifts by
  ~+1,000 u and CP2's structure is unchanged: the human's ramp is still a +1,564 u dip (74,629 vs
  73,065), and the agent's death point (71,894) still beats the human's level pass (72,361).
  `--race-kill-aware` cannot help here.
* **Tally at CP2 (all warm from the incumbent, 400M each, vs r1b3_ctl):** ungated held view offsets
  (collapse), gated off-policy held offsets, random velocity directions at frontier respawns,
  cell-uniform frontier respawns: 0 training episodes past the pit in ~1.3B steps. The control at
  T = 2 (ctl2): same. Search (position and position+vz cells): saturates at the pit.
* **r1b3_sou (launched 03:56):** FROM SCRATCH, the incumbent recipe + `--view-ou-sigma 0.8
  --view-ou-period 40 --view-ou-front 0.05 --view-ou-frac 0.5 --view-ou-offpol 1 --view-ou-temp 1`.
  T-scaled, frontier-gated, off-policy held view offsets: zero while the eval frontier moves, half
  the 1.6 s windows offset (sigma up to 0.8 at T = 2) inside 5% of d0 of the reservoir's frontier
  when stuck. The question is the early gates, which were turns the policy would not commit to (the
  first ramp "just has to turn left"). GR reference: trap exit 529M, window 755M.

### 2026-10-20 04:36 UTC - r1b3_sou (scratch, recipe + T-scaled gated held offsets): at the window by 302M, held by T = 0; continued as r1b3_sou2

* **r1b3_sou** (from scratch, 600M, RTX 4080 ~272k steps/s): the eval frontier read 1,156 / 4,497 /
  **8,232** (151M) / **14,644** (227M) / **20,424** (302M) / 20,271 / 20,301 / 20,309 u. Against GR at
  matched steps it is ~380M steps ahead: GR's frontier best was 4,337 u from 76M to 529M (8,573 at
  529M), 15,061 at 605M, 20,546 at 680M. **The mechanism did nothing in that phase:** the frontier
  kept moving, so T stayed 0.00 and `ou` 0.000 (no offset, no dropped row) until ~500M. It is run-to-run
  variance (one seed, a different card): the first-ramp trap exit has now ranged from < 151M to
  1,510M across five runs of near-identical recipes. Record it as noise, not evidence.
* It then sat at the window wall (20.3-20.4k u, 20.8%) from 302M; T rose from ~500M (0.48 at 600M,
  `ou` 0.195, 15% of rows offset and dropped).
* **The window is the cleaner test of held offsets:** its pass is a held 35-40 deg cut-back for ~0.4 s.
  GR / surf / cap2 passed it 75-150M steps after arriving, with T still elevated (decaying from the
  trap exit, 0.26-0.70 when passed); this run arrived with T = 0 and stalled, which fits "T passes the
  window". Continued as **r1b3_sou2** (resume of ckpt_final, every flag restored, +500M): a pass at
  T < ~1 would credit the offsets; a pass only at T >= 1 would not.

### 2026-10-20 05:06 UTC - r1b3_sou2: held offsets do NOT pass the window either (T to the cap, sigma 0.8, 450M steps); Go-Explore-in-the-loop launched on that stuck checkpoint

* **r1b3_sou2** (resume of r1b3_sou's 600M ckpt, +500M, every flag restored): T 0.48 -> 1.0 at
  ~700M -> 2.0 (cap) at ~900M; `ou` 0.195 -> 0.8; ~10-15% of rows offset and dropped. Evals 600M-978M:
  20.30 / 20.29 / 20.26 / 20.31 / 20.26 / 20.32k u (20.7-20.8%), mind 79.3-79.8%. NULL at the window
  in 800M steps since arriving (302M), 450M of them with T >= 0.5 and the offsets live. GR / surf /
  cap2 passed this gate with T alone 75-300M steps after arriving, so the frontier-gated held
  offsets did not help and may have hurt (10-15% of the zone's rows leave the PPO pool).
  **Verdict on `--view-ou-*`: null at the window from scratch, null at CP2 warm; destructive ungated.**
* **r1b3_gew (launched 05:03)** - Go-Explore in the loop, on that stuck checkpoint (a known-passable
  gate with an 800M-step stuck record): `--respawn-cells 0.5 --respawn-cells-front 0.05
  --respawn-cells-margin 0.25 --respawn-vel-rot 0.5 --respawn-vel-rot-front 0.05` (+ the restored
  offsets). Cell rows: 1,843 per pool over ~730-760 cells of the 0.25 s-margin archive, returned
  EXACTLY (no rotation, commit 7d4e8d2); ~10% of the other reservoir rows rotated. +400M.

### 2026-10-20 05:31 UTC - Go-Explore in the loop (r1b3_gew) NULL at the window; offsets-off diagnostic launched

* **r1b3_gew** (sou2's 1.1B ckpt + `--respawn-cells 0.5 --respawn-cells-front 0.05
  --respawn-cells-margin 0.25 --respawn-vel-rot 0.5`, with the restored offsets, 400M): 1,000-1,070
  frontier cells returned exactly each pool, ~10% rotated rows, T 2. Evals 20.79 / 20.94 / 20.76 /
  20.86 / 20.73 / 20.76%; mind 79.44-79.47% throughout. NULL: no state past the hole in 400M.
* All 9 episodes of this lineage die at the same window wall as sidGEOu: (1,206-1,277, -21..-133,
  12,811-12,904) at 18.7-19.5 s, on top of fail *89, vy -100..-590 (moving AWAY from the ridge
  after landing on the A-frame: the cut-back not taken).
* **The confound:** every run of this lineage (sou, sou2, gew) carried the held offsets, and it is the
  only lineage stuck at the window at T = 2; GR / surf / cap2 passed it with T alone. **r1b3_nou**
  (launched 05:29): sou2's ckpt with every `--view-ou-*` OFF (`--view-ou-sigma 0 --view-ou-period 0
  --view-ou-temp 0 --view-ou-frac 1 --view-ou-front 0 --view-ou-offpol 0`; T 2 restored), +400M.
  A quick pass says the offsets were BLOCKING the cut-back. Note for resumes: a checkpoint carrying
  the OU flags refuses `--view-ou-sigma 0` alone (the restored period/temp/frac/front/offpol each
  need a sigma) - pass all six.

### 2026-10-20 05:56 UTC - THE HELD OFFSETS WERE BLOCKING THE WINDOW: with them off the stuck checkpoint passes in < 75M steps; new branch best 27.517 (from scratch)

* **r1b3_nou** (sou2's 1.1B ckpt, every `--view-ou-*` off, T 2 restored, +400M): evals 1.101B 20.76 ->
  **1.177B 24.45 (CP1, through the hole)** -> 1.252B 27.44 -> 1.327B 27.45 -> **1.403B 27.517** ->
  1.478B 27.50; mind 74.4%, T falling 2.0 -> 0.3 after the new best. The same checkpoint had sat at
  the window for 800M steps with the offsets live (sou2, gew). So the frontier-gated off-policy held
  offsets did not just fail to help: they BLOCKED the cut-back the plain T = 2 rollouts learn in one
  eval period. Likely mechanism: 10-15% of the zone's decisions (the zone being the window) left the
  PPO pool and the rest learned around 1.6 s offset windows. **Do not use `--view-ou-*` at the frontier.**
* **The lineage r1b3_sou -> sou2 -> nou is from scratch** (the incumbent recipe; the OU flags were
  inert at T = 0 to ~500M, live 500M-1.1B, off after): first-ramp trap < 151M, window wall 302M,
  through the window 1.177B, CP2 wall 1.25B, best eval **27.517 at 1.403B** (branch best, above the
  round-0 best 27.5). Its 1.5B ckpt_final is pulled: runs/r1b3/r1b3_nou/ckpt_final.pt.
* **r1b3_gec (launched 05:55):** that checkpoint (at the CP2 wall, OU off) + `--respawn-cells 0.5
  --respawn-cells-front 0.05 --respawn-cells-margin 0.25 --respawn-vel-rot 0.5` (Go-Explore in the
  loop without the offsets), +400M.

### 2026-10-20 06:23 UTC - Go-Explore in the loop NULL at CP2 too; last experiment: non-episodic novelty keyed on climb (from scratch)

* **r1b3_gec** (nou's 1.5B ckpt at the CP2 wall, offsets off, `--respawn-cells 0.5 --respawn-cells-front
  0.05 --respawn-cells-margin 0.25 --respawn-vel-rot 0.5`, 400M): ~2,000 frontier cells returned
  exactly per pool, ~14-16% rotated rows, T back to the cap 2.0. Evals 27.51 / 27.48 / 27.51 / 27.30 /
  27.51 / 27.52%; mind 74.6% -> 73.7% (the reservoir crept ~1 s further along the same dive, still
  before the pit). NULL. Every in-loop exploration lever built this round is now null at CP2.
* **r1b3_isp (launched 06:20), from scratch:** the incumbent recipe + `--int-split --int-climb 3
  --int-view 0`. Burda et al. 2019's two-head critic (the count bonus in its own non-episodic stream,
  gamma_I 0.99 per decision, A = A_E + A_I), with the novelty key = 256 u cell x 3 horizontal-speed x
  3 climb-angle bands (no view sectors, so turning in place earns nothing). The idea: a dive that dies
  no longer cancels the value of reaching a never-seen (place, vertical motion), e.g. level in CP2's
  corridor, and the climb band separates that from diving through it. 30-min check against GR's
  marks (trap exit 529M, window 755M); stopped if behind. CPU smoke of the exact combination passed.

### 2026-10-20 07:57 UTC - r1b3_isp stopped; ROUND 1 LESSON (r1b3) - box released

* **r1b3_isp** (from scratch, recipe + `--int-split --int-climb 3 --int-view 0`): first-ramp trap exit
  between 529M and 605M (GR: 529M), then 14.0% (13.6-13.8k u, a minor gate at x ~5,850 before the
  window) at every eval from 755M to 1.359B, ~630M steps, T at the cap. Stopped at 1.39B by the rule
  set at 07:30. Behind GR (window passed at 755M). Non-episodic novelty did not help it through a gate
  the plain recipe passes; it never reached CP2, so its CP2 question stays open.
* Box 56818185 (RTX 4080, 0.267 $/h, ~268k steps/s warm, 240-270k scratch) released at 07:56,
  6.5 h of runs. Credit 98.5 at the end.

#### What round 1 (r1b3) knows that round 0 did not

1. **CP2's anatomy (analysis only, measured):** the passage west of the pit is a corridor ~220 u tall
   between lethal volumes (*86 top z 11,332 below, *99 bottom z 11,552 above, x 2,528-3,680), entered
   level from BELOW: the human comes up from a turn-around ramp (x 4,600-5,800, z ~11,000-11,330, under
   the lethal plates *84 z 12,256 and *45 z 11,904) and levels at z 11,469. The agent's line launches
   off the side ramps (x 2,200-2,900, y +-600, ~22 s) up to z 12,360 OVER those plates, U-turns
   mid-air at x ~4,600 and dives through the band at -1,000..-1,400 u/s onto a lethal floor at
   x ~3,000-3,100. The line splits ~3 s before the death. Three independently trained lineages (GR,
   surf, r1b3's from-scratch one) die at the same point (x ~3,000-3,100, y ~-900, z ~11,200).
   The goal field pays the dive at every tick until the human's detour is ~3 s along (its level pass
   71,352 is WORSE than the death point 70,882); the kill-aware field keeps that dip.
2. **Nulls at CP2 (each 400M warm, vs the control r1b3_ctl 27.2-27.44%):** held view offsets
   (`--view-ou-*`: ungated collapses the line to 4.4%; frontier-gated off-policy: null), random
   velocity directions at frontier respawns (`--respawn-vel-rot`), cell-uniform frontier respawns
   (`--respawn-cells`, also from a 0.25 s-margin archive with exact returns), T = 2 on GR (ctl2).
   Reward-free Go-Explore search from the policy's own 18-24.5 s states (position and position+vz
   cells) saturates AT the pit in ~8 min. None put one training episode past the pit.
3. **Held view offsets BLOCK the frontier.** A from-scratch checkpoint sat at the window for 800M
   steps with the gated offsets live (T to 2); with them off it passed the window within ONE eval
   period and reached CP2 (r1b3_nou). Do not use `--view-ou-*` near the frontier.
4. **Time-to-gate is seed-dominated:** the first-ramp trap exit ranged < 151M .. 1,510M across six
   near-identical recipe runs, the window 302M .. never-in-800M. One-seed time-to-gate comparisons of
   from-scratch recipes need large effects (CLAUDE.md's gate-ladder warning, re-measured).
5. **Best eval 27.517** (r1b3_nou 1.403B, 9/9 at the CP2 pit, x 2,995-3,039), from a FROM-SCRATCH
   lineage on the incumbent recipe (r1b3_sou -> sou2 -> nou; ckpt_final 1.5B pulled to
   ~/ar/runs/r1b3/r1b3_nou/ckpt_final.pt). It is the same wall as 27.5 - a few units deeper, not
   progress.

#### What to try next (my ranking)

* CP2 is a precision entry that local noise, start perturbations and position-keyed archives do not
  find. What is left in the generic space: search with the POLICY as the mover (it can ride the
  turn-around ramp; random macro-actions cannot) from its own states, with velocity in the cell key
  and a short harvest margin, and its finds fed back by self-imitation (expert iteration); or the
  planner (paused by the user on 10-18) - the problem is the one 0c describes.
* Tools this branch leaves (all default off, bit-identical off, tested): `--view-ou-temp/-frac/-front/
  -offpol` (documented negative), `--respawn-vel-rot`, `--respawn-cells` (+ `-front`, `-margin`),
  `explore_phase1 --cell-vz`. Scratchpad analysis scripts are not in the repo.

### 2026-10-20 08:41 UTC - the round continues (wrap-up is 11:50): Go-Explore in the loop + self-imitation at CP2

* RESULT.json set back to `final: false` (the 07:57 wrap-up was early; the round runs to wrapup_at).
* **Next arm r1b3_ges:** resume r1b3_nou's 1.5B ckpt (from scratch, at the CP2 wall, offsets off) +
  `--respawn-cells 0.5 --respawn-cells-front 0.05 --respawn-cells-margin 0.25 --respawn-vel-rot 0.5
  --sil-coef 0.1`. Why: r1b3_gec made the policy explore from rare frontier states, but anything rare
  it found was averaged away by PPO's on-policy step; SIL (Oh et al. 2018, existing flag) replays every
  transition whose return beat the critic, so a single lucky line past the pit would be imitated
  instead of diluted (expert iteration with the policy as both mover and student). CPU smoke of the
  exact combination: SIL buffer filling (~20k rows), loss finite, cells / vrot shares as configured.
* Read: mind below ~72.5% (a harvested state past the pit) and `sil gain` falling as the critic
  catches up; the eval frontier past 27.6% / CP2. A side effect to watch: SIL can also make the
  dive itself creep west (a deeper death, not a crossing) - judged by CP2, not by the score.

### 2026-10-20 09:18 UTC - r1b3_ges (Go-Explore + SIL) DEGRADES the line; relaunched as r1b3_gest with the sampling temperature off

* **r1b3_ges** (nou's 1.5B ckpt + cells 0.5 / front 0.05 / margin 0.25 + vel-rot 0.5 + `--sil-coef 0.1`,
  3090, stopped at ~1.82B): evals 27.51 / 27.49 / **16.05** / 27.37 (median 14.6) / **20.53**%; mind
  73.7-73.8%, nothing past the pit; SIL buffer at its 50k cap every rollout (~40% of transitions beat
  the critic), SIL loss rising 2.3 -> 3.8 as T rose 1.6 -> 2.0. The arm without SIL (gec) held
  27.3-27.5%, so SIL is what broke the line.
* **Why (a bad interaction, not a SIL verdict):** under `--unstuck` the rollout draws the view from
  N(mu, (1+T) sigma) and SIL scores those stored z under the UNTEMPERED policy, so at T = 2 it imitates
  high-temperature noise with large gradients. Anyone combining `--sil-coef` with `--unstuck` should
  either turn the sampling temperature off or score SIL rows under the temperature they were drawn at.
* **r1b3_gest (launched 09:17):** the same + `--unstuck-temp 0` (T still scales the entropy and novelty
  coefficients, no longer the sampling), so SIL imitates the policy's own on-distribution actions and
  the frontier diversity comes from the cell / rotation starts. +1.6B budget; runs to the wrap-up.
