# NOTES - the surf_za_loop lineage

The lineage's memory: every round's winner appends what it learned (what worked, what did not,
the numbers). Read it before you start; append to it before you wrap up.

## Round 0 (operator, 2026-10-31)

* A new lineage on a new map. The code on this branch is the unitfarmer2 lineage's best branch
  (ar/uf2r2b4, b93a7151: both unitfarmer2 rounds' tools) with the operator's harness. Earlier
  lineages' notes are NOT copied here; they are in this branch's git history:
  `git show b93a7151:autoresearch/NOTES.md` (unitfarmer2, rounds 1-2: its start pit, its own
  60-65% loop, the real-finish fix, the self-search loop that beat the record in the simulator)
  and `git show 8a825803:autoresearch/NOTES.md` (princessburglar, rounds 1-9 and the operator's
  expert loop: Go-Explore with energy cells pumped a half-pipe out of the start room).
* The map files came from the user on 2026-10-31 (a zip with the BSP, md5 e3ad293d, and its
  sky; the record's demo from speedrun.eu, analysis only). Zones: auto-detected and then
  declared (`maps/surf_za_loop.zones.json`, source manual): the start curtain *14, the FIVE
  finish slabs *15-*19 (any of them finishes; the harness's several-trigger finish), and the
  spawn at the start teleport's destination (every spawn of the map stands on a teleport there).
* The operator baked the caches on one box (smoke launch zlop_bake, RTX 5080; the goal field
  seeded on the five slabs: 8,141,468 reachable voxels, start d 6,361.8 u). The bake hit a
  latent crash in `--respawn-front-perturb` (an empty frontier band while every reservoir state
  is behind the start), fixed on this branch. The check launch zlop_check then ran 30M steps
  at 347k steps/s: the record gate passed (greedy, stoch, mixed, POV); its first eval (0.5M
  steps) scored 0.028, best progress 0.0%, 0/9; greedy episodes fell into the bottom teleport
  at 18-25 s; 62-68% of training episodes ended by the stall kill.
* The record's demo, parsed, is in `$AR_HOME/wr/`: `surf_za_loop_wr.jsonl` (a 10 ms resample),
  the raw 7 ms frames (`.frames.npz`) and the movevars. It is for ANALYSIS ONLY (CLAUDE.md
  section 0). It never becomes a spawn file, a spine, a target sequence, a BC set or a reward
  constant. PROTOCOL.md READ FIRST has what it shows.

## Round log

## zlr1b2 (round 1, slot 2) - 2026-10-31 17:10 UTC start

Box 58559243: RTX 5080 + Ryzen 9 7945HX (32 threads), 0.387 $/h, South Carolina (rented 17:14).

### The first loop, read off the record (ANALYSIS ONLY, CLAUDE.md section 0)
* The "loop" is a half-pipe along x: from the curtain the record drops to z ~-3,824 (t+6, 1,169 u/s)
  and then swings x -1,900 <-> +1,200 at y 50..800, z -3,330 (bottom) .. -2,600 (ends), one swing
  every ~2-4 s. E = z + |v|^2/1600: -3,180 at rest, -2,970 at t 6, -2,179 at t 34 (+28 u/s), a
  -780 u loss at t 34-38 (a mistake), back to -2,010 at t 64 (+34 u/s); at t ~68-70 it leaves the
  east end for a second half-pipe at y ~-600, z ~-2,250 (x -650 <-> 2,300), still +22 u/s of E.
  The field pays nothing until then (progress negative the whole first minute).
* Perfect air strafing (wishspeed cap 30 u/s per frame, perpendicular) is ~+80 u/s of E at 7 ms:
  the record pumps at a third to a half of that.

### Two rules that kill every pumping line, whatever the policy learns
* Training: `--stall-secs 15` kills an episode with no field record by > stall_eps in one call.
* Evals: `--eval-patience 30` cuts a greedy episode whose best field progress has not grown 32 u
  in 30 s. **The record's own line would be cut at 30 s with 0 progress** (it gains its first
  field progress at ~65 s). So the score cannot move past ~0 on this map under the recipe's eval,
  even for a policy that rides the loop like the record.
* Fix (7e5b6d0f, generic, flags off = bit-identical): `--stall-energy E` and
  `--eval-patience-energy E`: either clock ALSO re-arms when the episode's best energy height
  h = z + |v|^2/2g rises more than E above its value at the last re-arm (cumulative). Free fall
  and frictionless surfing keep h; only air acceleration / jumps raise it; h is bounded (height +
  sv_maxvelocity), so a farm still stalls once it saturates. E = 32 u (the patience rule's own
  32 u). Would I ship it on a map I never opened? Yes: it changes nothing where the field moves,
  and where it does not, a line still adding energy is not idle.
  Test: tests/python/test_stall_energy.py.

### Plan (17:23 UTC)
* `zlr1b2_ctl`: the recipe from scratch (control), 5080, 6e8 budget, stopped at ~2.3e8 to free
  the box. 0.5M / 76M / 151M greedy: fwd 49 / 83 / 311 u, peak speed 87 / 153 / 297 u/s,
  patience-cut 9 / 9 / 4 of 9, frontier 0 / 0 / 31 u. Training: 68-79% of episodes stall-killed,
  energy term (front bonus form) pays ~0, yaw sigma climbing to 1.2 (no signal).
* A1 `zlr1b2_en`: recipe + `--stall-energy 32 --eval-patience-energy 32` + PER-CALL energy
  shaping `--energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0` (pb_r26's form and K,
  carried unchanged from the princessburglar expert loop; not tuned here). Why K 12 and not the
  recipe's 3: the time penalty is 0.1 reward/s; at K 3 a pump at the record's ~30 u/s pays
  0.09/s, so a learner pumping 10 u/s is paid less than it is charged and dying early is the best
  loop-1 policy. At K 12 the same learner breaks even and the record's pace pays 3.6x the time
  penalty. Read with: energy gained per episode, stall share, greedy peak speed / E, where the
  greedy dies, and whether any greedy episode reaches the second half-pipe (progress > ~5%).

### Correction: the half-pipe's ends DO pay on the field (17:28 UTC, analysis of the record)
* config.field along the record (running-min progress, what the ratchet pays and the grader
  scores): 0.0% at t 10, **5.6% at t 12.5**, 6.2% at t 27.5, 7.7% at t 30, **10.2% at t 32.5**,
  flat to t 62.5, 12.6% at t 65, 17.5% at t 70 (second half-pipe), 23.7% at t 87.5. The ends of
  each swing climb toward the goal (z -2,856 at x -1,746: 3.1%; z -2,982 at x 1,129: 5.5%), so
  every higher swing is a new record: pumping IS paid, in steps, and the score can reach ~10%
  inside the first half-pipe. (PROTOCOL's "progress negative the whole first minute" reads the
  instantaneous d, not the running min.) The gaps between records are 10 s and 32 s: the
  training stall (15 s) and the eval patience (30 s) still cut the record's line without the
  energy re-arm.
* The first gate is the drop: from the curtain the record goes down a ramp to z -3,875 (t 5,
  1,172 u/s; the bottom teleport is at z -4,000) and up the far side into the half-pipe. The
  control's greedy at 151M falls to z -3,960 (dead) in 5 of 9 episodes and idles in the rest;
  its energy height never rises above the spawn's (h max -3,071 vs -3,074).

### A1 `zlr1b2_en` (energy re-arms + per-call energy K 12): SUICIDE, stopped at 382M (17:41 UTC)
* Greedy at 76 / 151 / 227 / 302 / 378M: every episode dives into the bottom teleport in 3.4-4.9 s
  (z -3,960; h max = the spawn's), path 224-509 u, peak 88-302 u/s; score <= 0.38%.
* Training: by 300M 92-93% of episodes end in a crash and 2-3% by the stall (the control at 208M
  the same: 93-94% crash); episode length 467; energy +167/-296 u/ep (net LOSS); yaw sigma pinned
  at 2.718 (no signal); the unstuck temperature had fired (T 0.20).
* Why: dying is free and living is charged. The time penalty is 0.1 reward/s; per-call energy at
  K 12 charges a learner ~1.2 for its first landing (the 100 u drop off the spawn ledge) and for
  every contact after; nothing charges a death. The cheapest episode is the shortest one, and
  both the control and A1 found it (A1 faster). The energy re-arm never got a pumping line to
  protect.

### A2 `zlr1b2_er` (17:42 UTC): the death made neutral in time, the energy term made non-negative
* `--fail-pen 2`: a death (fall, stall kill) costs the DISCOUNTED VALUE OF THE TIME-PENALTY STREAM,
  tp_tick / (1 - gamma_tick) = 0.0007 / 0.00035 = 2.0 (= 0.001 / 0.0005 at the 10 ms reference:
  the recipe's own time_pen and gamma, no map constant). With it, dying at any time t costs the same
  discounted total, tp/(1-g) x [(1 - g^t) + g^t]: WHEN to die is neutral, and any positive term
  (energy records, ratchet progress, novelty) tips the policy toward living. Generic: it is the
  termination-bias correction for a constant time penalty on every map.
* `--energy-ratchet 1` (4439aab8, new, off = bit-identical): `--energy-pot K` pays only NEW records of
  the episode's energy height and charges no loss. K 12 kept (pb_r26's). Test:
  tests/python/test_energy_ratchet.py.
* Plus A1's `--stall-energy 32 --eval-patience-energy 32`. Same box, 6e8 budget.

### A2 `zlr1b2_er`: no suicide bias left, but no pumping either; stopped at 243M (17:51 UTC)
* Greedy 76 / 151 / 227M: fwd 182 / 268 / 98 u, peak 267 / 261 / 186 u/s, patience-cut 3 / 3 / 6
  of 9, the rest fall into the pit at 5-25 s; h max = the spawn's (-3,071) in every episode. Best
  progress 0.70% (score ~0.70, = the control). Training: energy RECORDS +8..10 u/ep against
  -750..-1,090 u/ep of losses, stall 48-72%, yaw sigma pinned at 2.718 from ~170M (no signal).
* Why nothing to learn from: (1) the true start is 100 u above the floor, so the first landing
  burns ~106 u of h and an energy record needs > ~412 u/s on the floor first; (2) ~60% of the
  training spawn pool is progress-keyed rows that sit at the pit edge on this map: front-perturb
  (922 rows from 1-4 "frontier" reservoir states at progress 44 u), the greedy pool (states 2-0.25 s
  before each greedy death = falling into the pit), plus 10% goal-anchored rows at the top. The
  field's frontier IS the pit edge here, so the recipe trains where nothing can be learned.
* Zero-shot refs (local CPU, greedy, 30 s, 7 ms): pb_r26 idles on the start floor (vmax 443);
  uf2_x13 walks off the ledge into the pit at 10-11 s. Neither gains energy: a warm start buys no
  pumping here.
* The first gate, from the record (ANALYSIS ONLY): bhop the start floor to ~300-440 u/s, leave
  the platform's west edge, land on a curved ramp that bottoms out at z -3,930 (the kill plane is
  at -4,000..-4,050) and swings north-east into the half-pipe at ~830 u/s.

### Go-Explore from the true spawn DOES enter the half-pipe and pump (17:52-18:04 UTC, box CPU)
* `tools/explore_phase1.py --exact-chains --cell 64 --cell-energy 32 --cell-heading 1
  --view-follow=-90,-60,-30,-15,0,15,30,60,90 --select-energy 3 --act-every 4 --decisions 100
  --maxvel 3500 --tick-ms 7 --envs 4096 --threads 28 --max-minutes 12` (the pb/uf2 config at this
  map's tick; runs/ge/zl1 on the box). At 3 min: 868k cells, best progress **8.21%** at
  (1,047, 488, -2,879) = the half-pipe's EAST END (the record's east end reaches 6.2% at t 27.5),
  h max **-2,549** (+525 u over the spawn; the record has -2,844 at t 10 and -2,294 at t 30);
  123k states with h in -2,750..-2,500. Uniform HELD actions in the velocity frame pump on this
  map (pb's half-pipe stalled at +150 u). Report: autoresearch/analysis/zl_ge_report.py.
* Final (12 min, 28 threads): 1,595,692 cells, 10.4M runs, best progress **9.48%** at
  (1,226, 497, -2,801) (the half-pipe's east end), h max -2,341 (+733 u over the spawn), 0 goal
  hits. h bands: 97k cells at -2,500..-2,250. Best d flat at 5,759 from ~8 min: the next step up
  needs more energy than random bursts add quickly.
* `tools/ge_replay.py` -> runs/ge/zl1_line.npy on the box: 46 cells, 45/45 links exact, 3,961
  ticks (27.7 s): off the platform's west edge at ~3 s, pumps in the curved start ramp (z -3,200
  .. -3,815) to h -2,769 by 17 s, enters the half-pipe at 18 s (x -1,529), pumps to h -2,584 by
  27 s, ends at the east end (9.48%). `tools/spine_canon_view.py` -> zl1_line_canon.npy (views
  were a median 60 deg off the velocity).

### A3 `zlr1b2_ge` (18:05 UTC): A2 + the GE archive as 30% of training spawns
* Spawn file runs/ge/zl1_states.npy: the most energetic GE state per 256 u cell, kept if deeper
  than the start on the field OR above the spawn's energy height (1,360 states, h p50 -2,437);
  the trainer's own --ge-minutes keeps d < d0 only, which here would drop the half-pipe the run
  has to pump in. Provenance: Go-Explore from the true spawn, reward-free, no demo.
* At ~50M the file was swapped (atomically; the trainer re-reads it) for its view-canonical copy
  (spine_canon_view.py; views were 60 deg off the velocity, a spawn-semantics defect).
* GE-2 beside it (18:07, 12 threads, 25 min): explore_phase1 rooted at the line's last 1,461
  states (the half-pipe), stride 10 -> runs/ge/zl2: more energy, and the exit to the second
  half-pipe?
* A3 result (stopped at 232M, 18:13 UTC): greedy 76 / 151 / 227M = patience-cut 5 / 8 / 9 of 9,
  idling on the start platform at 260-530 u/s (score 0.70 at best, = the control). The suicide is
  gone (A2's fail-pen) but the archive spawns teach nothing that reaches the true start: nothing
  on the platform says "west, off the edge, fast". Yaw sigma back at its 2.718 cap by 227M.
  Training energy records +15..37 u/ep (A2: +10).

### A4 `zlr1b2_sp` (18:17 UTC): A3 + the GE line PINNED at the true start (uf2 x7's form)
* `--demo-file runs/ge/zl1_line_canon.npy --demo-tau0 0 --demo-rate 0.0 --demo-grow 5
  --demo-front-frac 0.5 --demo-window 5 --demo-grow-max 40 --demo-cooldown 5 --demo-min-ep 50
  --demo-success-d 100 --demo-success-bonus 50 --keys-hold-spawn-duck`: half the line draws
  start at rows [0, 4] (= the true start, where the eval starts), half uniform along the line;
  an episode whose ratchet record crosses d(line end) + 100 = 5,859 u (7.9%) is paid +50 once.
  `--keys-hold-spawn-duck`: 61% of the line's rows are crouched (GE's random duck bit).
* Recipe incompatibilities hit on the way (all three refused at startup, in this order):
  `--respawn-novel is not wired with --demo-file` -> `--no-respawn-novel`;
  `--respawn-backward-mix needs --respawn-novel` -> `--respawn-backward-mix 0`;
  `--int-real-only 1 needs --respawn-novel` -> `--int-real-only 0`. So this arm also loses the
  recipe's novel rows and goal-anchored rows (they were 30% of the non-start pool).
* Box extended to 20:15 UTC. GE-2 (rooted in the half-pipe) runs beside it until ~18:32.
* A4 result (stopped at 283M, 18:26 UTC): greedy 76 / 151 / 227M patience-cut 9 / 8 / 6, idling on
  the platform (peak 156-203 u/s); best 0.70. Spine success by decile at ~150M: 0% from rows
  0-90%, 19% from the last 10% (in the half-pipe, 2-3 s from the end). Yaw sigma 2.718 by 282M.

### The yaw rot is what stops every run from pumping (18:20 UTC, diagnostic on the box)
* record_ckpt of A4's 184M weights from 4 line states in the half-pipe (8 stochastic episodes,
  15 s): h max = h at spawn in EVERY episode (not one energy record), 0 swings; the survivors
  drift at 600-860 u/s in a narrow z band, the rest fall into the kill plane.
* Why: the yaw is an OFFSET from the velocity heading, off_warp(tanh z) (u 0.5 -> 10 deg, u 1 ->
  180 deg). At z sigma 0.3 (the init) the view stays within ~+-3 deg of the velocity: the strafe
  optimum, where any held side key adds energy. Every run here walked sigma to the 2.718 clamp
  (A2 by ~170M, A3 by 227M, A4 by 282M): offsets of ~+-170 deg, and in pm.c's air accelerate a
  wishdir with a backward component BRAKES (addspeed = 30 - v.wishdir, accelspeed up to
  airaccelerate x wishspeed x frametime = 175 u/s per tick). A policy trained on that noise
  cannot discover pumping, and its mean drifts. pb saw the same (0.08 -> 2.718); its
  `--yaw-entropy 0` stopped only the bonus part ("from scratch the policy gradient itself widens
  the view": pbr1b1_scr 0.31 -> 2.718 by 270M anyway).
* `--yaw-sigma-max S` (04524bae, new, off = bit-identical, mirrored in record_ckpt / beam_tas):
  the pitch head's ceiling applied to the yaw head(s) under --view-absolute; the mean is free.
  S 0.3 = the init value. Generic: the velocity-frame yaw's useful range is near zero on every
  map (large offsets only brake). Test: tests/python/test_yaw_sigma_max.py.

### A5 `zlr1b2_ys` (18:28 UTC): A4's flags + `--yaw-sigma-max 0.3` (one change)
* A5 result (stopped at 214M, 18:37 UTC): yaw sigma held at 0.300 (cap) the whole run, pitch
  0.30 -> 0.15; greedy 76 / 151M patience-cut 6 / 8, idling on the platform at up to 328 u/s;
  training energy RECORDS rose +8 -> +24 u/ep over 170M (A4: +7-8), stall 55-64%. The cap stops
  the rot but at 200M the policy still has no reason to leave the platform: the +50 at the line's
  end is ~28 s away and nothing in between pays for following the line.

### GE-2 (rooted on the GE-1 line's half-pipe, 25 min, 12 threads beside A3-A5)
* 2,511,985 cells, best **12.77%** (d 5,549) at (268, 1,336, -2,592): standing on a rim at
  z -2,592 north of the half-pipe (the record's best inside loop 1 is 12.64% at t 65); h max
  -2,130 (+944 u over the spawn; GE rides the deep U's near the kill plane at 1,700 u/s). 0 goal
  hits, no exit to the second half-pipe yet.
* ge_replay: 27 cells, 26/26 links exact, 2,364 ticks from line row 3,420; joined:
  runs/ge/zl12_line.npy = GE-1 line [0, 3,420) + that chain = **5,784 ticks (40.5 s) from the
  true spawn to 12.77%**; arc route runs/ge/zl12_route.npz (28,032 u, 220 vertices @ 128 u).

### A6 `zlr1b2_arc` (18:39 UTC): ARC-LENGTH shaping along the GE line (the xSELF construction)
* The field cannot see the loops; arc length along a line found by reward-free search from the
  true start is monotone through them (CLAUDE.md: "the reference line supplies the ORDERING" -
  xSELF finished 47/102 on cannonball with a line from the policy's own runs). Generic recipe:
  search -> line -> arc coordinate; no demo, no map constant.
* Flags: the recipe WITHOUT `--race-ratchet` (RaceReward refuses ratchet + arc as "a second
  treatment of the same defect"; launched with --no-recipe and the recipe minus that flag),
  `--race-arc runs/ge/zl12_route.npz --stall-arc 1` (corridor 1,500, window +-16 = the defaults;
  scale 100 / 28,032 u), `--spawn-states runs/ge/zl12_spawn.npy --spawn-states-frac 0.4` (the
  line's 5,784 states, views canonical, + GE-1's 1,360), `--stall-energy 32
  --eval-patience-energy 32 --energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0
  --energy-ratchet 1 --fail-pen 2 --yaw-sigma-max 0.3 --keys-hold-spawn-duck`.
