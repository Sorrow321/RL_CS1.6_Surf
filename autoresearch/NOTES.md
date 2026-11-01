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
* **A6 at 151M (18:44 UTC): the first arm off the start platform.** Greedy 76M: all 9 leave the
  platform, catch the curved start ramp (z -3,912..-3,964) and swing in it (5-11 swings, vmax
  ~1,150 u/s, h max -2,986..-3,014 = +60-90 u), dying at 30-36 s; 151M: 17-43 swings, h max
  -2,952..-2,977 (+100-120 u), episodes 32-64 s (the energy re-arm keeps them past the 30 s
  patience), one episode 1.34% (score **1.341**; every earlier arm <= 0.704). Training: reward
  13.9/ep, arc gain 2,113 u/ep, energy records +48 u/ep, yaw sigma 0.17 (cap 0.3), ep len 2,452,
  ~356k steps/s with GE-3 beside it. The policy follows the line's order (the GE line itself pumps
  in this ramp from 3 to 17 s, h -3,074 -> -2,769, before entering the half-pipe at 18 s).
* Promising by the protocol's rule (a gate no other arm cleared): extend.
* A6 to its 600M budget (19:03 UTC): greedy 227 / 302 / 378 / 453 / 529M all ride and pump the
  curved start ramp (path 34-51k u, peak 1,167-1,258 u/s, episodes ~35-50 s, patience-cut 7-9 of
  9) without the line's transition into the half-pipe (the line makes it at 17 s with h -2,769;
  the greedy's h max ~-2,950); field progress 0.0% (the ramp lies behind the start on the
  field). Training reward 13.9 -> 25.8/ep, energy records +48 -> +110 u/ep, arc gain 2,113 ->
  2,524 u/ep, yaw sigma 0.14. Best score stays 1.341 (151M). ckpt_final pulled (md5 308e7709).
* A6b `zlr1b2_arc2` (19:04 UTC): resume of ckpt_final to 3.6e9 (everything restored from the
  checkpoint: route, spawn states, energy rules, yaw cap; verified in the log).
* GE-3 (rooted on the joined line's last 2,364 rows, 25 min, 12 threads): 2,660,118 cells, best
  **13.37%** at (384, 487, -2,550), h max -2,034. East of the half-pipe it covers x 1,800-2,600
  (213k states) but only low (z <= -2,773), its high-energy states falling near the kill plane.
  The record (ANALYSIS ONLY) leaves loop 1 eastward at h ~-1,950 and reaches (2,317, -680,
  -2,252) at t 70: random bursts pump to ~-2,050 but not along that exit. No new route from it.
* Beam b1 (19:07 UTC, on the box beside A6b): beam_tas from the TRUE START on A6's 600M weights,
  --score dh --score-energy 1 (rank by -d + energy height), 4,096 envs, 25-decision generations,
  14,300 ticks (100 s), --save-frontier: does the policy's own pumping + selection make the
  ramp -> half-pipe transition and go past the GE line?
* **A6b 676M (19:08 UTC): score 9.905.** Greedy 6/9 episodes pump to h ~-2,000 (+1,074 u over the
  spawn; the record leaves loop 1 at ~-1,950) and reach 9.54-9.91% on the half-pipe's ends
  (z max -2,465..-2,506), flying 76-120 s with 68-104 swings; 3/9 still stay in the start ramp
  (h ~-2,850). median 9.63%. (600.8M, the resume's first eval: frontier 565 u, cover 2.8%.)
  Snapshot of the weights 12M steps later: runs/zlr1b2_arc2/ckpt_688390144.pt (pulled).
* Beam b1 (beam_tas on A6's 600M weights, --score dh): beam_tas and surfgym.bc.make_eval_feeds
  refused every --race-arc checkpoint; the arc reaches inference only through --obs-reward's
  slot 12, so both now refuse arc + obs-reward only (840ea283, test_arc_beam_feeds.py). b1's
  population at t 6,000: min d 5,975-6,050, median ~6,120 (3.8%).

### Harness issue (box.py pull --snapshot)
* `box.py pull --snapshot STEP` fetches `ckpt_<STEP>.pt` with STEP as a plain int, but the
  trainer names snapshots `ckpt_{step:010d}.pt` (zero-padded): below 1e9 steps they never match
  (a resumed run at 600M with --ckpt-every 1e9 writes its first snapshot at 1.6B). Workaround: a
  hardlink with the unpadded name on the box (`ln ckpt_0688390144.pt ckpt_688390144.pt`), then
  `pull --snapshot 688390144`.
* Beam b1 (A6's 600M weights, 100 s, 131 s of search): frontier d 5,482 = **13.83%** at 91 s - the
  policy's own line, past every GE line: ~50 s pumping the start ramp (h -2,926), into the
  half-pipe at ~55-60 s, ends high on the north-west rim (-1,904, 1,322, -2,522), h max -2,075.
  Beam b2 (A6b's 688M weights, 150 s, 231 s of search): into the half-pipe ~20 s sooner, h max
  -2,161, but its frontier is 9.0% (d 5,790 at 118 s). Neither leaves loop 1.
* A6b 752M: frontier 625 u (9.8%), cover 8.6%, all 9 episodes ended by the patience rule. 819M greedy recorded on the box (6 eps x 120 s, full
  states): 4/6 pump to h **-1,912..-1,927** (above the record's ~-1,950 at its loop-1 exit) and
  fly the whole 120 s; 2/6 stay in the start ramp (h ~-2,800). The policy has the energy; the
  arc route ends at 12.77% and nothing pays leaving loop 1.
* GE-4 (19:17 UTC, 12 threads, 20 min): rooted at 48 of those episodes' own highest-energy states
  (8 per episode, >= 400 ticks apart; runs/ge/roots4.npy) - return-then-explore from the
  policy's own frontier. A chain that leaves loop 1 + the recorded greedy prefix = a line from
  the true start for the next route.
* A6b 827M: all 9 greedy episodes stayed in the start ramp (0%), 902M: back in the half-pipe
  (frontier 553 u = 8.7%, cover 8.6%): the ramp -> half-pipe transition is still a coin flip for
  the deterministic greedy at this stage (CLAUDE.md's gate ladder).

### `--race-arc-tail U` (a705f5f0, 19:22 UTC): the field takes over past the route's end
* The arc route ends at 12.77% and nothing past it pays (the field's ratchet is refused with
  --race-arc), while the 819M greedy already pumps to the record's loop-exit energy. With
  `--race-arc-tail U`, an episode whose arc anchor reaches the route's last U units switches the
  field's ratchet on for the rest of the episode (records of d paid at 100/d0, a record re-arms
  the stall). The route covers what the field cannot see; the field leads after it, so a route
  found by search need not reach the finish. U = 1,500 = the arc corridor (the arc mode's own
  constant). Off = bit-identical, TRAIN_ONLY, refused with --obs-reward.
  Test: tests/python/test_race_arc_tail.py.
* Deploying during a run: deploy_box.sh's GPU health check fails while the trainer / GE / beam
  load the card ("below spec"); the checkout and build before it had run (git log a705f5f0 on the
  box), the restamp after it had not (ran by hand: caches matching). Not a bad host.
* GE-4 (rooted at the 819M greedy's own highest-energy states, 20 min): 3,052,701 cells, best
  **14.32%** at (341, 483, -2,488), h max **-1,838** (above the record's ~-1,950 at its exit), east
  of the half-pipe z max -2,621 (x 1,800-2,600): energy is no longer the gap - the direction is;
  no exit from loop 1 in 20 min.
* **A6b evals (score): 0.60B 9.104, 0.68B 9.905, 0.75B 10.043, 0.83B 0.000, 0.90B 8.918, 0.98B
  11.041, 1.05B 0.000, 1.13B 12.420, 1.20B 14.093 (best 14.1%, median 13.8%), 1.28B 13.806,
  1.36B ~13.7% (median low).** The policy reaches the route's end (12.77%) and a little past it;
  the 0.000 evals are the deterministic greedy staying in the start ramp (coin-flip gate).
  Snapshot pulled: runs/zlr1b2_arc2/ckpt_1379926016.pt (md5 b8beeff0).
* A6c `zlr1b2_arc3` (19:43 UTC): resume of that snapshot + `--race-arc-tail 1500`, to 4.4e9 (route,
  spawns, energy rules, yaw cap restored; verified). Box extended.

### The loop-1 exit, read off the record (ANALYSIS ONLY - it must not pick a route or a reward)
* t 66.7-67.1: along the half-pipe bottom eastward at ~1,480 u/s (h ~-1,970); t 67.5-68.3: up the
  east wall, a carve to the SOUTH over its top (x ~1,180, z -2,745 -> -2,633, vy -860..-940; field
  11.9%); t 68.7-69.3: lands on a ledge at z ~-2,490 south of the half-pipe (y ~-400) and runs east
  at ~1,000 u/s (field 10.7-11.6%); t 69.5-70.3: up a ramp at x 2,000-2,500 into loop 2 (field
  12.1 -> 19.4%). h at the carve ~-1,950.
* The GE-2 route ends on the NORTH rim (12.77%), so with `--race-arc-tail` the field's record
  starts at 12.77% and the exit's 10.7-12% stretch is unpaid. A route rule that picks by the
  record would be demo supervision (section 0); the generic alternative is the tail ON FROM THE
  START (`--race-arc-tail` >= the route length: the field's ratchet beside the arc everywhere),
  which pays the east-wall climb whenever it beats the episode's own record. Next arm if A6c
  stalls at ~14%.

### A6c `zlr1b2_arc3` (the arc tail): OUT OF LOOP 1 (20:10 UTC)
* Scores: 1.38B 11.664, 1.46B 18.133, **1.53B 19.747 (9/9 at 19.0-19.75%, h max -1,109..-1,160,
  z max -2,146..-2,192, 91-159 s)**, 1.61B 19.539, 1.68B 0.000, 1.76B 0.000, 1.83B 17.762,
  **1.91B 21.388 (2/9 at 21.1-21.4%, h ~-1,090, z max -2,036; 7/9 stay in the start ramp)**,
  1.98B 18.391. The field's ratchet past the route's end took the policy over the loop-1 exit
  and into loop 2 within ~80M steps of the switch (the record is at 17.5% entering loop 2 at
  t 70 s and 22% at t 95 s). Training at 2.02B: reward 52/ep, arc gain 5,157 u/ep, tail field
  progress 194 u/ep, energy records +177 u/ep, reservoir deepest 21.5%, yaw sigma 0.096.
* The remaining weakness: the start-ramp -> half-pipe transition is a coin flip for the greedy
  (whole evals of 0%). Snapshot pulled: runs/zlr1b2_arc3/ckpt_2028470272.pt.
* A6c 2.06B frontier 1,329 u (20.9%), 2.14B 1,295 u (20.4%): holding at ~20-21% (loop 2).

### `--ge-arc U` (6d0f04f5, 20:15 UTC): the procedure as ONE generic flag
* `--ge-minutes M --ge-arc U --no-race-ratchet`: Go-Explore phase 1 from the true spawn for M
  minutes on all CPUs (energy cells 64 u x 32 u of h, heading, view offsets in the velocity
  frame, select-energy 3, exact chains, at the run's whole-ms tick) -> the deepest cell's exact
  chain replayed into a line (ge_replay) -> the line at 128 u = runs/<run>/ge_route.npz as
  --race-arc with --race-arc-tail U and --stall-arc 1 -> spawn rows (--ge-frac, default 0.4) =
  the line + the archive's most energetic state per 256 u cell that is deeper than the start on
  the field OR above its energy height + 32, views along the velocity. The config records the
  route and the spawn file as race_arc / spawn_states (a resume reads them, never re-searches).
  `--no-race-ratchet` (new) turns a preset's ratchet off (--race-arc refuses it). Off =
  bit-identical. Test: tests/python/test_ge_arc.py (refusals; a 3-s search on this map -> route
  + rows).
* The recipe this lineage would ship (constants set once, no map knowledge): the incumbent recipe
  without --race-ratchet + `--ge-minutes 12 --ge-arc 1500 --stall-energy 32
  --eval-patience-energy 32 --energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0
  --energy-ratchet 1 --fail-pen 2 --yaw-sigma-max 0.3 --keys-hold-spawn-duck`. Not yet run from
  scratch as one launch: the A6 lineage reached it in steps (arc + spawns at 0, the tail at
  1.38B, a two-round GE line to 12.77% instead of one 12-min search to 9.48%).
* A6c 2.21-2.74B scores: 18.251, 21.671, 20.475, **22.404 (2.44B, median 22.1%)**, 22.029, 20.975,
  0.000 (2.66B, start ramp), 20.553: medians now 20-22% (most greedy episodes in loop 2), the
  frontier creeping 21.4 -> 22.4%. Snapshot pulled: runs/zlr1b2_arc3/ckpt_2788163584.pt.
* record_ckpt of that snapshot (6 x 150 s, no eval patience): only 2/6 leave the start ramp
  (20.5% with h -1,107; 16.2%), 4/6 pump in the ramp for 97-150 s (h ~-2,800): the start-ramp ->
  half-pipe transition is the weak gate, not loop 2.
* GE-5 (20:47 UTC, 12 threads, 20 min): rooted at the 16 recorded states with h > -2,200 (all in
  loop 2; runs/ge/roots5b.npy) - the loop-2 exit, to extend the route like GE-4 should have.

### Why the greedy's start is a coin flip: the route punishes the quick exit (20:49 UTC)
* In the recorded 2.79B greedy, ep_0001 leaves the start ramp for the half-pipe at **7.2 s** with
  h -2,861 (the record enters the half-pipe at ~7 s too); ep_0004 only at 72.7 s (h -2,755);
  ep_0000 never (h max -2,808). The GE-1 line the route is built on pumps in the ramp until 17 s.
  A quick exit jumps ahead of the arc anchor by more than its +-16-vertex window (2,048 u), so
  the anchor freezes and the rest of the route pays nothing: the arc reward PUNISHES the
  policy's best opening and rewards the GE line's 17 s of ramp pumping.
* The generic fix is CLAUDE.md's xSELF construction: rebuild the route from the policy's OWN best
  greedy episode (the deepest one, cut at its deepest point), extended by a search from its end.
  ep_0001: 20.47% at 134.4 s, 163,875 u of arc to there, h up to -1,237.
* A6c 2.81-3.27B (stopped 21:08 UTC): 18.649, 0.000, 20.161, 20.806, 0.000, 21.167, 0.000: the
  frontier stays at ~20-21% and the whole-eval start-ramp failures come back more often (the GE
  route keeps rewarding the ramp pumping). Best of the run: 22.404 at 2.44B.
* GE-5 (rooted at 16 loop-2 states, 20 min): 3,942,750 cells, nothing deeper than its roots
  (archive best 14.79% outside them), h max -1,061: random bursts do not find loop 2's exit.

### A7 `zlr1b2_self` (21:08 UTC): the xSELF route
* Resume of the 2.79B snapshot with `--race-arc runs/ge/self5_route.npz` (the policy's own greedy
  ep_0001 from the true start, cut at its deepest point: 19,203 ticks, 20.47%, 163,840 u of arc
  @ 128 u; scale 0.00061/u), `--race-arc-tail 1500`, `--spawn-states runs/ge/self5_spawn.npy`
  (that episode's states, views canonical, + GE-1's 1,360 forward cells) at 0.4. Box extended
  to 23:39 UTC. Read against A6c at matched steps: the share of whole evals stuck in the start
  ramp, the median, the frontier past 22.4%.
* **A7 (the self-route) works: 2.86B 21.966 (median 21.9%), 2.94B 23.367 (median 23.1%), 3.02B
  22.837, 3.09B 23.273 (median 23.1%), 3.17B 22.376 (median 22.1%).** The whole-eval start-ramp
  failures are gone (1 of 5 evals with a low median, vs half of A6c's); the frontier 22-23.4%.
  Snapshots pulled: runs/zlr1b2_self/ckpt_3115319296.pt (and 3208642560 on the box).

### `--race-arc-self M` (a2ed6852): the xSELF route, iterated automatically
* After every greedy eval, if the deepest episode beats the current route's end by more than
  M x d0 on the shaping field, the route becomes that episode's own line from the true start to
  its deepest tick (same spacing / corridor / window); every training env is re-anchored, the
  arc records re-armed, the tail re-armed lazily; runs/<run>/arc_self.npz and the checkpoint
  config point at it. M 0.01 (64 u here). Off = bit-identical, TRAIN_ONLY. Test:
  tests/python/test_race_arc_self.py.
* **BUG caught at launch (fixed, bb0d2cda):** `--no-race-ratchet` (6d0f04f5) was registered as a
  store_false BEFORE `--race-ratchet`, so argparse took ITS default (True) for the shared dest:
  every run passing neither flag would have trained with the ratchet ON. It surfaced as the A8
  resume's refusal ("--race-ratchet with --race-arc"). Registered after the original with
  default=None now; two regression tests in test_ge_arc.py fail on the old parser. No training
  run used it (the box ran a705f5f0 until 21:27).
* A8 `zlr1b2_aself` (21:31 UTC): resume of A7's 3.21B weights + `--race-arc-self 0.01`.
* **A8 (automatic self-route): the route swapped at 3.21B (end d 4,879 = 23.3%; 126,336 u) and at
  3.28B (end d 4,758 = 25.2%; 168,064 u). Evals 3.21B 23.300 (median 23.1%), 3.28B 25.203
  (median 25.1%), 3.36B 22.914 (median 22.7%).** Training at 3.38B: reservoir deepest 25.6%,
  arc gain 58k u/ep, tail 247 u/ep, energy records +247 u/ep. Snapshot pulled:
  runs/zlr1b2_aself/ckpt_3369598976.pt.
* Plan: A8 runs while it keeps setting records; ~23:30 UTC the box switches to a FROM-SCRATCH
  launch of the whole recipe (`--ge-minutes 12 --ge-arc 1500 --race-arc-self 0.01` + the
  energy rules, fail-pen, yaw cap) to confirm it in one launch.
* A8 3.44-3.89B: 23.379, 22.942, 24.415, 25.842 (median 25.5%), **26.870 (3.74B, median 26.8%)**,
  22.992, 23.338. Third route swap at 3.74B: end d 4,652 (26.9%), 2,014 vertices = 257,664 u
  (scale 0.00039/u). Snapshot pulled: runs/zlr1b2_aself/ckpt_3923247104.pt.
* A cost of the swap rule as written: the DEEPEST episode is also the one that swung longest to
  get there, so the route grows (126k -> 168k -> 258k u) and the arc pays less per unit (100/L);
  the medians dipped to 22.5-23.1% right after the third swap. A shorter-line tie-break (the
  fastest episode within the margin of the deepest) is the obvious refinement - not built.
* A8 after the third swap (3.81-4.42B): 22.992, 23.338, 23.699, 23.520, 23.260, 3.846, 23.811,
  23.438 (median 2.2%), 21.150 - a plateau at 23-24% with collapses; no record after 26.870.
  Stopped 22:28 UTC. The shortest-line tie-break (18ecfd30) is the response for the next runs.

### B1 `zlr1b2_one` (22:42 UTC): the WHOLE recipe FROM SCRATCH, one launch
* `box.py launch` (the incumbent recipe prepended) + `--no-race-ratchet --ge-minutes 9 --ge-arc
  1500 --race-arc-self 0.01 --stall-energy 32 --eval-patience-energy 32 --energy-pot 12
  --energy-pot-front 0 --energy-pot-front-bonus 0 --energy-ratchet 1 --fail-pen 2
  --yaw-sigma-max 0.3 --keys-hold-spawn-duck`. Nothing hand-made: the trainer's 9-minute search
  (30 threads) reproduced GE-1 exactly (same seed: the 9.48% cell, the 3,961-tick line); route
  21,888 u (172 vertices); spawn rows 5,294 (line + 1,333 forward archive cells).
* Two launch defects found and fixed on the way (f391270a, 50423480): `--ge-arc` named its spawn
  file in args before the trainer's SELF_STATES provenance check (refused after the search; the
  run's own reward-free search is exempt like --ge-minutes' pool), and set --stall-arc only when
  None while a default of 0 is set earlier (it ran with the field stall clock for a minute).
  The relaunch reused the run's search files (`_ge_phase1` reuses runs/<run>/ge_states.npy).
* Read against the A6 lineage at matched steps (A6: off the platform 151M; A6b 9.9% at 676M,
  14.1% at 1.20B; tail from 1.38B: 19.7% at 1.53B; self-route from 2.79B: 23.4% at 2.94B,
  25.2% at 3.28B, 26.9% at 3.74B).
* B1 0.5-680M: every greedy eval 0.0% (all 9 episodes pump in the start ramp until the patience
  cut; 604M: 44-60 swings, h max -2,692..-2,733 - more than A6 at 600M, ~-2,950; 0% because the
  ramp lies behind the start on the field). Training reward 38/ep at 680M (A6: 26 at 600M), arc
  1,713 u/ep, tail 52 u/ep. A6b first made the ramp -> half-pipe transition at 676M.
* B1 0.68-1.35B: 0.0% except 1.06B (9.449: one episode to the route's end, the half-pipe's east
  end); greedy episodes mostly pump in the start ramp. Training reward 48/ep, arc 2,051 u/ep,
  tail 13 u/ep, reservoir deepest 7.2%. Slower than the A6 lineage at matched steps (A6b:
  9.9% at 676M, 14.1% at 1.2B) - one seed each through a binary gate (the ramp -> half-pipe
  transition), and the route here ends at 9.48% (one 9-minute search) where A6's joined line
  ended at 12.77% (two searches). Box extended to 04:39 UTC (the round's wrap-up is 04:29).
* **B1 takes off (00:06 UTC): 1.43B 6.954, 1.51B 8.328 (median 7.9%), 1.59B 8.491 (median 8.4%),
  1.66B 13.401 (median 9.0%), 1.74B 0.009, 1.81B 12.546 (median 11.2%), 1.89B 12.770 (median
  12.4%), 1.96B 13.752 (median 12.9%).** Its own self-route swap at 1.66B (route end d 5,509 =
  13.4%, 55,808 u - the tie-break kept it short). Training at 2.0B: reward 68/ep, reservoir
  deepest 20.6% (stochastic episodes already leave loop 1), tail 283 u/ep, arc 25k u/ep. The
  one-launch recipe reproduces the A6 lineage's path with no hand-made step, a few hundred
  million steps behind it (A6c: 19.7% at 1.53B, 21.4% at 1.91B).
* B1 2.04-2.57B: 14.034, 15.273 (2.11B, median 14.3%), 12.936, 14.347, 14.653, 14.341, 14.364,
  13.295: medians 13-14% every eval (the greedy reliably on the first half-pipe's ends, not out of
  loop 1 yet); second self-route swap at 2.11B (end 14.9%, 58,496 u); reservoir deepest 20.6%.
  Snapshots on the box: ckpt_1000341504.pt, ckpt_2000683008.pt.
* Prepared for the last hours: the tie-break applied to A8's best eval (3.74B, 26.87%): the
  shortest line within 32 u of the deepest reaches 26.64% in 125.2 s (the deepest: 170.2 s) ->
  a 180,800 u route (vs 256,616 u) = runs/ge/a9_route.npz on the box. Plan: ~01:30 UTC, A9 =
  A8's 3.92B weights resumed on that route (+ the self-route rule, the tail) to push the score.
* **B1 result (stopped at 3.36B, 01:05 UTC): the one-launch recipe from scratch reaches 15.3%
  (2.11B: 15.273, median 14.3%) and holds 13-15% with medians 13-14% in every eval from 2.0B to
  3.32B (15.171 at 3.32B), but its greedy does not leave loop 1 in 1.3B steps on its plateau.**
  Off the start platform, through the start ramp into the half-pipe and up to the ends of loop 1
  with no hand-made step (route, spawns, swaps all automatic); the exit the A6 lineage found
  after its tail switched on (19.7% at 1.53B) did not come here. One seed against one seed: the
  route B1's swaps chose ends on the half-pipe at 14.9% (the policy's own deepest line), A6c's
  ran on the hand-joined GE line to the north rim. Snapshots on the box: ckpt_1000341504,
  ckpt_2000683008, ckpt_3001024512.
* A9 `zlr1b2_a9` (01:06 UTC): A8's 3.92B weights resumed on runs/ge/a9_route.npz (181,632 u),
  self-route + tie-break + tail - the last ~3 h of the round.
* **A9 (A8's weights on the shortest route): 3.92-5.66B scores 22.7-26.9 with medians mostly
  23-26%: 4.83B 25.979, 5.06B 26.449 (median 26.1%), 5.28B 26.858 (median 26.6%), 5.51B 26.531,
  5.66B 26.248 (median 26.0%)** - level with A8's best (26.870) and steadier (A8 after its third
  swap: 23-24% medians). No self-route swap yet (none beat 26.64% + 64 u by enough); training
  reservoir deepest ~28%. Snapshots pulled: runs/zlr1b2_a9/ckpt_4923588608.pt,
  ckpt_5760352256.pt.
* A9 5.81-6.34B: 25.258, 22.904, 23.702, 25.382, 25.720, 25.032, 25.747, 24.540 (medians
  22.8-25.6%): no record; the greedy holds ~25% while the TRAINING reservoir's deepest state is
  now **34.8%** (front band at 34.4%) - the stochastic policy explores well into the next loop.
* A9 6.49-7.02B: 24.602, 24.652, 23.660, 24.908, 23.878, 25.998, 24.558, 25.805 (medians
  23.3-25.7%): steady at ~25%, no record (best of A9: 26.858 at 5.28B).
* The handoff (orchestrator rule): the best eval is A8's 3.74B (26.870); the snapshot at or before
  it is runs/zlr1b2_aself/ckpt_3369598976.pt (3.37B), the one after it ckpt_3923247104.pt
  (3.92B, 0.186B after vs 0.368B before: just short of "more than twice as near"). A9's
  ckpt_5760352256.pt (medians 24-26%) is the stronger policy of this lineage if a later round
  can reach it; its route is runs/ge/a9_route.npz on the box (181,632 u, also in this branch's
  history only as a description: rebuild it from traj_3737649152 with arc_self_candidate tie 32).
