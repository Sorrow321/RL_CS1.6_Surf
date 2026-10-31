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
