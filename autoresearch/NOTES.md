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
