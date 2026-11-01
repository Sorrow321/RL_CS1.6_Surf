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

### zlr1b1 (round 1, slot 1) - 2026-10-31 17:34 UTC: energy as progress for the stall clocks
* **What the recipe cannot see, measured (record demo, ANALYSIS ONLY; d on config.field, its
  own first row as d0 6,441 u, the spawn's 6,362):** the first loop is a ~3,000 u half-pipe
  running east-west (x -1,900..+1,250, its floor z ~-3,300 at y ~50). The record crosses it
  every ~4 s; each pass climbs higher on an end wall, so the FIELD does pay the loop at the
  turnarounds: best progress 1.1% (t 16 s) -> 5.5% (20) -> 7.5% (28) -> 8.7% (32), swinging to
  -4% at the floor in between. At t 34-38 the record loses ~800 u of energy height
  (h = z + |v|^2/1600: -2,166 -> -2,962, a 380 u drop gaining almost no speed), and neither its
  d record nor its h record moves by 32 u again until t ~62. So the record's own line would be
  cut by the eval patience rule at 61.6 s and by the 15 s training stall at 46.6 s, with or
  without an energy channel. The second half-pipe (y ~-600, z -2,250..-2,550) starts at t ~70:
  progress 17-25% there through t 130.
* **What the recipe does pay on the loop:** the ratchet pays the higher turnarounds; nothing
  pays the energy built between them (`--energy-pot 3 --energy-pot-front 0.01
  --energy-pot-front-bonus 1` pays NOTHING per call, only once at the reservoir's frontier
  band); the novelty key's speed bins are 1,333 u/s wide (`--int-speed 3`), so a first-loop pump
  (345 -> 1,300 u/s) stays in bin 0. The training stall rule re-arms only on ONE decision that
  beats the running best d by 3.5 u (125 u/s toward the goal at the record), so a pass that
  tops the last one by under ~10 u does not re-arm it, and `stagnant_mask` (3 s without a
  re-arm) keeps every such state out of the reservoir.
* **New flags (2f3d07d7, all default off, bit-identical: tests/python/test_stall_energy.py runs
  the trainer at HEAD~ and after the change on the same smoke and compares config,
  progress.csv, eval bytes and weights; the smoke falls back to surf_za_loop where sidistic is
  not installed):**
  * `--stall-energy EPS`: the training stall timer also re-arms when h has risen EPS u above
    its value at the last energy re-arm (cumulative, any rate). Also un-stagnates pumping states
    for the reservoir. Step line `se N%` = share of re-arms made by energy alone. EPS 32 = the
    eval patience's own constant (generic, not read off this map).
  * `--eval-patience-energy 1`: the eval patience clock gets the same channel, same 32 u.
  * `--energy-pot-rel 1`: `--energy-pot K` in units of the shaping scale (100/d0 per u), so K 1
    pays one u of energy height like one u of field progress on every map (15.75 per 1000 u
    here; 0.5 on cannonball). A unit, not a per-map constant.
* **First arm (planned): `zlr1b1_en`** = the recipe + `--energy-pot 1 --energy-pot-rel 1
  --energy-pot-front 0 --energy-pot-front-bonus 0 --stall-energy 32 --eval-patience-energy 1`,
  scratch, 30 min; then the recipe alone as the scratch control on the same card type.
  Would I run exactly this on a map I never opened? Yes: energy height and the field distance
  are in the same unit on every map, and the 32 u is the eval rule's own constant.
* Box: `rent` REFUSED at 17:32 on the fleet's budget pace (two boxes up, burn 0.59 of 0.619 $/h).
* 18:29 UTC: still no box (budget pace full since 17:32: two boxes, 0.59 of 0.619 $/h). A single
  background waiter retries `rent` once a minute; on a box it deploys with
  `--ckpt runs/warm_uf2x13.pt` and launches TWO arms side by side (the trainer takes half the
  container's CPUs, so two fit a 32-thread box; steps, not wall time, are compared):
  * `zlr1b1_en`: scratch, the recipe + `--energy-pot 1 --energy-pot-rel 1 --energy-pot-front 0
    --energy-pot-front-bonus 0 --stall-energy 32 --eval-patience-energy 1`, 5e8 steps.
  * `zlr1b1_wen`: the same flags WARM from uf2_x13's weights (`tools/warm_from_ref.py`: weights
    and Adam moments of uf2_x13, the config of the operator's scratch check run on this map, so
    the resume trains exactly the scratch recipe; step 0; uf2's spine / BC / spawn files and its
    novelty counts, reservoir and greedy pool dropped; all shape keys verified equal). It asks a
    different question: does a policy that already surfs at 1,000+ u/s learn the pump. Zero-shot
    it falls into the kill plane at 7.8 s (recorded on CPU, peak 1,164 u/s).
* Map entities (analysis): no trigger_push, no other teleport than the start room's eight (*4-*11,
  all to the destination "map") and the bottom kill plane *21: the loops are plain ramp geometry,
  nothing in the core's trigger handling is involved. Still no box at 19:35 UTC (the pace has been
  full for 2 h; the waiter retries every minute).

### Harness issue (zlr1b1, 2026-10-31 20:05 UTC): the budget pace starves a branch for hours
* The pace ((credit - reserve) / hours left x 1.1 = 0.618 $/h) admits about two cheap boxes. Two
  boxes were up from before 17:32 UTC, burning 0.59 $/h together, and stayed up: this branch's
  `rent` has been refused every minute from 17:32 to (at least) 20:05 UTC (~85 tries by a single
  background waiter; headroom 0.028 $/h, no offer that cheap). With 4 branches and arms allowed up
  to 6 h, a branch that does not get a box in the first minutes of a round can lose most of it.
  Nothing on this branch works around it. Suggestions for the operator: a per-agent share of the
  pace (e.g. limit / branches), or a cap on how long one box may hold the pace while another
  branch waits.
* Zero-shot on this map (CPU, `record_ckpt` of `tools/warm_from_ref.py` checkpoints = the refs'
  weights under the recipe's config here, 42 s cap; all four are shape-compatible with the recipe):
  none pumps. uf2_x13 heads off the start floor and dives into the kill plane (greedy 7.8 s,
  stochastic 6.9 / 19.0 s, peak ~1,160 u/s, energy never above the spawn's); pb_r15 / pb_r26 /
  sid_ref mostly stay on the start floor until the cap (best 0.0-0.7%, peak ~450 u/s; one
  stochastic pb_r26 episode gained 29 u of energy height before it fell). So a warm start brings
  surfing skill, not the loop: it still has to learn the pump. uf2_x13 stays the warm arm (it at
  least goes where the half-pipe is).

### zlr1b1 round 1 lesson (2026-11-01 03:50 UTC): no box all round, so no evidence; what is ready
* **No run.** `box.py rent` was refused on the fleet's budget pace on every try from 17:32 to 03:46
  UTC (~385 tries by one background waiter, once a minute): two boxes held 0.59 of a 0.614-0.619
  $/h limit the whole round, and no single-GPU offer costs the 0.03 $/h of headroom. The waiter
  was stopped at the 03:45 cutoff (a later box could only have recorded an untrained eval before
  the wrap-up). See "Harness issue" above. This branch has no score; nothing here is measured on a
  box.
* **Measured on CPU / from the record (analysis only):**
  * the field DOES pay the first loop at its turnarounds (each higher pass up a half-pipe end
    wall is a d record: 1.1% at 16 s -> 8.7% at 32 s in the record), but the record's own line
    has a 28 s stretch (34 -> 62 s) with no 32 u record of d OR of energy height, after losing
    ~800 u of energy at 34-38 s: it would be cut by the 30 s eval patience and killed by the 15 s
    training stall with or without an energy channel. The second half-pipe (17-25%) starts ~70 s.
  * the recipe pays nothing for energy built between turnarounds (front-bonus mode), its speed
    novelty bins (1,333 u/s) cannot see a 345 -> 1,300 u/s pump, and `stagnant_mask` keeps
    states with no stall re-arm in 3 s out of the reservoir.
  * zero-shot, none of the four refs pumps here (uf2_x13 dives into the kill plane in 7-19 s; the
    pb refs and sid_ref stay on the start floor): a warm start brings surfing, not the loop.
* **Ready to run (pushed, tested, default off, bit-identical):** `--stall-energy EPS`,
  `--eval-patience-energy 1`, `--energy-pot-rel 1` (2f3d07d7, tests/python/test_stall_energy.py,
  14 tests incl. the flag-off bit-identity run against the trainer before the change);
  `tools/warm_from_ref.py` (refs' weights + this map's scratch config, step 0, no foreign
  run-state; all four refs shape-compatible); `tools/eval_energy_report.py`; `tools/zl_watch.sh`.
* **The first two arms, for whoever has a box** (both as `box.py launch` extras on top of the
  recipe; the warm one `--resume` after `box.py deploy --ckpt runs/warm_uf2x13.pt`, built with
  `tools/warm_from_ref.py --ref $AR_HOME/incumbent/refs/uf2_x13/ckpt.pt --template
  $AR_HOME/runs/zlop/zlop_check/run.json --set energy_pot=1 --set energy_pot_rel=1 --set
  stall_energy=32 --set eval_patience_energy=1 --drop energy_pot_front --drop
  energy_pot_front_bonus`):
  `--energy-pot 1 --energy-pot-rel 1 --energy-pot-front 0 --energy-pot-front-bonus 0
  --stall-energy 32 --eval-patience-energy 1`, scratch and warm, side by side (the trainer takes
  half the container's CPUs), then the plain recipe as the scratch control. Read the evals with
  `tools/eval_energy_report.py` (h max climbing pass after pass = pumping; best progress at the
  turnarounds = the wall height), and expect the score to sit near the record's 8.7% wall height
  until something leaves the first half-pipe (the record does at ~70 s, to 17%+).
* Wrap-up 04:30 UTC (ROUND OVER): no box was ever rented, `runs/zlr1b1/` is empty, score.py finds
  0 valid evals; RESULT.json final with best null.
