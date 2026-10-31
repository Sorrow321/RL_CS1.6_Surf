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

## zlr1b3 (round 1, slot 3) - 2026-10-31 17:10 UTC start

### What loop 1 is (ANALYSIS ONLY: the record as an instrument, CLAUDE.md section 0)
* The record's first 120 s, 1 s steps, on config.field (t from the start curtain): it drops
  into the start dip (z -3,926 at t+2, 74 u above the bottom teleport), turns north and reaches
  loop 1 at t+4 heading east. **Loop 1 is ONE long surf ramp along x** (a wedge from x ~-2,100
  to ~+900 whose face rises to the north: at x -800 the face runs from (y 0, z -3,400) to
  (y 380, z -3,050); open air to the south, the kill floor at z -4,096 below). The player
  rides it end to end (~4-5 s per pass) and turns round at each end by riding up the face
  (z -2,600..-3,000). Each turn is where speed is pumped: E = z + |v|^2/1600 rises -3,004
  (t+2) -> -1,956 (t+64), ~17 u/s. Loop 2 is the same thing ~800 u higher at y ~-450
  (x -1,000..2,700); the record jumps there from loop 1's east end at t+65 (E ~-1,920).
* **The field does see the ends.** Every swing up an end sets a small new record: 4.3% (t+8),
  4.5 (12), 6.1 (17), 7.1 (25), 8.2 (32), then the record LOSES ~700 u of E at t+33 (a hit at
  the east end) and sets nothing until 9.4% at t+61: 29 s without a field record, so the
  recipe's 15 s stall-kill would have killed the record itself at ~t+47, and --eval-patience
  30 would cut an eval episode there too. Loop 2's end swings reach 20-24%.
* **The reward balance on loop 1 under the recipe**: the ratchet pays the end records,
  ~9.4 over 60 s = 0.16/s at the record's pace, less for a weaker pumper; the time penalty is
  0.1/s (0.001 per 10 ms tick, per-second invariant); the recipe's energy term
  (--energy-pot-front-bonus) pays only the kinetic energy carried into the frontier band, once.
  Pumping slowly is roughly break-even against the time penalty, and a death costs nothing.
* Go-Explore smoke (local, 1 thread, 3 min; pbf config at --tick-ms 7 --maxvel 3500): 191k
  cells, 8,199 of them on loop 1's ramp (y 0..500, x up to -401) but E max -2,985 there (no
  pumping yet); max progress 2.2%. Rate 0.16M ticks/s/thread.

### --stall-energy EPS (51b5b635, generic)
The stall clock (training kill, the reservoir's stagnant mask) and --eval-patience also re-arm
once the episode's best energy height has grown EPS u since the last re-arm (cumulative). Would
I ship it for every map? Yes: an episode gaining energy is not stalled anywhere; bounded by
maxvel (h <= z + 7,656 u at 3,500 u/s). EPS 32 = the eval-patience's own field eps. Off is
byte-identical; tests/python/test_stall_energy.py.
