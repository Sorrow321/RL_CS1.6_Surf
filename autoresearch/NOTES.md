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
* Go-Explore with STRONGER energy selection (local, 1 thread, 5 min; `--select-energy 30`
  instead of 3, everything else the pbf config at --tick-ms 7): 329k cells, best 6.4% (5,955 u)
  vs 2.2% at K 3, E max -2,800 vs -2,948, loop-1 cells to x 1,832 (the east end). Its deepest
  cells are NOT pumping: they stand/hover on the TOP RIDGE of loop 1's wedge (x 170-950,
  y 76-440, z -2,879..-3,189, |v| 31-466). Geometry (point traces): the wedge's ridge is at z
  -2,960 (y ~410) for x -1,800..800; a 200 u slab spans z -2,872..-2,672 over x ~200-800,
  y ~300..1,400+ (88 u above the ridge); field progress on the ridge 1.6% (x <= -600) .. 5.0%
  (x 0), on the slab's top 10.8-11.6%. The record never stands on either; its transfer to loop
  2 passes the slab's east end at z ~-2,650 (11.9%, t+65). So the field has a DECEPTIVE local
  optimum on loop 1 (climb onto the ridge: ~5% in seconds) that pays more than the record's
  first ~30 s of pumping. K 30 is a generic choice (energy tops dominate selection on any
  map), not read off this map.
* GE K 30, 15 min on 2 local threads (755k cells): best **8.66%** at loop 1's EAST END
  ((1,160, 487, -2,850), |v| 207, E -2,824, 23.7 s from the spawn); the east-end cells at
  7-8.7% fly with |v| 200-550 and E -2,695..-2,830. E max on loop 1 -2,592 (3 min: -2,985,
  5 min: -2,800). The search climbs the end by spending its speed at the apex; the record
  carries ~600-750 u MORE energy through that end (E -1,956 at t+64) and needs it for the
  jump to loop 2. No loop-2 cell yet. Energy grows slowly with compute (sub-linear).
* `--energy-pot-rel KAPPA` (235bde20, generic): the energy term's K = KAPPA x 1000 x the
  slot's progress scale (100/d0 x --race-shaping), i.e. one unit of energy height shaped like
  KAPPA units of geodesic progress on any map. KAPPA 1 = the unit equivalence, the value I
  would ship everywhere (K 3 is 0.19 of a progress unit here, 0.92 on unitfarmer2). On loop 1
  the record's +1,050 u of E would pay 16.5 at KAPPA 1 against 9.4 from the end records, and
  it is paid during the turn that gains it, not ~4 s later at the next apex.
* Plan (no box yet at 18:52: the fleet's pace allows ~1.6 boxes and two were taken at 17:15):
  arm A `zlr1b3_enr` = recipe + `--stall-energy 32 --energy-pot-rel 1 --energy-pot-front 0
  --energy-pot-front-bonus 0 --time-pen 0 --death-charge 1 --death-charge-window 2` (6e8) with
  GE K 30 (12 threads, 25 min) beside it, then the plain-recipe control `zlr1b3_ctl` (6e8) on
  the same box. Why --time-pen 0: 0.1/s is about what pumping earns early, so it made a quick
  death the better return; with the ratchet and gamma a sooner record is still worth more.
* Process hygiene: a `pgrep -af explore_phase1` at 18:37 also listed another agent's launch
  command line (a local process). Not acted on; listings are filtered to my own paths now.
* `--energy-death-window W` (e17b8429, generic): a death takes back what the energy term paid
  over the episode's last W s if it was a net gain (the --death-charge-window rule on the energy
  term). Why: with KAPPA 1 a policy that cannot pump yet earns ~+0.8 by air-strafing (up to ~80
  u of h per second of perfect strafe at any speed: 450/|v| u/s per tick x |v| / g) on its way
  into a fatal fall; refunded, a death keeps nothing it strafed up in its last 2 s. Arm A now
  carries `--energy-death-window 2` (smoke: K 15.719 on this map, 286 calls per 2 s at 7 ms).

### Harness issue: one box for four agents (zlr1b3, 19:20 UTC)
The fleet pace (credit - reserve) / hours left x 1.1 = 0.62 $/h fits one 5080-class box
(0.35-0.39) plus a cheap card. From ~17:15 two boxes held the whole pace (burn 0.59) for over
two hours, so this branch has had no box at all in the first 2 h 10 min of its 12 h round
(every `rent` refused: "no offer passes ... within the budget pace"). A per-agent share (a
queue, or a cap on one agent's box-hours while another waits) would keep a branch from sitting
idle for the first third of its round.
* 19:45 UTC, still no box. Plan revised: NO Go-Explore beside arm A. Plain GE's loop-1 energy
  grows ~log(compute) (E max -2,985 / -2,800 / -2,592 at 3 / 5 / 30 thread-min); loop 2 needs
  ~-1,950, i.e. ~200x the 30 thread-min, so on the box it would only cost arm A ~20% of its steps.
  Control `zlr1b3_ctl` cut to 4e8 (enough for a matched-step reading of arm A's gates).
* What the pump IS (record, analysis only): at each end the heading reverses through ~120 deg in
  ~1 s with the view ALONG the velocity (yaw within a few deg of the heading) and ONE strafe key
  held toward the inside of the turn (right at one end, left at the other); E +40-55 u per
  end-turn. That is the plain air-strafe turn, an action the recipe's policy already expresses
  (view offset ~0 in --view-absolute velocity, one side key). It is a skill to discover, not a
  representational gap.
* 20:06 UTC, still no box (3 h). The offer pool is thin too: the cheapest acceptable offer is a
  4080S at 0.41 $/h (then a 3090 at 0.62, a 5080 at 0.62), so even with one of the two live
  boxes released, a 0.41 box fits the 0.62 pace only if the remaining one burns <= 0.21.
* GE K 30 for 40 min (local, `--threads 3` but effectively ONE core: 100% CPU, 0.14-0.17M
  ticks/s, the same as 1 thread): 1.28M cells, loop-1 E max **-2,329** (above the ~-2,400 I
  estimate the jump to loop 2 needs), best progress 9.35% (5,767 u, reached by minute ~18 and
  never beaten), the highest states z -2,664 at (x -800..-1,100, y 670-860) north of the wedge;
  **0 loop-2 cells**. The archive has the energy but not the transfer (up the east end and over
  to loop 2's face, ~800 u higher to the south). The global top-energy cells are useless as
  roots: all 64 are falls into the void (z -3,656..-3,980, 1,464-1,635 u/s) strafed on the way
  down. GE is not the route to loop 2 at local compute; arm A (RL) stays the first box run.
* The transfer loop 1 -> loop 2 (record, ANALYSIS ONLY, to read evals): at the bottom of loop 1
  (x 427, z -3,323, 1,479 u/s east, E -1,956, t+64.0) it climbs the east end and turns RIGHT
  (south) while climbing - heading 55 -> 6 -> -61 -> -98 deg, vz up to +931 - to (1,184, 75,
  -2,648) at t+65.0 (11.9%), flies south over the gap at z ~-2,490..-2,560 ((1,268, -283) at
  t+65.4, 983 u/s) and lands on loop 2's ramp at (1,612, -448, -2,499) heading east at t+65.8;
  loop 2's first end swing reaches 19.4% at t+67. Every loop-1 pass before it turned LEFT
  (north, back west) at that end. A greedy eval that turns right at the east end with
  |v| >~ 1,000 at the bottom is attempting the transfer.
* The transfer's energy floor (ballistics, ANALYSIS ONLY): from the record's turn point on loop
  1's east end (1,184, 75, -2,648) to its landing on loop 2 (1,612, -448, -2,499) is 676 u
  horizontally and +151 u up; the slowest projectile that makes it leaves at
  v^2 = g (dz + sqrt(dz^2 + dx^2)) = 800 x 843.7 -> 822 u/s, i.e. E >= -2,648 + 422 = **-2,226**
  (the record had -1,923 there). GE's -2,329 is ~100 u short of even the ideal shot.
* 22:13 UTC: GE with 128 u cells (instead of 64), 30 min on one local core, to see whether a
  coarser archive pumps faster per CPU-minute (fewer cells, more selections per energy top).
