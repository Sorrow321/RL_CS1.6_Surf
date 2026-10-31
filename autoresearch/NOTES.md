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

## zlr1b4 (round 1, slot 4) - 2026-10-31 17:10 UTC start

Box 58559819: RTX 3090 + Ryzen 9 5950X (32 threads), 0.203 $/h, Ontario CA (rented 17:20,
deployed 17:21). 237-248k steps/s on the recipe. (A deploy run WHILE a trainer runs fails its
gpu_health check - the trainer is using the GPU - after the git checkout has already happened;
that is not a bad host. box.py's restamp step is then skipped: run tools/restamp_maps.py by hand.)

### The record's first loop (ANALYSIS ONLY, CLAUDE.md section 0)
`$AR_HOME/wr/surf_za_loop_wr.jsonl`, t from the curtain exit (record tick 327):
* t+0..4 s: from the curtain (-1,696, -1,839, -3,147) it drops south-west down a ramp to
  (-2,304, -1,337, -3,926) at 1,215 u/s (75 u above the bottom teleport *21), swings north along
  x ~ -2,300 and climbs into the west end of a long east-west trough at (-1,513, 50, -3,211).
* t+4..62 s: it swings ACROSS the trough along x, -1,700 <-> +1,300 at y 0..300, a half-period of
  ~5 s (12 swings), the floor at z ~-3,330, turning round in a loop of the curved ramp at each end
  (west end up to y ~1,000, z ~-2,500 late). E = z + |v|^2/1600 rises -3,076 -> -2,050 (~30 u/s
  while pumping; a bad landing at t+32 drops it 700 u and it pumps it back), speed 345 -> 1,390.
  It is airborne almost all the time (onground spikes only).
* t+62: it leaves for the next trough up (x -1,000..2,900, y -300..-1,500, z -2,500..-1,700 by
  t+100) and pumps that one the same way. The map is a stack of half-pipes.
* Physics of the pump: optimal air strafing adds a CONSTANT ~0.51 u of E per 7 ms frame at any
  speed (accelspeed 21 = 100 x 30 x 0.007; d(v^2) = 2 x 21 x 9 + 21^2 = 819), i.e. <= 73 u/s of
  E; the record pumps at ~40% of that. Gravity is conservative (GoldSrc's half-step keeps E
  exactly), so energy comes ONLY from strafing and is lost only at contacts.

### Why the recipe cannot climb this (and the scratch control)
* `--energy-pot 3 --energy-pot-front 0.01 --energy-pot-front-bonus 1` pays NOTHING on the way and
  a kinetic bonus once at the frontier band; with an empty reservoir the band edge is d0, so the
  bonus is paid at the spawn (v = 0) and nothing after. No energy signal at all here.
* `--time-pen 0.001` (0.1/s) with no other income makes dying early optimal.
* `--stall-secs 15` kills a training episode with no field progress for 15 s, and
  `--eval-patience 30` cuts a GREEDY EVAL episode with no field progress for 30 s: the record
  pumps ~60 s with NEGATIVE progress before it gains any. So even a policy that pumps could never
  show it in an eval: the grader would see it cut at 30 s.
* Scratch control `zlr1b4_ctl` (the recipe as is, 3.5e8 steps): 0.029 (0.5M) / 0.541 (76M).
  Greedy at 76M (zl_eval_ends.py): 6/9 walk north from the spawn and drop through a hole at
  (-1,650, -1,100) into the bottom teleport at 16-28 s; 3/9 wander the start floor until the
  30 s patience cut; E max -3,072 (the spawn's -3,074): no energy ever gained.

### E1: energy is progress (generic, both changes)
* NEW `--energy-stall EPS` (commit 5d07d0df, tests/python/test_energy_stall.py): the training
  stall kill AND the eval patience also re-arm when the episode's best energy height has risen EPS
  above its value at the last energy re-arm (cumulative, any rate, like --eval-patience). EPS 32 =
  the eval patience's own field threshold. Generic: on any map a player adding energy is doing
  something the field cannot see; nothing adds energy at rest / at a constant speed / in free fall,
  so those still stall. No constant reads this map. Off = no state, bit-identical by construction.
* `--energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0`: the per-call energy
  potential everywhere - the form AND the constant uf2_x13 and pb_r26 (both lineages' best
  policies) were trained with. At K 12 a strafe pays up to ~0.9 reward/s against the 0.1/s time
  penalty, the record's pump ~0.36/s.
* Read it on: E max and the share of greedy time gaining E (zl_eval_ends.py), then progress
  (the first trough's swing tops reach z ~-2,900..-2,500), then the next trough.
* Control `zlr1b4_ctl` stopped at 157M (flat): 0.029 / 0.541 / **0.587** (151M). At 151M all 9
  greedy episodes dive into the bottom teleport within 3.6-6.1 s (the time-penalty trap learned).

### E1 `zlr1b4_e1` (17:33-17:50 UTC, 291M): a NULL
Recipe + `--energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0 --energy-stall 32`.
Evals 0.343 (76M) / 0.200 (151M) / 0.174 (227M) vs the control's 0.541 / 0.587. Greedy E max
-3,068 (spawn -3,074): 9/9 walk NORTH-west off the start platform at (-1,780, -950) and fall to
the bottom teleport at 8-19 s. Training episodes did gain energy (+1,717 u/ep at 10M, +650 u/ep
later, losses 800-2,250), stall kills 41-54%, yaw sigma at its cap 2.718 from ~150M (the "rot":
no signal in the yaw advantages). The reservoir never got deeper than 0.64%. So the energy
objective is not the first blocker: nothing the policy does reaches a place where pumping pays.
The record drops off the platform's WEST edge onto the start basin's ramp; the policy walks off
the north edge into open air.

### Go-Explore from the true spawn FINDS the first trough (17:50-18:02 UTC, box CPU, 12 min)
`tools/explore_phase1.py --exact-chains --cell 64 --cell-energy 32 --cell-heading 1
--view-follow=-90,-60,-30,-15,0,15,30,60,90 --select-energy 3 --act-every 4 --decisions 100
--maxvel 3500 --tick-ms 7 --envs 4096 --threads 30 --max-minutes 12` (the pb config at this map's
tick; runs/ge/zl1 on the box): 1.54M cells, 9.1M runs, best d 5,759 = **9.48%** (from 8.7 min on),
E max -2,367 (the spawn -3,074: uniform bursts in the velocity frame DO pump here, unlike pb's
half-pipe). Progress bands: 159 cells in [9,10)%, 732 in [8,9), 1,349 in [7,8); the deepest are
airborne at the trough's EAST end top (x 1,000-1,550, y 100-550, z -2,800..-2,900, |v| 200-700).
`tools/ge_replay.py` replays the deepest chain (46 cells) EXACTLY, 45/45 links, into a 3,960-tick
line (27.7 s; `autoresearch/spines/zlr1b4_ge1_line_a.npy`, md5 ccb2c661):
* t 0-3 s west from the spawn over the curtain, south-west to (-1,887, -1,776);
* t 3-17 s down into the START BASIN and swinging round its west side (x -2,200..-1,700,
  y -2,500..-300, z -3,800..-3,200), pumping E -3,150 -> -2,769 (~27 u/s: the record does not
  pump here - it enters the trough at t+4 with E -2,915);
* t 17-19 s up into the trough's west end (-1,529, 127, -3,190) -> (-1,268, 295, -2,897);
* t 19-27.7 s swinging east along the trough, E -2,750 -> -2,584, ending at its east-end top
  (1,218, 495, -2,801), 9.48%.

### E2 `zlr1b4_e2` (18:04 UTC): from scratch, E1's energy flags + a backward curriculum on the GE line
`--demo-file runs/ge/zl1/line_a.npy` (box) with uf2r1b1 A13's curriculum constants
(`--demo-window 5 --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5 --demo-rate 0.2
--demo-min-ep 50 --demo-front-frac 0.5 --demo-success-d 100 --demo-success-bonus 50`),
`--no-respawn-novel` (it refuses a spine) with `--respawn-backward-mix 0 --int-real-only 0` (both need
--respawn-novel; the first launch died on that). No BC (a GE line's actions are uniform noise). The
search -> line -> curriculum loop is the lineage's generic mechanism (README: princessburglar,
unitfarmer2); nothing names a place. Read it on: the curriculum position (demo starts [tau, ..]),
then the greedy from the true start.
* E2 at 215M (18:17 UTC, 288k steps/s): the curriculum walked tau 3959 -> 2154 (t 15 s on the
  line, the basin pump) - 28% band success at [3139..], 7% at [2384..], 85% at [2154..]; 33k ->
  83k spine-line bonuses paid; reservoir min d 5,746 (9.7%), training reward +45/ep, energy
  +422/-247 u/ep, stall kills 11%, yaw sigma 0.11 (no rot: the spine gives the yaw advantages a
  signal). Greedy from the true start: 0.481 (76M) / 0.004 (151M) - it idles on the start floor
  until the 30 s cut; expected until the band reaches the start (10% of spawns are the start).
* GE zl2 (18:18 UTC, beside E2 with 14 threads; the trainer uses ~15 of the 30.7-CPU quota):
  the same explorer rooted at line_a's last 1,400 ticks (stride 10, 133 roots: the trough
  stretch), 15 min - can the trough be pumped higher and left for the next one? A deeper chain
  splices onto line_a's prefix (tools/ge_spine.py) as the next spine, the old success line kept
  as a --demo-milestones payment.
* **E2 WORKS: the policy climbs the first loop from the true start (18:20-18:37 UTC).** Greedy evals
  0.029 / 0.481 / 0.004 / 0.671 / **10.853** (302M) / 11.148 / **16.383** (453M) / **24.031** (529M).
  The curriculum reached tau 114 (0.8 s from the true start) by ~500M; the success line (9.48%)
  was crossed by 9/9 greedy episodes from 302M on.
  * 453M (zl_eval_ends.py): 9/9 alike - west off the platform, the drop into the start basin and
    the swing north along its west wall (the RECORD's start, not the GE line's basin pumping), in
    the trough at 5 s, pumping across it at ~54 u/s of E (the record ~30), E -2,890 -> -1,924 by
    23 s, the east-end top at 12.9% (20.9 s), then out over the trough's west part NORTH
    (-500, 1,000..1,900, z -2,450) to 15.5-16.4% at 24 s and a fall into the bottom teleport at 26 s.
  * 529M: 9/9 at 22.2-24.0% by ~30 s, E max -1,369..-1,714, z up to -1,875; 5/9 alive in the
    record's SECOND trough region (x 900..2,200, y -1,950..-340, z -2,050..-2,600) until the 30 s
    eval-patience cut at 59-66 s (pumping there without new energy), 4/9 fell. The record: 11% at
    t+90 s, 43% at t+150.
  * Training at 510-560M: reward 44-56/ep, energy +1,208/-485 u/ep, stall kills 4%, episode
    length ~20-32 s, reservoir deepest 82.0% of d0 (18%) at 510M.
  * Stopped at 559.9M for E3 (ckpt_latest 559,415,296, md5 7c95d700..., pulled: the hand-on for the
    24.031 eval - no ckpt_<step> before it, --ckpt-every 1e9).

### E3 `zlr1b4_e3` (18:38 UTC): E2's 559M weights + the auto-spine loop
Resume (the recipe, the energy flags and the curriculum restore from the checkpoint) +
`--auto-spine --auto-spine-async --auto-spine-milestones --auto-spine-patience 3e8
--auto-spine-delay 5e7 --ckpt-every 2e8 --demo-tau0 114 --demo-cooldown 5 --demo-grow-max 40`
(the uf2r1b1 A11 loop: after an eval whose quorum (5/9) crosses the spine's success line - or
3e8 steps without a new spine - a beam from the TRUE start on the newest weights (GE from its
last seconds if the beam falls short) gives a spine 200 u past the greedy frontier, the
curriculum restarts at its end, old success lines keep paying as milestones) + snapcopy beside it.
Why: E2's spine ends at 9.48% while its greedy reaches 24%; 90% of its spawns are on the first
27 s of the line, so the frontier trains only through respawn-greedy rows and long episodes.
* **E3 (18:38-19:00 UTC): 26.307 / 30.872 / 33.645 / 35.070** at 560 / 635 / 711 / 786M (box
  extended to 22:00 UTC). At 786M 9/9 alike: up the first loop by ~23 s, the second trough
  (x -1,600..3,000, y -1,000..-100, z -2,560..-1,240) pumped from E -833 to -116 by ~57 s
  (2,490 u/s max), 34.0-35.1% at its east end top (2,404, -877, -1,241), then north-east off the
  east end and a fall into the bottom teleport at (2,450, -500) at 59 s. The record (ANALYSIS ONLY)
  leaves the second trough at its WEST end at about the same energy (E -131 at t+135, (-2,032,
  2,560, -999)) into a third structure it swings along y (x -2,300..-750, y 0..3,000, z -1,300
  ..-120; 30-47% at t+140..195).
  * Hand-on for 35.07: snapcopy's ckpt_785383424.pt (1M before the eval, pulled).
  * auto-spine search 1 (on the 711M weights, beside training, slow: >17 min - the beam has to fly
    ~60 s lines at 4,096 envs; training drops 286k -> 234k steps/s while it runs): beam + 5 branched
    beams end at d 4,381-4,420, behind the moving greedy frontier (4,294 at 711M, 4,179 at 786M,
    4,094 at 862M), so it went on to its Go-Explore stage (10 min, rooted before the line's stall).
    The policy is outrunning its own search here.
