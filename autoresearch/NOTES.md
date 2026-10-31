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
* **E3 19:00-19:25 UTC: 36.386 (862M) / 46.904 / 55.332 / 57.416 (1,088M) / 56.679 (1,164M).**
  * auto-spine search 1 (711M weights) ended with a NEW SPINE (8,797 states = 61.6 s; success line
    d <= 4,376 = 31.2%; the GE line's 5,859 kept as a milestone) - the curriculum walked it back
    from 8,796 to ~8,040 by 1.2B at 79-91% band success. Search 2 (1,088M weights) running: its
    beams reach d ~2,796, about the greedy frontier (2,752), so its GE stage decides.
  * 1,088M (hand-on: ckpt_1088421888.pt at the eval's own step, pulled): 9/9 alike, 56.4-57.4% at
    ~97-102 s, E up to +1,600, 2,980 u/s: the first loop, the second trough, then the record's
    THIRD structure, swung along y (x -2,300..-1,600, y 70..2,800, z -1,300..+250) - and out of it
    to the WEST (x -2,850..-2,940) into a fall at ~104 s. The record (ANALYSIS ONLY) swings the
    same structure up to E ~2,100 (55-57% at t+200..213 s) and at t+305..310 drops onto a platform
    at z ~640 round (-1,560, 1,790) (E 2,176 -> 648: a landing), 60.9% at t+310, then climbs
    (70% at t+351 z 1,266; 86% at t+388 z 2,075; 90% at t+412; the finish at t+450, z 3,029).
  * Training at 1.2B: stall kills up to 38% (the new spine's band at 56 s on its line), reservoir
    deepest 43.3% of d0 (56.7%), energy +1,517/-1,114 u/ep, 235k steps/s with the search beside it.

### E3 REGRESSED after a STALE auto-spine (19:25-19:50 UTC) -> `--auto-spine-recheck`
* Evals after 1,088M: 56.679 (1,164M) / **33.583 (1,239M)** / 34.453 / 34.983 / 34.067 / 33.960
  (1,541M); stopped at 1,678M. The greedy fell back to the SECOND trough: 9/9 pump it only to E
  ~-550 (at 786M it reached -116 there and at 1,088M left it west for the third structure), then
  fall off its east end or stall to the patience cut at 31.7-34.0%. Training: stall kills 38% ->
  67%, the reservoir's deepest 43.3% -> 66.0% of d0 (56.7% -> 34%), unstuck T at its 0.2 cap.
* Cause: search 1 ran on the 711M weights (greedy frontier d 4,511) for ~22 min beside the
  training; its spine (8,797 states, end d 4,376 = 31%: that policy's beam line, ending at the
  second trough's east-end top - a DEAD END, where the 786M greedy also died) was adopted at
  ~1,072M, when the greedy was already at d 2,752 (57%). The curriculum restarted at its end with
  90% of spawns on its late part and the +50 on its 31% success line; within 75M steps the greedy
  had re-learned the dead end. The search compared itself with the frontier of the eval that
  STARTED it; nothing re-checked at adoption.
* Fix (commit cf1a7bc0, tests/python/test_auto_spine_recheck.py): `--auto-spine-recheck` - adopt a
  spine only if its end still beats the LATEST eval's greedy frontier by --auto-spine-gain;
  otherwise discard it and retry after --auto-spine-retry. Generic (a search result must still be
  ahead of the policy when it is used); off = as before.
* (ops) `pgrep -f '<pattern>'` inside a box.py ssh command matches the ssh shell's own command line
  and kills the session (CLAUDE.md's warning, met again): kill by pid file, or pattern on a
  process name that is not in the command.

### E4 `zlr1b4_e4` (19:57 UTC): resume of E3's 1,088M snapshot (57.4%) + the auto-spine loop WITH the recheck
`runs_ckpt.pt` = ckpt_1088421888 (md5 3c500701); the GE-line spine (success line 9.48%) restores
from the checkpoint - the spine under which 559M -> 1,088M went 24% -> 57% by itself; E3's flags
+ `--auto-spine-recheck`, snapcopy beside it.
* **E4 (19:57-20:24 UTC): 57.384 / 57.513 / 57.863 (1,239M) / 54.413 / 55.626 (1,391M)** - the 57%
  level HOLDS with the recheck (E3 fell to 34% within 150M steps of the stale spine), but it is a
  plateau: the eval frontier did not move for 375M steps, unstuck T at its 0.2 cap. Hand-on for
  57.863: ckpt_1239941120.pt (the eval's own step, pulled). Stopped at 1,450M (ckpt_latest
  1,438,646,272, md5 a692435c). Its search 1 (1,164M weights): branched beams to d 2,686 (57.8%),
  i.e. the greedy's own frontier; it was in its GE stage when stopped.
* **The 57% barrier (ANALYSIS ONLY - the record):** at t+305-306.5 the record swings the third
  structure at E ~2,100-2,200, flies up at (-1,635, 1,539, -569) with vz +1,396, hits a wall face at
  x ~-1,616 that takes its horizontal speed (v (0, 104, 996) at t+307: E 1,903 -> 654, a loss of
  ~1,250 u) and rises straight onto a platform at z ~617-640 round (-1,590, 1,790): 60.9% at t+310,
  then it climbs from there (70% at t+351 z 1,266 ... the finish at t+450). The 1,239M greedy comes
  within 640-820 u of that platform at 80-90 s at (-1,870, 1,300, 80..275) with E 1,000-1,370, then
  falls out of the structure (dead at 91-99 s).
* **Why the objective blocks it:** the per-call energy potential charges that conversion of speed
  into a stop on a higher ledge: ~1,250 u x K 12 / 1,000 = -15 reward on one call, against +3 of
  field progress (57.85% -> 60.9%) - and the climb that pays afterwards is 40-80 s away, past the
  20 s horizon. Energy is the resource the loops build; charging its SPENDING at the frontier is
  wrong on any map where a ledge has to be landed.

### E5 `zlr1b4_e5` (20:26 UTC): E4's 1,438.6M weights + `--energy-pot-front 0.01`
The recipe's own form of the fix (incumbent recipe: `--energy-pot-front 0.01`): the per-call energy
term is paid only until the episode's ratchet record reaches the frontier band (progress >= 0.99 x
the reservoir's deepest, re-read every iteration); inside the band energy is free to spend - so
the pumping behind the frontier stays paid and the landing at the frontier costs nothing.
`--energy-pot-front-bonus` stays 0 (no one-time bonus: the per-call signal is what taught the
pump). Everything else as E4 (auto-spine + recheck, snapcopy).
* **E5 (20:26-20:50 UTC): 56.232 / 55.415 / 56.743 / 55.772** (1,439-1,666M) - no break yet; only
  1-4% of training episodes reach the band (where energy is now free); the greedy now dies at
  ~82 s flying NORTH out of the third structure (y 2,540-3,265). Eval frontier unchanged since
  1,088M (713M steps).
* auto-spine search 1 in E5 (1,439M weights): beam + 5 branched beams all end at d 2,816 (55.7%,
  greedy frontier 2,889); its GE stage HIT --max-cells 4,000,000 in 2.9 min (64 u x energy-32
  cells explode at 1,000-2,900 u/s in a 700 x 2,700 x 1,500 u structure) and could add nothing;
  killed. auto_spine.py's GE cell sizes are fixed (not a trainer flag).
* GE zl3 (20:52 UTC, by hand, beside E5, 14 threads, 15 min): rooted at the auto-spine beam line's
  OWN third-structure states (beam_frontier.npz states 7,700..11,699 every 25 = 160 roots, the
  policy's search line from the true start), COARSER cells `--cell 128 --cell-energy 64
  --max-cells 8000000`, the rest the explorer config above. Looking for the next ledge (the
  record's platform at 60.9%); a deeper chain splices onto the beam line (ge_spine.py) as a spine.
* (ops, again) a box.py ssh command that kills by `grep -E '[x]yz'` still matches its OWN shell
  when the same command later LAUNCHES xyz (the text is in bash -c's argv): kill and launch in
  separate ssh calls, or put the work in a script file on the box.
* **E5 final (stopped 21:25 UTC at 2.16B): 56.2 / 55.4 / 56.7 / 55.8 / 55.7 / 56.7 / 56.0 / 55.9** -
  `--energy-pot-front 0.01` did NOT open the platform, and it is the wrong form here: inside the band
  the energy term pays nothing, so a policy that cannot make a new record there earns only the
  time penalty by staying alive - dying becomes the better end. E5's greedy died EARLIER (~82 s
  against E4's 91-105 s), and the energy it needs is only built by staying alive (the record has
  E ~1,900 at its approach to the pillar; the policy peaks at 1,500-1,700 and is still GAINING
  when it flies out of the structure at ~100 s).
* GE zl3 (15 min, coarse cells, 160 roots in the third structure from e5's beam line): 2.18M cells,
  best 56.09% - nothing past the plateau. Its top cells sit at exactly z 164.19 (1,281 cells, vz
  ~0, over x -3,031..-16, y 1,227..3,344), but the core's physics flies a player from those spots
  freely to 566-626 u (no clip ceiling, no water): the explorer's roots carried at most E ~1,360
  and uniform bursts do not pump this structure, so 164 is just where their energy tops out.
  The record's platform is the top of a solid pillar at (-1,590, 1,790) (occupancy: solid z
  -706..~573), reached by flying into its west face and rising along it.

### E6 `zlr1b4_e6` (21:27 UTC): back to E4's objective + a spine on the policy's OWN current route
Resume of E4's ckpt_latest (1,438.6M, md5 a692435c; per-call energy everywhere, as E4) with the
spine = e5's auto-spine beam line (`autoresearch/spines/zlr1b4_e5beam_line.npy`, md5 0f507f6b:
beam_tas from the TRUE start on the policy's weights, 11,700 ticks = 81.9 s, start -> both troughs
-> the third structure, 55.7%), the band PINNED at the true start (`--demo-tau0 0 --demo-rate 0`,
uf2r2b4 x7's rule: half the spine draws at the eval's own start, half uniform along the line - a
third of the line is the third structure, where the policy dies), the GE line's 9.48% success
line kept paying as `--demo-milestones 5859`. No auto-spine (its searches could not beat the
greedy here, cost ~18% throughput, and a new spine with demo-rate 0 would pin the band at its end).
Why: the frontier skill is SURVIVING and pumping the third structure up to ~1,900 u; the GE-line
spine put 90% of spawns on the first 27 s of the route.
* **E6 (21:27-21:45 UTC, 1.44-1.77B): 56.232 / 57.505 / 56.623 / 55.945 / 56.084** - the plateau
  again in 330M steps (stall kills ~40%: the beam line's last 20 s are a crawl, so spawns there go
  nowhere). Its greedy dies at 95-112 s. Traced (1,666M ep0): a west-end swing top at (-2,342,
  1,308, 143) = 55.5%, then the swing east passes the pillar's west face at (-1,707, 1,796, -111)
  - right where the record climbs - but on the DOWNswing (vz -640; the record hits it rising at
  +1,396), and carries on north-east off the structure into the void (z -1,450 at (-932, 2,955)
  where the record rides at ~-120) and the bottom teleport.
* The frontier skill is SURVIVAL in the third structure: the record spends ~150 s there pumping E
  ~500 -> ~2,100 (t+150..305); the policy enters at ~80 s with E ~1,000 and is dead ~25 s later,
  still gaining energy. The objective makes dying free (no death refund on the energy term, the
  ratchet keeps paid progress; only the lost future income argues for surviving).

### E7 `zlr1b4_e7` (21:47 UTC): E6 + `--fail-pen 10`
Resume of E6's ckpt_latest (1,764.75M, md5 7ec885e9), the same spine flags, plus the trainer's
flat terminal penalty on death (falls, kill triggers, stall kills; truncation exempt): 10 = a fifth
of the success bonus, a constant that reads nothing on the map. CLAUDE.md records --fail-pen 10 as
inconclusive on cannonball scratch arms (4.3x for on one card, 8.7x against on another) - this is a
different question (a trained policy that dies at its frontier while still gaining).
* **E7 (21:47-22:05 UTC, 1.76-2.17B): 56.077 / 55.215 / 56.860 / 56.194 / 55.780 / 55.98** - no
  break. `--fail-pen 10` changed the BEHAVIOUR as intended but the wrong way for the score: the
  greedy now survives the third structure (episodes end at 117-145 s by the eval's patience cut,
  alive at z 65..-750) but plays safe - E max ~1,040 (E4: 1,500-1,700), speed max ~1,920 (E4:
  2,900) - and stops gaining. Training stall kills 63% (each a -10 now). It traded energy for
  survival; the platform needs both.
* GE zl4 (22:07 UTC, full CPU, 28 threads): rooted at E4-1239M's OWN greedy states (record_ckpt
  --dump-states, 3 episodes, ticks 10,000..death-300 every 15 = 624 roots, E 639-1,467) with fine
  cells (64 u x energy 48): 6.35M cells in 3.6 min, best d stuck at 2,689 (start 2,707) from the
  first 20 s - killed. Uniform bursts LOSE energy at these speeds (the explorer's energy never
  exceeds its roots'), so they cannot build the ~1,900 u the pillar climb needs.
* Beam dh1 (22:12 UTC): `tools/beam_tas.py` from the TRUE start on E4-1239M (the policy's own
  proposals, 4,096 envs, resample 25) ranked by `--score dh --score-energy 2` (-d + 2 x energy
  height: lineages that carry energy are kept over the ones that die), 28,571 ticks (200 s).
* Beam dh1 result: ALL 4,096 lineages dead by t = 91.7 s (population extinct at 13,100 ticks); the
  frontier d 2,731 (57.1%) at 88.2 s. The policy's own action distribution, selected every 25
  decisions for depth + energy, cannot sustain the third structure past ~90 s: a skill gap, not
  bad luck in the greedy. (A beam ranked by plain d ends at the same depth: auto-spine's beams.)
* So far the two ends: E4 (no death penalty) pumps to E 1,500-1,700 and dies at ~100 s; E7
  (fail-pen 10) survives but stops pumping (E ~1,040). The platform needs survival AND ~1,900 u.

### E8 `zlr1b4_e8` (22:18 UTC): E4's 1,239M weights (the 57.86% policy) + `--fail-pen 3`
A milder death penalty on the aggressive policy (E4's config otherwise restored: the GE-line
spine, per-call energy K 12, energy-stall 32), to look for survival AND energy at the frontier.
* **E8 BREAKS THE 57% PLATEAU (22:18-22:40 UTC): 57.735 (1,240M) / 58.751 / 58.858 / 62.286
  (1,467M) / 62.327 (1,542M).** `--fail-pen 3` is the middle the two ends lacked: the policy keeps
  pumping AND survives the third structure ~20 s longer. 1,542M (hand-on ckpt_1542455296.pt, the
  eval's own step, pulled): 9/9 at 61.8-62.3% by ~110-116 s, E max 2,520-2,630 (the record has
  ~2,100 at its pillar approach), speed to 3,240 u/s. Traced (ep0): 95 -> 117 s swinging the third
  structure along y (x -2,300..-550), E 1,519 -> 2,336 (~37 u/s), the swing tops rising with the
  energy - 57.8% (z 385) -> 59.5% (459) -> 61.0% (516) -> 60.9% (551) - then at ~119 s (E 2,219) it
  swings EAST out of the structure ((575, 2,155, 571) -> (2,905, 377, -709)) and falls. It has the
  energy now; the climb onto the pillar (60.9%) and on up (the record: 70% at z 1,266, 86% at
  z 2,075) is the next skill. Training: reservoir deepest 37.5% of d0 (62.5%), stall kills 10%.
* E8 23:05 UTC: 61.242 (1,693M) / 62.079 / **62.774 (1,844M)** / 61.898 / 60.829 (1,995M). Hand-on
  for 62.774: ckpt_1844445184.pt (the eval's own step, pulled). The 62.77% run's swing tops have
  moved EAST (x -110..+220, y 1,150-1,700, z ~600, E ~2,350-2,640); it leaves the structure at
  ~117 s and falls. Some evals now lose 1-3 episodes early (14-75 s): fail-pen 3 costs a little
  reliability at the start.
* What lies beyond (ANALYSIS ONLY - the record): the platform at z 641 is the floor of a FOURTH loop
  structure above the third (x -1,600..-760, y 0..2,700, z 630-1,500): from t+310 (E 648, nearly at
  rest) it swings that one along y pumping E back to ~2,600 by t+378 (61-72%), then climbs out of
  its top (70% at t+382 z 1,272; 78% t+386 z 1,680; 84% t+390 z 2,041; finish t+450 z 3,029). So the
  next skill is getting from the third structure (swing tops z ~600) up INTO the fourth.
* Beam d2 (23:10 UTC, beside E8): beam_tas --score d from the TRUE start on E8-1844M (4,096 envs,
  200 s) - the policy survives the third structure now, so its own lineages may reach further than
  E4's (all dead by 92 s).
* Beam d2 result: frontier d 2,428 (61.8%) at 126.7 s, the population extinct at 130.9 s (E4's beam:
  92 s) - E8's policy survives ~40 s longer, but no lineage of its own enters the fourth structure
  and none beats its greedy (62.77%).
* E8 23:15 UTC: 60.829 / 61.116 / 61.004 (1,995-2,146M): a second plateau ~61-62.8%. Stopped at
  2,181M (ckpt_latest 2,181,038,080, md5 b543d331).

### E9 `zlr1b4_e9` (23:18 UTC): E8's weights + a whole-route spine on its own beam line
Resume of E8's ckpt_latest with the spine = beam d2's line (`autoresearch/spines/
zlr1b4_e8beam_d2_line.npy`, md5 4bd2c9a3: beam_tas from the TRUE start on E8-1844M, replay ok,
18,100 ticks = 126.7 s to 61.8%, through all three structures), band pinned at the true start
(`--demo-tau0 0 --demo-rate 0`), the GE line's 9.48% kept as a milestone, --fail-pen 3 restored.
~20% of all spawns now land in the third structure (the GE-line spine put 90% of spawns on the
first 27 s of the route, which the policy has mastered). E6 tried this with fail-pen 0 and its
spawns there just died; E8's policy survives there now.
