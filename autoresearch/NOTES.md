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
* **E9 (23:18-23:45 UTC, 2.18-2.58B): 62.553 / 62.294 / 61.233 / 61.426 / 62.252 / 61.631** - the
  whole-route own-beam spine holds the ~62% level but does not move it (band success at the true
  start 54%: the full 126.7 s line from the start). Stopped at 2,577M (ckpt 2,562,719,744, md5
  207c4704).

### E10 `zlr1b4_e10` (23:46 UTC): E9's weights + `--fail-pen 5`; GE zl5 beside it
* E10: one change on E9 - a firmer death penalty (3 -> 5) for a policy that now pumps to E ~2,600
  and dies by over-swinging out of the structure at ~120 s; does more survival pressure buy
  control at the top (E7 showed 10 from a low-energy policy makes it stop pumping).
* GE zl5 (14 threads, 20 min, coarse 128 u x energy-64 cells): rooted at E8-1844M's OWN greedy states
  (record_ckpt --dump-states, 4 episodes; the last ~25 s before each fatal fall of the 3 that
  reach 61.5-62.1%, every 20 ticks = 492 roots, E up to 2,431, z up to 580) - the first explorer
  roots with the energy the climb needs: does anything get up into the fourth structure?
* GE zl5 (20 min, 492 roots from E8's own high-energy swing states, E up to 2,431): 3.25M cells, best
  never past its roots' (2,376 on the explorer's field). Third explorer null at this stretch: uniform
  bursts cannot hold or use the energy (zl3, zl4, zl5).
* **E10 (23:46-00:06 UTC, 2.56-2.91B): 60.430 / 61.961 / 60.948 / 61.539 / 61.397** - fail-pen 5 =
  fail-pen 3 here: the ~61-62% level. Stopped at 2,914M (ckpt 2,907,701,248, md5 12b2199e).

### E11 `zlr1b4_e11` (00:08 UTC): E10's weights + `--front-band 0.05`
The frontier temperature (`--front-temp 1`: every head sampled at T 2) applied in the top 5% of
the explored progress range instead of the top 1% - i.e. over the upper part of every swing in the
third structure, not only at the swing tops. The policy is near-deterministic there (yaw sigma
0.04-0.08) and the climb is a different swing ENDING (into the pillar's face, rising) that its own
samples never try (beam d2: no lineage). Evals stay greedy. Generic: a band relative to the run's
own frontier; no map constant.
* **E11 (00:08-00:30 UTC, 2.91-3.21B): 29.012 (its first eval, an outlier) / 61.669 / 61.128 /
  61.950 / 61.647** - the wider frontier temperature band holds ~62% and does not move it (training
  frontier 62.8%, stall kills 36%).
* Branched beams (00:31 UTC, beside E11): tools/beam_tas.py on E8-1844M from the TRUE start with
  half the population (2,048 envs) forked at 91 / 96 / 101 / 106 / 111 s - inside the third structure,
  before the swings that pass the pillar - into held random VIEW offsets (`--branch-at tT:2048
  --branch-burst 140 --branch-hold 70 --branch-protect 5 --branch-view-only`: the auto-spine's
  branched-beam tool, which only forks 0.5 s before the greedy's death). A manoeuvre the policy's
  own narrow proposals never try (yaw sigma 0.03-0.08): does any branch climb into the fourth
  structure?
* Branched beams (E8-1844M, forks at 91/96/101/106/111 s, 2,048 envs into held random view offsets,
  protected 5 generations): frontiers d 2,401 / 2,379 / 2,410 / 2,417 / 2,372 (62.0-62.7%) at
  127-132 s - none past the greedy's 62.77%, none into the fourth structure.
* E11 later: 61.786 / 61.122 / 61.906 / 61.006 (3.36-3.59B); stopped at 3,623M.

### E12 `zlr1b4_e12` (01:00 UTC): E8's 62.77% weights + `--int-coef 1.0`
The recipe's count-based novelty (cell x view-sector x speed key) at 4x (0.25 -> 1.0) on the best
weights, E8's config otherwise: a stronger push toward unvisited cells in the upper third structure
(the faces of the pillar, the space above it), where no search has found a way up.
* **E12 (01:00-01:27 UTC): 62.679 (its own first eval) / 45.355 / 53.803 / 54.329 / 53.511 /
  53.077 - NEGATIVE.** 4x novelty pulled the greedy back from 62% to ~53%: the count bonus paid 34
  reward/episode (E8: ~4-8) and the training frontier fell to 57%. Keep `--int-coef 0.25`.

### E13 `zlr1b4_e13` (01:28 UTC): E8's exact config from its 62.77% snapshot, to the round's end
Nothing new: more greedy evals of the best configuration (each a chance above 62.774) and, if one
comes, a stronger hand-on for the next round.
* **E13 (01:28-01:55 UTC): 62.303 (its own first eval) / 61.948 / 61.441 / 61.811 / 61.974 /
  62.971 (2,222M) - a new best by 0.2, inside the level's noise.** At 2,222M: 4/9 reach the third
  structure's top (62.4-62.97%, E up to 3,033 u), but 5/9 die at ~51 s at the second trough's exit
  ((-2,330, -44), 31.6-31.8%): the max is what scores, the median (31.8%) shows the fragility.
  Hand-on: ckpt_2216165376.pt (6.3M before the eval, the newest at or before it; pulled).
* **E13 02:20 UTC: 62.512 / 62.560 / 63.888 (2,600M) / 62.686.** At 2,600M 9/9 alike (median
  63.69%): alive 190-215 s (E8: ~120 s), E max 4,780-5,070 (twice E8's), speed up to 3,800 u/s, swing
  tops at z 623-663 (above the record's platform level z ~641) = 63.3-63.9%, dying at 190-215 s off
  the NORTH edge ((-1,540..-2,640, 3,100-3,340)). The same flags as E8, ~750M steps later: longer
  training keeps paying in survival and energy. Training frontier 64.4%, stall kills 0%. Hand-on:
  ckpt_2559049728.pt (41M before the eval: the newest at or before it - snapcopy had no copy
  between 2,559M and 2,620M); ckpt_2620391424.pt pulled too.
* **E13 02:50 UTC: ... / 64.241 (2,902M, median 64.0%) / 63.947 / 64.008 / 63.578 (3,128M).** Still
  creeping up with nothing changed - training time alone (the 62% plateau of E8-E12 was partly just
  impatience: E8 had ~600M steps past its jump, E13 ~1.3B). Hand-on for 64.241: ckpt_2900885504.pt
  (1M before the eval; pulled). Training frontier 64.6%.
* E13 03:22 UTC: 64.428 at 3,506M (median 64.05%) - hand-on ckpt_3499622400.pt (6.3M before; pulled).
* E13 03:52 UTC: 64.671 at 4,034M (median 64.57%) - hand-on ckpt_4032299008.pt (2.1M before; pulled).

### zlr1b4 round-1 lesson (FINAL, wrap-up 04:31 UTC)

**Result: 64.671** (E13 `zlr1b4_e13` at 4,034M; median 64.57%, 9/9 at the third structure's top) -
surf_za_loop from the TRUE start, champion-free, no demo in training, from scratch on 2026-10-31
(one RTX 3090 box, 0.20 $/h). Scratch control (the recipe as it stood): **0.587**. The record's
progress for reference (analysis only): 11% at t+90 s, 43% at t+150, 63% at t+330 - the policy is at
64% by ~160 s. Hand-on: ckpt_4032299008.pt. The lineage of weights: e2 (scratch, 559M) -> e3 1,088M
-> e4 1,239M -> e8 1,844M -> e13.

**What worked, in order (each a generic mechanism, constants that read nothing on the map):**
1. **Energy as progress for the stall rules: NEW `--energy-stall 32`** (training stall kill AND eval
   patience re-arm on an energy-height record, cumulative) + the per-call energy potential
   everywhere (`--energy-pot 12 --energy-pot-front 0 --energy-pot-front-bonus 0`, the form and K
   the uf2/pb best policies trained with). Necessary (the record pumps 60 s with NEGATIVE field
   progress; the 30 s eval patience would cut any pumping line) but NOT sufficient: E1 alone never
   reached the ramp where pumping pays.
2. **A Go-Explore line from the true spawn as the first spine**: `explore_phase1 --exact-chains
   --cell 64 --cell-energy 32 --cell-heading 1 --view-follow=-90,...,90 --select-energy 3 --act-every 4
   --maxvel 3500 --tick-ms 7`, 12 min of box CPU -> 9.48% (uniform bursts in the velocity frame DO
   pump the start basin), `ge_replay` exact line -> backward curriculum (uf2 A13 constants) on top
   of 1. **From scratch to 24% in 530M steps** (E2): the policy took the record's own start route
   and pumped ~2x the record's rate.
3. **`--auto-spine --auto-spine-async`** (beam from the true start on the policy's own weights ->
   next spine): 24 -> 57.4% by 1,088M (E3). **But an async search can return STALE** (22 min on
   the 711M weights, adopted when the greedy was 26 points further; it dragged the greedy back to
   34% within 75M steps) -> **NEW `--auto-spine-recheck`** (adopt only if still ahead of the LATEST
   greedy frontier).
4. **`--fail-pen 3`** (E8): the policy at 57% pumped the third structure to E ~1,700 and died at
   ~100 s by over-swinging out of it - dying was free. fail-pen 3 kept the pumping AND added
   survival -> 62.8%; fail-pen 10 (E7) killed the pumping instead (survives at E ~1,040).
5. **Training time** (E13 = E8's flags from its best snapshot, nothing changed): 62.8 -> 64.7% over
   ~2.2B more steps, still creeping up at the end (alive ~200 s, E ~5,000). The 62% "plateau" of
   E9-E12 was partly impatience.

**What did not (all measured here, one seed each):** `--energy-pot-front 0.01` (energy free in the
frontier band: dying becomes the better end there, the greedy died earlier); spines on the policy's
own beam lines (E6, E9) and a wider `--front-band 0.05` (E11): no gain over plain training;
`--int-coef 1.0` (E12, 4x novelty): NEGATIVE, 62 -> 53%; `--fail-pen 5` (E10) = 3. Searches past
57-64% found nothing: Go-Explore x3 from the policy's own states (uniform bursts lose energy at
1,000-3,000 u/s, the explorer never exceeds its roots' energy; auto_spine.py's fixed 64 u x energy-32
cells hit the 4M cap in 3 min), an energy-ranked beam (all 4,096 of E4's lineages dead by 92 s), a
depth-ranked beam on E8 (dead by 131 s) and 5 branched beams forked inside the third structure.

**Where the next round starts (the open problem):** the FOURTH loop structure. ANALYSIS ONLY (the
record): at E ~2,100 it flies, rising (vz +1,396), into the west face of a pillar at (-1,590, 1,790)
(solid to z ~573), rises along it to the top (z ~617-641, 60.9%), then pumps a fourth structure above
the third (x -1,600..-760, y 0..2,700, z 630-1,500) from E ~650 to ~2,600 (61-72%) and climbs out to
the finish (84% at t+390, finish t+450 at z 3,029). The policy pumps the third structure to E ~5,000,
survives ~200 s, its swing tops at z ~650 (64%), and dies off the north edge. It passes the pillar
on DOWNswings. Ideas not tried: auto-spine with recheck on the E13 weights (its beams now live
~200 s); a search that keeps the policy's own pumping actions but forks the swing TIMING; longer
training of E13 (it never stopped improving).

**Harness / ops notes:** a `box.py deploy` while a trainer runs fails gpu_health (the GPU is busy)
after the checkout has already happened - run restamp_maps by hand; a box.py ssh command that kills
by pattern (pgrep -f, grep '[x]yz') matches its own shell when the same command text names the
target - kill by pid file, launch in a separate call; snapcopy can leave a 40M gap before an eval
(pull --snapshot the newest at or before it).

* **Wrap-up (04:31 UTC):** E13 stopped at ~4.66B with the round; its last evals 63.81 / 63.94 / 63.46 /
  63.60 / 64.03 / 63.23 (4.26-4.64B, medians 63.0-63.9%) - the level holds, 9/9 every eval. **Final:
  64.671** (best of 133 greedy evals over 13 runs; `score.py --best runs/zlr1b4/*/`), hand-on
  **ckpt_4032299008.pt** (zlr1b4_e13, 2.1M before the eval; pulled with E13's ckpt_latest). Box
  58559819 (RTX 3090, 0.20 $/h, ~11.2 h ~ $2.3) released.

## zlr2b3 (round 2, slot 3) - 2026-11-01 05:21 UTC start

Box 58621513: RTX 5060 Ti + Threadripper 3970X (64 threads), 0.189 $/h, Germany (rented 05:28,
deployed 05:30 with the incumbent ckpt, md5 036960d8).

### Where the incumbent fails (ANALYSIS ONLY for the record; CLAUDE.md section 0)
* The incumbent's greedy (zlr1b4_e13 at 4,034M, 9/9 alike) flies ONE big circuit round the third
  bowl, period ~15 s: north along the bowl's WEST bottom (x ~-2,300, z ~-1,300, 2,300-2,900 u/s),
  up and round the north end (y ~3,000-3,300), south along the EAST side HIGH (x 0..+800, z
  400-700: the 60-64.7% swing tops), west and down round the south end (y -300..-900) to the west
  bottom again. E rises ~1,950 -> ~4,500 over 100-175 s (the energy potential pays every gain);
  the circuit widens until it flies out east/north-east and falls (171-187 s).
* The record flies THE SAME circuit at E ~1,900-2,100 (t+270..306). At t+305.9 it is at the west
  bottom (-2,271, 973, -1,277) going north at 2,300 u/s - exactly where the policy passes every
  ~15 s (t 100: (-2,309, 1,211, -1,332) v (-167, 2,287, -196)) - and instead of carrying on north it
  curves EAST up the bowl's floor (vz 72 -> 1,556 in 0.3 s), hits the vertical WEST FACE of a solid
  block in the middle of the bowl (x -1,616; at y 1,500 the block spans x -1,580..-1,150 from the
  bottom up to z ~600, its top a ramp rising east to z ~990) at (-1,635, 1,539, -569) with v
  (1,391, 267, 1,396), loses its horizontal speed there (E 1,903 -> 747), rises straight up the
  face ballistically and lands on the block's top at z ~620 (t+308, 60.9%), then pumps that top
  structure from ~100 u/s (the fourth structure).
* So the missing skill is ONE turn: at the west bottom, turn right (east) into the block instead
  of north. Both reward terms argue against it locally: the ratchet pays nothing for landing at
  60.9% (the episode already reached ~64% on the circuit's east side) and the per-call energy
  potential charges the impact (~1,250 u x 12/1000 = -15, more at the policy's speeds); the
  payoff (pumping the top structure above 64.7%) is ~70 s away, past the 20 s horizon.
  The policy is near-deterministic there (yaw sigma 0.04-0.05): it never samples the turn.

### EZ1 `zlr2b3_ez1` (05:33 UTC): incumbent weights + ez-greedy `--ez-eps 0.01`
Resume of the incumbent ckpt (4,032.3M; everything restored, the GE-line spine re-pointed at
autoresearch/spines/zlr1b4_ge1_line_a.npy, `--demo-tau0 114`) + temporally-extended
epsilon-greedy (Dabney et al. 2021: with prob 0.01 per decision an env holds ONE uniform random
action - yaw offset in the velocity frame and keys - for n ~ zeta(2) decisions, cap 60 = 1.7 s;
burst rows are excluded from the PPO loss). The paper's constants, nothing read off the map. Why:
the turn is a COMMITTED deviation of ~0.5 s; i.i.d. per-decision noise never produces it, a held
burst can. Read it on: does any training episode reach the block's top / fourth structure (the
reservoir's deepest, the frontier perturb best), then the greedy evals vs the incumbent's
progress.csv at matched steps (E13: 63.2-64.7% at 4.0-4.7B).
* **EZ1 = NEGATIVE, stopped at ~4,165M (05:47 UTC, ~130M steps).** Evals: 64.1% (4,032.8M, the
  resumed weights) / **54.07** (4,108M: 9/9 die at 57-96 s, several off the WEST side at the west
  bottom). Training: yaw sigma 0.044 -> 0.077, the frontier band 63.6% -> 59.3% (the reservoir's
  deepest receding), stall kills ~17%. Why: the burst rows are dropped from the PPO loss but the
  GAE returns of the ON-policy rows before a burst include the burst's outcome (often a fall), so
  those actions get noise advantages - the policy is pushed off its own line (sigma inflates).
  ez-greedy as implemented here is not sound for an on-policy learner at eps 0.01; the trace would
  have to be cut at the burst start (bootstrap V there). Pulled; ckpt_4132438016 not handed on.

### The turn, measured (ANALYSIS of the record + a diagnostic on the policy; nothing enters training)
* The incumbent makes the SAME right turn up the bowl's curved floor at the west bottom as the
  record - side key held right, view on the velocity - but ~600 u further NORTH: its heading
  passes 88 deg at y ~1,476 and 12 deg at y ~2,340 (t 94.9-95.5 s), so it rises past the block's
  NORTH end (the block spans y ~400..2,000) into open air and carries on round the circuit. The
  record bottoms out at y ~973 and is heading east at y ~1,567, into the block's west face. In the
  velocity frame the record holds its view ~1.5 deg RIGHT of the velocity through the curve
  (yaw - v-heading -1.2..-1.7), the policy 0..+0.3 deg; their approaches differ too (heading 116 vs
  131 deg at the same point, the policy's south loop is wider). So the gap is a held ~1-2 deg view
  offset / a tighter approach, not a big discrete manoeuvre - but between the policy's line and the
  record's lies the block's north end (a crash zone), which the policy has learned to clear.
* Diagnostic 1 (record_ckpt --spawn-states from the policy's own greedy states at the west bottom,
  the velocity ROTATED 15-75 deg east, 20 s greedy): 0/17 reach the block top - an instant velocity
  rotation slams the player into the curved floor (E 2,001 -> 1,449 / 883 / 286 within 0.1 s at
  30 / 45 / 60 deg). A bad proxy: the record turns by surfing the curvature (E ~conserved).
* Diagnostic 2 (running): a HELD yaw-command offset (record_ckpt --nudge-yaw -1..-8 deg, --nudge-hold
  15 / 35 decisions) from the policy's states 2.0 / 1.4 / 0.8 s before the turn.
* Control `zlr2b3_ctl` (05:49 UTC): the incumbent's exact config continued on this box (same-card
  baseline; every eval a chance above 64.671).
* Diagnostic 2 result: 0/30 - held yaw offsets of -1..-8 deg for 0.4-1.0 s from 2.0 / 1.4 / 0.8 s
  before the turn never reach the block top; nudges from 2.0 / 1.4 s are corrected (the circuit
  resumes), and from 0.8 s most kill it within 5-6 s at the block's NORTH-WEST corner
  (-1,500..-1,780, 1,900-2,200) - the crash zone between the policy's line and the record's.
* **Control `zlr2b3_ctl` first eval (4,032.8M, the incumbent's weights + one PPO iteration) =
  65.040** (median 64.6%, 9/9 alive to 186-198 s; ep0 pumped to E 4,970 and its swing top reached
  z 734). Eval variance of the same circuit on this card (RTX 5060 Ti; the lidar march is not
  bit-identical across GPU architectures), not a mechanism: in the circuit regime the progress is
  the swing-top height, which rises with the energy, i.e. with how long the episode survives.
  Hand-on: ckpt_4032299008.pt (= the incumbent ckpt, md5 036960d8, copied into the run dir on the
  box and pulled as the snapshot at or before the eval). Next evals 63.0 / 63.1 (4,108 / 4,184M).
* (tool) `tools/beam_tas.py --dump-paths NPZ --dump-every K`: every env's position and live flag
  every K ticks in population mode (off = byte-identical), so a sweep can be read for WHERE its
  lineages went. 1,024 envs x 30 s from a start state in 23 s on this box (193k env-steps/s).
  Sweep 1 (one start state 0.7 s before... ±6 deg held 0.5 s, redrawn for 5 s): 0/1,024 on the block,
  no new cell above z 500, the swing tops LOWER (z <= 408: the offsets cost energy).
* Control `zlr2b3_ctl` later evals: 63.0 / 63.1 / 62.9 / 63.3 (4,108-4,335M) - on this card the
  plain continuation sits ~63 (E13's level was ~64 on its 3090). Stopped 06:24 at ~4.35B.
* Sweep 2 (beam_tas --branch-at --branch-view-only, ONE held offset U(-10,10) deg for 1/2/3 s,
  forked from 21 of the policy's own greedy states over one 5 s circuit; 23 of 63 runs read,
  ~23,500 lineages): 0 on the block. The circuit absorbs held view offsets.

### GE from the policy's own greedy states with POSITION-ONLY cells FINDS THE BLOCK TOP (70.69%)
* `explore_phase1 --roots-spine <the incumbent's greedy recording, ticks 8,500..27,150 every 10 =
  1,865 roots, its last 130 s> --roots-keep-all --cell 128 --cell-energy 0 --cell-heading 0
  --exact-chains --view-follow=-90,...,90 --act-every 4 --decisions 100 --maxvel 3500 --tick-ms 7
  --envs 2048 --threads 24` (06:13 UTC, 15 min on the box CPU beside training): within 4 min the
  archive held 187 cells on the block top and its best cell at (-1,152, 1,371, 1,097) = **70.69%**
  on the grader's field (d 1,865; the circuit's best is ~65%). Round 1's explorer runs rooted in
  the third bowl (zl3/zl4/zl5) keyed cells on ENERGY (64-128 u x energy 32-64 u): 2-6M cells, each
  visited a few times, and their uniform bursts cannot pump at 2,000+ u/s, so the energy keys
  were unreachable noise; a position-only key (~77k cells) concentrates the search on new PLACES.
* The chain (ge_replay exact, 7/7 links, 703 ticks = 4.9 s): root = the greedy's state at t
  140.9 s at the circuit's NORTH end (-1,714, 2,988, -668), v (1,381, 1,444, 1,477), E 3,190.
  A burst runs it into the north-end wall: its northward speed is killed (E 3,190 -> 1,565 in
  0.24 s) and it leaves rising, heading SOUTH (v (-251, -1,153, 1,281)); a ballistic arc over the
  bowl's west part to z ~690 at y ~1,200, it lands on the block top from the north-west at
  (-1,458, 1,086, 764) (+2.0 s) and the bursts walk it east up the block's top ramp to z 1,097
  (+4.9 s). A different entry from the record's (which comes up the west bottom into the block's
  west face): the search found its own.
* Spine `autoresearch/spines/zlr2b3_ge1_s70.npy` (md5 5356e48b, tools/ge_spine.py): the greedy
  recording's ticks 0..20,129 + the chain (view canonicalised) = 20,833 states from the true start
  to d 1,864.8 (70.69%). Its success line (end + 100 u = 69.1%) is above anything the circuit
  reaches (~65%), so the d-rule cannot be satisfied without the new route, and from the launch to
  the line is ~5 s (inside the 20 s horizon: the +50 can outweigh the energy charge of the wall
  hit, -1,625 u x 12/1000 = -19.5).

### S70 `zlr2b3_s70` (06:25 UTC): incumbent weights + the backward curriculum on the s70 spine
Resume of the incumbent ckpt + `--demo-file autoresearch/spines/zlr2b3_ge1_s70.npy` (fresh
curriculum from the spine's end) with round 1's E2 constants (`--demo-window 5 --demo-grow 5
--demo-grow-max 40 --demo-cooldown 5 --demo-rate 0.2 --demo-min-ep 50 --demo-front-frac 0.5
--demo-success-d 100 --demo-success-bonus 50`), the old GE line's 9.48% line kept as
`--demo-milestones 5859`, and `--demo-front-temp 1 --demo-front-ticks 300` (keys at T 2 and yaw
sigma x2 for the first 2.1 s of a band spawn: exploration where the band is). Everything else
restored (per-call energy K 12, fail-pen 3, energy-stall 32, respawn-greedy, front-perturb).
* GE stage 2 (06:27, box CPU, 18 min): rooted at the spine's last 450 rows (block landing + top)
  with the pb config (64 u x energy-32 x heading cells, --select-energy 3): can uniform macros pump
  the top structure and climb out (the record pumps it ~70 s from ~100 u/s, ANALYSIS ONLY)?
* **S70 06:25-06:45 UTC (4.03-4.22B):** the curriculum walked from the spine's end (20,832) past
  the launch (row 20,130) to ~19,830 (1.8 s before the north-wall hit) in ~180M steps - 28.6% band
  success there, 8,144 wins over 10,907 spine episodes. The policy learned the block top AND what
  lies past it on its own: the reservoir's deepest state is at **91.3%** (min depth 8.1%; the
  frontier temperature band now >91%), 39 states >= 90% and 140 in [85, 90) at z ~2,340-2,460
  around (-1,500..+700, -50..2,180), E ~3,100-3,600 - in the open air between the platforms at
  z ~1,630-1,820 and the finish-level structure (solid z 2,814-3,006, the slabs on top at
  3,008-3,210). (The record, ANALYSIS ONLY, passes that region at 78-90%, t+383..412.) GE stage 2
  (pb energy cells from the block top) filled its 4M cells in 17 min with best 70.9%: the policy's
  own pumping went where uniform macros could not.
* The greedy from the true start has not taken the launch yet: 64.4 (4,032.8M) / 62.6 / 61.4
  (4,184M); the circuit is in flux (E max 5,100 at 4,108M, ~2,400 at 4,184M; one episode died at
  65 s). Box extended to 09:43 UTC.
* **S70 eval at 4,259M = 77.956 - NEW BEST** (`score.py`; median 62.2%). 1/9 greedy episodes from
  the TRUE START took the new route: in the circuit's north end at ~118 s it reverses WEST off the
  wall (v (1,102, -1,175, 1,307) -> (-1,586, 264, 909)), rises south-west to (-2,371, 1,718, 427),
  turns south-east onto the block's top ramp at (-1,306, 844, 912) and launches off its east end
  (vz 1,003) to (0, 1,343, 1,542) = 77.96%, then falls east (dead at 125 s, (2,635, -119)). The
  other 8 make the same north-end reversal and fall at (-2,400, 2,100-2,600) or off the block's
  west edge at ~117-128 s (~62%). So the policy's own launch differs from the GE chain's (it
  learned its own version of the wall reversal) and is still unreliable. Evals after: 59.6 / 59.9
  (4,335 / 4,410M) - the level moves while the curriculum walks (band ~17,100 = ~25 s before the
  spine's launch at 4.45B). Hand-on: ckpt_4232577024.pt (26.7M before the eval; pulled).
* GE stage 3 (rooted at 452 of S70's own reservoir states at 80-91.6%, position-only 128 u cells,
  15 min): 97,751 cells, 0 finishes, best 92.37% at (-386, -1,294, 2,479) hovering at z ~2,475
  with E ~2,960. The finish triggers (z 3,008-3,210) sit ON TOP of the upper structure (solid
  z 2,814-3,006 under their footprints): a finish is a landing on that structure, and uniform
  bursts top out just under the needed energy. The policy carries E 3,100-3,600 up there.
* Policy beams (07:12): beam_tas on S70's 4,432M weights from its own reservoir states >= 85%,
  ranked by d, 1,024 envs, 42 s each - can its own proposals land on the finish structure?
* **S70 COLLAPSED after 4,485M** (stopped 07:31 at ~4.74B): evals 60.8 (4,485M: 2/9 already dead
  at ~50 s in the second trough, 12.5%) -> **0.54 / 0.4 / 0.3** (4,561 / 4,637 / 4,712M: all 9
  wander the start platform at <= 470 u/s until the 30 s patience cut). Training: crawl kills
  0.1 -> 0.4, stall kills 0.25, episodes ~1,000 ticks, entropy up. **Cause: catastrophic
  forgetting of the early route.** The anchored-and-widening curriculum draws its spine starts from
  [tau, n-1]; on a 145 s spine with tau still at ~115 s (16,500 of 20,833) every spine start is in
  the last 30 s, and the demo replaces the whole reservoir share - so the first ~115 s of the route
  were practised only by the 10% true-start spawns. (Round 1's E2 used a 27.7 s spine, which the
  range covered entirely as tau walked back.) A long spine needs the early route kept in the pool.
* Policy beams (S70's 4,432M weights, beam_tas --score d, 1,024 envs, 42 s, from 8 of its own
  reservoir states at 75-85%): frontiers d 945-1,011 (85.1%), 0 finishes.

### S70b `zlr2b3_s70b` (07:33 UTC): S70 from its 77.96%-eval snapshot + the early route in the pool
Resume of ckpt_4232577024 (S70's hand-on; md5 f7aba3ce, staged as runs_ckpt.pt on the box; the
incumbent kept as runs_ckpt_incumbent.pt) with S70's curriculum flags at `--demo-tau0 19500` (where
its band was at 4.23B), WITHOUT `--demo-front-temp`, plus `--spawn-states
autoresearch/spines/zlr2b3_inc_early_route.npy --spawn-states-frac 0.3` (the incumbent's own greedy
recording, true start -> t 140.9 s, every 5th tick: 30% of every pool is uniform over the early
route the curriculum's range no longer covers). Generic: a long spine's backward curriculum keeps
the part of the route it has walked past in the pool; nothing reads the map.
* S70b 07:33-07:56 (4.23-4.47B): no collapse (the early route holds: the start, the first loops,
  the third bowl), but 0 launches in 4 evals: 61.6 / 59.8 / 59.8 / 60.8. The greedy's circuit now
  carries only E ~1,900-2,030 (the incumbent's ~4,500) and at ~120-156 s tries the north-end
  reversal and falls off the WEST edge (-3,120, 2,200-2,400) or the east (2,944, -400); 1/9 fell at
  the start (8.9 s). The spine's launch (the incumbent's line at t 140.9 s) had E 3,190; the
  policy's own circuit no longer matches it. Training: reservoir deepest 91.5%, band ~16,465
  (t 115 s of the spine), 47% spine-episode success.
* Beam from the TRUE start on S70b's newest weights (07:58, beam_tas --score d, 2,048 envs, 16
  greedy, 210 s: round 1's auto-spine beam by hand): a spine through the policy's OWN launch.
* Beam from the true start on S70b (4.47B weights; 2,048 envs, --score d): population extinct at
  139 s, frontier 60.0% at 106 s - no lineage of the policy's own proposals makes the launch from
  the true start. S70b eval 4,535M: 58.4%. So the curriculum teaches the launch from the
  INCUMBENT's circuit states (E 2,500-3,190 at t 110-141 s of the spine) while the policy's own
  circuit no longer passes through them (E ~2,000): band success 21% from spine states, 0 from its
  own line.
* Plan B (08:12): record S70b's CURRENT greedy line (record_ckpt --dump-states) and rerun the stage-1
  Go-Explore (position-only 128 u cells) rooted at ITS states (t 60 s -> death), 10 min: a launch
  that starts from the circuit the policy actually flies, for a spine whose band matches it.
* GE4 (stage-1 Go-Explore rooted at S70b's CURRENT greedy line, 730 roots from t 52 s to its death
  at 103 s, 10 min): 55,601 cells, best = a root (57%) - NO launch from the circuit the policy now
  flies (E ~2,000), against 4 min from the incumbent's (E 3,190). S70b final: 61.8 / 60.0 / 60.1 /
  60.8 / 58.6 / 58.8 / 57.0 / 55.7 (4.23-4.76B) - a steady decline, 0/72 greedy launches; stopped
  08:25. The curriculum keeps the policy launching from the INCUMBENT's states while its own circuit
  loses the energy the launch needs; the greedy then tries the reversal at every north-end pass and
  falls.
* Box redeployed on ar/zlr2b3 b266e9ad (08:26; the base had no --energy-pot-loss).

### S70c `zlr2b3_s70c` (08:27 UTC): S70b's exact config + `--energy-pot-loss 0`
From the same snapshot (S70's ckpt_4232577024), one change: energy losses are no longer charged
(gains still paid whole). The launch is a deliberate speed-for-height trade - the north-wall hit
costs E 3,190 -> 1,565 = -19.5 reward on one call under the full potential, more than the block
top's ratchet pays - so under the full potential every exploratory reversal from the policy's own
states is punished before its payoff. Generic (the flag reads nothing on the map; 0 = gains
only). Read against S70b at matched steps: greedy launches and the eval level.
* **S70c = NEGATIVE** (08:27-08:52, stopped at ~4.49B): 61.9 / 57.5 / 58.7 / 53.3 vs S70b's 61.8 /
  60.0 / 60.1 / 60.8 at matched steps, 0 launches. With losses free the policy bleeds energy and
  the --energy-stall rule (re-arm only on a NEW energy record) then cannot re-arm: training stall
  kills 5-8% -> **74%**, and 5-7 of 9 greedy episodes per eval end by the patience cut. A loss-free
  energy term needs a stall rule that does not read the episode's energy record; not pursued.

### S70d `zlr2b3_s70d` (08:53 UTC): S70 from the INCUMBENT again, with the early route in the pool
S70's one greedy launch came ~230M steps into the curriculum from the incumbent (4,259M), while
the incumbent's circuit (E ~4,500) was still intact; afterwards the circuit lost its energy and
the launch never came back (S70 later, S70b, S70c: 0/~180 greedy episodes). S70d restarts from the
incumbent ckpt with S70's exact flags (incl. --demo-front-temp 1 --demo-front-ticks 300) + the
early-route spawns (30%) from the start, denser evals (--record-every 37.5e6: twice the greedy
evals in the window where the launch appeared) and snapshots every 5e7 (a hand-on near any good
eval).
* **S70d WORKS (08:53-09:20 UTC): 64.9 / 63.2 / 63.5 / 63.8 / 75.8 (4,184M) / 78.789 (4,222M,
  median 78.1%: 7/9 launches) / 78.858 (4,259M, median 77.3%) - NEW BEST.** With the early route in
  the pool from the start and the incumbent's circuit intact, the policy learned the launch in
  ~150M steps and took it reliably from the true start. Every launching episode now ends the same
  way: off the block-top ramp east to ~78% at z ~1,570-1,610 (x ~-500..0), then a fall to the east
  (2,000-2,300, 1,600-3,300). Training: reservoir deepest 93.8%, band ~19,177 (7 s before the
  spine's launch), 45% band success. Hand-on: ckpt_4233625600.pt (25.7M before the eval, pulled).
* Beam from the TRUE start on S70d's 4,284M snapshot (09:22; 2,048 envs, --score d, 238 s): the
  auto-spine step by hand - a line past the greedy's 78% (into the upper loop) for the next spine.
* **S70d 4,297-4,373M: 77.227 / 89.519 (4,335M) / 80.648 - NEW BEST 89.519.** At 4,335M 2/9
  launched: ep3 80.4%, ep8 **89.52%** (z 2,291, 141.2 s): north-end launch at ~130 s, the block at
  137 s, the ramp launch (-1,043, 1,148, 1,190) at 138 s, then INTO THE UPPER LOOP (x ~0..300,
  y 300-1,400, z 1,700-2,270; the record's fifth structure, ANALYSIS ONLY) to its swing top at
  141 s, then out east (717, 1,266, 1,706) and a fall. Launch reliability swings between evals
  (7/9 at 4,222M, 2/9 at 4,335M). Hand-on: ckpt_4334288896.pt (0.5M before the eval, pulled).
* Beam from the true start on S70d's 4,284M weights: the population collapsed into one stuck
  place (all 2,048 lineages at d 5,046 = 20.7% for 80 generations) - killed. (A d-ranked beam
  over a 200 s line on this map keeps whatever lineage is momentarily deepest; a lineage parked
  on a ledge at 20% outlived every flying one.)
* **S70d 4,410-4,675M: 89.8 / 90.0 / 91.691 (4,486M, median 90.8%) / 91.729 (4,524M, median
  90.9%) / 90.9 / 89.4 / 90.8 / 89.8 (mean of the 9 episodes' best up to 87%) - NEW BEST 91.729.**
  Most greedy episodes from the true start now take the whole new route to the upper loop's first
  swing top (z 2,350-2,430, ~91%). All of them then fall out of it within ~5 s (E 2,800-3,100 at
  the top; the record, ANALYSIS ONLY, pumps that loop for 65 s to E ~4,000 and launches north onto
  the top platform). One of 9 at 4,524M died at the start (13 s). Hand-on: ckpt_4485283840.pt (the
  newest at or before the 4,524M eval, 0.5M before the 91.69 eval; pulled). Stopped 09:59 at
  4,679M (ckpt_latest pulled, md5 5b7eea3f).

### S70e `zlr2b3_s70e` (10:00 UTC): S70d continued + practice in the upper loop
Resume of S70d's ckpt_latest (4,678.7M) with S70d's flags at `--demo-tau0 14900` (its band), plus:
* `--spawn-states autoresearch/spines/zlr2b3_early_plus_upper.npy --spawn-states-frac 0.35`: the
  early route as before + S70d's OWN reservoir states at 75-92% (1,609 from its 4,384M checkpoint)
  - spawns where the greedy now arrives and falls out;
* `--respawn-greedy-contact`: the greedy-failure rows anchored at the failed episode's LAST MAP
  CONTACT instead of its last tick (its deaths are ballistic falls after leaving the loop's
  surface, so the old 0.25-2 s window sat mid-fall).
Both generic (the policy's own states; a rule about ballistic deaths). Read it on: time alive in
the upper loop, its E, swing tops above 92%, a finish.
* S70e 10:00-10:23 (4.68-4.91B): 89.9 / 90.1 / 89.3 / 90.3 / 89.7 / 89.6 / 90.0 (frontiers 5,682-
  5,743 u; means of the 9 episodes' best 66-89%): the level holds (most episodes reach the upper
  loop's first swing top) but does not move; reservoir deepest 91.8%. The upper loop (ANALYSIS of
  the geometry): at x 0 a solid slab z 1,630-1,950 over y ~-500..1,100, the finish-level blocks
  z 2,782-2,974 above it (y -750..16 and 528..1,296); the policy's swing top sits ~450 u above the
  slab and it falls off the slab's south edge. Without a finish the loop's tops cap at ~92-93%.
* Upper-loop beams (10:25): S70e's newest snapshot, --score dh --score-energy 2 (lineages that keep
  and pump energy survive selection), 1,024 envs, 63 s, from its own reservoir states >= 85%.
* **Upper-loop beams (S70e 4,880M weights, --score dh --score-energy 2, from its own reservoir
  states >= 85%):** 4/6 populations extinct within 4-8 s; 2/6 survived the whole 63 s; **bu_5
  (root at 90%, (180, 580, 2,359)) swung the upper loop for 12 s pumping E 2,674 -> 3,150 and reached
  d 351 = 94.49%** at (534, 491, 2,588) (replay exact). Ranked on depth + energy, the policy's own
  proposals DO stay in and pump the loop. S70e final: 90.2 / 90.3 / 89.5 / 90.5 / 89.9 / 89.8 /
  90.2 / 90.8 / 90.0 / 91.2 (4.68-5.02B); stopped 10:43 at 5,061M (ckpt_latest pulled).

### S70f `zlr2b3_s70f` (10:44 UTC): the spine extended into the upper loop
Spine `autoresearch/spines/zlr2b3_s70_bu5.npy` (md5 5c25bdff) = the s70 spine (true start -> block
top, 20,833 rows) + bu_5's line (1,700 rows: the upper loop at 90% -> 94.49%), cut at its best d:
22,506 rows, end d 350.7. The two pieces do not join (the gap - the ramp launch into the upper loop
- is what the greedy already flies by itself); the curriculum restarts at the new end. Success line
d <= 450.7 (92.9%), above the loop's first swing top (~91%): only pumping the loop reaches it. The
old lines keep paying: `--demo-milestones 5859,1965` (the GE line's 9.48% and the block top's
69.1%). Spawn file `zlr2b3_early_upper2.npy` (md5 4bccafdf): the early route + S70e's own reservoir
states at 75-93.6% (2,586) + bu_5's line, 35% of the pool. Else S70e's flags. Resumed from S70e's
ckpt_latest 5,061,476,352.
* Finish beams (S70e 4,880M weights from bu_5's own rows 1,100 / 1,350 / 1,600 / 1,699, --score d
  and dh, 1,024 envs, 63 s): no finish; no frontier past the start state (d 351-460); from the
  line's last rows (94%, E 3,150) every lineage falls within 4-6 s. bu_5's own population "survived"
  63 s only by falling onto LOWER structures (d 5,944 = 6.6% at 35 s, 5,072 at 60 s): ranked on
  d + 2 x energy height, a fall keeps its energy, so dh does not select survival in a loop.
* S70f 10:44-10:58 (5.06-5.21B): 90.6 / 90.5 / 89.4 / 90.6 / 91.07 (frontiers 5,690-5,794);
  band in the bu_5 segment at row 21,715 (882 of 1,700) with 32% success.
* (ops) Box 58621513 extended at 11:09 to 16:38:54 UTC - the wrap-up (16:40) must pull and release
  before then (or extend by a few minutes first).
* **S70f at 5,251M = 92.433 - NEW BEST** (median 89.8%): **9/9** greedy episodes from the true start
  launched and reached the upper loop (79.9-92.4%); ep2's swing top 92.43% at z 2,509 (123 s); all
  die at 116-130 s out of the loop. Hand-on ckpt_5212471296.pt (38M before; pulled). Then 90.7 /
  90.6 / 90.0 / 89.7 / 89.5 / 90.7 / 91.0 / 90.5 / 91.1 (5.29-5.59B); band 21,440 (row 607 of the
  1,700-row pumping segment) at 17% success; training reservoir deepest 94.65%.
* **S70f 5.63-5.85B: 91.6 / 90.6 / 91.4 / 92.8 / 93.2 / 96.944 (5,817M, median 95.1%) / 96.376
  (5,855M, median 96.2%) - NEW BEST 96.944.** The greedy now PUMPS the upper loop (E 3,190 ->
  3,466 over 148-160 s, swing tops 91.6-92.4%) and launches off its north end: ep0 to (-367, 1,669,
  2,764) = 96.94% with v (596, 877, -13) - its apex ~240 u under the finish platform (the slab
  footprints' solid tops at z 3,006; this one x -320..-64, y 1,312..3,360) and 47 u west of it; then
  a fall east. Energy-limited: E 3,466 at the launch (the record, ANALYSIS ONLY, launched there with
  E ~4,090). Training reservoir deepest 98.06%; curriculum band at row 21,245 (68% success).
  Hand-on ckpt_5816451072.pt (0.5M before the eval; pulled).
* GE stage 5 (12:08): rooted at S70f's own reservoir states >= 95%, position-only 64 u cells,
  --goals 3 (stops after 3 finishes), 12 min: a short chain onto a finish slab -> a spine to the finish.
