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

## zlr2b1 (round 2, slot 1) - 2026-11-01 05:20 UTC start

Box 58621920: RTX 5080, 0.337 $/h, Quebec CA (machine 153623), 96 threads (quota 92 CPUs), rented
05:37. 249-271k steps/s on the incumbent's resume. (The first rent, 58621552, a 3090 on machine
153342, was power-capped at 260 W: gpu_health 89% of reference -> blocklisted, released.)

### The incumbent's 64.67% is BALLISTIC: free-flight apexes the map never supports (05:45 UTC)
Analysis of `incumbent/r1/traj_4034396160.jsonl` (score 64.671), ep6; contact = a tick whose vz
change departs from free fall (air strafing never changes vz) or onground:
* after 52 s it touches EXACTLY ONE surface, every ~5 s for ~1 s: the crest of the west ramp of
  the third structure (x ~-1,965..-2,000, z ~-930..-1,000, sliding along y 300 -> 2,800). Contact
  progress there 32.6-33.1% (best ever at contact 38.89%, at 53 s). Everything else is flight: a
  giant ballistic loop (west ramp crest -> north-east up -> apex over open air east of x -400 at
  z ~600-710 -> south -> back to the crest), pumping E 1,261 -> 4,500 (~33 u/s) at 2,000-2,900
  u/s. The apex height rises with E, and the free-flight field reads apex height as progress:
  that is the whole 64.7% and the slow creep of E13 (62.8 -> 64.7 over 2.2B steps).
* The occupancy (`maps/surf_za_loop.occ_32.npz`, axes [z, y, x]) at y 1,539: the west ramp is a
  320 u wide pillar whose top slopes UP eastward (~50 deg, z -1,404 at x -2,276 -> -1,020 at
  -1,956, the crest), then a void (x -1,956..-1,636, down to the bottom teleport), then the RIDGE
  (x -1,636..-1,188, y 430..1,950): vertical west face up to z ~560-605, top sloping up east to
  ~950-1,150, vertical east face. Above z ~1,600 a huge block (x -2,900..250, y -470..1,000).
* The record (ANALYSIS ONLY): at t+306 it rides UP the west ramp's slope eastward (not along the
  crest), leaves the crest at ~45 deg (v (1,391, 267, 1,396) at (-1,635, 1,539, -569): vz ~1,620 at
  crest height), flies across the void into the ridge's west face, wall-slides up ~600 u
  (v (0, 104, 996)) and pops onto the ridge top at z ~620 (60.8%), then surfs the ridge's top
  along y up to z ~1,500 (76% by t+370) and launches east past the big block (82% at t+384).
* **The record's contact-only progress tracks its all-state progress within 0.1-2.6 points the
  whole way; the incumbent's gap is 26 points (64.7% vs 38.9%).** On the field the true exit (the
  ridge top, 60.9%) reads LOWER than the policy's apexes (64.5%), so the ratchet pays nothing for
  it and the energy term charges the wall impact (~-15): the objective says the dead end is ahead.

### NEW `--ratchet-contact 1` (commit 0bb26ad1, tests/python/test_ratchet_contact.py)
The ratchet PAYS a second record that moves only on supported ticks (walkable ground, or the
--surf-bonus contact test: airborne and vz gave back > 1 u/s more than free fall; the
-sv_maxvelocity clamp excluded; a vertical wall or a ceiling never counts). A landing pays
everything flown since the last contact (the teleport clip scales with the flight:
max_step x (calls since the last support + 1), the control's clip on a surface). The (d - b)/d0
observation column keeps the ALL-STATE record, so the policy's input is the control's exactly
(no recorder mirror needed: TRAIN_ONLY). Off = bit-identical to the base commit (the test runs
`git show 401f4007:python/surfgym/rewards.py` side by side over 600 random calls). Generic: no
constant reads a map; the record a policy is paid for becomes the furthest SURFACE it reached.
Step line: `rc <mean>/<max> all <mean>/<max> paid <share>` (contact vs all-state record per ended
training episode; the max includes reservoir spawns, whose record starts at the spawn's d).

### c1 `zlr2b1_c1` (05:39-05:55 UTC): the control - a plain resume of the incumbent
`--resume` (every flag restored) + `--demo-file autoresearch/spines/zlr1b4_ge1_line_a.npy
--demo-tau0 114 --ckpt-every 2e8` (the curriculum's tau is NOT in a checkpoint; the incumbent's
spine path `runs/ge/zl1/line_a.npy` does not exist on a new box). Evals **64.417** (its own first,
4,033M) / **63.050** / **63.289** (4,184M): the level, on this card. Stopped at 4,240M.

### rc1 `zlr2b1_rc1` (05:56 UTC): the incumbent + `--ratchet-contact 1` (warm, measuring AT the wall)
c1's flags + `--ratchet-contact 1`. Evals **64.719** (its own first at 4,032.8M = the incumbent's
weights after 0.5M steps - eval noise, but the round's best so far; hand-on ckpt_4032823296.pt,
pulled) / **60.672** (4,108M: the apex credit is gone, swing tops z ~450 instead of ~700, dying
at 146-155 s). Training: reward ~53/ep (c1 ~115), contact record 38-46% mean per episode against
48-54% all-state.
* rc1 later: **61.211** (4,184M) / **62.316** (4,259M), recovering; every greedy episode's best
  CONTACT stays 37.8-39.1% (the crest): the flag alone did not find the ridge in 320M steps.
  Stopped at 4,354M (ckpt_latest md5 7e551f2e, pulled) - the warm start of rc2.

### Go-Explore from the policy's OWN crest contacts FINDS THE RIDGE TOP (06:05-06:13 UTC, box CPU)
* Roots: `tools/contact_roots.py` (NEW) on `record_ckpt --dump-states` of the incumbent (3 greedy
  episodes; ep2 idled at the start and is skipped by `--min-record-frac 0.9`): every 8th contact
  tick after the episode's contact record last improved = 1,060 roots, all on the west ramp crest
  (progress 28-38.6%). Generic: the rule is the policy's own contact history, no place named.
* `tools/explore_phase1.py --roots-spine contact_roots.npy --roots-keep-all --exact-chains --cell
  128 --cell-energy 64 --cell-heading 1 --view-follow=-90,-60,-30,-15,0,15,30,60,90 --select-energy
  3 --act-every 4 --decisions 100 --maxvel 3500 --tick-ms 7 --envs 4096 --threads 40 --max-minutes
  15` beside the trainer (0.66-0.85M ticks/s): 3.29M cells at 7 min (iteration 200).
* `tools/ge_contact_pick.py` (NEW: the deepest archive cell whose state the map holds - one null
  tick from it shows ground or a surface pushing vz up): of the 600k deepest cells (mostly
  apexes), **232 are contacts at 55-70%, ALL on the ridge top** (x -1,330..-1,150, z 890-1,050),
  deepest **69.69%** at (-1,213, 1,333, 1,023). The incumbent's best contact: 38.9%.
* `ge_replay.py --cell 3194239`: 9/9 links exact, 448 ticks (3.1 s): from a crest contact
  (-1,965, 2,730, -933) already climbing (v (1,366, 1,395, 1,581): the incumbent's own north-end
  exit), an impact near (-1,200, 2,900, -300) costs ~1,800 u of E, then a ballistic flight back
  south-west over the ridge's north end, an air-strafed left turn, a landing on the ridge's sloped
  top at z ~684 (2.0 s) and a surf UP it to z 1,023 with |v| 339 left. Not the record's
  wall-slide: the policy's own exit, turned.
* `tools/contact_spine.py` (NEW): the incumbent's own greedy episode 1 up to the root (index
  19,656 = 137.6 s) + that line, cut at the deepest contact tick = `autoresearch/spines/
  zlr2b1_ge1_spine.npy` (20,103 states = 140.7 s, end contact 69.66%, md5 0c95e87e).
* NEW `explore_phase1 --cell-contact` (archive only states the map holds; off = byte-identical,
  selftest 99/99, tests/python/test_cell_contact.py) for the next stage: the flight cells are what
  filled the archive (3.3M at 7 min).

### rc2 `zlr2b1_rc2` (06:17 UTC): rc1's weights + the ridge-top spine (backward curriculum)
`--resume` of rc1's 4,354M (`--ratchet-contact 1` restored) + `--demo-file autoresearch/spines/
zlr2b1_ge1_spine.npy --demo-milestones 5859 --ckpt-every 2e8`; no --demo-tau0, so the curriculum
starts at the spine's END and walks back (window 5, grow 5..40, cooldown 5, rate 0.2, front-frac
0.5, success-d 100, bonus 50 - all restored). Success line d <= 2,030 (68.1%) read on the CONTACT
record (paid_record: an apex past it never counts). The old GE line's 9.48% stays a milestone.
* **rc2 (06:17-06:23 UTC, 4,354-4,430M): 60.391 (its first) / 63.211.** Training read-out: episodes
  spawned at the spine's END (the ridge top) reach CONTACT progress up to 79.7% within 50M steps
  (reservoir deepest 79.1%) - the policy pumps the fourth structure by itself once it is there.
  But the greedy from the true start still never touches the ridge (best contact 38.5-39.1%), and
  the curriculum walked back only 25 states in ~100 iterations (ridge-top episodes run ~5,000
  ticks before they end, so the band's min_ep / cooldown pace is slow): the 447-state GE part
  alone would take ~15 min and the 19,656-state base is the policy's own line anyway. Stopped.
* The GE line's E drop (3,028 -> 1,193 in 0.5 s) is AIR-BRAKING, not a collision (no solid there):
  keys held against the motion decelerate ~3,000 u/s^2; the line reverses east -> south-west.
* Record (ANALYSIS ONLY), t+306..392 contacts: after the ridge landing it swings the ridge's length
  (y ~400 <-> ~1,960) mostly on walkable ground (onground), with 5-frame touches at the swing ends
  on surfaces that rise each lap (y > 2,000 and < 430: z 940 -> 1,580, 76% by t+370), then 82% at
  (140, 1,223, 1,864). Those surfaces are not in the 32 u occupancy slices (thin brushes fall
  between voxel centres): read the map with physics-based contact tests, not the occ grid.
* GE ge2 (06:21 UTC, 20 min, 40 threads): `--cell-contact --cell 64 --cell-energy 32`, rooted at
  the spine's last 161 states (the ridge top after the landing, stride 2 = 81 roots): 279k contact
  cells at 5.6 min, then stalled; nothing deeper than the roots (uniform bursts do not pump the
  ridge top).

### rc3 `zlr2b1_rc3` (06:23 UTC): rc2's weights + the band at the crest ROOT (`--demo-tau0 19656`)
The same spine and flags as rc2 plus `--demo-tau0 19656` (the root = the crest contact the GE line
starts from): demo draws half on [19,656..19,660], half uniform over the GE part [19,656..20,102];
the curriculum's own backoff moves the band later if the launch from the root fails.
* rc3 evals: **63.647** (its first = rc2's 4,430M weights) / **63.252** (4,506M); greedy best contact
  still the crest (38.2-38.8%), two episodes died early (13 s, 64 s). The band backed off 19,656 ->
  19,676 at 0% success: spawns past the success line live long on the ridge top (mean episode
  11,276 ticks = 79 s), so the curriculum's outcomes arrive from the FAST deaths first (flight
  states that miss the ridge). Slow feedback, not a bug (DemoCurriculum.note_outcomes counts an
  episode when it ENDS).
* ge2 stopped at 10.6 min (331k contact cells, nothing deeper than its roots).
* **rc3 BREAKS THE THIRD STRUCTURE (06:23-06:55 UTC): 63.647 / 63.252 / 62.088 / 60.693 / 61.450 /
  62.076 / 72.105 (4,884M, median 59.9%).** The curriculum backed off to ~19,690 at first, then
  walked back PAST the root (19,576 by 4,906M) at 95-97% band success: from its own crest approach
  the policy now launches onto the ridge top. At 4,884M 2/9 greedy episodes do it from the TRUE
  start: ep4 lands (contact 68.8% at 148 s) and climbs to 72.10% (z 1,183, 151.9 s), ep2 to 70.87%
  (contact 69.3%); five still fly the old loop (best 59.7-62.6%), two die early in the second
  trough (60 s, 84 s). Training: contact records up to 92.0% per episode, reservoir deepest 92%
  (8.0% of d0), unstuck T 0.20 -> 0.17 (the eval frontier moved). Hand-on: ckpt_4883742720.pt
  (snapcopy, the eval's own step; pulled, md5 8b85ad3f). The incumbent never cleared this gate.
* **rc3 07:21 UTC: 60.963 / 62.777 / 77.176 (5,110M, median 71.7%) / 61.657 / 62.264 / 59.033
  (5,337M).** At 5,110M 7/9 greedy episodes leave the third structure onto the ridge (67-77%);
  best 77.18% at z 1,506 (136 s; the record has 76% at z ~1,580 at t+370), all dead by 127-241 s.
  The next three evals fell back to the old loop (59-62%): the launch is a close call in the
  greedy and flips. Curriculum band at 19,346 (310 states before the root) at 93-96% success;
  training contact records up to 93.9%, reservoir deepest 94% (6.0% of d0). Hand-on for 77.176:
  ckpt_5110235136.pt (snapcopy, the eval's own step; pulled, md5 e3b26209).
* rc3 last eval **61.017** (5,412M): 6/9 fly off the third structure to the south-east at 102-107 s
  (E ~2,100 - the 5,110M launches came at ~120 s, E ~2,440), 2/9 die at ~63 s, 1 at the start. The
  spawn mix is the suspect: with the band at 19,326 the demo (90% of spawns) is the spine's last 777
  states, so the third structure's long pumping phase (~80-137 s) trains only on the 10% fresh
  starts and the greedy rows. rc3 done at 5,430.6M (ckpt_final md5 99a1395b, pulled).

### rc4 `zlr2b1_rc4` (07:26 UTC): rc3's final + the band moved back into the third structure
Same spine and flags, `--demo-tau0 12000` (84 s on the spine): uniform demo draws over [12,000,
20,102] cover the pumping, the launch and the ridge top; the curriculum's own rule moves the band
from there. Budget 2e9.
* **rc4 (07:26-07:50 UTC): 60.143 (its first) / 79.168 / 81.217 (5,582M, median 78.3%) / 77.242 /
  77.148 / 78.103 (5,809M, median 76.0%).** Moving the band back into the third structure fixed the
  flip-flop: the greedy now leaves for the ridge in most episodes of every eval. Curriculum walking
  back from 12,000 at 86-87% success (11,935 at 5,833M). Hand-on for 81.217: ckpt_5582094336.pt
  (snapcopy, the eval's own step; pulled, md5 3d4f5bd9).
* **The 76-81% is ballistic again.** The 81.2% eval's best episodes touch surfaces only up to
  68.8-71.0% (the ridge top, (-734, 708, 1,598) at best); 80-81% are apexes over the structure
  (z 1,700-1,780 at x ~-1,800, north-west of the big block), then a fall west. The grader scores
  them, so 81.217 stands, but the surface frontier is ~71%. The training read-out's contact max
  (~90%) is inflated the same way: front-perturb spawns at the reservoir's (all-state, apex)
  frontier start their contact record at the spawn's d. The record's next surfaces (ANALYSIS
  ONLY): east of the big block, x 0..300, z 1,700-2,200 (82-88%), reached by a fast launch east
  from the ridge top's low part at t+381.
* The 5,110M greedy re-recording (record_ckpt --dump-states, 9 episodes): its deepest contact line
  ends at 69.82% - no deeper than the GE spine; not used.
* rc4 later: 75.617 (5,884M) / 77.850 (5,960M, median 77.2%) - the ~76-81% level holds; curriculum
  at 11,865 (81% success).
* GE ge3 (07:58 UTC, 15 min, ge1's config): roots = the 5,582M policy's OWN contact ticks after its
  contact record stopped improving (record_ckpt --dump-states, 9 greedy episodes, 8 on the ridge
  with contact records 69.2-69.7%, apexes to 82.0%; contact_roots --stride 4 --min-record-frac 0.9
  = 462 roots at 60-70%, E ~2,500-2,900). Looking for the next SURFACE (the record's: east of the
  big block, 82-88%).
* ge3 snapshot (6 min, 2.48M cells): 4,614 contact cells, deepest **70.05%** (flat ground on the
  ridge top at (-1,151, 1,345, 1,052)); only 2 contact cells above 70%. Deepest cell overall an apex,
  81.91% at (-22, 2,588, 1,807) north of the big block.
* The next surface (ANALYSIS ONLY: core.trace point probes under the record's 82-88% contacts): the
  big block's EAST face is a surf ramp (normal (0.76, 0, 0.65), ~49 deg, facing east) at x ~175-253,
  z ~1,670-1,780 (y -950..1,370); the record flies UNDER the block's underside (z ~1,630, x
  -2,900..250, y -470..1,000) from the fourth structure at t+381 and lands on that east-facing
  slope, then pumps it along y up to z ~2,180 (87.75%). The policy's apexes go up NORTH of the block.
* GE ge3 final (15 min, 2.99M cells): still no contact past 70.05%. ge4 (NEW tools/ge_frontier_roots.py:
  roots = ge3's 2,000 deepest cells, apexes at z 1,725-1,807 north of the block) started 08:13 and
  was stopped at 08:16: the RL got there first.
* **rc4 FINDS THE NEXT STRUCTURE BY ITSELF (08:15 UTC): 78.833 (6,035M) / 89.872 (6,111M, median
  88.0%) / 89.442 (6,186M, median 88.6%).** 6/9 episodes at 87-89.9%; CONTACT records 85.3-85.6% at
  (-14, 186, 2,029), (-92, 420, 2,043), (-9, 303, 2,023) - real surfaces, the record's 83-88%
  region. ep1 (89.87%): launch at ~136 s, ridge at 137.9 s, over it east to (218, 397, 1,760) - the
  block's east-face ramp - at 139.9 s, up to z 2,300 (88.6%) and (-391, -675, 2,255) (88.3%), then off
  west and a fall at 148 s. Remaining to the finish slabs (z 3,008): ~700-800 u of climb; the
  record does 87.75% (t+390, z 2,181) -> finish (t+450) in 60 s. Hand-on for 89.872:
  ckpt_6110576640.pt (snapcopy, the eval's own step; pulled, md5 b77a0749).
* **rc4 08:39 UTC: 88.631 / 89.110 / 89.101 / 90.081 (6,564M) / 89.040 (6,639M)** - a plateau at
  88.6-90.1% (contact ~85.6%). 6,564M: 3/9 at 89.8-90.1% but 6/9 die early, several stalled in the
  FIRST trough (~12%, the 30 s patience cut): with the band at ~11,500 (80 s) 90% of spawns sit at
  80-140 s on the route and the first 80 s train on the 10% fresh starts only. Hand-on for 90.081:
  ckpt_6563561472.pt (pulled, md5 71c5ba98). No training finish yet (win 0.00).
* The record (ANALYSIS ONLY) after 88%: a FIFTH loop on the big block's east side - swings along y
  (-1,700 <-> +2,250), the east ramp (x ~100-230, z 1,750-1,990) in the middle and higher surfaces
  at the swing ends (z 2,180 -> 2,522), ~55 s pumping E 2,650 -> 4,070 (t+389..444), then the climb
  to the finish at t+450. The policy enters it (z ~2,380, E ~3,100) and dies after 1-3 swings.
* rc4 last: 89.029 / 88.820 (6,790M, medians 88-89%). Stopped at 6,865.6M (ckpt_latest md5 c7061e31).
  The 6,111M greedy re-recording (9 episodes): 4 reach contact 84.7-85.5%, 4 stop at the ridge top
  (68.8-69.6%), and one sat in the FIRST trough for the whole 910 s cap (contact 11.7%) - the early
  route has become fragile.

### rc5 `zlr2b1_rc5` (08:51 UTC): rc4's weights + the policy's OWN line as the spine, band at the start
`--demo-file autoresearch/spines/zlr2b1_rc4_6110_spine.npy` (its own greedy episode to contact
85.55%) `--demo-tau0 0 --demo-milestones 5859,2030 --ckpt-every 2e8`: half the demo draws at the
true start, half uniform over the whole own line (the early route trains again; every structure
keeps its share); the success line moves to d <= 1,019 (84.0% contact), the ridge (2,030) and the
old 9.48% line stay as milestones (all read on the contact record). Round 1's steady state (E3-E13)
had the band at the start too.
* **rc5 REGRESSED (08:51-09:17 UTC): 89.204 (its first = rc4's weights) / 65.368 / 63.785 / 60.800 /
  61.432 / 62.572 (7,244M).** With the band at the true start the launch region gets ~2.5% of the
  spawns and the greedy forgot the launch within 75M steps (back to the third-structure loop); the
  curriculum backed off (0% success near the start). The launch is a fragile skill that needs heavy
  training where it happens: keep the band inside the third structure. Stopped at ~7,285M.

### rc6 `zlr2b1_rc6` (09:18 UTC): rc4's final weights + the own-line spine, band inside the third structure
Resume of rc4's ckpt_latest (6,865.6M, md5 c7061e31 - not rc5's) + `--demo-file autoresearch/spines/
zlr2b1_rc4_6110_spine.npy --demo-tau0 11400 --demo-milestones 5859,2030 --ckpt-every 2e8`: the band
~57 s before the launch (where rc4's sat on the GE spine), uniform draws over the third structure,
the launch, the ridge and the block's east side; success line 84.0% contact.
* **rc6 REGRESSED too (09:18-09:45 UTC): 90.073 (its first = rc4's weights) / 63.837 / 32.013 /
  60.628 / 59.969 / 62.266.** Same restored config and trainer code as rc4 (diffed); the differences
  are the own-line spine and its 84% success line (band success 0%, backing off). Both rc5 and rc6
  lost the launch within 75M steps of the spine switch: the greedy's launch is a close call that
  flips under any change of the spawn mix (rc3 flip-flopped the same way). Lesson: once a spine
  works, do not swap it; add frontier spawns BESIDE it.

### rc7 `zlr2b1_rc7` (09:45 UTC): rc4's exact curriculum + 15% own frontier spawns
rc4's final weights (6,865.6M) + `--demo-file autoresearch/spines/zlr2b1_ge1_spine.npy --demo-tau0
11530 --demo-milestones 5859` (rc4's spine, band and milestone exactly) + NEW pool share
`--spawn-states runs/sp2/frontier_6110.npy --spawn-states-frac 0.15` (= autoresearch/spines/
zlr2b1_frontier_6110.npy: the policy's own states around its last contact-record improvement,
tools/frontier_states.py, 4 episodes at contact 84.7-85.6%). The file is re-read when it changes,
so the frontier pool can be refreshed from newer re-recordings without a restart.
* **rc7 (09:45-10:10 UTC): 89.385 (its first) / 89.030 / 88.558 / 88.437 / 88.248 / 88.718 (7,244M),
  medians 87-88.5%** - rc4's curriculum holds the level (the spine swap was the regression), no gain
  past 90 yet.
* The finish (ANALYSIS ONLY, t+441..450.6): the record pumps the fifth loop to E ~4,100, then rides
  north up the end of the block's east ramp at ~1,960 u/s ((310, 504, 1,666) at t+449.0 -> (59, 1,048,
  1,956) -> (13, 1,404, 2,434) -> (4, 1,860, 2,785)) and swings straight up into slab *16 at (-48,
  2,017, 3,007). The policy enters the loop at E ~3,100 and dies after 1-3 swings: survival + pumping
  there (~55 s for the record), then that swing.
* 10:11 UTC: re-recording rc7's 7,302M weights (dump states) to REFRESH the --spawn-states pool in
  place (the trainer re-reads the file when it changes; the 15% share stays).
* 10:17 UTC: rc7's 7,302M greedy re-recording: 6/9 reach contact 85.5% at 114-119 s (20 s faster
  than at 6,111M) and die ~7 s later; 2/9 sit 910 s in the first trough (contact ~20%). The new pool
  (tools/frontier_states.py, 4,859 states, autoresearch/spines/zlr2b1_frontier_rc7.npy, md5
  a64769ff) replaced the spawn-states file in place.
* **rc7 10:46 UTC: 88.654 / 88.416 / 88.636 / 88.532 / 89.429 / 88.688 (7,848M)** - flat. All 8 good
  episodes of 7,848M die the same way at ~122 s: the block's east side at ~112 s, z ~2,250 (88%) at
  114 s, then on the swing back NORTH along the east ramp ((307, 803, 1,657) at 1,300 u/s) they reach
  the north end heading north-west, low (z ~1,680), and fly off; the record rides up that north end
  to z ~2,190 at the same energy (E ~2,800, t+393.5): a line problem, not energy.
* 10:48 UTC: the pool narrowed to that struggle stretch (frontier_states --before 2 --stride 1: 2 s
  before the last contact-record improvement to 1.5 s before death, 5,431 states, md5 d1518d2f),
  swapped in place. GE ge5 (ge1's config, 15 min): roots = the same episodes' contact ticks after the
  record (contact_roots --stride 1 --min-record-frac 0.97, 582 roots at 78-83% on the block's east
  ramp) - can random bursts ride up the north end (contacts at 87-88%)?
* rc7 last: 88.870 / 89.115 / 88.349 (8,074M); stopped at 8,113.4M (ckpt_latest md5 50f1485a). ge5
  (from the policy's own contacts on the block's east ramp): deepest contact 85.91% at 6 min - a
  walkable ledge at z ~2,045 (x -100..0, y 80-550) the policy already reaches; no north-end ride.
  Stopped. Random-burst search does not ride the high-speed swings (ge2, ge3, ge5).

### NEW `--front-contact 1` (commit bb285780) and rc8 `zlr2b1_rc8` (11:01 UTC)
Under --ratchet-contact the reservoir's deepest states are ballistic apexes, so `--front-temp`'s
all-state band heated envs at the apexes (z ~2,300, 89-90%), not where the line ends on a surface.
With `--front-contact 1` an env is hot (every head at T 2) while its episode's PAID record is within
the band (1%) of the deepest contact record a training episode EARNED over the last 50 iterations
(RaceReward.rc_paid_min_d: paid episodes only, so spawn-initialised apex records cannot inflate
it). Generic (the same band rule, on the record the run is paid for). Off = unchanged.
rc8 = rc7's 8,113M weights + rc7's exact mix (GE spine, `--demo-tau0 10930` = rc7's last tau,
milestone 5859, the 15% frontier pool) + `--front-contact 1`. At start 4-6% of envs hot, band edge
85.1% (the struggle after the 85.5% contact). (ops: `train_fast.py --help` crashes on three
pre-existing help strings with stray '%' - --demo-success-strict/-alive, --auto-spine-recheck;
harmless for training.)
* **rc8 (11:01-11:26 UTC): 88.405 (its first) / 89.791 / 89.265 / 89.605 / 91.426 / 91.505 (8,491M,
  median 90.2%).** Past the 88-90 plateau: at 8,491M all 9 episodes reach the 85.5% contact (a
  walkable ledge at z ~2,040, x ~-20, y 120-200 on the block's east side) and survive 10-15 s longer
  than rc7's (best at 122-129 s, apexes to z 2,418); still no higher CONTACT - the gain is energy and
  survival in the fifth loop. Hot envs 4% -> 26% (more episodes at the contact frontier). Hand-on
  for 91.505: ckpt_8491368448.pt (snapcopy, the eval's own step; pulled, md5 70afe666).
* 11:36 UTC: the frontier pool refreshed from rc8's 8,491M greedy re-recording (9/9 at contact
  85.5-85.6%, ~15 s of survival past it; 21,157 states, autoresearch/spines/zlr2b1_frontier_rc8.npy,
  md5 50ed1814), swapped in place. Box extended to 14:28 UTC.
* **rc8 later: 91.742 (8,567M; 5/9 at 88.8-91.7%) then COLLAPSE: 30.320 (8,642M) / 30.634 / 63.706 /
  60.738 / 61.746 / 61.168 (9,020M)** - back to the third-structure loop, stall kills 21%, reward
  18/ep. The collapse precedes the 8,668.6M pool swap, so it is the run's own: --front-contact made
  26% of envs hot at 8,548M (every successful episode reaches the 85.5% frontier and stays hot ~15 s;
  every frontier-pool spawn starts hot) - a quarter of the batch at T 2 on every head, against ~1-6%
  under the all-state band. Hand-on for 91.742: ckpt_8566865920.pt (pulled, md5 9762d1a1).

### rc9 `zlr2b1_rc9` (12:01 UTC): rc8's 8,491M weights + the same mix + a GENTLER --front-contact
Resume of ckpt_8491368448 (the steadier 91.505 point, 9/9 at contact 85.6%) + rc8's flags with
`--front-temp 0.5` (hot = every head at T 1.5 instead of 2), `--demo-tau0 10735` (rc8's tau then).
If it collapses again: drop --front-contact (rc7's stable mix).
* **rc9 (12:01-12:23 UTC): 91.204 (its first) / 92.324 / 91.536 / 92.327 (8,718M, median 91.0%) /
  92.126 / 90.458** - T 1.5 holds where T 2 collapsed (rc8 fell at the same step); hot envs still
  ~23%; training apexes to 96-97% (reservoir deepest 96.1%). Hand-on for 92.327: ckpt_8718385152.pt
  (pulled, md5 f3d4a750).
* **rc9 12:49 UTC: 93.276 (9,020M) / 92.206 / 93.892 (9,171M, median 92.3%) / 93.761 / 93.883
  (9,322M).** Still climbing: at 9,171M 8/9 reach z 2,415-2,670 with E max 3,550-3,950 (was
  3,100-3,500); the best apex z 2,670 is ~340 u below the finish slabs (z 3,008); training's deepest
  state 97.2%; hot envs 38%. Hand-on for 93.892: ckpt_9171369984.pt (pulled, md5 8981a09b). No finish
  yet. 12:52: re-recording 9,171M to refresh the frontier pool; box extended.
* 12:57 UTC: pool refreshed from rc9's 9,171M re-recording (9/9 at contact 85.5-85.6% by 110-115 s,
  alive 20-30 s past it; 34,387 states, autoresearch/spines/zlr2b1_frontier_rc9.npy, md5 771c31ea).
* **rc9 13:19 UTC: 95.689 (9,549M) / 95.049 / 95.708 / 96.101 (9,775M, median 95.4%).** 8/9 at
  95.0-96.1%, apexes to z 2,814 (194 u under the finish slabs, z 3,008), E max 3,900-4,390 (the
  record's finish swing had ~4,100): the energy is there; the swing shape (horizontal speed at the
  apex) is what still keeps them under the slabs. Training's deepest state 98.4%. No finish yet.
  Hand-on for 96.101: ckpt_9775349760.pt (pulled, md5 aa33a79a).
* **rc9 final (13:56 UTC, 10,491.5M): ... 94.891 / 94.891 / 97.107 (10,002M) / 96.770 / 97.339
  (10,153M, median 95.2%) / 95.359 / 97.120 / 95.994 / 96.842.** Hand-on for 97.339:
  ckpt_10152837120.pt (pulled, md5 6cd08865). ckpt_final md5 1858dee3.
  (ops: `ls traj_*.jsonl | tail` sorts the 10-billion-step evals BEFORE the 9-billion ones - score
  with `score.py --best <run dir>`.)

### rc10 `zlr2b1_rc10` (14:04 UTC): rc9 continued
rc9's ckpt_final + rc9's exact flags (`--front-contact 1 --front-temp 0.5`, GE spine, `--demo-tau0
9800` = rc9's last tau, milestone 5859, the rc9 frontier pool at 15%). Budget 2e9 (to ~16:00 UTC,
then the wrap-up).
* **rc10 (14:04-14:29 UTC): 96.125 (its first) / 95.996 / 97.293 / 96.477 / 96.585 / 96.420** - a
  plateau at 96-97%, no finish. Why (checked against zones.json): every high apex (z 2,550-2,909,
  E 3,850-4,260) is at a SWING END (x ~500-900, y +1,600..2,300 or -1,100..-1,500) - outside every
  finish slab's footprint (the slabs cover the middle band y -736..1,312 and the strip x -320..-64
  north/south of it). Height alone does not finish: it needs another swing shape (the record,
  ANALYSIS ONLY: up from the east ramp's north end, curving west into the strip at x ~-48, y 2,017).
* GE ge6 (14:31 UTC, 20 min): roots = rc9's 9,171M re-recording contact ticks after the contact
  record (3,576 roots on the block's east ramp, E up to ~4,000); explore_phase1 stops and writes
  win_<k>.spine.npy on any slab entry.
* ge6 (20 min, 3.83M cells): 0 goal hits; its closest cell is an apex 242 u from a slab. Random-burst
  search does not find the finish swing either.
* **rc10 14:53 UTC: ... 97.687 (11,021M) / 98.329 (11,096M, median 95.4%) / 96.936 / 94.946.**
  98.33% = ep8's apex at z 2,857 (E max 4,407) at 158.5 s, still outside the slab footprints; 3/9
  died early in the first troughs (22-28%). Hand-on for 98.329: ckpt_11096031232.pt (pulled, md5
  3c40f24d). 14:55: re-recording 11,096M to refresh the frontier pool.
* 15:00 UTC: pool refreshed from rc10's 11,096M re-recording (8/9 at contact 85.5-85.6%, alive 35-46 s
  past it - was ~15 s at 8,491M; 47,793 states, autoresearch/spines/zlr2b1_frontier_rc10.npy, md5
  2b29ec9e).
* rc10 15:27 UTC: ... 96.799 / 94.966 / 95.736 / 96.318 / 97.156 (11,851M): the 95-98 level holds,
  several evals with 4-6/9 early deaths in the first troughs (medians ~12%); best still 98.329.
  One training iteration logged win 0.91% @ 0.8 s - a frontier-pool spawn beside a slab flying in
  (harvest, CLAUDE.md's win-rate trap), not a policy finish; it does say a slab is within a small
  perturbation of the policy's own fifth-loop states.
* rc10 final (16:02 UTC, 12,491.7M): last evals 95.5-96.9; best 98.329 (11,096M). Pulled ckpt_final
  (md5 d1a079ab) and ckpt_11096031232 (md5 3c40f24d). rc11 (16:07 UTC) = rc10 continued for 4e8 steps
  (a few more greedy evals before the wrap-up), same flags, `--demo-tau0 8980`.

### zlr2b1 round-2 lesson

**Result: 98.329** (rc10 at 11,096M, median 95.4%; 0 finishes) from the round-1 incumbent's 64.671 -
surf_za_loop from the TRUE start, champion-free, no demo in training, one RTX 5080 (0.34 $/h, ~11 h).
Hand-on: `runs/zlr2b1/zlr2b1_rc10/ckpt_11096031232.pt`. Weights: incumbent -> rc1 -> rc2 -> rc3 ->
rc4 -> rc7 -> rc8 (8,491M snapshot) -> rc9 -> rc10. Evals over time: 64 -> 72 (rc3, the launch) -> 81
(rc4) -> 90 (rc4, the block found by the RL) -> 91.7 (rc8) -> 93.9 / 96.1 / 97.3 (rc9) -> 98.3 (rc10).

**The insight: the incumbent's 64.67% was ballistic.** After 52 s it touched one surface (a ramp
crest, contact progress 38.9%); the rest was flight apexes over open air, which the free-flight voxel
field reads as progress. The record's contact-only progress tracks its all-state progress within
2.6 points. On the field the true exit (the ridge top, 60.9%) read LOWER than the dead-end apexes.

**What worked, in order (each generic, no constant reads the map):**
1. NEW `--ratchet-contact 1`: the ratchet pays a record that moves only on supported ticks (ground,
   or a surface pushing vz back up); a landing pays everything flown since the last contact. Alone it
   did not find the exit (rc1, 320M steps), but it is what makes the exit worth taking.
2. Go-Explore rooted at the policy's OWN contact ticks after its contact record stopped improving
   (NEW tools/contact_roots.py; ge1's config: --cell 128 --cell-energy 64 --cell-heading 1
   --view-follow=-90..90 --select-energy 3, 40 threads beside the trainer): 232 contact cells on the
   ridge top within 6 min. NEW tools/ge_contact_pick.py picks the deepest CONTACT cell (one null tick
   shows the surface pushing back), ge_replay makes the exact line (an air-braked reversal off the
   crest), NEW tools/contact_spine.py splices it onto the policy's own greedy line (record_ckpt
   --dump-states), cut at the deepest contact -> autoresearch/spines/zlr2b1_ge1_spine.npy.
3. The spine's curriculum with the band where the decision is made: at the GE root (rc3:
   --demo-tau0 19656) the greedy launched in 2/8 evals (72, 77); moving the band back into the third
   structure (rc4: --demo-tau0 12000) trained the long pumping before the launch too -> 79-81 steadily,
   then the RL found the big block's east ramp BY ITSELF (contact 85.6%, 89.9%).
4. 15% own frontier spawns (NEW tools/frontier_states.py: states from 2 s before the last contact-record
   improvement to 1.5 s before death, from greedy re-recordings) via --spawn-states, REFRESHED in place
   (the trainer re-reads the file) every ~1-2 h as the frontier moved.
5. NEW `--front-contact 1` (--front-temp's band read on the EARNED contact record, not the all-state
   reservoir whose deepest states are apexes): broke the 88-90 plateau (rc8). At T 2 it collapsed after
   ~450M steps (26-38% of envs hot); at `--front-temp 0.5` (T 1.5) it held and climbed 91 -> 98.3.

**What did not / what to avoid:**
* Never swap a working spine: rc5 (own-line spine, band at the start) and rc6 (own-line spine, band in
  the third structure) both lost the launch within 75M steps (back to ~61%). The launch is a close call
  in the greedy; add frontier states BESIDE the spine instead (rc7).
* Random-burst Go-Explore cannot ride the high-speed loops: ge2 (ridge top, contact-only cells), ge3/ge4
  (fourth structure), ge5/ge6 (fifth loop) found nothing past their roots. It found the low-speed exit
  (ge1) only.
* T 2 frontier temperature on a large hot fraction destabilises (rc8).

**Where the next round starts:** the greedy reaches the fifth loop on the big block's east side every
time (contact 85.5% at ~110-115 s, alive 35-46 s past it, E up to ~4,400 = the record's finish
energy) and its apexes reach z ~2,900 (98.3%). No finish: every apex is at a SWING END (x ~500-900,
y +1,600..2,300 / -1,100..-1,500), outside every finish slab's footprint (zones.json: the middle band
y -736..1,312 and the strip x -320..-64). The record (ANALYSIS ONLY) finishes by riding up the east
ramp's north end at ~1,960 u/s and curving WEST into the strip at (-48, 2,017, 3,007). One training
iteration showed a frontier-pool spawn flying into a slab in 0.8 s: the slabs are a small perturbation
away. Ideas: a search on the policy's own fifth-loop states for a slab entry with the POLICY's actions
plus held view offsets (beam_tas --branch-at with a finish objective) rather than uniform bursts; the
early-route fragility (2-6/9 episodes die in the first troughs) needs fresh starts in the mix.
* rc11 (16:07-16:31 UTC, 12,491.7-12,891.7M): 96.814 / 96.238 / 95.598 / 95.808 / 96.991 / 94.463 - no
  gain on 98.329. Box 58621920 released at 16:34 UTC (idle; every cited run pulled).
* **Wrap-up (16:41 UTC): Final 98.329** (zlr2b1_rc10, traj_11096031232; best of 12 runs scored with
  `score.py --best <run dir>`: c1 64.417, rc1 64.719, rc2 63.211, rc3 77.176, rc4 90.081, rc5 89.204,
  rc6 90.073, rc7 89.429, rc8 91.742, rc9 97.339, rc10 98.329, rc11 96.991). Hand-on
  **ckpt_11096031232.pt** (rc10, the eval's own step; md5 3c40f24d) with rc10's ckpt_final (md5
  d1a079ab), both pulled. Box 58621920 (RTX 5080, 0.34 $/h, 05:37-16:34 UTC, ~$3.7) released.

## zlr3b2 (round 3, slot 2) - 2026-11-01 17:32 UTC start

Box 58689691: RTX 5080 + Ryzen 9 7945HX (32 threads, quota 30.7 CPUs), 0.387 $/h, South Carolina
(machine 154401), rented 17:39. c1 trains at ~236k steps/s with searches beside it.

### Where the incumbent misses the slabs (17:35 UTC, analysis of incumbent/r2/traj_11096031232)
The core's finish test (src/env.c): a STANDING player's origin inside [slab - hull] = z >= 2,972
(3,008 - 36) and x/y within the slab footprint +16 u. The five slabs: two wide east-west bands
(x -2,360..3,008, y -736..40 and 536..1,312), x -2,360..-64 y 40..536, and two strips x -320..-64
north (y 1,312..3,360) and south (y -2,784..-736). In the 98.329 eval the best-progress points are at
the footprint EDGE (gapH 0-46 u) but 115-350 u too LOW: ep8 (98.33%) at (-382, 2,627, 2,857)
descending (vz -254, |v_h| 1,400, E 4,118); the others at z 2,628-2,740. The apexes at the swing ends
(x 500-900) are 350-650 u outside the footprint. So the line is near; it needs ~120-350 u more height
INSIDE a footprint.

### The greedy in the beam core (beam_tas, 16 greedy envs, all identical in one batch)
Seed 0 idles in the first trough for 210 s (d 5,235); seeds 1-4 reach best d 286 / 256 / 228 / 317
(95.5 / 96.0 / 96.4 / 95.0%) and die at 155-168 s. Seeds 1 and 3 cross the NORTH STRIP rising at
(-52, 1,995, 2,632) t146.0 and (-46, 2,036, 2,718) t156.9 (vz +450..480, |v_h| ~1,350, E ~3,950):
250-340 u under the trigger at almost exactly the record's entry point (-48, 2,017, 3,007).
(ops: beam_tas without a finish loads --route-file for its death report and crashes on the default
cannonball path: pass `--route-file none`.)

### Finish search: policy beams from the incumbent's own greedy lines (17:45-18:05 UTC, box beside c1)
`tools/beam_tas.py runs_ckpt.pt --envs 4096 --greedy-envs 1 --prefix-line <greedy line>:<T>
--max-ticks T+SPAN --save-frontier --keep-finishers 16 --allow-nonfinisher --skip-gate --route-file
none` (the greedy line = beam_tas `--envs 16 --greedy-envs 16 --save-frontier --frontier-last` at
spawn seed s; the prefix is replayed open-loop, the search is the policy's own proposals, cloned
every 25 decisions). ~270k env-steps/s on the 5080 beside training; a 42 s span takes ~1.5 min.
* p3a (seed 3, T 145.6 s, 29 s span, score d): frontier d 129 (98.0%) at 163.8 s, 0 finishes.
* q1d (seed 1, T 133 s, 42 s, d): d 77 (98.8%) at 165.8 s, 0 finishes.
* q3dh (seed 3, T 145.6 s, 42 s, `--score dh --score-energy 1`: -d + energy height): d 80 at
  176.4 s, E to 4,509, 0 finishes - the energy rank keeps lineages alive and pumping.
* The near misses are all alike: the north swing's apex at x -348..-383 (the strip's west edge is
  -336), y 2,630-2,790, z 2,830-2,890, heading north at |v_h| ~1,500 with vz ~0 and E ~4,300:
  the apex needs E >= ~4,400 (z_apex = E - v_h^2/1600) and ~50 u further east.
* **r3dhL (seed 3, T 145.6 s, 105 s span, dh): FINISH at tick 28,569 (199.98 s)**, env 1632, gen 77
  - the first finishing line from the true start on this map (the greedy prefix + the policy's own
  sampled actions, energy-ranked selection).
  The line (`autoresearch/spines/zlr3b2_finish1.npy`, 28,569 states = 199.98 s, md5 0bb7d19e; the box's
  runs/beam/r3dhL/beam_frontier.npz md5 07ffa6a5, replay bit-exact): the greedy prefix to 145.6 s, then
  ~9 swings of the fifth loop pumping E 3,593 -> ~4,500 (the greedy dies at ~150-160 s with E ~4,300),
  and at 198.5 s a swing WEST off the east ramp: up south-west to (-1,276, -1,065, 2,886) with vz +491,
  a wall takes the horizontal speed (E 4,441 -> 3,054, v (-1,496, 96, 491) -> (-164, 0, 104)) and the
  upward momentum carries it ~145 u into the south-west wide slab at (-1,792, -752, 3,030) (slab y
  -736..40, hull edge -752). Not the record's north-strip entry: a route of its own. Physics checked:
  vz 491 at z 2,886 gives z ~3,031 after 0.5 s of gravity.
* 18:00 UTC: c1's pool swapped in place: `tools/finish_pool.py --base <rc10 frontier pool> --line
  runs/beam/r3dhL/beam_frontier.npz --secs 30 --share 0.5` (95,586 states: the rc10 frontier pool +
  the line's last 30 s x11.2 = half; md5 9f7b952e). Re-read by the trainer at step 11,452M. Its
  training win rate jumped to 5-13% at "@ 0.1 s" - the pool's last-tick spawns finishing at once
  (the harvest, CLAUDE.md section 3), not the policy; the greedy eval decides.
* **r3dh2 (seed 3, T 145.6 s, 105 s span, `--score dh --score-energy 2`): 11 finishers, the first at
  tick 30,219 (211.53 s)**. NEW `tools/finisher_states.py` replays every kept finisher of a
  beam_best.npz (acts_all / view_all) exactly as save_frontier_line replays the winner: 11/11 ok.
  All 11 are near-clones entering the NORTH STRIP at its west edge: from (-769, 1,906, 2,847) at
  211.0 s heading north-east (v (474, 1,328, 565), E 4,289) into a ledge face at x -320 (vx -> 0) with
  the hull already above z 2,991 -> origin (-336, 2,285..2,408, 3,027..3,046) = the trigger's face
  touched (the core's inclusive hull box; GoldSrc's absmin/absmax are even 1 u wider). Saved:
  `autoresearch/spines/zlr3b2_finish2.npy` (30,219 states, md5 96d6fe85). A higher energy weight
  (2 vs 1) found 11 finishers to 1: the selection keeps the lineages that pump.
* 18:04 UTC: c1's pool rebuilt from BOTH routes (`--line r3dhL --line r3dh2 --secs 30 --share 0.5`,
  95,586 states, md5 a95d0112).
* Side effect to watch: the reservoir harvests a finishing episode's whole chain (no pre-end margin
  on a success), so its deepest states are now beside the slab, and --respawn-front-perturb (25% of
  the non-start rows, ~19% of all spawns) perturbs THOSE ("front-perturb: 922 rows from 2
  frontier-band reservoir states (d <= 32u)"): trivial wins at "@ 0.1 s". Harvest, not evidence.
* **r1dhL (seed 1, T 133 s, 105 s span, dh 1): 31+ finishers, the first at tick 27,527 (192.69 s)**;
  16 kept, 16/16 replay ok, all into the north strip's west edge (x -336..-344, y 2,219-2,655,
  z 3,031-3,044, 192.69-192.79 s). Three searches of three found finishes once the span let the
  lineages live past ~190 s: the energy-ranked selection (survive and pump) is what the greedy lacks.
* c1 evals after the pool swap: 11,474M 0/9 (deaths 157-188 s), 11,549M 0/9 (deaths 128-229 s; 3/9
  stalled in the first trough at ~30 s). Training win rate 0-1.2% "@ 0.7 s": the policy's own
  samples rarely reproduce a finish from a pool state more than ~1 s out.
* 18:08 UTC: pool -> the last 10 s of the three winners (r3dhL, r3dh2, r1dhL) at 70% + the rc10
  frontier states at 30% (159,310 states, md5 2dce6db4), swapped at ~11,589M: a manual backward step
  (the trainer's own curriculum holds the GE spine); widen the window once the training win rate
  shows finishes from further out.
* r2dh2 (seed 2, T 136.5 s, 105 s span, dh 2): 36 finishers, the first at tick 25,845 (**180.91 s**),
  into the north strip flying EAST at z 3,036 ((-344, 2,749, 3,036), v (1,493, -229, 11)) - no wall.
* c1 final: evals 95.3 / 96.0 / 96.2 / 95.7 / 96.2 / 97.0 / 95.7 / 96.2 / 97.8 (11,096-11,700M; best
  97.797), 0 finishes in any of them; stopped 18:15 UTC at ~11,725M.

### Finish-probability bench (18:14 UTC): why a uniform pool window does not teach the finish
c1's 11,6xxM weights spawned at r1dhL's line N seconds before its finish, 512 independent
stochastic rollouts each (beam_tas --start-state, no cloning): **1 s: 170/512 finish (33%); 3 s: 17
(3.3%); 6 s: 0; 10 s: 1; 20 s: 1.** The finish hinges on a swing 3-6 s before it; the 10-30 s pool
windows spent most of their spawns where the policy's own samples never finish.

### NEW `--finish-line` (commit 8c95f310, tests/python/test_finish_line.py)
A SECOND backward curriculum (Salimans-Chen; the trainer's own DemoCurriculum, grow + front band) on
a finishing line of the policy's own, on its own share of the spawn pool, beside the --demo-file
curriculum (which keeps its spine, band and share): starts at the line's last state, walks back
while the band FINISHES >= 20% (min 50 eps, cooldown 5, step 0.25 s doubling to 4 s, band 0.5 s,
half the draws on the band), backs off below. Success = a finish (the core's goal hit). Flags:
`--finish-line FILE --finish-line-frac 0.15` (+ window / grow / grow-max / rate in seconds). Off =
byte-identical (every statement behind FINL / args.finish_line; DemoCurriculum without the new
arguments = the base commit's class, run side by side over 400 iterations in the test).
TRAIN_ONLY in record_ckpt. Generic: any map, any finishing line the policy's own search returns.

### c2 `zlr3b2_c2` (18:17 UTC): c1's final weights + `--finish-line` on r2dh2's 180.91 s line
c1's ckpt_latest (md5 e08fb8c1) + c1's flags with `--spawn-states runs/sp/base_rc10.npy` (the rc10
frontier pool alone again) `--finish-line runs/sp/fin_r2dh2.npy --finish-line-frac 0.15` (the line:
25,845 states, md5 0190559e = autoresearch/spines/zlr3b2_finish3.npy), `--demo-tau0 9215` (c1's
last GE tau). First iterations: the finish-line band walked 0 -> 0.73 s before the finish at 57-95%
band success.
* c2 18:29 UTC (11,75-11,98B): greedy evals 96.706 / 95.603 / 96.757 / 95.654, 0 finishes. The
  finish-line band reached ~4.2 s before the line's finish in ~10 min and oscillates there at 16-22%
  band finish rate (advance/backoff around the 20% rule) - the bench's hard zone. The greedy's good
  episodes pass the middle band's north edge (x 740-840, y ~1,300-1,360) 290-350 u under the slab with
  E max 4,145-4,361 at death; 3/9 die at the fifth loop's entry (~78%, z ~1,650, 126-147 s).
* More lines: r4dh2 (seed 4, T 129.5 s, dh 2) 1 finisher at 205.35 s; r1dh2 (seed 1, dh 2, on c1's
  weights) 0 finishers (frontier d 0 at 186.9 s without a goal hit).
* **Bench of c2's weights (~12.0B, 18:31 UTC) on r2dh2's line, 512 own samples each: 2 s: 218/512
  (43%); 4 s: 128 (25%); 6 s: 7 (1.4%); 10 s: 215 (42%); 20 s: 6 (1.2%); 40 s: 0** - against c1's
  33 / 3.3 / 0 / 0.2 / 0.2% (1/3/6/10/20 s, r1dhL). The curriculum teaches the finish; 10 s (42%) >
  6 s (1.4%) says the 10 s state has a finish on an earlier swing the 6 s one has already missed.
* c2 18:38 UTC: finish-line band at 7.35 s (16-30% band success); training win 7.3% "@ 4.1 s" (finishes
  now ~4 s after a finish-line spawn). Greedy evals 97.950 (12,054M) / 95.250 / 97.344 (12,205M), 0
  finishes. The greedy's behaviour moved: at 12,205M the good episodes cross the NORTH STRIP rising
  (x -55..-67, y 1,769-2,036; ep5 97.34% at (-368, 2,542, 2,789) heading east at 180.5 s) - the
  line's finishing swing - but with E 3,440-3,990, 190-420 u short: it swings for the slab before it
  has pumped (the line swings at E ~4,300+). The curriculum has not yet reached the pumping part.
* **c2 FINISHES FROM THE TRUE START (18:50 UTC): 98.938 (12,356M) / 109.242 (12,431M: 1/9, 168.25 s) /
  110.409 (12,507M: 2/9, best 163.15 s, mean 165.9 s).** The first greedy finishes on surf_za_loop,
  champion-free, no demo in training. Both enter the north strip at its west edge like the search
  lines ((-340, 2,351, 3,038) at 163.1 s, (-336, 2,203, 3,065) at 168.6 s; E 4,045-4,106); 450.46 s is
  the record on our clock (491.65 s listed). The finish-line band had walked back to 63.7 s before
  the line's finish (training win 30.5% "@ 52 s": finishes ~a minute after a finish-line spawn).
  Hand-on candidates pulled: ckpt_12352225280.pt (154.7M before the 110.409 eval, md5 0c8a37fe) and
  ckpt_12552503296.pt (45.6M after, md5 5d587b5b); snapcopy started beside c2 (18:54) so later evals
  get a checkpoint at their own step.
* **c2 19:17 UTC: 116.037 (12,733M: 7/9, best 156.69 s) / 114.960 (6/9, 153.56 s) / 112.611 (4/9) /
  112.582 (4/9) / 117.151 (13,035M: 8/9, best 156.45 s, median 100%) / 117.109 (13,111M: 8/9, 160.15 s).**
  Within ~25 min of the first finish the greedy finishes 4-8 of 9. The finish-line band is at tau 3,149
  of 25,845 = 158.9 s before the line's finish (nearly the whole line, 66% band success); training win
  48% "@ 80 s". The greedy now finishes FASTER than the line it learned from (153-168 s vs 180.9 s).
  Hand-on for 117.151: snapcopy's ckpt_13035372544.pt (the eval's own step; pulled, md5 fe9c190b);
  ckpt_13110870016.pt (117.109) pulled too (md5 b712d950).
* **c2 19:38 UTC: 117.272 (13,337M: 8/9, best 145.55 s) / 116.182 (7/9, 143.65 s) / 114.975 (6/9) /
  118.403 (13,564M: 9/9, best 143.76 s).** Every greedy episode finishes. Hand-on: snapcopy's
  ckpt_13563854848.pt (the eval's own step; pulled, md5 717b00cf). The finish-line band is at tau
  ~1,364 = 171 s before the line's finish (9.5 s into the line); training win 67%.
* Speed beams on c2's ~13.1B weights from the TRUE start (4,096 envs, no prefix): `--score d`: first
  crossing 156.71 s (env 0, the greedy itself; 4 finishers); `--score dh 1`: 147.55 s (20,088
  finishers - the whole population finishes). Both slower than the greedy's own 143.65-145.55 s by
  then: the beam's resample-by-d is no speed search once the policy finishes.
* c2 later: 115.068 (13,639M: 6/9, 143.89 s) / 116.181 (13,715M: 7/9, 143.72 s). **c2 ENDED at its
  budget** (`done: 13,751,549,952 steps` at 19:46 UTC, ckpt_final md5 e1234304): --steps 2e9 at 376k
  steps/s is 1.5 h, and the box then sat idle until 20:10 (my wakeup was 25 min) - CLAUDE.md's 5-minute
  rule broken. Size the budget past the next wakeup (c3: 6e9). The best times plateaued at ~143.7 s
  over the last 5 evals (13,413-13,715M); finishes per eval 6-9 of 9.

### c3 `zlr3b2_c3` (20:13 UTC): c2 continued
c2's ckpt_final + c2's exact flags (`--demo-tau0 8510` = c2's last GE tau; the finish-line curriculum
restarts at its line's end - its position is not in a checkpoint), budget 6e9, snapcopy beside it.
* c3 20:41 UTC (13.75-14.43B): 117.284 (8/9, 144.48 s) / 116.204 (7/9, 141.61 s) / 116.156 / **118.426
  (13,979M: 9/9, 141.68 s)** / 116.201 / 116.210 / 115.086 / 115.101 / **118.451 (14,356M: 9/9, best
  139.42 s)** / 113.969 (5/9). 5-9 finishes per eval; the best times creep down 144 -> 139 s. Hand-on
  for 118.451: snapcopy's ckpt_14356054016.pt (the eval's own step; pulled, md5 9693ab6b). The
  finish-line curriculum re-walked to 163 s before its line's finish within ~20 min of the launch.
* c3 to 15.34B (stopped 21:20 UTC): 9/9 evals at 139.55 / 141.34 / 141.48 s besides the 139.42 s one;
  4-9 finishes per eval; the best times flat at 139.4-141.5 s for ~1 h. Every greedy finish checked
  real: the last ticks fly EAST at vx 1,100-1,250 u/s with the origin at z 3,050-3,100 (hull bottom
  above the slab's z 3,008) and cross x -336 by ~8 u per tick, the finishing origin ~6-8 u INSIDE the
  slab's clip hull, no wall contact; score.py's box_only 0 on every eval (the uf2 AABB lesson).
* Pumping efficiency (13,564M-era greedy, 10 s windows): net +46-49 u/s of energy height over the
  run, gross gains 52-69 u/s against the ~73 u/s an optimal air strafe adds (0.51 u per 7 ms
  frame), losses 2-44 u/s at contacts (second trough -> third structure entry, the launch).

### Speed search: an ENERGY-ranked beam is a speed search (21:17 UTC)
beam_tas from the TRUE start on c3's 15.1B weights, 4,096 envs, `--score dh --score-energy 10` (the
energy height dominates -d): **first crossing 134.60 s** (the greedy's best 139.42 s; all 4,096
lineages finish; replay bit-exact) -> `autoresearch/spines/zlr3b2_finish_fe10.npy` (19,229 states,
md5 3efae3ed). Earlier full-route beams on c2's weights: `--score d` 156.71 s (the greedy env itself),
`--score dh 1` 147.55 s - a d rank selects the phase of the swing, not the pace. The 134.6 s line
gains in the second trough (30% at 50.6 s vs the greedy's 56.3 s) and the third structure (60% at
101.4 vs ~111 s); the fifth loop is the same ~30 s.

### NEW `--auto-finish` + tools/finish_search.py (commit 7d84be4e)
The round's manual loop as a trainer rule: after a greedy eval with fewer than --auto-finish-quorum
(1) finishes and no --finish-line armed, finish_search.py runs BESIDE the training on a copy of
ckpt_latest: greedy lines at 4 spawn seeds (beam_tas, all envs greedy, --frontier-last), the deepest
one's own states 10 and 25 s before its end as roots, energy-ranked beams (dh 2, 105 s, 4,096 envs);
the first finishing line becomes the --finish-line curriculum (its share and constants). Off =
byte-identical; TRAIN_ONLY in record_ckpt; tests in tests/python/test_finish_line.py. Not yet run
end to end on a box.

### c4 `zlr3b2_c4` (21:20 UTC): c3's 15.34B weights + the finish-line curriculum on the 134.6 s line
c3's ckpt_latest (md5 bc51a503) + c3's flags with `--finish-line runs/sp/fin_fe10.npy` (the energy
beam's line instead of r2dh2's 180.9 s one), `--demo-tau0 8000` (c3's last GE tau). Expert
iteration (uf2's search -> own faster line -> curriculum loop): does the greedy follow the faster line?
* c4 (21:20-22:05 UTC, 15.34-16.38B): 118.436 (9/9, 140.81 s) / 116.207 / 115.100 / 118.382 (9/9,
  145.60) / 112.880 / 117.335 / 114.011 (5/9, 138.99 s) / 118.445 (9/9, 139.93 s) / 115.107 / 115.108 /
  117.322 / 116.229 (7/9, 139.36 s) / 116.171 / 117.328 - **the curriculum on the faster line did
  NOT speed the greedy up** (its best finishes 139-141 s, as c3's); no new best. Spawning along a
  faster line teaches finishing from its states, not its pace.
* Speed beams on c4's 15.9B weights from the true start: `--score dh --score-energy 10` -> 128.59 s
  (13,055 finishers); `--score-energy 30` -> **127.95 s** (9,914 finishers) =
  autoresearch/spines/zlr3b2_finish_ge30.npy. The search side of expert iteration keeps finding
  faster own lines (139.4 greedy -> 134.6 -> 128.0 s); the policy side is what has to follow.

### NEW `--finish-line-pace FRAC SECS` (+ `--finish-line-pace-bonus`, commits cc187785, 074c730c)
uf2's speed mechanism (x11/x12: a pinned, pace-gated spine on the newest beam's line, 36.67 ->
35.80 s) cannot run on this lineage's main spine: under --ratchet-contact the demo bonus pays only
when the CONTACT record crosses the success line, and a finish here is airborne (the finishing row
is an ended row, which the bonus never pays). So the pace gate is on the finish itself, in the
finish-line curriculum: a finish from a finish-line row r pays the bonus (50, on top of the
finish's own 50) and counts as the curriculum's success only within (rows left after r) x (1 +
FRAC) + SECS; an unpaced finish is a curriculum failure. Off = byte-identical; TRAIN_ONLY.

### c5 `zlr3b2_c5` (22:06 UTC): c4's 16.38B weights + the PACE-GATED finish line on the 127.95 s line
c4's ckpt_latest (md5 5be13595) + c4's flags with `--finish-line runs/sp/fin_ge30.npy
--finish-line-pace 0.05 1.0 --finish-line-pace-bonus 50` (5% + 1 s; from the true start that is
135.4 s against the greedy's 139-141), `--demo-tau0 7710` (c4's last GE tau). The curriculum walks
back from the line's end only as far as the policy keeps the line's pace.
