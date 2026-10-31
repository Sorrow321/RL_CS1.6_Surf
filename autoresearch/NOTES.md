# NOTES - the surf_unitfarmer2 lineage

The lineage's memory: every round's winner appends what it learned (what worked, what did not,
the numbers). Read it before you start; append to it before you wrap up.

## Round 0 (operator, 2026-10-30)

* A new lineage on a new map. The two earlier fork-and-flush programs on this harness -
  surf_src_sidistic (2026-10-20..23) and surf_princessburglar_b26 (2026-10-23..30) - left their
  code and tools in this branch's history. Their notes are NOT copied here; they are in this
  branch's own git history (`git show 8a825803:autoresearch/NOTES.md`: the first
  princessburglar lineage, rounds 1-9, and the operator's expert loop after it). The second
  princessburglar lineage (pbf*, the corrected core) is not in this history; what transferred
  from it is summarised in PROTOCOL.md **This lineage**, with its exact flags.
* unitfarmer2's own history (2026-09..10-17, before this harness existed: the curriculum, novelty,
  frontier-loop and search arms) is in `docs/research-results.md` (grep `uf2` / `unitfarmer2`).
  Read it before you plan: it lists what failed, why, and which checkpoints are demo-trained
  (CLAUDE.md section 0: never resume, benchmark or build on those).
* The map files came from the user on 2026-10-30 (the BSP, md5 9834c24c; its WR demo, analysis
  only). The operator baked the caches on one box (smoke launch uf2op_bake, 40M steps from
  scratch at 227k steps/s on an RTX 4080). That launch passed the record gate (greedy, stoch,
  mixed, POV), and its 0.5M-step eval scored 2.831 (best progress 2.8%, 0/9).
* The WR demo, parsed, is in `$AR_HOME/wr/`: `surf_unitfarmer2_wr.jsonl` at 10 ms, the raw
  frames, and the movevars. It is for ANALYSIS ONLY (CLAUDE.md section 0). It never becomes a
  spawn file, a spine, a target sequence, a BC set or a reward constant.

## Round log

## uf2r1b1 (round 1, slot 1) - 2026-10-30 21:40 UTC start

Box 58455319: RTX 5080 + i9-14900K (32 threads, cgroup quota 30.7 CPUs), 0.38 $/h, Utah.

### The grader's field along the record (ANALYSIS ONLY, to read results)
`autoresearch/analysis/uf2_ge_report.py` reads a GE archive on config.field; the record on the
same field (goalc_32, d0 30,647.5), t from the start curtain: running-min progress 1.76% at t+1.5
(the platform's edge), then the pit: progress -7.56% at t+3.5 (d 32,965), back to 0.85% at t+6,
**5.48% at t+8.0 (out of the pit, 1,104 u/s, z 354)**, 8.7% at t+9.5, 10.2% at t+10, 15% at t+11,
31% at t+15, 48% at t+20. E = z + |v|^2/1600: 1,306 at the curtain, 1,361 at t+1 (air strafe off
the edge), 1,204 at t+3, 1,047-1,145 from t+4.5 to t+8.

### The start, in our core (analysis, 21:50-22:00 UTC)
* Kill geometry (BSP triggers, all fatal under --teleport-fail): `*18` recomeco is a 1-u SHEET at
  z 160-161 over x -3,456..-896, y -2,400..-448 (north of the platform, under the record's pit-exit
  flight, which passes its WEST edge at x < -3,456 before dropping below it at t+10); `*7`
  recomeco's AABB covers the pit and the space under the platform's edges up to z 577 (the pit
  floor's kill and the catches under the platform); `*8` a sheet at z -768 further north.
* **The trap**: the field (free-space geodesic) runs over the `*18` sheet. The scratch control's
  greedy at 76M (`uf2r1b1_ctl`, score 4.910) walks off the platform's east side and dies on the
  sheet at (-1,767, -2,190, 198) at 4.7 s, 4.9% progress, 9/9 identical; E 1,228 -> 575.
* pb_r26 zero-shot (greedy, 9 eps, recorded on the box): drops off the platform's south side
  and dies in `*7`'s catch under it at z ~505-590 after 6-13 s, 2.2-2.9% (score 2.860).
* Go-Explore from the true spawn, pbf config at 10 ms (`--exact-chains --cell 64 --cell-energy 32
  --cell-heading 1 --view-follow=-90..90 --select-energy 3 --act-every 4 --decisions 100 --envs
  4096`, 12 threads beside the control; `runs/ge/uf2a` on the box): 672k cells in 10 min. It
  matches the record's energy down to the south ramp (same-motion E at t+1 / t+2: 1,294 / 1,182
  vs the record's 1,361 / 1,272), then its pit states are FREE FALLS (z -1,000..-800 with E up
  to 1,233 but vz -1,808; same-motion E at t+3 / t+3.5 / t+4.5: 705 / 189 / 38 vs 1,204 /
  1,201 / 1,113): uniform bursts do not surf the bowl. Its deepest cells (8.5%) are falls onto
  the `*18` sheet north of the platform (E 250-900). No pit exit, nothing past the sheet.

### Plan after the control (22:00 UTC)
The control reaches the trap in 151M steps and stays there (5.608 / 5.568 at 151M / 227M). Two
generic levers are untested together on this map:
* the DEATH CHARGE (Grzes 2017, CLAUDE.md 0b's example of a generic objective fix): with the
  ratchet it charges back exactly what the episode banked, so the trap's fall nets 0;
* PER-CALL ENERGY SHAPING (`--energy-pot K --energy-pot-front 0 --energy-pot-front-bonus 0`, the
  form pb_r26 was trained with at K 12): every loss of E = z + |v|^2/2g is charged when it
  happens and air acceleration is paid when gained. September's best pit arms (death charge +
  speed-keyed novelty, ledger round 41) reached the lip at the record's speed but bled ~30% at
  the second ramp and fell back; nothing in their reward charged the bleed.
Arm A1 `uf2r1b1_dce`: recipe + `--death-charge 1 --energy-pot 12 --energy-pot-front 0
--energy-pot-front-bonus 0` + September's view-free upward speed novelty at 4x (`--int-coef 1.0
--int-view 0 --int-speed 8 --int-speed-weight 3 --int-speed-weight-up`). Three changes at once,
on purpose: round 1 needs an exit, and the attribution comes after. Read with uf2_pit.py
(entry, E at the bottom, climb, out).
* Control final (`uf2r1b1_ctl`, 400M steps, 378k steps/s at the end with GE beside it for 25
  min): 3.128 / 4.910 / **5.608** (151M) / 5.568 / 5.454 / 5.546 (378M). All 9 greedy episodes
  every eval: the east fall onto `*18` at 5.0 s. Checkpoint pulled (ckpt_final.pt, 400M).
* A1 `uf2r1b1_dce` launched 22:03:47 UTC (6e8 steps), record gate passed.

### A1 `uf2r1b1_dce` (death charge 1 + per-call energy K 12 + upward speed novelty 4x): FROZEN, stopped at 129M
* Greedy 0.5M / 76M: 1.882 / 1.826 (control at 76M: 4.910). The 76M greedy wanders the platform
  for 45-106 s (E ~1,160-1,250, never below z 1,057) until the eval-patience cut: 0/9 pit.
* Training: progress record 576 u (1.9%) from 13M to 129M, reservoir min-depth 97.6% (2.4%),
  61-67% of episodes stall-killed, energy -640..-805 u/ep against +570..+720 (x K 12 = -8 reward
  per episode), novelty 0.04-0.11 per episode, yaw sigma 0.32 -> 0.75 (the pb "rot": no signal in
  the advantages). With the trap charged back and every landing charged at K 12, nothing on the
  map pays: standing still is the cheapest loss. Lesson: K 12 per-call energy is far too
  expensive to explore under; the novelty never got a foothold off the platform.
* A2 `uf2r1b1_nov` (22:13 UTC): September's entry recipe on the incumbent's - `--death-charge 1`
  + view-free upward speed novelty at 10x (`--int-coef 2.5 --int-view 0 --int-speed 8
  --int-speed-weight 3 --int-speed-weight-up`) - with per-call energy at the recipe's K 3
  (`--energy-pot-front 0 --energy-pot-front-bonus 0`): the novelty should dominate the energy
  term (~-2/ep) until the pit is entered.

### A2 `uf2r1b1_nov`, A3 `uf2r1b1_temp`, and a death-charge BUG (22:10-22:25 UTC)
* A2 (death charge 1 + upward speed novelty 10x + per-call energy K 3): 2.492 / 2.738 / 1.474
  (0.5 / 76 / 151M). At 76M the greedy walks off the platform's SOUTH edge after 9-13 s and dies
  in `*7`'s catch under it (z ~525): the right direction, too slow (the record leaves the edge
  at ~400 u/s and clears the catch by ~300 u). At 151M it stays by the spawn; record 723 u flat
  from 16M, novelty 0.05-0.14/ep, yaw sigma 2.2 at 207M. Stopped.
* A3 (A2 + `--yaw-entropy 0` + keys temperature cap 2, patience/period 2e7 on the reservoir
  reach - September's schedule): T 1.65 by 97M, yaw sigma held 0.35, 2.708 / 2.777 at 0.5 /
  76M. Stopped at 97M because of the bug below.
* **BUG (fixed in ee29bffb): `--death-charge` under `--race-ratchet` charged a death for the
  NEXT spawn's depth.** On the death call the row already holds the next episode's spawn, and
  the ratchet min-ed the record with it before the charge read it: a deeper next spawn - a
  reservoir row, or a `--respawn-backward-mix` goal-anchored row 256-1,532 u from the finish -
  made the dying episode pay for ground it never covered (old code: -9,000 for a 1,000 bank in
  the test). That is the -7..-12 reward per training episode in A1-A3 (10% of the non-start
  pool are goal-anchored rows): their deaths were charged a lottery, so A1-A3 say nothing about
  the death charge itself. The charge now reads the record before the death call (`_rec_boot`).
  September's death-charge arms had no goal-anchored rows and a reservoir near the start, so the
  gap was small there, but not zero.
* New: `--death-charge-window W` (s): a death takes back only what the ratchet paid in its last
  W s; progress the episode survived past for W s is kept. W 2 = the recipe's own
  `--respawn-margin 2` reading of a doomed tail; on this map it charges back the trap's fall
  (the last ~1.6 s) and keeps the platform's 1.7%. Test: tests/python/test_death_charge_window.py.
* A4 `uf2r1b1_dcw` (22:26 UTC): A3's flags with the fixed charge windowed at 2 s.
* A4 `uf2r1b1_dcw` first 175M (22:26-22:33 UTC): with the fix the training reward is positive
  (+1..+5 per episode), the critic's explained variance 0.91-0.96 (A1-A3: 0.1-0.2), stall-kills
  0-8% (A1-A3: 45-67%), yaw sigma 0.17-0.28 (no rot). Greedy evals: 2.763 / 4.805 / 3.059 at
  0.5 / 76 / 151M. At 76M 8/9 walk off the platform's south-east corner (x ~-1,640) into the catch
  at 3.25%, 1/9 takes the east trap (4.80); at 151M 9/9 walk off the SOUTH edge at x -1,880..-1,990
  (the record leaves it at x ~-2,240 at ~375 u/s) and die in the catch at z 540-573 at 3.6 s, E
  max 1,218 (no prestrafe gain). At 175M (T at its cap 2): novelty 0.2 -> 3.4 per episode and the
  mean dip depth at death 0.5-0.8 -> 2.97 (% of d0): training episodes are diving into new
  ground below the start.
* Measured from the record (ANALYSIS ONLY): it prestrafes to 272 u/s on the platform, bunny-hops
  to 341 at the curtain, leaves the south edge at ~375 u/s and air-strafes to 494 during a 1.4 s
  fall, passing over the catch at z 585 (the catch top 577) before it lands on the south ramp at
  t+1.5 (E 1,381 -> 1,295). A walk-off at 250 u/s falls short into the catch. The novelty's speed
  key (`--int-speed 8` = 500 u/s bins) cannot tell 250 from 494: nothing pays the launch speed.
* **A4 ENTERS THE PIT at 227M** (22:35 UTC, `traj_0227016704`): 9/9 greedy episodes dive off the
  south edge, enter the pit at t 3.35 s with E ~1,200, E max 1,369 on the way down (the record
  peaks at 1,381 there: the policy now prestrafes / air-strafes), 1,710 u/s in the pit (record
  1,822) - then bleed it in the bowl (E at the pit bottom -109..772 against the record's 1,117),
  lap and die on the pit floor's kill after 8-32 s; 0 out. Score 1.717 (a dive costs field
  progress; only an exit pays). Training at 243M: novelty 12/ep, reward +11, explained variance
  0.95, crawl 51%. September's entry stage in ~230M steps from scratch.
* Beam on A4's 257M weights (`beam_tas --score d --envs 4096 --resample-every 25`, 19 s on the
  box): no exit; the population's best d settles at ~30,26x for the last 10 s (the pit's climb
  side), never below the platform edge's 30,112.
* A5 `uf2r1b1_dcw2` (22:38 UTC): RESUME of A4's 279M checkpoint (md5 32089245...) with
  `--energy-pot 12` (per-call, pb_r26's K: the bowl's ~700 u bleed now costs ~-8 against novelty's
  ~+12) and `--unstuck-max 0.5` (the dive is in the greedy now; T 2 on the keys is noise in a
  precision bowl), the rest restored. Warm from its own run: the run measures AT the pit wall.
* A4 302M (its last eval): 9/9 in the pit and alive to the eval-patience cut (31.8 s), climbing to
  z ~0-17 at best, E at the pit bottom 261-956 (record 1,117). Score 1.749.
* A5 `uf2r1b1_dcw2` (K 12, T cap 0.5), 355M (76M after the resume): 9/9 in the pit, E at the
  bottom **811-912** (A4: mostly 260-600), vmax 1,732-1,749 (record 1,822); all 9 now hit the pit
  floor's kill at ~12 s at ~1,700 u/s instead of riding the bowl's walls. Training bleed -687 ->
  -448 u per episode, novelty 5.5/ep, explained variance 0.95. Score 1.759. K 12 holds the entry
  and buys energy; no exit.
* A6 `uf2r1b1_gsp` (22:44 UTC): resume of A5's 379.6M checkpoint + `--respawn-greedy-speed 1.0 1.3`
  (the recipe's front-perturb range, on the greedy pool's rows: its 2 s before each death and 200
  states of each episode's stalled stretch - here the whole dive and bowl, from the 1.9 s
  best-progress tick to the ~12 s floor death) + `--respawn-greedy-temp 1` (keys at 2 and yaw
  sigma x2 for 1 s after such a spawn): the policy's own pit states with up to +69% kinetic
  energy, so climbs out can happen in training and get valued.
* A6 455M: 9/9 in the pit at **record energy** - vmax 1,804-1,814 u/s (record 1,822), E ~1,050
  at the bottom (record 1,117) - but none U-turns: all fly straight up the pit's EAST side to its
  north end (y ~-460) and die on the floor's kill at 5.9 s (the record turns east at y ~-1,400
  and comes back south along x ~-1,850 - the greedy's north run is the record's return leg run
  backwards). The field does not pay that run (d rises 31.9k -> 33.9k northwards along it). Why
  it is a local optimum: per-call energy has NO death refund (dying while carrying energy keeps
  it, and the fatal tick's own loss is never charged: `dh[ended] = 0`), the time penalty favours
  an early death, novelty pays the fresh high-speed keys of the run, and nothing in the pit pays
  survival. From the TRUE start the ratchet's record is the platform edge (d 30,145), so nothing
  in the pit pays until the climb has fully exited; only pit-SPAWNED rows (record gap 0) are
  paid for climbing, and the climb has to transfer from the gap-0 context to a real dive's gap
  of 6-13% (obs column 27).
* **A6 530M: the greedy LAPS the pit** (`traj_0531103744`, 9/9 alike, 9.4-9.6 s): in at t 3.3 with
  E ~1,200 -> north up the EAST side (E 1,195 -> 982 in 0.8 s: the bleed) -> U-turn at the north
  end (y ~-500) -> south down the WEST side -> round the SOUTH end climbing (t 7.6: (-1,684,
  -3,127, -220), vz +377, E 911, d 30,866 vs the platform-edge record 30,129) -> north again ->
  the floor's kill at y ~-880. The record laps the other way and climbs out in a spiral around
  that same south end ~200-400 u higher with E ~1,080-1,117: the greedy is ~150-200 u of energy
  short of the climb, most of it lost on its first (east-side) run.
* Beam on A6's 553M weights (d-ranked, 4096 x 30 s): every lineage alive and lapping from 10 s
  on (d 32-33k); resampling on the current d does not drive a climb. No exit.
* A6 605M: a different mode - the greedy walks to the platform's SE corner, hops onto a ledge at
  z ~1,000 ((-1,664, -2,722) at t 2.7: d 29,966 = 2.2%, the score's 2.296), falls south into the
  pit's SE corner and dies on its floor at 4.8 s with E 1,215-1,285 and 1,877-1,906 u/s (more
  than the record's). The ledge's progress is banked 2.1 s before the death - just outside the
  2 s window: **any finite window can be beaten by surviving just past it** (or by waiting on a
  ledge for the stall-kill). It is worth 0.5% over the platform edge; an exit (>= 5.5%) beats it.
  The policy flips between the lapping mode (530M) and this one (kl 0.03-0.04).
* A6 ~675M greedy (`record_ckpt --dump-states`, 3 eps on the box): it now laps the pit ~6 times in
  30 s, COUNTER-clockwise (north up the east side, south down the west side), bleeding 60-80 u of
  E per lap (1,213 at the entry -> 520), highest at the north end (z ~-30..38). The record goes
  CLOCKWISE (north on the west side at the floor, U-turn, south on the east side), then spirals
  up the south end and the south-west wall to z 587 and flies north along the west rim. The
  novelty key (cell x speed, no heading: `--int-heading 0`) pays nothing for lapping the other
  way; the d-ranked beam culls exactly the record's line (it runs north into the dip first).
* Beam on A6's 675M weights ranked by `--score dh --score-energy 3`: all 4,096 lineages alive and
  lapping (d 32.5-33.5k), no exit.
* A6 final (757-908M): 9/9 lap the pit for 22-32 s every eval, E at the bottom 610-1,023, the
  highest climb z 110 (757M ep0, at 31 s); 0 out; scores 1.714-1.838.
* A7 `uf2r1b1_tick` (23:07 UTC): resume of A6's final checkpoint (979.9M, md5 280d1bc3...) +
  `--tick-ms-schedule 10:7.63:3e8` (the physics tick the record and the pb policies run at: 30%
  more air-acceleration impulses per second, CLAUDE.md `--tick-ms`) + `--respawn-greedy-temp 1`
  (not restored by a resume), 1.2B steps. Note: my `--death-charge-window` ring is sized at the
  launch tick (200 calls), so at 7.67 ms it spans 1.53 s, not 2.
* Queued idea: `--respawn-greedy-turn T` (the greedy rows' velocity and view turned together by
  U(-T, T) deg; the recipe's front-perturb rows use 90) to show the policy the CLOCKWISE lap the
  record's climb needs - the novelty key cannot pay a direction.
* A7 1,056M (tick ~9.5 ms mid-ramp): 9/9 dive with E 1,240-1,253 at the entry and 1,825-1,840
  u/s (the record's 1,822), but back in the quick-death mode (the floor at 6.1 s). The training
  reservoir's deepest state is still the trap's 3.94% (96.063%): no training episode has ever
  climbed out, so the critic has never seen an exit to value.
* A8 `uf2r1b1_turn` (23:14 UTC): resume of A7's 1,080M checkpoint (md5 de446323..., the tick ramp
  continues from its stored origin) + `--respawn-greedy-turn 180` (the greedy pit rows - the
  policy's own laps - turned to a uniformly random heading, so the clockwise lap the record's
  climb needs appears in training) + `--respawn-greedy-temp 1`.
* Plan-B probe (analysis): `tools/mpc_bench.py --objective height --tick-ms 10 --duration 4`
  (receding-horizon CEM over A/D + view offset, 1.2 s plan, replan every 4 ticks) from the
  policy's own pit states (A6 675M greedy, `--dump-states`): from the entry (tick 339, E 1,213)
  it climbs to z 75 in 1.6 s, then hits a surface (E 554 -> 93) and dies at 3.4 s; from tick 389
  (E 1,054) it stalls at z -572. A height-greedy MPC from these states does not find the climb.
* **A8: THE FIRST EXIT IN TRAINING (by 1,132M, ~23:16 UTC).** The training reservoir's deepest
  state went 96.06% -> **93.38%** of d0 (progress 3.94% -> 6.6%), the unstuck reach 840 u ->
  1,934 u, front-perturb episodes from it 2,286 u (7.5%). The one deep reservoir state: a
  STANDING state on a ledge north of the pit, (-2,026, -1,196, 382), progress 6.55% (the record's
  exit passes ~300-500 u west of it at z 354-504). Only real episodes are harvested (novel and
  goal-anchored rows are not), so a training episode climbed out of the pit - most likely a
  turned and speed-boosted greedy pit row. The greedy itself (1,156M) still laps for 30 s, now
  with E ~1,125 at the bottom (record 1,117) and climbs to z ~-230..-300 at 7.9 s. Score 1.926.
* Beam on A8's 1,181M weights from the TRUE start (d-ranked, 4096, 29 s at the 8.4 ms mid-ramp
  tick): every lineage laps (best d 30.5-31.7k), no exit - the training exit came from a
  perturbed (turned / boosted) pit row, not from the policy's own line. The ledge (-2,026,
  -1,196, z 382) is above the north part of the pit's EAST side - where the greedy's
  counter-clockwise lap climbs (its highest points z ~-30..38 nearby, E ~1,000 there; standing on
  the ledge needs ~400-600): the lap the greedy already flies may lead to this exit.
* A8 1,307M (the ramp done: 7.67 ms): the greedy's pit energy now BEATS the record's - in at E
  1,301-1,304 (record 1,294), at the bottom 1,215-1,226 (record 1,117), 1,828-1,865 u/s - and it
  laps higher: south down the west side (z ~-830), round the south end climbing, then north up
  the EAST side at z -16..-61 with E ~1,240, west at the north end and down again; alive to the
  26.6 s cut. Closest approach to the training ledge 323-467 u at E 1,120-1,282 (the ledge needs
  ~300-400): a steering gap, not an energy gap. Score 1.956.
* The training frontier is a LEDGE: 4,479 of the reservoir's 20,000 states (22%) stand on it at
  (-1,970, -1,075, z ~300), progress 6.95%, |v| ~0 - climbed out by perturbed pit rows, then
  waiting for the stall-kill keeps the record (the finite window's exploit again). A beam from
  the true start on the 1,353M weights still finds no exit (best d 30,081 at 19.9 s).
* **Why the greedy does not follow the training exit (field + geometry, 00:00 UTC):** at the
  ledge's x, y the field jumps from d 32,874 at z 100 to 28,480 at z 150 - the pit's whole
  NORTH part is roofed by the `*18` kill sheet (z 161), and the ledge (z ~300) stands ABOVE it.
  The training exits climbed out at the pit's SOUTH end (no sheet there, y < -2,400) above z
  ~200 and flew north over the sheet onto the ledge - the record's kind of exit. The greedy laps
  under the sheet (its north-end peaks z ~29-110, the sheet's underside) and at the south end
  turns north at z ~0-50 with E ~1,150-1,235 (the record climbs the same south-east ramp to z
  320 and the west side to 587 with E ~1,100): a steering gap. Trend of the greedy's first-lap
  south-end peak (median of 9): z -400 (980M) -> -226 (1,156M) -> -1..47 (1,383-1,458M) -> -323..31
  (1.5-2.1B), single episodes at z 317-458 (1,232M, 1,836M, 1,987M, 2,138M) - i.e. at the record's
  climb height, none turning north there yet.
* Beams on A8's 2,063M weights - critic-ranked (`--score v`), d-ranked with `--keys-temp 2
  --view-temp 2`, and d-ranked from the greedy's own exact south-end states (`--start-state`,
  ticks 829 / 879 / 904 of a dumped greedy episode) - none climbs out (best d ~30.5k): the
  policy's proposals do not contain the higher line yet.
* A9 `uf2r1b1_turn2` (~00:02 UTC): A8 continued unchanged (2.4B steps) - the climb is converging.
* The greedy's highest south-end climbs (z 378-458; 1,836M ep2 at t 10.5 s, 1,987M ep3 at 10.6 s,
  2,138M ep3 at 13.9 s) come on LATER laps with E ~950-1,070 (bled on the earlier laps) and peak
  heading EAST (vz ~0, |v_h| ~950-990), then fall back into the pit. The record climbs on its
  FIRST lap with E ~1,100 and turns NORTH at z 485-587 to cross over the sheet. What is missing:
  the high climb on the first (full-energy) lap and a north turn at its top.
* Monte Carlo from the TRUE start (`beam_tas --elite-frac 1.0`: 4,096 independent rollouts, no
  cloning) on A8's final weights: keys-temp 1 -> best d 30,020; keys-temp 1.5 -> 29,987 at 30.6 s:
  one lineage climbed round the pit's south end on its 4th lap onto the TOP of the solid block
  south of the platform ((-2,577, -2,763, 508), E 1,200, heading east) - the record crosses ~110 u
  above that block (y -2,500..-2,700 is solid up to z ~475 on the field) and drops north into the
  open air above the sheet (y ~-2,300, z >= 175). Beams rooted at that lineage's states just
  before (`--start-state`, d-ranked, keys-temp 1.5, view-temp 1 / 3 / 6): none crosses (best d
  ~29,900 at 0.8 s, then back into the pit).
* A10 `uf2r1b1_cur` (00:31 UTC): RESUME of A9's 2.36B weights + a BACKWARD CURRICULUM on that
  Monte Carlo line (`autoresearch/spines/uf2r1b1_mcline.npy`, the policy's own rollout - provenance
  in spines/README.md): `--demo-file ... --demo-success-d -400 --demo-success-bonus 50
  --demo-window 5 --demo-grow 5 --demo-grow-max 40 --demo-cooldown 5 --demo-front-frac 0.5
  --demo-front-temp 0.5 --demo-front-ticks 40 --demo-front-view 6` (the pb playbook; success =
  the record 400 u past the line's end = across the block, over the sheet) + `--no-respawn-novel
  --respawn-backward-mix 0 --int-real-only 0` (both refuse a demo spine) + `--respawn-greedy-temp 1`.
* **A10 at 2.87B (00:29 UTC, ~20 min in): training flies out of the pit.** The reservoir's deepest
  state 93.05% -> **90.71%** of d0 (9.3% progress); 151 reservoir states past 5%, most of them
  FLYING above the sheet north at ~1,170-1,190 u/s with E 1,180-1,350 (e.g. (-2,069, -1,432, 455)
  E 1,354; (-743, -1,108, 325) at 9.29% heading north-east), some at z ~700 with E ~1,800-2,000;
  front-perturb episodes reach 4,221 u (13.8%). 164 spine-line bonuses paid (records past d 29,584),
  though the curriculum's own frontier-band success is still 0% at tau 3999. The greedy: 9/9 dive
  with E 1,200+ at the pit bottom (peaks 1,403-1,444); one episode (2,868M ep4) climbed out of the
  pit's EAST side to (-1,503, -2,512, 583) with E 1,306 at 24 s - 3.73%, the greedy's first
  progress past the platform edge - and died there (the catch's top is z 577). Score 3.729.
* **THE GREEDY CLIMBS OUT OF THE PIT: 8.846 at 2,944M (A10, 00:32 UTC, ~23 min into the
  curriculum).** `traj_2943877120`: best 8.85%, median 6.5%; 7/9 episodes past 6% after leaving
  the pit at 13-18 s - ep1 / ep3 die at (-2,480, -1,353, 182) / (-2,456, -1,215, 180), i.e. ON the
  `*18` sheet right under the record's S19 touch ((-2,571, -1,357, 504) at t+8.5): across the
  block and north, but too low; ep6 reaches 8.85% at (-1,064, -1,270, 183) (the sheet, further
  east); ep0 / ep2 leave by the pit's east side into the catch (4.0-4.1%). The lineage's first
  gate ("out of the pit from the true start, flying past 5%"). Checkpoint pulled: ckpt_latest at
  2,985M (41M after the eval; no milestone before it).
* A10 to 3,456M (stopped 00:55 UTC): evals 7.378 / **9.126** (3,095M, median 7.9%) / 6.453 /
  4.047 / 6.673 / 6.672 - the greedy exits every eval and dies within a few % after. The
  curriculum walked all the way back: tau 0 (the true start) with 95.8% success over 500 eps
  (success = the record past d 29,584). The TRAINING frontier is ~20% of the map: the
  reservoir's deepest 79.31% of d0, front-perturb best 6,411 u (20.9%), training reward ~59/ep.
  The greedy lags the stochastic training line by ~10% of the map.
* A11 `uf2r1b1_auto` (00:56 UTC): resume of A10's 3,456M weights (md5 44f472af...) + the
  pb lineage's search -> curriculum loop: `--auto-spine --auto-spine-async --auto-spine-milestones
  --auto-spine-gain 500 --auto-spine-ge-minutes 0 --auto-spine-delay 1e8` (a beam from the TRUE
  start on the newest weights after each eval whose quorum crosses the spine's success line; a
  line that beats the greedy frontier by 500 u becomes the next spine; the old success line
  keeps paying as a milestone) + `--ckpt-every 2e8` + `tools/snapcopy.py` beside it (a hand-on
  near every eval); A10's curriculum flags with `--demo-tau0 0`.
* **A11: 12.229 at 3,683M (01:05 UTC) - the auto-spine loop works here.** Search 1 (at 3,608M,
  beside the training, 114 s): a NEW SPINE from the beam on the newest weights from the TRUE
  start (2,900 states, end d 27,763, 918 u past the greedy frontier 28,681); success line d <=
  27,363; the old line 29,584 kept as a milestone. The next eval: 5/9 greedy episodes reach
  11.2-12.2% at ~29 s around (-3,061..-3,650, -140..-439, -740) - past the `*18` sheet's WEST edge
  on the record's own route (the record: (-3,382, -787, -220) at t+10, (-3,161, 712, -591) at t+11)
  - and die on the `*8` sheet (z -767..-768) where the record surfs higher (z -591 -> -102);
  3/9 still die in the pit at 7-8 s, 1 on `*18` (7.87%). Search 2 started (5/9 past 27,363).
  Hand-on snapshot pulled: ckpt_3662675968 (20M before the eval; snapcopy).
* A11 3,759M: **12.524** (best 12.5%, median 12.1%) - 9/9 greedy episodes alike: out of the pit
  at ~11 s, then 11.1-12.5% at ~16.5 s round (-3,450, -15), dying on the `*8` sheet at z ~-746
  (the record surfs ~250 u higher there: (-3,295, -86, -498) at t+10.5). 3,834M: 11.672.
  Search 2 (3,683M -> 3,779M, 375 s): NO new spine - the beam's frontier d 26,842 against the greedy
  frontier 27,220 = +378 u, under the 500 u gain (the trainer's default gain is 200: with it this
  spine would have been taken). Training: the reservoir's deepest 59.32% of d0 = **40.7%**,
  front-perturb best 12,478 u (40.7%) - the stochastic frontier is 3x the greedy's.
  Hand-on snapshot pulled: ckpt_3749183488 (9M before the eval; md5 8771e806...).
* **A11 3,985M: 38.339 (01:20 UTC)** - best 38.3%, median 34.5%: 8/9 greedy episodes leave the
  pit (~10.6 s), cross `*18`, pass its west edge and fly on to 32.8-38.3% by ~24-26 s round
  (530..2,510, 3,290..3,610) - the record's ~t+16 region (31.3% at t+15) - then fall (z ~-860);
  1/9 back on `*18` (6.5%). The jump came between the 3,910M eval (12.457) and this one, with no
  new spine (search 3 had not run): the policy itself extended past the `*8` sheet.
  snapcopy had silently stopped copying at 01:12 (3,857M; restarted 01:21), so the snapshots
  either side of the eval are 3,857M (pre-jump) and 4,050M (saved by hand, md5 66760a96...; its
  own fresh 9-episode greedy recording: 23.85% best / 23.6% median). Both pulled.
* A11 4,061M: 23.725 (median 23.6%); **4,136M: 39.090** (best 39.1%, median 35.5%): 9/9 out of the
  pit by ~10 s, 30.8-39.1% at 23-28 s - the 38% level holds, not a fluke. Hand-on snapshot pulled:
  ckpt_4120903680 (15M before, md5 fdaeb1e2...).
* Search 3 (-> 4,161M, 91 s): NEW SPINE from the beam on the newest weights (3,500 states, end d
  17,676 = 42.3%, the greedy frontier 19,765): success line d <= 17,276; milestones 29,584 and
  27,363 keep paying. The loop now has three spines; the curriculum restarts at the new end.
* A11 4,212-4,514M: 38.489 / 35.846 / 39.248 / 21.426 / **41.668** (4,514M: best 41.7%, median
  39.1%; 9/9 at 35.6-41.7%, dying in the `*9` kill volume (z <= -592) round (1,200..3,500,
  2,300..3,600) where the record climbs east out of it: (162, 3,593, -486) at t+15 -> (2,735,
  622, 461) at t+20). Hand-on: ckpt_4509925376 (3.7M before; md5 7e40760e...). Training frontier:
  the reservoir's deepest 49.96% of d0 = **50%**, front-perturb best 16,119 u (52.6%).
  The k03 curriculum is stuck at its end (0% of band spawns 400 u past d 17,676) and no greedy
  episode crosses its success line (43.6%), so the loop does not search again (no
  --auto-spine-patience set).
* A11 4,589-4,740M: 39.595 / 35.864 / **44.455** (4,740M: best 44.5%, median 39.4%; ep8 past the
  `*9` climb to (3,277, 1,419, -189), 44.5%, on its way to the record's t+20 point (2,735, 622,
  461) = 48%; 8/9 at 37-40%). 1/9 past the k03 success line (43.6%): the quorum is 5. Hand-on:
  ckpt_4738514944 (1.6M before; md5 430b0d0c...). Training: the reservoir's deepest 44.66% of d0 =
  55.3%, front-perturb best 17,070 u (55.7%).
* **A11 4,816M: 47.982** (best 48.0%, median 45.9%): 9/9 at 44.8-48.0% round (2,600..3,300,
  700..1,600, z 217..811) - the record's t+19..20 stretch (48% at t+20) - dying at z ~-185 beyond.
  Hand-on: ckpt_4815585280 (= the eval step; md5 f7578dee...). Search 4 (46 s): NEW SPINE (5,091
  states, end d 13,665 = 55.4%, the greedy frontier 16,572); success line d <= 13,265;
  milestones 29,584 / 27,363 / 17,276. Four spines so far: 1.9% (the MC line) -> 9.4% -> 42.3% ->
  55.4%.
* **A11 4,891M: 50.801** (best 50.8%, median 49.5%): 6/9 reach 49.5-50.8% at ~27 s round
  (3,320..3,450, -380..-30) and die at z ~-186 (the bottom of the `*10` trigger, z -224); 2/9 at
  46%, 1/9 at 24.7%. 4,967M: 47.477. Hand-on: ckpt_4874305536 (17M before; md5 e88fd75b...).
  0/9 past the k04 success line (56.7%) yet.
* A11 5,193-5,420M: 48.312 / 49.864 / 52.461 / **53.590** (5,420M: best 53.6%, median 53.0%;
  7/9 at 50.9-53.6% round (2,710..2,900, -150..-830) after 38-48 s - slower than the 4.9B line -
  dying on the `*11` sheet at z ~-1,023). Hand-on: ckpt_5408030720 (11.5M before; md5
  892cd43b...). Training frontier: the reservoir's deepest 32.50% of d0 = 67.5%, front-perturb
  best 20,905 u (68.2%). 0/9 past the k04 success line (56.7%) in four evals.
* A11 5,495-5,646M: 51.633 / 45.195 / 46.698; stopped at 5,668M (02:33 UTC). **The k04 curriculum
  never left the spine's last state** (tau 5090/5091, 0% success over 100 band episodes): my
  `--demo-success-d -400` asks for 400 u PAST the line's end, and the beam's line is cut at its
  best d - a state from which 400 u more is often impossible - so the curriculum never walks back
  along the line and half the demo draws are spent on one dead state; the quorum (5/9 past 56.7%)
  never forms, so the loop stops searching. -400 was right for the first spine (it had to force
  the crossing out of the pit); for the loop's spines the pb playbook's `--demo-success-d 100`
  (success = within 100 u of the end) is the rule: the curriculum walks the whole line back and
  the quorum fires once the greedy reaches the end.
* A12 `uf2r1b1_auto2` (02:35 UTC): resume of A11's 5,668M weights (md5 6d8e0110...) + the k04
  spine (`autoresearch/spines/uf2r1b1_k04.npy`, the trainer's own search 4) with its keys,
  `--demo-success-d 100`, `--demo-milestones 29584,27363,17276`, the same auto-spine loop, 8e9
  steps (to the wrap-up); snapcopy beside it. Box extended to 09:32 UTC.
* **A12 5,820M: 55.779** (best 55.8%, median 46.3%): ep4 reaches the k04 spine's end region
  ((2,095, -562, -201), 55.8%, at 44 s); 6/9 still stop at the ~46% gate (3,150..3,450, 1,000..1,140,
  z ~790), 1 at 51.6%. With `--demo-success-d 100` the curriculum WALKS BACK at once: tau 5090 ->
  3035 in ~150M steps, 94% success over 312 band episodes. Hand-on: ckpt_5807538176 (12M before;
  md5 88e3458a...).
* A12 5,895M: 55.508 (median 55.5%); **5,971M: 57.749** (best 57.8%, median 57.5%): 7/9 die at the
  SAME spot ~31 s, (1,530..1,650, -300..-355, -331), with E 54-89 - an ENERGY gate: the line
  arrives with ~1,300 u of E at 50% and bleeds it all by 57% (the record passes this stretch
  with E ~1,000: (1,586, 481, -726) at t+25 = 63%). A12's search 1 (5,909M, 55 s): new spine
  (4,585 states, end d 13,132 = 57.2%); 7/9 past its line -> search 2 running. The k01 curriculum
  at tau 3549 with 99.8% success. Hand-on: ckpt_5961678848 (9M before; md5 915931b4...).
* **A12 6,046M: 66.202** (best 66.2%, median 57.4%): through the 57% energy gate in 4/9 - ep0 to
  66.2% at (1,393, 1,909, -430) at 37 s, three at 61.5-62.5% (the record's ~t+25 stretch: 63% at
  (1,586, 481, -726)) - dying on the `*12` floor at z ~-1,640; 2/9 still at 57%, 3/9 early (9%,
  9%, 23%). Search 2 (169 s): no new spine (beam 12,994 vs greedy 13,013). Hand-on:
  ckpt_6030360576 (16M before; md5 1786148f...).
* A12 6,122M: 66.043 (median 62.3%); **6,197M: 68.707** (best 68.7%, median 67.9%): 9/9 past the
  57% gate; 7/9 at 67.0-68.7% round (870..1,280, 1,560..1,880, z ~0) at ~37 s, then down onto the
  `*12` floor (z ~-1,640). Hand-on: ckpt_6176112640 (21M before; md5 9e2ddd98...). **Training
  reaches the end zone**: the reservoir's deepest state 5.50% of d0 = **94.5% progress**,
  front-perturb best 29,204 u (95.3%); training reward 186/ep.
* A12 6,273-6,726M: 9 evals at 66-69%, best **69.163** (6,726M: best 69.2%, median 67.9%): the
  greedy holds ~68% (the `*12` floor at z ~-1,640 after 67-69%). **Search 3 (6,359M, 44 s): NEW
  SPINE from the beam on the policy's own weights from the TRUE start: 5,200 states, end d 4,782 =
  84.4% of the map** (the greedy frontier 9,845 = 67.9%); success line d <= 4,882; the curriculum
  (tau 4759/5200 at 6.75B, 37% band success) teaches it from its end. Training's reservoir
  frontier 94.7%, front-perturb best 29,698 u (96.9%). Hand-on: ckpt_6717702144 (8M before; md5
  178214a5...).
* A12 6,800-7,330M: 69.059 / 69.057 / 68.777 / 67.853 / 68.985 - the greedy holds 68-69% for ~1B
  steps. The k03 curriculum (the 84.4% beam line) STALLS at its end: band tau 4744-4759 of 5,200
  (~71%), success 37% -> 17%; success = reaching the line's end through 4800-5200 (72.6% -> 84.4%),
  which the beam itself flew at E ~250-290 (a barely-surviving stretch). The greedy follows that
  line to ~rows 4,550-4,600 (67-69%) and fails; nothing teaches 68-72%.
* A13 `uf2r1b1_cut` (03:53 UTC): resume of A12's 7.38B weights with the k03 line CUT at row 4800
  (d 8,408 = 72.6%; success line 8,508 - ~3.5% past the greedy) - the `--demo-stall-cut` rule
  (pbr6b4) applied by hand: cut where the backward curriculum stalls past the greedy's frontier;
  milestones 29,584 / 27,363 / 17,276 / 13,765 / 13,232; the same loop; 6.5e9 steps (to the wrap-up).
* **A13 7,783M: 86.071** (best 86.1%, median 85.6%; 04:07 UTC): the cut worked. The first evals
  after the restart collapsed while the new curriculum moved the policy (1.794 / 9.057 / 1.900 /
  51.237 at 7.48-7.71B), then 7/9 reach 85.5-86.1% at 41-48 s round (530..680, -2,550..-2,830,
  -160..-440) - ~2,700 u west of the finish box (x 3,328-3,936, same y band) - and fall to a floor
  at z ~-1,350; 2/9 still fail at the pit's east side (3.6%). 7/9 past the cut's line (8,515) ->
  search 1 running. The curriculum walked back to tau 2299 of 4,800 within ~400M steps.
  Hand-on: ckpt_7780433920 (2.6M before; md5 4ba632fe...).
* **A13 7,859M: 88.920** (best 88.9%, median 85.6%): 9/9 out of the pit and at 84.9-88.9%; ep7 to
  (1,628, -2,797, -345) at 54.8 s, ~1,700 u west of the finish box's west edge (x 3,328). Search 1
  (115 s): NEW SPINE (6,500 states, end d 3,376 = 89.0%); success line 3,476; milestones now
  29,584 / 27,363 / 17,276 / 13,765 / 13,232 / 8,515. Training: the reservoir's deepest 1.42% of
  d0 = **98.6%**; a front-perturb episode reached d ~0 (30,648 u), no finish counted yet.
  Hand-on: ckpt_7837057024 (21.5M before; md5 57f63cd8...).
* **A13 7,934M: 94.570** (best 94.6%, median 93.7%, 0/9 finishes; 04:14 UTC): 9/9 out of the pit
  and at 88.9-94.6%. Five of nine reach 93.7-94.6% at ~44-45 s round (1,720..1,905, -1,265..-1,475,
  -760..-850), ~1,480 u west of the finish box (x 3,328), lose the line and fall to the z ~-1,350
  floor by 46-49 s; three die at 90.2% round (1,670..1,730, -2,320..-2,370, z -1,350); one at 88.9%.
  Training: the front-perturbed rows FINISH (fpert fin 90/142, best 30,648 u = 100%), training win
  rate 11-14% at 0.4 s (rows harvested 2 s from the goal - the harvest, not the policy); the
  reservoir's deepest 1.195% of d0. Curriculum on k01 at tau 5,484-5,799 of 6,500, 93% success.
  9/9 past k01's line (3,476) -> search 2 (73 s): NEW SPINE `auto/k02` (5,800 states, end d 386 =
  **98.7%**, md5 6f135705...): the beam bounces off a ramp at (1,842, -1,400, -951) (vz +531, where
  the greedy now falls), runs south to (2,384, -1,515, -625) and ends at (2,923, -2,052, -753) moving
  (1,114, -585, -472) - BELOW the finish box's floor (z -640) and falling: it passes under the box.
  The beam's population died below d ~1,100 after tick ~6,000 (min_d 1,101 -> 1,594): no env of
  4,096 found an entry into the box from that approach; the finish needs the box entered at
  z >= -640. Milestones + 3,476; success line 486. Hand-on: ckpt_7934050304 (the eval's own step;
  md5 a11560b1...). k01 / k02 copied to `autoresearch/spines/uf2r1b1_cut_k0{1,2}.npy` (+ keys).
* **A13 8,010M: 99.094** (best 99.1%, median 98.5%, 0/9 finishes; 04:20 UTC), one eval after k02
  took over: five of nine at 98.5-99.1% (3,133..3,253, -1,739..-1,810, -893..-1,052) at ~45 s -
  171 u west of the box's x edge and 120 u north of its y edge but **250-410 u below its floor**
  (z -640) - then drift back south-west falling to the z ~-1,350 floor; 2/9 die at 72-73% round
  (-134..-208, 700..1,010), 1 at 89%, 1 in the pit (1.8%). Training: reservoir mind 0.379%
  (d ~116 - the under-box approach), and the front-perturbed rows that FINISHED 90/142 at 7.98B
  now finish **0/71-74**: the k02 curriculum's harvest filled the reservoir's front with the
  doomed under-box states (small d, no way up). The geodesic field ranks "under the box" next to
  "in the box" - the same deception as the cannonball descent. Hand-on: ckpt_8005877760 (3.7M
  before; md5 2a5f39d1...).
* **A13 8,085M: 99.689** (best 99.7%, median 99.4%, 0/9 finishes; 04:22 UTC): 8/9 at 99.28-99.69%,
  the best to (3,281, -1,834, -781) - 47 u west of the finish box's west edge, 54 u north of it,
  **141 u below its floor**; then they overshoot east along the box's north side (to x ~3,900) or
  turn back west, and fall to the z ~-1,350 floor. The field reads d 0 inside the box and 34,659
  60 u under its floor: the box sits ON a solid block and must be entered from above. Hand-on:
  ckpt_8080850944 (4.2M before; md5 ea634269...).
* Search 3 (8.01B, 267 s): NO new spine - beam frontier d 349, branched beams at 5,615 / 5,354 /
  5,121 (energy forks before losses of 218 u at 5,187 and 114 u at 5,475) d 342-379, 0 finishes in
  4,096 envs each, the population extinct by t 6,000. **And the loop cannot hand back a finish any
  more**: a spine must end `--auto-spine-gain` (500) u deeper than the greedy frontier, which is
  now 177 u from the goal - even a FINISHING line (d 0) fails `0 < 177 - 500`. A defect of the
  gain rule near the goal (a finisher should always beat a non-finisher).
* ANALYSIS (the record, insight only): it climbs the same ramp (1,790..1,940, -1,580..-1,190) at
  35.1-35.4 s with vz +320..+780 and arcs east at z -625..-448 into the box's west face at
  (3,334, -2,125, -510); at x 2,384 its energy height is ~556 against the k02 line's ~324 - the
  policy comes off that ramp ~230 u of energy short, which is what puts it under the box floor.
* Manual search m1 (04:31 UTC, beside A13 on the 8.081B weights): auto_spine's ENERGY stage alone
  (`--energy-gain 150` = the pbr5b2 constant, `--ge-boost ""`, GE 8 min, no branches): roots
  before the line's largest energy losses, a uniform explorer keyed on position x energy, the
  deepest cell keeping >= 150 u more energy than the line at the same progress.
* **A13 8,161M: 118.500 - 9/9 GREEDY FINISHES from the true start** (45.00-45.54 s; 04:40 UTC).
  Every episode dives into the start pit, climbs out, flies the whole course and enters the box
  through its west face at (3,303..3,312, -1,926..-1,980, -526..-594), 45-115 u above its floor -
  ONE eval after the greedy passed 141 u under it (99.689 at 8.085B). No new spine did it (search 3
  found none; the gain rule could not have handed one back): the k02 curriculum walking back
  (tau 2,394 of 5,800 at 8.23B, 84% success) and the training's own finishes (fpert fin 90/142 at
  7.98B; win 74% at 8.23B) taught the last 141 u of height. Hand-on: ckpt_8152154112 (8.4M before;
  md5 d8d47113...), VERIFIED by its own greedy recording on the box (record_ckpt, 9 episodes):
  **9/9 finishes, 44.74-45.52 s, score 118.509** (`runs/uf2r1b1_cut/verify/greedy_8152154112.jsonl`).
  Next: the time (record 35.17 s; the greedy is ~10 s slower).
* Manual search m1 (beside A13 on the 8.081B weights, 04:23-04:31): its beam reached d 63 (no
  finish); the energy stage had started its GE (roots before losses of 227 / 216 / 106 u at ticks
  4,160 / 5,144 / 5,428) when the greedy finished on its own - stopped (kill by pid).
* ANALYSIS (the record as an instrument only): split times by best field progress, the record
  vs the 8.152B greedy (median of 9): 2% 8.33 vs 11.88 s (**+3.55 s in the start pit** - the
  greedy's energy laps vs the record's direct exit), 40% 18.88 vs 23.69 (+1.26 over 2-40%), 60%
  25.26 vs 31.23 (+1.17), 70% 29.29 vs 36.88 (**+1.61 over 60-70%**), 90% 34.38 vs 42.68 (+0.71),
  100% 36.76 vs 45.16 (+0.10 on the final approach) - 8.4 s in all, 42% of it in the pit. The
  score's time term is 10 x (1 - t/300): the record's pace is worth +0.33 over 45 s; 9/9 is worth
  far more than speed (one failure costs 1.1), so the hand-on is always the best EVAL.
* A13 8.16-8.92B, eleven evals: 9/9 at 8.161 (45.00 s) / 8.236 (44.77) / 8.312 (44.97) / 8.538
  (45.36) / **8.840B (43.98 s, 118.534)**; 8/9 at 8.463 / 8.689 / 8.765 / 8.916; **0/9 at 8.387B**
  (all nine die in the start pit at 1.8%) and 2/9 at 8.614B (seven die there at ~1.7%): the pit
  entry is still the fragile part, the rest of the course holds. The curriculum walked back to
  tau 674 of 5,800 by 8.97B; search 5 (8.85B, greedy_d 0) found no spine (as it cannot).
  Hand-on for 118.534 = ckpt_8819048448 (21M before the eval; the next snapshot is after it),
  VERIFIED by its own greedy recording on the box: **9/9, 44.44-44.96 s, score 118.519**, into the
  box 130-200 u above its floor (`verify/greedy_8819048448.jsonl`; md5 d772ebb4...).
* The two bad evals (analysis): at 8.387B (0/9) all nine dive into the pit with full energy
  (E 1,294 at the entry, ~1,100 at the bottom) and then LAP in it without taking the exit until
  the 30 s eval-patience cut (best 1.77% at 1.66 s); at 8.614B (2/9) four mis-dive into a catch at
  (-2,272, -3,084, 232) at 2.48 s, two lap, one dies at 64.6%. The rest of the course holds in
  every eval after 8.16B: the start pit (its entry and its exit decision) is where this policy
  is fragile AND where it is slow (3.55 s of the 8.4 s). Both are round 2's problem.
* Not run: the expert loop (`tools/exit_local.py`) for speed - the grader counts only the
  trainer's own `traj_<step>.jsonl` evals, the loop trains with evals off, and its gain on
  princessburglar (64.15 -> 62.11 s over many rounds) is worth ~0.07 here at 0.033 per second.
  A13 keeps training (its own evals are scored) to the wrap-up.
* A13 8.99-9.67B: 9/9 at 8.991 (44.99 s) / 9.067 (44.41) / 9.142 (44.05) / 9.293 (43.92) / 9.369
  (44.14) / 9.444 (43.83) / **9.670B (43.51 s, 118.550)**; 7/9 at 9.519 / 9.595, 5/9 at 9.218.
  The curriculum reached tau 0 (the whole k02 line from the true spawn) by 9.76B.
* Hand-on audit (greedy re-recordings on the box, record_ckpt 9 episodes): ckpt_9661579264
  (8.9M before the 118.550 eval) **8/9** (one dies at 64.4%), best 43.65 s; ckpt_9637986304
  **6/9**, best 44.02 s; ckpt_8819048448 **9/9 again with --seed 7** (best 44.54 s; 18/18 over
  two seeds). The late policy's evals overstate it: 9/9 evals sit between 6/9-8/9 snapshots.
  So the 9.6B snapshots are NOT pulled, and the hand-on for the 118.550 eval stays
  ckpt_8819048448 (the newest pulled snapshot at or before it; PROTOCOL's rule) - 1 s slower,
  reliable. RESULT.json says so.
* **FIXED (58574f32): the loop could never hand back a FINISHING line.** Two defects:
  `beam_tas --save-frontier` wrote `beam_frontier.npz` only when the search did NOT cross (a beam
  that finished left auto_spine no line: search 5's `{"beam_rc": 0}` with no frontier fields), and
  `de < greedy_d - gain` rejects every line, a finisher included, once the greedy is within
  --gain of the goal. Now a crossing also saves the finishing winner (states to the pre-finish
  tick, `finished` 1), and `auto_spine.beats_greedy` lets a finishing line beat a greedy whose
  median episode does not finish. Unit tests + a box test on ckpt_8819048448: the patched beam
  finished in **43.18 s** (8,190 finishes in 4,096 envs, bit-exact), 5,632 states, handed back at
  greedy_d 177 / gain 500, refused at greedy_d 0. A13 itself runs the old code (deployed
  ee29bffb); it did not need the fix (its greedy closed the last 141 u itself).
* The policy's own beam (4,096 envs, 8.819B weights) finishes in 43.18 s against its greedy's
  44.4-45.0 s: a search line ~1.3 s faster exists in its own distribution - the expert loop's
  (beam -> BC) starting point for round 2.
* A13 9.75-10.20B: 9/9 at 9.746 (44.02 s) / 9.821 (43.72) / 9.897 (43.56) / 10.048 (43.70) /
  **10.123B (42.62 s, 118.579)**; 8/9 at 9.972 and 10.199. The snapshot before the best eval is
  ckpt_9980346368 (143M before: snapcopy copied nothing 05:54-06:04 - it copies only when
  ckpt_latest's mtime moves, and PRUNES its own copies after 45 min, so a candidate must be
  pulled within 45 min) and it re-records **8/9** (best 43.03 s). Every late snapshot re-recorded
  so far is 6/9-8/9 while the evals around it read 9/9 (both seed their spawns from the step,
  `step & 0x7FFFFFFF`, so each is another draw): the late policy finishes ~90% of its episodes,
  the 8.819B one 18/18.
  Hand-on stays ckpt_8819048448.
* A13 10.20-10.95B: 9/9 at 10.274 (43.13 s) / 10.350 (42.90) / 10.425 (42.73) / 10.576 / 10.652 /
  10.727 (42.73) / 10.878 / **10.954B (42.58 s, 118.581)**; 8/9 at 10.199 / 10.501; **2/9 at
  10.803B** (seven die in the start pit at 1.7%). The time has flattened at 42.6-43.1 s. The
  snapshot before the best eval, ckpt_10944512000, re-records **8/9** (best 43.25 s): the hand-on
  stays ckpt_8819048448.
