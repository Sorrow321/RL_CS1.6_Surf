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
* A13 11.03-12.39B: 9/9 at 11.029 / 11.180 / 11.256 (42.57 s, ties 118.581) / 11.482 / 11.633 /
  11.859 / 12.011 / 12.086; 8/9 at 11.104 / 11.407 / 11.558 / 11.708 / 11.784 / 11.935 / 12.162
  / 12.237; **0/9 at 11.331, 12.313 and 12.388B** (all nine lap or die in the start pit at
  1.7-1.8%). No gain in speed after ~10.4B (42.6-43.5 s), and the whole-eval pit failures come
  more often late: long training on this mixture does not stabilise the pit.
* **A13 12.46-13.29B: the late policy steadied.** 9/9 at 12.539 / 12.615 / 12.766 / 12.841 /
  12.917 (42.45 s) / 12.992 (42.47) / **13.068B (42.25 s, 118.592)** / 13.143 / 13.219 (42.35) /
  13.294 - seven 9/9 evals in a row from 12.77B; 7/9 at 12.464, 8/9 at 12.690. The snapshot
  before the best eval, **ckpt_13045858304, re-records 9/9 TWICE** (default seed best 42.07 s,
  score 118.598; --seed 7 best 42.13 s): the first late snapshot to do so. Pulled (md5
  359ab38c...) - the new hand-on, 18/18 like ckpt_8819048448 and ~2.4 s faster.
* **A13 ended at its budget, 13,905,166,336 steps (08:47 UTC; 365,604 steps/s on the 5080).**
  Last evals 13.37-13.90B: 9/9 at 13.370 / 13.445 / 13.521 / 13.672 (42.47 s) / 13.747 / 13.823 /
  13.898, 7/9 at 13.596 - nothing past 118.592. Harvested before release: all 14 runs (204 evals,
  run.json each), uf2r1b1_cut's ckpt_latest (md5 f1ee081d...) and the hand-on ckpt_13045858304
  (md5 359ab38c...), and `uf2r1b1_cut/extras/` (the trainer log, the auto-spine searches' logs
  and spine.json files, the 43.18 s finishing beam line of the box test in `fintest/`, the
  re-recording logs). Box 58455319 released at 08:54 UTC (idle since 08:47).

### uf2r1b1 round-1 lesson (wrap-up, 2026-10-31 09:03 UTC)

**Result: 118.592** - surf_unitfarmer2 FINISHED from the true start, champion-free, 9/9 greedy
(A13 `uf2r1b1_cut` at 13.068B, best 42.25 s; the first 9/9 at 8.161B, 45.00 s; record 35.17 s).
Scratch control (the round's base recipe) 5.608. Hand-on: **ckpt_13045858304** (md5 359ab38c...,
22M before the best eval), re-recorded 9/9 twice (18/18, best 42.07 / 42.13 s). One lineage on
one RTX 5080, 13.9B steps; 204 greedy evals over 14 runs.

**What worked, in order** (A4 from scratch, every later arm a resume of the same weights):
1. A correct death charge. `--death-charge` under `--race-ratchet` billed a dying episode for
   the NEXT spawn's depth (fixed, ee29bffb; it made A1-A3 uninformative). Fixed and windowed
   (`--death-charge-window 2`) the critic went from ev 0.1-0.2 to 0.9+, and A4 dived into the pit
   at 227M with speed-bin novelty (`--int-coef 2.5 --int-view 0 --int-speed 8 --int-speed-weight 3
   --int-speed-weight-up`), `--yaw-entropy 0` and keys temperature.
2. Energy, then exits: per-call `--energy-pot 12` (A5); greedy-row respawns at x1.0-1.3 speed
   (A6) with the heading turned U(-180, 180) (A8) -> the first exit in training (1.13B);
   `--tick-ms-schedule 10:7.63:3e8` (A7).
3. A backward curriculum on the policy's OWN Monte Carlo line from the true spawn (A10, by hand)
   -> the greedy left the pit (2.94B, 8.8).
4. The trainer's `--auto-spine` loop (beam_tas from the true spawn on its newest weights -> next
   spine; milestones keep the old lines paying) -> 12.5 -> 69.2 (A11-A12); then, by hand,
   `--demo-success-d 100` instead of -400 and a cut of the 84% spine where its curriculum
   stalled (A13) -> 86 -> 99.7 -> 9/9. The last 141 u of height under the finish box's floor were
   learned by the policy (the curriculum walking back + its own finishes), not found by a search.
5. Long training then paid in time and, late, in consistency: 45.0 s (8.16B) -> 42.6 s (10.4B,
   then flat) -> seven 9/9 evals in a row from 12.77B, best 42.25 s.

**What did not / watch:**
* `--demo-success-d -400` stalls a curriculum at a beam spine's end; the line belongs at the
  spine's end + 100.
* The loop could never hand back a FINISHING line (beam_tas wrote no frontier on a crossing; the
  gain rule rejected a finish within --gain of the goal): FIXED in 58574f32 with tests and a box
  test, not yet run inside a trainer.
* The start pit stays fragile: whole evals die there (0/9 at 8.39, 11.33, 12.31, 12.39B; 2/9 at
  8.61, 10.80B - laps without the exit, or a mis-dive into a catch), and it holds 3.55 s of the
  8.4 s gap measured at 8.15B.
* Evals overstate a policy: snapshots between 9.6 and 10.9B re-recorded 6/9-8/9 beside 9/9
  evals. Re-record a candidate twice before handing it on; snapcopy prunes its copies after
  45 min and copies only when ckpt_latest's mtime moves.
* `score.py --best $AR_HOME/runs/$AR_AGENT/` does not recurse into the run directories (0 evals);
  score a run directory, or all of them (`.../$AR_AGENT/*/`).

**Next (round 2):** reliability in the pit first (the score ranks it first), then speed: the
policy's own beam finishes in 43.18 s on the 8.82B weights (`uf2r1b1_cut/extras/fintest/`), the
expert loop's seed (beam -> plan_to_bc -> --bc-file; keep the trainer's evals on, the grader
counts only `traj_<step>.jsonl`). For the recipe (0b): make the three manual steps automatic (a
first spine from the true start without a hand-run MC line; the stall cut without
--auto-spine-energy) and repeat on a second map with the same flags and constants.

## uf2r2b1 (round 2, slot 1) - 2026-10-31 09:44 UTC start

Box 58518461: RTX 5080 + i9-14900K (machine 147030, round 1's host), 0.38 $/h, Utah. Incumbent
118.592 = uf2r1b1's ckpt_13045858304 (9/9, best 42.25 s); the score needs 9/9 AND a faster best
finish (1 s = 0.033), so this round is about TIME with the pit reliability kept.

### Where the 42.2 s goes (ANALYSIS ONLY, the record as an instrument; 09:45-09:50 UTC)
Split times by best field progress (`autoresearch/r2b1/` has no script for it yet; scratch
splits.py), the incumbent's 13.068B eval (median of 9) vs the record (file clock, its curtain at
~1.55 s): 1% 1.29 vs 2.15 / **2% 11.24 vs 8.33** / 5% 12.44 vs 9.41 / 20% 16.77 vs 13.49 / 40% 22.29
vs 18.88 / 60% 28.80 vs 25.26 / **70% 34.40 vs 29.29** / 90% 39.97 vs 34.39 / 100% 42.54 vs 36.76.
* The pit (1% -> 2%): 9.95 s vs 6.18 s = **+3.8 s**. The greedy laps the pit TWICE counter-clockwise
  (in going north on the EAST side, west at the north end, south down the WEST side, round the south
  end at y -3,755 and z ~-700 east, north again on the east side to y -645, west, south on the west
  side, and only then climbs the south-west wall: z -783 at t 10.0 with vz +777 -> z 272 at t 11.0
  -> out north at t ~12). The record laps ONCE clockwise (north on the west side, U-turn east at the
  north end, south on the east side, climbs the south end at t 6.5-7.5, out at t ~8). On its first
  pass of the south end the greedy is LOW (z -760..-670, vz +382) with E 1,150-1,160 - more than the
  record's 1,054-1,100 during its climb; the second lap buys +120 u of E (1,155 -> 1,277).
* 60% -> 70%: 5.60 vs 4.03 s = **+1.6 s**: an extra half-loop in the bowl at z ~-1,150 round
  (1,100..1,850, -490..+740) at t 30-31.5 (east, SOUTH to y -488, back north) where the record goes
  east and straight north. The greedy carries more E there (845-890 vs 608-707).
* Both losses are extra loops on an energy-conservative line.

### Searches on the incumbent's weights (box, 09:57-10:00 UTC; 28 s each)
`autoresearch/r2b1/beams1.sh`: beam_tas from the TRUE start, 4096 envs, d-ranked, 32 greedy envs,
--save-frontier --keep-finishers 8 --gens 3. S1 plain: **41.17 s** (5,752 finishes). S2 + a view-only
fork at tick 650 (t 5.0 s, half the population, jitter 45, protect 3): **41.14 s**. S3 keys-temp 1.5
view-temp 2: 41.36 s. All lap the pit twice (2% at 11.1-11.3 s): the gain is in the late course
(60-70% 5.26 s vs 5.60; 70-100% 7.56 vs 8.14).

### B1 `uf2r2b1_pace` (launched 09:59:57 UTC): the speed curriculum
Warm resume of the incumbent (13,045,858,304; md5 359ab38c) - measures the speed lever AT the
policy's frontier. The S2 line (5,366 states, `runs/beam/r2s2.npy`, md5 4a2f3f1a; keys 06864232)
as the spine, taught with the PACE rule (pbr9b1's `--demo-success-pace`): a spine episode
succeeds and is paid the 50 only if it crosses d <= 100 within the line's own time from its spawn
row + 0.5 s (from the map start 5,417 ticks = 41.55 s; the greedy takes ~42.2 s). Milestones keep
A13's paid lines (29584, 27363, 17276, 13765, 13232, 8515, 3476, 486); curriculum flags as A13;
--respawn-greedy-temp 1; no auto-spine (refused with the pace rule); --ckpt-every 2e8 + snapcopy;
6.5e8 steps (~30 min). Flags restored from the checkpoint: energy-pot 12, death charge 1 / 2 s,
novelty 2.5, greedy rows x1-1.3 turned 180, tick 7.63 ms, etc. (the launch log's "restored" line).
* **B1 result: STOPPED at 13.234B (10:14 UTC), a clear negative.** Evals 9/9 42.30 s (118.590, the
  resumed weights) -> 8/9 42.41 s (117.475) -> 7/9 42.74 s (116.353) at 13.046 / 13.122 / 13.197B.
  The pace curriculum walked back to row ~4,070 of 5,366 in ~60M steps and stalled there (band
  success 3.6-26%: the last ~10 s of the beam line is ~0.5 s faster than the policy's habit and it
  carries much less energy: E 593 at 90% and 463 at 95% vs the greedy's ~850). Training crashes 8%
  -> 27-33%, training win rate 70% -> 22-60%. Failures: one in the pit at 4.3 s, one at 37.7%, one
  at 89.5%. **Design flaw**: a NEW spine restarts its curriculum at its END, so every demo draw sat
  in [4,070, 5,365] - the start and the pit lost the spawns the incumbent was trained on (A13's k02
  curriculum sat at tau 0: half the demo draws on the true start, half uniform over the line). A
  warm policy's distribution must not be moved off its own start by a speed spine.
* Pit searches (10:02-10:17 UTC, beside B1): view-only forks of 3,072 envs at ticks 600 / 700 / 800
  (t 4.6 / 5.4 / 6.1 s; jitter 0 or 30, burst 1.0-1.5 s, protect 4-5) -> 41.14-41.35 s; forks at the
  pit ENTRY (ticks 300 / 400, t 2.3 / 3.1 s, where the record curves west and the policy east;
  jitter 0 / 60 / 90, protect 7-8) -> 41.15-41.32 s; held offsets on every proposal env
  (`--view-hold` 0.5-2.0 x 10-25 decisions) -> 42.12-42.67 s (they hurt the late course). **No
  search left the pit on the first lap**: the best branch lineages were still at d ~31,200 (progress
  -2%) at 7.7 s and culled by 15 s. One `--view-hold 0.5 10` lineage reached 7.4% at 7.7 s by a
  different route - off the platform's SE corner at t 2, landing on ledges (E 1,208 -> 638 at t 3)
  and WALKING north at 300-500 u/s to round 1's training ledge (-2,000, -1,177, 343) at t 7: no
  energy for the course (the beam's winner was a two-lap line).
* B2 `uf2r2b1_bc1` (10:16 UTC): resume of the incumbent with A13's own end distribution (the k02
  spine, `--demo-tau0 0`, success line 486, milestones 29584..3476, no auto-spine) + EXPERT
  ITERATION: `--bc-file runs/bc/r2_bc1.npz` (plan_to_bc of the 16 fastest finishing lines of four
  beams on the incumbent's weights - S1, S2, j652, u652 - 21,479 decisions, best 41.14 s,
  line-weight-decay 20), `--bc-coef 0.5 -> 0` over the 6.5e8-step run, `--bc-keys-engine`,
  `--bc-pitch-coef 0`. Start: head-acc 0.86, joint-acc 0.54, nll 0.83 against the lines.
* Go-Explore rooted at the policy's OWN first-lap pit states (`autoresearch/r2b1/ge_pit1.sh`: 41
  rows 450-850 of the r2s2 line, t 3.45-6.5 s; `--exact-chains --cell 64 --cell-energy 32
  --cell-heading 1 --view-follow=-90..90 --select-energy 3 --decisions 100 --tick-ms 8`, 8 min,
  5.9M runs, 1.28M cells): **0 cells past 1.7% progress**; the best climbs reach 0.6% at
  (-1,557, -3,216, 51) with E ~300 (`autoresearch/r2b1/ge_early_exits.py`). Uniform held keys
  cannot surf the climb out with the energy the first lap has.
* **B2 `uf2r2b1_bc1`: 118.611 at 13.197B (9/9, best 41.67 s, mean 42.22 s; 10:26 UTC)** - the
  first score over the incumbent. Evals: 6/9 42.70 s (after ONE iteration: the BC's first steps
  move the policy; B1's first eval at the same step was 9/9) -> 9/9 42.37 -> **9/9 41.67** -> 9/9
  42.32 (13.273B). BC: nll 0.83 -> 0.05 (head-acc 0.97, joint-acc 0.91) in ~40M steps, -0.24 /
  0.996 / 0.983 by 13.27B; kl 0.13-0.24 (high: BC drags the policy); training crashes 0-18%.
  Hand-on: ckpt_13190561792 (snapcopy, 6.8M before the eval), pulled.
* New code (386cd649): `--respawn-greedy-plateau S` (+ `-n N`, default 200): the greedy pool also
  keeps N evenly spaced states of every MID-EPISODE stall >= S s (the running best d gains < 64 u,
  then gains) of every greedy episode, FINISHED ones included (`surfgym.greedy_spawn.plateaus`).
  Without it the pool is empty once the greedy finishes 9/9 (only failed episodes are harvested),
  so the turned / boosted greedy rows that gave round 1 its first pit exit (A8) no longer see the
  pit laps. Off = bit-identical (unit tests: tests/python/test_respawn_greedy_plateau.py).
* Expert-iteration round 2 (10:33 UTC): `autoresearch/r2b1/beams_it.sh` (six beams: plain x3
  seeds, view-only forks at t 5.0 s, t 2.3 s and t 29.9 s - the 60-70% loop) on B2's 13.294B
  weights -> BC file 2 for B3 (a resume of B2's final checkpoint).
* Hand-on audit: **ckpt_13190561792 re-records 9/9 twice** (record_ckpt on the box, 9 eps: default
  seed best 42.21 s / score 118.593; --seed 7 best 41.49 s / 118.617; `uf2r2b1_bc1/verify/`).
* B2 later evals: 9/9 42.32 s (13.273B), 9/9 42.41 s (13.348B); stopped at 13.408B (10:44 UTC) to
  iterate. The BC drags the greedy only partly: its median finish stays 42.2-42.9 s (the
  incumbent's 42.54) while the lines it clones run 41.1-41.6 s.
* **Iteration 2 searches** (six beams on B2's 13.294B weights, `beams_it.sh`): **40.80-40.96 s**
  (iteration 1 on the incumbent: 41.14-41.35) - the search improves with the policy. The 40.79 s
  line vs the incumbent's median: 2% 11.22 vs 11.24 (both pit laps still), 60% 28.54 vs 28.80, 70%
  33.77 vs 34.40, 80% 36.17 vs 37.04, 90% 38.62 vs 39.97, 100% 40.77 vs 42.54: all of the gain
  after 60%. A fork at t 29.9 s (the 60-70% loop) did not find a direct line (40.96 s).
* BC file 2 `runs/bc/r2_bc2.npz` (16 lines, best 40.80 s, md5 d7f48d0b). The box was moved to
  bf59e8d4 with `box.py deploy` (code only: the run dirs are untracked); its gpu_health step
  FAILED because B2 was training on the GPU (a GEMM benchmark under load) - not a host defect,
  the same card passed at the first deploy, so it is NOT blocklisted. Maps restamped by hand.
* **B3 `uf2r2b1_bc2pl`** (10:45 UTC): resume of B2's 13,407,617,024 weights (md5 f42e1e0e), B2's
  flags with `--bc-file runs/bc/r2_bc2.npz` + **`--respawn-greedy-plateau 2`**. The first eval's
  pool: 18 plateaus = every episode's start pit (1.5-11.6 s) and its 60-70% loop (29.6-32.5 s),
  4,427 states, turned U(-180, 180) and x1-1.3 (the restored greedy-row perturbations), 25% of the
  non-start pool. First eval (one iteration in): 7/9, 42.94 s (the BC's first steps again).
* **B3 `uf2r2b1_bc2pl` result (stopped 13.849B, 11:14 UTC): FASTER, LESS RELIABLE.** Evals 7/9
  42.94 s / 8/9 41.86 / 8/9 **41.47** / **9/9 41.67 (118.611, ties B2)** / 7/9 41.79 / 8/9 41.67 at
  13.408-13.786B. Median finish 41.75-42.05 s (B2 42.2-42.9, the incumbent 42.54): 80% at 36.6-36.8
  (37.04), 90% at 39.4-39.6 (39.97). The pit is unchanged (2% at 11.2-11.3 s; the plateaus stay
  1.5-11.2 s), so ~400M steps of turned / boosted pit-lap respawns did not teach a first-lap exit.
  Failures: late course (37.4, 38.7, 39.3, 42.6 s - one at 99.98%, 27 u west of the finish box and
  53 u BELOW its floor), 19.2 s, one in the pit at 7.9 s. Training crashes ~20%.
* Iteration 3 (beams on B3's 13.762B weights): **40.53-40.77 s** (still two pit laps: 2% at 11.1 s;
  60% at 28.0, 70% at 33.1-33.3). BC file 3 `runs/bc/r2_bc3.npz` (16 lines, best 40.53 s, md5
  d47e0432).
* **B4 `uf2r2b1_bc3`** (11:15 UTC): resume of B3's 13,848,543,232 weights (md5 0e4246b3) with BC
  file 3, `--bc-coef 0.3` (B2 / B3 lost 2-3 episodes in the eval after one iteration at 0.5), and
  NO plateau rows (the pit did not move; they take 25% of the pool from the k02 line).
* **B4 `uf2r2b1_bc3`: 9/9 again, and faster: 41.43 s (118.619, 13.849B) and 41.37 s (118.621,
  13.925B; mean 41.96-41.97 s).** BC coef 0.3 and no plateau rows: reliability is back (the
  plateau rows, not BC2, most likely cost B3 its 7-8/9). Hand-on for 118.621: ckpt_13920370688
  (4.2M before the eval), pulled.
* **BREAKTHROUGH: a FIRST-LAP PIT EXIT found by the policy's own Monte Carlo (11:18 UTC).**
  `autoresearch/r2b1/beams_mc.sh`: beam_tas `--elite-frac 1.0` (no cloning: 4,096 independent
  stochastic rollouts of the policy from the true start, 32 greedy) + view-only forks of 3,072 envs
  at tick 400 (t 3.07 s) holding a heading offset U(-J, J) for 0.46-0.92 s, on B3's 13.762B weights:
  * J 60, burst 120 ticks, hold 120: a fork lineage **finishes in 37.34 s**; J 40 / 120 / 60: a
    plain (non-fork) lineage finishes in **38.00 s**; the other six settings (J 0-60, forks at tick
    400 or 500) 40.80-40.96 s.
  * The 37.34 s line's pit (ANALYSIS): north up the east side as always, but at t 4.0 (-1,918,
    -1,444, -732, vz +418) it carries on NORTH to (-2,284, -811, -623) at t 4.5 instead of turning
    west at y ~-1,400, comes south down the west side (t 5.0-5.5) and climbs out at the south end on
    the FIRST pass: t 6.0 (-2,273, -2,968, -499) vz +1,290 -> z 45 at 6.5 -> z 389 at 7.0 -> out
    (3.0%) at 7.5. **2% at 7.12 s** (the greedy 11.2 s; the record 8.33 s on its file clock, ~7.4 s
    on ours); 20% 12.73, 60% 24.46, 70% 29.85, 100% 37.31. It is the second lap's own north
    excursion flown on the first lap.
  * **Why every earlier search missed it**: a d-ranked beam culls the north excursion (d rises
    going north: progress -9% at t 4.5 vs the lappers' -7%) before it pays off ~3 s later; the
    forks were protected 3-8 generations but resampled on d around them. Monte Carlo does not cull.
  * More Monte Carlo (`beams_mc2.sh`, the J 60 / tick 400 fork, new seeds): 37.75 s, 37.67 s ... -
    it is reproducible in the policy's own distribution with a held heading offset at the turn.
* Next (11:30 UTC): prefix beams (`beams_pf.sh`: replay a 37.x s line's first 8-9 s with
  `--prefix-line`, then a d-ranked search for the rest on B4's 13.986B weights) -> BC file 4 with the
  one-lap exit, + the line's pit states as `--spawn-states` -> B5.
* Monte Carlo 2 (`beams_mc2.sh`, B3's 13.762B weights, forks at tick 400): **8 of 12 runs give a
  first-lap exit**, finishing in 37.51-38.14 s (several of them NON-fork lineages: 37.76, 37.91 s);
  forks held for 60 ticks (redrawn) and plain Monte Carlo (no fork) give none (40.70-41.17 s). The
  37.34 s winner's held yaw offset turns out to be SMALL (its executed yaw stays within +-2 deg of the
  policy's own through the burst): the one-lap exit is the policy's own key choices near t 4 s; what
  finds it is Monte Carlo whose dead lineages (the big offsets die) are refilled from survivors at
  the resample boundaries, i.e. many more samples of that decision, and NO d-ranked culling.
* Prefix beams (`beams_pf.sh`: `--prefix-line` = a Monte Carlo line's first 1,043-1,174 ticks, then
  a d-ranked beam for the rest on B4's 13.986B weights): **36.59 / 36.62 / 36.66 / 36.91 / 37.24 s**,
  all replayed bit-exact. BC file 4 `runs/bc/r2_bc4.npz` (16 lines, best 36.59 s, md5 4c2a63a1; the
  prefix rows' z rebuilt from the executed view, keys rebuilt from the engine rows - consistent
  with --bc-keys-engine). Spawn states `runs/beam/r2_pit_states.npy` (368 states, t 2.8-8.4 s of the
  36.59 s line, md5 5eafbf60).
* B4 later: 9/9 41.48 s (14.076B); stopped at 14.141B (11:43 UTC). Hand-on for 118.627 (14.000B):
  ckpt_13973848064 (26M before; the next copy is after the eval) re-records 9/9 41.28 s (118.624)
  and 8/9 41.48 s with --seed 7 (17/18).
* **B5 `uf2r2b1_pit1`** (11:44 UTC): resume of B4's 14,141,095,936 weights (md5 7b898296) + BC file 4
  (coef 0.3 -> 0) + `--spawn-states runs/beam/r2_pit_states.npy --spawn-states-frac 0.1`; k02
  curriculum at tau 0 as before. Goal: the greedy flies the one-lap exit (~37 s).
* **B5 `uf2r2b1_pit1`: 118.633 at 14.368B (9/9, best 41.00 s, mean 41.39 s)** - the branch's best.
  Evals 7/9 40.98 s / 6/9 41.05 / 9/9 41.70 / **9/9 41.00**. BUT the greedy never takes the one-lap
  exit: 2% at 10.8-11.2 s in every episode. BC fits the one-lap lines at their own states (joint-acc
  0.999) and the spawn states let the policy practise them, yet the closed loop from the true start
  still turns west at the first lap's north end (the decision is made in states the BC rows do not
  quite cover - the Monte Carlo prefix's first 3 s were sampled, not the greedy's). The late course
  did get faster (finishes 41.0-41.9 s; one fail-free eval at mean 41.39 s). Hand-on:
  ckpt_14361821184 (6.3M before), pulled.
* Next: a PACE-GATED backward curriculum on a one-lap line from the CURRENT weights
  (`beams_mc3.sh`: Monte Carlo on B5's 14.454B weights -> the fastest finisher -> prefix beams), the
  curriculum started just PAST the pit exit (`--demo-tau0` ~ the exit row) so the uniform draws cover
  the whole rest of the course (B1's lesson) and the band walks back INTO the pit, where only a
  one-lap exit keeps the line's pace.
* Monte Carlo 3 on B5's 14.454B weights (`beams_mc3.sh`): **all 8 runs give a one-lap exit**
  (37.24-37.91 s), the two PLAIN Monte Carlo runs (no fork) included - on B3's weights plain Monte
  Carlo gave none. BC4 + the pit spawn states moved the policy's stochastic distribution onto the
  one-lap exit; only its deterministic mode (the greedy) still turns west. Prefix beams 36.77-36.96 s.
  B5's last evals: 8/9 41.68 s, 9/9 41.77 s; stopped at 14.568B (12:08 UTC).
* **B6 `uf2r2b1_pace3`** (12:11 UTC): resume of B5's 14,568,390,656 weights (md5 12210a5c); the 36.77 s
  one-lap line as the curriculum spine (`runs/beam/r2_one621.npy`, md5 a480c602, + keys) started at
  row 1,050 (just past the pit exit, t 8.0 s), PACE-gated: `--demo-success-pace 0.05 1.0` (from the
  map start 5,153 ticks = 39.5 s: only a one-lap exit makes it, the two-lap greedy takes 41+ s);
  milestones 29584..3476 + 486 (k02's line); BC file 5 (16 one-lap lines from B5's weights, best
  36.77 s) at 0.3; spawn states = the spine's t 2.8-8.4 s (10%). A first try with `0.02 0.5`
  (`uf2r2b1_pace2_aborted`) had 0% band success at row 1,050 (the line's late course is beam-optimised,
  ~0.5-1 s faster than the policy's own) and backs off only 5 rows a move: relaunched at 5% + 1 s,
  where the band at 1,050 succeeds 55% and the curriculum walks back into the pit.
* Box extended to 15:12 UTC.
* **B6 `uf2r2b1_pace3`: THE GREEDY LEAVES THE START PIT ON ITS FIRST LAP (12:25 UTC).** Evals: 7/9
  41.25 s / 6/9 41.58 / 9/9 41.69 (14.720B) / **9/9, best 38.23 s = 118.726 (14.795B)** - three
  episodes one-lap (38.23, 38.30, 39.42 s), six still two-lap (42.6-43.1 s) - / **8/9, ALL NINE
  one-lap (2% at 7.65-7.93 s), finishes 37.90-38.41 s, mean 38.23 s (14.871B)**; the one failure
  died at 43.3% (22.9 s), mid-course, not in the pit. The curriculum walked back from row 1,050
  (t 8.0 s) to ~845 (t 6.5 s, the climb) in ~300M steps (band success ~55-60%; one 0% band at 845
  backed off once). Hand-on for 118.726: ckpt_14791213056 (4.2M before), pulled.
  **What did it**: the policy's own Monte Carlo found the line, BC + pit spawn states put it in the
  policy's stochastic distribution (B5), and the pace gate made it the only way to succeed from the
  pit rows, so the greedy committed to it. ~3.5 s of the 6 s gap to the record, in ~40 min.
* B6 continued: 8/9 37.18 s (14.946B; fail at 59.4%), 8/9 37.60 s (15.022B; fail at 15.2 s at the
  pit EXIT). Its curriculum band stalls at rows 840-845 (t 6.5 s, the climb out): success flips
  between 0% (845) and 52-57% (840), i.e. it is training exactly the move that fails.
* **Hand-on audit of 118.726 (ckpt_14791213056)**: default seed 9/9, best 38.40 s (118.720; 7 of 9
  one-lap, 2 two-lap at 43.0-43.2 s); **--seed 7: 6/9** (best 38.36 s) - 15/18. The three seed-7
  failures (and the 15.022B eval's pit failure) die at the SAME place, the top of the one-lap climb,
  (-1,840, -2,650, z ~525), ~80 u west of where the spine line passes (-1,757, -2,639, 535): the
  one-lap exit skims the catch volume under the platform's south-east edge. Without eval patience
  (record_ckpt runs 362 s) such an episode LAPS the pit for 70-350 s - the second-lap exit is
  forgotten once the first lap is learnt - before dying there. Two failure modes now: that climb,
  and scattered mid-course deaths (43%, 59%).
* Iteration-4 searches on B6's 14.951B weights (`beams_it4.sh`): the plain d-ranked beam from the
  TRUE start now finds the one-lap line by itself (36.97, 37.22 s); Monte Carlo 36.88-37.13 s;
  prefix beams **36.65-36.69 s**. BC file 6 `runs/bc/r2_bc6.npz` (16 lines, best 36.65 s, md5
  90cada01); spine candidate `runs/beam/r2_one723.npy` (36.65 s, md5 1c1dc348); pit states
  `runs/beam/r2_pit_states3.npy` (md5 54365e81).
* B6 ended at its budget (15.368B, 13:13 UTC): 9/9 37.89 s / 9/9 37.95 / 9/9 37.53 / 9/9 37.82 s
  (15.097-15.324B; scored 118.72-118.74 by score.py) - every episode one-lap, reliability back.

### !! The finishes were FAKE: the core scored the finish trigger's BOUNDING BOX (13:15 UTC)
A memory note of the operator's session (2026-10-31 13:04, "finish-trigger-aabb", from the user's
eye on a uf2 video) reports: the core's goal test (src/env.c, the swept segment vs the
hull-inflated AABB from zones.json) is not the game's - GoldSrc fires a brush trigger only inside
the model's player-expanded clip hull - and surf_unitfarmer2's finish (*23 -> counter_stop_button)
is three 1 u curtains (west face x 3,328 for y <= -2,144; a diagonal (3,328, -2,144)-(3,584, -1,888);
north face y -1,888 for x >= 3,584) whose box has an EMPTY north-west corner. Checked on my own
evals: **every "finish" of B6 (and B2-B5, and the incumbent's 13.068B eval but one episode) ends at
(3,302..3,330, -1,865..-1,950, -520..-650)** - the hull-inflated corner, outside the real trigger
(a real crossing needs y <= -2,128 at the west face, or the diagonal / north face hull). So
118.611-118.74 here, and the incumbent's 118.592, are scores of finishes the game would not count.
* **FIX (85592304, my branch)**: `surf_set_goal_model` (additive C export) + `SurfCore.set_goal_box`
  arms the trigger_multiple whose model bounds ARE the box (`surfgym.core.finish_trigger_model`, the
  BSP's own lumps; *23 here): a crossing counts only where the tick's segment (points <= 16 u apart)
  is inside that model's clip hull (`trigger_contains`, the test the core already used for
  teleports / hurts - so the PIT-EXIT catch deaths above are real; only the finish was wrong). A
  button box matches no trigger and keeps the box test; `SURF_GOAL_AABB=1` restores the old test.
  Tests on the real map (`tests/python/test_goal_model.py`): the corner skim finished on the box
  and does not on the trigger; a west-curtain crossing finishes under both. Every tool (trainer,
  evals, beam_tas, record_ckpt, explore) arms the goal through set_goal_box, so all get it. Box:
  core rebuilt (`./build.sh`) at 85592304, tests pass there.
* **B7 `uf2r2b1_real1`** (13:23 UTC): resume of B6's final 15,368,454,144 (md5 fd3a6826) on the fixed
  core; the r2_one621 curriculum (band from row 840, pace 0.05 1.0) and pit spawn states as B6; **no
  BC** (every BC line ends in the corner skim). The pace line d <= 100 is still reached by the
  skim, the finish bonus (50) no longer: the policy has to carry on ~250 u into a real curtain.

### Harness issue (for the operator)
`autoresearch/score.py` (a harness file) counts `end == "done"` from the trajectory trailer, i.e.
whatever goal test the recording core ran. Evals recorded on the old core count the corner skim as a
finish (the incumbent's 118.592 is 1/9 real by the memory note's count). Evals of this branch from
B7 (uf2r2b1_real1) on are recorded on the fixed core, so their finishes are real; everything
earlier in `$AR_HOME/runs/uf2r2b1/` (B1-B6, scores up to 118.74) is NOT. A grader fix would re-score
recordings on the trigger's hull from the positions (the memory note's plan).
* **B7 `uf2r2b1_real1` on the FIXED core - REAL finishes: 118.730 at 15.595B (9/9, best 38.10 s, mean
  38.36 s).** Evals: 6/9 38.41 s (B6's weights: the old line mostly carries on into the diagonal
  ~0.3 s after the corner) / **0/9** (15.444B: all nine patience-cut in the pit - a transient) / 9/9
  39.05 / **9/9 38.10** / 8/9 38.34 / 9/9 38.23 s (15.746B). Finishes now end inside the box at the
  diagonal curtain, (3,345-3,469, -1,972..-2,097, -457..-594). Hand-on: ckpt_15584985088 (10.5M
  before), pulled; re-recording on the box. Real-finish searches (`beams_it4.sh` on B7's weights)
  running for a real-finish BC file.
* OPERATOR NOTE (13:50 UTC): the finish fix is in the harness (uf2/main 2a27047f, PROTOCOL READ FIRST;
  found independently by uf2r2b1 / b2 / b4 the same hour, merged); score.py counts a "done" only if
  the finishing step entered the trigger's clip hull (box-only = progress, `box_only`): the round-1
  incumbent re-scores 118.592 -> 109.692 (1/9 real). The user: keep the checkpoints and train a little
  more on the fixed finish, no scratch retraining. Re-scored with the operator's score.py (copied
  read-only to the scratchpad): B7's evals are real (118.730 stands), B6's 118.726 eval -> 100.000.
* B7 hand-on ckpt_15584985088 re-records **9/9 REAL twice** (default seed 38.30 s = 118.723; --seed 7
  38.14 s = 118.729): 18/18. B7 later: 8/9 38.20 s (15.822B); the band sits at the climb (845: 0%).
* Real-finish searches on B7's 15.77B weights (`beams_it4.sh`): d-ranked beams 37.57 / 37.63 s, Monte
  Carlo 37.66-37.94 s (the box-finish lines were 36.65-36.97 s: a real finish costs ~0.6 s more).
* Real-finish prefix beams (B7's 15.77B weights): **37.50-37.60 s** (best real line 37.50 s). BC file 7
  `runs/bc/r2_bc7.npz` (16 real-finish lines, md5 1239d576).
* **Where the 38.1 s still loses (ANALYSIS, B7 15.595B median vs the record ~0.86 s file offset)**:
  the pit +0.24 s (the one-lap exit is near the record's), 2-60% +0.6 s, **60-70% +1.4 s** (B7 5.45 s
  vs the record's 4.03 s: at t 25.5 both are at ~(900, -240) heading north; the record turns EAST
  within 0.5 s and dives into the bowl north-bound at x ~1,600; B7 carries on north to y ~700, turns
  back SOUTH through (1,358, -7, -1,243) and comes north again 1.5 s behind), 80-90% +0.4 s. The
  real-finish search lines keep the same loop (60-70% in 5.26 s).
* Loop Monte Carlo (`beams_loop.sh`: --prefix-line of the 37.50 s line to tick 3,100-3,300, then
  `--elite-frac 1.0` + view-only held-offset forks of 3,072 envs at ticks 3,200-3,400 = t 24.5-26.1 s,
  J 30-90, 0.5-0.92 s): **8 of 8 null** (37.63-37.74 s, every winner a non-fork lineage). Unlike the
  pit, a held heading offset there does not put the policy onto the record's turn.
* B7 final evals: 9/9 38.35 s (15.897B), 5/9 37.98 s (15.973B); stopped at 15.975B (14:09 UTC).
* **B8 `uf2r2b1_real2`** (14:10 UTC): resume of B7's 15,975,055,360 (md5 0bd5b400); spine = the
  37.50 s REAL line (`runs/beam/r2_real723.npy` md5 528f703f, from B7's own weights: its pit climb is
  the current policy's, where B7's band was stuck at r2_one621's row 845 with 0% for ~0.5B steps),
  tau0 1,070 (just past its exit, row 1,067), pace 0.05 1.0; BC file 7 at 0.3; its pit states
  (`r2_pit_states4.npy`) at 10%. Band success 77.6% at 1,070 at once.
* **B8 `uf2r2b1_real2`: 118.744 (16.202B: 9/9 REAL, best 37.69 s)** - four 9/9 evals in a row: 37.89
  (118.737) / 38.08 / **37.69** / 37.78 s (16.051-16.278B; official re-score identical); first eval
  8/9 37.72. The curriculum on the policy's own real-finish line walked from row 1,070 back to tau 0
  (the true start) in ~250M steps (56-64% paced success). Hand-on: ckpt_16193683456 (8.4M before),
  pulled; re-recording. Next search round (`beams_it4.sh`) on B8's 16.31B weights.
