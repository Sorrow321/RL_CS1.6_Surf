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
