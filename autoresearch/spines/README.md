# Spawn / spine sources for surf_princessburglar_b26 (pbr1b1, round 1)

Machine-generated in the simulator from the POLICY'S OWN states (CLAUDE.md section 0 allows
these in training; declare them with `box.py launch --self-states "<provenance>"`). No demo,
no record, no human data entered either file. Both are STATE_DTYPE `.npy` (clocks zeroed).

* `pbr1b1_line_a.npy` (1,881 states, md5 4296eab791f13b104d33afc52bb76eed): one exact per-tick
  line from the crouch tunnel (the policy's own state at (224, -2,885, 3,722)) past the pit to
  3.93% on the grader field. `tools/explore_phase1.py --exact-chains` (reward-free uniform bursts,
  64 u x --cell-energy 32 cells, rooted at 64 of pbr1b1_g2b's OWN greedy tunnel-exit states,
  6 min on a 5080 box) + `tools/ge_replay.py` (deepest chain, 24/24 links replayed within 0.5 u).
  Route: back east through the tunnel (run-up), slide west at 300-380 u/s, drop onto the ridge at
  ~770 u/s, launch off its east end over the pit, ducked air turn into the east wall, ramp
  U-turn, north, U-turn west. Used by pbr1b1_g2cur..g2cur5 as `--demo-file` (backward curriculum).
* `pbr1b1_entry_ge_states.npy` (7,399 states, md5 3d014fb2abc6d883fd51b839611889b6): the start
  room's own-search states - 71 spawn->plate->tunnel lines from `tools/entry_search.py`
  (uniform random bursts rooted at the map's true spawn, every 9th tick) plus the forward cells
  of two Go-Explore phase-1 archives rooted at the true spawn. Used as `--spawn-states` (0.3-0.4)
  to keep the start room practised. Caveat (NOTES.md): the two entry_search stage target boxes
  were chosen from core probes of this map.
* `pbr2b3_line_a_canon.npy` (1,881 states, md5 8de7e89f9e2b9abedc13b37f68a7bf68): pbr1b1_line_a.npy
  with the VIEW re-aimed along the velocity on every row with |v_h| >= 100 u/s (1,316 rows;
  `tools/spine_canon_view.py`); every other field is identical. The explorer's random actions had
  left the view a median 48 deg (176 deg through the flight) off the velocity, while a
  --view-absolute velocity policy looks within +-7 deg of it and turns at ~4 deg/tick at speed.
  Same provenance as line_a (pbr2b3, round 2).
* `pbr2b3_line_a_canon_780.npy` (1,101 states, md5 6a0373013ebbfda3e51cb16020d52b0a): pbr2b3_line_a_canon.npy from state 780
  on - the westward slide through the crouch tunnel, where the policy's OWN greedy line from the
  true start joins the spine (pbr2b3_canon3's 3.644B eval enters the tunnel at x ~640 heading west);
  states 0-779 are the explorer's run-up back east through the tunnel, which the greedy line never
  flies. Same provenance (pbr2b3, round 2).
* `pbr2b3_beam_g3b.npy` (1,066 states, md5 ccce8fa9a36d173b67f7e1e7df6707c8): the policy's OWN line
  from the true spawn to 6.62% on the grader field, past gate 3. `tools/beam_tas.py` (policy-guided
  population search: 4,096 stochastic copies of pbr2b3_canon5's 4.29B weights, 32 greedy envs as a
  floor, the leaders by goal-field d cloned every 25 decisions, `--save-frontier --route-file none`,
  20 s cap, 102 s on the box) found the frontier lineage at 6.70% at 14 s and replayed it exactly
  (runs/pbr2b3/_beam_g3b/beam_frontier.npz); cut at its best progress before it stalls in a hollow
  at x ~-1,015 (tick 1,065). Its views are the policy's own (within ~7 deg of the velocity at speed).
  No demo, no record data (pbr2b3, round 2).
* `pbr2b3_beam_ge_g4.npy` (909 states, md5 a28ef67685dd77ad89bce2be3a91b96e): the policy's own beam line from the true spawn
  (beam_tas on pbr2b3_beam1's 4.548B weights, runs/pbr2b3/_ge_g4/beam_g4a_frontier.npz) up to its
  tick 720 (the drop after gate 3), then the deepest chain of a Go-Explore run rooted at 24 of that
  line's own states (explore_phase1 --exact-chains, 64 u x energy 32 cells, 10 min) replayed
  exactly by ge_replay (7/7 links) from that same state: it surfs the drop at up to 1,058 u/s and
  climbs to 7.07% at (-1,002, -2,994, 3,240). Views re-aimed along the velocity (spine_canon_view).
  No demo, no record data (pbr2b3, round 2).
* `pbr2b3_beam_g8a.npy` (1,338 states, md5 f0e76e8019d2472a14d63ff3096035d2): the policy's own line
  from the true spawn past the x -1,015 wall to 12.34% (beam_tas on pbr2b3_beam7's 6.98B weights,
  4,096 copies, 32 greedy envs, d-ranked; frontier 12.52% at 24 s), cut at its best progress before
  it stalls at x ~-3,400 (tick 1,337); views along the velocity. No demo, no record data (pbr2b3).
* `pbr2b3_beam_g9a.npy` (1,441 states, md5 7ab294dd8eb440f6ff3c2913c9326b3a): the policy's own line from the true spawn
  through the U-turn climb past the 13% fall catch to 15.36% (beam_tas on pbr2b3_beam8's 7.33B weights;
  frontier 15.66% at 15 s, every lineage dead by 16 s), cut 5 ticks before the impact at x ~-1,965
  (tick 1445); views along the velocity. No demo, no record data (pbr2b3).
* `pbr2b3_beam_g10a.npy` (1,736 states, md5 1ae3ae5821bc860f0986c549780769d4): the policy's own line from the true spawn to
  18.26% (beam_tas on pbr2b3_beam9's 7.67B weights; frontier 18.52% at 19 s, the population
  extinct by 28 s), cut at a healthy state at or before its best; views along the velocity. No demo,
  no record data (pbr2b3).
* `pbr2b3_beam_g10a_1560.npy` (1,561 states, md5 698e8ec75b45d269c4bd41d6e14439a5): the same beam_g10a line cut at tick 1,560
  (16.68%, just past the fall catch at x ~-1,700..-2,000 that ends beam9's greedy line), so the
  curriculum's success line sits right after that catch. Same provenance (pbr2b3).
* `pbr2b3_beam_ge_g16.npy` (2,381 states, md5 84fb90934c54195c1c78012e076fb367): the beam_g10a line (the policy's own) to its
  tick 1,455 (just past the ~16% ramp impact) + the deepest chain of a Go-Explore run rooted at 25 of
  that line's own states before the impact (32 u x energy 16 cells, 12 min, 1.27M cells; ge_replay
  15/15 links): ~6.8 s of back-and-forth on the ramps there (h 2,972 -> ~3,220, a small pump), then
  east at 640-1,078 u/s to 18.86%. Views along the velocity. No demo, no record data (pbr2b3).
* `pbr3b4_ge_r3d.npy` (1,814 states, md5 f716d4f9149a582edd452a1897570c0d): the policy's own line
  from the true spawn past the 20.4% fall catch to 21.54% (pbr3b4, round 3). The first 1,500 states
  are beam_tas's frontier lineage (4,096 stochastic copies of the incumbent's 9.48B weights, d-ranked,
  runs/pbr3b4/_beam_r3a); state 1,500 onward is the deepest chain of an explore_phase1
  --exact-chains run (64 u x energy 32 cells, 15 min, uniform random actions) rooted at 66 of that
  line's own states over 14.0-15.6 s at velocity x1.00 and x1.04, replayed exactly by ge_replay
  (13/13 links): its root is the line's state at 15.0 s (the launch off the ramp at x -868) with the
  velocity x1.04 - a sharper right turn at x 440-550 into the south corridor, a tangential landing
  on its west surf ramp, a U-turn at (484, -3,443) and a climb east to (1,103, -2,852, 2,833) at
  ~1,015 u/s. Views along the velocity on the explorer's part (tools/ge_spine.py). No demo, no record.
* `pbr3b4_auto_k02.npy` (3,348 states, md5 5cf733e9ae4a6a773c515b0682cf0281): found BY THE TRAINER
  (train_fast --auto-spine, search 2 of pbr3b4_as3 at 10.72B): tools/auto_spine.py's branched beam
  - beam_tas from the true spawn on the run's own weights, 4,096 copies, half of them forked at
  17.0 s (0.5 s before the greedy line's median launch) with held view offsets U(-45, 45) deg for
  0.5 s each over 2 s, the policy's own keys - over the 21.5% wall (gate 9) to a hard landing at x
  ~2,370 and 13 s of crawling at 23.9-24.2% (gate 10); cut at its best d (24.19%). No speed boost,
  no demo, no record. Used by pbr3b4_as3 (10.72-10.87B: greedy 21.5 -> 24.29), as4, as5.
* `pbr3b4_auto_k03.npy` (2,284 states, md5 0bbda0750625edaefc32be2ae7ca9f85): found BY THE TRAINER
  (search 3 of pbr3b4_as3 at 10.87B): the run's own beam line from the true spawn to 17.2 s + the
  deepest exact Go-Explore chain (64 u x energy 32 cells, 8 min, stage 2 of the escalation ladder)
  rooted at that line's state at 17.2 s with the velocity x1.12, replayed exactly, views along the
  velocity: over the gate-9 wall further south, a ramp catch at x ~2,444 and up and north to 26.66%.
  No demo, no record. The x1.12 root is out of the greedy line's reach so far: unprotected it eroded
  the route (as3 10.87-11.16B); under --demo-milestones it holds the route but its ramp-catch band
  sits at ~25% (as6, as7); pbr3b4_as8 continues it with --demo-rate 0.2.
* `pbr3b4_spawn_both2.npy` (9,316 states, md5 35f28b37ddecd9d9f5cecefc6bb4d510): a `--spawn-states`
  file for the first-ramp question (the user's insight, 2026-10-25, NOTES.md): the 7,399 states of
  pbr1b1_entry_ge_states.npy (start-room, deeper than the start) + the 1,917 cells BEHIND the start
  (shaping-field distance above the spawn's) of a 6-min explore_phase1 --exact-chains archive rooted
  at the true spawn (uniform random actions, 64 u x energy 32) from which >= 25% of 32 uniform-random
  rollouts survive 1 s (x 863-2,380: the east half-pipe; energy median 3,649, max 3,815). The
  explorer's states in both directions, nothing aimed at a ramp; used by pbr3b4_as12 (the policy
  learned to survive there and to come back to the tunnel; it did not pump). No demo, no record.
* `pbr4b3_beam_g10.npy` (2,400 states, md5 7fa685aa6ffabd1d3e220f9f3928ebec): the policy's OWN line
  from the true spawn over gate 10 (the ramp climb at x ~2,450-3,040 after the gate-9 wall) to
  26.36% on the grader field: tools/beam_tas.py on the round-3 incumbent's weights (pbr3b4_as10,
  13.4999B), 4,096 stochastic copies, 32 greedy envs, d-ranked, resample every 25 decisions, a
  branched half forked at 16.0 s (every branched lineage died; the frontier is an unbranched
  lineage of the policy's own proposals), 26 s cap, --save-frontier replay exact (d 42,078). It
  rides UP the ramp (z 2,785 -> 3,204 at 19.4-20.8 s), lands on the floor beyond it at z 3,197 and
  walks/hops north at 100-440 u/s. Views are the policy's own. No demo, no record (pbr4b3, round 4).
* `pbr4b3_ge12_x132.npy` (2,841 states, md5 942cb89a439a8cd193ae6d8e7a869e99): gate 11 (the lethal
  floor under a 1,000 u flight at 26.4%) by search - the policy's OWN branched-beam line from the true
  spawn (beam_tas on pbr4b3_t1's 14.97B weights at their own tick ~8.25 ms, the trainer's auto-spine
  search 2, half the copies forked with held view offsets) to its tick 2,299 on the gate-10 ramp, then
  the exact Go-Explore chain (explore_phase1 --exact-chains --tick-ms 8, uniform random bursts, 64 u x
  energy 32 cells, 12 min) rooted at that line's ramp state with the velocity x1.32, replayed 16/16
  links exactly: off the ramp top high, onto the ridge, surfing it north, over the sheet and down the
  hole at 27.9%; cut just after the hole's landing (28.41%). Views of the chain along the velocity
  (spine_canon_view). No demo, no record (pbr4b3, round 4). The x1.32 root is a big boost: under
  milestones the old success lines keep the route paid.
* `pbr4b3_beam_g11.npy` (2,900 states, md5 ad6f945f669b526921297705409a23a9): the policy's OWN line from
  the true spawn over gate 11 at its own tick (7.67 ms): beam_tas on pbr4b3_t2's 15.47B weights (4,096
  copies, 32 greedy, d-ranked, torch seed 4), replayed exactly - the ridge north, the launch, the flight
  over the lethal floor, down the hole at 28.0% and on to 29.31% where it falls toward the trigger_teleport
  sheet *128 (z ~2,577, a fall catch; gate 12). Cut at its best d. Views are the policy's own. No boost,
  no demo, no record (pbr4b3, round 4).
* `pbr4b3_ge13.npy` (3,054 states, md5 e7bf7a39310025e4736fac27ab43fa98): gate 12 (the drop out of the
  gate-11 hole) - pbr4b3_beam_g11.npy (the policy's own 7.67 ms beam line) to its state 2,690 over the
  lethal floor, then the exact Go-Explore chain (explore_phase1 --exact-chains --tick-ms 8, uniform random
  bursts, 64 u x energy 32 cells, 12 min) rooted at that state with the velocity x1.04, replayed 7/7 links
  exactly: down the hole heading SOUTH-WEST, a hard landing on the ramp at (2,880, -1,683, 2,781), west then
  north at ~820-900 u/s, to 30.44% (falling toward the catch sheet *128 at its end). Cut at its best d;
  the chain's views along the velocity. No demo, no record (pbr4b3, round 4).
* `pbr4b3_ge14.npy` (3,230 states, md5 d5ba35e9937b7b8d6188017034ce98dd): gate 12 WITHOUT a boost - pbr4b3_beam_g11.npy (the policy's own 7.67 ms
  beam line) to its state 2,750 just before the gate-11 hole, then the exact Go-Explore chain
  (explore_phase1 --exact-chains --tick-ms 8, uniform random bursts, 64 u x energy 32 cells, 15 min) rooted
  at that state UNBOOSTED (x1.000), replayed 12/12 links exactly: a bounce off the room's north side turns it
  south-west into the hole, onto the ramp at (2,759, -1,751, 2,950), north along x ~2,400-2,560 to (2,383,
  -23, 2,661) = 31.79%. Cut at its best d; the chain's views along the velocity. No demo, no record.
* `pbr4b3_ge16vf.npy` (3,074 states, md5 4d6815e097e64594bfa4cb7d15d82b35): gate 12 searched IN THE POLICY'S
  VIEW FRAME - pbr4b3_beam_g11.npy (the policy's own 7.67 ms beam line) to its state 2,743 just before the
  gate-11 hole, then the exact chain of explore_phase1 --exact-chains --tick-ms 8
  --view-follow=-90,-60,-30,-15,0,15,30,60,90 (bursts hold a view offset from the velocity heading; 15 min)
  rooted there UNBOOSTED, replayed 42/42 links exactly, cut at 30.28% (its hard landing at (2,371, -873,
  2,614), before a 25 s crawl near the catch). Views as the explorer had them (offsets of the velocity
  frame; no canon_view): out of the hole it holds forward with the view ~89 deg left of the velocity.
  No demo, no record (pbr4b3, round 4).
* `pbr4b3_ge16ck.npy` + `pbr4b3_ge16ck.keys.npy` (3,074 rows; md5 f60c3f012b2361bf77f106b279fc2948 /
  41d4b5c2b1d1fdebecbcd6ee4a123c2f): pbr4b3_ge16vf.npy in the POLICY's frame - tools/spine_canon_keys.py puts
  every row's view along the velocity and translates the held keys so the wish direction survives (to within
  22.5 deg): on the gate-12 ramp the explorer's forward-at-+89 deg becomes a LEFT strafe. The keys (fwd, side,
  duck; engine bins) are the policy's own resolved beam keys on the g11 rows and the explorer's on the chain
  rows (pbr4b3_ge16vf.keys.npy, md5 baf91cbccc1580b0145cd6b609b9dc21, before translation); for train_fast
  --demo-keys. No demo, no record (pbr4b3, round 4).
* `pbr4b3_auto_t9k01.npy` + `.keys.npy` (5,100 rows; md5 32fbfb3b05a6251f71a9404715701a4b /
  26abfd8a3fedb29086b0c76fe360f976): found BY THE TRAINER (pbr4b3_t9's auto-spine search 1 at 19.598B, 64 s):
  its branched beam - beam_tas from the true spawn on t9's weights at their own 7.67 ms tick, half the copies
  forked 1 s before the greedy line's stall with held view offsets - past gate 12 to 34.99%. The keys are the
  beam's own resolved held keys per row (auto_spine.keys_from_acts). No demo, no record (pbr4b3, round 4).
* `pbr4b3_auto_t9k04.npy` + `.keys.npy` (4,700 rows; md5 f92b0ad2ed39e53a23ed80a256cea868 /
  d97584f6519581659368871f242ad83c): found BY THE TRAINER (pbr4b3_t9's auto-spine search 4 at 20.127B, 74 s):
  its branched beam on t9's own weights from the true spawn (7.67 ms) to 45.18%; keys = the beam's own held
  keys per row. No demo, no record (pbr4b3, round 4).
* `pbr4b3_ge18ck.npy` + `.keys.npy` (4,134 rows; md5 edb1a8a33acdea05284adcc78a25026b /
  717f5e12ed93de3f09bea31a4ba2164f): gate 14 (a crash into a floor + wall corner at 44.2%) - the policy's own
  7.67 ms beam line (beam_tas on pbr4b3_t10's 20.54B weights, seed 51) to its state 3,557 (42.08%), then the exact
  view-follow Go-Explore chain (explore_phase1 --exact-chains --tick-ms 8 --view-follow, 12 min) rooted there
  UNBOOSTED, replayed 14/14 links exactly: it rebounds off the corner instead of stopping and flies west along the
  record's corridor to 47.18%. Views along the velocity with the held keys translated (spine_canon_keys); the
  beam rows' keys are the beam's own. No demo, no record (pbr4b3, round 4).
* `pbr4b3_auto_t11k03.npy` + `.keys.npy` (4,438 rows; md5 e10027824d48e385a37af9989c9ed0b3 /
  34a04785b6299fd27de9385430f64613): gate 15 (a launch off a ramp lip at 47.2%) - found BY THE TRAINER
  (pbr4b3_t11's auto-spine search 3 on patience at 22.910B): t11's own 7.67 ms beam line to its state 3,819, then
  the view-follow Go-Explore stage-2 chain (x1.08 at its root, 619 states, explore_phase1 --exact-chains --tick-ms 8
  --view-follow) to d 28,683 = 49.81%. The greedy passed it ~150M steps after the swap (50.72 at 23.06B). Views
  along the velocity, keys translated (spine_canon_keys); the beam rows' keys are the beam's own. No demo, no
  record (pbr4b3, round 4).
* `pbr4b3_auto_t11k04.npy` + `.keys.npy` (4,700 rows; md5 3c52f0b059a1850065cee4587a751b28 /
  6b676e96d47c8f006a34a4fdc4fd16e9): found BY THE TRAINER (pbr4b3_t11's auto-spine search 4 at 23.062B, 37 s, the
  greedy's quorum past k03): the plain beam on t11's own weights from the true spawn at 7.67 ms, to d 26,901 =
  52.93%; keys = the beam's own held keys per row. The spine t11 trained on at the end of round 4 and the natural
  --demo-file / --demo-keys for a resume of its checkpoint. No demo, no record (pbr4b3, round 4).
* `pbr4b3_auto_t11k05.npy` + `.keys.npy` (5,002 rows; md5 65e08c9f4ac0d1e3da24b6592dc442e9 /
  0cafdaaf895b919bfa8e7a44db22e86d): found BY THE TRAINER (pbr4b3_t11's auto-spine search 5 at 23.364B, 638 s,
  the greedy's quorum past k04): t11's own 7.67 ms beam line to its state 4,344, then an UNBOOSTED (x1.0)
  view-follow Go-Explore stage-1 chain (663 states) to d 25,021 = 56.22%. Views along the velocity, keys translated;
  the beam rows' keys are the beam's own. The spine t11 trained on at the end of round 4 and the --demo-file /
  --demo-keys for a resume of its checkpoint. No demo, no record (pbr4b3, round 4).
* `pbr5b2_e1.npy` + `.keys.npy` (10,952 rows; md5 1711b7d8028edb3eb083c64ddaa75c8b /
  b1ae7456fc168da111ba4eedc543eb5a): princessburglar's 56.6% shaft climb, stitched by tools/stitch_spine.py from
  four exact segments, each the policy's own or the run's own reward-free search, each starting at the previous
  one's state: (1) the policy's own 7.67 ms beam line from the true spawn (pbr5b2_c1's auto-spine search 1, beam_tas
  on its 24.080B weights) to its state 4,372 (51.2%); (2) the exact view-follow Go-Explore chain of that search's
  stage 1 (roots x1.0 / 1.04 over the beam line's last 4 s; this chain's root x1.04) to its most energetic
  non-plunging U-bottom cell (energy height 2,741 vs the beam line's ~2,400), replayed 13/13; (3) the policy's own
  greedy rollout from that state (record_ckpt --dump-states, c1's 24.080B weights) for 64 ticks; (4) the exact
  Go-Explore chain (explore_phase1 --exact-chains --cell-heading 1 --view-follow --tick-ms 8, 10 min, roots on
  segments 2-3 x1.0 / 1.04 / 1.08; this chain's root x1.08) up the shaft in one rise and over the upper level to
  62.25% (d 21,577), replayed 46/46. Cut at its best d. Views along the velocity on the chains, keys translated
  with them; the beam rows' keys are the beam's own, the rollout's from its trajectory. No demo, no record
  (pbr5b2, round 5).
* `pbr5b2_e1c.npy` + `.keys.npy` (5,005 rows; md5 adae45ae3cb2c2afffdbde98d4e41e61 / 7ff2bc12313b0290ad11010426bf04ff):
  pbr5b2_e1 cut at its row 5,004 (57.20%, d 24,459), just past the shaft climb on the upper level - the next gate only
  (the full e1's curriculum stalled at once on an explorer impact at 61.5%, far past what the greedy flies). Same
  provenance as pbr5b2_e1.
* `pbr5b2_e1d.npy` + `.keys.npy` (4,597 rows; md5 below in NOTES): pbr5b2_e1 cut at row 4,596 = the end of its GE chain A,
  the most energetic U-bottom state (energy height 2,741; the policy's own line ~2,300 there): the cleaner descent
  over 51.2-54.5%, for a curriculum whose success needs the ENERGY (train_fast --demo-success-energy), not depth.
  Same provenance as pbr5b2_e1.
* `pbr5b2_s6.npy` + `.keys.npy` (6,399 rows; md5 a84256d093930a3e3851fe57807f9205 / 199f9fb7357974cdb92e784747087b0b):
  the POLICY'S OWN plain beam line (tools/auto_spine.py: beam_tas from the true spawn, 4,096 stochastic copies, d-ranked,
  on pbr5b2_c5's 25.876B weights at their 7.67 ms tick) to d 21,524 = 62.34%: after c5 learned the cleaner 51.7-54.5%
  descent (energy height 2,962 at 53%) the policy's own population goes up the 56.3% shaft and over the upper level by
  itself, unboosted. Keys = the beam's own held keys. No demo, no record (pbr5b2, round 5).
* `pbr5b2_c7k01.npy` + `.keys.npy` (5,142 rows; md5 fb5d000a28d3bc9574c4a77511ba0e3a / 75874412312ab0137543e2240f3ab3c2):
  the first ENERGY spine of the trainer's own loop (train_fast --auto-spine-energy 150, run pbr5b2_c7 search 1 on c6's
  62.367 weights at 29.048B): the POLICY'S OWN plain beam line (tools/auto_spine.py: beam_tas from the true spawn) to its
  row 4,814, then one exact explore_phase1 --exact-chains chain (reward-free search; 328 ticks from that row's state with
  its velocity x1.04) to archive cell 251,579 at 59.55% with energy height 2,701, +217 u over the beam line there (the
  energy stage's pick: the deepest cell keeping >= 150 u). Keys = the beam's and the chain's own held keys. Taught with the
  energy success rule (h >= 2,626 at d <= 23,216) it put the greedy's h at 58.2% from ~2,480 to 2,683-2,751 in 75M steps.
  No demo, no record (pbr5b2, round 5).
* `pbr5b2_c7k02.npy` + `.keys.npy` (5,824 rows; md5 71c6f3e22042d30d70fc0641cd0928e9 / 7d3c279bacc6f144fb5855c4ab1c8b80):
  the first line PAST princessburglar's 62.3% drop (gate 19): run pbr5b2_c7's search 2 on its 29.124B weights (after the
  energy spine k01 taught the 58% corner) - the POLICY'S OWN plain beam line from the true spawn to its row 5,017 (~60%,
  the next corner), then one exact explore_phase1 --exact-chains chain (reward-free search; 807 ticks from that row's state
  with its velocity x1.12, the ladder's third stage) to d 20,489 = 64.15%. Keys = the beam's and the chain's own held keys.
  No demo, no record (pbr5b2, round 5).
* `pbr6b4_k02e.npy` + `.keys.npy` (5,421 rows; md5 c54ecba0b952d488611e2135758be3e9 / 6010cf9384a9a56203cccd7253178d9b):
  `pbr5b2_c7k02` (the loop's own line past the 62.3% drop: the policy's plain beam from the true spawn to row 5,017,
  then one exact reward-free explore_phase1 chain from that row's state x1.12) CUT at row 5,420 (61.65%, h 2,721),
  just before the landing band [5,423, 5,427] its depth curriculum could not pass (12-16% for ~15 min in pbr6b4_c8).
  Taught with the energy rule (train_fast --demo-success-energy 150: cross d(end)+100 with h >= 2,571) for the
  stretch where the greedy leaves the line (the 59.9% corner and the east lane). No demo, no record (pbr6b4, round 6).
* `pbr6b4_c9k01.npy` + `.keys.npy` (5,700 rows; md5 6023ae916420321e9680d41bb7bc980d / 4bc964a7280e57e42fb0b0c10173f60a):
  the first line of the POLICY'S OWN past princessburglar's 62.3% drop (gate 19): run pbr6b4_c9's search 1 on its 29.958B
  weights (after the energy spine pbr6b4_k02e taught the 59.9% corner) - tools/auto_spine.py's plain beam_tas from the true
  spawn, 4,096 stochastic copies of the policy, d-ranked, no fork, no boost, no explorer chain - to d 20,132 = 64.77%. Keys =
  the beam's own held keys. No demo, no record (pbr6b4, round 6).
* `pbr6b4_c9k02.npy` + `.keys.npy` (7,328 rows; md5 121ba05fe03e0ed183419040a44cb745 / b76a0f43199ccb90680400460792b761):
  run pbr6b4_c9's search 2 on its 30.260B weights (7/9 greedy episodes past the 62.3% drop): the policy's own branched
  beam line (tools/auto_spine.py: beam_tas from the true spawn, half the copies forked with held view offsets) to its row
  5,976 (65.47%), then one exact explore_phase1 --exact-chains chain (reward-free search, that row's state x1.04) to
  d 16,970 = 70.31%. Keys = the beam's and the chain's own held keys. No demo, no record (pbr6b4, round 6).
* `pbr6b4_c9k02c.npy` + `.keys.npy` (6,301 rows; md5 7999b8d227db83c12e0bd6f7a20216ba / 3c30cae919c342025d6ebfbdd96bff68):
  `pbr6b4_c9k02` cut at row 6,300 (66.42%, success line 66.25%), before the explorer chain's slow hopping at 66.5% where
  c9's curriculum oscillated (one gate past the greedy at a time; round 2's lesson). No demo, no record (pbr6b4, round 6).
* `pbr6b4_c11k01.npy` + `.keys.npy` (9,000 rows; md5 392147eb7a57d7664fa8ee888fa7dc16 / 68057ce2fabc77f34146f3b8aca34ac7):
  run pbr6b4_c11's search 1 on its 32.390B weights (5/9 greedy episodes past the 70.13% walkway line): the policy's own
  plain beam_tas line from the true spawn (tools/auto_spine.py, 4,096 stochastic copies, d-ranked, no fork, no boost, no
  explorer chain) to d 15,471 = 72.93%. Keys = the beam's own held keys. No demo, no record (pbr6b4, round 6).
* `pbr6b4_c12k02.npy` + `.keys.npy` (6,852 rows; md5 0aced6925bb21e8b7ec73c434dcfa7e5 / 8b172e13693f889c3997b17f648aaa7b):
  run pbr6b4_c12's search 2 on its 36.446B weights (7/9 greedy episodes past the 71.3% landing with the energy spine's
  energy): tools/auto_spine.py's plain beam_tas line from the true spawn (the policy's own 4,096 stochastic copies,
  d-ranked) to its row 6,387, then one exact explore_phase1 --exact-chains chain (reward-free search, that row's state
  x1.04, view-follow, rising split) to d 14,673 = 74.33% (success line d <= 14,773). Keys = the beam's and the chain's
  own held keys. Copied unchanged from the box's runs/pbr6b4_c12/auto/k02 (pulled at round 6's wrap-up). No demo, no
  record (pbr6b4, round 6; committed by pbr7b2, round 7).
* `pbr7b2_e1.npy` + `.keys.npy` (6,813 rows; md5 aa851e756427a0ed4c4991b72b86a6b1 / e6a29daa51635ee673ce161bac93888b):
  an ENERGY spine built with the loop's own energy-stage rule by tools/energy_splice.py (auto_spine.energy_cell: the
  deepest archive cell keeping >= 150 u more energy height than the line at the same progress, on the route, not
  plunging, not slower): `pbr6b4_c12k02` rows 0-6,468 (unchanged, keys included) + one exact explore_phase1
  --exact-chains chain (reward-free search, view-follow, rising split, --select-energy 3; rooted at k02 rows
  6,400-6,600 x1.0 / 1.04, here an UNBOOSTED root at row 6,468, 71.86%) to cell 48,251 at 74.30% with h 2,407 (+153 u
  over k02's line there), views along the velocity, the chain's keys translated with them. Taught with
  train_fast --demo-success-energy 75 (cross d(end)+100 with h >= 2,332). No demo, no record (pbr7b2, round 7).
* `pbr7b2_e3.npy` + `.keys.npy` (5,884 rows; md5 a1922fb2ee92dc8308266242eca56b34 / 72af08fdce73e868e7c4db5bfc48267e): an ENERGY spine by the loop's
  energy-stage rule (tools/energy_splice.py, G 150): the policy's OWN plain beam_tas line from the true spawn (run
  pbr7b2_c3's 38.629B weights, 4,096 stochastic copies, d-ranked; the loop's settings) to its row 5,711 (65.18%), then one
  exact explore_phase1 --exact-chains chain (reward-free search, view-follow, rising split, --select-energy 3; rooted at
  that row's state x1.04) over gate 20's catch sheet, west of the walkway strip, to cell 60,145 at 66.59% with h 2,684
  (+153 u over the policy's line at the same progress, 108 ticks sooner). Keys = the beam's own and the chain's
  translated. Taught with train_fast --demo-success-energy 75. No demo, no record (pbr7b2, round 7).
* `pbr7b2_e5.npy` + `.keys.npy` (6,426 rows; md5 17ecf712c7871dd63e324a6d70c939a4 / 72e9e5a5fcfd13932d9939a6322b3681): an ENERGY spine by the loop's
  energy-stage rule (tools/energy_splice.py, G 150): the policy's OWN plain beam_tas line from the true spawn (run
  pbr7b2_c4's 40.246B weights, after it learned e3's walkway corner; the loop's settings) to its row 6,320 (71.08%), then
  one exact explore_phase1 --exact-chains chain (reward-free, view-follow, rising split, --select-energy 3; rooted at that
  row's state x1.04) that stays airborne past the 71.4% platform (no hop) to cell 57,109 at 72.32% with h 2,735 (+217 u
  over the policy's line there, 105 ticks against its 197). Keys = the beam's own and the chain's translated. Taught with
  train_fast --demo-success-energy 75. No demo, no record (pbr7b2, round 7).
* `pbr7b2_e6.npy` + `.keys.npy` (6,431 rows; md5 f8038cdd9c5345ca6a93ada500a3b158 / e53834bfb1d6988564515532e0aeccc7): an ENERGY spine by the loop's
  energy-stage rule (tools/energy_splice.py, G 150, window 72.2-72.8% = the launch ramp's approach): the policy's OWN plain
  beam_tas line from the true spawn (run pbr7b2_c5's ~41.2B weights; the loop's settings) to its row 6,294 (71.01%), then one
  exact explore_phase1 --exact-chains chain (reward-free, view-follow, rising split, --select-energy 3; rooted at that row's
  state x1.04) airborne over the 71.4% platform and back onto the ramp line, to cell 86,087 at 72.47% ((-1,255, 2,521,
  1,993) heading south at 1,045 u/s, h 2,675 = +205 u over the policy's line there). Keys = the beam's own and the chain's
  translated. Taught with train_fast --demo-success-energy 75. No demo, no record (pbr7b2, round 7).
* `pbr7b2_d2.npy` + `.keys.npy` (8,520 rows; md5 5e7424fde691687cd5c0b1ba963a13d9 / be2b1d40f899fda52ec64f9ef7dc7084): the first
  line of this lineage OVER the 74.4% block: the policy's OWN energy-ranked branched beam line from the true spawn (run
  pbr7b2_c6's ~42.2B weights; beam_tas --score dh --score-energy 3, half the copies forked at 70.25% with held view offsets) to
  its row 6,356 (73.0%, on the launch ramp's approach at ~1,124 u/s), then one exact explore_phase1 --exact-chains chain
  (reward-free; rooted at that row's own state with its horizontal velocity TURNED to head straight south, speed kept, no
  boost - the recipe's front-perturb heading turn as a search root) that rides the ramp, crosses the block ducked and crawls on
  to d 12,601 = 77.95%. Keys = the beam's own and the chain's translated (tools/stitch_spine.py). No demo, no record.
* `pbr7b2_d2c.npy` + `.keys.npy` (7,011 rows; md5 30b4d558489a50f0ccfac27c8d9ae66e / 95d3a8d68adb3d8ec36597676b4a299c): d2 cut at
  row 7,010 (74.80%, on the floor past the block; success line d <= 14,499 = 74.63%, out of reach of any line that does not
  clear the block). No demo, no record (pbr7b2, round 7).
* `pbr7b2_k04c.npy` + `.keys.npy` (7,451 rows; md5 814044ec2e5a1c726e58a743f489406d / ffb3843d5d6043dc67f501690f4dbceb): the
  trainer loop's own search-4 spine (run pbr7b2_c8, auto/k04: the PLAIN beam_tas line on c8's own 44,891,635,712-step weights
  from the true spawn to d 4,265 = 92.54%, 9,200 states, its own held keys; md5 f3f74722... / 36d3f5e9...) cut at row 7,450
  (82.04%, past the turn north at ~80.4% where every greedy episode dies on the fall-catch sheet *265; success line d <= 10,366
  = 81.86%, out of a death-dive's reach: the sheet sits at d ~11,050+). No demo, no record (pbr7b2, round 7).
* `pbr8b1_k04.npy` + `.keys.npy` (9,200 rows; md5 f3f74722b2f182d2ead9c818210800f7 / 36d3f5e9ec96b3db850990768364878d): the
  same search-4 spine in FULL (run pbr7b2_c8, auto/k04: the PLAIN beam_tas line on c8's own 44,891,635,712-step weights from
  the true spawn to d 4,265 = 92.54%; k04c is its first 7,451 rows). Its end was cut by the beam's HORIZON (--beam-ticks
  9,262), not by a death: row 9,199 is (637, 2,918, 470) falling at 1,017 u/s into the final shaft. Success line d <= 4,365
  (92.36%). No demo, no record (pbr8b1, round 8).
* `pbr8b1_k04s.npy` + `.keys.npy` (11,771 rows; md5 1953271132ec4e4d1b192a7b7e5b9ab4 / 84144fcb43a126469ff14f8f683a4ac9): the
  FIRST LINE FROM THE TRUE START TO THE FINISH BOX. pbr8b1_k04 (rows 0-9,199, the policy's own beam to 92.54%) + a 2,571-state
  exact Go-Explore chain (tools/explore_phase1.py --exact-chains, the loop's GE settings, rooted at 96 states of k04's own rows
  8,950-9,199 x1.0 / x1.04, 13.9 min, 3.2M runs, one goal hit; tools/ge_replay.py 35/35 links) rooted at k04's LAST row x1.04,
  stitched with tools/stitch_spine.py (views canonical, keys translated). The chain dives the final shaft's kill sheets (z 463
  -> -939 east of the z ~200 and z -800 sheets), floats ~10 s in a trigger_push field at z -939, descends through the z -1,300
  west gap, the z -1,700 ring and the z -2,700 hole, and ends at d 8 above the finish trigger. No demo, no record (pbr8b1).
* `pbr8b1_k04t.npy` + `.keys.npy` (9,807 rows; md5 edaf415bd5e107d83e17fb8df981e1b6 / 1801de5c39c055ed51bf210f2d0c5d02): the
  FASTER LINE FROM THE TRUE START TO THE FINISH BOX (use this one, not k04s). pbr8b1_k04 rows 0-9,071 (91.27%) + a 736-state
  exact Go-Explore chain (explore_phase1 --exact-chains, the loop's GE settings, rooted at 64 states of k04's own rows
  9,050-9,199 x1.0 only - no boost - 1.3 min to the first goal hit; ge_replay 16/16 links) rooted at k04 row 9,071 UNBOOSTED.
  The chain spirals down the shaft between the kill sheets in 5.9 s (|v| 550 -> 1,930 u/s) to d 9 above the finish
  trigger. No demo, no record (pbr8b1).
* `pbr8b1_entry_shaft_states.npy` (14,739 rows; md5 86ce11fa084409657060903362bf3152): `--spawn-states` pool = the 7,399
  start-room states of pbr1b1_entry_ge_states.npy + pbr8b1_k04t's 734 shaft-chain rows 9,072-9,805 (the GE 2 chain, own
  search from the policy's own k04 states) tiled 10x, so the 20% spawn-states share splits ~50/50 between the start room and
  the final shaft: the shaft is practised while the curriculum works on the middle of the line. Clocks zeroed. No demo.
* `pbr8b1_entry_mid_shaft_states.npy` (21,849 rows; md5 718b3a08864b2a0a3a91f40f378fa901): `--spawn-states` pool = the 7,399
  start-room states + pbr8b1_k04 rows 7,420-8,130 (the policy's own beam line from just past the 81.86% milestone, d 10,341,
  to the curriculum's hop band, d 9,004: where the greedy dies) tiled 10x + k04t's shaft rows 9,072-9,805 tiled 10x: ~1/3
  each of the 20% spawn-states share. Clocks zeroed. No demo.
* `pbr8b1_b3f.npy` + `.keys.npy` (9,574 rows; md5 ce58c2884380c193396a4bde8a90bef0 / 2850804f272bce947a6f2f382b662b71): A LINE
  FROM THE TRUE START TO THE FINISH, 96% OF IT THE POLICY'S OWN. Rows 0-9,179: the trainer loop's own search-1 spine of run
  pbr8b1_c1 (auto/k01: tools/auto_spine.py on c1's own 46,174,568,448-step weights; its THIRD branched beam_tas line, forked
  at tick 6,936, ran to d 2,020 = 96.47% in 9,200 states, cut by its horizon; md5 b92fbd61b983c6ef4b06febe3ca8b648) up to
  row 9,179 (96.0%, z -1,295 in the final shaft) + a 395-state exact Go-Explore chain (explore_phase1 --exact-chains, the
  loop's GE settings, rooted at 48 of that line's own rows 9,080-9,199 x1.0, goal hit after 0.2 min; ge_replay 7/7 links)
  rooted UNBOOSTED at row 9,179, to d 4.6 above the finish trigger. Stitched with tools/stitch_spine.py. No demo, no record.
* `pbr8b1_entry_b3mid_states.npy` (14,903 rows; md5 f3368dd076f4d76e4cc49a0c96a001f2): `--spawn-states` pool = the 7,399
  start-room states + pbr8b1_b3f rows 7,163-8,100 (the policy's own beam line from just past the 81.86% milestone, d 10,299,
  over the platform walk and the hops to the drop at 85.46%) tiled 8x: ~half of the 20% spawn-states share each. Clocks
  zeroed. No demo.
* `pbr8b1_b3c.npy` + `.keys.npy` (8,201 rows; md5 b53ed8d3c1c5c089ed23e1eff6df286b / f3a1bac6081fa04db95e969fa6a66a56):
  pbr8b1_b3f cut at row 8,200 (86.31%, d 7,826: past the drop to the lower level and its landing; success line d <= 7,926 =
  86.13%, out of reach of a failed platform hop-off, which dies at d ~8,800+). The k04c-style cut: the curriculum works the
  platform landing / walk / hop-off (81.9-84.6%), the greedy's current failure, with the success bonus right after it.
* `pbr8b1_entry_shaft2_states.npy` (14,964 rows; md5 7756adc55aa19bf9ad8f5d0bacf7c966): `--spawn-states` pool = the 7,399
  start-room states + b3f rows 9,250-9,350 tiled 37x (the shaft frontier c3's curriculum had reached: ~25%) + b3f rows
  8,935-9,572 tiled 6x (the shaft from just past the 92.36% milestone, d 4,300, so no spawn collects an easy +50: ~25%), to
  keep the shaft practised while the curriculum works the platform.
* `pbr8b1_entry_shaft3_states.npy` (15,001 rows; md5 e51d44c27bdd87967761683c7c3358ab): as shaft2 with the focus window moved back to b3f rows
  9,150-9,260 (tiled 34x) - the hand-run parallel curriculum's next band (rows 9,300+ finish 9-12/12 on c4's 48.2B weights).
* `pbr8b1_b3s.npy` + `.keys.npy` (9,340 rows; md5 a92aa28155fa3e3d0a0d2123bb2581d5 / 470056cde169d32bb9ab35badd4c41f7): A LINE
  FROM THE TRUE START TO THE FINISH, ALL OF IT THE POLICY'S OWN. Rows 0-8,900: pbr8b1_b3f (c1's search-1 branched beam, 46.17B
  weights) to 92.17% on the final shaft's rim; rows 8,901-9,339: the PLAIN beam_tas winner of pbr8b1_c4's own 48.58B weights
  started at b3f row 8,900 (4,096 copies: 1,604 finishes within 700 ticks; the fastest, 440 ticks = 3.37 s, replayed
  bit-exact by tools/beam_best_states.py and appended with tools/stitch_spine.py's roll segment, keys from its trajectory).
  End d 6.4. No explorer chain, no demo, no record (pbr8b1).
* `pbr8b1_entry_b3sshaft_states.npy` (14,635 rows; md5 92e4471739f0913f59f92e9fda2f773e): `--spawn-states` pool = the 7,399
  start-room states + b3s rows 8,937-9,338 (its own-beam shaft line from just past the 92.36% milestone) tiled 18x (~half).
* `pbr8b1_entry_b3sw2_states.npy` (14,719 rows; md5 in NOTES): `--spawn-states` pool = the 7,399 start-room states + two
  focused windows of b3s's own shaft line - rows 9,130-9,205 (the 96.5% / 98.4% deaths) tiled 48x and rows 8,960-9,095 (the
  94.6% death) tiled 27x: the hand-run parallel curriculum, one step ahead of the trainer's band (9,239).
* `pbr8b1_entry_b3sw3_states.npy` (md5 in NOTES): start room + b3s rows 9,170-9,238 (the z -2,700 hole approach, right above
  the trainer's band) tiled 53x + rows 9,100-9,169 (the z -1,700 ring) tiled 52x: windows whose finish bonus is ~1-1.5 s away.
* `pbr8b1_b3sw4_states.npy` (md5 in NOTES): `--spawn-states` pool of b3s's own shaft line ONLY (no start-room states): rows
  9,170-9,238 (the z -2,700 hole approach) 60x, 9,100-9,169 (the z -1,700 ring) 55x, 8,940-9,099 (the upper sheets) 19x.
  Why: start-room spawns each run a full ~70 s episode while a shaft spawn lives ~1 s, so they took most of the training
  samples; the start room is practised by the 10% fresh map starts anyway.
* `pbr8b1_b3sw5_states.npy` (md5 in NOTES): b3s's shaft rows only - 8,937-9,039 (the upper sheets, past the 92.36% milestone)
  34x, 9,040-9,169 (the 94.6-96.5% sheets, the weakest on the 50.2B bench) 35x, 9,170-9,249 (the hole approach) 30x.
* `pbr8b1_f1.npy` + `.keys.npy` (9,205 rows; md5 fd2e95323440d7d322401f42e1168d5b / 4fc85e3ff59d2fdbd0ed50367b93fda5): a
  FASTER LINE (clock 70.81 s; the greedy's best so far 73.04 s): the plain beam_tas finish-objective winner on pbr8b1_c5's own
  50.69B weights (72.65 s; plan_to_bc's spine, its keys from the action table) to row 7,320 (80.4%) + an exact Go-Explore
  chain (GE 6: explore_phase1 rooted at that line's own rows 7,270-7,345 x1.0 / x1.04, --select-energy 3; cell 163849, picked
  by the EARLIEST arrival on the lower level, d 7,885 at 62.05 s; root row 7,320 at x1.04) that FLIES OVER the 82-86% platform
  the policy otherwise walks + the plain beam_tas finish of c5's own weights from the chain's end (8.77 s). No demo, no record.
* `pbr8b1_f1w_states.npy` (11,176 rows; md5 a17948a7f0ad062972f0680ec41a3a2f): `--spawn-states` pool = f1 rows 7,450-7,582 (the
  airborne crossing past the 81.86% milestone) 12x + rows 7,583-8,061 (the chain's drop to the lower level, 84-86.2%) 20x.
* `pbr8b1_f1w2_states.npy` (md5 in NOTES): f1 rows 7,450-7,640 (the airborne crossing past the 81.86% milestone, to 84.06%) 25x
  + rows 7,641-7,800 (the landing on the platform's west end and the hop-off) 20x - one step ahead of c6's band.
* `pbr8b1_f1e.npy` + `.keys.npy` (7,521 rows; md5 0ccaa042bbc79b71078e0645c7f80310 / 9275c8b0f3ffef3837e6ba49952d15e4): f1 cut at
  row 7,520 (82.99%, d 9,721, airborne over the platform with energy height 2,323 u) - for --demo-success-energy: success only
  for an AIRBORNE crossing (the walking route crosses 83% at ~1,600 u), so the curriculum forces the faster route at the
  80.4% junction.
* `pbr8b1_f1after_states.npy` (8,115 rows; md5 c9a7fb08cb4beb684e80122eaba02b58): f1 rows 7,521-8,061 (past the f1e cut: the
  platform-end landing, the hop-off and the drop to the lower level) tiled 15x, practised while f1e's spine stops at 83%.
* `pbr9b1_fo1.npy` + `.keys.npy` (7,742 rows; md5 3c4d23c0417b3de28979c2a31ecf78b9 / a249811d22a398803507e002406f633d): the
  incumbent's (pbr8b1_c12 ckpt_55038705664) own greedy line from the true start (record_ckpt --dump-states, episode 1, a
  69.02 s finish; runs/pbr9b1/look/g0_states.npz md5 6c32b52b...) rows 0-7,512 (to 83.59%), then a Go-Explore chain
  (explore_phase1 --exact-chains, the loop's GE settings, 15 min) rooted at that line's row 7,512 with its velocity x1.04,
  replayed exactly by ge_replay (7/7 links; cell 172811): it turns north at x ~1,450 and FLIES over the platform's end
  (h 2,279, vh 964; the greedy lands there at 84.47% and loses ~500 u), spirals down the drop and reaches the lower level
  (d 7,799, 86.35%) at 59.42 s from the spawn on the line's clock (the greedy 60.52 s). Stitched with tools/stitch_spine.py.
  No demo, no record data (pbr9b1, round 9).
* `pbr9b1_fo1e.npy` + `.keys.npy` (7,603 rows; md5 b1967be397ef241604b68779d49b3203 / c0008d72ff1a7724119ac51d3a572976):
  fo1 cut at row 7,602 (84.92%, airborne over the platform's end, h 2,279) - for `--demo-success-energy 150`: only a crossing
  of d <= 8,719 with h >= 2,129 succeeds (the greedy's landing route crosses it at h ~1,700).
* `pbr9b1_fo1after_states.npy` (2,780 rows; md5 90aa9780d2c848e3c7e90ced0bb78e01): fo1 rows 7,603-7,741 (past the fo1e cut:
  the drop and the lower-level landing) tiled 20x, a `--spawn-states` pool.
* `pbr9b1_fo1dec_states.npy` (4,280 rows; md5 0fa67ef5dd931ed5f4f6be0a2355a176): a `--spawn-states` pool on the fly-over
  DECISION: fo1 rows 7,380-7,511 (the incumbent's own greedy line, 82.1-83.6%, the approach to the junction) x15 + rows
  7,512-7,602 (the chain's fly-over to the fo1e cut) x10 + rows 7,603-7,741 (past the cut) x10. Swapped into pbr9b1_c2's
  pool path (pbr9b1_fo1after_states.npy on the box) at 21:36 UTC through the trainer's mtime re-read.
* `pbr9b1_fo2.npy` + `.keys.npy` (7,889 rows; md5 7b3ece416cd8edea942929bd9ef2ab57 / 4b8bdda9ae29ec66f5cdc5274de60418): fo1 rows
  0-7,638 (the greedy line + the fly-over chain into the drop's spiral, 85.32%) + a second Go-Explore chain (GE 3: rooted at
  43 of fo1's own drop states, rows 7,596-7,680, unboosted; cell 141561, ge_replay 4/4 links) that LANDS SOFTLY on the lower
  level (h 2,023 -> 1,741 at 85.66%; fo1's chain lands at 1,560) and flies it at vh 1,170-1,256 to d 6,790 (88.12%, h 1,807)
  at 60.60 s on the line's clock (fo1's chain to 88%: 60.76 s; the incumbent's greedy 61.89 s). For an energy cut at its end
  (`--demo-success-energy 150` -> h >= 1,657 at d <= 6,890). No demo, no record data (pbr9b1, round 9).
* `pbr9b1_fo2dec_states.npy` (md5 in NOTES): fo2 rows 7,380-7,511 (the greedy approach) x10 + 7,512-7,637 (the fly-over) x8 +
  7,638-7,888 (the soft landing and the lower level) x8, a `--spawn-states` pool.
* `pbr9b1_s71c.npy` + `.keys.npy` (6,465 rows; md5 b8950d1c242c117036c1ed72ec0525fe / 2c11b5da062e4089a86b3a9c3b339757): fo2 rows
  0-6,274 (the incumbent's own greedy line to 69.75%) + a Go-Explore chain (GE 4: 161 roots = 23 of the line's own states over
  69.1-71.4% x the velocity heading turned 0 / +-4 / +-8 / +-12 deg, unboosted; cell 104115, root row 6,274 turned +12 deg,
  ge_replay 6/6 links) that turns south ~150 u earlier and skips the greedy's bounce-and-loop at 71.5-72.1%; cut at chain index
  190 (72.41%, d 15,766): a beam of pbr9b1_c5's weights (beam_tas, 4,096 copies, --objective finish) from there FINISHES (6,987
  finishes; first at 16.96 s -> 66.58 s from the spawn on the line's clock), while from the chain's end (73.76%) every copy dies
  within 1.5 s. For `--demo-success-pace` (time, not energy: the chain reaches 72.41% ~0.7 s sooner with LESS energy). No demo,
  no record data (pbr9b1, round 9).
* `pbr9b1_s71beam_states.npy` (2,212 rows; md5 in NOTES): that beam's per-tick line from s71c's last state to the finish
  (tools/beam_best_states.py, bit-exact replay).
* `pbr9b1_s71dec_states.npy` (md5 in NOTES): s71c rows 6,150-6,273 (the approach) x20 + 6,274-6,464 (the chain) x15 + the beam's
  first 900 ticks (72.4-83.3%) x2 + pbr9b1_fo2dec_states x1 (the fly-over and the soft landing), a `--spawn-states` pool.
* `pbr9b1_s71dec2_states.npy` (13,584 rows; md5 72aba8a25abc402633f68c20884ca2c5): pbr9b1_s71dec_states + the s71 beam line's own
  shaft passage (ticks 1,511-2,211: 90% -> the finish, after its soft landing) x3, a `--spawn-states` pool.
* `pbr9b1_s71d.npy` + `.keys.npy` (6,477 rows; md5 39b920caed473f7d1ca030ce29615fde / 9679b98ee41ecf418258203046956233): fo2 rows
  0-6,388 (the incumbent's own greedy line to 71.2%, 0.15 s before its bounce) + GE 4's chain cell 46772 (root row 6,388 with the
  heading turned -4 deg, unboosted; ge_replay 4/4 links): it meets the 71.3% surface heading south-west and goes south at x
  ~-1,150 instead of bouncing north-east into the loop; cut at chain index 88 (72.31%, d 15,825), ~0.56 s ahead of the greedy line.
  Beams of pbr9b1_c6's weights from chain index 60 / 88 finish (67.21 s / 67.28 s from the spawn on the line's clock; from index
  100 every copy dies). For `--demo-success-pace`. No demo, no record data (pbr9b1, round 9).
* `pbr9b1_s71dbeam_states.npy` (2,297 rows; md5 22d3d1b8d4b994503126f521fc009327): that beam's line from s71d's last state to the finish.
* `pbr9b1_s71ddec_states.npy` (11,334 rows; md5 2151cc182b69b5ec96987c690aebd10c): s71d rows 6,300-6,387 (the approach) x20 + the chain
  (6,388-6,476) x15 + its beam's first 900 ticks x2 + pbr9b1_fo2dec_states x1 + the s71 beam's shaft passage x3.
* `pbr9b1_fo2t.npy` + `.keys.npy` (889 rows; md5 in NOTES): fo2 rows 7,000-7,888 (77.1-88.1%: the greedy approach, the fly-over chain,
  the soft landing) - fo2 with its first 7,000 rows dropped, so a curriculum on it can never walk its band back into the 71%
  region (whose bounce every band that crossed it destabilised, pbr9b1 c2/c3/c5/c6/c7).
* `pbr9b1_late_states.npy` (md5 in NOTES): pbr9b1_fo2dec_states + the s71 beam's shaft passage (90% -> the finish) x3, a
  `--spawn-states` pool for the late course.
* `pbr9b1_late2_states.npy` (md5 in NOTES): pbr9b1_fo2dec_states + the s71 beam's shaft passage x2 + the 89-94.5% shaft entries of
  three FINISHING greedy episodes of pbr9b1_c8's own 57.34B weights (record_ckpt --dump-states, seed 11) x8 - a `--spawn-states`
  pool aimed at the shaft's first kill sheet (z ~265), c8's commonest death.

## pbfr1b4 (corrected core, lineage pbf*, round 1) - 2026-10-29

* `pbfr1b4_stair_a.npy` + `.keys.npy` (1,753 rows; md5 3892849cd9f832d4f3ac7321f51b01cd /
  11084147e642f251c6e483ee93c8aa6d): Go-Explore phase 1 (`tools/explore_phase1.py --cell 64 --cell-energy 32
  --exact-chains --act-every 4 --decisions 100 --maxvel 3500 --tick-ms 8 --select-energy 2`, 256 envs, 5 min on the
  droplet CPU, 273,677 runs) from the TRUE spawn on the CORRECTED core (6bd7f931 tables: func_conveyor *249 and *250 and
  trigger_push *252 / *253 all loaded). Reward-free uniform random actions; the deepest exact chain (24 cells, 23/23 links
  bit-exact) replayed by `tools/ge_line_keys.py` (the line + the engine fwd/side/duck it was flown with) and cut at its
  first row at 1.39% (the crouch tunnel's west end, before the drop); views/keys put in the policy's velocity frame by
  `tools/spine_canon_keys.py` (689 rows at >= 100 u/s). Its route: crouch-jump from the spawn floor west into the
  doorway's basin (func_conveyor *249, top z 3657, 41 u up), a duck-jump inside the 2500 u/s push-down field onto the
  *250 strip (top 3681), a running jump off the strip's north-east end onto the ramp north of the corridor, west along
  the passage ducked. No demo, no record data, no box or place chosen by hand. OPEN QUESTION for the operator (NOTES):
  whether that staircase exists in the real game.
