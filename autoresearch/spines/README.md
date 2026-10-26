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
* `pbr6b3_k02e.npy` + `.keys.npy` (5,431 rows; md5 7ec93bd8379710dc96e4c6577e91e7b0 / da5ef616c5681d43bc82eff17d55518d):
  `pbr5b2_c7k02` cut at its row 5,430 (61.72%, d 21,875, energy height 2,721), the last row before the chain's landing at
  the east lane's south end (row 5,433, -172 u) - an ENERGY spine for `--demo-success-energy`: through 59.6-61.7% the
  chain flies the corner off the north wall and the airborne east lane keeping ~+250 u over the greedy (h 2,721 vs
  2,437-2,486 at 61.5%; the greedy lands on the corridor floor at 59.9%, -340 u, and takes the west strip). Same
  provenance as `pbr5b2_c7k02` (the policy's own beam line + one exact reward-free explore_phase1 chain); no demo, no
  record (pbr6b3, round 6).
