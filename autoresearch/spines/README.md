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
* `pbr8b4_k04d.npy` + `.keys.npy` (8,376 rows; md5 ac3760af979e24251de882683f50ea48 / c5cd9eeed6e26cae6205fb790a125d4c): the
  trainer loop's own search-4 spine (run pbr7b2_c8, auto/k04: the PLAIN beam_tas line on c8's own 44,891,635,712-step weights
  from the true spawn to d 4,265 = 92.54%, its own held keys; md5 f3f74722...) cut at row 8,375 (85.71%, d 8,168: past the
  platform walk at z ~1,565 over 82.15-83.6% and the drop through the hole in fall-catch sheet *265 at ~84.7-85.5%, on the
  lower level heading west at ~880 u/s; success line d <= 8,268 = 85.53%, reached only through the hole). Its first 7,451 rows
  are pbr7b2_k04c exactly. No demo, no record (pbr8b4, round 8).
* `pbr8b4_f1.npy` + `.keys.npy` (10,519 rows; md5 46a16fdd53f1b086dc49b6d50ca776d3 / b709399413bc3954a5312dd679bff8ef): the
  lineage's FIRST LINE TO THE FINISH (end d 10.8 at (819, 2,762, -3,424)): the trainer loop's own search-4 spine (run pbr7b2_c8,
  auto/k04: the PLAIN beam_tas line on c8's own 44.892B weights from the true spawn, its own held keys) to its row 9,185 (92.2%,
  inside the top of the finish tube), then ONE exact explore_phase1 --exact-chains chain (runs/la1 on the pbr8b4 box: reward-free,
  auto_spine's GE settings - 64 u x energy 32 cells, rising split, view-follow, 8 ms - rooted at 31 of k04's own rows 9,050-9,199,
  x1.0; the first goal hit after 0.8 min, ge_replay 21/21 links exact, 1,334 states) down the tube to the finish box. Keys = k04's
  own and the chain's translated (tools/stitch_spine.py). No demo, no record (pbr8b4, round 8).
* `pbr8b4_mix1.npy` (4,286 rows; md5 837a974e94dc6a893a67fccec13213c0): a --spawn-states pool, half and half: 2,143 rows drawn
  (seed 0, sorted) from `pbr1b1_entry_ge_states.npy` (start-room states from reward-free search lines) and pbr8b4_f1's rows
  8,376-10,518 (85.71% -> the finish: the loop's search-4 beam line on its own weights from k04d's end to 92.2%, then la1's
  reward-free Go-Explore chain down the finish tube). Every row is the policy's own line or a reward-free search from it; no
  demo, no record (pbr8b4, round 8). Swapped in for pbr8b4_c1's spawn-states pool on the box (atomic replace of the launch
  path; the trainer re-reads the file on an mtime change).
* `pbr8b4_mix2.npy` (4,094 rows; md5 below in NOTES): mix1 reweighted toward the finish tube's kill plates: 1,071 start-room
  rows (seed 1) + f1 rows 8,376-10,518 once + f1 rows 9,185-9,335 (the z ~250 plate approach) and 9,335-9,405 (the z ~-136
  plate) four more times each. Same provenance as mix1 (policy's own line + a reward-free chain from it); no demo, no record.
* `pbr8b4_b1.npy` + `.keys.npy` (10,877 rows; md5 60eeeaae0756377b441d2ddf48397375 / 24cf64917c8098f877a33c0fc870c3a6): a BRIDGE
  from the policy's own fast approach: its PLAIN beam_tas line on run pbr8b4_c1's own 48,487,202,816-step weights from the true
  spawn (the loop's settings, 4,096 envs; extinct at 83.0% on catch sheet *265) to its row 6,911 (80.20%, 1,118 u/s), then ONE
  exact explore_phase1 --exact-chains chain (runs/la2 on the pbr8b4 box: reward-free, auto_spine's GE settings, rooted at 37 of
  that beam line's own states over 78.8-80.9% x1.0, 10 min) that surfs up the east face, lands on the one-hull ledge at 82.5%,
  walks it, drops through the sheet's hole, flies the lower level and threads the finish tube's top plates and the hover zone
  to d 1,989 = 96.5% (ge_replay 50/50 links). Keys = the beam's own and the chain's translated (tools/stitch_spine.py). No demo,
  no record (pbr8b4, round 8).
* `pbr8b4_b1c.npy` + `.keys.npy` (9,101 rows; md5 46086c49bc753efa564f677e9ce8eb42 / bacb9f09d252fd5edbffbff513a6000a): b1 cut at
  row 9,100 (85.92%, on the lower level flying west past the hole; success line d <= 8,147 = 85.74%, reached only through the
  hole). No demo, no record (pbr8b4, round 8).
* `pbr8b4_b2.npy` + `.keys.npy` (11,715 rows; md5 bd970fc057a25987d5749e1de167121d / 44dc14fc0eb1b4be9b6606dba4d43ddd): b1 to its row
  10,755 (in the finish tube above the hover zone, z ~-500) + ONE exact explore_phase1 --exact-chains chain (runs/la5 on the
  pbr8b4 box: reward-free, auto_spine's GE settings, rooted at 41 of b1's own rows with z -256..-919, x1.0; the first goal hit
  after 0.2 min, ge_replay 9/9 links, 960 states) through the hover zone and the lower kill plates (the z -1,100 shaft, the
  z ~-1,700 chamber's ring, -2,700 / -2,850 / -3,200) to d 0 at (835, 2,774, -3,441) in the finish box. A finishing line from
  the true start: the policy's own fast beam line + two reward-free chains. No demo, no record (pbr8b4, round 8).
* `pbr8b4_mix4.npy` (4,137 rows; md5 546970dca2d072430e3781d8fa3e6c03): a --spawn-states pool for the tube curriculum, keeping
  the 80-86% stretch trained beside it: 600 start-room rows (seed 4) + b2 rows 6,800-7,211 (the policy's own beam approach at
  78.6-80.2% and la2's face surf to the ledge landing at 82.5%) x4 + b2 rows 7,212-9,100 (la2's ledge walk, the hole, the lower
  level to 85.9%) once: 15% / 40% / 46%. The policy's own lines and reward-free chains from them; no demo, no record.
* `pbr8b4_mix5.npy` (5,837 rows): mix4 + b2 rows 9,101-9,950 twice (la2's lower level from the hole's exit at 85.9% round the
  90.7% pocket into the tube's top at 91.9%). Same provenance; no demo, no record. Swapped in for c5's spawn pool (the mix4
  launch path on the box, atomic replace; the trainer re-reads on an mtime change).
* `pbr8b4_b3.npy` + `.keys.npy` (11,588 rows; md5 a68d4948fee4e1d518d56355ac74a946 / eaa017c87fb22f236f61a91383bbf367): b2 to row
  11,159 + the policy's OWN beam_tas line (runs/bm3b: c5's 51.094B weights, `--start-state pbr8b4_b2.npy:11160` = b2's hover exit,
  4,096 envs, the loop's settings, --save-frontier now saving a crossing winner) from b2's row 11,160 to the finish box in 428
  ticks (3.28 s, replay bit-exact). It replaces la5's lower tube, whose chamber passage clipped a slanted lip from 5 rows the
  policy cannot win from. Keys: b2's + the beam's own (auto_spine.keys_from_acts). No demo, no record (pbr8b4, round 8).
* `pbr8b4_b3p.npy` + `.keys.npy` (9,951 rows; md5 77dc4cb9032984de1d46be10c7552f28 / 150932bc86e7c17eabce9687052f7777): b3 cut at
  row 9,950 (91.79%, standing on the ledge inside the finish tube's top at (750, 2,983, 709), past la2's way round the 90.7%
  pocket; success line d <= 4,791 = 91.62%, out of reach from the pocket, whose states sit at d 5,230-5,407). No demo, no record.
* `pbr8b4_mix6.npy` (6,613 rows; md5 f9ad3bf1cb3f007147a7fd01384d72f6): mix5 + b3 rows 10,812-11,587 (the hover to the finish, the
  part of the tube the policy has learned) once, so the tube skill keeps training while a curriculum works elsewhere.
* `pbr8b4_c8k01.npy` + `.keys.npy` (9,600 rows): the trainer loop's own search-1 spine of run pbr8b4_c8 (auto/k01: the PLAIN
  beam_tas line on c8's own 52.63B weights from the true spawn, 4,096 envs, horizon 98.5 s; extinct at 74 s) to d 551 = 99.04%
  at (676, 2,798, -2,884), above the finish tube's last two kill plates; ducked through the whole tube. Its own held keys. No
  demo, no record (pbr8b4, round 8).
* `pbr8b4_fin1.npy` + `.keys.npy` (8,852 rows; md5 2a212d4b2225d0438ca4aa212fa51265 / f0526a2c95d9ecac9d0e45f96f01cc14): **the policy's
  own FINISHING line**: the PLAIN beam_tas line (runs/bm5 on the pbr8b4 box: run pbr8b4_c8's own 53,151,268,864-step weights, from
  the TRUE spawn, 4,096 envs, the loop's settings, --max-ticks 14000) that CROSSES the finish in 8,852 ticks = 67.87 s (replay
  bit-exact; saved by beam_tas --save-frontier's crossing case), its own held keys (auto_spine.keys_from_acts). No demo, no
  record, no search chain: every row is the policy's own population (pbr8b4, round 8).
* `pbr8b4_c9k02.npy` + `.keys.npy` (8,864 rows; md5 d83eecee3c710c651501e1cea6154e6b / ab935e3ee50482f9f6812ab9767140d2): the trainer
  loop's own search-2 spine of run pbr8b4_c9 (auto/k02: the PLAIN beam_tas line on c9's own 54.86B weights from the true spawn,
  4,096 envs) that FINISHES (8,864 ticks = 67.96 s), its own held keys. No demo, no record (pbr8b4, round 8).
