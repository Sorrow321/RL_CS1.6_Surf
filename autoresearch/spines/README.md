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
