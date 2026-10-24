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

## pbr2b1 (round 2)

* `pbr2b1_line_b.npy` (2,024 states, md5 18713796bc1f1e79f152f50d1f5c8b86): `pbr1b1_line_a.npy`
  states 0-1200 (unchanged) + an 823-state tail from line_a's state 1201 (over the pit, mid-flight)
  to 3.81% on the grader field (the SOUTH side above the pit, (619, -3,312, 3,432)). The tail is
  `tools/explore_phase1.py --view-vel --exact-chains` (reward-free uniform bursts in the trainer's
  velocity-frame view: random keys + a view offset from a table dense near 0; 64 u x --cell-energy
  32 cells, 4,096 envs, 12 min on a 5080 box) rooted at line_a's own flight states 1150-1245 (every
  3rd, yaw re-aimed along the velocity), replayed exactly by `tools/ge_replay.py` (9/9 links; chain
  end = archive cell 3258) and joined by `tools/splice_line.py`. No demo, no record data. Its views
  are the policy's kind (median 9 deg off the velocity at >= 100 u/s); line_a's head keeps its
  random-walk views (use `--demo-view-vel`).
* `pbr2b1_north_line.npy` (763 states, md5 986cc9a57986cf839ae9cc7650b0cf87): a building block for
  the NORTH route (not trained on yet). States 0-596 = the policy's OWN greedy episode from the true
  start (pbr2b1_lb2 ckpt_latest 4.446B via `record_ckpt --dump-states`, episode 0: start room,
  tunnel, ridge launch, flight over the pit); then a 166-tick tail from that episode's state 597
  (over the pit, h ~3,720) to the pit's NORTH side, (634, -2,687, 3,431) = 3.78% at 579 u/s:
  `explore_phase1 --view-vel --exact-chains` rooted at 160 of the policy's own over-the-pit states,
  replayed by `ge_replay` (5/5 links, cell 15292) and joined by `splice_line`. At this energy the
  north route cannot clear the block at 6.7% (NOTES.md, the feasibility check); it needs ~+130 u
  of energy height (the record's pump + strafed tunnel) first.
