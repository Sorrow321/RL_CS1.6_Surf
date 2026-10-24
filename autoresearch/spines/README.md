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

# pbr2b4 (round 2)

* `pbr2b4_chain3b.npy` (298 states, md5 946dc2ee28caa699581088c4408e081e): an exact per-tick chain
  from `pbr1b1_line_a.npy` state 1058 (the drop onto the ridge, ~650 u/s) to 3.95% on the grader
  field: surf the ridge east at ~780 u/s, launch north-east over the pit, turn on the far east
  side, fly west along the north side. `tools/explore_phase1.py --exact-chains` (reward-free
  uniform bursts, 64 u x --cell-energy 32, 4,096 envs, K 4, 100 decisions, --maxvel 3500, 12 min)
  rooted at line_a states 900-1250 (stride 2), then `tools/ge_replay.py --cell 16115` (9/9 links).
* `pbr2b4_spine_b.npy` (1,916 states, md5 df440ae6f70437b72e3f7fe40c5f9b4d): the backward
  curriculum's spine from the TRUE START = pbr1b1_g2cur6's own greedy line from the true start
  (record_ckpt --dump-states, episode 2, ticks 0-559, into the crouch tunnel) + line_a states
  0-1057 (the tunnel run-up) + chain3b. Junctions within 8 u. 1,574 of 1,916 states ducked.
* `pbr2b4_spine_c.npy` (1,146 states, md5 0816a498b603f76771a04e8874f5e12d): pbr2b4_cur1's own
  greedy line from the true start (ckpt 4.327B, record_ckpt --dump-states, episode 0, ticks 0-598:
  start room, tunnel slide, ridge launch, far-side turn) + an exact-chain GE chain (GE5: rooted at
  that line's last 3 s, 64 u x energy 32, 4,096 envs, 11 min; the near-deepest cell with the
  shortest chain, 546 ticks; ge_replay 16/16 links) west along the north side over the fall catch,
  down at x ~200 and west to 6.67% at (-1,008,-3,124,3,045), where it arrives nearly stopped.
* `pbr2b4_platform_chain.npy` (2,154 states, md5 40317f67523add07979e6c91384290ad): an exact-chain GE
  chain (64 u x energy 32, 6 min, ge_replay 24/24 links) rooted at pbr2b4_cur2's OWN greedy
  launch-ramp states (ticks 860-960 of its 4.861B line, x -219..-769) with the velocity SCALED x1.5
  (the recipe's front-perturb speed idea, a speed-handicap probe): from (-415,-2,771,2,918) |v| 1,135
  up onto a platform at z ~3,328 (state ~150), slowly west along it (to x ~-1,558, states 150-1,650),
  then down to 10-12.6% (-3,303,-3,384,2,790). Rows 150+ are physical (no scaling after the root).
  cur4's --spawn-states = pbr1b1_entry_ge_states + 3 x rows 150-2,153 (runs/pbr2b4/_ge6/
  spawn_entry_platform.npy, md5 267696e9...).
* `pbr2b4_spine_d.npy` (898 states, md5 fbc3f8298fad8bf7b96e9ca803ae12f1): pbr2b4_cur4's own greedy
  line from the true start (5.656B weights, record_ckpt --dump-states, episode 0, ticks 0-664) + an
  exact-chain GE chain (GE rim1: roots = that line's own states 1.0-0.1 s before its largest one-tick
  energy drop, 64 u x --cell-energy 16, 50-decision bursts, 10 min; ge_replay 8/8 links, 233 ticks)
  that passes the rim ~75 u higher, keeps the energy and reaches the wall top at 7.03%
  (-1,001,-2,983,3,231).
* `pbr2b4_spine_e.npy` (900 states, md5 8da828ad3f813a50cfe3424300952c13): pbr2b4_cur6's own greedy
  line from the true start (6.777B, episode 1, ticks 0-663) + GE8's chain (pb_frontier_step --roots
  sink: rooted at that line's own states 1.0-0.1 s before its largest one-tick energy drop - the
  rim - 64 u x --cell-energy 16, 50-decision bursts, 10 min; shortest chain within 32 u of the
  deepest, 236 ticks, ge_replay 12/12) to the wall top at 7.09% (-1,001,-2,996,3,252).
* `pbr2b4_spine_f.npy` (1,356 states, md5 88cd5d53177236fceb376b0c97e086cc): pbr2b4_cur10's own
  greedy line from the true start (8.369B, episode 2, ticks 0-1016: start room, tunnel, ridge launch,
  far side, rim cleared, platform jump, platform route) + GE11's chain (--roots sink at that line's
  largest energy drop after the platform, tick 1,061; 48 u x --cell-energy 16, 100-decision bursts,
  10 min; 338 ticks, ge_replay 12/12) to 12.74%.
