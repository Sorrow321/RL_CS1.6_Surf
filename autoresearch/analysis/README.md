# r1b1 analysis scripts (ANALYSIS ONLY - never training; CLAUDE.md sections 0 / 0b)

Both run ON THE BOX from /root/RL_Surf (copy the .py files and the .npy there), the box idle
(record_ckpt is CPU-bound: ~2 min for 20 episodes idle, ~4 min for 3 beside a trainer).

* `sid_cp2_nudge_sweep.py <ckpt> <out_dir> [ticks] [holds]` - WON'T vs CAN'T at the CP2 pit:
  one greedy episode per (tick, hold, nudge) through `record_ckpt.py --nudge-*`, scored on the
  goal field. Result on the wall checkpoint (905M): no short yaw deviation crosses.
* `sid_corridor_bench.py <ckpt> <tag> [n_stoch] [n_greedy] [states.npy]` - start the policy
  in the corridor under kill plate *45 moving east (its OWN frontier states moved there, the
  way --respawn-novel moves them; `sid_corridor_east.npy` both lanes, `_py.npy` the +y lane)
  and report how far east it gets, whether it reaches the ramp (x > 5,300) and how far west
  it comes back (`xret`). Map coordinates are measurement boxes only.

pbr7b4 (round 7) additions, ANALYSIS ONLY:

* `pb_energy_at.py <traj> L1,L2,...` - each greedy episode's energy height at its first arrival at the given
  progress levels (where the greedy loses a gate's energy).
* `pb_bench_spine.py CKPT SPINE LO HI OUT [N] [L1,L2] [HMIN]` - on the box: record_ckpt greedy episodes from exact
  spine rows [LO, HI), the best progress and the energy at two levels per episode, and how many cross L2 with
  h >= HMIN. Benching just before and just after a contact separates "the contact is missing" from "what follows it
  is missing". It also runs locally from the worktree root (the maps are symlinks to the main checkout's prebaked
  fields): ~35 s for 8 short episodes, a few minutes for long ones (CUDA_VISIBLE_DEVICES=-1 nice -n 10).
* `pb_approach.py L1,L2,... traj.jsonl|spine.npy[:lo:hi] ...` - the state at the first arrival at each progress level
  (position, |vxy|, heading, vz, h), per episode or spine: where two lines part at a gate.
