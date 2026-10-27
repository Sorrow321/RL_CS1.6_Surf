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

# pbr7b2 (round 7) analysis scripts (ANALYSIS ONLY - never training; CLAUDE.md sections 0 / 0b)

* `pb_energy_profile.py LEVELS traj.jsonl [...]` - per episode, at the first arrival at each progress level: time, energy
  height z + |v|^2/1600, speed and position. Run it on an eval and on a reference (the policy's own beam line saved as a
  trajectory, or the WR demo `$AR_HOME/incumbent/wr/pb_wr.jsonl` as a yardstick) to see where a gate's energy is lost.
