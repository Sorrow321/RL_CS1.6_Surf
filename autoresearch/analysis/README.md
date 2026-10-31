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

# uf2r1b2 (surf_unitfarmer2, round 1) analysis scripts (MEASUREMENT ONLY - never training)

Copied to the box and run from /root/RL_Surf (paths inside assume it): `uf2_ge_top.py ARCHIVE N`
(deepest cells on the grader field), `uf2_ge_chain.py ARCHIVE CELL` (a cell's chain), `uf2_ge_pitexit.py`
/ `uf2_ge_front.py` (chains that dove and came back; energy per progress band), `uf2_ge_pareto.py`
(earliest arrival per progress level), `uf2_ge_root_need.py` (root energy a deep exit needs),
`uf2_pit_bench_states.py` + `uf2_pit_bench_read.py` (bench a policy from given states with
record_ckpt --spawn-states), `uf2_line_profile.py` (a spine's path / energy / contact), `uf2_states_dist.py`
(a spawn file's progress / energy mix), `uf2_spawn_paths.py` (set_state vs spawn-pool restore),
`uf2_pumped_states.py` (a bench window). Local: `uf2_ep_ends.py` / `uf2_ep_profile.py` (eval episodes),
`uf2_wr_profile.py` (the WR demo as a YARDSTICK only, CLAUDE.md section 0).
