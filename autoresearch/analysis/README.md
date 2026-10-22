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

# r6b4 analysis scripts (2026-10-22; ANALYSIS ONLY except where a file says it builds own-state spawns)

* `sid_end_bench_demo.py` - MEASUREMENT ONLY: spawn a checkpoint from the WR demo's final-stretch states
  (record_ckpt --spawn-states) and report finishes / hole entries per stage. Never a training input.
* `sid_uturn_stats.py` - per greedy episode: the first basement U-turn's strafe offset (view yaw - velocity
  heading), strafe key share, max z, min speed, and the west leg's speed / height / lane.
* `sid_cem_macros.py` - CEM over held strafe macros from a state of the policy's own greedy eval (deceptive:
  converges on the hole-edge plateau).
* `sid_goexp_macros.py` - Go-Explore in the simulator (SurfCore.play_step) rooted at the policy's own greedy
  eval states; cells of position / speed / heading; random held strafe macros; optional energy-biased
  selection and archive continuation. `sid_goexp_path_states.py` rebuilds an archive path by exact replay and
  writes its states as STATE_DTYPE (an own-state archive file for --spawn-states, box.py --self-states).
Paths inside the scripts point at the r6b4 worktree / scratchpad; adjust before reuse.
