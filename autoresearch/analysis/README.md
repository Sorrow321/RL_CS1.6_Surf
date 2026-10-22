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

# r5b3 analysis scripts (2026-10-22, ANALYSIS ONLY)

Both run locally from the repo root on the CPU (no checkpoint needed for the first).

* `sid_killprobe.py {xy|xz|yz} a0 a1 da b0 b1 db c [duck]` - kill / solid / free map of a plane, from the
  simulator's own trigger test (holes in kill plates show up; the bsp AABBs hide them). Found the 95% slit,
  the hole in kill *10, the second thin wall's underpass, the closed chimney above the finish.
* `sid_macro_climb.py <ckpt> [n] [seed]` - random held macros on the raw core from the policy's OWN lower-lane
  states (its reservoir + its greedy-eval arrivals): can anything climb out of the lane into the chimney?
  Result on the 15.67B checkpoint: 0 / 2,048 finish, 1 above z 1,260.
