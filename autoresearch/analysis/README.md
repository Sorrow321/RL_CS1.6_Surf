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

# r5b2 tools (round 5; ANALYSIS ONLY - never training)

`autoresearch/analysis/r5b2/`, run from the repo root on CPU (`CUDA_VISIBLE_DEVICES=-1`). No matplotlib
on the dev machine: `png.py` writes PNGs with zlib.

* `xz_img.py OUT X0 X1 Z0 Z1 YLO YHI TRAJ EPS TMIN PMIN` / `xy_img.py OUT X0 X1 Y0 Y1 Z1,Z2,.. PMIN` /
  `yz_img.py OUT Y0 Y1 Z0 Z1 X1,X2,.. PMIN` (env `RES` = u per pixel): sections of the goal field's
  reachable air (colour = progress %, black = solid / unreachable), kill volumes in red (hull_probe:
  the standing-hull origins that die), eval episodes overlaid (labels = seconds).
* `descent.py X Y Z [EVERY]`: the field's steepest-descent route from a point (the field ignores
  triggers: check the route against the kill volumes).
* `eprep.py TRAJ...`: per eval episode - best progress, where, speed, end.
* `attempts.py OUT.npy TRAJGLOB...`: the greedy line's own LAUNCH-ATTEMPT states (0.2-0.6 s before the
  first tick rising at vz > 300 near *160's holes) as STATE_DTYPE.
* `macro2.py CKPT SEED N1 REPS N2`: two-stage random-held-macro search on the raw core from a ckpt's
  greedy-pool room states: into the shaft, then from the post-launch states to the funnel.
* `search_att.py STATES.npy N REPS SEED SCALE [FAST_OUT.npy]`: random held macros (5 segments, 4 s)
  from STATES with the velocity x SCALE: shaft entries, entries at vz > 600, funnel, finish depth; dumps
  the plans of the fast entries and (FAST_OUT) the fast post-launch states for a policy probe
  (`record_ckpt.py --spawn-states FAST_OUT.npy`).
* `roomstats.py RUN_DIR...`: per run, the greedy evals that reach the final room - entry speed, launch
  attempts (max vz, max |y|), shaft entries, finishes.
