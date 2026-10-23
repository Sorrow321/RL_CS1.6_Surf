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

## surf_princessburglar_b26 (pbr1b4, round 1) - champion-free artifacts of the start-trap pass

None of these came from a human demo: a reward-free Go-Explore archive from the TRUE START,
the goal field, and the lineage's own policy-guided search (CLAUDE.md section 0 provenance).

* `pb_goexplore_chain73.npy` - STATE_DTYPE, root first: the tools/explore_phase1.py archive's
  chain from the true start to its best goalc-field cell (3.21%: start -> conveyor plate ->
  south crawl tunnel -> the chamber floor). Built by tools/archive_spawns.py. STITCHED (each row
  is its cell's best state), so the crux rows are poor spawn states (see NOTES 17:31).
* `pb_goexplore_chain_north.npy` - the chain into the north tunnel's shallowest archive cell
  (1.95 s deep: a running duck-jump from the floor at (758, -2944)).
* `pb_self_route.npz` - tools/self_route.py: chain73 + the goalc field's steepest descent to the
  finish (448 pts @ 128 u); the --route-file of tools/beam_tas.py --objective progress and the
  --route of tools/plan_to_bc.py.
* `pb_spine_s0/s4/s5.npy` - the best replayed beam_tas lines (spawn seeds 0, 4, 5) from the TRUE
  START into the chamber (per-tick STATE_DTYPE, physically continuous; written by
  tools/plan_to_bc.py --spine). The climb windows (500 ticks before the first tick at x < 600,
  z > 3690, to 100 after) x4 + the whole spines were the --spawn-states of pbr1b4_bc2.
