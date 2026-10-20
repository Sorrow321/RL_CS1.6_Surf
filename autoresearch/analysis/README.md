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

## r2b3 (round 2) analysis scripts - ANALYSIS ONLY, never training

* `sid_slot_shoot.py <traj> <ep> <t,...> [n] [secs] [seed]` - random held-usercmd rollouts through
  the live-trigger physics from a recorded episode's own states; reports window crossings, how
  low and how far any continuation gets (CAN'T vs WON'T past the CP2 slot). Runs here (CPU).
* `sid_goexplore_probe.py <traj> <ep> <t,...> [iters] [explore_s] [seed]` - Go-Explore in the
  simulator (cell archive, 1/sqrt(1+chosen) returns, random exploration) seeded with a recording's
  states; prints cells below the slot and climbs back into the shaft.
* `sid_arch_regions.py <ckpt>` / `sid_arch_speeds.py <ckpt>` / `sid_tunnel_states.py <ckpt>` - read a
  --respawn-rare-arch checkpoint's archive by region (measurement boxes), its speeds, and its states
  in the window tunnels (with the ballistic exit height). Copy to /root/RL_Surf on the box.
* `sid_surf_states.py <n> <seed> <out.npy> [helixside]` - surf-aligned states from geometry alone
  (SDF + hull + field) in the region under the slot, for `record_ckpt --spawn-states` benches.
