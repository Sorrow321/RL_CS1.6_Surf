# RL_Surf

![surfing surf_ski_2 in the playable client](media/demo.gif)

*Human gameplay in the playable client (`python/play.py`) — the real `surfcore.dll`
physics at 100 Hz on surf_ski_2, 2000+ u/s lines. [Longer clip](media/demo.mp4).*

Train an RL agent to surf in CS 1.6. The GoldSrc/CS 1.6 movement physics
(ReGameDLL_CS `pm_shared`, vanilla path) is reimplemented as a standalone,
vectorized C environment that collides against real `.bsp` maps — so a policy
trained here can later be deployed on a real ReHLDS server with no sim2real gap.
Full design in [`docs/`](docs/00-overview.md).

## Status — env built, all gates green

| Gate | Result |
|---|---|
| A — BSP v30 loader + entities | surf_ski_2: 3396 planes, 7379 clipnodes, 50 models, 10 teleports, 7 pushes, 18 spawns ✓ |
| B — hull tracing (`PM_RecursiveHullCheck` port) | all invariants ✓ (floor rest z 41.03, 381 ramp-normal hits) |
| C — physics port + analytic tests | 8/8 ✓ incl. **real ramp slide: 196→584 u/s, never grounded** |
| D — vectorized env + benchmark | **1.42 M steps/s single-thread, 8.4 M steps/s on 32 threads** (256 envs, random actions, zero NaN) |
| E — visualizer (three.js) + map exporter | mesh + trajectory playback + waypoint editor ✓ |
| F — integration | Python binding 16/16 tests; 9.2 M steps/s at 1024 envs from Python; recorded surf demo `runs/ramp_slide.traj.jsonl` (597.9 u/s) ✓ |

## Training status (living log)

The agent genuinely surfs. Highlights of the program so far:

- **Blind era**: scalar-only MLP policies learned full-speed lane surfing
  (~6,100u path per 10s episode, 1,370 u/s peaks) from game-authentic
  platform spawns — but by memorizing coordinates, one groove per run,
  with physics-equivalent alien conventions (surfing sideways/backwards:
  air-strafe acceleration only depends on wishdir-vs-velocity angle).
- **Trainer**: custom GPU-resident PPO matches SB3's sample efficiency
  exactly (verified by config-clone bisection) at far higher throughput.
  Hard lessons encoded in the code: update density beats raw steps/s;
  `ep_rew_mean` must exclude the truncation bootstrap; the greedy argmax
  drifts weak under high entropy (record stochastic rollouts to see what
  the curve measures).
- **Eyes**: 6-head action space (yaw, view pitch, move, jump, duck) and a
  128x64 equiangular depth image rendered on the GPU from a precomputed
  map SDF (~6ms per 2048-env batch; exact per-ray traces were 300x too
  slow). First sighted runs promptly discovered *sensor collapse* —
  surfing while staring at featureless sky (constant input, zero gradient
  on where to look) — hence the asymmetric view clamp and fixed-gaze
  experiment modes.
- **Current experiment** (`--gps` off by default): absolute position and
  compass heading are hidden from the network — honest proprioception +
  vision only — so the policy must surf by sight instead of GPS-indexed
  memory. Network 2x'd where compute is free (GEMMs, not conv channels).
- **Tooling**: local W&B-style dashboard (live curves, rollout artifacts,
  one-click 3D replay with per-episode reward reconstruction, record-at-
  latest-checkpoint buttons) and first-person POV videos of exactly what
  the CNN sees (`tools/render_pov.py`, inline in the viewer).

## Quickstart

Linux (Ubuntu — training box):

```bash
sudo apt install build-essential            # gcc + libgomp
./build.sh --test                           # libsurfcore.so + full test suite + benchmark
python3 python/benchmark.py                 # throughput through the ctypes binding
python3 python/demo_ramp.py                 # record a surf slide (headless, no GL needed)
# play.py / the viewer work too (pip install pyglet; needs a display + OpenGL 3.2)
```

Windows:

```powershell
.\build.ps1                                  # MSVC: surfcore.dll + tests (needs VS2022)
.\build\test_trace.exe   maps\surf_ski_2.bsp # trace invariants
.\build\test_physics.exe maps\surf_ski_2.bsp # analytic physics tests
.\build\bench.exe        maps\surf_ski_2.bsp 256 2000

python python\benchmark.py                   # throughput through the ctypes binding
python python\demo_ramp.py                   # record a surf slide -> runs/ramp_slide.traj.jsonl
python python\demo_strafer.py                # scripted strafe bot, 256 envs

python python\play.py                        # PLAY IT: first-person, CS 1.6 bindings,
                                             # real DLL physics at 100Hz (see --help)

python python\train_speed.py                 # SB3 PPO baseline (docs/09), ~190k steps/s
python python\train_fast.py                  # GPU-resident PPO with depth VISION:
                                             # 128x64 SDF lidar (triton ray-march),
                                             # CNN policy, curriculum rewards, ~45k
                                             # steps/s; --ckpt resume; see --help
                                             # (scalar-only MLP era: ~900k steps/s)

# runs dashboard (local W&B-lite: metrics, artifacts, one-click playback)
python tools\dashboard.py                    # -> http://localhost:8000/

# bare viewer: serve and open http://localhost:8000  (auto-loads the map mesh;
# drag any runs/*.traj.jsonl in; press E for the waypoint editor)
cd viewer; python -m http.server
```

Regenerate the viewer mesh for another map: `python tools\export_map.py maps\<map>.bsp`.

## Layout

```
src/        C core: bsp.c trace.c pm.c env.c, surfcore.h (the ABI)
python/     surfgym package (ctypes, VecEnv/gym adapters, policies, recorder) + demos
tools/      export_map.py (bsp -> viewer mesh), make_sample_traj.py
viewer/     three.js trajectory player + waypoint editor (vendored three r147)
maps/       surf_ski_2.bsp + entity dump (+ your <map>.waypoints.json)
tests/      C test suites (Gates A-D)
docs/       design docs (00-06) + docs/research/ (sourced research)
third_party/ upstream engine reference sources — populate with tools\fetch_third_party.ps1
```

## License

This repository's code is **GPLv3** (matching the ReHLDS/ReGameDLL_CS lineage the
physics port derives from). Engine reference sources are fetched from their public
upstream repos rather than redistributed; full provenance and attributions in
[NOTICE.md](NOTICE.md). Non-commercial project, not affiliated with Valve.

## Ideas / roadmap

Exploration (the open problem — every run so far collapses to one route;
action-level entropy only widens the tube, it never jumps ramps):

1. **Intrinsic novelty (RND)**: learned bonus for visiting unfamiliar
   states — turns "never seen that ramp" into a reason to go. The
   structurally right fix; next in line.
2. **Weighted spawn mix**: mostly game-authentic platform starts + a
   minority of scattered map-wide starts, so other ramps acquire value
   estimates at all (pure exploring-starts already proved the mechanism).

Perception / architecture:

3. **Strided frame stacking** (3 depth frames ~50ms apart) to make
   ego-motion visually observable; store frames once, gather stacks at
   minibatch time (naive stacking triples the 8.6GB rollout buffer).
4. **Goal-conditioned surfing**: "start at X, reach Y minimal-time" — see
   docs/09 notes; ego-frame goal encoding + geodesic distance field over
   the SDF's free space as shaped reward.

Deployment / infrastructure:

5. ReHLDS replay parity harness and the Metamod fake-client deployment
   plugin (docs/05, docs/06 backlog).
6. Track spline authoring stays available (viewer `E`) for route-following
   rewards; current rewards are position-derived and spline-free.

Physics config is the CS 1.6 surf convention (`sv_airaccelerate 100`, gravity 800,
knife 250, 100 fps ticks) — every constant sourced in
[docs/01-physics-spec.md](docs/01-physics-spec.md).
