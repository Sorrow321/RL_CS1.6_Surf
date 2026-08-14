
## prior_art_markdown
# Prior art: q1physrl (Matthew Earl)

**Repo:** [matthewearl/q1physrl](https://github.com/matthewearl/q1physrl) — RL agent that beat the human WR on Quake 1's "100m" speedrun practice map. Closest existing analog to this project.

## Structure
- Two packages: `q1physrl_env` (pip-installable, pure-NumPy physics + gym env, no engine dependency) and `q1physrl` (RLlib training/analysis code), plus `docker/` utilities that replay trained-agent actions back into the real Quake engine to produce `.dem` files. **Copy this split**: env package with zero training deps, trainer package on top, and a separate "replay into the real game" path for validation.
- Key insight in `q1physrl_env/env.py`: the core class is `VectorPhysEnv`, which is **natively batch-vectorized** — physics written as NumPy ops over an `(N, ...)` batch, with a `vector_step()` taking batched actions and returning batched obs/rew/done. The single-env `PhysEnv` is just a batch-of-1 wrapper. He did in NumPy exactly what you're planning to do in C; your C core is the same design one level down.

## Observation design
6-dim float box, normalized by `[time_limit, 90, 100, 200, 200, 200]`:
1. time remaining in episode, 2. player yaw, 3. z position, 4–6. velocity x/y/z.

Notably **no x/y position and no map geometry** — 100m is an open runway, so speed is the whole state. For surf you cannot get away with this: you'll need at least position + some local geometry encoding (e.g., ramp-relative frame, raycast probes, or per-map checkpoint progress).

## Action design (worth stealing wholesale)
Tuple of:
- 3 discrete keys: strafe-left, strafe-right, forward (+ optional jump key if `allow_jump=True` and `auto_jump=False`);
- mouse-x as a **continuous yaw delta clipped to ±26°/frame** (`action_range`), or optionally discretized (`discrete_yaw_steps`).

Human-plausibility constraints he baked in: `key_press_delay=0.3s` rate-limits key re-presses; `smooth_keys` half-applies movement on key-transition frames (mimics Quake's input averaging). The ±26°/frame mouse cap is what kept the agent's demos looking legal/human-comparable.

## Reward + episode design
- Dense per-frame reward: `time_delta * forward_velocity` (option `speed_reward=True` uses 2D speed magnitude instead). No sparse goal reward at all.
- `time_delta = 1/72 s` (Quake demo tick), `time_limit = 10 s` episodes.
- **Randomized starts as implicit curriculum**: only `zero_start_prob=1%` of episodes start from the true deterministic spawn; the rest start from randomized states. The reported metric is `zero_start_total_reward_mean` (~5700 at convergence). There's also a `hover` config (gravity off) for isolated air-strafe practice — directly analogous to a "hover on a ramp" curriculum stage for surf.

## Training setup + throughput
- RLlib PPO on Ray; converged after ~150M env steps, ~1 day on an i7-6700K → roughly **1.5–2k steps/s end-to-end** (env + PPO learner) from a pure-Python/NumPy env. This is the number your C core should beat by 2–3 orders of magnitude on the env side.

## Lessons transferable to surf
- Vectorize natively; never step envs one at a time from Python.
- Dense velocity reward alone is enough to learn strafe acceleration — for surf, expect to need it *plus* progress/checkpoint terms since velocity alone won't route through a map.
- Keep a path to replay agent actions in the real engine (for CS 1.6: via a TAS tool / BXT script or a bot input injector) to measure sim divergence — he did demo-file generation for exactly this reason.
- Constrain actions to human-plausible rates from day one if you ever want the output comparable to real runs.

(His blog matthewearl.github.io hosts a companion post and the video ["Teaching a computer to strafe jump in Quake with reinforcement learning"](https://www.youtube.com/watch?v=hx7kvTZLHYI); the [HN thread](https://news.ycombinator.com/item?id=23052152) has author Q&A but was rate-limited during this research.)


## binding_recommendation_markdown
# C-core batch env from Python

## Binding: use ctypes on a plain-C ABI DLL (skip pybind11 for a one-night build)
- **ctypes (recommended)**: stdlib, no build-system coupling, works with any Python version, and zero-copy is trivial — allocate all buffers as NumPy arrays on the Python side and pass raw pointers:
  ```python
  lib.step_batch(state, acts.ctypes.data_as(POINTER(c_float)),
                 obs.ctypes.data_as(POINTER(c_float)),
                 rew.ctypes.data_as(POINTER(c_float)),
                 done.ctypes.data_as(POINTER(c_uint8)), c_int(n))
  ```
  Always set `argtypes`/`restype` explicitly (silent 32-bit truncation of pointers is the classic x64 ctypes bug). ctypes releases the GIL during the call, so the C core can use all cores freely. Per-call overhead (~1–2 µs) is irrelevant amortized over a 1024-env batch.
- **cffi**: equivalent capability (ABI mode), slightly faster call overhead, but an extra dependency for no real gain here.
- **pybind11**: what EnvPool uses, and the right answer if you later want C++ classes, async send/recv, or buffer-protocol objects created C-side — but it drags in a C++ toolchain, per-Python-version binary compatibility, and CMake time you don't have tonight.

## C API shape
One opaque `SimState*` holding **SoA float32 arrays** (`pos[3][N]`, `vel[3][N]`, `yaw[N]`, `on_ground[N]`, per-env RNG state) plus the loaded map (BSP clipnodes). Exports: `create(n, map_path)`, `reset_all(seed)`, `reset_indices(idx*, k)`, `step_batch(...)`, `destroy`. Give each env its own PCG32 RNG so results are independent of thread count.

## Gymnasium VectorEnv compliance
Subclass `gymnasium.vector.VectorEnv` **directly** (do NOT wrap N single envs in `SyncVectorEnv` — that defeats the whole point). You must provide `num_envs`, `single_observation_space`, `single_action_space`, and implement `reset(seed, options)` and `step(actions) -> (obs, rew, terminated, truncated, infos)` with batched arrays. Gymnasium ≥1.0 uses **next-step autoreset** semantics (env returns final obs with `terminated=True`, then the reset obs on the following step) — implement that convention inside the C core and you're drop-in compatible with CleanRL and most modern trainers. Docs: [gymnasium.farama.org/api/vector](https://gymnasium.farama.org/api/vector/). Also look at PufferLib if you want a shim to many trainers.

## Threading: OpenMP inside the C core, not Python multiprocessing
- Python multiprocessing costs pickling/IPC per step and duplicates map data per process. A `#pragma omp parallel for schedule(static)` over envs inside `step_batch` shares the (read-only) BSP, has zero serialization cost, and scales linearly for a uniform-cost sim.
- [EnvPool](https://github.com/sail-sg/envpool) ([paper, ICLR 2022](https://arxiv.org/pdf/2206.10558)) lessons: native thread pool + batched API is what reaches ~1M FPS; their async `send/recv` with `batch_size < num_envs` exists to fix the *long-tail* problem of variable-cost envs (Atari). A surf physics tick is near-constant cost, so **synchronous batch stepping is within a few percent of async** and much simpler. Skip async for v1.

## Windows build specifics
- MSVC: `cl /LD /O2 /fp:precise /openmp surf.c /Fe:surfcore.dll`. MSVC's `/openmp` is OpenMP 2.0 only — `parallel for` is fine; use `/openmp:llvm` if you need more. If you compile as C++, wrap exports in `extern "C" __declspec(dllexport)`.
- clang-cl also works (`clang-cl /O2 /LD /openmp -ffp-contract=off ...`) and gives you better FP control; it links against LLVM's libomp (ship `libomp.dll` next to your DLL).
- **ctypes loading pitfalls**: since Python 3.8, Windows DLL *dependency* resolution ignores `PATH` — load your DLL by absolute path (`ctypes.CDLL(str(Path(__file__).parent / "surfcore.dll"))`) and call `os.add_dll_directory()` for any directory containing dependent DLLs (e.g., `libomp.dll`, or `vcomp140.dll` if not in system32). Use `CDLL` (cdecl); on x64 there's only one calling convention anyway. A cryptic `WinError 126` usually means a *dependency* of your DLL wasn't found, not the DLL itself — check with `dumpbin /dependents`.


## visualizer_markdown
# Visualizer

## Existing browser GoldSrc renderers
- **[hlviewer.js](https://github.com/skyrim/hlviewer.js)** — the library that already does *most* of this: loads GoldSrc BSP v30 maps with WAD textures and skies **and plays .dem replays entirely in-browser**, with play/pause/scrub UI. Fork [WebHL](https://github.com/x8BitRain/webhl/) loads assets from local disk via the File System Access API. Caveats: oldish codebase, its playback is tied to the .dem format, so feeding it your own JSONL trajectories means writing a custom camera/entity driver against its internals. Best option if you want textured maps for free; overkill for tonight.
- **[bspview](https://github.com/sbuggay/bspview)** (sbuggay) — three.js + TypeScript Quake/GoldSrc map explorer with basic texture/lightmap support; small codebase, good place to **crib a BSP v30 parser** if you want in-browser parsing.
- BSP→standard-format exporters (recommended path): **[lewa-j/hlbsp-converter](https://github.com/lewa-j/hlbsp-converter)** converts GoldSrc BSP (v30) to **glTF** including textures; **[newbspguy](https://github.com/UnrealKaraulov/newbspguy)** is a GoldSrc BSP viewer/editor that exports **OBJ**; **[GoldImporter](https://github.com/Stalker2106/GoldImporter)** imports BSP into Blender if you need manual cleanup.

## Do textures matter?
No. For RL trajectory debugging, **flat shading by face normal is strictly better than textures**: color faces by `n.z` so surf ramps (GoldSrc treats plane normal `z < 0.7` as non-walkable → surfable) render in a distinct color from floors. One `MeshLambertMaterial` + vertex colors, done.

## Recommended minimal design (one night, ~300 lines of three.js)
1. **Offline**: `hlbsp-converter map.bsp` → `map.gltf` once per map. No BSP code in the browser.
2. **Data**: env writes trajectories as JSONL — one object per tick: `{t, pos:[x,y,z], vel:[vx,vy,vz], yaw, keys, done}` — or, for long runs, a single binary `Float32Array` blob with a tiny JSON header (much smaller/faster). Static files, no server beyond `python -m http.server`.
3. **Scene**: `GLTFLoader` + flat-shaded material (Z-up: rotate scene -90° about X or set `camera.up`); `THREE.Line` polyline of the full path with per-vertex color mapped to speed (instant visual of where the agent gains/loses velocity).
4. **Playback**: frame index + fractional interpolation between ticks; controls = play/pause, scrub slider, speed (0.25×–8×), step ±1 tick. Keep it in a rAF loop with `simTime += dt * speedMul`.
5. **Cameras**: OrbitControls free mode + follow mode (`camPos = pos - forward(yaw)*150 + [0,0,60]`, lerped at ~10 Hz equivalent) toggled with a key.
6. **Overlays**: `ArrowHelper` at the player for velocity (length ∝ |v|), a small box/capsule for the player hull (32×32×72), HUD `<div>` with speed in ups + tick number. Optionally render N trajectories at once with different hues to compare policy checkpoints.

This is deliberately dumber than hlviewer.js — but it's fully yours, trivially supports overlaying arbitrary agent data, and each piece (GLTFLoader, OrbitControls, Line, ArrowHelper) is stock three.js.


## perf_notes_markdown
# Performance expectations & float32 determinism

## Realistic steps/sec
- Baselines: q1physrl's pure-NumPy vector env achieved ~150M steps in ~1 day on an i7-6700K **including PPO training** (~1.7k steps/s end-to-end). [EnvPool](https://arxiv.org/pdf/2206.10558) reaches ~1M FPS (Atari) / ~3M FPS (Mujoco) on a 256-core DGX; on ordinary desktops it reports ~50–100k FPS Atari.
- A surf tick is far cheaper than either: ~a few hundred FLOPs of movement math + one hull trace (BSP clipnode traversal, tens of node visits; up to 4 traces in PM_FlyMove when clipping along ramps). Expect roughly **0.5–2M env-steps/s single-threaded** in optimized C with batch ≥1024, hull tracing dominating; **several million steps/s on an 8-core desktop with OpenMP**. Practical planning number: budget conservatively for **≥1M steps/s total**, and treat "env is never the bottleneck vs. the GPU PPO learner" (>200–500k steps/s) as the actual success criterion. At a 100 Hz tick, 1M steps/s is 10,000× realtime — q1physrl's entire 150M-step training run would take ~2.5 minutes of env time.
- Biggest perf levers, in order: batch size per call (amortize call overhead + cache-warm map data), clipnode trace efficiency (keep the node array flat and hot), SoA float32 layout, OpenMP static scheduling. Don't bother with SIMD hand-tuning night one.

## float32 determinism (train on many machines, replay anywhere)
- **The enemy is FMA contraction and fast-math**, which make results differ across compilers and across FMA/non-FMA CPUs:
  - MSVC: use `/fp:precise` (default) or `/fp:strict`; **never `/fp:fast`**. See [MSVC /fp docs](https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior). x64 MSVC uses SSE2 scalar math, which is IEEE-reproducible.
  - clang/clang-cl: pass **`-ffp-contract=off`** explicitly — clang's default (`on`) may fuse `a*b+c` into FMA, giving different low bits per machine. Also avoid `-ffast-math`/`/clang:-ffast-math`.
  - gcc (if cross-compiling for Linux training boxes): `-ffp-contract=off`, avoid `-Ofast`; on x86-64, SSE2 is default so no x87 excess-precision issue.
- Also: don't let the compiler auto-vectorize reductions differently per build (there are none to speak of in this sim if you keep per-env code scalar); avoid `libm` transcendentals in the hot loop where cross-platform bit-equality matters — yaw handling needs `sin/cos`, and libm implementations differ between MSVC UCRT and glibc, so if you need bit-identical Windows↔Linux replays, ship your own `sinf/cosf` (e.g., a small polynomial) — otherwise accept per-OS determinism only.
- Per-env RNG state (PCG32) and OpenMP `schedule(static)` over independent envs guarantee results are independent of thread count and scheduling.
- Perspective: GoldSrc itself was built with old MSVC x87 quirks; **bit-exact parity with the real engine is unattainable and unnecessary** — target a self-consistent deterministic sim, then validate statistically against HLStrafe/BXT predictions and real-engine replays (q1physrl's approach).


## sources
https://github.com/matthewearl/q1physrl

https://github.com/matthewearl/q1physrl/blob/master/q1physrl_env/q1physrl_env/env.py

https://www.youtube.com/watch?v=hx7kvTZLHYI

https://news.ycombinator.com/item?id=23052152

https://arxiv.org/pdf/2206.10558

https://github.com/sail-sg/envpool

https://gymnasium.farama.org/api/vector/

https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior

https://github.com/skyrim/hlviewer.js

https://github.com/x8BitRain/webhl/

https://github.com/sbuggay/bspview

https://github.com/lewa-j/hlbsp-converter

https://github.com/UnrealKaraulov/newbspguy

https://github.com/Stalker2106/GoldImporter

https://github.com/rehlds/ReGameDLL_CS

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.cpp

https://github.com/HLTAS/hlstrafe

https://github.com/YaLTeR/BunnymodXT

https://github.com/ratmarrow/GoldSrc-Character-Controller

https://forums.unrealengine.com/t/free-plugin-full-c-port-of-goldsrc-quake-movement-physics-ue-5-1/2702987

https://github.com/Olezen/UnitySourceMovement

https://github.com/khanghugo/dem

