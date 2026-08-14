# RL_Surf — Design Overview

**Goal:** train an RL agent to surf in CS 1.6. To iterate fast, we replicate the *exact* GoldSrc/CS 1.6 movement physics as a standalone, vectorized environment (C core + Python bindings), train against real surf `.bsp` maps, and later deploy the policy on a real ReHLDS server via a fake-client plugin — where the identical physics code runs, so there is no sim2real gap by construction.

**Tonight's scope:** the environment + visualizer, not the trained agent.

## Why this works

GoldSrc player movement is deliberately isolated in one module (`pm_shared.c`) with no dependency on the rest of the game. Its only external need is a collision-trace function against the map. Surfing is not special-cased anywhere: a ramp face whose plane normal has `z < 0.7` can never set "on ground", so air acceleration applies while `PM_ClipVelocity` slides velocity along the ramp plane — gravity's along-ramp component does the rest. Port `pm_shared` + BSP hull tracing and you have surf, bit-for-bit.

Precedent: [q1physrl](https://github.com/matthewearl/q1physrl) did exactly this for Quake 1 (same physics lineage) — reimplemented movement as a vectorized env, PPO'd ~150M steps in a day on a desktop CPU, beat the human speedrun record; [surfski2.com](https://www.surfski2.com/) reimplements GoldSrc surf standalone in a browser. Our C core should out-throughput q1physrl's NumPy env by 2–3 orders of magnitude.

## Repo state

`third_party/` already contains the vendored reference sources (ReGameDLL_CS `pm_shared.cpp` — the CS 1.6 authority — plus HLSDK, ReHLDS engine trace/hull/wrapper files, hlbsp format spec, `weapons.h`, a canonical surf server cfg). `maps/surf_ski_2.bsp` (the most-played CS 1.6 surf map) is downloaded, with its entity dump beside it. `docs/research/` holds the full sourced research behind every constant in these docs.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                surfcore.dll (C)              │
 maps/*.bsp ──────▶ │  bsp.c    BSP v30 loader + entity parser     │
                    │  trace.c  clipnode hull tracing (PM_Recur…)  │
                    │  pm.c     pm_shared port (move/jump/surf)    │
                    │  env.c    N envs, batch step, obs/reward     │
                    └───────▲──────────────────────┬───────────────┘
                            │ ctypes (zero-copy numpy)
                    ┌───────┴──────────────────────▼───────────────┐
                    │  python/surfgym: VecEnv + gymnasium adapters │
                    │  training (PPO: SB3 / CleanRL)               │
                    └───────┬──────────────────────────────────────┘
                            │ trajectory JSONL          ┌──────────────────┐
                            └─────────────────────────▶ │ viewer/ three.js │
 maps/*.bsp ── tools/export_map.py ── mesh JSON ──────▶ │ playback + edit  │
                                                        └──────────────────┘
 Later: policy → Metamod fake-client plugin on ReHLDS (deployment, out of scope tonight)
```

## Repo layout

```
RL_Surf/
  docs/               this design (+ docs/research/: sourced research dumps)
  src/                C core: bsp.c trace.c pm.c env.c surfcore.h
  python/surfgym/     ctypes binding, VecEnv/gym adapters, scripted test policies
  tools/              export_map.py (bsp → viewer mesh), make_waypoints.py
  viewer/             index.html + vendored three.js, trajectory player
  maps/               .bsp files + entity dumps + <map>.waypoints.json
  tests/              analytic physics tests, trace tests
  third_party/        vendored reference sources (ReGameDLL_CS, HLSDK, ReHLDS) — read-only
```

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Physics source of truth | ReGameDLL_CS `pm_shared` (CS 1.6 variant), HLSDK as cross-ref | CS 1.6 differs from HL (stamina, bhop cap, duck speeds) — see [01-physics-spec](01-physics-spec.md) |
| Collision | Real BSP clipnode hull tracing against the actual map file | Zero geometry approximation → zero domain gap; ~200 LOC |
| Core language | C11, single DLL, OpenMP over envs | One-night buildable, trivial ctypes binding, no deps |
| Precision | `float` (32-bit) everywhere, `/fp:precise` | Matches engine `vec3_t`; no FMA/reassociation drift |
| Tick | Fixed 10 ms (100 fps equivalent) | GoldSrc physics is frametime-dependent; 100 fps is the community-standard reference (see [01](01-physics-spec.md)) |
| Obs (v1) | Spline/track parameterization + lookahead, egocentric | Racing-RL standard; dense, Markovian, cheap. Ray sensors deferred (v2, for cross-map generalization) |
| Actions | MultiDiscrete (yaw-delta bins + side/fwd + jump + duck) | Pure-discrete keeps PPO plumbing trivial in SB3/CleanRL |
| Visualizer | Static mesh export + three.js trajectory player in browser | No engine needed to *see* the agent; doubles as waypoint editor |

## Document map

- [01-physics-spec.md](01-physics-spec.md) — the `pm_shared` port spec: tick pipeline, every function, every constant, server cvars for surf. **The most important doc.**
- [02-bsp-collision.md](02-bsp-collision.md) — BSP v30 loading, hull tracing, entities (teleports/hurt/spawns).
- [03-env-design.md](03-env-design.md) — env API, state layout, obs/action/reward, vectorization, performance budget.
- [04-visualizer.md](04-visualizer.md) — mesh export, trajectory format, three.js player + waypoint editor.
- [05-validation.md](05-validation.md) — analytic unit tests, engine parity plan.
- [06-night-plan.md](06-night-plan.md) — hour-by-hour build order with cut lines.

## Non-goals (tonight)

Ladders, water swimming, weapons/stamina-accurate loadouts beyond a maxspeed constant, multiplayer, netcode, the Metamod deployment plugin, training to convergence. Each has a marked hook where it slots in later.
