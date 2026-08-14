# 06 — One-Night Build Plan

Target: by morning, a scripted strafe bot demonstrably surfs a real map in the visualizer, and a vectorized env runs ≥100 k steps/s ready for PPO. Times assume Claude Code doing the typing; milestones are hard gates, hours are estimates.

## T−0 · Preflight — ✅ mostly done during research
- `third_party/` already holds the vendored references (HLSDK `pm_shared.c`/`pm_math.c`, ReGameDLL_CS `pm_shared.cpp/.h`/`pm_math.cpp`/`triggers.cpp`, ReHLDS `world.cpp`/`pmovetst.cpp`/`pmove.cpp`/`model.cpp`, hlbsp `bspdef.h`).
- `maps/surf_ski_2.bsp` already downloaded (+ entity-lump dump alongside).
- Remaining: toolchain check (`cl`/`clang-cl`, Python 3.10+, numpy), vendor three.js into `viewer/vendor/`.

## A · BSP loader + entities (~1.5 h)
`src/bsp.c`: v30 header/lumps → in-memory arrays (planes, clipnodes, models, nodes/leafs, render lumps), entity-lump key-value parser.
**Gate A:** CLI dump for the chosen map: model count, hull headnodes, spawn points, trigger entities with their brush AABBs. Numbers sane vs. what the map should contain.

## B · Hull tracing (~1.5 h)
`src/trace.c`: `PM_RecursiveHullCheck` port — the **pmove** variant per [02 §3](02-bsp-collision.md), not the SV_ one (they differ: empty-hull guard, 0.05 vs 0.1 backup) — explicit stack, hull selection, brush-model trace offset, point-contents incl. the contents-entity scan.
**Gate B:** trace tests pass — vertical down-trace from a spawn hits floor at plausible z with normal (0,0,1); trace into a ramp returns the ramp's tilted normal; start-in-solid reports `startsolid`. Print-compare a handful of traces by eye against the map geometry in the (not-yet-interactive) exported mesh.

## C · Physics port (~2–2.5 h) — the heart
`src/pm.c` from [01-physics-spec.md](01-physics-spec.md): tick pipeline with split gravity, friction+accelerate (ground), airaccelerate + FlyMove/ClipVelocity (air), jump with edge-trigger + bhop cap + stamina (all three are a handful of lines each, exact formulas in [01 §4](01-physics-spec.md); flags default vanilla-on), categorize position, maxvelocity clamps, basevelocity fold for trigger_push. Duck behind a config flag, OFF tonight.
**Gate C (analytic, from [05](05-validation.md)):** jump apex ≈ 45 u; flat-ground friction decay matches closed form; single-tick air-accel delta matches formula; ramp steady-state slide matches analytic solution. All four in `tests/test_physics.c` (asserts, not eyeballs).

## D · Env + binding (~1 h)
`src/env.c` (batch step, obs, reward, autoreset per [03](03-env-design.md)) + `python/surfgym` ctypes wrapper + `SurfVecEnv`.
**Gate D:** random-action rollout runs 10 k ticks × 256 envs without NaN/crash/stuck-in-solid; **benchmark**: single-thread ≥100 k steps/s (spline obs), OpenMP scaling visible.

## E · Visualizer (~1.5 h)
`tools/export_map.py` + `viewer/` playback per [04-visualizer.md](04-visualizer.md).
**Gate E:** load map mesh + a random-policy trajectory, scrub it, follow-cam it. *The demo moment:* run `ScriptedStrafer` spawned above a ramp — it must visibly gain speed sliding the ramp. If it does, the physics is real.

## F · Track + first training smoke (~1 h)
Waypoint-edit the spline for the map in the viewer → `surf_set_waypoints` → spline obs + progress reward live. Fallback if E slips (F's only dependency on E): hand-write `maps/<map>.waypoints.json` from the spawn and teleport-destination coordinates that Gate A already dumps — the format ([04](04-visualizer.md)) is a bare point list, and surf_ski_2's lane needs ~15 points.
**Gate F:** `ScriptedStrafer` reward curve is positive along the first ramp; PPO (SB3, defaults, 256 envs) *launches and learns something nonzero* for 10 min. Training to competence is tomorrow's job.

## Cut lines (pull these first when behind)
1. Ducking (fixed standing hull) — fine for most classic ramps.
2. Benign stage-teleport remapping — night one may treat *every* teleport as episode fail (surf_ski_2's teleports are jail/fail anyway).
3. trigger_push basevelocity handling — cut only if the chosen route uses no boosters; log a warning when a pushed state is hit unimplemented.
4. Ghosts/sparkline in viewer.
5. edgefriction probe (keep plain friction) — restore before serious training.
6. Water/point-contents (fail on func_water AABB entry + kill-z instead; then stub `PM_CheckWater` to `waterlevel = 0; watertype = CONTENTS_EMPTY; return false` — the pipeline calls it unconditionally, and the stub makes CategorizePosition's snap-down guard always pass).

**Never cut:** split-gravity ordering, ClipVelocity/FlyMove fidelity, float32 discipline, the four analytic tests. That's the difference between "a physics-ish toy" and "the CS 1.6 movement model".

## Morning-after backlog
Ray obs (v2) · curriculum spawns tuning · `.npz` trajectories · trigger containment for *rotated* brush entities (axis-aligned exact containment ships in v1) · ReHLDS replay parity harness ([05](05-validation.md)) · Metamod fake-client deployment plugin · multi-map training.
