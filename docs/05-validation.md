# 05 — Validation

Two tiers: **analytic tests tonight** (physics matches closed-form math), **engine parity later** (physics matches a real ReHLDS server tick-for-tick). The port is only trustworthy once both pass; tonight's bar is tier 1.

## Tier 1 — Analytic unit tests (`tests/test_physics.c`, Gate C)

All with `frametime = 0.01`, default surf cvars from [01](01-physics-spec.md). Tolerances are small multiples of float32 eps at the value's scale, not "close enough" fudge.

1. **Jump apex.** From flat ground, jump velocity `v0 = Q_sqrt(2.0·800.0f·45.0f)` computed in double, stored float32 = **268.328155517578125** (`0x43862A01`). With the engine's split-gravity integrator *and* PM_Jump's own FixupGravityVelocity call ([01 §3](01-physics-spec.md)), simulated apex height must match the discrete closed form derived in the test (the half-step scheme lands within ~1 u of the continuous 45 u; assert the discrete-exact value, tolerance ~1e-3 u).
2. **Friction decay.** On flat ground, speed 250, no input: each tick `v ← v·(1 − friction_eff·ft·stopspeed_factor)` per the exact PM_Friction formula; simulate 100 ticks, compare against the same recurrence computed independently in the test in float64, assert ≤1e-2 u/s drift.
3. **Air-accel single tick.** Airborne, speed `v`, wishdir at angle θ to velocity: post-tick velocity must equal the closed-form `PM_AirAccelerate` update (30-cap regime and sub-cap regime — two cases). This test alone catches most transcription bugs in the most important function.
4. **Ramp steady state.** A synthetic infinite ramp (single plane, normal `(sinα, 0, cosα)` with `cosα < 0.7`): player holding no input slides; the per-tick acceleration must equal the ClipVelocity∘gravity composition **`a⃗ = g·(−ẑ + n_z·n̂)`, magnitude `g·√(1 − n_z²) = g·sin α`** (the projection of gravity onto the plane — NOT `g·(1 − n_z²)`, which is only its vertical component and is 29% low at 45°). Assert the per-tick velocity delta vector matches this exactly (modulo the STOP_EPSILON band).
5. **ClipVelocity table.** Hand-built cases: head-on into wall (velocity zeroed along normal, STOP_EPSILON behavior), 45° graze, floor landing, steep-ramp graze — expected outputs computed by hand in the test file.
6. **Stationarity.** AFK on flat ground for 1000 ticks: origin drift = 0 exactly; onground stays true; no NaN. Then AFK on a 30° (walkable) slope: must *not* slide (GoldSrc has no walkable-slope sliding).
7. **Bhop cap & stamina (when flags on).** Jump at 400 u/s ground speed with knife (maxspeed 250): `maxscaledspeed = 1.2f·250 = 300.0f` exactly, `< 400` → velocity scaled by float `(300/400)·0.8 = 0.60000002384f` ⇒ post-jump speed **240.0000153…** (one ulp above 240 — `0.8` is inexact in binary; assert the ulp-exact value, not `== 240.0f`). Stamina: set `fuser2 = 1315.789429` directly via `surf_pm_step_usercmd` state (a literal "immediate second jump" can't observe the full value — ReduceTimers decays it and the edge trigger needs a release tick); the jump's `v_z` scale, evaluated **left-to-right in double as the code does**, is `(100.0 − fuser2·0.001·19.0)·0.01 = 0.7500000085449219` — just *above* 0.75; pre-folding the constants gives 0.7499999985 and is the wrong association. Horizontal velocity shrinks by the same formula every ground frame while `fuser2 > 0`.
8. **Trace invariants** (`tests/test_trace.c`, Gate B): down-trace onto known floor; ramp normal readback; startsolid detection; trace symmetry (A→B fraction 1 ⇒ B→A fraction 1 in open space); brush-entity offset traces.

## Tier 2 — Engine parity (after tonight)

The decisive test: **bit-replay against ReHLDS.**

1. Tiny Metamod/ReAPI plugin: for a chosen player, log per tick `msec, buttons, viewangles, forwardmove/sidemove` (the usercmd) and post-move `origin, velocity, onground`.
2. A human (or the bot via fake-client) plays the surf map on a stock surf-configured ReHLDS.
3. Feed the recorded usercmd stream through `surf_pm_step_usercmd` (the float-level entry — discrete actions can't represent recorded usercmds) starting from the recorded initial state, on the same `.bsp`, same cvars.
4. Compare trajectories: assert max |Δorigin| over a 60 s run. Goal: < 1 u drift over 60 s. Perfect bit-parity with retail is unattainable (the original cs.so was an x87 build; ReGameDLL itself needs double `real_t` intermediates to match it — we reproduce those sites per [01 §7](01-physics-spec.md), which gets drift down to float-noise), so the actionable signal is *step discontinuities* in the divergence curve: a jump in drift at one tick = a missed branch (duck edge case, stuck nudge, edgefriction probe, msec-timer truncation), not accumulated rounding. Chase those, accept smooth micro-drift.

This same plugin skeleton later *is* the deployment vehicle (fake client driven by the policy), so tier 2 work is not throwaway.

## Continuous guards

- `surfgym` debug flag: NaN/inf scan on state after every batch step (cheap, off for training).
- Fuzz: 64 envs × 100 k random-action ticks nightly; assert no *persistent* stuck-in-solid (the env's 5-tick stuck rule in [03](03-env-design.md) must fire instead) and `max_i |v_i| ≤ sv_maxvelocity + ε` **per axis** — a magnitude assert would false-fail correct physics, since legal diagonal speed reaches ~2828 horizontally.
- Determinism check in CI: same seed, same binary → byte-identical trajectory dump twice.
