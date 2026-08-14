# 03 — Environment Design (C core + Python bindings)

The env is a batch of `N` independent surf players stepped in lockstep by one C call. All physics state is `float32`. All heavy work (traces, movement, obs, reward) happens in C; Python only moves numpy pointers.

## C API (`surfcore.h`)

> **The authoritative ABI is `src/surfcore.h`** — the sketches below are design-time
> illustrations; where they differ (field names, action slot order), the header wins.

```c
// ---- configuration -------------------------------------------------------
typedef struct SurfPhys {           // physics cvars — defaults = CS 1.6 surf convention ([01] §1)
    float sv_gravity;               // 800
    float sv_airaccelerate;         // 100 (engine default 10; 100 is THE surf setting)
    float sv_accelerate;            // 5   (CS stock config; engine default 10)
    float sv_friction;              // 4
    float edgefriction;             // 2
    float sv_stopspeed;             // 75  (CS stock config; engine default 100)
    float sv_maxspeed;              // 320 (ceiling only — not a velocity cap)
    float player_maxspeed;          // 250 (knife; scout 260, rifles 210-240, P90 245)
    float sv_maxvelocity;           // 2000 — the real cap, per-axis
    float sv_stepsize;              // 18
    float sv_bounce;                // 1 (enters player physics only as 1.0+bounce*(1-friction)=1.0)
    int   msec;                     // 10 → frametime = (float)(msec*0.001), computed engine-style
    int   enable_stamina;           // fuser2; default 1 (vanilla) — never binds mid-surf
    int   enable_bhop_cap;          // PM_PreventMegaBunnyJumping 1.2f/0.8; default 1 (vanilla)
} SurfPhys;

typedef struct SurfEnvConfig {
    int      num_envs;
    int      max_episode_ticks;     // truncation
    int      lookahead_k;           // spline lookahead points in obs (default 8)
    float    lookahead_dt;          // seconds between lookahead points (default 0.25)
    int      spawn_mode;            // 0 = map spawns, 1 = uniform along spline (curriculum)
    float    yaw_rate_max_deg;      // action clamp per tick (default 10)
    SurfPhys phys;
} SurfEnvConfig;

// ---- lifecycle -----------------------------------------------------------
SurfSim* surf_create(const char* bsp_path, const SurfEnvConfig* cfg, char* err, int errlen);
void     surf_destroy(SurfSim* s);
int      surf_obs_dim(const SurfSim* s);
void     surf_set_waypoints(SurfSim* s, const float* xyz, int count);   // polyline, map units

// ---- the hot path --------------------------------------------------------
// actions: int32 [N x 5]  (see Action space)
// obs:     float32 [N x obs_dim]   rewards: float32 [N]
// done/trunc: uint8 [N].  Auto-reset: a done env returns its *new* episode's
// first obs in `obs` this same call, and its TRUE final observation in
// terminal_obs[i] (garbage for non-done envs). Both the SB3 adapter
// (terminal_observation info) and the gymnasium >=1.0 next-step shim need it —
// Python cannot reconstruct it after the in-place reset.
void surf_step(SurfSim* s, const int32_t* actions,
               float* obs, float* rewards, uint8_t* done, uint8_t* trunc,
               float* terminal_obs);
void surf_reset_all(SurfSim* s, uint64_t seed, float* obs);

// ---- state access (recording, curriculum, debugging) ---------------------
typedef struct SurfState {          // one env, plain float32/int32 — POD
    float origin[3], velocity[3], basevelocity[3];
    float yaw;                      // pitch fixed at 0 — irrelevant to horizontal wishdir
    float stamina;                  // fuser2
    int32_t onground, ducked, jump_held, tick, seg_hint;
    float progress, best_progress;
} SurfState;
void surf_get_states(SurfSim* s, SurfState* out);            // [N]
void surf_set_state (SurfSim* s, int env, const SurfState*); // teleport an env (curriculum)

// ---- exposed internals for tests ----------------------------------------
void surf_trace(SurfSim* s, const float* start, const float* end, int usehull,
                /*out*/ float* endpos, float* plane_normal, float* fraction,
                int32_t* startsolid, int32_t* allsolid);
// usercmd-level single step — the primitive the parity harness ([05] tier 2) replays
// recorded server usercmds through; discrete actions cannot represent arbitrary floats.
void surf_pm_step_usercmd(SurfSim* s, SurfState* st, float yaw, float pitch,
                          float fmove, float smove, int32_t buttons, int32_t msec);
void surf_pm_step_single(SurfSim* s, SurfState* st, const int32_t action[5]); // thin discretizing wrapper
// (SurfState will grow flDuckTime/bInDuck/waterjumptime + full oldbuttons when tier-2 replay lands)
```

Notes:
- `SurfSim` owns one immutable BSP + one waypoint spline, shared read-only by all envs and all threads. Different maps → different `SurfSim` instances.
- `surf_step` is `#pragma omp parallel for` over envs. Envs never interact → embarrassingly parallel, no locks.
- `surf_pm_step_single` + `surf_trace` exist so unit tests and the parity harness can drive physics without the env wrapper.

## Per-env state (in C, AoS)

One `~128 B` struct per env (the `SurfState` above plus duck timers and cached plane normal). AoS beats SoA here: a step touches all of one env's fields and traces dominate runtime anyway; cache behavior is fine.

## Action space — `MultiDiscrete([15, 3, 3, 2, 2])`

| Slot | Meaning | Values |
|---|---|---|
| 0 | yaw delta this tick | 15 bins, non-uniform, **ascending**: `−10, −7, −4, −2, −1, −0.5, −0.25, 0, +0.25 … +10` (index 7 = zero), scaled so the largest bin equals `yaw_rate_max_deg` (default ±10°/tick = 1000°/s at 100 fps — top-human flick range; q1physrl used ±26°/frame and that kept its record-beating demos looking human-legal). Accumulated yaw wraps into [0, 360) each tick before building viewangles ([01 §5](01-physics-spec.md)). |
| 1 | forwardmove | −1/0/+1 → `−400 / 0 / +400` (rarely needed mid-surf; needed at starts) |
| 2 | sidemove | −1/0/+1 (A / none / D) |
| 3 | jump | IN_JUMP held or not — **the engine jump is edge-triggered** (must see a release before re-jump; see [01](01-physics-spec.md)), the env passes the raw button and lets ported `PM_Jump` handle `oldbuttons` |
| 4 | duck | IN_DUCK held or not (v1 may compile ducking out — see cut lines in [06](06-night-plan.md)) |

Non-uniform yaw bins give fine control near 0 (smooth air-strafing is gentle, continuous yaw-velocity coupling) while keeping snap turns available. Pitch is frozen: GoldSrc horizontal `wishdir` is built from yaw only while airborne.

If PPO struggles with the coarse bins, plan B is hybrid: continuous `Box(-1,1)` yaw + MultiDiscrete rest (CleanRL makes this easy, SB3 does not). Start discrete.

## Observation (v1: track parameterization) — ~62 floats

All vectors expressed in the **player's local yaw frame** (rotation by −yaw about z): egocentric ⇒ policy is invariant to world orientation. Normalization constants in brackets.

| Block | Dim | Contents |
|---|---|---|
| velocity_local | 3 | player velocity rotated into yaw frame [÷1000] |
| speed_h, vel_z | 2 | horizontal speed magnitude [÷1000], vertical velocity [÷1000] — speed matters separately because air-strafe gain depends on it |
| flags | 3 | onground, ducked, jump_held (0/1) |
| track error | 3 | signed lateral offset, height offset from spline [÷500], fraction of track remaining `1 − progress/total_length` |
| heading vs track | 2 | sin, cos of (yaw − spline tangent yaw at nearest point) |
| lookahead | 8×6 = 48 | K=8 points at 0.25 s spacing of *anticipated travel* ahead along the spline: relative position in ego frame (3) [÷2000] + spline tangent dir in ego frame (3). This is what lets the agent set up ramp transitions early — the core skill of surfing. |
| last yaw action | 1 | previous tick's yaw delta [÷10°] |
| **total** | **62** | |

Lookahead spacing is *arc-length* based: point `i` sits at `progress + max(speed_h, 500) * lookahead_dt * (i+1)` along the spline — adapts to current speed so fast agents see farther.

**v2 (deferred): ray sensors** for cross-map generalization — 16–32 hull traces in a fixed egocentric fan (downward/outward ring + forward cone), each returning hit fraction + hit-normal (ego frame). Costs ~10× env throughput; a config flag, not a redesign — the trace function is already there.

## Track spline

- A polyline of waypoints (`maps/<map>.waypoints.json`), authored by clicking in the visualizer ([04](04-visualizer.md)); 30–80 points for a typical linear surf map.
- Preprocessing (C, at `surf_create`/`surf_set_waypoints`): cumulative arc length per vertex, unit tangents.
- **Progress** = arc length at the projection of player origin onto the polyline. To survive self-approaching layouts (folded ramps), projection searches only segments `[seg_hint − 2, seg_hint + 4]`; `seg_hint` advances monotonically per env and resets on teleport/spawn. No global nearest-segment search — O(1) and immune to snapping across folds.
- `progress`, `best_progress` tracked per env; the spline's total length marks completion.

## Reward

```
r_t = 0.05 * (progress_t − progress_{t−1})        # dense, ~+1.0/tick at 2000 u/s
    + 50.0  * [reached end]                        # completion bonus, then done
fail (hurt trigger / teleport-to-start / fell out) → done, no explicit penalty:
    lost progress is already the penalty
truncation at max_episode_ticks (progress stall optional later)
```

Delta-progress can be briefly negative (backwards motion) — that's correct and wanted. No per-tick time penalty: progress delta already prices time.

## Episode logic

- **Spawn** (`spawn_mode=0`): random `info_player_*` spawn, yaw = `SpawnPoint.yaw` + small seeded jitter. `spawn_mode=1` (curriculum): uniform-random progress point on the spline, placed at spline height + 20u, yaw = spline tangent yaw (+ jitter), velocity = tangent × modest speed, `seg_hint` = the sampled segment, then settled by one downward trace. Both modes zero the rest of the state: `fuser2`, `basevelocity`, `oldbuttons/jump_held`, `ducked`, timers. Curriculum mode is how you train long maps without waiting to re-reach ramp 7 — q1physrl used exactly this (99% randomized starts, 1% true-spawn starts, with true-spawn reward as the eval metric); adopt that split and report `spawn0_return` as the headline number.
- **Fail** → `done`: entering a `trigger_hurt` with lethal damage; a `trigger_teleport` whose destination is *behind* current progress (jail/restart detection: destination progress < current − 500u); origin below the map's kill-z; **waterlevel ≥ 2** (swimming is out of scope — on surf maps water = missed the ramp); or **stuck**: `PM_CheckStuck` true / FlyMove allsolid for 5 consecutive ticks (otherwise a trapped env silently freezes, burning ticks until truncation and poisoning the batch).
- **Benign teleports** (stage transitions): teleport the state, remap `seg_hint`/`progress` to destination, episode continues.
- **Triggers** (v1, *exact*): **run AFTER the physics tick**, matching the engine's post-move `SV_TouchLinks` ([01 §2](01-physics-spec.md) step 5 — a booster affects motion starting the *next* tick; testing before/inside the move breaks the basevelocity fold). AABB precheck then the real engine containment test — `HullPointContents(trigger_model.hull[player_hull], player_origin − trigger_origin) == CONTENTS_SOLID` ([02 §4](02-bsp-collision.md)). Teleport behavior is faithful `TeleportTouch`: dest by targetname, `z −= mins.z, z += 1`, yaw from dest angles, velocity and basevelocity zeroed, no re-test that tick. `trigger_push` boosters set basevelocity per [01 §6](01-physics-spec.md) — surf_ski_2 alone has 7 of them.

## Python layer (`python/surfgym`)

- `ctypes` binding (stdlib, no build coupling; releases the GIL during `surf_step` so OpenMP owns all cores). Set `argtypes`/`restype` explicitly — the classic x64 pointer-truncation bug. Load the DLL by absolute path (`Path(__file__).parent`), `os.add_dll_directory()` for `libomp.dll`/`vcomp140.dll` if needed; `WinError 126` almost always means a *dependency* DLL, check `dumpbin /dependents`. numpy arrays allocated once, raw pointers passed each step — zero copies, zero allocation in the loop.
- `SurfVecEnv` — SB3 `VecEnv`-compatible (same-step autoreset, `terminal_observation` in infos).
- `SurfGymVecEnv` — subclasses `gymnasium.vector.VectorEnv` *directly* (never `SyncVectorEnv`-of-1s); shims gymnasium ≥1.0's next-step autoreset convention over the C core's same-step behavior, for CleanRL compatibility.
- `gymnasium` single-env wrapper (`SurfEnv`) around `num_envs=1` for API compliance and debugging.
- `surfgym.policies.ScriptedStrafer` — sinusoidal strafe-sync bot (yaw oscillation phase-locked to sidemove sign). Not RL: the physics smoke test. If the strafer gains speed on a ramp, the port breathes.
- `surfgym.record` — roll any policy, dump trajectory JSONL for the viewer.

## Determinism & precision

- All physics in `float32`; compile with `/fp:precise` (MSVC) or `-ffp-contract=off` (clang) — no FMA contraction, no reassociation ⇒ run-to-run and machine-to-machine reproducibility for the same binary.
- No RNG inside physics. Env-level RNG (spawn jitter) = per-env `splitmix64` counters from the reset seed.
- Fixed `frametime`; no wall-clock anywhere.
- Engine parity is *qualitative tonight, bit-level later* — see [05-validation.md](05-validation.md).

## Performance budget

Per env-tick: ~10² flops of movement math + **2–6 hull traces** (FlyMove bumps, ground categorize, edgefriction probe) + O(K) spline math. Clipnode traces on a surf map (7 k clipnodes) visit tens of nodes ≈ a few hundred ns each.

Calibration: q1physrl's pure-NumPy vector env trained 150 M steps in ~a day *including PPO* (~1.7 k steps/s end-to-end); EnvPool-class C envs reach 10⁵–10⁶ steps/s on desktops. Realistic expectation for this sim: **0.5–2 M steps/s single-threaded** at batch ≥1024.

| Milestone | Gate (conservative) |
|---|---|
| Single thread, spline obs | ≥ 100 k steps/s (expect 0.5 M+) — same number as Gate D in [06](06-night-plan.md) |
| All cores (OpenMP, `schedule(static)`) | ≥ 1.5 M steps/s |
| With 32-ray obs (v2) | ≥ 100 k steps/s aggregate |

The true success criterion is "env is never the bottleneck vs. the PPO learner" (≥ 200–500 k steps/s). If single-thread lands under ~50 k, profile traces first (flat contiguous clipnode arrays, `restrict`, explicit-stack recursion). Synchronous batch stepping only — async send/recv buys nothing for a constant-cost tick.

## Config surface kept honest

Everything a surf server could set lives in `SurfPhys` — nothing hardcoded that ReHLDS exposes as a cvar. Deployment day = read the server's cvars into `SurfPhys`, retrain nothing.
