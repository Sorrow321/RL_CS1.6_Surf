# 01 — Physics Spec: the `pm_shared` port

**Source of truth:** ReGameDLL_CS `regamedll/pm_shared/pm_shared.cpp` — the decompile-faithful CS 1.6 game DLL — taking only the *vanilla* path (everything inside `#ifdef REGAMEDLL_ADD` / `REGAMEDLL_FIXES` is NOT retail CS 1.6 and is excluded). HLSDK `pm_shared.c` is the cross-reference. Engine-side behavior (frametime, hulls, writebacks) from ReHLDS. All reference sources are vendored in `third_party/` (fetched from the repos listed at the bottom).

## 1. Server configuration — the surf convention

Verified engine defaults (read from ReHLDS source — vendored: `third_party/engine_sv_phys.cpp` / `engine_sv_main.cpp` / `engine_sv_user.cpp` / `engine_host.cpp`) and what CS 1.6 surf servers actually run. The "CS stock" values 75/5 come from the shipped cstrike `server.cfg` convention (community-corroborated, see `docs/research/research_cvars.md`), not the engine. **These are the `SurfPhys` defaults.**

| cvar | engine default | CS 1.6 surf standard | role in movement |
|---|---|---|---|
| `sv_airaccelerate` | 10 | **100** — "a must in all surf servers" (UltimateSurf cfg); the single defining surf setting. Easy/fun servers go higher (150–800). | `PM_AirAccelerate` rate. At aa=100, knife: `accelspeed = 100·250·0.01 = 250 ≫ 30`, so the 30-unit projection cap saturates **every frame** → per-frame air steering is effectively instantaneous; the yaw rate becomes the skill bottleneck (and higher client fps gains real speed through turns). |
| `sv_gravity` | 800 | 800 | per-frame `v_z −= g·dt` (split half/half). Along-ramp gravity component is the engine of surf speed. |
| `sv_friction` | 4 | 4 | ground friction only — never runs mid-surf (`onground == -1`). |
| `edgefriction` | 2 | 2 | friction ×2 when a probe 16u ahead / 34u below the hull bottom finds no floor — i.e. on every ramp-top launch platform lip. |
| `sv_stopspeed` | 100 (engine) | **75** (CS stock config; the value CS servers actually run) | friction floor at low speeds; ground only. |
| `sv_accelerate` | 10 (engine) | **5** (CS stock config) | ground acceleration; start platforms only. |
| `sv_maxspeed` | 320 | 320 (some legacy cfgs 900 — behaviorally identical, see below) | ceiling for the per-player cap; **not** a velocity cap. |
| weapon speed | — | **250 (knife)** — surfers hold knife. Scout is 260, pistols 250. | `pev->maxspeed → pmove->clientmaxspeed`; the effective `pmove->maxspeed = min(250, sv_maxspeed)`. |
| `sv_maxvelocity` | 2000 | 2000 | **the real speed ceiling**, clamped PER AXIS in `PM_CheckVelocity` (diagonal can reach ~2828 horizontally). |
| `sv_stepsize` | 18 | 18 | step-up height in `PM_WalkMove`. |
| `sv_bounce` | 1 | 1 | enters player physics only via FlyMove's `1.0 + bounce·(1 − pmove->friction)` overbounce path = exactly 1.0 with player friction 1. |
| `fps_max` (client) | 100 | 99.5/100 → **the physics dt**. | GoldSrc physics is client-frame-timed: `usercmd.msec` (integer ms) is the dt. 100 fps → `msec = 10` exactly, zero truncation — the community reference rate. `sys_ticrate` does **not** affect player movement. |

Env default: `msec = 10`, `frametime = (float)(msec * 0.001)` — computed exactly as the engine does (`= 0.009999999776…f`), never a `0.01` double.

Stamina (`fuser2`) and the bhop cap are **vanilla-on** for combat surf; both bind only at jump/ground events, so pure ramp surfing never touches them. Modern movement servers (ReHLDS + ReGameDLL) can disable via `sv_enablebunnyhopping` / `sv_autobunnyhopping` / stamina plugins. Env: both toggleable, default on.

## 2. Engine wrapper duties (what `env.c` does around `pm.c`)

Per usercmd, ReHLDS `SV_RunCmd`:
1. `msec > 50` → command chopped into two halves `(byte)(msec / 2.0)` — a physics step never exceeds 50 ms. (Env uses fixed msec 10; keep the rule for robustness.)
2. **Basevelocity fold** (`SV_CheckMovingGround`): if `FL_BASEVELOCITY` is *not* set (no push/conveyor touched us since last frame): `velocity += (frametime*0.5f + 1.0f) * basevelocity; basevelocity = 0`. Always clear the flag afterwards; trigger_push Touch re-sets it each frame the player is inside.
3. Fill pmove: `usehull = (flags & FL_DUCKING) ? 1 : 0`, `maxspeed = sv_maxspeed`, `clientmaxspeed = pev->maxspeed` (250 knife), `oldbuttons`, `fuser2`, `frametime`.
4. Run `PM_PlayerMove` (below), then write back. Two writeback quirks that matter: `pev->flDuckTime = (int)pmove->flDuckTime` (**int truncation** every command) and `pev->oldbuttons = pmove->cmd.buttons` — the pm-internal `oldbuttons |= IN_JUMP` latches only live within one command; **the effective jump edge-trigger is "previous cmd had no IN_JUMP"** (why humans scroll-wheel bhop).
5. **AFTER the writeback: trigger touch** (`SV_LinkEdict → SV_TouchLinks`). This ordering is load-bearing: `CTriggerPush::Touch` sets `basevelocity` + `FL_BASEVELOCITY` *post-move*, so a booster first affects motion on the **next** tick — which is exactly why step 2's fold rule works (`FL_BASEVELOCITY` still set = "we were inside a push field last tick"). Teleports likewise move the player post-move and are not re-tested that tick. An env that tests triggers before or inside the pm tick gets one-tick-shifted boosts and a broken fold.

## 3. `PM_PlayerMove` — the per-tick pipeline (CS order)

```c
PM_CheckParameters();          // clamp |{fwd,side,up}move| to maxspeed; build pmove->angles (yaw > 180 → −360)
pmove->frametime = pmove->cmd.msec * 0.001;      // float; byte msec
PM_ReduceTimers();             // duck/step/swim ms timers −= msec; CS: fuser2 −= msec (floor 0)
AngleVectors(angles, forward, right, up);        // pm_math conventions, §5
if (spectator) { ... return; }                   // out of scope
if (PM_CheckStuck()) return;   // CS: NO duck-retry (HL retries once with PM_Duck)
PM_CategorizePosition();       // §4 — waterlevel + onground; snap-down
if (onground == -1) flFallVelocity = -velocity[2];
PM_Duck();                     // CS order: Duck BEFORE UpdateStepSound
// (ladder hook here — out of scope)
// MOVETYPE_WALK main path:
if (!PM_InWater()) PM_AddCorrectGravity();       // first half-gravity + basevelocity[2] consumption
// (waterjump / swim hooks here — out of scope)
if (cmd.buttons & IN_JUMP) PM_Jump();            // JUMP BEFORE FRICTION — the bhop mechanism
else oldbuttons &= ~IN_JUMP;
if (onground != -1) { velocity[2] = 0; PM_Friction(); }
PM_CheckVelocity();
if (onground != -1) PM_WalkMove(); else PM_AirMove();
PM_CategorizePosition();                         // re-evaluate ground AFTER the move
velocity -= basevelocity;      // pull push/conveyor velocity back out
PM_CheckVelocity();
if (!PM_InWater()) PM_FixupGravityVelocity();    // second half-gravity
if (onground != -1) velocity[2] = 0;             // final ground clamp
```

**Split gravity (leapfrog):** `AddCorrectGravity`: `v_z −= (ent_gravity · g · 0.5f · dt); v_z += basevelocity_z·dt; basevelocity_z = 0`. `FixupGravityVelocity`: `v_z −= (g · dt · ent_gravity · 0.5)`. The two halves round differently: `ent_gravity` is double (real_t) and *leads* the Add expression (all multiplies double), but in Fixup the leading `g · dt` is a float×float product **rounded to float** before the double enters; literals differ too (`0.5f` vs `0.5`). Copy both verbatim. Position integrates with mid-frame velocity. `PM_Jump` overwrites `v_z` (discarding the already-applied first half) and calls `FixupGravityVelocity` itself; the pipeline-end fixup still runs → exactly one full gravity per frame and correct 45u apex. Emulating "one gravity per frame" naively breaks jump height by half a tick.

## 4. Function specs (exact)

### PM_CategorizePosition — ground rule (the surf gate)
```c
PM_CheckWater();
if (velocity[2] > 180) { onground = -1; return; }             // int literal
tr = PlayerTrace(origin, origin - (0,0,2), hull);              // 2-unit down probe
onground = (tr.plane.normal[2] < 0.7f) ? -1 : tr.ent;          // CS literal 0.7f — steeper than ~45.57° can NEVER be ground
if (onground != -1) {
    waterjumptime = 0;
    if (waterlevel < 2 && !tr.startsolid && !tr.allsolid)      // note the water guard
        origin = tr.endpos;                                    // snap down ≤2u every ground frame
}
```

### PM_Accelerate (ground) / PM_AirAccelerate (air)
```c
// ground
currentspeed = DotProduct(velocity, wishdir);
addspeed = wishspeed - currentspeed;   if (addspeed <= 0) return;
accelspeed = accel * frametime * wishspeed * pmove->friction;  // accel = sv_accelerate
if (accelspeed > addspeed) accelspeed = addspeed;
velocity += accelspeed * wishdir;

// air — the surf primitive
wishspd = min(wishspeed, 30);                                   // THE 30-unit cap (int literal)
addspeed = wishspd - DotProduct(velocity, wishdir);  if (addspeed <= 0) return;
accelspeed = accel * wishspeed * frametime * pmove->friction;   // UNCAPPED wishspeed in the rate!
if (accelspeed > addspeed) accelspeed = addspeed;
velocity += accelspeed * wishdir;
```
Air speed is limited only *along the current wishdir* (30u projection); perpendicular velocity is untouched — that asymmetry is airstrafing. At `aa=100` the rate term saturates addspeed every frame.

### PM_AirMove / PM_WalkMove — wishdir construction
```c
forward[2] = 0; right[2] = 0; VectorNormalize(forward); VectorNormalize(right);   // PITCH IRRELEVANT in air/ground
wishvel[i] = forward[i]*fmove + right[i]*smove  (i = 0,1);  wishvel[2] = 0;
wishdir = wishvel; wishspeed = VectorNormalize(wishdir);
if (wishspeed > maxspeed) { scale wishvel; wishspeed = maxspeed; }               // 250 with knife
// AirMove: PM_AirAccelerate(...); velocity += basevelocity; PM_FlyMove();
// WalkMove: CS stamina drag FIRST (§ stamina); velocity[2]=0; PM_Accelerate; velocity[2]=0;
//   velocity += basevelocity; if (speed < 1.0) { velocity = 0; return; }
//   direct horizontal trace; if blocked → try flat slide AND step-up-slide-step-down (18u,
//   reject step if landing normal[2] < 0.7f), keep whichever travels farther in 2D —
//   BUT when the step path wins, velocity[2] is overwritten with the flat-slide result
//   (cs_pm_shared.cpp:1243-1247 "copy z value from slide move").
```

### PM_FlyMove — the collision slide (heart of surf)
```c
numbumps = 4; time_left = frametime; allFraction = 0; numplanes = 0;
original_velocity = primal_velocity = velocity;
for (bump = 0; bump < 4; bump++) {
    if (velocity == 0) break;
    end = origin + time_left * velocity;
    trace = PlayerTrace(origin, end, hull);
    allFraction += trace.fraction;
    if (trace.allsolid) { velocity = 0; return 4; }             // trapped
    if (trace.fraction > 0) { origin = trace.endpos; original_velocity = velocity; numplanes = 0; }
    if (trace.fraction == 1) break;
    time_left -= time_left * trace.fraction;
    if (numplanes >= 5 /*MAX_CLIP_PLANES*/) { velocity = 0; break; }
    planes[numplanes++] = trace.plane.normal;
    if (numplanes == 1 && movetype == MOVETYPE_WALK && (onground == -1 || pmove->friction != 1)) {
        // the airborne first-plane branch — THE surf path
        overbounce = (planes[0][2] > 0.7f) ? 1 : 1.0 + movevars->bounce * (1.0 - pmove->friction); // both = 1.0 stock
        PM_ClipVelocity(original_velocity, planes[0], velocity, overbounce);
        original_velocity = velocity;
    } else {
        for (i = 0; i < numplanes; i++) {                        // find a plane we exit along
            PM_ClipVelocity(original_velocity, planes[i], velocity, 1);
            for (j = 0; j < numplanes; j++) if (j != i && DotProduct(velocity, planes[j]) < 0) break;
            if (j == numplanes) break;
        }
        if (i == numplanes) {                                    // crease: slide along the intersection
            if (numplanes != 2) { velocity = 0; break; }
            dir = CrossProduct(planes[0], planes[1]);             // NOT normalized — |n1×n2| scales velocity
            velocity = dir * DotProduct(dir, velocity);
        }
        if (DotProduct(velocity, primal_velocity) <= 0) { velocity = 0; break; }   // anti-oscillation
    }
}
if (allFraction == 0) velocity = 0;    // all 4 bumps started blocked → full stop (ramp-seam head-on)
```

### PM_ClipVelocity
```c
backoff = DotProduct(in, normal) * overbounce;                  // CS: double intermediate
out[i] = in[i] - normal[i] * backoff;
if (out[i] > -0.1 && out[i] < 0.1) out[i] = 0;                  // STOP_EPSILON = 0.1, per component, EVERY clip
```
Overbounce 1.0 = pure tangential projection `v − n(v·n)`. Gravity's per-tick increment, clipped against a steep ramp, leaves its along-plane component → continuous down-slope thrust with zero friction. That *is* surfing.

### PM_Jump (CS vanilla)
```c
if (waterjumptime || waterlevel >= 2) { ... return; }           // water hooks
if (onground == -1) { oldbuttons |= IN_JUMP; return; }          // air: latch only
if (oldbuttons & IN_JUMP) return;                               // EDGE TRIGGER — must release between jumps
if (bInDuck && (flags & FL_DUCKING)) return;                    // CS-only guard
onground = -1;
PM_PreventMegaBunnyJumping();                                    // CS: cap 1.2f·maxspeed, rescale ×(cap/spd)·0.8
velocity[2] = Q_sqrt(2.0 * 800.0f * 45.0f);                     // DOUBLE sqrt → stored float 268.328155517578125
                                                                 // 800 HARDCODED — sv_gravity does not change jump-off speed
if (fuser2 > 0.0f) velocity[2] *= (100.0 - fuser2 * 0.001 * 19.0) * 0.01;   // stamina, double math
pmove->fuser2 = 1315.789429;                                     // stamina reset (ms); full penalty ratio = 0.75
PM_FixupGravityVelocity();
oldbuttons |= IN_JUMP;
```

### Stamina (fuser2) — CS only
- Set to `1315.789429` on every jump; decays `−= msec` per tick (`PM_ReduceTimers`), zero after ~1.316 s.
- Ratio `= (100 − fuser2·0.001·19) · 0.01` ≈ `1 − 0.00019·fuser2`, from ~0.75 (exactly 0.7500000085449219 at full debt, evaluating left-to-right in double as the code does — do NOT pre-fold the constants) up to 1.0.
- Applied to `velocity[2]` once at jump, AND to `velocity[0..1]` **every ground frame** at the top of `PM_WalkMove`.
- Never triggers during pure surfing (no jumps, never onground) — `enable_stamina` off loses zero fidelity on ramp-only tasks.

### PM_PreventMegaBunnyJumping — CS only fires at jump time
```c
maxscaledspeed = 1.2f * pmove->maxspeed;         // 300 with knife  (HL: 1.7f)
spd = Length(velocity);
if (spd > maxscaledspeed) velocity *= (maxscaledspeed / spd) * 0.8;   // → post-jump speed = 240·… exactly maxscaledspeed·0.8  (HL: ·0.65)
```
Never limits mid-air surf speed; punishes jumping at speed (relevant to jump-off-ramp finishes).

### PM_Friction (ground only — call site gated on onground, v_z already zeroed)
```c
speed = sqrt(v·v);  if (speed < 0.1f) return;
// edgefriction probe: start = origin + vel/speed·16 (xy), z = origin.z + player_mins[usehull].z; down 34
friction = movevars->friction * (probe_trace.fraction == 1.0 ? movevars->edgefriction : 1) * pmove->friction;
control = max(speed, movevars->stopspeed);
drop = friction * (control * frametime);          // CS association; HL: control*friction*frametime
newspeed = max(0, speed - drop) / speed;          // real_t (double)
newvel[0] = vel[0] * newspeed;                    // float × DOUBLE — one rounding
newvel[1] = vel[1] * (float)newspeed;             // newspeed rounded to float FIRST
newvel[2] = vel[2] * (float)newspeed;             //   (asymmetric per-component, faithful to the binary)
```

### PM_Duck / PM_UnDuck (compile-flagged; cut line #1 in [06](06-night-plan.md))
`flDuckTime = 1000` ms on press; ducked state (`FL_DUCKING`, hull 1) after 400 ms (`TIME_TO_DUCK 0.4`) — **instantly if airborne**. On-ground duck shifts origin z by −18 (`player_mins[1]−player_mins[0]`) + `PM_FixPlayerCrouchStuck` nudges up; in-air duck does NOT shift origin (feet rise 18u — the duck-jump). UnDuck reverses (+18 on ground, test standing hull first, stay ducked if blocked). Cmd moves ×`0.333` while gate passes (CS: while pressed/transitioning/ducked; HL: only fully ducked). View heights: CS standing 17, ducked 12.

### PM_CheckVelocity — the only hard cap
Per component: NaN scrub on velocity **and origin** (bit test `(bits & (255<<23)) == 255<<23` per HLSDK's `IS_NAN`/`nanmask`, not `isnan()`; origin is scrubbed but only velocity is clamped), then clamp velocity to ±`sv_maxvelocity`. Runs 4+ times per tick (both gravity halves, post-friction, post-move).

### PM_CheckStuck — port the SERVER path, not the client path
If hull position free → not stuck (reset per-player offset index). The full 54-entry do/while nudge loop is **client-prediction only** (`if (!pmove->server)` — cs_pm_shared.cpp:1736); the server path the env emulates is: throttle to one attempt per 0.05 s (source uses `Sys_FloatTime()` wall clock — env substitutes **sim time**, 5 ticks at msec 10, per-env state), take a **single** offset from the 54-slot table via a persistent cycling index, and commit the freed position to origin **only when the index ≥ 27** (second-half "big" offsets); a free first-half offset returns "not stuck" *without moving the player*. Table (`PM_CreateStuckTable`): ±0.125 single-axis (incl. 0) + 8 corner combos, then z {0,1,6} / x,y {−2,0,+2} singles and their 27 combos (53 filled entries, 54 slots cycled). Skip the stuck-in-player grid (no other players).

## 5. AngleVectors + usercmd (pm_math)

Degrees; `PITCH=0` (positive looks down), `YAW=1`, `ROLL=2`:
```c
forward = ( cp·cy,  cp·sy,  −sp );
right   = ( −sr·sp·cy + cr·sy,  −sr·sp·sy − cr·cy,  −sr·cp );   // simplified from source's −1· forms
up      = ( cr·sp·cy + sr·sy,   cr·sp·sy − sr·cy,   cr·cp );
```
With roll = 0 (env): `right = (sy, −cy, 0)` — note the sign convention. Precision quirk (CS only): in `cs_pm_math.cpp` the locals are float **except `angle` and `cy` which are real_t (double)** — so terms containing cos(yaw) round from double while e.g. `forward[1] = cp·sy` is pure float (HL uses all-float). Because ground/air wishdir zeroes and re-normalizes z, **pitch never scales wishspeed** — a pitched-down forward still yields |wishvel| = fmove. Env freezes pitch = roll = 0 and wraps accumulated yaw into [0, 360) each tick before building viewangles (client `anglemod` convention — the ported `>180 → −360` mapping assumes it).

`usercmd`: `byte msec`, float `forwardmove/sidemove/upmove` (client sends ±`cl_*speed`, default 400; ±2047 net cap), buttons bitmask (`IN_JUMP 2`, `IN_DUCK 4`). `PM_CheckParameters` rescales the 3-vector to `pmove->maxspeed` — so any client value ≥ 250 is equivalent; env emits ±400 like humans.

## 6. trigger_push / basevelocity (surf boosters — surf_ski_2 has 7)

`CTriggerPush::Touch` per frame inside a push field: `basevelocity = speed · movedir` (+= if `FL_BASEVELOCITY` already set); sets `FL_BASEVELOCITY`. `movedir` from `SetMovedir`: angles `(0,-1,0)`→up, `(0,-2,0)`→down, else forward of angles (`"angles" "-90 0 0"` = straight up, typical booster, speeds 1000–4000). `SF_TRIGGER_PUSH_ONCE (1)`: adds directly to velocity once, removes trigger.
In-move: horizontal basevelocity added before FlyMove, subtracted after; vertical consumed as acceleration in `AddCorrectGravity`. On leaving the field, `SV_CheckMovingGround` folds `(dt·0.5 + 1)·basevelocity` into real velocity.

## 7. Float32 parity rules for the port

1. Storage is `float` (`vec3_t`); **intermediates at marked sites are double** — ReGameDLL's `real_t` (double under `PLAY_GAMEDLL`) encodes the original x87 build's excess precision. Key real_t sites: ClipVelocity `backoff`; Accelerate/AirAccelerate `currentspeed`/`accelspeed`; Friction `newspeed/control/friction/drop` (`speed` is float! and the velocity rescale is per-component asymmetric — see its spec); WalkMove `spd/wishspeed`/stamina ratio; FlyMove `time_left·velocity[i]`; Jump ratio + `Q_sqrt`; both gravity halves (`ent_gravity` double, asymmetric association — §3); AngleVectors `angle`/`cy` only. **Caution on the math helpers:** `DotProduct`, `VectorNormalize`'s radicand, and the `Q_sqrt(real_t(...))` sums compute their products/sums **in float** and only widen to double at the return/cast boundary; only `Length` genuinely accumulates in double. Computing dot products in double "to be safe" *diverges* from the engine. Port rule: **copy each expression's operand types and order verbatim; widen exactly where the source widens, never earlier.**
2. Copy literal types exactly: CS uses `0.7f` (= 0.699999988…) where HL uses double `0.7`; stamina constants are double (`0.001`, `19.0`); `180` and `30` are ints.
3. Copy association: e.g. CS `drop += friction * (control * frametime)`.
4. `frametime = (float)(msec * 0.001)` — never `1.0/fps`.
5. No RNG touches physics (sounds only; stuck table is deterministic). No wall clock.
6. Bit-exactness vs. the original x87 binary is asymptotic, not absolute — the goal is: same binary → deterministic, and vs. ReHLDS server → sub-unit drift over a minute ([05](05-validation.md)).

## 8. Port checklist → `src/pm.c`

| Port (tonight) | Behind flag | Skip (hook comment only) |
|---|---|---|
| CheckParameters (clamp), ReduceTimers, AngleVectors, CategorizePosition, **CheckWater/InWater** (waterlevel via point contents, skip the CONTENTS_CURRENT block — the pipeline calls them unconditionally; when cut per [06] line 6, stub to `waterlevel=0; watertype=EMPTY; return false`), AddCorrectGravity/FixupGravityVelocity, Friction (+edge probe), Accelerate, AirAccelerate, AirMove, WalkMove (+step), FlyMove, ClipVelocity, CheckVelocity, Jump (+edge trigger, bhop cap), CheckStuck (server path), basevelocity fold + trigger_push | stamina (2 lines, default on), duck/unduck (default off tonight), bhop-cap toggle | ladders, water swim/waterjump (waterlevel≥2 = episode fail), spectator, noclip, toss, punchangle/roll, sounds, longjump |

Sources: ReGameDLL_CS (`pm_shared.cpp/.h`, `pm_math.cpp`, `pm_defs.h`, `triggers.cpp`, `player.cpp`, `weapons.h`, `game.cpp`), HLSDK (`pm_shared.c`, `pm_math.c`), ReHLDS (`sv_user.cpp`, `sv_phys.cpp`, `sv_main.cpp`, `pmove.cpp`, `host.cpp`), jwchong.com/hl (msec/slowdown math, stamina), UltimateSurf cfg + AlliedModders (surf server conventions). Full URL list in `docs/research/` outputs.
