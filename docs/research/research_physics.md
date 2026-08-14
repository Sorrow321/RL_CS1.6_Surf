
## tick_pipeline_markdown
# GoldSrc / CS 1.6 player-movement per-tick pipeline (exact call order)

## 0. Engine wrapper (per usercmd) — ReHLDS `SV_RunCmd` (rehlds/engine/sv_user.cpp)

For each received `usercmd_t` (one "tick" of movement — there is no fixed tickrate; the sim step is the command's own duration):

1. If `cmd.msec > 50`: the command is **chopped in half** and run as two recursive `SV_RunCmd` calls, each with `cmd.msec = (byte)(ucmd->msec / 2.0)` (second half gets `impulse = 0`). So a single physics step never exceeds 50 ms.
2. `frametime = float(ucmd->msec * 0.001)` (float multiply; `msec` is a **byte**, 0–255).
3. `SV_CheckMovingGround(sv_player, frametime)`:
   - If `FL_ONGROUND` and groundentity has `FL_CONVEYOR`: `basevelocity += speed*movedir` (or `= speed*movedir` if `FL_BASEVELOCITY` not set), sets `FL_BASEVELOCITY`.
   - If `FL_BASEVELOCITY` **not** set (i.e. no trigger_push/conveyor touched us since last frame): `velocity = velocity + (frametime * 0.5f + 1.0f) * basevelocity; basevelocity = {0,0,0}` — this is how a trigger_push impulse gets folded into real velocity the frame you leave the push volume.
   - Always clears `FL_BASEVELOCITY` (trigger Touch during the game frame re-sets it).
4. PreThink/Think run (game dll; trigger_push `Touch` fires from the physics linking, setting `pev->basevelocity = pev->speed * pev->movedir` and `FL_BASEVELOCITY` — ReGameDLL_CS triggers.cpp:1712-1719; the `SF_TRIGGER_PUSH_ONCE` variant instead adds directly to `velocity` and clears `FL_ONGROUND` if pushed up).
5. `pmove` struct is filled from entvars: notably `pmove->usehull = (flags & FL_DUCKING) == FL_DUCKING`, `pmove->maxspeed = sv_maxspeed.value`, `pmove->clientmaxspeed = pev->maxspeed` (CS sets pev->maxspeed per weapon, e.g. 250), `pmove->gravity = pev->gravity`, `pmove->friction = pev->friction`, `pmove->waterjumptime = pev->teleport_time`, `pmove->oldbuttons = pev->oldbuttons`, `pmove->fuser2 = pev->fuser2` (CS stamina), `pmove->basevelocity = pev->basevelocity`, `pmove->frametime = frametime`, `pmove->cmd = cmd`.
6. `gEntityInterface.pfnPM_Move(pmove, TRUE)` → game dll `PM_Move`.
7. Writeback: origin, velocity, basevelocity, flags (FL_ONGROUND set from `pmove->onground != -1`, `groundentity` from `physents[onground].info`), `pev->flDuckTime = (int)pmove->flDuckTime` (**int truncation!**), and — important — `sv_player->v.oldbuttons = pmove->cmd.buttons` (the engine overwrites oldbuttons with THIS command's buttons, ignoring pm-internal `oldbuttons |= IN_JUMP` edits between commands).

## 1. `PM_Move(ppmove, server)` (pm_shared)

```
pmove = ppmove;
PM_PlayerMove(server != 0);
if (pmove->onground != -1)  pmove->flags |= FL_ONGROUND;  else pmove->flags &= ~FL_ONGROUND;
if (!pmove->multiplayer && pmove->movetype == MOVETYPE_WALK)  pmove->friction = 1.0f;   // SP only
```

## 2. `PM_PlayerMove(server)` — exact order

```
pmove->server = server;
PM_CheckParamters();                       // CS: PM_CheckParameters()   [clamp cmd moves to maxspeed, build pmove->angles]
pmove->numtouch = 0;
pmove->frametime = pmove->cmd.msec * 0.001;   // float; recomputed here from the byte msec
PM_ReduceTimers();                          // step-sound / duck / swim timers; CS ALSO decays fuser2 (stamina) here
AngleVectors(pmove->angles, pmove->forward, pmove->right, pmove->up);

if (spectator || iuser1 > 0)                // CS adds: && PM_ShouldDoSpectMode() i.e. (iuser3 <= 0 || deadflag == DEAD_DEAD)
    { PM_SpectatorMove(); PM_CatagorizePosition(); return; }

if (movetype != MOVETYPE_NOCLIP && movetype != MOVETYPE_NONE)
{
    HL:  if (PM_CheckStuck()) { PM_Duck(); if (PM_CheckStuck()) return; }   // duck-retry
    CS:  if (PM_CheckStuck()) return;                                       // NO duck retry
}

PM_CatagorizePosition();                    // sets waterlevel + onground (0.7 / 180 rules, 2-unit down trace + snap)
pmove->oldwaterlevel = pmove->waterlevel;
if (pmove->onground == -1) pmove->flFallVelocity = -pmove->velocity[2];

g_onladder = 0;
if (!dead && !(flags & FL_ONTRAIN)) { pLadder = PM_Ladder(); if (pLadder) g_onladder = 1; }

HL:  PM_UpdateStepSound();  PM_Duck();      // HL order
CS:  PM_Duck();  PM_UpdateStepSound();      // CS order (swapped)

if (!dead && !(flags & FL_ONTRAIN))
{
    if (pLadder) PM_LadderMove(pLadder);                       // <-- LADDER HOOK (sets movetype = MOVETYPE_FLY, gravity = 0)
    else if (movetype != MOVETYPE_WALK && movetype != MOVETYPE_NOCLIP) movetype = MOVETYPE_WALK;
}

switch (movetype)
{
case MOVETYPE_NONE: break;
case MOVETYPE_NOCLIP: PM_NoClip(); break;
case MOVETYPE_TOSS: case MOVETYPE_BOUNCE: PM_Physics_Toss(); break;   // dead bodies
case MOVETYPE_FLY:                                                     // ladder movement
    PM_CheckWater();
    if (cmd.buttons & IN_JUMP) { if (!pLadder) PM_Jump(); } else oldbuttons &= ~IN_JUMP;
    velocity += basevelocity;  PM_FlyMove();  velocity -= basevelocity;
    break;
case MOVETYPE_WALK:                                                    // <<< THE MAIN PATH
    if (!PM_InWater())              PM_AddCorrectGravity();            // half-gravity + basevelocity[2] consumption
    if (waterjumptime)              { PM_WaterJump(); PM_FlyMove(); PM_CheckWater(); return; }   // WATERJUMP HOOK
    if (waterlevel >= 2)            { ... PM_WaterMove() ... }         // SWIMMING HOOK (skipped in scope)
    else
    {
        // -------- jump BEFORE friction: this ordering is the bunnyhop mechanism --------
        if (cmd.buttons & IN_JUMP)  { if (!pLadder) PM_Jump(); }       // PM_Jump sets onground = -1 on success
        else                        oldbuttons &= ~IN_JUMP;

        if (onground != -1)         { velocity[2] = 0;  PM_Friction(); }   // HL: 0.0 literal; CS: 0
        PM_CheckVelocity();                                            // NaN + per-axis ±sv_maxvelocity clamp

        if (onground != -1)         PM_WalkMove();                     // ground move (accelerate + step logic)
        else                        PM_AirMove();                      // air move  (airaccelerate + PM_FlyMove)

        PM_CatagorizePosition();                                       // re-evaluate ground after the move
        velocity -= basevelocity;                                      // pull conveyor/push velocity back out
        PM_CheckVelocity();
        if (!PM_InWater())          PM_FixupGravityVelocity();         // second half-gravity
        if (onground != -1)         velocity[2] = 0;                   // final ground clamp
        PM_CheckFalling();
    }
    PM_PlayWaterSounds();
    break;
}
```

### The split-gravity (leapfrog) scheme
`PM_AddCorrectGravity` (before the move): `velocity[2] -= ent_gravity * movevars->gravity * 0.5 * frametime; velocity[2] += basevelocity[2] * frametime; basevelocity[2] = 0;` then `PM_CheckVelocity()`.
`PM_FixupGravityVelocity` (after the move): `velocity[2] -= ent_gravity * movevars->gravity * frametime * 0.5;` then `PM_CheckVelocity()`. (`ent_gravity = pmove->gravity` if nonzero else `1.0`; both no-op while `waterjumptime`.)
Position is therefore integrated with velocity sampled at the mid-point of the frame (semi-implicit / leapfrog). On the frame you jump, `PM_Jump` OVERWRITES `velocity[2]` (discarding the already-applied first half-gravity) and then calls `PM_FixupGravityVelocity()` itself ("Decay it for simulation") so the move runs with `jumpvel - 0.5*g*ft`; the pipeline-end fixup then brings the frame total to `jumpvel - g*ft`. Net: exactly one full gravity per frame, and jump apex math matches `sqrt(2*800*45)` for 45 units at gravity 800.

## 3. Why steep planes (normal.z < 0.7) never set onground — and why that + PM_ClipVelocity = surfing

1. **Ground detection is a thresholded trace, not a contact test.** `PM_CatagorizePosition` traces the player hull from `origin` to `origin - (0,0,2)`. If the hit plane's `normal[2] < 0.7` (HL literal `0.7`, CS literal `0.7f`) it sets `onground = -1` — "too steep". 0.7 ≈ cos(45.57°): any brush face steeper than ~45.57° from horizontal can NEVER be ground, no matter how firmly you rest on it. Also `velocity[2] > 180` forces `onground = -1` unconditionally (fresh jumps: 268.33 > 180).
2. **Everything ground-specific is gated on `onground != -1`.** On a surf ramp you therefore permanently skip: `velocity[2] = 0` (both clamp points), `PM_Friction()` (never called → zero speed bleed), `PM_WalkMove` (and its maxspeed-capped `PM_Accelerate`), and the snap-down `origin = tr.endpos`. Instead `PM_AirMove` runs every tick: `PM_AirAccelerate` (wishspeed capped at 30 for the add-speed test → you can only steer, not pump speed directly) followed by `PM_FlyMove`.
3. **PM_FlyMove + PM_ClipVelocity turn gravity into along-ramp thrust.** Each tick gravity adds `-g*dt` to `velocity[2]`; the attempted move immediately hits the ramp plane; `PM_ClipVelocity(v, n, out, 1.0)` computes `backoff = DotProduct(v, n) * 1.0; out[i] = v[i] - n[i]*backoff` — i.e. it removes only the into-plane component and keeps the full tangential component. The tangential projection of the per-tick gravity increment points down-slope, so the player continuously accelerates along the ramp: `a_tangent = g * sin(slope)` in the plane, with NO friction because `onground == -1`. Riding down/along a steep ramp thus gains speed every tick; the only caps are per-axis `sv_maxvelocity` (2000) in `PM_CheckVelocity`.
4. The `overbounce` path in `PM_FlyMove`'s first-plane branch (`numplanes == 1 && movetype == MOVETYPE_WALK && (onground == -1 || friction != 1)`): planes with `normal[2] > 0.7` are clipped with overbounce `1`, others with `1.0 + movevars->bounce * (1 - pmove->friction)` — with defaults `sv_bounce 1`, player `friction 1` this is also exactly `1.0`, i.e. a pure slide, no reflection.
5. `STOP_EPSILON 0.1` zeroes any post-clip component in (-0.1, 0.1), which is why you can come to a perfect stop wedged in a ramp trough.
6. Because `PM_CatagorizePosition` runs again AFTER the move, landing on a shallow (walkable) surface at the ramp bottom instantly re-enables friction and the `velocity[2] = 0` clamps — the "landing" speed loss surfers avoid by jumping off ramp ends (`velocity[2] > 180` keeps them airborne after boosts).


## functions

### item 0

#### name
PM_CheckParamters (HL) / PM_CheckParameters (CS)


#### spec_markdown
Clamps the wish-move vector and derives view angles. Exact algorithm (CS spelling; HL identical unless noted):
```c
spd = sqrt(fwd*fwd + side*side + up*up);           // HL order: fwd,side,up; CS order: side,fwd,up (identical result)
                                                    // CS: Q_sqrt(real_t(...)) i.e. double sqrt of double sum
maxspeed = pmove->clientmaxspeed;                   // per-player (CS: per-weapon speed, e.g. 250)
if (maxspeed != 0.0)  pmove->maxspeed = min(maxspeed, pmove->maxspeed);   // pmove->maxspeed arrives as sv_maxspeed

// HL master ONLY (#if !defined(_TFC)) — NOT in CS:
if ((pmove->onground != -1) && (pmove->cmd.buttons & IN_USE))  pmove->maxspeed *= 1.0f / 3.0f;

if (spd != 0.0 && spd > pmove->maxspeed) {
    fRatio = pmove->maxspeed / spd;                 // CS: real_t fRatio
    cmd.forwardmove *= fRatio; cmd.sidemove *= fRatio; cmd.upmove *= fRatio;
}
if ((flags & (FL_FROZEN|FL_ONTRAIN)) || dead)  cmd.forwardmove = cmd.sidemove = cmd.upmove = 0;
PM_DropPunchAngle(punchangle);                      // len = VectorNormalize(pa); len -= (10.0 + len*0.5)*frametime; len = max(len, 0); scale back
if (!dead) {
    v_angle = cmd.viewangles + punchangle;
    angles[ROLL]  = PM_CalcRoll(v_angle, velocity, movevars->rollangle, movevars->rollspeed) * 4;
    angles[PITCH] = v_angle[PITCH];
    angles[YAW]   = v_angle[YAW];
} else angles = oldangles;
if (dead) view_ofs[2] = PM_DEAD_VIEWHEIGHT;         // -8;  CS REGAMEDLL_FIXES also UnDucks first
if (angles[YAW] > 180.0f) angles[YAW] -= 360.0f;
```
Note the clamp compares the 3-D length of (forwardmove, sidemove, upmove) against maxspeed and rescales proportionally — this is the usercmd clamp; forwardmove/sidemove are floats sent by the client (default client cl_forwardspeed etc. 400, ±2047 hard net limit).


#### name_note



### item 1

#### name
PM_ReduceTimers


#### spec_markdown
Millisecond timers decremented by `pmove->cmd.msec` (byte), floored at 0:
```c
flTimeStepSound -= cmd.msec (if > 0, floor 0)
flDuckTime      -= cmd.msec (if > 0, floor 0)
flSwimTime      -= cmd.msec (if > 0, floor 0)
// CS ONLY — stamina decay:
if (pmove->fuser2 > 0.0) { pmove->fuser2 -= pmove->cmd.msec; if (pmove->fuser2 < 0.0) pmove->fuser2 = 0; }
```
fuser2 counts down in milliseconds from 1315.789429; it reaches 0 after ~1.316 s.


### item 2

#### name
AngleVectors (+ wishdir construction)


#### spec_markdown
pm_math.c / pm_math.cpp. Angle order: `PITCH=0` (up/down, positive = looking DOWN gives forward[2] = -sp), `YAW=1`, `ROLL=2`, degrees.
```c
angle = angles[YAW]   * (M_PI*2 / 360); sy = sin(angle); cy = cos(angle);
angle = angles[PITCH] * (M_PI*2 / 360); sp = sin(angle); cp = cos(angle);
angle = angles[ROLL]  * (M_PI*2 / 360); sr = sin(angle); cr = cos(angle);
forward = ( cp*cy,  cp*sy,  -sp );
right   = ( -1*sr*sp*cy + -1*cr*-sy,  -1*sr*sp*sy + -1*cr*cy,  -1*sr*cp );
up      = ( cr*sp*cy + -sr*-sy,  cr*sp*sy + -sr*cy,  cr*cp );
```
M_PI = 3.14159265358979323846. HL: all-float locals (double promotions via sin/cos). CS: `real_t cy; real_t angle;` — angle and cy are double under PLAY_GAMEDLL, sr/sp/sy/cr/cp are float (exactly as decompiled).
**Wishdir construction (WalkMove/AirMove):** `forward[2] = 0; right[2] = 0; VectorNormalize(forward); VectorNormalize(right);` then `wishvel[i] = forward[i]*fmove + right[i]*smove` for i=0,1; `wishvel[2] = 0; wishdir = wishvel; wishspeed = VectorNormalize(wishdir);` — i.e. view pitch does NOT affect ground/air wishdir (2-D re-normalized).


### item 3

#### name
PM_CatagorizePosition (HL) / PM_CategorizePosition (CS)


#### spec_markdown
```c
PM_CheckWater();                              // waterlevel/watertype; water-current adds to basevelocity (50.0*waterlevel)
point = { origin[0], origin[1], origin[2] - 2 };   // 2-unit down probe
if (velocity[2] > 180)                         // int literal 180: 'shooting up really fast'
    onground = -1;                             // CS: early return here; HL: else-block (same effect)
else {
    tr = PM_PlayerTrace(origin, point, PM_NORMAL, -1);
    if (tr.plane.normal[2] < 0.7)              // HL: 0.7 (double); CS: 0.7f  → too steep, NOT ground
        onground = -1;
    else
        onground = tr.ent;                     // physent index (0 = world)
    if (onground != -1) {
        waterjumptime = 0;
        if (waterlevel < 2 && !tr.startsolid && !tr.allsolid)
            origin = tr.endpos;                // snap down (up to 2 units) every ground frame
    }
    if (tr.ent > 0) PM_AddToTouched(tr, velocity);
}
```
Called: after unstuck, after WalkMove/AirMove, from PM_Duck/PM_UnDuck after origin shifts, and after spectator move.


### item 4

#### name
PM_CheckStuck


#### spec_markdown
```c
hitent = PM_TestPlayerPosition(origin, &traceresult);
if (hitent == -1) { PM_ResetStuckOffsets(player_index, server); return 0; }   // not stuck
base = origin;
// nudge phase — HL: if (!server || !multiplayer);  CS: if (!server)  [client prediction only]
//   if (hitent == 0 || physents[hitent].model != NULL)  // world or bsp model
//       loop nReps 0..53: offset = rgv3tStuckTable[idx++ % 54]; test = base + offset;
//       if TestPlayerPosition(test) == -1 { reset offsets; origin = test; return 0; }
idx = server ? 0 : 1;
fTime = Sys_FloatTime();
if (rgStuckCheckTime[player_index][idx] >= fTime - PM_CHECKSTUCK_MINTIME) return 1;   // 0.05 s throttle
rgStuckCheckTime[player_index][idx] = fTime;
PM_StuckTouch(hitent, &traceresult);
i = PM_GetRandomStuckOffsets(...); test = base + offset;
if (TestPlayerPosition(test) == -1) { reset offsets; if (i >= 27 /*CS: ARRAYSIZE/2*/) origin = test; return 0; }
// stuck-in-player escape: if (cmd.buttons & (IN_JUMP|IN_DUCK|IN_ATTACK)) && physents[hitent].player:
//   grid search x,y in {-8,0,8} (xystep 8.0), z in {0,18,36,54,72} (zstep 18.0, zminmax 4*18) — first free spot wins
return 1;
```
Stuck table (PM_CreateStuckTable, 54 entries): ±0.125 single-axis (x,y,z, 3 entries each incl. 0), 8 corner combos of ±0.125, then 'big moves': z in {0.0f, 1.0f, 6.0f}, x/y in {-2.0f, 0, 2.0f} single-axis and the 27 xyz combos.


### item 5

#### name
PM_AddCorrectGravity / PM_FixupGravityVelocity / PM_AddGravity


#### spec_markdown
```c
void PM_AddCorrectGravity() {
    if (waterjumptime) return;
    ent_gravity = pmove->gravity ? pmove->gravity : 1.0;       // per-player gravity multiplier
    velocity[2] -= ent_gravity * movevars->gravity * 0.5 * frametime;   // CS literal: 0.5f
    velocity[2] += basevelocity[2] * frametime;                 // vertical push acts as acceleration
    basevelocity[2] = 0;
    PM_CheckVelocity();
}
void PM_FixupGravityVelocity() {
    if (waterjumptime) return;
    ent_gravity = pmove->gravity ? pmove->gravity : 1.0;
    velocity[2] -= ent_gravity * movevars->gravity * frametime * 0.5;   // HL order: g*ft*0.5
    // CS order:  velocity[2] -= (movevars->gravity * frametime * ent_gravity * 0.5);
    PM_CheckVelocity();
}
void PM_AddGravity() {   // used only by PM_Physics_Toss (dead bodies)
    velocity[2] -= ent_gravity * movevars->gravity * frametime;
    velocity[2] += basevelocity[2] * frametime; basevelocity[2] = 0; PM_CheckVelocity();
}
```


### item 6

#### name
PM_Friction


#### spec_markdown
Ground friction incl. edgefriction look-ahead. Never called in air (call site is gated on `onground != -1`, with `velocity[2]` already zeroed).
```c
if (waterjumptime) return;
vel = pmove->velocity;
speed = sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);   // CS: Q_sqrt(real_t(...)), result stored in float speed
if (speed < 0.1f) return;                                      // friction dead-zone
drop = 0;
if (pmove->onground != -1) {
    // EDGEFRICTION LOOK-AHEAD: probe 16 units ahead along velocity dir, from bottom of hull, 34 units down
    start[0] = stop[0] = origin[0] + vel[0]/speed*16;
    start[1] = stop[1] = origin[1] + vel[1]/speed*16;
    start[2] = origin[2] + player_mins[usehull][2];
    stop[2]  = start[2] - 34;
    trace = PM_PlayerTrace(start, stop, PM_NORMAL, -1);
    if (trace.fraction == 1.0)  friction = movevars->friction * movevars->edgefriction;   // near a ledge: friction*2 (default)
    else                        friction = movevars->friction;
    friction *= pmove->friction;               // player friction modifier (func_friction), normally 1.0
    control = (speed < movevars->stopspeed) ? movevars->stopspeed : speed;
    HL: drop += control * friction * pmove->frametime;
    CS: drop += friction * (control * pmove->frametime);       // different FP association!
}
newspeed = speed - drop;  if (newspeed < 0) newspeed = 0;
newspeed /= speed;
newvel[0] = vel[0]*newspeed; newvel[1] = vel[1]*newspeed; newvel[2] = vel[2]*newspeed;
// CS decompile detail: newvel[0] = vel[0] * newspeed (double), newvel[1] = vel[1] * float(newspeed), newvel[2] = vel[2] * float(newspeed)
velocity = newvel;
```
CS types: `float speed; real_t newspeed, control, friction, drop;`


### item 7

#### name
PM_Accelerate


#### spec_markdown
Ground acceleration (also water via separate code).
```c
void PM_Accelerate(vec3_t wishdir, float wishspeed /*CS: real_t*/, float accel) {
    if (pmove->dead) return;
    if (pmove->waterjumptime) return;
    currentspeed = DotProduct(velocity, wishdir);        // CS: real_t
    addspeed = wishspeed - currentspeed;                 // float in both
    if (addspeed <= 0) return;                           // projection cap — the Quake steering trick
    accelspeed = accel * pmove->frametime * wishspeed * pmove->friction;   // CS: real_t
    if (accelspeed > addspeed) accelspeed = addspeed;
    for (i = 0; i < 3; i++) velocity[i] += accelspeed * wishdir[i];
}
```
With `accel = movevars->accelerate`. Per-frame ground speed-up is `accelerate * wishspeed * frametime` capped so the velocity projection onto wishdir never exceeds wishspeed.


### item 8

#### name
PM_AirAccelerate


#### spec_markdown
```c
void PM_AirAccelerate(vec3_t wishdir, float wishspeed, float accel) {
    float wishspd = wishspeed;
    if (pmove->dead || pmove->waterjumptime) return;     // HL: two separate ifs
    if (wishspd > 30) wishspd = 30;                      // THE 30-UNIT AIR WISHSPEED CAP (int literal 30)
    currentspeed = DotProduct(velocity, wishdir);        // CS: real_t
    addspeed = wishspd - currentspeed;                   // capped wishspd used for the add test...
    if (addspeed <= 0) return;
    accelspeed = accel * wishspeed * frametime * friction;   // ...but UNCAPPED wishspeed used for the rate; CS: real_t
    if (accelspeed > addspeed) accelspeed = addspeed;
    for (i = 0; i < 3; i++) velocity[i] += accelspeed * wishdir[i];
}
```
With `accel = movevars->airaccelerate`. Consequence: max +30 u/s in wishdir per contact frame, gained at rate `airaccelerate * wishspeed * frametime`; strafing sideways keeps `currentspeed` (projection) small → free speed gain by turning — the basis of airstrafing/surf steering.


### item 9

#### name
PM_AirMove


#### spec_markdown
```c
fmove = cmd.forwardmove; smove = cmd.sidemove;
forward[2] = 0; right[2] = 0; VectorNormalize(forward); VectorNormalize(right);
for (i = 0; i < 2; i++) wishvel[i] = forward[i]*fmove + right[i]*smove;
wishvel[2] = 0;
wishdir = wishvel; wishspeed = VectorNormalize(wishdir);
if (wishspeed > pmove->maxspeed) { VectorScale(wishvel, pmove->maxspeed/wishspeed, wishvel); wishspeed = pmove->maxspeed; }
PM_AirAccelerate(wishdir, wishspeed, movevars->airaccelerate);
VectorAdd(velocity, basevelocity, velocity);
PM_FlyMove();
```
No friction, no maxspeed clamp on actual velocity (only on wishspeed). CS: identical logic (wrapped in PM_AirMove_internal via API hook).


### item 10

#### name
PM_WalkMove


#### spec_markdown
```c
// ---- CS ONLY: stamina drag (runs first) ----
if (pmove->fuser2 > 0.0) {
    real_t flRatio = (100 - pmove->fuser2 * 0.001 * 19) * 0.01;   // fuser2 in ms; at 1315.789429 → ~0.75
    velocity[0] *= flRatio;  velocity[1] *= flRatio;              // horizontal only, EVERY ground frame while fuser2 > 0
}
fmove = cmd.forwardmove; smove = cmd.sidemove;
forward[2] = 0; right[2] = 0; VectorNormalize(forward); VectorNormalize(right);
for (i = 0; i < 2; i++) wishvel[i] = forward[i]*fmove + right[i]*smove;
wishvel[2] = 0;
wishdir = wishvel; wishspeed = VectorNormalize(wishdir);           // CS: real_t wishspeed
if (wishspeed > maxspeed) { VectorScale(wishvel, maxspeed/wishspeed, wishvel); wishspeed = maxspeed; }
velocity[2] = 0;
PM_Accelerate(wishdir, wishspeed, movevars->accelerate);
velocity[2] = 0;
VectorAdd(velocity, basevelocity, velocity);
spd = Length(velocity);
if (spd < 1.0f) { VectorClear(velocity); return; }                 // CS: spd < 1.0 (real_t spd)
oldonground = pmove->onground;
// direct horizontal attempt (z held constant):
dest = { origin[0] + velocity[0]*frametime, origin[1] + velocity[1]*frametime, origin[2] };
trace = PM_PlayerTrace(origin, dest, PM_NORMAL, -1);
if (trace.fraction == 1) { origin = trace.endpos; return; }        // clean move, done (NO step logic, NO basevel subtract here — that happens in PlayerMove)
if (oldonground == -1 && waterlevel == 0) return;                  // don't step while airborne
if (waterjumptime) return;
// try both: (a) slide at current height, (b) step up, slide, step down — keep whichever goes farther
original = origin; originalvel = velocity;
clip = PM_FlyMove();                                                // (a) ground slide
down = origin; downvel = velocity;
origin = original; velocity = originalvel;
dest = origin; dest[2] += movevars->stepsize;                       // (b) up 18
trace = PM_PlayerTrace(origin, dest, ...); if (!trace.startsolid && !trace.allsolid) origin = trace.endpos;
clip = PM_FlyMove();
dest = origin; dest[2] -= movevars->stepsize;                       // press back down 18
trace = PM_PlayerTrace(origin, dest, ...);
if (trace.plane.normal[2] < 0.7) goto usedown;                      // stepping onto too-steep surface → use slide result (HL 0.7, CS 0.7f)
if (!trace.startsolid && !trace.allsolid) origin = trace.endpos;
pmove->up = origin;
downdist = (down[0]-original[0])^2 + (down[1]-original[1])^2;       // 2-D distances
updist   = (up[0]-original[0])^2   + (up[1]-original[1])^2;
if (downdist > updist) { usedown: origin = down; velocity = downvel; }
else velocity[2] = downvel[2];                                      // stepped: keep slide-move z velocity
```
HL-master extra (not in CS, not in classic goldsrc pm): none inside WalkMove — the old SDK's `IN_USE → VectorScale(velocity, 0.3, velocity)` was moved to PM_CheckParamters as `maxspeed *= 1.0f/3.0f`. CS has neither (no +use slowdown).
CS REGAMEDLL_ADD extra (non-vanilla, off by default): `IN_RUN`/fuser3 speed-up; stamina_restore_rate fps-normalized stamina.


### item 11

#### name
PM_FlyMove


#### spec_markdown
The multi-plane clipping slide (used for ground slide inside WalkMove, air movement, ladder). Exact algorithm:
```c
numbumps = 4;
blocked = 0; numplanes = 0;
original_velocity = primal_velocity = velocity;
allFraction = 0;
time_left = pmove->frametime;
for (bumpcount = 0; bumpcount < numbumps; bumpcount++) {
    if (!velocity[0] && !velocity[1] && !velocity[2]) break;
    for (i = 0; i < 3; i++) end[i] = origin[i] + time_left * velocity[i];   // CS: real_t flScale = time_left*velocity[i]; end[i] = origin[i] + flScale;
    trace = PM_PlayerTrace(origin, end, PM_NORMAL, -1);
    allFraction += trace.fraction;
    if (trace.allsolid) { velocity = {0,0,0}; return 4; }                    // trapped in solid
    if (trace.fraction > 0) { origin = trace.endpos; original_velocity = velocity; numplanes = 0; }
    if (trace.fraction == 1) break;                                          // moved full distance
    PM_AddToTouched(trace, velocity);
    if (trace.plane.normal[2] > 0.7)  blocked |= 1;   // floor  (HL 0.7 / CS 0.7f)
    if (!trace.plane.normal[2])       blocked |= 2;   // wall/step
    time_left -= time_left * trace.fraction;
    if (numplanes >= MAX_CLIP_PLANES) { velocity = {0,0,0}; break; }         // MAX_CLIP_PLANES == 5
    planes[numplanes++] = trace.plane.normal;
    if (numplanes == 1 && movetype == MOVETYPE_WALK && (onground == -1 || pmove->friction != 1)) {
        // reflect/slide branch (first plane only)
        for (i = 0; i < numplanes; i++)
            if (planes[i][2] > 0.7) { PM_ClipVelocity(original_velocity, planes[i], new_velocity, 1); original_velocity = new_velocity; }
            else PM_ClipVelocity(original_velocity, planes[i], new_velocity, 1.0 + movevars->bounce * (1 - pmove->friction));   // CS: (1.0 - pmove->friction)
        velocity = new_velocity; original_velocity = new_velocity;
    } else {
        for (i = 0; i < numplanes; i++) {
            PM_ClipVelocity(original_velocity, planes[i], velocity, 1);
            for (j = 0; j < numplanes; j++) if (j != i) { if (DotProduct(velocity, planes[j]) < 0) break; }
            if (j == numplanes) break;    // doesn't re-enter any other plane — good
        }
        if (i == numplanes) {             // couldn't find a good plane → crease
            if (numplanes != 2) { velocity = {0,0,0}; break; }
            CrossProduct(planes[0], planes[1], dir);
            d = DotProduct(dir, velocity);
            VectorScale(dir, d, velocity);           // NOTE: dir NOT normalized — velocity scales by |n1 x n2|
        }
        if (DotProduct(velocity, primal_velocity) <= 0) { velocity = {0,0,0}; break; }   // anti-oscillation
    }
}
if (allFraction == 0) velocity = {0,0,0};             // never moved at all this frame → don't stick
return blocked;
```
CS also contains `PM_FlyMove_New()` (Q3-style, `flymove_method` cvar, REGAMEDLL_ADD, default off) — not vanilla.


### item 12

#### name
PM_ClipVelocity


#### spec_markdown
```c
int PM_ClipVelocity(vec3_t in, vec3_t normal, vec3_t out, float overbounce) {
    angle = normal[2];                       // CS: real_t angle, real_t backoff; HL: float
    blocked = 0x00;
    if (angle > 0) blocked |= 0x01;          // floor
    if (!angle)    blocked |= 0x02;          // wall/step
    backoff = DotProduct(in, normal) * overbounce;
    for (i = 0; i < 3; i++) {
        HL: change = normal[i]*backoff;  out[i] = in[i] - change;
        CS: change = in[i] - normal[i] * backoff;  out[i] = change;   // backoff is double → different rounding
        if (out[i] > -STOP_EPSILON && out[i] < STOP_EPSILON) out[i] = 0;   // STOP_EPSILON == 0.1
    }
    return blocked;
}
```
With overbounce 1.0 this is a pure tangential projection `v - n(v.n)` — the surf primitive. Overbounce > 1 reflects (used only via bounce*(1-friction) path and Physics_Toss).


### item 13

#### name
PM_CheckVelocity


#### spec_markdown
Per-component NaN scrub and speed clamp — the ONLY hard velocity cap in the game:
```c
for (i = 0; i < 3; i++) {
    if (IS_NAN(velocity[i])) { Con_Printf(...); velocity[i] = 0; }
    if (IS_NAN(origin[i]))   { Con_Printf(...); origin[i] = 0; }
    if      (velocity[i] >  movevars->maxvelocity) velocity[i] =  movevars->maxvelocity;
    else if (velocity[i] < -movevars->maxvelocity) velocity[i] = -movevars->maxvelocity;
}
```
`IS_NAN(x)` is `(((*(int *)&x) & nanmask) == nanmask)`, `nanmask = 255 << 23`. Called from: PM_AddCorrectGravity, PM_FixupGravityVelocity, PM_AddGravity, and twice in the WALK branch of PM_PlayerMove (after friction, after the move+basevel subtract). Clamp is PER-AXIS (speed can reach maxvelocity*sqrt(3) diagonally).


### item 14

#### name
PM_Jump


#### spec_markdown
**HL version:**
```c
if (dead) { oldbuttons |= IN_JUMP; return; }
tfc = physinfo"tfc" == 1;  if (tfc && deadflag == DEAD_DISCARDBODY+1) return;
if (waterjumptime) { waterjumptime -= cmd.msec; if (< 0) = 0; return; }
if (waterlevel >= 2) { onground = -1; velocity[2] = 100 (WATER) / 80 (SLIME) / 50 (LAVA); swim sound; return; }
if (onground == -1) { oldbuttons |= IN_JUMP; return; }     // in air: no effect, latch button
if (oldbuttons & IN_JUMP) return;                          // EDGE TRIGGER: don't pogo — must release between jumps
onground = -1;
PM_PreventMegaBunnyJumping();
PM_PlayStepSound(..., 1.0);                                // (tfc: plyrjmp8.wav)
cansuperjump = physinfo"slj" == 1;
if (bInDuck || (flags & FL_DUCKING)) {
    if (cansuperjump && (cmd.buttons & IN_DUCK) && flDuckTime > 0 && Length(velocity) > 50) {   // longjump module
        punchangle[0] = -5;
        velocity[0..1] = forward[0..1] * PLAYER_LONGJUMP_SPEED * 1.6;   // 350 * 1.6
        velocity[2] = sqrt(2 * 800 * 56.0);
    } else velocity[2] = sqrt(2 * 800 * 45.0);             // = 268.32815729997476  (float-stored)
} else velocity[2] = sqrt(2 * 800 * 45.0);
PM_FixupGravityVelocity();                                 // pre-decay half gravity (see split-gravity note)
oldbuttons |= IN_JUMP;
```
**CS version (vanilla path, REGAMEDLL_ADD off):**
```c
if (dead) { oldbuttons |= IN_JUMP; return; }
if (waterjumptime != 0.0f) { waterjumptime -= cmd.msec; if (< 0) = 0; return; }
if (waterlevel >= 2) { ...same 100/80/50... return; }
if (onground == -1) { oldbuttons |= IN_JUMP; return; }
if (oldbuttons & IN_JUMP) return;                          // + REGAMEDLL_ADD: unless sv_autobunnyhopping > 0
if (bInDuck && (flags & FL_DUCKING)) return;               // CS-only guard
PM_CatagorizeTextureType();
onground = -1;
PM_PreventMegaBunnyJumping();                              // (REGAMEDLL_ADD can disable via sv_enablebunnyhopping)
real_t fvel = Length(velocity); float fvol = 1.0f;
if (fvel >= 150.0f) PM_PlayStepSound(PM_MapTextureTypeStepType(chtexturetype), fvol);   // sound only if moving >= 150
velocity[2] = PM_JumpHeight(false);                        // Q_sqrt(2.0 * 800.0f * 45.0f) — computed in DOUBLE, stored float
                                                           // (no duck/longjump special-case in vanilla CS)
// ---- STAMINA ----
if (pmove->fuser2 > 0.0f) {
    real_t flRatio = (100.0 - pmove->fuser2 * 0.001 * 19.0) * 0.01;   // double math — 'NOTE: don't do it in .f (float)'
    pmove->velocity[2] *= flRatio;                          // scales the jump velocity down (min ×0.75 right after a jump)
}
pmove->fuser2 = 1315.789429;                               // stamina timer reset (ms); 1315.789429*0.001*19 == 25.000000151
PM_FixupGravityVelocity();
oldbuttons |= IN_JUMP;
```
**fJumpHeld / edge trigger semantics:** the gate is `pmove->oldbuttons & IN_JUMP`. Within pm code, holding jump keeps IN_JUMP latched into oldbuttons; when jump is NOT pressed the pipeline executes `oldbuttons &= ~IN_JUMP`. Server-side, ReHLDS writes back `pev->oldbuttons = pmove->cmd.buttons` after each cmd, so effectively oldbuttons is the previous command's button mask — you must have at least one command without IN_JUMP between jumps (hence scroll-wheel bhop).
**Stamina timeline:** fuser2 = 1315.789429 at jump; decays by msec per tick in PM_ReduceTimers; while > 0 every ground frame PM_WalkMove multiplies horizontal velocity by `(100 - fuser2*0.001*19)*0.01` (0.75 → 1.0 as it decays over ~1.316 s), and a jump within that window has velocity[2] scaled by the same formula — consecutive jumps are both slower and lower.


### item 15

#### name
PM_PreventMegaBunnyJumping


#### spec_markdown
```c
maxscaledspeed = BUNNYJUMP_MAX_SPEED_FACTOR * pmove->maxspeed;   // HL: 1.7f * maxspeed;  CS: 1.2f * maxspeed
if (maxscaledspeed <= 0.0f) return;
spd = Length(velocity);                                          // CS: real_t
if (spd <= maxscaledspeed) return;
HL: fraction = (maxscaledspeed / spd) * 0.65;   //Returns the modifier for the velocity
CS: fraction = (maxscaledspeed / spd) * 0.8;
VectorScale(velocity, fraction, velocity);       // all 3 components scaled
```
Called only from PM_Jump, before the vertical velocity is set. Note the rescale math overshoots: the resulting speed is maxscaledspeed*0.65 (HL) / maxscaledspeed*0.8 (CS), i.e. CS caps a pre-jump 250-maxspeed player to 1.2*250 = 300, and if above that chops to 0.8 of it = 240 * (300/spd normalization) — exactly `spd_new = maxscaledspeed * 0.8` (since scaling is linear).


### item 16

#### name
PM_Duck


#### spec_markdown
**HL:**
```c
buttonsChanged = oldbuttons ^ cmd.buttons;  nButtonPressed = buttonsChanged & cmd.buttons;
if (cmd.buttons & IN_DUCK) oldbuttons |= IN_DUCK; else oldbuttons &= ~IN_DUCK;
if (iuser3 || dead) { if (flags & FL_DUCKING) PM_UnDuck(); return; }
if (flags & FL_DUCKING) { cmd.forwardmove *= PLAYER_DUCKING_MULTIPLIER; cmd.sidemove *= ...; cmd.upmove *= ...; }   // 0.333, only when fully ducked
if ((cmd.buttons & IN_DUCK) || bInDuck || (flags & FL_DUCKING)) {
    if (cmd.buttons & IN_DUCK) {
        if ((nButtonPressed & IN_DUCK) && !(flags & FL_DUCKING)) { flDuckTime = 1000; bInDuck = true; }
        time = max(0.0, (1.0 - (float)flDuckTime / 1000.0));
        if (bInDuck) {
            if (((float)flDuckTime / 1000.0 <= (1.0 - TIME_TO_DUCK)) || (onground == -1)) {   // 400 ms elapsed OR airborne → finish now
                usehull = 1;  view_ofs[2] = VEC_DUCK_VIEW /*12*/;  flags |= FL_DUCKING;  bInDuck = false;
                if (onground != -1) {                                    // ON GROUND: shift origin DOWN so feet stay planted
                    for (i=0;i<3;i++) origin[i] -= (player_mins[1][i] - player_mins[0][i]);   // z -= 18 with engine hulls
                    PM_FixPlayerCrouchStuck(STUCK_MOVEUP /*1*/);         // up to 36 (CS: HalfHumanHeight) 1-unit nudges up
                    PM_CatagorizePosition();
                }                                                        // IN AIR: NO origin shift → feet effectively rise 18 (duck-jump)
            } else {                                                     // transition: animate eye height
                fMore = (VEC_DUCK_HULL_MIN - VEC_HULL_MIN);              // (-18) - (-36) = 18
                duckFraction = PM_SplineFraction(time, (1.0/TIME_TO_DUCK));
                view_ofs[2] = ((VEC_DUCK_VIEW - fMore) * duckFraction) + (VEC_VIEW * (1-duckFraction));
            }
        }
    } else PM_UnDuck();
}
```
`PM_SplineFraction(value, scale)`: `value = scale*value; v2 = value*value; return 3*v2 - 2*v2*value;`
**CS differences:** gate first: `if (dead || (!(cmd.buttons & IN_DUCK) && !bInDuck && !(flags & FL_DUCKING))) return;` then `real_t mult = PLAYER_DUCKING_MULTIPLIER /*0.333*/;` applied to cmd moves **whenever the gate passes** (i.e. already while duck is merely pressed / in transition, not only when FL_DUCKING). CS view constants: `PM_VEC_DUCK_VIEW 12`, `PM_VEC_VIEW 17`, `fMore = (PM_VEC_DUCK_HULL_MIN - PM_VEC_HULL_MIN)` = 18 (REGAMEDLL_FIXES uses `player_mins[1][2]-player_mins[0][2]`). Origin shift non-fixes: `origin[2] = origin[2] - 18.0;`. No iuser3 duck-prevention in vanilla CS. Timing identical: `flDuckTime = 1000` ms, finish at `flDuckTime/1000.0 <= (1.0 - TIME_TO_DUCK)` with `TIME_TO_DUCK 0.4` → 400 ms, instant when airborne.


### item 17

#### name
PM_UnDuck


#### spec_markdown
```c
newOrigin = origin;
if (onground != -1)
    HL: for (i=0;i<3;i++) newOrigin[i] += (player_mins[1][i] - player_mins[0][i]);   // z += 18
    CS(non-fixes): newOrigin[2] += 18.0;
trace = PM_PlayerTrace(newOrigin, newOrigin, PM_NORMAL, -1);   // test target spot with CURRENT (duck) hull
if (!trace.startsolid) {
    usehull = 0;                                                // switch to standing hull
    trace = PM_PlayerTrace(newOrigin, newOrigin, PM_NORMAL, -1);// re-test with standing hull
    if (trace.startsolid) { usehull = 1; return; }              // blocked overhead: stay ducked
    flags &= ~FL_DUCKING;  bInDuck = false;  view_ofs[2] = VEC_VIEW /*HL 28; CS PM_VEC_VIEW 17*/;  flDuckTime = 0;
    CS extra: flTimeStepSound -= 100; if (< 0) = 0;
    origin = newOrigin;
    PM_CatagorizePosition();
}
```
In air, unducking does NOT shift origin (feet drop 18 relative to hull) — the counterpart of the duck-jump gain. CS REGAMEDLL_ADD `unduck_method`/`PLAYER_PREVENT_DDUCK` adds an early-out resetting hull/view if unducking mid-transition (non-vanilla).


### item 18

#### name
PM_GetHullBounds / GetHullBounds + engine hull tables


#### spec_markdown
There is no PM_GetHullBounds in HLSDK pm_shared.c; `pmove->player_mins/player_maxs[4]` are seeded by the ENGINE (ReHLDS engine/pmove.cpp `PM_Init`):
```c
vec3_t player_mins[MAX_MAP_HULLS] = {
    { -16.0f, -16.0f, -36.0f, },   // hull 0: standing
    { -16.0f, -16.0f, -18.0f, },   // hull 1: ducked
    {   0.0f,   0.0f,   0.0f, },   // hull 2: point
    { -32.0f, -32.0f, -32.0f, }    // hull 3: large
};
vec3_t player_maxs[MAX_MAP_HULLS] = {
    { 16.0f, 16.0f, 36.0f, },
    { 16.0f, 16.0f, 18.0f, },
    {  0.0f,  0.0f,  0.0f, },
    { 32.0f, 32.0f, 32.0f, }
};
```
The game dll exports `GetHullBounds(hullnumber, mins, maxs)`; in BOTH vanilla HLSDK (dlls/client.cpp:1854) and vanilla CS it only assigns the local pointer parameters (`mins = VEC_HULL_MIN;`) — a no-op bug — so the engine tables above are authoritative. ReGameDLL_FIXES actually memcpy's CS's values: hull0 `Vector(-16, -16, -36)`/`Vector(16, 16, 36)`, hull1 `Vector(-16, -16, -18)`/`Vector(16, 16, 32)` (note CS duck maxs z = **32**, but only effective with REGAMEDLL_FIXES builds), hull2 zero. HLSDK util.h: `VEC_HULL_MIN Vector(-16, -16, -36)`, `VEC_HULL_MAX Vector( 16,  16,  36)`, `VEC_DUCK_HULL_MIN Vector(-16, -16, -18 )`, `VEC_DUCK_HULL_MAX Vector( 16,  16,  18)`. BSP hull selection (engine world.cpp PM_HullForBsp): xy size <= 36 → z size <= 36 ? map hull 3 (crouch) : map hull 1 (human); else map hull 2 (large); traces return endpoints for the hull's point-expanded planes — collision resolution itself lives in the engine trace code, out of pm_shared.


### item 19

#### name
PM_SpectatorMove / PM_NoClip / PM_Physics_Toss (hook points only)


#### spec_markdown
Out of requested scope; where they hook: spectator handled at top of PM_PlayerMove (`spectator || iuser1 > 0`; CS additionally `PM_ShouldDoSpectMode()` = `iuser3 <= 0 || deadflag == DEAD_DEAD`) — roaming applies friction*1.5 and accelerate against `movevars->spectatormaxspeed` (500). NOCLIP: direct origin integration `origin += frametime * wishvel`, velocity cleared. TOSS/BOUNCE (dead bodies): PM_AddGravity + PM_PushEntity + PM_ClipVelocity with backoff `2.0 - friction` (BOUNCE) / `1` else; stop when `normal[2] > 0.7` and `vel dot vel < (30 * 30)`.


### item 20

#### name
SV_RunCmd / SV_CheckMovingGround (engine wrapper — basevelocity & trigger_push)


#### spec_markdown
See tick pipeline section 0 for full detail. Key exact snippets (ReHLDS sv_user.cpp):
```c
if (cmd.msec > 50) { cmd.msec = (byte)(ucmd->msec / 2.0); SV_RunCmd(&cmd, seed, fNetCmd, TRUE);
                     cmd.msec = (byte)(ucmd->msec / 2.0); cmd.impulse = 0; SV_RunCmd(&cmd, seed, fNetCmd, TRUE); return; }
frametime = float(ucmd->msec * 0.001);
// SV_CheckMovingGround: conveyor → basevelocity = movedir*speed (accumulate if FL_BASEVELOCITY)
// no FL_BASEVELOCITY this frame → VectorMA(velocity, frametime * 0.5f + 1.0f, basevelocity, velocity); VectorClear(basevelocity);
pmove->usehull = (sv_player->v.flags & FL_DUCKING) == FL_DUCKING;
pmove->maxspeed = sv_maxspeed.value;  pmove->clientmaxspeed = sv_player->v.maxspeed;
pmove->waterjumptime = sv_player->v.teleport_time;
... pfnPM_Move(pmove, TRUE) ...
sv_player->v.flDuckTime = (int)pmove->flDuckTime;   // TRUNCATED to int each cmd
sv_player->v.oldbuttons = pmove->cmd.buttons;        // oldbuttons := this cmd's buttons
```
trigger_push (ReGameDLL triggers.cpp CTriggerPush::Touch): continuous push field → `vecPush = pev->speed * pev->movedir; if (FL_BASEVELOCITY) vecPush += basevelocity; basevelocity = vecPush; flags |= FL_BASEVELOCITY;`. PUSH_ONCE → `velocity += speed*movedir; if (velocity.z > 0) flags &= ~FL_ONGROUND; remove trigger`. Inside pm code, horizontal basevelocity is added before the slide (`VectorAdd(velocity, basevelocity, velocity)` in WalkMove/AirMove) and subtracted after `PM_CatagorizePosition` back in PlayerMove; vertical basevelocity is consumed as an acceleration in PM_AddCorrectGravity (`velocity[2] += basevelocity[2]*frametime; basevelocity[2] = 0`).


## constants

### item 0

#### name
STOP_EPSILON


#### value
0.1


#### where
pm_shared.c:88 (HLSDK) / pm_shared.h:53 (ReGameDLL_CS); used in PM_ClipVelocity


#### notes
Per-component zeroing band after every clip: out[i] in (-0.1, 0.1) → 0


### item 1

#### name
plane-normal ground threshold


#### value
0.7 (HLSDK, double literal) / 0.7f (ReGameDLL_CS)


#### where
PM_CatagorizePosition, PM_FlyMove, PM_WalkMove step-down, both repos


#### notes
normal[2] < 0.7 → never onground (surf). 0.7f = 0.699999988079071 — see gotchas


### item 2

#### name
onground vertical-velocity cutoff


#### value
180


#### where
PM_CatagorizePosition (both): if (pmove->velocity[2] > 180)


#### notes
int literal; jump vel 268.33 > 180 → instantly airborne


### item 3

#### name
ground-probe depth


#### value
2


#### where
PM_CatagorizePosition: point[2] = pmove->origin[2] - 2


#### notes
down-trace distance; also snap-down distance


### item 4

#### name
numbumps


#### value
4


#### where
PM_FlyMove (both)


#### notes
max clip iterations per move


### item 5

#### name
MAX_CLIP_PLANES


#### value
5


#### where
pm_defs.h (both)


#### notes
plane buffer in PM_FlyMove


### item 6

#### name
air wishspeed cap


#### value
30


#### where
PM_AirAccelerate (both): if (wishspd > 30) wishspd = 30;


#### notes
caps addspeed test only; accel rate still uses full wishspeed


### item 7

#### name
half-gravity factor


#### value
0.5 (HL) / 0.5f (CS AddCorrectGravity), 0.5 (both Fixup)


#### where
PM_AddCorrectGravity / PM_FixupGravityVelocity


#### notes
leapfrog split


### item 8

#### name
jump velocity


#### value
sqrt(2 * 800 * 45.0) (HLSDK) / Q_sqrt(2.0 * 800.0f * 45.0f) (CS PM_JumpHeight)


#### where
PM_Jump; hl_pm_shared.c:2596,2601 / cs_pm_shared.cpp:2629


#### notes
= 268.32815729997476…, stored into float velocity[2]; hard-coded 800 regardless of sv_gravity


### item 9

#### name
longjump velocity


#### value
sqrt(2 * 800 * 56.0); xy = forward * PLAYER_LONGJUMP_SPEED * 1.6


#### where
PM_Jump (HL; CS only with REGAMEDLL_ADD)


#### notes
PLAYER_LONGJUMP_SPEED 350 (HL) / 350.0f (CS)


### item 10

#### name
BUNNYJUMP_MAX_SPEED_FACTOR


#### value
1.7f (HLSDK) / 1.2f (ReGameDLL_CS)


#### where
pm_shared.c:2436 / pm_shared.h:74


#### notes
threshold = factor * pmove->maxspeed


### item 11

#### name
bunnyhop crop factor


#### value
0.65 (HLSDK) / 0.8 (ReGameDLL_CS)


#### where
PM_PreventMegaBunnyJumping: fraction = (maxscaledspeed / spd) * 0.65 / * 0.8


#### notes
post-crop speed = maxscaledspeed * factor


### item 12

#### name
stamina reset value (fuser2)


#### value
1315.789429


#### where
cs_pm_shared.cpp:2805 (PM_Jump)


#### notes
ms countdown; 1315.789429 * 0.001 * 19 = 25.000000151 → initial ratio ≈ 0.75


### item 13

#### name
stamina ratio formula


#### value
(100 - pmove->fuser2 * 0.001 * 19) * 0.01 (WalkMove) / (100.0 - pmove->fuser2 * 0.001 * 19.0) * 0.01 (Jump)


#### where
cs_pm_shared.cpp:1062, 2800


#### notes
real_t (double) math; applied to velocity[0..1] every ground frame, and to velocity[2] once at jump


### item 14

#### name
TIME_TO_DUCK


#### value
0.4


#### where
pm_shared.c:77 / pm_shared.h:61


#### notes
seconds; duck completes when flDuckTime/1000.0 <= 1.0 - 0.4


### item 15

#### name
duck timer start


#### value
1000


#### where
PM_Duck: pmove->flDuckTime = 1000


#### notes
ms; comment 'Use 1 second so super long jump will work'


### item 16

#### name
PLAYER_DUCKING_MULTIPLIER


#### value
0.333


#### where
pm_shared.c:123 / pm_shared.h:55


#### notes
cmd move scale while ducked (CS: while ducking at all); 0.333 not 1/3


### item 17

#### name
duck origin shift


#### value
player_mins[1] - player_mins[0] = (0, 0, 18); CS non-fixes literal 18.0


#### where
PM_Duck / PM_UnDuck (both)


#### notes
applied only when onground != -1


### item 18

#### name
view heights


#### value
HL: VEC_VIEW 28, VEC_DUCK_VIEW 12; CS: PM_VEC_VIEW 17, PM_VEC_DUCK_VIEW 12; PM_DEAD_VIEWHEIGHT -8


#### where
pm_shared.c:78-87 / pm_shared.h:31,64-67


#### notes
fMore = VEC_DUCK_HULL_MIN - VEC_HULL_MIN = -18 - -36 = 18 in transition formula


### item 19

#### name
engine player hulls


#### value
hull0 (-16,-16,-36)/(16,16,36); hull1 (-16,-16,-18)/(16,16,18); hull2 (0,0,0); hull3 (-32,-32,-32)/(32,32,32)


#### where
ReHLDS engine/pmove.cpp:36-48 (authoritative — game-dll GetHullBounds is a no-op bug in vanilla)


#### notes
CS ReGameDLL_FIXES GetHullBounds would set duck maxs (16,16,32) (util.h VEC_DUCK_HULL_MAX Vector(16, 16, 32))


### item 20

#### name
edgefriction probe


#### value
16 ahead (vel/speed*16), start z = origin.z + player_mins[usehull][2], down 34


#### where
PM_Friction (both)


#### notes
trace.fraction == 1.0 → friction *= movevars->edgefriction


### item 21

#### name
friction dead-zone / walk clear


#### value
speed < 0.1f (PM_Friction return); spd < 1.0f (HL) / 1.0 (CS) → VectorClear (PM_WalkMove)


#### where
PM_Friction / PM_WalkMove


#### notes



### item 22

#### name
PM_CHECKSTUCK_MINTIME


#### value
0.05


#### where
pm_shared.c:1632 / pm_shared.h:56


#### notes
seconds between expensive unstick attempts


### item 23

#### name
stuck table size


#### value
54


#### where
rgv3tStuckTable[54], PM_CreateStuckTable (both)


#### notes
nudges ±0.125 and big moves x/y ±2.0f, z {0.0f,1.0f,6.0f}


### item 24

#### name
frametime derivation


#### value
pmove->frametime = pmove->cmd.msec * 0.001


#### where
PM_PlayerMove (both); engine: frametime = float(ucmd->msec * 0.001)


#### notes
cmd.msec is byte; engine chops msec > 50 into two (byte)(msec / 2.0) halves


### item 25

#### name
sv_gravity default


#### value
"800"


#### where
ReHLDS engine/sv_phys.cpp:49


#### notes
movevars->gravity


### item 26

#### name
sv_stopspeed default


#### value
"100"


#### where
ReHLDS engine/sv_phys.cpp:53


#### notes
CS servers conventionally run 75 via config


### item 27

#### name
sv_friction default


#### value
"4"


#### where
ReHLDS engine/sv_phys.cpp:52


#### notes



### item 28

#### name
edgefriction default


#### value
"2"


#### where
ReHLDS engine/sv_user.cpp:50 (cvar name is edgefriction)


#### notes



### item 29

#### name
sv_accelerate default


#### value
"10"


#### where
ReHLDS engine/sv_user.cpp:52


#### notes
CS servers conventionally run 5 via config


### item 30

#### name
sv_airaccelerate default


#### value
"10"


#### where
ReHLDS engine/sv_main.cpp:127


#### notes
surf servers typically set 100+


### item 31

#### name
sv_maxspeed default


#### value
"320"


#### where
ReHLDS engine/sv_user.cpp:51


#### notes
pmove->maxspeed; CS clientmaxspeed (pev->maxspeed per weapon, e.g. 250) min'd in via PM_CheckParameters; CS servers usually sv_maxspeed 900


### item 32

#### name
sv_maxvelocity default


#### value
"2000"


#### where
ReHLDS engine/sv_phys.cpp:48


#### notes
per-axis clamp in PM_CheckVelocity


### item 33

#### name
sv_stepsize default


#### value
"18"


#### where
ReHLDS engine/sv_phys.cpp:51


#### notes
movevars->stepsize in PM_WalkMove step logic


### item 34

#### name
sv_bounce default


#### value
"1"


#### where
ReHLDS engine/sv_phys.cpp:50


#### notes
overbounce = 1.0 + bounce*(1-friction) = 1.0 with friction 1


### item 35

#### name
sv_spectatormaxspeed default


#### value
"500"


#### where
ReHLDS engine/sv_main.cpp:126


#### notes



### item 36

#### name
CS jump-sound speed threshold


#### value
150.0f


#### where
cs_pm_shared.cpp:2748


#### notes
sound only; no physics effect


### item 37

#### name
IS_NAN / nanmask


#### value
#define IS_NAN(x) (((*(int *)&x)&nanmask)==nanmask); nanmask = 255<<23 (HL: int; CS: const int, pm_math.cpp:4)


#### where
HLSDK common/mathlib.h:40-42; ReGameDLL pm_math.cpp


#### notes
exponent-bits NaN/Inf test on float32 bit pattern


### item 38

#### name
waterjump velocity constants (hook reference)


#### value
waterjumptime = 2000; velocity[2] = 225 (HL) / 2000.0f, 225.0f (CS); WJ_HEIGHT 8; swim jump 100/80/50


#### where
PM_CheckWaterJump / PM_Jump


#### notes
out of scope but pipeline-visible


### item 39

#### name
CS fall-sound constants


#### value
PM_PLAYER_FALL_PUNCH_THRESHHOLD 250, PM_PLAYER_MAX_SAFE_FALL_SPEED 580, PM_PLAYER_MIN_BOUNCE_SPEED 350 (HL: 350 / 580 / 200); punch: flFallVelocity * 0.013


#### where
pm_shared.h:69-71 / pm_shared.c:115-119


#### notes
audio/viewpunch only; fall damage is computed in game dll from flFallVelocity


## cs_deviations_markdown
# Every movement-relevant difference: HLSDK pm_shared.c vs ReGameDLL_CS pm_shared.cpp

ReGameDLL_CS is a decompile of the shipped CS 1.6 gamedll (HLDS build 6153 cs.so); code inside `#ifdef REGAMEDLL_ADD` / `REGAMEDLL_FIXES` / `REGAMEDLL_API` is NOT vanilla CS 1.6 — the vanilla behavior is the unguarded path. Differences of the vanilla CS path vs HLSDK:

1. **Stamina system (fuser2)** — CS only:
   - `PM_Jump` sets `pmove->fuser2 = 1315.789429;` after a successful jump.
   - `PM_ReduceTimers` decays it: `if (fuser2 > 0.0) { fuser2 -= cmd.msec; if (< 0.0) fuser2 = 0; }` (~1.316 s).
   - `PM_WalkMove` (top of function, every ground frame while fuser2 > 0): `real_t flRatio = (100 - fuser2 * 0.001 * 19) * 0.01; velocity[0] *= flRatio; velocity[1] *= flRatio;` (0.75 → 1.0 as it decays).
   - `PM_Jump` (after setting velocity[2]): `real_t flRatio = (100.0 - fuser2 * 0.001 * 19.0) * 0.01; velocity[2] *= flRatio;` — a jump made <1.316 s after the previous is up to 25% lower.
   - `CBasePlayer::ResetStamina()` zeroes fuser1/2/3 (spawn etc.).
2. **PM_PreventMegaBunnyJumping**: HL `BUNNYJUMP_MAX_SPEED_FACTOR 1.7f`, crop `* 0.65`; CS `1.2f`, crop `* 0.8`. CS threshold uses pmove->maxspeed (min of sv_maxspeed and per-weapon speed, typically 250 → cap kicks in at 300).
3. **PM_Jump**:
   - CS adds `if (pmove->bInDuck && (pmove->flags & FL_DUCKING)) return;` guard.
   - CS calls `PM_CatagorizeTextureType()` before jumping, plays the jump step-sound only when `Length(velocity) >= 150.0f` (HL: always, volume 1.0).
   - Vanilla CS has NO longjump module and no duck special-case for jump height: always `Q_sqrt(2.0 * 800.0f * 45.0f)` (the duck/slj branch exists only under REGAMEDLL_ADD). HL supports `physinfo "slj"` longjump (350*1.6 fwd, sqrt(2*800*56.0) up, punchangle[0] = -5) and TFC feign-death checks.
   - CS jump height computed via `PM_JumpHeight` in double, HL via `sqrt(2 * 800 * 45.0)` (int*int*double).
4. **PM_Duck / PM_UnDuck**:
   - CS gate: `if (dead || (!(cmd.buttons & IN_DUCK) && !bInDuck && !(flags & FL_DUCKING))) return;` — dead players just return (HL tries UnDuck when iuser3||dead). Vanilla CS has no iuser3 duck-prevention (HL does: `if (pmove->iuser3 || pmove->dead)`).
   - CS applies the 0.333 cmd multiplier whenever the gate passes (pressed OR transitioning OR ducked); HL only when `FL_DUCKING` is already set.
   - View heights: HL VEC_VIEW 28; CS PM_VEC_VIEW 17 (duck view 12 in both).
   - CS PM_Duck computes `real_t time = (1.0 - flDuckTime / 1000.0)` guarded `if (time >= 0.0)` instead of HL's `max(0.0, …)`; duckFraction initialized to `PM_VEC_VIEW` (decompile artifact, unreachable).
   - CS origin shifts use the literal `18.0` (non-fixes) instead of the `player_mins[1]-player_mins[0]` loop.
   - CS PM_UnDuck additionally does `flTimeStepSound -= 100` (floor 0).
   - CS duck hull maxs per util.h is `Vector(16, 16, 32)` vs HL `Vector(16, 16, 18)` — only effective in REGAMEDLL_FIXES builds (vanilla GetHullBounds is a no-op; engine table hull1 maxs z = 18 applies).
5. **PM_PlayerMove ordering**: CS calls `PM_Duck(); PM_UpdateStepSound();` — HL calls `PM_UpdateStepSound(); PM_Duck();`.
6. **Stuck handling**: CS `if (PM_CheckStuck()) return;` — HL retries once with `PM_Duck()` between two `PM_CheckStuck()` calls. Inside PM_CheckStuck, HL nudges when `(!server || !multiplayer)`, CS only when `!server`.
7. **Spectator gate**: CS adds `PM_ShouldDoSpectMode()` (`iuser3 <= 0 || deadflag == DEAD_DEAD`) to the `spectator || iuser1 > 0` test.
8. **maxspeed handling**: identical min(clientmaxspeed, maxspeed) logic, but HLSDK master additionally has the `IN_USE` on-ground `maxspeed *= 1.0f / 3.0f` in PM_CheckParamters (older SDKs: `VectorScale(velocity, 0.3, velocity)` inside PM_WalkMove). CS has NO +use slowdown at all. CS relies on `pev->maxspeed` per weapon (assigned by game code, e.g. 250) flowing into `clientmaxspeed`.
9. **PM_CheckWater**: CS REGAMEDLL_FIXES adds a dead-player early-out; vanilla identical. Function renamed PM_CatagorizePosition → PM_CategorizePosition (behavior identical apart from the early `return` when `velocity[2] > 180` and a NOCLIP/observer fix under REGAMEDLL_FIXES).
10. **PM_FlyMove**: logic identical to HL; CS computes `end[i] = origin[i] + (real_t)(time_left * velocity[i])` in double, and the bounce overbounce as `1.0 + movevars->bounce * (1.0 - pmove->friction)` (HL: `(1-pmove->friction)`, same value). CS additionally contains `PM_FlyMove_New()` (Quake-3 style with plane-nudging), selected by `flymove_method` cvar — REGAMEDLL_ADD, default off, NOT vanilla.
11. **PM_Friction**: CS `drop += friction * (control * pmove->frametime)` vs HL `drop += control*friction*pmove->frametime` (different FP association); CS `real_t` (double) for newspeed/control/friction/drop and the mixed `float(newspeed)` casts on components 1 and 2.
12. **PM_ClipVelocity**: CS `angle`/`backoff` are `real_t` (double); change computed as `in[i] - normal[i] * backoff` in one expression.
13. **PM_Accelerate / PM_AirAccelerate**: CS uses `real_t` for wishspeed param (PM_Accelerate), currentspeed and accelspeed; HL all float.
14. **PM_CheckFalling constants**: threshold 250 (HL 350), min-bounce 350 (HL 200); CS has no fall-pain sounds from pm code (fvol table only) — audio/view only.
15. **Literals**: CS consistently uses `0.7f` where HL uses `0.7` (double) in plane-normal comparisons — one representable-value edge difference (see gotchas).
16. **REGAMEDLL_ADD (non-vanilla, default-off) extras to be aware of and NOT implement for a vanilla port**: sv_autobunnyhopping (skips the oldbuttons jump gate), sv_enablebunnyhopping (skips PreventMegaBunnyJumping), jump_height cvar, stamina_restore_rate (fps-normalized stamina pow), IN_RUN/fuser3 speed boost in WalkMove/NoClip/Spectator, freezetime duck/jump prevention, PLAYER_PREVENT_* iuser3 flags, unduck_method, hull-bounds cvar, PM_FlyMove_New."


## gotchas_markdown
# Float32 bit-parity gotchas

1. **Storage vs intermediates**: `vec_t` is `float` (float32); origin/velocity/basevelocity/angles are all `vec3_t` = `float[3]`. But the ORIGINAL shipped mp.dll/cs.so were 32-bit x87 builds where intermediates lived at higher precision. ReGameDLL encodes this explicitly: `#ifdef PLAY_GAMEDLL typedef double real_t; #else typedef float real_t; #endif` (regamedll/common/mathlib.h:31-37, "NOTE: In some cases we need high precision of floating-point, so use double instead of float, otherwise unittest will fail"). **For bit-parity with real CS 1.6 servers you must reproduce the `real_t`(=double) intermediates at exactly the marked sites and truncate to float32 exactly where the code stores back into vec_t.** Notable real_t sites: PM_ClipVelocity `angle`/`backoff`; PM_Accelerate `wishspeed`(param)/`currentspeed`/`accelspeed`; PM_AirAccelerate `currentspeed`/`accelspeed`; PM_Friction `newspeed`/`control`/`friction`/`drop` (but `speed` is float!); PM_WalkMove `spd`/`wishspeed`/stamina `flRatio`; PM_FlyMove `flScale = time_left * velocity[i]`; PM_Jump `fvel`, stamina `flRatio`, `PM_JumpHeight` (`Q_sqrt(2.0*800.0f*45.0f)` in double → float on store); PM_CheckParameters `maxspeed`/`fRatio` and `Q_sqrt(real_t(...))`; PM_DropPunchAngle `len`; AngleVectors `angle`/`cy` (only those two — sy/sp/cp/sr/cr stay float); `Length()` and `VectorNormalize()` return real_t and compute `Q_sqrt` on a double sum; `DotProduct` (vector.h inline) returns real_t.
2. **Order of operations differs between HL and CS and must be copied verbatim**: e.g. HL `drop += control*friction*pmove->frametime` (left-to-right float) vs CS `drop += friction * (control * pmove->frametime)` (double, different association); HL ClipVelocity `change = normal[i]*backoff; out[i] = in[i] - change` vs CS `out[i] = in[i] - normal[i]*backoff` with double backoff; CS PM_Friction writes `newvel[0] = vel[0] * newspeed` (double multiply) but `newvel[1] = vel[1] * float(newspeed)` and `newvel[2] = vel[2] * float(newspeed)` (float multiply) — asymmetric per-component rounding, faithful to the decompiled binary.
3. **frametime**: `pmove->frametime = pmove->cmd.msec * 0.001` — int byte times double 0.001, truncated to float member. 0.001 is not representable; msec=10 gives 0.009999999776482582… Do NOT substitute 1.0/fps or double frametime. The engine independently computes `frametime = float(ucmd->msec * 0.001)` for pre-move logic. Commands with msec > 50 are split server-side into two `(byte)(ucmd->msec / 2.0)` halves (odd msec loses 1 ms total).
4. **Literal width matters**: HL compares plane normals with double literal `0.7` (float normal promoted exactly to double), CS with `0.7f` (= 0.699999988079071). A plane whose float normal.z is exactly 0.7f is "ground" (>= tests are `<`/`>` so: `0.7f < 0.7` is TRUE in HL → not ground; `0.7f < 0.7f` FALSE in CS → ground). Same for `velocity[2] > 180` (int → exact) and `speed < 0.1f` vs stamina constants written in double (`0.001`, `19.0`, `0.01`, `100.0`). Copy each literal's type exactly as quoted.
5. **sqrt flavors**: HL calls `sqrt()` (double) on double-promoted float expressions; CS `Q_sqrt` maps to `sqrt` (double) or SSE `M_sqrt` depending on build (`#define Q_sqrt M_sqrt` under HAVE_SSE, else `#define Q_sqrt sqrt`, public/strtools.h:92/133). x87-parity builds use libm double sqrt. `sqrt(2 * 800 * 45.0)` = 268.32815729997476 stored as float32 268.32815551757812 (0x43862A01).
6. **STOP_EPSILON quantization**: every PM_ClipVelocity call zeroes any output component in (-0.1, 0.1) — this happens per bump iteration inside PM_FlyMove and feeds back into subsequent clips; skipping it desyncs immediately on ramps.
7. **Split gravity + jump overwrite**: PM_Jump overwrites velocity[2] AFTER PM_AddCorrectGravity already applied the first half-gravity, then calls PM_FixupGravityVelocity itself; the pipeline-end Fixup still runs. Emulating "one gravity per frame" naively breaks jump height by exactly half a tick of gravity.
8. **PM_CheckVelocity is per-axis**, ±sv_maxvelocity (2000), applied at 4+ points per tick; NaN scrub uses the bit-pattern macro `IS_NAN(x) (((*(int *)&x)&nanmask)==nanmask)`, `nanmask = 255<<23` — reproduce as bit test on the float32 pattern, not isnan().
9. **Engine RNG does NOT touch physics**: `pmove->RandomLong` is used only in PM_PlayStepSound / swim / water sounds and PM_GetRandomStuckOffsets is deterministic (cyclic table index per player, reset on unstick). No random numbers enter velocity/origin math. `shared_rand`/random_seed affects weapons only.
10. **State writeback truncation (server)**: `pev->flDuckTime = (int)pmove->flDuckTime` and `flSwimTime` likewise — ms timers quantize to whole ints between commands; `flTimeStepSound` is declared `int` in playermove_t. And `pev->oldbuttons = pmove->cmd.buttons` after each move — the pm-internal `oldbuttons |= IN_JUMP` latches only matter within one command; the effective jump edge-trigger is "previous cmd had no IN_JUMP".
11. **usehull is recomputed by the engine** each command from FL_DUCKING (`pmove->usehull = (flags & FL_DUCKING) == FL_DUCKING`), so pm-side hull switches (PM_Duck/PM_UnDuck) persist only via the FL_DUCKING flag.
12. **pmove->friction is player friction (pev->friction, default 1.0)** — distinct from movevars->friction (sv_friction 4). PM_Move resets it to 1.0f only in single player; func_friction can change it in MP and it alters both PM_Friction and the FlyMove overbounce (`1.0 + bounce*(1-friction)`).
13. **VectorNormalize mutates in place and returns pre-normalization length**; forward/right are re-normalized after z-zeroing each move call — pmove->forward/right/up are scratch, rebuilt from angles each tick by AngleVectors first.
14. **PM_CatagorizePosition snaps origin down (≤2 units) every ground frame** and PM_WalkMove's step-down decides via `normal[2] < 0.7`; both must run in the exact pipeline positions or ramp-edge behavior (surf takeoffs, "sticky" stairs) diverges.
15. **allFraction == 0 kill switch**: if all 4 bumps started solid-blocked (`trace.fraction` summed to 0), velocity is fully zeroed — this is why hitting a ramp seam dead-on stops you; emulate exactly (it sums fractions as floats).
16. **Angles**: pmove->angles come from cmd.viewangles + punchangle with the roll formula (`PM_CalcRoll * 4`); anglemod/±180 wrap (`if (angles[YAW] > 180.0f) angles[YAW] -= 360.0f`) affects AngleVectors input bit-exactly. PITCH positive = down; wishdir ignores pitch on ground/air because forward.z is zeroed then re-normalized (a pitched-down forward still yields |wishvel| = fmove after renormalization — NOT scaled by cos(pitch)).\n17. **Trace dependency**: bit-parity of the movement math is necessary but positions also depend on `PM_PlayerTrace` (engine BSP hull clipping: fraction, endpos with its own epsilon nudges (DIST_EPSILON 0.03125), plane normals). A standalone port needs a byte-faithful GoldSrc hull trace (see ReHLDS engine/world.cpp / pm's PM_PlayerTrace) — endpos fractions feed directly into `time_left -= time_left * trace.fraction`."


## sources
https://raw.githubusercontent.com/ValveSoftware/halflife/master/pm_shared/pm_shared.c (HLSDK master, read in full; note: master includes Valve's post-2023 'JoshA' change moving the IN_USE slowdown into PM_CheckParamters)

https://raw.githubusercontent.com/ValveSoftware/halflife/master/pm_shared/pm_math.c (AngleVectors, Length, VectorNormalize)

https://raw.githubusercontent.com/ValveSoftware/halflife/master/pm_shared/pm_movevars.h (movevars_t layout)

https://raw.githubusercontent.com/ValveSoftware/halflife/master/common/mathlib.h (IS_NAN, nanmask)

https://raw.githubusercontent.com/ValveSoftware/halflife/master/dlls/client.cpp (GetHullBounds)

https://raw.githubusercontent.com/ValveSoftware/halflife/master/dlls/util.h (VEC_HULL_MIN/MAX, VEC_DUCK_*)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/pm_shared/pm_shared.cpp (CS 1.6 authoritative movement, read in full)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/pm_shared/pm_shared.h (CS constants: BUNNYJUMP_MAX_SPEED_FACTOR 1.2f, PM_VEC_VIEW 17, etc.)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/pm_shared/pm_math.cpp (CS AngleVectors/VectorNormalize with real_t)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/pm_shared/pm_defs.h (playermove_t, MAX_CLIP_PLANES 5)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/common/usercmd.h (usercmd_t: byte msec, float forwardmove/sidemove/upmove)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/common/mathlib.h (real_t typedef: double under PLAY_GAMEDLL)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/public/strtools.h (Q_sqrt/Q_min/Q_max mappings)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/client.cpp (CS GetHullBounds)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/util.h (CS hull vectors incl. VEC_DUCK_HULL_MAX Vector(16,16,32))

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/vector.h (DotProduct returning real_t)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/player.cpp (CBasePlayer::ResetStamina fuser2=0; basevelocity reset on spawn)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/triggers.cpp (CTriggerPush::Touch basevelocity/FL_BASEVELOCITY)

https://raw.githubusercontent.com/rehlds/rehlds/master/rehlds/engine/sv_user.cpp (SV_RunCmd: msec chop >50, frametime, pmove setup/writeback, oldbuttons=cmd.buttons, edgefriction/sv_maxspeed/sv_accelerate cvar defaults)

https://raw.githubusercontent.com/rehlds/rehlds/master/rehlds/engine/sv_phys.cpp (sv_gravity 800, sv_maxvelocity 2000, sv_friction 4, sv_stopspeed 100, sv_stepsize 18, sv_bounce 1)

https://raw.githubusercontent.com/rehlds/rehlds/master/rehlds/engine/sv_main.cpp (sv_airaccelerate 10, sv_wateraccelerate 10, sv_spectatormaxspeed 500)

https://raw.githubusercontent.com/rehlds/rehlds/master/rehlds/engine/pmove.cpp (engine player_mins/player_maxs hull tables — authoritative due to GetHullBounds no-op bug)

https://raw.githubusercontent.com/rehlds/rehlds/master/rehlds/engine/world.cpp (PM_HullForBsp hull selection by size)

Fetched reference files (hl_pm_shared.c, cs_pm_shared.cpp, cs_pm_shared.h, cs_pm_math.cpp, cs_pm_defs.h, cs_mathlib.h, cs_usercmd.h, cs_util.h, cs_client.cpp, hl_client.cpp, cs_triggers.cpp, cs_player.cpp, rehlds_sv_user.cpp, rehlds_sv_phys.cpp, rehlds_sv_main.cpp, rehlds_pmove.cpp, rehlds_world.cpp)

