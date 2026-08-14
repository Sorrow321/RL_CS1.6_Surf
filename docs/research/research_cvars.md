
## cvars

### item 0

#### name
sv_airaccelerate


#### engine_default
10 (ReHLDS sv_main.cpp: cvar_t sv_airaccelerate = {"sv_airaccelerate", "10", FCVAR_SERVER}). Same default in HL and CS 1.6.


#### surf_typical
100 — the CS 1.6 surf standard. UltimateSurf (AMXX surf management plugin) ships `sv_airaccelerate "100"` commented "This is a must in all surf servers"; GameBanana CS 1.6 surf map pages recommend 100; the fix_fps_speed AMXX plugin is explicitly tuned for aa=100 servers. Variants: fun/easy servers run higher (150/200/400/800/1000 — a difficulty ladder popularized by Source-era guides); one guide (tobyscs) claims 300 for 1.6 but the author self-flags uncertainty.


#### why_it_matters
THE surf value. In PM_AirAccelerate: accelspeed = airaccelerate * wishspeed * frametime * friction, capped by addspeed = min(wishspeed,30) - dot(vel,wishdir). At aa=10 with knife (wishspeed 250, dt 0.01) accelspeed=25 < 30 so accel is rate-limited; at aa=100 accelspeed=250 >> 30, so the 30-unit projection saturates every single frame — air control becomes effectively instantaneous per-tick, which is what makes GoldSrc surfing and sharp ramp transitions viable. Saturation also makes speed gained through turns scale with client FPS (more frames = more re-aims per second), the known 'high-FPS advantage' on 1.6 surf.


#### confidence
high — engine default read from ReHLDS source; surf value 100 confirmed by multiple 1.6-specific artifacts (plugin config, map pages, fps-fix plugin).


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_main.cpp

https://github.com/tonykaram1993/UltimateSurf/blob/master/configs/UltimateSurf.cfg

https://github.com/igorkfmoura/amxx-plugins

https://gamebanana.com/mods/cats/5501

https://www.tobyscs.com/cs-surf-settings/


### item 1

#### name
sv_accelerate


#### engine_default
10 (engine, ReHLDS sv_user.cpp) — this is the raw HL/HLDS default; CS 1.6's stock server config sets 5, and 5 is universally treated as the 'CS 1.6 default'.


#### surf_typical
5 (left at CS default on virtually all 1.6 surf servers; a minority of configs use 8-10). Guides warn values >10 make ground strafing feel exaggerated.


#### why_it_matters
Ground acceleration: accelspeed = accelerate * frametime * wishspeed * player_friction, capped at (wishspeed - dot(vel,wishdir)). Governs start/stop feel between ramps and on stages/platforms; irrelevant while airborne.


#### confidence
high on defaults (engine source + widely documented CS config value); medium on surf-typical (few 1.6 surf cfg pastes explicitly set it, meaning they inherit 5).


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_user.cpp

https://gist.github.com/Log1x/b99213403bcbef9b5f32e0b11c419f19

http://gaming-blog.blogspot.com/2008/12/counter-strike-16-dedicated-server.html


### item 2

#### name
sv_gravity


#### engine_default
800 (ReHLDS sv_phys.cpp). Same in HL.


#### surf_typical
800. CS 1.6 surf maps are built for stock gravity; per-map exceptions exist (some space/fun maps 500-700), usually applied by per-map cfg. Note: sv_gravity does NOT change jump-off velocity in CS 1.6 (see maxspeed/mechanics section — jump vz is hardcoded sqrt(2*800*45)).


#### why_it_matters
Downward acceleration each frame (v_z -= g*dt, applied half before/half after move via PM_FixupGravityVelocity). On a surf ramp, gravity's component along the ramp plane is the engine of speed gain; changing it rescales every map's intended lines.


#### confidence
high — engine source; 800 on surf confirmed across 1.6 configs and map guidance.


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_phys.cpp

https://gist.github.com/Log1x/b99213403bcbef9b5f32e0b11c419f19


### item 3

#### name
sv_friction


#### engine_default
4 (ReHLDS sv_phys.cpp). Same in HL.


#### surf_typical
4 (unchanged). A few fun-surf cfgs float 5; not standard.


#### why_it_matters
Ground friction only (PM_Friction skipped while airborne, which is why surf speed persists). drop = friction * edgefriction? * max(speed, stopspeed) * dt. Also multiplied into ground/air accelspeed via player friction (pmove->friction, normally 1).


#### confidence
high — engine source; surf servers demonstrably leave it stock.


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_phys.cpp

https://gist.github.com/Log1x/b99213403bcbef9b5f32e0b11c419f19


### item 4

#### name
edgefriction


#### engine_default
2 (ReHLDS sv_user.cpp: cvar name is literally "edgefriction", variable sv_edgefriction). Same in HL.


#### surf_typical
2 (unchanged; rarely even mentioned in surf cfgs).


#### why_it_matters
When a trace 16 units ahead of the player and ~34-64 units down finds no floor (player near a ledge — i.e., the top of every surf ramp platform), ground friction is multiplied by this factor, decelerating players 2x faster near edges. Matters for start-platform launches.


#### confidence
high — engine source. Surf-typical = default is an inference from absence in all surveyed configs (medium there).


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_user.cpp


### item 5

#### name
sv_stopspeed


#### engine_default
100 (engine/HL default, ReHLDS sv_phys.cpp). CS 1.6 stock config sets 75, and 75 is the value CS servers actually run and log as default.


#### surf_typical
75 (CS convention, left untouched).


#### why_it_matters
Floor for the friction calculation: below this ground speed, friction drops speed by stopspeed*friction*dt per frame (fast stop). Ground-only; no effect mid-surf.


#### confidence
high — engine source for 100; CS=75 corroborated by multiple config/doc sources and server logs.


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_phys.cpp

http://gaming-blog.blogspot.com/2008/12/counter-strike-16-dedicated-server.html

https://gist.github.com/Log1x/b99213403bcbef9b5f32e0b11c419f19


### item 6

#### name
sv_maxspeed


#### engine_default
320 (ReHLDS sv_user.cpp). CS 1.6's game code briefly pushes 900 at startup/for observers (Valve closed halflife#1307 'Not a Bug'); servers then run 320 or 900 depending on config.


#### surf_typical
320 (most 1.6 surf configs) or 900 (some legacy configs). Largely cosmetic either way: effective per-player cap is min(pev->maxspeed_from_weapon, sv_maxspeed), and weapon speeds top out at 260, so any value >= 260 behaves identically. It does NOT cap actual velocity — only wishspeed and the bhop-cap reference.


#### why_it_matters
PM_CheckParameters: pmove->maxspeed = min(clientmaxspeed (weapon-based), sv_maxspeed); usercmd (forwardmove,sidemove,upmove) vector is rescaled to this magnitude. Also feeds the 1.2x bunnyhop cap. Surf velocity itself (1000+ ups on ramps) is only capped by sv_maxvelocity.


#### confidence
high — read directly from ReGameDLL pm_shared.cpp (PM_CheckParameters) and player.cpp (ResetMaxSpeed: observer 900, no-weapon 240, else weapon GetMaxSpeed).


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_user.cpp

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.cpp

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/dlls/player.cpp

https://github.com/ValveSoftware/halflife/issues/1307


### item 7

#### name
sv_maxvelocity


#### engine_default
2000 (ReHLDS sv_phys.cpp). Same in HL.


#### surf_typical
2000 (standard). This is the real speed ceiling on surf: clamped PER AXIS (each of x,y,z independently clamped to +/-2000), so diagonal horizontal speed can technically reach ~2828. A few speed-oriented servers raise it; combat-surf plugins (e.g. igorkfmoura Maxspeed) default their own surf cap to 2000 to match.


#### why_it_matters
PM_CheckVelocity clamps every velocity component each frame. Defines terminal falling speed and maximum attainable surf speed; an RL environment must clamp per-component, not by vector norm.


#### confidence
high — engine source.


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_phys.cpp

https://github.com/igorkfmoura/amxx-plugins


### item 8

#### name
sv_bounce


#### engine_default
1 (ReHLDS sv_phys.cpp).


#### surf_typical
1 (unchanged).


#### why_it_matters
Bounce multiplier for MOVETYPE_BOUNCE entities (grenades etc.), NOT player-vs-ramp collisions. Player velocity clipping on ramps uses PM_ClipVelocity with overbounce hardcoded to 1.0 in pm_shared — sv_bounce never enters player surf physics. Include it only if the env simulates grenades.


#### confidence
high — engine source + pm_shared reading.


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_phys.cpp

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.cpp


### item 9

#### name
sv_stepsize


#### engine_default
18 (ReHLDS sv_phys.cpp).


#### surf_typical
18 (unchanged).


#### why_it_matters
Max height PM_FlyMove/step logic will teleport the player up when walking; also used in the up-then-down ramp-hug trace. Affects whether small map lips are climbable and interacts with ramp-edge 'stuck' bugs surf mods patch.


#### confidence
high — engine source.


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_phys.cpp


### item 10

#### name
sys_ticrate


#### engine_default
100.0 (ReHLDS host.cpp: {"sys_ticrate", "100.0"}). HLDS-only (server framerate).


#### surf_typical
Set high — 1000 was the classic 'boosted 1000 FPS server' marketing standard for 1.6 movement/surf hosts (real achieved fps depends on OS timer granularity; many servers actually ran ~500-512 or ~1000). ReHLDS made high, stable server fps trivial. Values like sys_ticrate 10000 in rental-host cfgs just mean 'uncapped'.


#### why_it_matters
Caps the SERVER frame rate. Crucially, player movement in GoldSrc is integrated per-usercmd using the CLIENT's msec, so sys_ticrate does not directly change surf physics — it changes command processing latency, grenade/pusher physics, and fuser2-independent server-side timing. Don't model server tickrate as the physics dt; the client frame time is the physics dt.


#### confidence
high on default (source); medium on 'typical' (community convention, era-dependent).


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/host.cpp

https://www.jwchong.com/hl/game.html


### item 11

#### name
fps_max


#### engine_default
100.0 (host.cpp, FCVAR_ARCHIVE). Modern Steam builds cap effective fps at ~100.5 unless fps_override 1 (Valve re-enabled the 100 cap in the Feb 2013 update); ancient WON-era default was 72.


#### surf_typical
99.5 or 100 on clients — 100 fps is the community reference frame rate. 99.5 vs 100 depends on build: on current SteamPipe builds 99.5 yields a clean, stable 100fps/10ms frame; older builds want 100. Pre-2013, movement players ran fps_override 1 with higher fps for physics advantages.


#### why_it_matters
Client-side, but in GoldSrc the client frame IS the physics tick (usercmd.msec = frame time in integer ms; pmove->frametime = msec*0.001). fps determines dt and, at surf airaccelerate values, speed-gain-per-second. 100fps gives msec exactly 10 with no rounding loss (1000/100 integer, jwchong's 'no-slowdown' condition).


#### confidence
high on default and cap (source + Valve issue #452); high on 99.5/100 convention (speedrun/movement community docs).


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/host.cpp

https://github.com/ValveSoftware/halflife/issues/452

https://www.speedrun.com/hl1/forums/mog36#nqmld

https://www.jwchong.com/hl/game.html


### item 12

#### name
mp_footsteps


#### engine_default
1 (game DLL cvar).


#### surf_typical
1 (or 0 on some servers). NOT physics-relevant: it only gates footstep sound emission. The related physics fact is the reverse direction — pm_shared plays step/landing sounds based on speed thresholds (e.g. >=150 on jump) — sound never feeds back into movement.


#### why_it_matters
Include in the env only if modeling audio observations; zero effect on dynamics.


#### confidence
high.


#### sources
https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.cpp


### item 13

#### name
cl_forwardspeed / cl_sidespeed / cl_backspeed (client)


#### engine_default
400 (GoldSrc client default for all three; CS 1.6 keeps 400).


#### surf_typical
400 (players almost never change them; some old cfgs set 999 — no effect, see clamp).


#### why_it_matters
Determines the magnitude written into usercmd.forwardmove/sidemove when +forward/+moveleft etc. are held. Server-side PM_CheckParameters then rescales the (f,s,u) vector down to pmove->maxspeed (250 with knife), so any client value >= maxspeed is equivalent. For an RL action space: emitting forwardmove/sidemove = +/-400 (or anything >= 250) reproduces human input; intermediate analog values are legal and scale wishspeed.


#### confidence
medium-high — clamp logic read from ReGameDLL pm_shared.cpp (high); the 400 client default is community-documented rather than verified from client source (medium).


#### sources
https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.cpp

https://www.jwchong.com/hl/game.html


### item 14

#### name
sv_enablebunnyhopping / sv_autobunnyhopping / mp_jump_height / mp_stamina_restore_rate (ReGameDLL-only additions)


#### engine_default
0 / 0 / 45 / 0 (game.cpp of ReGameDLL_CS; these cvars DO NOT EXIST on vanilla Valve CS 1.6 servers — sv_enablebunnyhopping is otherwise a Source-engine cvar).


#### surf_typical
Combat-surf servers: usually left at defaults (surfing needs no jumps). Bhop/kreedz-flavored or 'fun' surf servers on ReHLDS+ReGameDLL: sv_enablebunnyhopping 1 (disables the 1.2x jump speed crop) and often sv_autobunnyhopping 1; mp_stamina_restore_rate > 0 or plugin fuser2 zeroing to kill stamina. Configs that list sv_enablebunnyhopping 1 on vanilla HLDS are cargo-cult (the cvar is ignored there).


#### why_it_matters
These are the modern, engine-level switches for the two CS 1.6 anti-bhop mechanisms (PreventMegaBunnyJumping cap and fuser2 stamina). An RL env should expose both as toggles.


#### confidence
high — read from ReGameDLL_CS game.cpp and pm_shared.cpp.


#### sources
https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/dlls/game.cpp

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.cpp


### item 15

#### name
sv_wateraccelerate / sv_wateramp / sv_spectator_maxspeed


#### engine_default
10 / 0 / 500 (ReHLDS sv_main.cpp).


#### surf_typical
Defaults (some surf maps have water hazards; water accel = accelerate*friction*dt*wishspeed with wishspeed*0.8 and a -60 sink drift).


#### why_it_matters
Only if maps include water. Spectator max speed relevant only to observer camera.


#### confidence
high — engine source.


#### sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_main.cpp


## maxspeed_mechanics_markdown
# How player max speed actually works in CS 1.6

Verified directly against ReGameDLL_CS source (a decompile-faithful reimplementation of the retail CS 1.6 game DLL).

## Three different "speeds" that must not be conflated

1. **`sv_maxspeed`** (engine movevar, default 320): a *ceiling* passed into player movement.
2. **`pev->maxspeed` -> `pmove->clientmaxspeed`** (per-player, set by the game DLL): the weapon-dependent speed. `CBasePlayer::ResetMaxSpeed()` (player.cpp): observer = **900**, freeze period = **1**, VIP = **227**, no weapon = **240**, otherwise `m_pActiveItem->GetMaxSpeed()`.
3. **Actual velocity**: unbounded by either of the above while airborne — only clamped per-axis by `sv_maxvelocity` (2000). This is why surfers reach 1000+ ups with a 250-speed knife.

## The effective cap (PM_CheckParameters, pm_shared.cpp)

```c
maxspeed = pmove->clientmaxspeed;
if (maxspeed != 0.0f)
    pmove->maxspeed = min(maxspeed, pmove->maxspeed /* = sv_maxspeed */);
spd = sqrt(f*f + s*s + u*u);   // usercmd forwardmove/sidemove/upmove
if (spd > pmove->maxspeed) { scale f,s,u by pmove->maxspeed/spd; }
```

So the usercmd *input vector* is rescaled to `min(weapon_speed, sv_maxspeed)` — with knife on a `sv_maxspeed 320` server that is **250**. This value then becomes `wishspeed` (capped again to `pmove->maxspeed` in PM_WalkMove/PM_AirMove) and is the reference for the bhop cap.

## Weapon speed table (ReGameDLL weapons.h constants, retail-accurate)

| Speed | Weapons |
|---|---|
| 260 | Scout (220 zoomed) — fastest gun |
| 250 | **Knife** (180 w/ shield), all pistols (Glock, USP, P228, Deagle, Five-Seven, Elites), C4, grenades (180 w/ shield), MP5, TMP, MAC-10, UMP-45 |
| 245 | P90 |
| 240 | AUG, XM1014, Galil, FAMAS |
| 235 | SG552 (200 zoomed) |
| 230 | M4A1, M3 |
| 221 | AK-47 |
| 220 | M249 |
| 210 | AWP (150 zoomed), G3SG1 (150 zoomed), SG550 (150 zoomed) |

**Knife = 250 confirmed** (`KNIFE_MAX_SPEED = 250.0f`). Surfers hold knife, so 250 is the canonical wishspeed reference for a surf RL env.

## cl_forwardspeed / cl_sidespeed

Client-side cvars (default 400) that set the magnitude the client writes into `usercmd.forwardmove/sidemove` while movement keys are held. Because PM_CheckParameters rescales the whole vector down to `pmove->maxspeed`, any client value >= the weapon speed is equivalent; 400 is what everyone runs. For the env: continuous actions `forwardmove, sidemove ∈ [-400, 400]` with the server-side rescale reproduces the real input pipeline exactly.

## Air acceleration detail that makes surf work (PM_AirAccelerate)

```c
wishspd = min(wishspeed, 30);                       // the famous 30-unit air cap
addspeed = wishspd - DotProduct(velocity, wishdir);
if (addspeed <= 0) return;
accelspeed = airaccelerate * wishspeed * frametime * pmove->friction;  // NOTE: uses UNCAPPED wishspeed
velocity += min(accelspeed, addspeed) * wishdir;
```

Air speed is only limited *along the current wish direction* (30 ups projection) — velocity perpendicular to wishdir is untouched, which is the entire basis of strafe/surf acceleration. With surf's `sv_airaccelerate 100`: accelspeed = 100 * 250 * 0.01 = 250 >> 30, so `addspeed` saturates **every frame** — per-frame air control is instantaneous and the practical turn/gain rate is set by frames per second.

## Jump (PM_Jump) — two anti-bhop mechanisms + a gravity quirk

- Jump-off vertical velocity is `sqrt(2 * 800 * 45) = 268.328` with **800 hardcoded** — changing `sv_gravity` does NOT change initial jump velocity (only the subsequent arc). ReGameDLL exposes `mp_jump_height` (default 45).
- **Bhop cap**: `PM_PreventMegaBunnyJumping()` runs on every jump in vanilla CS 1.6: if `|velocity| > 1.2 * pmove->maxspeed` (`BUNNYJUMP_MAX_SPEED_FACTOR = 1.2f`, so 300 with knife), velocity is scaled by `(1.2*maxspeed/|v|) * 0.8`. This fires only at jump time — it never limits speed while surfing, but it slashes speed if you jump at the end of a run (relevant for jump-to-platform finishes).
- **Stamina**: see stamina section.


## stamina_and_plugins_markdown
# Stamina (fuser2), the bhop cap, and what surf servers actually run

## Vanilla CS 1.6 stamina mechanics (verified in ReGameDLL pm_shared.cpp; matches jwchong's and KZ-Guide's write-ups)

- `pmove->fuser2` starts at 0; **every jump sets `fuser2 = 1315.789429`** (milliseconds-flavored units).
- Decay: `fuser2 -= cmd.msec` every player frame (so ~10/frame at 100fps; full recovery in ~1.316 s; decay rate is client-framerate-expressed but time-consistent since msec is real milliseconds).
- Penalty factor: `ratio = (100 - fuser2 * 0.001 * 19) * 0.01` = `1 - 0.00019 * fuser2`. At full stamina debt: **0.75**.
  - Applied to the new jump's vertical velocity in PM_Jump (`v_z = ratio * 268.328` -> min 201.2).
  - Applied to **horizontal velocity every frame while on the ground** in PM_WalkMove (`v_x,v_y *= ratio` — a brutal geometric slow that makes vanilla bhop chains speed-losing).
- Note stamina punishes only jumping + ground contact. **Pure surfing never touches it** — you don't jump on ramps and you're never onground, so a surf run is stamina-free in vanilla.

## Do surf servers alter it?

- **Combat/classic 1.6 surf servers (the majority)**: no. Standard surf configs (e.g. UltimateSurf) change `sv_airaccelerate` to 100, freezetime/buytime, respawn/semiclip behavior — and leave stamina and the bhop cap alone, because ramp movement never triggers them. "Standard surf movement settings" for 1.6 = vanilla movement + `sv_airaccelerate 100`.
- **Bhop-friendly / skill-surf / hybrid servers**: yes, via plugins. The canonical AMXX approach (AlliedModders "Bunny Hop Enabler", thread t=1262, and countless derivatives like bhop.amxx) is `entity_set_float(id, EV_FL_fuser2, 0.0)` each frame or on jump — zeroing stamina removes both the jump-height loss and the landing slow. Auto-bhop plugins additionally re-press jump for held +jump.
- **The 1.2x jump speed crop** (`PM_PreventMegaBunnyJumping`) cannot be cleanly removed by classic AMXX (it's compiled into pm_shared in both client prediction and game DLL); old plugins worked around it by re-setting velocity after the jump. On modern **ReHLDS + ReGameDLL_CS** servers it's a cvar: `sv_enablebunnyhopping 1` skips the crop, `sv_autobunnyhopping 1` gives held-space bhop, `mp_stamina_restore_rate` tunes stamina recovery, and ReAPI hooks (`PM_Jump`, `m_bMegaBunnyJumping` per-player flag, `m_flJumpHeight`) give per-player control. Modern 1.6 movement servers are near-universally ReHLDS/ReGameDLL.
- **Other movement-relevant surf plugins**: semiclip (teammates pass through each other — changes collision, standard on crowded surf servers), spawn/weapon managers, anti-speed-abuse caps for combat surf (e.g. igorkfmoura's Maxspeed: default air-jump cap 400, surf cap 2000, referenced to knife 250), and `fix_fps_speed`-type plugins that compensate the high-FPS acceleration advantage at `sv_airaccelerate 100`. Ramp-stuck fixes for high-FPS clients exist as metamod plugins (Surf-Mod-mm for ReGameDLL/ReHLDS).

## RL-env recommendation

Model stamina and the 1.2x crop as **toggleable, default-on** (vanilla combat surf), with the understanding that they only bind at jump/landing events; a pure ramp-surfing task can disable them with zero fidelity loss.


## fps_and_tickrate_markdown
# FPS, tickrate, and why 100 fps is the physics reference

## The core architectural fact

GoldSrc player physics is **client-frame-timed, not server-tick-timed**. Each client frame produces one `usercmd` whose `msec` field (an 8-bit integer, 0-255 ms) is the frame duration in whole milliseconds; the server (and client prediction) runs `PM_PlayerMove` with `pmove->frametime = cmd.msec * 0.001` (pm_shared.cpp, verbatim). So the client's framerate *is* the physics dt.

## msec rounding and the 100 fps sweet spot

Because msec is an integer, frame time is truncated to milliseconds: effective player dt = floor(1000/fps)/1000. jwchong formalizes the "slowdown factor" eta = f_g/1000 * floor(1000/f_g): **eta = 1 exactly iff 1000/fps is an integer** (100, 125, 200, 250, 500, 1000...). At 100 fps, msec = 10 exactly — zero truncation loss. Non-divisor framerates lose real speed (extreme example: 501 fps -> eta ~ 0.5). This, plus the engine's 100 fps cap, makes **100 fps / dt = 0.01 s the canonical reference for CS 1.6 physics**, and the correct fixed dt for an RL environment.

## fps_max conventions

- Modern Steam CS 1.6: default `fps_max 100`, hard-capped at ~100.5 unless `fps_override 1` (Valve re-enabled the cap in the Feb 2013 update; halflife issue #452 documents the 100.5 behavior).
- **99.5 vs 100**: movement communities run `fps_max 99.5` on current SteamPipe builds (yields a stable true 100 fps) and `fps_max 100` on older clients — build-dependent, per speedrun.com/GoldSrc Package guidance. Either way the intended physics rate is 100 fps.
- Higher fps changes physics: more frames = more air-accel applications per second (decisive at surf `sv_airaccelerate 100` where per-frame gain saturates — high-fps clients out-accelerate 100fps clients through turns, hence server-side fixer plugins), different jump/duck frame quantization, and pre-2013 1000fps clients had famous exploits. jwchong also notes builds >= 6027 changed the old rounding behavior.

## Server side: sys_ticrate

- `sys_ticrate` default **100.0** (ReHLDS host.cpp); it caps HLDS server fps. The 2000s-era "1000 FPS boosted server" fashion (sys_ticrate 1000+, kernel timer tweaks) mattered for command processing latency, `pev` physics (grenades, pushers, trains/platforms on surf maps) and timing granularity — but **not** for the player-movement integrator, which uses client msec. ReHLDS provides stable high fps without OS hacks.
- Practical surf-host convention: sys_ticrate 1000 (or an uncapped large value) on dedicated movement servers; plain 100 is also common and does not change ramp physics.

## RL environment takeaway

Fixed dt = 10 ms reproduces the reference client. If you want to model the real player population, dt per-agent-frame with integer-millisecond quantization (msec = floor(1000/fps)) is the faithful formulation; per-frame air-accel saturation at aa=100 means results are fps-sensitive, so pin 100 fps for canonical evaluation.


## disagreements_markdown
# Where sources disagree

1. **Surf sv_airaccelerate for CS 1.6: 100 vs 300 vs a difficulty ladder.** CS 1.6-native artifacts (UltimateSurf plugin cfg: `sv_airaccelerate "100" // This is a must in all surf servers`; GameBanana 1.6 surf map pages; the fps-fix plugin "works good on sv_airaccelerate 100") converge on **100**. Tobys CS claims 1.6 surf uses `sv_airaccelerate 300` plus a nonexistent cvar `sv_airmove 10`, but the author explicitly writes "I'm not 100% sure" — I weight this near zero (sv_airmove is not a GoldSrc cvar). The 100/150/200/400/800/1000 "difficulty tiers" repeated by csdownload.net and CS:GO-era guides describe the Source/CS:GO surf scene; on 1.6 the observed norm is 100, with higher values only on easy/fun servers.

2. **Engine vs game defaults (sv_accelerate, sv_stopspeed).** The raw engine defaults are `sv_accelerate 10`, `sv_stopspeed 100` (ReHLDS source = HL defaults). Nearly all CS-focused documentation calls **5 and 75** "the defaults" because CS 1.6's stock config/game sets them. Both statements are true at different layers; an RL env replicating CS 1.6 should use 5 / 75. csdownload.net's claim of defaults "sv_accelerate 5.6, sv_airaccelerate 12" matches no engine source (5.6 is a CS:GO-era number) — treat as wrong for 1.6.

3. **sv_maxspeed 320 vs 900.** Server logs show 900 transiently at startup (game DLL/observer logic) before configs set 320; Valve closed halflife#1307 as Not-a-Bug. Community configs split between 320 and 900; behaviorally identical for armed players since weapon speed (<= 260) is the binding min().

4. **fps_max 99.5 vs 100.** Speedrun/movement sources agree the target is a true 100 fps but disagree on the cvar value because it depends on client build (SteamPipe: 99.5; older builds: 100). Not a substantive physics disagreement.

5. **sv_enablebunnyhopping on 1.6.** Several config pastes (e.g. the Log1x 1.6 cfg) include it; it is a Source-engine cvar that only exists on 1.6 via ReGameDLL_CS (default 0). On vanilla HLDS it's silently ignored — configs containing it are not evidence the cap was off.

6. **Stamina decay framing.** KZ-Guide/jwchong describe identical math but different units framing ("decreases by frame ms" vs "sigma - tau"); both equal fuser2 -= msec per player frame. No real conflict; noted because secondary sources sometimes misstate it as fps-dependent in *rate* (it is fps-dependent only through millisecond truncation).


## sources
https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_main.cpp (engine cvar defaults: sv_airaccelerate 10, sv_wateraccelerate 10, sv_wateramp 0)

https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_phys.cpp (sv_gravity 800, sv_maxvelocity 2000, sv_bounce 1, sv_stepsize 18, sv_friction 4, sv_stopspeed 100)

https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/sv_user.cpp (edgefriction 2, sv_maxspeed 320, sv_accelerate 10)

https://github.com/rehlds/ReHLDS/blob/master/rehlds/engine/host.cpp (sys_ticrate 100.0, fps_max 100.0)

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.cpp (PM_AirAccelerate 30-unit cap, PM_CheckParameters clamp, PM_PreventMegaBunnyJumping, fuser2=1315.789429, jump vz sqrt(2*800*45), frametime=msec*0.001)

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/pm_shared/pm_shared.h (BUNNYJUMP_MAX_SPEED_FACTOR 1.2f)

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/dlls/game.cpp (sv_enablebunnyhopping 0, sv_autobunnyhopping 0, mp_jump_height 45, mp_stamina_restore_rate 0)

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/dlls/player.cpp (ResetMaxSpeed: observer 900, VIP 227, no-weapon 240, weapon GetMaxSpeed)

https://github.com/rehlds/ReGameDLL_CS/blob/master/regamedll/dlls/weapons.h (full weapon MAX_SPEED table: knife 250, scout 260, AWP 210/150, AK 221, etc.)

https://www.jwchong.com/hl/game.html (Half-Life Physics Reference: msec serialization, slowdown factor, no-slowdown framerates)

https://www.jwchong.com/posts/counter-strike-stamina/ (stamina math: 1315.789, 1-0.00019*sigma factor, 268.3282 jump vz)

https://kzguide.gitlab.io/techniques/stamina/ (fuser2 mechanics, 100fps decay)

https://github.com/tonykaram1993/UltimateSurf/blob/master/configs/UltimateSurf.cfg (1.6 surf plugin: sv_airaccelerate 100 'a must in all surf servers', semiclip, respawn)

https://github.com/igorkfmoura/amxx-plugins (fix_fps_speed for aa=100 servers, Maxspeed combat-surf caps, fuser2-based multijump)

https://forums.alliedmods.net/archive/index.php/t-1262.html (Bunny Hop Enabler: EV_FL_fuser2 = 0 technique)

https://github.com/mEldevlp/Surf-Mod-mm (ReGameDLL/ReHLDS surf metamod plugin, ramp-stuck fix)

https://github.com/ValveSoftware/halflife/issues/1307 (sv_maxspeed 320 vs 900, closed Not-a-Bug)

https://github.com/ValveSoftware/halflife/issues/452 (100.5 fps cap with fps_override 0)

https://www.speedrun.com/hl1/forums/mog36#nqmld (fps_max 99.5 vs 100 per build)

https://gist.github.com/Log1x/b99213403bcbef9b5f32e0b11c419f19 (1.6 movement server cfg: aa=100, accel 5, stopspeed 75, maxspeed 320, friction 4, edgefriction 2)

https://www.tobyscs.com/cs-surf-settings/ (dissenting: aa=300 for 1.6, author self-flagged unsure)

https://csdownload.net/cs-surf-settings-csgo-source-1-6/ (surf difficulty ladder 100-1000, Source-era framing)

https://gamebanana.com/mods/cats/5501 (CS 1.6 surf maps recommending sv_airaccelerate 100)

https://www.fiz-x.com/how-to-setup-cs-1-6-in-2026-fps-ping-mouse-and-best-console-commands/ (2013 fps cap re-enable, fps_override)

http://gaming-blog.blogspot.com/2008/12/counter-strike-16-dedicated-server.html (CS 1.6 cvar list: sv_accelerate 5, sv_stopspeed 75)

