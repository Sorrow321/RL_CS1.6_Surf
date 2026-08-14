# 08 — Ladders & Water Implementation Plan

Vanilla transcription targets in `third_party/cs_pm_shared.cpp` (fetch via
`tools/fetch_third_party.ps1`). Est. ~3–4 h total. Ladders have **no ABI
impact**; water grows `SurfState` again. Both behind flags, default ON after
gates, matching the duck rollout pattern (docs/07 worked first-run — reuse it).

## Part A — Water (do first; ladders reuse nothing from it)

### A1. Exact behavior (source refs)

- **Pipeline** (`:3258–3307`, we have it verbatim): in MOVETYPE_WALK, after
  gravity: if `waterjumptime != 0` → `PM_WaterJump(); PM_FlyMove();
  PM_CheckWater(); return;` (early-out replaces the whole walk branch).
  If `waterlevel >= 2`: `PM_CheckWaterJump()` only at exactly 2; falling
  (`velocity[2] < 0`) clears waterjumptime; IN_JUMP → `PM_Jump()` (swim pop —
  already ported: 100/80/50 by watertype); `PM_WaterMove()`; subtract
  basevelocity; `PM_CategorizePosition()`. Else the existing dry branch.
- **PM_WaterMove** (`:1376–1489`): wishvel from full 3D forward/right ×
  fmove/smove (pitch MATTERS underwater — unlike air/ground); idle sink
  `wishvel[2] -= 60`; jump… `upmove` adds up (our upmove is 0 — swimming up is
  via IN_JUMP/pitch); wishspeed clamp to maxspeed then `*= 0.8`; water
  friction `speed - friction*pfriction*frametime*speed` (uses sv_friction —
  no waterfriction cvar in CS); accelerate with `movevars->accelerate` toward
  `wishvel`. **Transcribe the move-execution tail `:1463–1489`** (dest trace +
  step-up attempt) — the one region not yet read; do it at implementation.
- **PM_CheckWaterJump** (`:2817–2890`): only at waterlevel==2, not while
  `velocity[2] < -180`, not backing up; probe from `origin.z + WJ_HEIGHT(=8)`
  24u along flat-forward with the **point hull** (usehull 2, saved/restored);
  near-vertical wall (`|normal.z| < 0.1f`) AND clear at
  `origin.z + player_maxs[hull].z - 8` → `waterjumptime = 2000; velocity[2] =
  225; movedir = -50 * wall_normal; oldbuttons |= IN_JUMP`.
- **PM_WaterJump** (`:2379–2401`): clamp waterjumptime ≤ 10000; decay by msec;
  clear when `< 0 || !waterlevel`; every tick force `velocity[0..1] =
  movedir[0..1]` (the ledge-hop push).
- **Re-enable the waterjumptime guards we stubbed** (currently dead because
  waterjumptime ≡ 0): early-returns in AddCorrectGravity, FixupGravityVelocity,
  Friction, Accelerate, AirAccelerate; PM_Jump's `if (waterjumptime) { decay;
  return; }` branch (replace the "not modeled" comment).
- Water currents (CONTENTS_CURRENT_*): still skipped — no surf map uses them.

### A2. ABI + config (one commit, all mirrors)

- `SurfState` += `float waterjumptime; float movedir[3];` → 88 → **104 B**.
  Mirrors: core.py `_fields_` + `STATE_DTYPE` + binding layout test.
- `SurfEnvConfig` += `int32_t water_fail;` (default **1**: keep the current
  episode-fail on waterlevel ≥ 2 — on surf maps water still means failed run)
  → 96 → **100 B**. `water_fail = 0` ⇒ swimming runs (play client uses this).
  Mirrors: core.py + `default_config` + layout test + bench.c `defcfg`.
- `SurfPhys` unchanged (no new cvar; gate implementation on
  `cfg.water_fail == 0`? NO — physics must swim even when water_fail=1 for the
  ticks before the env kills the episode; implement unconditionally, keep
  `water_fail` as the env-level termination policy only).

### A3. The duck+jump surface-skim tech = acceptance test, not code

The tech must EMERGE; nothing special is coded. Mechanism to verify:
waterlevel is computed with the CURRENT hull (our `pm_check_water` is already
usehull-parameterized); ducking in water completes instantly (`onground == -1`
finish condition) and shrinks the hull so the center/feet probes read air near
the surface → waterlevel drops below 2 → the DRY branch runs and a real jump
(268) or the prior swim-pop arc carries you over the surface; falling back in
repeats. **W4**: on a flat deep-water area (scan `func_water` volumes for one
with ≥ 200u open water ahead), script `duck+jump` alternation vs plain
hold-forward swimming for the same tick budget — assert the skim run covers
> 1.3× the horizontal distance (that speed edge is why the tech exists).

### A4. Water tests

- **W1 sink**: idle in deep water → settles into slow descent (−60 drift
  capped by water friction), never accelerates unbounded.
- **W2 swim pop**: IN_JUMP at waterlevel ≥ 2 → `velocity[2] == 100` exactly
  (CONTENTS_WATER), repeatable every tick (no edge trigger under water).
- **W3 waterjump**: place at water surface facing a vertical wall ≤ 24u ahead
  (scan for a func_water edge abutting world geometry) → waterjumptime 2000,
  vz 225, xy locked to movedir while active; SKIP-print if no spot found.
- **W4 skim** (§A3).
- **W5 policy**: `water_fail=1` env still terminates on waterlevel ≥ 2
  (existing behavior byte-identical); `water_fail=0` env survives and swims.
- **Regression**: full suite + stationarity with water code compiled in but
  player dry — bit-identical (T15 pattern).

## Part B — Ladders

### B1. Exact behavior (source refs)

- **Detection — PM_Ladder** (`:2344–2375`): iterate ladder entities (ours:
  `map.contents[]` with `skin == CONTENTS_LADDER` — surf_ski_2 has 6), brush
  model only; take the model's hull for the player's CURRENT usehull
  (`PM_HullForBsp` — same select as `trigger_contains`), offset = clip_mins −
  player_mins + ent origin, and test `HullPointContents(origin − offset) !=
  CONTENTS_EMPTY` → on ladder. This is `trigger_contains` with an EMPTY-test
  instead of SOLID — generalize that helper (`model_contains(m, model, origin,
  usehull, mode)`).
- **PM_LadderMove** (`:2218–2342`): sets `movetype = MOVETYPE_FLY`, gravity 0
  (our FLY branch simply never calls gravity — no per-player gravity field
  needed); `ladderCenter = (model mins + maxs)/2` (+ ent origin);
  `onFloor` = point contents SOLID at `origin.z + player_mins[usehull].z − 1`;
  `PM_TraceModel(pLadder, origin, ladderCenter)` — a trace against ONLY that
  model (our `clip_to_model`; **read engine `PM_TraceModel` in
  `engine_pmovetst.cpp` first to confirm which hull it uses**); if it hits:
  climb speed `MAX_CLIMB_SPEED` (**verify define — HLSDK value 200**), capped
  at maxspeed, ×0.333 when FL_DUCKING; direction from BUTTON BITS
  (IN_FORWARD/IN_BACK/IN_MOVELEFT/IN_MOVERIGHT — see B2) scaled into
  forward/right; **IN_JUMP → detach: `movetype = WALK; velocity = 270 *
  ladder_normal`** (`:2293`); else the lateral/normal decomposition
  (`:2300–2334`, transcribe verbatim incl. the un-normalized cross products
  and the `onFloor && normal > 0` push-away); no input → `velocity = 0`
  (motionless hang).
- **Pipeline**: `pLadder` found after CategorizePosition/Duck →
  `PM_LadderMove(pLadder)`; then MOVETYPE_FLY branch (`:3230–3256`):
  `PM_CheckWater(); if (IN_JUMP) { if (!pLadder) PM_Jump(); } else clear jump
  latch; velocity += basevelocity; PM_FlyMove(); velocity -= basevelocity;`.
  Our pm_tick grows a `movetype` LOCAL (WALK/FLY per tick — no persistence;
  the engine resets non-WALK to WALK next tick when off ladder `:3160s`).

### B2. Input plumbing (the one real integration task)

`PM_LadderMove` reads direction from usercmd BUTTON bits, not fmove/smove
signs. The client sets IN_FORWARD when +forward is held, etc. Add to
`pm_tick`/`surf_play_step`/env action decode: derive
`IN_FORWARD (1<<3) / IN_BACK (1<<4) / IN_MOVELEFT (1<<9) / IN_MOVERIGHT (1<<10)`
from the signs of fmove/smove before the tick (HLSDK `in_buttons.h`
convention — **verify the bit values**; add `in_buttons.h` to
`fetch_third_party.ps1`). Zero ABI impact; play client and env both get ladder
control for free through existing actions.

### B3. Ladder tests

- **L1 hang**: place inside a ladder volume (entity AABBs known from the map),
  no input → velocity 0, zero gravity, position frozen.
- **L2 climb**: face the ladder, hold forward → velocity magnitude ≈ 200
  directed up the ladder plane; reaches the top and exits.
- **L3 detach**: IN_JUMP on ladder → speed 270 along the ladder normal,
  gravity resumes next tick.
- **L4 duck-climb**: ×0.333 speed.
- **Regression**: full suite dry/off-ladder bit-identical; the ramp-slide test
  MUST stay identical (ladder volumes on surf_ski_2 sit near the jail, not the
  ramps, but the detection scan now runs every tick — assert perf impact < 5%
  in bench).

## Rollout order & gates

1. Water guards + state/ABI + WaterMove/CheckWaterJump/WaterJump + pipeline →
   W1/W2/W5 + regression.
2. W3/W4 (map-scan based, SKIP-tolerant).
3. Button-bit plumbing (B2, verify in_buttons.h) → ladder detection + move →
   L1–L4 + bench check.
4. play.py: `water_fail=0`, "WATER"/"LADDER" HUD tags; feel pass — swim, the
   skim tech (you know how it should feel), ladder climb in the jail area.
5. Defaults + docs sync (01 §8 table, 03 config sketch, README), commit.

## Risks / open items

- `PM_TraceModel` semantics unverified (read before B1 coding).
- `MAX_CLIMB_SPEED` and IN_* bit values need one-line upstream verification.
- WaterMove tail `:1463–1489` unread (transcribe at implementation).
- `movedir` persistence: TeleportTouch zeroes velocity/basevelocity but NOT
  movedir/waterjumptime in-engine — copy that (don't zero on teleport).
- Env obs carries no water/ladder signal — fine while `water_fail=1` and
  ladders are off-route; note as future obs extension if a ladder map is ever
  trained.
