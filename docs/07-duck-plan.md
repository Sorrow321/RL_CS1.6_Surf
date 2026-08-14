# 07 — Ducking Implementation Plan

Port `PM_Duck` / `PM_UnDuck` / `PM_FixPlayerCrouchStuck` from
`third_party/cs_pm_shared.cpp:2001–2214` (**vanilla path** — everything under
`REGAMEDLL_FIXES/ADD/API` excluded, same policy as the rest of `pm.c`).
Estimated effort: ~2–3 h including tests. All source line refs below are into
the fetched `cs_pm_shared.cpp`.

## 1. Exact behavior to port (vanilla literals)

Constants (`cs_pm_shared.h`): `PLAYER_DUCKING_MULTIPLIER 0.333`,
`TIME_TO_DUCK 0.4`, `STUCK_MOVEUP 1`, `PM_VEC_DUCK_VIEW 12`, `PM_VEC_VIEW 17`,
hull shift = 18 (vanilla uses the literal `18.0`, not the mins difference).
`HalfHumanHeight` = 36 (loop bound in FixPlayerCrouchStuck; defined in an
upstream header we don't fetch — HLSDK uses the literal 36 in the same loop;
**verify against upstream when implementing**).

**PM_Duck (:2105)** — called every tick from the pipeline:
1. `buttonsChanged = oldbuttons ^ cmd.buttons; nButtonPressed = buttonsChanged & cmd.buttons`.
   Latch: `IN_DUCK` set/cleared in `oldbuttons` from this cmd (:2113–2120).
   NOTE: compute `nButtonPressed` BEFORE the latch mutates oldbuttons.
2. Gate (:2136): `if (dead || (!(buttons & IN_DUCK) && !bInDuck && !FL_DUCKING)) return;`
3. Cmd scaling (:2141–2150): `real_t mult = 0.333; fmove *= mult; smove *= mult; upmove *= mult;`
   — applied whenever the gate passes (pressed OR transitioning OR ducked; CS
   differs from HL here). Mutates the working cmd values used by Walk/AirMove.
4. If `IN_DUCK` held (:2152):
   - press edge && !FL_DUCKING → `flDuckTime = 1000; bInDuck = TRUE` (:2154).
   - if `bInDuck`: **finish** when `(flDuckTime/1000.0) <= (1.0 - 0.4)` (400 ms
     elapsed) **or `onground == -1` (instant in air)** (:2164):
     `usehull = 1; view_ofs_z = 12; flags |= FL_DUCKING; bInDuck = FALSE;`
     and if on ground: `origin[2] -= 18.0;` then `FixPlayerCrouchStuck(+1)`,
     then `PM_CategorizePosition()` (:2172–2187). In air: NO origin shift
     (feet effectively rise 18 — the duck-jump).
     Else (mid-transition): view-height spline animation only (:2191–2205) —
     see §4 for what we do with it.
5. Else (duck released) → `PM_UnDuck()` (:2210) — **unconditionally**, even if
   still mid-transition. See the quirk below.

**PM_UnDuck (:2033)** — vanilla:
1. `newOrigin = origin; if (onground != -1) newOrigin[2] += 18.0;`
2. Trace `newOrigin→newOrigin` with the CURRENT (duck) hull; if `startsolid` → do nothing.
3. `usehull = 0`; re-trace with the standing hull; if `startsolid` →
   `usehull = 1; return` (blocked overhead: stay ducked).
4. Clear `FL_DUCKING`/`bInDuck`; `view_ofs_z = 17; flDuckTime = 0;`
   (`flTimeStepSound -= 100` — not modeled, skip); `origin = newOrigin;
   PM_CategorizePosition()`.

**PM_FixPlayerCrouchStuck(+1) (:2001):** if hull position free → return; else
up to 36 iterations of `origin[2] += 1` until free; if never free, restore the
original origin.

**⚠ Authentic vanilla quirk to KEEP:** releasing duck mid-transition
(`bInDuck && !FL_DUCKING`, origin never lowered) still runs UnDuck's
`+18` on-ground offset → the player pops up 18u if there's headroom. This is
real CS 1.6 behavior (the "duck-peek"); `REGAMEDLL_FIXES` patches it out, we
don't. Cover it with a test, don't "fix" it.

**PM_Jump guard** (already stubbed in pm.c): `if (bInDuck && FL_DUCKING) return;`

**PM_ReduceTimers**: add `flDuckTime -= msec (floor 0)` next to the existing
fuser2 decay (source :3086–3094).

**Engine wrapper duties** (docs/01 §2): each cmd starts with
`usehull = FL_DUCKING ? 1 : 0` (pm-internal switches persist only via the
flag), and writeback truncates `flDuckTime = (int)flDuckTime`. Mirror both:
set `ctx.usehull` from `st->ducked` at tick start; truncate `st->duck_time`
to whole ms at tick end.

## 2. State / ABI change (breaking — coordinate all mirrors)

`SurfState` gains two fields (append before `stuck_ticks` is NOT allowed —
append at END to keep offsets of existing fields stable... **decision: append
at end**): `int32_t induck; float duck_time;` → sizeof 80 → 88.
Update in the same commit:
- `python/surfgym/core.py`: `SurfState._fields_`, `STATE_DTYPE` (order must
  match), size assert self-updates.
- `python/tests/test_binding.py`: expected sizeof (80 → 88) in the layout test.
- `surfcore.h` header comment + the "SurfState will grow" note (this is that
  growth, minus waterjump which stays out of scope).
- `play.py`/`record.py` use fields by name — no change needed; recheck
  `record.py`'s tick row (add ducked? viewer row format change — OPTIONAL,
  skip in v1 to avoid breaking existing trajectories).

## 3. Code changes by file (implementation order)

1. **`src/sim.h`**: nothing (PmPersist unchanged).
2. **`src/pm.c`** (~120 lines):
   a. `PmCtx` gains `induck`, `duck_time` working copies? NO — mutate
      `st->induck`/`st->duck_time` directly (like fuser2). `ctx.usehull` set
      from `st->ducked` at tick start; `ctx.view_ofs_z` = ducked ? 12 : 17.
   b. `pm_reduce_timers`: duck_time decay.
   c. `pm_spline_fraction` (float, real_t square — :1981), needed only if we
      animate view_ofs; see §4.
   d. `pm_fix_player_crouch_stuck(c, +1)` using `test_player_position_free`.
   e. `pm_unduck(c)`, `pm_duck(c)` — transcribe per §1; they mutate
      `c->fmove/smove/upmove` (scaling), `st->ducked` (FL_DUCKING),
      `st->induck`, `st->duck_time`, `c->usehull`, `st->origin`.
      All behind `if (c->ph->enable_duck)` at the call site.
   f. Pipeline: insert `if (ph->enable_duck) pm_duck(&c);` right after the
      first `pm_categorize_position(&c)` (CS order: Duck before step-sound,
      before the gravity/jump block — source :3169 area).
   g. `pm_jump`: replace the skip comment with the real guard
      `if (st->induck && st->ducked) return;`.
   h. Duck-button edge state: PM_Duck reads `oldbuttons ^ buttons` — our
      `st->oldbuttons` currently only tracks IN_JUMP semantics; the engine
      writeback (`oldbuttons = cmd.buttons`) happens per-cmd in SV_RunCmd.
      CHECK: our pm_tick leaves oldbuttons managed inside Jump/Duck latches
      only — after porting Duck's latch (:2113), IN_DUCK edge detection works
      exactly like source. Keep the jump path's existing handling untouched.
3. **`src/env.c`**:
   a. `apply_triggers`: `int uh = st->ducked ? 1 : 0;` — use in
      `trigger_contains(m, t, uh, ...)` AND the AABB precheck extents
      (`g_player_mins[uh]`/`maxs[uh]`).
   b. Default `SurfPhys.enable_duck` stays 0 until Gate T passes, then flip
      docs + `_PHYS_DEFAULTS` + bench defaults to 1.
4. **`python/surfgym/core.py`**: §2 mirrors; `default_config` enable_duck.
5. **`python/play.py`**: Ctrl already sends IN_DUCK. Add: eye height from a
   client-side 400 ms spline (17→12) matching PM_SplineFraction for feel
   (cosmetic only — see §4); third-person hull box 36 tall when
   `st.ducked`; HUD "DUCK" tag next to GROUND/air.
6. **`viewer/`**: skip in v1 (traj format unchanged).

## 4. Accepted deviations (document in pm.c comments)

- **view_ofs during the 400 ms transition**: engine animates it and
  `PM_CheckWater` uses it for the eye-underwater (waterlevel 3) test. We fail
  episodes at waterlevel ≥ 2, so waterlevel 3 is unreachable-relevant; physics
  uses binary 17/12. The play client animates cosmetically.
- `flTimeStepSound -= 100` in UnDuck: sounds not modeled, skipped.

## 5. Tests (extend `tests/test_physics.c`; rest z references: standing rest
41.031, floor surface 5.031 under spawn 0)

- **T9 ground duck**: hold duck 50 ticks on flat ground → `ducked == 1` only
  after ~40 ticks (400 ms), origin z settles ≈ **23.03** (surface + 18),
  onground stays true throughout, never airborne.
- **T10 duck speed**: duck held + full forward on ground → steady h-speed
  ≈ 250·0.333 = **83.3** (assert 75–90 band; exact recurrence optional).
- **T11 instant air duck**: place airborne, press duck → `ducked == 1` after
  ONE tick, origin unchanged by the duck itself.
- **T12 duck-jump**: jump, press duck at apex → hull switches (verify via a
  zero-length standing-hull trace at origin being MORE constrained than duck
  hull), unduck in air → origin unchanged (feet drop back).
- **T13 unduck blocked**: scan the map for a spot with clearance in [40, 70]u
  (duck fits, standing doesn't — down-trace + up-trace per column); if found:
  walk there ducked, release duck → stays `ducked == 1`; else print SKIP.
- **T14 duck-peek quirk**: on flat ground press duck for ~10 ticks (< 400 ms,
  still `induck`), release → origin z jumps to ≈ 41.03+18 minus snap...
  assert origin z INCREASED ~18 that tick then CategorizePosition/gravity
  settle back to 41.03 over subsequent ticks (assert transient max > 50).
- **T15 stationarity regression**: T6 (AFK 1000 ticks) re-run with
  enable_duck=1 and no duck input — must stay bit-identical (duck gate
  returns immediately when nothing pressed; proves the flag is free when
  unused).
- Re-run ALL existing tests with enable_duck=1 to prove no interference, and
  once with =0 to prove the old path is untouched.

## 6. Verification order (gates)

1. ABI: `test_binding.py` green after struct growth (both platforms' sizeof).
2. T15 + full old suite with duck enabled-but-unused → bit-identical physics.
3. T9–T14.
4. Play-feel pass (the human gate): Ctrl feel vs real CS — duck-jump onto the
   spawn-room ledge, crouch-surf a ramp (hull 3 rides ramps at different
   clearance), duck-peek pop.
5. Flip `enable_duck` default to 1 everywhere + README/docs sync
   (docs/01 §8 table, docs/03 SurfPhys comment, NOTICE untouched).

## 7. Risk notes

- The `-18` on-ground shift + `FixPlayerCrouchStuck` + snap-down interact;
  transcribe ordering exactly (shift → fix → categorize) or ramp-edge ducks
  will stick/pop.
- Duck hull traces use BSP hull 3 — already loaded and mapped
  (`bsphull_for_usehull(1) == 3`), zero collision-layer work needed.
- `PM_CheckWater`'s probe points use `player_mins[usehull]` — already
  parameterized, verify only.
- Curriculum spawns/reset: memset covers the new fields; teleports zero
  nothing duck-related in TeleportTouch (engine doesn't either — a ducked
  player stays ducked through a teleport).
