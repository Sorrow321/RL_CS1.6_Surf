# 02 — BSP v30 Loading & Hull Collision

Everything here is read from ReHLDS engine source (`world.cpp`, `pmovetst.cpp`, `model.cpp`, `bspfile.h`) and verified against a real map: `maps/surf_ski_2.bsp` parses as v30 with 7379 clipnodes and 50 models (worldspawn + 49 `*N` brush models). We port the **pmove** trace path (`PM_RecursiveHullCheck`) — that is what player movement actually uses.

## 1. File format (little-endian)

Header: `int version` (must be **30**) + 15 × `{ int fileofs; int filelen; }` = 124 bytes.

| idx | lump | elem | size | needed for |
|---|---|---|---|---|
| 0 | ENTITIES | ASCII text | — | spawns/triggers |
| 1 | PLANES | `{float normal[3]; float dist; int type;}` | 20 | collision + mesh |
| 2 | TEXTURES | miptex directory | var | mesh (names only) |
| 3 | VERTEXES | `float[3]` | 12 | mesh |
| 5 | NODES | `{int planenum; short children[2]; short mins[3],maxs[3]; ushort firstface,numfaces;}` | 24 | **hull 0 build** |
| 6 | TEXINFO | `{float vecs[2][4]; int miptex; int flags;}` | 40 | mesh (skip-texture filter) |
| 7 | FACES | `{short planenum, side; int firstedge; short numedges, texinfo; byte styles[4]; int lightofs;}` | 20 | mesh |
| 9 | CLIPNODES | `{int planenum; short children[2];}` | 8 | **collision** |
| 10 | LEAFS | `{int contents; int visofs; short mins[3],maxs[3]; ushort firstmarksurface,nummarksurfaces; byte ambient[4];}` | 28 | hull 0 / contents |
| 12 | EDGES | `ushort v[2]` | 4 | mesh |
| 13 | SURFEDGES | `int32` (signed) | 4 | mesh |
| 14 | MODELS | `{float mins[3],maxs[3]; float origin[3]; int headnode[4]; int visleafs; int firstface, numfaces;}` | 64 | both |

Node/clipnode `children[i]`: `>= 0` → child node index; `< 0` → for NODES: leaf index `~child`; for CLIPNODES: **the contents value itself** (`CONTENTS_*`).

**Face → polygon:** for `i` in `[firstedge, firstedge+numedges)`: `se = surfedges[i]`; vertex = `se >= 0 ? edges[se].v[0] : edges[-se].v[1]`. Winding-ordered convex polygon → fan-triangulate. Normal = `planes[planenum].normal`, negated if `side != 0`.

## 2. Hulls

Clipnodes contain **compiler-pre-expanded (Minkowski) geometry per hull** — no runtime brush expansion, which is why traces are point-vs-tree and cheap. All hulls share the PLANES array. Hulls 1–3 of model *m* are subtrees of the one clipnode array rooted at `dmodel.headnode[1..3]`. **Hull 0 (point) is built at load** (`Mod_MakeHull0`): mirror the render NODES into clipnode form (`children[j] = child is leaf ? leaf->contents : child index`) — needed for point contents (water/solid) and trigger tests.

| BSP hull | clip_mins → clip_maxs | pmove `usehull` | use |
|---|---|---|---|
| 0 | point | 2 | point contents + SOLID_NOT contents-entity tests (NOT the player trigger test — that uses the trigger model's hull 1/3 per the player's box) |
| 1 | (−16,−16,−36)→(16,16,36) | **0 standing** (32×32×72) | player |
| 2 | (−32,−32,−32)→(32,32,32) | 3 large | unused for surf |
| 3 | (−16,−16,−18)→(16,16,18) | **1 ducked** (32×32×36) | player crouched |

**Player origin is the hull center** (feet = z−36 standing / z−18 ducked; eye = z+17, ducked z+12). Engine `player_mins/maxs[usehull]` exactly equal the map hulls' clip_mins/maxs, so for the world the trace offset is zero: trace the origin directly through the pre-expanded tree. General offset rule (`PM_HullForBsp`): `offset = hull->clip_mins − player_mins[usehull] + entity->origin`; `start_l = start − offset`. (Rotated brush entities need angle transforms — solids only rotate if door/button logic is implemented, which we don't; assert angles == 0 on solids and see the jail-door caveat in §4.)

Trace-side hull selection for arbitrary boxes (`SV_HullForBsp`, needed for the trigger test): `size = maxs−mins`; `size[0] <= 8` → hull 0; `size[0] <= 36` → (`size[2] <= 36` ? hull 3 : hull 1); else hull 2.

## 3. The trace — `PM_RecursiveHullCheck` (port this variant)

Init: `trace = {fraction: 1.0, endpos: end, allsolid: true, ent: -1}`.

```c
#define DIST_EPSILON 0.03125f                     // 1/32
bool RecursiveHullCheck(hull, num, p1f, p2f, p1, p2, trace):
  if (num < 0) {                                   // leaf: num IS the contents — checked FIRST (engine order)
      if (num != CONTENTS_SOLID) { trace->allsolid = false;
          if (num == CONTENTS_EMPTY) trace->inopen = true; else trace->inwater = true; }
      else trace->startsolid = true;
      return true;                                 // empty leaf: keep going
  }
  // pmove variant only, AFTER the leaf check: empty-hull guard
  if (hull->firstclipnode >= hull->lastclipnode) { trace->allsolid = false; trace->inopen = true; return true; }
  node = clipnodes[num]; plane = planes[node.planenum];
  t1 = plane.type < 3 ? p1[plane.type] - dist : dot(normal, p1) - dist;   // axial fast path
  t2 = ... p2 ...;
  if (t1 >= 0 && t2 >= 0) return RecursiveHullCheck(children[0], p1f, p2f, p1, p2);
  if (t1 <  0 && t2 <  0) return RecursiveHullCheck(children[1], ...);
  frac = clamp( t1 < 0 ? (t1 + DIST_EPSILON)/(t1 - t2) : (t1 - DIST_EPSILON)/(t1 - t2), 0, 1);
  midf = p1f + (p2f - p1f)*frac;  mid = p1 + frac*(p2 - p1);
  side = (t1 < 0);
  if (!RecursiveHullCheck(children[side], p1f, midf, p1, mid)) return false;      // near side
  if (HullPointContents(children[side^1], mid) != CONTENTS_SOLID)                  // past the node
      return RecursiveHullCheck(children[side^1], midf, p2f, mid, p2);
  if (trace->allsolid) return false;                                               // never left solid
  trace->plane = side ? {-normal, -dist} : {normal, dist};                         // ALWAYS faces the motion
  while (HullPointContents(hull, firstclipnode, mid) == CONTENTS_SOLID) {          // numeric backup
      frac -= 0.05f;                              // pmove variant (SV variant: 0.1)
      if (frac < 0) { trace->fraction = midf; trace->endpos = mid; return false; }
      midf = ...; mid = ...;
  }
  trace->fraction = midf; trace->endpos = mid; return false;
```
After: pmove semantics `if (allsolid) startsolid = true; if (startsolid) fraction = 0;` and if `fraction < 1`: `endpos = start + fraction·(end − start)` recomputed in world space. Implementation note: convert to an explicit stack; keep clipnodes/planes as flat contiguous arrays.

`HullPointContents(hull, num, p)`: walk `num = (dist < 0 ? children[1] : children[0])` until negative → contents.

**Multi-entity trace** (`_PM_PlayerTrace`): clip against world model + every solid brush entity; when a per-entity trace has *strictly smaller* fraction, copy it **wholesale** (do NOT OR-merge allsolid/startsolid across entities — a startsolid trace already forces its own fraction to 0, which is how it wins; ties keep the earlier entity). Surf maps: world + a handful of `func_wall`s — precompute the solid-entity list at load.

Contents codes: `EMPTY −1, SOLID −2, WATER −3, SLIME −4, LAVA −5, SKY −6, TRANSLUCENT −15, LADDER −16` (currents −9…−14 count as water).

## 4. Entities

`LUMP_ENTITIES` = text: `{ "key" "value" ... }` blocks; brush entities carry `"model" "*N"`. Parse into a classname-indexed list. From surf_ski_2's actual census (10 trigger_teleport, 7 trigger_push, 6 func_ladder, 5 func_wall, 4 func_water, 10 func_illusionary, 9+9 spawns…):

**Solid to movement** (clip via their model's clipnode subtree + origin offset): `func_wall`, `func_breakable`, `func_pushable`, `func_door*`/`func_button`/`func_train` (SOLID_BSP; doors with `"skin" != 0` are *water*, not solid — `cs_doors.cpp`), `func_conveyor`.
**Never solid:** `func_water` (`skin −3` contents volume), `func_ladder` (`skin` CONTENTS_LADDER), all triggers, and `func_illusionary` — **but** a SOLID_NOT brush entity with `skin < −1` is a live *contents volume*, not render-only (the engine links it and `PM_LinkContents` returns its skin): surf_ski_2's `func_illusionary` "water wall" (`*2`, skin −3) reads as water to `PM_CheckWater`. Rule: SOLID_NOT + skin −1/unset → render-only; SOLID_NOT + skin < −1 → treat exactly like func_water.
One caveat on rotation: surf_ski_2's jail has a `func_door_rotating` (`*19`) — a solid that *would* rotate if button/door logic were implemented. Its spawn angles are 0, so the "assert no rotated solids" stance holds as long as door logic stays unimplemented (the jail door just stays shut).

**Trigger touch test — exact engine semantics, cheap to replicate:** per moved player, for each trigger whose padded AABB overlaps the player's AABB:
```c
hull = trigger_model.hull[by SV_HullForBsp for the PLAYER's box];   // standing player → trigger's hull 1
inside = HullPointContents(hull, headnode, player_origin − trigger_origin) == CONTENTS_SOLID;
```
(A trigger brush's interior reads SOLID within its own model subtree.) This is *exact*, not an approximation — v1 ships it.

**Behaviors** (ReGameDLL `triggers.cpp`):
- `trigger_teleport`: destination = entity with `targetname == target` (usually `info_teleport_destination`, but surf_ski_2 uses `info_target` — look up by targetname over ALL point entities). Set `origin = dest.origin; origin.z −= player_mins.z (+36 standing); origin.z += 1`; clear onground; copy dest `angles` to **both** angles and view angles (full vector, pitch included, `fixangle = 1` — env may keep yaw-only since pitch is frozen, but note the deviation); **zero velocity and basevelocity** (stock CS).
- `trigger_hurt`: `dmg` is damage **per second**; applies `dmg × 0.5` per touch, rate-limited to one touch per 0.5 s. Env: `dmg ≥ 100`-ish → instant fail; negative dmg (heal pools) → ignore.
- `trigger_push`: see [01 §6](01-physics-spec.md) — sets basevelocity each frame inside.
- Spawns: `info_player_start` = CT, `info_player_deathmatch` = T (both usable directly as hull-center player origins — but mappers place them loosely: surf_ski_2's sit ~74u above the floor, so the player free-falls ~38u on spawn; settle with one down-trace if you want grounded starts).

**Water:** worldspawn water = hull-0 leaf contents −3; `func_water` (and skin<−1 illusionaries) = SOLID_NOT entities whose hull-0 tree + `skin` give contents. **Point contents must therefore scan contents entities after the world query** — engine `PM_PointContents` = world hull-0 lookup, then `PM_LinkContents` over non-solid physents returning `pe->skin` on containment. surf_ski_2's water bottom is 4 `func_water` volumes: a worldspawn-only point contents is blind to the map's primary landing volume. `PM_CheckWater` probes `origin + (0, 0, mins.z + 1)` (feet); waterlevel 1 = feet, 2 = center (swim physics engage — out of scope → episode fail), 3 = eyes.

## 5. First maps & acquisition

`surf_ski_2` (already in `maps/`, from `https://en.ds-servers.com/maps/goldsrc/cstrike/surf_ski_2.zip`) — the canonical 1.6 surf map, #1 by server count: one huge room, two mirrored forgiving ramps, water bottom, jail-teleport fails. Next: `surf_ski`, `surf_egypt`, `surf_green`, `surf_iceday`, `surf_minilevels_v3` (short linear stages — natural curriculum). Sources: ds-servers.com (direct zips), GameBanana CS 1.6, varq.net surf list, loadcs.com packs. Note: ramps are plain worldspawn brushes — there is no "surf entity"; the map + `sv_airaccelerate 100` is the whole game.

Textures: mostly embedded in the TEXTURES lump; when `miptex.offsets[0] == 0` (external WAD) the visualizer falls back to flat shading — collision never cares.

## 6. `src/bsp.c` / `src/trace.c` deliverables

```c
typedef struct BspMap {           // immutable after load, shared by all envs/threads
    Plane*     planes;            // {float n[3], dist; int type;}
    ClipNode*  clipnodes;         // world + entity subtrees + generated hull0 array
    ClipNode*  hull0nodes;        //   (from NODES/LEAFS via MakeHull0)
    Model*     models;            // headnode[4] per model, origin, bounds
    SolidEnt*  solids;            // model idx + origin, precomputed list
    ContentsEnt* contents_ents;   // SOLID_NOT volumes: model idx + origin + skin (func_water, skin<-1 illusionary, func_ladder)
    TriggerEnt* triggers;         // kind (tele/hurt/push), model idx, AABB, target data
    SpawnPoint* spawns;           // origin, yaw
    /* render lumps kept only by tools/export_map.py — not loaded by the sim */
} BspMap;

void trace_hull(const BspMap*, int usehull, const float* start, const float* end, Trace* out);   // world + solids
int  point_contents(const BspMap*, const float* p);   // world hull 0, THEN contents_ents scan (returns their skin)
int  trigger_test(const BspMap*, int usehull, const float* origin, int trigger_idx);             // exact containment
```
Validation gates for this layer are in [06-night-plan.md](06-night-plan.md) (Gates A/B) and [05-validation.md](05-validation.md) §8.
