
## format_markdown
# GoldSrc BSP v30 — file format and which lumps you need

All data is **little-endian** (the engine loads with `LittleLong`/`LittleShort`, which are no-ops on x86). File begins with a 4-byte version that must be `30` (`HLBSP_VERSION 30`; Quake1 is 29), followed by a directory of 15 lumps. Header size = 4 + 15*8 = **124 bytes**.

```c
typedef struct lump_s { int fileofs; int filelen; } lump_t;      // 8 bytes
typedef struct dheader_s { int version; lump_t lumps[15]; } dheader_t;  // 124 bytes
```

## Lump directory (from ReHLDS `public/rehlds/bspfile.h`, identical to the hlbsp spec)

| idx | name | element type | elem size (bytes) | count = filelen/size |
|-----|------|--------------|-------------------|----------------------|
| 0 | LUMP_ENTITIES | ASCII text (null-terminated) | 1 | — |
| 1 | LUMP_PLANES | `dplane_t` | 20 | ✔ |
| 2 | LUMP_TEXTURES | `dmiptexlump_t` header + `miptex_t` (40) | var | — |
| 3 | LUMP_VERTEXES | `dvertex_t` (3×float) | 12 | ✔ |
| 4 | LUMP_VISIBILITY | RLE-compressed PVS bits | 1 | — |
| 5 | LUMP_NODES | `dnode_t` | 24 | ✔ |
| 6 | LUMP_TEXINFO | `texinfo_t` | 40 | ✔ |
| 7 | LUMP_FACES | `dface_t` | 20 | ✔ |
| 8 | LUMP_LIGHTING | RGB8 lightmap samples | 3 | — |
| 9 | LUMP_CLIPNODES | `dclipnode_t` | 8 | ✔ |
| 10 | LUMP_LEAFS | `dleaf_t` | 28 | ✔ |
| 11 | LUMP_MARKSURFACES | `uint16` face index | 2 | ✔ |
| 12 | LUMP_EDGES | `dedge_t` (2×uint16) | 4 | ✔ |
| 13 | LUMP_SURFEDGES | `int32` (signed!) | 4 | ✔ |
| 14 | LUMP_MODELS | `dmodel_t` | 64 | ✔ |

`HEADER_LUMPS = 15`, `MAX_MAP_HULLS = 4`, `MAX_MAP_LEAFS = 32767` (signed-short limit; hlbsp spec lists compiler limits `MAX_MAP_CLIPNODES = 32767`, `MAX_MAP_NODES = 32767` "because negative shorts are leaves").

Verified against a real map: `surf_ski_2.bsp` parses as version 30 with exactly this table (e.g., CLIPNODES off=347532 len=59032 → 7379 clipnodes; MODELS len=3200 → 50 models = worldspawn + 49 `*N` brush models).

## Which lumps you need

**(a) Collision (hull traces, CS 1.6-identical):**
- `LUMP_PLANES` (1) — shared by render BSP and clip hulls.
- `LUMP_CLIPNODES` (9) — one flat array; hulls 1–3 of every model are subtrees inside it (roots = `dmodel_t.headnode[1..3]`). Clipnodes already contain the **compiler-expanded (Minkowski) geometry per hull** — the engine never expands brushes at runtime.
- `LUMP_MODELS` (14) — bounds, `headnode[4]`, and `*N` sub-model roots.
- `LUMP_NODES` (5) + `LUMP_LEAFS` (10) — required to build **hull 0** (the point hull). At load, `Mod_MakeHull0` converts the render node tree into a parallel clipnode array: `planenum = node.plane index`, `children[j] = leaf ? leaf->contents : child node index`. Point-contents (water/solid/sky detection) runs on hull 0, so you need NODES+LEAFS even for pure collision.

**(b) Entities:** `LUMP_ENTITIES` (0) — plain text, see entities section.

**(c) Visualization mesh:** `LUMP_VERTEXES` (3), `LUMP_EDGES` (12), `LUMP_SURFEDGES` (13), `LUMP_FACES` (7); `LUMP_TEXINFO` (6) for UVs / texture assignment; `LUMP_TEXTURES` (2) for texture names+embedded miptex pixels (WAD3 palette format; name lookup in external .wad if `offsets[0]==0`); optional `LUMP_MODELS` to know which faces belong to which brush entity (`firstface/numfaces`), `LUMP_MARKSURFACES`+`LUMP_LEAFS`+`LUMP_VISIBILITY` only if you want PVS culling, `LUMP_LIGHTING` for lightmaps.

**Face → triangle recipe:** for face `f`, iterate `i` in `[f.firstedge, f.firstedge+f.numedges)`; `se = surfedges[i]`; vertex index = `se >= 0 ? edges[se].v[0] : edges[-se].v[1]`. This yields the convex polygon's vertices in winding order — fan-triangulate. Face normal = `planes[f.planenum].normal`, negated if `f.side != 0`. UVs: `u = dot(vert, texinfo.vecs[0].xyz) + texinfo.vecs[0].w`, `v = dot(vert, texinfo.vecs[1].xyz) + texinfo.vecs[1].w`, divide by miptex width/height.


## structs_markdown
# Byte-exact struct layouts (quoted from ReHLDS `rehlds/public/rehlds/bspfile.h`)

```c
#define Q1BSP_VERSION  29   // quake1 regular version
#define HLBSP_VERSION  30   // half-life regular version
#define MAX_MAP_HULLS  4
#define MAX_MAP_LEAFS  32767 // signed short limit

typedef struct lump_s {
    int   fileofs;
    int   filelen;
} lump_t;                                    // 8 bytes

typedef struct dheader_s {
    int    version;                          // must be 30
    lump_t lumps[15];
} dheader_t;                                 // 124 bytes

typedef struct dplane_s {
    float normal[3];
    float dist;
    int   type;      // PLANE_X=0, PLANE_Y=1, PLANE_Z=2 (axial), PLANE_ANYX=3, ANYY=4, ANYZ=5
} dplane_t;                                  // 20 bytes

typedef struct dvertex_s { float point[3]; } dvertex_t;   // 12 bytes

typedef struct dnode_s {
    int            planenum;
    short          children[2];  // >=0: node index; <0: -(leafindex+1)  (i.e. ~leaf)
    short          mins[3];      // bbox for culling
    short          maxs[3];
    unsigned short firstface;
    unsigned short numfaces;
} dnode_t;                                   // 24 bytes

typedef struct dclipnode_s {
    int   planenum;
    short children[2];   // negative numbers are contents (CONTENTS_*)
} dclipnode_t;                               // 8 bytes  (children are SIGNED int16)

typedef struct dleaf_s {
    int            contents;         // CONTENTS_* (negative)
    int            visofs;           // -1 = no vis info
    short          mins[3], maxs[3];
    unsigned short firstmarksurface, nummarksurfaces;
    byte           ambient_level[4];
} dleaf_t;                                   // 28 bytes

typedef struct dmodel_s {
    float mins[3], maxs[3];
    float origin[3];
    int   headnode[MAX_MAP_HULLS];   // [0]=root NODE index (render tree / hull0)
                                     // [1..3]=root CLIPNODE indices for hulls 1..3
    int   visleafs;                  // not including the solid leaf 0
    int   firstface, numfaces;
} dmodel_t;                                  // 64 bytes

typedef struct dedge_s { unsigned short v[2]; } dedge_t;   // 4 bytes
// surfedges lump: int32 per entry; negative = walk edge backwards (use v[1])
// marksurfaces lump: uint16 per entry (face index)

typedef struct dface_s {
    short planenum;
    short side;         // nonzero => normal is flipped
    int   firstedge;    // into surfedges
    short numedges;
    short texinfo;
    byte  styles[4];
    int   lightofs;     // into LIGHTING lump, -1 = none
} dface_t;                                   // 20 bytes

typedef struct texinfo_s {
    float vecs[2][4];   // [s/t][xyz + offset]
    int   _miptex;      // index into miptex list
    int   flags;        // 0 or TEX_SPECIAL(1) = no lightmap (sky/liquids)
} texinfo_t;                                 // 40 bytes

typedef struct dmiptexlump_s {
    int _nummiptex;
    int dataofs[4];     // actually dataofs[nummiptex], -1 = missing
} dmiptexlump_t;

typedef struct miptex_s {
    char     name[16];
    unsigned width, height;
    unsigned offsets[4]; // offsets to 4 mip levels from miptex start; 0 => pixels in external WAD
} miptex_t;                                  // 40 bytes + pixel data
```

The hlbsp project spec (`bernhardmgruber/hlbsp src/bspdef.h`, port of the SourceForge "Unofficial BSP v30 File Format Specification") defines identical layouts using stdint types (`ClipNode { int32_t planeIndex; int16_t childIndex[2]; }`, `Model { vec3 lower,upper; vec3 origin; int32_t headNodesIndex[4]; int32_t visLeaves; int32_t firstFace, faceCount; }`, etc.) and adds compiler limits: `MAX_MAP_PLANES/NODES/CLIPNODES = 32767`, `MAX_MAP_VERTS/FACES/MARKSURFACES = 65535`, `MAX_MAP_ENTSTRING = 128*1024`, `MAX_KEY = 32`, `MAX_VALUE = 1024`.

# Runtime structs the trace code uses (from `common/com_model.h` / engine)

```c
typedef struct mplane_s {
    vec3_t normal;
    float  dist;
    byte   type;      // for fast axial side tests (< 3 = axial)
    byte   signbits;  // unused by hull code
    byte   pad[2];
} mplane_t;

typedef struct hull_s {
    dclipnode_t *clipnodes;
    mplane_t    *planes;      // shared model plane array
    int          firstclipnode;
    int          lastclipnode;
    vec3_t       clip_mins;   // hull dimensions (see hulls table)
    vec3_t       clip_maxs;
} hull_t;
```

Hull wiring at load (`Mod_LoadClipnodes` + submodel setup in `Mod_LoadBrushModel`, ReHLDS `engine/model.cpp`): all four hulls of a model share `planes`; hulls 1–3 share the single clipnode array with `firstclipnode = dmodel.headnode[h]`, `lastclipnode = numclipnodes-1`; hull 0 uses the `Mod_MakeHull0` node-mirror array with `firstclipnode = headnode[0]`, `lastclipnode = numnodes-1`.

# Trace result structs

```c
typedef struct pmplane_s { vec3_t normal; float dist; } pmplane_t;

typedef struct pmtrace_s {                    // pm_shared (player movement)
    qboolean  allsolid;      // if true, plane is not valid
    qboolean  startsolid;    // if true, the initial point was in a solid area
    qboolean  inopen, inwater; // End point is in empty space or in water
    float     fraction;      // time completed, 1.0 = didn't hit anything
    vec3_t    endpos;        // final position
    pmplane_t plane;         // surface normal at impact
    int       ent;           // entity (physent index) at impact
    vec3_t    deltavelocity; // change in player velocity caused by impact (server only)
    int       hitgroup;
} pmtrace_t;
```
The server-side `trace_t` is identical except `ent` is an `edict_t*`.


## hulls

### item 0

#### index
0

#### mins
(0, 0, 0)


#### maxs
(0, 0, 0)


#### used_for
Point hull. pmove usehull 2 (point traces, ground/stuck probes, hitscan). Built at load from NODES+LEAFS by Mod_MakeHull0, so it also serves SV_PointContents / PM_PointContents (water, sky, solid tests). SV_HullForBsp selects it when trace bbox size[0] <= 8.


### item 1

#### index
1

#### mins
(-16, -16, -36)


#### maxs
(16, 16, 36)


#### used_for
STANDING PLAYER (32x32x72). pmove usehull 0 -> PM_HullForBsp default case -> hulls[1]. SV_HullForBsp selects it when size[0] <= 36 && size[2] > 36. clip_mins/clip_maxs hardcoded in Mod_LoadClipnodes; identical to engine player_mins[0]/player_maxs[0], so trace offset vs world is exactly 0 and the trace runs in player-origin space.


### item 2

#### index
2

#### mins
(-32, -32, -32)


#### maxs
(32, 32, 32)


#### used_for
LARGE hull (64x64x64). pmove usehull 3 -> hulls[2]. SV_HullForBsp selects it when size[0] > 36. Used by big monsters/pushables; irrelevant for surf players but must exist in the loader.


### item 3

#### index
3

#### mins
(-16, -16, -18)


#### maxs
(16, 16, 18)


#### used_for
CROUCHED PLAYER (32x32x36). pmove usehull 1 -> PM_HullForBsp case 1 -> hulls[3]. SV_HullForBsp selects it when size[0] <= 36 && size[2] <= 36. Matches player_mins[1]/player_maxs[1] = (-16,-16,-18)..(16,16,18).


## trace_algorithm_markdown
# CS 1.6 hull tracing, exactly as the engine does it

## Player origin is at hull center — CONFIRMED
Engine `player_mins/player_maxs` (ReHLDS `engine/pmove.cpp`, indexed by **pmove usehull**, not BSP hull):
```c
vec3_t player_mins[MAX_MAP_HULLS] = {
    { -16.0f, -16.0f, -36.0f }, // usehull 0: standing
    { -16.0f, -16.0f, -18.0f }, // usehull 1: ducked
    {   0.0f,   0.0f,   0.0f }, // usehull 2: point
    { -32.0f, -32.0f, -32.0f }, // usehull 3: large
};
vec3_t player_maxs[MAX_MAP_HULLS] = {
    { 16.0f, 16.0f, 36.0f },
    { 16.0f, 16.0f, 18.0f },
    {  0.0f,  0.0f,  0.0f },
    { 32.0f, 32.0f, 32.0f },
};
```
The box is symmetric about the origin in all axes ⇒ origin is the AABB center: feet = origin.z − 36 standing / − 18 ducked. ReGameDLL `TeleportTouch` even comments: *"make origin adjustments in case the teleportee is a player. (origin in center, not at feet)"* and does `tmp.z -= pOther->pev->mins.z`. Eye height: `PM_VEC_VIEW = 17`, `PM_VEC_DUCK_VIEW = 12` (view_ofs.z above origin), `PM_VEC_HULL_MIN = -36`, `PM_VEC_DUCK_HULL_MIN = -18` (ReGameDLL `pm_shared.h`). usehull is set to 0 when standing/unducking and 1 while `FL_DUCKING`/in-duck (pm_shared.cpp).

## Hull selection — server path (`SV_HullForBsp`, ReHLDS `engine/world.cpp`)
Selection is by the **size of the tracing bbox** (`size = maxs - mins`), vanilla behavior (non-REHLDS_FIXES branch):
```c
VectorSubtract(maxs, mins, size);
if (size[0] <= 8.0f) {
    hull = &model->hulls[0];
    VectorCopy(hull->clip_mins, offset);       // hull0 clip_mins is zero
} else {
    if (size[0] <= 36.0f) {
        if (size[2] <= 36.0f) hull = &model->hulls[3];  // crouch
        else                  hull = &model->hulls[1];  // standing
    } else                    hull = &model->hulls[2];  // large
    VectorSubtract(hull->clip_mins, mins, offset);  // center-origin correction
}
VectorAdd(offset, ent->v.origin, offset);
```
`SV_HullForEntity`: `SOLID_BSP` entities (must be MOVETYPE_PUSH/PUSHSTEP, else Sys_Error) go through `SV_HullForBsp`; all other entities get a temp 6-plane box hull via `SV_HullForBox(ent->v.mins - maxs, ent->v.maxs - mins)` with `offset = ent->v.origin` (Minkowski expansion of the AABB by the trace box).

## Hull selection — pmove path (`PM_HullForBsp`, ReHLDS `engine/pmovetst.cpp`)
Direct usehull→BSP-hull map (this answers "standing traces which hull?"):
```c
switch (pmove->usehull) {
    case 1:  hull = &pe->model->hulls[3]; break;  // ducked  -> crouch hull
    case 2:  hull = &pe->model->hulls[0]; break;  // point   -> point hull
    case 3:  hull = &pe->model->hulls[2]; break;  // large
    default: hull = &pe->model->hulls[1]; break;  // standing -> human hull
}
offset = hull->clip_mins - player_mins[usehull] + pe->origin;
```
For the world (`pe->origin = 0`) and player hulls, `clip_mins == player_mins[usehull]`, so **offset = 0**: the player's center-origin is traced directly through the pre-expanded clipnode hull. Then `start_l = start - offset; end_l = end - offset` and, for rotated SOLID_BSP entities with nonzero angles, start/end are rotated into model space with `AngleVectors` (and the hit normal rotated back with `AngleVectorsTranspose`).

## Trace wrapper (`SV_SingleClipMoveToEntity` / `_PM_PlayerTrace`)
Init per entity: `memset(trace,0)`, `trace.fraction = 1.0f`, `trace.endpos = end`, `trace.allsolid = TRUE` (pmove also `trace.ent = -1`). Then `SV_RecursiveHullCheck(hull, hull->firstclipnode, 0.0f, 1.0f, start_l, end_l, &trace)`. Afterwards:
- pmove: `if (total.allsolid) total.startsolid = 1; if (total.startsolid) total.fraction = 0;`
- if `fraction != 1.0`: `endpos = start + (end-start)*fraction` (in world space).
- SV_Move first clips against world (`edicts[0]`), then `SV_ClipToLinks` clips against every linked solid entity in the areanode tree, keeping the trace with `allsolid || startsolid || fraction < best.fraction`; final `clip.trace.fraction *= trace_fraction` (fractions are relative to the world-clipped segment). pmove `_PM_PlayerTrace` loops all `physents` and keeps min-fraction, setting `trace.ent = i` (physent index).

## SV_RecursiveHullCheck — complete algorithm (ReHLDS `engine/world.cpp`, vanilla branch)
```c
const float DIST_EPSILON = 0.03125f;   // 1/32

qboolean SV_RecursiveHullCheck(hull_t *hull, int num, float p1f, float p2f,
                               const vec_t *p1, const vec_t *p2, trace_t *trace)
{
    float pdif = p2f - p1f;
    if (num < 0) {                              // reached a leaf (contents)
        if (num != CONTENTS_SOLID) {
            trace->allsolid = FALSE;
            if (num == CONTENTS_EMPTY)            trace->inopen  = TRUE;
            else if (num != CONTENTS_TRANSLUCENT) trace->inwater = TRUE;
        } else {
            trace->startsolid = TRUE;
        }
        return TRUE;                            // empty: keep going
    }
    // (bounds check num against hull->firstclipnode/lastclipnode -> Sys_Error)

    node  = &hull->clipnodes[num];
    plane = &hull->planes[node->planenum];
    if (plane->type < 3) { t1 = p1[plane->type] - plane->dist;
                           t2 = p2[plane->type] - plane->dist; }
    else                 { t1 = DotProduct(plane->normal, p1) - plane->dist;
                           t2 = DotProduct(plane->normal, p2) - plane->dist; }

    if (t1 >= 0.0f && t2 >= 0.0f) return SV_RecursiveHullCheck(hull, node->children[0], p1f, p2f, p1, p2, trace);
    if (t1 <  0.0f && t2 <  0.0f) return SV_RecursiveHullCheck(hull, node->children[1], p1f, p2f, p1, p2, trace);

    // put the crosspoint DIST_EPSILON pixels on the near side
    if (t1 < 0.0f) frac = (t1 + DIST_EPSILON) / (t1 - t2);
    else           frac = (t1 - DIST_EPSILON) / (t1 - t2);
    frac = clamp(frac, 0.0f, 1.0f);
    if (IS_NAN(frac)) return FALSE;             // not a number

    midf = p1f + pdif * frac;
    mid  = p1 + frac * (p2 - p1);
    side = (t1 < 0.0f) ? 1 : 0;

    // move up to the node
    if (!SV_RecursiveHullCheck(hull, node->children[side], p1f, midf, p1, mid, trace))
        return FALSE;

    if (SV_HullPointContents(hull, node->children[side^1], mid) != CONTENTS_SOLID)
        // go past the node
        return SV_RecursiveHullCheck(hull, node->children[side^1], midf, p2f, mid, p2, trace);

    if (trace->allsolid) return FALSE;          // never got out of the solid area

    // the other side of the node is solid, this is the impact point
    if (!side) { trace->plane.normal =  plane->normal; trace->plane.dist =  plane->dist; }
    else       { trace->plane.normal = -plane->normal; trace->plane.dist = -plane->dist; }  // SIDE FLIP

    // back off if mid is (numerically) inside solid — "shouldn't really happen, but does occasionally"
    while (SV_HullPointContents(hull, hull->firstclipnode, mid) == CONTENTS_SOLID) {
        frac -= 0.1f;
        if (frac < 0.0f) { trace->fraction = midf; trace->endpos = mid;
                           Con_DPrintf("backup past 0\n"); return FALSE; }
        midf = p1f + pdif * frac;
        mid  = p1 + frac * (p2 - p1);
    }
    trace->fraction = midf;
    trace->endpos   = mid;      // caller recomputes endpos from fraction in world space
    return FALSE;
}
```

**Semantics:**
- `allsolid` starts TRUE and is cleared the moment any non-SOLID leaf is visited: TRUE afterwards ⇒ the whole swept segment stayed in solid.
- `startsolid` is set whenever a segment-start lands in a CONTENTS_SOLID leaf (in practice: trace began inside solid). The engine still lets the move escape (`SV_Move` comment: "if the starting point is in a solid, it will be allowed to move out to an open area"). pmove forces `fraction = 0` when startsolid.
- `inopen` / `inwater`: endpoint region flags; CONTENTS_EMPTY sets inopen, any other non-solid contents except CONTENTS_TRANSLUCENT sets inwater.
- **Side flip:** the reported plane always faces the incoming motion — if the impact came from the plane's back side (`side==1`), `normal` and `dist` are negated.
- `DIST_EPSILON = 0.03125f` (1/32): the crossing point is nudged toward the near side so the recursion never starts exactly on a plane.

**pmove variant (`PM_RecursiveHullCheck`, `engine/pmovetst.cpp`) differences** — algorithm identical except: local `float DIST_EPSILON = 0.03125f`; empty-hull guard `if (hull->firstclipnode >= hull->lastclipnode) { allsolid=FALSE; inopen=TRUE; return TRUE; }`; the leaf branch sets `inwater` for *anything* non-solid non-empty (no TRANSLUCENT exception); the backup loop steps `midf -= 0.05` (not 0.1) and re-derives frac; there is no NaN check.

## Point contents (`SV_HullPointContents` — the primitive everything uses)
```c
int SV_HullPointContents(hull_t *hull, int num, const vec_t *p) {
    while (num >= 0) {
        node  = &hull->clipnodes[num];
        plane = &hull->planes[node->planenum];
        d = (plane->type < 3) ? p[plane->type] - plane->dist
                              : DotProduct(plane->normal, p) - plane->dist;
        num = (d < 0) ? node->children[1] : node->children[0];
    }
    return num;   // a CONTENTS_* value
}
```

## CONTENTS_* codes (exact values, `common/const.h` + `bspfile.h`)
```c
CONTENTS_EMPTY        -1
CONTENTS_SOLID        -2
CONTENTS_WATER        -3
CONTENTS_SLIME        -4
CONTENTS_LAVA         -5
CONTENTS_SKY          -6
CONTENTS_ORIGIN       -7   // removed at csg time
CONTENTS_CLIP         -8   // changed to CONTENTS_SOLID at csg time
CONTENTS_CURRENT_0    -9
CONTENTS_CURRENT_90   -10
CONTENTS_CURRENT_180  -11
CONTENTS_CURRENT_270  -12
CONTENTS_CURRENT_UP   -13
CONTENTS_CURRENT_DOWN -14
CONTENTS_TRANSLUCENT  -15
CONTENTS_LADDER       -16
CONTENT_FLYFIELD      -17
CONTENT_GRAVITY_FLYFIELD -18
CONTENT_FOG           -19
```
Solid types: `SOLID_NOT 0, SOLID_TRIGGER 1, SOLID_BBOX 2, SOLID_SLIDEBOX 3, SOLID_BSP 4`. Move types of interest: `MOVETYPE_WALK 3 (players), MOVETYPE_PUSH 7 (brush ents), MOVETYPE_NONE 0`. Trace types: `MOVE_NORMAL 0, MOVE_NOMONSTERS 1, MOVE_MISSILE 2`; pmove flags `PM_NORMAL 0, PM_STUDIO_IGNORE 1, PM_STUDIO_BOX 2, PM_GLASS_IGNORE 4, PM_WORLD_ONLY 8`.


## entities_markdown
# Entity lump, brush entities, triggers, and water

## LUMP_ENTITIES format
Null-terminated ASCII text: a sequence of `{ "key" "value" ... }` blocks, one per entity; the first block is `worldspawn` (carries `"wad"` list, `"skyname"`, `"MaxRange"`). Limits per hlbsp spec: key ≤ 32 chars, value ≤ 1024 chars, entstring ≤ 128 KiB. A brush entity references its geometry with `"model" "*N"` where N indexes `LUMP_MODELS` (model 0 is the world). Point entities have `"origin" "x y z"` and often `"angles"`. Real example from `surf_ski_2.bsp`:
```
{ "model" "*23" "targetname" "j" "killtarget" "j" "target" "jail" "classname" "trigger_teleport" }
{ "origin" "-451.661 3140.53 812" "targetname" "jail" "classname" "info_target" }
{ "model" "*3" "skin" "-3" "WaveHeight" "0" "wait" "4" "classname" "func_water" ... }
{ "model" "*12" "speed" "1200" "angles" "-90 0 0" "classname" "trigger_push" }
```

## Which brush entity classes are SOLID to movement (verified in ReGameDLL_CS)
`SOLID_BSP` + `MOVETYPE_PUSH` (clip via their `*N` model's clipnode hulls, exactly like the world):
- `func_wall` (`pev->solid = SOLID_BSP; pev->flags |= FL_WORLDBRUSH` — "If it can't move/go away, it's really part of the world"), and its children `func_wall_toggle` (toggles between SOLID_BSP and SOLID_NOT + EF_NODRAW), `func_conveyor`, `func_monsterclip` (FL_MONSTERCLIP — ignored for players).
- `func_breakable`, `func_pushable` (breakable is SOLID_BSP until broken), `func_door` / `func_door_rotating` / `func_rotating` / `func_train` / `func_button` (SOLID_BSP unless SF_DOOR_PASSABLE).
- **`func_door` special case = `func_water`**: doors.cpp — `if (pev->skin == 0) solid = SOLID_BSP else solid = SOLID_NOT` — a nonzero `skin` (contents value, e.g. `-3` water) makes the "door" a **non-solid contents volume**.

`SOLID_NOT` (never collide):
- `func_illusionary` — Spawn(): `pev->movetype = MOVETYPE_NONE; /* always solid_not */ pev->solid = SOLID_NOT;` (skin key = optional contents). Render-only geometry.
- `func_water` (skin != 0, usually -3) — contents volume, see water below.
- `func_ladder` — `pev->solid = SOLID_NOT; pev->skin = CONTENTS_LADDER;` (contents volume; pm_shared finds ladders by scanning physents for skin == CONTENTS_LADDER).

For a standalone surf engine: clip movement against world model + every SOLID_BSP brush entity (each `*N` model has its own clipnode subtree via `dmodel_t.headnode[hullIdx]`; add the entity's `origin` into the trace offset). Skip SOLID_NOT entirely for collision; use their hull-0 tree + `skin` for point contents.

## SOLID_TRIGGER volumes and touch detection (engine `world.cpp`)
`CBaseTrigger::InitTrigger()`: `pev->solid = SOLID_TRIGGER; pev->movetype = MOVETYPE_NONE; SET_MODEL(...model...)` (+ optional `SetMovedir` from angles). Trigger entities are linked into areanode `trigger_edicts` chains, never into `solid_edicts` — `SV_ClipToLinks` even `Sys_Error`s on a trigger in the clipping list, so **triggers never block movement**.

Touch detection runs whenever an entity is (re)linked after moving — `SV_LinkEdict(ent, touch_triggers=true)` → `SV_TouchLinks`, i.e. every physics frame the player moves. The exact test, per trigger:
1. `touch->v.solid == SOLID_TRIGGER` and touch != ent.
2. Coarse: `BoundsIntersect(ent->v.absmin, ent->v.absmax, touch->v.absmin, touch->v.absmax)` (absmin/absmax = origin+mins/maxs padded ±1 by SetObjectCollisionBox).
3. **Exact (brush triggers): a point-vs-hull test using the toucher's size**:
```c
if (Mod_GetType(touch->v.modelindex) == mod_brush) {
    hull_t *hull = SV_HullForBsp(touch, ent->v.mins, ent->v.maxs, offset);  // trigger's hull for the PLAYER's bbox size
    VectorSubtract(ent->v.origin, offset, localPosition);
    if (SV_HullPointContents(hull, hull->firstclipnode, localPosition) != CONTENTS_SOLID)
        continue;   // not really touching
}
gEntityInterface.pfnTouch(touch, ent);   // fire Touch callback
```
So: the trigger's own clipnode hull (selected by the *player's* hull size — standing player ⇒ trigger model's hull 1) is queried at the player's origin; "inside" reads CONTENTS_SOLID because a trigger brush's interior is solid within its own model subtree. This is what you replicate for teleport/kill/boost zones: after each movement step, for each trigger whose AABB overlaps yours, run `HullPointContents(triggerModel.hull[playerHull], headnode, playerOrigin - triggerOrigin) == CONTENTS_SOLID`.

### Trigger behaviors (ReGameDLL_CS `dlls/triggers.cpp`)
- **trigger_teleport** → `TeleportTouch`: only FL_CLIENT|FL_MONSTER; finds destination by `FIND_ENTITY_BY_TARGETNAME(pev->target)` (usually `info_teleport_destination`, but **any targetnamed point entity works — surf_ski_2 uses `info_target`**); `tmp = dest.origin; if (player) tmp.z -= pOther->pev->mins.z;` ("origin in center, not at feet"), `tmp.z++`, clears `FL_ONGROUND`, sets origin, copies dest `angles` into `angles`/`v_angle` with `fixangle = 1`, and zeroes `velocity` and `basevelocity` (stock CS; ReGameDLL adds optional keep-angles/keep-velocity spawnflags 256/512/1024).
- **trigger_hurt** → `HurtTouch`: requires `takedamage`; applies `fldmg = pev->dmg * 0.5f` per touch (dmg key is damage/sec), rate-limited via `pev->dmgtime = time + 0.5` (per-player mask `pev->impulse` in MP); negative dmg heals (`TakeHealth`). Kill zones are just `dmg` ≥ 2×health.
- **trigger_push** → `Touch`: ignores MOVETYPE_NONE/PUSH/NOCLIP/FOLLOW; with `SF_TRIGGER_PUSH_ONCE(1)`: `velocity += speed * movedir` then remove; else (push field, the surf booster case): `vecPush = speed * movedir; if (flags & FL_BASEVELOCITY) vecPush += basevelocity; basevelocity = vecPush; flags |= FL_BASEVELOCITY;` — i.e. it sets **basevelocity every frame you're inside**; pmove adds basevelocity into the move and the flag decays when you leave. Direction from `SetMovedir(pev)` (dlls/subs.cpp): `angles (0,-1,0) → movedir (0,0,1) up; (0,-2,0) → down; else movedir = forward vector of angles` (so `"angles" "-90 0 0"` = straight up — exactly what surf_ski_2's boosters use, speeds 1000–4000).
- **trigger_multiple/once**: generic `target` firing with `wait`; `trigger_gravity` sets toucher gravity.

## Point contents / water detection
World-only: `SV_PointContents` → `SV_HullPointContents(worldmodel->hulls[0], 0, p)`; currents (-9..-14) are mapped to `CONTENTS_WATER`; then non-solid contents entities are consulted: `SV_LinkContents` scans linked entities with `solid == SOLID_NOT` and a brush model (func_water/func_ladder/func_illusionary-with-skin), does the AABB check, then `SV_HullPointContents(entity hull0 via SV_HullForBsp(touch, 0,0, offset), p - offset) != CONTENTS_EMPTY` and returns `touch->v.skin` (the contents value, e.g. -3). pmove mirror: `PM_PointContents` / `PM_LinkContents` (`return pe->skin;`), and `PM_TruePointContents` returns raw world contents including currents.

Player water state — `PM_CheckWater` (ReGameDLL `pm_shared.cpp`), the function you need for `waterlevel`:
```c
point = origin + { (mins+maxs)/2 in x,y ; mins.z + 1 };   // just above the feet
cont = PM_PointContents(point, &truecont);
if (cont <= CONTENTS_WATER && cont > CONTENTS_TRANSLUCENT) {   // -3 .. -14
    watertype = cont; waterlevel = 1;
    point.z = origin.z + (mins.z + maxs.z) * 0.5;   // hull midpoint
    if (in water) waterlevel = 2;
    point.z = origin.z + view_ofs.z;                 // eyes
    if (in water) waterlevel = 3;
    // CONTENTS_CURRENT_* in truecont add 50*waterlevel*current_dir to basevelocity
}
```
waterlevel 0 = dry, 1 = feet, 2 = waist (swimming physics engage: `waterlevel > 1`), 3 = eyes under. In surf maps, water at the bottom of ramps is either worldspawn water brushes (leaf contents -3 in hull0 + hull leaves in clipnodes as non-solid CONTENTS_WATER) or `func_water` entities with `skin -3` (surf_ski_2 uses four func_water volumes).


## surf_map_conventions_markdown
# CS 1.6 surf map conventions (ground truth: parsed entity lump of surf_ski_2.bsp)

## What a classic surf map actually contains
Full classname census of **surf_ski_2** (the most-played 1.6 surf map; still #1 by server count today): `17 armoury_entity, 10 func_illusionary, 10 trigger_teleport, 10 info_target, 9 info_player_start, 9 info_player_deathmatch, 7 trigger_push, 6 func_ladder, 5 func_wall, 4 func_water, 2 func_breakable, 1 func_door, 1 func_door_rotating, 1 func_button, 1 func_pushable, 1 func_buyzone, light entities, worldspawn`.

Key takeaways for a standalone implementation:
- **Ramps are plain world geometry** (worldspawn brushes, often with func_wall/func_illusionary dressing). There is no special "surf entity": surfing emerges from air-strafe physics against sloped planes with server settings `sv_airaccelerate 100` (the defining surf cvar; default 10), `sv_gravity 800`, `sv_maxvelocity 2000`, usually `mp_footsteps 0`/`sv_maxspeed 320`.
- **trigger_teleport is the workhorse**: fail-teleports at the bottom of pits back to spawn/start (`target` → a targetnamed point entity — surf_ski_2 uses plain `info_target`, not `info_teleport_destination`; both work since the engine looks up by targetname), jail teleports (`"target" "jail"` sends caught players into a jail box; a floor trigger in jail (`"target" "jailed"`) keeps them there until freed via a button/door), gun-room teleports at the end of a lane (`"target" "groom"`), and secret-room teleports.
- **trigger_push** = boosters: brush over a surface with `"angles" "-90 0 0"` (straight up) or a yaw direction, `"speed" 1000–4000`. Continuous push fields set basevelocity each frame (see entities section).
- **trigger_hurt** = kill zones in many other surf maps (not in surf_ski_2, which uses jail/teleports instead); `"dmg"` large (e.g. 10000) for instant kill at the bottom. Negative dmg = heal pools, a common "spawn heal" convention.
- **func_water with `"skin" "-3"`** at the bottom of the arena (4 volumes in surf_ski_2) — soft landing + swim back.
- **Spawns:** `info_player_start` = **CT** spawns, `info_player_deathmatch` = **T** spawns (ReGameDLL `EntSelectSpawnPoint`: "the counter-terrorist spawns at info_player_start"). surf_ski_2 has 9 of each on one shared start platform at the top of the two symmetric lanes — the classic "both teams spawn together, race down mirrored ramps" convention. Each spawn has `origin` (z = floor + 36, since origin is hull center) and `angles`.
- **armoury_entity** (17 of them) = free weapons lying in the gun room / spawn; `func_buyzone` with `"team" "0"` (both teams) so buying works despite nonstandard layout.
- Start-zone convention in 1.6 skill-surf is informal (no dedicated start/end timer entities in the map — timers were AMX Mod X plugins keying off zones or teleport destinations). Linear "stage" maps chain: spawn platform → ramp section → catch platform with teleport to next stage → gun room/end arena.

## Good first maps for an RL agent (simple, verified names)
1. **surf_ski_2** — THE canonical CS 1.6 surf map: one huge room, two mirrored wide ramps, forgiving, water bottom, minimal routing. Best first target (this session's parsed copy came from ds-servers, direct zip: `https://en.ds-servers.com/maps/goldsrc/cstrike/surf_ski_2.zip`). Also see the browser remake at surfski2.com proving GoldSrc surf physics standalone is feasible.
2. **surf_ski** / **surf_ski_5** — same family, simpler/variant layouts.
3. **surf_egypt** — classic easy themed map, wide ramps.
4. **surf_green** — beginner staple.
5. **surf_iceday** / **surf_ice** — easy classics.
6. **surf_minilevels_v3** — short linear stages (good curriculum structure).
7. **surf_greatriver**, **surf_leet_xl** — bigger classics once the agent generalizes (combat-oriented, more geometry).

## Where to download .bsp files
- **ds-servers**: `https://en.ds-servers.com/maps/goldsrc/cstrike/<mapname>.zip` — direct zips containing `maps/<name>.bsp` (+ .res/.txt); verified working in this session for surf_ski_2.
- **GameBanana**: `https://gamebanana.com/games/48` (Counter-Strike 1.6) — maps section, search "surf".
- **GameMaps**: `https://www.gamemaps.com/details/12462` (surf_ski_2) and siblings.
- **varq.net / tsarvar** map DB: `https://varq.net/en/maps/counter-strike-1.6/mod:surf` — popularity-ranked list of surf maps with per-map pages and server fastdl links (93k-map database).
- **loadcs.com**: `https://loadcs.com/maps/download-Popular-Surf-maps` — curated popular surf pack.
(17buddies, the old canonical archive, is defunct.)

Note on textures: .bsp files may reference external WADs (`worldspawn "wad"` key — surf_ski_2 references halflife.wad but its TEXTURES lump is 1.4 MB, i.e. most miptex pixel data is embedded). For a visualizer, fall back to flat/checker colors when `miptex.offsets[0] == 0` and the WAD is absent — collision is never affected.


## sources
https://github.com/rehlds/ReHLDS — rehlds/engine/world.cpp (SV_RecursiveHullCheck, SV_HullForBsp, SV_HullForEntity, SV_SingleClipMoveToEntity, SV_ClipMoveToEntity, SV_Move, SV_TouchLinks, SV_LinkEdict, SV_HullPointContents, SV_PointContents, SV_LinkContents, SV_HullForBox, DIST_EPSILON)

https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds/public/rehlds/bspfile.h — lump indices and byte-exact dheader_t/lump_t/dplane_t/dnode_t/dclipnode_t/dleaf_t/dmodel_t/dface_t/dedge_t/texinfo_t/miptex_t

https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds/engine/model.cpp — Mod_LoadClipnodes (hull 1/2/3 clip_mins/clip_maxs), Mod_MakeHull0, submodel headnode wiring

https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds/engine/pmove.cpp — player_mins/player_maxs per usehull

https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds/engine/pmovetst.cpp — PM_HullForBsp usehull→BSP-hull map, _PM_PlayerTrace, PM_RecursiveHullCheck, PM_HullPointContents, PM_PointContents, PM_TruePointContents, PM_WaterEntity, PM_LinkContents

https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds/common/const.h — CONTENTS_*, SOLID_*, MOVETYPE_* values

https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds/common/com_model.h — hull_t, mplane_t, model_t runtime structs

https://raw.githubusercontent.com/rehlds/ReHLDS/master/rehlds/engine/world.h — MOVE_NORMAL/MOVE_NOMONSTERS/MOVE_MISSILE

https://github.com/rehlds/ReGameDLL_CS — regamedll/pm_shared/pm_shared.cpp (PM_CheckWater, usehull duck logic), pm_shared/pm_shared.h (PM_VEC_VIEW 17, PM_VEC_DUCK_VIEW 12, hull-min constants), pm_shared/pm_defs.h (PM_NORMAL etc.)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/triggers.cpp — InitTrigger, TeleportTouch, HurtTouch, CTriggerPush::Touch, CLadder (func_ladder), trigger_gravity

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/bmodels.cpp — func_wall, func_wall_toggle, func_illusionary, func_conveyor, func_monsterclip solidity

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/doors.cpp — func_water = CBaseDoor with skin!=0 → SOLID_NOT

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/subs.cpp — SetMovedir (trigger_push direction rules)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/dlls/player.cpp — EntSelectSpawnPoint (CT=info_player_start, T=info_player_deathmatch)

https://raw.githubusercontent.com/rehlds/ReGameDLL_CS/master/regamedll/common/pmtrace.h — pmtrace_t/pmplane_t

https://github.com/bernhardmgruber/hlbsp — src/bspdef.h, faithful port of the hlbsp SourceForge 'Unofficial BSP v30 File Format Specification' (original at http://hlbsp.sourceforge.net/index.php?content=bspdef, now offline)

https://developer.valvesoftware.com/wiki/BSP_(GoldSrc) — Valve Developer Community format reference (cross-check; fetch returned 403 this session)

https://developer.valvesoftware.com/wiki/Trigger_teleport_(GoldSrc)

surf_ski_2.bsp (v30) downloaded and parsed this session from https://en.ds-servers.com/maps/goldsrc/cstrike/surf_ski_2.zip — lump table + full entity lump dump

https://varq.net/en/maps/counter-strike-1.6/mod:surf — CS 1.6 surf map popularity list (surf_ski_2 #1)

https://loadcs.com/maps/download-Popular-Surf-maps

https://www.gamemaps.com/details/12462 — surf_ski_2

https://www.surfski2.com/ — browser GoldSrc surf reimplementation (precedent)

https://gamebanana.com/tuts/10630 — GoldSrc trigger setup tutorial

