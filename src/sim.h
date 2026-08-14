/* sim.h — internal structures shared by bsp.c / trace.c / pm.c / env.c.
 * Layouts of B* structs are byte-compatible with the BSP v30 lumps (docs/02 §1)
 * so lumps are memcpy'd straight in. Little-endian x64 assumed. */
#ifndef SIM_H
#define SIM_H

#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "surfcore.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- contents / misc engine constants (common/const.h) ------------------ */
#define CONTENTS_EMPTY        (-1)
#define CONTENTS_SOLID        (-2)
#define CONTENTS_WATER        (-3)
#define CONTENTS_SLIME        (-4)
#define CONTENTS_LAVA         (-5)
#define CONTENTS_SKY          (-6)
#define CONTENTS_CURRENT_0    (-9)
#define CONTENTS_CURRENT_DOWN (-14)
#define CONTENTS_TRANSLUCENT  (-15)
#define CONTENTS_LADDER       (-16)

#define DIST_EPSILON   0.03125f
#define STOP_EPSILON   0.1
#define MAX_CLIP_PLANES 5

/* ---- BSP lump structs (byte-exact) -------------------------------------- */
#pragma pack(push, 1)
typedef struct { int32_t fileofs, filelen; } BLump;
typedef struct { int32_t version; BLump lumps[15]; } BHeader;            /* 124 */
typedef struct { float normal[3]; float dist; int32_t type; } BPlane;    /* 20 */
typedef struct { int32_t planenum; int16_t children[2]; } BClipNode;     /* 8; children<0 = CONTENTS_* */
typedef struct {                                                         /* 24 */
    int32_t planenum; int16_t children[2];
    int16_t mins[3], maxs[3]; uint16_t firstface, numfaces;
} BNode;
typedef struct {                                                         /* 28 */
    int32_t contents, visofs; int16_t mins[3], maxs[3];
    uint16_t firstmarksurface, nummarksurfaces; uint8_t ambient[4];
} BLeaf;
typedef struct {                                                         /* 64 */
    float mins[3], maxs[3], origin[3];
    int32_t headnode[4]; int32_t visleafs, firstface, numfaces;
} BModel;
#pragma pack(pop)

#define LUMP_ENTITIES  0
#define LUMP_PLANES    1
#define LUMP_NODES     5
#define LUMP_CLIPNODES 9
#define LUMP_LEAFS     10
#define LUMP_MODELS    14

/* ---- runtime hull ------------------------------------------------------- */
typedef struct {
    const BClipNode* clipnodes;      /* shared clipnode array (or hull0 mirror) */
    const BPlane*    planes;
    int32_t firstclipnode, lastclipnode;
    float clip_mins[3], clip_maxs[3];
} Hull;

/* ---- parsed entities ---------------------------------------------------- */
typedef enum { TRIG_TELEPORT = 0, TRIG_HURT = 1, TRIG_PUSH = 2 } TrigKind;

typedef struct {
    int32_t kind;                    /* TrigKind */
    int32_t model;                   /* *N brush model index */
    float origin[3];                 /* entity origin (usually 0 for brush ents) */
    float absmin[3], absmax[3];      /* model bounds + origin, padded +-1 */
    /* teleport */
    int32_t has_dest;
    float dest[3], dest_yaw;
    float dest_progress;             /* filled by surf_set_waypoints */
    /* hurt */
    float dmg;
    /* push */
    float pushvel[3];                /* movedir * speed (SetMovedir rules) */
    int32_t push_once;               /* SF_TRIGGER_PUSH_ONCE */
} Trigger;

typedef struct { int32_t model; float origin[3]; float absmin[3], absmax[3]; } SolidEnt;
typedef struct { int32_t model; float origin[3]; int32_t skin; float absmin[3], absmax[3]; } ContentsEnt;
typedef struct { float origin[3]; float yaw; } SpawnPoint;

#define MAX_SOLIDS   128
#define MAX_TRIGGERS 128
#define MAX_CONTENTS 64
#define MAX_SPAWNS   64

typedef struct BspMap {
    BPlane*    planes;      int32_t numplanes;
    BClipNode* clipnodes;   int32_t numclipnodes;
    BClipNode* hull0nodes;  int32_t numhull0;    /* MakeHull0 mirror of NODES */
    BModel*    models;      int32_t nummodels;
    char*      entstring;   int32_t entlen;

    SolidEnt    solids[MAX_SOLIDS];     int32_t numsolids;      /* excludes world (implicit idx 0) */
    Trigger     triggers[MAX_TRIGGERS]; int32_t numtriggers;
    ContentsEnt contents[MAX_CONTENTS]; int32_t numcontents;
    SpawnPoint  spawns[MAX_SPAWNS];     int32_t numspawns;

    float world_mins[3], world_maxs[3];
} BspMap;

/* engine player hulls, indexed by pmove usehull (0 stand, 1 duck, 2 point, 3 large) */
extern const float g_player_mins[4][3];
extern const float g_player_maxs[4][3];

/* ---- traces ------------------------------------------------------------- */
typedef struct {
    int32_t allsolid, startsolid, inopen, inwater;
    float fraction;
    float endpos[3];
    float plane_normal[3];
    float plane_dist;
    int32_t ent;                     /* -1 none, 0 world, 1.. = solids[ent-1] */
} PmTrace;

/* bsp.c */
int  bsp_load(BspMap* m, const char* path, char* err, int errlen);      /* 0 ok */
void bsp_free(BspMap* m);
/* Hull of (model, bsp hull index 0..3); model referenced by BModel*. */
void bsp_model_hull(const BspMap* m, const BModel* mod, int bsphull, Hull* out);

/* trace.c */
/* usehull: 0 stand, 1 duck, 2 point. Traces world + all solid brush ents. */
void trace_player(const BspMap* m, int usehull, const float* start, const float* end, PmTrace* tr);
int  hull_point_contents(const Hull* h, int num, const float* p);
int  point_contents(const BspMap* m, const float* p);                    /* world hull0 + contents ents */
int  trigger_contains(const BspMap* m, const Trigger* t, int usehull, const float* origin);

/* pm.c — one full player-movement tick on a SurfState (no env logic).
 * PmPersist carries per-player stuck bookkeeping the engine keeps in statics. */
typedef struct { int32_t stuck_idx; int32_t stuck_last_tick; } PmPersist;
void pm_init(void);   /* build the stuck table; MUST run before any parallel pm_tick */
void pm_tick(const BspMap* m, const SurfPhys* ph, SurfState* st, PmPersist* pp,
             float yaw, float pitch, float fmove, float smove, int buttons, int msec,
             int* out_waterlevel, int* out_blocked_solid);

/* small vector helpers (float; see pm.c for double-discipline sites) */
static __inline void v3copy(float* d, const float* s) { d[0]=s[0]; d[1]=s[1]; d[2]=s[2]; }
static __inline void v3zero(float* d) { d[0]=d[1]=d[2]=0.0f; }

#endif /* SIM_H */
