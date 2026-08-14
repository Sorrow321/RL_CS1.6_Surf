/* surfcore.h — public C API of the surf simulation core (surfcore.dll)
 *
 * Contract for the Python layer (python/surfgym) and tests. See docs/03-env-design.md.
 * All arrays are caller-allocated. All floats are float32. Thread-safety: one SurfSim
 * may be stepped by one caller at a time; internally steps envs in parallel (OpenMP).
 *
 * Action encoding (int32 [num_envs x 5], see docs/03):
 *   a[0] yaw bin   0..14  -> yaw delta deg = YAW_BINS[a0] scaled so max == yaw_rate_max_deg.
 *                            ASCENDING: {-10,-7,-4,-2,-1,-0.5,-0.25, 0, 0.25,0.5,1,2,4,7,10}
 *                            (index 7 = 0 deg, index 14 = +10 deg at default rate)
 *   a[1] forward   0..2   -> forwardmove = {-400, 0, +400}[a1]
 *   a[2] side      0..2   -> sidemove    = {-400, 0, +400}[a2]
 *   a[3] jump      0..1   -> IN_JUMP held
 *   a[4] duck      0..1   -> IN_DUCK held (vanilla duck state machine when phys.enable_duck)
 *
 * Observation layout (float32 [num_envs x obs_dim], obs_dim = 14 + 6*lookahead_k; 62 at k=8):
 *   [0..2]   velocity in local yaw frame / 1000
 *   [3]      horizontal speed / 1000
 *   [4]      vertical velocity / 1000
 *   [5..7]   flags: onground, ducked, jump_held (0/1)
 *   [8]      signed lateral offset from spline / 500
 *   [9]      height offset from spline / 500
 *   [10]     fraction of track remaining (1 - progress/total_len)
 *   [11,12]  sin, cos of (yaw - spline tangent yaw)
 *   [13]     previous yaw action delta / yaw_rate_max_deg
 *   [14..]   lookahead_k blocks of 6: spline point rel. position in ego frame /2000 (3),
 *            spline tangent dir in ego frame (3)
 */
#ifndef SURFCORE_H
#define SURFCORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  ifdef SURFCORE_BUILD
#    define SURF_API __declspec(dllexport)
#  else
#    define SURF_API
#  endif
#else
#  define SURF_API __attribute__((visibility("default")))
#endif

/* usercmd button bits (HLSDK convention) */
#define SURF_IN_JUMP 2
#define SURF_IN_DUCK 4

typedef struct SurfPhys {
    float sv_gravity;         /* 800 */
    float sv_airaccelerate;   /* 100 (surf standard; engine default 10) */
    float sv_accelerate;      /* 5   (CS stock) */
    float sv_friction;        /* 4 */
    float edgefriction;       /* 2 */
    float sv_stopspeed;       /* 75  (CS stock) */
    float sv_maxspeed;        /* 320 */
    float player_maxspeed;    /* 250 (knife) */
    float sv_maxvelocity;     /* 2000, clamped per-axis */
    float sv_stepsize;        /* 18 */
    float sv_bounce;          /* 1 */
    int32_t msec;             /* 10 -> frametime = (float)(msec*0.001) */
    int32_t enable_stamina;   /* 1 = vanilla CS fuser2 */
    int32_t enable_bhop_cap;  /* 1 = vanilla PM_PreventMegaBunnyJumping (1.2f/0.8) */
    int32_t enable_duck;      /* 1 = vanilla PM_Duck/PM_UnDuck (docs/07) */
} SurfPhys;

typedef struct SurfEnvConfig {
    int32_t num_envs;
    int32_t max_episode_ticks;      /* truncation */
    int32_t lookahead_k;            /* 8 */
    float   lookahead_dt;           /* 0.25 s per lookahead point (arc length = speed*dt) */
    int32_t spawn_mode;             /* 0 = map spawns; 1 = curriculum (spline + map spawns mix) */
    float   spawn_curriculum_frac;  /* P(spline spawn) when spawn_mode==1, e.g. 0.99 */
    float   yaw_rate_max_deg;       /* 10 */
    float   yaw_jitter_deg;         /* 5 */
    float   kill_z;                 /* fail below this; <= -1e30f -> auto (map min z - 256) */
    SurfPhys phys;
} SurfEnvConfig;

/* Full per-env state — POD, for recording/curriculum/debug and single-step drivers.
 * Will grow duck/waterjump fields when tier-2 replay lands (docs/05). */
typedef struct SurfState {
    float origin[3];
    float velocity[3];
    float basevelocity[3];
    float yaw;                /* deg, kept in [0,360) */
    float fuser2;             /* stamina ms countdown */
    int32_t base_vel_flag;    /* FL_BASEVELOCITY equivalent */
    int32_t onground;         /* -1 airborne, else solid index (0 = world) */
    int32_t oldbuttons;       /* previous cmd buttons (jump edge trigger) */
    int32_t ducked;           /* reserved; 0 while enable_duck == 0 */
    int32_t tick;             /* ticks since episode start */
    int32_t seg_hint;         /* spline segment window anchor */
    float progress;           /* arc length along spline */
    float best_progress;
    int32_t stuck_ticks;      /* consecutive stuck/allsolid ticks (5 -> fail) */
    int32_t induck;           /* bInDuck: duck transition in progress */
    float duck_time;          /* flDuckTime ms countdown (1000 at press) */
} SurfState;

typedef struct SurfTrace {
    float endpos[3];
    float normal[3];          /* faces the motion */
    float dist;
    float fraction;           /* 1.0 = clean */
    int32_t startsolid;
    int32_t allsolid;
    int32_t ent;              /* -1 none, 0 world, >0 solid brush entity index */
} SurfTrace;

typedef struct SurfSim SurfSim;

/* ---- lifecycle ---------------------------------------------------------- */
SURF_API SurfSim* surf_create(const char* bsp_path, const SurfEnvConfig* cfg,
                              char* err, int32_t errlen);   /* NULL on error, err filled */
SURF_API void     surf_destroy(SurfSim* s);
SURF_API int32_t  surf_obs_dim(const SurfSim* s);
SURF_API int32_t  surf_num_envs(const SurfSim* s);

/* Track polyline in map units; >= 2 points. Returns 0 ok, -1 bad input. */
SURF_API int32_t  surf_set_waypoints(SurfSim* s, const float* xyz, int32_t count);

/* ---- hot path ----------------------------------------------------------- */
/* Same-step autoreset: done envs are reset in place; obs row = NEW episode's first obs,
 * terminal_obs row = the ended episode's true final obs (rows for non-done envs untouched). */
SURF_API void surf_reset_all(SurfSim* s, uint64_t seed, float* obs);
SURF_API void surf_step(SurfSim* s, const int32_t* actions,
                        float* obs, float* rewards, uint8_t* done, uint8_t* trunc,
                        float* terminal_obs);

/* ---- state access ------------------------------------------------------- */
SURF_API void surf_get_states(SurfSim* s, SurfState* out /* [num_envs] */);
SURF_API void surf_set_state(SurfSim* s, int32_t env, const SurfState* st);

/* ---- exposed internals (tests, tools, parity) --------------------------- */
SURF_API void    surf_trace(SurfSim* s, const float* start, const float* end,
                            int32_t usehull /* 0 stand, 1 duck, 2 point */, SurfTrace* out);
SURF_API int32_t surf_point_contents(SurfSim* s, const float* p); /* CONTENTS_* (world + contents ents) */

/* One physics tick at usercmd level — the parity-harness primitive (docs/05 tier 2). */
SURF_API void surf_pm_step_usercmd(SurfSim* s, SurfState* st, float yaw, float pitch,
                                   float fmove, float smove, int32_t buttons, int32_t msec);

/* One INTERACTIVE tick: usercmd step + live map triggers (teleports/boosters/hurt),
 * using env slot 0's push-once bookkeeping. For the human play client.
 * Returns flags: 1 = teleported, 2 = killed (trigger_hurt / destless teleport),
 * 4 = waterlevel >= 2 (swimming, unsupported), 8 = stuck/trapped in solid. */
SURF_API int32_t surf_play_step(SurfSim* s, SurfState* st, float yaw, float pitch,
                                float fmove, float smove, int32_t buttons, int32_t msec);
/* Thin discretizing wrapper over the above using the action encoding at top. */
SURF_API void surf_pm_step_single(SurfSim* s, SurfState* st, const int32_t action[5]);

/* ---- map info ----------------------------------------------------------- */
SURF_API int32_t surf_num_spawns(const SurfSim* s);
SURF_API void    surf_get_spawn(const SurfSim* s, int32_t i, float* origin /*3*/, float* yaw);
SURF_API void    surf_map_bounds(const SurfSim* s, float* mins /*3*/, float* maxs /*3*/);

#ifdef __cplusplus
}
#endif
#endif /* SURFCORE_H */
