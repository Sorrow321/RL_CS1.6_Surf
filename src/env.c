/* env.c — vectorized RL environment over pm.c. Spec: docs/03-env-design.md.
 * Batch step is OpenMP-parallel; envs never interact. Same-step autoreset with
 * terminal_obs. Triggers run AFTER the physics tick (docs/01 §2 step 5). */
#include <stdlib.h>
#include <stdio.h>
#include "sim.h"

#define OBS_FIXED 15

typedef struct SurfSim {
    BspMap map;
    SurfEnvConfig cfg;
    int obs_dim;
    float frametime;
    float kill_z;
    /* spline */
    float* wp; int nwp;            /* waypoints xyz */
    float* cum;                    /* cumulative arc length per vertex */
    float* tang;                   /* unit tangent per segment [3*(nwp-1)] */
    float total_len;
    /* spawn pool (spawn_mode 2) */
    SurfState* spawn_pool;
    int32_t spawn_pool_n;
    float map_center[3];
    /* per env */
    SurfState* st;
    PmPersist* pp;
    float* last_yaw_delta;
    float* last_pitch_delta;
    uint64_t* rng;
    uint64_t (*once_used)[2];      /* consumed SF_TRIGGER_PUSH_ONCE triggers per env */
    PmPersist single_pp;           /* for the single-step API (single caller) */
} SurfSim;

/* yaw action bins, ASCENDING (deg at yaw_rate_max_deg = 10): index 7 = 0, 14 = +10.
 * Must match surfgym.core.YAW_BINS. */
static const float YAW_BINS[15] = { -10.0f, -7.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.25f,
                                    0.0f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 7.0f, 10.0f };

/* pitch action bins, ASCENDING (deg at pitch_rate_max_deg = 10): index 3 = 0.
 * Must match surfgym.core.PITCH_BINS. */
static const float PITCH_BINS[7] = { -10.0f, -5.0f, -2.0f, 0.0f, 2.0f, 5.0f, 10.0f };

/* ---- rng ----------------------------------------------------------------- */
static uint64_t sm64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static float frand01(uint64_t* s) { return (float)((sm64(s) >> 40) * (1.0 / 16777216.0)); }

/* ---- spline -------------------------------------------------------------- */
static void spline_free(SurfSim* s) {
    free(s->wp); free(s->cum); free(s->tang);
    s->wp = NULL; s->cum = NULL; s->tang = NULL; s->nwp = 0; s->total_len = 0;
}

/* closest point on the polyline near seg_hint; returns progress, updates hint,
 * fills lateral (signed horizontal) and height offsets and tangent yaw */
static float spline_project(const SurfSim* s, const float* p, int32_t* hint,
                            float* lateral, float* height, float* tan_out) {
    if (s->nwp < 2) { *lateral = 0; *height = 0; tan_out[0] = 1; tan_out[1] = 0; tan_out[2] = 0; return 0; }
    int nseg = s->nwp - 1;
    if (*hint < 0) *hint = 0;
    if (*hint > nseg - 1) *hint = nseg - 1;          /* stale hints must not empty the window */
    int lo = *hint - 2, hi = *hint + 4;
    if (lo < 0) lo = 0;
    if (hi > nseg - 1) hi = nseg - 1;
    float bestd = 1e30f, bestprog = 0, bestlat = 0, besth = 0;
    int bestseg = *hint;
    const float* bt = &s->tang[0];
    for (int seg = lo; seg <= hi; seg++) {
        const float* a = &s->wp[3 * seg];
        const float* b = &s->wp[3 * (seg + 1)];
        float ab[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
        float ap[3] = { p[0]-a[0], p[1]-a[1], p[2]-a[2] };
        float len2 = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2];
        float t = len2 > 0 ? (ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2]) / len2 : 0;
        if (t < 0) t = 0; else if (t > 1) t = 1;
        float q[3] = { a[0]+ab[0]*t, a[1]+ab[1]*t, a[2]+ab[2]*t };
        float d[3] = { p[0]-q[0], p[1]-q[1], p[2]-q[2] };
        float dist2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        if (dist2 < bestd) {
            bestd = dist2;
            bestseg = seg;
            bestprog = s->cum[seg] + sqrtf(len2) * t;
            const float* tg = &s->tang[3 * seg];
            /* signed horizontal lateral: cross(tangent_xy, d_xy).z */
            bestlat = tg[0]*d[1] - tg[1]*d[0];
            besth = d[2];
            bt = tg;
        }
    }
    *hint = bestseg;
    *lateral = bestlat;
    *height = besth;
    tan_out[0] = bt[0]; tan_out[1] = bt[1]; tan_out[2] = bt[2];
    return bestprog;
}

/* position + tangent at arc distance u (clamped) */
static void spline_sample(const SurfSim* s, float u, float* pos, float* tang, int* seg_out) {
    if (s->nwp < 2) { pos[0]=pos[1]=pos[2]=0; tang[0]=1; tang[1]=0; tang[2]=0; if (seg_out) *seg_out = 0; return; }
    if (u < 0) u = 0;
    if (u > s->total_len) u = s->total_len;
    int nseg = s->nwp - 1, seg = 0;
    while (seg < nseg - 1 && s->cum[seg + 1] < u) seg++;
    float seglen = s->cum[seg + 1] - s->cum[seg];
    float t = seglen > 0 ? (u - s->cum[seg]) / seglen : 0;
    const float* a = &s->wp[3 * seg];
    const float* b = &s->wp[3 * (seg + 1)];
    for (int i = 0; i < 3; i++) pos[i] = a[i] + (b[i] - a[i]) * t;
    v3copy(tang, &s->tang[3 * seg]);
    if (seg_out) *seg_out = seg;
}

int32_t surf_set_waypoints(SurfSim* s, const float* xyz, int32_t count) {
    if (!xyz || count < 2) return -1;
    spline_free(s);
    s->wp = (float*)malloc(sizeof(float) * 3 * (size_t)count);
    s->cum = (float*)malloc(sizeof(float) * (size_t)count);
    s->tang = (float*)malloc(sizeof(float) * 3 * (size_t)count);
    if (!s->wp || !s->cum || !s->tang) { spline_free(s); return -1; }
    memcpy(s->wp, xyz, sizeof(float) * 3 * (size_t)count);
    s->nwp = count;
    s->cum[0] = 0;
    for (int i = 0; i < count - 1; i++) {
        float d[3] = { s->wp[3*(i+1)] - s->wp[3*i], s->wp[3*(i+1)+1] - s->wp[3*i+1], s->wp[3*(i+1)+2] - s->wp[3*i+2] };
        float len = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        if (len < 1e-3f) len = 1e-3f;
        s->tang[3*i] = d[0]/len; s->tang[3*i+1] = d[1]/len; s->tang[3*i+2] = d[2]/len;
        s->cum[i+1] = s->cum[i] + len;
    }
    v3copy(&s->tang[3*(count-1)], &s->tang[3*(count-2)]);
    s->total_len = s->cum[count - 1];
    /* teleport destination progress (global projection, no hint window) */
    for (int i = 0; i < s->map.numtriggers; i++) {
        Trigger* t = &s->map.triggers[i];
        if (t->kind != TRIG_TELEPORT || !t->has_dest) continue;
        float best = 1e30f; float prog = 0;
        for (int seg = 0; seg < count - 1; seg++) {
            int32_t h = seg; float lat, hgt, tg[3];
            /* reuse projector with a forced window of one segment */
            int32_t hh = seg;
            (void)h;
            const float* a = &s->wp[3*seg]; const float* b = &s->wp[3*(seg+1)];
            float ab[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
            float ap[3] = { t->dest[0]-a[0], t->dest[1]-a[1], t->dest[2]-a[2] };
            float len2 = ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2];
            float tt = len2 > 0 ? (ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2])/len2 : 0;
            if (tt < 0) tt = 0; else if (tt > 1) tt = 1;
            float q[3] = { a[0]+ab[0]*tt, a[1]+ab[1]*tt, a[2]+ab[2]*tt };
            float dx=t->dest[0]-q[0], dy=t->dest[1]-q[1], dz=t->dest[2]-q[2];
            float d2 = dx*dx+dy*dy+dz*dz;
            if (d2 < best) { best = d2; prog = s->cum[seg] + sqrtf(len2)*tt; }
            (void)lat; (void)hgt; (void)tg; (void)hh;
        }
        t->dest_progress = prog;
    }
    /* re-clamp every env's hint into the new spline and reproject its progress */
    for (int e = 0; e < s->cfg.num_envs; e++) {
        SurfState* st = &s->st[e];
        if (st->seg_hint < 0) st->seg_hint = 0;
        if (st->seg_hint > count - 2) st->seg_hint = count - 2;
        float lat, hgt, tg[3];
        st->progress = spline_project(s, st->origin, &st->seg_hint, &lat, &hgt, tg);
        if (st->best_progress > s->total_len) st->best_progress = st->progress;
    }
    return 0;
}

/* ---- observation --------------------------------------------------------- */
/* Depth "eyes": lidar_h x lidar_w point-hull rays around (yaw, pitch) from eye
 * height (GoldSrc view offsets: 17 standing, 12 ducked). Row 0 looks up,
 * col 0 looks left; value = hit distance / lidar_range, 1.0 = clear. */
static void write_lidar(const SurfSim* s, const SurfState* st, float* o) {
    int W = s->cfg.lidar_w, H = s->cfg.lidar_h;
    if (W <= 0 || H <= 0) return;
    float eye[3];
    v3copy(eye, st->origin);
    eye[2] += st->ducked ? 12.0f : 17.0f;
    float d2r = (float)(M_PI / 180.0);
    float range = s->cfg.lidar_range > 1.0f ? s->cfg.lidar_range : 2000.0f;
    for (int r = 0; r < H; r++) {
        float poff = s->cfg.lidar_vfov_deg * (0.5f - (r + 0.5f) / (float)H);
        float p = (st->pitch + poff) * d2r;
        float cp = cosf(p), sp = sinf(p);
        for (int c = 0; c < W; c++) {
            float yoff = s->cfg.lidar_hfov_deg * (0.5f - (c + 0.5f) / (float)W);
            float yw = (st->yaw + yoff) * d2r;
            float end[3] = { eye[0] + cp * cosf(yw) * range,
                             eye[1] + cp * sinf(yw) * range,
                             eye[2] + sp * range };
            PmTrace tr;
            trace_player(&s->map, 2, eye, end, &tr);   /* 2 = point hull */
            o[r * W + c] = tr.startsolid ? 0.0f : tr.fraction;
        }
    }
}

static void write_obs(const SurfSim* s, const SurfState* st, float last_yd,
                      float last_pd, float* o) {
    float yawr = st->yaw * (float)(M_PI / 180.0);
    float cy = cosf(yawr), sy = sinf(yawr);
    /* ego frame: x' = along yaw, y' = left */
    float vx = st->velocity[0], vy = st->velocity[1], vz = st->velocity[2];
    o[0] = ( vx * cy + vy * sy) / 1000.0f;
    o[1] = (-vx * sy + vy * cy) / 1000.0f;
    o[2] = vz / 1000.0f;
    o[3] = sqrtf(vx*vx + vy*vy) / 1000.0f;
    o[4] = st->onground != -1 ? 1.0f : 0.0f;
    o[5] = st->ducked ? 1.0f : 0.0f;
    o[6] = (st->oldbuttons & SURF_IN_JUMP) ? 1.0f : 0.0f;
    o[7] = sy;                                   /* sin(yaw): absolute heading */
    o[8] = cy;
    o[9] = st->pitch / 90.0f;
    o[10] = s->cfg.yaw_rate_max_deg > 0 ? last_yd / s->cfg.yaw_rate_max_deg : 0.0f;
    o[11] = s->cfg.pitch_rate_max_deg > 0 ? last_pd / s->cfg.pitch_rate_max_deg : 0.0f;
    o[12] = (st->origin[0] - s->map_center[0]) / 2000.0f;
    o[13] = (st->origin[1] - s->map_center[1]) / 2000.0f;
    o[14] = (st->origin[2] - s->map_center[2]) / 2000.0f;
    write_lidar(s, st, &o[OBS_FIXED]);
}

/* ---- spawn / reset ------------------------------------------------------- */
static float wrap_yaw(float y) {
    y = fmodf(y, 360.0f);
    if (y < 0) y += 360.0f;
    return y;
}

static void reset_env(SurfSim* s, int i) {
    SurfState* st = &s->st[i];
    uint64_t* r = &s->rng[i];
    memset(st, 0, sizeof(*st));
    memset(&s->pp[i], 0, sizeof(PmPersist));
    s->once_used[i][0] = s->once_used[i][1] = 0;
    s->last_yaw_delta[i] = 0;
    s->last_pitch_delta[i] = 0;
    st->onground = -1;
    if (s->cfg.spawn_mode == 2 && s->spawn_pool_n > 0) {
        int k = (int)(sm64(r) % (uint64_t)s->spawn_pool_n);
        *st = s->spawn_pool[k];
        st->tick = 0;
        st->stuck_ticks = 0;
        st->yaw = wrap_yaw(st->yaw + s->cfg.yaw_jitter_deg * (2.0f * frand01(r) - 1.0f));
        if (s->nwp >= 2) {
            float lat, hgt, tg[3];
            if (st->seg_hint < 0 || st->seg_hint > s->nwp - 2) st->seg_hint = 0;
            st->progress = spline_project(s, st->origin, &st->seg_hint, &lat, &hgt, tg);
        } else {
            st->progress = 0;
        }
        st->best_progress = st->progress;
        return;
    }
    int curriculum = (s->cfg.spawn_mode == 1 && s->nwp >= 2 &&
                      frand01(r) < s->cfg.spawn_curriculum_frac);
    if (curriculum) {
        float u = frand01(r) * s->total_len * 0.98f;
        float pos[3], tg[3]; int seg;
        spline_sample(s, u, pos, tg, &seg);
        st->origin[0] = pos[0]; st->origin[1] = pos[1]; st->origin[2] = pos[2] + 20.0f;
        st->velocity[0] = tg[0] * 250.0f;
        st->velocity[1] = tg[1] * 250.0f;
        st->velocity[2] = tg[2] * 250.0f;
        st->yaw = wrap_yaw(atan2f(tg[1], tg[0]) * (float)(180.0 / M_PI)
                           + s->cfg.yaw_jitter_deg * (2.0f * frand01(r) - 1.0f));
        st->seg_hint = seg;
        st->progress = st->best_progress = u;
        /* settle onto walkable ground if it is just below (not onto ramps) */
        float down[3] = { st->origin[0], st->origin[1], st->origin[2] - 40.0f };
        PmTrace tr;
        trace_player(&s->map, 0, st->origin, down, &tr);
        if (tr.startsolid) st->origin[2] += 40.0f;
        else if (tr.fraction < 1.0f && tr.plane_normal[2] >= 0.7f) v3copy(st->origin, tr.endpos);
    } else {
        int n = s->map.numspawns > 0 ? s->map.numspawns : 1;
        int k = (int)(sm64(r) % (uint64_t)n);
        if (s->map.numspawns > 0) {
            v3copy(st->origin, s->map.spawns[k].origin);
            st->yaw = wrap_yaw(s->map.spawns[k].yaw
                               + s->cfg.yaw_jitter_deg * (2.0f * frand01(r) - 1.0f));
        }
        float lat, hgt, tg[3];
        st->seg_hint = 0;
        st->progress = st->best_progress = spline_project(s, st->origin, &st->seg_hint, &lat, &hgt, tg);
    }
}

/* ---- triggers (post-move; docs/01 §2 step 5) ----------------------------- */
/* returns: 0 none, 1 fail, 2 teleported (benign) */
static int apply_triggers(SurfSim* s, int env, SurfState* st) {
    const BspMap* m = &s->map;
    int uh = (s->cfg.phys.enable_duck && st->ducked) ? 1 : 0;
    for (int i = 0; i < m->numtriggers; i++) {
        const Trigger* t = &m->triggers[i];
        /* player AABB vs trigger AABB precheck */
        if (st->origin[0] + g_player_maxs[uh][0] < t->absmin[0] || st->origin[0] + g_player_mins[uh][0] > t->absmax[0] ||
            st->origin[1] + g_player_maxs[uh][1] < t->absmin[1] || st->origin[1] + g_player_mins[uh][1] > t->absmax[1] ||
            st->origin[2] + g_player_maxs[uh][2] < t->absmin[2] || st->origin[2] + g_player_mins[uh][2] > t->absmax[2])
            continue;
        if (!trigger_contains(m, t, uh, st->origin)) continue;

        if (t->kind == TRIG_HURT) {
            if (t->dmg >= 90.0f) return 1;
        } else if (t->kind == TRIG_PUSH) {
            if (t->push_once) {
                uint64_t* used = &s->once_used[env][i >> 6];
                uint64_t bit = 1ULL << (i & 63);
                if (!(*used & bit)) {
                    *used |= bit;
                    st->velocity[0] += t->pushvel[0];
                    st->velocity[1] += t->pushvel[1];
                    st->velocity[2] += t->pushvel[2];
                    if (t->pushvel[2] > 0) st->onground = -1;
                }
            } else {
                float push[3];
                v3copy(push, t->pushvel);
                if (st->base_vel_flag) {
                    push[0] += st->basevelocity[0];
                    push[1] += st->basevelocity[1];
                    push[2] += st->basevelocity[2];
                }
                v3copy(st->basevelocity, push);
                st->base_vel_flag = 1;
            }
        } else if (t->kind == TRIG_TELEPORT) {
            if (!t->has_dest) return 1;
            if (s->nwp >= 2 && t->dest_progress < st->progress - 500.0f) return 1;   /* jail/restart */
            /* benign teleport: TeleportTouch */
            st->origin[0] = t->dest[0];
            st->origin[1] = t->dest[1];
            st->origin[2] = t->dest[2] - g_player_mins[uh][2]; /* origin at hull center (current hull) */
            st->origin[2] += 1.0f;
            st->yaw = wrap_yaw(t->dest_yaw);
            v3zero(st->velocity);
            v3zero(st->basevelocity);
            st->base_vel_flag = 0;
            st->onground = -1;
            if (s->nwp >= 2) {
                /* remap via the GLOBAL dest projection (computed in surf_set_waypoints) —
                 * a windowed re-projection from a stale hint collapses on far stage jumps */
                int nseg = s->nwp - 1, seg = 0;
                while (seg < nseg - 1 && s->cum[seg + 1] < t->dest_progress) seg++;
                st->seg_hint = seg;
                st->progress = t->dest_progress;
            }
            return 2;
        }
    }
    return 0;
}

/* ---- lifecycle ----------------------------------------------------------- */
int32_t surf_abi_version(void) { return SURF_ABI_VERSION; }

SurfSim* surf_create(const char* bsp_path, const SurfEnvConfig* cfg, char* err, int32_t errlen) {
    SurfSim* s = (SurfSim*)calloc(1, sizeof(SurfSim));
    if (!s) return NULL;
    if (bsp_load(&s->map, bsp_path, err, errlen) != 0) { free(s); return NULL; }
    pm_init();                                 /* stuck table, before any parallel stepping */
    s->cfg = *cfg;
    if (s->cfg.num_envs < 1) s->cfg.num_envs = 1;
    if (s->cfg.lookahead_k < 0) s->cfg.lookahead_k = 0;
    if (s->cfg.phys.msec < 1) s->cfg.phys.msec = 1;
    if (s->cfg.phys.msec > 50) s->cfg.phys.msec = 50;   /* engine chops >50ms; batch path clamps */
    if (s->cfg.lidar_w < 0) s->cfg.lidar_w = 0;
    if (s->cfg.lidar_h < 0) s->cfg.lidar_h = 0;
    if (s->cfg.lidar_w == 0 || s->cfg.lidar_h == 0) s->cfg.lidar_w = s->cfg.lidar_h = 0;
    if (s->cfg.lidar_range <= 1.0f) s->cfg.lidar_range = 2000.0f;
    /* 0 is meaningful: pitch frozen at its spawn value (fixed-gaze mode);
     * only negative means "use the default" */
    if (s->cfg.pitch_rate_max_deg < 0.0f) s->cfg.pitch_rate_max_deg = 10.0f;
    s->obs_dim = OBS_FIXED + s->cfg.lidar_w * s->cfg.lidar_h;
    s->frametime = (float)(s->cfg.phys.msec * 0.001);
    s->kill_z = s->cfg.kill_z <= -1e30f ? s->map.world_mins[2] - 256.0f : s->cfg.kill_z;
    for (int k = 0; k < 3; k++)
        s->map_center[k] = 0.5f * (s->map.world_mins[k] + s->map.world_maxs[k]);
    int n = s->cfg.num_envs;
    s->st = (SurfState*)calloc((size_t)n, sizeof(SurfState));
    s->pp = (PmPersist*)calloc((size_t)n, sizeof(PmPersist));
    s->last_yaw_delta = (float*)calloc((size_t)n, sizeof(float));
    s->last_pitch_delta = (float*)calloc((size_t)n, sizeof(float));
    s->rng = (uint64_t*)calloc((size_t)n, sizeof(uint64_t));
    s->once_used = (uint64_t(*)[2])calloc((size_t)n, sizeof(uint64_t[2]));
    if (!s->st || !s->pp || !s->last_yaw_delta || !s->last_pitch_delta ||
        !s->rng || !s->once_used) {
        surf_destroy(s);
        if (err && errlen > 0) { strncpy(err, "oom", (size_t)errlen - 1); err[errlen-1] = 0; }
        return NULL;
    }
    return s;
}

void surf_destroy(SurfSim* s) {
    if (!s) return;
    bsp_free(&s->map);
    spline_free(s);
    free(s->spawn_pool);
    free(s->st); free(s->pp); free(s->last_yaw_delta); free(s->last_pitch_delta);
    free(s->rng); free(s->once_used);
    free(s);
}

int32_t surf_set_spawn_pool(SurfSim* s, const SurfState* pool, int32_t n) {
    if (!pool || n < 1) return -1;
    SurfState* copy = (SurfState*)malloc(sizeof(SurfState) * (size_t)n);
    if (!copy) return -1;
    memcpy(copy, pool, sizeof(SurfState) * (size_t)n);
    free(s->spawn_pool);
    s->spawn_pool = copy;
    s->spawn_pool_n = n;
    return 0;
}

int32_t surf_obs_dim(const SurfSim* s) { return s->obs_dim; }
int32_t surf_num_envs(const SurfSim* s) { return s->cfg.num_envs; }
int32_t surf_num_spawns(const SurfSim* s) { return s->map.numspawns; }
void surf_get_spawn(const SurfSim* s, int32_t i, float* origin, float* yaw) {
    if (i < 0 || i >= s->map.numspawns) { origin[0]=origin[1]=origin[2]=0; *yaw = 0; return; }
    v3copy(origin, s->map.spawns[i].origin);
    *yaw = s->map.spawns[i].yaw;
}
void surf_map_bounds(const SurfSim* s, float* mins, float* maxs) {
    v3copy(mins, s->map.world_mins);
    v3copy(maxs, s->map.world_maxs);
}

void surf_reset_all(SurfSim* s, uint64_t seed, float* obs) {
    for (int i = 0; i < s->cfg.num_envs; i++) {
        /* per-env stream derived through the mixer — a raw gamma-stride seed makes
         * neighboring envs' streams identical one position apart */
        uint64_t t = seed + (uint64_t)i;
        s->rng[i] = sm64(&t) | 1;
        reset_env(s, i);
        write_obs(s, &s->st[i], 0.0f, 0.0f, &obs[(size_t)i * s->obs_dim]);
    }
}

/* ---- the hot path -------------------------------------------------------- */
void surf_step(SurfSim* s, const int32_t* actions,
               float* obs, float* rewards, uint8_t* done, uint8_t* trunc,
               float* terminal_obs) {
    int n = s->cfg.num_envs;
    int i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++) {
        SurfState* st = &s->st[i];
        const int32_t* a = &actions[(size_t)i * 6];

        /* decode action */
        int yb = a[0] < 0 ? 0 : (a[0] > 14 ? 14 : a[0]);
        float yd = YAW_BINS[yb] * (s->cfg.yaw_rate_max_deg / 10.0f);
        int pb = a[1] < 0 ? 0 : (a[1] > 6 ? 6 : a[1]);
        float pd = PITCH_BINS[pb] * (s->cfg.pitch_rate_max_deg / 10.0f);
        float fmove = (a[2] <= 0) ? -400.0f : (a[2] >= 2 ? 400.0f : 0.0f);
        float smove = (a[3] <= 0) ? -400.0f : (a[3] >= 2 ? 400.0f : 0.0f);
        int buttons = 0;
        if (a[4]) buttons |= SURF_IN_JUMP;
        if (a[5]) buttons |= SURF_IN_DUCK;   /* inert while enable_duck == 0 */
        st->yaw = wrap_yaw(st->yaw + yd);
        s->last_yaw_delta[i] = yd;
        st->pitch += pd;
        /* asymmetric view clamp: staring at featureless sky is a sensor-
         * collapse attractor (constant input -> zero gradient on where to
         * look -> permanently blind). +30 up keeps 15 deg of below-horizon
         * world in the 90-deg vfov; -70 covers every downward surf view. */
        if (st->pitch > 30.0f) st->pitch = 30.0f;
        if (st->pitch < -70.0f) st->pitch = -70.0f;
        s->last_pitch_delta[i] = pd;

        /* SV_CheckMovingGround basevelocity fold */
        if (!st->base_vel_flag) {
            float sc = s->frametime * 0.5f + 1.0f;
            st->velocity[0] += sc * st->basevelocity[0];
            st->velocity[1] += sc * st->basevelocity[1];
            st->velocity[2] += sc * st->basevelocity[2];
            v3zero(st->basevelocity);
        }
        st->base_vel_flag = 0;   /* Touch re-sets it post-move */

        float prev_progress = st->progress;
        int wl = 0, blocked = 0;
        /* pitch stays out of pm_tick: the contract is "lidar aim only" — and
         * the state convention (+up) is inverted vs GoldSrc angles (+down),
         * so passing it would steer water/ladder movement backwards */
        pm_tick(&s->map, &s->cfg.phys, st, &s->pp[i],
                st->yaw, 0.0f, fmove, smove, buttons, s->cfg.phys.msec, &wl, &blocked);
        st->tick++;

        int fail = 0, complete = 0;
        int trig = apply_triggers(s, i, st);
        if (trig == 1) fail = 1;

        /* progress + reward */
        float r = 0;
        if (s->nwp >= 2) {
            float lat, hgt, tg[3];
            st->progress = spline_project(s, st->origin, &st->seg_hint, &lat, &hgt, tg);
            if (trig != 2) r = 0.05f * (st->progress - prev_progress);
            if (st->progress > st->best_progress) st->best_progress = st->progress;
            if (st->progress >= s->total_len - 50.0f) {
                /* clamped projections report progress==total from anywhere past the
                 * last segment — completion also requires being NEAR the finish */
                const float* fin = &s->wp[3 * (s->nwp - 1)];
                float dx = st->origin[0] - fin[0], dy = st->origin[1] - fin[1], dz = st->origin[2] - fin[2];
                if (dx * dx + dy * dy + dz * dz < 150.0f * 150.0f) { complete = 1; r += 50.0f; }
            }
        }

        /* fail conditions (docs/03; water_fail=0 lets the agent swim) */
        if (wl >= 2 && s->cfg.water_fail) fail = 1;
        if (st->origin[2] < s->kill_z) fail = 1;
        st->stuck_ticks = blocked ? st->stuck_ticks + 1 : 0;
        if (st->stuck_ticks >= 5) fail = 1;

        int truncated = st->tick >= s->cfg.max_episode_ticks;
        int d = fail || complete;
        rewards[i] = r;
        done[i] = (uint8_t)d;
        trunc[i] = (uint8_t)(truncated && !d);

        float* orow = &obs[(size_t)i * s->obs_dim];
        if (d || truncated) {
            write_obs(s, st, s->last_yaw_delta[i], s->last_pitch_delta[i],
                      &terminal_obs[(size_t)i * s->obs_dim]);
            reset_env(s, i);
            write_obs(s, &s->st[i], 0.0f, 0.0f, orow);
        } else {
            write_obs(s, st, s->last_yaw_delta[i], s->last_pitch_delta[i], orow);
        }
    }
}

/* ---- state access -------------------------------------------------------- */
SurfState* surf_states_ptr(SurfSim* s) { return s->st; }

void surf_get_states(SurfSim* s, SurfState* out) {
    memcpy(out, s->st, sizeof(SurfState) * (size_t)s->cfg.num_envs);
}
void surf_set_state(SurfSim* s, int32_t env, const SurfState* st) {
    if (env < 0 || env >= s->cfg.num_envs) return;
    s->st[env] = *st;
    if (s->st[env].seg_hint < 0) s->st[env].seg_hint = 0;
    if (s->nwp >= 2 && s->st[env].seg_hint > s->nwp - 2) s->st[env].seg_hint = s->nwp - 2;
}

/* ---- exposed internals --------------------------------------------------- */
void surf_trace(SurfSim* s, const float* start, const float* end, int32_t usehull, SurfTrace* out) {
    PmTrace tr;
    trace_player(&s->map, usehull, start, end, &tr);
    v3copy(out->endpos, tr.endpos);
    v3copy(out->normal, tr.plane_normal);
    out->dist = tr.plane_dist;
    out->fraction = tr.fraction;
    out->startsolid = tr.startsolid;
    out->allsolid = tr.allsolid;
    out->ent = tr.ent;
}
int32_t surf_point_contents(SurfSim* s, const float* p) { return point_contents(&s->map, p); }

void surf_occupancy_grid(SurfSim* s, const float* mins, float cell,
                         int32_t nx, int32_t ny, int32_t nz, uint8_t* out) {
    int iz;
#pragma omp parallel for schedule(static)
    for (iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                float p[3] = { mins[0] + (ix + 0.5f) * cell,
                               mins[1] + (iy + 0.5f) * cell,
                               mins[2] + (iz + 0.5f) * cell };
                /* point_contents covers only the world hull; a zero-length
                 * point trace also clips solid brush entities (func_wall &
                 * co) exactly like movement physics and the C lidar do */
                int pc = point_contents(&s->map, p);
                int solid = (pc == -2 || pc == -6);
                if (!solid) {
                    PmTrace tr;
                    trace_player(&s->map, 2, p, p, &tr);
                    solid = tr.startsolid;
                }
                out[(size_t)ix + (size_t)nx * ((size_t)iy + (size_t)ny * iz)] =
                    (uint8_t)solid;
            }
        }
    }
}

void surf_pm_step_usercmd(SurfSim* s, SurfState* st, float yaw, float pitch,
                          float fmove, float smove, int32_t buttons, int32_t msec) {
    /* SV_RunCmd duties: msec>50 chop into two halves, basevelocity fold, pm tick.
     * No triggers/env logic. */
    if (msec > 50) {
        int32_t half = (int32_t)(uint8_t)(msec / 2.0);
        surf_pm_step_usercmd(s, st, yaw, pitch, fmove, smove, buttons, half);
        surf_pm_step_usercmd(s, st, yaw, pitch, fmove, smove, buttons, half);
        return;
    }
    float ft = (float)(msec * 0.001);
    if (!st->base_vel_flag) {
        float sc = ft * 0.5f + 1.0f;
        st->velocity[0] += sc * st->basevelocity[0];
        st->velocity[1] += sc * st->basevelocity[1];
        st->velocity[2] += sc * st->basevelocity[2];
        v3zero(st->basevelocity);
    }
    st->base_vel_flag = 0;
    int wl, blocked;
    pm_tick(&s->map, &s->cfg.phys, st, &s->single_pp, yaw, pitch, fmove, smove, buttons, msec, &wl, &blocked);
    st->tick++;
}

int32_t surf_play_step(SurfSim* s, SurfState* st, float yaw, float pitch,
                       float fmove, float smove, int32_t buttons, int32_t msec) {
    if (msec > 50) {                                 /* SV_RunCmd chop */
        int32_t half = (int32_t)(uint8_t)(msec / 2.0);
        int32_t f = surf_play_step(s, st, yaw, pitch, fmove, smove, buttons, half);
        return f | surf_play_step(s, st, yaw, pitch, fmove, smove, buttons, half);
    }
    float ft = (float)(msec * 0.001);
    if (!st->base_vel_flag) {                        /* basevelocity fold */
        float sc = ft * 0.5f + 1.0f;
        st->velocity[0] += sc * st->basevelocity[0];
        st->velocity[1] += sc * st->basevelocity[1];
        st->velocity[2] += sc * st->basevelocity[2];
        v3zero(st->basevelocity);
    }
    st->base_vel_flag = 0;
    int wl = 0, blocked = 0;
    pm_tick(&s->map, &s->cfg.phys, st, &s->single_pp, yaw, pitch, fmove, smove, buttons, msec,
            &wl, &blocked);
    st->tick++;
    int trig = apply_triggers(s, 0, st);             /* post-move, like SV_TouchLinks */
    int32_t flags = 0;
    if (trig == 2) flags |= 1;
    if (trig == 1) flags |= 2;
    if (wl >= 2) flags |= 4;
    if (blocked) flags |= 8;
    return flags;
}

void surf_pm_step_single(SurfSim* s, SurfState* st, const int32_t action[6]) {
    int yb = action[0] < 0 ? 0 : (action[0] > 14 ? 14 : action[0]);
    float yd = YAW_BINS[yb] * (s->cfg.yaw_rate_max_deg / 10.0f);
    int pb = action[1] < 0 ? 0 : (action[1] > 6 ? 6 : action[1]);
    float pd = PITCH_BINS[pb] * (s->cfg.pitch_rate_max_deg / 10.0f);
    float fmove = (action[2] <= 0) ? -400.0f : (action[2] >= 2 ? 400.0f : 0.0f);
    float smove = (action[3] <= 0) ? -400.0f : (action[3] >= 2 ? 400.0f : 0.0f);
    int buttons = 0;
    if (action[4]) buttons |= SURF_IN_JUMP;
    if (action[5]) buttons |= SURF_IN_DUCK;
    st->yaw = wrap_yaw(st->yaw + yd);
    st->pitch += pd;
    if (st->pitch > 30.0f) st->pitch = 30.0f;
    if (st->pitch < -70.0f) st->pitch = -70.0f;
    /* pitch is lidar-aim only (and +up vs GoldSrc's +down): keep physics at 0 */
    surf_pm_step_usercmd(s, st, st->yaw, 0.0f, fmove, smove, buttons, s->cfg.phys.msec);
}
