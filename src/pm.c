/* pm.c — CS 1.6 player movement, transcribed from ReGameDLL_CS
 * regamedll/pm_shared/pm_shared.cpp (vanilla path, REGAMEDLL_ADD/FIXES excluded)
 * and pm_math.cpp. See docs/01-physics-spec.md.
 *
 * Precision discipline: vec3 storage is float; `double` locals mirror the
 * source's real_t sites exactly (docs/01 §7). DotProduct/VectorNormalize
 * radicands compute in FLOAT and widen at the return/cast boundary; only
 * Length accumulates in double. Do not "fix" the mixed arithmetic. */
#include "sim.h"

#define PITCH 0
#define YAW   1
#define ROLL  2
#define BUNNYJUMP_MAX_SPEED_FACTOR 1.2f
#define PM_VEC_VIEW      17.0f
#define PM_VEC_DUCK_VIEW 12.0f
#define TIME_TO_DUCK     0.4
#define PLAYER_DUCKING_MULTIPLIER 0.333
#define HALF_HUMAN_HEIGHT 36    /* FixPlayerCrouchStuck loop bound (HLSDK literal) */
#define MAX_CLIMB_SPEED  200
#define WJ_HEIGHT        8
/* usercmd button bits, HLSDK in_buttons.h (verified upstream) */
#define SURF_IN_FORWARD   (1 << 3)
#define SURF_IN_BACK      (1 << 4)
#define SURF_IN_MOVELEFT  (1 << 9)
#define SURF_IN_MOVERIGHT (1 << 10)
#define MT_WALK 0
#define MT_FLY  1

typedef struct PmCtx {
    const BspMap* map;
    const SurfPhys* ph;
    SurfState* st;
    PmPersist* pp;
    float frametime;
    float maxspeed;               /* effective, after CheckParameters */
    float pfriction;              /* pmove->friction (player), always 1.0 here */
    float angles[3];
    float forward[3], right[3], up[3];
    float fmove, smove, upmove;
    int buttons;
    int msec;
    int usehull;                  /* 0 standing (duck disabled tonight) */
    int waterlevel, watertype;
    float view_ofs_z;
    int blocked_solid;            /* FlyMove trapped / CheckStuck failed */
} PmCtx;

/* ---- pm_math (CS variants) ---------------------------------------------- */

/* DotProduct: float products+sums, widened to real_t only on return */
static double dot_wide(const float* a, const float* b) {
    float s = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return (double)s;
}
/* Length: float products, DOUBLE accumulator (the one genuinely-double helper) */
static double length_wide(const float* v) {
    double length = 0.0f;
    for (int i = 0; i < 3; i++) length += v[i] * v[i];
    return sqrt(length);
}
/* VectorNormalize: FLOAT radicand -> double sqrt; double reciprocal scale */
static double normalize_v(float* v) {
    double length = sqrt((double)(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
    if (length != 0.0) {
        double ilength = 1.0 / length;
        v[0] = (float)(v[0] * ilength);
        v[1] = (float)(v[1] * ilength);
        v[2] = (float)(v[2] * ilength);
    }
    return length;
}
static int is_nan_f(float x) {          /* IS_NAN: exponent-bits test, nanmask = 255<<23 */
    int32_t bits; memcpy(&bits, &x, 4);
    return (bits & (255 << 23)) == (255 << 23);
}

/* AngleVectors (CS): all trig locals float EXCEPT cy and angle (real_t). */
static void angle_vectors(const float* angles, float* fwd, float* right, float* up) {
    float sr, sp, sy, cr, cp;
    double cy, angle;
    angle = (double)(angles[YAW] * (M_PI * 2 / 360));
    sy = (float)sin(angle);
    cy = cos(angle);
    angle = (double)(angles[PITCH] * (M_PI * 2 / 360));
    sp = (float)sin(angle);
    cp = (float)cos(angle);
    angle = (double)(angles[ROLL] * (M_PI * 2 / 360));
    sr = (float)sin(angle);
    cr = (float)cos(angle);
    if (fwd) {
        fwd[0] = (float)(cp * cy);
        fwd[1] = cp * sy;
        fwd[2] = -sp;
    }
    if (right) {
        right[0] = (float)(-1 * sr * sp * cy + -1 * cr * -sy);
        right[1] = (float)(-1 * sr * sp * sy + -1 * cr * cy);
        right[2] = -1 * sr * cp;
    }
    if (up) {
        up[0] = (float)(cr * sp * cy + -sr * -sy);
        up[1] = (float)(cr * sp * sy + -sr * cy);
        up[2] = cr * cp;
    }
}

/* ---- core movement functions -------------------------------------------- */

static void pm_check_velocity(PmCtx* c) {
    SurfState* st = c->st;
    for (int i = 0; i < 3; i++) {
        if (is_nan_f(st->velocity[i])) st->velocity[i] = 0;
        if (is_nan_f(st->origin[i])) st->origin[i] = 0;
        if (st->velocity[i] > c->ph->sv_maxvelocity) st->velocity[i] = c->ph->sv_maxvelocity;
        else if (st->velocity[i] < -c->ph->sv_maxvelocity) st->velocity[i] = -c->ph->sv_maxvelocity;
    }
}

static int pm_clip_velocity(const float* in, const float* normal, float* out, float overbounce) {
    double angle = normal[2];                       /* real_t */
    int blocked = 0x00;
    if (angle > 0) blocked |= 0x01;
    if (!angle) blocked |= 0x02;
    /* the source's DotProduct macro keeps the *overbounce multiply in FLOAT too;
     * widen only at the real_t assignment */
    float backoff_f = (in[0] * normal[0] + in[1] * normal[1] + in[2] * normal[2]) * overbounce;
    double backoff = backoff_f;                           /* real_t */
    for (int i = 0; i < 3; i++) {
        float change = (float)(in[i] - normal[i] * backoff);
        out[i] = change;
        if (out[i] > -STOP_EPSILON && out[i] < STOP_EPSILON) out[i] = 0;
    }
    return blocked;
}

static void pm_add_correct_gravity(PmCtx* c) {
    if (c->st->waterjumptime) return;
    double ent_gravity = 1.0f;                      /* pmove->gravity == 0 -> 1.0f */
    /* source is a float compound-assign with a double RHS: the SUBTRACTION happens
     * in double and rounds to float ONCE (Add's expression is double-led, literal 0.5f) */
    c->st->velocity[2] = (float)(c->st->velocity[2] - ent_gravity * c->ph->sv_gravity * 0.5f * c->frametime);
    c->st->velocity[2] += c->st->basevelocity[2] * c->frametime;
    c->st->basevelocity[2] = 0;
    pm_check_velocity(c);
}
static void pm_fixup_gravity_velocity(PmCtx* c) {
    if (c->st->waterjumptime) return;
    double ent_gravity = 1.0;
    /* Fixup: (gravity * frametime) is a FLOAT product before the double enters;
     * the subtraction itself is double with a single rounding to float */
    c->st->velocity[2] = (float)(c->st->velocity[2] - (double)(c->ph->sv_gravity * c->frametime) * ent_gravity * 0.5);
    pm_check_velocity(c);
}

static void pm_friction(PmCtx* c) {
    SurfState* st = c->st;
    if (st->waterjumptime) return;
    float* vel = st->velocity;
    float speed = (float)sqrt((double)(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]));
    if (speed < 0.1f) return;
    double drop = 0, friction, control, newspeed;   /* real_t */

    if (st->onground != -1) {
        float start[3], stop[3];
        PmTrace trace;
        start[0] = stop[0] = st->origin[0] + vel[0] / speed * 16;
        start[1] = stop[1] = st->origin[1] + vel[1] / speed * 16;
        start[2] = st->origin[2] + g_player_mins[c->usehull][2];
        stop[2] = start[2] - 34;
        trace_player(c->map, c->usehull, start, stop, &trace);
        if (trace.fraction == 1.0f)
            friction = c->ph->sv_friction * c->ph->edgefriction;
        else
            friction = c->ph->sv_friction;
        friction *= c->pfriction;
        control = (speed < c->ph->sv_stopspeed) ? c->ph->sv_stopspeed : speed;
        drop += friction * (control * c->frametime);
    }
    newspeed = speed - drop;
    if (newspeed < 0) newspeed = 0;
    newspeed /= speed;
    /* asymmetric per-component rounding, faithful to the binary (docs/01 §4) */
    float newvel[3];
    newvel[0] = (float)(vel[0] * newspeed);
    newvel[1] = vel[1] * (float)newspeed;
    newvel[2] = vel[2] * (float)newspeed;
    v3copy(st->velocity, newvel);
}

static void pm_accelerate(PmCtx* c, const float* wishdir, double wishspeed, float accel) {
    if (c->st->waterjumptime) return;
    double currentspeed = dot_wide(c->st->velocity, wishdir);
    float addspeed = (float)(wishspeed - currentspeed);
    if (addspeed <= 0) return;
    /* (accel * frametime) is a float product, then widened by the double wishspeed */
    double accelspeed = accel * c->frametime * wishspeed * c->pfriction;
    if (accelspeed > addspeed) accelspeed = addspeed;
    for (int i = 0; i < 3; i++)
        c->st->velocity[i] = (float)(c->st->velocity[i] + accelspeed * wishdir[i]);
}

static void pm_air_accelerate(PmCtx* c, const float* wishdir, float wishspeed, float accel) {
    if (c->st->waterjumptime) return;
    float wishspd = wishspeed;
    if (wishspd > 30) wishspd = 30;                 /* THE 30-unit air cap */
    double currentspeed = dot_wide(c->st->velocity, wishdir);
    float addspeed = (float)(wishspd - currentspeed);
    if (addspeed <= 0) return;
    /* all-float product (UNCAPPED wishspeed), widened on assignment */
    float accel_f = accel * wishspeed * c->frametime * c->pfriction;
    double accelspeed = accel_f;
    if (accelspeed > addspeed) accelspeed = addspeed;
    for (int i = 0; i < 3; i++)
        c->st->velocity[i] = (float)(c->st->velocity[i] + accelspeed * wishdir[i]);
}

static int pm_fly_move(PmCtx* c) {
    SurfState* st = c->st;
    int numbumps = 4, blocked = 0x00, numplanes = 0;
    float planes[MAX_CLIP_PLANES][3];
    float primal_velocity[3], original_velocity[3], new_velocity[3];
    float end[3], dir[3];
    PmTrace trace;
    float time_left = c->frametime, allFraction = 0;

    v3copy(original_velocity, st->velocity);
    v3copy(primal_velocity, st->velocity);

    for (int bumpcount = 0; bumpcount < numbumps; bumpcount++) {
        if (!st->velocity[0] && !st->velocity[1] && !st->velocity[2]) break;

        for (int i = 0; i < 3; i++) {
            double flScale = time_left * st->velocity[i];    /* float product, real_t local */
            end[i] = (float)(st->origin[i] + flScale);
        }
        trace_player(c->map, c->usehull, st->origin, end, &trace);
        allFraction += trace.fraction;

        if (trace.allsolid) {                        /* trapped in solid */
            v3zero(st->velocity);
            c->blocked_solid = 1;
            return 4;
        }
        if (trace.fraction > 0.0f) {
            v3copy(st->origin, trace.endpos);
            v3copy(original_velocity, st->velocity);
            numplanes = 0;
        }
        if (trace.fraction == 1.0f) break;

        if (trace.plane_normal[2] > 0.7f) blocked |= 0x01;
        if (!trace.plane_normal[2]) blocked |= 0x02;

        time_left -= time_left * trace.fraction;

        if (numplanes >= MAX_CLIP_PLANES) { v3zero(st->velocity); break; }
        v3copy(planes[numplanes], trace.plane_normal);
        numplanes++;

        if (numplanes == 1 && (st->onground == -1 || c->pfriction != 1)) {
            /* MOVETYPE_WALK airborne first-plane branch — the surf path */
            for (int i = 0; i < numplanes; i++) {
                if (planes[i][2] > 0.7f) {
                    pm_clip_velocity(original_velocity, planes[i], new_velocity, 1);
                    v3copy(original_velocity, new_velocity);
                } else {
                    pm_clip_velocity(original_velocity, planes[i], new_velocity,
                                     (float)(1.0 + c->ph->sv_bounce * (1.0 - c->pfriction)));
                }
            }
            v3copy(st->velocity, new_velocity);
            v3copy(original_velocity, new_velocity);
        } else {
            int i, j;
            for (i = 0; i < numplanes; i++) {
                pm_clip_velocity(original_velocity, planes[i], st->velocity, 1);
                for (j = 0; j < numplanes; j++)
                    if (j != i && dot_wide(st->velocity, planes[j]) < 0) break;
                if (j == numplanes) break;
            }
            if (i == numplanes) {
                if (numplanes != 2) { v3zero(st->velocity); break; }
                dir[0] = planes[0][1] * planes[1][2] - planes[0][2] * planes[1][1];
                dir[1] = planes[0][2] * planes[1][0] - planes[0][0] * planes[1][2];
                dir[2] = planes[0][0] * planes[1][1] - planes[0][1] * planes[1][0];
                float d = (float)dot_wide(dir, st->velocity);   /* dir NOT normalized */
                st->velocity[0] = d * dir[0];
                st->velocity[1] = d * dir[1];
                st->velocity[2] = d * dir[2];
            }
            if (dot_wide(st->velocity, primal_velocity) <= 0) { v3zero(st->velocity); break; }
        }
    }
    if (allFraction == 0.0f) v3zero(st->velocity);
    return blocked;
}

static void pm_walk_move(PmCtx* c) {
    SurfState* st = c->st;
    float wishvel[3], wishdir[3], dest[3];
    float original[3], originalvel[3], down[3], downvel[3], up_pos[3];
    PmTrace trace;

    /* CS stamina drag: every ground frame while fuser2 > 0 */
    if (c->ph->enable_stamina && st->fuser2 > 0.0) {
        double flRatio = (100 - st->fuser2 * 0.001 * 19) * 0.01;
        st->velocity[0] = (float)(st->velocity[0] * flRatio);
        st->velocity[1] = (float)(st->velocity[1] * flRatio);
    }

    float fmove = c->fmove, smove = c->smove, maxspeed = c->maxspeed;
    c->forward[2] = 0; c->right[2] = 0;
    normalize_v(c->forward);
    normalize_v(c->right);
    for (int i = 0; i < 2; i++)
        wishvel[i] = c->forward[i] * fmove + c->right[i] * smove;
    wishvel[2] = 0;
    v3copy(wishdir, wishvel);
    double wishspeed = normalize_v(wishdir);        /* real_t */
    if (wishspeed > maxspeed) {
        float scale = (float)(maxspeed / wishspeed);    /* VectorScale takes vec_t */
        wishvel[0] *= scale; wishvel[1] *= scale; wishvel[2] *= scale;
        wishspeed = maxspeed;
    }
    st->velocity[2] = 0;
    pm_accelerate(c, wishdir, wishspeed, c->ph->sv_accelerate);
    st->velocity[2] = 0;
    st->velocity[0] += st->basevelocity[0];
    st->velocity[1] += st->basevelocity[1];
    st->velocity[2] += st->basevelocity[2];

    double spd = length_wide(st->velocity);
    if (spd < 1.0) { v3zero(st->velocity); return; }

    int oldonground = st->onground;
    dest[0] = st->origin[0] + st->velocity[0] * c->frametime;
    dest[1] = st->origin[1] + st->velocity[1] * c->frametime;
    dest[2] = st->origin[2];
    trace_player(c->map, c->usehull, st->origin, dest, &trace);
    if (trace.fraction == 1.0f) { v3copy(st->origin, trace.endpos); return; }
    if (oldonground == -1 && c->waterlevel == 0) return;

    v3copy(original, st->origin);
    v3copy(originalvel, st->velocity);

    pm_fly_move(c);                                  /* (a) flat slide */
    v3copy(down, st->origin);
    v3copy(downvel, st->velocity);

    v3copy(st->origin, original);
    v3copy(st->velocity, originalvel);
    v3copy(dest, st->origin);
    dest[2] += c->ph->sv_stepsize;                   /* (b) step up 18 */
    trace_player(c->map, c->usehull, st->origin, dest, &trace);
    if (!trace.startsolid && !trace.allsolid) v3copy(st->origin, trace.endpos);
    pm_fly_move(c);

    v3copy(dest, st->origin);                        /* press back down */
    dest[2] -= c->ph->sv_stepsize;
    trace_player(c->map, c->usehull, st->origin, dest, &trace);
    int usedown = (trace.plane_normal[2] < 0.7f);    /* stepped onto too-steep -> slide result */
    if (!usedown) {
        if (!trace.startsolid && !trace.allsolid) v3copy(st->origin, trace.endpos);
        v3copy(up_pos, st->origin);
        float downdist = (down[0] - original[0]) * (down[0] - original[0])
                       + (down[1] - original[1]) * (down[1] - original[1]);
        float updist = (up_pos[0] - original[0]) * (up_pos[0] - original[0])
                     + (up_pos[1] - original[1]) * (up_pos[1] - original[1]);
        if (downdist > updist) usedown = 1;
        else st->velocity[2] = downvel[2];           /* stepped: keep slide-move z velocity */
    }
    if (usedown) {
        v3copy(st->origin, down);
        v3copy(st->velocity, downvel);
    }
}

static void pm_air_move(PmCtx* c) {
    SurfState* st = c->st;
    float wishvel[3], wishdir[3];
    float fmove = c->fmove, smove = c->smove;
    c->forward[2] = 0; c->right[2] = 0;
    normalize_v(c->forward);
    normalize_v(c->right);
    for (int i = 0; i < 2; i++)
        wishvel[i] = c->forward[i] * fmove + c->right[i] * smove;
    wishvel[2] = 0;
    v3copy(wishdir, wishvel);
    float wishspeed = (float)normalize_v(wishdir);   /* HL/CS AirMove: float wishspeed */
    if (wishspeed > c->maxspeed) {
        float scale = c->maxspeed / wishspeed;
        wishvel[0] *= scale; wishvel[1] *= scale; wishvel[2] *= scale;
        wishspeed = c->maxspeed;
    }
    pm_air_accelerate(c, wishdir, wishspeed, c->ph->sv_airaccelerate);
    st->velocity[0] += st->basevelocity[0];
    st->velocity[1] += st->basevelocity[1];
    st->velocity[2] += st->basevelocity[2];
    pm_fly_move(c);
}

static void pm_check_water(PmCtx* c) {
    SurfState* st = c->st;
    float point[3];
    c->waterlevel = 0;
    c->watertype = CONTENTS_EMPTY;
    point[0] = st->origin[0] + (g_player_mins[c->usehull][0] + g_player_maxs[c->usehull][0]) * 0.5f;
    point[1] = st->origin[1] + (g_player_mins[c->usehull][1] + g_player_maxs[c->usehull][1]) * 0.5f;
    point[2] = st->origin[2] + g_player_mins[c->usehull][2] + 1;
    int cont = point_contents(c->map, point);
    if (cont <= CONTENTS_WATER && cont > CONTENTS_TRANSLUCENT) {
        c->watertype = cont;
        c->waterlevel = 1;
        float height = g_player_mins[c->usehull][2] + g_player_maxs[c->usehull][2];
        float heightover2 = height * 0.5f;
        point[2] = st->origin[2] + heightover2;
        cont = point_contents(c->map, point);
        if (cont <= CONTENTS_WATER && cont > CONTENTS_TRANSLUCENT) {
            c->waterlevel = 2;
            point[2] = st->origin[2] + c->view_ofs_z;
            cont = point_contents(c->map, point);
            if (cont <= CONTENTS_WATER && cont > CONTENTS_TRANSLUCENT)
                c->waterlevel = 3;
        }
        /* water currents: none in surf maps; CONTENTS_CURRENT basevelocity add skipped */
    }
}

static void pm_categorize_position(PmCtx* c) {
    SurfState* st = c->st;
    float point[3];
    PmTrace tr;
    pm_check_water(c);
    point[0] = st->origin[0];
    point[1] = st->origin[1];
    point[2] = st->origin[2] - 2;
    if (st->velocity[2] > 180) { st->onground = -1; return; }
    trace_player(c->map, c->usehull, st->origin, point, &tr);
    if (tr.plane_normal[2] < 0.7f) st->onground = -1;
    else st->onground = tr.ent;
    if (st->onground != -1) {
        /* waterjumptime = 0 (not modeled) */
        if (c->waterlevel < 2 && !tr.startsolid && !tr.allsolid)
            v3copy(st->origin, tr.endpos);
    }
}

/* ---- stuck (SERVER path: single throttled attempt, i>=27 commit rule) ---- */
static float g_stuck_table[54][3];
static int g_stuck_init = 0;
static void create_stuck_table(void) {           /* PM_CreateStuckTable, verbatim loops */
    float x, y, z, zi[3];
    int idx = 0, i;
    memset(g_stuck_table, 0, sizeof(g_stuck_table));
    x = y = 0;
    for (z = -0.125f; z <= 0.125f; z += 0.125f) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=z; idx++; }
    x = z = 0;
    for (y = -0.125f; y <= 0.125f; y += 0.125f) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=z; idx++; }
    y = z = 0;
    for (x = -0.125f; x <= 0.125f; x += 0.125f) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=z; idx++; }
    for (x = -0.125f; x <= 0.125f; x += 0.250f)
        for (y = -0.125f; y <= 0.125f; y += 0.250f)
            for (z = -0.125f; z <= 0.125f; z += 0.250f) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=z; idx++; }
    x = y = 0;
    zi[0] = 0.0f; zi[1] = 1.0f; zi[2] = 6.0f;
    for (i = 0; i < 3; i++) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=zi[i]; idx++; }
    x = z = 0;
    for (y = -2.0f; y <= 2.0f; y += 2.0f) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=z; idx++; }
    y = z = 0;
    for (x = -2.0f; x <= 2.0f; x += 2.0f) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=z; idx++; }
    for (i = 0; i < 3; i++) {
        z = zi[i];
        for (x = -2.0f; x <= 2.0f; x += 2.0f)
            for (y = -2.0f; y <= 2.0f; y += 2.0f) { g_stuck_table[idx][0]=x; g_stuck_table[idx][1]=y; g_stuck_table[idx][2]=z; idx++; }
    }
    g_stuck_init = 1;
}

void pm_init(void) {                             /* serial init; parallel regions must not race it */
    if (!g_stuck_init) create_stuck_table();
}

static int test_player_position_free(PmCtx* c, const float* pos) {
    PmTrace tr;
    trace_player(c->map, c->usehull, pos, pos, &tr);
    return !tr.startsolid;
}

static int pm_check_stuck(PmCtx* c) {
    SurfState* st = c->st;
    if (test_player_position_free(c, st->origin)) {
        c->pp->stuck_idx = 0;                        /* PM_ResetStuckOffsets */
        return 0;
    }
    float base[3], test[3];
    v3copy(base, st->origin);
    /* server path: throttle to one attempt per 0.05 s (5 ticks at msec 10, and
     * ticks * msec < 50 in general, so the throttle keeps its 50 ms meaning at
     * a 7-8 ms tick; sim time, not the source's Sys_FloatTime wall clock).
     * stuck_last_tick stores tick+1 so zero means "never attempted" — the
     * FIRST attempt is never throttled (as in the source, where the wall clock
     * always exceeds the zeroed check time). */
    if (c->pp->stuck_last_tick != 0
        && (st->tick + 1 - c->pp->stuck_last_tick) * c->msec < 50) return 1;
    c->pp->stuck_last_tick = st->tick + 1;
    int i = c->pp->stuck_idx++ % 54;                 /* PM_GetRandomStuckOffsets */
    const float* off = g_stuck_table[i];
    test[0] = base[0] + off[0];
    test[1] = base[1] + off[1];
    test[2] = base[2] + off[2];
    if (test_player_position_free(c, test)) {
        c->pp->stuck_idx = 0;
        if (i >= 27)                                 /* only second-half offsets commit */
            v3copy(st->origin, test);
        return 0;
    }
    return 1;
}

/* ---- water (PM_WaterMove/PM_CheckWaterJump/PM_WaterJump, :1376-2401) ------ */

static void pm_water_move(PmCtx* c) {
    SurfState* st = c->st;
    float wishvel[3], wishdir[3], start[3], dest[3], temp[3];
    PmTrace trace;
    double speed, accelspeed, wishspeed;             /* real_t */
    float newspeed, addspeed;

    for (int i = 0; i < 3; i++)
        wishvel[i] = c->forward[i] * c->fmove + c->smove * c->right[i];
    if (!c->fmove && !c->smove && !c->upmove)
        wishvel[2] -= 60.0f;                         /* idle: drift toward the bottom */
    else
        wishvel[2] += c->upmove;

    v3copy(wishdir, wishvel);
    wishspeed = normalize_v(wishdir);
    if (wishspeed > c->maxspeed) {
        float scale = (float)(c->maxspeed / wishspeed);
        wishvel[0] *= scale; wishvel[1] *= scale; wishvel[2] *= scale;
        wishspeed = c->maxspeed;
    }
    wishspeed *= 0.8;
    st->velocity[0] += st->basevelocity[0];
    st->velocity[1] += st->basevelocity[1];
    st->velocity[2] += st->basevelocity[2];

    v3copy(temp, st->velocity);
    speed = normalize_v(temp);
    if (speed) {
        /* (friction*pfriction*frametime) is a float product before the double speed */
        newspeed = (float)(speed - (double)(c->ph->sv_friction * c->pfriction * c->frametime) * speed);
        if (newspeed < 0.0f) newspeed = 0.0f;
        float scale = (float)(newspeed / speed);
        st->velocity[0] *= scale; st->velocity[1] *= scale; st->velocity[2] *= scale;
    } else
        newspeed = 0;

    if ((float)wishspeed < 0.1f) return;
    addspeed = (float)((float)wishspeed - newspeed);
    if (addspeed > 0.0f) {
        normalize_v(wishvel);
        float acc_f = c->ph->sv_accelerate * c->pfriction * c->frametime * (float)wishspeed;
        accelspeed = acc_f;
        if (accelspeed > addspeed) accelspeed = addspeed;
        for (int i = 0; i < 3; i++)
            st->velocity[i] = (float)(st->velocity[i] + accelspeed * wishvel[i]);
    }

    /* move tail (:1463-1481): press down from stepheight above, else FlyMove */
    for (int i = 0; i < 3; i++)
        dest[i] = st->origin[i] + c->frametime * st->velocity[i];
    v3copy(start, dest);
    start[2] += c->ph->sv_stepsize + 1;
    trace_player(c->map, c->usehull, start, dest, &trace);
    if (!trace.startsolid && !trace.allsolid) {
        v3copy(st->origin, trace.endpos);
        return;
    }
    pm_fly_move(c);
}

static void pm_check_water_jump(PmCtx* c) {
    SurfState* st = c->st;
    float vecStart[3], vecEnd[3], flatforward[3], flatvelocity[3];
    double curspeed;
    PmTrace tr;

    if (st->waterjumptime) return;
    if (st->velocity[2] < -180) return;              /* just jumped in: only hop while rising */

    flatvelocity[0] = st->velocity[0];
    flatvelocity[1] = st->velocity[1];
    flatvelocity[2] = 0;
    curspeed = normalize_v(flatvelocity);
    flatforward[0] = c->forward[0];
    flatforward[1] = c->forward[1];
    flatforward[2] = 0;
    normalize_v(flatforward);
    if (curspeed != 0.0 && dot_wide(flatvelocity, flatforward) < 0.0) return;   /* backing up */

    v3copy(vecStart, st->origin);
    vecStart[2] += WJ_HEIGHT;
    for (int i = 0; i < 3; i++) vecEnd[i] = vecStart[i] + 24 * flatforward[i];

    int savehull = c->usehull;
    c->usehull = 2;                                  /* point hull probes */
    trace_player(c->map, 2, vecStart, vecEnd, &tr);
    if (tr.fraction < 1.0f && fabs((double)tr.plane_normal[2]) < 0.1f) {   /* near-vertical wall */
        vecStart[2] += g_player_maxs[savehull][2] - WJ_HEIGHT;
        for (int i = 0; i < 3; i++) {
            vecEnd[i] = vecStart[i] + 24 * flatforward[i];
            st->movedir[i] = -50 * tr.plane_normal[i];
        }
        trace_player(c->map, 2, vecStart, vecEnd, &tr);
        if (tr.fraction == 1.0f) {                   /* clear above: leap the ledge */
            st->waterjumptime = 2000.0f;
            st->velocity[2] = 225.0f;
            st->oldbuttons |= SURF_IN_JUMP;
        }
    }
    c->usehull = savehull;
}

static void pm_water_jump(PmCtx* c) {
    SurfState* st = c->st;
    if (st->waterjumptime > 10000) st->waterjumptime = 10000;
    if (!st->waterjumptime) return;
    st->waterjumptime -= c->msec;
    if (st->waterjumptime < 0 || !c->waterlevel) st->waterjumptime = 0;
    st->velocity[0] = st->movedir[0];                /* xy locked to the ledge push */
    st->velocity[1] = st->movedir[1];
}

/* ---- ladders (PM_Ladder/PM_LadderMove, :2218-2375) ------------------------ */

static int pm_ladder(PmCtx* c) {
    const BspMap* m = c->map;
    for (int i = 0; i < m->numcontents; i++) {
        const ContentsEnt* e = &m->contents[i];
        if (e->skin != CONTENTS_LADDER) continue;
        if (c->st->origin[0] < e->absmin[0] - 20 || c->st->origin[0] > e->absmax[0] + 20 ||
            c->st->origin[1] < e->absmin[1] - 20 || c->st->origin[1] > e->absmax[1] + 20 ||
            c->st->origin[2] < e->absmin[2] - 40 || c->st->origin[2] > e->absmax[2] + 40)
            continue;
        /* PM_Ladder: the ladder model's hull for the player's CURRENT usehull,
         * containment = anything but EMPTY */
        int bsph = c->usehull == 1 ? 3 : (c->usehull == 2 ? 0 : 1);
        Hull h;
        bsp_model_hull(m, &m->models[e->model], bsph, &h);
        float local[3];
        for (int k = 0; k < 3; k++)
            local[k] = c->st->origin[k] - (h.clip_mins[k] - g_player_mins[c->usehull][k] + e->origin[k]);
        if (hull_point_contents(&h, h.firstclipnode, local) != CONTENTS_EMPTY)
            return i;
    }
    return -1;
}

static void pm_ladder_move(PmCtx* c, int ladder_idx, int* movetype) {
    SurfState* st = c->st;
    const ContentsEnt* lad = &c->map->contents[ladder_idx];
    const BModel* mod = &c->map->models[lad->model];
    float ladderCenter[3], floorp[3];
    PmTrace trace;

    /* PM_GetModelBounds: map-coord bounds, no origin add */
    for (int i = 0; i < 3; i++)
        ladderCenter[i] = (mod->mins[i] + mod->maxs[i]) * 0.5f;

    *movetype = MT_FLY;                              /* gravity off while attached */

    v3copy(floorp, st->origin);
    floorp[2] += g_player_mins[c->usehull][2] - 1;
    int onFloor = point_contents(c->map, floorp) == CONTENTS_SOLID;

    /* PM_TraceModel: POINT hull vs this model only */
    trace_one_model(c->map, lad->model, lad->origin, 2, st->origin, ladderCenter, &trace);
    if (trace.fraction != 1.0f) {
        float fwd = 0, right = 0;
        float vpn[3], v_right[3], vup[3];
        float flSpeed = MAX_CLIMB_SPEED;
        if (flSpeed > c->maxspeed) flSpeed = c->maxspeed;
        angle_vectors(c->angles, vpn, v_right, vup);
        if (st->ducked) flSpeed = (float)(flSpeed * PLAYER_DUCKING_MULTIPLIER);

        if (c->buttons & SURF_IN_BACK)      fwd -= flSpeed;
        if (c->buttons & SURF_IN_FORWARD)   fwd += flSpeed;
        if (c->buttons & SURF_IN_MOVELEFT)  right -= flSpeed;
        if (c->buttons & SURF_IN_MOVERIGHT) right += flSpeed;

        if (c->buttons & SURF_IN_JUMP) {             /* detach: leap off along the normal */
            *movetype = MT_WALK;
            for (int i = 0; i < 3; i++)
                st->velocity[i] = trace.plane_normal[i] * 270;
        } else if (fwd != 0 || right != 0) {
            float vel[3], perp[3], cross[3], lateral[3], tmp[3];
            const float* n = trace.plane_normal;
            for (int i = 0; i < 3; i++)
                vel[i] = vpn[i] * fwd + right * v_right[i];
            v3zero(tmp);
            tmp[2] = 1;
            perp[0] = tmp[1]*n[2] - tmp[2]*n[1];     /* CrossProduct(up, normal) */
            perp[1] = tmp[2]*n[0] - tmp[0]*n[2];
            perp[2] = tmp[0]*n[1] - tmp[1]*n[0];
            normalize_v(perp);
            float normal = (float)dot_wide(vel, n);  /* velocity into the ladder face */
            for (int i = 0; i < 3; i++) cross[i] = n[i] * normal;
            for (int i = 0; i < 3; i++) lateral[i] = vel[i] - cross[i];
            tmp[0] = n[1]*perp[2] - n[2]*perp[1];    /* CrossProduct(normal, perp) */
            tmp[1] = n[2]*perp[0] - n[0]*perp[2];
            tmp[2] = n[0]*perp[1] - n[1]*perp[0];
            for (int i = 0; i < 3; i++)
                st->velocity[i] = lateral[i] + -normal * tmp[i];
            if (onFloor && normal > 0)               /* on ground pushing away: shove off */
                for (int i = 0; i < 3; i++)
                    st->velocity[i] += MAX_CLIMB_SPEED * n[i];
        } else {
            v3zero(st->velocity);                    /* motionless hang */
        }
    }
}

/* ---- duck (vanilla PM_Duck/PM_UnDuck, cs_pm_shared.cpp:2001-2214) --------- */

/* nudge up 1u at a time (duck hull) until free; restore on failure */
static void pm_fix_player_crouch_stuck(PmCtx* c, int direction) {
    float test[3];
    if (test_player_position_free(c, c->st->origin)) return;
    v3copy(test, c->st->origin);
    for (int i = 0; i < HALF_HUMAN_HEIGHT; i++) {
        c->st->origin[2] += direction;
        if (test_player_position_free(c, c->st->origin)) return;
    }
    v3copy(c->st->origin, test);                     /* failed */
}

static void pm_unduck(PmCtx* c) {
    SurfState* st = c->st;
    PmTrace trace;
    float newOrigin[3];
    v3copy(newOrigin, st->origin);
    if (st->onground != -1)
        newOrigin[2] = (float)(newOrigin[2] + 18.0); /* vanilla literal — incl. the
                                                        mid-transition duck-peek pop */
    trace_player(c->map, c->usehull, newOrigin, newOrigin, &trace);
    if (!trace.startsolid) {
        c->usehull = 0;
        trace_player(c->map, 0, newOrigin, newOrigin, &trace);
        if (trace.startsolid) {                      /* blocked overhead: stay ducked */
            c->usehull = 1;
            return;
        }
        st->ducked = 0;
        st->induck = 0;
        c->view_ofs_z = PM_VEC_VIEW;
        st->duck_time = 0;
        /* flTimeStepSound -= 100: sounds not modeled */
        v3copy(st->origin, newOrigin);
        pm_categorize_position(c);
    }
}

static void pm_duck(PmCtx* c) {
    SurfState* st = c->st;
    int buttonsChanged = st->oldbuttons ^ c->buttons;
    int nButtonPressed = buttonsChanged & c->buttons;
    if (c->buttons & SURF_IN_DUCK) st->oldbuttons |= SURF_IN_DUCK;
    else st->oldbuttons &= ~SURF_IN_DUCK;

    if (!(c->buttons & SURF_IN_DUCK) && !st->induck && !st->ducked)
        return;

    double mult = PLAYER_DUCKING_MULTIPLIER;         /* real_t; scales the clamped cmd */
    c->fmove = (float)(c->fmove * mult);
    c->smove = (float)(c->smove * mult);
    c->upmove = (float)(c->upmove * mult);

    if (c->buttons & SURF_IN_DUCK) {
        if ((nButtonPressed & SURF_IN_DUCK) && !st->ducked) {
            st->duck_time = 1000;
            st->induck = 1;
        }
        if (st->induck) {
            /* finish after 400 ms, or instantly when airborne (duck-jump: no shift) */
            if ((st->duck_time / 1000.0) <= (1.0 - TIME_TO_DUCK) || st->onground == -1) {
                c->usehull = 1;
                c->view_ofs_z = PM_VEC_DUCK_VIEW;
                st->ducked = 1;
                st->induck = 0;
                if (st->onground != -1) {
                    st->origin[2] = (float)(st->origin[2] - 18.0);
                    pm_fix_player_crouch_stuck(c, 1);
                    pm_categorize_position(c);
                }
            }
            /* else: view-height spline animation only — physics view_ofs stays
               binary (docs/07 §4); the play client animates cosmetically */
        }
    } else {
        pm_unduck(c);
    }
}

/* ---- jump ---------------------------------------------------------------- */
static void pm_prevent_mega_bunny_jumping(PmCtx* c) {
    float maxscaledspeed = BUNNYJUMP_MAX_SPEED_FACTOR * c->maxspeed;
    if (maxscaledspeed <= 0.0f) return;
    double spd = length_wide(c->st->velocity);       /* real_t */
    if (spd <= maxscaledspeed) return;
    float fraction = (float)((maxscaledspeed / spd) * 0.8);
    c->st->velocity[0] *= fraction;
    c->st->velocity[1] *= fraction;
    c->st->velocity[2] *= fraction;
}

static void pm_jump(PmCtx* c) {
    SurfState* st = c->st;
    if (st->waterjumptime != 0.0f) {                 /* mid water-leap: just decay */
        st->waterjumptime -= c->msec;
        if (st->waterjumptime < 0) st->waterjumptime = 0;
        return;
    }
    if (c->waterlevel >= 2) {                        /* swimming: swim pop */
        st->onground = -1;
        if (c->watertype == CONTENTS_WATER) st->velocity[2] = 100;
        else if (c->watertype == CONTENTS_SLIME) st->velocity[2] = 80;
        else st->velocity[2] = 50;
        return;
    }
    if (st->onground == -1) { st->oldbuttons |= SURF_IN_JUMP; return; }
    if (st->oldbuttons & SURF_IN_JUMP) return;       /* edge trigger: don't pogo */
    if (st->induck && st->ducked) return;            /* CS-only duck guard */
    st->onground = -1;
    if (c->ph->enable_bhop_cap) pm_prevent_mega_bunny_jumping(c);
    /* PM_JumpHeight: DOUBLE sqrt, stored to float velocity[2] */
    st->velocity[2] = (float)sqrt(2.0 * 800.0f * 45.0f);
    if (c->ph->enable_stamina) {
        if (st->fuser2 > 0.0f) {
            double flRatio = (100.0 - st->fuser2 * 0.001 * 19.0) * 0.01;
            st->velocity[2] = (float)(st->velocity[2] * flRatio);
        }
        st->fuser2 = (float)1315.789429;
    }
    pm_fixup_gravity_velocity(c);                    /* pre-decay half gravity */
    st->oldbuttons |= SURF_IN_JUMP;
}

/* ---- the tick ------------------------------------------------------------ */
void pm_tick(const BspMap* m, const SurfPhys* ph, SurfState* st, PmPersist* pp,
             float yaw, float pitch, float fmove, float smove, int buttons, int msec,
             int* out_waterlevel, int* out_blocked_solid) {
    if (!g_stuck_init) create_stuck_table();         /* idempotent; race-safe (same values) */

    PmCtx c;
    memset(&c, 0, sizeof(c));
    c.map = m; c.ph = ph; c.st = st; c.pp = pp;
    c.pfriction = 1.0f;
    /* engine recomputes usehull from FL_DUCKING every cmd (SV_RunCmd) */
    c.usehull = (ph->enable_duck && st->ducked) ? 1 : 0;
    c.view_ofs_z = (ph->enable_duck && st->ducked) ? PM_VEC_DUCK_VIEW : PM_VEC_VIEW;
    c.buttons = buttons;
    /* the client sets direction button bits from held keys; derive from move signs
     * (ladder movement reads these, not fmove/smove) */
    if (fmove > 0) c.buttons |= SURF_IN_FORWARD;
    else if (fmove < 0) c.buttons |= SURF_IN_BACK;
    if (smove > 0) c.buttons |= SURF_IN_MOVERIGHT;
    else if (smove < 0) c.buttons |= SURF_IN_MOVELEFT;
    c.msec = msec;
    c.fmove = fmove; c.smove = smove; c.upmove = 0;
    c.maxspeed = ph->sv_maxspeed;

    /* --- PM_CheckParameters --- */
    float spd = (float)sqrt((double)(c.smove * c.smove + c.fmove * c.fmove + c.upmove * c.upmove));
    double maxspeed_d = ph->player_maxspeed;         /* clientmaxspeed, real_t */
    if (maxspeed_d != 0.0f)
        c.maxspeed = (float)(maxspeed_d < (double)c.maxspeed ? maxspeed_d : (double)c.maxspeed);
    if (spd != 0.0f && spd > (double)c.maxspeed) {
        double fRatio = (double)(c.maxspeed / spd);  /* float division, real_t local */
        c.fmove = (float)(c.fmove * fRatio);
        c.smove = (float)(c.smove * fRatio);
        c.upmove = (float)(c.upmove * fRatio);
    }
    c.angles[PITCH] = pitch;
    c.angles[YAW] = yaw;
    c.angles[ROLL] = 0;
    if (c.angles[YAW] > 180.0f) c.angles[YAW] -= 360.0f;

    /* --- frametime + PM_ReduceTimers --- */
    c.frametime = (float)(msec * 0.001);
    if (st->duck_time > 0) {
        st->duck_time -= msec;
        if (st->duck_time < 0) st->duck_time = 0;
    }
    if (st->fuser2 > 0.0) {
        st->fuser2 -= msec;
        if (st->fuser2 < 0.0) st->fuser2 = 0;
    }

    angle_vectors(c.angles, c.forward, c.right, c.up);

    /* --- PM_CheckStuck (CS: no duck retry) --- */
    if (pm_check_stuck(&c)) {
        c.blocked_solid = 1;
        *out_waterlevel = 0;
        *out_blocked_solid = 1;
        return;
    }

    pm_categorize_position(&c);

    int ladder_idx = pm_ladder(&c);                  /* source: scan before Duck */
    if (ph->enable_duck) pm_duck(&c);                /* CS order: Duck before step sound */

    int movetype = MT_WALK;
    if (ladder_idx >= 0) pm_ladder_move(&c, ladder_idx, &movetype);

    if (movetype == MT_FLY) {
        /* on the ladder: no gravity; jump was handled inside LadderMove (detach) */
        pm_check_water(&c);
        if (!(c.buttons & SURF_IN_JUMP)) st->oldbuttons &= ~SURF_IN_JUMP;
        st->velocity[0] += st->basevelocity[0];
        st->velocity[1] += st->basevelocity[1];
        st->velocity[2] += st->basevelocity[2];
        pm_fly_move(&c);
        st->velocity[0] -= st->basevelocity[0];
        st->velocity[1] -= st->basevelocity[1];
        st->velocity[2] -= st->basevelocity[2];
    } else {
        /* --- MOVETYPE_WALK --- */
        if (!(c.waterlevel > 1)) pm_add_correct_gravity(&c);

        if (st->waterjumptime != 0.0f) {             /* leaping out of the water */
            pm_water_jump(&c);
            pm_fly_move(&c);
            pm_check_water(&c);
        } else if (c.waterlevel >= 2) {              /* swimming */
            if (c.waterlevel == 2) pm_check_water_jump(&c);
            if (st->velocity[2] < 0 && st->waterjumptime) st->waterjumptime = 0;
            if (c.buttons & SURF_IN_JUMP) pm_jump(&c);
            else st->oldbuttons &= ~SURF_IN_JUMP;
            pm_water_move(&c);
            st->velocity[0] -= st->basevelocity[0];
            st->velocity[1] -= st->basevelocity[1];
            st->velocity[2] -= st->basevelocity[2];
            pm_categorize_position(&c);
        } else {
            if (c.buttons & SURF_IN_JUMP) pm_jump(&c);
            else st->oldbuttons &= ~SURF_IN_JUMP;

            if (st->onground != -1) {
                st->velocity[2] = 0;
                pm_friction(&c);
            }
            pm_check_velocity(&c);

            if (st->onground != -1) pm_walk_move(&c);
            else pm_air_move(&c);

            pm_categorize_position(&c);

            st->velocity[0] -= st->basevelocity[0];
            st->velocity[1] -= st->basevelocity[1];
            st->velocity[2] -= st->basevelocity[2];
            pm_check_velocity(&c);

            if (!(c.waterlevel > 1)) pm_fixup_gravity_velocity(&c);
            if (st->onground != -1) st->velocity[2] = 0;
        }
    }

    /* SV_RunCmd writeback truncation: pev->flDuckTime = (int)pmove->flDuckTime */
    st->duck_time = (float)(int)st->duck_time;

    *out_waterlevel = c.waterlevel;
    *out_blocked_solid = c.blocked_solid;
}
