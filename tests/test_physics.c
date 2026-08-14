/* test_physics.c — Gate C analytic tests (docs/05 tier 1) run on the real map. */
#include <stdio.h>
#include <stdlib.h>
#include "../src/sim.h"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; \
    printf("FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static SurfPhys surf_phys(void) {
    SurfPhys p;
    p.sv_gravity = 800; p.sv_airaccelerate = 100; p.sv_accelerate = 5;
    p.sv_friction = 4; p.edgefriction = 2; p.sv_stopspeed = 75;
    p.sv_maxspeed = 320; p.player_maxspeed = 250; p.sv_maxvelocity = 2000;
    p.sv_stepsize = 18; p.sv_bounce = 1; p.msec = 10;
    p.enable_stamina = 1; p.enable_bhop_cap = 1; p.enable_duck = 0;
    return p;
}

static SurfState mkstate(float x, float y, float z) {
    SurfState s; memset(&s, 0, sizeof(s));
    s.origin[0] = x; s.origin[1] = y; s.origin[2] = z;
    s.onground = -1;
    return s;
}

/* run one tick; returns waterlevel */
static int tick(const BspMap* m, const SurfPhys* ph, SurfState* st, PmPersist* pp,
                float yaw, float fmove, float smove, int buttons) {
    int wl = 0, blocked = 0;
    pm_tick(m, ph, st, pp, yaw, 0.0f, fmove, smove, buttons, ph->msec, &wl, &blocked);
    st->tick++;
    return wl;
}

static float hspeed(const SurfState* s) {
    return (float)sqrt((double)(s->velocity[0]*s->velocity[0] + s->velocity[1]*s->velocity[1]));
}

/* find a clear-air point with lots of space below (for air/ramp tests) */
static int find_air(const BspMap* m, float* out, float clearance) {
    for (int ix = 2; ix < 38; ix++) for (int iy = 2; iy < 38; iy++) {
        float x = m->world_mins[0] + (m->world_maxs[0]-m->world_mins[0]) * (ix+0.5f) / 40.0f;
        float y = m->world_mins[1] + (m->world_maxs[1]-m->world_mins[1]) * (iy+0.5f) / 40.0f;
        for (float z = m->world_maxs[2] - 300.0f; z > m->world_mins[2] + 500.0f; z -= 200.0f) {
            float p[3] = { x, y, z };
            if (point_contents(m, p) != CONTENTS_EMPTY) continue;
            PmTrace t0; trace_player(m, 0, p, p, &t0);
            if (t0.startsolid) continue;
            float down[3] = { x, y, z - clearance };
            PmTrace tr; trace_player(m, 0, p, down, &tr);
            float side[3] = { x + 60, y, z };
            PmTrace ts; trace_player(m, 0, p, side, &ts);
            if (tr.fraction == 1.0f && ts.fraction == 1.0f) { out[0]=x; out[1]=y; out[2]=z; return 1; }
        }
    }
    return 0;
}

/* find a surfable ramp: down-trace hit with 0.35 < nz < 0.68 and clear air above */
static int find_ramp(const BspMap* m, float* pos_out, float* n_out) {
    for (int ix = 1; ix < 79; ix++) for (int iy = 1; iy < 79; iy++) {
        float x = m->world_mins[0] + (m->world_maxs[0]-m->world_mins[0]) * (ix+0.5f) / 80.0f;
        float y = m->world_mins[1] + (m->world_maxs[1]-m->world_mins[1]) * (iy+0.5f) / 80.0f;
        for (float z = m->world_maxs[2] - 200.0f; z > m->world_mins[2] + 200.0f; z -= 150.0f) {
            float p[3] = { x, y, z };
            if (point_contents(m, p) != CONTENTS_EMPTY) continue;
            PmTrace t0; trace_player(m, 0, p, p, &t0);
            if (t0.startsolid) continue;
            float down[3] = { x, y, z - 400.0f };
            PmTrace tr; trace_player(m, 0, p, down, &tr);
            if (tr.fraction < 1.0f && !tr.startsolid &&
                tr.plane_normal[2] > 0.35f && tr.plane_normal[2] < 0.68f &&
                (z - tr.endpos[2]) > 60.0f) {
                pos_out[0] = tr.endpos[0]; pos_out[1] = tr.endpos[1]; pos_out[2] = tr.endpos[2] + 40.0f;
                v3copy(n_out, tr.plane_normal);
                return 1;
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "maps\\surf_ski_2.bsp";
    BspMap m; char err[256];
    if (bsp_load(&m, path, err, sizeof(err))) { printf("load: %s\n", err); return 2; }
    SurfPhys ph = surf_phys();
    const float dt = (float)(10 * 0.001);

    /* flat rest position under spawn 0 */
    const float* sp = m.spawns[0].origin;
    float s0[3] = { sp[0], sp[1], sp[2] }, e0[3] = { sp[0], sp[1], sp[2] - 1000 };
    PmTrace tr; trace_player(&m, 0, s0, e0, &tr);
    float restz = tr.endpos[2];

    /* ---- 6. Stationarity: AFK on flat ground, 1000 ticks ---- */
    {
        SurfState st = mkstate(sp[0], sp[1], restz);
        PmPersist pp = {0};
        tick(&m, &ph, &st, &pp, 0, 0, 0, 0);         /* settle tick */
        float ox = st.origin[0], oy = st.origin[1], oz = st.origin[2];
        int always_ground = 1;
        for (int i = 0; i < 1000; i++) {
            tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
            if (st.onground == -1) always_ground = 0;
        }
        CHECK(st.origin[0] == ox && st.origin[1] == oy && st.origin[2] == oz,
              "T6 drift: (%g %g %g) vs (%g %g %g)", st.origin[0], st.origin[1], st.origin[2], ox, oy, oz);
        CHECK(always_ground, "T6: left ground while AFK");
    }

    /* ---- 1. Jump apex ~= 45u; grounded state machine sane ---- */
    {
        SurfState st = mkstate(sp[0], sp[1], restz);
        PmPersist pp = {0};
        tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
        float z0 = st.origin[2];
        tick(&m, &ph, &st, &pp, 0, 0, 0, SURF_IN_JUMP);   /* jump */
        CHECK(st.onground == -1, "T1: still grounded after jump");
        CHECK(fabsf(st.velocity[2] - (268.328155517578125f - 800.0f * dt)) < 0.01f,
              "T1: post-jump-tick vz %f", st.velocity[2]);
        float apex = z0;
        for (int i = 0; i < 200 && st.onground == -1; i++) {
            tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
            if (st.origin[2] > apex) apex = st.origin[2];
        }
        CHECK(fabsf((apex - z0) - 45.0f) < 0.35f, "T1: apex %.4f (want ~45)", apex - z0);
        CHECK(st.onground != -1, "T1: never landed");
        CHECK(st.fuser2 > 0.0f && st.fuser2 < 1315.789429, "T1: stamina not decaying: %f", st.fuser2);
    }

    /* ---- 2. Friction decay matches the exact recurrence ---- */
    {
        SurfState st = mkstate(sp[0], sp[1], restz);
        PmPersist pp = {0};
        tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
        st.velocity[0] = 250.0f;
        double ref = 250.0;
        for (int i = 0; i < 40; i++) {
            tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
            /* reference recurrence (assumes no edgefriction on the big platform) */
            double control = ref < 75.0 ? 75.0 : ref;
            double drop = 4.0 * (control * (double)dt);
            ref -= drop; if (ref < 0) ref = 0;
        }
        CHECK(fabs(hspeed(&st) - ref) < 1.0, "T2: speed %.3f vs ref %.3f", hspeed(&st), ref);
    }

    /* ---- 3. Air-accel single tick: saturated and rate-limited regimes ---- */
    {
        float air[3];
        CHECK(find_air(&m, air, 300.0f), "T3: no clear air point found");
        SurfState st = mkstate(air[0], air[1], air[2]);
        st.velocity[0] = 300.0f;
        PmPersist pp = {0};
        tick(&m, &ph, &st, &pp, 0.0f, 0, 400.0f, 0);      /* yaw 0, full sidemove: wishdir (0,-1,0) */
        CHECK(st.velocity[1] == -30.0f, "T3a: vy %f (want -30, saturated)", st.velocity[1]);
        CHECK(st.velocity[0] == 300.0f, "T3a: vx changed: %f", st.velocity[0]);
        CHECK(fabsf(st.velocity[2] + 800.0f * dt) < 1e-3f, "T3a: vz %f", st.velocity[2]);

        SurfPhys ph10 = ph; ph10.sv_airaccelerate = 10;   /* rate-limited: 10*250*0.01 = 25 < 30 */
        SurfState st2 = mkstate(air[0], air[1], air[2]);
        st2.velocity[0] = 300.0f;
        PmPersist pp2 = {0};
        tick(&m, &ph10, &st2, &pp2, 0.0f, 0, 400.0f, 0);
        CHECK(fabsf(st2.velocity[1] + 25.0f) < 1e-4f, "T3b: vy %f (want -25)", st2.velocity[1]);
    }

    /* ---- 7. Bhop cap + stamina exact values ---- */
    {
        SurfState st = mkstate(sp[0], sp[1], restz);
        PmPersist pp = {0};
        tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
        st.velocity[0] = 400.0f;                          /* over the 1.2*250=300 cap */
        tick(&m, &ph, &st, &pp, 0, 0, 0, SURF_IN_JUMP);
        /* Expected: PreventMegaBunnyJumping runs AFTER AddCorrectGravity, so the 3D
         * Length includes v_z = -half-gravity; fraction divides by ~400.02, not 400. */
        {
            float vz_pre = (float)(1.0 * 800.0f * 0.5f * dt);  /* 4 */
            double spd = 0.0f;                                  /* Length: double accumulate */
            spd += 400.0f * 400.0f; spd += 0.0f * 0.0f; spd += (-vz_pre) * (-vz_pre);
            spd = sqrt(spd);
            float fraction = (float)(((1.2f * 250.0f) / spd) * 0.8);
            float expect_h = 400.0f * fraction;
            CHECK(fabsf(hspeed(&st) - expect_h) < 1e-3f,
                  "T7a: capped speed %f (want %f)", hspeed(&st), expect_h);
        }

        SurfState st3 = mkstate(sp[0], sp[1], restz);
        PmPersist pp3 = {0};
        tick(&m, &ph, &st3, &pp3, 0, 0, 0, 0);
        st3.fuser2 = (float)1315.789429;                  /* full stamina debt */
        tick(&m, &ph, &st3, &pp3, 0, 0, 0, SURF_IN_JUMP);
        /* expected, mirroring code order: ReduceTimers decays 10ms FIRST, then jump ratio */
        double f2 = (double)(float)1315.789429 - 10.0;
        double ratio = (100.0 - f2 * 0.001 * 19.0) * 0.01;
        float expect = (float)((double)(float)sqrt(2.0*800.0f*45.0f) * ratio) - 800.0f * dt;
        CHECK(fabsf(st3.velocity[2] - expect) < 0.01f, "T7b: stamina vz %f want %f", st3.velocity[2], expect);
    }

    /* ---- 8. Per-axis maxvelocity clamp ---- */
    {
        float air[3];
        find_air(&m, air, 300.0f);
        SurfState st = mkstate(air[0], air[1], air[2]);
        st.velocity[0] = 3000.0f; st.velocity[1] = -3000.0f;
        PmPersist pp = {0};
        tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
        CHECK(st.velocity[0] <= 2000.0f && st.velocity[0] > 1900.0f, "T8: vx %f", st.velocity[0]);
        CHECK(st.velocity[1] >= -2000.0f && st.velocity[1] < -1900.0f, "T8: vy %f", st.velocity[1]);
    }

    /* ---- 5. THE SURF TEST: drop onto a real ramp, no input -> slides, gains speed,
     *        never grounded ---- */
    {
        float rpos[3], rn[3];
        CHECK(find_ramp(&m, rpos, rn), "T5: no ramp found on map");
        printf("ramp at (%.0f %.0f %.0f) normal (%.3f %.3f %.3f)\n",
               rpos[0], rpos[1], rpos[2], rn[0], rn[1], rn[2]);
        SurfState st = mkstate(rpos[0], rpos[1], rpos[2]);
        PmPersist pp = {0};
        int grounded_ticks = 0, wl = 0;
        float speed_at_50 = 0, speed_at_150 = 0;
        for (int i = 0; i < 150; i++) {
            wl = tick(&m, &ph, &st, &pp, 0, 0, 0, 0);
            if (st.onground != -1) grounded_ticks++;
            if (i == 49) speed_at_50 = hspeed(&st);
            if (i == 149) speed_at_150 = hspeed(&st);
            if (wl >= 2) break;                            /* fell into water: end */
        }
        CHECK(grounded_ticks == 0, "T5: onground %d ticks on a surf ramp", grounded_ticks);
        CHECK(speed_at_50 > 100.0f, "T5: not accelerating (%.1f at t=50)", speed_at_50);
        if (wl < 2)
            CHECK(speed_at_150 > speed_at_50, "T5: speed fell: %.1f -> %.1f", speed_at_50, speed_at_150);
        printf("surf slide: h-speed %.1f u/s at tick 50, %.1f at tick 150\n", speed_at_50, speed_at_150);
    }

    bsp_free(&m);
    if (fails == 0) { printf("test_physics: ALL OK\n"); return 0; }
    printf("test_physics: %d FAILURES\n", fails);
    return 1;
}
