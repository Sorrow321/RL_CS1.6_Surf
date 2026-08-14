/* test_trace.c — Gate B: trace invariants on the real map (docs/05 §8, docs/06 Gate B). */
#include <stdio.h>
#include <stdlib.h>
#include "../src/sim.h"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; \
    printf("FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "maps\\surf_ski_2.bsp";
    BspMap m;
    char err[256];
    if (bsp_load(&m, path, err, sizeof(err)) != 0) { printf("load failed: %s\n", err); return 2; }
    CHECK(m.numspawns > 0, "no spawns");
    const float* sp = m.spawns[0].origin;

    /* T1: standing down-trace from spawn hits floor with up normal */
    float start[3] = { sp[0], sp[1], sp[2] };
    float end[3] = { sp[0], sp[1], sp[2] - 1000.0f };
    PmTrace tr;
    trace_player(&m, 0, start, end, &tr);
    CHECK(tr.fraction < 1.0f, "T1: no floor below spawn");
    CHECK(!tr.startsolid && !tr.allsolid, "T1: spawn in solid");
    CHECK(tr.plane_normal[2] > 0.99f, "T1: floor normal z=%f", tr.plane_normal[2]);
    float floor_z = tr.endpos[2];
    printf("floor under spawn: hull-1 rest z = %.3f (expect ~41)\n", floor_z);
    CHECK(floor_z > 35.0f && floor_z < 45.0f, "T1: floor z %f", floor_z);

    /* T2: start inside expanded solid -> startsolid, fraction 0 */
    float s2[3] = { sp[0], sp[1], floor_z - 6.0f };
    trace_player(&m, 0, s2, end, &tr);
    CHECK(tr.startsolid, "T2: expected startsolid");
    CHECK(tr.fraction == 0.0f, "T2: fraction %f", tr.fraction);

    /* T3/T9: open air trace, both directions clean (short — the spawn room has a ceiling) */
    float lo9[3] = { sp[0], sp[1], sp[2] + 5.0f };
    float up[3] = { sp[0], sp[1], sp[2] + 40.0f };
    trace_player(&m, 0, lo9, up, &tr);
    CHECK(tr.fraction == 1.0f, "T3: up blocked f=%f", tr.fraction);
    trace_player(&m, 0, up, lo9, &tr);
    CHECK(tr.fraction == 1.0f, "T9: reverse blocked f=%f", tr.fraction);

    /* T5: point contents below/above the real floor surface (point hull) */
    trace_player(&m, 2, start, end, &tr);
    float surf_z = tr.endpos[2];
    printf("floor surface z = %.3f (expect ~5)\n", surf_z);
    float below[3] = { sp[0], sp[1], surf_z - 2.0f };
    float above[3] = { sp[0], sp[1], surf_z + 2.0f };
    CHECK(point_contents(&m, below) == CONTENTS_SOLID, "T5: below floor not solid: %d", point_contents(&m, below));
    CHECK(point_contents(&m, above) == CONTENTS_EMPTY, "T5: above floor not empty: %d", point_contents(&m, above));

    /* T6: func_water volume reads as water via contents entities */
    int water_ok = 0;
    for (int i = 0; i < m.numcontents && !water_ok; i++) {
        if (m.contents[i].skin != CONTENTS_WATER) continue;
        float c[3] = { (m.contents[i].absmin[0] + m.contents[i].absmax[0]) * 0.5f,
                       (m.contents[i].absmin[1] + m.contents[i].absmax[1]) * 0.5f,
                       (m.contents[i].absmin[2] + m.contents[i].absmax[2]) * 0.5f };
        if (point_contents(&m, c) == CONTENTS_WATER) water_ok = 1;
    }
    CHECK(water_ok, "T6: no func_water volume read as CONTENTS_WATER");

    /* T7: the map contains surf ramps — scan down-traces for 0.1 < nz < 0.7.
     * Columns can start inside the outer shell: find open air first via point contents. */
    int ramp_hits = 0;
    for (int ix = 0; ix < 40; ix++) {
        for (int iy = 0; iy < 40; iy++) {
            float x = m.world_mins[0] + (m.world_maxs[0] - m.world_mins[0]) * (ix + 0.5f) / 40.0f;
            float y = m.world_mins[1] + (m.world_maxs[1] - m.world_mins[1]) * (iy + 0.5f) / 40.0f;
            float a[3] = { x, y, 0 }, probe[3] = { x, y, 0 };
            int found_air = 0;
            for (float z = m.world_maxs[2] - 40.0f; z > m.world_mins[2]; z -= 64.0f) {
                probe[2] = z;
                if (point_contents(&m, probe) != CONTENTS_EMPTY) continue;
                PmTrace tt;                       /* need hull-1 CLEARANCE, not just point air */
                trace_player(&m, 0, probe, probe, &tt);
                if (!tt.startsolid) { a[2] = z; found_air = 1; break; }
            }
            if (!found_air) continue;
            float b[3] = { x, y, m.world_mins[2] + 10.0f };
            trace_player(&m, 0, a, b, &tr);
            if (tr.fraction < 1.0f && !tr.startsolid &&
                tr.plane_normal[2] > 0.1f && tr.plane_normal[2] < 0.7f) {
                float len2 = tr.plane_normal[0]*tr.plane_normal[0] +
                             tr.plane_normal[1]*tr.plane_normal[1] +
                             tr.plane_normal[2]*tr.plane_normal[2];
                CHECK(len2 > 0.99f && len2 < 1.01f, "T7: unnormalized normal");
                ramp_hits++;
            }
        }
    }
    printf("ramp-normal down-trace hits: %d\n", ramp_hits);
    CHECK(ramp_hits > 5, "T7: too few ramp hits (%d)", ramp_hits);

    /* T8: giant floor teleport trigger contains its own center, not a point far above */
    int tele = -1;
    float best_area = 0;
    for (int i = 0; i < m.numtriggers; i++) {
        if (m.triggers[i].kind != TRIG_TELEPORT) continue;
        float area = (m.triggers[i].absmax[0] - m.triggers[i].absmin[0]) *
                     (m.triggers[i].absmax[1] - m.triggers[i].absmin[1]);
        if (area > best_area) { best_area = area; tele = i; }
    }
    CHECK(tele >= 0, "T8: no teleport trigger");
    if (tele >= 0) {
        const Trigger* t = &m.triggers[tele];
        float c[3] = { (t->absmin[0] + t->absmax[0]) * 0.5f,
                       (t->absmin[1] + t->absmax[1]) * 0.5f,
                       (t->absmin[2] + t->absmax[2]) * 0.5f };
        CHECK(trigger_contains(&m, t, 0, c), "T8: center not contained");
        float far_up[3] = { c[0], c[1], c[2] + 500.0f };
        CHECK(!trigger_contains(&m, t, 0, far_up), "T8: far point contained");
    }

    bsp_free(&m);
    if (fails == 0) { printf("test_trace: ALL OK\n"); return 0; }
    printf("test_trace: %d FAILURES\n", fails);
    return 1;
}
