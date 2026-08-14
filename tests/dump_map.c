/* dump_map.c — Gate A check: load a BSP, print counts + entity summary (docs/06 Gate A). */
#include <stdio.h>
#include "../src/sim.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: dump_map <map.bsp>\n"); return 2; }
    BspMap m;
    char err[256] = {0};
    if (bsp_load(&m, argv[1], err, sizeof(err)) != 0) {
        fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }
    printf("planes      %d\n", m.numplanes);
    printf("clipnodes   %d\n", m.numclipnodes);
    printf("hull0nodes  %d\n", m.numhull0);
    printf("models      %d\n", m.nummodels);
    printf("solids      %d\n", m.numsolids);
    printf("contents    %d\n", m.numcontents);
    printf("triggers    %d\n", m.numtriggers);
    printf("spawns      %d\n", m.numspawns);
    printf("world mins  %.1f %.1f %.1f\n", m.world_mins[0], m.world_mins[1], m.world_mins[2]);
    printf("world maxs  %.1f %.1f %.1f\n", m.world_maxs[0], m.world_maxs[1], m.world_maxs[2]);
    printf("world headnodes %d %d %d %d\n",
           m.models[0].headnode[0], m.models[0].headnode[1],
           m.models[0].headnode[2], m.models[0].headnode[3]);
    static const char* kinds[] = { "teleport", "hurt", "push" };
    for (int i = 0; i < m.numtriggers; i++) {
        const Trigger* t = &m.triggers[i];
        printf("trigger %2d %-8s model *%-3d aabb (%.0f %.0f %.0f)-(%.0f %.0f %.0f)",
               i, kinds[t->kind], t->model,
               t->absmin[0], t->absmin[1], t->absmin[2],
               t->absmax[0], t->absmax[1], t->absmax[2]);
        if (t->kind == TRIG_TELEPORT)
            printf(" dest %s (%.0f %.0f %.0f) yaw %.0f", t->has_dest ? "ok" : "MISSING",
                   t->dest[0], t->dest[1], t->dest[2], t->dest_yaw);
        if (t->kind == TRIG_PUSH)
            printf(" pushvel (%.0f %.0f %.0f)%s", t->pushvel[0], t->pushvel[1], t->pushvel[2],
                   t->push_once ? " ONCE" : "");
        if (t->kind == TRIG_HURT) printf(" dmg %.0f", t->dmg);
        printf("\n");
    }
    for (int i = 0; i < m.numcontents; i++)
        printf("contents %d model *%d skin %d\n", i, m.contents[i].model, m.contents[i].skin);
    for (int i = 0; i < m.numspawns && i < 4; i++)
        printf("spawn %d (%.1f %.1f %.1f) yaw %.0f\n", i,
               m.spawns[i].origin[0], m.spawns[i].origin[1], m.spawns[i].origin[2], m.spawns[i].yaw);
    bsp_free(&m);
    return 0;
}
