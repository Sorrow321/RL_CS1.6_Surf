/* trace.c — GoldSrc hull collision, transcribed from ReHLDS engine_pmovetst.cpp
 * (PM_RecursiveHullCheck / PM_HullPointContents / _PM_PlayerTrace / PM_PointContents).
 * Simplifications valid for this env: brush models only (no studio/box hulls),
 * no rotated solids, no ignore lists. Float arithmetic mirrors the source. */
#include "sim.h"

/* _DotProduct: float products/sums (docs/01 §7 — widen only at return; here callers are float) */
static float dotf(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

int hull_point_contents(const Hull* hull, int num, const float* p) {
    if (hull->firstclipnode >= hull->lastclipnode)
        return CONTENTS_EMPTY;                                   /* -1 */
    while (num >= 0) {
        const BClipNode* node = &hull->clipnodes[num];
        const BPlane* plane = &hull->planes[node->planenum];
        float d = (plane->type >= 3) ? dotf(p, plane->normal) - plane->dist
                                     : p[plane->type] - plane->dist;
        num = (d >= 0.0f) ? node->children[0] : node->children[1];
    }
    return num;                                                  /* CONTENTS_* */
}

/* PM_RecursiveHullCheck, vanilla pmove variant (engine_pmovetst.cpp:614). */
static int recursive_hull_check(const Hull* hull, int num, float p1f, float p2f,
                                const float* p1, const float* p2, PmTrace* trace) {
    float mid[3], pdif, frac, t1, t2, midf;

    if (num < 0) {
        if (num == CONTENTS_SOLID) {
            trace->startsolid = 1;
        } else {
            trace->allsolid = 0;
            if (num == CONTENTS_EMPTY) trace->inopen = 1;
            else trace->inwater = 1;
        }
        return 1;
    }
    if (hull->firstclipnode >= hull->lastclipnode) {
        trace->allsolid = 0;
        trace->inopen = 1;
        return 1;
    }

    const BClipNode* node = &hull->clipnodes[num];
    const BPlane* plane = &hull->planes[node->planenum];
    if (plane->type >= 3) {
        t1 = dotf(p1, plane->normal) - plane->dist;
        t2 = dotf(p2, plane->normal) - plane->dist;
    } else {
        t1 = p1[plane->type] - plane->dist;
        t2 = p2[plane->type] - plane->dist;
    }
    if (t1 >= 0.0f && t2 >= 0.0f)
        return recursive_hull_check(hull, node->children[0], p1f, p2f, p1, p2, trace);

    if (t1 >= 0.0f) {
        midf = t1 - DIST_EPSILON;
    } else {
        if (t2 < 0.0f)
            return recursive_hull_check(hull, node->children[1], p1f, p2f, p1, p2, trace);
        midf = t1 + DIST_EPSILON;
    }
    midf = midf / (t1 - t2);
    if (midf >= 0.0f) { if (midf > 1.0f) midf = 1.0f; }
    else midf = 0.0f;

    pdif = p2f - p1f;
    frac = pdif * midf + p1f;
    mid[0] = (p2[0] - p1[0]) * midf + p1[0];
    mid[1] = (p2[1] - p1[1]) * midf + p1[1];
    mid[2] = (p2[2] - p1[2]) * midf + p1[2];

    int side = (t1 >= 0.0f) ? 0 : 1;

    if (!recursive_hull_check(hull, node->children[side], p1f, frac, p1, mid, trace))
        return 0;

    if (hull_point_contents(hull, node->children[side ^ 1], mid) != CONTENTS_SOLID)
        return recursive_hull_check(hull, node->children[side ^ 1], frac, p2f, mid, p2, trace);

    if (trace->allsolid)
        return 0;

    if (side) {
        trace->plane_normal[0] = -plane->normal[0];
        trace->plane_normal[1] = -plane->normal[1];
        trace->plane_normal[2] = -plane->normal[2];
        trace->plane_dist = -plane->dist;
    } else {
        v3copy(trace->plane_normal, plane->normal);
        trace->plane_dist = plane->dist;
    }

    if (hull_point_contents(hull, hull->firstclipnode, mid) != CONTENTS_SOLID) {
        trace->fraction = frac;
        v3copy(trace->endpos, mid);
        return 0;
    }

    /* numeric backup: mid landed in solid — step back along the segment */
    for (;;) {
        midf = (float)(midf - 0.05);
        if (midf < 0.0f) break;
        frac = pdif * midf + p1f;
        mid[0] = (p2[0] - p1[0]) * midf + p1[0];
        mid[1] = (p2[1] - p1[1]) * midf + p1[1];
        mid[2] = (p2[2] - p1[2]) * midf + p1[2];
        if (hull_point_contents(hull, hull->firstclipnode, mid) != CONTENTS_SOLID) {
            trace->fraction = frac;
            v3copy(trace->endpos, mid);
            return 0;
        }
    }
    trace->fraction = frac;      /* backed up past 0 */
    v3copy(trace->endpos, mid);
    return 0;
}

/* pmove usehull -> BSP hull index (PM_HullForBsp switch) */
static int bsphull_for_usehull(int usehull) {
    switch (usehull) {
        case 1: return 3;    /* ducked  -> crouch hull */
        case 2: return 0;    /* point   -> point hull */
        case 3: return 2;    /* large */
        default: return 1;   /* standing -> human hull */
    }
}

/* Clip a segment against one brush model (world or entity); mirrors the per-entity
 * body of _PM_PlayerTrace. */
static void clip_to_model(const BspMap* m, const BModel* mod, const float* ent_origin,
                          int usehull, const float* start, const float* end, PmTrace* total) {
    Hull hull;
    float offset[3], start_l[3], end_l[3];
    bsp_model_hull(m, mod, bsphull_for_usehull(usehull), &hull);
    for (int i = 0; i < 3; i++) {
        offset[i] = hull.clip_mins[i] - g_player_mins[usehull][i] + ent_origin[i];
        start_l[i] = start[i] - offset[i];
        end_l[i] = end[i] - offset[i];
    }
    memset(total, 0, sizeof(*total));
    v3copy(total->endpos, end);
    total->fraction = 1.0f;
    total->allsolid = 1;
    total->ent = -1;

    recursive_hull_check(&hull, hull.firstclipnode, 0.0f, 1.0f, start_l, end_l, total);

    if (total->allsolid) total->startsolid = 1;
    if (total->startsolid) total->fraction = 0.0f;
    if (total->fraction != 1.0f) {
        total->endpos[0] = (end[0] - start[0]) * total->fraction + start[0];
        total->endpos[1] = (end[1] - start[1]) * total->fraction + start[1];
        total->endpos[2] = (end[2] - start[2]) * total->fraction + start[2];
    }
}

void trace_player(const BspMap* m, int usehull, const float* start, const float* end, PmTrace* tr) {
    PmTrace total;

    /* physent 0: world */
    clip_to_model(m, &m->models[0], m->models[0].origin, usehull, start, end, &total);
    *tr = total;
    tr->ent = 0;
    if (total.fraction == 1.0f && !total.startsolid) tr->ent = -1;

    /* solid brush entities; AABB reject padded by hull extents */
    float pad = 40.0f;
    for (int i = 0; i < m->numsolids; i++) {
        const SolidEnt* s = &m->solids[i];
        float lo[3], hi[3];
        for (int k = 0; k < 3; k++) {
            lo[k] = (start[k] < end[k] ? start[k] : end[k]) - pad;
            hi[k] = (start[k] > end[k] ? start[k] : end[k]) + pad;
        }
        if (lo[0] > s->absmax[0] || hi[0] < s->absmin[0] ||
            lo[1] > s->absmax[1] || hi[1] < s->absmin[1] ||
            lo[2] > s->absmax[2] || hi[2] < s->absmin[2]) continue;

        clip_to_model(m, &m->models[s->model], s->origin, usehull, start, end, &total);
        if (total.fraction < tr->fraction) {          /* strictly smaller: copy wholesale */
            int ent = i + 1;
            *tr = total;
            tr->ent = ent;
        }
    }
}

/* PM_PointContents: world hull0, currents -> WATER, then contents entities (skin). */
int point_contents(const BspMap* m, const float* p) {
    Hull h0;
    bsp_model_hull(m, &m->models[0], 0, &h0);
    int cont = hull_point_contents(&h0, h0.firstclipnode, p);
    if (cont <= CONTENTS_CURRENT_0 && cont >= CONTENTS_CURRENT_DOWN)
        cont = CONTENTS_WATER;
    else if (cont == CONTENTS_SOLID)
        return CONTENTS_SOLID;

    for (int i = 0; i < m->numcontents; i++) {
        const ContentsEnt* c = &m->contents[i];
        if (c->skin == CONTENTS_EMPTY) continue;     /* skin -1 = render-only, never a volume */
        if (p[0] < c->absmin[0] || p[0] > c->absmax[0] ||
            p[1] < c->absmin[1] || p[1] > c->absmax[1] ||
            p[2] < c->absmin[2] || p[2] > c->absmax[2]) continue;
        float local[3] = { p[0] - c->origin[0], p[1] - c->origin[1], p[2] - c->origin[2] };
        Hull eh;
        bsp_model_hull(m, &m->models[c->model], 0, &eh);
        if (hull_point_contents(&eh, eh.firstclipnode, local) != CONTENTS_EMPTY)
            return c->skin;
    }
    return cont;
}

/* SV_TouchLinks containment: trigger model's hull selected by the PLAYER's box size,
 * queried at player origin in trigger-local space. Interior of a trigger brush reads SOLID. */
int trigger_contains(const BspMap* m, const Trigger* t, int usehull, const float* origin) {
    /* No AABB precheck here: the trigger's clipnode hull is pre-expanded by the
     * player hull, so containment legitimately extends up to 36u beyond the brush
     * AABB. Callers precheck player-box vs trigger-box (engine BoundsIntersect). */
    Hull h;
    bsp_model_hull(m, &m->models[t->model], bsphull_for_usehull(usehull), &h);
    float local[3];
    for (int i = 0; i < 3; i++) {
        float off = h.clip_mins[i] - g_player_mins[usehull][i] + t->origin[i];
        local[i] = origin[i] - off;
    }
    return hull_point_contents(&h, h.firstclipnode, local) == CONTENTS_SOLID;
}
