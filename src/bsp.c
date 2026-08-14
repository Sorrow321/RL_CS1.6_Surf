/* bsp.c — GoldSrc BSP v30 loader + entity parser. Spec: docs/02-bsp-collision.md.
 * Reference: third_party/engine_bspfile.h, engine_model.cpp (Mod_MakeHull0, hull wiring). */
#include <stdio.h>
#include <stdlib.h>
#include "sim.h"

const float g_player_mins[4][3] = {
    { -16.0f, -16.0f, -36.0f },   /* usehull 0: standing */
    { -16.0f, -16.0f, -18.0f },   /* usehull 1: ducked */
    {   0.0f,   0.0f,   0.0f },   /* usehull 2: point */
    { -32.0f, -32.0f, -32.0f },   /* usehull 3: large */
};
const float g_player_maxs[4][3] = {
    { 16.0f, 16.0f, 36.0f },
    { 16.0f, 16.0f, 18.0f },
    {  0.0f,  0.0f,  0.0f },
    { 32.0f, 32.0f, 32.0f },
};

/* BSP hull dims (Mod_LoadClipnodes), indexed by BSP hull 0..3 */
static const float k_hull_mins[4][3] = {
    { 0, 0, 0 }, { -16, -16, -36 }, { -32, -32, -32 }, { -16, -16, -18 },
};
static const float k_hull_maxs[4][3] = {
    { 0, 0, 0 }, { 16, 16, 36 }, { 32, 32, 32 }, { 16, 16, 18 },
};

static void seterr(char* err, int errlen, const char* msg) {
    if (err && errlen > 0) { strncpy(err, msg, (size_t)errlen - 1); err[errlen - 1] = 0; }
}

static void* load_lump(const uint8_t* data, long fsize, const BLump* l, int elemsize,
                       int32_t* count, char* err, int errlen) {
    if (l->fileofs < 0 || l->filelen < 0 ||
        (int64_t)l->fileofs + (int64_t)l->filelen > (int64_t)fsize) {
        seterr(err, errlen, "lump out of file bounds"); return NULL;
    }
    if (elemsize > 1 && (l->filelen % elemsize) != 0) {
        seterr(err, errlen, "lump size not multiple of element size"); return NULL;
    }
    void* p = malloc(l->filelen > 0 ? (size_t)l->filelen : 1);
    if (!p) { seterr(err, errlen, "out of memory"); return NULL; }
    memcpy(p, data + l->fileofs, (size_t)l->filelen);
    *count = elemsize > 1 ? l->filelen / elemsize : l->filelen;
    return p;
}

/* Mod_MakeHull0: mirror the render NODES into clipnode form; leaf children become contents. */
static int make_hull0(BspMap* m, const BNode* nodes, int numnodes, const BLeaf* leafs, int numleafs,
                      char* err, int errlen) {
    m->hull0nodes = (BClipNode*)malloc(sizeof(BClipNode) * (numnodes > 0 ? (size_t)numnodes : 1));
    if (!m->hull0nodes) { seterr(err, errlen, "oom hull0"); return -1; }
    m->numhull0 = numnodes;
    for (int i = 0; i < numnodes; i++) {
        m->hull0nodes[i].planenum = nodes[i].planenum;
        for (int j = 0; j < 2; j++) {
            int16_t c = nodes[i].children[j];
            if (c >= 0) {
                m->hull0nodes[i].children[j] = c;
            } else {
                int leaf = -1 - (int)c;
                if (leaf < 0 || leaf >= numleafs) { seterr(err, errlen, "bad leaf ref"); return -1; }
                m->hull0nodes[i].children[j] = (int16_t)leafs[leaf].contents;
            }
        }
    }
    return 0;
}

void bsp_model_hull(const BspMap* m, const BModel* mod, int bsphull, Hull* out) {
    out->planes = m->planes;
    if (bsphull == 0) {
        out->clipnodes = m->hull0nodes;
        out->firstclipnode = mod->headnode[0];
        out->lastclipnode = m->numhull0 - 1;
    } else {
        out->clipnodes = m->clipnodes;
        out->firstclipnode = mod->headnode[bsphull];
        out->lastclipnode = m->numclipnodes - 1;
    }
    v3copy(out->clip_mins, k_hull_mins[bsphull]);
    v3copy(out->clip_maxs, k_hull_maxs[bsphull]);
}

/* ---- entity lump parsing ------------------------------------------------ */

#define MAX_KV 40
typedef struct { char key[40]; char val[520]; } KV;

/* Advance *pp past one { ... } block, filling kv. Returns kv count, or -1 at end of lump. */
static int next_entity(const char** pp, const char* end, KV* kv) {
    const char* p = *pp;
    while (p < end && *p && *p != '{') p++;
    if (p >= end || !*p) return -1;
    p++;
    int n = 0;
    for (;;) {
        while (p < end && *p && *p != '"' && *p != '}') p++;
        if (p >= end || !*p || *p == '}') { if (p < end && *p == '}') p++; break; }
        /* key */
        p++; const char* ks = p;
        while (p < end && *p && *p != '"') p++;
        int klen = (int)(p - ks); if (p < end) p++;
        /* value */
        while (p < end && *p && *p != '"') p++;
        if (p >= end || !*p) break;
        p++; const char* vs = p;
        while (p < end && *p && *p != '"') p++;
        int vlen = (int)(p - vs); if (p < end) p++;
        if (n < MAX_KV) {
            if (klen > 39) klen = 39;
            if (vlen > 519) vlen = 519;
            memcpy(kv[n].key, ks, (size_t)klen); kv[n].key[klen] = 0;
            memcpy(kv[n].val, vs, (size_t)vlen); kv[n].val[vlen] = 0;
            n++;
        }
    }
    *pp = p;
    return n;
}

static const char* kv_get(const KV* kv, int n, const char* key) {
    for (int i = 0; i < n; i++) if (!strcmp(kv[i].key, key)) return kv[i].val;
    return NULL;
}
static float kv_float(const KV* kv, int n, const char* key, float dflt) {
    const char* v = kv_get(kv, n, key);
    return v ? (float)atof(v) : dflt;
}
static void kv_vec(const KV* kv, int n, const char* key, float* out3) {
    out3[0] = out3[1] = out3[2] = 0.0f;
    const char* v = kv_get(kv, n, key);
    if (v) sscanf(v, "%f %f %f", &out3[0], &out3[1], &out3[2]);
}
static int kv_model(const KV* kv, int n) {          /* "*N" -> N, else -1 */
    const char* v = kv_get(kv, n, "model");
    if (v && v[0] == '*') return atoi(v + 1);
    return -1;
}

/* SetMovedir (cs_subs.cpp): angles (0,-1,0) -> up, (0,-2,0) -> down, else forward of angles. */
static void set_movedir(const float* angles, float* movedir) {
    if (angles[0] == 0 && angles[1] == -1 && angles[2] == 0) { movedir[0]=0; movedir[1]=0; movedir[2]=1; return; }
    if (angles[0] == 0 && angles[1] == -2 && angles[2] == 0) { movedir[0]=0; movedir[1]=0; movedir[2]=-1; return; }
    double pitch = angles[0] * (M_PI * 2 / 360), yaw = angles[1] * (M_PI * 2 / 360);
    movedir[0] = (float)(cos(pitch) * cos(yaw));
    movedir[1] = (float)(cos(pitch) * sin(yaw));
    movedir[2] = (float)(-sin(pitch));
}

static void ent_bounds(const BspMap* m, int model, const float* origin, float* absmin, float* absmax) {
    const BModel* mod = &m->models[model];
    for (int i = 0; i < 3; i++) {                    /* SetObjectCollisionBox pads +-1 */
        absmin[i] = mod->mins[i] + origin[i] - 1.0f;
        absmax[i] = mod->maxs[i] + origin[i] + 1.0f;
    }
}

/* point entities kept for teleport targetname lookup */
typedef struct { char targetname[64]; float origin[3]; float yaw; } PointEnt;

static float parse_yaw(const KV* kv, int n) {
    const char* a = kv_get(kv, n, "angles");
    if (a) { float v[3] = {0,0,0}; sscanf(a, "%f %f %f", &v[0], &v[1], &v[2]); return v[1]; }
    return kv_float(kv, n, "angle", 0.0f);
}

static int parse_entities(BspMap* m, char* err, int errlen) {
    static const char* solid_classes[] = {
        "func_wall", "func_breakable", "func_pushable", "func_button", "func_train",
        "func_conveyor", "func_wall_toggle", "func_rotating", "func_door_rotating", NULL
    };
    PointEnt* points = (PointEnt*)calloc(1024, sizeof(PointEnt));
    int npoints = 0;
    typedef struct { int trig; char target[64]; } Pending;
    Pending pend[MAX_TRIGGERS]; int npend = 0;

    const char* p = m->entstring;
    const char* end = m->entstring + m->entlen;
    KV kv[MAX_KV];
    int n;
    while ((n = next_entity(&p, end, kv)) >= 0) {
        const char* cls = kv_get(kv, n, "classname");
        if (!cls) continue;
        int model = kv_model(kv, n);
        if (model >= m->nummodels) continue;         /* malformed *N reference */
        float origin[3]; kv_vec(kv, n, "origin", origin);

        if (!strncmp(cls, "info_player_start", 17) || !strncmp(cls, "info_player_deathmatch", 22)) {
            if (m->numspawns < MAX_SPAWNS) {
                v3copy(m->spawns[m->numspawns].origin, origin);
                m->spawns[m->numspawns].yaw = parse_yaw(kv, n);
                m->numspawns++;
            }
        } else if (!strcmp(cls, "trigger_teleport") && model >= 0) {
            if (m->numtriggers < MAX_TRIGGERS) {
                Trigger* t = &m->triggers[m->numtriggers];
                memset(t, 0, sizeof(*t));
                t->kind = TRIG_TELEPORT; t->model = model; v3copy(t->origin, origin);
                ent_bounds(m, model, origin, t->absmin, t->absmax);
                const char* tgt = kv_get(kv, n, "target");
                if (tgt && npend < MAX_TRIGGERS) {
                    pend[npend].trig = m->numtriggers;
                    strncpy(pend[npend].target, tgt, 63); pend[npend].target[63] = 0;
                    npend++;
                }
                m->numtriggers++;
            }
        } else if (!strcmp(cls, "trigger_hurt") && model >= 0) {
            if (m->numtriggers < MAX_TRIGGERS) {
                Trigger* t = &m->triggers[m->numtriggers];
                memset(t, 0, sizeof(*t));
                t->kind = TRIG_HURT; t->model = model; v3copy(t->origin, origin);
                ent_bounds(m, model, origin, t->absmin, t->absmax);
                t->dmg = kv_float(kv, n, "dmg", 10.0f);
                m->numtriggers++;
            }
        } else if (!strcmp(cls, "trigger_push") && model >= 0) {
            if (m->numtriggers < MAX_TRIGGERS) {
                Trigger* t = &m->triggers[m->numtriggers];
                memset(t, 0, sizeof(*t));
                t->kind = TRIG_PUSH; t->model = model; v3copy(t->origin, origin);
                ent_bounds(m, model, origin, t->absmin, t->absmax);
                float angles[3]; kv_vec(kv, n, "angles", angles);
                /* legacy "angle" key -> angles (0, V, 0), incl. the -1/-2 up/down codes */
                if (!kv_get(kv, n, "angles") && kv_get(kv, n, "angle")) {
                    angles[0] = angles[2] = 0;
                    angles[1] = kv_float(kv, n, "angle", 0.0f);
                }
                /* CTriggerPush::Spawn: zero angles -> yaw 360 (movedir = +x); replicate */
                if (angles[0] == 0 && angles[1] == 0 && angles[2] == 0) angles[1] = 360.0f;
                float movedir[3]; set_movedir(angles, movedir);
                float speed = kv_float(kv, n, "speed", 0.0f);
                if (speed == 0.0f) speed = 100.0f;   /* engine: if (pev->speed == 0) speed = 100 */
                for (int i = 0; i < 3; i++) t->pushvel[i] = movedir[i] * speed;
                t->push_once = ((int)kv_float(kv, n, "spawnflags", 0) & 1) ? 1 : 0;
                m->numtriggers++;
            }
        } else if ((!strcmp(cls, "func_water") || !strcmp(cls, "func_ladder") ||
                    (!strcmp(cls, "func_illusionary") && kv_float(kv, n, "skin", -1.0f) < -1.0f) ||
                    (!strcmp(cls, "func_door") && kv_float(kv, n, "skin", 0.0f) < -1.0f)) && model >= 0) {
            /* SOLID_NOT contents volumes (docs/02 §4): func_water, skin<-1 illusionary,
             * doors with skin!=0; func_ladder = CONTENTS_LADDER */
            if (m->numcontents < MAX_CONTENTS) {
                ContentsEnt* c = &m->contents[m->numcontents];
                c->model = model; v3copy(c->origin, origin);
                c->skin = !strcmp(cls, "func_ladder") ? CONTENTS_LADDER
                                                      : (int32_t)kv_float(kv, n, "skin", -3.0f);
                ent_bounds(m, model, origin, c->absmin, c->absmax);
                m->numcontents++;
            }
        } else if (model >= 0) {
            int solid = 0;
            for (int i = 0; solid_classes[i]; i++) if (!strcmp(cls, solid_classes[i])) { solid = 1; break; }
            if (!strcmp(cls, "func_door")) solid = 1;   /* skin==0 doors handled above */
            if (solid && m->numsolids < MAX_SOLIDS) {
                SolidEnt* s = &m->solids[m->numsolids];
                s->model = model; v3copy(s->origin, origin);
                ent_bounds(m, model, origin, s->absmin, s->absmax);
                m->numsolids++;
            }
            /* func_illusionary skin>=-1, triggers we don't model, buyzones etc: non-solid, ignored */
        } else {
            /* point entity: remember targetname for teleport destination lookup */
            const char* tn = kv_get(kv, n, "targetname");
            if (tn && npoints < 1024) {
                strncpy(points[npoints].targetname, tn, 63); points[npoints].targetname[63] = 0;
                v3copy(points[npoints].origin, origin);
                points[npoints].yaw = parse_yaw(kv, n);
                npoints++;
            }
        }
    }

    /* resolve teleport destinations */
    for (int i = 0; i < npend; i++) {
        Trigger* t = &m->triggers[pend[i].trig];
        for (int j = 0; j < npoints; j++) {
            if (!strcmp(points[j].targetname, pend[i].target)) {
                t->has_dest = 1;
                v3copy(t->dest, points[j].origin);
                t->dest_yaw = points[j].yaw;
                break;
            }
        }
    }
    free(points);
    (void)err; (void)errlen;
    return 0;
}

int bsp_load(BspMap* m, const char* path, char* err, int errlen) {
    memset(m, 0, sizeof(*m));
    FILE* f = fopen(path, "rb");
    if (!f) { seterr(err, errlen, "cannot open bsp file"); return -1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc((size_t)fsize);
    if (!data || fread(data, 1, (size_t)fsize, f) != (size_t)fsize) {
        seterr(err, errlen, "read failed"); fclose(f); free(data); return -1;
    }
    fclose(f);

    if (fsize < (long)sizeof(BHeader)) { seterr(err, errlen, "file too small"); free(data); return -1; }
    const BHeader* h = (const BHeader*)data;
    if (h->version != 30) { seterr(err, errlen, "not a BSP v30 file"); free(data); return -1; }

    int ok = 1;
    int32_t numnodes = 0, numleafs = 0;
    BNode* nodes = NULL; BLeaf* leafs = NULL;
    m->planes    = (BPlane*)load_lump(data, fsize, &h->lumps[LUMP_PLANES], sizeof(BPlane), &m->numplanes, err, errlen);
    m->clipnodes = (BClipNode*)load_lump(data, fsize, &h->lumps[LUMP_CLIPNODES], sizeof(BClipNode), &m->numclipnodes, err, errlen);
    nodes        = (BNode*)load_lump(data, fsize, &h->lumps[LUMP_NODES], sizeof(BNode), &numnodes, err, errlen);
    leafs        = (BLeaf*)load_lump(data, fsize, &h->lumps[LUMP_LEAFS], sizeof(BLeaf), &numleafs, err, errlen);
    m->models    = (BModel*)load_lump(data, fsize, &h->lumps[LUMP_MODELS], sizeof(BModel), &m->nummodels, err, errlen);
    m->entstring = (char*)load_lump(data, fsize, &h->lumps[LUMP_ENTITIES], 1, &m->entlen, err, errlen);
    if (!m->planes || !m->clipnodes || !nodes || !leafs || !m->models || !m->entstring) ok = 0;

    if (ok && make_hull0(m, nodes, numnodes, leafs, numleafs, err, errlen) != 0) ok = 0;
    free(nodes); free(leafs); free(data);
    if (!ok) { bsp_free(m); return -1; }

    if (m->nummodels < 1) { seterr(err, errlen, "no models"); bsp_free(m); return -1; }
    v3copy(m->world_mins, m->models[0].mins);
    v3copy(m->world_maxs, m->models[0].maxs);

    if (parse_entities(m, err, errlen) != 0) { bsp_free(m); return -1; }
    return 0;
}

void bsp_free(BspMap* m) {
    free(m->planes); free(m->clipnodes); free(m->hull0nodes);
    free(m->models); free(m->entstring);
    memset(m, 0, sizeof(*m));
}
