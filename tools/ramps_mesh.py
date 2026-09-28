"""ramps_mesh.py - the map's surfaces from its FACES, grounded in its COLLISION (v4, 2026-09-28).

The exported render mesh (viewer/assets/<map>.mesh.json: the world + the solid brush entities;
triggers, teleports and func_illusionary dropped) is grouped into surfaces, and every surface a
player can target is checked against the simulator's own collision:

    python tools/ramps_mesh.py maps_pool/<map>.bsp [--out ramps_mesh_<map>.npz] [--sample 32]

1. ORIENTATION: the exporter's face normals. The mesh winds clockwise - its per-vertex normals are
   the BSP face's plane normal with the face's side, opposite the winding on 97.7% of utopia's
   world triangles - so the stored normal IS the face's outward side (v3 guessed it from point
   contents, which cannot see brush entities, and reversed 82% of the faces it called ambiguous;
   Codex 23:16Z). Against point traces including entities, the stored side is the open side on
   81.1% of sampled utopia faces and the opposite side on 0.2%; the rest are faces no point
   reaches (under a solid func_wall, pressed into the world, player-hull-only entities).
2. GROUPS: plane key = (outward normal to 1e-3, plane distance to 0.5 u); triangles of one plane
   that TOUCH form a piece. Touching is exact: every edge is sampled every EDGE_STEP u and two
   triangles touch when samples of theirs lie within EDGE_R u (a shared edge or a T-junction;
   v3 bucketed samples into 16 u cells, which both split real junctions and joined gaps). Then
   touching pieces within MERGE_DEG of each other merge (a curved ramp's facets), each group
   staying within CAP_DEG of its largest piece.
3. CATEGORIES from the outward n_z, the engine's rules: floor >= 0.7 (walkable), ramp in
   (0.02, 0.7), wall |n_z| <= 0.02, ceiling < -0.02; KILL = a surface whose contact origins are
   mostly inside a kill trigger; HIDDEN = a floor or ramp NO player hull can touch (a face under
   a solid func_wall, inside the world): it is never a target.
4. CONTACT ORIGINS, validated: sample points on every floor / ramp surface (one per --sample u
   square, at least one per triangle, at most MAX_SAMPLES per surface) and, for the STANDING and
   the DUCKED hull, a hull trace along -n onto each. A sample is a contact origin only when the
   trace stops exactly on that face's plane - not start-solid, the hit normal within ~5.7 deg of
   the face's, and the hull centre at the support distance (16|nx| + 16|ny| + 36|nz| standing,
   18|nz| ducked) within 1 u - so every origin is a place the player's origin really is when its
   hull touches that surface (v3 pushed samples out by the support without a trace: 33-48% of
   them were inside solid). The standing ones are the RampMap points; the ducked ones are kept
   beside them.
5. CLIP DRESSING (optional, --clip-dressing): hlcsg compiles CLIP brushes into the player hulls
   only - no rendered face, no point-hull solid - so a ramp built as a clip brush has no solid
   triangle (utopia's curved ramps). The faces of non-solid func_illusionary pieces that lie ON
   the player's collision surface can stand in for it (tri_src 1). It is a HEURISTIC, off by
   default (Codex 21:16Z: art can be offset or partial; the collision is the authority).
6. PROVENANCE: the npz records the map, the .bsp's size + mtime signature and the mesh's sha1;
   surfgym.rampvocab refuses a vocabulary whose signature does not match the map it is used on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

VERSION = 4
CATS = ("floor", "ramp", "wall", "ceiling", "kill", "hidden")
HULL_HALF = {0: np.array([16.0, 16.0, 36.0]), 1: np.array([16.0, 16.0, 18.0])}  # stand, duck
MERGE_DEG = 10.0       # touching pieces this close in normal are one (curved) surface
CAP_DEG = 25.0         # ... while every piece stays this close to the group's largest piece
EDGE_STEP = 4.0        # u between edge samples
EDGE_R = 2.5           # u: two triangles touch when edge samples of theirs are this close
MAX_SAMPLES = 2000     # contact samples per surface at most (the spacing grows past it)
CONTACT_TOL = 1.0      # u: the hull centre's distance from the plane vs the support distance
CONTACT_COS = 0.995    # the hit plane vs the face normal


def category(nz):
    nz = np.asarray(nz)
    return np.where(nz >= 0.7, 0, np.where(nz > 0.02, 1, np.where(nz >= -0.02, 2, 3)))


def bsp_signature(bsp) -> str:
    """the .bsp's identity as every per-map cache in this repo records it (size + mtime_ns)"""
    st = Path(bsp).stat()
    return f"{st.st_size}_{st.st_mtime_ns}"


def load_faces(mesh_path, bsp_path=None):
    """the solid faces of the exported mesh -> (tris (T,3,3) float64, outward normals (T,3),
    model (T,) int: 0 = world, k = brush model *k). World + the brush entities physics collides
    with (surfgym.vision.SOLID_ENT_CLASSES, NOTSOLID conveyors dropped) - the same triangle set
    as surfgym.surfmask.load_solid_tris, plus the exporter's face normals."""
    from surfgym.surfmask import _notsolid_conveyor_models
    from surfgym.vision import SOLID_ENT_CLASSES
    doc = json.loads(Path(mesh_path).read_text(encoding="utf-8"))
    brushes = doc.get("brushes") or []
    notsolid = ()
    if bsp_path and any(b.get("classname") == "func_conveyor" for b in brushes):
        notsolid = _notsolid_conveyor_models(bsp_path)
    parts = [(0, doc["world"])]
    parts += [(int(b.get("model") or -1), b) for b in brushes
              if b.get("classname") in SOLID_ENT_CLASSES and b.get("model") not in notsolid]
    T, N, M = [], [], []
    for mdl, part in parts:
        idx = np.asarray(part["indices"], dtype=np.int64).reshape(-1, 3)
        if not idx.size:
            continue
        pos = np.asarray(part["positions"], dtype=np.float64).reshape(-1, 3)
        nrm = np.asarray(part["normals"], dtype=np.float64).reshape(-1, 3)
        T.append(pos[idx])
        N.append(nrm[idx[:, 0]])
        M.append(np.full(len(idx), mdl, np.int64))
    if not T:
        return np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    n = np.concatenate(N)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.concatenate(T), n, np.concatenate(M)


def clip_dressing(core, mesh_path, samples=5, need=3, tol=8.0, seed=0):
    """Triangles of the NON-SOLID visible brush entities (func_illusionary) that lie on the
    PLAYER's collision surface - the visible dressing of a CLIP brush, which has no rendered
    face of its own. A triangle is kept when >= `need` of `samples` points on it have a
    ducked-hull contact (a trace along the face normal from either side) within +-`tol` u of the
    point pushed out by the ducked box's support, with the same plane normal (dot > 0.98); the
    side that contact came from is the triangle's outward side. A HEURISTIC (--clip-dressing).
    -> (tris (K, 3, 3) float64, outward unit normals (K, 3))"""
    doc = json.loads(Path(mesh_path).read_text(encoding="utf-8"))
    parts = [b for b in (doc.get("brushes") or []) if b.get("classname") == "func_illusionary"]
    rng = np.random.default_rng(seed)
    duck = HULL_HALF[1]
    keep_t, keep_n = [], []
    for b in parts:
        idx = np.asarray(b["indices"], dtype=np.int64)
        if not idx.size:
            continue
        pos = np.asarray(b["positions"], dtype=np.float64).reshape(-1, 3)
        for tri in pos[idx].reshape(-1, 3, 3):
            cr = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            ar = 0.5 * float(np.linalg.norm(cr))
            if ar <= 1e-3:
                continue
            n = cr / (2.0 * ar)
            votes = {1: 0, -1: 0}
            for _ in range(samples):
                r1, r2 = rng.random(2)
                s1 = np.sqrt(r1)
                q = (1 - s1) * tri[0] + s1 * (1 - r2) * tri[1] + s1 * r2 * tri[2]
                for sg in (1, -1):
                    nn = n * sg
                    h = float(duck @ np.abs(nn))
                    tr = core.trace((q + nn * (h + 24.0)).tolist(), (q - nn * 24.0).tolist(), 1)
                    if tr.startsolid or tr.fraction >= 1.0:
                        continue
                    c = np.array(tr.endpos[:], np.float64)
                    if (abs(float((c - q) @ nn) - h) <= tol
                            and float(np.array(tr.normal[:], np.float64) @ nn) > 0.98):
                        votes[sg] += 1
                        break
            sg = 1 if votes[1] >= votes[-1] else -1
            if votes[sg] >= need:
                keep_t.append(tri)
                keep_n.append(n * sg)
    if not keep_t:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    return np.asarray(keep_t), np.asarray(keep_n)


def touching_pairs(tris, step=EDGE_STEP, radius=EDGE_R):
    """(T,3,3) -> (P, 2) int64 pairs (i < j) of triangles whose edges come within `radius` u of
    each other (edge samples every `step` u; a shared edge or a T-junction)"""
    from scipy.spatial import cKDTree
    tids, pts = [], []
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        seg = tris[:, b_] - tris[:, a_]
        k = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / step).astype(np.int64))
        tid = np.repeat(np.arange(len(tris)), k + 1)
        # the fraction along each edge: 0..1 in k steps, built without a Python loop per edge
        off = np.concatenate([[0], np.cumsum(k + 1)[:-1]])
        j = np.arange(len(tid)) - np.repeat(off, k + 1)
        frac = j / np.repeat(k, k + 1)
        tids.append(tid)
        pts.append(tris[tid, a_] + seg[tid] * frac[:, None])
    tid = np.concatenate(tids)
    p = np.concatenate(pts)
    pr = cKDTree(p).query_pairs(r=radius, output_type="ndarray")
    a, b = tid[pr[:, 0]], tid[pr[:, 1]]
    keep = a != b
    pr = np.stack([np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])], axis=1)
    return np.unique(pr, axis=0) if len(pr) else np.zeros((0, 2), np.int64)


def group_surfaces(tris, n, area, pairs):
    """plane pieces (same plane + touching), then curved merges -> per triangle a surface label"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    T = len(tris)
    d = np.einsum("ij,ij->i", n, tris.mean(axis=1))
    pk = np.concatenate([np.round(n * 1000.0), np.round(d * 2.0)[:, None]], axis=1).astype(np.int64)
    _u, plane_id = np.unique(pk, axis=0, return_inverse=True)
    plane_id = plane_id.reshape(-1)
    same = plane_id[pairs[:, 0]] == plane_id[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    sp = pairs[same]
    g = coo_matrix((np.ones(len(sp)), (sp[:, 0], sp[:, 1])), shape=(T, T))
    _n, piece = connected_components(g, directed=False)
    piece = np.unique(piece, return_inverse=True)[1].reshape(-1)
    P = int(piece.max()) + 1 if T else 0
    p_area = np.bincount(piece, weights=area, minlength=P)
    p_n = np.zeros((P, 3))
    np.add.at(p_n, piece, n * area[:, None])
    p_n /= np.maximum(np.linalg.norm(p_n, axis=1, keepdims=True), 1e-9)
    # touching pieces (any plane) within MERGE_DEG merge, each group within CAP_DEG of its seed
    pp = np.unique(np.sort(np.stack([piece[pairs[:, 0]], piece[pairs[:, 1]]], axis=1), axis=1),
                   axis=0) if len(pairs) else np.zeros((0, 2), np.int64)
    pp = pp[pp[:, 0] != pp[:, 1]]
    cos_m, cos_c = np.cos(np.radians(MERGE_DEG)), np.cos(np.radians(CAP_DEG))
    parent = np.arange(P)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seed_n = p_n.copy()                 # a group's seed: its LARGEST piece's normal
    order = np.argsort(-np.maximum(p_area[pp[:, 0]], p_area[pp[:, 1]])) if len(pp) else []
    for q in order:
        i_, j_ = int(pp[q, 0]), int(pp[q, 1])
        if float(p_n[i_] @ p_n[j_]) < cos_m:
            continue
        ri, rj = find(i_), find(j_)
        if ri == rj:
            continue
        if float(seed_n[ri] @ p_n[j_]) < cos_c or float(seed_n[rj] @ p_n[i_]) < cos_c:
            continue
        big, small = (ri, rj) if p_area[ri] >= p_area[rj] else (rj, ri)
        parent[small] = big
    lab = np.array([find(int(x)) for x in piece]) if T else np.zeros(0, np.int64)
    return np.unique(lab, return_inverse=True)[1].reshape(-1) if T else lab


def validate_contacts(core, pts, n, plane, hull):
    """per sample point on a face with outward normal n: a `hull` trace (0 standing, 1 ducked)
    along -n onto it -> (accepted (M,) bool, contact origins (M, 3), stop offsets (M,)).

    Where the hull stops is NOT the box support: the compiler's hull expansion varies by map and
    by plane - utopia's and blue025's ramps stop at 16|nx| + 16|ny| + 36|nz| to 0.03 u, while
    petrus's, cannonball's and uf2's stop 11-16 u short standing (7-12 u ducked), and floors are
    mostly 1.57 u short. So the stop is MEASURED: a sample's hit counts when it is not start-solid,
    the hit normal is within CONTACT_COS of n and the stop offset e = (stop - p) . n lies in
    [0.25 h, h + 1]; then, per exact plane (`plane`, one id per sample) the median offset is that
    plane's expansion and only samples within CONTACT_TOL of it are accepted - a sample that fell
    through a hole onto a parallel surface below disagrees with its plane's other samples. The
    accepted stops are where the player's origin really is when its hull touches that face."""
    half = HULL_HALF[hull]
    e = np.full(len(pts), np.nan)
    org = np.zeros((len(pts), 3))
    for i in range(len(pts)):
        h = float(half @ np.abs(n[i]))
        tr = core.trace((pts[i] + n[i] * (h + 24.0)).tolist(), (pts[i] - n[i] * 8.0).tolist(),
                        hull)
        if tr.startsolid or tr.allsolid or tr.fraction >= 1.0:
            continue
        if float(np.array(tr.normal[:], np.float64) @ n[i]) < CONTACT_COS:
            continue
        stop = np.array(tr.endpos[:], np.float64)
        off = float((stop - pts[i]) @ n[i])
        if 0.25 * h <= off <= h + 1.0:
            e[i] = off
            org[i] = stop
    ok = np.zeros(len(pts), bool)
    fin = np.isfinite(e)
    for pl in np.unique(plane[fin]):
        m = fin & (plane == pl)
        med = float(np.median(e[m]))
        ok[m] = np.abs(e[m] - med) <= CONTACT_TOL
    return ok, org, e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bsp")
    ap.add_argument("--mesh", default=None, help="default viewer/assets/<map>.mesh.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=float, default=32.0,
                    help="one contact sample per this many u squared of floor / ramp face")
    ap.add_argument("--min-area", type=float, default=256.0,
                    help="surfaces smaller than this (u^2) are dropped as trim")
    ap.add_argument("--clip-dressing", action="store_true",
                    help="HEURISTIC: add the func_illusionary faces that lie on the player's "
                         "collision surface (the dressing of CLIP-brush ramps)")
    a = ap.parse_args(argv)
    from surfgym.core import SurfCore, default_config
    from ramps import in_kill
    bsp = Path(a.bsp)
    mesh = Path(a.mesh) if a.mesh else ROOT / "viewer" / "assets" / f"{bsp.stem}.mesh.json"
    tris, n, model = load_faces(str(mesh), str(bsp))
    cr = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 1e-3
    tris, n, model, area = tris[ok], n[ok], model[ok], area[ok]
    tri_src = np.zeros(len(tris), np.int8)
    core = SurfCore(str(bsp), default_config(num_envs=1))
    if a.clip_dressing:
        dt, dn = clip_dressing(core, mesh)
        if len(dt):
            d_ar = 0.5 * np.linalg.norm(np.cross(dt[:, 1] - dt[:, 0], dt[:, 2] - dt[:, 0]), axis=1)
            tris = np.concatenate([tris, dt])
            n = np.concatenate([n, dn])
            area = np.concatenate([area, d_ar])
            model = np.concatenate([model, np.full(len(dt), -1, np.int64)])
            tri_src = np.concatenate([tri_src, np.ones(len(dt), np.int8)])
        print(f"ramps_mesh {bsp.stem}: --clip-dressing: {len(dt):,} func_illusionary triangles on "
              f"the player's collision surface added")
    pairs = touching_pairs(tris)
    lab = group_surfaces(tris, n, area, pairs)
    S0 = int(lab.max()) + 1 if len(lab) else 0
    s_area = np.bincount(lab, weights=area, minlength=S0)
    keep = s_area >= float(a.min_area)
    remap = -np.ones(S0, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    tri_surf = remap[lab]
    S = int(keep.sum())
    s_n = np.zeros((S, 3))
    s_a = np.zeros(S)
    for i in np.flatnonzero(tri_surf >= 0):
        s_n[tri_surf[i]] += n[i] * area[i]
        s_a[tri_surf[i]] += area[i]
    s_n /= np.maximum(np.linalg.norm(s_n, axis=1, keepdims=True), 1e-9)
    cat = category(s_n[:, 2])
    # contact samples on floors and ramps only (the only targets), area-uniform per surface; each
    # sample carries its triangle's EXACT plane (the key group_surfaces joins pieces on)
    d_tri = np.einsum("ij,ij->i", n, tris.mean(axis=1))
    pk = np.concatenate([np.round(n * 1000.0), np.round(d_tri * 2.0)[:, None]], axis=1)
    tri_plane = np.unique(pk.astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
    rng = np.random.default_rng(0)
    sp, snm, ssid, spl = [], [], [], []
    for s in np.flatnonzero(np.isin(cat, (0, 1))):
        ti = np.flatnonzero(tri_surf == s)
        spacing = max(float(a.sample), float(np.sqrt(s_a[s] / MAX_SAMPLES)))
        for i in ti:
            k = max(1, int(np.ceil(area[i] / spacing ** 2)))
            r1, r2 = rng.random(k), rng.random(k)
            s1 = np.sqrt(r1)
            p = ((1 - s1)[:, None] * tris[i, 0] + (s1 * (1 - r2))[:, None] * tris[i, 1]
                 + (s1 * r2)[:, None] * tris[i, 2])
            sp.append(p)
            snm.append(np.repeat(n[i][None], k, axis=0))
            ssid.append(np.full(k, s))
            spl.append(np.full(k, tri_plane[i]))
    sp = np.concatenate(sp) if sp else np.zeros((0, 3))
    snm = np.concatenate(snm) if snm else np.zeros((0, 3))
    ssid = np.concatenate(ssid) if ssid else np.zeros(0, np.int64)
    spl = np.concatenate(spl) if spl else np.zeros(0, np.int64)
    ok_s, org_s, e_s = validate_contacts(core, sp, snm, spl, 0)
    ok_d, org_d, e_d = validate_contacts(core, sp, snm, spl, 1)
    touch = np.zeros(S)
    for s in np.unique(ssid):
        m = ssid == s
        touch[s] = float((ok_s[m] | ok_d[m]).mean())
    hidden = np.isin(cat, (0, 1)) & (touch == 0.0)
    cat[hidden] = 5
    # kill: a floor / ramp whose standing contact origins are mostly inside a kill trigger
    kp = in_kill(bsp, org_s[ok_s]) if ok_s.any() else np.zeros(0, bool)
    sid_s = ssid[ok_s]
    for s in np.unique(sid_s):
        m = sid_s == s
        if kp[m].mean() > 0.5:
            cat[s] = 4
    pts, nrm, sid = org_s[ok_s], snm[ok_s], ssid[ok_s]
    dpts, dnrm, dsid = org_d[ok_d], snm[ok_d], ssid[ok_d]
    # per surface: the median measured stop offset minus the box support (0 = the exact box
    # expansion; negative = the compile expanded that plane less), standing and ducked
    sup_s = np.abs(snm) @ HULL_HALF[0]
    sup_d = np.abs(snm) @ HULL_HALF[1]
    exp_s = np.full(S, np.nan)
    exp_d = np.full(S, np.nan)
    for s in range(S):
        m = (ssid == s) & ok_s
        if m.any():
            exp_s[s] = float(np.median(e_s[m] - sup_s[m]))
        m = (ssid == s) & ok_d
        if m.any():
            exp_d[s] = float(np.median(e_d[m] - sup_d[m]))
    # the goal potential per surface (its standing contact origins), as tools/ramps.py does
    gpaths = sorted(bsp.parent.glob(f"{bsp.stem}.goal_*.npz"))
    gpaths = [p for p in gpaths if p.stem.split(".goal_")[-1].isdigit()]
    d_p10 = np.full(S, np.nan)
    d_med = np.full(S, np.nan)
    if gpaths and len(pts):
        from surfgym.goalfield import load_goal_field
        gf = load_goal_field(str(gpaths[0]))
        dd = gf.sample(pts)
        okd = gf.reachable(pts)
        for s in range(S):
            m = (sid == s) & okd
            if m.any():
                d_p10[s] = float(np.percentile(dd[m], 10))
                d_med[s] = float(np.median(dd[m]))
    # order like ramps.py: by category, then far-from-goal first
    order = sorted(range(S), key=lambda s: (int(cat[s]), -(d_p10[s] if np.isfinite(d_p10[s])
                                                           else -1.0)))
    new_id = np.empty(S, np.int64)
    new_id[order] = np.arange(S)
    sid, dsid = new_id[sid], new_id[dsid]
    tri_surf = np.where(tri_surf >= 0, new_id[np.maximum(tri_surf, 0)], -1)
    cat, s_n, s_a, d_p10, d_med, touch = (cat[order], s_n[order], s_a[order], d_p10[order],
                                          d_med[order], touch[order])
    exp_s, exp_d = exp_s[order], exp_d[order]
    s_d = np.array([float(np.average(np.einsum("ij,j->i", tris[tri_surf == s].mean(axis=1), s_n[s]),
                                     weights=area[tri_surf == s])) for s in range(S)])
    cnt = {c: int((cat == i).sum()) for i, c in enumerate(CATS)}
    n_samp = int(len(sp))
    print(f"ramps_mesh {bsp.stem} (v{VERSION}): {len(tris):,} solid triangles, {len(pairs):,} "
          f"touching pairs -> {S} surfaces >= {a.min_area:g} u^2: "
          + ", ".join(f"{v} {k}" for k, v in cnt.items()))
    print(f"   contact samples on floors / ramps: {n_samp:,}; validated standing "
          f"{int(ok_s.sum()):,} ({100 * ok_s.mean() if n_samp else 0:.1f}%), ducked "
          f"{int(ok_d.sum()):,} ({100 * ok_d.mean() if n_samp else 0:.1f}%); "
          f"{int(hidden.sum())} floor / ramp surfaces no hull touches (hidden)")
    out = Path(a.out) if a.out else ROOT / "runs" / "research" / f"ramps_mesh_{bsp.stem}.npz"
    mesh_sha1 = hashlib.sha1(Path(mesh).read_bytes()).hexdigest()
    np.savez_compressed(
        out, points=pts.astype(np.float32), normals=nrm.astype(np.float32), surf=sid,
        duck_points=dpts.astype(np.float32), duck_normals=dnrm.astype(np.float32), duck_surf=dsid,
        cat=cat, n=np.bincount(sid, minlength=S),
        centroid=np.array([pts[sid == s].mean(0) if (sid == s).any()
                           else tris[tri_surf == s].mean(axis=(0, 1)) for s in range(S)]),
        normal=s_n, area=s_a, plane_d=s_d, touchable=touch, expand_stand=exp_s,
        expand_duck=exp_d, d_min=d_p10, d_p10=d_p10,
        d_med=d_med, d_max=d_med, map=bsp.stem, cell=float(a.sample), version=VERSION,
        source="mesh+collision", bsp_sig=bsp_signature(bsp), mesh_sha1=mesh_sha1,
        params=json.dumps({"sample": float(a.sample), "min_area": float(a.min_area),
                           "clip_dressing": bool(a.clip_dressing), "merge_deg": MERGE_DEG,
                           "cap_deg": CAP_DEG, "edge_step": EDGE_STEP, "edge_r": EDGE_R,
                           "contact_tol": CONTACT_TOL, "contact_cos": CONTACT_COS}),
        tris=tris.astype(np.float32), tri_n=n.astype(np.float32), tri_surf=tri_surf,
        tri_src=tri_src, tri_model=model)
    print(f"   -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
