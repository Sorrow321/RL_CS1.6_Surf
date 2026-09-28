"""ramps_mesh.py - surfaces straight from the map's FACES (the user, 2026-09-28: "in a BSP these
ramps should be easily extractable"). The exported render mesh (viewer/assets/<map>.mesh.json,
the triangles the viewer and the surf mask use; surfgym.surfmask.load_solid_tris - world + the
solid brush entities, triggers dropped) is grouped into flat surfaces: triangles on the SAME
plane that share a vertex are one surface - a flat ramp is exactly one group, its normal is its
plane's. No sampling, no region growing.

    python tools/ramps_mesh.py maps_pool/<map>.bsp [--out ramps_mesh_<map>.npz] [--sample 32]

1. ORIENTATION: the mesh's winding is not reliable (both sides of a thin brush are exported), so
   each plane's outward side is decided by the collision geometry: the point contents just in
   front of and behind the face (the sim's own point test); ambiguous faces take the upper
   hemisphere.
2. GROUPS: plane key = (outward normal to 1e-3, plane distance to 0.5 u); pieces = triangles of
   one plane that TOUCH (their edges, sampled every 8 u, share a 16 u cell - so a T-junction
   joins); then adjacent pieces within MERGE_DEG of each other merge (a curved ramp's facets),
   capped at CAP_DEG from the group's largest piece.
3. CATEGORIES from the outward n_z, the engine's rules: floor >= 0.7 (walkable), ramp in
   (0.02, 0.7), wall |n_z| <= 0.02, ceiling < -0.02; KILL = a surface whose standing-hull contact
   origins are mostly inside a kill trigger (zones.kill_zones + hull_probe, as tools/ramps.py).
4. CLIP DRESSING: hlcsg compiles CLIP brushes into the player hulls only - no rendered face, no
   point-hull solid - so a ramp built as a clip brush has no solid triangle at all (utopia's
   curved ramps: CLIP brushes under non-solid func_illusionary pieces). The faces of non-solid
   visible brush entities that lie ON the player's collision surface are therefore taken as that
   surface (clip_dressing; --no-clip-dressing drops them), oriented by the side the collision
   came from; the npz's tri_src says which triangles they are (1) against solid faces (0).
5. The output is tools/ramps.py's RampMap format, so the ramp operator runs on it unchanged:
   per surface, CONTACT ORIGINS - points on its triangles (one per --sample u square, at least
   3 per triangle) pushed out along the normal by the standing hull's support 16|nx| + 16|ny| +
   36|nz| (where the player's origin is when its hull touches that plane) - plus, per surface,
   its triangles (for the viewer) and plane (normal, distance).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

CATS = ("floor", "ramp", "wall", "ceiling", "kill")
STAND = np.array([16.0, 16.0, 36.0])
MERGE_DEG = 10.0       # adjacent pieces this close in normal are one (curved) surface
CAP_DEG = 25.0         # ... while every piece stays this close to the group's largest piece


def category(nz):
    nz = np.asarray(nz)
    return np.where(nz >= 0.7, 0, np.where(nz > 0.02, 1, np.where(nz >= -0.02, 2, 3)))


def orient(core, cen, n, probe=4.0):
    """-> +1 / -1 per triangle: the side of n that is EMPTY (the player's side), 0 = ambiguous"""
    out = np.zeros(len(cen), np.int8)
    for i in range(len(cen)):
        a = core.point_contents((cen[i] + n[i] * probe).tolist())
        b = core.point_contents((cen[i] - n[i] * probe).tolist())
        a_solid, b_solid = a == -2, b == -2          # CONTENTS_SOLID
        if b_solid and not a_solid:
            out[i] = 1
        elif a_solid and not b_solid:
            out[i] = -1
    return out


def clip_dressing(core, mesh_path, samples=5, need=3, tol=8.0, seed=0):
    """Triangles of the NON-SOLID visible brush entities (func_illusionary) that lie on the
    PLAYER's collision surface - the visible dressing of a CLIP brush, which has no rendered
    face of its own. A triangle is kept when >= `need` of `samples` points on it have a
    ducked-hull contact (a trace along the face normal from either side) within +-`tol` u of the
    point pushed out by the ducked box's support, with the same plane normal (dot > 0.98); the
    side that contact came from is the triangle's outward side.
    -> (tris (K, 3, 3) float64, outward unit normals (K, 3))"""
    import json
    doc = json.loads(Path(mesh_path).read_text(encoding="utf-8"))
    parts = [b for b in (doc.get("brushes") or []) if b.get("classname") == "func_illusionary"]
    rng = np.random.default_rng(seed)
    duck = np.array([16.0, 16.0, 18.0])
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bsp")
    ap.add_argument("--mesh", default=None, help="default viewer/assets/<map>.mesh.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=float, default=32.0,
                    help="one contact origin per this many u squared of face (and >= 3 per tri)")
    ap.add_argument("--min-area", type=float, default=256.0,
                    help="surfaces smaller than this (u^2) are dropped as trim")
    ap.add_argument("--no-clip-dressing", action="store_true",
                    help="leave out the func_illusionary faces that lie on the player's "
                         "collision surface (the dressing of CLIP-brush ramps)")
    a = ap.parse_args(argv)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from surfgym.core import SurfCore, default_config
    from surfgym.surfmask import load_solid_tris
    from ramps import in_kill
    bsp = Path(a.bsp)
    mesh = Path(a.mesh) if a.mesh else ROOT / "viewer" / "assets" / f"{bsp.stem}.mesh.json"
    tris = load_solid_tris(str(mesh), str(bsp))
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 1e-3
    tris, cr, area = tris[ok], cr[ok], area[ok]
    n = cr / (2.0 * area[:, None])
    cen = tris.mean(axis=1)
    core = SurfCore(str(bsp), default_config(num_envs=1))
    sgn = orient(core, cen, n)
    # an ambiguous face takes the upper hemisphere (surfmask's canonical sign)
    amb = sgn == 0
    sgn[amb] = np.where(n[amb, 2] >= 0.0, 1, -1)
    n = n * sgn[:, None]
    tri_src = np.zeros(len(tris), np.int8)
    if not a.no_clip_dressing:
        dt, dn = clip_dressing(core, mesh)
        if len(dt):
            d_ar = 0.5 * np.linalg.norm(np.cross(dt[:, 1] - dt[:, 0], dt[:, 2] - dt[:, 0]), axis=1)
            tris = np.concatenate([tris, dt])
            n = np.concatenate([n, dn])
            area = np.concatenate([area, d_ar])
            cen = tris.mean(axis=1)
            tri_src = np.concatenate([tri_src, np.ones(len(dt), np.int8)])
        print(f"ramps_mesh {bsp.stem}: {len(dt):,} func_illusionary triangles lie on the player's "
              f"collision surface (CLIP-brush dressing), {float(d_ar.sum()) if len(dt) else 0.0:,.0f} "
              f"u^2 - added as solid faces")
    d = np.einsum("ij,ij->i", n, tris[:, 0])
    # plane keys, then connected components over shared vertices WITHIN a plane
    pk = np.concatenate([np.round(n * 1000.0), np.round(d * 2.0)[:, None]], axis=1).astype(np.int64)
    _u, plane_id = np.unique(pk, axis=0, return_inverse=True)
    plane_id = plane_id.reshape(-1)
    T = len(tris)
    # EDGE cells: every triangle's edges sampled every 8 u into 16 u cells; triangles sharing a
    # cell TOUCH (a T-junction vertex lies in a cell its neighbour's edge passes through)
    eds = []
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        seg = tris[:, b_] - tris[:, a_]
        ln = np.linalg.norm(seg, axis=1)
        k = np.maximum(1, np.ceil(ln / 8.0).astype(np.int64))
        tid = np.repeat(np.arange(T), k + 1)
        frac = np.concatenate([np.linspace(0.0, 1.0, kk + 1) for kk in k])
        eds.append((tid, tris[tid, a_] + seg[tid] * frac[:, None]))
    e_tid = np.concatenate([e[0] for e in eds])
    e_cell = np.floor(np.concatenate([e[1] for e in eds]) / 16.0).astype(np.int64)
    _c, cell_id = np.unique(e_cell, axis=0, return_inverse=True)
    cell_id = cell_id.reshape(-1)
    # 1) same plane + touching -> one flat piece
    key = np.stack([plane_id[e_tid], cell_id], axis=1)
    _k, kid = np.unique(key, axis=0, return_inverse=True)
    kid = kid.reshape(-1)
    g = coo_matrix((np.ones(len(e_tid)), (e_tid, kid + T)), shape=(T + len(_k), T + len(_k)))
    ncomp, lab = connected_components(g, directed=False)
    piece = np.unique(lab[:T], return_inverse=True)[1].reshape(-1)
    P = int(piece.max()) + 1
    p_area = np.bincount(piece, weights=area, minlength=P)
    p_n = np.zeros((P, 3))
    np.add.at(p_n, piece, n * area[:, None])
    p_n /= np.maximum(np.linalg.norm(p_n, axis=1, keepdims=True), 1e-9)
    # 2) adjacent pieces (sharing an edge cell, any plane) within MERGE_DEG of each other are one
    # surface - a curved ramp's facets - while the group stays within CAP_DEG of its first piece
    pc = np.unique(np.stack([piece[e_tid], cell_id], axis=1), axis=0)
    by_cell = {}
    for pp, cc in pc:
        by_cell.setdefault(int(cc), []).append(int(pp))
    pairs = set()
    for lst in by_cell.values():
        if len(lst) > 1:
            for i_ in range(len(lst)):
                for j_ in range(i_ + 1, len(lst)):
                    pairs.add((lst[i_], lst[j_]))
    cos_m, cos_c = np.cos(np.radians(MERGE_DEG)), np.cos(np.radians(CAP_DEG))
    parent = np.arange(P)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seed_n = p_n.copy()                 # a group's seed: its LARGEST piece's normal
    order_p = sorted(pairs, key=lambda q: -max(p_area[q[0]], p_area[q[1]]))
    for i_, j_ in order_p:
        if float(p_n[i_] @ p_n[j_]) < cos_m:
            continue
        ri, rj = find(i_), find(j_)
        if ri == rj:
            continue
        if float(seed_n[ri] @ p_n[j_]) < cos_c or float(seed_n[rj] @ p_n[i_]) < cos_c:
            continue
        big, small = (ri, rj) if p_area[ri] >= p_area[rj] else (rj, ri)
        parent[small] = big
    tri_lab = np.array([find(int(x)) for x in piece])
    _l, tri_surf = np.unique(tri_lab, return_inverse=True)
    S0 = int(tri_surf.max()) + 1
    s_area = np.bincount(tri_surf, weights=area, minlength=S0)
    keep = s_area >= float(a.min_area)
    remap = -np.ones(S0, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    tri_surf = remap[tri_surf]
    S = int(keep.sum())
    # contact origins: area-uniform points on each triangle, pushed out by the standing support
    rng = np.random.default_rng(0)
    pts, nrm, sid = [], [], []
    for i in np.flatnonzero(tri_surf >= 0):
        k = max(3, int(np.ceil(area[i] / (a.sample ** 2))))
        r1, r2 = rng.random(k), rng.random(k)
        s1 = np.sqrt(r1)
        p = ((1 - s1)[:, None] * tris[i, 0] + (s1 * (1 - r2))[:, None] * tris[i, 1]
             + (s1 * r2)[:, None] * tris[i, 2])
        sup = float(STAND @ np.abs(n[i]))
        pts.append(p + n[i] * sup)
        nrm.append(np.repeat(n[i][None], k, axis=0))
        sid.append(np.full(k, tri_surf[i]))
    pts = np.concatenate(pts)
    nrm = np.concatenate(nrm)
    sid = np.concatenate(sid)
    # per surface: category (kill if its contact origins are mostly inside a kill trigger)
    s_n = np.zeros((S, 3))
    s_a = np.zeros(S)
    for i in np.flatnonzero(tri_surf >= 0):
        s_n[tri_surf[i]] += n[i] * area[i]
        s_a[tri_surf[i]] += area[i]
    s_n /= np.maximum(np.linalg.norm(s_n, axis=1, keepdims=True), 1e-9)
    cat = category(s_n[:, 2])
    kp = in_kill(bsp, pts)
    for s in range(S):
        m = sid == s
        if m.any() and kp[m].mean() > 0.5:
            cat[s] = 4
    # the goal potential per surface (reachable samples), as tools/ramps.py does
    gpaths = sorted(bsp.parent.glob(f"{bsp.stem}.goal_*.npz"))
    gpaths = [p for p in gpaths if p.stem.split(".goal_")[-1].isdigit()]
    d_p10 = np.full(S, np.nan)
    d_med = np.full(S, np.nan)
    if gpaths:
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
    sid = new_id[sid]
    tri_surf = np.where(tri_surf >= 0, new_id[np.maximum(tri_surf, 0)], -1)
    cat, s_n, s_a, d_p10, d_med = cat[order], s_n[order], s_a[order], d_p10[order], d_med[order]
    s_d = np.array([float(np.median(np.einsum("ij,j->i", pts[sid == s], s_n[s])))
                    for s in range(S)])
    cnt = {c: int((cat == i).sum()) for i, c in enumerate(CATS)}
    print(f"ramps_mesh {bsp.stem}: {len(tris):,} solid triangles ({int(amb.sum()):,} with an "
          f"ambiguous side) -> {S} surfaces >= {a.min_area:g} u^2: "
          + ", ".join(f"{v} {k}" for k, v in cnt.items())
          + f"; {len(pts):,} contact origins")
    out = Path(a.out) if a.out else ROOT / "runs" / "research" / f"ramps_mesh_{bsp.stem}.npz"
    np.savez_compressed(out, points=pts.astype(np.float32), normals=nrm.astype(np.float32),
                        surf=sid, cat=cat, n=np.bincount(sid, minlength=S),
                        centroid=np.array([pts[sid == s].mean(0) for s in range(S)]),
                        normal=s_n, area=s_a, plane_d=s_d, d_min=d_p10, d_p10=d_p10,
                        d_med=d_med, d_max=d_med, map=bsp.stem, cell=float(a.sample),
                        version=3, source="mesh",
                        tris=tris.astype(np.float32), tri_surf=tri_surf, tri_src=tri_src)
    print(f"   -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
