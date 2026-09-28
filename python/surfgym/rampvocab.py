"""rampvocab.py - a map's TARGET surfaces (tools/ramps_mesh.py v4: faces grounded in collision),
bound to the .bsp they were extracted from, and the collision-telemetry classifier: which of those
surfaces did each env's hull touch on this tick (SurfCore.get_touch, the planes the movement hit).

    voc = RampVocab("runs/research/ramps_mesh_<map>.npz", bsp_path)   # refuses another map
    ids = voc.classify(counts, normals, points, ducked)                # (N, 8) surface ids, -1
    voc.targets                                                        # floors + ramps a hull touches

A touch is matched to the nearest VALIDATED contact origin of the same hull (standing / ducked):
the places ramps_mesh measured the player's origin to be when its hull touches that face. Those
origins lie on the compiled hull's own expanded plane, which is NOT the box support on every map
(ramps_mesh.validate_contacts: petrus / cannonball / uf2 ramps stop 11-16 u short), so the plane
test here is exact whatever the compiler did: the touch's origin must lie on the sample's plane
(PLANE_TOL), with the same normal (NORMAL_COS), within the surface's own sample spacing laterally.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PLANE_TOL = 2.5                        # u: a grounded player hovers up to ~2 u over its floor
NORMAL_COS = float(np.cos(np.radians(2.0)))
LATERAL_MIN = 48.0                     # u: the lateral match radius is max(this, 1.5 x spacing)
K_NEAR = 8


def edge_pieces(tris, ts, n_surf):
    """surface -> PIECE id (-1 for a surface with no triangle here): surfaces whose triangle edges
    share a 16 u cell (edge samples every 8 u) are one physical piece - a wedge's two sides and
    its end caps. tris (m, 3, 3), ts (m,) the surface of each triangle."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    obj = -np.ones(n_surf, np.int64)
    if len(ts) == 0:
        return obj
    eds = []
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        seg = tris[:, b_] - tris[:, a_]
        k = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / 8.0).astype(np.int64))
        tid = np.repeat(np.arange(len(tris)), k + 1)
        frac = np.concatenate([np.linspace(0.0, 1.0, kk + 1) for kk in k])
        eds.append((ts[tid], np.floor((tris[tid, a_] + seg[tid] * frac[:, None]) / 16.0)))
    sid = np.concatenate([e[0] for e in eds])
    cell = np.concatenate([e[1] for e in eds]).astype(np.int64)
    _c, cid = np.unique(cell, axis=0, return_inverse=True)
    cid = cid.reshape(-1)
    g = coo_matrix((np.ones(len(sid)), (sid, n_surf + cid)),
                   shape=(n_surf + len(_c), n_surf + len(_c)))
    _n, lab = connected_components(g, directed=False)
    used = np.unique(ts)
    _l, compact = np.unique(lab[used], return_inverse=True)
    obj[used] = compact.reshape(-1)
    return obj


def bsp_signature(bsp) -> str:
    st = Path(bsp).stat()
    return f"{st.st_size}_{st.st_mtime_ns}"


class RampVocab:
    def __init__(self, npz_path, bsp_path=None, check: bool = True):
        from scipy.spatial import cKDTree
        z = np.load(npz_path, allow_pickle=False)
        ver = int(z["version"]) if "version" in z else 0
        if ver < 4:
            raise ValueError(f"{npz_path}: ramps vocabulary v{ver} - re-extract with "
                             "tools/ramps_mesh.py (v4 is grounded in collision; earlier "
                             "versions' contact origins are not)")
        self.path = str(npz_path)
        self.map = str(z["map"])
        self.bsp_sig = str(z["bsp_sig"])
        self.mesh_sha1 = str(z["mesh_sha1"])
        self.version = ver
        if bsp_path is not None and check:
            if Path(bsp_path).stem != self.map:
                raise ValueError(f"{npz_path} was extracted from {self.map}, not "
                                 f"{Path(bsp_path).stem}")
            sig = bsp_signature(bsp_path)
            if sig != self.bsp_sig:
                raise ValueError(f"{npz_path} was extracted from a {self.map}.bsp with signature "
                                 f"{self.bsp_sig}; this one is {sig} - re-extract (or restore "
                                 "the map's mtime, tools/restamp_maps.py)")
        self.cat = z["cat"].astype(np.int64)
        self.normal = z["normal"].astype(np.float64)
        self.area = z["area"].astype(np.float64)
        self.touchable = z["touchable"].astype(np.float64)
        self.n_surf = len(self.cat)
        self.points = z["points"].astype(np.float64)          # standing contact origins
        self.normals = z["normals"].astype(np.float64)
        self.surf = z["surf"].astype(np.int64)
        self.dpoints = z["duck_points"].astype(np.float64)    # ducked contact origins
        self.dnormals = z["duck_normals"].astype(np.float64)
        self.dsurf = z["duck_surf"].astype(np.int64)
        # the targets: floors and ramps the STANDING hull touches (hidden / kill / walls /
        # ceilings are not; nor is a surface only a ducked player can reach)
        n_stand = np.bincount(self.surf, minlength=self.n_surf)
        self.targets = np.flatnonzero(np.isin(self.cat, (0, 1)) & (self.touchable > 0)
                                      & (n_stand > 0))
        self.is_target = np.zeros(self.n_surf, bool)
        self.is_target[self.targets] = True
        # PIECES: target surfaces joined at an edge (a wedge's two sides and its end caps) are
        # one physical piece - what the ramp windows choose, capture and leave (a target with no
        # triangle in the file, or a vocabulary without triangles: its own piece)
        self.piece = np.full(self.n_surf, -1, np.int64)
        if "tris" in z.files and "tri_surf" in z.files and len(self.targets):
            tr = z["tris"].astype(np.float64)
            ts = z["tri_surf"].astype(np.int64)
            sel = np.isin(ts, self.targets)
            self.piece = edge_pieces(tr[sel], ts[sel], self.n_surf)
        lone = self.targets[self.piece[self.targets] < 0]
        self.piece[lone] = int(self.piece.max()) + 1 + np.arange(len(lone))
        self.piece_faces = {}
        for s in self.targets:
            self.piece_faces.setdefault(int(self.piece[s]), []).append(int(s))
        # per surface the lateral match radius: 1.5 x its sample spacing, at least LATERAL_MIN
        n_s = np.bincount(self.surf, minlength=self.n_surf).astype(np.float64)
        spacing = np.sqrt(self.area / np.maximum(n_s, 1.0))
        self.lateral = np.maximum(LATERAL_MIN, 1.5 * spacing)
        self._tree = [cKDTree(self.points) if len(self.points) else None,
                      cKDTree(self.dpoints) if len(self.dpoints) else None]
        self._pn = [self.normals, self.dnormals]
        self._pp = [self.points, self.dpoints]
        self._ps = [self.surf, self.dsurf]
        self._rmax = float(self.lateral.max()) if self.n_surf else LATERAL_MIN
        # per target: its standing origins (subsampled) for arc queries by the window planner
        rng = np.random.default_rng(0)
        self.tp, self.tn = {}, {}
        for s in self.targets:
            m = np.flatnonzero(self.surf == s)
            if len(m) > 400:
                m = np.sort(rng.choice(m, 400, replace=False))
            self.tp[int(s)] = self.points[m]
            self.tn[int(s)] = self.normals[m]

    def describe(self) -> str:
        c = {k: int((self.cat == i).sum()) for i, k in
             enumerate(("floor", "ramp", "wall", "ceiling", "kill", "hidden"))}
        return (f"ramp vocabulary {Path(self.path).name} (v{self.version}, {self.map}, sig "
                f"{self.bsp_sig}): {len(self.targets)} targets "
                f"({int(np.isin(self.cat[self.targets], 1).sum())} ramps, "
                f"{int(np.isin(self.cat[self.targets], 0).sum())} floors) of {self.n_surf} "
                f"surfaces ({', '.join(f'{v} {k}' for k, v in c.items())}); "
                f"{len(self.points):,} standing / {len(self.dpoints):,} ducked contact origins")

    def classify(self, counts, normals, points, ducked):
        """collision truth -> (N, T) int64 surface ids per telemetry slot, -1 where the touch is
        not a surface of this vocabulary (a wall, a ceiling, trim, a clip brush no face stands
        for). counts (N,), normals / points (N, T, 3) as SurfCore.get_touch returns them;
        ducked (N,) the hull each env used (0 standing, 1 ducked)."""
        counts = np.asarray(counts)
        N, T = normals.shape[0], normals.shape[1]
        out = np.full((N, T), -1, np.int64)
        rows, cols = np.nonzero(np.arange(T)[None, :] < counts[:, None])
        if len(rows) == 0:
            return out
        tp = np.asarray(points[rows, cols], np.float64)
        tn = np.asarray(normals[rows, cols], np.float64)
        hull = (np.asarray(ducked)[rows] != 0).astype(np.int64)
        for h in (0, 1):
            sel = np.flatnonzero(hull == h)
            if not len(sel) or self._tree[h] is None:
                continue
            dist, idx = self._tree[h].query(tp[sel], k=K_NEAR, distance_upper_bound=self._rmax)
            ok = np.isfinite(dist)
            ii = np.where(ok, idx, 0)
            pn = self._pn[h][ii]                                        # (m, K, 3)
            pp = self._pp[h][ii]
            ps = self._ps[h][ii]
            ok &= np.einsum("mkj,mj->mk", pn, tn[sel]) >= NORMAL_COS
            ok &= np.abs(np.einsum("mkj,mkj->mk", pn, tp[sel][:, None, :] - pp)) <= PLANE_TOL
            ok &= dist <= self.lateral[ps]
            dd = np.where(ok, dist, np.inf)
            j = np.argmin(dd, axis=1)
            hit = np.isfinite(dd[np.arange(len(sel)), j])
            sid = np.where(hit, ps[np.arange(len(sel)), j], -1)
            out[rows[sel], cols[sel]] = sid
        return out

    def contact_of(self, origins, ducked, plane_tol: float = PLANE_TOL):
        """(M, 3) player origins, (M,) hulls -> (M,) the surface each one is resting on / touching
        (its origin on a validated contact plane within plane_tol, laterally within the
        surface's radius), -1 if none - a spawn's SOURCE, which is never drawn as its target"""
        o = np.atleast_2d(np.asarray(origins, np.float64))
        hull = (np.asarray(ducked).reshape(-1) != 0).astype(np.int64)
        out = np.full(len(o), -1, np.int64)
        for h in (0, 1):
            sel = np.flatnonzero(hull == h)
            if not len(sel) or self._tree[h] is None:
                continue
            dist, idx = self._tree[h].query(o[sel], k=K_NEAR, distance_upper_bound=self._rmax)
            ok = np.isfinite(dist)
            ii = np.where(ok, idx, 0)
            pn, pp, ps = self._pn[h][ii], self._pp[h][ii], self._ps[h][ii]
            ok &= np.abs(np.einsum("mkj,mkj->mk", pn, o[sel][:, None, :] - pp)) <= plane_tol
            ok &= dist <= self.lateral[ps]
            dd = np.where(ok, dist, np.inf)
            j = np.argmin(dd, axis=1)
            hit = np.isfinite(dd[np.arange(len(sel)), j])
            out[sel] = np.where(hit, ps[np.arange(len(sel)), j], -1)
        return out

    def targets_in(self, ids):
        """(N, T) classify output -> (N, T) bool: the touch is a TARGET surface"""
        ids = np.asarray(ids)
        return (ids >= 0) & self.is_target[np.maximum(ids, 0)]
