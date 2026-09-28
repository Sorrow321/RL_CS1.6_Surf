"""targetmask.py - the TARGET-RAMP channel: where in the lidar's view the next ramp is (+1) and the
one after it (-1); everything else 0 (the user, 2026-09-28: "the geometry comes from the depth
render; the destination ramps are a separate channel ... rasterize, ignoring walls").

The target is drawn like a HUD objective marker: WITHOUT occlusion - a wall between the eye and
the target does not hide it - so it always carries the target's direction and shape, and the
depth channel carries the walls. (The occluded variant was built first, a4f8b0e: it needed the
depth march's hit per ray and a patch for the march's grazing error, and cost 3x the depth render.)

It is close to free: a target surface is a few PLANES (one for a flat ramp; a curved ramp or a
whole object is several), and each plane is baked once into a 2D occupancy mask in its own
coordinates (8 u cells, one cell of dilation so the BSP's T-junction cracks close). Per ray and
per plane: one ray-plane intersection and one mask lookup - independent of how finely the BSP
compiler tessellated the surface. The nearer of the two targets wins where they overlap.

    tm = TargetMask(mesh_npz, finish_box=None, device="cuda", unit="face")   # or unit="object"
    tm.set_targets(n_envs, t1, t2)                   # surface / object ids, FIN, NONE
    chan = tm.render(lidar, origin, yaw, pitch, ducked)   # (N, H, W) in {-1, 0, +1}
    tm.set_slots(n_envs, ids, vals)   # up to three slots with any values (a cross-fade)

unit="object": a target is a whole connected object (e.g. an A-frame's two slopes and its caps:
target faces touching at an edge), not one slope - coarser but more forgiving.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                       # pragma: no cover
    HAVE_TRITON = False

FIN = -2          # the finish box as a target id
NONE = -1
TM_BLOCK = 128
RES = 8.0         # u: the plane masks' cell
MERGE_DEG = 3.0   # a target's pieces within this angle and MERGE_DIST of a plane share its mask
MERGE_DIST = 16.0
MAX_CELLS = 1 << 22   # per-plane mask budget; a larger plane coarsens its cell


def _plane_groups(tris, ts):
    """(T,3,3), (T,) target ids -> per triangle a PLANE id: within one target, the largest
    triangle seeds a plane and every triangle within MERGE_DEG of its normal and MERGE_DIST of it
    joins (the BSP compile leaves near-coplanar pieces of one ramp with slightly different planes;
    a curved ramp becomes a few planes, one per MERGE_DEG of turn)"""
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(2.0 * area[:, None], 1e-12)
    flip = (n[:, 2] < -1e-6) | ((np.abs(n[:, 2]) <= 1e-6) & ((n[:, 1] < -1e-6)
                                                           | ((np.abs(n[:, 1]) <= 1e-6)
                                                              & (n[:, 0] < 0))))
    n[flip] *= -1.0
    cen = tris.mean(axis=1)
    pid = -np.ones(len(tris), np.int64)
    cos_m = np.cos(np.radians(MERGE_DEG))
    nxt = 0
    for s in np.unique(ts):
        idx = np.flatnonzero(ts == s)
        idx = idx[np.argsort(-area[idx])]
        free = np.ones(len(idx), bool)
        for a in range(len(idx)):
            if not free[a]:
                continue
            seed = idx[a]
            ns = n[seed]
            ds = float(ns @ cen[seed])
            cand = idx[free]
            close = (n[cand] @ ns >= cos_m) & (np.abs(cen[cand] @ ns - ds) <= MERGE_DIST)
            pid[cand[close]] = nxt
            free[np.isin(idx, cand[close])] = False
            nxt += 1
    return pid, n, area


def _rasterize(uv_tris, res):
    """(k,3,2) triangles in plane coordinates -> (u0, v0, nu, nv, mask uint8 [nv, nu])"""
    lo = uv_tris.reshape(-1, 2).min(0) - 2 * res
    hi = uv_tris.reshape(-1, 2).max(0) + 2 * res
    nu = int(np.ceil((hi[0] - lo[0]) / res))
    nv = int(np.ceil((hi[1] - lo[1]) / res))
    m = np.zeros((nv, nu), np.uint8)
    for tr in uv_tris:
        a, b, c = tr
        bl = np.floor((np.minimum(np.minimum(a, b), c) - lo) / res).astype(int)
        bh = np.ceil((np.maximum(np.maximum(a, b), c) - lo) / res).astype(int)
        iu = np.arange(max(bl[0], 0), min(bh[0] + 1, nu))
        iv = np.arange(max(bl[1], 0), min(bh[1] + 1, nv))
        if not len(iu) or not len(iv):
            continue
        U, V = np.meshgrid(lo[0] + (iu + 0.5) * res, lo[1] + (iv + 0.5) * res)
        v0, v1 = b - a, c - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-9:
            continue
        pu, pv = U - a[0], V - a[1]
        s = (pu * v1[1] - v1[0] * pv) / den
        t = (v0[0] * pv - pu * v0[1]) / den
        # a half-cell margin, so a sliver still covers the cells it crosses
        eps = 0.5 * res / max(np.sqrt(abs(den)), 1e-9)
        ins = (s >= -eps) & (t >= -eps) & (s + t <= 1.0 + eps)
        m[np.ix_(iv, iu)] |= ins.astype(np.uint8)
    # one cell of dilation: closes the cracks the compiler's T-junctions leave between pieces
    dm = m.copy()
    dm[1:] |= m[:-1]
    dm[:-1] |= m[1:]
    dm[:, 1:] |= m[:, :-1]
    dm[:, :-1] |= m[:, 1:]
    return float(lo[0]), float(lo[1]), nu, nv, dm


if HAVE_TRITON:
    @triton.jit
    def _tm_plane_kernel(dx_ptr, dy_ptr, dz_ptr, eye_ptr, sid_ptr, pstart_ptr, pcount_ptr,
                         pf_ptr, pi_ptr, mask_ptr, sph_ptr, t_ptr, R, BLOCK: tl.constexpr):
        """one program = one env x BLOCK rays: the nearest distance at which each ray crosses one
        of its target's planes INSIDE that plane's mask (inf = never). pf = 16 floats per plane
        (n, d, origin, e_u, e_v, u0, v0, cell), pi = (mask offset, nu, nv)"""
        e = tl.program_id(0)
        rb = tl.program_id(1)
        offs = rb * BLOCK + tl.arange(0, BLOCK)
        m = offs < R
        base = e * R + offs
        dx = tl.load(dx_ptr + base, mask=m, other=0.0)
        dy = tl.load(dy_ptr + base, mask=m, other=0.0)
        dz = tl.load(dz_ptr + base, mask=m, other=0.0)
        ox = tl.load(eye_ptr + e * 3 + 0)
        oy = tl.load(eye_ptr + e * 3 + 1)
        oz = tl.load(eye_ptr + e * 3 + 2)
        s = tl.load(sid_ptr + e)
        best = tl.full([BLOCK], float("inf"), tl.float32)
        p0 = tl.where(s >= 0, tl.load(pstart_ptr + tl.maximum(s, 0)), 0)
        pc = tl.where(s >= 0, tl.load(pcount_ptr + tl.maximum(s, 0)), 0)
        for j in range(0, pc):
            # the plane's bounding sphere: a block none of whose rays passes it skips the plane
            sx = tl.load(sph_ptr + (p0 + j) * 4 + 0) - ox
            sy = tl.load(sph_ptr + (p0 + j) * 4 + 1) - oy
            sz = tl.load(sph_ptr + (p0 + j) * 4 + 2) - oz
            sr = tl.load(sph_ptr + (p0 + j) * 4 + 3)
            tcl = sx * dx + sy * dy + sz * dz
            cc = sx * sx + sy * sy + sz * sz
            near = ((cc - tcl * tcl) <= sr * sr) & ((tcl > 0.0) | (cc <= sr * sr)) & m
            if tl.sum(near.to(tl.int32), axis=0) > 0:
                best = _tm_plane_one(dx, dy, dz, ox, oy, oz, pf_ptr, pi_ptr, mask_ptr, p0 + j,
                                     best, m)
        tl.store(t_ptr + base, best, mask=m)

    @triton.jit
    def _tm_target_best(dx, dy, dz, ox, oy, oz, s, pstart_ptr, pcount_ptr, pf_ptr, pi_ptr,
                        mask_ptr, sph_ptr, m, BLOCK: tl.constexpr):
        """the nearest masked-plane hit of each ray on target s (inf = none)"""
        best = tl.full([BLOCK], float("inf"), tl.float32)
        p0 = tl.where(s >= 0, tl.load(pstart_ptr + tl.maximum(s, 0)), 0)
        pc = tl.where(s >= 0, tl.load(pcount_ptr + tl.maximum(s, 0)), 0)
        for j in range(0, pc):
            sx = tl.load(sph_ptr + (p0 + j) * 4 + 0) - ox
            sy = tl.load(sph_ptr + (p0 + j) * 4 + 1) - oy
            sz = tl.load(sph_ptr + (p0 + j) * 4 + 2) - oz
            sr = tl.load(sph_ptr + (p0 + j) * 4 + 3)
            tcl = sx * dx + sy * dy + sz * dz
            cc = sx * sx + sy * sy + sz * sz
            near = ((cc - tcl * tcl) <= sr * sr) & ((tcl > 0.0) | (cc <= sr * sr)) & m
            if tl.sum(near.to(tl.int32), axis=0) > 0:
                best = _tm_plane_one(dx, dy, dz, ox, oy, oz, pf_ptr, pi_ptr, mask_ptr, p0 + j,
                                     best, m)
        return best

    @triton.jit
    def _tm_fused_kernel(yaw_ptr, pitch_ptr, eye_ptr, yoff_ptr, poff_ptr, sid_ptr, val_ptr,
                         pstart_ptr, pcount_ptr, pf_ptr, pi_ptr, mask_ptr, sph_ptr, out_ptr,
                         H, W, BLOCK: tl.constexpr, MAXV: tl.constexpr,
                         PINHOLE: tl.constexpr = False):
        """one program = one env x BLOCK pixels: the equiangular camera's ray (the lidar's own
        _dirs_equiangular, term for term), the env's three target SLOTS (ids and values), and the
        channel: the value of the nearest slot whose value is not 0 (MAXV: the LARGEST value
        among the slots the ray crosses), else 0"""
        e = tl.program_id(0)
        rb = tl.program_id(1)
        R = H * W
        offs = rb * BLOCK + tl.arange(0, BLOCK)
        m = offs < R
        row = offs // W
        col = offs % W
        d2r = 0.017453292519943295
        if PINHOLE:
            # --pinhole: the lidar's rectilinear ray (_dirs_pinhole, term for term) - yoff /
            # poff carry the TANGENT-PLANE offsets uoff / voff here
            yw = tl.load(yaw_ptr + e) * d2r
            pt = tl.load(pitch_ptr + e) * d2r
            cy = tl.cos(yw)
            sy = tl.sin(yw)
            cpt = tl.cos(pt)
            spt = tl.sin(pt)
            uo = tl.load(yoff_ptr + col, mask=m, other=0.0)
            vo = tl.load(poff_ptr + row, mask=m, other=0.0)
            dx = cpt * cy + uo * sy - vo * spt * cy
            dy = cpt * sy - uo * cy - vo * spt * sy
            dz = spt + vo * cpt
            inv = 1.0 / tl.sqrt(dx * dx + dy * dy + dz * dz)
            dx = dx * inv
            dy = dy * inv
            dz = dz * inv
        else:
            pa = tl.load(pitch_ptr + e) * d2r + tl.load(poff_ptr + row, mask=m, other=0.0)
            ya = tl.load(yaw_ptr + e) * d2r + tl.load(yoff_ptr + col, mask=m, other=0.0)
            cp = tl.cos(pa)
            dx = cp * tl.cos(ya)
            dy = cp * tl.sin(ya)
            dz = tl.sin(pa)
        ox = tl.load(eye_ptr + e * 3 + 0)
        oy = tl.load(eye_ptr + e * 3 + 1)
        oz = tl.load(eye_ptr + e * 3 + 2)
        inf = float("inf")
        best = tl.full([BLOCK], inf, tl.float32)
        val = tl.zeros([BLOCK], tl.float32)
        for k in range(0, 3):
            s = tl.load(sid_ptr + e * 3 + k)
            v = tl.load(val_ptr + e * 3 + k)
            s = tl.where(v != 0.0, s, -1)
            b = _tm_target_best(dx, dy, dz, ox, oy, oz, s, pstart_ptr, pcount_ptr,
                                pf_ptr, pi_ptr, mask_ptr, sph_ptr, m, BLOCK)
            if MAXV:
                val = tl.where((b < inf) & (v > val), v, val)
            else:
                closer = b < best
                best = tl.where(closer, b, best)
                val = tl.where(closer, v, val)
        tl.store(out_ptr + e * R + offs, val, mask=m)

    @triton.jit
    def _tm_plane_one(dx, dy, dz, ox, oy, oz, pf_ptr, pi_ptr, mask_ptr, pj, best, m):
        """one plane: the ray-plane distance, the mask lookup, the running nearest hit"""
        if True:
            q = pj * 16
            nx = tl.load(pf_ptr + q + 0)
            ny = tl.load(pf_ptr + q + 1)
            nz = tl.load(pf_ptr + q + 2)
            d = tl.load(pf_ptr + q + 3)
            px0 = tl.load(pf_ptr + q + 4)
            py0 = tl.load(pf_ptr + q + 5)
            pz0 = tl.load(pf_ptr + q + 6)
            ux = tl.load(pf_ptr + q + 7)
            uy = tl.load(pf_ptr + q + 8)
            uz = tl.load(pf_ptr + q + 9)
            vx = tl.load(pf_ptr + q + 10)
            vy = tl.load(pf_ptr + q + 11)
            vz = tl.load(pf_ptr + q + 12)
            u0 = tl.load(pf_ptr + q + 13)
            v0 = tl.load(pf_ptr + q + 14)
            res = tl.load(pf_ptr + q + 15)
            off = tl.load(pi_ptr + pj * 3 + 0)
            nu = tl.load(pi_ptr + pj * 3 + 1)
            nv = tl.load(pi_ptr + pj * 3 + 2)
            nd = dx * nx + dy * ny + dz * nz
            ok = tl.abs(nd) > 1e-9
            tt = (d - (ox * nx + oy * ny + oz * nz)) / tl.where(ok, nd, 1.0)
            rx = ox + dx * tt - px0
            ry = oy + dy * tt - py0
            rz = oz + dz * tt - pz0
            uu = rx * ux + ry * uy + rz * uz
            vv = rx * vx + ry * vy + rz * vz
            iu = tl.floor((uu - u0) / res).to(tl.int64)
            iv = tl.floor((vv - v0) / res).to(tl.int64)
            inb = ok & (tt > 0.0) & (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv) & m
            mv = tl.load(mask_ptr + off + iv * nu + iu, mask=inb, other=0)
            hit = inb & (mv > 0)
            best = tl.where(hit & (tt < best), tt, best)
        return best


def takeoff_fade_values(leave_ticks, t, fade_ticks):
    """The CONTINUOUS target schedule (the user, 2026-09-28: no jump when a ramp is reached): the
    route's targets 0, 1, 2, ... (the last may be FIN), ``leave_ticks[i]`` the tick the agent LEFT
    target i (its takeoff; +inf while not yet left). Target i's value at tick t is

        0.5 * ramp(t - L[i-2]) + 0.5 * ramp(t - L[i-1]) - 1.0 * ramp(t - L[i]),

    ramp(x) = clip(x / fade_ticks, 0, 1), L[-1] = L[-2] = -inf: it fades in as the one AFTER
    (0 -> 0.5) when the ramp two before it is left, brightens to NEXT (0.5 -> 1) when the ramp
    before it is left, and fades out (1 -> 0) when it is left itself. A TOUCH changes nothing;
    every value is a sum of continuous ramps, so no timing of the takeoffs makes it jump.
    -> (n,) values."""
    L = np.asarray(leave_ticks, np.float64)
    n = len(L)

    def ramp(x):
        return np.clip(np.asarray(x, np.float64) / max(float(fade_ticks), 1e-9), 0.0, 1.0)
    prev1 = np.concatenate(([-np.inf], L[:-1])) if n else L
    prev2 = np.concatenate(([-np.inf, -np.inf], L[:-2]))[:n] if n else L
    # +-inf need no special case: ramp(t + inf) = 1 (the virtual predecessors), ramp(t - inf) = 0
    # (a ramp never left)
    return 0.5 * ramp(t - prev2) + 0.5 * ramp(t - prev1) - ramp(t - L)


class TargetMask:
    def __init__(self, mesh_npz, finish_box=None, device="cuda", unit: str = "face",
                 res: float = RES):
        if unit not in ("face", "object"):
            raise ValueError(f"TargetMask: unit face|object, got {unit!r}")
        z = np.load(mesh_npz)
        tris = z["tris"].astype(np.float64)
        ts = z["tri_surf"].astype(np.int64)
        cat = z["cat"].astype(np.int64)
        self.unit = unit
        self.device = torch.device(device)
        keep = ts >= 0
        tris, ts = tris[keep], ts[keep]
        n_surf = int(len(cat))
        if unit == "object":
            # a target is a whole OBJECT: target faces (floors, ramps) touching at an edge (their
            # edge samples share a 16 u cell), whatever their angle
            tgt = np.isin(ts, np.flatnonzero((cat == 0) | (cat == 1)))
            obj = self._objects(tris[tgt], ts[tgt], n_surf)
            self.obj_of_surf = obj
            ts = np.where(tgt, obj[ts], -1)
            keep = ts >= 0
            tris, ts = tris[keep], ts[keep]
            self.n_targets = int(obj.max()) + 1 if (obj >= 0).any() else 0
        else:
            self.obj_of_surf = np.arange(n_surf)
            self.n_targets = n_surf
        pid, nrm, area = _plane_groups(tris, ts)
        planes = []                      # (target, pf[16], mask, bounding sphere)
        for p in np.unique(pid):
            sel = pid == p
            T = tris[sel]
            w = area[sel]
            n = (nrm[sel] * w[:, None]).sum(0)
            n = n / max(np.linalg.norm(n), 1e-12)
            eu = np.cross(n, [0.0, 0.0, 1.0])
            if np.linalg.norm(eu) < 1e-3:
                eu = np.array([1.0, 0.0, 0.0])
            eu = eu / np.linalg.norm(eu)
            ev = np.cross(n, eu)
            o = (T.mean(axis=1) * w[:, None]).sum(0) / max(w.sum(), 1e-12)
            rel = T - o
            uv = np.stack([rel @ eu, rel @ ev], axis=-1)            # (k, 3, 2)
            ext = uv.reshape(-1, 2).max(0) - uv.reshape(-1, 2).min(0)
            r = max(float(res), float(np.sqrt((ext[0] + 4 * res) * (ext[1] + 4 * res)
                                              / MAX_CELLS)))
            u0, v0, nu, nv, mk = _rasterize(uv, r)
            pf = np.array([n[0], n[1], n[2], float(n @ o), o[0], o[1], o[2], eu[0], eu[1], eu[2],
                           ev[0], ev[1], ev[2], u0, v0, r], np.float32)
            vv = T.reshape(-1, 3)
            sc = vv.mean(0)
            sph = np.array([sc[0], sc[1], sc[2],
                            float(np.linalg.norm(vv - sc, axis=1).max()) + 2.0 * r], np.float32)
            planes.append((int(ts[sel][0]), pf, mk, sph))
        planes.sort(key=lambda x: x[0])
        start = np.zeros(self.n_targets, np.int64)
        count = np.zeros(self.n_targets, np.int64)
        pi, masks, off = [], [], 0
        for i, (s, _pf, mk, _sph) in enumerate(planes):
            if count[s] == 0:
                start[s] = i
            count[s] += 1
            pi.append((off, mk.shape[1], mk.shape[0]))
            masks.append(mk.reshape(-1))
            off += mk.size
        self.n_planes = len(planes)
        self.planes_per_target = count
        self.pf = torch.as_tensor(np.stack([p[1] for p in planes]) if planes
                                  else np.zeros((0, 16), np.float32), device=self.device)
        self.pi = torch.as_tensor(np.asarray(pi, np.int64).reshape(-1, 3), device=self.device)
        self.mask = torch.as_tensor(np.concatenate(masks) if masks else np.zeros(1, np.uint8),
                                    device=self.device)
        self.sph = torch.as_tensor(np.stack([p[3] for p in planes]) if planes
                                   else np.zeros((0, 4), np.float32), device=self.device)
        self.pstart = torch.as_tensor(start, device=self.device)
        self.pcount = torch.as_tensor(count, device=self.device)
        self.mask_bytes = int(off)
        if finish_box is not None and isinstance(finish_box, dict):
            finish_box = (finish_box["mins"], finish_box["maxs"])
        self.fin = (None if finish_box is None or finish_box[0] is None else
                    (torch.as_tensor(np.asarray(finish_box[0], np.float32), device=self.device),
                     torch.as_tensor(np.asarray(finish_box[1], np.float32), device=self.device)))
        self.t1 = None
        self.t2 = None
        self.sids = None
        self.svals = None
        self.combine = "nearest"

    @staticmethod
    def _objects(tris, ts, n_surf):
        """surface -> object id: target surfaces whose edges share a 16 u cell are one object
        (surfgym.rampvocab.edge_pieces - the ramp windows' pieces are the same rule)"""
        from .rampvocab import edge_pieces
        return edge_pieces(tris, ts, n_surf)

    def set_targets(self, n_envs, t1, t2):
        """per env the next target (t1, +1) and the one after (t2, -1): surface ids (object ids
        with unit="object" - see obj_of_surf), FIN for the finish box, NONE for no target"""
        t1 = np.asarray(t1, np.int64).reshape(n_envs)
        t2 = np.asarray(t2, np.int64).reshape(n_envs)
        self.set_slots(n_envs, np.stack([t1, t2, np.full(n_envs, NONE)], 1),
                       np.tile(np.array([1.0, -1.0, 0.0], np.float32), (n_envs, 1)),
                       combine="nearest")

    def set_slots(self, n_envs, ids, vals, combine: str = "max"):
        """per env up to three target SLOTS: ids (n, K) (surface / object ids, FIN, NONE) and
        their channel values (n, K) - e.g. a cross-fade: the ramp just left fading 1 -> 0, the
        next 0.5 -> 1, the new one after 0 -> 0.5. A pixel shows the value of the nearest slot
        with a non-zero value (combine="nearest"), or the LARGEST value among the slots the ray
        crosses (combine="max", the default here: with brightness as the order, a pixel is then
        the max of continuously changing values and cannot jump)."""
        if combine not in ("max", "nearest"):
            raise ValueError(f"TargetMask: combine max|nearest, got {combine!r}")
        self.combine = combine
        ids = np.asarray(ids, np.int64).reshape(n_envs, -1)
        vals = np.asarray(vals, np.float32).reshape(n_envs, -1)
        k = ids.shape[1]
        if k > 3:
            raise ValueError("TargetMask: at most three target slots")
        pad_i = np.full((n_envs, 3), NONE, np.int64)
        pad_v = np.zeros((n_envs, 3), np.float32)
        pad_i[:, :k] = ids
        pad_v[:, :k] = vals
        self.sids = torch.as_tensor(pad_i, device=self.device).contiguous()
        self.svals = torch.as_tensor(pad_v, device=self.device).contiguous()
        self.t1 = self.sids[:, 0]
        self.t2 = self.sids[:, 1]

    def _plane_hit_torch(self, sid, eye, dirs):
        """the torch reference of _tm_plane_kernel: (N,), (N,3), (N,R,3) -> (N,R) distances"""
        N, R, _ = dirs.shape
        best = torch.full((N, R), float("inf"), device=self.device)
        cnt = torch.where(sid >= 0, self.pcount[sid.clamp(min=0)], torch.zeros_like(sid))
        st = torch.where(sid >= 0, self.pstart[sid.clamp(min=0)], torch.zeros_like(sid))
        for j in range(int(cnt.max().item()) if N else 0):
            act = cnt > j
            pidx = (st + j).clamp(max=max(self.n_planes - 1, 0))
            pf = self.pf[pidx]                                    # (N, 16)
            pi = self.pi[pidx]
            n = pf[:, 0:3]
            nd = (dirs * n[:, None]).sum(-1)
            ok = nd.abs() > 1e-9
            tt = ((pf[:, 3] - (eye * n).sum(-1))[:, None]
                  / torch.where(ok, nd, torch.ones_like(nd)))
            rel = eye[:, None] + dirs * tt[..., None] - pf[:, None, 4:7]
            uu = (rel * pf[:, None, 7:10]).sum(-1)
            vv = (rel * pf[:, None, 10:13]).sum(-1)
            iu = torch.floor((uu - pf[:, 13:14]) / pf[:, 15:16]).long()
            iv = torch.floor((vv - pf[:, 14:15]) / pf[:, 15:16]).long()
            nu, nv = pi[:, 1:2], pi[:, 2:3]
            inb = ok & (tt > 0) & (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv) & act[:, None]
            flat = (pi[:, 0:1] + iv.clamp(min=0) * nu + iu.clamp(min=0)).clamp(
                0, max(self.mask.numel() - 1, 0))
            hit = inb & (self.mask[flat] > 0)
            best = torch.where(hit & (tt < best), tt, best)
        return best

    def _box_hit(self, eye, dirs):
        """entry distance of each ray into the finish box (inf = misses it)"""
        lo, hi = self.fin
        tmin = torch.zeros(dirs.shape[:2], device=self.device)
        tmax = torch.full(dirs.shape[:2], float("inf"), device=self.device)
        for a in range(3):
            d = dirs[..., a]
            inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
            t0 = (lo[a] - eye[:, a:a + 1]) * inv
            t1 = (hi[a] - eye[:, a:a + 1]) * inv
            tmin = torch.maximum(tmin, torch.minimum(t0, t1))
            tmax = torch.minimum(tmax, torch.maximum(t0, t1))
        return torch.where(tmax >= tmin, tmin, torch.full_like(tmin, float("inf")))

    def _dist(self, sid, eye, dx, dy, dz, force_torch=False):
        N, H, W = dx.shape
        R = H * W
        if HAVE_TRITON and self.device.type == "cuda" and not force_torch:
            t = torch.empty(N, R, device=self.device, dtype=torch.float32)
            _tm_plane_kernel[(N, triton.cdiv(R, TM_BLOCK))](
                dx.contiguous(), dy.contiguous(), dz.contiguous(), eye, sid.contiguous(),
                self.pstart, self.pcount, self.pf, self.pi, self.mask, self.sph, t, R,
                BLOCK=TM_BLOCK)
        else:
            dirs = torch.stack([dx, dy, dz], dim=-1).reshape(N, R, 3)
            t = self._plane_hit_torch(sid, eye, dirs)
        if self.fin is not None and bool((sid == FIN).any()):
            dirs = torch.stack([dx, dy, dz], dim=-1).reshape(N, R, 3)
            t = torch.where((sid == FIN).view(N, 1), self._box_hit(eye, dirs), t)
        return t

    @torch.no_grad()
    def render(self, lidar, origin, yaw_deg, pitch_deg, ducked, depth=None, force_torch=False):
        """(N, H, W): +1 where the view sees the next target, -1 the one after, 0 elsewhere - no
        occlusion; the nearer target wins where the two overlap. `depth` is not needed."""
        N = origin.shape[0]
        eye = torch.stack([origin[:, 0], origin[:, 1],
                           origin[:, 2] + torch.where(ducked.bool(), 12.0, 17.0)],
                          dim=1).float().contiguous()
        if self.sids is None:
            self.set_slots(N, np.full((N, 1), NONE), np.zeros((N, 1)))
        sids, svals = self.sids, self.svals
        if (HAVE_TRITON and self.device.type == "cuda" and not force_torch
                and not bool((sids == FIN).any())):
            out = torch.empty(N, lidar.H, lidar.W, device=self.device, dtype=torch.float32)
            R = lidar.H * lidar.W
            ph = bool(getattr(lidar, "pinhole", False))
            _tm_fused_kernel[(N, triton.cdiv(R, TM_BLOCK))](
                yaw_deg.float().contiguous(), pitch_deg.float().contiguous(), eye,
                lidar.uoff if ph else lidar.yoff, lidar.voff if ph else lidar.poff,
                sids, svals, self.pstart, self.pcount, self.pf, self.pi,
                self.mask, self.sph, out, lidar.H, lidar.W, BLOCK=TM_BLOCK,
                MAXV=self.combine == "max", PINHOLE=ph)
            return out
        lidar._ensure_buffers(N)
        dirs = lidar._dirs_pinhole if lidar.pinhole else lidar._dirs_equiangular
        dirs(N, yaw_deg, pitch_deg, np.pi / 180.0)
        dx, dy, dz = lidar._dx, lidar._dy, lidar._dz
        R = lidar.H * lidar.W
        best = torch.full((N, R), float("inf"), device=self.device)
        out = torch.zeros(N, R, device=self.device)
        for k in range(3):
            s = torch.where(svals[:, k] != 0, sids[:, k], torch.full_like(sids[:, k], NONE))
            b = self._dist(s, eye, dx, dy, dz, force_torch)
            v = svals[:, k:k + 1].expand(-1, R)
            if self.combine == "max":
                out = torch.where(torch.isfinite(b) & (v > out), v, out)
            else:
                closer = b < best
                best = torch.where(closer, b, best)
                out = torch.where(closer, v, out)
        return out.reshape(N, lidar.H, lidar.W)
