"""targetmask.py - the TARGET-RAMP channel: which of the lidar's pixels see the next ramp (+1) and
the one after it (-1); everything else 0 (the user, 2026-09-28: "the geometry comes from the depth
render; the destination ramps are a separate channel").

No new render and no voxel bake: the depth march already cast every ray, so each ray is tested
against ONLY the two target surfaces' triangles (tools/ramps_mesh.py - a flat ramp is a handful of
triangles) with Moller-Trumbore, and it shows a target when the triangle hit lies at or before the
ray's own depth hit (+ a tolerance: the march stops inside the solid, up to a cell past the
surface). Occlusion is therefore exactly the depth image's (a ray clear to the lidar's range
shows a target at ANY distance - direction without occlusion past the range), walls and every
other surface read 0,
and the finish - a trigger box, not a surface - is a target like a ramp through the same slab
test as --obs-potential-curtain. One batched torch pass per render; no per-env Python.

    tm = TargetMask(mesh_npz, finish_box, device, max_tris=128)
    tm.set_targets(env_ids, t1_surface, t2_surface)       # -1 = none, FIN = the finish
    chan = tm.render(lidar, origin, yaw, pitch, ducked, depth)   # (N, H, W) in {-1, 0, +1}
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

if HAVE_TRITON:
    @triton.jit
    def _tm_kernel(dx_ptr, dy_ptr, dz_ptr, eye_ptr, sid_ptr, cnt_ptr,
                   pn_ptr, pv0_ptr, pa1_ptr, pa2_ptr, pd_ptr, out_ptr, R, MT,
                   BLOCK: tl.constexpr):
        """one program = one env x BLOCK rays: the nearest hit distance of each ray on that env's
        target surface (inf = none) - the torch path's ray-plane distance and dual-vector
        barycentrics, term for term, looping over the surface's own triangles"""
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
        c = tl.where(s >= 0, tl.load(cnt_ptr + tl.maximum(s, 0)), 0)
        sb = tl.maximum(s, 0) * MT
        for k in range(0, c):
            q = (sb + k) * 3
            nx = tl.load(pn_ptr + q + 0)
            ny = tl.load(pn_ptr + q + 1)
            nz = tl.load(pn_ptr + q + 2)
            vx = tl.load(pv0_ptr + q + 0)
            vy = tl.load(pv0_ptr + q + 1)
            vz = tl.load(pv0_ptr + q + 2)
            ax = tl.load(pa1_ptr + q + 0)
            ay = tl.load(pa1_ptr + q + 1)
            az = tl.load(pa1_ptr + q + 2)
            bx = tl.load(pa2_ptr + q + 0)
            by = tl.load(pa2_ptr + q + 1)
            bz = tl.load(pa2_ptr + q + 2)
            d = tl.load(pd_ptr + sb + k)
            nd = dx * nx + dy * ny + dz * nz
            ok = tl.abs(nd) > 1e-9
            tt = (d - (ox * nx + oy * ny + oz * nz)) / tl.where(ok, nd, 1.0)
            ovx = ox - vx
            ovy = oy - vy
            ovz = oz - vz
            u = (ovx * ax + ovy * ay + ovz * az) + tt * (dx * ax + dy * ay + dz * az)
            v = (ovx * bx + ovy * by + ovz * bz) + tt * (dx * bx + dy * by + dz * bz)
            hit = ok & (tt > 0.0) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
            best = tl.where(hit & (tt < best), tt, best)
        tl.store(out_ptr + base, best, mask=m)


class TargetMask:
    def __init__(self, mesh_npz, finish_box=None, device="cuda", max_tris: int = 128,
                 tol: float = 48.0):
        z = np.load(mesh_npz)
        tris = z["tris"].astype(np.float32)
        ts = z["tri_surf"].astype(np.int64)
        self.n_surf = int(len(z["cat"]))
        self.device = torch.device(device)
        self.max_tris = int(max_tris)
        self.tol = float(tol)
        # per surface: its (up to max_tris LARGEST) triangles, padded: (S, max_tris, 3, 3)
        area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]),
                                    axis=1)
        pad = np.zeros((self.n_surf, self.max_tris, 3, 3), np.float32)
        cnt = np.zeros(self.n_surf, np.int64)
        for s in range(self.n_surf):
            idx = np.flatnonzero(ts == s)
            if len(idx) > self.max_tris:
                idx = idx[np.argsort(-area[idx])[:self.max_tris]]
            pad[s, :len(idx)] = tris[idx]
            cnt[s] = len(idx)
        self.tris = torch.as_tensor(pad, device=self.device)
        self.cnt = torch.as_tensor(cnt, device=self.device)
        # per triangle: the (unnormalised) plane normal n and offset d = n.v0, and the DUAL vectors
        # a1, a2 with u = (p - v0).a1, v = (p - v0).a2 for a point p of the plane - so a ray's hit
        # is three batched matmuls (dir.n, dir.a1, dir.a2) and elementwise work, no cross products
        v0 = pad[:, :, 0].astype(np.float64)
        e1 = pad[:, :, 1] - v0
        e2 = pad[:, :, 2] - v0
        nn = np.cross(e1, e2)
        c1 = np.cross(e2, nn)
        c2 = np.cross(nn, e1)
        den1 = np.einsum("smk,smk->sm", e1, c1)
        den2 = np.einsum("smk,smk->sm", e2, c2)
        good = (np.abs(den1) > 1e-9) & (np.abs(den2) > 1e-9)
        a1 = np.where(good[..., None], c1 / np.where(good, den1, 1.0)[..., None], 0.0)
        a2 = np.where(good[..., None], c2 / np.where(good, den2, 1.0)[..., None], 0.0)
        f32 = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)  # noqa: E731
        self.pn, self.pv0, self.pa1, self.pa2 = f32(nn), f32(v0), f32(a1), f32(a2)
        self.pd = f32(np.einsum("smk,smk->sm", nn, v0))
        self.fin = (None if finish_box is None else
                    (torch.as_tensor(np.asarray(finish_box[0], np.float32), device=self.device),
                     torch.as_tensor(np.asarray(finish_box[1], np.float32), device=self.device)))
        self.t1 = None
        self.t2 = None

    def set_targets(self, n_envs, t1, t2):
        """per env the next target (t1, drawn +1) and the one after (t2, drawn -1): surface ids,
        FIN for the finish box, NONE for no target"""
        self.t1 = torch.as_tensor(np.asarray(t1, np.int64).reshape(n_envs), device=self.device)
        self.t2 = torch.as_tensor(np.asarray(t2, np.int64).reshape(n_envs), device=self.device)

    def _tri_hit(self, sid, ex, ey, ez, dx, dy, dz, chunk: int = 64):
        if HAVE_TRITON and self.device.type == "cuda":
            N, H, W = dx.shape
            R = H * W
            out = torch.empty(N, R, device=self.device, dtype=torch.float32)
            eye = torch.cat([ex, ey, ez], dim=-1).reshape(N, 3).contiguous()
            _tm_kernel[(N, triton.cdiv(R, TM_BLOCK))](
                dx.contiguous(), dy.contiguous(), dz.contiguous(), eye, sid.contiguous(), self.cnt,
                self.pn, self.pv0, self.pa1, self.pa2, self.pd, out, R, self.max_tris,
                BLOCK=TM_BLOCK)
            return out.reshape(N, H, W)
        return self._tri_hit_torch(sid, ex, ey, ez, dx, dy, dz, chunk)

    def _tri_hit_torch(self, sid, ex, ey, ez, dx, dy, dz, chunk: int = 64):
        """(N,) surface ids, eye (N,1,1), dirs (N,H,W) -> the nearest hit distance of each ray on
        that surface's triangles (inf = none). Per env the ray-plane distance and the two
        barycentric coordinates are batched matmuls of the ray directions against the triangles'
        precomputed normals and dual vectors; `chunk` envs at a time, padded to the chunk's own
        largest triangle count."""
        N, H, W = dx.shape
        R = H * W
        out = torch.full((N, R), float("inf"), device=self.device)
        cnt_all = torch.where(sid >= 0, self.cnt[sid.clamp(min=0)], torch.zeros_like(sid))
        dirs = torch.stack([dx, dy, dz], dim=-1).reshape(N, R, 3)
        eye = torch.cat([ex, ey, ez], dim=-1).reshape(N, 3)
        for c0 in range(0, N, chunk):
            sl = slice(c0, min(N, c0 + chunk))
            M = int(cnt_all[sl].max().item()) if cnt_all[sl].numel() else 0
            if M == 0:
                continue
            s_ = sid[sl].clamp(min=0)
            n_, v0 = self.pn[s_, :M], self.pv0[s_, :M]            # (n, M, 3)
            a1, a2, d_ = self.pa1[s_, :M], self.pa2[s_, :M], self.pd[s_, :M]
            valid = torch.arange(M, device=self.device)[None, :] < cnt_all[sl][:, None]
            dr, o = dirs[sl], eye[sl]
            nd = torch.bmm(dr, n_.transpose(1, 2))                # (n, R, M)
            no = (n_ * o[:, None]).sum(-1)                        # (n, M)
            ok_nd = nd.abs() > 1e-9
            tt = (d_ - no)[:, None] / torch.where(ok_nd, nd, torch.ones_like(nd))
            ov = o[:, None] - v0                                   # (n, M, 3)
            u = (ov * a1).sum(-1)[:, None] + tt * torch.bmm(dr, a1.transpose(1, 2))
            v = (ov * a2).sum(-1)[:, None] + tt * torch.bmm(dr, a2.transpose(1, 2))
            hit = ok_nd & (tt > 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & valid[:, None]
            out[sl] = torch.where(hit, tt, torch.full_like(tt, float("inf"))).min(dim=-1).values
        return out.reshape(N, H, W)

    def _box_hit(self, ex, ey, ez, dx, dy, dz):
        """entry distance of each ray into the finish box (inf = misses it)"""
        lo, hi = self.fin
        o = [ex, ey, ez]
        dd = [dx, dy, dz]
        tmin = torch.zeros_like(dx)
        tmax = torch.full_like(dx, float("inf"))
        for a in range(3):
            inv = 1.0 / torch.where(dd[a].abs() < 1e-9, torch.full_like(dd[a], 1e-9), dd[a])
            t0 = (lo[a] - o[a]) * inv
            t1 = (hi[a] - o[a]) * inv
            tmin = torch.maximum(tmin, torch.minimum(t0, t1))
            tmax = torch.minimum(tmax, torch.maximum(t0, t1))
        return torch.where(tmax >= tmin, tmin, torch.full_like(tmin, float("inf")))

    @torch.no_grad()
    def render(self, lidar, origin, yaw_deg, pitch_deg, ducked, depth):
        """the channel for a render whose depth channel is ``depth`` (N, H, W): the lidar's own
        rays (its _dirs_* and eye height), the decoded hit distance, then the two targets"""
        N = origin.shape[0]
        lidar._ensure_buffers(N)
        ex = origin[:, 0].view(N, 1, 1).float()
        ey = origin[:, 1].view(N, 1, 1).float()
        ez = (origin[:, 2] + torch.where(ducked.bool(), 12.0, 17.0)).view(N, 1, 1).float()
        dirs = lidar._dirs_pinhole if lidar.pinhole else lidar._dirs_equiangular
        dirs(N, yaw_deg, pitch_deg, np.pi / 180.0)
        dx, dy, dz = lidar._dx, lidar._dy, lidar._dz
        t_depth = lidar.decode_depth(depth.reshape(N, lidar.H, lidar.W))
        out = torch.zeros(N, lidar.H, lidar.W, device=self.device)
        o3 = (ex, ey, ez)
        for sid, val in ((self.t2, -1.0), (self.t1, 1.0)):     # t1 drawn last: it wins a tie
            if sid is None:
                continue
            t_hit = self._tri_hit(sid, *o3, dx, dy, dz)
            if self.fin is not None:
                tf = self._box_hit(*o3, dx, dy, dz)
                t_hit = torch.where((sid == FIN).view(N, 1, 1), tf, t_hit)
            # visible: the target's hit is at or before the ray's own depth hit - or the ray saw
            # NOTHING within the lidar's range (clear to range): then a farther target still shows
            # its direction (occlusion past the range is unknown to the depth render)
            clear = t_depth >= lidar.range - 1.0
            # the march stops on the voxel grid, up to ~a cell (along a grazing ray, more) away
            # from the exact triangle: the tolerance scales with the lidar's own cell
            tol = max(self.tol, 3.0 * float(getattr(lidar, "cell", 16.0)))
            vis = torch.isfinite(t_hit) & ((t_hit <= t_depth + tol) | clear)
            out = torch.where(vis, torch.full_like(out, val), out)
        return out
