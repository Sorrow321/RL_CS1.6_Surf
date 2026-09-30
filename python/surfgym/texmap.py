"""--obs-texture (the user, 2026-09-30): the map's own textures as three more image channels.

Depth alone cannot tell a ramp from the wall beside it at the same distance, and far away it is a
fog (ledger 2026-09-28 04:24). The game separates them with textures. This module gives the depth
march (surfgym/vision.py) what it needs to colour its hit:

* the map's faces, texinfo and embedded textures, parsed from the .bsp (GoldSrc v30), every mip
  converted through the texture's own palette into one RGB atlas, and each texture's AVERAGE
  colour - the far level: at 64 x 32 a pixel spans ~33 u at 1,000 u, so past mip 3 the texture
  is its average or it aliases into frame-to-frame noise;
* a FACE-ID GRID on the depth field's own voxels: every solid voxel within two voxels of air
  (the march stops inside the first solid voxel it samples) holds its nearest visible triangle's
  face, within three cells, else -1 (baked once per map on the GPU, cached beside the .bsp as the
  sparse pairs, keyed on the depth field's signature);
* per face: its texinfo (s and t axes + offsets), its plane (the hit is projected onto it, so the
  texture coordinates are the surface's, not the march's point up to ~29 u inside the solid), its
  texels per unit, its texture.

Invisible brushes (triggers, clip, null, hint, skip, origin; trigger-class entities and fully
transparent ones) have no visible face; sky faces are a flat sky colour; a miss or a solid voxel
with no visible face in reach is black. Unlit (the texture's albedo, no lightmaps).
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

TEXMAP_SEMANTICS = "t1"             # bump when the bake or the face set changes
SKIP_TEX = {"aaatrigger", "clip", "null", "origin", "hint", "skip", "bevel", "trigger"}
SKIP_CLASS = ("trigger_", "func_buyzone", "func_bomb_target", "func_hostage_rescue",
              "func_vip_safetyzone", "func_escapezone")
SKY_RGB = (190, 205, 225)
TEX_FAR_MIP = 3                     # past this mip level: the texture's average colour
BOUNDARY_VOXELS = 2                 # solid voxels this close (Chebyshev) to air get a face
FACE_REACH_CELLS = 3.0              # a voxel's nearest face must be within this many cells


def pack_rgb(rgb):
    rgb = np.asarray(rgb, np.int64)
    return (rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)).astype(np.int32)


# ---------------------------------------------------------------------------------- the .bsp
def load_bsp(path):
    """-> verts, edges, surfedges, faces, texinfo, models, entity text, textures (name, w, h,
    the 4 mips as RGB through the texture's palette or None when external, the average)"""
    f = Path(path).read_bytes()
    ver = struct.unpack_from("<i", f, 0)[0]
    if ver != 30:
        raise ValueError(f"{path}: BSP v{ver}; --obs-texture reads GoldSrc v30")

    def lump(i):
        return struct.unpack_from("<ii", f, 4 + i * 8)
    vo, vl = lump(3)
    verts = np.frombuffer(f, np.float32, vl // 4, vo).reshape(-1, 3).astype(np.float64)
    eo, el = lump(12)
    edges = np.frombuffer(f, np.uint16, el // 2, eo).reshape(-1, 2)
    so, sl = lump(13)
    surfedges = np.frombuffer(f, np.int32, sl // 4, so)
    fo, fl = lump(7)
    faces = [struct.unpack_from("<HHiHH4Bi", f, fo + 20 * i) for i in range(fl // 20)]
    to, tl_ = lump(6)
    texinfo = [struct.unpack_from("<8fii", f, to + 40 * i) for i in range(tl_ // 40)]
    mo, ml = lump(14)
    models = [struct.unpack_from("<9f4i3i", f, mo + 64 * i) for i in range(ml // 64)]
    xo, xl = lump(0)
    ents = f[xo:xo + xl].decode("latin1", errors="replace")
    ao, _al = lump(2)
    n = struct.unpack_from("<i", f, ao)[0]
    offs = struct.unpack_from(f"<{n}i", f, ao + 4)
    tex = []
    for o in offs:
        if o < 0:
            tex.append({"name": "", "w": 1, "h": 1, "mips": None, "avg": (128, 128, 128)})
            continue
        base = ao + o
        name = f[base:base + 16].split(b"\0")[0].decode("latin1").lower()
        w, h = struct.unpack_from("<II", f, base + 16)
        mo4 = struct.unpack_from("<4I", f, base + 24)
        if mo4[0] == 0:                                   # external (a .wad): no pixels here
            tex.append({"name": name, "w": w, "h": h, "mips": None, "avg": (128, 128, 128)})
            continue
        pal_at = base + mo4[3] + (w >> 3) * (h >> 3) + 2
        pal = np.frombuffer(f, np.uint8, 768, pal_at).reshape(256, 3)
        mips = []
        for lv in range(4):
            wl, hl = max(w >> lv, 1), max(h >> lv, 1)
            idx = np.frombuffer(f, np.uint8, wl * hl, base + mo4[lv]).reshape(hl, wl)
            mips.append(pal[idx])
        m0 = mips[0].reshape(-1, 3).astype(np.float64)
        keep = (np.frombuffer(f, np.uint8, w * h, base + mo4[0]) != 255
                if name.startswith("{") else np.ones(w * h, bool))
        avg = tuple(int(round(x)) for x in (m0[keep].mean(0) if keep.any() else m0.mean(0)))
        tex.append({"name": name, "w": w, "h": h, "mips": mips, "avg": avg})
    return verts, edges, surfedges, faces, texinfo, models, ents, tex


def visible_models(ents, n_models):
    """the models to draw: the world and every brush entity that is not a trigger / invisible"""
    skip = set()
    for blk in re.findall(r"\{([^{}]*)\}", ents):
        kv = dict(re.findall(r'"([^"]*)"\s*"([^"]*)"', blk))
        m = kv.get("model", "")
        if not m.startswith("*"):
            continue
        cls = kv.get("classname", "")
        invisible = kv.get("rendermode", "0") not in ("0", "") and kv.get("renderamt", "255") == "0"
        if cls.startswith(SKIP_CLASS) or invisible:
            skip.add(int(m[1:]))
    return [i for i in range(n_models) if i not in skip]


def visible_triangles(path):
    """-> tris (T, 3, 3) float64, tface (T,) the face of each, and load_bsp's faces / texinfo /
    textures, plus (models drawn, models)"""
    verts, edges, surfedges, faces, texinfo, models, ents, tex = load_bsp(path)
    keep = visible_models(ents, len(models))
    tris, tface = [], []
    for mi in keep:
        first, num = models[mi][14], models[mi][15]   # dmodel_t: 9 floats, headnode[4], visleafs, firstface, numfaces
        for fi in range(first, first + num):
            _pl, _side, fe, ne, ti, *_rest = faces[fi]
            if tex[texinfo[ti][8]]["name"] in SKIP_TEX:
                continue
            vs = []
            for k in range(ne):
                se = int(surfedges[fe + k])
                vs.append(verts[edges[se][0]] if se >= 0 else verts[edges[-se][1]])
            for k in range(1, ne - 1):
                tris.append((vs[0], vs[k], vs[k + 1]))
                tface.append(fi)
    return (np.asarray(tris, np.float64).reshape(-1, 3, 3), np.asarray(tface, np.int64),
            faces, texinfo, tex, len(keep), len(models))


def face_table(tris, tface, faces, texinfo):
    """per face (F, 16) float32: s (x, y, z, offset), t (x, y, z, offset), the plane's unit normal
    (from its first triangle) and distance, texels per unit, 3 pad; and (F,) its texture"""
    nf = len(faces)
    fd = np.zeros((nf, 16), np.float32)
    ft = np.zeros(nf, np.int32)
    for fi in range(nf):
        ti = texinfo[faces[fi][4]]
        fd[fi, 0:4] = ti[0:4]
        fd[fi, 4:8] = ti[4:8]
        fd[fi, 12] = max(np.linalg.norm(ti[0:3]), np.linalg.norm(ti[4:7]))
        ft[fi] = ti[8]
    seen = np.zeros(nf, bool)
    for tr, fi in zip(tris, tface):
        if seen[fi]:
            continue
        c = np.cross(tr[1] - tr[0], tr[2] - tr[0])
        ln = float(np.linalg.norm(c))
        if ln > 1e-9:
            nrm = c / ln
            fd[fi, 8:11] = nrm
            fd[fi, 11] = float(nrm @ tr[0])
            seen[fi] = True
    return fd, ft


def atlas(tex):
    """every texture's 4 mips as one flat packed-RGB int32 array; per texture (w, h), the 4 mip
    offsets, the packed average and a sky flag"""
    chunks, base, dims, avg, sky = [], [], [], [], []
    at = 0
    for t in tex:
        row = []
        for lv in range(4):
            m = (t["mips"][lv] if t["mips"] is not None
                 else np.full((1, 1, 3), 128, np.uint8))
            row.append(at)
            chunks.append(pack_rgb(m.reshape(-1, 3)))
            at += m.shape[0] * m.shape[1]
        base.append(row)
        dims.append((t["w"], t["h"]) if t["mips"] is not None else (1, 1))
        avg.append(t["avg"])
        sky.append(int(t["name"].startswith("sky")))
    return (np.concatenate(chunks).astype(np.int32), np.asarray(base, np.int32).reshape(-1),
            np.asarray(dims, np.int32).reshape(-1), pack_rgb(np.asarray(avg)),
            np.asarray(sky, np.int32))


# ------------------------------------------------------------------ the bake (GPU, triton)
try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                       # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _nearest_tri(p_ptr, tri_ptr, out_i, out_d2, NP, T,
                     BP: tl.constexpr, BT: tl.constexpr):
        """per point: the nearest triangle (exact point-triangle distance, brute force)"""
        pid = tl.program_id(0) * BP + tl.arange(0, BP)
        pm = pid < NP
        px = tl.load(p_ptr + pid * 3 + 0, mask=pm, other=0.0)[:, None]
        py = tl.load(p_ptr + pid * 3 + 1, mask=pm, other=0.0)[:, None]
        pz = tl.load(p_ptr + pid * 3 + 2, mask=pm, other=0.0)[:, None]
        best = tl.full([BP], 3.0e38, tl.float32)
        bi = tl.full([BP], -1, tl.int32)
        for j in range(0, T, BT):
            tid = j + tl.arange(0, BT)
            tm = tid < T
            ax = tl.load(tri_ptr + tid * 9 + 0, mask=tm, other=0.0)[None, :]
            ay = tl.load(tri_ptr + tid * 9 + 1, mask=tm, other=0.0)[None, :]
            az = tl.load(tri_ptr + tid * 9 + 2, mask=tm, other=0.0)[None, :]
            e1x = tl.load(tri_ptr + tid * 9 + 3, mask=tm, other=0.0)[None, :]
            e1y = tl.load(tri_ptr + tid * 9 + 4, mask=tm, other=0.0)[None, :]
            e1z = tl.load(tri_ptr + tid * 9 + 5, mask=tm, other=0.0)[None, :]
            e2x = tl.load(tri_ptr + tid * 9 + 6, mask=tm, other=0.0)[None, :]
            e2y = tl.load(tri_ptr + tid * 9 + 7, mask=tm, other=0.0)[None, :]
            e2z = tl.load(tri_ptr + tid * 9 + 8, mask=tm, other=0.0)[None, :]
            wx = px - ax
            wy = py - ay
            wz = pz - az
            d00 = e1x * e1x + e1y * e1y + e1z * e1z
            d01 = e1x * e2x + e1y * e2y + e1z * e2z
            d11 = e2x * e2x + e2y * e2y + e2z * e2z
            d20 = wx * e1x + wy * e1y + wz * e1z
            d21 = wx * e2x + wy * e2y + wz * e2z
            den = d00 * d11 - d01 * d01
            okd = den > 1e-9
            iden = 1.0 / tl.where(okd, den, 1.0)
            bv = (d11 * d20 - d01 * d21) * iden
            bw = (d00 * d21 - d01 * d20) * iden
            inside = okd & (bv >= 0.0) & (bw >= 0.0) & (bv + bw <= 1.0)
            # inside: the distance to the plane
            qx = wx - bv * e1x - bw * e2x
            qy = wy - bv * e1y - bw * e2y
            qz = wz - bv * e1z - bw * e2z
            dpl = qx * qx + qy * qy + qz * qz
            # outside: the nearest of the three edges
            s1 = tl.minimum(tl.maximum(d20 / tl.maximum(d00, 1e-12), 0.0), 1.0)
            ux = wx - s1 * e1x
            uy = wy - s1 * e1y
            uz = wz - s1 * e1z
            de = ux * ux + uy * uy + uz * uz
            s2 = tl.minimum(tl.maximum(d21 / tl.maximum(d11, 1e-12), 0.0), 1.0)
            ux = wx - s2 * e2x
            uy = wy - s2 * e2y
            uz = wz - s2 * e2z
            de = tl.minimum(de, ux * ux + uy * uy + uz * uz)
            e3x = e2x - e1x
            e3y = e2y - e1y
            e3z = e2z - e1z
            vx = wx - e1x
            vy = wy - e1y
            vz = wz - e1z
            d33 = e3x * e3x + e3y * e3y + e3z * e3z
            s3 = tl.minimum(tl.maximum((vx * e3x + vy * e3y + vz * e3z)
                                       / tl.maximum(d33, 1e-12), 0.0), 1.0)
            ux = vx - s3 * e3x
            uy = vy - s3 * e3y
            uz = vz - s3 * e3z
            de = tl.minimum(de, ux * ux + uy * uy + uz * uz)
            d2 = tl.where(inside, dpl, de)
            d2 = tl.where(tm[None, :], d2, 3.0e38)
            dmin = tl.min(d2, axis=1)
            imin = tl.argmin(d2, axis=1).to(tl.int32) + j
            better = dmin < best
            best = tl.where(better, dmin, best)
            bi = tl.where(better, imin, bi)
        tl.store(out_i + pid, bi, mask=pm)
        tl.store(out_d2 + pid, best, mask=pm)


def nearest_triangles(points, tri9):
    """(N, 3) float32 cuda points, (T, 9) triangles (a, b - a, c - a) -> (tri index, dist^2)"""
    import torch
    n = points.shape[0]
    oi = torch.empty(n, dtype=torch.int32, device=points.device)
    od = torch.empty(n, dtype=torch.float32, device=points.device)
    if n:
        _nearest_tri[(triton.cdiv(n, 32),)](points.contiguous(), tri9, oi, od, n,
                                            tri9.shape[0], BP=32, BT=64, num_warps=4)
    return oi, od


def nearest_triangles_torch(points, tri9, chunk=256):
    """the same, in torch (CPU / the reference for tests)"""
    import torch
    a, e1, e2 = tri9[:, 0:3], tri9[:, 3:6], tri9[:, 6:9]
    out_i, out_d = [], []
    for s in range(0, points.shape[0], chunk):
        w = points[s:s + chunk, None, :] - a[None]
        d00 = (e1 * e1).sum(-1)
        d01 = (e1 * e2).sum(-1)
        d11 = (e2 * e2).sum(-1)
        d20 = (w * e1).sum(-1)
        d21 = (w * e2).sum(-1)
        den = d00 * d11 - d01 * d01
        okd = den > 1e-9
        iden = 1.0 / torch.where(okd, den, torch.ones_like(den))
        bv = (d11 * d20 - d01 * d21) * iden
        bw = (d00 * d21 - d01 * d20) * iden
        inside = okd & (bv >= 0) & (bw >= 0) & (bv + bw <= 1)
        q = w - bv[..., None] * e1 - bw[..., None] * e2
        dpl = (q * q).sum(-1)
        s1 = (d20 / d00.clamp(min=1e-12)).clamp(0, 1)
        de = ((w - s1[..., None] * e1) ** 2).sum(-1)
        s2 = (d21 / d11.clamp(min=1e-12)).clamp(0, 1)
        de = torch.minimum(de, ((w - s2[..., None] * e2) ** 2).sum(-1))
        e3 = e2 - e1
        v = w - e1
        s3 = ((v * e3).sum(-1) / (e3 * e3).sum(-1).clamp(min=1e-12)).clamp(0, 1)
        de = torch.minimum(de, ((v - s3[..., None] * e3) ** 2).sum(-1))
        d2 = torch.where(inside, dpl, de)
        dm, im = d2.min(1)
        out_i.append(im.to(torch.int32))
        out_d.append(dm)
    return torch.cat(out_i), torch.cat(out_d)


class TextureMap:
    """--obs-texture's data on the device, for GpuLidar: the face-id grid (int16, the depth
    field's voxels, -1 = no visible face), the per-face table and the texture atlas."""

    def __init__(self, fid, fdata, ftex, atlas_rgb, tmip, tdim, tavg, tsky, describe=""):
        self.fid, self.fdata, self.ftex = fid, fdata, ftex
        self.atlas, self.tmip, self.tdim, self.tavg, self.tsky = atlas_rgb, tmip, tdim, tavg, tsky
        self.sky_packed = int(pack_rgb(np.asarray(SKY_RGB)))
        self._describe = describe

    def describe(self):
        return self._describe

    @classmethod
    def for_lidar(cls, bsp_path, sdf_flat, dims, mins, cell, sdf_sig, device, cache_dir=None):
        """parse the .bsp, then load or bake the face-id grid on the depth field's voxels"""
        import torch
        tris, tface, faces, texinfo, tex, nkeep, nmod = visible_triangles(bsp_path)
        fd, ft = face_table(tris, tface, faces, texinfo)
        at, tmip, tdim, tavg, tsky = atlas(tex)
        bsp = Path(bsp_path)
        cache = Path(cache_dir) if cache_dir else bsp.parent
        cfile = cache / f"{bsp.stem}.faceid_{float(cell):g}.npz"
        sig = f"{sdf_sig}_{TEXMAP_SEMANTICS}"
        nz, ny, nx = dims
        fid = None
        if cfile.exists():
            z = np.load(cfile, allow_pickle=False)
            if str(z["sig"]) == sig:
                fid = torch.full((nz * ny * nx,), -1, dtype=torch.int16, device=device)
                fid[torch.as_tensor(z["idx"].astype(np.int64), device=device)] = \
                    torch.as_tensor(z["face"], device=device)
        baked = fid is None
        if baked and (not HAVE_TRITON or torch.device(device).type != "cuda"):
            raise RuntimeError(
                f"--obs-texture: {cfile.name} is not baked for this map and cell, and the bake "
                "needs CUDA + triton; run the trainer (or record_ckpt) once on a GPU - it caches")
        if baked:
            tri9 = torch.as_tensor(np.concatenate([tris[:, 0], tris[:, 1] - tris[:, 0],
                                                   tris[:, 2] - tris[:, 0]], 1),
                                   dtype=torch.float32, device=device).contiguous()
            fid, idx, face = bake_face_grid(sdf_flat, dims, mins, cell, tri9,
                                            torch.as_tensor(tface, device=device))
            np.savez_compressed(cfile, idx=idx.astype(np.int32), face=face.astype(np.int16),
                                sig=np.str_(sig))
        n_set = int((fid >= 0).sum())
        msg = (f"texture: {len(tris):,} visible triangles of {len(set(tface.tolist())):,} faces "
               f"({nkeep}/{nmod} models), {len(tex)} textures ({at.size * 4 / 1e6:.1f} MB RGBA), "
               f"face-id grid {'BAKED' if baked else 'cached'}: {n_set:,} surface voxels "
               f"({fid.numel() * 2 / 1e9:.2f} GB int16) -> +3 channels (R, G, B; far = the "
               f"texture's average, past mip {TEX_FAR_MIP})")
        t = torch.as_tensor
        return cls(fid, t(fd, device=device).reshape(-1).contiguous(), t(ft, device=device),
                   t(at, device=device), t(tmip, device=device), t(tdim, device=device),
                   t(tavg, device=device), t(tsky, device=device), describe=msg)


def bake_face_grid(sdf_flat, dims, mins, cell, tri9, tface, reach_cells=FACE_REACH_CELLS):
    """the face-id grid: every SOLID voxel within BOUNDARY_VOXELS (Chebyshev) of air gets the
    face of its nearest triangle, if that is within reach_cells cells; the rest -1.
    -> (the dense int16 grid on the device, the set voxels' flat indices, their faces)"""
    import torch
    import torch.nn.functional as F
    nz, ny, nx = dims
    dev = sdf_flat.device
    s3 = sdf_flat.view(nz, ny, nx)
    k = BOUNDARY_VOXELS
    picks = []
    for z0 in range(0, nz, 32):
        z1 = min(nz, z0 + 32)
        a0, a1 = max(0, z0 - k), min(nz, z1 + k)
        air = (s3[a0:a1] > 0).to(torch.float16)[None, None]
        near = F.max_pool3d(air, 2 * k + 1, stride=1, padding=k)[0, 0]
        near = near[z0 - a0:z0 - a0 + (z1 - z0)] > 0
        b = ((s3[z0:z1] == 0) & near).nonzero()
        b[:, 0] += z0
        picks.append(b)
    zyx = torch.cat(picks)
    flat = (zyx[:, 0] * ny + zyx[:, 1]) * nx + zyx[:, 2]
    mn = torch.as_tensor(np.asarray(mins, np.float32), device=dev)
    centers = mn + (zyx[:, [2, 1, 0]].float() + 0.5) * float(cell)
    tri_i = torch.empty(len(centers), dtype=torch.int32, device=dev)
    d2 = torch.empty(len(centers), dtype=torch.float32, device=dev)
    step = 1 << 20
    for s in range(0, len(centers), step):
        tri_i[s:s + step], d2[s:s + step] = nearest_triangles(centers[s:s + step], tri9)
    face = tface[tri_i.clamp(min=0).long()]
    face = torch.where((d2 <= (reach_cells * float(cell)) ** 2) & (tri_i >= 0), face,
                       torch.full_like(face, -1))
    fid = torch.full((nz * ny * nx,), -1, dtype=torch.int16, device=dev)
    keep = face >= 0
    fid[flat[keep]] = face[keep].to(torch.int16)
    return fid, flat[keep].cpu().numpy(), face[keep].to(torch.int16).cpu().numpy()
