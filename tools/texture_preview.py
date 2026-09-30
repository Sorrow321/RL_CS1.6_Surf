"""texture_preview.py - what a TEXTURED policy image would look like, on a recorded rollout (the
user, 2026-09-30: "take some rollout and visualize what the render would look like").

Visualization only (nothing here trains). For every frame of an episode:

  1. the CURRENT input: the depth channel as the policy receives it (GpuLidar, legacy encoding);
  2. the TEXTURED view at the policy's own resolution and camera (64 x 32, the equiangular 120 x 90
     lidar), prefiltered the way a trainable render must be: the mip level whose texel matches the
     pixel's footprint, and past mip 3 the texture's AVERAGE colour (at 64 x 32 a pixel spans ~33 u
     at 1,000 u, so finer detail would only alias into frame-to-frame noise);
  3. the same textured render at a higher resolution, for a human to see what is there - NOT the
     policy's view.

Textures, texinfo and faces come from the .bsp itself (all embedded on cannonball). Hits are exact
ray-triangle intersections (a brute-force Triton kernel over every visible triangle) - the ideal a
voxel face-id grid would approximate. Unlit (the texture's albedo, no lightmaps). Invisible brushes
(triggers, clip, null, hint, skip, origin) are not drawn; sky faces are a flat sky colour.

    python tools/texture_preview.py runs/dencCTL/traj_1435500544.jsonl --episode best \\
        --out runs/research/texture_preview/dencCTL_textured.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import triton                      # noqa: E402
import triton.language as tl       # noqa: E402

SKIP_TEX = {"aaatrigger", "clip", "null", "origin", "hint", "skip", "bevel", "trigger"}
SKIP_CLASS = ("trigger_", "func_buyzone", "func_bomb_target", "func_hostage_rescue",
              "func_vip_safetyzone", "func_escapezone")
SKY_RGB = (190, 205, 225)


# ---------------------------------------------------------------------------------- the .bsp
def load_bsp(path):
    f = open(path, "rb").read()
    ver = struct.unpack_from("<i", f, 0)[0]
    if ver != 30:
        raise SystemExit(f"{path}: BSP v{ver}, this reads GoldSrc v30")

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
    # the textures: name, and each of the 4 mips as RGB through the texture's own palette
    ao, _al = lump(2)
    n = struct.unpack_from("<i", f, ao)[0]
    offs = struct.unpack_from(f"<{n}i", f, ao + 4)
    tex = []
    for o in offs:
        base = ao + o
        name = f[base:base + 16].split(b"\0")[0].decode("latin1").lower()
        w, h = struct.unpack_from("<II", f, base + 16)
        mo4 = struct.unpack_from("<4I", f, base + 24)
        if mo4[0] == 0:                                   # external (.wad): not embedded
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
        keep = np.frombuffer(f, np.uint8, w * h, base + mo4[0]) != 255 if name.startswith("{") \
            else np.ones(w * h, bool)
        avg = tuple(int(x) for x in (m0[keep].mean(0) if keep.any() else m0.mean(0)))
        tex.append({"name": name, "w": w, "h": h, "mips": mips, "avg": avg})
    return verts, edges, surfedges, faces, texinfo, models, ents, tex


def visible_models(ents, n_models):
    """model indices to draw: the world and every brush entity that is not a trigger / invisible"""
    skip = set()
    for blk in re.findall(r"\{([^{}]*)\}", ents):
        kv = dict(re.findall(r'"([^"]*)"\s*"([^"]*)"', blk))
        m = kv.get("model", "")
        if not m.startswith("*"):
            continue
        cls = kv.get("classname", "")
        invisible = (kv.get("rendermode", "0") not in ("0", "") and kv.get("renderamt", "255") == "0")
        if cls.startswith(SKIP_CLASS) or invisible:
            skip.add(int(m[1:]))
    return [i for i in range(n_models) if i not in skip]


def triangles(path):
    verts, edges, surfedges, faces, texinfo, models, ents, tex = load_bsp(path)
    keep_models = visible_models(ents, len(models))
    tris, tface = [], []
    for mi in keep_models:
        first, num = models[mi][14], models[mi][15]   # dmodel_t: 9 floats, headnode[4], visleafs, firstface, numfaces
        for fi in range(first, first + num):
            _pl, _side, fe, ne, ti, *_rest = faces[fi]
            name = tex[texinfo[ti][8]]["name"]
            if name in SKIP_TEX:
                continue
            vs = []
            for k in range(ne):
                se = int(surfedges[fe + k])
                vs.append(verts[edges[se][0]] if se >= 0 else verts[edges[-se][1]])
            for k in range(1, ne - 1):
                tris.append((vs[0], vs[k], vs[k + 1]))
                tface.append(fi)
    tris = np.asarray(tris, np.float64)
    return tris, np.asarray(tface, np.int64), faces, texinfo, tex, len(keep_models), len(models)


# ------------------------------------------------------------------------ the exact ray cast
@triton.jit
def _raycast(o_ptr, d_ptr, tri_ptr, out_t, out_i, R, T, BR: tl.constexpr, BT: tl.constexpr):
    rid = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rid < R
    ox = tl.load(o_ptr + rid * 3 + 0, mask=rm, other=0.0)[:, None]
    oy = tl.load(o_ptr + rid * 3 + 1, mask=rm, other=0.0)[:, None]
    oz = tl.load(o_ptr + rid * 3 + 2, mask=rm, other=0.0)[:, None]
    dx = tl.load(d_ptr + rid * 3 + 0, mask=rm, other=0.0)[:, None]
    dy = tl.load(d_ptr + rid * 3 + 1, mask=rm, other=0.0)[:, None]
    dz = tl.load(d_ptr + rid * 3 + 2, mask=rm, other=0.0)[:, None]
    best_t = tl.full([BR], 3.0e38, tl.float32)
    best_i = tl.full([BR], -1, tl.int32)
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
        px = dy * e2z - dz * e2y
        py = dz * e2x - dx * e2z
        pz = dx * e2y - dy * e2x
        det = e1x * px + e1y * py + e1z * pz
        ok = tl.abs(det) > 1e-9
        inv = 1.0 / tl.where(ok, det, 1.0)
        tx = ox - ax
        ty = oy - ay
        tz = oz - az
        u = (tx * px + ty * py + tz * pz) * inv
        qx = ty * e1z - tz * e1y
        qy = tz * e1x - tx * e1z
        qz = tx * e1y - ty * e1x
        v = (dx * qx + dy * qy + dz * qz) * inv
        t = (e2x * qx + e2y * qy + e2z * qz) * inv
        hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.5) & tm[None, :]
        t = tl.where(hit, t, 3.0e38)
        tmin = tl.min(t, axis=1)
        imin = tl.argmin(t, axis=1).to(tl.int32) + j
        better = tmin < best_t
        best_t = tl.where(better, tmin, best_t)
        best_i = tl.where(better, imin, best_i)
    tl.store(out_t + rid, best_t, mask=rm)
    tl.store(out_i + rid, best_i, mask=rm)


def raycast(o, d, tri9):
    R, T = d.shape[0], tri9.shape[0]
    out_t = torch.empty(R, device=d.device, dtype=torch.float32)
    out_i = torch.empty(R, device=d.device, dtype=torch.int32)
    BR, BT = 32, 64
    _raycast[(triton.cdiv(R, BR),)](o.contiguous(), d.contiguous(), tri9, out_t, out_i, R, T,
                                    BR=BR, BT=BT, num_warps=4)
    return out_t, out_i


# ------------------------------------------------------------------------------ the textures
class Atlas:
    """every texture's 4 mips as one flat RGB array + per texture offsets / dims + its average"""

    def __init__(self, tex, device):
        chunks, base, dims, avg, sky = [], [], [], [], []
        at = 0
        for t in tex:
            row = []
            for lv in range(4):
                if t["mips"] is None:
                    m = np.full((1, 1, 3), 128, np.uint8)
                else:
                    m = t["mips"][lv]
                row.append(at)
                chunks.append(m.reshape(-1, 3))
                at += m.shape[0] * m.shape[1]
            base.append(row)
            dims.append((t["w"], t["h"]) if t["mips"] is not None else (1, 1))
            avg.append(t["avg"])
            sky.append(t["name"].startswith("sky"))
        self.rgb = torch.as_tensor(np.concatenate(chunks), device=device)
        self.base = torch.as_tensor(np.asarray(base, np.int64), device=device)
        self.dims = torch.as_tensor(np.asarray(dims, np.int64), device=device)
        self.avg = torch.as_tensor(np.asarray(avg, np.uint8), device=device)
        self.sky = torch.as_tensor(np.asarray(sky, bool), device=device)


def shade(o, d, t, tri_id, tface_t, finfo, atlas, pix_rad, prefilter=True):
    """the hit's colour: the texture at the hit's (u, v), at the mip matching the footprint, and
    past mip 3 the texture's average; sky a flat colour; a miss black"""
    R = d.shape[0]
    rgb = torch.zeros(R, 3, dtype=torch.uint8, device=d.device)
    hit = tri_id >= 0
    if not hit.any():
        return rgb
    fi = tface_t[tri_id[hit].long()]
    s = finfo["s"][fi]
    tt = finfo["t"][fi]
    ti = finfo["tex"][fi]
    n = finfo["n"][fi]
    p = o[hit] + d[hit] * t[hit, None]
    u = (p * s[:, :3]).sum(1) + s[:, 3]
    v = (p * tt[:, :3]).sum(1) + tt[:, 3]
    # footprint: texels per pixel along the ray, stretched by the grazing angle
    tex_per_u = torch.maximum(s[:, :3].norm(dim=1), tt[:, :3].norm(dim=1))
    cosi = (d[hit] * n).sum(1).abs().clamp(min=0.15)
    foot = t[hit] * pix_rad * tex_per_u / cosi
    lev = torch.floor(torch.log2(foot.clamp(min=1.0))) if prefilter else torch.zeros_like(foot)
    far = lev > 3
    lv = lev.clamp(0, 3).long()
    w = (atlas.dims[ti, 0] >> lv).clamp(min=1)
    h = (atlas.dims[ti, 1] >> lv).clamp(min=1)
    sc = torch.pow(2.0, -lv.double()).float()
    iu = torch.remainder(torch.floor(u * sc).long(), w)
    iv = torch.remainder(torch.floor(v * sc).long(), h)
    col = atlas.rgb[atlas.base[ti, lv] + iv * w + iu]
    col = torch.where(far[:, None], atlas.avg[ti], col)
    col = torch.where(atlas.sky[ti][:, None], torch.tensor(SKY_RGB, dtype=torch.uint8,
                                                           device=d.device), col)
    rgb[hit] = col
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", help="a recorded episode file (record_rollout JSONL)")
    ap.add_argument("--episode", default="best", help="index, or 'best' (the longest)")
    ap.add_argument("--map", default=None, help="the .bsp (default: the file's header map)")
    ap.add_argument("--every", type=int, default=4, help="every Nth tick (4 = 25 fps at 10 ms)")
    ap.add_argument("--hires", default="256x128", help="the reference panel's resolution")
    ap.add_argument("--panel", default="512x384", help="each panel's display size")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from surfgym import SurfCore, default_config
    from surfgym.vision import GpuLidar, pick_cell
    dev = "cuda"
    # the episode
    eps, rows, hdr = [], [], None
    for line in open(a.traj, encoding="utf-8"):
        r = json.loads(line)
        if isinstance(r, list):
            rows.append(r)
        elif "end" in r:
            eps.append((np.asarray(rows, np.float64), r))
            rows = []
        elif "episode" in r or "map" in r:
            hdr = hdr or r
            rows = []
    if a.episode == "best":
        k = int(np.argmax([len(e[0]) for e in eps]))
    else:
        k = int(a.episode)
    rows = eps[k][0][::a.every]
    if a.max_frames:
        rows = rows[:a.max_frames]
    stem = (hdr or {}).get("map") or "surf_src_cannonball"
    bsp = a.map or next((str(d / f"{stem}.bsp") for d in (ROOT / "maps", ROOT / "maps_pool")
                         if (d / f"{stem}.bsp").is_file()), None)
    print(f"{a.traj} episode {k}: {len(eps[k][0])} ticks -> {len(rows)} frames; map {bsp}")
    # the map: triangles, per-face texinfo, the atlas
    tris, tface, faces, texinfo, tex, nkeep, nmod = triangles(bsp)
    print(f"{len(tris):,} triangles from {len(set(tface.tolist())):,} faces "
          f"({nkeep}/{nmod} models drawn), {len(tex)} textures")
    tri9 = torch.as_tensor(np.concatenate([tris[:, 0], tris[:, 1] - tris[:, 0],
                                           tris[:, 2] - tris[:, 0]], 1), dtype=torch.float32,
                           device=dev).contiguous()
    nf = len(faces)
    s = np.zeros((nf, 4), np.float32)
    t4 = np.zeros((nf, 4), np.float32)
    ti = np.zeros(nf, np.int64)
    nrm = np.zeros((nf, 3), np.float32)
    for fi in range(nf):
        tinf = texinfo[faces[fi][4]]
        s[fi] = tinf[0:4]
        t4[fi] = tinf[4:8]
        ti[fi] = tinf[8]
    for tr, fi in zip(tris, tface):
        c = np.cross(tr[1] - tr[0], tr[2] - tr[0])
        ln = np.linalg.norm(c)
        if ln > 0:
            nrm[fi] = c / ln
    finfo = {"s": torch.as_tensor(s, device=dev), "t": torch.as_tensor(t4, device=dev),
             "tex": torch.as_tensor(ti, device=dev), "n": torch.as_tensor(nrm, device=dev)}
    tface_t = torch.as_tensor(tface, device=dev)
    atlas = Atlas(tex, dev)
    core = SurfCore(bsp, default_config(num_envs=1, lidar_w=0, lidar_h=0))
    cell = 32.0 if "cannonball" in Path(bsp).stem else pick_cell(core)
    lid = GpuLidar(core, 64, 32, range_units=11500.0, near_range=2000.0, cell=cell, device=dev)
    hw, hh = (int(x) for x in a.hires.split("x"))
    # the policy's camera and a finer one with the same fov and projection (directions only)
    import cv2
    PW, PH = (int(x) for x in a.panel.split("x"))

    def dirs(lidar_like, N, yw, pt):
        lidar_like._ensure_buffers(N)
        lidar_like._dirs_equiangular(N, yw, pt, float(np.pi / 180.0))
        return torch.stack((lidar_like._dx, lidar_like._dy, lidar_like._dz), -1)

    class _Cam:
        pass
    hi = _Cam()
    hi.H, hi.W, hi.device, hi._buf_n = hh, hw, lid.device, 0
    d2r = np.pi / 180.0
    hi.yoff = torch.as_tensor((120.0 * (0.5 - (np.arange(hw) + 0.5) / hw)) * d2r,
                              dtype=torch.float32, device=dev)
    hi.poff = torch.as_tensor((90.0 * (0.5 - (np.arange(hh) + 0.5) / hh)) * d2r,
                              dtype=torch.float32, device=dev)
    hi._ensure_buffers = GpuLidar._ensure_buffers.__get__(hi)
    hi._dirs_equiangular = GpuLidar._dirs_equiangular.__get__(hi)
    pix_lo = float(np.radians(90.0 / 32))                     # the coarser of the two axes
    pix_hi = float(np.radians(90.0 / hh))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    FW, FH = 3 * PW, PH + 40
    ff = shutil.which("ffmpeg")
    enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{FW}x{FH}", "-r", str(round(100.0 / a.every)), "-i", "-",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)

    def label(img, txt, y=20, scale=0.5):
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                    cv2.LINE_AA)

    B = 32
    agree = []
    for s0 in range(0, len(rows), B):
        rb = rows[s0:s0 + B]
        n = len(rb)
        org = torch.as_tensor(rb[:, 1:4], dtype=torch.float32, device=dev)
        yw = torch.as_tensor(rb[:, 7], dtype=torch.float32, device=dev)
        pt = torch.as_tensor(rb[:, 12] if rb.shape[1] > 12 else np.zeros(n), dtype=torch.float32,
                             device=dev)
        dk = torch.as_tensor(((rb[:, 8].astype(np.int64) & 4) != 0).astype(np.int32), device=dev)
        depth = lid.render(org, yw, pt, dk).cpu().numpy()
        eye = org.clone()
        eye[:, 2] += torch.where(dk != 0, 12.0, 17.0)
        out_img = []
        for cam, pw, ph, prad in ((lid, 64, 32, pix_lo), (hi, hw, hh, pix_hi)):
            D = dirs(cam, n, yw, pt).reshape(-1, 3)
            O = eye.repeat_interleave(pw * ph, 0)
            tt, tri = raycast(O, D, tri9)
            col = shade(O, D, tt, tri, tface_t, finfo, atlas, prad)
            out_img.append(col.reshape(n, ph, pw, 3).cpu().numpy())
            if cam is lid:
                # the exact hit vs the depth march, for the log: the same surfaces?
                t_sdf = lid.decode_depth(torch.as_tensor(depth, device=dev)).reshape(-1)
                both = (tt < 11500) & (t_sdf < 11400)
                if both.any():
                    agree.append(float(((tt[both] - t_sdf[both]).abs() < 64.0).float().mean()))
        for i in range(n):
            dimg = (np.clip(1.0 - depth[i] / 1.25, 0, 1) * 255).astype(np.uint8)
            p1 = cv2.resize(cv2.applyColorMap(dimg, cv2.COLORMAP_TURBO), (PW, PH),
                            interpolation=cv2.INTER_NEAREST)
            p2 = cv2.resize(np.ascontiguousarray(out_img[0][i][..., ::-1]), (PW, PH),
                            interpolation=cv2.INTER_NEAREST)
            p3 = cv2.resize(np.ascontiguousarray(out_img[1][i][..., ::-1]), (PW, PH),
                            interpolation=cv2.INTER_AREA)
            label(p1, "NOW: depth channel, 64x32")
            label(p2, "TEXTURED, policy res 64x32")
            label(p2, "(far = texture's average colour)", 40, 0.45)
            label(p3, f"textured {hw}x{hh} - reference only,")
            label(p3, "NOT what the policy would see", 40, 0.45)
            for p in (p1, p2, p3):
                cv2.drawMarker(p, (PW // 2, PH // 2), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
            bar = np.full((40, FW, 3), 24, np.uint8)
            r = rb[i]
            label(bar, f"{Path(a.traj).parent.name} ep {k}  t {r[0] / 100.0:5.1f} s  speed "
                       f"{np.hypot(r[4], r[5]):5.0f} u/s   unlit texture albedo, exact ray cast",
                  26, 0.55)
            frame = np.vstack((np.hstack((p1, p2, p3)), bar))
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        print(f"  {min(s0 + B, len(rows))}/{len(rows)} frames", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"exact hit within 64 u of the depth march on {np.mean(agree):.1%} of the policy pixels")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
