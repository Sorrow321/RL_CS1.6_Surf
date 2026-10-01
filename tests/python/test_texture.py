"""--obs-texture (surfgym/texmap.py + vision.py's texture kernels, the user 2026-09-30).

(a) the .bsp read: cannonball's 103 textures are all embedded, every mip converted, the average
    of a texture is the mean of its mip 0; the per-face table carries unit plane normals.
(b) CUDA: the brute-force nearest-triangle kernel equals its torch reference.
(c) CUDA, cannonball's caches: the texture lidar's depth channel is the shipped kernel's bit for
    bit (plain and with the potential, whose channel is unchanged too); the RGB of the triton
    kernel equals the torch reference; the face-id grid names the face an EXACT ray cast hits on
    most pixels (tools/texture_preview.py); refusals.
(d) a trainer smoke (CPU, the cached face grid): trains, in_ch 5 with the potential, record_ckpt
    mirrors it, a resume restores it, a mismatch is refused, off writes no key.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym import texmap, vision                               # noqa: E402
from test_obs_potential import (ABS, CANNONBALL, CPOSES, RECORD,   # noqa: E402
                                _run, _train, needs_cuda_caches, scene)  # noqa: F401
from test_view_continuous import needs_run                         # noqa: E402

FACEID = CANNONBALL.parent / "surf_src_cannonball.faceid_32.npz"


@pytest.mark.skipif(not CANNONBALL.exists(), reason="needs cannonball")
def test_the_bsp_read():
    tris, tface, faces, texinfo, tex, nkeep, nmod = texmap.visible_triangles(str(CANNONBALL))
    assert len(tex) == 103 and all(t["mips"] is not None for t in tex)
    for t in tex[:5]:
        for lv in range(4):
            assert t["mips"][lv].shape == (max(t["h"] >> lv, 1), max(t["w"] >> lv, 1), 3)
    t0 = next(t for t in tex if not t["name"].startswith("{"))
    assert np.allclose(t0["avg"], t0["mips"][0].reshape(-1, 3).mean(0), atol=0.51)
    fd, ft = texmap.face_table(tris, tface, faces, texinfo)
    used = np.unique(tface)
    nrm = np.linalg.norm(fd[used, 8:11], axis=1)
    assert np.allclose(nrm, 1.0, atol=1e-5)
    assert np.all(fd[used, 12] > 0)                     # texels per unit
    at, tmip, tdim, tavg, tsky = texmap.atlas(tex)
    assert at.size == 6_634_760 and tmip.size == 4 * 103 and tsky.sum() >= 1


@pytest.mark.skipif(not (torch.cuda.is_available() and texmap.HAVE_TRITON), reason="needs CUDA")
def test_nearest_triangle_kernel_equals_the_reference():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(300, 3, generator=g) * 1000
    tri9 = torch.cat([a, torch.randn(300, 3, generator=g) * 100,
                      torch.randn(300, 3, generator=g) * 100], 1).cuda()
    pts = (torch.rand(5000, 3, generator=g) * 1000).cuda()
    i_t, d_t = texmap.nearest_triangles(pts, tri9)
    i_r, d_r = texmap.nearest_triangles_torch(pts, tri9)
    assert torch.allclose(d_t, d_r, rtol=1e-4, atol=1e-2)
    same = (i_t == i_r) | torch.isclose(d_t, d_r, rtol=1e-5, atol=1e-3)
    assert same.all()


def test_refusals(scene):
    for kw in ({"pinhole": True}, {"normals": True}, {"surf_mask": True},
               {"depth_enc": "log"}):
        with pytest.raises(ValueError):
            vision.GpuLidar(None, 16, 8, cell=8.0, device="cpu", texture=True, **kw)


@needs_cuda_caches
def test_texture_kernels_on_cannonball():
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import build_goal_field
    from surfgym.zones import load_zones
    import texture_preview as TP
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    kw = dict(range_units=11500.0, near_range=2000.0, cell=32.0)
    lg = vision.GpuLidar(core, 64, 32, device="cuda", texture=True, **kw)
    assert lg.channels == 4 and lg.tex_channel == 1
    lc = vision.GpuLidar(core, 64, 32, device="cpu", texture=True, **kw)     # the cached grid
    ship = vision.GpuLidar(core, 64, 32, device="cuda", **kw)
    a = np.asarray(CPOSES, np.float32)

    def tens(dev):
        return (torch.as_tensor(a[:, 0:3]).to(dev), torch.as_tensor(a[:, 3]).to(dev),
                torch.as_tensor(a[:, 4]).to(dev), torch.as_tensor(a[:, 5]).to(dev))
    og = lg.render(*tens("cuda")).cpu()
    oc = lc.render(*tens("cpu"))
    assert torch.equal(og[..., 0], ship.render(*tens("cuda")).cpu()), "depth ABI drift"
    same = (og[..., 1:] - oc[..., 1:]).abs().max(-1).values < 1e-6
    assert same.float().mean() > 0.999
    assert og[..., 1:].max() <= 1.0 and og[..., 1:].min() >= 0.0
    # with the potential: (depth, potential, R, G, B), depth + potential the shipped pot kernel's
    gf = build_goal_field(core, load_zones(str(CANNONBALL))["end"], cell=32.0)
    lp = vision.GpuLidar(core, 64, 32, device="cuda", texture=lg.texmap,
                         potential=vision.LidarPotential(gf, "norm", device="cuda"), **kw)
    sp = vision.GpuLidar(core, 64, 32, device="cuda",
                         potential=vision.LidarPotential(gf, "norm", device="cuda"), **kw)
    assert (lp.channels, lp.pot_channel, lp.tex_channel) == (5, 1, 2)
    op = lp.render(*tens("cuda")).cpu()
    assert torch.equal(op[..., :2], sp.render(*tens("cuda")).cpu())
    assert torch.equal(op[..., 2:], og[..., 1:])
    # the grid's face against an exact ray cast at the policy's pixels
    tris, tface, faces, texinfo, tex, _, _ = TP.triangles(str(CANNONBALL))
    tri9 = torch.as_tensor(np.concatenate([tris[:, 0], tris[:, 1] - tris[:, 0],
                                           tris[:, 2] - tris[:, 0]], 1),
                           dtype=torch.float32, device="cuda").contiguous()
    o, yw, pt, dk = tens("cuda")
    N = len(a)
    lg._ensure_buffers(N)
    lg._dirs_equiangular(N, yw, pt, float(np.pi / 180))
    D = torch.stack((lg._dx, lg._dy, lg._dz), -1).reshape(-1, 3)
    eye = o.clone()
    eye[:, 2] += torch.where(dk != 0, 12.0, 17.0)
    O = eye.repeat_interleave(64 * 32, 0)
    _tt, tri = TP.raycast(O, D, tri9)
    fx = torch.as_tensor(tface, device="cuda")[tri.clamp(min=0).long()]
    t_m = lg.decode_depth(og[..., 0].cuda()).reshape(-1)
    p = O + D * t_m[:, None]
    ix = ((p[:, 0] - lg.mins_f[0]) / 32).long().clamp(0, lg.nx - 1)
    iy = ((p[:, 1] - lg.mins_f[1]) / 32).long().clamp(0, lg.ny - 1)
    iz = ((p[:, 2] - lg.mins_f[2]) / 32).long().clamp(0, lg.nz - 1)
    fg = lg.texmap.fid[iz * lg.stride_z + iy * lg.stride_y + ix].long()
    both = (tri >= 0) & (fg >= 0)
    assert float((fg >= 0).float().mean()) > 0.99
    assert float((fg[both] == fx[both]).float().mean()) > 0.85


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
def test_trainer_smoke_trains_records_and_resumes():
    run = "cya_tex"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-texture", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--obs-texture: surf_src_cannonball texture:" in r.stdout
    assert "-> in_ch 5" in r.stdout
    d = ROOT / "runs" / run
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["obs_texture"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 5, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    re = _train("cya_tex_re", ABS, steps="8192", ckpt=d / "ckpt_final.pt")
    assert re.returncode == 0, re.stdout[-4000:] + re.stderr[-4000:]
    assert "obs_texture=1" in re.stdout
    bad = _train("cya_tex_bad", ABS + ["--obs-texture", "0"], steps="8192",
                 ckpt=d / "ckpt_final.pt")
    assert bad.returncode != 0 and "--obs-texture changes the conv trunk" in bad.stdout + bad.stderr
    off = _train("cya_tex_off", ABS + ["--obs-potential", "norm"])
    assert off.returncode == 0
    oc = json.loads((ROOT / "runs" / "cya_tex_off" / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert "obs_texture" not in oc
    for n in (run, "cya_tex_re", "cya_tex_bad", "cya_tex_off"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


@needs_cuda_caches
def test_face_normal_mode_on_cannonball():
    """--obs-normal: the same face lookup, the face's unit normal turned to the camera and rotated
    into the ego frame (x fwd, y left, z up); triton == torch; the texture mode's colours are
    unchanged by the switch; a down-looking pose reads floors as (0, 0, 1)"""
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    kw = dict(range_units=11500.0, near_range=2000.0, cell=32.0)
    lt = vision.GpuLidar(core, 64, 32, device="cuda", texture=True, **kw)
    ln = vision.GpuLidar(core, 64, 32, device="cuda", texture=lt.texmap, texture_mode="normal", **kw)
    lc = vision.GpuLidar(core, 64, 32, device="cpu", texture=True, texture_mode="normal", **kw)
    a = np.asarray(CPOSES, np.float32)

    def tens(dev):
        return (torch.as_tensor(a[:, 0:3]).to(dev), torch.as_tensor(a[:, 3]).to(dev),
                torch.as_tensor(a[:, 4]).to(dev), torch.as_tensor(a[:, 5]).to(dev))
    ot = lt.render(*tens("cuda")).cpu()
    on = ln.render(*tens("cuda")).cpu()
    oc = lc.render(*tens("cpu"))
    assert torch.equal(on[..., 0], ot[..., 0])
    assert (on[..., 1:] - oc[..., 1:]).abs().max(-1).values.lt(1e-4).float().mean() > 0.999
    nrm = on[..., 1:].norm(dim=-1)
    hit = nrm > 0
    assert hit.float().mean() > 0.99 and torch.allclose(nrm[hit], torch.ones_like(nrm[hit]),
                                                        atol=1e-4)
    # facing the camera: the normal and the ray point against each other (ego x > 0 means the
    # surface faces forward-ward... the facing test is on the world vectors, re-derived here)
    ln._ensure_buffers(len(a))
    ln._dirs_equiangular(len(a), tens("cuda")[1], tens("cuda")[2], float(np.pi / 180))
    yp = tens("cuda")[1].view(-1, 1, 1) * float(np.pi / 180)
    fx, fy, fz = (on[..., 1].cuda(), on[..., 2].cuda(), on[..., 3].cuda())
    wx = fx * torch.cos(yp) - fy * torch.sin(yp)
    wy = fx * torch.sin(yp) + fy * torch.cos(yp)
    facing = (wx * ln._dx + wy * ln._dy + fz * ln._dz)
    assert float((facing[hit.cuda()] <= 1e-4).float().mean()) > 0.999
    # the bottom rows of the steepest down-looking pose are mostly floor or ramp: z up > 0.5
    k = int(np.argmin(a[:, 4]))
    assert float((on[k, -4:, :, 3] > 0.5).float().mean()) > 0.5


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
def test_normal_trainer_smoke():
    run = "cya_nrm"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-normal", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--obs-normal: surf_src_cannonball the hit face's ego-frame unit normal" in r.stdout
    d = ROOT / "runs" / run
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["obs_normal"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 5, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    bad = _train("cya_nrm_bad", ABS + ["--obs-potential", "norm", "--obs-normal", "1",
                                       "--obs-texture", "1"])
    assert bad.returncode != 0 and "pick one" in bad.stdout + bad.stderr
    for n in (run, "cya_nrm_bad"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)


@needs_cuda_caches
def test_slope_mode_on_cannonball():
    """--obs-slope: one channel, acos(the facing normal's z) / pi off the same face lookup;
    triton (a polynomial acos) == torch (exact) to 1e-4; floors ~0 on a down-looking pose; the
    slope is the normal mode's z read as an angle"""
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    kw = dict(range_units=11500.0, near_range=2000.0, cell=32.0)
    lt = vision.GpuLidar(core, 64, 32, device="cuda", texture=True, **kw)
    ls = vision.GpuLidar(core, 64, 32, device="cuda", texture=lt.texmap, texture_mode="slope", **kw)
    ln = vision.GpuLidar(core, 64, 32, device="cuda", texture=lt.texmap, texture_mode="normal", **kw)
    lc = vision.GpuLidar(core, 64, 32, device="cpu", texture=True, texture_mode="slope", **kw)
    assert ls.channels == 2 and ls.tex_channel == 1
    a = np.asarray(CPOSES, np.float32)

    def tens(dev):
        return (torch.as_tensor(a[:, 0:3]).to(dev), torch.as_tensor(a[:, 3]).to(dev),
                torch.as_tensor(a[:, 4]).to(dev), torch.as_tensor(a[:, 5]).to(dev))
    os_ = ls.render(*tens("cuda")).cpu()
    oc = lc.render(*tens("cpu"))
    on = ln.render(*tens("cuda")).cpu()
    assert torch.equal(os_[..., 0], on[..., 0])
    assert (os_[..., 1] - oc[..., 1]).abs().max() < 1e-4
    sl = os_[..., 1]
    ok = sl >= 0
    assert ok.float().mean() > 0.99 and float(sl[ok].max()) <= 1.0
    want = torch.acos(on[..., 3].clamp(-1, 1)) / np.pi
    assert (sl[ok] - want[ok]).abs().max() < 1e-4
    k = int(np.argmin(a[:, 4]))
    assert float((sl[k, -4:] < 0.25).float().mean()) > 0.5          # floors / gentle ramps below


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
def test_slope_trainer_smoke():
    run = "cya_slope"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-slope", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "-> in_ch 3" in r.stdout
    d = ROOT / "runs" / run
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["obs_slope"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 3, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    shutil.rmtree(d, ignore_errors=True)


@needs_cuda_caches
def test_edges_on_cannonball():
    """--obs-edges: the last channel; the fused kernel == the torch reference; the other channels
    are the normal mode's; values in [0, 1]; a sizeable but minority share of edge pixels"""
    from surfgym import SurfCore, default_config
    core = SurfCore(str(CANNONBALL), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    kw = dict(range_units=11500.0, near_range=2000.0, cell=32.0)
    ln = vision.GpuLidar(core, 64, 32, device="cuda", texture=True, texture_mode="normal", **kw)
    le = vision.GpuLidar(core, 64, 32, device="cuda", texture=ln.texmap, texture_mode="normal",
                         edges=True, **kw)
    lc = vision.GpuLidar(core, 64, 32, device="cpu", texture=True, texture_mode="normal",
                         edges=True, **kw)
    assert (le.channels, le.edge_channel) == (5, 4)
    a = np.asarray(CPOSES, np.float32)

    def tens(dev):
        return (torch.as_tensor(a[:, 0:3]).to(dev), torch.as_tensor(a[:, 3]).to(dev),
                torch.as_tensor(a[:, 4]).to(dev), torch.as_tensor(a[:, 5]).to(dev))
    oe = le.render(*tens("cuda")).cpu()
    assert torch.equal(oe[..., :4], ln.render(*tens("cuda")).cpu())
    oc = lc.render(*tens("cpu"))
    assert (oe[..., 4] - oc[..., 4]).abs().max() < 1e-3
    e = oe[..., 4]
    assert float(e.min()) >= 0.0 and float(e.max()) <= 1.0
    assert 0.05 < float((e > 0.5).float().mean()) < 0.6
    with pytest.raises(ValueError):
        vision.GpuLidar(core, 64, 32, device="cuda", edges=True, **kw)    # no face lookup


@needs_run
@pytest.mark.skipif(not FACEID.exists(), reason="needs the baked cannonball face grid (cell 32)")
def test_edges_trainer_smoke():
    run = "cya_edges"
    r = _train(run, ABS + ["--obs-potential", "norm", "--obs-normal", "1", "--obs-edges", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "-> in_ch 6" in r.stdout
    d = ROOT / "runs" / run
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["obs_edges"] == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert tuple(ck["policy"]["conv.0.weight"].shape) == (16, 6, 5, 5)
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"),
                "--map", str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-4000:] + rec.stderr[-4000:]
    alone = _train("cya_edges_bad", ABS + ["--obs-potential", "norm", "--obs-edges", "1"])
    assert alone.returncode != 0 and "--obs-edges reads the face-id grid" in alone.stdout + alone.stderr
    for n in (run, "cya_edges_bad"):
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
