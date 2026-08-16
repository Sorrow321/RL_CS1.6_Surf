"""The surfability mask (--surf-mask): bake, cache, render, policy shape.

CPU only — the bake is numpy and the renderer's torch fallback marches the
same grid the triton kernel does, so everything here runs without a GPU.
What a GPU still owes: that _march_kernel_nz agrees with the fallback pixel
for pixel.

Four things have to hold, and three of them are silent if they break:

  1. the baked values ARE the faces' |n_z| — a floor reads 1, a 45-degree
     ramp 0.707, a wall 0;
  2. the cache invalidates on a content-tag bump, or a semantics change
     serves stale normals next to a fresh SDF;
  3. with the flag off nothing about GpuLidar moves — no grid, no buffer,
     and the depth pixels are the same numbers;
  4. Policy(in_ch=1) is untouched by the feature (identical state_dict) and
     Policy(in_ch=2) is a working network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import surfmask, vision                        # noqa: E402
from train_fast import N_SCALAR, NVEC, Policy               # noqa: E402


# ---------------------------------------------------------------- the bake --
# One synthetic mesh, three faces, each in its own corner of a 128u box so
# no two AABBs overlap: a floor at z=20 (n_z 1), a 45-degree ramp rising in
# y (n_z 0.7071), and a vertical wall (n_z 0).
CELL = 8.0
DIMS = (16, 16, 16)                       # nx, ny, nz -> 128u cube at mins 0
FLOOR = [[8, 8, 20], [56, 8, 20], [56, 56, 20]]
RAMP = [[72, 8, 8], [120, 8, 8], [72, 56, 56]]
WALL = [[8, 72, 8], [8, 120, 8], [8, 72, 56]]


def _mesh_doc(world_tris, brushes=()):
    """The subset of tools/export_map.py's format the bake reads."""
    def part(tris):
        pos = [c for tri in tris for v in tri for c in v]
        return {"positions": pos, "normals": [0.0] * len(pos),
                "indices": list(range(3 * len(tris)))}
    doc = {"map": "synthetic", "world": part(world_tris), "brushes": [],
           "markers": [], "bounds": {"mins": [0, 0, 0], "maxs": [128, 128, 128]}}
    for classname, tris in brushes:
        b = {"classname": classname, "model": 1, "targetname": "",
             "target": "", "skin": 0}
        b.update(part(tris))
        doc["brushes"].append(b)
    return doc


def _write_mesh(path, world_tris, brushes=()):
    path.write_text(json.dumps(_mesh_doc(world_tris, brushes)),
                    encoding="utf-8")
    return path


def _bake(world_tris, brushes=(), tmp_path=None):
    mesh = _write_mesh(tmp_path / "m.mesh.json", world_tris, brushes)
    tris = surfmask.load_solid_tris(mesh)
    return surfmask.rasterize_surfnz(tris, np.zeros(3), CELL, *DIMS)


def _at(grid, x, y, z):
    """Sample by world position (the grid is [nz, ny, nx])."""
    i = [int(v // CELL) for v in (x, y, z)]
    return float(grid[i[2], i[1], i[0]]) / 127.0


def test_baked_values_are_the_faces_normals(tmp_path):
    grid = _bake([FLOOR, RAMP, WALL], tmp_path=tmp_path)
    assert _at(grid, 20, 12, 20) == pytest.approx(1.0, abs=0.05), "floor"
    # the ramp's plane is z == y; sample a voxel centre sitting on it
    assert _at(grid, 100, 20, 20) == pytest.approx(0.7071, abs=0.05), "ramp"
    # 0 is also "no surface" — a wall is unmarked by construction, and the
    # depth channel is what separates a wall pixel from a sky pixel
    assert _at(grid, 12, 76, 12) == 0.0, "wall"
    assert _at(grid, 68, 100, 100) == 0.0, "empty space"


def test_bake_is_a_grid_not_a_smear(tmp_path):
    """A conservative rasterizer still has to be thin: the floor may claim a
    cell either side of its plane, never the whole column."""
    grid = _bake([FLOOR], tmp_path=tmp_path)
    column = grid[:, 1, 2]                              # x=20, y=12, all z
    assert np.count_nonzero(column) <= 3, f"floor smeared over {column}"
    assert column[2] == 127                             # z=20, on the plane


def test_largest_area_triangle_wins(tmp_path):
    """A small ramp inside a big floor's slab must not repaint it — the
    agent lands on the surface, not on the trim bolted to it."""
    small = [[16, 16, 22], [24, 16, 22], [24, 24, 30]]  # n_z 0.707, tiny
    a = _bake([FLOOR, small], tmp_path=tmp_path)
    b = _bake([small, FLOOR], tmp_path=tmp_path)        # order must not matter
    assert _at(a, 20, 20, 20) == pytest.approx(1.0, abs=0.05)
    assert np.array_equal(a, b)


def test_non_solid_brushes_are_ignored(tmp_path):
    """Triggers are draped over the track and physics ignores them; a
    trigger face inside a ramp voxel would overwrite the ramp's slope."""
    both = _bake([RAMP], [("trigger_teleport", [FLOOR]),
                          ("func_wall", [WALL])], tmp_path=tmp_path)
    assert _at(both, 20, 12, 20) == 0.0, "trigger brush was rasterized"
    solid = _bake([RAMP, WALL], tmp_path=tmp_path)
    assert np.array_equal(both, solid)


# ------------------------------------------------------------- the cache ----
class _FakeCore:
    """Enough of SurfCore for the bake: a path to stat and map bounds."""

    def __init__(self, bsp):
        self.bsp_path = str(bsp)

    def map_bounds(self):
        return np.zeros(3), np.full(3, 96.0)


@pytest.fixture
def baked(tmp_path):
    bsp = tmp_path / "synthetic.bsp"
    bsp.write_bytes(b"not a bsp, only stat()ed")
    mesh = _write_mesh(tmp_path / "synthetic.mesh.json", [FLOOR, RAMP, WALL])
    core = _FakeCore(bsp)
    grid, _ = surfmask.build_surfnz(core, CELL, cache_dir=tmp_path,
                                    mesh_path=mesh)
    return core, mesh, tmp_path / f"synthetic.surfnz_{CELL:g}.npz", grid


def test_cache_signature_tracks_the_content_tag(baked, tmp_path, monkeypatch):
    core, mesh, cache_file, honest = baked
    assert cache_file.exists()

    # poison the cache under its own signature: a hit must return THIS
    z = np.load(cache_file, allow_pickle=False)
    poison = np.full_like(honest, 42)
    np.savez_compressed(cache_file, nz=poison, mins=z["mins"], cell=z["cell"],
                        sig=z["sig"])
    hit, _ = surfmask.build_surfnz(core, CELL, cache_dir=tmp_path,
                                   mesh_path=mesh)
    assert np.array_equal(hit, poison), "the cache was not consulted"

    # bump the bake's semantics: the stale grid must be rejected and rebaked
    monkeypatch.setattr(surfmask, "_SURFNZ_SEMANTICS", "zz")
    fresh, _ = surfmask.build_surfnz(core, CELL, cache_dir=tmp_path,
                                     mesh_path=mesh)
    assert np.array_equal(fresh, honest), "content tag did not invalidate"


def test_bake_lands_on_the_vision_grid(baked, tmp_path):
    """Same mins, same dims as build_sdf's grid — the march reads both with
    one voxel index, so a half-cell offset would silently shift the mask."""
    core, _, _, grid = baked
    mins, nx, ny, nz = vision.grid_dims(core, CELL)
    assert grid.shape == (nz, ny, nx)
    _, cmins = surfmask.build_surfnz(core, CELL, cache_dir=tmp_path,
                                     mesh_path=tmp_path / "synthetic.mesh.json")
    assert np.allclose(cmins, mins)


# ------------------------------------------------------------ the renderer --
# A 512x512x256u world with one solid floor slab. The SDF is analytic
# (distance to the slab) so no map, no scipy, no DLL is involved.
RCELL = 16.0
RNX, RNY, RNZ = 32, 32, 16


@pytest.fixture
def fake_grids(monkeypatch):
    sdf = (np.arange(RNZ, dtype=np.float32) * RCELL)[:, None, None] \
        * np.ones((1, RNY, RNX), dtype=np.float32)
    mins = np.zeros(3, dtype=np.float32)
    snz = np.zeros((RNZ, RNY, RNX), dtype=np.int8)
    snz[0] = 127                                        # the floor is flat
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None: (sdf, mins, cell))
    monkeypatch.setattr(surfmask, "build_surfnz",
                        lambda core, cell, cache_dir=None, mesh_path=None:
                        (snz, mins))
    return sdf, snz


def _poses():
    #                     x     y      z (eye = z + 17)   yaw   pitch  duck
    a = np.array([[200.0, 200.0, 183.0, 0.0, -90.0, 0],   # straight down
                  [200.0, 200.0, 183.0, 0.0, 80.0, 0],    # up into the sky
                  [200.0, 200.0, 183.0, 45.0, -30.0, 0]], dtype=np.float32)
    return (torch.as_tensor(a[:, 0:3]), torch.as_tensor(a[:, 3]),
            torch.as_tensor(a[:, 4]), torch.as_tensor(a[:, 5]))


def test_flag_off_changes_nothing(fake_grids):
    off = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu")
    on = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu", surf_mask=True)
    tensors = {k for k, v in vars(off).items() if torch.is_tensor(v)}
    assert tensors == {"sdf_flat", "mins", "yoff", "poff"}
    assert {k for k, v in vars(on).items() if torch.is_tensor(v)} - tensors \
        == {"snz_flat"}, "the mask grid leaked into the depth-only lidar"
    assert (off.surf_mask, off.channels) == (False, 1)
    assert (on.surf_mask, on.channels) == (True, 2)

    d = off.render(*_poses())
    both = on.render(*_poses())
    assert d.shape == (3, 4, 8) and both.shape == (3, 4, 8, 2)
    assert torch.equal(both[..., 0], d), "the mask perturbed the depth channel"


def test_mask_channel_reads_the_hit_surface(fake_grids):
    lid = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu", surf_mask=True)
    out = lid.render(*_poses())
    down, up = out[0], out[1]
    assert down[..., 0].max() < 0.2, "the floor should be close below"
    assert torch.all(down[..., 1] == 1.0), "flat floor must read |n_z| = 1"
    assert torch.all(up[..., 0] > 0.9), "the sky pixels should run to range"
    assert torch.all(up[..., 1] == 0.0), "open sky must read 0"


def test_flat_image_is_channel_fastest(fake_grids):
    """The trainer flattens the render into the obs row and Policy restrides
    it back — that only works if the channel is the fastest axis."""
    lid = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu", surf_mask=True)
    out = lid.render(*_poses())
    flat = out.reshape(3, -1)
    assert torch.equal(flat.reshape(-1, 4, 8, 2), out)
    im = flat.reshape(-1, 4, 8, 2).permute(0, 3, 1, 2)
    assert im.stride()[1] == 1, "channel stride must be 1 (NHWC)"


# ------------------------------------------------- against the real tracer --
SKI = ROOT / "maps" / "surf_ski_2.bsp"
SKI_SDF = ROOT / "maps" / "surf_ski_2.sdf_16.npz"
SKI_MESH = ROOT / "viewer" / "assets" / "surf_ski_2.mesh.json"
TW, TH, TRANGE = 16, 8, 2000.0


@pytest.mark.skipif(not (SKI_SDF.exists() and SKI_MESH.exists()),
                    reason="needs the tracked surf_ski_2 SDF cache + mesh")
def test_mask_matches_the_exact_trace_on_a_real_map(tmp_path, monkeypatch):
    """The end-to-end claim, on real geometry: what the mask channel says a
    pixel is standing on is what the C tracer's plane normal says.

    Everything upstream can be individually right and this still fail — the
    bake writing into voxels the march never stops in would read 0 ("wall")
    on open floor, and nothing else here would notice.

    Judged where the bake HAS data, because on surf_ski_2 it sometimes
    cannot: ~20% of the pixels these poses see are solid world brushes with
    no face in the BSP at all (verified — the nearest face plane to those
    hits is 200u away), so no face-based bake can know their normal, and
    they render 0 = "wall". The trained map is better behaved: the same
    measurement on surf_src_cannonball's ramp views is 98.7% band agreement
    overall, 99.8% on the surfable band.
    """
    from surfgym import SurfCore, default_config
    from surfgym.rewards import ramp_spawn_pool

    # Both grids come from outside maps/: the mask is baked into tmp_path,
    # and the SDF is READ from the tracked cache instead of rebuilt. A bare
    # build_sdf here would overwrite that tracked file on every run — the
    # .bsp's mtime changes on any fresh checkout, so its signature never
    # matches and the 70M-voxel field rebuilds itself into the worktree.
    real = surfmask.build_surfnz
    monkeypatch.setattr(
        surfmask, "build_surfnz",
        lambda core, cell, cache_dir=None, mesh_path=None:
        real(core, cell, cache_dir=tmp_path, mesh_path=mesh_path))
    cached = np.load(SKI_SDF, allow_pickle=False)
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None:
                        (cached["sdf"], cached["mins"], float(cached["cell"])))
    core = SurfCore(str(SKI), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    lid = vision.GpuLidar(core, TW, TH, range_units=TRANGE, cell=16.0,
                          device="cpu", surf_mask=True)

    # hovering over ramp faces, looking down: the view the objective is about
    pool = ramp_spawn_pool(core)
    sel = np.random.default_rng(1).choice(len(pool), size=6, replace=False)
    origin = np.ascontiguousarray(pool["origin"][sel]).astype(np.float64)
    origin[:, 2] += 200.0
    yaw = pool["yaw"][sel].astype(np.float64)
    pitch = np.full(len(sel), -45.0)
    out = lid.render(torch.as_tensor(origin, dtype=torch.float32),
                     torch.as_tensor(yaw, dtype=torch.float32),
                     torch.as_tensor(pitch, dtype=torch.float32),
                     torch.zeros(len(sel)))
    depth, mask = out[..., 0].numpy(), out[..., 1].numpy()

    d2r = np.pi / 180.0
    yoff = (120.0 * (0.5 - (np.arange(TW) + 0.5) / TW)) * d2r
    poff = (90.0 * (0.5 - (np.arange(TH) + 0.5) / TH)) * d2r
    truth = np.zeros((len(sel), TH, TW))
    hit = np.zeros((len(sel), TH, TW), dtype=bool)
    for i in range(len(sel)):
        eye = origin[i] + [0.0, 0.0, 17.0]
        for r in range(TH):
            pt = pitch[i] * d2r + poff[r]
            for c in range(TW):
                yw = yaw[i] * d2r + yoff[c]
                d = np.array([np.cos(pt) * np.cos(yw), np.cos(pt) * np.sin(yw),
                              np.sin(pt)])
                tr = core.trace(eye, eye + d * TRANGE, hull=2)
                if tr.fraction < 1.0 and not tr.startsolid:
                    hit[i, r, c] = True
                    truth[i, r, c] = abs(tr.normal[2])

    both = hit & (depth < 0.995)
    assert both.sum() > 300, "too few compared pixels to mean anything"
    err = np.abs(mask[both] - truth[both])
    assert (err <= 0.05).mean() > 0.75, (
        f"only {(err <= 0.05).mean():.1%} of ALL pixels within 0.05 — under "
        f"the floor the map's faceless brushes alone explain")

    # the decision the mask exists to support: wall / steep / surfable /
    # walkable / flat, split at the physics threshold (0.7, src/pm.c)
    def band(v):
        return np.digitize(v, [0.1, 0.3, 0.7, 0.95])
    bt, bm = band(truth[both]), band(mask[both])
    have = mask[both] > 0                      # the bake knows this surface
    assert have.sum() > 250, "almost nothing was baked — the grid is empty"
    assert (err[have] <= 0.05).mean() > 0.90, \
        f"where baked, only {(err[have] <= 0.05).mean():.1%} within 0.05"
    assert (bt[have] == bm[have]).mean() > 0.90, \
        f"band agreement {(bt[have] == bm[have]).mean():.1%}"
    ramp = have & (bt == 2)
    assert ramp.sum() > 30, "these poses see no baked ramp — vacuous"
    assert (bm[ramp] == 2).mean() > 0.85, \
        f"ramps misread {(bm[ramp] != 2).mean():.1%} of the time"


# --------------------------------------------------------------- the policy --
W, H = 32, 16
KEYS = ["conv.0.weight", "conv.0.bias", "conv.2.weight", "conv.2.bias",
        "conv.4.weight", "conv.4.bias", "conv.8.weight", "conv.8.bias",
        "pi.0.weight", "pi.0.bias", "pi.2.weight", "pi.2.bias",
        "vf.0.weight", "vf.0.bias", "vf.2.weight", "vf.2.bias",
        "action_head.weight", "action_head.bias",
        "value_head.weight", "value_head.bias"]


def test_depth_only_policy_is_untouched():
    """Every existing checkpoint has to keep loading: same keys, same
    shapes, conv1 still one input channel."""
    p = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24)
    sd = p.state_dict()
    assert list(sd) == KEYS
    assert tuple(sd["conv.0.weight"].shape) == (16, 1, 5, 5)
    assert p.in_ch == 1


def test_two_channel_policy_runs():
    torch.manual_seed(0)
    p = Policy(N_SCALAR + W * H * 2, W, H, emb=32, hidden=24, in_ch=2).eval()
    sd = p.state_dict()
    assert list(sd) == KEYS, "the mask must not add or rename parameters"
    assert tuple(sd["conv.0.weight"].shape) == (16, 2, 5, 5)
    one = Policy(N_SCALAR + W * H, W, H, emb=32, hidden=24).state_dict()
    assert all(sd[k].shape == one[k].shape for k in KEYS
               if k != "conv.0.weight"), "the mask resized the trunk"

    obs = torch.rand(6, N_SCALAR + W * H * 2)
    with torch.no_grad():
        logits, value = p(obs)
        split = p.forward_split(obs[:, :N_SCALAR], obs[:, N_SCALAR:])
    assert logits.shape == (6, sum(NVEC)) and value.shape == (6,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert torch.equal(logits, split[0]) and torch.equal(value, split[1])


def test_two_channel_image_is_a_free_restride():
    """The image slice must reach the channels_last trunk without a copy —
    the whole reason the renderer interleaves instead of stacking planes."""
    img = torch.rand(6, H * W * 2)
    im = img.reshape(-1, H, W, 2).permute(0, 3, 1, 2)
    assert im.stride()[1] == 1, "channel stride must be 1 (NHWC)"
    assert im.data_ptr() == img.data_ptr(), "the restride copied"
    assert torch.equal(im[:, 0], img.reshape(6, H, W, 2)[..., 0])
