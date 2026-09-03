"""--normals: the full-normal bake, the 4-channel march, the ego-frame
convention, the --lidar-hfov/--lidar-vfov plumbing and the goal-ball
wrapper over a multi-channel lidar.

CPU torch throughout (the renderer's torch fallback marches the same grid
the triton kernel does), plus one CUDA-only test that holds the triton
kernel to the fallback pixel for pixel.

The convention under test, stated once (vision.py module docstring):
the normal faces the RAY, and is expressed in the player's ego frame
rotated by the view yaw only - x forward, y left, z up. So a floor reads
(0, 0, 1) whatever the gaze, the wall ahead (-1, 0, 0), a wall on the
player's right (0, 1, 0) (its normal points left, at the player), a ceiling
(0, 0, -1), and a miss (0, 0, 0).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import surfmask, vision                        # noqa: E402
from surfgym.goalball import GoalBallLidar                  # noqa: E402


# ---------------------------------------------------------------- the bake --
CELL = 8.0
DIMS = (16, 16, 16)                       # nx, ny, nz -> 128u cube at mins 0
FLOOR = [[8, 8, 20], [56, 8, 20], [56, 56, 20]]
RAMP = [[72, 8, 8], [120, 8, 8], [72, 56, 56]]       # plane z == y
WALL = [[8, 72, 8], [8, 120, 8], [8, 72, 56]]        # plane x == 8


def _mesh_doc(world_tris):
    pos = [c for tri in world_tris for v in tri for c in v]
    part = {"positions": pos, "normals": [0.0] * len(pos),
            "indices": list(range(3 * len(world_tris)))}
    return {"map": "synthetic", "world": part, "brushes": [], "markers": [],
            "bounds": {"mins": [0, 0, 0], "maxs": [128, 128, 128]}}


def _write_mesh(path, world_tris):
    path.write_text(json.dumps(_mesh_doc(world_tris)), encoding="utf-8")
    return path


def _at(grid, x, y, z):
    i = [int(v // CELL) for v in (x, y, z)]
    return grid[i[2], i[1], i[0]].astype(np.float64) / 127.0


def test_canonical_sign_is_the_upper_hemisphere():
    n = surfmask.canonical_normals([[0, 0, -1], [0, -1, 0], [-1, 0, 0],
                                    [0.6, 0, -0.8], [0, 0.6, 0.8]])
    assert np.allclose(n, [[0, 0, 1], [0, 1, 0], [1, 0, 0],
                           [-0.6, 0, 0.8], [0, 0.6, 0.8]])


def test_baked_values_are_the_faces_canonical_normals():
    tris = np.asarray([FLOOR, RAMP, WALL], dtype=np.float64)
    grid = surfmask.rasterize_surfnormal(tris, np.zeros(3), CELL, *DIMS)
    assert grid.shape == (16, 16, 16, 3) and grid.dtype == np.int8
    assert np.allclose(_at(grid, 20, 12, 20), [0, 0, 1], atol=0.01), "floor"
    assert np.allclose(_at(grid, 100, 20, 20), [0, -0.7071, 0.7071],
                       atol=0.01), "ramp (z == y rises in y: n_y < 0)"
    assert np.allclose(_at(grid, 12, 76, 12), [1, 0, 0], atol=0.01), \
        "wall: canonical sign is n_x > 0 when n_z == n_y == 0"
    assert np.array_equal(_at(grid, 68, 100, 100), [0, 0, 0]), "empty"
    # every claimed voxel holds a unit vector (to int8 rounding)
    claimed = grid.any(axis=-1)
    lens = np.linalg.norm(grid[claimed].astype(np.float64) / 127.0, axis=1)
    assert claimed.sum() > 50 and np.abs(lens - 1.0).max() < 0.01


def test_z_byte_is_the_surfnz_grid():
    """Same claim rules by construction: the surf-mask bake must be
    recoverable from this one, or the two channels would disagree about
    what a pixel is standing on."""
    tris = np.asarray([FLOOR, RAMP, WALL], dtype=np.float64)
    full = surfmask.rasterize_surfnormal(tris, np.zeros(3), CELL, *DIMS)
    nz = surfmask.rasterize_surfnz(tris, np.zeros(3), CELL, *DIMS)
    assert np.array_equal(full[..., 2], nz)
    small = [[16, 16, 22], [24, 16, 22], [24, 24, 30]]  # trim on the floor
    a = surfmask.rasterize_surfnormal(np.asarray([FLOOR, small], float),
                                      np.zeros(3), CELL, *DIMS)
    b = surfmask.rasterize_surfnormal(np.asarray([small, FLOOR], float),
                                      np.zeros(3), CELL, *DIMS)
    assert np.array_equal(a, b), "triangle order leaked into the bake"
    assert np.allclose(_at(a, 20, 20, 20), [0, 0, 1], atol=0.01), \
        "the largest-area face must win"


class _FakeCore:
    def __init__(self, bsp):
        self.bsp_path = str(bsp)

    def map_bounds(self):
        return np.zeros(3), np.full(3, 96.0)


def test_cache_tracks_mesh_content_and_semantics(tmp_path, monkeypatch):
    bsp = tmp_path / "synthetic.bsp"
    bsp.write_bytes(b"not a bsp, only stat()ed")
    mesh = _write_mesh(tmp_path / "synthetic.mesh.json", [FLOOR, RAMP, WALL])
    core = _FakeCore(bsp)
    honest, mins = surfmask.build_surfnormal(core, CELL, cache_dir=tmp_path,
                                             mesh_path=mesh)
    cache_file = tmp_path / f"synthetic.surfn_{CELL:g}.npz"
    assert cache_file.exists()
    gmins, nx, ny, nz = vision.grid_dims(core, CELL)
    assert honest.shape == (nz, ny, nx, 3) and np.allclose(mins, gmins)

    z = np.load(cache_file, allow_pickle=False)
    poison = np.full_like(honest, 42)
    np.savez_compressed(cache_file, n=poison, mins=z["mins"], cell=z["cell"],
                        sig=z["sig"])
    hit, _ = surfmask.build_surfnormal(core, CELL, cache_dir=tmp_path,
                                       mesh_path=mesh)
    assert np.array_equal(hit, poison), "the cache was not consulted"

    # a re-exported mesh (same size and name, different content) must miss
    _write_mesh(mesh, [FLOOR, WALL, RAMP])
    fresh, _ = surfmask.build_surfnormal(core, CELL, cache_dir=tmp_path,
                                         mesh_path=mesh)
    assert np.array_equal(fresh, honest), "mesh content did not invalidate"

    np.savez_compressed(cache_file, n=poison, mins=z["mins"], cell=z["cell"],
                        sig=np.load(cache_file, allow_pickle=False)["sig"])
    monkeypatch.setattr(surfmask, "_SURFN_SEMANTICS", "zz")
    fresh, _ = surfmask.build_surfnormal(core, CELL, cache_dir=tmp_path,
                                         mesh_path=mesh)
    assert np.array_equal(fresh, honest), "content tag did not invalidate"


# ------------------------------------------------------------ the renderer --
# A 512x512x256u world of axis-aligned half-spaces: a floor (voxel layer 0
# solid), optionally a wall (x >= WALL_X solid) and a ceiling (z >= CEIL_Z
# solid). The SDF is what the EDT would produce for such a union - distance
# to the nearest solid voxel CENTRE, so the march never stops in an air
# voxel - and the normal grid marks the solid voxels with each face's
# canonical normal. No map, no scipy, no DLL.
RCELL = 16.0
RNX, RNY, RNZ = 32, 32, 16
WALL_X = 24                                # voxel index: x >= 384 u solid
CEIL_Z = 15                                # voxel index: z >= 240 u solid
                                           # (the eye sits at z = 200)


def _scene(wall=False, ceil=False):
    k = np.arange(RNZ, dtype=np.float32)[:, None, None]
    i = np.arange(RNX, dtype=np.float32)[None, None, :]
    sdf = np.broadcast_to(k * RCELL, (RNZ, RNY, RNX)).copy()
    n = np.zeros((RNZ, RNY, RNX, 3), dtype=np.int8)
    n[0] = (0, 0, 127)
    if wall:
        sdf = np.minimum(sdf, np.broadcast_to(np.maximum(WALL_X - i, 0) * RCELL,
                                              sdf.shape))
        n[:, :, WALL_X:] = (127, 0, 0)
    if ceil:
        sdf = np.minimum(sdf, np.broadcast_to(np.maximum(CEIL_Z - k, 0) * RCELL,
                                              sdf.shape))
        n[CEIL_Z:] = (0, 0, 127)
    return sdf.astype(np.float32), n


@pytest.fixture
def scene(monkeypatch):
    def install(wall=False, ceil=False):
        sdf, n = _scene(wall, ceil)
        mins = np.zeros(3, dtype=np.float32)
        monkeypatch.setattr(vision, "build_sdf",
                            lambda core, cell, cache_dir=None: (sdf, mins, cell))
        monkeypatch.setattr(surfmask, "build_surfnormal",
                            lambda core, cell, cache_dir=None, mesh_path=None:
                            (n, mins))
        return sdf, n
    return install


def _pose(x=200.0, y=200.0, z=183.0, yaw=0.0, pitch=0.0, duck=0):
    a = np.array([[x, y, z, yaw, pitch, duck]], dtype=np.float32)
    return (torch.as_tensor(a[:, 0:3]), torch.as_tensor(a[:, 3]),
            torch.as_tensor(a[:, 4]), torch.as_tensor(a[:, 5]))


def _poses(*ps):
    parts = [_pose(**p) for p in ps]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(4))


def _legacy_render_torch(lid, origin, yaw_deg, pitch_deg, ducked):
    """GpuLidar._render_torch's depth-only path as it was before --normals,
    verbatim (the march, the early exit, the encoding) - the reference the
    flag-off render is held bit-identical to."""
    N = origin.shape[0]
    lid._ensure_buffers(N)
    d2r = np.pi / 180.0
    ex = origin[:, 0].view(N, 1, 1)
    ey = origin[:, 1].view(N, 1, 1)
    ez = (origin[:, 2] + torch.where(ducked.bool(), 12.0, 17.0)).view(N, 1, 1)
    lid._dirs_equiangular(N, yaw_deg, pitch_deg, d2r)
    t, alive = lid._t, lid._alive
    t.zero_()
    alive.fill_(True)
    hit_eps = 0.6 * lid.cell
    min_step = 0.3 * lid.cell
    inv_cell = 1.0 / lid.cell
    mx, my, mz = lid.mins[0], lid.mins[1], lid.mins[2]
    for it in range(lid.max_steps):
        ix = ((ex + lid._dx * t - mx) * inv_cell).long().clamp_(0, lid.nx - 1)
        iy = ((ey + lid._dy * t - my) * inv_cell).long().clamp_(0, lid.ny - 1)
        iz = ((ez + lid._dz * t - mz) * inv_cell).long().clamp_(0, lid.nz - 1)
        d = lid.sdf_flat[(iz * lid.stride_z + iy * lid.stride_y + ix)
                         .reshape(-1)].reshape(N, lid.H, lid.W).float()
        alive &= (d > hit_eps) & (t < lid.range)
        t.add_(torch.clamp(d * 0.9, min=min_step) * alive)
        if (it & 7) == 7 and not alive.any():
            break
    t = torch.clamp(t, max=lid.range)
    return (torch.clamp(t, max=lid.near) / lid.near
            + 0.25 * (1.0 - torch.exp(-torch.clamp(t - lid.near, min=0.0)
                                      / 2500.0)))


VIEWS = ({"pitch": -90.0}, {"pitch": 80.0}, {"yaw": 45.0, "pitch": -30.0},
         {"yaw": 200.0, "pitch": -10.0, "duck": 1}, {"yaw": 90.0})


def test_flag_off_is_bit_identical_and_allocation_free(scene):
    scene(wall=True, ceil=True)
    off = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu")
    on = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu", normals=True)
    tensors = {k for k, v in vars(off).items() if torch.is_tensor(v)}
    assert tensors == {"sdf_flat", "mins", "yoff", "poff"}
    assert {k for k, v in vars(on).items() if torch.is_tensor(v)} - tensors \
        == {"snrm_flat"}, "the normal grid leaked into the depth-only lidar"
    assert (off.normals, off.channels) == (False, 1)
    assert (on.normals, on.channels) == (True, 4)

    p = _poses(*VIEWS)
    d = off.render(*p)
    assert d.shape == (len(VIEWS), 4, 8)
    assert torch.equal(d, _legacy_render_torch(off, *p)), \
        "the flag-off render drifted from the pre-normals code"
    both = on.render(*p)
    assert both.shape == (len(VIEWS), 4, 8, 4)
    assert torch.equal(both[..., 0], d), "normals perturbed the depth channel"
    flat = both.reshape(len(VIEWS), -1)
    assert torch.equal(flat.reshape(-1, 4, 8, 4), both)   # channel-fastest


def test_exclusive_with_surf_mask_and_pinhole(scene):
    scene()
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu", normals=True,
                        surf_mask=True)
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu", normals=True,
                        pinhole=True)


# A ray that exhausts the march's step budget ends in AIR with depth < 1 and
# normal 0 - the march's own (pre-existing) behaviour, and a grazing ray over
# this scene's floor does it at the default 64 steps. The tests that equate
# "depth < 1" with "hit" give the march enough steps to settle every ray.
STEPS = 256


def test_unit_length_on_a_hit_and_zero_on_a_miss(scene):
    scene()
    lid = vision.GpuLidar(None, 16, 8, cell=RCELL, device="cpu", normals=True,
                          max_steps=STEPS)
    out = lid.render(*_poses({"pitch": -90.0}, {"pitch": 80.0},
                             {"yaw": 30.0, "pitch": -20.0}))
    depth, nrm = out[..., 0], out[..., 1:]
    sky = 1.0                                   # near == range: legacy encoding
    hit = depth < sky
    assert hit[0].all() and not hit[1].any(), "poses do not split hit/miss"
    assert 0.2 < hit[2].float().mean() < 0.8, "the third pose is degenerate"
    length = torch.linalg.norm(nrm, dim=-1)
    assert torch.allclose(length[hit], torch.ones_like(length[hit]), atol=1e-5)
    assert torch.all(length[~hit] == 0.0)


def test_ego_frame_convention(scene):
    scene(wall=True, ceil=True)
    lid = vision.GpuLidar(None, 16, 8, cell=RCELL, device="cpu", normals=True,
                          max_steps=STEPS)
    # floor: (0, 0, 1) from any yaw, at any duck
    out = lid.render(*_poses({"pitch": -90.0}, {"yaw": 137.0, "pitch": -90.0},
                             {"yaw": 270.0, "pitch": -90.0, "duck": 1}))
    assert torch.allclose(out[..., 1:], torch.tensor([0.0, 0.0, 1.0])
                          .expand_as(out[..., 1:]), atol=1e-5)
    # the wall ahead (x = 384, player at x = 200 facing +x): (-1, 0, 0) on
    # the centre rows; the ceiling seen from below: (0, 0, -1)
    out = lid.render(*_poses({"yaw": 0.0}, {"pitch": 89.0}))
    centre = out[0, 3:5, 6:10, 1:]
    assert torch.allclose(centre, torch.tensor([-1.0, 0.0, 0.0])
                          .expand_as(centre), atol=1e-5)
    assert torch.allclose(out[1, ..., 1:], torch.tensor([0.0, 0.0, -1.0])
                          .expand_as(out[1, ..., 1:]), atol=1e-5)
    # facing +y the same wall is on the player's RIGHT: its normal points
    # left, at the player -> (0, +1, 0) on the right-hand columns
    out = lid.render(*_poses({"yaw": 90.0}))
    right = out[0, 3:5, 14:16, 1:]
    assert out[0, 3:5, 14:16, 0].max() < 0.2, "the right edge must see the wall"
    assert torch.allclose(right, torch.tensor([0.0, 1.0, 0.0])
                          .expand_as(right), atol=1e-5)
    # ...and on the player's LEFT when facing -y: (0, -1, 0)
    out = lid.render(*_poses({"yaw": -90.0}))
    left = out[0, 3:5, 0:2, 1:]
    assert torch.allclose(left, torch.tensor([0.0, -1.0, 0.0])
                          .expand_as(left), atol=1e-5)


def test_fov_flags_move_the_pixel_grid(scene):
    scene()
    d2r = math.pi / 180.0
    dflt = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu")
    assert (dflt.hfov_deg, dflt.vfov_deg) == (120.0, 90.0)
    assert abs(float(dflt.yoff[0]) - 120.0 * (0.5 - 0.5 / 8) * d2r) < 1e-6
    assert abs(float(dflt.poff[0]) - 90.0 * (0.5 - 0.5 / 4) * d2r) < 1e-6
    lid = vision.GpuLidar(None, 8, 4, cell=RCELL, device="cpu",
                          hfov_deg=90.0, vfov_deg=60.0)
    assert abs(float(lid.yoff[0]) - 90.0 * (0.5 - 0.5 / 8) * d2r) < 1e-6
    assert abs(float(lid.yoff[-1]) + 90.0 * (0.5 - 0.5 / 8) * d2r) < 1e-6
    assert abs(float(lid.poff[0]) - 60.0 * (0.5 - 0.5 / 4) * d2r) < 1e-6
    assert abs(float(lid.poff[-1]) + 60.0 * (0.5 - 0.5 / 4) * d2r) < 1e-6
    # the ball wrapper backs the fov out of exactly these offsets
    gb = GoalBallLidar(lid, 1, views=1)
    assert abs(math.degrees(gb.hfov) - 90.0) < 1e-4
    assert abs(math.degrees(gb.vfov) - 60.0) < 1e-4
    # a narrower vfov looking down at -30 sees LESS sky than the default
    p = _pose(pitch=-30.0)
    assert float((lid.render(*p) >= 1.0).float().mean()) \
        < float((dflt.render(*p) >= 1.0).float().mean())


# --------------------------------------------------- the goal-ball wrapper --
W, H = 64, 32


class _Stub4:
    """A 4-channel lidar: constant, distinct planes so every channel can be
    told apart after the wrapper reassembles them."""

    def __init__(self):
        d2r = math.pi / 180.0
        self.W, self.H, self.channels, self.pinhole = W, H, 4, False
        self.device = torch.device("cpu")
        self.near, self.range = 2000.0, 11500.0
        self.yoff = torch.as_tensor(
            (120.0 * (0.5 - (np.arange(W) + 0.5) / W)) * d2r, dtype=torch.float32)
        self.poff = torch.as_tensor(
            (90.0 * (0.5 - (np.arange(H) + 0.5) / H)) * d2r, dtype=torch.float32)

    def render(self, origin, yaw_deg, pitch_deg, ducked):
        n = origin.shape[0]
        return torch.tensor([0.5, -1.0, 0.25, 1.0]).view(1, 1, 1, 4) \
            .expand(n, H, W, 4).contiguous()


def test_goal_ball_appends_views_after_all_lidar_channels():
    for views in (1, 4):
        gb = GoalBallLidar(_Stub4(), 2, views=views)
        assert gb.channels == 4 + views
        gb.set_goals([0, 1], [[3000.0, 0.0, 17.0], [3000.0, 0.0, 17.0]])
        o = torch.zeros((2, 3))
        img = gb.render(o, torch.zeros(2), torch.zeros(2),
                        torch.zeros(2, dtype=torch.int32))
        assert img.shape == (2, H, W, 4 + views)
        assert torch.all(img[..., 0] == 0.5), "channel 0 must stay the depth"
        assert torch.all(img[..., 1] == -1.0) and torch.all(img[..., 2] == 0.25) \
            and torch.all(img[..., 3] == 1.0), "the lidar's channels moved"
        front = img[0, :, :, 4]
        assert (front[H // 2 - 1: H // 2 + 1, W // 2 - 1: W // 2 + 1] > 0).all()
        assert img.reshape(2, -1).reshape(-1, H, W, 4 + views).equal(img)
    # (the 1-channel case, (N, H, W, 1 + views), is tests/python/test_goalball.py)


# --------------------------------------------------- against the real tracer --
SKI = ROOT / "maps" / "surf_ski_2.bsp"
SKI_SDF = ROOT / "maps" / "surf_ski_2.sdf_16.npz"
SKI_MESH = ROOT / "viewer" / "assets" / "surf_ski_2.mesh.json"
TW, TH, TRANGE = 16, 8, 2000.0


@pytest.mark.skipif(not (SKI_SDF.exists() and SKI_MESH.exists()),
                    reason="needs the tracked surf_ski_2 SDF cache + mesh")
def test_ego_normals_match_the_exact_trace_on_a_real_map(tmp_path, monkeypatch):
    """The end-to-end claim on real geometry: the ego-frame normal a pixel
    carries is the C tracer's plane normal at that hit, flipped to face the
    ray and rotated by the view yaw - sign included, which is the half the
    surf mask could not define.

    Judged where the bake HAS data, for the reason test_surfmask.py gives:
    a fifth of what these poses see on surf_ski_2 is faceless solid (clip
    brushes, faceless world brushes), which no mesh bake can know and which
    renders (0, 0, 0). Same measurement on surf_src_cannonball's ramp views:
    78.6% baked, and where baked 96.9% within 10 degrees, median 0.05.
    """
    from surfgym import SurfCore, default_config
    from surfgym.rewards import ramp_spawn_pool

    real = surfmask.build_surfnormal
    monkeypatch.setattr(
        surfmask, "build_surfnormal",
        lambda core, cell, cache_dir=None, mesh_path=None:
        real(core, cell, cache_dir=tmp_path, mesh_path=mesh_path))
    cached = np.load(SKI_SDF, allow_pickle=False)
    monkeypatch.setattr(vision, "build_sdf",
                        lambda core, cell, cache_dir=None:
                        (cached["sdf"], cached["mins"], float(cached["cell"])))
    core = SurfCore(str(SKI), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    lid = vision.GpuLidar(core, TW, TH, range_units=TRANGE, cell=16.0,
                          device="cpu", normals=True)

    pool = ramp_spawn_pool(core)
    sel = np.random.default_rng(1).choice(len(pool), size=6, replace=False)
    origin = np.ascontiguousarray(pool["origin"][sel]).astype(np.float64)
    origin[:, 2] += 200.0
    yaw = pool["yaw"][sel].astype(np.float64)
    pitch = np.full(len(sel), -45.0)
    out = lid.render(torch.as_tensor(origin, dtype=torch.float32),
                     torch.as_tensor(yaw, dtype=torch.float32),
                     torch.as_tensor(pitch, dtype=torch.float32),
                     torch.zeros(len(sel))).numpy()
    depth, nrm = out[..., 0], out[..., 1:]

    d2r = np.pi / 180.0
    yoff = (120.0 * (0.5 - (np.arange(TW) + 0.5) / TW)) * d2r
    poff = (90.0 * (0.5 - (np.arange(TH) + 0.5) / TH)) * d2r
    truth = np.zeros((len(sel), TH, TW, 3))
    hit = np.zeros((len(sel), TH, TW), dtype=bool)
    for i in range(len(sel)):
        eye = origin[i] + [0.0, 0.0, 17.0]
        cy, sy = np.cos(yaw[i] * d2r), np.sin(yaw[i] * d2r)
        for r in range(TH):
            pt = pitch[i] * d2r + poff[r]
            for c in range(TW):
                yw = yaw[i] * d2r + yoff[c]
                d = np.array([np.cos(pt) * np.cos(yw), np.cos(pt) * np.sin(yw),
                              np.sin(pt)])
                tr = core.trace(eye, eye + d * TRANGE, hull=2)
                if tr.fraction < 1.0 and not tr.startsolid:
                    hit[i, r, c] = True
                    n = np.asarray(tr.normal, dtype=np.float64)
                    if n @ d > 0:                      # face the ray
                        n = -n
                    truth[i, r, c] = [n[0] * cy + n[1] * sy,   # forward
                                      n[1] * cy - n[0] * sy,   # left
                                      n[2]]                    # up

    both = hit & (depth < 0.995)
    assert both.sum() > 300, "too few compared pixels to mean anything"
    have = np.linalg.norm(nrm[both], axis=-1) > 0
    assert have.sum() > 200, "almost nothing was baked - the grid is empty"
    cosang = np.clip((nrm[both] * truth[both]).sum(-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))[have]
    assert (ang < 10.0).mean() > 0.90, \
        f"where baked, only {(ang < 10.0).mean():.1%} within 10 degrees"
    assert (cosang[have] > 0).mean() > 0.95, \
        f"the sign disagrees with the tracer on {(cosang[have] <= 0).mean():.1%}"


# --------------------------------------------------- triton vs the fallback --
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not vision.HAVE_TRITON, reason="needs triton")
def test_triton_kernel_agrees_with_the_fallback(scene):
    scene(wall=True, ceil=True)
    cpu = vision.GpuLidar(None, 32, 16, cell=RCELL, device="cpu", normals=True,
                          max_steps=STEPS)
    gpu = vision.GpuLidar(None, 32, 16, cell=RCELL, device="cuda", normals=True,
                          max_steps=STEPS)
    p = _poses(*VIEWS, {"yaw": 90.0, "pitch": -5.0}, {"yaw": -90.0, "pitch": 5.0})
    ref = cpu.render(*p)
    out = gpu.render(*[t.cuda() for t in p]).cpu()
    assert out.shape == ref.shape == (len(VIEWS) + 2, 16, 32, 4)
    # depth: the same march on the same grid (cos/sin differ in the last ulp
    # between the two, which can move a voxel boundary by one ray)
    d_ok = torch.isclose(out[..., 0], ref[..., 0], atol=1e-4)
    assert d_ok.float().mean() > 0.995, f"depth agreement {d_ok.float().mean():.4f}"
    n_ok = torch.isclose(out[..., 1:], ref[..., 1:], atol=1e-3).all(dim=-1)
    assert n_ok.float().mean() > 0.995, f"normal agreement {n_ok.float().mean():.4f}"
    length = torch.linalg.norm(out[..., 1:], dim=-1)
    hit = out[..., 0] < 1.0
    assert torch.allclose(length[hit], torch.ones_like(length[hit]), atol=1e-5)
    assert torch.all(length[~hit] == 0.0)
