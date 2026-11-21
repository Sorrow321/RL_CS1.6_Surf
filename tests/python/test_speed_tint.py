"""--obs-speed-tint (the user, 2026-11-21): the depth render coloured by the player's SPEED, a
scalar - more red = faster, more blue = slower. The channel is the red part: |v| / 4000 (clipped
to [0, 1]) x the near-bright depth 1 - enc / 1.25.

What could be silently wrong: the channel is not that formula, the direction of the velocity
leaks in (the user asked for the speed, a scalar), the depth channel moves, or a render without
the velocity goes through."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym import vision                                         # noqa: E402
from test_obs_potential import CELL, VIEWS, _poses, scene          # noqa: E402,F401

KW = dict(cell=CELL, device="cpu", max_steps=256)


def test_the_channel_is_the_speed_share_times_the_nearness(scene):
    p = _poses(*VIEWS)
    n = len(VIEWS)
    base = vision.GpuLidar(None, 16, 8, **KW)
    lt = vision.GpuLidar(None, 16, 8, speed_tint=True, **KW)
    assert (lt.channels, lt.tint_channel, lt.uses_velocity) == (2, 1, True)
    d = base.render(*p)
    near = (1.0 - d / 1.25).clamp(0.0, 1.0)
    assert float(near.min()) >= 0.0 and float(near.max()) > 0.5       # the floor is near-bright
    for sp, share in ((0.0, 0.0), (1000.0, 0.25), (4000.0, 1.0), (9000.0, 1.0)):
        v = torch.tensor([[0.6 * sp, -0.8 * sp, 0.0]] * n)             # |v| = sp
        out = lt.render(*p, velocity=v)
        assert out.shape == (n, 8, 16, 2)
        assert torch.equal(out[..., 0], d)                              # the depth is untouched
        assert torch.allclose(out[..., 1], share * near, atol=1e-6)
    # the speed, not the direction
    a = lt.render(*p, velocity=torch.tensor([[3000.0, 0.0, 0.0]] * n))[..., 1]
    b = lt.render(*p, velocity=torch.tensor([[0.0, 0.0, -3000.0]] * n))[..., 1]
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        lt.render(*p)                                                   # no velocity, no channel
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, speed_tint=True, depth_enc="log", **KW)


def test_the_velocity_mode_colours_the_camera_frame_components(scene):
    """--obs-speed-tint 2: R, G, B = nearness x (1 + c) / 2, c = (forward, left, up) velocity in
    the camera's yaw frame / 4000 clipped to [-1, 1]: mid-grey at rest, the depth untouched, the
    scalar mode unchanged by the mode's code."""
    import numpy as np
    p = _poses(*VIEWS)
    n = len(VIEWS)
    base = vision.GpuLidar(None, 16, 8, **KW)
    lv = vision.GpuLidar(None, 16, 8, speed_tint=2, **KW)
    assert (lv.channels, lv.tint_channel, lv.uses_velocity) == (4, 1, True)
    d = base.render(*p)
    near = (1.0 - d / 1.25).clamp(0.0, 1.0)
    still = lv.render(*p, velocity=torch.zeros(n, 3))
    assert torch.equal(still[..., 0], d)
    assert torch.allclose(still[..., 1:], 0.5 * near[..., None].expand(-1, -1, -1, 3), atol=1e-6)
    v = torch.tensor([[1200.0, -2500.0, -900.0], [9000.0, 300.0, 5000.0]] * n)[:n]
    out = lv.render(*p, velocity=v)
    yr = torch.as_tensor(p[1], dtype=torch.float32).reshape(n) * (np.pi / 180.0)
    cy, sy = torch.cos(yr), torch.sin(yr)
    c = torch.stack((v[:, 0] * cy + v[:, 1] * sy, -v[:, 0] * sy + v[:, 1] * cy, v[:, 2]), 1)
    col = 0.5 * (1.0 + (c / 4000.0).clamp(-1.0, 1.0))
    for k in range(3):
        assert torch.allclose(out[..., 1 + k], col[:, k].reshape(n, 1, 1) * near, atol=1e-5)
    # the speed itself, not the world direction, survives a turn of the camera and the velocity
    # together: rotate both by the same angle and the colours do not move
    rot = torch.tensor(30.0 * np.pi / 180.0)
    v2 = torch.stack((v[:, 0] * torch.cos(rot) - v[:, 1] * torch.sin(rot),
                      v[:, 0] * torch.sin(rot) + v[:, 1] * torch.cos(rot), v[:, 2]), 1)
    yaw2 = torch.as_tensor(p[1], dtype=torch.float32) + 30.0
    a = lv._append_speed_tint(d, v, p[1])[..., 1:]
    b = lv._append_speed_tint(d, v2, yaw2)[..., 1:]
    assert torch.allclose(a, b, atol=1e-5)
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, speed_tint=4, **KW)


def _flat(n, yaw=0.0, pitch=0.0):
    return (torch.full((n,), float(yaw)), torch.full((n,), float(pitch)))


def test_the_uniform_map_is_the_velocity_at_every_pixel(scene):
    p = _poses(*VIEWS)
    n = len(VIEWS)
    lu = vision.GpuLidar(None, 16, 8, speed_tint=3, **KW)
    assert (lu.channels, lu.tint_channel) == (4, 1)
    out = lu.render(*p, velocity=torch.tensor([[2000.0, 0.0, -4000.0]] * n))
    img = out[..., 1:]
    assert torch.allclose(img, img[:, :1, :1, :].expand_as(img))       # one colour per frame
    assert float(img[..., 2].max()) == 0.0                             # straight down: blue 0


def test_the_flow_is_the_ego_motion_of_each_pixel(scene):
    lf = vision.GpuLidar(None, 16, 8, flow=1, **KW)
    ld = vision.GpuLidar(None, 16, 8, flow=2, **KW)
    assert (lf.channels, lf.flow_channel, lf.uses_velocity) == (3, 1, True)
    n = 1
    y, pt = _flat(n)
    dep = torch.full((n, 8, 16), 0.3)
    zero = lf._append_flow(dep, y, pt, torch.zeros(n, 3))[..., 1:]
    assert float(zero.abs().max()) == 0.0
    # forward (yaw 0 = +x): the points spread from the middle - left half to the left (columns
    # fall), right half to the right, top up (rows fall), bottom down
    # channel 0 is the depth; 1 = columns, 2 = rows
    f = lf._append_flow(dep, y, pt, torch.tensor([[3000.0, 0.0, 0.0]]))[0]
    assert (f[:, :8, 1] < 0).all() and (f[:, 8:, 1] > 0).all()
    assert (f[:4, :, 2] < 0).all() and (f[4:, :, 2] > 0).all()
    # to the LEFT (+y at yaw 0): every point slides right on the image
    s = lf._append_flow(dep, y, pt, torch.tensor([[0.0, 3000.0, 0.0]]))[0]
    assert (s[..., 1] > 0).all()
    # real depth: the nearer half of the image moves more
    d2 = dep.clone()
    d2[..., :8] = 0.05                                  # the left half near, the right half far
    r = ld._append_flow(d2, y, pt, torch.tensor([[0.0, 3000.0, 0.0]]))[0]
    assert float(r[:, :8, 1].abs().mean()) > 2.0 * float(r[:, 8:, 1].abs().mean())
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, flow=1, pinhole=True, **KW)


def test_the_blob_sits_where_the_velocity_points(scene):
    lb = vision.GpuLidar(None, 16, 8, vel_blob=True, **KW)
    assert (lb.channels, lb.blob_channel, lb.uses_velocity) == (2, 1, True)
    n = 1
    y, pt = _flat(n)
    dep = torch.full((n, 8, 16), 0.3)
    assert float(lb._append_vel_blob(dep, y, pt, torch.zeros(n, 3))[..., 1].max()) == 0.0
    b = lb._append_vel_blob(dep, y, pt, torch.tensor([[4000.0, 0.0, 0.0]]))[0, ..., 1]
    r, c = divmod(int(b.argmax()), 16)
    # dead ahead (the 16 x 8 test camera's middle pixels sit 3.75 / 5.6 deg off it), full speed
    assert r in (3, 4) and c in (7, 8) and float(b.max()) > 0.65
    left = lb._append_vel_blob(dep, y, pt, torch.tensor([[2000.0, 2000.0, 0.0]]))[0, ..., 1]
    assert int(left.argmax()) % 16 < 4                                   # 45 deg left: left side


def test_the_warp_reads_the_depth_where_its_content_comes_from(scene):
    lw = vision.GpuLidar(None, 16, 8, warp=True, **KW)
    assert (lw.channels, lw.warp_channel, lw.uses_velocity) == (2, 1, True)
    n = 1
    y, pt = _flat(n)
    dep = torch.linspace(0.05, 1.2, 8 * 16).reshape(n, 8, 16)          # every pixel different
    still = lw._append_warp(dep, y, pt, torch.zeros(n, 3))
    assert torch.equal(still[..., 0], dep)
    assert torch.allclose(still[..., 1], dep, atol=1e-5)                 # no motion, no warp
    for v in ([0.0, 3000.0, 0.0], [3000.0, 0.0, 0.0], [500.0, -2000.0, -1500.0]):
        vel = torch.tensor([v])
        got = lw._append_warp(dep, y, pt, vel)[0, ..., 1]
        # the reference: x reads the depth at x - F(x) (F capped by 8 tanh(F / 8)), bilinear,
        # the coordinate clamped to the image (grid_sample's border padding)
        D = lw.decode_depth(dep).clamp(min=vision.FLOW_DMIN)
        col, row = lw._flow_px(D, y, pt, vel, vision.WARP_DT)
        col = (8.0 * torch.tanh(col / 8.0))[0]
        row = (8.0 * torch.tanh(row / 8.0))[0]
        want = torch.empty(8, 16)
        for r in range(8):
            for c in range(16):
                sx = min(max(c - float(col[r, c]), 0.0), 15.0)
                sy = min(max(r - float(row[r, c]), 0.0), 7.0)
                x0, y0 = int(sx), int(sy)
                x1, y1 = min(x0 + 1, 15), min(y0 + 1, 7)
                fx, fy = sx - x0, sy - y0
                d = dep[0]
                want[r, c] = ((1 - fy) * ((1 - fx) * d[y0, x0] + fx * d[y0, x1])
                              + fy * ((1 - fx) * d[y1, x0] + fx * d[y1, x1]))
        assert torch.allclose(got, want, atol=1e-4), (v, float((got - want).abs().max()))
    # sliding LEFT: the scene slides right, so a pixel reads the depth from its left
    left = lw._append_warp(dep, y, pt, torch.tensor([[0.0, 3000.0, 0.0]]))[0, ..., 1]
    assert float((left[:, 1:] - dep[0, :, 1:]).mean()) < 0.0              # dep rises to the right
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, warp=True, pinhole=True, **KW)


def _lit(img):
    return {(int(r), int(c)) for r, c in zip(*torch.nonzero(img > 0, as_tuple=True))}


def test_the_arrow_points_where_the_player_moves(scene):
    la = vision.GpuLidar(None, 64, 32, glyph=1, **KW)
    assert (la.channels, la.glyph_channel, la.uses_velocity) == (2, 1, True)
    n = 1
    y, pt = _flat(n)
    dep = torch.full((n, 32, 64), 0.3)
    still = la._append_glyph(dep, y, pt, torch.zeros(n, 3))[0, ..., 1]
    assert len(_lit(still)) == 1 and float(still.max()) == 1.0           # one dot at the base
    (r0, c0), = _lit(still)
    assert c0 in (31, 32) and r0 > 16                                     # centred, below the middle

    def head(v):
        g = la._append_glyph(dep, y, pt, torch.tensor([v]))[0, ..., 1]
        assert 0.0 <= float(g.min()) and float(g.max()) <= 1.0
        r, c = divmod(int(g.argmax()), 64)
        return g, r, c
    g, r, c = head([0.0, -4000.0, 0.0])                                   # to the right
    # the tail stays at the base, dim (a pixel keeps its brightest sample: the tail's few)
    assert c > c0 + 8 and float(g[r0, c0]) < 0.5 and float(g.max()) == 1.0
    g, r, c = head([0.0, 4000.0, 0.0])                                    # to the left
    assert c < c0 - 8
    g, r, c = head([4000.0, 0.0, 0.0])                                    # ahead: toward the horizon
    assert r < r0 and abs(c - c0) <= 1
    g, r, c = head([-4000.0, 0.0, 0.0])                                   # back: down, on the image
    assert r > r0 and abs(c - c0) <= 1 and r < 32
    g, r, c = head([0.0, 0.0, -2000.0])                                   # falling: down
    assert r > r0
    # the arrow turns with the camera: the same velocity seen from a camera turned 90 deg left
    gt = la._append_glyph(dep, torch.full((n,), 90.0), pt, torch.tensor([[4000.0, 0.0, 0.0]]))
    assert divmod(int(gt[0, ..., 1].argmax()), 64)[1] > c0 + 8           # +x is now to the right


def test_the_particles_streak_along_the_synthetic_flow(scene):
    lp = vision.GpuLidar(None, 16, 8, glyph=2, **KW)
    n = 1
    y, pt = _flat(n)
    dep = torch.full((n, 8, 16), 0.3)
    still = lp._append_glyph(dep, y, pt, torch.zeros(n, 3))[0, ..., 1]
    grid = {(r, c) for r in (2, 6) for c in (2, 6, 10, 14)}
    assert _lit(still) == grid and float(still.max()) == 1.0
    # sliding LEFT (+y at yaw 0): every particle's streak runs to the right, head brightest
    s = lp._append_glyph(dep, y, pt, torch.tensor([[0.0, 4000.0, 0.0]]))[0, ..., 1]
    for r in (2, 6):
        row = s[r]
        for c in (2, 6, 10):
            seg = row[c:c + 4]
            assert 0.0 < float(seg[0]) < float(seg.max())                # the dim tail at it
            assert int(seg.argmax()) > 0                                  # the head to its right
    # ahead: the streaks spread from the middle (the left particles' heads further left)
    f = lp._append_glyph(dep, y, pt, torch.tensor([[4000.0, 0.0, 0.0]]))[0, ..., 1]
    assert float(f[2, 1]) > 0.0 or float(f[1, 2]) > 0.0 or float(f[1, 1]) > 0.0
    with pytest.raises(ValueError):
        vision.GpuLidar(None, 16, 8, glyph=3, **KW)
