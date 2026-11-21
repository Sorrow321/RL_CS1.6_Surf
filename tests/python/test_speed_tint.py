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
