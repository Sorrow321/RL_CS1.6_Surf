"""GoalBallLidar: the goal sphere as a depth channel, with the off-screen
border marker. CPU torch, a stub depth lidar (no map, no GPU)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from surfgym.goalball import GoalBallLidar  # noqa: E402

W, H = 64, 32
HFOV, VFOV = 120.0, 90.0
NEAR, RANGE = 2000.0, 11500.0


class StubLidar:
    """The attributes GoalBallLidar reads, the shipped camera's numbers."""

    def __init__(self):
        d2r = math.pi / 180.0
        self.W, self.H = W, H
        self.channels = 1
        self.pinhole = False
        self.device = torch.device("cpu")
        self.near, self.range = NEAR, RANGE
        self.yoff = torch.as_tensor(
            (HFOV * (0.5 - (np.arange(W) + 0.5) / W)) * d2r, dtype=torch.float32)
        self.poff = torch.as_tensor(
            (VFOV * (0.5 - (np.arange(H) + 0.5) / H)) * d2r, dtype=torch.float32)

    def render(self, origin, yaw_deg, pitch_deg, ducked):
        return torch.zeros((origin.shape[0], H, W))


def enc(t):
    t = min(t, RANGE)
    return min(t, NEAR) / NEAR + 0.25 * (1.0 - math.exp(-max(t - NEAR, 0.0) / 2500.0))


def pose(n=1, yaw=0.0, pitch=0.0):
    o = torch.zeros((n, 3))
    return (o, torch.full((n,), float(yaw)), torch.full((n,), float(pitch)),
            torch.zeros((n,), dtype=torch.int32))


def test_fov_recovered_and_channels():
    gb = GoalBallLidar(StubLidar(), 4, views=1)
    assert gb.channels == 2
    assert GoalBallLidar(StubLidar(), 4).channels == 5
    assert abs(math.degrees(gb.hfov) - HFOV) < 1e-4
    assert abs(math.degrees(gb.vfov) - VFOV) < 1e-4


def test_ball_straight_ahead_hits_centre_with_its_depth():
    gb = GoalBallLidar(StubLidar(), 1, views=1)
    gb.set_goals([0], [[3000.0, 0.0, 17.0]])       # dead ahead at eye height
    img = gb.render(*pose())
    assert img.shape == (1, H, W, 2)
    ball = img[0, :, :, 1]
    # centre pixels (the two middle columns, two middle rows) are hit
    hit = ball > 0
    assert hit[H // 2 - 1: H // 2 + 1, W // 2 - 1: W // 2 + 1].all()
    # depth value = front intersection at dist - R (R = 192 > angular floor)
    front = enc(3000.0 - 192.0)
    assert abs(float(ball[H // 2, W // 2]) - front) < 5e-3
    # nothing painted far from the ball
    assert not hit[0, 0] and not hit[H - 1, W - 1]
    # depth channel untouched (stub returns zeros)
    assert float(img[..., 0].abs().sum()) == 0.0


def test_far_ball_keeps_minimum_angular_size():
    gb = GoalBallLidar(StubLidar(), 1, min_px=1.5, views=1)
    gb.set_goals([0], [[9000.0, 0.0, 17.0]])
    ball = gb.render(*pose())[0, :, :, 1]
    n_hit = int((ball > 0).sum())
    assert n_hit >= 4, n_hit                 # ~3 px across at the floor
    assert n_hit <= 40


def test_ball_behind_paints_border_marker_with_distance():
    gb = GoalBallLidar(StubLidar(), 1, marker_px=2, views=1)
    gb.set_goals([0], [[-2500.0, 10.0, 17.0]])    # behind, slightly left
    ball = gb.render(*pose())[0, :, :, 1]
    hit = ball > 0
    assert hit.any()
    rows, cols = np.nonzero(hit.numpy())
    # a 2x2-ish block on the LEFT border (yaw_g ~ +pi -> col 0 side),
    # vertically centred (eye height)
    assert cols.max() <= 1
    assert abs(rows.mean() - (H / 2 - 0.5)) < 2.0
    assert abs(float(ball[hit].max()) - enc(math.hypot(2500.0, 10.0))) < 5e-3


def test_ball_above_fov_marks_top_row():
    gb = GoalBallLidar(StubLidar(), 1, views=1)
    gb.set_goals([0], [[500.0, 0.0, 17.0 + 3000.0]])   # ~80 deg up: out of vfov/2
    ball = gb.render(*pose())[0, :, :, 1]
    rows, cols = np.nonzero((ball > 0).numpy())
    assert rows.max() <= 1
    assert abs(cols.mean() - (W / 2 - 0.5)) < 2.0


def test_nan_goal_and_off_mode_render_nothing():
    gb = GoalBallLidar(StubLidar(), 2, views=1)
    gb.set_goals([1], [[3000.0, 0.0, 17.0]])
    img = gb.render(*pose(2))
    assert float(img[0, :, :, 1].abs().sum()) == 0.0     # NaN centre
    assert float(img[1, :, :, 1].abs().sum()) > 0.0
    gb.mode = "off"
    assert float(gb.render(*pose(2))[..., 1].abs().sum()) == 0.0


def test_yaw_rotation_moves_the_ball_across_columns():
    gb = GoalBallLidar(StubLidar(), 1, views=1)
    gb.set_goals([0], [[3000.0, 0.0, 17.0]])
    # looking 30 deg left: the ball (dead ahead in world) appears RIGHT of centre
    ball = gb.render(*pose(yaw=30.0))[0, :, :, 1]
    cols = np.nonzero((ball > 0).numpy())[1]
    assert cols.mean() > W / 2 + 4


def test_subset_render_uses_the_rows_goals():
    gb = GoalBallLidar(StubLidar(), 3, views=1)
    gb.set_goals([0, 1, 2], [[3000.0, 0.0, 17.0], [-3000.0, 0.0, 17.0],
                             [3000.0, 0.0, 17.0]])
    o, y, p, d = pose(1)
    front = gb.render(o, y, p, d, idx=[2])[0, :, :, 1]
    back = gb.render(o, y, p, d, idx=[1])[0, :, :, 1]
    assert float(front[H // 2, W // 2]) > 0.0
    assert float(back[H // 2, W // 2]) == 0.0 and (back > 0).any()


def _centre_hit(chan):
    return bool((chan[H // 2 - 1: H // 2 + 1, W // 2 - 1: W // 2 + 1] > 0).all())


def test_four_views_put_the_goal_in_exactly_the_right_view():
    gb = GoalBallLidar(StubLidar(), 4)          # views=4 default
    gb.set_goals([0, 1, 2, 3], [[3000.0, 0.0, 17.0], [-3000.0, 0.0, 17.0],
                                [0.0, 3000.0, 17.0], [0.0, -3000.0, 17.0]])
    img = gb.render(*pose(4))
    assert img.shape == (4, H, W, 5)
    # channel order: depth, front, back, left, right; env i's goal sits in
    # view i, dead centre, and NOWHERE else (no marker in 4-view mode)
    for i in range(4):
        for v in range(4):
            chan = img[i, :, :, 1 + v]
            if v == i:
                assert _centre_hit(chan), (i, v)
            else:
                assert float(chan.abs().sum()) == 0.0, (i, v)


def test_four_views_track_yaw():
    gb = GoalBallLidar(StubLidar(), 1)
    gb.set_goals([0], [[3000.0, 0.0, 17.0]])    # +x in the world
    img = gb.render(*pose(1, yaw=180.0))        # player faces -x: goal is behind
    assert _centre_hit(img[0, :, :, 2])          # back view
    assert float(img[0, :, :, 1].abs().sum()) == 0.0
    img = gb.render(*pose(1, yaw=90.0))         # facing +y: goal is to the right
    assert _centre_hit(img[0, :, :, 4])          # right view
