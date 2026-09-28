"""The target-ramp channel (surfgym.targetmask), on a synthetic mesh (no map files needed): the
Triton kernel equals the torch reference; a target is drawn with NO occlusion; the nearer of the
two targets wins an overlap; a target split into many coplanar pieces (a BSP tessellation with
T-junctions) draws as one solid region; the finish box is a target; unit="object" joins an
A-frame's two slopes into one target."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import targetmask as tmm  # noqa: E402


def _quad(p0, p1, p2, p3, split=1):
    """a planar quad as 2*split*split triangles (a fine tessellation, like the BSP's)"""
    tris = []
    a = np.array(p0, float)
    u = (np.array(p1, float) - a) / split
    v = (np.array(p3, float) - a) / split
    for i in range(split):
        for j in range(split):
            q0 = a + u * i + v * j
            tris.append([q0, q0 + u, q0 + u + v])
            tris.append([q0, q0 + u + v, q0 + v])
    return np.array(tris)


def _mesh(tmp_path):
    # surface 0: a ramp in the plane y = 0 (x 0..1000, z 0..500), tessellated 8 x 8
    r0 = _quad([0, 0, 0], [1000, 0, 0], [1000, 0, 500], [0, 0, 500], split=8)
    # surface 1: a ramp farther along +y (y = 3000), same extent
    r1 = _quad([0, 3000, 0], [1000, 3000, 0], [1000, 3000, 500], [0, 3000, 500], split=2)
    # surfaces 2 + 3: an A-frame's two slopes meeting at a ridge (x 4000..5000)
    a = _quad([4000, 0, 0], [5000, 0, 0], [5000, 300, 300], [4000, 300, 300])
    b = _quad([4000, 300, 300], [5000, 300, 300], [5000, 600, 0], [4000, 600, 0])
    tris = np.concatenate([r0, r1, a, b]).astype(np.float32)
    ts = np.concatenate([np.full(len(r0), 0), np.full(len(r1), 1), np.full(len(a), 2),
                         np.full(len(b), 3)])
    p = tmp_path / "mesh.npz"
    np.savez(p, tris=tris, tri_surf=ts, cat=np.array([1, 1, 1, 1]))
    return p


class _Lidar:
    def __init__(self, H=24, W=48, dev="cpu"):
        self.H, self.W, self.pinhole, self.device = H, W, False, torch.device(dev)
        self.yoff = torch.linspace(-0.9, 0.9, W, device=self.device)
        self.poff = torch.linspace(-0.6, 0.6, H, device=self.device)

    def _ensure_buffers(self, n):
        self._dx = torch.zeros(n, self.H, self.W, device=self.device)
        self._dy = torch.zeros_like(self._dx)
        self._dz = torch.zeros_like(self._dx)

    def _dirs_equiangular(self, N, yaw_deg, pitch_deg, d2r):
        p = pitch_deg.view(N, 1, 1) * d2r + self.poff.view(1, self.H, 1)
        y = yaw_deg.view(N, 1, 1) * d2r + self.yoff.view(1, 1, self.W)
        cp = torch.cos(p)
        self._dx.copy_(cp * torch.cos(y))
        self._dy.copy_(cp * torch.sin(y))
        self._dz.copy_(torch.sin(p).expand_as(self._dz))


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _view(tm, lid, t1, t2, origin, yaw, pitch=0.0, force_torch=False):
    N = len(t1)
    tm.set_targets(N, t1, t2)
    o = torch.as_tensor(np.asarray(origin, np.float32).reshape(N, 3), device=DEV)
    return tm.render(lid, o, torch.as_tensor(np.asarray(yaw, np.float32), device=DEV),
                     torch.full((N,), float(pitch), device=DEV),
                     torch.zeros(N, dtype=torch.int64, device=DEV), force_torch=force_torch)


def test_kernel_matches_reference_no_occlusion_and_nearer_wins(tmp_path):
    tm = tmm.TargetMask(str(_mesh(tmp_path)), None, DEV)
    lid = _Lidar(dev=DEV)
    # from (500, -2000, 250 - eye 17) looking +y (yaw 90): ramp 0 is in front (y = 0, 2000 u),
    # ramp 1 behind it (y = 3000, 5000 u) - the SAME direction
    orig = [[500, -2000, 233]] * 4
    yaw = [90.0] * 4
    got = _view(tm, lid, [0, 1, 0, 1], [1, 0, -1, -1], orig, yaw)
    ref = _view(tm, lid, [0, 1, 0, 1], [1, 0, -1, -1], orig, yaw, force_torch=True)
    assert torch.equal(got, ref)
    r, c = lid.H // 2, lid.W // 2
    assert got[0, r, c] == 1.0          # t1 = ramp 0 (nearer): +1
    assert got[1, r, c] == -1.0         # t1 = ramp 1 (farther), t2 = ramp 0 nearer: -1 wins
    assert got[3, r, c] == 1.0          # ramp 1 alone, BEHIND ramp 0: drawn (no occlusion)
    # the finely tessellated ramp 0 (128 triangles, shared edges) draws as ONE solid region
    lit = (got[2] == 1.0).cpu().numpy()
    for row in np.flatnonzero(lit.any(1)):
        cols = np.flatnonzero(lit[row])
        assert cols.max() - cols.min() + 1 == len(cols)       # no hole inside a row


def test_finish_box_is_a_target(tmp_path):
    tm = tmm.TargetMask(str(_mesh(tmp_path)),
                        {"mins": (2000, -100, 0), "maxs": (2100, 100, 100)}, DEV)
    lid = _Lidar(dev=DEV)
    ch = _view(tm, lid, [tmm.FIN], [tmm.NONE], [[2050, -2000, 33]], [90.0])
    assert (ch == 1.0).sum() > 0 and (ch == -1.0).sum() == 0


def test_object_unit_joins_an_a_frame(tmp_path):
    tm = tmm.TargetMask(str(_mesh(tmp_path)), None, DEV, unit="object")
    obj = tm.obj_of_surf
    assert obj[2] == obj[3]                 # the A-frame's two slopes are one object
    assert len({int(obj[0]), int(obj[1]), int(obj[2])}) == 3
    lid = _Lidar(dev=DEV)
    # from above and before the ridge, looking along it (+x) and down: both slopes are the one
    # target, so the lit region spans both sides of the ridge
    ch = _view(tm, lid, [int(obj[2])], [tmm.NONE], [[3000, 300, 700]], [0.0], pitch=-20.0)
    lit = (ch[0] == 1.0).cpu().numpy()
    assert lit.sum() > 0
    cols = np.flatnonzero(lit.any(0))
    assert cols.min() < lid.W // 2 < cols.max()   # left and right of the centre line


def test_takeoff_fade_is_continuous_and_ignores_touches():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 9))
        leave = np.sort(rng.uniform(0, 400, n))
        leave[rng.random(n) < 0.2] = np.inf          # some ramps never left
        leave = np.sort(leave)
        prev = tmm.takeoff_fade_values(leave, -1.0, 30.0)
        assert np.allclose(prev[:2], [1.0, 0.5]) and np.allclose(prev[2:], 0.0)
        for t in np.arange(0.0, 500.0, 1.0):
            v = tmm.takeoff_fade_values(leave, t, 30.0)
            assert np.all(v >= -1e-9) and np.all(v <= 1.0 + 1e-9)
            assert np.max(np.abs(v - prev)) <= 1.0 / 30.0 + 1e-9    # at most one ramp step per tick
            prev = v


def test_slots_max_combine_and_zero_slots(tmp_path):
    tm = tmm.TargetMask(str(_mesh(tmp_path)), None, DEV)
    lid = _Lidar(dev=DEV)
    o = torch.as_tensor(np.asarray([[500, -2000, 233]], np.float32), device=DEV)
    yaw = torch.as_tensor([90.0], device=DEV)
    z0 = torch.zeros(1, device=DEV)
    dk = torch.zeros(1, dtype=torch.int64, device=DEV)
    r, c = lid.H // 2, lid.W // 2
    # ramp 0 (near, 0.5) in front of ramp 1 (far, 1.0): MAX shows 1.0 where they overlap
    tm.set_slots(1, [[0, 1]], [[0.5, 1.0]])
    assert float(tm.render(lid, o, yaw, z0, dk)[0, r, c]) == 1.0
    tm.set_slots(1, [[0, 1]], [[0.5, 1.0]], combine="nearest")
    assert float(tm.render(lid, o, yaw, z0, dk)[0, r, c]) == 0.5
    # a slot at value 0 neither draws nor hides anything
    tm.set_slots(1, [[0, 1]], [[0.0, 0.7]])
    ch = tm.render(lid, o, yaw, z0, dk)
    assert float(ch[0, r, c]) == pytest.approx(0.7)
    tm.set_slots(1, [[0]], [[0.0]])
    assert float(tm.render(lid, o, yaw, z0, dk).abs().sum()) == 0.0
