"""The target-ramp channel (surfgym.targetmask): the Triton kernel equals the torch reference, the
channel is +1 / -1 / 0, the next target wins a tie, and occlusion follows the depth hit."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym import targetmask as tmm  # noqa: E402

MESH = ROOT / "runs" / "research" / "ramps_mesh_surf_edgeflow_blue025.npz"


class _Lidar:
    """the minimal lidar surface TargetMask.render reads: rays, the depth decoder, range, cell"""

    def __init__(self, n, H=16, W=32, dev="cpu"):
        self.H, self.W, self.pinhole = H, W, False
        self.range, self.cell, self.device = 11500.0, 32.0, torch.device(dev)
        self.yoff = torch.linspace(-1.0, 1.0, W, device=self.device)
        self.poff = torch.linspace(-0.7, 0.2, H, device=self.device)
        self._dx = self._dy = self._dz = None

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

    def decode_depth(self, enc):
        return enc.float()


@pytest.mark.skipif(not MESH.exists(), reason="needs the local blue025 face extraction")
def test_kernel_matches_the_torch_reference_and_the_channel_is_signed():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tm = tmm.TargetMask(str(MESH), None, dev)
    z = np.load(MESH)
    ramps = [int(s) for s in np.flatnonzero(z["cat"] == 1)][:4]
    N = 8
    rng = np.random.default_rng(0)
    c = z["centroid"][ramps[0]]
    origin = torch.as_tensor(c[None] + rng.normal(0, 300, (N, 3)) + [0, 0, 400.0],
                             dtype=torch.float32, device=dev)
    lid = _Lidar(N, dev=dev)
    lid._ensure_buffers(N)
    yaw = torch.as_tensor(rng.uniform(0, 360, N), dtype=torch.float32, device=dev)
    pitch = torch.full((N,), -30.0, device=dev)
    lid._dirs_equiangular(N, yaw, pitch, np.pi / 180.0)
    ex = origin[:, 0].view(N, 1, 1)
    ey = origin[:, 1].view(N, 1, 1)
    ez = (origin[:, 2] + 17.0).view(N, 1, 1)
    sid = torch.as_tensor([ramps[i % len(ramps)] for i in range(N)], device=dev)
    ref, kref = tm._tri_hit_torch(sid, ex, ey, ez, lid._dx, lid._dy, lid._dz)
    got, kgot = tm._tri_hit(sid, ex, ey, ez, lid._dx, lid._dy, lid._dz)
    fin = torch.isfinite(ref)
    assert torch.equal(fin, torch.isfinite(got))
    assert torch.allclose(ref[fin], got[fin], rtol=1e-4, atol=0.5)
    assert torch.equal(kref[fin].long(), kgot[fin].long())
    # the channel: depth = the next target's own hit -> +1 there; an occluder in front -> 0
    tm.set_targets(N, sid.cpu().numpy(), np.full(N, -1))
    depth = torch.where(fin, ref, torch.full_like(ref, lid.range))
    ch = tm.render(lid, origin, yaw, pitch, torch.zeros(N, dtype=torch.int64, device=dev), depth)
    assert set(torch.unique(ch).tolist()) <= {-1.0, 0.0, 1.0}
    assert bool((ch[fin] == 1.0).all())
    occl = torch.where(fin, ref - 500.0, depth)          # something 500 u in front of the ramp
    # (500 u in front along the ray is > GRAZE cells off the ramp's plane unless the ray grazes)
    ch2 = tm.render(lid, origin, yaw, pitch, torch.zeros(N, dtype=torch.int64, device=dev), occl)
    # a ray whose occluder point (500 u short of the ramp) is more than GRAZE cells off the ramp's
    # plane is occluded; a GRAZING ray (the occluder point still within GRAZE cells of the plane,
    # as the depth march's early stops are) keeps seeing the target by design
    s_ = sid.view(N, 1, 1).expand_as(kref)
    nrm = tm.pn[s_, kref]
    cosang = ((nrm[..., 0] * lid._dx + nrm[..., 1] * lid._dy + nrm[..., 2] * lid._dz).abs()
              / nrm.norm(dim=-1).clamp_min(1e-9))
    off_plane = 500.0 * cosang > tmm.GRAZE * lid.cell + 1.0
    assert not bool((ch2[fin & off_plane] == 1.0).any())
    assert bool(off_plane[fin].any())
