"""goalball.py - the goal as a second DEPTH channel (user, 2026-09-02).

Render the goal sphere alone - no map geometry - in the SAME camera as the
map depth: one analytic ray-sphere intersection per pixel against the rays
the equiangular march casts (surfcore.h convention: row 0 looks up, col 0
looks left, eye at origin + 17 u, 12 u ducked). The pixel VALUE is the
ball's depth in the march's own encoding (``t/near`` plus the slow tail), so
the two channels share one distance scale; no ball on a ray = 0, which the
depth channel never produces (its minimum is a contact).

Two things a bare render would get wrong, both fixed here:

* a 192 u ball at 5,000 u is under a pixel at 64x32 - the ball is drawn
  with a MINIMUM angular radius (``min_px`` pixels) so it never vanishes;
  distance still lives in the value, not the size;
* a memoryless policy loses a goal that leaves the field of view - so when
  the goal is off-screen its direction is clamped to the image border and
  a ``marker_px`` block is painted there with the goal's depth value: the
  game-HUD off-screen arrow, in pixels. The channel always says where to
  look.

Wraps a :class:`surfgym.vision.GpuLidar` (equiangular camera only; the
pinhole variant is a separate experiment). The ball views are appended
after ALL of the wrapped lidar's channels: channel 0 stays the map depth,
so the ball shares its pixels with it, and a 4-channel --normals lidar
keeps (depth, nx, ny, nz) in front of the views. ``channels`` is
``lidar.channels + views``, so the trunk's ``in_ch`` follows exactly as it
does for --surf-mask.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["GoalBallLidar"]


class GoalBallLidar:
    # yaw offsets of the four views, degrees: front, back, left, right
    # (forward = (cos yaw, sin yaw), left = +90 in this convention)
    VIEW_YAW = {1: (0.0,), 4: (0.0, 180.0, 90.0, -90.0)}

    def __init__(self, lidar, n_envs: int, radius: float = 192.0,
                 min_px: float = 1.5, marker_px: int = 2, views: int = 4):
        import torch
        if int(views) not in self.VIEW_YAW:
            raise ValueError("views must be 1 (front + border marker) or 4 "
                             "(front/back/left/right, no marker needed)")
        self.views = int(views)
        if getattr(lidar, "pinhole", False):
            raise ValueError("GoalBallLidar mirrors the equiangular camera; "
                             "--pinhole is a separate experiment")
        self.lidar = lidar
        self.N = int(n_envs)
        # channels 0..base-1 = whatever the lidar renders (depth first, so
        # the FRONT ball view shares its pixels with the map depth; under
        # --normals the ego normal rides in 1..3), then the ball in each view
        self.base = int(getattr(lidar, "channels", 1))
        if self.base < 1:
            raise ValueError("the wrapped lidar renders no channels")
        self.channels = self.base + self.views
        self.W, self.H = int(lidar.W), int(lidar.H)
        self.device = lidar.device
        self.near, self.range = float(lidar.near), float(lidar.range)
        self.yoff, self.poff = lidar.yoff, lidar.poff          # (W,), (H,) rad
        # the camera's fov, back out of the pixel-centre grid
        self.hfov = float(self.yoff[0]) / (0.5 - 0.5 / self.W) if self.W > 1 \
            else 0.0
        self.vfov = float(self.poff[0]) / (0.5 - 0.5 / self.H) if self.H > 1 \
            else 0.0
        self.min_ang = float(min_px) * (self.hfov / max(self.W, 1))
        self.marker_px = int(marker_px)
        self.mode = "live"                       # "off" zeroes the ball channel
        self.center = torch.full((self.N, 3), float("nan"), dtype=torch.float32,
                                 device=self.device)
        self.radius = torch.full((self.N,), float(radius), dtype=torch.float32,
                                 device=self.device)
        self._hh = torch.arange(self.H, device=self.device, dtype=torch.float32)
        self._ww = torch.arange(self.W, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------ goals
    def set_goals(self, idx, centers, radius=None) -> None:
        import torch
        idx = torch.as_tensor(np.asarray(idx, np.int64), device=self.device)
        self.center[idx] = torch.as_tensor(np.asarray(centers, np.float32),
                                           device=self.device)
        if radius is not None:
            self.radius[idx] = torch.as_tensor(
                np.asarray(radius, np.float32), device=self.device)

    def describe(self) -> str:
        tail = (f"off-screen border marker {self.marker_px}px" if self.views == 1
                else "front/back/left/right views, no marker")
        return (f"goal ball: depth channel(s) of the goal sphere in the "
                f"{self.W}x{self.H} camera (hfov {math.degrees(self.hfov):.0f}, "
                f"vfov {math.degrees(self.vfov):.0f}), min angular radius "
                f"{math.degrees(self.min_ang):.2f} deg, {tail}, after the "
                f"lidar's {self.base} channel(s) -> in_ch {self.channels}")

    # ------------------------------------------------------------ render
    def _encode(self, t):
        import torch
        t = torch.clamp(t, max=self.range)
        return (torch.clamp(t, max=self.near) / self.near
                + 0.25 * (1.0 - torch.exp(-torch.clamp(t - self.near, min=0.0)
                                          / 2500.0)))

    def ball(self, origin, yaw_deg, pitch_deg, ducked, center=None,
             radius=None, view_yaw: float = 0.0, marker: bool = True):
        """(N, H, W) ball-depth channel for the given poses, looking
        ``view_yaw`` degrees off the player's yaw; ``center`` / ``radius``
        default to the per-env goals (rows 0..N-1 of the pose batch).
        ``marker`` paints the off-screen border marker (single view)."""
        import torch
        yaw_deg = yaw_deg + float(view_yaw)
        n = origin.shape[0]
        c = self.center[:n] if center is None else center
        R = self.radius[:n] if radius is None else radius
        out = torch.zeros((n, self.H, self.W), dtype=torch.float32,
                          device=self.device)
        if self.mode == "off":
            return out
        d2r = math.pi / 180.0
        ez = origin[:, 2] + torch.where(ducked.bool(), 12.0, 17.0)
        eye = torch.stack([origin[:, 0], origin[:, 1], ez], dim=1)     # (n,3)
        v = c - eye                                                    # (n,3)
        dist = torch.linalg.norm(v, dim=1)                             # (n,)
        ok = torch.isfinite(dist) & (dist > 1e-3)
        if not bool(ok.any()):
            return out
        # rays, exactly the march's equiangular grid
        p = pitch_deg.view(n, 1, 1) * d2r + self.poff.view(1, self.H, 1)
        y = yaw_deg.view(n, 1, 1) * d2r + self.yoff.view(1, 1, self.W)
        cp = torch.cos(p)
        dx = cp * torch.cos(y)
        dy = cp * torch.sin(y)
        dz = torch.sin(p).expand(n, self.H, self.W)
        # apparent radius floor: never under min_ang of angular size
        r_eff = torch.maximum(R, dist * math.tan(self.min_ang))
        b = (dx * v[:, 0].view(n, 1, 1) + dy * v[:, 1].view(n, 1, 1)
             + dz * v[:, 2].view(n, 1, 1))                            # (n,H,W)
        disc = b * b - (dist * dist - r_eff * r_eff).view(n, 1, 1)
        hit = (disc >= 0.0) & (b > 0.0) & ok.view(n, 1, 1)
        t = b - torch.sqrt(torch.clamp(disc, min=0.0))
        out = torch.where(hit, self._encode(torch.clamp(t, min=0.0)), out)
        # off-screen: clamp the goal's direction to the border, paint the
        # marker there with the goal's depth
        yaw_g = torch.atan2(v[:, 1], v[:, 0]) - yaw_deg * d2r
        yaw_g = torch.remainder(yaw_g + math.pi, 2.0 * math.pi) - math.pi
        pitch_g = torch.asin(torch.clamp(v[:, 2] / torch.clamp(dist, min=1e-3),
                                         -1.0, 1.0)) - pitch_deg * d2r
        off = ok & ((yaw_g.abs() > 0.5 * self.hfov)
                    | (pitch_g.abs() > 0.5 * self.vfov))
        if marker and bool(off.any()):
            col = torch.clamp((0.5 - yaw_g / max(self.hfov, 1e-6)) * self.W - 0.5,
                              0.0, self.W - 1.0)
            row = torch.clamp((0.5 - pitch_g / max(self.vfov, 1e-6)) * self.H
                              - 0.5, 0.0, self.H - 1.0)
            half = 0.5 * (self.marker_px - 1)
            mh = (self._hh.view(1, self.H) - row.view(n, 1)).abs() <= half + 0.5
            mw = (self._ww.view(1, self.W) - col.view(n, 1)).abs() <= half + 0.5
            mark = mh.view(n, self.H, 1) & mw.view(n, 1, self.W) \
                & off.view(n, 1, 1)
            val = self._encode(dist).view(n, 1, 1).expand(n, self.H, self.W)
            out = torch.where(mark, val, out)
        return out

    def render(self, origin, yaw_deg, pitch_deg, ducked, idx=None):
        """(N, H, W, base + views): the lidar's channels [map depth, ...],
        then the goal ball per view. ``idx`` selects which envs' goals a
        SUBSET batch belongs to (the truncation bootstrap renders a few
        reconstructed rows)."""
        import torch
        img = self.lidar.render(origin, yaw_deg, pitch_deg, ducked)
        if img.dim() == 3:                       # depth-only lidar: (N, H, W)
            img = img.unsqueeze(-1)
        if idx is not None:
            ii = torch.as_tensor(np.asarray(idx, np.int64), device=self.device)
            c, R = self.center[ii], self.radius[ii]
        else:
            c = R = None
        chans = [img]
        for vy in self.VIEW_YAW[self.views]:
            chans.append(self.ball(origin, yaw_deg, pitch_deg, ducked,
                                   center=c, radius=R, view_yaw=vy,
                                   marker=(self.views == 1)).unsqueeze(-1))
        return torch.cat(chans, dim=-1)
