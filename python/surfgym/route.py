"""route.py - ego-frame lookahead geometry, the one observation every
superhuman racing agent has and this project did not.

The survey's finding (docs/research-litsurvey.md section 0): explicit
lookahead geometry in the observation is UNIVERSAL among systems that reached
or beat top-human performance, and removing it is the single largest ablation
reported in any of those papers.

  * GT Sophy (Nature 602:223): 60 ego-frame points per track edge spanning
    ~6 s at the CURRENT velocity. Ablation on Maggiore: no course points
    costs +2.64 s on a 114 s lap - larger than dropping the distributional
    critic (+0.69 s) or n-step returns (+1.48 s).
  * Fuchs et al. (RA-L 2021): 10 curvature samples at 0.2 s intervals.
  * Linesight (Trackmania WRs): 40 ego-frame virtual checkpoints, ~400 m
    ahead, taken from a reference line that "does not need to be fast...
    usually the centerline".
  * Swift (Nature 620:982): the next gate's four corners in ego frame.

What this module builds is the same object: a reference polyline through the
map resampled at constant arc length, and a featurizer that reports where
that line goes NEXT, in the player's own frame, at horizons that scale with
how fast the player is currently moving.

Three design points that are not arbitrary:

* **Span scales with speed** (Sophy). A fixed metric horizon is the wrong
  observation for an agent whose speed varies 5x across a run: at 500 u/s a
  15,000 u lookahead is 30 seconds of future and useless, at 2,500 u/s it is
  6 seconds and exactly the planning horizon. ``speed_floor`` stops the whole
  fan from collapsing onto one point at a standing spawn.

* **Ego frame, and normalized per point.** Raw offsets span 0-15,000 u, which
  a Tanh tower saturates on contact. Each point is divided by its OWN nominal
  span (speed * t_k), so a point that sits exactly where the route predicts
  reads ~1.0 forward and ~0 lateral, and the vector's DIRECTION carries the
  curvature Fuchs samples explicitly. Off-route states read large lateral
  components, which is the signal that matters.

* **This is route information, not policy information.** The line is a path
  through the map; it carries no actions, no timing and no control. Every
  system above has the track layout available a priori - a human racer walks
  the course. It does mean an agent trained with it is not "honest-perception"
  in the --gps sense: the fan localizes you against a known route. That is
  the treatment, deliberately, and it is what the papers evaluate.

Cost: one (N, L) argmin per decision. At L ~ 2,500 (128 u spacing over
cannonball's route) and N = 2,048 that is a 20 MB matmul+argmin, ~2% of the
lidar render it sits next to.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["RouteLine", "resample_polyline", "episodes_from_traj",
           "DEFAULT_OFFSETS"]

# Lookahead horizons in SECONDS, multiplied by current speed to get arc
# offsets. Dense near the player (the next ramp entry is what the next 20
# decisions are about) and sparse far out (where only the coarse direction of
# the route is actionable) - the same geometric spread Fuchs samples densely
# at 0.2 s and Sophy spans to ~6 s.
DEFAULT_OFFSETS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)

# Arc-length spacing of the resampled line, map units. 128 u at a champion's
# ~2,500 u/s is 51 ms of route per sample - finer than one decision at
# act_every=3 (30 ms) is pointless, and coarser starts rounding ramp entries.
DEFAULT_SPACING = 128.0


def episodes_from_traj(path):
    """Split a record_rollout .jsonl into per-episode float arrays.

    Format (tools/record_ckpt.py): a JSON dict header per episode, then rows
    ``[tick, x, y, z, vx, vy, vz, yaw, ...]``, then a footer dict. Recorders
    that omit the header are handled by also splitting where the tick counter
    goes backwards.
    """
    eps, cur, prev_tick = [], [], None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":                      # header or footer
                if cur:
                    eps.append(np.asarray(cur, np.float64))
                    cur = []
                prev_tick = None
                continue
            row = json.loads(line)
            if not isinstance(row, list) or len(row) < 8:
                continue
            tick = row[0]
            if prev_tick is not None and tick <= prev_tick:
                if cur:
                    eps.append(np.asarray(cur, np.float64))
                    cur = []
            prev_tick = tick
            cur.append(row[:8])
    if cur:
        eps.append(np.asarray(cur, np.float64))
    return [e for e in eps if len(e) > 1]


def resample_polyline(xyz: np.ndarray, spacing: float = DEFAULT_SPACING):
    """Constant-arc-length resample of a 3D polyline.

    Uniform spacing is what makes the featurizer's arc lookup pure index
    arithmetic instead of a searchsorted per point per env. Duplicate/
    stationary samples (a player standing still for 200 ticks at a respawn)
    collapse to one vertex, so they cannot inflate the arc coordinate.
    """
    p = np.asarray(xyz, np.float64).reshape(-1, 3)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.concatenate(([True], step > 1e-6))
    p = p[keep]
    if len(p) < 2:
        raise ValueError("route polyline has fewer than 2 distinct points")
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0),
                                                        axis=1))))
    total = float(s[-1])
    n = max(2, int(round(total / float(spacing))) + 1)
    q = np.linspace(0.0, total, n)
    out = np.empty((n, 3), np.float64)
    for k in range(3):
        out[:, k] = np.interp(q, s, p[:, k])
    return out.astype(np.float32), total


class RouteLine:
    """Reference route + the ego-frame lookahead fan.

    ``pts`` is (L, 3) in map units at constant ``spacing``; everything else is
    derived. Built once at startup and held on the training device.
    """

    def __init__(self, pts, spacing: float = DEFAULT_SPACING,
                 offsets=DEFAULT_OFFSETS, speed_floor: float = 500.0,
                 near_scale: float = 500.0, clamp: float = 4.0,
                 device=None, source: str = ""):
        import torch
        self.spacing = float(spacing)
        self.offsets = tuple(float(o) for o in offsets)
        self.speed_floor = float(speed_floor)
        self.near_scale = float(near_scale)
        self.clamp = float(clamp)
        self.source = source
        dev = device
        self.pts = torch.as_tensor(np.ascontiguousarray(pts),
                                   dtype=torch.float32, device=dev)
        self.L = int(self.pts.shape[0])
        # |R|^2, the constant half of the expanded squared distance: the
        # argmin then costs one (N,3)x(3,L) matmul instead of materializing
        # an (N, L, 3) difference tensor
        self.sq = (self.pts * self.pts).sum(1)
        self.t = torch.as_tensor(self.offsets, dtype=torch.float32,
                                 device=dev)
        self.length = float((self.L - 1) * self.spacing)

    # ---------------------------------------------------------------- build
    @classmethod
    def from_points(cls, xyz, spacing: float = DEFAULT_SPACING, **kw):
        pts, total = resample_polyline(xyz, spacing)
        line = cls(pts, spacing, **kw)
        line.raw_length = total
        return line

    @classmethod
    def load(cls, path, spacing: float = DEFAULT_SPACING, **kw):
        """Load a route .npy/.npz saved by tools/build_route.py.

        A saved route is ALREADY resampled; re-resampling at the same spacing
        is a no-op interpolation, so it is skipped and the stored spacing
        wins. That keeps a resumed run's features bit-identical to the run
        that produced the checkpoint.
        """
        p = Path(path)
        if p.suffix == ".npz":
            z = np.load(p)
            pts = np.asarray(z["route"], np.float32)
            spacing = float(z["spacing"]) if "spacing" in z.files else spacing
        else:
            pts = np.asarray(np.load(p), np.float32)
        return cls(pts, spacing, source=str(p), **kw)

    # ------------------------------------------------------------- features
    @property
    def n_features(self) -> int:
        """3 dims per point: the nearest point, then one per horizon."""
        return 3 * (1 + len(self.offsets))

    def describe(self) -> str:
        return (f"route {Path(self.source).name or '<memory>'}: {self.L} pts "
                f"@ {self.spacing:g}u ({self.length:,.0f}u), "
                f"{len(self.offsets)} horizons "
                f"{self.offsets[0]:g}-{self.offsets[-1]:g}s -> "
                f"{self.n_features} features")

    def features(self, origin, yaw_deg, speed):
        """(N,3) positions + (N,) yaw in DEGREES + (N,) horizontal speed
        -> (N, n_features) ego-frame lookahead, all on ``origin``'s device.

        Yaw convention matches src/env.c:247 exactly - forward is
        (cos yaw, sin yaw), left is (-sin yaw, cos yaw) - so these features
        live in the same frame as the velocity scalars the policy already
        reads, and a "route goes left" feature agrees in sign with a
        "velocity points left" one.
        """
        import torch
        p = origin
        # nearest vertex: |p|^2 is common to all L and drops out of the argmin
        d2 = self.sq.unsqueeze(0) - 2.0 * (p @ self.pts.t())
        i0 = d2.argmin(dim=1)                             # (N,)
        # Refine the vertex to a continuous ARC coordinate by projecting onto
        # the local tangent. Without this the whole fan inherits the vertex
        # snap (up to spacing/2 = 64 u), which at the shortest horizon
        # (125 u at the speed floor) is most of the feature.
        ti = torch.clamp(i0, max=self.L - 2)
        tang = self.pts[ti + 1] - self.pts[ti]             # (N, 3), ~spacing
        tl = tang.norm(dim=1).clamp_min(1e-6)
        proj = ((p - self.pts[ti]) * tang).sum(1) / tl     # units along the line
        s0 = ti.to(p.dtype) * self.spacing + proj.clamp(0.0, self.spacing)

        # arc offsets: uniform spacing makes the lookup index arithmetic, and
        # a lerp between neighbouring vertices removes the rounding the
        # shortest horizons cannot afford
        scale = torch.clamp(speed, min=self.speed_floor)  # (N,)
        span = scale.unsqueeze(1) * self.t.unsqueeze(0)   # (N, M) map units
        fidx = torch.clamp((s0.unsqueeze(1) + span) / self.spacing,
                           0.0, float(self.L - 1))
        lo = fidx.floor()
        w = (fidx - lo).unsqueeze(2)
        li = lo.long()
        hi = torch.clamp(li + 1, max=self.L - 1)
        tgt = self.pts[li] * (1.0 - w) + self.pts[hi] * w  # (N, M, 3)
        near = self.pts[i0].unsqueeze(1)                   # (N, 1, 3)
        d = torch.cat([near, tgt], dim=1) - p.unsqueeze(1)  # (N, 1+M, 3)

        yr = yaw_deg * (np.pi / 180.0)
        cy, sy = torch.cos(yr).unsqueeze(1), torch.sin(yr).unsqueeze(1)
        fwd = d[:, :, 0] * cy + d[:, :, 1] * sy
        left = -d[:, :, 0] * sy + d[:, :, 1] * cy
        up = d[:, :, 2]

        # per-point normalization by that point's OWN nominal span, so the fan
        # is scale-free: an on-route agent reads ~(1, 0, 0) at every horizon
        # regardless of speed, and the deviations are what carry information
        norm = torch.cat([torch.full_like(scale.unsqueeze(1), self.near_scale),
                          torch.clamp(span, min=self.near_scale)], dim=1)
        out = torch.stack([fwd / norm, left / norm, up / norm], dim=2)
        return torch.clamp(out, -self.clamp, self.clamp).reshape(p.shape[0], -1)

    def features_np(self, origin, yaw_deg, speed, device=None):
        """numpy-in convenience for the eval/record path (small batches)."""
        import torch
        dev = device if device is not None else self.pts.device
        o = torch.as_tensor(np.ascontiguousarray(origin), dtype=torch.float32,
                            device=dev)
        y = torch.as_tensor(np.ascontiguousarray(yaw_deg), dtype=torch.float32,
                            device=dev)
        s = torch.as_tensor(np.ascontiguousarray(speed), dtype=torch.float32,
                            device=dev)
        return self.features(o, y, s)
