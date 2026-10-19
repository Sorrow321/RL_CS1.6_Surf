"""``--aug-mirror`` / ``--eval-mirror``: fly the LEFT-RIGHT MIRROR IMAGE of the map.

The user (2026-10-16): "mirror the image left to the right" - an augmentation
against a policy that memorises every point of view of its one map.

Flipping the IMAGE alone would be wrong: the policy would see a left turn and
be paid for steering left into a right turn. What makes the mirror exact is
that GoldSrc's movement physics is REFLECTION-SYMMETRIC. src/pm.c builds the
wish direction as ``forward * fmove + right * smove`` with, at roll 0,
``forward = (cos yaw, sin yaw)`` and ``right = (sin yaw, -cos yaw)``. Reflect
the world in a vertical plane (y -> -y) and yaw -> -yaw maps ``forward`` onto
its reflection while ``right`` lands on MINUS its reflection - so the
reflected trajectory is the one flown with the yaw AND the side key
reflected. Gravity, friction, air acceleration and every clip against a
(reflected) plane commute with the reflection.

So a policy that SEES the reflected observation, and whose yaw and side
commands are reflected on their way to the engine, is - exactly - an agent
flying the mirror-image map: the same ramps at the same speeds, every turn
the other way round. Nothing in the engine, the reward, the novelty counts or
the reservoir changes: the real state is always the real map, the geodesic
field is reflection-invariant, and a mirrored episode that finishes flew the
real line, perceived the other way round. The mirror image is therefore a
genuine second map with exactly the first one's difficulty - and, for a
policy that never trained on it, a held-out map (``--eval-mirror``,
``record_ckpt.py --mirror``) whose physics, route and metrics are all the
original's.

What the reflection touches:

* **the image** - columns reversed. The camera's rays sit at
  ``hfov * (0.5 - (i + 0.5) / W)`` (surfgym/vision.py), symmetric about the
  view direction, so column ``i <-> W - 1 - i`` is the exact reflection; and
  every channel that carries a LEFT component is negated (``--obs-normal``:
  the ego-frame normal is (x forward, y left, z up)).
* **the core scalars** (src/env.c write_obs): 1 = lateral velocity in the
  view frame, 10 = the last yaw delta. Under ``--gps`` also 7 = sin(yaw) and
  13 = (y - map centre) / 2000 - the reflection in the plane y = centre.
* **the arrow** of ``--ramp-obs-vec``: its LEFT component.
* **the engine row** - the side key's bins ``{-400, 0, +400}[i]`` become
  ``2 - i``; the view command's yaw is negated (an offset from the velocity
  heading under ``--view-absolute velocity``, a world heading under
  ``world``, a per-tick multiple under the delta command: all three reflect
  by a sign - src/env.c surf_yaw_delta_abs / surf_yaw_delta_cont).

The held keys of ``--keys-hold`` stay in the POLICY's frame: the policy chose
them looking at the mirror image, so they are what it believes it holds, and
only the row leaving for the engine is reflected.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["Mirror", "H_SIDE", "SCAL_NEG", "SCAL_NEG_GPS", "mirror_actions_"]

#: the engine's side-key head (surfcore.h: a[3] = {-400, 0, +400}[i])
H_SIDE = 3

#: core scalar columns whose sign a left-right reflection flips (src/env.c
#: write_obs): 1 = lateral velocity in the view frame, 10 = last yaw delta
SCAL_NEG = (1, 10)

#: ... and with --gps (absolute heading + position re-included): 7 = sin(yaw),
#: 13 = (y - map centre) / 2000
SCAL_NEG_GPS = (7, 13)


def mirror_actions_(act, view=None, mask=None):
    """Reflect ENGINE-space action rows IN PLACE (host numpy).

    ``act`` (n, 6) integer, the row the engine is about to receive (after any
    ``--keys-hold`` resolve); ``view`` (n, 2) float32 view command or None;
    ``mask`` (n,) bool of the rows to reflect, None = every row. Returns
    ``act``.
    """
    rows = slice(None) if mask is None else np.flatnonzero(np.asarray(mask, bool))
    if mask is not None and rows.size == 0:
        return act
    act[rows, H_SIDE] = 2 - act[rows, H_SIDE]
    if view is not None:
        view[rows, 0] = -view[rows, 0]
    return act


class Mirror:
    """The observation half of the reflection, for one row layout.

    A row is ``[scalar half | image]`` with the image ``frame_h x frame_w x
    channels`` channel-fastest (NHWC) in its LAST ``frame_h * frame_w *
    channels`` columns - the layout the renderer emits and Policy.features
    restrides. ``neg_cols`` are scalar-half columns, ``neg_ch`` image
    channels, whose sign the reflection flips.
    """

    def __init__(self, frame_h: int, frame_w: int, channels: int,
                 neg_cols=SCAL_NEG, neg_ch=()):
        self.h, self.w, self.c = int(frame_h), int(frame_w), int(channels)
        self.frame = self.h * self.w * self.c
        self.neg_cols = tuple(sorted({int(c) for c in neg_cols}))
        self.neg_ch = tuple(sorted({int(c) for c in neg_ch}))
        if any(not 0 <= c < self.c for c in self.neg_ch):
            raise ValueError(f"mirror: image channels {self.neg_ch} outside 0..{self.c - 1}")
        self._signs = {}

    def _sign(self, n, cols, ref):
        key = (n, cols, ref.device, ref.dtype)
        s = self._signs.get(key)
        if s is None:
            s = torch.ones(n, device=ref.device, dtype=ref.dtype)
            s[list(cols)] = -1.0
            self._signs[key] = s
        return s

    def rows(self, x):
        """The mirror image of every row of ``x`` (B, scal + frame), as a NEW
        tensor; an involution (``rows(rows(x)) == x`` exactly)."""
        b, d = x.shape
        ns = d - self.frame
        if ns < 0:
            raise ValueError(f"mirror: a row of {d} columns cannot hold a {self.h}x{self.w}x"
                             f"{self.c} image")
        if self.neg_cols and max(self.neg_cols) >= ns:
            raise ValueError(f"mirror: scalar column {max(self.neg_cols)} is not in the "
                             f"{ns}-wide scalar half")
        img = x[:, ns:].reshape(b, self.h, self.w, self.c).flip(2)
        if self.neg_ch:
            img = img * self._sign(self.c, self.neg_ch, x)
        scal = x[:, :ns]
        if self.neg_cols:
            scal = scal * self._sign(ns, self.neg_cols, x)
        return torch.cat([scal, img.reshape(b, self.frame)], dim=1)

    def obs_(self, x, mask=None):
        """Reflect the rows of ``x`` IN PLACE where ``mask`` ((B,) bool
        tensor) is True; ``mask`` None reflects every row. Returns ``x``."""
        m = self.rows(x)
        if mask is None:
            x.copy_(m)
        else:
            x.copy_(torch.where(mask.reshape(-1, 1), m, x))
        return x

    def describe(self) -> str:
        return (f"image {self.h}x{self.w}x{self.c} columns reversed"
                + (f", channels {list(self.neg_ch)} negated" if self.neg_ch else "")
                + f"; scalar columns {list(self.neg_cols)} negated; engine side key "
                  f"i -> 2 - i, view yaw -> -yaw")


def layout_for(lidar, gps: bool = False, rvec0: int = -1):
    """``Mirror`` for a renderer's row: its frame and channels, the LEFT
    channels of its normal block, and the scalar columns of the core (plus
    ``--gps``'s and the ``--ramp-obs-vec`` arrow's when present).

    Refuses the renderers whose image is not one NHWC camera frame."""
    views = int(getattr(lidar, "views", 1) or 1)
    if views > 1:
        raise ValueError("mirror: --obs-views rings are not one camera frame")
    h, w = int(lidar.H), int(lidar.W)
    c = int(lidar.channels)
    neg_ch = []
    if getattr(lidar, "normals", False):
        neg_ch.append(2)                      # --normals: (depth, nx, ny, nz)
    if (getattr(lidar, "texmap", None) is not None
            and getattr(lidar, "texture_mode", "") == "normal"):
        neg_ch.append(int(lidar.tex_channel) + 1)    # --obs-normal: (x fwd, y left, z up)
    neg_cols = list(SCAL_NEG) + (list(SCAL_NEG_GPS) if gps else [])
    if rvec0 >= 0:
        neg_cols.append(int(rvec0) + 1)       # the arrow: (forward, left, up, length)
    return Mirror(h, w, c, neg_cols=neg_cols, neg_ch=neg_ch)
