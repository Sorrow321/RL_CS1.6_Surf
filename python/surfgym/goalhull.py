"""goalhull.py - the engine's brush-trigger test for a finish zone, in python, for graders and
analysis of recordings (the core runs the same rule: src/env.c goal_match_model / trigger_contains).

GoldSrc fires a brush trigger only with the player origin inside the trigger model's CLIP HULL
(hull 1 standing, hull 3 ducked) at the end of a move (SV_TouchLinks); the bounding box is only
the pre-check. zones.json's "end" on a type-1 map is the trigger_multiple's bounding box, so a
model that is not a plain box (surf_unitfarmer2's *23: three 1 u curtains whose box adds an empty
triangle) needs this test to tell a real finish from a box-only one (operator, 2026-10-31).

    gh = GoalHull.for_box("maps/surf_unitfarmer2.bsp", zones["end"])   # None = a plain box / no model
    gh.finished(rows, tick_s)        # did the recorded episode's finishing step enter the hull?
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_LUMP_PLANES, _LUMP_CLIPNODES, _LUMP_MODELS = 1, 9, 14
_EMPTY, _SOLID = -1, -2
_GROW = (16.0, 16.0, 36.0)              # the standing player's half extents (hull 1)


class GoalHull:
    def __init__(self, bsp_path, model: int):
        data = Path(bsp_path).read_bytes()
        lumps = [struct.unpack_from("<ii", data, 4 + 8 * i) for i in range(15)]
        off, ln = lumps[_LUMP_PLANES]
        self.planes = [struct.unpack_from("<4fi", data, off + 20 * i) for i in range(ln // 20)]
        off, ln = lumps[_LUMP_CLIPNODES]
        self.clip = [struct.unpack_from("<ihh", data, off + 8 * i) for i in range(ln // 8)]
        off, ln = lumps[_LUMP_MODELS]
        self.models = [struct.unpack_from("<9f4i3i", data, off + 64 * i) for i in range(ln // 64)]
        self.model = int(model)
        m = self.models[self.model]
        self.mins, self.maxs, self.origin, self.head = m[0:3], m[3:6], m[6:9], m[9:13]

    @classmethod
    def match(cls, bsp_path, box, tol: float = 0.5) -> "GoalHull | None":
        """The brush model whose bounds ARE the box (the trigger a type-1 zone was read from),
        or None (a padded button box, a hand-made box)."""
        g = cls(bsp_path, 0)
        lo, hi = box["mins"], box["maxs"]
        for i in range(1, len(g.models)):
            m = g.models[i]
            if all(abs(m[k] - lo[k]) <= tol and abs(m[3 + k] - hi[k]) <= tol for k in range(3)):
                g.model = i
                g.mins, g.maxs, g.origin, g.head = m[0:3], m[3:6], m[6:9], m[9:13]
                return g
        return None

    @classmethod
    def for_box(cls, bsp_path, box) -> "GoalHull | None":
        """The finish model when its clip hull is NOT a plain box (only then does the engine rule
        differ from the box), else None - the core's goal_match_model rule."""
        g = cls.match(bsp_path, box)
        return None if g is None or g.plain_box() else g

    def plain_box(self) -> bool:
        """Six axial planes on the faces of the bounds grown by the standing hull, one EMPTY
        side each: the clip hull IS the grown box (src/env.c model_is_plain_box)."""
        num, n, faces = self.head[1], 0, 0
        while num >= 0:
            n += 1
            if n > 6:
                return False
            pn, c0, c1 = self.clip[num]
            nx, ny, nz, dist, typ = self.planes[pn]
            if typ >= 3:
                return False
            lo, hi = self.mins[typ] - _GROW[typ], self.maxs[typ] + _GROW[typ]
            if abs(dist - lo) <= 0.01:
                faces |= 1 << (2 * typ)
            elif abs(dist - hi) <= 0.01:
                faces |= 2 << (2 * typ)
            else:
                return False
            if c0 == _EMPTY:
                num = c1
            elif c1 == _EMPTY:
                num = c0
            else:
                return False
        return n == 6 and num == _SOLID and faces == 0x3F

    def _contents(self, hull: int, p) -> int:
        num = self.head[hull]
        q = (p[0] - self.origin[0], p[1] - self.origin[1], p[2] - self.origin[2])
        while num >= 0:
            pn, c0, c1 = self.clip[num]
            nx, ny, nz, dist, typ = self.planes[pn]
            d = (q[typ] - dist) if typ < 3 else (nx * q[0] + ny * q[1] + nz * q[2] - dist)
            num = c0 if d >= 0 else c1
        return num

    def contains(self, p) -> bool:
        """Origin p inside the standing (hull 1) or the ducked (hull 3) clip hull - a recording
        does not say which stance the player was in."""
        return self._contents(1, p) == _SOLID or self._contents(3, p) == _SOLID

    def finished(self, rows: np.ndarray, tick_s: float, tail: int = 3, slack: float = 1.0) -> bool:
        """Did an episode the recorder ended "done" enter the hull? Rows are [tick, x, y, z, vx,
        vy, vz, ...]; the last row is the state BEFORE the finishing tick, so its end point is
        extrapolated (one tick of velocity; gravity moves it < 0.05 u) and tested with a +-slack
        neighbourhood, together with the last `tail` recorded rows."""
        r = np.asarray(rows, np.float64)
        for p in r[-tail:, 1:4]:
            if self.contains(p):
                return True
        q = r[-1, 1:4] + r[-1, 4:7] * float(tick_s)
        for dx, dy, dz in ((0, 0, 0), (slack, 0, 0), (-slack, 0, 0), (0, slack, 0),
                           (0, -slack, 0), (0, 0, slack), (0, 0, -slack)):
            if self.contains((q[0] + dx, q[1] + dy, q[2] + dz)):
                return True
        return False
