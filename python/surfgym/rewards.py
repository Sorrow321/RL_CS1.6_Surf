"""Pluggable reward functions + spawn-pool builders (docs/03 basis, first
training task: docs/09 — maximize speed on a ramp).

Reward functions follow the ``SurfVecEnv`` hook contract::

    fn(prev_obs, obs, terminal_obs, base_rewards, done, trunc, core) -> (N,) f32

Observation indices used here (surfcore.h layout): obs[3] = horizontal speed
/ 1000. For ended envs, ``obs`` rows are the NEW episode's first obs — the
final-tick value lives in ``terminal_obs``.
"""
from __future__ import annotations

import numpy as np

from .core import STATE_DTYPE, SurfCore

__all__ = ["SpeedReward", "ProgressPlusSpeedReward", "ramp_spawn_pool"]


class SpeedReward:
    """``r_t = (h_speed_t − h_speed_{t−1}) * scale``.

    Telescopes over an episode to ``(final speed − spawn speed) * scale`` —
    literally "maximize speed at the horizon" — while staying dense.  Losing
    speed is negative reward; dying early forfeits all future gain.
    """

    def __init__(self, scale: float = 0.01) -> None:
        self.scale = float(scale)

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        ended = (done | trunc).astype(bool)
        cur = np.where(ended, terminal_obs[:, 3], obs[:, 3]) * 1000.0
        prev = prev_obs[:, 3] * 1000.0
        return ((cur - prev) * self.scale).astype(np.float32)


class ProgressPlusSpeedReward:
    """The core's spline-progress reward plus a speed-delta term — the natural
    next rung once waypoints exist: route-following that still prizes speed."""

    def __init__(self, speed_scale: float = 0.005) -> None:
        self._speed = SpeedReward(speed_scale)

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        return (base_rewards +
                self._speed(prev_obs, obs, terminal_obs, base_rewards, done, trunc, core)
                ).astype(np.float32)


def ramp_spawn_pool(
    core: SurfCore,
    grid: int = 48,
    nz_range: tuple[float, float] = (0.35, 0.68),
    height_above: float = 30.0,
    min_drop: float = 80.0,
    initial_speed: float = 0.0,
) -> np.ndarray:
    """Scan the map for surfable ramp faces and build a ``STATE_DTYPE`` spawn
    pool: one entry per found spot, placed ``height_above`` units over the
    ramp, yaw facing down-slope, optional initial speed along it.

    ``min_drop``: require that much open air below the scan point before the
    ramp hit (skips walkable slopes / clutter near floors).
    """
    from .core import SurfState

    mins, maxs = core.map_bounds()
    spots = []
    for ix in range(1, grid):
        for iy in range(1, grid):
            x = mins[0] + (maxs[0] - mins[0]) * (ix + 0.5) / grid
            y = mins[1] + (maxs[1] - mins[1]) * (iy + 0.5) / grid
            z = maxs[2] - 200.0
            while z > mins[2] + 200.0:                       # walk the whole column
                p = (x, y, z)
                if core.point_contents(p) == -1:             # CONTENTS_EMPTY
                    t0 = core.trace(p, p, hull=0)            # hull clearance
                    if not t0.startsolid:
                        tr = core.trace(p, (x, y, z - 600.0), hull=0)
                        if (tr.fraction < 1.0 and not tr.startsolid and
                                nz_range[0] < tr.normal[2] < nz_range[1] and
                                (z - tr.endpos[2]) > min_drop):
                            spots.append((tuple(tr.endpos), tuple(tr.normal)))
                            break                            # one spot per column
                        if tr.fraction < 1.0 and not tr.startsolid:
                            z = tr.endpos[2]                 # skip past this surface
                z -= 150.0
    if not spots:
        raise RuntimeError("ramp_spawn_pool: no surfable ramp faces found")

    # audition: keep only spawns that actually SLIDE (>=120 u/s within 80 ticks
    # of zero input) — filters roofs/clutter whose normal merely looks surfable
    pool_rows = []
    for end, n in spots:
        h = float(np.hypot(n[0], n[1])) or 1.0
        dx, dy = n[0] / h, n[1] / h                          # down-slope horizontal dir
        yaw = float(np.degrees(np.arctan2(dy, dx))) % 360.0
        st = SurfState()
        st.origin[0], st.origin[1], st.origin[2] = end[0], end[1], end[2] + height_above
        st.velocity[0], st.velocity[1] = dx * initial_speed, dy * initial_speed
        st.yaw = yaw
        st.onground = -1
        for _ in range(80):
            core.pm_step_usercmd(st, yaw, 0.0, 0.0, 0.0, 0, 10)
        if float(np.hypot(st.velocity[0], st.velocity[1])) < 120.0:
            continue
        pool_rows.append(((end[0], end[1], end[2] + height_above),
                          (dx * initial_speed, dy * initial_speed, 0.0), yaw))
    if not pool_rows:
        raise RuntimeError("ramp_spawn_pool: no candidate survived the slide audition")

    pool = np.zeros(len(pool_rows), dtype=STATE_DTYPE)
    for i, (origin, vel, yaw) in enumerate(pool_rows):
        pool[i]["origin"] = origin
        pool[i]["velocity"] = vel
        pool[i]["yaw"] = yaw
        pool[i]["onground"] = -1
    return pool
