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


def _states(core: SurfCore) -> np.ndarray:
    """Zero-copy view when the DLL provides it, else a copy."""
    try:
        return core.states_view
    except Exception:
        return core.get_states()


__all__ = ["SpeedReward", "AvgSpeedReward", "ForwardProgressReward",
           "PathLengthReward", "ProgressPlusSpeedReward", "BlendedReward",
           "ramp_spawn_pool", "platform_spawn_pool", "drop_spawn_pool"]


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


class AvgSpeedReward:
    """``r_t = h_speed_t * scale`` — maximizes AVERAGE speed (~distance).

    Denser shaping than the terminal telescope: being fast mid-episode pays
    even if the run later ends in the trough, so "stay on the ramp" gets a
    gradient long before a full 5-second hold is discovered. q1physrl's
    reward was this family."""

    def __init__(self, scale: float = 0.0005) -> None:
        self.scale = float(scale)

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        ended = (done | trunc).astype(bool)
        cur = np.where(ended, terminal_obs[:, 3], obs[:, 3]) * 1000.0
        return (cur * self.scale).astype(np.float32)


class ForwardProgressReward:
    """Maximize horizontal distance along each episode's spawn-facing yaw.

    ``mode="max"`` (default): ``r_t = relu(proj_t − best_proj) * scale`` —
    only NEW forward maxima pay, telescoping to the episode's furthest point.
    A great run followed by a fall or a jail teleport keeps its full credit
    (the retreat earns 0, not negative) — matching "best distance reached",
    which is what we actually want optimized.

    ``mode="net"``: signed delta (telescopes to FINAL position). Kept for
    comparison; punishes post-run teleports/retreats retroactively.

    Teleports never pay: any per-tick displacement above ``max_step`` (the
    physics ceiling is ~28u/tick horizontal) shifts the reference frame by
    the jump, so a forward map teleport earns 0 that tick and later progress
    is measured from the destination — without this, a stage-advance pad
    that lands thousands of units down-course pays more in one tick than
    surfing the whole lane. Backward teleports were already safe under
    ``mode="max"`` (relu of a retreat is 0).

    Ungameable by in-place tricks either way: pogo/circles earn ~0. Uses
    ``core.get_states()`` per tick (absolute positions aren't in obs);
    anchors re-snapshot automatically on autoreset."""

    def __init__(self, scale: float = 0.01, mode: str = "max",
                 max_step: float = 50.0) -> None:
        assert mode in ("max", "net")
        self.scale = float(scale)
        self.mode = mode
        self.max_step = float(max_step)
        self._dir: np.ndarray | None = None      # (N,2) unit forward per env
        self._ref: np.ndarray | None = None      # (N,2) spawn xy
        self._proj: np.ndarray | None = None     # (N,) last/best projection
        self._prev: np.ndarray | None = None     # (N,2) last tick's xy

    def _snapshot(self, states, idx) -> None:
        yaw = np.radians(states["yaw"][idx].astype(np.float64))
        self._dir[idx, 0] = np.cos(yaw)
        self._dir[idx, 1] = np.sin(yaw)
        self._ref[idx] = states["origin"][idx, :2]
        self._proj[idx] = 0.0
        self._prev[idx] = states["origin"][idx, :2]

    def on_reset(self, core) -> None:
        n = core.num_envs
        self._dir = np.zeros((n, 2), np.float64)
        self._ref = np.zeros((n, 2), np.float64)
        self._proj = np.zeros(n, np.float64)
        self._prev = np.zeros((n, 2), np.float64)
        self._snapshot(_states(core), np.arange(n))

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        states = _states(core)
        if self._dir is None:
            self.on_reset(core)
            return np.zeros(len(done), np.float32)
        pos = states["origin"][:, :2].astype(np.float64)
        delta = pos - self._prev
        tel = np.hypot(delta[:, 0], delta[:, 1]) > self.max_step
        if tel.any():                            # teleport: carry the frame along
            self._ref[tel] += delta[tel]
        self._prev = pos
        d = pos - self._ref
        proj = d[:, 0] * self._dir[:, 0] + d[:, 1] * self._dir[:, 1]
        if self.mode == "max":
            r = np.maximum(proj - self._proj, 0.0) * self.scale
            self._proj = np.maximum(self._proj, proj)
        else:
            r = (proj - self._proj) * self.scale
            self._proj = proj
        ended = (done | trunc).astype(bool)
        if ended.any():
            # states for ended envs are already the NEW episode's spawn:
            # drop the bogus cross-episode delta and re-anchor
            r[ended] = 0.0
            self._snapshot(states, np.flatnonzero(ended))
        return r.astype(np.float32)


class PathLengthReward:
    """``r_t = |Δxy| * scale`` — total HORIZONTAL distance traveled.

    Direction-agnostic: rewards sustained movement anywhere (ramp transfers,
    turns, loops), unlike ForwardProgressReward's single spawn-facing axis.
    Horizontal only — falling must not farm free vertical "distance".

    Teleports are filtered, not rewarded: the maximum legitimate move is
    ``sv_maxvelocity * frametime`` per axis (~28u horizontal per tick), so any
    per-tick displacement above ``max_step`` (default 50u) is a teleport and
    counts as zero. The relocation itself still happens (map behavior stays
    authentic) — the agent just can't cash it in, and time lost in jail is
    its own penalty."""

    def __init__(self, scale: float = 0.01, max_step: float = 50.0) -> None:
        self.scale = float(scale)
        self.max_step = float(max_step)
        self._pos: np.ndarray | None = None

    def on_reset(self, core) -> None:
        self._pos = _states(core)["origin"][:, :2].astype(np.float64)

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        states = _states(core)
        pos = states["origin"][:, :2].astype(np.float64)
        if self._pos is None:
            self._pos = pos
            return np.zeros(len(done), np.float32)
        step = np.hypot(pos[:, 0] - self._pos[:, 0], pos[:, 1] - self._pos[:, 1])
        step[step > self.max_step] = 0.0            # teleport: not travel
        ended = (done | trunc).astype(bool)
        step[ended] = 0.0                           # autoreset jump: not travel
        self._pos = pos
        return (step * self.scale).astype(np.float32)


class BlendedReward:
    """Curriculum blend of two reward fns: ``r = (1−w)·a + w·b`` with ``w``
    annealed linearly from 0 to 1 over global env-steps ``[t0, t1]``.

    First teach the narrow skill (e.g. ForwardProgressReward down the spawn
    lane), then hand over to the open-ended objective (e.g. PathLengthReward
    — surf as far as possible, anywhere) without a reward cliff the value
    function would have to relearn from scratch.

    Step tracking: ``__call__`` self-counts env-steps (len(done) per tick), so
    it works under any trainer; a trainer that knows better (checkpoint
    resume!) should call ``set_step(global_step)`` each iteration, which
    switches to external authority permanently. Both children are evaluated
    every tick regardless of ``w`` so their internal anchors stay live across
    the phase boundary."""

    def __init__(self, a, b, t0: float = 100e6, t1: float = 200e6) -> None:
        assert t1 > t0 >= 0
        self.a, self.b = a, b
        self.t0, self.t1 = float(t0), float(t1)
        self._step = 0.0
        self._external = False

    @property
    def weight(self) -> float:
        return min(1.0, max(0.0, (self._step - self.t0) / (self.t1 - self.t0)))

    def set_step(self, global_step: int) -> None:
        self._step = float(global_step)
        self._external = True

    def on_reset(self, core) -> None:
        self.a.on_reset(core)
        self.b.on_reset(core)

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        ra = self.a(prev_obs, obs, terminal_obs, base_rewards, done, trunc, core)
        rb = self.b(prev_obs, obs, terminal_obs, base_rewards, done, trunc, core)
        if not self._external:
            self._step += len(done)
        w = self.weight
        return ((1.0 - w) * ra + w * rb).astype(np.float32)


class ProgressPlusSpeedReward:
    """The core's spline-progress reward plus a speed-delta term — the natural
    next rung once waypoints exist: route-following that still prizes speed."""

    def __init__(self, speed_scale: float = 0.005) -> None:
        self._speed = SpeedReward(speed_scale)

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        return (base_rewards +
                self._speed(prev_obs, obs, terminal_obs, base_rewards, done, trunc, core)
                ).astype(np.float32)


def platform_spawn_pool(
    core: SurfCore,
    edge_back: float = 35.0,
    probe_dirs: int = 24,
    max_walk: float = 700.0,
    drop_min: float = 100.0,
    nz_range: tuple[float, float] = (0.35, 0.68),
) -> np.ndarray:
    """Game-authentic spawns: the map's real start platform, standing near the
    edge, facing a ramp. The agent must walk/jump off, then strafe + steer.

    For each map spawn point and each probe direction: march outward to the
    platform edge (floor drops > ``drop_min``), require a surfable face below
    the far side, then place the spawn ``edge_back`` units before the edge,
    grounded, yaw facing out. Audition: holding +forward for 300 ticks from
    there must reach 120 u/s (i.e. walking off really lands on a ramp).
    """
    from .core import SurfState

    rows = []
    for origin, _syaw in core.spawns():
        # ground the reference point
        t0 = core.trace(origin, (origin[0], origin[1], origin[2] - 200.0), hull=0)
        if t0.fraction >= 1.0 or t0.startsolid:
            continue
        gx, gy, gz = float(t0.endpos[0]), float(t0.endpos[1]), float(t0.endpos[2])
        for di in range(probe_dirs):
            ang = 2.0 * np.pi * di / probe_dirs
            dx, dy = float(np.cos(ang)), float(np.sin(ang))
            edge_d = None
            d = 25.0
            while d <= max_walk:
                p = (gx + dx * d, gy + dy * d, gz + 20.0)
                tr = core.trace(p, (p[0], p[1], p[2] - drop_min - 40.0), hull=0)
                if tr.fraction >= 1.0:              # floor fell away: the edge
                    edge_d = d
                    break
                if tr.startsolid:                    # wall: dead direction
                    break
                d += 25.0
            if edge_d is None:
                continue
            # a surfable face must catch the fall beyond the edge
            q = (gx + dx * (edge_d + 60.0), gy + dy * (edge_d + 60.0), gz + 20.0)
            tq = core.trace(q, (q[0], q[1], q[2] - 900.0), hull=0)
            if (tq.fraction >= 1.0 or tq.startsolid or
                    not (nz_range[0] < tq.normal[2] < nz_range[1])):
                continue
            # spawn: settled on the platform, edge_back before the drop
            sx, sy = gx + dx * (edge_d - edge_back), gy + dy * (edge_d - edge_back)
            ts = core.trace((sx, sy, gz + 30.0), (sx, sy, gz - 60.0), hull=0)
            if ts.fraction >= 1.0 or ts.startsolid or ts.normal[2] < 0.7:
                continue
            yaw = float(np.degrees(np.arctan2(dy, dx))) % 360.0
            st = SurfState()
            st.origin[0], st.origin[1], st.origin[2] = sx, sy, float(ts.endpos[2])
            st.yaw = yaw
            st.onground = -1
            peak = 0.0
            for _ in range(300):                     # audition: hold +forward
                core.pm_step_usercmd(st, yaw, 0.0, 400.0, 0.0, 0, 10)
                peak = max(peak, float(np.hypot(st.velocity[0], st.velocity[1])))
            if peak < 120.0:
                continue
            rows.append(((sx, sy, float(ts.endpos[2])), yaw))
    if not rows:
        raise RuntimeError("platform_spawn_pool: no edge-facing-ramp spawn found")

    pool = np.zeros(len(rows), dtype=STATE_DTYPE)
    for i, (origin, yaw) in enumerate(rows):
        pool[i]["origin"] = origin
        pool[i]["yaw"] = yaw
        pool[i]["onground"] = -1
    return pool


def _scan_ramp_faces(core: SurfCore, grid: int, nz_range, min_drop: float):
    """Grid-scan the map for surfable ramp faces; returns [(endpos, normal)]."""
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
        raise RuntimeError("ramp scan: no surfable ramp faces found")
    return spots


def drop_spawn_pool(
    core: SurfCore,
    h_range: tuple[float, float] = (400.0, 800.0),
    speed_range: tuple[float, float] = (100.0, 400.0),
    pitch_range: tuple[float, float] = (-45.0, 15.0),
    variants: int = 6,
    grid: int = 48,
    nz_range: tuple[float, float] = (0.35, 0.68),
    min_drop: float = 80.0,
    seed: int = 17,
) -> np.ndarray:
    """Exploration spawn pool: high drops onto surfable faces with randomized
    entry state. For each scanned ramp face, ``variants`` entries are drawn
    with height ~ U(h_range) above the face, a random-direction horizontal
    velocity ~ U(speed_range), uniform random yaw, and view pitch ~
    U(pitch_range) (meaningful under fixed-gaze mode, where pitch stays at
    its spawn value). Each candidate is auditioned no-input for the full
    fall + slide; it must end still inside the map and moving >= 120 u/s
    horizontally — flat-floor landings, void falls and stuck spawns fail.
    """
    from .core import SurfState

    mins, maxs = core.map_bounds()
    spots = _scan_ramp_faces(core, grid, nz_range, min_drop)
    rng = np.random.default_rng(seed)
    rows = []
    max_aud = int(100 * np.sqrt(2 * h_range[1] / 800.0)) + 200
    for end, _n in spots:
        for _ in range(variants):
            h = float(rng.uniform(*h_range))
            spd = float(rng.uniform(*speed_range))
            ang = float(rng.uniform(0, 2 * np.pi))
            yaw = float(rng.uniform(0, 360))
            pitch = float(rng.uniform(*pitch_range))
            oz = end[2] + h
            if oz > maxs[2] - 64:
                oz = maxs[2] - 64
            if oz - end[2] < h_range[0] * 0.5:
                continue                             # ceiling ate the drop
            # structural checks only — no passive-fall audition: the agent
            # has ~2s of air-strafe authority (hundreds of units of steering)
            # to convert a drop above a known surfable face, and unconvertible
            # rolls end cheaply under the teleport rule. A ragdoll test
            # rejected ~99.6% of learnable situations.
            t0 = core.trace((end[0], end[1], oz), (end[0], end[1], oz), hull=0)
            if t0.startsolid:
                continue                             # spawn inside geometry
            if core.point_contents((end[0], end[1], oz)) == -3:
                continue                             # CONTENTS_WATER
            tdn = core.trace((end[0], end[1], oz),
                             (end[0], end[1], oz - h_range[0] * 0.5), hull=0)
            if tdn.fraction < 1.0:
                continue                             # ledge right beneath spawn
            rows.append(((end[0], end[1], oz),
                         (np.cos(ang) * spd, np.sin(ang) * spd, 0.0),
                         yaw, pitch))
    if not rows:
        raise RuntimeError("drop_spawn_pool: no candidate survived audition")
    pool = np.zeros(len(rows), dtype=STATE_DTYPE)
    for i, (origin, vel, yaw, pitch) in enumerate(rows):
        pool[i]["origin"] = origin
        pool[i]["velocity"] = vel
        pool[i]["yaw"] = yaw
        pool[i]["pitch"] = pitch
        pool[i]["onground"] = -1
    return pool


def ramp_spawn_pool(
    core: SurfCore,
    grid: int = 48,
    nz_range: tuple[float, float] = (0.35, 0.68),
    height_above: float = 30.0,
    min_drop: float = 80.0,
    initial_speed: float = 0.0,
    audition_ticks: int = 80,
) -> np.ndarray:
    """Scan the map for surfable ramp faces and build a ``STATE_DTYPE`` spawn
    pool: one entry per found spot, placed ``height_above`` units over the
    ramp, yaw facing down-slope, optional initial speed along it.

    ``min_drop``: require that much open air below the scan point before the
    ramp hit (skips walkable slopes / clutter near floors).
    """
    from .core import SurfState

    spots = _scan_ramp_faces(core, grid, nz_range, min_drop)

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
        for _ in range(audition_ticks):
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
