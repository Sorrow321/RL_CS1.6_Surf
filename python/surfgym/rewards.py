"""Pluggable reward functions + spawn-pool builders (docs/03 basis, first
training task: docs/09 — maximize speed on a ramp).

Reward functions follow the ``SurfVecEnv`` hook contract::

    fn(prev_obs, obs, terminal_obs, base_rewards, done, trunc, core) -> (N,) f32

Observation indices used here (surfcore.h layout): obs[3] = horizontal speed
/ 1000. For ended envs, ``obs`` rows are the NEW episode's first obs — the
final-tick value lives in ``terminal_obs``.
"""
from __future__ import annotations

import math

import numpy as np

from .core import STATE_DTYPE, SurfCore
from .dipmeter import DipMeter


def _states(core: SurfCore) -> np.ndarray:
    """Zero-copy view when the DLL provides it, else a copy."""
    try:
        return core.states_view
    except Exception:
        return core.get_states()


__all__ = ["SpeedReward", "AvgSpeedReward", "ForwardProgressReward",
           "PathLengthReward", "ProgressPlusSpeedReward", "BlendedReward",
           "MaxSpeedReward", "CoverageSpeedReward", "AcroCoverageReward",
           "RaceReward",
           "ramp_spawn_pool", "platform_spawn_pool", "drop_spawn_pool",
           "map_spawn_pool"]


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


class MaxSpeedReward:
    """``r_t = relu(speed_t − best_so_far) * scale`` — telescopes to the
    episode's TOP speed (only new personal bests pay, retreats cost 0).

    Horizontal speed by default: a 3D maximum is farmable by free-falling
    (gravity donates ~1800 u/s per 2000u of drop); horizontal speed must be
    earned on ramps. Anchors re-snapshot automatically on autoreset."""

    def __init__(self, scale: float = 0.05, horizontal: bool = True) -> None:
        self.scale = float(scale)
        self.horizontal = bool(horizontal)
        self._best: np.ndarray | None = None

    def _speed(self, states) -> np.ndarray:
        v = states["velocity"].astype(np.float64)
        if self.horizontal:
            return np.hypot(v[:, 0], v[:, 1])
        return np.sqrt((v * v).sum(axis=1))

    def on_reset(self, core) -> None:
        self._best = self._speed(_states(core))

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        states = _states(core)
        if self._best is None:
            self.on_reset(core)
            return np.zeros(len(done), np.float32)
        s = self._speed(states)
        r = np.maximum(s - self._best, 0.0) * self.scale
        self._best = np.maximum(self._best, s)
        ended = (done | trunc).astype(bool)
        if ended.any():
            # states for ended envs are already the NEW episode's spawn
            r[ended] = 0.0
            self._best[ended] = s[ended]
        return r.astype(np.float32)


class CoverageSpeedReward:
    """Cinematic objective: ``r_t = h_speed_t * scale`` ONLY when entering a
    map cell not yet visited this episode; revisits pay zero.

    Coverage isn't a bonus term — it gates the speed reward. Loops die (their
    cells are spent after one pass), strolling through fresh cells pays
    crumbs, and ramp-to-ramp flight across chains of untouched air cells at
    full speed is the highest-paying move in the game. Horizontal speed so
    dive-bombing stays worthless. Per-episode novelty: the visited set clears
    on autoreset, so every episode must re-earn its route."""

    def __init__(self, scale: float = 0.001, cell: float = 512.0,
                 columns: bool = False, revisit_pen: float = 0.25) -> None:
        self.scale = float(scale)
        self.cell = float(cell)
        # 512u 3D voxels: big enough that one ramp corridor is only a handful
        # of cells (the 256u-3D altitude-farming leak), 3D because the map
        # stacks elements vertically (columns=True conflates layers).
        self.columns = bool(columns)
        # entering an ALREADY-visited cell costs revisit_pen (transitions
        # only — no per-tick bleed, which would make dying preferable to
        # crossing spent ground). Kept small vs the ~1.0/new-cell income so
        # journeys through old territory to fresh frontiers stay net-positive.
        self.revisit_pen = float(revisit_pen)
        self._visited: np.ndarray | None = None
        self._prev_cell: np.ndarray | None = None
        self._mins = None
        self._dims = None

    def _cells(self, states) -> np.ndarray:
        p = states["origin"].astype(np.float64)
        ix = np.clip(((p[:, 0] - self._mins[0]) // self.cell).astype(np.int64),
                     0, self._dims[0] - 1)
        iy = np.clip(((p[:, 1] - self._mins[1]) // self.cell).astype(np.int64),
                     0, self._dims[1] - 1)
        if self.columns:
            return ix + self._dims[0] * iy
        iz = np.clip(((p[:, 2] - self._mins[2]) // self.cell).astype(np.int64),
                     0, self._dims[2] - 1)
        return ix + self._dims[0] * (iy + self._dims[1] * iz)

    def on_reset(self, core) -> None:
        mins, maxs = core.map_bounds()
        self._mins = mins.astype(np.float64)
        self._dims = tuple(int(np.ceil((maxs[i] - mins[i]) / self.cell)) + 1
                           for i in range(3))
        n = core.num_envs
        ncells = self._dims[0] * self._dims[1] * (1 if self.columns
                                                 else self._dims[2])
        self._visited = np.zeros((n, ncells), dtype=bool)
        rows = np.arange(n)
        c0 = self._cells(_states(core))
        self._visited[rows, c0] = True          # spawn cell free
        self._prev_cell = c0.copy()

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        states = _states(core)
        if self._visited is None:
            self.on_reset(core)
            return np.zeros(len(done), np.float32)
        rows = np.arange(len(done))
        cell = self._cells(states)
        moved = cell != self._prev_cell
        new = moved & ~self._visited[rows, cell]
        revisit = moved & self._visited[rows, cell] & ~new
        v = states["velocity"]
        r = (np.hypot(v[:, 0], v[:, 1]) * new * self.scale
             - self.revisit_pen * revisit)
        self._visited[rows, cell] = True
        self._prev_cell = cell.copy()
        ended = (done | trunc).astype(bool)
        if ended.any():
            # states for ended envs are already the NEW episode's spawn:
            # wipe their visited set and comp the spawn cell
            r[ended] = 0.0
            self._visited[ended] = False
            ei = np.flatnonzero(ended)
            self._visited[ei, cell[ei]] = True
        return r.astype(np.float32)


class AcroCoverageReward(CoverageSpeedReward):
    """Coverage income + style bonuses for cinematic flight.

    - Air spins: yaw rotation while fully airborne is credited at most
      ``spin_rate_cap`` deg/tick — spinning faster than ~600 deg/s earns
      nothing extra, so the optimal trick LOOKS human. Each full 360 (up to
      ``max_spins`` per flight) banks ``spin_bonus``, paid only on a survived
      landing after a real flight (>= ``min_air`` ticks): no reward for
      spinning into the void.
    - Switch landings: touching down from a real flight at >= 300 u/s with
      yaw within +-40 deg of BACKWARDS pays ``switch_bonus``.
    """

    def __init__(self, scale: float = 0.001, cell: float = 512.0,
                 revisit_pen: float = 1.0, spin_bonus: float = 1.5,
                 switch_bonus: float = 2.0, spin_rate_cap: float = 6.0,
                 max_spins: int = 2, min_air: int = 30,
                 back_bonus: float = 0.01) -> None:
        super().__init__(scale, cell, columns=False, revisit_pen=revisit_pen)
        self.spin_bonus = float(spin_bonus)
        self.switch_bonus = float(switch_bonus)
        self.spin_rate_cap = float(spin_rate_cap)
        self.max_spins = int(max_spins)
        self.min_air = int(min_air)
        # per-tick pay for SUSTAINED backwards ramp-riding (yaw opposite
        # travel while surfing) — the ramp-riding state is the non-ballistic
        # airborne one: velocity clips against the face every tick
        self.back_bonus = float(back_bonus)

    def on_reset(self, core) -> None:
        super().on_reset(core)
        st = _states(core)
        n = core.num_envs
        ph = core.config.phys
        # the MEAN tick under a --tick-ms pattern (8,8,7 -> 7.667 ms); a core
        # without the attribute reads as its config msec, as before
        _tick = float(getattr(core, "tick_ms", ph.msec))
        self._g_tick = float(ph.sv_gravity) * _tick * 1e-3
        self._prev_yaw = st["yaw"].astype(np.float64)
        self._prev_v = st["velocity"].astype(np.float64).copy()
        self._streak = np.zeros(n, np.int64)     # consecutive BALLISTIC ticks
        self._spin_acc = np.zeros(n, np.float64)

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc, core):
        r = super().__call__(prev_obs, obs, terminal_obs, base_rewards,
                             done, trunc, core)
        states = _states(core)
        yaw = states["yaw"].astype(np.float64)
        v = states["velocity"].astype(np.float64)
        hspd = np.hypot(v[:, 0], v[:, 1])

        # surf subtlety: riding a ramp KEEPS onground == -1 (that IS the surf
        # mechanic), so "airborne" conflates flight and ramp-riding. True
        # ballistic flight = velocity evolved by gravity alone (+ small
        # air-strafe increments); a ramp catch is a large velocity clip.
        dv = v - self._prev_v
        contact = ((np.abs(dv[:, 0]) + np.abs(dv[:, 1]) > 40.0)
                   | (np.abs(dv[:, 2] + self._g_tick) > 12.0))
        grounded = states["onground"] != -1
        ballistic = ~grounded & ~contact

        dyaw = np.abs((yaw - self._prev_yaw + 180.0) % 360.0 - 180.0)
        # a flight ends when a real flight streak hits a surface (ramp catch
        # or floor) — pay the banked style THEN. Spin pay is CONTINUOUS in
        # rotation (measured policies rotate 25-305 deg per flight — an
        # all-or-nothing 360 gate paid exactly zero times, no gradient)
        flight_end = (self._streak >= self.min_air) & (contact | grounded)
        spin_frac = np.minimum(self._spin_acc, 360.0 * self.max_spins) / 360.0
        style = self.spin_bonus * spin_frac * flight_end
        heading = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        yaw_vs_travel = np.abs((yaw - heading + 180.0) % 360.0 - 180.0)
        style = style + self.switch_bonus * (flight_end & (hspd >= 300.0)
                                             & (yaw_vs_travel >= 140.0))
        # sustained backwards surfing: riding a ramp (airborne + clipping)
        # fast, with yaw within 40 deg of opposite to travel
        style = style + self.back_bonus * (~grounded & contact
                                           & (hspd >= 500.0)
                                           & (yaw_vs_travel >= 140.0))

        # trackers: spin credit accrues only across consecutive ballistic
        # ticks; everything clears once the flight is over
        self._spin_acc = np.where(
            ballistic,
            self._spin_acc + np.minimum(dyaw, self.spin_rate_cap)
            * (self._streak > 0),
            0.0)
        self._streak = np.where(ballistic, self._streak + 1, 0)
        self._prev_yaw = yaw
        self._prev_v = v.copy()

        ended = (done | trunc).astype(bool)
        if ended.any():
            # states for ended envs are the NEW episode's spawn: no
            # cross-episode landings, trackers re-anchor
            style[ended] = 0.0
            ei = np.flatnonzero(ended)
            self._prev_yaw[ei] = yaw[ei]
            self._prev_v[ei] = v[ei]
            self._streak[ei] = 0
            self._spin_acc[ei] = 0.0
        return (r + style.astype(np.float32)).astype(np.float32)


# --------------------------------------------------------------------------
# --curiosity-cond (docs/curiosity_cond.md): the per-env exploration weight
# T of an Agent57-style family (Badia et al. 2020). ONE network reads its T
# through one observation column t = log1p(T) / log1p(T_max); every env
# draws its own T at every episode start (spawn AND respawn) from a mixture
# - T = 0 with probability p0, else log-uniform on [tmin, tmax] - and is paid
# a per-env mix of the race reward: the shaping term x (1 - T/T_max), the
# count bonus x T, the success bonus and the fail penalty unchanged. The
# draw is a HASH of (seed, env, episode index), not a stream: env i's k-th
# episode gets the same T whatever the other envs did, so a run is
# reproducible and any tool can re-derive an env's T from the three numbers.
# --------------------------------------------------------------------------
CC_P0, CC_TMIN, CC_TMAX, CC_BUCKETS, CC_TEMP_GAIN = 0.5, 0.05, 2.0, 4, 0.25
_CC_M64 = 0xFFFFFFFFFFFFFFFF


def cc_encode(T, tmax: float):
    """T -> the observation column t = log1p(T) / log1p(tmax): 0 at T = 0,
    1 at T = tmax, concave in between so the log-uniform continuum is
    spread over the unit interval. float64 (a float in -> a 0-d array)."""
    return np.log1p(np.asarray(T, np.float64)) / math.log1p(float(tmax))


def cc_decode(t, tmax: float):
    """The inverse of :func:`cc_encode`."""
    return np.expm1(np.asarray(t, np.float64) * math.log1p(float(tmax)))


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64's finaliser over a uint64 array (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def cc_uniform(seed: int, env, ep, salt: int) -> np.ndarray:
    """One uniform in [0, 1) per (env, episode): a pure function of
    (seed, env, ep, salt), 53 random bits, vectorised over env/ep."""
    env = np.asarray(env, np.int64).astype(np.uint64).reshape(-1)
    ep = np.broadcast_to(np.asarray(ep, np.int64).astype(np.uint64),
                         env.shape)
    with np.errstate(over="ignore"):
        x = (env + np.uint64(1)) * np.uint64(0xC2B2AE3D27D4EB4F)
        x ^= (ep + np.uint64(1)) * np.uint64(0x165667B19E3779F9)
        x ^= (np.array([int(seed) & _CC_M64], np.uint64)
              * np.uint64(0x9E3779B97F4A7C15))
        x ^= (np.array([int(salt) & _CC_M64], np.uint64)
              * np.uint64(0x27D4EB2F165667C5))
    z = _splitmix64(x)
    return (z >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def cc_draw(seed: int, env, ep, p0: float, tmin: float, tmax: float) -> np.ndarray:
    """T per (env, episode): 0 with probability ``p0``, else log-uniform on
    [tmin, tmax]. Two independent uniforms per draw (salts 1 and 2), so
    the T = 0 decision and the level are decoupled. float64 (n,)."""
    u0 = cc_uniform(seed, env, ep, 1)
    u1 = cc_uniform(seed, env, ep, 2)
    lo, hi = math.log(float(tmin)), math.log(float(tmax))
    return np.where(u0 < float(p0), 0.0, np.exp(lo + (hi - lo) * u1))


def cc_bucket_edges(p0: float, tmin: float, tmax: float, buckets: int) -> np.ndarray:
    """The advantage-normalisation buckets of train_fast --cc-buckets, as
    ascending edges in t (``np.searchsorted(edges, t, side='right')`` /
    ``torch.bucketize(t, edges, right=True)``: bucket k holds
    edges[k-1] <= t < edges[k]). Bucket 0 is the T = 0 family when
    p0 > 0 (its edge sits halfway to t(tmin), so it holds T = 0 alone);
    the remaining buckets cut the log-uniform continuum into
    equal-PROBABILITY slices (equal widths in log T). ``buckets`` 1 is
    one bucket (no edges). Returns buckets - 1 edges."""
    B = int(buckets)
    if B < 1:
        raise ValueError(f"cc buckets must be >= 1, got {buckets}")
    if B == 1:
        return np.zeros(0, np.float64)
    n_cont = B - 1 if float(p0) > 0.0 else B
    edges = []
    if float(p0) > 0.0:
        edges.append(0.5 * float(cc_encode(tmin, tmax)))
    for k in range(1, n_cont):
        Tk = float(tmin) * (float(tmax) / float(tmin)) ** (k / n_cont)
        edges.append(float(cc_encode(Tk, tmax)))
    return np.asarray(edges, np.float64)


def cc_bucket_of(t, edges) -> np.ndarray:
    """Bucket index per t (see :func:`cc_bucket_edges`)."""
    return np.searchsorted(np.asarray(edges, np.float64),
                           np.asarray(t, np.float64), side="right")


class RaceReward:
    """Start-to-finish speedrun objective (linear maps with a labeled end).

    ``r_t = scale * (d_{t-1} - d_t) - time_pen``, where ``d`` is the GEODESIC
    distance-to-finish from :mod:`goalfield`, plus ``success_bonus`` on the
    tick the env crosses the goal box (``core.goal_hits``, set by the C step
    when a goal box is armed via ``core.set_goal_box``).

    Why this shape: the shaping term is potential-based, so it telescopes —
    looping a ramp nets exactly 0, backtracking refunds itself, and total
    collectible shaping over any successful run is the same ``scale * d_0``.
    With progress income fixed, return = const - time_pen * T + bonus:
    maximizing it IS minimizing time to the finish. Straight-line distance
    would pay for pointing at walls; the geodesic follows the track.

    Stagnation: an env whose best-ever ``d`` this episode hasn't improved by
    ``stall_eps`` units for ``stall_ticks`` physics ticks is reported in
    :meth:`pop_stall_mask` — the trainer forwards it to ``core.force_fail``
    ("kill the agent if its score hasn't improved in 10-20 s"). The shaping
    already pays 0 for loitering; the kill just frees the env slot and makes
    stalling terminal instead of merely unprofitable.

    ``d_floor`` (``--race-dfloor``, 0 = off) CLAMPS the potential from below:
    ``d_eff = max(d, d_floor)``, ``Phi = -d_eff``. Every state closer to the
    goal than ``d_floor`` then shares one potential, so inside that shell the
    shaping pays exactly zero in BOTH directions - it neither rewards getting
    closer nor charges getting further. Still potential-based and still a
    function of state alone (NOT a running-minimum ratchet, which would be
    history-dependent and unrepresentable by the critic), so it telescopes
    exactly like the unclamped term and a closed loop still nets 0.

    Why: on surf_src_cannonball the voxel geodesic believes the player can
    glide ~8,700 u level across open air from route vertex 1600, so the
    field's low-``d`` shell reaches down into a lethal void and PAYS the
    policy ~+2 reward for falling into it. The clamp deletes that income
    without deleting anything above the shell. The floor is a distance, not a
    place: nothing about the route or the finishing line is needed to set it.

    ``d_latch`` (``--race-latch``, 0 = off) is the LATCH: once an episode has
    reached ``d <= d_latch``, the shaping term pays exactly zero for the rest
    of that episode - it neither pays for approaching nor charges for
    leaving. The flag is per-env, set on any tick whose ``d`` is at or under
    the threshold (the spawn tick included), and cleared at every episode
    start. The clamp is the weaker relative: it flattens the potential INSIDE
    the shell but still charges the full climb back out of it, which on this
    map is the whole valley (route vertices 1600 -> 1680 raise ``d``
    6,632 -> 14,976, all of it above the floor, charged -4.02). The latch
    deletes that charge without any reference line.

    The latch is EPISODE HISTORY, not state, so it is fed to the network as
    one extra observation feature (the trainer's route block, 1 wide, last
    column) - otherwise the critic cannot see which regime it is in and the
    value function is unlearnable. :meth:`latch_flags` is that column;
    :meth:`latch_boot` is the same flag one call ago, which is what the
    truncation bootstrap's reconstructed terminal row needs.

    The RAW ``d`` is what still drives the stall detector and the respawn
    ``stagnant`` mask - those are liveness rules, not the objective, and
    inside a clamped shell every state would otherwise read as stagnant and
    the 15 s stall-kill would fire on a correct final approach. Moving them
    would be a second treatment. The latch needs this even harder: past the
    switch EVERY state pays 0, so on the clamped value nothing would ever
    look like progress and the 15 s kill would fire on the whole final
    descent.

    ``ratchet`` (``--race-ratchet``) is the RATCHET: the episode keeps a
    RECORD ``b`` of the best (smallest) ``d`` it has reached, ``b_0`` = the
    distance at the episode's own start (spawn AND every reservoir respawn),
    ``b_{t+1} = min(b_t, d_{t+1})``, and the shaping term becomes
    ``r = scale * (b_t - b_{t+1}) >= 0``. Only NEW progress records are paid;
    losing ground costs nothing and re-gaining it is not paid twice.
    ``d 10 -> 8 -> 12 -> 8 -> 6`` pays ``+2, 0, 0, +2`` where the stock
    signed term pays ``+2, -4, +4, +2``. Time penalty, success bonus, fail
    penalty and the liveness rules on the RAW ``d`` are untouched.

    Why: round 19 measured that a FLAT potential inside the trap is not
    enough (xCLAMP, ``--race-dfloor``: 0/99 finishes) and that making
    LEAVING free is (xLATCH, ``--race-latch``: 52/102). The latch needs a
    map-specific threshold picked from a champion trace; the ratchet is the
    same "leaving is free" property with no threshold and no reference line
    at all - every detour anywhere on the map is uncharged, not just the one
    below ``d_latch``.

    Like the latch this is EPISODE HISTORY, not state, so the network is fed
    one extra observation column, ``(d_t - b_t) / d0`` (:meth:`ratchet_gap`,
    the trailing scalar block the latch uses): 0 exactly when the agent is
    at a new record and positive by how far it has backed off. Without it
    the critic cannot tell "10 units out on the way in" (pays) from "10
    units out on the way back" (pays nothing) and the value function is
    unlearnable. :meth:`ratchet_boot` is the record one call ago, which is
    what the truncation bootstrap's reconstructed terminal row needs.

    ``arc`` (an :class:`surfgym.route.ArcProgress`, i.e. ``--race-arc``)
    REPLACES the geodesic term with arc length along a reference line:
    ``r_t = arc_scale * (a_t - a_{t-1}) - time_pen`` inside a corridor of the
    line and ``-time_pen`` outside it. Same shape, same telescoping, same
    total collectible budget (``arc_scale = 100/route_length`` against
    ``scale = 100/d0``) - but arc length is monotone along the route by
    construction, so it cannot have the interior local minimum the voxel
    geodesic has at route vertex 1601 on surf_src_cannonball. The geodesic
    field is still sampled every call and still drives the stall detector and
    the respawn ``stagnant`` mask: those are liveness rules, not the
    objective, and moving them would be a second treatment. Measured on the
    champion's own finishing runs, the longest no-geodesic-improvement
    stretch is 13.1-13.3 s against a 15 s kill, so the detector does not
    misfire on a correct descent - that is true of the control too.

    Intrinsic exploration (``int_coef > 0``): count-based novelty,
    ``r_int = int_coef / sqrt(N(cell))`` on each transition into a 256u map
    cell, with visit counts N GLOBAL across all envs and all episodes.
    Fresh territory beyond the current fail-wall pays a premium over the
    shaping rate; anything the fleet already frequents decays toward zero
    (2048 envs share one count table, so beaten paths wear out in minutes).
    Not potential-based, but self-annealing — bouncing between two cells
    inflates their own counts and pays ~2c*sqrt(n)/n -> 0.

    ``cc_tmax > 0`` (train_fast ``--curiosity-cond``, the module notes
    above :func:`cc_draw`) makes the reward a PER-ENV mix: env i carries
    an exploration weight T_i, redrawn at every episode start, and is
    paid ``(1 - T_i/T_max) * shaping + T_i * int_coef / sqrt(N + 1)`` with
    the success bonus and the fail penalty untouched. T_i = 0 is the plain
    race reward with NO intrinsic term; T_i = T_max is pure novelty. The
    T vector is the observation the trainer feeds the network
    (:meth:`cc_obs`); :meth:`cc_boot` is the T of the episode that just
    ended, for the truncation bootstrap's reconstructed terminal row.
    """

    def __init__(self, field, scale: float, time_pen: float = 0.005,
                 success_bonus: float = 50.0, stall_ticks: int = 1500,
                 stall_eps: float = 32.0, max_step: float = 100.0,
                 int_coef: float = 0.0, int_cell: float = 256.0,
                 int_view: int = 0, int_speed: int = 0,
                 speed_equiv: float = 0.0, fail_pen: float = 0.0,
                 finish_k: float = 0.0, finish_tref: float = 120.0,
                 every: int = 1, d_floor: float = 0.0,
                 d_latch: float = 0.0, ng: int = 0, ng_gamma: float = 0.0,
                 ng_d0: float = 0.0, death_charge: float = 0.0,
                 arc=None, arc_scale: float = 0.0,
                 ratchet: bool = False, ratchet_d0: float = 0.0,
                 d0_per_env: bool = False, tick_ms: float = 10.0,
                 cc_tmax: float = 0.0, cc_p0: float = CC_P0,
                 cc_tmin: float = CC_TMIN, cc_seed: int = 0,
                 dip: bool = True, frontier_d0: float = 0.0,
                 frontier_start_eps: float = 256.0,
                 frontier_anchor_speed: float = 0.0) -> None:
        self.field = field
        # --respawn-frontier: the START-ANCHORED frontier tracker. OFF
        # unless frontier_d0 > 0 (it IS the map's start geodesic d0), in
        # which case every episode END records the d it SPAWNED at and the
        # smallest d it REACHED. pop_stats then reports P_max over the
        # episodes that spawned AT THE MAP START only - the one progress
        # measure a forward spawn curriculum cannot inflate, because no
        # frontier spawn is a start spawn. LOGGING ONLY: no reward term, no
        # observation, no RNG. See surfgym.respawn.FrontierSpawnSampler.
        self.frontier_d0 = float(frontier_d0)
        self.frontier_start_eps = float(frontier_start_eps)
        # --respawn-frontier-anchor: > 0 makes "start-anchored" also mean
        # FROM REST - a spawn placed within start_eps of d0 but launched at
        # speed (a frontier row that landed near the start, a reservoir row
        # harvested mid-flight) is not a start and must not move P_max.
        # 0.0 = the distance rule alone, byte-identical to before.
        self.frontier_anchor_speed = float(frontier_anchor_speed)
        self._fr_spawn_spd: np.ndarray | None = None
        # --respawn-frontier-quantile: the per-episode START-ANCHORED
        # reaches of the last pop_stats window, for a quantile instead of
        # a max (pop_frontier_reaches)
        self._fr_anch_reach = np.zeros(0, np.float64)
        self._fr_best: np.ndarray | None = None
        self._fr_spawn: np.ndarray | None = None
        self.fr_pairs: list[tuple] = []
        # --tick-ms: the MEAN physics tick, ms. Every tick counter in here
        # (stall_ticks, the finish clock, stagnant_mask's window) stays in
        # ticks; this is what turns them into seconds. 10.0 = today, and the
        # seconds conversions below reduce to the legacy `/ 100.0` exactly.
        self.tick_ms = float(tick_ms)
        # --goal-reward euclid + --death-charge: the potential's origin is
        # PER ENV - the distance to this env's goal at assignment (the goal
        # system writes it via set_d0) - not the map's start geodesic.
        # False leaves the geodesic death-charge path untouched.
        self.d0_per_env = bool(d0_per_env)
        self._d0: np.ndarray | None = None
        # --race-ng: Ng-conformant shaping (research question 4, round 27).
        # The stock term Phi(s')-Phi(s) does not telescope under gamma < 1;
        # the residue is a per-call leak of ~(1-gamma^every)*banked, ~9x the
        # explicit time penalty deep into a map, and death (r[ended]=0)
        # keeps all collected shaping income for free. ng=1 makes the
        # discounted sum telescope exactly: per call the tax
        # (1-gamma^every)*Phi(s') is charged, and a terminal transition
        # charges the full remaining potential -Phi (death AND goal: shaping
        # nets zero over any episode from its own spawn, so behavior is
        # driven by success_bonus/time_pen alone - Ng invariance including
        # termination). ng=2 is the bond variant: death still forfeits the
        # bank, finishing keeps it (~+Phi(goal) extra success payment).
        # Truncation stays exempt on purpose - the bootstrap carries V.
        # 0 = off, and off touches no array the control did not.
        self.ng = int(ng)
        self.ng_d0 = float(ng_d0)
        self._ng_g = (float(ng_gamma) ** float(every)) if ng else 1.0
        if self.ng and (float(d_floor) > 0.0 or float(d_latch) > 0.0):
            raise ValueError("--race-ng with --race-dfloor/--race-latch is "
                             "untested; run it as its own arm")
        # --race-dfloor: potential floor, d_eff = max(d, d_floor). 0 = off,
        # and off is the control path byte for byte (no array is touched).
        self.d_floor = float(d_floor)
        # --race-latch: once d <= d_latch this episode, the shaping term is
        # skipped entirely for the rest of it. 0 = off, and off never touches
        # an array or a branch that the control did not.
        self.d_latch = float(d_latch)
        self._latched: np.ndarray | None = None
        self._latch_boot: np.ndarray | None = None
        # d0_per_env (--goals with a euclid goal field) moves the potential's
        # origin to a PER-ENV quantity: the straight-line distance to THIS
        # env's goal at assignment, set through set_d0, typically a few
        # hundred to a few thousand units and different in every env. All
        # three of ng, d_floor and d_latch are absolute GEODESIC-scale
        # numbers against the map's start distance - ng's conformant tax is
        # (1-gamma^k)*(ng_d0 - d)*scale with ng_d0 the map's start geodesic,
        # and the two thresholds are compared to d directly. Composing them
        # is not a mis-tuned constant, it is two different distance scales in
        # one expression, so refuse rather than run something whose reward is
        # per-env nonsense with nothing in the log saying so.
        if self.d0_per_env and (self.ng or float(d_floor) > 0.0
                                or float(d_latch) > 0.0):
            raise ValueError(
                "a per-env goal potential (d0_per_env) with --race-ng/"
                "--race-dfloor/--race-latch: those are geodesic-scale "
                "quantities against the map's start distance and this "
                "potential is a per-env euclid distance to a moving goal - "
                "run them as separate arms")
        # --race-arc: replace the geodesic potential with ARC LENGTH along a
        # reference line (surfgym.route.ArcProgress). Arc length is monotone
        # along the route by construction, so unlike the voxel geodesic it
        # cannot have an interior local minimum for the agent to stop at.
        # The geodesic field is still sampled every call - the stall detector
        # and the respawn `stagnant` mask are defined on it and are NOT part
        # of this treatment. arc=None is the control path, byte for byte.
        # Composing arc with ng/d_floor/d_latch is untested (each was
        # validated as its own arm); refuse loudly rather than run silently.
        self.arc = arc
        self.arc_scale = float(arc_scale)
        if arc is not None and (self.ng or float(d_floor) > 0.0
                                or float(d_latch) > 0.0):
            raise ValueError("--race-arc with --race-ng/--race-dfloor/"
                             "--race-latch is untested; run it as its own arm")
        # --race-ratchet: pay only NEW progress records inside an episode.
        # False is the control path byte for byte (no array, no branch the
        # control did not take). It is a DIFFERENT treatment of the same
        # defect as d_floor/d_latch/ng/arc - composing two of them would
        # measure neither, so refuse rather than run something whose
        # objective nothing in the log describes.
        self.ratchet = bool(ratchet)
        if self.ratchet and (self.ng or float(d_floor) > 0.0
                             or float(d_latch) > 0.0 or arc is not None):
            raise ValueError("--race-ratchet with --race-ng/--race-dfloor/"
                             "--race-latch/--race-arc is a second treatment "
                             "of the same defect; run it as its own arm")
        if self.ratchet and self.d0_per_env:
            raise ValueError("--race-ratchet with a per-env goal potential "
                             "(d0_per_env) is untested; run it as its own arm")
        # the normaliser for the observation column, (d - b)/d0. The trainer
        # passes THIS map's start geodesic (rf_d0); falling back to 100/scale
        # recovers the same number whenever scale is the stock 100/d0.
        self.ratchet_d0 = (float(ratchet_d0) if float(ratchet_d0) > 0.0
                           else (100.0 / float(scale) if float(scale) > 0.0
                                 else 1.0))
        self._rec: np.ndarray | None = None
        self._rec_boot: np.ndarray | None = None
        self.scale = float(scale)
        self.time_pen = float(time_pen)
        self.success_bonus = float(success_bonus)
        # time-scaled finish bonus (0 = off): += finish_k * (tref - T_ep)
        # seconds, clamped at 0, paid ON the finish tick. Equivalent to a
        # per-second time cost charged only to successful episodes — the
        # gradient wrt finish time is -finish_k everywhere, but there is no
        # suicide channel (non-finishers never touch it) and no
        # avoid-the-curtain channel (the clamp keeps the paid bonus >= 0, so
        # crossing always beats stalling). The tref/spawn-dependent offset is
        # a per-spawn constant the value baseline absorbs.
        self.finish_k = float(finish_k)
        self.finish_tref = float(finish_tref)
        # call cadence in physics ticks (frame-skip trainers pass K and call
        # once per decision): the potential shaping TELESCOPES across the
        # skipped ticks, so the per-decision sum is exactly the per-tick sum;
        # time_pen and the tick counters scale by `every`, the teleport clamp
        # widens by `every`, and stall/finish thresholds stay in tick units.
        # The caller must pass OR-accumulated done/trunc masks and the
        # OR-accumulated goal mask (goal=) — core.goal_hits mutates per tick.
        self.every = max(1, int(every))
        self.stall_ticks = int(stall_ticks)
        self.stall_eps = float(stall_eps)
        # a legit tick moves <= ~35u (sv_maxvelocity * 10ms); anything larger
        # is a teleport-ish relocation and must not cash shaping
        self.max_step = float(max_step)
        self.int_coef = float(int_coef)
        self.int_cell = float(int_cell)
        # view-aware novelty (0 = off): the depth image depends on gaze as
        # much as position — the same voxel looking left vs right is a
        # different observation to the CNN. int_view = number of yaw sectors
        # in the count key, so scanning a familiar place from a new angle
        # still pays first-visit novelty (and wears out globally like any
        # other cell). Multiplies the count table by int_view.
        self.int_view = int(int_view)
        # speed buckets in the novelty key (0 = off): arriving at a known
        # place at a NEW speed is a new state for exploration purposes —
        # walls here are speed-gated. Bin width assumes the 4000 u/s server
        # cap; faster states clip into the top bin.
        self.int_speed = int(int_speed)
        self._speed_bin = 4000.0 / max(1, self.int_speed)
        # speed folded into the POTENTIAL (0 = off): d_eff = d - beta*s.
        # Unlike speed_coef (pays the speed LEVEL per tick, changes the
        # optimum), this stays potential-based — closed loops in position
        # AND speed net zero, so a speed-building detour becomes locally
        # profitable exactly when it is time-profitable. The death refund
        # in __call__ is load-bearing: without it "accelerate and die"
        # farms the credit.
        self.speed_equiv = float(speed_equiv)
        self._s: np.ndarray | None = None
        # explicit death cost: at an unlearned frontier, V(beyond) is ~0, so
        # "die at max progress" and "survive past it" pay the same — shaping
        # alone cannot prefer the catch until catches exist. A terminal
        # penalty creates the differential immediately (truncation exempt:
        # it is bootstrapped, not a death)
        self.fail_pen = float(fail_pen)
        # --death-charge kappa (round 27): at death, charge kappa*Phi(last
        # pre-death state) ON TOP of the stock scheme - no per-step tax, no
        # goal charge. A doomed run still nets (1-kappa)*Phi(death)-Phi(spawn)
        # of shaping, so depth keeps paying (the curriculum survives), while
        # a deliberate deep dive is taxed in proportion to the bank it
        # abandons. kappa=0 is the control byte for byte; kappa=1 removes
        # the depth income at death entirely (measured 2026-08-25 on xNGS:
        # with the conformant tax as well, that collapses to fast suicide -
        # do not run kappa=1 from scratch expecting learning). Requires
        # ng_d0. Mutually exclusive with --race-ng terminal charges.
        self.death_charge = float(death_charge)
        if self.death_charge and self.ng:
            raise ValueError("--death-charge and --race-ng both charge "
                             "terminals; pick one")
        if self.death_charge and not self.ng_d0:
            raise ValueError("--death-charge needs ng_d0 (the map's start "
                             "geodesic) to define Phi")
        # --curiosity-cond (cc_draw's notes): the per-env exploration
        # weight. cc_tmax 0 = off, and off touches no array and no branch
        # the control did not. The T vector is redrawn for an env at every
        # episode start inside __call__ (where the latch and the arc anchor
        # are reset too), from a hash of (seed, env, episode index), unless
        # set_cc_T pinned it. Composing it with --race-ng / --death-charge /
        # a per-env goal potential / --speed-equiv is untested: those add
        # terms of their own to the shaping side, and "which of them scales
        # with (1 - w)" has no measured answer - refuse, as the arc does.
        self.cc_tmax = float(cc_tmax)
        self.cc_p0, self.cc_tmin = float(cc_p0), float(cc_tmin)
        self.cc_seed = int(cc_seed)
        if self.cc_tmax > 0.0:
            if not 0.0 <= self.cc_p0 <= 1.0:
                raise ValueError(f"cc_p0 must be in [0, 1], got {cc_p0}")
            if not 0.0 < self.cc_tmin <= self.cc_tmax:
                raise ValueError(f"cc_tmin must be in (0, cc_tmax], got "
                                 f"{cc_tmin} with cc_tmax {cc_tmax}")
            if (self.ng or self.death_charge or self.d0_per_env
                    or float(speed_equiv) > 0.0):
                raise ValueError("--curiosity-cond with --race-ng / "
                                 "--death-charge / a per-env goal potential "
                                 "/ --speed-equiv is untested; run it as "
                                 "its own arm")
        self._cc_T: np.ndarray | None = None       # the live T per env
        self._cc_ep: np.ndarray | None = None      # episode index per env
        self._cc_boot: np.ndarray | None = None    # T of the episode that ended
        self._cc_shape: np.ndarray | None = None   # (1 - T/T_max), float32
        self._cc_fixed: np.ndarray | None = None   # set_cc_T's pin
        # per-tick horizontal-speed bonus: speed_coef * h_speed/1000 — tilts
        # line choice toward carrying speed (speed-gated jumps). Not farmable
        # here: racing collects the same income PLUS shaping, and circling
        # gets stall-killed in 15s
        self.speed_coef = 0.0
        # --surf-bonus / --dive-pen (round 40, pnSURF): "surfing = good,
        # diving = bad" as a dense per-tick term. Both are REWARD UNITS PER
        # SECOND on the flag; the trainer converts to per-tick. 0/0 is off
        # and bit-identical (the whole block is skipped: no state, no RNG).
        #
        # THE CONTACT TEST, and why it is the cheapest honest one. Riding a
        # surfable face KEEPS onground == -1 - that IS the surf mechanic
        # (pm.c sets onground only at plane_normal[2] >= 0.7), so "airborne"
        # alone conflates flight with ramp-riding. What separates them is
        # whether the map pushed back VERTICALLY this tick:
        #
        #     dev = (vz - vz_prev) + g*dt      (0 in exact free fall)
        #     supported = airborne AND dev > surf_dvz
        #
        # This is the same "last tick the map pushed back" detector
        # tools/pick_selfline.py trims self-lines with (CLAUDE.md), reused
        # here rather than re-derived. It costs one subtract and one compare
        # over the (N,) velocity column the reward already fetches - no grid
        # sample, no extra trace, no C change, nothing per-pixel.
        #
        # It brackets the surfable band |n_z| in (0, 0.7) from BOTH sides,
        # using the engine's own arithmetic:
        #   * the upper edge is exact - onground == -1 IS n_z < 0.7;
        #   * the lower edge falls out of the physics - a near-vertical wall
        #     (n_z ~ 0) clips only the HORIZONTAL velocity, so it returns
        #     dev = 0 and pays nothing, which is the behaviour wanted;
        #   * a ceiling bonk pushes DOWN (dev < 0) and is excluded by the
        #     sign, which is why the test is signed and not on |dev|.
        # Measured on 18 recorded petrus flights of this round's own control:
        # 74% of airborne ticks sit at dev == 0 exactly (free fall), 23-25%
        # at dev > 1, and 0.01-0.10% at dev < -1. The split is essentially
        # BINARY, so every threshold in 0.05..4.0 gives the same 24.6-25.1%,
        # and surf_dvz = 1.0 sits in the middle of that plateau.
        self.surf_bonus = 0.0                    # per TICK once resolved
        self.dive_pen = 0.0                      # per TICK once resolved
        self.surf_dvz = 1.0                      # u/s of vertical give-back
        self.surf_hspd = 0.0                     # optional speed floor, u/s
        self._vz = None
        # diagnostics the arm is read on (CLAUDE.md: a bonus whose paid
        # fraction saturates is measuring a farm, not the behaviour)
        self.surf_paid_ticks = 0.0
        self.dive_ticks = 0.0
        self.surf_air_ticks = 0.0
        self._d: np.ndarray | None = None
        # the CLAMPED previous distance: what the shaping differences. Kept
        # separate from self._d so the liveness counters below keep seeing the
        # raw geodesic. With d_floor = 0 the two are the same values.
        self._dc: np.ndarray | None = None
        self._best: np.ndarray | None = None
        self._since: np.ndarray | None = None
        self._ticks: np.ndarray | None = None
        self._counts: np.ndarray | None = None   # global cell visit counts
        self._pending_counts: np.ndarray | None = None
        # DDP counts sharing (docs/ddp-plan.md §3a): _counts_base is the
        # table as of the last cross-rank sync, so the sync exchanges
        # DELTAS, never absolutes — that is what makes the checkpoint
        # round-trip structurally safe (a restored table is loaded on every
        # rank and the first sync adds only new visits).
        # track_touched makes the sync O(cells entered) instead of
        # O(table): with --int-view 8 --int-speed 3 the champion table is
        # ~32M cells (~256 MB int64), so the dense subtract/all-reduce/
        # apply cycle would cost hundreds of ms per iteration — while the
        # cells actually entered per iteration are a few tens of thousands.
        self._counts_base: np.ndarray | None = None
        self.track_touched = False               # DDP trainer sets True
        self._touched: list[np.ndarray] = []
        self._prev_cell: np.ndarray | None = None
        self._mins = None
        self._dims = None
        # episode-outcome accumulators, drained by pop_stats()
        self.n_success = 0
        self.n_fail = 0
        self.n_trunc = 0
        self.int_paid = 0.0
        self.finish_ticks: list[int] = []
        # --race-arc diagnostics, drained by pop_stats(): the arc actually
        # realized per episode, checked against the recorded trajectories.
        # An arc reward computed from the agent's own position is farmable if
        # the corridor/order rules leak, and "arc gained per episode" is the
        # number that would show it.
        self._arc_spawn: np.ndarray | None = None
        self._arc_max: np.ndarray | None = None
        self._arc_off: np.ndarray | None = None
        self.arc_gain: list[float] = []
        self.arc_reach: list[float] = []
        self.arc_off_frac: list[float] = []

        # -- the `dip/*` diagnostic (surfgym/dipmeter.py) ------------------
        # How much geodesic potential the policy gives up and RECOVERS: the
        # per-episode running minimum b of d, the instantaneous
        # depth = (d - b) * scale in reward units, and every maximal stretch
        # of depth > 0 classified as survived (a new record closed it) or
        # failed (the episode ended inside it). Exactly --race-ratchet's own
        # bookkeeping, maintained whether or not that flag is on, and
        # LOGGING ONLY: it is read after every term of r is final, it feeds
        # no observation and it draws from no RNG. ON BY DEFAULT (it is a
        # diagnostic); `dip=False` allocates nothing and takes no branch the
        # control did not.
        #
        # Refused under a PER-ENV goal potential (--goals + a euclid field):
        # there d is the distance to a MOVING goal that is reassigned
        # without an episode end, so every reassignment would read as a
        # fabricated dip. Degrade to logging nothing rather than logging
        # nonsense.
        self.dip = bool(dip) and not self.d0_per_env
        self._dip: DipMeter | None = None

    def _cells(self, states) -> np.ndarray:
        p = states["origin"].astype(np.float64)
        ix = np.clip(((p[:, 0] - self._mins[0]) // self.int_cell).astype(np.int64),
                     0, self._dims[0] - 1)
        iy = np.clip(((p[:, 1] - self._mins[1]) // self.int_cell).astype(np.int64),
                     0, self._dims[1] - 1)
        iz = np.clip(((p[:, 2] - self._mins[2]) // self.int_cell).astype(np.int64),
                     0, self._dims[2] - 1)
        key = ix + self._dims[0] * (iy + self._dims[1] * iz)
        if self.int_view > 0:
            yb = np.floor((states["yaw"].astype(np.float64) % 360.0)
                          / 360.0 * self.int_view).astype(np.int64)
            np.clip(yb, 0, self.int_view - 1, out=yb)
            key = key * self.int_view + yb
        if self.int_speed > 0:
            v = states["velocity"]
            sb = (np.hypot(v[:, 0], v[:, 1]) // self._speed_bin).astype(np.int64)
            np.clip(sb, 0, self.int_speed - 1, out=sb)
            key = key * self.int_speed + sb
        return key

    def _clamp(self, d: np.ndarray) -> np.ndarray:
        """d_eff = max(d, d_floor). Off (0) returns the array unchanged, so
        the control path allocates and computes exactly what it always did."""
        if self.d_floor <= 0.0:
            return d
        return np.maximum(d, self.d_floor)

    def on_reset(self, core) -> None:
        n = core.num_envs
        self._d = self.field.sample(_states(core)["origin"]).astype(np.float64)
        self._dc = self._clamp(self._d)
        v0 = _states(core)["velocity"]
        self._s = np.hypot(v0[:, 0], v0[:, 1]).astype(np.float64)
        # --surf-bonus / --dive-pen: the previous tick's vz and the gravity
        # step the free-fall comparison is made against. `tick_ms` is the
        # MEAN tick under a --tick-ms pattern, the same reading every other
        # seconds conversion in this class takes.
        self._vz = v0[:, 2].astype(np.float64).copy()
        self._g_tick = float(core.config.phys.sv_gravity) * self.tick_ms * 1e-3
        self._best = self._d.copy()
        if self.frontier_d0 > 0.0:
            # --respawn-frontier: its own copies, because self._best folds
            # the NEXT episode's spawn in before the `ended` block runs
            self._fr_best = self._d.copy()
            self._fr_spawn = self._d.copy()
            self._fr_spawn_spd = np.hypot(v0[:, 0], v0[:, 1]).astype(np.float64)
            self.fr_pairs.clear()
        if self.d0_per_env:
            self._d0 = self._d.copy()
        # a spawn already inside the shell IS a tick with d <= d_latch,
        # so it arms the latch immediately - at --respawn-margin 2 the
        # reservoir really does place starts past the wall, and charging
        # those for leaving is exactly the term this arm removes
        self._latched = (self._d <= self.d_latch if self.d_latch > 0.0
                         else np.zeros(n, bool))
        self._latch_boot = self._latched.copy()
        # --race-ratchet: the record starts AT the episode's own start
        # distance, so an episode start - spawn or reservoir respawn - can
        # never itself generate progress reward, exactly as the latch is
        # armed off the spawn tick.
        if self.ratchet:
            self._rec = self._d.copy()
            self._rec_boot = self._rec.copy()
        self._since = np.zeros(n, np.int64)
        self._ticks = np.zeros(n, np.int64)
        if self.dip:
            # the accumulator survives a re-reset (it is drained per
            # iteration, not per reset); only the per-env record restarts
            if self._dip is None or self._dip.depth.shape != (n,):
                self._dip = DipMeter(n)
            self._dip.start(self._d)
        if self.cc_tmax > 0.0:
            # every env starts episode 0 with its own draw (spawn counts as
            # an episode start, like every later respawn)
            self._cc_ep = np.zeros(n, np.int64)
            self._cc_T = self._cc_draw(np.arange(n))
            self._cc_boot = self._cc_T.copy()
            self._cc_apply()
        if self.arc is not None:
            self.arc.reset(_states(core)["origin"])
            self._arc_spawn = self.arc.arc.copy()
            self._arc_max = self.arc.arc.copy()
            self._arc_off = np.zeros(n, np.int64)
        if self.int_coef > 0.0:
            mins, maxs = core.map_bounds()
            self._mins = mins.astype(np.float64)
            self._dims = tuple(int(np.ceil((maxs[i] - mins[i]) / self.int_cell))
                               + 1 for i in range(3))
            ncells = (self._dims[0] * self._dims[1] * self._dims[2]
                      * max(1, self.int_view) * max(1, self.int_speed))
            if (self._pending_counts is not None
                    and len(self._pending_counts) == ncells):
                # checkpointed table: a resume must NOT re-pay first-visit
                # novelty for the fleet's entire beaten path (racing line
                # included — the exact opposite of exploration pressure)
                self._counts = self._pending_counts.astype(np.int64)
            elif self._counts is None or len(self._counts) != ncells:
                self._counts = np.zeros(ncells, np.int64)   # survives resets
            self._pending_counts = None
            # DDP only (track_touched is set by the trainer BEFORE reset):
            # a resume (or a fresh table) starts with a ZERO delta. Never
            # allocate the base unconditionally — with keyed tables it is
            # a ~256 MB duplicate nothing on a single-GPU run ever reads.
            # And never lazily zero it later: a zero base after a resume
            # would report the restored history as the first delta (the
            # multiply-by-R failure plan §3a calls structurally impossible).
            self._counts_base = (self._counts.copy() if self.track_touched
                                 else None)
            self._touched.clear()
            self._prev_cell = self._cells(_states(core))

    def counts_state(self) -> np.ndarray | None:
        """Visit-count table for checkpointing (uint32 copy, ~5 MB)."""
        if self._counts is None:
            return None
        return np.minimum(self._counts, 0xFFFFFFFF).astype(np.uint32)

    def restore_counts(self, arr) -> None:
        """Hand back a checkpointed table; applied on the next on_reset
        (dims are validated there — a different map/cell just re-zeroes)."""
        if arr is not None:
            self._pending_counts = np.asarray(arr)

    def decay_counts(self, factor: float) -> int:
        """train_fast --unstuck: multiply every visit count by ``factor``
        (floored to an integer), so the novelty bonus ``int_coef /
        sqrt(N + 1)`` of a worn-out cell recovers - on the beaten path the
        counts are 1-2 million (bonus ~0.0002) and a halving barely moves
        them, in a lightly visited region it doubles the bonus in a few
        pulses. In place on the int64 table (``>> 1`` for the 0.5 default,
        no float copy of a 256 MB table). Counts pending from a checkpoint
        are decayed too, so the pulse is not lost across the next reset.
        -> the number of non-zero cells afterwards (0 with no table).
        Refused under DDP counts sharing: the delta sync keys on a base
        table this would silently desynchronise."""
        f = float(factor)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"count decay factor must be in [0, 1], got {f}")
        if self.track_touched:
            raise RuntimeError("decay_counts under DDP counts sharing would "
                               "desynchronise the delta base")
        if f == 1.0:
            return int(np.count_nonzero(self._counts)) if self._counts is not None else 0
        n = 0
        for name in ("_counts", "_pending_counts"):
            c = getattr(self, name)
            if c is None:
                continue
            if c.dtype != np.int64:
                c = c.astype(np.int64)
            if f == 0.5:
                np.right_shift(c, 1, out=c)
            else:
                c[:] = np.floor(c * f).astype(np.int64)
            setattr(self, name, c)
            if name == "_counts":
                n = int(np.count_nonzero(c))
        return n

    # -- DDP counts sharing (docs/ddp-plan.md §3a) --------------------------
    def counts_delta(self) -> np.ndarray | None:
        """Increments since the last sync, int32 (never absolutes)."""
        if self._counts is None:
            return None
        if (self._counts_base is None
                or len(self._counts_base) != len(self._counts)):
            self._counts_base = np.zeros_like(self._counts)
        return (self._counts - self._counts_base).astype(np.int32)

    def apply_counts_delta(self, fleet_delta: np.ndarray) -> None:
        """``fleet_delta`` is the ALL-RANK sum of :meth:`counts_delta` —
        including this rank's own, so counts = base + fleet sum."""
        np.add(self._counts_base, fleet_delta, out=self._counts,
               casting="unsafe")
        self._counts_base[:] = self._counts

    def counts_delta_sparse(self):
        """(cells, increments) touched since the last sync — O(entries),
        never O(table). Needs ``track_touched``; increments are exact
        because untouched cells cannot have moved off the base."""
        if self._counts is None:
            return None, None
        if not self._touched:
            return (np.empty(0, np.int64), np.empty(0, np.int32))
        u = np.unique(np.concatenate(self._touched))
        self._touched.clear()
        return u, (self._counts[u] - self._counts_base[u]).astype(np.int32)

    def apply_counts_delta_sparse(self, cells: np.ndarray,
                                  incs: np.ndarray) -> None:
        """``cells``/``incs`` are the CONCATENATION of every rank's
        :meth:`counts_delta_sparse` (this rank's included; duplicate cells
        across ranks legal — np.add.at sums them)."""
        np.add.at(self._counts_base, cells, incs.astype(np.int64))
        self._counts[cells] = self._counts_base[cells]

    def counts_check(self) -> tuple[int, int]:
        """Cheap per-iteration cross-rank divergence probe: total visits
        plus a strided sample digest (the full-table hash runs on the
        slow cadence — hashing 256 MB every iteration is real money)."""
        if self._counts is None:
            return (0, 0)
        import hashlib
        h = hashlib.blake2b(memoryview(np.ascontiguousarray(
            self._counts[::257])), digest_size=8).digest()
        return (int(self._counts.sum()),
                int.from_bytes(h, "little", signed=True))

    def __call__(self, prev_obs, obs, terminal_obs, base_rewards, done, trunc,
                 core, goal=None):
        if self._d is None:
            self.on_reset(core)
            return np.zeros(len(done), np.float32)
        d = self.field.sample(_states(core)["origin"]).astype(np.float64)
        if goal is None:
            goal = core.goal_hits.astype(bool)
        else:
            goal = np.asarray(goal, bool)
        ended = (done | trunc).astype(bool)
        dc = self._clamp(d)
        clip = self.max_step * self.every
        if self.arc is None:
            if self.ratchet:
                # b_{t+1} = min(b_t, d_{t+1}); the term is the RECORD's
                # improvement, never negative. The record as it stood at
                # t-1 is snapshotted first: that is what the network was
                # shown at t-1 and what the truncation bootstrap needs to
                # rebuild the terminal row (ratchet_boot), exactly the
                # contract latch_boot has.
                self._rec_boot = self._rec.copy()
                new_rec = np.minimum(self._rec, d)
                delta = self._rec - new_rec
                # same teleport guard as the control: a relocation must not
                # cash shaping. The record still moves, so the ground the
                # teleport covered is simply never paid for.
                np.clip(delta, -clip, clip, out=delta)
                self._rec = new_rec
            else:
                delta = self._dc - dc
                np.clip(delta, -clip, clip, out=delta)
            if self.d_latch > 0.0:
                # the flag as it stood at t-1 is what governs THIS
                # transition's shaping, and it is also the flag the network
                # was shown at t-1 - so the reward the critic has to predict
                # is one it could actually see. Snapshotted for the
                # truncation bootstrap, which rebuilds a terminal row from
                # outside this call after the autoreset has moved on.
                self._latch_boot = self._latched.copy()
                delta[self._latched] = 0.0
            r = (delta * self.scale - self.time_pen * self.every) \
                .astype(np.float32)
            if self.ng:
                # conformant tax on the post-step potential; ended rows are
                # wiped below and then charged their terminal potential
                # instead
                r -= ((1.0 - self._ng_g) * (self.ng_d0 - dc)
                      * self.scale).astype(np.float32)
        else:
            # arc-length shaping. `advance` pays 0 and freezes its anchor
            # outside the corridor, so leaving the line stops the clock and
            # can never be cashed; inside it the term is the signed arc
            # delta, i.e. potential-based exactly like the geodesic one.
            # The same max_step clip applies: a legal tick moves <= ~35u of
            # arc, anything larger is a relocation.
            pos = _states(core)["origin"]
            arc_before = self.arc.arc.copy()
            delta, inside = self.arc.advance(pos)
            np.clip(delta, -clip, clip, out=delta)
            r = (delta * self.arc_scale - self.time_pen * self.every) \
                .astype(np.float32)
            # diagnostics: on an ended row `pos` is already the NEXT
            # episode's spawn, so the dying episode's last arc is the
            # pre-advance one
            cur = np.where(ended, arc_before, self.arc.arc)
            self._arc_max = np.maximum(self._arc_max, cur)
            self._arc_off += (~inside) & ~ended
        v = _states(core)["velocity"]
        s = np.hypot(v[:, 0], v[:, 1]).astype(np.float64)
        if self.speed_coef > 0.0:
            r += (self.speed_coef / 1000.0) * s.astype(np.float32)
        if self.surf_bonus > 0.0 or self.dive_pen > 0.0:
            # --surf-bonus / --dive-pen: "surfing = good, diving = bad".
            # The contact test is derived in __init__. Over `every` ticks
            # the free-fall baseline is `every` gravity steps, so the same
            # expression holds at --reward-per-decision.
            vz = v[:, 2]
            dev = (vz - self._vz) + self._g_tick * float(self.every)
            airborne = _states(core)["onground"] == -1
            supported = airborne & (dev > self.surf_dvz)
            if self.surf_hspd > 0.0:
                # optional guard (OFF by default, its own arm): a bonus with
                # no speed floor invites "slide gently forever"
                supported = supported & (s >= self.surf_hspd)
            if self.surf_bonus > 0.0:
                r = r + np.float32(self.surf_bonus) * supported
            if self.dive_pen > 0.0:
                # the symmetric half: unsupported free fall. Airborne with
                # nothing holding the player up IS dev <= surf_dvz, so this
                # needs no downward ray of its own.
                r = r - np.float32(self.dive_pen) * (airborne & ~supported)
            self.surf_paid_ticks += float(np.count_nonzero(supported))
            self.dive_ticks += float(np.count_nonzero(airborne & ~supported))
            self.surf_air_ticks += float(np.count_nonzero(airborne))
            self._vz = vz.astype(np.float64)

        if self.speed_equiv > 0.0:
            # potential term for d_eff = d - beta*s: gaining speed pays now,
            # losing it pays back — the ended mask below wipes the garbage
            # (post-autoreset) rows exactly like the distance term
            r += (self.scale * self.speed_equiv) * (s - self._s) \
                .astype(np.float32)
        if self._cc_shape is not None:
            # --curiosity-cond: everything above this line is the SHAPING
            # side (the potential difference, the time penalty, the ng tax
            # and the speed terms) - scaled per env by (1 - T/T_max). The
            # outcome terms below (success bonus, fail penalty, the
            # terminal charges) are paid in full to every member.
            r *= self._cc_shape
        # ended rows: states are already the NEW episode's spawn — the final
        # approach tick's shaping is forfeited (<= ~35u, noise next to the
        # bonus); outcome pays instead
        r[ended] = 0.0
        r[goal] += self.success_bonus
        if self.finish_k > 0.0 and goal.any():
            tsec = (self._ticks[goal].astype(np.float64)
                    + float(self.every)) / (1000.0 / self.tick_ms)
            r[goal] += (self.finish_k
                        * np.maximum(0.0, self.finish_tref - tsec)
                        ).astype(np.float32)
        if self.fail_pen > 0.0:
            r[done.astype(bool) & ~goal] -= self.fail_pen
        if self.ng in (1, 2):
            # terminal potential charge, computed from the LAST pre-death
            # distance (self._dc): the post-step states of ended rows are
            # already the next episode's spawn. Death forfeits the bank;
            # ng=1 charges the finish too (strict invariance), ng=2 lets a
            # finisher keep it. ng=3 taxes per step but leaves terminals
            # stock (question 4's fix (a) verbatim). Truncation exempt.
            # MEASURED 2026-08-25 (xNGS, from scratch): ng=1 collapses to
            # fast suicide - the charge cancels Phi(death) exactly, so a
            # never-finishing policy has nothing left to earn.
            phi_prev = ((self.ng_d0 - self._dc) * self.scale) \
                .astype(np.float32)
            dead = done.astype(bool) & ~goal
            r[dead] -= phi_prev[dead]
            if self.ng == 1:
                r[goal] -= phi_prev[goal]
        if self.death_charge > 0.0:
            # kappa-scaled death charge on the stock scheme: doomed depth
            # still nets (1-kappa)*Phi, deliberate dives pay kappa*Phi
            phi_prev = ((self.ng_d0 - self._dc) * self.scale) \
                .astype(np.float32)
            dead = done.astype(bool) & ~goal
            if self._d0 is not None:
                # goal arms: the bank is measured from THIS episode's
                # assignment distance, not the map's start geodesic
                phi_prev = ((self._d0 - self._dc) * self.scale).astype(np.float32)
            r[dead] -= self.death_charge * phi_prev[dead]
        if self.speed_equiv > 0.0:
            # death refund — load-bearing: without it "accelerate and die"
            # farms the speed credit. Cached self._s, NOT s: ended rows'
            # post-step states are the next episode's spawn. Not on goal
            # (arriving fast is the objective), not on truncation (the
            # bootstrap carries the potential).
            dead = done.astype(bool) & ~goal
            r[dead] -= (self.scale * self.speed_equiv
                        * self._s[dead]).astype(np.float32)
            self._s = s
        if self.int_coef > 0.0:
            cell = self._cells(_states(core))
            # ended ticks are masked: post-autoreset states are the NEW
            # episode's spawn, so the "transition" is a respawn relocation.
            # Cost: the cell entered on the exact death tick is neither paid
            # nor counted (~one cell per episode, unobservable Python-side).
            moved = (cell != self._prev_cell) & ~ended
            if moved.any():
                mi = np.flatnonzero(moved)
                mc = cell[mi]
                bonus = self.int_coef / np.sqrt(self._counts[mc] + 1.0)
                if self._cc_T is not None:
                    # --curiosity-cond: the count bonus x T, so a T = 0
                    # member is paid NO novelty and the T_max member is
                    # paid it in full (the counts are still shared: the
                    # table is the fleet's, whoever visited)
                    bonus = bonus * self._cc_T[mi]
                r[mi] += bonus.astype(np.float32)
                self.int_paid += float(bonus.sum())
                # count each entry once even when several envs share a cell
                # this tick (np.add.at handles duplicate indices)
                np.add.at(self._counts, mc, 1)
                if self.track_touched:
                    self._touched.append(mc.copy())
            self._prev_cell = cell
        self._ticks += self.every
        self.n_success += int(goal.sum())
        self.n_fail += int((done.astype(bool) & ~goal).sum())
        self.n_trunc += int((ended & ~done.astype(bool)).sum())
        if goal.any():
            self.finish_ticks.extend(self._ticks[goal].tolist())
        improved = d < self._best - self.stall_eps
        self._best = np.minimum(self._best, d)
        self._since = np.where(improved, 0, self._since + self.every)
        if self.d_latch > 0.0:
            self._latched |= d <= self.d_latch
        self._d = d
        self._dc = dc
        if self._fr_best is not None:
            # --respawn-frontier's frontier tracker. `ended` rows already
            # hold the NEW episode's spawn d, so the pair is emitted BEFORE
            # the running minimum is allowed to see it - the bug that would
            # otherwise make every episode look like it reached its own
            # successor's spawn.
            if ended.any():
                _ei = np.flatnonzero(ended)
                self.fr_pairs.extend(zip(self._fr_spawn[_ei].tolist(),
                                         self._fr_best[_ei].tolist(),
                                         self._fr_spawn_spd[_ei].tolist()))
            _nm = ~ended
            self._fr_best[_nm] = np.minimum(self._fr_best[_nm], d[_nm])
            if ended.any():
                self._fr_best[ended] = d[ended]
                self._fr_spawn[ended] = d[ended]
                self._fr_spawn_spd[ended] = s[ended]
        if ended.any():
            if self.arc is not None:
                ei = np.flatnonzero(ended)
                nt = np.maximum(self._ticks[ei], 1).astype(np.float64)
                self.arc_gain.extend(
                    (self._arc_max[ei] - self._arc_spawn[ei]).tolist())
                self.arc_reach.extend(self._arc_max[ei].tolist())
                self.arc_off_frac.extend(
                    (self._arc_off[ei] * float(self.every) / nt).tolist())
                # a respawn relocates arbitrarily: the local window cannot
                # follow it, so the anchor is rebuilt with a global search
                self.arc.reset(_states(core)["origin"], mask=ended)
                self._arc_spawn[ended] = self.arc.arc[ended]
                self._arc_max[ended] = self.arc.arc[ended]
                self._arc_off[ended] = 0
            self._best[ended] = d[ended]
            if self._vz is not None:
                # --surf-bonus / --dive-pen: the vz tracker re-anchors on
                # the fresh spawn, or the first tick of every new episode
                # reads a cross-episode velocity jump as a giant "the map
                # pushed back" (the rule _prev_v follows in
                # AcroCoverageReward, for the same reason)
                self._vz[ended] = v[ended, 2]
            if self.ratchet:
                # the ended rows already hold the NEW episode's spawn: the
                # record restarts there, so the fresh episode's first call
                # pays only what it actually gains from its own start
                self._rec[ended] = d[ended]
            if self._d0 is not None:
                self._d0[ended] = d[ended]
            self._since[ended] = 0
            self._ticks[ended] = 0
            if self.d_latch > 0.0:
                # ended rows already hold the NEW episode's spawn, so
                # this both CLEARS the flag and re-arms it when that
                # spawn is itself inside the shell
                self._latched[ended] = d[ended] <= self.d_latch
            if self._cc_T is not None:
                # --curiosity-cond: the ended rows start a NEW episode
                # here, with a fresh T. The old one is kept in _cc_boot
                # for the truncation bootstrap (the terminal row of the
                # episode that just ended still carries ITS T).
                ei = np.flatnonzero(ended)
                self._cc_boot[ei] = self._cc_T[ei]
                self._cc_ep[ei] += 1
                self._cc_T[ei] = self._cc_draw(ei)
                self._cc_apply()
        if self._dip is not None:
            # dip/* diagnostic - LAST, after every term of `r` is final and
            # every piece of episode state has been updated, so it can only
            # READ. On an ended row `d` already holds the NEXT episode's
            # spawn and the dying episode's last state is the previous
            # call's: that is exactly DipMeter.update's contract. `dt` is
            # taken live because --tick-ms-schedule moves tick_ms during a
            # run.
            self._dip.update(d, ended, self.scale,
                             float(self.every) * self.tick_ms / 1000.0)
        return r

    # -- --curiosity-cond -----------------------------------------------------
    def _cc_draw(self, idx) -> np.ndarray:
        idx = np.asarray(idx, np.int64)
        if self._cc_fixed is not None:
            return self._cc_fixed[idx].astype(np.float64)
        return cc_draw(self.cc_seed, idx, self._cc_ep[idx], self.cc_p0,
                       self.cc_tmin, self.cc_tmax)

    def _cc_apply(self) -> None:
        self._cc_shape = (1.0 - self._cc_T / self.cc_tmax).astype(np.float32)

    def set_cc_T(self, values) -> None:
        """Pin every env's T to ``values`` (one per env, in [0, cc_tmax]):
        the eval mirrors and the tools use it (T = 0 is the exploit
        member). The pin also replaces the mixture draw at every later
        episode start. ``None`` returns to the mixture from the next
        episode start on. Needs the flag (cc_tmax > 0)."""
        if self.cc_tmax <= 0.0:
            raise ValueError("set_cc_T needs --curiosity-cond (cc_tmax > 0)")
        if values is None:
            self._cc_fixed = None
            return
        v = np.asarray(values, np.float64).reshape(-1)
        if np.any(v < 0.0) or np.any(v > self.cc_tmax):
            raise ValueError(f"T must lie in [0, {self.cc_tmax:g}]")
        self._cc_fixed = v
        if self._cc_T is not None:
            if len(v) != len(self._cc_T):
                raise ValueError(f"set_cc_T: {len(v)} values for "
                                 f"{len(self._cc_T)} envs")
            self._cc_T = v.copy()
            self._cc_boot = v.copy()
            self._cc_apply()

    def cc_T(self) -> np.ndarray | None:
        """The live exploration weight per env (None before on_reset, or
        without the flag)."""
        return self._cc_T

    def cc_obs(self) -> np.ndarray | None:
        """The observation column, float32 (N,): t = log1p(T)/log1p(T_max)
        of the state the next decision acts on (like :meth:`latch_flags`,
        read after the reward call)."""
        if self._cc_T is None:
            return None
        return cc_encode(self._cc_T, self.cc_tmax).astype(np.float32)

    def cc_boot(self) -> np.ndarray | None:
        """The T of the episode that ENDED at the last call, per env (the
        live T for envs whose episode did not end) - the truncation
        bootstrap's reconstructed terminal row needs the old one."""
        return self._cc_boot

    def cc_obs_boot(self, idx, live: bool = False) -> np.ndarray:
        """The column at a truncated episode's terminal state for the envs
        ``idx``: from :meth:`cc_boot` (the reward has already been called
        for this tick and moved the rows on), or from the live vector
        (``live=True``: the per-decision reward path calls the reward
        AFTER the bootstrap, so nothing has moved yet)."""
        src = self._cc_T if live else self._cc_boot
        return cc_encode(src[np.asarray(idx, np.int64)],
                         self.cc_tmax).astype(np.float32)

    def cc_summary(self) -> dict:
        """``frac0`` (share of envs at T = 0), ``T_mean`` over all envs,
        ``T_pos_mean`` over the T > 0 envs, ``T_max``."""
        T = self._cc_T
        if T is None or len(T) == 0:
            return {"frac0": float("nan"), "T_mean": float("nan"),
                    "T_pos_mean": float("nan"), "T_max": float("nan")}
        pos = T[T > 0.0]
        return {"frac0": float(np.mean(T == 0.0)), "T_mean": float(T.mean()),
                "T_pos_mean": float(pos.mean()) if len(pos) else 0.0,
                "T_max": float(T.max())}

    def set_d0(self, idx, values) -> None:
        """Per-env potential origin for the death charge (goal arms): the
        bank an episode forfeits at death is scale*(d0 - d_last)."""
        if self._d0 is not None:
            self._d0[np.asarray(idx, np.int64)] = np.asarray(values, np.float64)

    def dist_now(self) -> np.ndarray | None:
        """The geodesic distance this reward is shaping on, one per env, as
        of the LAST call - i.e. of the state the next decision acts on.

        Read by --priv-critic (surfgym/privfeat.py): the critic is given the
        very number the reward is built from rather than a second sample of
        the field, so the two can never disagree. None before the first
        :meth:`on_reset`.
        """
        return self._d

    def latch_flags(self) -> np.ndarray | None:
        """The ``--race-latch`` flag as it stands, one bool per env.

        This is the observation feature. The trainer writes it into the
        last column of the route block after every reward call, so the
        row the policy acts on at t carries the flag that decides
        whether t+1 pays shaping. Without it the switch is invisible
        episode history and one input has two different returns."""
        return self._latched

    def ratchet_gap(self) -> np.ndarray | None:
        """``(d_t - b_t) / d0`` per env - the ``--race-ratchet`` observation
        column, as of the state the next decision acts on.

        Exactly 0 at a new record and positive by how far the episode has
        backed off it. Read AFTER the reward call (fill_vision), like
        :meth:`latch_flags`: by then ``_rec`` already carries this state's
        own distance and the ended rows have been restarted at their new
        spawn."""
        if not self.ratchet or self._d is None:
            return None
        return ((self._d - self._rec) / self.ratchet_d0).astype(np.float32)

    def ratchet_boot(self) -> np.ndarray | None:
        """The RECORD one reward call ago (raw units, not the column).

        The truncation bootstrap rebuilds ``s_T`` from outside this call,
        after the autoreset has already restarted ``_rec`` on the ended
        rows; the terminal row's column is
        ``(d_T - min(ratchet_boot, d_T)) / d0``."""
        if not self.ratchet:
            return None
        return self._rec_boot

    def ratchet_gap_of(self, d_t, rec_prev) -> np.ndarray:
        """The column at an arbitrary reconstructed state: the record rule
        applied once to ``d_t`` from ``rec_prev``. One implementation, used
        by the trainer's terminal row and by the tests."""
        d_t = np.asarray(d_t, np.float64)
        rec = np.minimum(np.asarray(rec_prev, np.float64), d_t)
        return ((d_t - rec) / self.ratchet_d0).astype(np.float32)

    def latch_boot(self) -> np.ndarray | None:
        """The flag one call ago - what the terminal state of a
        TRUNCATED episode carried. The trainer's V(s_T) bootstrap
        rebuilds that row from the terminal scalars, after the autoreset
        has already moved the live states (and :meth:`latch_flags`) on
        to the next episode's spawn."""
        return self._latch_boot

    def stagnant_mask(self, ticks: int | None = None) -> np.ndarray | None:
        """Envs that haven't improved their best distance for ``ticks`` —
        respawn snapshots taken there would seed provably-stuck states.
        Default: 3 s (300 ticks at 10 ms, scaled by the tick)."""
        if self._since is None:
            return None
        if ticks is None:
            ticks = 300 if self.tick_ms == 10.0 else int(3000.0 / self.tick_ms)
        return self._since >= ticks

    def pop_stall_mask(self) -> np.ndarray | None:
        """uint8 mask of envs past the stagnation window, for
        ``core.force_fail``; counters re-arm so a kill fires once."""
        if self._since is None or self.stall_ticks <= 0:
            return None
        stall = self._since >= self.stall_ticks
        if not stall.any():
            return None
        self._since[stall] = 0
        return stall.astype(np.uint8)

    def pop_dip_raw(self) -> dict | None:
        """Drain the ``dip/*`` window: the raw per-dip arrays, so pooling
        over maps is a concatenation and the percentiles are taken once over
        the pool. ``None`` when the diagnostic is off (``dip=False``, or a
        per-env goal potential). See :mod:`surfgym.dipmeter`."""
        if self._dip is None:
            return None
        return self._dip.pop_raw()

    def pop_stats(self) -> dict:
        """Episode outcomes since the last call (per-iteration logging)."""
        n_ep = self.n_success + self.n_fail + self.n_trunc
        out = {
            "success_rate": (self.n_success / n_ep) if n_ep else float("nan"),
            "finish_s": (float(np.mean(self.finish_ticks))
                         / (1000.0 / self.tick_ms)
                         if self.finish_ticks else float("nan")),
            "episodes": n_ep,
            "int_per_ep": (self.int_paid / n_ep) if n_ep else float("nan"),
        }
        if self.surf_air_ticks > 0.0:
            # the fraction of AIRBORNE ticks the bonus actually paid on.
            # Saturating near 1.0 means the agent found a way to be
            # permanently "in contact" - a farm, not surfing - and is the
            # first thing to look at on a pnSURF-family arm.
            out["surf_paid_frac"] = self.surf_paid_ticks / self.surf_air_ticks
            out["dive_frac"] = self.dive_ticks / self.surf_air_ticks
            # the RAW counts as well, so MapFleet can pool the fraction over
            # maps by airborne TICKS rather than averaging two per-map means
            out["surf_air_ticks"] = self.surf_air_ticks
            out["surf_paid_ticks"] = self.surf_paid_ticks
            out["surf_dive_ticks"] = self.dive_ticks
        self.surf_paid_ticks = self.dive_ticks = self.surf_air_ticks = 0.0
        if self.arc is not None:
            out["arc_gain"] = (float(np.mean(self.arc_gain))
                               if self.arc_gain else float("nan"))
            out["arc_reach"] = (float(np.max(self.arc_reach))
                                if self.arc_reach else float("nan"))
            out["arc_p90"] = (float(np.percentile(self.arc_reach, 90))
                              if self.arc_reach else float("nan"))
            out["arc_off"] = (float(np.mean(self.arc_off_frac))
                              if self.arc_off_frac else float("nan"))
            self.arc_gain.clear()
            self.arc_reach.clear()
            self.arc_off_frac.clear()
        if self._fr_best is not None:
            # --respawn-frontier: four numbers, and they are meant to be
            # read TOGETHER (CLAUDE.md's win-rate trap).
            #   front_pmax     the START-ANCHORED frontier - the deepest
            #                  progress reached by an episode that spawned
            #                  at the true map start. This is what drives
            #                  the cap, and no frontier spawn can inflate
            #                  it.
            #   front_pmax_all the same over EVERY episode. The gap between
            #                  the two IS the curriculum's reach, and if
            #                  win rate rises while only this one moves,
            #                  the run is measuring the harvest.
            #   front_spawn_*  median / p90 of the REALISED spawn progress
            #                  over every episode that ended - the pool,
            #                  the reservoir contamination and all.
            if self.fr_pairs:
                sp = np.asarray([p[0] for p in self.fr_pairs], np.float64)
                bs = np.asarray([p[1] for p in self.fr_pairs], np.float64)
                anch = sp >= self.frontier_d0 - self.frontier_start_eps
                if self.frontier_anchor_speed > 0.0:
                    # --respawn-frontier-anchor: a start is a start FROM
                    # REST. A frontier row that landed near d0 is launched
                    # at reservoir speed x U(0.9, 5) and can fly further
                    # than any real start; it must not be the frontier.
                    spd = np.asarray([p[2] for p in self.fr_pairs],
                                     np.float64)
                    anch &= spd <= self.frontier_anchor_speed
                self._fr_anch_reach = (self.frontier_d0
                                       - bs[anch]).astype(np.float64)
                out["front_pmax"] = (float(self.frontier_d0 - bs[anch].min())
                                     if anch.any() else float("nan"))
                out["front_anch_eps"] = int(anch.sum())
                out["front_pmax_all"] = float(self.frontier_d0 - bs.min())
                pr = self.frontier_d0 - sp
                out["front_spawn_med"] = float(np.median(pr))
                out["front_spawn_p90"] = float(np.percentile(pr, 90))
            else:
                self._fr_anch_reach = np.zeros(0, np.float64)
                out["front_pmax"] = float("nan")
                out["front_anch_eps"] = 0
                out["front_pmax_all"] = float("nan")
                out["front_spawn_med"] = float("nan")
                out["front_spawn_p90"] = float("nan")
            self.fr_pairs.clear()
        self.n_success = self.n_fail = self.n_trunc = 0
        self.int_paid = 0.0
        self.finish_ticks.clear()
        return out

    def pop_frontier_reaches(self) -> np.ndarray:
        """--respawn-frontier-quantile: the START-ANCHORED reaches (progress
        d0 - best d, map units) of every anchored episode counted by the
        last :meth:`pop_stats`, one entry per episode, then cleared. The
        trainer keeps a window of these and takes a QUANTILE for P_max, so
        one lucky episode - or one first finish - does not move the cap:
        the frontier is where start episodes usually get to."""
        r = self._fr_anch_reach
        self._fr_anch_reach = np.zeros(0, np.float64)
        return r

    # -- DDP fleet metrics (docs/ddp-plan.md step 12a) ----------------------
    # A per-rank success rate is a win rate over N/R envs: same expectation,
    # R x the variance of the single-GPU baseline it gets plotted against.
    # The trainer all-reduces this raw vector and recomputes the rates from
    # fleet totals (a list of finish ticks cannot be all-reduced, hence the
    # running sum + count representation).
    def stats_vector(self) -> np.ndarray:
        """Raw outcome counters since the last clear, as a reducible f64
        vector: [n_success, n_fail, n_trunc, int_paid, finish_ticks_sum,
        finish_count]."""
        return np.array([self.n_success, self.n_fail, self.n_trunc,
                         self.int_paid, float(sum(self.finish_ticks)),
                         float(len(self.finish_ticks))], np.float64)

    def clear_stats(self) -> None:
        self.n_success = self.n_fail = self.n_trunc = 0
        self.int_paid = 0.0
        self.finish_ticks.clear()

    @staticmethod
    def stats_from_vector(v, tick_ms: float = 10.0) -> dict:
        """Same shape as :meth:`pop_stats`, from a (fleet-summed) vector.
        ``tick_ms`` turns the summed finish TICKS into seconds."""
        n_ep = float(v[0] + v[1] + v[2])
        return {
            "success_rate": (v[0] / n_ep) if n_ep else float("nan"),
            "finish_s": ((v[4] / v[5] / (1000.0 / float(tick_ms)))
                         if v[5] else float("nan")),
            "episodes": n_ep,
            "int_per_ep": (v[3] / n_ep) if n_ep else float("nan"),
        }


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


def map_spawn_pool(core: SurfCore, yaw: np.ndarray | float | None = None
                   ) -> np.ndarray:
    """Game-authentic spawns: the map's own spawn points, standing, as a
    player joining the server gets them. The race objective uses these — the
    run must start where the map says runs start. ``yaw`` overrides the
    (often unreliable) entity yaw; pass e.g. ``GoalField.descent_yaw`` output
    to face the track."""
    from .core import SurfState  # noqa: F401  (STATE_DTYPE is module-level)

    spawns = list(core.spawns())
    if not spawns:
        raise RuntimeError("map_spawn_pool: map has no spawn points")
    pool = np.zeros(len(spawns), dtype=STATE_DTYPE)
    for i, (origin, syaw) in enumerate(spawns):
        pool[i]["origin"] = origin
        pool[i]["yaw"] = syaw
        pool[i]["onground"] = -1
    if yaw is not None:
        pool["yaw"] = yaw
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
