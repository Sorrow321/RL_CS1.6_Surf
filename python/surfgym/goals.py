"""goals.py - per-env goals, per-env reference lines, and the curriculum that
decides how far away a goal is allowed to be.

``route.py`` gives the agent ONE line through ONE map, fixed at startup: the
map has a start and a finish and the reference line runs between them.
Goal-conditioned training does not work that way. Every env gets its OWN goal
- a sphere somewhere in flyable air - and its OWN reference line ending at
that goal, redrawn on every reset, and the policy has to read "where does MY
line go next" out of the same 27 numbers it already reads.

What is here:

* :class:`MultiLine` - RouteLine's lookahead fan with the line index moved
  into the batch. The same math, the same yaw convention (src/env.c:247), the
  same per-point normalization and the same clamp, so a policy trained on
  RouteLine features and one trained on these read the SAME observation and
  the goal-conditioned arm stays comparable to every number in the ledger.
  The equivalence is asserted numerically in tests/python/test_goals.py, not
  by inspection, because two copies of a featurizer drift otherwise.
* :func:`chord_line` / :func:`segment_line` - the two ways a line to a goal
  gets built when no recording of it exists. Round 18's finding is what makes
  them enough: **the reference line supplies the ORDERING, not the line**.
  xAUTO's 58 straight chords, 1,131 u off the champion and a quarter of its
  vertices inside solid geometry, matched the full champion line on every
  axis. A chord to a goal the agent has never reached is in exactly that
  class of object.
* :class:`SphereGoals` - the goal test. A sphere, not a box, because a goal
  drawn in open air has no natural axes to align a box to.
* :class:`AirSampler` - rejection sampling of goal positions inside a
  reachability predicate.
* :class:`KCurriculum` - how far away, in SECONDS of flight, the next goal is
  allowed to be, moved by the measured success rate at the FAR end of the
  current band (Florensa's rule: expand while the frontier is being solved).
* :class:`GoalStats` - what to print, split by goal kind and by distance,
  because an aggregate success rate over a curriculum whose band is moving is
  not a comparable number between two logs.

Only numpy at import time; torch is imported inside the methods that need it,
the same way route.py keeps ``import surfgym`` dependency-free.
"""
from __future__ import annotations

import numpy as np

from .route import DEFAULT_OFFSETS, DEFAULT_SPACING, resample_polyline

def resample_polyline_np(pts, spacing=None):
    """Constant-spacing resample of a raw polyline -> float32 (L, 3),
    L >= 2 (the route-goal line slice, before it enters MultiLine)."""
    from .route import DEFAULT_SPACING, resample_polyline
    out, _ = resample_polyline(np.asarray(pts, np.float64),
                               spacing or DEFAULT_SPACING)
    if len(out) < 2:
        p = np.asarray(pts, np.float64)
        out = np.vstack([p[0], p[-1] + np.array([0.0, 0.0, 1e-3])])
    return np.asarray(out, np.float32)


__all__ = ["MultiLine", "SphereGoals", "AirSampler", "KCurriculum",
           "GoalStats", "chord_line", "segment_line"]

# A degenerate chord (start == goal) still has to be a LINE - two distinct
# points - or the featurizer's tangent collapses and resample_polyline
# refuses it. 1 u is below any physical displacement in this game (a tick of
# champion flight is ~35 u) and far above the 1e-6 the tangent norm is
# clamped at, so the fallback line is exact rather than clamped.
_DEGENERATE_DZ = 1.0


class MultiLine:
    """N reference polylines, one per env, behind RouteLine's featurizer.

    ``pts`` is (N, l_max, 3) with the first ``length[n]`` rows of row n valid;
    the rest is padding that no read may ever reach. Storage is a single dense
    block rather than a list of arrays because the featurizer is a per-tick
    cost on the training path: one padded tensor makes the nearest-vertex
    search one batched matmul, where a ragged list would be N small ones.

    The price of the padding is that EVERY index into ``pts`` has to be
    clamped per env instead of per line - the nearest-vertex argmin masks the
    invalid tail with +inf, the tangent index clamps at ``length - 2`` and the
    lookahead index clamps at ``length - 1``. Miss one of those and an env
    with a short line reads a neighbour's leftovers, which looks like a
    perfectly plausible route and is silent.

    Lines start as a degenerate 2-point line at the origin (``length = 2``)
    so ``features`` is finite before the first :meth:`set_lines`: the zero
    tangent hits the 1e-6 norm clamp and the projection is 0, never NaN.
    """

    def __init__(self, n_envs, l_max: int = 768,
                 spacing: float = DEFAULT_SPACING, offsets=DEFAULT_OFFSETS,
                 speed_floor: float = 500.0, near_scale: float = 500.0,
                 clamp: float = 4.0, device=None):
        import torch
        self.n_envs = int(n_envs)
        self.l_max = int(l_max)
        if self.n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {self.n_envs}")
        if self.l_max < 2:
            raise ValueError(f"l_max must be >= 2, got {self.l_max}")
        self.spacing = float(spacing)
        self.offsets = tuple(float(o) for o in offsets)
        self.speed_floor = float(speed_floor)
        self.near_scale = float(near_scale)
        self.clamp = float(clamp)
        self.pts = torch.zeros((self.n_envs, self.l_max, 3),
                               dtype=torch.float32, device=device)
        self.length = torch.full((self.n_envs,), 2, dtype=torch.int64,
                                 device=device)
        self.t = torch.as_tensor(self.offsets, dtype=torch.float32,
                                 device=device)
        # column indices, held once: the validity mask is rebuilt every call
        # (the lines change every reset) and allocating an arange per tick on
        # the training path is not free
        self._col = torch.arange(self.l_max, device=device)

    # ----------------------------------------------------------------- build
    def set_lines(self, idx, lines) -> None:
        """Install ``lines[k]`` into env ``idx[k]``.

        ``lines`` are ALREADY resampled at ``self.spacing`` - the featurizer's
        arc lookup is pure index arithmetic and is wrong on a line that is
        not uniform, so resampling is the caller's job (:func:`chord_line`,
        :func:`segment_line`, ``route.resample_polyline``).

        One padded numpy block and one host->device copy for the whole batch:
        a reset can touch a thousand envs at once, and a thousand small
        copies is a thousand synchronizations. The pad is the line's own last
        point repeated, not zeros - it is masked by ``length`` and never
        read, but if a clamp is ever missed the failure is "the env stares at
        its goal" instead of "the env flies at the map origin".
        """
        import torch
        idx = np.asarray(idx, np.int64).reshape(-1)
        lines = list(lines)
        if len(idx) != len(lines):
            raise ValueError(f"set_lines: {len(idx)} indices but "
                             f"{len(lines)} lines")
        if len(idx) == 0:
            return
        if idx.min() < 0 or idx.max() >= self.n_envs:
            raise ValueError(f"set_lines: env index out of range "
                             f"[0, {self.n_envs}): min {idx.min()}, "
                             f"max {idx.max()}")
        arrs = []
        lens = np.empty(len(lines), np.int64)
        for k, ln in enumerate(lines):
            a = np.ascontiguousarray(np.asarray(ln, np.float32))
            if a.ndim != 2 or a.shape[1] != 3:
                raise ValueError(f"set_lines: line {k} has shape {a.shape}, "
                                 f"want (L, 3)")
            if len(a) < 2 or len(a) > self.l_max:
                raise ValueError(f"set_lines: line {k} has L={len(a)}, need "
                                 f"2 <= L <= l_max={self.l_max}")
            arrs.append(a)
            lens[k] = len(a)
        w = int(lens.max())
        block = np.empty((len(idx), w, 3), np.float32)
        for k, a in enumerate(arrs):
            block[k, :lens[k]] = a
            block[k, lens[k]:] = a[-1]
        dev = self.pts.device
        it = torch.as_tensor(idx, device=dev)
        self.pts[it, :w] = torch.as_tensor(block, device=dev)
        self.length[it] = torch.as_tensor(lens, device=dev)

    # -------------------------------------------------------------- features
    @property
    def n_features(self) -> int:
        """3 dims per point: the nearest point, then one per horizon."""
        return 3 * (1 + len(self.offsets))

    def describe(self) -> str:
        lens = self.length.detach().to("cpu").numpy()
        return (f"multiline {self.n_envs} envs x <= {self.l_max} pts @ "
                f"{self.spacing:g}u (arc "
                f"{float((lens.min() - 1) * self.spacing):,.0f}-"
                f"{float((lens.max() - 1) * self.spacing):,.0f}u), "
                f"{len(self.offsets)} horizons "
                f"{self.offsets[0]:g}-{self.offsets[-1]:g}s -> "
                f"{self.n_features} features")

    def _gather1(self, idx):
        """pts[n, idx[n]] for an (N,) index -> (N, 3)."""
        return self.pts.gather(1, idx.reshape(-1, 1, 1).expand(-1, 1,
                                                               3)).squeeze(1)

    def _gather2(self, idx):
        """pts[n, idx[n, m]] for an (N, M) index -> (N, M, 3)."""
        return self.pts.gather(1, idx.unsqueeze(2).expand(-1, -1, 3))

    def _anchor(self, origin):
        """-> (nearest VALID vertex index, continuous arc coordinate).

        The tail mask is the whole difference from RouteLine: a padded vertex
        can sit anywhere - it is the previous line's geometry, or a repeat of
        this line's endpoint - and a global argmin would happily pick one.
        Masking the squared distance with +inf AFTER it is computed also
        swallows a NaN left in the tail, which an arithmetic mask would not.
        """
        import torch
        p = origin
        if p.shape[0] != self.n_envs:
            raise ValueError(f"features: got {p.shape[0]} rows for "
                             f"{self.n_envs} envs")
        # |R|^2 - 2 p.R, same expansion RouteLine uses: the argmin never
        # materializes the (N, L, 3) difference. |R|^2 is recomputed rather
        # than cached because the lines are rewritten on every reset and a
        # stale cache here is a silently wrong nearest vertex.
        sq = (self.pts * self.pts).sum(2)                       # (N, l_max)
        dot = torch.matmul(self.pts, p.unsqueeze(2)).squeeze(2)  # (N, l_max)
        valid = self._col.unsqueeze(0) < self.length.unsqueeze(1)
        d2 = (sq - 2.0 * dot).masked_fill(~valid, float("inf"))
        i0 = d2.argmin(dim=1)                                   # (N,)
        # Refine to a continuous ARC coordinate along the local tangent.
        # Without it the fan inherits the vertex snap (up to spacing/2 = 64 u)
        # which at the shortest horizon (125 u at the speed floor) is most of
        # the feature.
        ti = torch.minimum(i0, self.length - 2)
        p0 = self._gather1(ti)
        tang = self._gather1(ti + 1) - p0
        tl = tang.norm(dim=1).clamp_min(1e-6)
        proj = ((p - p0) * tang).sum(1) / tl
        s0 = ti.to(p.dtype) * self.spacing + proj.clamp(0.0, self.spacing)
        return i0, s0

    def features(self, origin, yaw_deg, speed):
        """(N,3) positions + (N,) yaw in DEGREES + (N,) horizontal speed
        -> (N, n_features) ego-frame lookahead, on ``origin``'s device.

        Identical in every arithmetic step to ``RouteLine.features`` - see
        that docstring for why the span scales with speed, why each point is
        normalized by its own nominal span, and why forward is
        (cos yaw, sin yaw). The only additions are per-env clamps where
        RouteLine has one global L.
        """
        import torch
        p = origin
        i0, s0 = self._anchor(p)
        scale = torch.clamp(speed, min=self.speed_floor)        # (N,)
        span = scale.unsqueeze(1) * self.t.unsqueeze(0)         # (N, M)
        last = self.length - 1                                  # (N,)
        fidx = torch.minimum(
            torch.clamp((s0.unsqueeze(1) + span) / self.spacing, min=0.0),
            last.to(p.dtype).unsqueeze(1))
        lo = fidx.floor()
        w = (fidx - lo).unsqueeze(2)
        li = lo.long()
        hi = torch.minimum(li + 1, last.unsqueeze(1))
        tgt = self._gather2(li) * (1.0 - w) + self._gather2(hi) * w
        near = self._gather1(i0).unsqueeze(1)                   # (N, 1, 3)
        d = torch.cat([near, tgt], dim=1) - p.unsqueeze(1)      # (N, 1+M, 3)

        yr = yaw_deg * (np.pi / 180.0)
        cy, sy = torch.cos(yr).unsqueeze(1), torch.sin(yr).unsqueeze(1)
        fwd = d[:, :, 0] * cy + d[:, :, 1] * sy
        left = -d[:, :, 0] * sy + d[:, :, 1] * cy
        up = d[:, :, 2]
        norm = torch.cat([torch.full_like(scale.unsqueeze(1), self.near_scale),
                          torch.clamp(span, min=self.near_scale)], dim=1)
        out = torch.stack([fwd / norm, left / norm, up / norm], dim=2)
        return torch.clamp(out, -self.clamp, self.clamp).reshape(p.shape[0],
                                                                 -1)

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

    # ------------------------------------------------------------------- arc
    def arc_position(self, origin):
        """(N,3) -> (N,) continuous arc coordinate of the nearest point.

        This is the reward-side quantity: arc length along a line is monotone
        by construction, so unlike a distance-to-goal field it cannot have an
        interior minimum that pays the agent to turn back (ledger round 18,
        route vertex 1601). ``ArcProgress`` is the windowed, order-only
        version of the same coordinate for the single-line case; this one is
        a GLOBAL search, so it is an anchor, not a progress signal - a line
        that folds back on itself will jump.
        """
        return self._anchor(origin)[1]

    def total_arc(self):
        """(N,) arc length of each env's line, in map units."""
        import torch
        return (self.length - 1).to(torch.float32) * self.spacing


# ------------------------------------------------------------- line builders
def chord_line(start, goal, spacing: float = DEFAULT_SPACING):
    """The straight segment start -> goal, resampled at ``spacing``.

    The weakest possible reference line, and round 18 says that is enough:
    xAUTO's 58 straight chords - up to 1,131 u from the champion line and a
    quarter of their vertices inside solid geometry - matched the full
    champion line on corridor MAX, on episodes past the wall and on finishes,
    with a better best time. What a line supplies is the ORDERING.

    A zero-length chord (start == goal, which a sampler can produce) would
    trip resample_polyline's "fewer than 2 distinct points"; it degenerates
    to two points 1 u apart instead, so downstream code never has to special
    case it.
    """
    a = np.asarray(start, np.float64).reshape(3)
    b = np.asarray(goal, np.float64).reshape(3)
    if float(np.linalg.norm(b - a)) <= 1e-6:
        out = np.stack([a, b + np.array([0.0, 0.0, _DEGENERATE_DZ])])
        return out.astype(np.float32)
    pts, _ = resample_polyline(np.stack([a, b]), spacing)
    return pts


def _rdp(p, eps: float):
    """Douglas-Peucker on a 3D polyline -> the kept vertices (a subsequence).

    Iterative with an explicit stack, not recursive: a raw trajectory is
    thousands of points long and the recursion depth of the classic form is
    O(K) in the worst case (a monotone staircase), which is a RecursionError
    in the middle of a reset rather than at import.

    The distance is to the SEGMENT (the parameter is clamped to [0, 1]), not
    to the infinite line through the endpoints. On a flight line that doubles
    back - and surf lines do - a point far beyond an endpoint has a small
    perpendicular distance to the infinite line and would be dropped, taking
    the doubling-back with it.
    """
    p = np.asarray(p, np.float64).reshape(-1, 3)
    n = len(p)
    if n < 2:
        raise ValueError(f"segment_line needs at least 2 points, got {n}")
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a = p[i]
        ab = p[j] - a
        seg = p[i + 1:j] - a
        den = float(ab @ ab)
        if den <= 1e-12:                      # the two ends coincide
            dist = np.linalg.norm(seg, axis=1)
        else:
            t = np.clip((seg @ ab) / den, 0.0, 1.0)
            dist = np.linalg.norm(seg - t[:, None] * ab, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return p[keep]


def segment_line(points, spacing: float = DEFAULT_SPACING,
                 rdp_eps: float = 512.0):
    """Simplify a raw path to its skeleton, then resample it at ``spacing``.

    For a line built out of something noisy - a recorded flight, a
    steepest-descent walk down a voxel field, a chain of sampled waypoints -
    the noise is not information: the featurizer reads DIRECTION, so a jitter
    of tens of units on a 128 u sampling turns into lookahead vectors that
    swing about a route that is actually straight. Douglas-Peucker keeps the
    corners that carry the ordering and drops everything the agent cannot act
    on.

    ``rdp_eps`` defaults to 4x the resample spacing, so the simplification
    never removes a feature the resampled line could have represented, and
    never keeps one it could not.
    """
    skel = _rdp(points, float(rdp_eps))
    pts, _ = resample_polyline(skel, spacing)
    return pts


# -------------------------------------------------------------------- goals
class SphereGoals:
    """Per-env goal spheres and the arrival test.

    A sphere rather than an AABB because a goal drawn in open air has no
    axes to align a box to, and because the test then has no direction
    dependence: "within r of the point" is the same requirement whichever way
    the agent arrives. Map finish zones stay AABBs - they are map geometry
    (CLAUDE.md section 4b) - and this is a different object.

    ``center`` is NaN and ``active`` False until :meth:`set`. Both matter:
    the NaN makes an unset goal impossible to hit even if something marks it
    active by mistake, since every comparison against NaN is False.
    """

    def __init__(self, n_envs, radius: float = 192.0):
        self.n_envs = int(n_envs)
        self.center = np.full((self.n_envs, 3), np.nan, np.float32)
        self.radius = np.full(self.n_envs, float(radius), np.float32)
        self.active = np.zeros(self.n_envs, bool)

    def set(self, idx, centers, radius=None) -> None:
        """Place (and activate) the goals of envs ``idx``."""
        idx = np.asarray(idx, np.int64).reshape(-1)
        c = np.asarray(centers, np.float32).reshape(-1, 3)
        if len(c) != len(idx):
            raise ValueError(f"set: {len(idx)} indices but {len(c)} centers")
        self.center[idx] = c
        if radius is not None:
            self.radius[idx] = np.asarray(radius, np.float32)
        self.active[idx] = True

    def clear(self, idx) -> None:
        """Deactivate, and blank the centre so a stale one cannot come back."""
        idx = np.asarray(idx, np.int64).reshape(-1)
        self.active[idx] = False
        self.center[idx] = np.nan

    def hit(self, origin):
        """(N,3) -> (N,) bool: active, and inside (or exactly on) the sphere.

        float64 for the difference: at map coordinates of ~1e4 a float32
        squared distance loses the last couple of units, which is a real
        fraction of a 192 u goal radius.
        """
        p = np.asarray(origin, np.float64).reshape(-1, 3)
        d = p - self.center
        d2 = (d * d).sum(1)
        r = self.radius.astype(np.float64)
        with np.errstate(invalid="ignore"):     # NaN centre -> False, quietly
            return self.active & (d2 <= r * r)


class AirSampler:
    """Uniform goal positions inside a reachability predicate.

    The predicate is the caller's (an occupancy/SDF lookup, a "is this voxel
    in the baked field" test, a clearance threshold); this only does the
    rejection loop, in BATCHES, because the predicate is a vectorized array
    query and calling it once per candidate would cost more than the sampling.

    Failure is raised, not returned short. A sampler that quietly hands back
    fewer goals than asked turns into envs with no goal at all, which reads
    downstream as a policy that stopped finishing - hours later, on a rented
    box. The count in the message is what says whether the predicate is too
    tight or the box is in the wrong place.
    """

    def __init__(self, mins, maxs, reachable_fn, exclude_fn=None,
                 max_tries: int = 64):
        self.mins = np.asarray(mins, np.float64).reshape(3)
        self.maxs = np.asarray(maxs, np.float64).reshape(3)
        if np.any(self.maxs < self.mins):
            raise ValueError(f"AABB is inverted: mins {self.mins}, "
                             f"maxs {self.maxs}")
        self.reachable_fn = reachable_fn
        self.exclude_fn = exclude_fn
        self.max_tries = int(max_tries)

    def _keep(self, cand):
        ok = np.asarray(self.reachable_fn(cand), bool).reshape(-1)
        if len(ok) != len(cand):
            raise ValueError(f"reachable_fn returned {len(ok)} flags for "
                             f"{len(cand)} points")
        if self.exclude_fn is not None:
            bad = np.asarray(self.exclude_fn(cand), bool).reshape(-1)
            if len(bad) != len(cand):
                raise ValueError(f"exclude_fn returned {len(bad)} flags for "
                                 f"{len(cand)} points")
            ok = ok & ~bad
        return cand[ok]

    def _collect(self, n, draw):
        n = int(n)
        if n <= 0:
            return np.zeros((0, 3), np.float32)
        got, have = [], 0
        for _ in range(self.max_tries):
            kept = self._keep(draw(n))
            if len(kept):
                got.append(kept)
                have += len(kept)
            if have >= n:
                break
        if have < n:
            raise RuntimeError(
                f"AirSampler: only {have} of {n} candidates passed in "
                f"{self.max_tries} batches of {n} - predicate too tight, or "
                f"the AABB {self.mins.tolist()}..{self.maxs.tolist()} is in "
                f"the wrong place")
        return np.ascontiguousarray(np.concatenate(got)[:n], np.float32)

    def sample(self, n, rng):
        """-> (n, 3) float32 uniform in the AABB and passing the predicates."""
        return self._collect(
            n, lambda m: rng.uniform(self.mins, self.maxs, (m, 3)))

    def sample_near(self, n, anchor, r_min, r_max, rng):
        """-> (n, 3) drawn from a spherical shell around ``anchor``.

        This is the curriculum's sampler: "a goal about k seconds away" is a
        shell, not a box. The radius is uniform in r rather than in volume,
        which biases toward the near edge of the shell on purpose - the point
        of a band is the whole band, and volume-uniform sampling would put
        7/8 of the draws in the outer half of a 1:2 shell.

        Points are CLIPPED to the AABB, so against a tight box the result can
        land inside r_min. The predicates still hold; the shell does not.
        """
        a = np.asarray(anchor, np.float64).reshape(3)
        lo, hi = float(r_min), float(r_max)

        def draw(m):
            u = rng.normal(size=(m, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True).clip(1e-12)
            r = rng.uniform(lo, hi, (m, 1))
            return np.clip(a + u * r, self.mins, self.maxs)

        return self._collect(n, draw)


class KCurriculum:
    """How far away, in SECONDS of flight, the next goal may be.

    Florensa et al. (1705.06366) train on goals of INTERMEDIATE difficulty -
    the ones the current policy solves sometimes - and move the band as the
    policy improves. The band here is a time, not a distance, because the
    thing that makes a goal hard on this task is how much flight it takes to
    reach, and a straight-line distance divided by a speed the agent does not
    hold is not that.

    Only the TOP THIRD of the band votes. The near end of the band is solved
    by construction after the first expansion - counting it would let a wide
    band's easy majority carry the average over ``hi`` forever, expanding
    until the far end is hopeless. The frontier is what has to be succeeding.

    ``k_min`` is fixed: an agent that stops being able to reach a 1-second
    goal has a problem no curriculum should paper over.
    """

    def __init__(self, k_min: float = 1.0, k_max: float = 5.0,
                 k_cap: float = 60.0, k_floor: float = 2.0,
                 step: float = 2.0, lo: float = 0.10, hi: float = 0.50,
                 min_episodes: int = 64):
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.k_cap = float(k_cap)
        self.k_floor = float(k_floor)
        self.step = float(step)
        self.lo = float(lo)
        self.hi = float(hi)
        self.min_episodes = int(min_episodes)
        self.n = 0
        self.wins = 0
        self.updates = 0

    def band_lo(self) -> float:
        """Lower edge of the top third of the current band."""
        return self.k_min + (2.0 / 3.0) * (self.k_max - self.k_min)

    def draw(self, n, rng):
        """-> (n,) float64 seconds, uniform over the whole band.

        The whole band, not the top third: the near end is the retention set.
        Dropping it is how a curriculum forgets what it could already do.
        """
        return rng.uniform(self.k_min, max(self.k_min, self.k_max), int(n))

    def note(self, k, success) -> None:
        """Record episode outcomes, keeping only the top-third draws.

        Scalars or arrays: envs report in batches and unpacking them at the
        call site is how a loop ends up in the reset path.
        """
        kk = np.atleast_1d(np.asarray(k, np.float64))
        ss = np.atleast_1d(np.asarray(success, bool))
        if ss.shape != kk.shape:
            ss = np.broadcast_to(ss, kk.shape)
        top = kk >= self.band_lo()
        self.n += int(top.sum())
        self.wins += int((top & ss).sum())

    def update(self):
        """Move the far edge if enough frontier episodes have been seen.

        -> the new ``k_max``. Below ``min_episodes`` nothing moves AND
        nothing is reset: a band whose sample is too small to read is not
        evidence for holding still either, so the outcomes keep accumulating
        until they are.
        """
        if self.n < self.min_episodes:
            return self.k_max
        rate = self.wins / float(self.n)
        if rate > self.hi:
            self.k_max = min(self.k_cap, self.k_max + self.step)
        elif rate < self.lo:
            self.k_max = max(self.k_floor, self.k_max - self.step)
        self.n = 0
        self.wins = 0
        self.updates += 1
        return self.k_max

    def state(self) -> dict:
        """Flat dict for the progress csv."""
        return {"k_min": self.k_min, "k_max": self.k_max,
                "k_band_lo": self.band_lo(), "k_n": self.n,
                "k_wins": self.wins,
                "k_rate": (self.wins / float(self.n) if self.n
                           else float("nan")),
                "k_updates": self.updates}


class GoalStats:
    """Success accounting for goal episodes, split by KIND and by DISTANCE.

    One aggregate success rate is not a comparable number here, for the same
    reason ``race/win_rate`` was not (ledger round 19): the rate can rise
    because the curriculum handed out easier goals. Splitting by k-bin makes
    that visible - a rate that rises inside a FIXED bin is the policy, a rate
    that rises only in aggregate is the harvest.

    The kinds are the three things a goal episode can be asked to do:
    ``achieved`` (a sampled goal the agent reached), ``air`` (a goal in open
    air with no map feature at it) and ``finish`` (the map's own finish zone).
    They are not pooled by default because a null on one says nothing about
    the others - the same reason type-1 and type-3 map finishes must not be
    aggregated.
    """

    KINDS = ("achieved", "air", "finish")
    # seconds; the last bin is open-ended
    EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, float("inf"))
    LABELS = ("0-2", "2-5", "5-10", "10-20", "20+")

    def __init__(self):
        self._k = []
        self._kind = []
        self._ok = []
        self._ticks = []

    def note(self, k, kind, success, ticks) -> None:
        """One finished goal episode.

        An unknown ``kind`` raises rather than being bucketed away: a typo in
        a logging call is invisible in the output and silently removes a
        whole arm's episodes from the table.
        """
        if kind not in self.KINDS:
            raise ValueError(f"unknown goal kind {kind!r}, "
                             f"want one of {self.KINDS}")
        self._k.append(float(k))
        self._kind.append(kind)
        self._ok.append(bool(success))
        self._ticks.append(float(ticks))

    def __len__(self) -> int:
        return len(self._k)

    def pop(self) -> dict:
        """-> the table, and clear. NaN wherever there is no data."""
        k = np.asarray(self._k, np.float64)
        kinds = np.asarray(self._kind, dtype=object)
        ok = np.asarray(self._ok, bool)
        ticks = np.asarray(self._ticks, np.float64)
        self.__init__()
        n = len(k)
        nan = float("nan")

        def rate(mask):
            m = int(mask.sum())
            return (float(ok[mask].sum()) / m) if m else nan

        by_kind = {}
        for kind in self.KINDS:
            m = (kinds == kind) if n else np.zeros(0, bool)
            by_kind[kind] = {"n": int(m.sum()), "success_rate": rate(m)}

        # np.digitize on the interior edges: bin index IS the bucket, and a
        # k below the first edge (which should not happen) lands in bin 0
        # rather than being dropped
        b = (np.digitize(k, np.asarray(self.EDGES[1:-1]))
             if n else np.zeros(0, np.int64))
        counts = np.zeros(len(self.LABELS), np.int64)
        rates = np.full(len(self.LABELS), nan)
        for i in range(len(self.LABELS)):
            m = (b == i) if n else np.zeros(0, bool)
            counts[i] = int(m.sum())
            rates[i] = rate(m)

        return {"n": n,
                "success_rate": (float(ok.sum()) / n) if n else nan,
                "ticks_mean": (float(ticks[ok].mean()) if ok.any() else nan),
                "kind": by_kind,
                "k_bins": {"labels": list(self.LABELS),
                           "edges": list(self.EDGES),
                           "n": counts, "success_rate": rates}}
