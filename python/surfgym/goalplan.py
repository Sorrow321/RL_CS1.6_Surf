"""goalplan.py - the deterministic BFS PLANNER behind ``--goal-planner bfs``.

The user's design (2026-09-23): "we need the planner ... something that can
produce some potential movement directions, like a polyline ... and then the
executor that actually executes this plan ... the planner is actually
deterministic. So it's not trainable ... selecting some point in the map and
drawing the path towards that point using just some BFS ... And the executor
just executes it." The executor is the goal-conditioned PPO policy that
already reads a per-env polyline through the lookahead FAN
(:class:`surfgym.goals.MultiLine`); this file is the planner that writes the
polyline. Nothing here is trained and nothing here reads a demo, a route
file or a map-specific constant.

WALKABLE GRAPH (built once at startup, on the run's goal-cell occupancy -
vision's slab-catching grid, the one the geodesic goal field is baked on):

* a NODE is a FREE cell with SOLID within ``PLAN_SUPPORT_U`` (64 u) below it,
  i.e. within ``r = max(1, round(64 / cell))`` cells. 64 u is a PLAYER
  constant, not a map one: the standing hull's origin rides 36 u above the
  floor, so the cell the origin sits in is the first or second cell above
  the floor at every cell size this project uses (32 / 48 / 64);
* the node and the solid that supports it both sit ABOVE THE KILL CEILING:
  the top of the highest kill volume (a trigger_teleport with a live
  destination, or a trigger_hurt with dmg >= 90 - the simulator's own rule,
  :func:`surfgym.zones.kill_zones`). A fall net supports nothing. A map
  with no kill volume has no ceiling;
* an EDGE joins two nodes whose (dz, dy, dx) offset is in the 26-cell
  neighbourhood - the 8 horizontal neighbours plus one-cell vertical steps,
  so ramps and stairs connect - and only when the whole axis-aligned box
  the two cells span is free (no squeezing diagonally past a solid corner).
  Its cost is the Euclidean step length, cell * sqrt(dx^2 + dy^2 + dz^2).

TARGETS AND FIELDS. ``n_targets`` nodes drawn uniformly (seeded) from the
graph, plus THE FINISH: every node inside the ARMED finish box, inflated
exactly like the geodesic field's seed box (``goalfield._zone_seed_box``),
is a source, so the finish field is the planner's distance to ENTERING the
box - the real task, not a point near it. One Dijkstra distance field per
target over the graph (numba, parallel over targets), float32
``(n_targets + 1, n_nodes)``; unreachable nodes hold +inf.

PLANS. From a start position: snap it to the nearest node (Euclidean), walk
the steepest descent of the target's field (the neighbour minimising
``d[v] + w``, which is the shortest-path predecessor), lift every node to
the START's height above its floor (a 2-D plan that follows the ground),
prepend the start itself, append the box centre for the finish, simplify
with Douglas-Peucker at ONE CELL (the lattice's own staircase is the noise;
every corridor corner is larger) and resample at the fan's 128 u spacing
with the existing helpers (:func:`surfgym.goals.segment_line`).

POTENTIAL (``--goal-reward plan``, :class:`PlanDistField`): per env, that
env's own target field sampled trilinearly over the WALKABLE corners (the
honest-corner rule of ``GoalField.sample``); a position with no walkable
corner reads the nearest node's value plus the Euclidean offset to it. This
is the planner's exact cost-to-go to its target, so the shaping it pays
telescopes along the very plan the fan shows.
"""
from __future__ import annotations

import time
from typing import NamedTuple, Optional

import numpy as np

__all__ = ["BFSPlanner", "Plan", "PlanDistField", "PLAN_SUPPORT_U",
           "PLAN_SEED_OFFSET", "kill_ceiling", "walkable_mask",
           "parse_fan_offsets", "make_plan_hooks"]

# Support reach below a node, map units. The standing player's origin is 36 u
# above the floor (half the 72 u hull); 64 u reaches the cell that origin sits
# in plus one cell of quantisation slack at 32 u. A physics constant of the
# player, identical on every map.
PLAN_SUPPORT_U = 64.0

# The target draw is seeded from the run's --seed plus this offset (and the
# recorder uses the same, so a recording builds the trainer's target set).
PLAN_SEED_OFFSET = 9173

# Refuse a planner whose distance table alone would exceed this (bytes). The
# labyrinths need ~1.4 MB; a whole surf map at cell 32 can need tens of GB,
# which is a job for a coarser --goal-cell or fewer --goal-plan-targets, not
# an OOM on a rented box.
PLAN_MEM_CAP = 4 << 30


def parse_fan_offsets(spec):
    """``--goal-fan-offsets``: "0.25,0.5,..." (or a list restored from a
    checkpoint config) -> a tuple of positive, strictly increasing seconds.
    None stays None (today's offsets)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        parts = list(spec)
    try:
        offs = tuple(float(p) for p in parts)
    except (TypeError, ValueError):
        raise ValueError(f"--goal-fan-offsets: not a comma list of seconds: "
                         f"{spec!r}")
    if not offs:
        raise ValueError("--goal-fan-offsets: empty list")
    if any(not np.isfinite(o) or o <= 0.0 for o in offs):
        raise ValueError(f"--goal-fan-offsets: every offset must be a "
                         f"positive number of seconds, got {offs}")
    if any(b <= a for a, b in zip(offs, offs[1:])):
        raise ValueError(f"--goal-fan-offsets: offsets must be strictly "
                         f"increasing, got {offs}")
    return offs


def kill_ceiling(bsp_path) -> float:
    """Top of the highest KILL volume (surfgym.zones.kill_zones: destful
    teleports and dmg >= 90 hurts, the simulator's own rule), world z;
    -inf when the map has none. A brush entity's origin is added to its
    model's box."""
    from .zones import kill_zones
    top = -np.inf
    for k in kill_zones(bsp_path):
        oz = float((k.get("origin") or [0.0, 0.0, 0.0])[2])
        top = max(top, float(k["maxs"][2]) + oz)
    return float(top)


def walkable_mask(solid, mins, cell, kill_z=-np.inf,
                  support_u=PLAN_SUPPORT_U):
    """(nz, ny, nx) bool solid -> (walkable mask, support reach in cells,
    per-cell floor height).

    Walkable = free, with a live solid within ``r`` cells BELOW, and a voxel
    centre above the kill ceiling. The floor height of a walkable cell is the
    TOP face of the nearest supporting solid cell (NaN elsewhere)."""
    solid = np.asarray(solid, bool)
    nz = solid.shape[0]
    zc = float(mins[2]) + (np.arange(nz) + 0.5) * float(cell)
    dead = (zc <= float(kill_z))[:, None, None]
    r = max(1, int(round(float(support_u) / float(cell))))
    live = solid & ~dead                       # a fall net supports nothing
    floor = np.full(solid.shape, np.nan, np.float64)
    # nearest support wins: fill from the farthest reach to the nearest
    for s in range(r, 0, -1):
        m = np.zeros_like(solid)
        m[s:] = live[:-s]                      # solid s cells below
        top = float(mins[2]) + (np.arange(nz) - s + 1) * float(cell)
        floor = np.where(m, top[:, None, None], floor)
    walk = (~solid) & np.isfinite(floor) & ~dead
    floor = np.where(walk, floor, np.nan)
    return walk, r, floor


def _build_kernels():
    """The numba kernels. Everything the planner does per call is here, so
    the per-episode cost is microseconds, not a Python loop over the path."""
    from numba import njit, prange

    @njit(cache=True)
    def neighbours(node_of, solid, coords):
        """(M, 26) int32 neighbour table, -1 where there is no edge. The
        offset order is (dz, dy, dx) in -1..1 nested, (0,0,0) skipped - the
        order _offset_weights() uses."""
        m = coords.shape[0]
        nz, ny, nx = node_of.shape
        nbr = np.full((m, 26), -1, np.int32)
        for i in range(m):
            z = coords[i, 0]
            y = coords[i, 1]
            x = coords[i, 2]
            k = -1
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        k += 1
                        z2 = z + dz
                        y2 = y + dy
                        x2 = x + dx
                        if (z2 < 0 or z2 >= nz or y2 < 0 or y2 >= ny
                                or x2 < 0 or x2 >= nx):
                            continue
                        j = node_of[z2, y2, x2]
                        if j < 0:
                            continue
                        ok = True
                        for az in range(min(z, z2), max(z, z2) + 1):
                            for ay in range(min(y, y2), max(y, y2) + 1):
                                for ax in range(min(x, x2), max(x, x2) + 1):
                                    if solid[az, ay, ax]:
                                        ok = False
                        if ok:
                            nbr[i, k] = j
        return nbr

    @njit(cache=True)
    def _sift_up(heap, pos, key, i):
        v = heap[i]
        kv = key[v]
        while i > 0:
            p = (i - 1) >> 1
            u = heap[p]
            if key[u] <= kv:
                break
            heap[i] = u
            pos[u] = i
            i = p
        heap[i] = v
        pos[v] = i

    @njit(cache=True)
    def _sift_down(heap, pos, key, i, size):
        v = heap[i]
        kv = key[v]
        while True:
            c = 2 * i + 1
            if c >= size:
                break
            if c + 1 < size and key[heap[c + 1]] < key[heap[c]]:
                c += 1
            u = heap[c]
            if key[u] >= kv:
                break
            heap[i] = u
            pos[u] = i
            i = c
        heap[i] = v
        pos[v] = i

    @njit(cache=True)
    def dijkstra(nbr, wk, sources, out):
        """Single-source-SET Dijkstra with an indexed binary heap (the heap
        never holds more than M entries, whatever the degree)."""
        m = nbr.shape[0]
        key = np.full(m, np.inf)
        heap = np.empty(m, np.int64)
        pos = np.full(m, -1, np.int64)
        done = np.zeros(m, np.bool_)
        size = 0
        for s in sources:
            if key[s] > 0.0:
                key[s] = 0.0
                heap[size] = s
                pos[s] = size
                size += 1
                _sift_up(heap, pos, key, size - 1)
        while size > 0:
            u = heap[0]
            size -= 1
            pos[u] = -1
            done[u] = True
            if size > 0:
                last = heap[size]
                heap[0] = last
                pos[last] = 0
                _sift_down(heap, pos, key, 0, size)
            du = key[u]
            for k in range(nbr.shape[1]):
                v = nbr[u, k]
                if v < 0 or done[v]:
                    continue
                nd = du + wk[k]
                if nd < key[v]:
                    key[v] = nd
                    if pos[v] < 0:
                        heap[size] = v
                        pos[v] = size
                        size += 1
                    _sift_up(heap, pos, key, pos[v])
        for i in range(m):
            out[i] = key[i]

    @njit(cache=True, parallel=True)
    def fields(nbr, wk, src, src_ptr, out):
        for t in prange(out.shape[0]):
            dijkstra(nbr, wk, src[src_ptr[t]:src_ptr[t + 1]], out[t])

    @njit(cache=True)
    def descend(d, s, nbr, wk):
        """Steepest descent of field ``d`` from node ``s``: the neighbour
        minimising d[v] + w (the shortest-path predecessor), strictly
        downhill, until a source (d == 0). -> node indices, s first."""
        m = d.shape[0]
        path = np.empty(m, np.int64)
        path[0] = s
        n = 1
        cur = s
        while d[cur] > 0.0 and n < m:
            best = np.inf
            bv = -1
            for k in range(nbr.shape[1]):
                v = nbr[cur, k]
                if v < 0:
                    continue
                if not (d[v] < d[cur]):
                    continue
                val = d[v] + wk[k]
                if val < best:
                    best = val
                    bv = v
            if bv < 0:
                break
            cur = bv
            path[n] = cur
            n += 1
        return path[:n]

    @njit(cache=True)
    def sample(pos, tgt, dtab, rmax, node_of, mins, cell, out, miss):
        """Per env: its target's field, trilinear over the WALKABLE corners
        that carry a finite distance (GoalField.sample's honest-corner rule;
        corners outside the grid are simply absent). ``miss`` flags rows
        with no such corner, for the nearest-node fallback."""
        n = pos.shape[0]
        nz, ny, nx = node_of.shape
        for i in range(n):
            t = tgt[i]
            miss[i] = False
            if t < 0:
                out[i] = 0.0
                continue
            lim = rmax[t]
            gx = (pos[i, 0] - mins[0]) / cell - 0.5
            gy = (pos[i, 1] - mins[1]) / cell - 0.5
            gz = (pos[i, 2] - mins[2]) / cell - 0.5
            ix0 = int(np.floor(gx))
            iy0 = int(np.floor(gy))
            iz0 = int(np.floor(gz))
            fx = gx - ix0
            fy = gy - iy0
            fz = gz - iz0
            num = 0.0
            den = 0.0
            for dz in range(2):
                kz = iz0 + dz
                if kz < 0 or kz >= nz:
                    continue
                wz = fz if dz == 1 else 1.0 - fz
                for dy in range(2):
                    ky = iy0 + dy
                    if ky < 0 or ky >= ny:
                        continue
                    wy = fy if dy == 1 else 1.0 - fy
                    for dx in range(2):
                        kx = ix0 + dx
                        if kx < 0 or kx >= nx:
                            continue
                        j = node_of[kz, ky, kx]
                        if j < 0:
                            continue
                        v = dtab[t, j]
                        if not (v <= lim):
                            continue
                        wx = fx if dx == 1 else 1.0 - fx
                        w = (wx * wy) * wz
                        num += w * v
                        den += w
            if den > 1e-6:
                out[i] = num / den
            else:
                out[i] = 0.0
                miss[i] = True

    return neighbours, fields, descend, sample


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _build_kernels()
    return _KERNELS


def _offset_weights(cell: float) -> np.ndarray:
    """Edge cost per neighbour-table column, in the kernel's offset order."""
    w = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                w.append(float(cell) * float(np.sqrt(dz * dz + dy * dy
                                                     + dx * dx)))
    return np.asarray(w, np.float64)


class Plan(NamedTuple):
    """One planned goal. ``line`` is ready for MultiLine / MultiArcProgress
    (resampled at the fan spacing, starting at the start position)."""
    line: np.ndarray          # (L, 3) float32
    goal: np.ndarray          # (3,) float64: the target point / box centre
    length: float             # graph distance from the snapped start, u
    target: int               # row of BFSPlanner.dist
    finish: bool              # the target is the finish box
    start: int                # the snapped start node
    raw: np.ndarray           # (P, 3) the lifted path before simplification


class BFSPlanner:
    """The walkable graph, its target fields, and plans on it.

    ``occ`` is the (nz, ny, nx) occupancy (nonzero = solid) with voxel
    ``(iz, iy, ix)`` centred at ``mins + (i + 0.5) * cell``; ``finish_box``
    is the armed finish zone (``{"mins", "maxs"}``) or None. Rows
    ``0 .. n_targets-1`` of :attr:`dist` are the random targets and row
    :attr:`fin` (when there is a finish) is the finish box."""

    def __init__(self, occ, mins, cell, finish_box=None, kill_z=-np.inf,
                 n_targets: int = 256, seed: int = 0,
                 spacing: Optional[float] = None,
                 mem_cap: int = PLAN_MEM_CAP):
        from scipy.spatial import cKDTree
        from .route import DEFAULT_SPACING
        t0 = time.perf_counter()
        neighbours, fields, self._descend, self._sample = _kernels()
        solid = np.ascontiguousarray(np.asarray(occ) != 0)
        self.mins = np.asarray(mins, np.float64).reshape(3)
        self.cell = float(cell)
        self.kill_z = float(kill_z)
        self.spacing = float(spacing or DEFAULT_SPACING)
        walk, self.support_cells, floor = walkable_mask(
            solid, self.mins, self.cell, self.kill_z)
        coords = np.argwhere(walk).astype(np.int64)          # C order
        m = len(coords)
        if m == 0:
            raise RuntimeError(
                "planner: no walkable cell (free, solid within "
                f"{PLAN_SUPPORT_U:g} u below, above the kill ceiling "
                f"z={self.kill_z:g}) - wrong occupancy or wrong cell?")
        self.n_nodes = m
        self.shape = solid.shape
        # the solid grid itself (a reference, no copy): the surf planner's
        # occupancy slabs and its 3-D solid-crossing diagnostic read it
        # (surfgym/goalsurf.py); nothing on the walkable path does
        self.solid = solid
        self.node_of = np.full(solid.shape, -1, np.int32)
        self.node_of[walk] = np.arange(m, dtype=np.int32)
        self.coords = coords
        self.xyz = self.mins[None, :] + (coords[:, ::-1] + 0.5) * self.cell
        self.floor = floor[walk].astype(np.float64)
        self.nbr = neighbours(self.node_of, solid, coords)
        self.wk = _offset_weights(self.cell)
        self.n_edges = int((self.nbr >= 0).sum())
        self._tree = cKDTree(self.xyz)
        self.fin = None
        self.finish_center = None
        self.finish_seed = "none"
        self.finish_nodes = np.zeros(0, np.int64)
        if finish_box is not None:
            from .goalfield import _zone_seed_box
            lo, hi = _zone_seed_box(finish_box, self.cell)
            inside = np.all((self.xyz >= lo[None, :])
                            & (self.xyz <= hi[None, :]), axis=1)
            fn = np.flatnonzero(inside).astype(np.int64)
            self.finish_center = 0.5 * (np.asarray(finish_box["mins"],
                                                   np.float64)
                                        + np.asarray(finish_box["maxs"],
                                                     np.float64))
            self.finish_seed = "box"
            if len(fn) == 0:
                # no walkable cell inside the (inflated) box - e.g. a button
                # hanging in the air: the node nearest its centre stands in
                fn = np.asarray([int(self._tree.query(self.finish_center)[1])],
                                np.int64)
                self.finish_seed = "nearest"
            self.finish_nodes = fn
        # random targets: uniform over the graph's nodes OUTSIDE the finish
        # box (the finish has its own share, and a target inside the armed
        # box could never be reached - the box ends the episode first),
        # seeded, distinct
        pool = np.setdiff1d(np.arange(m, dtype=np.int64), self.finish_nodes)
        rng = np.random.default_rng(int(seed))
        nt = int(min(max(0, int(n_targets)), len(pool)))
        self.targets = (np.sort(rng.choice(pool, size=nt, replace=False))
                        if nt else np.zeros(0, np.int64)).astype(np.int64)
        self.n_rand = nt
        srcs = [np.asarray([t], np.int64) for t in self.targets]
        if finish_box is not None:
            srcs.append(self.finish_nodes)
            self.fin = len(srcs) - 1
        self.n_fields = len(srcs)
        need = self.n_fields * m * 4
        if need > mem_cap:
            raise RuntimeError(
                f"planner: {self.n_fields} fields x {m:,} nodes x 4 B = "
                f"{need / 2**30:.1f} GiB of distance table, over the "
                f"{mem_cap / 2**30:.0f} GiB cap - raise --goal-cell or lower "
                f"--goal-plan-targets")
        src_ptr = np.zeros(self.n_fields + 1, np.int64)
        for i, s in enumerate(srcs):
            src_ptr[i + 1] = src_ptr[i] + len(s)
        src = (np.concatenate(srcs) if srcs else np.zeros(0, np.int64))
        self.dist = np.empty((self.n_fields, m), np.float32)
        t1 = time.perf_counter()
        if self.n_fields:
            fields(self.nbr, self.wk, src, src_ptr, self.dist)
        self.field_secs = time.perf_counter() - t1
        fin_ok = np.isfinite(self.dist)
        self.rmax = np.array([float(self.dist[i][fin_ok[i]].max())
                              if fin_ok[i].any() else 0.0
                              for i in range(self.n_fields)], np.float64)
        self.reach_frac = (float(fin_ok.mean()) if self.n_fields else 1.0)
        self.build_secs = time.perf_counter() - t0
        self.mem_bytes = int(self.dist.nbytes + self.nbr.nbytes
                             + self.node_of.nbytes + self.xyz.nbytes
                             + self.floor.nbytes + self.coords.nbytes)

    # ------------------------------------------------------------ building
    @classmethod
    def for_core(cls, core, cell, finish_box, n_targets: int = 256,
                 seed: int = 0, cache_dir=None, **kw):
        """The planner of ``core``'s map at goal cell ``cell``: the goal
        graph's own occupancy (cached next to the bsp) and the map's kill
        ceiling."""
        from .goalfield import goal_occupancy
        occ, mins = goal_occupancy(core, float(cell), cache_dir)
        return cls(occ, mins, float(cell), finish_box=finish_box,
                   kill_z=kill_ceiling(core.bsp_path), n_targets=n_targets,
                   seed=seed, **kw)

    def describe(self) -> str:
        kz = ("none" if not np.isfinite(self.kill_z)
              else f"z={self.kill_z:,.0f}")
        fin = ("no finish box" if self.fin is None else
               f"finish = {len(self.finish_nodes)} node(s) "
               f"({self.finish_seed}), start->finish reach "
               f"{self.rmax[self.fin]:,.0f} u max")
        return (f"planner bfs: {self.n_nodes:,} walkable nodes at cell "
                f"{self.cell:g} (free, solid within {self.support_cells} "
                f"cell(s) = {PLAN_SUPPORT_U:g} u below, kill ceiling {kz}), "
                f"{self.n_edges:,} directed edges (26-nbhd, |dz| <= 1, no "
                f"corner cutting), {self.n_rand} random targets + {fin}; "
                f"{self.n_fields} Dijkstra fields, "
                f"{self.dist.nbytes / 2**20:.2f} MiB table "
                f"({self.mem_bytes / 2**20:.2f} MiB total), "
                f"{100.0 * self.reach_frac:.1f}% of (field, node) pairs "
                f"reachable; built in {self.build_secs:.2f} s "
                f"(fields {self.field_secs:.2f} s)")

    # ------------------------------------------------------------- queries
    def snap(self, pos) -> np.ndarray:
        """(k, 3) positions -> (k,) nearest node (Euclidean)."""
        p = np.atleast_2d(np.asarray(pos, np.float64))
        return np.asarray(self._tree.query(p)[1], np.int64).reshape(-1)

    def node_distance(self, t: int, s: int) -> float:
        return float(self.dist[int(t), int(s)])

    def choose(self, s: int, rng, p_finish: float, dmin: float,
               dmax: float) -> int:
        """The training draw for a start node ``s``: the finish with
        probability ``p_finish`` (when it is reachable), else a random target
        whose path length from ``s`` lies in [dmin, dmax], uniform among the
        eligible. Fallbacks: the reachable target nearest the band, then the
        finish, then -1 (nothing reachable). Always draws exactly one
        uniform, plus one integer when it picks among the eligible."""
        s = int(s)
        coin = float(rng.random())
        fin_ok = self.fin is not None and np.isfinite(self.dist[self.fin, s])
        if fin_ok and coin < float(p_finish):
            return int(self.fin)
        col = self.dist[:self.n_rand, s].astype(np.float64)
        ok = np.isfinite(col) & (col >= float(dmin)) & (col <= float(dmax))
        if ok.any():
            cand = np.flatnonzero(ok)
            return int(cand[int(rng.integers(len(cand)))])
        reach = np.isfinite(col) & (col > 0.0)
        if reach.any():
            gap = (np.maximum(float(dmin) - col, 0.0)
                   + np.maximum(col - float(dmax), 0.0))
            gap[~reach] = np.inf
            return int(np.argmin(gap))
        if fin_ok:
            return int(self.fin)
        return -1

    def plan(self, origin, target: int) -> Optional[Plan]:
        """The plan from ``origin`` (3,) to row ``target``; None when the
        target is unreachable from the snapped start."""
        o = np.asarray(origin, np.float64).reshape(3)
        t = int(target)
        s = int(self.snap(o[None, :])[0])
        d = self.dist[t]
        if not np.isfinite(d[s]):
            return None
        nodes = np.asarray(self._descend(d, s, self.nbr, self.wk), np.int64)
        fin = self.fin is not None and t == self.fin
        h = float(o[2] - self.floor[s])          # the start's height above
        pts = self.xyz[nodes].copy()             # its floor, kept all along
        pts[:, 2] = self.floor[nodes] + h
        raw = np.vstack([o[None, :], pts[1:]])
        if fin:
            g = self.finish_center.copy()
            g[2] = float(self.floor[nodes[-1]] + h)
            raw = np.vstack([raw, g[None, :]])
        else:
            g = pts[-1].copy()
        line = self.line_from(raw, o, g)
        return Plan(line=line, goal=g, length=float(d[s]), target=t,
                    finish=bool(fin), start=s, raw=raw)

    def line_from(self, raw, o, g) -> np.ndarray:
        """Douglas-Peucker at one cell, then the constant-spacing resample
        (the existing goals.segment_line); a degenerate path falls back to
        the straight chord."""
        from .goals import chord_line, segment_line
        try:
            return np.asarray(segment_line(raw, self.spacing,
                                           rdp_eps=self.cell), np.float32)
        except ValueError:
            return np.asarray(chord_line(o, g, self.spacing), np.float32)

    def chord_plan(self, origin, goal, target: int, finish: bool) -> Plan:
        """The fallback when the graph cannot reach the target: the straight
        chord (the goal system's air-goal line)."""
        o = np.asarray(origin, np.float64).reshape(3)
        g = np.asarray(goal, np.float64).reshape(3)
        line = self.line_from(np.vstack([o, g]), o, g)
        return Plan(line=line, goal=g, length=float(np.linalg.norm(g - o)),
                    target=int(target), finish=bool(finish),
                    start=int(self.snap(o[None, :])[0]),
                    raw=np.vstack([o, g]))

    def sample(self, pos, tgt) -> np.ndarray:
        """(n, 3) positions and (n,) target rows -> (n,) float64 cost-to-go.
        -1 targets read 0. See the module docstring for the rule."""
        p = np.ascontiguousarray(np.atleast_2d(pos), np.float64)
        tt = np.ascontiguousarray(np.asarray(tgt, np.int64).reshape(-1))
        if len(tt) != len(p):
            raise ValueError(f"sample: {len(p)} positions, {len(tt)} targets")
        out = np.empty(len(p), np.float64)
        miss = np.empty(len(p), np.bool_)
        self._sample(p, tt, self.dist, self.rmax, self.node_of, self.mins,
                     self.cell, out, miss)
        if miss.any():
            mi = np.flatnonzero(miss)
            dd, nn = self._tree.query(p[mi])
            v = self.dist[tt[mi], np.asarray(nn, np.int64)].astype(np.float64)
            bad = ~np.isfinite(v)
            if bad.any():
                # the target cannot be reached from the nearest node: a
                # finite sentinel past the field's own reach, like
                # GoalField's, so the reward never sees inf
                v[bad] = self.rmax[tt[mi][bad]] + 2.0 * self.cell
            out[mi] = v + np.asarray(dd, np.float64)
        return out


def make_plan_hooks(planner: BFSPlanner, core, ev: dict, *, line=None,
                    ball=None, dist_field=None, radius: float = 192.0,
                    finish_radius: float = 192.0, dmin: float = 256.0,
                    dmax: float = 4096.0, rng=None,
                    random_targets: bool = False,
                    replan_len: Optional[float] = None, act_every: int = 1,
                    tick_ms: float = 10.0):
    """(episode_meta, on_tick) for ``record_rollout`` on a core whose env 0
    is being recorded - the ONE implementation of the planner's eval, shared
    by the trainer (GoalSystem.plan_eval_hooks) and tools/record_ckpt.py.

    The HEADLINE (``random_targets=False``) is the real task: from wherever
    the core spawned env 0 (the map start, on the trainer's eval core and in
    a default recording), the goal is the finish BOX and the line is the
    planner's path to it; success is the core's own box test - exactly what
    the race eval metrics count - so no sphere is armed. The secondary eval
    (``random_targets=True``) draws BFSPlanner.choose with no finish share
    and arms a sphere of ``radius`` at the target; entry force-fails env 0
    like every other goal eval. ``ev`` is the caller's tally dict (n, succ,
    ticks, dists, pending, center, t0), updated in place."""
    rng = rng if rng is not None else np.random.default_rng(0)
    ev.update({"n": 0, "succ": 0, "pending": False, "center": None,
               "ticks": [], "dists": [], "t0": 0, "box": False})
    P = planner
    # replan_len (u): the BFS planner on the LEARNED planner's schedule -
    # every call plans from the agent's CURRENT position and hands the
    # executor only the first replan_len u of that path; a plan closes when
    # 90% of its arc is covered, when its budget (length / 250 u/s x 1.5)
    # runs out, or at the episode's end, and the next is issued at the next
    # executor decision (goallearn.PlanState's rule, shared tracker). None
    # = the original single plan per episode, bit for bit.
    RP = replan_len is not None and float(replan_len) > 0.0
    if RP:
        from .goalarc import MultiArcProgress
        from .goallearn import BUDGET_MULT, BUDGET_SPEED_U, COMPLETE_FRAC
        _keep = max(2, int(round(float(replan_len) / float(P.spacing))) + 1)
        _trk = MultiArcProgress(1, l_max=_keep + 1, spacing=float(P.spacing),
                                corridor=float(radius), window=16)
        _budget = int(np.ceil((_keep - 1) * float(P.spacing) / BUDGET_SPEED_U
                              * BUDGET_MULT * 1000.0 / float(tick_ms) - 1e-6))
        _K = max(1, int(act_every))
        rp = {"active": False, "need": False, "elapsed": 0, "target": -1}
        ev.update({"plans": 0, "closed": 0, "complete": 0, "plan_log": [],
                   "tick": 0, "ep_cur": 0, "shapes": [], "wall": 0})

    def _issue(o, t):
        """Plan from ``o`` to target row ``t``, cut to the first replan_len u
        (RP) -> (the full Plan, the line handed to the executor)."""
        pl = P.plan(o, t) if t >= 0 else None
        if pl is None:
            g0 = (P.finish_center if P.finish_center is not None
                  else o + np.array([0.0, 0.0, 1.0]))
            pl = P.chord_plan(o, g0, -1, P.finish_center is not None)
        ln = np.asarray(pl.line, np.float32)
        if RP:
            ln = ln[:_keep]
            _trk.set_lines(np.array([0]), [ln])
            rp.update(active=True, need=False, elapsed=0)
            ev["plans"] += 1
            ev["shapes"].append(-1)
            ev["plan_log"].append({"ep": int(ev["ep_cur"]),
                                   "tick": int(ev["tick"]), "shape": -1,
                                   "anchor": [float(v) for v in o],
                                   "line": [[float(v) for v in q] for q in ln]})
        if line is not None:
            line.set_lines(np.array([0]), [ln])
        return pl, ln

    def episode_meta(ep):
        o = core.states_view["origin"][0].astype(np.float64)
        s = int(P.snap(o[None, :])[0])
        t = (P.choose(s, rng, 0.0, dmin, dmax) if random_targets
             else (P.fin if P.fin is not None else -1))
        if RP:
            ev["ep_cur"] = int(ev["n"])
            rp["target"] = int(t)
        pl, _ln0 = _issue(o, t)
        g = np.asarray(pl.goal, np.float64)
        if ball is not None:
            ball.set_goals([0], [g])
        if dist_field is not None:
            dist_field.set([0], [g])
        ev["box"] = bool(pl.finish)
        ev["center"] = None if pl.finish else g
        ev["radius"] = float(finish_radius if pl.finish else radius)
        ev["pending"] = False
        ev["n"] += 1
        ev["dists"].append(float(pl.length))
        thin = pl.line[:: max(1, len(pl.line) // 64)]
        return {"goal": {"center": [float(v) for v in g],
                         "radius": float(ev["radius"])},
                "line": [[float(v) for v in p] for p in thin],
                "plan": {"planner": "bfs",
                         "target": ("finish" if pl.finish
                                    else int(pl.target)),
                         "length": round(float(pl.length), 1)}}

    def on_tick(t, states, rewards, done, trunc):
        if RP:
            ev["tick"] = int(t) + 1
        if bool(done[0]) or bool(trunc[0]):
            won = ev["pending"] or (ev["box"] and bool(done[0]) and bool(
                np.asarray(core.goal_hits)[0]))
            if won:
                ev["succ"] += 1
                ev["ticks"].append(t - ev["t0"])
            ev["pending"] = False
            ev["t0"] = t + 1
            if RP and rp["active"]:
                ev["closed"] += 1
                rp["active"] = False
            return
        if RP:
            o = core.states_view["origin"][0].astype(np.float64)
            if rp["active"]:
                _trk.advance(o[None, :].astype(np.float32))
                rp["elapsed"] += 1
                comp = bool(_trk.arc[0] >= COMPLETE_FRAC * _trk.total_arc()[0])
                if comp or rp["elapsed"] >= _budget:
                    ev["closed"] += 1
                    ev["complete"] += int(comp)
                    rp.update(active=False, need=True)
            if rp["need"] and (t + 1) % _K == 0:
                _issue(o, rp["target"])
        if ev["pending"] or ev["center"] is None:
            return
        o = core.states_view["origin"][0].astype(np.float64)
        if np.linalg.norm(o - ev["center"]) <= float(ev["radius"]):
            m = np.zeros(core.num_envs, np.uint8)
            m[0] = 1
            core.force_fail(m)
            ev["pending"] = True

    return episode_meta, on_tick


class PlanDistField:
    """``--goal-reward plan``: the per-env potential RaceReward shapes on -
    the distance along the walkable graph to THIS env's planned target (the
    planner's exact cost-to-go), with the ``GoalField.sample`` interface.
    Row-aligned to the envs like :class:`surfgym.goals.GoalDistField`; the
    goal system sets the targets on every assignment
    (:meth:`set_targets`). An env with no target (-1) reads 0."""

    def __init__(self, planner: BFSPlanner, n_envs: int):
        self.planner = planner
        self.tgt = np.full(int(n_envs), -1, np.int64)
        self.reach_max = float("inf")

    def set_targets(self, idx, targets) -> None:
        idx = np.asarray(idx, np.int64).reshape(-1)
        self.tgt[idx] = np.asarray(targets, np.int64).reshape(-1)

    def sample(self, pos) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pos, np.float64))
        n = len(p)
        if n != len(self.tgt):
            raise ValueError(f"PlanDistField is row-aligned to its "
                             f"{len(self.tgt)} envs; got {n} positions")
        return self.planner.sample(p, self.tgt).astype(np.float32)

    def reachable(self, pos) -> np.ndarray:
        return np.ones(len(np.atleast_2d(pos)), bool)

    def describe(self) -> str:
        return ("goal distance: the PLANNER's own cost-to-go - each env's "
                "target field over the walkable graph (trilinear over "
                "walkable corners; off the graph: nearest node + Euclidean "
                "offset)")
