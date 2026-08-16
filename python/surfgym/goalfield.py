"""goalfield.py — geodesic distance-to-finish over the map's free space.

Euclidean distance to the end zone is a broken race proxy on a winding surf
map: straight-line distance happily decreases through walls or across a
hairpin the track has to swing around, so shaping on it rewards the wrong
direction for whole stages. Instead we run a one-time multi-source shortest
path from the finish zone across every FREE voxel of an occupancy grid
(26-connected, edge cost = euclidean step length) and shape on *that*. The
result is "meters of track left to the finish", which telescopes cleanly
into a potential-based reward.

Correctness over convenience (both learned from adversarial review of v1):

* The goal graph uses its own SLAB-CATCHING occupancy, not vision's
  point-sampled one: each axis is sampled on an 8u lattice (shifted
  occupancy_grid calls, OR-ed), so a 10u floor between two track stages
  reads solid instead of letting the "geodesic" tunnel through it and paint
  a permanent reward trap on the near side. This dilates geometry slightly
  (conservative — false walls only make shaping less generous, never
  farmable); brushes under ~8u can still slip through.
* Solid and unreachable voxels hold a SENTINEL, never bled distances:
  :meth:`GoalField.sample` renormalizes trilinear weights over honest
  corners only. v1 bled values into solids after convergence, which leaked
  low distances through thin walls — the cached field had >100k corrupted
  sites paying fake progress for hugging geometry.

Cost: Bellman-Ford wavefront relaxation on the GPU. For cannonball's 671M
voxels that is ~9-11 GB of VRAM resident (three f32 grids + masks) and a
few minutes once per (map, zone, cell); cached to
``maps/<map>.goal_<cell>.npz`` (uint16, cell/8-unit quantization). A GPU
under ~12 GB will not fit the default 700M-voxel budget — lower it via
``pick_cell(core, budget_voxels=...)`` / ``--lidar-cell``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .vision import _map_sig, slab_occupancy

__all__ = ["GoalField", "EuclidField", "build_goal_field"]

# v3: goal graph shares vision's slab occupancy (thin-entity rasterization
# included) — glass panes are walls for the geodesic, not just for physics
# v4: NOTSOLID func_conveyors excluded from the solid set (src/bsp.c parity)
_GOAL_BUILDER_VERSION = 4


class GoalField:
    """Sampled geodesic distance-to-goal, map units. CPU/numpy — reward math
    runs on 2048 states per tick, 8 gathers is nothing.

    ``grid`` holds honest distances on reachable free voxels and
    ``sentinel`` (> reach_max) on solid or unreachable ones; sampling
    renormalizes over honest corners so values near surfaces are one-sided
    extrapolations, never mixtures with wall interiors."""

    def __init__(self, grid: np.ndarray, mins, cell: float, reach_max: float):
        self.grid = grid                      # (nz, ny, nx) float32, units
        self.mins = np.asarray(mins, np.float64)
        self.cell = float(cell)
        self.reach_max = float(reach_max)     # finite geodesic ceiling
        self.sentinel = float(reach_max + 2.0 * cell)
        self._valid_max = self.reach_max + 0.5 * self.cell
        self.dims = np.array(grid.shape[::-1])  # (nx, ny, nz)

    def sample(self, pos: np.ndarray) -> np.ndarray:
        """Trilinear geodesic distance at ``pos`` (N, 3) -> (N,) float32,
        weighted over honest corners only; ``sentinel`` where no honest
        corner exists (deep wall / disconnected area)."""
        g = (np.atleast_2d(np.asarray(pos, np.float64)) - self.mins) \
            / self.cell - 0.5
        i0 = np.floor(g).astype(np.int64)
        f = (g - i0).astype(np.float32)
        num = np.zeros(len(g), np.float32)
        den = np.zeros(len(g), np.float32)
        nx, ny, nz = self.dims
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix = np.clip(i0[:, 0] + dx, 0, nx - 1)
                    iy = np.clip(i0[:, 1] + dy, 0, ny - 1)
                    iz = np.clip(i0[:, 2] + dz, 0, nz - 1)
                    v = self.grid[iz, iy, ix]
                    w = ((f[:, 0] if dx else 1.0 - f[:, 0])
                         * (f[:, 1] if dy else 1.0 - f[:, 1])
                         * (f[:, 2] if dz else 1.0 - f[:, 2])
                         * (v < self._valid_max))
                    num += w * v
                    den += w
        out = np.where(den > 1e-6, num / np.maximum(den, 1e-6),
                       self.sentinel).astype(np.float32)
        return out

    def reachable(self, pos: np.ndarray) -> np.ndarray:
        """True where a finite path to the goal exists (filters spawn pools
        out of disconnected bonus areas)."""
        return self.sample(pos) < self.reach_max - 0.5 * self.cell

    def descent_yaw(self, pos: np.ndarray) -> np.ndarray:
        """Yaw (deg) of steepest horizontal descent — 'face the track'."""
        p = np.atleast_2d(np.asarray(pos, np.float64))
        h = self.cell
        dx = self.sample(p + [h, 0, 0]) - self.sample(p - [h, 0, 0])
        dy = self.sample(p + [0, h, 0]) - self.sample(p - [0, h, 0])
        return (np.degrees(np.arctan2(-dy, -dx))) % 360.0


class EuclidField:
    """Straight-line distance to the goal AABB, ignoring walls — the A*
    heuristic as a shaping potential. Zero precompute, so it scales to any
    number of maps (the geodesic field costs minutes of GPU per map/zone).

    The trade: it under-informs. Around a hairpin or a wall the true track
    direction can point AWAY from the finish, so immediate shaping goes
    negative and the agent must discover the detour on its own — more
    exploration pressure, slower early learning, but no per-map bake. Same
    interface as :class:`GoalField`; ``reachable`` is trivially True (a
    disconnected bonus area cannot be detected without geometry)."""

    def __init__(self, zone):
        self.mins = np.asarray(zone["mins"], np.float64)
        self.maxs = np.asarray(zone["maxs"], np.float64)

    def sample(self, pos: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pos, np.float64))
        q = np.clip(p, self.mins, self.maxs)       # nearest point of the AABB
        return np.linalg.norm(p - q, axis=1).astype(np.float32)

    def reachable(self, pos: np.ndarray) -> np.ndarray:
        return np.ones(len(np.atleast_2d(pos)), bool)

    def descent_yaw(self, pos: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pos, np.float64))
        q = np.clip(p, self.mins, self.maxs)
        d = q - p
        return (np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360.0


def goal_occupancy(core, cell: float, cache_dir=None):
    """Occupancy for the goal graph — vision's slab-catching grid (per-axis
    8u sampling lattice + exact rasterization of thin solid entities), so a
    glass pane physics collides with is a wall for the geodesic too."""
    return slab_occupancy(core, cell, cache_dir)


def _zone_seed_box(zone, cell: float):
    """Inflate a zone AABB so even a 1u-thin timer curtain covers at least
    one full layer of voxel centers (plus the player-hull reach the env's
    goal test also grants)."""
    grow = max(0.75 * cell, 20.0)
    mins = np.asarray(zone["mins"], np.float64) - grow
    maxs = np.asarray(zone["maxs"], np.float64) + grow
    return mins, maxs


def build_goal_field(core, zone, cell: float, cache_dir=None,
                     device="cuda") -> GoalField:
    """Build (or load) the geodesic distance field toward ``zone``
    (``{"mins": [...], "maxs": [...]}``, map units)."""
    import torch

    bsp = Path(core.bsp_path)
    box = np.round(np.asarray(zone["mins"] + zone["maxs"], np.float64), 1)
    sig = (f"g{_GOAL_BUILDER_VERSION}_{_map_sig(bsp)}_"
           + "_".join(f"{v:g}" for v in box))
    cache = Path(cache_dir) if cache_dir else bsp.parent
    cache_file = cache / f"{bsp.stem}.goal_{cell:g}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sig" in z and str(z["sig"]) == sig:
            q = float(z["quant"])
            grid = z["grid"].astype(np.float32) * q
            return GoalField(grid, z["mins"], float(z["cell"]),
                             float(z["reach_max"]))

    occ, mins = goal_occupancy(core, cell, cache_dir)
    nz, ny, nx = occ.shape
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    solid = torch.as_tensor(occ != 0, device=dev)
    INF = float("inf")
    dpad = torch.full((nz + 2, ny + 2, nx + 2), INF, device=dev)
    d = torch.full((nz, ny, nx), INF, device=dev)

    smin, smax = _zone_seed_box(zone, cell)
    lo = np.maximum(np.floor((smin - mins) / cell - 0.5), 0).astype(int)
    hi = np.minimum(np.ceil((smax - mins) / cell - 0.5),
                    [nx - 1, ny - 1, nz - 1]).astype(int)
    seed = torch.zeros_like(solid)
    seed[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = True
    seed &= ~solid
    n_seed = int(seed.sum())
    if n_seed == 0:
        raise RuntimeError("goal zone contains no free voxel — wrong box?")
    d[seed] = 0.0
    del seed              # min-relaxation can never raise the 0s — no refill

    offsets = [(oz, oy, ox)
               for oz in (-1, 0, 1) for oy in (-1, 0, 1) for ox in (-1, 0, 1)
               if (oz, oy, ox) != (0, 0, 0)]
    weights = [cell * float(np.sqrt(oz * oz + oy * oy + ox * ox))
               for oz, oy, ox in offsets]
    scratch = torch.empty_like(d)

    def relax():
        dpad[1:-1, 1:-1, 1:-1].copy_(d)
        for (oz, oy, ox), w in zip(offsets, weights):
            src = dpad[1 + oz:1 + oz + nz, 1 + oy:1 + oy + ny,
                       1 + ox:1 + ox + nx]
            torch.add(src, w, out=scratch)
            torch.minimum(d, scratch, out=d)
        d.masked_fill_(solid, INF)

    # convergence probe in float64: a float32 sum over ~1e13 has ~1e6-unit
    # ULPs — a slow correction wave improving less than that per window
    # would read as "converged" while distant stages stay overestimated
    prev = None
    it = 0
    while it < 8000:
        relax()
        it += 1
        if it % 64 == 0:
            fin = torch.isfinite(d)
            cur = (int(fin.sum()),
                   float(d[fin].sum(dtype=torch.float64)))
            if cur == prev:
                break
            prev = cur
    reach_max = float(d[torch.isfinite(d)].max())
    print(f"goal field: {it} sweeps, {n_seed} seed voxels, "
          f"max geodesic {reach_max:.0f}u")

    # solid + unreachable stay at the sentinel — sampling renormalizes over
    # honest corners (v1's "bleed into solids" leaked low distances through
    # thin walls and made the shaping farmable)
    sentinel = reach_max + 2.0 * cell
    d.nan_to_num_(posinf=sentinel)
    del dpad, scratch, solid

    quant = cell / 8.0
    d.div_(quant).round_().clamp_(0, 65535)
    grid_q = d.to(torch.int32).cpu().numpy().astype(np.uint16)
    del d
    np.savez_compressed(cache_file, grid=grid_q, quant=np.float32(quant),
                        mins=mins, cell=np.float32(cell),
                        reach_max=np.float32(reach_max), sig=np.str_(sig))
    grid = grid_q.astype(np.float32) * quant
    return GoalField(grid, mins, cell, reach_max)
