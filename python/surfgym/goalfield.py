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

Gravity is the third correctness problem (``gravity_dir=True``, opt-in).
The plain graph is UNDIRECTED, so it happily routes the player back UP a
shaft it can only fall down, and a mid-route pit that drains toward the
finish reads as a low-distance basin the shaping then pulls agents into.
Measured on surf_src_cannonball: the reachable minimum along policy
rollouts is a d~21.5k basin off-route, while the winning line reads
31k -> 107k -> unreachable. The directional mode keeps falling and
air-strafing free but allows a climb only where geometry supports one, so
"distance" means distance the player can actually travel.

Cost: Bellman-Ford wavefront relaxation on the GPU. For cannonball's 671M
voxels that is ~9-11 GB of VRAM resident (three f32 grids + masks) and a
few minutes once per (map, zone, cell); cached to
``maps/<map>.goal_<cell>.npz`` (uint16, cell/8-unit quantization). A GPU
under ~12 GB will not fit the default 700M-voxel budget — lower it via
``pick_cell(core, budget_voxels=...)`` / ``--lidar-cell``. Directional
bakes add two bool grids (~1.4 GB at cannonball's size) and ~30% per sweep.
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

# Directional-graph semantics, versioned SEPARATELY so tightening the climb
# rule invalidates only the goalg_/goalgk_ caches and never the plain ones
# (a 10-minute re-bake per map, times every map in the fleet).
# v1: a forward move with dz > 0 needs either end surface-adjacent.
_GRAVITY_RULE_VERSION = 1

# How far below a voxel a floor still counts as climbable support, in cells.
# 1 is already covered by the 6-neighbourhood; 2 lets the wavefront stand one
# cell off a ledge, which the 32u lattice needs when a ramp only clips a
# voxel's corner and the airspace the player rides is the cell above it.
_SUPPORT_DROP_CELLS = 2


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


def _surface_support(solid, torch, out=None):
    """(nz, ny, nx) bool: voxels where a player could plausibly GAIN height —
    anything with a solid 6-neighbour, or a floor within
    ``_SUPPORT_DROP_CELLS`` below. Ramps, walls, stairs and ledges are
    surface-adjacent; open sky is not, and in open sky the player only falls
    (and strafes sideways) — which is the whole point of the directional
    graph. Solid voxels themselves come out True and are harmless: they hold
    the sentinel regardless.

    ``out`` (a zeroed tensor or view) writes in place — the caller hands in
    the interior of the padded mask so a 671M-voxel bake never holds two
    full bool grids at once."""
    sup = torch.zeros_like(solid) if out is None else out
    sup[:, :, :-1] |= solid[:, :, 1:]      # +x neighbour solid
    sup[:, :, 1:] |= solid[:, :, :-1]      # -x
    sup[:, :-1, :] |= solid[:, 1:, :]      # +y
    sup[:, 1:, :] |= solid[:, :-1, :]      # -y
    sup[:-1, :, :] |= solid[1:, :, :]      # ceiling overhead
    for k in range(1, _SUPPORT_DROP_CELLS + 1):
        sup[k:, :, :] |= solid[:-k, :, :]  # floor k cells below
    return sup


def _bfs_geodesic(occ, seed, cell: float, gravity_dir: bool = False,
                  device="cuda", max_sweeps: int = 8000, verbose: bool = True):
    """Multi-source geodesic over the free voxels of ``occ`` (nz, ny, nx),
    seeded at ``seed`` (bool, same shape). Returns
    ``(d, reach_max, sweeps)``: a torch float32 grid in map units holding
    ``+inf`` on solid and unreachable voxels, the finite maximum, and the
    sweep count.

    Edge semantics — read this before touching the offsets. Relaxation is
    ``d[A] = min(d[A], d[A + off] + w)`` and ``d`` is distance TO the goal,
    so a relaxation along ``off`` encodes the path ``A -> A+off -> ... ->
    goal``: the wavefront travels backward from the finish, but ``off`` is
    the direction the PLAYER moves. World +z is +z index, so ``off[0] > 0``
    is exactly a player CLIMB.

    ``gravity_dir`` therefore gates only the nine ``off[0] > 0`` offsets: a
    climb relaxes only where the source voxel or its destination is
    surface-adjacent (:func:`_surface_support`). Descending and lateral
    offsets stay unconstrained — the player can always fall, and air-strafe
    while falling. Weights are untouched (euclidean step length), so a
    directional field is still metres-of-track and still telescopes.

    ``device="cpu"`` never queries CUDA — the selftests must not touch a GPU
    a trainer is using."""
    import torch

    if str(device).startswith("cpu"):
        dev = torch.device("cpu")
    else:
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
    nz, ny, nx = occ.shape
    solid = torch.as_tensor(np.asarray(occ) != 0, device=dev)
    INF = float("inf")
    dpad = torch.full((nz + 2, ny + 2, nx + 2), INF, device=dev)
    d = torch.full((nz, ny, nx), INF, device=dev)

    seed_t = torch.as_tensor(np.asarray(seed, bool), device=dev) & ~solid
    n_seed = int(seed_t.sum())
    if n_seed == 0:
        raise RuntimeError("goal zone contains no free voxel — wrong box?")
    d[seed_t] = 0.0
    del seed_t            # min-relaxation can never raise the 0s — no refill

    offsets = [(oz, oy, ox)
               for oz in (-1, 0, 1) for oy in (-1, 0, 1) for ox in (-1, 0, 1)
               if (oz, oy, ox) != (0, 0, 0)]
    weights = [cell * float(np.sqrt(oz * oz + oy * oy + ox * ox))
               for oz, oy, ox in offsets]
    scratch = torch.empty_like(d)

    spad = support = blocked = None
    if gravity_dir:
        # pad with False: outside the sampled box there is no geometry to
        # climb, and the grid carries 4 cells of margin past the map bounds
        spad = torch.zeros((nz + 2, ny + 2, nx + 2), dtype=torch.bool,
                           device=dev)
        support = spad[1:-1, 1:-1, 1:-1]      # view — the source-side term
        _surface_support(solid, torch, out=support)
        blocked = torch.empty_like(solid)     # reused per upward offset
        if verbose:
            # ASCII only: this lands on a cp1251 console
            print(f"goal graph: gravity-directional (rule "
                  f"v{_GRAVITY_RULE_VERSION}, support drop "
                  f"{_SUPPORT_DROP_CELLS} cells), "
                  f"{int(support.sum()):,} / {support.numel():,} voxels "
                  f"surface-adjacent")

    def relax():
        dpad[1:-1, 1:-1, 1:-1].copy_(d)
        for (oz, oy, ox), w in zip(offsets, weights):
            src = dpad[1 + oz:1 + oz + nz, 1 + oy:1 + oy + ny,
                       1 + ox:1 + ox + nx]
            torch.add(src, w, out=scratch)
            if gravity_dir and oz > 0:
                # the player would be climbing A -> A+off here: legal only
                # along geometry, so kill the edge where NEITHER end is
                # surface-adjacent (free-fall-only shafts stop being
                # two-way and the pits below them stop reading as shortcuts)
                dst = spad[1 + oz:1 + oz + nz, 1 + oy:1 + oy + ny,
                           1 + ox:1 + ox + nx]
                torch.logical_or(support, dst, out=blocked)
                torch.logical_not(blocked, out=blocked)
                scratch.masked_fill_(blocked, INF)
            torch.minimum(d, scratch, out=d)
        d.masked_fill_(solid, INF)

    # convergence probe in float64: a float32 sum over ~1e13 has ~1e6-unit
    # ULPs — a slow correction wave improving less than that per window
    # would read as "converged" while distant stages stay overestimated
    prev = None
    it = 0
    while it < max_sweeps:
        relax()
        it += 1
        if it % 64 == 0:
            fin = torch.isfinite(d)
            cur = (int(fin.sum()),
                   float(d[fin].sum(dtype=torch.float64)))
            if cur == prev:
                break
            prev = cur
    if it >= max_sweeps:
        # directional graphs need MORE sweeps than the plain one (detours
        # replace one-way shortcuts), so this cap is reachable now. Exiting
        # on it means the wavefront never settled and far stages are
        # OVERESTIMATED - loud, because the npz that follows looks normal.
        print(f"goal field: WARNING - hit the {max_sweeps}-sweep cap without "
              f"converging; distances are overestimated, do not ship this "
              f"bake (raise max_sweeps)")
    fin = torch.isfinite(d)
    if not bool(fin.any()):
        raise RuntimeError("goal field has no reachable voxel at all")
    reach_max = float(d[fin].max())
    if verbose:
        print(f"goal field: {it} sweeps, {n_seed} seed voxels, "
              f"{int(fin.sum()):,} reachable voxels, "
              f"max geodesic {reach_max:.0f}u")
    # returning drops dpad/scratch/solid (+ the support masks) with the
    # frame, so the caching allocator has them back before the caller casts
    # `d` to int32 — the one allocation left in the bake
    return d, reach_max, it


def build_goal_field(core, zone, cell: float, cache_dir=None,
                     device="cuda", mask_kill: bool = False,
                     gravity_dir: bool = False) -> GoalField:
    """Build (or load) the geodesic distance field toward ``zone``
    (``{"mins": [...], "maxs": [...]}``, map units).

    ``mask_kill`` (arm S2): rasterize kill volumes (destful teleports,
    fatal hurt triggers — see zones.kill_zones) as walls in the goal graph
    ONLY, so the shaping gradient routes around fail nets instead of
    through them. Physics and vision are untouched; the caller must verify
    the start stays reachable (a disconnect means the voxelized route
    model is wrong and the arm must not run).

    ``gravity_dir``: make the graph GRAVITY-DIRECTIONAL — the player may
    fall and air-strafe anywhere, but may only gain height along geometry
    (:func:`_bfs_geodesic`). Without it the undirected BFS lets voxels
    "reach" the finish through one-way falls, which on surf_src_cannonball
    paints an off-route pit as the global minimum of the shaping potential.
    Combines with ``mask_kill``; each combination gets its own cache
    (``goal_`` / ``goalk_`` / ``goalg_`` / ``goalgk_``), so the plain fields
    every finished arm was trained against stay bit-identical."""
    import torch

    bsp = Path(core.bsp_path)
    box = np.round(np.asarray(zone["mins"] + zone["maxs"], np.float64), 1)
    # the non-directional signature is unchanged on purpose — a stale-cache
    # miss here costs a 10-minute GPU bake per map
    sig = (f"g{_GOAL_BUILDER_VERSION}_{'k1_' if mask_kill else ''}"
           f"{f'd{_GRAVITY_RULE_VERSION}_' if gravity_dir else ''}"
           f"{_map_sig(bsp)}_" + "_".join(f"{v:g}" for v in box))
    cache = Path(cache_dir) if cache_dir else bsp.parent
    tag = f"{'g' if gravity_dir else ''}{'k' if mask_kill else ''}"
    cache_file = cache / f"{bsp.stem}.goal{tag}_{cell:g}.npz"
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        if "sig" in z and str(z["sig"]) == sig:
            q = float(z["quant"])
            grid = z["grid"].astype(np.float32) * q
            return GoalField(grid, z["mins"], float(z["cell"]),
                             float(z["reach_max"]))

    occ, mins = goal_occupancy(core, cell, cache_dir)
    if mask_kill:
        from .zones import hull_probe, kill_zones
        occ = occ.copy()
        gz, gy, gx = occ.shape
        kz = kill_zones(bsp)
        contains = hull_probe(bsp)
        # classify voxel centers through each trigger's HULL-1 clipnodes —
        # the sim's own kill test. AABBs are broadphase only: triggers are
        # often thin curtains inside huge boxes, and box-filling was
        # measured to wall off ~10% of the on-track airspace, ~98% of it
        # not actually deadly
        masked = 0
        for k in kz:
            bmn = np.asarray(k["mins"], np.float64) - 1.0
            bmx = np.asarray(k["maxs"], np.float64) + 1.0
            lo = np.maximum(np.floor((bmn - mins) / cell - 0.5), 0).astype(int)
            hi = np.minimum(np.ceil((bmx - mins) / cell - 0.5) + 1,
                            [gx, gy, gz]).astype(int)
            if (hi <= lo).any():
                continue
            xs = mins[0] + (np.arange(lo[0], hi[0]) + 0.5) * cell
            ys = mins[1] + (np.arange(lo[1], hi[1]) + 0.5) * cell
            zs = mins[2] + (np.arange(lo[2], hi[2]) + 0.5) * cell
            gzz, gyy, gxx = np.meshgrid(zs, ys, xs, indexing="ij")
            pts = np.stack([gxx.ravel(), gyy.ravel(), gzz.ravel()], 1)
            inside = contains(int(k["model"][1:]),
                              pts - np.asarray(k["origin"], np.float64))
            if inside.any():
                win = occ[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
                m = inside.reshape(win.shape)
                masked += int((m & (win == 0)).sum())
                win[m] = 1
        print(f"goal graph: {len(kz)} kill volumes hull-masked "
              f"({masked} free voxels -> wall)")
    nz, ny, nx = occ.shape
    smin, smax = _zone_seed_box(zone, cell)
    lo = np.maximum(np.floor((smin - mins) / cell - 0.5), 0).astype(int)
    hi = np.minimum(np.ceil((smax - mins) / cell - 0.5),
                    [nx - 1, ny - 1, nz - 1]).astype(int)
    seed = np.zeros(occ.shape, bool)
    seed[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = True

    d, reach_max, _ = _bfs_geodesic(occ, seed, cell, gravity_dir=gravity_dir,
                                    device=device)
    del seed, occ

    # solid + unreachable stay at the sentinel — sampling renormalizes over
    # honest corners (v1's "bleed into solids" leaked low distances through
    # thin walls and made the shaping farmable)
    sentinel = reach_max + 2.0 * cell
    d.nan_to_num_(posinf=sentinel)

    quant = cell / 8.0
    if sentinel / quant > 65535:
        # a saturated sentinel would quantize DOWN into the valid range and
        # make sample() pay finite potential inside walls/unreachable space.
        # Directional graphs stretch geodesics (detours replace one-way
        # shortcuts), so widen the LSB rather than throw away the bake —
        # but never past one cell, or rounding could lift a reachable value
        # over sample()'s honest-corner threshold (reach_max + cell/2).
        # round-trip through f32 first: the npz stores it as f32 and load
        # divides by THAT, so quantize with the number readers will see
        quant = float(np.float32(min(sentinel / 65500.0, cell)))
        print(f"goal field: sentinel {sentinel:.0f}u overflows uint16 at "
              f"{cell / 8.0:g}u/LSB, quantization widened to {quant:g}u")
    if sentinel / quant > 65535:
        raise RuntimeError(
            f"goal field sentinel {sentinel:.0f}u exceeds the uint16 range "
            f"at quant {quant:g}u — widen quant or store an explicit "
            f"unreachable mask")
    d.div_(quant).round_().clamp_(0, 65535)
    grid_q = d.to(torch.int32).cpu().numpy().astype(np.uint16)
    del d
    np.savez_compressed(cache_file, grid=grid_q, quant=np.float32(quant),
                        mins=mins, cell=np.float32(cell),
                        reach_max=np.float32(reach_max), sig=np.str_(sig))
    grid = grid_q.astype(np.float32) * quant
    return GoalField(grid, mins, cell, reach_max)
