"""explore_phase1.py - Go-Explore Phase 1 (reward-free discovery) on a surf map.

Why this exists
---------------
The shaped race reward on these maps is DECEPTIVE: a mid-route basin traps
PPO, and the route that actually wins is reward-adversarial for ~50 s. No
amount of shaping tuning fixes a reward whose gradient points the wrong way
for a minute of play. Go-Explore's answer (Ecoffet et al., arXiv 1901.10995;
Nature, arXiv 2004.12919) is to stop optimizing a reward during discovery and
run a pure archive search instead:

    1. keep an archive of cells (discretized states) and, per cell, the best
       state ever seen there;
    2. SELECT a cell with probability proportional to W = 1/sqrt(C_seen + 1);
    3. RETURN to it - here by native state restore, not action replay
       (the paper measures restore as ~45x cheaper than replay);
    4. EXPLORE from it with random actions for k decisions, repeating the
       previous action with probability 0.95;
    5. archive everything new, repeat until the goal box is crossed.

Nothing in this file reads the environment reward. The only objective signal
is ``core.goal_hits`` (did anyone cross the finish curtain), and a distance
field used ONLY for progress printing.

What it produces
----------------
On the first goal crossing (and every crossing after, if ``--goals`` allows):

    <out>/win_<k>.traj.jsonl   the winning exploration burst, per physics
                               tick, in the repo's docs/04 JSONL format
                               (readable by tools/tas_search.py, the viewer)
    <out>/win_<k>.spine.npy    TIME-ORDERED STATE_DTYPE array from a map
                               spawn to the goal - the demo spine for the
                               Salimans-Chen robustifier
                               (train_fast.py --demo-file)
    <out>/win_<k>.chain.json   the archive cell chain behind that spine
    <out>/archive.npz          states + counts + provenance

The spine is STITCHED, not replayed: consecutive rows can come from different
exploration bursts (each row is the best state ever archived for its cell,
and every row is reachable from its predecessor's cell in the recorded tick
gap). That is exactly what a backward curriculum needs - it only ever RESETS
into those states, it never replays between them.

Faithfulness notes / deliberate deviations
------------------------------------------
* Cell selection weight is the Nature formula W = 1/sqrt(C_seen + 1), and
  C_seen counts EXPLORATION RUNS that visited the cell (once per run, however
  many ticks were spent there) - not ticks, not times-chosen. The 2019
  CntScore/NeighScore machinery is deliberately not implemented: its tuned
  optimum was the same 1/sqrt shape, and the domain-knowledge terms need a
  notion of "neighbour cell" this map does not give us for free.
* Exploration burst: k decisions (default 100) with 95% action repeat, a
  decision being ``--act-every`` physics ticks (3, matching the trainer), and
  a fresh action drawn uniformly over the full MultiDiscrete action set.
  Aborted at episode end, exactly as in the paper.
* DEVIATION (``--restart-dead 0`` turns it off): when an env's episode ends
  mid-chunk the paper would leave that worker idle until the batch finishes.
  On a surf map a random-action burst usually dies in well under a second, so
  idling wastes most of the batch; instead the env is immediately re-seeded
  with a freshly selected cell and starts a new run. This is a worker pool,
  not an algorithm change - runs stay independent and C_seen bookkeeping is
  per run either way.
* Restored states get ``tick`` and ``stuck_ticks`` zeroed (``--keep-tick``
  keeps them). Otherwise a deep cell inherits its episode clock and truncates
  immediately. Cumulative distance from the map start is tracked Python-side
  as ``depth`` (physics ticks), which is also the "better trajectory"
  criterion for replacing an archived cell.
* Pitch is fixed to the neutral bin by default: src/env.c passes 0 pitch into
  pm_tick, so the pitch action is provably inert for physics and randomizing
  it only adds noise to nothing. ``--random-pitch`` restores it.

Usage
-----
    python tools/explore_phase1.py --map maps/surf_src_cannonball.bsp \\
        --envs 512 --out runs/explore_cb

    python tools/explore_phase1.py --selftest        # CPU, no DLL, no map

Ctrl-C saves the archive before exiting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import ACTION_NVEC, STATE_DTYPE                # noqa: E402

# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------


class CellHash:
    """Position (plus optional gaze / speed) -> int64 cell key.

    Mirrors the trainer's novelty key (surfgym/rewards.py ``_cells``) so an
    archive built here is directly comparable with the intrinsic-count cells
    PPO already pays for: 3D grid over the map AABB at ``cell`` units, index
    ``ix + nx*(iy + ny*iz)``, optionally multiplied by a yaw sector and a
    horizontal-speed bucket.

    Position-only is the default. Gaze and speed split a place into several
    cells, which is more faithful to "state" but multiplies the archive: a
    surf map is speed-gated, so ``--cell-speed`` is the option worth trying
    second, after a position-only run maps the route topology.
    """

    def __init__(self, mins, maxs, cell: float = 256.0, view: int = 0,
                 speed: int = 0, speed_max: float = 3500.0) -> None:
        self.mins = np.asarray(mins, np.float64)
        self.cell = float(cell)
        self.view = int(view)
        self.speed = int(speed)
        self.dims = tuple(int(np.ceil((float(maxs[i]) - self.mins[i]) / self.cell)) + 1
                          for i in range(3))
        self.speed_bin = float(speed_max) / max(1, self.speed)

    @property
    def capacity(self) -> int:
        """Number of distinct keys this hash can produce."""
        n = self.dims[0] * self.dims[1] * self.dims[2]
        return n * max(1, self.view) * max(1, self.speed)

    def keys(self, states: np.ndarray) -> np.ndarray:
        """STATE_DTYPE array of any shape -> int64 keys of the same shape."""
        p = states["origin"].astype(np.float64)
        ix = np.clip(((p[..., 0] - self.mins[0]) // self.cell).astype(np.int64),
                     0, self.dims[0] - 1)
        iy = np.clip(((p[..., 1] - self.mins[1]) // self.cell).astype(np.int64),
                     0, self.dims[1] - 1)
        iz = np.clip(((p[..., 2] - self.mins[2]) // self.cell).astype(np.int64),
                     0, self.dims[2] - 1)
        key = ix + self.dims[0] * (iy + self.dims[1] * iz)
        if self.view > 0:
            yb = np.floor((states["yaw"].astype(np.float64) % 360.0)
                          / 360.0 * self.view).astype(np.int64)
            np.clip(yb, 0, self.view - 1, out=yb)
            key = key * self.view + yb
        if self.speed > 0:
            v = states["velocity"]
            sb = (np.hypot(v[..., 0], v[..., 1]) // self.speed_bin).astype(np.int64)
            np.clip(sb, 0, self.speed - 1, out=sb)
            key = key * self.speed + sb
        return key


def group_entries(keys: np.ndarray, run_id: np.ndarray):
    """Reduce a (T, N) rollout to one record per (run, cell) first visit.

    ``keys``   (T, N) int64 cell key of the state at the START of tick t.
    ``run_id`` (T, N) int32 exploration-run id, -1 for rows to ignore (an env
               idling after its episode ended).

    Returns ``(g_key, g_run, g_tick, g_env)``, sorted by run and then by tick,
    one entry per (run, cell) pair holding the EARLIEST tick that run entered
    that cell. That is simultaneously the C_seen unit ("+1 per exploration run
    in which the cell is visited, even if visited many times") and the state
    to archive ("first / lowest-depth state that reached the cell").

    Only cell TRANSITIONS are candidates, which is what keeps this cheap: at
    256 u a run crosses a new cell every few ticks at best, so the sort sees a
    small fraction of the T*N rows.
    """
    live = run_id >= 0
    entry = np.empty(keys.shape, bool)
    entry[0] = live[0]
    entry[1:] = live[1:] & ((keys[1:] != keys[:-1]) | (run_id[1:] != run_id[:-1]))
    flat = np.flatnonzero(entry.ravel())
    if flat.size == 0:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    n_env = keys.shape[1]
    k = keys.ravel()[flat]
    r = run_id.ravel()[flat].astype(np.int64)
    t = flat // n_env
    e = flat - t * n_env
    order = np.lexsort((t, k, r))                 # group by (run, cell)
    k, r, t, e = k[order], r[order], t[order], e[order]
    first = np.empty(k.size, bool)
    first[0] = True
    first[1:] = (r[1:] != r[:-1]) | (k[1:] != k[:-1])
    k, r, t, e = k[first], r[first], t[first], e[first]
    order = np.lexsort((t, r))                    # replay order within a run
    return k[order], r[order], t[order], e[order]


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


class Archive:
    """Cell -> best known state, with counts and parent provenance.

    Columns (all parallel, length ``size``):

    ``key``     cell key
    ``state``   STATE_DTYPE row: the lowest-``depth`` state seen in that cell
    ``seen``    C_seen - exploration runs that visited the cell (Nature's W)
    ``chosen``  times selected as a return target (bookkeeping / diagnostics;
                reset when the cell's state is improved, per 1901.10995 2.1.1)
    ``parent``  archive index of the cell entered just before this one, on the
                run that produced ``state``; -1 for map-start roots
    ``depth``   physics ticks from a map spawn along that provenance chain
    ``dist``    distance to the goal box (progress printing only)
    """

    def __init__(self, hasher: CellHash, capacity: int = 1 << 16,
                 max_cells: int = 4_000_000) -> None:
        self.h = hasher
        self.max_cells = int(max_cells)
        self.size = 0
        self._grow(int(capacity))
        self.index: dict[int, int] = {}
        self._cum: np.ndarray | None = None
        self.n_full_warned = False

    def _grow(self, cap: int) -> None:
        old = getattr(self, "key", None)
        n = self.size
        self.key = np.zeros(cap, np.int64)
        self.state = np.zeros(cap, STATE_DTYPE)
        self.seen = np.zeros(cap, np.int64)
        self.chosen = np.zeros(cap, np.int64)
        self.parent = np.full(cap, -1, np.int64)
        self.depth = np.zeros(cap, np.int64)
        self.dist = np.full(cap, np.inf, np.float64)
        if old is not None and n:
            self.key[:n] = old[:n]
            self.state[:n] = self._old_state[:n]
            self.seen[:n] = self._old_seen[:n]
            self.chosen[:n] = self._old_chosen[:n]
            self.parent[:n] = self._old_parent[:n]
            self.depth[:n] = self._old_depth[:n]
            self.dist[:n] = self._old_dist[:n]
        self.capacity = cap

    def _reserve(self, extra: int) -> None:
        if self.size + extra <= self.capacity:
            return
        cap = self.capacity
        while cap < self.size + extra:
            cap *= 2
        (self._old_state, self._old_seen, self._old_chosen, self._old_parent,
         self._old_depth, self._old_dist) = (self.state, self.seen, self.chosen,
                                             self.parent, self.depth, self.dist)
        self._grow(cap)
        del (self._old_state, self._old_seen, self._old_chosen,
             self._old_parent, self._old_depth, self._old_dist)

    # -- writing ------------------------------------------------------------

    def add(self, key: int, state, depth: int = 0, parent: int = -1) -> int:
        """Insert a cell (no-op returning the existing index if present)."""
        key = int(key)
        got = self.index.get(key)
        if got is not None:
            return got
        self._reserve(1)
        i = self.size
        self.key[i] = key
        self.state[i] = state
        self.depth[i] = int(depth)
        self.parent[i] = int(parent)
        self.index[key] = i
        self.size = i + 1
        return i

    def ingest(self, g_key, g_run, g_tick, g_state, run_t0, run_d0,
               run_origin) -> np.ndarray:
        """Fold one chunk's (run, cell) first-visit records into the archive.

        ``run_t0`` / ``run_d0`` / ``run_origin`` are per-run arrays: the tick
        the run's first recorded row sits at, the cumulative depth of the cell
        it was restored from, and that cell's archive index.

        Returns the archive indices of cells added by this call (for the
        caller's incremental distance bookkeeping).
        """
        if len(g_key) == 0:
            return np.zeros(0, np.int64)
        depth = run_d0[g_run] + (g_tick - run_t0[g_run])
        self._reserve(len(g_key))
        keys = g_key.tolist()
        runs = g_run.tolist()
        ticks = depth.tolist()
        origins = run_origin.tolist()
        index = self.index
        added: list[int] = []
        prev_run = -1
        prev_idx = -1
        for j in range(len(keys)):
            r = runs[j]
            if r != prev_run:
                prev_run = r
                prev_idx = origins[r]
            k = keys[j]
            d = ticks[j]
            i = index.get(k)
            if i is None:
                if self.size >= self.max_cells:
                    if not self.n_full_warned:
                        print(f"archive hit --max-cells {self.max_cells:,}; "
                              "new cells are dropped from here on "
                              "(counts on known cells keep updating)")
                        self.n_full_warned = True
                    continue
                i = self.size
                self.key[i] = k
                self.state[i] = g_state[j]
                self.seen[i] = 1
                self.depth[i] = d
                self.parent[i] = prev_idx
                index[k] = i
                self.size = i + 1
                added.append(i)
            else:
                self.seen[i] += 1
                if d < self.depth[i]:
                    # "better trajectory" is shorter here - the run reached the
                    # same cell in fewer ticks from the map start. 1901.10995
                    # 2.1.1 resets the choice counters on an improvement (a new
                    # way in may be a better stepping stone) but never C_seen.
                    self.state[i] = g_state[j]
                    self.depth[i] = d
                    self.parent[i] = prev_idx
                    self.chosen[i] = 0
            prev_idx = i
        self._cum = None
        return np.asarray(added, np.int64)

    # -- selection ----------------------------------------------------------

    def build_sampler(self) -> None:
        """Cache the cumulative selection weight. W = 1/sqrt(C_seen + 1)
        (Nature Extended Data Table 1a, robustification treatment)."""
        w = 1.0 / np.sqrt(self.seen[:self.size].astype(np.float64) + 1.0)
        self._cum = np.cumsum(w)

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``n`` archive indices with probability proportional to W."""
        if self._cum is None:
            self.build_sampler()
        total = float(self._cum[-1])
        idx = np.searchsorted(self._cum, rng.random(n) * total, side="right")
        return np.clip(idx, 0, self.size - 1)

    # -- provenance ---------------------------------------------------------

    def chain(self, i: int) -> list[int]:
        """Root-first archive indices from a map spawn down to cell ``i``.

        ``depth`` is strictly increasing from parent to child at write time
        and parents only ever get shallower, so the chain cannot cycle; the
        guard is belt and braces against a corrupted snapshot.
        """
        out: list[int] = []
        seen: set[int] = set()
        while i >= 0:
            if i in seen or len(out) > self.size:
                print(f"WARNING: parent chain cycle at index {i}; truncated")
                break
            seen.add(i)
            out.append(int(i))
            i = int(self.parent[i])
        out.reverse()
        return out

    # -- io -----------------------------------------------------------------

    def save(self, path, meta: dict | None = None) -> None:
        n = self.size
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 key=self.key[:n], state=self.state[:n], seen=self.seen[:n],
                 chosen=self.chosen[:n], parent=self.parent[:n],
                 depth=self.depth[:n], dist=self.dist[:n])
        if meta is not None:
            Path(str(path).replace(".npz", ".meta.json")).write_text(
                json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# goal distance (progress printing only - never a reward)
# ---------------------------------------------------------------------------


class EuclidDist:
    """Straight-line distance to the finish AABB. Zero precompute."""

    def __init__(self, zone) -> None:
        self.mins = np.asarray(zone["mins"], np.float64)
        self.maxs = np.asarray(zone["maxs"], np.float64)
        self.name = "euclid"

    def sample(self, pos: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pos, np.float64))
        q = np.clip(p, self.mins, self.maxs)
        return np.linalg.norm(p - q, axis=1)


class GridDist:
    """Cached geodesic field from surfgym/goalfield.py, loaded straight from
    the npz so this tool never imports torch. Same honest-corner trilinear
    sampling as GoalField.sample."""

    def __init__(self, npz_path: Path) -> None:
        z = np.load(npz_path, allow_pickle=False)
        self.grid = z["grid"].astype(np.float32) * float(z["quant"])
        self.mins = np.asarray(z["mins"], np.float64)
        self.cell = float(z["cell"])
        self.reach_max = float(z["reach_max"])
        self.sentinel = self.reach_max + 2.0 * self.cell
        self._valid_max = self.reach_max + 0.5 * self.cell
        self.dims = np.array(self.grid.shape[::-1])
        self.name = f"geodesic {npz_path.name}"

    def sample(self, pos: np.ndarray) -> np.ndarray:
        g = (np.atleast_2d(np.asarray(pos, np.float64)) - self.mins) / self.cell - 0.5
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
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6),
                        self.sentinel).astype(np.float64)


def make_dist(bsp: Path, zone):
    """Cached geodesic field if one is sitting next to the map, else euclid.
    Building a field is a multi-minute GPU bake (goalfield.py) and this tool
    never does that - it only reads a cache that a trainer run left behind."""
    for p in sorted(bsp.parent.glob(f"{bsp.stem}.goal*_*.npz")):
        try:
            return GridDist(p)
        except Exception as exc:                       # noqa: BLE001
            print(f"cached goal field {p.name} unusable ({exc}); "
                  "falling back to euclid")
    return EuclidDist(zone)


# ---------------------------------------------------------------------------
# trajectory output
# ---------------------------------------------------------------------------

SURF_IN_JUMP, SURF_IN_DUCK = 2, 4


def write_traj_jsonl(path, states, actions, rewards, map_name: str,
                     tick_ms: int, phys: dict, end: str) -> None:
    """docs/04 JSONL: header, one row per physics tick, trailer.

    Row layout is record.py's exactly, so tools/tas_search.py, the viewer and
    tools/render_pov.py read these files without a special case:
    ``[t, x,y,z, vx,vy,vz, yaw, buttons, onground, progress, reward, pitch,
    fwd, side]``. ``progress`` is 0 here: the tool sets no waypoint spline
    (there is no reward to shape), so the core never fills that field.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rnd = lambda v: round(float(v), 2)                       # noqa: E731
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"map": map_name, "tick_ms": int(tick_ms),
                            "phys": phys, "episode": 0},
                           separators=(",", ":")) + "\n")
        for t in range(len(states)):
            s = states[t]
            a = actions[t]
            buttons = (SURF_IN_JUMP if a[4] else 0) | (SURF_IN_DUCK if a[5] else 0)
            o, v = s["origin"], s["velocity"]
            f.write(json.dumps(
                [t, rnd(o[0]), rnd(o[1]), rnd(o[2]),
                 rnd(v[0]), rnd(v[1]), rnd(v[2]), rnd(s["yaw"]),
                 int(buttons), int(int(s["onground"]) >= 0),
                 rnd(s["progress"]), round(float(rewards[t]), 5),
                 rnd(s["pitch"]), int(a[2]), int(a[3])],
                separators=(",", ":")) + "\n")
        f.write(json.dumps({"end": end, "ticks": int(len(states)),
                            "best_progress": 0.0},
                           separators=(",", ":")) + "\n")


def dedup_spine(states: np.ndarray) -> np.ndarray:
    """Drop rows whose origin repeats at 0.1 u resolution.

    surfgym/respawn.py's DemoCurriculum re-identifies realized spawns by
    matching ``round(origin, 1)`` against a dict built from the spine, so two
    rows with the same rounded origin would collide and one of them would
    never be creditable. Keeps the earliest occurrence.
    """
    keep = []
    seen: set[tuple] = set()
    for i, r in enumerate(states):
        k = tuple(np.round(np.asarray(r["origin"], np.float64), 1))
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    return states[np.asarray(keep, np.int64)] if keep else states[:0]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def explore(args) -> int:
    from surfgym.core import SurfCore, default_config, phys_to_dict
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones

    bsp = Path(args.map).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = int(args.envs)
    act_every = int(args.act_every)
    ticks = int(args.decisions) * act_every
    min_run_ticks = max(1, int(args.min_run) * act_every)

    cfg = default_config(num_envs=n, spawn_mode=2,
                         max_episode_ticks=int(args.ep_ticks),
                         water_fail=1, yaw_jitter_deg=0.0,
                         lidar_w=0, lidar_h=0)     # eyeless: no obs needed
    core = SurfCore(str(bsp), cfg)
    if not args.keep_teleports:
        core.set_teleport_fail(True)

    zones = load_zones(str(bsp))
    if not zones.get("end"):
        raise SystemExit(
            f"no end zone for {bsp.stem}: auto-extraction found none - hand "
            f"label maps/{bsp.stem}.zones.json (see surfgym/zones.py). "
            "Without a finish box there is nothing to terminate on.")
    goal_box = zones["end"]
    core.set_goal_box(goal_box["mins"], goal_box["maxs"])

    dist = make_dist(bsp, goal_box)
    roots = map_spawn_pool(core)
    if args.face_goal:
        h = 64.0
        o = roots["origin"].astype(np.float64)
        gx = dist.sample(o + [h, 0, 0]) - dist.sample(o - [h, 0, 0])
        gy = dist.sample(o + [0, h, 0]) - dist.sample(o - [0, h, 0])
        roots["yaw"] = np.degrees(np.arctan2(-gy, -gx)) % 360.0
    core.set_spawn_pool(roots)                     # post-death autoreset fallback

    mins, maxs = core.map_bounds()
    hasher = CellHash(mins, maxs, cell=args.cell, view=args.cell_view,
                      speed=args.cell_speed, speed_max=args.cell_speed_max)
    arch = Archive(hasher, max_cells=args.max_cells)
    rk = hasher.keys(roots)
    for j in range(len(roots)):
        i = arch.add(int(rk[j]), roots[j], depth=0, parent=-1)
        arch.dist[i] = dist.sample(roots[j]["origin"][None, :])[0]

    print(f"map {bsp.stem} | envs {n} | cell {args.cell:g}u"
          f"{f' x view{args.cell_view}' if args.cell_view else ''}"
          f"{f' x speed{args.cell_speed}' if args.cell_speed else ''}"
          f" | grid {hasher.dims} ({hasher.capacity:,} keys)")
    print(f"burst {args.decisions} decisions x {act_every} ticks "
          f"({ticks} ticks = {ticks * cfg.phys.msec / 1000:.1f}s) | "
          f"repeat p {args.repeat_p:g} | restart-dead {int(args.restart_dead)} | "
          f"goal box {goal_box['mins']} .. {goal_box['maxs']}")
    print(f"progress metric: {dist.name} | roots {arch.size} | "
          f"start distance {arch.dist[:arch.size].min():.0f}u | out {out}")

    # rollout ring: full states are cheap to keep (N*ticks*108 bytes) and let
    # the whole chunk be folded into the archive with numpy instead of a
    # per-tick python loop over N envs
    hist = np.zeros((ticks, n), STATE_DTYPE)
    acts_h = np.zeros((ticks, n, 6), np.int8)
    rew_h = np.zeros((ticks, n), np.float32)
    rid_h = np.full((ticks, n), -1, np.int32)
    act = np.zeros((n, 6), np.int32)
    cur_run = np.full(n, -1, np.int32)
    sv = core.states_view
    pitch_bin = 3                                  # PITCH_BINS[3] == 0 deg

    def fresh_actions(idx: np.ndarray) -> None:
        m = idx.size
        if m == 0:
            return
        act[idx, 0] = rng.integers(0, ACTION_NVEC[0], m)
        act[idx, 1] = (rng.integers(0, ACTION_NVEC[1], m) if args.random_pitch
                       else pitch_bin)
        act[idx, 2] = rng.integers(0, ACTION_NVEC[2], m)
        act[idx, 3] = rng.integers(0, ACTION_NVEC[3], m)
        act[idx, 4] = rng.integers(0, ACTION_NVEC[4], m)
        act[idx, 5] = rng.integers(0, ACTION_NVEC[5], m) if args.duck else 0

    # per-chunk run bookkeeping (parallel lists indexed by run id)
    r_t0: list[int] = []
    r_d0: list[int] = []
    r_org: list[int] = []

    def start_runs(idx: np.ndarray, t0: int) -> None:
        """Return step: select cells by W, restore them into envs ``idx``."""
        if idx.size == 0:
            return
        picks = arch.draw(idx.size, rng)
        for j in range(idx.size):
            e = int(idx[j])
            a = int(picks[j])
            st = arch.state[a:a + 1].copy()      # copy: set_state must not
            if not args.keep_tick:               # edit the archived row
                # a restored deep state carries its episode clock; leave it and
                # the env truncates on arrival. Depth is tracked python-side.
                st["tick"] = 0
                st["stuck_ticks"] = 0
            core.set_state(e, st)
            cur_run[e] = len(r_t0)
            r_t0.append(t0)
            r_d0.append(int(arch.depth[a]))
            r_org.append(a)
            arch.chosen[a] += 1
        fresh_actions(idx)

    n_goal = 0
    n_runs = 0
    n_deaths = 0
    total_ticks = 0
    t_start = time.perf_counter()
    print_every = max(1, int(args.print_every))
    map_name = bsp.stem
    phys = phys_to_dict(cfg.phys)
    all_idx = np.arange(n)
    stop = False
    it = 0
    try:
        while not stop and (args.max_iters <= 0 or it < args.max_iters):
            # a full reset clears each env's consumed push-once triggers and
            # stuck bookkeeping (surf_set_state does not); then every env is
            # placed on its own selected cell.
            core.reset(args.seed + it)
            r_t0.clear()
            r_d0.clear()
            r_org.clear()
            cur_run[:] = -1
            rid_h[:] = -1
            arch.build_sampler()
            start_runs(all_idx, 0)

            for t in range(ticks):
                hist[t] = sv
                rid_h[t] = cur_run
                if t % act_every == 0:
                    # Go-Explore burst: hold the previous action with prob
                    # repeat_p, else redraw uniformly over the whole action set
                    live = np.flatnonzero((cur_run >= 0)
                                          & (rng.random(n) >= args.repeat_p))
                    fresh_actions(live)
                acts_h[t] = act
                _, rew, done, trunc, _ = core.step(act)
                rew_h[t] = rew
                ended = (done | trunc) != 0
                hit = np.flatnonzero(np.asarray(core.goal_hits, bool)
                                     & (cur_run >= 0))
                for e in hit.tolist():
                    n_goal += 1
                    run = int(cur_run[e])
                    print(f"GOAL #{n_goal} at iter {it} tick {t} env {e} "
                          f"(archive {arch.size:,} cells, burst from cell "
                          f"depth {arch.depth[r_org[run]]}t)")
                    dump_win(out, n_goal, arch, hist, acts_h, rew_h, e,
                             r_t0[run], t, r_org[run], map_name,
                             int(cfg.phys.msec), phys, act_every, goal_box)
                    if args.goals > 0 and n_goal >= args.goals:
                        stop = True
                close = np.flatnonzero(ended & (cur_run >= 0))
                if close.size:
                    n_deaths += int(close.size)
                    cur_run[close] = -1
                    if (args.restart_dead and not stop
                            and ticks - (t + 1) >= min_run_ticks):
                        start_runs(close, t + 1)
                total_ticks += n
                if stop:
                    break

            n_runs += len(r_t0)
            fold_chunk(arch, hasher, hist, rid_h, r_t0, r_d0, r_org, dist)
            it += 1

            if it % print_every == 0 or stop:
                el = time.perf_counter() - t_start
                best = int(np.argmin(arch.dist[:arch.size]))
                print(f"it {it:6d} | archive {arch.size:9,} | runs {n_runs:9,} "
                      f"| deaths {n_deaths:9,} | best {arch.dist[best]:9.0f}u "
                      f"(depth {arch.depth[best]:6d}t) | "
                      f"{total_ticks / max(el, 1e-9) / 1e6:5.2f}M ticks/s | "
                      f"{it / max(el, 1e-9):5.2f} it/s | {el / 60:.1f} min")
            if args.snapshot_every > 0 and it % args.snapshot_every == 0:
                arch.save(out / "archive.npz",
                          meta_dict(args, bsp, hasher, arch, it, n_goal))
                print(f"snapshot -> {out / 'archive.npz'} "
                      f"({arch.size:,} cells)")
    except KeyboardInterrupt:
        print("\ninterrupted - saving archive")
    finally:
        arch.save(out / "archive.npz",
                  meta_dict(args, bsp, hasher, arch, it, n_goal))
        el = time.perf_counter() - t_start
        best = int(np.argmin(arch.dist[:arch.size])) if arch.size else 0
        print(f"done: {it} iters, {n_runs:,} runs, {arch.size:,} cells, "
              f"{n_goal} goal hits, best distance "
              f"{arch.dist[best] if arch.size else float('nan'):.0f}u, "
              f"{el / 60:.1f} min -> {out}")
        core.close()
    return 0 if n_goal else 1


def fold_chunk(arch: Archive, hasher: CellHash, hist, rid_h, r_t0, r_d0,
               r_org, dist) -> None:
    """Archive everything one chunk of rollout discovered."""
    if not r_t0:
        return
    keys = hasher.keys(hist)
    g_key, g_run, g_tick, g_env = group_entries(keys, rid_h)
    if g_key.size == 0:
        return
    g_state = hist[g_tick, g_env]
    added = arch.ingest(g_key, g_run, g_tick, g_state,
                        np.asarray(r_t0, np.int64), np.asarray(r_d0, np.int64),
                        np.asarray(r_org, np.int64))
    if added.size:
        arch.dist[added] = dist.sample(arch.state[added]["origin"])


def dump_win(out: Path, k: int, arch: Archive, hist, acts_h, rew_h, env: int,
             t0: int, t1: int, origin_idx: int, map_name: str, tick_ms: int,
             phys: dict, stride: int = 3, goal_box=None) -> None:
    """Save the winning burst plus the archive chain that led into it."""
    rows = hist[t0:t1 + 1, env]
    write_traj_jsonl(out / f"win_{k}.traj.jsonl", rows, acts_h[t0:t1 + 1, env],
                     rew_h[t0:t1 + 1, env], map_name, tick_ms, phys, "done")

    chain = arch.chain(origin_idx)
    # spine = archived states root -> the cell this burst started from, then
    # the burst's own states at decision granularity (a reset point mid
    # decision is not something the trainer can represent). Rows before the
    # burst come from many different bursts; a backward curriculum only
    # RESETS into them, never replays between them, so the stitch is sound.
    tail = rows[::max(1, int(stride))]
    if len(rows) and (len(rows) - 1) % max(1, int(stride)):
        tail = np.concatenate([tail, rows[-1:]])
    spine = dedup_spine(np.concatenate(
        [arch.state[np.asarray(chain, np.int64)], tail]))
    np.save(out / f"win_{k}.spine.npy", spine)
    depth = int(arch.depth[origin_idx]) + int(t1 - t0 + 1)
    (out / f"win_{k}.chain.json").write_text(json.dumps({
        "goal_index": k,
        "map": map_name,
        "goal_box": goal_box,
        "burst_ticks": int(t1 - t0 + 1),
        "chain_len": len(chain),
        "spine_len": int(len(spine)),
        "chain_depth_ticks": depth,
        "chain_depth_seconds": round(depth * tick_ms / 1000.0, 2),
        "note": "chain_depth is the summed tick cost of the archived hops, a "
                "LOWER BOUND on a real single-episode route time: consecutive "
                "states come from different exploration bursts.",
        "chain": [{"idx": int(i), "key": int(arch.key[i]),
                   "depth": int(arch.depth[i]),
                   "dist": float(arch.dist[i]),
                   "origin": [round(float(v), 2)
                              for v in arch.state[i]["origin"]]}
                  for i in chain],
    }, indent=2), encoding="utf-8")
    print(f"  -> win_{k}.traj.jsonl ({t1 - t0 + 1} ticks), "
          f"win_{k}.spine.npy ({len(spine)} states, chain {len(chain)}), "
          f"win_{k}.chain.json | chain depth {depth}t "
          f"({depth * tick_ms / 1000.0:.1f}s)")


def meta_dict(args, bsp, hasher, arch, it, n_goal) -> dict:
    return {
        "map": bsp.stem,
        "tool": "explore_phase1",
        "cell": args.cell,
        "cell_view": args.cell_view,
        "cell_speed": args.cell_speed,
        "cell_speed_max": args.cell_speed_max,
        "grid_dims": list(hasher.dims),
        "grid_mins": [float(v) for v in hasher.mins],
        "envs": args.envs,
        "decisions": args.decisions,
        "act_every": args.act_every,
        "repeat_p": args.repeat_p,
        "restart_dead": bool(args.restart_dead),
        "keep_tick": bool(args.keep_tick),
        "random_pitch": bool(args.random_pitch),
        "seed": args.seed,
        "iters": int(it),
        "cells": int(arch.size),
        "goals": int(n_goal),
        "weight": "1/sqrt(C_seen+1)",
    }


# ---------------------------------------------------------------------------
# selftest (CPU, no DLL, no map)
# ---------------------------------------------------------------------------


def _fake_states(origins) -> np.ndarray:
    o = np.asarray(origins, np.float32)
    s = np.zeros(o.shape[:-1], STATE_DTYPE)
    s["origin"] = o
    s["onground"] = -1
    return s


class FakeCore:
    """Stand-in for SurfCore with toy physics - CPU only, no DLL, no map.

    Mimics the parts of the C contract the loop depends on, especially the
    awkward ones: ``states_view`` is a live zero-copy read-only view that
    mutates in place, ``goal_hits`` is 1 only on the crossing tick, and a
    done/trunc env is AUTORESET from the spawn pool inside the same ``step``
    (so the row a caller reads after a done step is already the new episode).
    Physics is a corridor: +x per forward bin, +y per strafe bin, a small
    per-tick death chance, and a finish box out at +x.
    """

    def __init__(self, bsp_path, config, dll_path=None) -> None:
        self.bsp_path = str(bsp_path)
        self._cfg = config
        self.n = int(config.num_envs)
        self._st = np.zeros(self.n, STATE_DTYPE)
        self._st["onground"] = -1
        self._gh = np.zeros(self.n, np.uint8)
        self._obs = np.zeros((self.n, 1), np.float32)
        self._rew = np.zeros(self.n, np.float32)
        self._done = np.zeros(self.n, np.uint8)
        self._trunc = np.zeros(self.n, np.uint8)
        self._pool = None
        self._goal = None
        self.rng = np.random.default_rng(4242)
        self.n_set_state = 0
        self.n_reset = 0

    @property
    def num_envs(self) -> int:
        return self.n

    @property
    def config(self):
        return self._cfg

    @property
    def states_view(self) -> np.ndarray:
        v = self._st.view()
        v.flags.writeable = False
        return v

    @property
    def goal_hits(self) -> np.ndarray:
        v = self._gh.view()
        v.flags.writeable = False
        return v

    def map_bounds(self):
        return (np.array([-2000.0] * 3, np.float32),
                np.array([2000.0] * 3, np.float32))

    def set_goal_box(self, mins, maxs) -> None:
        self._goal = (np.asarray(mins, np.float64), np.asarray(maxs, np.float64))

    def set_teleport_fail(self, enable: bool = True) -> None:
        pass

    def set_spawn_pool(self, states) -> None:
        self._pool = np.ascontiguousarray(states, STATE_DTYPE).copy()

    def set_state(self, i: int, state) -> None:
        self._st[i] = np.asarray(state).reshape(-1)[0]
        self.n_set_state += 1

    def _respawn(self, idx) -> None:
        k = self.rng.integers(0, len(self._pool), len(idx))
        self._st[idx] = self._pool[k]
        self._st["tick"][idx] = 0
        self._st["stuck_ticks"][idx] = 0

    def reset(self, seed: int = 0) -> np.ndarray:
        self.n_reset += 1
        self._respawn(np.arange(self.n))
        self._gh[:] = 0
        return self._obs

    def step(self, actions):
        st = self._st
        st["origin"][:, 0] += (actions[:, 2].astype(np.float64) - 1.0) * 25.0
        st["origin"][:, 1] += (actions[:, 3].astype(np.float64) - 1.0) * 25.0
        st["velocity"][:, 0] = (actions[:, 2].astype(np.float32) - 1.0) * 250.0
        st["yaw"] = (st["yaw"] + (actions[:, 0].astype(np.float32) - 7.0)) % 360.0
        st["tick"] += 1
        p = st["origin"].astype(np.float64)
        gmin, gmax = self._goal
        hit = np.all((p >= gmin) & (p <= gmax), axis=1)
        die = ((self.rng.random(self.n) < 0.01) | (np.abs(p[:, 1]) > 1200.0)
               | (p[:, 0] < -600.0))
        trunc = st["tick"] >= self._cfg.max_episode_ticks
        self._gh[:] = hit.astype(np.uint8)
        done = hit | (die & ~hit)
        self._done[:] = done.astype(np.uint8)
        self._trunc[:] = (trunc & ~done).astype(np.uint8)
        idx = np.flatnonzero(done | trunc)
        if idx.size:
            self._respawn(idx)
        return self._obs, self._rew, self._done, self._trunc, self._obs

    def close(self) -> None:
        pass


def _loop_selftest(check) -> None:
    """Drive the real explore() loop against FakeCore in a temp dir."""
    import tempfile
    import surfgym.core as sgcore
    import surfgym.rewards as sgrew
    import surfgym.zones as sgzones

    # far enough out that no single 40-tick burst can reach it from a spawn:
    # the goal is only findable by CHAINING archived cells, which is the whole
    # point of the algorithm and the only way the provenance chain gets tested
    end = {"mins": [2600.0, -1200.0, -50.0], "maxs": [2900.0, 1200.0, 50.0]}

    def fake_pool(core, yaw=None):
        pool = np.zeros(2, STATE_DTYPE)
        pool["onground"] = -1
        pool["origin"] = [[0.0, 0.0, 0.0], [0.0, 50.0, 0.0]]
        if yaw is not None:
            pool["yaw"] = yaw
        return pool

    saved = (sgcore.SurfCore, sgrew.map_spawn_pool, sgzones.load_zones)
    sgcore.SurfCore = FakeCore
    sgrew.map_spawn_pool = fake_pool
    sgzones.load_zones = lambda p, create=True: {"end": end, "start": None}
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run"
            args = build_parser().parse_args([
                "--map", str(Path(td) / "fake.bsp"), "--out", str(out),
                "--envs", "16", "--cell", "128", "--decisions", "20",
                "--act-every", "2", "--min-run", "5", "--max-iters", "400",
                "--print-every", "100", "--snapshot-every", "0", "--seed", "3",
            ])
            rc = explore(args)
            check("loop reached the goal", rc == 0, f"rc={rc}")
            with np.load(out / "archive.npz", allow_pickle=False) as arc:
                key, state, seen = arc["key"], arc["state"], arc["seen"]
                par, dep = arc["parent"], arc["depth"]
            n = len(key)
            check("archive npz written", n > 20, n)
            check("archive states dtype", state.dtype == STATE_DTYPE)
            check("cell keys are unique", len(set(key.tolist())) == n, n)
            check("counts are per run, not per tick",
                  int(seen.max()) >= 1 and int(seen.sum()) >= n,
                  (int(seen.max()), int(seen.sum())))
            check("roots keep parent -1", int((par < 0).sum()) >= 1,
                  int((par < 0).sum()))
            check("every non-root parent is a real index",
                  bool(np.all(par < n)), par.max())
            inner = par >= 0
            check("depth strictly increases parent -> child",
                  bool(np.all(dep[inner] > dep[par[inner]])),
                  int((dep[inner] <= dep[par[inner]]).sum()))
            check("meta json written", (out / "archive.meta.json").exists())
            meta = json.loads((out / "archive.meta.json").read_text("utf-8"))
            check("meta records the weight", meta["weight"] == "1/sqrt(C_seen+1)")

            lines = [json.loads(x) for x in
                     (out / "win_1.traj.jsonl").read_text("utf-8").splitlines()]
            check("win traj header+rows+trailer", len(lines) >= 3, len(lines))
            check("win traj rows are docs/04 wide",
                  all(len(r) == 15 for r in lines[1:-1]))
            check("win traj ticks count out",
                  lines[-1]["ticks"] == len(lines) - 2, lines[-1])
            xs = [r[1] for r in lines[1:-1]]
            check("win traj ends near the finish box", max(xs) >= 2500.0, max(xs))

            spine = np.load(out / "win_1.spine.npy", allow_pickle=False)
            check("spine dtype", spine.dtype == STATE_DTYPE, spine.dtype)
            check("spine non-trivial", len(spine) > 3, len(spine))
            check("spine origins unique at 0.1u",
                  len({tuple(np.round(np.asarray(r["origin"], np.float64), 1))
                       for r in spine}) == len(spine))
            check("spine starts at a map spawn",
                  abs(float(spine[0]["origin"][0])) < 1e-3,
                  spine[0]["origin"])
            check("spine ends at the finish",
                  float(spine[-1]["origin"][0]) >= 2500.0, spine[-1]["origin"])
            ch = json.loads((out / "win_1.chain.json").read_text("utf-8"))
            check("chain spans several bursts", ch["chain_len"] >= 3,
                  ch["chain_len"])
            check("chain json depths ascend",
                  all(b["depth"] > a["depth"]
                      for a, b in zip(ch["chain"], ch["chain"][1:])),
                  [c["depth"] for c in ch["chain"]][:8])
            check("chain json distances trend to the goal",
                  ch["chain"][-1]["dist"] < ch["chain"][0]["dist"],
                  (ch["chain"][0]["dist"], ch["chain"][-1]["dist"]))

            # the paper's literal batch (idle after death) must also run
            out2 = Path(td) / "run2"
            args2 = build_parser().parse_args([
                "--map", str(Path(td) / "fake.bsp"), "--out", str(out2),
                "--envs", "8", "--cell", "128", "--decisions", "10",
                "--act-every", "2", "--restart-dead", "0", "--max-iters", "20",
                "--goals", "0", "--print-every", "100", "--snapshot-every", "0",
            ])
            rc2 = explore(args2)
            check("--restart-dead 0 runs to the iter cap", rc2 == 1, rc2)
            with np.load(out2 / "archive.npz", allow_pickle=False) as a2:
                check("idle-batch archive still grows", len(a2["key"]) > 5,
                      len(a2["key"]))
    finally:
        sgcore.SurfCore, sgrew.map_spawn_pool, sgzones.load_zones = saved


def selftest() -> int:
    ok = 0
    fail = 0

    def check(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS {name}")
        else:
            fail += 1
            print(f"  FAIL {name} {extra}")

    print("cell hash")
    h = CellHash(mins=(-1000.0, -1000.0, -1000.0), maxs=(1000.0, 1000.0, 1000.0),
                 cell=256.0)
    check("dims", h.dims == (9, 9, 9), h.dims)
    s = _fake_states([[-1000, -1000, -1000], [-1000 + 255.9, -1000, -1000],
                      [-1000 + 256.0, -1000, -1000], [-1000, -1000 + 256.0, -1000],
                      [-1000, -1000, -1000 + 256.0], [5000, 5000, 5000]])
    k = h.keys(s)
    check("origin cell is 0", k[0] == 0, k[0])
    check("within-cell equal", k[1] == k[0], k[:2])
    check("+x steps by 1", k[2] == k[0] + 1, k[2])
    check("+y steps by nx", k[3] == k[0] + h.dims[0], k[3])
    check("+z steps by nx*ny", k[4] == k[0] + h.dims[0] * h.dims[1], k[4])
    check("out of bounds clamps", 0 <= k[5] < h.capacity, k[5])
    # trainer parity: rewards.py builds the same key from map_bounds mins
    ix = int((-1000 + 256.0 - -1000) // 256.0)
    check("formula parity", k[2] == ix + h.dims[0] * (0 + h.dims[1] * 0), k[2])

    hv = CellHash((-1000.0,) * 3, (1000.0,) * 3, 256.0, view=8)
    sv = _fake_states([[0, 0, 0], [0, 0, 0]])
    sv["yaw"] = [0.0, 200.0]
    kv = hv.keys(sv)
    check("view splits a cell", kv[0] != kv[1], kv)
    check("view keeps position", kv[0] // 8 == kv[1] // 8, kv)
    hs = CellHash((-1000.0,) * 3, (1000.0,) * 3, 256.0, speed=4, speed_max=4000.0)
    ss = _fake_states([[0, 0, 0], [0, 0, 0]])
    ss["velocity"] = [[10.0, 0, 0], [3000.0, 0, 0]]
    ks = hs.keys(ss)
    check("speed splits a cell", ks[0] != ks[1], ks)

    print("group_entries")
    # 2 envs, 6 ticks. env0 = run 0 the whole time, visiting A A B B A C;
    # env1 = run 1 for 3 ticks (D D E) then idle.
    keys = np.array([[10, 40], [10, 40], [20, 50], [20, -1], [10, -1], [30, -1]],
                    np.int64)
    rid = np.array([[0, 1], [0, 1], [0, 1], [0, -1], [0, -1], [0, -1]], np.int32)
    gk, gr, gt, ge = group_entries(keys, rid)
    check("run0 distinct cells", sorted(gk[gr == 0].tolist()) == [10, 20, 30],
          gk[gr == 0])
    check("A counted once despite revisit", int((gk[gr == 0] == 10).sum()) == 1,
          gk[gr == 0])
    check("first tick of A is 0", int(gt[(gr == 0) & (gk == 10)][0]) == 0, gt)
    check("first tick of B is 2", int(gt[(gr == 0) & (gk == 20)][0]) == 2, gt)
    check("run1 stops at idle", sorted(gk[gr == 1].tolist()) == [40, 50],
          gk[gr == 1])
    check("idle rows dropped", int((gr < 0).sum()) == 0, gr)
    check("tick ordered within run",
          bool(np.all(np.diff(gt[gr == 0]) > 0)), gt[gr == 0])
    check("env recorded", sorted(set(ge[gr == 1].tolist())) == [1], ge)

    print("archive ingest / provenance")
    h2 = CellHash((0.0,) * 3, (10000.0,) * 3, 256.0)
    a = Archive(h2, capacity=4)
    root = _fake_states([[0.0, 0.0, 0.0]])
    ri = a.add(int(h2.keys(root)[0]), root[0], depth=0, parent=-1)
    check("root index 0", ri == 0)
    # one run from the root: enters cells 100, 200, 300 at ticks 0(root),2,5,9
    g_key = np.array([a.key[0], 100, 200, 300], np.int64)
    g_run = np.array([0, 0, 0, 0], np.int64)
    g_tick = np.array([0, 2, 5, 9], np.int64)
    g_state = _fake_states([[0, 0, 0], [300, 0, 0], [600, 0, 0], [900, 0, 0]])
    added = a.ingest(g_key, g_run, g_tick, g_state, np.array([0]), np.array([0]),
                     np.array([0]))
    check("3 new cells", added.tolist() == [1, 2, 3], added)
    check("root C_seen +1", a.seen[0] == 1, a.seen[:4])
    check("depths are ticks from start", a.depth[:4].tolist() == [0, 2, 5, 9],
          a.depth[:4])
    check("parent chain is per cell entry",
          a.parent[:4].tolist() == [-1, 0, 1, 2], a.parent[:4])
    check("chain root-first", a.chain(3) == [0, 1, 2, 3], a.chain(3))

    # a second run from cell 200 (depth 5) reaches 300 in 1 tick -> depth 6,
    # worse than 9? no: 5+1=6 < 9, so it must replace and re-parent.
    a.chosen[2] = 7
    g2k = np.array([200, 300], np.int64)
    g2r = np.array([0, 0], np.int64)
    g2t = np.array([0, 1], np.int64)
    g2s = _fake_states([[600, 0, 0], [901, 0, 0]])
    a.ingest(g2k, g2r, g2t, g2s, np.array([0]), np.array([5]), np.array([2]))
    check("improved depth", a.depth[3] == 6, a.depth[3])
    check("improved state", abs(float(a.state[3]["origin"][0]) - 901.0) < 1e-3,
          a.state[3]["origin"])
    check("re-parented to 200", a.parent[3] == 2, a.parent[3])
    check("C_seen accumulated", a.seen[3] == 2, a.seen[3])
    check("C_chosen kept for unimproved", a.chosen[2] == 7, a.chosen[2])
    # a worse run must not overwrite
    a.ingest(np.array([300], np.int64), np.array([0], np.int64),
             np.array([50], np.int64), _fake_states([[999, 0, 0]]),
             np.array([0]), np.array([0]), np.array([0]))
    check("worse depth rejected", a.depth[3] == 6, a.depth[3])
    check("worse state rejected",
          abs(float(a.state[3]["origin"][0]) - 901.0) < 1e-3, a.state[3]["origin"])
    check("C_seen still counted", a.seen[3] == 3, a.seen[3])

    print("archive growth")
    a2 = Archive(CellHash((0.0,) * 3, (1e6,) * 3, 256.0), capacity=2)
    m = 500
    ks2 = np.arange(m, dtype=np.int64) * 7 + 1
    st2 = _fake_states(np.stack([np.arange(m) * 300.0, np.zeros(m),
                                 np.zeros(m)], 1))
    a2.ingest(ks2, np.zeros(m, np.int64), np.arange(m, dtype=np.int64), st2,
              np.array([0]), np.array([0]), np.array([-1]))
    check("grew past capacity", a2.size == m, a2.size)
    check("keys intact", a2.key[:m].tolist() == ks2.tolist())
    check("states intact",
          abs(float(a2.state[m - 1]["origin"][0]) - (m - 1) * 300.0) < 1e-2)
    check("chain from -1 parent", a2.chain(0) == [0], a2.chain(0))

    print("selection weights")
    a3 = Archive(CellHash((0.0,) * 3, (1e5,) * 3, 256.0), capacity=8)
    for j, seen_j in enumerate([0, 3, 15, 99]):
        a3.add(j + 1, _fake_states([[j * 300.0, 0, 0]])[0])
        a3.seen[j] = seen_j
    a3.build_sampler()
    draws = a3.draw(400_000, np.random.default_rng(7))
    freq = np.bincount(draws, minlength=4) / 400_000.0
    w = 1.0 / np.sqrt(np.array([0, 3, 15, 99], np.float64) + 1.0)
    want = w / w.sum()
    err = float(np.max(np.abs(freq - want)))
    check("W = 1/sqrt(C_seen+1) empirically", err < 0.004,
          f"max abs err {err:.4f} freq {np.round(freq, 4)} want {np.round(want, 4)}")
    check("never starves a hot cell", freq[3] > 0.02, freq[3])
    check("prefers the frontier", freq[0] > freq[3] * 2.5, freq)

    print("cycle guard")
    a4 = Archive(CellHash((0.0,) * 3, (1e5,) * 3, 256.0), capacity=4)
    a4.add(1, _fake_states([[0.0, 0, 0]])[0])
    a4.add(2, _fake_states([[300.0, 0, 0]])[0])
    a4.parent[0] = 1
    a4.parent[1] = 0
    ch = a4.chain(1)
    check("cycle truncated, no hang", len(ch) <= a4.size + 1, ch)

    print("spine dedup")
    sp = _fake_states([[0, 0, 0], [0.04, 0, 0], [100, 0, 0], [0, 0, 0]])
    d = dedup_spine(sp)
    check("collapses 0.1u duplicates", len(d) == 2, len(d))
    check("keeps first occurrence", abs(float(d[0]["origin"][0])) < 1e-6)

    print("jsonl round trip")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.traj.jsonl"
        st = _fake_states([[1, 2, 3], [4, 5, 6]])
        st["velocity"] = [[10, 0, 0], [20, 0, 0]]
        st["yaw"] = [90.0, 91.0]
        acts = np.array([[7, 3, 2, 0, 1, 0], [7, 3, 0, 2, 0, 1]], np.int8)
        write_traj_jsonl(p, st, acts, np.zeros(2, np.float32), "m", 10, {}, "done")
        lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    check("header + 2 rows + trailer", len(lines) == 4, len(lines))
    check("header has map", lines[0].get("map") == "m", lines[0])
    check("row width 15", len(lines[1]) == 15, len(lines[1]))
    check("row xyz", lines[1][1:4] == [1.0, 2.0, 3.0], lines[1][1:4])
    check("jump button", lines[1][8] == SURF_IN_JUMP, lines[1][8])
    check("duck button", lines[2][8] == SURF_IN_DUCK, lines[2][8])
    check("fwd/side echoed", lines[2][13:15] == [0, 2], lines[2][13:15])
    check("trailer", lines[3].get("end") == "done", lines[3])

    print("euclid distance")
    e = EuclidDist({"mins": [0, 0, 0], "maxs": [10, 10, 10]})
    d2 = e.sample(np.array([[5.0, 5.0, 5.0], [0.0, 0.0, -30.0]]))
    check("inside is 0", abs(d2[0]) < 1e-9, d2[0])
    check("outside is the gap", abs(d2[1] - 30.0) < 1e-6, d2[1])

    print("end-to-end loop against a fake core")
    _loop_selftest(check)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Go-Explore Phase 1 reward-free discovery on a surf map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_src_cannonball.bsp"),
                    help="path to the .bsp")
    ap.add_argument("--out", default=str(ROOT / "runs" / "explore"),
                    help="output directory (archive, winning trajectories)")
    ap.add_argument("--envs", type=int, default=512,
                    help="parallel envs; each holds one exploration run")
    ap.add_argument("--cell", type=float, default=256.0,
                    help="cell size in map units (256 matches the trainer's "
                         "novelty cells)")
    ap.add_argument("--cell-view", type=int, default=0,
                    help="yaw sectors folded into the cell key (0 = off)")
    ap.add_argument("--cell-speed", type=int, default=0,
                    help="horizontal-speed buckets folded into the cell key "
                         "(0 = off); surf walls are speed-gated, so the same "
                         "place at a new speed is arguably a new state")
    ap.add_argument("--cell-speed-max", type=float, default=3500.0,
                    help="top of the speed-bucket range")
    ap.add_argument("--decisions", type=int, default=100,
                    help="exploration burst length, k in the paper "
                         "(100 for Atari, 30 for robotics)")
    ap.add_argument("--act-every", type=int, default=3,
                    help="physics ticks per decision (the trainer's frameskip)")
    ap.add_argument("--repeat-p", type=float, default=0.95,
                    help="probability of repeating the previous action at each "
                         "decision (paper: 0.95 Atari, 0.90 robotics)")
    ap.add_argument("--min-run", type=int, default=25,
                    help="do not restart a dead env with fewer than this many "
                         "decisions left in the chunk; it idles instead")
    ap.add_argument("--restart-dead", type=int, default=1, choices=(0, 1),
                    help="1: re-seed an env the moment its episode ends "
                         "(worker pool). 0: the paper's literal batch, where "
                         "dead envs idle to the end of the chunk")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="stop after this many chunks (0 = until the goal)")
    ap.add_argument("--goals", type=int, default=1,
                    help="stop after this many goal crossings (0 = never)")
    ap.add_argument("--ep-ticks", type=int, default=6000,
                    help="core max_episode_ticks; only bites if a burst is "
                         "longer than this")
    ap.add_argument("--snapshot-every", type=int, default=200,
                    help="save the archive every N chunks (0 = only at exit)")
    ap.add_argument("--print-every", type=int, default=10,
                    help="progress line every N chunks")
    ap.add_argument("--max-cells", type=int, default=4_000_000,
                    help="hard cap on archive entries (memory guard: each "
                         "entry is ~150 bytes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP_NUM_THREADS for the core (0 = leave as is)")
    ap.add_argument("--keep-tick", action="store_true",
                    help="do NOT zero tick/stuck_ticks on restore (a deep "
                         "state then truncates on arrival)")
    ap.add_argument("--keep-teleports", action="store_true",
                    help="do not treat a map teleport as a fail (surf maps "
                         "teleport fallers into a jail cell)")
    ap.add_argument("--random-pitch", action="store_true",
                    help="randomize the pitch action too; inert for physics "
                         "(src/env.c passes 0 pitch into pm_tick)")
    ap.add_argument("--duck", type=int, default=1, choices=(0, 1),
                    help="include the duck bit in the random action set")
    ap.add_argument("--face-goal", action="store_true",
                    help="face root spawns down the distance gradient instead "
                         "of using the map's (often unreliable) entity yaw")
    ap.add_argument("--selftest", action="store_true",
                    help="run the CPU unit tests (no DLL, no map) and exit")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.selftest:
        return selftest()
    if args.threads > 0:
        # must precede the DLL load for OpenMP to pick it up
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if not 0.0 <= args.repeat_p < 1.0:
        raise SystemExit(f"--repeat-p must be in [0, 1), got {args.repeat_p}")
    if args.act_every < 1:
        raise SystemExit("--act-every must be >= 1")
    if args.decisions < 1:
        raise SystemExit("--decisions must be >= 1")
    if args.envs < 1:
        raise SystemExit("--envs must be >= 1")
    return explore(args)


if __name__ == "__main__":
    raise SystemExit(main())
