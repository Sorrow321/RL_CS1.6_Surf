#!/usr/bin/env python3
"""certify_field.py - a REACHABILITY-CERTIFIED goal potential from an archive.

The trainer's goal field (`surfgym.goalfield.build_goal_field`) is a BFS
over the map's FREE SPACE: it asks whether a voxel is occupiable, never
whether the player can get from one voxel to the next. On a surf map that
plans glides through open air no policy can fly (the cannonball wall, the
unitfarmer pit, the edgeflow pit), and the shaping then descends into the
obstacle. Robotics' answer (Roth 2025, SoRB, Wellhausen 2021, RL-RRT): the
high level must only use edges the low level can traverse.

This tool builds that field from an archive whose edges were CERTIFIED by
running the simulator: `tools/explore_phase1.py`'s `archive.npz` records,
for every cell, the cell it was first reached FROM by a random
macro-action burst that survived (its `parent`). Dijkstra from the goal
over that provenance tree (edge cost = the Euclidean distance between the
two cells' states; the spawn roots joined at zero cost - they are all the
map start; plus the last leg into the finish box) gives a potential on the
archive cells. It is then extended to every voxel of the trainer's lattice
as

    Phi(v) = d_cert(c*) + |v - c*|,   c* = the archive cell nearest v

so every state the policy visits has a value and an off-tree state is
pulled toward the certified cell it lies nearest to (a global min over
cells would re-create the free-flight shortcut through the goal cell). The result is written in the
trainer's `GoalField` cache format (uint16 grid x quant, mins, cell,
reach_max) so `--goal-field-file` can load it in place of the BFS field.

Generic: no constant is read off a map (the cell size is the archive's,
the lattice the trainer's). The archive can come from the reward-free
search or from the POLICY'S OWN rollouts; re-run as the policy improves.

    python tools/certify_field.py --map maps_pool/surf_edgeflow_blue050.bsp \\
        --archive runs/explore_blue050/archive.npz \\
        --spine runs/explore_blue050/win_1.spine.npy \\
        --out runs/explore_blue050/certified_goal_32.npz
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import GoalField           # noqa: E402
from surfgym.zones import load_zones              # noqa: E402


def certified_cells(archive: Path, goal_pos: np.ndarray, box: tuple[np.ndarray, np.ndarray],
                    hop_xy: float = 256.0, hop_z: float = 96.0, dense: float = 64.0):
    """Dijkstra over the archive's provenance tree from the cell nearest
    ``goal_pos`` (the winning chain's last state). Returns (positions,
    distances) for every archive cell, distances in map units."""
    z = np.load(archive, allow_pickle=True)
    pos = z["state"]["origin"].astype(np.float64)
    parent = np.asarray(z["parent"], np.int64)
    n = len(pos)
    goal = int(np.argmin(np.linalg.norm(pos - goal_pos, axis=1)))
    roots = np.flatnonzero(parent < 0)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for i, p in enumerate(parent):
        if p >= 0:
            w = float(np.linalg.norm(pos[i] - pos[p]))
            adj[i].append((int(p), w))
            adj[int(p)].append((i, w))
    for a in roots:                       # every root is a map spawn: one start
        for b in roots:
            if a != b:
                adj[int(a)].append((int(b), 0.0))
    # LOCAL HOPS: the provenance tree keeps one parent per cell, so two cells
    # that sit on the same platform can carry wildly different tree
    # distances (blue050: a cell 260 u from the goal-side cell, reached first
    # from a dead-end branch, read 4,609 u against its neighbour's 208 u).
    # Two cells whose states are within one cell horizontally and within a
    # jump's reach vertically are joined - the player's own hop, a physics
    # constant, never a map one. A rim cell and a pit cell are 300 u apart
    # vertically and stay disconnected.
    if hop_xy > 0.0:
        from scipy.spatial import cKDTree
        pairs = cKDTree(pos[:, :2]).query_pairs(hop_xy)
        n_hop = 0
        for i, j in pairs:
            if abs(pos[i, 2] - pos[j, 2]) <= hop_z:
                w = float(np.linalg.norm(pos[i] - pos[j]))
                adj[int(i)].append((int(j), w)); adj[int(j)].append((int(i), w)); n_hop += 1
        print(f"local hops: {n_hop} cell pairs within {hop_xy:.0f} u horizontally and {hop_z:.0f} u vertically")
    d = np.full(n, np.inf)
    d[goal] = 0.0
    pq = [(0.0, goal)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > d[u]:
            continue
        for v, w in adj[u]:
            if du + w < d[v]:
                d[v] = du + w
                heapq.heappush(pq, (d[v], v))
    gmin, gmax = box
    leg = float(np.linalg.norm(np.clip(pos[goal], gmin, gmax) - pos[goal]))
    d = d + leg
    # the LAST LEG: the goal-side node sits outside the box, and without a
    # node inside it every voxel between them reads node + distance and
    # RISES toward the finish; a node at the box's nearest point with d = 0
    # (a child of the goal node) makes the leg a straight certified descent
    box_pt = np.clip(pos[goal], gmin, gmax)
    pos = np.vstack([pos, box_pt[None, :]])
    d = np.concatenate([d, [0.0]])
    parent = np.concatenate([parent, [goal]])
    # DENSIFY the tree's edges: a chain node is the state at CELL ENTRY, so
    # consecutive chain nodes can be several hundred units apart when the
    # burst was fast, and the voxels between them would otherwise fall to
    # whatever dead-end node lies within the extension radius. Points every
    # `dense` units along each parent->child segment inherit an interpolated
    # distance, so the whole segment is a certified descent.
    if dense > 0.0:
        extra_pos, extra_d = [], []
        for i, pp in enumerate(parent):
            if pp < 0 or not (np.isfinite(d[i]) and np.isfinite(d[pp])):
                continue
            seg = pos[i] - pos[pp]
            L = float(np.linalg.norm(seg))
            n_pts = int(L // dense)
            for k in range(1, n_pts + 1):
                f = k * dense / L
                extra_pos.append(pos[pp] + f * seg)
                extra_d.append(d[pp] + f * (d[i] - d[pp]))
        if extra_pos:
            pos = np.vstack([pos, np.array(extra_pos)])
            d = np.concatenate([d, np.array(extra_d)])
            print(f"densified: {len(extra_pos)} points every {dense:.0f} u along the tree's edges")
    return pos, d, goal, roots


def certify_transitions(bsp: Path, archive: Path, meta: dict, *, envs: int = 512,
                        iters: int = 100, decisions: int = 100, act_every: int = 4,
                        repeat_p: float = 0.95, ep_ticks: int = 3000, seed: int = 0,
                        cert_cell: float = 0.0, support_reach: float = 160.0,
                        lattice_cell: float = 32.0, vz_bins: bool = True,
                        vz_fall: float = 400.0, vz_rise: float = 150.0):
    """Roll random macro-action bursts from EVERY archive cell's state and
    record every alive cell -> cell transition (and cell -> GOAL on a finish
    hit). Returns (node_pos {key: xyz}, edges {(a, b): count}, goal_keys).
    The provenance tree keeps one parent per cell; this keeps every move
    the simulator actually allowed, in the direction it allowed it."""
    sys.path.insert(0, str(ROOT / "tools"))
    import explore_phase1 as ep
    from surfgym.core import SurfCore, default_config
    from surfgym.zones import load_zones as _lz
    z = np.load(archive, allow_pickle=True)
    states = z["state"]
    n_cells = len(states)
    n = int(envs)
    ticks = int(decisions) * int(act_every)
    cfg = default_config(num_envs=n, spawn_mode=2, max_episode_ticks=int(ep_ticks),
                         water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    core = SurfCore(str(bsp), cfg)
    core.set_teleport_fail(True)
    zones = _lz(str(bsp))
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    mins, maxs = core.map_bounds()
    ccell = float(cert_cell) if cert_cell else float(meta["cell"])
    hasher = ep.CellHash(mins, maxs, cell=ccell, view=int(meta["cell_view"]),
                         speed=int(meta["cell_speed"]), speed_max=float(meta["cell_speed_max"]))
    rng = np.random.default_rng(seed)
    act = np.zeros((n, 6), np.int32)
    pitch_bin = 3

    def fresh(idx):
        m = idx.size
        if m == 0:
            return
        for j, hi in enumerate(ep.ACTION_NVEC):
            act[idx, j] = rng.integers(0, hi, m)
        act[idx, 1] = pitch_bin
    core.set_spawn_pool(states)
    sv = core.states_view
    # the support mask: solid within `support_reach` below, above the kill
    # ceiling, on the trainer's 32 u lattice (dip_probe's route model)
    sup_mask = None
    if support_reach > 0.0:
        import dip_probe as dp
        from surfgym.goalfield import goal_occupancy
        occ, omins = goal_occupancy(core, float(lattice_cell), None)
        occ = np.asarray(occ, bool)
        kz = dp.kill_ceiling(str(bsp))
        zs = omins[2] + (np.arange(occ.shape[0]) + 0.5) * float(lattice_cell)
        dead = np.zeros(occ.shape, bool)
        dead[zs <= kz, :, :] = True
        sup_mask = dp.supported(occ, max(1, int(round(support_reach / float(lattice_cell)))), dead)
        omins = np.asarray(omins, np.float64)

        def supported(origins):
            f = np.floor((np.asarray(origins, np.float64) - omins) / float(lattice_cell)).astype(int)
            f[:, 0] = np.clip(f[:, 0], 0, occ.shape[2] - 1)
            f[:, 1] = np.clip(f[:, 1], 0, occ.shape[1] - 1)
            f[:, 2] = np.clip(f[:, 2], 0, occ.shape[0] - 1)
            return sup_mask[f[:, 2], f[:, 1], f[:, 0]]
        print(f"support rule: a move counts only into a state with solid within {support_reach:.0f} u "
              f"below it ({int(sup_mask.sum()):,} supported voxels of {int((~occ).sum()):,} free)")
    def vz_bin(st):
        vz = np.asarray(st["velocity"], np.float64)[..., 2]
        return np.where(vz < -float(vz_fall), 0, np.where(vz > float(vz_rise), 2, 1)).astype(np.int64)

    def keyed(st):
        k = hasher.keys(st).astype(np.int64)
        return k * 3 + vz_bin(st) if vz_bins else k
    node_pos = {}
    for k, st in zip(keyed(states), states):
        node_pos.setdefault(int(k), np.array(st["origin"], np.float64))
    edges = {}
    goal_keys = {}
    deaths = {}
    live = np.zeros(n, bool)
    t0 = time.perf_counter()
    for it in range(int(iters)):
        core.reset(seed + it)
        picks = (np.arange(n) + it * n) % n_cells          # every cell, in turn
        for e in range(n):
            st = states[picks[e]:picks[e] + 1].copy()
            st["tick"] = 0
            st["stuck_ticks"] = 0
            core.set_state(e, st)
        live[:] = True
        fresh(np.arange(n))
        keys_before = keyed(sv)
        for t in range(ticks):
            if t % act_every == 0:
                fresh(np.flatnonzero(live & (rng.random(n) >= repeat_p)))
            _, _, done, trunc, _ = core.step(act)
            ended = ((done | trunc) != 0) & live
            hit = np.asarray(core.goal_hits, bool) & live
            keys_after = keyed(sv)
            moved = live & ~ended & (keys_after != keys_before)
            sup_now = None
            if sup_mask is not None:
                # only SUPPORTED states are nodes; while airborne the env keeps
                # its last supported key, so a ramp-to-ramp flight certifies
                # the move between the two ramps and the air is never a node
                sup_now = supported(sv["origin"])
                moved &= sup_now
            for e in np.flatnonzero(moved):
                a, b = int(keys_before[e]), int(keys_after[e])
                edges[(a, b)] = edges.get((a, b), 0) + 1
                if b not in node_pos:
                    node_pos[b] = np.array(sv["origin"][e], np.float64)
            for e in np.flatnonzero(hit):
                goal_keys[int(keys_before[e])] = goal_keys.get(int(keys_before[e]), 0) + 1
            for e in np.flatnonzero(ended & ~hit):
                deaths[int(keys_before[e])] = deaths.get(int(keys_before[e]), 0) + 1
            live &= ~ended
            adv = live if sup_now is None else (live & sup_now)
            keys_before = np.where(adv, keys_after, keys_before)
            if not live.any():
                break
    core.close()
    print(f"certification: {iters} x {n} bursts of {decisions} decisions from {n_cells} cells in "
          f"{time.perf_counter() - t0:.0f}s -> {len(node_pos)} nodes, {len(edges)} directed edges, "
          f"{sum(goal_keys.values())} finish hits from {len(goal_keys)} cells, deaths in {len(deaths)} cells")
    certify_transitions.deaths = deaths
    return node_pos, edges, goal_keys


def certified_from_graph(node_pos: dict, edges: dict, goal_keys: dict, box, min_count: int = 1):
    """Distance-to-goal along CERTIFIED FORWARD moves: Dijkstra from GOAL over
    the reversed edges, edge cost = the Euclidean distance between the two
    nodes' positions. Nodes with no certified way to the goal (the pit) get
    the largest connected value plus their distance to the nearest connected
    node, so the extension still pulls them back toward the certified set."""
    keys = sorted(node_pos)
    idx = {k: i for i, k in enumerate(keys)}
    pos = np.array([node_pos[k] for k in keys])
    gmin, gmax = box
    radj = {i: [] for i in range(len(keys))}
    for (a, b), c in edges.items():
        if c >= min_count and a in idx and b in idx:
            w = float(np.linalg.norm(pos[idx[a]] - pos[idx[b]]))
            radj[idx[b]].append((idx[a], w))          # reversed: b <- a
    d = np.full(len(keys), np.inf)
    pq = []
    for k in goal_keys:
        if k in idx:
            i = idx[k]
            d[i] = float(np.linalg.norm(np.clip(pos[i], gmin, gmax) - pos[i]))
            heapq.heappush(pq, (d[i], i))
    while pq:
        du, u = heapq.heappop(pq)
        if du > d[u]:
            continue
        for v, w in radj[u]:
            if du + w < d[v]:
                d[v] = du + w
                heapq.heappush(pq, (d[v], v))
    ok = np.isfinite(d)
    if ok.any() and (~ok).any():
        from scipy.spatial import cKDTree
        dist, near = cKDTree(pos[ok]).query(pos[~ok])
        d[~ok] = d[ok].max() + dist
    return pos, d, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--archive", required=True, help="explore_phase1's archive.npz")
    ap.add_argument("--spine", required=True,
                    help="the winning spine (.npy); its last state marks the goal-side cell")
    ap.add_argument("--goal-cell", type=float, default=32.0,
                    help="the trainer's lattice; the BFS cache <map>.goal_<cell>.npz supplies the lattice and the free-voxel mask")
    ap.add_argument("--hop-xy", type=float, default=256.0,
                    help="join archive cells within this horizontal distance (one archive cell) ...")
    ap.add_argument("--hop-z", type=float, default=96.0,
                    help="... and within this vertical distance (a jump's reach); 0 = provenance tree only")
    ap.add_argument("--certify-iters", type=int, default=100,
                    help="re-certify edges by rolling random bursts from every archive cell "
                         "(iterations x --certify-envs bursts); 0 = the provenance tree + local hops only")
    ap.add_argument("--certify-envs", type=int, default=512)
    ap.add_argument("--support-reach", type=float, default=160.0,
                    help="a certified move must enter a state with solid within this many u "
                         "below it (a ramp face under a surfer, a floor under a walker); 0 = off")
    ap.add_argument("--vz-bins", type=int, default=1, choices=(0, 1),
                    help="key the certification nodes by a vertical-velocity bin too (falling / "
                         "level-or-riding / rising), so a player falling through a cell and one "
                         "riding a ramp in it are different nodes; 1 = on")
    ap.add_argument("--vz-fall", type=float, default=400.0, help="falling bin below -this (u/s)")
    ap.add_argument("--vz-rise", type=float, default=150.0, help="rising bin above +this (u/s)")
    ap.add_argument("--dense", type=float, default=64.0,
                    help="tree mode: interpolate points every this many u along the tree's edges before the extension; 0 = off")
    ap.add_argument("--certify-cell", type=float, default=0.0,
                    help="cell size for the certification graph (0 = the archive's, 256 u); the "
                         "archive's states stay the roots, the transitions are hashed finer")
    ap.add_argument("--extend-radius", type=float, default=1.5,
                    help="voxels take the best [d(node) + distance] over nodes within this many "
                         "ARCHIVE cells (256 u each); 0 = the single nearest node")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bsp = Path(a.map)
    zones = load_zones(str(bsp))
    gmin, gmax = np.asarray(zones["end"]["mins"], float), np.asarray(zones["end"]["maxs"], float)
    cache = bsp.parent / f"{bsp.stem}.goal_{int(a.goal_cell)}.npz"
    if not cache.exists():
        raise SystemExit(f"no BFS cache {cache}: run the trainer or bev_dip once to bake the lattice")
    zc = np.load(cache, allow_pickle=True)
    grid = zc["grid"]
    quant, cell = float(zc["quant"]), float(zc["cell"])
    mins = np.asarray(zc["mins"], np.float64)
    reach_bfs = float(zc["reach_max"])
    bfs = grid.astype(np.float64) * quant
    valid = (grid != np.iinfo(grid.dtype).max) & (bfs < reach_bfs - 0.5 * cell)
    nz, ny, nx = grid.shape

    spine = np.load(a.spine)
    goal_pos = spine["origin"][-1].astype(np.float64)
    meta = json.load(open(str(Path(a.archive).with_name("archive.meta.json")), encoding="utf-8"))
    acell = float(meta["cell"])                  # the archive's cell size (256 u)
    if int(a.certify_iters) > 0:
        node_pos, edges, goal_keys = certify_transitions(
            bsp, Path(a.archive), meta, envs=int(a.certify_envs), iters=int(a.certify_iters),
            decisions=int(meta.get("decisions", 100)), act_every=int(meta.get("act_every", 4)),
            repeat_p=float(meta.get("repeat_p", 0.95)), cert_cell=float(a.certify_cell),
            support_reach=float(a.support_reach), lattice_cell=float(a.goal_cell),
            vz_bins=bool(a.vz_bins), vz_fall=float(a.vz_fall), vz_rise=float(a.vz_rise))
        pos, dcert, ok = certified_from_graph(node_pos, edges, goal_keys, (gmin, gmax))
        acell = float(a.certify_cell) if float(a.certify_cell) > 0 else acell
        # void diagnostic: walk the greedy certified descent from the spawn
        # and trace straight down from each node - a node with no surface
        # within 160 u below it hangs in the air (over the kill box on
        # edgeflow); a 256 u cell can straddle a ramp and the void
        from surfgym.core import SurfCore, SurfEnvConfig
        core = SurfCore(str(bsp), SurfEnvConfig(num_envs=1))
        keys = sorted(node_pos); kidx = {k: i for i, k in enumerate(keys)}
        fadj = {}
        for (ka, kb), c in edges.items():
            fadj.setdefault(kidx[ka], []).append(kidx[kb])
        from surfgym.rewards import map_spawn_pool
        s0 = np.median(map_spawn_pool(core)["origin"].astype(np.float64), axis=0)
        i0 = int(np.argmin(np.linalg.norm(pos - s0, axis=1)))
        path = [i0]
        for _ in range(200):
            nxt = [j for j in fadj.get(path[-1], []) if dcert[j] < dcert[path[-1]]]
            if not nxt:
                break
            path.append(min(nxt, key=lambda j: dcert[j]))
        void = 0
        deaths = certify_transitions.deaths
        ex = {}
        for (ka, kb), c in edges.items():
            ex[ka] = ex.get(ka, 0) + c
        for j in path:
            q = tuple(float(v) for v in pos[j])
            tr = core.trace(q, (q[0], q[1], q[2] - 160.0), hull=0)
            if tr.fraction >= 1.0:
                void += 1
            kj = keys[j]
            dj, xj = deaths.get(kj, 0), ex.get(kj, 0)
            print(f"      ({pos[j][0]:6.0f},{pos[j][1]:6.0f},{pos[j][2]:4.0f}) d {dcert[j]:5.0f} exits {xj:5d} deaths {dj:5d} "
                  f"death share {dj / max(1, dj + xj):.2f} {'VOID' if tr.fraction >= 1.0 else ''}")
        nsup = 0
        for q in spine["origin"].astype(np.float64):
            q = tuple(float(v) for v in q)
            tr = core.trace(q, (q[0], q[1], q[2] - 160.0), hull=0)
            nsup += int(tr.fraction < 1.0)
        print(f"the winning spine's own states with a surface within 160 u below: {nsup} of {len(spine)}")
        print(f"greedy certified descent from the spawn: {len(path)} nodes, leftmost x "
              f"{pos[path][:, 0].min():.0f}, ends at d {dcert[path[-1]]:.0f}; nodes with no surface "
              f"within 160 u below them: {void} of {len(path)}")
        core.close()
        print(f"graph: {len(pos)} nodes, {int(ok.sum())} with a certified way to the finish; "
              f"the rest get the largest connected value + their distance to it")
        ok = np.ones(len(pos), bool)            # every node carries a value now
    else:
        pos, dcert, goal, roots = certified_cells(Path(a.archive), goal_pos, (gmin, gmax),
                                                  hop_xy=float(a.hop_xy), hop_z=float(a.hop_z),
                                                  dense=float(a.dense))
        ok = np.isfinite(dcert)
        print(f"archive: {len(pos)} cells, {int(ok.sum())} connected to the goal-side cell "
              f"{goal} at {pos[goal].round(0).tolist()}, {len(roots)} spawn roots")

    # extend to the lattice: Phi(v) = min_c [ d_cert(c) + |v - c| ] over connected cells
    zs = mins[2] + (np.arange(nz) + 0.5) * cell
    ys = mins[1] + (np.arange(ny) + 0.5) * cell
    xs = mins[0] + (np.arange(nx) + 0.5) * cell
    idx = np.argwhere(valid)
    centers = np.c_[xs[idx[:, 2]], ys[idx[:, 1]], zs[idx[:, 0]]]
    cpos, cd = pos[ok], dcert[ok]
    # NEAREST-cell assignment, not a min over all cells: a global
    # min_c [d(c) + |v - c|] is just the straight-line distance to the goal
    # cell wherever the line is shorter than the tree path, i.e. the
    # free-flight shortcut again. A voxel takes the certified distance of
    # the cell it lies nearest to, plus the way there.
    from scipy.spatial import cKDTree
    tree = cKDTree(cpos)
    dist, near = tree.query(centers, workers=-1)
    phi = cd[near] + dist
    # ... but the single nearest node is fragile: a dead-end node (bursts
    # from it only die) can sit one cell from a route node. Within a local
    # radius take the best node, [d(c) + |v - c|] minimised over the nodes
    # within R = --extend-radius cells; farther voxels keep the nearest
    # node. R is short enough that no gap of the map's scale is bridged.
    R = float(a.extend_radius) * acell          # in ARCHIVE cells, not lattice cells
    if R > 0.0:
        k = min(len(cpos), 64)                    # all nodes within R matter, not the 8 nearest
        dk, nk = tree.query(centers, k=k, workers=-1)
        dk = np.atleast_2d(dk); nk = np.atleast_2d(nk)
        cand = cd[nk] + dk
        cand[dk > R] = np.inf
        best = np.min(cand, axis=1)
        phi = np.where(np.isfinite(best), best, phi)
    # the finish box itself is the sink
    inbox = np.all((centers >= gmin) & (centers <= gmax), axis=1)
    phi[inbox] = 0.0

    out_grid = np.full(grid.shape, np.iinfo(np.uint16).max, np.uint16)
    reach_max = float(phi.max()) + cell
    q = np.clip(np.round(phi / quant), 0, np.iinfo(np.uint16).max - 1).astype(np.uint16)
    out_grid[idx[:, 0], idx[:, 1], idx[:, 2]] = q
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, grid=out_grid, quant=np.float32(quant), mins=mins, cell=np.float32(cell),
             reach_max=np.float32(reach_max), sig=zc["sig"],
             certified_from=str(a.archive), cells=int(ok.sum()))
    print(f"wrote {out}: {int(valid.sum()):,} voxels, quant {quant:g}, reach_max {reach_max:,.0f}")

    # self-check: the field must fall along the spine and must NOT fall along
    # the straight line the free-space field descends
    gf = GoalField(out_grid.astype(np.float32) * quant, mins, cell, reach_max)
    sp = spine["origin"].astype(np.float64)
    along = gf.sample(sp)
    rise = float(np.max(along - np.minimum.accumulate(along)))
    print(f"self-check along the spine ({len(sp)} states): {along[0]:,.0f} -> {along[-1]:,.0f} u, "
          f"max rise above the running minimum {rise:,.0f} u")
    s0 = sp[0]
    gc = (gmin + gmax) / 2.0
    line = s0[None, :] + np.linspace(0, 1, 60)[:, None] * (gc - s0)[None, :]
    lv = gf.sample(line)
    lv = lv[lv < reach_max - cell]
    lrise = float(np.max(lv - np.minimum.accumulate(lv)))
    print(f"self-check along the straight line spawn -> finish ({len(lv)} valid samples): "
          f"{lv[0]:,.0f} -> {lv[-1]:,.0f} u; max rise above the running minimum {lrise:,.0f} u "
          f"(the barrier the field now puts across the free-space shortcut; the free-space BFS field has 0)")
    meta = {"map": bsp.stem, "archive": str(a.archive), "spine": str(a.spine),
            "cells_connected": int(ok.sum()),
            "spine_rise": rise, "line_rise": lrise}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
