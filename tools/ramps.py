"""ramps.py - a map's contactable SURFACES, grouped into ramps / floors / walls / ceilings, from the
player's own collision (the user's ramp-graph idea, 2026-09-27; v2 after Codex's and Fable's
reviews). No per-map constant.

    python tools/ramps.py maps_pool/<map>.bsp [--cell 16] [--goal-cell 32] [--out ramps.npz]
        [--png ramps.png]

Method (v2):
  1. probe points: the free voxels of the cached point occupancy (<map>.occ_<cell>.npz) that touch
     a solid voxel; for each solid neighbour direction d, a STANDING-hull trace (the hull movement
     uses: core.trace(..., hull=0)) from 48 u back along -d to one cell past the surface. The hit is
     a CONTACT ORIGIN (where the player's origin is when its hull touches) plus the plane normal -
     collision-only (CLIP) geometry included, since the hull trace sees it;
  2. category by the normal (the engine's ground test, pm.c: n_z >= 0.7 is ground): FLOOR
     n_z >= 0.7, RAMP 0.02 < n_z < 0.7 (surfable), WALL |n_z| <= 0.02, CEILING n_z < -0.02;
  3. region growing within one category: a sample joins when within 1.5 cells of a member and its
     normal is within CAP_DEG of BOTH that member's and the region SEED's normal - the seed cap
     keeps a curved run (uf2's 65-degree arch) from chaining into one surface;
  4. per surface: its contact origins, mean normal, size, and the goal field's distance over the
     REACHABLE samples only (GoalField.reachable, not isfinite: the unreachable sentinel is finite):
     min, p10 (the best end), median, max.

RampMap (import from here): loads the .npz; contact(origins) -> surface id per origin, or -1:
the origin is within CONTACT_TOL of a nearby contact origin's plane (a touching hull sits at
~DIST_EPSILON from it; a fly-by is farther), laterally within 1.5 cells.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

WALKABLE_NZ = 0.7          # the engine's ground test (pm.c: plane normal z >= 0.7 is ground)
WALL_NZ = 0.02             # |n_z| below this: a vertical wall
CAP_DEG = 25.0             # a surface's normals stay within this of its seed and of each neighbour
STAND_HALF = np.array([16.0, 16.0, 36.0])   # the standing hull's half extents
BACK_MARGIN = 16.0         # the hull probe starts this far beyond its contact distance
CONTACT_TOL = 2.0          # u: an origin this close to a contact plane is touching it
CATS = ("floor", "ramp", "wall", "ceiling", "kill")   # kill: touching it ends the episode
DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float64)


def category(nz):
    nz = np.asarray(nz)
    return np.where(nz >= WALKABLE_NZ, 0, np.where(nz > WALL_NZ, 1,
                                                   np.where(nz >= -WALL_NZ, 2, 3)))


def extract(core, occ, mins, cell):
    """-> contact origins (N,3), plane normals (N,3) seen by the standing hull from the free
    boundary voxels of the point occupancy grid (occ [nz, ny, nx], 1 = solid)."""
    nz_, ny, nx = occ.shape
    solid = occ.astype(bool)
    pts, nrm, seen = [], [], set()
    for d in DIRS:
        dz, dy, dx = int(d[2]), int(d[1]), int(d[0])
        b = np.zeros_like(solid)
        zs = slice(max(0, -dz), nz_ - max(0, dz))
        ys = slice(max(0, -dy), ny - max(0, dy))
        xs = slice(max(0, -dx), nx - max(0, dx))
        zt = slice(max(0, dz), nz_ - max(0, -dz))
        yt = slice(max(0, dy), ny - max(0, -dy))
        xt = slice(max(0, dx), nx - max(0, -dx))
        b[zs, ys, xs] = solid[zt, yt, xt]
        iz, iy, ix = np.nonzero(~solid & b)
        for k in range(len(iz)):
            c = mins + (np.array([ix[k], iy[k], iz[k]], np.float64) + 0.5) * cell
            # pass 1: a POINT probe finds the surface and its plane; pass 2 places the standing
            # hull on that plane's normal at its own support distance 16|nx|+16|ny|+36|nz| (+ a
            # margin) and traces back along -n. A fixed axis back-off (the old BACK = 48) left
            # the hull start inside the brush for any plane tilted in yaw AND pitch - a 45-deg-yaw
            # ramp needs ~40 u along its normal and 48 u along an axis gives 27.7 (Codex
            # 2026-09-27), so those surfaces were silently never sampled
            tp = core.trace(c.tolist(), (c + d * cell).tolist(), 2)
            if tp.startsolid or tp.allsolid or tp.fraction >= 1.0:
                continue
            q = np.array(tp.endpos[:], np.float64)
            n = np.array(tp.normal[:], np.float64)
            sup = STAND_HALF @ np.abs(n)
            s = q + n * (sup + BACK_MARGIN)
            e = q + n * (sup - BACK_MARGIN)
            tr = core.trace(s.tolist(), e.tolist(), 0)
            if tr.startsolid or tr.allsolid or tr.fraction >= 1.0:
                continue                            # the standing hull cannot stand off here
            p = np.array(tr.endpos[:], np.float64)
            key = tuple(np.round(p / 4.0).astype(int))
            if key in seen:
                continue
            seen.add(key)
            pts.append(p)
            nrm.append(np.array(tr.normal[:], np.float64))
    return np.asarray(pts), np.asarray(nrm)


def group(pts, nrm, cell):
    """region growing within a category, normals capped against the member AND the seed"""
    from scipy.spatial import cKDTree
    n = len(pts)
    cat = category(nrm[:, 2])
    tree = cKDTree(pts)
    cos_c = np.cos(np.radians(CAP_DEG))
    lab = np.full(n, -1, np.int64)
    cur = 0
    for seed in range(n):
        if lab[seed] >= 0:
            continue
        lab[seed] = cur
        sn = nrm[seed]
        stack = [seed]
        while stack:
            i = stack.pop()
            for j in tree.query_ball_point(pts[i], 1.5 * cell):
                if lab[j] >= 0 or cat[j] != cat[seed]:
                    continue
                if float(nrm[j] @ nrm[i]) >= cos_c and float(nrm[j] @ sn) >= cos_c:
                    lab[j] = cur
                    stack.append(j)
        cur += 1
    return lab


def in_kill(bsp, pts):
    """(N,3) contact origins -> bool: inside a kill trigger by the sim's own test (zones.kill_zones
    + hull_probe: the model's HULL-1 clipnodes, the standing-player-inflated hull the engine's
    trigger test walks) - a surface touched there ends the episode, so it is never a target"""
    from surfgym.zones import hull_probe, kill_world_box, kill_zones
    out = np.zeros(len(pts), bool)
    kz = kill_zones(str(bsp))
    if not kz:
        return out
    contains = hull_probe(str(bsp))
    for k in kz:
        lo, hi = kill_world_box(k, (40.0, 40.0, 40.0))    # world box (+ origin), hull-grown
        m = np.all((pts >= lo) & (pts <= hi), axis=1)
        if m.any():
            out[m] |= contains(int(k["model"][1:]), pts[m] - np.asarray(k["origin"], np.float64))
    return out


def merge_coplanar(pts, nrm, lab, cell, min_samples):
    """merge pieces of ONE plane that the probe sampling left apart (e.g. an A-frame face and its
    strip below the ridge): same category, mean normals within 5 deg, plane offsets within 8 u,
    nearest samples within 4 cells - separate parallel or collinear faces stay apart"""
    from scipy.spatial import cKDTree
    ids = [r for r in range(int(lab.max()) + 1) if int((lab == r).sum()) > 0]
    info = {}
    for r in ids:
        m = np.flatnonzero(lab == r)
        nm = nrm[m].mean(0)
        nm = nm / max(np.linalg.norm(nm), 1e-9)
        info[r] = (m, nm, float(nm @ pts[m].mean(0)), int(category(nm[2:3])[0]))
    parent = {r: r for r in ids}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r
    trees = {r: cKDTree(pts[info[r][0]]) for r in ids}
    cos5 = np.cos(np.radians(5.0))
    for i, ri in enumerate(ids):
        mi, ni, oi, ci = info[ri]
        for rj in ids[i + 1:]:
            mj, nj, oj, cj = info[rj]
            if ci != cj or float(ni @ nj) < cos5 or abs(oi - oj) > 8.0:
                continue
            d, _ = trees[ri].query(pts[mj], k=1, distance_upper_bound=4.0 * cell)
            if np.isfinite(d).any():
                a_, b_ = find(ri), find(rj)
                if a_ != b_:
                    parent[a_] = b_
    out = lab.copy()
    for r in ids:
        out[info[r][0]] = find(r)
    return out


class RampMap:
    """The surfaces of one map (a ramps.py v2 .npz) + the contact test."""

    def __init__(self, path):
        from scipy.spatial import cKDTree
        z = np.load(path, allow_pickle=False)
        self.points = z["points"].astype(np.float64)
        self.normals = z["normals"].astype(np.float64)
        self.surf = z["surf"].astype(np.int64)            # per point; -1 = dropped (tiny)
        self.cat = z["cat"].astype(np.int64)              # per surface
        self.d_p10 = z["d_p10"]
        self.d_med = z["d_med"]
        self.normal = z["normal"].astype(np.float64)      # per surface: the mean normal
        self.n_surf = len(self.cat)
        self.cell = float(z["cell"])
        keep = self.surf >= 0
        self.kp, self.kn, self.ks = self.points[keep], self.normals[keep], self.surf[keep]
        self.tree = cKDTree(self.kp)
        self.members = [np.flatnonzero(self.ks == s) for s in range(self.n_surf)]

    def touch_sets(self, counts, normals, points):
        """collision TRUTH (core.get_touch: the planes the movement actually hit, with the player
        origin at impact) -> per env the SET of surface ids touched on that tick; a touch that
        matches no extracted surface (a piece below --min-samples) is -4 (unknown). A touch is
        matched to the nearest contact sample within 1.5 cells whose normal is within 15 deg."""
        counts = np.asarray(counts)
        out = [set() for _ in range(len(counts))]
        rows, cols = np.nonzero(np.arange(normals.shape[1])[None, :] < counts[:, None])
        if len(rows) == 0:
            return out
        pts = np.asarray(points[rows, cols], np.float64)
        nrm = np.asarray(normals[rows, cols], np.float64)
        dist, idx = self.tree.query(pts, k=8, distance_upper_bound=1.5 * self.cell)
        ok = np.isfinite(dist)
        ii = np.where(ok, idx, 0)
        good = ok & (np.einsum("mkj,mj->mk", self.kn[ii], nrm) >= np.cos(np.radians(15.0)))
        dd = np.where(good, dist, np.inf)
        j = np.argmin(dd, axis=1)
        hit = np.isfinite(dd[np.arange(len(pts)), j])
        sid = np.where(hit, self.ks[ii[np.arange(len(pts)), j]], -4)
        for r, sv in zip(rows, sid):
            out[int(r)].add(int(sv))
        return out

    def contact(self, origins, targetable=False):
        """(M,3) -> (M,) surface id the origin touches, or -1 (vectorized). targetable=True:
        only floors and ramps count (the ramp-command contract: grazing a wall, a ramp's bevel or
        a ceiling does not end a command - Codex 17:35Z 'first new TARGETABLE contact')"""
        o = np.atleast_2d(np.asarray(origins, np.float64))
        dist, idx = self.tree.query(o, k=8, distance_upper_bound=1.5 * self.cell)
        ok = np.isfinite(dist)
        if targetable:
            ok &= np.isin(self.cat[self.ks[np.where(ok, idx, 0)]], (0, 1))
        ii = np.where(ok, idx, 0)
        off = np.abs(np.einsum("mkj,mkj->mk", self.kn[ii], o[:, None, :] - self.kp[ii]))
        off = np.where(ok & (off <= CONTACT_TOL), off, np.inf)
        j = np.argmin(off, axis=1)
        hit = np.isfinite(off[np.arange(len(o)), j])
        return np.where(hit, self.ks[ii[np.arange(len(o)), j]], -1).astype(np.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bsp")
    ap.add_argument("--cell", type=float, default=16.0)
    ap.add_argument("--goal-cell", type=int, default=32)
    ap.add_argument("--min-samples", type=int, default=8,
                    help="surfaces with fewer contact samples (~cell^2 each) are dropped")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--kill-aware", action="store_true",
                    help="also mark as KILL every floor / ramp whose contact origins are mostly "
                         "unreachable in the KILL-AWARE geodesic field (build_goal_field "
                         "mask_kill=True: kill volumes are walls) - a surface you can only reach "
                         "through a kill volume (edgeflow's ground under its kill layer) is never "
                         "a target")
    a = ap.parse_args(argv)
    from surfgym.core import SurfCore, default_config
    from surfgym.goalfield import load_goal_field
    bsp = Path(a.bsp)
    core = SurfCore(str(bsp), default_config(num_envs=1))
    z = np.load(bsp.with_name(f"{bsp.stem}.occ_{a.cell:g}.npz"))
    occ, mins = z["occ"], np.asarray(z["mins"], np.float64)
    pts, nrm = extract(core, occ, mins, a.cell)
    lab = merge_coplanar(pts, nrm, group(pts, nrm, a.cell), a.cell, a.min_samples)
    cat_pt = category(nrm[:, 2])
    kill_pt = in_kill(bsp, pts)
    unreach_k = np.zeros(len(pts), bool)
    if a.kill_aware:
        from surfgym.goalfield import build_goal_field
        from surfgym.zones import load_zones
        gk = build_goal_field(core, load_zones(str(bsp))["end"], cell=a.goal_cell,
                              device="cuda", mask_kill=True)
        unreach_k = ~gk.reachable(pts)
    gpath = bsp.with_name(f"{bsp.stem}.goal_{a.goal_cell}.npz")
    gf = load_goal_field(str(gpath)) if gpath.exists() else None
    d_all = gf.sample(pts) if gf is not None else np.full(len(pts), np.nan)
    ok_all = gf.reachable(pts) if gf is not None else np.zeros(len(pts), bool)
    surf = np.full(len(pts), -1, np.int64)
    rows = []
    for r in range(int(lab.max()) + 1):
        m = np.flatnonzero(lab == r)
        if len(m) < a.min_samples:
            continue
        d = d_all[m][ok_all[m]]
        # a surface whose contact origins are mostly inside a kill trigger is a KILL surface
        cat_r = (4 if (float(kill_pt[m].mean()) > 0.5
                       or (int(cat_pt[m[0]]) in (0, 1) and float(unreach_k[m].mean()) > 0.5))
                 else int(cat_pt[m[0]]))
        rows.append({"m": m, "cat": cat_r, "n": len(m),
                     "centroid": pts[m].mean(0), "normal": nrm[m].mean(0),
                     "n_ok": len(d),
                     "d_min": float(d.min()) if len(d) else np.nan,
                     "d_p10": float(np.percentile(d, 10)) if len(d) else np.nan,
                     "d_med": float(np.median(d)) if len(d) else np.nan,
                     "d_max": float(d.max()) if len(d) else np.nan})
    # order: by category (floor, ramp, wall, ceiling), then by the best end's potential, far first
    rows.sort(key=lambda r: (r["cat"], -(r["d_p10"] if np.isfinite(r["d_p10"]) else -1.0)))
    for i, r in enumerate(rows):
        surf[r["m"]] = i
    counts = {c: sum(1 for r in rows if r["cat"] == k) for k, c in enumerate(CATS)}
    print(f"   contact samples inside kill triggers: {int(kill_pt.sum()):,}")
    print(f"ramps.py v2 {bsp.stem}: {len(pts):,} standing-hull contact samples at cell {a.cell:g}; "
          f"{len(rows)} surfaces >= {a.min_samples} samples: " +
          ", ".join(f"{v} {k}" for k, v in counts.items()))
    print("   id  cat      samples  centroid (x, y, z)        n_z   reachable  potential d p10 / "
          "median / max")
    for i, r in enumerate(rows):
        if r["cat"] not in (0, 1):
            continue
        c = r["centroid"]
        print(f"  {i:3d}  {CATS[r['cat']]:7s} {r['n']:7d}  ({c[0]:7.0f},{c[1]:7.0f},{c[2]:6.0f})  "
              f"{r['normal'][2]:5.2f}  {r['n_ok']:7d}   {r['d_p10']:8,.0f} / {r['d_med']:8,.0f} / "
              f"{r['d_max']:8,.0f}")
    if a.out:
        np.savez_compressed(
            a.out, points=pts.astype(np.float32), normals=nrm.astype(np.float32),
            surf=surf.astype(np.int32), cat=np.array([r["cat"] for r in rows], np.int32),
            n=np.array([r["n"] for r in rows], np.int32),
            centroid=np.array([r["centroid"] for r in rows], np.float32).reshape(-1, 3),
            normal=np.array([r["normal"] for r in rows], np.float32).reshape(-1, 3),
            d_min=np.array([r["d_min"] for r in rows]), d_p10=np.array([r["d_p10"] for r in rows]),
            d_med=np.array([r["d_med"] for r in rows]), d_max=np.array([r["d_max"] for r in rows]),
            map=np.array(bsp.stem), cell=np.float32(a.cell), version=np.int32(2))
        print(f"   -> {a.out}")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        for i, r in enumerate(rows):
            if r["cat"] == 0:
                ax.scatter(pts[r["m"], 0], pts[r["m"], 1], s=1, c="0.85")
        for i, r in enumerate(rows):
            if r["cat"] != 1:
                continue
            p = pts[r["m"]]
            ax.scatter(p[:, 0], p[:, 1], s=2, c="0.2")
            ax.annotate(f"R{i}", r["centroid"][:2], fontsize=8, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))
        ax.set_aspect("equal")
        ax.set_title(f"{bsp.stem}: ramps (dark, labelled R<id>) and floors (light), ramps.py v2")
        ax.set_xlabel("x (u)")
        ax.set_ylabel("y (u)")
        fig.savefig(a.png, dpi=110, bbox_inches="tight")
        print(f"   plot -> {a.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
