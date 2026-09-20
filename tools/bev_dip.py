#!/usr/bin/env python3
"""bev_dip.py - one figure per map: the goal potential from above, the route a
surfer has to take, and what the policy actually did.

Top row, per map: a bird's-eye heat map of the geodesic potential d (the
thing the shaping reward descends), the map's geometry in grey, the
ride-shell route from :mod:`dip_probe` in green (the route the physics
allows), and each recorded trajectory on top - so "the field points across
the pit and the policy follows it" is visible rather than argued.

Bottom row: d against distance travelled, for the same route and the same
trajectories. The DIP is the rise of the green curve above its own running
minimum, and a policy that dies in the pit shows as a curve that just
stops.

    python tools/bev_dip.py --maps maps_pool/surf_edgeflow_blue*.bsp \\
        --traj blue025=runs/efRAT_blue025/traj_0907018240.jsonl \\
        --out runs/research/gate_bench/edgeflow_bev.png

Measurement only (CLAUDE.md 0b): nothing here enters a reward or a spawn
rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

import dip_probe as dp                                   # noqa: E402
from surfgym.core import SurfCore, SurfEnvConfig         # noqa: E402
from surfgym.goalfield import build_goal_field, goal_occupancy   # noqa: E402
from surfgym.zones import load_zones                     # noqa: E402


def episodes(path: Path):
    eps, cur = [], None
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        if isinstance(o, dict):
            if cur:
                eps.append(np.asarray(cur, float))
            cur = []
        else:
            cur.append(o[:7])
    if cur:
        eps.append(np.asarray(cur, float))
    return [e for e in eps if len(e) > 20]


def longest(path: Path):
    eps = episodes(path)
    return max(eps, key=len) if eps else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--traj", nargs="*", default=[],
                    help="<map-tag>=<traj.jsonl>[:<label>] - the episode that got furthest is drawn")
    ap.add_argument("--goal-cell", type=float, default=32.0)
    ap.add_argument("--hop", type=float, default=96.0)
    ap.add_argument("--ride", type=float, default=256.0)
    ap.add_argument("--zslab", type=float, nargs=2, default=None,
                    help="height band the BEV takes its minimum over (default: above the fall net)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trajs: dict[str, list] = {}
    for spec in a.traj:
        tag, _, rest = spec.partition("=")
        p, _, label = rest.partition(":")
        trajs.setdefault(tag, []).append((Path(p), label or Path(p).parent.name))

    n = len(a.maps)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 11.0),
                             gridspec_kw={"height_ratios": [2.0, 1.0]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, bsp in enumerate(a.maps):
        bsp = str(bsp)
        stem = Path(bsp).stem
        tag = stem.split("_")[-1]
        zones = load_zones(bsp)
        core = SurfCore(bsp, SurfEnvConfig(num_envs=1))
        gf = build_goal_field(core, zones["end"], a.goal_cell, device="cpu")
        occ, omins = goal_occupancy(core, a.goal_cell, None)
        occ = np.asarray(occ, bool)
        d = np.asarray(gf.grid, np.float64)
        cell = float(a.goal_cell)
        zs = omins[2] + (np.arange(d.shape[0]) + 0.5) * cell
        ys = omins[1] + (np.arange(d.shape[1]) + 0.5) * cell
        xs = omins[0] + (np.arange(d.shape[2]) + 0.5) * cell
        kz = dp.kill_ceiling(bsp)
        lo, hi = (a.zslab if a.zslab else (kz + cell, zs.max() - 4 * cell))
        band = (zs > lo) & (zs < hi)

        # the potential a column offers, and the geometry in it
        dd = np.where(np.isfinite(d) & (d < 1e8), d, np.nan)[band]
        with np.errstate(invalid="ignore"):
            dmin = np.nanmin(dd, axis=0)
        solid = occ[band].any(axis=0)

        ax = axes[0, col]
        ext = [xs[0], xs[-1], ys[0], ys[-1]]
        im = ax.imshow(dmin, origin="lower", extent=ext, aspect="equal",
                       cmap="viridis_r", interpolation="nearest")
        ax.imshow(np.where(solid, 1.0, np.nan), origin="lower", extent=ext,
                  aspect="equal", cmap="Greys", vmin=0, vmax=1.6,
                  interpolation="nearest", alpha=0.55)
        plt.colorbar(im, ax=ax, fraction=0.035, label="geodesic d to finish (u)")

        # the ride-shell route
        dead = np.zeros(occ.shape, bool)
        dead[zs <= kz, :, :] = True
        free = ~occ
        above = np.zeros_like(free)
        above[zs > kz, :, :] = True
        sup = dp.supported(occ, max(1, int(round(a.ride / cell))), dead)
        shell = free & above & np.isfinite(d) & dp.hop(sup, free & above,
                                                       max(1, int(round(a.hop / cell))))
        gmin, gmax = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])

        def vox(p):
            f = (np.asarray(p, float) - np.asarray(omins, float)) / cell
            i = np.floor(f).astype(int)
            return (int(np.clip(i[2], 0, d.shape[0] - 1)),
                    int(np.clip(i[1], 0, d.shape[1] - 1)),
                    int(np.clip(i[0], 0, d.shape[2] - 1)))

        glo, ghi = vox(gmin), vox(gmax)
        gm = np.zeros_like(free)
        gm[glo[0]:ghi[0] + 1, glo[1]:ghi[1] + 1, glo[2]:ghi[2] + 1] = True
        gm &= free
        S = dp.shell_field(shell, [tuple(v) for v in np.argwhere(gm)])
        sp = dp.spawn_points(bsp)
        starts = [vox(q) for q in sp]
        live = [s for s in starts if shell[s] and np.isfinite(S[s])]
        route_xy, route_d, dip = None, None, None
        if live:
            path = dp.trace(S, min(live, key=lambda v: S[v]))
            route_xy = np.array([[omins[0] + (v[2] + .5) * cell,
                                  omins[1] + (v[1] + .5) * cell] for v in path])
            route_d = np.array([float(d[v]) for v in path])
            dip = dp.dip_of(d, path)[0]
            ax.plot(route_xy[:, 0], route_xy[:, 1], "-", color="#22cc55", lw=3.0,
                    label=f"route the physics allows (dip {dip:,.0f}u)")

        ax.add_patch(plt.Rectangle((gmin[0], gmin[1]), gmax[0] - gmin[0], gmax[1] - gmin[1],
                                   fill=False, ec="#ff2d95", lw=2.2, label="finish"))
        if len(sp):
            ax.plot(sp[:, 0], sp[:, 1], "w*", ms=11, mec="k", label="spawns")

        prof = []
        for k, (tp, label) in enumerate(trajs.get(tag, [])):
            if not tp.exists():
                continue
            # the episode that got FURTHEST (lowest geodesic d), not the
            # longest: a finisher's longest episode is one of its failures
            eps_ = episodes(tp)
            if not eps_:
                continue
            ep = min(eps_, key=lambda e: (float(np.min(gf.sample(e[:, 1:4]))), len(e)))
            p = ep[:, 1:4]
            col_ = ["#e8412f", "#2f6be8", "#f0a30a"][k % 3]
            ax.plot(p[:, 0], p[:, 1], "-", color=col_, lw=2.0, label=label)
            ax.plot(p[-1, 0], p[-1, 1], "x", color=col_, ms=11, mew=3)
            dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
            prof.append((dist, gf.sample(p), col_, label, len(ep) / 100.0))

        ax.set_title(f"{stem}\nd0 {float(np.median(gf.sample(sp))):,.0f}u" +
                     (f"  |  dip {dip:,.0f}u = {100 * dip / float(np.median(gf.sample(sp))):.1f}% of d0"
                      if dip is not None else ""), fontsize=11)
        ax.set_xlabel("x (u)")
        if col == 0:
            ax.set_ylabel("y (u)")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

        bx = axes[1, col]
        if route_d is not None:
            rd = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route_xy, axis=0), axis=1))]
            bx.plot(rd, route_d, "-", color="#22cc55", lw=2.5, label="route the physics allows")
            run = np.minimum.accumulate(route_d)
            bx.fill_between(rd, run, route_d, where=route_d > run, color="#22cc55", alpha=0.25,
                            label="the give-back (the dip)")
        for dist, dv, col_, label, secs in prof:
            bx.plot(dist, dv, "-", color=col_, lw=2.0, label=f"{label} ({secs:.1f}s)")
            bx.plot(dist[-1], dv[-1], "x", color=col_, ms=10, mew=3)
        bx.set_xlabel("distance travelled (u)")
        if col == 0:
            bx.set_ylabel("geodesic d to finish (u)")
        bx.grid(alpha=0.3)
        bx.legend(fontsize=8)
        bx.invert_yaxis()

    fig.suptitle("edgeflow: the potential points straight across the pit; the route has to go left",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
