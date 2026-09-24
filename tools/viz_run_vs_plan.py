"""viz_run_vs_plan.py - what greedy episodes did, against a planner route (one picture).

  panel 1  top view: the surf graph's cells (light), the planned route (thick, marked every
           1,000 u), the finish box, every episode's path (line styles), its end (x = died,
           star = finished)
  panel 2  progress ALONG the route over time - the --race-arc rule (tools/plan_route.py file,
           surfgym.route.ArcProgress: corridor 1,500 u, window 16), i.e. what the arc reward pays
  panel 3  height over time

    python tools/viz_run_vs_plan.py <map.bsp> <route.npz> <out.png> <label>=<traj.jsonl> [...]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import BFSPlanner              # noqa: E402
from surfgym.route import ArcProgress, episodes_from_traj   # noqa: E402
from surfgym.tick import episode_seconds             # noqa: E402,F401
from surfgym.zones import load_zones                 # noqa: E402

bsp, route, out = sys.argv[1:4]
runs = [a.split("=", 1) for a in sys.argv[4:]]
zones = load_zones(bsp)
core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=1, lidar_w=0, lidar_h=0))
core.reset(0)
g = BFSPlanner.for_core(core, 32.0, zones["end"], n_targets=0, seed=0, graph_kind="ride")
core.close()
xyz = np.asarray(g.xyz)
line = np.asarray(np.load(route)["route"], np.float64)
arc_len = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())

fig, axs = plt.subplots(len(runs), 3, figsize=(21, 6.6 * len(runs)), squeeze=False,
                        gridspec_kw={"width_ratios": [1.25, 1, 1]})
styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2)),
          (0, (2, 2, 6, 2))]
for r, (label, traj) in enumerate(runs):
    eps, hdrs = episodes_from_traj(traj, with_headers=True)
    a1, a2, a3 = axs[r]
    a1.scatter(xyz[::3, 0], xyz[::3, 1], s=0.2, c="0.8")
    a1.plot(line[:, 0], line[:, 1], "-", color="black", lw=4, alpha=0.35, label="planned route")
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
    for km in range(1000, int(s[-1]), 1000):
        p = np.array([np.interp(km, s, line[:, k]) for k in range(2)])
        a1.plot(*p, "o", color="black", ms=4)
        a1.text(p[0] + 40, p[1] + 40, f"{km // 1000}k", fontsize=8)
    fm, fx = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])
    a1.add_patch(plt.Rectangle((fm[0], fm[1]), fx[0] - fm[0], fx[1] - fm[1], fill=False,
                               ec="black", lw=2, ls="--"))
    a1.text(fm[0], fx[1] + 60, "FINISH", fontsize=10, weight="bold")
    n_fin = 0
    for k, e in enumerate(eps):
        e = np.asarray(e, np.float64)
        if len(e) < 2:
            continue
        p = e[:, 1:4]
        fin = bool(hdrs[k].get("_footer", {}).get("finished", False)) if hdrs else False
        inbox = bool(np.all((p[-1] >= fm - 64) & (p[-1] <= fx + 64)))
        fin = fin or inbox
        n_fin += fin
        ls = styles[k % len(styles)]
        a1.plot(p[:, 0], p[:, 1], ls=ls, color="0.15", lw=1.3,
                label=f"episode {k}" if k < 4 else None)
        a1.plot(*p[-1, :2], "*" if fin else "x", color="black", ms=12 if fin else 9, mew=2)
        ap = ArcProgress(line, spacing=float(np.load(route)["spacing"]))
        ap.reset(p[:1])
        arc = [float(ap.arc[0])]
        for q in p[1:]:
            ap.advance(q[None, :])
            arc.append(float(ap.arc[0]))
        tick_ms = float(hdrs[k].get("tick_ms", 10.0)) if hdrs else 10.0
        t = (e[:, 0] - e[0, 0]) * tick_ms / 1000.0
        a2.plot(t, arc, ls=ls, color="0.15", lw=1.3)
        a2.plot(t[-1], arc[-1], "*" if fin else "x", color="black", ms=11 if fin else 8, mew=2)
        a3.plot(t, p[:, 2], ls=ls, color="0.15", lw=1.3)
        a3.plot(t[-1], p[-1, 2], "*" if fin else "x", color="black", ms=11 if fin else 8, mew=2)
    a1.plot(line[0, 0], line[0, 1], "s", color="black", ms=9)
    a1.text(line[0, 0] + 60, line[0, 1] - 90, "START", fontsize=10, weight="bold")
    a1.set_aspect("equal"); a1.set_xticks([]); a1.set_yticks([])
    a1.legend(loc="lower left", fontsize=8)
    a1.set_title(f"{label}: top view, {len(eps)} greedy episodes, {n_fin} finished\n"
                 "x = episode ended without finishing, star = finished", fontsize=10)
    a2.axhline(arc_len, color="black", ls="--", lw=1)
    a2.text(0.2, arc_len * 0.97, f"end of the route ({arc_len:,.0f} u)", fontsize=8, va="top")
    a2.set_xlabel("seconds"); a2.set_ylabel("progress along the planned route (u)")
    a2.set_ylim(-200, arc_len * 1.05)
    a2.set_title("progress along the route (what the arc reward pays)", fontsize=10)
    if np.isfinite(g.kill_z):
        a3.axhline(g.kill_z, color="black", ls="--", lw=1)
        a3.text(0.2, g.kill_z + 10, "kill plane", fontsize=8)
    a3.set_xlabel("seconds"); a3.set_ylabel("z (u)")
    a3.set_title("height", fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=80)
print(out)
