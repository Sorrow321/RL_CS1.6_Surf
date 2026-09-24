"""viz_surf_algo.py - how the surf planner's graph and plans are made (--plan-graph ride), on one map.

  panel 1  a vertical cross-section: solid geometry, the cells with a surface within 256 u below them
           ("supported"), the extra cells within one 128 u hop of those, the kill plane
  panel 2  the whole graph from above, the TEST plan (map start -> finish, the shortest route on the
           graph) and a few TRAINING plans (shortest routes to random targets)
  panel 3  the training loop, in words

    python tools/viz_surf_algo.py <map.bsp> <out.png> [slice_y]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import (BFSPlanner, RIDE_SUPPORT_U, RIDE_HOP_U,  # noqa: E402
                              _shift)
from surfgym.zones import load_zones                 # noqa: E402

bsp, out = sys.argv[1:3]
core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=1, lidar_w=0, lidar_h=0))
core.reset(0)
sp = np.asarray(core.states_view["origin"][0], np.float64)
zones = load_zones(bsp)
g = BFSPlanner.for_core(core, 32.0, zones["end"], n_targets=6, seed=3, graph_kind="ride")
core.close()
solid = np.asarray(g.solid, bool)
nz, ny, nx = solid.shape
cell = float(g.cell)
mins = np.asarray(g.mins, np.float64)
zc = mins[2] + (np.arange(nz) + 0.5) * cell
dead = np.broadcast_to((zc <= g.kill_z)[:, None, None], solid.shape)
free_live = (~solid) & ~dead
sup = np.zeros_like(solid)
for s in range(1, int(round(RIDE_SUPPORT_U / cell)) + 1):
    sup |= _shift(solid & ~dead, s, 0)
sup &= free_live
ride = np.asarray(g.node_of) >= 0
hop_only = ride & ~sup

# ---- panel 1: the vertical cross-section at y = slice_y (default: halfway along the course)
yslice = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5 * (sp[1] + float(np.asarray(g.finish_center)[1]))
iy = int(np.clip((yslice - mins[1]) / cell, 0, ny - 1))
img = np.zeros((nz, nx), np.int8)            # 0 air, 1 solid, 2 supported, 3 hop
img[solid[:, iy, :]] = 1
img[sup[:, iy, :]] = 2
img[hop_only[:, iy, :]] = 3
ext = (mins[0], mins[0] + nx * cell, mins[2], mins[2] + nz * cell)

fig = plt.figure(figsize=(20, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.12, hspace=0.25)
ax = fig.add_subplot(gs[0, 0])
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["white", "0.25", "#7fb3d5", "#f5cba7"])
ax.imshow(img, cmap=cmap, vmin=0, vmax=3, origin="lower", extent=ext, interpolation="nearest", aspect="auto")
if np.isfinite(g.kill_z):
    ax.axhline(g.kill_z, color="black", ls="--", lw=1)
    ax.text(ext[0] + 50, g.kill_z + 25, "kill plane (fall = death)", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="0.25", label="solid (ramps, walls)"),
                   Patch(color="#7fb3d5", label=f"GRAPH: a surface within {RIDE_SUPPORT_U:g} u below"),
                   Patch(color="#f5cba7", label=f"GRAPH: within one {RIDE_HOP_U:g} u hop of those")],
          loc="upper right", fontsize=8)
ax.set_xlabel("x (u)"); ax.set_ylabel("z (u)")
ax.set_title(f"1. which 32 u cells are graph nodes - a vertical cut through the map at y = {yslice:,.0f}\n"
             "a surfer rides a surface or crosses a short gap; anything further from a surface is a fall",
             fontsize=10)
zs_used = np.flatnonzero(img.any(axis=1))
if len(zs_used):
    ax.set_ylim(mins[2] + (zs_used.min() - 2) * cell, mins[2] + (zs_used.max() + 3) * cell)

# ---- panel 2: the graph from above with the plans
a2 = fig.add_subplot(gs[:, 1])
xyz = np.asarray(g.xyz)
a2.scatter(xyz[::2, 0], xyz[::2, 1], s=0.3, c="0.75")
test = g.plan(sp, g.fin)
a2.plot(test.raw[:, 0], test.raw[:, 1], "-", color="black", lw=3, label="TEST plan: start -> finish")
styles = ["--", ":", "-."]
k = 0
for t in range(g.n_rand):
    pl = g.plan(sp, t)
    if pl is None:
        continue
    a2.plot(pl.raw[:, 0], pl.raw[:, 1], styles[k % 3], color="0.35", lw=1.8,
            label="TRAINING plan to a random target" if k == 0 else None)
    a2.plot(pl.goal[0], pl.goal[1], "x", color="0.2", ms=9, mew=2)
    k += 1
    if k >= 3:
        break
fmn, fmx = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])
a2.add_patch(plt.Rectangle((fmn[0], fmn[1]), fmx[0] - fmn[0], fmx[1] - fmn[1], fill=False, ec="black", lw=2,
                           ls="--"))
a2.text(fmn[0], fmx[1] + 50, "FINISH", fontsize=10, weight="bold")
a2.plot(sp[0], sp[1], "s", color="black", ms=10)
a2.text(sp[0] + 60, sp[1] - 40, "START", fontsize=10, weight="bold")
a2.axhline(yslice, color="0.5", lw=1, ls=(0, (1, 3)))
a2.text(xyz[:, 0].min(), yslice + 40, "cut of panel 1", fontsize=8, color="0.4")
a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])
a2.legend(loc="lower left", fontsize=8)
a2.set_title(f"2. the graph from above ({g.n_nodes:,} cells) and its plans: shortest routes\n"
             "(Dijkstra, one cell to the next, one level up or down per step, no corner cutting)", fontsize=10)

# ---- panel 3: the loop
a3 = fig.add_subplot(gs[1, 0]); a3.axis("off")
a3.text(0.0, 1.0,
        "3. how the EXECUTOR is trained on these plans (stage 1, the same recipe as the maze):\n\n"
        "   every episode:  pick a target - 80% a random graph cell 256-4,096 u of route away,\n"
        "                   20% the finish box\n"
        "                -> plan = the shortest route to it on the graph (simplified, a point every 128 u)\n"
        "                -> the executor sees the plan as the FAN (+ the drawn plan under fanline)\n"
        "                -> reward = progress along the plan (arc length) + a bonus on reaching the target\n"
        "   90% of episodes start from places the executor reached before (its own reservoir).\n\n"
        "   TEST: from the map start, the plan is the shortest route to the finish - 'finished' means the\n"
        "   executor surfed it to the end. The jump planner replaces that one long plan with 750-1,500 u\n"
        "   jumps chosen by its search (the maze's algorithm, run on this graph).",
        va="top", family="monospace", fontsize=9.5, transform=a3.transAxes)
fig.suptitle(f"The surf planner on {Path(bsp).stem}: a graph of where a surfer can be, shortest routes on it, "
             "an executor trained to follow them", fontsize=13)
fig.savefig(out, dpi=85, bbox_inches="tight")
print(out, "ride", int(ride.sum()), "supported", int(sup.sum()), "hop-only", int(hop_only.sum()))
