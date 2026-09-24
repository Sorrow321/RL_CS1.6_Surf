"""viz_jump_probes.py - HOW the jump planner's proposals are made on a walking map (one picture):

  panel 1  the walkable graph: 32 u floor cells (free, with floor below); an edge joins two
           neighbouring floor cells, never through a wall and never cutting a wall's corner
  panel 2  Dijkstra on that graph from the agent's cell, stopped at 750 u (~3 s of walking):
           every floor cell reachable in 3 s, coloured by walking distance - it flows along the
           corridors and never through a wall, because the graph has no edge there
  panel 3  the 8 probes: for each compass direction, the reachable cell that gets furthest in
           that direction, and the shortest path to it; probes whose ends are within 200 u of
           each other ALONG THE FLOOR merge into one option
    python tools/viz_jump_probes.py <map.bsp> <out.png> [x y]   (x y: where the agent stands)
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import BFSPlanner              # noqa: E402
from surfgym.zones import load_zones                 # noqa: E402

bsp, out = sys.argv[1:3]
at = np.array([float(sys.argv[3]), float(sys.argv[4]), 40.0]) if len(sys.argv) > 4 else None
JUMP_U, MIN_ADV, MERGE_U = 750.0, 225.0, 200.0
DIRS = [np.array([np.cos(a), np.sin(a)]) for a in np.radians(np.arange(0, 360, 45))]
NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=1, lidar_w=0, lidar_h=0))
core.reset(0)
zones = load_zones(bsp)
g = BFSPlanner.for_core(core, 32.0, zones["end"], n_targets=0, seed=0)
core.close()
XY = np.asarray(g.xyz, np.float64)[:, :2]
n = len(XY)
nbr = np.asarray(g.nbr)
rows, cols = np.repeat(np.arange(n), nbr.shape[1]), nbr.reshape(-1)
wts = np.tile(np.asarray(g.wk, np.float64), n)
m = cols >= 0
A = sp.csr_matrix((wts[m], (rows[m], cols[m])), shape=(n, n))
s = int(g.snap((at if at is not None else np.array([-32.0, 32.0, 40.0]))[None, :])[0])

d, pred = dijkstra(A, directed=True, indices=s, limit=JUMP_U, return_predecessors=True)
idx = np.flatnonzero(np.isfinite(d))


def path(t):
    p = [t]
    while p[-1] != s:
        p.append(int(pred[p[-1]]))
    return p[::-1]


probes = []
for k, u in enumerate(DIRS):
    pr = (XY[idx] - XY[s]) @ u
    j = int(np.argmax(pr - 0.02 * d[idx]))
    probes.append(None if pr[j] < MIN_ADV else (k, int(idx[j])))
opts = []
for pb in probes:
    if pb is None:
        continue
    dd, _ = dijkstra(A, directed=True, indices=pb[1], limit=MERGE_U, return_predecessors=True)
    for o in opts:
        if np.isfinite(dd[o["node"]]):
            o["dirs"].append(pb[0])
            break
    else:
        opts.append({"node": pb[1], "dirs": [pb[0]]})

# ---- drawing, zoomed on the ball
cell = 32.0
x0, y0 = XY[:, 0].min() - 3 * cell, XY[:, 1].min() - 3 * cell
nx = int(np.ceil((XY[:, 0].max() + 3 * cell - x0) / cell))
ny = int(np.ceil((XY[:, 1].max() + 3 * cell - y0) / cell))
gi = np.clip(((XY[:, 1] - y0) / cell).astype(int), 0, ny - 1)
gj = np.clip(((XY[:, 0] - x0) / cell).astype(int), 0, nx - 1)
floor = np.zeros((ny, nx), bool)
floor[gi, gj] = True
dist_img = np.full((ny, nx), np.nan)
dist_img[gi[idx], gj[idx]] = d[idx]
ext = (x0, x0 + nx * cell, y0, y0 + ny * cell)
pad = JUMP_U + 250
lim = (XY[s, 0] - pad, XY[s, 0] + pad, XY[s, 1] - pad, XY[s, 1] + pad)
col = plt.get_cmap("tab10")

fig, axs = plt.subplots(1, 3, figsize=(19, 7.2))
for ax in axs:
    ax.imshow(np.where(floor, 0.93, 0.25), cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext,
              interpolation="nearest")
    ax.plot(*XY[s], "o", color="red", ms=11, zorder=8)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
# panel 1: the graph's nodes and edges near the agent
near = np.flatnonzero(np.linalg.norm(XY - XY[s], axis=1) < pad * 1.2)
for v in near[::1]:
    for w in nbr[v]:
        if w > v and w in set(near.tolist()):
            axs[0].plot(*zip(XY[v], XY[w]), "-", color="steelblue", lw=0.25, alpha=0.6)
axs[0].plot(XY[near, 0], XY[near, 1], ".", color="steelblue", ms=2)
axs[0].set_title("1. the walkable graph (built once from the map)\n"
                 "node = 32 u floor cell; edge = step to a neighbour cell\nno edge through a wall -> "
                 "walls are simply missing links", fontsize=10)
# panel 2: the reachable ball
im = axs[1].imshow(np.ma.masked_invalid(dist_img), cmap="viridis", origin="lower", extent=ext,
                   interpolation="nearest", vmin=0, vmax=JUMP_U)
fig.colorbar(im, ax=axs[1], fraction=0.04, label="walking distance from the agent (u)")
axs[1].set_title(f"2. Dijkstra on the graph from the agent, stopped at {JUMP_U:.0f} u (~3 s)\n"
                 f"= every floor cell reachable in one jump ({len(idx):,} cells)\n"
                 "it flows along corridors, never through walls", fontsize=10)
# panel 3: probes and options
axs[2].imshow(np.ma.masked_invalid(dist_img), cmap="Greys", origin="lower", extent=ext,
              interpolation="nearest", vmin=0, vmax=JUMP_U * 3, alpha=0.5)
for pb in probes:
    if pb is None:
        continue
    k, t = pb
    oi = next(i for i, o in enumerate(opts) if k in o["dirs"])
    p = XY[path(t)]
    axs[2].plot(p[:, 0], p[:, 1], "-", color=col(oi), lw=2.5)
    axs[2].annotate("", XY[s] + 260 * DIRS[k], XY[s], arrowprops=dict(arrowstyle="->", color="0.3",
                                                                   lw=1))
    axs[2].text(*(XY[s] + 300 * DIRS[k]), NAMES[k], ha="center", va="center", fontsize=9, color="0.2")
for k, pb in enumerate(probes):
    if pb is None:
        axs[2].text(*(XY[s] + 300 * DIRS[k]), NAMES[k] + "\n(blocked)", ha="center", va="center",
                    fontsize=8, color="crimson")
for i, o in enumerate(opts):
    axs[2].plot(*XY[o["node"]], "o", color=col(i), ms=14, mec="black", zorder=9)
    axs[2].text(*(XY[o["node"]] + 45), f"option {i + 1}: {'+'.join(NAMES[k] for k in o['dirs'])}",
                color=col(i), fontsize=9, weight="bold", zorder=10)
axs[2].set_title("3. probe in direction d = the reachable cell furthest along d,\n"
                 "reached by its shortest path; ends within 200 u along the floor merge\n"
                 f"-> {sum(p is not None for p in probes)} probes, {len(opts)} options", fontsize=10)
fig.suptitle(f"How the proposals are made on a walking map ({Path(bsp).stem}): pure geometry, no learning",
             fontsize=13)
fig.tight_layout()
fig.subplots_adjust(top=0.84)
fig.savefig(out, dpi=95)
print(f"{out}: {len(idx)} cells in the ball, {sum(p is not None for p in probes)} probes, {len(opts)} options")
