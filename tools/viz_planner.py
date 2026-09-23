"""viz_planner.py - top-down visualisation of the learned planner + executor on a labyrinth.

Inputs: a recording (tools/record_ckpt.py JSONL) and its --dump-plans JSON (every planner call:
episode, tick, shape, anchor, polyline). Draws the walkable floor (the planner's own graph), the
finish box, the agent's trail, the CURRENT plan (bold) and the earlier plans (faded).
    python viz_planner.py <traj.jsonl> <plans.json> <map.bsp> <out_prefix> [episode] [title]
Writes <out_prefix>.mp4 (animation) and <out_prefix>.png (whole episode, every plan numbered).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                      # noqa: E402

sys.path.insert(0, r"C:/RL_Surf/python")
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import BFSPlanner              # noqa: E402
from surfgym.zones import load_zones                 # noqa: E402

traj_p, plans_p, bsp, outp = sys.argv[1:5]
EP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
TITLE = sys.argv[6] if len(sys.argv) > 6 else Path(bsp).stem

# ---- episode rows
eps, cur, hdrs = [], None, []
for line in open(traj_p, encoding="utf-8"):
    if line.startswith("["):
        cur.append(json.loads(line))
    else:
        r = json.loads(line)
        if "end" in r:
            eps.append((cur, r)); cur = None
        else:
            hdrs.append(r); cur = []
rows, trailer = eps[EP]
rows = np.asarray(rows, np.float64)
T = rows[:, 0].astype(int)
xy = rows[:, 1:3]
yaw = rows[:, 7]
tick_ms = float(hdrs[EP].get("tick_ms", 10.0))

# ---- plans of this episode, ticks relative to the episode start
dump = json.load(open(plans_p, encoding="utf-8"))
pl = [p for p in dump["plans"] if int(p["ep"]) == EP]
t0 = min(int(p["tick"]) for p in pl)
for p in pl:
    p["rt"] = int(p["tick"]) - t0
pl.sort(key=lambda p: p["rt"])

# ---- the floor: the planner's walkable graph, flattened to x/y
core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
zones = load_zones(bsp)
g = BFSPlanner.for_core(core, 32.0, zones["end"], n_targets=1, seed=0)
core.close()
nodes = np.asarray(g.xyz, np.float64)
cell = 32.0
x0, y0 = nodes[:, 0].min() - 3 * cell, nodes[:, 1].min() - 3 * cell
x1, y1 = nodes[:, 0].max() + 3 * cell, nodes[:, 1].max() + 3 * cell
nx, ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
floor = np.zeros((ny, nx), bool)
ix = ((nodes[:, 0] - x0) / cell).astype(int)
iy = ((nodes[:, 1] - y0) / cell).astype(int)
floor[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)] = True
fmin, fmax = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])


def base(ax):
    ax.imshow(np.where(floor, 0.93, 0.25), cmap="gray", vmin=0, vmax=1, origin="lower",
              extent=(x0, x0 + nx * cell, y0, y0 + ny * cell), interpolation="nearest")
    ax.add_patch(plt.Rectangle((fmin[0], fmin[1]), fmax[0] - fmin[0], fmax[1] - fmin[1],
                               fc=(0.2, 0.8, 0.3, 0.35), ec="green", lw=2))
    ax.text(0.5 * (fmin[0] + fmax[0]), fmax[1] + 60, "FINISH", color="green", ha="center",
            fontsize=10, weight="bold")
    ax.plot(xy[0, 0], xy[0, 1], "o", color="royalblue", ms=9)
    ax.text(xy[0, 0] + 60, xy[0, 1] - 40, "start", color="royalblue", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


secs = (T[-1] - T[0] + 1) * tick_ms / 1000.0
_m = 48.0
won = ("FINISHED" if (fmin[0] - _m <= xy[-1, 0] <= fmax[0] + _m and fmin[1] - _m <= xy[-1, 1] <= fmax[1] + _m)
       else f"not finished ({trailer.get('end')})")
cmap = plt.get_cmap("plasma")

# ---- static overview: the whole path and every plan, numbered
fig, ax = plt.subplots(figsize=(8, 8))
base(ax)
ax.plot(xy[:, 0], xy[:, 1], "-", color="black", lw=1.5, alpha=0.8, label="agent path")
for i, p in enumerate(pl):
    ln = np.asarray(p["line"])[:, :2]
    c = cmap(i / max(1, len(pl) - 1))
    ax.plot(ln[:, 0], ln[:, 1], "-", color=c, lw=3, alpha=0.85)
    ax.plot(ln[0, 0], ln[0, 1], "o", color=c, ms=5)
    ax.text(ln[0, 0] + 25, ln[0, 1] + 25, str(i + 1), color=c, fontsize=9, weight="bold")
ax.set_title(f"{TITLE}\nepisode {EP}: {len(pl)} planner calls, {secs:.1f} s, {won}", fontsize=11)
fig.tight_layout(); fig.savefig(outp + ".png", dpi=110); plt.close(fig)

# ---- animation
STEP = 8                                    # 80 ms of game per frame
frames = list(range(0, len(T), STEP)) + [len(T) - 1]
fig, ax = plt.subplots(figsize=(7.5, 7.5))
base(ax)
trail, = ax.plot([], [], "-", color="black", lw=1.5, alpha=0.8)
dot, = ax.plot([], [], "o", color="red", ms=9, zorder=5)
arrow = [None]
past = []
curline, = ax.plot([], [], "-", color="orange", lw=4, alpha=0.95, zorder=4)
curpts, = ax.plot([], [], "o", color="orange", ms=4, zorder=4)
title = ax.set_title("")


def draw(fi):
    i = frames[fi]
    rt = T[i] - T[0]
    trail.set_data(xy[: i + 1, 0], xy[: i + 1, 1])
    dot.set_data([xy[i, 0]], [xy[i, 1]])
    if arrow[0] is not None:
        arrow[0].remove()
    a = np.deg2rad(yaw[i])
    arrow[0] = ax.arrow(xy[i, 0], xy[i, 1], 140 * np.cos(a), 140 * np.sin(a), width=18,
                        color="red", zorder=6)
    active = [p for p in pl if p["rt"] <= rt]
    for ln_ in past:
        ln_.remove()
    past.clear()
    for p in active[:-1]:
        ln = np.asarray(p["line"])[:, :2]
        past.append(ax.plot(ln[:, 0], ln[:, 1], "-", color="purple", lw=2, alpha=0.25)[0])
    if active:
        ln = np.asarray(active[-1]["line"])[:, :2]
        curline.set_data(ln[:, 0], ln[:, 1]); curpts.set_data(ln[:, 0], ln[:, 1])
    title.set_text(f"{TITLE}\nt = {rt * tick_ms / 1000:5.1f} s   plan #{len(active)} of {len(pl)}"
                   f" (shape {active[-1]['shape'] if active else '-'})")
    return trail, dot, curline, curpts, title


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=False)
anim.save(outp + ".mp4", writer=animation.FFMpegWriter(fps=25, bitrate=1800), dpi=100)
plt.close(fig)
print(f"{outp}.png / .mp4: episode {EP}, {len(pl)} plans, {secs:.1f} s, {len(frames)} frames")
