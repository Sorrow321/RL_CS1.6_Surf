"""viz_planner_bfs.py - the DETERMINISTIC (BFS) planner + executor on a walking map, top-down.

The stage-1 planner (--goal-planner bfs) plans ONCE per episode: the whole shortest path on the
walkable graph from the spawn to the finish, stored in each recorded episode's header ("line").
The executor never sees the whole plan - it reads the FAN: 8 points of the plan ahead of its
projection, at the run's --goal-fan-offsets (seconds) x max(horizontal speed, 500 u/s). This
draws the whole plan (faded), the part the executor sees right now (bold orange, the 8 fan
points as dots), and the agent's trail.
    python tools/viz_planner_bfs.py <traj.jsonl> <run.json> <map.bsp> <out_prefix> [episode] [title]
Writes <out_prefix>.mp4 and <out_prefix>.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import BFSPlanner              # noqa: E402
from surfgym.zones import load_zones                 # noqa: E402

traj_p, runjson, bsp, outp = sys.argv[1:5]
EP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
TITLE = sys.argv[6] if len(sys.argv) > 6 else Path(bsp).stem
SPEED_FLOOR = 500.0                                   # the fan's span floor (surfgym.goals)

cfg = json.load(open(runjson, encoding="utf-8"))["config"]
offs = np.asarray(cfg.get("goal_fan_offsets") or [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0],
                  np.float64)

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
vxy = rows[:, 4:6]
yaw = rows[:, 7]
hdr = hdrs[EP]
tick_ms = float(hdr.get("tick_ms", 10.0))
plan = np.asarray(hdr["line"], np.float64)[:, :2]

# densify the plan to 16 u so the projection and the fan points are smooth
seg = np.linalg.norm(np.diff(plan, axis=0), axis=1)
arc = np.r_[0.0, np.cumsum(seg)]
L = float(arc[-1])
s_d = np.arange(0.0, L + 1e-6, 16.0)
dense = np.c_[np.interp(s_d, arc, plan[:, 0]), np.interp(s_d, arc, plan[:, 1])]


def at(s):
    s = np.clip(s, 0.0, L)
    return np.c_[np.interp(s, arc, plan[:, 0]), np.interp(s, arc, plan[:, 1])]


# a MONOTONE-windowed projection: the nearest plan point within [-200, +800] u of the last one
proj = np.zeros(len(T))
s_prev = 0.0
for i in range(len(T)):
    lo, hi = np.searchsorted(s_d, s_prev - 200.0), np.searchsorted(s_d, s_prev + 800.0)
    lo, hi = max(0, lo), max(lo + 1, min(len(s_d), hi))
    j = lo + int(np.argmin(np.linalg.norm(dense[lo:hi] - xy[i], axis=1)))
    s_prev = float(s_d[j]); proj[i] = s_prev

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
floor[np.clip(((nodes[:, 1] - y0) / cell).astype(int), 0, ny - 1),
      np.clip(((nodes[:, 0] - x0) / cell).astype(int), 0, nx - 1)] = True
fmin, fmax = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])
secs = (T[-1] - T[0] + 1) * tick_ms / 1000.0
_m = 48.0
won = ("FINISHED" if (fmin[0] - _m <= xy[-1, 0] <= fmax[0] + _m and fmin[1] - _m <= xy[-1, 1] <= fmax[1] + _m)
       else f"not finished ({trailer.get('end')})")


def base(ax):
    ax.imshow(np.where(floor, 0.93, 0.25), cmap="gray", vmin=0, vmax=1, origin="lower",
              extent=(x0, x0 + nx * cell, y0, y0 + ny * cell), interpolation="nearest")
    ax.add_patch(plt.Rectangle((fmin[0], fmin[1]), fmax[0] - fmin[0], fmax[1] - fmin[1],
                               fc=(0.2, 0.8, 0.3, 0.35), ec="green", lw=2))
    ax.text(0.5 * (fmin[0] + fmax[0]), fmax[1] + 60, "FINISH", color="green", ha="center",
            fontsize=10, weight="bold")
    ax.plot(plan[:, 0], plan[:, 1], "--", color="royalblue", lw=2, alpha=0.45,
            label=f"BFS plan ({L:,.0f} u, computed once)")
    ax.plot(xy[0, 0], xy[0, 1], "o", color="royalblue", ms=9)
    ax.text(xy[0, 0] + 60, xy[0, 1] - 40, "start", color="royalblue", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


fig, ax = plt.subplots(figsize=(8, 8))
base(ax)
ax.plot(xy[:, 0], xy[:, 1], "-", color="black", lw=1.5, alpha=0.85, label="agent path")
ax.legend(loc="lower left", fontsize=8)
ax.set_title(f"{TITLE}\nepisode {EP}: ONE planner call (the whole path), {secs:.1f} s, {won}",
             fontsize=11)
fig.tight_layout(); fig.savefig(outp + ".png", dpi=110); plt.close(fig)

STEP = 8
frames = list(range(0, len(T), STEP)) + [len(T) - 1]
fig, ax = plt.subplots(figsize=(7.5, 7.5))
base(ax)
ax.legend(loc="lower left", fontsize=8)
trail, = ax.plot([], [], "-", color="black", lw=1.5, alpha=0.85)
dot, = ax.plot([], [], "o", color="red", ms=9, zorder=6)
win, = ax.plot([], [], "-", color="orange", lw=4, alpha=0.95, zorder=4)
fan, = ax.plot([], [], "o", color="darkorange", ms=6, zorder=5, mec="black", mew=0.6)
anc, = ax.plot([], [], "s", color="orange", ms=6, zorder=5)
arrow = [None]
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
                        color="red", zorder=7)
    spd = float(np.hypot(*vxy[i]))
    span = max(spd, SPEED_FLOOR) * offs
    s0 = proj[i]
    ss = np.linspace(s0, min(L, s0 + span[-1]), 40)
    w = at(ss)
    win.set_data(w[:, 0], w[:, 1])
    fp = at(s0 + span)
    fan.set_data(fp[:, 0], fp[:, 1])
    p0 = at(np.array([s0]))
    anc.set_data(p0[:, 0], p0[:, 1])
    title.set_text(f"{TITLE}\nt = {rt * tick_ms / 1000:5.1f} s   speed {spd:3.0f} u/s   "
                   f"the executor sees the next {min(span[-1], L - s0):,.0f} u (8 fan points)")
    return trail, dot, win, fan, anc, title


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=False)
anim.save(outp + ".mp4", writer=animation.FFMpegWriter(fps=25, bitrate=1800), dpi=100)
plt.close(fig)
print(f"{outp}.png / .mp4: episode {EP}, plan {L:,.0f} u, {secs:.1f} s, {won}, {len(frames)} frames")
