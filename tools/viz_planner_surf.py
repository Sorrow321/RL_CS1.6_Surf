"""viz_planner_surf.py - planner + executor visualisation for SURF maps (3-D).

Like tools/viz_planner.py, but surf needs height: the left panel is a TOP-DOWN height map (the
highest solid above the kill plane per 32 u column; black = void, nothing to land on), the right
panel is a SIDE view along the map's long axis (altitude vs position) with the kill plane and the
finish box. Every planner call is drawn in both panels: the current plan bold orange, earlier
plans faded purple; the agent's trail black.
    python tools/viz_planner_surf.py <traj.jsonl> <plans.json> <map.bsp> <out_prefix> [episode] [title]
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

traj_p, plans_p, bsp, outp = sys.argv[1:5]
EP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
TITLE = sys.argv[6] if len(sys.argv) > 6 else Path(bsp).stem

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
P = rows[:, 1:4]
yaw = rows[:, 7]
tick_ms = float(hdrs[EP].get("tick_ms", 10.0))

dump = json.load(open(plans_p, encoding="utf-8"))
pl = [p for p in dump["plans"] if int(p["ep"]) == EP]
t0 = min(int(p["tick"]) for p in pl) if pl else 0
for p in pl:
    p["rt"] = int(p["tick"]) - t0
pl.sort(key=lambda p: p["rt"])

core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
zones = load_zones(bsp)
g = BFSPlanner.for_core(core, 32.0, zones["end"], n_targets=1, seed=0)
core.close()
solid = np.asarray(g.solid, bool)                     # (nz, ny, nx)
mins, cell = np.asarray(g.mins, np.float64), float(g.cell)
nz, ny, nx = solid.shape
zc = mins[2] + (np.arange(nz) + 0.5) * cell
kill_z = float(getattr(g, "kill_z", -np.inf))
live = solid & (zc > kill_z)[:, None, None]
# the FLOOR a top-down camera sees: the highest solid cell that has open space above it. Surf
# maps sit in a sealed skybox, so the topmost solid in every column is the ceiling - skip it:
# find each column's highest FREE cell, then the highest solid below that
free = ~solid
has_free = free.any(0)
top_free = nz - 1 - np.argmax(free[::-1], axis=0)
below = live & (np.arange(nz)[:, None, None] < top_free[None, :, :])
has = has_free & below.any(0)
top_idx = nz - 1 - np.argmax(below[::-1], axis=0)
height = np.where(has, zc[top_idx] + 0.5 * cell, np.nan)
ext = (mins[0], mins[0] + nx * cell, mins[1], mins[1] + ny * cell)
fmin, fmax = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])
# the side view runs along the map's long axis (start -> finish)
fc = 0.5 * (fmin + fmax)
AX = 1 if abs(fc[1] - P[0, 1]) >= abs(fc[0] - P[0, 0]) else 0
axn = "y" if AX == 1 else "x"
secs = (T[-1] - T[0] + 1) * tick_ms / 1000.0
_m = 48.0
inbox = np.all((P[-1] >= fmin - _m) & (P[-1] <= fmax + _m))
fell = np.isfinite(kill_z) and P[-1, 2] <= kill_z + 32
won = "FINISHED" if inbox else ("fell into the void" if fell else f"not finished ({trailer.get('end')})")
cmap = plt.get_cmap("plasma")


def base(axt, axs):
    hm = np.ma.masked_invalid(height)
    cm = plt.get_cmap("terrain").copy(); cm.set_bad("black")
    axt.imshow(hm, cmap=cm, origin="lower", extent=ext, interpolation="nearest")
    axt.add_patch(plt.Rectangle((fmin[0], fmin[1]), fmax[0] - fmin[0], fmax[1] - fmin[1],
                                fc=(0.2, 0.9, 0.3, 0.35), ec="lime", lw=2))
    axt.text(fc[0], fmax[1] + 80, "FINISH", color="lime", ha="center", fontsize=10, weight="bold")
    axt.plot(P[0, 0], P[0, 1], "o", color="deepskyblue", ms=9)
    axt.text(P[0, 0] + 80, P[0, 1] - 60, "start", color="deepskyblue", fontsize=10)
    axt.set_aspect("equal"); axt.set_xticks([]); axt.set_yticks([])
    axt.set_title("top-down (colour = height, black = void)", fontsize=9)
    # side: the geometry's silhouette along the long axis (highest solid per slice)
    # height is indexed [y, x]: the profile along y (AX 1) is the max over x (axis 1)
    prof_h = np.nanmax(np.where(np.isfinite(height), height, -np.inf), axis=AX)
    prof_h = np.where(np.isfinite(prof_h), prof_h, np.nan)
    s_ax = (mins[AX] + (np.arange(height.shape[1 - AX]) + 0.5) * cell)
    axs.fill_between(s_ax, np.nanmin(zc) - cell, prof_h, color="0.75", step="mid")
    if np.isfinite(kill_z):
        axs.axhline(kill_z, color="red", ls="--", lw=1)
        axs.text(s_ax[0], kill_z + 20, "kill plane", color="red", fontsize=8)
    axs.add_patch(plt.Rectangle((fmin[AX], fmin[2]), fmax[AX] - fmin[AX], fmax[2] - fmin[2],
                                fc=(0.2, 0.9, 0.3, 0.35), ec="green", lw=2))
    axs.set_xlabel(f"{axn} (u)"); axs.set_ylabel("z (u)")
    axs.set_title(f"side view along {axn} (grey = highest geometry)", fontsize=9)


# ---- static overview
fig, (axt, axs) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.2]})
base(axt, axs)
axt.plot(P[:, 0], P[:, 1], "-", color="white", lw=1.5)
axs.plot(P[:, AX], P[:, 2], "-", color="black", lw=1.5)
for i, p in enumerate(pl):
    ln = np.asarray(p["line"])
    c = cmap(i / max(1, len(pl) - 1))
    axt.plot(ln[:, 0], ln[:, 1], "-", color=c, lw=3, alpha=0.9)
    axt.text(ln[0, 0] + 40, ln[0, 1] + 40, str(i + 1), color=c, fontsize=9, weight="bold")
    axs.plot(ln[:, AX], ln[:, 2], "-", color=c, lw=2.5, alpha=0.9)
fig.suptitle(f"{TITLE}\nepisode {EP}: {len(pl)} planner calls, {secs:.1f} s, {won}", fontsize=11)
fig.tight_layout(); fig.savefig(outp + ".png", dpi=100); plt.close(fig)

# ---- animation
STEP = 4                                              # 40 ms of game per frame (surf is fast)
frames = list(range(0, len(T), STEP)) + [len(T) - 1]
fig, (axt, axs) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.2]})
base(axt, axs)
trt, = axt.plot([], [], "-", color="white", lw=1.5)
trs, = axs.plot([], [], "-", color="black", lw=1.5)
dt, = axt.plot([], [], "o", color="red", ms=9, zorder=6)
ds, = axs.plot([], [], "o", color="red", ms=9, zorder=6)
clt, = axt.plot([], [], "-", color="orange", lw=4, zorder=5)
cls, = axs.plot([], [], "-", color="orange", lw=4, zorder=5)
past = []
arrow = [None]
sup = fig.suptitle("")


def draw(fi):
    i = frames[fi]
    rt = T[i] - T[0]
    trt.set_data(P[: i + 1, 0], P[: i + 1, 1]); trs.set_data(P[: i + 1, AX], P[: i + 1, 2])
    dt.set_data([P[i, 0]], [P[i, 1]]); ds.set_data([P[i, AX]], [P[i, 2]])
    if arrow[0] is not None:
        arrow[0].remove()
    a = np.deg2rad(yaw[i])
    arrow[0] = axt.arrow(P[i, 0], P[i, 1], 220 * np.cos(a), 220 * np.sin(a), width=28,
                         color="red", zorder=7)
    for l_ in past:
        l_.remove()
    past.clear()
    active = [p for p in pl if p["rt"] <= rt]
    for p in active[:-1]:
        ln = np.asarray(p["line"])
        past.append(axt.plot(ln[:, 0], ln[:, 1], "-", color="violet", lw=2, alpha=0.4)[0])
        past.append(axs.plot(ln[:, AX], ln[:, 2], "-", color="violet", lw=2, alpha=0.4)[0])
    if active:
        ln = np.asarray(active[-1]["line"])
        clt.set_data(ln[:, 0], ln[:, 1]); cls.set_data(ln[:, AX], ln[:, 2])
    spd = float(np.hypot(rows[i, 4], rows[i, 5]))
    sup.set_text(f"{TITLE}\nt = {rt * tick_ms / 1000:5.2f} s   plan #{len(active)} of {len(pl)}"
                 f"   speed {spd:,.0f} u/s   z {P[i, 2]:,.0f}")
    return trt, trs, dt, ds, clt, cls


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=40, blit=False)
anim.save(outp + ".mp4", writer=animation.FFMpegWriter(fps=25, bitrate=2400), dpi=90)
plt.close(fig)
print(f"{outp}.png / .mp4: episode {EP}, {len(pl)} plans, {secs:.1f} s, {won}, {len(frames)} frames")
