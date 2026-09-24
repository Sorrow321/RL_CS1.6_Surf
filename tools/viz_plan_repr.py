"""viz_plan_repr.py - WHAT THE EXECUTOR RECEIVES about its plan, frame by frame, on a real recorded episode.

Replays a recording of a --goals checkpoint and, at every executor decision (every act_every ticks),
re-computes the two plan representations with the SAME code the trainer uses:

  * the FAN (--goal-obs fan and fanline): 9 points of the current plan - the point nearest the agent
    and 8 points ahead of it at 0.25 ... 2 s x max(speed, 500 u/s) of arc - each as (forward, left,
    up) in the agent's own frame, divided by that point's nominal distance: 27 numbers
    (surfgym.goals.MultiLine.features);
  * the DRAWN PLAN (--goal-obs fanline only): the next 12 plan vertices (128 u apart) drawn as equal
    dots into one extra depth channel of the agent's own 64x32 camera, value = depth
    (surfgym.goalball.PlanLineLidar), next to the map depth channel every executor gets.

    python tools/viz_plan_repr.py <traj.jsonl> <run.json> <map.bsp> <out_prefix> [episode] [title]
Writes <out_prefix>.mp4 and <out_prefix>.png (one annotated frame).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import BFSPlanner              # noqa: E402
from surfgym.goals import MultiLine                  # noqa: E402
from surfgym.goalball import PlanLineLidar           # noqa: E402
from surfgym.vision import GpuLidar                  # noqa: E402
from surfgym.zones import load_zones                 # noqa: E402

traj_p, runjson, bsp, outp = sys.argv[1:5]
EP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
TITLE = sys.argv[6] if len(sys.argv) > 6 else Path(bsp).stem
cfg = json.load(open(runjson, encoding="utf-8"))["config"]
K = int(cfg.get("act_every", 4))
offs = cfg.get("goal_fan_offsets") or [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- the episode: rows [t, x, y, z, vx, vy, vz, yaw, buttons, onground, progress, reward, pitch]
eps, cur, hdrs = [], None, []
for line in open(traj_p, encoding="utf-8"):
    if line.startswith("["):
        cur.append(json.loads(line))
    else:
        r = json.loads(line)
        if "end" in r:
            eps.append(np.asarray(cur, np.float64)); cur = None
        else:
            hdrs.append(r); cur = []
rows = eps[EP]
tick_ms = float(hdrs[EP].get("tick_ms", 10.0))
P, V, YAW = rows[:, 1:4], rows[:, 4:7], rows[:, 7]
PITCH = rows[:, 12] if rows.shape[1] > 12 else np.zeros(len(rows))
DUCK = (rows[:, 8].astype(np.int64) & 4) != 0

# ---- the plan the eval gave: the BFS route on the checkpoint's graph from the spawn to the finish
core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=1, lidar_w=0, lidar_h=0))
zones = load_zones(bsp)
g = BFSPlanner.for_core(core, float(cfg.get("goal_cell") or 32.0), zones["end"], n_targets=0, seed=0,
                        **({"graph_kind": "ride"} if cfg.get("plan_graph") == "ride" else {}))
plan = g.plan(P[0], g.fin).line
lid = GpuLidar(core, int(cfg.get("lidar_w", 64)), int(cfg.get("lidar_h", 32)),
               hfov_deg=float(cfg.get("lidar_hfov", 120.0)), vfov_deg=float(cfg.get("lidar_vfov", 90.0)),
               range_units=float(cfg.get("lidar_range", 11500.0)), cell=float(cfg.get("lidar_cell", 32.0)),
               near_range=float(cfg.get("lidar_near", 2000.0)), device=dev)
pll = PlanLineLidar(lid, 1)
ml = MultiLine(1, device=dev, offsets=offs)
ml.set_lines(np.array([0]), [plan])
pll.line = ml
fmin, fmax = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])


def fan_world(p, speed):
    """The fan's 9 points in WORLD coordinates (the same math as MultiLine.features)."""
    o = torch.as_tensor(p[None, :], dtype=torch.float32, device=dev)
    i0, s0 = ml._anchor(o)
    span = max(float(speed), ml.speed_floor) * ml.t.cpu().numpy()
    L = int(ml.length[0]) - 1
    f = np.minimum(np.maximum((float(s0[0]) + span) / ml.spacing, 0.0), L)
    lo = np.floor(f).astype(int); w = (f - lo)[:, None]; hi = np.minimum(lo + 1, L)
    pts = ml.pts[0].cpu().numpy()
    tgt = pts[lo] * (1 - w) + pts[hi] * w
    return np.vstack([pts[int(i0[0])][None, :], tgt])


frames = []
for i in range(0, len(rows), K):
    p = P[i]
    sp = float(np.hypot(V[i, 0], V[i, 1]))
    o = torch.as_tensor(p[None, :], dtype=torch.float32, device=dev)
    y = torch.as_tensor([YAW[i]], dtype=torch.float32, device=dev)
    pi = torch.as_tensor([PITCH[i]], dtype=torch.float32, device=dev)
    dk = torch.as_tensor([int(DUCK[i])], dtype=torch.int32, device=dev)
    img = pll.render(o, y, pi, dk)[0].detach().cpu().numpy()          # (H, W, 2)
    feat = ml.features(o, y, torch.as_tensor([sp], dtype=torch.float32, device=dev))[0]
    frames.append({"i": i, "p": p, "yaw": YAW[i], "speed": sp, "depth": img[:, :, 0],
                   "line": img[:, :, 1], "fan": feat.detach().cpu().numpy().reshape(9, 3),
                   "fanw": fan_world(p, sp)})
core.close()

# ---- the map from above: the highest ride-shell cell per column (a surf map has no floor to draw)
xyz = np.asarray(g.xyz)
cell = float(g.cell)
x0, y0 = xyz[:, 0].min() - 2 * cell, xyz[:, 1].min() - 2 * cell
nx = int(np.ceil((xyz[:, 0].max() + 2 * cell - x0) / cell)); ny = int(np.ceil((xyz[:, 1].max() + 2 * cell - y0) / cell))
hmap = np.full((ny, nx), np.nan)
ix = ((xyz[:, 0] - x0) / cell).astype(int); iy = ((xyz[:, 1] - y0) / cell).astype(int)
order = np.argsort(xyz[:, 2])
hmap[iy[order], ix[order]] = xyz[order, 2]
ext = (x0, x0 + nx * cell, y0, y0 + ny * cell)
LAB = ["0 (nearest)"] + [f"{k}: +{t:g}s" for k, t in enumerate(offs, 1)]

fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.0], height_ratios=[1, 1], hspace=0.32, wspace=0.18)


def draw(k):
    fr = frames[k]
    fig.clf()
    ax = fig.add_subplot(gs[:, 0])
    cm = plt.get_cmap("cividis").copy(); cm.set_bad("black")
    ax.imshow(np.ma.masked_invalid(hmap), cmap=cm, origin="lower", extent=ext, interpolation="nearest",
              alpha=0.8)
    ax.plot(plan[:, 0], plan[:, 1], "--", color="white", lw=1.5, label="the plan (whole route)")
    ax.plot(P[: fr["i"] + 1, 0], P[: fr["i"] + 1, 1], "-", color="red", lw=1.5, label="agent so far")
    fw = fr["fanw"]
    ax.plot(fw[:, 0], fw[:, 1], "o", color="white", ms=6, mec="black")
    for j in (0, 2, 4, 8):
        ax.annotate(str(j), fw[j, :2], xytext=(6, 4), textcoords="offset points", color="white", fontsize=9,
                    weight="bold")
    a = np.radians(fr["yaw"])
    for s in (-60.0, 60.0):
        b = a + np.radians(s)
        ax.plot([fr["p"][0], fr["p"][0] + 900 * np.cos(b)], [fr["p"][1], fr["p"][1] + 900 * np.sin(b)], ":",
                color="orange", lw=1.2)
    ax.annotate("", fr["p"][:2] + 350 * np.array([np.cos(a), np.sin(a)]), fr["p"][:2],
                arrowprops=dict(arrowstyle="-|>", color="orange", lw=2))
    ax.add_patch(plt.Rectangle((fmin[0], fmin[1]), fmax[0] - fmin[0], fmax[1] - fmin[1], fill=False,
                               ec="white", lw=1.5, ls="--"))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("the map from above (for you, NOT an input)\nwhite dots 0-8 = the fan's 9 points; "
                 "orange = view direction and the 120 deg camera", fontsize=9)
    ax.legend(loc="lower left", fontsize=7)
    # the two image channels
    a1 = fig.add_subplot(gs[0, 1])
    a1.imshow(fr["depth"], cmap="gray", interpolation="nearest")
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("INPUT the executor always gets: its depth camera (64 x 32)", fontsize=9)
    a2 = fig.add_subplot(gs[1, 1])
    a2.imshow(fr["depth"], cmap="gray", interpolation="nearest", alpha=0.35)
    ln = np.ma.masked_where(fr["line"] <= 0, fr["line"])
    a2.imshow(ln, cmap="autumn", interpolation="nearest")
    ys, xs = np.nonzero(fr["line"] > 0)
    a2.set_xticks([]); a2.set_yticks([])
    a2.set_title("DRAWN PLAN (fanline only): the next 12 plan points as dots\nin a 2nd channel of the SAME "
                 f"camera (shown over a faded depth image) - {len(xs)} px lit", fontsize=9)
    # the fan in the agent's frame: positions in u (what the 27 numbers encode), then the numbers
    a3 = fig.add_subplot(gs[:, 2])
    fan = fr["fan"]
    nom = np.r_[ml.near_scale, np.maximum(max(fr["speed"], ml.speed_floor) * np.asarray(offs), ml.near_scale)]
    fwd_u, left_u = fan[:, 0] * nom, fan[:, 1] * nom
    a3.axhline(0, color="0.7", lw=0.8); a3.axvline(0, color="0.7", lw=0.8)
    a3.plot(-left_u, fwd_u, "-", color="0.5", lw=1)
    a3.scatter(-left_u, fwd_u, s=55, c="black", zorder=3)
    for j in range(9):
        a3.annotate(str(j), (-left_u[j], fwd_u[j]), xytext=(5, 3), textcoords="offset points", fontsize=9,
                    weight="bold")
    a3.plot(0, 0, "^", color="orange", ms=14, mec="black", zorder=4)
    lim = max(300.0, 1.15 * float(np.max(np.abs(np.r_[fwd_u, left_u]))))
    a3.set_xlim(-lim, lim); a3.set_ylim(-0.45 * lim, lim)
    a3.set_aspect("equal")
    a3.set_xlabel("<- left     sideways (u)     right ->")
    a3.set_ylabel("forward (u)")
    a3.set_title("THE FAN (fan and fanline), in the agent's own frame\n(agent = triangle facing up; "
                 "points 0-8 along the plan)", fontsize=9)
    txt = "the 27 numbers it gets\n(fwd, left, up) / distance\n" + "\n".join(
        f"{j}: {fan[j, 0]:+.2f} {fan[j, 1]:+.2f} {fan[j, 2]:+.2f}" for j in range(9))
    a3.text(0.02, 0.02, txt, transform=a3.transAxes, family="monospace", fontsize=7.5, va="bottom",
            bbox=dict(fc="white", ec="0.6", alpha=0.9))
    fig.suptitle(f"{TITLE} - t = {fr['i'] * tick_ms / 1000:5.2f} s, speed {fr['speed']:,.0f} u/s "
                 f"(the fan's points spread with speed: 0.25-2 s x max(speed, 500))", fontsize=12)


mid = range(len(frames) // 5, 4 * len(frames) // 5)
k_png = max(mid, key=lambda k: int((frames[k]["line"] > 0).sum()))   # a frame with the plan in view
draw(k_png)
fig.savefig(outp + ".png", dpi=90)
anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=100, blit=False)
anim.save(outp + ".mp4", writer=animation.FFMpegWriter(fps=10, bitrate=2400), dpi=75)
plt.close(fig)
print(f"{outp}.mp4 ({len(frames)} decisions, {len(rows) * tick_ms / 1000:.1f} s) / .png")
