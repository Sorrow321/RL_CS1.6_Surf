"""viz_planner_input.py - what the LEARNED planner sees and what it outputs, at every call.

Replays a recorded episode (tools/record_ckpt.py JSONL + its --dump-plans JSON): rebuilds the
planner's observation at each call exactly as the eval does (surfgym.goallearn.build_obs on the
WalkMap patch, this episode's VisitGrid and the 9 scalars), runs the checkpoint's planner network
on it and checks that its greedy choice is the shape the recording logged. Per call it draws
  the map (context only - NOT an input), with the 2,048 u square the planner sees
  input 1: the walkable-floor patch (32 x 32 cells of 64 u, world-aligned, north up)
  input 2: this episode's visit counts on the same cells
  input 3: the 9 scalars
  the output: every shape of the vocabulary coloured by the network's probability, and the top 8
    python tools/viz_planner_input.py <ckpt.pt> <traj.jsonl> <plans.json> <map.bsp> <out_prefix>
                                      [episode] [png_call] [title]
Writes <out_prefix>.mp4 (1.5 s per call) and <out_prefix>.png (call png_call, default the middle).
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
from surfgym.goalplan import BFSPlanner, PLAN_SEED_OFFSET  # noqa: E402
from surfgym.goallearn import (WalkMap, VisitGrid, build_obs,  # noqa: E402
                               planner_from_state)
from surfgym.zones import load_zones                 # noqa: E402

ckpt_p, traj_p, plans_p, bsp, outp = sys.argv[1:6]
EP = int(sys.argv[6]) if len(sys.argv) > 6 else 0
PNG_CALL = int(sys.argv[7]) if len(sys.argv) > 7 else -1
TITLE = sys.argv[8] if len(sys.argv) > 8 else Path(bsp).stem

ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
cfg = ck.get("config") or {}
net, vocab, spec = planner_from_state(ck["planner"], "cpu")
if spec.get("vocab") not in (None, "walk"):
    raise SystemExit(f"a walking planner only (this one is {spec.get('vocab')!r})")

# ---- the recorded episode: rows [tick, x, y, z, vx, vy, vz, yaw, ...]
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
rows = np.asarray(eps[EP][0], np.float64)
T = rows[:, 0].astype(int) - int(rows[0, 0])
P, V, YAW = rows[:, 1:4], rows[:, 4:7], rows[:, 7]
tick_ms = float(hdrs[EP].get("tick_ms", 10.0))
pl = sorted((p for p in json.load(open(plans_p, encoding="utf-8"))["plans"]
             if int(p["ep"]) == EP), key=lambda p: int(p["tick"]))
t_base = int(pl[0]["tick"])

# ---- the planner's world: stage 1's walkable graph at the recording's graph cell
cell = float((cfg.get("heldout_goal_cells") or {}).get(Path(bsp).stem)
             or cfg.get("goal_cell") or 32.0)
core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=2, lidar_w=0, lidar_h=0))
zones = load_zones(bsp)
g = BFSPlanner.for_core(core, cell, zones["end"], n_targets=0,
                        seed=int(cfg.get("seed") or 0) + PLAN_SEED_OFFSET)
core.close()
wm = WalkMap(g, spec["patch_cell_u"], spec["patch_n"], spec["patch_layers"])
vis = VisitGrid(1, wm)
finish = np.asarray(g.finish_center, np.float64)
fmin, fmax = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])

# ---- replay: the visit grid is updated every tick with the post-step position (row t+1 is
# the state after step t, row 0 the spawn), and a plan logged at tick t is chosen after that
# tick's update - exactly the order of make_learned_hooks' episode_meta / on_tick
calls = []
vis.reset([0])
vis.update(P[0:1])
nxt = 1
for p in pl:
    rt = int(p["tick"]) - t_base
    while nxt <= rt:
        vis.update(P[nxt:nxt + 1]); nxt += 1
    i = int(np.searchsorted(T, rt))
    img, scal = build_obs(wm, vis, np.zeros(1, np.int64), P[i:i + 1], V[i:i + 1], YAW[i:i + 1],
                          finish, int(spec["visit_cap"]))
    with torch.no_grad():
        lg, val = net(torch.as_tensor(img), torch.as_tensor(scal))
    pr = torch.softmax(lg.float(), -1)[0].numpy().astype(np.float64)
    calls.append({"i": i, "rt": rt, "img": img[0], "scal": scal[0], "prob": pr,
                  "val": float(val[0]), "k": int(p["shape"]), "k_net": int(np.argmax(pr)),
                  "anchor_err": float(np.linalg.norm(np.asarray(p["anchor"]) - P[i]))})
n_ok = sum(c["k"] == c["k_net"] for c in calls)
print(f"{len(calls)} planner calls; replayed observation -> network argmax == logged shape on "
      f"{n_ok}/{len(calls)}; max anchor error {max(c['anchor_err'] for c in calls):.3f} u")

# ---- drawing helpers
nodes = np.asarray(g.xyz, np.float64)
x0, y0 = nodes[:, 0].min() - 3 * cell, nodes[:, 1].min() - 3 * cell
x1, y1 = nodes[:, 0].max() + 3 * cell, nodes[:, 1].max() + 3 * cell
nx, ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
floor = np.zeros((ny, nx), bool)
floor[np.clip(((nodes[:, 1] - y0) / cell).astype(int), 0, ny - 1),
      np.clip(((nodes[:, 0] - x0) / cell).astype(int), 0, nx - 1)] = True
N, pc, h = wm.n, wm.pc, wm.half
K = vocab.K
raw = np.asarray(vocab.raw, np.float64)[:, :, :2]


def extent(p):
    cy, cx = wm.cells(p[None, :])
    xa = wm.mins[0] + (int(cx[0]) - h) * pc
    ya = wm.mins[1] + (int(cy[0]) - h) * pc
    return (xa, xa + N * pc, ya, ya + N * pc)


def sdesc(k):
    hd, a, ta = vocab.heading_deg[k], vocab.turn_deg[k], vocab.turn_at[k]
    if ta < 0:
        return (f"#{k:3d} head {hd:5.1f}, " + ("straight" if a == 0 else f"arc {a:+.0f}"))
    return f"#{k:3d} head {hd:5.1f}, {ta:.2g} then {a:+.0f}"


def draw_call(fig, j, full=True):
    c = calls[j]
    i, p = c["i"], P[c["i"]]
    ext = extent(p)
    s = c["scal"]
    fig.clf()
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.05, 1, 1],
                          hspace=0.28, wspace=0.12)
    # A - the map: context for the viewer, the planner never sees it
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(np.where(floor, 0.93, 0.25), cmap="gray", vmin=0, vmax=1, origin="lower",
              extent=(x0, x0 + nx * cell, y0, y0 + ny * cell), interpolation="nearest")
    ax.add_patch(plt.Rectangle((fmin[0], fmin[1]), fmax[0] - fmin[0], fmax[1] - fmin[1],
                               fc=(0.2, 0.8, 0.3, 0.35), ec="green", lw=2))
    ax.text(0.5 * (fmin[0] + fmax[0]), fmax[1] + 60, "FINISH", color="green", ha="center",
            fontsize=9, weight="bold")
    ax.plot(P[: i + 1, 0], P[: i + 1, 1], "-", color="black", lw=1.3, alpha=0.8)
    for q in calls[:j]:
        ln = raw[q["k"]] + P[q["i"], :2]
        ax.plot(ln[:, 0], ln[:, 1], "-", color="purple", lw=1.5, alpha=0.3)
    ln = raw[c["k"]] + p[:2]
    ax.plot(ln[:, 0], ln[:, 1], "-", color="orange", lw=3.5)
    ax.plot(p[0], p[1], "o", color="red", ms=8, zorder=6)
    ax.add_patch(plt.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False,
                               ec="deepskyblue", lw=2.2, ls="--", zorder=5))
    ax.set_xlim(x0, x0 + nx * cell); ax.set_ylim(y0, y0 + ny * cell)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("the map - for you, NOT an input\nblue square = the 2,048 u the planner sees",
                 fontsize=10)
    # B - input 1: the walkable floor
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(c["img"][0], cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext,
              interpolation="nearest")
    ax.plot(p[0], p[1], "o", color="red", ms=7)
    L = 700.0
    ax.arrow(p[0], p[1], L * s[0], L * s[1], width=22, color="limegreen", zorder=6)
    sp = float(np.hypot(s[4], s[5]))
    if sp > 1e-3:
        ax.arrow(p[0], p[1], 350 * s[4] / sp, 350 * s[5] / sp, width=16, color="deepskyblue",
                 zorder=6)
    ax.arrow(p[0], p[1], 250 * s[8], 250 * s[7], width=12, color="red", zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("INPUT 1 (image): walkable floor, 32 x 32 cells of 64 u\n"
                 "white = floor, black = wall; north up, agent at the centre", fontsize=9)
    # C - input 2: visits
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(c["img"][0], cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext,
              interpolation="nearest", alpha=0.25)
    vm = np.ma.masked_where(c["img"][1] <= 0, c["img"][1])
    ax.imshow(vm, cmap="autumn_r", vmin=0, vmax=1, origin="lower", extent=ext,
              interpolation="nearest")
    ax.plot(p[0], p[1], "o", color="red", ms=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("INPUT 2 (image): cells visited THIS episode\n"
                 "entries per cell, capped at 4 (yellow 1 ... red 4)", fontsize=9)
    # D - input 3: the 9 scalars
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    d = 1000.0 * float(np.expm1(s[3]))
    yaw = float(np.degrees(np.arctan2(s[7], s[8])))
    txt = ("INPUT 3: 9 numbers (world frame)\n\n"
           f"  unit vector to finish  ({s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f})\n"
           f"                         (green arrow, straight line)\n"
           f"  log(1 + dist/1000)     {s[3]:.2f}   = {d:,.0f} u straight-line\n"
           f"  velocity / 1000        ({s[4]:+.2f}, {s[5]:+.2f}, {s[6]:+.2f})\n"
           f"                         = {1000 * sp:,.0f} u/s (blue arrow)\n"
           f"  sin yaw, cos yaw       ({s[7]:+.2f}, {s[8]:+.2f})  = {yaw:.0f} deg (red)\n\n"
           "NOT inputs: the depth render, the key state,\n"
           "the geodesic distance, anything outside the square\n\n"
           f"OUTPUT, top 8 of {K} shapes (probability):\n")
    pr = c["prob"]
    top = np.argsort(-pr)[:8]
    txt += "\n".join(f"  {pr[k]:5.1%}  {sdesc(k)}" + ("  <- chosen" if k == c["k"] else "")
                     for k in top)
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=9, va="top", transform=ax.transAxes)
    # E - the output
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(c["img"][0], cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext,
              interpolation="nearest", alpha=0.55)
    order = np.argsort(pr)
    cm = plt.get_cmap("viridis")
    pmax = float(pr.max())
    for k in order:
        ln = raw[k] + p[:2]
        a = float(pr[k] / pmax)
        ax.plot(ln[:, 0], ln[:, 1], "-", color=cm(a), lw=0.6 + 2.0 * a, alpha=0.12 + 0.88 * a)
    ln = raw[c["k"]] + p[:2]
    ax.plot(ln[:, 0], ln[:, 1], "-", color="orange", lw=3.5, zorder=5)
    ax.plot(p[0], p[1], "o", color="red", ms=7, zorder=6)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"OUTPUT: softmax over {K} fixed shapes (all drawn,\n"
                 "brighter = more probable); orange = the argmax", fontsize=9)
    ok = "yes" if c["k"] == c["k_net"] else f"NO (net {c['k_net']})"
    fig.suptitle(f"{TITLE} - learned planner call {j + 1} of {len(calls)} at t = "
                 f"{c['rt'] * tick_ms / 1000:.1f} s    planner value {c['val']:+.2f}    "
                 f"replayed input -> logged choice: {ok}", fontsize=11)


fig = plt.figure(figsize=(18, 10))
jp = PNG_CALL if PNG_CALL >= 0 else len(calls) // 2
draw_call(fig, jp)
fig.savefig(outp + ".png", dpi=100, bbox_inches="tight")
frames = [j for j in range(len(calls)) for _ in range(3)]
anim = animation.FuncAnimation(fig, lambda f: draw_call(fig, frames[f]), frames=len(frames),
                               interval=500, blit=False)
anim.save(outp + ".mp4", writer=animation.FFMpegWriter(fps=2, bitrate=2400), dpi=80)
plt.close(fig)
print(f"{outp}.png (call {jp + 1}) / .mp4 ({len(calls)} calls, 1.5 s each)")
