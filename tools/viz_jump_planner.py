"""viz_jump_planner.py - an explainer video of the proposed jump-point planner (docs/planner-design.md
section 7), on a walking maze, computed on the maze's REAL walkable graph (no training involved).

  scene 1  one decision: probe 8 directions (~3 s of walking each) -> merge the probes that end in
           the same place -> the distinct options -> pi = softmax(U / T) -> draw one
  scene 2  a trajectory = a chain of such draws (ancestral sampling), low vs high temperature, and a
           novelty U
  scene 3  search: a small tree of jump points, leaves scored by U, only the first jump executed

    python tools/viz_jump_planner.py <map.bsp> <out_prefix> [seed]
Writes <out_prefix>.mp4 and <out_prefix>.png (the scene-1 decision and the chains, one picture).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import BFSPlanner              # noqa: E402
from surfgym.zones import load_zones                 # noqa: E402

bsp, outp = sys.argv[1:3]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
JUMP_U = 750.0          # one jump = ~3 s of walking at 250 u/s
MIN_ADV = 0.3 * JUMP_U  # a probe that cannot advance this far in its direction is dropped
MERGE_U = 200.0         # probe ends this close ALONG THE FLOOR are the same option
NOV_CELL = 128.0
DIRS = [np.array([np.cos(a), np.sin(a)]) for a in np.radians(np.arange(0, 360, 45))]
DIR_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
PCOL = plt.get_cmap("tab10")

# ---- the maze's walkable graph (stage 1's BFS planner graph)
core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=1, lidar_w=0, lidar_h=0))
core.reset(0)
spawn = np.asarray(core.states_view["origin"][0], np.float64)
zones = load_zones(bsp)
g = BFSPlanner.for_core(core, 32.0, zones["end"], n_targets=0, seed=0)
core.close()
XY = np.asarray(g.xyz, np.float64)[:, :2]
n = len(XY)
nbr = np.asarray(g.nbr)
rows = np.repeat(np.arange(n), nbr.shape[1])
cols = nbr.reshape(-1)
wts = np.tile(np.asarray(g.wk, np.float64), n)
m = cols >= 0
A = sp.csr_matrix((wts[m], (rows[m], cols[m])), shape=(n, n))
fmin, fmax = np.asarray(zones["end"]["mins"]), np.asarray(zones["end"]["maxs"])
FIN = np.asarray(g.finish_center, np.float64)[:2]
s_start = int(g.snap(spawn[None, :])[0])


DFIN = np.asarray(g.dist[g.fin], np.float64)   # path distance to the finish, per node


def progress(pts):
    """How far along the maze a chain got: 1 - (best path distance to the finish) / (start's)."""
    return 1.0 - float(np.min(DFIN[np.asarray(pts)])) / float(DFIN[pts[0]])


def in_finish(v):
    p = XY[v]
    return bool(fmin[0] - 48 <= p[0] <= fmax[0] + 48 and fmin[1] - 48 <= p[1] <= fmax[1] + 48)


_reach = {}


def reach(s, limit=JUMP_U):
    key = (s, limit)
    if key not in _reach:
        _reach[key] = dijkstra(A, directed=True, indices=s, limit=limit, return_predecessors=True)
    return _reach[key]


def path_nodes(pred, s, t):
    out = [t]
    while out[-1] != s and out[-1] >= 0:
        out.append(int(pred[out[-1]]))
    return out[::-1]


def probe(s):
    """8 probes from node s -> list of (dir index, end node, path) or None (could not advance)."""
    d, pred = reach(s)
    idx = np.flatnonzero(np.isfinite(d))
    disp = XY[idx] - XY[s]
    res = []
    for k, u in enumerate(DIRS):
        pr = disp @ u
        j = int(np.argmax(pr - 0.02 * d[idx]))
        if pr[j] < MIN_ADV:
            res.append(None)
            continue
        t = int(idx[j])
        res.append((k, t, path_nodes(pred, s, t)))
    return res


def close_on_floor(a, b):
    d, _ = reach(a, MERGE_U)
    return bool(np.isfinite(d[b]))


def merge(res):
    ch = []
    for r in res:
        if r is None:
            continue
        k, t, path = r
        for c in ch:
            if close_on_floor(c["node"], t):
                c["dirs"].append(k)
                break
        else:
            ch.append({"node": t, "dirs": [k], "path": path})
    return ch


def options(s):
    return merge(probe(s))


def u_euclid(v, visits=None):
    return -float(np.linalg.norm(XY[v] - FIN)) / 1000.0


def cell(v):
    return tuple(np.floor(XY[v] / NOV_CELL).astype(int))


def u_novelty(v, visits):
    return 1.0 / np.sqrt(1.0 + visits[cell(v)])


def softmax(u, T):
    z = np.asarray(u, np.float64) / T
    z = np.exp(z - z.max())
    return z / z.sum()


def sample_chain(s0, T, U, rng, max_jumps=40):
    s, pts, paths, visits = s0, [s0], [], Counter()
    visits[cell(s0)] += 1
    for _ in range(max_jumps):
        ch = options(s)
        if not ch:
            break
        p = softmax([U(c["node"], visits) for c in ch], T)
        c = ch[int(rng.choice(len(ch), p=p))]
        for v in c["path"][::4]:
            visits[cell(v)] += 1
        paths.append(c["path"])
        s = c["node"]
        pts.append(s)
        if in_finish(s):
            break
    return pts, paths


# ---- the scene-1 node: a junction (the most distinct options) near the maze's middle
cand = np.arange(0, n, 7)
mid = np.array([0.5 * (XY[:, 0].min() + XY[:, 0].max()), 0.5 * (XY[:, 1].min() + XY[:, 1].max())])
best = None
for v in cand[np.argsort(np.linalg.norm(XY[cand] - mid, axis=1))][:120]:
    k = len(options(int(v)))
    if best is None or k > best[0]:
        best = (k, int(v))
    if k >= 4:
        break
s_j = best[1]
ch_j = options(s_j)
res_j = probe(s_j)
print(f"junction node {s_j} at {XY[s_j].round()}: {sum(r is not None for r in res_j)} probes advance, "
      f"{len(ch_j)} distinct options")

# ---- scene 2: chains
rng = np.random.default_rng(SEED)
CH = {}
for name, T, U in (("low T (0.05), U = straight-line closeness", 0.05, u_euclid),
                   ("high T (1.0), U = straight-line closeness", 1.0, u_euclid),
                   ("U = novelty (unvisited by this chain), T 0.1", 0.1, u_novelty)):
    CH[name] = [sample_chain(s_start, T, U, rng) for _ in range(8)]
    fins = sum(in_finish(c[0][-1]) for c in CH[name])
    prog = [progress(c[0]) for c in CH[name]]
    print(f"{name}: {fins}/8 chains reached the finish; furthest along the maze "
          f"{100 * max(prog):.0f}%, median {100 * np.median(prog):.0f}%")

# ---- scene 3: a depth-2 tree from the junction
tree = []
for c in ch_j:
    kids = options(c["node"])
    tree.append((c, kids))

# ---- drawing
nodes3 = np.asarray(g.xyz, np.float64)
cellg = 32.0
x0, y0 = nodes3[:, 0].min() - 3 * cellg, nodes3[:, 1].min() - 3 * cellg
x1, y1 = nodes3[:, 0].max() + 3 * cellg, nodes3[:, 1].max() + 3 * cellg
nx, ny = int(np.ceil((x1 - x0) / cellg)), int(np.ceil((y1 - y0) / cellg))
floor = np.zeros((ny, nx), bool)
floor[np.clip(((nodes3[:, 1] - y0) / cellg).astype(int), 0, ny - 1),
      np.clip(((nodes3[:, 0] - x0) / cellg).astype(int), 0, nx - 1)] = True


def base(ax):
    ax.imshow(np.where(floor, 0.93, 0.25), cmap="gray", vmin=0, vmax=1, origin="lower",
              extent=(x0, x0 + nx * cellg, y0, y0 + ny * cellg), interpolation="nearest")
    ax.add_patch(plt.Rectangle((fmin[0], fmin[1]), fmax[0] - fmin[0], fmax[1] - fmin[1],
                               fc=(0.2, 0.8, 0.3, 0.35), ec="green", lw=2))
    ax.text(FIN[0], fmax[1] + 50, "FINISH", color="green", ha="center", fontsize=9, weight="bold")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def pline(ax, path, **kw):
    p = XY[np.asarray(path)]
    ax.plot(p[:, 0], p[:, 1], **kw)


TXT = {
    "s1a": ("ONE DECISION\n\n1. The agent stands at a jump point s (red).\n"
            "   The planner decides once per jump\n   (~3 s of play), not 25 times a second."),
    "s1b": ("2. PROBE: try 8 directions. A probe =\n   walk ~3 s (750 u) along the floor, as far\n"
            "   as possible in that direction.\n   The walk is EXACT and DETERMINISTIC:\n"
            "   the maze graph here; on surf, a copied\n   simulator running the executor."),
    "s1c": ("3. MERGE: probes that end in the same\n   place are ONE option. 8 probes ->\n"
            "   {k} distinct places = the real choices.\n   Corridor: 1-2. Junction: 3-4.\n"
            "   Mid-air on surf: 1 (no choice)."),
    "s1d": ("4. SCORE each option:\n   pi(option) = softmax( U(option) / T )\n"
            "   U = any proxy (here: straight-line\n   closeness to the finish), T = temperature.\n"
            "   Bars: pi at T = 0.05 and T = 1."),
    "s1e": ("5. DRAW one option from pi.\n   THIS is the randomness: the jumps are\n"
            "   deterministic, the CHOICE is random.\n   Higher T -> more random choices."),
    "s1f": ("So pi is NOT a network that outputs the\nnext state. The probes LIST the options;\n"
            "pi only says how likely each one is.\n\nv1: pi = softmax(U/T), no network at all.\n"
            "Later: a network SCORES each option\n(AlphaZero's policy head), and a learned\n"
            "generator can PROPOSE options instead\nof 8 fixed probes (your latent idea)."),
    "s2": ("A TRAJECTORY = a chain of such draws:\nrepeat 1-5 from each new jump point\n"
           "(ancestral sampling; the chain is the\nprobability model). 8 samples, <= 40\n"
           "jumps each, NO search:\n\n{name}\n"
           "{fin}/8 reached the finish; furthest\nalong the maze: {best:.0f}% (median {med:.0f}%)."),
    "s3": ("SEARCH: instead of one draw, look ahead.\nExpand the options of the options (a tree\n"
           "of jump points), score the LEAVES by U,\nexecute only the FIRST jump (bold), then\n"
           "repeat from there, re-using the tree.\n\nDepth 2 here; U = straight-line closeness."),
}

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.25, 1], wspace=0.05, hspace=0.12)


def frame(sc, step=0):
    fig.clf()
    ax = fig.add_subplot(gs[:, 0])
    base(ax)
    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
    axb = fig.add_subplot(gs[1, 1])
    axb.set_visible(False)
    if sc.startswith("s1"):
        ax.plot(*XY[s_j], "o", color="red", ms=10, zorder=6)
        order = ["s1a", "s1b", "s1c", "s1d", "s1e", "s1f"]
        stage = order.index(sc)
        if stage == 1:
            for k, r in enumerate(res_j[:step + 1]):
                if r is not None:
                    pline(ax, r[2], color=PCOL(k), lw=2.5, alpha=0.9)
                    ax.plot(*XY[r[1]], "o", color=PCOL(k), ms=6)
                    ax.text(*(XY[r[1]] + 40), DIR_NAMES[k], color=PCOL(k), fontsize=8)
                else:
                    u = DIRS[k]
                    ax.annotate("", XY[s_j] + 180 * u, XY[s_j], arrowprops=dict(
                        arrowstyle="->", color="0.5", ls="--"))
        if stage >= 2:
            for i, c in enumerate(ch_j):
                col = PCOL(i)
                pline(ax, c["path"], color=col, lw=3)
                ax.plot(*XY[c["node"]], "o", color=col, ms=12, mec="black", zorder=7)
                ax.text(*(XY[c["node"]] + 50), f"option {i + 1}\n({', '.join(DIR_NAMES[k] for k in c['dirs'])})",
                        color=col, fontsize=8, weight="bold")
        if stage >= 3:
            axb.set_visible(True)
            u = [u_euclid(c["node"]) for c in ch_j]
            xs = np.arange(len(ch_j))
            for off, T, a in ((-0.2, 0.05, 1.0), (0.2, 1.0, 0.45)):
                axb.bar(xs + off, softmax(u, T), width=0.38, color=[PCOL(i) for i in xs], alpha=a,
                        edgecolor="black", label=f"T = {T:g}")
            axb.set_xticks(xs); axb.set_xticklabels([f"option {i + 1}" for i in xs])
            axb.set_ylim(0, 1); axb.set_ylabel("pi (probability)"); axb.legend(fontsize=8)
        if stage == 4:
            pick = int(np.random.default_rng(SEED + step).choice(len(ch_j), p=softmax(
                [u_euclid(c["node"]) for c in ch_j], 1.0)))
            ax.plot(*XY[ch_j[pick]["node"]], "*", color="gold", ms=28, mec="black", zorder=8)
            axb.set_title(f"draw at T = 1: option {pick + 1}", fontsize=10)
        axt.text(0, 1, TXT[sc].format(k=len(ch_j)), va="top", family="monospace", fontsize=11,
                 transform=axt.transAxes)
    elif sc == "s2":
        name = list(CH)[step]
        cols = plt.get_cmap("plasma")
        for i, (pts, paths) in enumerate(CH[name]):
            c = cols(i / 8)
            for pth in paths:
                pline(ax, pth, color=c, lw=2, alpha=0.8)
            ax.plot(XY[pts, 0], XY[pts, 1], "o", color=c, ms=3.5)
        ax.plot(*XY[s_start], "o", color="red", ms=10, zorder=6)
        fin = sum(in_finish(c[0][-1]) for c in CH[name])
        prog = [100 * progress(c[0]) for c in CH[name]]
        axt.text(0, 1, TXT["s2"].format(name=name, fin=fin, best=max(prog), med=np.median(prog)),
                 va="top", family="monospace", fontsize=11, transform=axt.transAxes)
    elif sc == "s3":
        ax.plot(*XY[s_j], "o", color="red", ms=10, zorder=6)
        best_i, best_v = None, -1e9
        for i, (c, kids) in enumerate(tree):
            leaf_u = [u_euclid(k["node"]) for k in kids] or [u_euclid(c["node"])]
            if max(leaf_u) > best_v:
                best_i, best_v = i, max(leaf_u)
        cm = plt.get_cmap("viridis")
        allu = [u_euclid(k["node"]) for _, kids in tree for k in kids] or [0.0]
        lo, hi = min(allu), max(allu)
        for i, (c, kids) in enumerate(tree):
            pline(ax, c["path"], color="black", lw=4 if i == best_i else 1.5, alpha=0.9)
            ax.plot(*XY[c["node"]], "o", color="black", ms=8)
            for k in kids:
                pline(ax, k["path"], color="0.35", lw=1, alpha=0.8)
                v = (u_euclid(k["node"]) - lo) / max(hi - lo, 1e-9)
                ax.plot(*XY[k["node"]], "o", color=cm(v), ms=9, mec="black")
        axt.text(0, 1, TXT["s3"], va="top", family="monospace", fontsize=11, transform=axt.transAxes)
        axb.set_visible(True); axb.axis("off")
        axb.text(0, 0.8, "leaf colour: U (yellow = closer to the finish)", fontsize=9,
                 transform=axb.transAxes)
    fig.suptitle(f"The jump-point planner, explained on {Path(bsp).stem} (the real walkable graph)",
                 fontsize=13)


script = ([("s1a", 0)] * 3 + [("s1b", k) for k in range(8) for _ in range(2)] + [("s1b", 7)] * 3
          + [("s1c", 0)] * 6 + [("s1d", 0)] * 7 + [("s1e", k) for k in range(4) for _ in range(2)]
          + [("s1f", 0)] * 10 + [("s2", i) for i in range(3) for _ in range(7)] + [("s3", 0)] * 10)
anim = animation.FuncAnimation(fig, lambda f: frame(*script[f]), frames=len(script),
                               interval=1000, blit=False)
anim.save(outp + ".mp4", writer=animation.FFMpegWriter(fps=1.4, bitrate=2400), dpi=80)

# the still: the decision with its bars, and the three chain panels
fig2, axs = plt.subplots(1, 4, figsize=(20, 8))
base(axs[0]); axs[0].plot(*XY[s_j], "o", color="red", ms=10, zorder=6)
for i, c in enumerate(ch_j):
    pline(axs[0], c["path"], color=PCOL(i), lw=3)
    axs[0].plot(*XY[c["node"]], "o", color=PCOL(i), ms=12, mec="black")
u = [u_euclid(c["node"]) for c in ch_j]
axs[0].set_title("one decision, pi at T=0.05 / T=1:\n" + ", ".join(
    f"opt {i + 1}: {p:.2f}/{q:.2f}" for i, (p, q) in enumerate(zip(softmax(u, 0.05), softmax(u, 1.0)))),
    fontsize=9)
for ax, name in zip(axs[1:], CH):
    base(ax)
    cols = plt.get_cmap("plasma")
    for i, (pts, paths) in enumerate(CH[name]):
        for pth in paths:
            pline(ax, pth, color=cols(i / 8), lw=2, alpha=0.8)
    ax.plot(*XY[s_start], "o", color="red", ms=10)
    prog = [100 * progress(c[0]) for c in CH[name]]
    ax.set_title(f"{name}\n{sum(in_finish(c[0][-1]) for c in CH[name])}/8 reached the finish, furthest "
                 f"{max(prog):.0f}% of the maze", fontsize=9)
fig2.tight_layout(); fig2.subplots_adjust(top=0.9); fig2.savefig(outp + ".png", dpi=90)
print(f"{outp}.mp4 ({len(script)} frames) / .png")
