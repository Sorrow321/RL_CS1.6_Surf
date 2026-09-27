"""plot_archive_bev.py - where an edge archive reached on a map, top-down and from the side.

    python tools/plot_archive_bev.py runs/research/archive_<name>/nodes.npz --map maps_pool/<m>.bsp
        [--occ-cell 32] [--line <route .npz or recording .jsonl> --line-label "..."] --out <png>

A MEASUREMENT of the search (edge_archive.py --dump-nodes): every live node's position over the
map's solid geometry (grey: any solid cell in the column / row), the root (square, "start") and
the finish (star, "finish"). --line overlays a reference path for ANALYSIS only - a self-route, or
a human record (CLAUDE.md section 0: a record is an analysis instrument, never a training input);
it is drawn dashed with a dot every 2 s (a .jsonl recording) and labelled. Grey scale and marker
shapes only, every element labelled in text (the user is colour-blind).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_line(path):
    """-> (points (N, 3), seconds per point or None)."""
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        return np.asarray(z["route"], np.float64), None
    pts, tick_ms = [], 10.0
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("{"):
                h = json.loads(ln)
                if "end" in h and pts:
                    break                       # the first episode only
                tick_ms = float(h.get("tick_ms", tick_ms))
                continue
            r = json.loads(ln)
            pts.append(r[1:4])
    return np.asarray(pts, np.float64), tick_ms / 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nodes")
    ap.add_argument("--map", required=True)
    ap.add_argument("--occ-cell", type=int, default=32)
    ap.add_argument("--line", default=None)
    ap.add_argument("--line-label", default="reference line")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(a.nodes)
    o = np.asarray(z["origin"], np.float64)
    root = np.asarray(z["root"], np.float64)
    fin = np.asarray(z["finish"], np.float64)
    bsp = Path(a.map)
    oc = np.load(bsp.with_name(f"{bsp.stem}.occ_{a.occ_cell}.npz"))
    occ = oc["occ"].astype(bool)                       # [z, y, x]
    mins = np.asarray(oc["mins"], np.float64)
    cell = float(a.occ_cell)
    line, dt = (load_line(a.line) if a.line else (None, None))
    pts = [o, root[None], fin[None]] + ([line] if line is not None else [])
    allp = np.vstack(pts)
    lo = allp.min(0) - 600.0
    hi = allp.max(0) + 600.0

    def idx(v, ax_):
        return int(np.clip(np.floor((v - mins[ax_]) / cell), 0, occ.shape[2 - ax_] - 1))
    x0, x1 = idx(lo[0], 0), idx(hi[0], 0) + 1
    y0, y1 = idx(lo[1], 1), idx(hi[1], 1) + 1
    z0, z1 = idx(lo[2], 2), idx(hi[2], 2) + 1
    sub = occ[z0:z1, y0:y1, x0:x1]
    ext_xy = (mins[0] + x0 * cell, mins[0] + x1 * cell, mins[1] + y0 * cell, mins[1] + y1 * cell)
    ext_xz = (mins[0] + x0 * cell, mins[0] + x1 * cell, mins[2] + z0 * cell, mins[2] + z1 * cell)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7.5),
                                 gridspec_kw={"width_ratios": [1.0, 1.0]})
    # the SHARE of solid cells in each column / row of the plotted box: open air light, walls and
    # floors darker (a surf map is sealed, so "any solid in the column" would be grey everywhere)
    a1.imshow(sub.mean(0), origin="lower", extent=ext_xy, cmap="Greys", vmin=0.0, vmax=1.6,
              interpolation="nearest")
    a2.imshow(sub.mean(1), origin="lower", extent=ext_xz, cmap="Greys", vmin=0.0, vmax=1.6,
              interpolation="nearest", aspect="auto")
    for ax, (i, j) in ((a1, (0, 1)), (a2, (0, 2))):
        ax.scatter(o[:, i], o[:, j], s=4, c="black", marker=".", linewidths=0,
                   label=f"archive nodes ({len(o):,}): where the search reached")
        if line is not None:
            ax.plot(line[:, i], line[:, j], ls="--", lw=1.4, c="black", label=a.line_label)
            if dt:
                k = max(1, int(round(2.0 / dt)))
                ax.plot(line[::k, i], line[::k, j], ls="none", marker="o", ms=4, mfc="white",
                        mec="black", label=f"{a.line_label}: every 2 s")
        ax.plot(root[i], root[j], marker="s", ms=11, mfc="white", mec="black", mew=2, ls="none",
                label="start")
        ax.plot(fin[i], fin[j], marker="*", ms=17, mfc="white", mec="black", mew=1.5, ls="none",
                label="finish")
        ax.annotate("start", (root[i], root[j]), xytext=(8, 8), textcoords="offset points",
                    fontsize=11, weight="bold")
        ax.annotate("finish", (fin[i], fin[j]), xytext=(8, -14), textcoords="offset points",
                    fontsize=11, weight="bold")
    a1.set_xlabel("x (u)")
    a1.set_ylabel("y (u)")
    a1.set_title("top-down (darker: more solid in the column)")
    a2.set_xlabel("x (u)")
    a2.set_ylabel("z, height (u)")
    a2.set_title("side view (darker: more solid along y)")
    a1.legend(loc="best", fontsize=9, framealpha=0.9)
    a1.set_aspect("equal")
    fig.suptitle(a.title or f"edge archive on {bsp.stem}: {len(o):,} nodes", fontsize=13)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"plot_archive_bev: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
