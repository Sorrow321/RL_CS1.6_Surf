#!/usr/bin/env python3
"""bev_branch.py - a bird's-eye view of a critical point, with both branches
and the shaping field's own arrows on it.

Three layers over one window of the map:

  1. what is UNDER the flight, taken over the z-slab the two branches
     actually occupy: solid that a player could RIDE (0.1 <= |n_z| <= 0.7,
     the surf band) in one shade, solid he could only stand on or bounce off
     in another, open air left blank. This is the layer the goal field does
     not have.
  2. the goal field's own STEEPEST-DESCENT direction at the flight's median
     height, as a quiver - the direction the shaping reward pays for.
  3. the two branches: the failing policy's own greedy episode and the
     surviving line, with a marker every second.

    python tools/bev_branch.py --map C:/RL_Surf/maps/surf_petrus_lite.bsp \
        --naive C:/RL_Surf_pr1/runs/prRATCH/traj_2757754880.jsonl \
        --correct runs/research/cornerdiag/wr/petrus_wr.jsonl \
        --window -1200 3900 1200 2100 --t0 5.5 --t1 10.5 \
        --out docs/img/bev_petrus_lbend.png

Read-only; the field and the surface mask come out of the baked caches next
to the .bsp, so nothing can bake. Absolute main-checkout map path only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from dip_report import load_field, load_traj     # noqa: E402
from deception_profile import descent_dir        # noqa: E402
from field_probe import Field                    # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


def pick_ep(path, gf, act_every, episode="best"):
    eps = load_traj(path)
    if episode == "best":
        k = int(np.argmin([float(gf.sample(a[:, 1:4]).min())
                           for _f, _h, a in eps]))
    else:
        k = int(episode)
    foot, hdr, a = eps[k]
    tm = float(hdr.get("tick_ms", 10.0)) if hdr else 10.0
    return a, tm, k


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--naive", required=True)
    ap.add_argument("--naive-episode", default="best")
    ap.add_argument("--naive-label", default="the failing policy")
    ap.add_argument("--correct", required=True)
    ap.add_argument("--correct-episode", default="best")
    ap.add_argument("--correct-label", default="the surviving line")
    ap.add_argument("--window", nargs=4, type=float, required=True,
                    metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ap.add_argument("--naive-window", nargs=2, type=float, default=None,
                    metavar=("T0", "T1"), help="seconds of the naive episode")
    ap.add_argument("--correct-window", nargs=2, type=float, default=None)
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--quiver-step", type=int, default=6, help="in cells")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gf = load_field(Path(a.map), a.cell)
    fp = Field(Path(a.map), a.cell)
    na, ntm, nk = pick_ep(a.naive, gf, a.act_every, a.naive_episode)
    ca, ctm, ck = pick_ep(a.correct, gf, a.act_every, a.correct_episode)

    def cut(arr, tm, win):
        t = np.arange(len(arr)) * tm / 1000.0
        if win is None:
            return arr, t
        m = (t >= win[0]) & (t <= win[1])
        return arr[m], t[m]

    na, nt = cut(na, ntm, a.naive_window)
    ca, ct = cut(ca, ctm, a.correct_window)

    x0, y0, x1, y1 = a.window
    zs = np.concatenate([na[:, 3], ca[:, 3]])
    zlo, zhi = float(zs.min()) - 64.0, float(zs.max()) + 64.0
    zmed = float(np.median(zs))

    cell = float(fp.cell)
    ix0 = int(np.floor((x0 - fp.mins[0]) / cell))
    ix1 = int(np.ceil((x1 - fp.mins[0]) / cell))
    iy0 = int(np.floor((y0 - fp.mins[1]) / cell))
    iy1 = int(np.ceil((y1 - fp.mins[1]) / cell))
    iz0 = max(0, int(np.floor((zlo - fp.mins[2]) / cell)))
    iz1 = min(fp.shape[0], int(np.ceil((zhi - fp.mins[2]) / cell)))
    sub_occ = fp.occ[iz0:iz1, iy0:iy1, ix0:ix1]
    ride = (fp.surfy[iz0:iz1, iy0:iy1, ix0:ix1] if fp.surfy is not None
            else np.zeros_like(sub_occ))
    layer = np.where(ride.any(0), 2, np.where(sub_occ.any(0), 1, 0))

    ext = (fp.mins[0] + ix0 * cell, fp.mins[0] + ix1 * cell,
           fp.mins[1] + iy0 * cell, fp.mins[1] + iy1 * cell)
    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    from matplotlib.colors import ListedColormap
    ax.imshow(layer, origin="lower", extent=ext, aspect="equal",
              cmap=ListedColormap([SURF, "#c9c7c0", "#9fc6ea"]),
              interpolation="nearest", zorder=0)

    q = a.quiver_step
    gx = np.arange(ext[0] + cell * q / 2, ext[1], cell * q)
    gy = np.arange(ext[2] + cell * q / 2, ext[3], cell * q)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel(),
                  np.full(GX.size, zmed)], axis=1)
    d = gf.sample(P)
    u = descent_dir(gf, P)
    ok = (d < gf.reach_max - cell) & (np.linalg.norm(u, axis=1) > 1e-6)
    ax.quiver(P[ok, 0], P[ok, 1], u[ok, 0], u[ok, 1], color=INK2,
              alpha=0.55, width=0.0026, scale=32, zorder=2)

    for arr, t, col, lab, tm in ((ca, ct, C[0], a.correct_label, ctm),
                                 (na, nt, C[1], a.naive_label, ntm)):
        ax.plot(arr[:, 1], arr[:, 2], "-", color=col, lw=2.6, label=lab,
                zorder=4)
        step = int(round(1000.0 / tm))
        ax.plot(arr[::step, 1], arr[::step, 2], "o", color=col, ms=5,
                mec=SURF, mew=1.0, zorder=5)
        for i in range(0, len(arr), step):
            ax.annotate(f"{t[i]:.0f}s", (arr[i, 1], arr[i, 2]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=col, zorder=6)
        ax.plot(arr[-1, 1], arr[-1, 2], "X", color=col, ms=13, mew=2.2,
                zorder=6)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("x (map units)"); ax.set_ylabel("y (map units)")
    ax.set_title(a.title or
                 f"{Path(a.map).stem}: the branch, the surfaces under it, "
                 f"and the field's own arrows", loc="left")
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="#9fc6ea", label="RIDABLE surface below "
                                           f"(z {zlo:.0f}..{zhi:.0f})"),
          Patch(facecolor="#c9c7c0", label="solid, not ridable"),
          plt.Line2D([], [], color=INK2, marker=">", ls="-",
                     label=f"field steepest descent at z = {zmed:.0f}")]
    ax.legend(handles=h, loc="best", frameon=True, framealpha=0.92,
              fontsize=8)
    ax.grid(True, color=GRID, lw=0.5, zorder=1)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    plt.close(fig)
    print(a.out)


if __name__ == "__main__":
    main()
