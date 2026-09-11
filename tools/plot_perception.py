#!/usr/bin/env python3
"""plot_perception.py - the M6 figures, from perception_probe.py's dumps.

Pure numpy + matplotlib over ``m6_frames.npz`` and ``m6_table.json``; no
GPU, no map, no bake, so it can run while a trainer owns the card.

    python tools/plot_perception.py --in runs/research/petrusperc/m6 \
        --out docs/img --stem petrus_bend
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Rectangle                         # noqa: E402

ENC_MAX = 1.25          # the depth encoding's clear value with a near range
WALL_NZ, GROUND_NZ = 0.1, 0.7


def outline(ax, m, color="#ff2d55", lw=0.9):
    """Draw the boundary of a boolean pixel mask as cell edges."""
    H, W = m.shape
    for r in range(H):
        for c in range(W):
            if not m[r, c]:
                continue
            if r == 0 or not m[r - 1, c]:
                ax.plot([c - .5, c + .5], [r - .5, r - .5], color=color, lw=lw)
            if r == H - 1 or not m[r + 1, c]:
                ax.plot([c - .5, c + .5], [r + .5, r + .5], color=color, lw=lw)
            if c == 0 or not m[r, c - 1]:
                ax.plot([c - .5, c - .5], [r - .5, r + .5], color=color, lw=lw)
            if c == W - 1 or not m[r, c + 1]:
                ax.plot([c + .5, c + .5], [r - .5, r + .5], color=color, lw=lw)


def show(ax, img, kind):
    if kind == "depth":
        ax.imshow(1.0 - np.clip(img / ENC_MAX, 0, 1), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    elif kind == "mask":
        ax.imshow(img, cmap="turbo", vmin=0.0, vmax=1.0,
                  interpolation="nearest", aspect="auto")
    else:
        ax.imshow(img, cmap="RdBu", vmin=-3, vmax=3,
                  interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--stem", default="petrus_bend")
    ap.add_argument("--strip-at", default="6.40,6.64,6.76,6.80,6.88,6.96,7.08,7.20")
    a = ap.parse_args()
    src = Path(a.inp)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(src / "m6_frames.npz")
    J = json.load(open(src / "m6_table.json", encoding="utf-8"))
    R, M = J["rows"], J["meta"]
    t = z["t"]

    # ---------------------------------------------------- 1. the frame strip
    want = [float(x) for x in a.strip_at.split(",")]
    idx = [int(np.argmin(np.abs(t - w))) for w in want]
    n = len(idx)
    fig, axes = plt.subplots(3, n, figsize=(2.05 * n, 5.4))
    for j, k in enumerate(idx):
        rm = z["on_ramp"][k]
        for i, (kind, arr) in enumerate((("depth", z["depth"][k]),
                                         ("mask", z["mask"][k]),
                                         ("pot", z["pot"][k]))):
            ax = axes[i, j]
            show(ax, arr, kind)
            if rm.any():
                outline(ax, rm)
            if i == 0:
                ax.set_title(f"t {t[k]:.2f}s\n{int(rm.sum())} px",
                             fontsize=8.5)
            if j == 0:
                ax.set_ylabel({"depth": "DEPTH\n(what it has)",
                               "mask": "SURF MASK\n|n_z|  (--surf-mask)",
                               "pot": "POTENTIAL\nnorm (on today)"}[kind],
                              fontsize=8)
    fig.suptitle("surf_petrus_lite, prRATCH's own greedy episode: the "
                 "surviving ramp (outlined) in the three channels\n"
                 "64x32 = 2,048 px, 120x90 deg FOV. It is inside the FOV "
                 "throughout and OCCLUDED until t = 6.76 s; the branch is "
                 "at 6.80 s and it dies at 8.28 s.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / f"{a.stem}_frames.png", dpi=170)
    plt.close(fig)

    # ------------------------------------------------ 2. the surf-mask crux
    kb = int(np.argmin(np.abs(t - 7.04)))
    rm = z["on_ramp"][kb]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 2.9))
    show(axes[0], z["depth"][kb], "depth")
    outline(axes[0], rm)
    r = R[kb]
    axes[0].set_title("DEPTH (today's channel)\nramp %.3f vs wall %.3f  "
                      "= %+.2f sigma" % (r["ramp_depth_mean"],
                                         r["wall_depth_mean"],
                                         r["depth_vs_wall_sigma"]),
                      fontsize=9)
    show(axes[1], z["mask"][kb], "mask")
    outline(axes[1], rm, color="#111111")
    axes[1].set_title("SURF MASK |n_z| (--surf-mask 1)\nramp %.3f vs wall "
                      "%.3f  = %+.2f sigma" % (r["ramp_mask_mean"],
                                               r["wall_mask_mean"],
                                               r["mask_vs_wall_sigma"]),
                      fontsize=9)
    ridable = ((z["mask"][kb] >= WALL_NZ) & (z["mask"][kb] <= GROUND_NZ) &
               ~z["sky"][kb])
    axes[2].imshow(ridable, cmap="Greys_r", vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    outline(axes[2], rm)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title("the mask THRESHOLDED to the ridable band\n"
                      "%.0f%% of ramp px vs %.0f%% of wall px" %
                      (100 * r["ramp_surfy_frac"], 100 * r["wall_surfy_frac"]),
                      fontsize=9)
    fig.suptitle("t = %.2f s, 0.24 s after the branch: what --surf-mask "
                 "adds that depth cannot supply" % t[kb], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / f"{a.stem}_surfmask.png", dpi=170)
    plt.close(fig)

    # ------------------------------------------------------ 3. the timeseries
    tt = np.array([r["t"] for r in R])
    ang = np.array([r["ang_n_pix_angular"] for r in R], float)
    ren = np.array([r["ramp_px"] for r in R], float)
    occ = np.array([100 * r.get("ang_occ_frac", np.nan) for r in R])
    gap = np.array([r.get("ang_occ_gap_med", np.nan) for r in R])

    def col(k):
        return np.array([r.get(k, np.nan) for r in R], float)

    fig, ax = plt.subplots(4, 1, figsize=(9.2, 9.4), sharex=True)
    ax[0].plot(tt, ang, "-", color="#888", label="ANGULAR: px the ramp would "
               "fill with no occluder")
    ax[0].plot(tt, ren, "-", color="#c62828", lw=2,
               label="RENDERED: px the march actually returns")
    ax[0].axvline(6.80, color="#1565c0", ls="--", lw=1)
    ax[0].axvline(8.28, color="k", ls=":", lw=1)
    ax[0].set_ylabel("pixels of 2,048")
    ax[0].legend(fontsize=8)
    ax[0].set_title("M6: the surviving ramp is in the FOV the whole approach "
                    "and OCCLUDED until the branch decision itself",
                    fontsize=10)
    ax[1].plot(tt, occ, "-", color="#6a1b9a", lw=2)
    ax[1].set_ylabel("% of the ramp's own\npixels occluded")
    ax[1].axvline(6.80, color="#1565c0", ls="--", lw=1)
    ax2 = ax[1].twinx()
    ax2.plot(tt, gap, "-", color="#ef6c00", lw=1)
    ax2.set_ylabel("median gap to the\noccluder (u)", color="#ef6c00",
                   fontsize=8)
    ax[2].plot(tt, col("depth_vs_wall_sigma"), "-o", ms=2.5, color="#37474f",
               label="DEPTH: ramp vs not-ridable beside it")
    ax[2].plot(tt, col("mask_vs_wall_sigma"), "-o", ms=2.5, color="#2e7d32",
               label="SURF MASK |n_z|")
    ax[2].plot(tt, col("pot_contrast_sigma"), "-o", ms=2.5, color="#1565c0",
               label="POTENTIAL norm")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].axvline(6.80, color="#1565c0", ls="--", lw=1)
    ax[2].set_ylabel("contrast, in local\npixel sigma")
    ax[2].legend(fontsize=8)
    ax[3].plot(tt, col("yaw_minus_heading"), "-", color="#ad1457", lw=1.4)
    ax[3].axhline(0, color="k", lw=0.6)
    ax[3].axvline(6.80, color="#1565c0", ls="--", lw=1)
    ax[3].set_ylabel("camera yaw minus\nheading(v), deg")
    ax[3].set_xlabel("seconds of prRATCH's own greedy episode "
                     "(branch 6.80 s dashed, death 8.28 s dotted)")
    fig.tight_layout()
    fig.savefig(out / f"{a.stem}_timeseries.png", dpi=170)
    plt.close(fig)

    # ---------------------------------------------------- 4. the yaw sweep
    swt, swo = z["sw_t"], z["sw_off"]
    times = sorted(set(swt.tolist()))
    offs = sorted(set(swo.tolist()))
    fig, axes = plt.subplots(len(times), len(offs),
                             figsize=(1.5 * len(offs), 1.35 * len(times)))
    axes = np.atleast_2d(axes)
    for i, tq in enumerate(times):
        for j, o in enumerate(offs):
            k = int(np.flatnonzero((swt == tq) & (swo == o))[0])
            ax = axes[i, j]
            show(ax, z["sw_mask"][k], "mask")
            if z["sw_ramp"][k].any():
                outline(ax, z["sw_ramp"][k], color="#111111", lw=0.7)
            if i == 0:
                ax.set_title(f"{o:+.0f} deg", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"t {tq:.2f}s", fontsize=8.5)
            ax.set_xlabel(f"{int(z['sw_ramp'][k].sum())} px", fontsize=7.5,
                          labelpad=1)
    fig.suptitle("the 90-degree question: the SURF MASK at "
                 "heading(v) + offset (positive = left).\n"
                 "At the branch (6.80 s) NO camera offset reveals the ramp; "
                 "0.4 s later, +45/+90 deg reveals 1.6-3.1x more than "
                 "looking along velocity - which the policy never does "
                 "(|yaw - heading| <= 1.8 deg, always).", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / f"{a.stem}_yawsweep.png", dpi=170)
    plt.close(fig)
    print("wrote", ", ".join(str(out / f"{a.stem}_{s}.png") for s in
                             ("frames", "surfmask", "timeseries", "yawsweep")))


if __name__ == "__main__":
    main()
