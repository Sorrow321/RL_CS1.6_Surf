#!/usr/bin/env python3
"""plot_cornerdiag.py - the figures for round 37 (the corner diagnosis).

Four figures, all from JSON the measurement tools already wrote:

  potential_vs_time.png  the headline the user asked for: the goal field
                         sampled at the agent's OWN position at every
                         decision, against episode time, oriented so that
                         PROGRESS IS UP and a setback is a DIP. Two
                         normalisations, one per row (never two y-scales on
                         one axis): raw map units of geodesic ground banked,
                         and REWARD UNITS (scale = 100/d0, so the whole map
                         is worth 100 on either map - the only fair
                         cross-map axis). Deaths marked; the break-even
                         slope (banking exactly enough per second to pay the
                         time penalty) drawn as the reference.
  dip_ladder.png         every DIP in those curves, depth against duration,
                         with the GAE weight that survives at that delay.
  deception_<map>.png    M1: along the reference line, the angle between the
                         field's steepest descent and the flyable tangent,
                         the share of the field's own descent that runs over
                         nothing ridable, and the d-per-unit ratio of the
                         two lines.
  branch_<map>.png       M2: the two branches at the critical point, per
                         decision, on a common time axis.

Palette: the dataviz skill's reference categorical palette, slots taken in
fixed order (the validator needs node, which this box does not have, so the
palette is used UNMODIFIED rather than re-derived).

    python tools/plot_cornerdiag.py --dips runs/research/cornerdiag/dips \
        --m1 runs/research/cornerdiag --m2 runs/research/cornerdiag \
        --out docs/img
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

# dataviz skill, references/palette.md - categorical slots 1..8, in order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
     "#e87ba4", "#008300", "#4a3aa7", "#8a5a2b"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# curve -> (colour slot, linestyle, label)
STYLE = {
    "cb_exitABS_r9_FINISH":  (0, "-",  "exitABS r9 (ours, FINISHES 70.7 s)"),
    "cb_cySPINEW_FINISH":    (2, "-",  "cySPINEW (ours, FINISHES 75.7 s)"),
    "cb_beam_68.54_FINISH":  (6, "--", "searched line 68.54 s (FINISHES)"),
    "cb_cyKEYPOT_WALL":      (1, "-",  "cyKEYPOT (wall, dies)"),
    "cb_cyPOTNC_WALL":       (3, "-",  "cyPOTNC (wall, dies)"),
    "cb_cyRATCH_WALL":       (4, "-",  "cyRATCH (wall, dies)"),
    "pt_WRdemo_FINISH":      (0, "-",  "human WR demo (FINISHES 29.9 s)"),
    "pt_prRATCH_DIES":       (1, "-",  "prRATCH (dies 8.3 s)"),
    "pt_pdKEYPOT_DIES":      (3, "-",  "pdKEYPOT (dies 6.8 s)"),
}


def _panel(ax, curves, names, reward_units, title, tmax):
    for nm in names:
        c = curves[nm]
        slot, ls, lab = STYLE[nm]
        t = np.asarray(c["t"])
        d = np.asarray(c["d"])
        y = (c["d0"] - d) * (c["scale"] if reward_units else 1.0)
        ax.plot(t, y, ls, color=C[slot], lw=1.6, label=lab, zorder=3)
        if not c.get("finished"):
            ax.plot(t[-1], y[-1], "x", color=C[slot], ms=9, mew=2.2, zorder=4)
    ax.set_xlim(0, tmax)
    ax.grid(True, axis="y", zorder=0)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("episode time (s)")
    if reward_units:
        d0 = curves[names[0]]["d0"]
        # the break-even slope: banking this fast exactly pays the time
        # penalty (time_pen_tick per tick = 0.5 reward/s at 0.005/10 ms)
        tp = curves[names[0]]["time_pen_tick"]
        tms = curves[names[0]]["tick_ms"]
        rate = tp * 1000.0 / tms
        ax.plot([0, tmax], [0, rate * tmax], ":", color=INK2, lw=1.2,
                zorder=1)
        ax.annotate(f"break-even {rate:.2f} reward/s\n(the time penalty)",
                    xy=(tmax * 0.62, rate * tmax * 0.62),
                    xytext=(tmax * 0.40, rate * tmax * 0.86),
                    color=INK2, fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_ylabel(f"reward units banked  (100/d0, d0 = {d0:,.0f} u)")
        ax.set_ylim(0, 105)
    else:
        ax.set_ylabel("geodesic ground banked  d0 - d  (map units)")


def _depth_panel(ax, curves, names, title, tmax):
    for nm in names:
        c = curves[nm]
        slot, ls, lab = STYLE[nm]
        t = np.asarray(c["t"])
        ax.plot(t, np.asarray(c["depth"]), ls, color=C[slot], lw=1.6,
                label=lab, zorder=3)
        if not c.get("finished"):
            ax.plot(t[-1], c["depth"][-1], "x", color=C[slot], ms=9,
                    mew=2.2, zorder=4)
    ax.set_xlim(0, tmax)
    ax.set_ylim(-0.15, 4.6)
    ax.grid(True, axis="y", zorder=0)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("dip depth  (d - running min) * 100/d0")


def fig_potential(curves, out: Path):
    cb = [n for n in STYLE if n.startswith("cb_") and n in curves]
    pt = [n for n in STYLE if n.startswith("pt_") and n in curves]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0))
    _panel(axes[0][0], curves, cb, False,
           "A  surf_src_cannonball - raw map units", 80)
    _panel(axes[0][1], curves, pt, False,
           "B  surf_petrus_lite - raw map units", 33)
    _panel(axes[1][0], curves, cb, True,
           "C  surf_src_cannonball - REWARD units (100 = the whole map)", 80)
    _panel(axes[1][1], curves, pt, True,
           "D  surf_petrus_lite - REWARD units (100 = the whole map)", 33)
    _depth_panel(axes[2][0], curves, cb,
                 "E  surf_src_cannonball - the DIP itself", 80)
    _depth_panel(axes[2][1], curves, pt,
                 "F  surf_petrus_lite - the DIP itself (there is none)", 33)
    for ax in axes[:2].ravel():
        ax.legend(loc="upper left", frameon=False)
    axes[2][0].legend(loc="upper left", frameon=False)
    axes[2][1].annotate("no episode on this map, winner or loser, ever\n"
                        "gives back more than 0.141 reward units",
                        xy=(6, 2.2), fontsize=9, color=INK2)
    fig.suptitle("The goal field at the agent's own position, against time. "
                 "Progress is UP; a setback is a DIP. x = death.",
                 y=0.992, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = out / "potential_vs_time.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_zoom(curves, out: Path):
    """The two critical windows, side by side, in the SAME reward units and
    over the SAME 12-second span, so the two demands are directly
    comparable by eye."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    spans = [("cannonball", [n for n in STYLE if n.startswith("cb_")], 61.0),
             ("petrus", [n for n in STYLE if n.startswith("pt_")], 5.0)]
    for ax, (lab, names, t0) in zip(axes, spans):
        for nm in names:
            if nm not in curves:
                continue
            c = curves[nm]
            slot, ls, l = STYLE[nm]
            t = np.asarray(c["t"])
            y = np.asarray(c["depth"])
            m = (t >= t0) & (t <= t0 + 12.0)
            ax.plot(t[m] - t0, y[m], ls, color=C[slot], lw=1.8, label=l)
            if not c.get("finished") and t[-1] <= t0 + 12.0:
                ax.plot(t[-1] - t0, y[-1], "x", color=C[slot], ms=11, mew=2.4)
        ax.set_xlim(0, 12)
        ax.grid(True, axis="y", zorder=0)
        ax.set_xlabel(f"seconds after t = {t0:g} s")
        ax.set_title(f"{lab}: the dip at the critical point", loc="left")
        ax.legend(loc="center right", frameon=False, fontsize=8)
    axes[0].set_ylabel("dip depth (reward units)")
    axes[0].annotate("the finishers PAY 4.16-4.26 reward units\n"
                     "over 5.4-5.7 s, and recover",
                     xy=(0.15, 3.55), fontsize=9, color=INK2)
    axes[1].annotate("the WR's worst give-back anywhere on petrus is\n"
                     "0.141 reward units over 0.28 s; our two losers\n"
                     "give back NOTHING and die",
                     xy=(0.15, 3.55), fontsize=9, color=INK2)
    fig.suptitle("Same axis, same units: the tolerance each map's critical "
                 "point actually asks for", y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = out / "potential_zoom.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_dips(curves, out: Path):
    """Panel A puts BOTH maps on one axis, so identity is never colour alone:
    the MAP is the marker family (round = cannonball, square = petrus) and
    the arm is the colour; filled = survived, hollow = died inside it."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax, ax2 = axes
    MK = {"cb": ("o", "s"), "pt": ("s", "D")}      # (survived, failed)
    for nm, c in curves.items():
        if nm not in STYLE:
            continue
        slot, ls, lab = STYLE[nm]
        fam = MK["cb" if nm.startswith("cb_") else "pt"]
        for dp in c["dips"]:
            if dp["depth"] <= 0:
                continue
            surv = dp["survived"]
            ax.scatter(dp["secs"], dp["depth"], s=80 if surv else 120,
                       marker=fam[0] if surv else fam[1],
                       facecolor=(C[slot] if surv else "none"),
                       edgecolor=C[slot], linewidths=1.9, zorder=3)
    for nm in STYLE:
        if nm in curves:
            slot, ls, lab = STYLE[nm]
            mk = "o" if nm.startswith("cb_") else "s"
            tag = "cannonball" if nm.startswith("cb_") else "petrus"
            ax.plot([], [], mk, color=C[slot], ls="none",
                    label=f"{tag}: {lab}")
    ax.plot([], [], "o", color=INK2, ls="none", label="filled = SURVIVED")
    ax.plot([], [], "s", color=INK2, mfc="none", ls="none",
            label="hollow = FAILED (died inside it)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dip duration (s)")
    ax.set_ylabel("dip depth (reward units = % of d0)")
    ax.set_title("A  every dip on either map, depth against duration",
                 loc="left")
    ax.grid(True, which="both", zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=7)

    secs = np.linspace(0.02, 8.0, 400)
    for lam, slot in ((0.95, 0), (0.99, 1), (1.0, 2)):
        g = (0.9995 ** 4) * lam
        ax2.plot(secs, g ** (secs / 0.04), "-", color=C[slot], lw=1.8,
                 label=f"GAE lambda = {lam:g}")
    ax2.plot(secs, 0.9995 ** (secs * 100.0), ":", color=INK2, lw=1.4,
             label="pure discount gamma^ticks")
    for x, txt, va in ((0.68, "petrus corner\n0.57 rew over 0.68 s\n"
                              "-> 40.4% of the credit arrives", "bottom"),
                       (5.37, "cannonball ending\n4.16 rew over 5.37 s\n"
                              "-> 0.010% arrives", "top")):
        ax2.axvline(x, color=INK2, lw=1.0, ls="--", zorder=0)
        ax2.annotate(txt, xy=(x + 0.12, 2e-4 if va == "bottom" else 3e-2),
                     fontsize=8, color=INK2)
    ax2.set_yscale("log"); ax2.set_xlim(0, 8); ax2.set_ylim(1e-5, 1.4)
    ax2.set_xlabel("delay to the payoff (s)")
    ax2.set_ylabel("share of the payoff reaching the first action")
    ax2.set_title("B  what survives the trainer's own credit path "
                  "(act_every 4, 10 ms)", loc="left")
    ax2.grid(True, which="both", zorder=0)
    ax2.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    p2 = out / "dip_ladder.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p2


def fig_deception(m1, label, out: Path, mark=None):
    R = m1["rows"]
    ap = np.array([r["arc_pct"] for r in R])
    ang = np.array([r["angle"] if r["angle"] is not None else np.nan
                    for r in R])
    void = 100.0 * np.array([r["field_void_surfy"] for r in R])
    rvoid = 100.0 * np.array([r["ref_support"] is None for r in R], float)
    fe = np.array([r["field_eff"] if r["field_eff"] is not None else np.nan
                   for r in R])
    re = np.array([r["ref_eff"] if r["ref_eff"] is not None else np.nan
                   for r in R])
    dip = np.array([r["dip"] for r in R])

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.6), sharex=True)
    axes[0].plot(ap, ang, "-", color=C[0], lw=1.4)
    axes[0].axhline(90, color=GRID, lw=1)
    axes[0].set_ylabel("angle (deg)")
    axes[0].set_title("A  angle between the field's steepest descent and the "
                      "flyable line's tangent", loc="left")
    axes[1].plot(ap, void, "-", color=C[1], lw=1.4,
                 label="the FIELD's own descent")
    axes[1].plot(ap, rvoid, "-", color=C[2], lw=1.4,
                 label="the REFERENCE line itself")
    axes[1].set_ylabel("% over nothing ridable")
    axes[1].set_title("B  share with no ridable surface (0.1 <= |n_z| <= 0.7) "
                      "within 192 u", loc="left")
    axes[1].legend(loc="upper left", frameon=False)
    axes[2].plot(ap, fe / np.maximum(re, 1e-9), "-", color=C[3], lw=1.4)
    axes[2].axhline(1.0, color=GRID, lw=1)
    axes[2].set_ylabel("field / reference")
    axes[2].set_title("C  d banked per unit of path travelled - above 1 the "
                      "reward PREFERS the field's line", loc="left")
    axes[3].plot(ap, dip, "-", color=C[4], lw=1.6)
    axes[3].set_ylabel("dip (reward units)")
    axes[3].set_xlabel("arc along the reference line (%)")
    axes[3].set_title("D  the reference line's OWN dip depth - the tolerance "
                      "the map asks a follower for", loc="left")
    for ax in axes:
        ax.grid(True, axis="y", zorder=0)
        if mark:
            for x0, x1, lab in mark:
                ax.axvspan(x0, x1, color=C[1], alpha=0.10, zorder=0)
    if mark:
        axes[0].annotate(mark[0][2], xy=(mark[0][1], axes[0].get_ylim()[1]),
                         xytext=(mark[0][1] + 1.5,
                                 axes[0].get_ylim()[1] * 0.85),
                         color=INK2, fontsize=8)
    fig.suptitle(f"Deception profile - {label}", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = out / f"deception_{label}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_branch(B, label, out: Path):
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 8.6), sharex="col")
    for col, key in enumerate(("naive", "correct")):
        b = B[key]
        t = np.asarray(b["t"])
        slot = 1 if key == "naive" else 0
        axes[0][col].plot(t, b["cum"], "-", color=C[slot], lw=1.8)
        axes[0][col].axhline(0, color=GRID, lw=1)
        axes[0][col].set_ylabel("cumulative reward")
        axes[0][col].set_title(
            f"{'AB'[col]}  {key.upper()} branch - "
            f"{Path(B['meta'][key].split('#')[0]).parent.name}", loc="left")
        axes[1][col].plot(t, b["speed"], "-", color=C[slot], lw=1.5,
                          label="speed")
        axes[1][col].plot(t, np.abs(b["vz"]), "-", color=C[3], lw=1.0,
                          label="|vz|")
        axes[1][col].set_ylabel("u/s")
        axes[1][col].legend(loc="upper left", frameon=False)
        sup = np.array([np.nan if s is None else s for s in b["support"]])
        gap = np.where(np.isnan(sup), 300.0, sup)
        axes[2][col].plot(t, gap, "-", color=C[slot], lw=1.5)
        axes[2][col].fill_between(t, 0, 300, where=np.isnan(sup),
                                  color=C[1], alpha=0.15, step="mid")
        axes[2][col].set_ylabel("ridable below (u); shaded = NONE")
        axes[2][col].set_xlabel("seconds after the branch point")
        for ax in (axes[0][col], axes[1][col], axes[2][col]):
            ax.grid(True, axis="y", zorder=0)
    fig.suptitle(f"The two branches at {label} (arc "
                 f"{B['meta']['branch_arc_pct']:g}%)", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = out / f"branch_{label}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p



def fig_credit(cr, label, out: Path, mark_t=None):
    """M2b: one branch, per decision - the reward, the critic, the empirical
    return, and the advantage the trainer would assign."""
    t = np.asarray(cr["t"])
    fig, ax = plt.subplots(5, 1, figsize=(11.5, 11.6), sharex=True)
    ax[0].plot(t, cr["d"], "-", color=C[0], lw=1.6)
    ax[0].set_ylabel("geodesic d (u)")
    ax[0].set_title("A  distance to the finish (monotone: no dip anywhere)",
                    loc="left")
    ax[1].plot(t, cr["cum"], "-", color=C[2], lw=1.6, label="cumulative")
    ax[1].plot(t, np.asarray(cr["reward"]) * 20.0, "-", color=C[3], lw=1.0,
               label="per-decision reward x20")
    ax[1].axhline(0, color=GRID, lw=1)
    ax[1].set_ylabel("reward units")
    ax[1].set_title("B  the reward it is actually being paid", loc="left")
    ax[1].legend(loc="upper left", frameon=False)
    ax[2].plot(t, cr["V"], "-", color=C[0], lw=1.8, label="V(s) - the critic")
    ax[2].plot(t, cr["G"], "--", color=C[1], lw=1.4,
               label="G - the realised discounted return")
    ax[2].set_ylabel("reward units")
    ax[2].set_title("C  the critic against the truth of THIS episode",
                    loc="left")
    ax[2].legend(loc="upper right", frameon=False)
    ax[3].fill_between(t, cr["A_trainer_lo"], cr["A_trainer_hi"],
                       color=C[0], alpha=0.18, lw=0,
                       label="trainer GAE, buffer-phase envelope")
    ax[3].plot(t, cr["A_trainer"], "-", color=C[0], lw=1.6,
               label=f"trainer GAE (lambda {cr['lam']:g}, "
                     f"n_steps {cr['n_steps']})")
    ax[3].plot(t, cr["A_lambda1"], "-", color=C[1], lw=1.2,
               label="lambda = 1, whole episode (G - V)")
    ax[3].axhline(0, color=GRID, lw=1)
    ax[3].set_ylabel("advantage")
    ax[3].set_title("D  the gradient signal at each decision", loc="left")
    ax[3].legend(loc="upper left", frameon=False, fontsize=7.5)
    sup = np.array([np.nan if v is None else v for v in cr["support"]])
    ax[4].plot(t, cr["speed"], "-", color=C[3], lw=1.6, label="speed (u/s)")
    ax4b = ax[4]
    ax4b.fill_between(t, 0, np.nanmax(cr["speed"]) * 1.05,
                      where=np.isnan(sup), color=C[1], alpha=0.15, step="mid",
                      label="nothing RIDABLE within 192 u")
    ax4b.set_ylabel("u/s")
    ax4b.set_xlabel("episode time (s)")
    ax4b.set_title("E  speed, and where the ramp runs out", loc="left")
    ax4b.legend(loc="upper left", frameon=False)
    for a_ in ax:
        a_.grid(True, axis="y", zorder=0)
        if mark_t is not None:
            a_.axvline(mark_t, color=INK2, lw=1.0, ls=":")
    fig.suptitle(f"{label}: per decision, with the critic and the advantage",
                 y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p2 = out / f"credit_{label}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dips", default="runs/research/cornerdiag/dips")
    ap.add_argument("--m1", default="runs/research/cornerdiag")
    ap.add_argument("--m2", default="runs/research/cornerdiag")
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    curves = {c["name"]: c
              for c in json.loads((Path(a.dips) / "curves.json")
                                  .read_text(encoding="utf-8"))}
    made = [fig_potential(curves, out), fig_zoom(curves, out),
            fig_dips(curves, out)]
    for f, lab, mk in (("m1_petrus_wr.json", "petrus",
                        [(16.3, 18.6, "the L-bend:\nprRATCH leaves here")]),
                       ("m1_cannonball_exitABS.json", "cannonball",
                        [(88.0, 95.0, "the 88.8% wall")])):
        p = Path(a.m1) / f
        if p.exists():
            made.append(fig_deception(
                json.loads(p.read_text(encoding="utf-8")), lab, out, mk))
    for f, lab in (("m2_petrus.json", "petrus"),
                   ("m2_cannonball.json", "cannonball")):
        p = Path(a.m2) / f
        if p.exists():
            made.append(fig_branch(
                json.loads(p.read_text(encoding="utf-8")), lab, out))
    for f, lab, mt in (("m2v/credit_prRATCH_ctl.json", "prRATCH", 6.76),
                       ("m2v/credit_cyPOTNC_ctl.json", "cyPOTNC", 67.84)):
        p = Path(a.m2) / f
        if p.exists():
            made.append(fig_credit(
                json.loads(p.read_text(encoding="utf-8")), lab, out, mt))
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
