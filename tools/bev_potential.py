#!/usr/bin/env python3
"""bev_potential.py - the potential field painted on the room where a policy
dies, with the policy's dying line and a human record over it, and the
potential-versus-time of both around the death.

Two figures from one call (user's ask, 2026-09-12, celestial):

  <out>_bev.png        bird's-eye view of the window around the death: the
                       goal potential (min over the z-slab the two lines
                       occupy, reachable cells only) as a heat map, solid
                       geometry in that slab hatched over it, the field's
                       own steepest-descent arrows at the slab's median
                       height, the record line and the policy's line with a
                       marker every second, and the ticks at which the map
                       PUSHED BACK on each line (vertical acceleration off
                       the gravity step - the ramp contacts) drawn bold.
  <out>_potential.png  potential versus time, both lines, clock centred on
                       the policy's death (t = 0) and bounded by +-window s;
                       the record's clock is shifted so that the two lines
                       coincide where the policy LEAVES the record line (the
                       last tick within --divergence-u of it), which is
                       what "the same place" means when the two do not run
                       at the same speed. The z (height) of both underneath.

--mirror-y C adds the record line MIRRORED about the plane y = C (dashed):
a fork whose two branches are mirror images lets the record, taken on one
branch, be compared with a policy that took the other. Find C with the
surface-voxel mirror test in the ledger (celestial: y = 552, IoU 0.995).

Both trajectories are record_ckpt-format .jsonl (tools/demo/parse_hldemo.py
writes a human demo in that format). The field comes out of the baked
goal_<cell>.npz next to the .bsp and is never built here; the occupancy out
of occ_<occ-cell>.npz. Nothing bakes.

    python tools/bev_potential.py --map maps_pool/surf_src_celestial.bsp \
        --goal-cell 48 --occ-cell 32 \
        --agent runs/jt3ANCHU/traj_14901313536_celestial.jsonl \
        --wr <scratch>/celestial_wr/surf_src_celestial.jsonl \
        --mirror-y 552 --bev-window -7500 1500 -4200 5000 \
        --out docs/img/celestial_gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.goalfield import GoalField                  # noqa: E402
from deception_profile import descent_dir                # noqa: E402

C_WR, C_AG, C_MIR = "#2a78d6", "#eb6834", "#7b3fb8"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
GRAVITY = 800.0


def load_goal(bsp: Path, cell: int) -> GoalField:
    p = Path(f"{Path(bsp).with_suffix('')}.goal_{cell:g}.npz")
    if not p.exists():
        raise SystemExit(f"no baked field {p} - refusing to bake")
    z = np.load(p, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]),
                     z["mins"], float(z["cell"]), float(z["reach_max"]))


def load_occ(bsp: Path, cell: int):
    p = Path(f"{Path(bsp).with_suffix('')}.occ_{cell:g}.npz")
    if not p.exists():
        raise SystemExit(f"no baked occupancy {p}")
    z = np.load(p, allow_pickle=False)
    return z["occ"].astype(bool), np.asarray(z["mins"], np.float64), float(cell)


def load_episodes(path: Path):
    """Every episode's raw rows [tick, x, y, z, vx, vy, vz, yaw, ...] and its
    header (record_ckpt format: a JSON dict before the rows, one after)."""
    eps, hdrs, cur, hdr = [], [], [], None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                if cur:
                    eps.append(np.asarray(cur, np.float64)); hdrs.append(hdr)
                    cur, hdr = [], None
                if hdr is None:
                    hdr = json.loads(line)
                continue
            cur.append(json.loads(line))
    if cur:
        eps.append(np.asarray(cur, np.float64)); hdrs.append(hdr)
    return eps, hdrs


class Line:
    def __init__(self, rows: np.ndarray, hdr: dict | None, label: str):
        tm = float((hdr or {}).get("tick_ms", 10.0))
        self.t = rows[:, 0] * tm / 1000.0
        self.p = rows[:, 1:4].copy()
        self.v = rows[:, 4:7].copy()
        self.label = label
        az = np.gradient(self.v[:, 2], self.t)
        self.push = np.abs(az + GRAVITY) > 150.0      # the map pushed back

    def mirrored(self, c: float, label: str) -> "Line":
        m = Line.__new__(Line)
        m.t, m.label, m.push = self.t, label, self.push
        m.p = self.p.copy(); m.p[:, 1] = 2.0 * c - self.p[:, 1]
        m.v = self.v.copy(); m.v[:, 1] = -self.v[:, 1]
        return m

    def launch_before(self, t_end: float):
        """The last tick before t_end at which the map pushed back."""
        k = np.where(self.push & (self.t <= t_end))[0]
        if len(k) == 0:
            return None
        i = int(k[-1])
        v = self.v[i]
        return dict(t=float(self.t[i]), pos=self.p[i], v=v, speed=float(np.linalg.norm(v)),
                    heading=float(np.degrees(np.arctan2(v[1], v[0]))),
                    climb=float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))))


def fmt_launch(L: dict) -> str:
    return (f"t = {L['t']:.2f} s at ({L['pos'][0]:.0f}, {L['pos'][1]:.0f}, {L['pos'][2]:.0f}), "
            f"|v| = {L['speed']:.0f} u/s, heading {L['heading']:.0f} deg, climb {L['climb']:.0f} deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="the .bsp; caches sit next to it")
    ap.add_argument("--goal-cell", type=int, default=32)
    ap.add_argument("--occ-cell", type=int, default=32)
    ap.add_argument("--agent", required=True, help="the policy's recording (.jsonl)")
    ap.add_argument("--agent-episode", default=None,
                    help="episode index; default = the one whose end is nearest "
                         "the median of all ends (the typical death)")
    ap.add_argument("--agent-label", default="policy (greedy eval)")
    ap.add_argument("--wr", required=True, help="the record line (.jsonl)")
    ap.add_argument("--wr-label", default="human record")
    ap.add_argument("--mirror-y", type=float, default=None,
                    help="also draw the record mirrored about the plane y = C")
    ap.add_argument("--divergence-u", type=float, default=200.0,
                    help="the policy 'leaves' a line at the last tick within this distance of it")
    ap.add_argument("--window", type=float, default=10.0, help="seconds each side of the death")
    ap.add_argument("--bev-secs", type=float, default=6.0,
                    help="seconds of each line before the death drawn on the BEV")
    ap.add_argument("--wr-secs-after", type=float, default=None,
                    help="seconds of the record after the (aligned) death drawn; default bev-secs")
    ap.add_argument("--bev-window", nargs=4, type=float, default=None,
                    metavar=("X0", "X1", "Y0", "Y1"), help="override the BEV window")
    ap.add_argument("--zslab", nargs=2, type=float, default=None, metavar=("ZLO", "ZHI"),
                    help="override the z-slab the heat map and the solid mask are taken over")
    ap.add_argument("--pad", type=float, default=400.0, help="BEV window padding, u")
    ap.add_argument("--quiver-step", type=int, default=5, help="in goal cells")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, help="prefix; _bev.png and _potential.png are added")
    a = ap.parse_args()

    gf = load_goal(Path(a.map), a.goal_cell)
    occ, omins, ocell = load_occ(Path(a.map), a.occ_cell)

    weps, whdrs = load_episodes(Path(a.wr))
    W = Line(weps[0], whdrs[0], a.wr_label)
    eps, hdrs = load_episodes(Path(a.agent))
    if a.agent_episode is None:
        ends = np.asarray([e[-1, 1:4] for e in eps])
        k = int(np.argmin(np.linalg.norm(ends - np.median(ends, axis=0), axis=1)))
    else:
        k = int(a.agent_episode)
    A = Line(eps[k], hdrs[k], a.agent_label)
    t_death = float(A.t[-1])
    ad, wd = gf.sample(A.p), gf.sample(W.p)
    M = W.mirrored(a.mirror_y, f"{a.wr_label}, mirrored about y = {a.mirror_y:.0f}") \
        if a.mirror_y is not None else None
    md = gf.sample(M.p) if M is not None else None
    if md is not None:
        # a mirrored hull position can land a hair inside the wall the real
        # one skimmed; those unreachable samples would plot as spikes
        md = np.where(md < gf.reach_max - gf.cell, md, np.nan)

    # where the policy leaves the record line, and the clock offset there
    from scipy.spatial import cKDTree

    def leaves(L: Line):
        dist, idx = cKDTree(L.p).query(A.p)
        near = np.where(dist < a.divergence_u)[0]
        if len(near) == 0:
            return None
        i = int(near[-1])
        return i, int(idx[i]), float(L.t[int(idx[i])]) - float(A.t[i])

    div = leaves(W)
    if div is None:
        raise SystemExit("the policy is never within --divergence-u of the record line")
    i_div, j_div, shift = div            # record clock = policy clock + shift
    t_div = float(A.t[i_div])
    W.t_al = W.t - shift                 # record time on the policy's clock
    if M is not None:
        M.t_al = W.t_al
        divm = leaves(M)

    def at_time(tarr, darr, tq):
        return float(darr[int(np.argmin(np.abs(tarr - tq)))])

    span = t_death - t_div
    wd_death = at_time(W.t_al, wd, t_death)
    print(f"policy episode {k} of {len(eps)}: dies at t = {t_death:.2f} s, d = {ad[-1]:.0f} u "
          f"(best {ad.min():.0f} u at {A.t[int(np.argmin(ad))]:.2f} s)")
    print(f"leaves the record line at t = {t_div:.2f} s (last tick within {a.divergence_u:.0f} u; "
          f"the record was there at {W.t[j_div]:.2f} s, so it leads by {shift:+.2f} s), "
          f"d = {ad[i_div]:.0f} u, z = {A.p[i_div, 2]:.0f}, |v| = {np.linalg.norm(A.v[i_div]):.0f} u/s")
    print(f"from there to the death ({span:.2f} s): the policy descends the potential at "
          f"{(ad[i_div] - ad[-1]) / max(span, 1e-6):.0f} u/s, the record at "
          f"{(wd[j_div] - wd_death) / max(span, 1e-6):.0f} u/s over the same {span:.2f} s "
          f"(record d at the aligned death time: {wd_death:.0f} u)")
    if M is not None:
        if divm is None:
            print("the policy is never within --divergence-u of the MIRRORED record line")
        else:
            im, jm, _ = divm
            print(f"leaves the MIRRORED record line at t = {A.t[im]:.2f} s "
                  f"(mirrored record there at {M.t[jm]:.2f} s), d = {ad[im]:.0f} u")
    mwin = (W.t_al >= t_death - a.window) & (W.t_al <= t_death + a.window)
    dd, tt = wd[mwin], W.t_al[mwin]
    rises, i = [], 0
    while i < len(dd) - 1:
        if dd[i + 1] > dd[i]:
            j = i
            while j < len(dd) - 1 and dd[j + 1] >= dd[j] - 1e-6:
                j += 1
            rises.append((float(tt[i] - t_death), float(tt[j] - t_death), float(dd[j] - dd[i])))
            i = j
        else:
            i += 1
    if rises:
        big = max(rises, key=lambda r: r[2])
        print(f"largest RISE of the record's potential within +-{a.window:.0f} s of the death: "
              f"+{big[2]:.0f} u over t = {big[0]:+.2f}..{big[1]:+.2f} s")
    # the flattest second of the record (a plateau is a give-up too)
    if mwin.sum() > 100:
        rate = -np.gradient(dd, tt)
        kern = int(round(1.0 / max(np.median(np.diff(tt)), 1e-3)))
        sm = np.convolve(rate, np.ones(kern) / kern, mode="same")
        j = int(np.argmin(sm[kern:-kern])) + kern if len(sm) > 2 * kern else int(np.argmin(sm))
        print(f"slowest 1 s of the record's descent in the window: {sm[j]:.0f} u/s at "
              f"t = {tt[j] - t_death:+.2f} s (its mean in the window: {np.mean(rate):.0f} u/s)")
    La = A.launch_before(t_death)
    Lw = W.launch_before(t_death + shift)
    if La:
        print(f"policy's last ramp contact before the death: {fmt_launch(La)}")
    if Lw:
        s = fmt_launch(Lw)
        if M is not None:
            Lm = dict(Lw); Lm["heading"] = float(np.degrees(np.arctan2(-Lw['v'][1], Lw['v'][0])))
            Lm["pos"] = M.p[int(np.argmin(np.abs(M.t - Lw['t'])))]
            s += f"  [mirrored: heading {Lm['heading']:.0f} deg at ({Lm['pos'][0]:.0f}, {Lm['pos'][1]:.0f})]"
        print(f"record's last ramp contact before the aligned death: {s}")

    # ------------------------------------------------------------ BEV -----
    after = a.bev_secs if a.wr_secs_after is None else a.wr_secs_after
    ma = (A.t >= t_death - a.bev_secs)
    mw = (W.t_al >= t_death - a.bev_secs) & (W.t_al <= t_death + after)
    drawn = [(W, mw, wd, C_WR, "-")] + ([(M, mw, md, C_MIR, "--")] if M is not None else []) \
        + [(A, ma, ad, C_AG, "-")]
    allp = np.concatenate([L.p[mm] for L, mm, _d, _c, _s in drawn])
    if a.bev_window:
        x0, x1, y0, y1 = a.bev_window
    else:
        x0, y0 = allp[:, :2].min(0) - a.pad
        x1, y1 = allp[:, :2].max(0) + a.pad
    if a.zslab:
        zlo, zhi = a.zslab
    else:
        zlo, zhi = float(allp[:, 2].min()) - 64.0, float(allp[:, 2].max()) + 64.0
    zmed = float(np.median(allp[:, 2]))

    gc = float(gf.cell)
    gx = np.arange(x0, x1 + gc, gc)
    gy = np.arange(y0, y1 + gc, gc)
    GX, GY = np.meshgrid(gx, gy)
    heat = np.full(GX.shape, np.nan)
    for zz in np.arange(zlo, zhi + gc, gc):
        P = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, zz)], axis=1)
        d = gf.sample(P).reshape(GX.shape)
        d = np.where(d < gf.reach_max - gc, d, np.nan)
        heat = np.where(np.isnan(heat), d, np.where(np.isnan(d), heat, np.minimum(heat, d)))

    ix0 = max(0, int(np.floor((x0 - omins[0]) / ocell)))
    ix1 = min(occ.shape[2], int(np.ceil((x1 - omins[0]) / ocell)))
    iy0 = max(0, int(np.floor((y0 - omins[1]) / ocell)))
    iy1 = min(occ.shape[1], int(np.ceil((y1 - omins[1]) / ocell)))
    iz0 = max(0, int(np.floor((zlo - omins[2]) / ocell)))
    iz1 = min(occ.shape[0], int(np.ceil((zhi - omins[2]) / ocell)))
    solid = occ[iz0:iz1, iy0:iy1, ix0:ix1].any(0)
    oext = (omins[0] + ix0 * ocell, omins[0] + ix1 * ocell,
            omins[1] + iy0 * ocell, omins[1] + iy1 * ocell)

    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    ext = (gx[0] - gc / 2, gx[-1] + gc / 2, gy[0] - gc / 2, gy[-1] + gc / 2)
    vmin, vmax = np.nanpercentile(heat, 2), np.nanpercentile(heat, 98)
    im = ax.imshow(heat / 1000.0, origin="lower", extent=ext, aspect="equal",
                   cmap="viridis_r", vmin=vmin / 1000.0, vmax=vmax / 1000.0,
                   interpolation="nearest", zorder=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"goal potential d, thousand u (min over z {zlo:.0f}..{zhi:.0f}; "
                 "brighter = closer to the finish)")
    ax.imshow(np.where(solid, 1.0, np.nan), origin="lower", extent=oext,
              aspect="equal", cmap=matplotlib.colors.ListedColormap(["#6b6963"]),
              alpha=0.55, interpolation="nearest", zorder=1)

    q = a.quiver_step
    qx = np.arange(x0 + gc * q / 2, x1, gc * q)
    qy = np.arange(y0 + gc * q / 2, y1, gc * q)
    QX, QY = np.meshgrid(qx, qy)
    P = np.stack([QX.ravel(), QY.ravel(), np.full(QX.size, zmed)], axis=1)
    d = gf.sample(P)
    u = descent_dir(gf, P)
    ok = (d < gf.reach_max - gc) & (np.linalg.norm(u, axis=1) > 1e-6)
    ax.quiver(P[ok, 0], P[ok, 1], u[ok, 0], u[ok, 1], color=SURF, alpha=0.7,
              width=0.0022, scale=36, zorder=2)

    for L, mm, _d, col, ls in drawn:
        pts, tt = L.p[mm], (L.t_al if hasattr(L, "t_al") else L.t)[mm]
        ax.plot(pts[:, 0], pts[:, 1], ls, color=col, lw=2.6, label=L.label, zorder=4)
        pb = L.push[mm]
        ax.plot(pts[pb, 0], pts[pb, 1], ".", color=col, ms=9, zorder=5)
        for s in np.arange(np.ceil(tt[0]), tt[-1] + 1e-9, 1.0):
            i = int(np.argmin(np.abs(tt - s)))
            ax.plot(pts[i, 0], pts[i, 1], "o", color=col, ms=5, mec=SURF, mew=1.0, zorder=5)
            ax.annotate(f"{s - t_death:+.0f}s z{pts[i, 2]:.0f}", (pts[i, 0], pts[i, 1]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7,
                        color=col, zorder=6)
    ax.plot(A.p[-1, 0], A.p[-1, 1], "X", color=C_AG, ms=14, mew=2.2, zorder=6)
    ax.plot(A.p[i_div, 0], A.p[i_div, 1], "D", color=INK, ms=9, mfc="none", mew=2.0, zorder=6)
    ax.annotate("leaves the record line", (A.p[i_div, 0], A.p[i_div, 1]),
                textcoords="offset points", xytext=(8, -14), fontsize=8, color=INK, zorder=6)
    if a.mirror_y is not None:
        ax.axhline(a.mirror_y, color=INK, ls=":", lw=1.0, zorder=3)
        ax.text(x0 + 40, a.mirror_y + 30, f"mirror plane y = {a.mirror_y:.0f}", fontsize=8, color=INK)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("x (map units)"); ax.set_ylabel("y (map units)")
    ax.set_title(a.title or f"{Path(a.map).stem}: the potential under the death, "
                 f"the record line and the policy's line (clock: seconds to the death)",
                 loc="left", fontsize=10)
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="#6b6963", alpha=0.55, label=f"solid somewhere in z {zlo:.0f}..{zhi:.0f}"),
          plt.Line2D([], [], color=INK2, marker=">", ls="-",
                     label=f"field steepest descent at z = {zmed:.0f}"),
          plt.Line2D([], [], color=INK2, marker=".", ms=9, ls="",
                     label="bold dots: the map pushes back (ramp contact)"),
          plt.Line2D([], [], color=INK, marker="D", mfc="none", ls="",
                     label=f"policy leaves the record line ({a.divergence_u:.0f} u)"),
          plt.Line2D([], [], color=C_AG, marker="X", ls="", label="death")]
    ax.legend(handles=h, loc="best", frameon=True, framealpha=0.92, fontsize=8)
    ax.grid(True, color=GRID, lw=0.4, zorder=1)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}_bev.png", dpi=150)
    plt.close(fig)
    print(f"{out}_bev.png")

    # ------------------------------------------------- potential vs time --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    lo, hi = -a.window, a.window
    mA = (A.t - t_death >= lo) & (A.t - t_death <= hi)
    ax1.plot(W.t_al[mwin] - t_death, wd[mwin] / 1000.0, "-", color=C_WR, lw=2.2,
             label=f"{a.wr_label} (clock aligned where the policy leaves its line)")
    if M is not None:
        ax1.plot(M.t_al[mwin] - t_death, md[mwin] / 1000.0, "--", color=C_MIR, lw=1.8,
                 label=f"{M.label} (the field on the policy's side)")
    ax1.plot(A.t[mA] - t_death, ad[mA] / 1000.0, "-", color=C_AG, lw=2.2, label=a.agent_label)
    for L, col in ((W, C_WR), (A, C_AG)):
        tt = (L.t_al if hasattr(L, "t_al") else L.t)
        mm = (tt - t_death >= lo) & (tt - t_death <= hi) & L.push
        dcol = wd if L is W else ad
        ax1.plot(tt[mm] - t_death, dcol[mm] / 1000.0, ".", color=col, ms=7)
    ax1.axvline(t_div - t_death, color=INK, ls=":", lw=1.2)
    ax1.axvline(0.0, color=C_AG, ls="--", lw=1.2)
    y_top = ax1.get_ylim()[1]
    ax1.text(t_div - t_death, y_top, " leaves the record line", va="top", fontsize=8, color=INK)
    ax1.text(0.0, y_top, " death", va="top", fontsize=8, color=C_AG)
    ax1.set_ylabel("goal potential d (thousand u)")
    ax1.set_title(a.title or f"{Path(a.map).stem}: potential over time, +-{a.window:.0f} s "
                  f"around the policy's death (bold dots: ramp contact)", loc="left", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92)
    ax1.grid(True, color=GRID, lw=0.5)
    ax2.plot(W.t_al[mwin] - t_death, W.p[mwin, 2], "-", color=C_WR, lw=1.8)
    ax2.plot(A.t[mA] - t_death, A.p[mA, 2], "-", color=C_AG, lw=1.8)
    for L, col in ((W, C_WR), (A, C_AG)):
        tt = (L.t_al if hasattr(L, "t_al") else L.t)
        mm = (tt - t_death >= lo) & (tt - t_death <= hi) & L.push
        ax2.plot(tt[mm] - t_death, L.p[mm, 2], ".", color=col, ms=7)
    ax2.axvline(t_div - t_death, color=INK, ls=":", lw=1.2)
    ax2.axvline(0.0, color=C_AG, ls="--", lw=1.2)
    ax2.set_ylabel("height z (u)")
    ax2.set_xlabel("seconds relative to the policy's death")
    ax2.grid(True, color=GRID, lw=0.5)
    ax2.set_xlim(lo, hi)
    fig.tight_layout()
    fig.savefig(f"{out}_potential.png", dpi=150)
    plt.close(fig)
    print(f"{out}_potential.png")


if __name__ == "__main__":
    main()
