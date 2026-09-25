"""primitive_coverage.py - how many numbers does a motion primitive need to cover a world record?

ANALYSIS ONLY (CLAUDE.md section 0: demos are for insight, never for training). Cuts each record's
flight into windows of W seconds and asks, per window, how well a primitive family redraws the path.

A path that leaves along its velocity is fixed by two functions of time: the heading's sideways
turn rate and its vertical turn rate. Each family is a model of those two functions, fitted to the
record's own heading and pitch by least squares; the path is then redrawn by integrating the
record's ACTUAL speed along the fitted heading, so the error measures the SHAPE a primitive can
express (a second number redraws it at the start speed, to show what speed changes add).

    straight   0 numbers   heading and pitch frozen at the start
    arc        2           constant sideways + constant vertical turn rate
    ours       4           delay d, then the sideways rate moves linearly from h1 to h2, a
                           constant vertical rate c (tools/viz_curve_sliders.py's family)
    quad       6           turn rates quadratic in time (3 + 3)
    cubic      8           turn rates cubic in time (4 + 4)

Per window it also labels the CONTACT: a sample touches a ramp or floor when its vertical
acceleration departs from free fall by more than 300 u/s^2 (the ramp's push), or the recorder's
onground flag is set; windows are "air", "surface" or "mixed".

    python tools/primitive_coverage.py [out_dir] [smooth_s] [label=path.npz|path.jsonl ...]

smooth_s > 0 first removes the air-strafe WEAVE (the left-right heading oscillation, ~1-2 Hz on
petrus / unitfarmer2) with a Gaussian of that sigma in seconds, so the families are fitted to the
ROUTE the weave rides along - what a planner specifies; the weave itself is the executor's.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RECORDS = {
    "cannonball": "runs/research/wr_demo/wr_cannonball.frames.npz",
    "unitfarmer2": "runs/research/uf2_wr/surf_unitfarmer2.frames.npz",
    "petrus": "runs/m1__research/cornerdiag/wr/petrus_wr.frames.npz",
}
DT = 0.01
WINDOWS = (1.0, 2.0, 3.0)
STRIDE = 0.25
G = 800.0
MODELS = ("straight", "arc", "ours", "quad", "cubic")
NPARAM = {"straight": 0, "arc": 2, "ours": 4, "quad": 6, "cubic": 8}
TOLS = (64.0, 128.0, 256.0)


def load(path):
    """-> a list of (T, P, V, OG) sequences on the DT grid: one for a demo's frames.npz, one per
    episode (longer than 3 s) for a recorder / eval trajectory .jsonl (rows tick,x,y,z,vx,vy,vz,..)."""
    if str(path).endswith(".jsonl"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        from surfgym.route import episodes_from_traj
        eps, hdrs = episodes_from_traj(str(path), with_headers=True)
        out = []
        for e, h in zip(eps, hdrs):
            a = np.asarray(e, np.float64)
            tick_s = float(h.get("tick_ms", 10.0)) / 1000.0
            if len(a) * tick_s < 3.0:
                continue
            out.append(_resample(a[:, 0] * tick_s, a[:, 1:4], a[:, 4:7], np.zeros(len(a))))
        return out
    z = np.load(path)
    return [_resample(z["time"], z["simorg"].astype(np.float64), z["simvel"].astype(np.float64),
                      z["onground"])]


def _resample(t, p, v, og):
    sp = np.linalg.norm(v, axis=1)
    mv = np.flatnonzero(sp > 50)
    a, b = mv[0], mv[-1]
    t, p, v, og = t[a:b + 1], p[a:b + 1], v[a:b + 1], og[a:b + 1]
    grid = np.arange(t[0], t[-1], DT)
    P = np.stack([np.interp(grid, t, p[:, k]) for k in range(3)], 1)
    V = np.stack([np.interp(grid, t, v[:, k]) for k in range(3)], 1)
    OG = np.interp(grid, t, (og > 0).astype(float)) > 0.5
    return grid - grid[0], P, V, OG


def contact(V, OG):
    az = np.gradient(V[:, 2], DT)
    az = np.convolve(az, np.ones(5) / 5, mode="same")        # 50 ms smoothing
    return (np.abs(az + G) > 300.0) | OG


def integrate(speed, psi, phi):
    d = np.stack([np.cos(phi) * np.cos(psi), np.cos(phi) * np.sin(psi), np.sin(phi)], 1)
    step = 0.5 * (speed[1:, None] * d[1:] + speed[:-1, None] * d[:-1]) * DT
    return np.vstack([np.zeros(3), np.cumsum(step, 0)])


def _lsq(cols, y):
    A = np.stack(cols, 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ c, c


def fit(x, dpsi, dphi, W):
    """-> {model: (psi_hat - psi0, phi_hat - phi0, params)}"""
    out = {"straight": (np.zeros_like(x), np.zeros_like(x), ())}
    hp, a = _lsq([x], dpsi)
    vp, b = _lsq([x], dphi)
    out["arc"] = (hp, vp, (a[0], b[0]))
    best = None
    for d in np.arange(0.0, 0.5 * W + 1e-9, 0.05):
        u = np.clip(x - d, 0.0, None)                       # time since the delay ended
        L = W - d
        H2 = u * u / (2.0 * L)                              # integral of f = u / L
        H1 = u - H2                                         # integral of (1 - f)
        hh, hc = _lsq([H1, H2], dpsi)
        vv, vc = _lsq([u], dphi)
        sse = float(((hh - dpsi) ** 2).sum() + ((vv - dphi) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, hh, vv, (d, hc[0], hc[1], vc[0]))
    out["ours"] = (best[1], best[2], best[3])
    for name, deg in (("quad", 3), ("cubic", 4)):
        cols = [x ** k for k in range(1, deg + 1)]
        hh, hc = _lsq(cols, dpsi)
        vv, vc = _lsq(cols, dphi)
        out[name] = (hh, vv, tuple(hc) + tuple(vc))
    return out


def analyse(name, path, smooth_s=0.0):
    rows, first, rev_n, rev_t = [], None, 0.0, 0.0
    for T, P, V, OG in load(path):
        r_, tr_ = _analyse_one(name, T, P, V, OG, smooth_s)
        rows += r_
        first = first or tr_
        rev_n += tr_["reversals"]
        rev_t += float(T[-1])
    first["reversals_per_s"] = rev_n / max(rev_t, 1e-9)
    return rows, first


def _analyse_one(name, T, P, V, OG, smooth_s):
    if smooth_s > 0:
        from scipy.ndimage import gaussian_filter1d
        P = np.stack([gaussian_filter1d(P[:, k], smooth_s / DT, mode="nearest") for k in range(3)], 1)
        V = np.gradient(P, DT, axis=0)
    C = contact(V, OG)
    sp = np.linalg.norm(V, axis=1)
    psi = np.unwrap(np.arctan2(V[:, 1], V[:, 0]))
    phi = np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]))
    rows = []
    for W in WINDOWS:
        n = int(round(W / DT)) + 1
        for i0 in range(0, len(T) - n, int(round(STRIDE / DT))):
            s = slice(i0, i0 + n)
            x = T[s] - T[i0]
            true = P[s] - P[i0]
            dpsi, dphi = psi[s] - psi[i0], phi[s] - phi[i0]
            fits = fit(x, dpsi, dphi, W)
            cfrac = float(C[s].mean())
            r = {"map": name, "W": W, "t0": float(T[i0]), "contact": cfrac,
                 "kind": "air" if cfrac < 0.05 else ("surface" if cfrac > 0.95 else "mixed"),
                 "speed": float(sp[i0]), "length": float(np.linalg.norm(np.diff(true, axis=0), axis=1).sum())}
            for m, (hp, vp, prm) in fits.items():
                est = integrate(sp[s], psi[i0] + hp, phi[i0] + vp)
                r[m] = float(np.linalg.norm(est - true, axis=1).max())
                if m == "ours":
                    est0 = integrate(np.full(n, sp[i0]), psi[i0] + hp, phi[i0] + vp)
                    r["ours_const_speed"] = float(np.linalg.norm(est0 - true, axis=1).max())
                    r["ours_params"] = prm
            rows.append(r)
    # the sideways turn rate itself: how often it changes sign (steering reversals per second)
    wr = np.gradient(psi, DT)
    wr_s = np.convolve(wr, np.ones(25) / 25, mode="same")    # 250 ms smoothing
    sig = np.sign(wr_s[np.abs(wr_s) > np.radians(10)])
    return rows, {"T": T, "P": P, "V": V, "C": C, "psi": psi, "phi": phi, "sp": sp,
                  "reversals": float((np.diff(sig) != 0).sum()), "turn_rate": np.degrees(wr_s)}


def main(out, smooth_s=0.0, sources=None):
    global RECORDS
    if sources:
        RECORDS = dict(sources)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    allrows, traces = [], {}
    for name, path in RECORDS.items():
        rows, tr = analyse(name, path, smooth_s)
        allrows += rows
        traces[name] = tr
    lines = [f"# Primitive coverage of: {', '.join(RECORDS)} (weave smoothing {smooth_s:g} s)\n",
             "Max position error of the redrawn path over each window, with the record's own speed profile. "
             "Coverage = share of windows redrawn within the tolerance.\n"]
    for W in WINDOWS:
        lines.append(f"\n## Windows of {W:g} s (stride {STRIDE:g} s)\n")
        lines.append("| map | windows | model | numbers | median err (u) | p90 err (u) | "
                     + " | ".join(f"<= {t:g} u" for t in TOLS) + " |")
        lines.append("|---|---|---|---|---|---|" + "---|" * len(TOLS))
        for name in RECORDS:
            rs = [r for r in allrows if r["map"] == name and r["W"] == W]
            for m in MODELS:
                e = np.array([r[m] for r in rs])
                cov = " | ".join(f"{100 * float((e <= t).mean()):.0f}%" for t in TOLS)
                lines.append(f"| {name} | {len(rs)} | {m} | {NPARAM[m]} | {np.median(e):.0f} | "
                             f"{np.percentile(e, 90):.0f} | {cov} |")
    lines.append("\n## The 4-number family at 2 s, split by contact\n")
    lines.append("| map | kind | windows | median err (u) | p90 (u) | <= 128 u | median err at constant speed (u) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in RECORDS:
        for kind in ("air", "surface", "mixed"):
            rs = [r for r in allrows if r["map"] == name and r["W"] == 2.0 and r["kind"] == kind]
            if not rs:
                continue
            e = np.array([r["ours"] for r in rs])
            e0 = np.array([r["ours_const_speed"] for r in rs])
            lines.append(f"| {name} | {kind} | {len(rs)} | {np.median(e):.0f} | {np.percentile(e, 90):.0f} | "
                         f"{100 * float((e <= 128).mean()):.0f}% | {np.median(e0):.0f} |")
    lines.append("\n## Fitted parameters of the 4-number family at 2 s (percentiles 5 / 25 / 50 / 75 / 95)\n")
    lines.append("| map | delay (s) | start turn rate (deg/s) | end turn rate (deg/s) | vertical rate (deg/s) | speed (u/s) | steering reversals per s |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in RECORDS:
        rs = [r for r in allrows if r["map"] == name and r["W"] == 2.0]
        pr = np.array([r["ours_params"] for r in rs])
        spd = np.array([r["speed"] for r in rs])
        q = lambda a, deg=False: " / ".join(f"{(np.degrees(v) if deg else v):.0f}" if not (not deg and a is pr[:, 0]) else f"{v:.2f}"
                                            for v in np.percentile(a, [5, 25, 50, 75, 95]))
        lines.append(f"| {name} | {q(pr[:, 0])} | {q(pr[:, 1], True)} | {q(pr[:, 2], True)} | "
                     f"{q(pr[:, 3], True)} | {q(spd)} | {traces[name]['reversals_per_s']:.2f} |")
    (out / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # figure 1: coverage curves at W = 2 s
    styles = {"straight": ("-", "o"), "arc": ("--", "s"), "ours": ("-", "^"), "quad": (":", "D"), "cubic": ("-.", "v")}
    fig, axs = plt.subplots(1, len(RECORDS), figsize=(5.7 * len(RECORDS), 5), sharey=True, squeeze=False)
    axs = axs[0]
    for ax, name in zip(axs, RECORDS):
        rs = [r for r in allrows if r["map"] == name and r["W"] == 2.0]
        tol = np.linspace(0, 1000, 201)
        for m in MODELS:
            e = np.array([r[m] for r in rs])
            cov = [(e <= t).mean() for t in tol]
            ls, mk = styles[m]
            ax.plot(tol, cov, ls, marker=mk, markevery=20, color="black" if m == "ours" else "0.45",
                    lw=2.5 if m == "ours" else 1.5, label=f"{m} ({NPARAM[m]} numbers)")
        ax.set_title(f"{name}: {len(rs)} windows of 2 s", fontsize=11)
        ax.set_xlabel("max error of the redrawn path (u)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("share of windows redrawn within the error")
    axs[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("How many numbers a 2-second primitive needs to redraw the recorded motion", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "coverage_2s.png", dpi=80)

    # figure 2: the sideways turn rate along each record, with contact shaded
    fig, axs = plt.subplots(len(RECORDS), 1, figsize=(17, 3.4 * len(RECORDS)), squeeze=False)
    axs = axs[:, 0]
    for ax, name in zip(axs, RECORDS):
        tr = traces[name]
        ax.plot(tr["T"], tr["turn_rate"], "-", color="black", lw=0.8, label="sideways turn rate (deg/s, 250 ms smoothed)")
        ax.fill_between(tr["T"], -400, 400, where=tr["C"], color="0.85", step="mid", label="touching a ramp or floor")
        ax.set_ylim(-400, 400)
        ax.set_ylabel("deg/s")
        ax.set_title(f"{name}: steering reverses {tr['reversals_per_s']:.2f} times per second", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    axs[-1].set_xlabel("time in the record (s)")
    fig.tight_layout()
    fig.savefig(out / "turn_rate_timeline.png", dpi=70)
    print((out / "coverage.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/research/primitives",
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.0,
         [a.split("=", 1) for a in sys.argv[3:]] or None)
