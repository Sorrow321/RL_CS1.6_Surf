#!/usr/bin/env python3
"""plot_m3_recheck.py - the Round 37 M3 reconciliation.

Recomputes the petrus L-bend shortfall THREE independent ways and draws
them on one axis, so the sign of "worst cumulative shortfall of the
correct branch against the naive branch" is settled by arithmetic rather
than by which script ran:

  1. tools/branch_table.py's own output (runs/.../m2_petrus.json), read back;
  2. the same rule recomputed from scratch over
     runs/.../dips/curves.json with a segment-projected arc alignment;
  3. an ALIGNMENT-FREE statistic - geodesic d banked per unit travelled -
     which no choice of t = 0 can move.

    python tools/plot_m3_recheck.py --curves <dips/curves.json> \\
        --m2 <m2_petrus.json> --route C:/RL_Surf/maps/surf_petrus_lite.wrroute.npz \\
        --out docs/img/bend_derivative_recheck.png --json <out.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

WINDOW = 36          # decisions: the naive branch's whole remaining life


def arc_seg(pts, xyz):
    a, b = pts[:-1], pts[1:]
    ab = b - a
    L2 = (ab * ab).sum(1)
    seglen = np.sqrt(L2)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    out = np.empty(len(xyz))
    for k, p in enumerate(xyz):
        t = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
        q = a + ab * t[:, None]
        dd = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(dd))
        out[k] = cum[j] + t[j] * seglen[j]
    return 100.0 * out / cum[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--m2", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--branch-pct", type=float, default=17.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    pts = np.load(a.route)["route"]
    C = {e["name"]: e for e in json.load(open(a.curves, encoding="utf-8"))}
    rep = {}

    # ---- 1. branch_table.py's own numbers, read back -------------------
    B = json.load(open(a.m2, encoding="utf-8"))
    nb, cb = B["naive"], B["correct"]
    m1 = min(nb["n"], cb["n"])
    g1 = np.asarray(nb["cum"][:m1]) - np.asarray(cb["cum"][:m1])
    t1 = np.asarray(nb["t"][:m1])
    rep["branch_table"] = dict(
        max_naive_lead=float(g1.max()), at_t=float(t1[int(np.argmax(g1))]),
        min_gap=float(g1.min()), min_at_t=float(t1[int(np.argmin(g1))]),
        secs_naive_ahead=float((g1 > 0).sum() * nb["dt"]),
        window_s=float(t1[-1]), n=int(m1))

    # ---- 2. recomputed from curves.json --------------------------------
    R = {}
    for nm in ("pt_prRATCH_DIES", "pt_WRdemo_FINISH"):
        e = C[nm]
        xyz = np.asarray(e["xyz"])
        pct = arc_seg(pts, xyz)
        i0 = int(np.argmax(pct >= a.branch_pct))
        d = np.asarray(e["d"])
        t = np.asarray(e["t"])
        sc = 100.0 / e["d0"]
        sl = slice(i0, i0 + WINDOW)
        R[nm] = dict(t=t[sl] - t[i0],
                     cum=(d[i0] - d[sl]) * sc - 0.5 * (t[sl] - t[i0]),
                     d=d[sl], xyz=xyz[sl], i0=i0, pct0=float(pct[i0]),
                     t0=float(t[i0]))
    n, w = R["pt_prRATCH_DIES"], R["pt_WRdemo_FINISH"]
    g2 = n["cum"] - w["cum"]
    r2 = np.diff(n["cum"]) / np.diff(n["t"])
    r2w = np.diff(w["cum"]) / np.diff(w["t"])
    rep["curves_recompute"] = dict(
        naive_t0=n["t0"], naive_arc0=n["pct0"],
        wr_t0=w["t0"], wr_arc0=w["pct0"],
        max_naive_lead=float(g2.max()),
        at_t=float(n["t"][int(np.argmax(g2))]),
        min_gap=float(g2.min()), min_at_t=float(n["t"][int(np.argmin(g2))]),
        n_samples_naive_ahead=int((g2 > 0).sum()), n=int(len(g2)),
        peak_rate_naive=float(r2.max()),
        peak_rate_naive_t=float(n["t"][int(np.argmax(r2)) + 1]),
        peak_rate_wr=float(r2w.max()),
        peak_rate_wr_t=float(w["t"][int(np.argmax(r2w)) + 1]))

    # ---- 2b. the SAME recompute under branch_table's own arc rule -------
    # tools/branch_table.py aligns on surfgym.route.ArcProgress (order-only,
    # corridor 1,500 u, window 16), not on a raw segment projection. Doing
    # it that way from curves.json reproduces the ledger's number exactly,
    # which is what makes this a reconciliation rather than a third opinion.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from surfgym.route import ArcProgress                        # noqa: E402
    _tot = float(np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))[-1])

    def _order_only(xyz):
        r = ArcProgress.load(a.route, corridor=1500.0, window=16)
        r.reset(xyz[:1])
        best = float(r.arc[0])
        o = [best]
        for k in range(1, len(xyz)):
            r.advance(xyz[k:k + 1])
            best = max(best, float(r.arc[0]))
            o.append(best)
        return 100.0 * np.asarray(o) / _tot

    OO = {}
    for nm in ("pt_prRATCH_DIES", "pt_WRdemo_FINISH"):
        e = C[nm]
        xyz = np.asarray(e["xyz"])
        pc = _order_only(xyz)
        i0 = int(np.argmax(pc >= a.branch_pct))
        d = np.asarray(e["d"])
        t = np.asarray(e["t"])
        sl = slice(i0, i0 + WINDOW)
        OO[nm] = dict(t=t[sl] - t[i0],
                      cum=(d[i0] - d[sl]) * (100.0 / e["d0"])
                      - 0.5 * (t[sl] - t[i0]),
                      t0=float(t[i0]), pct0=float(pc[i0]))
    g2b = OO["pt_prRATCH_DIES"]["cum"] - OO["pt_WRdemo_FINISH"]["cum"]
    tb = OO["pt_prRATCH_DIES"]["t"]
    rep["order_only_recompute"] = dict(
        naive_t0=OO["pt_prRATCH_DIES"]["t0"],
        naive_arc0=OO["pt_prRATCH_DIES"]["pct0"],
        wr_t0=OO["pt_WRdemo_FINISH"]["t0"],
        wr_arc0=OO["pt_WRdemo_FINISH"]["pct0"],
        max_naive_lead=float(g2b.max()),
        at_t=float(tb[int(np.argmax(g2b))]),
        min_gap=float(g2b.min()), min_at_t=float(tb[int(np.argmin(g2b))]))

    # ---- 3. alignment-free: d banked per unit travelled ----------------
    def per_unit(e):
        dist = np.linalg.norm(np.diff(e["xyz"], axis=0), axis=1).sum()
        return float((e["d"][0] - e["d"][-1]) / dist), float(dist)
    pn, dn = per_unit(n)
    pw, dw = per_unit(w)
    rep["alignment_free"] = dict(naive_d_per_u=pn, naive_travelled=dn,
                                 wr_d_per_u=pw, wr_travelled=dw,
                                 ratio=pn / pw)

    # ---- 4. alignment sweep: can any offset flip the sign? -------------
    sweep = []
    for k in range(-15, 16):
        e = C["pt_WRdemo_FINISH"]
        d = np.asarray(e["d"])
        t = np.asarray(e["t"])
        j = w["i0"] + k
        cw = (d[j] - d[j:j + WINDOW]) * (100.0 / e["d0"]) \
            - 0.5 * (t[j:j + WINDOW] - t[j])
        g = n["cum"][:len(cw)] - cw
        sweep.append(dict(shift=k, max_naive_lead=float(g.max()),
                          min_gap=float(g.min())))
    rep["wr_alignment_sweep"] = sweep
    ok = [s["shift"] for s in sweep if s["max_naive_lead"] > 0.05]
    rep["sweep_verdict"] = (
        "the naive branch leads by more than 0.05 reward units at every WR "
        "alignment shift in [%+d, %+d] decisions; only a shift of %+d or "
        "more (the WR read %.2f s LATER than its own arc-%.1f%% crossing) "
        "erases it" % (min(ok), max(ok), max(ok) + 1,
                       (max(ok) + 1) * 0.04, a.branch_pct))

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")

    fig, ax = plt.subplots(3, 1, figsize=(8.6, 8.6))
    ax[0].plot(n["t"], n["cum"], "-o", ms=3, color="#c62828",
               label="NAIVE / prRATCH (dies at 8.28 s)")
    ax[0].plot(w["t"], w["cum"], "-o", ms=3, color="#2e7d32",
               label="CORRECT / human WR (finishes)")
    ax[0].set_ylabel("cumulative reward from the branch\n"
                     "(shaping at 100/d0, minus 0.5/s)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Round 37 M3, recomputed: the DYING branch leads in "
                    "reward for %d of %d decisions" %
                    ((g2 > 0).sum(), len(g2)), fontsize=10)
    ax[1].plot(n["t"], g2, "-o", ms=3, color="#4527a0")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].annotate("max naive lead %+.3f at t = %.2f s" %
                   (g2.max(), n["t"][int(np.argmax(g2))]),
                   (n["t"][int(np.argmax(g2))], g2.max()),
                   textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax[1].set_ylabel("naive minus correct\n(reward units)")
    ax[2].plot(n["t"][1:], r2, "-", color="#c62828", label="naive")
    ax[2].plot(w["t"][1:], r2w, "-", color="#2e7d32", label="correct (WR)")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("instantaneous rate\n(reward units / s)")
    ax[2].set_xlabel("seconds after each line's own arc-%.1f%% crossing"
                     % a.branch_pct)
    ax[2].legend(fontsize=8)
    ax[2].set_title("the WR's PEAK rate is higher (%.2f vs %.2f) but arrives "
                    "LATER (t = %.2f vs %.2f s) - which is why it is behind "
                    "on the cumulative" %
                    (r2w.max(), r2.max(), w["t"][int(np.argmax(r2w)) + 1],
                     n["t"][int(np.argmax(r2)) + 1]), fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)
    print(json.dumps(rep, indent=1))
    print("wrote", a.out, "and", a.json)


if __name__ == "__main__":
    main()
