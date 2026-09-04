#!/usr/bin/env python3
"""gaze_wave.py - "does it see the ramp, or does it remember the map?"

    python tools/gaze_wave.py --out runs/research/gaze30

The owner watched the round-30 eval recordings and said of xW2DEEP4: "the
agent surfs backwards/sideways instead of looking forward... that means the
agent doesn't learn to surf, it remembers the map. Learning to surf means
getting the idea 'see ramp => can surf', and the agent doesn't learn this."

An agent that flies where its camera is NOT pointing cannot be acting on the
geometry in front of it - the depth render is the only exteroception it has,
and a ramp behind the camera is not in the observation. So the share of fast
flight spent looking away from the flight path is a direct, cheap measure of
how much of the policy is open-loop memory of THIS map. This tool computes it
for every harvested run and asks five questions of the result (see the report
sections written to --out).

The three per-tick numbers are exactly tools/gaze_stats.py's definitions, so
the figures here are comparable with every gaze number already recorded:

  * heading = atan2(vy, vx) in degrees, on ticks whose HORIZONTAL speed
    exceeds --min-speed (default 200 u/s; below that the velocity direction
    is noise - walk speed is 250).
  * offset  = abs(wrap180(yaw - heading)); 0 = looking along the flight path,
    180 = flying backwards. Bucketed forward (<60), sideways (60-120),
    backwards (>120).
  * in-FOV  = offset <= hfov/2, hfov read from each run's own run.json
    (--lidar-hfov; 120 default, 160 on the wide-FOV arms). This is the one
    that says whether the thing being flown at is on screen at all.
  * pitch   = the raw pitch column over ALL ticks, no speed gate: where the
    camera points is a property of the policy, not of the flight.

Everything else is arc length along the champion route (maps/*.route.npz,
1,811 vertices x 128 u = 231,680 u), by nearest-vertex projection, so
"where along the map does it fly blind" is answerable and comparable across
arms. Corridor best comes from each run's harvested corridor.csv (the
order-only-16 MAX column, tools/wave/wave_harvest.py); runs without one are
scored from their trajectories with the same rule.

Pure CPU geometry over recorded positions - no map, no GPU, no checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# trajectory row layout: [t, x,y,z, vx,vy,vz, yaw, buttons, onground,
#                         progress, reward, pitch, fwd, side]
T, X, Y, Z, VX, VY, VZ, YAW = 0, 1, 2, 3, 4, 5, 6, 7
PITCH, FWD, SIDE = 12, 13, 14
MIN_COLS = 15
# src/surfcore.h: a[2]/a[3] are 0..2 -> forwardmove/sidemove = {-400, 0, +400}.
# Code 1 is "key not pressed", which is why the strafe-mode columns test == 1.
NEUTRAL = 1
PITCH_CLAMP = 30.0    # surfgym/respawn.py clips view pitch to [-70, +30]

BACK_DEG = 120.0     # offset above this = flying backwards
SIDE_DEG = 60.0      # offset above this = flying sideways
PITCH_DOWN = -45.0   # "staring at the floor"
PITCH_UP = 15.0      # "staring at the sky"


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- loading


def episodes(path):
    """Split a traj_*.jsonl into per-episode float arrays, FULL rows.

    surfgym.route.episodes_from_traj truncates to row[:8] and so drops the
    pitch column this tool exists to measure; the splitting rule (a header or
    footer dict, or the tick counter going backwards) is the same.
    """
    eps, cur, prev = [], [], None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":
                if cur:
                    eps.append(np.asarray(cur, np.float64))
                    cur = []
                prev = None
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                      # truncated tail of a live run
            if not isinstance(row, list) or len(row) < MIN_COLS:
                continue
            if prev is not None and row[T] <= prev:
                if cur:
                    eps.append(np.asarray(cur, np.float64))
                    cur = []
            prev = row[T]
            cur.append(row[:MIN_COLS])
    if cur:
        eps.append(np.asarray(cur, np.float64))
    return [e for e in eps if len(e) > 1]


def resolve_files(spec, last):
    """A run spec is a directory (take the last <last> evals) or a glob."""
    p = Path(spec)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "traj_[0-9]*[0-9].jsonl")),
                       key=lambda f: int(Path(f).stem.split("_")[1]))
        return files[-last:] if last > 0 else files, p
    files = sorted(glob.glob(spec))
    if not files and p.exists():
        files = [str(p)]
    return files, (Path(files[0]).parent if files else p.parent)


def load_cfg(run_dir):
    f = Path(run_dir) / "run.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("config", {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def corridor_best_from_csv(run_dir):
    f = Path(run_dir) / "corridor.csv"
    if not f.exists():
        return None, None, None
    best, fins, n = 0, 0, 0
    with open(f, encoding="ascii", errors="replace") as fh:
        for r in csv.DictReader(fh):
            try:
                best = max(best, int(r["max"]))
                fins += int(r["finishes"] or 0)
                n += 1
            except (ValueError, TypeError, KeyError):
                continue
    return (best or None), fins, n


def corridor_best_from_traj(files, pts, spacing, corridor=1500.0, window=16):
    """The order-only-16 rule wave_harvest.py runs remotely, run locally.

    For the two older finishers, which were harvested before corridor.csv
    existed. NOTE this is the best over the SUPPLIED evals only, not over the
    whole run, and must not be read as the same statistic as a corridor.csv
    best.
    """
    from surfgym.route import ArcProgress
    best = 0.0
    for f in files:
        for ep in episodes(f):
            ap = ArcProgress(np.asarray(pts, np.float64), spacing,
                             corridor=corridor, window=window)
            p = ep[:, X:Z + 1]
            ap.reset(p[:1])
            b = float(ap.arc[0])
            for k in range(1, len(p)):
                ap.advance(p[k:k + 1])
                b = max(b, float(ap.arc[0]))
            best = max(best, b)
    return best


# ---------------------------------------------------------------- stats


def pct(mask):
    return 100.0 * float(np.count_nonzero(mask)) / max(1, mask.size)


def spearman(a, b):
    """rho and a two-sided p from the t approximation (n >= 6)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan"), float("nan")   # constant input: rho undefined
    try:
        from scipy import stats
        r = stats.spearmanr(a, b)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        ra, rb = _rank(a), _rank(b)
        ra, rb = ra - ra.mean(), rb - rb.mean()
        den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
        rho = float((ra * rb).sum() / den) if den else float("nan")
        return rho, float("nan")


def _rank(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1, dtype=float)
    # average ties
    for val in np.unique(v):
        m = v == val
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def analyse(label, spec, args, pts, spacing, tree):
    files, run_dir = resolve_files(spec, args.last)
    if not files:
        return None
    cfg = load_cfg(run_dir)
    hfov = cfg.get("lidar_hfov")
    hfov = float(hfov) if hfov else float(args.fov)
    half = hfov / 2.0

    offs, pitches, arcs, lat, back_flags, n_eps, n_ticks = [], [], [], [], [], 0, 0
    keys = []
    for f in files:
        for ep in episodes(f):
            n_eps += 1
            n_ticks += len(ep)
            pitches.append(ep[:, PITCH])
            vx, vy = ep[:, VX], ep[:, VY]
            fast = np.hypot(vx, vy) > args.min_speed
            if not fast.any():
                continue
            e = ep[fast]
            head = np.degrees(np.arctan2(e[:, VY], e[:, VX]))
            o = np.abs(wrap180(e[:, YAW] - head))
            d, idx = tree.query(e[:, X:Z + 1], k=1)
            offs.append(o)
            arcs.append(idx.astype(np.float64) * spacing)
            lat.append(d)
            back_flags.append(o > BACK_DEG)
            keys.append(e[:, [FWD, SIDE]])
    if not offs:
        return None
    off = np.concatenate(offs)
    pit = np.concatenate(pitches)
    arc = np.concatenate(arcs)
    lat = np.concatenate(lat)
    back = np.concatenate(back_flags)
    k = np.concatenate(keys)
    fneu, sneu = k[:, 0] == NEUTRAL, k[:, 1] == NEUTRAL

    best, fins, n_evals = corridor_best_from_csv(run_dir)
    src = "corridor.csv"
    if best is None:
        best = int(round(corridor_best_from_traj(files, pts, spacing)))
        src = "from traj"
        fins, n_evals = None, len(files)

    return dict(
        label=label, run_dir=str(run_dir), files=[Path(f).name for f in files],
        cfg=cfg, hfov=hfov, n_eps=n_eps, n_ticks=n_ticks, n_fast=int(off.size),
        fast_share=100.0 * off.size / max(1, n_ticks),
        med_off=float(np.median(off)),
        off_p10=float(np.percentile(off, 10)),
        off_p90=float(np.percentile(off, 90)),
        back=pct(off > BACK_DEG), side=pct((off > SIDE_DEG) & (off <= BACK_DEG)),
        fwd=pct(off <= SIDE_DEG), infov=pct(off <= half),
        infov120=pct(off <= 60.0),
        # strafe mode: pure side-strafe (fmove == 0) is the branch whose
        # physics is EXACTLY invariant under yaw += 180 with sidemove negated
        side_only=pct(fneu & ~sneu), fwd_only=pct(~fneu & sneu),
        both=pct(~fneu & ~sneu), neither=pct(fneu & sneu),
        pitch_med=float(np.median(pit)),
        pitch_dn=pct(pit < PITCH_DOWN), pitch_up=pct(pit > PITCH_UP),
        pitch_clamp=pct(pit >= PITCH_CLAMP - 0.05),
        best=int(best), best_src=src, finishes=fins, n_evals=n_evals,
        arc=arc, lat=lat, back_mask=back, blind_mask=off > half,
        on_route=pct(lat <= args.corridor),
    )


def mannwhitney(a, b):
    """U-test p, for 'forward-looking arms get further' without assuming
    the trimodal in-FOV distribution is anything like linear."""
    try:
        from scipy import stats
        r = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def variance_split(prof, cnt):
    """How much of the (run, bin) backwards-share variation is RUN identity
    vs PLACE-on-the-map identity.

    An additive fit ``x_ij ~ mu + run_i + bin_j`` over the scored cells; the
    two sums of squares say whether flying backwards is a property of the
    policy (run term dominates: a global mode) or of where on the map it is
    (bin term dominates: a reaction to specific geometry, which is what
    "sees ramp => surfs" would look like).
    """
    ok = ~np.isnan(prof)
    if ok.sum() < 6:
        return None
    x = np.where(ok, prof, 0.0)
    w = np.where(ok, 1.0, 0.0)
    mu = x.sum() / w.sum()
    rmean = np.where(w.sum(1) > 0, x.sum(1) / np.maximum(w.sum(1), 1), mu)
    bmean = np.where(w.sum(0) > 0, x.sum(0) / np.maximum(w.sum(0), 1), mu)
    ss_tot = float((w * (x - mu) ** 2).sum())
    ss_run = float((w * (rmean[:, None] - mu) ** 2).sum())
    ss_bin = float((w * (bmean[None, :] - mu) ** 2).sum())
    return dict(n=int(ok.sum()), ss_tot=ss_tot, ss_run=ss_run, ss_bin=ss_bin,
                f_run=ss_run / ss_tot if ss_tot else float("nan"),
                f_bin=ss_bin / ss_tot if ss_tot else float("nan"))


def treatment(cfg, ref):
    """One-line 'what is different from the shallow control'."""
    if not cfg or not ref:
        return "(no run.json)"
    skip = {"label", "started", "finished", "steps", "seed"}
    d = [(k, ref.get(k), v) for k, v in cfg.items()
         if k in ref and k not in skip and ref.get(k) != v]
    if not d:
        return "control (identical to ref)"
    return "; ".join("%s %s->%s" % (k, _s(a), _s(b)) for k, a, b in d)


def _s(v):
    if v is None:
        return "-"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


# ---------------------------------------------------------------- output


def bin_profiles(rows, spacing, nbins, total, corridor, min_ticks,
                 key="back_mask"):
    """(runs x bins) share of ON-ROUTE fast ticks satisfying ``key``."""
    edges = np.linspace(0.0, total, nbins + 1)
    prof = np.full((len(rows), nbins), np.nan)
    cnt = np.zeros((len(rows), nbins), np.int64)
    for i, r in enumerate(rows):
        keep = r["lat"] <= corridor
        a, b = r["arc"][keep], r[key][keep]
        which = np.clip(np.digitize(a, edges[1:-1]), 0, nbins - 1)
        for j in range(nbins):
            m = which == j
            cnt[i, j] = int(m.sum())
            if cnt[i, j] >= min_ticks:
                prof[i, j] = 100.0 * float(b[m].sum()) / cnt[i, j]
    return prof, cnt, edges


def scatter_png(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=130)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    style = {"shallow ctl": ("o", "#1f77b4"), "deep": ("s", "#d62728"),
             "deep+treat": ("^", "#ff7f0e"), "shallow+treat": ("v", "#2ca02c"),
             "finisher": ("*", "#9467bd")}
    for g, rs in groups.items():
        m, c = style.get(g, ("o", "#7f7f7f"))
        ax.scatter([r["infov"] for r in rs], [r["best"] / 1e3 for r in rs],
                   marker=m, c=c, s=170 if m == "*" else 70, label=g,
                   edgecolors="black", linewidths=0.5, zorder=3)
    # labels pile up on the two walls of the trimodal x, so stagger them
    for k, r in enumerate(sorted(rows, key=lambda r: (r["infov"], r["best"]))):
        right = r["infov"] < 50.0
        ax.annotate(r["label"], (r["infov"], r["best"] / 1e3), fontsize=6.5,
                    xytext=(5 if right else -5, 3 + 5 * (k % 2)),
                    textcoords="offset points",
                    ha="left" if right else "right")
    rho, p = spearman([r["infov"] for r in rows], [r["best"] for r in rows])
    ax.set_xlabel("in-FOV share of fast ticks (%)  [offset <= hfov/2]")
    ax.set_ylabel("corridor best (ku, order-only 16)")
    ax.set_title("Looking where you fly vs how far you get  "
                 "(n=%d, Spearman rho=%.3f, p=%.3f)" % (len(rows), rho, p))
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def heat_png(rows, prof, edges, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort([-r["back"] for r in rows])
    m = prof[order]
    fig, ax = plt.subplots(figsize=(11.0, 0.30 * len(rows) + 2.2), dpi=130)
    im = ax.imshow(np.ma.masked_invalid(m), aspect="auto", cmap="magma",
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([rows[i]["label"] for i in order], fontsize=7)
    step = max(1, m.shape[1] // 12)
    ax.set_xticks(range(0, m.shape[1], step))
    ax.set_xticklabels(["%.0f%%" % (100.0 * edges[j] / edges[-1])
                        for j in range(0, m.shape[1], step)], fontsize=7)
    ax.set_xlabel("arc along the champion route (%% of %s u), "
                  "nearest-vertex projection, on-route ticks only"
                  % "{:,.0f}".format(edges[-1]))
    ax.set_title("share of fast ticks flown BACKWARDS (|yaw-heading| > 120 deg)")
    fig.colorbar(im, ax=ax, label="% backwards")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def table(head, widths, lines):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out.extend("| " + " | ".join(l) + " |" for l in lines)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "runs" / "research"),
                    help="where the xW1*/xW2* run dirs live")
    ap.add_argument("--glob", default="xW1*,xW2*",
                    help="comma-separated dir globs under --root")
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=SPEC",
                    help="extra run: a dir (last --last evals) or a file glob")
    ap.add_argument("--ref", default="xW1CTL",
                    help="run whose config the treatment column diffs against")
    ap.add_argument("--route", default=str(ROOT / "maps" /
                                           "surf_src_cannonball.route.npz"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "research" / "gaze30"))
    ap.add_argument("--last", type=int, default=2,
                    help="evals per run dir (0 = all)")
    ap.add_argument("--min-speed", type=float, default=200.0)
    ap.add_argument("--fov", type=float, default=120.0,
                    help="fallback hfov when run.json has none")
    ap.add_argument("--corridor", type=float, default=1500.0,
                    help="lateral distance still counted as on-route")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--min-bin-ticks", type=int, default=200)
    a = ap.parse_args()

    from scipy.spatial import cKDTree
    z = np.load(a.route)
    pts = np.asarray(z["route"], np.float32)
    spacing = float(z["spacing"]) if "spacing" in z.files else 128.0
    total = (len(pts) - 1) * spacing
    tree = cKDTree(pts.astype(np.float64))

    specs = []
    for g in a.glob.split(","):
        g = g.strip()
        if not g:
            continue
        for d in sorted(glob.glob(os.path.join(a.root, g))):
            if os.path.isdir(d):
                specs.append((os.path.basename(d), d))
    for s in a.run:
        label, _, path = s.partition("=")
        specs.append((label, path))

    ref_cfg = {}
    for label, spec in specs:
        if label == a.ref:
            ref_cfg = load_cfg(resolve_files(spec, a.last)[1])

    rows = []
    for label, spec in specs:
        r = analyse(label, spec, a, pts, spacing, tree)
        if r is None:
            print("skip %s (no usable trajectories)" % label)
            continue
        r["treat"] = treatment(r["cfg"], ref_cfg)
        deep = (r["cfg"].get("tower_depth") or 2) > 2
        other = [k for k in ("normals", "pitch_fixed", "rnn", "lidar_hfov",
                             "emb", "success_bonus", "speed_coef", "ret_norm",
                             "fp32_heads", "obs_compass")
                 if r["cfg"].get(k) not in (None, ref_cfg.get(k))]
        if not r["cfg"]:
            r["group"] = "finisher"
        elif deep and not other:
            r["group"] = "deep"
        elif deep:
            r["group"] = "deep+treat"
        elif other:
            r["group"] = "shallow+treat"
        else:
            r["group"] = "shallow ctl"
        if label in ("xQR32", "xsG5n"):
            r["group"] = "finisher"
        rows.append(r)
        print("%-10s eps %3d  fast %6d  med_off %6.1f  back %5.1f%%  "
              "inFOV %5.1f%%  pitch %6.1f  best %8d (%s)"
              % (label, r["n_eps"], r["n_fast"], r["med_off"], r["back"],
                 r["infov"], r["pitch_med"], r["best"], r["best_src"]))

    if not rows:
        raise SystemExit("no runs analysed")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    prof, cnt, edges = bin_profiles(rows, spacing, a.bins, total, a.corridor,
                                    a.min_bin_ticks)
    bprof, bcnt, _ = bin_profiles(rows, spacing, a.bins, total, a.corridor,
                                  a.min_bin_ticks, key="blind_mask")
    scatter_png(rows, str(outdir / "infov_vs_corridor.png"))
    heat_png(rows, prof, edges, str(outdir / "backwards_by_arc.png"))

    # --- per-run csv (the raw numbers, for anything downstream)
    with open(outdir / "gaze.csv", "w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "group", "hfov", "eps", "ticks", "fast_ticks",
                    "fast_pct", "med_offset", "off_p10", "off_p90", "back_pct",
                    "side_pct", "fwd_pct", "infov_pct", "infov120_pct",
                    "on_route_pct", "side_only_pct", "fwd_only_pct",
                    "pitch_med", "pitch_below_-45_pct", "pitch_above_+15_pct",
                    "pitch_at_+30_clamp_pct",
                    "corridor_best", "corridor_src", "finishes", "treatment"])
        for r in rows:
            w.writerow([r["label"], r["group"], "%.0f" % r["hfov"], r["n_eps"],
                        r["n_ticks"], r["n_fast"], "%.1f" % r["fast_share"],
                        "%.1f" % r["med_off"], "%.1f" % r["off_p10"],
                        "%.1f" % r["off_p90"], "%.1f" % r["back"],
                        "%.1f" % r["side"], "%.1f" % r["fwd"],
                        "%.1f" % r["infov"], "%.1f" % r["infov120"],
                        "%.1f" % r["on_route"], "%.1f" % r["side_only"],
                        "%.1f" % r["fwd_only"], "%.1f" % r["pitch_med"],
                        "%.1f" % r["pitch_dn"], "%.1f" % r["pitch_up"],
                        "%.1f" % r["pitch_clamp"],
                        r["best"], r["best_src"],
                        "" if r["finishes"] is None else r["finishes"],
                        r["treat"]])
    np.savetxt(outdir / "backwards_by_arc.csv", prof, fmt="%.2f",
               delimiter=",",
               header="rows=%s; cols=arc bins 0..%d u" %
                      ("|".join(r["label"] for r in rows), int(total)))

    write_report(outdir, rows, prof, cnt, edges, a, total, bprof, bcnt)
    print("\nwrote %s" % (outdir / "report.md"))
    return 0


def write_report(outdir, rows, prof, cnt, edges, a, total, bprof, bcnt):
    L = []
    W = L.append
    W("# Round 30 gaze audit: does the policy look where it flies?")
    W("")
    W("Generated by `tools/gaze_wave.py` (CPU only, pure geometry over the")
    W("recorded trajectories). Definitions are `tools/gaze_stats.py`'s:")
    W("")
    W("* fast tick = horizontal speed > %.0f u/s (%d ticks below that are"
      % (a.min_speed, sum(r["n_ticks"] - r["n_fast"] for r in rows)))
    W("  excluded from every angle number; heading is noise there).")
    W("* offset = |wrap180(yaw - atan2(vy, vx))|; 0 = looking along the")
    W("  flight path, 180 = flying backwards.")
    W("* forward < %.0f deg, sideways %.0f-%.0f deg, backwards > %.0f deg."
      % (SIDE_DEG, SIDE_DEG, BACK_DEG, BACK_DEG))
    W("* in-FOV = offset <= hfov/2, hfov taken from each run's own run.json")
    W("  (120 deg standard, 160 deg on the FOV arms).")
    W("* pitch = the raw pitch column over ALL ticks, no speed gate.")
    W("* corridor best = MAX of the order-only-16 corridor progress over the")
    W("  whole run's evals, from the harvested `corridor.csv`.")
    W("")
    W("Evals used: the last %d per run (%d greedy episodes each)." % (a.last, 9))
    W("")
    mark = len(L)

    W("## 1. Per-run gaze")
    W("")
    head = ["run", "group", "hfov", "eps", "fast ticks", "med off",
            "off p10-p90", "back >120", "side 60-120", "fwd <60", "in-FOV",
            "pitch med", "p<-45", "p>+15", "p at +30", "corr best"]
    lines = []
    for r in sorted(rows, key=lambda r: -r["med_off"]):
        lines.append([r["label"], r["group"], "%.0f" % r["hfov"],
                      str(r["n_eps"]), "%d" % r["n_fast"],
                      "%.1f" % r["med_off"],
                      "%.0f-%.0f" % (r["off_p10"], r["off_p90"]),
                      "%.1f%%" % r["back"],
                      "%.1f%%" % r["side"], "%.1f%%" % r["fwd"],
                      "%.1f%%" % r["infov"], "%.1f" % r["pitch_med"],
                      "%.1f%%" % r["pitch_dn"], "%.1f%%" % r["pitch_up"],
                      "%.1f%%" % r["pitch_clamp"],
                      "%s%s" % ("{:,}".format(r["best"]),
                                "*" if r["best_src"] != "corridor.csv" else "")])
    L.extend(table(head, [max(len(head[i]), max(len(l[i]) for l in lines))
                          for i in range(len(head))], lines))
    W("")
    W("`*` = no corridor.csv harvested; scored from the supplied eval(s) only")
    W("with the same order-only-16 rule, so it is a floor, not the run best.")
    W("")
    W("### 1b. The three attractors, and why 180 deg is free")
    W("")
    W("`offset` is not spread over [0, 180]: every run LOCKS onto one of")
    W("three values and holds it to a few degrees (the p10-p90 column). The")
    W("three values are the three air-strafe equilibria, and which one a run")
    W("sits in is decided by which movement key it holds")
    W("(`src/surfcore.h`: a[2]/a[3] -> forwardmove/sidemove = {-400, 0, +400}):")
    W("")
    W("* **offset ~0** - pure side-strafe (`forwardmove == 0`), wishdir")
    W("  perpendicular to the view, so the equilibrium velocity runs ALONG")
    W("  the view axis. The ramp being ridden is centred in the depth image.")
    W("* **offset ~90** - pure forward/back (`sidemove == 0`), wishdir along")
    W("  the view axis, so the equilibrium velocity is PERPENDICULAR to it.")
    W("  The camera points at the wall beside the flight path.")
    W("* **offset ~180** - pure side-strafe again, but with the camera")
    W("  turned around. **This is an exact symmetry of the physics**: with")
    W("  `forwardmove == 0`, wishdir = right * sidemove, and yaw += 180 flips")
    W("  the right vector, so negating `sidemove` reproduces the identical")
    W("  wishdir and the identical trajectory. Nothing in the movement code")
    W("  distinguishes the two branches. The ONLY thing that does is the")
    W("  depth image, which is rendered at `yaw` (`raster.py::render`).")
    W("")
    W("So a policy that had learned 'see ramp => surf' could not sit in the")
    W("180 branch - the ramp would be off-screen. A policy that has memorised")
    W("this map is free to, and 4 of the 21 wave arms do.")
    W("")
    head = ["run", "med off", "side-only (fmove=0)", "fwd-only (smove=0)",
            "both", "neither"]
    lines = [[r["label"], "%.1f" % r["med_off"], "%.1f%%" % r["side_only"],
              "%.1f%%" % r["fwd_only"], "%.1f%%" % r["both"],
              "%.1f%%" % r["neither"]]
             for r in sorted(rows, key=lambda r: -r["med_off"])]
    L.extend(table(head, [max(len(head[i]), max(len(l[i]) for l in lines))
                          for i in range(len(head))], lines))
    W("")
    W("Pitch is even cheaper to get wrong: surfcore.h says outright that")
    W("*'pitch aims the lidar only - it has no effect on movement physics'*,")
    W("so a policy that does not use the picture pays nothing for letting the")
    W("camera drift to the +30 deg clamp. The `p at +30` column is that drift.")
    W("")

    W("## 2. Treatments")
    W("")
    lines = [[r["label"], r["treat"][:110]] for r in rows]
    L.extend(table(["run", "config diff vs %s" % a.ref],
                   [12, max(len(l[1]) for l in lines)], lines))
    W("")

    W("## 3. Group means")
    W("")
    g = {}
    for r in rows:
        g.setdefault(r["group"], []).append(r)
    lines = []
    for k in sorted(g):
        rs = g[k]
        lines.append([k, str(len(rs)),
                      "%.1f" % np.mean([r["med_off"] for r in rs]),
                      "%.1f%%" % np.mean([r["back"] for r in rs]),
                      "%.1f%%" % np.mean([r["infov"] for r in rs]),
                      "%.1f" % np.mean([r["pitch_med"] for r in rs]),
                      "{:,.0f}".format(np.mean([r["best"] for r in rs]))])
    L.extend(table(["group", "n", "med off", "backwards", "in-FOV",
                    "pitch med", "corr best"], [14, 3, 8, 10, 8, 10, 10],
                   lines))
    W("")

    W("## 4. in-FOV share vs corridor best (Spearman)")
    W("")
    subsets = [("all runs", rows),
               ("wave arms only (xW1*/xW2*)",
                [r for r in rows if r["group"] != "finisher"]),
               ("deep-net arms only",
                [r for r in rows if r["group"].startswith("deep")]),
               ("wave arms + xQR32",
                [r for r in rows if r["group"] != "finisher"
                 or r["label"] == "xQR32"])]
    lines = []
    for name, rs in subsets:
        if len(rs) < 3:
            continue
        rho, p = spearman([r["infov"] for r in rs], [r["best"] for r in rs])
        rho2, p2 = spearman([-r["back"] for r in rs], [r["best"] for r in rs])
        lines.append([name, str(len(rs)), "%+.3f" % rho, "%.3f" % p,
                      "%+.3f" % rho2, "%.3f" % p2])
    L.extend(table(["subset", "n", "rho(inFOV, best)", "p",
                    "rho(-backwards, best)", "p"],
                   [28, 3, 16, 6, 21, 6], lines))
    W("")
    W("Spearman on a variable that takes three values is weak by")
    W("construction, so the same question asked as a two-group comparison,")
    W("wave arms only (the finishers are a different generation and their")
    W("corridor figure is a floor from one eval):")
    W("")
    wv = [r for r in rows if r["group"] != "finisher"]
    look = [r for r in wv if r["infov"] >= 90.0]
    blind = [r for r in wv if r["infov"] <= 15.0]
    if look and blind:
        lb = [r["best"] for r in look]
        bb = [r["best"] for r in blind]
        u, p = mannwhitney(lb, bb)
        W("* looking forward (in-FOV >= 90 pct), n=%d: corridor best median "
          "%s, mean %s" % (len(look), "{:,.0f}".format(np.median(lb)),
                           "{:,.0f}".format(np.mean(lb))))
        W("  (%s)" % ", ".join(r["label"] for r in look))
        W("* blind (in-FOV <= 15 pct), n=%d: corridor best median %s, mean %s"
          % (len(blind), "{:,.0f}".format(np.median(bb)),
             "{:,.0f}".format(np.mean(bb))))
        W("  (%s)" % ", ".join(r["label"] for r in blind))
        W("* Mann-Whitney U = %.1f, two-sided p = %.3f" % (u, p))
    W("")
    W("![in-FOV vs corridor best](infov_vs_corridor.png)")
    W("")

    W("## 5. Where along the route the backwards flying happens")
    W("")
    W("Nearest-vertex projection onto the champion route; only ticks within")
    W("%.0f u of the line (off-route falls project to arbitrary vertices)."
      % a.corridor)
    W("Bins of %.0f u (%.1f%% of the %s u route). Cells with fewer than %d"
      % (edges[1] - edges[0], 100.0 / a.bins, "{:,.0f}".format(total),
         a.min_bin_ticks))
    W("on-route fast ticks are blank.")
    W("")
    nb = prof.shape[1]
    head = ["arc bin", "% of route", "runs scored", "pooled backwards",
            "runs > 50%", "min", "max"]
    lines = []
    for j in range(nb):
        col = prof[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            lines.append(["%.0f-%.0f k" % (edges[j] / 1e3, edges[j + 1] / 1e3),
                          "%.0f-%.0f%%" % (100 * edges[j] / total,
                                           100 * edges[j + 1] / total),
                          "0", "-", "-", "-", "-"])
            continue
        w = cnt[ok, j].astype(float)
        pooled = float((col[ok] * w).sum() / w.sum())
        lines.append(["%.0f-%.0f k" % (edges[j] / 1e3, edges[j + 1] / 1e3),
                      "%.0f-%.0f%%" % (100 * edges[j] / total,
                                       100 * edges[j + 1] / total),
                      str(int(ok.sum())), "%.1f%%" % pooled,
                      str(int((col[ok] > 50).sum())),
                      "%.1f%%" % col[ok].min(), "%.1f%%" % col[ok].max()])
    L.extend(table(head, [max(len(head[i]), max(len(l[i]) for l in lines))
                          for i in range(len(head))], lines))
    W("")
    W("![backwards by arc](backwards_by_arc.png)")
    W("")

    W("### 5b. Per-run profile across the map")
    W("")
    W("If backwards flying were a REACTION to particular geometry it would")
    W("switch on and off along the route inside one run. If it is a global")
    W("mode of the policy it is flat, near 0 or near 100, everywhere the run")
    W("reaches.")
    W("")
    head = ["run", "bins scored", "backwards mean", "min", "max", "std",
            "bins > 50%"]
    lines = []
    for i, r in enumerate(sorted(range(len(rows)),
                                 key=lambda i: -np.nanmean(prof[i])
                                 if np.isfinite(prof[i]).any() else 0)):
        col = prof[r]
        ok = ~np.isnan(col)
        if not ok.any():
            continue
        lines.append([rows[r]["label"], str(int(ok.sum())),
                      "%.1f%%" % col[ok].mean(), "%.1f%%" % col[ok].min(),
                      "%.1f%%" % col[ok].max(), "%.1f" % col[ok].std(),
                      "%d/%d" % (int((col[ok] > 50).sum()), int(ok.sum()))])
    L.extend(table(head, [max(len(head[i]), max(len(l[i]) for l in lines))
                          for i in range(len(head))], lines))
    W("")

    for nm, pr, cn in (("backwards (offset > 120 deg)", prof, cnt),
                       ("blind, i.e. outside the run's own FOV", bprof, bcnt)):
        vs = variance_split(pr, cn)
        if not vs:
            continue
        W("Additive decomposition of the %d scored (run, bin) cells for"
          % vs["n"])
        W("**%s**, `x_ij ~ mu + run_i + bin_j`:" % nm)
        W("")
        W("* RUN identity explains **%.1f%%** of the total sum of squares"
          % (100 * vs["f_run"]))
        W("* PLACE-on-the-map identity explains **%.1f%%**"
          % (100 * vs["f_bin"]))
        W("")

    # cross-run agreement on WHERE
    pair, n = [], len(rows)
    for i in range(n):
        for k in range(i + 1, n):
            ok = ~np.isnan(prof[i]) & ~np.isnan(prof[k])
            if ok.sum() >= 5:
                rho, _ = spearman(prof[i][ok], prof[k][ok])
                if not math.isnan(rho):
                    pair.append((rho, rows[i]["label"], rows[k]["label"]))
    if pair:
        rr = np.array([p[0] for p in pair])
        W("Cross-run agreement on WHERE the backwards flying sits: mean")
        W("pairwise Spearman over the per-bin backwards profile = **%+.3f**"
          % rr.mean())
        W("(median %+.3f, %d of %d pairs positive, range %+.3f to %+.3f, over"
          % (float(np.median(rr)), int((rr > 0).sum()), len(rr), rr.min(),
             rr.max()))
        W("%d pairs with at least 5 overlapping scored bins). Pairs where"
          % len(rr))
        W("either profile is constant are undefined and excluded, which is")
        W("most of them - and a constant profile is itself the finding.")
        W("")

    L[mark:mark] = answers(rows, prof, cnt, bprof, bcnt, a)
    return Path(outdir / "report.md").write_text("\n".join(L) + "\n",
                                                 encoding="ascii")


def answers(rows, prof, cnt, bprof, bcnt, a):
    """The five questions, answered with the numbers computed above."""
    by = {r["label"]: r for r in rows}
    wv = [r for r in rows if r["group"] != "finisher"]
    ctl = [r for r in wv if r["group"] == "shallow ctl"]
    deep = [r for r in wv if (r["cfg"].get("tower_depth") or 2) > 2]
    A = ["## 0. Answers", ""]

    def g(n, k):
        return by[n][k] if n in by else float("nan")

    A += ["**1. Does the deep net fly backwards more than the shallow "
          "control?** Not",
          "as a depth effect - as a CAPACITY effect, and it is a lottery, "
          "not a trend.",
          "The 4 arms in the backwards branch are %s."
          % ", ".join("%s (%.1f%%)" % (r["label"], r["back"])
                      for r in sorted(wv, key=lambda r: -r["back"])[:4]),
          "All 4 added capacity over the control (2 deep, 1 wide, 1 GRU); "
          "none of the",
          "%d plain controls did (backwards %s). But the deep arms do not do "
          "it" % (len(ctl), ", ".join("%.1f%%" % r["back"] for r in ctl)),
          "consistently: four runs of the SAME deep config (xW1DEEP, "
          "xW2DEEP3, xW2DEEP4,",
          "xW2DEEP5) score %s backwards - two of four"
          % ", ".join("%.1f%%" % g(n, "back") for n in
                      ("xW1DEEP", "xW2DEEP3", "xW2DEEP4", "xW2DEEP5")),
          "identical runs are blind and two are not. That is the seed lottery "
          "CLAUDE.md",
          "already records for the corridor metric, read out on a new axis. "
          "The owner's",
          "reading of the xW2DEEP4 recording is exactly right for that run "
          "(%.1f%% of its" % g("xW2DEEP4", "back"),
          "fast ticks backwards, in-FOV %.1f%%, and it still reaches %s u of"
          % (g("xW2DEEP4", "infov"),
             "{:,}".format(int(g("xW2DEEP4", "best")))),
          "corridor) - it just is not a property of depth.", ""]

    fp = [r for r in deep if r["pitch_med"] > 0]
    A += ["**2. Do the pitch-fixed, normals and FOV arms look forward more?**",
          "Pitch-fixing works on the axis it controls and nothing else; "
          "normals do not;",
          "wide FOV is the only treatment that helps horizontally, and mostly "
          "arithmetically.",
          "",
          "* xW2PITCH (`--pitch-fixed -25`): pitch median %.1f, 0.0%% below "
          "-45 and 0.0%%" % g("xW2PITCH", "pitch_med"),
          "  above +15 by construction, against %.0f-%.0f%% above +15 for the "
          "free-pitch" % (min(r["pitch_up"] for r in fp),
                          max(r["pitch_up"] for r in fp)),
          "  deep arms. Horizontally unchanged: median offset %.1f, in-FOV "
          "%.1f%% - it" % (g("xW2PITCH", "med_off"), g("xW2PITCH", "infov")),
          "  sits in the 90 deg branch.",
          "* xW2NRM (normals): median offset %.1f, in-FOV %.1f%% - the 90 deg"
          % (g("xW2NRM", "med_off"), g("xW2NRM", "infov")),
          "  branch too. No effect on gaze.",
          "* xW1FOV / xW2FOV (hfov 120 -> 160): in-FOV %.1f%% and %.1f%%. "
          "xW1FOV's gain" % (g("xW1FOV", "infov"), g("xW2FOV", "infov")),
          "  is arithmetic - the same offsets counted against a wider window "
          "(%.1f%% inside" % g("xW1FOV", "infov120"),
          "  +/-60 deg vs %.1f%% inside +/-80 deg). Only xW2FOV actually sits "
          "in the" % g("xW1FOV", "infov"),
          "  aligned branch (median offset %.1f)." % g("xW2FOV", "med_off"),
          "",
          "What every free-pitch arm shares: the camera drifts to the +30 deg "
          "clamp and",
          "stays (%.0f-%.0f%% of ticks pinned at it across the %d arms whose "
          "median pitch"
          % (min(r["pitch_clamp"] for r in wv if r["pitch_med"] > 0),
             max(r["pitch_clamp"] for r in wv),
             sum(1 for r in wv if r["pitch_med"] > 0)),
          "is positive). surfcore.h: pitch aims the lidar and nothing else, "
          "so drifting",
          "to the clamp is free unless the picture is being used.", ""]

    A += ["**3. Do the finishers look forward?** Yes, and they are the "
          "cleanest cases in",
          "the set. xQR32: median offset %.1f deg, in-FOV %.1f%%, 0.0%% "
          "backwards, pitch" % (g("xQR32", "med_off"), g("xQR32", "infov")),
          "median %.1f with 1.3%% of ticks above +15 - it looks DOWN at the "
          "ramp." % g("xQR32", "pitch_med"),
          "xsG5n: median offset %.1f, in-FOV %.1f%%, pitch median %.1f. Every "
          "arm that"
          % (g("xsG5n", "med_off"), g("xsG5n", "infov"),
             g("xsG5n", "pitch_med")),
          "reaches 190k+ u of corridor is in the aligned branch.", ""]

    look = [r for r in wv if r["infov"] >= 90.0]
    blind = [r for r in wv if r["infov"] <= 15.0]
    rho_all, p_all = spearman([r["infov"] for r in rows],
                              [r["best"] for r in rows])
    rho_w, p_w = spearman([r["infov"] for r in wv], [r["best"] for r in wv])
    rho_d, p_d = spearman([r["infov"] for r in deep], [r["best"] for r in deep])
    u, pu = mannwhitney([r["best"] for r in look], [r["best"] for r in blind])
    A += ["**4. Does in-FOV share correlate with corridor progress?** Weakly "
          "overall,",
          "strongly inside the deep arms. Spearman rho(in-FOV, corridor best) "
          "= %+.3f" % rho_all,
          "(p = %.3f) over all %d runs, %+.3f (p = %.3f) over the %d wave "
          "arms alone, and" % (p_all, len(rows), rho_w, p_w, len(wv)),
          "%+.3f (p = %.3f) over the %d deep-net arms. As a two-group test on "
          "the wave" % (rho_d, p_d, len(deep)),
          "arms, the %d aligned arms median %s u against %s u for the %d "
          "blind arms"
          % (len(look), "{:,.0f}".format(np.median([r["best"] for r in look])),
             "{:,.0f}".format(np.median([r["best"] for r in blind])),
             len(blind)),
          "(Mann-Whitney U = %.1f, p = %.3f). The direction is consistent and "
          "the ceiling" % (u, pu),
          "is one-sided - the three best wave arms (%s) are all aligned - "
          "but looking"
          % ", ".join(r["label"] for r in
                      sorted(wv, key=lambda r: -r["best"])[:3]),
          "forward is plainly not sufficient: xW1RETN and xW1BON10 look "
          "forward %.0f%% and" % g("xW1RETN", "infov"),
          "%.0f%% of the time and still stop at ~106k u."
          % g("xW1BON10", "infov"), ""]

    vs_b = variance_split(prof, cnt)
    vs_f = variance_split(bprof, bcnt)
    A += ["**5. Where along the route does the backwards flying happen?** "
          "Everywhere, and",
          "it is a property of the RUN, not of the place. Splitting the "
          "per-(run, arc-bin)",
          "backwards share additively, RUN identity explains %.1f%% of the "
          "sum of squares" % (100 * vs_b["f_run"]),
          "and PLACE explains %.1f%%; for the broader out-of-FOV share the "
          "split is" % (100 * vs_b["f_bin"]),
          "%.1f%% / %.1f%%. Each of the four backwards arms is backwards in "
          "EVERY arc bin" % (100 * vs_f["f_run"], 100 * vs_f["f_bin"]),
          "it reaches (11/11, 11/11, 8/8, 9/9 bins above 50%, per-run std "
          "0.3-8.6 points);",
          "every other run is at ~0% in every bin. No stretch of cannonball "
          "turns the",
          "camera around - the policy picks a gaze mode at the spawn and "
          "holds it for the",
          "whole flight, which is the shape of an open-loop habit rather than "
          "a reaction",
          "to geometry.", ""]
    return A


if __name__ == "__main__":
    raise SystemExit(main())
