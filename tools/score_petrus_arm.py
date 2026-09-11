#!/usr/bin/env python3
"""score_petrus_arm.py - one arm, both rulers, plus the metrics CLAUDE.md
insists are read TOGETHER.

CLAUDE.md: `race/eval_progress` has been blind (round 18) and
anti-correlated (round 18 xARC); `race/win_rate` is deceptive on its own
and must be reported next to reservoir min-depth (round 19 xPSSR, on this
very map). Round 37 adds `dip/*`. So every petrus arm is scored the same
way here and nowhere else:

  * corridor MAX and finishes, --order-only 16, on BOTH
    surf_petrus_lite.fieldroute.npz and .wrroute.npz;
  * per-eval, so a TIME-TO-EVENT (the step at which a threshold is first
    cleared) can be reported instead of an end-of-run mean, which the
    retraction section says is a coin flip;
  * `race/win_rate` and reservoir min-depth (`mind`, % of d0 REMAINING) on
    the same rows;
  * the `dip/*` columns;
  * the greedy END POSITIONS and their spread.

    python tools/score_petrus_arm.py --run runs/prCTL
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.route import ArcProgress                            # noqa: E402


sys.path.insert(0, str(ROOT / "tools" / "demo"))
from render_channels import load_episodes                         # noqa: E402


def episodes(path):
    """Every episode of a traj_*.jsonl as (N, 15) float rows (the recorder's
    own split rule, shared with tools/demo/render_channels.py)."""
    eps, _hdrs = load_episodes(Path(path))
    return eps


def corridor(route_path, eps, window=16, corridor_u=1500.0):
    r = ArcProgress.load(str(route_path), corridor=corridor_u, window=window)
    out = []
    for e in eps:
        p = e[:, 1:4]
        r.reset(p[:1])
        best = float(r.arc[0])
        for k in range(1, len(p)):
            r.advance(p[k:k + 1])
            best = max(best, float(r.arc[0]))
        out.append(best)
    return np.asarray(out), float(r.length)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--maps-dir", default="C:/RL_Surf/maps")
    ap.add_argument("--map", default="surf_petrus_lite")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    md = Path(a.maps_dir)
    zones = json.loads((md / f"{a.map}.zones.json").read_text())
    end = zones["end"]
    box = (np.asarray(end["mins"] if "mins" in end else end[0], float),
           np.asarray(end["maxs"] if "maxs" in end else end[1], float))

    rows = list(csv.DictReader(open(run / "progress.csv", encoding="utf-8")))
    # the reservoir depth is NOT a csv column - it lives only in the
    # trainer's own step line ("res 3,783 mind 99.552%"), UTF-16 on Windows.
    logp = run.parent / f"{run.name}_launch.txt"
    logsteps, logres, logmind, logwin = [], [], [], []
    if logp.exists():
        import re
        txt = logp.read_bytes().decode("utf-16", errors="replace")
        for ln in txt.splitlines():
            m = re.match(r"^step\s+([\d,]+).*?res\s+([\d,]+)\s+mind\s+"
                         r"([\d.]+)%", ln)
            if m:
                logsteps.append(int(m.group(1).replace(",", "")))
                logres.append(int(m.group(2).replace(",", "")))
                logmind.append(float(m.group(3)))
            mw = re.search(r"win\s+([\d.]+)%", ln)
            if m and mw:
                logwin.append(float(mw.group(1)))
    logsteps = np.asarray(logsteps)
    trajs = sorted(run.glob("traj_*.jsonl"))
    rep = {"run": str(run), "evals": []}
    print(f"== {run.name}: {len(trajs)} evals, {len(rows)} progress rows")
    hdr = ("  step        field MAX   %    wr MAX     %   fin  eps  "
           "eval_prog   win%%   mind%%   dip max/fail  endz spread")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tp in trajs:
        step = int(tp.stem.split("_")[1])
        eps = episodes(tp)
        if not eps:
            continue
        fa, flen = corridor(md / f"{a.map}.fieldroute.npz", eps)
        wa, wlen = corridor(md / f"{a.map}.wrroute.npz", eps)
        endp = np.stack([e[-1, 1:4] for e in eps])
        fin = int(sum(bool(np.all(p >= box[0]) and np.all(p <= box[1]))
                      for p in endp))
        # the progress row nearest this step
        r = min(rows, key=lambda q: abs(int(float(q["time/total_timesteps"]))
                                        - step)) if rows else {}
        j = int(np.argmin(np.abs(logsteps - step))) if len(logsteps) else None

        def g(k, d=float("nan")):
            try:
                return float(r.get(k, d))
            except (TypeError, ValueError):
                return d
        e = dict(step=step, n_eps=len(eps),
                 field_max=float(fa.max()), field_pct=100 * fa.max() / flen,
                 field_mean=float(fa.mean()),
                 wr_max=float(wa.max()), wr_pct=100 * wa.max() / wlen,
                 wr_mean=float(wa.mean()), finishes=fin,
                 eval_progress=g("race/eval_progress"),
                 win_rate=g("race/success_rate"),
                 res_n=(logres[j] if j is not None else float("nan")),
                 res_mind=(logmind[j] if j is not None else float("nan")),
                 log_win=(logwin[j] if j is not None and j < len(logwin)
                          else float("nan")),
                 dip_max=g("dip/max_survived_depth"),
                 dip_fail=g("dip/fail_depth"),
                 end_z_mean=float(endp[:, 2].mean()),
                 end_spread_u=float(np.linalg.norm(
                     endp - endp.mean(0), axis=1).max()),
                 end_pos=endp.tolist())
        rep["evals"].append(e)
        print("  %-11d %8.0f %5.1f %8.0f %5.1f %4d %4d %10.0f %6.2f %7.3f "
              "%5.2f/%-5.2f %7.0f %7.0f" % (
                  step, e["field_max"], e["field_pct"], e["wr_max"],
                  e["wr_pct"], fin, len(eps), e["eval_progress"],
                  e["win_rate"] * (100 if e["win_rate"] <= 1 else 1),
                  e["res_mind"], e["dip_max"], e["dip_fail"],
                  e["end_z_mean"], e["end_spread_u"]))
    if rows:
        last = rows[-1]
        rep["last_row"] = {k: last.get(k) for k in
                           ("step", "race/win_rate", "dip/max_survived_depth",
                            "dip/fail_frac", "train/ep_len_mean") if k in last}
        print("  progress.csv columns available: "
              + ", ".join(k for k in rows[0] if k.startswith(("race/", "dip/"))))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print("  wrote", a.json)


if __name__ == "__main__":
    main()
