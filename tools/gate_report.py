#!/usr/bin/env python3
"""gate_report.py - one table per run: did the agent even TRY the dip?

Reads runs/<run>/progress.csv and prints, at a few points along the run,
the start-line eval (`race/eval_progress`, `race/map_pct`) beside the gate
columns: pit/ramp-box visits per iteration, the share of episodes that gave
back more than 1,000 u of potential (`gate/dip_frac`), the p90 accepted
rise (`gate/rise_p90`) and the top speed (`gate/vmax_mean`), plus the
unstuck temperature. The record's start on unitfarmer2 is a +2,840 u rise
at 1,800 u/s, so an arm that "tries the ramps below" shows dip_frac > 0 and
vmax climbing before eval_progress moves.

    python tools/gate_report.py uf2RATCH uf2TP uf2SURF uf2PIT [--rows 6]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def col(rows, key):
    for k in rows[-1]:
        if k == key or k.startswith(key + "."):
            return k
    return None


def f(v, nd=0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{x:,.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rows", type=int, default=6, help="points along the run to print")
    a = ap.parse_args()
    for run in a.runs:
        p = ROOT / "runs" / run / "progress.csv"
        if not p.exists():
            print(f"== {run}: no progress.csv")
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows:
            print(f"== {run}: empty")
            continue
        keys = {n: col(rows, n) for n in ("time/total_timesteps", "race/eval_progress", "race/map_pct",
                                          "race/eval_finish_s", "gate/hit_frac", "gate/dip_frac",
                                          "gate/rise_p90", "gate/vmax_mean", "unstuck/T", "rollout/ep_len_mean",
                                          "rollout/ep_rew_mean")}
        boxes = [k for k in rows[-1] if k.startswith("gate/") and k.endswith("_eps") or
                 (k.startswith("gate/") and "_eps." in k)]
        ev = [r for r in rows if r.get(keys["race/eval_progress"] or "", "")]
        best_ev = max((float(r[keys["race/eval_progress"]]) for r in ev), default=float("nan"))
        fin = [float(r[keys["race/eval_finish_s"]]) for r in rows if keys["race/eval_finish_s"] and r.get(keys["race/eval_finish_s"])]
        dips = [float(r[keys["gate/dip_frac"]]) for r in rows if keys["gate/dip_frac"] and r.get(keys["gate/dip_frac"])]
        vis = [int(float(r[b])) for b in boxes for r in rows if r.get(b)]
        print(f"== {run}: {len(rows)} iterations, {int(float(rows[-1]['time/total_timesteps'])):,} steps | "
              f"best eval_progress {best_ev:,.0f} u | evals with a finish {len(fin)}"
              + (f" (best {min(fin):.2f} s)" if fin else "")
              + f" | box visits total {sum(vis)} | max dip_frac {max(dips) if dips else '-'}")
        hdr = ["steps", "eval_prog", "map_pct", "hit%", "dip%>1k", "rise_p90", "vmax", "T", "ep_len", "ep_rew"] + [b.split("/")[-1] for b in boxes]
        print("   " + " | ".join(hdr))
        n = len(rows)
        picks = sorted(set([0] + [int(i * (n - 1) / max(a.rows - 1, 1)) for i in range(a.rows)] + [n - 1]))
        for i in picks:
            r = rows[i]
            vals = [f(r.get("time/total_timesteps")), f(r.get(keys["race/eval_progress"] or "")),
                    f(r.get(keys["race/map_pct"] or ""), 1),
                    f(float(r.get(keys["gate/hit_frac"] or "") or 0) * 100, 1) if r.get(keys["gate/hit_frac"] or "") else "-",
                    f(float(r.get(keys["gate/dip_frac"] or "") or 0) * 100, 1) if r.get(keys["gate/dip_frac"] or "") else "-",
                    f(r.get(keys["gate/rise_p90"] or "")), f(r.get(keys["gate/vmax_mean"] or "")),
                    f(r.get(keys["unstuck/T"] or ""), 2), f(r.get(keys["rollout/ep_len_mean"] or "")),
                    f(r.get(keys["rollout/ep_rew_mean"] or ""), 1)] + [f(r.get(b)) for b in boxes]
            print("   " + " | ".join(vals))


if __name__ == "__main__":
    main()
