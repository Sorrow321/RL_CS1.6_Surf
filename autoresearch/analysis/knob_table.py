#!/usr/bin/env python3
"""knob_table.py - one row per knob search of a run (unr1b4, 2026-11-08): when it ran, the line
it rooted on, the knob's range over the search, each mode's new cells per 1k decisions, the
deepest cell and the modes of the chain that reached it, and what the trainer did with it.

    python autoresearch/analysis/knob_table.py RUN_LAUNCH_LOG AUTO_DIR [--d0 99675]

RUN_LAUNCH_LOG = runs/<run>_launch.txt (the trainer's log: search starts and outcomes);
AUTO_DIR = runs/<run>/auto (k01/, k02/, ... each with knob/knob_summary.json, knob.log).
Analysis only: it reads logs, it decides nothing."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def lam_range(knob_log: Path):
    lams = []
    if knob_log.exists():
        for ln in knob_log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"lambda ([0-9.]+)", ln)
            if m and ln.startswith("knob-GE ") and "s  cells" in ln:
                lams.append(float(m.group(1)))
    return (min(lams), max(lams)) if lams else (None, None)


def outcomes(launch_log: Path) -> dict:
    out = {}
    if not launch_log.exists():
        return out
    for ln in launch_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"\[\s*([\d,]+)\] auto-spine search (\d+): (NEW SPINE|NO new spine|STALE|"
                      r"spine kept|running beside)", ln)
        if not m:
            continue
        k, what = int(m.group(2)), m.group(3)
        step = int(m.group(1).replace(",", ""))
        rec = out.setdefault(k, {})
        if what == "running beside":
            rec["start"] = step
        else:
            rec["end"] = step
            rec["outcome"] = {"NEW SPINE": "adopted", "NO new spine": "none", "STALE": "stale",
                              "spine kept": "not deeper"}[what]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("launch_log")
    ap.add_argument("auto_dir")
    ap.add_argument("--d0", type=float, default=99675.0)
    a = ap.parse_args(argv)
    oc = outcomes(Path(a.launch_log))
    d0 = float(a.d0)
    print("| search | start step | line best | deepest cell | knob range | cells/1k dec "
          "policy / random | chain (P policy, R random) | outcome |")
    print("|---|---|---|---|---|---|---|---|")
    for kd in sorted(Path(a.auto_dir).glob("k[0-9][0-9]")):
        k = int(kd.name[1:])
        sp = kd / "knob" / "knob_summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        lo, hi = lam_range(kd / "knob.log")
        cb, db = s.get("cells_by_mode") or [0, 0], s.get("decisions_by_mode") or [1, 1]
        yp = 1000.0 * cb[0] / max(db[0], 1)
        yr = 1000.0 * cb[1] / max(db[1], 1)
        lb = s.get("line_best_d")
        bd = s.get("best_d")
        chain = "".join("P" if m == 0 else "R" for m in (s.get("chain_modes") or []))
        o = oc.get(k, {})
        print(f"| {k} | {o.get('start', 0) / 1e6:,.0f}M | "
              f"{100 * (d0 - lb) / d0 if lb else float('nan'):.1f}% | "
              f"{100 * (d0 - bd) / d0 if bd else float('nan'):.1f}% | "
              f"{lo if lo is not None else '-'}-{hi if hi is not None else '-'} | "
              f"{yp:.2f} / {yr:.2f} | {chain or '-'} | {o.get('outcome', '?')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
