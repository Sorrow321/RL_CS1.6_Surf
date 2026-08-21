"""compare_runs.py - line up several training runs at matched steps.

Every arm we run is n=1 against enormous seed variance (measured: two
champion-recipe seeds sat at 9.6k and 27k track at the same 500M steps),
so a single pair of curves means little without the historical band next
to it. This prints arms side by side at fixed step marks and, when the
reference runs are present, the min-max band they spanned.

  python tools/compare_runs.py sYAW sCTL sSTR2
  python tools/compare_runs.py --dir runs --marks 250e6,500e6,1e9 A B

Remote arms: pull their progress.csv first, e.g.
  scp -P <port> root@<host>:/root/RL_Surf/runs/<run>/progress.csv \
      runs/<run>/progress.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

# champion-mechanism scratch seeds, for the band
REFERENCE = ("sIS_long", "sIS_b")
COLS = {
    "track": "race/eval_progress",
    "rew": "rollout/ep_rew_mean",
    "len": "rollout/ep_len_mean",
    "spd": "eval/speed_max",
    "fin": "race/eval_finish_s",
}


def load(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((int(float(r["time/total_timesteps"])), r))
                except (ValueError, KeyError, TypeError):
                    pass
    except FileNotFoundError:
        return None
    return rows or None


def at(rows, mark, tol):
    """Latest row at or before `mark`, if within tol."""
    ok = [r for r in rows if r[0] <= mark]
    if not ok:
        return None
    step, row = ok[-1]
    return row if mark - step <= tol else None


def val(row, key):
    if row is None:
        return None
    raw = row.get(COLS[key], "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default="runs")
    ap.add_argument("--marks", default="250e6,500e6,1e9,2e9,4e9")
    ap.add_argument("--metric", default="track", choices=list(COLS))
    args = ap.parse_args()

    marks = [float(m) for m in args.marks.split(",")]
    base = Path(args.dir)
    data = {r: load(base / r / "progress.csv") for r in args.runs}
    refs = {r: load(base / r / "progress.csv") for r in REFERENCE}
    refs = {k: v for k, v in refs.items() if v}

    missing = [r for r, v in data.items() if v is None]
    for r in missing:
        print(f"  (no progress.csv for {r})")
    data = {k: v for k, v in data.items() if v}
    if not data:
        raise SystemExit("nothing to compare")

    print(f"\nmetric: {COLS[args.metric]}   (tolerance: half a mark gap)")
    hdr = "  " + "run".ljust(14) + "".join(f"{m/1e9:>10.2f}B" for m in marks)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, rows in data.items():
        cells = []
        for i, m in enumerate(marks):
            tol = m / 2 if i == 0 else (m - marks[i - 1])
            v = val(at(rows, m, tol), args.metric)
            cells.append("       ---" if v is None else f"{v:>10,.0f}")
        last = rows[-1][0]
        print("  " + name.ljust(14) + "".join(cells) + f"   (to {last/1e9:.2f}B)")

    if refs:
        print()
        for i, m in enumerate(marks):
            tol = m / 2 if i == 0 else (m - marks[i - 1])
            vs = [val(at(r, m, tol), args.metric) for r in refs.values()]
            vs = [v for v in vs if v is not None]
            if not vs:
                continue
            print(f"  reference seed band @{m/1e9:.2f}B: "
                  f"{min(vs):,.0f} .. {max(vs):,.0f}  "
                  f"({', '.join(refs)})")


if __name__ == "__main__":
    main()
