"""wave_table.py - matched-step tables from harvested corridor.csv files.

    python wave_table.py <runs_root> <groups.json> [--steps 1.0,1.5,2.0,2.5,3.0]

groups.json: {"3090": ["xW1CTL", "xW1RETN", ...], "4090": [...], ...}
For each group prints, per run, the order-only corridor MAX and mean at the
eval nearest each matched step (within 0.12B), the best MAX so far, the
number of evals, finishes, and the last eval's step.
"""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="ascii", errors="replace")):
        try:
            rows.append((int(r["step"]), int(r["mean"]), int(r["max"]), int(r["finishes"] or 0)))
        except ValueError:
            continue
    return rows


def at(rows, step, tol=0.12e9):
    best = min(rows, key=lambda x: abs(x[0] - step), default=None)
    return best if best and abs(best[0] - step) <= tol else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("groups")
    ap.add_argument("--steps", default="1.0,1.5,2.0,2.5,3.0")
    a = ap.parse_args()
    steps = [float(s) * 1e9 for s in a.steps.split(",")]
    groups = json.loads(Path(a.groups).read_text(encoding="utf-8"))
    for card, runs in groups.items():
        print(f"\n=== {card}   corridor MAX (mean) in ku at matched steps; best MAX; evals; finishes; last step")
        print(f"{'run':12s} " + " ".join(f"{s/1e9:>11.1f}B" for s in steps) + f" {'best':>8s} {'n':>4s} {'fin':>4s} {'last':>7s}")
        for run in runs:
            p = Path(a.root) / run / "corridor.csv"
            if not p.exists():
                print(f"{run:12s} (no corridor.csv)")
                continue
            rows = load(p)
            cells = []
            for s in steps:
                r = at(rows, s)
                cells.append(f"{r[2]/1e3:5.1f}({r[1]/1e3:4.1f})" if r else f"{'-':>11s}")
            best = max((r[2] for r in rows), default=0)
            fin = sum(r[3] for r in rows)
            print(f"{run:12s} " + " ".join(f"{c:>12s}" for c in cells) + f" {best/1e3:8.1f} {len(rows):4d} {fin:4d} {rows[-1][0]/1e9 if rows else 0:6.2f}B")


if __name__ == "__main__":
    main()
