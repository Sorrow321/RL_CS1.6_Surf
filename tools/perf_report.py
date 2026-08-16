#!/usr/bin/env python3
"""perf_report.py — read train_fast.py --timing logs, report per-phase medians.

The benchmark protocol (docs/perf-implementation-plan.md 0.2) is: run 40
iterations from the frozen checkpoint and take the MEDIAN of the last 30
TIMING lines — the first few carry warm-up, the optional recording, and the
cudnn.benchmark autotune, and a median ignores the periodic-checkpoint
iterations instead of letting them skew a mean.

    python3 tools/perf_report.py runs/pb_baseline.log
    python3 tools/perf_report.py runs/pb_s2.log --vs runs/pb_baseline.log

With --vs, phases are shown side by side with the delta; the headline is the
end-to-end speedup on `total`. Also prints the spread (p10..p90) of `total`
so a claimed win can be checked against the run's own noise.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# order matters only for display; unknown fields are appended
ORDER = ("pool", "rollout_fwd", "sync_copy", "env", "reward_py", "boot",
         "book", "respawn", "vis_cpu", "lidar", "rollout_wall",
         "gae", "gae_gpu", "update", "update_gpu", "mb_gpu",
         "ckpt", "record", "misc", "total")
# phases whose times overlap another phase rather than adding to the wall
NESTED = {"rollout_fwd", "sync_copy", "env", "reward_py", "boot", "book",
          "respawn", "vis_cpu", "lidar", "gae_gpu", "update_gpu", "mb_gpu"}


def parse(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("TIMING "):
            continue
        row = {}
        for tok in line.split()[1:]:
            k, _, v = tok.partition("=")
            if k == "iter":                  # the line's own index, not a phase
                continue
            try:
                row[k] = float(v)
            except ValueError:
                continue
        if "total" in row:
            rows.append(row)
    return rows


def medians(rows: list[dict], last: int) -> dict[str, float]:
    tail = rows[-last:] if last and len(rows) > last else rows
    keys = {k for r in tail for k in r}
    return {k: statistics.median([r[k] for r in tail if k in r]) for k in keys}


def fields(*mds: dict) -> list[str]:
    seen = [k for k in ORDER if any(k in m for m in mds)]
    seen += sorted({k for m in mds for k in m} - set(ORDER))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--vs", type=Path, default=None, help="baseline log")
    ap.add_argument("--last", type=int, default=30,
                    help="median over the last N iterations (default 30)")
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        raise SystemExit(f"no TIMING lines in {args.log} — was --timing passed?")
    md = medians(rows, args.last)
    tail = rows[-args.last:] if len(rows) > args.last else rows
    totals = sorted(r["total"] for r in tail)
    p10 = totals[max(0, int(0.1 * len(totals)) - 1)]
    p90 = totals[min(len(totals) - 1, int(0.9 * len(totals)))]

    base = medians(parse(args.vs), args.last) if args.vs else None
    if args.vs and not base:
        raise SystemExit(f"no TIMING lines in {args.vs}")

    print(f"{args.log}  ({len(rows)} iters, median of last {min(args.last, len(rows))})")
    if base:
        print(f"vs {args.vs}")
    print()
    hdr = f"{'phase':<13}{'ms':>10}"
    if base:
        hdr += f"{'base ms':>10}{'delta':>10}{'x':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k in fields(md, base or {}):
        v = md.get(k, float("nan"))
        mark = " " if k not in NESTED else "."     # '.' = nested/overlapping
        line = f"{mark}{k:<12}{v:>10.1f}"
        if base:
            b = base.get(k, float("nan"))
            d = v - b
            x = (b / v) if v else float("nan")
            line += f"{b:>10.1f}{d:>+10.1f}{x:>8.3f}"
        print(line)
    print()
    print(f"total spread p10..p90: {p10:.0f}..{p90:.0f} ms "
          f"({100.0 * (p90 - p10) / md['total']:.1f}% of median)")
    ticks = 786432.0
    print(f"throughput: {ticks / (md['total'] / 1e3):,.0f} ticks/s")
    if base:
        print(f"END-TO-END SPEEDUP: {base['total'] / md['total']:.3f}x  "
              f"({base['total']:.0f} -> {md['total']:.0f} ms/iter)")
    print("('.' phases are nested inside another phase / GPU-side and do not "
          "add to the wall)")


if __name__ == "__main__":
    sys.exit(main())
