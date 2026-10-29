"""ge_start_report.py - what a Go-Explore archive rooted at the true start found (pbfr1b4, 2026-10-29).

    python autoresearch/analysis/ge_start_report.py runs/<ge>/archive.npz [--top 12]

ANALYSIS ONLY: reads an explore_phase1 archive (reward-free uniform-action search from the map
spawn) and reports, on the GRADER's field (config.field): the deepest cell and its chain, the
progress histogram of the archive, and the energy height h = z + |v|^2/1600 of the cells BEHIND
the start (d above the spawn's). Nothing here names a place on the map; progress is the field's.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score  # noqa: E402


def chain_to(parent, i):
    out = []
    while i >= 0:
        out.append(int(i))
        i = int(parent[i])
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    z = np.load(a.archive, allow_pickle=False)
    st = z["state"]
    parent = z["parent"] if "parent" in z.files else None
    F, D0 = score.field(), score.START_D
    o = st["origin"].astype(np.float64)
    v = st["velocity"].astype(np.float64)
    d = F.sample(o).astype(np.float64)
    bad = d >= F.reach_max - 0.5 * F.cell
    prog = np.where(bad, -np.inf, 100.0 * (D0 - d) / D0)
    h = o[:, 2] + (v ** 2).sum(1) / 1600.0
    print(f"{len(st)} cells; off-field {int(bad.sum())}")
    for lo in (-3, -1, 0, 0.3, 0.45, 0.6, 1.0, 1.4, 2.0, 3.0):
        print(f"  progress >= {lo:5.2f}%: {int((prog >= lo).sum())}")
    behind = (~bad) & (d > D0)
    if behind.any():
        print(f"behind the start: {int(behind.sum())} cells, h median {np.median(h[behind]):.0f} "
              f"p90 {np.percentile(h[behind], 90):.0f} max {h[behind].max():.0f} "
              f"(spawn h {o[0, 2] + (v[0] ** 2).sum() / 1600:.0f})")
    order = np.argsort(-prog)
    print("deepest cells:")
    for i in order[:a.top]:
        sp = np.hypot(v[i, 0], v[i, 1])
        line = (f"  cell {i:7d} prog {prog[i]:6.2f}% at ({o[i, 0]:6.0f},{o[i, 1]:7.0f},{o[i, 2]:6.0f}) "
                f"vh {sp:5.0f} vz {v[i, 2]:5.0f} h {h[i]:5.0f}")
        if parent is not None:
            ch = chain_to(parent, int(i))
            line += f" chain {len(ch)}"
        print(line)
    if parent is not None:
        i = int(order[0])
        ch = chain_to(parent, i)
        print(f"chain of the deepest cell ({len(ch)} cells): prog / h / origin")
        for c in ch:
            print(f"  {c:7d} {prog[c]:6.2f}% h {h[c]:5.0f} ({o[c, 0]:6.0f},{o[c, 1]:7.0f},{o[c, 2]:6.0f}) "
                  f"v ({v[c, 0]:5.0f},{v[c, 1]:5.0f},{v[c, 2]:5.0f})")
    meta = Path(a.archive).with_suffix(".meta.json")
    if meta.exists():
        m = json.loads(meta.read_text())
        print("meta:", {k: m[k] for k in list(m)[:30] if not isinstance(m[k], (list, dict))})


if __name__ == "__main__":
    main()
