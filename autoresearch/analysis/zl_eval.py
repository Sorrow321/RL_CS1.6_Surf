#!/usr/bin/env python3
"""zl_eval.py - read a greedy eval (traj_<step>.jsonl) on surf_za_loop, per episode.

Prints, per episode: its length (s, on the header's tick), how it ended, best progress on the
grader's field (config.field, d0 = the episode's first row), the time of that best, energy height
E = z + |v|^2 / 1600 at the start / its max / at the end, the highest z, the end position, and E
every 10 s. Reporting only: nothing here feeds a reward, a spawn rule or a schedule.

    python autoresearch/analysis/zl_eval.py runs/zlr1b3/<run>/traj_<step>.jsonl [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text())


def field():
    from surfgym.goalfield import GoalField
    z = np.load(REPO / CFG["field"], allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def episodes(path):
    eps, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append((hdr, np.asarray(cur, np.float64), d))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def main():
    F = field()
    for path in sys.argv[1:]:
        eps = episodes(path)
        print(f"== {path}: {len(eps)} episodes")
        for hdr, r, tr in eps:
            if not len(r):
                continue
            pat = hdr.get("tick_pattern_ms")
            dt = (float(np.mean(pat)) if pat else float(hdr.get("tick_ms", 10.0))) / 1000.0
            p, v = r[:, 1:4], r[:, 4:7]
            d = F.sample(p).astype(np.float64)
            ok = d < F.reach_max - 0.5 * F.cell
            d0 = float(d[0])
            prog = np.where(ok, 100.0 * (d0 - d) / d0, -np.inf)
            k = int(np.argmax(prog))
            E = p[:, 2] + (v ** 2).sum(1) / 1600.0
            t = np.arange(len(r)) * dt
            marks = " ".join(f"{E[int(s / dt)]:.0f}" for s in range(10, int(t[-1]) + 1, 10)
                             if int(s / dt) < len(E))
            print(f"ep{hdr.get('episode', '?')}: {t[-1]:6.1f}s {tr.get('end'):5s} best "
                  f"{prog[k]:5.1f}% @{t[k]:5.1f}s  E0 {E[0]:6.0f} Emax {E.max():6.0f} "
                  f"Eend {E[-1]:6.0f} zmax {p[:, 2].max():6.0f}  end ({p[-1, 0]:.0f}, "
                  f"{p[-1, 1]:.0f}, {p[-1, 2]:.0f}) |v| {np.linalg.norm(v[-1]):.0f}  E/10s: {marks}")


if __name__ == "__main__":
    main()
