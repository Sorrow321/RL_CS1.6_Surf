"""mcts_line_report.py - progress and checkpoint curtains of a search line (beam_frontier.npz),
graded with score.py's own field and curtains, plus the search's summary.json.

    python autoresearch/analysis/mcts_line_report.py <out_dir> [<out_dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "autoresearch"))

import score  # noqa: E402


def report(d: Path) -> dict:
    out = {"dir": str(d)}
    s = d / "summary.json"
    if s.exists():
        sm = json.loads(s.read_text())
        for k in ("mode", "waves", "nodes", "cells", "best_pct", "greedy_pct", "best_depth",
                  "max_depth", "finishes", "search_wall_s", "env_steps_per_s", "greedy_end",
                  "frontier_s"):
            if k in sm:
                out[k] = sm[k]
    f = d / "beam_frontier.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        st = z["states"]
        p = np.asarray(st["origin"], np.float64)
        fld = score.field()
        dd = fld.sample(p).astype(np.float64)
        ok = dd < fld.reach_max - 0.5 * fld.cell
        d0 = float(score.START_D)
        out["line_pct"] = round(100.0 * (d0 - float(dd[ok].min())) / d0, 2) if ok.any() else None
        cp = 0
        for n, lo, hi in score.curtains():
            if np.any(np.all((p >= lo) & (p <= hi), axis=1)):
                cp = max(cp, n)
        out["line_cp"] = cp
        out["line_s"] = round(len(st) / 100.0, 2)
        out["replay_ok"] = int(z["replay_ok"])
        out["finished"] = int(z["finished"]) if "finished" in z.files else 0
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(json.dumps(report(Path(a))))
