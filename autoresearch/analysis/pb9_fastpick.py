"""Earliest arrival in a Go-Explore exact-chains archive rooted on the policy's own line (pbr9b2).

    python autoresearch/analysis/pb9_fastpick.py ARCHIVE.npz LINE.npy PCT [PCT ...] [--line-ms 7.6667 --ge-ms 8]

For each progress level PCT (% of config.start_d on config.field): the line's own first arrival
(row x --line-ms) against every archive cell at or past PCT, arrival = its root's row on the line x
--line-ms + its chain depth x --ge-ms. Prints the fastest cells per level with their energy height
h = z + |v|^2/1600 and the gain in seconds; the cell id goes to tools/ge_replay.py --cell.
Round 8 picked GE 5 / GE 6 this way (runs/pbr8b1_look/fast_pick.py, never committed)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
CFG = json.loads((REPO / "autoresearch" / "config.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive"); ap.add_argument("line"); ap.add_argument("pct", type=float, nargs="+")
    ap.add_argument("--line-ms", type=float, default=23.0 / 3.0)
    ap.add_argument("--ge-ms", type=float, default=8.0)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--near", type=float, default=0.0, help="only cells within this many u of the "
                    "line's own states (0 = any)")
    a = ap.parse_args()
    from scipy.spatial import cKDTree
    from surfgym.goalfield import GoalField
    z = np.load(REPO / CFG["field"], allow_pickle=False)
    f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]),
                  float(z["reach_max"]))
    d0 = float(CFG["start_d"])

    def prog(o):
        d = f.sample(np.asarray(o, np.float64)).astype(np.float64)
        d = np.where(d < f.reach_max - 16, d, np.inf)
        return 100 * (d0 - d) / d0

    A = np.load(a.archive, allow_pickle=False)
    st, par, dep = A["state"], np.asarray(A["parent"], np.int64), np.asarray(A["depth"], np.int64)
    L = np.load(a.line)
    Lo = np.asarray(L["origin"], np.float64)
    Lp = np.maximum.accumulate(np.where(np.isfinite(prog(Lo)), prog(Lo), -1e9))
    root = np.empty(len(par), np.int64)
    for i in range(len(par)):
        root[i] = i if par[i] < 0 else root[par[i]]
    o = np.asarray(st["origin"], np.float64)
    v = np.asarray(st["velocity"], np.float64)
    h = o[:, 2] + (v * v).sum(1) / 1600
    p = prog(o)
    tree = cKDTree(Lo)
    rd, rj = tree.query(o[root])
    t_arr = rj * a.line_ms / 1000 + dep * a.ge_ms / 1000
    near_ok = np.ones(len(o), bool)
    if a.near > 0:
        dd, _ = tree.query(o)
        near_ok = dd <= a.near
    print(f"archive {len(o):,} cells, {len(np.unique(root))} roots (root rows {rj[par < 0].min()}-"
          f"{rj[par < 0].max()}, max root offset {rd[par < 0].max():.1f} u)")
    for P in a.pct:
        k = np.flatnonzero(Lp >= P)
        tl = k[0] * a.line_ms / 1000 if len(k) else np.nan
        hl = (Lo[k[0], 2] + float((np.asarray(L["velocity"][k[0]], np.float64) ** 2).sum()) / 1600
              if len(k) else np.nan)
        m = np.flatnonzero((p >= P) & np.isfinite(p) & near_ok)
        print(f"-- {P:.2f}%: line arrives {tl:.2f} s (h {hl:.0f}); {len(m):,} cells at or past it")
        if not len(m):
            continue
        order = m[np.argsort(t_arr[m])][:a.top]
        for i in order:
            print(f"   cell {i}: {t_arr[i]:.2f} s ({t_arr[i] - tl:+.2f}) at {p[i]:.2f}% h {h[i]:.0f} "
                  f"|v| {np.linalg.norm(v[i]):.0f} pos {o[i].round(0)} root row {rj[i]} depth {dep[i]}")


if __name__ == "__main__":
    main()
