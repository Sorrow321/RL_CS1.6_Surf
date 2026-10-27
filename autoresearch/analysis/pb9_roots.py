"""Go-Explore roots from the policy's OWN greedy line (record_ckpt --dump-states), pbr9b2.

    python autoresearch/analysis/pb9_roots.py STATES.npz EP LO_PCT HI_PCT N OUT_ROOTS.npy [--line OUT_LINE.npy]

Takes episode EP of a --dump-states npz, finds the ticks whose running-best progress on the
grader's field (config.field) lies in [LO, HI] %, keeps the airborne ones (onground < 0), and
writes N of them, evenly spaced, as a STATE_DTYPE .npy for explore_phase1 --roots-spine.
--line also writes the whole episode as a .npy (the reference line for picks and stitching)."""
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
    ap.add_argument("states"); ap.add_argument("ep", type=int)
    ap.add_argument("lo", type=float); ap.add_argument("hi", type=float)
    ap.add_argument("n", type=int); ap.add_argument("out")
    ap.add_argument("--line", default=None)
    ap.add_argument("--all", action="store_true", help="keep ground ticks too")
    a = ap.parse_args()
    from surfgym.goalfield import GoalField
    z = np.load(REPO / CFG["field"], allow_pickle=False)
    f = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]),
                  float(z["reach_max"]))
    s = np.load(a.states)
    ep = s[f"ep_{a.ep:04d}"] if hasattr(s, "files") else s
    d = f.sample(ep["origin"].astype(np.float64)).astype(np.float64)
    d = np.where(d < f.reach_max - 16, d, 1e9)
    d0 = float(CFG["start_d"])
    prog = 100 * (d0 - np.minimum.accumulate(d)) / d0
    m = (prog >= a.lo) & (prog <= a.hi)
    if not a.all:
        m &= ep["onground"] < 0
    idx = np.flatnonzero(m)
    if not len(idx):
        sys.exit("no ticks in the window")
    pick = idx[np.unique(np.linspace(0, len(idx) - 1, min(a.n, len(idx))).round().astype(int))]
    np.save(a.out, ep[pick])
    print(f"{len(pick)} roots, ticks {pick[0]}..{pick[-1]} ({prog[pick[0]]:.2f}-{prog[pick[-1]]:.2f}%), "
          f"of {len(idx)} in the window -> {a.out}")
    if a.line:
        np.save(a.line, ep)
        print(f"line: {len(ep)} states -> {a.line}")


if __name__ == "__main__":
    main()
