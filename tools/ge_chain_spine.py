#!/usr/bin/env python3
"""ge_chain_spine.py - a curriculum spine from a Go-Explore archive's provenance chain, picked by
EARLIEST ARRIVAL at a progress level (pbfr1b1, 2026-10-29).

    python tools/ge_chain_spine.py ARCHIVE.npz OUT.npy --level 3.0 [--exclude-box x0 y0 z0 x1 y1 z1]

For an archive rooted at the map's true spawn (tools/explore_phase1.py; with or without
--exact-chains), the cells whose distance on the shaping field is at least LEVEL % of the start's
distance below it are ranked by depth (physics ticks from the spawn along their chain) and the
shortest chain wins: root -> ... -> that cell, each row the cell's archived state. Without
--exact-chains the archive keeps every cell's SHORTEST arrival and re-parents it, so the chain is
the explorer's fastest known way there, stitched from several runs - a time-ordered list of
states, which is all a backward curriculum (train_fast --demo-file) spawns from; it is not a
replayable line. Views are re-aimed along the velocity (tools/spine_canon_view.py).
--exclude-box (repeatable) skips chains with any state whose origin lies in that box - a way to
keep a SUSPECT simulator move (one not yet checked against the engine) out of a spine; it is a
validity filter on the simulator, never a recipe component. Prints a JSON summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from spine_canon_view import canon_view                     # noqa: E402
from surfgym.core import STATE_DTYPE                        # noqa: E402
from surfgym.goalfield import GoalField                     # noqa: E402


def chain_to(parent: np.ndarray, i: int) -> list:
    out, seen = [], set()
    while i >= 0 and i not in seen:
        seen.add(i)
        out.append(int(i))
        i = int(parent[i])
    return out[::-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--level", type=float, required=True, help="progress %% of the start's distance")
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"))
    ap.add_argument("--exclude-box", type=float, nargs=6, action="append", default=[],
                    metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    ap.add_argument("--max-tries", type=int, default=2000)
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    st, parent, depth = z["state"], z["parent"], z["depth"]
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    o = st["origin"].astype(np.float64)
    d = f.sample(o).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    roots = np.flatnonzero(parent < 0)
    d0 = float(np.min(d[roots]))
    cand = np.flatnonzero(d <= d0 * (1.0 - a.level / 100.0))
    cand = cand[np.argsort(depth[cand], kind="stable")]
    boxes = [np.asarray(b, np.float64).reshape(2, 3) for b in a.exclude_box]
    pick, chain, skipped = -1, [], 0
    for c in cand[:a.max_tries]:
        ch = chain_to(parent, int(c))
        oc = o[ch]
        bad = any(np.any(np.all((oc >= b[0]) & (oc <= b[1]), axis=1)) for b in boxes)
        if bad:
            skipped += 1
            continue
        pick, chain = int(c), ch
        break
    if pick < 0:
        raise SystemExit(f"no chain to {a.level}% outside the excluded boxes ({len(cand)} cells, {skipped} skipped)")
    sp = np.asarray(st[np.asarray(chain, np.int64)], STATE_DTYPE)
    sp, m = canon_view(sp)
    sp = np.array(sp, copy=True)
    sp["tick"] = 0
    sp["stuck_ticks"] = 0
    np.save(a.out, sp)
    dc = d[chain]
    hh = o[chain, 2] + (st["velocity"][chain].astype(np.float64) ** 2).sum(1) / 1600.0
    info = {"cell": pick, "rows": len(chain), "depth_ticks": int(depth[pick]), "skipped_chains": skipped,
            "start_d": round(float(dc[0]), 1), "end_d": round(float(dc[-1]), 1),
            "end_pct": round(100.0 * (d0 - float(dc[-1])) / d0, 3),
            "max_d": round(float(np.max(dc[np.isfinite(dc)])), 1), "h_max": round(float(hh.max()), 1),
            "end_origin": [round(float(v), 1) for v in sp[-1]["origin"]], "canon_rows": int(m.sum()),
            "md5": hashlib.md5(Path(a.out).read_bytes()).hexdigest()}
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
