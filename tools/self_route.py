#!/usr/bin/env python3
"""self_route.py - a champion-free reference route: a Go-Explore chain from the map's TRUE START,
continued by the goal field's own steepest-descent line to the finish.

    python tools/self_route.py runs/ge/chain.npy --map maps/<m>.bsp --out runs/ge/self_route.npz

The first part is the chain tools/archive_spawns.py cuts out of a tools/explore_phase1.py archive
(reward-free random macro-action bursts from the true start, returned to by state restore: no
demo, no policy, no reward) - the only part of the line that knows the way through a deceptive
start. From the chain's last state on, the line is tools/build_route.py --from-field's walk down
the baked goal field (the no-corner-cut field the recipe shapes on, unless --field goal), which
is a function of the geometry alone. Resampled at constant arc length (DEFAULT_SPACING) and
written in the format surfgym.route.ArcProgress / RouteLine load, for tools/beam_tas.py
--objective progress --route-file and tools/plan_to_bc.py --route. No map constant: the chain is
whatever the search found and the tail is whatever the field says.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chain", help="STATE_DTYPE .npy, root first (tools/archive_spawns.py)")
    ap.add_argument("--map", required=True)
    ap.add_argument("--field", default="goalc", choices=("goalc", "goal"))
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from surfgym.goalfield import GoalField
    from surfgym.route import DEFAULT_SPACING, resample_polyline
    from build_route import trace_descent
    bsp = Path(a.map)
    z = np.load(bsp.parent / f"{bsp.stem}.{a.field}_{a.cell}.npz", allow_pickle=False)
    field = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                      float(z["cell"]), float(z["reach_max"]))
    ch = np.load(a.chain)
    head = ch["origin"].astype(np.float64)
    tail, info = trace_descent(field, head[-1])
    pts = np.concatenate([head, tail[1:]])
    line, total = resample_polyline(pts, DEFAULT_SPACING)
    out = Path(a.out)
    np.savez_compressed(out, route=np.asarray(line, np.float32),
                        spacing=np.float32(DEFAULT_SPACING), map=str(bsp.stem),
                        source="self: explore_phase1 chain + field descent",
                        chain_len=np.int32(len(head)), d_end=np.float32(info["d_end"]))
    meta = {"chain": str(a.chain), "chain_len": int(len(head)), "field": f"{a.field}_{a.cell}",
            "descent_steps": int(info["n_steps"]), "descent_d0": float(info["d0"]),
            "descent_d_end": float(info["d_end"]), "points": int(len(line)),
            "length_u": float(total)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    if info["d_end"] > 2.0 * a.cell:
        print(f"WARNING: the descent stopped in a local minimum at d {info['d_end']:,.0f}u")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
