#!/usr/bin/env python3
"""archive_to_spawn.py - a Go-Explore archive (tools/explore_phase1.py) -> a STATE_DTYPE .npy for
the trainer's --spawn-states.

The archive holds, per cell, the first state an exploration run from the MAP START reached there
(random held-key bursts in the simulator: no demo, no policy, no reward, no route). This keeps
the cells whose goal-field distance is below the start's - the part of the explored space that
is ahead of the spawn on the field the reward shapes on - zeroes each row's episode clocks, and
writes a .json sidecar with the provenance and the progress range. Nothing here reads the map
beyond its own goal field (the same field the reward and the grader use).

    python tools/archive_to_spawn.py runs/ge/archive.npz runs/ge/spawn.npy \
        --field maps/surf_princessburglar_b26.goalc_32.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def load_field(path):
    from surfgym.goalfield import GoalField
    z = np.load(path, allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def select(states, d, d0: float, min_progress: float = 0.0, reach_max: float = np.inf):
    """Rows whose field progress 1 - d / d0 exceeds min_progress (and that are on the field)."""
    d = np.asarray(d, np.float64)
    ok = (d < reach_max) & (d < d0 * (1.0 - float(min_progress)))
    out = np.array(states[ok], copy=True)
    for k in ("tick", "stuck_ticks"):
        if k in out.dtype.names:
            out[k] = 0
    return out, d[ok]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("out")
    ap.add_argument("--field", required=True, help="the goal field .npz the reward shapes on")
    ap.add_argument("--start-d", type=float, default=None,
                    help="the start's field distance (default: the archive's root row, depth 0)")
    ap.add_argument("--min-progress", type=float, default=0.0,
                    help="keep rows with 1 - d/d0 above this (default 0: ahead of the start)")
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    st, depth = z["state"], z["depth"]
    f = load_field(a.field)
    d = f.sample(st["origin"]).astype(np.float64)
    d0 = float(a.start_d) if a.start_d else float(d[int(np.argmin(depth))])
    out, dk = select(st, d, d0, a.min_progress, f.reach_max - 0.5 * f.cell)
    np.save(a.out, out)
    meta = {"archive": str(a.archive), "rows": int(len(out)), "of": int(len(st)),
            "start_d": d0, "min_progress": float(a.min_progress),
            "progress_max": float(1.0 - dk.min() / d0) if len(dk) else None,
            "provenance": "Go-Explore phase 1 (tools/explore_phase1.py) from the map start: "
                          "random held-key bursts in the simulator, cells = position (+ speed); "
                          "no demo, no policy, no reward, no route"}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
