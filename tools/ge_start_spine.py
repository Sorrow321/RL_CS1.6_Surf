#!/usr/bin/env python3
"""ge_start_spine.py - a curriculum spine from ONE Go-Explore exact chain rooted at the map start
(pbfr1b1, 2026-10-29).

    python tools/ge_start_spine.py LINE.npy OUT.npy [--field F] [--end-margin 0]

LINE.npy is tools/ge_replay.py's per-tick line (its first row = the archive's root, the map's true
spawn for an archive rooted there; ge_replay also writes LINE.keys.npy). The spine is the line with
the explorer's views re-aimed along the velocity (tools/spine_canon_view.py: a --view-absolute
velocity policy looks within a few degrees of it) and cut at its best (lowest) distance on the
shaping field - the backward curriculum's success line is the spine's last state
(train_fast --demo-success-d). Writes OUT.npy, OUT.keys.npy (held fwd / side / duck, engine bins,
cut alike) and prints a JSON summary. Nothing here names a place: on any map it is "the explorer's
deepest line from the start, as a spine"."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("line")
    ap.add_argument("out")
    ap.add_argument("--field", default=str(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz"))
    a = ap.parse_args(argv)
    line = np.load(a.line)
    kp = Path(a.line).with_suffix(".keys.npy")
    keys = np.load(kp) if kp.exists() else None
    if keys is not None and len(keys) != len(line):
        raise SystemExit(f"{kp}: {len(keys)} rows for a {len(line)}-row line")
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    d = f.sample(line["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    e = int(np.argmin(d))
    sp, m = canon_view(np.asarray(line[:e + 1], STATE_DTYPE))
    sp = np.array(sp, copy=True)
    sp["tick"] = 0
    sp["stuck_ticks"] = 0
    np.save(a.out, sp)
    if keys is not None:
        np.save(str(Path(a.out).with_suffix(".keys.npy")), keys[:e + 1].astype(np.int64))
    d0 = float(d[0])
    info = {"n": int(len(sp)), "start_d": round(d0, 1), "end_d": round(float(d[e]), 1),
            "end_gain_pct": round(100.0 * (d0 - float(d[e])) / d0, 3),
            "max_d": round(float(np.max(d[:e + 1][np.isfinite(d[:e + 1])])), 1),
            "end_origin": [round(float(v), 1) for v in sp[-1]["origin"]],
            "canon_rows": int(m.sum()), "keys": keys is not None,
            "md5": hashlib.md5(Path(a.out).read_bytes()).hexdigest()}
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
