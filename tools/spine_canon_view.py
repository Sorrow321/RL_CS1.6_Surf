#!/usr/bin/env python3
"""spine_canon_view.py - put a spine's VIEW where the policy's view parametrisation lives.

A line found by a random explorer (tools/explore_phase1.py --exact-chains + ge_replay.py, or
entry_search.py) carries whatever view yaw its uniform actions happened to leave: on
pbr1b1_line_a.npy the view is a median 176 deg off the velocity heading through the whole
flight from the ridge launch to the wall bounce. A --view-absolute velocity policy commands its
yaw as an OFFSET from the velocity heading (it looks within +-7 deg of it at speed), and the
core turns the view at most atan(30 / |v|) per tick (--yaw-adaptive: ~4 deg/tick at 420 u/s),
so a curriculum spawn on such a state spends its first ~35 ticks swinging the view round while
its keys push in directions it never learned - a SPAWN-SEMANTICS defect of the same family as
the unducked crouched spawns (--keys-hold-spawn-duck).

The physics state does not depend on the view (it only reaches the dynamics through the next
usercmd), so the canonical spawn is the same state with the view along the velocity: every row
whose horizontal speed is >= --min-speed (the core's absolute-view regime, src/env.c, 100 u/s)
gets yaw = the velocity heading. Slower rows keep their yaw: there the policy's command is a
per-tick delta from the current view, and any choice would be arbitrary. Pitch is untouched (it
moves only the camera).

    python tools/spine_canon_view.py autoresearch/spines/pbr1b1_line_a.npy \
        autoresearch/spines/pbr2b3_line_a_canon.npy
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def canon_view(states: np.ndarray, min_speed: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """A copy of ``states`` with yaw = the velocity heading (degrees, [0, 360)) on every row
    whose horizontal speed is >= ``min_speed``; returns (copy, mask of the rows changed)."""
    out = np.array(states, copy=True)
    v = out["velocity"].astype(np.float64)
    hs = np.hypot(v[:, 0], v[:, 1])
    m = hs >= float(min_speed)
    hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    out["yaw"][m] = hd[m].astype(out["yaw"].dtype)
    return out, m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-speed", type=float, default=100.0)
    a = ap.parse_args(argv)
    from surfgym.core import STATE_DTYPE
    s = np.load(a.src)
    if s.dtype != STATE_DTYPE:
        raise SystemExit(f"{a.src}: not a STATE_DTYPE array ({s.dtype})")
    out, m = canon_view(s, a.min_speed)
    v = s["velocity"].astype(np.float64)
    hd = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    off = np.abs((s["yaw"].astype(np.float64) - hd + 180.0) % 360.0 - 180.0)
    np.save(a.dst, out)
    md5 = hashlib.md5(Path(a.dst).read_bytes()).hexdigest()
    print(f"{a.dst}: {len(out)} states, {int(m.sum())} re-aimed along the velocity (|v_h| >= "
          f"{a.min_speed:g}; their view was a median {np.median(off[m]):.1f} deg off it), "
          f"md5 {md5}")


if __name__ == "__main__":
    main()
