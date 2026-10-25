#!/usr/bin/env python3
"""spine_canon_keys.py - a spine in the POLICY's frame, its held keys translated with it (pbr4b3,
2026-10-25).

tools/spine_canon_view.py puts a searched line's view along the velocity (the frame a
--view-absolute velocity policy flies in), but a search's KEYS were pressed relative to the
search's own view: forward held with the view 89 deg left of the velocity is, physically, a left
strafe. Canonicalising the view alone therefore hides the manoeuvre (round 4, gate 12: the
explorer surfed a ramp with forward + a view far left of the velocity; the policy, spawned with
its view along the velocity and its keys reset, strafed right and fell).

This tool takes a spine (STATE_DTYPE .npy) and the held keys each row was reached with ((n, 3)
fwd, side, duck in engine bins: fwd 0/1/2 = back/none/forward, side 0/1/2 = left/none/right) and
writes, for every row whose horizontal speed is >= --min-speed:

* the view along the velocity (spine_canon_view's rule), and
* the (fwd, side) pair whose wish direction, relative to that canonical view, is the closest of
  the eight key directions to the wish direction the original keys had relative to the original
  view - so the physical input survives the change of frame to within 22.5 deg. No wish (both
  none) stays none; duck is unchanged.

Slower rows keep their view and keys (below 100 u/s the policy's view is a per-tick delta).
The keys go to train_fast --demo-keys.

    python tools/spine_canon_keys.py SPINE.npy KEYS.npy OUT.npy OUT_KEYS.npy [--min-speed 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# engine bins -> unit move: fwd bin 2 = +forward, side bin 0 = LEFT (sidemove -400)
_FWD = np.array([-1.0, 0.0, 1.0])
_LEFT = np.array([1.0, 0.0, -1.0])
_DIRS = [(f, s) for f in (0, 1, 2) for s in (0, 1, 2) if not (f == 1 and s == 1)]


def wish_angle(fwd, side):
    """Wish direction of a key pair relative to the view, deg (0 = ahead, +90 = left); NaN =
    no key held."""
    x = _FWD[np.asarray(fwd, np.int64)]
    y = _LEFT[np.asarray(side, np.int64)]
    out = np.degrees(np.arctan2(y, x))
    return np.where((x == 0.0) & (y == 0.0), np.nan, out)


def translate(keys, yaw, heading):
    """(n, 3) keys at view ``yaw`` -> keys at view ``heading`` with the closest wish direction."""
    keys = np.asarray(keys, np.int64).copy()
    phi = wish_angle(keys[:, 0], keys[:, 1]) + (np.asarray(yaw, np.float64)
                                                - np.asarray(heading, np.float64))
    cand = np.array([wish_angle(f, s) for f, s in _DIRS], np.float64)
    for i in np.flatnonzero(~np.isnan(phi)):
        dd = np.abs((cand - phi[i] + 180.0) % 360.0 - 180.0)
        keys[i, 0], keys[i, 1] = _DIRS[int(np.argmin(dd))]
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spine")
    ap.add_argument("keys")
    ap.add_argument("out")
    ap.add_argument("out_keys")
    ap.add_argument("--min-speed", type=float, default=100.0)
    a = ap.parse_args(argv)
    sp = np.array(np.load(a.spine), copy=True)
    keys = np.asarray(np.load(a.keys), np.int64)
    if keys.shape != (len(sp), 3):
        raise SystemExit(f"keys {keys.shape} do not match the spine's {len(sp)} rows")
    v = np.asarray(sp["velocity"], np.float64)
    fast = np.hypot(v[:, 0], v[:, 1]) >= float(a.min_speed)
    hd = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    out_keys = keys.copy()
    out_keys[fast] = translate(keys[fast], np.asarray(sp["yaw"], np.float64)[fast], hd[fast])
    sp["yaw"][fast] = hd[fast].astype(sp["yaw"].dtype)
    np.save(a.out, sp)
    np.save(a.out_keys, out_keys.astype(np.int32))
    ch = int((out_keys[:, :2] != keys[:, :2]).any(axis=1).sum())
    print(f"{len(sp)} rows, {int(fast.sum())} put along the velocity, {ch} with their keys "
          f"translated -> {a.out}, {a.out_keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
