#!/usr/bin/env python3
"""line_actions.py - recover the per-tick actions of a STATE_DTYPE line (pbr2b2, 2026-10-24).

A line replayed by tools/ge_replay.py (or any time-ordered spine of exact consecutive states) keeps
only the states, not the actions that produced them. The core is deterministic, so the actions
can be read back one tick at a time: from row t, step every forward/side key pair (3 x 3) with
the jump and duck bits row t+1's oldbuttons records and the view moved to row t+1's yaw/pitch
(core view_mode 2, a world-yaw target, exact within the 10 deg/tick ceiling), and keep the pair
that lands on row t+1. Nothing comes from a demo or a record: the line is the input.

Output (.npz): keys (n, 4) int32 engine bins [fwd, side, jump, duck] for the tick FROM each row
(the last row repeats the one before), err (n,) the landing error in u (origin + velocity), and
the line's md5. A row whose best pair misses by more than --tol is reported; the curriculum
(train_fast --demo-keys) uses the keys as the held state an exact spawn starts with.

    python tools/line_actions.py --map maps/X.bsp --line autoresearch/spines/line.npy \\
        --out autoresearch/spines/line.keys.npz
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402

PAIRS = [(f, s) for f in range(3) for s in range(3)]


def infer(core: SurfCore, line: np.ndarray, maxvel=None):
    n = len(line)
    keys = np.zeros((n, 4), np.int32)
    err = np.zeros(n, np.float64)
    m = len(PAIRS)
    acts = np.zeros((m, 6), np.int32)
    acts[:, 0], acts[:, 1] = 7, 3
    acts[:, 2] = [p[0] for p in PAIRS]
    acts[:, 3] = [p[1] for p in PAIRS]
    view = np.zeros((m, 2), np.float32)
    core.reset(0)
    for t in range(n - 1):
        s0 = line[t:t + 1].copy()
        s0["tick"] = 0
        s0["stuck_ticks"] = 0
        for i in range(m):
            core.set_state(i, s0[0])
        nb = int(line[t + 1]["oldbuttons"])
        acts[:, 4] = 1 if nb & 2 else 0
        acts[:, 5] = 1 if nb & 4 else 0
        view[:, 0] = float(line[t + 1]["yaw"])
        view[:, 1] = float(line[t + 1]["pitch"])
        core.step(acts, view=view)
        st = core.get_states()
        e = (np.linalg.norm(st["origin"].astype(np.float64)
                            - line[t + 1]["origin"].astype(np.float64), axis=1)
             + 0.01 * np.linalg.norm(st["velocity"].astype(np.float64)
                                     - line[t + 1]["velocity"].astype(np.float64), axis=1))
        k = int(np.argmin(e))
        keys[t] = (PAIRS[k][0], PAIRS[k][1], acts[0, 4], acts[0, 5])
        err[t] = float(e[k])
    keys[-1] = keys[-2] if n > 1 else keys[-1]
    err[-1] = err[-2] if n > 1 else 0.0
    return keys, err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--line", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxvel", type=float, default=3500.0)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args(argv)
    line = np.load(a.line, allow_pickle=False)
    if line.dtype != STATE_DTYPE:
        raise SystemExit(f"{a.line}: not a STATE_DTYPE array")
    cfg = default_config(num_envs=len(PAIRS), spawn_mode=2, max_episode_ticks=1_000_000,
                         water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0,
                         sv_maxvelocity=float(a.maxvel), view_mode=2)
    core = SurfCore(str(a.map), cfg)
    core.set_teleport_fail(True)
    keys, err = infer(core, line)
    bad = np.flatnonzero(err[:-1] > a.tol)
    md5 = hashlib.md5(Path(a.line).read_bytes()).hexdigest()
    np.savez(a.out, keys=keys, err=err, line_md5=md5)
    print(f"{len(line)} rows -> {a.out}; landing error max {err[:-1].max():.4f} u, "
          f"{len(bad)} rows above --tol {a.tol}" + (f" (first {bad[:10].tolist()})" if len(bad)
                                                   else ""))
    held = np.array([k[0] != 1 or k[1] != 1 for k in keys])
    print(f"fwd/side held on {held.mean():.0%} of ticks; duck on {keys[:, 3].mean():.0%}")
    return 0 if not len(bad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
