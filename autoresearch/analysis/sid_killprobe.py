"""ANALYSIS ONLY (never training): map kill triggers / solid / free space with the simulator itself.

A standing (or ducked) player is placed at each grid point with v = 0: 'K' = it fails on the first tick
(inside a kill trigger - the core's exact brush test, holes included, which the trigger AABBs in the bsp
entity list hide), '#' = a zero-length hull trace starts solid, '.' = free. Run from the repo root:

    python autoresearch/analysis/sid_killprobe.py xy x0 x1 dx y0 y1 dy z     [duck]
    python autoresearch/analysis/sid_killprobe.py xz x0 x1 dx z0 z1 dz y     [duck]
    python autoresearch/analysis/sid_killprobe.py yz y0 y1 dy z0 z1 dz x     [duck]

Rows print top-down (largest y or z first). A probe of ~3,000 points takes a few seconds on one CPU core.
Mind the grid: a 64 u step missed the 32 u walls of sidistic's finish shaft (r5b3, 2026-10-22)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from surfgym.core import SurfCore, default_config  # noqa: E402

BSP = str(Path(__file__).resolve().parents[2] / "maps" / "surf_src_sidistic.bsp")


def probe(pts, bsp=BSP, ducked=False):
    """(n, 3) points -> (kill (n,) bool, solid (n,) bool)."""
    n = len(pts)
    core = SurfCore(bsp, default_config(num_envs=n))
    core.set_teleport_fail(True)
    core.reset(0)
    st = core.get_states()
    hull = 1 if ducked else 0
    solid = np.zeros(n, bool)
    for i, p in enumerate(pts):
        s = st[i:i + 1].copy()
        s["origin"][0] = p
        s["velocity"][0] = 0.0
        s["basevelocity"][0] = 0.0
        s["onground"][0] = -1
        s["tick"][0] = 0
        s["stuck_ticks"][0] = 0
        s["ducked"][0] = 1 if ducked else 0
        core.set_state(i, s)
        solid[i] = bool(core.trace(p, p, hull=hull).startsolid)
    act = np.zeros((n, 6), np.int32)
    act[:, 0] = 7
    _, _, done, _, _ = core.step(act)
    kill = done.astype(bool).copy()
    core.close()
    return kill, solid


def main(argv):
    mode = argv[0]
    a0, a1, da, b0, b1, db, c = map(float, argv[1:8])
    ducked = len(argv) > 8 and argv[8] == "duck"
    us = np.arange(a0, a1 + 1e-6, da)
    vs = np.arange(b0, b1 + 1e-6, db)
    if mode == "xy":
        pts = [(u, v, c) for v in vs for u in us]
    elif mode == "xz":
        pts = [(u, c, v) for v in vs for u in us]
    elif mode == "yz":
        pts = [(c, u, v) for v in vs for u in us]
    else:
        raise SystemExit("mode is xy, xz or yz")
    k, s = probe(np.array(pts, np.float32), ducked=ducked)
    k = k.reshape(len(vs), len(us))
    s = s.reshape(len(vs), len(us))
    fixed = {"xy": "z", "xz": "y", "yz": "x"}[mode]
    print(f"{mode} slice at {fixed} = {c:g}: cols {a0:g}..{a1:g} step {da:g}, "
          f"rows {b1:g} (top) .. {b0:g}, {'ducked' if ducked else 'standing'}")
    for j in range(len(vs) - 1, -1, -1):
        print("%6d " % vs[j] + "".join("K" if k[j, i] else ("#" if s[j, i] else ".")
                                       for i in range(len(us))))


if __name__ == "__main__":
    main(sys.argv[1:])
