#!/usr/bin/env python3
"""fail_probe.py - paint where the player dies, from the core's own trigger test (ANALYSIS ONLY).

A map's fail volumes are trigger brushes, and their AABBs are badly misleading on maps built from
slanted fail sheets: on surf_src_sidistic the agent's surviving lines pass straight through the
AABB of `*99`, and the CP2 "pit" turned out to be a thin slanted sheet under a ramp, not the box
`*86`. This places a STILL standing hull on a grid over one plane, steps the core one tick with
--teleport-fail on, and paints every point whose episode ends. Grey is solid (the baked occupancy
grid), orange is death. Trajectories (.jsonl, record_ckpt format) can be drawn over it; a dot
every second.

The probe is a measurement instrument, like the gate boxes and BEV plots (CLAUDE.md 0b): what it
finds never enters a reward, a spawn rule or a schedule. A still hull misses a sheet thinner
than the grid step; a MOVING hull touches a sheet ~14 u away in one tick at 2,000 u/s (measured
on sidistic), so read the map to within one cell.

    python tools/fail_probe.py --map maps/surf_src_sidistic.bsp --plane xz --at -900 \
        --a 0 7200 --b 10000 13600 --step 32 --traj runs/x/traj_*.jsonl --out side.png
    python tools/fail_probe.py --map maps/surf_src_sidistic.bsp --plane xy --at 11450 \
        --a 1500 7000 --b -1400 1400 --out top.png
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import SurfCore, default_config      # noqa: E402

AX = {"xz": (0, 2, 1), "xy": (0, 1, 2), "yz": (1, 2, 0)}   # (horizontal, vertical, fixed)


def probe(core, n, pts):
    """(M, 3) points -> (M,) bool: True where a still standing hull dies on the next tick."""
    out = np.zeros(len(pts), bool)
    st0 = core.get_states()
    act = np.zeros((n, 6), np.int32)
    act[:, 0], act[:, 1], act[:, 2], act[:, 3] = 7, 3, 1, 1      # neutral
    for s in range(0, len(pts), n):
        p = pts[s:s + n]
        st = st0.copy()
        st["velocity"][:] = 0
        st["basevelocity"][:] = 0
        st["onground"] = -1
        st["tick"] = 10
        st["origin"][:len(p)] = p
        st["origin"][len(p):] = p[0]
        for i in range(n):
            core.set_state(i, st[i])
        core.step(act)
        out[s:s + len(p)] = np.asarray(core._done[:len(p)], bool)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--plane", choices=sorted(AX), default="xz")
    ap.add_argument("--at", type=float, required=True, help="the fixed coordinate of the plane")
    ap.add_argument("--a", type=float, nargs=2, required=True, help="horizontal range")
    ap.add_argument("--b", type=float, nargs=2, required=True, help="vertical range")
    ap.add_argument("--step", type=float, default=32.0)
    ap.add_argument("--px", type=int, default=3, help="pixels per grid step")
    ap.add_argument("--traj", nargs="*", default=[], help="trajectories to draw (globs)")
    ap.add_argument("--near", type=float, default=400.0,
                    help="draw only trajectory points within this distance of the plane")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import cv2
    ha, va, fa = AX[a.plane]
    hs = np.arange(a.a[0], a.a[1], a.step) + a.step / 2
    vs = np.arange(a.b[0], a.b[1], a.step) + a.step / 2
    H, V = np.meshgrid(hs, vs)
    pts = np.zeros((H.size, 3), np.float32)
    pts[:, ha], pts[:, va], pts[:, fa] = H.ravel(), V.ravel(), a.at
    core = SurfCore(str(a.map), default_config(num_envs=a.envs, max_episode_ticks=100000))
    core.set_teleport_fail(True)
    core.reset(0)
    dead = probe(core, a.envs, pts).reshape(V.shape)
    solid = np.zeros(V.shape, bool)
    occ_p = Path(a.map).with_name(Path(a.map).stem + ".occ_32.npz")
    if occ_p.exists():
        z = np.load(occ_p, allow_pickle=False)
        O, mins = z["occ"], z["mins"]
        idx = np.floor((pts.astype(np.float64) - mins) / 32.0).astype(np.int64)
        okk = np.all((idx >= 0) & (idx < np.array(O.shape[::-1])), axis=1)
        sol = np.zeros(len(pts), bool)
        sol[okk] = O[idx[okk, 2], idx[okk, 1], idx[okk, 0]] > 0
        solid = sol.reshape(V.shape)
    img = np.full(V.shape + (3,), 255, np.uint8)
    img[solid] = (140, 140, 140)
    img[dead] = (0, 165, 255)
    img = cv2.resize(img, (len(hs) * a.px, len(vs) * a.px),
                     interpolation=cv2.INTER_NEAREST)[::-1].copy()
    hh = img.shape[0]

    def P(u, w):
        return (int((u - a.a[0]) / a.step * a.px), int(hh - (w - a.b[0]) / a.step * a.px))
    cols = [(255, 0, 0), (0, 0, 255), (0, 128, 0), (128, 0, 128)]
    for k, pat in enumerate(a.traj):
        for path in glob.glob(pat):
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.startswith("[")]
            r = np.asarray(rows, np.float64)
            if not len(r):
                continue
            near = np.abs(r[:, 1 + fa] - a.at) < a.near
            for i in range(1, len(r)):
                if near[i] and near[i - 1] and r[i, 0] > r[i - 1, 0]:
                    cv2.line(img, P(r[i - 1, 1 + ha], r[i - 1, 1 + va]),
                             P(r[i, 1 + ha], r[i, 1 + va]), cols[k % len(cols)], 1)
            for i in range(0, len(r), 100):
                if near[i]:
                    cv2.circle(img, P(r[i, 1 + ha], r[i, 1 + va]), 2, cols[k % len(cols)], -1)
    lab = "xyz"
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"{Path(a.map).stem} {a.plane} at {lab[fa]}={a.at:g}: grey solid, orange "
                f"DEATH (still hull, core trigger test), step {a.step:g}u", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.imwrite(a.out, np.vstack([bar, img]))
    print(f"{int(dead.sum()):,} of {dead.size:,} probe points die -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
