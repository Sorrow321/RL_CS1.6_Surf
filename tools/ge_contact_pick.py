#!/usr/bin/env python3
"""ge_contact_pick.py - the deepest CONTACT cell of a Go-Explore archive (zlr2b1, 2026-11-01).

tools/ge_replay.py replays the archive's deepest cell on the goal field by default. On a map whose
field flies through open air (surf_za_loop) that cell is a ballistic apex over the void: a line
to it ends in a fall. This picks instead the deepest cell whose state the map HOLDS: walkable
ground (onground != -1), or a surfable face pushing back up over the next tick (the
--ratchet-contact test: step the state one tick with no input, vz gave back more than free fall,
the -sv_maxvelocity clamp excluded). Prints the cell index for `ge_replay.py --cell` and a
progress histogram of the contact cells. Nothing here reads a place on the map.

    python tools/ge_contact_pick.py --map maps/X.bsp --archive runs/ge/archive.npz \\
        --field maps/X.goalc_32.npz [--d0 6361.8] [--top 400000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402
from surfgym.goalfield import load_goal_field                     # noqa: E402


def contact_cells(core, states: np.ndarray, msec: float, vmax: float, batch: int,
                  tol: float = 1.0, gravity: float = 800.0) -> np.ndarray:
    """bool per state: ground, or one null-input tick from it shows the map pushing back up."""
    out = np.zeros(len(states), bool)
    null = np.zeros((batch, 6), np.int32)
    null[:, 0], null[:, 1], null[:, 2], null[:, 3] = 7, 3, 1, 1
    for b0 in range(0, len(states), batch):
        chunk = states[b0:b0 + batch]
        core.reset(0)
        for i in range(len(chunk)):
            s = chunk[i:i + 1].copy()
            s["tick"] = 0
            s["stuck_ticks"] = 0
            core.set_state(i, s)
        _, _, done, trunc, _ = core.step(null)
        after = core.get_states()[:len(chunk)]
        vz0 = chunk["velocity"][:, 2].astype(np.float64)
        vz1 = after["velocity"][:, 2].astype(np.float64)
        dev = (vz1 - vz0) + gravity * msec / 1000.0
        alive = ~((np.asarray(done)[:len(chunk)] | np.asarray(trunc)[:len(chunk)]) != 0)
        surf = alive & (chunk["onground"] == -1) & (dev > tol) & (vz1 > -(vmax - 0.5))
        out[b0:b0 + len(chunk)] = (chunk["onground"] != -1) | surf
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--d0", type=float, default=None, help="progress normaliser (default: the "
                    "field at the archive's first cell)")
    ap.add_argument("--top", type=int, default=400000,
                    help="test only the TOP deepest cells on the field (0 = all)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--json", default=None, help="also write the summary here")
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    meta = json.loads(Path(a.archive).with_suffix(".meta.json").read_text())
    st = z["state"]
    f = load_goal_field(a.field)
    d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    d0 = float(a.d0) if a.d0 else float(d[0])
    order = np.argsort(d)
    if a.top:
        order = order[:a.top]
    order = order[np.isfinite(d[order])]
    msec = float(meta.get("tick_ms") or 10)
    vmax = float(meta.get("maxvel") or 2000.0)
    cfg = default_config(num_envs=a.batch, spawn_mode=2,
                         max_episode_ticks=int(meta.get("ep_ticks", 6000)),
                         water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    cfg.phys.sv_maxvelocity = vmax
    cfg.phys.msec = int(msec)
    core = SurfCore(str(a.map), cfg)
    if not meta.get("keep_teleports", False):
        core.set_teleport_fail(True)
    c = contact_cells(core, st[order], msec, vmax, a.batch)
    cand = order[c]
    prog = 100.0 * (d0 - d) / d0
    print(f"archive {len(st):,} cells; tested the {len(order):,} deepest; {len(cand):,} are "
          f"contact cells")
    print(f"deepest cell overall: {int(order[0])} at {prog[order[0]]:.2f}% "
          f"{np.asarray(st[order[0]]['origin']).round(0)}")
    res = {"cells": int(len(st)), "tested": int(len(order)), "contact": int(len(cand))}
    if len(cand):
        best = int(cand[0])
        o = np.asarray(st[best]["origin"]).round(0)
        v = np.asarray(st[best]["velocity"]).round(0)
        print(f"DEEPEST CONTACT cell {best}: {prog[best]:.2f}% d {d[best]:,.0f} at {o} v {v} "
              f"onground {int(st[best]['onground'])}")
        edges = [-100, 0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 62, 64, 66, 70, 80, 100]
        h, _ = np.histogram(prog[cand], bins=edges)
        print("contact cells by progress band: " + ", ".join(
            f"[{lo},{hi}) {n:,}" for lo, hi, n in zip(edges[:-1], edges[1:], h) if n))
        print("top 12 contact cells:")
        for i in cand[:12]:
            print(f"  cell {int(i)}: {prog[i]:6.2f}%  at {np.asarray(st[i]['origin']).round(0)}  "
                  f"|v| {float(np.linalg.norm(st[i]['velocity'])):5.0f}  og {int(st[i]['onground'])}")
        res.update({"best_cell": best, "best_pct": float(prog[best]), "best_d": float(d[best]),
                    "best_origin": o.tolist()})
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
