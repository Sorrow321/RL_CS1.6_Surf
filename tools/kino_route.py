#!/usr/bin/env python3
"""kino_route.py - the route a tools/kino_search.py archive found: rebuild the control sequence
of its fastest goal hit, REPLAY it in the engine from the map's spawn (it must reproduce the
search exactly - every edge was simulated from its parent's exact state), and save the flight.

    python tools/kino_route.py runs/research/kino/uf2.npz [--ref runs/research/uf2_wr/surf_unitfarmer2.frames.npz]

--ref is a MEASURING STICK only (CLAUDE.md section 0: a record is analysis, never training):
it prints where the found route and the reference are at the same game times, and the plot
draws both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym import SurfCore, default_config          # noqa: E402
from surfgym.zones import load_zones                  # noqa: E402
from kino_search import states_rw                     # noqa: E402


def chain(z, node):
    """root -> node: [(control, ticks)] along the archive's parent links."""
    par, ctl, tick = z["parent"], z["ctrl"], z["tick"]
    seq = []
    while par[node] >= 0:
        seq.append((int(ctl[node]), int(tick[node] - tick[par[node]])))
        node = int(par[node])
    return seq[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--map", default=None, help="the .bsp (default: maps_pool/ or maps/<map>.bsp)")
    ap.add_argument("--ref", default=None, help="a parsed demo (frames.npz) to compare against")
    ap.add_argument("--maxvel", type=float, default=4000.0)
    ap.add_argument("--plot", default=None, help="write a top-down + height plot (png)")
    a = ap.parse_args()

    z = np.load(a.archive)
    stem = str(z["map"])
    bsp = Path(a.map) if a.map else next(p for p in (ROOT / "maps_pool" / f"{stem}.bsp",
                                                     ROOT / "maps" / f"{stem}.bsp") if p.exists())
    goals = z["goals"]
    if not len(goals):
        raise SystemExit("this archive has no goal hit")
    g = goals[np.argmin(goals[:, 2])]
    node, gctl, garr = int(g[0]), int(g[1]), int(g[2])
    seq = chain(z, node)
    controls = z["controls"]
    print(f"fastest goal hit: {garr / 100:.2f} s, {len(seq) + 1} edges "
          f"(the last: control {gctl} into the finish)")

    zones = load_zones(str(bsp))
    core = SurfCore(str(bsp), default_config(
        num_envs=1, lidar_w=0, lidar_h=0, view_mode=1, lookahead_k=0,
        sv_maxvelocity=a.maxvel, max_episode_ticks=10 ** 8, spawn_mode=1, yaw_jitter_deg=0.0))
    core.set_teleport_fail(True)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    core.reset(0)
    rw = states_rw(core)
    rw["velocity"][0] = (0.0, 0.0, 0.0)
    rw["tick"][0] = 0
    pos, vel, t = [], [], 0
    finished = died = False
    for ci, ticks in seq + [(gctl, 10 ** 6)]:
        c = controls[ci]
        act = np.array([[7, 3, int(c[0]), int(c[1]), int(c[2]), 0]], np.int32)
        view = np.array([[c[3], -10.0]], np.float32)
        # the search restarts every edge with the episode and stuck counters at 0: so does this
        rw["tick"][0] = 0
        rw["stuck_ticks"][0] = 0
        for k in range(ticks):
            s = core.get_states()[0]
            pos.append(np.asarray(s["origin"], np.float64))
            vel.append(np.asarray(s["velocity"], np.float64))
            _o, _r, done, trunc, _t = core.step(act, view=view)
            t += 1
            if k == 0:
                act[0, 4] = 0
            if done[0] or trunc[0]:
                finished = bool(core.goal_hits[0])
                died = not finished
                break
        if finished or died:
            break
    pos, vel = np.asarray(pos), np.asarray(vel)
    spd = np.linalg.norm(vel, axis=1)
    print(f"replay: {'FINISHED' if finished else 'DIED' if died else 'ran out'} at {t / 100:.2f} s "
          f"(the search said {garr / 100:.2f} s); peak speed {spd.max():,.0f} u/s; "
          f"z {pos[:, 2].min():,.0f} .. {pos[:, 2].max():,.0f}")
    out = Path(a.archive).with_suffix("")
    np.savez_compressed(f"{out}.route.npz", pos=pos, vel=vel, controls=np.asarray(seq),
                        finished=finished, seconds=t / 100.0)
    print(f"wrote {out}.route.npz")

    ref = None
    if a.ref:
        w = np.load(a.ref)
        ref = (w["t"] - w["t"][0], w["simorg"], w["simvel"])
        print("  game s | found route: x y z, speed          | reference: x y z, speed")
        for s in range(0, int(max(t / 100.0, ref[0][-1])) + 1, 2):
            i = min(s * 100, len(pos) - 1)
            j = min(int(np.searchsorted(ref[0], s)), len(ref[0]) - 1)
            print(f"  {s:6d} | {pos[i, 0]:6.0f} {pos[i, 1]:6.0f} {pos[i, 2]:6.0f} {spd[i]:6.0f}"
                  f"   | {ref[1][j, 0]:6.0f} {ref[1][j, 1]:6.0f} {ref[1][j, 2]:6.0f} "
                  f"{np.linalg.norm(ref[2][j]):6.0f}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].plot(pos[:, 0], pos[:, 1], "-", color="#1f6fb4", lw=1.6, label="found by search")
        ax[1].plot(np.arange(len(pos)) / 100.0, pos[:, 2], "-", color="#1f6fb4", lw=1.6,
                   label="found by search")
        if ref is not None:
            ax[0].plot(ref[1][:, 0], ref[1][:, 1], "--", color="#d9771e", lw=1.4,
                       label="world record (reference only)")
            ax[1].plot(ref[0], ref[1][:, 2], "--", color="#d9771e", lw=1.4,
                       label="world record (reference only)")
        ax[0].plot(pos[0, 0], pos[0, 1], "o", color="k", ms=7)
        ax[0].annotate("start", (pos[0, 0], pos[0, 1]), xytext=(6, 6), textcoords="offset points")
        ax[0].set_title("top-down (x, y)")
        ax[0].set_aspect("equal")
        ax[1].set_title("height over time")
        ax[1].set_xlabel("game seconds")
        ax[1].set_ylabel("z (units)")
        for x in ax:
            x.legend(loc="best")
            x.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
