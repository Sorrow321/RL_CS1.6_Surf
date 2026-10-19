#!/usr/bin/env python3
"""go_cem_route.py - the route a tools/go_cem.py archive found, replayed from the spawn.

Walks from a goal hit (default: the fastest) or a cell back through the expansions to the
spawn, then REPLAYS the whole control sequence from the spawn in a fresh engine and checks it
lands exactly where the archive says (the engine is deterministic, so it must).

    python tools/go_cem_route.py runs/research/kino/gocem_uf2.npz [--cell best-progress]
        [--ref runs/research/uf2_wr/surf_unitfarmer2.frames.npz] [--plot out.png]

--ref is a measuring stick only (CLAUDE.md section 0).
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


def chain(z, exp_id, ticks):
    """[(expansion, ticks)] from the spawn to (exp_id, ticks)"""
    segs = []
    while exp_id >= 0:
        segs.append((int(exp_id), int(ticks)))
        exp_id, ticks = int(z["exp_src_exp"][exp_id]), int(z["exp_src_tick"][exp_id])
    return segs[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--cell", default="goal",
                    help="'goal' (the fastest goal hit), 'best-progress', or a cell index")
    ap.add_argument("--map", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    z = np.load(a.archive)
    stem = str(z["map"])
    bsp = Path(a.map) if a.map else next(p for p in (ROOT / "maps_pool" / f"{stem}.bsp",
                                                     ROOT / "maps" / f"{stem}.bsp") if p.exists())
    zones = load_zones(str(bsp))
    tick_ms, knot = int(z["tick_ms"]), int(z["knot"])
    TS = tick_ms / 1000.0
    core = SurfCore(str(bsp), default_config(
        num_envs=1, lidar_w=0, lidar_h=0, view_mode=1, lookahead_k=0, sv_maxvelocity=4000.0,
        max_episode_ticks=10 ** 8, spawn_mode=1, yaw_jitter_deg=0.0), tick_ms=float(tick_ms))
    core.set_teleport_fail(True)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    core.reset(0)
    gf = None
    if a.cell == "best-progress":
        from surfgym.goalfield import build_goal_field
        gf = build_goal_field(core, zones["end"], cell=32)
    if a.cell == "goal":
        g = z["goals"]
        if not len(g):
            raise SystemExit("no goal hit in this archive (try --cell best-progress)")
        g = g[np.argmin(g[:, 3])]
        target, segs, want = "goal", chain(z, int(g[0]), int(g[2])), None
        print(f"target: the fastest goal hit, {g[3] * TS:.2f} s")
    else:
        if a.cell == "best-progress":
            d = gf.sample(z["origin"].astype(np.float64))
            d[d >= gf.reach_max] = np.inf
            c = int(np.argmin(d))
            print(f"target: the best-progress cell {c}, progress "
                  f"{1 - d[c] / float(z['d0']):.1%}, reached at {z['t'][c] * TS:.2f} s")
        else:
            c = int(a.cell)
        target, segs, want = f"cell {c}", chain(z, int(z["exp"][c]), int(z["tick"][c])), z["origin"][c]
    print(f"{len(segs)} segments, {sum(t for _, t in segs) * TS:.2f} s of game time")

    rw = states_rw(core)
    rw["velocity"][0] = (0.0, 0.0, 0.0)
    sv = core.states_view
    act = np.array([[7, 3, 1, 1, 0, 0]], np.int32)
    view = np.array([[0.0, -10.0]], np.float32)
    pos, vel = [], []
    end = "ran out"
    for e, ticks in segs:
        side, fwd, off = z["exp_side"][e], z["exp_fwd"][e], z["exp_off"][e]
        jmp = z["exp_jump"][e] if "exp_jump" in z else np.zeros(len(off), np.int8)
        K = len(off)
        rw["tick"][0] = 0
        rw["stuck_ticks"][0] = 0
        for k in range(ticks):
            j, f = divmod(k, knot)
            j1 = min(j + 1, K - 1)
            view[0, 0] = off[j] + (off[j1] - off[j]) * (f / knot)
            act[0, 2], act[0, 3], act[0, 4] = int(fwd[j]), int(side[j]), int(jmp[j])
            _o, _r, done, trunc, _t = core.step(act, view=view)
            pos.append(np.asarray(sv["origin"][0], np.float64))
            vel.append(np.asarray(sv["velocity"][0], np.float64))
            if done[0] or trunc[0]:
                end = "FINISHED" if core.goal_hits[0] else "DIED"
                break
        if end != "ran out":
            break
    pos, vel = np.asarray(pos), np.asarray(vel)
    spd = np.linalg.norm(vel, axis=1)
    print(f"replay: {end} after {len(pos) * TS:.2f} s; peak speed {spd.max():,.0f} u/s")
    if want is not None:
        print(f"  lands {np.linalg.norm(pos[-1] - want):.2f} u from the archived state")
    out = Path(a.archive).with_suffix("")
    np.savez_compressed(f"{out}.route.npz", pos=pos, vel=vel, segs=np.asarray(segs),
                        end=end, tick_ms=tick_ms)
    print(f"wrote {out}.route.npz")
    ref = None
    if a.ref:
        w = np.load(a.ref)
        ref = (w["t"] - w["t"][0], w["simorg"], w["simvel"])
        print("  game s | route x y z speed            | reference x y z speed")
        for s in range(0, int(len(pos) * TS) + 1, 2):
            i = min(int(s / TS), len(pos) - 1)
            j = min(int(np.searchsorted(ref[0], s)), len(ref[0]) - 1)
            print(f"  {s:6d} | {pos[i, 0]:6.0f} {pos[i, 1]:6.0f} {pos[i, 2]:6.0f} {spd[i]:6.0f}   | "
                  f"{ref[1][j, 0]:6.0f} {ref[1][j, 1]:6.0f} {ref[1][j, 2]:6.0f} "
                  f"{np.linalg.norm(ref[2][j]):6.0f}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        tt = np.arange(len(pos)) * TS
        ax[0].plot(pos[:, 0], pos[:, 1], "-", color="#1f6fb4", lw=1.6, label=f"found by search ({end.lower()})")
        ax[1].plot(tt, pos[:, 2], "-", color="#1f6fb4", lw=1.6, label="found by search")
        if ref is not None:
            ax[0].plot(ref[1][:, 0], ref[1][:, 1], "--", color="#d9771e", lw=1.3,
                       label="world record (reference only)")
            ax[1].plot(ref[0], ref[1][:, 2], "--", color="#d9771e", lw=1.3,
                       label="world record (reference only)")
        ax[0].plot(pos[0, 0], pos[0, 1], "o", color="k", ms=7)
        ax[0].annotate("spawn", (pos[0, 0], pos[0, 1]), xytext=(6, 6), textcoords="offset points")
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
