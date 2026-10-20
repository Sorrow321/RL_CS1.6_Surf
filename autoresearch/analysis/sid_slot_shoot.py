#!/usr/bin/env python3
"""sid_slot_shoot.py - ANALYSIS ONLY (CLAUDE.md sections 0 / 0b): CAN'T vs WON'T past the CP2 slot.

From the policy's OWN greedy states in the slot (a recorded eval, traj_*.jsonl), random-shoot
held usercmd primitives through the live-trigger physics (SurfCore.play_step) and report what
any continuation can reach: the best goal-field progress, whether a rollout gets through the
wall's windows (x < 1,940 inside the tunnel's height), into the region below the slot
(z < 10,600), and where the deepest survivors are. Nothing here enters training.

    python sid_slot_shoot.py <traj.jsonl> <episode> <t_start_s,...> [n_per_start] [secs] [seed]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "python")
from surfgym.core import SurfCore, SurfState, SURF_IN_DUCK  # noqa: E402
from surfgym.goalfield import GoalField  # noqa: E402

BSP = "maps/surf_src_sidistic.bsp"
z = np.load("maps/surf_src_sidistic.goal_32.npz", allow_pickle=False)
GF = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
               float(z["cell"]), float(z["reach_max"]))
D0 = 97768.0


def episodes(path):
    eps, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append((hdr, np.asarray(cur, np.float64), d))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def make_state(row):
    s = SurfState()
    for i in range(3):
        s.origin[i] = row[1 + i]
        s.velocity[i] = row[4 + i]
    s.yaw = row[7]
    s.pitch = row[12]
    s.onground = -1 if row[9] < 0.5 else 0
    s.ducked = 1 if (int(row[8]) & SURF_IN_DUCK) else 0
    return s


def copy_state(a):
    b = SurfState()
    ctypes_memmove(b, a)
    return b


def ctypes_memmove(dst, src):
    import ctypes
    ctypes.memmove(ctypes.addressof(dst), ctypes.addressof(src), ctypes.sizeof(SurfState))


def main():
    path, ep = sys.argv[1], int(sys.argv[2])
    t_starts = [float(v) for v in sys.argv[3].split(",")]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    secs = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    rng = np.random.default_rng(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    core = SurfCore(BSP)
    _, a, _ = episodes(path)[ep]
    max_ticks = int(secs * 100)
    for ts in t_starts:
        row = a[int(round(ts * 100))]
        s0 = make_state(row)
        d_start = float(GF.sample(np.array([row[1:4]]))[0])
        res = []
        t0 = time.time()
        for k in range(n):
            s = SurfState()
            ctypes_memmove(s, s0)
            best_d, best_p = d_start, tuple(row[1:4])
            window = False
            low = False
            dead = False
            tick = 0
            path_pts = []
            while tick < max_ticks:
                # one held primitive: yaw offset from the velocity heading, keys, duck
                hold = int(rng.integers(1, 9)) * 4
                off = float(rng.choice([0, 0, 0, 10, -10, 30, -30, 60, -60, 90, -90, 135, -135, 180]))
                fm = float(rng.choice([-250.0, 0.0, 0.0, 250.0]))
                sm = float(rng.choice([-250.0, 0.0, 250.0]))
                bt = SURF_IN_DUCK if rng.random() < 0.4 else 0
                for _ in range(hold):
                    vx, vy = s.velocity[0], s.velocity[1]
                    head = np.degrees(np.arctan2(vy, vx)) if (vx * vx + vy * vy) > 1.0 else s.yaw
                    fl = core.play_step(s, head + off, 0.0, fm, sm, bt, 10)
                    tick += 1
                    if fl & (SurfCore.PLAY_KILLED | SurfCore.PLAY_TELEPORTED):
                        dead = True
                        break
                    p = (s.origin[0], s.origin[1], s.origin[2])
                    if tick % 4 == 0:
                        d = float(GF.sample(np.array([p]))[0])
                        if d < best_d and d < GF.reach_max - 16:
                            best_d, best_p = d, p
                        path_pts.append(p)
                    if p[0] < 1940 and 10950 < p[2] < 11200:
                        window = True
                    if p[2] < 10600:
                        low = True
                    if tick >= max_ticks:
                        break
                if dead:
                    break
            res.append(dict(prog=100 * (D0 - best_d) / D0, best=[round(v) for v in best_p],
                            window=window, low=low, dead=dead, ticks=tick,
                            last=[round(s.origin[i]) for i in range(3)]))
        progs = np.array([r["prog"] for r in res])
        order = np.argsort(-progs)
        print(f"== start t={ts:.2f}s pos={[round(v) for v in row[1:4]]} v={[round(v) for v in row[4:7]]} "
              f"start_prog={100 * (D0 - d_start) / D0:.2f}%  n={n}  {time.time() - t0:.0f}s")
        print(f"   window {sum(r['window'] for r in res)}  low(<10600) {sum(r['low'] for r in res)}  "
              f"survived {sum(not r['dead'] for r in res)}  prog p50 {np.median(progs):.2f} "
              f"p99 {np.percentile(progs, 99):.2f} max {progs.max():.2f}")
        for i in order[:6]:
            print("   top", json.dumps(res[i]))


if __name__ == "__main__":
    main()
