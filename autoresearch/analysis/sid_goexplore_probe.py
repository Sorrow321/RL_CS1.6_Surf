#!/usr/bin/env python3
"""sid_goexplore_probe.py - ANALYSIS ONLY (CLAUDE.md sections 0 / 0b): is there ANY way on past
the CP2 slot from the policy's own line, and can archive-and-return exploration find it?

Go-Explore (Ecoffet et al. 2021) in the simulator, from the policy's OWN greedy states of a
recorded eval (traj_*.jsonl, analysis only): an archive of cells (position 128 u + a coarse
velocity heading/speed bucket), each holding the deepest-progress state that reached it; pick a
cell with weight 1/sqrt(1 + times chosen), restore its state, take random held usercmd
primitives through the live-trigger physics (SurfCore.play_step) for up to `explore_s`, add
every new cell. Reports the frontier (best goal-field progress) over iterations and where the
deepest cells are. Nothing here enters training.

    python sid_goexplore_probe.py <traj.jsonl> <episode> <t0,t1,...> [iters] [explore_s] [seed]
"""
import ctypes
import json
import sys
import time

import numpy as np

sys.path.insert(0, "python")
from surfgym.core import SURF_IN_DUCK, SurfCore, SurfState  # noqa: E402
from surfgym.goalfield import GoalField  # noqa: E402

BSP = "maps/surf_src_sidistic.bsp"
z = np.load("maps/surf_src_sidistic.goal_32.npz", allow_pickle=False)
GF = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
               float(z["cell"]), float(z["reach_max"]))
D0 = 97768.0
LIM = float(z["reach_max"]) - 16.0


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


def clone(a):
    b = SurfState()
    ctypes.memmove(ctypes.addressof(b), ctypes.addressof(a), ctypes.sizeof(SurfState))
    return b


def cell_of(s):
    vx, vy, vz = s.velocity[0], s.velocity[1], s.velocity[2]
    h = int(((np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0) // 90.0)   # 4 headings
    sp = min(2, int(np.hypot(vx, vy) // 800.0))                            # 3 speed buckets
    up = 1 if vz > 0 else 0
    return (int(s.origin[0] // 128), int(s.origin[1] // 128), int(s.origin[2] // 128), h, sp, up)


def prog(p):
    d = float(GF.sample(np.array([p]))[0])
    return 100.0 * (D0 - d) / D0 if d < LIM else -1.0


def main():
    path, ep = sys.argv[1], int(sys.argv[2])
    t_starts = [float(v) for v in sys.argv[3].split(",")]
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    explore_s = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    rng = np.random.default_rng(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    core = SurfCore(BSP)
    _, a, _ = episodes(path)[ep]
    arch = {}   # cell -> [state, progress, chosen, t_since_start(s)]
    keys = []   # selection order; chosen counts live in `chosen` (numpy, grows by doubling)
    chosen = np.zeros(1024, np.float64)

    def add_key(c):
        nonlocal chosen
        if len(keys) >= len(chosen):
            chosen = np.concatenate([chosen, np.zeros(len(chosen))])
        keys.append(c)

    for ts in t_starts:
        row = a[int(round(ts * 100))]
        s = make_state(row)
        c0 = cell_of(s)
        if c0 not in arch:
            add_key(c0)
        arch[c0] = [s, prog(tuple(row[1:4])), 0, 0.0]
    best = max(v[1] for v in arch.values())
    t0 = time.time()
    ticks = 0
    for it in range(iters):
        nk = len(keys)
        w = 1.0 / np.sqrt(1.0 + chosen[:nk])
        cw = np.cumsum(w)
        j = int(np.searchsorted(cw, rng.random() * cw[-1], side="right"))
        j = min(j, nk - 1)
        k = keys[j]
        chosen[j] += 1.0
        s = clone(arch[k][0])
        age = arch[k][3]
        n = int(explore_s * 100)
        tick = 0
        while tick < n:
            hold = int(rng.integers(1, 9)) * 4
            off = float(rng.choice([0, 0, 10, -10, 30, -30, 60, -60, 90, -90, 135, -135, 180]))
            fm = float(rng.choice([-250.0, 0.0, 0.0, 250.0]))
            sm = float(rng.choice([-250.0, 0.0, 250.0]))
            bt = SURF_IN_DUCK if rng.random() < 0.4 else 0
            dead = False
            for _ in range(hold):
                vx, vy = s.velocity[0], s.velocity[1]
                head = np.degrees(np.arctan2(vy, vx)) if (vx * vx + vy * vy) > 1.0 else s.yaw
                fl = core.play_step(s, head + off, 0.0, fm, sm, bt, 10)
                tick += 1
                ticks += 1
                if fl & (SurfCore.PLAY_KILLED | SurfCore.PLAY_TELEPORTED):
                    dead = True
                    break
                if tick % 4 == 0:
                    c = cell_of(s)
                    pr = prog((s.origin[0], s.origin[1], s.origin[2]))
                    cur = arch.get(c)
                    if cur is None or pr > cur[1] + 1e-6:
                        if cur is None:
                            add_key(c)
                        arch[c] = [clone(s), pr, 0, age + tick / 100.0]
                        if pr > best:
                            best = pr
                if tick >= n:
                    break
            if dead:
                break
        if (it + 1) % 2000 == 0:
            # climbed back through the centre gap into the shaft: x < 2,520, z 10,850-11,200, vz > 0
            gap_up = sum(1 for kk, v in arch.items()
                         if v[0].origin[0] < 2520 and 10850 < v[0].origin[2] < 11200
                         and v[0].velocity[2] > 0 and abs(v[0].origin[1]) < 1000)
            print(f"           climbing in the shaft (x<2520, z 10850-11200, vz>0): {gap_up}", flush=True)
            low = sum(1 for kk, v in arch.items() if v[0].origin[2] < 10600)
            print(f"iter {it + 1:6d}  cells {len(arch):6d}  best {best:6.2f}%  cells below z 10600: "
                  f"{low}  ticks {ticks:,}  {time.time() - t0:.0f}s", flush=True)
    top = sorted(arch.items(), key=lambda kv: -kv[1][1])[:10]
    for kk, v in top:
        print("  top", json.dumps(dict(prog=round(v[1], 2), pos=[round(v[0].origin[i]) for i in range(3)],
                                       vel=[round(v[0].velocity[i]) for i in range(3)], t=round(v[3], 2))))


if __name__ == "__main__":
    main()
