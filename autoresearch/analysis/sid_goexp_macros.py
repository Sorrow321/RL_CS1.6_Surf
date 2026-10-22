#!/usr/bin/env python3
"""goexp.py - Go-Explore in the simulator rooted at the POLICY'S OWN greedy-eval states (no demo input).

Archive cells = (x/128, y/128, z/64, |v_h|/150, heading/30deg). Each iteration: pick cells with
probability ~ 1/sqrt(1 + times chosen), restore the cell's stored state, run 1-3 random held macros
(strafe key L/none/R, forward key, view offset ~ N(0, 3 deg) from the velocity heading, 0.1-0.8 s),
add every new cell met (every 4 ticks) with its exact state and a parent pointer. Stops at a finish.
Objective-free (novelty); the goal field is only logged. Kill / teleport ends a rollout.

    python3 goexp.py <traj.jsonl> <eps,..> <t_after,..> <out.npz> [minutes] [batch] [seed]
"""
import json, sys, math, time, pickle
import numpy as np
sys.path.insert(0, "/home/steam/ar/wt/b4/python")
from surfgym.core import SurfCore, default_config, SurfState
from surfgym.goalfield import GoalField

R = "/home/steam/ar/wt/b4/"
core = SurfCore(R + "maps/surf_src_sidistic.bsp", default_config(num_envs=1, lidar_w=0, lidar_h=0))
zf = np.load(R + "maps/surf_src_sidistic.goal_32.npz")
GF = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"], float(zf["cell"]), float(zf["reach_max"]))
D0 = 97768.0
FIN = ((2872.0, -88.0, 176.0), (3048.0, 88.0, 184.017))
SIDE = (-250.0, 0.0, 250.0)
FWD = (0.0, 250.0)


def episodes(path):
    eps, hdr, cur = [], None, []
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None: eps.append((np.asarray(cur, float), d)); hdr, cur = None, []
            else: hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def cell(o, v):
    vh = math.hypot(v[0], v[1])
    hd = math.degrees(math.atan2(v[1], v[0])) % 360.0
    return (int(o[0] // 128), int(o[1] // 128), int(o[2] // 64), int(vh // 150), int(hd // 30))


def mkstate(o, v, duck):
    st = SurfState()
    st.origin[:] = [float(x) for x in o]; st.velocity[:] = [float(x) for x in v]
    st.onground = -1; st.ducked = int(duck)
    return st


def snap(st):
    return (tuple(st.origin), tuple(st.velocity), int(st.onground), int(st.ducked), int(st.induck),
            float(st.duck_time), int(st.oldbuttons))


def restore(s):
    st = SurfState()
    st.origin[:] = s[0]; st.velocity[:] = s[1]; st.onground = s[2]; st.ducked = s[3]
    st.induck = s[4]; st.duck_time = s[5]; st.oldbuttons = s[6]
    return st


def explore(s, macros, on_tick):
    """Run macros from snapshot s; on_tick(k, st) after every tick (k = tick index, 1-based).
    -> 'fin' | 'dead' | 'ok'."""
    st = restore(s)
    k = 0
    for side, fwd, off, n in macros:
        for _ in range(n):
            v = st.velocity
            hd = math.degrees(math.atan2(v[1], v[0]))
            prev_z = st.origin[2]
            f = core.play_step(st, hd + off, -10.0, FWD[fwd], SIDE[side], 0, 10)
            k += 1
            if f & 3:
                return "dead", k
            o = st.origin
            if (FIN[0][0] <= o[0] <= FIN[1][0] and FIN[0][1] <= o[1] <= FIN[1][1]
                    and o[2] <= FIN[1][2] < prev_z):
                return "fin", k
            on_tick(k, st)
    return "ok", k


def main():
    path = sys.argv[1]
    eps = [int(x) for x in sys.argv[2].split(",")]
    tafter = [float(x) for x in sys.argv[3].split(",")]
    out = sys.argv[4]
    minutes = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
    batch = int(sys.argv[6]) if len(sys.argv) > 6 else 256
    rng = np.random.default_rng(int(sys.argv[7]) if len(sys.argv) > 7 else 0)
    E = episodes(path)
    arch = {}          # cell -> dict(s=snapshot, n=times chosen, parent=cell|None, macros=..., d=field d)
    roots = []
    for e in eps:
        a = E[e][0]
        p = a[:, 1:4]
        ib = np.flatnonzero((p[:, 0] > 2280) & (p[:, 2] < 1250))
        if not len(ib): continue
        for ta in tafter:
            i0 = int(ib[0] + round(ta * 100))
            if i0 >= len(a): continue
            st = mkstate(a[i0, 1:4], a[i0, 4:7], 1 if (int(a[i0, 8]) & 4) else 0)
            c = cell(st.origin, st.velocity)
            d = float(GF.sample(np.array([tuple(st.origin)], np.float32))[0])
            arch[c] = dict(s=snap(st), n=0, parent=None, macros=None, d=d, root=(e, round(ta, 2)))
            roots.append(c)
    print(f"{len(roots)} roots, cells {len(arch)}", flush=True)
    t0 = time.time()
    best_d, it, found = 1e9, 0, None
    while time.time() - t0 < minutes * 60 and found is None:
        keys = list(arch.keys())
        w = np.array([1.0 / math.sqrt(1.0 + arch[c]["n"]) for c in keys])
        pick = rng.choice(len(keys), size=batch, p=w / w.sum())
        new = 0
        for j in pick:
            c0 = keys[j]
            arch[c0]["n"] += 1
            nm = int(rng.integers(1, 4))
            macros = []
            for _ in range(nm):
                macros.append((int(rng.choice(3, p=[0.4, 0.2, 0.4])), int(rng.choice(2, p=[0.7, 0.3])),
                               float(np.clip(rng.normal(0.0, 3.0), -12, 12)),
                               int(rng.integers(10, 81))))
            seen = []

            def on_tick(k, st):
                if k % 4 == 0:
                    seen.append((k, snap(st)))
            res, k = explore(arch[c0]["s"], macros, on_tick)
            if res == "fin":
                found = (c0, macros, k)
                break
            for kk, s in seen:
                c = cell(s[0], s[1])
                if c not in arch:
                    d = float(GF.sample(np.array([s[0]], np.float32))[0])
                    # the macros that lead from the parent's state to this one: the prefix of k ticks
                    pre, left = [], kk
                    for m in macros:
                        if left <= 0: break
                        take = min(m[3], left)
                        pre.append((m[0], m[1], m[2], take)); left -= take
                    arch[c] = dict(s=s, n=0, parent=c0, macros=pre, d=d)
                    new += 1
                    if d < best_d and d < GF.reach_max - 16:
                        best_d = d
        it += 1
        if it % 5 == 0 or found:
            zmax_room = max((arch[c]["s"][0][2] for c in arch
                             if 2460 < arch[c]["s"][0][0] < 3420 and abs(arch[c]["s"][0][1]) < 700), default=0)
            fastest = max(math.hypot(arch[c]["s"][1][0], arch[c]["s"][1][1]) for c in arch)
            print(f"it {it:4d} cells {len(arch):6d} new {new:4d} best prog {100*(D0-best_d)/D0:6.2f} "
                  f"room zmax {zmax_room:5.0f} fastest |vh| {fastest:5.0f}  {time.time()-t0:.0f}s", flush=True)
    with open(out, "wb") as fh:
        pickle.dump(dict(arch=arch, found=found), fh)
    print("FOUND" if found else "no finish", found[0] if found else "", flush=True)


if __name__ == "__main__":
    main()
