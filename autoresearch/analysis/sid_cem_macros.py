#!/usr/bin/env python3
"""cem_macro.py - planning in the simulator, rooted at the POLICY'S OWN state (no demo input).

CEM over K held macros (strafe key in {L, none, R}, forward key in {none, F}, view offset from the
velocity heading in degrees, duration in seconds), executed tick by tick with SurfCore.play_step
(usercmd physics + live triggers), then a free-fall tail. Objective: the goal field's minimum d
reached (old field, the grader's), a finish = d 0. Start: a state of a greedy eval episode of
the incumbent, ~T s after it first enters the region x > X0, z < Z0 (the basement - a measurement
box picked from the eval, only to choose WHICH own state to root the search at).

    python3 cem_macro.py <traj.jsonl> <ep> <t_after_entry> <out.npz> [iters] [pop] [K] [seed]
"""
import json, sys, math, time
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


SIDE = np.array([-250.0, 0.0, 250.0])
FWD = np.array([0.0, 250.0])


def rollout(s0, macros, tail=1.5, keep=False):
    st = SurfState()
    st.origin[:] = [float(x) for x in s0[0]]; st.velocity[:] = [float(x) for x in s0[1]]
    st.onground = -1; st.ducked = int(s0[2])
    dmin, fin, P = 1e9, False, []
    plan = []
    for side, fwd, off, dur in macros:
        plan += [(side, fwd, off)] * max(1, int(round(dur * 100)))
    plan += [(1, 0, 0.0)] * int(round(tail * 100))
    for k, (side, fwd, off) in enumerate(plan):
        v = st.velocity
        hd = math.degrees(math.atan2(v[1], v[0]))
        prev = (st.origin[0], st.origin[1], st.origin[2])
        f = core.play_step(st, hd + off, -10.0, float(FWD[fwd]), float(SIDE[side]), 0, 10)
        if f & 3:
            break
        o = (st.origin[0], st.origin[1], st.origin[2])
        if keep:
            P.append((o[0], o[1], o[2], st.velocity[0], st.velocity[1], st.velocity[2]))
        if (k % 4) == 0 or o[2] < 400:
            d = float(GF.sample(np.array([o], np.float32))[0])
            if d < GF.reach_max - 16:
                dmin = min(dmin, d)
        if (FIN[0][0] <= o[0] <= FIN[1][0] and FIN[0][1] <= o[1] <= FIN[1][1]
                and o[2] <= FIN[1][2] < prev[2]):
            fin = True; dmin = 0.0
            break
    return dmin, fin, (np.array(P) if keep else None)


def main():
    path, ep, t_after, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    pop = int(sys.argv[6]) if len(sys.argv) > 6 else 384
    K = int(sys.argv[7]) if len(sys.argv) > 7 else 5
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    rng = np.random.default_rng(seed)
    a, _ = episodes(path)[ep]
    p = a[:, 1:4]
    ib = np.flatnonzero((p[:, 0] > 2280) & (p[:, 2] < 1250))
    i0 = int(ib[0] + round(t_after * 100))
    s0 = (a[i0, 1:4].copy(), a[i0, 4:7].copy(), 1 if (int(a[i0, 8]) & 4) else 0)
    print(f"root: ep {ep} t {a[i0,0]/100:.2f} pos {s0[0].round(0)} vel {s0[1].round(0)} duck {s0[2]}", flush=True)
    # distributions
    ps = np.full((K, 3), 1 / 3); pf = np.full((K, 2), 0.5)
    mo = np.zeros(K); so = np.full(K, 4.0)
    md = np.full(K, 0.6); sd = np.full(K, 0.4)
    best = (1e9, None, False)
    t0 = time.time()
    for it in range(iters):
        sides = np.stack([rng.choice(3, pop, p=ps[k]) for k in range(K)], 1)
        fwds = np.stack([rng.choice(2, pop, p=pf[k]) for k in range(K)], 1)
        offs = rng.normal(mo, so, (pop, K)).clip(-12, 12)
        durs = rng.normal(md, sd, (pop, K)).clip(0.08, 2.0)
        J = np.empty(pop)
        fins = 0
        for i in range(pop):
            mac = [(int(sides[i, k]), int(fwds[i, k]), float(offs[i, k]), float(durs[i, k])) for k in range(K)]
            dmin, fin, _ = rollout(s0, mac)
            J[i] = dmin; fins += fin
            if dmin < best[0]:
                best = (dmin, mac, fin)
        el = np.argsort(J)[: max(8, pop // 16)]
        for k in range(K):
            ps[k] = 0.7 * ps[k] + 0.3 * (np.bincount(sides[el, k], minlength=3) / len(el))
            pf[k] = 0.7 * pf[k] + 0.3 * (np.bincount(fwds[el, k], minlength=2) / len(el))
        mo = 0.7 * mo + 0.3 * offs[el].mean(0); so = np.maximum(0.3, 0.7 * so + 0.3 * offs[el].std(0))
        md = 0.7 * md + 0.3 * durs[el].mean(0); sd = np.maximum(0.05, 0.7 * sd + 0.3 * durs[el].std(0))
        print(f"it {it:2d} best prog {100*(D0-best[0])/D0:6.2f} (fin {best[2]})  elite med {100*(D0-np.median(J[el]))/D0:6.2f}  "
              f"pop fins {fins}  {time.time()-t0:.0f}s", flush=True)
    dmin, fin, P = rollout(s0, best[1], keep=True)
    np.savez(out, macros=np.array(best[1], dtype=float), traj=P, root=np.concatenate([s0[0], s0[1]]),
             dmin=dmin, fin=fin)
    print("BEST", 100 * (D0 - dmin) / D0, "fin", fin, "macros", [tuple(round(x, 2) for x in m) for m in best[1]])


if __name__ == "__main__":
    main()
