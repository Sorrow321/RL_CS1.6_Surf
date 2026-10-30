#!/usr/bin/env python3
"""ANALYSIS BENCH (CLAUDE.md section 0): closed-loop MPC (tools/mpc_bench.py's receding-horizon CEM)
with a FIELD-PROGRESS objective, started from the record's state at game time --bench-time with its
speed rescaled to a target energy height. Measures how much energy the pit's north-end exit into
the north room needs. The record's inputs are never read; nothing here trains anything."""
import argparse, sys, time
from pathlib import Path
import numpy as np
import os
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python")); sys.path.insert(0, str(ROOT / "tools"))
from surfgym import SurfCore, default_config
from surfgym.zones import load_zones
from surfgym.goalfield import GoalField
from kino_search import states_rw
G = 800.0
ap = argparse.ArgumentParser()
ap.add_argument("--map", default=str(ROOT / "maps/surf_unitfarmer2.bsp"))
ap.add_argument("--tick-ms", type=int, default=8)
ap.add_argument("--bench-time", type=float, default=9.5)
ap.add_argument("--energy", type=float, default=0.0, help="rescale speed to this energy height (0 = as is)")
ap.add_argument("--duration", type=float, default=4.0)
ap.add_argument("--plan", type=float, default=1.2)
ap.add_argument("--exec", type=int, default=8)
ap.add_argument("--knot", type=int, default=4)
ap.add_argument("--pop", type=int, default=256)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--max-offset", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
z = np.load(ROOT / "maps/surf_unitfarmer2.goalc_32.npz", allow_pickle=False)
F = GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"], float(z["cell"]), float(z["reach_max"]))
D0 = 30647.5
zones = load_zones(a.map)
core = SurfCore(a.map, default_config(num_envs=a.pop, lidar_w=0, lidar_h=0, view_mode=1, lookahead_k=0,
                sv_maxvelocity=3500.0, max_episode_ticks=10 ** 8, spawn_mode=1, yaw_jitter_deg=0.0),
                tick_ms=float(a.tick_ms))
core.set_teleport_fail(True)
core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
core.reset(0)
rw = states_rw(core); sv = core.states_view
cur = core.get_states()[0].copy()
w = np.load(os.path.join(os.environ.get("AR_HOME", "/home/steam/ar_uf2"), "wr", "surf_unitfarmer2_wr.frames.npz"))
wt = w["t"] - w["t"][0]
i = int(np.searchsorted(wt, a.bench_time))
o0, v0 = w["simorg"][i].astype(np.float64), w["simvel"][i].astype(np.float64)
E0 = o0[2] + (v0 ** 2).sum() / (2 * G)
if a.energy > 0:
    sp = np.sqrt(max(0.0, 2 * G * (a.energy - o0[2])))
    v0 = v0 / np.linalg.norm(v0) * sp
cur["origin"], cur["velocity"] = o0, v0
cur["yaw"], cur["onground"] = float(w["viewangles"][i, 1]), -1
cur["tick"], cur["stuck_ticks"] = 0, 0
cur["basevelocity"] = (0.0, 0.0, 0.0)
print(f"BENCH t {a.bench_time} at ({o0[0]:.0f},{o0[1]:.0f},{o0[2]:.0f}) record E {E0:.0f} -> start E "
      f"{o0[2] + (v0**2).sum()/(2*G):.0f} |v| {np.linalg.norm(v0):.0f}; tick {a.tick_ms} ms", flush=True)
TS = a.tick_ms / 1000.0
Hp = int(round(a.plan / TS)); K = Hp // a.knot + 1
n_rep = int(round(a.duration / TS)) // a.exec
shift = a.exec / a.knot
mu = np.zeros(K); sig = np.full(K, 8.0); pD = np.full(K, 0.5)
n_el = max(4, int(a.pop * 0.05))
act = np.zeros((a.pop, 6), np.int32); act[:, 0], act[:, 1], act[:, 2] = 7, 3, 1
view = np.zeros((a.pop, 2), np.float32); view[:, 1] = -10.0
best_plan = None; traj = []; end = "ran out"; t0 = time.perf_counter()
def dsample(o):
    return F.sample(o.astype(np.float32)).astype(np.float64)
for r in range(n_rep):
    for it in range(a.iters):
        off = np.clip(mu + sig * rng.standard_normal((a.pop, K)), -a.max_offset, a.max_offset)
        side = np.where(rng.random((a.pop, K)) < pD, 2, 0).astype(np.int32)
        if best_plan is not None:
            off[0], side[0] = best_plan
        rw[:] = cur
        alive = np.ones(a.pop, bool)
        dmin = np.full(a.pop, np.inf)
        for k in range(Hp):
            j, f = divmod(k, a.knot); j1 = min(j + 1, K - 1)
            view[:, 0] = off[:, j] + (off[:, j1] - off[:, j]) * (f / a.knot)
            act[:, 3] = side[:, j]
            _o, _rr, done, trunc, _t = core.step(act, view=view)
            alive &= ~((done | trunc).astype(bool))
            if k % 4 == 3:
                dd = dsample(np.asarray(sv["origin"]))
                dmin = np.where(alive, np.minimum(dmin, dd), dmin)
        oo = np.asarray(sv["origin"], np.float64); vv = np.asarray(sv["velocity"], np.float64)
        E = oo[:, 2] + (vv * vv).sum(1) / (2 * G)
        score = -dmin + 0.25 * E
        score[~alive] = -np.inf
        el = np.argsort(-score)[:n_el]; el = el[np.isfinite(score[el])]
        if len(el) == 0:
            continue
        best_plan = (off[el[0]].copy(), side[el[0]].copy())
        if len(el) >= 2:
            mu = 0.7 * off[el].mean(0) + 0.3 * mu
            sig = np.maximum(0.7 * off[el].std(0) + 0.3 * sig, 0.5)
            pD = np.clip(0.7 * (side[el] == 2).mean(0) + 0.3 * pD, 0.02, 0.98)
    if best_plan is None:
        end = "no surviving plan"; break
    off_b, side_b = best_plan
    rw[:] = cur
    died = False
    for k in range(a.exec):
        j, f = divmod(k, a.knot); j1 = min(j + 1, K - 1)
        view[:, 0] = off_b[j] + (off_b[j1] - off_b[j]) * (f / a.knot)
        act[:, 3] = side_b[j]
        _o, _rr, done, trunc, _t = core.step(act, view=view)
        s0 = sv[0]
        traj.append(np.concatenate([np.asarray(s0["origin"], np.float64), np.asarray(s0["velocity"], np.float64)]))
        if done[0] or trunc[0]:
            died = True; break
    if died:
        end = "DIED"; break
    cur = sv[0].copy(); cur["tick"], cur["stuck_ticks"] = 0, 0
    s = int(round(shift))
    mu = np.r_[mu[s:], np.repeat(mu[-1], s)]
    sig = np.r_[np.maximum(sig[s:], 2.0), np.full(s, 8.0)]
    pD = np.r_[pD[s:], np.full(s, 0.5)]
    best_plan = (np.r_[off_b[s:], np.repeat(off_b[-1], s)], np.r_[side_b[s:], np.repeat(side_b[-1], s)])
traj = np.asarray(traj)
if len(traj) == 0:
    print(f"MPC: {end} at once"); sys.exit()
d = dsample(traj[:, :3]); sp = np.linalg.norm(traj[:, 3:], axis=1); E = traj[:, 2] + sp ** 2 / (2 * G)
best = d.min()
print(f"MPC {end} after {len(traj)*TS:.2f} s: best d {best:.0f} = {100*(D0-best)/D0:.2f}% ; end ({traj[-1,0]:.0f},{traj[-1,1]:.0f},{traj[-1,2]:.0f}) E {E[-1]:.0f}; "
      f"room {'YES' if (traj[:,1] > -400).any() else 'no'}  [{time.perf_counter()-t0:.0f}s]")
for q in np.arange(0, len(traj) * TS, 0.5):
    k = min(int(q / TS), len(traj) - 1)
    print(f"   +{q:3.1f}s ({traj[k,0]:6.0f},{traj[k,1]:6.0f},{traj[k,2]:5.0f}) |v| {sp[k]:5.0f} E {E[k]:5.0f} prog {100*(D0-d[k])/D0:5.2f}%")
