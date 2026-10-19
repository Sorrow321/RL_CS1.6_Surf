#!/usr/bin/env python3
"""mpc_bench.py - CLOSED-LOOP optimal control in the real engine: receding-horizon CEM (MPC).

cem_bench.py optimises ONE open-loop sequence; a long precise segment (unitfarmer2's U-turn
through the pit bottom) defeats it: at 28 ms knots it keeps 869 u of energy height where the
record keeps 1,118, and at 14 / 7 ms knots no random candidate even survives round 0. MPC plans
--plan seconds ahead from the CURRENT state, executes the first --exec ticks of the best plan,
and replans from where it landed, warm-starting from the previous plan shifted forward - so the
controls are corrected every few ticks, the way the record's are.

Controls per knot (--knot ticks): side key A or D (Bernoulli) and the view yaw offset from the
live velocity heading (Gaussian, linearly interpolated), as in cem_bench.py. Objectives, all
generic: energy (z + |v|^2/2g at the plan's end, alive) or height.

    python tools/mpc_bench.py --map maps_pool/surf_unitfarmer2.bsp \\
        --bench-demo runs/research/uf2_wr/surf_unitfarmer2.frames.npz --bench-time 3.5 \\
        --duration 2.5 --plan 1.2 --exec 4

--bench-demo starts from a parsed record's state: an ANALYSIS BENCH (CLAUDE.md section 0) -
the record never enters the objective and its inputs are never read; --ref-report prints the
record's own energy at the same game times as a measuring stick.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym import SurfCore, default_config          # noqa: E402
from surfgym.zones import load_zones                  # noqa: E402
from kino_search import states_rw                     # noqa: E402

G = 800.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--tick-ms", type=int, default=7)
    ap.add_argument("--bench-demo", default=None)
    ap.add_argument("--bench-time", type=float, default=0.0)
    ap.add_argument("--start-npz", default=None, help="instead: an archive and --start-node")
    ap.add_argument("--start-node", type=int, default=-1)
    ap.add_argument("--duration", type=float, default=2.5, help="seconds of closed-loop flight")
    ap.add_argument("--plan", type=float, default=1.2, help="planning horizon, seconds")
    ap.add_argument("--exec", type=int, default=4, help="ticks executed per replan")
    ap.add_argument("--knot", type=int, default=4)
    ap.add_argument("--pop", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--elite", type=float, default=0.05)
    ap.add_argument("--max-offset", type=float, default=8.0)
    ap.add_argument("--objective", default="energy", choices=("energy", "height"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    bsp = Path(a.map)
    zones = load_zones(str(bsp))
    core = SurfCore(str(bsp), default_config(
        num_envs=a.pop, lidar_w=0, lidar_h=0, view_mode=1, lookahead_k=0,
        sv_maxvelocity=4000.0, max_episode_ticks=10 ** 8, spawn_mode=1, yaw_jitter_deg=0.0),
        tick_ms=float(a.tick_ms))
    core.set_teleport_fail(True)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    core.reset(0)
    rw = states_rw(core)
    sv = core.states_view
    cur = core.get_states()[0].copy()
    ref = None
    if a.bench_demo:
        w = np.load(a.bench_demo)
        wt = w["t"] - w["t"][0]
        i = int(np.searchsorted(wt, a.bench_time))
        cur["origin"], cur["velocity"] = w["simorg"][i], w["simvel"][i]
        cur["yaw"], cur["onground"] = float(w["viewangles"][i, 1]), -1
        ref = (wt, w["simorg"], w["simvel"])
        print(f"!! ANALYSIS BENCH: start = {Path(a.bench_demo).name} at {a.bench_time:g} s; the "
              f"record's inputs are never read", flush=True)
    elif a.start_npz:
        st = np.load(Path(a.start_npz).with_suffix(".states.npy"))
        cur = st[a.start_node].copy()
    cur["tick"], cur["stuck_ticks"] = 0, 0
    cur["basevelocity"] = (0.0, 0.0, 0.0)

    TS = a.tick_ms / 1000.0
    Hp = int(round(a.plan / TS))
    K = Hp // a.knot + 1
    n_rep = int(round(a.duration / TS)) // a.exec
    shift = a.exec / a.knot                                # knots consumed per replan
    mu = np.zeros(K)
    sig = np.full(K, 3.0)
    pD = np.full(K, 0.5)
    n_el = max(4, int(a.pop * a.elite))
    act = np.zeros((a.pop, 6), np.int32)
    act[:, 0], act[:, 1], act[:, 2] = 7, 3, 1
    view = np.zeros((a.pop, 2), np.float32)
    view[:, 1] = -10.0
    best_plan = None
    traj = []
    t0 = time.perf_counter()
    end = "ran out"

    def energy(o, v):
        return o[..., 2] + (v * v).sum(-1) / (2 * G)

    for r in range(n_rep):
        for it in range(a.iters):
            off = np.clip(mu + sig * rng.standard_normal((a.pop, K)), -a.max_offset, a.max_offset)
            side = np.where(rng.random((a.pop, K)) < pD, 2, 0).astype(np.int32)
            if best_plan is not None:
                off[0], side[0] = best_plan
            rw[:] = cur
            alive = np.ones(a.pop, bool)
            for k in range(Hp):
                j, f = divmod(k, a.knot)
                j1 = min(j + 1, K - 1)
                view[:, 0] = off[:, j] + (off[:, j1] - off[:, j]) * (f / a.knot)
                act[:, 3] = side[:, j]
                _o, _rr, done, trunc, _t = core.step(act, view=view)
                alive &= ~((done | trunc).astype(bool))
            o = np.asarray(sv["origin"], np.float64)
            v = np.asarray(sv["velocity"], np.float64)
            score = energy(o, v) if a.objective == "energy" else o[:, 2].copy()
            score[~alive] = -np.inf
            el = np.argsort(-score)[:n_el]
            el = el[np.isfinite(score[el])]
            if len(el) == 0:
                continue
            best_plan = (off[el[0]].copy(), side[el[0]].copy())
            if len(el) >= 2:
                mu = 0.7 * off[el].mean(0) + 0.3 * mu
                sig = np.maximum(0.7 * off[el].std(0) + 0.3 * sig, 0.15)
                pD = np.clip(0.7 * (side[el] == 2).mean(0) + 0.3 * pD, 0.02, 0.98)
        if best_plan is None:
            end = "no surviving plan"
            break
        # execute the first --exec ticks of the best plan from the current state
        off_b, side_b = best_plan
        rw[:] = cur
        died = False
        for k in range(a.exec):
            j, f = divmod(k, a.knot)
            j1 = min(j + 1, K - 1)
            view[:, 0] = off_b[j] + (off_b[j1] - off_b[j]) * (f / a.knot)
            act[:, 3] = side_b[j]
            _o, _rr, done, trunc, _t = core.step(act, view=view)
            s0 = sv[0]
            traj.append(np.concatenate([np.asarray(s0["origin"], np.float64),
                                        np.asarray(s0["velocity"], np.float64)]))
            if done[0] or trunc[0]:
                died = True
                break
        if died:
            end = "DIED"
            break
        cur = sv[0].copy()
        cur["tick"], cur["stuck_ticks"] = 0, 0
        # warm start: shift the distribution and the incumbent forward by the executed knots
        s = int(round(shift))
        mu = np.r_[mu[s:], np.repeat(mu[-1], s)]
        sig = np.r_[np.maximum(sig[s:], 1.0), np.full(s, 3.0)]
        pD = np.r_[pD[s:], np.full(s, 0.5)]
        best_plan = (np.r_[off_b[s:], np.repeat(off_b[-1], s)],
                     np.r_[side_b[s:], np.repeat(side_b[-1], s)])
        if r % 10 == 0:
            o_, v_ = traj[-1][:3], traj[-1][3:]
            print(f"  t +{len(traj) * TS:5.2f} s: z {o_[2]:6.0f}  speed {np.linalg.norm(v_):5.0f}  "
                  f"energy {energy(o_, v_):6.0f}  [{time.perf_counter() - t0:5.1f}s]", flush=True)
    traj = np.asarray(traj)
    if len(traj) == 0:
        print(f"MPC: {end} from the start state (every candidate of the first plan died within "
              f"{a.plan:g} s) [{time.perf_counter() - t0:.1f}s compute]")
        return
    sp = np.linalg.norm(traj[:, 3:], axis=1)
    E = traj[:, 2] + sp ** 2 / (2 * G)
    print(f"MPC: {end} after {len(traj) * TS:.2f} s; end z {traj[-1, 2]:.0f}, speed {sp[-1]:.0f}, "
          f"energy {E[-1]:.0f}; min energy {E.min():.0f}; [{time.perf_counter() - t0:.1f}s compute]")
    if ref is not None:
        wt, wo, wv = ref
        print("  game s | MPC z  energy | record z  energy (reference only)")
        for q in np.arange(0.0, len(traj) * TS + 1e-9, 0.5):
            i = min(int(q / TS), len(traj) - 1)
            jj = min(int(np.searchsorted(wt, a.bench_time + q)), len(wt) - 1)
            print(f"  {a.bench_time + q:6.2f} | {traj[i, 2]:6.0f} {E[i]:6.0f} | "
                  f"{wo[jj, 2]:6.0f} {wo[jj, 2] + (wv[jj] ** 2).sum() / (2 * G):6.0f}")
    if a.out:
        np.savez_compressed(a.out, traj=traj, tick_ms=a.tick_ms)


if __name__ == "__main__":
    main()
