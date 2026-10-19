#!/usr/bin/env python3
"""cem_bench.py - local optimal control in the real engine: the cross-entropy method over a
control SEQUENCE from one start state (the user, 2026-10-16: "it's just optimal control").

A candidate is an open-loop sequence over --horizon seconds: the side key (A or D) and the view
yaw OFFSET from the live velocity heading (the trainer's --view-absolute velocity mode), both
set at knots every --knot ticks, the offset linearly interpolated between knots. Every round
simulates --pop candidates in parallel from the same start state (the engine is
deterministic), keeps the --elite best by the objective, and refits the sampling distribution
(per-knot Gaussian offsets, per-knot Bernoulli side keys) to them.

Objectives (generic: nothing in them is read off a map):
    height   the z at the end of the horizon, alive
    energy   z + |v|^2 / 2g at the end, alive
    field    geodesic progress at the end (the map's goal field), alive

    python tools/cem_bench.py --map maps_pool/surf_unitfarmer2.bsp --tick-ms 7 \\
        --bench-demo runs/research/uf2_wr/surf_unitfarmer2.frames.npz --bench-time 6.0 \\
        --horizon 2.4 --objective height

--bench-demo starts from a parsed record's state: an ANALYSIS BENCH (CLAUDE.md section 0) -
the record never enters the objective, and its inputs are never read.
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
    ap.add_argument("--start-npz", default=None,
                    help="instead: a kino_search archive and --start-node")
    ap.add_argument("--start-node", type=int, default=-1)
    ap.add_argument("--horizon", type=float, default=2.4, help="seconds")
    ap.add_argument("--knot", type=int, default=4, help="ticks between control knots")
    ap.add_argument("--pop", type=int, default=4096)
    ap.add_argument("--elite", type=float, default=0.04)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--max-offset", type=float, default=8.0, help="degrees")
    ap.add_argument("--objective", default="height", choices=("height", "energy", "field"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save the best sequence and its flight (.npz)")
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
    start = core.get_states()[0].copy()
    if a.bench_demo:
        w = np.load(a.bench_demo)
        i = int(np.searchsorted(w["t"] - w["t"][0], a.bench_time))
        start["origin"], start["velocity"] = w["simorg"][i], w["simvel"][i]
        start["yaw"], start["onground"] = float(w["viewangles"][i, 1]), -1
        print(f"!! ANALYSIS BENCH: start = {Path(a.bench_demo).name} at {a.bench_time:g} s "
              f"(z {w['simorg'][i][2]:.0f}, speed {np.linalg.norm(w['simvel'][i]):.0f}); the "
              f"record's inputs are never read", flush=True)
    elif a.start_npz:
        st = np.load(Path(a.start_npz).with_suffix(".states.npy"))
        start = st[a.start_node].copy()
    start["tick"], start["stuck_ticks"] = 0, 0
    start["basevelocity"] = (0.0, 0.0, 0.0)
    gf = None
    if a.objective == "field":
        from surfgym.goalfield import build_goal_field
        gf = build_goal_field(core, zones["end"], cell=32)

    H = int(round(a.horizon * 1000.0 / a.tick_ms))
    K = H // a.knot + 1
    mu = np.zeros(K)
    sig = np.full(K, 3.0)
    pD = np.full(K, 0.5)                                   # probability of D (else A)
    n_el = max(4, int(a.pop * a.elite))
    best = (-np.inf, None)
    t0 = time.perf_counter()
    for it in range(a.iters):
        off = np.clip(mu + sig * rng.standard_normal((a.pop, K)), -a.max_offset, a.max_offset)
        side = np.where(rng.random((a.pop, K)) < pD, 2, 0).astype(np.int32)
        if best[1] is not None:                            # keep the incumbent in the population
            off[0], side[0] = best[1]
        rw[:] = start
        alive = np.ones(a.pop, bool)
        act = np.zeros((a.pop, 6), np.int32)
        act[:, 0], act[:, 1], act[:, 2] = 7, 3, 1
        view = np.zeros((a.pop, 2), np.float32)
        view[:, 1] = -10.0
        for k in range(H):
            j, f = divmod(k, a.knot)
            j1 = min(j + 1, K - 1)
            view[:, 0] = off[:, j] + (off[:, j1] - off[:, j]) * (f / a.knot)
            act[:, 3] = side[:, j]
            _o, _r, done, trunc, _t = core.step(act, view=view)
            alive &= ~((done | trunc).astype(bool))
        o = np.asarray(sv["origin"], np.float64)
        v = np.asarray(sv["velocity"], np.float64)
        if a.objective == "height":
            score = o[:, 2].copy()
        elif a.objective == "energy":
            score = o[:, 2] + (v * v).sum(1) / (2 * G)
        else:
            score = -gf.sample(o)
        score[~alive] = -np.inf
        el = np.argsort(-score)[:n_el]
        if np.isfinite(score[el[0]]) and score[el[0]] > best[0]:
            best = (float(score[el[0]]), (off[el[0]].copy(), side[el[0]].copy()))
        ok = el[np.isfinite(score[el])]
        if len(ok) >= 2:
            mu = 0.7 * off[ok].mean(0) + 0.3 * mu
            sig = np.maximum(0.7 * off[ok].std(0) + 0.3 * sig, 0.15)
            pD = np.clip(0.7 * (side[ok] == 2).mean(0) + 0.3 * pD, 0.02, 0.98)
        if it % 5 == 0 or it == a.iters - 1:
            print(f"iter {it:3d}: alive {alive.mean():6.1%}  best {best[0]:8.1f}  elite mean "
                  f"{np.mean(score[ok]) if len(ok) else float('nan'):8.1f}  sigma {sig.mean():.2f}  "
                  f"[{time.perf_counter() - t0:5.1f}s]", flush=True)
    # replay the incumbent on env 0 and report its flight
    off_b, side_b = best[1]
    rw[:] = start
    traj = []
    for k in range(H):
        j, f = divmod(k, a.knot)
        j1 = min(j + 1, K - 1)
        view[:, 0] = off_b[j] + (off_b[j1] - off_b[j]) * (f / a.knot)
        act[:, 3] = side_b[j]
        _o, _r, done, trunc, _t = core.step(act, view=view)
        s0 = sv[0]
        traj.append(np.concatenate([np.asarray(s0["origin"], np.float64),
                                    np.asarray(s0["velocity"], np.float64)]))
        if done[0] or trunc[0]:
            break
    traj = np.asarray(traj)
    sp = np.linalg.norm(traj[:, 3:], axis=1)
    E = traj[:, 2] + sp ** 2 / (2 * G)
    print(f"best: objective {best[0]:.1f}; end z {traj[-1, 2]:.0f}, speed {sp[-1]:.0f}, energy "
          f"height {E[-1]:.0f}; max z {traj[:, 2].max():.0f}; |offset| p50 "
          f"{np.median(np.abs(off_b)):.2f} deg, side switches {int((np.diff(side_b) != 0).sum())}")
    if a.out:
        np.savez_compressed(a.out, off=off_b, side=side_b, traj=traj, knot=a.knot,
                            tick_ms=a.tick_ms, horizon=a.horizon)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
