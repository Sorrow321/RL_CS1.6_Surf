#!/usr/bin/env python3
"""shoot_finish.py - open-loop random shooting for a FINISH around the policy's OWN line
(zlr3b4, 2026-11-01).

The policy's greedy line (a tools/beam_tas.py --save-frontier npz: its per-decision engine rows,
its view rows and the per-tick states of their exact replay) passes within a few hundred units of
the finish at the top of its last structure without entering it. Its finishing variant is a
different manoeuvre over a few seconds - a held change of heading through a launch and the climb
after it - which the policy's own per-decision noise does not propose and which a population
search ranked every 0.7 s culls before it pays. This tool asks the physics alone:

* for every start second S in --at (seconds of the line; negative = from its end), the line's own
  state at S is cloned into --envs envs (env 0 keeps the line unchanged: the control);
* every other env replays the line's own rows for --horizon seconds with a PIECEWISE-CONSTANT
  perturbation: --segments segments of random length, each adding a held yaw offset
  U(-yaw_max, yaw_max) deg to the line's own view target (the velocity frame) and, with
  probability --side-flip, a held random side key instead of the line's; then --tail seconds of
  the line's own rows (open loop) after it;
* a finish is the core's own goal test (any of the zone's trigger models); every finishing env's
  rows are replayed on a fresh 1-env core from the same state and must finish on the same tick;
  each verified line is saved as beam_frontier-style npz (the line's states up to S + the
  shot's states, finished = 1, replay_ok) for tools/line_pool.py.

Nothing here reads a place on the map: the start states and the base rows are the policy's own
line, the perturbations are generic held offsets, the test is the map's finish trigger. It runs
on CPU (physics only, no policy forward): --threads caps OpenMP.

    python tools/shoot_finish.py runs/x/ckpt.pt runs/fs/greedy/beam_frontier.npz --out runs/sf \\
        --at=-36:-2:1 --envs 2048 --horizon 4 --tail 2 --segments 3 --yaw-max 30 --side-flip 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def starts_from(spec: str, line_s: float):
    out = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        if ":" in x:
            lo, hi, st = (float(u) for u in x.split(":"))
            lo, hi = (lo + line_s if lo < 0 else lo), (hi + line_s if hi < 0 else hi)
            out += list(np.arange(lo, hi + 1e-9, st))
        else:
            v = float(x)
            out.append(v + line_s if v < 0 else v)
    return [float(v) for v in out]


def plan(rng, n: int, h_ticks: int, segments: int, yaw_max: float, side_flip: float):
    """-> yaw offset (n, h_ticks) float32 deg, side override (n, h_ticks) int32 (-1 = the line's
    own key), one piecewise-constant plan per env; env 0 is the unperturbed control."""
    yaw = np.zeros((n, h_ticks), np.float32)
    side = np.full((n, h_ticks), -1, np.int32)
    for i in range(1, n):
        cuts = np.sort(rng.integers(1, h_ticks, size=max(0, segments - 1)))
        edges = np.concatenate([[0], cuts, [h_ticks]])
        for a, b in zip(edges[:-1], edges[1:]):
            yaw[i, a:b] = rng.uniform(-yaw_max, yaw_max)
            if rng.random() < side_flip:
                side[i, a:b] = rng.integers(0, 3)
    return yaw, side


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt", help="the checkpoint the line came from (its config builds the core)")
    ap.add_argument("line", help="beam_frontier.npz of the policy's own line (acts, view, states)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--at", default="-36:-2:1")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--horizon", type=float, default=4.0, help="perturbed seconds")
    ap.add_argument("--tail", type=float, default=2.0, help="the line's own rows after (s)")
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--yaw-max", type=float, default=30.0)
    ap.add_argument("--side-flip", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=1, help="batches of --envs per start")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=int, default=8, help="finishing lines saved per start")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))

    import torch
    import beam_tas as bt
    from surfgym.goalfield import build_goal_field
    from surfgym.rewards import map_spawn_pool
    from surfgym.tick import TickClock
    from surfgym.zones import load_zones
    from surfgym.core import STATE_DTYPE

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    z = np.load(a.line, allow_pickle=False)
    K = int(z["act_every"])
    acts = np.repeat(np.asarray(z["acts"], np.int32), K, axis=0)          # per tick
    view = (np.repeat(np.asarray(z["view"], np.float32).reshape(-1, 2), K, axis=0)
            if "view" in z.files else None)
    if view is None:
        raise SystemExit("the line carries no view rows (a --view-continuous checkpoint's line "
                         "is needed)")
    states = np.asarray(z["states"], STATE_DTYPE)
    gate_seed = int(z["gate_seed"]) if "gate_seed" in z.files else 0
    tick_ms = float(cfg.get("tick_ms") or 10.0)
    TICK = TickClock(tick_ms)
    tps = 1000.0 / tick_ms
    n_line = len(states)
    line_s = n_line / tps
    map_path = bt.resolve_map(None, cfg.get("map"))
    ep_cap = max(int(cfg.get("ep_ticks", 130000)), n_line + int((a.horizon + a.tail + 5) * tps))
    core1 = bt.build_sim(cfg, map_path, 1, ep_cap, tick=TICK)
    zones = load_zones(core1.bsp_path)
    gf = build_goal_field(core1, zones["end"], cell=float(cfg.get("lidar_cell") or 32.0),
                          no_corner_cut=bool(int(cfg.get("race_no_corner_cut") or 0)))
    pool = map_spawn_pool(core1)

    def arm(core):
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        core.set_teleport_fail(True)
        core.set_spawn_pool(pool)

    arm(core1)
    N = int(a.envs)
    coreN = bt.build_sim(cfg, map_path, N, ep_cap, tick=TICK)
    arm(coreN)
    h_t = int(round(a.horizon * tps))
    t_t = int(round(a.tail * tps))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    d_line = gf.sample(np.asarray(states["origin"], np.float64)).astype(np.float64)
    print(f"line {a.line}: {n_line} ticks ({line_s:.1f}s), best d {d_line.min():,.0f} at "
          f"{np.argmin(d_line) / tps:.1f}s; K {K}; envs {N}; horizon {a.horizon}s + tail "
          f"{a.tail}s", flush=True)
    summ = open(out / "summary.jsonl", "a", encoding="utf-8")
    n_saved = 0
    for s in starts_from(a.at, line_s):
        t0 = int(round(s * tps))
        t0 -= t0 % K
        if t0 <= 0 or t0 >= n_line - 1:
            continue
        S = np.array(states[t0:t0 + 1], copy=True)
        for rnd in range(max(1, a.rounds)):
            t_start = time.time()
            yaw, side = plan(rng, N, h_t, a.segments, a.yaw_max, a.side_flip)
            coreN.reset(gate_seed)
            for i in range(N):
                coreN.set_state(i, S[0])
            alive = np.ones(N, bool)
            fin_t = np.full(N, -1, np.int64)
            dmin = np.full(N, np.inf)
            zmax = np.full(N, -np.inf)
            total = h_t + t_t
            for k in range(total):
                li = min(t0 + k, n_line - 1)
                A = np.repeat(acts[li:li + 1], N, axis=0)
                V = np.repeat(view[li:li + 1], N, axis=0)
                if k < h_t:
                    V[:, 0] = ((V[:, 0] + yaw[:, k] + 180.0) % 360.0) - 180.0
                    m = side[:, k] >= 0
                    A[m, 3] = side[m, k]
                _o, _r, done, trunc, _ = coreN.step(np.ascontiguousarray(A),
                                                     view=np.ascontiguousarray(V))
                done = np.asarray(done, bool) | np.asarray(trunc, bool)
                o = np.asarray(coreN.states_view["origin"], np.float64)
                if k % 4 == 0 or done.any():
                    dd = gf.sample(o).astype(np.float64)
                    dmin = np.where(alive, np.minimum(dmin, dd), dmin)
                    zmax = np.where(alive, np.maximum(zmax, o[:, 2]), zmax)
                if done.any():
                    gh = np.asarray(coreN.goal_hits, bool)
                    hit = done & alive & gh
                    fin_t[hit] = k + 1
                    alive &= ~done
                if not alive.any():
                    break
            fins = np.flatnonzero(fin_t > 0)
            row = {"start_s": round(s, 2), "start_tick": t0, "round": rnd, "envs": N,
                   "finishes": int(len(fins)), "control_dmin": float(dmin[0]),
                   "best_dmin": float(dmin[np.isfinite(dmin)].min()) if np.isfinite(dmin).any()
                   else None, "zmax": float(zmax.max()), "alive_end": int(alive.sum()),
                   "wall_s": round(time.time() - t_start, 1)}
            # replay-verify and save up to --keep finishing shots, fastest first
            saved = []
            for i in fins[np.argsort(fin_t[fins])][:max(0, a.keep)]:
                ft = int(fin_t[i])
                core1.reset(gate_seed)
                core1.set_state(0, S[0])
                st_new, ok = [], False
                rows_a, rows_v = [], []
                for k in range(ft):
                    li = min(t0 + k, n_line - 1)
                    A = acts[li:li + 1].copy()
                    V = view[li:li + 1].copy()
                    if k < h_t:
                        V[0, 0] = ((V[0, 0] + yaw[i, k] + 180.0) % 360.0) - 180.0
                        if side[i, k] >= 0:
                            A[0, 3] = side[i, k]
                    rows_a.append(A[0])
                    rows_v.append(V[0])
                    st_new.append(core1.get_states()[0].copy())
                    _o, _r, done, trunc, _ = core1.step(np.ascontiguousarray(A),
                                                        view=np.ascontiguousarray(V))
                    if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
                        ok = bool(np.asarray(done)[0]) and bool(core1.goal_hits[0]) and k + 1 == ft
                        break
                full = np.concatenate([states[:t0], np.array(st_new, STATE_DTYPE)])
                tag = f"s{int(round(s * 10)):05d}_r{rnd}_e{int(i):05d}"
                np.savez(out / f"{tag}.npz", states=full, finished=np.int32(1),
                         replay_ok=np.int32(1 if ok else 0), start_tick=np.int32(t0),
                         finish_tick=np.int32(t0 + ft), acts=np.asarray(rows_a, np.int32),
                         view=np.asarray(rows_v, np.float32), act_every=np.int32(K),
                         yaw_plan=yaw[i], side_plan=side[i], tick_ms=np.float64(tick_ms),
                         frontier_d=np.float64(0.0), frontier_tick=np.int32(t0 + ft))
                saved.append({"tag": tag, "finish_s": round((t0 + ft) / tps, 2),
                              "replay_ok": int(ok)})
                n_saved += 1
            row["saved"] = saved
            print(json.dumps(row), flush=True)
            summ.write(json.dumps(row) + "\n")
            summ.flush()
    print(f"done: {n_saved} finishing lines saved under {out}", flush=True)


if __name__ == "__main__":
    main()
