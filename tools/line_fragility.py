#!/usr/bin/env python3
"""line_fragility.py - how fragile is a searched line (beam_best.npz)?

The policy finishes ~0.6 s behind its own planner's line, on every loop
(ledger 2026-09-05). Is that because the line is a knife edge that no
closed-loop executor can hold, or because the imitation is loose? This
replays the line's per-decision action table open-loop on the real
physics, N envs at once, each with one small perturbation, and reports
who still finishes and when:

  identity     no perturbation (must reproduce the npz's finish tick exactly)
  vel          spawn velocity scaled by (1 +- e)          e = 0.002, 0.01, 0.05
  pos          spawn origin jittered by +- u horizontally  u = 1, 4, 16
  delay        the whole table shifted late by 1 tick from a random tick
  bin          one random decision's yaw bin moved one bin (a --view-continuous
               line: its yaw command moved by +-0.25 K instead)
  room-*       the same, applied at --at-tick (e.g. 7800 = the finish-room entry)
  blend        --yaw-blend b: the core low-passes the yaw command (needs the
               rebuilt surfcore with cfg.yaw_blend)

    python tools/line_fragility.py runs/research/tas_68.54/beam_best.npz \
        --ckpt <the line's checkpoint> --n 64 --at-tick 7800

Everything is the agent's own line; nothing from the record.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE          # noqa: E402
from surfgym.tick import TickClock            # noqa: E402
import beam_tas as bt                         # noqa: E402


def load_line(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    acts = np.asarray(z["acts"], np.int32)                 # (decisions, 6)
    # --view-continuous lines carry their float view per decision
    view = np.asarray(z["view"], np.float32) if "view" in z.files else None
    k = int(z["act_every"])
    spawn = np.asarray(z["spawn_state"], STATE_DTYPE)
    if spawn.ndim == 0:
        spawn = spawn.reshape(1)
    fin = int(z["finish_ticks"]) if "finish_ticks" in z else -1
    tick_ms = float(z["tick_ms"]) if "tick_ms" in z else 10.0
    ck = str(z["ckpt"]) if "ckpt" in z else None
    return acts, k, spawn[0], fin, tick_ms, ck, view


def replay(core, states, acts_ticks, max_ticks, view_ticks=None):
    """states: (N,) STATE_DTYPE; acts_ticks: (T, N, 6) int32 (+ view_ticks
    (T, N, 2) float32 for a --view-continuous line) -> finish tick per env
    (-1 = no)"""
    n = len(states)
    for i in range(n):
        core.set_state(i, states[i])
    fin = np.full(n, -1, np.int64)
    alive = np.ones(n, bool)
    T = min(int(max_ticks), acts_ticks.shape[0])
    for t in range(T):
        a = np.ascontiguousarray(acts_ticks[t], dtype=np.int32)
        if view_ticks is None:
            _o, _r, done, trunc, _term = core.step(a)
        else:
            _o, _r, done, trunc, _term = core.step(
                a, view=np.ascontiguousarray(view_ticks[t], dtype=np.float32))
        hits = np.asarray(core.goal_hits, np.int64) > 0
        newly = alive & hits & (fin < 0)
        fin[newly] = t + 1
        ended = np.asarray(done, bool) | np.asarray(trunc, bool) | hits
        alive &= ~ended
        if not alive.any():
            break
    return fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--ckpt", default=None, help="checkpoint whose config built the line (default: the npz's)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--at-tick", type=int, default=0, help="apply the perturbation at this tick (0 = spawn)")
    ap.add_argument("--yaw-blend", type=float, default=None, help="override cfg.yaw_blend (rebuilt core)")
    ap.add_argument("--kinds", default="identity,vel0.002,vel0.01,vel0.05,pos1,pos4,pos16,delay,bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    acts, k, spawn, fin_ref, tick_ms, ck_path, view = load_line(a.npz)
    ck_path = a.ckpt or ck_path
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("config") or {})
    if a.yaw_blend is not None:
        cfg["yaw_blend"] = float(a.yaw_blend)
    # absolute path into the MAIN checkout (CLAUDE.md: a worktree copy has other mtimes and rebakes every cache)
    stem = Path(str(cfg.get("map") or "surf_src_cannonball")).stem
    map_path = str(Path("C:/RL_Surf/maps") / f"{stem}.bsp")
    tick = TickClock(tick_ms)
    T_dec = acts.shape[0]
    T = T_dec * k
    ep_cap = int(cfg.get("ep_ticks") or 12000)
    core = bt.build_sim(cfg, map_path, a.n, max(ep_cap, T + 10), tick=tick)
    zones = json.loads((Path(map_path).with_suffix(".zones.json")).read_text(encoding="utf-8"))
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    base_ticks = np.repeat(acts, k, axis=0)                       # (T, 6)
    base_view = None if view is None else np.repeat(view, k, axis=0)   # (T, 2)
    if (view is not None) != bool(cfg.get("view_continuous")):
        raise SystemExit("the line and the checkpoint disagree about --view-continuous")
    if cfg.get("view_absolute"):
        raise SystemExit(f"checkpoint trained with --view-absolute "
                         f"{cfg['view_absolute']}: its lines carry targets, "
                         "and the +-0.25 K perturbation is a delta-space one; "
                         "not implemented")
    rng = np.random.default_rng(a.seed)
    results = {}
    print(f"line {Path(a.npz).name}: {T_dec} decisions x {k} = {T} ticks, npz finish {fin_ref}, "
          f"tick {tick_ms} ms, ckpt {Path(ck_path).name}, yaw_blend {cfg.get('yaw_blend', 1.0)}, at-tick {a.at_tick}"
          + (", continuous view" if view is not None else ""))

    # state at --at-tick: replay the unperturbed line to it once
    states0 = np.repeat(spawn.reshape(1), a.n, axis=0).copy()
    if a.at_tick > 0:
        for i in range(a.n):
            core.set_state(i, spawn)
        for t in range(a.at_tick):
            _a = np.ascontiguousarray(np.repeat(base_ticks[t].reshape(1, 6), a.n, axis=0), np.int32)
            if base_view is None:
                core.step(_a)
            else:
                core.step(_a, view=np.ascontiguousarray(np.repeat(base_view[t].reshape(1, 2), a.n, axis=0), np.float32))
        st_at = np.asarray(core.states_view)[0].copy()
        states0 = np.repeat(st_at.reshape(1), a.n, axis=0).copy()
    t_start = a.at_tick

    for kind in a.kinds.split(","):
        kind = kind.strip()
        states = states0.copy()
        table = np.repeat(base_ticks[t_start:][:, None, :], a.n, axis=1).copy()   # (T', N, 6)
        vtable = (None if base_view is None
                  else np.repeat(base_view[t_start:][:, None, :], a.n, axis=1).copy())   # (T', N, 2)
        if kind.startswith("vel"):
            e = float(kind[3:])
            f = rng.uniform(1 - e, 1 + e, size=(a.n, 1)).astype(np.float32)
            states["velocity"] = (states["velocity"] * f).astype(states["velocity"].dtype)
        elif kind.startswith("pos"):
            u = float(kind[3:])
            j = rng.uniform(-u, u, size=(a.n, 3)).astype(np.float32); j[:, 2] = 0.0
            states["origin"] = (states["origin"] + j).astype(states["origin"].dtype)
        elif kind == "delay":
            for i in range(a.n):
                t0 = int(rng.integers(0, max(1, table.shape[0] - 2)))
                table[t0 + 1:, i] = table[t0:-1, i]
                if vtable is not None:
                    vtable[t0 + 1:, i] = vtable[t0:-1, i]
        elif kind == "bin":
            for i in range(a.n):
                d0 = int(rng.integers(0, max(1, (table.shape[0] // k) - 1)))
                sl = slice(d0 * k, (d0 + 1) * k)
                if vtable is None:
                    cur = int(table[sl, i, 0][0])
                    table[sl, i, 0] = min(14, max(0, cur + (1 if rng.random() < 0.5 else -1)))
                else:
                    # the continuous twin of "one bin over": +-0.25 K
                    vtable[sl, i, 0] = np.clip(vtable[sl, i, 0] + (0.25 if rng.random() < 0.5 else -0.25), -20.0, 20.0)
        elif kind != "identity":
            print("unknown kind", kind); continue
        fin = replay(core, states, table, table.shape[0], vtable)
        finished = fin[fin >= 0] + t_start
        rate = float((fin >= 0).mean())
        line = (f"{kind:10s} finish {rate*100:5.1f}%  "
                + (f"ticks min/med/max {finished.min()}/{int(np.median(finished))}/{finished.max()} "
                   f"(ref {fin_ref}; +{(np.median(finished) - fin_ref) * tick_ms / 1000:.3f} s median)" if len(finished) else "no finishes"))
        print(line)
        results[kind] = {"finish_rate": rate, "finish_ticks": finished.tolist()}
    if a.out:
        Path(a.out).write_text(json.dumps({"npz": a.npz, "ckpt": ck_path, "at_tick": a.at_tick,
                                           "yaw_blend": cfg.get("yaw_blend", 1.0), "results": results}, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
