#!/usr/bin/env python3
"""cem_to_plan.py - pack tools/cem_finish.py lines into beam_best.npz-layout plan files.

tools/plan_to_bc.py turns a planner's finishing lines into behaviour-cloning rows (and replays
each one, exact, to record the per-decision states and core scalars). Its input is beam_tas's
beam_best.npz: one spawn state, the per-decision ENGINE action rows of every kept line
(acts_all / acts_len / finish_ticks_all) and, for a --view-continuous checkpoint, the per-
decision view rows (view_all: yaw target deg in the velocity frame, pitch target deg). The CEM
lines are exactly that shape, one root (= spawn) per group: this writes one plan file per root.

The decision-0 scalars (obs_start) are the map spawn's reset row, as beam_tas --start-state
does for a mid-line root (set_state does not refresh the obs buffer; the core recomputes them
on the first step).

    python tools/cem_to_plan.py CKPT 'autoresearch/spines/zlr3b3_cem/*_fin_*.npz' --out runs/plans
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def engine_rows(z) -> tuple[np.ndarray, np.ndarray]:
    """One CEM line -> (acts (D, 6) engine bins, view (D, 2) f32)."""
    side = np.asarray(z["side"], np.int64)
    D = len(side)
    acts = np.zeros((D, 6), np.int64)
    acts[:, 0] = 7                      # yaw bin: ignored under --view-continuous
    acts[:, 1] = 3                      # pitch bin: ignored likewise
    acts[:, 2] = np.asarray(z["fwd"], np.int64)
    acts[:, 3] = side
    acts[:, 4] = 0
    acts[:, 5] = (np.asarray(z["duck"]) > 0).astype(np.int64)
    view = np.zeros((D, 2), np.float32)
    view[:, 0] = np.asarray(z["yaw"], np.float32)
    return acts, view


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("lines", help="glob of cem_finish fin_*.npz files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    import beam_tas as BT
    from surfgym.tick import TickClock
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tick_ms = float(cfg.get("tick_ms") or 10.0)
    map_path = str(ROOT / "maps" / f"{cfg['map']}.bsp")
    core = BT.build_sim(cfg, map_path, 1, int(cfg.get("ep_ticks") or 130000),
                        tick=TickClock(tick_ms))
    obs0 = np.asarray(core.reset(0), np.float32)[0].copy()
    groups = defaultdict(list)
    for f in sorted(glob.glob(a.lines)):
        z = np.load(f, allow_pickle=False)
        if int(z["finished"]) != 1 or int(z["replay_ok"]) != 1:
            continue
        groups[(str(z["pool"]), int(z["root"]))].append((f, z))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for (pool_path, root), items in sorted(groups.items(), key=lambda kv: kv[0][1]):
        pool = np.load(pool_path if Path(pool_path).is_absolute() else ROOT / pool_path,
                       allow_pickle=False)
        st = pool[root:root + 1].copy()
        st["tick"] = 0
        st["stuck_ticks"] = 0
        K = int(items[0][1]["act_every"])
        rows = [engine_rows(z) for _, z in items]
        Dmax = max(len(r[0]) for r in rows)
        aa = np.zeros((len(rows), Dmax, 6), np.int8)
        va = np.zeros((len(rows), Dmax, 2), np.float32)
        al = np.zeros(len(rows), np.int32)
        ft = np.zeros(len(rows), np.int32)
        for j, ((acts, view), (_, z)) in enumerate(zip(rows, items)):
            # the action table runs to the finishing decision (fin_tick ticks)
            n_d = min(len(acts), int(np.ceil(int(z["fin_tick"]) / K)))
            aa[j, :n_d] = acts[:n_d]
            va[j, :n_d] = view[:n_d]
            al[j] = n_d
            ft[j] = int(z["fin_tick"])
        best = int(np.argmin(ft))
        dst = out / f"plan_root{root:06d}.npz"
        np.savez(dst, acts=aa[best, :al[best]], view=va[best, :al[best]],
                 acts_all=aa, acts_len=al, finish_ticks_all=ft, view_all=va,
                 finish_ticks=np.int32(ft[best]), spawn_state=st, obs_start=obs0,
                 gate_seed=np.int32(0), greedy_ticks=np.int32(0), act_every=np.int32(K),
                 view_continuous=np.int32(1), view_mode=np.int32(1),
                 tick_ms=np.float64(tick_ms), tick_ms_requested=np.float64(tick_ms),
                 tick_pattern_ms=np.array([int(round(tick_ms))], np.int32),
                 map=np.str_(map_path), ckpt=np.str_(str(a.ckpt)),
                 objective=np.str_("finish"))
        print(f"{dst}: root {root}, {len(rows)} line(s), finish ticks {ft.tolist()}")


if __name__ == "__main__":
    main()
