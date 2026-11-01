#!/usr/bin/env python3
"""finisher_states.py - replay every kept finishing lineage of a tools/beam_tas.py search
(beam_best.npz: acts_all / view_all / acts_len / finish_ticks_all, --keep-finishers K) open-loop on
a 1-env core from the search's spawn row, exactly as beam_tas's save_frontier_line replays its
winner, and save each line's per-tick STATE_DTYPE states up to the pre-finish tick (zlr3b2,
2026-11-01).

A line whose replay does not finish on the tick the search recorded is reported and skipped. The
output .npz holds line_000, line_001, ... (fastest first), each a STATE_DTYPE array that
tools/finish_pool.py takes as --line <npz>:<key> (or use --split to write one .npy per line).

    python tools/finisher_states.py runs_ckpt.pt runs/beam/X/beam_best.npz --out runs/beam/X/fin_states.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None):
    import torch
    import beam_tas as B
    from surfgym.tick import TickClock
    from surfgym.zones import load_zones
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("best", help="beam_best.npz of a search that finished")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lines", type=int, default=0, help="0 = every kept finisher")
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    z = np.load(a.best, allow_pickle=False)
    K = int(z["act_every"])
    tick = TickClock(float(z["tick_ms"]) if "tick_ms" in z.files else float(cfg.get("tick_ms") or 10.0))
    map_path = B.resolve_map(None, cfg.get("map"))
    n = len(z["acts_len"])
    if a.max_lines > 0:
        n = min(n, a.max_lines)
    ep_cap = int(max(z["finish_ticks_all"][:n].max(), 1)) + 100
    core = B.build_sim(cfg, map_path, 1, ep_cap, tick=tick)
    zones = load_zones(core.bsp_path)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)
    row0 = z["spawn_state"]
    seed = int(z["gate_seed"])
    view_all = z["view_all"] if "view_all" in z.files else None
    out, kept = {}, 0
    for j in range(n):
        L = int(z["acts_len"][j])
        ft = int(z["finish_ticks_all"][j])
        if ft <= 0:
            continue
        acts_t = np.repeat(np.asarray(z["acts_all"][j, :L], np.int32), K, axis=0)
        view_t = (B.line_view_ticks(np.asarray(view_all[j, :L], np.float32), K)
                  if view_all is not None else None)
        core.reset(seed)
        core.set_state(0, row0)
        states, fin_t = [], -1
        for t in range(min(len(acts_t), ft)):
            states.append(core.get_states()[0].copy())
            act = np.ascontiguousarray(acts_t[t].reshape(1, 6), dtype=np.int32)
            if view_t is not None:
                _o, _r, done, trunc, _ = core.step(
                    act, view=np.ascontiguousarray(view_t[t].reshape(1, 2), dtype=np.float32))
            else:
                _o, _r, done, trunc, _ = core.step(act)
            if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
                if bool(np.asarray(done)[0]) and bool(core.goal_hits[0]):
                    fin_t = t + 1
                break
        ok = fin_t == ft
        st = np.array(states)
        e = st[-1] if len(st) else None
        print(f"line {j}: finish tick {ft} ({tick.ms * ft / 1000.0:.2f} s) replay "
              f"{'ok' if ok else f'DIVERGED (ended at {fin_t})'}"
              + (f"  last ({e['origin'][0]:.0f},{e['origin'][1]:.0f},{e['origin'][2]:.0f})"
                 if e is not None else ""))
        if ok:
            out[f"line_{kept:03d}"] = st
            kept += 1
    if not out:
        raise SystemExit("no finishing line replayed")
    np.savez(a.out, **out)
    print(f"finisher_states: {kept}/{n} lines -> {a.out}")


if __name__ == "__main__":
    main()
