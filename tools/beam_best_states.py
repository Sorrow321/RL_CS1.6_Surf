#!/usr/bin/env python3
"""beam_best_states.py - per-tick STATE_DTYPE states of a beam_tas FINISHING line (pbr8b1).

beam_tas --save-frontier writes the per-tick states of a NON-finishing frontier line
(beam_frontier.npz); a search that finishes writes only the winner's action table
(beam_best.npz) and trajectory (beam_best.jsonl). This replays the table open-loop on a fresh
1-env core from the search's own spawn row (exactly as save_frontier_line does) and writes
{"states": (T,) STATE_DTYPE} - the input of tools/stitch_spine.py's roll:NPZ:JSONL segment
(the jsonl gives the held keys), e.g. after a spine cut at the beam's --start-state row:

    python tools/beam_best_states.py --ckpt C.pt --beam runs/x/beam_best.npz --out runs/x/line.npz
    python tools/stitch_spine.py --out S.npy --field F.npz spine:CUT.npy:CUT.keys.npy \\
        roll:runs/x/line.npz:runs/x/beam_best.jsonl

The replay must finish (the core's own goal box) at the table's finish tick, or it exits 1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="the checkpoint the search ran (its config)")
    ap.add_argument("--beam", required=True, help="beam_best.npz of a finishing search")
    ap.add_argument("--map", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from beam_tas import build_sim, line_view_ticks, resolve_map
    from surfgym.tick import TickClock
    from surfgym.zones import load_zones
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    z = np.load(a.beam, allow_pickle=False)
    k = int(z["act_every"])
    acts = np.asarray(z["acts"], np.int32)
    view = np.asarray(z["view"], np.float32) if "view" in z.files else None
    fin = int(z["finish_ticks"])
    tick = TickClock(float(z["tick_ms_requested"]) if "tick_ms_requested" in z.files
                     else float(cfg.get("tick_ms") or 10.0))
    map_path = resolve_map(a.map, cfg.get("map", "surf_ski_2"))
    core = build_sim(cfg, map_path, 1, max(fin + 10, int(cfg.get("ep_ticks", 700))), tick=tick)
    zones = load_zones(core.bsp_path)
    core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
    if cfg.get("teleport_fail") or cfg.get("reward") == "race":
        core.set_teleport_fail(True)
    row0 = np.asarray(z["spawn_state"]).reshape(-1)[:1]
    core.set_spawn_pool(row0)
    core.reset(int(z["gate_seed"]))
    core.set_state(0, row0[0])
    acts_t = np.repeat(acts, k, axis=0)
    view_t = line_view_ticks(view, k)
    states, ended = [], None
    for t in range(len(acts_t)):
        states.append(core.get_states()[0].copy())
        at = np.ascontiguousarray(acts_t[t].reshape(1, 6), dtype=np.int32)
        if view_t is not None:
            _o, _r, done, trunc, info = core.step(
                at, view=np.ascontiguousarray(view_t[t].reshape(1, 2), dtype=np.float32))
        else:
            _o, _r, done, trunc, info = core.step(at)
        if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
            ended = t + 1
            break
    states = np.array(states)
    np.savez(a.out, states=states)
    ok = ended == fin
    print(f"{len(states)} states from {a.beam} (finish table tick {fin}, replay ended at "
          f"{ended}) -> {a.out}" + ("" if ok else "  !! the replay did NOT end at the finish tick"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
