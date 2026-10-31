#!/usr/bin/env python3
"""ckpt_retarget.py - a single-map checkpoint made into a warm start on ANOTHER map.

train_fast resumes a checkpoint with its own config, and several entries of that config are bound
to the map it was trained on: the map itself (run_arm.sh's resume branch passes no --map, and its
pre-launch record gate records the source checkpoint on the map its config names), the spawn /
spine / BC files of that map's curricula, and the run state the checkpoint carries (the respawn
reservoir, the novelty count table, the unstuck plateau tracker, all in that map's coordinates or
progress units). This writes a copy whose config names the new map, with every map-bound file
cleared, the demo-curriculum knobs that refuse to run without a spine switched off, and that run
state dropped. The weights, the optimizer moments, the step counter (a finished --tick-ms-schedule
keys on it) and every shape flag are kept as they are: what the policy sees and does is unchanged.

    python tools/ckpt_retarget.py --ckpt refs/pb_r26/ckpt.pt --map surf_unitfarmer2 \\
        --out runs/pb_r26_on_uf2.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# map-bound config entries: files of the source map's curricula, and the knobs that only mean
# something (or refuse to run) with them
CLEAR = {
    "demo_file": None, "demo_keys": None, "bc_file": None, "spawn_states": None,
    "spawn_states_frac": None, "route": None, "goal_route": None, "ramp_sequence": None,
    "demo_front_frac": 0.0, "demo_window": None, "demo_rate": None, "demo_min_ep": None,
    "demo_grow": None, "demo_success_d": None, "demo_success_bonus": None,
    "ge_minutes": None, "ge_frac": None, "ge_cell": None, "ge_speed": None,
    "obs_potential_d0": None,
}
# run state in the source map's coordinates / progress units
RUN_STATE = ("respawn", "int_counts", "unstuck")


def retarget(ck: dict, map_stem: str) -> tuple[dict, list[str]]:
    out = dict(ck)
    cfg = dict(ck.get("config") or {})
    notes = [f"map {cfg.get('map')!r} -> {map_stem!r}"]
    cfg["map"] = map_stem
    for k, v in CLEAR.items():
        if cfg.get(k) not in (None, "", 0, 0.0) and cfg.get(k) != v:
            notes.append(f"{k} {cfg.get(k)!r} -> {v!r}")
        if k in cfg or v is not None:
            cfg[k] = v
    out["config"] = cfg
    for k in RUN_STATE:
        if out.get(k) is not None:
            notes.append(f"run state {k!r} dropped")
        out.pop(k, None)
    return out, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map", required=True, help="the target map's stem (maps/<stem>.bsp)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if (ck.get("config") or {}).get("n_maps", 1) not in (None, 1):
        raise SystemExit("a --maps checkpoint: use tools/strip_ckpt.py first")
    out, notes = retarget(ck, a.map)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"{a.ckpt} -> {a.out} (step {int(out.get('global_step', 0)):,})")
    for n in notes:
        print("  " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
