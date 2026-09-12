#!/usr/bin/env python3
"""strip_ckpt.py - a SINGLE-MAP warm-start checkpoint out of a --maps one.

The gate benchmark (tools/gate_bench.py) trains a policy that already
reaches a gate, from a window of pre-gate states, on ONE map. train_fast
resumes only through --ckpt, and a joint checkpoint resumed with --map
restores its own multi-map config: the map list, per-map reservoirs, the
frontier curriculum that owns the spawn pool (--demo-file refuses to share
it). This writes a checkpoint the trainer can resume as a plain single-map
run: the weights (and the optimizer moments, by default), the config with
the map list collapsed to the one map, the frontier / backward / demo
curricula switched off, the step counter reset to 0 (--keep-step keeps it),
and every per-map run-state entry dropped. Nothing about what the policy
sees or does changes: keys-hold, the potential channel, the view mode and
the action layout are carried over as they are.

    python tools/strip_ckpt.py --ckpt runs/jt3ANCHU/ckpt_final.pt \
        --map surf_src_celestial --goal-cell 48 \
        --out runs/research/gate_bench/jt3_celestial_init.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.mapfleet import map_tag      # noqa: E402

CURRICULA_OFF = {
    "respawn_frontier": False, "respawn_frontier_anchor": False,
    "respawn_frontier_uniform": False, "respawn_frontier_quantile": 100.0,
    "respawn_backward": False, "goal_frontier": 0,
    "demo_file": None, "demo_window": None, "demo_rate": None,
    "demo_min_ep": None, "demo_grow": None,
    "heldout_maps": None, "heldout_goal_cell": None,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map", required=True, help="map stem, e.g. surf_src_celestial")
    ap.add_argument("--goal-cell", type=float, required=True)
    ap.add_argument("--cell", type=float, default=None,
                    help="lidar/occupancy cell (default: the map's entry in map_cells, else 32)")
    ap.add_argument("--keep-step", action="store_true")
    ap.add_argument("--drop-optimizer", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("config") or {})
    stem, tag = a.map, map_tag(a.map)
    maps = list(cfg.get("maps") or [])
    if maps and stem not in maps:
        raise SystemExit(f"{stem!r} is not one of the checkpoint's maps {maps}")
    cells = dict(cfg.get("map_cells") or {})
    cell = float(a.cell if a.cell is not None else cells.get(tag, 32.0))
    cfg.update(map=stem, maps=None, map_cells={tag: cell},
               goal_cell=f"{a.goal_cell:g}", goal_cells={tag: float(a.goal_cell)})
    cfg.update(CURRICULA_OFF)
    out = {"policy": ck["policy"], "config": cfg,
           "global_step": int(ck.get("global_step", 0)) if a.keep_step else 0}
    if not a.drop_optimizer and "optimizer" in ck:
        out["optimizer"] = ck["optimizer"]
    for k in ck:
        if k not in out and k not in ("optimizer", "respawn", "respawn_frontier",
                                      "respawn_backward", "int_counts", "demo"):
            v = ck[k]
            # per-map run state keyed by stem or tag collapses to this map's entry
            if isinstance(v, dict) and (stem in v or tag in v):
                out[k] = v.get(stem, v.get(tag))
            else:
                out[k] = v
    ic = ck.get("int_counts")
    if isinstance(ic, dict) and stem in ic:
        out["int_counts"] = ic[stem]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    n = sum(int(v.numel()) for v in ck["policy"].values() if hasattr(v, "numel"))
    print(f"{a.out}: {stem} (tag {tag}, cell {cell:g}, goal cell {a.goal_cell:g}), "
          f"policy {n:,} params, optimizer {'kept' if 'optimizer' in out else 'dropped'}, "
          f"global_step {out['global_step']:,}, keys {sorted(out)}")
    on = {k: cfg.get(k) for k in ("keys_hold", "obs_potential", "obs_potential_curtain",
                                  "view_continuous", "view_absolute", "act_every",
                                  "n_steps", "gae", "ent", "tick_ms", "race_field_blur")}
    print(f"  carried: {on}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
