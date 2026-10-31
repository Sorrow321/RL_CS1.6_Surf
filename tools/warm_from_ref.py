#!/usr/bin/env python3
"""warm_from_ref.py - a clean WARM START on a new map from a policy trained on another one.

A resume of a foreign checkpoint restores its whole config: its map's spine, BC set and spawn
files (paths that do not exist on this map's box), its reward constants, its curricula. This
writes a checkpoint whose WEIGHTS (and Adam moments) are the reference's and whose CONFIG is a
from-scratch run's own on the new map (the template: that run's run.json "config", e.g. the
recipe's check run), so a `--resume` trains exactly the scratch recipe from the reference's
weights. The step counter is 0 (matched-step comparisons against scratch runs read directly),
and none of the reference's run-state (novelty counts, reservoir, greedy pool, unstuck state)
is carried: it belongs to the other map.

It refuses when a SHAPE key differs between the two configs (what the policy sees, how its
actions are read, its architecture): those weights would be read in another frame.

    python tools/warm_from_ref.py --ref ~/ar_loop/incumbent/refs/uf2_x13/ckpt.pt \
        --template ~/ar_loop/runs/zlop/zlop_check/run.json --out runs/warm_uf2x13.pt \
        --set energy_pot=1 --set energy_pot_rel=1 --drop energy_pot_front --drop energy_pot_front_bonus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

# what the policy SEES, how an action is READ, and the network's shape: must agree
SHAPE_KEYS = ("view_continuous", "view_absolute", "keys_hold", "obs_potential",
              "obs_potential_curtain", "simba", "obs_normal", "obs_surf", "obs_compass",
              "obs_pot_rate", "lidar_w", "lidar_h", "lidar_cell", "lidar_range", "lidar_near",
              "emb", "hidden", "race_ratchet", "act_every", "pitch_rate", "yaw_adaptive",
              "obs_reward", "int_split", "frame_stack", "gps", "obs_texture", "obs_speed",
              "pitch_entropy", "chunk", "obs_gaze")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _val(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="the reference checkpoint (weights + optimizer)")
    ap.add_argument("--template", required=True,
                    help="run.json (or a checkpoint) of a from-scratch run on the NEW map")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="set a config key (JSON value), e.g. stall_energy=32")
    ap.add_argument("--drop", action="append", default=[], metavar="KEY",
                    help="remove a config key (so a resume uses the flag or its default)")
    a = ap.parse_args()

    ref = Path(a.ref).expanduser()
    ck = torch.load(ref, map_location="cpu", weights_only=False)
    tp = Path(a.template).expanduser()
    if tp.suffix == ".json":
        tcfg = dict(json.loads(tp.read_text(encoding="utf-8"))["config"])
    else:
        tcfg = dict(torch.load(tp, map_location="cpu", weights_only=False)["config"])
    rcfg = dict(ck.get("config") or {})
    bad = [k for k in SHAPE_KEYS if rcfg.get(k) != tcfg.get(k)]
    if bad:
        raise SystemExit("shape keys differ (reference vs template): "
                         + ", ".join(f"{k}={rcfg.get(k)!r}/{tcfg.get(k)!r}" for k in bad))
    for kv in a.set:
        k, _, v = kv.partition("=")
        tcfg[k] = _val(v)
    for k in a.drop:
        tcfg.pop(k, None)
    out = {"policy": ck["policy"], "optimizer": ck["optimizer"], "global_step": 0,
           "config": tcfg,
           # provenance, outside the config (the recorder audits config keys only)
           "warm_from": {"ref": str(ref), "md5": _md5(ref),
                         "ref_step": int(ck.get("global_step", 0)),
                         "ref_map": rcfg.get("map"), "ref_tick_ms": rcfg.get("tick_ms"),
                         "template": str(tp)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"wrote {a.out}: weights of {ref} ({rcfg.get('map')}, step "
          f"{int(ck.get('global_step', 0)):,}, tick {rcfg.get('tick_ms')}) + the config of {tp} "
          f"(map {tcfg.get('map')}, tick {tcfg.get('tick_ms')}); step 0; run-state dropped "
          f"({', '.join(k for k in ck if k not in ('policy', 'optimizer', 'global_step', 'config'))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
