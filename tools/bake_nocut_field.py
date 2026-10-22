#!/usr/bin/env python3
"""bake_nocut_field.py - bake the goal field WITHOUT corner cutting and check it.

The trainer's goal field is a 26-neighbour wavefront over the map's free voxels. A
diagonal step there may squeeze between two solid voxels that touch only along an edge,
so a one-voxel-thick SLANTED wall is no wall: on surf_src_sidistic the finish funnel's
walls read ~300 u shorter from outside than the real way over their rim, and the
shaping pays hugging the wall (autoresearch r5b4, 2026-10-22). This bakes
``build_goal_field(..., no_corner_cut=True)`` (cache ``<map>.goalc_<cell>.npz``, the
trainer's GoalField format, loadable with ``--goal-field-file``) and reports, against the
shipped field:

* the start's geodesic in both (the new one must be finite - a disconnect is refused);
* with ``--traj``: both fields along greedy eval episodes - the difference d_new - d_old
  along each episode (a constant offset changes no reward under the ratchet and no
  observation under norm / compass / pot-rate; jumps are where the shape changed);
* with ``--probe x,y,z ...``: both fields at those points.

Generic: one grid rule, no map constant.

    python tools/bake_nocut_field.py --map maps/surf_src_sidistic.bsp \\
        --traj runs/r5b4_ps05/traj_15796797440.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def _episodes(path):
    eps, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                eps.append(np.asarray(cur, np.float64))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return eps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", default=str(ROOT / "maps" / "surf_src_sidistic.bsp"))
    ap.add_argument("--cell", type=float, default=32.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--probe", nargs="*", default=[])
    a = ap.parse_args(argv)

    from surfgym.core import SurfCore, default_config
    from surfgym.goalfield import build_goal_field
    from surfgym.rewards import map_spawn_pool
    from surfgym.zones import load_zones

    core = SurfCore(a.map, default_config(1))
    zone = load_zones(a.map)["end"]
    t0 = time.time()
    old = build_goal_field(core, zone, cell=a.cell, device=a.device)
    new = build_goal_field(core, zone, cell=a.cell, device=a.device, no_corner_cut=True)
    print(f"fields ready in {time.time() - t0:.0f} s; new reach_max {new.reach_max:.0f} "
          f"(old {old.reach_max:.0f})")
    sp = map_spawn_pool(core)["origin"]
    d_old, d_new = old.sample(sp), new.sample(sp)
    print(f"start geodesic: old {np.mean(d_old):.0f}  new {np.mean(d_new):.0f} "
          f"(+{np.mean(d_new) - np.mean(d_old):.0f})")
    if not new.reachable(sp).all():
        raise SystemExit("no-corner-cut field DISCONNECTS the start from the finish - "
                         "do not train on it")
    for s in a.probe:
        p = np.array([[float(v) for v in s.split(",")]])
        print(f"probe {s}: old {float(old.sample(p)[0]):.0f}  new {float(new.sample(p)[0]):.0f}")
    for path in a.traj:
        for i, r in enumerate(_episodes(path)):
            p = r[:, 1:4]
            do, dn = old.sample(p).astype(np.float64), new.sample(p).astype(np.float64)
            ok = (do < old.reach_max - 0.5 * old.cell) & (dn < new.reach_max - 0.5 * new.cell)
            if not ok.any():
                continue
            diff = (dn - do)[ok]
            jumps = np.abs(np.diff(diff))
            bad = int((~(dn < new.reach_max - 0.5 * new.cell) & (do < old.reach_max - 0.5 * old.cell)).sum())
            best_o, best_n = float(do[ok].min()), float(dn[ok].min())
            print(f"{Path(path).name} ep{i}: {len(r)} ticks, diff new-old min {diff.min():.0f} "
                  f"median {np.median(diff):.0f} max {diff.max():.0f}; largest tick-to-tick "
                  f"change of the diff {jumps.max() if len(jumps) else 0:.0f} u; ticks reachable "
                  f"only in the old field {bad}; best d old {best_o:.0f} new {best_n:.0f}")
            # where the diff moves: report the 5 largest step changes with positions
            if len(jumps):
                top = np.argsort(-jumps)[:5]
                idx = np.flatnonzero(ok)
                for k in sorted(top):
                    q = p[idx[k + 1]]
                    print(f"    t={idx[k + 1] / 100:7.2f}s at ({q[0]:.0f},{q[1]:.0f},{q[2]:.0f}): "
                          f"diff {diff[k]:.0f} -> {diff[k + 1]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
