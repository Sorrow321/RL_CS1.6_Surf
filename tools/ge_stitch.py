#!/usr/bin/env python3
"""ge_stitch.py - turn a Go-Explore archive's provenance chain into one REPLAYABLE line
(pbr1b1, 2026-10-23).

tools/explore_phase1.py keeps one state per cell and the parent it was first reached from, but
not the burst between them, so the chain root -> ... -> deepest cell is a list of states, not a
line a backward curriculum can spawn along. This walks the chain of the archive's deepest cell
(on a GoalField) and, link by link, re-finds a burst from the LINE'S current end state into the
next chain cell (its 3-D cell box, from the archive's own cell size; uniform random actions with
95% repeat, the archive's exploration rule; tools/entry_search.search), keeping every per-tick
state. Targets come from the archive itself - nothing from a demo, a record or a hand-picked box.

    python tools/ge_stitch.py --map maps/X.bsp --archive runs/ge/archive.npz \\
        --field maps/X.goalc_32.npz --out runs/ge/line.npy [--minutes-per-link 1]

A link that cannot be re-found within its budget ends the line there (the output then stops at
the last cell it did reach, and says so)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402
from surfgym.goalfield import GoalField                          # noqa: E402
import entry_search                                              # noqa: E402


def chain_to(parent: np.ndarray, i: int) -> list:
    out = []
    while i >= 0:
        out.append(int(i))
        i = int(parent[i])
    return out[::-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--minutes-per-link", type=float, default=1.0)
    ap.add_argument("--maxvel", type=float, default=3500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed-frac", type=float, default=0.9,
                    help="a link must also arrive with at least this share of the archived "
                         "cell's speed (the archive's energy dimension; 0 = position only)")
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=True)
    meta = json.loads(Path(a.archive).with_suffix(".meta.json").read_text()) \
        if Path(a.archive).with_suffix(".meta.json").exists() else {}
    cell = float(meta.get("cell", 64.0))
    st, parent = z["state"], z["parent"]
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    best = int(np.argmin(d))
    chain = chain_to(parent, best)
    print(f"deepest cell {best}: d {d[best]:,.0f} at {np.asarray(st[best]['origin']).round(0)}; "
          f"chain of {len(chain)} cells")
    cfg = default_config(num_envs=a.envs, sv_maxvelocity=float(a.maxvel), lidar_w=0, lidar_h=0,
                         spawn_mode=2, water_fail=1, yaw_jitter_deg=0.0)
    core = SurfCore(str(a.map), cfg)
    core.set_teleport_fail(True)
    core.reset(0)
    rng = np.random.default_rng(a.seed)
    root = np.zeros(1, STATE_DTYPE)
    for name in STATE_DTYPE.names:
        root[name] = st[chain[0]][name]
    line = [root[0].copy()]
    reached = 0
    for k in range(1, len(chain)):
        o = np.asarray(st[chain[k]]["origin"], np.float64)
        box = [o[0] - cell / 2, o[0] + cell / 2, o[1] - cell / 2, o[1] + cell / 2,
               o[2] - cell / 2, o[2] + cell / 2]
        t0 = time.time()
        vt = np.asarray(st[chain[k]]["velocity"], np.float64)
        smin = float(a.speed_frac) * float(np.sqrt((vt * vt).sum())) if a.speed_frac > 0 else None
        hits, n = entry_search.search(core, line[-1], a.envs, a.ticks, 4, 0.95, box, False,
                                      None, 1, time.time() + 60.0 * a.minutes_per_link, rng,
                                      speed_min=smin)
        if not hits:
            print(f"link {k}/{len(chain) - 1} -> cell {chain[k]} not re-found in {n:,} bursts "
                  f"({time.time() - t0:.0f}s): the line stops at link {reached}")
            break
        seg = hits[0][0]
        line.extend(seg[1:])
        reached = k
        print(f"link {k}/{len(chain) - 1}: {len(seg) - 1} ticks after {n:,} bursts, now at "
              f"{np.asarray(seg[-1]['origin']).round(0)}", flush=True)
    out = np.array(line, dtype=STATE_DTYPE)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    np.save(a.out, out)
    dl = f.sample(out["origin"].astype(np.float64))
    print(f"{len(out)} states -> {a.out}; links {reached}/{len(chain) - 1}; line end d "
          f"{float(dl[-1]):,.0f}, best d {float(dl.min()):,.0f}")
    return 0 if reached == len(chain) - 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
