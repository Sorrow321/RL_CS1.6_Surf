#!/usr/bin/env python3
"""ge_replay.py - replay a Go-Explore chain into ONE per-tick line (pbr1b1, 2026-10-23).

An archive written by tools/explore_phase1.py --exact-chains keeps, for every cell, the cell
its run was restored from (parent), that parent's state unchanged, and the run's actions from
the restore to the first entry (link_acts). Replaying the actions link by link from each
parent's archived state therefore reproduces the chain exactly; the per-tick states are the
line a backward curriculum spawns along (train_fast --demo-file). The chain end is the
archive's deepest cell on a GoalField (or --cell); nothing comes from a demo or a record.

    python tools/ge_replay.py --map maps/X.bsp --archive runs/ge/archive.npz \\
        --field maps/X.goalc_32.npz --out runs/ge/line.npy

Each link's replayed end is checked against the child's archived state (origin within
--tol u); a link that diverges is reported and the line stops there."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402
from surfgym.goalfield import GoalField                          # noqa: E402


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
    ap.add_argument("--cell", type=int, default=None,
                    help="the chain's end cell (archive index); default the deepest on the field")
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args(argv)
    z = np.load(a.archive, allow_pickle=False)
    if "link_acts" not in z.files:
        raise SystemExit(f"{a.archive} has no link_acts: run explore_phase1 with --exact-chains")
    meta = json.loads(Path(a.archive).with_suffix(".meta.json").read_text())
    st, parent = z["state"], z["parent"]
    acts, off = z["link_acts"], z["link_off"]
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
    d[d >= f.reach_max - 0.5 * f.cell] = np.inf
    end = int(np.argmin(d)) if a.cell is None else int(a.cell)
    chain = chain_to(parent, end)
    print(f"{'deepest' if a.cell is None else 'chosen'} cell {end}: d {d[end]:,.0f} at "
          f"{np.asarray(st[end]['origin']).round(0)}; chain of {len(chain)} cells")
    cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=int(meta.get("ep_ticks", 6000)),
                         water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    if meta.get("maxvel") is not None:
        cfg.phys.sv_maxvelocity = float(meta["maxvel"])
    if meta.get("tick_ms") is not None:
        # explore_phase1 --tick-ms: replay at the archive's own (whole-ms) tick
        cfg.phys.msec = int(meta["tick_ms"])
    core = SurfCore(str(a.map), cfg)
    if not meta.get("keep_teleports", False):
        core.set_teleport_fail(True)
    line = [st[chain[0]].copy()]
    ok = 0
    for k in range(1, len(chain)):
        p, c = chain[k - 1], chain[k]
        la = acts[off[c]:off[c + 1]].astype(np.int32)
        core.reset(0)
        s0 = st[p:p + 1].copy()
        s0["tick"] = 0
        s0["stuck_ticks"] = 0
        core.set_state(0, s0)
        seg = []
        dead = False
        for t in range(len(la)):
            _, _, done, trunc, _ = core.step(la[t][None, :])
            if (done[0] | trunc[0]) != 0:
                dead = True
                break
            seg.append(core.get_states()[0].copy())
        err = (float(np.linalg.norm(seg[-1]["origin"].astype(np.float64)
                                    - st[c]["origin"].astype(np.float64)))
               if seg and not dead else float("inf"))
        if dead or err > a.tol:
            print(f"link {k} ({p} -> {c}, {len(la)} ticks) DIVERGED: "
                  f"{'died' if dead else f'end off by {err:.2f} u'}; the line stops at link {ok}")
            break
        line.extend(seg)
        ok = k
    out = np.array(line, dtype=STATE_DTYPE)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    np.save(a.out, out)
    dl = f.sample(out["origin"].astype(np.float64))
    print(f"{len(out)} states -> {a.out}; links {ok}/{len(chain) - 1}; line end d "
          f"{float(dl[-1]):,.0f}, best d {float(dl.min()):,.0f}")
    return 0 if ok == len(chain) - 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
