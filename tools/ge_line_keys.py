#!/usr/bin/env python3
"""ge_line_keys.py - replay a Go-Explore chain into a per-tick line AND its held keys
(pbfr1b4, 2026-10-29).

tools/ge_replay.py writes the chain's per-tick states; a --keys-hold policy spawned on one of
them should also start holding the keys the line was flown with (train_fast --demo-keys: a
(rows, 3) int array of engine bins fwd, side, duck). The explorer's actions are engine
MultiDiscrete rows [yaw, pitch, fwd, side, jump, duck], so the held keys of row i are the
fwd / side / duck bins of the action that produced it (row 0, the chain's root, holds none:
fwd 1, side 1, duck 0). The replay and its checks are ge_replay's (every link's end against
the child's archived state, --tol u).

    python tools/ge_line_keys.py --map maps/X.bsp --archive runs/ge/archive.npz \\
        --field maps/X.goalc_32.npz --out runs/ge/line.npy [--cell N] [--cut-prog P]

--cut-prog P keeps the line up to its first row at >= P percent progress on --field (the
progress the grader reads, from the chain's root), so a spine can end where it does what the
greedy does not. Writes OUT and OUT with .keys.npy. Nothing comes from a demo or a record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402
from surfgym.goalfield import GoalField                          # noqa: E402
from ge_replay import chain_to                                   # noqa: E402

NEUTRAL_KEYS = (1, 1, 0)     # fwd none, side none, duck off (engine bins)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=None,
                    help="the chain's end cell (archive index); default the deepest on the field")
    ap.add_argument("--cut-prog", type=float, default=None,
                    help="keep rows up to the first one at >= this progress percent (from the root)")
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
    print(f"{'deepest' if a.cell is None else 'chosen'} cell {end}: d {d[end]:,.0f}; chain of "
          f"{len(chain)} cells")
    cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=int(meta.get("ep_ticks", 6000)),
                         water_fail=1, yaw_jitter_deg=0.0, lidar_w=0, lidar_h=0)
    if meta.get("maxvel") is not None:
        cfg.phys.sv_maxvelocity = float(meta["maxvel"])
    if meta.get("tick_ms") is not None:
        cfg.phys.msec = int(meta["tick_ms"])
    core = SurfCore(str(a.map), cfg)
    if not meta.get("keep_teleports", False):
        core.set_teleport_fail(True)
    line = [st[chain[0]].copy()]
    keys = [NEUTRAL_KEYS]
    for k in range(1, len(chain)):
        p, c = chain[k - 1], chain[k]
        la = acts[off[c]:off[c + 1]].astype(np.int32)
        core.reset(0)
        s0 = st[p:p + 1].copy()
        s0["tick"] = 0
        s0["stuck_ticks"] = 0
        core.set_state(0, s0)
        seg, kseg = [], []
        for t in range(len(la)):
            _, _, done, trunc, _ = core.step(la[t][None, :])
            if (done[0] | trunc[0]) != 0:
                raise SystemExit(f"link {k} ({p} -> {c}) died at tick {t}: not an exact chain")
            seg.append(core.get_states()[0].copy())
            kseg.append((int(la[t][2]), int(la[t][3]), int(la[t][5])))
        err = float(np.linalg.norm(seg[-1]["origin"].astype(np.float64)
                                   - st[c]["origin"].astype(np.float64)))
        if err > a.tol:
            raise SystemExit(f"link {k} ({p} -> {c}) end off by {err:.2f} u: not an exact chain")
        line.extend(seg)
        keys.extend(kseg)
    out = np.array(line, dtype=STATE_DTYPE)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    kk = np.asarray(keys, np.int64)
    root_d = float(f.sample(out["origin"][:1].astype(np.float64))[0])
    prog = 100.0 * (root_d - f.sample(out["origin"].astype(np.float64))) / root_d
    if a.cut_prog is not None:
        hit = np.flatnonzero(prog >= float(a.cut_prog))
        if not len(hit):
            raise SystemExit(f"the line never reaches {a.cut_prog}% (max {prog.max():.2f}%)")
        out, kk, prog = out[:hit[0] + 1], kk[:hit[0] + 1], prog[:hit[0] + 1]
    np.save(a.out, out)
    kp = str(Path(a.out).with_suffix("")) + ".keys.npy"
    np.save(kp, kk)
    print(f"{len(out)} rows -> {a.out} (+ {kp}); end progress {prog[-1]:.2f}%, max {prog.max():.2f}%, "
          f"{len(out) * int(cfg.phys.msec) / 1000.0:.2f} s at {int(cfg.phys.msec)} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
