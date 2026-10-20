#!/usr/bin/env python3
"""ge_spawns.py - policy-guided Go-Explore: roots from a checkpoint's own reservoir, and a
--spawn-states file from the archives tools/explore_phase1.py grows from those roots.

The loop (autoresearch r1b4, 2026-10-20, surf_src_sidistic's CP2 wall):

    1. roots    the DEEPEST Q of a checkpoint's own respawn reservoir by goal-field distance
                (the states the policy itself reaches nearest where it dies);
    2. search   tools/explore_phase1.py --roots-spine <roots> grows a cell archive from them
                with random held macro-actions, by native state restore (no policy, no reward);
    3. spawns   the archive's cells that are (a) NOVEL: outside every 128 u cell (and its 3x3x3
                block) the policy's own reservoir and greedy evals visited, (b) NEVER PAST THE
                POLICY'S OWN BEST: goal-field d >= d_best (no forward teleport), and (c) at
                least as FAST as the policy's own frontier states (median horizontal speed of
                the deepest Q of its reservoir) - random macro-actions mostly stall, and a
                stalled state teaches nothing about a surf map; plus, optionally, the chains
                (root -> cell) leading to the archive's K deepest cells, weighted xW: the
                approaches to the deepest points the search reached, cut at d_best;
    4. train    train_fast.py --spawn-states <file> --spawn-states-frac F (SELF_STATES=1 at
                tools/run_arm.sh: the file is rooted in the policy's own states and extended
                by search; no human demo). The trainer re-reads the file when its mtime
                changes, so a refreshed file can be mv'd in under a running arm.

Every input is generic: the field d, the policy's own states and visits, its own speed. No map
box, no demo, no threshold read off a map.

    python tools/ge_spawns.py roots runs/x/ckpt_final.pt --out roots.npy --q 0.03
    python tools/explore_phase1.py --map maps/<m>.bsp --out runs/ge --roots-spine roots.npy \
        --cell 128 --cell-speed 8 --cell-view 8 --decisions 100 --act-every 4 --max-minutes 8
    python tools/ge_spawns.py spawns runs/x/ckpt_final.pt runs/ge/archive.npz \
        --evals runs/x/traj_*.jsonl --chains 300 --chain-weight 30 --out spawns.npy

The checkpoint's map and field are read from its config (maps/<map>.bsp, goal_32.npz).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                     # noqa: E402
from surfgym.goalfield import GoalField                  # noqa: E402

CELL = 128.0


def load_field(map_name: str, cell: int = 32) -> GoalField:
    z = np.load(ROOT / "maps" / f"{map_name}.goal_{cell}.npz", allow_pickle=False)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]), z["mins"],
                     float(z["cell"]), float(z["reach_max"]))


def load_ckpt(path: str):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    res = (ck.get("respawn") or {}).get("states")
    if res is None or not len(res):
        raise SystemExit(f"{path}: no respawn reservoir in the checkpoint")
    return str(cfg.get("map")), np.asarray(res)


def field_d(f: GoalField, states) -> np.ndarray:
    return np.asarray(f.sample(np.asarray(states["origin"], np.float64)), np.float64)


def deepest(f: GoalField, states, q: float):
    """(rows, threshold): the states at or under the q-quantile of field distance."""
    d = field_d(f, states)
    lim = f.reach_max - 0.5 * f.cell
    ok = d < lim
    thr = float(np.quantile(d[ok], q))
    return states[ok & (d <= thr)], thr


def episode_points(paths) -> list:
    """Every recorded position of the given trajectories (record_ckpt / eval .jsonl format)."""
    pts = []
    for pat in paths:
        for p in glob.glob(pat):
            for line in open(p, encoding="utf-8"):
                if line.startswith("["):
                    r = json.loads(line)
                    pts.append(r[1:4])
    return pts


def _keys(c: np.ndarray) -> np.ndarray:
    return (c[:, 0] * 100003 + c[:, 1]) * 100019 + c[:, 2]


def visited_set(f: GoalField, pts: np.ndarray) -> set:
    c = np.floor((np.asarray(pts, np.float64) - f.mins) / CELL).astype(np.int64)
    out = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out.update(_keys(c + [dx, dy, dz]).tolist())
    return out


def novel_mask(f: GoalField, states, vset: set) -> np.ndarray:
    c = np.floor((np.asarray(states["origin"], np.float64) - f.mins) / CELL).astype(np.int64)
    return np.array([k not in vset for k in _keys(c).tolist()], bool)


def chains_to_deepest(f: GoalField, st, parent, k: int, d_floor: float):
    """Archive indices on the root -> cell chains of the k deepest cells, cut at d_floor."""
    d = field_d(f, st)
    lim = f.reach_max - 0.5 * f.cell
    ok = d < lim
    order = np.flatnonzero(ok)[np.argsort(d[ok])][:k]
    keep = set()
    for i in order:
        j = int(i)
        while j >= 0 and j not in keep:
            keep.add(j)
            j = int(parent[j])
    idx = np.array(sorted(keep), np.int64)
    if not len(idx):
        return idx
    return idx[ok[idx] & (d[idx] >= d_floor)]


def cmd_roots(a) -> None:
    mp, res = load_ckpt(a.ckpt)
    f = load_field(mp)
    rows, thr = deepest(f, res, a.q)
    rows = rows.copy()
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    np.save(a.out, rows)
    print(f"{len(rows)} roots from {a.ckpt} ({mp}): the deepest {a.q:.0%} of its reservoir, "
          f"field d <= {thr:,.0f}")


def cmd_spawns(a) -> None:
    mp, res = load_ckpt(a.ckpt)
    f = load_field(mp)
    lim = f.reach_max - 0.5 * f.cell
    d_res = field_d(f, res)
    pts = [np.asarray(res["origin"], np.float64)]
    ev = episode_points(a.evals or [])
    # the policy's own best: the reservoir harvests only states >= --respawn-margin before an
    # episode's end, so its deepest row is short of where the greedy evals get; take both
    d_all = d_res
    if ev:
        pts.append(np.asarray(ev, np.float64))
        d_all = np.concatenate([d_res, np.asarray(f.sample(pts[-1]), np.float64)])
    d_best = (float(a.d_best) if a.d_best is not None
              else float(d_all[d_all < lim].min()))
    vset = visited_set(f, np.concatenate(pts))
    front, _ = deepest(f, res, a.q)
    v = np.asarray(front["velocity"], np.float64)
    speed_ref = float(np.median(np.hypot(v[:, 0], v[:, 1])))
    out = []
    for arch in a.archives:
        z = np.load(arch, allow_pickle=False)
        st, par = z["state"], z["parent"]
        d = field_d(f, st)
        vv = np.asarray(st["velocity"], np.float64)
        hs = np.hypot(vv[:, 0], vv[:, 1])
        keep = (d < lim) & (d >= d_best) & (hs >= speed_ref) & novel_mask(f, st, vset)
        out.append(st[keep])
        msg = f"{arch}: {len(st):,} cells, {int(keep.sum()):,} novel & fast & under d_best"
        if a.chains > 0:
            ci = chains_to_deepest(f, st, par, a.chains, d_best)
            out.extend([st[ci]] * max(1, int(a.chain_weight)))
            msg += f"; {len(ci):,} chain states to its {a.chains} deepest cells x{a.chain_weight}"
        print(msg)
    rows = np.concatenate(out) if out else np.zeros(0, STATE_DTYPE)
    if not len(rows):
        raise SystemExit("no spawn state passed the filters")
    rows = rows.copy()
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    np.save(a.out, rows)
    print(f"{len(rows):,} spawn rows -> {a.out}  (d_best {d_best:,.0f}, frontier speed median "
          f"{speed_ref:,.0f} u/s, {len(vset):,} visited-cell keys)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("roots", help="the deepest Q of a checkpoint's own reservoir")
    p.add_argument("ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--q", type=float, default=0.03)
    p.set_defaults(fn=cmd_roots)
    p = sub.add_parser("spawns", help="a --spawn-states file from explore_phase1 archives")
    p.add_argument("ckpt", help="the policy whose visits, best and speed define the filters")
    p.add_argument("archives", nargs="+")
    p.add_argument("--evals", nargs="*", default=None,
                   help="the policy's greedy eval .jsonl files (globs): more visited cells")
    p.add_argument("--q", type=float, default=0.03,
                   help="the reservoir share that defines the frontier speed")
    p.add_argument("--d-best", type=float, default=None,
                   help="override the policy's best field distance (default: its reservoir's)")
    p.add_argument("--chains", type=int, default=0,
                   help="also the chains to the archive's K deepest cells (cut at d_best)")
    p.add_argument("--chain-weight", type=int, default=1)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_spawns)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
