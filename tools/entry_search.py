#!/usr/bin/env python3
"""entry_search.py - find short lines from a root state into a target region by random bursts,
and keep their per-tick states (pbr1b1, 2026-10-23).

The Go-Explore archive (tools/explore_phase1.py) keeps one state per cell, not the line that
reached it, and on surf_princessburglar_b26 the start room's way out is a two-step duck-jump
that uniform random bursts complete about once in 50,000 tries. This runs that same burst rule
(uniform MultiDiscrete actions held with 95% repeat per decision, the archive's exploration
step) from ONE root state on a batched eyeless core until K bursts reach the target region,
and saves each successful burst's per-tick STATE_DTYPE rows (the root first, the target entry
last) for use as spawn states of the agent's own search (train_fast --spawn-states).

Stages chain: --stage-roots FILE takes the LAST state of each line in a previous stage's
output as roots (one search per root), so spawn -> ledge -> tunnel is two short searches, not
one long one. The target is a box in map units plus optional conditions (on the ground, feet
height); nothing here reads a reward, a demo or a record.

    python tools/entry_search.py --map maps/X.bsp --out a.npz --target "640 725 -2965 -2900 3600 3800" \\
        --onground --feet-min 3674 --k 8 --ticks 300 --minutes 3
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

from surfgym.core import ACTION_NVEC, STATE_DTYPE, SurfCore, default_config  # noqa: E402


def feet_z(st: np.ndarray) -> np.ndarray:
    """The hull's bottom: standing hull half-height 36, ducked 18."""
    return st["origin"][..., 2] - np.where(st["ducked"] > 0, 18.0, 36.0)


def in_target(st: np.ndarray, box, onground: bool, feet_min) -> np.ndarray:
    o = st["origin"]
    m = ((o[..., 0] >= box[0]) & (o[..., 0] <= box[1]) & (o[..., 1] >= box[2])
         & (o[..., 1] <= box[3]) & (o[..., 2] >= box[4]) & (o[..., 2] <= box[5]))
    if onground:
        m &= st["onground"] != -1
    if feet_min is not None:
        m &= feet_z(st) >= float(feet_min)
    return m


def search(core, root, n: int, ticks: int, act_every: int, repeat_p: float, box, onground,
           feet_min, k: int, deadline: float, rng) -> list:
    """Random bursts from ``root`` until ``k`` reach the target or the deadline passes;
    returns [(states (t+1,) STATE_DTYPE, actions (t+1, 6) int8)] per success."""
    root = root.copy()
    root["tick"] = 0
    root["stuck_ticks"] = 0
    hits = []
    act = np.zeros((n, 6), np.int32)

    def fresh(m):
        c = int(m.sum())
        if c:
            act[m, 0] = rng.integers(0, ACTION_NVEC[0], c)
            act[m, 1] = 3                                  # pitch is inert (src/env.c)
            for h in (2, 3, 4, 5):
                act[m, h] = rng.integers(0, ACTION_NVEC[h], c)

    bursts = 0
    while len(hits) < k and time.time() < deadline:
        for j in range(n):
            core.set_state(j, root)
        fresh(np.ones(n, bool))
        traj = np.zeros((ticks + 1, n), STATE_DTYPE)
        hist = np.zeros((ticks + 1, n, 6), np.int8)
        traj[0] = core.get_states()
        alive = np.ones(n, bool)
        first = np.full(n, -1)
        for t in range(ticks):
            if t % act_every == 0 and t:
                fresh(rng.random(n) > repeat_p)
            hist[t] = act
            _, _, done, trunc, _ = core.step(act)
            traj[t + 1] = core.get_states()
            alive &= ~((done > 0) | (trunc > 0))
            new = alive & (first < 0) & in_target(traj[t + 1], box, onground, feet_min)
            first[new] = t + 1
            if (first >= 0).sum() >= k - len(hits):
                break
        bursts += n
        for j in np.where(first >= 0)[0][: k - len(hits)]:
            hits.append((traj[: first[j] + 1, j].copy(), hist[: first[j] + 1, j].copy()))
    return hits, bursts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-spawn", action="store_true",
                    help="the root is the map's true spawn (the zones file's spawn)")
    ap.add_argument("--root-archive", default=None,
                    help="ARCHIVE.npz:INDEX - an explore_phase1 archive row as the root")
    ap.add_argument("--stage-roots", default=None,
                    help="a previous entry_search output: the last state of each line is a root")
    ap.add_argument("--target", required=True, help="xmin xmax ymin ymax zmin zmax")
    ap.add_argument("--onground", action="store_true")
    ap.add_argument("--feet-min", type=float, default=None)
    ap.add_argument("--k", type=int, default=8, help="successes to keep per root")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--repeat-p", type=float, default=0.95)
    ap.add_argument("--maxvel", type=float, default=3500.0)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    box = [float(v) for v in a.target.split()]
    cfg = default_config(num_envs=a.envs, sv_maxvelocity=float(a.maxvel), lidar_w=0, lidar_h=0,
                         spawn_mode=2, water_fail=1, yaw_jitter_deg=0.0)
    core = SurfCore(str(a.map), cfg)
    core.set_teleport_fail(True)
    core.reset(0)
    roots = []
    if a.root_spawn:
        from surfgym.rewards import map_spawn_pool
        p = map_spawn_pool(core)
        roots.append(np.asarray(p[0]).copy())
    if a.root_archive:
        f, i = a.root_archive.rsplit(":", 1)
        st = np.load(f, allow_pickle=True)["state"]
        r = np.zeros(1, STATE_DTYPE)
        for name in STATE_DTYPE.names:
            r[name] = st[int(i)][name]
        roots.append(r[0])
    prev = None
    if a.stage_roots:
        prev = np.load(a.stage_roots, allow_pickle=True)
        for line in prev["lines"]:
            roots.append(line[-1].copy())
    if not roots:
        raise SystemExit("no root: --root-spawn, --root-archive or --stage-roots")
    rng = np.random.default_rng(a.seed)
    deadline = time.time() + 60.0 * a.minutes
    lines, acts, parent, tried = [], [], [], 0
    for ri, root in enumerate(roots):
        per_root = time.time() + (deadline - time.time()) / max(1, len(roots) - ri)
        hits, n = search(core, root, a.envs, a.ticks, a.act_every, a.repeat_p, box, a.onground,
                         a.feet_min, a.k, per_root, rng)
        tried += n
        for s, h in hits:
            lines.append(s)
            acts.append(h)
            parent.append(ri)
        print(f"root {ri} {np.asarray(root['origin']).round(0).tolist()}: {len(hits)} hits in "
              f"{n:,} bursts", flush=True)
    meta = {"tool": "entry_search", "map": Path(a.map).stem, "target": box,
            "onground": bool(a.onground), "feet_min": a.feet_min, "envs": a.envs,
            "ticks": a.ticks, "act_every": a.act_every, "repeat_p": a.repeat_p,
            "maxvel": a.maxvel, "seed": a.seed, "roots": len(roots), "bursts": tried,
            "hits": len(lines), "stage_roots": a.stage_roots, "root_archive": a.root_archive,
            "root_spawn": bool(a.root_spawn),
            "provenance": "machine-generated: uniform random bursts from the given roots in the "
                          "simulator; no demo, no record, no policy, no reward"}
    obj = np.empty(len(lines), dtype=object)
    obj[:] = lines
    aobj = np.empty(len(acts), dtype=object)
    aobj[:] = acts
    np.savez(a.out, lines=obj, acts=aobj, parent=np.asarray(parent, np.int64),
             meta=json.dumps(meta))
    print(f"{len(lines)} lines -> {a.out} ({tried:,} bursts)")
    return 0 if lines else 1


if __name__ == "__main__":
    raise SystemExit(main())
