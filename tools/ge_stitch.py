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
the last cell it did reach, and says so). --backtrack 16,32,64,128 first retries such a link from
that many ticks EARLIER on the line (the line is cut back to the root that worked): a cell is a
position and a speed, not a heading, so the line can arrive in the right cell too late to make the
next one - a surf launch, say, whose approach has to start before the last cell. --prefix resumes
from a previous run's line (with --prefix-links, the chain links it had reached)."""
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


def search_any(core, root, targets, cell, a, rng, deadline):
    """Uniform bursts (95% repeat per 4-tick decision) from ``root`` until one enters one of
    ``targets``' cell boxes with >= --speed-frac of its speed; the FURTHEST target any burst of
    a round entered wins. -> (target index, per-tick states, bursts) or None."""
    from surfgym.core import ACTION_NVEC
    n, ticks = a.envs, a.ticks
    boxes = []
    for t in targets:
        o = np.asarray(t["origin"], np.float64)
        v = np.asarray(t["velocity"], np.float64)
        boxes.append((o - cell / 2, o + cell / 2,
                      float(a.speed_frac) * float(np.sqrt((v * v).sum()))))
    root = root.copy()
    root["tick"] = 0
    root["stuck_ticks"] = 0
    act = np.zeros((n, 6), np.int32)

    def fresh(m):
        c = int(m.sum())
        if c:
            act[m, 0] = rng.integers(0, ACTION_NVEC[0], c)
            act[m, 1] = 3
            for h in (2, 3, 4, 5):
                act[m, h] = rng.integers(0, ACTION_NVEC[h], c)

    bursts = 0
    while time.time() < deadline:
        for j in range(n):
            core.set_state(j, root)
        fresh(np.ones(n, bool))
        traj = np.zeros((ticks + 1, n), STATE_DTYPE)
        traj[0] = core.get_states()
        alive = np.ones(n, bool)
        hit_t = np.full((len(boxes), n), -1)
        for t in range(ticks):
            if t % 4 == 0 and t:
                fresh(rng.random(n) > 0.95)
            _, _, done, trunc, _ = core.step(act)
            s = core.get_states()
            traj[t + 1] = s
            alive &= ~((done > 0) | (trunc > 0))
            o = s["origin"].astype(np.float64)
            sp = np.sqrt((s["velocity"].astype(np.float64) ** 2).sum(axis=1))
            for b, (lo, hi, smin) in enumerate(boxes):
                inb = alive & (hit_t[b] < 0) & np.all((o >= lo) & (o <= hi), axis=1) & (sp >= smin)
                hit_t[b][inb] = t + 1
            if (hit_t[-1] >= 0).any():
                break
        bursts += n
        for b in range(len(boxes) - 1, -1, -1):
            e = np.flatnonzero(hit_t[b] >= 0)
            if len(e):
                env = int(e[0])
                return b, traj[: hit_t[b][env] + 1, env].copy(), bursts
    return None


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
    ap.add_argument("--lookahead", type=int, default=4,
                    help="target any of the next N chain cells and jump to the furthest reached")
    ap.add_argument("--speed-frac", type=float, default=0.9,
                    help="a link must also arrive with at least this share of the archived "
                         "cell's speed (the archive's energy dimension; 0 = position only)")
    ap.add_argument("--backtrack", default="",
                    help="comma list of tick offsets: a link not re-found from the line's end is "
                         "retried from that many ticks earlier on the line, e.g. 16,32,64,128")
    ap.add_argument("--cell", type=int, default=None,
                    help="the chain's end cell (archive index); default the deepest on the field")
    ap.add_argument("--prefix", default=None,
                    help="resume from a previous ge_stitch line (.npy) of the same chain")
    ap.add_argument("--prefix-links", type=int, default=0,
                    help="with --prefix: the chain links that line had reached")
    a = ap.parse_args(argv)
    backtrack = [int(b) for b in a.backtrack.split(",") if b.strip()]
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
    best = int(np.argmin(d)) if a.cell is None else int(a.cell)
    chain = chain_to(parent, best)
    print(f"{'deepest' if a.cell is None else 'chosen'} cell {best}: d {d[best]:,.0f} at "
          f"{np.asarray(st[best]['origin']).round(0)}; chain of {len(chain)} cells")
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
    if a.prefix:
        line = [r.copy() for r in np.load(a.prefix, allow_pickle=False)]
        reached = int(a.prefix_links)
        print(f"resuming from {a.prefix}: {len(line)} states, link {reached}")
    while reached < len(chain) - 1:
        # the next --lookahead chain cells are all targets; the burst that enters the FURTHEST
        # one wins (a link the archive reached through a later-replaced parent state is skipped)
        ks = list(range(reached + 1, min(len(chain), reached + 1 + int(a.lookahead))))
        t0 = time.time()
        got, back = None, 0
        for back in [0] + backtrack:
            if back >= len(line):
                break
            got = search_any(core, line[-1 - back], [st[chain[k]] for k in ks], cell, a, rng,
                             time.time() + 60.0 * a.minutes_per_link)
            if got is not None:
                break
            if backtrack:
                print(f"  links {ks[0]}..{ks[-1]} not re-found {back} ticks back", flush=True)
        if got is None:
            print(f"links {ks[0]}..{ks[-1]} of {len(chain) - 1} not re-found "
                  f"({time.time() - t0:.0f}s): the line stops at link {reached}")
            break
        j, seg, n = got
        if back:
            del line[-back:]
        line.extend(seg[1:])
        reached = ks[j]
        print(f"link {reached}/{len(chain) - 1}: {len(seg) - 1} ticks after {n:,} bursts"
              f"{f' ({back} ticks back)' if back else ''}, now at "
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
