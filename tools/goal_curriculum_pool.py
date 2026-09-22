"""goal_curriculum_pool.py - a goal-rooted archive -> a reverse-curriculum start pool.

Florensa, Held, Wulfmeier, Zhang, Abbeel, "Reverse Curriculum Generation for
Reinforcement Learning" (CoRL 2017, arXiv 1707.05300) start training at the
goal and move the start distribution backward as the policy succeeds, with
new starts produced by random-action ("Brownian") walks from the goal. Here
the walks are ``tools/explore_phase1.py --roots-goal`` (random macro-action
bursts from standing states inside the finish box, Go-Explore archive, one
state per cell), and this tool writes that archive as the STATE_DTYPE spine
``train_fast.py --demo-file`` reads (surfgym/respawn.py DemoCurriculum):

    row 0        the state FARTHEST from the goal (largest depth)
    ...
    last row     the state NEAREST the goal, outside the finish box - the
                 curriculum's tau starts here ("just before the goal")

``depth`` is physics ticks FROM THE GOAL along the archive's provenance
chain (a root in the finish box has depth 0), so descending depth is the
curriculum's backward order. The pool is a TREE around the goal, not one
route: every cell the backward search reached is a row, dead ends included,
ordered only by how far from the goal the search found it.

What is dropped, and why:
  * states whose origin lies inside the HULL-INFLATED finish box
    (src/env.c: origin in [box.mins - hull.maxs, box.maxs - hull.mins], the
    ducked hull when ducked). The core's swept-segment test completes such
    an episode on its first tick, a +50 no action earns; ``--keep-goal-states``
    keeps them;
  * states whose origin repeats at 0.1 u (DemoCurriculum re-identifies a
    realized spawn by its rounded origin, so a duplicate could never be
    credited); the goal-nearest one is kept.

Velocity (``--velocity``):
  zero     (default) the walker standing still: velocity and basevelocity
           zeroed. The time-reversal-free choice: nothing about the search's
           own direction of travel (AWAY from the goal) is carried over.
  keep     the archived velocity as the search had it (moving away from the
           goal).
  reverse  velocity negated and yaw turned 180 deg (basevelocity zeroed): the
           time reverse of the search's motion, i.e. moving TOWARD the goal.
           Only approximately a forward-reachable state - friction, air
           acceleration and clip-velocity are not time-reversible.
Every row gets tick = 0, stuck_ticks = 0, progress = best_progress = 0,
seg_hint = 0 (the core re-zeroes tick / stuck on reset anyway).

Provenance (``<out>.json`` next to the .npy): the command line, the git hash,
the archive and its meta (map, cell, seed, minutes, roots), the row counts,
the depth range, whether the backward search reached a map spawn (the
closest exported origin to a spawn, within one archive cell = reached) and,
for analysis only, the rank correlation between depth and the archive's
goal-distance column (the cached geodesic field when one sat next to the
map, else euclid) - it is printed, never used.

    python tools/goal_curriculum_pool.py runs/rc_search/labyrinth_left100/archive.npz \\
        --out runs/rc_pools/labyrinth_left100_goalpool.npy

No demo, no policy, no reward: the rows are machine search states rooted in
the task's own finish box.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                             # noqa: E402

# src/bsp.c g_player_mins / g_player_maxs: usehull 0 standing, 1 ducked
HULL_MINS = np.array([[-16.0, -16.0, -36.0], [-16.0, -16.0, -18.0]])
HULL_MAXS = np.array([[16.0, 16.0, 36.0], [16.0, 16.0, 18.0]])
VELOCITY_MODES = ("zero", "keep", "reverse")


def in_goal(states: np.ndarray, box) -> np.ndarray:
    """True where the core would complete the episode on its FIRST tick: the
    origin inside the finish box inflated by the player's hull (the ducked
    hull for a ducked state), exactly env.c's test with prev == cur."""
    o = np.asarray(states["origin"], np.float64).reshape(-1, 3)
    uh = (np.asarray(states["ducked"]).reshape(-1) != 0).astype(np.int64)
    lo = np.asarray(box["mins"], np.float64)[None, :] - HULL_MAXS[uh]
    hi = np.asarray(box["maxs"], np.float64)[None, :] - HULL_MINS[uh]
    return np.all((o >= lo) & (o <= hi), axis=1)


def rank_corr(a, b) -> float:
    """Spearman rank correlation (average ranks for ties), numpy only."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) < 3:
        return float("nan")

    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average the ranks of ties
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def build_pool(state, depth, key, box=None, velocity: str = "zero",
               keep_goal_states: bool = False):
    """The archive columns -> (pool STATE_DTYPE ordered by DESCENDING depth,
    source archive row of each pool row, counts dict).

    Order: ascending depth (ties by cell key), duplicates at 0.1 u dropped
    keeping the goal-nearest, then reversed - row 0 is the farthest from the
    goal, the last row the nearest."""
    if velocity not in VELOCITY_MODES:
        raise ValueError(f"velocity must be one of {VELOCITY_MODES}, got {velocity!r}")
    state = np.asarray(state, STATE_DTYPE)
    depth = np.asarray(depth, np.int64)
    key = np.asarray(key, np.int64)
    n = len(state)
    keep = np.ones(n, bool)
    n_goal = 0
    if box is not None and not keep_goal_states:
        g = in_goal(state, box)
        n_goal = int(g.sum())
        keep &= ~g
    idx = np.flatnonzero(keep)
    idx = idx[np.lexsort((key[idx], depth[idx]))]          # goal-nearest first
    seen: set = set()
    uniq = []
    for i in idx.tolist():
        k = tuple(np.round(np.asarray(state[i]["origin"], np.float64), 1))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(i)
    src = np.asarray(uniq[::-1], np.int64)                  # farthest first
    pool = state[src].copy()
    if velocity == "zero":
        pool["velocity"] = 0.0
        pool["basevelocity"] = 0.0
    elif velocity == "reverse":
        pool["velocity"] = -pool["velocity"]
        pool["basevelocity"] = 0.0
        pool["yaw"] = np.mod(pool["yaw"].astype(np.float64) + 180.0,
                             360.0).astype(np.float32)
    pool["tick"] = 0
    pool["stuck_ticks"] = 0
    pool["progress"] = 0.0
    pool["best_progress"] = 0.0
    pool["seg_hint"] = 0
    counts = {"archive_cells": int(n), "excluded_in_goal": n_goal,
              "dedup_dropped": int(len(idx) - len(uniq)), "rows": int(len(pool))}
    return pool, src, counts


def _git_hash() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=str(ROOT), timeout=30)
        h = r.stdout.strip() if r.returncode == 0 else "unknown"
        d = subprocess.run(["git", "status", "--porcelain", "--", "tools",
                            "python"], capture_output=True, text=True,
                           cwd=str(ROOT), timeout=30)
        if d.returncode == 0 and d.stdout.strip():
            h += "+dirty"
        return h
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def export(archive: Path, out: Path, velocity: str = "zero",
           keep_goal_states: bool = False, argv=None) -> dict:
    archive = Path(archive)
    meta_p = Path(str(archive).replace(".npz", ".meta.json"))
    if not meta_p.exists():
        raise SystemExit(f"no {meta_p.name} next to {archive}: the archive's "
                         "provenance is part of the pool's")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    if meta.get("roots") != "goal":
        raise SystemExit(
            f"{archive} is not a goal-rooted archive (meta roots="
            f"{meta.get('roots')!r}): its depth counts from the MAP START, so "
            "descending depth would put the goal FIRST. Run "
            "explore_phase1.py --roots-goal")
    with np.load(archive, allow_pickle=False) as z:
        state, depth, key = z["state"], z["depth"], z["key"]
        dist = z["dist"] if "dist" in z.files else None
    box = meta.get("goal_box")
    pool, src, counts = build_pool(state, depth, key, box, velocity,
                                   keep_goal_states)
    if len(pool) == 0:
        raise SystemExit("no rows left to export (every archived state is in "
                         "the finish box?)")
    tick_ms = float(meta.get("tick_ms", 10))
    cell = float(meta.get("cell", 0.0))
    d_out = depth[src].astype(np.int64)
    spawns = np.asarray(meta.get("spawns") or [], np.float64).reshape(-1, 3)
    gap, row = float("inf"), -1
    if len(spawns):
        o = pool["origin"].astype(np.float64)
        dd = np.full(len(o), np.inf)
        for p in spawns:
            np.minimum(dd, np.linalg.norm(o - p, axis=1), out=dd)
        row = int(np.argmin(dd))
        gap = float(dd[row])
    rc = float("nan")
    if dist is not None:
        dv = np.asarray(dist, np.float64)[src]
        fin = np.isfinite(dv)
        rc = rank_corr(d_out[fin], dv[fin]) if fin.sum() >= 3 else float("nan")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pool)
    prov = {
        "tool": "goal_curriculum_pool",
        "method": "reverse curriculum (Florensa et al. 2017, arXiv 1707.05300) "
                  "start pool from a goal-rooted Go-Explore archive",
        "provenance": "machine-generated: explore_phase1 --roots-goal random "
                      "macro-action bursts from standing states inside the "
                      "map's own finish box; no human demo, no policy, no reward",
        "argv": list(argv if argv is not None else sys.argv),
        "git": _git_hash(),
        "archive": str(archive),
        "archive_meta": {k: meta.get(k) for k in (
            "map", "cell", "cell_view", "cell_speed", "envs", "decisions",
            "act_every", "repeat_p", "seed", "iters", "minutes", "max_minutes",
            "ep_ticks", "roots", "roots_goal_n", "roots_floor",
            "roots_fallback", "goal_box", "tick_ms", "depth_unit", "argv",
            "start_gap", "start_reached", "start_gap_depth", "depth_max",
            "dist_max")},
        "map": meta.get("map"),
        "out": str(out),
        "order": "DESCENDING depth: row 0 = farthest from the goal, last row = "
                 "nearest the goal (outside the finish box)",
        "velocity": velocity,
        "keep_goal_states": bool(keep_goal_states),
        **counts,
        "depth_ticks": [int(d_out.min()), int(d_out.max())],
        "depth_seconds": [round(float(d_out.min()) * tick_ms / 1000.0, 3),
                          round(float(d_out.max()) * tick_ms / 1000.0, 3)],
        "first_row_depth": int(d_out[0]),
        "last_row_depth": int(d_out[-1]),
        "start_gap": round(gap, 1) if np.isfinite(gap) else None,
        "start_reached": bool(np.isfinite(gap) and cell > 0 and gap <= cell),
        "start_row": row,
        "start_row_depth": int(d_out[row]) if row >= 0 else None,
        "start_row_depth_seconds": (round(float(d_out[row]) * tick_ms / 1000.0, 3)
                                    if row >= 0 else None),
        "rank_corr_depth_vs_goal_dist": (round(rc, 4) if rc == rc else None),
        "note": "depth is the summed tick cost of stitched archive hops, an "
                "upper bound on the search's own shortest time from the goal; "
                "the goal-distance correlation is analysis only",
    }
    js = out.with_suffix(".json")
    js.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return prov


def main() -> int:
    ap = argparse.ArgumentParser(
        description="goal-rooted explore_phase1 archive -> reverse-curriculum "
                    "start pool (STATE_DTYPE .npy, descending depth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("archive", help="archive.npz from explore_phase1 --roots-goal "
                                    "(its archive.meta.json must sit next to it)")
    ap.add_argument("--out", required=True, help="the pool .npy to write "
                                                 "(provenance goes to <out>.json)")
    ap.add_argument("--velocity", default="zero", choices=VELOCITY_MODES,
                    help="zero: standing still (default, time-reversal-free); "
                         "keep: the search's own velocity (away from the goal); "
                         "reverse: negated velocity and yaw + 180 (toward the goal)")
    ap.add_argument("--keep-goal-states", action="store_true",
                    help="keep states inside the hull-inflated finish box (they "
                         "complete on their first tick)")
    args = ap.parse_args()
    prov = export(Path(args.archive), Path(args.out), args.velocity,
                  args.keep_goal_states)
    print(f"{prov['map']}: {prov['rows']} rows (archive {prov['archive_cells']} cells, "
          f"{prov['excluded_in_goal']} in the finish box dropped, "
          f"{prov['dedup_dropped']} duplicate origins dropped) -> {prov['out']}")
    print(f"  depth {prov['depth_ticks'][0]}..{prov['depth_ticks'][1]} ticks = "
          f"{prov['depth_seconds'][0]:.2f}..{prov['depth_seconds'][1]:.2f} s from the "
          f"goal; row 0 depth {prov['first_row_depth']}, last row depth "
          f"{prov['last_row_depth']}; velocity {prov['velocity']}")
    print(f"  start {'REACHED' if prov['start_reached'] else 'NOT reached'}: the "
          f"closest row to a map spawn is row {prov['start_row']} at "
          f"{prov['start_gap']} u (depth {prov['start_row_depth']} ticks)"
          f" | rank corr(depth, goal distance) {prov['rank_corr_depth_vs_goal_dist']}"
          " (analysis only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
