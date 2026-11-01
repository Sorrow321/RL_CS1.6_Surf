#!/usr/bin/env python3
"""macro_finish.py - physics-only random-shooting search for a goal entry from the policy's OWN states.

Roots: a 1-D STATE_DTYPE pool of the policy's own states (tools/frontier_states.py on its greedy
re-recordings), every --stride-th row. Proposals: sequences of --segs held MACROS in the policy's
own action frame (--view-absolute velocity): a forward key (back / none / forward), a side key
(left / none / right), duck, jump, and a yaw TARGET offset from the horizontal velocity heading
(the core turns the view toward heading + offset at its per-tick ceiling, exactly as it does for
the policy's own commands), each macro held for a log-uniform --hold MIN..MAX seconds. Every env
runs one (root, plan) for --secs seconds, all in lockstep on one batched core (no policy, no
vision: the C physics only). A goal entry is recorded with its tick and REPLAYED on a fresh 1-env
core from the same root; it must finish on the same tick, and its per-tick states (root ->
pre-finish) are saved as <out>/fin_<k>.npz in the beam_frontier.npz layout tools/finish_pool.py
reads (states, finished, replay_ok).

--rounds > 1 adds a cross-entropy step: the plans with the lowest goal distance reached (the
grader's field) are kept unchanged and perturbed (offsets +- --jit deg, holds x U(0.85, 1.2), a
key redrawn with p 0.05) for the next round, from the same roots. Nothing here reads the map beyond the goal
field and the finish triggers every search already uses; every constant is a physics or search
scale, set once.

    python tools/macro_finish.py CKPT POOL.npy --out runs/mf/a --stride 25 --plans 256 --secs 2.5
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "autoresearch"))

# a plan: (S, 7) float rows = (fwd 0..2, side 0..2, duck 0/1, jump 0/1, yaw offset deg,
# pitch deg, end tick (exclusive, cumulative))
F_FWD, F_SIDE, F_DUCK, F_JUMP, F_YAW, F_PITCH, F_END = range(7)


def draw_plans(rng, n: int, segs: int, hold: tuple[float, float], tick_s: float,
               t_max: int, p_duck: float, p_jump: float, p_small: float,
               small_sd: float, wide: float) -> np.ndarray:
    """(n, segs, 7) random plans; the last segment runs to t_max."""
    p = np.zeros((n, segs, 7), np.float64)
    p[..., F_FWD] = rng.integers(0, 3, (n, segs))
    p[..., F_SIDE] = rng.integers(0, 3, (n, segs))
    p[..., F_DUCK] = rng.random((n, segs)) < p_duck
    p[..., F_JUMP] = rng.random((n, segs)) < p_jump
    small = rng.random((n, segs)) < p_small
    p[..., F_YAW] = np.where(small, rng.normal(0.0, small_sd, (n, segs)),
                             rng.uniform(-wide, wide, (n, segs)))
    lo, hi = np.log(hold[0]), np.log(hold[1])
    dur = np.exp(rng.uniform(lo, hi, (n, segs))) / tick_s
    end = np.minimum(np.cumsum(np.maximum(1.0, np.round(dur)), axis=1), t_max)
    end[:, -1] = t_max
    p[..., F_END] = end
    return p


def perturb(rng, plans: np.ndarray, jit: float, t_max: int, p_key: float = 0.05,
            dur_jit: tuple[float, float] = (0.85, 1.2)) -> np.ndarray:
    q = plans.copy()
    n, s, _ = q.shape
    q[..., F_YAW] += rng.normal(0.0, jit, (n, s))
    for f, k in ((F_FWD, 3), (F_SIDE, 3)):
        m = rng.random((n, s)) < p_key
        q[..., f] = np.where(m, rng.integers(0, k, (n, s)), q[..., f])
    dur = np.diff(np.concatenate([np.zeros((n, 1)), q[..., F_END]], axis=1), axis=1)
    dur = np.maximum(1.0, np.round(dur * rng.uniform(dur_jit[0], dur_jit[1], (n, s))))
    end = np.minimum(np.cumsum(dur, axis=1), t_max)
    end[:, -1] = t_max
    q[..., F_END] = end
    return q


def plan_rows(plans: np.ndarray, t: int):
    """(acts (n, 6) int32, view (n, 2) float32) of every plan at tick t."""
    n = len(plans)
    k = (plans[..., F_END] <= t).sum(axis=1)
    k = np.minimum(k, plans.shape[1] - 1)
    row = plans[np.arange(n), k]
    act = np.zeros((n, 6), np.int32)
    act[:, 0] = 7
    act[:, 1] = 3
    act[:, 2] = row[:, F_FWD]
    act[:, 3] = row[:, F_SIDE]
    act[:, 4] = row[:, F_JUMP]
    act[:, 5] = row[:, F_DUCK]
    view = np.stack([row[:, F_YAW], row[:, F_PITCH]], axis=1).astype(np.float32)
    return act, view


# the player hulls (src/bsp.c g_player_mins / g_player_maxs): standing, ducked
HULL_MINS = np.array([[-16.0, -16.0, -36.0], [-16.0, -16.0, -18.0]])
HULL_MAXS = np.array([[16.0, 16.0, 36.0], [16.0, 16.0, 18.0]])


def box_dist(o: np.ndarray, ducked: np.ndarray, boxes) -> np.ndarray:
    """Euclidean distance from each origin to the nearest goal box INFLATED by its hull - the
    core's own goal test region (origin in [gmin - maxs, gmax - mins]); 0 inside."""
    h = (np.asarray(ducked) != 0).astype(np.int64)
    best = np.full(len(o), np.inf)
    for mn, mx in boxes:
        lo = np.asarray(mn, np.float64)[None, :] - HULL_MAXS[h]
        hi = np.asarray(mx, np.float64)[None, :] - HULL_MINS[h]
        gap = np.maximum(np.maximum(lo - o, 0.0), np.maximum(o - hi, 0.0))
        best = np.minimum(best, np.sqrt((gap * gap).sum(axis=1)))
    return best


def reach_points(core, boxes, hull: int, step_xy: float = 16.0, step_z: float = 8.0):
    """Origins inside the hull-inflated goal boxes where that hull FITS (a zero-length hull
    trace that does not start in solid): the part of the goal test region a player can occupy.
    A trigger lying on a solid top is only enterable from above it; touching the face below it
    leaves the origin DIST_EPSILON outside the inflated box, so the boxes alone overstate it."""
    pts = []
    for mn, mx in boxes:
        lo = np.asarray(mn, np.float64) - HULL_MAXS[hull]
        hi = np.asarray(mx, np.float64) - HULL_MINS[hull]
        xs = np.arange(lo[0] + 0.5, hi[0], step_xy)
        ys = np.arange(lo[1] + 0.5, hi[1], step_xy)
        zs = np.arange(lo[2] + 0.5, hi[2], step_z)
        for x in xs:
            for y in ys:
                for z in zs:
                    if not core.trace((x, y, z), (x, y, z), hull).startsolid:
                        pts.append((x, y, z))
    return np.asarray(pts, np.float64).reshape(-1, 3)


class Reach:
    """Distance to the reachable goal region (reach_points), per hull, through a KD-tree."""

    def __init__(self, core, boxes, cache: Path | None = None):
        from scipy.spatial import cKDTree
        self.trees = []
        for h in (0, 1):
            f = None if cache is None else cache.with_name(cache.stem + f"_h{h}.npy")
            if f is not None and f.exists():
                pts = np.load(f)
            else:
                pts = reach_points(core, boxes, h)
                if f is not None:
                    np.save(f, pts)
            self.trees.append(cKDTree(pts) if len(pts) else None)
            print(f"reach: hull {h}: {len(pts):,} free goal origins"
                  + (f", z {pts[:, 2].min():.0f}..{pts[:, 2].max():.0f}" if len(pts) else ""),
                  flush=True)

    def dist(self, o, ducked, near=None, cap: float = 600.0):
        """Distance to the nearest free goal origin of each state's hull; only the rows in
        ``near`` are queried (a KD query far from every point costs ~0.1 ms a row) and a row
        with nothing within ``cap`` reads inf - callers keep max(field, box) there."""
        out = np.full(len(o), np.inf)
        h = (np.asarray(ducked) != 0).astype(np.int64)
        for k in (0, 1):
            m = h == k
            if near is not None:
                m &= near
            if m.any() and self.trees[k] is not None:
                out[m] = self.trees[k].query(o[m], distance_upper_bound=cap)[0]
        return out


def goal_score(o, ducked, gf, boxes, reach=None):
    """The search's distance: the goal field (geodesic, voxel cells), but never below the exact
    distance to the hull-inflated goal boxes, which the field's cells cannot resolve within a
    cell of a slab (a field d of 0 can still be 13 u short of the trigger), nor below the
    distance to the part of those boxes the hull fits in (Reach)."""
    d = gf.sample(o).astype(np.float64)
    if boxes:
        db = box_dist(o, ducked, boxes)
        d = np.maximum(d, db)
        if reach is not None:
            near = db < 300.0
            if near.any():
                dr = reach.dist(o, ducked, near=near)
                d = np.where(near & np.isfinite(dr), np.maximum(d, dr), d)
    return d


def rollout(core, roots_states, root_idx, plans, t_max, gf, d_every: int = 4, boxes=(),
            reach=None):
    """Run every env's (root, plan) for t_max ticks -> (fin_tick (n,) 0 = none, min_d (n,),
    end_E (n,), alive (n,) bool at the end)."""
    n = len(plans)
    core.reset(0)
    for e in range(n):
        core.set_state(e, roots_states[root_idx[e]:root_idx[e] + 1])
    alive = np.ones(n, bool)
    fin = np.zeros(n, np.int32)
    min_d = np.full(n, np.inf)
    sv = core.states_view
    E = np.zeros(n)
    for t in range(t_max):
        act, view = plan_rows(plans, t)
        _, _, done, trunc, _ = core.step(act, view=view)
        gh = np.asarray(core.goal_hits, bool) & alive
        if gh.any():
            fin[gh] = t + 1
        ended = ((np.asarray(done) | np.asarray(trunc)) != 0) & alive
        alive &= ~(ended | gh)
        if t % d_every == 0 or t == t_max - 1:
            o = np.asarray(sv["origin"], np.float64)
            d = goal_score(o, sv["ducked"], gf, boxes, reach)
            min_d = np.where(alive, np.minimum(min_d, d), min_d)
            v = np.asarray(sv["velocity"], np.float64)
            E = np.where(alive, o[:, 2] + (v * v).sum(1) / 1600.0, E)
    min_d[fin > 0] = 0.0
    return fin, min_d, E, alive


def replay(build, root_state, plan, fin_tick):
    """1-env exact replay -> (states up to the pre-finish state, finished on fin_tick)."""
    core = build(1)
    core.reset(0)
    core.set_state(0, root_state)
    states = []
    ok = False
    for t in range(int(fin_tick)):
        states.append(core.get_states()[0].copy())
        act, view = plan_rows(plan[None], t)
        _, _, done, trunc, _ = core.step(act, view=view)
        if bool(np.asarray(core.goal_hits, bool)[0]):
            ok = (t + 1) == int(fin_tick)
            break
        if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
            break
    return np.array(states), ok


def describe_best(build, pool, root, plan, t_max, gf, tick_s, boxes=(), reach=None):
    """Replay one plan on a 1-env core and print where it came closest to the goal."""
    core = build(1)
    core.reset(0)
    core.set_state(0, pool[root:root + 1])
    best = (np.inf, None)
    for t in range(t_max):
        act, view = plan_rows(plan[None], t)
        _, _, done, trunc, _ = core.step(act, view=view)
        st = core.get_states()[0]
        o = np.asarray(st["origin"], np.float64)
        d = float(goal_score(o[None], np.array([st["ducked"]]), gf, boxes, reach)[0])
        if d < best[0]:
            best = (d, (t + 1, o.copy(), np.asarray(st["velocity"], np.float64).copy(),
                        int(st["ducked"])))
        if bool(np.asarray(core.goal_hits, bool)[0]) or bool(np.asarray(done)[0]) \
                or bool(np.asarray(trunc)[0]):
            break
    if best[1] is not None:
        t, o, v, du = best[1]
        r0 = pool["origin"][root]
        print(f"   best: root {root} ({r0[0]:.0f},{r0[1]:.0f},{r0[2]:.0f}) -> d {best[0]:.0f} at "
              f"{t * tick_s:.2f}s ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f}) v ({v[0]:.0f},{v[1]:.0f},"
              f"{v[2]:.0f}) ducked {du}; plan "
              f"{np.round(plan[:, [F_FWD, F_SIDE, F_DUCK, F_JUMP, F_YAW, F_END]], 1).tolist()}",
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt", help="the checkpoint whose config builds the core (physics parity)")
    ap.add_argument("pool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--plans", type=int, default=256, help="plans per root per round")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--secs", type=float, default=2.5)
    ap.add_argument("--segs", type=int, default=3)
    ap.add_argument("--hold", type=float, nargs=2, default=(0.1, 1.0))
    ap.add_argument("--p-duck", type=float, default=0.1)
    ap.add_argument("--p-jump", type=float, default=0.05)
    ap.add_argument("--p-small", type=float, default=0.5)
    ap.add_argument("--small-sd", type=float, default=3.0)
    ap.add_argument("--wide", type=float, default=120.0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--elite", type=float, default=0.05)
    ap.add_argument("--jit", type=float, default=2.0)
    ap.add_argument("--elite-roots", type=int, default=0,
                    help="> 0: the elite is the best --elite-per plans of each of this many roots "
                         "(diversity across roots) instead of the global --elite share")
    ap.add_argument("--elite-per", type=int, default=16)
    ap.add_argument("--max-fin", type=int, default=20, help="stop after this many finishes")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reach", type=int, default=1,
                    help="1 = score also by the distance to the goal origins the hull fits in")
    ap.add_argument("--reach-cache", default=None,
                    help="npy path stem for the reach point clouds (built once, ~20 s per hull)")
    a = ap.parse_args(argv)

    import torch
    import beam_tas as BT
    from surfgym.tick import TickClock
    import score as S

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tick_ms = float(cfg.get("tick_ms") or 10.0)
    tick = TickClock(tick_ms)
    tick_s = tick_ms / 1000.0
    if int(cfg.get("view_continuous") or 0) != 1 or cfg.get("view_absolute") != "velocity":
        raise SystemExit("macro_finish: the plans are velocity-frame view targets; the "
                         "checkpoint must be --view-continuous --view-absolute velocity")
    map_path = str(ROOT / "maps" / f"{cfg['map']}.bsp")
    from surfgym.zones import load_zones
    zones = load_zones(map_path)

    def build(n):
        core = BT.build_sim(cfg, map_path, n, int(cfg.get("ep_ticks") or 130000), tick=tick)
        core.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        core.set_teleport_fail(True)
        return core

    gf = S.field()
    # the goal boxes the core tests (zones.json "end": its "boxes", else the union box)
    _end = zones["end"]
    boxes = [(b["mins"], b["maxs"]) for b in (_end.get("boxes") or [_end])]
    reach = None
    if a.reach:
        _c = build(1)
        reach = Reach(_c, boxes, cache=Path(a.reach_cache) if a.reach_cache else None)
    pool = np.load(a.pool, allow_pickle=False).copy()
    pool["tick"] = 0
    pool["stuck_ticks"] = 0
    roots = np.arange(0, len(pool), max(1, a.stride))
    t_max = int(round(a.secs / tick_s))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    core = build(a.envs)
    print(f"macro_finish: {len(roots)} roots (stride {a.stride} of {len(pool):,}), {a.plans} "
          f"plans each, {a.segs} macros x hold {a.hold[0]:g}-{a.hold[1]:g}s, {a.secs:g}s "
          f"({t_max} ticks at {tick_ms:g} ms), {a.envs} envs/batch, {a.rounds} round(s)",
          flush=True)
    t_start = time.time()
    n_fin = 0
    best_by_root = np.full(len(roots), np.inf)
    elite_plans, elite_roots = None, None
    log = []
    for rnd in range(a.rounds):
        if rnd == 0:
            r_idx = np.repeat(roots, a.plans)
            plans = draw_plans(rng, len(r_idx), a.segs, tuple(a.hold), tick_s, t_max,
                               a.p_duck, a.p_jump, a.p_small, a.small_sd, a.wide)
        else:
            reps = max(1, int(round(len(roots) * a.plans / max(1, len(elite_plans)))))
            # the elite itself rides along unperturbed (a CEM round never loses its best plan)
            r_idx = np.concatenate([elite_roots, np.repeat(elite_roots, reps - 1)])
            plans = np.concatenate([elite_plans, perturb(rng, np.repeat(elite_plans, reps - 1, axis=0),
                                                         a.jit, t_max)])
        order = rng.permutation(len(r_idx))
        r_idx, plans = r_idx[order], plans[order]
        all_d = np.full(len(r_idx), np.inf)
        for b0 in range(0, len(r_idx), a.envs):
            b1 = min(len(r_idx), b0 + a.envs)
            n = b1 - b0
            if n < a.envs:     # pad the last batch with copies (ignored)
                pr = np.concatenate([plans[b0:b1], plans[b0:b0 + 1].repeat(a.envs - n, 0)])
                rr = np.concatenate([r_idx[b0:b1], r_idx[b0:b0 + 1].repeat(a.envs - n)])
            else:
                pr, rr = plans[b0:b1], r_idx[b0:b1]
            fin, min_d, E, alive = rollout(core, pool, rr, pr, t_max, gf, boxes=boxes,
                                           reach=reach)
            all_d[b0:b1] = min_d[:n]
            for e in np.nonzero(fin[:n] > 0)[0]:
                st, ok = replay(build, pool[rr[e]:rr[e] + 1], pr[e], int(fin[e]))
                n_fin += 1
                o = pool["origin"][rr[e]]
                print(f"FINISH #{n_fin}: root {int(rr[e])} ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f}) "
                      f"tick {int(fin[e])} ({fin[e] * tick_s:.2f}s) replay {'ok' if ok else 'DIVERGED'} "
                      f"plan {np.round(pr[e][:, [F_FWD, F_SIDE, F_DUCK, F_JUMP, F_YAW, F_END]], 1).tolist()}",
                      flush=True)
                np.savez(out / f"fin_{n_fin:03d}.npz", states=st, finished=np.int32(1),
                         replay_ok=np.int32(1 if ok else 0), root=np.int64(rr[e]),
                         plan=pr[e], fin_tick=np.int32(fin[e]), tick_ms=np.float64(tick_ms),
                         pool=np.str_(str(a.pool)), ckpt=np.str_(str(a.ckpt)))
                if n_fin >= a.max_fin:
                    break
            if n_fin >= a.max_fin:
                break
            if a.max_minutes > 0 and time.time() - t_start > 60 * a.max_minutes:
                print("--max-minutes reached", flush=True)
                break
        # per-root best and the round's elite
        rpos = np.searchsorted(roots, r_idx)
        np.minimum.at(best_by_root, rpos, all_d)
        k = max(1, int(round(a.elite * len(r_idx))))
        if a.elite_roots > 0:
            # per-root elite: the best --elite-per plans of each of the --elite-roots roots whose
            # best plan this round came closest, so one root cannot take the whole next round
            rbest = {}
            for j in np.argsort(all_d):
                r = int(r_idx[j])
                if r not in rbest:
                    if len(rbest) >= a.elite_roots:
                        continue
                    rbest[r] = []
                if len(rbest[r]) < a.elite_per:
                    rbest[r].append(int(j))
            top = np.array([j for v in rbest.values() for j in v], np.int64)
            top = top[np.argsort(all_d[top])]
        else:
            top = np.argsort(all_d)[:k]
        elite_plans, elite_roots = plans[top], r_idx[top]
        fin_d = all_d[np.isfinite(all_d)]
        msg = {"round": rnd, "rollouts": int(len(r_idx)), "finishes_total": n_fin,
               "min_d": float(fin_d.min()) if len(fin_d) else None,
               "p1_d": float(np.percentile(fin_d, 1)) if len(fin_d) else None,
               "elite_max_d": float(all_d[top].max()), "wall_s": round(time.time() - t_start, 1)}
        log.append(msg)
        print(json.dumps(msg), flush=True)
        best = np.argsort(best_by_root)[:8]
        for j in best:
            o = pool["origin"][roots[j]]
            print(f"   root {int(roots[j])} ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f}) best d "
                  f"{best_by_root[j]:.0f}", flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        jb = int(top[0])
        describe_best(build, pool, int(r_idx[jb]), plans[jb], t_max, gf, tick_s, boxes, reach)
        np.savez(out / f"best_round{rnd}.npz", root=np.int64(r_idx[jb]), plan=plans[jb],
                 min_d=np.float64(all_d[jb]))
        if n_fin >= a.max_fin or (a.max_minutes > 0 and time.time() - t_start > 60 * a.max_minutes):
            break
    np.save(out / "best_by_root.npy", np.stack([roots.astype(np.float64), best_by_root]))
    print(f"macro_finish: {n_fin} finish(es) in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
