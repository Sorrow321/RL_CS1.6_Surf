"""edge_archive.py - Go-Explore over the learned executor's plans (Codex's v1 spec, agent bus
2026-09-26 21:37; the user's pivot after the 3-choice experiment).

    python tools/edge_archive.py <ckpt> --out runs/research/archive_<name> [--minutes 60]
        [--parents 64] [--seed 0] [--map maps_pool/<map>.bsp]

WHY. Under the planner's refund_i reward every failed episode nets 0, so nothing is learned before
the first finish, and a policy's progress into a detour is not KEPT anywhere (the reservoir keeps
states, never the chain that reached them; SIL keeps only finished episodes). An archive keeps
every new kind of state the agent reaches, and how it got there, whether or not that episode later
dies - progress is kept because it is NEW, not because it is rewarded. No reward, no potential, no
map constant enters the archive.

WHAT. A NODE is an exact simulator state (STATE_DTYPE row + the executor wrapper's held keys + the
core observation), reached from its parent by one PLAN - one of the checkpoint's --plan-choices
rays (forward / left / right) - flown by the checkpoint's own executor, SAMPLING its actions (its
native temperature), for the plan's duration (--plan-close commit: the checkpoint's own period),
closing at the executor's next decision tick like the trainer. A node's KEY (Codex's v1):
  * position: 128 u XYZ cell;
  * horizontal speed: < 128, 128-256, 256-512, 512-1024, 1024-2048, >= 2048 u/s;
  * horizontal velocity azimuth: 8 bins when moving (>= 128 u/s), else one 'stationary' bin;
  * vertical velocity: down / level / up with a 128 u/s dead band;
  * contact: onground != -1.
Held keys, view, yaw are stored in the node, NOT in the key. Two elites per key: the FIRST arrival
(fewest ticks from the root) and the FASTEST (highest speed); a new arrival replaces one only if it
is strictly better on that axis.

LOOP. Pick --parents nodes with weight 1 / sqrt(1 + times selected) (count-only: reward-free,
potential-free), fly ALL C choices from each in one batch on a scratch core (the recorder's own
construction, record_ckpt.build --plan-scratch), admit every live endpoint, repeat. A restored
node's episode clock is zeroed (a deep node must not inherit the cap; the node's own tick count
from the root is kept separately). Stops at the first chain that reaches the finish, --minutes, or
--expansions.

OUTPUT (<out>/): progress.jsonl (one line per report: expansions, nodes, keys, position cells,
admissions / duplicates / replacements, the best distance to the finish, the deepest node, novel-
child yield per choice, simulator steps / s), chain.json (the first finishing chain: every node's
origin, the choice that reached it, its tick from the root, and the flown path points), and a
replay of that chain's choices from the true start, without restores, 32 times (the executor
samples): how often the same plan sequence finishes - a chain that exists only as individually
restored edges is not yet a policy trajectory (Codex).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

POS_CELL = 128.0
HSPD_EDGES = np.array([128.0, 256.0, 512.0, 1024.0, 2048.0])
N_AZ = 8
VZ_DEAD = 128.0
REPLAYS = 32


def keys_of(st, mins) -> list:
    """The archive key of each STATE_DTYPE row (Codex's v1 quantiser)."""
    o = np.asarray(st["origin"], np.float64)
    v = np.asarray(st["velocity"], np.float64)
    cell = np.floor((o - mins[None, :]) / POS_CELL).astype(np.int64)
    vh = np.hypot(v[:, 0], v[:, 1])
    sb = np.searchsorted(HSPD_EDGES, vh, side="right")
    az = np.where(vh >= HSPD_EDGES[0],
                  np.floor((np.arctan2(v[:, 1], v[:, 0]) + math.pi) / (2.0 * math.pi) * N_AZ)
                  .astype(np.int64) % N_AZ, N_AZ)
    vz = np.where(v[:, 2] > VZ_DEAD, 2, np.where(v[:, 2] < -VZ_DEAD, 0, 1))
    g = (np.asarray(st["onground"]) != -1).astype(np.int64)
    return [(int(cell[i, 0]), int(cell[i, 1]), int(cell[i, 2]), int(sb[i]), int(az[i]),
             int(vz[i]), int(g[i])) for i in range(len(o))]


class Archive:
    """Nodes (parallel lists) + the key index (key -> [first-arrival id, fastest id])."""

    def __init__(self):
        self.state, self.keys_state, self.obs, self.key = [], [], [], []
        self.parent, self.move, self.depth, self.t, self.speed = [], [], [], [], []
        self.path = []
        self.n_sel = []
        self.index = {}
        self.cells = set()
        self.admit_new = self.admit_dup = self.replaced = 0

    def __len__(self):
        return len(self.state)

    def add(self, st, ks, obs, key, parent, move, depth, t, path):
        nid = len(self.state)
        self.state.append(st.copy())
        self.keys_state.append(ks)
        self.obs.append(np.array(obs, np.float32, copy=True))
        self.key.append(key)
        self.parent.append(int(parent))
        self.move.append(int(move))
        self.depth.append(int(depth))
        self.t.append(int(t))
        self.speed.append(float(np.linalg.norm(np.asarray(st["velocity"], np.float64))))
        self.path.append(path)
        self.n_sel.append(0)
        return nid

    def admit(self, st, ks, obs, key, parent, move, depth, t, path) -> str:
        """-> 'new' | 'replaced' | 'dup'."""
        slot = self.index.get(key)
        sp = float(np.linalg.norm(np.asarray(st["velocity"], np.float64)))
        if slot is None:
            nid = self.add(st, ks, obs, key, parent, move, depth, t, path)
            self.index[key] = [nid, nid]
            self.cells.add(key[:3])
            self.admit_new += 1
            return "new"
        first, fast = slot
        out = "dup"
        if t < self.t[first]:
            slot[0] = self.add(st, ks, obs, key, parent, move, depth, t, path)
            out = "replaced"
        if sp > self.speed[fast]:
            slot[1] = self.add(st, ks, obs, key, parent, move, depth, t, path)
            out = "replaced"
        if out == "replaced":
            self.replaced += 1
        else:
            self.admit_dup += 1
        return out

    def live_ids(self):
        """The nodes the index currently holds (an elite that was replaced is retired)."""
        return sorted({i for s in self.index.values() for i in s})

    def select(self, n, rng):
        ids = np.asarray(self.live_ids(), np.int64)
        w = 1.0 / np.sqrt(1.0 + np.asarray([self.n_sel[i] for i in ids], np.float64))
        pick = rng.choice(ids, size=min(n, len(ids)), replace=len(ids) < n, p=w / w.sum())
        for i in pick:
            self.n_sel[int(i)] += 1
        return [int(i) for i in pick]

    def chain(self, nid):
        out = []
        while nid >= 0:
            out.append(nid)
            nid = self.parent[nid]
        return out[::-1]


class Flyer:
    """Flies batches of (node, choice) on the recorder's scratch core with the checkpoint's own
    executor wrapper, the way goalsearch.PrimMCTS._expand does, but one DIFFERENT node per slot
    group (every node is captured at an executor decision tick, so all slots share the phase)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.P = ctx.planner
        self.sc = ctx.scratch
        self.core = self.sc.core
        self.S = int(self.core.num_envs)
        self.C = int(getattr(self.P, "n_choice", 0))
        if self.C <= 0:
            raise SystemExit("edge_archive: the checkpoint's planner has no --plan-choices (v1 "
                             "flies the fixed choices)")
        pol = self.sc.make_policy(self.core, self.sc.line)
        self.K = max(1, int(getattr(pol, "_k", 1)))
        self.keys_hold = bool(getattr(pol, "keys_hold", False))
        from surfgym.goalprimplan import L_MAX
        self.L_MAX = L_MAX
        self.dur = int(getattr(self.P, "commit_ticks", 0) or self.P.budget_ticks)
        self.steps = 0

    def fresh_keys(self):
        """The held-keys state of a fresh episode (what the wrapper starts from)."""
        if not self.keys_hold:
            return None
        from surfgym.keyshold import KeysHold
        k = KeysHold(1)
        return (k.state[0].copy(), k.boot[0].copy())

    def fly(self, jobs, arch, finish, record_path=True):
        """``jobs``: [(node id, choice)] (<= S). -> per job dict(end, obs, keys, died, fin,
        ticks, path)."""
        core, P, S, K = self.core, self.P, self.S, self.K
        n = len(jobs)
        assert 0 < n <= S
        st_all = np.empty(S, dtype=arch.state[jobs[0][0]].dtype)
        for i in range(S):
            nid = jobs[min(i, n - 1)][0]
            st = arch.state[nid].copy()
            st["tick"] = 0
            st["stuck_ticks"] = 0
            st_all[i] = st
            core.set_state(i, st)
        o = st_all["origin"].astype(np.float64)
        v = st_all["velocity"].astype(np.float64)
        y = st_all["yaw"].astype(np.float64)
        lines = []
        for i in range(S):
            k = jobs[min(i, n - 1)][1]
            ln, _pts = P.line_and_curve_of(o[i], v[i], float(y[i]), P.choice_nums[int(k)],
                                           int(k))
            lines.append(ln[:self.L_MAX])
        self.sc.line.set_lines(np.arange(S), lines)
        pol = self.sc.make_policy(core, self.sc.line)
        if self.keys_hold:
            from surfgym.keyshold import KeysHold
            pol.keys = KeysHold(S)
            for i in range(S):
                ks = arch.keys_state[jobs[min(i, n - 1)][0]]
                if ks is not None:
                    pol.keys.state[i] = ks[0]
                    pol.keys.boot[i] = ks[1]
            # the restored clocks are 0: the previous decision was one period earlier, so the
            # wrapper's episode-start detector does not collapse the held keys
            pol._keys_tick = np.full(S, -int(getattr(pol, "_period", K)), np.int64)
        pol._tick = 0
        obs = np.ascontiguousarray(np.stack([arch.obs[jobs[min(i, n - 1)][0]]
                                             for i in range(S)]).astype(np.float32))
        open_ = np.zeros(S, bool)
        open_[:n] = True
        died = np.zeros(S, bool)
        fnd = np.zeros(S, bool)
        ticks = np.zeros(S, np.int64)
        end = [None] * n
        end_obs = [None] * n
        end_keys = [None] * n
        paths = [[o[i].copy()] for i in range(n)] if record_path else None
        for t in range(self.dur + K):
            acts = pol.act(obs)
            view = getattr(pol, "view", None)
            obs, _r, done, trunc, _term = (core.step(acts) if view is None
                                           else core.step(acts, view=view))
            self.steps += S
            done = np.asarray(done, bool)
            ended = open_ & (done | np.asarray(trunc, bool))
            if ended.any():
                hits = np.asarray(core.goal_hits, bool)
                fnd |= ended & done & hits
                died |= ended & ~(done & hits)
                ticks[ended] = t + 1
                open_ &= ~ended
            if paths is not None and t % 10 == 9:
                cur_o = core.states_view["origin"]
                for i in np.flatnonzero(open_[:n]):
                    paths[i].append(cur_o[i].astype(np.float64).copy())
            if t + 1 >= self.dur and int(pol._tick) % K == 0 and open_[:n].any():
                cur = core.get_states()
                for i in np.flatnonzero(open_[:n]):
                    end[i] = cur[i].copy()
                    end_obs[i] = np.array(obs[i], np.float32, copy=True)
                    end_keys[i] = ((pol.keys.state[i].copy(), pol.keys.boot[i].copy())
                                   if self.keys_hold and pol.keys is not None else None)
                    ticks[i] = t + 1
                open_[:n] = False
            if not open_[:n].any():
                break
        out = []
        for i in range(n):
            out.append({"end": end[i], "obs": end_obs[i], "keys": end_keys[i],
                        "died": bool(died[i]), "fin": bool(fnd[i]), "ticks": int(ticks[i]),
                        "path": (None if paths is None else
                                 np.round(np.asarray(paths[i]), 1).tolist())})
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default=None, help="another map than the checkpoint's own")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--expansions", type=int, default=0, help="stop after this many (0 = none)")
    ap.add_argument("--parents", type=int, default=64, help="nodes expanded per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-secs", type=float, default=30.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop at the first finish (count finishing chains)")
    a = ap.parse_args(argv)
    import record_ckpt
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(a.seed))
    probe = _ckpt_choices(a.ckpt)
    S = int(a.parents) * probe
    rargv = [str(a.ckpt), "--episodes", "1", "--stochastic", "--plan-scratch", str(S)]
    if a.map:
        rargv += ["--map", str(a.map)]
    ctx = record_ckpt.build(rargv, device=a.device)
    if ctx.planner is None or getattr(ctx, "scratch", None) is None:
        raise SystemExit("edge_archive: the recorder built no primitive planner / scratch core")
    fl = Flyer(ctx)
    P = ctx.planner
    fin = np.asarray(P.finish, np.float64)
    mins = np.asarray(ctx.core.map_bounds()[0], np.float64)
    # the root: the map start, reset like a recording's first episode
    core1 = ctx.core
    core1.set_spawn_pool(np.asarray(ctx.pool)[0:1])
    obs0 = core1.reset(int(a.seed))
    st0 = core1.get_states()[0].copy()
    arch = Archive()
    root = arch.add(st0, fl.fresh_keys(), np.asarray(obs0)[0], keys_of(st0[None], mins)[0], -1,
                    -1, 0, 0, [np.round(st0["origin"].astype(np.float64), 1).tolist()])
    arch.index[arch.key[root]] = [root, root]
    arch.cells.add(arch.key[root][:3])
    d0 = float(np.linalg.norm(st0["origin"].astype(np.float64) - fin))
    print(f"edge_archive: {a.ckpt} on {Path(ctx.map_path).name}: root at "
          f"{np.round(st0['origin'], 0).tolist()}, {d0:,.0f} u from the finish; {fl.C} choices x "
          f"{a.parents} parents per batch = {S} envs; plan {fl.dur} ticks + decision alignment; "
          f"executor SAMPLES (native temperature); count-only selection 1/sqrt(1+n)", flush=True)
    prog_f = open(out / "progress.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    t_rep = t0
    expansions = 0
    yield_new = np.zeros(fl.C, np.int64)
    tried = np.zeros(fl.C, np.int64)
    deaths = fins = 0
    best_d = d0
    best_node = root
    finishers = []

    def report(final=False):
        ids = arch.live_ids()
        dmin = min(float(np.linalg.norm(arch.state[i]["origin"].astype(np.float64) - fin))
                   for i in ids)
        rec = {"secs": round(time.time() - t0, 1), "expansions": expansions,
               "nodes": len(arch), "keys": len(arch.index), "cells": len(arch.cells),
               "new": arch.admit_new, "dup": arch.admit_dup, "replaced": arch.replaced,
               "deaths": deaths, "finishes": fins, "best_dist": round(dmin, 1),
               "best_progress": round(1.0 - dmin / d0, 4),
               "max_depth": max(arch.depth[i] for i in ids),
               "yield": [round(float(yield_new[k]) / max(1, int(tried[k])), 4)
                         for k in range(fl.C)],
               "sim_steps_per_s": round(fl.steps / max(1e-9, time.time() - t0)),
               "final": bool(final)}
        prog_f.write(json.dumps(rec) + "\n")
        prog_f.flush()
        print(f"[{rec['secs']:7.1f}s] exp {expansions:,} nodes {rec['nodes']:,} keys "
              f"{rec['keys']:,} cells {rec['cells']:,} | new {arch.admit_new:,} dup "
              f"{arch.admit_dup:,} repl {arch.replaced:,} | deaths {deaths:,} fin {fins} | "
              f"best {rec['best_progress']:.1%} ({rec['best_dist']:,.0f} u) depth "
              f"{rec['max_depth']} | yield F/L/R {rec['yield']} | {rec['sim_steps_per_s']:,} "
              f"steps/s", flush=True)
        return rec

    while True:
        parents = arch.select(int(a.parents), rng)
        jobs = [(p, k) for p in parents for k in range(fl.C)]
        res = fl.fly(jobs, arch, fin)
        expansions += len(parents)
        for (p, k), r in zip(jobs, res):
            tried[k] += 1
            if r["fin"]:
                fins += 1
                nid = arch.add(arch.state[p], arch.keys_state[p], arch.obs[p], ("FIN",), p, k,
                               arch.depth[p] + 1, arch.t[p] + r["ticks"], r["path"])
                finishers.append(nid)
                continue
            if r["died"] or r["end"] is None:
                deaths += 1
                continue
            key = keys_of(r["end"][None], mins)[0]
            what = arch.admit(r["end"], r["keys"], r["obs"], key, p, k, arch.depth[p] + 1,
                              arch.t[p] + r["ticks"], r["path"])
            if what == "new":
                yield_new[k] += 1
                d = float(np.linalg.norm(r["end"]["origin"].astype(np.float64) - fin))
                if d < best_d:
                    best_d, best_node = d, len(arch) - 1
        now = time.time()
        if now - t_rep >= float(a.report_secs):
            report()
            t_rep = now
        if finishers and not a.keep_going:
            break
        if a.expansions and expansions >= int(a.expansions):
            break
        if now - t0 >= 60.0 * float(a.minutes):
            break
    rec = report(final=True)
    prog_f.close()
    summary = dict(rec, ckpt=str(a.ckpt), map=Path(ctx.map_path).name, d0=d0,
                   choices=fl.C, parents=int(a.parents), plan_ticks=fl.dur,
                   finishers=len(finishers))
    if finishers:
        nid = min(finishers, key=lambda i: arch.t[i])
        ch = arch.chain(nid)
        moves = [arch.move[i] for i in ch[1:]]
        chain = {"ticks_from_root": arch.t[nid], "secs": arch.t[nid] * float(ctx.tick.ms) / 1000.0,
                 "moves": moves,
                 "nodes": [{"origin": np.round(arch.state[i]["origin"].astype(np.float64), 1)
                            .tolist(), "t": arch.t[i], "move": arch.move[i],
                            "path": arch.path[i]} for i in ch]}
        (out / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
        print(f"edge_archive: FIRST FINISHING CHAIN after {expansions:,} expansions "
              f"({rec['secs']:.0f} s): {len(moves)} plans, {chain['secs']:.1f} s from the root, "
              f"moves {moves}", flush=True)
        rep = replay(fl, arch, root, moves, fin)
        summary["replay"] = rep
        print(f"edge_archive: the chain's plan sequence replayed from the true start, no "
              f"restores, executor sampling: {rep['finished']}/{rep['n']} finish; median plans "
              f"completed {rep['median_plans']}", flush=True)
        # the chain's exact states, root first, clocks zeroed: a SELF_STATES spine for the
        # trainer's --demo-file (the agent's own search states - no human input)
        sp = np.stack([arch.state[i] for i in ch]).copy()
        sp["tick"] = 0
        sp["stuck_ticks"] = 0
        np.save(out / "chain_states.npy", sp)
        fid = edge_fidelity(fl, arch, ch, fin)
        summary["edge_fidelity"] = fid
        chain["edge_fidelity"] = fid
        (out / "chain.json").write_text(json.dumps(chain), encoding="utf-8")
        pr = float(np.prod([max(e["survived"], 0) / e["n"] for e in fid]))
        summary["fidelity_product"] = pr
        print(f"edge_archive: product of the edges' survival rates {pr:.4f}", flush=True)
        print("edge_archive: each chain edge re-flown 32 x from its EXACT parent state "
              "(survived / finished): "
              + " ".join(f"{e['move']}:{e['survived']}/{e['finished']}" for e in fid), flush=True)
    else:
        print(f"edge_archive: no finish after {expansions:,} expansions; best "
              f"{rec['best_progress']:.1%} of the start distance (node depth "
              f"{arch.depth[best_node]})", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


def replay(fl, arch, root, moves, fin):
    """The chain's plan sequence from the true start, REPLAYS parallel copies, no restores between
    plans: each copy flies plan i from wherever plan i-1 left it."""
    n = min(REPLAYS, fl.S)
    rep = Archive()
    r0 = rep.add(arch.state[root], arch.keys_state[root], arch.obs[root], ("R",), -1, -1, 0, 0,
                 None)
    cur = [r0] * n
    alive = np.ones(n, bool)
    done_plans = np.zeros(n, np.int64)
    finished = np.zeros(n, bool)
    for m in moves:
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        res = fl.fly([(cur[i], m) for i in idx], rep, fin, record_path=False)
        for i, r in zip(idx, res):
            if r["fin"]:
                finished[i] = True
                alive[i] = False
                done_plans[i] += 1
            elif r["died"] or r["end"] is None:
                alive[i] = False
            else:
                cur[i] = rep.add(r["end"], r["keys"], r["obs"], ("R",), cur[i], m, 0, 0, None)
                done_plans[i] += 1
    return {"n": int(n), "finished": int(finished.sum()),
            "median_plans": float(np.median(done_plans)), "plans": int(len(moves))}


def edge_fidelity(fl, arch, chain_ids, fin):
    """Each edge of the chain re-flown REPLAYS times from its exact parent state (the node the
    archive stored): how often the executor survives that one plan, and finishes (the last edge).
    A chain that finishes once but whose edges survive rarely is a lucky path, not a route the
    executor can fly."""
    n = min(REPLAYS, fl.S)
    out = []
    for a_, b_ in zip(chain_ids[:-1], chain_ids[1:]):
        m = arch.move[b_]
        res = fl.fly([(a_, m)] * n, arch, fin, record_path=False)
        out.append({"from": np.round(arch.state[a_]["origin"].astype(np.float64), 0).tolist(),
                    "move": int(m), "survived": int(sum((not r["died"]) and
                                                        (r["end"] is not None or r["fin"])
                                                        for r in res)),
                    "finished": int(sum(r["fin"] for r in res)), "n": int(n)})
    return out


def _ckpt_choices(path) -> int:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = int((ck.get("config") or {}).get("plan_choices") or 0)
    if c <= 0:
        raise SystemExit("edge_archive: v1 needs a --plan-choices checkpoint")
    return c


if __name__ == "__main__":
    sys.exit(main())
