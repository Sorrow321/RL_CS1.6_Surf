"""chain_check.py - the ramp-command chain invariant (Codex 2026-09-27 20:16Z): from a policy's OWN
recorded states, fly a CHAIN of ramp commands A -> B -> C UNINTERRUPTED - no state restore between
commands: at a direct capture the next command's line is swapped into the running flight at the
decision boundary - and report per stage the exact first-contact sets, deaths, timeouts and the
arrival states. MEASUREMENT only; a record's states are a ruler here, never training data
(CLAUDE.md section 0) - the default states file is our own jt3ANCHU finisher.

    python tools/chain_check.py <mover ckpt> --map maps_pool/<m>.bsp --ramps <ramps.npz>
        --states <states.npz>[:key] --starts 4.0,5.5,6.0 --chain 26,29,33 [--n 64] [--greedy]

The contract is edge_archive.Flyer's: the source set is fixed at the command start (the capture
set that ended the previous command, the proximity source and the first tick's contacts); before
the departure (DEPART_TICKS ticks touching none of it) the first NON-source contact is the
outcome, after it any contact is; the command ends at the first decision boundary at or after the
contact, or at RAMP_TIMEOUT. A stage is DIRECT when its first new contact set is exactly {target}.
Each command's line comes from edge_archive.RampOperator on a coast of the flight's CURRENT state,
simulated in a second scratch core (the flight's own core is never touched).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--map", required=True)
    ap.add_argument("--ramps", required=True)
    ap.add_argument("--states", required=True, help="an .npz of recorded state arrays (key "
                    "after ':' or the first array) - one state per physics tick from t = 0")
    ap.add_argument("--starts", default="4.0,5.5,6.0", help="seconds into the recording")
    ap.add_argument("--chain", required=True, help="surface ids, comma separated")
    ap.add_argument("--n", type=int, default=64, help="parallel copies per start")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--exec-view-scale", type=float, default=None,
                    help="sampled executor: the continuous view heads' sigma multiplier only "
                         "(record_ckpt --exec-view-scale; the keys stay at their temperature)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    from ramps import CATS, RampMap
    torch.manual_seed(int(a.seed))
    chain = [int(x) for x in a.chain.split(",")]
    n = int(a.n)
    base = [str(a.ckpt), "--episodes", "1", "--plan-scratch", str(n), "--ep-ticks",
            str(int(round(ea.RAMP_TIMEOUT * 100)) * (len(chain) + 1) + 400), "--map", str(a.map)]
    base += [] if a.greedy else ["--stochastic"]
    if a.exec_view_scale is not None:
        base += ["--exec-view-scale", str(float(a.exec_view_scale))]
    ctx = record_ckpt.build(base)
    ctxc = record_ckpt.build(base)          # a second scratch core for the coasts
    core = ctx.scratch.core
    fb = ctx.finish_box
    fin = 0.5 * (np.asarray(fb[0], np.float64) + np.asarray(fb[1], np.float64))
    rm = RampMap(a.ramps)
    op = ea.RampOperator(float(ctx.tick.ms), fin, rm, 4)
    ctx.planner = op
    fl = ea.Flyer(ctx)
    fl.ramp_map = rm
    K = fl.K

    class _Coast:
        pass
    flc = _Coast()
    flc.core = ctxc.scratch.core
    flc.live_ticks = 0
    spath, _, skey = str(a.states).partition(":")
    z = np.load(spath)
    rec = z[skey or z.files[0]]
    kidx = [op.targets.index(s) for s in chain]
    neutral = np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (n, 1))
    t_cmd_max = int(round(ea.RAMP_TIMEOUT * 1000.0 / float(ctx.tick.ms)))
    tick_s = float(ctx.tick.ms) / 1000.0
    # the recording's own first contact with each chain surface (the ruler)
    rc = rm.contact(rec["origin"].astype(np.float64))
    ref = {}
    for s in chain:
        w = np.flatnonzero(rc == s)
        if len(w):
            v = rec["velocity"][w[0]].astype(np.float64)
            ref[s] = (w[0] * tick_s, float(np.linalg.norm(v)))

    def lines_for(states, envs, ks):
        """coast each env's CURRENT state in the coast core; its command line to target ks[i]"""
        tmp = ea.Archive()
        nids = []
        for i in envs:
            st = states[i].copy()
            nids.append(tmp.add(st, None, np.zeros(1, np.float32), ("C",), -1, -1, 0, 0, None))
        op.plan(flc, tmp, nids)
        out = []
        for i, nid in zip(envs, nids):
            op.queue = [(tmp.token, nid)]
            st = states[i]
            ln, _pts = op.line_and_curve_of(st["origin"].astype(np.float64),
                                            st["velocity"].astype(np.float64), float(st["yaw"]),
                                            op.choice_nums[ks[i]], ks[i])
            out.append((ln, float(op.last_tc)))
        return out

    print(f"chain_check: {Path(a.ckpt).name} on {Path(a.map).stem}, chain "
          + " -> ".join(f"{CATS[rm.cat[s]]} {s}" for s in chain)
          + f", {n} {'greedy' if a.greedy else 'sampled'} copies per start"
          + (f" (view sigma x {a.exec_view_scale:g})" if a.exec_view_scale is not None else "")
          + ", uninterrupted")
    for s in chain:
        if s in ref:
            print(f"   the recording first touches {s} at {ref[s][0]:.2f} s, |v| {ref[s][1]:,.0f}")
    for T in [float(x) for x in a.starts.split(",")]:
        st0 = rec[int(round(T / tick_s))].copy()
        st0["tick"] = 0
        st0["stuck_ticks"] = 0
        for i in range(n):
            core.set_state(i, st0)
        obs, _r, _d, _t, _ = core.step(neutral)
        obs = np.ascontiguousarray(np.asarray(obs, np.float32))
        cur = core.get_states()
        pol = ctx.scratch.make_policy(core, ctx.scratch.line)
        if fl.keys_hold:
            from surfgym.keyshold import KeysHold
            pol.keys = KeysHold(n)
            pol._keys_tick = cur["tick"].astype(np.int64) - int(getattr(pol, "_period", K))
        pol._tick = 0
        stage = np.zeros(n, np.int64)
        active = np.ones(n, bool)
        ks = {i: kidx[0] for i in range(n)}
        lt = lines_for(cur, list(range(n)), ks)
        ctx.scratch.line.set_lines(np.arange(n), [x[0] for x in lt])
        tc = {i: lt[i][1] for i in range(n)}
        p0 = rm.contact(cur["origin"].astype(np.float64))
        src = [({int(p0[i])} if p0[i] >= 0 else set()) for i in range(n)]
        departed = np.zeros(n, bool)
        away = np.zeros(n, np.int64)
        t_cmd = np.zeros(n, np.int64)
        pend = [None] * n
        hit_at = np.full(n, -1, np.int64)
        log = [[] for _ in range(n)]
        for t in range(t_cmd_max * len(chain) + 4 * K):
            acts = pol.act(obs)
            view = getattr(pol, "view", None)
            obs, _r, done, trunc, _ = (core.step(acts) if view is None
                                       else core.step(acts, view=view))
            obs = np.ascontiguousarray(np.asarray(obs, np.float32))
            done = np.asarray(done, bool)
            ended = active & (done | np.asarray(trunc, bool))
            cnt, tn_, tp_ = core.get_touch()
            sets = rm.touch_sets(cnt, tn_, tp_)
            for i in np.flatnonzero(active & (hit_at < 0)):      # the death tick included
                ts_ = sets[i]
                if t_cmd[i] == 0:
                    src[i] |= ts_
                    continue
                new = ts_ if departed[i] else (ts_ - src[i])
                if new:
                    pend[i] = sorted(new)
                    hit_at[i] = t_cmd[i] + 1
                elif not departed[i]:
                    if ts_ & src[i]:
                        away[i] = 0
                    else:
                        away[i] += 1
                        departed[i] = away[i] >= ea.DEPART_TICKS
            t_cmd[active] += 1
            if ended.any():
                gh = np.asarray(core.goal_hits, bool)
                for i in np.flatnonzero(ended):
                    log[i].append((int(stage[i]), "fin" if (done[i] and gh[i]) else "died",
                                   pend[i], int(hit_at[i]), None, None))
                active &= ~ended
            if int(pol._tick) % K == 0 and active.any():
                cs = core.get_states()
                adv = []
                for i in np.flatnonzero(active):
                    if hit_at[i] >= 0:
                        s_t = chain[stage[i]]
                        direct = pend[i] == [s_t]
                        v = cs[i]["velocity"].astype(np.float64)
                        nb = rm.normal[s_t] / max(1e-9, float(np.linalg.norm(rm.normal[s_t])))
                        log[i].append((int(stage[i]), "direct" if direct else "wrong", pend[i],
                                       int(hit_at[i]), float(np.linalg.norm(v)),
                                       float(v @ nb) / max(1e-9, float(np.linalg.norm(v)))))
                        if direct and stage[i] + 1 < len(chain):
                            adv.append(int(i))
                        else:
                            active[i] = False
                    elif t_cmd[i] >= t_cmd_max:
                        log[i].append((int(stage[i]), "timeout", None, -1, None, None))
                        active[i] = False
                if adv:
                    for i in adv:
                        stage[i] += 1
                        ks[i] = kidx[stage[i]]
                    lt = lines_for(cs, adv, ks)
                    ctx.scratch.line.set_lines(np.asarray(adv), [x[0] for x in lt])
                    for i, x in zip(adv, lt):
                        tc[i] = x[1]
                        src[i] = set(pend[i])
                        departed[i] = False
                        away[i] = 0
                        t_cmd[i] = 0
                        pend[i] = None
                        hit_at[i] = -1
            if not active.any():
                break
        for i in np.flatnonzero(active):
            log[i].append((int(stage[i]), "unfinished", None, -1, None, None))
        # per stage: how many copies were commanded it, and their first outcomes
        print(f"start t {T:.2f} s (|v| {np.linalg.norm(st0['velocity']):,.0f}, source "
              f"{sorted(src[0]) if not log[0] else '-'}):")
        for j, s in enumerate(chain):
            rows = [r for L in log for r in L if r[0] == j]
            if not rows:
                print(f"   stage {j} ({s}): commanded 0 times")
                break
            kinds = {}
            for r in rows:
                kinds[r[1]] = kinds.get(r[1], 0) + 1
            wrong = {}
            for r in rows:
                if r[1] in ("wrong", "died") and r[2]:
                    kk = ",".join(f"{CATS[rm.cat[x]][0]}{x}" if x >= 0 else str(x) for x in r[2])
                    wrong[kk] = wrong.get(kk, 0) + 1
            dr = [r for r in rows if r[1] == "direct"]
            sp = np.array([r[4] for r in dr]) if dr else np.zeros(0)
            vn = np.array([r[5] for r in dr]) if dr else np.zeros(0)
            ht = np.array([r[3] for r in dr]) * tick_s if dr else np.zeros(0)
            print(f"   stage {j} ({CATS[rm.cat[s]]} {s}): commanded {len(rows)} -> " +
                  ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
                  + (f"; direct: capture after {np.median(ht):.2f} s (median), |v| median "
                     f"{np.median(sp):,.0f} [{sp.min():,.0f}-{sp.max():,.0f}], v.n/|v| median "
                     f"{np.median(vn):+.2f}" if dr else "")
                  + (f"; other first contacts {dict(sorted(wrong.items(), key=lambda x: -x[1])[:5])}"
                     if wrong else ""))
        full = sum(1 for L in log if any(r[0] == len(chain) - 1 and r[1] == "direct" for r in L))
        print(f"   the whole chain direct and uninterrupted: {full}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
