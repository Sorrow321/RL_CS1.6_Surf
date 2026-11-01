"""finish_mpc.py - receding-horizon CEM (MPC) in the real engine, from the policy's OWN states, toward
the FINISH (zlr3b1, 2026-11-01).

tools/mpc_bench.py's planner (plan --plan s ahead, execute --exec ticks of the best plan, replan from
where it landed, warm-started), with the finish as the objective. A candidate plan scores

  finished inside the plan (the core's goal test, any finish model)  -> 1e6 - its finish tick;
  alive at the plan's end -> -(the smallest exact hull gap to the REACHABLE finish region over the
      plan, every --dist-every ticks) + --energy-w x (energy height at the plan's end);
  died inside the plan -> discarded.

The reachable region is beam_tas --score gap's: each finish box inflated by the hull, its floor raised
to clear a solid floor under the box (tools/beam_tas.goal_box_floors). Controls per knot (--knot
ticks): the view yaw TARGET offset from the live velocity heading (degrees, Gaussian, interpolated
between knots, clipped to --max-offset), the side key (A / none / D) and duck (Bernoulli); forward
and jump stay released. No policy, no reward, no reference line: every constant is generic and the
start states are the policy's own (a --spawn-states pool or a record_ckpt --dump-states file).

The core is built from the CHECKPOINT's config (tools/beam_tas.build_sim: its tick, maxvel, view
mode and ceilings), so a line found here is a line the trained policy's physics can fly. On a
finish the executed actions are replayed open-loop from the root on a fresh 1-env core; the replay
must finish at the same tick, and the line's per-tick states (root .. finish) are written for
--spawn-states.

    python tools/finish_mpc.py --ckpt runs_ckpt.pt --states autoresearch/spines/X.npy \
        --rows 41316,41345 --out runs/fmpc/a
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

G = 800.0


class _ZeroField:
    def sample(self, p):
        return np.zeros(len(np.atleast_2d(p)), np.float32)


def run_root(core, rw, sv, row0, scorer, a, rng, K_ticks):
    """MPC from one root state. -> dict(best_gap, finished, finish_tick, acts, view, ticks)."""
    TS = a.tick_s
    Hp = int(round(a.plan / TS))
    Kn = Hp // a.knot + 1
    n_rep = int(round(a.duration / TS)) // a.exec
    shift = a.exec / a.knot
    P = a.pop
    mu = np.zeros(Kn)
    sig = np.full(Kn, a.sigma0)
    pside = np.tile(np.array([0.45, 0.10, 0.45]), (Kn, 1))      # A / none / D
    pduck = np.full(Kn, 0.3)
    n_el = max(4, int(P * a.elite))
    act = np.zeros((P, 6), np.int32)
    act[:, 0], act[:, 1], act[:, 2] = 7, 3, 1
    view = np.zeros((P, 2), np.float32)
    view[:, 1] = -10.0
    best_plan = None
    cur = row0.copy()
    ex_off, ex_side, ex_duck = [], [], []      # executed controls, per tick
    best_gap = np.inf
    fin_tick = None
    t_exec = 0
    died = False
    for r in range(n_rep):
        plan_fin = False
        for it in range(a.iters):
            off = np.clip(mu + sig * rng.standard_normal((P, Kn)), -a.max_offset, a.max_offset)
            u = rng.random((P, Kn))
            cs = np.cumsum(pside, axis=1)
            side = (u[:, :, None] > cs[None, :, :]).sum(-1).astype(np.int32)
            side = np.minimum(side, 2)
            duck = (rng.random((P, Kn)) < pduck).astype(np.int32)
            if best_plan is not None:
                off[0], side[0], duck[0] = best_plan
            rw[:] = cur
            alive = np.ones(P, bool)
            ftick = np.full(P, -1)
            mgap = np.full(P, np.inf)
            for k in range(Hp):
                j, f = divmod(k, a.knot)
                j1 = min(j + 1, Kn - 1)
                view[:, 0] = off[:, j] + (off[:, j1] - off[:, j]) * (f / a.knot)
                act[:, 3] = side[:, j]
                act[:, 5] = duck[:, j]
                _o, _rr, done, trunc, _t = core.step(act, view=view)
                ended = (np.asarray(done) | np.asarray(trunc)).astype(bool)
                hit = np.asarray(core.goal_hits).astype(bool) & alive & (ftick < 0)
                ftick[hit] = k
                alive &= ~(ended & ~hit)
                alive &= ftick < 0
                if k % a.dist_every == 0 or k == Hp - 1:
                    _s, g = scorer(sv)
                    mgap = np.where(alive, np.minimum(mgap, g), mgap)
            o = np.asarray(sv["origin"], np.float64)
            v = np.asarray(sv["velocity"], np.float64)
            en = o[:, 2] + (v * v).sum(-1) / (2 * G)
            score = -mgap + a.energy_w * en
            score[~alive] = -np.inf
            fin = ftick >= 0
            score[fin] = 1e6 - ftick[fin]
            best_gap = min(best_gap, float(np.min(np.where(np.isfinite(score), mgap, np.inf))))
            el = np.argsort(-score)[:n_el]
            el = el[np.isfinite(score[el])]
            if len(el) == 0:
                continue
            best_plan = (off[el[0]].copy(), side[el[0]].copy(), duck[el[0]].copy())
            plan_fin = bool(fin[el[0]])
            if len(el) >= 2:
                mu = 0.7 * off[el].mean(0) + 0.3 * mu
                sig = np.maximum(0.7 * off[el].std(0) + 0.3 * sig, a.sigma_min)
                ps = np.stack([(side[el] == s).mean(0) for s in range(3)], axis=1)
                pside = np.clip(0.7 * ps + 0.3 * pside, 0.02, 0.98)
                pside /= pside.sum(1, keepdims=True)
                pduck = np.clip(0.7 * duck[el].mean(0) + 0.3 * pduck, 0.02, 0.98)
            if fin.any():
                break
        if best_plan is None:
            break
        # execute: up to --exec ticks of the best plan (all of it to its finish tick if it finishes)
        off_b, side_b, duck_b = best_plan
        rw[:] = cur
        # a finishing plan is executed to its finish; otherwise its first --exec ticks
        for k in range(Hp if plan_fin else a.exec):
            j, f = divmod(k, a.knot)
            j1 = min(j + 1, Kn - 1)
            view[:, 0] = off_b[j] + (off_b[j1] - off_b[j]) * (f / a.knot)
            act[:, 3] = side_b[j]
            act[:, 5] = duck_b[j]
            ex_off.append(float(view[0, 0]))
            ex_side.append(int(side_b[j]))
            ex_duck.append(int(duck_b[j]))
            _o, _rr, done, trunc, _t = core.step(act, view=view)
            t_exec += 1
            if bool(np.asarray(core.goal_hits)[0]):
                fin_tick = t_exec
                break
            if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
                died = True
                break
        if fin_tick is not None or died:
            break
        cur = sv[0].copy()
        cur["tick"], cur["stuck_ticks"] = 0, 0
        s = int(round(shift))
        mu = np.r_[mu[s:], np.repeat(mu[-1], s)]
        sig = np.r_[np.maximum(sig[s:], a.sigma_min), np.full(s, a.sigma0)]
        pside = np.r_[pside[s:], np.tile(np.array([[0.45, 0.10, 0.45]]), (s, 1))]
        pduck = np.r_[pduck[s:], np.full(s, 0.3)]
        best_plan = (np.r_[off_b[s:], np.repeat(off_b[-1], s)],
                     np.r_[side_b[s:], np.repeat(side_b[-1], s)],
                     np.r_[duck_b[s:], np.repeat(duck_b[-1], s)])
    return {"best_gap": best_gap, "finish_tick": fin_tick, "ticks": t_exec, "died": died,
            "off": np.asarray(ex_off, np.float32), "side": np.asarray(ex_side, np.int32),
            "duck": np.asarray(ex_duck, np.int32)}


def replay(core1, row0, res):
    """Open-loop replay of the executed controls on a 1-env core -> (states, finish tick or None)."""
    core1.set_state(0, row0)
    act = np.zeros((1, 6), np.int32)
    act[:, 0], act[:, 1], act[:, 2] = 7, 3, 1
    view = np.zeros((1, 2), np.float32)
    view[:, 1] = -10.0
    states = [core1.get_states()[0].copy()]
    for t in range(len(res["off"])):
        view[0, 0] = res["off"][t]
        act[0, 3] = res["side"][t]
        act[0, 5] = res["duck"][t]
        _o, _r, done, trunc, _i = core1.step(act, view=view)
        if bool(np.asarray(core1.goal_hits)[0]):
            return np.array(states), t + 1
        states.append(core1.get_states()[0].copy())
        if bool(np.asarray(done)[0]) or bool(np.asarray(trunc)[0]):
            return np.array(states), None
    return np.array(states), None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint whose CONFIG builds the core")
    ap.add_argument("--map", default=None)
    ap.add_argument("--states", required=True, help="STATE_DTYPE .npy of the policy's own states")
    ap.add_argument("--rows", required=True, help="comma-separated root rows")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--pop", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--elite", type=float, default=0.05)
    ap.add_argument("--plan", type=float, default=1.5, help="planning horizon, s")
    ap.add_argument("--exec", type=int, default=8, help="ticks executed per replan")
    ap.add_argument("--knot", type=int, default=4)
    ap.add_argument("--duration", type=float, default=3.0, help="game s per root")
    ap.add_argument("--max-offset", type=float, default=150.0, help="deg")
    ap.add_argument("--sigma0", type=float, default=30.0, help="deg")
    ap.add_argument("--sigma-min", type=float, default=2.0, help="deg")
    ap.add_argument("--energy-w", type=float, default=0.05)
    ap.add_argument("--dist-every", type=int, default=2)
    ap.add_argument("--floors", choices=("none", "trace", "mix"), default="none",
                    help="none (default): the gap to the hull-inflated finish boxes - a hull "
                         "sliding up a platform's SIDE touches the box once its top passes the "
                         "box bottom, which the core counts (the collision creeps onto the "
                         "inflated face) and so does the grader (1 u slack); trace: the region "
                         "must also clear a solid floor under each box (stricter); mix: none "
                         "+ --mix-w x trace - at a platform's side the hull touches the box "
                         "(gap ~0) but the core fires only when the collision creeps onto the "
                         "face, so among those the higher apex (clearing the floor, a definite "
                         "finish) ranks first")
    ap.add_argument("--mix-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from beam_tas import build_sim, resolve_map, make_scorer, goal_box_floors
    from surfgym.tick import TickClock
    from surfgym.zones import load_zones
    from surfgym.core import STATE_DTYPE
    from kino_search import states_rw

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    tick = TickClock(float(cfg.get("tick_ms") or 10.0))
    a.tick_s = float(tick.ms) / 1000.0
    map_path = resolve_map(a.map, cfg.get("map", "surf_ski_2"))
    cap = 10 ** 8
    core = build_sim(cfg, map_path, a.pop, cap, tick=tick)
    core1 = build_sim(cfg, map_path, 1, cap, tick=tick)
    zones = load_zones(core.bsp_path)
    for c in (core, core1):
        c.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        if cfg.get("teleport_fail") or cfg.get("reward") == "race":
            c.set_teleport_fail(True)
    pool = np.load(a.states)
    if pool.dtype != STATE_DTYPE:
        raise SystemExit(f"{a.states}: not a STATE_DTYPE array")
    boxes = zones["end"].get("boxes") or [{"mins": zones["end"]["mins"],
                                           "maxs": zones["end"]["maxs"]}]
    floors = goal_box_floors(core1, boxes) if a.floors in ("trace", "mix") else None
    _bx = [[b["mins"], b["maxs"]] for b in boxes]
    scorer = make_scorer(_ZeroField(), None, 0.0, "gap", goal_boxes=_bx,
                         goal_floors=floors if a.floors == "trace" else None)
    if a.floors == "mix":
        _plain, _floored = scorer, make_scorer(_ZeroField(), None, 0.0, "gap", goal_boxes=_bx,
                                               goal_floors=floors)

        def scorer(states):
            _s, g0 = _plain(states)
            _s, g1 = _floored(states)
            g = g0 + float(a.mix_w) * g1
            return -g, g
    print(f"finish_mpc: {Path(map_path).name} tick {a.tick_s * 1000:.2f} ms, maxvel "
          f"{cfg.get('maxvel')}, view {cfg.get('view_absolute')}, floors {floors}; pop {a.pop} "
          f"plan {a.plan}s exec {a.exec} knot {a.knot} iters {a.iters}", flush=True)
    for c in (core, core1):
        c.set_spawn_pool(pool[:1].copy())
        c.reset(0)
    rw = states_rw(core)
    sv = core.states_view
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    summary = []
    for rs in a.rows.split(","):
        r = int(rs)
        row0 = pool[r].copy()
        row0["tick"], row0["stuck_ticks"] = 0, 0
        t0 = time.perf_counter()
        res = run_root(core, rw, sv, row0, scorer, a, rng, None)
        line = {"row": r, "best_gap": round(float(res["best_gap"]), 1),
                "finish_tick": res["finish_tick"], "ticks": res["ticks"], "died": res["died"],
                "secs": round(time.perf_counter() - t0, 1)}
        if res["finish_tick"] is not None:
            states, ft = replay(core1, row0, res)
            line["replay_finish_tick"] = ft
            line["replay_ok"] = ft == res["finish_tick"]
            if ft is not None:
                np.save(out / f"line_r{r}.npy", states)
                np.savez(out / f"line_r{r}_acts.npz", off=res["off"], side=res["side"],
                         duck=res["duck"], root=row0, tick_ms=a.tick_s * 1000.0)
                line["states"] = len(states)
        print(json.dumps(line), flush=True)
        summary.append(line)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
