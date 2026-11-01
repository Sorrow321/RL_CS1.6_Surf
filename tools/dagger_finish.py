"""dagger_finish.py - DAgger labels for the finish: the policy's OWN greedy line from the true start,
cut shortly before each of its closest passes to the finish, continued by tools/finish_mpc.py's engine
MPC to a finish (zlr3b1, 2026-11-01).

Behaviour cloning on MPC lines alone (rooted at mid-episode states) teaches only foreign decisions, and
on this map one update of it (BC 0.5) broke the whole greedy route (zlr3b1 f1). This writes, per root,
a SPLICED plan in beam_best format: the greedy's own per-decision actions from the true start up to
the root, then the MPC's held per-decision controls to the finish - so tools/plan_to_bc.py replays it
from the true start (and asserts the finish) and the BC rows are the policy's own decisions along the
whole route plus the finishing suffix (the uf2 x7 recipe: BC on the line the policy itself flies). It
also writes the suffix alone (root -> finish) so the suffix can be up-weighted by merging it again.

The greedy line is beam_tas's `--envs N --greedy-envs N --save-frontier --frontier-last` output
(beam_frontier.npz: acts, view, spawn_state, obs_start, per-tick states). Roots, generic: the K
closest passes of the line to the finish (the exact hull gap to the finish boxes, minima >= 2 s apart),
the last map contact before each pass (the exit), and --leads seconds before it, on the decision grid.
Nothing reads a place on the map.

    python tools/dagger_finish.py --ckpt C.pt --greedy runs/dag/g0/beam_frontier.npz \
        --out runs/dag/p0 [--passes 4 --leads 0.25,0.45,0.65,0.9 + finish_mpc's CEM flags]
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


def pass_roots(states, gap, tick_s, K, passes, leads, sep_s=2.0):
    """-> sorted unique root ticks (multiples of K): `leads` s before the last contact preceding
    each of the `passes` closest approaches (gap minima >= sep_s apart)."""
    from contact_roots import contact_mask
    c = contact_mask(states, tick_s * 1000.0)
    order = np.argsort(gap)
    picked = []
    sep = int(round(sep_s / tick_s))
    for i in order:
        if all(abs(int(i) - j) > sep for j in picked):
            picked.append(int(i))
        if len(picked) >= passes:
            break
    roots = set()
    for i in picked:
        lc = np.flatnonzero(c[:i])
        if not len(lc):
            continue
        x = int(lc[-1])
        for ld in leads:
            r = x - int(round(ld / tick_s))
            r -= r % K
            if r > 0:
                roots.add(r)
    return sorted(roots), picked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--greedy", required=True, help="beam_frontier.npz of an all-greedy beam_tas")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--leads", default="0.25,0.45,0.65,0.9")
    ap.add_argument("--min-start-s", type=float, default=60.0,
                    help="ignore passes earlier than this (the route's first minutes)")
    ap.add_argument("--death-roots", default=None, metavar="S1,S2,...",
                    help="instead of the closest passes: roots these many seconds before the "
                         "greedy line's END (its death), on the decision grid - where a greedy "
                         "that dies one swing short of a finishable energy needs labels")
    # finish_mpc's planner
    ap.add_argument("--pop", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--elite", type=float, default=0.05)
    ap.add_argument("--plan", type=float, default=2.0)
    ap.add_argument("--exec", type=int, default=12)
    ap.add_argument("--knot", type=int, default=8)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--max-offset", type=float, default=90.0)
    ap.add_argument("--sigma0", type=float, default=5.0)
    ap.add_argument("--sigma-min", type=float, default=1.0)
    ap.add_argument("--energy-w", type=float, default=0.0)
    ap.add_argument("--dist-every", type=int, default=2)
    ap.add_argument("--objective", choices=("finish", "height"), default="height")
    ap.add_argument("--floors", choices=("none", "trace", "mix"), default="none")
    ap.add_argument("--mix-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    a.hold = True
    import torch
    from beam_tas import build_sim, resolve_map, make_scorer, goal_box_floors
    from surfgym.tick import TickClock
    from surfgym.zones import load_zones
    from kino_search import states_rw
    import finish_mpc as FM

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    K = int(cfg.get("act_every") or 1)
    if a.knot % K or a.exec % K:
        raise SystemExit("--knot and --exec must be multiples of act_every")
    tick = TickClock(float(cfg.get("tick_ms") or 10.0))
    a.tick_s = float(tick.ms) / 1000.0
    map_path = resolve_map(a.map, cfg.get("map", "surf_ski_2"))
    g = np.load(a.greedy, allow_pickle=False)
    if int(g["act_every"]) != K:
        raise SystemExit(f"greedy line act_every {int(g['act_every'])} != checkpoint's {K}")
    gst = np.asarray(g["states"])
    gacts = np.asarray(g["acts"])
    gview = np.asarray(g["view"], np.float32)
    zones = load_zones(map_path)
    boxes = zones["end"].get("boxes") or [{"mins": zones["end"]["mins"],
                                           "maxs": zones["end"]["maxs"]}]
    core = build_sim(cfg, map_path, a.pop, 10 ** 8, tick=tick)
    core1 = build_sim(cfg, map_path, 1, 10 ** 8, tick=tick)
    for c in (core, core1):
        c.set_goal_box(zones["end"]["mins"], zones["end"]["maxs"])
        if cfg.get("teleport_fail") or cfg.get("reward") == "race":
            c.set_teleport_fail(True)
    _bx = [[b["mins"], b["maxs"]] for b in boxes]
    floors = goal_box_floors(core1, boxes) if a.floors in ("trace", "mix") else None

    class _Z:
        def sample(self, p):
            return np.zeros(len(np.atleast_2d(p)), np.float32)
    plain = make_scorer(_Z(), None, 0.0, "gap", goal_boxes=_bx, goal_floors=None)
    if a.floors == "trace":
        scorer = make_scorer(_Z(), None, 0.0, "gap", goal_boxes=_bx, goal_floors=floors)
    elif a.floors == "mix":
        _fl = make_scorer(_Z(), None, 0.0, "gap", goal_boxes=_bx, goal_floors=floors)

        def scorer(s):
            _x, g0 = plain(s)
            _x, g1 = _fl(s)
            gg = g0 + a.mix_w * g1
            return -gg, gg
    else:
        scorer = plain
    # the greedy line's closest passes (after --min-start-s) and their roots
    _s, gap = plain(gst)
    gap = np.asarray(gap, np.float64).copy()
    gap[: int(round(a.min_start_s / a.tick_s))] = np.inf
    leads = [float(x) for x in a.leads.split(",")]
    if a.death_roots:
        n = len(gst)
        roots = sorted({(n - int(round(float(x) / a.tick_s))) // K * K
                        for x in a.death_roots.split(",")} - {0})
        picked = [n - 1]
    else:
        roots, picked = pass_roots(gst, gap, a.tick_s, K, a.passes, leads)
    print(f"dagger_finish: {Path(a.greedy).parent.name}: {len(gst)} ticks, passes at "
          f"{[round(p * a.tick_s, 1) for p in picked]} s (gaps "
          f"{[round(float(gap[p]), 0) for p in picked]}), {len(roots)} roots", flush=True)
    for c in (core, core1):
        c.set_spawn_pool(gst[:1].copy())
        c.reset(0)
    rw = states_rw(core)
    sv = core.states_view
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    summary = []
    spawn = np.asarray(g["spawn_state"]).reshape(-1)[:1].copy()
    for r in roots:
        row0 = gst[r].copy()
        row0["tick"], row0["stuck_ticks"] = 0, 0
        t0 = time.perf_counter()
        res = FM.run_root(core, rw, sv, row0, scorer, a, rng, None)
        line = {"root_tick": int(r), "root_s": round(r * a.tick_s, 2),
                "best_gap": round(float(res["best_gap"]), 1),
                "finish_tick": res["finish_tick"], "secs": round(time.perf_counter() - t0, 1)}
        if res["finish_tick"] is not None:
            states, ft = FM.replay(core1, row0, res)
            line["replay_ok"] = ft == res["finish_tick"]
            if ft is not None:
                n_dec = -(-len(res["off"]) // K)
                acts = np.zeros((n_dec, 6), gacts.dtype)
                acts[:, 0], acts[:, 1], acts[:, 2] = 7, 3, 1
                acts[:, 3] = res["side"][::K][:n_dec]
                acts[:, 5] = res["duck"][::K][:n_dec]
                vw = np.zeros((n_dec, 2), np.float32)
                vw[:, 0] = res["off"][::K][:n_dec]
                vw[:, 1] = -10.0
                d0 = r // K
                common = dict(gate_seed=int(g["gate_seed"]), act_every=K,
                              map=Path(map_path).stem, greedy_ticks=0, view_continuous=1,
                              view_mode=int(g["view_mode"]) if "view_mode" in g.files else 1,
                              tick_ms=a.tick_s * 1000.0,
                              tick_ms_requested=float(cfg.get("tick_ms") or 10.0),
                              tick_pattern_ms=np.array([int(round(a.tick_s * 1000.0))]),
                              objective="finish", source="tools/dagger_finish.py")
                # the SPLICED plan: the greedy's own decisions from the true start, then the MPC
                np.savez(out / f"splice_t{r}.npz", spawn_state=spawn,
                         obs_start=np.asarray(g["obs_start"], np.float32),
                         finish_ticks=int(r + ft),
                         acts=np.concatenate([gacts[:d0], acts]),
                         view=np.concatenate([gview[:d0], vw]), root_decision=d0, **common)
                # the suffix alone (root -> finish), for up-weighting
                core1.set_spawn_pool(row0[None].copy())
                obs0 = np.array(core1.reset(0))[0].astype(np.float32)
                np.savez(out / f"suffix_t{r}.npz", spawn_state=row0[None].copy(),
                         obs_start=obs0, finish_ticks=int(ft), acts=acts, view=vw, **common)
                core1.set_spawn_pool(gst[:1].copy())
                core1.reset(0)
                np.save(out / f"line_t{r}.npy", states)
                line["states"] = len(states)
        print(json.dumps(line), flush=True)
        summary.append(line)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
