"""route_follow.py - can the trained executor FLY a route the archive found, given it as ONE fixed
line (no planner, no replanning)?

    python tools/route_follow.py <ckpt> <chain.json> [--n 64] [--greedy] [--rounds 2]

WHY (2026-09-27). A chain found by tools/edge_archive.py is a sequence of RAY choices; replayed
open-loop from the start it fails (0-1 of 32): the rays are relative to the current velocity, so a
small drift turns the same choice into a different line. The executor, however, was trained to
follow a LINE shown in its lookahead fan, and a fixed line in world coordinates is corrected
against continuously. If the executor can fly the whole found route from the start as one line,
the macro problem reduces to finding the route (the archive does it in seconds) and the micro
problem to following a line - the project's xSELF construction (a line from the agent's own
trajectories), with no learned planner.

WHAT. The chain's flown path (every node's recorded flight, root first) is resampled at the fan
spacing into one polyline. The scratch core is reset from the map's own start pool; every env gets
that polyline as its line, the checkpoint's executor flies (sampling, or --greedy) until each env
finishes, dies, or reaches the episode cap. Reported: finishes, their times, and for the rest the
share of the route's arc reached (the line's own coordinate, no corridor).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def route_of(chain_path, spacing):
    from surfgym.route import resample_polyline
    c = json.loads(Path(chain_path).read_text(encoding="utf-8"))
    pts = []
    for n in c["nodes"]:
        for p in (n.get("path") or [n["origin"]]):
            q = np.asarray(p, np.float64)
            if not pts or np.linalg.norm(q - pts[-1]) > 1.0:
                pts.append(q)
    line, total = resample_polyline(np.asarray(pts), spacing)
    return np.asarray(line, np.float32), float(total), c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("chains", nargs="+")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--temp", type=float, default=None, help="all heads' temperature")
    ap.add_argument("--keys-temp", type=float, default=None, help="the keys heads' temperature")
    ap.add_argument("--view-scale", type=float, default=None, help="the view sigma multiplier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    import record_ckpt
    from surfgym.goalarc import MultiArcProgress
    rargv = [str(a.ckpt), "--episodes", "1", "--plan-scratch", str(int(a.n))]
    if not a.greedy:
        rargv.append("--stochastic")
        for flag, val in (("--exec-temp", a.temp), ("--exec-keys-temp", a.keys_temp),
                          ("--exec-view-scale", a.view_scale)):
            if val is not None:
                rargv += [flag, str(val)]
    ctx = record_ckpt.build(rargv, device=a.device)
    sc = ctx.scratch
    core, S = sc.core, int(sc.core.num_envs)
    spacing = float(ctx.planner.prim.spacing)
    tick_s = float(ctx.tick.ms) / 1000.0
    pool = np.asarray(ctx.pool)
    for cp in a.chains:
        line, total, c = route_of(cp, spacing)
        fins, arcs, times = 0, [], []
        n_ep = 0
        for rnd in range(int(a.rounds)):
            core.set_spawn_pool(pool)
            obs = core.reset(int(a.seed) * 1000 + rnd)
            sc.line.set_lines(np.arange(S), [line] * S)
            trk = MultiArcProgress(S, l_max=len(line) + 8, spacing=spacing, corridor=1.0e6,
                                   window=16)
            trk.set_lines(np.arange(S), [line] * S, origin=core.states_view["origin"]
                          .astype(np.float64))
            pol = sc.make_policy(core, sc.line)
            alive = np.ones(S, bool)
            best = np.zeros(S)
            t = 0
            while alive.any():
                acts = pol.act(obs)
                view = getattr(pol, "view", None)
                obs, _r, done, trunc, _term = (core.step(acts) if view is None
                                               else core.step(acts, view=view))
                t += 1
                trk.advance(core.states_view["origin"].astype(np.float32))
                best = np.where(alive, np.maximum(best, trk.arc), best)
                ended = alive & (np.asarray(done, bool) | np.asarray(trunc, bool))
                if ended.any():
                    hits = np.asarray(core.goal_hits, bool)
                    for i in np.flatnonzero(ended):
                        n_ep += 1
                        if bool(done[i]) and hits[i]:
                            fins += 1
                            times.append(t * tick_s)
                        else:
                            arcs.append(float(best[i]) / max(1.0, float(trk.total_arc()[i])))
                    alive &= ~ended
        mode = ("greedy" if a.greedy else
                f"sampling (temp {a.temp}, keys {a.keys_temp}, view x {a.view_scale})")
        arcs_a = np.asarray(arcs) if arcs else np.zeros(0)
        print(f"route_follow {Path(cp).parent.name}: route {total:,.0f} u ({len(line)} vertices, "
              f"chain {c['secs']:.1f} s); executor {mode}; {fins}/{n_ep} finished"
              + (f" (median {np.median(times):.1f} s)" if times else "")
              + (f"; the others reached median {np.median(arcs_a):.0%} of the route's arc "
                 f"(p25 {np.percentile(arcs_a, 25):.0%}, p75 {np.percentile(arcs_a, 75):.0%})"
                 if len(arcs_a) else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
