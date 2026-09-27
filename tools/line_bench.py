"""line_bench.py - can a line-following mover fly a piece of route when it is SHOWN the right
line from the right state? A map-agnostic version of tools/uf2_landing_bench.py for any map and
any recording with full states (record_ckpt.py --dump-states). MEASUREMENT only: the recording is
a ruler (CLAUDE.md section 0 - with a human record it must never train anything).

    python tools/line_bench.py <mover ckpt> --states <rec_states.npz> [--ep ep_0000] --t 7.0
        --map maps_pool/<m>.bsp [--goal-cell 72] [--secs 4] [--n 64] [--greedy] [--seed 0]

From the recording's state at --t the mover flies, for --secs:
  record : the recording's own path from --t, resampled at 128 u (the line it actually flew);
  coast  : straight along the current 3-D velocity (edge_archive's rays4 continuation);
  ray0   : the forward level ray (edge_archive --rays 3, choice 0);
  prim   : random step-1 primitives from the mover's own config.
Reported per line: alive at the end, geodesic progress gained (the map's goal field) against the
recording's own gain over the same time, and the speed at the end.
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
    ap.add_argument("--states", required=True)
    ap.add_argument("--ep", default="ep_0000")
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--goal-cell", type=int, default=32)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vscale", type=float, default=1.0,
                    help="scale the starting velocity (how much speed does this piece need?)")
    a = ap.parse_args(argv)
    import torch
    import record_ckpt
    import edge_archive as ea
    from surfgym.goalfield import load_goal_field
    from surfgym.route import resample_polyline
    bsp = Path(a.map)
    gf = load_goal_field(str(bsp.with_name(f"{bsp.stem}.goal_{a.goal_cell}.npz")))
    rec = np.load(a.states)[a.ep]
    k0 = int(round(a.t * 100))
    st = rec[k0].copy()
    st["velocity"] = st["velocity"] * np.float32(a.vscale)
    st["tick"] = 0
    st["stuck_ticks"] = 0
    torch.manual_seed(int(a.seed))
    # the scratch core's episode cap must exceed the flight: a mover's own cap (e.g. 400 ticks for
    # a 2 s-primitive mover) would TRUNCATE a longer flight and the Flyer counts that as a death
    ctx = record_ckpt.build([str(a.ckpt), "--episodes", "1", "--plan-scratch", str(a.n),
                             "--ep-ticks", str(int(a.secs * 100) + 400)]
                            + ([] if a.greedy else ["--stochastic"]) + ["--map", str(bsp)])
    fin = np.zeros(3)
    dur = int(round(a.secs * 100))
    path = rec["origin"][k0:k0 + dur + 50].astype(np.float64)
    rline, _ = resample_polyline(path, ea.RAY_SPACING)
    rline = np.asarray(rline, np.float32)

    class _Fixed:
        n_choice = 1
        commit_ticks = budget_ticks = dur
        choice_nums = np.zeros((1, 1))
        finish = fin

        def line_and_curve_of(self, origin, velocity, yaw_deg, nums, k=None):
            return rline, rline
    ops = {"record": (_Fixed(), 0), "coast": (ea.RayOperator(float(ctx.tick.ms), fin, n=4), 3),
           "ray0": (ea.RayOperator(float(ctx.tick.ms), fin, n=3), 0),
           "prim": (ea.PrimOperator(float(ctx.tick.ms), fin, 1, a.seed,
                                    cfg=getattr(ctx, "cfg", None)), 0),
           # the simulator-trace lines (edge_archive): free flight to the first surface, then
           # along it (surf) / arriving in its plane (tangent)
           "surf": (ea.SurfOperator(float(ctx.tick.ms), fin, 0, a.seed, ctx.scratch.core), 0),
           "tangent": (ea.SurfOperator(float(ctx.tick.ms), fin, 0, a.seed, ctx.scratch.core,
                                       kind="tangent"), 0)}
    core = ctx.scratch.core
    for i in range(core.num_envs):
        core.set_state(i, st)
    obs, *_ = core.step(np.tile(np.array([7, 3, 1, 1, 0, 0], np.int32), (core.num_envs, 1)))
    st1 = core.get_states()[0].copy()
    st1["tick"] = 0
    st1["stuck_ticks"] = 0
    obs0 = np.array(obs[0], np.float32, copy=True)
    d_start = float(gf.sample(st1["origin"].astype(np.float64)[None])[0])
    k1 = min(k0 + dur, len(rec) - 1)
    rec_gain = d_start - float(gf.sample(rec[k1]["origin"].astype(np.float64)[None])[0])
    print(f"line_bench: {Path(a.ckpt).name} ({'greedy' if a.greedy else 'sampled'}) from "
          f"{Path(a.states).name}:{a.ep} at t {a.t:.2f} s: pos "
          f"{np.round(st1['origin']).astype(int).tolist()}, |v| "
          f"{np.linalg.norm(st1['velocity']):,.0f}; {a.secs:g} s flights; the recording gains "
          f"{rec_gain:,.0f} u of route and ends at |v| {np.linalg.norm(rec[k1]['velocity']):,.0f}")
    for name, (op, choice) in ops.items():
        ctx.planner = op
        fl = ea.Flyer(ctx)
        fl.dur = dur
        arch = ea.Archive()
        nid = arch.add(st1, fl.fresh_keys(), obs0, ("R",), -1, -1, 0, 0, None)
        res = fl.fly([(nid, choice)] * min(a.n, fl.S), arch, fin, record_path=False)
        al = [r for r in res if r["end"] is not None]
        gains = np.array([d_start - float(gf.sample(r["end"]["origin"].astype(np.float64)[None])[0])
                          for r in al])
        sp = np.array([float(np.linalg.norm(r["end"]["velocity"])) for r in al])
        good = int((gains >= 0.8 * rec_gain).sum()) if len(gains) else 0
        print(f"   {name:6s}: alive {len(al)}/{len(res)} | route gained median "
              f"{(np.median(gains) if len(gains) else 0):,.0f} max {(gains.max() if len(gains) else 0):,.0f}"
              f" | >= 80% of the recording's gain: {good}/{len(res)} | end |v| median "
              f"{(np.median(sp) if len(sp) else 0):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
