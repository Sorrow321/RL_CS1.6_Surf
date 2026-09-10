#!/usr/bin/env python3
"""branch_table.py - the two branches at a critical point, per decision.

At a place where a policy dies, two trajectories pass through nearly the same
state and go different ways: the NAIVE branch (the failing policy's own
greedy episode) and the CORRECT branch (a finisher's, or the record's). This
lines them up at a common branch point - matched by ARC along a shared
reference route, not by wall clock, because the two arrive at different
times - and prints, per DECISION, exactly what the trainer would see:

    position, speed, vertical speed
    geodesic d, and the dip depth (d - running min) in reward units
    the ridable support below (field_probe: 0.1 <= |n_z| <= 0.7 within reach)
    the SHAPING reward   scale * (d_prev - d),  scale = 100/d0
    the TIME penalty     -time_pen_tick * act_every
    the per-decision reward and its cumulative sum
    the discounted cumulative sum from the branch point

and then the TOLERANCE INTEGRAL for each branch: how long and how much the
branch is behind the other one, and what fraction of the eventual payoff
survives the trainer's GAE weighting at that delay
((gamma^act_every * lambda)^k).

    python tools/branch_table.py --map C:/RL_Surf/maps/surf_petrus_lite.bsp \
        --route C:/RL_Surf/maps/surf_petrus_lite.wrroute.npz --d0 35636.65625 \
        --naive C:/RL_Surf_pr1/runs/prRATCH/traj_2757754880.jsonl \
        --correct runs/research/cornerdiag/wr/petrus_wr.jsonl \
        --branch-arc-pct 17.1 --window 6 --out runs/research/cornerdiag/m2_petrus.json

Read-only, no GPU, no bake (the field comes out of the baked cache).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from dip_report import load_field, load_traj      # noqa: E402
from field_probe import Field                      # noqa: E402
from surfgym.route import ArcProgress              # noqa: E402


def arc_of(pts, xyz, spacing=128.0, corridor=6000.0, window=16):
    ap = ArcProgress(np.asarray(pts, np.float64), spacing,
                     corridor=corridor, window=window)
    ap.reset(xyz[:1])
    out = [float(ap.arc[0])]
    for i in range(1, len(xyz)):
        ap.advance(xyz[i:i + 1])
        out.append(float(ap.arc[0]))
    _a, off = ap.locate(xyz)
    return np.asarray(out), np.asarray(off, float)


def pick(traj, episode, gf, act_every):
    eps = load_traj(traj)
    if episode == "best":
        k = int(np.argmin([float(gf.sample(a[:, 1:4]).min())
                           for _f, _h, a in eps]))
    else:
        k = int(episode)
    foot, hdr, a = eps[k]
    return k, foot, hdr, a[::act_every]


def branch(name, rows, gf, fp, pts, d0, act_every, tick_ms, time_pen_tick,
           gamma_tick, lam, arc0_pct, window_s):
    xyz = rows[:, 1:4]
    arcs, off = arc_of(pts, xyz)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.cumsum(seg)[-1])
    tgt = arc0_pct / 100.0 * total
    i0 = int(np.argmax(arcs >= tgt)) if (arcs >= tgt).any() else 0
    dt = act_every * tick_ms / 1000.0
    n = min(len(rows) - i0, int(round(window_s / dt)) + 1)
    sl = slice(i0, i0 + n)

    d = gf.sample(xyz[sl]).astype(np.float64)
    scale = 100.0 / float(d0)
    b = np.minimum.accumulate(d)
    dip = (d - b) * scale
    shap = np.concatenate([[0.0], scale * (d[:-1] - d[1:])])
    tp = -float(time_pen_tick) * int(act_every)
    rew = shap + tp
    rew[0] = 0.0
    cum = np.cumsum(rew)
    gd = float(gamma_tick) ** int(act_every)
    disc = np.cumsum(rew * (gd ** np.arange(n)))
    spd = np.linalg.norm(rows[sl, 4:7], axis=1)
    sup, hold, fdrop = [], [], []
    for p in xyz[sl]:
        s = fp.support(p, reach=192.0, surf_only=True)
        h = fp.support(p, reach=192.0, surf_only=False)
        sup.append(None if not np.isfinite(s) else float(s))
        hold.append(None if not np.isfinite(h) else float(h))
        fdrop.append(fp.floor_drop(p))
    return dict(
        name=name, i0=int(i0), n=int(n), dt=dt, scale=scale,
        arc_pct=(100.0 * arcs[sl] / total).tolist(),
        off_line=off[sl].tolist(),
        t=(np.arange(n) * dt).tolist(),
        xyz=xyz[sl].tolist(), speed=spd.tolist(), vz=rows[sl, 6].tolist(),
        d=d.tolist(), dip=dip.tolist(),
        shaping=shap.tolist(), time_pen=tp,
        reward=rew.tolist(), cum=cum.tolist(), disc_cum=disc.tolist(),
        support=sup, hold=hold, floor_drop=fdrop,
        gamma_dec=gd, lam=float(lam),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--d0", type=float, required=True)
    ap.add_argument("--naive", required=True)
    ap.add_argument("--naive-episode", default="best")
    ap.add_argument("--naive-act-every", type=int, default=4)
    ap.add_argument("--naive-tick-ms", type=float, default=10.0)
    ap.add_argument("--correct", required=True)
    ap.add_argument("--correct-episode", default="best")
    ap.add_argument("--correct-act-every", type=int, default=4)
    ap.add_argument("--correct-tick-ms", type=float, default=10.0)
    ap.add_argument("--branch-arc-pct", type=float, required=True)
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--time-pen-tick", type=float, default=0.005)
    ap.add_argument("--correct-time-pen-tick", type=float, default=None,
                    help="the CORRECT branch's own time_pen_tick when it was "
                         "recorded at a different tick_ms (the trainer scales "
                         "it so reward per SECOND is equal); defaults to "
                         "--time-pen-tick")
    ap.add_argument("--gamma-tick", type=float, default=0.9995)
    ap.add_argument("--correct-gamma-tick", type=float, default=None)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gf = load_field(Path(a.map), a.cell)
    fp = Field(Path(a.map), a.cell)
    pts = np.asarray(np.load(a.route)["route"], np.float64)

    kn, fn, hn, rn = pick(a.naive, a.naive_episode, gf, a.naive_act_every)
    kc, fc, hc, rc = pick(a.correct, a.correct_episode, gf, a.correct_act_every)
    B = {}
    B["naive"] = branch("naive", rn, gf, fp, pts, a.d0, a.naive_act_every,
                        a.naive_tick_ms, a.time_pen_tick, a.gamma_tick,
                        a.gae, a.branch_arc_pct, a.window)
    B["correct"] = branch("correct", rc, gf, fp, pts, a.d0,
                          a.correct_act_every, a.correct_tick_ms,
                          (a.correct_time_pen_tick
                           if a.correct_time_pen_tick is not None
                           else a.time_pen_tick),
                          (a.correct_gamma_tick
                           if a.correct_gamma_tick is not None
                           else a.gamma_tick), a.gae,
                          a.branch_arc_pct, a.window)
    B["meta"] = dict(map=a.map, route=a.route, d0=a.d0,
                     branch_arc_pct=a.branch_arc_pct, window=a.window,
                     naive=f"{a.naive}#ep{kn}", correct=f"{a.correct}#ep{kc}",
                     naive_end=fn.get("end"), correct_end=fc.get("end"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(B), encoding="utf-8")

    for k in ("naive", "correct"):
        b = B[k]
        print(f"\n=== {k}: {B['meta'][k]}   branch at arc {a.branch_arc_pct}%")
        print("  t     arc%   off    x      y      z     spd    vz     d    "
              "  dip   shap   rew    cum   disc  supp  floor")
        for i in range(b["n"]):
            s = b["support"][i]
            print("  %5.2f %5.1f %5.0f %6.0f %6.0f %6.0f %6.0f %6.0f %7.0f "
                  "%6.3f %6.3f %6.3f %6.2f %6.2f %5s %6.0f"
                  % (b["t"][i], b["arc_pct"][i], b["off_line"][i],
                     b["xyz"][i][0], b["xyz"][i][1], b["xyz"][i][2],
                     b["speed"][i], b["vz"][i], b["d"][i], b["dip"][i],
                     b["shaping"][i], b["reward"][i], b["cum"][i],
                     b["disc_cum"][i],
                     ("%.0f" % s) if s is not None else "NONE",
                     b["floor_drop"][i]))
    n, c = B["naive"], B["correct"]
    m = min(n["n"], c["n"])
    gap = np.asarray(n["cum"][:m]) - np.asarray(c["cum"][:m])
    print(f"\ncumulative reward: naive - correct over the window")
    for i in range(0, m, max(1, m // 12)):
        k = i
        gw = (n["gamma_dec"] * n["lam"]) ** k
        print(f"  t {n['t'][i]:5.2f}s  naive {n['cum'][i]:7.3f}  correct "
              f"{c['cum'][i]:7.3f}  naive-correct {gap[i]:+7.3f}  "
              f"gae_w(k={k}) {gw:.4g}")
    print(f"  max naive lead {gap.max():+.3f} at t "
          f"{n['t'][int(np.argmax(gap))]:.2f}s")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
