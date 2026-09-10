#!/usr/bin/env python3
"""forced_branch.py - WON'T or CAN'T? Steer the policy onto the other branch
for a moment, then let go.

A policy that dies at one place either (a) cannot execute the surviving
branch - control precision, and no reward or tolerance mechanism helps - or
(b) can execute it perfectly well and simply never chooses it, in which case
the branch is a credit / exploration problem. The two are distinguished by
one experiment: put the policy ON the other branch and release it.

This drives ``tools/record_ckpt.py --nudge-tick/--nudge-hold/--nudge-yaw``
over a grid. The intervention changes exactly one number: for ``hold``
decisions starting at physics tick ``tick``, the policy's own yaw COMMAND is
offset by ``yaw`` degrees. Everything else - the observation, every other
action head, the pitch, the sampling draw - is the policy's own, and after
the window the override stops and the policy flies free. Nothing from a
champion, a demo or a route enters the intervention: the sweep is over
offsets and the surviving offset, if any, is DISCOVERED.

Every cell is then scored with the same rulers an arm is scored with:
order-only corridor arc (tools/eval_honesty's rule), the geodesic ground
banked in reward units, the episode length, and finishes.

    python tools/forced_branch.py C:/RL_Surf_pr1/runs/prRATCH/ckpt_latest.pt \
        --map C:/RL_Surf/maps/surf_petrus_lite.bsp \
        --route C:/RL_Surf/maps/surf_petrus_lite.wrroute.npz \
        --d0 35636.65625 --at-ticks 676 --holds 5,12,25,50 \
        --offsets -60,-40,-25,-15,0,15,25,40,60 \
        --greedy-eps 2 --sampled-eps 6 \
        --out runs/research/cornerdiag/m4_petrus

Absolute main-checkout map path only (CLAUDE.md's worktree bake trap).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from dip_report import load_field, load_traj      # noqa: E402
from surfgym.route import ArcProgress             # noqa: E402


def order_arc(xyz, pts, spacing, corridor=1500.0, window=16):
    ap = ArcProgress(np.asarray(pts, np.float64), spacing,
                     corridor=corridor, window=window)
    p = np.asarray(xyz, np.float64)
    ap.reset(p[:1])
    best = float(ap.arc[0])
    for k in range(1, len(p)):
        ap.advance(p[k:k + 1])
        best = max(best, float(ap.arc[0]))
    return best


def score(path, gf, pts, spacing, d0, corridor):
    out = []
    for foot, hdr, a in load_traj(path):
        d = gf.sample(a[:, 1:4]).astype(np.float64)
        out.append(dict(
            end=str(foot.get("end", "")), ticks=int(foot.get("ticks", len(a))),
            arc=order_arc(a[:, 1:4], pts, spacing, corridor),
            banked=float((d[0] - d.min()) * 100.0 / d0),
            end_xyz=[float(v) for v in a[-1, 1:4]],
            speed_end=float(np.linalg.norm(a[-1, 4:7])),
            finished=bool(d.min() <= 150.0),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--map", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--d0", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--at-ticks", default="676")
    ap.add_argument("--holds", default="5,12,25,50")
    ap.add_argument("--offsets", default="-60,-40,-25,-15,0,15,25,40,60")
    ap.add_argument("--mode", choices=["offset", "abs", "vel"],
                    default="offset",
                    help="offset: add --offsets to the policy's OWN yaw "
                         "target for the window (an offset follows the turn "
                         "the policy is already making). abs: HOLD the yaw "
                         "target at each of --offsets as an ABSOLUTE world "
                         "heading - which is what 'keep going straight for K "
                         "decisions' actually is. vel: rotate the BODY's "
                         "horizontal velocity once at --at-ticks, the state "
                         "the policy would have been in on the other line")
    ap.add_argument("--greedy-eps", type=int, default=2)
    ap.add_argument("--sampled-eps", type=int, default=6)
    ap.add_argument("--corridor", type=float, default=1500.0)
    ap.add_argument("--ep-ticks", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gf = load_field(Path(a.map))
    z = np.load(a.route)
    pts = np.asarray(z["route"], np.float64)
    spacing = float(z["spacing"]) if "spacing" in z.files else 128.0
    total = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    ticks = [int(x) for x in a.at_ticks.split(",") if x.strip()]
    holds = [int(x) for x in a.holds.split(",") if x.strip()]
    offs = [float(x) for x in a.offsets.split(",") if x.strip()]

    rows, t0 = [], time.perf_counter()
    for T in ticks:
        for hold in holds:
            for off in offs:
                for mode, neps in (("greedy", a.greedy_eps),
                                   ("sampled", a.sampled_eps)):
                    if neps <= 0:
                        continue
                    tag = f"{a.mode}_t{T}_h{hold}_o{off:+g}_{mode}"
                    jl = out / f"{tag}.jsonl"
                    cmd = [sys.executable, str(ROOT / "tools" / "record_ckpt.py"),
                           str(a.ckpt), "--map", str(a.map),
                           "--episodes", str(neps), "--seed", "0",
                           "--out", str(jl)]
                    if mode == "sampled":
                        cmd.append("--stochastic")
                    if a.ep_ticks:
                        cmd += ["--ep-ticks", str(a.ep_ticks)]
                    if a.mode == "vel":
                        if off != 0.0:
                            cmd += ["--nudge-tick", str(T), "--nudge-hold",
                                    "0", "--nudge-vel", f"{off:g}"]
                    elif a.mode == "abs":
                        if hold > 0:
                            cmd += ["--nudge-tick", str(T), "--nudge-hold",
                                    str(hold), "--nudge-yaw-abs", f"{off:g}"]
                    elif off != 0.0 and hold > 0:
                        cmd += ["--nudge-tick", str(T), "--nudge-hold",
                                str(hold), "--nudge-yaw", f"{off:g}"]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  {tag}: FAILED\n{r.stdout[-800:]}"
                              f"\n{r.stderr[-800:]}")
                        continue
                    eps = score(jl, gf, pts, spacing, a.d0, a.corridor)
                    arcs = np.array([e["arc"] for e in eps])
                    tk = np.array([e["ticks"] for e in eps])
                    rows.append(dict(tick=T, hold=hold, offset=off, mode=mode,
                                     kind=a.mode,
                                     n=len(eps), eps=eps,
                                     arc_max=float(arcs.max()),
                                     arc_mean=float(arcs.mean()),
                                     arc_pct_max=100.0 * float(arcs.max()) / total,
                                     ticks_max=int(tk.max()),
                                     ticks_mean=float(tk.mean()),
                                     banked_max=float(max(e["banked"] for e in eps)),
                                     fin=int(sum(e["finished"] for e in eps))))
                    print(f"  t{T} hold {hold:3d} off {off:+6.1f} {mode:8s} "
                          f"arc max {arcs.max():8,.0f} ({100*arcs.max()/total:5.1f}%)"
                          f" mean {arcs.mean():8,.0f}  ticks max {tk.max():5d} "
                          f"mean {tk.mean():7.1f}  banked "
                          f"{max(e['banked'] for e in eps):6.2f}  "
                          f"fin {sum(e['finished'] for e in eps)}/{len(eps)}")
    (out / f"forced_branch_{a.mode}.json").write_text(json.dumps(dict(
        ckpt=str(a.ckpt), map=str(a.map), route=str(a.route), d0=a.d0,
        total_arc=total, rows=rows)), encoding="utf-8")
    print(f"wrote {out}/forced_branch_{a.mode}.json  "
          f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
