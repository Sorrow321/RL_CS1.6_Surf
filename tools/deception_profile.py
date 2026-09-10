#!/usr/bin/env python3
"""deception_profile.py - where, along a reference line, does the shaping
field ask for something the physics cannot do?

At every sample of a reference line (a route .npz or a recorded trajectory)
this reports four things the goal field itself never asks:

  1. ANGLE      the angle between the field's steepest-descent direction at
                that point (a central difference of GoalField.sample on the
                field's own cell, the same quantity GoalField.descent_yaw
                uses, extended to 3D) and the reference line's TANGENT.
                0 deg means the field points exactly where the flyable line
                goes; 90 deg means it points sideways off it.
  2. SUPPORT    follow the field's own descent from that point for
                --descent-steps cells and ask, at each step, whether a
                RIDABLE surface (0.1 <= |n_z| <= 0.7) is within --reach.
                The fraction of that descent trace with nothing ridable in
                reach is the field's "over void" share.
  3. EFFICIENCY d banked per unit of path length, for the field's descent
                trace and for the reference line over the same span. The
                shaping reward is exactly scale * (-dd), and the time
                penalty is proportional to path time, so a field line with a
                HIGHER d-per-unit is a line the reward PREFERS.
  4. DIP        the reference line's own instantaneous dip depth in reward
                units (see tools/dip_report.py) at that arc position, so the
                deception profile and the tolerance profile share an x axis.

Read-only: it np.loads the baked caches next to the .bsp and never calls
build_goal_field, so it cannot trigger a bake. Pass an ABSOLUTE main-checkout
map path (CLAUDE.md's worktree trap).

    python tools/deception_profile.py \
        --map C:/RL_Surf/maps/surf_petrus_lite.bsp \
        --route C:/RL_Surf/maps/surf_petrus_lite.wrroute.npz \
        --out runs/research/cornerdiag/m1_petrus.json
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

from surfgym.goalfield import GoalField            # noqa: E402
from field_probe import Field, ballistic_speed     # noqa: E402
from dip_report import load_field, load_traj       # noqa: E402


def descent_dir(gf: GoalField, p, h=None):
    """Unit vector of steepest descent of the trilinear field at p (N,3).

    Central differences on the field's own cell - the 3D extension of
    GoalField.descent_yaw, which uses exactly this construction in xy."""
    p = np.atleast_2d(np.asarray(p, np.float64))
    h = float(gf.cell) if h is None else float(h)
    g = np.zeros_like(p)
    for k in range(3):
        e = np.zeros(3); e[k] = h
        g[:, k] = gf.sample(p + e) - gf.sample(p - e)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    out = np.where(n > 1e-9, -g / np.maximum(n, 1e-9), 0.0)
    return out


def descent_trace(gf: GoalField, p0, steps, step_u):
    """Greedy descent on the field from p0: (K,3) points, d at each."""
    pts = [np.asarray(p0, np.float64)]
    ds = [float(gf.sample(np.asarray(p0, np.float64)[None])[0])]
    for _ in range(int(steps)):
        u = descent_dir(gf, pts[-1][None])[0]
        if not np.any(u):
            break
        q = pts[-1] + u * step_u
        dq = float(gf.sample(q[None])[0])
        if not np.isfinite(dq) or dq >= gf.reach_max:
            break
        pts.append(q); ds.append(dq)
        if dq <= 1.0:
            break
    return np.asarray(pts), np.asarray(ds)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--route", help="a route .npz (key 'route')")
    ap.add_argument("--traj", help="a trajectory .jsonl instead of a route")
    ap.add_argument("--episode", default="best")
    ap.add_argument("--every", type=int, default=1,
                    help="sample every N-th route vertex / trajectory decision")
    ap.add_argument("--act-every", type=int, default=4)
    ap.add_argument("--cell", type=int, default=32)
    ap.add_argument("--reach", type=float, default=192.0)
    ap.add_argument("--descent-steps", type=int, default=24)
    ap.add_argument("--d0", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bsp = Path(a.map)
    gf = load_field(bsp, a.cell)
    fp = Field(bsp, a.cell)

    if a.route:
        z = np.load(a.route)
        pts = np.asarray(z["route"], np.float64)[::a.every]
        src = a.route
    else:
        eps = load_traj(a.traj)
        if a.episode == "best":
            k = int(np.argmin([float(gf.sample(x[:, 1:4]).min())
                               for _f, _h, x in eps]))
        else:
            k = int(a.episode)
        pts = eps[k][2][::a.act_every, 1:4][::a.every]
        src = f"{a.traj}#ep{k}"

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)

    d = gf.sample(pts).astype(np.float64)
    scale = 100.0 / float(a.d0)
    b = np.minimum.accumulate(d)
    dip = (d - b) * scale

    u = descent_dir(gf, pts)
    cosang = np.clip((u * tang).sum(1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    ang[np.linalg.norm(u, axis=1) < 1e-6] = np.nan

    rows = []
    step_u = float(a.cell)
    for i, p in enumerate(pts):
        tr, td = descent_trace(gf, p, a.descent_steps, step_u)
        sup = np.array([fp.support(q, reach=a.reach, surf_only=True)
                        for q in tr])
        hold = np.array([fp.support(q, reach=a.reach, surf_only=False)
                         for q in tr])
        void = float(np.mean(~np.isfinite(sup)))
        voidhold = float(np.mean(~np.isfinite(hold)))
        tlen = float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()) \
            if len(tr) > 1 else 0.0
        f_eff = (float(td[0] - td[-1]) / tlen) if tlen > 1e-6 else np.nan
        # the reference line's own efficiency over the SAME path length
        j = int(np.searchsorted(arc, arc[i] + max(tlen, 1e-6)))
        j = min(j, len(pts) - 1)
        rlen = float(arc[j] - arc[i])
        r_eff = (float(d[i] - d[j]) / rlen) if rlen > 1e-6 else np.nan
        # support under the reference line itself
        r_sup = fp.support(p, reach=a.reach, surf_only=True)
        r_hold = fp.support(p, reach=a.reach, surf_only=False)
        # ballistic bound over the descent trace's biggest unsupported run
        climb = float(np.sum(np.maximum(np.diff(tr[:, 2]), 0.0))) \
            if len(tr) > 1 else 0.0
        rows.append(dict(
            i=i, arc=float(arc[i]), arc_pct=100.0 * arc[i] / max(total, 1e-9),
            xyz=[float(v) for v in p], d=float(d[i]),
            d_pct=100.0 * float(d[i]) / float(a.d0),
            dip=float(dip[i]), angle=float(ang[i]) if np.isfinite(ang[i]) else None,
            field_void_surfy=void, field_void_hold=voidhold,
            field_eff=f_eff if np.isfinite(f_eff) else None,
            ref_eff=r_eff if np.isfinite(r_eff) else None,
            ref_support=(None if not np.isfinite(r_sup) else float(r_sup)),
            ref_hold=(None if not np.isfinite(r_hold) else float(r_hold)),
            field_trace_len=tlen, field_trace_climb=climb,
            field_end=[float(v) for v in tr[-1]],
            field_drop=float(tr[0, 2] - tr[-1, 2]),
        ))

    out = dict(map=str(bsp), source=src, cell=a.cell, d0=float(a.d0),
               scale=scale, reach=a.reach, descent_steps=a.descent_steps,
               total_arc=total, rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")

    ang_ok = np.array([r["angle"] for r in rows if r["angle"] is not None])
    vv = np.array([r["field_void_surfy"] for r in rows])
    fe = np.array([r["field_eff"] for r in rows if r["field_eff"] is not None])
    re = np.array([r["ref_eff"] for r in rows if r["ref_eff"] is not None])
    print(f"{src}\n  samples {len(rows)}  arc {total:,.0f} u")
    print(f"  angle(field descent, line tangent): median {np.median(ang_ok):.1f} deg "
          f"p90 {np.percentile(ang_ok, 90):.1f}  max {ang_ok.max():.1f}")
    print(f"  field descent over NO ridable surface: mean {100*vv.mean():.1f}% "
          f"of trace; {100*np.mean(vv > 0.5):.1f}% of samples are >50% void")
    print(f"  d per unit path: field {np.mean(fe):.4f}  reference {np.mean(re):.4f} "
          f"(field/ref {np.mean(fe)/max(np.mean(re),1e-9):.3f}x)")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
