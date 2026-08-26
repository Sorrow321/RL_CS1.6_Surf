"""Pick, trim and spine-ify one round's best greedy rollout (xLOOP).

One link of the self-improvement loop: given the 20 greedy evals recorded
from a round's final checkpoint (tools/record_ckpt.py --episodes 20
--dump-states), choose the episode that reaches the MINIMUM geodesic
distance to goal, trim its fall, and write the surviving prefix as the
time-ordered STATE_DTYPE .npy that the next round spawns from.

Why the states and not the .jsonl: a trajectory row is lossy (no
basevelocity, no duck bookkeeping, onground flattened to 0/1), and a
spawn pool copies whole states. --dump-states hands us the real thing,
episode-aligned with the trajectory.

The trim is tools/pick_selfline.py's contact_cut, imported rather than
re-derived: every non-finishing episode ends in a fall, and spawning
mid-fall seeds states no policy can recover from. The rule is the last
tick whose vertical acceleration departs from the gravity step - the
last tick the map pushed back - and it needs no champion, no route and
no goal field. A FINISHING episode is not trimmed.

Selection is by minimum d, as the experiment specifies. That number is
dive-flattered by construction (this map's goal field has a mid-route
minimum at the wall, and goal-adjacent airspace below the ramp scores
well), so the corridor MAX of the chosen episode is reported alongside
it - if a round's pick is a dive, the summary says so rather than
hiding it. The trim is what keeps a dive from poisoning the next
round's spawns.

    python tools/loop_spine.py --states s.npz --traj e.jsonl \\
        --ckpt runs/xLOOP/round_0/ckpt_final.pt --out spine.npy \\
        --summary-out pick.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch

import beam_tas
from eval_honesty import corridor_progress, load_route
from pick_selfline import contact_cut

FINISH_PAD = 64.0          # eval_honesty's finish tolerance


def main():
    ap = argparse.ArgumentParser(description="pick+trim+spine for xLOOP")
    ap.add_argument("--states", required=True, help="--dump-states .npz")
    ap.add_argument("--traj", default=None, help="matching .jsonl (optional)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map", default="C:/RL_Surf/maps/surf_src_cannonball.bsp")
    ap.add_argument("--route",
                    default="C:/RL_Surf/maps/surf_src_cannonball.route.npz")
    ap.add_argument("--out", required=True, help="spine .npy")
    ap.add_argument("--summary-out", default=None, help="pick summary .json")
    ap.add_argument("--corridor", type=float, default=1500.0)
    ap.add_argument("--contact-tol", type=float, default=1.0)
    args = ap.parse_args()

    cfg = (torch.load(args.ckpt, map_location="cpu", weights_only=False)
           .get("config") or {})
    core = beam_tas.build_sim(cfg, args.map, 1,
                              int(cfg.get("ep_ticks", 12000)))
    from surfgym.goalfield import build_goal_field
    from surfgym.zones import load_zones
    zones = load_zones(core.bsp_path)
    gcell = float(cfg.get("goal_cell") or cfg.get("lidar_cell") or 32.0)
    gf = build_goal_field(core, zones["end"], cell=gcell)
    lo = np.asarray(zones["end"]["mins"], np.float64)
    hi = np.asarray(zones["end"]["maxs"], np.float64)

    z = np.load(args.states, allow_pickle=False)
    eps = [z[k] for k in sorted(z.files)]
    if not eps:
        raise SystemExit(f"no episodes in {args.states}")

    pts, spacing = load_route(Path(args.route))
    rows = []
    for i, e in enumerate(eps):
        o = np.asarray(e["origin"], np.float64)
        d = gf.sample(o).astype(np.float64)
        fin = bool(np.any(np.all((o >= lo - FINISH_PAD)
                                 & (o <= hi + FINISH_PAD), axis=1)))
        corr, _off = corridor_progress(o.astype(np.float32), pts, spacing,
                                       args.corridor)
        rows.append({"ep": i, "ticks": int(len(e)), "min_d": float(d.min()),
                     "end_d": float(d[-1]), "finished": fin,
                     "corridor": float(corr)})
    best = min(rows, key=lambda r: r["min_d"])
    print(f"{len(eps)} episodes; min_d {min(r['min_d'] for r in rows):,.0f}"
          f"..{max(r['min_d'] for r in rows):,.0f}u, "
          f"{sum(r['finished'] for r in rows)} finished")
    for r in sorted(rows, key=lambda r: r["min_d"])[:5]:
        print(f"  ep{r['ep']:3d} {r['ticks']:6d}t  min_d {r['min_d']:9,.0f}u"
              f"  corridor {r['corridor']:9,.0f}u"
              f"{'  FINISH' if r['finished'] else ''}")

    e = eps[best["ep"]]
    o = np.asarray(e["origin"], np.float64)
    v = np.asarray(e["velocity"], np.float64)
    if best["finished"]:
        cut, g = len(e) - 1, 0.0
        why = "finisher: no trim"
    else:
        # contact_cut reads [tick, x, y, z, vx, vy, vz, ...]: column 6 = vz
        t = np.arange(len(e), dtype=np.float64)
        packed = np.column_stack([t, o, v, np.asarray(e["yaw"], np.float64)])
        cut, g = contact_cut(packed, args.contact_tol)
        why = (f"trimmed at tick {cut} of {len(e) - 1} (last tick the map "
               f"pushed back; gravity step {g:g} u/tick^2), "
               f"{(len(e) - 1 - cut) / 100:.2f}s of fall dropped")
    spine = np.ascontiguousarray(e[:cut + 1]).copy()
    d_spine = gf.sample(np.asarray(spine["origin"], np.float64))
    print(f"chosen ep{best['ep']}: {why}")
    print(f"  spine {len(spine)} states, d {d_spine[0]:,.0f} -> "
          f"{d_spine[-1]:,.0f}u (min along spine {d_spine.min():,.0f}u), "
          f"corridor MAX {best['corridor']:,.0f}u")
    np.save(args.out, spine)
    print(f"spine -> {args.out}")

    summary = {"episodes": len(eps), "chosen_ep": best["ep"],
               "chosen_min_d": best["min_d"],
               "chosen_corridor": best["corridor"],
               "chosen_finished": best["finished"],
               "chosen_ticks": best["ticks"],
               "any_finished": int(sum(r["finished"] for r in rows)),
               "best_corridor_any": max(r["corridor"] for r in rows),
               "spine_len": int(len(spine)),
               "spine_end_d": float(d_spine[-1]),
               "spine_min_d": float(d_spine.min()),
               "trim_ticks_dropped": int(len(e) - 1 - cut),
               "gravity_step": float(g), "spine": str(args.out)}
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
