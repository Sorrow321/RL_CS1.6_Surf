#!/usr/bin/env python3
"""reward_replay.py - replay recorded episodes through BOTH race shaping
computations and print what each one paid.

The pre-flight check for ``--race-ratchet`` (docs/race_ratchet.md). A one
hour arm cannot tell you whether the mechanism does what it claims - it can
only tell you what the policy did afterwards. This does the deterministic
half in a second, on trajectories that already exist:

  * **stock** - the signed potential difference the control trains on,
    ``scale * (d_{t-1} - d_t)`` per decision, clipped at ``max_step*every``;
  * **ratchet** - ``scale * (b_{t-1} - b_t)`` with ``b`` the episode's
    running minimum of ``d``, restarted at every episode's own start.

Both are read off the recorded POSITIONS through ``goalfield.GoalField``, so
this is the same geodesic the trainer shapes on and nothing here needs a
GPU, a map or a running box.

Per episode it prints:

  ``shaping``   total shaping paid over the episode, both schemes;
  ``detour``    what the RISING stretches cost - every call where d grows.
                ``worst`` is the single longest contiguous rise, which on
                cannonball is the final ramp the ledger charges -4.02 for;
  ``revisit``   what re-closing ground the episode had already lost pays -
                the stock term's refund, which the ratchet deletes.

Expected under the ratchet: ``detour`` is exactly 0 and ``revisit`` is
exactly 0, while the total stays ``scale * (d_start - d_best)``.

    python tools/reward_replay.py runs/research/xARC/traj_*.jsonl
    python tools/reward_replay.py --every 3 --per-episode runs/.../traj_*.jsonl
"""
import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from surfgym.goalfield import GoalField        # noqa: E402
from surfgym.route import episodes_from_traj   # noqa: E402


def load_field(path):
    """Load a baked goal field straight out of its cache npz.

    The signature check the trainer does needs the .bsp; this is a read-only
    diagnostic over recorded positions, so the file is taken at face value
    and its name is printed with every number that came out of it."""
    z = np.load(path, allow_pickle=False)
    grid = z["grid"].astype(np.float32) * float(z["quant"])
    return GoalField(grid, z["mins"], float(z["cell"]), float(z["reach_max"]))


def replay(d, scale, every, max_step):
    """-> dict of what each scheme pays over one episode's decision series.

    ``d`` is the geodesic at each DECISION (already subsampled), so the
    per-call clip is the trainer's ``max_step * every``.
    """
    clip = max_step * every
    dd = np.diff(d)                       # d_t - d_{t-1}
    stock = np.clip(-dd, -clip, clip)     # progress positive, like the trainer
    b = np.minimum.accumulate(d)
    rat = np.clip(-np.diff(b), -clip, clip)   # >= 0 by construction

    rising = dd > 0.0                     # the calls that LOSE ground
    # the single longest contiguous rise: on cannonball that is the ramp
    worst_i = worst_n = 0
    i = 0
    while i < len(rising):
        if rising[i]:
            j = i
            while j < len(rising) and rising[j]:
                j += 1
            if j - i > worst_n:
                worst_n, worst_i = j - i, i
            i = j
        else:
            i += 1
    worst = slice(worst_i, worst_i + worst_n)

    # a REVISIT: closing ground while still above the episode's own record,
    # i.e. re-earning what a rise already refunded
    closing = dd < 0.0
    above = d[1:] >= b[:-1] - 1e-9
    revisit = closing & above

    return {
        "n": len(d),
        "d0": float(d[0]),
        "dmin": float(b[-1]),
        "stock": float(stock.sum() * scale),
        "ratchet": float(rat.sum() * scale),
        "stock_detour": float(stock[rising].sum() * scale),
        "ratchet_detour": float(rat[rising].sum() * scale),
        "stock_worst": float(stock[worst].sum() * scale) if worst_n else 0.0,
        "ratchet_worst": float(rat[worst].sum() * scale) if worst_n else 0.0,
        "worst_n": int(worst_n),
        "worst_rise": float(d[worst_i + worst_n] - d[worst_i]) if worst_n else 0.0,
        "stock_revisit": float(stock[revisit].sum() * scale),
        "ratchet_revisit": float(rat[revisit].sum() * scale),
        "budget": float((d[0] - b[-1]) * scale),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="+", help="traj_*.jsonl (globs allowed)")
    ap.add_argument("--field", default=str(ROOT / "maps"
                                           / "surf_src_cannonball.goal_32.npz"),
                    help="the baked goal field npz the trainer shapes on")
    ap.add_argument("--d0", type=float, default=198380.0,
                    help="the map's start geodesic; scale = 100/d0 * shaping")
    ap.add_argument("--shaping", type=float, default=1.0,
                    help="--race-shaping, the scale multiplier")
    ap.add_argument("--every", type=int, default=3,
                    help="act_every: rows are per TICK, the reward is per "
                         "DECISION (the stuck checkpoint is 3)")
    ap.add_argument("--max-step", type=float, default=100.0)
    ap.add_argument("--per-episode", action="store_true",
                    help="one line per episode instead of a per-file summary")
    args = ap.parse_args()

    field = load_field(args.field)
    scale = 100.0 / args.d0 * args.shaping
    paths = []
    for p in args.traj:
        paths.extend(sorted(glob.glob(p)) or [p])

    print(f"field {Path(args.field).name}   scale {scale:.6g} "
          f"(100/{args.d0:,.0f} x {args.shaping:g})   act_every {args.every}")
    print()
    grand = []
    for path in paths:
        eps = episodes_from_traj(path)
        rows = []
        for e in eps:
            if len(e) < 2 * args.every + 2:
                continue
            xyz = np.ascontiguousarray(e[::args.every, 1:4], np.float64)
            d = field.sample(xyz).astype(np.float64)
            rows.append(replay(d, scale, args.every, args.max_step))
        if not rows:
            continue
        grand.extend(rows)
        name = Path(path).name
        if args.per_episode:
            print(f"== {name}   {len(rows)} episodes")
            print("   ep    d0        dmin      shaping stock/ratchet   "
                  "detour stock/ratchet   worst rise (calls)   "
                  "revisit stock/ratchet")
            for k, r in enumerate(rows):
                print(f"   {k:<4d} {r['d0']:>9,.0f} {r['dmin']:>9,.0f}   "
                      f"{r['stock']:>8.3f} / {r['ratchet']:<8.3f}  "
                      f"{r['stock_detour']:>8.3f} / {r['ratchet_detour']:<6.3f}  "
                      f"{r['stock_worst']:>7.3f} ({r['worst_rise']:>8,.0f}u "
                      f"in {r['worst_n']:>4d})  "
                      f"{r['stock_revisit']:>7.3f} / {r['ratchet_revisit']:<6.3f}")
            print()
        else:
            m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            print(f"{name:<28s} {len(rows):>3d} eps  "
                  f"d0 {m['d0']:>8,.0f} -> dmin {m['dmin']:>8,.0f}   "
                  f"shaping {m['stock']:>7.3f} / {m['ratchet']:<7.3f}  "
                  f"detour {m['stock_detour']:>7.3f} / {m['ratchet_detour']:<5.3f}  "
                  f"worst {m['stock_worst']:>6.3f} ({m['worst_rise']:>7,.0f}u)  "
                  f"revisit {m['stock_revisit']:>6.3f} / "
                  f"{m['ratchet_revisit']:<5.3f}")

    if grand:
        m = {k: float(np.mean([r[k] for r in grand])) for k in grand[0]}
        print()
        print(f"ALL {len(grand)} episodes, per-episode means:")
        print(f"  shaping paid       stock {m['stock']:>8.3f}   "
              f"ratchet {m['ratchet']:>8.3f}   "
              f"(scale*(d0-dmin) = {m['budget']:.3f})")
        print(f"  charged on rises   stock {m['stock_detour']:>8.3f}   "
              f"ratchet {m['ratchet_detour']:>8.3f}")
        print(f"  longest single rise stock {m['stock_worst']:>7.3f}   "
              f"ratchet {m['ratchet_worst']:>8.3f}   "
              f"({m['worst_rise']:,.0f}u over {m['worst_n']:.0f} calls)")
        print(f"  paid for revisits  stock {m['stock_revisit']:>8.3f}   "
              f"ratchet {m['ratchet_revisit']:>8.3f}")


if __name__ == "__main__":
    main()
