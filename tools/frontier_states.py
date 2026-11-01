#!/usr/bin/env python3
"""frontier_states.py - a --spawn-states pool from a policy's OWN greedy re-recording: the states
around where each episode's CONTACT record last improved (zlr2b1, 2026-11-01).

For every dumped episode whose contact record (the --ratchet-contact test) is within --frac of
the best episode's: the states from --before seconds before the record's last improvement up to
--after-death seconds before the episode's end (the fatal fall is left out). That is the policy's
frontier - where the next manoeuvre has to start - as exact own states for --spawn-states. Nothing
here reads a place on the map.

    python tools/frontier_states.py runs/x/states.npz --field maps/X.goalc_32.npz --tick-ms 7 \\
        --out runs/x/frontier.npy [--before 8 --after-death 1.5 --stride 2 --frac 0.97]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from contact_roots import contact_mask   # noqa: E402


def main(argv=None):
    from surfgym.core import STATE_DTYPE
    from surfgym.goalfield import load_goal_field
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("states")
    ap.add_argument("--field", required=True)
    ap.add_argument("--tick-ms", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--before", type=float, default=8.0)
    ap.add_argument("--after-death", type=float, default=1.5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.97,
                    help="keep episodes whose contact record >= frac x the best episode's")
    a = ap.parse_args(argv)
    f = load_goal_field(a.field)
    z = np.load(a.states)
    eps = [np.asarray(z[k], STATE_DTYPE) for k in sorted(z.files)]
    d0 = float(f.sample(eps[0]["origin"][:1].astype(np.float64))[0])
    tps = 1000.0 / a.tick_ms
    info = []
    for st in eps:
        if len(st) < 2:
            info.append(None)
            continue
        d = f.sample(st["origin"].astype(np.float64)).astype(np.float64)
        prog = (d0 - d) / d0
        pc = np.where(contact_mask(st, a.tick_ms), prog, -np.inf)
        run = np.maximum.accumulate(pc)
        last = int(np.flatnonzero(pc >= run[-1] - 1e-12)[0]) if np.isfinite(run[-1]) else 0
        info.append((float(run[-1]), last))
    top = max(i[0] for i in info if i)
    out = []
    for k, (st, inf) in enumerate(zip(eps, info)):
        if not inf or inf[0] < a.frac * top:
            continue
        lo = max(0, inf[1] - int(a.before * tps))
        hi = max(lo + 1, len(st) - int(a.after_death * tps))
        sel = st[lo:hi:max(1, a.stride)]
        out.append(sel)
        print(f"ep{k}: contact record {100 * inf[0]:.2f}% at {inf[1] / tps:.1f}s; states "
              f"{lo / tps:.1f}..{hi / tps:.1f}s -> {len(sel)}")
    pool = np.concatenate(out)
    pool["tick"] = 0
    pool["stuck_ticks"] = 0
    np.save(a.out, pool)
    print(f"{len(pool)} frontier states -> {a.out}")


if __name__ == "__main__":
    main()
