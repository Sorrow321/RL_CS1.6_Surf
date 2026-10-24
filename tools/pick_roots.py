#!/usr/bin/env python3
"""pick_roots.py - Go-Explore roots from the policy's OWN greedy line (pbr2b1, 2026-10-24).

Reads a record_ckpt --dump-states .npz (ep_0000, ep_0001, ... STATE_DTYPE per tick) and, per
episode, finds the tick where it was deepest on a goal field (its frontier), keeps the ticks
from --window HI to LO seconds BEFORE it (the approach to the frontier: where the line commits),
and writes up to --n of the kept states, evenly spaced, as one STATE_DTYPE .npy with clocks
zeroed - the --roots-spine input of tools/explore_phase1.py. Generic: the window is in seconds
before the policy's own frontier, the recipe's --respawn-greedy-window rule; nothing is read off
the map beyond the shaping field the trainer already uses.

    python tools/pick_roots.py runs/x/greedy_states.npz --field maps/X.goalc_32.npz \\
        --window 0.25 4 --n 64 --out runs/x/roots.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from surfgym.core import STATE_DTYPE  # noqa: E402


def pick(eps, dist, window=(0.25, 4.0), n: int = 64, tick_s: float = 0.01,
         max_d: float = float("inf")) -> np.ndarray:
    """``eps``: per-episode STATE_DTYPE arrays; ``dist``: origins (k, 3) -> field distance.
    ``max_d``: only episodes whose deepest distance is at most this (the ones that reached the
    frontier; default all)."""
    lo, hi = (int(round(float(w) / tick_s)) for w in window)
    rows = []
    for st in eps:
        st = np.asarray(st, STATE_DTYPE)
        if not len(st):
            continue
        d = np.asarray(dist(st["origin"].astype(np.float64)), np.float64)
        b = int(np.argmin(d))
        if d[b] > max_d:
            continue
        rows.append(st[max(0, b - hi):max(0, b - lo) + 1])
    allr = np.concatenate(rows) if rows else np.zeros(0, STATE_DTYPE)
    if len(allr) > n:
        allr = allr[np.linspace(0, len(allr) - 1, n).round().astype(int)]
    out = allr.copy()
    out["tick"] = 0
    out["stuck_ticks"] = 0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump")
    ap.add_argument("--field", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--window", type=float, nargs=2, default=(0.25, 4.0),
                    help="seconds before each episode's deepest tick: LO HI")
    ap.add_argument("--tick-ms", type=float, default=10.0)
    ap.add_argument("--max-d", type=float, default=float("inf"),
                    help="only episodes whose deepest field distance is at most this")
    a = ap.parse_args(argv)
    from surfgym.goalfield import GoalField
    zf = np.load(a.field, allow_pickle=False)
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    z = np.load(a.dump, allow_pickle=False)
    eps = [z[k] for k in sorted(z.files) if k.startswith("ep_")]
    r = pick(eps, f.sample, tuple(a.window), a.n, a.tick_ms / 1000.0, a.max_d)
    if not len(r):
        raise SystemExit("no states in the window")
    np.save(a.out, r)
    o = r["origin"]
    d = f.sample(o.astype(np.float64))
    print(f"{len(r)} roots from {len(eps)} episodes -> {a.out}; d {d.min():,.0f}..{d.max():,.0f}; "
          f"x {o[:, 0].min():.0f}..{o[:, 0].max():.0f} y {o[:, 1].min():.0f}..{o[:, 1].max():.0f} "
          f"z {o[:, 2].min():.0f}..{o[:, 2].max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
