#!/usr/bin/env python3
"""contact_roots.py - Go-Explore roots from a policy's OWN greedy states: its CONTACT ticks after
its contact record stopped improving (zlr2b1, 2026-11-01).

The goal field is a voxel BFS that flies through open air, so a policy's progress record can be
set by ballistic apexes the map never supports (surf_za_loop's incumbent: 64.7% all-state, 38.9%
at contacts). The ticks where the map actually holds the player - walkable ground, or a surfable
face pushing vz back up (the --ratchet-contact / --surf-bonus test, the -sv_maxvelocity clamp
excluded) - are where its next manoeuvre has to start. After the episode's CONTACT record last
improved, those ticks are where it is stuck; they are the roots.

Input: a tools/record_ckpt.py --dump-states .npz (keys ep_0000, ...; full STATE_DTYPE per tick).
Output: a STATE_DTYPE .npy of roots (explore_phase1.py --roots-spine) and a summary. Nothing here
reads a place on the map: the rule is the policy's own contact history.

    python tools/contact_roots.py runs/x/states.npz --field maps/<map>.goalc_32.npz \
        --tick-ms 7 --stride 10 --out runs/x/contact_roots.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def contact_mask(st: np.ndarray, tick_ms: float, gravity: float = 800.0,
                 vmax: float = 3500.0, tol: float = 1.0) -> np.ndarray:
    """Per tick of one episode: did the map hold the player up during the tick that starts here
    (ground at its start, or vz gave back more than free fall by its end)? The last tick has no
    successor and reads its ground flag alone."""
    og = np.asarray(st["onground"]) != -1
    vz = np.asarray(st["velocity"], np.float64)[:, 2]
    dev = np.zeros(len(st))
    dev[:-1] = (vz[1:] - vz[:-1]) + gravity * tick_ms / 1000.0
    surf = (dev > tol) & (vz > -(vmax - 0.5))
    surf[-1] = False
    return og | surf


def pick(eps, field, d0: float, tick_ms: float, stride: int, min_after_s: float = 0.0,
         min_record_frac: float = 0.0):
    """-> (roots, rows of summary) from the episodes' states. An episode whose contact record
    is below min_record_frac x the best episode's contributes no roots (a greedy episode that
    stalled at the start is not where the policy is stuck)."""
    out, rows = [], []
    recs = []
    for st in eps:
        if len(st) < 2:
            recs.append(-np.inf)
            continue
        c = contact_mask(st, tick_ms)
        d = field.sample(np.asarray(st["origin"], np.float64)).astype(np.float64)
        recs.append(float(np.max(np.where(c, (d0 - d) / d0, -np.inf))))
    top = max(recs) if recs else -np.inf
    for k, st in enumerate(eps):
        if recs[k] < min_record_frac * top:
            rows.append((k, len(st), recs[k], -1, 0, 0, float("nan"), float("nan"),
                         float("nan")))
            continue
        if len(st) < 2:
            continue
        c = contact_mask(st, tick_ms)
        d = field.sample(np.asarray(st["origin"], np.float64)).astype(np.float64)
        prog = (d0 - d) / d0
        pc = np.where(c, prog, -np.inf)
        run = np.maximum.accumulate(pc)
        best = float(run[-1])
        # the last tick at which the contact record improved
        imp = np.flatnonzero(pc >= run - 1e-12)
        t_last = int(imp[-1]) if len(imp) else 0
        t0 = max(t_last + 1, int(min_after_s * 1000.0 / tick_ms))
        idx = np.flatnonzero(c[t0:]) + t0
        sel = idx[::max(1, stride)]
        out.append(st[sel])
        rows.append((k, len(st), best, t_last, len(idx), len(sel),
                     float(prog.max()), float(prog[sel].min()) if len(sel) else float("nan"),
                     float(prog[sel].max()) if len(sel) else float("nan")))
    return (np.concatenate(out) if out else np.zeros(0)), rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("states", help="record_ckpt --dump-states .npz")
    ap.add_argument("--field", required=True, help="the goal field .npz (progress read-out)")
    ap.add_argument("--d0", type=float, default=None,
                    help="the start distance (default: the field at the first episode's tick 0)")
    ap.add_argument("--tick-ms", type=float, required=True)
    ap.add_argument("--stride", type=int, default=10, help="keep every k-th contact tick")
    ap.add_argument("--min-record-frac", type=float, default=0.0,
                    help="skip an episode whose contact record is below this fraction of the "
                         "best episode's (0 = keep every episode)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from surfgym.goalfield import load_goal_field
    field = load_goal_field(a.field)
    z = np.load(a.states)
    eps = [z[k] for k in sorted(z.files)]
    d0 = a.d0 if a.d0 else float(field.sample(np.asarray(eps[0]["origin"][:1], np.float64))[0])
    roots, rows = pick(eps, field, d0, a.tick_ms, a.stride,
                       min_record_frac=a.min_record_frac)
    for k, n, best, tl, nc, ns, pmax, plo, phi in rows:
        print(f"ep{k}: {n} ticks ({n * a.tick_ms / 1000:.1f}s), contact record {100 * best:.2f}% "
              f"last improved at {tl * a.tick_ms / 1000:.1f}s; all-state best {100 * pmax:.2f}%; "
              f"{nc} contact ticks after it -> {ns} roots (progress {100 * plo:.1f}..{100 * phi:.1f}%)")
    np.save(a.out, roots)
    print(f"{len(roots)} roots -> {a.out}")


if __name__ == "__main__":
    main()
