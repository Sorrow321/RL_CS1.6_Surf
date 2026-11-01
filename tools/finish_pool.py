#!/usr/bin/env python3
"""finish_pool.py - a --spawn-states pool from the policy's own finishing search lines.

Inputs: a base pool (tools/frontier_states.py: the policy's own greedy states at its frontier)
and the lines of searches that ENTERED the goal (tools/finish_beams.py ->
<dir>/r*/beam_frontier.npz with finished == 1 and replay_ok == 1: the policy's own proposals
from its own states, replayed bit-exactly). From each line it keeps the states of the last
--secs seconds before the goal entry (the approach the policy has never trained on), repeats
them so the finishing approaches make up --frac of the written pool, and writes the result
atomically (write + rename), so a trainer running with --spawn-states on that path re-reads it
in place.

Generic: the rule is "the last S seconds of a line that finished"; nothing reads the map.

    python tools/finish_pool.py --base POOL.npy --lines 'runs/fb/a/r*/beam_frontier.npz' \
        --secs 8 --frac 0.4 --out runs/sp/pool.npy
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def finishing_tails(paths, secs: float, tick_s: float):
    """[(path, tail states)] for every finished, replay-verified line; the tail is its last
    ``secs`` seconds of states (all of it when shorter)."""
    out = []
    k = max(1, int(round(secs / tick_s)))
    for p in paths:
        z = np.load(p, allow_pickle=False)
        if int(z["finished"]) != 1 or int(z["replay_ok"]) != 1:
            continue
        s = np.asarray(z["states"])
        if len(s) == 0:
            continue
        out.append((p, s[-k:].copy()))
    return out


def mix(base: np.ndarray, tails, frac: float, seed: int = 0) -> np.ndarray:
    """base + the tails repeated so they are ~``frac`` of the result (each tail repeated about
    equally); frac 0 or no tails returns base unchanged."""
    if not tails or frac <= 0.0:
        return base.copy()
    if not 0.0 < frac < 1.0:
        raise SystemExit("--frac must be in (0, 1)")
    t = np.concatenate([s for _, s in tails])
    want = int(round(frac * len(base) / (1.0 - frac)))
    reps = max(1, int(round(want / len(t))))
    rows = np.concatenate([base] + [t] * reps)
    rows = rows.copy()
    rows["tick"] = 0
    rows["stuck_ticks"] = 0
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--lines", action="append", required=True,
                    help="glob(s) of beam_frontier.npz files (repeatable)")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--tick-ms", type=float, default=7.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from surfgym.core import STATE_DTYPE
    base = np.load(a.base, allow_pickle=False)
    if base.dtype != STATE_DTYPE:
        raise SystemExit(f"--base {a.base}: not a STATE_DTYPE array ({base.dtype})")
    paths = sorted({p for g in a.lines for p in glob.glob(g)})
    tails = finishing_tails(paths, a.secs, a.tick_ms / 1000.0)
    for p, s in tails:
        o = s["origin"]
        print(f"line {p}: {len(s)} tail states, ({o[0][0]:.0f},{o[0][1]:.0f},{o[0][2]:.0f}) -> "
              f"({o[-1][0]:.0f},{o[-1][1]:.0f},{o[-1][2]:.0f})")
    rows = mix(base, tails, a.frac)
    if rows.dtype != STATE_DTYPE:
        raise SystemExit("internal: dtype changed")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, rows)
    os.replace(tmp, out)
    n_t = len(rows) - len(base)
    print(f"finish_pool: {len(paths)} line file(s), {len(tails)} finished; wrote {len(rows):,} "
          f"states ({len(base):,} base + {n_t:,} finishing-approach rows = "
          f"{n_t / max(1, len(rows)):.1%}) -> {out}")


if __name__ == "__main__":
    main()
