#!/usr/bin/env python3
"""line_pool.py - a --spawn-states pool from the policy's OWN search lines (zlr3b4, 2026-11-01).

Takes the per-tick states of one or more tools/beam_tas.py --save-frontier lines
(beam_frontier.npz: the policy's own greedy prefix plus its own search suffix, replay-checked)
and writes a STATE_DTYPE .npy for train_fast --spawn-states: the last --secs seconds of each
line (where a finishing line's manoeuvre happens), every --stride-th tick, each line's rows
repeated --repeat times, optionally appended to an existing pool (--merge) so the spawn mix
keeps what it already trains on. The trainer re-reads a --spawn-states file when it changes, so
writing the result over the running pool (atomically: a .tmp then a rename) swaps it in place.
Nothing here reads a place on the map.

    python tools/line_pool.py runs/fs/s140_k0/beam_frontier.npz --secs 8 --stride 2 \
        --repeat 2 --merge runs/sp3/pool.npy --out runs/sp3/pool.npy --tick-ms 7
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main(argv=None):
    from surfgym.core import STATE_DTYPE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lines", nargs="+", help="beam_frontier.npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tick-ms", type=float, required=True)
    ap.add_argument("--secs", type=float, default=8.0, help="the last N seconds of each line")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--finished-only", action="store_true",
                    help="skip a line whose npz does not say finished = 1")
    ap.add_argument("--merge", default=None, help="an existing pool to append to")
    ap.add_argument("--share", type=float, default=None,
                    help="with --merge: repeat the new rows (instead of --repeat) so they are this "
                         "share of the merged pool, in (0, 1)")
    a = ap.parse_args(argv)
    n_t = int(round(a.secs * 1000.0 / a.tick_ms))
    parts = []
    for p in a.lines:
        z = np.load(p, allow_pickle=False)
        fin = int(z["finished"]) if "finished" in z.files else 0
        ok = int(z["replay_ok"]) if "replay_ok" in z.files else 0
        if a.finished_only and not fin:
            print(f"{p}: not a finishing line, skipped")
            continue
        st = np.asarray(z["states"], STATE_DTYPE)
        sel = st[max(0, len(st) - n_t)::max(1, a.stride)]
        print(f"{p}: {len(st)} states, finished {fin}, replay_ok {ok} -> {len(sel)} rows "
              f"x{a.repeat}")
        parts += [sel] * max(1, a.repeat)
    if not parts:
        raise SystemExit("no rows")
    new = np.concatenate(parts)
    if a.merge and Path(a.merge).exists():
        old = np.asarray(np.load(a.merge), STATE_DTYPE)
        print(f"merge: {len(old)} rows from {a.merge}")
        if a.share is not None:
            if not 0.0 < a.share < 1.0:
                raise SystemExit("--share must be in (0, 1)")
            uniq = np.concatenate([p for i, p in enumerate(parts)
                                   if i % max(1, a.repeat) == 0])
            want = int(round(a.share / (1.0 - a.share) * len(old)))
            reps = max(1, int(round(want / max(1, len(uniq)))))
            new = np.concatenate([uniq] * reps)
            print(f"--share {a.share:g}: the new rows x{reps} = {len(new)} "
                  f"({100.0 * len(new) / (len(new) + len(old)):.1f}% of the pool)")
        new = np.concatenate([old, new])
    out = Path(a.out)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, new)
    os.replace(tmp, out)
    print(f"-> {out}: {len(new)} rows")


if __name__ == "__main__":
    main()
