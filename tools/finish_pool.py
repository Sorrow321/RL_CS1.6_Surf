#!/usr/bin/env python3
"""finish_pool.py - a --spawn-states pool that mixes a run's own frontier states with the last
seconds of the policy's OWN finishing search lines (zlr3b2, 2026-11-01).

A finishing line is a tools/beam_tas.py --save-frontier output (beam_frontier.npz with
finished = 1: the per-tick states of the search's finishing winner, replayed open-loop from the
true start through the policy's own greedy prefix; or any STATE_DTYPE .npy that ends at a
finish). Its states from --secs seconds before the finish up to --tail-skip seconds before it
are taken at --stride, replicated so that the finishing states make up --share of the output
pool, and appended to the base pool (a frontier_states.py file, or nothing). The trainer re-reads
--spawn-states when the file changes, so the output is written to a temporary file and renamed
over the target.

Nothing here names a place: the lines are the policy's own (its greedy prefix plus its own
proposals selected by a search), and the window is a time before the line's own finish.

    python tools/finish_pool.py --base runs/sp/base.npy --line runs/beam/X/beam_frontier.npz \\
        [--line ...] --secs 20 --share 0.5 --out runs/sp/pool.npy
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def line_states(path: str, tick_ms: float, secs: float, tail_skip: float, stride: int,
                need_finish: bool = True) -> np.ndarray:
    from surfgym.core import STATE_DTYPE
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        if need_finish and ("finished" not in z.files or int(z["finished"]) != 1):
            raise SystemExit(f"{path}: not a finishing line (finished != 1)")
        st = np.asarray(z["states"], STATE_DTYPE)
    else:
        st = np.asarray(np.load(path, allow_pickle=False), STATE_DTYPE)
    tps = 1000.0 / float(tick_ms)
    n = len(st)
    lo = max(0, n - int(round(secs * tps)))
    hi = max(lo + 1, n - int(round(tail_skip * tps)))
    out = st[lo:hi:max(1, int(stride))].copy()
    out["tick"] = 0
    out["stuck_ticks"] = 0
    return out


def main(argv=None):
    from surfgym.core import STATE_DTYPE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=None, help="the current own-state pool (.npy), kept whole")
    ap.add_argument("--line", action="append", required=True,
                    help="a finishing line (beam_frontier.npz finished=1, or a .npy)")
    ap.add_argument("--tick-ms", type=float, default=7.0)
    ap.add_argument("--secs", type=float, default=20.0,
                    help="take the line's states from this many seconds before its finish")
    ap.add_argument("--tail-skip", type=float, default=0.0,
                    help="... up to this many seconds before it")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--share", type=float, default=0.5,
                    help="the finishing states' share of the output pool (replicated to it)")
    ap.add_argument("--allow-unfinished", action="store_true",
                    help="accept a line whose npz does not say finished = 1 (analysis only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not 0.0 < a.share <= 1.0:
        raise SystemExit("--share must be in (0, 1]")
    fin = [line_states(p, a.tick_ms, a.secs, a.tail_skip, a.stride,
                       need_finish=not a.allow_unfinished) for p in a.line]
    fin = np.concatenate(fin) if fin else np.zeros(0, STATE_DTYPE)
    base = (np.asarray(np.load(a.base, allow_pickle=False), STATE_DTYPE)
            if a.base else np.zeros(0, STATE_DTYPE))
    if len(fin) == 0:
        raise SystemExit("no finishing states")
    if len(base) and a.share < 1.0:
        want = int(round(a.share * len(base) / (1.0 - a.share)))
        reps = max(1, int(np.ceil(want / len(fin))))
        fin_r = np.tile(fin, reps)[:max(want, len(fin))]
    else:
        fin_r = fin
    out = np.concatenate([base, fin_r]) if len(base) else fin_r
    out = np.ascontiguousarray(out, STATE_DTYPE)
    tmp = a.out + ".tmp.npy"
    np.save(tmp, out)
    os.replace(tmp, a.out)
    print(f"finish_pool: {len(base):,} base + {len(fin):,} finishing states x"
          f"{len(fin_r) / len(fin):.1f} = {len(fin_r):,} ({len(fin_r) / len(out):.0%}) "
          f"-> {a.out} ({len(out):,} states)")


if __name__ == "__main__":
    main()
