"""pool_add_lines.py - a --spawn-states pool = a base pool of the policy's own states + search lines
from those states, the lines replicated to a target share (zlr3b1, 2026-11-01).

The trainer draws --spawn-states rows uniformly, so a handful of short search lines (a few hundred
states each, e.g. tools/finish_mpc.py's line_r<row>.npy from the policy's own frontier states) would
be ~1% of a 50k-state frontier pool. --share sets the fraction of the OUTPUT pool that is line
states (each line weighted equally, whatever its length). Nothing here reads a place on the map.

    python tools/pool_add_lines.py --base autoresearch/spines/X_frontier.npy \
        --lines runs/fmpc/m1/line_r*.npy --share 0.3 --out autoresearch/spines/Y_pool.npy
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def build(base, lines, share):
    """-> the pool: base + every line repeated so that lines are `share` of the result, each line
    getting the same number of rows (its states tiled)."""
    if not lines:
        return base.copy()
    if not 0.0 < share < 1.0:
        raise ValueError("share must be in (0, 1)")
    n_line_rows = int(round(len(base) * share / (1.0 - share)))
    per = max(1, n_line_rows // len(lines))
    parts = [base]
    for ln in lines:
        parts.append(np.resize(ln, per))      # tiles the line's states to `per` rows
    out = np.concatenate(parts)
    out["tick"] = 0
    out["stuck_ticks"] = 0
    return out


def main(argv=None):
    from surfgym.core import STATE_DTYPE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--lines", nargs="+", required=True, help="line .npy files (globs ok)")
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = np.load(a.base)
    files = sorted({f for g in a.lines for f in glob.glob(g)})
    lines = []
    for f in files:
        ln = np.load(f)
        if ln.dtype != STATE_DTYPE or ln.ndim != 1 or len(ln) < 1:
            raise SystemExit(f"{f}: not a 1-D STATE_DTYPE array")
        lines.append(ln)
    if base.dtype != STATE_DTYPE:
        raise SystemExit(f"{a.base}: not a STATE_DTYPE array")
    pool = build(base, lines, a.share)
    tmp = Path(a.out).with_suffix(".tmp.npy")
    np.save(tmp, pool)
    tmp.replace(a.out)                 # atomic: the trainer re-reads the file when it changes
    print(f"{len(base)} base + {len(lines)} lines ({sum(len(x) for x in lines)} states) -> "
          f"{len(pool)} rows, {100.0 * (len(pool) - len(base)) / len(pool):.1f}% line states "
          f"-> {a.out}")
    for f, ln in zip(files, lines):
        print(f"  {f}: {len(ln)} states")


if __name__ == "__main__":
    main()
