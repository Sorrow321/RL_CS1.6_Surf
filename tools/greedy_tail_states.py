#!/usr/bin/env python3
"""greedy_tail_states.py - spawn states from the policy's OWN greedy line, just before it fails.

From a `record_ckpt.py --dump-states` recording (greedy episodes from the true start), keep the
full core states of the last --secs seconds before each FAILED episode's end (every --every
ticks, skipping the last --skip seconds, which are already doomed), and write them as one
STATE_DTYPE .npy for `train_fast.py --spawn-states` (CLAUDE.md section 0: the policy's own
recording, so SELF_STATES=1 applies; no demo). Truncated episodes (the cap or a loiter) and
episodes longer than --max-secs are skipped: their tail is not where the line fails.

The file is written to a temp name and renamed, so a trainer that re-reads --spawn-states on
an mtime change never sees a half-written file.

    python tools/record_ckpt.py CKPT --episodes 9 --dump-states OUT/g.npz --out OUT/g.jsonl
    python tools/greedy_tail_states.py OUT/g.npz OUT/g.jsonl OUT/tail.npy --secs 3 --every 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def episode_ends(jsonl: Path) -> list[dict]:
    """The trailer of every episode in a record.py trajectory, in order, plus its tick_ms."""
    out, tick_ms = [], None
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if not line or line[0] != "{":
            continue
        d = json.loads(line)
        if "end" in d:
            d["tick_ms"] = tick_ms
            out.append(d)
        elif "tick_ms" in d:
            tick_ms = float(d["tick_ms"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="record_ckpt --dump-states .npz")
    ap.add_argument("jsonl", help="the .jsonl recorded with it (episode ends, tick_ms)")
    ap.add_argument("out", help=".npy of STATE_DTYPE rows")
    ap.add_argument("--secs", type=float, default=3.0, help="seconds before the end to keep from")
    ap.add_argument("--skip", type=float, default=0.25, help="seconds before the end to stop at")
    ap.add_argument("--every", type=int, default=5, help="keep every k-th tick")
    ap.add_argument("--max-secs", type=float, default=120.0,
                    help="skip episodes longer than this (a loiter's end is not a failure point)")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="drop states slower than this (u/s, 3-D |v|, the reservoir's "
                         "--respawn-min-speed rule); 0 = keep all")
    ap.add_argument("--min-rows", type=int, default=50,
                    help="refuse to write fewer rows (keeps the previous file in place)")
    a = ap.parse_args(argv)
    from surfgym.core import STATE_DTYPE
    from surfgym.greedy_tail import cut_tails      # the trainer's --greedy-tail-frac cut
    dump = dict(np.load(a.dump, allow_pickle=False))
    ends = episode_ends(Path(a.jsonl))
    tms = {e.get("tick_ms") for e in ends}
    if not ends or None in tms or len(tms) != 1:
        raise SystemExit("the trajectory headers carry no (single) tick_ms: refusing to assume 10 ms")
    eps = [(dump[f"ep_{i:04d}"], e.get("end")) for i, e in enumerate(ends) if f"ep_{i:04d}" in dump]
    rows, notes = cut_tails(eps, 1000.0 / tms.pop(), a.secs, a.skip, a.every, a.max_secs,
                            a.min_speed)
    for s in notes:
        print("  " + s)
    if len(rows) < a.min_rows:
        print(f"!! {len(rows)} rows < --min-rows {a.min_rows}: {a.out} left as it was")
        return 1
    if rows.dtype != STATE_DTYPE:
        raise SystemExit(f"dump rows are {rows.dtype}, not STATE_DTYPE")
    out = Path(a.out)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, rows)
    os.replace(tmp, out)
    print(f"wrote {len(rows):,} greedy-tail states from {sum(1 for s in notes if 'kept' in s)} "
          f"failed episodes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
