#!/usr/bin/env python3
"""How much does the policy CHATTER on the movement keys?

The behavioural measurement the ``--keys-hold`` arm exists to move.  The
defect, from the macro-hold work and ``docs/research-results.md``: A/D flips
per second are **0.42** on the human world record, **7.01** on the exitABS r9
policy and **18.91** at ``--act-every 1``.  A key that has to be re-decided
from scratch every decision is a key that gets dropped by accident, and the
frontier metrics cannot see it at all.

This reads the ENGINE actions a recording already carries - ``traj_*.jsonl``
column 13 (fwd: 0 = S, 1 = none, 2 = W), column 14 (side: 0 = A, 1 = none,
2 = D) and the duck bit of the button mask in column 8 - so it measures the
same thing for both arms with no trainer change, and it works on every
recording ever made.

Reported per file, over its greedy episodes:

  * ``side chg/s``   any change of the side key, per second of flight
  * ``A<->D/s``      direction reversals only (A to D or D to A, however
                     many neutral ticks are in between) - the number the
                     0.42 / 7.01 / 18.91 series is in
  * ``fwd chg/s``    any change of the fwd key
  * ``duck chg/s``   any change of the duck button
  * ``hold``         mean / median seconds a non-neutral key stays held,
                     per key, over every run of that key in the file

Usage:

    python tools/key_chatter.py runs/cyKEYH/traj_*.jsonl
    python tools/key_chatter.py --per-episode runs/cyKEYC/traj_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

SURF_IN_DUCK = 4          # surfgym/core.py, HLSDK convention
I_BUTTONS, I_FWD, I_SIDE = 8, 13, 14
NEUTRAL_FWD = NEUTRAL_SIDE = 1


def episodes(path):
    """Yield ``(header, rows)`` per episode of a traj file.

    A file is header dict / tick rows / trailer dict, repeated once per
    episode.  Rows shorter than 15 columns predate the fwd/side append and
    are skipped with a note rather than silently scored as neutral.
    """
    hdr = None
    rows = []
    short = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line)
            if isinstance(v, dict):
                if "map" in v or "tick_ms" in v:
                    hdr = v
                    rows = []
                else:                       # trailer: the episode is closed
                    if rows:
                        yield hdr or {}, np.asarray(rows, np.int32), v
                    rows = []
                continue
            if len(v) < 15:
                short += 1
                continue
            rows.append((int(v[I_BUTTONS]), int(v[I_FWD]), int(v[I_SIDE])))
    if short:
        print(f"  ! {short} rows in {Path(path).name} predate the fwd/side "
              f"columns and were skipped", file=sys.stderr)


def runs_of(a, live):
    """Lengths (in ticks) of every maximal run where ``live`` holds."""
    m = np.asarray(live, bool)
    if not m.any():
        return np.zeros(0, np.int64)
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def score_file(path, per_episode=False):
    tot = dict(ticks=0, side_chg=0, ad=0, fwd_chg=0, duck_chg=0, eps=0,
               finishes=0)
    holds = {"A": [], "D": [], "W/S": [], "duck": []}
    tick_s = 0.010
    for hdr, arr, tr in episodes(path):
        tick_s = float(hdr.get("tick_ms", 10)) / 1000.0
        btn, fwd, side = arr[:, 0], arr[:, 1], arr[:, 2]
        duck = (btn & SURF_IN_DUCK) > 0
        n = len(arr)
        side_chg = int((side[1:] != side[:-1]).sum())
        fwd_chg = int((fwd[1:] != fwd[:-1]).sum())
        duck_chg = int((duck[1:] != duck[:-1]).sum())
        # A<->D reversals: collapse the neutral ticks, then count sign
        # changes in what is left. Tapping A, releasing, tapping A again is
        # chatter on `side chg` but is NOT a reversal.
        nz = side[side != NEUTRAL_SIDE]
        ad = int((nz[1:] != nz[:-1]).sum()) if len(nz) > 1 else 0
        tot["ticks"] += n
        tot["side_chg"] += side_chg
        tot["ad"] += ad
        tot["fwd_chg"] += fwd_chg
        tot["duck_chg"] += duck_chg
        tot["eps"] += 1
        tot["finishes"] += 1 if tr.get("end") == "done" else 0
        holds["A"].append(runs_of(side, side == 0))
        holds["D"].append(runs_of(side, side == 2))
        holds["W/S"].append(runs_of(fwd, fwd != NEUTRAL_FWD))
        holds["duck"].append(runs_of(duck, duck))
        if per_episode:
            secs = max(n * tick_s, 1e-9)
            print(f"    ep {tot['eps'] - 1:2d}  {secs:6.2f}s  "
                  f"side {side_chg / secs:6.2f}/s  A<->D {ad / secs:6.2f}/s  "
                  f"fwd {fwd_chg / secs:5.2f}/s  duck {duck_chg / secs:5.2f}/s"
                  f"  ({tr.get('end')})")
    secs = max(tot["ticks"] * tick_s, 1e-9)
    out = {
        "file": Path(path).name,
        "eps": tot["eps"],
        "finishes": tot["finishes"],
        "secs": secs,
        "side_per_s": tot["side_chg"] / secs,
        "ad_per_s": tot["ad"] / secs,
        "fwd_per_s": tot["fwd_chg"] / secs,
        "duck_per_s": tot["duck_chg"] / secs,
    }
    for k, v in holds.items():
        r = np.concatenate(v) if v else np.zeros(0, np.int64)
        out[f"hold_{k}_mean"] = float(r.mean() * tick_s) if len(r) else 0.0
        out[f"hold_{k}_med"] = float(np.median(r) * tick_s) if len(r) else 0.0
        out[f"hold_{k}_n"] = int(len(r))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--per-episode", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object per file, for the ledger")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        g = sorted(glob.glob(f))
        paths.extend(g if g else [f])

    rows = []
    for p in paths:
        if args.per_episode:
            print(p)
        try:
            rows.append(score_file(p, args.per_episode))
        except Exception as exc:                       # pragma: no cover
            print(f"  ! {p}: {exc!r}", file=sys.stderr)
    if args.json:
        for r in rows:
            print(json.dumps(r))
        return
    print(f"{'file':<34} {'eps':>4} {'fin':>4} {'secs':>7} "
          f"{'side/s':>7} {'A<->D/s':>8} {'fwd/s':>7} {'duck/s':>7} "
          f"{'holdA':>7} {'holdD':>7} {'holdW/S':>8} {'holdDuck':>9}")
    for r in rows:
        print(f"{r['file']:<34} {r['eps']:>4} {r['finishes']:>4} "
              f"{r['secs']:>7.1f} {r['side_per_s']:>7.2f} "
              f"{r['ad_per_s']:>8.2f} {r['fwd_per_s']:>7.2f} "
              f"{r['duck_per_s']:>7.2f} {r['hold_A_mean']:>7.3f} "
              f"{r['hold_D_mean']:>7.3f} {r['hold_W/S_mean']:>8.3f} "
              f"{r['hold_duck_mean']:>9.3f}")
    if len(rows) > 1:
        w = np.asarray([r["secs"] for r in rows], np.float64)
        def wm(k):
            return float(np.sum([r[k] * r["secs"] for r in rows]) / w.sum())
        print(f"{'ALL (time-weighted)':<34} "
              f"{sum(r['eps'] for r in rows):>4} "
              f"{sum(r['finishes'] for r in rows):>4} {w.sum():>7.1f} "
              f"{wm('side_per_s'):>7.2f} {wm('ad_per_s'):>8.2f} "
              f"{wm('fwd_per_s'):>7.2f} {wm('duck_per_s'):>7.2f} "
              f"{wm('hold_A_mean'):>7.3f} {wm('hold_D_mean'):>7.3f} "
              f"{wm('hold_W/S_mean'):>8.3f} {wm('hold_duck_mean'):>9.3f}")


if __name__ == "__main__":
    main()
