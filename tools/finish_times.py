#!/usr/bin/env python3
"""finish_times.py - pooled finish TIMES over recorded greedy episodes.

CLAUDE.md rule 3: a run that finishes the map is judged on wall clock. Every
arm since xARC has measured it the same way, and this is that measurement
made reusable:

    time = (first tick inside the finish box, +64 u pad) - (first recorded
    tick), in seconds at 100 ticks/s.

That is NOT the episode's length: an episode can keep recording after it
crosses. It is also not the trainer's own ``fin`` field, which differs by
0.02-0.06 s because it counts from the spawn tick of the env rather than the
first recorded one.

    python tools/finish_times.py runs/research/xNOSHP/traj_*.jsonl

Prints one row per file (n finishers, best, mean) and then the pooled mean,
median and sd over every finisher, which is what distinguishes a real gain
from variance.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from surfgym.tick import episode_seconds  # noqa: E402  (the recording's tick)


def episodes_from_traj(path, with_headers=False):
    """Split a record_rollout .jsonl into per-episode arrays.

    Same contract as surfgym.route.episodes_from_traj, with one difference
    that matters when an arm is harvested while a recording is still being
    written: a TRUNCATED final row is skipped instead of raising. xLAT3's
    last file ends mid-line and that alone made the whole eval unscoreable.
    ``with_headers=True`` also returns each episode's header dict (None
    where a recorder wrote none): the recording's time base (tick_ms).
    """
    eps, cur, prev_tick = [], [], None
    hdrs, cur_hdr = [], None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line[0] == "{":                      # header or footer
                if cur:
                    eps.append(np.asarray(cur, np.float64))
                    hdrs.append(cur_hdr)
                    cur = []
                prev_tick = None
                try:
                    d = json.loads(line)
                except ValueError:
                    d = None
                cur_hdr = (d if isinstance(d, dict) and "end" not in d
                           else None)
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue                            # truncated tail row
            if not isinstance(row, list) or len(row) < 8:
                continue
            tick = row[0]
            if prev_tick is not None and tick <= prev_tick:
                if cur:
                    eps.append(np.asarray(cur, np.float64))
                    hdrs.append(cur_hdr)
                    cur = []
                    cur_hdr = None
            prev_tick = tick
            cur.append(row[:8])
    if cur:
        eps.append(np.asarray(cur, np.float64))
        hdrs.append(cur_hdr)
    if with_headers:
        return eps, hdrs
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", nargs="+")
    ap.add_argument("--map", default="surf_src_cannonball")
    ap.add_argument("--pad", type=float, default=64.0)
    a = ap.parse_args()

    end = json.loads((ROOT / "maps" / f"{a.map}.zones.json").read_text(
        encoding="utf-8"))["end"]
    lo, hi = np.asarray(end["mins"]), np.asarray(end["maxs"])

    files = []
    for pat in a.traj:
        files.extend(sorted(glob.glob(pat)) or
                     ([pat] if Path(pat).exists() else []))
    if not files:
        raise SystemExit("no trajectory files matched")

    pooled = []
    for f in files:
        eps, hdrs = episodes_from_traj(f, with_headers=True)
        times = []
        for i, (ep, hdr) in enumerate(zip(eps, hdrs)):
            xyz = ep[:, 1:4].astype(np.float32)
            inside = np.all((xyz >= lo - a.pad) & (xyz <= hi + a.pad), axis=1)
            if inside.any():
                # seconds at the recording's OWN tick (header tick_ms /
                # pattern); a header-less episode refuses, never 10 ms
                times.append(episode_seconds(hdr, int(np.argmax(inside)),
                                             f"{Path(f).name} ep{i}"))
        pooled.extend(times)
        if times:
            print("%-28s fin %d/%-2d  best %6.2fs  mean %6.2fs  %s"
                  % (Path(f).name, len(times), len(eps), min(times),
                     float(np.mean(times)),
                     " ".join("%.2f" % t for t in sorted(times))))
        else:
            print("%-28s fin 0/%-2d" % (Path(f).name, len(eps)))

    if pooled:
        p = np.asarray(pooled)
        print("\npooled finishers n=%d  min %.2fs  mean %.2fs  median %.2fs  "
              "max %.2fs  sd %.2f"
              % (len(p), p.min(), p.mean(), float(np.median(p)), p.max(),
                 float(p.std(ddof=1)) if len(p) > 1 else 0.0))
    else:
        print("\nno finishers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
