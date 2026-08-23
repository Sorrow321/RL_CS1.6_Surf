#!/usr/bin/env python3
"""gaze_stats.py - is the policy looking where it is going?

    python tools/gaze_stats.py runs/research/xPET/traj_*.jsonl

xPET (scratch, surf_petrus_lite, 64x32) flies BACKWARDS staring at the
floor: |yaw - heading| median 177.8 deg, heading inside the +/-60 deg FOV
on 0.0% of fast ticks, pitch median -69.2 against its own [-70, +30]
clamp. A second scratch run on the same map does not do this (1.8 deg,
100%, -13.1) and stalls at the same place, so the attractor and the wall
are separate defects. Any arm on this map has to say which of the two it
is looking at, which means these three numbers, computed the same way.

The definition, fixed so runs are comparable:
  * heading  = atan2(vy, vx) in degrees, over ticks whose HORIZONTAL
    speed exceeds --min-speed (default 200 u/s). Below that the velocity
    direction is noise and the angle is meaningless - walk speed is 250.
  * offset   = abs(wrap180(yaw - heading)); 0 = looking along the flight
    path, 180 = flying backwards.
  * in-FOV   = offset <= --fov/2 (default 120 deg wide, i.e. +/-60), the
    horizontal half of the depth render's field of view. This is the one
    that says whether the ramp being flown at is on screen at all.
  * pitch    = the raw pitch column, all ticks, no speed gate: where the
    camera points is a property of the policy, not of the flight.

Trajectory rows are the flat lists in traj_*.jsonl (the first line of
each file is the header object): yaw is field 7, pitch field 12,
velocity fields 4/5/6.
"""
import argparse
import glob
import json
import math
import statistics
import sys

YAW, PITCH = 7, 12
VX, VY = 4, 5


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def read_rows(path):
    """Yield the flat per-tick lists, skipping the header object."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("["):
                continue          # header object, or a blank tail line
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # a truncated last line while a run is live
            if isinstance(row, list) and len(row) > PITCH:
                yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-speed", type=float, default=200.0,
                    help="horizontal u/s below which heading is noise")
    ap.add_argument("--fov", type=float, default=120.0,
                    help="horizontal FOV in degrees (in-FOV is +/- fov/2)")
    ap.add_argument("--per-file", action="store_true")
    a = ap.parse_args()

    paths = []
    for pat in a.files:
        hit = sorted(glob.glob(pat))
        paths.extend(hit if hit else [pat])

    half = a.fov / 2.0
    tot_off, tot_pitch, tot_ticks, tot_fast = [], [], 0, 0
    for p in paths:
        offs, pitches, ticks, fast = [], [], 0, 0
        for row in read_rows(p):
            ticks += 1
            pitches.append(float(row[PITCH]))
            vx, vy = float(row[VX]), float(row[VY])
            if math.hypot(vx, vy) <= a.min_speed:
                continue
            fast += 1
            head = math.degrees(math.atan2(vy, vx))
            offs.append(abs(wrap180(float(row[YAW]) - head)))
        tot_off.extend(offs)
        tot_pitch.extend(pitches)
        tot_ticks += ticks
        tot_fast += fast
        if a.per_file and offs:
            print("%-52s |yaw-head| med %6.1f  inFOV %5.1f%%  pitch med %6.1f"
                  % (p.split("/")[-1], statistics.median(offs),
                     100.0 * sum(o <= half for o in offs) / len(offs),
                     statistics.median(pitches)))

    if not tot_off:
        print("no ticks above %.0f u/s horizontal in %d file(s) - the policy "
              "never moves, and the gaze question does not apply"
              % (a.min_speed, len(paths)))
        return 1

    med = statistics.median(tot_off)
    infov = 100.0 * sum(o <= half for o in tot_off) / len(tot_off)
    pmed = statistics.median(tot_pitch)
    print("files %d | ticks %d | fast ticks (>%.0f u/s horiz) %d (%.1f%%)"
          % (len(paths), tot_ticks, a.min_speed, tot_fast,
             100.0 * tot_fast / max(tot_ticks, 1)))
    print("|yaw - heading| median : %6.1f deg   (xPET 177.8 = backwards, "
          "xMM 1.8 = along the path)" % med)
    print("heading inside +/-%.0f    : %6.1f %%     (xPET 0.0, xMM 100.0)"
          % (half, infov))
    print("pitch median           : %6.1f deg   (xPET -69.2 at its -70 "
          "clamp, xMM -13.1)" % pmed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
