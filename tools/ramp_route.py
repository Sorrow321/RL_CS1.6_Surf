"""ramp_route.py - which SURFACES does a trajectory touch, in what order, at what speed? Maps a
recorded episode onto the surfaces tools/ramps.py (v2) extracted. MEASUREMENT: with a human record
it is a ruler, never training data (CLAUDE.md section 0).

    python tools/ramp_route.py runs/research/ramps2_<map>.npz <trajectory.jsonl> [--episode 0]

Contact: ramps.RampMap.contact - the origin within CONTACT_TOL of a standing-hull contact plane
(the extractor's samples ARE player origins at contact). Consecutive contact ticks on one surface
form a visit; the route is the visit sequence with entry / exit time and speed (walls and ceilings
are listed too - they end a flight in the ramp-command contract).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ramps import CATS, RampMap  # noqa: E402


def load_episode(path, ep):
    rows, k = [], 0
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("["):
            if k == ep:
                rows.append(json.loads(ln)[:7])
        elif ln.startswith('{"end'):
            if k == ep:
                break
            k += 1
    return np.asarray(rows, np.float64)


def visits_of(contact, min_ticks=2, merge_gap=5):
    out, t = [], 0
    while t < len(contact):
        if contact[t] < 0:
            t += 1
            continue
        r, t0 = contact[t], t
        while t < len(contact) and contact[t] == r:
            t += 1
        if t - t0 >= min_ticks:
            if out and out[-1][0] == r and t0 - out[-1][2] <= merge_gap:
                out[-1][2] = t
            else:
                out.append([int(r), t0, t])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ramps")
    ap.add_argument("traj")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--tick-ms", type=float, default=10.0)
    a = ap.parse_args(argv)
    rm = RampMap(a.ramps)
    traj = load_episode(a.traj, a.episode)
    c = rm.contact(traj[:, 1:4])
    sp = np.linalg.norm(traj[:, 4:7], axis=1)
    dt = a.tick_ms / 1000.0
    vs = visits_of(c)
    print(f"ramp_route: {len(traj)} ticks ({len(traj) * dt:.1f} s), {int((c >= 0).sum())} ticks in "
          f"contact ({100.0 * float((c >= 0).mean()):.0f}%); {len(vs)} surface visits:")
    print("   surface      potential p10   enter s  leave s   |v| in   |v| out")
    for r, t0, t1 in vs:
        print(f"   {CATS[rm.cat[r]][0].upper()}{r:<4d} {CATS[rm.cat[r]]:7s} {rm.d_p10[r]:9,.0f}   "
              f"{t0 * dt:7.2f}  {t1 * dt:7.2f}   {sp[t0]:6,.0f}   {sp[min(t1, len(sp) - 1)]:6,.0f}")
    print("   sequence: " + " -> ".join(f"{CATS[rm.cat[r]][0].upper()}{r}" for r, _a, _b in vs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
