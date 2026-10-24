#!/usr/bin/env python3
"""energy_budget.py - where a trajectory loses its energy (pbr2b1, 2026-10-24).

Mechanical energy per unit mass, as a height: h = z + |v|^2 / (2 g) (g = sv_gravity, 800 by
default). Free fall and frictionless surfing keep h; air strafing raises it; a contact's lost
normal component, ground friction, and the speed clamp lower it. An energy GATE (a launch that
needs more height than the line carries) is diagnosed by where h is lost, not by where the line
dies. Reads the repo's per-tick trajectory JSONL (record_ckpt / trainer evals: rows
[tick, x, y, z, vx, vy, vz, ...]) and prints, per episode, h at even intervals plus every drop
of more than --loss u within --window ticks, with where it happened. A measurement instrument:
nothing here reads the map, and nothing feeds a reward or a spawn.

    python tools/energy_budget.py runs/x/traj_000123.jsonl --episodes 0 1 --every 50 --loss 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episodes(path):
    """[(header, rows (n, 7) float64)] from a trajectory JSONL."""
    out, cur, hdr = [], None, None
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        if isinstance(o, dict):
            if "episode" in o:
                hdr, cur = o, []
                out.append((hdr, cur))
        elif cur is not None:
            cur.append(o[:7])
    return [(h, np.asarray(r, np.float64).reshape(-1, 7)) for h, r in out if r]


def energy_height(rows: np.ndarray, gravity: float = 800.0) -> np.ndarray:
    v = rows[:, 4:7]
    return rows[:, 3] + (v * v).sum(1) / (2.0 * gravity)


def losses(h: np.ndarray, window: int = 20, loss: float = 60.0):
    """[(t0, t1, dh)] - non-overlapping stretches of <= window ticks over which h fell by more
    than ``loss`` (greedy, earliest first)."""
    out, t = [], 0
    n = len(h)
    while t < n - 1:
        hi = min(n - 1, t + window)
        j = t + int(np.argmin(h[t:hi + 1]))
        if h[t] - h[j] > loss:
            out.append((t, j, float(h[j] - h[t])))
            t = j
        else:
            t += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traj")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--every", type=int, default=50, help="print h every N ticks")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--loss", type=float, default=60.0)
    ap.add_argument("--gravity", type=float, default=None,
                    help="default: the header's phys.sv_gravity, else 800")
    a = ap.parse_args(argv)
    for k, (hdr, rows) in enumerate(episodes(a.traj)):
        if a.episodes is not None and k not in a.episodes:
            continue
        g = a.gravity or float((hdr.get("phys") or {}).get("sv_gravity", 800.0))
        h = energy_height(rows, g)
        print(f"== episode {hdr.get('episode', k)}: {len(rows)} ticks, h {h[0]:.0f} -> {h[-1]:.0f} "
              f"(max {h.max():.0f})")
        for t in range(0, len(rows), max(1, a.every)):
            p, v = rows[t, 1:4], rows[t, 4:7]
            print(f"   t {t:6d} pos ({p[0]:7.0f},{p[1]:7.0f},{p[2]:6.0f}) |v| {np.linalg.norm(v):6.0f} "
                  f"h {h[t]:6.0f}")
        for t0, t1, dh in losses(h, a.window, a.loss):
            p = rows[t1, 1:4]
            print(f"   LOSS {dh:7.0f} u over ticks {t0}-{t1} at ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
