"""Energy height at first arrival per progress level: greedy evals against a spine (or any line).

The energy playbook's first step (autoresearch/NOTES.md, pbr5b2 / pbr6b3): an energy gate shows as a
line that arrives where it dies with too little h = z + |v|^2 / (2 g), and the place it lost it is
the first progress level where the greedy's h falls below the reference line's. For each level L (%
of the start's d on the goal field) this prints the reference's row / h / horizontal speed at its
first arrival and, per eval file, how many of its first --episodes episodes got there and the range
of their h and horizontal speed.

    python tools/energy_profile.py --field maps/surf_princessburglar_b26.goalc_32.npz \\
        --levels 58,59,60,61,62,62.5 --ref autoresearch/spines/pbr6b3_b2k01.npy runs/<run>/traj_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

G = 800.0


def energy(pos, vel) -> np.ndarray:
    pos = np.asarray(pos, np.float64).reshape(-1, 3)
    vel = np.asarray(vel, np.float64).reshape(-1, 3)
    return pos[:, 2] + (vel * vel).sum(axis=1) / (2.0 * G)


def first_at(pct, L: float):
    """index of the first row at or past level L, or None."""
    k = np.flatnonzero(np.asarray(pct) >= float(L))
    return int(k[0]) if len(k) else None


def episodes(path, n: int = 9):
    """[(pos (k, 3), vel (k, 3))] of a record.py trajectory's first n episodes (header index < n)."""
    out, hdr, cur = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            d = json.loads(line)
            if "end" in d and hdr is not None:
                if not isinstance(hdr.get("episode"), int) or n <= 0 or hdr["episode"] < n:
                    r = np.asarray(cur, np.float64)
                    if len(r):
                        out.append((r[:, 1:4], r[:, 4:7]))
                hdr, cur = None, []
            else:
                hdr, cur = d, []
            continue
        cur.append(json.loads(line))
    return out


def profile(field, d0: float, pos, vel):
    """-> (progress %, h, horizontal speed) per row."""
    d = np.asarray(field.sample(np.asarray(pos, np.float64)), np.float64)
    d = np.where(d < field.reach_max - 0.5 * field.cell, d, np.inf)
    vel = np.asarray(vel, np.float64)
    return 100.0 * (d0 - d) / d0, energy(pos, vel), np.hypot(vel[:, 0], vel[:, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("evals", nargs="+", help="greedy eval trajectories (traj_<step>.jsonl)")
    ap.add_argument("--field", required=True, help="the goal field the progress is read on")
    ap.add_argument("--levels", required=True, help="comma list of progress levels, %%")
    ap.add_argument("--ref", default=None, help="a STATE_DTYPE .npy (a spine) to compare with")
    ap.add_argument("--d0", type=float, default=None,
                    help="the start's d (default: the first eval episode's first row)")
    ap.add_argument("--episodes", type=int, default=9, help="first N episodes (0 = all)")
    a = ap.parse_args(argv)
    from auto_spine import load_field
    field = load_field(a.field)
    levels = [float(x) for x in a.levels.split(",") if x.strip()]
    E = [episodes(p, a.episodes) for p in a.evals]
    d0 = a.d0
    if d0 is None:
        first = next((e[0][0][:1] for e in E if e), None)
        if first is None:
            ap.error("no episode in the evals and no --d0")
        d0 = float(field.sample(np.asarray(first, np.float64))[0])
    ref = None
    if a.ref:
        s = np.load(a.ref)
        ref = profile(field, d0, s["origin"], s["velocity"])
        print(f"reference {a.ref}: {len(s)} rows, max {np.nanmax(ref[0][np.isfinite(ref[0])]):.2f}%")
    P = [[profile(field, d0, pos, vel) for pos, vel in e] for e in E]
    print("level  " + ("ref(row h hs)       " if ref is not None else "")
          + " | ".join(Path(p).name for p in a.evals))
    for L in levels:
        line = f"{L:5.1f}  "
        if ref is not None:
            i = first_at(ref[0], L)
            line += (f"{i:5d} {ref[1][i]:5.0f} {ref[2][i]:4.0f}       " if i is not None
                     else "    -                ")
        cells = []
        for eps in P:
            got = [(pr[1][i], pr[2][i]) for pr in eps for i in [first_at(pr[0], L)] if i is not None]
            if got:
                h = [g[0] for g in got]
                v = [g[1] for g in got]
                cells.append(f"{len(got)}/{len(eps)} h {min(h):4.0f}-{max(h):4.0f} "
                             f"hs {min(v):3.0f}-{max(v):3.0f}")
            else:
                cells.append(f"0/{len(eps)}")
        print(line + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
