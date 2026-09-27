"""uf2_traj_probe.py - per episode of recorded unitfarmer2 trajectories: did it enter the pit, how
fast did it go in there, did it get out of the start shaft?

    python tools/uf2_traj_probe.py runs/<run>/traj_<step>.jsonl [...]

MEASUREMENT only (CLAUDE.md 0b), the same rungs as tools/uf2_archive_probe.py: pit contact (the
docs/gate_boxes.json pit box) -> in-pit horizontal speed >= 1,400 u/s -> out of the shaft (y > -900,
or x < -3,000 / x > -1,400) and alive 3 s later. Rows are record_ckpt's [tick, x, y, z, vx, vy, vz,
...]; an episode ends at its {"end": ...} line.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# the geodesic goal field, as a MEASUREMENT (the uf2 review's PASS: progress d0 - d >= 5,600 u,
# alive 3 s later). "Out of the shaft" alone counts the lower level's north end, which drops
# into a region ~3,000 u FARTHER from the goal than the start (2026-09-27 loop round 1)
GOAL_FIELD = ROOT / "maps_pool" / "surf_unitfarmer2.goal_32.npz"
START = (-2176.0, -2272.0, 1192.0)


def depth_of(o):
    """Geodesic progress d0 - d of positions (N, 3) on unitfarmer2 (NaN where unreachable)."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "python"))
    from surfgym.goalfield import load_goal_field
    gf = load_goal_field(str(GOAL_FIELD))
    d0 = float(gf.sample(np.asarray([START]))[0])
    d = gf.sample(np.asarray(o, np.float64)).astype(np.float64)
    return d0 - d


def episodes(path):
    ep, head = [], None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("["):
                ep.append(json.loads(ln)[:7])
            elif ln.startswith("{"):
                h = json.loads(ln)
                if "end" in h:
                    yield head, h, np.asarray(ep, np.float64)
                    ep = []
                else:
                    head = h


def main(argv=None) -> int:
    g = json.loads((ROOT / "docs" / "gate_boxes.json").read_text(encoding="utf-8"))
    box = g["surf_unitfarmer2"]["boxes"][0]
    lo, hi = np.asarray(box["mins"], np.float64), np.asarray(box["maxs"], np.float64)
    for p in (argv if argv is not None else sys.argv[1:]):
        rows = []
        for head, end, e in episodes(p):
            if not len(e):
                continue
            tick_s = float((head or {}).get("tick_ms", 10.0)) / 1000.0
            o, v = e[:, 1:4], e[:, 4:7]
            vh = np.hypot(v[:, 0], v[:, 1])
            inp = np.all((o >= lo) & (o <= hi), axis=1)
            out = (o[:, 1] > -900.0) | (o[:, 0] < -3000.0) | (o[:, 0] > -1400.0)
            t_out = int(np.argmax(out)) if out.any() else -1
            alive3 = t_out >= 0 and (len(e) - 1 - t_out) * tick_s >= 3.0
            dep = depth_of(o)
            dep = np.where(np.isfinite(dep), dep, -1e9)
            k5 = np.flatnonzero(dep >= 5600.0)
            pass_ = bool(len(k5)) and (len(e) - 1 - int(k5[0])) * tick_s >= 3.0
            rows.append((bool(inp.any()), float(vh[inp].max()) if inp.any() else 0.0,
                         bool(out.any()), bool(alive3), float(o[:, 1].max()), len(e) * tick_s,
                         str(end.get("end")), float(dep.max()), pass_))
        n = len(rows)
        print(f"{Path(p).name}: {n} episodes | entered the pit {sum(r[0] for r in rows)} | "
              f"in-pit vh >= 1,400: {sum(r[1] >= 1400 for r in rows)} (max "
              f"{max((r[1] for r in rows), default=0):,.0f}) | out of the shaft "
              f"{sum(r[2] for r in rows)} (alive 3 s after: {sum(r[3] for r in rows)}) | "
              f"max depth {max((r[7] for r in rows), default=0):,.0f} u, PASS (>= 5,600 u, "
              f"alive 3 s) {sum(r[8] for r in rows)} | "
              f"ends: " + ", ".join(f"{r[6]} {r[5]:.1f}s/{r[7]:,.0f}u" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
