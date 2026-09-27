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
            rows.append((bool(inp.any()), float(vh[inp].max()) if inp.any() else 0.0,
                         bool(out.any()), bool(alive3), float(o[:, 1].max()), len(e) * tick_s,
                         str(end.get("end"))))
        n = len(rows)
        print(f"{Path(p).name}: {n} episodes | entered the pit {sum(r[0] for r in rows)} | "
              f"in-pit vh >= 1,400: {sum(r[1] >= 1400 for r in rows)} (max "
              f"{max((r[1] for r in rows), default=0):,.0f}) | out of the shaft "
              f"{sum(r[2] for r in rows)} (alive 3 s after: {sum(r[3] for r in rows)}) | "
              f"ends: " + ", ".join(f"{r[6]} {r[5]:.1f}s" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
