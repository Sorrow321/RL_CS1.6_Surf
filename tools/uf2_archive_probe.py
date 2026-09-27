"""uf2_archive_probe.py - did an edge archive surf unitfarmer2's pit, and did it get out?

    python tools/uf2_archive_probe.py runs/research/archive_<name>/nodes.npz [...]

A MEASUREMENT of a finished search (edge_archive.py --dump-nodes), map-specific on purpose and
never an input to anything (CLAUDE.md 0b: gate boxes are bench metrics). The pit box comes from
docs/gate_boxes.json; the rungs are the ones the uf2 review set (docs/uf2-exploration-review.md):
pit contact -> in-pit horizontal speed >= 1,400 u/s (the first ramp surfed; the human record
reaches ~1,765) -> out of the start shaft. "Out" = any node north of the shaft's north wall
(y > -900: the record crosses it at z ~500 around t 10 s) or west / east of the shaft's walls
(x < -3,000 or x > -1,400).
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


def main(argv=None) -> int:
    paths = (argv if argv is not None else sys.argv[1:])
    g = json.loads((ROOT / "docs" / "gate_boxes.json").read_text(encoding="utf-8"))
    box = g["surf_unitfarmer2"]["boxes"][0]
    lo, hi = np.asarray(box["mins"], np.float64), np.asarray(box["maxs"], np.float64)
    for p in paths:
        z = np.load(p)
        o = z["origin"].astype(np.float64)
        v = z["velocity"].astype(np.float64)
        vh = np.hypot(v[:, 0], v[:, 1])
        inpit = np.all((o >= lo) & (o <= hi), axis=1)
        out_n = o[:, 1] > -900.0
        out_w = o[:, 0] < -3000.0
        out_e = o[:, 0] > -1400.0
        out = out_n | out_w | out_e
        vp = vh[inpit]
        print(f"{Path(p).parent.name}: {len(o):,} nodes | in the pit box {int(inpit.sum()):,} "
              f"| in-pit horizontal speed max {vp.max() if len(vp) else 0:,.0f} u/s, "
              f"p99 {np.percentile(vp, 99) if len(vp) else 0:,.0f}, "
              f">= 1,400: {int((vp >= 1400).sum())} | out of the shaft: {int(out.sum())} "
              f"(north {int(out_n.sum())}, west {int(out_w.sum())}, east {int(out_e.sum())}) "
              f"| max y {o[:, 1].max():,.0f}, z range {o[:, 2].min():,.0f}..{o[:, 2].max():,.0f}, "
              f"max |v| {np.linalg.norm(v, axis=1).max():,.0f}")
        dep = depth_of(o)
        dep = np.where(np.isfinite(dep), dep, -1e9)
        jd = int(np.argmax(dep))
        print(f"   geodesic progress (d0 - d): max {dep[jd]:,.0f} u at "
              f"{np.round(o[jd], 0).astype(int).tolist()} ({np.hypot(v[jd, 0], v[jd, 1]):,.0f} u/s); "
              f"nodes >= 2,500: {int((dep >= 2500).sum())}, >= 3,600 (the record at 12 s): "
              f"{int((dep >= 3600).sum())}, >= 5,600 (PASS depth): {int((dep >= 5600).sum())}")
        if out.any():
            i = np.flatnonzero(out)
            j = i[np.argmax(o[i, 1])]
            print(f"   the northernmost escaped node: {np.round(o[j], 0).tolist()} at "
                  f"{np.hypot(v[j, 0], v[j, 1]):,.0f} u/s, depth {int(z['depth'][j])}, "
                  f"{int(z['t'][j]) / 100.0:.1f} s from the root")
            if "all_parent" in z.files:
                # its chain back to the root, one point per move (analysis only)
                ao, ap = z["all_origin"], z["all_parent"]
                nid, pts = int(z["node_id"][j]), []
                while nid >= 0:
                    pts.append(np.round(ao[nid], 0).astype(int).tolist())
                    nid = int(ap[nid])
                print("   its chain, one point per 2 s move: " + " -> ".join(
                    f"({a},{b},{c})" for a, b, c in pts[::-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
