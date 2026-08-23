"""traj_ends.py - where a policy's greedy episodes END, with no reference line.

    python tools/traj_ends.py runs/xPET/traj_*.jsonl \
        --field maps/surf_petrus_lite.goal_32.npz \
        --zones maps/surf_petrus_lite.zones.json

`tools/eval_honesty.py` is the honest read on surf_src_cannonball, but it
needs a champion route (`--order-only 16` is an ORDER along that polyline) and
no such file exists for a map nobody has ever finished. This is the
route-free half of the same read, and it is the half that actually carried
Round 19's verdict: *one identical stopping point, reached by falling* is the
cannonball signature, and it is visible without any line at all.

Per eval file it prints:

  * frontier as **% of d0** - `d0` is the geodesic distance at the spawn, so
    `100 * (d0 - min d) / d0` is directly the number the trainer logs as
    `track a/b`, and comparable across maps;
  * **finishes**, tested against the map's own `zones.json` `end` box, padded,
    which is the only definition of success that does not need a route;
  * **where the episodes stop**: mean and spread of the last position, the
    wall clock, the horizontal speed and `vz` there. A tight spread with a
    large negative `vz` is a potential-barrier wall - every episode leaving at
    the same place, by falling - and is the thing to look at, not the
    percentage.

Champion-free by construction: the policy's own recordings, the map's own
distance field, the map's own finish zone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.goalfield import GoalField            # noqa: E402
from surfgym.route import episodes_from_traj       # noqa: E402


def load_field(path: Path) -> GoalField:
    z = np.load(path)
    return GoalField(z["grid"].astype(np.float32) * float(z["quant"]),
                     z["mins"], float(z["cell"]), float(z["reach_max"]))


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.exists() else ROOT / p


def in_box(xyz: np.ndarray, box: dict, pad: float) -> np.ndarray:
    lo = np.asarray(box["mins"], np.float64) - pad
    hi = np.asarray(box["maxs"], np.float64) + pad
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="+", help="record_rollout .jsonl files")
    ap.add_argument("--field", required=True, help="<map>.goal_32.npz")
    ap.add_argument("--zones", required=True, help="<map>.zones.json")
    ap.add_argument("--pad", type=float, default=64.0,
                    help="finish-box pad, u (default 64, as eval_honesty.py)")
    ap.add_argument("--tick-ms", type=float, default=10.0)
    a = ap.parse_args()

    field = load_field(_resolve(a.field))
    end_box = json.loads(_resolve(a.zones).read_text())["end"]

    print(f"{'eval file':<28} {'eps':>4} {'d0':>9} {'min d':>10} "
          f"{'% of d0':>8} {'fin':>5}  where the episodes end")
    tot_eps = tot_fin = 0
    for f in a.traj:
        eps = episodes_from_traj(_resolve(f))
        if not eps:
            continue
        d0s, mind, ends, fins = [], [], [], 0
        for ep in eps:
            xyz = ep[:, 1:4]
            d = field.sample(xyz)
            if not np.isfinite(d).any():
                continue
            d0s.append(float(d[0]))
            mind.append(float(np.nanmin(d)))
            fins += int(in_box(xyz, end_box, a.pad).any())
            last = ep[-1]
            ends.append([last[1], last[2], last[3],
                         float(np.hypot(last[4], last[5])), last[6],
                         float(last[0]) * a.tick_ms / 1000.0])
        if not ends:
            continue
        E = np.asarray(ends, np.float64)
        d0 = float(np.median(d0s))
        m = float(np.min(mind))
        pos, spread = E[:, :3].mean(0), E[:, :3].std(0)
        tot_eps += len(ends)
        tot_fin += fins
        print(f"{Path(f).name:<28} {len(ends):>4} {d0:>9,.0f} {m:>10,.0f} "
              f"{100.0 * (d0 - m) / max(d0, 1.0):>7.1f}% {fins:>2}/{len(ends):<2} "
              f"({pos[0]:,.0f}+-{spread[0]:.0f}, {pos[1]:,.0f}+-{spread[1]:.0f}, "
              f"{pos[2]:,.0f}+-{spread[2]:.0f}) "
              f"t {E[:, 5].mean():.1f}s vh {E[:, 3].mean():,.0f} "
              f"vz {E[:, 4].mean():,.0f}")
    print(f"\ntotal finishes {tot_fin}/{tot_eps}")


if __name__ == "__main__":
    main()
