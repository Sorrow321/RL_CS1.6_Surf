"""plan_progress.py - greedy evals scored ALONG a planner route, one line per eval.

For every traj_<step>.jsonl of each run (held-out and recording files skipped): each episode's
furthest progress along the route under the --race-arc rule (surfgym.route.ArcProgress,
corridor 1,500 u, window 16 - the anchor cannot jump across the map) and whether it ended in
the finish box. The same yardstick for a flat arm and a planner arm on the same map.

    python tools/plan_progress.py <route.npz> <finish_zone_map.bsp> <run_dir> [<run_dir> ...]
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.route import ArcProgress, episodes_from_traj   # noqa: E402
from surfgym.zones import load_zones                        # noqa: E402

route, bsp = sys.argv[1:3]
z = np.load(route)
line = np.asarray(z["route"], np.float64)
spacing = float(z["spacing"])
L = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
end = load_zones(bsp)["end"]
fm, fx = np.asarray(end["mins"], np.float64), np.asarray(end["maxs"], np.float64)

for run in sys.argv[3:]:
    files = sorted(f for f in Path(run).glob("traj_*.jsonl")
                   if re.fullmatch(r"traj_\d+\.jsonl", f.name))
    print(f"== {Path(run).name}: route {L:,.0f} u")
    for f in files:
        step = int(f.stem.split("_")[1])
        best, fins = [], 0
        for e in episodes_from_traj(str(f)):
            p = np.asarray(e, np.float64)[:, 1:4]
            ap = ArcProgress(line, spacing=spacing)
            ap.reset(p[:1])
            a0, amax = float(ap.arc[0]), float(ap.arc[0])
            for q in p[1:]:
                ap.advance(q[None, :])
                amax = max(amax, float(ap.arc[0]))
            best.append(amax - a0 if amax > a0 else 0.0)
            fins += bool(np.all((p[-1] >= fm - 64) & (p[-1] <= fx + 64)))
        b = np.array(best)
        print(f"  {step / 1e6:8.0f}M  along the plan: max {b.max():6,.0f} u ({100 * b.max() / L:5.1f}%)  "
              f"median {np.median(b):6,.0f} u  finished {fins}/{len(b)}")
