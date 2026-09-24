"""plan_route.py - export a map's start->finish PLAN as a --race-arc route file.

The line is the goal planner's shortest route on its graph (goalplan.BFSPlanner, the walking or
the surf graph), so it comes from map GEOMETRY only: no recording, no demo, no policy. It is
the same line the stage-1 executor is shown through the fan at test time (BFSPlanner.plan:
Douglas-Peucker at one cell, resampled at 128 u).

    python tools/plan_route.py <map.bsp> <out.npz> [ride|tight|walk] [cell]

Every spawn of the map's race pool is planned; the median-length plan is written (the
reward's ArcProgress anchors each spawn on the line globally, so one line serves them all).
The npz carries route (L, 3), spacing, and provenance: source = "planner:<graph>", cell, the
spawn it starts from and every spawn's plan length.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from surfgym.core import SurfCore, default_config    # noqa: E402
from surfgym.goalplan import BFSPlanner              # noqa: E402
from surfgym.rewards import map_spawn_pool           # noqa: E402
from surfgym.zones import load_zones                 # noqa: E402

bsp, out = sys.argv[1:3]
kind = sys.argv[3] if len(sys.argv) > 3 else "ride"
cell = float(sys.argv[4]) if len(sys.argv) > 4 else 32.0
if kind not in ("ride", "tight", "walk"):
    raise SystemExit(f"graph must be ride, tight or walk, not {kind!r}")

core = SurfCore(bsp, default_config(num_envs=1, spawn_mode=1, lidar_w=0, lidar_h=0))
core.reset(0)
spawns = np.asarray(map_spawn_pool(core)["origin"], np.float64)
g = BFSPlanner.for_core(core, cell, load_zones(bsp)["end"], n_targets=0, seed=0,
                        **({"graph_kind": kind} if kind in ("ride", "tight") else {}))
core.close()
print(g.describe())
if g.fin is None:
    raise SystemExit("no finish box on this map")
plans = [g.plan(s, g.fin) for s in spawns]
ok = [i for i, p in enumerate(plans) if p is not None]
if not ok:
    raise SystemExit("no spawn reaches the finish on this graph")
lens = np.array([plans[i].length for i in ok])
pick = ok[int(np.argsort(lens)[len(lens) // 2])]
pl = plans[pick]
line = np.asarray(pl.line, np.float64)
seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
spacing = float(seg.mean())            # the resample is uniform: total / (n - 1)
np.savez(out, route=line.astype(np.float32), spacing=np.float64(spacing),
         source=np.array(f"planner:{kind}"), cell=np.float64(cell),
         spawn=spawns[pick], plan_lengths=np.array([p.length if p is not None else np.nan
                                                    for p in plans]))
print(f"{out}: {len(line)} pts @ {spacing:.2f} u (segments {seg.min():.2f}-{seg.max():.2f}) = "
      f"{seg.sum():,.0f} u of line, graph route {pl.length:,.0f} u, from spawn {pick} "
      f"{spawns[pick].round(0).tolist()}; {len(ok)}/{len(spawns)} spawns reach the finish, "
      f"plans {lens.min():,.0f}-{lens.max():,.0f} u")
print(f"line z {line[:, 2].min():.0f}..{line[:, 2].max():.0f}, x {line[:, 0].min():.0f}..{line[:, 0].max():.0f}, "
      f"y {line[:, 1].min():.0f}..{line[:, 1].max():.0f}, ends {line[-1].round(0).tolist()}")
