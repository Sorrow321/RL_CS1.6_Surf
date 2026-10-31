#!/usr/bin/env python3
"""uf2_real_finish.py - re-score greedy evals with the ENGINE's finish rule (uf2r2b3, 2026-10-31).

The core ends an episode as a finish when the player's swept segment crosses the hull-inflated
AABB of the finish trigger (zones.json "end"), but GoldSrc fires a brush trigger only when the
player origin is inside the trigger model's clip hull at the end of a move (ReHLDS SV_TouchLinks).
unitfarmer2's finish (*23) is three 1 u curtains, so most of its AABB - its NW triangle and its
interior - is no finish at all. This replays the last recorded tick of every "done" episode by
one tick of its own velocity and asks the core's clip-hull test (surf_model_contains, the
--goal-hull rule) whether the finishing position is inside the trigger, standing or ducked as
the row's duck key says. Reporting only.

    python autoresearch/analysis/uf2_real_finish.py runs/<agent>/<run>/traj_*.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "autoresearch"))
import score as S  # noqa: E402
from surfgym.core import SurfCore, SurfEnvConfig  # noqa: E402

CFG = json.loads((REPO / "autoresearch" / "config.json").read_text(encoding="utf-8"))
BSP = REPO / CFG["map"]
ZONES = json.loads(BSP.with_suffix(".zones.json").read_text(encoding="utf-8"))
DUCK = 4


def real_finish(core, model, rows, tick_s):
    """(real?, how): the finishing tick's position = the last row + one tick of its velocity
    (and the swept points between), tested in the trigger's clip hull."""
    last = rows[-1]
    p0, v = last[1:4], last[4:7]
    ducked = bool(int(last[8]) & DUCK)
    pts = [p0 + v * tick_s * f for f in (1.0, 0.75, 0.5, 0.25, 0.0)]
    inside = core.model_contains(model, pts, [ducked] * len(pts))
    if inside.any():
        return True, "end"
    return False, "box only"


def main():
    core = SurfCore(str(BSP), SurfEnvConfig(num_envs=1))
    model = core.find_box_model(ZONES["end"]["mins"], ZONES["end"]["maxs"])
    if model < 1:
        raise SystemExit("the finish box is not a brush model's bounds")
    print(f"finish trigger model *{model}")
    for path in sys.argv[1:]:
        p = Path(path)
        eps = S.first_episodes(S.episodes(p), S.SCORE_EPISODES)
        box, real, t_real, t_box = 0, 0, [], []
        for h, rows, tr in eps:
            if tr.get("end") != "done":
                continue
            tick = float(h.get("tick_ms", 10.0)) / 1000.0
            box += 1
            t_box.append(len(rows) * tick)
            ok, _ = real_finish(core, model, rows, tick)
            if ok:
                real += 1
                t_real.append(len(rows) * tick)
        print(f"{p.name}: {len(eps)} eps, core finishes {box} (best {min(t_box) if t_box else float('nan'):.2f} s), "
              f"REAL {real}" + (f" (best {min(t_real):.2f} s)" if t_real else ""))


if __name__ == "__main__":
    main()
