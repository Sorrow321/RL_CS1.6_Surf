"""--race-ground (the user, 2026-10-03, skate_laby: "maybe the potential field that we baked is bad
there"): the ground-bound goal field - standable floors linked by step-height climbs, drops and
ducked jumps through raised windows - on the map that needed it. The 3-D air field routed over its
32 u maze walls (start 87.5k u); the ground field walks the corridors (start ~119k u, which at the
1:14.90 record is ~1,590 u/s - the start push's speed, held). Skipped without maps/skate_laby.bsp
(not tracked)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SKATE = ROOT / "maps" / "skate_laby.bsp"
needs_skate = pytest.mark.skipif(not SKATE.exists(), reason="needs maps/skate_laby.bsp")


@needs_skate
def test_the_ground_field_walks_the_maze():
    from surfgym import SurfCore, default_config
    from surfgym.goalfield import build_goal_field, build_ground_field
    from surfgym.zones import load_zones
    core = SurfCore(str(SKATE), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    end = load_zones(str(SKATE))["end"]
    gw = build_ground_field(core, end)
    g3 = build_goal_field(core, end, 32.0)
    start = np.asarray([[-1056.0, -1000.0, -220.0]], np.float32)
    dw, d3 = float(gw.sample(start)[0]), float(g3.sample(start)[0])
    assert dw < gw.reach_max and d3 < g3.reach_max
    # the corridors are ~1.35x the air route over the walls
    assert dw > 1.25 * d3 and dw > 100_000
    # next to the finish both read ~0
    near = np.asarray([[2100.0, -352.0, -220.0]], np.float32)
    assert float(gw.sample(near)[0]) < 200.0


@needs_skate
def test_race_ground_trains_and_records():
    from test_obs_potential import ABS, RECORD, _run, _train
    run = "cya_ground"
    r = _train(run, ABS + ["--map", str(SKATE), "--obs-potential", "norm", "--race-ground", "1",
                           "--sv-friction", "0"], steps="6144")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["race_ground"] == 1 and cfg["sv_friction"] == 0.0
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"), "--map", str(SKATE),
                "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-3000:] + rec.stderr[-3000:]
    shutil.rmtree(d, ignore_errors=True)
