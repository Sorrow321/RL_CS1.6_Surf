"""train_fast --conveyor / SURF_CONVEYOR=1 (pbr5b3, 2026-10-26): the core's experimental conveyor physics.

* off (SURF_CONVEYOR unset): a hull dropped onto surf_princessburglar_b26's upward func_conveyor *162
  lands and slides on it with ground friction, exactly as before the change (the plate is a floor);
* on: the same drop is launched - one frame of friction on the plate, then vz ~ +250 (the conveyor's
  speed) and the horizontal speed kept, what the WR demo shows there;
* the rollout with the flag off is bit-identical to the core built from the parent commit's sources
  (BASE_SURFCORE_DLL, when given: the test builds nothing itself).

Each case runs in a subprocess: the core reads SURF_CONVEYOR when it loads the map.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "maps" / "surf_princessburglar_b26.bsp"
SPINE = ROOT / "autoresearch" / "spines" / "pbr4b3_auto_t11k05.npy"

PROBE = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1] + "/python")
from surfgym.core import SurfCore, default_config
cfg = default_config(num_envs=1, spawn_mode=2, max_episode_ticks=6000, water_fail=1, yaw_jitter_deg=0.0,
                     lidar_w=0, lidar_h=0)
cfg.phys.sv_maxvelocity = 3500.0
cfg.phys.msec = 8
core = SurfCore(sys.argv[2], cfg)
core.set_teleport_fail(True)
sp = np.load(sys.argv[3])
core.reset(0)
s0 = sp[4405:4406].copy(); s0["tick"] = 0; s0["stuck_ticks"] = 0
s0["origin"][0] = (-2345.0, 620.0, 2600.0)       # 20 u over the middle of conveyor *162
s0["velocity"][0] = (700.0, 300.0, -500.0)
s0["onground"] = -1
core.set_state(0, s0)
out = []
for t in range(int(sys.argv[4])):
    _, _, done, trunc, _ = core.step(np.array([[7, 3, 1, 1, 0, 0]], np.int32))
    s = core.get_states()[0]
    out.append([float(x) for x in s["origin"]] + [float(x) for x in s["velocity"]] + [int(s["onground"])])
    if done[0] or trunc[0]:
        break
print(json.dumps(out))
'''


def _run(env_extra, ticks=30, dll=None):
    env = dict(os.environ)
    env.pop("SURF_CONVEYOR", None)
    env.update(env_extra)
    if dll:
        env["SURFCORE_DLL"] = str(dll)
    r = subprocess.run([sys.executable, "-c", PROBE, str(ROOT), str(MAP), str(SPINE), str(ticks)],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return np.asarray(json.loads(r.stdout.strip().splitlines()[-1]), np.float64)


pytestmark = pytest.mark.skipif(not (MAP.exists() and SPINE.exists()),
                                reason="needs the princessburglar map and the k05 spine")


def test_off_plate_is_a_floor():
    tr = _run({})
    og = tr[:, 6]
    on = np.flatnonzero(og > 0)
    assert len(on) >= 10, "the hull should stand on the plate for many ticks"
    vh = np.hypot(tr[:, 3], tr[:, 4])
    assert vh[on[-1]] < vh[on[0]] - 150, "ground friction should eat the horizontal speed"
    assert tr[:, 5].max() < 1.0


def test_on_plate_launches():
    tr = _run({"SURF_CONVEYOR": "1"})
    og = tr[:, 6]
    first = int(np.flatnonzero(og > 0)[0])
    vh = np.hypot(tr[:, 3], tr[:, 4])
    assert tr[first + 2, 5] > 200.0, f"vz after the plate {tr[first + 2, 5]:.1f}: no launch"
    assert (og[first + 1:first + 10] <= 0).all(), "the launched hull should be airborne"
    assert vh[first + 9] > vh[first] - 40.0, "at most one frame of friction"


def test_off_bit_identical_to_base():
    base = os.environ.get("BASE_SURFCORE_DLL")
    if not base:
        pytest.skip("BASE_SURFCORE_DLL not set (the parent commit's core)")
    a = _run({}, ticks=60, dll=base)
    b = _run({}, ticks=60)
    assert a.shape == b.shape and np.array_equal(a, b)
