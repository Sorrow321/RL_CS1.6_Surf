"""--goal-hull (uf2r2b3, 2026-10-31): a finish is the engine's trigger touch on the finish
trigger's brush model at the end of a tick - player box vs the model's bounds, then the model's
CLIP HULL for the player's size contains the origin (ReHLDS SV_TouchLinks; the rule the core
already applies to kill / push / teleport triggers) - instead of the swept segment vs the
trigger's bounding box. surf_unitfarmer2's finish (*23) is three 1 u curtains, so the box's NW
triangle and its interior are no finish in the game.

(a) the core finds *23 from the zones' end box and its clip hull is the curtains, not the box;
(b) stepping: a player flying into the NW triangle finishes under the box test and NOT under
    --goal-hull; one flying through the west curtain finishes under both;
(c) off = the core before the change, byte for byte (the same seeded rollout under the old and
    the new library, in subprocesses);
(d) the flag surface: the trainer flag, its dump and restore, the recorder and the tools mirror it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

UF2 = ROOT / "maps" / "surf_unitfarmer2.bsp"
ZONES = UF2.with_suffix(".zones.json")
LIB = ROOT / "build" / "libsurfcore.so"
BASE = "1214a468"     # the commit before the change (its src/ is the core without --goal-hull)
needs_uf2 = pytest.mark.skipif(not (UF2.exists() and ZONES.exists() and LIB.exists()),
                               reason="needs the built core and surf_unitfarmer2")


def _core(n=1):
    from surfgym.core import SurfCore, default_config
    return SurfCore(str(UF2), default_config(num_envs=n))      # kill_z auto (map min - 256)


def _end():
    return json.loads(ZONES.read_text(encoding="utf-8"))["end"]


@needs_uf2
def test_the_finish_is_three_curtains_not_the_box():
    c = _core()
    e = _end()
    m = c.find_box_model(e["mins"], e["maxs"])
    assert m == 23
    assert c.find_box_model([0, 0, 0], [1, 1, 1]) == -1
    inside = {(3400, -1950, -560): False,      # the box's empty NW triangle
              (3310, -1950, -560): False,      # where the round-1 incumbent entered
              (3700, -2300, -300): False,      # the box's interior
              (3340, -2300, -560): True,       # the west curtain, south of y -2144
              (3460, -2010, -560): True,       # the diagonal
              (3700, -1890, -300): True}       # the north curtain
    for p, want in inside.items():
        assert bool(c.model_contains(m, [p])[0]) is want, p
        assert bool(c.model_contains(m, [p], [True])[0]) is want, p
    with pytest.raises(ValueError):
        c.set_goal_hull([0, 0, 0], [1, 1, 1])


def _fly(c, origin, vel):
    from surfgym.core import STATE_DTYPE
    c.reset(0)
    st = np.zeros(1, STATE_DTYPE)
    st["origin"][0] = origin
    st["velocity"][0] = vel
    st["yaw"][0] = float(np.degrees(np.arctan2(vel[1], vel[0])))
    st["onground"][0] = -1
    c.set_state(0, st[0])
    hits = 0
    for _ in range(40):
        c.step(np.array([[7, 3, 1, 1, 0, 0]], np.int32))
        hits += int(c.goal_hits[0])
        if hits:
            break
    return hits


@needs_uf2
def test_the_triangle_finishes_only_under_the_box_rule():
    e = _end()
    # east into the NW triangle at the incumbent's entry height (no curtain on that path
    # for the first 40 ticks at 300 u/s: ~90 u)
    tri = ((3290.0, -1950.0, -560.0), (300.0, 0.0, 0.0))
    # east through the west curtain, south of y -2144
    west = ((3290.0, -2300.0, -560.0), (300.0, 0.0, 0.0))
    box = _core()
    box.set_goal_box(e["mins"], e["maxs"])
    assert _fly(box, *tri) == 1
    assert _fly(box, *west) == 1
    hull = _core()
    assert hull.set_goal_hull(e["mins"], e["maxs"]) == 23
    assert _fly(hull, *tri) == 0
    assert _fly(hull, *west) == 1
    hull.set_goal_model(-1)                              # back to the box test
    assert _fly(hull, *tri) == 1


_ROLL = r'''
import sys, hashlib, json, numpy as np
sys.path.insert(0, sys.argv[1] + "/python")
from surfgym.core import SurfCore, default_config
c = SurfCore(sys.argv[2], default_config(num_envs=16))
e = json.load(open(sys.argv[3]))["end"]
c.set_goal_box(e["mins"], e["maxs"])
c.set_teleport_fail(True)
c.reset(7)
rng = np.random.default_rng(3)
h = hashlib.sha256()
nv = [15, 7, 3, 3, 2, 2]
for t in range(1500):
    a = np.stack([rng.integers(0, k, 16) for k in nv], 1).astype(np.int32)
    obs, r, d, tr = c.step(a)[:4]
    h.update(np.ascontiguousarray(c.get_states()["origin"]).tobytes())
    h.update(np.asarray(d, np.uint8).tobytes())
    h.update(np.asarray(c.goal_hits, np.uint8).tobytes())
print(h.hexdigest())
'''


@needs_uf2
def test_off_is_the_core_before_the_change_byte_for_byte(tmp_path):
    # the core at the commit before the change, built beside the new one
    old = tmp_path / "libsurfcore_base.so"
    src = tmp_path / "src"
    src.mkdir()
    for f in ("bsp.c", "trace.c", "pm.c", "env.c", "sim.h", "surfcore.h"):
        r = subprocess.run(["git", "show", f"{BASE}:src/{f}"], capture_output=True, cwd=str(ROOT))
        assert r.returncode == 0, f
        (src / f).write_bytes(r.stdout)
    r = subprocess.run(["gcc", "-O2", "-fPIC", "-ffp-contract=off", "-fopenmp", "-DSURFCORE_BUILD",
                        "-fvisibility=hidden", "-shared"]
                       + [str(src / f) for f in ("bsp.c", "trace.c", "pm.c", "env.c")]
                       + ["-o", str(old), "-lm"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    script = tmp_path / "roll.py"
    script.write_text(_ROLL, encoding="utf-8")
    out = []
    for lib in (old, LIB):
        env = dict(os.environ, SURFCORE_DLL=str(lib), CUDA_VISIBLE_DEVICES="-1",
                   OMP_NUM_THREADS="2")
        r = subprocess.run([sys.executable, str(script), str(ROOT), str(UF2), str(ZONES)],
                           capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        out.append(r.stdout.strip().splitlines()[-1])
    assert out[0] == out[1] and len(out[0]) == 64


def test_the_flag_surface():
    tf = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--goal-hull", action="store_const", const=1, default=None' in tf
    assert '**({"goal_hull": 1} if args.goal_hull else {}),' in tf      # dumped only when set
    assert 'if args.goal_hull is None and ck_cfg.get("goal_hull"):' in tf   # a resume keeps it
    assert tf.count("set_goal_hull(") == 3                               # train, eval, held-out
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '{"goal_hull": cfg.get("goal_hull")}' in rc                   # MIRRORED
    assert rc.count("arm_finish(") == 2
    for f in ("beam_tas.py", "plan_to_bc.py", "beam_best_states.py"):
        assert "arm_finish(core, zones[\"end\"], cfg)" in (ROOT / "tools" / f).read_text(
            encoding="utf-8"), f
    from surfgym.core import arm_finish

    class _C:
        def __init__(self):
            self.calls = []

        def set_goal_box(self, a, b):
            self.calls.append("box")

        def set_goal_hull(self, a, b):
            self.calls.append("hull")
            return 23
    c = _C()
    box = {"mins": [0, 0, 0], "maxs": [1, 1, 1]}
    assert arm_finish(c, box, {}) == -1 and arm_finish(c, box, None) == -1
    assert arm_finish(c, box, {"goal_hull": 1}) == 23
    assert c.calls == ["box", "box", "hull"]
