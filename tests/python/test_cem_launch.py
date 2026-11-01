"""tools/cem_launch.py: the physics-exact CEM over the policy's own line (zlr2b4, 2026-11-01).

What must hold: a rollout is a pure function of (root state, control) - so the search's best is
reproducible and its replay is a line the core flies; the offset unit is the optimal strafe
angle (a zero offset with a side key held turns the velocity, a large one coasts); and the spine
is the line's own prefix up to the root plus the replayed flight, clocks zeroed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

MAP = ROOT / "maps" / "surf_za_loop.bsp"
FIELD = ROOT / "maps" / "surf_za_loop.goalc_32.npz"
pytestmark = pytest.mark.skipif(not (MAP.exists() and FIELD.exists()),
                                reason="surf_za_loop map/field not present")

cl = pytest.importorskip("cem_launch")
from surfgym.core import STATE_DTYPE, SurfCore, default_config  # noqa: E402


def _line(tmp_path, n=400):
    """A short 'line': the true spawn state stepped with no input (a fall), saved like
    beam_tas --save-frontier."""
    core = SurfCore(str(MAP), default_config(num_envs=1, spawn_mode=2, sv_maxvelocity=3500.0),
                    tick_ms=7)
    core.reset()
    from surfgym.zones import load_zones
    z = load_zones(str(MAP))
    sp = np.array(z["spawns"][0]["origin"] if isinstance(z.get("spawns"), list) and z["spawns"]
                  and isinstance(z["spawns"][0], dict) else core.get_states()[0]["origin"],
                  np.float32)
    row = core.get_states()[0:1].copy()
    row["origin"][0] = sp
    row["velocity"][0] = np.array([800.0, 0.0, 0.0], np.float32)
    core.set_state(0, row[0])
    states = []
    for _ in range(n):
        states.append(core.get_states()[0].copy())
        core.step(np.array([[7, 3, 1, 1, 0, 0]], np.int32))
    p = tmp_path / "line.npz"
    np.savez(p, states=np.array(states, dtype=STATE_DTYPE), spawn_state=row)
    return p


def _args(line, **kw):
    import argparse
    a = argparse.Namespace(line=str(line), map=str(MAP), field=str(FIELD), tick_ms=7.0,
                           maxvel=3500.0, horizon=0.2, coast=0.2, knot=8, sample_every=4)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_rollout_is_deterministic(tmp_path):
    S = cl.Searcher(_args(_line(tmp_path)))
    rng = np.random.default_rng(1)
    u = rng.normal(0, 3, S.nk)
    sd = rng.integers(0, 3, S.nk)
    fw = (rng.random(S.nk) < 0.5).astype(float)
    a = S.roll(10, u, sd, fw, keep=True)
    b = S.roll(10, u, sd, fw, keep=True)
    assert a[0] == b[0] and a[1] == b[1] and len(a[2]) == len(b[2])
    assert all(np.array_equal(np.asarray(x["origin"]), np.asarray(y["origin"]))
               for x, y in zip(a[2], b[2]))


def test_spine_is_prefix_plus_replay(tmp_path):
    line = _line(tmp_path)
    S = cl.Searcher(_args(line))
    root = 50
    res = [{"root": root, "t": root * 0.007, "own": 0.0, "cem": 0.0, "best_tick": 0,
            "u": [0.0] * S.nk, "sd": [1] * S.nk, "fw": [0.0] * S.nk,
            "horizon": 0.2, "coast": 0.2, "knot": 8}]
    best, bt, _ = S.roll(root, res[0]["u"], res[0]["sd"], res[0]["fw"])
    res[0]["cem"] = best
    rj = tmp_path / "result.json"
    rj.write_text(json.dumps(res))
    out = tmp_path / "spine.npy"
    assert cl.main(["--line", str(line), "--roots", "0", "--map", str(MAP), "--field", str(FIELD),
                    "--spine", f"{rj}:0", "--spine-out", str(out)]) == 0
    sp = np.load(out)
    src = np.load(line)["states"]
    assert sp.dtype == STATE_DTYPE
    assert np.array_equal(sp["origin"][:root + 1], src["origin"][:root + 1])
    assert len(sp) > root + 1
    assert not sp["tick"].any()
