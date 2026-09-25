"""--goal-planner prim (surfgym/goalprim.py): random motion primitives as the executor's plans.

(a) the curve: leaves along the velocity (the view direction below the speed floor), constant knots
    give a circular arc with the right heading change, the rate profile goes through its knots;
(b) the draw stays inside its ranges and the success table bins what it is told;
(c) the trainer + recorder on the small maze (CPU): a prim run trains, dumps its knobs, and
    record_ckpt.py records its checkpoint with the same kind of primitives.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from surfgym.goalprim import PrimitivePlanner, curve, rate_profile   # noqa: E402

LAB = ROOT / "maps_pool" / "labyrinth_left100.bsp"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_core = pytest.mark.skipif(not (DLL.exists() and LAB.exists()),
                                reason="needs the built core + maps_pool/labyrinth_left100")


def test_leaves_along_the_velocity():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(0, 600, 3)
        p = rng.uniform(-150, 150, 6)
        c = curve(np.zeros(3), v, 0.0, p, 2.0, 3, 300.0)
        d0 = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
        vd = v / np.linalg.norm(v)
        # pitch is clipped at +-85 deg, so compare only when the velocity is not steeper. The
        # first 10 ms CHORD is off by half that step's turn (<= 0.75 deg at 150 deg/s); the
        # tangent itself starts exactly along the velocity
        if abs(np.degrees(np.arcsin(vd[2]))) < 84:
            assert np.degrees(np.arccos(min(1.0, float(d0 @ vd)))) < 1.1


def test_slow_start_follows_the_view_and_the_floor():
    c = curve(np.zeros(3), np.zeros(3), 90.0, np.zeros(6), 2.0, 3, 300.0)
    assert np.allclose(c[-1], [0.0, 600.0, 0.0], atol=1.0)     # 2 s at 300 u/s along +y


def test_constant_knots_are_an_arc():
    c = curve(np.zeros(3), np.array([800.0, 0.0, 0.0]), 0.0,
              [90.0, 90.0, 90.0, 0.0, 0.0, 0.0], 2.0, 3, 300.0)
    tang = c[-1] - c[-2]
    assert abs(np.degrees(np.arctan2(tang[1], tang[0])) - 180.0) < 1.0   # turned 90 deg/s x 2 s
    r = 800.0 / np.radians(90.0)                                          # radius v / omega
    assert np.allclose(np.linalg.norm(c[:, :2] - [0.0, r], axis=1), r, atol=2.0)


def test_rate_profile_hits_its_knots():
    t = np.array([0.0, 1.0, 2.0])
    assert np.allclose(rate_profile([10.0, -40.0, 70.0], 2.0, t), [10.0, -40.0, 70.0])


def test_draw_ranges_and_table():
    P = PrimitivePlanner(side=100.0, down=50.0, up=20.0, n_envs=4)
    rng = np.random.default_rng(1)
    s = np.array([P.sample(rng) for _ in range(2000)])
    assert s[:, :3].min() >= -100 and s[:, :3].max() <= 100
    assert s[:, 3:].min() >= -50 and s[:, 3:].max() <= 20
    pl = P.goal(2, np.zeros(3), np.array([500.0, 0.0, 0.0]), 0.0, rng)
    assert pl.line.shape[1] == 3 and np.allclose(pl.line[0], 0.0) and not pl.finish
    P.note(2, True)
    P.note(2, False)
    assert int(P.bin_n.sum()) == 2 and int(P.bin_ok.sum()) == 1
    assert "primitive success" in P.table()


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


FLAGS = ["--map", str(LAB), "--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--goal-cell", "32",
         "--lidar-range", "11500", "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
         "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2", "--ep-ticks", "300",
         "--stall-secs", "30", "--maxvel", "4000", "--yaw-adaptive", "--respawn-frac", "0.9",
         "--respawn-margin", "0.1", "--respawn-reservoir", "1000", "--ckpt-every", "1e9",
         "--record-every", "4096", "--eval-eps", "2", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold",
         "--goals", "1", "--goal-obs", "fan", "--goal-planner", "prim", "--goal-reward", "arc",
         "--goal-fan-offsets", "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0", "--goal-kcap", "1"]


@needs_core
def test_trainer_and_recorder_run_prim():
    run = "prim_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "12288"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "MOTION PRIMITIVES" in r.stdout and "prim-eval" in r.stdout, r.stdout[-2000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["goal_planner"] == "prim" and cfg["prim_knots"] == 3 and cfg["prim_secs"] == 2.0
    out = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(out), "--episodes", "2"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    head = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert head["plan"]["planner"] == "prim" and len(head["plan"]["numbers"]) == 6
    shutil.rmtree(d, ignore_errors=True)
