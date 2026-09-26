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


def test_level_frame_leaves_level_along_the_horizontal_velocity():
    """--prim-frame level: the curve is laid on the velocity's projection onto the horizontal
    plane - it leaves level along the horizontal heading and is traced at the horizontal speed.
    The velocity frame (the default) dives with a falling agent."""
    v = np.array([400.0, 300.0, -900.0])
    lv = curve(np.zeros(3), v, 0.0, np.zeros(6), 2.0, 3, 300.0, frame="level")
    assert np.allclose(lv[:, 2], 0.0)                          # no height change at all
    assert np.allclose(lv[-1], [800.0, 600.0, 0.0], atol=1.0)  # 2 s x 500 u/s along (0.8, 0.6)
    vel = curve(np.zeros(3), v, 0.0, np.zeros(6), 2.0, 3, 300.0)
    assert vel[-1, 2] < -1000.0                                # the default follows the fall


def test_level_frame_draws_the_same_shape_climbing_or_falling():
    rng = np.random.default_rng(4)
    for _ in range(50):
        p = np.concatenate([rng.uniform(-180, 180, 3), rng.uniform(-120, 90, 3)])
        vxy = rng.normal(0, 500, 2)
        c = [curve(np.zeros(3), np.array([vxy[0], vxy[1], vz]), 0.0, p, 2.0, 3, 300.0,
                   frame="level") for vz in (-900.0, 0.0, 700.0)]
        assert np.array_equal(c[0], c[1]) and np.array_equal(c[1], c[2])


def test_level_frame_vertical_rates_bend_from_level():
    c = curve(np.zeros(3), np.array([500.0, 0.0, -800.0]), 0.0,
              [0.0, 0.0, 0.0, 30.0, 30.0, 30.0], 2.0, 3, 300.0, frame="level")
    t0, t1 = c[1] - c[0], c[-1] - c[-2]
    assert abs(np.degrees(np.arctan2(t0[2], np.hypot(t0[0], t0[1])))) < 0.2    # leaves level
    assert abs(np.degrees(np.arctan2(t1[2], np.hypot(t1[0], t1[1]))) - 60.0) < 0.5


def test_level_frame_slow_horizontal_follows_the_view_and_the_floor():
    # falling straight down: no horizontal heading, so the view yaw and the floor speed
    c = curve(np.zeros(3), np.array([10.0, 0.0, -900.0]), 90.0, np.zeros(6), 2.0, 3, 300.0,
              frame="level")
    assert np.allclose(c[-1], [0.0, 600.0, 0.0], atol=1.0)


def test_planner_threads_the_frame_and_refuses_an_unknown_one():
    v = np.array([300.0, -200.0, -600.0])
    p = np.array([40.0, -20.0, 10.0, -30.0, 15.0, 5.0])
    P = PrimitivePlanner(n_envs=1, frame="level")
    _line, end, _total = P.line_of(np.zeros(3), v, 0.0, p)
    assert np.allclose(end, curve(np.zeros(3), v, 0.0, p, 2.0, 3, 300.0, frame="level")[-1])
    assert "LEVEL frame" in P.describe()
    assert PrimitivePlanner(n_envs=1).frame == "velocity"
    with pytest.raises(ValueError):
        PrimitivePlanner(n_envs=1, frame="bogus")


def test_map_frame_headings_are_absolute_and_ignore_the_motion():
    """--prim-frame map: the sideways numbers are ABSOLUTE map headings at the knots - the agent's
    velocity direction changes nothing, only its horizontal speed sets the length."""
    p = [90.0, 90.0, 90.0, 0.0, 0.0, 0.0]                     # due +y, level
    for v in ([500.0, 0.0, 0.0], [-300.0, 400.0, -900.0], [0.0, -500.0, 600.0]):
        c = curve(np.zeros(3), np.array(v), 45.0, p, 2.0, 3, 300.0, frame="map")
        assert np.allclose(c[:, 2], 0.0) and np.allclose(c[:, 0], 0.0, atol=1e-6)
        assert np.allclose(c[-1], [0.0, 1000.0, 0.0], atol=1.0)          # 2 s x 500 u/s
    # slow: the floor speed, and still the planned heading (not the view yaw)
    c = curve(np.zeros(3), np.zeros(3), 0.0, [180.0, 180.0, 180.0, 0.0, 0.0, 0.0], 2.0, 3,
              300.0, frame="map")
    assert np.allclose(c[-1], [-600.0, 0.0, 0.0], atol=1.0)
    # height as in the level frame: leaves level, the vertical rates bend it
    c = curve(np.zeros(3), np.array([0.0, 500.0, -800.0]), 0.0,
              [0.0, 0.0, 0.0, 30.0, 30.0, 30.0], 2.0, 3, 300.0, frame="map")
    t0, t1 = c[1] - c[0], c[-1] - c[-2]
    assert abs(np.degrees(np.arctan2(t0[2], np.hypot(t0[0], t0[1])))) < 0.2
    assert abs(np.degrees(np.arctan2(t1[2], np.hypot(t1[0], t1[1]))) - 60.0) < 0.5


def test_map_frame_turns_the_short_way_across_the_seam():
    """Knot headings 170 / -170 / -150 are a 40 deg LEFT turn through 180, not a 320 deg swing."""
    c = curve(np.zeros(3), np.array([400.0, 0.0, 0.0]), 0.0,
              [170.0, -170.0, -150.0, 0.0, 0.0, 0.0], 2.0, 3, 300.0, frame="map")
    d = np.diff(c[:, :2], axis=0)
    h = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
    assert np.abs(np.diff(h)).max() < 1.0
    assert abs(h[0] - 170.0) < 1.5 and abs(h[-1] - 210.0) < 1.5


def test_map_frame_sample_draws_any_heading():
    P = PrimitivePlanner(n_envs=1, side=45.0, frame="map")
    rng = np.random.default_rng(2)
    s = np.array([P.sample(rng) for _ in range(3000)])
    assert s[:, :3].min() < -170.0 and s[:, :3].max() > 170.0   # headings, not --prim-side rates
    assert s[:, 3:].min() >= -120.0 and s[:, 3:].max() <= 90.0
    assert "MAP frame" in P.describe()


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


@needs_core
def test_trainer_and_recorder_run_prim_level_frame():
    """--prim-frame level: the trainer dumps it, prints it, and record_ckpt.py mirrors it."""
    run = "prim_smoke_level"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "4096", "--prim-frame", "level"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "LEVEL frame" in r.stdout, r.stdout[-2000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["prim_frame"] == "level"
    out = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(out), "--episodes", "1"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "LEVEL frame" in r2.stdout, r2.stdout[-2000:]
    shutil.rmtree(d, ignore_errors=True)


def test_a_loop_back_onto_its_own_end_is_redrawn():
    """A tight turn at the floor speed loops onto its start; its end sphere could be entered
    without following the curve. draw() never returns one."""
    P = PrimitivePlanner(n_envs=1, radius=192.0)
    rng = np.random.default_rng(3)
    for _ in range(400):
        p, line, end, total = P.draw(np.zeros(3), np.zeros(3), 0.0, rng)
        pts = curve(np.zeros(3), np.zeros(3), 0.0, p, P.secs, P.knots, P.floor)
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        early = pts[s < s[-1] - 2 * 192.0]
        assert len(early) == 0 or np.min(np.linalg.norm(early - pts[-1], axis=1)) >= 192.0
    loop = [180.0, 180.0, 180.0, 0.0, 0.0, 0.0]          # 360 deg in 2 s at 300 u/s
    pts = curve(np.zeros(3), np.zeros(3), 0.0, loop, 2.0, 3, 300.0)
    assert np.linalg.norm(pts[-1] - pts[0]) < 192.0       # the case the rule exists for
