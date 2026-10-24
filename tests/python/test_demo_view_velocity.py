"""--demo-view-velocity and --demo-cooldown N (pbr2b2, 2026-10-24).

* surfgym.respawn.align_view_to_velocity: a row moving at >= 100 u/s horizontally gets yaw =
  its velocity heading ([0, 360)); slower rows and every other field are untouched.
* DemoCurriculum(cooldown=N): the position moves at most once per N iterations (20 = default).
* the trainer (CPU, princessburglar): both flags train with a spine and are dumped; the view flag
  without a spine / spawn-state file is refused; both flags off are bit-identical to the trainer
  before the change (config, progress.csv, the eval trajectories).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, _env, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def test_align_view_to_velocity():
    from surfgym.core import STATE_DTYPE
    from surfgym.respawn import align_view_to_velocity
    s = np.zeros(4, STATE_DTYPE)
    s["velocity"] = [[-300.0, 0.0, -50.0], [0.0, -400.0, 10.0], [60.0, 60.0, 0.0],
                     [99.0, 0.0, -900.0]]
    s["yaw"] = [10.0, 20.0, 30.0, 40.0]
    s["origin"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    s["ducked"] = [1, 0, 1, 0]
    out = align_view_to_velocity(s)
    assert out is not s
    np.testing.assert_allclose(out["yaw"], [180.0, 270.0, 30.0, 40.0], atol=1e-4)
    for k in STATE_DTYPE.names:
        if k != "yaw":
            np.testing.assert_array_equal(out[k], s[k])
    np.testing.assert_array_equal(s["yaw"], [10.0, 20.0, 30.0, 40.0])   # input unchanged


def test_demo_cooldown():
    from surfgym.core import STATE_DTYPE
    from surfgym.respawn import DemoCurriculum
    S = np.zeros(100, STATE_DTYPE)
    S["origin"][:, 0] = np.arange(100, dtype=np.float32)
    for cool, want in ((None, 20), (3, 3)):
        kw = {} if cool is None else {"cooldown": cool}
        dc = DemoCurriculum(S, window=5, min_ep=1, grow=5, **kw)
        dc.note_outcomes(np.arange(95, 100), np.ones(5))
        dc._move()
        assert dc.tau == 94 and dc._cool == want
        moves = 0
        for _ in range(40):
            lo, hi = dc._band()
            dc.note_outcomes(np.arange(lo, hi + 1), np.ones(hi - lo + 1))
            t = dc.tau
            dc._move()
            moves += dc.tau != t
        assert moves == 40 // want
    with pytest.raises(ValueError):
        DemoCurriculum(S, cooldown=0)


@needs_pb
def test_the_flags_train_and_are_dumped(tmp_path):
    run = "dvv_on"
    try:
        r = _run_demo(run, ["--demo-view-velocity", "--demo-cooldown", "5"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-view-velocity: --demo-file / --spawn-states rows" in out
        assert "--demo-cooldown 5: the curriculum position moves" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_view_velocity"] == 1 and cfg["demo_cooldown"] == 5
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    from test_ge_minutes import SMOKE
    bad = subprocess.run([sys.executable, "-u", str(TRAIN), "--run", "dvv_bad"] + SMOKE
                         + ["--steps", "2048", "--demo-view-velocity"], capture_output=True,
                         text=True, env=_env(), cwd=str(ROOT), timeout=600, encoding="utf-8",
                         errors="replace")
    assert bad.returncode != 0 and "give one of them" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dvv_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-view-velocity" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flags_off_are_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dvv.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("dvv_off_new", TRAIN), ("dvv_off_old", pre)):
            r = _run_demo(run, ["--record-every", "2048"], tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dvv_off_new", ROOT / "runs" / "dvv_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_view_velocity" not in ca and "demo_cooldown" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
        assert ta and [p.name for p in ta] == [p.name for p in tb]
        for p, q in zip(ta, tb):
            assert p.read_bytes() == q.read_bytes(), p.name
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
