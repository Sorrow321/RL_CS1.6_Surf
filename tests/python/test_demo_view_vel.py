"""--demo-view-vel and --demo-cooldown (pbr2b1, 2026-10-24).

--demo-view-vel: under --demo-file, spine spawn rows moving at >= 100 u/s horizontally spawn with
their view yaw along their horizontal velocity (a random-action search's spine carries a random
view the policy never holds); position, velocity and stance stay exact.
--demo-cooldown N: iterations between two curriculum moves (default 20, the old constant).

CPU: the helper re-aims only the fast rows; the curriculum moves on its cooldown; princessburglar:
a short spine run with both flags trains and dumps the keys; flags off are bit-identical to the
trainer before the change."""
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
from surfgym.core import STATE_DTYPE  # noqa: E402
from surfgym.respawn import DemoCurriculum, view_along_velocity  # noqa: E402
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def test_view_along_velocity_reaims_only_fast_rows():
    r = np.zeros(4, STATE_DTYPE)
    r["velocity"] = [[400.0, -300.0, -50.0], [50.0, 60.0, 0.0], [-100.0, 0.0, 300.0],
                     [0.0, 0.0, -900.0]]
    r["yaw"] = [139.0, 77.0, 5.0, 210.0]
    r["origin"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    r["ducked"] = [1, 0, 1, 0]
    out = view_along_velocity(r)
    assert out is not r and r["yaw"][0] == 139.0                 # a copy, input untouched
    assert np.isclose(out["yaw"][0], np.degrees(np.arctan2(-300.0, 400.0)) % 360.0)
    assert out["yaw"][1] == 77.0                                  # |v_h| 78 < 100: unchanged
    assert np.isclose(out["yaw"][2], 180.0)                       # exactly 100 u/s: re-aimed
    assert out["yaw"][3] == 210.0                                 # falling straight down
    for k in STATE_DTYPE.names:
        if k != "yaw":
            assert np.array_equal(out[k], r[k]), k
    assert len(view_along_velocity(np.zeros(0, STATE_DTYPE))) == 0


def test_cooldown_sets_the_pace():
    s = np.zeros(100, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(100)
    # a move at rebuild 1 and then every cd-th: 21 rebuilds = 2 moves at 20, 7 at 3
    for cd, want in ((20, 99 - 5 * 2), (3, 99 - 5 * 7)):
        c = DemoCurriculum(s, window=5, grow=5, min_ep=1, cooldown=cd)
        for _ in range(21):
            lo, hi = c._band()
            c.note_outcomes(np.arange(lo, hi + 1), np.ones(hi - lo + 1))
            c.build_pool(s[:1], pool_size=64)
        assert c.tau == want, (cd, c.tau)
    with pytest.raises(ValueError):
        DemoCurriculum(s, cooldown=0)


@needs_pb
def test_flags_train_and_are_dumped(tmp_path):
    run = "dvv_smoke"
    try:
        r = _run_demo(run, ["--demo-view-vel", "--demo-cooldown", "5"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-view-vel:" in out and "--demo-cooldown 5:" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_view_vel"] == 1 and cfg["demo_cooldown"] == 5
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-view-vel" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flags_off_are_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dvv.py"
    try:
        if _pre_change_trainer(pre) is None:
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
        assert ca == cb and "demo_view_vel" not in ca and "demo_cooldown" not in ca
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
