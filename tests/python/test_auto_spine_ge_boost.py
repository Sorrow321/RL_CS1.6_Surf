"""--auto-spine-ge-boost (pbr8b3, 2026-10-27): the loop's search runs its Go-Explore stages on the
given ladder (tools/auto_spine.py --ge-boost), e.g. unboosted roots as a stage of their own.
TRAIN_ONLY in record_ckpt; a smoke run in which the trainer hands the ladder to the search; the
refusals; flag off bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
LOOP = ["--demo-success-d", "50", "--demo-success-bonus", "50",
        "--auto-spine", "--auto-spine-quorum", "1", "--auto-spine-patience", "1",
        "--auto-spine-envs", "16", "--auto-spine-ge-minutes", "0",
        "--auto-spine-extra-secs", "1", "--auto-spine-retry", "1",
        "--record-every", "2048", "--eval-eps", "2", "--ep-ticks", "300"]


def test_train_only():
    import record_ckpt
    assert "auto_spine_ge_boost" in record_ckpt.TRAIN_ONLY


@needs_pb
def test_the_trainer_hands_the_ladder_to_the_search(tmp_path):
    run = "asgb_smoke"
    try:
        r = _run_demo(run, LOOP + ["--auto-spine-ge-boost", "1.0;1.04"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-ge-boost 1.0;1.04: the search's Go-Explore stages" in out
        assert "auto-spine search 1: " in out and "--ge-boost 1.0;1.04" in out
        log = (ROOT / "runs" / run / "auto" / "k01" / "auto_spine.log").read_text()
        assert "AUTO_SPINE " in log
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_ge_boost"] == "1.0;1.04"
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra, msg in (("asgb_bad1", ["--auto-spine-ge-boost", "1.0"], "give --auto-spine"),
                             ("asgb_bad2", LOOP + ["--auto-spine-ge-boost", "x;1.0"],
                              "positive speed factors")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), name
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine-ge-boost" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    """The blocking loop (a search that fires and ends) and its numbers, against the trainer
    before the change."""
    pre = ROOT / "python" / "_train_fast_pre_asgb.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("asgb_off_new", TRAIN), ("asgb_off_old", pre)):
            r = _run_demo(run, LOOP, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "asgb_off_new", ROOT / "runs" / "asgb_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine_ge_boost" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
