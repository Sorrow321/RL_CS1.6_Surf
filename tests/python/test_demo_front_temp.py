"""--demo-front-temp T (pbr2b3, 2026-10-24): under --demo-file, an env whose episode STARTED on
the backward curriculum's frontier band samples the keys and the yaw at 1 + T (the yaw sigma
at --demo-front-view when given; the camera-only pitch head stays cold) for its first
--demo-front-ticks physics ticks; every other env keeps its own temperature, and PPO scores
each row at the temperature it was drawn at (the --front-temp slots).

(a) the hot mask: band membership, spawns off the spine, the window;
(b) the trainer (CPU, princessburglar): it says so, the step line carries the hot share, the
    config dumps the keys; the refusals;
(c) flag off - with and without --front-temp on the same run - is bit-identical to the trainer
    before the change (config and progress.csv, fps aside).
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
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def test_the_hot_mask():
    from train_fast import demo_hot_mask
    idx = np.array([-1, 3, 4, 7, 8, 9, 5, 6])
    ln = np.array([0, 0, 99, 50, 0, 0, 100, 150])
    m = demo_hot_mask(idx, (4, 8), ln, 100)
    # -1 is off the spine, 3 and 9 are outside the band, 5 / 6 are past the window
    assert m.tolist() == [False, False, True, True, True, False, False, False]
    assert not demo_hot_mask(np.full(4, -1), (0, 10), np.zeros(4), 100).any()


@needs_pb
def test_on_dumped_and_refused(tmp_path):
    run = "dft_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-front-temp", "2",
                            "--demo-front-ticks", "60", "--demo-front-view", "8",
                            "--steps", "8192"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert ("--demo-front-temp 2: an env whose episode started on the curriculum's "
                "frontier band samples the keys at 3 and the yaw sigma x 8 for its first "
                "60 ticks") in out
        assert "demohot " in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_front_temp"] == 2.0 and cfg["demo_front_ticks"] == 60
        assert cfg["demo_front_view"] == 8.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra, msg in (
            ("dft_bad0", ["--demo-front-temp", "0"], "--demo-front-temp T needs T > 0"),
            ("dft_bad1", ["--demo-front-ticks", "60"], "need --demo-front-temp")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), name
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-front-temp" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
@pytest.mark.parametrize("front", [False, True])
def test_flag_off_is_bit_identical(tmp_path, front):
    pre = ROOT / "python" / "_train_fast_pre_dft.py"
    extra = ["--demo-success-d", "50", "--demo-grow", "5", "--demo-front-frac", "0.67",
             "--steps", "8192"]
    if front:
        extra += ["--front-temp", "1", "--front-band", "0.01"]
    tag = "f" if front else "n"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in ((f"dft_off_new_{tag}", TRAIN), (f"dft_off_old_{tag}", pre)):
            r = _run_demo(run, extra, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / f"dft_off_new_{tag}", ROOT / "runs" / f"dft_off_old_{tag}"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_front_temp" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
