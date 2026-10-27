"""--demo-success-strict (pbr9b3, 2026-10-27): under --demo-success-energy the curriculum's outcome
is the gated crossing alone - a finish is no success of its own. The trainer's print-free dump and
refusal (CPU, princessburglar) and the flag off bit-identical to the trainer before it."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

BASE = "f146a8b"      # the commit before the flag

GATE = ["--demo-success-d", "50", "--demo-success-bonus", "50", "--demo-success-energy", "150"]


@needs_pb
def test_trainer_dumps_and_refuses(tmp_path):
    run = "dss_on"
    try:
        r = _run_demo(run, GATE + ["--demo-success-strict"], tmp_path)
        assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_strict"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dss_bad", GATE[:4] + ["--demo-success-strict"], tmp_path)
    assert bad.returncode != 0 and "--demo-success-strict makes" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dss_bad", ignore_errors=True)


@needs_pb
def test_flag_off_is_the_trainer_before_it(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dss.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--demo-success-strict" not in src
    pre.write_bytes(src)
    a, b = ROOT / "runs" / "dss_off_new", ROOT / "runs" / "dss_off_old"
    try:
        for run, script in ((a.name, ROOT / "python" / "train_fast.py"), (b.name, pre)):
            r = _run_demo(run, GATE, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_success_strict" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        pre.unlink(missing_ok=True)
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
