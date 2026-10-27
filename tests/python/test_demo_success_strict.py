"""--demo-success-strict (pbr9b1, 2026-10-27): under --demo-success-energy the curriculum's outcome
for a spine episode is its energetic crossing alone (a finish without one is a failure). The pure
rule (train_fast.strict_outcomes), the trainer's print / dump / refusals (CPU, princessburglar), and
the flag off bit-identical to the trainer before it."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

BASE = "5b4765c"      # the commit before the flag

ENERGY = ["--demo-success-d", "50", "--demo-success-bonus", "50", "--demo-success-energy", "150"]


def test_strict_outcomes_count_the_energetic_crossing_alone():
    from train_fast import strict_outcomes
    finished = np.array([True, True, False, False])
    crossed = np.array([True, False, True, False])
    assert strict_outcomes(finished, crossed).tolist() == [True, False, True, False]
    # before the first reward call nothing has crossed: every ended episode failed, finish or not
    assert strict_outcomes(finished, None).tolist() == [False, False, False, False]
    # the default rule (a finish OR the crossing) would have counted row 1
    assert (finished | crossed).tolist() == [True, True, True, False]


@needs_pb
def test_trainer_says_dumps_and_refuses(tmp_path):
    run = "dss_on"
    try:
        r = _run_demo(run, ENERGY + ["--demo-success-strict"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-strict: a spine episode's curriculum outcome is its energetic" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_strict"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dss_bad", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                "--demo-success-strict"], tmp_path)
    assert bad.returncode != 0 and "give --demo-success-energy" in (bad.stdout + bad.stderr)
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
            r = _run_demo(run, ENERGY, tmp_path, script=script)
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
