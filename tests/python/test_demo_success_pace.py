"""--demo-success-pace FRAC SECS (pbr9b1, 2026-10-28): a spine episode succeeds (and is paid the
--demo-success-bonus) only if it crosses the success line no slower than the spine did from the
episode's spawn row. The pure rules (train_fast.pace_allowed / pace_gate), the trainer's print /
dump / refusals (CPU, princessburglar), and the flag off bit-identical to the trainer before it."""
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

BASE = "b727719"      # the commit before the flag

LINE = ["--demo-success-d", "50", "--demo-success-bonus", "50"]


def test_pace_allowed_is_the_spines_own_time_plus_slack():
    from train_fast import pace_allowed
    a = pace_allowed(10, 8, 0.25, 3)
    # row 0: 8 rows to the crossing row x 1.25 + 3 = 13; row 6: 2 x 1.25 + 3 = 5.5 -> 5
    assert a.tolist() == [13, 11, 10, 9, 8, 6, 5, 4, 3, 3]
    assert pace_allowed(4, 2, 0.0, 0).tolist() == [2, 1, 0, 0]


def test_pace_gate_pays_only_paced_crossings_on_the_spine():
    from train_fast import pace_gate
    allowed = np.array([10, 8, 6, 4])
    hit = np.array([True, True, True, False, False])
    ended = np.array([False, False, False, True, False])
    paid = np.array([False, False, False, False, True])
    idx = np.array([0, 1, -1, 2, 3])
    elapsed = np.array([10, 9, 1, 0, 0])
    rec = np.array([90.0, 90.0, 90.0, 300.0, 50.0])     # row 3: its NEW spawn's record
    pay, ok, boot = pace_gate(hit, ended, paid, None, idx, elapsed, allowed, rec, 100.0)
    # row 0 in time (10 <= 10), row 1 late (9 > 8), row 2 off the spine: never paced
    assert pay.tolist() == [True, False, False, False, False]
    # row 4 spawned past the line: a success, unpaid
    assert boot.tolist() == [True, False, False, False, True]
    # the ended row re-anchors on its new spawn (record 300 > 100: not past the line)
    assert ok.tolist() == [True, False, False, False, True]


@needs_pb
def test_trainer_says_dumps_and_refuses(tmp_path):
    run = "dsp_on"
    try:
        r = _run_demo(run, LINE + ["--demo-success-pace", "0.05", "0.25"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-pace 0.05 0.25: a spine episode succeeds only if it crosses" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_pace"] == [0.05, 0.25]
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra, msg in (
            ("dsp_bad1", ["--demo-success-d", "50", "--demo-success-pace", "0.05", "0.25"],
             "give both"),
            ("dsp_bad2", LINE + ["--demo-success-energy", "150", "--demo-success-pace", "0", "0"],
             "its own success rule")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), (bad.stdout + bad.stderr)[-2000:]
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


@needs_pb
def test_flag_off_is_the_trainer_before_it(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dsp.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--demo-success-pace" not in src
    pre.write_bytes(src)
    a, b = ROOT / "runs" / "dsp_off_new", ROOT / "runs" / "dsp_off_old"
    try:
        for run, script in ((a.name, ROOT / "python" / "train_fast.py"), (b.name, pre)):
            r = _run_demo(run, LINE, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_success_pace" not in ca
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
