"""--demo-success-time SECS (pbr9b4, 2026-10-27): a spine episode succeeds (and is paid the
--demo-success-bonus) only if it crosses the success line no later than the spine itself did
from the episode's spawn row, plus SECS. The pure gate (train_fast.time_gate_ok), its use through
energy_gate's bookkeeping, the trainer's print / dump / refusals (CPU, princessburglar), and the
flag off bit-identical to the trainer before it."""
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

BASE = "a9e11cb"      # the commit before the flag


def test_time_gate_ok_is_the_spines_clock_plus_slack():
    from train_fast import time_gate_ok
    # spine crossing row 100, slack 10 ticks
    idx = np.array([0, 50, 50, 50, -1, 100, 120, 99])
    tick = np.array([110, 60, 61, 0, 10 ** 6, 10 ** 6, 10 ** 6, 12])
    ok = time_gate_ok(tick, idx, 100, 10)
    # row 0: 100 + 10 = 110 ticks allowed; row 50: 60 allowed (61 is late); a fresh spawn (tick 0)
    # is in time; off the spine (-1), at or past the crossing row: ungated; row 99: 1 + 10 = 11
    assert ok.tolist() == [True, True, False, True, True, True, True, False]
    assert time_gate_ok(np.array([0]), np.array([0]), 0, 0).tolist() == [True]


def test_late_crossings_are_spent_through_energy_gate():
    """The trainer runs energy_gate on the 0 / 1 'gate holds' height: a late crossing is not paid,
    the episode's chance is spent, and the in-time one is paid and succeeds."""
    from train_fast import energy_gate, time_gate_ok
    hit = np.array([True, True, False])
    ended = np.zeros(3, bool)
    paid = np.zeros(3, bool)
    rec = np.array([90.0, 90.0, 900.0])
    g = time_gate_ok(np.array([40, 80, 5]), np.array([10, 10, 10]), 60, 5)   # allowed 55 ticks
    pay, ok, boot = energy_gate(hit, ended, paid, None, g.astype(np.float64), rec, 100.0, 0.5)
    assert pay.tolist() == [True, False, False]
    assert boot.tolist() == [True, False, False] and ok.tolist() == [True, False, False]
    # the late row's chance is spent: once paid (the caller marks it), it never pays again
    pay2, _, _ = energy_gate(np.zeros(3, bool), ended, paid | hit, ok,
                             np.ones(3), rec, 100.0, 0.5)
    assert not pay2.any()


@needs_pb
def test_trainer_says_dumps_and_refuses(tmp_path):
    run = "dst_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-success-time", "0.5"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-time 0.5: a spine episode succeeds only if it crosses" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_time"] == 0.5
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dst_bad", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                "--demo-success-time", "0.5", "--demo-success-alive", "1"],
                    tmp_path)
    assert bad.returncode != 0 and "rule of ONE fixed spine" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dst_bad", ignore_errors=True)


@needs_pb
def test_flag_off_is_the_trainer_before_it(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dst.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--demo-success-time" not in src
    pre.write_bytes(src)
    a, b = ROOT / "runs" / "dst_off_new", ROOT / "runs" / "dst_off_old"
    try:
        for run, script in ((a.name, ROOT / "python" / "train_fast.py"), (b.name, pre)):
            r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                "--demo-success-energy", "150"], tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_success_time" not in ca
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
