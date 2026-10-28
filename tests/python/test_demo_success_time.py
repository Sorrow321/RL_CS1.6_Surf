"""--demo-success-time SECS (pbr9b2, 2026-10-27): a spine episode succeeds (and is paid the
--demo-success-bonus) only if its record crosses the success line no later than the spine itself
did from the row the episode spawned at, plus SECS. The pure gate (train_fast.time_gate_ok, and
its composition with train_fast.energy_gate as the trainer calls it), the trainer's print / dump /
refusals (CPU, princessburglar), and the flag off bit-identical to the trainer before it - with
--demo-success-energy on, the branch the flag touched."""
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

BASE = "c052bbb"      # the commit before the flag


def test_time_gate_allows_the_spines_own_pace_plus_slack():
    from train_fast import time_gate_ok
    # spine crosses its line at row 100; slack 10 ticks
    idx = np.array([0, 60, 60, 95, 100, 130, -1, -1])
    ticks = np.array([110, 50, 51, 15, 10, 11, 0, 99999])
    ok = time_gate_ok(ticks, idx, 100, 10)
    # row 0: 100 + 10 allowed; row 60: 40 + 10 -> 50 ok, 51 late; row 95: 5 + 10; at/past the
    # crossing row only the slack; off the spine never gated
    assert ok.tolist() == [True, True, False, True, True, False, True, True]
    assert time_gate_ok(np.array([0]), np.array([5]), 100, 0).tolist() == [True]


def test_time_gate_through_the_energy_gate_pays_only_crossings_in_time():
    from train_fast import energy_gate, time_gate_ok
    hit = np.array([True, True, False, False])
    ended = np.array([False, False, False, True])
    paid = np.array([False, False, True, False])
    idx = np.array([60, 60, 60, 60])
    ticks = np.array([50, 51, 20, 0])           # row 3 ended: its NEW spawn, tick 0
    h = np.array([3000.0, 3000.0, 3000.0, 3000.0])
    rec = np.array([100.0, 100.0, 100.0, 400.0])
    m = np.where(time_gate_ok(ticks, idx, 100, 10), h, -1e30)
    pay, ok, boot = energy_gate(hit, ended, paid, None, m, rec, 500.0, -1e29)
    assert pay.tolist() == [True, False, False, False]
    # the new spawn of row 3 is past the line (rec 400 <= 500) at tick 0: it counts as spawned past
    assert ok.tolist() == [True, False, True, True]
    assert boot.tolist() == [True, False, True, False]
    # with the energy rule too, both must hold
    pay2, _, _ = energy_gate(hit, ended, paid, None, np.where(time_gate_ok(ticks, idx, 100, 10),
                                                              np.array([2000.0, 3000, 3000, 3000]),
                                                              -1e30), rec, 500.0, 2500.0)
    assert pay2.tolist() == [False, False, False, False]


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
    for name, extra, msg in (
            ("dst_bad1", ["--demo-success-d", "50", "--demo-success-time", "0.5"], "give all three"),
            ("dst_bad2", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                          "--demo-success-time", "-1"], "must be >= 0")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr)
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


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


@needs_pb
def test_additive_time_bonus_says_dumps_and_refuses(tmp_path):
    run = "dstb_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-success-time", "0.5", "--demo-success-time-bonus", "25"],
                      tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-time-bonus 25: ADDITIVE" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_time_bonus"] == 25.0 and cfg["demo_success_time"] == 0.5
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dstb_bad", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                 "--demo-success-time-bonus", "25"], tmp_path)
    assert bad.returncode != 0 and "additive form of" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dstb_bad", ignore_errors=True)
