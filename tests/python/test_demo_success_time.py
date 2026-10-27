"""--demo-success-time SECS (pbr9b3, 2026-10-27): under --demo-success-energy, a spine spawn's
crossing of the success line counts (pays the --demo-success-bonus, succeeds for the curriculum)
only if it comes within the spine's own rows from the spawn row to its crossing row plus SECS.
The pure gate (train_fast.time_gate), the trainer's print / dump / refusals (CPU,
princessburglar), and the flag off bit-identical to the trainer before it."""
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

BASE = "a3d63d2"      # the commit before the flag

GATE = ["--demo-success-d", "50", "--demo-success-bonus", "50", "--demo-success-energy", "150"]


def test_time_gate_counts_only_on_time_spine_crossings():
    from train_fast import time_gate
    hit = np.array([True, True, True, True, True, False])
    idx = np.array([100, 100, -1, 0, 950, 100])       # spawn rows (-1 off the spine, 0 the start)
    tick = np.array([890, 960, 5000, 5000, 40, 10])  # ticks since each spawn
    # the spine crosses its line at row 1,000; 20 ticks of slack: row 100 has 900 + 20
    ok = time_gate(hit, idx, tick, 1000, 20)
    assert ok.tolist() == [True, False, True, True, True, False]
    # a negative slack asks for a crossing faster than the spine's own
    ok2 = time_gate(hit, idx, tick, 1000, -20)
    assert ok2.tolist() == [False, False, True, True, False, False]   # row 4: 40 > 50 - 20
    # rows that do not cross are never on time
    assert not time_gate(np.zeros(6, bool), idx, tick, 1000, 10**6).any()


@needs_pb
def test_trainer_says_dumps_and_refuses(tmp_path):
    run = "dst_on"
    try:
        r = _run_demo(run, GATE + ["--demo-success-time", "0.5"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-time 0.5: a spine spawn at row i succeeds only if it crosses" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_time"] == 0.5
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra, msg in (
            ("dst_bad1", GATE[:4] + ["--demo-success-time", "0.5"], "rides on"),
            ("dst_bad2", GATE + ["--demo-success-time", "0.5", "--demo-success-alive", "1"],
             "not with --demo-success-alive")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), \
            (bad.stdout + bad.stderr)[-2000:]
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
            r = _run_demo(run, GATE, tmp_path, script=script)
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
