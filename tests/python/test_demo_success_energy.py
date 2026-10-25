"""--demo-success-energy M (pbr5b2, 2026-10-25): a spine episode succeeds (and is paid the
--demo-success-bonus) only if it crosses the success line with energy height >= the spine end
state's - M. The pure gate (train_fast.energy_gate), the trainer's print / dump / refusals (CPU,
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

BASE = "10456fb"      # the commit before the flag


def test_energy_gate_pays_only_energetic_crossings():
    from train_fast import energy_gate
    hit = np.array([True, True, False, False, False])
    ended = np.array([False, False, False, True, False])
    paid = np.array([False, False, True, True, False])
    h = np.array([3000.0, 2000.0, 2800.0, 3100.0, 3000.0])
    rec = np.array([100.0, 100.0, 100.0, 900.0, 600.0])     # row 3: its NEW spawn's record
    pay, ok, boot = energy_gate(hit, ended, paid, None, h, rec, 500.0, 2500.0)
    assert pay.tolist() == [True, False, False, False, False]
    # first call: ok starts as "spawned past the line with the energy" (row 2, row 3)
    assert boot.tolist() == [True, False, True, True, False]
    # an ended row re-anchors on its new spawn (rec 900 > 500: not past the line)
    assert ok.tolist() == [True, False, True, False, False]
    # a later crossing of row 1 with the energy: the episode was spent (paid by the caller)
    pay2, ok2, _ = energy_gate(np.array([False, False, False, False, True]),
                               np.zeros(5, bool), paid | hit, ok, h, rec, 500.0, 2500.0)
    assert pay2.tolist() == [False, False, False, False, True] and ok2[4]


@needs_pb
def test_trainer_says_dumps_and_refuses(tmp_path):
    run = "dse_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-success-energy", "150"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-energy 150: a spine episode succeeds only if it crosses" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_energy"] == 150.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra in (("dse_bad1", ["--demo-success-d", "50", "--demo-success-energy", "150"]),
                        ("dse_bad2", ["--demo-success-energy", "150"])):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and "give all three" in (bad.stdout + bad.stderr)
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


@needs_pb
def test_flag_off_is_the_trainer_before_it(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dse.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--demo-success-energy" not in src
    pre.write_bytes(src)
    a, b = ROOT / "runs" / "dse_off_new", ROOT / "runs" / "dse_off_old"
    try:
        for run, script in ((a.name, ROOT / "python" / "train_fast.py"), (b.name, pre)):
            r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50"],
                          tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_success_energy" not in ca
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
