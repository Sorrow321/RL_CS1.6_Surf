"""--demo-milestone-energy D:H,... (pbr9b3, 2026-10-27): further milestones, each paid the
--demo-success-bonus once per episode whose record crosses distance D with energy height >= H; a
crossing without the energy is spent. The pure rule (train_fast.milestone_pay with floors), the
trainer's print / dump / refusals (CPU, princessburglar), and the flag off bit-identical to the
trainer before it (milestones on, so the changed milestone_pay is exercised)."""
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

BASE = "76735f4"      # the commit before the flag

BONUS = ["--demo-success-d", "50", "--demo-success-bonus", "50"]


def test_milestone_floors_pay_only_energetic_crossings():
    from train_fast import milestone_pay
    ms = np.array([5000.0, 3000.0])
    rec = np.array([4900.0, 4900.0, 2900.0, 6000.0])
    ended = np.array([False, False, False, True])
    h = np.array([2500.0, 1500.0, 1500.0, 9999.0])
    # ungated (no floors): the old rule
    n0, p0 = milestone_pay(rec, ended, ms, None)
    n1, p1 = milestone_pay(rec, ended, ms, None, h, np.array([-np.inf, -np.inf]))
    assert n0.tolist() == n1.tolist() and (p0 == p1).all()
    # first call: rows past a milestone at the start are marked paid, nothing pays
    assert n0.tolist() == [0, 0, 0, 0]
    rec2 = np.array([4800.0, 4800.0, 2800.0, 6000.0])
    p = np.zeros((4, 2), bool)
    n, p = milestone_pay(np.array([5100.0, 5100.0, 3100.0, 6000.0]), np.zeros(4, bool), ms, p,
                         h, np.array([2000.0, -np.inf]))
    assert n.tolist() == [0, 0, 0, 0]
    n, p = milestone_pay(rec2, np.zeros(4, bool), ms, p, h, np.array([2000.0, -np.inf]))
    # rows 0 / 1 cross 5,000: row 0 with the energy (paid), row 1 without (spent); row 2 crosses
    # both, 3,000 ungated (paid) and 5,000 gated without the energy (spent)
    assert n.tolist() == [1, 0, 1, 0]
    assert p.tolist() == [[True, False], [True, False], [True, True], [False, False]]
    n, p = milestone_pay(rec2, np.zeros(4, bool), ms, p, np.full(4, 9999.0),
                         np.array([2000.0, -np.inf]))
    assert n.tolist() == [0, 0, 0, 0]          # spent stays spent


@needs_pb
def test_trainer_says_dumps_and_refuses(tmp_path):
    run = "dme_on"
    try:
        r = _run_demo(run, BONUS + ["--demo-milestones", "40000",
                                    "--demo-milestone-energy", "30000:2500"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-milestone-energy: 30,000 for energy height >= 2,500" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_milestone_energy"] == "30000:2500"
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra in (("dme_bad1", ["--demo-milestone-energy", "30000:2500"]),
                        ("dme_bad2", BONUS + ["--demo-milestone-energy", "30000"])):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and "D:H" in (bad.stdout + bad.stderr), \
            (bad.stdout + bad.stderr)[-2000:]
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


@needs_pb
def test_flag_off_is_the_trainer_before_it(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dme.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--demo-milestone-energy" not in src
    pre.write_bytes(src)
    a, b = ROOT / "runs" / "dme_off_new", ROOT / "runs" / "dme_off_old"
    try:
        for run, script in ((a.name, ROOT / "python" / "train_fast.py"), (b.name, pre)):
            r = _run_demo(run, BONUS + ["--demo-milestones", "40000,30000"], tmp_path,
                          script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_milestone_energy" not in ca
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
