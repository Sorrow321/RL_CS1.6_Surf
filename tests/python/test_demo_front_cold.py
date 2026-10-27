"""--demo-front-cold F (pbr8b4, 2026-10-27): under --demo-front-temp, a share F of the
frontier-band spawns runs COLD (no band heat) and the backward curriculum's advance / backoff
counts only the band episodes that ran cold (every off-band spawn counts as before). Measured
on princessburglar's finish tube: a band the policy finished 6/8 cold (greedy and stochastic)
scored 5-31% under the band heat, and the curriculum sat on it.

(a) the trainer (CPU, princessburglar): it says so, the config dumps the key; the refusals;
(b) flag off - with --demo-front-temp on - is bit-identical to the trainer before the change
    (config and progress.csv, fps aside).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
HOT = ["--demo-success-d", "50", "--demo-grow", "5", "--demo-front-frac", "0.67",
       "--demo-front-temp", "2", "--demo-front-ticks", "60", "--steps", "8192"]


@needs_pb
def test_on_dumped_and_refused(tmp_path):
    run = "dfc_on"
    try:
        r = _run_demo(run, HOT + ["--demo-front-cold", "0.5"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert ("--demo-front-cold 0.5: 50% of the band spawns run cold (no band heat); the "
                "curriculum's advance / backoff counts only the band episodes that ran cold") in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_front_cold"] == 0.5 and cfg["demo_front_temp"] == 2.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    for name, extra, msg in (
            ("dfc_bad0", ["--demo-success-d", "50", "--demo-front-cold", "0.5"],
             "--demo-front-cold splits --demo-front-temp's band spawns: give it"),
            ("dfc_bad1", HOT + ["--demo-front-cold", "1.5"], "--demo-front-cold F needs 0 < F < 1")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), name
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-front-cold" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dfc.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("dfc_off_new", TRAIN), ("dfc_off_old", pre)):
            r = _run_demo(run, HOT, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dfc_off_new", ROOT / "runs" / "dfc_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_front_cold" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
