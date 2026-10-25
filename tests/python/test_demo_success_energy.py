"""--demo-success-energy TOL (pbr3b3, 2026-10-25, the user's momentum-ramp insight): with
--demo-success-d and --demo-success-bonus, a spine episode succeeds (and is paid) only while its
record is past the success line AND its energy height z + |v|^2/2g is at least the spine end's
minus TOL - so a line that buys depth with momentum is not matched by a slower line reaching the
same depth. CPU, princessburglar: the rule is announced and dumped; it needs the bonus; an energy
threshold no state reaches pays nothing where the plain rule pays; flag off is bit-identical to
the trainer before the flag."""
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
PAY = "--demo-success-bonus: first payment at step"


def _go(run, extra, tmp_path, script=TRAIN):
    r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50"] + list(extra),
                  tmp_path, script=script)
    return r, r.stdout + r.stderr


@needs_pb
def test_the_rule_is_on_dumped_and_needs_the_bonus(tmp_path):
    run = "dse_on"
    try:
        r, out = _go(run, ["--demo-success-energy", "30"], tmp_path)
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-energy 30: a spine episode succeeds (and is paid) only past" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_energy"] == 30.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dse_bad", ["--demo-success-d", "50", "--demo-success-energy", "30"], tmp_path)
    shutil.rmtree(ROOT / "runs" / "dse_bad", ignore_errors=True)
    assert bad.returncode != 0 and "--demo-success-energy conditions the --demo-success-bonus" in (
        bad.stdout + bad.stderr)


@needs_pb
def test_an_unreachable_energy_pays_nothing(tmp_path):
    """The success line placed 2 u inside the true start on the shaping field: an episode from
    the map start crosses it with its first step toward the goal, so the plain rule pays; an
    energy threshold no state reaches must pay nothing."""
    import numpy as np
    from surfgym.goalfield import GoalField
    from test_demo_success_d import _spine
    from test_ge_minutes import SPAWN
    zf = np.load(ROOT / "maps" / "surf_princessburglar_b26.goalc_32.npz")
    f = GoalField(zf["grid"].astype(np.float32) * float(zf["quant"]), zf["mins"],
                  float(zf["cell"]), float(zf["reach_max"]))
    S = np.load(_spine(tmp_path))
    d_end = float(f.sample(np.asarray(S[-1]["origin"], np.float64)[None])[0])
    d0 = float(f.sample(np.asarray(SPAWN, np.float64)[None])[0])
    m = f"{d0 - d_end - 2.0:.3f}"
    try:
        ra = _run_demo("dse_plain", ["--demo-success-d", m, "--demo-success-bonus", "50",
                                     "--steps", "8192"], tmp_path)
        rb = _run_demo("dse_high", ["--demo-success-d", m, "--demo-success-bonus", "50",
                                    "--steps", "8192", "--demo-success-energy", "-100000"],
                       tmp_path)
        outa, outb = ra.stdout + ra.stderr, rb.stdout + rb.stderr
        assert ra.returncode == 0 and rb.returncode == 0, (outa + outb)[-4000:]
        assert PAY in outa, outa[-3000:]
        assert PAY not in outb
    finally:
        for run in ("dse_plain", "dse_high"):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-success-energy" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dse.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("dse_off_new", TRAIN), ("dse_off_old", pre)):
            r, out = _go(run, [], tmp_path, script=script)
            assert r.returncode == 0, out[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dse_off_new", ROOT / "runs" / "dse_off_old"
    try:
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
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
