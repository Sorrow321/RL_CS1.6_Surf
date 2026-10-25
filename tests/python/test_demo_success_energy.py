"""--demo-success-energy E (pbr3b2, 2026-10-25): with --demo-success-bonus, the spine's success line
is paid (and counts as a curriculum success) only when the episode crosses it with an energy height
z + |v|^2/2g within E of the spine end's - the curriculum teaches a line's state, not only its
position. CPU, princessburglar: the trainer says so and dumps the key; without the bonus it is
refused; flag off is bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


@needs_pb
def test_the_energy_rule_is_on_and_dumped(tmp_path):
    run = "dse_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-success-energy", "40", "--record-every", "2048"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-energy 40: the line is paid / a success only with energy" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_energy"] == 40.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dse_bad", ["--demo-success-d", "50", "--demo-success-energy", "40"], tmp_path)
    try:
        assert bad.returncode != 0 and "needs --demo-success-d and --demo-success-bonus" in (
            bad.stdout + bad.stderr)
    finally:
        shutil.rmtree(ROOT / "runs" / "dse_bad", ignore_errors=True)


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
    import torch
    pre = ROOT / "python" / "_train_fast_pre_dse.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("dse_off_new", TRAIN), ("dse_off_old", pre)):
            r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                "--record-every", "2048"], tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
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
        sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
