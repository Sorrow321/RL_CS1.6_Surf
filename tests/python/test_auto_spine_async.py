"""--auto-spine-async (pbr7b2, 2026-10-26): the trainer's search runs BESIDE the training on a
copy of ckpt_latest instead of pausing it, and its spine is swapped in at the first iteration
after it ends. CPU, princessburglar: a smoke run in which a search starts, the training goes on
while it runs, it ends and its outcome is applied; the refusal without --auto-spine; TRAIN_ONLY
in record_ckpt; flag off bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import re
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
LOOP = ["--demo-success-d", "50", "--demo-success-bonus", "50",
        "--auto-spine", "--auto-spine-quorum", "1", "--auto-spine-patience", "1",
        "--auto-spine-envs", "16", "--auto-spine-ge-minutes", "0",
        "--auto-spine-extra-secs", "1", "--auto-spine-retry", "1",
        "--record-every", "2048", "--eval-eps", "2", "--ep-ticks", "300"]


def test_train_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert "auto_spine_async" in record_ckpt.TRAIN_ONLY


@needs_pb
def test_the_search_runs_beside_the_training(tmp_path):
    run = "aspa_smoke"
    try:
        # a longer run than the blocking smoke test: the search has to end while training goes on
        r = _run_demo(run, LOOP + ["--auto-spine-async", "--steps", "24576"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-async: the search runs BESIDE the training" in out
        m = re.search(r"\[\s*([\d,]+)\] auto-spine search 1: running beside the training", out)
        assert m, out[-6000:]
        started = int(m.group(1).replace(",", ""))
        e = re.search(r"\[\s*([\d,]+)\] auto-spine search 1: ended beside the training", out)
        assert e, out[-6000:]
        assert int(e.group(1).replace(",", "")) > started      # the training went on meanwhile
        assert ("NEW SPINE" in out) or ("NO new spine" in out)
        k1 = ROOT / "runs" / run / "auto" / "k01"
        assert (k1 / "ckpt.pt").exists()                        # the search's own weights
        log = (k1 / "auto_spine.log").read_text()
        assert "AUTO_SPINE " in log and str(k1 / "ckpt.pt") in log
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_async"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("aspa_bad", ["--auto-spine-async"], tmp_path)
    assert bad.returncode != 0 and "--auto-spine-async runs --auto-spine's search" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "aspa_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine-async" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    """The blocking loop (a search that fires and ends) and its numbers, against the trainer
    before the change."""
    pre = ROOT / "python" / "_train_fast_pre_aspa.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("aspa_off_new", TRAIN), ("aspa_off_old", pre)):
            r = _run_demo(run, LOOP, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "aspa_off_new", ROOT / "runs" / "aspa_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine_async" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        assert (a / "auto" / "k01" / "auto_spine.log").exists()
        assert not (a / "auto" / "k01" / "ckpt.pt").exists()      # the blocking path reads ckpt_latest
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
