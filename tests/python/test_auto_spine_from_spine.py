"""--auto-spine-from-spine (pbr8b3, 2026-10-27): the search's beams start from the CURRENT spine
S seconds before it reaches the greedy frontier (tools/auto_spine.py --start-spine / --start-row)
instead of the true spawn, and the new line is the spine's rows before that row + the beam's own
states. TRAIN_ONLY in record_ckpt; a smoke run in which the trainer hands the spine to the search
and the beam starts on it; the refusal; flag off bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
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
    import record_ckpt
    assert "auto_spine_from_spine" in record_ckpt.TRAIN_ONLY


@needs_pb
def test_the_beam_starts_on_the_spine(tmp_path):
    run = "asfs_smoke"
    try:
        r = _run_demo(run, LOOP + ["--auto-spine-from-spine", "0.05"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-from-spine 0.05: the search's beams start from the current spine" in out
        assert "auto-spine search 1: " in out and "--start-spine" in out and "--start-row" in out
        k1 = ROOT / "runs" / run / "auto" / "k01"
        assert (k1 / "start_spine.npy").exists()
        log = (k1 / "auto_spine.log").read_text()
        assert "AUTO_SPINE " in log
        blog = (k1 / "beam.log").read_text()
        assert "--start-state start_spine.npy[" in blog          # the population starts on it
        res = json.loads([ln for ln in log.splitlines() if ln.startswith("AUTO_SPINE ")][-1][11:])
        if res.get("spine"):
            import numpy as np
            sp = np.load(res["spine"])
            st = np.load(k1 / "start_spine.npy")
            row = int(out.split("--start-row ")[1].split()[0])
            # the line's rows before the start row are the spine's own
            assert (sp["origin"][:row] == st["origin"][:row]).all()
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_from_spine"] == 0.05
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("asfs_bad", ["--auto-spine-from-spine", "1"], tmp_path)
    assert bad.returncode != 0 and "--auto-spine-from-spine starts --auto-spine's beams" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "asfs_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine-from-spine" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    """The blocking loop (a search that fires and ends) and its numbers, against the trainer
    before the change."""
    pre = ROOT / "python" / "_train_fast_pre_asfs.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("asfs_off_new", TRAIN), ("asfs_off_old", pre)):
            r = _run_demo(run, LOOP, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "asfs_off_new", ROOT / "runs" / "asfs_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine_from_spine" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        assert not (a / "auto" / "k01" / "start_spine.npy").exists()
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
