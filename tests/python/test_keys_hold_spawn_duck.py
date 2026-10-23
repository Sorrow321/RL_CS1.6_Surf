"""--keys-hold-spawn-duck (pbr1b1, 2026-10-23): under --keys-hold an episode that spawns crouched
starts with duck HELD instead of neutral. CPU, princessburglar: a run with the flag trains and
dumps the key; flag off is bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_ge_minutes import _csv, _train, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


@needs_pb
def test_the_flag_trains_and_is_dumped():
    run = "khsd_on"
    try:
        r = _train(run, ["--keys-hold-spawn-duck"])
        assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["keys_hold_spawn_duck"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--keys-hold-spawn-duck" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical():
    pre = ROOT / "python" / "_train_fast_pre_khsd.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("khsd_off_new", TRAIN), ("khsd_off_old", pre)):
            r = _train(run, ["--record-every", "2048"], script=script, steps="4096")
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "khsd_off_new", ROOT / "runs" / "khsd_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "keys_hold_spawn_duck" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
        assert ta and [p.name for p in ta] == [p.name for p in tb]
        for p, q in zip(ta, tb):
            assert p.read_bytes() == q.read_bytes(), p.name
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
