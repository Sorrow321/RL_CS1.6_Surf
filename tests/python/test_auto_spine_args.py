"""--auto-spine-args STR (pbr7b1, 2026-10-26): extra arguments appended to every tools/auto_spine.py
search command the trainer's --auto-spine loop runs (its explorer's --ge-select-energy, ladder,
roots ... without a trainer flag each). CPU, princessburglar: (a) a smoke run with the flag prints
its notice, puts the arguments LAST on the search line and dumps the config key; (b) refused without
--auto-spine and when empty; (c) TRAIN_ONLY in record_ckpt; (d) flag off is bit-identical to the
trainer before the change (config, every progress.csv number but time/*)."""
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
sys.path.insert(0, str(ROOT / "tools"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
LOOP = ["--demo-success-d", "50", "--demo-success-bonus", "50",
        "--auto-spine", "--auto-spine-quorum", "1",
        "--auto-spine-patience", "1", "--auto-spine-envs", "16",
        "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
        "--auto-spine-retry", "1", "--record-every", "2048",
        "--eval-eps", "2", "--ep-ticks", "300"]


@needs_pb
def test_the_search_line_carries_the_args(tmp_path):
    run = "asargs_smoke"
    try:
        r = _run_demo(run, LOOP + ["--auto-spine-args", "--ge-select-energy 3 --roots-n 24"],
                      tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert ("--auto-spine-args: every tools/auto_spine.py search also gets "
                "--ge-select-energy 3 --roots-n 24") in out, out[-6000:]
        lines = [ln for ln in out.splitlines() if "auto-spine search 1: " in ln
                 and "auto_spine.py" in ln]
        assert lines, out[-6000:]
        assert lines[0].rstrip().endswith("--ge-select-energy 3 --roots-n 24"), lines[0]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_args"] == "--ge-select-energy 3 --roots-n 24"
        log = (ROOT / "runs" / run / "auto" / "k01" / "auto_spine.log").read_text()
        assert "AUTO_SPINE " in log                     # auto_spine.py parsed the arguments
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_pb
def test_refusals(tmp_path):
    for name, extra, msg in (
            ("asargs_bad1", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                             "--auto-spine-args", "--ge-select-energy 3"],
             "--auto-spine-args goes on --auto-spine's search command"),
            ("asargs_bad2", LOOP + ["--auto-spine-args", " "], "--auto-spine-args: empty")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), \
            (bad.stdout + bad.stderr)[-3000:]
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "auto_spine_args" in record_ckpt.TRAIN_ONLY


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine-args" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_asargs.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("asargs_off_new", TRAIN), ("asargs_off_old", pre)):
            r = _run_demo(run, LOOP, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "asargs_off_new", ROOT / "runs" / "asargs_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine_args" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
