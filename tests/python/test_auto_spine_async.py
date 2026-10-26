"""--auto-spine-async (pbr7b4, 2026-10-26): the --auto-spine search runs in the BACKGROUND on its
own copy of the weights while training goes on, and its spine is swapped in at the first
iteration after it ends (round 6's two searches paused c12's training for ~45 min). CPU,
princessburglar: a smoke run in which the search is launched in the background and its result
is picked up (the run's end waits for it), the refusal without --auto-spine, TRAIN_ONLY in
record_ckpt, and the synchronous loop (flag off) bit-identical to the trainer before the change -
the result handling moved into a function both paths share."""
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
LOOP = ["--demo-success-d", "50", "--demo-success-bonus", "50",
        "--auto-spine", "--auto-spine-quorum", "1",
        "--auto-spine-patience", "1", "--auto-spine-envs", "16",
        "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
        "--auto-spine-retry", "1", "--record-every", "2048",
        "--eval-eps", "2", "--ep-ticks", "300"]


@needs_pb
def test_the_search_runs_in_the_background(tmp_path):
    run = "aspa_smoke"
    try:
        r = _run_demo(run, LOOP + ["--auto-spine-async"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine-async: the search runs in the background" in out
        assert "auto-spine search 1: running in the background (pid" in out, out[-6000:]
        assert "auto-spine search 1: the background search (launched at step" in out, out[-6000:]
        assert ("NEW SPINE" in out) or ("NO new spine" in out)
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine"] == 1 and cfg["auto_spine_async"] == 1
        k1 = ROOT / "runs" / run / "auto" / "k01"
        # the search read its own copy of the weights, not ckpt_latest
        assert (k1 / "search_ckpt.pt").exists()
        assert f"--ckpt {k1 / 'search_ckpt.pt'}" in out
        assert "AUTO_SPINE " in (k1 / "auto_spine.log").read_text()
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("aspa_bad", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                 "--auto-spine-async"], tmp_path)
    assert bad.returncode != 0 and "--auto-spine-async runs --auto-spine's search" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "aspa_bad", ignore_errors=True)


def test_train_only_in_record_ckpt():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"auto_spine_async"' in src


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
def test_the_synchronous_loop_is_bit_identical(tmp_path):
    """Flag off, the loop ON: the search pauses training and swaps its spine exactly as the
    trainer before the change did (same config, same progress.csv numbers but time/*)."""
    pre = ROOT / "python" / "_train_fast_pre_aspa.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        outs = {}
        for run, script in (("aspa_off_new", TRAIN), ("aspa_off_old", pre)):
            r = _run_demo(run, LOOP, tmp_path, script=script)
            outs[run] = r.stdout + r.stderr
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
            assert "auto-spine search 1:" in outs[run]
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
        # the search's outcome line is the same (its seconds aside)
        pick = [ln.split(" in ")[0].replace(run, "RUN") for run, o in outs.items()
                for ln in o.splitlines() if "auto-spine search 1: N" in ln]
        assert len(pick) == 2 and pick[0] == pick[1], pick
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
