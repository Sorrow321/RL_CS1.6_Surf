"""--demo-pull-d M2 --demo-pull-bonus B2 (pbr2b4, 2026-10-24): a second spine line beside the
--demo-success-d anchor, paying B2 once per episode. CPU, princessburglar: the trainer prints both
lines and dumps the keys; the pair is refused without the anchor bonus or when half given; off
(None) leaves the bonus path as before (one `is not None` test)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import needs_pb  # noqa: E402


@needs_pb
def test_the_pull_line_is_on_and_dumped(tmp_path):
    run = "dpull_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "100", "--demo-success-bonus", "50",
                            "--demo-pull-d", "0", "--demo-pull-bonus", "25"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-pull-d 0: a second line at d <=" in out and "pays 25 once more" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_pull_d"] == 0.0 and cfg["demo_pull_bonus"] == 25.0
        assert cfg["demo_success_d"] == 100.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_pb
def test_the_pull_line_needs_its_pair_and_the_anchor(tmp_path):
    for run, extra, msg in (
            ("dpull_half", ["--demo-success-d", "100", "--demo-success-bonus", "50",
                            "--demo-pull-d", "0"], "--demo-pull-d and --demo-pull-bonus go together"),
            ("dpull_noanchor", ["--demo-success-d", "100", "--demo-pull-d", "0",
                                "--demo-pull-bonus", "25"], "a second line beside")):
        try:
            r = _run_demo(run, extra, tmp_path)
            assert r.returncode != 0 and msg in (r.stdout + r.stderr), (r.stdout + r.stderr)[-2000:]
        finally:
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
