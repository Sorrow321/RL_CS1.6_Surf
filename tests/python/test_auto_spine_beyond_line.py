"""train_fast --auto-spine-beyond-line (pbr7b4, 2026-10-27): when the greedy median has not reached a
DEPTH spine's success line (a --auto-spine-patience search), the new spine must beat that line by
the gain, not only the greedy median - a search never swaps in a line shallower than the one being
taught (pbr7b4's manual search on c17: the policy's own beam reached 74.41%, 650 u past the greedy
median but short of the taught spine's end). CPU, princessburglar: a smoke run with a patience
search (the reference in its command and the notice), the config key, the refusal, TRAIN_ONLY."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import needs_pb  # noqa: E402


@needs_pb
def test_a_patience_search_measures_from_the_line(tmp_path):
    run = "asbl_smoke"
    try:
        # quorum 9 of 2 episodes: only the patience fires the search
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "9",
                            "--auto-spine-patience", "1", "--auto-spine-envs", "16",
                            "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
                            "--auto-spine-retry", "1", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300",
                            "--auto-spine-beyond-line"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine_beyond_line"] == 1
        # the quorum line of the eval that fired the search (its "-> SEARCH" suffix)
        line = re.search(r"success line d <= ([\d,]+); greedy frontier \(median deepest d\) "
                         r"([\d,]+)  -> SEARCH", out)
        assert line, out[-4000:]
        succ, med = (float(x.replace(",", "")) for x in line.groups())
        cmd = re.search(r"auto-spine search 1: \S+ -u \S+auto_spine.py .*--greedy-d ([\d.]+)", out)
        assert cmd, out[-4000:]
        ref = float(cmd.group(1))
        if succ < med:
            # the greedy has not reached the line: the search is measured from the line
            assert "--auto-spine-beyond-line - the new spine must beat the success line" in out
            assert abs(ref - succ) < 1.0, (ref, succ, med)
        else:
            assert abs(ref - med) < 1.0, (ref, succ, med)      # med printed rounded
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("asbl_bad", ["--demo-success-d", "50", "--auto-spine-beyond-line"], tmp_path)
    assert bad.returncode != 0 and "--auto-spine-beyond-line is a rule of --auto-spine's" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "asbl_bad", ignore_errors=True)


def test_the_reference_rule():
    import importlib
    tf = importlib.import_module("train_fast")
    # off: the greedy median
    assert tf.search_ref_d(15240.0, 14808.0, None, False) == 15240.0
    # on, a depth spine the greedy has not reached: its line
    assert tf.search_ref_d(15240.0, 14808.0, None, True) == 14808.0
    # on, the greedy already past the line (the quorum case): the median
    assert tf.search_ref_d(14700.0, 14808.0, None, True) == 14700.0
    # on, an energy spine (its line no deeper than the greedy): the median
    assert tf.search_ref_d(15240.0, 14808.0, 2411.0, True) == 15240.0
    assert tf.search_ref_d(15240.0, None, None, True) == 15240.0


def test_train_only_in_record_ckpt():
    assert '"auto_spine_beyond_line"' in (ROOT / "tools" / "record_ckpt.py").read_text(
        encoding="utf-8")
