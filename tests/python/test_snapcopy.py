"""tools/snapcopy.py's pruning rule (pbr8b3, 2026-10-27)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import snapcopy  # noqa: E402


def test_keep_recent_and_the_newest_before_each_eval():
    now = 10_000.0
    snaps = {100: now - 5000, 200: now - 4000, 300: now - 3000, 400: now - 100, 500: now - 10}
    # evals at 250 (newest at or before: 200) and 1000 (500, also recent)
    keep = snapcopy.keep_set(snaps, [250, 1000], now, keep_secs=1800)
    assert keep == {200, 400, 500}


def test_no_eval_before_any_snapshot_keeps_only_recent():
    now = 10_000.0
    snaps = {100: now - 5000, 200: now - 10}
    assert snapcopy.keep_set(snaps, [50], now, keep_secs=1800) == {200}
