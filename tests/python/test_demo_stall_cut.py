"""--demo-stall-cut STEPS (pbr6b4, 2026-10-26): under --auto-spine with --auto-spine-energy G, a
backward curriculum whose tau reaches no new minimum for STEPS steps has its spine CUT one
--demo-window before that minimum; the cut is taught with the energy rule (the cut end's energy
height - G) when the greedy frontier already crosses its line, else with the depth rule, and the
old line stays a milestone. (a) the rule (stall_cut_plan); (b) a CPU trainer run on princessburglar
cuts once and dumps the key; without --auto-spine-energy it is refused; (c) record_ckpt knows it."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))


def _spine(n=100):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 10.0   # x 0 .. 990
    s["origin"][:, 2] = 100.0
    s["velocity"][:, 0] = 400.0                                  # h = 100 + 400^2/1600 = 200
    return s


def _sample(p):
    """d = 10,000 - x: the spine runs toward the goal."""
    return 10000.0 - np.asarray(p, np.float64)[:, 0]


def test_stall_cut_plan_picks_the_rule_against_the_greedy():
    from train_fast import stall_cut_plan
    sp = _spine()
    # cut at row 60: d 9,400, its line 9,500
    p = stall_cut_plan(sp, 60, _sample, 100.0, 9450.0, 150.0)      # the greedy is past it
    assert p["line"] == pytest.approx(9500.0) and p["h_min"] == pytest.approx(200.0 - 150.0)
    p = stall_cut_plan(sp, 60, _sample, 100.0, 9500.0, 150.0)      # exactly on it: energy
    assert p["h_min"] == pytest.approx(50.0)
    p = stall_cut_plan(sp, 60, _sample, 100.0, 9600.0, 150.0)      # short of it: depth rule
    assert p["line"] == pytest.approx(9500.0) and p["h_min"] is None
    assert stall_cut_plan(sp, 60, _sample, 100.0, None, 150.0) is None   # no greedy eval yet
    assert stall_cut_plan(sp, 0, _sample, 100.0, 9450.0, 150.0) is None  # nothing left
    assert stall_cut_plan(sp, 99, _sample, 100.0, 9450.0, 150.0) is None  # nothing cut


def test_the_trainer_cuts_a_stalled_curriculum(tmp_path):
    from test_demo_success_d import _run_demo
    from test_ge_minutes import PB
    if not PB.exists():
        pytest.skip("no princessburglar")
    run = "dsc_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "9",
                            "--auto-spine-energy", "150", "--auto-spine-ge-minutes", "0",
                            "--demo-stall-cut", "1", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300", "--steps", "8192"],
                      tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--demo-stall-cut 1: a curriculum whose tau reaches no new minimum" in out
        assert "--demo-stall-cut 1: tau had not fallen below 39" in out, out[-6000:]
        assert "the spine is cut at row 34" in out, out[-6000:]
        cut = json.loads((ROOT / "runs" / run / "auto" / "cut01.json").read_text())
        assert cut["cut_row"] == 34 and cut["n"] == 35
        assert cut["old_success_d"] in cut["milestones"]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_stall_cut"] == 1.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dsc_bad", ["--demo-success-d", "50", "--demo-success-bonus", "50",
                                "--auto-spine", "--demo-stall-cut", "1"], tmp_path)
    assert bad.returncode != 0 and "give --auto-spine and --auto-spine-energy" in (
        bad.stdout + bad.stderr), (bad.stdout + bad.stderr)[-3000:]
    shutil.rmtree(ROOT / "runs" / "dsc_bad", ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "demo_stall_cut" in record_ckpt.TRAIN_ONLY
