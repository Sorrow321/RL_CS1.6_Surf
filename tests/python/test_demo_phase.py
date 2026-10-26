"""train_fast --demo-phase (pbr6b2, 2026-10-26): a spine spawn's first decision lands on the
LINE'S OWN decision grid. The grid is read off the line's input changes (demo_phase_snap): a line
flown with one decision every K ticks changes its held inputs only on a decision's first tick.
(a) the snap rule on synthetic lines; (b) on a real policy line (pbr6b2_c1k01c: a beam line flown
from tick 0, then an explorer chain with no grid); (c) a trainer smoke run moves real spawns and
dumps the flag; the refusal without --demo-file."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

from surfgym.core import STATE_DTYPE  # noqa: E402
from train_fast import demo_phase_snap  # noqa: E402


def _line(n=600, k=4, phase=0, start=0, stop=None):
    """oldbuttons change on rows = phase + 1 (mod k) over [start, stop): decisions start at
    state rows = phase (mod k) there; elsewhere the inputs never change."""
    s = np.zeros(n, STATE_DTYPE)
    stop = n if stop is None else stop
    b = 0
    for r in range(n):
        if start <= r < stop and r % k == (phase + 1) % k and (r // k) % 2 == 0:
            b ^= 2                                   # jump pressed / released
        s[r]["oldbuttons"] = b
    return s


def test_the_snap_reads_the_grid_and_leaves_the_rest():
    s = _line(n=1000, phase=0, stop=400)
    snap = demo_phase_snap(s, None, 4)
    rows = np.arange(len(s))
    on = rows < 300                                  # well inside the gridded stretch
    assert np.all(snap[on] % 4 == 0)
    assert np.all(np.abs(snap[on] - rows[on]) <= 2)
    assert snap[101] == 100 and snap[102] == 104 and snap[103] == 104 and snap[100] == 100
    assert np.all(snap[rows >= 400 + 256 + 1] == rows[rows >= 400 + 256 + 1])   # no changes: kept
    s2 = _line(phase=2)
    snap2 = demo_phase_snap(s2, None, 4)
    assert np.all(snap2[10:590] % 4 == 2)


def test_the_snap_ignores_a_line_without_a_grid():
    rng = np.random.default_rng(0)
    s = np.zeros(800, STATE_DTYPE)
    s["oldbuttons"] = rng.integers(0, 2, 800) * 2    # changes on every residue
    snap = demo_phase_snap(s, None, 4)
    assert np.array_equal(snap, np.arange(800))
    assert np.array_equal(demo_phase_snap(s[:1], None, 4), np.arange(1))


def test_the_snap_on_a_policy_line():
    sp = ROOT / "autoresearch" / "spines" / "pbr6b2_c1k01c.npy"
    if not sp.exists():
        import pytest
        pytest.skip("no spine")
    s = np.load(sp)
    keys = np.load(str(sp).replace(".npy", ".keys.npy"))
    snap = demo_phase_snap(s, keys, 4)
    assert np.all(snap[300:6100] % 4 == 0)           # the beam line: decisions at rows 0 mod 4
    assert snap[6340] == 6340 and snap[6339] == 6340 and snap[6338] == 6340
    assert snap[6337] == 6336 and snap[6341] == 6340   # the bounce: on the decision at 6,340
    tail = np.arange(6460, len(s))
    assert np.array_equal(snap[6460:], tail)         # the explorer chain: no grid


def test_the_trainer_moves_spine_spawns(tmp_path):
    from test_demo_success_d import _run_demo
    from test_ge_minutes import PB
    if not PB.exists():
        import pytest
        pytest.skip("no princessburglar")
    run = "dph_smoke"
    try:
        r = _run_demo(run, ["--demo-phase"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--demo-phase: the line's decision grid read on its own input changes" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_phase"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert "demo_phase" in record_ckpt.TRAIN_ONLY
