"""--demo-report-bands A:B[,C:D...] (ab/loop, 2026-11-14): the curriculum's finish rate at chosen
spine rows, printed beside its per-decile report.

Why: the princessburglar instruments train a fresh network on spawns spread uniformly along a
crossing line (the xLOOP distribution, ledger Round 27 addendum 4), and the read-out is the
success at the line's launch and at its far-wall landing. On a 2,664-row line both sit in the
same decile (rows 1,332-1,598), so the decile report cannot separate them. A read-out only: the
spawns, the curriculum's moves and every reward are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def _spine(n):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n, dtype=np.float32) * 10.0
    return s


def test_the_band_report_counts_the_rows_it_names():
    from surfgym.respawn import DemoCurriculum
    dc = DemoCurriculum(_spine(100), window=100, rate=2.0, min_ep=1e9)
    # rows 10-19: 10 episodes each, 3 of 10 rows' episodes succeed; rows 50-59: all succeed
    for _ in range(10):
        dc.note_outcomes(np.arange(10, 20), np.r_[np.ones(3), np.zeros(7)])
        dc.note_outcomes(np.arange(50, 60), np.ones(10))
    rep = dc.band_report([(10, 19), (50, 59), (90, 99)])
    assert rep.startswith("demo bands (eps/finish, spine rows) ")
    parts = rep.split(") ", 1)[1].split()
    assert parts[0].startswith("[10,19]:") and parts[0].endswith("/30%")
    assert parts[1].startswith("[50,59]:") and parts[1].endswith("/100%")
    assert parts[2] == "[90,99]:0/0%"                           # never spawned
    # the decile report is unchanged by it
    assert dc.region_report().startswith("demo regions (eps/finish, decile of spine) ")


def test_the_uniform_window_spans_the_whole_line():
    # the xLOOP / spine-testbed setting: window = the line, rate 2.0, min-ep 1e9 - the window
    # never moves and every demo draw is uniform over [0, n-1]
    from surfgym.respawn import DemoCurriculum
    dc = DemoCurriculum(_spine(2664), window=2664, rate=2.0, min_ep=1e9, seed=3)
    for _ in range(50):
        dc.note_outcomes(np.arange(0, 2664, 7), np.ones(len(range(0, 2664, 7))))
    pool = dc.build_pool(_spine(1), pool_size=4096, fresh_frac=0.10)
    rows = np.round(pool["origin"][:, 0] / 10.0).astype(int)
    demo_rows = rows[int(round(4096 * 0.10)):]
    assert dc.tau == 2663 and dc._draw() == (0, 2663)
    assert demo_rows.min() < 50 and demo_rows.max() > 2600
    assert abs(float(np.mean(demo_rows)) - 1331.5) < 60.0      # uniform over the line


def test_train_only():
    import record_ckpt
    assert "demo_report_bands" in record_ckpt.TRAIN_ONLY


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--demo-report-bands"' in src
    assert 'raise SystemExit("--demo-report-bands reads the --demo-file curriculum: give it")' \
        in src
    assert 'meta["config"]["demo_report_bands"] = [list(_b) for _b in DEMO_BANDS]' in src
    i_reg = src.index('print(f"  {demo.region_report()}")')
    i_band = src.index('print(f"  {demo.band_report(DEMO_BANDS)}")')
    assert i_reg < i_band < i_reg + 200
