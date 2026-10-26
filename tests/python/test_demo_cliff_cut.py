"""train_fast --demo-cliff-cut K (pbr6b3, 2026-10-26): a curriculum SUCCESS CLIFF - a band that keeps
backing off with success under a third of the advance rate at one place while the band after it
passes - is detected by DemoCurriculum (which only watches its own moves: tau's trajectory is the
same with the detector on or off) and the trainer cuts the spine before it. (a) the detector; (b) the
cut; (c) the trainer's notice, config key and refusal."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE  # noqa: E402
from surfgym.respawn import DemoCurriculum  # noqa: E402


def _spine(n=600):
    s = np.zeros(n, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(n) * 10.0                 # distinct origins: match() works
    return s


def _drive(k, cliff=300, n=600, iters=400, passes_after=None):
    """the band's spawns succeed iff their row >= ``cliff`` (from iteration ``passes_after`` on,
    every row succeeds) -> (tau trajectory, cliff() per iteration)."""
    dc = DemoCurriculum(_spine(n), window=5, rate=0.3, min_ep=20, grow=5, front_frac=0.8,
                        cooldown=1, grow_max=40)
    dc.cliff_k = k
    taus, cl = [], []
    for it in range(iters):
        lo, hi = dc._band()
        idx = np.repeat(np.arange(lo, hi + 1), 6)
        ok = (idx >= cliff) | (passes_after is not None and it >= passes_after)
        dc.note_outcomes(idx, ok.astype(float))
        dc._move()
        taus.append(dc.tau)
        cl.append(dc.cliff())
    return taus, cl


def test_the_detector_finds_the_cliff_and_never_moves_tau():
    t0, c0 = _drive(0)
    t3, c3 = _drive(3)
    assert t0 == t3                                          # the detector changes nothing
    assert all(c is None for c in c0)
    found = [c for c in c3 if c is not None]
    assert found, "no cliff found"
    assert 300 - 4 * 5 <= found[0] < 300                    # the first row of the failing band
    first = next(i for i, c in enumerate(c3) if c is not None)
    assert min(t3[:first + 1]) >= 300 - 5 * 8               # the band was oscillating at it


def test_the_detector_resets_when_the_band_gets_past():
    t, c = _drive(3, passes_after=200)
    assert any(x is not None for x in c[:200])
    assert all(x is None for x in c[-50:]) and min(t[-50:]) < 300 - 20


def test_reset_spine_forgets_the_cliff():
    dc = DemoCurriculum(_spine(), window=5, rate=0.3, min_ep=20, grow=5, cooldown=1)
    dc.cliff_k = 1
    dc._cliff_lo, dc._cliff_n = 123, 5
    assert dc.cliff() == 123
    dc.reset_spine(_spine(200))
    assert dc.cliff() is None


def test_cliff_cut_spine():
    from train_fast import cliff_cut_spine
    s = _spine(50)
    k = np.arange(150, dtype=np.int32).reshape(50, 3)
    cs, ck = cliff_cut_spine(s, k, 20)
    assert len(cs) == 20 and float(cs[-1]["origin"][0]) == 190.0 and ck.shape == (20, 3)
    cs[0]["origin"][0] = -1.0                                # a copy, not a view
    assert float(s[0]["origin"][0]) == 0.0
    cs2, ck2 = cliff_cut_spine(s, None, 10)
    assert len(cs2) == 10 and ck2 is None
    import pytest
    with pytest.raises(ValueError):
        cliff_cut_spine(s, None, 0)


def test_the_trainer_announces_dumps_and_refuses(tmp_path):
    import shutil
    from test_demo_success_d import _run_demo
    from test_ge_minutes import PB
    if not PB.exists():
        import pytest
        pytest.skip("no princessburglar")
    run = "dcc_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--demo-cliff-cut", "3"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--demo-cliff-cut 3: a band that backs off 3 times" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_cliff_cut"] == 3
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dcc_bad", ["--demo-cliff-cut", "3"], tmp_path)
    assert bad.returncode != 0 and "--demo-cliff-cut moves the --demo-success-d line" in \
        (bad.stdout + bad.stderr), (bad.stdout + bad.stderr)[-3000:]
    shutil.rmtree(ROOT / "runs" / "dcc_bad", ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "demo_cliff_cut" in record_ckpt.TRAIN_ONLY
