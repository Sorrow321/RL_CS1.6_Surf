"""--demo-match-band (pbr6b4, 2026-10-26): a spine that stood still repeats one origin over several
rows; the curriculum re-identifies spawns by origin and its key keeps only the LAST copy, so a
frontier band inside such a run never sees an outcome and tau stops moving (pbr6b4_c11 at the
loop's spine row 6,819). With the flag, such a spawn is credited to its copy inside the band.
(a) the matcher, off and on; (b) the band's outcome count moves only with the flag; (c) a CPU
trainer run dumps the key; (d) record_ckpt knows it."""
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


def _spine():
    from surfgym.core import STATE_DTYPE
    s = np.zeros(30, STATE_DTYPE)
    s["origin"][:, 0] = np.arange(30, dtype=np.float32) * 10.0
    s["origin"][8:16] = s["origin"][8]          # rows 8..15 stood still
    return s


def test_a_repeated_origin_is_credited_to_its_copy_in_the_band():
    from surfgym.respawn import DemoCurriculum
    sp = _spine()
    off = DemoCurriculum(sp, window=5, grow=5, tau0=9)
    on = DemoCurriculum(sp, window=5, grow=5, tau0=9, match_band=True)
    assert off._band() == (9, 13) and on._band() == (9, 13)
    o = sp["origin"][[8, 12, 20]].astype(np.float64)
    assert off.match(o).tolist() == [15, 15, 20]          # the last copy: outside the band
    assert on.match(o).tolist() == [9, 9, 20]             # its copy in the band; others unchanged
    on.tau = 20                                           # no copy in the band: as before
    assert on.match(o).tolist() == [15, 15, 20]
    # the band's outcome count moves only when the spawns are credited to it
    for c in (off, on):
        c.tau = 9
        idx = c.match(np.repeat(o[:1], 60, axis=0))
        c.note_outcomes(idx, np.ones(60, bool))
    assert off.ep[9:14].sum() == 0.0 and on.ep[9:14].sum() > 40.0   # 60 decayed by 0.99: 45.3
    # a swapped-in spine rebuilds the copies
    on.reset_spine(sp[:12])
    assert on._dups and max(max(v) for v in on._dups.values()) == 11


def test_off_keeps_the_old_matcher_exactly():
    from surfgym.respawn import DemoCurriculum
    rng = np.random.default_rng(3)
    sp = _spine()
    c = DemoCurriculum(sp, window=5, grow=5, tau0=9)
    old = {tuple(np.round(np.asarray(r["origin"], np.float64), 1)): i for i, r in enumerate(sp)}
    pts = np.concatenate([sp["origin"].astype(np.float64), rng.normal(0, 50, (20, 3))])
    assert c.match(pts).tolist() == [old.get(tuple(np.round(p, 1)), -1) for p in pts]
    assert c._dups == {}


def test_the_trainer_dumps_the_flag(tmp_path):
    from test_demo_success_d import _run_demo
    from test_ge_minutes import PB
    if not PB.exists():
        pytest.skip("no princessburglar")
    run = "dmb_smoke"
    try:
        r = _run_demo(run, ["--demo-match-band"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_match_band"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    import record_ckpt
    assert "demo_match_band" in record_ckpt.TRAIN_ONLY
