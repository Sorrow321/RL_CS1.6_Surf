"""--speed-pot K: potential-based speed shaping on the race reward, Phi = K * |v_xy| / 1000.

The user, 2026-10-04 (skate_laby): "maybe our reward is not that sensitive to the timer ... a
proxy metric, for example ... lost speed on turns". Each call pays K/1000 x (the change of
horizontal speed); the sum over an episode telescopes; an ended row (whose state is already the
NEXT episode's spawn) pays 0 and re-anchors; K = 0 leaves the reward untouched.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

from surfgym.rewards import RaceReward                       # noqa: E402
from test_arc_death_charge import _FakeCore, _FlatField, _step   # noqa: E402
from test_view_continuous import needs_run                  # noqa: E402

K = 10.0


def _rr(k):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9)
    rr.speed_pot = k
    rr.on_reset(core)
    return rr, core


def _at_speed(core, v):
    core.states_view["velocity"][:] = np.asarray([[v, 0.0, 0.0]], np.float32)
    return core


def test_each_call_pays_the_change_of_speed():
    rr, core = _rr(K)
    out = [float(_step(rr, _at_speed(core, v))[0]) for v in (1000.0, 1200.0, 900.0, 900.0)]
    # the first call anchors (0), then +200, -300, 0 u/s
    assert out == pytest.approx([0.0, K * 0.2, -K * 0.3, 0.0], abs=1e-5)


def test_the_sum_telescopes():
    rr, core = _rr(K)
    vs = [1500.0, 1700.0, 1100.0, 1300.0, 2400.0, 1800.0]
    paid = sum(float(_step(rr, _at_speed(core, v))[0]) for v in vs)
    assert paid == pytest.approx(K * (vs[-1] - vs[0]) / 1000.0, abs=1e-4)


def test_an_ended_row_pays_nothing_and_reanchors():
    rr, core = _rr(K)
    _step(rr, _at_speed(core, 1800.0))
    r_end = float(_step(rr, _at_speed(core, 0.0), done=1)[0])      # autoreset: the new spawn
    assert r_end == pytest.approx(0.0, abs=1e-6)
    r_next = float(_step(rr, _at_speed(core, 600.0))[0])           # pays from the spawn's 0
    assert r_next == pytest.approx(K * 0.6, abs=1e-5)


def test_zero_is_off():
    a, ca = _rr(0.0)
    b, cb = _rr(K)
    for v in (1000.0, 1300.0, 700.0):
        ra = float(_step(a, _at_speed(ca, v))[0])
        assert ra == pytest.approx(0.0, abs=1e-7)
    assert a._spd_prev is None


@needs_run
def test_the_trainer_carries_it_and_the_recorder_accepts_it():
    """--speed-pot reaches run.json (only when on) and record_ckpt records the checkpoint (the key
    is TRAIN_ONLY)."""
    import json
    import shutil
    from test_obs_potential import ABS, CANNONBALL, RECORD, _run, _train
    r = _train("cya_spot", ABS + ["--speed-pot", "10"], steps="6144")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = ROOT / "runs" / "cya_spot"
    assert json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]["speed_pot"] == 10.0
    rec = _run([sys.executable, "-u", str(RECORD), str(d / "ckpt_final.pt"), "--map",
                str(CANNONBALL), "--episodes", "1", "--out", str(d / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-3000:] + rec.stderr[-3000:]
    shutil.rmtree(d, ignore_errors=True)
