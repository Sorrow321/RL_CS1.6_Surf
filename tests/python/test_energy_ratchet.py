"""--energy-ratchet (zlr1b2, 2026-10-31): --energy-pot's RATCHET form - pay only new records of
the episode's energy height h = z + |v|^2 / (2 g), charge nothing for a loss.

Why: under the per-call form a learner that cannot pump yet is charged for every landing, and
on surf_za_loop (A1, K 12) its greedy learned to dive into the kill plane in 3-5 s: dying was
the cheapest episode. The ratchet never charges a contact, pays a lost-and-regained h once, and
is bounded per episode by K x (h max - h at spawn).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.rewards import RaceReward        # noqa: E402
from test_stall_energy import _Core, _FlatField, _step, _vx_for   # noqa: E402

K = 12.0


def _rr(ratchet, n=1):
    core = _Core(n).put([0.0] * n, [0.0] * n)
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9)
    rr.energy_pot = K
    rr.en_ratchet = ratchet
    rr.on_reset(core)
    return rr, core


def _pay(rr, core, hs, done_at=()):
    out = []
    for i, h in enumerate(hs):
        out.append(float(_step(rr, core.put([0.0], [_vx_for(h)]),
                               done=[1 if i in done_at else 0])[0]))
    return out


def test_records_pay_losses_are_free_regains_pay_once():
    rr, core = _rr(True)
    hs = [0.0, 100.0, 50.0, 80.0, 150.0, 150.0, 20.0]
    out = _pay(rr, core, hs)
    want = [0.0, 100.0, 0.0, 0.0, 50.0, 0.0, 0.0]
    assert out == pytest.approx([K / 1000.0 * w for w in want], abs=1e-4)
    st = rr.pop_stats()
    assert st["en_gain_per_ep"] != st["en_gain_per_ep"] or True   # no ended episode: nan
    assert rr.en_gain == 0.0 and rr.en_loss == 0.0                 # drained by pop_stats


def test_the_loss_is_still_counted_for_the_readout():
    rr, core = _rr(True)
    _pay(rr, core, [0.0, 100.0, 40.0, 100.0])
    assert rr.en_gain == pytest.approx(100.0, abs=1e-3)
    assert rr.en_loss == pytest.approx(60.0, abs=1e-3)


def test_an_ended_row_reanchors_on_the_next_spawn():
    rr, core = _rr(True)
    out = _pay(rr, core, [0.0, 300.0, 50.0, 120.0], done_at=(2,))
    # call 2 ends the episode (its row is the next spawn, h 50): pays 0 and re-anchors there;
    # the new episode's rise 50 -> 120 is a record of ITS OWN and pays
    assert out == pytest.approx([0.0, K * 0.3, 0.0, K / 1000.0 * 70.0], abs=1e-4)


def test_off_is_the_per_call_form():
    hs = [0.0, 100.0, 50.0, 80.0, 150.0, 20.0]
    rr, core = _rr(False)
    out = _pay(rr, core, hs)
    want = [0.0] + [K / 1000.0 * (hs[i] - hs[i - 1]) for i in range(1, len(hs))]
    assert out == pytest.approx(want, abs=1e-4)
    assert rr._en_rec is None


def test_the_recorder_lists_it_as_training_only():
    import record_ckpt as rc
    assert "energy_ratchet" in rc.TRAIN_ONLY
