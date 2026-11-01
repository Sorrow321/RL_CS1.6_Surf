"""--energy-pot-loss L (zlr2b3, 2026-11-01): the share of an energy-height LOSS the per-call
--energy-pot term charges; gains are always paid whole.

(a) L = 1 (the default) is the potential exactly as before (each call pays K/1000 x dh, the sum
    telescopes); L = 0 pays only the gains (a contact that kills speed costs nothing); 0 < L < 1
    charges L x the loss; an ended row still pays 0 and re-anchors.
(b) The trainer refuses L outside [0, 1] and L != 1 without --energy-pot; the recorder lists the
    key as training-only (so the record gate accepts a checkpoint that carries it).
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

from surfgym.rewards import RaceReward                       # noqa: E402
from test_arc_death_charge import _FakeCore, _FlatField, _step   # noqa: E402

K = 12.0
G = 800.0


def _rr(k, loss=None):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9)
    rr.energy_pot = k
    if loss is not None:
        rr.en_loss_frac = loss
    rr.on_reset(core)
    return rr, core


def _put(core, z, v):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:] = np.asarray([v], np.float32)
    return core


def _h(z, v):
    return z + float(np.dot(v, v)) / (2.0 * G)


# gain, loss (a wall takes vx), gain, loss (a floor takes vz)
PTS = [(0.0, (1000.0, 0.0, 0.0)), (0.0, (1200.0, 0.0, 0.0)), (0.0, (200.0, 0.0, 600.0)),
       (100.0, (300.0, 0.0, 600.0)), (100.0, (300.0, 0.0, 0.0))]


def _pay(loss):
    rr, core = _rr(K, loss)
    return [float(_step(rr, _put(core, z, v))[0]) for z, v in PTS]


def _want(loss):
    hs = [_h(z, np.asarray(v)) for z, v in PTS]
    out = [0.0]
    for i in range(1, len(hs)):
        dh = hs[i] - hs[i - 1]
        out.append(K / 1000.0 * (dh if dh >= 0.0 else loss * dh))
    return out


def test_default_is_the_potential_bit_for_bit():
    a = _pay(None)
    b = _pay(1.0)
    assert a == b                                  # not merely close: the same floats
    assert a == pytest.approx(_want(1.0), abs=1e-5)
    hs = [_h(z, np.asarray(v)) for z, v in PTS]
    assert sum(a) == pytest.approx(K / 1000.0 * (hs[-1] - hs[0]), abs=1e-4)   # telescopes


@pytest.mark.parametrize("loss", [0.0, 0.25, 0.5])
def test_a_loss_is_charged_its_share_and_a_gain_whole(loss):
    got = _pay(loss)
    assert got == pytest.approx(_want(loss), abs=1e-5)
    want_gain = [w for w in _want(1.0) if w > 0.0]
    assert [g for g in got if g > 0.0] == pytest.approx(want_gain, abs=1e-5)


def test_gains_only_never_charges():
    got = _pay(0.0)
    assert min(got) >= 0.0
    assert sum(got) > 0.0


def test_an_ended_row_pays_nothing_and_reanchors():
    rr, core = _rr(K, 0.0)
    _step(rr, _put(core, 0.0, (1000.0, 0.0, 0.0)))
    r = float(_step(rr, _put(core, 0.0, (0.0, 0.0, 0.0)), done=1)[0])
    assert r == pytest.approx(0.0, abs=1e-6)       # the new episode's spawn: no charge, no pay
    r2 = float(_step(rr, _put(core, 0.0, (400.0, 0.0, 0.0)))[0])
    assert r2 == pytest.approx(K / 1000.0 * _h(0.0, np.asarray((400.0, 0.0, 0.0))), abs=1e-5)


def test_recorder_lists_the_key_training_only():
    import record_ckpt
    assert "energy_pot_loss" in record_ckpt.TRAIN_ONLY


def test_trainer_validates_the_flag():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--energy-pot-loss"' in src
    assert "--energy-pot-loss L needs 0 <= L <= 1" in src
    assert "--energy-pot-loss shapes --energy-pot" in src
    assert "_s.reward_fn.en_loss_frac = float(args.energy_pot_loss)" in src
