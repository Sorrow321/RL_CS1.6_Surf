"""beam_tas --score dh (pbr2b4, 2026-10-24): rank by -(d - h_coef * h), h = z + |v|^2 / (2 g).
The default 'd' ranking is unchanged; 'dh' prefers, at equal distance, the lineage with more
energy height, and at equal energy the nearer one. CPU, no map: a stub field."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import beam_tas  # noqa: E402
from surfgym.core import STATE_DTYPE  # noqa: E402


class _Field:
    def sample(self, o):
        o = np.asarray(o, np.float64)
        return 10000.0 - o[:, 0]               # d falls with x


def _states(rows):
    s = np.zeros(len(rows), STATE_DTYPE)
    for i, (x, z, v) in enumerate(rows):
        s["origin"][i] = (x, 0.0, z)
        s["velocity"][i] = (v, 0.0, 0.0)
    return s


def test_d_mode_unchanged():
    sc = beam_tas.make_scorer(_Field(), None, 0.0, "d")
    st = _states([(100, 0, 0), (200, 0, 0)])
    score, d = sc(st)
    assert np.allclose(d, [9900, 9800]) and np.allclose(score, -d)


def test_dh_prefers_energy_at_equal_distance_and_distance_at_equal_energy():
    sc = beam_tas.make_scorer(_Field(), None, 0.0, "dh", h_coef=1.0)
    st = _states([(100, 0, 0), (100, 0, 400), (200, 0, 0)])   # h = 0, 100, 0
    score, d = sc(st)
    assert np.allclose(d, [9900, 9900, 9800])
    assert score[1] > score[0], "more energy height at the same d ranks higher"
    assert score[2] > score[0], "nearer at the same energy ranks higher"
    assert np.isclose(score[1] - score[0], 100.0)              # 400^2 / 1600 = 100 u of h
    sc2 = beam_tas.make_scorer(_Field(), None, 0.0, "dh", h_coef=0.5)
    s2, _ = sc2(st)
    assert np.isclose(s2[1] - s2[0], 50.0)
