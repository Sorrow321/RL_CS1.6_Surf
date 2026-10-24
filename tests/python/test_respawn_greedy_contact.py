"""--respawn-greedy-contact (pbr3b4, 2026-10-24): a failed greedy episode that ends in a
ballistic flight is anchored at its LAST MAP CONTACT instead of its last tick. Synthetic
episodes (a ride then a free fall; a death on impact; a short final hop), and the flag off
reproduces the death anchor exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                     # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool, last_contact  # noqa: E402


def _ride_then_fall(n_ride=1500, n_fall=170, g=-8.0):
    ep = np.zeros(n_ride + n_fall, STATE_DTYPE)
    vz = np.zeros(n_ride + n_fall)
    vz[:n_ride] = 60.0 * np.sin(np.arange(n_ride) / 9.0)      # ramp contact: not ballistic
    vz[n_ride:] = vz[n_ride - 1] + g * np.arange(1, n_fall + 1)
    ep["velocity"][:, 2] = vz
    ep["origin"][:, 0] = np.arange(n_ride + n_fall)          # an index tag
    return ep


def test_last_contact_finds_the_launch():
    ep = _ride_then_fall()
    c = last_contact(ep, 25)
    assert 1490 <= c <= 1500, c
    # a death on impact (no ballistic tail) keeps the last index
    hit = _ride_then_fall(n_fall=0)
    assert last_contact(hit, 25) == len(hit) - 1


def _pool(contact):
    p = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0, contact=contact)
    return p


def _build(p, eps, ends):
    p._eps = list(eps)
    p._cur = []
    p._par = None
    return p.build([{"end": e} for e in ends], 10.0)


def test_contact_anchor_takes_the_launch_window():
    ep = _ride_then_fall()
    off, on = _pool(False), _pool(True)
    _build(off, [ep], ["fail"])
    _build(on, [ep], ["fail"])
    t_off = off.rows["origin"][:, 0]
    t_on = on.rows["origin"][:, 0]
    # off: [n - 200, n - 25) - the last two seconds, inside the fall
    n = len(ep)
    assert t_off.min() == n - 200 and t_off.max() == n - 26
    # on: the 175 ticks up to and including the launch (~1500)
    assert 1490 <= t_on.max() <= 1500 and t_on.max() - t_on.min() == 175
    assert on.info["launches_s"] and abs(on.info["launches_s"][0] - t_on.max() / 100) < 0.011
    assert "last contact" in on.line()


def test_short_final_hop_and_truncs_keep_the_death_anchor():
    # a final flight shorter than LO (0.25 s) is not a "fatal flight": the death anchor
    ep = _ride_then_fall(n_fall=15)
    off, on = _pool(False), _pool(True)
    _build(off, [ep], ["fail"])
    _build(on, [ep], ["fail"])
    assert np.array_equal(off.rows, on.rows)
    # a truncated episode is untouched by the flag
    off2, on2 = _pool(False), _pool(True)
    _build(off2, [_ride_then_fall()], ["trunc"])
    _build(on2, [_ride_then_fall()], ["trunc"])
    assert np.array_equal(off2.rows, on2.rows)


def test_flag_off_build_is_unchanged():
    """The constructor default is off and the rows equal the explicit off pool's."""
    ep = _ride_then_fall()
    a = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0)
    b = _pool(False)
    _build(a, [ep], ["fail"])
    _build(b, [ep], ["fail"])
    assert np.array_equal(a.rows, b.rows) and "launches_s" not in a.info
    assert "last contact" not in a.line()
