"""--energy-pot-ng (uf2r2b3, 2026-10-31): the --race-ng treatment of --energy-pot.

The stock term pays the per-call change Phi(s') - Phi(s) of Phi = K h / 1000 (h = z + |v|^2/2g).
Under gamma < 1 that does not telescope: its discounted residue is ~(1 - g) Phi(s') per call
for being ALIVE at energy h (g = the per-call discount). With the flag, every live call is also
taxed (1 - g) Phi(s') and a death or a finish is charged the potential of its last
pre-terminal state, so the DISCOUNTED shaping over any episode is exactly -Phi(spawn) (Ng et al.
1999; Grzes 2017 for the terminal), whatever the line flew.

(a) per call: the stock change minus the tax on the post-step potential;
(b) a death and a finish are charged -Phi(last pre-terminal state), a truncation is not;
(c) the invariance: two very different episodes from one spawn have the same discounted sum,
    -Phi(spawn), while the stock form pays the one that lingers at high energy more;
(d) off = the stock form (the existing test_energy_pot.py), the flag surface and refusals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.rewards import RaceReward                       # noqa: E402
from test_arc_death_charge import _FakeCore, _FlatField, _step   # noqa: E402

K = 12.0
G = 800.0
GC = 0.99                     # a per-call discount far from 1, so the tax is visible


def _rr(ng=True, g=GC):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9)
    rr.energy_pot = K
    if ng:
        rr.energy_ng = True
        rr._en_ng_g = g
    rr.on_reset(core)
    return rr, core


def _put(core, z, v):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:] = np.asarray([v], np.float32)
    return core


def _phi(z, v):
    v = np.asarray(v, np.float64)
    return K / 1000.0 * (z + float(v @ v) / (2.0 * G))


def test_each_live_call_pays_the_change_minus_the_tax():
    rr, core = _rr()
    pts = [(1000.0, (800.0, 0.0, 0.0)), (1000.0, (900.0, 0.0, 0.0)),
           (900.0, (900.0, 0.0, 0.0)), (900.0, (900.0, 0.0, -400.0))]
    out = [float(_step(rr, _put(core, z, v))[0]) for z, v in pts]
    ph = [_phi(z, v) for z, v in pts]
    want = [-(1 - GC) * ph[0]] + [ph[i] - ph[i - 1] - (1 - GC) * ph[i]
                                  for i in range(1, len(ph))]
    assert out == pytest.approx(want, abs=1e-4)


@pytest.mark.parametrize("kind", ["death", "finish", "trunc"])
def test_a_terminal_is_charged_its_last_potential_and_a_truncation_is_not(kind):
    rr, core = _rr()
    _step(rr, _put(core, 3000.0, (1800.0, 0.0, 0.0)))
    last = _phi(3000.0, (1800.0, 0.0, 0.0))
    spawn = _put(core, 100.0, (0.0, 0.0, 0.0))                     # the next episode's spawn
    if kind == "death":
        r = float(_step(rr, spawn, done=1)[0])
        assert r == pytest.approx(-last, abs=1e-4)
    elif kind == "finish":
        r = float(_step(rr, spawn, done=1, goal=1)[0])
        assert r == pytest.approx(rr.success_bonus - last, abs=1e-4)
    else:
        r = float(_step(rr, spawn, trunc=1)[0])
        assert r == pytest.approx(0.0, abs=1e-6)
    # the next call pays from the new spawn, as the stock form does
    r1 = float(_step(rr, _put(core, 100.0, (400.0, 0.0, 0.0)))[0])
    p0, p1 = _phi(100.0, (0.0, 0.0, 0.0)), _phi(100.0, (400.0, 0.0, 0.0))
    assert r1 == pytest.approx(p1 - p0 - (1 - GC) * p1, abs=1e-4)


def _episode(rr, core, line, end="death"):
    """the rewards of one episode: the call that re-anchors at line[0] (the previous episode's
    terminal) is not part of it; then one call per later state, and the terminal call."""
    _step(rr, _put(core, *line[0]), done=1)
    rs = [float(_step(rr, _put(core, z, v))[0]) for z, v in line[1:]]
    rs.append(float(_step(rr, _put(core, 0.0, (0.0, 0.0, 0.0)), done=1,
                          goal=int(end == "finish"))[0]))
    if end == "finish":
        rs[-1] -= rr.success_bonus
    return rs


def _disc(rs, g=GC):
    return float(sum(r * g ** i for i, r in enumerate(rs)))


def test_the_discounted_shaping_of_any_episode_is_minus_phi_of_its_spawn():
    spawn = (1200.0, (0.0, 0.0, 0.0))
    rng = np.random.default_rng(0)
    # a quick line that spends its energy, and a long one that pumps and keeps it
    quick = [spawn] + [(1200.0 - 40.0 * i, (300.0 + 10 * i, 0.0, 0.0)) for i in range(1, 12)]
    slow = [spawn] + [(1200.0, (float(rng.uniform(900, 1400)), 0.0, 0.0)) for _ in range(80)]
    for end in ("death", "finish"):
        rr, core = _rr()
        _step(rr, _put(core, *spawn))
        a = _disc(_episode(rr, core, quick, end))
        b = _disc(_episode(rr, core, slow, end))
        assert a == pytest.approx(-_phi(*spawn), abs=1e-3)
        assert b == pytest.approx(-_phi(*spawn), abs=1e-3)
    # the stock form pays the lingering, energy-keeping episode more (the leak)
    rr, core = _rr(ng=False)
    _step(rr, _put(core, *spawn))
    a = _disc(_episode(rr, core, quick))
    b = _disc(_episode(rr, core, slow))
    assert b > a + 1.0


def test_off_is_the_stock_form_and_the_stats_appear_only_when_on():
    a, ca = _rr(ng=False)
    b, cb = _rr(ng=False)
    b.energy_ng = False                                    # the default, set explicitly
    for z, v in ((1000.0, (500.0, 0.0, 0.0)), (800.0, (900.0, 0.0, 0.0)),
                 (800.0, (100.0, 0.0, 0.0))):
        assert float(_step(a, _put(ca, z, v))[0]) == float(_step(b, _put(cb, z, v))[0])
    assert not any(k.startswith("en_ng") for k in a.pop_stats())
    c, cc = _rr()
    for z, v in ((1000.0, (500.0, 0.0, 0.0)), (800.0, (900.0, 0.0, 0.0))):
        _step(c, _put(cc, z, v))
    _step(c, _put(cc, 0.0, (0.0, 0.0, 0.0)), done=1)
    st = c.pop_stats()
    assert "en_ng_tax_per_ep" in st and "en_ng_term_per_ep" in st


def test_the_flag_surface():
    tf = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--energy-pot-ng", action="store_const", const=1, default=None' in tf
    assert '**({"energy_pot_ng": 1}' in tf                   # dumped only when set
    assert "--energy-pot-ng conforms --energy-pot: it needs --energy-pot K > 0" in tf
    assert "--energy-pot-ng with --energy-pot-front" in tf
    # the per-call discount at startup and on every tick retune
    assert tf.count("_en_ng_g = float(GAMMA_T) ** float(") == 2
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"energy_pot_ng",' in rc                          # TRAIN_ONLY
