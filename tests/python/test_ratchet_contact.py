"""--ratchet-contact (zlr2b1, 2026-11-01): the ratchet PAYS a record that moves only on supported
ticks - walkable ground (onground != -1) or a surfable face pushing back up (the --surf-bonus
contact test) - so an apex over open air, which the free-flight goal field reads as progress,
pays nothing until a landing reaches that far.

(a) off is the control exactly: same rewards, same records, same observation column, no state;
(b) on: free flight above the record pays nothing; the landing that reaches it pays everything
    flown since the last contact (the teleport clip scales with the flight); ground counts; a
    vertical wall (only horizontal velocity clipped), a ceiling (pushed down) and a fall pinned
    at the -sv_maxvelocity clamp do not;
(c) the (d - b)/d0 observation column and ratchet_boot keep the ALL-STATE record;
(d) ended rows: the next episode's spawn never moves the dying episode's record, the read-out
    gets the dying episode's two records, the paid record restarts at the new spawn;
(e) the recorder lists the flag as training-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.rewards import RaceReward                  # noqa: E402
from test_arc_death_charge import _FakeCore, _step      # noqa: E402

G = 800.0
TICK = 10.0              # ms: RaceReward's default tick_ms
GS = G * TICK / 1000.0   # free-fall vz loss per call (8 u/s)


class _ZField:
    """d = 10,000 - z: 'height still to climb', the free-flight field's shape on surf_za_loop."""

    def sample(self, pos):
        return 10000.0 - np.asarray(np.atleast_2d(pos), np.float64)[:, 2]


def _rr(contact, n=1, z0=0.0, vz0=0.0, ground=True, **kw):
    core = _FakeCore(n).at([0.0] * n)
    core.config.phys.sv_maxvelocity = 3500.0
    _put(core, z0, vz0, ground)
    rr = RaceReward(_ZField(), scale=0.01, time_pen=0.0, stall_ticks=10 ** 9,
                    ratchet=True, ratchet_d0=10000.0, max_step=100.0, **kw)
    rr.ratchet_contact = contact
    rr.on_reset(core)
    return rr, core


def _put(core, z, vz, ground):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:, 2] = np.float32(vz)
    core.states_view["onground"][:] = 0 if ground else -1
    return core


def _flight(z0, vz0, n):
    """Exact free fall (GoldSrc half-step gravity), call by call: [(z, vz)]."""
    out, z, vz, dt = [], z0, vz0, TICK / 1000.0
    for _ in range(n):
        vh = vz - 0.5 * G * dt
        z += vh * dt
        vz = vh - 0.5 * G * dt
        out.append((z, vz))
    return out


def _run(rr, core, seq, ends=()):
    """seq: [(z, vz, ground)] -> per-call rewards"""
    return [float(_step(rr, _put(core, z, vz, g), done=1 if t in ends else 0)[0])
            for t, (z, vz, g) in enumerate(seq)]


# ==========================================================================
# (a) off is the control
# ==========================================================================
BASE = "401f40072ceac0df34cadd7363a29b4e64d7c970"   # the branch base, before the flag


def _control_class():
    """RaceReward as it was before the flag (git show BASE), or None without git."""
    import importlib.util
    import subprocess
    try:
        src = subprocess.run(["git", "show", f"{BASE}:python/surfgym/rewards.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    src = src.replace("from .core import", "from surfgym.core import") \
             .replace("from .dipmeter import", "from surfgym.dipmeter import")
    spec = importlib.util.spec_from_loader("_rewards_base", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "rewards_base.py", "exec"), mod.__dict__)
    return mod.RaceReward


def test_off_is_the_control_exactly():
    base = _control_class()
    if base is None:
        pytest.skip("no git history to read the pre-flag RaceReward from")
    rng = np.random.default_rng(3)
    seq = [(float(z), float(vz), bool(g)) for z, vz, g in
           zip(rng.uniform(-50, 400, 600).cumsum() % 3000, rng.normal(0, 400, 600),
               rng.random(600) < 0.3)]
    a, ca = _rr(False)
    cb = _FakeCore(1).at([0.0])
    cb.config.phys.sv_maxvelocity = 3500.0
    _put(cb, 0.0, 0.0, True)
    b = base(_ZField(), scale=0.01, time_pen=0.0, stall_ticks=10 ** 9,
             ratchet=True, ratchet_d0=10000.0, max_step=100.0)
    b.on_reset(cb)
    for t, (z, vz, g) in enumerate(seq):
        ra = _step(a, _put(ca, z, vz, g), done=1 if t in (200, 401) else 0)
        rb = _step(b, _put(cb, z, vz, g), done=1 if t in (200, 401) else 0)
        assert np.array_equal(ra, rb)
        assert np.array_equal(a.ratchet_gap(), b.ratchet_gap())
        assert np.array_equal(a._rec, b._rec) and np.array_equal(a._vz, b._vz)
    assert a._rec_c is None and a._rc_air is None
    sa, sb = a.pop_stats(), b.pop_stats()
    assert "rc_mean" not in sa and sa.keys() == sb.keys()
    assert all(sa[k] == sb[k] or (sa[k] != sa[k] and sb[k] != sb[k]) for k in sa)


# ==========================================================================
# (b) the paid record
# ==========================================================================
def test_an_apex_over_open_air_pays_nothing_until_a_landing_reaches_it():
    # a take-off from the ground at z 0 with vz 800: apex ~400 u; then it lands on a ledge at
    # z 300 (ground). The control pays every tick of the climb (400 u) and nothing at the
    # ledge; the contact ratchet pays nothing in the air and 300 u at the landing.
    fl = _flight(0.0, 800.0, 140)                 # up and back down past z 300
    k_land = max(i for i, (z, vz) in enumerate(fl) if z >= 300.0 and vz < 0)
    seq = [(z, vz, False) for z, vz in fl[:k_land + 1]] + [(300.0, 0.0, True)] * 5
    on, con = _rr(True)
    off, coff = _rr(False)
    r_on = _run(on, con, seq)
    r_off = _run(off, coff, seq)
    zmax = max(z for z, _ in fl[:k_land + 1])
    assert sum(r_off) == pytest.approx(0.01 * zmax, rel=1e-4)
    # the take-off call reads as support (the jump IS a push off the ground): its few units
    assert r_on[0] == pytest.approx(0.01 * fl[0][0], rel=1e-5)
    assert all(r == 0.0 for r in r_on[1:k_land + 1])
    # the landing pays everything from the take-off call to the ledge in ONE call - 292 u,
    # past the control's 100 u clip, which scales with the flight
    assert r_on[k_land + 1] == pytest.approx(0.01 * (300.0 - fl[0][0]), rel=1e-5)
    assert sum(r_on) == pytest.approx(0.01 * 300.0, rel=1e-5)


def test_a_surfable_face_counts_a_wall_a_ceiling_and_the_clamp_do_not():
    # airborne throughout: a spawn at z 1000 with vz_prev, then one call to z 1010
    def one(z_next, vz_prev, vz_next):
        rr, core = _rr(True, z0=1000.0, vz0=vz_prev, ground=False)
        return float(_step(rr, _put(core, z_next, vz_next, False))[0])
    # a ramp pushes back up: vz loses less than the free-fall step
    assert one(1010.0, -100.0, -100.0 - GS + 50.0) == pytest.approx(0.01 * 10.0, rel=1e-5)
    # a vertical wall clips only horizontal velocity: vz follows free fall exactly
    assert one(1010.0, -100.0, -100.0 - GS) == 0.0
    # a ceiling pushes DOWN
    assert one(1010.0, 300.0, 0.0) == 0.0
    # a fall pinned at the -sv_maxvelocity clamp stops losing vz but is free fall
    assert one(1010.0, -3500.0, -3500.0) == 0.0


def test_ground_counts_and_a_lower_contact_pays_nothing():
    rr, core = _rr(True, z0=500.0)
    seq = [(400.0, 0.0, True), (450.0, 0.0, True), (520.0, 0.0, True), (510.0, 0.0, True)]
    r = _run(rr, core, seq)
    assert r[:2] == [0.0, 0.0]                              # below the start record
    assert r[2] == pytest.approx(0.01 * 20.0, rel=1e-5)
    assert r[3] == 0.0


def test_a_teleport_while_on_a_surface_keeps_the_control_clip():
    rr, core = _rr(True, z0=0.0)
    r = _run(rr, core, [(0.0, 0.0, True), (5000.0, 0.0, True)])   # relocated in one call
    assert r[1] == pytest.approx(0.01 * 100.0, rel=1e-5)          # max_step x every
    assert rr._rec_c[0] == pytest.approx(10000.0 - 5000.0)        # the record still moves


# ==========================================================================
# (c) the observation column keeps the all-state record
# ==========================================================================
def test_the_observation_column_is_the_control_s():
    fl = _flight(0.0, 800.0, 140)
    seq = [(z, vz, False) for z, vz in fl] + [(300.0, 0.0, True)] * 3
    on, con = _rr(True)
    off, coff = _rr(False)
    for z, vz, g in seq:
        _step(on, _put(con, z, vz, g))
        _step(off, _put(coff, z, vz, g))
        assert np.array_equal(on.ratchet_gap(), off.ratchet_gap())
        assert np.array_equal(on.ratchet_boot(), off.ratchet_boot())


# ==========================================================================
# (d) ended rows and the read-out
# ==========================================================================
def test_ended_rows_restart_the_paid_record_and_report_both():
    rr, core = _rr(True, n=1, z0=0.0)
    # episode 0: ground at 200 (contact record 200), then flight up to ~600 (all-state ~600)
    seq = [(200.0, 0.0, True)] + [(z + 200.0, vz, False) for z, vz in _flight(0.0, 800.0, 60)]
    _run(rr, core, seq)
    zmax = max(z for z, _, _ in seq)
    # the call that ends it: the state is ALREADY episode 1's spawn - ground at 5000 (deeper):
    # it must not move episode 0's record nor pay
    r = float(_step(rr, _put(core, 5000.0, 0.0, True), done=1)[0])
    assert r == 0.0
    st = rr.pop_stats()
    # the contact record: the ground at 200 and the take-off call's few units above it
    assert st["rc_mean"] == pytest.approx((200.0 + _flight(0.0, 800.0, 1)[0][0]) / 10000.0,
                                          abs=1e-6)
    assert st["rc_all_mean"] == pytest.approx(zmax / 10000.0, abs=1e-4)
    assert st["rc_paid"] == 1.0
    # episode 1 starts its paid record at its own spawn: a contact below it pays nothing,
    # one above it pays from the spawn
    r = _run(rr, core, [(4990.0, 0.0, True), (5030.0, 0.0, True)])
    assert r[0] == 0.0 and r[1] == pytest.approx(0.01 * 30.0, rel=1e-5)


# ==========================================================================
# (e) the recorder
# ==========================================================================
def test_the_recorder_lists_the_flag_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    head = src[src.index("TRAIN_ONLY = frozenset({"):]
    head = head[:head.index("})")]
    assert '"ratchet_contact"' in head


# ==========================================================================
# (f) the curriculum reads the PAID record
# ==========================================================================
def test_paid_record_is_the_contact_record_under_the_flag():
    fl = _flight(0.0, 800.0, 140)
    k_land = max(i for i, (z, vz) in enumerate(fl) if z >= 300.0 and vz < 0)
    seq = [(z, vz, False) for z, vz in fl[:k_land + 1]]
    on, con = _rr(True)
    off, coff = _rr(False)
    _run(on, con, seq)
    _run(off, coff, seq)
    # off: the paid record is the all-state one, exactly
    assert off.paid_record() is off._rec and off.paid_record(boot=True) is off._rec_boot
    # on: the apex (~400 u up) moved the all-state record but not the paid one
    assert on._rec[0] < 10000.0 - 390.0
    assert on.paid_record()[0] == pytest.approx(10000.0 - fl[0][0])
    # the landing call ends the episode: boot = the dying episode's paid record (before the
    # call, the landing itself is never credited on an ended row), then the reset
    _step(on, _put(con, 300.0, 0.0, True), done=1)
    assert on.paid_record(boot=True)[0] == pytest.approx(10000.0 - fl[0][0])
    assert on.paid_record()[0] == pytest.approx(10000.0 - 300.0)
