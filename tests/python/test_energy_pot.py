"""--energy-pot C (2026-10-23, autoresearch r7b1): potential-based shaping on the mechanical ENERGY
HEIGHT h_E = z + |v|^2 / (2 g), Phi = C x scale x h_E (scale = the race reward's own 100/d0 per u).

Why: on sidistic the greedy line reaches the last gate with too little energy to rise over the
finish funnel's rim; it bleeds it in hard landings and wall hits between the chimney and the
basement, and nothing in the race reward charges a loss when it happens. --speed-pot charges every
climb (speed traded for height); this does not.

(a) The reward: each call pays C x scale x (the change of h_E); a climb or a fall that trades speed
    for height at constant h_E pays 0; the sum telescopes; an ended row (whose state is already the
    NEXT episode's spawn) pays 0 and re-anchors; a gain above max_step per tick (a relocation) is
    clipped; C = 0 leaves the reward untouched.
(b) The trainer (CPU, sidistic): the key reaches run.json only when on, a bare resume restores it,
    a negative C is refused, record_ckpt records the checkpoint; flag off is bit-identical to the
    trainer before the change.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.rewards import RaceReward                       # noqa: E402
from test_arc_death_charge import _FakeCore, _FlatField, _step   # noqa: E402
from test_ez_viewc import SID, SMOKE, TRAIN, _run, needs_run    # noqa: E402

C = 2.0
SCALE = 0.001
G = 800.0
RECORD = ROOT / "tools" / "record_ckpt.py"


def _rr(c, max_step=100.0):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=SCALE, time_pen=0.0, stall_ticks=10 ** 9,
                    max_step=max_step)
    rr.energy_pot = c
    rr.on_reset(core)
    return rr, core


def _put(core, z, v):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:] = np.asarray([v], np.float32)
    return core


def _h(z, v):
    return z + float(np.dot(v, v)) / (2.0 * G)


def test_each_call_pays_the_change_of_energy_height():
    rr, core = _rr(C)
    pts = [(1000.0, (1200.0, 0.0, 0.0)), (1000.0, (1000.0, 300.0, 0.0)),
           (1010.0, (1000.0, 300.0, 0.0)), (1010.0, (400.0, 0.0, -100.0))]
    out = [float(_step(rr, _put(core, z, v))[0]) for z, v in pts]
    want = [0.0] + [C * SCALE * (_h(*pts[i]) - _h(*pts[i - 1])) for i in range(1, len(pts))]
    assert out == pytest.approx(want, abs=1e-5)
    assert out[-1] < 0.0                      # the hard stop is charged at once


def test_a_climb_at_constant_energy_pays_nothing():
    """Speed traded for height (surfing up a ramp, a ballistic rise) keeps h_E: no charge, where
    --speed-pot would charge the lost horizontal speed."""
    rr, core = _rr(C)
    z0, v0 = 900.0, 1600.0
    h0 = _h(z0, (v0, 0.0, 0.0))
    out = []
    for z in (900.0, 1000.0, 1200.0, 1500.0, 1100.0):
        v = float(np.sqrt(2.0 * G * (h0 - z)))
        out.append(float(_step(rr, _put(core, z, (v * 0.6, 0.0, v * 0.8)))[0]))
    assert out == pytest.approx([0.0] * len(out), abs=1e-4)


def test_the_sum_telescopes():
    rr, core = _rr(C, max_step=1e9)       # these steps are far above one tick's physics
    pts = [(1500.0, (900.0, 100.0, -50.0)), (1450.0, (1100.0, 0.0, -300.0)),
           (1300.0, (700.0, 400.0, 200.0)), (1320.0, (1500.0, -200.0, 0.0)),
           (1100.0, (1300.0, 0.0, -600.0))]
    paid = sum(float(_step(rr, _put(core, z, v))[0]) for z, v in pts)
    assert paid == pytest.approx(C * SCALE * (_h(*pts[-1]) - _h(*pts[0])), abs=1e-4)


def test_an_ended_row_pays_nothing_and_reanchors():
    rr, core = _rr(C)
    _step(rr, _put(core, 2000.0, (1800.0, 0.0, 0.0)))
    r_end = float(_step(rr, _put(core, 15000.0, (0.0, 0.0, 0.0)), done=1)[0])   # the new spawn
    assert r_end == pytest.approx(0.0, abs=1e-6)
    r_next = float(_step(rr, _put(core, 14990.0, (0.0, 0.0, -100.0)))[0])      # from the spawn
    assert r_next == pytest.approx(C * SCALE * (_h(14990.0, (0, 0, -100.0)) - 15000.0), abs=1e-5)


def test_a_relocation_gain_is_clipped_and_a_loss_is_not():
    rr, core = _rr(C, max_step=100.0)
    _step(rr, _put(core, 1000.0, (1000.0, 0.0, 0.0)))
    up = float(_step(rr, _put(core, 6000.0, (1000.0, 0.0, 0.0)))[0])     # +5,000 u in one call
    assert up == pytest.approx(C * SCALE * 100.0, abs=1e-5)
    down = float(_step(rr, _put(core, 6000.0, (0.0, 0.0, 0.0)))[0])      # a dead stop
    assert down == pytest.approx(-C * SCALE * 1000.0 ** 2 / (2 * G), abs=1e-4)


def test_zero_is_off():
    a, ca = _rr(0.0)
    for z, v in ((1000.0, (1000.0, 0.0, 0.0)), (1300.0, (100.0, 0.0, 0.0)),
                 (700.0, (0.0, 0.0, 0.0))):
        assert float(_step(a, _put(ca, z, v))[0]) == pytest.approx(0.0, abs=1e-7)
    assert a._eh_prev is None


class _XField:
    """d = 10,000 - x: moving +x is progress."""
    def sample(self, pos):
        return 10000.0 - np.atleast_2d(pos)[:, 0].astype(np.float64)


def _rr_front(thr):
    """The energy term alone: a pair of rewards on two identical cores, C and 0 (the field's
    progress shaping cancels in the difference)."""
    pair = []
    for c in (C, 0.0):
        core = _FakeCore(1).at([0.0])
        rr = RaceReward(_XField(), scale=SCALE, time_pen=0.0, stall_ticks=10 ** 9, max_step=1e9)
        rr.energy_pot = c
        rr.energy_front_d = thr
        rr.on_reset(core)
        pair.append((rr, core))
    return pair


def _go(pair, x, z, v, done=0):
    out = []
    for rr, core in pair:
        core.states_view["origin"][:, 0] = np.float32(x)
        out.append(float(_step(rr, _put(core, z, v), done=done)[0]))
    return out[0] - out[1]


def test_front_gate_stops_the_term_once_the_record_is_in_the_band():
    """--energy-pot-front: paid while the episode's record d > thr, nothing after - even when the
    agent falls back out of the band (the record only improves)."""
    pair = _rr_front(thr=4000.0)                          # the band: d <= 4,000, i.e. x >= 6,000
    _go(pair, 1000.0, 2000.0, (1000.0, 0.0, 0.0))           # anchors
    a = _go(pair, 3000.0, 2000.0, (800.0, 0.0, 0.0))        # outside the band: charged
    assert a == pytest.approx(C * SCALE * (_h(2000, (800, 0, 0)) - _h(2000, (1000, 0, 0))), abs=1e-5)
    b = _go(pair, 6500.0, 2000.0, (600.0, 0.0, 0.0))        # the call that ENTERS the band: still paid
    assert b == pytest.approx(C * SCALE * (_h(2000, (600, 0, 0)) - _h(2000, (800, 0, 0))), abs=1e-5)
    c = _go(pair, 6600.0, 2000.0, (100.0, 0.0, 0.0))        # inside: a big loss, not charged
    assert c == pytest.approx(0.0, abs=1e-7)
    e = _go(pair, 2000.0, 2000.0, (900.0, 0.0, 0.0))        # back out of the band: the record stays
    assert e == pytest.approx(0.0, abs=1e-7)


def test_front_gate_spawn_in_the_band_gets_nothing_and_a_new_episode_restarts_the_record():
    pair = _rr_front(thr=4000.0)
    _go(pair, 7000.0, 2000.0, (1000.0, 0.0, 0.0))           # spawned in the band
    assert _go(pair, 7100.0, 2000.0, (200.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-7)
    # the episode ends; the new spawn is far behind the band, so its energy is paid again
    assert _go(pair, 500.0, 3000.0, (0.0, 0.0, 0.0), done=1) == pytest.approx(0.0, abs=1e-7)
    f = _go(pair, 600.0, 2990.0, (0.0, 0.0, -150.0))
    assert f == pytest.approx(C * SCALE * (_h(2990, (0, 0, -150)) - 3000.0), abs=1e-5)


def test_front_gate_off_is_the_plain_term():
    pair = _rr_front(thr=float("inf"))
    b, cb = _rr(C, max_step=1e9)
    for x, z, v in ((1000.0, 2000.0, (1000.0, 0.0, 0.0)), (7000.0, 1900.0, (300.0, 0.0, 0.0)),
                    (9000.0, 1500.0, (1500.0, 0.0, -400.0))):
        assert _go(pair, x, z, v) == pytest.approx(float(_step(b, _put(cb, z, v))[0]), abs=1e-5)
    assert pair[0][0]._eh_dmin is None


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_carries_restores_and_refuses_it_and_the_recorder_accepts_it():
    run, run2 = "epot_on", "epot_on_re"
    for rn in (run, run2, "epot_bad"):
        shutil.rmtree(ROOT / "runs" / rn, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "2048", "--energy-pot", "1.5"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "ENERGY POTENTIAL - Phi = 1.5 x scale x (z + |v|^2 / 2g)" in r.stdout
    assert _cfg(run)["energy_pot"] == 1.5
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + ["--steps", "4096"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "energy_pot=1.5" in r2.stdout and _cfg(run2)["energy_pot"] == 1.5
    r3 = _run([sys.executable, "-u", str(TRAIN), "--run", "epot_bad"] + SMOKE
              + ["--steps", "2048", "--energy-pot", "-1"])
    assert r3.returncode != 0 and "--energy-pot must be finite and >= 0" in r3.stdout + r3.stderr
    # --energy-pot-front on a resume of a seeded reservoir: the band edge becomes finite
    from test_respawn_greedy import _seeded_ckpt
    d0s, seeded = _seeded_ckpt("epot_front_base")
    run3 = "epot_front"
    shutil.rmtree(ROOT / "runs" / run3, ignore_errors=True)
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", run3, "--ckpt", str(seeded)] + SMOKE
              + ["--steps", "8192", "--respawn-reservoir", "5000", "--energy-pot", "1",
                 "--energy-pot-front", "0.02"])
    assert r4.returncode == 0, r4.stdout[-4000:] + r4.stderr[-4000:]
    assert "--energy-pot-front 0.02: the energy term stops" in r4.stdout
    assert "  epot<" in r4.stdout, r4.stdout[-3000:]
    assert _cfg(run3)["energy_pot_front"] == 0.02
    r5 = _run([sys.executable, "-u", str(TRAIN), "--run", "epot_front_re", "--ckpt",
               str(ROOT / "runs" / run3 / "ckpt_final.pt")] + SMOKE + ["--steps", "10240"])
    assert r5.returncode == 0, r5.stdout[-4000:] + r5.stderr[-4000:]
    assert "energy_pot_front=0.02" in r5.stdout
    r6 = _run([sys.executable, "-u", str(TRAIN), "--run", "epot_bad2"] + SMOKE
              + ["--steps", "2048", "--energy-pot-front", "0.02"])
    assert r6.returncode != 0 and "needs --energy-pot C > 0" in r6.stdout + r6.stderr
    for rn in (run3, "epot_front_re", "epot_bad2"):
        shutil.rmtree(ROOT / "runs" / rn, ignore_errors=True)
    shutil.rmtree(d0s, ignore_errors=True)
    rec = _run([sys.executable, "-u", str(RECORD), str(ROOT / "runs" / run2 / "ckpt_final.pt"),
                "--map", str(SID), "--episodes", "1", "--ep-ticks", "200",
                "--out", str(ROOT / "runs" / run2 / "rec.jsonl")])
    assert rec.returncode == 0, rec.stdout[-3000:] + rec.stderr[-3000:]
    for rn in (run, run2, "epot_bad"):
        shutil.rmtree(ROOT / "runs" / rn, ignore_errors=True)


def _first_without(path: str, needle: bytes):
    """(ref, source) of `path` at the first commit in the first-parent history that lacks
    `needle` - the module as it was before the change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and needle not in s.stdout:
            return ref, s.stdout
    return None, None


def test_the_reward_with_the_flag_off_is_bit_identical_to_the_module_before_the_change():
    """The pre-change RaceReward against the current one, flag off, on a ratchet + time penalty +
    novelty configuration over random states with deaths: identical reward arrays, bit for bit."""
    import importlib
    ref, src = _first_without("python/surfgym/rewards.py", b"energy_pot")
    if ref is None:
        pytest.skip("no pre-change rewards.py in the first-parent history")
    tmp = ROOT / "python" / "surfgym" / "_rewards_pre_epot.py"
    tmp.write_bytes(src)
    try:
        old = importlib.import_module("surfgym._rewards_pre_epot")
        n = 16

        class _Core(_FakeCore):
            def map_bounds(self):
                return (np.array([-4096.0, -4096.0, -4096.0], np.float32),
                        np.array([4096.0, 4096.0, 4096.0], np.float32))

        out = []
        for mod in (old, sys.modules["surfgym.rewards"]):
            rng = np.random.default_rng(7)
            core = _Core(n).at(np.zeros(n))
            rr = mod.RaceReward(_FlatField(), scale=SCALE, time_pen=0.005, stall_ticks=10 ** 9,
                                ratchet=True, ratchet_d0=5000.0, int_coef=0.25, int_view=0)
            rr.on_reset(core)
            seq = []
            for t in range(200):
                core.states_view["origin"][:] = rng.uniform(-3000, 3000, (n, 3)).astype(np.float32)
                core.states_view["velocity"][:] = rng.normal(0, 900, (n, 3)).astype(np.float32)
                done = (rng.random(n) < 0.05).astype(np.uint8)
                seq.append(rr(None, None, None, np.zeros(n, np.float32), done,
                              np.zeros(n, np.uint8), core).copy())
            out.append(np.stack(seq))
        assert out[0].dtype == out[1].dtype and np.array_equal(out[0], out[1])
    finally:
        tmp.unlink(missing_ok=True)
        sys.modules.pop("surfgym._rewards_pre_epot", None)


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    """The pre-change trainer against the current one (the same surfgym package), flag off: same
    config, csv, eval bytes and weights."""
    import torch
    from test_respawn_backward_mix import _csv
    ref, src = _first_without("python/train_fast.py", b"--energy-pot")
    if ref is None:
        pytest.skip("no pre-change trainer in the first-parent history")
    pre = ROOT / "python" / "_train_fast_pre_epot.py"
    pre.write_bytes(src)
    try:
        for run, script in (("epot_off_new", TRAIN), ("epot_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                     + ["--steps", "4096", "--record-every", "2048", "--eval-eps", "2",
                        "--eval-stall", "1", "--stall-secs", "2"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "epot_off_new", ROOT / "runs" / "epot_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "energy_pot" not in ca
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert ta and [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
