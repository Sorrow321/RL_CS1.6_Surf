"""--energy-pot K (2026-10-23, autoresearch r7b3): potential-based shaping on the mechanical energy
height h_E = z + |v|^2 / (2 g), Phi = K * scale * h_E (scale = the race reward's own progress
scale), paid per call as the change of Phi.

(a) RaceReward: each call pays K x scale x (the change of h_E); free fall (height traded for speed)
    pays nothing; the sum over an episode telescopes; an ended row (whose state is already the NEXT
    episode's spawn) pays 0 and re-anchors; K = 0 leaves the reward untouched.
(b) The trainer (CPU, sidistic): the key reaches run.json only when on, a bare resume restores it,
    record_ckpt records the checkpoint (TRAIN_ONLY), a negative K is refused; flag off is
    bit-identical to the trainer before the change (config, csv, eval bytes, weights).
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
from test_ez_viewc import SMOKE, TRAIN, _csv, _run, needs_run    # noqa: E402

K = 2.0
SCALE = 0.01
G = 800.0


def _rr(k):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=SCALE, time_pen=0.0, stall_ticks=10 ** 9)
    rr.energy_pot = k
    rr.on_reset(core)
    return rr, core


def _put(core, z, v):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:] = np.asarray([v], np.float32)
    return core


def _eh(z, v):
    return z + float(np.dot(v, v)) / (2.0 * G)


def test_each_call_pays_the_change_of_energy_height():
    rr, core = _rr(K)
    seq = [(1000.0, (800.0, 0.0, 0.0)), (1000.0, (1200.0, 0.0, 0.0)),
           (1300.0, (600.0, 0.0, 300.0)), (1300.0, (600.0, 0.0, 300.0))]
    out = [float(_step(rr, _put(core, z, v))[0]) for z, v in seq]
    want = [0.0] + [K * SCALE * (_eh(*seq[i]) - _eh(*seq[i - 1])) for i in (1, 2, 3)]
    assert out == pytest.approx(want, abs=1e-4)
    assert out[1] > 0.0 and out[2] < 0.0 and out[3] == pytest.approx(0.0, abs=1e-6)


def test_free_fall_trades_height_for_speed_at_no_cost():
    rr, core = _rr(K)
    z0, dt = 5000.0, 0.01
    paid = []
    for i in range(60):
        t = i * dt
        paid.append(float(_step(rr, _put(core, z0 - 0.5 * G * t * t, (900.0, 0.0, -G * t)))[0]))
    assert max(abs(p) for p in paid) < 1e-4


def test_a_climb_that_turns_speed_into_height_is_not_charged():
    """The horizontal speed potential charges this move (|v_xy| falls); the energy one does not."""
    rr, core = _rr(K)
    z, vx = 1000.0, 1600.0
    _step(rr, _put(core, z, (vx, 0.0, 0.0)))
    # all of the horizontal speed turned into height (an ideal ramp)
    r = float(_step(rr, _put(core, z + vx * vx / (2.0 * G), (0.0, 0.0, 0.0)))[0])
    assert r == pytest.approx(0.0, abs=1e-4)


def test_the_sum_telescopes():
    rr, core = _rr(K)
    seq = [(4000.0, (1500.0, 0.0, -100.0)), (3900.0, (1700.0, 200.0, -300.0)),
           (3700.0, (1100.0, -50.0, 0.0)), (3800.0, (1300.0, 400.0, 250.0)),
           (2500.0, (2400.0, 0.0, -800.0)), (2600.0, (1800.0, 0.0, 0.0))]
    paid = sum(float(_step(rr, _put(core, z, v))[0]) for z, v in seq)
    assert paid == pytest.approx(K * SCALE * (_eh(*seq[-1]) - _eh(*seq[0])), abs=1e-3)


def test_an_ended_row_pays_nothing_and_reanchors():
    rr, core = _rr(K)
    _step(rr, _put(core, 3000.0, (1800.0, 0.0, 0.0)))
    r_end = float(_step(rr, _put(core, 15000.0, (0.0, 0.0, 0.0)), done=1)[0])   # the new spawn
    assert r_end == pytest.approx(0.0, abs=1e-6)
    r_next = float(_step(rr, _put(core, 15000.0, (400.0, 0.0, 0.0)))[0])         # from the spawn
    assert r_next == pytest.approx(K * SCALE * 400.0 ** 2 / (2.0 * G), abs=1e-5)
    r_tr = float(_step(rr, _put(core, 9000.0, (0.0, 0.0, 0.0)), trunc=1)[0])     # truncation too
    assert r_tr == pytest.approx(0.0, abs=1e-6)


def test_zero_is_off():
    a, ca = _rr(0.0)
    for z, v in ((1000.0, (1000.0, 0.0, 0.0)), (900.0, (1300.0, 0.0, 0.0)), (2000.0, (700.0, 0, 0))):
        assert float(_step(a, _put(ca, z, v))[0]) == pytest.approx(0.0, abs=1e-7)
    assert a._eh_prev is None


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_carries_it_restores_it_and_the_recorder_accepts_it():
    run, run2 = "epot_on", "epot_on_re"
    for d in (run, run2, "epot_bad"):
        shutil.rmtree(ROOT / "runs" / d, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "6144", "--energy-pot", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert _cfg(run)["energy_pot"] == 1.0
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + ["--steps", "10240"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "energy_pot=1" in r2.stdout
    assert _cfg(run2)["energy_pot"] == 1.0
    rec = ROOT / "runs" / run / "rec.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "epot_bad"] + SMOKE
              + ["--steps", "2048", "--energy-pot", "-1"])
    assert r4.returncode != 0 and "--energy-pot K needs a finite K >= 0" in r4.stdout + r4.stderr
    for d in (run, run2, "epot_bad"):
        shutil.rmtree(ROOT / "runs" / d, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--energy-pot" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_epot.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("epot_off_new", TRAIN), ("epot_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                     + ["--steps", "6144", "--record-every", "4096"])
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
