"""--energy-pot K (2026-10-23, autoresearch r7b2): the --speed-pot form on the ENERGY HEIGHT,
Phi = K * h / 1000 with h = z + |v|^2 / (2 g), paid as its change every call.

(a) RaceReward: each call pays K/1000 x the change of h; GoldSrc's half-step gravity keeps h
    EXACTLY in free fall (so a fall pays 0 and only contacts / friction / air acceleration move
    it); the sum telescopes; an ended row pays 0 and re-anchors; K = 0 touches nothing; pop_stats
    reports the gained / lost energy only when on.
(b) The real core: an airborne player's h is constant to float precision until it lands.
(c) The trainer (CPU, sidistic): the startup refusal, the config key, the step line, a bare
    resume restores it, record_ckpt records the checkpoint; flag off is bit-identical to the
    trainer before the change.
"""
from __future__ import annotations

import json
import re
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
from test_ez_viewc import SID, SMOKE, TRAIN, _run, needs_run  # noqa: E402

K = 2.0
G = 800.0


def _rr(k):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9)
    rr.energy_pot = k
    rr.on_reset(core)
    return rr, core


def _put(core, z, v):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:] = np.asarray([v], np.float32)
    return core


def _h(z, v):
    return z + float(np.dot(v, v)) / (2.0 * G)


def test_each_call_pays_the_change_of_energy_height():
    rr, core = _rr(K)
    pts = [(1000.0, (800.0, 0.0, 0.0)), (1000.0, (900.0, 0.0, 0.0)),
           (900.0, (900.0, 0.0, 0.0)), (900.0, (900.0, 0.0, -400.0))]
    out = [float(_step(rr, _put(core, z, v))[0]) for z, v in pts]
    hs = [_h(z, np.asarray(v)) for z, v in pts]
    want = [0.0] + [K / 1000.0 * (hs[i] - hs[i - 1]) for i in range(1, len(hs))]
    assert out == pytest.approx(want, abs=1e-5)


def test_free_fall_pays_nothing_and_a_contact_is_charged():
    """GoldSrc integrates gravity in two half steps around the move, which keeps z + v^2/2g
    exactly: a 2 s fall pays ~0; then a landing that kills vz is charged its energy."""
    rr, core = _rr(K)
    z, vx, vz, dt = 5000.0, 700.0, 300.0, 0.01
    paid = []
    for _ in range(200):
        paid.append(float(_step(rr, _put(core, z, (vx, 0.0, vz)))[0]))
        vh = vz - 0.5 * G * dt
        z += vh * dt
        vz = vh - 0.5 * G * dt
    assert abs(sum(paid)) < 1e-3
    h_before = _h(z, np.asarray((vx, 0.0, vz)))
    _step(rr, _put(core, z, (vx, 0.0, vz)))
    r_land = float(_step(rr, _put(core, z, (vx, 0.0, 0.0)))[0])       # the floor took vz
    assert r_land == pytest.approx(K / 1000.0 * (_h(z, np.asarray((vx, 0.0, 0.0))) - h_before),
                                   abs=1e-5)
    assert r_land < 0.0


def test_the_sum_telescopes():
    rr, core = _rr(K)
    rng = np.random.default_rng(0)
    pts = [(float(rng.uniform(0, 9000)), tuple(rng.normal(0, 900, 3))) for _ in range(40)]
    paid = sum(float(_step(rr, _put(core, z, v))[0]) for z, v in pts)
    h0, h1 = _h(pts[0][0], np.asarray(pts[0][1], np.float32)), _h(pts[-1][0], np.asarray(
        pts[-1][1], np.float32))
    assert paid == pytest.approx(K / 1000.0 * (h1 - h0), abs=2e-3)


def test_an_ended_row_pays_nothing_and_reanchors():
    rr, core = _rr(K)
    _step(rr, _put(core, 3000.0, (1800.0, 0.0, 0.0)))
    r_end = float(_step(rr, _put(core, 100.0, (0.0, 0.0, 0.0)), done=1)[0])   # the new spawn
    assert r_end == pytest.approx(0.0, abs=1e-6)
    r_next = float(_step(rr, _put(core, 100.0, (400.0, 0.0, 0.0)))[0])        # from the spawn
    assert r_next == pytest.approx(K / 1000.0 * 400.0 ** 2 / (2 * G), abs=1e-5)


def test_zero_is_off_and_the_stats_appear_only_when_on():
    a, ca = _rr(0.0)
    b, cb = _rr(K)
    for z, v in ((1000.0, (500.0, 0.0, 0.0)), (800.0, (900.0, 0.0, 0.0)),
                 (800.0, (100.0, 0.0, 0.0))):
        assert float(_step(a, _put(ca, z, v))[0]) == pytest.approx(0.0, abs=1e-7)
        _step(b, _put(cb, z, v))
    assert a._en_prev is None
    sa, sb = a.pop_stats(), b.pop_stats()
    assert not any(k.startswith("en_") for k in sa)
    assert "en_gain_per_ep" in sb and "en_loss_per_ep" in sb


class _LineField:
    """d = x: the field falls along +x (the goal at x 0)."""
    reach_max = 1e9
    cell = 32.0

    def sample(self, p):
        return np.asarray(p, np.float64)[:, 0].astype(np.float32)


def _rr_front(k, edge, ratchet=True):
    core = _FakeCore(1).at([5000.0])
    rr = RaceReward(_LineField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9, ratchet=ratchet)
    rr.energy_pot = k
    rr.en_front_d = edge
    rr.on_reset(core)
    return rr, core


def _go(rr, core, x, vx, done=0):
    core.states_view["origin"][:, 0] = np.float32(x)
    core.states_view["velocity"][:] = np.asarray([[vx, 0.0, 0.0]], np.float32)
    return float(_step(rr, core, done=done)[0])


def _e(v):
    return K / 1000.0 * v * v / (2 * G)


def test_front_latch_pays_up_to_the_band_and_stops_inside_it():
    """moves of <= 100 u per call (the ratchet's teleport clip), the band edge at d 3,000"""
    rr, core = _rr_front(K, edge=3000.0)
    _go(rr, core, 3100.0, 1000.0)                                  # anchors
    # outside the band: the ratchet pays 50 and the energy change is paid
    assert _go(rr, core, 3050.0, 1200.0) == pytest.approx(50.0 + _e(1200) - _e(1000), abs=1e-3)
    # the transition that ENTERS the band (record 2990 <= 3000) is still paid
    assert _go(rr, core, 2990.0, 1500.0) == pytest.approx(60.0 + _e(1500) - _e(1200), abs=1e-3)
    # inside it the energy term is off: only the ratchet pays, a speed loss costs nothing
    assert _go(rr, core, 2940.0, 500.0) == pytest.approx(50.0, abs=1e-3)
    # backing out of the band keeps the latch (the record is still inside)
    assert _go(rr, core, 3020.0, 400.0) == pytest.approx(0.0, abs=1e-3)
    assert bool(rr._en_off[0])
    # an ended row: its new episode spawns OUTSIDE the band and is paid again from there
    assert _go(rr, core, 6000.0, 0.0, done=1) == pytest.approx(0.0, abs=1e-6)
    assert not bool(rr._en_off[0])
    assert _go(rr, core, 6000.0, 800.0) == pytest.approx(_e(800), abs=1e-3)


def test_front_latch_a_spawn_inside_the_band_is_off_from_the_start():
    rr, core = _rr_front(K, edge=3000.0)
    _go(rr, core, 3100.0, 1000.0)
    _go(rr, core, 2000.0, 0.0, done=1)                             # the new episode spawns in the band
    assert bool(rr._en_off[0])
    assert _go(rr, core, 2000.0, 900.0) == pytest.approx(0.0, abs=1e-6)


def test_front_latch_off_is_the_plain_term():
    """edge = inf (no reservoir yet, or the flag off): the ratchet plus the plain energy term on
    every call, inside the would-be band too, and no latch state"""
    rr, core = _rr_front(K, edge=float("inf"))
    _go(rr, core, 3100.0, 1000.0)
    rec, prev = 3100.0, 1000.0
    for x, v in ((3050.0, 1300.0), (2990.0, 700.0), (2940.0, 1500.0), (2960.0, 200.0)):
        pay = max(0.0, rec - x)
        rec = min(rec, x)
        assert _go(rr, core, x, v) == pytest.approx(pay + _e(v) - _e(prev), abs=1e-3)
        prev = v
    assert rr._en_off is None


@needs_run
def test_the_real_core_keeps_energy_height_in_free_flight():
    """Launched up from the spawn platform with no keys held: on every tick where only gravity
    acted (vz fell by exactly g dt, the horizontal velocity unchanged) h stays put to float
    precision; the ceiling bonk and the landing are the ticks that change it, and they lower it."""
    from surfgym import SurfCore, default_config
    from surfgym.core import ACTION_NVEC, STATE_DTYPE
    sc = SurfCore(str(SID), default_config(num_envs=1, spawn_mode=2, max_episode_ticks=400,
                                           sv_maxvelocity=4000.0, lidar_w=0, lidar_h=0))
    try:
        sc.reset(0)
        st = np.array(sc.get_states()[0:1], STATE_DTYPE).copy()
        st["velocity"] = np.asarray([[300.0, 0.0, 700.0]], np.float32)
        st["origin"][0, 2] += 40.0
        st["onground"] = -1
        sc.set_state(0, st[0])
        # no yaw / pitch change, no keys: zero wishspeed, so nothing but gravity acts in the air
        idle = np.asarray([[ACTION_NVEC[0] // 2, ACTION_NVEC[1] // 2, 1, 1, 0, 0]], np.int32)
        g = float(sc.config.phys.sv_gravity)
        dt = sc.tick_ms * 1e-3
        z, v = [], []
        for _ in range(150):
            s = sc.get_states()[0]
            z.append(float(s["origin"][2]))
            v.append(np.asarray(s["velocity"], np.float64))
            sc.step(idle)
        z, v = np.asarray(z), np.asarray(v)
        h = z + (v * v).sum(1) / (2 * g)
        dh = np.diff(h)
        free = ((np.abs(np.diff(v[:, 2]) + g * dt) < 1e-2)
                & (np.abs(np.diff(v[:, 0])) < 1e-3) & (np.abs(np.diff(v[:, 1])) < 1e-3))
        assert free.sum() > 20, "the player should fly for a while"
        assert np.max(np.abs(dh[free])) < 0.05, dh[free][:10]
        assert (~free).any() and dh[~free].min() < -1.0      # a contact took energy
    finally:
        sc.close()


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_carries_restores_and_records_it():
    run = "enpot_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "6144", "--energy-pot", "1.5", "--ep-ticks", "40"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert _cfg(run)["energy_pot"] == 1.5
    assert "  en +" in r.stdout and "u/ep" in r.stdout, r.stdout[-3000:]
    run2 = "enpot_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + ["--steps", "10240"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "energy_pot=1.5" in r2.stdout
    assert _cfg(run2)["energy_pot"] == 1.5
    rec = ROOT / "runs" / run / "rec.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "enpot_bad"] + SMOKE
              + ["--steps", "2048", "--energy-pot", "-1"])
    assert r4.returncode != 0 and "--energy-pot K needs K >= 0" in r4.stdout + r4.stderr
    r5 = _run([sys.executable, "-u", str(TRAIN), "--run", "enpot_bad"] + SMOKE
              + ["--steps", "2048", "--energy-pot-front", "0.01"])
    assert r5.returncode != 0 and "it needs --energy-pot K > 0" in r5.stdout + r5.stderr
    for dd in (run, run2, "enpot_bad"):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)


@needs_run
def test_the_front_latch_runs_on_a_seeded_reservoir_and_is_restored():
    from test_respawn_greedy import GREEDY_SMOKE, _seeded_ckpt
    d0, seeded = _seeded_ckpt("enfront_base")
    run = "enfront_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)] + SMOKE
             + GREEDY_SMOKE + ["--steps", "10240", "--energy-pot", "1", "--energy-pot-front",
                               "0.5", "--ep-ticks", "60"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "--energy-pot-front 0.5: the energy term is paid only until" in out
    assert re.search(r"  en \+[0-9,]+/-[0-9,]+u/ep off \d+%", out), out[-3000:]
    cfg = _cfg(run)
    assert cfg["energy_pot"] == 1.0 and cfg["energy_pot_front"] == 0.5
    run2 = "enfront_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + GREEDY_SMOKE
              + ["--steps", "14336"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "energy_pot_front=0.5" in r2.stdout and _cfg(run2)["energy_pot_front"] == 0.5
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


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
    from test_respawn_backward_mix import _csv
    pre = ROOT / "python" / "_train_fast_pre_enpot.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("enpot_off_new", TRAIN), ("enpot_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                     + ["--steps", "6144", "--record-every", "4096", "--eval-eps", "2",
                        "--eval-stall", "1", "--stall-secs", "2"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "enpot_off_new", ROOT / "runs" / "enpot_off_old"
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
