"""--stall-energy EPS, --eval-patience-energy and --energy-pot-rel (zlr1b1, 2026-10-31).

A loop (ride a ramp side to side until the speed carries you to the next one) gains energy
height h = z + |v|^2 / (2 g) while the field distance does not move. The training stall kill,
the reservoir's stagnant filter and the eval patience rule all watch the field alone, so every
loop is killed (training), never harvested (stagnant_mask) and cut (eval). These flags add an
energy channel to the stall clocks and a unit for --energy-pot in the field's own scale.

(a) RaceReward --stall-energy: an episode whose h keeps rising (by less than EPS per call, so
    the per-call field rule would never fire) is never stall-killed; one whose h is flat is
    killed after the window; the anchor is the h at the last energy re-arm (cumulative); an
    ended row re-anchors on its new spawn; stagnant_mask sees the re-arm; 0 = off allocates
    nothing; pop_stats reports en_stall_frac only when on.
(b) The eval patience hooks (single and side by side), energy=True: a flat-field line that is
    pumping is never cut, a flat-energy one is cut after the window; energy=False is the
    shipped rule.
(c) The recorder lists the three flags as training-only.
(d) The trainer (CPU, sidistic): the flags print, the config carries them, a bare resume
    restores them, bad values are refused, record_ckpt records the checkpoint.
(e) Flag off: the trainer before this change and after it give the same config, progress.csv
    (fps aside), eval trajectory bytes and weights on the same smoke.
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

from surfgym.core import STATE_DTYPE                              # noqa: E402
from surfgym.rewards import RaceReward                            # noqa: E402
from test_arc_death_charge import _FakeCore, _FlatField, _step    # noqa: E402
from test_ez_viewc import DLL, SID, SMOKE, TRAIN, _run          # noqa: E402

# the smoke runs on sidistic like the rest of the suite, or on this lineage's own map where
# sidistic is not installed (the agent host carries surf_za_loop and its baked fields only)
_ZL = ROOT / "maps" / "surf_za_loop.bsp"
_MAP = SID if (SID.exists() and SID.with_name("surf_src_sidistic.goal_32.npz").exists()) else _ZL
SMOKE = [str(_MAP) if a == str(SID) else a for a in SMOKE]
needs_run = pytest.mark.skipif(
    not (_MAP.exists() and _MAP.with_name(_MAP.stem + ".goal_32.npz").exists() and DLL.exists()),
    reason="needs the built core + a smoke map and its prebaked goal field")

G = 800.0


class _XField:
    """d = x (moving -x is progress)"""
    reach_max = 1e9
    cell = 32.0

    def sample(self, pts):
        return np.asarray(pts, np.float64)[:, 0].copy()


def _rr(eps, window=100):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=window, stall_eps=32.0)
    rr.stall_energy = eps
    rr.on_reset(core)
    return rr, core


def _put(core, z, vx):
    core.states_view["origin"][:, 2] = np.float32(z)
    core.states_view["velocity"][:] = np.asarray([[vx, 0.0, 0.0]], np.float32)
    return core


def _speed_for(h, z=0.0):
    return float(np.sqrt(max(h - z, 0.0) * 2.0 * G))


def _ride(rr, core, hs, ends=()):
    """one call per energy height h (z 0, the speed that gives h); -> the calls that killed"""
    kills = []
    for t, h in enumerate(hs):
        _step(rr, _put(core, 0.0, _speed_for(h)), done=int(t in ends))
        m = rr.pop_stall_mask()
        if m is not None and m[0]:
            kills.append(t)
    return kills


# ==========================================================================
# (a) the reward's stall channel
# ==========================================================================
def test_off_the_field_alone_kills_a_pumping_episode():
    rr, core = _rr(0.0)
    # the reward's clock counts every call from the reset, so a 100-call window fires on call 99
    kills = _ride(rr, core, 100.0 + 2.0 * np.arange(350))        # h +2u per call, d flat
    assert kills == [99, 199, 299]
    assert rr._hst is None


def test_on_a_pumping_episode_is_never_killed_however_slowly():
    rr, core = _rr(32.0)
    # +2u per call: far below 32u in ONE call (the field rule's form), but 32u every 17 calls
    assert _ride(rr, core, 100.0 + 2.0 * np.arange(1000)) == []
    # +0.33u per call: 32u every 97 calls, inside a 100-call window
    rr, core = _rr(32.0)
    assert _ride(rr, core, 100.0 + 0.33 * np.arange(1000)) == []


def test_on_a_flat_or_falling_energy_is_killed_after_the_window():
    rr, core = _rr(32.0)
    hs = [400.0] + [400.0 - (i % 9) * 3.0 for i in range(350)]  # wobbles, never +32u
    assert _ride(rr, core, hs) == [99, 199, 299]
    rr, core = _rr(32.0)
    assert _ride(rr, core, 2000.0 - 5.0 * np.arange(250)) == [99, 199]


def test_the_anchor_is_the_last_energy_rearm_not_the_running_max():
    """a rise to the anchor + 31 and back never re-arms; the next re-arm needs anchor + 32"""
    rr, core = _rr(32.0)
    hs = [500.0] + [531.0, 500.0] * 60 + [533.0] + [500.0] * 99 + [500.0]
    kills = _ride(rr, core, hs)
    # killed on call 99 (no re-arm yet); call 121 (533 > 500 + 32) re-arms; 100 calls later
    assert kills == [99, 221]
    assert float(rr._hst[0]) == pytest.approx(533.0, abs=0.05)


def test_an_ended_row_reanchors_on_its_new_spawn():
    rr, core = _rr(32.0, window=50)
    # pump to h 2,000, then the episode ends and the new one spawns at h 100: a rise to 140
    # (+40 over the NEW spawn, far under the old anchor) re-arms it
    hs = list(100.0 + 20.0 * np.arange(96)) + [100.0] + [100.0] * 40 + [140.0] + [140.0] * 60
    kills = _ride(rr, core, hs, ends={96})
    assert kills == [96 + 41 + 50]
    assert float(rr._hst[0]) == pytest.approx(140.0, abs=0.05)


def test_stagnant_mask_sees_the_energy_rearm():
    """the reservoir never harvests a stagnant state; a pumping one is not stagnant"""
    rr, core = _rr(32.0, window=10 ** 9)
    _ride(rr, core, 100.0 + 2.0 * np.arange(600))
    assert not bool(rr.stagnant_mask(ticks=300)[0])
    off, core2 = _rr(0.0, window=10 ** 9)
    _ride(off, core2, 100.0 + 2.0 * np.arange(600))
    assert bool(off.stagnant_mask(ticks=300)[0])


def test_the_field_still_rearms_and_the_stats_appear_only_when_on():
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_XField(), scale=1.0, time_pen=0.0, stall_ticks=100, stall_eps=32.0)
    rr.stall_energy = 32.0
    rr.on_reset(core)
    for t in range(300):                                          # field progress, flat energy
        core.states_view["origin"][:, 0] = np.float32(-40.0 * (t + 1))
        _step(rr, _put(core, 0.0, 400.0))
        assert rr.pop_stall_mask() is None
    st = rr.pop_stats()
    assert "en_stall_frac" in st and st["en_stall_frac"] == pytest.approx(0.0)
    rr2, core2 = _rr(32.0)
    _ride(rr2, core2, 100.0 + 2.0 * np.arange(200))
    assert rr2.pop_stats()["en_stall_frac"] == pytest.approx(1.0)
    off, core3 = _rr(0.0)
    _ride(off, core3, 100.0 + 2.0 * np.arange(50))
    assert "en_stall_frac" not in off.pop_stats()


def test_the_flat_field_class_samples_flat():
    """guard for the fixtures above: _FlatField pays no progress, so only energy can re-arm"""
    assert np.all(_FlatField().sample(np.zeros((3, 3))) == 0.0)


# ==========================================================================
# (b) the eval patience hooks
# ==========================================================================
class _EvalCore:
    def __init__(self, n=1):
        from types import SimpleNamespace
        self.num_envs = n
        self.failed = np.zeros(n, np.uint8)
        self.states_view = np.zeros(n, STATE_DTYPE)
        self.config = SimpleNamespace(phys=SimpleNamespace(sv_gravity=G))

    def force_fail(self, mask):
        self.failed |= np.asarray(mask, np.uint8)


def _drive(core, hook, hs, n=1):
    kills = []
    for t, h in enumerate(hs):
        core.states_view["origin"][:, 0] = 1000.0                 # the field never moves
        core.states_view["velocity"][:, 0] = _speed_for(h)
        core.failed[:] = 0
        hook(t, None, None, np.zeros(n, bool), np.zeros(n, bool))
        if core.failed[0]:
            kills.append(t)
    return kills


@pytest.mark.parametrize("par", [False, True])
def test_eval_patience_energy_never_cuts_a_pumping_line(par):
    from train_fast import make_eval_patience_hook, make_eval_patience_hook_par
    pump = 100.0 + 0.4 * np.arange(2000)                          # 32u every 80 ticks
    flat = np.full(400, 300.0)
    for energy, hs, want in ((True, pump, []), (False, pump, [100, 200, 300]),
                             (True, flat, [100, 200, 300])):
        core = _EvalCore(1)
        hook = (make_eval_patience_hook_par(core, _XField(), 100, 1, energy=energy) if par
                else make_eval_patience_hook(core, _XField(), 100, energy=energy))
        kills = _drive(core, hook, hs)
        assert kills[:3] == want, (energy, kills[:5])


def test_eval_patience_energy_par_is_per_env():
    from train_fast import make_eval_patience_hook_par
    core = _EvalCore(2)
    hook = make_eval_patience_hook_par(core, _XField(), 100, 2, energy=True)
    killed = []
    for t in range(250):
        core.states_view["origin"][:, 0] = 1000.0
        core.states_view["velocity"][0, 0] = _speed_for(100.0 + 0.5 * t)   # env 0 pumps
        core.states_view["velocity"][1, 0] = _speed_for(300.0)              # env 1 does not
        core.failed[:] = 0
        hook(t, None, None, np.zeros(2, bool), np.zeros(2, bool))
        killed.append(tuple(core.failed.tolist()))
    assert not any(k[0] for k in killed)
    assert sum(k[1] for k in killed) == 2 and killed[100] == (0, 1)


# ==========================================================================
# (c) the recorder
# ==========================================================================
def test_the_recorder_lists_the_flags_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    for k in ("stall_energy", "eval_patience_energy", "energy_pot_rel"):
        assert k in record_ckpt.TRAIN_ONLY, k


# ==========================================================================
# (d) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_the_trainer_carries_restores_refuses_and_records_them():
    run = "stallen_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "6144", "--stall-energy", "32", "--energy-pot", "1",
                "--energy-pot-rel", "1", "--eval-patience", "5", "--eval-patience-energy", "1",
                "--record-every", "4096", "--ep-ticks", "40"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "--stall-energy 32: the stall timer also re-arms" in out, out[-3000:]
    assert "--energy-pot-rel: energy height paid at 1 x the shaping scale" in out, out[-3000:]
    assert "nor its energy height" in out, out[-3000:]
    assert "  se " in out, out[-3000:]
    cfg = _cfg(run)
    assert cfg["stall_energy"] == 32.0 and cfg["energy_pot_rel"] == 1
    assert cfg["eval_patience_energy"] == 1 and cfg["eval_patience"] == 5.0
    trajs = sorted((ROOT / "runs" / run).glob("traj_*.jsonl"))
    assert trajs
    hdr = json.loads(trajs[0].open(encoding="utf-8").readline())
    assert hdr.get("eval_patience_energy") == 1
    run2 = "stallen_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + ["--steps", "10240"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    for s in ("stall_energy=32", "energy_pot_rel=1", "eval_patience_energy=1"):
        assert s in r2.stdout, s
    c2 = _cfg(run2)
    assert c2["stall_energy"] == 32.0 and c2["energy_pot_rel"] == 1
    assert c2["eval_patience_energy"] == 1
    rec = ROOT / "runs" / run / "rec.jsonl"
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
               str(ROOT / "runs" / run / "ckpt_final.pt"), "--out", str(rec),
               "--episodes", "1", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    for bad, msg in ((["--stall-energy", "-1"], "--stall-energy EPS needs a finite EPS >= 0"),
                     (["--energy-pot-rel", "1"], "--energy-pot-rel says what --energy-pot K"),
                     (["--eval-patience-energy", "1"], "it needs --eval-patience SECS > 0")):
        r4 = _run([sys.executable, "-u", str(TRAIN), "--run", "stallen_bad"] + SMOKE
                  + ["--steps", "2048"] + bad)
        assert r4.returncode != 0 and msg in r4.stdout + r4.stderr, (bad, r4.stdout[-800:])
    for dd in (run, run2, "stallen_bad"):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)


# ==========================================================================
# (e) flag off is bit-identical
# ==========================================================================
def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--stall-energy" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    from test_respawn_backward_mix import _csv
    pre = ROOT / "python" / "_train_fast_pre_stallen.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("stallen_off_new", TRAIN), ("stallen_off_old", pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                     + ["--steps", "6144", "--record-every", "4096", "--eval-eps", "2",
                        "--eval-patience", "5", "--energy-pot", "1", "--stall-secs", "2"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "stallen_off_new", ROOT / "runs" / "stallen_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "stall_energy" not in ca and "energy_pot_rel" not in ca
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
