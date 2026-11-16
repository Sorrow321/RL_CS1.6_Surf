"""--energy-ratchet (zlr1b2 4439aab8, 2026-10-31; ported to ab/eratchet 2026-11-16): --energy-pot's
RATCHET form - pay only new records of the episode's energy height h = z + |v|^2 / (2 g), charge
nothing for a loss.

Why: under the per-call form a learner that cannot pump yet is charged for every landing, and on
surf_za_loop (zlr1b2 A1, K 12) its greedy learned to dive into the kill plane in 3-5 s: dying was
the cheapest episode. Recipe v14's band-bonus form pays ~0 on za_loop, where the ~27% plateau sits
in a half-pipe the policy has to pump (sub-agent P, w18). The ratchet never charges a contact,
pays a lost-and-regained h once, and is bounded per episode by K x (h max - h at spawn).

1. RaceReward: records pay, losses are free, a regain pays once; the read-out still counts the
   losses; an ended row re-anchors; off is the per-call form with no ratchet state.
2. record_ckpt and the wiring in train_fast (source).
3. CPU runs (the built core + a map with its prebaked goal field; ERATCHET_TEST_MAP /
   SURFCORE_DLL): the flag OFF is bit-identical to the parent commit - the config dump, the eval
   trajectories and every training row - with v14's band-bonus energy form and with the per-call
   form; ON writes its config key, announces the rule, reads out en +a/-b, refuses without
   --energy-pot or with --energy-pot-front, and a bare resume restores it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
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


# --- 1. RaceReward --------------------------------------------------------------------------------

def _rr(ratchet, k=K):
    core = _FakeCore(1).at([0.0])
    rr = RaceReward(_FlatField(), scale=1.0, time_pen=0.0, stall_ticks=10 ** 9)
    rr.energy_pot = k
    rr.en_ratchet = ratchet
    rr.on_reset(core)
    return rr, core


def _vx_for(h, z=0.0):
    """the horizontal speed that gives energy height h at height z"""
    return float(np.sqrt(max(0.0, 2.0 * G * (h - z))))


def _put(core, h):
    core.states_view["origin"][:, 2] = np.float32(0.0)
    core.states_view["velocity"][:] = np.asarray([[_vx_for(h), 0.0, 0.0]], np.float32)
    return core


def _pay(rr, core, hs, done_at=()):
    return [float(_step(rr, _put(core, h), done=1 if i in done_at else 0)[0])
            for i, h in enumerate(hs)]


def test_records_pay_losses_are_free_regains_pay_once():
    rr, core = _rr(True)
    out = _pay(rr, core, [0.0, 100.0, 50.0, 80.0, 150.0, 150.0, 20.0])
    want = [0.0, 100.0, 0.0, 0.0, 50.0, 0.0, 0.0]
    assert out == pytest.approx([K / 1000.0 * w for w in want], abs=1e-3)


def test_the_loss_is_still_counted_for_the_readout():
    rr, core = _rr(True)
    _pay(rr, core, [0.0, 100.0, 40.0, 100.0])
    assert rr.en_gain == pytest.approx(100.0, abs=0.05)          # what was paid: the record's rise
    assert rr.en_loss == pytest.approx(60.0, abs=0.05)           # what the map took (not charged)
    st = rr.pop_stats()
    assert "en_gain_per_ep" in st and "en_loss_per_ep" in st
    assert rr.en_gain == 0.0 and rr.en_loss == 0.0               # drained by pop_stats


def test_an_ended_row_reanchors_on_the_next_spawn():
    rr, core = _rr(True)
    out = _pay(rr, core, [0.0, 300.0, 50.0, 120.0], done_at=(2,))
    # call 2 ends the episode (its row is the next spawn, h 50): it pays 0 and re-anchors there;
    # the new episode's rise 50 -> 120 is a record of ITS OWN and pays
    assert out == pytest.approx([0.0, K * 0.3, 0.0, K / 1000.0 * 70.0], abs=1e-3)


def test_the_sum_is_bounded_by_the_episode_record():
    rr, core = _rr(True)
    rng = np.random.default_rng(0)
    hs = list(np.cumsum(rng.normal(5.0, 60.0, 200)) + 500.0)
    paid = sum(_pay(rr, core, hs))
    assert paid == pytest.approx(K / 1000.0 * (max(hs) - hs[0]), abs=0.05)
    assert paid >= 0.0


def test_off_is_the_per_call_form():
    hs = [0.0, 100.0, 50.0, 80.0, 150.0, 20.0]
    rr, core = _rr(False)
    out = _pay(rr, core, hs)
    want = [0.0] + [K / 1000.0 * (hs[i] - hs[i - 1]) for i in range(1, len(hs))]
    assert out == pytest.approx(want, abs=1e-3)
    assert rr._en_rec is None
    rr0, _ = _rr(False, k=0.0)
    assert rr0.en_ratchet is False and rr0._en_rec is None


# --- 2. record_ckpt and the wiring ----------------------------------------------------------------

def test_record_ckpt_treats_it_as_train_only():
    import record_ckpt
    assert "energy_ratchet" in record_ckpt.TRAIN_ONLY
    record_ckpt.audit_cfg({"energy_pot": 12.0, "energy_ratchet": 1})


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--energy-ratchet"' in src
    for refusal in ('"--energy-ratchet is a form of --energy-pot: it needs --energy-pot K > 0"',
                    '"--energy-ratchet replaces the per-call term; not with --energy-pot-front"'):
        assert refusal in src
    assert "_s.reward_fn.en_ratchet = bool(args.energy_ratchet)" in src
    # the config key is written ONLY when on (a config without it is the old one) and restored
    assert '**({"energy_ratchet": 1}\n                          if args.reward == "race" and ' \
           'args.energy_ratchet else {})' in src
    assert 'args.energy_ratchet = int(ck_cfg["energy_ratchet"])' in src


# --- 3. CPU runs ----------------------------------------------------------------------------------

MAP = Path(os.environ.get("ERATCHET_TEST_MAP", os.environ.get(
    "INTP_TEST_MAP", str(ROOT / "maps" / "surf_unitfarmer2.bsp"))))
DLL = Path(os.environ.get("SURFCORE_DLL", str(
    ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))))
needs_run = pytest.mark.skipif(
    not (MAP.exists() and DLL.exists() and MAP.with_suffix(".goal_32.npz").exists()),
    reason="needs the built core + a map with its prebaked goal field (ERATCHET_TEST_MAP, "
           "SURFCORE_DLL)")
SMOKE = ["--map", str(MAP), "--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--lidar-range", "11500",
         "--lidar-near", "2000", "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4", "--gamma", "0.9995",
         "--gae", "0.95", "--clip", "0.2", "--vf", "0.5", "--ent", "0.005", "--n-steps", "8",
         "--epochs", "1", "--minibatches", "2", "--ep-ticks", "600", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "3500", "--train-stride", "1", "--yaw-adaptive",
         "--race-ratchet", "--respawn-frac", "0.9", "--respawn-margin", "10",
         "--respawn-reservoir", "1000", "--int-coef", "2.5", "--int-view", "0", "--int-speed", "8",
         "--ckpt-every", "1e9", "--eval-eps", "3", "--eval-greedy-only", "--no-compile", "--fp32",
         "--seed", "0"]
BAND = ["--energy-pot", "3", "--energy-pot-front", "0.01", "--energy-pot-front-bonus", "1"]
PER_CALL = ["--energy-pot", "12"]
RATCHET = ["--energy-pot", "12", "--energy-pot-front", "0", "--energy-pot-front-bonus", "0",
           "--energy-ratchet", "1"]
# 64-tick episodes: every env ends one inside a 6,144-step run, so the energy read-out has episodes
SHORT = ["--ep-ticks", "64", "--stall-secs", "60"]


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", SURFCORE_DLL=str(DLL))


def _train(root, run, extra, timeout=1800):
    shutil.rmtree(Path(root) / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(Path(root) / "python" / "train_fast.py"), "--run", run] \
        + SMOKE + extra
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(root),
                          timeout=timeout)


def _base_tree(tmp_path):
    """<tmp>/base/python as it was before --energy-ratchet: the parent of the first commit that
    mentions it (ERATCHET_BASE_REF overrides)."""
    ref = os.environ.get("ERATCHET_BASE_REF")
    if not ref:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-S",
                            "energy-ratchet", "--", "python/train_fast.py"],
                           capture_output=True, text=True)
        shas = r.stdout.split()
        if r.returncode != 0 or not shas:
            pytest.skip("the flag is not committed yet; set ERATCHET_BASE_REF")
        ref = shas[-1] + "^"
    base = tmp_path / "base"
    base.mkdir(parents=True, exist_ok=True)
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "-o", str(tar), ref,
                        "python"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git archive {ref} failed: {r.stderr[-400:]}")
    with tarfile.open(tar) as tf:
        tf.extractall(base)
    assert "energy-ratchet" not in (base / "python" / "train_fast.py").read_text(encoding="utf-8")
    return base


def _rows(path, drop=("time/fps",)):
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    keep = [i for i, h in enumerate(head) if h not in drop]
    return [[r.split(",")[i] for i in keep] for r in lines[1:]]


@needs_run
@pytest.mark.parametrize("form, energy", [("band", BAND), ("percall", PER_CALL)])
def test_the_flag_off_is_bit_identical_to_the_parent_commit(tmp_path, form, energy):
    base = _base_tree(tmp_path)
    run = f"erat_off_{form}"
    extra = ["--steps", "6144", "--record-every", "2048"] + energy + SHORT
    r_old = _train(base, run, extra)
    assert r_old.returncode == 0, r_old.stdout[-3000:] + r_old.stderr[-3000:]
    r_new = _train(ROOT, run, extra)
    assert r_new.returncode == 0, r_new.stdout[-3000:] + r_new.stderr[-3000:]
    d_old, d_new = base / "runs" / run, ROOT / "runs" / run
    c_old = json.loads((d_old / "run.json").read_text(encoding="utf-8"))["config"]
    c_new = json.loads((d_new / "run.json").read_text(encoding="utf-8"))["config"]
    assert "energy_ratchet" not in c_new and c_new == c_old
    assert c_new["energy_pot"] == float(energy[1])
    t_old, t_new = sorted(d_old.glob("traj_*.jsonl")), sorted(d_new.glob("traj_*.jsonl"))
    assert len(t_old) >= 3 and [p.name for p in t_old] == [p.name for p in t_new]
    for a, b in zip(t_old, t_new):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs"
    rows_old, rows_new = _rows(d_old / "progress.csv"), _rows(d_new / "progress.csv")
    assert len(rows_new) >= 3 and rows_old == rows_new
    # the energy term ran (the step line's read-out), so the identity covers its code path
    assert re.search(r"\ben\s+\+[\d,.]+/-[\d,.]+u/ep", r_new.stdout), r_new.stdout[-2000:]
    shutil.rmtree(d_new, ignore_errors=True)


@needs_run
def test_the_flag_refuses_without_the_term_or_with_the_band():
    r = _train(ROOT, "erat_noterm", ["--steps", "2048", "--energy-ratchet", "1"], timeout=600)
    assert r.returncode != 0
    assert "--energy-ratchet is a form of --energy-pot: it needs --energy-pot K > 0" in (
        r.stdout + r.stderr)
    r = _train(ROOT, "erat_band", ["--steps", "2048", "--energy-ratchet", "1"] + BAND,
               timeout=600)
    assert r.returncode != 0
    assert "--energy-ratchet replaces the per-call term; not with --energy-pot-front" in (
        r.stdout + r.stderr)
    for run in ("erat_noterm", "erat_band"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_run
def test_on_writes_announces_reads_out_and_a_bare_resume_restores_it():
    r = _train(ROOT, "erat_on", ["--steps", "6144", "--record-every", "2048"] + RATCHET + SHORT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "--energy-ratchet: the energy term pays only NEW records" in r.stdout
    ens = re.findall(r"\ben\s+\+([\d,.]+)/-([\d,.]+)u/ep", r.stdout)
    assert ens, r.stdout[-3000:]
    assert all(float(a.replace(",", "")) >= 0.0 for a, _ in ens)
    d = ROOT / "runs" / "erat_on"
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["energy_ratchet"] == 1 and c["energy_pot"] == 12.0
    assert "energy_pot_front" not in c and "energy_pot_front_bonus" not in c
    r2 = _train(ROOT, "erat_on2", ["--steps", "8192", "--record-every", "1e12",
                                   "--no-eval-at-start", "--ckpt", str(d / "ckpt_final.pt")]
                + SHORT)
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "energy_ratchet=1" in r2.stdout, r2.stdout[-3000:]
    assert "--energy-ratchet: the energy term pays only NEW records" in r2.stdout
    c2 = json.loads((ROOT / "runs" / "erat_on2" / "run.json").read_text(encoding="utf-8"))
    assert c2["config"]["energy_ratchet"] == 1
    for run in ("erat_on", "erat_on2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
