"""--view-ou-temp: --view-ou-sigma's held yaw offset scaled by --unstuck's T.

sigma_eff = --view-ou-sigma x min(T, --unstuck-max) / --unstuck-max, written by
set_ou_temp wherever the unstuck schedule moves T (the restore and every
update), and read by the NEXT redraw of each env's held offset. With the flag
absent nothing reads it.

(a) the formula, pinned against the trainer's own expression;
(b) the refusals: no --view-ou-sigma, no --unstuck;
(c) CPU smoke (sidistic, tiny net): T rises under a short patience and the
    step line's "ou" column follows 0.8 x T / 4; the config dump carries
    view_ou_temp; a flagless resume restores it;
(d) flag-off identity: --view-ou-sigma WITHOUT --view-ou-temp (the touched
    init path) reproduces the parent commit's trainer - config dump,
    progress.csv (timing columns excluded), the greedy eval trajectory bytes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "python" / "train_fast.py"
MAPS = Path(os.environ.get("SURF_TEST_MAPS") or (ROOT / "maps"))
SID = MAPS / "surf_src_sidistic.bsp"
SID_FIELD = MAPS / "surf_src_sidistic.goal_32.npz"
DLL = ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so")
needs_run = pytest.mark.skipif(
    not (SID.exists() and SID_FIELD.exists() and DLL.exists()),
    reason="needs the built core + sidistic + its prebaked goal field")

SMOKE = ["--map", str(SID), "--reward", "race", "--envs", "32",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64", "--act-every", "4", "--pitch-rate", "1.33",
         "--teleport-fail", "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95",
         "--clip", "0.2", "--vf", "0.5", "--ent", "0.005", "--n-steps", "8",
         "--epochs", "1", "--minibatches", "2", "--ep-ticks", "3000",
         "--time-pen", "0.005", "--success-bonus", "50", "--finish-k", "0",
         "--stall-secs", "15", "--race-dist", "geodesic", "--maxvel", "4000",
         "--train-stride", "1", "--yaw-adaptive", "--respawn-frac", "0.9",
         "--respawn-margin", "10", "--respawn-reservoir", "1000",
         "--int-coef", "0.25", "--int-view", "8", "--int-speed", "3",
         "--ckpt-every", "1e9", "--eval-eps", "1", "--eval-greedy-only",
         "--seed", "7", "--view-continuous", "--view-absolute", "velocity"]
FAST_T = ["--unstuck", "--unstuck-patience", "1024", "--unstuck-period", "1024"]


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="3", NUMBA_NUM_THREADS="3")


def _run(cmd, timeout=900):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                          timeout=timeout, encoding="utf-8", errors="replace")


def _train(run, extra, script=TRAIN, steps="4096", record="1e12"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
             + ["--steps", steps, "--record-every", record] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


# ----------------------------------------------------------------- (a) formula
def _sigma(sig, T, cap):
    return sig * min(max(T, 0.0), cap) / cap


def test_formula_off_at_zero_full_at_the_cap_clamped_beyond():
    assert _sigma(0.8, 0.0, 2.0) == 0.0
    assert _sigma(0.8, 1.0, 2.0) == pytest.approx(0.4)
    assert _sigma(0.8, 2.0, 2.0) == pytest.approx(0.8)
    assert _sigma(0.8, 3.5, 2.0) == pytest.approx(0.8)      # never past the flag's sigma
    assert _sigma(0.8, -1.0, 2.0) == 0.0


def test_the_trainer_uses_that_expression_and_the_flag_defaults_off():
    src = TRAIN.read_text(encoding="utf-8")
    assert 'ap.add_argument("--view-ou-temp", type=int, default=None, choices=(0, 1)' in src
    assert "ou_sig_t[0] = OU_SIG * min(max(float(T), 0.0), OU_TMAX) / OU_TMAX" in src
    # the schedule moves the sigma at both places it moves T
    assert src.count("set_ou_temp(unstuck_T)") == 1
    assert src.count("set_ou_temp(_T_next)") == 1
    # flag off: the fixed sigma, exactly as before
    assert "ou_sig_t[0] = 0.0 if OU_TEMP else OU_SIG" in src
    rec = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"view_ou_temp",' in rec


# ---------------------------------------------------------------- (b) refusals
@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--unstuck", "--view-ou-temp", "1"], "it needs --view-ou-sigma > 0"),
    (["--view-ou-sigma", "0.8", "--view-ou-temp", "1"], "it needs --unstuck"),
])
def test_refusals(extra, msg):
    shutil.rmtree(ROOT / "runs" / "r1b3_ou_temp_refuse", ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "r1b3_ou_temp_refuse"] + SMOKE
             + ["--steps", "1024", "--record-every", "1e12"] + extra, timeout=600)
    shutil.rmtree(ROOT / "runs" / "r1b3_ou_temp_refuse", ignore_errors=True)
    assert r.returncode != 0
    assert msg in (r.stdout + r.stderr)


# ------------------------------------------------------------------ (c) smoke
@needs_run
def test_smoke_sigma_follows_T_dump_and_flagless_resume():
    run = "r1b3_ou_temp_smoke"
    r = _train(run, FAST_T + ["--view-ou-sigma", "0.8", "--view-ou-period", "4",
                              "--view-ou-temp", "1"])
    assert "--view-ou-temp: the offset's sigma follows --unstuck's T, 0.8 x min(T, 4) / 4" \
        in r.stdout
    pairs = re.findall(r"T (\d+\.\d+)(?:->(\d+\.\d+))? stuck [^\n]*? ou (\d+\.\d+)", r.stdout)
    assert len(pairs) >= 3, r.stdout[-3000:]
    for t_used, t_next, ou in pairs:
        t = float(t_next or t_used)
        assert float(ou) == pytest.approx(0.8 * min(t, 4.0) / 4.0, abs=1e-3)
    assert float(pairs[-1][2]) > 0.0           # T rose, so the offset is live
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["view_ou_temp"] == 1 and cfg["view_ou_sigma"] == 0.8 \
        and cfg["view_ou_period"] == 4
    run2 = "r1b3_ou_temp_smoke_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "6144", "--record-every", "1e12"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "view_ou_temp=1" in r2.stdout
    assert " ou " in r2.stdout
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


# --------------------------------------------------- (d) flag-off bit identity
def _parent_trainer() -> Path | None:
    """HEAD~1's train_fast.py when HEAD carries this flag (HEAD's otherwise),
    written beside the live one so its relative imports resolve."""
    ref = "HEAD"
    r = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:python/train_fast.py"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0 and "--view-ou-temp" in r.stdout:
        ref = "HEAD~1"
        r = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not r.stdout or "--view-ou-temp" in r.stdout:
        return None
    p = ROOT / "python" / "_train_fast_parent_for_ou_temp_test.py"
    p.write_text(r.stdout, encoding="utf-8")
    return p


@needs_run
def test_flag_off_is_bit_identical_to_the_parent_trainer():
    parent = _parent_trainer()
    if parent is None:
        pytest.skip("git could not serve a pre-flag train_fast.py")
    a, b = "r1b3_ou_temp_off_now", "r1b3_ou_temp_off_par"
    flags = FAST_T + ["--view-ou-sigma", "0.8", "--view-ou-period", "4"]
    try:
        _train(a, flags, record="2048")
        _train(b, flags, script=parent, record="2048")
        ca = json.loads((ROOT / "runs" / a / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((ROOT / "runs" / b / "run.json").read_text(encoding="utf-8"))["config"]
        assert ca == cb, "the flag-off config dump changed"
        ra, rb = _csv(a), _csv(b)
        assert list(ra[0]) == list(rb[0]), "the flag-off CSV header changed"
        assert len(ra) == len(rb)
        for xa, xb in zip(ra, rb):
            for k in xa:
                if k.startswith("time/"):
                    continue
                assert xa[k] == xb[k], (k, xa[k], xb[k])
        ta = sorted((ROOT / "runs" / a).glob("traj_*.jsonl"))
        tb = sorted((ROOT / "runs" / b).glob("traj_*.jsonl"))
        assert ta and [x.name for x in ta] == [x.name for x in tb]
        for xa, xb in zip(ta, tb):
            assert xa.read_bytes() == xb.read_bytes(), xa.name
    finally:
        parent.unlink(missing_ok=True)
        for dd in (a, b):
            shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)
