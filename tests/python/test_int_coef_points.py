"""--int-coef-progress as a POINT LIST (ab/gate, 2026-11-15): 'P0:C0,P1:C1,...', linear between
the points, flat before the first and beyond the last; the HI:LO:P form is read as before.

Why: one schedule has to serve maps whose stuck zones sit at different frontier progress
(celestial's fork at ~25-35%, cannonball's wall zone at ~85-97%), and HI:LO:P has one knee.

1. The pieces: both forms parse (the old one to the same tuple, through the same int_coef_at),
   the point list's checks, its interpolation, its start value, and '0:HI,P:LO' equal to HI:LO:P
   bit for bit.
2. record_ckpt: the key stays TRAIN_ONLY, and a point list in a checkpoint's config passes its
   audit.
3. CPU runs (the built core + a map with its prebaked goal field; INTP_TEST_MAP / SURFCORE_DLL
   point at them): the OLD form ON is bit-identical to the parent commit; a point list reads
   every greedy eval, writes its points to run.json, checkpoints the coefficient, and a resume
   restores it.
"""
from __future__ import annotations

import json
import os
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


# --- 1. the pieces --------------------------------------------------------------------------------

def test_the_old_form_parses_as_before():
    from train_fast import ProgressPoints, parse_int_coef_progress
    v = parse_int_coef_progress("2.5:0.25:0.1")
    assert v == (2.5, 0.25, 0.1) and type(v) is tuple and not isinstance(v, ProgressPoints)
    for bad in ("2.5:0.25", "a:b:c", "2.5:0.25:0", "2.5:0.25:1.5", "-1:0.25:0.1",
                "2.5:-0.1:0.1"):
        with pytest.raises(SystemExit):
            parse_int_coef_progress(bad)


def test_the_point_list_parses_and_bad_ones_are_refused():
    from train_fast import ProgressPoints, parse_int_coef_progress
    v = parse_int_coef_progress("0:2.5,0.1:0.25,0.5:0.25,0.9:0.05")
    assert isinstance(v, ProgressPoints)
    assert tuple(v) == ((0.0, 2.5), (0.1, 0.25), (0.5, 0.25), (0.9, 0.05))
    assert v.config() == [[0.0, 2.5], [0.1, 0.25], [0.5, 0.25], [0.9, 0.05]]
    assert v.text() == "0:2.5,0.1:0.25,0.5:0.25,0.9:0.05"
    assert tuple(parse_int_coef_progress("0.2:0.3")) == ((0.2, 0.3),)       # one point: flat
    assert tuple(parse_int_coef_progress(" 0:1, 1:0 ")) == ((0.0, 1.0), (1.0, 0.0))
    for bad in ("0:2.5,0.1", "0:2.5,0.1:0.25,0.1:0.3", "0:2.5,0.5:1,0.2:1", "0:2.5,1.5:0.1",
                "-0.1:1,0.5:1", "0:-1", "0:2.5,,0.1:1", "0:inf", "0:nan", "nan:1",
                "0:1:2,0.5:1", "x:1,0.5:1"):
        with pytest.raises(SystemExit):
            parse_int_coef_progress(bad)


def test_the_interpolation():
    from train_fast import int_coef_progress_at, parse_int_coef_progress
    v = parse_int_coef_progress("0:2.5,0.1:0.25,0.5:0.25,0.9:0.05")
    at = lambda x: int_coef_progress_at(x, v)                                # noqa: E731
    assert at(float("nan")) == 2.5                         # no reading yet = progress 0
    assert at(-0.08) == 2.5                                # a dip below the spawn's d
    assert at(0.0) == 2.5
    assert at(0.05) == pytest.approx(1.375)
    assert at(0.1) == 0.25 and at(0.3) == 0.25 and at(0.5) == 0.25
    assert at(0.7) == pytest.approx(0.15)
    assert at(0.9) == 0.05 and at(0.967) == 0.05 and at(1.0) == 0.05 and at(1.2) == 0.05
    w = parse_int_coef_progress("0.2:0.3,0.6:1.5")         # flat BEFORE the first point too
    assert int_coef_progress_at(0.0, w) == 0.3 and int_coef_progress_at(float("nan"), w) == 0.3
    assert int_coef_progress_at(0.4, w) == pytest.approx(0.9)
    assert int_coef_progress_at(0.9, w) == 1.5


def test_the_old_form_is_the_two_point_line_bit_for_bit():
    from train_fast import int_coef_at, int_coef_progress_at, parse_int_coef_progress
    old = parse_int_coef_progress("2.5:0.25:0.1")
    two = parse_int_coef_progress("0:2.5,0.1:0.25")
    for x in list(np.linspace(-0.3, 1.3, 20001)) + [float("nan")]:
        ref = int_coef_at(float(x), 2.5, 0.25, 0.1)
        assert int_coef_progress_at(float(x), old) == ref          # the old form, same call
        assert int_coef_progress_at(float(x), two) == ref          # and the same line


def test_the_start_value():
    from train_fast import int_coef_progress_start, parse_int_coef_progress
    assert int_coef_progress_start(parse_int_coef_progress("2.5:0.25:0.1")) == 2.5
    assert int_coef_progress_start(parse_int_coef_progress("0:2.5,0.1:0.25")) == 2.5
    assert int_coef_progress_start(parse_int_coef_progress("0.2:0.3,0.6:1.5")) == 0.3


# --- 2. record_ckpt ------------------------------------------------------------------------------

def test_record_ckpt_still_treats_it_as_train_only():
    import record_ckpt
    assert "int_coef_progress" in record_ckpt.TRAIN_ONLY
    record_ckpt.audit_cfg({"int_coef_progress": [[0.0, 2.5], [0.1, 0.25], [0.9, 0.05]]})
    record_ckpt.audit_cfg({"int_coef_progress": [2.5, 0.25, 0.1]})


def test_the_trainer_wires_both_forms():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert 'meta["config"]["int_coef_progress"] = INTP.config()' in src
    assert 'meta["config"]["int_coef_progress"] = [float(v) for v in INTP]' in src
    assert '"coef": None if INTP is None else int_coef_progress_start(INTP)' in src
    assert "c = int_coef_progress_at(prog, INTP)" in src
    assert "int_coef_at(prog, *INTP)" not in src


# --- 3. CPU runs ----------------------------------------------------------------------------------

MAP = Path(os.environ.get("INTP_TEST_MAP", str(ROOT / "maps" / "surf_unitfarmer2.bsp")))
DLL = Path(os.environ.get("SURFCORE_DLL", str(
    ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))))
needs_run = pytest.mark.skipif(
    not (MAP.exists() and DLL.exists() and MAP.with_suffix(".goal_32.npz").exists()),
    reason="needs the built core + a map with its prebaked goal field (INTP_TEST_MAP, "
           "SURFCORE_DLL)")
SMOKE = ["--map", str(MAP), "--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--lidar-range", "11500",
         "--lidar-near", "2000", "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4", "--gamma", "0.9995",
         "--gae", "0.95", "--clip", "0.2", "--vf", "0.5", "--ent", "0.005", "--n-steps", "8",
         "--epochs", "1", "--minibatches", "2", "--ep-ticks", "600", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "3500", "--train-stride", "1", "--yaw-adaptive",
         "--respawn-frac", "0.9", "--respawn-margin", "10", "--respawn-reservoir", "1000",
         "--int-coef", "2.5", "--int-view", "0", "--int-speed", "8", "--ckpt-every", "1e9",
         "--eval-eps", "3", "--eval-greedy-only", "--unstuck", "--unstuck-max", "0.5",
         "--unstuck-patience", "4096", "--unstuck-period", "4096", "--no-compile", "--fp32",
         "--seed", "0"]
POINTS = "0:2.5,0.0001:0.6,0.0005:0.3,0.9:0.05"   # knees a 6k-step smoke run's frontier crosses


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
    """<tmp>/base/python as it was before the point list: the parent of the first commit that
    defines ProgressPoints (GATE_BASE_REF overrides)."""
    ref = os.environ.get("GATE_BASE_REF")
    if not ref:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-S",
                            "class ProgressPoints", "--", "python/train_fast.py"],
                           capture_output=True, text=True)
        shas = r.stdout.split()
        if r.returncode != 0 or not shas:
            pytest.skip("the point list is not committed yet; set GATE_BASE_REF")
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
    src = (base / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert "class ProgressPoints" not in src and "--int-coef-progress" in src
    return base


def _rows(path, drop=("time/fps",)):
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    keep = [i for i, h in enumerate(head) if h not in drop]
    return [[r.split(",")[i] for i in keep] for r in lines[1:]]


@needs_run
def test_the_old_form_on_is_bit_identical_to_the_parent_commit(tmp_path):
    base = _base_tree(tmp_path)
    extra = ["--steps", "6144", "--record-every", "2048", "--int-coef-progress", "2.5:0.25:0.1"]
    r_old = _train(base, "gate_old", extra)
    assert r_old.returncode == 0, r_old.stdout[-3000:] + r_old.stderr[-3000:]
    r_new = _train(ROOT, "gate_old", extra)
    assert r_new.returncode == 0, r_new.stdout[-3000:] + r_new.stderr[-3000:]
    d_old, d_new = base / "runs" / "gate_old", ROOT / "runs" / "gate_old"
    c_old = json.loads((d_old / "run.json").read_text(encoding="utf-8"))["config"]
    c_new = json.loads((d_new / "run.json").read_text(encoding="utf-8"))["config"]
    assert c_new["int_coef_progress"] == [2.5, 0.25, 0.1] and c_new == c_old
    t_old, t_new = sorted(d_old.glob("traj_*.jsonl")), sorted(d_new.glob("traj_*.jsonl"))
    assert len(t_old) >= 3 and [p.name for p in t_old] == [p.name for p in t_new]
    for a, b in zip(t_old, t_new):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs"
    assert _rows(d_old / "progress.csv") == _rows(d_new / "progress.csv")
    reads = lambda s: [ln for ln in s.splitlines() if "--int-coef-progress: greedy" in ln]  # noqa
    assert reads(r_old.stdout) == reads(r_new.stdout) and len(reads(r_new.stdout)) >= 3
    shutil.rmtree(d_new, ignore_errors=True)


@needs_run
def test_a_point_list_reads_every_eval_checkpoints_and_restores():
    from train_fast import int_coef_progress_at, parse_int_coef_progress
    pts = parse_int_coef_progress(POINTS)
    r = _train(ROOT, "gate_pts", ["--steps", "6144", "--record-every", "2048",
                                  "--int-coef-progress", POINTS])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert f"--int-coef-progress {pts.text()}: the novelty coefficient" in r.stdout
    reads = [ln for ln in r.stdout.splitlines() if "--int-coef-progress: greedy frontier" in ln]
    assert len(reads) >= 3, r.stdout[-3000:]
    coefs = []
    for ln in reads:
        prog = float(ln.split("greedy frontier ")[1].split("%")[0]) / 100.0
        coef = float(ln.split("novelty coefficient ")[1].split()[0])
        # the log prints 2 decimals of the progress and 4 significant digits of the coefficient
        lo, hi = int_coef_progress_at(prog + 5e-5, pts), int_coef_progress_at(prog - 5e-5, pts)
        assert min(lo, hi) * (1 - 1e-3) <= coef <= max(lo, hi) * (1 + 1e-3), ln
        coefs.append(coef)
    d = ROOT / "runs" / "gate_pts"
    c = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert c["int_coef_progress"] == pts.config() and c["int_coef"] == 2.5
    import torch
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    st = ck["int_coef_progress"]
    assert st["coef"] == pytest.approx(coefs[-1], rel=1e-3)
    r2 = _train(ROOT, "gate_pts2", ["--steps", "8192", "--record-every", "1e12",
                                    "--no-eval-at-start", "--ckpt", str(d / "ckpt_final.pt"),
                                    "--int-coef-progress", POINTS])
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert (f"--int-coef-progress: restored the novelty coefficient {st['coef']:.4g}"
            in r2.stdout), r2.stdout[-3000:]
    for run in ("gate_pts", "gate_pts2"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
