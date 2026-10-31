"""--energy-pot-rel KAPPA (zlr1b3, 2026-10-31): with --energy-pot on, the per-call energy term's K
is replaced on every training slot by KAPPA x 1000 x the slot's progress scale (100 / d0 x
--race-shaping), so one unit of energy height h = z + |v|^2 / (2 g) is shaped like KAPPA units
of geodesic progress on any map, instead of a fixed K whose weight against the progress term
depends on the map's size (K 3: 0.19 of a progress unit on surf_za_loop, 0.92 on unitfarmer2).

(a) The trainer (CPU smoke on surf_za_loop): K = KAPPA x 1000 x 100 / d0 is printed, the config
    records energy_pot_rel, and a flagless resume restores it.
(b) It is refused without --energy-pot on, and with a negative KAPPA.
(c) The recorder lists it as training-only (a reward term).
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
sys.path.insert(0, str(ROOT / "tools"))
TRAIN = ROOT / "python" / "train_fast.py"
MAP = ROOT / "maps" / "surf_za_loop.bsp"
FIELDC = ROOT / "maps" / "surf_za_loop.goalc_32.npz"
DLL = ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so")
needs_run = pytest.mark.skipif(not (MAP.exists() and FIELDC.exists() and DLL.exists()),
                               reason="needs the built core + surf_za_loop and its goalc field")

SMOKE = ["--map", str(MAP), "--reward", "race", "--envs", "16", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--lidar-range", "11500",
         "--lidar-near", "2000", "--emb", "32", "--hidden", "32", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--n-steps", "8", "--epochs", "1",
         "--minibatches", "2", "--ep-ticks", "3000", "--time-pen", "0.001",
         "--success-bonus", "50", "--stall-secs", "15", "--race-dist", "geodesic",
         "--race-no-corner-cut", "1", "--race-ratchet", "--maxvel", "3500", "--tick-ms", "7",
         "--yaw-adaptive", "--ckpt-every", "1e9", "--record-every", "1e12", "--eval-eps", "1",
         "--eval-greedy-only", "--seed", "7", "--view-continuous", "--view-absolute",
         "velocity", "--keys-hold"]


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")


def _run(cmd, timeout=1200):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                          timeout=timeout, encoding="utf-8", errors="replace")


@needs_run
def test_k_is_kappa_progress_units_and_a_resume_restores_it():
    run = "eprel_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "1024", "--energy-pot", "3", "--energy-pot-rel", "1"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    m = re.search(r"--energy-pot-rel 1: energy K = ([0-9.]+) on .*\(scale ([0-9.e-]+)/u", r.stdout)
    assert m, r.stdout[-3000:]
    k, scale = float(m.group(1)), float(m.group(2))
    assert k == pytest.approx(1000.0 * scale, rel=1e-3)
    d0 = float(re.search(r"d0 = ([0-9,]+)u", r.stdout).group(1).replace(",", ""))
    assert scale == pytest.approx(100.0 / d0, rel=2e-3)
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["energy_pot_rel"] == 1.0 and cfg["energy_pot"] == 3.0
    run2 = "eprel_smoke_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    ck = d / "ckpt_final.pt"
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt", str(ck)] + SMOKE
              + ["--steps", "2048"])
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "energy_pot_rel=1" in r2.stdout
    assert "--energy-pot-rel 1: energy K" in r2.stdout
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
def test_refused_without_energy_pot_or_negative():
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "eprel_bad"] + SMOKE
             + ["--steps", "1024", "--energy-pot-rel", "1"])
    assert r.returncode != 0 and "needs --energy-pot K > 0" in (r.stdout + r.stderr)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "eprel_bad"] + SMOKE
             + ["--steps", "1024", "--energy-pot", "3", "--energy-pot-rel", "-1"])
    assert r.returncode != 0 and "KAPPA >= 0" in (r.stdout + r.stderr)
    shutil.rmtree(ROOT / "runs" / "eprel_bad", ignore_errors=True)


def test_record_ckpt_lists_it_as_training_only():
    import record_ckpt
    assert "energy_pot_rel" in record_ckpt.TRAIN_ONLY
