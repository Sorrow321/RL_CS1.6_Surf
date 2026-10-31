"""--auto-bc (uf2r2b2, 2026-10-31): the trainer runs the expert loop's search half
(tools/auto_bc.py) beside the training and swaps a faster line's BC rows + spine in.

* the tool's pure rules (its selftest);
* the trainer's refusals (CPU, surf_unitfarmer2 smoke);
* flag off: the trainer is the trainer before the change (config + progress.csv);
* end to end on the round-1 incumbent (CPU resume, 16 envs): a stand-in search tool
  (AUTO_BC_TOOL) hands back a line faster than the BC file's own, the trainer swaps it in.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "python" / "train_fast.py"
UF2 = ROOT / "maps" / "surf_unitfarmer2.bsp"
UF2_FIELDS = [ROOT / "maps" / f"surf_unitfarmer2.{k}_32.npz" for k in ("goal", "goalc")]
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_uf2 = pytest.mark.skipif(
    not (UF2.exists() and DLL.exists() and all(p.exists() for p in UF2_FIELDS)),
    reason="needs the built core + surf_unitfarmer2 and its prebaked goal fields")
AR_HOME = Path(os.environ.get("AR_HOME", "/home/steam/ar_uf2"))
INC = AR_HOME / "incumbent" / "ckpt.pt"
POOL = AR_HOME / "runs" / "uf2r2b2" / "sf_c3"
needs_inc = pytest.mark.skipif(
    not (INC.exists() and (POOL / "bc.npz").exists() and (POOL / "spine.npy").exists()),
    reason="needs the round-1 incumbent and the uf2r2b2 c3 BC pool")
BASE = "019b6848"      # the commit before the flag

SMOKE = ["--map", str(UF2), "--reward", "race", "--envs", "32",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
         "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2", "--vf", "0.5",
         "--ent", "0.005", "--n-steps", "8", "--epochs", "1",
         "--minibatches", "2", "--ep-ticks", "3000", "--time-pen", "0.001",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--race-no-corner-cut", "1", "--maxvel", "3500",
         "--train-stride", "1", "--yaw-adaptive", "--respawn-frac", "0.9",
         "--respawn-margin", "2", "--respawn-reservoir", "1000", "--ckpt-every", "1e9",
         "--record-every", "1e12", "--eval-eps", "2", "--eval-greedy-only",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold",
         "--no-compile", "--fp32", "--seed", "0"]


def _env(**kw):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8")
    env.pop("SELF_STATES", None)
    env.update(kw)
    return env


def _run(cmd, env=None, timeout=2400):
    return subprocess.run(cmd, capture_output=True, text=True, env=env or _env(), cwd=str(ROOT),
                          timeout=timeout, encoding="utf-8", errors="replace")


def _csv(run):
    with open(ROOT / "runs" / run / "progress.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_tool_selftest():
    r = _run([sys.executable, str(ROOT / "tools" / "auto_bc.py"), "--selftest"])
    assert r.returncode == 0 and "auto_bc selftest OK" in r.stdout, r.stdout + r.stderr


@needs_uf2
def test_trainer_refuses_auto_bc_without_its_files():
    run = "abc_bad"
    try:
        r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
                 + ["--steps", "2048", "--auto-bc"])
        out = r.stdout + r.stderr
        assert r.returncode != 0 and "--auto-bc needs --bc-file and --demo-file" in out, out[-3000:]
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_uf2
def test_flag_off_is_the_trainer_before_it():
    pre = ROOT / "python" / "_train_fast_pre_abc.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--auto-bc" not in src
    pre.write_bytes(src)
    a, b = "abc_off_new", "abc_off_old"
    try:
        for run, script in ((a, TRAIN), (b, pre)):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
            r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                     + ["--steps", "4096"])
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((ROOT / "runs" / a / "run.json").read_text())["config"]
        cb = json.loads((ROOT / "runs" / b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_bc" not in ca
        ra, rb = _csv(a), _csv(b)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        pre.unlink(missing_ok=True)
        for run in (a, b):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


FAKE = '''import argparse, json, sys
ap = argparse.ArgumentParser()
for k in ("--ckpt", "--map", "--out", "--waves", "--envs", "--score", "--keep", "--seed",
          "--max-ticks", "--timeout"):
    ap.add_argument(k)
a = ap.parse_args()
print("AUTO_BC " + json.dumps({"best_s": 1.0, "bc": %r, "spine": %r, "keys": %r,
                               "waves": []}))
'''


@needs_uf2
@needs_inc
def test_end_to_end_swap_on_the_incumbent(tmp_path):
    """A CPU resume of the incumbent with the c3 pool: the first greedy eval starts the search
    (a stand-in that hands back the same pool as a 1.0 s line), the next iteration swaps it in."""
    fake = tmp_path / "fake_auto_bc.py"
    fake.write_text(FAKE % (str(POOL / "bc.npz"), str(POOL / "spine.npy"),
                            str(POOL / "spine.keys.npy")))
    run = "abc_e2e"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    import torch
    step0 = int(torch.load(INC, map_location="cpu", weights_only=False)["global_step"])
    try:
        r = _run([sys.executable, "-u", str(TRAIN), "--ckpt", str(INC), "--run", run,
                  "--envs", "16", "--no-compile", "--fp32", "--steps", str(step0 + 3 * 8192),
                  "--record-every", "8192", "--eval-eps", "1", "--eval-greedy-only",
                  "--eval-patience", "3", "--ckpt-every", "1e12",
                  "--demo-file", str(POOL / "spine.npy"),
                  "--demo-keys", str(POOL / "spine.keys.npy"), "--demo-tau0", "0",
                  "--demo-milestones", "29584,27363,17276,13765,13232,8515,3476",
                  "--bc-file", str(POOL / "bc.npz"), "--bc-coef", "0.5",
                  "--bc-coef-final", "0.5", "--bc-target", "dist", "--bc-value-coef", "0",
                  "--bc-keys-engine", "--bc-pitch-coef", "0", "--bc-batch", "64",
                  "--lr", "3e-5", "--auto-bc", "--auto-bc-every", "0"],
                 env=_env(SELF_STATES="1", AUTO_BC_TOOL=str(fake)), timeout=5400)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-5000:]
        assert "--auto-bc: tools/auto_bc.py" in out, out[-3000:]
        assert "auto-bc search 1: running beside the training" in out, out[-3000:]
        assert "auto-bc search 1: NEW BC ROWS + SPINE" in out, out[-3000:]
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_bc"] == 1 and cfg["auto_bc_score"] == "dh"
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
