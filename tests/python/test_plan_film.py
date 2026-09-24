"""--plan-film: the plan fan gates the conv trunk (FiLM on the last conv block).

(a) the Policy: off builds nothing; on, every other tensor gets the same init as off and the
    zero-initialised gate makes the step-0 policy compute exactly the ungated function;
    gradients reach the gate.
(b) the trainer + recorder on the small maze (CPU): a --plan-film 1 run trains, dumps
    plan_film into its config, and record_ckpt.py rebuilds and records its checkpoint.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import train_fast as tf   # noqa: E402

LAB = ROOT / "maps_pool" / "labyrinth_left100.bsp"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_core = pytest.mark.skipif(not (DLL.exists() and LAB.exists()),
                                reason="needs the built core + maps_pool/labyrinth_left100")


def _policy(film, route_dim=27, seed=0):
    torch.manual_seed(seed)
    obs_dim = tf.N_SCALAR + route_dim + 16 * 8
    return tf.Policy(obs_dim, 16, 8, emb=32, hidden=32, route_dim=route_dim,
                     plan_film=film, film_dim=(27 if film else 0)), obs_dim


def test_off_is_the_old_policy_and_on_starts_identical():
    p0, obs_dim = _policy(0)
    p1, _ = _policy(1)
    assert p0.film is None and not any(k.startswith("film") for k in p0.state_dict())
    s0, s1 = p0.state_dict(), p1.state_dict()
    assert set(s1) - set(s0) == {"film.0.weight", "film.0.bias", "film.2.weight", "film.2.bias"}
    for k in s0:                                   # every other tensor: the same draw
        assert torch.equal(s0[k], s1[k]), k
    x = torch.randn(5, obs_dim)
    with torch.no_grad():
        a0, v0 = p0(x)
        a1, v1 = p1(x)
    assert torch.allclose(a0, a1, atol=1e-6) and torch.allclose(v0, v1, atol=1e-6)


def test_gradients_reach_the_gate():
    p1, obs_dim = _policy(1)
    x = torch.randn(7, obs_dim)
    a, v = p1(x)
    (a.pow(2).sum() + v.pow(2).sum()).backward()
    # the zero-init output layer gets a gradient at once; the hidden layer only after it moves
    assert p1.film[2].weight.grad is not None and p1.film[2].weight.grad.abs().sum() > 0


def test_refused_without_a_fan():
    with pytest.raises(SystemExit):
        torch.manual_seed(0)
        tf.Policy(tf.N_SCALAR + 16 * 8, 16, 8, emb=32, hidden=32, route_dim=0,
                  plan_film=1, film_dim=27)


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


FLAGS = ["--map", str(LAB), "--reward", "race", "--envs", "64", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32", "--goal-cell", "32",
         "--lidar-range", "11500", "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
         "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
         "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2", "--ep-ticks", "96",
         "--time-pen", "0.005", "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "30",
         "--maxvel", "4000", "--train-stride", "1", "--yaw-adaptive", "--respawn-frac", "0.9",
         "--respawn-margin", "0.1", "--respawn-reservoir", "1000", "--int-coef", "0.25",
         "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9", "--record-every", "4096",
         "--eval-eps", "2", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold",
         "--goals", "1", "--goal-obs", "fan", "--goal-planner", "bfs", "--goal-reward", "arc",
         "--goal-fan-offsets", "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0", "--plan-film", "1"]


@needs_core
def test_trainer_and_recorder_run_a_film_checkpoint():
    run = "pfilm_smoke"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run, "--steps", "12288"] + FLAGS,
                       capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg.get("plan_film") == 1
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sd = ck.get("policy", ck.get("model", {}))
    assert any(k.startswith("film.") for k in sd), list(sd)[:8]
    out = d / "rec.jsonl"
    r2 = subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"),
                         str(d / "ckpt_final.pt"), "--out", str(out), "--episodes", "1"],
                        capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                        timeout=900, encoding="utf-8", errors="replace")
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert out.exists() and out.stat().st_size > 0
    shutil.rmtree(d, ignore_errors=True)
