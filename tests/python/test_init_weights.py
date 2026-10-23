"""``--init-weights CKPT``: a FRESH run whose network starts from another run's weights.

The use (pbr1b4, 2026-10-23): the sidistic finisher's weights onto surf_princessburglar_b26.
A ``--ckpt`` resume onto another map would also restore that map's reservoir, its greedy-death
states, its novelty table and its --unstuck reading (the other map's frontier), so the flag
reads the policy's tensors and nothing else. What is pinned:

1. the weights ARE the checkpoint's: at --lr 0 the run's final weights equal the source's
   bit for bit, while the step counter is the new run's own (it starts at 0);
2. the provenance lands in run.json's config (path, md5, source step, source map), and the
   recorder accepts the key (TRAIN_ONLY);
3. an architecture mismatch and a --ckpt resume are refused with the reason;
4. no flag = no key in the config dump (the flag-off code path is the scratch path: the new
   block runs only when the flag is given).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/python/test_init_weights.py -q

CPU only. Uses surf_princessburglar_b26 (its fields are prebaked in maps/); set RL_SURF_MAPS
to a maps/ directory holding them when the worktree has none.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_DLL_NAME = "surfcore.dll" if os.name == "nt" else "libsurfcore.so"
DLL = Path(os.environ.get("SURFCORE_DLL") or (ROOT / "build" / _DLL_NAME))
STEM = "surf_princessburglar_b26"


def _maps_dir():
    env = os.environ.get("RL_SURF_MAPS")
    for d in ([Path(env)] if env else []) + [ROOT / "maps"]:
        if (d / f"{STEM}.bsp").exists() and (d / f"{STEM}.goalc_32.npz").exists():
            return d
    return None


MAPS = _maps_dir()
MAP = (MAPS or (ROOT / "maps")) / f"{STEM}.bsp"
needs_core = pytest.mark.skipif(not (MAP.exists() and DLL.exists() and MAPS is not None),
                                reason="needs the built core + the prebaked princessburglar fields")

SMOKE = ["--reward", "race", "--envs", "16", "--spawn", "platform",
         "--lidar-w", "16", "--lidar-h", "8", "--lidar-cell", "32",
         "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "32", "--hidden", "32",
         "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
         "--lr", "3e-4", "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2",
         "--vf", "0.5", "--ent", "0.005",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
         "--ep-ticks", "3000", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--race-no-corner-cut", "1", "--maxvel", "3500",
         "--train-stride", "1",
         "--respawn-frac", "0.9", "--respawn-margin", "10",
         "--respawn-reservoir", "1000", "--int-coef", "0.25",
         "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
         "--eval-eps", "1", "--eval-greedy-only", "--no-eval-at-start",
         "--record-every", "1e12", "--no-compile"]


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"),
             NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS", "2"),
             SURFCORE_DLL=str(DLL))
    if MAPS is not None:
        e["RL_SURF_MAPS"] = str(MAPS)
    return e


def _train(run, extra, check=True, timeout=1800):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
           "--map", str(MAP), "--run", run] + SMOKE + extra
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                       timeout=timeout, encoding="utf-8", errors="replace")
    if check:
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r, ROOT / "runs" / run


def _ck(run_dir):
    return torch.load(run_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)


@needs_core
def test_init_weights_loads_the_policy_and_nothing_else():
    try:
        _, src = _train("iw_src", ["--steps", "1024"])   # 2 iterations of 16 x 8 x 4
        a = _ck(src)
        assert a["global_step"] >= 1024
        assert "init_weights" not in json.loads(
            (src / "run.json").read_text(encoding="utf-8"))["config"]      # (4)

        # (1) at lr 0 the run's weights stay the source's, bit for bit; its own step counter
        r, dst = _train("iw_dst", ["--steps", "512", "--lr", "0",
                                   "--init-weights", str(src / "ckpt_final.pt")])
        assert "--init-weights" in r.stdout and "policy tensors loaded" in r.stdout
        b = _ck(dst)
        assert b["global_step"] < a["global_step"]
        for k, v in a["policy"].items():
            assert torch.equal(v, b["policy"][k]), k
        # (2) provenance in the config
        cfg = json.loads((dst / "run.json").read_text(encoding="utf-8"))["config"]
        iw = cfg["init_weights"]
        assert iw["path"].endswith("ckpt_final.pt") and len(iw["md5"]) == 32
        assert iw["step"] == a["global_step"] and iw["map"] == STEM
        sys.path.insert(0, str(ROOT / "tools"))
        import record_ckpt
        assert "init_weights" in record_ckpt.TRAIN_ONLY

        # (3) the refusals
        r, _ = _train("iw_bad", ["--steps", "128", "--hidden", "48",
                                 "--init-weights", str(src / "ckpt_final.pt")], check=False)
        assert r.returncode != 0
        assert "hidden" in (r.stdout + r.stderr)
        r, _ = _train("iw_both", ["--steps", "128", "--ckpt", str(src / "ckpt_final.pt"),
                                  "--init-weights", str(src / "ckpt_final.pt")], check=False)
        assert r.returncode != 0
        assert "pass one of them" in (r.stdout + r.stderr)
    finally:
        for run in ("iw_src", "iw_dst", "iw_bad", "iw_both"):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
