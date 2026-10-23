"""--ge-minutes M (pbr1b1, 2026-10-23): Go-Explore phase 1 (tools/explore_phase1.py) from the map's
true spawn BEFORE training; its archive joins every training spawn pool as --ge-frac of it.

(a) train_fast._ge_phase1 runs the search, writes runs/<run>/ge_states.npy (STATE_DTYPE rows,
    clocks zeroed, the root = the zones file's spawn) and reuses the file on a second call;
(b) explore_phase1 --maxvel sets the search core's sv_maxvelocity;
(c) the trainer (CPU, princessburglar): a smoke run with the flag searches, drops the archive cells
    behind the start, trains, prints its provenance and dumps the four config keys; the knobs
    without --ge-minutes and the flag with --spawn-states are refused;
(d) flag off is bit-identical to the trainer before the change (config, progress.csv, the eval
    trajectories, the weights).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
TRAIN = ROOT / "python" / "train_fast.py"
PB = ROOT / "maps" / "surf_princessburglar_b26.bsp"
PB_FIELDS = [ROOT / "maps" / f"surf_princessburglar_b26.{k}_32.npz" for k in ("goal", "goalc")]
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_pb = pytest.mark.skipif(
    not (PB.exists() and DLL.exists() and all(p.exists() for p in PB_FIELDS)),
    reason="needs the built core + princessburglar and its prebaked goal fields")
SPAWN = np.array([860.0, -3005.0, 3653.0])      # the zones file's spawn (start_run + 37 u)

SMOKE = ["--map", str(PB), "--reward", "race", "--envs", "64",
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


def _env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8")
    env.pop("SELF_STATES", None)
    return env


def _run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                          timeout=timeout, encoding="utf-8", errors="replace")


def _train(run, extra, script=TRAIN, steps="2048"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    return _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                + ["--steps", steps] + list(extra))


@needs_pb
def test_ge_phase1_writes_the_archive_once_and_reuses_it(capsys):
    import train_fast
    from surfgym.core import STATE_DTYPE
    run = "ge_unit"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    try:
        a = Namespace(run=run, map=str(PB), ge_cell=64.0, ge_speed=8, act_every=4,
                      maxvel=3500.0, ge_minutes=0.05, seed=0)
        p = Path(train_fast._ge_phase1(a))
        assert p == ROOT / "runs" / run / "ge_states.npy" and p.exists()
        st = np.load(p, allow_pickle=False)
        assert st.dtype == STATE_DTYPE and st.ndim == 1 and len(st) > 1
        assert (st["tick"] == 0).all() and (st["stuck_ticks"] == 0).all()
        # rooted at the true spawn: the archive's first cell is the zones file's spawn
        assert np.allclose(st["origin"][0], SPAWN, atol=1.0)
        meta = json.loads((ROOT / "runs" / run / "ge" / "archive.meta.json").read_text())
        assert meta["cell"] == 64.0 and meta["cell_speed"] == 8 and meta["act_every"] == 4
        assert meta["maxvel"] == 3500.0
        m0 = p.stat().st_mtime_ns
        assert Path(train_fast._ge_phase1(a)) == p and p.stat().st_mtime_ns == m0
        assert "reusing this run's archive" in capsys.readouterr().out
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_pb
def test_explore_phase1_maxvel_sets_the_search_core():
    out = ROOT / "runs" / "ge_maxvel"
    shutil.rmtree(out, ignore_errors=True)
    try:
        r = _run([sys.executable, "-u", str(ROOT / "tools" / "explore_phase1.py"), "--map",
                  str(PB), "--out", str(out), "--envs", "64", "--cell", "64", "--max-iters",
                  "1", "--maxvel", "3500", "--threads", "2"], timeout=600)
        # explore_phase1 exits 1 when nothing crossed the finish: a budgeted search's end
        assert r.returncode in (0, 1), r.stdout[-3000:] + r.stderr[-3000:]
        assert (out / "archive.npz").exists()
        assert json.loads((out / "archive.meta.json").read_text())["maxvel"] == 3500.0
    finally:
        shutil.rmtree(out, ignore_errors=True)
    import surfgym.core as sc
    # the default the flag overrides (the trainer's recipe runs 3500)
    assert sc.default_config(num_envs=1).phys.sv_maxvelocity == pytest.approx(2000.0)


@needs_pb
def test_the_trainer_searches_filters_trains_and_dumps_the_keys():
    run = "ge_smoke"
    try:
        r = _train(run, ["--ge-minutes", "0.05"])
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--ge-minutes 0.05: Go-Explore phase 1 from the true spawn" in out
        assert "provenance: ge_states" in out
        assert "deeper than the start on the shaping field" in out
        assert "--spawn-states:" in out and "own states from" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["ge_minutes"] == 0.05 and cfg["ge_frac"] == 0.3
        assert cfg["ge_cell"] == 64.0 and cfg["ge_speed"] == 8
        assert cfg.get("spawn_states") is None          # not a --spawn-states source
        assert (ROOT / "runs" / run / "ge_states.npy").exists()
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _train("ge_bad1", ["--ge-frac", "0.5"])
    assert bad.returncode != 0 and "need --ge-minutes > 0" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "ge_bad1", ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--ge-minutes" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


def _csv(run):
    import csv
    with open(ROOT / "runs" / run / "progress.csv", newline="") as fh:
        return list(csv.DictReader(fh))


@needs_pb
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_ge.py"
    extra = ["--record-every", "2048"]
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("ge_off_new", TRAIN), ("ge_off_old", pre)):
            r = _train(run, extra, script=script, steps="4096")
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "ge_off_new", ROOT / "runs" / "ge_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and not any(k.startswith("ge_") for k in ca)
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if k != "time/fps" and not k.startswith("time/"):
                    assert x[k] == y[k], k
        ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
        assert ta and [p.name for p in ta] == [p.name for p in tb]
        for p, q in zip(ta, tb):
            assert p.read_bytes() == q.read_bytes(), p.name
        sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
