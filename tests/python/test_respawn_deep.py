"""--respawn-snap / --respawn-deep (2026-10-20, autoresearch r2b4).

The reservoir harvests one snapshot per second of an episode and only those >= --respawn-margin
before its end, then draws spawns uniformly. On sidistic's incumbent (r1b1, 5.1B steps) that
left 3 of 20,000 reservoir states past 27% progress while every greedy episode dies at
27.5-28.6%: the place the run is stuck is never a spawn. Two generic knobs:

* --respawn-snap S: seconds between two snapshots of one episode (absent = the shipped 1 s,
  0.25 s under --goals);
* --respawn-deep F (+ --respawn-deep-q Q): F of the reservoir draws come from its deepest Q
  quantile by goal-field distance, the rest uniform (Go-Explore's return to the frontier over
  the policy's own harvested states).

(a) unit: the weights realise the requested shell share, ignore non-finite distances and are
    uniform at share 0; the snap helper keeps the shipped cadence when the flag is absent;
    a RespawnBuffer drawing through the weights lands the shell share it was asked for.
(b) CPU smoke on sidistic: both flags run, the config carries them, a flagless resume restores
    them; bad values are refused.
(c) flag off: the trainer before this change and after it give the same config, progress.csv
    (fps aside), eval trajectory bytes and weights on the same smoke.
"""
from __future__ import annotations

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
TRAIN = ROOT / "python" / "train_fast.py"
SID = ROOT / "maps" / "surf_src_sidistic.bsp"
SID_FIELD = ROOT / "maps" / "surf_src_sidistic.goal_32.npz"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_run = pytest.mark.skipif(
    not (SID.exists() and SID_FIELD.exists() and DLL.exists()),
    reason="needs the built core + sidistic and its prebaked goal field")

SMOKE = ["--map", str(SID), "--reward", "race", "--envs", "64",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
         "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2", "--vf", "0.5",
         "--ent", "0.005", "--n-steps", "8", "--epochs", "1",
         "--minibatches", "2", "--ep-ticks", "3000", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "4000", "--train-stride", "1",
         "--yaw-adaptive", "--respawn-frac", "0.9", "--respawn-margin", "2",
         "--respawn-reservoir", "1000", "--int-coef", "0.25", "--int-view", "8",
         "--int-speed", "3", "--ckpt-every", "1e9", "--record-every", "1e12",
         "--eval-eps", "1", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold"]


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout, encoding="utf-8",
                          errors="replace")


def _train(run, extra, script=TRAIN, steps="6144"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


# -- (a) unit ------------------------------------------------------------------------------

def test_deep_shell_weights_realise_the_share():
    from train_fast import deep_shell_weights
    d = np.arange(1000, dtype=np.float64)[::-1].copy()      # deepest = smallest d, at the end
    w = deep_shell_weights(d, 0.5, 0.1)
    assert w.shape == (1000,) and abs(w.sum() - 1.0) < 1e-12
    shell = d < 100.0                                         # the 100 deepest
    assert abs(w[shell].sum() - (0.5 + 0.5 * 0.1)) < 1e-12
    assert np.allclose(w[shell], w[shell][0]) and np.allclose(w[~shell], w[~shell][0])
    # share 0 is the uniform draw
    assert np.allclose(deep_shell_weights(d, 0.0, 0.1), 1.0 / 1000)
    # non-finite distances never enter the shell; the quantile is over the finite ones
    d2 = d.copy()
    d2[:500] = np.nan
    w2 = deep_shell_weights(d2, 0.5, 0.1)
    assert abs(w2.sum() - 1.0) < 1e-12
    sh2 = np.isfinite(d2) & (d2 < 50.0)                       # 10% of the 500 finite ones
    assert sh2.sum() == 50
    assert abs(w2[sh2].sum() - (0.5 + 0.5 * 50 / 1000)) < 1e-12
    assert np.all(w2[~np.isfinite(d2)] < w2[sh2][0])


def test_snap_helper_keeps_the_shipped_cadence():
    from types import SimpleNamespace
    from train_fast import _respawn_snap_secs
    assert _respawn_snap_secs(SimpleNamespace(respawn_snap=None, goals=None)) == 1.0
    assert _respawn_snap_secs(SimpleNamespace(respawn_snap=None, goals="x")) == 0.25
    assert _respawn_snap_secs(SimpleNamespace(respawn_snap=0.25, goals=None)) == 0.25
    assert _respawn_snap_secs(SimpleNamespace(goals=None)) == 1.0


def test_reservoir_draw_through_the_weights_lands_the_shell_share():
    from surfgym.respawn import RespawnBuffer, STATE_DTYPE
    from train_fast import deep_shell_weights
    rb = RespawnBuffer(4, reservoir=2000, margin_ticks=0, snap_every=1, seed=5)
    st = np.zeros(2000, STATE_DTYPE)
    st["origin"][:, 0] = np.arange(2000, dtype=np.float32)    # "distance" = x
    rb.push_many(st)
    assert rb.size == 2000
    rb.weight_fn = lambda o: deep_shell_weights(o[:, 0].astype(np.float64), 0.6, 0.05)
    start = np.zeros(8, STATE_DTYPE)
    start["origin"][:, 0] = -1.0
    pool = rb.build_pool(start, pool_size=20000, fresh_frac=0.10)
    re = pool[pool["origin"][:, 0] >= 0.0]
    got = float(np.mean(re["origin"][:, 0] < 100.0))          # the 5% deepest = x < 100
    assert abs(got - (0.6 + 0.4 * 0.05)) < 0.02, got


# -- (b) smoke -----------------------------------------------------------------------------

@needs_run
def test_snap_and_deep_run_and_a_flagless_resume_restores_them():
    """A 4 s episode cap and a 0.1 s margin fill the reservoir inside the smoke, so the deep
    draw is live; the config carries both keys and a flagless resume restores them."""
    run = "rdeep_on"
    r = _train(run, ["--respawn-snap", "0.25", "--respawn-deep", "0.5",
                     "--respawn-deep-q", "0.1", "--ep-ticks", "400",
                     "--respawn-margin", "0.1"], steps="40960")
    assert "--respawn-deep 0.5" in r.stdout and "--respawn-snap 0.25" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["respawn_snap"] == 0.25 and cfg["respawn_deep"] == 0.5
    assert cfg["respawn_deep_q"] == 0.1
    run2 = "rdeep_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    for k in ("respawn_snap=0.25", "respawn_deep=0.5", "respawn_deep_q=0.1"):
        assert k in r2.stdout, k
    bad = _run([sys.executable, "-u", str(TRAIN), "--run", "rdeep_bad"]
               + SMOKE + ["--steps", "2048", "--respawn-deep", "1.5"])
    assert bad.returncode != 0 and "--respawn-deep is a share" in (bad.stdout + bad.stderr)
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / "rdeep_bad"):
        shutil.rmtree(dd, ignore_errors=True)


# -- (c) flag off --------------------------------------------------------------------------

def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200",
                        "HEAD"], capture_output=True, text=True, cwd=str(ROOT),
                       timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--respawn-deep" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_rdeep.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("rdeep_off_new", [])
        _train("rdeep_off_old", [], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "rdeep_off_new", ROOT / "runs" / "rdeep_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and "respawn_deep" not in ca and "respawn_snap" not in ca
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
