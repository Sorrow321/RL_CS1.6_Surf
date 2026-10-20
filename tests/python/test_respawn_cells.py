"""--respawn-cells (2026-10-20, autoresearch r2b4): Go-Explore's return over the reservoir.

A share of the reservoir draws goes through CELLS (a 128 u position cube x a velocity class:
speed bucket, heading octant, vertical sign), drawn with weight 1/sqrt(1 + harvested + chosen),
then a uniform member state. Rare cells - the bottom of a detour the policy reached a few times -
get many returns; the mastered line few.

(a) unit: rare cells are preferred in proportion to the weights; every drawn index belongs to
    a drawn cell; the counters update; state_dict round trip; RespawnBuffer realises the share
    and checkpoints the counters.
(b) CPU smoke on sidistic: the flag runs, logs, the config carries it, a flagless resume
    restores it; bad values are refused.
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

def _rows(xs, vx=1000.0):
    from surfgym.respawn import STATE_DTYPE
    st = np.zeros(len(xs), STATE_DTYPE)
    st["origin"][:, 0] = np.asarray(xs, np.float32)
    st["velocity"][:, 0] = vx
    return st


def test_cell_weights_prefer_rare_cells_and_draw_members():
    from surfgym.respawn import CellReturn
    cr = CellReturn(0.5, 128.0, seed=3)
    common = _rows(np.full(900, 10.0))            # one cell, 900 states
    rare = _rows(np.arange(100) * 256.0 + 1000.0)  # 100 cells, one state each
    store = np.concatenate([common, rare])
    cr.note_pushed(store)
    rng = np.random.default_rng(0)
    idx = cr.draw(store, len(store), 20000, rng)
    assert idx.min() >= 0 and idx.max() < len(store)
    share_common = float(np.mean(idx < 900))
    # weights: common cell 1/sqrt(1+900) vs 100 rare cells 1/sqrt(2) each (chosen counts move
    # them during the draw only by this single batch): the common cell gets ~0.33%
    w_c, w_r = 1 / np.sqrt(901), 100 / np.sqrt(2)
    assert abs(share_common - w_c / (w_c + w_r)) < 0.01, share_common
    assert sum(cr.chosen.values()) == 20000
    # a cell is position cube x velocity class: same cube, opposite heading = two cells
    k = cr.keys(np.concatenate([_rows([10.0], 1000.0), _rows([10.0], -1000.0)]))
    assert k[0] != k[1]


def test_cell_state_dict_round_trip_and_size_guard():
    from surfgym.respawn import CellReturn
    a = CellReturn(0.4, 128.0)
    a.note_pushed(_rows([0.0, 0.0, 300.0]))
    a.draw(_rows([0.0, 300.0]), 2, 10, np.random.default_rng(1))
    b = CellReturn(0.4, 128.0)
    assert b.load_state_dict(a.state_dict())
    assert b.seen == a.seen and b.chosen == a.chosen
    assert not CellReturn(0.4, 64.0).load_state_dict(a.state_dict())


def test_reservoir_realises_the_cell_share_and_checkpoints_the_counters():
    from surfgym.respawn import CellReturn, RespawnBuffer
    rb = RespawnBuffer(4, reservoir=2000, margin_ticks=0, snap_every=1, seed=5)
    rb.cells = CellReturn(0.5, 128.0)
    store = np.concatenate([_rows(np.full(1900, 10.0)), _rows(np.arange(100) * 256.0 + 1000.0)])
    rb.push_many(store)
    assert sum(rb.cells.seen.values()) == 2000
    start = _rows([-5000.0] * 8)
    pool = rb.build_pool(start, pool_size=20000, fresh_frac=0.10)
    re = pool[pool["origin"][:, 0] > -4000.0]
    rare = float(np.mean(re["origin"][:, 0] >= 1000.0))
    # half the reservoir rows by cell (rare cells win almost all of it), half uniform (5% rare)
    assert abs(rare - (0.5 * 0.99 + 0.5 * 0.05)) < 0.03, rare
    d = rb.state_dict()
    assert "cells" in d
    rb2 = RespawnBuffer(4, reservoir=2000, margin_ticks=0, snap_every=1, seed=5)
    rb2.cells = CellReturn(0.5, 128.0)
    rb2.load_state_dict(d)
    assert rb2.cells.seen == rb.cells.seen and rb2.cells.chosen == rb.cells.chosen
    # off: no cells key, the uniform draw that shipped
    rb3 = RespawnBuffer(4, reservoir=2000, margin_ticks=0, snap_every=1, seed=5)
    rb3.push_many(store)
    assert "cells" not in rb3.state_dict()


# -- (b) smoke -----------------------------------------------------------------------------

@needs_run
def test_cells_run_log_and_a_flagless_resume_restores_them():
    run = "rcells_on"
    r = _train(run, ["--respawn-cells", "0.5", "--respawn-snap", "0.25", "--ep-ticks", "400",
                     "--respawn-margin", "0.1"], steps="40960")
    assert "--respawn-cells 0.5" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["respawn_cells"] == 0.5 and cfg["respawn_cells_size"] == 128.0
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert "cells" in ck["respawn"] and len(ck["respawn"]["cells"]["keys"]) > 0
    run2 = "rcells_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "respawn_cells=0.5" in r2.stdout and "--respawn-cells 0.5" in r2.stdout
    bad = _run([sys.executable, "-u", str(TRAIN), "--run", "rcells_bad"]
               + SMOKE + ["--steps", "2048", "--respawn-cells", "0.5", "--respawn-deep", "0.5"])
    assert bad.returncode != 0 and "pick one" in (bad.stdout + bad.stderr)
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / "rcells_bad"):
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
        if s.returncode == 0 and b"--respawn-cells" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_rcells.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("rcells_off_new", [])
        _train("rcells_off_old", [], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "rcells_off_new", ROOT / "runs" / "rcells_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and "respawn_cells" not in ca
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
