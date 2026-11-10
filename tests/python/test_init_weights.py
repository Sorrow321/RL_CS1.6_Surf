"""--init-weights: a FROM-SCRATCH run whose network starts as another run's.

Why the flag exists (the user, 2026-11-10: "launch A, but with network
initialization that came from sidistic map"): the second-map test runs one
frozen recipe on a new map, and the transfer arm asks what the first map's
network is worth there. A --ckpt resume is the wrong instrument for that -
it restores the source run's step counter, its novelty counts, its spawn
pools, its curriculum state and its spine path, so --auto-spine-scratch
would never write the new map's own spawn spine. --init-weights is a scratch
run in every respect but one: the policy weights, their Adam moments and
(under --ret-norm) the return normaliser are read from the checkpoint.

What this file pins:

1. The run starts at step 0, says so, never prints a "resumed" line, and its
   run.json is a scratch run's plus five provenance keys (file name, md5,
   step, map, spine file of the source).
2. The network IS the source's: with --lr 0 the final checkpoint's policy
   tensors equal the source checkpoint's bit for bit, and the optimizer's
   step count continues the source's (the Adam moments came over).
3. A run WITHOUT the flag has none of the keys (a control's config dump is
   unchanged).
4. Refusals: together with --ckpt; a demo_contaminated source (CLAUDE.md
   section 0); a network that does not fit; a setting the weights read or
   write through that differs from the source's (act_every here).
5. A crash-resume of such a run (plain --ckpt of its own checkpoint) keeps
   the provenance keys.
6. tools/record_ckpt.py declares the keys (its audit_cfg gate refuses a
   config key it never mentions, and the launcher's record gate would then
   kill the run).

    python -m pytest tests/python/test_init_weights.py -q

CPU only, tiny nets, a few thousand steps per run. The two maps default to
cannonball (source) and petrus_lite (target) in this checkout's maps/; set
INIT_W_SRC_MAP / INIT_W_DST_MAP to absolute .bsp paths that have their
prebaked fields beside them to run it elsewhere (a checkout without the
fields would bake for half an hour, so the file skips instead).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SRC_MAP = Path(os.environ.get("INIT_W_SRC_MAP",
                              ROOT / "maps" / "surf_src_cannonball.bsp"))
DST_MAP = Path(os.environ.get("INIT_W_DST_MAP",
                              ROOT / "maps" / "surf_petrus_lite.bsp"))
DLL = ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so")


def _field(bsp):
    return bsp.with_name(bsp.stem + ".goal_32.npz")


needs_run = pytest.mark.skipif(
    not (DLL.exists() and SRC_MAP.exists() and DST_MAP.exists()
         and _field(SRC_MAP).exists() and _field(DST_MAP).exists()),
    reason="needs the built core + two maps with their prebaked goal fields")

KEYS = ("init_weights", "init_weights_md5", "init_weights_step",
        "init_weights_map", "init_weights_demo_file")

SMOKE = ["--reward", "race", "--envs", "64",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500",
         "--lidar-near", "2000", "--emb", "64", "--hidden", "64",
         "--act-every", "4", "--pitch-rate", "1.33", "--teleport-fail",
         "--gamma", "0.9995", "--gae", "0.95",
         "--clip", "0.2", "--vf", "0.5", "--ent", "0.005",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
         "--ep-ticks", "3000", "--time-pen", "0.005",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "4000",
         "--train-stride", "1", "--yaw-adaptive",
         "--respawn-frac", "0.9", "--respawn-margin", "10",
         "--respawn-reservoir", "1000", "--int-coef", "0.25",
         "--int-view", "8", "--int-speed", "3", "--ckpt-every", "1e9",
         "--eval-eps", "1", "--eval-greedy-only",
         "--record-every", "1e12", "--no-eval-at-start"]

SRC_STEPS = 4096          # 2 iterations of 8 * 64 * 4 = 2,048 steps
DST_STEPS = 2048          # 1 iteration


def _env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES="-1",
                PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2",
                NUMBA_NUM_THREADS="2", SURFCORE_DLL=str(DLL))


def _train(run, map_path, extra, lr="3e-4", keep=False, timeout=1800):
    """Run python/train_fast.py; artifacts land in <ROOT>/runs/<run>."""
    if not keep:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    cmd = ([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
            "--run", run, "--map", str(map_path), "--lr", lr]
           + SMOKE + extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                          cwd=str(ROOT), timeout=timeout)


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json")
                      .read_text(encoding="utf-8"))["config"]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _out(r):
    return r.stdout[-6000:] + r.stderr[-6000:]


@pytest.fixture(scope="module")
def source():
    """A tiny scratch run on the source map; its final checkpoint is the
    network every other test starts from."""
    r = _train("iw_src", SRC_MAP, ["--steps", str(SRC_STEPS), "--seed", "1"])
    assert r.returncode == 0, _out(r)
    ck = ROOT / "runs" / "iw_src" / "ckpt_final.pt"
    assert ck.exists(), _out(r)
    return ck


# ==========================================================================
# 1-3. the run is a scratch run, the network is the source's
# ==========================================================================
@needs_run
def test_the_run_starts_at_step_0_with_the_source_network(source):
    import torch
    r = _train("iw_dst", DST_MAP,
               ["--steps", str(DST_STEPS), "--seed", "2",
                "--init-weights", str(source)], lr="0")
    assert r.returncode == 0, _out(r)
    log = r.stdout
    assert "--init-weights: the policy (" in log, _out(r)
    assert "step counter starts at 0" in log and "(not a resume)" in log
    assert "--init-weights: settings that differ from the source run's" in log
    assert not [ln for ln in log.splitlines() if ln.startswith("resumed ")], \
        "an --init-weights run must not take the resume path"

    src_cfg, dst_cfg = _cfg("iw_src"), _cfg("iw_dst")
    # 3. the control (no flag) has none of the keys
    assert not [k for k in src_cfg if k.startswith("init_weights")]
    # 1. a scratch run's config plus the five provenance keys, nothing else
    assert set(dst_cfg) - set(src_cfg) == set(KEYS)
    assert set(src_cfg) - set(dst_cfg) == set()
    assert dst_cfg["init_weights"] == source.name
    assert dst_cfg["init_weights_md5"] == _md5(source)
    assert dst_cfg["init_weights_step"] == SRC_STEPS
    assert dst_cfg["init_weights_map"] == src_cfg["map"]
    assert dst_cfg["init_weights_demo_file"] is None
    # the map is this run's own, not the source's
    assert Path(dst_cfg["map"]).stem == DST_MAP.stem != SRC_MAP.stem

    a = torch.load(source, map_location="cpu", weights_only=False)
    b = torch.load(ROOT / "runs" / "iw_dst" / "ckpt_final.pt",
                   map_location="cpu", weights_only=False)
    # the step counter started at 0 (a resume would say SRC + DST)
    assert a["global_step"] == SRC_STEPS and b["global_step"] == DST_STEPS
    # 2. --lr 0: the weights are the source's, bit for bit
    assert set(a["policy"]) == set(b["policy"])
    for k, v in a["policy"].items():
        assert torch.equal(v, b["policy"][k]), k
    # ... and the Adam state came with them: the step count continues
    sa = max(int(s["step"]) for s in a["optimizer"]["state"].values())
    sb = max(int(s["step"]) for s in b["optimizer"]["state"].values())
    assert sa > 0 and sb > sa, (sa, sb)
    # the checkpoint carries the provenance too
    assert all(b["config"][k] == dst_cfg[k] for k in KEYS)
    # nothing of the source's run state came over
    assert b["config"]["map"] == dst_cfg["map"]


@needs_run
def test_the_weights_then_train(source):
    """With a real learning rate the run trains on from the source network:
    it runs, counts from 0 and the weights MOVE (so test 2's equality is
    --lr 0's doing, not a frozen network)."""
    import torch
    r = _train("iw_dst_lr", DST_MAP,
               ["--steps", str(DST_STEPS), "--seed", "2",
                "--init-weights", str(source)])
    assert r.returncode == 0, _out(r)
    a = torch.load(source, map_location="cpu", weights_only=False)
    b = torch.load(ROOT / "runs" / "iw_dst_lr" / "ckpt_final.pt",
                   map_location="cpu", weights_only=False)
    assert b["global_step"] == DST_STEPS
    assert any(not torch.equal(v, b["policy"][k])
               for k, v in a["policy"].items())


# ==========================================================================
# 4. refusals
# ==========================================================================
@needs_run
def test_refused_together_with_ckpt(source):
    r = _train("iw_ref1", DST_MAP,
               ["--steps", str(DST_STEPS), "--init-weights", str(source),
                "--ckpt", str(source)])
    assert r.returncode != 0
    assert "Pass one of them" in _out(r)


@needs_run
def test_a_demo_contaminated_source_is_refused(source, tmp_path):
    import torch
    ck = torch.load(source, map_location="cpu", weights_only=False)
    ck["config"] = dict(ck["config"], demo_contaminated=True)
    bad = tmp_path / "contaminated.pt"
    torch.save(ck, bad)
    r = _train("iw_ref2", DST_MAP,
               ["--steps", str(DST_STEPS), "--init-weights", str(bad)])
    assert r.returncode != 0
    assert "demo_contaminated" in _out(r) and "section 0" in _out(r)
    assert not (ROOT / "runs" / "iw_ref2" / "ckpt_final.pt").exists()


@needs_run
def test_a_network_that_does_not_fit_is_refused(source):
    r = _train("iw_ref3", DST_MAP,
               ["--steps", str(DST_STEPS), "--init-weights", str(source),
                "--emb", "32"])
    assert r.returncode != 0
    assert "does not fit this run's architecture" in _out(r)


@needs_run
def test_a_setting_the_weights_read_through_must_match(source):
    # act_every changes no tensor shape, so nothing but the settings check
    # can catch it: the same weights would decide at 50 Hz instead of 25
    r = _train("iw_ref4", DST_MAP,
               ["--steps", str(DST_STEPS), "--init-weights", str(source),
                "--act-every", "2"])
    assert r.returncode != 0
    assert "act_every" in _out(r) and "differ from the source" in _out(r)
    assert not (ROOT / "runs" / "iw_ref4" / "ckpt_final.pt").exists()


@needs_run
def test_a_missing_checkpoint_is_refused():
    r = _train("iw_ref5", DST_MAP,
               ["--steps", str(DST_STEPS), "--init-weights",
                str(ROOT / "runs" / "iw_no_such" / "ckpt.pt")])
    assert r.returncode != 0
    assert "--init-weights: no checkpoint at" in _out(r)


# ==========================================================================
# 5. a crash-resume keeps the provenance
# ==========================================================================
@needs_run
def test_a_resume_of_the_run_keeps_the_provenance(source):
    if not (ROOT / "runs" / "iw_dst" / "ckpt_final.pt").exists():
        r = _train("iw_dst", DST_MAP,
                   ["--steps", str(DST_STEPS), "--seed", "2",
                    "--init-weights", str(source)], lr="0")
        assert r.returncode == 0, _out(r)
    first = _cfg("iw_dst")
    ck = ROOT / "runs" / "iw_dst" / "ckpt_final.pt"
    r = _train("iw_dst_res", DST_MAP,
               ["--steps", str(DST_STEPS), "--ckpt", str(ck)], lr="0")
    assert r.returncode == 0, _out(r)
    assert "--init-weights:" not in r.stdout          # a plain resume
    res = _cfg("iw_dst_res")
    assert all(res[k] == first[k] for k in KEYS)


# ==========================================================================
# 6. the recorder declares the keys
# ==========================================================================
def test_the_recorder_declares_the_provenance_keys():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    for k in KEYS:
        assert k in record_ckpt.TRAIN_ONLY, k
    record_ckpt.audit_cfg({"init_weights": "ckpt_latest.pt",
                           "init_weights_md5": "0" * 32,
                           "init_weights_step": 1,
                           "init_weights_map": "maps/x.bsp",
                           "init_weights_demo_file": None}, strict=True)
