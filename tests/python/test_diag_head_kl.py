"""--diag-head-kl N (2026-10-21, autoresearch r4b2): every N updates the trainer prints PPO's
drift PER ACTION HEAD over the whole update - mean(old - new log-prob) of the stored actions for
the keys joint and each Gaussian head (yaw, pitch), the mean shift in old sigmas, and the share
of rows whose ratio ends outside 1 +- clip with and without the pitch head. Why: approx_kl read
3-55 per update at the unstuck T cap 0.5 (r3b3) with the pitch sigma at 0.0067 in z; a head
that only aims the camera can then decide which rows PPO clips for every head.

* CPU smoke (sidistic, abs view, keys-hold, unstuck): the line is printed every update, its
  before-probe reproduces the rollout's own log-prob, the config carries the setting, and a
  bare resume does NOT restore it (a log line, not run state);
* refused on the bins (it reads the --view-continuous heads); the recorder lists it as
  training-only;
* flag off is bit-identical to the trainer before the change (config, csv, eval bytes, weights).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run   # noqa: E402

LINE = re.compile(r"head-kl it (\d+) step [\d,]+: keys (\S+)  z0 (\S+)  z1 (\S+)  "
                  r"\|dmu\|/sig (\S+)/(\S+)  clip% joint (\S+)% no-last (\S+)%  "
                  r"sig (\S+)/(\S+)  joint kl (\S+)  err (\S+)")


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


def test_the_recorder_treats_the_flag_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"diag_head_kl"' in src


@needs_run
def test_the_probe_prints_per_head_drift_and_is_not_restored():
    run = "hkl_on"
    r = _train(run, ["--unstuck", "--unstuck-reach", "eval", "--diag-head-kl", "1",
                     "--epochs", "2", "--minibatches", "4"])
    assert "--diag-head-kl 1: per-head drift of the update every 1 updates" in r.stdout
    rows = [m.groups() for m in LINE.finditer(r.stdout)]
    assert len(rows) == 3, r.stdout[-3000:]          # 6,144 steps = 3 updates of 2,048
    for g in rows:
        keys, z0, z1 = float(g[1]), float(g[2]), float(g[3])
        for v in (keys, z0, z1):
            assert v == v                                   # finite
        assert 0.0 <= float(g[6]) <= 100.0 and 0.0 <= float(g[7]) <= 100.0
        # the before-probe scores the stored actions under the rollout's own measure
        assert float(g[11]) < 5e-3, g
    # the joint is the sum of the parts
    g = rows[-1]
    assert abs(float(g[10]) - (float(g[1]) + float(g[2]) + float(g[3]))) < 2e-3
    assert _cfg(run)["diag_head_kl"] == 1
    d = ROOT / "runs" / run
    run2 = "hkl_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "head-kl it" not in r2.stdout and "diag_head_kl" not in _cfg(run2)
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
def test_the_bins_are_refused():
    smoke = [a for a in SMOKE if a not in ("--view-continuous", "--view-absolute", "velocity")]
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "hkl_bad"] + smoke
             + ["--steps", "2048", "--diag-head-kl", "1"])
    shutil.rmtree(ROOT / "runs" / "hkl_bad", ignore_errors=True)
    assert r.returncode != 0 and "--diag-head-kl reads the --view-continuous heads" in (
        r.stdout + r.stderr), r.stdout[-2000:]


def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--diag-head-kl" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_hkl.py"
    extra = ["--unstuck", "--unstuck-reach", "eval", "--record-every", "4096"]
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("hkl_off_new", extra)
        _train("hkl_off_old", extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "hkl_off_new", ROOT / "runs" / "hkl_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "diag_head_kl" not in ca
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
