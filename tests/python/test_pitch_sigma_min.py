"""--pitch-sigma-min S (2026-10-21, autoresearch r4b2): under --view-absolute the pitch head's
sigma is floored at S in z, kept by the same projection as its log 0.5 cap (the raw log sigma
is lifted to log S after every optimizer step and once on resume).

Why: pitch only aims the camera, so with its entropy term off (the --view-absolute default)
PPO drives its sigma to the LOG_STD_MIN clamp - r3b3's checkpoint holds a raw pitch log sigma
of -5.0015, below the clamp (sigma 0.0067 = 0.3 deg of target, gradient dead) - and the joint
PPO ratio then turns on that head: a 1-degree drift of the camera target is 3 sigma.

* CPU smoke (sidistic, abs view, keys-hold, unstuck): the config carries S; a resume of a
  checkpoint whose raw pitch log sigma sits at -5.0015 (r3b3's value) restores S bare, says it
  lifted the head, and every later save holds the raw value at log S or above; the yaw head is
  never touched by the floor;
* refused without --view-absolute and outside [exp(LOG_STD_MIN), 0.5]; the recorder lists it
  as training-only (the floor lives in the stored weights);
* flag off is bit-identical to the trainer before the change (config, csv, eval bytes, weights).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run   # noqa: E402


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


def test_the_recorder_treats_the_flag_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"pitch_sigma_min"' in src


@needs_run
def test_the_floor_lifts_a_collapsed_pitch_head_and_is_restored():
    import torch
    run = "psm_on"
    r = _train(run, ["--unstuck", "--unstuck-reach", "eval", "--pitch-sigma-min", "0.05"])
    assert "--pitch-sigma-min 0.05: the pitch head's sigma is floored at 0.05" in r.stdout
    assert _cfg(run)["pitch_sigma_min"] == 0.05
    d = ROOT / "runs" / run
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    yaw0 = float(ck["policy"]["view_std.log_std"][0])
    ck["policy"]["view_std.log_std"][-1] = -5.0015          # r3b3's collapsed pitch head
    bad = d / "ckpt_collapsed.pt"
    torch.save(ck, bad)
    run2 = "psm_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt", str(bad)]
              + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "pitch_sigma_min=0.05" in r2.stdout
    assert ("--pitch-sigma-min: the checkpoint's raw pitch log sigma -5.0015 lifted to "
            f"{math.log(0.05):.4f}") in r2.stdout
    sigs = re.findall(r"  sig (\d\.\d+)/(\d\.\d+)", r2.stdout)
    assert sigs and all(float(p) >= 0.05 - 5e-4 for _, p in sigs), sigs
    ck2 = torch.load(ROOT / "runs" / run2 / "ckpt_final.pt", map_location="cpu",
                     weights_only=False)
    ls = ck2["policy"]["view_std.log_std"]
    assert float(ls[-1]) >= math.log(0.05) - 1e-6
    # the yaw head is PPO's alone: it moved by optimizer steps only, never to the floor
    assert abs(float(ls[0]) - yaw0) < 0.05 and float(ls[0]) != math.log(0.05)
    assert _cfg(run2)["pitch_sigma_min"] == 0.05
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--pitch-sigma-min", "0.6"], "--pitch-sigma-min must lie in"),
    (["--pitch-sigma-min", "0.001"], "--pitch-sigma-min must lie in"),
])
def test_bad_settings_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "psm_bad"] + SMOKE
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "psm_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]


@needs_run
def test_the_relative_view_is_refused():
    smoke = [a for a in SMOKE if a not in ("--view-absolute", "velocity")]
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "psm_bad2"] + smoke
             + ["--steps", "2048", "--pitch-sigma-min", "0.05"])
    shutil.rmtree(ROOT / "runs" / "psm_bad2", ignore_errors=True)
    assert r.returncode != 0 and "it needs --view-continuous --view-absolute" in (
        r.stdout + r.stderr), r.stdout[-2000:]


def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--pitch-sigma-min" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_psm.py"
    extra = ["--unstuck", "--unstuck-reach", "eval", "--record-every", "4096"]
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("psm_off_new", extra)
        _train("psm_off_old", extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "psm_off_new", ROOT / "runs" / "psm_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "pitch_sigma_min" not in ca
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
