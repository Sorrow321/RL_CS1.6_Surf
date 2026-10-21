"""--pitch-std-min (a floor on the --view-absolute pitch head's sigma) and --diag-heads
(a print-only per-head gradient / KL diagnostic of the PPO update).

(a) the Policy: off is the shipped log_std() expression; the floor binds the pitch head
    only, passes its gradient whole at the floor, and project_log_std() lifts the raw
    parameter to it;
(b) the trainer (CPU smoke on sidistic): the config carries the floor, the resume lifts a
    raw pitch log sigma parked under it, a bare resume restores it, record_ckpt mirrors it
    (a stochastic recording runs), --diag-heads prints its line, bad settings are refused;
(c) both flags off are bit-identical to the trainer before the change (config, csv, eval
    bytes, weights).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/python/test_pitch_std_min.py -p no:cacheprovider
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402

import train_fast as tf                                                  # noqa: E402

RECORD = ROOT / "tools" / "record_ckpt.py"


def _policy(view_absolute="velocity"):
    torch.manual_seed(0)
    return tf.Policy(tf.N_SCALAR + 8 * 4, 8, 4, emb=16, hidden=12,
                     view_continuous=True, view_absolute=view_absolute)


# ==========================================================================
# (a) the Policy
# ==========================================================================
def test_off_is_the_shipped_expression():
    p = _policy()
    with torch.no_grad():
        p.view_std.log_std.copy_(torch.tensor([-2.5, -6.0]))
    want = torch.clamp(p.view_std.log_std.clamp(tf.LOG_STD_MIN, tf.LOG_STD_MAX),
                       max=p.view_std.log_std_hi)
    assert torch.equal(p.log_std(), want)
    assert getattr(p.view_std, "log_std_lo", None) is None
    assert "view_std.log_std_lo" not in p.state_dict()


def test_the_floor_binds_the_pitch_head_only_and_keeps_its_gradient():
    p = _policy()
    p.set_pitch_floor(0.05)
    lo = math.log(0.05)
    with torch.no_grad():
        p.view_std.log_std.copy_(torch.tensor([-4.9, -5.2]))
    ls = p.log_std()
    assert ls[0].item() == pytest.approx(-4.9)          # the yaw keeps LOG_STD_MIN
    assert ls[1].item() == pytest.approx(lo)            # the pitch is floored
    # non-persistent: the state_dict keys are the shipped ones
    assert "view_std.log_std_lo" not in p.state_dict()
    # the projection lifts the raw pitch parameter, leaves the yaw
    assert p.project_log_std() == 1
    assert p.view_std.log_std[1].item() == pytest.approx(lo)
    assert p.view_std.log_std[0].item() == pytest.approx(-4.9)
    # at the floor the clamp passes the gradient whole (a later PG step can lift it)
    p.view_std.log_std.grad = None
    p.log_std().sum().backward()
    assert p.view_std.log_std.grad.tolist() == [1.0, 1.0]
    # the cap still binds from above
    with torch.no_grad():
        p.view_std.log_std.copy_(torch.tensor([0.5, 0.5]))
    assert p.log_std()[1].item() == pytest.approx(math.log(0.5))


def test_the_floor_needs_the_absolute_view():
    p = _policy(view_absolute=None)
    with pytest.raises(SystemExit):
        p.set_pitch_floor(0.05)


# ==========================================================================
# (b) the trainer
# ==========================================================================
@needs_run
def test_trainer_floor_resume_restore_record_and_diag():
    d0 = ROOT / "runs" / "psm_base"
    _train("psm_base", [], steps="2048")
    ck0 = torch.load(d0 / "ckpt_final.pt", map_location="cpu", weights_only=False)
    # park the raw pitch log sigma under the floor, as on every round-1..3 checkpoint
    ck0["policy"]["view_std.log_std"][-1] = -5.0015
    seeded = d0 / "ckpt_parked.pt"
    torch.save(ck0, seeded)
    run = "psm_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + ["--steps", "6144", "--pitch-std-min", "0.05", "--diag-heads", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    out = r.stdout
    assert "--pitch-std-min 0.05: the pitch head's sigma is floored at 0.05" in out
    assert "projected into it" in out
    diag = re.findall(r"diag-heads it \d+ step [0-9,]+: \|grad\| shared actor keys (\S+) "
                      r"yaw (\S+) pitch (\S+) \(pitch (\d+)% of the norm\) \| kl keys (\S+) "
                      r"yaw (\S+) pitch (\S+) \| clip joint (\S+)% w/o pitch (\S+)% \| sig "
                      r"(\S+)/(\S+) \| logp check (\S+)", out)
    assert len(diag) >= 2, out[-3000:]
    for row in diag:
        assert all(np.isfinite(float(x)) for x in row)
        assert float(row[-2]) >= 0.05 - 1e-6           # the tempered pitch sigma >= floor
        assert float(row[-1]) < 1e-2                    # the split sums to the stored logp
    cfg = tf_cfg(run)
    assert cfg["pitch_std_min"] == 0.05
    ck = torch.load(ROOT / "runs" / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    assert ck["policy"]["view_std.log_std"][-1].item() >= math.log(0.05) - 1e-6
    # a bare resume restores the floor
    run2 = "psm_bare"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "pitch_std_min=0.05" in r2.stdout
    assert tf_cfg(run2)["pitch_std_min"] == 0.05
    # record_ckpt mirrors it: the audit passes and a stochastic recording runs
    r3 = _run([sys.executable, "-u", str(RECORD), str(ROOT / "runs" / run / "ckpt_final.pt"),
               "--episodes", "1", "--stochastic", "--ep-ticks", "300"])
    assert r3.returncode == 0, r3.stdout[-4000:] + r3.stderr[-4000:]
    assert "sigma " in r3.stdout and "/0.050" in r3.stdout, r3.stdout[-2000:]
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


def tf_cfg(run):
    import json
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--pitch-std-min", "-1"], "--pitch-std-min must be >= 0"),
    (["--pitch-std-min", "0.9"], "is above the pitch head's own cap"),
    (["--diag-heads", "-2"], "--diag-heads must be >= 0"),
])
def test_bad_settings_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "psm_bad"] + SMOKE
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "psm_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]


# ==========================================================================
# (c) flags off
# ==========================================================================
def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--pitch-std-min" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flags_off_are_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_psm.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("psm_off_new", [])
        _train("psm_off_old", [], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "psm_off_new", ROOT / "runs" / "psm_off_old"
    ca, cb = tf_cfg(a.name), tf_cfg(b.name)
    assert ca == cb and "pitch_std_min" not in ca
    ra, rb = _csv(a.name), _csv(b.name)
    assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
    for x, y in zip(ra, rb):
        for k in x:
            if k != "time/fps":
                assert x[k] == y[k], k
    ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
    assert [p.name for p in ta] == [p.name for p in tb]
    for p, q in zip(ta, tb):
        assert p.read_bytes() == q.read_bytes(), p.name
    sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k in sa["policy"]:
        assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
