"""--pitch-sigma-min S (2026-10-22, autoresearch r5b4): a floor on the --view-absolute pitch
head's sigma. The pitch moves only the camera, has no entropy term and sat on the global floor
exp(LOG_STD_MIN) = 0.0067, where a ~1 deg move of its mean is several nats of the joint PPO
log-ratio (approx_kl 1-22 per update on sidistic).

(a) The policy: log_std() floors the pitch head (the yaw head untouched), the state_dict keys
    are the flag-off ones, project_log_std lifts a raw parameter parked under the floor, and
    flag off builds no buffer; the delta / non-absolute policy refuses the floor.
(b) The trainer (CPU, sidistic): the config says so, the step line's sigma reads the floor, a
    bare resume restores it, record_ckpt reads the key (its config audit passes), and an
    out-of-range value is refused.
(c) Flag off: the trainer before this change and after it give the same config,
    progress.csv (fps aside), eval trajectory bytes and weights on the same smoke.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import _csv, _run, _train, needs_run  # noqa: E402


def _policy(**kw):
    import torch
    from train_fast import Policy
    torch.manual_seed(0)
    return Policy(15 + 32, 8, 4, emb=16, hidden=16, view_continuous=True,
                  view_absolute="velocity", **kw)


# ==========================================================================
# (a) the policy
# ==========================================================================
def test_the_pitch_head_is_floored_and_the_yaw_head_is_not():
    import torch
    from train_fast import LOG_STD_MIN
    p = _policy(pitch_sigma_min=0.05)
    with torch.no_grad():
        p.view_std.log_std.fill_(LOG_STD_MIN - 1.0)          # both heads parked deep under
    s = p.log_std().exp().tolist()
    assert s[-1] == pytest.approx(0.05, rel=1e-6)           # pitch: the floor
    assert s[0] == pytest.approx(float(np.exp(LOG_STD_MIN)), rel=1e-6)   # yaw: the global floor


def test_the_state_dict_keys_do_not_change():
    a, b = _policy(), _policy(pitch_sigma_min=0.05)
    assert list(a.state_dict()) == list(b.state_dict())
    for k in a.state_dict():
        assert a.state_dict()[k].shape == b.state_dict()[k].shape, k


def test_project_lifts_a_raw_parameter_under_the_floor():
    import torch
    p = _policy(pitch_sigma_min=0.05)
    with torch.no_grad():
        p.view_std.log_std.copy_(torch.tensor([-1.0, -6.0]))
    assert p.project_log_std() == 1
    assert float(p.view_std.log_std[-1]) == pytest.approx(float(np.log(0.05)), rel=1e-6)
    assert float(p.view_std.log_std[0]) == pytest.approx(-1.0)
    assert p.project_log_std() == 0                         # idempotent


def test_flag_off_builds_no_buffer_and_the_delta_policy_refuses_the_floor():
    import torch
    from train_fast import Policy
    assert getattr(_policy().view_std, "log_std_lo", None) is None
    assert getattr(_policy(pitch_sigma_min=0.0).view_std, "log_std_lo", None) is None
    torch.manual_seed(0)
    with pytest.raises(SystemExit):
        Policy(15 + 32, 8, 4, emb=16, hidden=16, view_continuous=True,
               view_absolute=None, pitch_sigma_min=0.05)


# ==========================================================================
# (b) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


SMOKE_PS = ["--record-every", "2048", "--eval-eps", "2"]


@needs_run
def test_the_trainer_floors_the_pitch_and_a_resume_restores_it():
    run = "psmin_on"
    r = _train(run, SMOKE_PS + ["--pitch-sigma-min", "0.05"], steps="4096")
    assert "--pitch-sigma-min 0.05: the pitch head's sigma is floored at 0.05" in r.stdout
    assert _cfg(run)["pitch_sigma_min"] == 0.05
    sig = re.findall(r"sig ([0-9.]+)/([0-9.]+)", r.stdout)
    assert sig and all(float(b) >= 0.0499 for _, b in sig)
    ck = ROOT / "runs" / run / "ckpt_final.pt"
    r2 = _run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"), "--ckpt", str(ck),
               "--run", run + "_res", "--steps", str(4096 + 2048), "--envs", "64",
               "--record-every", "1e12", "--ckpt-every", "1e9", "--eval-eps", "1",
               "--eval-greedy-only"])
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "pitch_sigma_min=0.05" in r2.stdout
    assert _cfg(run + "_res")["pitch_sigma_min"] == 0.05
    # the recorder reads the key: its config audit passes and it records
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"), str(ck),
               "--episodes", "1", "--ep-ticks", "200", "--stochastic",
               "--out", str(ROOT / "runs" / run / "rec_psmin.jsonl")])
    assert r3.returncode == 0, r3.stdout[-3000:] + r3.stderr[-3000:]
    assert "CONFIG MISMATCH" not in r3.stdout + r3.stderr
    for dd in (run, run + "_res"):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)


@needs_run
def test_an_out_of_range_floor_is_refused():
    for bad in ("0.001", "0.6"):
        r = _run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                  "--run", "psmin_bad", "--steps", "2048"]
                 + __import__("test_ez_viewc").SMOKE + ["--pitch-sigma-min", bad])
        assert r.returncode != 0
        assert "--pitch-sigma-min must lie in" in r.stdout + r.stderr
    shutil.rmtree(ROOT / "runs" / "psmin_bad", ignore_errors=True)


# ==========================================================================
# (c) flag off
# ==========================================================================
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
    pre = ROOT / "python" / "_train_fast_pre_psmin.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("psmin_off_new", SMOKE_PS)
        _train("psmin_off_old", SMOKE_PS, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "psmin_off_new", ROOT / "runs" / "psmin_off_old"
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
