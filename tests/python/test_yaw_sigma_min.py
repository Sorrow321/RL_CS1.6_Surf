"""--yaw-sigma-min S (pbfr1b4, 2026-10-29): a floor on the --view-absolute YAW head(s)' sigma, the
--pitch-sigma-min mechanism for every view head but the last. The expert loop's final policy
(pbEXIT2_r38) came out with the yaw sigma at 0.017 and its mean pinned by BC; resumed for a
backward curriculum without BC, approx_kl read 0.13 after ONE Adam step and 1-4.7 per update.

(a) The policy: log_std() floors the yaw head (the pitch head untouched, or on its own floor when
    both are set), the state_dict keys are the flag-off ones, project_log_std lifts a raw yaw
    parameter parked under the floor, flag off builds no buffer, the delta policy refuses it.
(b) The trainer (CPU, princessburglar): the config says so, the step line's yaw sigma reads the
    floor, a bare resume restores it, record_ckpt reads the key (its config audit passes and a
    stochastic recording runs), an out-of-range value is refused.
(c) Flag off: the trainer before this change and after it give the same config, progress.csv
    (fps aside), eval trajectory bytes and weights on the same smoke.
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

from test_ge_minutes import SMOKE as PB_SMOKE, _env, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
# the princessburglar smoke in the policy's own action / view layout (the incumbent's)
SMOKE = PB_SMOKE + ["--view-continuous", "--view-absolute", "velocity", "--keys-hold",
                    "--seed", "7"]


def _run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=str(ROOT),
                          timeout=timeout, encoding="utf-8", errors="replace")


def _train(run, extra, script=TRAIN, steps="2048"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(script), "--run", run] + SMOKE
             + ["--steps", steps] + list(extra))
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _csv(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [dict(zip(head, r.split(","))) for r in rows[1:]]


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


def _policy(**kw):
    import torch
    from train_fast import Policy
    torch.manual_seed(0)
    return Policy(15 + 32, 8, 4, emb=16, hidden=16, view_continuous=True,
                  view_absolute="velocity", **kw)


# ==========================================================================
# (a) the policy
# ==========================================================================
def test_the_yaw_head_is_floored_and_the_pitch_head_is_not():
    import torch
    from train_fast import LOG_STD_MIN
    p = _policy(yaw_sigma_min=0.05)
    with torch.no_grad():
        p.view_std.log_std.fill_(LOG_STD_MIN - 1.0)          # both heads parked deep under
    s = p.log_std().exp().tolist()
    assert s[0] == pytest.approx(0.05, rel=1e-6)            # yaw: the floor
    assert s[-1] == pytest.approx(float(np.exp(LOG_STD_MIN)), rel=1e-6)   # pitch: global floor


def test_both_floors_together():
    import torch
    from train_fast import LOG_STD_MIN
    p = _policy(yaw_sigma_min=0.05, pitch_sigma_min=0.1)
    with torch.no_grad():
        p.view_std.log_std.fill_(LOG_STD_MIN - 1.0)
    s = p.log_std().exp().tolist()
    assert s[0] == pytest.approx(0.05, rel=1e-6) and s[-1] == pytest.approx(0.1, rel=1e-6)


def test_a_pitch_only_floor_is_the_buffer_it_was():
    from train_fast import LOG_STD_MIN
    lo = _policy(pitch_sigma_min=0.05).view_std.log_std_lo.tolist()
    assert lo[0] == pytest.approx(LOG_STD_MIN) and lo[-1] == pytest.approx(float(np.log(0.05)))


def test_the_state_dict_keys_do_not_change():
    a, b = _policy(), _policy(yaw_sigma_min=0.05)
    assert list(a.state_dict()) == list(b.state_dict())
    for k in a.state_dict():
        assert a.state_dict()[k].shape == b.state_dict()[k].shape, k


def test_project_lifts_a_raw_yaw_parameter_under_the_floor():
    import torch
    p = _policy(yaw_sigma_min=0.05)
    with torch.no_grad():
        p.view_std.log_std.copy_(torch.tensor([float(np.log(0.017)), -1.0]))
    assert p.project_log_std() == 1
    assert float(p.view_std.log_std[0]) == pytest.approx(float(np.log(0.05)), rel=1e-6)
    assert float(p.view_std.log_std[-1]) == pytest.approx(-1.0)
    assert p.project_log_std() == 0                         # idempotent


def test_flag_off_builds_no_buffer_and_the_delta_policy_refuses_the_floor():
    import torch
    from train_fast import Policy
    assert getattr(_policy().view_std, "log_std_lo", None) is None
    assert getattr(_policy(yaw_sigma_min=0.0).view_std, "log_std_lo", None) is None
    torch.manual_seed(0)
    with pytest.raises(SystemExit):
        Policy(15 + 32, 8, 4, emb=16, hidden=16, view_continuous=True,
               view_absolute=None, yaw_sigma_min=0.05)


# ==========================================================================
# (b) the trainer
# ==========================================================================
@needs_pb
def test_the_trainer_floors_the_yaw_and_a_resume_restores_it():
    run = "ysmin_on"
    r = _train(run, ["--record-every", "2048", "--eval-eps", "2", "--yaw-sigma-min", "0.05"],
               steps="2048")
    assert "--yaw-sigma-min 0.05: the yaw head's sigma is floored at 0.05" in r.stdout
    assert _cfg(run)["yaw_sigma_min"] == 0.05
    sig = re.findall(r"sig ([0-9.]+)/([0-9.]+)", r.stdout)
    assert sig and all(float(a) >= 0.0499 for a, _ in sig)
    ck = ROOT / "runs" / run / "ckpt_final.pt"
    r2 = _run([sys.executable, "-u", str(TRAIN), "--ckpt", str(ck), "--run", run + "_res",
               "--steps", str(2048 + 1024), "--envs", "64", "--record-every", "1e12",
               "--ckpt-every", "1e9", "--eval-eps", "1", "--eval-greedy-only"])
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "yaw_sigma_min=0.05" in r2.stdout
    assert _cfg(run + "_res")["yaw_sigma_min"] == 0.05
    # the recorder reads the key: its config audit passes and a stochastic recording runs
    r3 = _run([sys.executable, "-u", str(ROOT / "tools" / "record_ckpt.py"), str(ck),
               "--episodes", "1", "--ep-ticks", "200", "--stochastic",
               "--out", str(ROOT / "runs" / run / "rec_ysmin.jsonl")])
    assert r3.returncode == 0, r3.stdout[-3000:] + r3.stderr[-3000:]
    assert "CONFIG MISMATCH" not in r3.stdout + r3.stderr
    for dd in (run, run + "_res"):
        shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)


@needs_pb
def test_an_out_of_range_floor_is_refused():
    for bad in ("0.001", "5.0"):
        r = _run([sys.executable, "-u", str(TRAIN), "--run", "ysmin_bad", "--steps", "1024"]
                 + SMOKE + ["--yaw-sigma-min", bad])
        assert r.returncode != 0
        assert "--yaw-sigma-min must lie in" in r.stdout + r.stderr
    shutil.rmtree(ROOT / "runs" / "ysmin_bad", ignore_errors=True)


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
        if s.returncode == 0 and b"--yaw-sigma-min" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_ysmin.py"
    extra = ["--record-every", "1024", "--eval-eps", "1", "--pitch-sigma-min", "0.05"]
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("ysmin_off_new", extra)
        _train("ysmin_off_old", extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "ysmin_off_new", ROOT / "runs" / "ysmin_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "yaw_sigma_min" not in ca
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
