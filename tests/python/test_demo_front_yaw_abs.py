"""ab/yawabs (2026-11-14): two yaw-noise knobs from October's princessburglar crossing.

October's lineage learned the pit's far-wall impact (pbr2b3, 2026-10-24) only when its curriculum
band spawns drew the yaw at sigma 0.066 (z) for 0.4 s - `--demo-front-view 6` on a policy whose
own yaw sigma was 0.011. That flag MULTIPLIES the policy's sigma, so on today's policies (sigma
0.2-0.5 at the cap) the same flag is a near-uniform view (ledger w3_pbi, arm ii) and October's
noise was never reproduced (ledger w9_pbhist).

* `--demo-front-yaw-abs S` (needs --demo-front-temp): band spawns draw the yaw at the ABSOLUTE
  sigma S for --demo-front-ticks, whatever the policy's own sigma; the per-row slot holds
  S / sigma (band_yaw_mult, re-read every iteration), so PPO scores each row at the sigma it was
  drawn at, as for the existing band temperature.
* `--yaw-entropy-progress HI:LO:P`: --yaw-entropy's coefficient read off the greedy frontier at
  every greedy eval (the --int-coef-progress reading): HI at progress 0 -> LO at fraction P;
  1:0:P reproduces --yaw-entropy 0 past P. The loss gets a 0-d tensor the trainer rewrites.

CPU: the multiplier, the loss with a tensor share equal to the float one, the parse and the
schedule, the recorder's TRAIN_ONLY keys; princessburglar trainer runs: on (lines, config, the
eval's reading), the refusals, and flag off bit-identical to the trainer before the change.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def test_the_band_multiplier():
    from train_fast import band_yaw_mult
    # October's policy (sigma 0.011) needs x6 for 0.066; today's capped one (0.48) needs x0.1375
    m = band_yaw_mult(0.066, np.log([0.011, 0.48]))
    assert m.dtype == np.float32
    assert m[0] == pytest.approx(6.0, rel=1e-5) and m[1] == pytest.approx(0.1375, rel=1e-5)
    # sigma x multiplier is S, for any sigma
    for sig in (0.003, 0.2, 0.5, 2.718):
        assert sig * float(band_yaw_mult(0.066, [np.log(sig)])[0]) == pytest.approx(0.066)


def test_a_tensor_share_is_the_float_share():
    import torch
    from train_fast import logprob_entropy_view, N_VIEW
    torch.manual_seed(0)
    B, nz = 6, 2
    padded = torch.randn(B, N_VIEW + 4, 5)                  # (rows, heads, padded bins)
    acts = torch.randint(0, 5, (B, N_VIEW + 4))
    mu, z = torch.randn(B, nz), torch.randn(B, nz)
    log_std = torch.tensor([-0.7, -1.1])
    for share in (0.0, 0.5, 2.0):
        for tv in (None, torch.full((B, nz), 1.3)):        # one sigma / one per row
            lf, ef = logprob_entropy_view(padded, acts, mu, log_std, z, 0.0, None, tv,
                                          yaw_ent=share)
            lt, et = logprob_entropy_view(padded, acts, mu, log_std, z, 0.0, None, tv,
                                          yaw_ent=torch.tensor(share))
            assert torch.equal(lf, lt)
            assert torch.allclose(ef, et, atol=0, rtol=1e-6), (share, tv is None)
    # a tensor share of 1 is the default's value (not its op order, so allclose)
    l1, e1 = logprob_entropy_view(padded, acts, mu, log_std, z, 0.0)
    lt, et = logprob_entropy_view(padded, acts, mu, log_std, z, 0.0, yaw_ent=torch.tensor(1.0))
    assert torch.equal(l1, lt) and torch.allclose(e1, et, rtol=1e-6)


def test_the_schedule():
    from train_fast import int_coef_at, parse_yaw_entropy_progress
    assert parse_yaw_entropy_progress("1:0:0.02") == (1.0, 0.0, 0.02)
    for bad in ("1:0", "a:b:c", "1:0:0", "1:-1:0.5", "-1:0:0.5", "1:0:1.5"):
        with pytest.raises(SystemExit):
            parse_yaw_entropy_progress(bad)
    hi, lo, p = parse_yaw_entropy_progress("1:0:0.02")
    assert int_coef_at(float("nan"), hi, lo, p) == 1.0          # no reading yet: HI
    assert int_coef_at(0.0, hi, lo, p) == 1.0
    assert int_coef_at(0.01, hi, lo, p) == pytest.approx(0.5)
    assert int_coef_at(0.02, hi, lo, p) == 0.0 and int_coef_at(0.9, hi, lo, p) == 0.0


def test_the_recorder_ignores_both():
    import record_ckpt
    assert "demo_front_yaw_abs" in record_ckpt.TRAIN_ONLY
    assert "yaw_entropy_progress" in record_ckpt.TRAIN_ONLY


@needs_pb
def test_on_dumped_and_read(tmp_path):
    run = "dfya_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-front-temp", "0.5",
                            "--demo-front-yaw-abs", "0.066", "--demo-front-ticks", "40",
                            "--yaw-entropy-progress", "1:0:0.02", "--steps", "8192",
                            "--record-every", "4096", "--eval-eps", "2"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert ("samples the keys at 1.5 and the yaw at an ABSOLUTE sigma 0.066 (z; "
                "--demo-front-yaw-abs) for its first 40 ticks") in out
        assert "--demo-front-yaw-abs: band rows drawn at yaw sigma 0.066 = the policy's" in out
        assert ("--yaw-entropy-progress 1:0:0.02: the yaw head's entropy share is 1 at frontier "
                "progress 0") in out
        assert "--yaw-entropy-progress: greedy frontier" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_front_yaw_abs"] == 0.066 and cfg["demo_front_view"] is None
        assert cfg["yaw_entropy_progress"] == [1.0, 0.0, 0.02]
        assert "yaw_entropy" not in cfg
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_pb
def test_the_refusals(tmp_path):
    for name, extra, msg in (
            ("dfya_bad0", ["--demo-front-yaw-abs", "0.066"], "need --demo-front-temp"),
            ("dfya_bad1", ["--demo-front-temp", "0.5", "--demo-front-view", "6",
                           "--demo-front-yaw-abs", "0.066"], "pick one"),
            ("dfya_bad2", ["--demo-front-temp", "0.5", "--demo-front-yaw-abs", "0"],
             "--demo-front-yaw-abs is a yaw sigma in z: > 0"),
            ("dfya_bad3", ["--yaw-entropy-progress", "1:0:0.02", "--yaw-entropy", "0"],
             "--yaw-entropy-progress and --yaw-entropy both set"),
            ("dfya_bad4", ["--yaw-entropy-progress", "1:0"], "HI:LO:P (three numbers)")):
        bad = _run_demo(name, extra, tmp_path)
        assert bad.returncode != 0 and msg in (bad.stdout + bad.stderr), name
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-front-yaw-abs" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
@pytest.mark.parametrize("band", [False, True])
def test_flag_off_is_bit_identical(tmp_path, band):
    pre = ROOT / "python" / "_train_fast_pre_yawabs.py"
    extra = ["--demo-success-d", "50", "--demo-grow", "5", "--demo-front-frac", "0.67",
             "--steps", "8192"]
    if band:
        extra += ["--demo-front-temp", "0.5", "--demo-front-view", "6", "--yaw-entropy", "0"]
    tag = "b" if band else "n"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in ((f"dfya_off_new_{tag}", TRAIN), (f"dfya_off_old_{tag}", pre)):
            r = _run_demo(run, extra, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / f"dfya_off_new_{tag}", ROOT / "runs" / f"dfya_off_old_{tag}"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_front_yaw_abs" not in ca and "yaw_entropy_progress" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
