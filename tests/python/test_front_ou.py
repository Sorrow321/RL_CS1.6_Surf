"""--front-heads yaw / --front-ou TAU (2026-10-22, autoresearch r6b4): the frontier temperature on
the yaw head alone, and inside the band the yaw head's standard-normal draw as a per-env AR(1)
(Ornstein-Uhlenbeck) sequence with unit marginal variance - colored noise in PPO's reparameterised
Gaussian (Hollenstein, Martius, Piater 2024), so the stored z is the executed z and its density is
the one the update uses.

(a) ou_step keeps a stationary N(0, 1) and has lag-k correlation rho^k; sample_view with eps draws
    exactly mu + sigma x eps and returns that z's log-density; the recorder treats both flags as
    training-only.
(b) The trainer (CPU, sidistic, a resume onto a filled reservoir): the band holds envs, the startup
    and step lines and the config say so, a bare resume restores both; the refusals.
(c) Flag off: the trainer before this change and after it give the same config, progress.csv (fps
    aside), eval trajectory bytes and weights on the same smoke.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402


def test_ou_step_is_stationary_unit_variance_with_geometric_correlation():
    from train_fast import ou_step
    rng = np.random.default_rng(0)
    rho = 0.96
    x = rng.standard_normal(200_000).astype(np.float32)
    x0 = x.copy()
    xs = [x0]
    for _ in range(10):
        x = ou_step(x, rng.standard_normal(x.shape).astype(np.float32), rho)
        xs.append(x)
    assert abs(float(xs[-1].std()) - 1.0) < 0.01 and abs(float(xs[-1].mean())) < 0.01
    for k in (1, 5, 10):
        c = float(np.corrcoef(xs[0], xs[k])[0, 1])
        assert abs(c - rho ** k) < 0.01, (k, c)
    # rho 0 is white noise: the next draw is the new normal itself
    n = rng.standard_normal(1000).astype(np.float32)
    assert np.allclose(ou_step(x0[:1000], n, 0.0), n)


def test_sample_view_with_eps_draws_mu_plus_sigma_eps_and_its_density():
    from train_fast import N_VIEW, NVEC, gauss_logp, sample_view
    g = torch.Generator().manual_seed(1)
    b, nz = 5, 2
    padded = torch.randn(b, len(NVEC), max(NVEC), generator=g)
    for h in range(N_VIEW, len(NVEC)):
        padded[:, h, NVEC[h]:] = -1e30
    mu = torch.randn(b, nz, generator=g)
    log_std = torch.tensor([-2.5, -1.0])
    tempv = torch.tensor([[3.0, 1.0]] * b)
    eps = torch.randn(b, nz, generator=g)
    act, z, logp = sample_view(padded, mu, log_std, torch.ones(b, 1, 1), tempv, eps=eps)
    ls = log_std + torch.log(tempv)
    assert torch.allclose(z, mu + ls.exp() * eps)
    # the joint log-prob is the categorical part plus the Gaussian density of THIS z
    act2, z2, logp2 = sample_view(padded, mu, log_std, torch.ones(b, 1, 1), tempv, eps=eps)
    assert torch.allclose(z, z2)
    lc = logp - gauss_logp(z, mu, ls)
    assert torch.isfinite(lc).all() and (lc <= 0).all()


def test_the_recorder_treats_the_flags_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert {"front_heads", "front_ou"} <= set(record_ckpt.TRAIN_ONLY)


def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_a_resume_runs_the_yaw_ou_band_and_restores_it():
    from test_respawn_greedy import _seeded_ckpt
    d0, seeded = _seeded_ckpt("frontou_base")
    run = "frontou_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    band_smoke = ["--n-steps", "64", "--stall-secs", "1", "--respawn-reservoir", "5000"]
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + band_smoke
             + ["--steps", "49152", "--front-temp", "2", "--front-band", "0.95",
                "--front-heads", "yaw", "--front-ou", "0.5",
                "--unstuck", "--unstuck-temp-heads", "keys"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "samples the YAW head at 3" in r.stdout
    assert "--front-ou 0.5: inside the band the yaw draw is an AR(1) sequence, rho 0.9231" in r.stdout
    shares = [int(m) for m in re.findall(r"  fronttemp\+ou (\d+)%", r.stdout)]
    assert shares and max(shares) > 0, r.stdout[-3000:]
    cfg = _cfg(run)
    assert cfg["front_heads"] == "yaw" and cfg["front_ou"] == 0.5 and cfg["front_temp"] == 2.0
    run2 = "frontou_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + band_smoke
              + ["--steps", "65536"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "front_heads=yaw" in r2.stdout and "front_ou=0.5" in r2.stdout
    c2 = _cfg(run2)
    assert c2["front_heads"] == "yaw" and c2["front_ou"] == 0.5
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--front-ou", "0.5"], "they need --front-temp T_F > 0"),
    (["--front-heads", "yaw"], "they need --front-temp T_F > 0"),
    (["--front-temp", "1", "--front-ou", "-1"], "--front-ou TAU needs TAU >= 0"),
])
def test_bad_settings_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "frontou_bad"] + SMOKE
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "frontou_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--front-ou" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_frontou.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("frontou_off_new", [])
        _train("frontou_off_old", [], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "frontou_off_new", ROOT / "runs" / "frontou_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and "front_ou" not in ca and "front_heads" not in ca
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
