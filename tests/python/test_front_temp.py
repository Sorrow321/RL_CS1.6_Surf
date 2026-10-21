"""--front-temp T_F (2026-10-21, autoresearch r4b3): a PER-ENV sampling temperature - every head
at 1 + T_F for an env inside the frontier band (progress >= (1 - --front-band) x the reservoir's
deepest), the global --unstuck temperature (on its heads) or none elsewhere; PPO scores each row
at the temperature it was drawn at.

(a) The update's log-prob / entropy with one temperature PER ROW (logprob_entropy_view): with
    every row at the same temperature it is the scalar path (log-prob row for row, the mean
    entropy), with mixed rows each row is the scalar path at its own temperature, for every
    --pitch-entropy weight.
(b) The trainer (CPU, sidistic, a resume onto a filled reservoir): the band holds envs, the step
    line and the config say so, a bare resume restores it; the refusals.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402


# ==========================================================================
# (a) the per-row log-prob and entropy
# ==========================================================================
def _batch(b=6, nz=2, seed=0):
    from train_fast import N_VIEW, NVEC
    g = torch.Generator().manual_seed(seed)
    heads = len(NVEC) - N_VIEW
    npad = max(NVEC)
    padded = torch.randn(b, len(NVEC), npad, generator=g)
    acts = torch.zeros(b, len(NVEC), dtype=torch.long)
    for h in range(N_VIEW, len(NVEC)):
        acts[:, h] = torch.randint(0, NVEC[h], (b,), generator=g)
        padded[:, h, NVEC[h]:] = -1e30
    mu = torch.randn(b, nz, generator=g)
    log_std = torch.tensor([-2.5, -1.0][:nz])
    z = mu + 0.3 * torch.randn(b, nz, generator=g)
    return padded, acts, mu, log_std, z, heads


@pytest.mark.parametrize("pitch_ent", [0.0, 0.5, 1.0])
def test_one_temperature_per_row_is_the_scalar_path(pitch_ent):
    from train_fast import logprob_entropy_view
    padded, acts, mu, log_std, z, _ = _batch()
    b, nz = mu.shape
    for t in (1.0, 1.5, 2.0):
        lp0, en0 = logprob_entropy_view(padded, acts, mu, log_std, z, pitch_ent,
                                        torch.tensor(t), torch.full((nz,), t))
        lp1, en1 = logprob_entropy_view(padded, acts, mu, log_std, z, pitch_ent,
                                        torch.full((b, 1, 1), t), torch.full((b, nz), t))
        torch.testing.assert_close(lp1, lp0, rtol=1e-5, atol=1e-5)
        assert en1.shape == (b,)
        torch.testing.assert_close(en1.mean(), en0.mean(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pitch_ent", [0.0, 1.0])
def test_mixed_rows_each_score_at_their_own_temperature(pitch_ent):
    from train_fast import logprob_entropy_view
    padded, acts, mu, log_std, z, _ = _batch(b=5)
    b, nz = mu.shape
    temps = torch.tensor([1.0, 2.0, 1.5, 1.0, 3.0])
    tv = torch.stack([temps, torch.ones(b)], 1)          # yaw tempered, pitch not
    lp, en = logprob_entropy_view(padded, acts, mu, log_std, z, pitch_ent,
                                  temps.view(b, 1, 1), tv)
    for i in range(b):
        lpi, eni = logprob_entropy_view(padded[i:i + 1], acts[i:i + 1], mu[i:i + 1],
                                        log_std, z[i:i + 1], pitch_ent,
                                        temps[i].view(1, 1, 1), tv[i])
        torch.testing.assert_close(lp[i], lpi[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(en[i], eni[0], rtol=1e-5, atol=1e-5)


def test_the_recorder_treats_the_flags_as_training_only():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert {"front_temp", "front_band"} <= set(record_ckpt.TRAIN_ONLY)


# ==========================================================================
# (b) the trainer
# ==========================================================================
def _cfg(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]


@needs_run
def test_a_resume_runs_with_envs_in_the_band_and_restores_it():
    from test_respawn_greedy import _seeded_ckpt
    d0, seeded = _seeded_ckpt("front_base")
    run = "front_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    # 64-decision rollouts and a 1 s stall kill: the first episodes end inside iteration 1 and
    # respawn from the seeded reservoir (5,000 cap keeps its 3,000 states), i.e. into the band
    band_smoke = ["--n-steps", "64", "--stall-secs", "1", "--respawn-reservoir", "5000"]
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run, "--ckpt", str(seeded)]
             + SMOKE + band_smoke
             + ["--steps", "49152", "--front-temp", "1", "--front-band", "0.95",
                "--unstuck", "--unstuck-temp-heads", "keys"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--front-temp 1: an env whose progress is >=" in r.stdout
    shares = [int(m) for m in re.findall(r"  fronttemp (\d+)%", r.stdout)]
    assert shares and max(shares) > 0, r.stdout[-3000:]
    cfg = _cfg(run)
    assert cfg["front_temp"] == 1.0 and cfg["front_band"] == 0.95
    run2 = "front_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(ROOT / "runs" / run / "ckpt_final.pt")] + SMOKE + band_smoke
              + ["--steps", "65536"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "front_temp=1" in r2.stdout and "front_band=0.95" in r2.stdout
    assert _cfg(run2)["front_temp"] == 1.0
    for dd in (d0, ROOT / "runs" / run, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--front-temp", "1", "--front-band", "1.5"], "--front-band B in (0, 1]"),
    (["--front-temp", "1", "--respawn-frac", "0"], "--respawn-frac 0"),
])
def test_bad_settings_are_refused(extra, msg):
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "front_bad"] + SMOKE
             + ["--steps", "2048"] + extra)
    shutil.rmtree(ROOT / "runs" / "front_bad", ignore_errors=True)
    assert r.returncode != 0 and msg in r.stdout + r.stderr, r.stdout[-2000:]


# ==========================================================================
# (c) flag off
# ==========================================================================
def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--front-temp" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_front.py"
    extra = ["--record-every", "2048", "--unstuck", "--unstuck-temp-heads", "keys"]
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("front_off_new", extra)
        _train("front_off_old", extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "front_off_new", ROOT / "runs" / "front_off_old"
    ca, cb = _cfg(a.name), _cfg(b.name)
    assert ca == cb and "front_temp" not in ca
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
