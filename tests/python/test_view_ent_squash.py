"""--view-ent-squash (2026-10-23, autoresearch pbr1b3): the entropy bonus of the SQUASHED
view action.

The view heads are Gaussians in z, and the core receives u = tanh(z). The update paid the
Gaussian's entropy H(z) = 0.5 + 0.5 log 2pi + log sigma, which grows without bound in sigma,
while the action's own entropy H(u) = H(z) + E[log(1 - tanh(z)^2)] peaks near sigma ~0.7 and
falls beyond it (the mass piles at u = +-1). With no reward signal on the yaw head (the
princessburglar start room pays the scratch recipe ~nothing) the bonus alone drove the yaw
sigma to the LOG_STD_MAX clamp, 2.718, in ~100M steps: a bang-bang look direction.

What is pinned here:

(a) flag off: logprob_entropy_view is the old arithmetic (the categorical entropy plus the
    Gaussian entropy, the pitch share scaled by pitch_ent) - the squash branch is never
    entered;
(b) flag on: the estimate's mean is the squashed action's entropy (against quadrature), and
    the log-prob is untouched;
(c) the gradient on log sigma: +1 per head without the flag at any sigma; with it, positive
    at small sigma and NEGATIVE at the clamp, so a signal-free head settles inside (-1, 1);
(d) the pitch share follows pitch_ent (0 under --view-absolute: its correction is not paid);
(e) per-row log sigma (--front-temp's tempered rows) is handled;
(f) a CPU smoke on princessburglar: the flag trains, writes view_ent_squash to the config and
    prints its line; the flag-off run's config has no such key.
"""
from __future__ import annotations

import json
import math
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

from train_fast import (LOG2PI, N_VIEW, NACT, logprob_entropy_padded,  # noqa: E402
                        logprob_entropy_view, squash_entropy_corr)


def _inputs(B=6, n_z=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    from train_fast import NEUTRAL_ACT, NVEC, HeadPacker
    padded = HeadPacker(torch.device("cpu")).pad(torch.randn(B, sum(NVEC), generator=g))
    act = torch.zeros(B, NACT, dtype=torch.long)
    act[:, 0], act[:, 1] = NEUTRAL_ACT[0], NEUTRAL_ACT[1]
    for h in range(N_VIEW, NACT):
        act[:, h] = torch.randint(0, NVEC[h], (B,), generator=g)
    mu = torch.randn(B, n_z, generator=g)
    log_std = torch.tensor([0.3, -0.7][:n_z] + [0.1] * max(0, n_z - 2))
    z = mu + log_std.exp() * torch.randn(B, n_z, generator=g)
    return padded, act, mu, log_std, z


def _sq_entropy_quad(mu: float, sigma: float) -> float:
    """H(tanh(z)), z ~ N(mu, sigma), by quadrature: H(z) + E[log(1 - tanh(z)^2)]."""
    x = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 400001)
    p = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    lj = 2.0 * (math.log(2.0) - x - np.logaddexp(0.0, -2.0 * x))
    return 0.5 + 0.5 * math.log(2 * math.pi) + math.log(sigma) + float(np.sum(0.5 * (p[1:] * lj[1:] + p[:-1] * lj[:-1]) * np.diff(x)))


def test_flag_off_is_the_old_arithmetic():
    padded, act, mu, log_std, z = _inputs()
    for pe in (1.0, 0.0, 0.5):
        logp, ent = logprob_entropy_view(padded, act, mu, log_std, z, pe)
        logp2, ent2 = logprob_entropy_view(padded, act, mu, log_std, z, pe, squash=False)
        assert torch.equal(logp, logp2) and torch.equal(ent, ent2)
        _, ent_c = logprob_entropy_padded(padded[:, N_VIEW:], act[:, N_VIEW:], None)
        h = 0.5 + 0.5 * LOG2PI + log_std
        want = ent_c + h[:-1].sum() + pe * h[-1]
        assert torch.allclose(ent, want, atol=1e-6)


def test_flag_on_is_the_squashed_entropy_and_keeps_the_log_prob():
    torch.manual_seed(1)
    B = 400000
    for m, s in ((0.0, 0.3), (0.0, 0.7), (0.0, 2.718), (0.8, 0.5), (-1.5, 1.2)):
        mu = torch.full((B, 2), m)
        ls = torch.full((2,), math.log(s))
        c = squash_entropy_corr(mu, ls)[:, 0]
        est = 0.5 + 0.5 * LOG2PI + math.log(s) + float(c.mean())
        assert abs(est - _sq_entropy_quad(m, s)) < 0.01, (m, s, est, _sq_entropy_quad(m, s))
    padded, act, mu, log_std, z = _inputs()
    logp, _ = logprob_entropy_view(padded, act, mu, log_std, z, 0.0)
    logp_sq, _ = logprob_entropy_view(padded, act, mu, log_std, z, 0.0, squash=True)
    assert torch.equal(logp, logp_sq)


def _dent_dls(s: float, squash: bool, B=200000) -> float:
    torch.manual_seed(2)
    padded, act, _, _, _ = _inputs(B=B)
    mu = torch.zeros(B, 2)
    ls = torch.tensor([math.log(s), math.log(0.3)], requires_grad=True)
    z = mu + ls.detach().exp() * torch.randn(B, 2)
    _, ent = logprob_entropy_view(padded, act, mu, ls, z, 0.0, squash=squash)
    ent.mean().backward()
    return float(ls.grad[0])


def test_gradient_turns_negative_at_the_clamp():
    assert abs(_dent_dls(2.718, False) - 1.0) < 1e-6
    assert abs(_dent_dls(0.2, False) - 1.0) < 1e-6
    assert _dent_dls(0.2, True) > 0.5            # still widens a narrow head
    assert _dent_dls(2.718, True) < -1.0         # and pulls a saturated one back
    # the squashed entropy's maximum for mu = 0 lies between 0.5 and 1.0
    hs = [_sq_entropy_quad(0.0, s) for s in np.linspace(0.3, 2.7, 25)]
    smax = float(np.linspace(0.3, 2.7, 25)[int(np.argmax(hs))])
    assert 0.5 <= smax <= 1.0, smax


def test_pitch_share_follows_pitch_ent():
    padded, act, mu, log_std, z = _inputs()
    for pe in (0.0, 1.0, 0.25):
        _, ent0 = logprob_entropy_view(padded, act, mu, log_std, z, pe)
        torch.manual_seed(3)
        _, ent1 = logprob_entropy_view(padded, act, mu, log_std, z, pe, squash=True)
        torch.manual_seed(3)
        c = squash_entropy_corr(mu, log_std)
        assert torch.allclose(ent1, ent0 + c[:, :-1].sum(-1) + pe * c[:, -1], atol=1e-6)


def test_per_row_log_sigma():
    padded, act, mu, log_std, z = _inputs(B=5)
    tv = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 1.0], [3.0, 1.0], [1.0, 1.0]])
    _, ent0 = logprob_entropy_view(padded, act, mu, log_std, z, 0.0, None, tv)
    torch.manual_seed(4)
    _, ent1 = logprob_entropy_view(padded, act, mu, log_std, z, 0.0, None, tv, squash=True)
    assert ent1.shape == ent0.shape == (5,)
    assert torch.isfinite(ent1).all() and not torch.equal(ent0, ent1)


PB = ROOT / "maps" / "surf_princessburglar_b26.bsp"
PB_FIELD = ROOT / "maps" / "surf_princessburglar_b26.goal_32.npz"
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else
       ROOT / "build" / ("surfcore.dll" if os.name == "nt" else "libsurfcore.so"))
needs_run = pytest.mark.skipif(not (PB.exists() and PB_FIELD.exists() and DLL.exists()),
                               reason="needs the built core + princessburglar and its field")

SMOKE = ["--map", str(PB), "--reward", "race", "--envs", "32",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--lidar-cell", "32", "--lidar-range", "11500", "--lidar-near", "2000",
         "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--pitch-rate", "1.33", "--teleport-fail", "--lr", "3e-4",
         "--gamma", "0.9995", "--gae", "0.95", "--clip", "0.2", "--vf", "0.5",
         "--ent", "0.005", "--n-steps", "8", "--epochs", "1",
         "--minibatches", "2", "--ep-ticks", "3000", "--time-pen", "0.001",
         "--success-bonus", "50", "--finish-k", "0", "--stall-secs", "15",
         "--race-dist", "geodesic", "--maxvel", "3500", "--train-stride", "1",
         "--yaw-adaptive", "--respawn-frac", "0.9", "--respawn-margin", "2",
         "--respawn-reservoir", "1000", "--int-coef", "0.25", "--int-view", "8",
         "--int-speed", "3", "--ckpt-every", "1e9", "--record-every", "1e12",
         "--eval-eps", "1", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold"]


def _train(run, extra, steps="1024"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-u", str(ROOT / "python" / "train_fast.py"),
                        "--run", run] + SMOKE + ["--steps", steps] + list(extra),
                       capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=1800,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


@needs_run
def test_cpu_smoke_on_and_off():
    try:
        r = _train("_t_vesq_on", ["--view-ent-squash", "1"])
        assert "--view-ent-squash 1:" in r.stdout
        cfg = json.loads((ROOT / "runs" / "_t_vesq_on" / "run.json").read_text())
        assert (cfg.get("config") or cfg).get("view_ent_squash") == 1
        r = _train("_t_vesq_off", [])
        assert "--view-ent-squash" not in r.stdout
        cfg = json.loads((ROOT / "runs" / "_t_vesq_off" / "run.json").read_text())
        assert "view_ent_squash" not in (cfg.get("config") or cfg)
    finally:
        for run in ("_t_vesq_on", "_t_vesq_off"):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _csv_rows(run):
    import csv
    with open(ROOT / "runs" / run / "progress.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    """(g) The trainer at the commit before this flag (91f9fbb, the round's base) and after
    it, flag off, on the same CPU smoke: same config, progress.csv (fps aside), eval
    trajectory bytes and final weights."""
    pre = ROOT / "python" / "_train_fast_pre_vesq.py"
    s = subprocess.run(["git", "show", "91f9fbb:python/train_fast.py"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8")
    if s.returncode != 0:
        pytest.skip("the base commit is not in this checkout's history")
    try:
        pre.write_text(s.stdout, encoding="utf-8")
        _train("_t_vesq_new", [])
        _train_script = [sys.executable, "-u", str(pre), "--run", "_t_vesq_old"]
        shutil.rmtree(ROOT / "runs" / "_t_vesq_old", ignore_errors=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
                   OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
        r = subprocess.run(_train_script + SMOKE + ["--steps", "1024"], capture_output=True,
                           text=True, env=env, cwd=str(ROOT), timeout=1800,
                           encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        a, b = ROOT / "runs" / "_t_vesq_new", ROOT / "runs" / "_t_vesq_old"
        ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
        assert ca == cb
        ra, rb = _csv_rows(a.name), _csv_rows(b.name)
        assert ra and len(ra) == len(rb) and list(ra[0]) == list(rb[0])
        for x, y in zip(ra, rb):
            for k in x:
                if k != "time/fps":
                    assert x[k] == y[k], k
        ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
        assert ta and [p.name for p in ta] == [p.name for p in tb]
        for p, q in zip(ta, tb):
            assert p.read_bytes() == q.read_bytes(), p.name
        ka = sorted(a.glob("ckpt_*.pt"))
        assert ka
        sa = torch.load(ka[-1], map_location="cpu", weights_only=False)
        sb = torch.load(b / ka[-1].name, map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        pre.unlink(missing_ok=True)
        for run in ("_t_vesq_new", "_t_vesq_old"):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
