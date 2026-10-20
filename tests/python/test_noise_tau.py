"""--noise-tau: AR(1)-correlated rollout sampling noise (train_fast.py,
sample_padded_ar / sample_view_ar).

What is pinned here:

(a) gumbel_from_normal turns a standard normal into a standard Gumbel
    (mean = Euler's gamma, variance = pi^2 / 6), with finite tails.
(b) ar1_step_ keeps the latent N(0, 1) and gives it lag-1 correlation rho.
(c) THE PER-DECISION DISTRIBUTION IS THE POLICY'S: the categorical draw's
    empirical frequencies match softmax(logits / temp) along one long
    correlated chain, and the view draw's z is N(mu, sigma * temp).
(d) The returned log-prob is exactly logprob_entropy_view's recomputation
    of the stored action (so the update needs nothing new).
(e) The draws PERSIST: consecutive actions agree far more often than under
    i.i.d. sampling, and long off-mode runs exist that i.i.d. never makes.
(f) The trainer: with the flag absent the run is bit-identical to the base
    commit's trainer (config, progress.csv minus fps, eval trajectory,
    weights); with the flag the run trains, the config records noise_tau,
    and the startup line names rho.

    python -m pytest tests/python/test_noise_tau.py -q
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from train_fast import (N_VIEW, NACT, NPAD, NEG, ar1_step_,     # noqa: E402
                        gumbel_from_normal, logprob_entropy_view,
                        sample_padded_ar, sample_view_ar)

EULER = 0.5772156649015329


def test_gumbel_from_normal_is_standard_gumbel():
    g = gumbel_from_normal(torch.randn(2_000_000, generator=torch.Generator().manual_seed(1)))
    assert torch.isfinite(g).all()
    assert abs(float(g.mean()) - EULER) < 5e-3
    assert abs(float(g.var()) - math.pi ** 2 / 6) < 1e-2
    # the tails stay finite where the naive -log(-log(Phi)) would not
    t = gumbel_from_normal(torch.tensor([-12.0, 0.0, 9.0, 40.0]))
    assert torch.isfinite(t).all() and bool((t[1:] > t[:-1]).all())


def test_ar1_is_stationary_with_lag1_rho():
    torch.manual_seed(2)
    rho = 0.9
    e = torch.randn(20_000)
    prev = e.clone()
    a, b = [], []
    for _ in range(50):
        ar1_step_(e, rho)
        a.append(prev.clone())
        b.append(e.clone())
        prev.copy_(e)
    a, b = torch.cat(a), torch.cat(b)
    assert abs(float(b.mean())) < 0.01 and abs(float(b.std()) - 1.0) < 0.01
    assert abs(float(torch.corrcoef(torch.stack([a, b]))[0, 1]) - rho) < 0.01


def _padded(B, logits_list):
    p = torch.full((B, len(logits_list), NPAD), NEG)
    for h, lg in enumerate(logits_list):
        p[:, h, :len(lg)] = torch.tensor(lg)
    return p


@pytest.mark.parametrize("temp", [None, 2.0])
def test_categorical_marginal_is_softmax_and_draws_persist(temp):
    torch.manual_seed(3)
    B, steps, rho = 256, 400, math.exp(-0.04 / 0.5)
    heads = [[2.0, 0.0, -1.0, 0.5], [0.0, 1.5], [1.0, 1.0, -0.5]]
    padded = _padded(B, heads)
    e = torch.randn(B, len(heads), NPAD)
    acts = []
    for _ in range(steps):
        a, _lp = sample_padded_ar(padded, e, rho,
                                  None if temp is None else torch.tensor(temp))
        acts.append(a)
    acts = torch.stack(acts)                                  # (T, B, H)
    t = 1.0 if temp is None else temp
    for h, lg in enumerate(heads):
        want = F.softmax(torch.tensor(lg) / t, -1)
        got = torch.bincount(acts[..., h].flatten(), minlength=len(lg)).float()
        got = got[:len(lg)] / got.sum()
        assert torch.allclose(got, want, atol=0.02), (h, got, want)
        assert int(acts[..., h].max()) < len(lg)              # NEG slots never drawn
    # persistence: P(a_t == a_t-1) under AR(1) vs under i.i.d. at the same marginal
    same = (acts[1:] == acts[:-1]).float().mean(dim=(0, 1))
    for h, lg in enumerate(heads):
        p = F.softmax(torch.tensor(lg) / t, -1)
        iid = float((p * p).sum())
        assert float(same[h]) > iid + 0.5 * (1.0 - iid), (h, float(same[h]), iid)


def test_long_off_mode_runs_exist():
    """A head that puts 0.9 on its mode: i.i.d. sampling makes a 25-decision
    run of the OTHER bin with probability 0.1^25; the correlated draw makes
    plenty at tau = 1 s (rho per 40 ms decision)."""
    torch.manual_seed(4)
    B, steps, rho = 512, 500, math.exp(-0.04 / 1.0)
    lg = [math.log(0.9), math.log(0.1)]
    padded = _padded(B, [lg])
    e = torch.randn(B, 1, NPAD)
    off = []
    for _ in range(steps):
        a, _ = sample_padded_ar(padded, e, rho)
        off.append(a[:, 0] == 1)
    off = torch.stack(off).numpy()                            # (T, B)
    best = 0
    for b in range(B):
        run = 0
        for v in off[:, b]:
            run = run + 1 if v else 0
            best = max(best, run)
    assert abs(off.mean() - 0.1) < 0.02
    assert best >= 25, best


def test_per_env_rho_pair_correlates_only_the_chosen_envs():
    """--noise-tau-frac: a (rho, innovation) tensor pair. rho-0 envs draw a
    fresh N(0, 1) every step (lag-1 correlation 0), the others AR(1)."""
    torch.manual_seed(6)
    B, rho = 4000, 0.9
    r = torch.zeros(B)
    r[::2] = rho
    pair = (r.view(B, 1), torch.sqrt(1.0 - r * r).view(B, 1))
    e = torch.randn(B, 3)
    xs = []
    for _ in range(60):
        xs.append(ar1_step_(e, pair).clone())
    xs = torch.stack(xs)                                      # (T, B, 3)
    assert abs(float(xs.std()) - 1.0) < 0.02
    a, b = xs[:-1], xs[1:]
    c_corr = float(torch.corrcoef(torch.stack([a[:, ::2].flatten(), b[:, ::2].flatten()]))[0, 1])
    c_iid = float(torch.corrcoef(torch.stack([a[:, 1::2].flatten(), b[:, 1::2].flatten()]))[0, 1])
    assert abs(c_corr - rho) < 0.02 and abs(c_iid) < 0.02


def test_view_draw_marginal_and_logprob_match_the_update():
    torch.manual_seed(5)
    B, rho = 4096, 0.95
    NZ = 2
    heads = [[0.0] * 15, [0.0] * 7, [1.0, 0.0, -1.0, 0.2], [0.3, 0.1, -0.2, 0.0],
             [0.0, 0.4], [0.0, -0.3, 0.2]]
    assert len(heads) == NACT
    padded = _padded(B, heads)
    mu = torch.randn(B, NZ) * 0.3
    log_std = torch.tensor([-2.0, -1.0])
    temp = torch.tensor(1.5)
    tempv = torch.tensor([1.5, 1.5])
    e_cat = torch.randn(B, NACT - N_VIEW, NPAD)
    e_view = torch.randn(B, NZ)
    zs = []
    for _ in range(30):
        act, z, logp = sample_view_ar(padded, mu, log_std, e_cat, e_view, rho,
                                      temp, tempv)
        lp2, _ = logprob_entropy_view(padded, act, mu, log_std, z,
                                      temp=temp, temp_view=tempv)
        assert torch.allclose(logp, lp2, atol=1e-5)
        assert int(act[:, 0].max()) == int(act[:, 0].min())  # neutral view bins
        zs.append((z - mu) / (log_std.exp() * 1.5))
    zs = torch.stack(zs)
    assert abs(float(zs.mean())) < 0.02 and abs(float(zs.std()) - 1.0) < 0.02


# ==========================================================================
# (f): the trainer, flag off = the base commit's trainer, flag on = trains
# ==========================================================================
_env_dll = os.environ.get("SURFCORE_DLL")
DLL = (Path(_env_dll) if _env_dll else ROOT / "build" / "libsurfcore.so")
MAPS = Path(os.environ.get("SURF_TEST_MAPS") or (ROOT / "maps"))
SID = MAPS / "surf_src_sidistic.bsp"
needs_run = pytest.mark.skipif(
    not (SID.exists() and DLL.exists()
         and (MAPS / "surf_src_sidistic.goal_32.npz").exists()),
    reason="needs the built core + sidistic + its prebaked goal field")
BASE = os.environ.get("NOISE_TAU_BASE", "877a2a626dc9003ec300342e93ee793fb1da389d")

SMOKE = ["--map", str(SID), "--reward", "race", "--envs", "32",
         "--spawn", "platform", "--lidar-w", "16", "--lidar-h", "8",
         "--emb", "64", "--hidden", "64", "--act-every", "4",
         "--n-steps", "8", "--epochs", "1", "--minibatches", "2",
         "--ep-ticks", "3000", "--respawn-frac", "0.9", "--respawn-margin", "2",
         "--respawn-reservoir", "1000", "--int-coef", "0.25", "--int-view", "8",
         "--int-speed", "3", "--ckpt-every", "1e9", "--record-every", "1e12",
         "--eval-eps", "1", "--eval-greedy-only", "--seed", "7",
         "--view-continuous", "--view-absolute", "velocity", "--keys-hold",
         "--no-compile", "--no-graphs"]


def _env():
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8",
             OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
    if _env_dll:
        e["SURFCORE_DLL"] = _env_dll
    return e


def _train(run, extra, script="train_fast.py", steps="2048"):
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    cmd = [sys.executable, "-u", str(ROOT / "python" / script), "--run", run,
           "--steps", steps] + SMOKE + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=_env(),
                       cwd=str(ROOT), timeout=1800, encoding="utf-8",
                       errors="replace")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r


def _rows(run):
    rows = (ROOT / "runs" / run / "progress.csv").read_text(
        encoding="utf-8").splitlines()
    head = rows[0].split(",")
    return [{k: v for k, v in zip(head, r.split(",")) if k != "time/fps"}
            for r in rows[1:]]


@needs_run
def test_trainer_flag_off_is_the_base_trainer_and_flag_on_trains():
    old = ROOT / "python" / "train_fast_prenoise.py"
    r = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"],
                       capture_output=True, cwd=str(ROOT))
    if r.returncode != 0:
        pytest.skip(f"base commit {BASE} not in this clone")
    old.write_bytes(r.stdout)
    try:
        _train("nt_ctl_new", [])
        _train("nt_ctl_old", [], script="train_fast_prenoise.py")
        on = _train("nt_on", ["--noise-tau", "0.5"])
        frac = _train("nt_frac", ["--noise-tau", "0.5", "--noise-tau-frac", "0.25"])
    finally:
        old.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "nt_ctl_new", ROOT / "runs" / "nt_ctl_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert "noise_tau" not in ca and ca == cb
    assert _rows("nt_ctl_new") == _rows("nt_ctl_old")
    ta = sorted(a.glob("traj_*.jsonl"))
    tb = sorted(b.glob("traj_*.jsonl"))
    assert [p.name for p in ta] == [p.name for p in tb]
    for pa, pb in zip(ta, tb):
        assert pa.read_bytes() == pb.read_bytes()
    ka = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
    kb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for k, v in ka["policy"].items():
        assert torch.equal(v, kb["policy"][k]), k
    # flag on: trains, records the key, and the draw differs from the control
    con = json.loads((ROOT / "runs" / "nt_on" / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert con["noise_tau"] == 0.5
    assert "--noise-tau 0.5 s" in on.stdout and "rho 0.9231" in on.stdout
    assert _rows("nt_on") != _rows("nt_ctl_new")
    assert "noise_tau_frac" not in con
    cf = json.loads((ROOT / "runs" / "nt_frac" / "run.json").read_text(
        encoding="utf-8"))["config"]
    assert cf["noise_tau"] == 0.5 and cf["noise_tau_frac"] == 0.25
    assert "--noise-tau-frac 0.25: 8 of 32 envs draw AR(1) noise" in frac.stdout
    for run in ("nt_ctl_new", "nt_ctl_old", "nt_on", "nt_frac"):
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
