"""--sil-tempered 1 (2026-10-21, autoresearch r3b3): under --unstuck the SIL buffer's actions
are scored under the CURRENT sampling temperature (sigma x (1+T), logits / (1+T)) - the
distribution the rollout drew them from - instead of the untempered policy, so imitating a
tempered sample moves the mean without inflating sigma or flattening the keys.

* CPU smoke (sidistic, abs view, keys-hold, unstuck): the flag runs with --sil-coef, says so,
  the config carries it and a flagless resume restores it;
* without --sil-coef it is refused; the recorder lists it as training-only;
* the tempered score is the behaviour policy's: for a Gaussian head, scoring z under
  sigma x tau gives exactly log N(z; mu, (sigma tau)^2), whose sigma-gradient vanishes in
  expectation over z ~ N(mu, (sigma tau)^2) while the untempered score's is positive.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ez_viewc import SMOKE, TRAIN, _run, needs_run           # noqa: E402


def test_the_tempered_gaussian_score_does_not_push_sigma_up():
    torch.manual_seed(0)
    mu, tau = 0.3, 3.0
    log_std = torch.tensor(math.log(0.05), requires_grad=True)
    z = mu + 0.05 * tau * torch.randn(200000)

    def nll(ls):
        return (0.5 * ((z - mu) / ls.exp()).pow(2) + ls).mean()
    g_un, = torch.autograd.grad(nll(log_std), log_std)
    g_te, = torch.autograd.grad(nll(log_std + math.log(tau)), log_std)
    # minimising the untempered NLL of tempered samples RAISES sigma (gradient < 0 on the loss)
    assert g_un.item() < -5.0
    assert abs(g_te.item()) < 0.02


def test_the_recorder_treats_the_flag_as_training_only():
    src = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"sil_tempered"' in src


@needs_run
def test_tempered_sil_runs_under_unstuck_and_resumes():
    run = "siltemp_on"
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "6144", "--unstuck", "--unstuck-reach", "eval",
                "--sil-coef", "0.1", "--sil-tempered", "1"])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "--sil-tempered: buffer rows are scored under the CURRENT sampling" in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["sil_tempered"] == 1 and cfg["sil_coef"] == 0.1
    run2 = "siltemp_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "sil_tempered=1" in r2.stdout and "--sil-tempered: buffer rows" in r2.stdout
    bad = _run([sys.executable, "-u", str(TRAIN), "--run", "siltemp_bad"] + SMOKE
               + ["--steps", "2048", "--sil-tempered", "1"])
    assert bad.returncode != 0 and "--sil-tempered needs --sil-coef > 0" in bad.stdout + bad.stderr
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / "siltemp_bad"):
        shutil.rmtree(dd, ignore_errors=True)
