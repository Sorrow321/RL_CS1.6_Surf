"""--sil-temper: the self-imitation replay scored at the rollout's sampling temperature
(2026-10-21, autoresearch r3b2).

Under --unstuck the rollout draws the view heads from N(mu, (sigma (1 + T))^2) and the
keys from softmax(logits / (1 + T)), and PPO's ratio scores them under that tempered
measure. The shipped SIL scored its replayed rows UNTEMPERED, so a row drawn at sigma x
(1 + T) reads as a (1 + T)-sigma outlier: the imitation gradient on log_std is
-((z - mu)^2 / sigma^2 - 1) ~ -((1 + T)^2 - 1) per row on average, a steady push of sigma UP.
--sil-temper 1 passes the same temperature tensors to the same helpers, so a tempered row
is, on average, exactly as likely as the tempered draw that made it (no push on log_std);
the mean is still pulled toward the imitated action.

(a) math: rows drawn at temperature 1 + T: the untempered SIL gradient on log_std is
    ~-((1 + T)^2 - 1) per row (loss falls as sigma rises), the tempered one ~0; both pull
    mu toward the imitated z.
(b) CPU smoke on sidistic (abs view, keys-hold, unstuck with T forced up): it runs, the
    note prints, the config carries sil_temper, a flagless resume restores it, and
    --sil-temper without --sil-coef is refused.
(c) flag off: --sil-temper absent leaves the config dump without the key (the SIL path
    itself is the shipped call: SIL_TEMP is None).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_fast import N_VIEW, NVEC, HeadPacker, logprob_entropy_view   # noqa: E402
from test_ez_viewc import SMOKE, TRAIN, _run, _train, needs_run          # noqa: E402

CPU = torch.device("cpu")


def _sil_grads(T: float, tempered: bool, n: int = 20000, seed: int = 0):
    """d(-logp)/d(log_std) and d(-logp)/d(mu) per row, averaged, for rows whose z was
    drawn at temperature 1 + T around mu = 0 with sigma = exp(log_std)."""
    g = torch.Generator().manual_seed(seed)
    padded = HeadPacker(CPU).pad(torch.zeros(n, sum(NVEC)))
    acts = torch.zeros((n, len(NVEC)), dtype=torch.long)
    log_std = torch.full((N_VIEW,), -2.5, requires_grad=True)
    mu = torch.zeros((n, N_VIEW), requires_grad=True)
    sig = float(torch.exp(log_std[0].detach()))
    eps = torch.randn((n, N_VIEW), generator=g)
    z = (0.3 + (1.0 + T) * sig * eps).detach()         # imitated rows sit around +0.3
    t = torch.full((), 1.0 + T) if tempered else None
    tv = torch.full((N_VIEW,), 1.0 + T) if tempered else None
    logp, _ = logprob_entropy_view(padded, acts, mu, log_std, z, 1.0, t, tv)
    (-logp.mean()).backward()
    return log_std.grad.detach(), mu.grad.detach().sum(0)


def test_untempered_rows_push_log_std_up_and_tempered_rows_do_not():
    T = 1.0
    g_un, m_un = _sil_grads(T, tempered=False)
    g_te, m_te = _sil_grads(T, tempered=True)
    # z is drawn around +0.3, so the per-head mean squared deviation includes the offset;
    # the noise part alone is (1 + T)^2 sigma^2. Untempered: loss gradient on log_std is
    # strongly NEGATIVE (gradient descent raises sigma) - at least the noise's share.
    assert torch.all(g_un < -((1.0 + T) ** 2 - 1.0) * 0.95), g_un
    # tempered: the noise share cancels; what is left is the offset's pull (0.3 / ((1+T)
    # sigma))^2, which with sigma 0.082 is large too - so check the noise share alone
    # below, at zero offset
    assert torch.all(g_te > g_un), (g_te, g_un)
    # both pull mu toward the imitated z (> 0): the loss gradient on mu is negative
    assert torch.all(m_un < 0.0) and torch.all(m_te < 0.0)
    # the tempered pull is (1 + T)^2 x weaker per row
    assert torch.allclose(m_un / m_te, torch.full_like(m_un, (1.0 + T) ** 2), rtol=1e-3)


def test_tempered_rows_at_zero_offset_leave_log_std_alone():
    """With the imitated rows centred on the mean (pure sampling noise), the tempered
    score has no net push on log_std; the untempered one pushes it up by ~(1+T)^2 - 1."""
    for T in (0.5, 1.0, 2.0):
        g = torch.Generator().manual_seed(1)
        n = 50000
        padded = HeadPacker(CPU).pad(torch.zeros(n, sum(NVEC)))
        acts = torch.zeros((n, len(NVEC)), dtype=torch.long)
        mu = torch.zeros((n, N_VIEW))
        for tempered in (False, True):
            log_std = torch.full((N_VIEW,), -2.5, requires_grad=True)
            sig = float(torch.exp(log_std[0].detach()))
            z = (1.0 + T) * sig * torch.randn((n, N_VIEW), generator=g)
            t = torch.full((), 1.0 + T) if tempered else None
            tv = torch.full((N_VIEW,), 1.0 + T) if tempered else None
            logp, _ = logprob_entropy_view(padded, acts, mu, log_std, z, 1.0, t, tv)
            (-logp.mean()).backward()
            gr = log_std.grad
            if tempered:
                assert torch.all(gr.abs() < 0.05), (T, gr)
            else:
                want = -((1.0 + T) ** 2 - 1.0)
                assert torch.allclose(gr, torch.full_like(gr, want), atol=0.08), (T, gr)


# ---------------------------------------------------------------- (b) the smoke
UNSTUCK_HOT = ["--unstuck", "--unstuck-reach", "eval", "--unstuck-patience", "0",
               "--unstuck-period", "2048", "--unstuck-rate", "1", "--unstuck-max", "1"]
SIL = ["--sil-coef", "0.1", "--sil-batch-size", "64", "--sil-buffer", "2000",
       "--sil-batches", "2"]


@needs_run
def test_sil_temper_smoke_runs_and_resumes():
    run = "siltemper_on"
    r = _train(run, UNSTUCK_HOT + SIL + ["--sil-temper", "1"], steps="10240")
    assert "--sil-temper 1: buffer rows scored at the rollout's sampling temperature " \
           "(the --unstuck T tensors PPO's ratio reads)" in r.stdout, r.stdout[-3000:]
    assert "  sil " in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["sil_temper"] == 1 and cfg["sil_coef"] == 0.1 and cfg["unstuck"]
    run2 = "siltemper_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "12288"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "sil_temper=1" in r2.stdout and "--sil-temper 1:" in r2.stdout
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
def test_sil_temper_needs_sil_coef_and_is_absent_from_the_dump_when_off():
    r = _run([sys.executable, "-u", str(TRAIN), "--run", "siltemper_bad"] + SMOKE
             + ["--steps", "2048", "--sil-temper", "1"])
    assert r.returncode != 0 and "--sil-temper 1 tempers the --sil-coef replay" in \
        (r.stdout + r.stderr)
    shutil.rmtree(ROOT / "runs" / "siltemper_bad", ignore_errors=True)
    run = "siltemper_off"
    _train(run, SIL, steps="4096")
    cfg = json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))["config"]
    assert "sil_temper" not in cfg and cfg["sil_coef"] == 0.1
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


# ------------------------------------------- (c) flag-off identity vs the pre-flag trainer
def _pre_flag_trainer() -> Path | None:
    """The newest committed train_fast.py WITHOUT --sil-temper, materialised beside the
    live one so its relative imports resolve; None when git cannot serve one."""
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-n", "40", "--",
                        "python/train_fast.py"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    for ref in r.stdout.split():
        s = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if s.returncode == 0 and s.stdout and "--sil-temper" not in s.stdout:
            p = ROOT / "python" / "_train_fast_pre_sil_temper.py"
            p.write_text(s.stdout, encoding="utf-8")
            return p
    return None


@needs_run
def test_sil_without_temper_is_bit_identical_to_the_pre_flag_trainer():
    """--sil-coef ON, --sil-temper absent: the tempered branch is never taken
    (SIL_TEMP is None, the shipped helper call), so the config dump and every
    non-timing progress.csv column match the trainer before the flag existed."""
    old = _pre_flag_trainer()
    if old is None:
        pytest.skip("no pre-flag train_fast.py in the history")
    a, b = "siltemper_id_now", "siltemper_id_old"
    try:
        _train(a, UNSTUCK_HOT + SIL, steps="6144")
        _train(b, UNSTUCK_HOT + SIL, script=old, steps="6144")
        ca = json.loads((ROOT / "runs" / a / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((ROOT / "runs" / b / "run.json").read_text(encoding="utf-8"))["config"]
        assert ca == cb, "the flag-off config dump changed"
        from test_ez_viewc import _csv
        ra, rb = _csv(a), _csv(b)
        assert list(ra[0]) == list(rb[0]), "the flag-off CSV header changed"
        assert len(ra) == len(rb) and len(ra) > 0
        for xa, xb in zip(ra, rb):
            for k in xa:
                if k.startswith("time/"):
                    continue
                assert xa[k] == xb[k], (k, xa[k], xb[k])
    finally:
        old.unlink(missing_ok=True)
        for dd in (a, b):
            shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)
