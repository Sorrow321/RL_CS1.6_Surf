"""--sil-coef: self-imitation learning as a default-off auxiliary loss (docs/sil.md).

(a) SILBuffer: only rows with R > V are kept; more than the capacity keeps
    the largest gains; FIFO eviction; priority sampling follows the gain;
    refresh rewrites the gain of the sampled rows; the extra columns (z,
    priv) ride along.
(b) sil_loss_terms: zero when R <= V; the policy term is -logp x gap with the
    gap detached; the value term's gradient pulls V UP toward R and is zero
    below R.
(c) CPU smoke (cannonball toy set): --sil-coef runs, the config carries the
    keys, progress.csv gains sil/buffer, sil/mean_gain, sil/loss, the buffer
    is non-empty after a few iterations, the step line prints the note, and
    a flagless resume restores the knobs.
(d) flag-off identity: the trainer of THIS tree with --sil-coef absent
    reproduces the parent commit's trainer on the toy set (same seed),
    every progress.csv column but the timing ones, and the config dump.
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.sil import SILBuffer, sil_loss_terms                # noqa: E402
from test_unstuck import ABS, TRAIN, _csv, _run, _train           # noqa: E402
from test_view_continuous import SMOKE_FLAGS, needs_run           # noqa: E402

DEV = torch.device("cpu")


def _rows(n, ret, val, seed=0):
    g = torch.Generator().manual_seed(seed)
    scal = torch.randn((n, 4), generator=g)
    img = torch.rand((n, 8), generator=g)
    act = torch.randint(0, 3, (n, 6), generator=g)
    z = torch.randn((n, 2), generator=g)
    return scal, img, act, torch.as_tensor(ret, dtype=torch.float32), \
        torch.as_tensor(val, dtype=torch.float32), z


# ------------------------------------------------------------- (a) the buffer
def test_buffer_keeps_only_rows_whose_return_beats_the_critic():
    b = SILBuffer(100, DEV, torch.float32, 4, 8, 6, nz=2, seed=1)
    scal, img, act, ret, val, z = _rows(6, [1.0, 0.5, 2.0, 0.0, 3.0, 1.0],
                                        [0.5, 0.5, 2.5, 0.0, 1.0, 2.0])
    n = b.add(scal, img, act, ret, val, z=z)
    assert n == 2 and b.size == 2                     # rows 0 and 4
    assert b.ret[:2].tolist() == [1.0, 3.0]
    assert b.gain[:2].tolist() == pytest.approx([0.5, 2.0])
    assert torch.equal(b.act[:2], act[[0, 4]]) and torch.equal(b.z[:2], z[[0, 4]])


def test_buffer_topk_when_more_than_capacity_qualify_and_fifo_eviction():
    b = SILBuffer(3, DEV, torch.float32, 4, 8, 6, seed=1)
    scal, img, act, ret, val, _ = _rows(5, [5, 4, 3, 2, 1], [0, 0, 0, 0, 0])
    assert b.add(scal, img, act, ret, val) == 3           # the three largest gains
    assert sorted(b.ret[:3].tolist()) == [3.0, 4.0, 5.0]
    scal, img, act, ret, val, _ = _rows(1, [9.0], [0.0], seed=2)
    b.add(scal, img, act, ret, val)                        # evicts the oldest slot
    assert b.size == 3 and 9.0 in b.ret[:3].tolist()
    assert b.head == 1


def test_priority_sampling_follows_the_gain_and_refresh_rewrites_it():
    b = SILBuffer(10, DEV, torch.float32, 4, 8, 6, seed=3)
    scal, img, act, ret, val, _ = _rows(2, [10.0, 1.001], [0.0, 1.0])
    b.add(scal, img, act, ret, val)                        # gains 10 and 0.001
    idx = b.sample(2000)
    frac_big = float((idx == 0).float().mean())
    assert frac_big > 0.98
    b.refresh(torch.tensor([0]), torch.tensor([-1.0]))     # V rose past R for row 0
    idx = b.sample(2000)
    assert float((idx == 1).float().mean()) > 0.98         # the other row now dominates
    assert b.mean_gain() == pytest.approx(0.0005, abs=1e-6)
    assert b.sample(5).numel() == 5 and SILBuffer(4, DEV, torch.float32, 4, 8, 6).sample(3).numel() == 0


# ----------------------------------------------------------------- (b) the loss
def test_loss_is_zero_below_the_critic_and_the_value_term_pulls_v_up():
    logp = torch.tensor([-1.0, -2.0])
    ent = torch.tensor([0.5, 0.5])
    v = torch.tensor([2.0, 2.0], requires_grad=True)
    loss, raw = sil_loss_terms(logp, ent, v, torch.tensor([1.0, 2.0]))
    assert float(loss) == 0.0 and raw.tolist() == [-1.0, 0.0]
    v2 = torch.tensor([1.0, 3.0], requires_grad=True)
    loss2, raw2 = sil_loss_terms(logp, ent, v2, torch.tensor([3.0, 2.0]))
    # row 0: gain 2 -> pg = -(-1) x 2 / 2 = 1.0 ; vl = 0.5 x 4 / 2 = 1.0 ; row 1 contributes 0
    assert float(loss2) == pytest.approx(2.0)
    loss2.backward()
    assert v2.grad[0] < 0 and v2.grad[1] == 0          # descending the loss RAISES V(row 0)
    assert raw2.tolist() == [2.0, -1.0]
    loss3, _ = sil_loss_terms(logp, ent, v2.detach(), torch.tensor([3.0, 2.0]), ent_coef=1.0)
    assert float(loss3) == pytest.approx(2.0 - 0.5)


# ---------------------------------------------------------------- (c) the smoke
@needs_run
def test_sil_smoke_runs_logs_and_resumes():
    chk = _run([sys.executable, "-c",
                "import torch; assert not torch.cuda.is_available()"], 300)
    assert chk.returncode == 0, chk.stderr
    run = "cya_sil"
    r = _train(run, ABS + ["--sil-coef", "0.1", "--sil-batch-size", "64",
                           "--sil-buffer", "2000", "--sil-batches", "2",
                           "--ep-ticks", "48"], steps="10240")
    assert "--sil-coef 0.1: buffer 2,000 rows, 2 x 64 rows per epoch" in r.stdout
    assert "  sil " in r.stdout
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["sil_coef"] == 0.1 and cfg["sil_buffer"] == 2000 \
        and cfg["sil_batches"] == 2 and cfg["sil_batch_size"] == 64 and cfg["sil_ent"] == 0.0
    rows = _csv(run)
    head = list(rows[0])
    i = head.index("sil/buffer")
    assert head[i:i + 3] == ["sil/buffer", "sil/mean_gain", "sil/loss"]
    assert int(rows[-1]["sil/buffer"]) > 0
    assert any(x["sil/loss"] not in ("", "nan") for x in rows)
    run2 = "cya_sil_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE_FLAGS + ["--steps", "12288"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "sil_coef=0.1" in r2.stdout and "sil_batch_size=64" in r2.stdout
    rows2 = _csv(run2)
    assert "sil/buffer" in rows2[0]
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


# --------------------------------------------------- (d) flag-off bit identity
def _parent_trainer() -> Path | None:
    """The parent commit's train_fast.py, materialised beside the live one
    so its relative imports resolve; None when git cannot serve it."""
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD~1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    ref = r.stdout.strip()
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:python/train_fast.py"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not r.stdout:
        return None
    p = ROOT / "python" / "_train_fast_parent_for_sil_test.py"
    p.write_text(r.stdout, encoding="utf-8")
    return p


@needs_run
def test_flag_off_is_bit_identical_to_the_parent_commit_trainer():
    parent = _parent_trainer()
    if parent is None:
        pytest.skip("git could not serve the parent commit's trainer")
    try:
        a = _train("cya_sil_off_now", ABS + ["--ep-ticks", "48"], steps="8192")
        b = _train("cya_sil_off_par", ABS + ["--ep-ticks", "48"], script=parent, steps="8192")
        ca = json.loads((ROOT / "runs" / "cya_sil_off_now" / "run.json").read_text(encoding="utf-8"))["config"]
        cb = json.loads((ROOT / "runs" / "cya_sil_off_par" / "run.json").read_text(encoding="utf-8"))["config"]
        assert ca == cb, "the flag-off config dump changed"
        ra, rb = _csv("cya_sil_off_now"), _csv("cya_sil_off_par")
        assert list(ra[0]) == list(rb[0]), "the flag-off CSV header changed"
        skip = {"time/fps", "time/elapsed_secs", "time/wall", "time/iter_s"}
        for xa, xb in zip(ra, rb):
            for k in xa:
                if k in skip or k.startswith("time/"):
                    continue
                assert xa[k] == xb[k], (k, xa[k], xb[k])
    finally:
        parent.unlink(missing_ok=True)
        for dd in ("cya_sil_off_now", "cya_sil_off_par"):
            shutil.rmtree(ROOT / "runs" / dd, ignore_errors=True)
