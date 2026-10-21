"""--sil-burst-only / --sil-view-mse (2026-10-21, autoresearch r3b4).

Exploration by held random bursts (--ez-eps, --ez-front) and consolidation by
self-imitation (--sil-coef) of exactly the bursts that beat the critic. PPO
drops the burst rows (b_ez), and its clipped ratio could not learn from a
far-off held action anyway (pi_new / pi_old ~ 0 out in the tail), so before
this change a burst that found a better line changed the reservoir but never
the policy.

(a) SILBuffer.add(mask=...): only masked rows with R > V are kept; mask None is
    the old behaviour.
(b) CPU smoke on sidistic (abs view, keys-hold): bursts + SIL burst-only + the
    bounded view loss run, the buffer fills with burst rows, the config carries
    both keys, and a flagless resume restores them; the refusals fire.
(c) Identity: the trainer before this change and after it give the same config,
    progress.csv (fps aside), eval trajectory bytes and weights - with SIL off,
    and with SIL on without the two new flags (the SIL code paths were touched).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surfgym.sil import SILBuffer                                   # noqa: E402
from test_ez_viewc import SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402

DEV = torch.device("cpu")
SIL_SMALL = ["--sil-coef", "0.1", "--sil-batch-size", "64", "--sil-buffer", "2000",
             "--sil-batches", "2"]


# ------------------------------------------------------------- (a) the buffer
def test_buffer_mask_keeps_only_masked_rows_that_beat_the_critic():
    g = torch.Generator().manual_seed(0)
    n = 6
    scal, img = torch.randn((n, 4), generator=g), torch.rand((n, 8), generator=g)
    act, z = torch.randint(0, 3, (n, 6), generator=g), torch.randn((n, 2), generator=g)
    ret = torch.tensor([1.0, 0.5, 2.0, 0.0, 3.0, 1.0])
    val = torch.tensor([0.5, 0.5, 2.5, 0.0, 1.0, 0.0])
    # gains: +0.5, 0, -0.5, 0, +2, +1 -> rows 0, 4, 5 qualify; the mask drops row 4
    mask = torch.tensor([True, True, True, False, False, True])
    b = SILBuffer(100, DEV, torch.float32, 4, 8, 6, nz=2, seed=1)
    assert b.add(scal, img, act, ret, val, z=z, mask=mask) == 2 and b.size == 2
    assert b.ret[:2].tolist() == [1.0, 1.0]
    assert torch.equal(b.z[:2], z[[0, 5]])
    nb = SILBuffer(100, DEV, torch.float32, 4, 8, 6, nz=2, seed=1)
    assert nb.add(scal, img, act, ret, val, z=z) == 3          # mask None: the old rule
    assert nb.ret[:3].tolist() == [1.0, 3.0, 1.0]


# ---------------------------------------------------------------- (b) the smoke
@needs_run
def test_burst_sil_runs_fills_the_buffer_with_burst_rows_and_resumes():
    run = "silburst_on"
    r = _train(run, ["--ez-eps", "0.3", "--sil-burst-only", "1", "--sil-view-mse", "1"]
               + SIL_SMALL)
    assert "--sil-burst-only: only BURST rows" in r.stdout
    assert "--sil-view-mse: the view heads are imitated" in r.stdout
    kept = [int(m) for m in re.findall(r"  sil [0-9,]+ rows \+([0-9]+)", r.stdout)]
    assert kept and sum(kept) > 0, r.stdout[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["sil_burst_only"] == 1 and cfg["sil_view_mse"] == 1 and cfg["sil_coef"] == 0.1
    rows = _csv(run)
    assert int(rows[-1]["sil/buffer"]) > 0
    assert any(x["sil/loss"] not in ("", "nan") for x in rows)
    run2 = "silburst_on_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "sil_burst_only=1" in r2.stdout and "sil_view_mse=1" in r2.stdout
    assert "--sil-burst-only: only BURST rows" in r2.stdout
    for dd in (d, ROOT / "runs" / run2):
        shutil.rmtree(dd, ignore_errors=True)


@needs_run
@pytest.mark.parametrize("extra,msg", [
    (["--sil-burst-only", "1", "--ez-eps", "0.3"], "without --sil-coef"),
    (["--sil-burst-only", "1"] + SIL_SMALL, "no burst rows to imitate"),
    (["--ez-eps", "0.3"] + SIL_SMALL, "without --sil-burst-only"),
])
def test_refusals(extra, msg):
    run = "silburst_bad"
    r = _run([sys.executable, "-u", str(TRAIN), "--run", run] + SMOKE
             + ["--steps", "2048"] + extra)
    assert r.returncode != 0 and msg in (r.stdout + r.stderr), (r.stdout + r.stderr)[-2000:]
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


# ------------------------------------------------------------ (c) the identity
def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before this change."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--sil-burst-only" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


def _same(a: Path, b: Path):
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb, "config dump changed"
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


@needs_run
@pytest.mark.parametrize("tag,extra", [("off", []), ("silon", SIL_SMALL)])
def test_identity_with_the_trainer_before_the_change(tag, extra):
    pre = ROOT / "python" / f"_train_fast_pre_silburst_{tag}.py"
    a, b = ROOT / "runs" / f"silburst_{tag}_new", ROOT / "runs" / f"silburst_{tag}_old"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train(a.name, extra)
        _train(b.name, extra, script=pre)
    finally:
        pre.unlink(missing_ok=True)
    _same(a, b)
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
