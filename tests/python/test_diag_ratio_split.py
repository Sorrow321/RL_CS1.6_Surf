"""train_fast --diag-ratio-split (pbr6b2, 2026-10-26): a MEASUREMENT of how PPO's joint ratio
is spent across the heads. Every update scores R of its rows before its first step and after
its last and splits the joint log-ratio into the keys heads', the yaw head's and the pitch
head's parts (ratio_split_parts / ratio_split_stats). (a) the parts sum to
logprob_entropy_view's joint, with and without temperatures; (b) the dead-row bookkeeping on
synthetic ratios; (c) a trainer smoke run prints the read-out and its parts match the stored
log-probs; (d) the flag off is bit-identical to the trainer before the change; (e) TRAIN_ONLY."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import train_fast  # noqa: E402
from test_ge_minutes import _csv, _train, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def _heads(B=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_cat = train_fast.NACT - train_fast.N_VIEW
    padded = torch.randn(B, train_fast.NACT, train_fast.NPAD, generator=g)
    acts = torch.zeros(B, train_fast.NACT, dtype=torch.long)
    for h in range(train_fast.N_VIEW, train_fast.NACT):
        acts[:, h] = torch.randint(0, 2, (B,), generator=g)
    n_z = train_fast.N_VIEW
    mu = torch.randn(B, n_z, generator=g) * 0.3
    log_std = torch.tensor([-3.9, -0.7])[:n_z]
    z = mu + torch.randn(B, n_z, generator=g) * log_std.exp()
    assert n_cat > 0
    return padded, acts, mu, log_std, z


def test_the_parts_sum_to_the_joint():
    padded, acts, mu, log_std, z = _heads()
    for temp, tv in ((None, None), (1.5, None), (1.5, torch.tensor([6.0, 1.0]))):
        joint, _ = train_fast.logprob_entropy_view(padded, acts, mu, log_std, z, 0.0, temp, tv)
        k, y, p = train_fast.ratio_split_parts(padded, acts, mu, log_std, z, temp, tv)
        assert k.shape == y.shape == p.shape == joint.shape
        assert torch.allclose(k + y + p, joint, atol=1e-5)
    # the yaw part is the first Gaussian's, the pitch part the last one's
    k, y, p = train_fast.ratio_split_parts(padded, acts, mu, log_std, z)
    ls = log_std
    want_y = -0.5 * ((z[:, 0] - mu[:, 0]) / ls[0].exp()) ** 2 - ls[0] - 0.5 * math.log(2 * math.pi)
    assert torch.allclose(y, want_y, atol=1e-5)


def test_the_dead_rows():
    n = 8
    zero = torch.zeros(n)
    adv = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    before = (zero, zero, zero)
    # rows 0-1: the yaw alone moves the joint past +0.18 (adv > 0: dead, cut by the view)
    # row 2: the keys alone past the clip (dead by the keys' own ratio too)
    # row 3: inside; rows 4-5: the yaw moves the joint below -0.22 (adv < 0: dead, by the view)
    # row 6: past the clip AGAINST its advantage (adv < 0, ratio up: the gradient flows)
    # row 7: inside
    keys = torch.tensor([0.0, 0.0, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0])
    yaw = torch.tensor([0.5, 0.3, 0.0, 0.05, -0.5, -0.4, 0.6, 0.1])
    after = (keys, yaw, zero)
    r = train_fast.ratio_split_stats(before, after, adv, 0.2)
    assert r["n"] == n
    assert r["out_joint"] == pytest.approx(6 / 8)      # rows 0, 1, 2, 4, 5, 6
    assert r["dead_joint"] == pytest.approx(5 / 8)     # rows 0, 1, 2, 4, 5
    assert r["dead_keys"] == pytest.approx(1 / 8)      # row 2
    assert r["dead_view"] == pytest.approx(4 / 8)      # rows 0, 1, 4, 5
    assert r["by_view"] == pytest.approx(4 / 8)
    assert r["kl_keys"] == pytest.approx(-float(keys.mean()))
    assert r["rms_pitch"] == 0.0


@needs_pb
def test_the_trainer_prints_the_split(tmp_path):
    run = "drs_smoke"
    try:
        r = _train(run, ["--diag-ratio-split", "256", "--epochs", "2"], steps="8192")
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--diag-ratio-split 256: every update scores 256 of its rows" in out
        lines = [ln for ln in out.splitlines() if ln.startswith("ratio-split [")]
        assert len(lines) >= 2, out[-4000:]
        for ln in lines:
            assert " 256 rows: past the clip " in ln and "cut by the view" in ln
            chk = float(ln.rsplit("|parts - stored logp|", 1)[1])
            assert chk < 1e-3, ln                       # the parts are the rollout's joint
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["diag_ratio_split"] == 256
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--diag-ratio-split" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_drs.py"
    a, b = ROOT / "runs" / "drs_off_new", ROOT / "runs" / "drs_off_old"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in ((a.name, TRAIN), (b.name, pre)):
            r = _train(run, ["--record-every", "2048"], script=script, steps="4096")
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "diag_ratio_split" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        pre.unlink(missing_ok=True)
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)


def test_record_ckpt_knows_the_flag():
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    assert "diag_ratio_split" in record_ckpt.TRAIN_ONLY
