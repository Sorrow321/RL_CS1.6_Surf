"""train_fast --ppo-head-clip (pbr6b2, 2026-10-26): PPO's clipped surrogate per head group (keys /
yaw / pitch), each on its own ratio. --diag-ratio-split measured the joint ratio's clip spent ~90%
by the view heads, cutting the keys' gradient on ~14% of each update's rows. (a) on-policy the
group surrogate's gradient is the joint one's; (b) a group past the clip goes flat alone, where
the joint ratio switches every group off; (c) a trainer smoke run with the flag (and with the
diagnostic beside it); (d) the flag off is bit-identical to the trainer before the change;
(e) TRAIN_ONLY."""
from __future__ import annotations

import json
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


def _joint(a, lp_new, lp_old, clip):
    r = torch.exp(lp_new.sum(-1) - lp_old.sum(-1))
    return torch.max(-a * r, -a * torch.clamp(r, 1 - clip, 1 + clip)).mean()


def test_on_policy_the_gradient_is_the_joint_one():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(32, generator=g)
    lp = torch.randn(32, 3, generator=g).requires_grad_(True)
    old = lp.detach().clone()
    gj, = torch.autograd.grad(_joint(a, lp, old, 0.2), lp)
    gg, = torch.autograd.grad(train_fast.ppo_group_surrogate(a, lp, old, 0.2), lp)
    assert torch.allclose(gj, gg, atol=1e-7)


def test_a_group_past_the_clip_goes_flat_alone():
    a = torch.tensor([1.0, 1.0])
    old = torch.zeros(2, 3)
    # row 0: the pitch group moved +0.5 (past +0.18 in the advantage's direction); row 1: nothing
    lp = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]], requires_grad=True)
    gj, = torch.autograd.grad(_joint(a, lp, old, 0.2), lp)
    gg, = torch.autograd.grad(train_fast.ppo_group_surrogate(a, lp, old, 0.2), lp)
    assert torch.all(gj[0] == 0)                  # the joint ratio: row 0 trains no head
    assert gg[0, 2] == 0                          # the group surrogate: only the pitch is flat
    assert gg[0, 0] != 0 and gg[0, 1] != 0        # the keys and the yaw still train on row 0
    assert torch.allclose(gg[1], gj[1])           # an untouched row: the same gradient


@needs_pb
def test_the_trainer_runs_the_group_clip():
    run = "phc_smoke"
    try:
        r = _train(run, ["--ppo-head-clip", "--diag-ratio-split", "256", "--epochs", "2"],
                   steps="8192")
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--ppo-head-clip: PPO's clipped surrogate per head group (keys / yaw / pitch)" in out
        assert any(ln.startswith("ratio-split [") for ln in out.splitlines())
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["ppo_head_clip"] == 1
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--ppo-head-clip" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_phc.py"
    a, b = ROOT / "runs" / "phc_off_new", ROOT / "runs" / "phc_off_old"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in ((a.name, TRAIN), (b.name, pre)):
            r = _train(run, ["--record-every", "2048"], script=script, steps="4096")
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "ppo_head_clip" not in ca
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
    assert "ppo_head_clip" in record_ckpt.TRAIN_ONLY
