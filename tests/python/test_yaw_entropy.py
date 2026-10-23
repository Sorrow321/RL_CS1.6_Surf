"""--yaw-entropy Y (pbr1b1, 2026-10-23): the yaw head(s)' share of the entropy bonus under
--view-continuous, like --pitch-entropy for the pitch head.

(a) logprob_entropy_view: Y = 1 is the old result exactly; Y = 0 drops the yaw Gaussian's
    entropy (and with it the gradient that walks a state-independent log sigma up); Y scales it;
    the log-prob never changes; the per-row (--front-temp) branch obeys the same rule;
(b) the trainer (CPU, princessburglar): --yaw-entropy 0 trains, says so and dumps the key;
    a negative value is refused;
(c) flag off is bit-identical to the trainer before the change.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_ge_minutes import _csv, _train, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def test_scale_one_is_the_old_entropy_and_zero_drops_the_yaw_head():
    import torch
    import train_fast as tf
    torch.manual_seed(0)
    B = 5
    ls = torch.tensor([-1.3, -2.2], requires_grad=True)          # yaw, pitch log sigma
    mu = torch.zeros(B, 2)
    z = torch.randn(B, 2)
    PAD, ACT = torch.zeros(B, 12), torch.zeros(B, 6, dtype=torch.long)
    # the categorical part is irrelevant to the rule: give it a fixed log-prob / entropy
    orig = tf.logprob_entropy_padded
    tf.logprob_entropy_padded = lambda padded, actions, temp=None: (
        torch.zeros(B), torch.full((B,), 0.7))
    try:
        lp1, e1 = tf.logprob_entropy_view(PAD, ACT, mu, ls, z,
                                          0.0)
        lp1b, e1b = tf.logprob_entropy_view(PAD, ACT, mu, ls,
                                            z, 0.0, yaw_ent=1.0)
        lp0, e0 = tf.logprob_entropy_view(PAD, ACT, mu, ls, z,
                                          0.0, yaw_ent=0.0)
        lph, eh = tf.logprob_entropy_view(PAD, ACT, mu, ls, z,
                                          1.0, yaw_ent=0.5)
    finally:
        tf.logprob_entropy_padded = orig
    assert torch.equal(lp1, lp1b) and torch.equal(e1, e1b)
    assert torch.equal(lp0, lp1) and torch.equal(lph, lp1)          # log-prob untouched
    h_yaw = tf.gauss_entropy(ls[:1])
    h_pitch = tf.gauss_entropy(ls[1:])
    assert torch.allclose(e1, 0.7 + h_yaw.expand(B))                 # pitch_ent 0
    assert torch.allclose(e0, torch.full((B,), 0.7))                 # yaw dropped too
    assert torch.allclose(eh, 0.7 + 0.5 * h_yaw + h_pitch)
    g, = torch.autograd.grad(e0.sum(), ls, allow_unused=True)
    assert g is None or float(g.abs().sum()) == 0.0                  # nothing walks sigma up


def test_the_per_row_branch_obeys_the_same_rule():
    import torch
    import train_fast as tf
    B = 4
    ls = torch.tensor([[-1.0, -2.0]] * B)
    orig = tf.logprob_entropy_padded
    orig_t = tf._temper_log_std
    tf.logprob_entropy_padded = lambda padded, actions, temp=None: (
        torch.zeros(B), torch.zeros(B))
    tf._temper_log_std = lambda log_std, t: log_std
    try:
        PAD, ACT = torch.zeros(B, 12), torch.zeros(B, 6, dtype=torch.long)
        _, e1 = tf.logprob_entropy_view(PAD, ACT, torch.zeros(B, 2), ls, torch.zeros(B, 2), 1.0)
        _, e0 = tf.logprob_entropy_view(PAD, ACT, torch.zeros(B, 2), ls, torch.zeros(B, 2), 1.0,
                                        yaw_ent=0.0)
    finally:
        tf.logprob_entropy_padded = orig
        tf._temper_log_std = orig_t
    h = 0.5 + 0.5 * tf.LOG2PI + ls
    assert torch.allclose(e1, h.sum(-1)) and torch.allclose(e0, h[:, -1])


@needs_pb
def test_the_trainer_carries_the_flag():
    run = "yawent_smoke"
    try:
        r = _train(run, ["--yaw-entropy", "0"])
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--yaw-entropy 0: the yaw head's entropy term is OFF" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["yaw_entropy"] == 0.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _train("yawent_bad", ["--yaw-entropy", "-1"])
    assert bad.returncode != 0 and "--yaw-entropy must be >= 0" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "yawent_bad", ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--yaw-entropy" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    import torch
    pre = ROOT / "python" / "_train_fast_pre_yawent.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("yawent_off_new", TRAIN), ("yawent_off_old", pre)):
            r = _train(run, ["--record-every", "2048"], script=script, steps="4096")
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "yawent_off_new", ROOT / "runs" / "yawent_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "yaw_entropy" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
        ta, tb = sorted(a.glob("traj_*.jsonl")), sorted(b.glob("traj_*.jsonl"))
        assert ta and [p.name for p in ta] == [p.name for p in tb]
        for p, q in zip(ta, tb):
            assert p.read_bytes() == q.read_bytes(), p.name
        sa = torch.load(a / "ckpt_final.pt", map_location="cpu", weights_only=False)
        sb = torch.load(b / "ckpt_final.pt", map_location="cpu", weights_only=False)
        for k in sa["policy"]:
            assert torch.equal(sa["policy"][k], sb["policy"][k]), k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
