"""--demo-front-hold (pbr4b2, 2026-10-25): inside --demo-front-temp's hot window an env draws with
HELD noise - the key heads' Gumbel uniforms and the view's standard normals are drawn at the band
spawn's first decision and reused for the window - so a band episode explores a sustained
manoeuvre instead of per-decision white noise, every draw still an exact sample of the tempered
policy (PPO scores it as before).

(a) sample_view / sample_padded: held rows repeat their draw for unchanged logits, free rows do
    not; the log-prob is the tempered distribution's of the drawn action either way;
(b) the trainer (CPU, princessburglar): on, it says so and dumps the key; refused without
    --demo-front-temp;
(c) flag off (with --demo-front-temp on) is bit-identical to the trainer before the change.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_demo_success_d import _run_demo  # noqa: E402
from test_ge_minutes import _csv, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def test_held_rows_repeat_their_draw():
    import torch
    import train_fast as tf
    torch.manual_seed(0)
    B, H, P = 64, tf.NACT - tf.N_VIEW, tf.NPAD
    padded = torch.randn(B, tf.NACT, P)
    mu = torch.zeros(B, 2)
    log_std = torch.zeros(2)
    m = torch.zeros(B)
    m[: B // 2] = 1.0
    u = torch.rand(B, H, P).clamp_min_(1e-20)
    e = torch.randn(B, 2)
    a1, z1, l1 = tf.sample_view(padded, mu, log_std, hold=(m, u, e))
    a2, z2, l2 = tf.sample_view(padded, mu, log_std, hold=(m, u, e))
    held, free = slice(0, B // 2), slice(B // 2, B)
    assert torch.equal(a1[held], a2[held]) and torch.equal(z1[held], z2[held])
    assert torch.allclose(z1[held], e[held])                  # mu 0, sigma 1: z = e exactly
    assert not torch.equal(z1[free], z2[free])                # fresh noise on free rows
    # the log-prob is the tempered joint's of whatever was drawn (here temperature 1)
    lp = tf.logprob_entropy_padded(padded[:, tf.N_VIEW:], a1[:, tf.N_VIEW:])[0] \
        + tf.gauss_logp(z1, mu, log_std.expand_as(mu))
    assert torch.allclose(lp, l1, atol=1e-5)
    # hold=None is the draw that shipped: the same RNG stream gives the same result
    torch.manual_seed(5)
    b1 = tf.sample_view(padded, mu, log_std)
    torch.manual_seed(5)
    b2 = tf.sample_view(padded, mu, log_std, hold=None)
    assert all(torch.equal(x, y) for x, y in zip(b1, b2))


@needs_pb
def test_on_dumped_and_refused(tmp_path):
    run = "dfh_on"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-front-temp", "2",
                            "--demo-front-ticks", "60", "--demo-front-hold",
                            "--steps", "8192"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-front-hold: inside that window the draws reuse ONE set of noise" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_front_hold"] == 1 and cfg["demo_front_temp"] == 2.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dfh_bad", ["--demo-front-hold"], tmp_path)
    assert bad.returncode != 0 and "need --demo-front-temp" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dfh_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-front-hold" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_dfh.py"
    extra = ["--demo-success-d", "50", "--demo-grow", "5", "--demo-front-frac", "0.67",
             "--demo-front-temp", "2", "--demo-front-ticks", "60", "--steps", "8192"]
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("dfh_off_new", TRAIN), ("dfh_off_old", pre)):
            r = _run_demo(run, extra, tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dfh_off_new", ROOT / "runs" / "dfh_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_front_hold" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
