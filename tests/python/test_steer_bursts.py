"""--steer-eps / --steer-front / --steer-dy / --steer-hold (2026-10-22, autoresearch r6b1).

A steer burst holds ONE offset dy ~ U(-D, D) degrees for n ~ U[LO, HI] decisions and adds
it to the yaw offset the policy just sampled (velocity frame); the keys, the pitch and the
policy's own steering stay its own (an ez burst replaces the keys too). Burst rows are
dropped from the PPO pool by b_ez, like ez rows.

(a) z_from_off_t inverts tanh-then-off_warp_t, so "offset + dy" lands where it should.
(b) CPU smoke on sidistic (abs view, keys-hold, unstuck): with --steer-front the band is
    live (the step line prints where it starts and a non-zero dropped share), the config
    carries the four keys, a flagless resume restores them; the refusals.
(c) flag off: the trainer before this change and after it give the same config,
    progress.csv (fps aside), eval trajectory bytes and weights on the same smoke.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from test_ez_viewc import ROOT, SMOKE, TRAIN, _csv, _run, _train, needs_run  # noqa: E402


def test_z_from_off_inverts_the_offset_warp():
    import train_fast as tf
    from surfgym.view import off_warp
    off = torch.tensor([-178.0, -90.0, -30.0, -10.0, -1.0, 0.0, 0.5, 10.0, 45.0, 120.0, 178.0])
    z = tf.z_from_off_t(off)
    back = tf.off_warp_t(torch.tanh(z))
    assert torch.allclose(back, off, atol=1e-2), (back, off)
    # the numpy reference agrees
    assert np.allclose(off_warp(np.tanh(z.numpy())), off.numpy(), atol=1e-2)
    # the policy's own U_CLIP (0.999, off_warp = 178.98 deg) bounds it: beyond that, and
    # beyond +-180, it saturates at the clip instead of producing inf
    zz = tf.z_from_off_t(torch.tensor([179.5, 400.0, -400.0]))
    assert torch.isfinite(zz).all() and float(zz[0]) > 0 > float(zz[2])
    assert torch.allclose(tf.off_warp_t(torch.tanh(zz)).abs(), torch.tensor(178.98), atol=0.01)


@needs_run
def test_steer_bursts_band_config_resume_and_refusals():
    run = "steer_front"
    r = _train(run, ["--steer-eps", "0.5", "--steer-front", "0.5", "--steer-dy", "60",
                     "--steer-hold", "2", "6", "--unstuck", "--unstuck-reach", "eval",
                     "--ep-ticks", "400", "--respawn-margin", "0.1"], steps="40960")
    m = re.findall(r"  ez +([0-9.]+)% steer>([0-9.]+)%", r.stdout)
    assert m and any(float(s) > 0.0 for s, _ in m), r.stdout[-3000:]
    d = ROOT / "runs" / run
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["steer_eps"] == 0.5 and cfg["steer_front"] == 0.5
    assert cfg["steer_dy"] == 60.0 and cfg["steer_hold"] == [2, 6]
    assert cfg["ez_eps"] == 0.0
    # the record gate: every new config key is TRAIN_ONLY for the recorder
    sys.path.insert(0, str(ROOT / "tools"))
    import record_ckpt
    record_ckpt.audit_cfg(cfg)
    assert {"steer_eps", "steer_front", "steer_dy", "steer_hold"} <= record_ckpt.TRAIN_ONLY
    run2 = "steer_front_re"
    shutil.rmtree(ROOT / "runs" / run2, ignore_errors=True)
    r2 = _run([sys.executable, "-u", str(TRAIN), "--run", run2, "--ckpt",
               str(d / "ckpt_final.pt")] + SMOKE + ["--steps", "8192"])
    assert r2.returncode == 0, r2.stdout[-4000:] + r2.stderr[-4000:]
    assert "steer_eps=0.5" in r2.stdout
    cfg2 = json.loads((ROOT / "runs" / run2 / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg2["steer_front"] == 0.5 and cfg2["steer_dy"] == 60.0 and cfg2["steer_hold"] == [2, 6]
    bad = _run([sys.executable, "-u", str(TRAIN), "--run", "steer_bad"]
               + SMOKE + ["--steps", "2048", "--steer-front", "0.5"])
    assert bad.returncode != 0 and "needs --steer-eps > 0" in (bad.stdout + bad.stderr)
    nov = [a for a in SMOKE if a not in ("--view-absolute", "velocity")]
    bad2 = _run([sys.executable, "-u", str(TRAIN), "--run", "steer_bad"]
                + nov + ["--steps", "2048", "--steer-eps", "0.1"])
    assert bad2.returncode != 0 and "--view-absolute velocity" in (bad2.stdout + bad2.stderr)
    for dd in (d, ROOT / "runs" / run2, ROOT / "runs" / "steer_bad"):
        shutil.rmtree(dd, ignore_errors=True)


def _pre_change_trainer(dst: Path):
    """train_fast.py of the last first-parent commit before --steer-eps."""
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--steer-eps" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_run
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change():
    pre = ROOT / "python" / "_train_fast_pre_steer.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        _train("steer_off_new", ["--unstuck", "--unstuck-reach", "eval"])
        _train("steer_off_old", ["--unstuck", "--unstuck-reach", "eval"], script=pre)
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "steer_off_new", ROOT / "runs" / "steer_off_old"
    ca = json.loads((a / "run.json").read_text(encoding="utf-8"))["config"]
    cb = json.loads((b / "run.json").read_text(encoding="utf-8"))["config"]
    assert ca == cb and "steer_eps" not in ca
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
    for dd in (a, b):
        shutil.rmtree(dd, ignore_errors=True)
