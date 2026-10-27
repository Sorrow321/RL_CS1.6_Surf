"""--bc-view-coef C (pbr9b4, 2026-10-27): the view heads' weight in the BC loss under
--view-continuous. 0 = BC on the four categorical (key) heads only - pbr8b1's round-8 distil
steps read kl ~17 on the first update at --bc-coef 0.5 AND 0.05 because the view targets sat ~1 z
off a yaw head running at sigma ~0.013. The trainer says it, dumps it (only off its default) and
trains on CPU; the default is byte-identical to the trainer before the flag."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import numpy as np  # noqa: E402

from test_view_absolute import TRAIN, _csv, _run  # noqa: E402
from test_ge_minutes import SMOKE as SMOKE_FLAGS, needs_pb as needs_run  # noqa: E402  (princessburglar)
from test_keys_hold_bc import _engine_line  # noqa: E402


def _write_bc(path, n=16):
    """A tiny --keys-hold --view-continuous --view-absolute velocity BC file (the princessburglar
    smoke's modes): an engine line's keys turned into policy-space rows + held-key columns, a
    finite absolute view column and (n, 2) z moments."""
    from surfgym.bc import keys_policy_rows, save_bc_dataset
    from surfgym.core import STATE_DTYPE
    rng = np.random.default_rng(3)
    pol, _held, feat = keys_policy_rows(_engine_line(n=n))
    view = np.stack([rng.uniform(-40, 40, n), rng.uniform(-60, 20, n)], 1).astype(np.float32)
    zmu = rng.standard_normal((n, 2)).astype(np.float32)
    zsd = np.abs(rng.standard_normal((n, 2))).astype(np.float32)
    meta = {"obs_reward": False, "n_latch": 0, "act_every": 4, "view_continuous": 1,
            "view_absolute": "velocity", "keys_hold": 1}
    save_bc_dataset(path, np.zeros(n, STATE_DTYPE), np.zeros((n, 15), np.float32),
                    np.zeros(n, np.float32), pol, np.ones(n, np.float32), np.zeros(n, np.int32),
                    meta, keys=feat, view=view, view_zmu=zmu, view_zsd=zsd)

BASE = "a9e11cb"      # the commit before the flag
BC = ["--bc-coef", "0.5", "--bc-target", "dist", "--bc-batch", "8"]


def _bc_run(run, bc_file, extra=(), script=TRAIN):
    import os
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8", SELF_STATES="1")     # a synthetic file, no demo
    return subprocess.run([sys.executable, "-u", str(script), "--run", run] + SMOKE_FLAGS
                          + ["--steps", "4096", "--bc-file", str(bc_file)] + BC + list(extra),
                          capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=1800,
                          encoding="utf-8", errors="replace")


@needs_run
def test_keys_only_bc_trains_says_and_dumps(tmp_path):
    f = tmp_path / "bc.npz"
    _write_bc(f, n=16)
    run = "bvc_on"
    try:
        r = _bc_run(run, f, ["--bc-view-coef", "0"])
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--bc-view-coef 0 - the view heads' terms weighted 0 in the BC loss (keys only)" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["bc_view_coef"] == 0.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_run
def test_default_is_the_trainer_before_it(tmp_path):
    f = tmp_path / "bc.npz"
    _write_bc(f, n=16)
    pre = ROOT / "python" / "_train_fast_pre_bvc.py"
    src = subprocess.run(["git", "show", f"{BASE}:python/train_fast.py"], capture_output=True,
                         cwd=str(ROOT), check=True).stdout
    assert b"--bc-view-coef" not in src
    pre.write_bytes(src)
    a, b = "bvc_off_new", "bvc_off_old"
    try:
        for run, script in ((a, TRAIN), (b, pre)):
            r = _bc_run(run, f, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        ca = json.loads((ROOT / "runs" / a / "run.json").read_text())["config"]
        cb = json.loads((ROOT / "runs" / b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "bc_view_coef" not in ca
        ra, rb = _csv(a), _csv(b)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        pre.unlink(missing_ok=True)
        for run in (a, b):
            shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
