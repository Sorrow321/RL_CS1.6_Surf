"""--demo-keys (pbr4b3, 2026-10-25): under --keys-hold a spine spawn starts holding the keys its
row was reached with. CPU, princessburglar: a run with the flag trains, applies the keys and dumps
the key; a bad keys file is refused; flag off is bit-identical to the trainer before the change."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from test_ge_minutes import SMOKE, _csv, _env, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"
SPINE = ROOT / "autoresearch" / "spines" / "pbr4b3_ge16ck.npy"
KEYS = ROOT / "autoresearch" / "spines" / "pbr4b3_ge16ck.keys.npy"
def _train(run, extra, script=None, steps="2048"):
    """test_ge_minutes._train with SELF_STATES=1: the spine is the policy's own line + its search."""
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    env = _env()
    env["SELF_STATES"] = "1"
    return subprocess.run([sys.executable, "-u", str(script or TRAIN), "--run", run] + SMOKE
                          + ["--steps", steps] + list(extra), capture_output=True, text=True,
                          env=env, cwd=str(ROOT), timeout=1800, encoding="utf-8",
                          errors="replace")


DEMO = ["--demo-file", str(SPINE), "--race-ratchet", "--demo-success-d", "100",
        "--demo-tau0", "2870", "--respawn-greedy", "0", "--no-respawn-novel",
        "--respawn-backward-mix", "0", "--respawn-front-perturb", "0"]


@needs_pb
def test_the_flag_trains_and_is_dumped():
    run = "dkeys_on"
    try:
        r = _train(run, DEMO + ["--demo-keys", str(KEYS)])
        assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        assert "--demo-keys" in r.stdout and "rows hold something other than neutral" in r.stdout
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_keys"] == str(KEYS)
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)


@needs_pb
def test_a_mismatched_keys_file_is_refused(tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.ones((5, 3), np.int32))
    r = _train("dkeys_bad", DEMO + ["--demo-keys", str(bad)])
    shutil.rmtree(ROOT / "runs" / "dkeys_bad", ignore_errors=True)
    assert r.returncode != 0 and "--demo-keys" in (r.stdout + r.stderr)


def _pre_change(dst: Path):
    s = subprocess.run(["git", "show", "HEAD:python/train_fast.py"], capture_output=True,
                       cwd=str(ROOT))
    if s.returncode != 0 or b'"--demo-keys"' in s.stdout:
        return None
    dst.write_bytes(s.stdout)
    return True


@needs_pb
def test_flag_off_is_bit_identical():
    pre = ROOT / "python" / "_train_fast_pre_dkeys.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("HEAD's trainer already has --demo-keys (run before committing)")
        for run, script in (("dkeys_off_new", TRAIN), ("dkeys_off_old", pre)):
            r = _train(run, DEMO + ["--record-every", "2048"], script=script, steps="4096")
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dkeys_off_new", ROOT / "runs" / "dkeys_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_keys" not in ca
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
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)
