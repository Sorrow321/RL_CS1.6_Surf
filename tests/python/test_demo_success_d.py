"""--demo-success-d M (pbr1b1, 2026-10-23): under --demo-file, a spine episode counts as a
success for the backward curriculum when its ratchet record reaches the spine end's distance
+ M (or it finishes - the default rule). CPU, princessburglar: a short spine from the true spawn
trains, the trainer prints the threshold and dumps the key; without --race-ratchet it is refused;
flag off is bit-identical to the trainer before the change."""
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
from test_ge_minutes import PB, _csv, _env, _train, needs_pb  # noqa: E402

TRAIN = ROOT / "python" / "train_fast.py"


def _spine(tmp_path):
    """A 40-state time-ordered spine of real states: the true spawn, stepped forward."""
    from surfgym.core import SurfCore, default_config
    from surfgym.rewards import map_spawn_pool
    core = SurfCore(str(PB), default_config(num_envs=1, lidar_w=0, lidar_h=0))
    core.set_teleport_fail(True)
    core.reset(0)
    st = map_spawn_pool(core)[0].copy()
    core.set_state(0, st)
    rows = []
    a = np.zeros((1, 6), np.int32)
    a[0, 0], a[0, 1], a[0, 2] = 7, 3, 2                 # no turn, neutral pitch, forward
    for _ in range(40):
        rows.append(core.get_states()[0].copy())
        core.step(a)
    p = tmp_path / "spine.npy"
    np.save(p, np.array(rows))
    return p


def _run_demo(run, extra, tmp_path, script=TRAIN, ratchet=True):
    sp = _spine(tmp_path)
    flags = ["--demo-file", str(sp), "--demo-window", "5"] + (["--race-ratchet"] if ratchet
                                                                else []) + list(extra)
    shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    env = _env()
    env["SELF_STATES"] = "1"
    from test_ge_minutes import SMOKE
    return subprocess.run([sys.executable, "-u", str(script), "--run", run] + SMOKE
                          + ["--steps", "4096"] + flags, capture_output=True, text=True,
                          env=env, cwd=str(ROOT), timeout=1800, encoding="utf-8",
                          errors="replace")


@needs_pb
def test_the_success_rule_is_on_and_dumped(tmp_path):
    run = "dsd_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "64"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-4000:]
        assert "--demo-success-d 64: a spine episode succeeds when its record reaches d <=" in out
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["demo_success_d"] == 64.0
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("dsd_bad", ["--demo-success-d", "64"], tmp_path, ratchet=False)
    assert bad.returncode != 0 and "needs --race-ratchet" in (bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "dsd_bad", ignore_errors=True)


def _pre_change_trainer(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--demo-success-d" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical_to_the_trainer_before_the_change(tmp_path):
    import torch
    pre = ROOT / "python" / "_train_fast_pre_dsd.py"
    try:
        if _pre_change_trainer(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("dsd_off_new", TRAIN), ("dsd_off_old", pre)):
            r = _run_demo(run, ["--record-every", "2048"], tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "dsd_off_new", ROOT / "runs" / "dsd_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "demo_success_d" not in ca
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
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)


def test_demo_tau0_starts_the_curriculum_where_asked():
    from surfgym.core import STATE_DTYPE
    from surfgym.respawn import DemoCurriculum
    S = np.zeros(30, STATE_DTYPE)
    S["origin"][:, 0] = np.arange(30, dtype=np.float32)
    assert DemoCurriculum(S).tau == 29                       # the paper's start, unchanged
    assert DemoCurriculum(S, tau0=7).tau == 7
    assert DemoCurriculum(S, tau0=99).tau == 29 and DemoCurriculum(S, tau0=-3).tau == 0
    c = DemoCurriculum(S, window=5, grow=5, tau0=10)
    pool = c.build_pool(np.zeros(4, STATE_DTYPE), pool_size=200, fresh_frac=0.1)
    o = pool["origin"][:, 0]
    assert ((o == 0) | (o >= 10)).all()                       # fresh rows or spine [10, 29]
