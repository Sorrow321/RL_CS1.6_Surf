"""--auto-spine (pbr3b4, 2026-10-24): the search -> curriculum loop run by the trainer
(tools/auto_spine.py: beam from the true start, the frontier line cut at its last contact and
its best d, Go-Explore from the line's own last seconds when the beam falls short). CPU,
princessburglar: the cut rules' selftest, DemoCurriculum.reset_spine, a smoke run in which the
trainer calls the search and logs its outcome, the refusals, and flag off bit-identical to the
trainer before the change."""
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


def test_cut_rules_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "auto_spine.py"), "--selftest"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" not in r.stdout


def test_stall_tick():
    sys.path.insert(0, str(ROOT / "tools"))
    import auto_spine
    d = np.concatenate([np.linspace(1000, 300, 70), np.full(30, 290.0), [280.0]])
    # first within 50 u of the best (280): 330 at index ~67
    st = auto_spine.stall_tick(d, 50.0)
    assert d[st] <= 330.0 and (st == 0 or d[st - 1] > 330.0)


def test_reset_spine():
    from surfgym.core import STATE_DTYPE
    from surfgym.respawn import DemoCurriculum
    a = np.zeros(30, STATE_DTYPE)
    a["origin"][:, 0] = np.arange(30)
    b = np.zeros(50, STATE_DTYPE)
    b["origin"][:, 0] = 100 + np.arange(50)
    c = DemoCurriculum(a, window=5, rate=0.3, min_ep=1, grow=5, front_frac=0.4,
                       cooldown=1, grow_max=20)
    c.note_outcomes(np.arange(25, 30), np.ones(5))
    c.build_pool(a[:1], pool_size=64)
    assert c.tau < 29
    c.reset_spine(b)
    assert c.n == 50 and c.tau == 49 and c.ep.sum() == 0 and c.win.sum() == 0
    assert c._step == c.grow and c._cool == 0
    assert c.match(b["origin"][[0, 49]]).tolist() == [0, 49]
    assert c.match(a["origin"][[3]]).tolist() == [-1]
    pool = c.build_pool(a[:1], pool_size=64)
    assert len(pool) == 64


@needs_pb
def test_the_trainer_runs_the_search_and_logs_it(tmp_path):
    run = "aspn_smoke"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50",
                            "--auto-spine", "--auto-spine-quorum", "1",
                            "--auto-spine-patience", "1", "--auto-spine-envs", "16",
                            "--auto-spine-ge-minutes", "0", "--auto-spine-extra-secs", "1",
                            "--auto-spine-retry", "1", "--record-every", "2048",
                            "--eval-eps", "2", "--ep-ticks", "300"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode == 0, out[-6000:]
        assert "--auto-spine: a greedy eval with >= 1 episodes" in out
        assert "auto-spine: " in out and "greedy episodes past the success line" in out
        assert "auto-spine search 1:" in out, out[-6000:]
        assert ("NEW SPINE" in out) or ("NO new spine" in out)
        cfg = json.loads((ROOT / "runs" / run / "run.json").read_text())["config"]
        assert cfg["auto_spine"] == 1 and cfg["auto_spine_quorum"] == 1
        k1 = ROOT / "runs" / run / "auto" / "k01"
        assert (k1 / "auto_spine.log").exists()
        assert "AUTO_SPINE " in (k1 / "auto_spine.log").read_text()
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
    bad = _run_demo("aspn_bad", ["--auto-spine"], tmp_path)
    assert bad.returncode != 0 and "--auto-spine moves the --demo-success-d line" in (
        bad.stdout + bad.stderr)
    shutil.rmtree(ROOT / "runs" / "aspn_bad", ignore_errors=True)


def _pre_change(dst: Path):
    r = subprocess.run(["git", "rev-list", "--first-parent", "--max-count=200", "HEAD"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    for ref in (r.stdout.split() if r.returncode == 0 else ()):
        s = subprocess.run(["git", "show", f"{ref}:python/train_fast.py"],
                           capture_output=True, cwd=str(ROOT))
        if s.returncode == 0 and b"--auto-spine" not in s.stdout:
            dst.write_bytes(s.stdout)
            return ref
    return None


@needs_pb
def test_flag_off_is_bit_identical(tmp_path):
    pre = ROOT / "python" / "_train_fast_pre_aspn.py"
    try:
        if _pre_change(pre) is None:
            pytest.skip("no pre-change trainer in the first-parent history")
        for run, script in (("aspn_off_new", TRAIN), ("aspn_off_old", pre)):
            r = _run_demo(run, ["--demo-success-d", "50", "--demo-success-bonus", "50"],
                          tmp_path, script=script)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    finally:
        pre.unlink(missing_ok=True)
    a, b = ROOT / "runs" / "aspn_off_new", ROOT / "runs" / "aspn_off_old"
    try:
        ca = json.loads((a / "run.json").read_text())["config"]
        cb = json.loads((b / "run.json").read_text())["config"]
        ca.pop("run", None), cb.pop("run", None)
        assert ca == cb and "auto_spine" not in ca
        ra, rb = _csv(a.name), _csv(b.name)
        assert ra and len(ra) == len(rb)
        for x, y in zip(ra, rb):
            for k in x:
                if not k.startswith("time/"):
                    assert x[k] == y[k], k
    finally:
        for dd in (a, b):
            shutil.rmtree(dd, ignore_errors=True)


def test_branched_beam_command_parses(tmp_path, monkeypatch):
    """The branched beam's flags are ones beam_tas.py's own parser accepts (round 3: a float
    --branch-jitter killed the first live branched beam with an argparse error)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import auto_spine
    seen = {}

    def fake_run(cmd, log, timeout):
        seen["cmd"] = [str(c) for c in cmd]
        return 0
    monkeypatch.setattr(auto_spine, "run", fake_run)
    a = auto_spine.main.__globals__["argparse"].Namespace(
        ckpt="c.pt", map="m.bsp", envs=64, greedy_envs=4, beam_ticks=100, seed=0, timeout=10,
        branch_burst_secs=2.0, branch_hold_secs=0.5, branch_jitter=45, branch_protect=3)
    auto_spine.beam(a, tmp_path, branch_tick=1700)
    cmd = seen["cmd"]
    i = cmd.index("--branch-at")
    assert cmd[i + 1] == "t1700:32"
    import beam_tas
    ap_src = Path(beam_tas.__file__).read_text(encoding="utf-8")
    for flag in ("--branch-burst", "--branch-hold", "--branch-jitter", "--branch-protect"):
        v = cmd[cmd.index(flag) + 1]
        assert v.lstrip("-").isdigit(), (flag, v)        # beam_tas parses these as int
        assert f'"{flag}", type=int' in ap_src
    assert "--branch-view-only" in cmd
