"""auto_spine --beam-extra (zlr2b2, 2026-11-01): extra beam_tas flags reach the PLAIN beam's
command only; absent, the command is the one before the flag."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "python"))

import auto_spine as A  # noqa: E402


def _cmd(tmp_path, beam_extra, branch_tick=0):
    seen = {}

    def fake_run(cmd, log, timeout):
        seen["cmd"] = [str(x) for x in cmd]
        return 1

    orig = A.run
    A.run = fake_run
    try:
        a = SimpleNamespace(ckpt="c.pt", map="m.bsp", envs=64, greedy_envs=4, beam_ticks=100,
                            seed=0, timeout=10.0, branch_burst_secs=2.0, branch_hold_secs=0.5,
                            branch_jitter=45, branch_protect=3, beam_extra=beam_extra)
        A.beam(a, tmp_path, branch_tick=branch_tick)
    finally:
        A.run = orig
    return seen["cmd"]


def test_absent_is_unchanged(tmp_path):
    c0 = _cmd(tmp_path, None)
    c1 = _cmd(tmp_path, "")
    assert c0 == c1
    assert "--view-hold" not in c0


def test_extra_reaches_the_plain_beam(tmp_path):
    c = _cmd(tmp_path, "--view-hold 0.3 25 --view-hold-stagger")
    i = c.index("--view-hold")
    assert c[i + 1:i + 3] == ["0.3", "25"] and "--view-hold-stagger" in c
    assert c[:len(_cmd(tmp_path, None))] == _cmd(tmp_path, None)


def test_branched_beam_keeps_its_own_fork(tmp_path):
    c = _cmd(tmp_path, "--view-hold 0.3 25", branch_tick=500)
    assert "--view-hold" not in c and "--branch-at" in c
