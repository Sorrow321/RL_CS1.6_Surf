"""ab/mc (2026-11-14): the search loop's no-cull Monte Carlo stage and the archive bar.

abuf2_all (the ablation round's all-in on surf_unitfarmer2, ledger ab/main c9a3f657) dove into the
pit from 227M and never left it by 3.5B. October's no-cull Monte Carlo (beam_tas --elite-frac 1.0
--keys-temp 1.5, 4,096 rollouts from the true start) on its final weights climbs out in 3 of 3
seeds (d 29,834-29,890), but the loop's tree prunes by the field and never keeps a rollout that
laps the pit for 20-40 s first, and its archive stage handed back three lines through the trough
north of the pit (searches 2, 4 and 5: d 28,599 / 28,937 / 28,714), each deeper on the field than
the exit and none learned by the curriculum. The two pieces, both off by default:

* auto_spine.py ``--mc-envs N``: the Monte Carlo stage, after the tree and before the archives;
  its line passes like the tree's, or becomes the fallback when it is the deeper of the two;
* train_fast ``--auto-spine-archive-bar``: a PATIENCE search marks the held line failed; an
  archive line's end then bars every later archive line (auto_spine.py ``--archive-bar``), while
  the policy's own lines (tree, Monte Carlo) are judged as before.

CPU: the trainer's helpers and wiring, the recorder's TRAIN_ONLY key, the refusal, and the
decision of auto_spine.py's main() with its stages replaced by stand-ins that return lines
ending at the d each real search measured (k05 of abuf2_all, k03 of fxb200_t1 and k02 of
fxpb_t1): with the bar the k05 search hands back the Monte Carlo exit instead of the trough;
blue200's goal line and princessburglar's crossing still come from the archive.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

GE_ARGS = ["--ge-cell-heading", "0", "--ge-view-follow", "none", "--ge-boost", "1.0",
           "--ge-held", "1", "--ge-cell-contact", "1"]


def test_train_only():
    import record_ckpt
    assert "auto_spine_archive_bar" in record_ckpt.TRAIN_ONLY


def test_the_helpers():
    from train_fast import archive_bar, is_archive_source, mark_failed_archive
    for s in ("beam+ge", "beam+ge2", "beam+ge_energy", "beam_br2+ge3"):
        assert is_archive_source(s), s
    for s in (None, "", "beam", "beam_br", "beam_br3", "mc"):
        assert not is_archive_source(s), s
    failed = []
    assert archive_bar(failed) is None
    # abuf2_all: the held lines when the patience searches 3, 4 (no line) and 5 fired
    assert mark_failed_archive(failed, "beam+ge", 28599.4)
    assert not mark_failed_archive(failed, "beam+ge", 28599.4)       # each end once
    assert not mark_failed_archive(failed, "beam", 29986.2)          # a tree line: no bar
    assert not mark_failed_archive(failed, "mc", 29834.0)            # the policy's own: no bar
    assert not mark_failed_archive(failed, "beam+ge", float("inf"))
    assert mark_failed_archive(failed, "beam+ge", 28936.6)
    assert archive_bar(failed) == pytest.approx(28599.4)


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-archive-bar"' in src
    assert ('raise SystemExit("--auto-spine-archive-bar marks a line failed at a PATIENCE '
            'search: give "') in src
    assert 'meta["config"]["auto_spine_archive_bar"] = 1' in src
    # the failure is read from the trigger (patience, no quorum) before the command is run
    i_go = src.index("go = (quorum or pat) and global_step >= AUTO[\"retry_at\"]")
    i_mark = src.index("if pat and not quorum and mark_failed_archive(_fa, AUTO.get(\"src\"), "
                       "_held):")
    i_bar = src.index('cmd += ["--archive-bar", f"{archive_bar(_fa):.1f}"]')
    i_run = src.index("t0 = time.time()\n        if args.auto_spine_async")
    assert i_go < i_mark < i_bar < i_run
    # the held line's source is remembered when a spine is adopted, before current.json
    i_reset = src.index("demo.reset_spine(sp)")
    i_src = src.index('AUTO["src"] = res.get("source")', i_reset)
    i_cur = src.index('(adir / "current.json").write_text(', i_reset)
    assert i_reset < i_src < i_cur
    from train_fast import AUTO_SPINE_OWN_FLAGS
    assert "--archive-bar" in AUTO_SPINE_OWN_FLAGS


def _line(d_end, n=40, d0=None):
    """A stand-in line whose d (= x on the stand-in field) falls linearly to ``d_end``."""
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    d0 = d_end + 2000.0 if d0 is None else d0
    s["origin"][:, 0] = np.linspace(d0, d_end, n)
    s["origin"][:, 2] = 500.0
    return s


class _Field:                      # d = x, reachable everywhere
    reach_max, cell = 1e9, 32.0

    def sample(self, o):
        return np.asarray(o, np.float64)[:, 0]


def _decide(monkeypatch, tmp_path, greedy, current, tree, mc=None, ge=(), bar=None,
            mc_envs=4096):
    """auto_spine.main() with its stages replaced: the tree's line ends at ``tree``, the Monte
    Carlo line at ``mc``, the archive stages' (held, then contact) at ``ge``. -> (result json,
    the stages that ran)."""
    import auto_spine as A
    ran = []
    monkeypatch.setattr(A, "ckpt_tick", lambda _c: 7.6667)
    monkeypatch.setattr(A, "load_field", lambda _p: _Field())

    def beam(a, out, branch_tick=0, tag=None):
        ran.append("tree")
        return _line(tree, n=300), {"beam_rc": 0, "beam_replay_ok": 1, "beam_finished": 0}

    def mcs(a, out, tag="mc"):
        ran.append("mc")
        assert int(a.mc_envs) == mc_envs and float(a.mc_keys_temp) == 1.5
        if mc is None:
            return None, {"mc_rc": 1}
        return _line(mc, n=400), {"mc_rc": 0, "mc_replay_ok": 1, "mc_finished": 0}

    gq = list(ge)

    def go_explore(a, out, line, end, boosts=None, tag="ge", line_keys=None, roots=None,
                   pick=None, kind=None):
        ran.append(f"ge:{kind}")
        if not gq:
            return None, {"ge_rc": 1}
        return _line(gq.pop(0), n=500), {"ge_rc": 0, "ge_kind": kind}

    monkeypatch.setattr(A, "beam", beam)
    monkeypatch.setattr(A, "mc", mcs)
    monkeypatch.setattr(A, "go_explore", go_explore)
    argv = ["--ckpt", "x.pt", "--map", "m.bsp", "--field", "f.npz", "--out", str(tmp_path),
            "--greedy-d", str(greedy), "--gain", "200", "--beam-ticks", "6127",
            "--envs", "2048", "--ge-minutes", "5", "--explorer", "mcts",
            "--current-d", str(current)] + GE_ARGS
    if mc_envs:
        argv += ["--mc-envs", str(mc_envs), "--mc-keys-temp", "1.5"]
    if bar is not None:
        argv += ["--archive-bar", str(bar)]
    A.KEYS.clear()
    rc = A.main(argv)
    res = json.loads((tmp_path / "spine.json").read_text()) if rc == 0 else None
    return res, ran


# abuf2_all search 5 (2,988.4M): greedy 30,300.6, the held trough line (k04) ends at 28,936.6,
# the tree 30,099.9, the archive's held cell 28,713.5; the failed archive lines k02 (28,599.4)
# and k04; the Monte Carlo exit on the final weights 29,834 (seed 0)
K05 = dict(greedy=30300.6, current=28936.6, tree=30099.9)


def test_k05_with_the_bar_hands_back_the_monte_carlo_exit(monkeypatch, tmp_path):
    res, ran = _decide(monkeypatch, tmp_path, mc=29834.0, ge=(28713.5, 28429.2),
                       bar=28599.4, **K05)
    assert ran == ["tree", "mc", "ge:held", "ge:contact"]
    assert res["source"] == "mc" and res["fallback"] == 1
    assert res["end_d"] == pytest.approx(29834.0)
    assert res["ge_barred"] == 1 and res["ge2_barred"] == 1
    assert res["archive_ref"] == pytest.approx(28599.4)


def test_k05_without_the_bar_takes_the_trough_as_before(monkeypatch, tmp_path):
    res, ran = _decide(monkeypatch, tmp_path, mc=29834.0, ge=(28713.5,), **K05)
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(28713.5)
    assert ran == ["tree", "mc", "ge:held"]
    assert "archive_bar" not in res


def test_k05_both_off_is_the_old_decision(monkeypatch, tmp_path):
    res, ran = _decide(monkeypatch, tmp_path, ge=(28713.5,), mc_envs=0, **K05)
    assert ran == ["tree", "ge:held"]
    assert res["source"] == "beam+ge"
    assert not any(k.startswith("mc_") or "barred" in k or "archive" in k for k in res)


def test_a_monte_carlo_line_that_misses_keeps_the_tree_fallback(monkeypatch, tmp_path):
    # no exit (the 2.0B weights): the rollouts' best stays on the start platform
    res, _ran = _decide(monkeypatch, tmp_path, mc=30150.0, ge=(28713.5, 28429.2),
                        bar=28599.4, **K05)
    assert res["source"] == "beam" and res["fallback"] == 1
    assert res["end_d"] == pytest.approx(30099.9)


def test_a_monte_carlo_line_past_the_reference_skips_the_archives(monkeypatch, tmp_path):
    res, ran = _decide(monkeypatch, tmp_path, mc=28500.0, ge=(28000.0,), **K05)
    assert ran == ["tree", "mc"]
    assert res["source"] == "mc" and "fallback" not in res


def test_blue200_goal_line_still_comes_from_the_archive(monkeypatch, tmp_path):
    # fxb200_t1 search 3: the held line k2 was a TREE line (no bar); the archive found the goal
    res, _ran = _decide(monkeypatch, tmp_path, greedy=1977.3, current=1660.9, tree=1725.0,
                        mc=1700.0, ge=(0.0,))
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(0.0)


def test_princessburglar_crossing_still_comes_from_the_archive(monkeypatch, tmp_path):
    # fxpb_t1 search 2: the held line k1 was an ARCHIVE line that failed (bar 55,353); the
    # archive's crossing ends 255 u past it
    res, _ran = _decide(monkeypatch, tmp_path, greedy=56962.0, current=55353.0, tree=56900.0,
                        mc=56850.0, ge=(55098.0,), bar=55353.0)
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(55098.0)


def test_the_mc_command(tmp_path, monkeypatch):
    """mc() runs beam_tas with October's flags: no culling, hotter keys, the beam's ticks."""
    import auto_spine as A
    seen = {}

    def run(cmd, log, timeout):
        seen["cmd"] = [str(c) for c in cmd]
        return 0
    monkeypatch.setattr(A, "run", run)
    a = SimpleNamespace(ckpt="c.pt", map="m.bsp", mc_envs=4096, greedy_envs=32, mc_ticks=0,
                        beam_ticks=6127, seed=5, mc_keys_temp=1.5, timeout=2400.0)
    line, info = A.mc(a, tmp_path)
    assert line is None and info == {"mc_rc": 0}
    c = seen["cmd"]
    for flag, val in (("--envs", "4096"), ("--elite-frac", "1.0"), ("--keys-temp", "1.5"),
                      ("--max-ticks", "6127"), ("--greedy-envs", "32"), ("--score", "d"),
                      ("--torch-seed", "5")):
        assert c[c.index(flag) + 1] == val, flag
    assert "--save-frontier" in c and "--mcts" not in c
    a.mc_ticks = 6000
    A.mc(a, tmp_path)
    assert seen["cmd"][seen["cmd"].index("--max-ticks") + 1] == "6000"


try:
    from test_demo_success_d import _run_demo  # noqa: E402
    from test_ge_minutes import needs_pb  # noqa: E402
except Exception:                                      # noqa: BLE001
    _run_demo = None
    needs_pb = pytest.mark.skip(reason="the princessburglar smoke helpers are unavailable")


@needs_pb
def test_refusal(tmp_path):
    run = "asmc_bad0"
    try:
        r = _run_demo(run, ["--demo-success-d", "50", "--auto-spine", "--auto-spine-envs", "32",
                            "--auto-spine-archive-bar"], tmp_path)
        out = r.stdout + r.stderr
        assert r.returncode != 0 and "--auto-spine-archive-bar marks a line failed" in out, (
            out[-3000:])
    finally:
        shutil.rmtree(ROOT / "runs" / run, ignore_errors=True)
