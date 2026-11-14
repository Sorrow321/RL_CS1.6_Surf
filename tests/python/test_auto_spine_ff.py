"""ab/ff (2026-11-14): a stepping-stone fallback after the held line failed.

The all-in on surf_princessburglar_b26 (abpb_all, ledger ab/main 64cc18fb / 20a23c95) handed back
an archive line at its first search that crossed the pit and then wandered 13 s on the far ledge
(end d 55,107). The curriculum never learned it, and its patience searches 2 and 3 found archive
lines from the start room into the pit (55,345-55,361, ~1,650 u past the greedy frontier) - the
kind of line fix-round try 3 learned to the spawn - and refused them: under
--auto-spine-current-ref they had to end 200 u past the held line's end, and under ab/mc's
--auto-spine-archive-bar past the failed archive end, and archive lines have no fallback path.

* auto_spine.py ``--failed-fallback ENDS``: in a patience search after the held line failed, when
  no line passes and neither the tree nor the Monte Carlo line is a fallback, the deepest ARCHIVE
  line that beats the greedy frontier by --gain and ends at least --gain away from every failed
  archive end is handed back (a stepping stone short of the failed line, not its dead end again);
* train_fast ``--auto-spine-failed-fallback`` (needs --auto-spine-archive-bar) passes the failed
  archive ends in exactly the searches ab/mc marks a held line failed (patience, no quorum).

CPU: the helpers, the trainer's wiring, the recorder's TRAIN_ONLY key, and auto_spine.py's
decision with its stages replaced by stand-ins ending at the d each real search measured: the
all-in's princessburglar searches 2 and 3 now hand back the room-to-pit line, while ab/mc's three
replayed decisions (abuf2_all search 5 -> the Monte Carlo exit; fxb200_t1 search 3 -> the
archive's goal line; fxpb_t1 search 2 -> the archive's crossing) do not change, and unitfarmer2's
trough lines beside the failed ends stay refused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

GE_ARGS = ["--ge-cell-heading", "0", "--ge-view-follow", "none", "--ge-boost", "1.0",
           "--ge-held", "1", "--ge-cell-contact", "1"]


def test_train_only():
    import record_ckpt
    assert "auto_spine_failed_fallback" in record_ckpt.TRAIN_ONLY


def test_the_helpers():
    from auto_spine import failed_ends, stepping_stone
    assert failed_ends(None) == [] and failed_ends("-") == [] and failed_ends("") == []
    assert failed_ends("55106.6") == [55106.6]
    assert failed_ends("28599.4,28936.6") == [28599.4, 28936.6]
    # princessburglar: the room-to-pit lines are 238-254 u short of the failed crossing's end
    assert stepping_stone(55345.0, 57004.3, 200.0, [55106.6])
    assert stepping_stone(55347.0, 56990.3, 200.0, [55106.6])
    # unitfarmer2: the trough lines lie within 200 u of a failed trough end
    assert not stepping_stone(28713.5, 30300.6, 200.0, [28599.4, 28936.6])
    assert not stepping_stone(28429.2, 30300.6, 200.0, [28599.4, 28936.6])
    # a line must still beat the greedy frontier by the gain
    assert not stepping_stone(56911.0, 57004.3, 200.0, [55106.6])
    # no failed archive end (the failed held line was the policy's own): the frontier alone
    assert stepping_stone(1500.0, 1977.3, 200.0, [])


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-failed-fallback"' in src
    assert ('raise SystemExit("--auto-spine-failed-fallback reads the failed archive ends of "'
            in src)
    assert 'meta["config"]["auto_spine_failed_fallback"] = 1' in src
    # the ends are passed after this search's own failure is marked, only in a failed search
    i_mark = src.index("if pat and not quorum and mark_failed_archive(_fa, AUTO.get(\"src\"), "
                       "_held):")
    i_bar = src.index('cmd += ["--archive-bar", f"{archive_bar(_fa):.1f}"]')
    i_ff = src.index("if args.auto_spine_failed_fallback and pat and not quorum:")
    i_cmd = src.index('cmd += ["--failed-fallback", ",".join(f"{v:.1f}" for v in _fa) or "-"]')
    i_run = src.index("t0 = time.time()\n        if args.auto_spine_async")
    assert i_mark < i_bar < i_ff < i_cmd < i_run
    from train_fast import AUTO_SPINE_OWN_FLAGS
    assert "--failed-fallback" in AUTO_SPINE_OWN_FLAGS


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
            mc_envs=4096, failed=None):
    """auto_spine.main() with its stages replaced (test_auto_spine_mc's harness) plus
    ``failed`` -> --failed-fallback. -> (result json or None when no line, the stages that ran,
    the result printed on the AUTO_SPINE line)."""
    import auto_spine as A
    ran = []
    monkeypatch.setattr(A, "ckpt_tick", lambda _c: 7.6667)
    monkeypatch.setattr(A, "load_field", lambda _p: _Field())

    def beam(a, out, branch_tick=0, tag=None):
        ran.append("tree")
        return _line(tree, n=300), {"beam_rc": 0, "beam_replay_ok": 1, "beam_finished": 0}

    def mcs(a, out, tag="mc"):
        ran.append("mc")
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
    if failed is not None:
        argv += ["--failed-fallback", failed]
    A.KEYS.clear()
    rc = A.main(argv)
    res = json.loads((tmp_path / "spine.json").read_text()) if rc == 0 else None
    return res, ran


# abpb_all (the all-in on princessburglar): the held line k01 (archive, crossing, end 55,106.6)
# failed; patience search 2 (680M): greedy 57,004.3, tree cut at 56,911, archives 55,354 (held)
# and 55,345 (contact); search 3 (1.36B): greedy 56,990.3, tree 56,933, archives 55,361 / 55,347.
# The Monte Carlo stage was not in the all-in; on its 2.0B weights it stays in the room (sub-agent
# P's bench: 56,906-56,913), so it is never a fallback here.
K02 = dict(greedy=57004.3, current=55106.6, tree=56911.0, mc=56913.0, ge=(55354.0, 55345.0),
           bar=55106.6)
K03 = dict(greedy=56990.3, current=55106.6, tree=56933.0, mc=56906.0, ge=(55361.0, 55347.0),
           bar=55106.6)


def test_allin_pb_search2_hands_back_the_room_to_pit_line(monkeypatch, tmp_path):
    res, ran = _decide(monkeypatch, tmp_path, failed="55106.6", **K02)
    assert ran == ["tree", "mc", "ge:held", "ge:contact"]
    assert res["source"] == "beam+ge2" and res["end_d"] == pytest.approx(55345.0)
    assert res["fallback"] == 1 and res["failed_fallback"] == 1
    assert res["failed_ends"] == [55106.6]


def test_allin_pb_search3_hands_back_the_room_to_pit_line(monkeypatch, tmp_path):
    res, _ran = _decide(monkeypatch, tmp_path, failed="55106.6", **K03)
    assert res["source"] == "beam+ge2" and res["end_d"] == pytest.approx(55347.0)
    assert res["failed_fallback"] == 1


def test_allin_pb_search2_without_the_monte_carlo_stage(monkeypatch, tmp_path):
    # the all-in itself had no Monte Carlo stage: the same decision
    res, ran = _decide(monkeypatch, tmp_path, failed="55106.6", mc_envs=0, **K02)
    assert ran == ["tree", "ge:held", "ge:contact"]
    assert res["source"] == "beam+ge2" and res["end_d"] == pytest.approx(55345.0)


def test_allin_pb_search2_without_the_rule_refuses_as_before(monkeypatch, tmp_path):
    res, _ran = _decide(monkeypatch, tmp_path, **K02)
    assert res is None                                   # no line: ab/mc's decision
    res, _ran = _decide(monkeypatch, tmp_path, mc_envs=0, **{k: v for k, v in K02.items()
                                                              if k not in ("bar", "mc")})
    assert res is None                                   # the all-in's own decision


# ab/mc's three replayed decisions, now with the rule on

K05 = dict(greedy=30300.6, current=28936.6, tree=30099.9)


def test_k05_still_hands_back_the_monte_carlo_exit(monkeypatch, tmp_path):
    res, ran = _decide(monkeypatch, tmp_path, mc=29834.0, ge=(28713.5, 28429.2),
                       bar=28599.4, failed="28599.4,28936.6", **K05)
    assert ran == ["tree", "mc", "ge:held", "ge:contact"]
    assert res["source"] == "mc" and res["fallback"] == 1 and "failed_fallback" not in res
    assert res["end_d"] == pytest.approx(29834.0)


def test_blue200_goal_line_still_comes_from_the_archive(monkeypatch, tmp_path):
    # fxb200_t1 search 3: the held line was a TREE line (no failed archive end)
    res, _ran = _decide(monkeypatch, tmp_path, greedy=1977.3, current=1660.9, tree=1725.0,
                        mc=1700.0, ge=(0.0,), failed="-")
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(0.0)
    assert "failed_fallback" not in res


def test_princessburglar_crossing_still_comes_from_the_archive(monkeypatch, tmp_path):
    # fxpb_t1 search 2: the failed archive line k1 (55,353); the crossing passes everything
    res, _ran = _decide(monkeypatch, tmp_path, greedy=56962.0, current=55353.0, tree=56900.0,
                        mc=56850.0, ge=(55098.0,), bar=55353.0, failed="55353.0")
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(55098.0)
    assert "failed_fallback" not in res


def test_the_trough_stays_refused_without_a_policy_fallback(monkeypatch, tmp_path):
    # abuf2_all search 5 with no Monte Carlo stage and a tree that is no fallback: both archive
    # lines end within 200 u of a failed trough end -> no line, the dead end is not handed back
    res, _ran = _decide(monkeypatch, tmp_path, greedy=30300.6, current=28936.6, tree=30250.0,
                        ge=(28713.5, 28429.2), bar=28599.4, mc_envs=0,
                        failed="28599.4,28936.6")
    assert res is None


def test_a_stepping_stone_far_from_the_failed_ends_is_handed_back(monkeypatch, tmp_path):
    # the rule as written: with no policy fallback, an archive line 514 u short of the nearest
    # failed end that beats the greedy frontier is handed back (search 5's final-weights 29,451)
    res, _ran = _decide(monkeypatch, tmp_path, greedy=30300.6, current=28936.6, tree=30250.0,
                        ge=(29451.0, 29102.0), bar=28599.4, mc_envs=0,
                        failed="28599.4,28936.6")
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(29451.0)
    assert res["failed_fallback"] == 1
