"""ab/probe (2026-11-14): auto_spine.py --end-probe-iters - is there a way on from a line's END?

Every unitfarmer2 search ranks the trough north of the pit deeper on the field than the pit's
exit, and a ban on where failed lines ended cannot tell the two apart (ledger w8, the tail ban:
the exit line ab4u_1 k05 ends 80 u from the end of the lip line k01 that failed at 0.53B). The
probe asks the end itself: the archive explorer (explore_phase1, uniform actions, no policy),
rooted at the line's last --end-probe-secs, runs a fixed count of chunks, and the line is handed
back only when the explorer reaches a state the map holds --end-probe-gain (default --gain)
deeper than the line's end, or the finish.

CPU: the verdict on stand-in archives, the explorer's command and roots with run() replaced,
and auto_spine.main()'s decisions with its stages and the probe replaced by stand-ins: off, the
old decisions; on, a line with no way on is refused at every hand-back (tree, Monte Carlo,
archives, the failed-fallback stepping stone), a finishing line is never probed, and each
stage's line is probed once and only when it would be handed back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

GE_ARGS = ["--ge-cell-heading", "0", "--ge-view-follow", "none", "--ge-boost", "1.0",
           "--ge-held", "1", "--ge-cell-contact", "1"]


class _Field:                      # d = x; reachable below 1e8
    reach_max, cell = 1e8, 32.0

    def sample(self, o):
        return np.asarray(o, np.float64)[:, 0]


def _line(d_end, n=40, d0=None):
    """A stand-in line whose d (= x on the stand-in field) falls linearly to ``d_end``."""
    from surfgym.core import STATE_DTYPE
    s = np.zeros(n, STATE_DTYPE)
    d0 = d_end + 2000.0 if d0 is None else d0
    s["origin"][:, 0] = np.linspace(d0, d_end, n)
    s["origin"][:, 2] = 500.0
    s["tick"] = np.arange(n)
    return s


def _archive(path, ds, held):
    from surfgym.core import STATE_DTYPE
    st = np.zeros(len(ds), STATE_DTYPE)
    st["origin"][:, 0] = ds
    np.savez(path, state=st, held=np.asarray(held, bool))


def _verdict(tmp_path, ds, held, end_d=28500.0, gain=200.0, goal=False):
    import auto_spine as A
    _archive(tmp_path / "a.npz", ds, held)
    with np.load(tmp_path / "a.npz", allow_pickle=False) as z:
        return A.probe_verdict(z, _Field(), end_d, gain, goal)


def test_the_verdict(tmp_path):
    # a held state 250 u deeper: a way on
    v = _verdict(tmp_path, [28700.0, 28250.0, 28600.0], [True, True, False])
    assert v["probe_way_on"] == 1 and v["probe_held_d"] == 28250.0 and v["probe_held_i"] == 1
    # 150 u deeper is not --gain deeper; a FLIGHT 900 u deeper does not count
    v = _verdict(tmp_path, [28350.0, 27600.0], [True, False])
    assert v["probe_way_on"] == 0
    assert v["probe_best_d"] == 27600.0 and v["probe_held_d"] == 28350.0
    # the finish is a way on whatever the archive holds
    assert _verdict(tmp_path, [28600.0], [True], goal=True)["probe_way_on"] == 1
    # nothing held at all, and unreachable cells (beyond the field's reach) never count
    v = _verdict(tmp_path, [28000.0], [False])
    assert v["probe_way_on"] == 0 and v["probe_held_i"] == -1
    v = _verdict(tmp_path, [2e8, 28400.0], [True, True])
    assert v["probe_way_on"] == 0 and v["probe_held_d"] == 28400.0
    # the gain is the caller's: 100 u deeper with --end-probe-gain 100
    assert _verdict(tmp_path, [28400.0], [True], gain=100.0)["probe_way_on"] == 1


def _probe_args(**kw):
    a = dict(map="m.bsp", ge_cell=64.0, ge_energy=32.0, act_every=4, maxvel=3500.0, threads=0,
             tick=7.6667, ge_cell_heading=0, ge_view_follow="none", seed=3, end_probe_iters=250,
             end_probe_envs=2048, end_probe_secs=2.0, end_probe_roots=16, end_probe_gain=None,
             gain=200.0)
    a.update(kw)
    return SimpleNamespace(**a)


def test_the_probe_command_and_its_roots(tmp_path, monkeypatch):
    import auto_spine as A
    monkeypatch.setattr(A, "TICKS_PER_S", 1000.0 / 7.6667)
    seen = {}

    def run(cmd, log, timeout):
        c = [str(x) for x in cmd]
        seen["cmd"] = c
        out = Path(c[c.index("--out") + 1])
        seen["roots"] = np.load(out / "roots.npy", allow_pickle=False)
        _archive(out / "archive.npz", [28800.0, 28200.0, 27900.0], [True, True, False])
        return 1                             # explore_phase1: 1 = no goal hit, the normal end
    monkeypatch.setattr(A, "run", run)
    line = _line(28500.0, n=1500)
    info = A.end_probe(_probe_args(), tmp_path, line, 1499, "beam", _Field())
    c = seen["cmd"]
    for flag, val in (("--max-iters", "250"), ("--envs", "2048"), ("--goals", "1"),
                      ("--snapshot-every", "0"), ("--decisions", "100"), ("--act-every", "4"),
                      ("--cell", "64.0"), ("--cell-energy", "32.0"), ("--tick-ms", "8"),
                      ("--maxvel", "3500.0"), ("--seed", "3")):
        assert c[c.index(flag) + 1] == val, flag
    for flag in ("--cell-held", "--exact-chains", "--roots-keep-all"):
        assert flag in c
    assert "--cell-heading" not in c and not any(x.startswith("--view-follow") for x in c)
    # 16 roots over the line's last 2 s (261 ticks at 7.67 ms), the end included
    r = seen["roots"]
    assert len(r) == 16 and int(r["tick"].sum()) == 0          # clean()ed, as the GE roots
    rx = np.asarray(r["origin"][:, 0], np.float64)
    lx = np.asarray(line["origin"][:, 0], np.float64)
    assert rx[-1] == lx[1499] and rx[0] == pytest.approx(lx[1499 - 261])
    assert info["probe_roots"] == [1499 - 261, 1499]
    assert info["probe_way_on"] == 1 and info["probe_held_d"] == 28200.0
    assert info["probe_end_d"] == 28500.0 and info["probe_gain"] == 200.0
    # the archive is gone, its deepest held state stays
    assert not (tmp_path / "beam_probe" / "archive.npz").exists()
    dh = np.load(tmp_path / "beam_probe" / "deepest_held.npy", allow_pickle=False)
    assert float(dh["origin"][0][0]) == 28200.0


def test_a_probe_that_leaves_no_archive_refuses_nothing(tmp_path, monkeypatch):
    import auto_spine as A
    monkeypatch.setattr(A, "run", lambda cmd, log, timeout: -9)
    info = A.end_probe(_probe_args(tick=10.0), tmp_path, _line(28500.0), 39, "mc", _Field())
    assert info["probe_way_on"] == 1 and info["probe_err"] == "no archive"


def test_the_probe_at_10_ms_and_with_the_ge_view_flags(tmp_path, monkeypatch):
    import auto_spine as A
    monkeypatch.setattr(A, "TICKS_PER_S", 100.0)
    seen = {}

    def run(cmd, log, timeout):
        seen["cmd"] = [str(x) for x in cmd]
        return 1
    monkeypatch.setattr(A, "run", run)
    a = _probe_args(tick=10.0, ge_cell_heading=1, ge_view_follow="-30,0,30", end_probe_secs=1.0,
                    end_probe_roots=8, end_probe_gain=500.0, threads=6, maxvel=None)
    info = A.end_probe(a, tmp_path, _line(1000.0, n=300), 299, "ge2", _Field())
    c = seen["cmd"]
    assert "--tick-ms" not in c and "--maxvel" not in c
    assert c[c.index("--cell-heading") + 1] == "1" and "--view-follow=-30,0,30" in c
    assert c[c.index("--threads") + 1] == "6"
    assert info["probe_roots"] == [199, 299] and info["probe_gain"] == 500.0


# ---- auto_spine.main() with its stages and the probe replaced ----------------------------------

def _decide(monkeypatch, tmp_path, greedy, current, tree, mc=None, ge=(), probe=None,
            dead=(), tree_fin=False, fallback=None, extra=()):
    """auto_spine.main(): the tree's line ends at ``tree``, the Monte Carlo line at ``mc``, the
    archive stages' at ``ge``; with ``probe`` (chunks) the probe says 'no way on' for a line
    that ends at a d in ``dead``. -> (result json or None, the stages that ran, the probes)."""
    import auto_spine as A
    ran, probes = [], []
    monkeypatch.setattr(A, "ckpt_tick", lambda _c: 7.6667)
    monkeypatch.setattr(A, "load_field", lambda _p: _Field())

    def beam(a, out, branch_tick=0, tag=None):
        ran.append("tree")
        return _line(tree, n=300), {"beam_rc": 0, "beam_replay_ok": 1,
                                    "beam_finished": int(tree_fin)}

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

    def end_probe(a, out, line, e, tag, field):
        de = float(line["origin"][e][0])
        probes.append((tag, round(de, 1)))
        assert int(a.end_probe_iters) == probe and e == len(line) - 1
        return {"probe_way_on": int(round(de, 1) not in dead), "probe_iters": probe,
                "probe_envs": 2048, "probe_secs": 1.0, "probe_end_d": de,
                "probe_held_d": de - 50.0, "probe_gain": 200.0}

    monkeypatch.setattr(A, "beam", beam)
    monkeypatch.setattr(A, "mc", mcs)
    monkeypatch.setattr(A, "go_explore", go_explore)
    monkeypatch.setattr(A, "end_probe", end_probe)
    argv = ["--ckpt", "x.pt", "--map", "m.bsp", "--field", "f.npz", "--out", str(tmp_path),
            "--greedy-d", str(greedy), "--gain", "200", "--beam-ticks", "6127",
            "--envs", "2048", "--ge-minutes", "5", "--explorer", "mcts",
            "--current-d", str(current), "--mc-envs", "4096"] + GE_ARGS + list(extra)
    if probe is not None:
        argv += ["--end-probe-iters", str(probe)]
    if fallback is not None:
        argv += ["--failed-fallback", fallback]
    A.KEYS.clear()
    rc = A.main(argv)
    res = json.loads((tmp_path / "spine.json").read_text()) if rc == 0 else None
    return res, ran, probes


# ab4u_1 search 5 (3,279M): greedy 30,218, the held line (Monte Carlo k04) ends at 29,980; the
# tree's line is the SE-lip exit (29,502); the archive's line runs into the trough (28,4xx)
K5 = dict(greedy=30218.0, current=29980.0, tree=29502.0)


def test_off_is_the_old_decision(monkeypatch, tmp_path):
    res, ran, probes = _decide(monkeypatch, tmp_path, **K5)
    assert res["source"] == "beam" and ran == ["tree"] and probes == []
    assert not any("probe" in k for k in res)
    res0, _r, _p = _decide(monkeypatch, tmp_path, extra=("--end-probe-iters", "0"), **K5)
    assert res0 == res


def test_a_line_with_a_way_on_passes_as_before(monkeypatch, tmp_path):
    res, ran, probes = _decide(monkeypatch, tmp_path, probe=250, **K5)
    assert res["source"] == "beam" and ran == ["tree"] and probes == [("beam", 29502.0)]
    assert res["beam_probe_way_on"] == 1 and res["beam_probe_iters"] == 250


def test_a_trough_line_is_refused_and_the_next_stage_answers(monkeypatch, tmp_path):
    # the tree and the Monte Carlo lines run into the trough (no way on); the archive's line
    # does not: it is handed back
    res, ran, probes = _decide(monkeypatch, tmp_path, greedy=30218.0, current=29980.0,
                               tree=28517.0, mc=28255.0, ge=(29300.0,), probe=250,
                               dead=(28517.0, 28255.0))
    assert ran == ["tree", "mc", "ge:held"]
    assert probes == [("beam", 28517.0), ("mc", 28255.0), ("ge", 29300.0)]
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(29300.0)
    assert res["beam_probe_way_on"] == 0 and res["mc_probe_way_on"] == 0
    # nothing else: no line at all
    res, ran, _p = _decide(monkeypatch, tmp_path, greedy=30218.0, current=29980.0,
                           tree=28517.0, mc=28255.0, ge=(28400.0, 28300.0), probe=250,
                           dead=(28517.0, 28255.0, 28400.0, 28300.0))
    assert res is None and ran == ["tree", "mc", "ge:held", "ge:contact"]


def test_a_finishing_line_is_never_probed(monkeypatch, tmp_path):
    res, _ran, probes = _decide(monkeypatch, tmp_path, greedy=1977.0, current=1661.0,
                                tree=0.0, tree_fin=True, probe=250, dead=(0.0,))
    assert res["source"] == "beam" and probes == []


def test_only_a_line_that_would_be_handed_back_is_probed(monkeypatch, tmp_path):
    # the tree's line does not beat the greedy frontier: no probe for it
    res, _ran, probes = _decide(monkeypatch, tmp_path, greedy=30218.0, current=29980.0,
                                tree=30150.0, mc=29500.0, probe=250)
    assert probes == [("mc", 29500.0)] and res["source"] == "mc"


def test_a_fallback_is_probed_too(monkeypatch, tmp_path):
    # the tree's line beats the greedy frontier, not the held line: a fallback - unless the
    # probe finds no way on from it
    kw = dict(greedy=30218.0, current=28500.0, tree=29900.0, mc=None, ge=())
    res, _ran, _p = _decide(monkeypatch, tmp_path, probe=250, **kw)
    assert res["source"] == "beam" and res["fallback"] == 1
    res, _ran, probes = _decide(monkeypatch, tmp_path, probe=250, dead=(29900.0,), **kw)
    assert res is None and probes == [("beam", 29900.0)]


def test_the_stepping_stone_is_the_deepest_with_a_way_on(monkeypatch, tmp_path):
    # a patience search after the held archive line (28,500) failed: two archive lines beat
    # the greedy frontier, neither the held line; the deeper one has no way on
    kw = dict(greedy=30218.0, current=28500.0, tree=30300.0, mc=None, ge=(29000.0, 29500.0),
              fallback="-")
    res, _ran, _p = _decide(monkeypatch, tmp_path, **kw)
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(29000.0)
    res, _ran, probes = _decide(monkeypatch, tmp_path, probe=250, dead=(29000.0,), **kw)
    assert probes == [("ge", 29000.0), ("ge2", 29500.0)]
    assert res["source"] == "beam+ge2" and res["end_d"] == pytest.approx(29500.0)
    assert res["failed_fallback"] == 1


def test_the_flags_parse_with_their_defaults():
    import auto_spine as A
    src = (ROOT / "tools" / "auto_spine.py").read_text(encoding="utf-8")
    for flag, default in (("--end-probe-iters", "default=0"), ("--end-probe-envs", "default=2048"),
                          ("--end-probe-secs", "default=2.0"), ("--end-probe-roots", "default=16"),
                          ("--end-probe-gain", "default=None")):
        i = src.index(f'ap.add_argument("{flag}"')
        assert default in src[i:i + 120], flag
    assert callable(A.end_probe) and callable(A.probe_verdict)
