"""ab/v2 (2026-11-14): the failed lines survive a resume, and a dead end of ANY source is refused.

1. Persistence. ab/mc's d-bar, ab/ban's end positions and the held line's source lived only in the
   trainer's AUTO dict, so a resumed run forgot every failed line (sub-agent U). The checkpoint now
   carries them (``auto_spine_failed``, written whenever --auto-spine-archive-bar is on) and the
   resume restores them; the held line's source only when the resumed spine ends where the saved
   one did.
2. ``train_fast --auto-spine-dead-end-ban`` (needs --auto-spine-archive-ban R, same R): a PATIENCE
   search records where the held line ended WHATEVER its source; every later search refuses a line
   of any stage that ENDS within R of such an end while the greedy has not come within the gain of
   it (``auto_spine.py --dead-end-ban``).

Why the END and not the path (as ab/ban does for archive chains): a failed princessburglar line
ends ON the route - m2ge_pb1's first tree line (failed at 0.38B, walk-back stalled at 34%) ends
on the room-exit path that 9 of the 14 later lines pass within 8-31 u of, and they END 495-606 u
from it. The trough re-entries on unitfarmer2 END in the trough. Why the lift: m2ge_pb1's
pit-floor tree line k15 failed at 5.96B and the greedy reached the pit floor at 6.19B; the
crossings END 97 u (fxpb_t1) and 124 u (the all-in) from that pit floor's tree-line ends.

The cases are REAL lines (tests/python/data/dead_end_cases.npz: every fourth row of each line,
its root and last row kept, with its d on the trainer's real field; positions and d only; every
line is the loop's own search, no demo): the frozen recipe's six failed trough tree lines in a
row (m6sib k02-k07, then k08), the fix round's four (fxuf2 k02-k05), ab5u_1's trough tree line,
the all-in's failed lip line and sub-agent U's Monte Carlo exits, the failed tree lines before
the finishing lines on blue200 / petrus / cannonball (fix round), and m2ge_pb1's room line, its
room-to-pit line and its pit-floor line. princessburglar's crossings come from
archive_ban_cases.npz.
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

R = 128.0
DATA = ROOT / "tests" / "python" / "data"
GE_ARGS = ["--ge-cell-heading", "0", "--ge-view-follow", "none", "--ge-boost", "1.0",
           "--ge-held", "1", "--ge-cell-contact", "1"]


def _fx():
    out = {}
    for f in ("dead_end_cases.npz", "archive_ban_cases.npz"):
        z = np.load(DATA / f, allow_pickle=False)
        for n in sorted({k.split("__")[0] for k in z.files}):
            out[n] = (z[f"{n}__o"].astype(np.float64), z[f"{n}__d"].astype(np.float64),
                      int(z[f"{n}__root"]))
    return out


def _end(fx, name):
    o, d, _r = fx[name]
    return o[-1], float(d[-1])


def _end_dist(fx, cand, ends):
    from auto_spine import end_near
    return end_near(_end(fx, cand)[0], np.array([_end(fx, e)[0] for e in ends]))


# --- the trainer side ----------------------------------------------------------------------------

def test_train_only():
    import record_ckpt
    assert "auto_spine_dead_end_ban" in record_ckpt.TRAIN_ONLY


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-dead-end-ban"' in src
    assert ('raise SystemExit("--auto-spine-dead-end-ban refuses lines within '
            "--auto-spine-archive-ban's \"") in src
    assert 'meta["config"]["auto_spine_dead_end_ban"] = 1' in src
    i_mark = src.index('if pat and not quorum and mark_dead_end(_de_all, AUTO.get("src"), _hp, '
                       "_held,")
    i_live = src.index("_live, _gone = live_dead_ends(_de_all, med, float(args.auto_spine_gain))")
    i_tok = src.index('cmd += ["--dead-end-ban" + "=" + ";".join(')
    i_run = src.index("t0 = time.time()\n        if args.auto_spine_async")
    assert i_mark < i_live < i_tok < i_run
    # the failed lines are saved with every checkpoint of an --auto-spine-archive-bar run and
    # restored where AUTO is made
    assert 'state["auto_spine_failed"] = auto_failed_state(AUTO, demo.S[-1]["origin"])' in src
    i_auto = src.index('    AUTO = {"eval": None, "last": None, "retry_at": 0, "k": 0, '
                       '"field": None}\n')
    i_rest = src.index('_srcok = restore_auto_failed(AUTO, ck["auto_spine_failed"], '
                       'demo.S[-1]["origin"])')
    assert i_auto < i_rest < i_mark
    from train_fast import AUTO_SPINE_OWN_FLAGS
    assert "--dead-end-ban" in AUTO_SPINE_OWN_FLAGS


def test_a_dead_end_is_recorded_once_never_at_the_goal_nor_the_spawn():
    from train_fast import mark_dead_end
    ends = []
    assert mark_dead_end(ends, "beam", [-1755.0, -1205.0, 196.0], 28314.5, 32.0)
    assert not mark_dead_end(ends, "beam", [-1755.4, -1205.0, 196.0], 28314.5, 32.0)   # again
    assert mark_dead_end(ends, "mc", [-1339.0, -945.0, 237.0], 27792.9, 32.0)
    assert not mark_dead_end(ends, "beam+ge", [2028.0, 1206.0, 317.0], 0.0, 32.0)   # the goal
    assert not mark_dead_end(ends, "beam", [0.0, 0.0, 0.0], float("inf"), 32.0)
    assert not mark_dead_end(ends, None, [-2176.0, -2272.0, 1192.0], 30748.0, 32.0)  # spine0
    assert ends == [[-1755.0, -1205.0, 196.0, 28314.5], [-1339.0, -945.0, 237.0, 27792.9]]


def test_the_greedy_reaching_a_dead_end_lifts_it():
    # m2ge_pb1's pit-floor tree line k15 (55,301.5) failed at 5.96B with the greedy in the start
    # room (56,960); the greedy reached the pit floor at 6.19B (55,350)
    from train_fast import live_dead_ends
    ends = [[808.0, -3285.0, 3377.0, 55301.5]]
    assert live_dead_ends(ends, 56960.0, 200.0) == (ends, [])
    assert live_dead_ends(ends, 55350.0, 200.0) == ([], ends)
    # unitfarmer2: the greedy dives to 30,245-30,328 and never comes near the trough (28,315)
    trough = [[-1755.0, -1205.0, 196.0, 28314.5]]
    assert live_dead_ends(trough, 30245.0, 200.0)[0] == trough


def test_the_failed_lines_survive_a_resume():
    from train_fast import auto_failed_state, restore_auto_failed
    held = np.array([-1706.0, -1178.0, 197.0])
    auto = {"failed_arch": [28516.7], "failed_arch_pos": [[-1969.0, -1083.0, 295.0]],
            "dead_ends": [[-1969.0, -1083.0, 295.0, 28516.7]], "src": "beam+ge2", "k": 3}
    st = json.loads(json.dumps(auto_failed_state(auto, held)))     # plain data in the checkpoint
    new = {"k": 0}
    assert restore_auto_failed(new, st, held + 0.4)
    assert new["failed_arch"] == [28516.7] and new["src"] == "beam+ge2"
    assert new["failed_arch_pos"] == [[-1969.0, -1083.0, 295.0]]
    assert new["dead_ends"] == [[-1969.0, -1083.0, 295.0, 28516.7]]
    from train_fast import archive_bar
    assert archive_bar(new["failed_arch"]) == 28516.7               # the d-bar is back
    other = {}
    assert not restore_auto_failed(other, st, held + 500.0)         # another held line
    assert other["failed_arch"] == [28516.7] and "src" not in other
    assert restore_auto_failed({}, {"failed_arch": []}, held) is False   # an old checkpoint's


# --- the geometry of the real lines at R = 128 ---------------------------------------------------

def test_the_fixture_is_the_real_lines():
    fx = _fx()
    runs = Path("/home/steam/ar_m2/runs")
    src = {"ab5u_k02": "ab5u/auto_P/k02", "m6_k03": "m6sib/uf2_auto/k03",
           "fx_k05": "fxuf2/auto_t1/k05", "b2_k03": "fxb200/auto_t1/k03",
           "g_k15": "m2ge/m2ge_pb1/auto_lines/k15"}
    for name, rel in src.items():
        js = runs / rel / "spine.json"
        if not js.exists():
            pytest.skip("the run bundles are not on this machine")
        assert _end(fx, name)[1] == pytest.approx(json.loads(js.read_text())["end_d"], abs=0.5)


DEAD_AGAIN = [   # (the next line, the failed lines before it), its end's distance to them
    ("m6_k03", ("m6_k02",), 68.0),
    ("m6_k04", ("m6_k02", "m6_k03"), 31.0),
    ("m6_k05", ("m6_k02", "m6_k03", "m6_k04"), 114.0),
    ("m6_k06", ("m6_k02", "m6_k03", "m6_k04", "m6_k05"), 37.0),
    ("fx_k05", ("fx_k02", "fx_k03", "fx_k04"), 69.0)]


@pytest.mark.parametrize("cand,ends,dist", DEAD_AGAIN)
def test_the_same_trough_again_ends_within_the_radius(cand, ends, dist):
    fx = _fx()
    got = _end_dist(fx, cand, ends)
    assert got == pytest.approx(dist, abs=3.0) and got < R


UF2_DEAD = ("ab5u_k02", "m6_k02", "m6_k03", "m6_k04", "m6_k05", "m6_k06", "m6_k07", "fx_k02",
            "fx_k03", "fx_k04", "fx_k05", "aa_k03")
OPEN = [
    ("m6_k07", ("m6_k02", "m6_k03", "m6_k04", "m6_k05", "m6_k06"), 355.0),  # a deeper trough spot
    ("m6_k08", ("m6_k02", "m6_k03", "m6_k04", "m6_k05", "m6_k06", "m6_k07"), 150.0),
    ("fx_k03", ("fx_k02",), 465.0),
    ("fx_k04", ("fx_k02", "fx_k03"), 135.0),
    ("k05r_mc", UF2_DEAD, 182.0),      # U's Monte Carlo exits, beside the all-in's failed lip line
    ("mc30_s0", UF2_DEAD, 317.0),
    ("mc30_s1", UF2_DEAD, 223.0),
    ("mc30_s2", UF2_DEAD, 168.0),
    ("b2_k03", ("b2_k02",), 1652.0),   # blue200's goal line after its failed tree line
    ("pe_k03", ("pe_k02",), 4167.0),   # petrus's search 3 after its failed tree line
    ("ca_k04", ("ca_k02",), 15497.0),  # cannonball's finishing curriculum line
    ("g_k02", ("g_k01",), 606.0)]      # princessburglar: room-to-pit beside the failed room line


@pytest.mark.parametrize("cand,ends,dist", OPEN)
def test_the_lines_that_must_stay_open_end_outside_it(cand, ends, dist):
    fx = _fx()
    got = _end_dist(fx, cand, ends)
    assert got == pytest.approx(dist, abs=3.0) and got > R


def test_the_margins_and_the_pit_floor_risk():
    fx = _fx()
    refused = max(_end_dist(fx, c, e) for c, e, _d in DEAD_AGAIN)
    kept = min(_end_dist(fx, c, e) for c, e, _d in OPEN)
    assert refused == pytest.approx(114.0, abs=3.0)       # m6sib search 5
    assert kept == pytest.approx(135.0, abs=3.0)          # fxuf2 search 4, a trough line kept
    # the risk the lift is for: princessburglar's crossings end beside m2ge_pb1's pit-floor
    # tree-line ends (they would be refused while the greedy has not reached the pit floor)
    assert _end_dist(fx, "t1_k02", ("g_k02",)) == pytest.approx(97.0, abs=3.0)
    assert _end_dist(fx, "ap_k01", ("g_k15",)) == pytest.approx(124.0, abs=3.0)
    # ab5u_1's trough line ends within R of 5 of the 11 trough tree-line ends of the frozen
    # recipe and the fix round (m6sib k02-k08, fxuf2 k02-k05)
    others = ("m6_k02", "m6_k03", "m6_k04", "m6_k05", "m6_k06", "m6_k07", "m6_k08", "fx_k02",
              "fx_k03", "fx_k04", "fx_k05")
    near = sorted(n for n in others if _end_dist(fx, n, ("ab5u_k02",)) < R)
    assert near == ["fx_k02", "fx_k05", "m6_k02", "m6_k03", "m6_k04"]


# --- auto_spine.main replayed on the real lines (stages stubbed, a nearest-neighbour field) ------

class _Field:
    """d of a point by its nearest fixture neighbour."""
    reach_max, cell = 1e9, 32.0

    def __init__(self, fx, names):
        self.o = np.concatenate([fx[n][0] for n in names])
        self.d = np.concatenate([fx[n][1] for n in names])

    def sample(self, o):
        o = np.asarray(o, np.float64).reshape(-1, 3)
        return np.array([self.d[int(np.argmin(np.linalg.norm(self.o - p, axis=1)))] for p in o])


def _states(origins):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(len(origins), STATE_DTYPE)
    s["origin"] = np.asarray(origins, np.float32)
    return s


def _decide(monkeypatch, tmp_path, capsys, fx, fields, greedy, current, tree, mc, ge=(),
            dead=(), finished=()):
    """``tree`` / ``mc`` / ``ge`` = fixture names (or None); ``dead`` = the failed lines whose
    ends are passed as --dead-end-ban. -> (the AUTO_SPINE result, the stages that ran)."""
    import auto_spine as A
    ran = []
    monkeypatch.setattr(A, "ckpt_tick", lambda _c: 7.6667)
    monkeypatch.setattr(A, "load_field", lambda _p: _Field(fx, fields))

    def _line(name):
        return None if name is None else _states(fx[name][0])
    monkeypatch.setattr(A, "beam", lambda a, out, branch_tick=0, tag=None: (
        ran.append("tree") or _line(tree),
        {"beam_rc": 0, "beam_replay_ok": 1, "beam_finished": int(tree in finished)}))
    monkeypatch.setattr(A, "mc", lambda a, out, tag="mc": (
        ran.append("mc") or _line(mc),
        {"mc_rc": 0, "mc_replay_ok": 1, "mc_finished": int(mc in finished)}))
    gq = list(ge)

    def go_explore(a, out, line, end, boosts=None, tag="ge", line_keys=None, roots=None,
                   pick=None, kind=None):
        ran.append(f"ge:{kind}")
        name = gq.pop(0) if gq else None
        if name is None:
            return None, {"ge_rc": 1}
        o, _d, r = fx[name]
        return _states(o), {"ge_rc": 0, "ge_kind": kind, "ge_root_index": r}

    monkeypatch.setattr(A, "go_explore", go_explore)
    argv = ["--ckpt", "x.pt", "--map", "m.bsp", "--field", "f.npz", "--out", str(tmp_path),
            "--greedy-d", str(greedy), "--gain", "200", "--beam-ticks", "6127",
            "--envs", "2048", "--ge-minutes", "5", "--explorer", "mcts",
            "--current-d", str(current), "--mc-envs", "4096", "--mc-keys-temp", "1.5"] + GE_ARGS
    if dead:
        pts = ";".join(",".join(f"{v:.1f}" for v in _end(fx, n)[0]) for n in dead)
        argv += [f"--dead-end-ban={pts}", "--archive-ban-r", f"{R:g}"]
    A.KEYS.clear()
    capsys.readouterr()
    A.main(argv)
    out = capsys.readouterr().out
    res = json.loads(out.split("AUTO_SPINE ", 1)[1].splitlines()[0])
    return res, ran


UF2 = ["ab5u_k02", "m6_k03", "k05r_mc", "mc30_s2"]


def test_ab5u_next_search_hands_back_the_exit_not_the_trough(monkeypatch, tmp_path, capsys):
    # ab5u_1 after its trough tree line (28,314.5) failed: the greedy dives (30,252); the tree
    # re-enters the trough (a real re-entry: the frozen recipe's search-3 line, ending 43 u from
    # ab5u's end) and the Monte Carlo stage finds U's exit (29,910). Without the rule the trough
    # line is the fallback (it beats the greedy frontier, and it is deeper than the exit)
    fx = _fx()
    kw = dict(greedy=30252.0, current=28314.5, tree="m6_k03", mc="k05r_mc")
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, UF2, **kw)
    assert res["source"] == "beam" and res["fallback"] == 1
    assert res["end_d"] == pytest.approx(28276.3, abs=1)
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, UF2, dead=("ab5u_k02",), **kw)
    assert res["beam_dead_end"] == 1 and res["beam_dead_end_dist"] == pytest.approx(43, abs=3)
    assert res["source"] == "mc" and res["fallback"] == 1
    assert res["end_d"] == pytest.approx(29909.9, abs=1)
    assert res["mc_dead_end_dist"] > R


def test_m6sib_search3_refuses_the_same_trough(monkeypatch, tmp_path, capsys):
    # the frozen recipe's search 3 (greedy 29,444): its tree line k03 ends 68 u from the failed
    # k02's end; with nothing else found there is no line instead of the same dead end
    fx = _fx()
    kw = dict(greedy=29444.0, current=28349.9, tree="m6_k03", mc=None)
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, ["m6_k02", "m6_k03"], **kw)
    assert res["source"] == "beam"
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, ["m6_k02", "m6_k03"],
                        dead=("m6_k02",), **kw)
    assert res["spine"] is None and res["beam_dead_end_dist"] == pytest.approx(68, abs=3)


@pytest.mark.parametrize("name,failed,greedy,current", [
    ("b2_k03", "b2_k02", 1977.3, 1660.9),       # blue200 search 3: the goal line
    ("pe_k03", "pe_k02", 30542.5, 28058.3)])    # petrus search 3
def test_blue200_and_petrus_lines_still_pass(monkeypatch, tmp_path, capsys, name, failed, greedy,
                                            current):
    # both after a failed TREE line, now a recorded dead end
    fx = _fx()
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, [failed, name], greedy=greedy,
                        current=current, tree=failed, mc=None, ge=(name,), dead=(failed,))
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(_end(fx, name)[1], abs=1)
    assert res["beam_dead_end"] == 1                     # the failed line itself again: refused
    assert res["ge_dead_end_dist"] > 1000.0


def test_pb_room_to_pit_line_passes_beside_a_failed_room_line(monkeypatch, tmp_path, capsys):
    # m2ge_pb1 search 2: the room line k01 (56,724) failed with the greedy in the room (56,968);
    # the tree's room-to-pit line k02 passes 8 u from k01's end on the way and ends 606 u away
    fx = _fx()
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, ["g_k01", "g_k02"], greedy=56968.0,
                        current=56724.2, tree="g_k02", mc=None, dead=("g_k01",))
    assert res["source"] == "beam" and res["end_d"] == pytest.approx(55302.1, abs=1)
    assert res["beam_dead_end_dist"] == pytest.approx(606, abs=3)


def test_pb_crossing_waits_for_the_greedy_at_the_pit_floor(monkeypatch, tmp_path, capsys):
    # the risk: a pit-floor tree line failed (m2ge_pb1 k15) and the all-in's crossing ends 124 u
    # from it. While the greedy is in the room the end is live and the crossing is refused; once
    # the greedy reaches the pit floor the trainer drops the end (live_dead_ends) and it passes
    from train_fast import live_dead_ends
    fx = _fx()
    end = [float(v) for v in _end(fx, "g_k15")[0]] + [_end(fx, "g_k15")[1]]
    kw = dict(current=56724.2, tree="g_k15", mc=None, ge=("ap_k01",))
    assert live_dead_ends([end], 56960.0, 200.0)[0] == [end]
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, ["g_k15", "ap_k01"], greedy=56960.0,
                        dead=("g_k15",), **kw)
    assert res["spine"] is None and res["beam_dead_end"] == 1 and res["ge_dead_end"] == 1
    assert res["ge_dead_end_dist"] == pytest.approx(124, abs=3)
    assert live_dead_ends([end], 55450.0, 200.0)[0] == []
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, ["g_k15", "ap_k01"], greedy=55450.0,
                        **kw)
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(55106.6, abs=1)


def test_a_finishing_line_is_never_a_dead_end(monkeypatch, tmp_path, capsys):
    fx = _fx()
    res, _ran = _decide(monkeypatch, tmp_path, capsys, fx, ["m6_k02", "m6_k03"], greedy=29444.0,
                        current=28349.9, tree="m6_k03", mc=None, dead=("m6_k02",),
                        finished=("m6_k03",))
    assert res["source"] == "beam" and "beam_dead_end" not in res
