"""ab/ban (2026-11-14): refuse an archive line whose chain runs through a failed archive line's end.

Recipe v1's unitfarmer2 run ab4u_1 (code ab/mc) adopted an archive TROUGH line at search 2 (end
(-1,969, -1,083, 295), d 28,517); its curriculum failed, and search 3 adopted a second trough
line (end (-1,706, -1,178, 197), d 28,255) that cleared ab/mc's d-bar (28,317) by 62 u. Its chain
runs within 112 u of the failed line's end. The d-bar compares depths only; this adds where.

* auto_spine.py ``--archive-ban 'X,Y,Z;...' --archive-ban-r R``: an archive line whose chain (from
  its root on) passes within R of a failed archive line's END POSITION is refused, as a pass and
  as a ``--failed-fallback``; the policy's own lines (tree, Monte Carlo) are not held to it;
* train_fast ``--auto-spine-archive-ban R`` (needs --auto-spine-archive-bar) records the end
  position of every archive line ab/mc marks failed and passes them to every later search.

The cases are REAL lines (tests/python/data/archive_ban_cases.npz: every second row of each line,
its end and root kept, and its d on the trainer's real field - positions and d only; every line is
the loop's own search, no demo): ab4u_1 searches 2-3, the all-in's unitfarmer2 searches 2/4/5,
sub-agent U's search-5 replays (ab/mc a6383c75), the fix round's princessburglar searches 1-2
(fxpb_t1), the all-in's princessburglar search 1, and the archive lines that finished blue200
(fxb200_t1 search 3) and moved petrus (fxpet_t1 search 3). One radius, R = 128 u (two 64 u archive
cells): every trough line that re-enters a failed end passes within 3-112 u of it; the lines that
must stay open keep 163 u (a room-to-pit line beside the all-in's failed crossing end) and 210 u
(fxpb_t1's crossing beside its failed pit line) away.
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
FIX = ROOT / "tests" / "python" / "data" / "archive_ban_cases.npz"
GE_ARGS = ["--ge-cell-heading", "0", "--ge-view-follow", "none", "--ge-boost", "1.0",
           "--ge-held", "1", "--ge-cell-contact", "1"]


def _fx():
    z = np.load(FIX, allow_pickle=False)
    names = sorted({k.split("__")[0] for k in z.files})
    return {n: (z[f"{n}__o"].astype(np.float64), z[f"{n}__d"].astype(np.float64),
                int(z[f"{n}__root"])) for n in names}


def _end(fx, name):
    o, d, _r = fx[name]
    return o[-1], float(d[-1])


def _chain_dist(fx, cand, ends):
    from auto_spine import chain_near
    o, _d, r = fx[cand]
    return chain_near(o[r:], np.array([_end(fx, e)[0] for e in ends]))[0]


def test_train_only():
    import record_ckpt
    assert "auto_spine_archive_ban" in record_ckpt.TRAIN_ONLY


def test_the_trainer_wires_it():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--auto-spine-archive-ban"' in src
    assert ('raise SystemExit("--auto-spine-archive-ban records the failed archive lines of "'
            in src)
    assert 'meta["config"]["auto_spine_archive_ban"] = float(args.auto_spine_archive_ban)' in src
    i_mark = src.index("if pat and not quorum and mark_failed_archive(_fa, AUTO.get(\"src\"), "
                       "_held):")
    i_pos = src.index('AUTO.setdefault("failed_arch_pos", []).append(_hp)')
    i_ban = src.index('cmd += ["--archive-ban" + "=" + _pts,')
    i_run = src.index("t0 = time.time()\n        if args.auto_spine_async")
    assert i_mark < i_pos < i_ban < i_run
    from train_fast import AUTO_SPINE_OWN_FLAGS
    assert {"--archive-ban", "--archive-ban-r"} <= AUTO_SPINE_OWN_FLAGS


def test_the_fixture_is_the_real_lines():
    fx = _fx()
    for name, d_end in (("u4_k02", 28516.7), ("u4_k03", 28254.6), ("aa_k02", 28599.4),
                        ("aa_k04", 28936.6), ("aa_k05", 28713.5), ("t1_k01", 55353.0),
                        ("t1_k02", 55098.2), ("ap_k01", 55106.6), ("b2_k03", 0.0),
                        ("pe_k03", 22364.4)):
        assert _end(fx, name)[1] == pytest.approx(d_end, abs=0.2), name


# which real chains run through a failed end (refused at R) and which keep away (open)
REFUSED = [("u4_k03", ["u4_k02"]),                      # ab4u_1 search 3: 112 u
           ("aa_k05", ["aa_k02", "aa_k04"]),            # the all-in's search 5: 3 u from k04's
           ("k05r30_ge2", ["aa_k02", "aa_k04"]),        # U's replays at search 5: troughs
           ("k05rE_ge", ["aa_k02", "aa_k04"]),
           ("k05rE_ge2", ["aa_k02", "aa_k04"]),
           ("k05rE_ge4", ["aa_k02", "aa_k04"])]
OPEN = [("t1_k02", ["t1_k01"]),                         # fxpb_t1 search 2: the crossing, 210 u
        ("t1_k01", ["ap_k01"]),                         # a room-to-pit line beside the all-in's
        ("k05r_ge", ["aa_k02", "aa_k04"]),              # U's ledge / lip lines: far
        ("k05r_ge2", ["aa_k02", "aa_k04"]),
        ("aa_k04", ["aa_k02"])]                         # another trough branch: 581 u (d-bar)


@pytest.mark.parametrize("cand,ends", REFUSED)
def test_trough_reentries_pass_within_the_radius(cand, ends):
    assert _chain_dist(_fx(), cand, ends) < R


@pytest.mark.parametrize("cand,ends", OPEN)
def test_open_lines_keep_away(cand, ends):
    assert _chain_dist(_fx(), cand, ends) >= R


def test_the_margins():
    fx = _fx()
    worst_refused = max(_chain_dist(fx, c, e) for c, e in REFUSED)
    best_open = min(_chain_dist(fx, c, e) for c, e in OPEN[:2])
    assert worst_refused == pytest.approx(112.0, abs=3.0)   # ab4u_1 search 3
    assert best_open == pytest.approx(163.0, abs=3.0)       # the room-to-pit line


# --- auto_spine.main replayed on the real lines (stages stubbed, a nearest-neighbour field) ---

class _Field:
    """d of a fixture point by nearest neighbour; a stand-in policy line far away reads d = x - 1e6."""
    reach_max, cell = 1e9, 32.0

    def __init__(self, fx, names):
        self.o = np.concatenate([fx[n][0] for n in names])
        self.d = np.concatenate([fx[n][1] for n in names])

    def sample(self, o):
        o = np.asarray(o, np.float64).reshape(-1, 3)
        out = np.empty(len(o))
        far = o[:, 0] > 5e5
        out[far] = o[far, 0] - 1e6
        for i in np.flatnonzero(~far):
            out[i] = self.d[int(np.argmin(np.linalg.norm(self.o - o[i], axis=1)))]
        return out


def _states(origins):
    from surfgym.core import STATE_DTYPE
    s = np.zeros(len(origins), STATE_DTYPE)
    s["origin"] = np.asarray(origins, np.float32)
    return s


def _policy_line(d_end, n=60):
    """A stand-in tree / Monte Carlo line, far from every map point, ending at d = d_end."""
    o = np.zeros((n, 3))
    o[:, 0] = 1e6 + np.linspace(d_end + 2000.0, d_end, n)
    return _states(o)


def _decide(monkeypatch, tmp_path, fx, field_names, greedy, current, tree, mc, ge, bar=None,
            failed=None, ban=()):
    """``ge`` = fixture names (or None) for the held and contact archive stages, in order."""
    import auto_spine as A
    ran = []
    monkeypatch.setattr(A, "ckpt_tick", lambda _c: 7.6667)
    monkeypatch.setattr(A, "load_field", lambda _p: _Field(fx, field_names))
    monkeypatch.setattr(A, "beam", lambda a, out, branch_tick=0, tag=None: (
        ran.append("tree") or _policy_line(tree),
        {"beam_rc": 0, "beam_replay_ok": 1, "beam_finished": 0}))
    monkeypatch.setattr(A, "mc", lambda a, out, tag="mc": (
        ran.append("mc") or _policy_line(mc), {"mc_rc": 0, "mc_replay_ok": 1, "mc_finished": 0}))
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
    if bar is not None:
        argv += ["--archive-bar", str(bar)]
    if failed is not None:
        argv += ["--failed-fallback", failed]
    if ban:
        pts = ";".join(",".join(f"{v:.1f}" for v in _end(fx, b)[0]) for b in ban)
        argv += [f"--archive-ban={pts}", "--archive-ban-r", f"{R:g}"]
    A.KEYS.clear()
    rc = A.main(argv)
    res = json.loads((tmp_path / "spine.json").read_text()) if rc == 0 else None
    return res, ran


UF2 = ["u4_k02", "u4_k03", "aa_k02", "aa_k04", "aa_k05", "k05rE_ge4"]


def test_ab4u_search3_refuses_the_second_trough_line(monkeypatch, tmp_path):
    # ab4u_1 search 3 (1,581.3M): greedy 30,288.8, the held trough line k02 (28,516.7) failed;
    # tree 29,738 (the fallback), Monte Carlo 30,044; the contact archive's trough line k03
    fx = _fx()
    kw = dict(greedy=30288.8, current=28516.7, tree=29738.0, mc=30044.0, ge=(None, "u4_k03"),
              bar=28516.7, failed="28516.7")
    res, _ran = _decide(monkeypatch, tmp_path, fx, UF2, **kw)
    assert res["source"] == "beam+ge2" and res["end_d"] == pytest.approx(28254.6, abs=1)
    res, _ran = _decide(monkeypatch, tmp_path, fx, UF2, ban=("u4_k02",), **kw)
    assert res["source"] == "beam" and res["fallback"] == 1          # the tree's line instead
    assert res["ge2_banned"] == 1 and res["ge2_ban_dist"] == pytest.approx(112, abs=3)


def test_allin_search5_still_hands_back_the_monte_carlo_exit(monkeypatch, tmp_path):
    # the all-in's search 5 with both failed trough ends: k05 (3 u from k04's end) and U's energy
    # replay's 28,430 trough line are refused; the Monte Carlo exit is handed back
    fx = _fx()
    res, _ran = _decide(monkeypatch, tmp_path, fx, UF2, greedy=30300.6, current=28936.6,
                        tree=30099.9, mc=29834.0, ge=("aa_k05", "k05rE_ge4"), bar=28599.4,
                        failed="28599.4,28936.6", ban=("aa_k02", "aa_k04"))
    assert res["source"] == "mc" and res["fallback"] == 1
    assert res["ge_banned"] == 1 and res["ge2_banned"] == 1


def test_a_trough_line_past_the_bar_is_still_refused(monkeypatch, tmp_path):
    # no policy fallback and the reference at the bar: the d-bar alone takes the 28,255 line,
    # the ban refuses it -> no line
    fx = _fx()
    kw = dict(greedy=29900.0, current=28516.7, tree=29850.0, mc=29860.0, ge=(None, "u4_k03"),
              bar=28516.7, failed="28516.7")
    res, _ran = _decide(monkeypatch, tmp_path, fx, UF2, **kw)
    assert res["source"] == "beam+ge2"
    res, _ran = _decide(monkeypatch, tmp_path, fx, UF2, ban=("u4_k02",), **kw)
    assert res is None


PB = ["t1_k01", "t1_k02", "ap_k01"]


def test_pb_crossing_still_passes(monkeypatch, tmp_path):
    # fxpb_t1 search 2: the failed pit line k01 (55,353); the crossing k02 keeps 210 u away
    fx = _fx()
    res, _ran = _decide(monkeypatch, tmp_path, fx, PB, greedy=56962.4, current=55353.0,
                        tree=56900.0, mc=56850.0, ge=("t1_k02",), bar=55353.0, failed="55353.0",
                        ban=("t1_k01",))
    assert res["source"] == "beam+ge" and res["end_d"] == pytest.approx(55098.2, abs=1)
    assert res["ge_ban_dist"] == pytest.approx(210, abs=3)


def test_pb_room_to_pit_stepping_stone_still_passes(monkeypatch, tmp_path):
    # the all-in's search 2: its held crossing (55,106.6) failed; a room-to-pit archive line (the
    # real one is not saved: fxpb_t1's first line, the same kind, 55,353) keeps 163 u from that
    # end and comes back as the stepping-stone fallback (ab/ff)
    fx = _fx()
    res, _ran = _decide(monkeypatch, tmp_path, fx, PB, greedy=57004.3, current=55106.6,
                        tree=56911.0, mc=56913.0, ge=("t1_k01",), bar=55106.6, failed="55106.6",
                        ban=("ap_k01",))
    assert res["source"] == "beam+ge" and res["failed_fallback"] == 1
    assert res["end_d"] == pytest.approx(55353.0, abs=1)
    assert res["ge_ban_dist"] == pytest.approx(163, abs=3)


@pytest.mark.parametrize("name,fields,greedy,current", [
    ("b2_k03", ["b2_k03"], 1977.3, 1660.9),       # fxb200_t1 search 3: the goal line
    ("pe_k03", ["pe_k03"], 30542.5, 28058.3)])    # fxpet_t1 search 3
def test_blue200_and_petrus_archive_lines_still_pass(monkeypatch, tmp_path, name, fields,
                                                    greedy, current):
    # both after a failed TREE line: no failed archive end, no ban - as before
    fx = _fx()
    res, _ran = _decide(monkeypatch, tmp_path, fx, fields, greedy=greedy, current=current,
                        tree=current + 50.0, mc=current + 60.0, ge=(name,), failed="-")
    assert res["source"] == "beam+ge"
    assert res["end_d"] == pytest.approx(_end(fx, name)[1], abs=1)
    assert "ge_banned" not in res
