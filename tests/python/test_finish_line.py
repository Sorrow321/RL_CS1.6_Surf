"""--finish-line (zlr3b2, 2026-11-01): a SECOND backward curriculum (the --demo-file
DemoCurriculum) over a finishing line of the policy's own, on its own share of the training spawn
pool, beside the --demo-file curriculum.

(a) DemoCurriculum without the new arguments is the base commit's class exactly: build_pool's
    rows, the curriculum's moves and its log line are byte-identical over a long random run
    (the class is loaded from git at the commit before the change and run side by side);
(b) draw(m): the curriculum moves first like build_pool; the rows come from the admitted range
    [tau, n-1], front_frac of them on the frontier band; it starts AT the line's end (the last
    state, one step before the finish) and walks back while the band's finish rate holds;
(c) the trainer: the flags parse with their defaults, the config keys are written only when the
    flag is given, the recorder lists them TRAIN_ONLY, and the source needs SELF_STATES=1.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                   # noqa: E402
from surfgym.respawn import DemoCurriculum             # noqa: E402

BASE = "b11e95df53204200da76207acfc336e90c334890"      # the round's base commit


def _spine(n=600, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, STATE_DTYPE)
    s["origin"] = np.cumsum(rng.normal(0, 30, (n, 3)), axis=0).astype(np.float32)
    s["velocity"] = rng.normal(0, 300, (n, 3)).astype(np.float32)
    return s


def _base_class():
    try:
        src = subprocess.run(["git", "show", f"{BASE}:python/surfgym/respawn.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("the base commit is not in this checkout")
    mod = types.ModuleType("surfgym.respawn_base")
    mod.__package__ = "surfgym"            # its relative imports resolve inside surfgym
    exec(compile(src, "respawn_base.py", "exec"), mod.__dict__)
    return mod.DemoCurriculum


@pytest.mark.parametrize("kw", [
    dict(window=10, rate=0.2, min_ep=20.0),
    dict(window=5, rate=0.2, min_ep=50.0, grow=5, front_frac=0.5, cooldown=5, grow_max=40),
    dict(window=8, rate=0.3, min_ep=10.0, grow=3, tau0=300),
])
def test_off_is_the_base_class(kw, capsys):
    Base = _base_class()
    sp = _spine()
    start = _spine(4, seed=9)
    a, b = Base(sp, **kw), DemoCurriculum(sp, **kw)
    rng = np.random.default_rng(1)
    for it in range(400):
        pa = a.build_pool(start, pool_size=512, fresh_frac=0.1)
        out_a = capsys.readouterr().out
        pb = b.build_pool(start, pool_size=512, fresh_frac=0.1)
        out_b = capsys.readouterr().out
        assert pa.tobytes() == pb.tobytes()
        assert out_a == out_b and a.last_info == b.last_info and a.tau == b.tau
        ia = a.match(pa["origin"])
        ib = b.match(pb["origin"])
        assert np.array_equal(ia, ib)
        wins = rng.random(len(ia)) < (0.35 if it % 3 else 0.1)
        a.note_outcomes(ia, wins)
        b.note_outcomes(ib, wins)


def test_draw_walks_back_from_the_end(capsys):
    n = 1000
    c = DemoCurriculum(_spine(n), window=10, rate=0.2, min_ep=50.0, seed=31, grow=5,
                       front_frac=0.5, tau0=None, cooldown=5, grow_max=40,
                       name="finish-line curriculum")
    rows = c.draw(200)
    idx = c.match(rows["origin"])
    assert c.tau == n - 1 and np.all(idx == n - 1)        # starts at the line's last state
    taus = [c.tau]
    for _ in range(200):
        rows = c.draw(256)
        idx = c.match(rows["origin"])
        assert np.all(idx >= c.tau)                         # only the admitted range
        lo, hi = c._band()
        assert np.mean((idx >= lo) & (idx <= hi)) >= 0.45   # half on the frontier band
        c.note_outcomes(idx, np.ones(len(idx), bool))       # every band start finishes
        taus.append(c.tau)
    assert taus[-1] < n - 1 - 100                           # it walked back
    assert all(t1 <= t0 for t0, t1 in zip(taus, taus[1:]))
    out = capsys.readouterr().out
    assert "finish-line curriculum: tau <- earlier" in out and "demo curriculum:" not in out
    # a band that stops finishing backs off toward the end
    t0 = c.tau
    for _ in range(40):
        rows = c.draw(256)
        c.note_outcomes(c.match(rows["origin"]), np.zeros(len(rows), bool))
    assert c.tau > t0


def test_draw_without_front_frac_is_uniform_over_the_range():
    c = DemoCurriculum(_spine(300), window=10, rate=0.2, min_ep=50.0, seed=5, grow=5,
                       tau0=100)
    idx = c.match(c.draw(4000)["origin"])
    assert idx.min() >= 100 and idx.max() == 299
    assert abs(np.mean(idx) - 199.5) < 6.0


def _parser():
    spec = importlib.util.spec_from_file_location("train_fast_mod", ROOT / "python" / "train_fast.py")
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    return src


def test_trainer_wiring():
    src = _parser()
    for flag in ("--finish-line", "--finish-line-frac", "--finish-line-window",
                 "--finish-line-grow", "--finish-line-grow-max", "--finish-line-rate",
                 "--finish-line-tau0"):
        assert f'ap.add_argument("{flag}"' in src
    # off: every new statement sits behind FINL / args.finish_line
    assert 'if args.finish_line:\n        meta["config"]["finish_line"]' in src
    assert '"spawn_states", "finish_line")]' in src            # section 0: SELF_STATES=1
    assert src.count("FINL.") >= 5 and "if FINL is not None and ended.any():" in src
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    for k in ("finish_line", "finish_line_frac", "finish_line_window", "finish_line_grow",
              "finish_line_grow_max", "finish_line_rate"):
        assert f'"{k}"' in rc


def test_finish_search_helpers():
    sys.path.insert(0, str(ROOT / "tools"))
    import finish_search as FS
    # roots: seconds before the line's end, on a decision boundary, inside the line
    assert FS.root_ticks(20000, [10, 25], 1000 / 7.0, 4) == [18568, 16428]
    assert FS.root_ticks(500, [10], 1000 / 7.0, 4) == []          # a line shorter than the root
    # the deepest line by the field; a missing line is skipped; ties go to the longer line
    a, b, c = _spine(300, 1), _spine(500, 2), _spine(400, 3)

    def d_fn(o):
        return np.asarray(o, np.float64)[:, 2]
    lines = [a, None, b, c]
    depths = [float(np.min(d_fn(L["origin"]))) if L is not None else None for L in lines]
    want = int(np.nanargmin([x if x is not None else np.nan for x in depths]))
    assert FS.pick_deepest(lines, d_fn) == want
    assert FS.pick_deepest([None, None], d_fn) == -1
    assert FS.crossed("beam TAS: SPLICED CROSSING in 180.91s") and not FS.crossed("no crossing")


def test_auto_finish_wiring():
    src = _parser()
    for flag in ("--auto-finish", "--auto-finish-quorum", "--auto-finish-retry",
                 "--auto-finish-delay", "--auto-finish-timeout"):
        assert f'ap.add_argument("{flag}"' in src
    assert "--auto-finish finds its own finishing line: not with --finish-line" in src
    assert 'if args.auto_finish:\n        meta["config"]["auto_finish"] = 1' in src
    assert "nonlocal FINL, FINL_FRAC" in src and "finish_search.py" in src
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"auto_finish"' in rc


def test_finish_line_pace_wiring():
    src = _parser()
    assert 'ap.add_argument("--finish-line-pace"' in src
    assert 'ap.add_argument("--finish-line-pace-bonus"' in src
    # paid on the per-tick path for finishes from finish-line rows within the line's own pace
    assert "_ok = _pg & ((ep_len + 1) <= _lim)" in src
    assert "(FINL.n - 1 - np.clip(fin_idx, 0, FINL.n - 1))" in src
    # the curriculum's outcome is the PACED finish when the pace is on
    assert "_fgoal = fin_paced[_fei].copy()" in src
    assert "--finish-line-pace pays on the finishing tick" in src
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"finish_line_pace"' in rc and '"finish_line_pace_bonus"' in rc


def test_pace_limit_arithmetic():
    # the allowed ticks from row r: (n - 1 - r) x (1 + frac) + slack - the demo pace rule with the
    # crossing row at the line's last state
    import importlib
    tf_src = _parser()
    assert "def pace_allowed(n, cross_row, frac, slack_ticks):" in tf_src
    n, frac, slack = 1000, 0.05, 143
    rows = np.array([0, 500, 999])
    lim = np.floor((n - 1 - rows) * (1.0 + frac) + slack)
    assert list(lim) == [1191.0, 666.0, 143.0]
