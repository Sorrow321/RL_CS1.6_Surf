"""tools/knob_ge.py (unr1b4): the knob rule, the archive, the chain bookkeeping - CPU, no map -
and the trainer / recorder / auto_spine wiring of --auto-spine-scratch, --auto-spine-knob and
--demo-share, read off the sources (the flags must be off by default, dumped only when on, and
declared TRAIN_ONLY for the record gate)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from surfgym.core import STATE_DTYPE                        # noqa: E402
from knob_ge import (Knob, KnobArchive, contact_mask, knob_share, link_ticks,  # noqa: E402
                     root_indices, stall_index)


def test_knob_share_is_probability_matching():
    assert knob_share(10, 100, 10, 100) == 0.5                     # equal yields: 50/50
    assert abs(knob_share(30, 100, 10, 100) - 0.75) < 1e-12        # random 3x: 0.75
    assert abs(knob_share(10, 100, 30, 100) - 0.25) < 1e-12        # policy 3x: 0.25
    assert knob_share(0, 100, 50, 100, floor=0.05) == 0.05         # floor
    assert knob_share(50, 100, 0, 100, floor=0.05) == 0.95         # 1 - floor
    assert knob_share(0, 100, 0, 100) == 0.5                       # nothing found: 0.5
    assert knob_share(5, 0, 5, 100) == 0.5                         # no evidence for a mode
    # per DECISION, not per run: 10 cells over 10 decisions beats 10 cells over 100
    assert knob_share(10, 10, 10, 100) > 0.9


def test_knob_warms_up_then_follows_the_yield_and_decays():
    k = Knob(horizon=100, floor=0.05, warm=4)
    assert k.lam == 0.5
    k.credit(np.array([1, 1, 1]), np.array([3, 3, 3]), np.array([10, 10, 10]))
    k.credit(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), np.array([10, 10, 10, 10]))
    assert k.lam == 0.5                          # random has 3 ends < warm 4
    k.credit(np.array([1]), np.array([3]), np.array([10]))
    assert 0.7 < k.lam < 0.8                     # 0.3 vs 0.1 per decision -> 0.75
    assert list(k.ends) == [4, 4] and list(k.cells) == [4, 12]
    # old evidence decays: a long stretch where only the policy finds cells turns it round
    for _ in range(50):
        k.credit(np.array([0, 1]), np.array([5, 0]), np.array([10, 10]))
    assert k.lam < 0.1
    f = Knob(horizon=10, fixed=0.0)
    f.credit(np.array([1]), np.array([100]), np.array([1]))
    assert f.lam == 0.0


def _st(x, d=0.0):
    s = np.zeros(1, STATE_DTYPE)
    s["origin"] = (x, 0.0, 0.0)
    return s[0]


def test_archive_select_chain_best():
    a = KnobArchive(15, STATE_DTYPE, cap=2)               # grows past its capacity
    r = a.add(11, _st(0), np.zeros(15), -1, -1, 0, 7, 500.0, 7, -1)
    c1 = a.add(12, _st(1), np.ones(15), r, 0, 40, -1, 400.0, 47, 0)
    c2 = a.add(13, _st(2), None, c1, 1, 12, -1, 300.0, 59, 1)
    c3 = a.add(14, _st(3), None, c1, 2, 30, -1, 300.0, 77, 1)
    assert a.n == 4 and a.cap >= 4 and a.index == {11: 0, 12: 1, 13: 2, 14: 3}
    assert a.chain(c2) == [r, c1, c2]
    assert a.best() == c2                       # d ties: the shallower chain
    assert float(a.obs[c1][0]) == 1.0 and a.root[r] == 7 and a.parent[c3] == c1
    f = a.add(-1, _st(4), None, c2, 3, 5, -1, 0.0, 64, 1, fin=True)
    assert -1 not in a.index                    # a finish is never a key
    assert a.best() == f                        # a finish beats any depth
    rng = np.random.default_rng(0)
    pick = a.select(rng, 4000)
    assert not np.any(pick == f)                # never explored from
    a.chosen[r] = 10 ** 6                       # W = 1/sqrt(1 + chosen) all but drops it
    pick = a.select(rng, 4000)
    assert (pick == r).mean() < 0.01
    assert set(np.unique(pick)) <= {r, c1, c2, c3}


def test_contact_mask():
    g = -8.0
    og = np.array([-1, 0, -1, -1])
    vz0 = np.array([0.0, 0.0, -100.0, -100.0])
    vz1 = np.array([-8.0, 0.0, -108.0, -40.0])   # free fall / ground / free fall / pushed back
    assert list(contact_mask(og, vz0, vz1, g)) == [False, True, False, True]


def test_stall_and_roots():
    d = np.array([1000, 900, 800, 640, 600, 620, 700, 900.0])
    assert stall_index(d, 50) == 3              # the first tick within 50 of the best (600)
    assert stall_index(d, 0) == 4
    assert stall_index(np.array([np.inf, np.inf]), 10) == 1
    r = root_indices(100, 60, 100.0, 0.25, 4)
    assert r[0] == 35 and r[-1] == 60 and np.all(np.diff(r) > 0)
    r = root_indices(100, 3, 100.0, 4.0, 4)
    assert r[0] == 0 and r[-1] == 3


def test_link_ticks():
    acts = np.arange(18).reshape(3, 6)
    view = np.arange(6, dtype=np.float32).reshape(3, 2)
    a, v = link_ticks(acts, view, 9, 4)
    assert a.shape == (9, 6) and v.shape == (9, 2)
    assert (a[:4] == acts[0]).all() and (a[4:8] == acts[1]).all() and (a[8] == acts[2]).all()
    a, v = link_ticks(acts, None, 2, 4)
    assert v is None and a.shape == (2, 6)
    try:
        link_ticks(acts, view, 13, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("a link longer than its run must refuse")


def test_trainer_wiring_off_by_default_and_dumped_only_when_on():
    src = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    for flag in ("--auto-spine-scratch", "--auto-spine-knob"):
        m = re.search(r'ap\.add_argument\("%s", action="store_true"' % re.escape(flag), src)
        assert m, flag
    assert re.search(r'ap\.add_argument\("--demo-share", type=float, default=None', src)
    assert 'meta["config"]["auto_spine_scratch"] = 1' in src
    assert 'meta["config"]["auto_spine_knob"] = 1' in src
    assert 'meta["config"]["demo_share"] = float(args.demo_share)' in src
    # flag off: the curriculum is the --demo-file one, live from the start, the replace rule
    assert "DEMO_ON = bool(args.demo_file) or AUTO_SCRATCH" in src
    assert "DEMO_LIVE = [not AUTO_SCRATCH]" in src
    assert "if demo is not None and DEMO_LIVE[0] and DEMO_SHARE is None:" in src
    assert 'cmd += ["--knob"]' in src
    # --demo-share keeps the recipe's pool, so the novel / backward-mix rows stay wired with it
    assert src.count('(demo is not None and DEMO_SHARE is None, "--demo-file")') == 2
    rec = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    block = rec[rec.index("TRAIN_ONLY = frozenset({"):]
    block = block[:block.index("})")]
    for k in ("auto_spine_scratch", "auto_spine_knob", "demo_share"):
        assert re.search(r'"%s"' % k, block), k


def test_auto_spine_knob_path_and_beam_tas_flags():
    a = (ROOT / "tools" / "auto_spine.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--knob", action="store_true"' in a
    assert '"--ge-knob", "--ge-knob-minutes", a.ge_minutes' in a
    assert "return finish_spine(out, res, best_src)" in a
    b = (ROOT / "tools" / "beam_tas.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--ge-knob", action="store_true"' in b
    assert "from knob_ge import knob_ge_search" in b
