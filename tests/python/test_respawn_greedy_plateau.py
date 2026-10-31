"""--respawn-greedy-plateau SECS N (uf2r2b3, 2026-10-31): the greedy-line pool also keeps N
evenly spaced states of every stretch of SECS s or more in which a greedy-eval episode - FINISHED
ones included - set no new progress record. A 9/9 policy has no failed episode, so without it
the pool is empty and its share of the spawn pool does nothing.

(a) record_plateaus finds the spans between records that last at least SECS, ignores the tail
    after the last record, and does not count improvements of eps u or less as records;
(b) the pool keeps the plateaus of a FINISHED episode, evenly spaced, and reports them;
(c) off = the pool of the code before the flag, row for row (the module at the base commit is
    loaded side by side), for every other option;
(d) the trainer's flag surface.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from surfgym.core import STATE_DTYPE                                   # noqa: E402
from surfgym.greedy_spawn import GreedyLinePool, record_plateaus       # noqa: E402

BASE = "ff2652b3"      # the commit before the flag
PER_S = 100.0          # 10 ms ticks


def _line(n=3000):
    """x = the tick; the field distance d = 5000 - x except two slow stretches: ticks 500-1400
    (a 9 s lap: d rises then comes back) and 2000-2150 (a 1.5 s loop)."""
    ep = np.zeros(n, STATE_DTYPE)
    ep["origin"][:, 0] = np.arange(n)
    ep["origin"][:, 2] = 1000.0
    ep["velocity"][:, 0] = 1500.0
    return ep


def _d(o):
    x = np.asarray(o, np.float64)[:, 0]
    d = 5000.0 - x
    lap = (x > 500) & (x < 1400)
    d[lap] = 4500.0 + 300.0 * np.sin(np.pi * (x[lap] - 500) / 900.0)
    loop = (x > 2000) & (x < 2150)
    d[loop] = 3000.0 + 50.0 * np.sin(np.pi * (x[loop] - 2000) / 150.0)
    return d


def test_record_plateaus_finds_the_slow_stretches():
    ep = _line()
    d = _d(ep["origin"])
    # records every 33 ticks at 1 u per tick (eps 32): the last one before the lap is tick 495,
    # the first after it 1400; the loop's 1994 -> 2150
    assert record_plateaus(d, PER_S, 1.0) == [(495, 1400), (1994, 2150)]
    # 1.6 s keeps only the long one (the loop is 1.56 s)
    assert record_plateaus(d, PER_S, 1.6) == [(495, 1400)]
    # a crawl of 100/999 u per tick gains 32 u per 3.2 s: a plateau every 320 ticks at eps 32,
    # and none at eps 0 (every tick is then a record)
    ramp = np.linspace(1000.0, 900.0, 1000)
    assert record_plateaus(np.r_[2000.0, ramp, 0.0], PER_S, 1.0) == [(1, 321), (321, 641),
                                                                       (641, 961)]
    assert record_plateaus(np.r_[2000.0, ramp, 0.0], PER_S, 1.0, eps=0.0) == []
    # the tail after the last record is not a plateau; off-field ticks never set one
    assert record_plateaus(np.r_[100.0, np.full(500, 100.0)], PER_S, 1.0) == []
    assert record_plateaus(np.r_[100.0, np.full(300, np.inf), 0.0], PER_S, 1.0) == [(0, 301)]


def _build(p, eps, ends, d_fn=None):
    p._eps = list(eps)
    p._cur = []
    p._par = None
    return p.build([{"end": e} for e in ends], 10.0, d_fn=d_fn)


def test_a_finished_episode_gives_its_plateaus():
    p = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0, plateau=1.0, plateau_n=50)
    info = _build(p, [_line()], ["done"], _d)
    x = np.sort(p.rows["origin"][:, 0].astype(int))
    assert len(x) == 100 and x.min() == 495 and x.max() == 2150
    assert np.all(((x >= 495) & (x <= 1400)) | ((x >= 1994) & (x <= 2150)))
    assert info["plateaus_s"] == [(4.95, 14.0), (19.94, 21.5)]
    assert np.all(p.rows["tick"] == 0)
    assert "record plateaus >= 1 s (5.0-14.0, 19.9-21.5)" in p.line()
    # a pool without the field cannot see a record
    q = GreedyLinePool(0.2, window=(0.25, 2.0), seed=0, plateau=1.0, plateau_n=50)
    _build(q, [_line()], ["done"], None)
    assert q.size == 0


def test_bad_values_are_refused():
    with pytest.raises(ValueError):
        GreedyLinePool(0.2, plateau=1.0, plateau_n=0)
    with pytest.raises(ValueError):
        GreedyLinePool(0.2, plateau=-1.0, plateau_n=10)


def _base_module():
    src = subprocess.run(["git", "show", f"{BASE}:python/surfgym/greedy_spawn.py"],
                         capture_output=True, text=True, cwd=str(ROOT), check=True).stdout
    path = Path(__file__).with_name("_greedy_spawn_base_plateau.py")
    path.write_text(src.replace("from .core import", "from surfgym.core import"),
                    encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("_greedy_spawn_base_plateau", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


def test_flag_off_is_the_code_before_the_flag():
    base = _base_module()
    rng = np.random.default_rng(3)
    eps = [_line(), _line(2500), _line(1800), _line(2200)]
    for ep in eps:
        ep["origin"][:, 1] = rng.normal(0.0, 50.0, len(ep))
    ends = ["fail", "trunc", "done", "fail"]
    for kw in ({}, {"stuck": 7}, {"contact": True}, {"turn": 30.0, "speed": (1.0, 1.1)},
               {"energy": 2, "energy_secs": 20.0}):
        for df in (None, _d):
            a = base.GreedyLinePool(0.2, window=(0.25, 2.0), seed=5, **kw)
            b = GreedyLinePool(0.2, window=(0.25, 2.0), seed=5, **kw)
            ia = _build(a, [e.copy() for e in eps], ends, df)
            ib = _build(b, [e.copy() for e in eps], ends, df)
            assert np.array_equal(a.rows, b.rows), kw
            assert ia == ib and a.line() == b.line()
            pool = np.zeros(400, STATE_DTYPE)
            assert np.array_equal(a.mix(pool, 10), b.mix(pool, 10))


def test_the_flag_surface():
    tf = (ROOT / "python" / "train_fast.py").read_text(encoding="utf-8")
    assert '"--respawn-greedy-plateau", type=float, nargs=2, default=None' in tf
    assert 'meta["config"]["respawn_greedy_plateau"] = [float(GREEDY.plateau),' in tf
    assert "--respawn-greedy-plateau adds rows to the --respawn-greedy pool" in tf
    assert "plateau=(float(args.respawn_greedy_plateau[0])" in tf
    rc = (ROOT / "tools" / "record_ckpt.py").read_text(encoding="utf-8")
    assert '"respawn_greedy_plateau",' in rc                       # TRAIN_ONLY
